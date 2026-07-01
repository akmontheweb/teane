"""LangGraph node that decomposes the approved requirements spec into stories.

This is the entry point of the Agile / per-story TDD path:

    human_gatekeeper(ARCHITECTURE) → decomposition_node
                                       → human_gatekeeper(STORIES)
                                       → batch_planner_node …

The node:

1. Reads ``<workspace>/docs/SPEC_REQUIREMENTS.md`` (and, when present,
   ``SPEC_ARCHITECTURE.md``) as the source material.
2. Calls the planning LLM with a structured prompt that asks for a
   list of vertical-slice stories, each with acceptance criteria,
   dependencies, and a scope_files hint.
3. Persists the stories into ``<workspace>/.teane/state.db`` via
   ``harness.story_state.create_stories``.
4. Regenerates ``docs/STORIES.md`` from the DB so the STORIES
   gatekeeper has a fresh view to show the operator.

LLM cost is capped: at most ``MAX_STORIES_PER_PASS`` stories per
call. If the model wants more, ``next_pass_summary`` carries
remaining intent into the next decomposition pass (future work — for
now a single pass is the contract).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


MAX_STORIES_PER_PASS = 20
"""Hard cap. The prompt instructs the LLM to merge or defer beyond this.
Large specs that need more than 20 stories should re-run decomposition
after the first batch lands — incremental decomposition is cheaper and
gives the operator a chance to course-correct after seeing real output."""

MAX_FEATURES_PER_PASS = 8
"""Soft-ish cap on features per decomposition pass. The decomposition
LLM is asked to keep features at sprint-scale (one or more vertical
slices that share a coherent user-facing capability), not epic-scale.
More than ~8 features per pass usually means the model is treating
features as line-items rather than capabilities — the planner is
instructed to merge or defer."""


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


SPEC_REQUIREMENTS_RELPATH = os.path.join("docs", "SPEC_REQUIREMENTS.md")


def _ingest_requirements(
    workspace: str,
    app_name: str,
    spec_text: str,
) -> tuple[int, int]:
    """Parse ``docs/SPEC_REQUIREMENTS.md`` and UPSERT rows into the
    ``requirements`` table.

    Runs once per ``decomposition_node`` invocation (greenfield AND
    augment) so the requirements table reflects the latest spec
    before the decomposition LLM is asked to cite ``requirement_keys``.
    Soft-fails on errors — the validator below will surface "unknown
    requirement key" if the table is empty when it shouldn't be.

    Returns ``(parsed, inserted_or_updated)`` so the caller can log
    a one-line summary.
    """
    from harness import story_state
    from harness.req_ids import parse_spec_requirements

    parsed = parse_spec_requirements(spec_text)
    if not parsed:
        return (0, 0)
    items = [
        {
            "req_key": p.req_key,
            "kind": p.kind,
            "title": p.title,
            "body": p.body or None,
            "source_path": SPEC_REQUIREMENTS_RELPATH,
            "source_line": p.source_line,
        }
        for p in parsed
    ]
    try:
        conn = story_state.open_story_db()
        try:
            story_state.create_requirements(conn, app_name, items)
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 — soft-fail; validator will catch
        logger.warning(
            "[decomposition] requirements_ingest skipped: %s", exc,
        )
        return (len(parsed), 0)
    return (len(parsed), len(items))


_REQ_KEY_LIST_CAP = 80
"""Max number of valid identifiers embedded in a planner prompt. The full
universe is always available to the validator (which echoes it back in
its rejection message), so capping here is purely a token-budget guard
for very large specs — 80 keys covers every workspace we've seen and
keeps the prompt deterministic instead of unbounded."""


def _format_requirement_keys_guidance(
    known_req_keys: Optional[set[str]],
) -> tuple[str, str]:
    """Render the example-token + constraint-paragraph pair the decomposition
    prompts splice into their JSON shape and Constraints sections.

    Returns ``(example_json, constraint_block)`` — the first is the inner
    payload for the ``requirement_keys`` example field (already quoted,
    comma-separated, no enclosing brackets), the second is the markdown
    bullet that goes under "## Constraints".

    When ``known_req_keys`` is non-empty, both halves reference the
    workspace's actual identifier list — agile workspaces get
    ``EPIC/FEAT/STORY/STORY-NFR`` keys, waterfall workspaces get
    ``FR/NFR/US`` keys, and the LLM doesn't have to second-guess the
    family from spec text. When empty (requirements ingest produced no
    headings), falls back to generic guidance that points the LLM at
    the spec without dictating either vocabulary.
    """
    if not known_req_keys:
        return (
            '"<one valid req_key>", "<another>"',
            (
                "- Every story MUST cite at least one ``requirement_keys`` "
                "entry declared as a heading in ``docs/SPEC_REQUIREMENTS.md``. "
                "Stories citing unknown identifiers are rejected with the "
                "full list of valid keys, so prefer to map every story back "
                "to the spec it implements. A story that implements pure "
                "scaffolding without a spec requirement is itself a sign "
                "the decomposition is wrong."
            ),
        )

    sorted_keys = sorted(known_req_keys)
    sample = sorted_keys[:2] if len(sorted_keys) >= 2 else sorted_keys
    example_json = ", ".join(f'"{k}"' for k in sample)
    embedded = sorted_keys[:_REQ_KEY_LIST_CAP]
    embedded_str = ", ".join(f"``{k}``" for k in embedded)
    truncated_note = (
        f" (showing first {_REQ_KEY_LIST_CAP} of {len(sorted_keys)} — "
        "the validator knows all of them)"
        if len(sorted_keys) > _REQ_KEY_LIST_CAP else ""
    )
    constraint_block = (
        "- Every story MUST cite at least one ``requirement_keys`` entry "
        "drawn from this workspace's actual spec identifiers"
        f"{truncated_note}: {embedded_str}. Do NOT invent identifiers in "
        "any other namespace — stories citing unknown keys are rejected. "
        "Do NOT append suffixes (``STORY-011A``, ``STORY-011B``), decimals "
        "(``STORY-011.1``), or otherwise extend a listed key — use the "
        "exact string as shown above. "
        "A story that implements pure scaffolding without a matching spec "
        "requirement is itself a sign the decomposition is wrong."
    )
    return (example_json, constraint_block)


def _build_decomposition_prompt(
    spec_requirements: str,
    spec_architecture: str,
    workspace_path: str,
    known_req_keys: Optional[set[str]] = None,
) -> str:
    """Compose the planner prompt. The LLM returns JSON; the body of
    this function is the contract every decomposition LLM must follow.

    ``known_req_keys`` is the set of valid requirement identifiers
    harvested from ``docs/SPEC_REQUIREMENTS.md`` in this workspace.
    When non-empty, the prompt embeds the list verbatim and the
    example block samples from it — so an agile workspace
    (``EPIC/FEAT/STORY/STORY-NFR``) and a waterfall workspace
    (``FR/NFR/US``) get the right vocabulary without hardcoding either
    family. Empty means the requirements ingest didn't produce any
    headings; we fall back to generic guidance pointing at the spec."""
    spec_block = "## SPEC_REQUIREMENTS.md\n\n" + (spec_requirements or "_(empty)_")
    if spec_architecture:
        spec_block += "\n\n## SPEC_ARCHITECTURE.md\n\n" + spec_architecture

    req_example, req_constraint = _format_requirement_keys_guidance(known_req_keys)

    return f"""You are an Agile delivery planner. Decompose the approved
specification below into a list of **features**, and within each feature
a list of **vertical-slice stories** that the teane code-generation
agent will implement using a test-first loop (acceptance tests → code
→ run → repair → commit).

Workspace: {workspace_path}

## Features vs stories

A **feature** is a user-facing capability that ships as a unit (e.g.
"user authentication", "payment checkout", "admin reporting"). It is
the organising boundary the harness uses to plan work: each batch the
patcher runs contains stories from exactly ONE feature, so a small
feature lands in a single batch while a larger feature spans several.

A **story** is one thin end-to-end slice of value INSIDE a feature
(e.g. "user can register with email + password and receive a
confirmation"). Stories are NOT horizontal layers — "set up the
database schema" by itself is not a story.

A good story:

- Has 1–4 concrete acceptance criteria that a behavioral test can
  exercise against the public surface (CLI command, HTTP endpoint,
  library function, UI route).
- Names the files it expects to touch in ``scope_files`` when you
  have a high-confidence guess.
- Declares hard dependencies on prior stories in ``depends_on``.
  Independent stories run in parallel, so omit deps where genuinely
  optional. Use the ``story_key`` strings you assign (STORY-1,
  STORY-2, …).
- Carries a ``feature`` field whose value is one of the
  ``feature_key`` strings you defined in the ``features`` block.

A good feature:

- Has a short kebab-case ``feature_key`` (``auth``, ``billing``,
  ``admin-reporting``) — used as the join key from stories.
- Has a human-readable ``name`` (3-6 words).
- Bundles a small handful of stories that share a coherent capability;
  if a feature would have only one story, ask whether it belongs in
  a sibling feature instead.

## Output

Output STRICT JSON in this exact shape — no markdown, no code fence,
no commentary:

{{
  "features": [
    {{
      "feature_key": "auth",
      "name": "User authentication",
      "description": "Registration, login, and session management."
    }}
  ],
  "stories": [
    {{
      "story_key": "STORY-1",
      "feature": "auth",
      "title": "User can register",
      "description": "1-2 sentence summary of intent.",
      "requirement_keys": [{req_example}],
      "acceptance_criteria": [
        "POST /register with valid payload returns 201",
        "Duplicate email returns 409"
      ],
      "depends_on": [],
      "scope_files": ["src/auth/register.py", "tests/test_register.py"]
    }}
  ],
  "summary": "1-line description of the decomposition shape"
}}

## Constraints

- AT MOST {MAX_FEATURES_PER_PASS} features per pass. Merge or defer
  beyond that — the harness can re-decompose after the first features
  land.
- AT MOST {MAX_STORIES_PER_PASS} stories per pass across ALL features.
  If the spec calls for more, merge the closest-coupled ones and put
  the leftovers in a final "polish" story that the next pass can
  re-decompose.
- ``story_key`` MUST be ``STORY-N`` with N starting at 1 and
  monotonically increasing across the whole response (not per-feature).
  The DB layer re-checks this — mismatches are rejected.
- Every story's ``feature`` MUST match a ``feature_key`` declared in
  the ``features`` block.
- Every feature MUST own at least one story.
- Every story MUST have at least one acceptance_criteria entry.
{req_constraint}
- ``depends_on`` may reference any other story_key declared in this
  same response (forward or backward — the validator topologically
  resolves the graph, only true cycles are rejected). Cross-feature
  dependencies are allowed; the batch planner honours them at runtime.

Specification follows:

{spec_block}
"""


def _build_decomposition_augment_prompt(
    existing_features: list[dict[str, Any]],
    existing_stories: list[dict[str, Any]],
    spec_requirements: str,
    spec_architecture: str,
    workspace_path: str,
    known_req_keys: Optional[set[str]] = None,
) -> str:
    """Compose the delta-only planner prompt for agile-patch augmentation.

    Shown to the LLM when the workspace already has stories on file (from
    a prior `teane build --agile` or `teane patch --agile`) and the spec
    has been revised or new CRs ingested this run. The LLM is asked to
    propose ONLY new stories — anything that overlaps an existing title
    or acceptance criterion is a duplicate the operator doesn't want.

    The augment prompt may also introduce NEW features when the revised
    spec demands capabilities the existing feature set doesn't cover.
    Stories may reference either an existing feature_key (showing the
    LLM the list below) or a brand-new feature defined in this same
    response.

    ``known_req_keys`` is the snapshot of valid identifiers in this
    workspace's spec — embedded so the LLM cites the right vocabulary
    family (agile vs waterfall) without a hardcoded hint. See
    :func:`_build_decomposition_prompt` for the same contract.
    """
    req_example, req_constraint = _format_requirement_keys_guidance(known_req_keys)
    features_block_lines: list[str] = []
    for f in existing_features:
        features_block_lines.append(
            f"- ``{f.get('feature_key')}`` — {f.get('name', '')}"
            + (f" ({f.get('description')})" if f.get('description') else "")
        )
    features_block = (
        "\n".join(features_block_lines) if features_block_lines else "_(none)_"
    )

    existing_lines: list[str] = []
    for s in existing_stories:
        ac = s.get("acceptance_criteria") or []
        if isinstance(ac, list):
            ac_str = " / ".join(str(x) for x in ac[:3])
            if len(ac) > 3:
                ac_str += f" / ... ({len(ac) - 3} more)"
        else:
            ac_str = ""
        existing_lines.append(
            f"- {s.get('story_key')} [{s.get('status', '?')}] "
            f"(feature: {s.get('feature_key') or '?'}) "
            f"{s.get('title', '')} — {ac_str}"
        )
    existing_block = (
        "\n".join(existing_lines) if existing_lines else "_(none)_"
    )

    spec_block = "## SPEC_REQUIREMENTS.md (current)\n\n" + (
        spec_requirements or "_(empty)_"
    )
    if spec_architecture:
        spec_block += "\n\n## SPEC_ARCHITECTURE.md (current)\n\n" + spec_architecture

    return f"""You are an Agile delivery planner in **augment mode**.

The workspace at ``{workspace_path}`` already has features and stories
tracked in ``.teane/state.db`` (listed below). The spec has been revised
or new change requests have been ingested this run, so the planner needs
to propose ONLY new work that fills gaps the existing set doesn't cover.

## Existing features

{features_block}

## Existing stories

{existing_block}

## What to do

1. Read the revised spec and any change-request preamble in the
   conversation above.
2. Identify functionality the existing stories DON'T cover.
3. Propose ONLY new vertical-slice stories for that gap. Do NOT
   re-emit an existing story under a new key — even if you'd refine
   the title. Existing DONE stories will be re-classified separately
   by ``story_reopen_node`` if their acceptance criteria drifted.
4. If a new story fits an existing feature, set its ``feature`` field
   to that feature_key. If it belongs to a brand-new capability,
   declare a new feature in the ``features`` block and reference it.
5. If no new stories are needed, return an empty ``stories`` list AND
   an empty ``features`` list.

Use placeholder keys ``STORY-NEW-1``, ``STORY-NEW-2``, … for new
stories. The DB allocator overwrites these with the next-available
``STORY-N`` when inserting, so the placeholders are just unique
markers inside this response.

Output STRICT JSON in this exact shape — no markdown, no code fence:

{{
  "features": [
    {{"feature_key": "new-cap", "name": "New capability", "description": "1-line"}}
  ],
  "stories": [
    {{
      "story_key": "STORY-NEW-1",
      "feature": "new-cap",
      "title": "1-line title",
      "description": "1-2 sentence summary of intent.",
      "requirement_keys": [{req_example}],
      "acceptance_criteria": ["..."],
      "depends_on": [],
      "scope_files": ["src/..."]
    }}
  ],
  "summary": "1-line description of the delta"
}}

Constraints:
- AT MOST {MAX_STORIES_PER_PASS} new stories per pass.
- AT MOST {MAX_FEATURES_PER_PASS} new features per pass.
- Every new story MUST have at least one acceptance criterion.
{req_constraint}
- Every new story's ``feature`` field MUST reference either an
  existing feature_key (from the list above) or a NEW feature_key
  declared in the ``features`` block of THIS response.
- ``depends_on`` may only reference other placeholders in this same
  response; cross-references to existing DONE stories are not allowed.
- If the existing set already covers everything in the revised spec,
  return ``"features": []`` and ``"stories": []`` — that's the valid
  "no-op" answer.

Specification follows:

{spec_block}
"""


def _validate_features_payload(
    data: Any, *, allow_empty: bool,
) -> list[dict[str, Any]]:
    """Sanity-check the ``features`` block of a decomposition response.

    Returns the cleaned feature list. ``allow_empty=True`` in augment
    mode (a no-op "no new features" answer is legal); ``False`` in
    initial decomposition (every story needs a feature to belong to).
    """
    features = data.get("features")
    if not isinstance(features, list):
        raise ValueError("'features' must be a list")
    if not features:
        if allow_empty:
            return []
        raise ValueError("'features' must be a non-empty list")
    if len(features) > MAX_FEATURES_PER_PASS:
        raise ValueError(
            f"too many features ({len(features)} > {MAX_FEATURES_PER_PASS}); "
            "merge or defer"
        )
    seen: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    for i, f in enumerate(features, start=1):
        if not isinstance(f, dict):
            raise ValueError(f"feature #{i} must be an object")
        key = (f.get("feature_key") or "").strip()
        if not key:
            raise ValueError(f"feature #{i} requires non-empty 'feature_key'")
        if key in seen:
            raise ValueError(f"duplicate feature_key {key!r}")
        seen.add(key)
        name = (f.get("name") or "").strip()
        if not name:
            raise ValueError(
                f"feature {key!r} requires non-empty 'name'"
            )
        cleaned.append({
            "feature_key": key,
            "name": name,
            "description": (f.get("description") or "").strip() or None,
        })
    return cleaned


def _validate_story_requirement_keys(
    story_key: str,
    raw: Any,
    known_req_keys: Optional[set[str]],
) -> list[str]:
    """Validate one story's ``requirement_keys`` and return the cleaned
    list, or raise ValueError with a precise message.

    Enforces:
      - ``requirement_keys`` is a non-empty list of strings.
      - When ``known_req_keys`` is provided, every cited key must
        appear in it. The error message lists up to 40 valid keys
        (sorted) so the operator immediately sees the universe of
        valid choices without re-opening the spec.

    ``known_req_keys=None`` means the validator only checks shape
    (used by unit tests that don't seed a DB). Production callers
    in ``decomposition_node`` always pass a real set.
    """
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"{story_key} must cite at least one 'requirement_keys' entry "
            "declared as a heading in docs/SPEC_REQUIREMENTS.md"
        )
    keys = [str(x).strip() for x in raw if str(x).strip()]
    if not keys:
        raise ValueError(
            f"{story_key} 'requirement_keys' must contain non-empty strings"
        )
    if known_req_keys is None:
        return keys
    unknown = [k for k in keys if k not in known_req_keys]
    if unknown:
        sample = sorted(known_req_keys)[:40]
        more = (
            f" (showing first 40 of {len(known_req_keys)})"
            if len(known_req_keys) > 40 else ""
        )
        raise ValueError(
            f"{story_key} cites unknown requirement_keys "
            f"{sorted(set(unknown))}. Known req_keys in this workspace"
            f"{more}: {sample}. Update docs/SPEC_REQUIREMENTS.md to add "
            "the missing requirement(s), or revise the story to cite "
            "an existing one."
        )
    return keys


def _check_depends_on_acyclic(
    story_keys: list[str],
    deps_per_story: list[list[str]],
    *,
    valid_targets: set[str],
    response_label: str,
) -> None:
    """Validate ``depends_on`` references full-set membership + acyclicity.

    Replaces the old strict "declared earlier in the array" check, which
    rejected LLM responses that happened to emit a forward reference
    (e.g. STORY-8 depends_on STORY-15 when STORY-15 hadn't appeared yet
    in the payload). LLMs routinely group stories by feature, not by
    topological order — and as long as the dependency graph is acyclic
    there is no semantic problem: ``get_planned_stories`` already gates
    each story on its deps being ``done`` at runtime, so a forward
    reference in the payload becomes a backward reference at runtime.

    Cycles remain a hard error: a dependency cycle would leave every
    story in the cycle blocked forever (each waiting on the others to
    finish), so the validator must reject them up front.
    """
    if len(story_keys) != len(deps_per_story):
        raise AssertionError("story_keys and deps_per_story must align 1:1")

    # 1) Membership check: every dep target must be a valid key.
    for key, deps in zip(story_keys, deps_per_story):
        for d in deps:
            if d not in valid_targets:
                raise ValueError(
                    f"{key} depends_on {d!r} which is not declared in "
                    f"{response_label}"
                )

    # 2) Cycle detection: iterative DFS with 3-color marking. Only edges
    #    INTO this-response keys participate; deps targeting any external
    #    set (valid_targets - this_response_keys) can never form a cycle
    #    with this batch.
    this_response_keys = set(story_keys)
    by_key = {k: i for i, k in enumerate(story_keys)}
    adj: list[list[int]] = [[] for _ in story_keys]
    for i, deps in enumerate(deps_per_story):
        for d in deps:
            if d in this_response_keys:
                adj[i].append(by_key[d])
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * len(story_keys)
    for start in range(len(story_keys)):
        if color[start] != WHITE:
            continue
        stack: list[tuple[int, int]] = [(start, 0)]
        path: list[int] = []
        color[start] = GRAY
        path.append(start)
        while stack:
            node, idx = stack[-1]
            if idx < len(adj[node]):
                stack[-1] = (node, idx + 1)
                nxt = adj[node][idx]
                if color[nxt] == GRAY:
                    cycle_start = path.index(nxt)
                    cycle_keys = [story_keys[n] for n in path[cycle_start:]] + [
                        story_keys[nxt]
                    ]
                    raise ValueError(
                        "depends_on cycle detected: "
                        + " → ".join(cycle_keys)
                    )
                if color[nxt] == WHITE:
                    color[nxt] = GRAY
                    path.append(nxt)
                    stack.append((nxt, 0))
            else:
                color[node] = BLACK
                path.pop()
                stack.pop()


def _validate_stories_against_features(
    stories: list[dict[str, Any]],
    declared_feature_keys: set[str],
    existing_feature_keys: Optional[set[str]] = None,
) -> None:
    """Enforce that every story.feature references a known feature_key.

    ``declared_feature_keys`` is the set defined in THIS response's
    ``features`` block. ``existing_feature_keys`` (augment mode only)
    is the set already on file in the DB. Together they define the
    valid namespace.
    """
    allowed = set(declared_feature_keys)
    if existing_feature_keys:
        allowed |= existing_feature_keys
    for s in stories:
        key = s.get("story_key", "?")
        fkey = s.get("feature")
        if not fkey:
            raise ValueError(f"{key} is missing 'feature'")
        if fkey not in allowed:
            raise ValueError(
                f"{key} references feature {fkey!r} which is not declared "
                f"in this response's 'features' block "
                f"({sorted(declared_feature_keys)}) "
                + (
                    f"nor in existing features ({sorted(existing_feature_keys)})"
                    if existing_feature_keys else ""
                )
            )


def _validate_augment_payload(
    data: Any,
    *,
    existing_feature_keys: Optional[set[str]] = None,
    known_req_keys: Optional[set[str]] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Augment-mode validator: same shape as _validate_stories_payload
    but tolerates an empty stories list as a legitimate "no new work"
    answer. Allows STORY-NEW-N placeholder keys in addition to STORY-N.

    Returns ``(features, stories)``. Both may be empty in the no-op
    answer. ``known_req_keys`` (when provided) cross-validates each
    story's ``requirement_keys`` against the spec — see
    :func:`_validate_story_requirement_keys`.
    """
    if not isinstance(data, dict):
        raise ValueError(f"top-level must be JSON object, got {type(data).__name__}")
    stories = data.get("stories")
    if not isinstance(stories, list):
        raise ValueError("'stories' must be a list (empty allowed in augment mode)")
    features_cleaned = _validate_features_payload(data, allow_empty=True)
    if not stories:
        return features_cleaned, []
    if len(stories) > MAX_STORIES_PER_PASS:
        raise ValueError(
            f"too many new stories ({len(stories)} > {MAX_STORIES_PER_PASS}); "
            "merge closely-coupled stories"
        )
    seen_keys: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    story_keys_in_order: list[str] = []
    deps_in_order: list[list[str]] = []
    for i, s in enumerate(stories, start=1):
        if not isinstance(s, dict):
            raise ValueError(f"story #{i} must be an object")
        key = s.get("story_key")
        if not isinstance(key, str) or not key.startswith("STORY-"):
            raise ValueError(f"story #{i} has invalid story_key: {key!r}")
        if key in seen_keys:
            raise ValueError(f"duplicate placeholder key {key}")
        seen_keys.add(key)
        title = s.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"{key} is missing a non-empty title")
        ac = s.get("acceptance_criteria") or []
        if not isinstance(ac, list) or not ac:
            raise ValueError(f"{key} must have at least one acceptance criterion")
        req_keys = _validate_story_requirement_keys(
            key, s.get("requirement_keys"), known_req_keys,
        )
        deps = s.get("depends_on") or []
        if not isinstance(deps, list):
            raise ValueError(f"{key} depends_on must be a list")
        scope = s.get("scope_files") or []
        if not isinstance(scope, list):
            raise ValueError(f"{key} scope_files must be a list")
        feature = (s.get("feature") or "").strip()
        deps_str = [str(x) for x in deps]
        story_keys_in_order.append(key)
        deps_in_order.append(deps_str)
        cleaned.append({
            "title": title.strip(),
            "feature": feature or None,
            "description": s.get("description") or None,
            "acceptance_criteria": [str(x) for x in ac],
            "requirement_keys": req_keys,
            "depends_on": deps_str,
            "scope_files": [str(x) for x in scope],
            "external_ref": s.get("external_ref") or None,
        })
    _check_depends_on_acyclic(
        story_keys_in_order, deps_in_order,
        valid_targets=seen_keys,
        response_label="this augment response",
    )
    _validate_stories_against_features(
        cleaned,
        {f["feature_key"] for f in features_cleaned},
        existing_feature_keys,
    )
    return features_cleaned, cleaned


def _validate_stories_payload(
    data: Any,
    *,
    known_req_keys: Optional[set[str]] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sanity-check the LLM's JSON. Returns ``(features, stories)``.

    Raises ValueError with a precise message on shape violations so
    the caller can surface it to the operator instead of writing a
    corrupt batch into the DB. ``known_req_keys`` (when provided)
    cross-validates each story's ``requirement_keys`` against the
    spec — see :func:`_validate_story_requirement_keys`.
    """
    if not isinstance(data, dict):
        raise ValueError(f"top-level must be JSON object, got {type(data).__name__}")
    features_cleaned = _validate_features_payload(data, allow_empty=False)
    stories = data.get("stories")
    if not isinstance(stories, list) or not stories:
        raise ValueError("'stories' must be a non-empty list")
    if len(stories) > MAX_STORIES_PER_PASS:
        raise ValueError(
            f"too many stories ({len(stories)} > {MAX_STORIES_PER_PASS}); "
            "merge closely-coupled stories"
        )

    seen_keys: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    story_keys_in_order: list[str] = []
    deps_in_order: list[list[str]] = []
    for i, s in enumerate(stories, start=1):
        if not isinstance(s, dict):
            raise ValueError(f"story #{i} must be an object")
        key = s.get("story_key")
        if not isinstance(key, str) or not key.startswith("STORY-"):
            raise ValueError(f"story #{i} has invalid story_key: {key!r}")
        if key in seen_keys:
            raise ValueError(f"duplicate story_key {key}")
        seen_keys.add(key)
        title = s.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"{key} is missing a non-empty title")
        ac = s.get("acceptance_criteria") or []
        if not isinstance(ac, list) or not ac:
            raise ValueError(f"{key} must have at least one acceptance criterion")
        req_keys = _validate_story_requirement_keys(
            key, s.get("requirement_keys"), known_req_keys,
        )
        deps = s.get("depends_on") or []
        if not isinstance(deps, list):
            raise ValueError(f"{key} depends_on must be a list")
        scope = s.get("scope_files") or []
        if not isinstance(scope, list):
            raise ValueError(f"{key} scope_files must be a list")
        feature = (s.get("feature") or "").strip()
        deps_str = [str(x) for x in deps]
        story_keys_in_order.append(key)
        deps_in_order.append(deps_str)
        cleaned.append({
            "title": title.strip(),
            "feature": feature or None,
            "description": s.get("description") or None,
            "acceptance_criteria": [str(x) for x in ac],
            "requirement_keys": req_keys,
            "depends_on": deps_str,
            "scope_files": [str(x) for x in scope],
            "external_ref": s.get("external_ref") or None,
        })
    _check_depends_on_acyclic(
        story_keys_in_order, deps_in_order,
        valid_targets=seen_keys,
        response_label="this response",
    )
    _validate_stories_against_features(
        cleaned, {f["feature_key"] for f in features_cleaned},
    )
    # Every declared feature must own at least one story.
    feature_owners: set[str] = {s["feature"] for s in cleaned if s.get("feature")}
    orphan_features = sorted(
        f["feature_key"] for f in features_cleaned
        if f["feature_key"] not in feature_owners
    )
    if orphan_features:
        raise ValueError(
            f"features {orphan_features} have no stories — "
            "every feature must own at least one story"
        )
    return features_cleaned, cleaned


def strip_json_fence(content: str) -> str:
    """Tolerate a code-fenced JSON response even though the prompt
    forbids it — some models add fences anyway. Thin wrapper around
    ``harness.trust.strip_code_fences`` (canonical implementation);
    kept under this name for backwards compatibility with
    ``harness.batch_sizing`` and other call sites.
    """
    from harness.trust import strip_code_fences
    return strip_code_fences(content)


def _build_cycle_repair_prompt(raw_json: str, cycle_msg: str) -> str:
    """Targeted one-shot repair prompt for `depends_on` cycles.

    Feeds the planner its own prior payload + the exact cycle path and
    asks it to drop the minimum edges. Kept narrow on purpose: removing
    deps is a local, low-risk edit; broader "fix the payload" prompts
    have historically drifted into renaming stories or dropping
    acceptance criteria, which then fails downstream traceability.
    """
    return (
        "Your previous decomposition response contained a circular "
        "`depends_on` dependency that makes the plan unschedulable:\n\n"
        f"  {cycle_msg}\n\n"
        "Fix the payload by removing the minimum number of `depends_on` "
        "edges needed to break the cycle. Prefer dropping the edge that "
        "is semantically weakest (e.g. if STORY-A is genuinely a "
        "prerequisite for STORY-B, then the reverse edge B → A is the "
        "one to drop). Do NOT remove or rename any stories, features, "
        "acceptance_criteria, requirement_keys, or scope_files. Edit "
        "ONLY the `depends_on` arrays.\n\n"
        "Return the COMPLETE corrected payload as a single JSON object "
        "with exactly the same shape as before. JSON only — no "
        "commentary, no code fences.\n\n"
        "Previous payload:\n"
        f"{raw_json}"
    )


_UNKNOWN_REQ_KEY_ERR_MARKER = "cites unknown requirement_keys"
"""Substring that identifies the validator error emitted by
``_validate_story_requirement_keys`` when a story references a
requirement_key that isn't in the workspace's requirements table.
Used by ``decomposition_node`` to detect the auto-repair-eligible
branch."""


def _build_unknown_req_repair_prompt(
    raw_json: str,
    err_msg: str,
    known_req_keys: set[str],
) -> str:
    """Targeted one-shot repair prompt for unknown ``requirement_keys``.

    Mirrors the cycle-repair contract: feed the planner its own prior
    payload plus the exact validator complaint (which already names the
    invented key AND lists the workspace's valid alternatives), and ask
    it to swap the offending entries for real ones without touching
    anything else. The narrow scope prevents the common "fix the
    payload" drift into renaming stories or shuffling acceptance
    criteria that then breaks downstream traceability.

    ``known_req_keys`` is re-listed in the repair prompt (up to the
    same 80-key cap the initial prompt uses) so the planner sees the
    universe of valid alternatives in one turn instead of having to
    re-parse the validator error.
    """
    sorted_keys = sorted(known_req_keys)
    embedded = sorted_keys[:_REQ_KEY_LIST_CAP]
    embedded_str = ", ".join(f"``{k}``" for k in embedded)
    truncated_note = (
        f" (showing first {_REQ_KEY_LIST_CAP} of {len(sorted_keys)})"
        if len(sorted_keys) > _REQ_KEY_LIST_CAP else ""
    )
    return (
        "Your previous decomposition response cited requirement_keys "
        "that do not exist in this workspace's specification:\n\n"
        f"  {err_msg}\n\n"
        "Valid requirement_keys for this workspace"
        f"{truncated_note}: {embedded_str}.\n\n"
        "Fix the payload by replacing every unknown key with an EXACT "
        "match from the valid list above. If a story's intent maps to "
        "no listed requirement, drop the offending key (leaving at "
        "least one valid entry per story) rather than inventing a new "
        "identifier. Do NOT append suffixes (A/B), decimals, or "
        "otherwise extend a listed key — use the exact string as "
        "shown. Do NOT remove or rename any stories, features, "
        "acceptance_criteria, depends_on, or scope_files. Edit ONLY "
        "the `requirement_keys` arrays.\n\n"
        "Return the COMPLETE corrected payload as a single JSON object "
        "with exactly the same shape as before. JSON only — no "
        "commentary, no code fences.\n\n"
        "Previous payload:\n"
        f"{raw_json}"
    )


async def decomposition_node(state: dict[str, Any]) -> dict[str, Any]:
    """Decompose the approved spec into stories, persist them, regenerate views.

    Returns a state-delta dict in the LangGraph convention. Sets:

    - ``stories_db_path`` — absolute path to the workspace state DB
    - ``current_gate`` = "STORIES" so the next hop into
      ``human_gatekeeper_node`` knows which gate to render
    - ``node_state.decomposition_complete`` boolean + story count
    """
    from harness.gateway import NodeRole
    from harness.graph import get_gateway
    from harness import story_state

    workspace = state.get("workspace_path") or os.getcwd()
    spec_req = _read_text(os.path.join(workspace, "docs", "SPEC_REQUIREMENTS.md"))
    if not spec_req.strip():
        logger.warning("[decomposition] SPEC_REQUIREMENTS.md is empty or missing")
        return {
            "current_gate": "STORIES",
            "node_state": {
                "current_node": "decomposition",
                "decomposition_complete": True,
                "error": "spec_requirements_missing",
                "story_count": 0,
            },
        }
    spec_arch = _read_text(os.path.join(workspace, "docs", "SPEC_ARCHITECTURE.md"))

    # Parse the §11 machine-readable summary the arch_doc skill embeds
    # in SPEC_ARCHITECTURE.md. Lazy + lenient — None on any failure,
    # which means downstream nodes fall back to prose-only handoff.
    # Stored on AgentState so patching_node and the batch planner don't
    # each re-read the file.
    from harness.arch_summary import load_arch_summary
    arch_summary = load_arch_summary(workspace) or {}

    gateway = get_gateway()
    if gateway is None:
        logger.error("[decomposition] No gateway configured.")
        return {
            "current_gate": "STORIES",
            "node_state": {
                "current_node": "decomposition",
                "decomposition_complete": True,
                "error": "no_gateway",
                "story_count": 0,
            },
        }

    budget = state.get("budget_remaining_usd", 0.0)
    if budget <= 0:
        logger.warning("[decomposition] Budget exhausted ($%.4f); skipping.", budget)
        return {
            "current_gate": "STORIES",
            "node_state": {
                "current_node": "decomposition",
                "decomposition_complete": True,
                "error": "budget_exhausted",
                "story_count": 0,
            },
            "budget_remaining_usd": budget,
        }

    # Augment mode: the workspace already has stories on file from a
    # prior agile run. Show the LLM the existing list and ask for
    # delta-only proposals. The PATCH flow upstream routes through
    # story_reopen_node first to flip drifted DONE stories to REOPENED;
    # this pass picks up brand-new stories the revised spec demands.
    app_name = story_state.app_name_for_workspace(workspace)

    # v5 requirements ingest. Parses FR/NFR/US headings from the spec
    # and UPSERTs the ``requirements`` table BEFORE the augment peek so
    # the augment prompt (and the validator) see the current set of
    # valid requirement_keys. Soft-fail; the validator below will
    # surface "unknown requirement key" if the table is empty when a
    # story tries to cite one.
    parsed_count, upserted_count = _ingest_requirements(
        workspace, app_name, spec_req,
    )
    if parsed_count:
        logger.info(
            "[decomposition] requirements_ingest: %d parsed, %d upserted",
            parsed_count, upserted_count,
        )

    augment_existing: list[dict[str, Any]] = []
    augment_existing_features: list[dict[str, Any]] = []
    known_req_keys: set[str] = set()
    # The peek-then-write pattern opens TWO sqlite connections to the
    # same file (this one and the writer below). That's safe because:
    # (a) the state.db has WAL enabled so readers/writers don't block
    # each other; (b) ``create_features`` is idempotent on duplicate
    # feature_key; (c) the LLM call between peek and write would force
    # us to hold a conn open for 30s+, which is worse than two short
    # connections. Operators MUST NOT run two ``teane`` processes
    # against the same workspace concurrently — that's contracted on
    # by the workspace lock taken at session start.
    try:
        _peek_conn = story_state.open_story_db()
        try:
            augment_existing = story_state.list_stories(_peek_conn, app_name)
            augment_existing_features = story_state.list_features(_peek_conn, app_name)
            # Snapshot the requirement universe so the validator can
            # reject stories citing unknown req_keys with a precise
            # error listing valid alternatives.
            known_req_keys = {
                r["req_key"]
                for r in story_state.list_requirements(_peek_conn, app_name)
            }
        finally:
            _peek_conn.close()
    except Exception as exc:  # noqa: BLE001 — fall back to from-scratch
        logger.info("[decomposition] augment-mode peek skipped: %s", exc)
        augment_existing = []
        augment_existing_features = []
        known_req_keys = set()

    augment_mode = bool(augment_existing)
    if augment_mode:
        logger.info(
            "[decomposition] augment mode active "
            "(%d existing features, %d existing stories); "
            "prompting LLM for delta-only proposals.",
            len(augment_existing_features), len(augment_existing),
        )
        prompt = _build_decomposition_augment_prompt(
            augment_existing_features, augment_existing,
            spec_req, spec_arch, workspace,
            known_req_keys=known_req_keys,
        )
    else:
        prompt = _build_decomposition_prompt(
            spec_req, spec_arch, workspace,
            known_req_keys=known_req_keys,
        )
    system_msg = state.get("messages", [{}])[0] if state.get("messages") else {}
    call_messages = [system_msg, {"role": "user", "content": prompt}]

    try:
        response, budget = await gateway.dispatch(
            messages=call_messages,
            role=NodeRole.PLANNING,
            budget_remaining_usd=budget,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[decomposition] gateway dispatch failed: %s", exc)
        return {
            "exit_code": 1,
            "node_state": {
                "current_node": "decomposition",
                "decomposition_complete": False,
                "decomposition_failed": True,
                "error": f"dispatch_failed: {exc}",
                "story_count": 0,
            },
            "budget_remaining_usd": budget,
        }

    raw = strip_json_fence(getattr(response, "content", "") or "")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("[decomposition] LLM returned invalid JSON: %s", exc)
        return {
            "exit_code": 1,
            "node_state": {
                "current_node": "decomposition",
                "decomposition_complete": False,
                "decomposition_failed": True,
                "error": f"invalid_json: {exc}",
                "story_count": 0,
            },
            "budget_remaining_usd": budget,
        }

    # Always pass the snapshot through — even an EMPTY set must
    # be threaded so the validator's cross-check fires and rejects
    # bogus req_keys against a workspace whose spec has no FR/NFR/US
    # headings yet. ``set() or None`` would collapse to None and
    # silently drop the validator into shape-only mode, leaving
    # the audit to "pass" vacuously (Phase 7 BUG #5). ``None`` is
    # reserved for unit tests that explicitly want shape-only.
    existing_keys: set[str] = set()
    if augment_mode:
        existing_keys = {
            f.get("feature_key") for f in augment_existing_features
            if f.get("feature_key")
        }

    def _run_validator(payload: dict[str, Any]) -> tuple[list[dict], list[dict]]:
        if augment_mode:
            return _validate_augment_payload(
                payload,
                existing_feature_keys=existing_keys,
                known_req_keys=known_req_keys,
            )
        return _validate_stories_payload(
            payload, known_req_keys=known_req_keys,
        )

    try:
        features_cleaned, cleaned = _run_validator(data)
    except ValueError as exc:
        # Two validation-error classes support a single targeted
        # auto-repair pass. Cycle errors: feed the payload back with
        # the cycle path and ask for the minimum edge removal.
        # Unknown-requirement_key errors: feed the payload back with
        # the offending key(s) + the full valid-key universe and ask
        # for an in-vocabulary swap. Both use narrow repair prompts
        # (edit only the offending arrays) so the model doesn't drift
        # into renaming stories or dropping traceability. Other
        # validation errors (malformed shape, feature mismatches) stay
        # structural and route to HITL.
        exc_str = str(exc)
        cycle_err = exc_str.startswith("depends_on cycle detected:")
        unknown_req_err = _UNKNOWN_REQ_KEY_ERR_MARKER in exc_str
        if (cycle_err or unknown_req_err) and budget > 0:
            repair_kind = "cycle" if cycle_err else "unknown_req_key"
            logger.warning(
                "[decomposition] %s — attempting 1-shot %s auto-repair "
                "(budget=$%.4f).",
                exc, repair_kind, budget,
            )
            if cycle_err:
                repair_prompt = _build_cycle_repair_prompt(raw, exc_str)
            else:
                repair_prompt = _build_unknown_req_repair_prompt(
                    raw, exc_str, known_req_keys,
                )
            repair_messages = [
                system_msg, {"role": "user", "content": repair_prompt},
            ]
            try:
                response, budget = await gateway.dispatch(
                    messages=repair_messages,
                    role=NodeRole.PLANNING,
                    budget_remaining_usd=budget,
                )
                repaired_raw = strip_json_fence(
                    getattr(response, "content", "") or ""
                )
                data = json.loads(repaired_raw)
                features_cleaned, cleaned = _run_validator(data)
                raw = repaired_raw
                logger.info(
                    "[decomposition] %s auto-repair succeeded; "
                    "budget_remaining=$%.4f.", repair_kind, budget,
                )
            except Exception as repair_exc:  # noqa: BLE001
                logger.error(
                    "[decomposition] %s auto-repair failed: %s",
                    repair_kind, repair_exc,
                )
                return {
                    "exit_code": 1,
                    "node_state": {
                        "current_node": "decomposition",
                        "decomposition_complete": False,
                        "decomposition_failed": True,
                        "error": (
                            f"validation: {exc}; "
                            f"repair_failed: {repair_exc}"
                        ),
                        "story_count": 0,
                    },
                    "budget_remaining_usd": budget,
                }
        else:
            logger.error(
                "[decomposition] payload validation failed: %s", exc,
            )
            # Failure must NOT advertise itself as a successful
            # completion — the prior shape ({decomposition_complete:
            # True, current_gate: "STORIES"}) caused the gatekeeper to
            # show an empty STORIES gate, the developer to approve it,
            # the planner to find no stories and report
            # ``all_complete=True``, and the rest of the pipeline to
            # generate code with zero traceability to the spec. Route
            # to HITL with a clear error instead.
            return {
                "exit_code": 1,
                "node_state": {
                    "current_node": "decomposition",
                    "decomposition_complete": False,
                    "decomposition_failed": True,
                    "error": f"validation: {exc}",
                    "story_count": 0,
                },
                "budget_remaining_usd": budget,
            }

    db_path = story_state.state_db_path()
    # app_name was already resolved during the augment-mode peek above.
    # CR-mode decompositions tag every story as a CR layer so the
    # traceability matrix can split greenfield work from incremental
    # change-request work. cr_ids is the integer set ingested in this
    # run; greenfield runs leave it None.
    if state.get("change_request_mode"):
        build_kind = story_state.BUILD_KIND_CR
        cr_ids = sorted({
            int(r.get("cr_id"))
            for r in (state.get("change_request_files") or [])
            if r.get("cr_id") is not None
        })
    else:
        build_kind = story_state.BUILD_KIND_CR if augment_mode else story_state.BUILD_KIND_GREENFIELD
        cr_ids = None
    # Augment mode + empty cleaned list = "no new stories needed". Skip
    # the DB insert entirely; the existing rows + any story_reopen
    # verdicts upstream are the full work set for this run.
    if augment_mode and not cleaned and not features_cleaned:
        conn = story_state.open_story_db()
        try:
            stories_md, _ = story_state.regenerate_markdown_views(conn, workspace)
        finally:
            conn.close()
        logger.info(
            "[decomposition] augment mode: no new stories needed. "
            "Existing %d stories carry forward.", len(augment_existing),
        )
        return {
            "stories_db_path": db_path,
            "arch_summary": arch_summary,
            "current_gate": "STORIES",
            "budget_remaining_usd": budget,
            "node_state": {
                "current_node": "decomposition",
                "decomposition_complete": True,
                "story_count": 0,
                "story_keys": [],
                "feature_count": 0,
                "feature_keys": [],
                "augment_mode": True,
                "augment_existing_count": len(augment_existing),
                "stories_md_path": stories_md,
                "summary": data.get("summary") or "no new stories needed",
            },
        }
    conn = story_state.open_story_db()
    try:
        # Features first — create_stories below resolves each story's
        # ``feature`` field to a feature_id by lookup, so the rows must
        # exist before stories reference them. ``create_features`` is
        # idempotent on duplicate feature_key (augment mode may declare
        # a feature_key that already exists on file).
        created_feature_keys = story_state.create_features(
            conn, app_name, features_cleaned,
        )
        created_keys = story_state.create_stories(
            conn, app_name, cleaned,
            build_kind=build_kind, cr_ids=cr_ids,
        )
        # v5: each cleaned story carries a validated ``requirement_keys``
        # list (Phase 2 contract). Write the story_satisfies_req edges
        # now that both sides exist in the DB. create_stories returns
        # keys in the same order as the input list, so a zip stays
        # aligned even though create_stories' public signature returns
        # list[str] (keeping it that way avoided churn across ~15 call
        # sites — see the plan's Phase 1 notes).
        for story_key, story_item in zip(created_keys, cleaned):
            req_keys = story_item.get("requirement_keys") or []
            if not req_keys:
                continue
            row = story_state.get_story(conn, app_name, story_key)
            if row is None:
                logger.warning(
                    "[decomposition] story %s vanished before req-link write",
                    story_key,
                )
                continue
            try:
                story_state.link_story_to_requirements(
                    conn, app_name, row["id"], req_keys,
                )
            except ValueError as exc:
                # Validator should have caught this; defensive log only.
                logger.error(
                    "[decomposition] req-link skipped for %s: %s",
                    story_key, exc,
                )
        stories_md, _ = story_state.regenerate_markdown_views(conn, workspace)
    finally:
        conn.close()

    logger.info(
        "[decomposition] created %d feature(s) [%s] and %d story(ies) [%s]; "
        "STORIES.md regenerated at %s",
        len(created_feature_keys), ", ".join(created_feature_keys),
        len(created_keys), ", ".join(created_keys),
        stories_md,
    )

    return {
        "stories_db_path": db_path,
        "arch_summary": arch_summary,
        "current_gate": "STORIES",
        "budget_remaining_usd": budget,
        "node_state": {
            "current_node": "decomposition",
            "decomposition_complete": True,
            "story_count": len(created_keys),
            "story_keys": created_keys,
            "feature_count": len(created_feature_keys),
            "feature_keys": created_feature_keys,
            "stories_md_path": stories_md,
            "summary": data.get("summary") or "",
            "augment_mode": augment_mode,
            "augment_existing_count": len(augment_existing),
        },
    }
