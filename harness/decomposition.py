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


def _build_decomposition_prompt(
    spec_requirements: str,
    spec_architecture: str,
    workspace_path: str,
) -> str:
    """Compose the planner prompt. The LLM returns JSON; the body of
    this function is the contract every decomposition LLM must follow."""
    spec_block = "## SPEC_REQUIREMENTS.md\n\n" + (spec_requirements or "_(empty)_")
    if spec_architecture:
        spec_block += "\n\n## SPEC_ARCHITECTURE.md\n\n" + spec_architecture

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
- ``depends_on`` may only reference story_keys that appear earlier
  in the same response. Cross-feature dependencies are allowed and
  the batch planner will honour them.

Specification follows:

{spec_block}
"""


def _build_decomposition_augment_prompt(
    existing_features: list[dict[str, Any]],
    existing_stories: list[dict[str, Any]],
    spec_requirements: str,
    spec_architecture: str,
    workspace_path: str,
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
    """
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Augment-mode validator: same shape as _validate_stories_payload
    but tolerates an empty stories list as a legitimate "no new work"
    answer. Allows STORY-NEW-N placeholder keys in addition to STORY-N.

    Returns ``(features, stories)``. Both may be empty in the no-op
    answer.
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
        deps = s.get("depends_on") or []
        if not isinstance(deps, list):
            raise ValueError(f"{key} depends_on must be a list")
        for d in deps:
            if d not in seen_keys:
                raise ValueError(
                    f"{key} depends_on '{d}' which is not declared earlier "
                    "in this augment response"
                )
        scope = s.get("scope_files") or []
        if not isinstance(scope, list):
            raise ValueError(f"{key} scope_files must be a list")
        feature = (s.get("feature") or "").strip()
        cleaned.append({
            "title": title.strip(),
            "feature": feature or None,
            "description": s.get("description") or None,
            "acceptance_criteria": [str(x) for x in ac],
            "depends_on": [str(x) for x in deps],
            "scope_files": [str(x) for x in scope],
            "external_ref": s.get("external_ref") or None,
        })
    _validate_stories_against_features(
        cleaned,
        {f["feature_key"] for f in features_cleaned},
        existing_feature_keys,
    )
    return features_cleaned, cleaned


def _validate_stories_payload(
    data: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sanity-check the LLM's JSON. Returns ``(features, stories)``.

    Raises ValueError with a precise message on shape violations so
    the caller can surface it to the operator instead of writing a
    corrupt batch into the DB.
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
        deps = s.get("depends_on") or []
        if not isinstance(deps, list):
            raise ValueError(f"{key} depends_on must be a list")
        for d in deps:
            if d not in seen_keys:
                raise ValueError(
                    f"{key} depends_on '{d}' which is not declared earlier"
                )
        scope = s.get("scope_files") or []
        if not isinstance(scope, list):
            raise ValueError(f"{key} scope_files must be a list")
        feature = (s.get("feature") or "").strip()
        cleaned.append({
            "title": title.strip(),
            "feature": feature or None,
            "description": s.get("description") or None,
            "acceptance_criteria": [str(x) for x in ac],
            "depends_on": [str(x) for x in deps],
            "scope_files": [str(x) for x in scope],
            "external_ref": s.get("external_ref") or None,
        })
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
    forbids it — some models add fences anyway. Shared with
    ``harness.batch_sizing``; same shape applies to every JSON-mode
    LLM call we make."""
    s = content.strip()
    if s.startswith("```"):
        first_newline = s.find("\n")
        if first_newline != -1:
            s = s[first_newline + 1:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


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
    augment_existing: list[dict[str, Any]] = []
    augment_existing_features: list[dict[str, Any]] = []
    try:
        _peek_conn = story_state.open_story_db()
        try:
            augment_existing = story_state.list_stories(_peek_conn, app_name)
            augment_existing_features = story_state.list_features(_peek_conn, app_name)
        finally:
            _peek_conn.close()
    except Exception as exc:  # noqa: BLE001 — fall back to from-scratch
        logger.info("[decomposition] augment-mode peek skipped: %s", exc)
        augment_existing = []
        augment_existing_features = []

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
        )
    else:
        prompt = _build_decomposition_prompt(spec_req, spec_arch, workspace)
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
            "current_gate": "STORIES",
            "node_state": {
                "current_node": "decomposition",
                "decomposition_complete": True,
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
            "current_gate": "STORIES",
            "node_state": {
                "current_node": "decomposition",
                "decomposition_complete": True,
                "error": f"invalid_json: {exc}",
                "story_count": 0,
            },
            "budget_remaining_usd": budget,
        }

    try:
        if augment_mode:
            existing_keys = {
                f.get("feature_key") for f in augment_existing_features
                if f.get("feature_key")
            }
            features_cleaned, cleaned = _validate_augment_payload(
                data, existing_feature_keys=existing_keys,
            )
        else:
            features_cleaned, cleaned = _validate_stories_payload(data)
    except ValueError as exc:
        logger.error("[decomposition] payload validation failed: %s", exc)
        return {
            "current_gate": "STORIES",
            "node_state": {
                "current_node": "decomposition",
                "decomposition_complete": True,
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
