"""Requirement gap-fill (P2).

When decomposition leaves a *feature* requirement with no covering story, the
end-of-run traceability gate (and the P1 fail-fast in ``spec_reconciler_node``)
blocks the build. Rather than always deferring to a human spec edit, this
module drafts a covering story with an LLM, optionally has a second LLM review
it, and writes it into ``docs/SPEC_REQUIREMENTS.md`` so the next
``reconcile_workspace_from_spec`` pass closes the gap durably.

Why write into the spec (not the DB): ``reconcile_workspace_from_spec`` wipes
and rebuilds the ``stories`` table from ``SPEC_REQUIREMENTS.md`` on every pass,
so a story only survives if it exists as a spec heading. A ``#### Story:``
heading carrying a ``**Parent feature:** FEAT-XXX`` marker is associated to its
feature by the marker (not document position), so appending a well-formed
block anywhere in the file is sufficient — the feature then rolls up as covered
(see ``story_state.requirements_without_satisfying_story``).

Scope: only ``kind == "feat"`` gaps are auto-filled — a child story rolls up to
its parent feature. Other kinds fall through to the P1 fail-fast. The two LLM
calls use ``NodeRole.DECOMPOSITION`` (generation) and, when configured,
``NodeRole.DECOMPOSITION_REVIEWER`` (review), so an operator can route them to
two different models from config.json.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger("harness.gap_fill")

SPEC_REQUIREMENTS_RELPATH = os.path.join("docs", "SPEC_REQUIREMENTS.md")

# Matches a functional story heading key (STORY-7) but NOT an NFR/enabler key
# (STORY-NFR-3), so freshly-minted keys never collide with either family.
_STORY_KEY_RE = re.compile(r"^####\s+Story:\s+STORY-(\d+)\b", re.MULTILINE)


def next_story_ordinal(spec_text: str) -> int:
    """Smallest functional-story ordinal not already used in the spec.

    Scans ``#### Story: STORY-<n>`` headings and returns ``max(n) + 1`` (or 1
    when none exist), so appended stories get collision-free keys.
    """
    ordinals = [int(m.group(1)) for m in _STORY_KEY_RE.finditer(spec_text or "")]
    return (max(ordinals) + 1) if ordinals else 1


def render_story_block(
    *,
    story_key: str,
    title: str,
    parent_feature: str,
    as_a: str,
    i_want: str,
    so_that: str,
    acceptance_criteria: list[str],
) -> str:
    """Render one SAFe story as a ``SPEC_REQUIREMENTS.md`` markdown block.

    The shape mirrors what ``req_ids.parse_spec_requirements`` and
    ``spec_reconciler.reconcile_workspace_from_spec`` parse: a ``#### Story:``
    heading, a ``**Parent feature:**`` marker, the user-story triplet, and one
    ``gherkin`` fenced block per acceptance criterion.
    """
    so_that = so_that.rstrip(". ")
    lines = [
        f"#### Story: {story_key} — {title.strip()}",
        f"**Parent feature:** {parent_feature.strip()}",
        "",
        f"**As a** {as_a.strip()}",
        f"**I want** {i_want.strip()}",
        f"**So that** {so_that}.",
        "",
    ]
    for ac in acceptance_criteria or []:
        scenario = str(ac).strip()
        if not scenario:
            continue
        # Accept either a bare scenario title or a full Given/When/Then body.
        if "\n" in scenario or scenario.lower().startswith("scenario:"):
            body = scenario
        else:
            body = (
                f"Scenario: {scenario}\n"
                "  Given the system is in a valid state\n"
                "  When the capability is exercised\n"
                "  Then the expected outcome holds"
            )
        if not body.lower().lstrip().startswith("scenario:"):
            body = "Scenario: " + body
        lines.append("```gherkin")
        lines.append(body)
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def append_stories_to_spec(
    spec_path: str, drafts: list[dict[str, Any]],
) -> list[str]:
    """Append rendered story blocks to ``SPEC_REQUIREMENTS.md``.

    ``drafts`` are validated story dicts (see :func:`_coerce_draft`). Assigns
    each a fresh, non-colliding ``STORY-<n>`` key based on the current spec
    content. Returns the list of assigned story keys (empty if nothing valid
    was appended). Idempotency is the caller's concern — this always appends.
    """
    if not drafts:
        return []
    try:
        with open(spec_path, "r", encoding="utf-8") as f:
            spec_text = f.read()
    except OSError as exc:
        logger.warning("[gap_fill] cannot read spec %s: %s", spec_path, exc)
        return []

    ordinal = next_story_ordinal(spec_text)
    blocks: list[str] = []
    keys: list[str] = []
    for d in drafts:
        story_key = f"STORY-{ordinal:03d}"
        blocks.append(render_story_block(
            story_key=story_key,
            title=d["title"],
            parent_feature=d["parent_feature"],
            as_a=d.get("as_a", "user"),
            i_want=d.get("i_want", d["title"]),
            so_that=d.get("so_that", "the requirement is satisfied"),
            acceptance_criteria=d.get("acceptance_criteria", []),
        ))
        keys.append(story_key)
        ordinal += 1

    addition = (
        "\n<!-- gap-fill: auto-generated stories closing requirement-coverage "
        "gaps left by decomposition (harness.gap_fill) -->\n\n"
        + "\n".join(blocks)
    )
    try:
        with open(spec_path, "a", encoding="utf-8") as f:
            f.write(addition)
    except OSError as exc:
        logger.warning("[gap_fill] cannot append to spec %s: %s", spec_path, exc)
        return []
    return keys


def _coerce_draft(
    raw: Any, fillable_by_key: dict[str, dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Validate one LLM-produced story dict against the fillable features.

    Returns a normalized draft, or ``None`` when the entry is malformed or its
    ``parent_feature`` isn't one of the uncovered features we asked to fill
    (guards against the model inventing coverage for the wrong requirement).
    """
    if not isinstance(raw, dict):
        return None
    parent = str(raw.get("parent_feature") or raw.get("requirement_key") or "").strip()
    if parent not in fillable_by_key:
        return None
    title = str(raw.get("title") or "").strip()
    if not title:
        return None
    acs = raw.get("acceptance_criteria")
    if not isinstance(acs, list):
        acs = []
    return {
        "parent_feature": parent,
        "title": title,
        "as_a": str(raw.get("as_a") or "user").strip(),
        "i_want": str(raw.get("i_want") or title).strip(),
        "so_that": str(raw.get("so_that") or "the requirement is satisfied").strip(),
        "acceptance_criteria": [str(a).strip() for a in acs if str(a).strip()],
    }


def _build_draft_prompt(fillable: list[dict[str, Any]]) -> str:
    """Prompt the DECOMPOSITION model to draft one covering story per feature."""
    lines = [
        "The following FEATURE requirements were decomposed into no covering "
        "story, so the build's requirement traceability is incomplete. For "
        "EACH feature below, write exactly one SAFe user story that, if "
        "implemented, satisfies that feature. A story parented under a feature "
        "counts as covering it.",
        "",
        "Return ONLY a JSON array (no prose, no code fence). Each element:",
        '{"parent_feature": "<FEAT-key>", "title": "<short title>", '
        '"as_a": "<role>", "i_want": "<capability>", '
        '"so_that": "<benefit>", "acceptance_criteria": '
        '["Scenario: ... Given ... When ... Then ...", ...]}',
        "",
        "One array element per feature. parent_feature MUST be the exact "
        "FEAT-key given. Acceptance criteria must be concrete and testable.",
        "",
        "Features needing coverage:",
    ]
    for f in fillable:
        lines.append(f"- {f['req_key']} — {f.get('title', '')}")
        body = (f.get("body") or "").strip()
        if body:
            lines.append(f"  {body[:600]}")
    return "\n".join(lines)


async def _dispatch_json_array(
    gateway: Any, role: Any, prompt: str, budget: float, cache_family: str,
) -> tuple[Optional[list], float]:
    """Dispatch a one-shot prompt and parse the reply as a JSON array.

    Returns ``(array, budget)`` or ``(None, budget)`` on any dispatch/parse
    failure — the caller treats ``None`` as "no result, fall through".
    """
    from harness.decomposition import strip_json_fence

    try:
        response, budget = await gateway.dispatch(
            messages=[{"role": "user", "content": prompt}],
            role=role,
            budget_remaining_usd=budget,
            cache_family=cache_family,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[gap_fill] dispatch (%s) failed: %s", cache_family, exc)
        return None, budget
    raw = strip_json_fence(getattr(response, "content", "") or "")
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("[gap_fill] %s returned non-JSON; ignoring.", cache_family)
        return None, budget
    if isinstance(data, dict):
        # Tolerate {"stories": [...]} envelopes.
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    return (data if isinstance(data, list) else None), budget


async def draft_gap_fill_stories(
    gateway: Any, fillable: list[dict[str, Any]], budget: float,
) -> tuple[list[dict[str, Any]], float]:
    """Draft one covering story per fillable feature (DECOMPOSITION role)."""
    from harness.gateway import NodeRole

    fillable_by_key = {f["req_key"]: f for f in fillable}
    arr, budget = await _dispatch_json_array(
        gateway, NodeRole.DECOMPOSITION, _build_draft_prompt(fillable), budget,
        "decomposition:gap_fill",
    )
    if not arr:
        return [], budget
    drafts = [_coerce_draft(x, fillable_by_key) for x in arr]
    drafts = [d for d in drafts if d]
    # At most one story per feature — keep the first draft for each.
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for d in drafts:
        if d["parent_feature"] in seen:
            continue
        seen.add(d["parent_feature"])
        deduped.append(d)
    return deduped, budget


async def review_gap_fill_stories(
    gateway: Any, drafts: list[dict[str, Any]], budget: float,
) -> tuple[list[dict[str, Any]], float]:
    """Optionally review drafts (DECOMPOSITION_REVIEWER role).

    Returns the accepted subset. When the reviewer role isn't configured
    (empty ``decomposition_reviewer_primary``), returns the drafts unchanged.
    A review that fails to dispatch/parse is non-fatal — drafts pass through.
    """
    from harness.gateway import NodeRole

    if not drafts:
        return [], budget
    if not gateway.select_model(NodeRole.DECOMPOSITION_REVIEWER):
        return drafts, budget

    by_feature = {d["parent_feature"]: d for d in drafts}
    prompt = (
        "Review these draft user stories, each meant to cover the named "
        "FEATURE requirement. For each, decide if it genuinely and "
        "sufficiently satisfies the feature's intent.\n\n"
        "Return ONLY a JSON array of the parent_feature keys you ACCEPT, e.g. "
        '["FEAT-002"]. Omit any story that is vague, off-target, or does not '
        "actually satisfy its feature.\n\nDrafts:\n"
        + json.dumps(drafts, indent=2)
    )
    arr, budget = await _dispatch_json_array(
        gateway, NodeRole.DECOMPOSITION_REVIEWER, prompt, budget,
        "decomposition_reviewer:gap_fill",
    )
    if arr is None:
        # Reviewer errored — fail-open, keep the drafts.
        return drafts, budget
    accepted_keys = {str(k).strip() for k in arr if str(k).strip()}
    accepted = [by_feature[k] for k in by_feature if k in accepted_keys]
    dropped = [k for k in by_feature if k not in accepted_keys]
    if dropped:
        logger.info("[gap_fill] reviewer rejected drafts for: %s", dropped)
    return accepted, budget


async def requirement_gap_fill_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node — draft + (optional) review + append covering stories
    for uncovered FEATURE requirements, then hand back to the reconciler.

    Bumps ``loop_counter['requirement_gap_fill_cycles']`` so the router can cap
    the draft→append→reconcile loop. Always routes back to
    ``spec_reconciler_node`` (which re-reconciles and re-checks coverage).
    Fail-open: any error leaves the spec untouched and the reconciler's next
    pass fails fast (P1) as before.
    """
    from harness.graph import get_gateway
    from harness import story_state

    workspace_path = state.get("workspace_path") or os.getcwd()
    app = story_state.app_name_for_workspace(workspace_path)
    budget = state.get("budget_remaining_usd", 0.0)
    loop_counter = dict(state.get("loop_counter") or {})
    cycle = int(loop_counter.get("requirement_gap_fill_cycles", 0) or 0) + 1
    loop_counter["requirement_gap_fill_cycles"] = cycle

    def _return(filled_keys: list[str]) -> dict[str, Any]:
        # Clear the reconciler's stale exit_code=1 while we remediate. If the
        # gap survives, the reconciler's next pass re-asserts exit_code=1 and
        # the router ENDs once the cap is hit.
        return {
            "loop_counter": loop_counter,
            "budget_remaining_usd": budget,
            "exit_code": 0,
            "node_state": {
                "current_node": "requirement_gap_fill",
                "gap_fill_cycle": cycle,
                "gap_fill_story_keys": filled_keys,
            },
        }

    conn = story_state.open_story_db()
    try:
        uncovered = story_state.requirements_without_satisfying_story(conn, app)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[gap_fill] coverage query failed (%s); skipping.", exc)
        return _return([])
    finally:
        conn.close()

    fillable = [r for r in uncovered if str(r.get("kind")) == "feat"]
    if not fillable:
        logger.info(
            "[gap_fill] cycle %d — no feature-level gaps to fill "
            "(%d uncovered, none of kind 'feat'). Deferring to fail-fast.",
            cycle, len(uncovered),
        )
        return _return([])

    gateway = get_gateway()
    if gateway is None:
        logger.warning("[gap_fill] no gateway available; skipping.")
        return _return([])

    logger.info(
        "[gap_fill] cycle %d — drafting covering stories for %d feature(s): %s",
        cycle, len(fillable), [f["req_key"] for f in fillable],
    )
    drafts, budget = await draft_gap_fill_stories(gateway, fillable, budget)
    if not drafts:
        logger.warning("[gap_fill] no usable drafts produced; deferring to fail-fast.")
        return _return([])

    drafts, budget = await review_gap_fill_stories(gateway, drafts, budget)
    if not drafts:
        logger.warning("[gap_fill] reviewer rejected all drafts; deferring to fail-fast.")
        return _return([])

    spec_path = os.path.join(workspace_path, SPEC_REQUIREMENTS_RELPATH)
    filled_keys = append_stories_to_spec(spec_path, drafts)
    if filled_keys:
        logger.info(
            "[gap_fill] appended %d covering story(ies) to the spec: %s "
            "(features: %s). Re-reconciling.",
            len(filled_keys), filled_keys,
            [d["parent_feature"] for d in drafts],
        )
    return _return(filled_keys)
