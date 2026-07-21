"""Semantic coverage review (P3).

The deterministic traceability gate (P0 rollup + P1 fail-fast) answers a
*syntactic* question: does every requirement have at least one story linked to
it? It cannot tell whether the linked story actually *satisfies* the
requirement's intent, or merely cites it — a story can be parented under a
feature (so the feature rolls up as "covered") while its acceptance criteria
don't fulfil what the feature asks for. That's reward-hackable / hollow
coverage.

This module adds an adversarial LLM pass over the decomposition: for each
FEATURE, it hands the reviewer the feature's intent plus the titles + AC text
of its covering stories and asks whether, taken together, they genuinely
satisfy the feature. Findings are surfaced (log + observability event) and,
when ``traceability.semantic_review_enforce`` is on, block the build the same
way a deterministic gap does.

Config-gated and OFF by default (``traceability.semantic_review``) — it costs
an extra LLM call per run and is a judgement call that can false-positive. Uses
``NodeRole.DECOMPOSITION_REVIEWER`` so it runs on whatever model the operator
routes review to (independent of decomposition generation).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("harness.semantic_review")


def _gather_feature_coverage(conn: Any, workspace: str) -> list[dict[str, Any]]:
    """For each feature requirement, collect its intent + the covering stories
    (title + acceptance-criteria text) parented under it.

    Coverage is read from the requirements' own ``**Parent feature:**`` markers
    (the same source the P0 rollup uses), so it's robust to the inconsistent
    ``features`` / ``stories.feature_id`` tables. Only features that actually
    have covering stories are returned — a feature with none is a *deterministic*
    gap, already handled upstream (P1/P2), not a semantic one.
    """
    from harness.story_state import _PARENT_MARKER_RE

    rows = conn.execute(
        "SELECT req_key, kind, title, body FROM requirements "
        "WHERE workspace = ? ORDER BY id",
        (workspace,),
    ).fetchall()
    reqs = [
        {"req_key": r[0], "kind": r[1], "title": r[2], "body": r[3] or ""}
        for r in rows
    ]
    # child req_key -> parent req_key (from the in-row marker).
    parent_of: dict[str, str] = {}
    for r in reqs:
        m = _PARENT_MARKER_RE.search(r["body"])
        if m:
            parent_of[r["req_key"]] = m.group(1)

    # story_key -> {title, acs[]}
    story_meta: dict[str, dict[str, Any]] = {}
    for r in conn.execute(
        "SELECT story_key, title FROM stories WHERE workspace = ?", (workspace,),
    ).fetchall():
        story_meta[r[0]] = {"title": r[1], "acs": []}
    for r in conn.execute(
        "SELECT s.story_key, ac.text FROM acceptance_criteria ac "
        "JOIN stories s ON s.id = ac.story_id "
        "WHERE ac.workspace = ? ORDER BY ac.ordinal",
        (workspace,),
    ).fetchall():
        if r[0] in story_meta and r[1]:
            story_meta[r[0]]["acs"].append(str(r[1]))

    out: list[dict[str, Any]] = []
    for r in reqs:
        if r["kind"] != "feat":
            continue
        children = [ck for ck, pk in parent_of.items() if pk == r["req_key"]]
        stories = [
            {"story_key": ck, "title": story_meta[ck]["title"],
             "acceptance_criteria": story_meta[ck]["acs"]}
            for ck in children if ck in story_meta
        ]
        if stories:
            out.append({
                "req_key": r["req_key"], "title": r["title"],
                "intent": r["body"], "stories": stories,
            })
    return out


def _build_review_prompt(coverage: list[dict[str, Any]]) -> str:
    lines = [
        "You are auditing a software decomposition for SEMANTIC coverage. For "
        "each FEATURE below you are given its intent and the user stories "
        "parented under it (with their acceptance criteria). Decide whether "
        "those stories, taken TOGETHER, genuinely and sufficiently satisfy the "
        "feature's intent — not merely whether they are related to it.",
        "",
        "Return ONLY a JSON array (no prose, no fence). One element per feature:",
        '{"req_key": "<FEAT-key>", "verdict": "satisfied" | "partial" | '
        '"unsatisfied", "gap": "<what intent is unmet; empty if satisfied>"}',
        "",
        "Judge strictly: a feature is 'partial' if a material part of its intent "
        "has no story/AC, 'unsatisfied' if the stories miss the point entirely.",
        "",
    ]
    for f in coverage:
        lines.append(f"FEATURE {f['req_key']}: {f['title']}")
        intent = (f.get("intent") or "").strip()
        if intent:
            lines.append(f"  Intent: {intent[:700]}")
        for s in f["stories"]:
            lines.append(f"  Story {s['story_key']}: {s['title']}")
            for ac in s["acceptance_criteria"][:12]:
                lines.append(f"    - AC: {ac}")
        lines.append("")
    return "\n".join(lines)


async def review_semantic_coverage(
    gateway: Any, coverage: list[dict[str, Any]], budget: float,
) -> tuple[list[dict[str, Any]], float]:
    """Ask the DECOMPOSITION_REVIEWER whether each feature is semantically
    satisfied by its stories. Returns ``(findings, budget)`` where findings are
    the non-satisfied verdicts. Empty on no coverage, no reviewer configured,
    or any dispatch/parse failure (fail-open — never blocks on its own error).
    """
    from harness.gateway import NodeRole
    from harness.decomposition import strip_json_fence

    if not coverage:
        return [], budget
    if not gateway.select_model(NodeRole.DECOMPOSITION_REVIEWER):
        logger.info(
            "[semantic_review] no decomposition_reviewer model configured; "
            "skipping semantic coverage review.")
        return [], budget

    try:
        response, budget = await gateway.dispatch(
            messages=[{"role": "user", "content": _build_review_prompt(coverage)}],
            role=NodeRole.DECOMPOSITION_REVIEWER,
            budget_remaining_usd=budget,
            cache_family="decomposition_reviewer:semantic_coverage",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[semantic_review] dispatch failed: %s; skipping.", exc)
        return [], budget

    raw = strip_json_fence(getattr(response, "content", "") or "")
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("[semantic_review] reviewer returned non-JSON; skipping.")
        return [], budget
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    if not isinstance(data, list):
        return [], budget

    valid_keys = {f["req_key"] for f in coverage}
    findings: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        rk = str(item.get("req_key") or "").strip()
        verdict = str(item.get("verdict") or "").strip().lower()
        if rk in valid_keys and verdict in ("partial", "unsatisfied"):
            findings.append({
                "req_key": rk, "verdict": verdict,
                "gap": str(item.get("gap") or "").strip(),
            })
    return findings, budget


async def semantic_coverage_review_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node — adversarial semantic-coverage review of the
    decomposition. Runs after the reconciler when
    ``traceability.semantic_review`` is on. Advisory by default (logs + emits a
    ``semantic_coverage_findings`` event); sets ``semantic_coverage_gap`` +
    ``exit_code=1`` when ``traceability.semantic_review_enforce`` is also on.
    Fail-open at every step.
    """
    from harness.graph import get_gateway
    from harness import story_state

    workspace_path = state.get("workspace_path") or os.getcwd()
    app = story_state.app_name_for_workspace(workspace_path)
    budget = state.get("budget_remaining_usd", 0.0)
    tr_cfg = (state.get("harness_config") or {}).get("traceability", {})
    enforce = bool(tr_cfg.get("semantic_review_enforce", False))

    node_state: dict[str, Any] = {"current_node": "semantic_coverage_review"}
    out: dict[str, Any] = {"node_state": node_state, "budget_remaining_usd": budget}

    conn = story_state.open_story_db()
    try:
        coverage = _gather_feature_coverage(conn, app)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[semantic_review] coverage gather failed: %s; skipping.", exc)
        return out
    finally:
        conn.close()

    gateway = get_gateway()
    if gateway is None or not coverage:
        return out

    findings, budget = await review_semantic_coverage(gateway, coverage, budget)
    out["budget_remaining_usd"] = budget
    node_state["semantic_coverage_findings"] = findings

    try:
        from harness.observability import emit_event
        emit_event(
            "semantic_coverage_findings",
            reviewed=len(coverage), findings=len(findings),
            keys=[f["req_key"] for f in findings], enforce=enforce,
        )
    except Exception:  # noqa: BLE001
        pass

    if not findings:
        logger.info(
            "[semantic_review] %d feature(s) reviewed — all semantically "
            "satisfied by their stories.", len(coverage))
        return out

    logger.warning(
        "[semantic_review] %d of %d feature(s) NOT fully satisfied by their "
        "stories (enforce=%s):", len(findings), len(coverage), enforce)
    for f in findings:
        logger.warning(
            "  - %s [%s]: %s", f["req_key"], f["verdict"],
            f["gap"] or "(no detail)")

    if enforce:
        print()
        print("===== SEMANTIC COVERAGE GAP (post-decomposition) =====")
        print(f"{len(findings)} feature(s) are not sufficiently satisfied by "
              "their stories:")
        for f in findings:
            print(f"  - {f['req_key']} [{f['verdict']}]: {f['gap'] or '(no detail)'}")
        print()
        print("Revise the spec (docs/SPEC_REQUIREMENTS.md) so each feature's "
              "stories cover its full intent, then re-run. Set "
              "traceability.semantic_review_enforce=false to downgrade this to "
              "an advisory warning.")
        print("======================================================")
        node_state["semantic_coverage_gap"] = True
        out["exit_code"] = 1

    return out
