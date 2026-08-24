"""Decomposition-quality review (ADR-0007).

`spec_reconciler` checks structural integrity and `semantic_review` checks
coverage/intent — neither judges *craftsmanship*. This module reviews the
decomposition artifact's quality along the axes those two miss: story
sizing/atomicity, overlap/duplication, AC testability/atomicity/ambiguity,
dependency correctness, and over-/under-decomposition.

Two tiers (the ADR-0003/0005 deterministic-plus-LLM split):

  * :func:`deterministic_findings` — provable defects computed from state.db
    (dangling / circular dependencies, zero-AC stories, duplicate stories, empty
    ACs). No LLM, zero false positives.
  * :func:`review_decomposition_quality` — an adversarial LLM pass on
    ``NodeRole.DECOMPOSITION_REVIEWER`` (independent of the decomposition model)
    scoring the subjective INVEST dimensions.

Config-gated OFF by default (``decomposition.quality_review``). Advisory unless
``decomposition.quality_enforce``. Fail-open at every step — a reviewer error or
a bad state.db never blocks the build on the review's own failure. Phase-1 scope:
findings + advisory/enforce; the Phase-2 bounded auto-remediation is separate.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger("harness.decomposition_review")


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------


def _gather_stories(conn: Any, workspace: str) -> list[dict[str, Any]]:
    """All stories for ``workspace`` with the fields the review needs.

    Reuses ``story_state.list_stories`` so ``depends_on`` / ``scope_files`` come
    parsed and ``acceptance_criteria`` is the ordered AC-text list.
    """
    from harness import story_state

    stories = story_state.list_stories(conn, workspace)
    return [
        {
            "story_key": s.get("story_key"),
            "title": s.get("title") or "",
            "description": s.get("description") or "",
            "feature_key": s.get("feature_key"),
            "depends_on": list(s.get("depends_on") or []),
            "scope_files": list(s.get("scope_files") or []),
            "acceptance_criteria": list(s.get("acceptance_criteria") or []),
        }
        for s in stories
        if s.get("story_key")
    ]


# ---------------------------------------------------------------------------
# Tier 1 — deterministic quality gate (no LLM)
# ---------------------------------------------------------------------------


def _finding(story_key: str, dimension: str, severity: str, problem: str,
             suggested_action: str, *, source: str) -> dict[str, Any]:
    return {
        "story_key": story_key,
        "dimension": dimension,
        "severity": severity,
        "problem": problem,
        "suggested_action": suggested_action,
        "source": source,
    }


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _detect_cycle(deps: dict[str, list[str]]) -> list[str]:
    """Return one dependency cycle (list of story_keys) if any, else []."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {k: WHITE for k in deps}
    stack: list[str] = []

    def visit(node: str) -> list[str]:
        color[node] = GRAY
        stack.append(node)
        for nxt in deps.get(node, []):
            if nxt not in color:
                continue  # dangling — handled separately
            if color[nxt] == GRAY:
                # cycle: from nxt's position in stack to the end
                idx = stack.index(nxt)
                return stack[idx:] + [nxt]
            if color[nxt] == WHITE:
                found = visit(nxt)
                if found:
                    return found
        color[node] = BLACK
        stack.pop()
        return []

    for k in deps:
        if color[k] == WHITE:
            found = visit(k)
            if found:
                return found
    return []


def deterministic_findings(stories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Provable decomposition defects. Pure — no DB, no LLM, no false positives."""
    findings: list[dict[str, Any]] = []
    keys = {s["story_key"] for s in stories}
    deps = {s["story_key"]: [d for d in s["depends_on"]] for s in stories}

    # Dangling dependencies.
    for s in stories:
        for dep in s["depends_on"]:
            if dep not in keys:
                findings.append(_finding(
                    s["story_key"], "dependency", "high",
                    f"depends_on references {dep!r}, which is not a defined story",
                    "add_dependency", source="deterministic"))

    # Circular dependency (report once, naming the cycle).
    cycle = _detect_cycle(deps)
    if cycle:
        findings.append(_finding(
            cycle[0], "dependency", "high",
            f"circular dependency: {' -> '.join(cycle)}",
            "resize", source="deterministic"))

    # Zero-AC stories + empty ACs.
    for s in stories:
        acs = [a for a in s["acceptance_criteria"]]
        if not acs:
            findings.append(_finding(
                s["story_key"], "ac_quality", "high",
                "story has no acceptance criteria — nothing to test or verify",
                "rewrite_ac", source="deterministic"))
        elif any(not str(a).strip() for a in acs):
            findings.append(_finding(
                s["story_key"], "ac_quality", "medium",
                "story has an empty/blank acceptance criterion",
                "rewrite_ac", source="deterministic"))

    # Duplicate stories: same normalized title AND overlapping scope_files.
    by_title: dict[str, list[dict[str, Any]]] = {}
    for s in stories:
        by_title.setdefault(_normalize_title(s["title"]), []).append(s)
    for norm, group in by_title.items():
        if norm and len(group) > 1:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    # Require an ACTUAL scope overlap — same title alone is not
                    # proof (two stories may share a title but touch different
                    # files, or have no scope yet). Keeps this tier false-positive
                    # free; the LLM tier catches same-title/no-scope overlaps.
                    if set(a["scope_files"]) & set(b["scope_files"]):
                        findings.append(_finding(
                            a["story_key"], "overlap", "medium",
                            f"duplicate of {b['story_key']} — same title "
                            f"({a['title']!r}) and overlapping scope",
                            "merge", source="deterministic"))
    return findings


# ---------------------------------------------------------------------------
# Tier 2 — adversarial LLM quality review
# ---------------------------------------------------------------------------

_QUALITY_RUBRIC = """\
You are auditing a software DECOMPOSITION for QUALITY (craftsmanship), not for \
coverage. You are given the user stories (with descriptions, acceptance criteria, \
and declared dependencies). Judge them against INVEST and report only real \
problems.

Check each story and each pair of stories for:
- right-sizing: a story too LARGE (bundles several behaviours — should be split; \
  name the seams) or trivially SMALL (should be merged).
- overlap: two stories covering the SAME behaviour (should be merged/disambiguated).
- ac_quality: an acceptance criterion that is NOT atomic (bundles behaviours with \
  "and"/"or"), NOT testable (an implementation detail, or no observable outcome), \
  or ambiguous ("works well", "handles errors", "fast").
- dependency: a story that clearly USES what only another story builds but does \
  not declare it in depends_on (a MISSING dependency).
- balance: a feature over-decomposed (fragmented) or under-decomposed (one story \
  hiding several behaviours).

Return ONLY a JSON array (no prose, no code fence). One element per problem found \
(empty array if the decomposition is clean):
{"story_key": "<STORY-key or the first of a pair>", "dimension": \
"right_sizing"|"overlap"|"ac_quality"|"dependency"|"balance", \
"severity": "high"|"medium"|"low", "problem": "<one sentence>", \
"suggested_action": "split"|"merge"|"rewrite_ac"|"add_dependency"|"resize"}

Be specific and strict, but do not invent problems — a clean, well-formed story \
produces no finding. Prefer high/medium severity only for issues that would cause \
real downstream churn.
"""

_VALID_DIMENSIONS = frozenset({
    "right_sizing", "overlap", "ac_quality", "dependency", "balance",
})
_VALID_ACTIONS = frozenset({
    "split", "merge", "rewrite_ac", "add_dependency", "resize",
})


def _build_quality_prompt(stories: list[dict[str, Any]], *, max_stories: int) -> str:
    lines = [_QUALITY_RUBRIC, "", "## Stories"]
    for s in stories[:max_stories]:
        feat = f" (feature {s['feature_key']})" if s.get("feature_key") else ""
        lines.append(f"\n{s['story_key']}: {s['title']}{feat}")
        desc = (s.get("description") or "").strip()
        if desc:
            lines.append(f"  Description: {desc[:400]}")
        if s["depends_on"]:
            lines.append(f"  depends_on: {', '.join(s['depends_on'])}")
        for ac in s["acceptance_criteria"][:12]:
            lines.append(f"  - AC: {ac}")
    if len(stories) > max_stories:
        lines.append(f"\n(+{len(stories) - max_stories} more stories not shown)")
    return "\n".join(lines)


async def review_decomposition_quality(
    gateway: Any, stories: list[dict[str, Any]], budget: float, *, max_stories: int = 60,
) -> tuple[list[dict[str, Any]], float]:
    """Ask DECOMPOSITION_REVIEWER to score decomposition craft. Returns
    ``(findings, budget)``. Fail-open: empty on no reviewer / dispatch error /
    non-JSON — never blocks on its own failure.
    """
    from harness.gateway import NodeRole
    from harness.decomposition import strip_json_fence

    if not stories:
        return [], budget
    if not gateway.select_model(NodeRole.DECOMPOSITION_REVIEWER):
        logger.info("[decomposition_review] no decomposition_reviewer model "
                    "configured; skipping quality review.")
        return [], budget

    try:
        response, budget = await gateway.dispatch(
            messages=[{"role": "user",
                       "content": _build_quality_prompt(stories, max_stories=max_stories)}],
            role=NodeRole.DECOMPOSITION_REVIEWER,
            budget_remaining_usd=budget,
            cache_family="decomposition_reviewer:quality",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[decomposition_review] dispatch failed: %s; skipping.", exc)
        return [], budget

    raw = strip_json_fence(getattr(response, "content", "") or "")
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("[decomposition_review] reviewer returned non-JSON; skipping.")
        return [], budget
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    if not isinstance(data, list):
        return [], budget

    valid_keys = {s["story_key"] for s in stories}
    findings: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        sk = str(item.get("story_key") or "").strip()
        dim = str(item.get("dimension") or "").strip().lower()
        sev = str(item.get("severity") or "").strip().lower()
        if sk not in valid_keys or dim not in _VALID_DIMENSIONS:
            continue
        if sev not in ("high", "medium", "low"):
            sev = "medium"
        action = str(item.get("suggested_action") or "").strip().lower()
        if action not in _VALID_ACTIONS:
            action = "resize"
        findings.append(_finding(
            sk, dim, sev, str(item.get("problem") or "").strip(), action,
            source="llm"))
    return findings, budget


# ---------------------------------------------------------------------------
# Report + node
# ---------------------------------------------------------------------------


def render_review_doc(findings: list[dict[str, Any]], story_count: int) -> str:
    lines = [
        "# Decomposition Quality Review (ADR-0007)",
        "",
        f"Reviewed {story_count} stor{'y' if story_count == 1 else 'ies'}; "
        f"{len(findings)} finding(s).",
        "",
    ]
    if not findings:
        lines.append("No quality findings — decomposition is well-formed.")
        return "\n".join(lines) + "\n"
    lines.append("| Story | Dimension | Severity | Source | Problem | Action |")
    lines.append("|---|---|---|---|---|---|")
    for f in sorted(findings, key=lambda x: (x["severity"] != "high", x["story_key"])):
        prob = f["problem"].replace("|", "\\|")
        lines.append(
            f"| {f['story_key']} | {f['dimension']} | {f['severity']} | "
            f"{f['source']} | {prob} | {f['suggested_action']} |")
    return "\n".join(lines) + "\n"


async def decomposition_quality_review_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node — ADR-0007 decomposition-quality review. Runs after
    reconciliation (and semantic-coverage review) when
    ``decomposition.quality_review`` is on; pass-through no-op otherwise. Advisory
    by default (logs + emits ``decomposition_quality_findings`` + writes
    ``docs/DECOMPOSITION_REVIEW.md``); sets ``decomposition_quality_gap`` +
    ``exit_code=1`` when ``decomposition.quality_enforce`` is also on. Fail-open.
    """
    from harness.graph import get_gateway
    from harness import story_state

    cfg = (state.get("decomposition_config") or {})
    node_state: dict[str, Any] = {"current_node": "decomposition_quality_review"}
    budget = state.get("budget_remaining_usd", 0.0)
    out: dict[str, Any] = {"node_state": node_state, "budget_remaining_usd": budget}

    if not bool(cfg.get("quality_review", False)):
        node_state["skipped"] = True
        return out

    enforce = bool(cfg.get("quality_enforce", False))
    max_stories = int(cfg.get("max_stories_per_review", 60))
    workspace_path = state.get("workspace_path") or os.getcwd()

    try:
        app = story_state.app_name_for_workspace(workspace_path)
        conn = story_state.open_story_db(workspace_path=workspace_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[decomposition_review] DB unavailable: %s; skipping.", exc)
        return out
    try:
        stories = _gather_stories(conn, app)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[decomposition_review] gather failed: %s; skipping.", exc)
        return out
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    if not stories:
        return out

    findings = deterministic_findings(stories)
    gateway = get_gateway()
    if gateway is not None:
        llm_findings, budget = await review_decomposition_quality(
            gateway, stories, budget, max_stories=max_stories)
        findings.extend(llm_findings)
    out["budget_remaining_usd"] = budget
    node_state["decomposition_quality_findings"] = findings

    try:
        doc_path = os.path.join(workspace_path, "docs", "DECOMPOSITION_REVIEW.md")
        os.makedirs(os.path.dirname(doc_path), exist_ok=True)
        with open(doc_path, "w", encoding="utf-8") as fh:
            fh.write(render_review_doc(findings, len(stories)))
    except OSError as exc:
        logger.debug("[decomposition_review] could not write review doc: %s", exc)

    try:
        from harness.observability import emit_event
        emit_event(
            "decomposition_quality_findings",
            reviewed=len(stories), findings=len(findings),
            high=sum(1 for f in findings if f["severity"] == "high"),
            enforce=enforce,
        )
    except Exception:  # noqa: BLE001
        pass

    if not findings:
        logger.info("[decomposition_review] %d stories reviewed — no quality "
                    "findings.", len(stories))
        return out

    high = [f for f in findings if f["severity"] == "high"]
    logger.warning("[decomposition_review] %d quality finding(s) across %d stories "
                   "(%d high, enforce=%s):", len(findings), len(stories),
                   len(high), enforce)
    for f in findings:
        logger.warning("  - %s [%s/%s]: %s -> %s", f["story_key"], f["dimension"],
                       f["severity"], f["problem"], f["suggested_action"])

    # Enforce blocks only on HIGH-severity findings — the provable/structural ones
    # and the model's most confident calls — so a subjective low/medium never
    # stalls a headless run.
    if enforce and high:
        print()
        print("===== DECOMPOSITION QUALITY GAP (post-decomposition) =====")
        print(f"{len(high)} high-severity decomposition quality issue(s):")
        for f in high:
            print(f"  - {f['story_key']} [{f['dimension']}]: {f['problem']} "
                  f"-> {f['suggested_action']}")
        print()
        print("Revise the spec / stories (see docs/DECOMPOSITION_REVIEW.md), then "
              "re-run. Set decomposition.quality_enforce=false to downgrade this "
              "to an advisory warning.")
        print("=========================================================")
        node_state["decomposition_quality_gap"] = True
        out["exit_code"] = 1

    return out


def route_after_decomposition_quality(state: dict[str, Any]) -> str:
    """Enforced high-severity gap -> END (bounded, one shot); else the STORIES
    gate (human_gatekeeper_node)."""
    from langgraph.graph import END

    ns = state.get("node_state", {}) or {}
    if ns.get("decomposition_quality_gap"):
        return END
    return "human_gatekeeper_node"


__all__ = [
    "deterministic_findings", "review_decomposition_quality",
    "render_review_doc", "decomposition_quality_review_node",
    "route_after_decomposition_quality",
]
