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
             suggested_action: str, *, source: str,
             remediable: bool = True) -> dict[str, Any]:
    """``remediable=False`` marks a finding the Phase-2 pass must not act
    on even though its ``suggested_action`` says otherwise.

    ``_VALID_ACTIONS`` has no "create" member, so the zero-AC check has to
    report ``rewrite_ac`` while its actual remedy is to WRITE criteria that
    do not exist. Remediation then selects it and finds nothing to rewrite,
    dispatching an LLM call that can only no-op. The outcome is correct but
    accidental; this makes the exclusion explicit.
    """
    return {
        "story_key": story_key,
        "dimension": dimension,
        "severity": severity,
        "problem": problem,
        "suggested_action": suggested_action,
        "source": source,
        "remediable": remediable,
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
                "rewrite_ac", source="deterministic", remediable=False))
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
  EXCEPTION — an acceptance criterion prefixed ``[NFR:<policy>]`` is a
  non-functional policy deliberately attached to every story it constrains
  (ADR-0004). Seeing the SAME ``[NFR:...]`` criterion in several stories is
  correct and expected; it is single-ownership fan-out, not duplication.
  NEVER report overlap on the basis of shared ``[NFR:...]`` criteria, and
  never suggest merging stories because they share one. Judge overlap on
  the UNTAGGED criteria only.
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

# Matches either the embedded-AC tag ``[NFR:NFR-002]`` or a bare policy
# key like ``NFR-002`` in a reviewer's prose, since the model paraphrases.
_NFR_TAG_RE = re.compile(r"\[NFR:[^\]]+\]|\bNFR[-_ ]?\d+\b", re.IGNORECASE)

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
        # Belt-and-braces behind the prompt's NFR exception. ADR-0004
        # deliberately fans one non-functional policy out across every
        # story it constrains, tagging each copy ``[NFR:<policy>]``. A
        # reviewer that has not internalised that reads the repeated
        # criteria as duplication and recommends merging stories that
        # ADR-0004 requires to stay separate — two harness features
        # contradicting each other, with the operator left to arbitrate.
        # lumina-fresh-20260911-1107 produced four of these
        # (overlap/high -> merge, STORY-001..005, all citing the shared
        # NFR-002 security criteria).
        if dim == "overlap" and _NFR_TAG_RE.search(
            str(item.get("problem") or "")
        ):
            logger.info(
                "[decomposition_review] Dropping overlap finding on %s — "
                "cites shared [NFR:...] criteria, which ADR-0004 fans out "
                "across stories on purpose.", sk,
            )
            continue
        findings.append(_finding(
            sk, dim, sev, str(item.get("problem") or "").strip(), action,
            source="llm"))
    return findings, budget


# ---------------------------------------------------------------------------
# Phase 2 — bounded AC remediation (ADR-0007)
# ---------------------------------------------------------------------------
#
# ``quality_enforce`` is binary: findings are either discarded entirely or
# they kill the build with exit_code=1 and a "revise the spec by hand"
# message. lumina-fresh-20260911-1107 produced 14 high-severity findings on
# an ordinary spec, so enforcing would have bricked the run at
# decomposition — strictly worse than ignoring them. The findings were
# discarded instead, and the vague ACs they named went on to feed test
# generation and acceptance-scenario generation.
#
# This is the middle path: one bounded LLM pass that rewrites ONLY the
# acceptance-criterion text the reviewer flagged, in place.
#
# Safety contract — every clause here exists to avoid breaking something
# that currently works:
#
#   * AC identity is (story_id, ac_key) and ``create_acceptance_criteria``
#     UPSERTs on it, preserving the row id. Rewriting text through that
#     path leaves every ``test_verifies_ac`` edge intact, so the
#     traceability audit (`teane audit`, which gates CI) sees no change.
#   * The AC COUNT per story never changes, and ordinals are preserved.
#     Adding or dropping an AC would silently change coverage denominators.
#   * ``[NFR:...]``-tagged criteria are never touched. ADR-0004 owns that
#     text and fans identical copies across stories; rewriting one copy
#     would break single-ownership and desynchronise the rest.
#   * Only ``ac_quality`` findings with ``suggested_action == "rewrite_ac"``
#     qualify. split / merge / resize / add_dependency are structural and
#     are NOT auto-applied — those stay advisory.
#   * Fail-open at every step. A malformed response, a budget refusal, a DB
#     error, an unknown ac_key, an empty rewrite — all leave the original
#     text exactly as it was.

_REMEDIATION_PROMPT = """\
You are tightening acceptance criteria that a quality review flagged as \
not testable or ambiguous.

For each item below, rewrite ONLY the criterion text so it names a \
concrete, observable outcome — a status code, a rendered string, a stored \
value, a specific error. Keep the original intent exactly; do not broaden \
it, do not narrow it, and do not split one criterion into several.

Rules:
- Return the SAME number of items you were given, with the same ac_key values.
- One sentence per criterion. No markdown, no numbering, no commentary.
- If a criterion is already fine, or you cannot improve it without \
inventing a requirement that is not implied, return its ORIGINAL text \
unchanged.
- Never invent a threshold, timeout, or numeric limit that the criterion \
does not already imply. "Fast" becomes an observable outcome, not "under \
200ms" unless 200ms was already stated.

Return ONLY a JSON array (no prose, no code fence):
[{"ac_key": "<unchanged>", "text": "<rewritten criterion>"}]
"""


def _remediable_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The subset this pass will act on: high-severity AC-quality findings
    whose recommended action is a rewrite.

    Structural recommendations (split / merge / resize / add_dependency)
    are deliberately excluded — acting on those would reshape the
    decomposition, which is the operator's call and would invalidate the
    story keys downstream nodes have already committed to.
    """
    return [
        f for f in findings
        if f.get("dimension") == "ac_quality"
        and f.get("severity") == "high"
        and f.get("suggested_action") == "rewrite_ac"
        and f.get("remediable", True)
    ]


def _ac_is_remediable(text: str) -> bool:
    """``[NFR:...]``-tagged criteria are owned by the ADR-0004 embedder,
    which fans identical copies across every story a policy constrains.
    Rewriting one copy desynchronises the rest and breaks the
    single-ownership the tag exists to express."""
    return not _NFR_TAG_RE.match(text.strip())


async def remediate_acceptance_criteria(
    gateway: Any,
    conn: Any,
    workspace: str,
    stories: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    budget: float,
) -> tuple[int, float]:
    """Rewrite flagged acceptance criteria in place. Returns (count, budget).

    Fail-open: returns ``(0, budget)`` unchanged on any problem.
    """
    from harness import story_state

    targets = _remediable_findings(findings)
    if not targets:
        return 0, budget
    flagged_keys = {f.get("story_key") for f in targets}

    # Build the work list from the DB, not from the finding text — the
    # finding says WHICH story is weak, the DB says what its criteria
    # actually are and which row each one is.
    items: list[dict[str, Any]] = []
    try:
        db_stories = story_state.list_stories(conn, workspace)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[decomposition_review] remediation skipped — story read "
            "failed: %s", exc,
        )
        return 0, budget
    story_by_key = {s.get("story_key"): s for s in db_stories}
    for skey in sorted(k for k in flagged_keys if k):
        story = story_by_key.get(skey)
        if not story:
            continue
        try:
            acs = story_state.list_acceptance_criteria(
                conn, workspace, int(story["id"]),
            )
        except Exception:  # noqa: BLE001
            continue
        for ac in acs:
            text = str(ac.get("text") or "")
            if not text or not _ac_is_remediable(text):
                continue
            items.append({
                "ac_key": ac.get("ac_key"),
                "text": text,
                "story_id": int(story["id"]),
                "ordinal": ac.get("ordinal"),
                "story_key": skey,
                "title": story.get("title") or "",
            })
    if not items:
        return 0, budget

    payload = json.dumps(
        [
            {"ac_key": i["ac_key"], "story": i["title"], "text": i["text"]}
            for i in items
        ],
        indent=1,
    )
    from harness.decomposition import strip_json_fence
    from harness.gateway import NodeRole
    try:
        response, budget = await gateway.dispatch(
            messages=[
                {"role": "system", "content": _REMEDIATION_PROMPT},
                {"role": "user", "content": payload},
            ],
            role=NodeRole.DECOMPOSITION_REVIEWER,
            budget_remaining_usd=budget,
            cache_family="decomposition_reviewer:ac_remediation",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[decomposition_review] remediation dispatch failed: %s; "
            "criteria left unchanged.", exc,
        )
        return 0, budget

    raw = strip_json_fence(str(getattr(response, "content", "") or ""))
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        logger.warning(
            "[decomposition_review] remediation response was not JSON; "
            "criteria left unchanged.",
        )
        return 0, budget
    if not isinstance(data, list):
        return 0, budget

    by_key = {i["ac_key"]: i for i in items}
    rewritten = 0
    for entry in data:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("ac_key") or "").strip()
        new_text = str(entry.get("text") or "").strip()
        item = by_key.get(key)
        if item is None or not new_text:
            continue
        if new_text == item["text"]:
            continue  # model judged it already fine
        if not _ac_is_remediable(new_text):
            # Never let a rewrite introduce an NFR tag — that would forge
            # ADR-0004 ownership onto a criterion the embedder did not
            # place.
            continue
        try:
            # Same ac_key, same ordinal → UPSERT preserves the row id, so
            # every test_verifies_ac edge pointing at it survives.
            story_state.create_acceptance_criteria(
                conn, workspace, item["story_id"],
                [{
                    "ac_key": key,
                    "text": new_text,
                    "ordinal": item["ordinal"],
                }],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[decomposition_review] remediation write failed for %s: "
                "%s; leaving original.", key, exc,
            )
            continue
        logger.info(
            "[decomposition_review] AC %s rewritten:\n    was: %s\n    now: %s",
            key, item["text"], new_text,
        )
        rewritten += 1

    if rewritten:
        logger.warning(
            "[decomposition_review] Remediated %d acceptance criterion(s) "
            "in place across %d flagged story(ies). Row ids and ordinals "
            "preserved — traceability edges are intact.",
            rewritten, len(flagged_keys),
        )
    return rewritten, budget


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
    # Phase-2 auto-remediation (ADR-0007). Independent of ``quality_enforce``
    # — that flag is binary (discard everything, or fail the build), and on
    # an ordinary spec the reviewer emits enough high-severity findings that
    # enforcing is unusable. Remediation is the middle path and is safe to
    # leave on: it rewrites flagged criterion TEXT in place, preserving row
    # ids, ordinals and AC counts, so traceability is untouched.
    remediate = bool(cfg.get("quality_remediate", True))
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

    # Phase 2 — bounded AC remediation. Runs BEFORE the enforce gate so a
    # finding the harness just fixed cannot also fail the build; ``high``
    # is recomputed from ``remaining`` below for exactly that reason.
    remediated = 0
    if remediate and gateway is not None:
        try:
            conn = story_state.open_story_db(workspace_path=workspace_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[decomposition_review] remediation skipped — DB "
                "unavailable: %s", exc,
            )
        else:
            try:
                remediated, budget = await remediate_acceptance_criteria(
                    gateway, conn, app, stories, findings, budget,
                )
                out["budget_remaining_usd"] = budget
            except Exception as exc:  # noqa: BLE001 — never break the build
                logger.warning(
                    "[decomposition_review] remediation raised (%s); "
                    "criteria left unchanged.", exc,
                )
            finally:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
    node_state["decomposition_acs_remediated"] = remediated

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

    # A finding whose criterion the harness just rewrote is no longer a
    # reason to stop the build. Without this, remediation and enforcement
    # would contradict each other: the harness fixes the AC and then fails
    # the run for the AC it fixed.
    if remediated:
        _fixed_stories = {f.get("story_key") for f in _remediable_findings(findings)}
        remaining = [
            f for f in findings
            if not (
                f.get("dimension") == "ac_quality"
                and f.get("suggested_action") == "rewrite_ac"
                and f.get("story_key") in _fixed_stories
            )
        ]
    else:
        remaining = findings
    high = [f for f in remaining if f["severity"] == "high"]
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
