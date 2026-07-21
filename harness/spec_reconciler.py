"""Deterministic reconciler: SPEC_REQUIREMENTS.md is the source of truth.

Runs after ``decomposition_node`` in agile mode. Parses
``docs/SPEC_REQUIREMENTS.md`` directly, then rewrites the workspace's
``features`` and ``stories`` rows using spec-authored IDs
(``FEAT-NNN``, ``STORY-NNN``, ``STORY-NFR-NNN``).  LLM-produced
``scope_files`` are preserved by matching LLM story rows to spec
stories on ``story_key`` first, then by title similarity.

Motivation — the decomposition LLM has three observed failure modes
that this reconciler contains:

1. Silently renumbering ``STORY-NNN`` → ``STORY-N``. The old prompt
   at ``_build_decomposition_prompt`` even instructed this
   explicitly, and ``story_state.create_stories`` overwrites
   whatever key the LLM sent. Result: downstream ``depends_on``
   refs inherited from the spec become dangling.
2. Dropping stories when the spec exceeds ``MAX_STORIES_PER_PASS``.
   The prompt tells the LLM to "merge the closest-coupled ones and
   put the leftovers in a final polish story". Real specs with 40+
   stories collapse to 16 silently.
3. Fabricating features that don't exist in the spec (e.g. bundling
   enabler stories into a synthesised "Platform" bucket instead of
   parking each ``STORY-NFR-NNN`` under the parent feature the spec
   nominates).

The reconciler eliminates all three by making the spec authoritative
for structure. LLM output is retained for path hints only.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from difflib import SequenceMatcher
from typing import Any, Optional

from harness import story_state
from harness.req_ids import canonicalize_req_key

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structural regexes — SPEC_REQUIREMENTS.md Agile format (Path A)
# ---------------------------------------------------------------------------

# ``### Feature: FEAT-001 — Company Search & Filing Discovery``
_FEAT_RE = re.compile(
    r"^###\s+Feature:\s+(FEAT-\d+)\s+—\s+(.+?)\s*$",
    re.M,
)

# Story headings — accepts both level-4 (regular stories under a
# feature) and level-3 (enabler stories often live at feature depth
# in SAFe specs).
#
#   ``#### Story: STORY-001 — Ticker & Name Search``     (level 4)
#   ``#### Enabler Story: STORY-NFR-006 — Doc Caching``  (level 4)
#   ``### Enabler Story: STORY-NFR-001 — Perf``          (level 3)
#
# 2026-07-04 fix — the ciod spec used ``### Enabler Story`` (level 3)
# and the strict ``^####`` prefix silently skipped every enabler
# story. All six of ciod's ``STORY-NFR-*`` requirements ended up in
# the ``requirements`` table (parsed by ``req_ids`` with a looser
# heading regex) but never landed in ``stories`` → the identity-link
# fix from earlier in the day had nothing to link. Loosening the
# heading level closes the gap without breaking existing
# ``#### Story`` layouts.
#
# Level range: at LEAST 3 hashes (SAFe convention floor) and up to
# 6 (Markdown's max h6). Uses a bounded quantifier so a stray
# ``## Story:`` at page-title level doesn't get promoted.
_STORY_RE = re.compile(
    r"^#{3,6}\s+(?:Enabler\s+)?Story:\s+(STORY-[\w-]+)\s+—\s+(.+?)\s*$",
    re.M,
)

# ``**Parent feature:** FEAT-001``
# Case-insensitive: a spec that capitalizes "Feature" must still link its
# stories to the parent feature. When this silently returned None the
# story→feature edge was never written and the feature read as untraced
# (the audit now also rolls coverage up structurally — see
# story_state.requirements_without_satisfying_story — but the direct edge
# should still be populated).
_PARENT_FEAT_RE = re.compile(r"\*\*Parent feature:\*\*\s+(FEAT-\d+)", re.IGNORECASE)

# ``**Parent epic:** EPIC-001`` — declared inside a feature body.
# Used by the reconciler to write ``story_satisfies_req`` edges from
# each story up through its feature to the enclosing epic, so
# structural parent requirements (EPIC / FEAT rows) are traced by
# their descendants and don't linger in the "untraced" report.
_PARENT_EPIC_RE = re.compile(r"\*\*Parent epic:\*\*\s+(EPIC-\d+)")

# ` ```gherkin ... ``` ` blocks inside a story body
_GHERKIN_BLOCK_RE = re.compile(r"```gherkin\s*\n(.+?)\n```", re.S)

# ``Scenario: <title>`` inside a Gherkin block
_SCENARIO_RE = re.compile(r"^\s*Scenario:\s*(.+?)\s*$", re.M)

# Story intent lines — reconstructed into ``description``
_AS_A_RE = re.compile(r"\*\*As a\*\*\s+(.+?)\s*\n", re.M)
_I_WANT_RE = re.compile(r"\*\*I want\*\*\s+(.+?)\s*\n", re.M)
_SO_THAT_RE = re.compile(r"\*\*So that\*\*\s+(.+?)\s*\n", re.M)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_spec_requirements(text: str) -> dict[str, Any]:
    """Extract features and stories from a SPEC_REQUIREMENTS.md agile-format text.

    Returns::

        {
            "features": [
                {"feature_key": "FEAT-001", "name": "...", "description": ""},
                ...,
            ],
            "stories": [
                {
                    "story_key": "STORY-001",
                    "title": "...",
                    "feature": "FEAT-001",
                    "description": "As a ..., I want ..., so that ...",
                    "acceptance_criteria": ["Scenario title", ...],
                },
                ...,
            ],
        }
    """
    features: list[dict[str, Any]] = []
    feat_matches = list(_FEAT_RE.finditer(text))
    for i, m in enumerate(feat_matches):
        # Body spans from THIS feature heading to the next feature
        # heading (or EOF for the last feature). Used to locate the
        # ``**Parent epic:** EPIC-N`` line for structural
        # traceability. 2026-07-04 extension — closes the ciod v12
        # 15/26 epic/feature gap.
        body_start = m.start()
        body_end = (
            feat_matches[i + 1].start()
            if i + 1 < len(feat_matches) else len(text)
        )
        body = text[body_start:body_end]
        pe = _PARENT_EPIC_RE.search(body)
        features.append({
            "feature_key": m.group(1),
            "name": m.group(2).strip(),
            "description": "",
            "parent_epic": pe.group(1) if pe else None,
        })

    stories: list[dict[str, Any]] = []
    seen_story_keys: set[str] = set()
    story_starts = [
        (m.start(), m.group(1), m.group(2).strip())
        for m in _STORY_RE.finditer(text)
    ]
    for i, (start, key, title) in enumerate(story_starts):
        # Some specs list an enabler story once under its parent feature
        # AND again in a standalone NFR appendix. First occurrence wins:
        # it's the one that carries the explicit ``**Parent feature:**``
        # marker (the appendix copy usually doesn't).
        if key in seen_story_keys:
            logger.info(
                "[spec_reconciler] spec has duplicate %s heading — keeping "
                "first occurrence, ignoring subsequent",
                key,
            )
            continue
        seen_story_keys.add(key)
        # A story body ends at the next story heading OR the next feature
        # heading — whichever comes first.
        candidates: list[int] = []
        if i + 1 < len(story_starts):
            candidates.append(story_starts[i + 1][0])
        next_feat = _FEAT_RE.search(text, start + 1)
        if next_feat is not None:
            candidates.append(next_feat.start())
        end = min(candidates) if candidates else len(text)
        body = text[start:end]

        pf = _PARENT_FEAT_RE.search(body)
        parent_feature = pf.group(1) if pf else None

        acs: list[str] = []
        for gb in _GHERKIN_BLOCK_RE.findall(body):
            for s in _SCENARIO_RE.finditer(gb):
                acs.append(s.group(1).strip())

        description = _reconstruct_intent(body)

        stories.append({
            "story_key": key,
            "title": title,
            "feature": parent_feature,
            "description": description,
            "acceptance_criteria": acs,
        })
    return {"features": features, "stories": stories}


def _reconstruct_intent(body: str) -> str:
    a = _AS_A_RE.search(body)
    iw = _I_WANT_RE.search(body)
    st = _SO_THAT_RE.search(body)
    if a and iw and st:
        return (
            f"As a {a.group(1).strip()}, "
            f"I want {iw.group(1).strip()}, "
            f"so that {st.group(1).strip()}."
        )
    return ""


# ---------------------------------------------------------------------------
# Fuzzy match — LLM story → spec story
# ---------------------------------------------------------------------------

_MATCH_THRESHOLD = 0.55


def _normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s or "")).strip().lower()


def _title_similarity(a: str, b: str) -> float:
    """Symmetric similarity: 50% word-set Jaccard + 50% sequence ratio."""
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0.0
    words_a, words_b = set(na.split()), set(nb.split())
    jacc = len(words_a & words_b) / max(len(words_a | words_b), 1)
    seq = SequenceMatcher(None, na, nb).ratio()
    return 0.5 * jacc + 0.5 * seq


def _match_llm_to_spec(
    spec_stories: list[dict[str, Any]],
    llm_stories: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return ``spec_story_key → LLM story dict``.

    Match precedence:

    1. Exact ``story_key`` match. The LLM sometimes emits spec-authored
       IDs verbatim (post-prompt-relaxation path). Trust when it does.
    2. Best fuzzy title similarity above ``_MATCH_THRESHOLD``.
    3. No match (spec story keeps empty ``scope_files``).
    """
    by_key: dict[str, dict[str, Any]] = {}
    llm_by_key = {ll["story_key"]: ll for ll in llm_stories}
    llm_unused = list(llm_stories)

    for spec in spec_stories:
        hit = llm_by_key.get(spec["story_key"])
        if hit is not None:
            by_key[spec["story_key"]] = hit
            if hit in llm_unused:
                llm_unused.remove(hit)

    for spec in spec_stories:
        if spec["story_key"] in by_key:
            continue
        best_score = 0.0
        best_ll: Optional[dict[str, Any]] = None
        for ll in llm_unused:
            score = _title_similarity(spec["title"], ll["title"])
            if score > best_score:
                best_score = score
                best_ll = ll
        if best_ll is not None and best_score >= _MATCH_THRESHOLD:
            by_key[spec["story_key"]] = best_ll
            llm_unused.remove(best_ll)
            logger.debug(
                "[spec_reconciler] fuzzy match %.2f: spec %s (%r) ← LLM %s (%r)",
                best_score, spec["story_key"], spec["title"],
                best_ll["story_key"], best_ll["title"],
            )

    if llm_unused:
        # Finsearch session 44c5e194 root cause B1: this was an INFO
        # log and got ignored. Bump to WARNING and be explicit about
        # what the operator should check — either SPEC_REQUIREMENTS.md
        # is missing rows the planner invented, or the planner
        # hallucinated stories that shouldn't have been created.
        # Either way, the run will produce features that no story
        # attribution can point at.
        first5 = ", ".join(
            f"{s['story_key']}={s['title']!r}" for s in llm_unused[:5]
        )
        logger.warning(
            "[spec_reconciler] SPEC DRIFT: %d LLM-invented story(s) "
            "have no counterpart in SPEC_REQUIREMENTS.md — the planner "
            "hallucinated them, or the spec is missing rows. Either "
            "add them to docs/SPEC_REQUIREMENTS.md (with feature + "
            "acceptance criteria) or fix the planner prompt before "
            "the next run. First 5: %s",
            len(llm_unused), first5,
        )
        # Full list at DEBUG for the operator digging in later.
        if len(llm_unused) > 5:
            logger.debug(
                "[spec_reconciler] full drift list: %s",
                ", ".join(
                    f"{s['story_key']}={s['title']!r}" for s in llm_unused
                ),
            )
    return by_key


# ---------------------------------------------------------------------------
# DB rewrite
# ---------------------------------------------------------------------------


def _wipe_workspace(conn: sqlite3.Connection, workspace: str) -> None:
    """Delete stories/features rows for ``workspace``.

    FK cascade wipes ``acceptance_criteria``, ``batch_stories``,
    ``defects``, ``file_links``, ``commits``, ``story_satisfies_req``,
    and ``test_verifies_ac`` transitively via ``story_id``. The
    ``foreign_keys=ON`` pragma is set by ``story_state.open_story_db``.
    """
    conn.execute("DELETE FROM stories WHERE workspace=?", (workspace,))
    conn.execute("DELETE FROM features WHERE workspace=?", (workspace,))


def _insert_features(
    conn: sqlite3.Connection,
    workspace: str,
    features: list[dict[str, Any]],
    now: str,
) -> dict[str, int]:
    """Insert features. Returns ``feature_key → feature_id``."""
    ids: dict[str, int] = {}
    for f in features:
        cur = conn.execute(
            "INSERT INTO features(workspace, feature_key, name, description, created_at)"
            " VALUES(?, ?, ?, ?, ?)",
            (workspace, f["feature_key"], f["name"], f.get("description", ""), now),
        )
        ids[f["feature_key"]] = int(cur.lastrowid)
    return ids


def _insert_story(
    conn: sqlite3.Connection,
    workspace: str,
    story_key: str,
    feature_id: int,
    title: str,
    description: str,
    scope_files: list[str],
    now: str,
) -> int:
    cur = conn.execute(
        "INSERT INTO stories("
        " workspace, story_key, feature_id, title, description,"
        " depends_on, scope_files, status, external_ref, build_kind, cr_ids,"
        " created_at"
        ") VALUES(?, ?, ?, ?, ?, '[]', ?, 'planned', NULL, 'greenfield', NULL, ?)",
        (
            workspace, story_key, feature_id, title, description,
            json.dumps(list(scope_files or [])), now,
        ),
    )
    return int(cur.lastrowid)


def _insert_acs(
    conn: sqlite3.Connection,
    workspace: str,
    story_id: int,
    story_key: str,
    acs: list[str],
) -> None:
    for i, text in enumerate(acs):
        conn.execute(
            "INSERT INTO acceptance_criteria("
            " workspace, story_id, ac_key, text, ordinal"
            ") VALUES(?, ?, ?, ?, ?)",
            (workspace, story_id, f"{story_key}.AC-{i + 1}", text, i),
        )


def reconcile_workspace_from_spec(
    conn: sqlite3.Connection,
    workspace: str,
    spec_path: str,
) -> dict[str, Any]:
    """Overwrite the workspace's stories/features rows with spec-authored data.

    LLM's ``scope_files`` are carried over by ``story_key`` (exact) or
    fuzzy title match. Everything else in ``stories`` / ``features`` /
    ``acceptance_criteria`` comes from the spec.
    """
    with open(spec_path, "r", encoding="utf-8") as fh:
        spec_text = fh.read()
    parsed = parse_spec_requirements(spec_text)

    llm_stories = story_state.list_stories(conn, workspace)
    llm_summary = {
        "story_count": len(llm_stories),
        "feature_count": len(story_state.list_features(conn, workspace)),
    }
    matches = _match_llm_to_spec(parsed["stories"], llm_stories)

    feature_keys = {f["feature_key"] for f in parsed["features"]}
    orphan_stories = [
        s for s in parsed["stories"]
        if not s["feature"] or s["feature"] not in feature_keys
    ]
    # Enabler stories often live under a standalone "### Enabler Stories —
    # Non-Functional Requirements" section without a per-story
    # ``**Parent feature:**`` line — they're cross-cutting concerns. Bundle
    # them into a synthesised ``PLATFORM`` feature so we don't silently
    # drop 10+ stories of NFR coverage.
    if orphan_stories:
        synth_key = "PLATFORM"
        if synth_key not in feature_keys:
            parsed["features"].append({
                "feature_key": synth_key,
                "name": "Platform (Non-Functional & Enabler Stories)",
                "description": (
                    "Synthesised bucket for spec-authored enabler stories "
                    "(NFRs, cross-cutting infra) that carry no explicit "
                    "**Parent feature:** marker."
                ),
            })
            feature_keys.add(synth_key)
        for s in orphan_stories:
            s["feature"] = synth_key
        logger.info(
            "[spec_reconciler] %d spec stories had no explicit parent "
            "feature — attached to synthesised %s: %s",
            len(orphan_stories), synth_key,
            ", ".join(s["story_key"] for s in orphan_stories),
        )
    orphan_stories_after = [
        s["story_key"] for s in parsed["stories"]
        if not s["feature"] or s["feature"] not in feature_keys
    ]

    now = story_state._utcnow_iso()
    with conn:
        _wipe_workspace(conn, workspace)
        feature_id_by_key = _insert_features(
            conn, workspace, parsed["features"], now,
        )
        # Snapshot the requirement universe AFTER wipe (which does NOT
        # touch ``requirements`` — cascades hit stories/features/ACs
        # only). Used below to write ``story_satisfies_req`` edges for
        # each spec-authored story.
        req_id_by_key: dict[str, int] = {
            r["req_key"]: r["id"]
            for r in story_state.list_requirements(conn, workspace)
        }
        stories_written = 0
        links_written = 0
        for spec in parsed["stories"]:
            fk = spec["feature"]
            if fk not in feature_id_by_key:
                continue
            # Fold the spec key to the DB storage form (canonical
            # zero-padded). The parser already produces canonical
            # form; ``_canon`` is idempotent so this is defence-in-
            # depth against a future spec parser change.
            spec_key = story_state._canon(spec["story_key"])
            llm_hit = matches.get(spec["story_key"])
            scope_files = (llm_hit or {}).get("scope_files", []) or []
            story_id = _insert_story(
                conn, workspace,
                story_key=spec_key,
                feature_id=feature_id_by_key[fk],
                title=spec["title"],
                description=spec["description"],
                scope_files=scope_files,
                now=now,
            )
            _insert_acs(
                conn, workspace, story_id, spec_key,
                spec["acceptance_criteria"],
            )
            stories_written += 1
            # Root-cause fix (2026-07-04) — ``_wipe_workspace`` above
            # cascades-deletes ``story_satisfies_req`` (FK on
            # story_id). Without re-populating the link table here, the
            # end-of-session traceability audit ALWAYS reports 0%
            # requirement coverage even when every spec story is
            # written to disk. Ciod session 523e86a7 hit this and
            # spun ~376 iterations in the traceability_block HITL loop
            # (see graph.route_after_installation_doc) before external
            # kill. The identity link (``story STORY-N`` satisfies
            # ``requirement STORY-N``) mirrors the SAFe spec convention
            # where every story heading is itself a requirement; when
            # the LLM's story cites additional ``requirement_keys``
            # (FEAT-*, FR-*, NFR-*), they are added on top. Missing
            # requirement rows (rare — happens when the spec was
            # revised between ingest and reconcile) are logged but
            # do not abort — a broken spec must never poison the run.
            candidate_req_keys: set[str] = set()
            self_ref = spec["story_key"]
            if self_ref in req_id_by_key:
                candidate_req_keys.add(self_ref)
            for llm_key in ((llm_hit or {}).get("requirement_keys") or []):
                canonical = canonicalize_req_key(str(llm_key))
                if canonical in req_id_by_key:
                    candidate_req_keys.add(canonical)
            # Structural parent inference (2026-07-04) — the ciod v12
            # 15/26 traceability report showed 5 requirement rows
            # (1 EPIC + 4 FEATs) untraced because coverage was only
            # counted via story identity links. Every story implicitly
            # satisfies its parent feature (via ``spec['feature']``)
            # and, transitively, that feature's parent epic. Write
            # both edges here so structural requirements get traced by
            # their descendants — no dedicated feature/epic story is
            # required.
            if fk in req_id_by_key:
                candidate_req_keys.add(fk)
            for feat in parsed["features"]:
                if feat.get("feature_key") == fk:
                    parent_epic = feat.get("parent_epic")
                    if parent_epic and parent_epic in req_id_by_key:
                        candidate_req_keys.add(parent_epic)
                    break
            for rk in candidate_req_keys:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO story_satisfies_req"
                    "(story_id, requirement_id) VALUES(?, ?)",
                    (story_id, req_id_by_key[rk]),
                )
                links_written += cur.rowcount or 0

    summary = {
        "features_written": len(feature_id_by_key),
        "stories_written": stories_written,
        "story_satisfies_req_written": links_written,
        "spec_features_seen": len(parsed["features"]),
        "spec_stories_seen": len(parsed["stories"]),
        "orphan_stories": len(orphan_stories_after),
        "synth_platform_absorbed": len(orphan_stories),
        "llm_before": llm_summary,
        "fuzzy_matched": len(matches),
    }
    logger.info(
        "[spec_reconciler] wrote %d features, %d stories, %d "
        "story_satisfies_req edges from spec; LLM had %d stories "
        "(%d matched by key/title); orphaned %d",
        summary["features_written"], summary["stories_written"],
        summary["story_satisfies_req_written"],
        summary["llm_before"]["story_count"], summary["fuzzy_matched"],
        summary["orphan_stories"],
    )
    return summary


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


SPEC_REQUIREMENTS_RELPATH = os.path.join("docs", "SPEC_REQUIREMENTS.md")


def spec_reconciler_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node — reconcile workspace stories against SPEC_REQUIREMENTS.md.

    Runs after ``decomposition_node`` in agile mode. Makes the spec
    authoritative for feature/story IDs, titles, parent-feature links,
    and acceptance criteria. LLM ``scope_files`` are preserved via
    key/title match.

    On any failure sets ``reconcile_failed`` in ``node_state`` so the
    router can divert to HITL rather than presenting the LLM's
    renumbered STORIES.md to the operator.
    """
    workspace_path = state.get("workspace_path", "")
    workspace = story_state.app_name_for_workspace(workspace_path)
    spec_path = os.path.join(workspace_path, SPEC_REQUIREMENTS_RELPATH)

    # CR mode adds a slice of new stories on top of previously-done rows —
    # a blind wipe would destroy that history. Skip until we add a
    # scoped-to-planned reconcile mode.
    if state.get("change_request_mode"):
        logger.info(
            "[spec_reconciler] change_request_mode=True — skipping reconcile "
            "so historical done rows are preserved."
        )
        return {
            "node_state": {
                "current_node": "spec_reconciler",
                "reconciled": False,
                "skipped_reason": "change_request_mode",
            },
        }

    if not os.path.isfile(spec_path):
        logger.warning(
            "[spec_reconciler] %s not found — skipping reconcile "
            "(non-agile flow?)",
            spec_path,
        )
        return {
            "node_state": {
                "current_node": "spec_reconciler",
                "reconciled": False,
                "skipped_reason": "spec_missing",
            },
        }

    conn = story_state.open_story_db()
    try:
        summary = reconcile_workspace_from_spec(conn, workspace, spec_path)
        story_state.regenerate_markdown_views(conn, workspace_path)
    except Exception as e:
        conn.close()
        logger.exception("[spec_reconciler] failed: %s", e)
        return {
            "node_state": {
                "current_node": "spec_reconciler",
                "reconcile_failed": True,
                "error": str(e),
            },
        }

    # P1 fail-fast requirement-coverage check. This is the SAME audit the
    # end-of-run traceability gate (installation_doc_node) runs — but a full
    # build and token budget later. Catching an untraced requirement HERE,
    # right after decomposition+reconcile, lets the operator fix the spec /
    # decomposition and re-run before spending any of it. Coverage rolls up
    # the parent hierarchy (story→feature→epic), so only a genuinely
    # uncovered requirement trips this. Fail-open: a coverage-check error must
    # never block the run on its own — the end gate stays as the backstop.
    untraced: list[dict[str, Any]] = []
    try:
        untraced = story_state.requirements_without_satisfying_story(
            conn, workspace,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "[spec_reconciler] early coverage check failed (%s); deferring "
            "to the end-of-run gate.", e,
        )
    finally:
        conn.close()

    node_state: dict[str, Any] = {
        "current_node": "spec_reconciler",
        "reconciled": True,
        **summary,
    }
    out: dict[str, Any] = {"node_state": node_state}

    if not untraced:
        logger.info(
            "[spec_reconciler] requirement coverage complete — every "
            "requirement has a satisfying story (hierarchy rollup applied)."
        )
        return out

    keys = [str(u.get("req_key")) for u in untraced]
    tr_cfg = (state.get("harness_config") or {}).get("traceability", {})
    enforce = bool(tr_cfg.get("enforce", True))
    enforce_reqs = bool(tr_cfg.get("enforce_reqs", enforce))
    logger.warning(
        "[spec_reconciler] %d requirement(s) have no satisfying story after "
        "decomposition: %s (enforce_reqs=%s)",
        len(untraced), ", ".join(keys), enforce_reqs,
    )
    if enforce_reqs:
        print()
        print("========= REQUIREMENT COVERAGE GAP (post-decomposition) =========")
        print(f"{len(untraced)} requirement(s) have no story that satisfies them:")
        for u in untraced:
            print(f"  - {u.get('req_key')} [{u.get('kind')}] {u.get('title')}")
        print()
        print("Decomposition produced no story covering these. Revise the spec")
        print("(docs/SPEC_REQUIREMENTS.md) or the decomposition and re-run.")
        print("Failing fast now saves the full build budget the end-of-run")
        print("traceability gate would spend before reporting the same gap.")
        print("Set traceability.enforce_reqs=false in config/config.json to")
        print("downgrade this to a warning (NOT RECOMMENDED).")
        print("=================================================================")
        node_state["early_req_coverage_gap"] = True
        out["exit_code"] = 1

    return out
