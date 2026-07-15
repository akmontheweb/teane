"""Per-story TDD orchestration nodes for the Agile decomposition path.

This module wires the planned-stories table to the existing patching /
compile / review chain. Three nodes live here today (more land in the
test-first / completion / traceability step):

- ``batch_planner_node`` — picks the next batch of dependency-ready
  stories from ``.teane/state.db`` and persists a ``batches`` row.
  No LLM call; pure DB orchestration.
- ``story_loop_node`` — advances the cursor to the next planned
  story inside the current batch. Sets ``current_story_id`` and
  ``story_scope_files`` so the downstream patcher knows what it is
  building.
- Routing helpers (``route_after_batch_planner``,
  ``route_after_story_loop``) — small dispatchers consumed by the
  graph's conditional edges.

The state of record is the workspace's ``.teane/state.db``. These
nodes read and write that DB via ``harness.story_state`` only —
they never compose SQL inline.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Optional

from harness import story_state
from harness.batch_sizing import DETERMINISTIC_BATCH_SIZE as DEFAULT_BATCH_SIZE
from harness.loop_counter_keys import PER_BATCH_CAP_COUNTERS

logger = logging.getLogger(__name__)


__all__ = ["DEFAULT_BATCH_SIZE", "STORY_ZERO_PATCH_CAP"]
"""Cap on stories selected per batch — aliased to
``harness.batch_sizing.DETERMINISTIC_BATCH_SIZE`` so the two modules
share one source of truth. The dependency-aware planner may return
fewer when the graph chokes earlier (e.g. only one story is unblocked
at this moment). The ``agile_defaults.batch_size`` config knob (set
in ``~/.harness/config.json``) overrides this via
``state['story_batch_size']`` — there is no longer a CLI flag for
this value."""

STORY_ZERO_PATCH_CAP = 3
"""Per-story patching attempts allowed before story_loop_node
auto-completes the story and advances the cursor.

When the patching turn lands zero successful patches against the same
``current_story_id`` ``STORY_ZERO_PATCH_CAP`` times in a row, the story
is considered vacuous / mis-identified (the planner generated a story
the patcher has nothing to do for, or the LLM cannot find anything to
emit). Rather than burn the run's budget in a tight patching ↔
story_loop cycle, story_loop_node marks the story ``done`` and lets
``_next_story_in_batch`` advance to the next eligible story. Default
3 — overridable via ``state['story_zero_patch_cap']``."""


# ---------------------------------------------------------------------------
# batch_planner_node
# ---------------------------------------------------------------------------

def batch_planner_node(state: dict[str, Any]) -> dict[str, Any]:
    """Select the next batch of ready stories and record it in the DB.

    Reads ``state['stories_db_path']`` (set by ``decomposition_node``)
    or falls back to ``state['workspace_path']``'s default location.

    **Feature-first invariant** (v4 schema, 2026-06-25): a batch never
    spans features. The planner picks the next feature (by feature_id
    ascending) that has at least one ready story and takes up to
    ``story_batch_size`` of THAT feature's ready stories. Small
    features land in one batch; larger features (more than
    story_batch_size stories) span multiple batches, all tagged with
    the same ``feature_id``.

    Behavior:

    - If every story is already ``done`` → ``batch_planned=False``,
      ``all_complete=True``. The router uses this to fall through to
      the security-scan / installation-doc tail of the graph.
    - If some stories are still ``planned`` but blocked (a dependency
      is ``blocked``/``in_progress``/missing across every feature) →
      returns no batch and logs the stall so the operator sees why
      nothing fired.
    - Otherwise creates a batch with up to ``story_batch_size``
      independent stories from the first feature with ready work,
      and sets ``current_batch_id``.

    Does NOT mark stories ``in_progress`` — that's ``story_loop_node``'s
    job, called once per story as it picks them up.
    """
    workspace_path = state.get("workspace_path", "")
    workspace = story_state.app_name_for_workspace(workspace_path)
    session_id = state.get("session_id", "")
    batch_size = int(
        state.get("story_batch_size") or DEFAULT_BATCH_SIZE
    )
    # Tag the batch as part of an incremental CR layer when the
    # session ingested change requests, so the traceability matrix
    # can split greenfield work from CR-driven increments.
    if state.get("change_request_mode"):
        batch_build_kind = story_state.BUILD_KIND_CR
        batch_cr_ids = sorted({
            int(r.get("cr_id"))
            for r in (state.get("change_request_files") or [])
            if r.get("cr_id") is not None
        }) or None
    else:
        batch_build_kind = story_state.BUILD_KIND_GREENFIELD
        batch_cr_ids = None

    conn = story_state.open_story_db()
    try:
        all_stories = story_state.list_stories(conn, workspace)
        if not all_stories:
            # An empty story DB is NEVER "all done" — it is failure.
            # Earlier shape (``all_complete: True, reason: "no_stories"``)
            # let the router fall through to ``traceability_node`` and
            # the pipeline proceeded to generate code with zero stories
            # backing it (a silent ``decomposition_node`` failure used
            # to land here too, before Bug 2 was fixed). The router now
            # diverts to HITL so the operator can see why no stories
            # were produced.
            logger.error(
                "[batch_planner] No stories in DB; nothing to plan. "
                "This is a failure path — decomposition likely never "
                "ran (non-agile flow misrouted) or its output failed "
                "validation. Routing to HITL."
            )
            return {
                "current_batch_id": 0,
                "exit_code": 1,
                "node_state": {
                    "current_node": "batch_planner",
                    "batch_planned": False,
                    "all_complete": False,
                    "decomposition_missing": True,
                    "error": "no stories in DB — decomposition produced none",
                },
            }

        # ``reopened`` rows count too — story_reopen_node flips drifted
        # DONE stories to that status and the planner needs to pick
        # them back up.
        planned_left = [
            s for s in all_stories if s["status"] in ("planned", "reopened")
        ]
        if not planned_left:
            logger.info("[batch_planner] All stories accounted for; no planned or reopened left.")
            done_count = sum(1 for s in all_stories if s["status"] == "done")
            return {
                "current_batch_id": 0,
                "node_state": {
                    "current_node": "batch_planner",
                    "batch_planned": False,
                    "all_complete": True,
                    "done_count": done_count,
                    "total_count": len(all_stories),
                },
            }

        ready = story_state.get_planned_stories(conn, workspace)
        if not ready:
            # Planned stories exist but none are unblocked. This is a
            # stall — typically a dep chain that hit `blocked` upstream.
            blockers = sorted({
                d for s in planned_left for d in s["depends_on"]
                if any(o["story_key"] == d and o["status"] != "done"
                       for o in all_stories)
            })
            logger.warning(
                "[batch_planner] %d planned stories but none unblocked. "
                "Outstanding deps: %s",
                len(planned_left), ", ".join(blockers) or "(unknown)",
            )
            return {
                "current_batch_id": 0,
                "node_state": {
                    "current_node": "batch_planner",
                    "batch_planned": False,
                    "all_complete": False,
                    "stalled": True,
                    "outstanding_deps": blockers,
                },
            }

        # Feature-first slicing: walk features in id-ascending order
        # and grab the first one that owns at least one ready story.
        # Within that feature we take up to story_batch_size ready
        # stories, preserving their list order (already insertion-sorted
        # by list_stories' ORDER BY s.id).
        features = story_state.list_features(conn, workspace)
        feature_order: list[Optional[int]] = [int(f["id"]) for f in features]
        # Add `None` (unassigned) at the end as a defensive fallback —
        # create_stories rejects feature-less inserts, but a corrupted
        # row or a forced ON DELETE SET NULL could leave a story with
        # feature_id=None. We don't want such stories to be invisible.
        if any(r.get("feature_id") is None for r in ready):
            feature_order.append(None)

        picked: list[dict[str, Any]] = []
        picked_feature_id: Optional[int] = None
        picked_feature_label: Optional[str] = None
        for fid in feature_order:
            slice_ready = [
                r for r in ready
                if (None if r.get("feature_id") is None else int(r["feature_id"])) == fid
            ]
            if not slice_ready:
                continue
            picked = slice_ready[:batch_size]
            picked_feature_id = fid
            if fid is None:
                picked_feature_label = "(unassigned)"
            else:
                picked_feature_label = (
                    picked[0].get("feature_key") or f"feature#{fid}"
                )
            break

        if not picked:
            # Belt-and-braces: ``ready`` is non-empty but no feature
            # matched. Should be unreachable — every ready story has
            # a feature_id (from create_stories) or hits the None
            # fallback above. Treat as a stall rather than crash.
            logger.error(
                "[batch_planner] %d ready stories but none matched any "
                "feature slice — likely a corrupted DB row.", len(ready),
            )
            return {
                "current_batch_id": 0,
                "node_state": {
                    "current_node": "batch_planner",
                    "batch_planned": False,
                    "all_complete": False,
                    "stalled": True,
                    "reason": "feature_slice_empty",
                },
            }

        picked_keys = [s["story_key"] for s in picked]
        batch_id = story_state.start_batch(
            conn, workspace, session_id, picked_keys,
            build_kind=batch_build_kind, cr_ids=batch_cr_ids,
            feature_id=picked_feature_id,
        )
    finally:
        conn.close()

    logger.info(
        "[batch_planner] batch %d planned with %d stories "
        "from feature %s: %s",
        batch_id, len(picked_keys),
        picked_feature_label, ", ".join(picked_keys),
    )
    return {
        "current_batch_id": batch_id,
        "current_story_id": "",
        "story_scope_files": [],
        # Phase E.3 cursor-advance fix: a fresh batch starts with no
        # patched stories so `_next_story_in_batch` is free to pick any
        # ready story by sequence. Without this reset, leftover keys
        # from the previous batch would silently skip ready stories.
        "batch_patched_story_keys": [],
        "node_state": {
            "current_node": "batch_planner",
            "batch_planned": True,
            "all_complete": False,
            "batch_id": batch_id,
            "story_keys": picked_keys,
            "batch_size": len(picked_keys),
            "feature_id": picked_feature_id,
            "feature_key": picked_feature_label,
        },
    }


# ---------------------------------------------------------------------------
# story_loop_node
# ---------------------------------------------------------------------------

def _next_story_in_batch(
    conn,
    workspace: str,
    batch_id: int,
    already_patched: Optional[set[str]] = None,
) -> dict[str, Any] | None:
    """Return the next batch story that isn't ``done``/``blocked`` AND
    whose intra-batch dependencies are all already ``done`` AND whose
    patching turn hasn't already run in the current batch.

    Ordering rules (highest priority first):

    1. ``in_progress`` rows beat ``planned`` rows — a resumed session
       picks up exactly where it left off rather than starting a
       parallel story while another is mid-repair.
    2. Within each status group, ``bs.sequence`` decides — the order
       the batch sizer or operator chose. Phase I requires this
       order to be topological for intra-batch deps, so honoring it
       implements the dependency contract.
    3. Phase I (defense in depth): even when the sequence is right,
       skip a candidate whose intra-batch deps are still ``planned``
       or ``in_progress``. The batch sizer's validator catches
       intra-batch ordering bugs, but a corrupted DB row or a
       resumed session that landed mid-rewind could still surface
       a story before its in-batch dep finished — in which case we
       wait. Cross-batch deps (deps that aren't in batch_stories
       for the current batch_id) are NOT enforced here; those are
       guaranteed ``done`` by ``batch_planner_node`` before the
       batch was even created.
    4. Phase E.3 cursor-advance: skip any story whose key is in
       ``already_patched``. Without this, an ``in_progress`` story
       whose patching turn just ran would sort first under rule (1)
       and be re-picked forever — patching never marks a story
       ``done`` in the per-batch model (batch_commit_node does that
       at end-of-batch), so the "in-progress means mid-repair on
       resume" assumption behind rule (1) breaks down within a single
       run through the batch.
    """
    already_patched = already_patched or set()
    rows = conn.execute(
        "SELECT s.story_key FROM batch_stories bs "
        "JOIN stories s ON s.id = bs.story_id "
        "WHERE bs.batch_id = ? AND s.workspace = ? "
        "ORDER BY (s.status != 'in_progress'), bs.sequence",
        (batch_id, workspace),
    ).fetchall()
    batch_keys = {key for (key,) in rows}
    for (key,) in rows:
        if key in already_patched:
            continue
        s = story_state.get_story(conn, workspace, key)
        if s is None:
            continue
        if s["status"] not in ("planned", "in_progress", "reopened"):
            continue
        # Defensive intra-batch dep check.
        deps = list(s.get("depends_on") or [])
        intra_batch_unmet = []
        for d in deps:
            if d not in batch_keys:
                continue  # cross-batch dep — already done by batch_planner
            dep_story = story_state.get_story(conn, workspace, d)
            if dep_story is None or dep_story.get("status") != "done":
                intra_batch_unmet.append(d)
        if intra_batch_unmet:
            logger.debug(
                "[story_loop] %s deferred — intra-batch deps unmet: %s",
                key, intra_batch_unmet,
            )
            continue
        return s
    return None


def story_loop_node(state: dict[str, Any]) -> dict[str, Any]:
    """Advance to the next planned/in-progress story in the current batch.

    Sets ``current_story_id`` and ``story_scope_files`` on the state
    so the patching chain knows what it's working on. When the batch
    is exhausted (every story ``done`` or ``blocked``), returns
    ``batch_complete=True`` and clears ``current_story_id`` so the
    router can hop to ``traceability_node``.

    Per-story zero-patch auto-advance (Layer 2 — added after a session
    burned ~1h22m and $18 looping story_loop ↔ patching with the same
    STORY-001 selected every cycle): before picking the next story, this
    node consults ``loop_counter['story_zero_patch_rounds']`` for the
    currently-cursored story. If that story has accumulated
    ``story_zero_patch_cap`` (default 3) consecutive patching turns with
    success_count=0, it is marked ``done`` and its counter cleared so
    ``_next_story_in_batch`` advances to the next eligible story. The
    rationale is that a story the patcher can produce nothing for is
    most likely vacuous, mis-identified, or already covered by an
    earlier story — and the batch should carry on rather than stall the
    run. The decision is logged with the story key and round count so
    the operator sees exactly what was skipped and why.
    """
    workspace_path = state.get("workspace_path", "")
    workspace = story_state.app_name_for_workspace(workspace_path)
    batch_id = int(state.get("current_batch_id") or 0)
    if batch_id <= 0:
        logger.warning(
            "[story_loop] no current_batch_id (%r); nothing to advance.",
            state.get("current_batch_id"),
        )
        return {
            "current_story_id": "",
            "story_scope_files": [],
            "node_state": {
                "current_node": "story_loop",
                "batch_complete": True,
                "reason": "no_batch_id",
            },
        }

    # Per-story zero-patch auto-advance. Must run BEFORE
    # `_next_story_in_batch` so the now-done story isn't re-picked.
    loop_counter = dict(state.get("loop_counter", {}) or {})
    cur_story_id = state.get("current_story_id") or ""
    # Phase E.3 cursor-advance: the story whose patching turn just ran
    # must be remembered so `_next_story_in_batch` doesn't pick it again
    # next iteration. We carry this in state as ``batch_patched_story_keys``
    # rather than a DB column because (a) it's per-run-through-the-batch
    # bookkeeping (resume should re-patch in_progress stories), and
    # (b) it composes with the existing batch_commit_node reset.
    patched_keys = list(state.get("batch_patched_story_keys") or [])

    # A1 fix (2026-07-11): read the just-finished patching turn's
    # ``patch_success`` count so the "should we advance off cur_story_id
    # or give it a retry?" decision below can be made. The pre-A1 code
    # unconditionally appended cur_story_id to ``patched_keys`` here,
    # which meant a story that produced 0 real patches was burned from
    # the batch queue and _next_story_in_batch never re-picked it — so
    # the story_zero_patch_cap retry budget was dead code and the
    # patcher's rejection feedback (echoed in the message trail) went
    # unused. Finsearch STORY-002/003/004 all advanced with 0 patches
    # and never got a second shot; STORY-006/008/012 the same. See
    # patching_node's return payload (graph.py:4622) for the
    # node_state.patch_success contract.
    _ns_in = state.get("node_state", {}) or {}
    _patch_success = int(_ns_in.get("patch_success", 0) or 0)
    _had_patching_turn = "patch_success" in _ns_in

    auto_completed_key: Optional[str] = None
    auto_completed_rounds = 0
    if cur_story_id:
        sz_raw = loop_counter.get("story_zero_patch_rounds", {}) or {}
        sz: dict[str, int] = {
            k: int(v) for k, v in sz_raw.items()
        } if isinstance(sz_raw, dict) else {}
        cap = int(
            state.get("story_zero_patch_cap")
            or STORY_ZERO_PATCH_CAP
        )
        cur_rounds = int(sz.get(cur_story_id, 0) or 0)
        if cur_rounds >= cap:
            auto_completed_key = cur_story_id
            auto_completed_rounds = cur_rounds
            sz.pop(cur_story_id, None)
            loop_counter["story_zero_patch_rounds"] = sz

    # A1 append gate: only mark cur_story_id as "patched this pass" when
    # we're actually advancing off of it — patching produced real code,
    # the retry cap just fired, or story_loop was entered without a
    # patching turn (defensive: matches the pre-A1 behaviour for
    # resume/re-entry paths where node_state.patch_success is absent).
    _should_advance = (
        _patch_success > 0
        or not _had_patching_turn
        or auto_completed_key is not None
    )
    if cur_story_id and cur_story_id not in patched_keys and _should_advance:
        patched_keys.append(cur_story_id)
    elif cur_story_id and not _should_advance:
        logger.info(
            "[story_loop] %s produced 0 patches this round "
            "(zero_patch_rounds=%d/%d); re-picking so the LLM gets "
            "another shot with the patcher's rejection feedback.",
            cur_story_id,
            int(loop_counter.get("story_zero_patch_rounds", {}).get(cur_story_id, 0) or 0),
            int(state.get("story_zero_patch_cap") or STORY_ZERO_PATCH_CAP),
        )

    conn = story_state.open_story_db()
    try:
        if auto_completed_key is not None:
            # Mark `done` (not `blocked`) — the user's directive is that
            # a vacuous story should be considered satisfied so the
            # batch makes progress. If the story really had work the
            # downstream verification chain (compile / review) will
            # catch the gap and route the run through repair_node /
            # human_intervention_node, where the operator has full
            # context to intervene.
            story_state.mark_done(conn, workspace, auto_completed_key)
            logger.warning(
                "[story_loop] %s auto-completed after %d zero-patch round(s) "
                "(cap=%d). Story may be vacuous, already covered by an "
                "earlier story, or mis-identified by the decomposer. "
                "Marking done and advancing.",
                auto_completed_key, auto_completed_rounds,
                int(state.get("story_zero_patch_cap") or STORY_ZERO_PATCH_CAP),
            )
        nxt = _next_story_in_batch(
            conn, workspace, batch_id, already_patched=set(patched_keys),
        )
        if nxt is None:
            # Batch fully resolved (every story done or blocked).
            blocked_count = conn.execute(
                "SELECT COUNT(*) FROM batch_stories bs "
                "JOIN stories s ON s.id = bs.story_id "
                "WHERE bs.batch_id = ? AND s.workspace = ? "
                "AND s.status = 'blocked'",
                (batch_id, workspace),
            ).fetchone()[0]
            # NOTE: we DO NOT call story_state.complete_batch() here. The
            # per-batch verification chain (speculative_node →
            # code_review_node → batch_commit_node) hasn't run yet; sealing
            # the batch row now would let external readers (dashboard,
            # traceability) see a "complete" batch that hasn't been
            # verified. batch_commit_node is the single source of truth
            # for the final batch state and writes it atomically via
            # seal_batch_atomically.
            logger.info(
                "[story_loop] batch %d ready for verification "
                "(%d blocked, %d patched this pass).",
                batch_id, blocked_count, len(patched_keys),
            )
            return {
                "current_story_id": "",
                "story_scope_files": [],
                "loop_counter": loop_counter,
                "batch_patched_story_keys": patched_keys,
                "node_state": {
                    "current_node": "story_loop",
                    "batch_complete": True,
                    "batch_id": batch_id,
                    "blocked_count": blocked_count,
                    "auto_completed_story": auto_completed_key,
                    "auto_completed_zero_rounds": auto_completed_rounds,
                    "patched_in_batch": list(patched_keys),
                },
            }

        moved = story_state.mark_in_progress(conn, workspace, nxt["story_key"])
        if moved == 0:
            # Either the row vanished (race with another process) or
            # the story landed in a terminal state we don't recognise.
            # Route back to story_loop_node (NOT patching_node): the
            # earlier fail-open shape routed to patching with
            # current_story_id="" which spun the patcher with no scope
            # and could loop indefinitely. Treating this as
            # batch_complete=True hands control to the verification
            # chain — if it really is the last unprocessable story, the
            # chain seals the batch as complete_with_blocks; otherwise
            # the next pass through story_loop_node will retry.
            logger.warning(
                "[story_loop] mark_in_progress matched 0 rows for %s; "
                "ending batch loop and handing to verification chain.",
                nxt["story_key"],
            )
            return {
                "current_story_id": "",
                "story_scope_files": [],
                "loop_counter": loop_counter,
                "batch_patched_story_keys": patched_keys,
                "node_state": {
                    "current_node": "story_loop",
                    "batch_complete": True,
                    "batch_id": batch_id,
                    "skipped": True,
                    "reason": "mark_in_progress_no_rows",
                    "story_key": nxt["story_key"],
                    "auto_completed_story": auto_completed_key,
                    "auto_completed_zero_rounds": auto_completed_rounds,
                    "patched_in_batch": list(patched_keys),
                },
            }
    finally:
        conn.close()

    logger.info(
        "[story_loop] next story: %s — %s (scope_files=%s)",
        nxt["story_key"], nxt["title"], nxt["scope_files"] or "(unscoped)",
    )
    # Snapshot the modified_files cursor so story_complete_node can
    # attribute only the files newly touched during this story to its
    # file_links rows.
    baseline = list(state.get("modified_files", []) or [])
    return {
        "current_story_id": nxt["story_key"],
        "story_scope_files": list(nxt["scope_files"] or []),
        "story_modified_baseline": baseline,
        "loop_counter": loop_counter,
        "batch_patched_story_keys": patched_keys,
        "node_state": {
            "current_node": "story_loop",
            "batch_complete": False,
            "story_key": nxt["story_key"],
            "story_title": nxt["title"],
            "acceptance_criteria": nxt["acceptance_criteria"],
            "auto_completed_story": auto_completed_key,
            "auto_completed_zero_rounds": auto_completed_rounds,
            "patched_in_batch": list(patched_keys),
        },
    }


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def route_after_batch_planner(state: dict[str, Any]) -> str:
    """After ``batch_planner_node``:

    - batch was created → ``story_loop_node`` (start the loop)
    - empty story DB (decomposition missing/failed) → ``human_intervention_node``
    - all stories already done → ``traceability_node`` (write final view)
    - stall (deps unmet) → ``traceability_node`` (flush state, exit)
    """
    ns = state.get("node_state", {})
    if ns.get("batch_planned"):
        return "story_loop_node"
    if ns.get("decomposition_missing"):
        return "human_intervention_node"
    return "traceability_node"


def route_after_story_loop(state: dict[str, Any]) -> str:
    """After ``story_loop_node``:

    - a story was picked → ``patching_node`` (the acceptance criteria
      are carried into the patching LLM via the story preamble that
      ``_build_story_preamble`` injects, so no separate test-stub
      generation step is needed — Phase F removed
      ``story_test_first_node``).
    - batch exhausted (every story patched) → enter the per-batch
      verification chain. Phase K consults
      ``batch_gate_progress[current_batch_id]`` and skips ahead to the
      next un-passed gate so a resumed session doesn't re-run gates
      that already cleared cleanly before the crash:

        * compile_passed=False → ``speculative_node`` (full chain)
        * compile_passed=True, review_passed=False → ``code_review_node``
        * compile_passed=True, review_passed=True → ``batch_commit_node``
          (everything passed — only reachable when the crash landed
          inside batch_commit itself; the seal idempotent re-runs).
    """
    ns = state.get("node_state", {})
    if not ns.get("batch_complete"):
        return "patching_node"

    batch_id = int(state.get("current_batch_id") or 0)
    bgp = state.get("batch_gate_progress") or {}
    entry = bgp.get(str(batch_id)) or {} if isinstance(bgp, dict) else {}
    compile_passed = bool(entry.get("compile_passed"))
    review_passed = bool(entry.get("review_passed"))

    if not compile_passed:
        return "speculative_node"
    if not review_passed:
        return "code_review_node"
    return "batch_commit_node"


# ---------------------------------------------------------------------------
# story_complete_node — mark outcome, link files, optional git commit
# ---------------------------------------------------------------------------

def _classify_file(rel_path: str) -> str:
    """Heuristic kind for file_links: code / test / doc / infra."""
    p = rel_path.replace("\\", "/")
    name = os.path.basename(p)
    if (
        p.startswith("tests/")
        or "/tests/" in p
        or "/__tests__/" in p
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name in ("conftest.py", "pytest.ini")
    ):
        return "test"
    if p.startswith("docs/") or name.endswith(".md"):
        return "doc"
    if name in (
        "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
        "Caddyfile", "Makefile",
    ) or name.endswith((".dockerfile", ".compose.yml", ".compose.yaml")):
        return "infra"
    return "code"


def _git(workspace: str, *args: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        ["git", "-C", workspace, *args],
        capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=60,
    )


def _is_git_repo(workspace: str) -> bool:
    if not os.path.isdir(os.path.join(workspace, ".git")):
        return False
    return _git(workspace, "rev-parse", "--git-dir").returncode == 0


def _stage_and_commit(
    workspace: str, message: str, log_tag: str
) -> Optional[str]:
    """Common path used by both per-story and per-batch commit helpers.

    Stages everything, commits with ``message``, and returns the new
    HEAD SHA. ``log_tag`` prefixes any warning logs (e.g. story_key or
    'batch_commit'). Returns None on no-op (nothing to commit) or any
    failure (hook rejection, etc.) — commit failures are never fatal."""
    if not _is_git_repo(workspace):
        return None

    add = _git(workspace, "add", "-A")
    if add.returncode != 0:
        logger.warning(
            "[%s] git add failed: %s", log_tag, add.stderr.strip(),
        )
        return None

    status = _git(workspace, "status", "--porcelain")
    if status.returncode == 0 and not status.stdout.strip():
        return None

    commit = _git(workspace, "commit", "-m", message)
    if commit.returncode != 0:
        logger.warning(
            "[%s] git commit failed: %s", log_tag, commit.stderr.strip(),
        )
        return None
    sha = _git(workspace, "rev-parse", "HEAD")
    if sha.returncode != 0:
        return None
    return sha.stdout.strip() or None


def _commit_for_story(
    workspace: str, story_key: str, title: str
) -> Optional[str]:
    """Stage + commit the working tree under a ``STORY-N: title`` message.

    Returns the new HEAD SHA, or None on any failure (not a git repo,
    nothing to commit, hook rejected, etc.). Failures are non-fatal —
    the story is still marked done; only the commits row is skipped.

    Retained during the transition to per-batch commits — Phase F's
    removal of ``story_complete_node`` will retire this helper.
    """
    message = f"{story_key}: {title}" if title else f"{story_key}: complete"
    return _stage_and_commit(workspace, message, log_tag=f"story_complete:{story_key}")


def _commit_for_batch(
    workspace: str,
    batch_id: int,
    stories: list[tuple[str, str]],
) -> Optional[str]:
    """Stage + commit the working tree under a ``BATCH-N: ...`` message.

    ``stories`` is ``[(story_key, title), ...]`` in batch order. The
    commit message lists every constituent story so ``git log --oneline``
    shows what landed in the batch. Returns the new HEAD SHA, or None
    on no-op / failure (commit failures are non-fatal — the batch is
    still marked complete; only the SHA stamping is skipped).
    """
    if stories:
        body = "; ".join(
            f"{key}: {title}".strip() if title else key
            for key, title in stories
        )
        message = f"BATCH-{batch_id}: {body}"
    else:
        message = f"BATCH-{batch_id}: complete"
    return _stage_and_commit(
        workspace, message, log_tag=f"batch_commit:{batch_id}",
    )


def story_complete_node(state: dict[str, Any]) -> dict[str, Any]:
    """Record the outcome of the current story and advance the cursor.

    Outcomes:

    - ``exit_code == 0`` → mark_done, resolve any open defects for the
      story, optional ``git commit`` when ``commit_on_story``.
    - ``total_repairs >= story_repair_cap`` → mark_blocked, record a
      defect describing the cap hit. Batch continues to next story.
    - Otherwise (rare — node fires only after the loop settles) →
      leave story ``in_progress`` so the next iteration can retry.

    Resets the loop counters and ``current_story_id`` so the next
    ``story_loop_node`` hop starts clean.
    """
    story_key = state.get("current_story_id") or ""
    if not story_key:
        return {
            "node_state": {
                "current_node": "story_complete",
                "skipped": True,
                "reason": "no_current_story",
            },
        }
    workspace_path = state.get("workspace_path") or ""
    workspace = story_state.app_name_for_workspace(workspace_path)
    session_id = state.get("session_id", "")
    exit_code = state.get("exit_code", -1)
    loop_counter = dict(state.get("loop_counter", {}) or {})
    repair_total = int(loop_counter.get("total_repairs", 0) or 0)
    repair_cap = int(state.get("story_repair_cap") or 3)
    current_batch_id = int(state.get("current_batch_id") or 0) or None

    baseline = list(state.get("story_modified_baseline", []) or [])
    current = list(state.get("modified_files", []) or [])
    baseline_set = set(baseline)
    new_files = [f for f in current if f not in baseline_set]

    success = exit_code == 0
    outcome: str
    committed_sha: Optional[str] = None
    defect_id: Optional[int] = None

    conn = story_state.open_story_db()
    try:
        for path in new_files:
            # Stamp with the current batch so per-batch repair
            # attribution can map compile errors back to the owning
            # story; without batch_id the link_file row defaults to
            # NULL and the join in files_for_batch returns nothing.
            story_state.link_file(
                conn, workspace, story_key, path, _classify_file(path),
                batch_id=current_batch_id,
            )

        if success:
            story_state.mark_done(conn, workspace, story_key)
            story_state.resolve_defects_for_story(conn, workspace, story_key)
            outcome = "done"
            story = story_state.get_story(conn, workspace, story_key)
            if state.get("commit_on_story") and story is not None:
                committed_sha = _commit_for_story(
                    workspace_path, story_key, story.get("title") or "",
                )
                if committed_sha:
                    story_state.record_commit(
                        conn,
                        workspace=workspace,
                        sha=committed_sha, story_key=story_key,
                        session_id=session_id,
                        message=f"{story_key}: {story.get('title', '')}",
                    )
        elif repair_total >= repair_cap:
            story_state.mark_blocked(conn, workspace, story_key)
            defect_id = story_state.record_defect(
                conn,
                workspace=workspace,
                story_key=story_key,
                session_id=session_id,
                severity="repair_cap_exceeded",
                summary=(
                    f"{story_key} hit repair cap "
                    f"({repair_total}/{repair_cap}); exit_code={exit_code}"
                ),
                diagnostic={
                    "compiler_errors": state.get("compiler_errors") or [],
                    "exit_code": exit_code,
                    "repair_total": repair_total,
                },
            )
            outcome = "blocked"
        else:
            outcome = "incomplete"
    finally:
        conn.close()

    # Reset per-story counters so the next story starts with a clean repair budget.
    for key in ("patching", "repair", "compiler", "total_repairs",
                "review_spec", "review_code"):
        loop_counter[key] = 0

    logger.info(
        "[story_complete] %s → %s (new_files=%d, sha=%s, defect=%s)",
        story_key, outcome, len(new_files), committed_sha or "—", defect_id or "—",
    )
    return {
        "current_story_id": "",
        "story_scope_files": [],
        "story_modified_baseline": [],
        "loop_counter": loop_counter,
        "node_state": {
            "current_node": "story_complete",
            "story_key": story_key,
            "outcome": outcome,
            "new_files": new_files,
            "committed_sha": committed_sha,
            "defect_id": defect_id,
        },
    }


def route_after_story_complete(state: dict[str, Any]) -> str:
    """After ``story_complete_node``: always loop back to ``story_loop_node``
    to pick the next story in the current batch. ``story_loop_node`` is
    responsible for noticing when the batch is exhausted and routing
    back to ``batch_planner_node``."""
    return "story_loop_node"


# ---------------------------------------------------------------------------
# batch_commit_node — per-batch sealing (replaces per-story commit)
# ---------------------------------------------------------------------------

def batch_commit_node(state: dict[str, Any]) -> dict[str, Any]:
    """Seal a batch after its verification pipeline has passed.

    In the per-batch verification pipeline (Phase E), the compile /
    review / test / security / regression loops have already finished
    against the batch's combined patches. This node:

    - Marks every still-pending story in ``current_batch_id`` as
      ``done`` and resolves their open defects.
    - Counts ``blocked`` stories (carried over from per-story repair-cap
      failures) and sets the batch's terminal status accordingly
      (``complete`` or ``complete_with_blocks``).
    - When ``commit_on_story`` (kept as the operator-facing flag through
      the transition) is True, calls ``_commit_for_batch`` and records
      a ``commits`` row per constituent story so traceability still
      attributes the commit to each story.
    - Persists the commit SHA via ``story_state.set_batch_committed_sha``.
    - Resets the batch-scoped state cursor:
      ``current_batch_id=0``, ``current_story_id=""``,
      ``story_scope_files=[]``, ``story_modified_baseline=[]``,
      ``batch_modified_files=[]``.
    - Resets the per-batch ``loop_counter`` keys so the next batch
      starts with a fresh repair budget.

    The router (``route_after_batch_commit``) sends control back to
    ``batch_planner_node`` to try the next batch; when no batches
    remain, the planner falls through to ``traceability_node`` and the
    end-of-session security + regression gates.
    """
    workspace_path = state.get("workspace_path") or ""
    batch_id = int(state.get("current_batch_id") or 0)
    if batch_id <= 0 or not workspace_path:
        return {
            "node_state": {
                "current_node": "batch_commit",
                "skipped": True,
                "reason": "no_batch_or_workspace",
            },
        }
    workspace = story_state.app_name_for_workspace(workspace_path)

    session_id = state.get("session_id", "")
    commit_enabled = bool(state.get("commit_on_story"))
    batch_files = list(state.get("batch_modified_files") or [])

    committed_sha: Optional[str] = None
    blocked_count = 0
    stories_in_batch: list[tuple[str, str, str]] = []
    done_keys: list[str] = []

    conn = story_state.open_story_db()
    try:
        rows = conn.execute(
            "SELECT s.story_key, s.title, s.status "
            "FROM batch_stories bs JOIN stories s ON s.id = bs.story_id "
            "WHERE bs.batch_id = ? AND s.workspace = ? ORDER BY bs.sequence",
            (batch_id, workspace),
        ).fetchall()
        stories_in_batch = [
            (r[0], r[1] or "", r[2] or "") for r in rows
        ]

        blocked_count = sum(
            1 for _, _, st in stories_in_batch if st == "blocked"
        )

        # Git commit FIRST — it can't be rolled back, so it has to
        # happen outside the DB transaction. Failure here yields
        # committed_sha=None and the seal proceeds without a SHA stamp.
        if commit_enabled and stories_in_batch:
            commit_stories = [(k, t) for k, t, _ in stories_in_batch]
            committed_sha = _commit_for_batch(
                workspace_path, batch_id, commit_stories,
            )

        batch_message: Optional[str] = None
        if committed_sha:
            # One commits row per SHA (commits.sha is the PRIMARY KEY).
            # The row carries no specific story_id — this is a
            # batch-level commit — and the message lists every
            # constituent story so log inspection still attributes the
            # change. The authoritative per-story landing point is
            # ``batches.committed_sha`` joined via ``batch_stories``.
            batch_message = (
                f"BATCH-{batch_id}: " + "; ".join(
                    f"{k}: {t}".strip() if t else k
                    for k, t, _ in stories_in_batch
                )
            )

        # All DB mutations in a single transaction so a crash here can
        # never leave the batch row ``running`` with stories already
        # marked ``done`` — on resume, batch_planner_node would have
        # seen inconsistent state.
        # Per-file (path, kind) for traceability attribution. batch_files
        # is a plain list of rel paths from state; classify each so
        # TRACEABILITY.md can split "code" vs "test" vs "doc" columns.
        classified_files = [(p, _classify_file(p)) for p in batch_files]
        done_keys, blocked_count = story_state.seal_batch_atomically(
            conn,
            workspace=workspace,
            batch_id=batch_id,
            stories_in_batch=stories_in_batch,
            blocked_count=blocked_count,
            committed_sha=committed_sha,
            batch_commit_message=batch_message,
            session_id=session_id,
            batch_files=classified_files,
        )
    finally:
        conn.close()

    loop_counter = dict(state.get("loop_counter", {}) or {})
    # Per-batch repair budgets reset between batches. Keys that don't
    # exist yet are tolerated — assignment, not increment.
    # PER_BATCH_CAP_COUNTERS is the canonical registry (see
    # harness/loop_counter_keys.py) shared with the two resume paths
    # in cli.py. Site-specific extras stay listed inline below.
    for key in PER_BATCH_CAP_COUNTERS:
        loop_counter[key] = 0
    for key in (
        "total_repairs",
        "consecutive_zero_patch_rounds", "missing_dep_consecutive_same",
        "diagnostics_rounds_since_compile",
    ):
        loop_counter[key] = 0
    # Layer 2 / Layer 3 — per-story zero-patch tally and the global
    # no-progress failsafe are both scoped to the current batch.
    # ``story_zero_patch_rounds`` carries story_keys that won't exist
    # in the next batch; ``progress_tracker`` carries a budget marker
    # that should re-baseline against the next batch's starting budget.
    loop_counter["story_zero_patch_rounds"] = {}
    loop_counter.pop("progress_tracker", None)

    # Phase K — pop the sealed batch's gate-progress entry so a future
    # batch with the same id (shouldn't happen, IDs are monotonic) or
    # an inspection tool sees a clean slate. The dict is rebuilt
    # immutably so we don't mutate state.
    existing_bgp = state.get("batch_gate_progress") or {}
    if isinstance(existing_bgp, dict):
        next_bgp = {
            k: dict(v) for k, v in existing_bgp.items()
            if k != str(batch_id)
        }
    else:
        next_bgp = {}

    blocked_keys = [k for k, _, st in stories_in_batch if st == "blocked"]
    if blocked_count:
        logger.error(
            "[batch_commit] batch %d sealed with %d blocked story(s): %s. "
            "Operator action required — these stories never passed their "
            "acceptance gates and were carried as defects.",
            batch_id, blocked_count, ", ".join(blocked_keys),
        )
    logger.info(
        "[batch_commit] batch %d sealed "
        "(stories=%d, marked_done=%d, blocked=%d, sha=%s, batch_files=%d)",
        batch_id, len(stories_in_batch), len(done_keys), blocked_count,
        committed_sha or "—", len(batch_files),
    )

    # NOTE (unit-test model): the batch seal used to run a
    # ``test_verifies_ac`` sweep here — parsing ``@verifies:`` markers
    # out of test files and even retroactively PREPENDING markers to
    # tests whose bodies mentioned a story informally. Both are gone
    # on purpose: unit tests generated during build / patch are linked
    # to the CODE under test (the ``@tests:`` marker), never to stories
    # or acceptance criteria. AC edges are written exclusively by the
    # ``teane test`` functional pack (harness/playwright_gen.py), and
    # the traceability AC gate only fires in that flow
    # (traceability.has_ac_gap). Session 22471c0c's seal inserted 18
    # bogus test→AC edges from unit tests this way.
    return {
        "current_batch_id": 0,
        "current_story_id": "",
        "story_scope_files": [],
        "story_modified_baseline": [],
        "batch_modified_files": [],
        "batch_gate_progress": next_bgp,
        # Phase E.3 cursor-advance fix: clear the patched-keys cursor so
        # the next batch (if any) starts clean. batch_planner_node also
        # resets this when it plans the next batch — keeping the reset
        # here too means inspection tools (dashboard, traceability) see
        # a clean state even if no next batch ever runs.
        "batch_patched_story_keys": [],
        "loop_counter": loop_counter,
        "node_state": {
            "current_node": "batch_commit",
            "batch_id": batch_id,
            "stories_in_batch": len(stories_in_batch),
            "marked_done": len(done_keys),
            "blocked_count": blocked_count,
            "blocked_story_keys": blocked_keys,
            "batch_has_blocks": blocked_count > 0,
            "committed_sha": committed_sha,
            "batch_files": batch_files,
        },
    }


def route_after_batch_commit(state: dict[str, Any]) -> str:
    """After ``batch_commit_node``: try to plan the next batch.

    ``batch_planner_node`` decides whether more batches are available
    (more planned stories whose deps are now satisfied) or whether
    every story is accounted for, in which case it falls through to
    ``traceability_node`` and the end-of-session gates."""
    return "batch_planner_node"


# ---------------------------------------------------------------------------
# traceability_node — regenerate the matrix view
# ---------------------------------------------------------------------------

def traceability_node(state: dict[str, Any]) -> dict[str, Any]:
    """Regenerate ``docs/STORIES.md`` and ``docs/TRACEABILITY.md`` from
    the DB. Idempotent; safe to call at any time. Fires at the end of
    every batch and at the very end of the run.

    When ``state["arch_summary"]`` is populated (by ``decomposition_node``
    on agile flows, or lazy-loaded by ``patching_node`` on monolithic
    ones), TRACEABILITY.md picks up an "Architecture coverage"
    section that surfaces gaps between the §11 endpoint / component
    map and the stories implementing them. Empty / missing summary
    keeps the byte-identical pre-existing output.
    """
    workspace_path = state.get("workspace_path") or ""
    if not workspace_path:
        return {
            "node_state": {
                "current_node": "traceability",
                "skipped": True,
                "reason": "no_workspace",
            },
        }

    arch_summary_dict = state.get("arch_summary") or {}
    if not arch_summary_dict:
        # Lazy load — covers the case where this node fires before
        # patching_node ever did (e.g. an immediate traceability
        # regeneration after decomposition on a budget-exhausted
        # session).
        from harness.arch_summary import load_arch_summary
        arch_summary_dict = load_arch_summary(workspace_path) or {}

    # NOTE (unit-test model): the NFR AC skip-stub backfill that ran
    # here is gone. Build / patch unit tests never carry AC linkage,
    # and the AC-coverage gate only fires during ``teane test``
    # (traceability.has_ac_gap), so there is no audit block to appease
    # with placeholder stubs. NFR verification is owned by the
    # ``teane test`` functional pack.

    conn = story_state.open_story_db()
    try:
        stories_md, trace_md = story_state.regenerate_markdown_views(
            conn, workspace_path,
            arch_summary=arch_summary_dict or None,
        )
    finally:
        conn.close()
    logger.info(
        "[traceability] regenerated %s and %s%s",
        os.path.relpath(stories_md, workspace_path),
        os.path.relpath(trace_md, workspace_path),
        " (with arch coverage)" if arch_summary_dict else "",
    )

    # v5 soft batch warnings — run the SQL audit at end-of-batch and
    # log gaps so the operator notices early, but DO NOT block.
    # End-of-session enforcement lives in installation_doc_node where
    # ``traceability.enforce`` (default true) can route to HITL.
    # Batches that hit a gap continue running so the audit doesn't
    # stall mid-feature; the operator sees the warning in the log and
    # the freshly-regenerated TRACEABILITY.md.
    audit_summary: dict[str, Any] = {}
    try:
        from harness.traceability import audit_workspace
        report = audit_workspace(workspace_path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[traceability] batch audit skipped: %s", exc)
        report = None
    if report is not None:
        audit_summary = {
            "total_reqs": report.total_reqs,
            "traced_reqs": report.traced_reqs,
            "untraced_count": len(report.untraced),
            "total_acs": report.total_acs,
            "verified_acs": report.verified_acs,
            "untested_count": len(report.untested_acs),
        }
        if report.has_failures():
            # AC coverage is closed by ``teane test`` (Playwright pack
            # generated against ACs), so the batch-level warning says
            # so — earlier revisions warned that end-of-session would
            # block on untested_acs, which is no longer accurate for
            # build/patch flows.
            logger.warning(
                "[traceability] batch gaps — reqs %d/%d (%.0f%%), "
                "ACs %d/%d (%.0f%%); untraced=%d, untested=%d. "
                "Soft warning at end-of-batch. End-of-session blocks "
                "only on untraced requirements; AC coverage is closed "
                "by `teane test` and does not block build/patch.",
                report.traced_reqs, report.total_reqs,
                report.req_coverage_pct,
                report.verified_acs, report.total_acs,
                report.ac_coverage_pct,
                len(report.untraced), len(report.untested_acs),
            )

    return {
        "arch_summary": arch_summary_dict,
        "node_state": {
            "current_node": "traceability",
            "skipped": False,
            "stories_md": stories_md,
            "traceability_md": trace_md,
            "arch_coverage_emitted": bool(arch_summary_dict),
            "audit_summary": audit_summary,
        },
    }
