"""Park a failing-test blocker instead of ending the run (ADR-0010, Step 1).

A headless build that cannot fix a test used to have exactly one way out of
the repair loop: exhaust an auto-resume cap and terminate. lumina run 12 ended
that way with 13 of 17 criteria verified and zero stories sealed; run 01a21db1
ended that way on three acceptance tests that were correct. In both, one
bounded set of failing tests cost every story in the batch and every batch
after it.

ADR-0006 already contains an acceptance failure that survives the acceptance
repair cap: ``acceptance_node._park_over_cap`` records a deferral, quarantines
the test with a tagged non-strict ``xfail`` and seals the batch. But that
runs only when control RETURNS to ``acceptance_node``, and the inner repair
loop that dies on a HITL cap never returns there. This module applies the same
containment at the level where runs actually die.

Deliberately not clever. Nothing here decides whether the test or the code is
at fault — that is the question the last three months of fixes kept getting
wrong in alternating directions. It only bounds what a wrong answer can cost:
the failing tests are carried as recorded, self-describing expected failures
and the build goes on.

And deliberately empirical. No attempt is made to prove in advance that a park
will turn the build green. The resumed compile says so; if the same blocker
comes back there is nothing new to park and the caller terminates as before,
one cycle later.

No dependency on ``harness.graph`` — ``cli`` imports this, and ``graph``
imports ``cli`` lazily.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_PARK_ON_CAP = True
DEFAULT_MAX_PARKED_BLOCKERS = 12
DEFAULT_MAX_TESTS_PER_PARK = 12

#: ``loop_counter`` keys. Both are plain JSON so they survive a checkpoint.
LC_PARKED_BLOCKERS = "parked_blockers"
LC_PARKED_NODEIDS = "parked_test_nodeids"

#: The ``defects.severity`` label for a test parked here.
DEFECT_SEVERITY = "parked_at_hitl_cap"

#: Trigger classes (the label before the first ``:``) whose blocker is NOT a
#: failing test, whatever stale diagnostics ``compiler_errors`` still holds.
#: Quarantining tests cannot unblock a sandbox that timed out, a security
#: finding or a traceability gap, and would leave markers on tests that had
#: nothing to do with it.
NON_TEST_TRIGGER_CLASSES = frozenset({
    "traceability_block",
    "decomposition_validation_failed",
    "decomposition_missing",
    "llm_behavior",
    "env_misconfig",
    "build_command_blocked",
    "build_command_cd_missing",
    "build_env_timeout",
    "security_fix_limit",
    "budget_exhausted",
    "budget_preflight",
    "discovery_interview_loop",
})

# pytest's short-summary rows. Same shape ``parser_registry`` reads, kept
# local so this module does not depend on a private pattern there.
_SUMMARY_ROW = re.compile(
    r"^(?P<kind>FAILED|ERROR)\s+(?P<file>[^\s:]+\.py)(?P<tail>::\S+)?"
    r"(?:\s+-\s+.*)?$"
)


def clamp_max_parked_blockers(raw: Any) -> int:
    """``hitl.max_parked_blockers`` → [0, 200]. 0 disables parking."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_PARKED_BLOCKERS
    return max(0, min(200, value))


def clamp_max_tests_per_park(raw: Any) -> int:
    """``hitl.max_parked_tests_per_blocker`` → [1, 200]."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_TESTS_PER_PARK
    return max(1, min(200, value))


@dataclass(frozen=True)
class FailingTest:
    """One failing pytest function, as named by its node id."""

    nodeid: str
    file: str       # workspace-relative, forward slashes
    function: str   # bare function name, parametrisation stripped


@dataclass
class ParkDecision:
    """Whether the current blocker can be parked, and what that would park."""

    tests: list[FailingTest] = field(default_factory=list)
    refusal: str = ""

    @property
    def parkable(self) -> bool:
        return bool(self.tests) and not self.refusal


@dataclass
class ParkOutcome:
    """What a park actually did."""

    parked: list[FailingTest] = field(default_factory=list)
    ac_keys: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    reason: str = ""


def _rel(path: str, workspace: str) -> str:
    """Workspace-relative, forward-slash form of ``path``."""
    p = str(path or "").strip().replace("\\", "/")
    ws = str(workspace or "").rstrip("/").replace("\\", "/")
    if ws and p.startswith(ws + "/"):
        p = p[len(ws) + 1:]
    while p.startswith("./"):
        p = p[2:]
    return p


def _split_nodeid(nodeid: str, workspace: str) -> Optional[FailingTest]:
    """``path.py::[Class::]name[param]`` → :class:`FailingTest`, else None."""
    nid = str(nodeid or "").strip()
    if "::" not in nid:
        return None
    file_part, _, _rest = nid.partition("::")
    function = nid.rsplit("::", 1)[-1].split("[", 1)[0].strip()
    file_rel = _rel(file_part, workspace)
    if not file_rel.endswith(".py") or not function:
        return None
    return FailingTest(nodeid=nid, file=file_rel, function=function)


def failing_tests(
    compiler_errors: Any, raw_output: str, workspace: str,
) -> tuple[list[FailingTest], list[str]]:
    """The failing pytest functions, plus anything that cannot be parked.

    Reads BOTH the structured diagnostics and the raw build output. The
    diagnostics alone under-report: ``PythonParser._dedup`` keys on
    ``(file, error_code)``, so three ``AssertionError`` failures in one file
    arrive as one diagnostic carrying one node id. Parking from that would
    quarantine one test, resume, and meet the second.

    Returns ``(tests, obstacles)``. An obstacle is a reason parking cannot
    make this build green — a module that fails to collect, or a failure that
    is not a Python test at all.
    """
    found: dict[tuple[str, str], FailingTest] = {}
    obstacles: list[str] = []

    def _add(nodeid: str) -> None:
        ft = _split_nodeid(nodeid, workspace)
        if ft is not None:
            found.setdefault((ft.file, ft.function), ft)

    loose: list[str] = []
    for d in compiler_errors or []:
        if not isinstance(d, dict):
            continue
        nid = str(d.get("pytest_nodeid") or "")
        if nid:
            _add(nid)
            continue
        f = _rel(str(d.get("file") or ""), workspace)
        if f:
            loose.append(f)

    for line in str(raw_output or "").splitlines():
        m = _SUMMARY_ROW.match(line.strip())
        if not m:
            continue
        if m.group("tail"):
            _add(m.group("file") + m.group("tail"))
        else:
            # "ERROR tests/test_x.py - ImportError: ..." — the module never
            # imported, so there is no function to mark and an xfail on one
            # would never be reached.
            obstacles.append(
                f"{_rel(m.group('file'), workspace)} fails to collect"
            )

    # A diagnostic with no node id in a non-Python file is a failure this
    # module has no marker for (a vitest suite, a tsc error). A loose ``.py``
    # diagnostic is tolerated: a failing test whose innermost frame is in
    # production code yields exactly that, and the resumed compile is what
    # tells a derivative traceback from an independent build error.
    for f in loose:
        if not f.endswith(".py"):
            obstacles.append(f"{f} is not a Python test failure")

    # Stable order so a park, its log line and its tests are reproducible.
    tests = sorted(found.values(), key=lambda t: (t.file, t.function))
    return tests, sorted(set(obstacles))


def decide(
    state: dict[str, Any], *, max_tests: int, max_blockers: int,
) -> ParkDecision:
    """Can the blocker in ``state`` be parked? Pure read."""
    loop_counter = state.get("loop_counter") or {}
    node_state = state.get("node_state") or {}
    workspace = str(state.get("workspace_path") or "")

    _history = loop_counter.get(LC_PARKED_BLOCKERS)
    parks_so_far = len(_history) if isinstance(_history, list) else 0
    if max_blockers <= 0:
        return ParkDecision(refusal="parking is disabled (max_parked_blockers=0)")
    if parks_so_far >= max_blockers:
        return ParkDecision(refusal=(
            f"{parks_so_far}/{max_blockers} blockers already parked this "
            "session"
        ))

    tests, obstacles = failing_tests(
        state.get("compiler_errors"),
        str(node_state.get("last_build_output") or ""),
        workspace,
    )
    if obstacles:
        return ParkDecision(refusal="; ".join(obstacles[:3]))
    if not tests:
        return ParkDecision(refusal="no failing pytest function identified")

    _already_raw = loop_counter.get(LC_PARKED_NODEIDS)
    already = {
        str(x) for x in (_already_raw if isinstance(_already_raw, list) else [])
    }
    fresh = [t for t in tests if f"{t.file}::{t.function}" not in already]
    if not fresh:
        # Every failing test already carries a marker and the build is still
        # red. The previous park did not work, and marking them again cannot.
        return ParkDecision(refusal=(
            "every failing test is already parked and the build is still red"
        ))
    if len(fresh) > max_tests:
        # A large failing set is a broken build, not a blocker. Sealing an
        # application behind a wall of quarantined tests is worse than
        # stopping.
        return ParkDecision(refusal=(
            f"{len(fresh)} failing tests exceed max_parked_tests_per_blocker "
            f"({max_tests})"
        ))
    return ParkDecision(tests=fresh)


def apply(
    workspace: str, tests: list[FailingTest], reason: str,
) -> ParkOutcome:
    """Write the quarantine markers. Returns what was actually marked.

    A test already carrying a marker, absent from its file, or in a file that
    will not parse is skipped rather than failing the whole park —
    ``quarantine_function`` returns None for each of those.
    """
    from harness import acceptance_gen as ag

    outcome = ParkOutcome(reason=reason)
    by_file: dict[str, list[FailingTest]] = {}
    for t in tests:
        by_file.setdefault(t.file, []).append(t)

    for rel, group in sorted(by_file.items()):
        abs_path = rel if os.path.isabs(rel) else os.path.join(workspace, rel)
        try:
            with open(abs_path, "r", encoding="utf-8") as fh:
                source = fh.read()
        except OSError as exc:
            logger.warning(
                "[park] cannot read %s to quarantine it: %s", abs_path, exc,
            )
            continue
        try:
            verifies = ag.verifies_by_function(source)
        except Exception:  # noqa: BLE001 — AC lookup must not block a park
            verifies = {}
        marked: list[FailingTest] = []
        for t in group:
            new = ag.quarantine_function(source, t.function, reason)
            if new:
                source = new
                marked.append(t)
        if not marked:
            continue
        try:
            with open(abs_path, "w", encoding="utf-8") as fh:
                fh.write(source)
        except OSError as exc:
            logger.warning(
                "[park] could not write the quarantine to %s: %s",
                abs_path, exc,
            )
            continue
        outcome.files.append(rel)
        outcome.parked.extend(marked)
        for t in marked:
            entry = verifies.get(t.function)
            if entry and entry[0] and entry[0] not in outcome.ac_keys:
                outcome.ac_keys.append(str(entry[0]))
    return outcome


def _record(
    state: dict[str, Any], outcome: ParkOutcome, *, trigger: str,
    cap_reason: str,
) -> None:
    """Deferrals for criteria, a defect row per test. Best-effort."""
    workspace = str(state.get("workspace_path") or "")
    session_id = str(state.get("session_id") or "")
    try:
        from harness import story_state
        from harness.acceptance_run import STATUS_ATTRIBUTABLE
        app = story_state.app_name_for_workspace(workspace)
        conn = story_state.open_story_db(workspace_path=workspace)
    except Exception as exc:  # noqa: BLE001 — the record must not undo a park
        logger.warning("[park] could not open the story DB to record: %s", exc)
        return
    try:
        detail = f"parked at the HITL cap ({cap_reason}): {outcome.reason}"
        for key in outcome.ac_keys:
            try:
                ac = story_state.get_ac_by_key(conn, app, key)
                if ac is not None:
                    story_state.record_acceptance_deferral(
                        conn, app, int(ac["id"]), STATUS_ATTRIBUTABLE,
                        detail[:500],
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug("[park] deferral for %s skipped: %s", key, exc)
        try:
            from harness import acceptance_gen as ag
        except Exception:  # noqa: BLE001
            ag = None  # type: ignore[assignment]
        for t in outcome.parked:
            story_key: Optional[str] = None
            if ag is not None:
                try:
                    abs_path = os.path.join(workspace, t.file)
                    with open(abs_path, "r", encoding="utf-8") as fh:
                        entry = ag.verifies_by_function(fh.read()).get(
                            t.function,
                        )
                    if entry and entry[0]:
                        story_key = str(entry[0]).split(".", 1)[0]
                except Exception:  # noqa: BLE001
                    story_key = None
            try:
                story_state.record_defect(
                    conn, workspace=app, story_key=story_key,
                    session_id=session_id, severity=DEFECT_SEVERITY,
                    summary=(
                        f"{t.file}::{t.function} parked as an expected "
                        f"failure ({cap_reason}; trigger {trigger})"
                    )[:500],
                    diagnostic={
                        "nodeid": t.nodeid, "file": t.file,
                        "function": t.function, "trigger": trigger,
                        "cap": cap_reason, "reason": outcome.reason[:300],
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("[park] defect for %s skipped: %s", t.nodeid, exc)
        try:
            conn.commit()
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def _spend_acceptance_rounds(state: dict[str, Any]) -> None:
    """Make an acceptance park stick across ``acceptance_node`` re-entry.

    ``acceptance_node`` regenerates every unverified scenario on entry, which
    drops the marker just written, and routes an attributable failure back to
    repair while the batch has rounds left — a second inner loop on the test
    that was just parked. Spending the batch's rounds sends that re-entry down
    the existing ``_park_over_cap`` path instead: quarantine what is still
    attributable, seal, move on.
    """
    cfg = state.get("acceptance_config") or (
        (state.get("harness_config") or {}).get("acceptance") or {}
    )
    try:
        max_repair = int(cfg.get("max_repair_rounds_per_batch", 2))
    except (TypeError, ValueError, AttributeError):
        max_repair = 2
    loop_counter = state.get("loop_counter")
    if not isinstance(loop_counter, dict):
        return
    try:
        batch_id = int(state.get("current_batch_id") or 0)
    except (TypeError, ValueError):
        batch_id = 0
    loop_counter["acceptance_repair"] = max(
        int(loop_counter.get("acceptance_repair", 0) or 0), max_repair,
    )
    loop_counter["acceptance_repair_batch"] = batch_id


def park_blocker(
    state: dict[str, Any], *, trigger: str, cap_reason: str,
    hitl_config: Optional[dict[str, Any]] = None,
) -> Optional[ParkOutcome]:
    """Park the blocker in ``state``, or return None when it cannot be parked.

    On success ``state`` is mutated: the park is appended to
    ``loop_counter["parked_blockers"]``, the parked functions join
    ``loop_counter["parked_test_nodeids"]``, the touched test files join
    ``modified_files``, and the flags that mark a test as un-authorable are
    cleared so they do not condemn the NEXT blocker on its first escalation.
    The caller resumes the graph. On None the caller terminates exactly as it
    did before this module existed.

    No auto-resume counter is touched. The class budget is refunded by the
    existing progress rule once the parked blocker is gone and a new one
    appears; if the park did not work, the same caps are still spent, the
    next trip finds nothing new to park, and the run ends one cycle later.
    """
    cfg = hitl_config or {}

    def _setting(name: str, default: Any) -> Any:
        _override = state.get(f"hitl_{name}")
        if _override is not None:
            return _override
        _value = cfg.get(name)
        return default if _value is None else _value

    if not bool(_setting("park_blocker_on_cap", DEFAULT_PARK_ON_CAP)):
        return None
    _trigger_class = str(trigger or "").split(":", 1)[0].strip()
    if _trigger_class in NON_TEST_TRIGGER_CLASSES:
        logger.info(
            "[park] trigger '%s' is not a failing-test blocker — nothing to "
            "park.", trigger,
        )
        return None
    max_blockers = clamp_max_parked_blockers(
        _setting("max_parked_blockers", DEFAULT_MAX_PARKED_BLOCKERS)
    )
    max_tests = clamp_max_tests_per_park(
        _setting("max_parked_tests_per_blocker", DEFAULT_MAX_TESTS_PER_PARK)
    )

    decision = decide(state, max_tests=max_tests, max_blockers=max_blockers)
    if not decision.parkable:
        logger.warning(
            "[park] blocker NOT parked (%s) — %s. Terminating as before.",
            cap_reason, decision.refusal,
        )
        try:
            from harness.observability import emit_event
            emit_event(
                "blocker_park_refused", trigger=trigger, cap=cap_reason,
                reason=decision.refusal[:300],
            )
        except Exception:  # noqa: BLE001 — telemetry must not block
            pass
        return None

    workspace = str(state.get("workspace_path") or "")
    reason = (
        f"parked by the harness after {cap_reason} on trigger '{trigger}' — "
        "no node could make this test pass; recorded as a defect"
    )
    outcome = apply(workspace, decision.tests, reason)
    if not outcome.parked:
        logger.warning(
            "[park] blocker NOT parked (%s) — none of %d failing test(s) "
            "could be marked. Terminating as before.",
            cap_reason, len(decision.tests),
        )
        return None

    loop_counter = state.get("loop_counter")
    if not isinstance(loop_counter, dict):
        loop_counter = {}
        state["loop_counter"] = loop_counter
    _history = loop_counter.get(LC_PARKED_BLOCKERS)
    history = list(_history) if isinstance(_history, list) else []
    history.append({
        "trigger": trigger,
        "cap": cap_reason,
        "tests": [f"{t.file}::{t.function}" for t in outcome.parked],
        "ac_keys": list(outcome.ac_keys),
    })
    loop_counter[LC_PARKED_BLOCKERS] = history
    _ids = loop_counter.get(LC_PARKED_NODEIDS)
    ids = list(_ids) if isinstance(_ids, list) else []
    for t in outcome.parked:
        key = f"{t.file}::{t.function}"
        if key not in ids:
            ids.append(key)
    loop_counter[LC_PARKED_NODEIDS] = ids

    if outcome.ac_keys:
        _spend_acceptance_rounds(state)

    modified = state.get("modified_files")
    if isinstance(modified, list):
        for rel in outcome.files:
            if rel not in modified:
                modified.append(rel)

    node_state = state.get("node_state")
    if isinstance(node_state, dict):
        for flag in ("test_regen_unsupported", "test_regen_stuck"):
            node_state.pop(flag, None)

    _record(state, outcome, trigger=trigger, cap_reason=cap_reason)

    logger.warning(
        "[park] PARKED blocker %d/%d instead of terminating (%s, trigger "
        "'%s'): %d test(s) in %d file(s) now carried as expected failures — "
        "%s%s. Recorded as defects; the build resumes and later batches run.",
        len(history), max_blockers, cap_reason, trigger,
        len(outcome.parked), len(outcome.files),
        ", ".join(f"{t.file}::{t.function}" for t in outcome.parked[:6]),
        (f" (+{len(outcome.parked) - 6} more)"
         if len(outcome.parked) > 6 else ""),
    )
    try:
        from harness.observability import emit_event
        emit_event(
            "blocker_parked", trigger=trigger, cap=cap_reason,
            tests=[f"{t.file}::{t.function}" for t in outcome.parked],
            ac_keys=list(outcome.ac_keys), files=list(outcome.files),
            parked_blockers=len(history), max_parked_blockers=max_blockers,
        )
    except Exception:  # noqa: BLE001 — telemetry must not block
        pass
    return outcome
