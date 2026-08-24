"""In-build acceptance-run engine (ADR-0006, Phase 1).

The graph-independent core that ``acceptance_node`` orchestrates: given generated
integration scenarios and a way to run them in the sandbox, it runs each AC once,
triages every failure into one of three buckets, and returns a structured result
the node uses to decide routing (repair vs forward) and what to record/defer.

Triage buckets (ADR-0006 §Decision.3):
  * ``attributable``            — the app responded and behaved wrong; a real
                                  code defect this batch can fix → repair loop.
  * ``deferred:blocked-by-dependency`` — a prerequisite isn't built yet (app
                                  won't import, route 404s, fixture missing) →
                                  defer + re-queue, DO NOT fail the batch.
  * ``test-bug``                — the generated test itself is broken → drop
                                  (Phase 1 does not regenerate; that's a later
                                  item). Never routed to repair.

Conservative-default safety property (mirrors ADR-0005's triage): on any doubt a
failure is ``deferred``, never ``attributable``. The engine can add a repair
round or defer, but can never manufacture a hard failure that stalls a headless
run — the finsearch 156032347 lesson. The end-of-run traceability gate and the
post-deploy ``teane test`` pass remain the backstops for anything deferred.

This module is pure: the sandbox runner is injected as a callable, so it is fully
unit-testable without a graph, a gateway, or a real sandbox.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from harness.acceptance_gen import (
    ALTITUDE_INTEGRATION,
    AcceptanceScenario,
    integration_function_name,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statuses
# ---------------------------------------------------------------------------

STATUS_PASSED = "passed"
STATUS_ATTRIBUTABLE = "attributable"
STATUS_DEFERRED_DEP = "deferred:blocked-by-dependency"
STATUS_DEFERRED_COLLECTION = "deferred:collection-error"
STATUS_DEFERRED_UI = "deferred:needs-browser"
STATUS_TEST_BUG = "test-bug"

# The statuses that mean "not verified, but not this batch's fault" — parked,
# re-queued, or handed to the post-deploy browser pass.
#
# ``deferred:collection-error`` is distinct from ``blocked-by-dependency`` on
# purpose (Fix B): the latter means a genuine prerequisite isn't built yet, the
# former means the acceptance SUITE ITSELF could not be collected/executed
# (the app didn't import, the sandbox lacked the app's deps, an install step
# failed). Both defer — the never-hard-fail safety property is preserved — but
# conflating them hid the real signal: "acceptance ran and everything was fine"
# vs "acceptance never actually ran". Keeping them apart makes the batch
# summary and traceability honest and lets the re-queue path retry a suite that
# was merely under-provisioned.
_DEFERRED = frozenset({
    STATUS_DEFERRED_DEP, STATUS_DEFERRED_COLLECTION,
    STATUS_DEFERRED_UI, STATUS_TEST_BUG,
})


class AcceptanceCollectionError(Exception):
    """Raised by a runner when the acceptance suite could not be
    collected/executed at all — a whole-suite import/collection failure or a
    failed environment provisioning step, as opposed to individual test
    failures. ``run_acceptance`` catches it and marks every runnable AC
    :data:`STATUS_DEFERRED_COLLECTION` (honest "not run", still deferred)
    rather than silently fanning the batch out to ``blocked-by-dependency``.
    """


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------


@dataclass
class TestOutcome:
    """One executed test's result, normalised from whatever the runner returns.

    ``nodeid`` is the pytest-style ``path::func`` id; ``message`` is the failure
    text (empty when passed). The node adapts the real sandbox runner into a list
    of these so the engine stays runner-agnostic.
    """

    __test__ = False  # not a pytest test class despite the ``Test`` prefix

    nodeid: str
    passed: bool
    message: str = ""


@dataclass
class ACOutcome:
    ac_key: str
    status: str
    altitude: str = ALTITUDE_INTEGRATION
    detail: str = ""

    @property
    def is_deferred(self) -> bool:
        return self.status in _DEFERRED


@dataclass
class AcceptanceRunResult:
    story_keys: list[str] = field(default_factory=list)
    outcomes: list[ACOutcome] = field(default_factory=list)
    ran: bool = True
    """False when nothing runnable was found / the run was skipped."""

    def by_status(self, status: str) -> list[ACOutcome]:
        return [o for o in self.outcomes if o.status == status]

    def passed(self) -> list[ACOutcome]:
        return self.by_status(STATUS_PASSED)

    def attributable(self) -> list[ACOutcome]:
        return self.by_status(STATUS_ATTRIBUTABLE)

    def deferred(self) -> list[ACOutcome]:
        return [o for o in self.outcomes if o.is_deferred]

    def has_attributable(self) -> bool:
        return any(o.status == STATUS_ATTRIBUTABLE for o in self.outcomes)


# ---------------------------------------------------------------------------
# Failure triage
# ---------------------------------------------------------------------------

# Dependency-blocked fingerprints — a prerequisite the AC needs isn't ready.
# These are NOT this batch's defect, so they defer (re-queue) rather than repair.
_DEP_BLOCKED_RES = (
    re.compile(r"\b(ModuleNotFoundError|ImportError)\b"),
    re.compile(r"\bcollection error\b", re.IGNORECASE),
    re.compile(r"\bconftest\b.*\berror\b", re.IGNORECASE),
    re.compile(r"fixture '(client|app|async_client)' not found"),
    re.compile(r"\b(ConnectionError|ConnectionRefused|Failed to establish a new connection)\b"),
    re.compile(r"\b404\b|\bNot Found\b"),
    re.compile(r"\bapp\b.*\bfailed to (import|start|boot)\b", re.IGNORECASE),
)

# Test-authoring bug fingerprints — the generated test itself is wrong. Reuses
# the ADR-0005 test-bug intuition; conservative (message-only, high confidence).
_TEST_BUG_RES = (
    re.compile(r"NameError: name '.*' is not defined"),
    re.compile(r"cannot import name '.*' from"),
    re.compile(r"AttributeError: .* does not have the attribute"),
    re.compile(r"SyntaxError|IndentationError"),
)

# Attributable — the app responded and behaved wrong. A plain assertion mismatch
# on a received response is the class the repair loop is good at.
_ATTRIBUTABLE_RES = (
    re.compile(r"\bAssertionError\b"),
    re.compile(r"\bassert\b.*\b(status_code|response|json)\b"),
    re.compile(r"\b(500|502|503)\b"),  # server error on a route that DID respond
)


def classify_acceptance_failure(message: str) -> str:
    """Bucket one failure message. Conservative: defaults to deferred on doubt.

    Order matters: dependency-blocked and test-bug are checked BEFORE
    attributable, because a 404/import-error can co-occur with an assertion in
    the same traceback and must not be mistaken for a real code defect.
    """
    m = message or ""
    for rx in _DEP_BLOCKED_RES:
        if rx.search(m):
            return STATUS_DEFERRED_DEP
    for rx in _TEST_BUG_RES:
        if rx.search(m):
            return STATUS_TEST_BUG
    for rx in _ATTRIBUTABLE_RES:
        if rx.search(m):
            return STATUS_ATTRIBUTABLE
    # Unknown failure shape → defer (never manufacture an attributable failure).
    return STATUS_DEFERRED_DEP


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

# A runner takes (list of test file paths, workspace) and returns normalised
# per-test outcomes. The node supplies the real sandbox runner; tests supply a fake.
AcceptanceRunner = Callable[[list[str], str], list[TestOutcome]]


def select_runnable(
    scenarios: list[AcceptanceScenario],
    *,
    already_passed: Optional[set[str]] = None,
) -> list[AcceptanceScenario]:
    """Integration scenarios whose AC hasn't already passed (run-each-AC-once).

    ui-only ACs never have an integration scenario, so they are naturally
    excluded here and handled as ``deferred:needs-browser`` by the caller.
    """
    already = already_passed or set()
    return [
        s for s in scenarios
        if s.altitude == ALTITUDE_INTEGRATION and s.verifies not in already
    ]


def run_acceptance(
    scenarios: list[AcceptanceScenario],
    written_paths: list[str],
    workspace_path: str,
    *,
    runner: AcceptanceRunner,
    story_keys: Optional[list[str]] = None,
    already_passed: Optional[set[str]] = None,
) -> AcceptanceRunResult:
    """Run the batch's integration scenarios and triage the results.

    ``written_paths`` are the integration files already rendered to disk (the node
    writes them); ``runner`` executes them in the sandbox. Each scenario maps to a
    pytest function via :func:`integration_function_name`; the outcome for that
    function determines the AC's status.
    """
    runnable = select_runnable(scenarios, already_passed=already_passed)
    if not runnable:
        return AcceptanceRunResult(story_keys=list(story_keys or []), ran=False)

    # func-name → scenario, to map nodeids back to ACs.
    by_func: dict[str, AcceptanceScenario] = {
        integration_function_name(s): s for s in runnable
    }

    try:
        outcomes = runner(written_paths, workspace_path)
    except AcceptanceCollectionError as exc:
        # The suite never ran (import/collection failure or a failed
        # provisioning step). Record it as its own honest bucket instead of
        # the misleading "a prerequisite isn't built yet" — the app may well
        # be complete; the run itself is what failed.
        logger.warning(
            "[acceptance_run] suite did not run (collection/provisioning "
            "error) — recording %d AC(s) as %s, not blocked-by-dependency.",
            len(runnable), STATUS_DEFERRED_COLLECTION,
        )
        return AcceptanceRunResult(
            story_keys=list(story_keys or []),
            outcomes=[ACOutcome(s.verifies, STATUS_DEFERRED_COLLECTION,
                                s.altitude, str(exc)[:500]) for s in runnable],
        )
    except Exception as exc:  # noqa: BLE001 — a runner crash defers the batch, never fails it
        logger.warning("[acceptance_run] runner crashed: %s — deferring batch", exc)
        return AcceptanceRunResult(
            story_keys=list(story_keys or []),
            outcomes=[ACOutcome(s.verifies, STATUS_DEFERRED_DEP, s.altitude,
                                f"runner error: {exc}") for s in runnable],
        )

    outcome_by_func = _index_outcomes_by_func(outcomes)
    ac_outcomes: list[ACOutcome] = []
    for func, sc in by_func.items():
        out = outcome_by_func.get(func)
        if out is None:
            # The test never ran (collection dropped it) → dependency-blocked.
            ac_outcomes.append(ACOutcome(sc.verifies, STATUS_DEFERRED_DEP, sc.altitude,
                                         "not collected"))
            continue
        if out.passed:
            ac_outcomes.append(ACOutcome(sc.verifies, STATUS_PASSED, sc.altitude))
        else:
            status = classify_acceptance_failure(out.message)
            ac_outcomes.append(ACOutcome(sc.verifies, status, sc.altitude,
                                         out.message[:500]))
    return AcceptanceRunResult(story_keys=list(story_keys or []), outcomes=ac_outcomes)


def _index_outcomes_by_func(outcomes: list[TestOutcome]) -> dict[str, TestOutcome]:
    """Map a pytest nodeid's function part → outcome.

    A nodeid looks like ``tests/acceptance/test_x.py::test_add_contact`` (or with a
    parametrisation suffix ``[...]``). We key on the bare function name so the
    scenario mapping is independent of the file path the node chose.
    """
    out: dict[str, TestOutcome] = {}
    for o in outcomes:
        func = o.nodeid.rsplit("::", 1)[-1]
        func = func.split("[", 1)[0].strip()
        # First failure wins over a later pass for the same func name (defensive).
        if func not in out or (out[func].passed and not o.passed):
            out[func] = o
    return out


# ---------------------------------------------------------------------------
# Pytest output parsing — adapt a sandbox run into TestOutcomes
# ---------------------------------------------------------------------------

# Pytest's ``-rA`` short-summary block lists every test with an outcome prefix.
# PASSED lines carry no message; FAILED/ERROR lines carry `- <message>`.
_SUMMARY_LINE_RE = re.compile(
    r"^(?P<status>PASSED|FAILED|ERROR|XFAIL|XPASS)\s+"
    r"(?P<nodeid>\S+::\S+?)\s*(?:-\s*(?P<message>.*))?$"
)

# Pytest emits "no tests ran" / collection-error banners when the whole file
# fails to import — every scenario in it is dependency-blocked, not failed.
_COLLECTION_ERROR_RE = re.compile(r"errors? during collection|ERROR collecting", re.IGNORECASE)


# A FAILURES/ERRORS detail block starts with an underscored header naming the
# test, e.g. ``____ test_add_valid_contact ____`` or
# ``__ ERROR at setup of test_add __``. Assertion/exception text follows on
# ``E   `` lines. Parsing these recovers the failure message even when pytest's
# short-summary line omits it (multi-line assertions), which triage NEEDS to tell
# an attributable failure from a deferred one.
_BLOCK_HEADER_RE = re.compile(r"^_{3,}\s+(?P<title>.+?)\s+_{3,}$")
_E_LINE_RE = re.compile(r"^E\s+(?P<text>.*)$")


def _extract_failure_messages(raw_output: str) -> dict[str, str]:
    """Map a test's function name → its joined ``E`` failure lines."""
    out: dict[str, str] = {}
    cur_func: Optional[str] = None
    cur_lines: list[str] = []

    def _flush() -> None:
        if cur_func and cur_lines:
            out.setdefault(cur_func, " ".join(cur_lines).strip())

    for line in (raw_output or "").splitlines():
        hm = _BLOCK_HEADER_RE.match(line.strip())
        if hm:
            _flush()
            title = hm.group("title")
            # "ERROR at setup of test_x" / "TestClass.test_x" / "test_x[param]"
            func = title.split()[-1]
            func = func.rsplit(".", 1)[-1].split("[", 1)[0].strip()
            cur_func, cur_lines = func, []
            continue
        em = _E_LINE_RE.match(line.rstrip())
        if em and cur_func:
            cur_lines.append(em.group("text").strip())
    _flush()
    return out


def parse_pytest_outcomes(raw_output: str) -> list[TestOutcome]:
    """Parse pytest ``-rA`` output into per-test :class:`TestOutcome`.

    Robust to version drift: keys off the short-summary block for pass/fail, and
    backfills each failure's message from the FAILURES/ERRORS detail sections
    (the summary line often omits multi-line assertion text). When a whole file
    failed to collect (import error), the summary lists ``ERROR <file>`` without a
    ``::func`` — surfaced as a collection-error outcome so the caller defers.
    """
    detail = _extract_failure_messages(raw_output)
    outcomes: list[TestOutcome] = []
    seen: set[str] = set()
    for line in (raw_output or "").splitlines():
        m = _SUMMARY_LINE_RE.match(line.strip())
        if not m:
            continue
        nodeid = m.group("nodeid")
        if nodeid in seen:
            continue
        seen.add(nodeid)
        status = m.group("status")
        passed = status in ("PASSED", "XFAIL")  # xfail = expected-fail, not a real failure
        message = (m.group("message") or "").strip()
        if not passed and not message:
            func = nodeid.rsplit("::", 1)[-1].split("[", 1)[0].strip()
            message = detail.get(func, "")
        if status == "ERROR" and not message:
            message = "collection/setup error"
        outcomes.append(TestOutcome(nodeid=nodeid, passed=passed, message=message))
    return outcomes


def is_collection_error(raw_output: str) -> bool:
    """True when pytest reported a collection/import error for the run.

    Signals the whole batch's acceptance file couldn't even import — every AC in
    it is dependency-blocked (a prerequisite module isn't built), not failed.
    """
    return bool(_COLLECTION_ERROR_RE.search(raw_output or ""))


def summarize(result: AcceptanceRunResult) -> dict[str, Any]:
    """Compact counts for logging / node_state."""
    counts: dict[str, int] = {}
    for o in result.outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
    return {
        "ran": result.ran,
        "total": len(result.outcomes),
        "counts": counts,
        "passed_acs": [o.ac_key for o in result.passed()],
        "attributable_acs": [o.ac_key for o in result.attributable()],
        "deferred_acs": [(o.ac_key, o.status) for o in result.deferred()],
    }


__all__ = [
    "STATUS_PASSED", "STATUS_ATTRIBUTABLE", "STATUS_DEFERRED_DEP",
    "STATUS_DEFERRED_COLLECTION", "STATUS_DEFERRED_UI", "STATUS_TEST_BUG",
    "AcceptanceCollectionError",
    "TestOutcome", "ACOutcome", "AcceptanceRunResult", "AcceptanceRunner",
    "classify_acceptance_failure", "select_runnable", "run_acceptance", "summarize",
    "parse_pytest_outcomes", "is_collection_error",
]
