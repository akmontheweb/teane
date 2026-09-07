"""The protected-test subsystem must not depend on the judge having a good
round.

lumina session 01a079dc: for five consecutive repair rounds the top failing
diagnostic sat inside a tamper-guarded test file, and the repair LLM was never
once told the file was guarded — let alone offered the ``UNSATISFIABLE_TEST``
escape that exists for exactly this case. Three independent couplings caused
it, all of them to the reflection verdict:

  * The enclosing branch in ``repair_node`` requires a verdict that is
    DISTRACTION/REGRESSION, carries a real_blocker, and is NOT low-signal.
    Every round returned the "insufficient data" sentinel, so the whole
    branch — MUST MODIFY, PERSISTENT BLOCKER, PROTECTED TEST LOCATION and the
    escape offer — never executed.
  * ``_guarded_test_lines`` is derived from ``_verdict_named_file_lines``, so
    a verdict naming no files yields no guarded set even when the failing
    diagnostics point straight at a guarded test.
  * ``_should_offer_unsatisfiable_escape`` keys on
    ``consecutive_distraction_rounds``, which the loop deliberately HOLDS at
    zero while the judge is low-signal. The threshold was unreachable by
    construction.

The repair LLM, given no way to report a defective test, diagnosed all three
failures correctly ("the test is checking the wrong thing"), wrote "But we
can't modify tests", and shipped a production regression instead.
"""

from __future__ import annotations

from harness.graph import (
    _guarded_test_lines_from_diagnostics,
    _should_offer_unsatisfiable_escape,
    _triage_flags_guarded_test_bug,
)

WS = "/home/akhila/work/projects/lumina"

CAPLOG_CTX = (
    "failing source: assert len(caplog.records) == 1\n"
    "assertion-rewrite:\n"
    "  assert 2 == 1\n"
    "  +  where 2 = len([<LogRecord: audit, 20, "
    "server/app/middleware/audit.py, 34, \"{...}\">, "
    "<LogRecord: httpx, 20, "
    "/tmp/teane-venv/lib/python3.12/site-packages/httpx/_client.py, 1025, "
    "\"HTTP Request: %s %s\">])\n"
)

# The real lumina round-8 failing set.
LUMINA_ERRORS = [
    {"file": "server/tests/test_main.py", "line": 22,
     "error_code": "RuntimeError", "message": "boom"},
    {"file": "server/tests/test_middleware.py", "line": 118,
     "error_code": "AssertionError", "message": "AssertionError: assert 2 == 1",
     "semantic_context": CAPLOG_CTX},
]


class TestGuardedLinesFromDiagnostics:
    def test_guarded_test_lines_are_found_without_a_verdict(self):
        out = _guarded_test_lines_from_diagnostics(
            LUMINA_ERRORS, WS, frozenset(),
        )
        assert ("server/tests/test_main.py", 22) in out
        assert ("server/tests/test_middleware.py", 118) in out

    def test_production_frames_are_excluded(self):
        errs = [{"file": "server/app/middleware/audit.py", "line": 21,
                 "error_code": "RuntimeError", "message": "boom"}]
        assert _guarded_test_lines_from_diagnostics(errs, WS, frozenset()) == []

    def test_absolute_paths_are_relativized(self):
        errs = [{"file": f"{WS}/server/tests/test_main.py", "line": 22,
                 "error_code": "RuntimeError", "message": "boom"}]
        out = _guarded_test_lines_from_diagnostics(errs, WS, frozenset())
        assert out == [("server/tests/test_main.py", 22)]

    def test_carveout_files_are_excluded(self):
        """A parse-broken test file is already editable by repair, so it is
        not a protected blocker — offering the escape on it would send a
        fixable file to regeneration."""
        out = _guarded_test_lines_from_diagnostics(
            LUMINA_ERRORS, WS, frozenset({"server/tests/test_main.py"}),
        )
        assert [f for f, _ in out] == ["server/tests/test_middleware.py"]

    def test_empty_and_malformed_input(self):
        assert _guarded_test_lines_from_diagnostics([], WS, frozenset()) == []
        assert _guarded_test_lines_from_diagnostics(
            [None, {}, {"file": ""}], WS, frozenset(),
        ) == []

    def test_deduplicates_and_caps(self):
        errs = [
            {"file": "server/tests/test_main.py", "line": 22,
             "error_code": "RuntimeError", "message": "boom"},
        ] * 4 + [
            {"file": f"server/tests/test_{i}.py", "line": i,
             "error_code": "AssertionError", "message": "x"}
            for i in range(10)
        ]
        out = _guarded_test_lines_from_diagnostics(
            errs, WS, frozenset(), limit=5,
        )
        assert len(out) == 5
        assert len(set(out)) == 5


class TestTriageFlagsGuardedTestBug:
    def test_lumina_caplog_failure_is_flagged(self):
        assert _triage_flags_guarded_test_bug(
            LUMINA_ERRORS, WS, frozenset(),
        ) is True

    def test_plain_behaviour_assertion_is_not_flagged(self):
        errs = [{"file": "server/tests/test_api.py", "line": 10,
                 "error_code": "AssertionError",
                 "message": "AssertionError: assert 500 == 200"}]
        assert _triage_flags_guarded_test_bug(errs, WS, frozenset()) is False

    def test_test_bug_in_a_carved_out_file_is_not_flagged(self):
        assert _triage_flags_guarded_test_bug(
            LUMINA_ERRORS, WS,
            frozenset({"server/tests/test_main.py",
                       "server/tests/test_middleware.py"}),
        ) is False

    def test_production_frame_is_not_flagged(self):
        errs = [{"file": "server/app/db.py", "line": 10,
                 "error_code": "NameError",
                 "message": "name 'patch' is not defined"}]
        assert _triage_flags_guarded_test_bug(errs, WS, frozenset()) is False

    def test_malformed_input_is_survivable(self):
        assert _triage_flags_guarded_test_bug(
            [None, {}, {"file": ""}], WS, frozenset(),
        ) is False


class TestEscapeGateNewTriggers:
    def test_low_signal_streak_unlocks_the_escape(self):
        """The lumina case: distraction_streak pinned at 0 by the low-signal
        sentinel while consecutive_low_signal_rounds climbed to 5."""
        assert _should_offer_unsatisfiable_escape(
            has_guarded_blocker=True,
            has_mandatable_target=True,
            distraction_streak=0,
            low_signal_streak=2,
        ) is True

    def test_single_low_signal_round_does_not_unlock(self):
        assert _should_offer_unsatisfiable_escape(
            has_guarded_blocker=True,
            has_mandatable_target=True,
            distraction_streak=0,
            low_signal_streak=1,
        ) is False

    def test_triage_test_bug_unlocks_immediately(self):
        """A positive classifier identification is evidence in its own right
        — it should not have to wait for the loop to burn two rounds."""
        assert _should_offer_unsatisfiable_escape(
            has_guarded_blocker=True,
            has_mandatable_target=True,
            distraction_streak=0,
            low_signal_streak=0,
            triage_test_bug=True,
        ) is True

    def test_guarded_blocker_still_required(self):
        """Neither new trigger may fire on a production blocker."""
        for kwargs in (
            {"low_signal_streak": 9},
            {"triage_test_bug": True},
        ):
            assert _should_offer_unsatisfiable_escape(
                has_guarded_blocker=False,
                has_mandatable_target=True,
                distraction_streak=9,
                **kwargs,
            ) is False

    def test_existing_behaviour_unchanged(self):
        # Defaults keep the pre-existing gate exactly as it was.
        assert _should_offer_unsatisfiable_escape(
            has_guarded_blocker=True, has_mandatable_target=True,
            distraction_streak=1,
        ) is False
        assert _should_offer_unsatisfiable_escape(
            has_guarded_blocker=True, has_mandatable_target=True,
            distraction_streak=2,
        ) is True
        assert _should_offer_unsatisfiable_escape(
            has_guarded_blocker=True, has_mandatable_target=False,
            distraction_streak=0,
        ) is True


class TestLuminaRound8EndToEnd:
    """The composed decision on the real round-8 inputs. Every one of these
    was False or empty at the time; the run got no banner and no escape."""

    def test_escape_would_now_be_offered(self):
        carveout = frozenset()
        guarded = _guarded_test_lines_from_diagnostics(
            LUMINA_ERRORS, WS, carveout,
        )
        assert guarded, "guarded blocker must be visible without a verdict"

        triage = _triage_flags_guarded_test_bug(LUMINA_ERRORS, WS, carveout)
        assert triage is True

        # The floor claims has_mandatable_target=True so the offer must earn
        # itself; on this run BOTH remaining triggers fire.
        assert _should_offer_unsatisfiable_escape(
            has_guarded_blocker=True,
            has_mandatable_target=True,
            distraction_streak=0,     # held at 0 by the low-signal sentinel
            low_signal_streak=5,
            triage_test_bug=triage,
        ) is True

    def test_ordinary_first_round_test_failure_gets_no_offer(self):
        """The guard against over-firing: an all-test failing set on round 1
        with no fingerprint and no streak is the common shape of a real code
        gap, and must still be sent to repair."""
        errs = [{"file": "server/tests/test_api.py", "line": 10,
                 "error_code": "AssertionError",
                 "message": "AssertionError: assert 500 == 404"}]
        carveout = frozenset()
        assert _guarded_test_lines_from_diagnostics(errs, WS, carveout)
        assert _should_offer_unsatisfiable_escape(
            has_guarded_blocker=True,
            has_mandatable_target=True,
            distraction_streak=0,
            low_signal_streak=0,
            triage_test_bug=_triage_flags_guarded_test_bug(
                errs, WS, carveout,
            ),
        ) is False
