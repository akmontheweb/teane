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
    _apply_unsat_offer_floor,
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


# ---------------------------------------------------------------------------
# The declined-offer floor.
#
# The escape is MODEL-DECLARED: the harness prints the offer and the model
# must emit an ``UNSATISFIABLE_TEST: <path>`` line for anything to happen. A
# model that never writes the line declines it forever.
#
# lumina-fresh-20260911-1107 is the case. The harness offered the escape 15
# times and was correct every time — ``server/tests/test_birthdays_api.py:36``
# builds dates of birth with ``today.replace(day=today.day + 1)``, which keeps
# the current year, so both POSTs are future-dated and the app's own
# ``validate_date_not_future`` rejects them with 422. The test never asserts
# the POST status, so it GETs an empty list and fails ``len == 2``. No
# production change can satisfy it. The model declined all 15 offers and
# worked through main.py, deps.py and config.py while the build died on the
# HITL auto-resume cap.
#
# The subsystem holding the correct diagnosis could only suggest; the
# reflection judge — wrong 13 rounds running — could compel via the
# MUST-MODIFY promotion. This floor gives the correct one a way to act.
# ---------------------------------------------------------------------------

_LUMINA_TEST = "server/tests/test_birthdays_api.py"


class TestDeclinedOfferFloor:

    def test_below_the_floor_only_counts(self):
        declined: dict[str, int] = {}
        for _ in range(2):
            assert _apply_unsat_offer_floor(
                declined, {_LUMINA_TEST},
                any_real_patch=False, already_declared=False, floor=3,
            ) is None
        assert declined[_LUMINA_TEST] == 2

    def test_floor_forces_the_declaration(self):
        declined: dict[str, int] = {}
        out = None
        for _ in range(3):
            out = _apply_unsat_offer_floor(
                declined, {_LUMINA_TEST},
                any_real_patch=False, already_declared=False, floor=3,
            )
        assert out is not None
        path, reason = out
        assert path == _LUMINA_TEST
        # The reason must say who declared it — a post-mortem reading this
        # should not mistake it for the model's own words.
        assert "harness-declared" in reason
        assert "3 declined" in reason

    def test_counter_clears_after_forcing(self):
        """So a later, genuinely different blocker starts from zero rather
        than tripping on the first offer."""
        declined: dict[str, int] = {}
        for _ in range(3):
            _apply_unsat_offer_floor(
                declined, {_LUMINA_TEST},
                any_real_patch=False, already_declared=False, floor=3,
            )
        assert _LUMINA_TEST not in declined

    def test_a_real_patch_resets_the_counter(self):
        """Production moving means the loop is working. The floor is for a
        loop that is stuck, not for a model taking a few rounds."""
        declined: dict[str, int] = {}
        for _ in range(2):
            _apply_unsat_offer_floor(
                declined, {_LUMINA_TEST},
                any_real_patch=False, already_declared=False, floor=3,
            )
        assert _apply_unsat_offer_floor(
            declined, {_LUMINA_TEST},
            any_real_patch=True, already_declared=False, floor=3,
        ) is None
        assert declined == {}
        # And the next decline starts from one, not three.
        _apply_unsat_offer_floor(
            declined, {_LUMINA_TEST},
            any_real_patch=False, already_declared=False, floor=3,
        )
        assert declined[_LUMINA_TEST] == 1

    def test_model_declaration_suppresses_forcing(self):
        """When the model speaks for itself there is nothing to force, and
        the round must not also count as a decline."""
        declined: dict[str, int] = {}
        assert _apply_unsat_offer_floor(
            declined, {_LUMINA_TEST},
            any_real_patch=False, already_declared=True, floor=3,
        ) is None
        assert declined == {}

    def test_floor_of_zero_disables_the_mechanism(self):
        declined: dict[str, int] = {}
        for _ in range(10):
            assert _apply_unsat_offer_floor(
                declined, {_LUMINA_TEST},
                any_real_patch=False, already_declared=False, floor=0,
            ) is None

    def test_empty_offer_set_is_a_no_op(self):
        declined: dict[str, int] = {}
        assert _apply_unsat_offer_floor(
            declined, set(), any_real_patch=False,
            already_declared=False, floor=3,
        ) is None
        assert declined == {}

    def test_the_worst_offender_is_chosen(self):
        """Offers can name several files; force the one that has been
        declined most, not an arbitrary member of the set."""
        declined = {"tests/a.py": 2, "tests/b.py": 0}
        out = _apply_unsat_offer_floor(
            declined, {"tests/a.py", "tests/b.py"},
            any_real_patch=False, already_declared=False, floor=3,
        )
        assert out is not None and out[0] == "tests/a.py"

    def test_counts_are_per_file(self):
        """Two files each declined twice must not sum to a trip at floor 3."""
        declined: dict[str, int] = {}
        for _ in range(2):
            assert _apply_unsat_offer_floor(
                declined, {"tests/a.py", "tests/b.py"},
                any_real_patch=False, already_declared=False, floor=3,
            ) is None
        assert declined == {"tests/a.py": 2, "tests/b.py": 2}

    def test_the_fifteen_decline_lumina_sequence_trips_at_three(self):
        """End-to-end shape of the real session: the offer repeats on the
        same file with no production patch ever landing. The build should
        stop losing rounds at 3, not 15."""
        declined: dict[str, int] = {}
        forced_at = None
        for round_no in range(1, 16):
            out = _apply_unsat_offer_floor(
                declined, {_LUMINA_TEST},
                any_real_patch=False, already_declared=False, floor=3,
            )
            if out is not None and forced_at is None:
                forced_at = round_no
        assert forced_at == 3


class TestAssertedCorrectWeighting:
    """An asserted-correct round is the escape claim without the keyword.

    lumina-run7-20260915-1345: the model re-emitted a byte-identical
    ``BodySizeLimitMiddleware`` five times (repair calls 0072/0075/0078/
    0081/0084, every response md5 8cea442d8d68) while the real defect sat in
    ``test_body_size_limit.py``, which posts raw bytes to an endpoint that
    parses JSON. Every one of those rounds scored zero real patches, drove
    ``zero_patch_loop`` to its 3/3 cap, and killed the build.

    "This production file is already correct" and "no production change can
    satisfy this test" are the same proposition, so the round counts double
    toward the floor rather than reading as a silent decline.
    """

    def test_asserted_round_reaches_floor_in_half_the_rounds(self) -> None:
        declined: dict[str, int] = {}
        offers = {"server/tests/test_body_size_limit.py"}
        kw = dict(any_real_patch=False, already_declared=False, floor=4)

        first = _apply_unsat_offer_floor(
            declined, offers, asserted_correct=True, **kw
        )
        assert first is None, "one asserted round must not reach a floor of 4"

        second = _apply_unsat_offer_floor(
            declined, offers, asserted_correct=True, **kw
        )
        assert second is not None, "two asserted rounds must reach floor 4"
        assert second[0] == "server/tests/test_body_size_limit.py"

    def test_silent_declines_are_unchanged(self) -> None:
        """The default path must keep its old arithmetic exactly."""
        declined: dict[str, int] = {}
        offers = {"server/tests/test_birthdays_api.py"}
        kw = dict(any_real_patch=False, already_declared=False, floor=3)

        assert _apply_unsat_offer_floor(declined, offers, **kw) is None
        assert _apply_unsat_offer_floor(declined, offers, **kw) is None
        forced = _apply_unsat_offer_floor(declined, offers, **kw)
        assert forced is not None, "third silent decline must reach floor 3"

    def test_a_real_patch_still_clears_an_asserted_counter(self) -> None:
        """Production moving outranks the claim that production is done."""
        declined: dict[str, int] = {}
        offers = {"server/tests/test_body_size_limit.py"}

        _apply_unsat_offer_floor(
            declined, offers, any_real_patch=False,
            already_declared=False, floor=4, asserted_correct=True,
        )
        assert declined, "the asserted round should have counted"

        _apply_unsat_offer_floor(
            declined, offers, any_real_patch=True,
            already_declared=False, floor=4, asserted_correct=True,
        )
        assert not declined, "a real patch must reset the counter"
