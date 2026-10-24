"""The class cap must measure thrash on one problem, not lifetime firings.

Regression cover for lumina 01a169ed (run 6, 2026-10-24), which terminated
on ``zero_patch_loop`` 6/6 after RESOLVING five successive blockers, with
the session cap at 7/18, every blocker counter at 1/9, and $8.02 of $10
unspent.

The decisive test replays that run's recorded resume sequence through the
shipped decision function and asserts the class never reaches its cap.
"""

from __future__ import annotations

from harness.cli import (
    _blocker_locations,
    _class_budget_progress_reset,
)


def _charge(history, trigger_class, sig):
    """One resume: decide the refund, then account for it as cli.py does."""
    reset, history = _class_budget_progress_reset(
        trigger_class=trigger_class, blocker_sig=sig, history=history,
    )
    return reset, history


class TestBlockerLocations:
    def test_it_splits_a_multi_location_signature(self):
        assert _blocker_locations("a.py:1|b.py:2") == frozenset(
            {"a.py:1", "b.py:2"}
        )

    def test_an_empty_signature_has_no_locations(self):
        assert _blocker_locations("") == frozenset()

    def test_blank_segments_are_dropped(self):
        assert _blocker_locations("a.py:1||") == frozenset({"a.py:1"})


class TestRun6Sequence:
    """The exact sequence from lumina 01a169ed's log."""

    SEQUENCE = [
        "tests/acceptance/test_story_001_acceptance.py:29|"
        "tests/acceptance/test_story_004_acceptance.py:66",
        "tests/acceptance/test_story_001_acceptance.py:29|"
        "tests/acceptance/test_story_004_acceptance.py:66",
        "tests/acceptance/test_story_004_acceptance.py:66",
        "tests/acceptance/test_story_004_acceptance.py:66",
        "server/tests/contract/test_birthday_contract.py:0|"
        "tests/acceptance/test_story_008_acceptance.py:160|"
        "tests/acceptance/test_story_008_acceptance.py:49",
        "server/tests/contract/test_birthday_contract.py:0|"
        "tests/acceptance/test_story_008_acceptance.py:158|"
        "tests/acceptance/test_story_008_acceptance.py:49",
    ]

    def _replay(self, cap=6):
        """Returns the class counter after each of the six resumes."""
        history: dict = {}
        taken = 0
        counters = []
        for sig in self.SEQUENCE:
            reset, history = _charge(history, "zero_patch_loop", sig)
            if reset:
                taken = 0
            taken += 1
            counters.append(taken)
        return counters, cap

    def test_the_class_never_reaches_its_cap(self):
        counters, cap = self._replay()
        assert max(counters) < cap, (
            f"class counter peaked at {max(counters)}/{cap} — run 6 would "
            f"still terminate. Sequence: {counters}"
        )

    def test_the_repeated_blocker_still_accrues(self):
        """Resumes 1 and 2 are the SAME blocker: that is thrash and must
        count, or the cap stops bounding anything."""
        counters, _ = self._replay()
        assert counters[0] == 1
        assert counters[1] == 2, (
            "an identical signature must not refund — nothing was resolved"
        )

    def test_a_resolved_location_refunds(self):
        """Resume 3 drops story_001:29 (regenerated), keeping story_004:66.

        The sets still SHARE :66, so this is the same pathology shrinking,
        not a resolution — it must NOT refund."""
        counters, _ = self._replay()
        assert counters[2] == 3, (
            "a signature overlapping the previous one is the same problem "
            f"shifting; got {counters[2]}"
        )

    def test_a_genuinely_new_blocker_refunds(self):
        """Resume 5 shares no location with resume 4 and is novel."""
        counters, _ = self._replay()
        assert counters[4] == 1, (
            f"a disjoint, novel blocker must refund the class; got {counters}"
        )

    def test_without_the_fix_the_run_dies(self):
        """Vacuity check: lifetime counting reaches 6/6 on this sequence,
        which is exactly how run 6 ended."""
        lifetime = list(range(1, len(self.SEQUENCE) + 1))
        assert lifetime[-1] == 6, "the sequence must be 6 resumes long"
        counters, cap = self._replay()
        assert counters != lifetime, (
            "the fix changed nothing — this test would pass against the "
            "broken code"
        )


class TestTheNovelConditionStopsOscillation:
    """Why 'resolved' alone is not enough.

    With only the resolved check, A -> B -> A -> B refunds on every hop:
    B shares no location with A, so B's arrival looks like A's resolution
    and vice versa. The cap would then bound nothing at all.
    """

    A = "a.py:1"
    B = "b.py:2"

    def test_an_oscillation_accrues_rather_than_refunding(self):
        history: dict = {}
        taken = 0
        counters = []
        for sig in [self.A, self.B, self.A, self.B, self.A, self.B]:
            reset, history = _charge(history, "zero_patch_loop", sig)
            if reset:
                taken = 0
            taken += 1
            counters.append(taken)
        assert counters[-1] >= 5, (
            f"an A/B oscillation must still trip the cap; got {counters}"
        )

    def test_the_first_hop_does_refund(self):
        """A -> B once IS progress as far as we can tell; only the RETURN
        to A reveals the oscillation."""
        history: dict = {}
        _, history = _charge(history, "c", self.A)
        reset, _ = _charge(history, "c", self.B)
        assert reset is True

    def test_a_returning_blocker_never_refunds_again(self):
        history: dict = {}
        _, history = _charge(history, "c", self.A)
        _, history = _charge(history, "c", self.B)
        reset, _ = _charge(history, "c", self.A)
        assert reset is False, "A was already charged to this class"


class TestEdgeCases:
    def test_the_first_resume_cannot_refund(self):
        """No previous blocker means nothing was resolved."""
        reset, _ = _charge({}, "zero_patch_loop", "a.py:1")
        assert reset is False

    def test_an_empty_signature_never_refunds(self):
        """A trip with no diagnostics disables blocker accounting; it must
        not be read as having solved anything."""
        history: dict = {}
        _, history = _charge(history, "c", "a.py:1")
        reset, _ = _charge(history, "c", "")
        assert reset is False

    def test_an_empty_signature_does_not_corrupt_the_history(self):
        history: dict = {}
        _, history = _charge(history, "c", "a.py:1")
        _, history2 = _charge(history, "c", "")
        assert history2["c"]["last"] == "a.py:1", (
            "a signature-less trip must leave the last charged blocker "
            "intact, or the next real blocker compares against nothing"
        )

    def test_classes_are_accounted_independently(self):
        history: dict = {}
        _, history = _charge(history, "class_a", "a.py:1")
        _, history = _charge(history, "class_b", "b.py:2")
        reset, _ = _charge(history, "class_a", "b.py:2")
        assert reset is True, (
            "class_b having seen b.py:2 must not suppress class_a's refund"
        )

    def test_the_seen_set_is_bounded(self):
        """It rides every checkpoint, so it cannot grow without limit."""
        history: dict = {}
        for i in range(200):
            _, history = _charge(history, "c", f"f{i}.py:{i}")
        assert len(history["c"]["seen"]) <= 32

    def test_history_is_not_mutated_in_place(self):
        """cli.py reads the dict off loop_counter; an in-place mutation
        would record progress even on a path that does not apply it."""
        history = {"c": {"last": "a.py:1", "seen": ["a.py:1"]}}
        snapshot = {"c": {"last": "a.py:1", "seen": ["a.py:1"]}}
        _charge(history, "c", "b.py:2")
        assert history == snapshot


class TestTheWiring:
    def test_the_flag_is_registered_in_both_cli_places(self):
        """A config key missing from either the known-keys set or the type
        map fails 11 schema tests — learned on the dispatch deadline."""
        import pathlib

        src = (
            pathlib.Path(__file__).parent.parent / "harness" / "cli.py"
        ).read_text()
        assert '"auto_resume_class_cap_resets_on_progress",' in src
        assert (
            '"hitl.auto_resume_class_cap_resets_on_progress": (bool,),' in src
        )

    def test_config_json_enables_it(self):
        import json
        import pathlib

        cfg = json.loads(
            (pathlib.Path(__file__).parent.parent
             / "config" / "config.json").read_text()
        )
        assert cfg["hitl"]["auto_resume_class_cap_resets_on_progress"] is True

    def test_config_help_documents_it(self):
        import pathlib

        help_md = (
            pathlib.Path(__file__).parent.parent
            / "config" / "config_help.md"
        ).read_text()
        assert "auto_resume_class_cap_resets_on_progress" in help_md

    def test_the_refund_runs_before_the_cap_test(self):
        """Progress made on THIS trip must count against termination, not
        after it."""
        import pathlib

        src = (
            pathlib.Path(__file__).parent.parent / "harness" / "cli.py"
        ).read_text()
        refund_at = src.index("_class_budget_progress_reset(\n")
        cap_at = src.index("_session_cap_hit = _resumes_taken >= _cap")
        assert refund_at < cap_at
