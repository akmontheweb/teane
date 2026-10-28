"""A rotating root cause must not buy more resumes than a stationary one.

The per-class auto-resume cap is meant to measure thrash on ONE problem. It has
now failed at that three times in three different directions, and this file
covers the third:

  * keyed on the raw trigger LABEL, so an embedded counter made every round a
    new trigger and the sub-cap never accumulated. Fixed by grouping on class.
  * counted lifetime firings, so RESOLVING five successive blockers still
    exhausted a class. Fixed by ``_class_budget_progress_reset`` (7bf5db8),
    proven live in run 10 — a class charged five times ended at 1/6.
  * **counted per CLASS, so one blocker presenting under three classes drew
    three budgets.** lumina run 5: a single blocker walked
    ``reflection_distraction_loop`` -> ``repair_loop_limit`` ->
    ``persistent_build_failure``, each staying around 1/6, so no class ever
    capped and the run spun until it was killed by hand. Run 10 charged
    ``_004:80`` to two classes.

``auto_resume_cap_per_blocker`` is the existing backstop, but at 9 against a
class cap of 6 it sits ABOVE the thing it backs up, so rotation still paid:
50% more resumes for presenting under a fresh label.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from harness.cli import _charged_trigger_class

BLOCKER_A = "server/app/birthdays.py:52"
BLOCKER_B = "server/app/dashboard.py:19"

ROTATION = (
    "reflection_distraction_loop",
    "repair_loop_limit",
    "persistent_build_failure",
)


def _charge(owners: dict, trigger_class: str, blocker: str, *, enabled=True):
    return _charged_trigger_class(
        trigger_class=trigger_class, blocker_sig=blocker,
        owners=owners, enabled=enabled,
    )


class TestTheRotationCostsNothing:
    def test_run_5s_three_classes_become_one(self):
        """The decisive case. Three hops, one budget."""
        owners: dict = {}
        charged = []
        for cls in ROTATION:
            got, owners = _charge(owners, cls, BLOCKER_A)
            charged.append(got)
        assert charged == [ROTATION[0]] * 3, (
            "each hop drew its own class budget, so no class ever capped"
        )

    def test_a_stationary_blocker_is_unaffected(self):
        owners: dict = {}
        for _ in range(3):
            got, owners = _charge(owners, "zero_patch_loop", BLOCKER_A)
            assert got == "zero_patch_loop"

    def test_rotation_and_stasis_now_cost_the_same(self):
        """The property worth stating outright: the accounting no longer
        cares how the blocker labels itself."""
        rot: dict = {}
        sta: dict = {}
        rot_charges, sta_charges = [], []
        for cls in ROTATION:
            got, rot = _charge(rot, cls, BLOCKER_A)
            rot_charges.append(got)
            got2, sta = _charge(sta, ROTATION[0], BLOCKER_A)
            sta_charges.append(got2)
        assert rot_charges == sta_charges

    def test_a_different_blocker_gets_its_own_owner(self):
        """A genuinely new problem is new work, not thrash — the session cap
        and the token budget are what bound a long tail of distinct
        problems."""
        owners: dict = {}
        a, owners = _charge(owners, "zero_patch_loop", BLOCKER_A)
        b, owners = _charge(owners, "repair_loop_limit", BLOCKER_B)
        assert a == "zero_patch_loop"
        assert b == "repair_loop_limit"

    def test_a_blocker_that_reappears_returns_to_its_owner(self):
        owners: dict = {}
        _a, owners = _charge(owners, "zero_patch_loop", BLOCKER_A)
        _b, owners = _charge(owners, "repair_loop_limit", BLOCKER_B)
        again, owners = _charge(owners, "persistent_build_failure", BLOCKER_A)
        assert again == "zero_patch_loop"

    def test_the_owner_map_is_returned_not_mutated_in_place(self):
        """It rides loop_counter through a checkpoint round-trip, so the
        caller owns when it is written back."""
        owners: dict = {}
        _got, out = _charge(owners, "zero_patch_loop", BLOCKER_A)
        assert owners == {}
        assert out == {BLOCKER_A: "zero_patch_loop"}


class TestTheEdges:
    def test_no_blocker_signature_charges_the_firing_class(self):
        """A trip with no diagnostics to key on has no blocker identity —
        per-blocker accounting is skipped for it, and so is this."""
        got, owners = _charged_trigger_class(
            trigger_class="zero_patch_loop", blocker_sig="",
            owners={}, enabled=True,
        )
        assert got == "zero_patch_loop"
        assert owners == {}

    def test_disabled_charges_the_firing_class(self):
        owners: dict = {BLOCKER_A: "zero_patch_loop"}
        got, out = _charge(owners, "repair_loop_limit", BLOCKER_A,
                           enabled=False)
        assert got == "repair_loop_limit"
        assert out == owners, "and it records nothing while off"

    def test_a_corrupt_owner_map_is_survivable(self):
        got, out = _charged_trigger_class(
            trigger_class="zero_patch_loop", blocker_sig=BLOCKER_A,
            owners=None, enabled=True,
        )
        assert got == "zero_patch_loop"
        assert out == {BLOCKER_A: "zero_patch_loop"}

    def test_an_empty_owner_entry_is_treated_as_unowned(self):
        got, out = _charge({BLOCKER_A: ""}, "repair_loop_limit", BLOCKER_A)
        assert got == "repair_loop_limit"
        assert out[BLOCKER_A] == "repair_loop_limit"


class TestTheWiring:
    def _src(self) -> str:
        return (
            pathlib.Path(__file__).parent.parent / "harness" / "cli.py"
        ).read_text()

    def test_the_cap_test_reads_the_charged_class(self):
        src = self._src()
        assert "_trigger_taken = int(_per_trigger.get(_charged_class, 0) or 0)" in src

    def test_the_increment_charges_the_charged_class(self):
        src = self._src()
        assert "_per_trigger[_charged_class] = int(" in src

    def test_the_refund_refunds_the_charged_class(self):
        """Otherwise the two halves of this counter's accounting would
        disagree about whose budget it is."""
        src = self._src()
        assert "trigger_class=_charged_class," in src
        assert "_per_trigger[_charged_class] = 0" in src

    def test_the_owner_map_survives_a_checkpoint(self):
        src = self._src()
        assert '_lc_for_cap["hitl_blocker_owner_class"] = _owners' in src

    def test_the_firing_class_is_still_reported(self):
        """A rotation must stay visible in the log and the termination
        banner — it just stops paying for itself."""
        src = self._src()
        assert "fired as '{_trigger_class}'" in src
        assert "_resumes_taken + 1, _cap, trigger, _charged_class," in src


class TestTheConfigKey:
    def test_it_is_registered_as_a_known_key(self):
        from harness.cli import _KNOWN_NESTED_KEYS

        assert "auto_resume_class_charged_to_first_owner" in (
            _KNOWN_NESTED_KEYS["hitl"]
        )

    def test_it_is_typed(self):
        from harness.cli import _TYPE_SCHEMA

        assert _TYPE_SCHEMA[
            "hitl.auto_resume_class_charged_to_first_owner"
        ] == (bool,)

    def test_the_shipped_config_enables_it(self):
        cfg = json.loads(
            (pathlib.Path(__file__).parent.parent / "config" / "config.json")
            .read_text()
        )
        assert cfg["hitl"]["auto_resume_class_charged_to_first_owner"] is True

    def test_it_is_documented(self):
        """House rule: every config change updates config_help.md."""
        doc = (
            pathlib.Path(__file__).parent.parent / "config" / "config_help.md"
        ).read_text()
        assert "auto_resume_class_charged_to_first_owner" in doc
        assert "hitl_blocker_owner_class" in doc

    @pytest.mark.parametrize("default", [True])
    def test_the_code_default_is_on(self, default):
        from harness.cli import _HITL_CLASS_CHARGED_TO_FIRST_OWNER

        assert _HITL_CLASS_CHARGED_TO_FIRST_OWNER is default
