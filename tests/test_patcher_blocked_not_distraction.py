"""A patcher-rejected round is a tooling failure, not a wrong direction.

lumina run 01a21db1 (2026-11-27) reached ``UNSATISFIABLE_TEST`` on two
CORRECT acceptance tests without a single component mis-reasoning:

``server/app/api/celebrations.py`` carried deliberate alias pairs --
``@router.post("/celebrations")`` stacked with
``@router.post("/birthdays", include_in_schema=False)`` on one function --
so every SEARCH anchor matched twice and the patcher refused to guess. Five
patch batches came back ``Applied 0/N`` (``ambiguous match`` x4,
``search miss`` x1). The reflection judge then scored each zero-byte round
DISTRACTION, *saying so in its own blocker text*: "The previous round landed
no patches (the only patch was REJECTED)". Three such verdicts saturated
``max_consecutive_distraction_rounds`` and forced the escape over:

  * the per-file production-fault veto, which refused 7 times (once:
    "accepting that would delete the requirement, not fix it"),
  * the harness's own ``triage_test_bug=False``,
  * the model declining the offer four times.

The repair direction had been right the entire time: the file genuinely had
no ``@router.get`` at all, and STORY-004/012/013's endpoints were unbuilt.
Damage was zero only because the tamper guard and ``no_failing_criterion``
each independently blocked the escape -- luck, not design.

These tests pin the classification: a round the PATCHER refused holds
``consecutive_distraction_rounds`` and ticks a bounded counter instead.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from harness.graph import (
    _prior_round_was_patcher_blocked,
    _summarize_round_patches,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class _FakeResult:
    """Minimal stand-in for ``harness.patcher.PatchResult``.

    ``_summarize_round_patches`` reads via ``getattr``, so the real dataclass
    is not needed -- and using a fake keeps this test from depending on the
    patcher's constructor signature.
    """

    def __init__(
        self,
        file: str,
        *,
        success: bool = False,
        no_op: bool = False,
        error: str = "",
        operation: str = "REPLACE_BLOCK",
    ) -> None:
        self.file = file
        self.success = success
        self.no_op = no_op
        self.error = error
        self.operation = operation


class TestTheRun01a21db1Shape:
    """The exact patch outcomes the run recorded must classify as blocked."""

    def test_the_ambiguous_match_round_is_patcher_blocked(self) -> None:
        # Verbatim shape of 21:42:05 and 21:50:40: "Applied 0/1 patches.
        # Other failures: server/app/api/celebrations.py (ambiguous match)."
        summary = _summarize_round_patches([
            _FakeResult(
                "server/app/api/celebrations.py",
                error="ambiguous match",
            ),
        ])
        assert summary == [{
            "file": "server/app/api/celebrations.py",
            "op": "REPLACE_BLOCK",
            "outcome": "REJECTED: ambiguous match",
        }]
        assert _prior_round_was_patcher_blocked(summary) is True

    def test_the_search_miss_round_is_also_patcher_blocked(self) -> None:
        # 21:52:01 -- the model widened the anchor to disambiguate and
        # overshot. Opposite failure mode, same tooling verdict.
        summary = _summarize_round_patches([
            _FakeResult(
                "server/app/api/celebrations.py",
                error="search miss",
            ),
        ])
        assert _prior_round_was_patcher_blocked(summary) is True

    def test_the_round_that_finally_landed_is_not_blocked(self) -> None:
        # 21:53:10 -- "Applied 1/1 patches", and @router.get went 0 -> 4.
        # This must clear the streak, not extend it.
        summary = _summarize_round_patches([
            _FakeResult("server/app/api/celebrations.py", success=True),
        ])
        assert summary[0]["outcome"] == "applied"
        assert _prior_round_was_patcher_blocked(summary) is False


class TestWhatMustNotCountAsPatcherBlocked:
    """The guard must not swallow the cases it is not about."""

    def test_a_round_with_no_patch_attempts_is_a_genuine_refusal(self) -> None:
        # The model emitted no patch blocks at all. That IS a wrong-direction
        # signal and must keep ticking the distraction counter.
        assert _prior_round_was_patcher_blocked([]) is False
        assert _prior_round_was_patcher_blocked(None) is False

    def test_a_partial_apply_is_not_blocked(self) -> None:
        # One landed, one rejected. Bytes changed, so the round is a real
        # attempt and the judge's verdict about it stands.
        summary = _summarize_round_patches([
            _FakeResult("server/app/api/celebrations.py", success=True),
            _FakeResult("server/app/main.py", error="ambiguous match"),
        ])
        assert _prior_round_was_patcher_blocked(summary) is False

    def test_a_noop_only_round_is_left_to_asserted_correct(self) -> None:
        # Byte-identical re-emission is the ASSERTED-CORRECT signal, which
        # has its own branch ("the model re-emitted byte-identical content
        # ... reading this as a claim that the file is already correct").
        # This guard must not mask it.
        summary = _summarize_round_patches([
            _FakeResult("server/app/main.py", success=True, no_op=True),
        ])
        assert summary[0]["outcome"] == "no-op (content already identical)"
        assert _prior_round_was_patcher_blocked(summary) is False

    def test_a_noop_plus_rejection_still_counts_as_blocked(self) -> None:
        # A no-op alongside a rejection: nothing applied and the patcher
        # refused, so the rejection is the operative signal.
        summary = _summarize_round_patches([
            _FakeResult("server/app/main.py", success=True, no_op=True),
            _FakeResult(
                "server/app/api/celebrations.py", error="ambiguous match",
            ),
        ])
        assert _prior_round_was_patcher_blocked(summary) is True

    def test_malformed_entries_do_not_crash_or_fire(self) -> None:
        # loop_counter survives checkpoint round-trips; defend the read.
        assert _prior_round_was_patcher_blocked([{}]) is False
        assert _prior_round_was_patcher_blocked(
            ["not-a-dict"],  # type: ignore[list-item]
        ) is False
        assert _prior_round_was_patcher_blocked(
            [{"file": "x.py", "outcome": ""}],
        ) is False


class TestTheBoundIsConfigDriven:
    """No hardcoded caps: the hold must be bounded by config."""

    def test_gateway_config_exposes_the_cap_with_a_default(self) -> None:
        from harness.gateway import GatewayConfig

        assert hasattr(GatewayConfig, "max_consecutive_patcher_blocked_rounds")

    def test_config_json_carries_the_key_under_node_throttle(self) -> None:
        cfg = json.loads((REPO_ROOT / "config" / "config.json").read_text())
        throttle = cfg["node_throttle"]
        assert "max_consecutive_patcher_blocked_rounds" in throttle
        assert isinstance(
            throttle["max_consecutive_patcher_blocked_rounds"], int,
        )

    def test_cli_accepts_and_type_checks_the_key(self) -> None:
        from harness import cli

        assert (
            "max_consecutive_patcher_blocked_rounds"
            in cli._KNOWN_NESTED_KEYS["node_throttle"]
        )
        assert cli._TYPE_SCHEMA[
            "node_throttle.max_consecutive_patcher_blocked_rounds"
        ] == (int,)

    def test_config_help_documents_the_key(self) -> None:
        # Standing rule: every config change updates config_help.md.
        help_text = (REPO_ROOT / "config" / "config_help.md").read_text()
        assert "max_consecutive_patcher_blocked_rounds" in help_text

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(0, 1), (-5, 1), (3, 3), (20, 20), (99, 20), ("nonsense", 3)],
    )
    def test_the_cap_is_clamped(self, raw: object, expected: int) -> None:
        from harness.gateway import create_gateway_from_config

        gw = create_gateway_from_config({
            "node_throttle": {
                "max_consecutive_patcher_blocked_rounds": raw,
            },
        })
        assert gw.config.max_consecutive_patcher_blocked_rounds == expected


class TestTheHoldBranchIsWired:
    """The branch must exist in repair_node's verdict chain, in order."""

    def test_the_hold_precedes_the_increment(self) -> None:
        # A regression here is silent: if the incrementing branch is
        # evaluated first, the hold can never be reached and run
        # 01a21db1 repeats. Assert source order.
        src = (REPO_ROOT / "harness" / "graph.py").read_text()
        hold = src.index("and _patcher_blocked_streak < _max_patcher_blocked")
        increment = src.index(
            'elif _v in {"DISTRACTION", "REGRESSION"} and not _low_signal:',
        )
        assert hold < increment, (
            "the patcher-blocked hold must be evaluated before the "
            "unconditional DISTRACTION increment"
        )

    def test_the_streak_resets_when_a_patch_lands(self) -> None:
        # Without a reset the streak leaks into an unrelated later stall and
        # the bound fires on a round that was never blocked.
        src = (REPO_ROOT / "harness" / "graph.py").read_text()
        assert 'loop_counter["consecutive_patcher_blocked_rounds"] = 0' in src
