"""Repair's in-turn correction loop (B2).

patching has fed each patch result back and re-dispatched inside one turn
since lumina 019f71b0 (``patching_emission_rounds``). repair never did: it
applied once, appended the composed feedback to ``messages`` and RETURNED,
so the model first learned its search string had not matched on the NEXT
repair round — after a full compiler cycle (build + prod-smoke; 1.2s warm
but 120-446s cold on lumina 01a164b2). A round that lands zero patches was
therefore the most expensive possible way to say "that did not apply".

Repair is ~77% of a run's LLM calls (profiled on 019ff418: 133 of 173), so
it was the half running the weaker protocol.

The loop accumulates into the same variable names the ~450 lines of
downstream bookkeeping already read, so that code sees ONE merged round —
which is what it is. It deliberately does not bump ``total_repairs``.
"""

import pytest

from harness.graph import (
    _repair_failures_are_policy_only,
    _resolve_repair_emission_rounds,
)
from harness.patcher import OperationType, PatchResult


def _result(success, error=None, no_op=False):
    return PatchResult(
        success=success, file="f.py", operation=OperationType.REPLACE_BLOCK,
        error=error, no_op=no_op,
    )


class TestTheCapIsConfigDriven:
    """No hardcoded caps: every budget/retry/timeout comes from config."""

    @pytest.mark.parametrize("cfg,expected", [
        ({}, 1),                                    # documented default
        ({"repair_emission_rounds": 0}, 0),         # explicitly disabled
        ({"repair_emission_rounds": 2}, 2),
        ({"repair_emission_rounds": 5}, 5),
        ({"repair_emission_rounds": 99}, 5),        # clamped up
        ({"repair_emission_rounds": -4}, 0),        # clamped down
        ({"repair_emission_rounds": "nonsense"}, 1),
        ({"repair_emission_rounds": None}, 1),
    ])
    def test_resolution_and_clamping(self, cfg, expected):
        state = {"llm_dispatch_config": cfg}
        assert _resolve_repair_emission_rounds(state) == expected

    def test_zero_disables_the_retry_entirely(self):
        """Operators must be able to restore pre-change behaviour."""
        assert _resolve_repair_emission_rounds(
            {"llm_dispatch_config": {"repair_emission_rounds": 0}}
        ) == 0

    def test_a_missing_config_section_still_resolves(self):
        assert _resolve_repair_emission_rounds({}) == 1

    def test_the_knob_is_registered_and_documented(self):
        """A config key the validator rejects, or that config_help.md does
        not describe, is a key operators cannot actually use."""
        import json
        import os

        import harness.graph as g

        root = os.path.dirname(os.path.dirname(os.path.abspath(g.__file__)))
        with open(os.path.join(root, "config", "config.json")) as fh:
            shipped = json.load(fh)
        assert "repair_emission_rounds" in shipped["llm_dispatch"]

        with open(os.path.join(root, "config", "config_help.md")) as fh:
            assert "repair_emission_rounds" in fh.read()

        from harness.cli import _TYPE_SCHEMA
        assert "llm_dispatch.repair_emission_rounds" in _TYPE_SCHEMA


class TestOnlyRetryWhatARetryCanFix:
    """A re-dispatch costs a full call. Spending it to be told the same
    thing is worse than closing the round."""

    def test_allowlist_rejections_do_not_retry(self):
        """The allowlist is fixed for the round — the model cannot widen
        it, so asking again produces the identical rejection."""
        assert _repair_failures_are_policy_only(
            [_result(False, "server/x.py not in skill allowlist")]
        )

    def test_test_protected_rejections_do_not_retry(self):
        """A protected test needs an UNSATISFIABLE_TEST declaration, which
        is a different action, not another patch."""
        assert _repair_failures_are_policy_only(
            [_result(False, "[test-protected] Editing test files is not")]
        )

    def test_off_scope_move_destination_does_not_retry(self):
        assert _repair_failures_are_policy_only(
            [_result(False, "move destination `x` is outside this round's "
                            "allowed paths")]
        )

    def test_operator_diff_rejection_does_not_retry(self):
        assert _repair_failures_are_policy_only(
            [_result(False, "rejected by operator via diff-approval gate")]
        )

    @pytest.mark.parametrize("error", [
        "Search text not found in file",
        "Post-patch validation failed: Undefined name(s): `re`",
        "You have not been shown the current bytes of `a.py`",
        "REPLACE_BLOCK no-op: your search and replace blocks are identical",
        "some message nobody has classified yet",
    ])
    def test_mechanically_correctable_failures_do_retry(self, error):
        """These are precisely what another attempt fixes — and the last
        case is the deliberate asymmetry: an UNRECOGNISED message reads as
        correctable, costing one extra call, rather than silently
        disabling the retry."""
        assert not _repair_failures_are_policy_only([_result(False, error)])

    def test_a_mix_still_retries(self):
        """One unfixable rejection must not suppress the retry that the
        OTHER, correctable failure deserves."""
        assert not _repair_failures_are_policy_only([
            _result(False, "not in skill allowlist"),
            _result(False, "Search text not found in file"),
        ])

    def test_no_failures_is_not_policy_only(self):
        assert not _repair_failures_are_policy_only([_result(True)])

    def test_nothing_parsed_is_not_policy_only(self):
        """An empty result set is the "model emitted nothing parseable"
        round — 157 of the 158 pre-change DSL emissions made zero tool
        calls, so it is the single most valuable case to retry."""
        assert not _repair_failures_are_policy_only([])


class TestNoOpsDoNotCountAsLanding:
    """The retry triggers on zero REAL successes. An idempotency no-op is
    the model re-emitting an already-applied patch; counting it as landing
    would mask exactly the stuck round this loop exists to break."""

    def test_a_no_op_is_not_a_real_success(self):
        results = [_result(True, no_op=True)]
        real = sum(1 for r in results if r.success and not r.no_op)
        assert real == 0

    def test_a_genuine_success_is(self):
        results = [_result(True)]
        real = sum(1 for r in results if r.success and not r.no_op)
        assert real == 1


class TestTheLoopIsWiredIntoRepair:
    def test_repair_node_consults_the_cap(self):
        import inspect

        import harness.graph as g

        src = inspect.getsource(g)
        assert "_resolve_repair_emission_rounds(state)" in src
        assert "_repair_failures_are_policy_only(patch_results)" in src

    @staticmethod
    def _retry_block() -> str:
        """The in-turn retry block's source, bounded by its own markers.

        Extracting it is fiddly enough to get wrong silently: the first
        version split on a 36-dash run, which matched inside the block's
        OWN header comment and captured a single line — so every assertion
        against it passed vacuously. Hence the length guard below: a
        source-scraping test must prove it captured something.
        """
        import inspect

        import harness.graph as g

        src = inspect.getsource(g)
        after = src.split("In-turn correction (B2)", 1)[1]
        # Skip the header line's own dashes, then stop at the closing rule.
        after = after.split("\n", 1)[1]
        block = after.split("# " + "-" * 60, 1)[0]
        assert len(block.splitlines()) > 50, (
            f"extracted only {len(block.splitlines())} line(s) — the block "
            f"markers drifted and this test would assert nothing"
        )
        return block

    def test_the_block_is_extractable(self):
        assert "while _inturn_cap > _inturn_round" in self._retry_block()

    def test_it_does_not_bump_the_repair_counter(self):
        """The retries merge into ONE repair iteration. Bumping
        total_repairs would make a single round consume the repair cap and
        would lie to the reflection judge about how many attempts ran."""
        code = [
            ln for ln in self._retry_block().splitlines()
            if not ln.strip().startswith("#")
        ]
        assert not any("total_repairs" in ln for ln in code)

    def test_every_exit_path_is_present(self):
        """An unbounded retry loop in the node that carries 77% of a run's
        calls would burn budget without limit."""
        block = self._retry_block()
        for guard in (
            "while _inturn_cap > _inturn_round",   # the cap
            "_inturn_round += 1",                  # always advances
            "if _rs:",                             # a real success ends it
            "_repair_failures_are_policy_only",    # nothing to retry
            "new_budget <= 0",                     # out of money
            "if not _retry_blocks:",               # nothing parseable
            "if not _retry_kept:",                 # nothing survived screening
        ):
            assert guard in block, f"missing exit guard: {guard}"

    def test_it_accumulates_rather_than_replaces(self):
        """~450 lines downstream read patch_results, modified_files,
        _blocks_kept and both rejection lists. The loop must merge into
        those names so that code sees one round — replacing them would
        drop the first pass's rejections from the feedback."""
        block = self._retry_block()
        assert "patch_results = list(patch_results) + list(_retry_results)" in block
        assert "_blocks_kept = list(_blocks_kept) + list(_retry_kept)" in block
        assert "_test_tamper_rejections = (" in block
        assert "_screen_rejections = (" in block

    def test_it_reuses_the_first_pass_guard_arguments(self):
        """Caught in review: omitting allow_parse_broken made a
        parse-broken test the first pass was ALLOWED to repair untouchable
        on the retry."""
        assert "allow_parse_broken=_parse_broken_tests" in self._retry_block()

    def test_it_tells_the_model_reads_are_not_resolved(self):
        """The retry strips READ_FILE without answering it, while the
        read-before-edit rejection in the feedback says to go and read.
        Without this line the two instructions contradict each other."""
        assert "file reads are NOT resolved here" in self._retry_block()

    def test_it_emits_telemetry(self):
        """Without an event, "did the retry help?" is unanswerable from a
        run log — the same gap that made the dual-dialect cost invisible."""
        import inspect

        import harness.graph as g

        assert '"repair_inturn_retry"' in inspect.getsource(g)


class TestTheRetryActuallyRuns:
    """The 33 tests above all asserted on SOURCE TEXT or exercised the
    helpers in isolation. None of them ever executed the retry block, so
    none could catch a runtime signature mismatch — and lumina 01a167cd,
    the first run in which a zero-landing round occurred, raised:

        TypeError: compose_patch_feedback() missing 3 required
        keyword-only arguments: 'success_count', 'fail_count',
        and 'no_op_count'

    A source-scanning test cannot see that. This one calls the real
    function with the real arguments the block passes.
    """

    def test_compose_patch_feedback_accepts_what_the_block_passes(self):
        from harness.patch_feedback import compose_patch_feedback
        from harness.patcher import OperationType, PatchResult

        results = [
            PatchResult(success=False, file="a.py",
                        operation=OperationType.REPLACE_BLOCK,
                        error="Search text not found in file"),
            PatchResult(success=True, file="b.py",
                        operation=OperationType.CREATE_FILE, no_op=True),
        ]
        success = sum(1 for r in results if r.success)
        no_op = sum(1 for r in results if r.success and r.no_op)
        msg, failures, rejections = compose_patch_feedback(
            results,
            ["a.py", "b.py"],
            "",
            prefix="[System]: Your patches did NOT land (in-turn retry 1/1):",
            success_count=success,
            fail_count=len(results) - success,
            no_op_count=no_op,
        )
        assert isinstance(msg, str) and msg
        assert "in-turn retry 1/1" in msg
        assert isinstance(failures, list) and isinstance(rejections, list)

    def test_the_blocks_call_supplies_every_required_kwarg(self):
        """Signature parity, checked against the live signature rather
        than against a copy of it — so a future kwarg added to
        compose_patch_feedback fails here instead of at runtime in a
        zero-landing repair round."""
        import inspect

        from harness.patch_feedback import compose_patch_feedback

        required = {
            name for name, p in
            inspect.signature(compose_patch_feedback).parameters.items()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
            and p.default is inspect.Parameter.empty
        }
        block = TestTheLoopIsWiredIntoRepair._retry_block()
        missing = [kw for kw in required if f"{kw}=" not in block]
        assert not missing, (
            f"the in-turn retry omits required keyword-only argument(s) "
            f"{missing} of compose_patch_feedback — this raises TypeError "
            f"the first time a repair round lands zero patches"
        )
