"""A correction round must not lose the work it did not touch.

Two defects, both in the bookkeeping around a re-prompt that writes files.

**One — the generated-test set was replaced by a per-call delta.**
``process_llm_patch_output`` returns ``existing_modified`` plus the files
*that call* applied; it does not accumulate, because every correction round
passes the same pre-node baseline. Both correction loops in
``test_generation_node`` recomputed the generated-test set from that return,
so:

  * a correction that rewrote ONE of three generated tests dropped the other
    two from the set, and
  * a correction that landed NOTHING -- the patcher rejects a byte-identical
    re-emission as a no-op, which is the likeliest outcome -- dropped all of
    them.

Marker persistence and the runner's test selection key off that list, so
either way tests that were generated perfectly well vanished from the build
with nothing in the log. Found by executing the spec gate rather than
grepping it; the contradiction gate beside it had the same defect, and both
now share :func:`_merge_corrected_tests`.

**Two -- the unit regeneration path re-asked an identical question.**
The router mints a regeneration request whenever repair is blocked needing a
test edit, so a file that is still red comes back even when the previous
attempt succeeded. The acceptance path has refused that since 01a17661; the
unit path only had a prompt note. Run 10 took ``test_main.py`` from
regeneration 1/2 to 2/2 in TEN SECONDS, both attempts writing the same bytes,
with the note logged each time -- shown, ignored, and paid for twice. Both
paths now share :func:`_regen_request_repeated`.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from harness.test_generation import _merge_corrected_tests
from harness.test_regeneration import (
    _REGEN_ATTEMPTS_KEY,
    _record_regen_request,
    _regen_request_repeated,
)


# ---------------------------------------------------------------------------
# One: the merge
# ---------------------------------------------------------------------------

class TestTheMergeKeepsWhatTheRoundDidNotTouch:
    """Executes the real helper. Every case here is a way the old
    delta-assignment lost a file."""

    @staticmethod
    def _ws(tmp_path, *rels: str) -> str:
        for rel in rels:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# test\n")
        return str(tmp_path)

    def test_a_correction_that_landed_nothing_keeps_the_whole_set(
        self, tmp_path,
    ):
        """The original finding. ``call_modified == existing_modified`` is
        exactly what the patcher returns for a no-op round."""
        ws = self._ws(tmp_path, "tests/a.py", "tests/b.py")
        merged, written = _merge_corrected_tests(
            ["tests/a.py", "tests/b.py"],
            ["src/app.py"],
            existing_modified=["src/app.py"],
            workspace_path=ws,
        )
        assert merged == ["tests/a.py", "tests/b.py"]
        assert written == []

    def test_a_correction_touching_one_file_keeps_the_other_two(
        self, tmp_path,
    ):
        """The stronger half, and the one a no-op test would have missed:
        the set was wrong even when the correction SUCCEEDED."""
        ws = self._ws(tmp_path, "tests/a.py", "tests/b.py", "tests/c.py")
        merged, written = _merge_corrected_tests(
            ["tests/a.py", "tests/b.py", "tests/c.py"],
            ["src/app.py", "tests/b.py"],
            existing_modified=["src/app.py"],
            workspace_path=ws,
        )
        assert merged == ["tests/a.py", "tests/b.py", "tests/c.py"]
        assert written == ["tests/b.py"]

    def test_a_brand_new_corrected_file_joins_the_set(self, tmp_path):
        ws = self._ws(tmp_path, "tests/a.py", "tests/new.py")
        merged, written = _merge_corrected_tests(
            ["tests/a.py"],
            ["tests/new.py"],
            existing_modified=[],
            workspace_path=ws,
        )
        assert merged == ["tests/a.py", "tests/new.py"]
        assert written == ["tests/new.py"]

    def test_a_file_the_correction_deleted_leaves_the_set(self, tmp_path):
        """DELETE_FILE and MOVE_FILE are real operations, and the
        contradiction re-prompt explicitly asks the author to REMOVE the
        losing test. A path the runner cannot open is worse than one it
        never knew about, so the union is filtered by what is on disk."""
        ws = self._ws(tmp_path, "tests/a.py")
        merged, _ = _merge_corrected_tests(
            ["tests/a.py", "tests/gone.py"],
            ["tests/a.py"],
            existing_modified=[],
            workspace_path=ws,
        )
        assert merged == ["tests/a.py"]

    def test_baseline_files_never_enter_the_set(self, tmp_path):
        """``existing_modified`` is production code patched before this node
        ran; it is not a generated test."""
        ws = self._ws(tmp_path, "tests/a.py", "src/app.py")
        merged, written = _merge_corrected_tests(
            ["tests/a.py"],
            ["src/app.py", "tests/a.py"],
            existing_modified=["src/app.py"],
            workspace_path=ws,
        )
        assert merged == ["tests/a.py"]
        assert written == ["tests/a.py"]

    def test_a_path_outside_the_workspace_is_excluded(self, tmp_path):
        """Defence in depth behind the patcher's own resolve guard."""
        ws = self._ws(tmp_path, "tests/a.py")
        merged, written = _merge_corrected_tests(
            ["tests/a.py"],
            ["/etc/passwd", "../outside.py"],
            existing_modified=[],
            workspace_path=ws,
        )
        assert merged == ["tests/a.py"]
        assert written == []

    def test_a_rewrite_of_a_file_already_in_the_set_does_not_duplicate_it(
        self, tmp_path,
    ):
        ws = self._ws(tmp_path, "tests/a.py")
        merged, _ = _merge_corrected_tests(
            ["tests/a.py"],
            ["tests/a.py"],
            existing_modified=[],
            workspace_path=ws,
        )
        assert merged == ["tests/a.py"]

    def test_it_tolerates_empty_and_none_inputs(self, tmp_path):
        ws = self._ws(tmp_path)
        assert _merge_corrected_tests(
            [], None, existing_modified=None, workspace_path=ws,
        ) == ([], [])


# ---------------------------------------------------------------------------
# One, continued: the gate that had the defect, executed
# ---------------------------------------------------------------------------

#: A proven cross-file pair — ContactUpdate(first_name="   ") required to
#: raise in one file and to succeed in the other. Verbatim shape from
#: tests/test_contradiction.py, which is what find_contradictions_across
#: proves.
SCHEMAS_FILE = '''# @tests: server/app/schemas/contact.py
import pytest
from pydantic import ValidationError
from server.app.schemas.contact import ContactUpdate


class TestContactUpdate:
    def test_empty_first_name_explicitly_sent_is_rejected(self):
        with pytest.raises(ValidationError, match="First name is required"):
            ContactUpdate(first_name="   ")
'''

SERVICE_FILE = '''# @tests: server/app/services/contact_service.py
import pytest
from fastapi import HTTPException
from server.app.schemas.contact import ContactUpdate
from server.app.services.contact_service import update_contact


class TestUpdateContact:
    def test_update_with_empty_first_name_raises_422(self, db_session):
        update_payload = ContactUpdate(first_name="   ")
        with pytest.raises(HTTPException) as exc_info:
            update_contact(db_session, 1, update_payload)
        assert exc_info.value.status_code == 422
'''

#: The reconciliation the re-prompt asks for: the service test stops
#: constructing the invalid value, so the pair is satisfiable again.
SERVICE_FIXED = '''# @tests: server/app/services/contact_service.py
import pytest
from fastapi import HTTPException
from server.app.schemas.contact import ContactUpdate
from server.app.services.contact_service import update_contact


class TestUpdateContact:
    def test_update_with_a_valid_first_name_succeeds(self, db_session):
        update_payload = ContactUpdate(first_name="New")
        assert update_contact(db_session, 1, update_payload) is not None
'''

SCHEMAS_REL = "server/tests/test_contact_schemas.py"
SERVICE_REL = "server/tests/test_contact_service.py"


def _rewrite(rel: str, body: str) -> str:
    """A real REWRITE_FILE block, per harness.patcher._BLOCK_PATTERNS."""
    return (
        "<<<REWRITE_FILE>>>\n"
        f"file: {rel}\n"
        "content:\n"
        + body +
        "<<<END_REWRITE_FILE>>>\n"
    )


class _GW:
    """Scripts the author's correction turns."""

    class config:
        repair_fallback = ""
        planning_fallback = ""

    def __init__(self, scripted: list[str]):
        self._scripted = list(scripted)
        self.roles: list[str] = []

    def select_model(self, _role):
        return "stub:model"

    async def dispatch(self, *, messages, role, budget_remaining_usd, **kw):
        self.roles.append(getattr(role, "value", str(role)))
        content = self._scripted.pop(0) if self._scripted else ""

        class _R:
            def __init__(self, c):
                self.content = c

                class _U:
                    cost_usd = 0.001
                    input_tokens = 10
                    output_tokens = 10
                    cached_tokens = 0
                    model = "stub"
                self.usage = _U()

        return _R(content), budget_remaining_usd - 0.001

    def aggregate_tokens(self, tracker, usage, role=None):
        out = dict(tracker or {})
        out["total_cost_usd"] = out.get("total_cost_usd", 0.0) + 0.001
        return out


class TestTheContradictionGateActuallyRuns:
    """The sibling loop, now extracted and therefore drivable.

    While it was inline nothing executed it, and it carried the merge defect
    for as long as that was true.
    """

    def _workspace(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "server" / "tests").mkdir(parents=True)
        (ws / SCHEMAS_REL).write_text(SCHEMAS_FILE)
        (ws / SERVICE_REL).write_text(SERVICE_FILE)
        return ws

    async def _run(self, tmp_path, scripted, *, cfg=None, tests=None):
        import harness.test_generation as tg

        ws = self._workspace(tmp_path)
        gw = _GW(scripted)
        out = await tg._run_contradiction_gate(
            gateway=gw,
            workspace_path=str(ws),
            generated_tests=(
                tests if tests is not None else [SCHEMAS_REL, SERVICE_REL]
            ),
            messages=[{"role": "user", "content": "generate tests"}],
            existing_modified=[],
            allowed_paths=None,
            budget=10.0,
            token_tracker={},
            cfg=cfg if cfg is not None else {},
        )
        return ws, gw, out

    @pytest.mark.asyncio
    async def test_a_proven_pair_is_reconciled_on_disk(self, tmp_path):
        """End to end: the pair is detected, the author is re-prompted, and
        the corrected bytes land before any build runs."""
        ws, gw, out = await self._run(
            tmp_path, [_rewrite(SERVICE_REL, SERVICE_FIXED)],
        )
        on_disk = (ws / SERVICE_REL).read_text()
        assert "test_update_with_a_valid_first_name_succeeds" in on_disk
        assert out["reprompts"] == 1
        assert gw.roles == ["patching"], (
            "one re-prompt reconciled it; the loop must not ask again"
        )

    @pytest.mark.asyncio
    async def test_the_file_the_correction_did_not_touch_stays_in_the_set(
        self, tmp_path,
    ):
        """THE defect, on the success path. The correction rewrote the
        service test; the schemas test was fine and must survive."""
        _, _, out = await self._run(
            tmp_path, [_rewrite(SERVICE_REL, SERVICE_FIXED)],
        )
        assert sorted(out["generated_tests"]) == sorted(
            [SCHEMAS_REL, SERVICE_REL]
        ), "a correction dropped the generated test it did not touch"

    @pytest.mark.asyncio
    async def test_a_reprompt_that_lands_nothing_keeps_the_set_and_stops(
        self, tmp_path,
    ):
        """THE defect, on the no-op path. A byte-identical re-emission is
        rejected by the patcher, so nothing is written — and nothing has
        changed for a second round to read."""
        _, gw, out = await self._run(
            tmp_path, [_rewrite(SERVICE_REL, SERVICE_FILE),
                       _rewrite(SERVICE_REL, SERVICE_FILE)],
            cfg={"max_contradiction_reprompts": 2},
        )
        assert sorted(out["generated_tests"]) == sorted(
            [SCHEMAS_REL, SERVICE_REL]
        ), "the generated-test set was wiped by a no-op re-prompt"
        assert gw.roles.count("patching") == 1, (
            "a re-prompt that wrote nothing will not write anything next "
            "round either — it must stop, not retry to the cap"
        )

    @pytest.mark.asyncio
    async def test_what_a_reprompt_wrote_is_reported_for_modified_files(
        self, tmp_path,
    ):
        """The node reports ``applied`` as part of ``modified_files``. A
        corrected file missing from it is written to disk and invisible to
        everything downstream."""
        _, _, out = await self._run(
            tmp_path, [_rewrite(SERVICE_REL, SERVICE_FIXED)],
        )
        assert out["applied"] == [SERVICE_REL]

    @pytest.mark.asyncio
    async def test_consistent_tests_cost_no_dispatch(self, tmp_path):
        _, gw, out = await self._run(tmp_path, [], tests=[SCHEMAS_REL])
        assert gw.roles == []
        assert out["reprompts"] == 0

    @pytest.mark.asyncio
    async def test_the_cap_bounds_a_model_that_cannot_reconcile(
        self, tmp_path,
    ):
        """Each round writes DIFFERENT still-contradictory content, because
        an identical re-emission exits on the no-write path above."""
        wrongs = [
            SERVICE_FILE.replace("422", f"42{i}") for i in range(3)
        ]
        _, gw, out = await self._run(
            tmp_path,
            [_rewrite(SERVICE_REL, w) for w in wrongs],
            cfg={"max_contradiction_reprompts": 2},
        )
        assert out["reprompts"] == 2
        assert gw.roles.count("patching") == 2

    @pytest.mark.asyncio
    async def test_a_zero_cap_disables_it(self, tmp_path):
        _, gw, out = await self._run(
            tmp_path, [], cfg={"max_contradiction_reprompts": 0},
        )
        assert gw.roles == []
        assert out["reprompts"] == 0

    @pytest.mark.asyncio
    async def test_the_reprompt_is_billed(self, tmp_path):
        _, _, out = await self._run(
            tmp_path, [_rewrite(SERVICE_REL, SERVICE_FIXED)],
        )
        assert out["budget"] < 10.0
        assert out["token_tracker"]["total_cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_a_gateway_refusal_does_not_block_the_build(self, tmp_path):
        import harness.test_generation as tg

        class _Refuses(_GW):
            async def dispatch(self, **_kw):
                raise RuntimeError("budget exhausted")

        ws = self._workspace(tmp_path)
        out = await tg._run_contradiction_gate(
            gateway=_Refuses([]),
            workspace_path=str(ws),
            generated_tests=[SCHEMAS_REL, SERVICE_REL],
            messages=[{"role": "user", "content": "go"}],
            existing_modified=[], allowed_paths=None,
            budget=10.0, token_tracker={}, cfg={},
        )
        assert sorted(out["generated_tests"]) == sorted(
            [SCHEMAS_REL, SERVICE_REL]
        )


class TestTheGatesSpendReachesTheNodesReport:
    """A leak found reviewing the extraction, not by a run.

    Every dispatch in ``test_generation_node`` threads ``new_budget`` and
    then copies it into ``budget``; both gates are the other way round --
    they take ``budget`` and return the remainder. So the node has to copy
    back, or the spend vanishes: each of its own returns, and the
    acceptance/build gate it feeds further down, carries ``new_budget``, and
    the graph would go on believing money already spent was still available.
    Run 6 died at $8.02 of $10, so an inflated remainder is not harmless.

    Source-order rather than executed, which is the honest description:
    nothing drives this node end to end -- it wants a sandbox, a gateway, a
    marker-bearing workspace and a dozen branches of config. The gates'
    own returns ARE executed, above and in test_generated_test_selfcheck.
    """

    @staticmethod
    def _node_src() -> str:
        import inspect

        import harness.test_generation as tg

        return inspect.getsource(tg.test_generation_node)

    def test_the_spend_is_resynced_after_both_gates(self):
        src = self._node_src()
        contra = src.index("_run_contradiction_gate(")
        spec = src.index("_run_spec_consistency_gate(")
        resync = src.index("\n    new_budget = budget\n", spec)
        assert contra < spec < resync

    def test_nothing_reports_the_budget_between_the_gates_and_the_resync(self):
        """A return in that window would carry the stale remainder."""
        src = self._node_src()
        contra = src.index("_run_contradiction_gate(")
        resync = src.index("\n    new_budget = budget\n", contra)
        assert '"budget_remaining_usd"' not in src[contra:resync]


# ---------------------------------------------------------------------------
# Two: the unit-path repeat guard
# ---------------------------------------------------------------------------

class TestTheSharedRepeatGuard:
    REL = "server/tests/test_main.py"

    def test_an_answered_request_asked_again_is_a_repeat(self):
        lc: dict = {}
        assert not _regen_request_repeated(lc, self.REL, "sig-a")
        _record_regen_request(lc, self.REL, "sig-a")
        assert _regen_request_repeated(lc, self.REL, "sig-a")

    def test_a_new_signature_proceeds(self):
        lc: dict = {}
        _record_regen_request(lc, self.REL, "sig-a")
        assert not _regen_request_repeated(lc, self.REL, "sig-b")

    def test_the_check_records_nothing_on_its_own(self):
        """The split that keeps a round which never got an answer from
        posing as one that did."""
        lc: dict = {}
        assert not _regen_request_repeated(lc, self.REL, "sig-a")
        assert not _regen_request_repeated(lc, self.REL, "sig-a")

    def test_a_refused_round_does_not_overwrite_the_signature(self):
        """Otherwise the third round gets through, because the second would
        have replaced what the first stored."""
        lc: dict = {}
        _record_regen_request(lc, self.REL, "sig-a")
        assert _regen_request_repeated(lc, self.REL, "sig-a")
        assert _regen_request_repeated(lc, self.REL, "sig-a")

    def test_files_are_tracked_independently(self):
        lc: dict = {}
        _record_regen_request(lc, "a.py", "sig")
        assert not _regen_request_repeated(lc, "b.py", "sig")

    def test_a_corrupt_store_is_survivable(self):
        lc: dict = {"test_regen_last_signature": "not a dict"}
        assert not _regen_request_repeated(lc, self.REL, "sig")
        _record_regen_request(lc, self.REL, "sig")  # must not raise


MODULE = "app/birthdays.py"
MODULE_SRC = "def upcoming(days, today):\n    return []\n"

UNIT_TEST = f"""# @tests: {MODULE}
from app.birthdays import upcoming


def test_upcoming_counts_the_window():
    assert len(upcoming(30, "2024-01-01")) == 2
"""

UNIT_REL = "tests/test_birthdays.py"


class TestTheUnitPathRefusesAnIdenticalRequest:
    """Drives the shipped node. Run 10's shape: regeneration 1/2 then 2/2
    ten seconds later, both writing the same bytes."""

    @staticmethod
    def _emission(body: str) -> str:
        return _rewrite(UNIT_REL, body)

    @staticmethod
    def _ws(ws: str, body: str = UNIT_TEST) -> None:
        os.makedirs(os.path.join(ws, "tests"), exist_ok=True)
        os.makedirs(os.path.join(ws, "app"), exist_ok=True)
        with open(os.path.join(ws, MODULE), "w") as fh:
            fh.write(MODULE_SRC)
        with open(os.path.join(ws, UNIT_REL), "w") as fh:
            fh.write(body)

    @staticmethod
    def _state(ws: str, *, reason: str = "two tests assert opposite counts",
               errors=None) -> dict:
        return {
            "workspace_path": ws,
            "node_state": {
                "unsatisfiable_test": UNIT_REL,
                "unsatisfiable_test_reason": reason,
            },
            "loop_counter": {},
            "test_regeneration_config": {
                "enabled": True, "max_attempts_per_test": 2,
                "require_code_linkage": True, "coverage_nonregression": False,
            },
            "messages": [{"role": "system", "content": "SRS spec"}],
            "budget_remaining_usd": 5.0,
            "compiler_errors": errors if errors is not None else [],
            "modified_files": [],
        }

    class _Queue:
        def __init__(self, contents):
            self._contents = list(contents)
            self.calls: list = []

        def select_model(self, _role):
            return "stub:model"

        async def dispatch(self, *, messages, **kw):
            self.calls.append(messages)
            content = self._contents.pop(0) if self._contents else ""

            class _R:
                def __init__(self, c):
                    self.content = c
                    self.finish_reason = "stop"
                    self.reasoning_content = ""

                    class _U:
                        cost_usd = 0.001
                        input_tokens = 10
                        output_tokens = 10
                        cached_tokens = 0
                        model = "stub"
                    self.usage = _U()

            return _R(content), kw.get("budget_remaining_usd", 5.0) - 0.001

        def aggregate_tokens(self, tracker, usage, role=None):
            return dict(tracker or {})

    @pytest.mark.asyncio
    async def test_an_unchanged_request_is_refused_before_dispatch(
        self, monkeypatch,
    ):
        """The file on disk, the declared reason and the failure are all
        what they were — so the author has already answered this, and its
        answer is the file being shown back to it."""
        import harness.graph as g

        with tempfile.TemporaryDirectory() as ws:
            self._ws(ws)
            # Round 1 emits a byte-identical file, so the patcher no-ops and
            # disk is unchanged going into round 2.
            first_gw = self._Queue([self._emission(UNIT_TEST)])
            monkeypatch.setattr(g, "get_gateway", lambda: first_gw)
            first = await regeneration_node_for_tests()(self._state(ws))

            second_gw = self._Queue([self._emission(UNIT_TEST)])
            monkeypatch.setattr(g, "get_gateway", lambda: second_gw)
            state2 = self._state(ws)
            state2["loop_counter"] = first["loop_counter"]
            out = await regeneration_node_for_tests()(state2)

        assert second_gw.calls == [], (
            "the second round re-asked an identical question at full price"
        )
        assert out["node_state"]["test_regeneration"]["status"] == (
            "unit_regen_repeat"
        )

    @pytest.mark.asyncio
    async def test_the_refused_attempt_is_given_back(self, monkeypatch):
        """Without the refund the file is a attempt poorer for a round that
        never happened, and ``max_attempts_per_test`` (2) routes it to HITL
        with one real attempt spent."""
        import harness.graph as g

        with tempfile.TemporaryDirectory() as ws:
            self._ws(ws)
            gw1 = self._Queue([self._emission(UNIT_TEST)])
            monkeypatch.setattr(g, "get_gateway", lambda: gw1)
            first = await regeneration_node_for_tests()(self._state(ws))
            charged = first["loop_counter"][_REGEN_ATTEMPTS_KEY][UNIT_REL]

            gw2 = self._Queue([self._emission(UNIT_TEST)])
            monkeypatch.setattr(g, "get_gateway", lambda: gw2)
            state2 = self._state(ws)
            state2["loop_counter"] = first["loop_counter"]
            out = await regeneration_node_for_tests()(state2)

        assert out["loop_counter"][_REGEN_ATTEMPTS_KEY][UNIT_REL] == charged, (
            "the refused round must cost nothing"
        )

    @pytest.mark.asyncio
    async def test_repeated_refusals_still_converge(self, monkeypatch):
        """The refusal makes no LLM call, so if it also refunded every time
        it would be free and unbounded — router -> regeneration(refused) ->
        compiler, forever. This file has run that pathology before: 813
        cycles in 68 minutes on lumina-verify-20260914-1151, with spend flat
        because budget is the usual backstop for a stuck graph.

        ``_refund_attempt`` is bounded to one reprieve per FILE and the
        refusal shares that budget, so the attempt counter climbs and the
        router's cap converges it to HITL.
        """
        import harness.graph as g

        with tempfile.TemporaryDirectory() as ws:
            self._ws(ws)
            gw = self._Queue([self._emission(UNIT_TEST)])
            monkeypatch.setattr(g, "get_gateway", lambda: gw)
            out = await regeneration_node_for_tests()(self._state(ws))

            counts = []
            for _ in range(4):
                state_n = self._state(ws)
                state_n["loop_counter"] = out["loop_counter"]
                monkeypatch.setattr(g, "get_gateway", lambda: self._Queue([]))
                out = await regeneration_node_for_tests()(state_n)
                counts.append(
                    out["loop_counter"][_REGEN_ATTEMPTS_KEY][UNIT_REL]
                )

        assert counts == sorted(counts), f"attempts must not fall: {counts}"
        assert counts[-1] >= 2, (
            f"the refusal never reached the attempt cap: {counts}"
        )

    @pytest.mark.asyncio
    async def test_a_round_that_got_no_answer_may_ask_again(self, monkeypatch):
        """Run 7 died on a 402 Payment Required. A provider failure says
        nothing about the question, so it must not stand as the file's last
        request and get the retry refused as a repeat — one external fault
        would otherwise turn into a dead story."""
        import harness.graph as g

        with tempfile.TemporaryDirectory() as ws:
            self._ws(ws)
            # Both dispatches of round 1 come back with no parseable block,
            # so the round ends at ``no_patch`` having written nothing.
            gw1 = self._Queue(["", ""])
            monkeypatch.setattr(g, "get_gateway", lambda: gw1)
            first = await regeneration_node_for_tests()(self._state(ws))
            assert first["node_state"]["test_regeneration"]["status"] == (
                "no_patch"
            )

            gw2 = self._Queue([self._emission(UNIT_TEST)])
            monkeypatch.setattr(g, "get_gateway", lambda: gw2)
            state2 = self._state(ws)
            state2["loop_counter"] = first["loop_counter"]
            await regeneration_node_for_tests()(state2)

        assert len(gw2.calls) == 1, (
            "a round that never got an answer poisoned the retry"
        )

    @pytest.mark.asyncio
    async def test_a_changed_file_still_gets_its_round(self, monkeypatch):
        """The productive case the note exists for: round 1 wrote something
        new, so round 2 has a different file to reason about and must not be
        refused."""
        import harness.graph as g

        changed = UNIT_TEST.replace("== 2", "== 1")
        with tempfile.TemporaryDirectory() as ws:
            self._ws(ws)
            gw1 = self._Queue([self._emission(changed)])
            monkeypatch.setattr(g, "get_gateway", lambda: gw1)
            first = await regeneration_node_for_tests()(self._state(ws))

            gw2 = self._Queue([self._emission(UNIT_TEST)])
            monkeypatch.setattr(g, "get_gateway", lambda: gw2)
            state2 = self._state(ws)
            state2["loop_counter"] = first["loop_counter"]
            await regeneration_node_for_tests()(state2)

        assert len(gw2.calls) == 1, (
            "a file that changed since the last attempt is a new question"
        )

    @pytest.mark.asyncio
    async def test_a_new_failure_still_gets_its_round(self, monkeypatch):
        """Repair moved production, so the same file now fails differently —
        there IS something new to work from, and byte-identity of the file
        alone would have refused it."""
        import harness.graph as g

        with tempfile.TemporaryDirectory() as ws:
            self._ws(ws)
            gw1 = self._Queue([self._emission(UNIT_TEST)])
            monkeypatch.setattr(g, "get_gateway", lambda: gw1)
            first = await regeneration_node_for_tests()(
                self._state(ws, errors=[{
                    "file": UNIT_REL, "line": 5,
                    "message": "AssertionError: assert 0 == 2",
                }])
            )

            gw2 = self._Queue([self._emission(UNIT_TEST)])
            monkeypatch.setattr(g, "get_gateway", lambda: gw2)
            state2 = self._state(ws, errors=[{
                "file": UNIT_REL, "line": 5,
                "message": "TypeError: upcoming() missing 1 required argument",
            }])
            state2["loop_counter"] = first["loop_counter"]
            await regeneration_node_for_tests()(state2)

        assert len(gw2.calls) == 1

    @pytest.mark.asyncio
    async def test_a_new_declared_reason_still_gets_its_round(
        self, monkeypatch,
    ):
        import harness.graph as g

        with tempfile.TemporaryDirectory() as ws:
            self._ws(ws)
            gw1 = self._Queue([self._emission(UNIT_TEST)])
            monkeypatch.setattr(g, "get_gateway", lambda: gw1)
            first = await regeneration_node_for_tests()(self._state(ws))

            gw2 = self._Queue([self._emission(UNIT_TEST)])
            monkeypatch.setattr(g, "get_gateway", lambda: gw2)
            state2 = self._state(
                ws, reason="the spec's ASM-009 fallback is not honoured",
            )
            state2["loop_counter"] = first["loop_counter"]
            await regeneration_node_for_tests()(state2)

        assert len(gw2.calls) == 1


def regeneration_node_for_tests():
    """Late import so the module imports cleanly under collection."""
    from harness.test_regeneration import test_regeneration_node

    return test_regeneration_node
