"""ADR-0005 item #2 — pre-repair triage gate in test_generation_node.

The gate diverts unambiguous test-authoring bugs to proactive regeneration and
re-runs BEFORE the repair loop, so repair only sees code-gaps. Exercised
directly via ``_run_pre_repair_triage_gate`` with injected fakes (no sandbox).
"""

from __future__ import annotations

import pytest

from harness.sandbox import DiagnosticObject
import harness.test_generation as tg
import harness.test_regeneration as trn


class _BR:
    """Minimal BuildResult stand-in."""
    def __init__(self, exit_code, diagnostics):
        self.exit_code = exit_code
        self.diagnostics = diagnostics
        self.raw_output = ""


class _Executor:
    def __init__(self, rerun_result):
        self._rerun = rerun_result
        self.runs = 0

    async def run(self, cmd):
        self.runs += 1
        return self._rerun


_ROW_BUG = DiagnosticObject(
    file="tests/unit/test_database.py", error_code="TypeError",
    message="tuple indices must be integers or slices, not str",
)
_CODE_GAP = DiagnosticObject(
    file="server/app/database.py", error_code="TypeError",
    message="'coroutine' object does not support the async context manager protocol",
)


@pytest.mark.asyncio
async def test_gate_regenerates_test_bug_and_reruns(monkeypatch):
    build_result = _BR(1, [_ROW_BUG, _CODE_GAP])
    green = _BR(0, [])
    executor = _Executor(green)

    seen_targets = []

    async def _fake_regen(state):
        rel = state["node_state"]["unsatisfiable_test"]
        seen_targets.append(rel)
        return {
            "modified_files": [rel],
            "token_tracker": state.get("token_tracker", {}),
            "budget_remaining_usd": state.get("budget_remaining_usd", 1.0),
            "loop_counter": state.get("loop_counter", {}),
            "node_state": {"test_regeneration": {"status": "regenerated"}},
        }

    monkeypatch.setattr(trn, "test_regeneration_node", _fake_regen)

    lc: dict = {}
    res = await tg._run_pre_repair_triage_gate(
        state={"test_regeneration_config": {}, "messages": []},
        build_result=build_result, executor=executor, test_cmd="pytest",
        workspace_path="/ws", loop_counter=lc, token_tracker={}, budget=1.0,
        max_regens=3,
    )
    assert res is not None
    new_build, _tok, _budget, extra_modified, summary = res
    # Only the test-bug file was regenerated — NOT the code-gap source file.
    assert seen_targets == ["tests/unit/test_database.py"]
    assert extra_modified == ["tests/unit/test_database.py"]
    assert new_build.exit_code == 0        # re-run picked up the rewrite
    assert executor.runs == 1              # re-ran exactly once
    assert summary["regenerated"] == 1
    assert lc["triage_gate_regenerated"] == 1


@pytest.mark.asyncio
async def test_gate_noop_when_no_test_bug():
    # Only a code-gap / behaviour assertion — nothing to divert.
    diags = [DiagnosticObject(
        file="tests/unit/test_x.py", error_code="AssertionError",
        message="assert 1 == 2",
    )]
    executor = _Executor(_BR(0, []))
    res = await tg._run_pre_repair_triage_gate(
        state={}, build_result=_BR(1, diags), executor=executor,
        test_cmd="pytest", workspace_path="/ws", loop_counter={},
        token_tracker={}, budget=1.0, max_regens=3,
    )
    assert res is None            # repair handles it unchanged
    assert executor.runs == 0     # no wasted re-run


@pytest.mark.asyncio
async def test_gate_noop_when_regeneration_writes_nothing(monkeypatch):
    async def _fake_regen(state):
        return {
            "modified_files": [],
            "node_state": {"test_regeneration": {"status": "no_patch"}},
        }

    monkeypatch.setattr(trn, "test_regeneration_node", _fake_regen)
    executor = _Executor(_BR(0, []))
    res = await tg._run_pre_repair_triage_gate(
        state={"test_regeneration_config": {}, "messages": []},
        build_result=_BR(1, [_ROW_BUG]), executor=executor, test_cmd="pytest",
        workspace_path="/ws", loop_counter={}, token_tracker={}, budget=1.0,
        max_regens=3,
    )
    assert res is None            # nothing rewritten → no-op
    assert executor.runs == 0     # did NOT re-run


@pytest.mark.asyncio
async def test_gate_bounds_regeneration_count(monkeypatch):
    # Three distinct test-bug files, but max_regens=2 → only 2 regenerated.
    diags = [
        DiagnosticObject(file=f"tests/unit/test_{n}.py", error_code="NameError",
                         message="name 'patch' is not defined")
        for n in ("a", "b", "c")
    ]
    calls = []

    async def _fake_regen(state):
        rel = state["node_state"]["unsatisfiable_test"]
        calls.append(rel)
        return {
            "modified_files": [rel],
            "node_state": {"test_regeneration": {"status": "regenerated"}},
        }

    monkeypatch.setattr(trn, "test_regeneration_node", _fake_regen)
    res = await tg._run_pre_repair_triage_gate(
        state={"test_regeneration_config": {}, "messages": []},
        build_result=_BR(1, diags), executor=_Executor(_BR(0, [])),
        test_cmd="pytest", workspace_path="/ws", loop_counter={},
        token_tracker={}, budget=1.0, max_regens=2,
    )
    assert res is not None
    assert len(calls) == 2        # bounded
    assert res[4]["detected"] == 3
