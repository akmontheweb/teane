"""Repair-level fanout (speculative.repair_fanout).

When the sequential repair loop accrues K consecutive no-progress rounds,
sample N repair variants, test-compile each in a worktree seeded with the
current dirty workspace, and hand the best response back to repair_node's
normal apply path. The workspace itself is never touched by the fanout.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from types import SimpleNamespace

import pytest

import harness.speculative as spec
from harness.speculative import (
    SpeculativeConfig,
    _seed_worktree_from_workspace,
    maybe_run_repair_fanout,
    repair_fanout_should_engage,
)


def _cfg(**over):
    raw = {"repair_fanout": True, **over}
    return SpeculativeConfig.normalize(raw)


class TestConfig:
    def test_defaults_off(self):
        cfg = SpeculativeConfig.normalize({})
        assert cfg.repair_fanout is False
        assert cfg.repair_fanout_variants == 3
        assert cfg.repair_fanout_after_rounds == 2

    def test_clamping(self):
        cfg = SpeculativeConfig.normalize({
            "repair_fanout_variants": 99, "repair_fanout_after_rounds": 0,
        })
        assert cfg.repair_fanout_variants == 5   # hi clamp
        assert cfg.repair_fanout_after_rounds == 1  # lo clamp


class TestEngage:
    def test_disabled_never_engages(self):
        assert not repair_fanout_should_engage(
            SpeculativeConfig.normalize({}), {"no_progress_repairs": 2},
        )

    def test_engages_exactly_at_threshold(self):
        cfg = _cfg(repair_fanout_after_rounds=2)
        assert not repair_fanout_should_engage(cfg, {"no_progress_repairs": 1})
        assert repair_fanout_should_engage(cfg, {"no_progress_repairs": 2})
        # Above the threshold means a fanout already ran this climb and
        # failed — don't burn N more dispatches every remaining round.
        assert not repair_fanout_should_engage(cfg, {"no_progress_repairs": 3})

    def test_missing_counter_is_zero(self):
        assert not repair_fanout_should_engage(_cfg(), {})


@pytest.fixture
def git_workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    subprocess.run(["git", "init", "-q", str(ws)], check=True)
    subprocess.run(["git", "-C", str(ws), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(ws), "config", "user.name", "t"], check=True)
    (ws / "committed.py").write_text("original\n")
    (ws / "doomed.py").write_text("delete me\n")
    subprocess.run(["git", "-C", str(ws), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(ws), "commit", "-qm", "init"], check=True)
    return ws


class TestWorktreeSeeding:
    def test_dirty_state_mirrored(self, git_workspace, tmp_path):
        ws = git_workspace
        # Dirty the workspace: modify tracked, add untracked (nested), delete tracked.
        (ws / "committed.py").write_text("modified\n")
        (ws / "pkg").mkdir()
        (ws / "pkg" / "new.py").write_text("untracked\n")
        (ws / "doomed.py").unlink()

        wt = str(tmp_path / "wt")
        assert spec._create_worktree(str(ws), wt)
        synced = _seed_worktree_from_workspace(str(ws), wt)
        assert synced == 3
        assert open(os.path.join(wt, "committed.py")).read() == "modified\n"
        assert open(os.path.join(wt, "pkg", "new.py")).read() == "untracked\n"
        assert not os.path.exists(os.path.join(wt, "doomed.py"))
        # Workspace untouched by seeding.
        assert (ws / "committed.py").read_text() == "modified\n"
        spec._remove_worktree(str(ws), wt)


def _response(tag: str):
    return SimpleNamespace(
        content=f"PATCH BODY {tag}", usage={"total_tokens": 10, "tag": tag},
    )


def _state(ws, **over):
    return {
        "workspace_path": str(ws),
        "speculative_config": {"repair_fanout": True, "repair_fanout_after_rounds": 2},
        "build_command": "echo build",
        **over,
    }


def _fake_apply(success=True):
    async def _apply(content, worktree, existing_modified_files):
        return ([SimpleNamespace(success=success, no_op=False)],
                ["some_file.py"] if success else [])
    return _apply


def _run(coro):
    return asyncio.run(coro)


class TestFanout:
    def test_config_off_is_inert_even_at_threshold(self, git_workspace):
        # repair_fanout=false (the shipped default) must be a hard no-op:
        # no variant dispatches, no worktrees — repair_node falls through
        # to its normal single sequential dispatch, exactly as before the
        # feature existed.
        calls = []

        async def dispatch(msgs, budget):  # pragma: no cover — must not run
            calls.append(msgs)
            return _response("x"), budget

        state = _state(git_workspace)
        state["speculative_config"] = {"repair_fanout": False}
        out = _run(maybe_run_repair_fanout(
            state=state, messages=[], dispatch=dispatch,
            workspace_path=str(git_workspace), budget=1.0,
            loop_counter={"no_progress_repairs": 2},  # trigger WOULD be met
        ))
        assert out is None
        assert calls == []

    def test_absent_speculative_section_is_inert(self, git_workspace):
        # A config with no speculative section at all (older configs) must
        # also mean fanout-off.
        async def dispatch(msgs, budget):  # pragma: no cover — must not run
            raise AssertionError("dispatch must not be called")

        out = _run(maybe_run_repair_fanout(
            state={"workspace_path": str(git_workspace)}, messages=[],
            dispatch=dispatch, workspace_path=str(git_workspace), budget=1.0,
            loop_counter={"no_progress_repairs": 2},
        ))
        assert out is None

    def test_trigger_not_met_returns_none_without_dispatch(self, git_workspace):
        calls = []

        async def dispatch(msgs, budget):
            calls.append(msgs)
            return _response("x"), budget

        out = _run(maybe_run_repair_fanout(
            state=_state(git_workspace), messages=[], dispatch=dispatch,
            workspace_path=str(git_workspace), budget=1.0,
            loop_counter={"no_progress_repairs": 1},
        ))
        assert out is None
        assert calls == []

    def test_winner_selected_and_budget_threaded(self, git_workspace, monkeypatch):
        monkeypatch.setattr(spec, "process_llm_patch_output", _fake_apply())
        dispatched = []

        async def dispatch(msgs, budget):
            i = len(dispatched)
            dispatched.append(msgs)
            return _response(f"v{i}"), budget - 0.01

        async def compile_variant(vr):
            vr.exit_code = 0 if vr.index == 1 else 1  # variant 1 compiles
            return vr

        out = _run(maybe_run_repair_fanout(
            state=_state(git_workspace), messages=[{"role": "user", "content": "fix"}],
            dispatch=dispatch, workspace_path=str(git_workspace), budget=1.0,
            loop_counter={"no_progress_repairs": 2},
            compile_variant=compile_variant,
        ))
        assert out is not None and out.won is True
        assert out.response.usage["tag"] == "v1"
        assert out.budget == pytest.approx(1.0 - 0.03)  # all 3 dispatches paid
        assert len(out.extra_usages) == 2               # losers' usage surfaced
        assert len(dispatched) == 3
        # Each variant got a distinct strategy directive appended.
        directives = {d[-1]["content"] for d in dispatched}
        assert len(directives) == 3
        # Workspace files untouched (fanout never writes to the workspace).
        assert (git_workspace / "committed.py").read_text() == "original\n"

    def test_no_winner_falls_back_to_best_applying(self, git_workspace, monkeypatch):
        monkeypatch.setattr(spec, "process_llm_patch_output", _fake_apply())

        async def dispatch(msgs, budget):
            return _response(f"v{len(msgs)}"), budget

        async def compile_variant(vr):
            vr.exit_code = 1
            return vr

        out = _run(maybe_run_repair_fanout(
            state=_state(git_workspace), messages=[], dispatch=dispatch,
            workspace_path=str(git_workspace), budget=1.0,
            loop_counter={"no_progress_repairs": 2},
            compile_variant=compile_variant,
        ))
        assert out is not None and out.won is False
        assert out.response is not None

    def test_all_dispatches_fail_returns_none(self, git_workspace):
        async def dispatch(msgs, budget):
            raise RuntimeError("provider down")

        out = _run(maybe_run_repair_fanout(
            state=_state(git_workspace), messages=[], dispatch=dispatch,
            workspace_path=str(git_workspace), budget=1.0,
            loop_counter={"no_progress_repairs": 2},
        ))
        assert out is None

    def test_unborn_head_returns_none(self, tmp_path):
        ws = tmp_path / "empty"
        ws.mkdir()
        subprocess.run(["git", "init", "-q", str(ws)], check=True)

        async def dispatch(msgs, budget):  # pragma: no cover — must not run
            raise AssertionError("dispatch must not be called")

        out = _run(maybe_run_repair_fanout(
            state=_state(ws), messages=[], dispatch=dispatch,
            workspace_path=str(ws), budget=1.0,
            loop_counter={"no_progress_repairs": 2},
        ))
        assert out is None

    def test_internal_error_is_fail_open(self, git_workspace, monkeypatch):
        async def dispatch(msgs, budget):
            return _response("x"), budget

        def _boom(*a, **k):
            raise RuntimeError("worktree layer exploded")

        monkeypatch.setattr(spec, "_create_worktree", _boom)
        out = _run(maybe_run_repair_fanout(
            state=_state(git_workspace), messages=[], dispatch=dispatch,
            workspace_path=str(git_workspace), budget=1.0,
            loop_counter={"no_progress_repairs": 2},
        ))
        # Worktree failures poison every variant -> no winner, best-effort
        # response still returned (never an exception).
        assert out is None or out.won is False
