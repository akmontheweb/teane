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


class TestStrictValidatorTypes:
    def test_string_false_is_rejected_not_silently_enabled(self):
        # Regression: the three repair_fanout keys were registered in
        # _KNOWN_NESTED_KEYS but not _TYPE_SCHEMA, so `"repair_fanout":
        # "false"` passed validation and bool("false") then ENABLED the
        # feature from a config that reads as off.
        import harness.cli as cli_mod
        bad_config = {
            "models": {"any": {"provider": "deepseek", "model_id": "x",
                               "api_key": ""}},
            "model_routing": {
                "planning_primary": "any",
                "patching_primary": "any",
                "repair_primary": "any",
            },
            "persistence": {"db_path": "~/.harness/x.db"},
            "token_budget": {"hard_cap_usd": 1.0},
            "sandbox": {"backend": "bare"},
            "product_spec_dir": "product_spec",
            "speculative": {"repair_fanout": "false"},
        }
        with pytest.raises(cli_mod.ConfigError) as ex:
            cli_mod.validate_config_strict(bad_config, source="<test>")
        assert "repair_fanout" in str(ex.value)

    def test_int_keys_reject_strings(self):
        import harness.cli as cli_mod
        for key in ("repair_fanout_variants", "repair_fanout_after_rounds"):
            bad_config = {
                "models": {"any": {"provider": "deepseek", "model_id": "x",
                                   "api_key": ""}},
                "model_routing": {
                    "planning_primary": "any",
                    "patching_primary": "any",
                    "repair_primary": "any",
                },
                "persistence": {"db_path": "~/.harness/x.db"},
                "token_budget": {"hard_cap_usd": 1.0},
                "sandbox": {"backend": "bare"},
                "product_spec_dir": "product_spec",
                "speculative": {key: "3"},
            }
            with pytest.raises(cli_mod.ConfigError):
                cli_mod.validate_config_strict(bad_config, source="<test>")


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

    def test_staged_rename_removes_origin_in_worktree(self, git_workspace, tmp_path):
        # Regression: `git status --porcelain=v1 -z` emits `R  new\0old` for
        # a STAGED rename; the seeder copied the destination but never
        # removed the origin — the worktree then held BOTH files (duplicate
        # pytest collection, shadowed imports), corrupting the variant's
        # compile verdict.
        ws = git_workspace
        subprocess.run(["git", "-C", str(ws), "mv", "committed.py", "renamed.py"],
                       check=True, capture_output=True)
        wt = str(tmp_path / "wt")
        assert spec._create_worktree(str(ws), wt)
        _seed_worktree_from_workspace(str(ws), wt)
        assert os.path.exists(os.path.join(wt, "renamed.py"))
        assert not os.path.exists(os.path.join(wt, "committed.py"))
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


def _fake_apply(success=True, seen=None):
    async def _apply(content, worktree, existing_modified_files,
                     allowed_paths=None):
        if seen is not None:
            seen.append({"content": content, "allowed_paths": allowed_paths})
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

    def test_apply_uses_allowlist_and_stripped_payload(self, git_workspace, monkeypatch):
        # Scoring parity with the real apply path (graph.py repair_node):
        # variants must be applied under the same source-root allowlist and
        # with READ_FILE blocks stripped — otherwise a variant that writes
        # outside the allowlist (or leads with READ_FILE) is scored under
        # laxer rules than the winner will face on the real apply.
        seen: list = []
        monkeypatch.setattr(spec, "process_llm_patch_output",
                            _fake_apply(seen=seen))

        async def dispatch(msgs, budget):
            resp = _response("v")
            resp.content = (
                "<<<READ_FILE>>>\nfile: app.py\n<<<END_READ_FILE>>>\n"
                "PATCH BODY"
            )
            return resp, budget

        async def compile_variant(vr):
            vr.exit_code = 0
            return vr

        out = _run(maybe_run_repair_fanout(
            state=_state(git_workspace), messages=[], dispatch=dispatch,
            workspace_path=str(git_workspace), budget=1.0,
            loop_counter={"no_progress_repairs": 2},
            compile_variant=compile_variant,
        ))
        assert out is not None
        assert seen, "apply seam never called"
        for call in seen:
            assert "READ_FILE" not in call["content"]
            assert "PATCH BODY" in call["content"]
            # graph._build_patcher_allowlist returns a non-None allowlist
            # for a real workspace — the fanout must thread it through.
            assert call["allowed_paths"] is not None

    def test_post_dispatch_failure_still_returns_paid_dispatches(
        self, git_workspace, monkeypatch,
    ):
        # Regression: an exception AFTER the variant dispatches (worktree
        # plumbing, selection, cleanup) fell into the outer fail-open and
        # returned None — repair_node then re-dispatched against the
        # PRE-fanout budget, so the N paid dispatches vanished from
        # budget/token accounting.
        monkeypatch.setattr(spec, "process_llm_patch_output", _fake_apply())

        def _boom(*a, **k):
            raise RuntimeError("post-dispatch plumbing failure")
        monkeypatch.setattr(spec, "_cleanup_worktrees", _boom)

        async def dispatch(msgs, budget):
            return _response(f"v{len(msgs)}"), budget - 0.01

        async def compile_variant(vr):
            vr.exit_code = 1
            return vr

        out = _run(maybe_run_repair_fanout(
            state=_state(git_workspace), messages=[], dispatch=dispatch,
            workspace_path=str(git_workspace), budget=1.0,
            loop_counter={"no_progress_repairs": 2},
            compile_variant=compile_variant,
        ))
        assert out is not None, "paid dispatches were dropped"
        assert out.won is False
        assert out.budget == pytest.approx(1.0 - 0.03)
        assert len(out.extra_usages) == 2

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
