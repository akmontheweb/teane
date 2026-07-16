"""Opt-in best-of-N runner (harness/best_of_n_runner.py).

Covers the parts that don't require a live LLM: config parsing / activation,
child-argv reconstruction, the real git worktree → diff → apply plumbing, the
recursion guard, and the full orchestration with a mock variant-runner
against a scratch repo (winner selection + winner-diff application).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile

import pytest

from harness import best_of_n_runner as R


class TestConfig:
    def test_defaults_off(self):
        c = R.BestOfNConfig.from_config(None)
        assert c.enabled is False and not c.is_active()

    def test_parse_and_activation(self):
        c = R.BestOfNConfig.from_config({"best_of_n": {"enabled": True, "n": 3}})
        assert c.enabled and c.n == 3 and c.is_active()

    def test_enabled_but_n1_is_inactive(self):
        assert not R.BestOfNConfig.from_config(
            {"best_of_n": {"enabled": True, "n": 1}}).is_active()

    def test_bad_strategy_falls_back(self):
        assert R.BestOfNConfig.from_config(
            {"best_of_n": {"strategy": "nope"}}).strategy == "first_success"

    def test_bad_types_dont_raise(self):
        c = R.BestOfNConfig.from_config({"best_of_n": {"n": "x", "max_concurrency": "y"}})
        assert c.n == 1 and c.max_concurrency == 3


class TestReconstructArgv:
    def test_strips_workspace_and_bestof_spaced(self):
        assert R.reconstruct_child_argv(
            ["teane", "build", "-w", "/repo", "--best-of", "3", "--git", "true"]
        ) == ["teane", "build", "--git", "true"]

    def test_strips_equals_forms(self):
        assert R.reconstruct_child_argv(
            ["teane", "build", "--workspace=/repo", "--best-of=3"]
        ) == ["teane", "build"]

    def test_leaves_other_args(self):
        argv = ["teane", "patch", "--agile", "true"]
        assert R.reconstruct_child_argv(argv) == argv


class TestChildGuard:
    def test_env_guard(self, monkeypatch):
        monkeypatch.delenv("TEANE_BEST_OF_N_CHILD", raising=False)
        assert not R.is_best_of_n_child()
        monkeypatch.setenv("TEANE_BEST_OF_N_CHILD", "1")
        assert R.is_best_of_n_child()


@pytest.fixture()
def repo():
    if shutil.which("git") is None:
        pytest.skip("git not available")
    ws = tempfile.mkdtemp()
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t.co",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t.co"}
    with open(os.path.join(ws, "app.py"), "w") as f:
        f.write("x = 1\n")
    for a in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "init"]):
        subprocess.run(["git", "-C", ws, *a], check=True, env=env,
                       capture_output=True)
    yield ws
    shutil.rmtree(ws, ignore_errors=True)


class TestOrchestrationOnRealRepo:
    def test_winner_selected_and_applied(self, repo):
        async def variant_runner(vid, wt, cfg):
            if vid == 0:  # big change, non-green
                with open(os.path.join(wt, "app.py"), "w") as f:
                    f.write("x = 1\nA=1\nB=2\nC=3\n")
                return (1, 0, 0, 0.1)
            with open(os.path.join(wt, "app.py"), "w") as f:  # green, small diff
                f.write("x = 2\n")
            return (0, 0, 0, 0.1)

        winner, results = asyncio.run(R.run_best_of_n_build(
            repo, "sess1234",
            R.BestOfNConfig(enabled=True, n=2, strategy="fewest_changes"),
            variant_runner=variant_runner,
        ))
        assert winner is not None and winner.variant_id == 1
        with open(os.path.join(repo, "app.py")) as f:
            assert f.read() == "x = 2\n"
        assert len(results) == 2

    def test_all_fail_leaves_workspace_untouched(self, repo):
        async def variant_runner(vid, wt, cfg):
            with open(os.path.join(wt, "app.py"), "w") as f:
                f.write("broken\n")
            return (1, 0, 0, 0.0)

        winner, _ = asyncio.run(R.run_best_of_n_build(
            repo, "sess5678",
            R.BestOfNConfig(enabled=True, n=2, strategy="first_success"),
            variant_runner=variant_runner,
        ))
        assert winner is None
        with open(os.path.join(repo, "app.py")) as f:
            assert f.read() == "x = 1\n"  # unchanged

    def test_worktrees_are_cleaned_up(self, repo):
        async def variant_runner(vid, wt, cfg):
            with open(os.path.join(wt, "app.py"), "w") as f:
                f.write("x = 3\n")
            return (0, 0, 0, 0.0)

        asyncio.run(R.run_best_of_n_build(
            repo, "cleanup01",
            R.BestOfNConfig(enabled=True, n=2, strategy="first_success"),
            variant_runner=variant_runner,
        ))
        bon_dir = os.path.join(os.path.dirname(os.path.abspath(repo)),
                               ".teane_bon_cleanup")
        # worktree parent dir may linger empty, but no v0/v1 checkouts remain
        assert not os.path.exists(os.path.join(bon_dir, "v0"))
