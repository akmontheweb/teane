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

    def test_committed_winner_work_still_lands(self, repo):
        # Regression: with agile_defaults.commit_on_story=true the child
        # teane process commits inside its worktree after each green story,
        # advancing the worktree branch. The old code diffed against the
        # worktree's OWN HEAD, so a fully-committed winner produced an empty
        # patch and apply_winner_diff returned True having landed NOTHING —
        # a silent total loss of the run's work.
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t.co",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t.co"}

        async def variant_runner(vid, wt, cfg):
            with open(os.path.join(wt, "app.py"), "w") as f:
                f.write("x = 42\n")
            with open(os.path.join(wt, "story2.py"), "w") as f:
                f.write("y = 7\n")
            # Commit ALL work, story-style — worktree HEAD moves past the
            # branch point and `git diff HEAD` in the worktree is empty.
            for a in (["add", "-A"], ["commit", "-qm", "STORY-1: done"]):
                subprocess.run(["git", "-C", wt, *a], check=True, env=env,
                               capture_output=True)
            return (0, 0, 0, 0.1)

        winner, _ = asyncio.run(R.run_best_of_n_build(
            repo, "commit01",
            R.BestOfNConfig(enabled=True, n=1, strategy="first_success"),
            variant_runner=variant_runner,
        ))
        assert winner is not None
        with open(os.path.join(repo, "app.py")) as f:
            assert f.read() == "x = 42\n"
        assert os.path.exists(os.path.join(repo, "story2.py"))

    def test_committed_work_counts_in_diffstat(self, repo):
        # The fewest_changes strategy reads _diff_stat — committed work must
        # count, or a fully-committed variant scores (0, 0) and games the
        # smallest-diff selection.
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t.co",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t.co"}
        captured: dict[int, tuple[int, int]] = {}

        async def variant_runner(vid, wt, cfg):
            with open(os.path.join(wt, "app.py"), "w") as f:
                f.write("x = 1\nextra = True\n")
            for a in (["add", "-A"], ["commit", "-qm", "STORY-1"]):
                subprocess.run(["git", "-C", wt, *a], check=True, env=env,
                               capture_output=True)
            return (0, 0, 0, 0.0)

        winner, results = asyncio.run(R.run_best_of_n_build(
            repo, "stat0001",
            R.BestOfNConfig(enabled=True, n=1, strategy="first_success"),
            variant_runner=variant_runner,
        ))
        assert winner is not None
        assert winner.changed_files == 1
        assert winner.lines_changed >= 1

    def test_failed_apply_saves_winner_patch_before_teardown(
        self, repo, monkeypatch, tmp_path,
    ):
        # Regression: a winner whose diff failed to apply (dirty operator
        # workspace) was force-removed with its worktree — with
        # commit_on_story=false the winning diff existed nowhere else.
        monkeypatch.setenv("HOME", str(tmp_path))

        async def variant_runner(vid, wt, cfg):
            with open(os.path.join(wt, "app.py"), "w") as f:
                f.write("x = 99\n")
            return (0, 0, 0, 0.0)

        winner, _ = asyncio.run(R.run_best_of_n_build(
            repo, "rescue01",
            R.BestOfNConfig(enabled=True, n=1, strategy="first_success"),
            variant_runner=variant_runner,
            diff_applier=lambda wt, ws: False,  # simulate apply failure
        ))
        assert winner is None
        rescue = tmp_path / ".harness" / "best_of_n" / "rescue01-winner.patch"
        assert rescue.exists()
        patch_text = rescue.read_text()
        assert "x = 99" in patch_text
        # Workspace untouched by the failed apply.
        with open(os.path.join(repo, "app.py")) as f:
            assert f.read() == "x = 1\n"


class TestSubprocessRunnerWiring:
    def _capture_spawn(self, monkeypatch):
        captured: dict = {}

        class _FakeProc:
            returncode = 0

            async def communicate(self):
                return (b"", None)

        async def fake_exec(*cmd, **kwargs):
            captured["cmd"] = list(cmd)
            captured["kwargs"] = kwargs
            return _FakeProc()

        monkeypatch.setattr(R.asyncio, "create_subprocess_exec", fake_exec)
        return captured

    def test_child_stdin_closed_and_auto_approve_pinned(self, monkeypatch):
        # Regression: children inherited the parent's interactive tty, so
        # HITL gates defaulted ON, wrote their menus into the unread stdout
        # pipe, and every variant blocked on input() until the timeout.
        captured = self._capture_spawn(monkeypatch)
        runner = R.make_subprocess_variant_runner(["teane", "build"])
        exit_code, *_ = asyncio.run(
            runner(0, "/tmp/wt0", R.BestOfNConfig(enabled=True, n=2))
        )
        assert exit_code == 0
        assert captured["kwargs"]["stdin"] == asyncio.subprocess.DEVNULL
        assert captured["kwargs"]["env"]["HARNESS_AUTO_APPROVE"] == "true"
        assert captured["kwargs"]["env"]["TEANE_BEST_OF_N_CHILD"] == "1"
        # Own process group, so the timeout path can killpg the whole
        # tree instead of orphaning sandbox/MCP grandchildren.
        assert captured["kwargs"]["start_new_session"] is True

    def test_per_variant_budget_passed_as_child_budget_flag(self, monkeypatch):
        # Regression: per_variant_budget_usd was a documented no-op (the
        # runner hardcoded cost=0.0 and nothing enforced the cap). The cap
        # is now enforced by the child's own budget gateway via --budget.
        captured = self._capture_spawn(monkeypatch)
        runner = R.make_subprocess_variant_runner(["teane", "build"])
        asyncio.run(runner(
            0, "/tmp/wt0",
            R.BestOfNConfig(enabled=True, n=2, per_variant_budget_usd=2.5),
        ))
        cmd = captured["cmd"]
        assert "--budget" in cmd
        assert cmd[cmd.index("--budget") + 1] == "2.5"

    def test_no_budget_flag_when_unset(self, monkeypatch):
        captured = self._capture_spawn(monkeypatch)
        runner = R.make_subprocess_variant_runner(["teane", "build"])
        asyncio.run(runner(0, "/tmp/wt0", R.BestOfNConfig(enabled=True, n=2)))
        assert "--budget" not in captured["cmd"]


class TestCleanup:
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
                               ".teane_bon_cleanup0")
        assert not os.path.exists(os.path.join(bon_dir, "v0"))
        # The parent dir and the agent/bon-* branches are swept too — a
        # leaked branch used to make a reused session id fail outright.
        assert not os.path.exists(bon_dir)
        branches = subprocess.run(
            ["git", "-C", repo, "branch", "--list", "agent/bon-*"],
            capture_output=True, text=True,
        ).stdout
        assert branches.strip() == ""

    def test_session_id_reuse_survives_leaked_branch(self, repo):
        # Simulate a crashed prior run: branch exists, worktree gone.
        subprocess.run(
            ["git", "-C", repo, "branch", "agent/bon-reuse123-v0"],
            check=True, capture_output=True,
        )
        wt = R.make_worktree(repo, "reuse123", 0)
        assert os.path.isdir(wt)
        R.remove_worktree(repo, wt)
