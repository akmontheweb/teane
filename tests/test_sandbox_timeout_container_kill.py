"""Tests for the sandbox timeout path: killing the container the outer
timeout orphans, and classifying such a timeout as an operator-action
condition rather than feeding it to the code-repair loop.

Reproduces lumina session 01a1648d. A stalling `httpcore==1.0.9` wheel
fetch pushed a prod-smoke run past `sandbox.timeout_seconds`. Two separate
defects compounded:

1. The timeout killed the host-side `docker run` client but left the
   container running, so each round competed with the orphans of every
   previous round — one timeout became a cascade.
2. The killed run produced no `FAIL:` lines, so the coarse fallback handed
   the repair LLM a network error at `<prod-import-smoke>:0:0`. The judge
   reported "insufficient data — no diagnostic locations available" and
   patch success fell 3/3 -> 1/2 -> 0/2 across three guessing rounds.
"""

import asyncio
import os


from harness.sandbox import (
    BuildResult,
    DockerBackend,
    _new_container_name,
)


class TestContainerName:
    def test_has_identifiable_prefix(self):
        # The prefix is what lets an operator spot strays in `docker ps`
        # when cleanup could not run at all (harness itself SIGKILLed).
        assert _new_container_name().startswith("teane-build-")

    def test_unique_per_call(self):
        # Per-invocation, not per-session: a repair loop runs many builds
        # and reusing a name races a container still shutting down, which
        # makes `docker run` fail outright on a name conflict.
        names = {_new_container_name() for _ in range(100)}
        assert len(names) == 100

    def test_is_a_legal_docker_name(self):
        import re
        # Docker requires [a-zA-Z0-9][a-zA-Z0-9_.-]*
        assert re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", _new_container_name())


class TestDockerCommandCarriesName:
    def test_name_flag_emitted(self, monkeypatch):
        monkeypatch.setattr(os, "getuid", lambda: 0)
        backend = DockerBackend(image="python:3.12-slim")
        cmd = backend._build_docker_command(
            "pytest -q", "/work", allow_network=False,
            cache_mounts=[], extra_env={}, timeout_seconds=60,
            container_name="teane-build-deadbeef",
        )
        assert "--name" in cmd
        assert cmd[cmd.index("--name") + 1] == "teane-build-deadbeef"

    def test_name_generated_when_omitted(self, monkeypatch):
        # Callers that only inspect the argv (and the existing test suite)
        # may omit it; the container must still be nameable.
        monkeypatch.setattr(os, "getuid", lambda: 0)
        backend = DockerBackend(image="python:3.12-slim")
        cmd = backend._build_docker_command(
            "pytest -q", "/work", allow_network=False,
            cache_mounts=[], extra_env={}, timeout_seconds=60,
        )
        assert cmd[cmd.index("--name") + 1].startswith("teane-build-")

    def test_rm_still_present(self):
        # --rm is what removes the container once it is dead; the kill only
        # makes it die. Losing --rm would trade a leaked running container
        # for a leaked stopped one.
        backend = DockerBackend(image="python:3.12-slim")
        cmd = backend._build_docker_command(
            "pytest -q", "/work", allow_network=False,
            cache_mounts=[], extra_env={}, timeout_seconds=60,
        )
        assert "--rm" in cmd


class TestTimeoutKillsContainer:
    def test_on_timeout_hook_is_passed_and_kills_by_name(self, monkeypatch):
        """The executor must receive an on_timeout hook, and invoking it
        must `docker kill` the same name that went into the argv."""
        captured: dict = {}

        async def _fake_exec(cmd, timeout_seconds, **kwargs):
            captured["cmd"] = cmd
            captured["on_timeout"] = kwargs.get("on_timeout")
            return (-9, "", True, False)

        killed: list[str] = []

        async def _fake_kill(self, name):
            killed.append(name)

        monkeypatch.setattr(
            "harness.sandbox._execute_subprocess_with_timeout", _fake_exec,
        )
        monkeypatch.setattr(DockerBackend, "_kill_container", _fake_kill)

        backend = DockerBackend(image="python:3.12-slim")
        asyncio.run(backend.run(
            command="sleep 60", workspace_path="/work",
            timeout_seconds=5, allow_network=False,
        ))

        hook = captured["on_timeout"]
        assert hook is not None, "docker backend must supply a timeout hook"

        # The hook is what the executor calls after killing the PGID.
        asyncio.run(hook())

        argv_name = captured["cmd"][captured["cmd"].index("--name") + 1]
        assert killed == [argv_name], (
            "the hook must kill the container actually launched, not a "
            "freshly generated name"
        )

    def test_executor_invokes_hook_on_timeout(self):
        """The shared executor must actually call the hook when it times
        out — the hook existing is no use if nothing fires it."""
        from harness.sandbox import _execute_subprocess_with_timeout

        fired = []

        async def _hook():
            fired.append(True)

        exit_code, _out, timed_out, _ = asyncio.run(
            _execute_subprocess_with_timeout(
                ["sleep", "30"], timeout_seconds=1, on_timeout=_hook,
            )
        )
        assert timed_out is True
        assert fired == [True]

    def test_executor_does_not_invoke_hook_on_clean_exit(self):
        from harness.sandbox import _execute_subprocess_with_timeout

        fired = []

        async def _hook():
            fired.append(True)

        exit_code, _out, timed_out, _ = asyncio.run(
            _execute_subprocess_with_timeout(
                ["echo", "ok"], timeout_seconds=30, on_timeout=_hook,
            )
        )
        assert exit_code == 0
        assert timed_out is False
        assert fired == []

    def test_hook_failure_does_not_mask_the_timeout(self):
        """A cleanup failure must never turn a reported timeout into
        something else — the timeout is the finding."""
        from harness.sandbox import _execute_subprocess_with_timeout

        async def _bad_hook():
            raise RuntimeError("docker daemon unreachable")

        exit_code, _out, timed_out, _ = asyncio.run(
            _execute_subprocess_with_timeout(
                ["sleep", "30"], timeout_seconds=1, on_timeout=_bad_hook,
            )
        )
        assert timed_out is True
        assert exit_code == -9


class TestKillContainerIsIdempotent:
    def test_already_gone_is_not_an_error(self, monkeypatch, caplog):
        """`docker kill` losing the race against --rm is the outcome we
        wanted, so it must not be logged as a failure."""
        class _Proc:
            returncode = 1
            async def communicate(self):
                return (b"", b"Error response from daemon: No such container: x")

        async def _fake_subprocess(*args, **kwargs):
            return _Proc()

        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", _fake_subprocess,
        )
        backend = DockerBackend(image="python:3.12-slim")
        with caplog.at_level("WARNING"):
            asyncio.run(backend._kill_container("teane-build-x"))
        assert "failed" not in caplog.text.lower()

    def test_daemon_error_is_warned(self, monkeypatch, caplog):
        class _Proc:
            returncode = 1
            async def communicate(self):
                return (b"", b"Cannot connect to the Docker daemon")

        async def _fake_subprocess(*args, **kwargs):
            return _Proc()

        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", _fake_subprocess,
        )
        backend = DockerBackend(image="python:3.12-slim")
        with caplog.at_level("WARNING"):
            asyncio.run(backend._kill_container("teane-build-x"))
        assert "teane-build-x" in caplog.text


class TestProdSmokeTimeoutClassification:
    """A timed-out smoke check must not be presented as a code defect."""

    def _run(self, monkeypatch, tmp_path, *, timed_out, raw_output, exit_code):
        from harness import graph

        (tmp_path / "requirements.txt").write_text("fastapi\n")
        (tmp_path / "app.py").write_text("x = 1\n")

        class _FakeExecutor:
            def __init__(self, *a, **kw):
                pass
            async def run(self, cmd):
                return BuildResult(
                    exit_code=exit_code,
                    raw_output=raw_output,
                    timed_out=timed_out,
                    elapsed_seconds=303.0,
                )

        monkeypatch.setattr("harness.sandbox.SandboxExecutor", _FakeExecutor)
        return asyncio.run(graph._run_prod_import_smoke_check(
            workspace_path=str(tmp_path),
            sandbox_config={},
            allow_network=True,
            install_step="uv pip install -r requirements.txt",
            session_id="test-sess",
        ))

    def test_timeout_yields_build_env_timeout(self, monkeypatch, tmp_path):
        diags = self._run(
            monkeypatch, tmp_path,
            timed_out=True,
            raw_output=(
                "Creating virtual environment at: /tmp/teane-venv\n"
                "  x Failed to download `httpcore==1.0.9`\n"
                "  |-> Request failed after 3 retries in 101.1s\n"
                "  `-> operation timed out\n"
            ),
            exit_code=-9,
        )
        assert len(diags) == 1
        assert diags[0]["error_code"] == "BUILD_ENV_TIMEOUT"
        # Not attributed to a workspace file — there is no code defect.
        assert diags[0]["file"] == "<harness:sandbox-timeout>"

    def test_timeout_message_states_nothing_was_proved(self, monkeypatch, tmp_path):
        diags = self._run(
            monkeypatch, tmp_path,
            timed_out=True, raw_output="stalled", exit_code=-9,
        )
        msg = diags[0]["message"]
        assert "No module was proved to fail" in msg
        # Must name the operator-side levers, since repair cannot act.
        assert "sandbox.timeout_seconds" in msg
        assert "cache_volumes" in msg

    def test_timeout_message_keeps_the_last_output(self, monkeypatch, tmp_path):
        diags = self._run(
            monkeypatch, tmp_path,
            timed_out=True,
            raw_output="x Failed to download `httpcore==1.0.9`",
            exit_code=-9,
        )
        # The stalling package is the actionable detail.
        assert "httpcore" in diags[0]["message"]

    def test_real_import_failure_still_classified_normally(self, monkeypatch, tmp_path):
        """Guard against the timeout branch swallowing genuine failures."""
        diags = self._run(
            monkeypatch, tmp_path,
            timed_out=False,
            raw_output=(
                "=== PROD_IMPORT_SMOKE_FAILURES ===\n"
                "FAIL: app: NameError: /work/app.py:3: name 'Generator' "
                "is not defined\n"
            ),
            exit_code=1,
        )
        assert all(d["error_code"] != "BUILD_ENV_TIMEOUT" for d in diags)

    def test_clean_exit_yields_no_diagnostics(self, monkeypatch, tmp_path):
        diags = self._run(
            monkeypatch, tmp_path,
            timed_out=False,
            raw_output="=== PROD_IMPORT_SMOKE_OK ===",
            exit_code=0,
        )
        assert diags == []

    def test_non_timeout_failure_without_fail_lines_is_not_called_a_module_count(
        self, monkeypatch, tmp_path, caplog,
    ):
        """The coarse fallback always appends exactly one diagnostic. It
        must not be reported as '1 production module(s) failed to import'
        — no module was shown to fail."""
        with caplog.at_level("WARNING"):
            diags = self._run(
                monkeypatch, tmp_path,
                timed_out=False,
                raw_output="error: Failed to install pydantic-core",
                exit_code=1,
            )
        assert len(diags) == 1
        assert "production module(s) failed to import" not in caplog.text
        assert "without naming a failing module" in caplog.text

    def test_identified_failures_are_still_counted(
        self, monkeypatch, tmp_path, caplog,
    ):
        with caplog.at_level("WARNING"):
            self._run(
                monkeypatch, tmp_path,
                timed_out=False,
                raw_output=(
                    "=== PROD_IMPORT_SMOKE_FAILURES ===\n"
                    "FAIL: app: NameError: /work/app.py:3: boom\n"
                ),
                exit_code=1,
            )
        assert "production module(s) failed to import" in caplog.text


class TestRouterEscalatesBuildEnvTimeout:
    def test_hitl_trigger_is_distinct(self):
        """A sandbox timeout must get its own trigger, not the generic
        persistent_build_failure story — that would send the operator
        hunting for a code defect that does not exist."""
        from harness.graph import _infer_hitl_trigger

        state = {
            "node_state": {"build_env_timeout": True},
            "exit_code": 1,
            "budget_remaining_usd": 9.0,
        }
        assert _infer_hitl_trigger(state, max_repair=5) == "build_env_timeout"

    def test_trigger_precedes_persistent_build_failure(self):
        """Both conditions coexist (exit_code != 0 is what got us here),
        so the specific label must win."""
        from harness.graph import _infer_hitl_trigger

        state = {
            "node_state": {"build_env_timeout": True},
            "exit_code": 1,
            "budget_remaining_usd": 9.0,
            "loop_counter": {"repair": 5},
        }
        assert _infer_hitl_trigger(state, max_repair=5) == "build_env_timeout"

    def test_absent_flag_does_not_produce_the_trigger(self):
        from harness.graph import _infer_hitl_trigger

        state = {
            "node_state": {},
            "exit_code": 1,
            "budget_remaining_usd": 9.0,
        }
        assert _infer_hitl_trigger(state, max_repair=5) != "build_env_timeout"

    def test_router_sends_it_to_hitl(self):
        """route_after_compiler must escalate rather than dispatch repair."""
        from harness.graph import route_after_compiler

        state = {
            "node_state": {"build_env_timeout": True, "current_node": "compiler"},
            "exit_code": 1,
            "budget_remaining_usd": 9.0,
            "loop_counter": {"repair": 0},
            "compiler_errors": [{
                "error_code": "BUILD_ENV_TIMEOUT",
                "message": "killed",
                "file": "<harness:sandbox-timeout>",
                "line": 0,
                "column": 0,
                "severity": "error",
            }],
        }
        assert route_after_compiler(state) == "human_intervention_node"

    def test_post_mortem_has_advice_for_it(self):
        from harness.post_mortem import _TRIGGER_ADVICE
        advice = _TRIGGER_ADVICE["build_env_timeout"]
        assert "no code defect" in advice
        assert "sandbox.timeout_seconds" in advice

    def test_counts_as_infrastructure_failure(self):
        """Exit code must say 'something outside the harness must change',
        not 'the build failed'."""
        from harness.cli import (
            EXIT_INFRASTRUCTURE_FAILURE,
            _resolve_cli_exit_code,
        )

        assert _resolve_cli_exit_code(
            graph_exit_code=1,
            final_state={"node_state": {"build_env_timeout": True}},
        ) == EXIT_INFRASTRUCTURE_FAILURE


class TestOperatorLogHandlerSurvives:
    def test_configure_logging_preserves_operator_handler(self, tmp_path):
        """`--log` attaches a root handler before the session's logging
        stack is configured. configure_logging clears root handlers to own
        the config; it must leave the operator's alone, or the file goes
        silent after startup (which is the opposite of what --log
        promises for headless runs)."""
        import logging
        from harness.observability import (
            OPERATOR_HANDLER_ATTR,
            configure_logging,
        )

        log_file = tmp_path / "operator.log"
        handler = logging.FileHandler(str(log_file), encoding="utf-8")
        setattr(handler, OPERATOR_HANDLER_ATTR, True)
        root = logging.getLogger()
        original = list(root.handlers)
        root.addHandler(handler)
        try:
            configure_logging(
                session_id="sess-test",
                log_dir=str(tmp_path / "logs"),
                max_bytes=0,
            )
            assert handler in logging.getLogger().handlers
            logging.getLogger("harness.test").warning("after configure")
            handler.flush()
            assert "after configure" in log_file.read_text()
        finally:
            root.removeHandler(handler)
            handler.close()
            for h in list(root.handlers):
                if h not in original:
                    root.removeHandler(h)

    def test_unmarked_handlers_are_still_cleared(self, tmp_path):
        """The ownership contract is unchanged for everything else."""
        import logging
        from harness.observability import configure_logging

        stray = logging.StreamHandler()
        root = logging.getLogger()
        original = list(root.handlers)
        root.addHandler(stray)
        try:
            configure_logging(
                session_id="sess-test-2",
                log_dir=str(tmp_path / "logs"),
                max_bytes=0,
            )
            assert stray not in logging.getLogger().handlers
        finally:
            for h in list(root.handlers):
                if h not in original:
                    root.removeHandler(h)
