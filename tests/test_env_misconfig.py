"""Tests for the env-misconfig short-circuit in harness/graph.py.

When the sandbox build fails because a required runtime is missing (pytest
not in `python:3.12-slim`, `npm: command not found` in a bare image), no
amount of LLM repair can fix it from inside the container. The harness
now detects these failures and routes straight to HITL with a focused
error message instead of burning 3 repair iterations.
"""

from __future__ import annotations

import pytest

from harness.graph import (
    _env_misconfig_hint,
    _is_env_misconfig,
    compiler_node,
    route_after_compiler,
)


# ---------------------------------------------------------------------------
# _is_env_misconfig — pattern detector
# ---------------------------------------------------------------------------

class TestIsEnvMisconfig:

    def test_python_no_module_named_pytest(self):
        # The exact wording from the user's incident.
        raw = "/usr/local/bin/python3: No module named pytest\n"
        assert _is_env_misconfig(raw) == ("pytest", "python")

    def test_modulenotfounderror(self):
        raw = (
            "Traceback (most recent call last):\n"
            "  File \"<stdin>\", line 1, in <module>\n"
            "ModuleNotFoundError: No module named 'requests'\n"
        )
        assert _is_env_misconfig(raw) == ("requests", "python")

    def test_command_not_found_npm(self):
        raw = "/bin/sh: 1: npm: not found\n"
        assert _is_env_misconfig(raw) == ("npm", "shell")

    def test_command_not_found_sh_no_bin_prefix(self):
        # python:3.12-slim's dash emits "sh: 1: <cmd>: not found" with no
        # leading /bin/ prefix. Session 51ecb569 hit this with
        # `sh: 1: make: not found` and the pre-fix regex missed it, so
        # the loop wasted 5 LLM repair iterations on a sandbox config
        # issue no patch could fix. The relaxed regex must match both
        # forms; the /bin/ prefix variant is still covered by the npm
        # test above.
        raw = "sh: 1: make: not found\n"
        assert _is_env_misconfig(raw) == ("make", "shell")

    def test_command_not_found_bash(self):
        raw = "bash: cargo: command not found\n"
        assert _is_env_misconfig(raw) == ("cargo", "shell")

    def test_docker_executable_not_found(self):
        raw = (
            'docker: Error response from daemon: failed to create task '
            'for container: exec: "go": executable file not found in $PATH.\n'
        )
        assert _is_env_misconfig(raw) == ("go", "shell")

    def test_empty_returns_none(self):
        assert _is_env_misconfig("") is None

    def test_regular_compile_error_returns_none(self):
        # A real Rust compile error must NOT trigger the env-misconfig
        # short-circuit — the LLM is supposed to fix this one.
        raw = (
            "error[E0425]: cannot find value `foo` in this scope\n"
            "  --> src/main.rs:2:13\n"
        )
        assert _is_env_misconfig(raw) is None

    def test_pytest_failure_returns_none(self):
        # pytest itself failing (a real test bug) must not trigger.
        raw = (
            "tests/test_x.py F\n"
            "======= 1 failed in 0.05s =======\n"
        )
        assert _is_env_misconfig(raw) is None

    def test_dotted_local_import_returns_none(self):
        # A dotted module name (e.g. 'api.database') is a local-import bug
        # in the user's code, not a missing system package. Must NOT trigger
        # the short-circuit — the repair loop should get a chance to fix it
        # (add pytest.ini pythonpath, create conftest.py, etc.).
        raw = (
            "Traceback (most recent call last):\n"
            "  File \"tests/api/test_database.py\", line 6, in <module>\n"
            "    from api.database import engine, SessionLocal, Base\n"
            "ModuleNotFoundError: No module named 'api.database'\n"
        )
        assert _is_env_misconfig(raw) is None

    def test_dotted_python_dash_m_returns_none(self):
        # `python3 -m foo.bar` where foo.bar isn't a runnable module is also
        # a local-code problem, not a missing system package.
        raw = "/usr/local/bin/python3: No module named myapp.cli\n"
        assert _is_env_misconfig(raw) is None

    def test_scans_tail_only_for_perf(self):
        # Even with a huge log, the helper finds the trailing
        # pytest-missing line.
        raw = ("filler line\n" * 5000) + "/usr/local/bin/python3: No module named pytest\n"
        assert _is_env_misconfig(raw) == ("pytest", "python")

    def test_python_app_dep_returns_python_kind(self):
        # Regression: httpx and other application deps (fastapi, pydantic,
        # sqlalchemy, …) used to be misclassified as ENV_MISCONFIG because
        # they weren't in the pip-installable-test-tools whitelist. The
        # `python` kind tag is now what the compiler_node keys on, and
        # MISSING_DEP applies to every Python ModuleNotFoundError.
        raw = "ModuleNotFoundError: No module named 'httpx'\n"
        assert _is_env_misconfig(raw) == ("httpx", "python")
        raw = "ModuleNotFoundError: No module named 'fastapi'\n"
        assert _is_env_misconfig(raw) == ("fastapi", "python")


# ---------------------------------------------------------------------------
# _env_misconfig_hint — actionable HITL message
# ---------------------------------------------------------------------------

class TestEnvMisconfigHint:

    def test_pytest_hint_suggests_pip_install(self):
        hint = _env_misconfig_hint("pytest", "python3 -m pytest -q")
        assert "pip install pytest" in hint
        assert "python3 -m pytest -q" in hint
        # The hint must say the LLM can't fix it.
        assert "LLM" in hint or "repair" in hint.lower()

    def test_node_hint_says_change_image(self):
        # node / npm cannot be pip-installed; tell the user to swap the
        # docker_image instead of trying to install it in the container.
        hint = _env_misconfig_hint("npm", "npm test")
        assert "docker_image" in hint
        assert "node" in hint.lower()


# ---------------------------------------------------------------------------
# compiler_node — synthesizes ENV_MISCONFIG diagnostic + flag
# ---------------------------------------------------------------------------

class _StubBuildResult:
    """Stand-in for harness.sandbox.BuildResult — just the fields compiler_node reads."""
    def __init__(self, exit_code: int, raw_output: str):
        self.exit_code = exit_code
        self.raw_output = raw_output
        self.diagnostics: list = []
        self.elapsed_seconds = 0.01
        self.timed_out = False
        self.log_truncated = False


class _StubSandboxExecutor:
    """SandboxExecutor stub that returns a pre-canned BuildResult."""
    canned: _StubBuildResult

    def __init__(self, **kwargs):
        pass

    async def run(self, build_command: str):
        return _StubSandboxExecutor.canned


@pytest.fixture
def stub_sandbox(monkeypatch, tmp_path):
    """Replace harness.sandbox.SandboxExecutor with our stub and yield a
    canned-result setter."""
    import harness.sandbox as sandbox_mod
    monkeypatch.setattr(sandbox_mod, "SandboxExecutor", _StubSandboxExecutor)

    def _set(exit_code: int, raw_output: str) -> None:
        _StubSandboxExecutor.canned = _StubBuildResult(exit_code, raw_output)

    return _set


class TestCompilerNodeShortCircuit:

    @pytest.mark.asyncio
    async def test_compiler_node_short_circuits_on_non_installable(
        self, stub_sandbox, tmp_path,
    ):
        # npm/node/cargo/go aren't pip-installable from inside the sandbox —
        # the image itself is wrong, so compiler_node must short-circuit:
        #   - synthesise a single ENV_MISCONFIG diagnostic
        #   - set node_state["env_misconfig"] = True
        #   - record the symbol so the router and HITL can surface it
        stub_sandbox(1, "/bin/sh: 1: npm: not found\n")

        state = {
            "workspace_path": str(tmp_path),
            "build_command": "npm install && npm test",
            "allow_network": False,
            "sandbox_config": {"docker_image": "python:3.12-slim"},
            "loop_counter": {},
        }
        result = await compiler_node(state)

        assert result["exit_code"] == 1
        assert result["node_state"]["env_misconfig"] is True
        assert result["node_state"]["env_misconfig_symbol"] == "npm"
        assert len(result["compiler_errors"]) == 1
        diag = result["compiler_errors"][0]
        assert diag["error_code"] == "ENV_MISCONFIG"
        assert "npm" in diag["message"]

    @pytest.mark.asyncio
    async def test_compiler_node_routes_pytest_through_repair_not_hitl(
        self, stub_sandbox, tmp_path,
    ):
        # Repro of the user's recurring HITL bounce: codegen produced a
        # requirements.txt without pytest, build fails with "No module
        # named pytest". Before the fix this short-circuited to HITL and
        # the repair loop never got to amend the manifest. Now:
        #   - emits a MISSING_DEP diagnostic pointing at requirements.txt
        #   - does NOT set node_state["env_misconfig"], so route_after_compiler
        #     falls through to the normal repair_node path
        stub_sandbox(1, "/usr/local/bin/python3: No module named pytest\n")

        state = {
            "workspace_path": str(tmp_path),
            "build_command": (
                "python3 -m pip install -r requirements.txt && python3 -m pytest -q"
            ),
            "allow_network": True,
            "sandbox_config": {"docker_image": "python:3.12-slim"},
            "loop_counter": {},
        }
        result = await compiler_node(state)

        assert result["exit_code"] == 1
        # No env_misconfig flag → router will go to repair_node, not HITL.
        assert "env_misconfig" not in result["node_state"]
        assert "env_misconfig_symbol" not in result["node_state"]
        assert len(result["compiler_errors"]) == 1
        diag = result["compiler_errors"][0]
        assert diag["error_code"] == "MISSING_DEP"
        assert "pytest" in diag["message"]
        # The diagnostic must point the repair LLM at requirements.txt
        # (the actionable fix), not at the legacy "swap docker_image" path.
        assert "requirements.txt" in diag["message"]
        assert "pyproject.toml" in diag["message"]

    @pytest.mark.asyncio
    async def test_compiler_node_does_not_short_circuit_on_real_error(
        self, stub_sandbox, tmp_path,
    ):
        # A real compile error (no "No module named" pattern) must NOT
        # trigger env_misconfig — the LLM repair path must still run.
        # We send raw output the generic parser CAN'T pattern-match, so
        # compiler_errors stays empty and the only thing that COULD flip
        # env_misconfig is _is_env_misconfig itself.
        stub_sandbox(
            1,
            "error: something went wrong without a recognized error pattern\n",
        )

        state = {
            "workspace_path": str(tmp_path),
            "build_command": "make build",
            "allow_network": False,
            "sandbox_config": {},
            "loop_counter": {},
        }
        result = await compiler_node(state)

        assert result["exit_code"] == 1
        assert "env_misconfig" not in result["node_state"]


# ---------------------------------------------------------------------------
# route_after_compiler — env_misconfig skips the repair loop
# ---------------------------------------------------------------------------

class TestRouterEnvMisconfig:

    def test_routes_to_hitl_on_env_misconfig_even_with_budget(self):
        # The router used to send exit≠0 + repairs<3 to repair_node
        # regardless of whether the LLM could fix it. Env-misconfig now
        # short-circuits straight to HITL even when budget and repair
        # counter are fresh.
        state = {
            "exit_code": 1,
            "budget_remaining_usd": 1.99,
            "loop_counter": {"total_repairs": 0},
            "node_state": {
                "env_misconfig": True,
                "env_misconfig_symbol": "pytest",
            },
        }
        assert route_after_compiler(state) == "human_intervention_node"

    def test_regular_failure_still_routes_to_repair(self):
        # Sanity check: when env_misconfig is NOT set, the original
        # repair path still triggers.
        state = {
            "exit_code": 1,
            "budget_remaining_usd": 1.99,
            "loop_counter": {"total_repairs": 0},
            "node_state": {},
        }
        assert route_after_compiler(state) == "repair_node"

    def test_success_still_routes_to_security_scan(self):
        state = {
            "exit_code": 0,
            "budget_remaining_usd": 1.99,
            "loop_counter": {"total_repairs": 0},
            "node_state": {},
        }
        assert route_after_compiler(state) == "security_scan_node"
