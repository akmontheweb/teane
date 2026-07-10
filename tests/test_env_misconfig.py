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
    _PIP_INSTALLABLE_SYMBOLS,
    _env_misconfig_hint,
    _is_env_misconfig,
    _warn_if_commits_will_be_no_ops,
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

    # -----------------------------------------------------------------
    # Fix H — RuntimeError "requires the X package" family
    # -----------------------------------------------------------------

    def test_starlette_testclient_requires_httpx2(self):
        # Verbatim message from starlette.testclient when httpx2 isn't
        # installed. Session fe51a89a-* saw this in
        # backend/tests/test_api_filings.py at test-collection time; the
        # pre-fix classifier missed it (it's a RuntimeError, not a
        # ModuleNotFoundError) so the LLM chased downstream 404s and
        # search-block misses for 8 rounds before HITL. Fix H tags it as
        # MISSING_DEP so _try_missing_dep can auto-append `httpx2` to
        # requirements.txt in one compile cycle.
        raw = (
            "  raise RuntimeError(\n"
            "RuntimeError: The starlette.testclient module requires the "
            "httpx2 package to be installed.\n"
            "    $ pip install httpx2\n"
        )
        assert _is_env_misconfig(raw) == ("httpx2", "python")

    def test_requires_package_family_generic(self):
        # The same shape shows up beyond starlette — SQLAlchemy dialects,
        # Pillow plugins, and any lib with optional runtime deps. Verify
        # the pattern captures a plain generic form so we don't need to
        # extend it every time a new lib adopts the idiom.
        raw = (
            "RuntimeError: The foo.bar module requires the baz-quux "
            "package to be installed.\n"
        )
        assert _is_env_misconfig(raw) == ("baz-quux", "python")

    def test_requires_package_does_not_match_prose(self):
        # A README / commit-message mentioning "requires the httpx2
        # package" without the RuntimeError prefix must NOT trigger.
        raw = (
            "The docs say: this feature requires the httpx2 package to "
            "be installed.\n"
        )
        assert _is_env_misconfig(raw) is None


# ---------------------------------------------------------------------------
# Whitelist — shell-invoked CLIs
# ---------------------------------------------------------------------------

class TestPipInstallableWhitelist:
    """Regression guard for the shell-CLI whitelist. Every entry gates a
    ``command not found`` shell miss: on the whitelist → repair loop
    autofix; off the whitelist → HITL short-circuit. Adding or removing
    an entry is a policy change; these tests make sure such a change
    lands deliberately, not by accident."""

    @pytest.mark.parametrize("symbol", [
        # Fix I additions.
        "alembic", "celery", "uvicorn", "gunicorn", "hypercorn",
        "django-admin", "flask", "pip-compile", "pip-tools",
        "sqlfluff", "pre-commit", "mkdocs",
        # Fix J additions — additional shell-invoked CLIs.
        "django", "daphne", "granian",
        "dramatiq", "arq", "taskiq",
        "httpie", "jupyter",
        # Pre-existing entries — asserting so a refactor can't silently
        # drop them.
        "pytest", "ruff", "mypy", "black",
    ])
    def test_symbol_in_whitelist(self, symbol):
        assert symbol in _PIP_INSTALLABLE_SYMBOLS

    def test_shell_command_not_covered_by_whitelist(self):
        # A symbol that is genuinely NOT pip-installable (base-image
        # concern, not a Python distribution) must NOT be on the list.
        # If someone adds these, they're either wrong or the semantics
        # changed and this test should be reviewed.
        for name in ("docker", "node", "npm", "make", "gcc", "cargo"):
            assert name not in _PIP_INSTALLABLE_SYMBOLS


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
# compiler_node — pytest exit 5 "no tests collected" carve-out (A2)
# ---------------------------------------------------------------------------

class TestNoTestsCollectedCarveOut:
    """Finsearch session 44c5e194 root cause A2: compiler_node used to
    fold pytest exit 5 to success whenever the workspace had source
    files. That's legitimate for genuine early greenfield rounds, but
    once test_generation has run, "no tests collected" means the runner
    isn't seeing our tests — a real failure. The narrow greenfield
    carve-out only lifts when test_generation has never fired."""

    @pytest.mark.asyncio
    async def test_greenfield_exit_5_still_folds_to_success(
        self, stub_sandbox, tmp_path,
    ):
        # test_generation counter is 0, no generated_tests — this is a
        # legitimate greenfield round. Preserve the old fold-to-success
        # so we don't over-correct.
        (tmp_path / "src.py").write_text("x = 1\n")
        stub_sandbox(5, "no tests ran in 0.01s\n")
        state = {
            "workspace_path": str(tmp_path),
            "build_command": "python3 -m pytest -q",
            "allow_network": False,
            "sandbox_config": {"docker_image": "python:3.12-slim"},
            "loop_counter": {"test_generation": 0},
            "generated_tests": [],
        }
        result = await compiler_node(state)
        assert result["exit_code"] == 0
        assert "tests_not_collected" not in result["node_state"]

    @pytest.mark.asyncio
    async def test_post_testgen_exit_5_surfaces_diagnostic(
        self, stub_sandbox, tmp_path,
    ):
        # test_generation has run and emitted files, yet the runner
        # still says no tests collected → the tests aren't being
        # collected (PYTHONPATH, conftest, testpaths mismatch, etc.).
        # This must become a repair-eligible diagnostic, not a silent
        # pass.
        (tmp_path / "src.py").write_text("x = 1\n")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_thing.py").write_text(
            "def test_thing():\n    assert True\n",
        )
        stub_sandbox(5, "no tests ran in 0.01s\n")
        state = {
            "workspace_path": str(tmp_path),
            "build_command": "python3 -m pytest -q",
            "allow_network": False,
            "sandbox_config": {"docker_image": "python:3.12-slim"},
            "loop_counter": {"test_generation": 3},
            "generated_tests": ["tests/test_thing.py"],
        }
        result = await compiler_node(state)
        assert result["exit_code"] == 5
        assert result["node_state"].get("tests_not_collected") is True
        assert len(result["compiler_errors"]) == 1
        diag = result["compiler_errors"][0]
        assert diag["error_code"] == "TESTS_NOT_COLLECTED"
        # Message must actionably point at the common causes.
        msg = diag["message"].lower()
        assert "pythonpath" in msg or "testpaths" in msg or "conftest" in msg

    @pytest.mark.asyncio
    async def test_testgen_iterated_but_zero_emit_still_surfaces(
        self, stub_sandbox, tmp_path,
    ):
        # Edge case: test_generation_node ran but emitted 0 tests (the
        # `test_generation_zero_emit` HITL trigger pattern). Even with
        # empty generated_tests, the fact that test_generation has run
        # AT ALL means we're past greenfield — exit 5 shouldn't pass.
        (tmp_path / "src.py").write_text("x = 1\n")
        stub_sandbox(5, "no tests ran in 0.01s\n")
        state = {
            "workspace_path": str(tmp_path),
            "build_command": "python3 -m pytest -q",
            "allow_network": False,
            "sandbox_config": {"docker_image": "python:3.12-slim"},
            "loop_counter": {"test_generation": 5},
            "generated_tests": [],
        }
        result = await compiler_node(state)
        assert result["exit_code"] == 5
        assert result["node_state"].get("tests_not_collected") is True


# ---------------------------------------------------------------------------
# _warn_if_commits_will_be_no_ops — session-start commit config warning (A5)
# ---------------------------------------------------------------------------

class TestCommitNoOpWarning:
    """Finsearch session 44c5e194 root cause A5: with the default
    ``commit_on_story=false`` or a non-git workspace, batch commits
    silently no-op. Warn once at session start so the operator sees it
    before burning hours on a run with no rollback point."""

    def test_warns_when_workspace_not_a_git_repo(self, caplog, tmp_path):
        # No .git dir; commit_on_story=True is still a no-op.
        import logging
        caplog.set_level(logging.WARNING, logger="harness.graph")
        _warn_if_commits_will_be_no_ops(
            str(tmp_path), decomposition_enabled=True, commit_on_story=True,
        )
        assert any(
            "not a git repository" in r.message for r in caplog.records
        ), caplog.records

    def test_warns_when_repo_present_but_flag_off(self, caplog, tmp_path):
        import logging
        (tmp_path / ".git").mkdir()
        caplog.set_level(logging.WARNING, logger="harness.graph")
        _warn_if_commits_will_be_no_ops(
            str(tmp_path), decomposition_enabled=True, commit_on_story=False,
        )
        assert any(
            "commit_on_story" in r.message and "false" in r.message
            for r in caplog.records
        ), caplog.records

    def test_no_warn_when_repo_present_and_flag_on(self, caplog, tmp_path):
        import logging
        (tmp_path / ".git").mkdir()
        caplog.set_level(logging.WARNING, logger="harness.graph")
        _warn_if_commits_will_be_no_ops(
            str(tmp_path), decomposition_enabled=True, commit_on_story=True,
        )
        assert not any(
            "[commit]" in r.message for r in caplog.records
        )

    def test_no_warn_when_decomposition_disabled(self, caplog, tmp_path):
        # Single-shot runs don't go through the batch commit path, so
        # the warn is irrelevant noise there.
        import logging
        caplog.set_level(logging.WARNING, logger="harness.graph")
        _warn_if_commits_will_be_no_ops(
            str(tmp_path), decomposition_enabled=False, commit_on_story=False,
        )
        assert not any(
            "[commit]" in r.message for r in caplog.records
        )


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
