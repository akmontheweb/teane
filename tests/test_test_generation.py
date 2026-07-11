"""Tests for harness/test_generation.py — the auto test-generation + deterministic
sandbox execution node wired between speculative_node and lintgate_node.
"""

from __future__ import annotations

from typing import Any

import pytest

from harness.test_generation import (
    _PRIMARY_STACK_PRIORITY,
    _STACK_TEST_COMMANDS,
    _is_test_file,
    _parse_verifies_marker,
    _persist_verifies_links,
    _pick_primary_stack,
    _stack_test_command,
    _inside_workspace,
    route_after_test_generation,
)
# Imported under a non-test_ alias so pytest's auto-collection doesn't try
# to invoke this graph node as if it were a test function.
from harness.test_generation import test_generation_node as run_test_generation


# ---------------------------------------------------------------------------
# Helpers — stubs for the gateway + sandbox so the node runs without touching
# the network or spinning up docker.
# ---------------------------------------------------------------------------

class _StubUsage:
    input_tokens = 100
    output_tokens = 80
    cached_tokens = 0
    cost_usd = 0.001
    model = "stub"


class _StubResponse:
    def __init__(self, content: str):
        self.content = content
        self.usage = _StubUsage()


class _StubGateway:
    """Records every dispatch + returns canned patch content."""

    def __init__(self, content: str):
        self._content = content
        self.dispatched = []

    class config:
        repair_fallback = ""
        planning_fallback = ""

    async def dispatch(self, *, messages, role, budget_remaining_usd, **kwargs):
        self.dispatched.append({"messages": list(messages), "role": role})
        return _StubResponse(self._content), budget_remaining_usd - 0.001

    def aggregate_tokens(self, tracker, usage, role=None):
        out = dict(tracker or {})
        out["total_cost_usd"] = out.get("total_cost_usd", 0.0) + float(usage.cost_usd)
        return out


class _StubBuildResult:
    def __init__(self, exit_code: int, raw_output: str):
        self.exit_code = exit_code
        self.raw_output = raw_output
        self.diagnostics = []
        self.elapsed_seconds = 0.01
        self.timed_out = False
        self.log_truncated = False


class _StubSandboxExecutor:
    """Records every run() invocation and returns a pre-canned BuildResult."""
    last_command: str = ""
    canned: _StubBuildResult = _StubBuildResult(0, "")

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def run(self, build_command: str):
        _StubSandboxExecutor.last_command = build_command
        return _StubSandboxExecutor.canned


@pytest.fixture
def stub_sandbox(monkeypatch):
    """Replace harness.sandbox.SandboxExecutor with the stub above and return
    a setter the test can use to pre-can the build result."""
    import harness.sandbox as sandbox_mod
    monkeypatch.setattr(sandbox_mod, "SandboxExecutor", _StubSandboxExecutor)

    def _set(exit_code: int, raw_output: str = "") -> None:
        _StubSandboxExecutor.canned = _StubBuildResult(exit_code, raw_output)
        _StubSandboxExecutor.last_command = ""

    _set(0, "")  # default to passing sandbox
    return _set


@pytest.fixture
def stub_gateway(monkeypatch):
    """Install a stub LLM gateway for the duration of the test."""
    from harness import graph as graph_mod

    holder: dict[str, _StubGateway] = {}

    def _set(content: str) -> _StubGateway:
        gw = _StubGateway(content)
        graph_mod.set_gateway(gw)
        holder["gw"] = gw
        return gw

    yield _set

    graph_mod.set_gateway(None)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_is_test_file_python(self):
        assert _is_test_file("tests/test_foo.py") is True
        assert _is_test_file("src/foo.py") is False

    def test_is_test_file_javascript(self):
        assert _is_test_file("src/foo.test.ts") is True
        assert _is_test_file("__tests__/foo.spec.js") is True
        assert _is_test_file("src/foo.ts") is False

    def test_is_test_file_java(self):
        assert _is_test_file("src/test/java/com/x/FooTest.java") is True
        assert _is_test_file("src/main/java/com/x/Foo.java") is False

    def test_pick_primary_stack_prefers_specific_over_generic(self):
        # typescript should win over node when both are present
        assert _pick_primary_stack({"node", "typescript"}) == "typescript"
        assert _pick_primary_stack({"python", "node"}) in ("node", "python")

    def test_pick_primary_stack_none_when_unknown(self):
        assert _pick_primary_stack({"cobol"}) is None
        assert _pick_primary_stack(set()) is None

    def test_stack_test_command_runs_pytest_for_python(self):
        # pytest is pre-baked into the builder image so the per-run install
        # step is gone — the command is now just the pytest invocation.
        cmd = _stack_test_command("python")
        assert cmd is not None
        assert "pytest" in cmd
        # Regression: the legacy `pip install -q pytest && ...` prefix must
        # NOT come back; that round-tripped to PyPI on every single test run.
        assert "pip install" not in cmd

    def test_stack_test_command_runs_jest_for_javascript(self):
        # jest is also pre-baked; `npx --no-install` resolves it from PATH.
        cmd = _stack_test_command("javascript")
        assert cmd is not None
        assert "jest" in cmd
        assert "npm install" not in cmd

    def test_every_priority_stack_has_a_test_command(self):
        # If we add a new stack to the priority list we must also add its
        # test command, otherwise the node silently skips the deterministic
        # run. Catch the omission in CI.
        for tag in _PRIMARY_STACK_PRIORITY:
            assert tag in _STACK_TEST_COMMANDS, (
                f"stack {tag!r} in _PRIMARY_STACK_PRIORITY but missing from "
                f"_STACK_TEST_COMMANDS"
            )

    def test_inside_workspace_accepts_relative(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_x.py").write_text("")
        assert _inside_workspace("tests/test_x.py", str(tmp_path)) is True

    def test_inside_workspace_rejects_absolute(self, tmp_path):
        assert _inside_workspace("/etc/passwd", str(tmp_path)) is False

    def test_inside_workspace_rejects_traversal(self, tmp_path):
        assert _inside_workspace("../outside.py", str(tmp_path)) is False


# ---------------------------------------------------------------------------
# Skip / gate behaviours
# ---------------------------------------------------------------------------

class TestSkipBehaviour:

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self, tmp_path, stub_sandbox, stub_gateway):
        # enabled: false → no work, no LLM call
        gw = stub_gateway("")
        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["foo.py"],
            "test_generation_config": {"enabled": False},
        })
        assert result == {}
        assert gw.dispatched == []

    @pytest.mark.asyncio
    async def test_skips_when_modified_files_empty(self, tmp_path, stub_sandbox, stub_gateway):
        gw = stub_gateway("")
        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": [],
        })
        assert result == {}
        assert gw.dispatched == []

    @pytest.mark.asyncio
    async def test_skips_when_only_test_files_modified(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        # modified files are themselves tests → nothing to do
        gw = stub_gateway("")
        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["tests/test_foo.py", "src/foo.test.ts"],
        })
        assert result == {}
        assert gw.dispatched == []

    @pytest.mark.asyncio
    async def test_skips_when_no_supported_stack(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        gw = stub_gateway("")
        # Unknown extension → no stack tag inferred
        (tmp_path / "x.cobol").write_text("DISPLAY 'hi'.\n")
        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["x.cobol"],
        })
        assert result == {}
        assert gw.dispatched == []

    @pytest.mark.asyncio
    async def test_routes_to_hitl_when_no_gateway(self, tmp_path, stub_sandbox):
        # gateway is None → env_misconfig diagnostic + route to HITL
        from harness import graph as graph_mod
        graph_mod.set_gateway(None)

        (tmp_path / "foo.py").write_text("def x(): return 1\n")
        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["foo.py"],
        })
        assert result["node_state"]["env_misconfig"] is True
        assert result["node_state"]["env_misconfig_symbol"] == "llm_api_key"
        # The synthetic diagnostic must spell out the fix the operator needs.
        msg = result["compiler_errors"][0]["message"]
        assert "LLM API key" in msg
        assert "ANTHROPIC_API_KEY" in msg
        # And the router must send it to HITL.
        assert route_after_test_generation(result) == "human_intervention_node"

    @pytest.mark.asyncio
    async def test_skips_nfr_only_batch_without_dispatching(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        """2026-07-10: finsearch session tripped
        ``env_misconfig:test_generation_zero_emit`` three times because the
        LLM refused to generate unit tests for pure-NFR batches
        (STORY-NFR-001, STORY-NFR-004). Verify test_generation now
        short-circuits cleanly when EVERY story key in the batch is a SAFe
        enabler / NFR story — no LLM dispatch, no HITL, no counter burn."""
        gw = stub_gateway("<<<CREATE_FILE:tests/test_x.py>>>\ndef test_x():\n    pass\n<<<END>>>\n")
        (tmp_path / "foo.py").write_text("def x(): return 1\n")
        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["foo.py"],
            "batch_patched_story_keys": ["STORY-NFR-001", "STORY-NFR-004"],
        })
        # No LLM call, no HITL.
        assert gw.dispatched == []
        assert "env_misconfig" not in (result.get("node_state") or {})
        assert "llm_behavior" not in (result.get("node_state") or {})
        # Result payload records the reason so downstream nodes / audit
        # can distinguish "skipped by policy" from "generated 0 tests".
        node_state = result["node_state"]
        assert node_state["test_generation"]["status"] == "skipped"
        assert node_state["test_generation"]["reason"] == "nfr_only_batch"
        assert node_state["test_generation"]["story_keys"] == [
            "STORY-NFR-001", "STORY-NFR-004",
        ]

    @pytest.mark.asyncio
    async def test_skips_when_current_story_id_is_nfr(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        """Single-story path (agile per-story mode): current_story_id
        points at STORY-NFR-*, batch_patched_story_keys not populated
        yet. Same skip applies."""
        gw = stub_gateway("...")
        (tmp_path / "foo.py").write_text("def x(): return 1\n")
        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["foo.py"],
            "current_story_id": "STORY-NFR-007",
        })
        assert gw.dispatched == []
        assert result["node_state"]["test_generation"]["reason"] == "nfr_only_batch"
        assert result["node_state"]["test_generation"]["story_keys"] == ["STORY-NFR-007"]

    @pytest.mark.asyncio
    async def test_mixed_nfr_and_regular_batch_still_dispatches(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        """Guardrail is intentionally narrow: mixed batches (NFR alongside
        a regular story) MUST still run test-gen — the regular story
        anchors the tests and the NFR ACs can ride along as ``@verifies:``
        citations."""
        gw = stub_gateway("<<<CREATE_FILE:tests/test_foo.py>>>\ndef test_foo():\n    from foo import x\n    assert x() == 1\n<<<END>>>\n")
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "foo.py").write_text("def x(): return 1\n")
        stub_sandbox(0, "1 passed in 0.01s\n")
        await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["foo.py"],
            "batch_patched_story_keys": ["STORY-005", "STORY-NFR-002"],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
        })
        # LLM was actually dispatched — the guardrail did NOT skip.
        assert len(gw.dispatched) >= 1


# ---------------------------------------------------------------------------
# Happy path: LLM emits a CREATE_FILE block, sandbox passes
# ---------------------------------------------------------------------------

class TestHappyPath:

    @pytest.mark.asyncio
    async def test_python_writes_test_runs_pytest_routes_to_lintgate(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        # Workspace with a Python source file
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "calculator.py").write_text(
            "def divide(a, b):\n"
            "    if b == 0:\n"
            "        raise ZeroDivisionError('cannot divide by zero')\n"
            "    return a // b\n"
        )

        # Stub the LLM to return one CREATE_FILE block for a test file.
        # v5 Phase 3 contract: every generated test MUST carry a
        # `# @verifies: STORY-N.AC-N` marker. This canned response
        # includes one so the marker gate passes; the ac_key references
        # an AC that doesn't exist in this test's state.db, which
        # produces a "dropped unknown ac_key" log but doesn't fail the
        # gate (Phase 3 warn-and-drop).
        gw = stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_calculator.py\n"
            "content:\n"
            "# @verifies: STORY-001.AC-1\n"
            "from calculator import divide\n"
            "def test_divide():\n"
            "    assert divide(10, 2) == 5\n"
            "<<<END_CREATE_FILE>>>\n"
        )

        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["calculator.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
        })

        # 1. LLM was called exactly once
        assert len(gw.dispatched) == 1
        sent = gw.dispatched[0]["messages"]
        # 2. The Python test guide was injected into the system messages
        guide_sys = [m for m in sent if m.get("role") == "system" and "monkeypatch" in m.get("content", "")]
        assert guide_sys, "system prompt should include the python test guide"
        # 3. The prompt explicitly forbids mocks
        user_msgs = [m for m in sent if m.get("role") == "user"]
        joined_user = "\n".join(m.get("content", "") for m in user_msgs)
        assert "Do NOT generate mocks" in joined_user or "do not generate mocks" in joined_user.lower()
        # 4. The deterministic test command ran in the sandbox. pytest is
        # pre-baked into the builder image so there's no install prefix —
        # just the bare pytest invocation.
        assert "pytest" in _StubSandboxExecutor.last_command
        assert "pip install" not in _StubSandboxExecutor.last_command
        # 5. The result reports a pass and lists the generated test
        assert result["node_state"]["test_generation"]["status"] == "passed"
        assert result["generated_tests"] == ["tests/test_calculator.py"]
        # 6. The router would proceed to lintgate
        assert route_after_test_generation(result) == "lintgate_node"
        # 7. The test file landed inside the workspace, not anywhere else
        assert (tmp_path / "tests" / "test_calculator.py").is_file()

    @pytest.mark.asyncio
    async def test_zero_patch_emission_retries_then_trips_hitl(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        # Fix 2a (2026-07-10): when the LLM returns zero patch blocks
        # that's a prompt-comprehension miss, not a benign "no tests
        # needed" pass-through. The node retries inline with a stronger
        # contract (max_zero_emit_reprompts=3 by default). If the LLM
        # STILL emits nothing, HITL fires with a distinct
        # llm_behavior symbol so the operator can distinguish this
        # failure class from the generic max_iterations one.
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "foo.py").write_text("def foo(): pass\n")
        stub_gateway("no patch blocks here, just prose")
        stub_sandbox(99, "this should not be observed because sandbox should not run")

        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["foo.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
        })
        # No deterministic run happened — we never got past the
        # zero-emit retry loop.
        assert _StubSandboxExecutor.last_command == ""
        # LLM-behavior HITL fires with the distinct zero_emit symbol
        # so a post-mortem can see WHICH failure class ate the budget.
        assert result["node_state"]["llm_behavior"] is True
        assert (
            result["node_state"]["llm_behavior_symbol"]
            == "test_generation_zero_emit"
        )
        assert result["exit_code"] == 1
        # The zero-emit budget is fully consumed but the real
        # test_generation iteration counter DID NOT advance — that's
        # the whole point of the sub-counter split.
        assert result["loop_counter"]["test_generation_zero_emit"] == 3
        assert "test_generation" not in result["loop_counter"] or \
            result["loop_counter"]["test_generation"] == 0
        # Router sends this to human_intervention on llm_behavior.
        assert route_after_test_generation(result) == "human_intervention_node"


# ---------------------------------------------------------------------------
# Failure path: sandbox exits non-zero → repair_node
# ---------------------------------------------------------------------------

class TestFailurePath:

    @pytest.mark.asyncio
    async def test_test_failure_routes_to_repair_with_test_failure_code(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "foo.py").write_text("def foo(): return 1\n")
        stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_foo.py\n"
            "content:\n"
            "# @verifies: STORY-001.AC-1\n"
            "from foo import foo\n"
            "def test_foo(): assert foo() == 2  # wrong on purpose\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(1, "tests/test_foo.py:2: assert 1 == 2\nFAILED tests/test_foo.py::test_foo")

        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["foo.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
        })

        assert result["node_state"]["test_generation"]["status"] == "failed"
        assert result["compiler_errors"], "must populate compiler_errors on failure"
        # Every diagnostic must carry the TEST_FAILURE prefix so repair_node's
        # framing tweak knows these came from the test runner.
        codes = [d["error_code"] for d in result["compiler_errors"]]
        assert all(c.upper().startswith("TEST_FAILURE") for c in codes), codes
        # Router sends to repair.
        assert route_after_test_generation(result) == "repair_node"


# ---------------------------------------------------------------------------
# Graph wiring smoke test
# ---------------------------------------------------------------------------

class TestGraphWiring:

    def test_graph_includes_test_generation_node(self):
        # The build_graph must register the new node and the edge from
        # speculative_node to it. We don't execute the graph here, just
        # confirm the wiring via the graph's compiled structure.
        from harness.graph import build_graph
        try:
            g = build_graph(checkpointer=None)
        except Exception as exc:
            pytest.skip(f"build_graph requires extra deps in this env: {exc}")
        # LangGraph compiled graph exposes node names via .nodes
        node_names = set(g.nodes.keys()) if hasattr(g, "nodes") else set()
        # build_graph returns a compiled graph; the source StateGraph has
        # different introspection. Either path is acceptable — just confirm
        # the node name appears somewhere in repr.
        graph_repr = repr(g)
        assert (
            "test_generation_node" in node_names
            or "test_generation_node" in graph_repr
        )


# ---------------------------------------------------------------------------
# repair_node framing for TEST_FAILURE diagnostics
# ---------------------------------------------------------------------------

class TestRepairFraming:

    @pytest.mark.asyncio
    async def test_repair_node_uses_test_failure_framing(self, tmp_path):
        # Feed repair_node a state whose only diagnostic carries TEST_FAILURE.
        # The prompt sent to the LLM must contain the new framing sentence,
        # NOT the generic "build failed" framing or the security framing.
        from harness import graph as graph_mod

        captured: dict[str, Any] = {}

        class StubResp:
            content = ""

            class usage:
                input_tokens = 0
                output_tokens = 0
                cached_tokens = 0
                cost_usd = 0.0
                model = "stub"

        class StubGW:
            class config:
                repair_fallback = ""
                planning_fallback = ""

            async def dispatch(self, *, messages, role, budget_remaining_usd, **kwargs):
                captured["messages"] = list(messages)
                return StubResp(), budget_remaining_usd

            def aggregate_tokens(self, tracker, usage, role=None):
                return tracker or {}

        graph_mod.set_gateway(StubGW())
        try:
            await graph_mod.repair_node({
                "workspace_path": str(tmp_path),
                "compiler_errors": [{
                    "file": "tests/test_x.py",
                    "line": 5,
                    "column": 0,
                    "severity": "error",
                    "error_code": "TEST_FAILURE:assertion",
                    "message": "assert 1 == 2",
                    "semantic_context": "",
                }],
                "loop_counter": {"total_repairs": 0, "repair": 0},
                "messages": [],
                "modified_files": [],
                "budget_remaining_usd": 1.0,
            })
        finally:
            graph_mod.set_gateway(None)

        user_msgs = [m for m in captured["messages"] if m.get("role") == "user"]
        joined = "\n".join(m.get("content", "") for m in user_msgs)
        assert "harness-generated unit tests" in joined, (
            "repair_node must use the TEST_FAILURE framing for these diagnostics"
        )
        assert "Do NOT add mocks" in joined or "do not add mocks" in joined.lower()


# ---------------------------------------------------------------------------
# v5 @verifies marker — parser + gate + link writer
# ---------------------------------------------------------------------------

class TestVerifiesMarkerParser:
    """Unit tests for _parse_verifies_marker. Permissive on whitespace,
    strict on key shape (STORY-N.AC-N anchored)."""

    def test_python_comment_style(self):
        assert _parse_verifies_marker(
            "# @verifies: STORY-003.AC-2\nimport pytest"
        ) == ["STORY-003.AC-2"]

    def test_js_java_comment_style(self):
        assert _parse_verifies_marker(
            "// @verifies: STORY-001.AC-1\nimport foo;"
        ) == ["STORY-001.AC-1"]

    def test_multi_ac_comma_separated(self):
        assert _parse_verifies_marker(
            "// @verifies: STORY-001.AC-1, STORY-001.AC-2, STORY-002.AC-3\n"
        ) == ["STORY-001.AC-1", "STORY-001.AC-2", "STORY-002.AC-3"]

    def test_permissive_whitespace(self):
        assert _parse_verifies_marker(
            "#  @verifies:   STORY-005.AC-7\n"
        ) == ["STORY-005.AC-7"]

    def test_missing_marker_returns_empty(self):
        assert _parse_verifies_marker(
            "import pytest\ndef test_foo(): pass\n"
        ) == []

    def test_malformed_key_filtered_out(self):
        # NOT-A-KEY doesn't match STORY-N.AC-N; the cleaning step drops it.
        assert _parse_verifies_marker("# @verifies: NOT-A-KEY\n") == []

    def test_mixed_valid_and_invalid_keeps_valid(self):
        assert _parse_verifies_marker(
            "# @verifies: STORY-001.AC-1, BOGUS, STORY-002.AC-3\n"
        ) == ["STORY-001.AC-1", "STORY-002.AC-3"]

    def test_marker_buried_past_scan_window_ignored(self):
        body = "\n".join(
            ["# preamble"] * 60 + ["# @verifies: STORY-001.AC-1"]
        )
        assert _parse_verifies_marker(body) == []

    def test_empty_body_returns_empty(self):
        assert _parse_verifies_marker("") == []


class TestVerifiesGate:
    """Integration tests for the markerless-test gate inside
    test_generation_node."""

    @pytest.mark.asyncio
    async def test_markerless_test_routes_to_repair(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        """A generated test without a `@verifies:` marker is rejected
        before the sandbox runs; the resulting compiler_errors route
        the flow to repair_node via the existing TEST_FAILURE path."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "calculator.py").write_text("def divide(a, b): return a // b\n")
        stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_calc.py\n"
            "content:\n"
            "from calculator import divide\n"
            "def test_divide(): assert divide(10, 2) == 5\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        # If the sandbox ran, the test would pass — gate must intercept first.
        stub_sandbox(0, "1 passed in 0.01s")

        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["calculator.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
            # Phase 6: marker gate only fires in agile mode.
            "decomposition_enabled": True,
        })

        assert result["compiler_errors"], "marker gate must populate compiler_errors"
        codes = [d["error_code"] for d in result["compiler_errors"]]
        assert all(c.startswith("TEST_FAILURE:missing_verifies_marker") for c in codes)
        assert result["node_state"]["test_generation"]["status"] == "missing_verifies_marker"
        assert route_after_test_generation(result) == "repair_node"

    @pytest.mark.asyncio
    async def test_malformed_marker_treated_as_missing(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "f.py").write_text("def f(): return 1\n")
        stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_f.py\n"
            "content:\n"
            "# @verifies: NOT-A-VALID-KEY\n"
            "from f import f\n"
            "def test_f(): assert f() == 1\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(0, "1 passed")

        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["f.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
            # Phase 6: marker gate only fires in agile mode.
            "decomposition_enabled": True,
        })

        assert result["node_state"]["test_generation"]["status"] == "missing_verifies_marker"
        assert route_after_test_generation(result) == "repair_node"


class TestVerifiesLinkPersistence:
    """_persist_verifies_links writes (test_path, ac_key) edges into
    the test_verifies_ac table after the sandbox passes."""

    def test_empty_input_no_op(self, tmp_path):
        assert _persist_verifies_links(str(tmp_path), {}) == (0, 0)

    def test_inserts_known_ac_keys_and_drops_unknown(self, tmp_path):
        from harness import story_state
        # Seed: one feature + one story + one AC on a fresh state.db
        # scoped to this tmp_path's basename (which is what
        # app_name_for_workspace derives).
        ws = tmp_path / "verify-link-ws"
        ws.mkdir()
        app = story_state.app_name_for_workspace(str(ws))
        conn = story_state.open_story_db()
        try:
            story_state.ensure_feature(conn, app, "f", name="F")
            keys = story_state.create_stories(conn, app, [{
                "title": "S", "feature": "f",
                "acceptance_criteria": ["only AC"],
            }])
            sid = story_state.get_story(conn, app, keys[0])["id"]
            ac = story_state.list_acceptance_criteria(conn, app, sid)[0]
            known_ac_key = ac["ac_key"]
        finally:
            conn.close()

        inserted, dropped = _persist_verifies_links(
            str(ws),
            {
                "tests/test_real.py": [known_ac_key, "STORY-099.AC-99"],
                "tests/test_other.py": [known_ac_key],
            },
        )
        assert inserted == 2  # one per file pointing at the known AC
        assert dropped == 1  # the STORY-099 key

        # Round-trip: link rows present
        conn = story_state.open_story_db()
        try:
            rows = conn.execute(
                "SELECT test_path FROM test_verifies_ac "
                "WHERE workspace = ? ORDER BY test_path", (app,),
            ).fetchall()
        finally:
            conn.close()
        assert [r[0] for r in rows] == [
            "tests/test_other.py", "tests/test_real.py",
        ]

    def test_idempotent_on_rerun(self, tmp_path):
        from harness import story_state
        ws = tmp_path / "idempotent-ws"
        ws.mkdir()
        app = story_state.app_name_for_workspace(str(ws))
        conn = story_state.open_story_db()
        try:
            story_state.ensure_feature(conn, app, "f", name="F")
            keys = story_state.create_stories(conn, app, [{
                "title": "S", "feature": "f",
                "acceptance_criteria": ["AC"],
            }])
            sid = story_state.get_story(conn, app, keys[0])["id"]
            ac_key = story_state.list_acceptance_criteria(conn, app, sid)[0]["ac_key"]
        finally:
            conn.close()
        # First call: 1 insert
        a, _ = _persist_verifies_links(str(ws), {"tests/t.py": [ac_key]})
        # Second call: composite PK → no new insert
        b, _ = _persist_verifies_links(str(ws), {"tests/t.py": [ac_key]})
        assert a == 1 and b == 0


# ---------------------------------------------------------------------------
# v5 Phase 6 — non-agile mode skips the @verifies machinery
# ---------------------------------------------------------------------------

class TestNonAgileSkipsVerifiesGate:
    """Phase 6 contract: the @verifies marker prompt + gate + link
    writer are all gated on ``state["decomposition_enabled"]``.

    Non-agile runs (monolithic ``teane build`` / ``teane patch``)
    have no acceptance_criteria rows to cite, so enforcing the
    marker would force the LLM to fabricate fake STORY-N.AC-N keys
    to pass syntactic validation — every link insert would then be
    silently dropped with a warning. Skipping keeps the prompt
    honest and the log noise zero.
    """

    @pytest.mark.asyncio
    async def test_rule5_absent_from_user_prompt_in_non_agile(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n")
        gw = stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_calc.py\n"
            "content:\n"
            "from calc import add\n"
            "def test_add(): assert add(1, 2) == 3\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(0, "1 passed")
        await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["calc.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
            # decomposition_enabled deliberately False / unset.
        })
        sent = gw.dispatched[0]["messages"]
        joined = "\n".join(m.get("content", "") for m in sent if m.get("role") == "user")
        # RULE 5 must NOT appear in non-agile prompts.
        assert "@verifies" not in joined, (
            "non-agile prompt must not require @verifies markers"
        )

    @pytest.mark.asyncio
    async def test_markerless_test_accepted_in_non_agile(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        """The marker gate is skipped entirely — a test without a
        @verifies marker passes through to the sandbox and lands as
        ``status=passed``, NOT ``missing_verifies_marker``."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n")
        stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_calc.py\n"
            "content:\n"
            "from calc import add\n"
            "def test_add(): assert add(1, 2) == 3\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(0, "1 passed in 0.01s")
        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["calc.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
        })
        assert "compiler_errors" not in result
        assert result["node_state"]["test_generation"]["status"] == "passed"
        # Agile-only fields are absent in non-agile node_state.
        assert "verifies_links_inserted" not in result["node_state"]["test_generation"]
        assert "verifies_links_dropped" not in result["node_state"]["test_generation"]
        assert route_after_test_generation(result) == "lintgate_node"

    @pytest.mark.asyncio
    async def test_rule5_present_in_user_prompt_when_agile(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        """The agile path still emits RULE 5 — Phase 3 contract intact."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n")
        gw = stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_calc.py\n"
            "content:\n"
            "# @verifies: STORY-001.AC-1\n"
            "from calc import add\n"
            "def test_add(): assert add(1, 2) == 3\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(0, "1 passed")
        await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["calc.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
            "decomposition_enabled": True,
        })
        sent = gw.dispatched[0]["messages"]
        joined = "\n".join(m.get("content", "") for m in sent if m.get("role") == "user")
        assert "@verifies" in joined
        assert "STORY-003.AC-2" in joined  # canonical example in RULE 5


class TestStoryPreambleInjectedIntoTestGenPrompt:
    """Phase 6b: _build_story_preamble is now prepended to the test-gen
    user prompt alongside the change-request and arch-summary preambles
    so the LLM actually sees the AC keys it's expected to cite.

    In non-agile / no-current-story mode the preamble is empty, so the
    prompt picks up zero extra content (the bytes match the pre-Phase-6
    non-preamble form).
    """

    @pytest.mark.asyncio
    async def test_preamble_present_when_current_story_active(
        self, tmp_path, stub_sandbox, stub_gateway, monkeypatch,
    ):
        from harness import story_state
        # Seed an agile workspace with a real STORY-001 + AC.
        ws = tmp_path / "agile-preamble-ws"
        ws.mkdir()
        (ws / "pyproject.toml").write_text("[project]\nname='x'\n")
        (ws / "calc.py").write_text("def add(a, b): return a + b\n")
        db = tmp_path / "state.db"
        monkeypatch.setenv("TEANE_STATE_DB", str(db))
        app = story_state.app_name_for_workspace(str(ws))
        conn = story_state.open_story_db()
        try:
            story_state.ensure_feature(conn, app, "core", name="Core")
            story_state.create_stories(conn, app, [{
                "title": "Add two numbers", "feature": "core",
                "acceptance_criteria": ["add(1, 2) returns 3"],
            }])
        finally:
            conn.close()

        gw = stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_calc.py\n"
            "content:\n"
            "# @verifies: STORY-001.AC-1\n"
            "from calc import add\n"
            "def test_add(): assert add(1, 2) == 3\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(0, "1 passed")
        await run_test_generation({
            "workspace_path": str(ws),
            "modified_files": ["calc.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
            "decomposition_enabled": True,
            "current_story_id": "STORY-001",
        })
        sent = gw.dispatched[0]["messages"]
        joined = "\n".join(m.get("content", "") for m in sent if m.get("role") == "user")
        # Story preamble rendered with the AC key so the LLM can cite it.
        assert "STORY-001.AC-1" in joined
        assert "add(1, 2) returns 3" in joined

    @pytest.mark.asyncio
    async def test_batch_preamble_renders_when_current_story_cleared(
        self, tmp_path, stub_sandbox, stub_gateway, monkeypatch,
    ):
        """Phase 7 BUG #2 regression: story_loop_node clears
        current_story_id="" before routing into batch verification.
        Without the batch-scope fallback, test_generation would emit
        RULE 5 but no preamble, leaving the LLM with no AC keys to
        cite. The fallback must render ACs from every story patched
        in the current batch."""
        from harness import story_state
        ws = tmp_path / "batch-preamble-ws"
        ws.mkdir()
        (ws / "pyproject.toml").write_text("[project]\nname='x'\n")
        (ws / "calc.py").write_text("def add(a, b): return a + b\n")
        db = tmp_path / "state.db"
        monkeypatch.setenv("TEANE_STATE_DB", str(db))
        app = story_state.app_name_for_workspace(str(ws))
        conn = story_state.open_story_db()
        try:
            story_state.ensure_feature(conn, app, "core", name="Core")
            story_state.create_stories(conn, app, [
                {
                    "title": "Add",
                    "feature": "core",
                    "acceptance_criteria": ["add(1, 2) returns 3"],
                },
                {
                    "title": "Subtract",
                    "feature": "core",
                    "acceptance_criteria": ["sub(2, 1) returns 1"],
                },
            ])
        finally:
            conn.close()

        gw = stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_calc.py\n"
            "content:\n"
            "# @verifies: STORY-001.AC-1, STORY-002.AC-1\n"
            "from calc import add\n"
            "def test_add(): assert add(1, 2) == 3\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(0, "1 passed")
        await run_test_generation({
            "workspace_path": str(ws),
            "modified_files": ["calc.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
            "decomposition_enabled": True,
            # Mid-verification state: story_loop already cleared this.
            "current_story_id": "",
            "current_batch_id": 1,
            "batch_patched_story_keys": ["STORY-001", "STORY-002"],
        })
        sent = gw.dispatched[0]["messages"]
        joined = "\n".join(m.get("content", "") for m in sent if m.get("role") == "user")
        # Batch preamble must list BOTH stories' AC keys so the LLM
        # has real keys to cite in the @verifies marker.
        assert "Batch Scope:" in joined
        assert "STORY-001.AC-1" in joined
        assert "STORY-002.AC-1" in joined
        assert "add(1, 2) returns 3" in joined
        assert "sub(2, 1) returns 1" in joined

    @pytest.mark.asyncio
    async def test_preamble_empty_when_no_current_story(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n")
        gw = stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_calc.py\n"
            "content:\n"
            "from calc import add\n"
            "def test_add(): assert add(1, 2) == 3\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(0, "1 passed")
        await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["calc.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
            # no current_story_id, no decomposition_enabled
        })
        sent = gw.dispatched[0]["messages"]
        joined = "\n".join(m.get("content", "") for m in sent if m.get("role") == "user")
        # Story preamble renders the empty string when no story is set —
        # the prompt has no "Story Scope:" header.
        assert "Story Scope:" not in joined


# ---------------------------------------------------------------------------
# Fix 2c (2026-07-10): the test-gen format reminder must document all four
# patch block types. Prior to this fix only CREATE_FILE and INSERT_AT_BLOCK
# were listed, but the LLM sees REPLACE_BLOCK examples in the messages
# history from patching_node — mismatch caused iter 4 of session 44c5e194
# to emit 5 REPLACE_BLOCKs that all rejected as "unknown format".
# ---------------------------------------------------------------------------

class TestFormatReminderDocumentsAllBlockTypes:

    def test_reminder_lists_all_four_patch_ops(self):
        from harness.test_generation import _PROMPT_FORMAT_REMINDER_BASE
        assert "<<<CREATE_FILE>>>" in _PROMPT_FORMAT_REMINDER_BASE
        assert "<<<REPLACE_BLOCK>>>" in _PROMPT_FORMAT_REMINDER_BASE
        assert "<<<REWRITE_FILE>>>" in _PROMPT_FORMAT_REMINDER_BASE
        assert "<<<INSERT_AT_BLOCK>>>" in _PROMPT_FORMAT_REMINDER_BASE

    def test_reminder_explains_when_to_use_each_op(self):
        # The "CHOOSING THE RIGHT BLOCK:" section is what steers the LLM
        # away from REPLACE_BLOCK-when-file-is-empty (Fix 4 bait).
        from harness.test_generation import _PROMPT_FORMAT_REMINDER_BASE
        assert "CHOOSING THE RIGHT BLOCK" in _PROMPT_FORMAT_REMINDER_BASE
        assert "REWRITE_FILE" in _PROMPT_FORMAT_REMINDER_BASE
        assert "small" in _PROMPT_FORMAT_REMINDER_BASE.lower()


# ---------------------------------------------------------------------------
# Fix 3 (2026-07-10): missing @verifies marker is autofixed deterministically
# from the current story's AC keys rather than routed to LLM repair. Only
# markerless files WITHOUT usable story context still route to repair.
# ---------------------------------------------------------------------------

class TestVerifiesMarkerAutofix:

    def test_marker_line_for_python_uses_hash_lead(self):
        from harness.test_generation import _marker_line_for
        line = _marker_line_for("python", ["STORY-3.AC-1", "STORY-3.AC-2"])
        assert line == "# @verifies: STORY-3.AC-1, STORY-3.AC-2"

    def test_marker_line_for_typescript_uses_slash_lead(self):
        from harness.test_generation import _marker_line_for
        line = _marker_line_for("typescript", ["STORY-3.AC-1"])
        assert line == "// @verifies: STORY-3.AC-1"

    def test_marker_line_for_no_keys_returns_none(self):
        from harness.test_generation import _marker_line_for
        assert _marker_line_for("python", []) is None

    def test_marker_line_for_drops_malformed_keys(self):
        # Bad keys are silently filtered — the persist gate would drop
        # them downstream anyway, so autofix shouldn't waste I/O
        # writing them.
        from harness.test_generation import _marker_line_for
        assert _marker_line_for("python", ["bogus", "STORY-1.AC-2"]) == (
            "# @verifies: STORY-1.AC-2"
        )
        assert _marker_line_for("python", ["bogus", "also-bad"]) is None

    def test_prepend_marker_writes_at_top_of_file(self, tmp_path):
        from harness.test_generation import _prepend_verifies_marker
        f = tmp_path / "test_x.py"
        f.write_text("def test_x(): pass\n")
        assert _prepend_verifies_marker(
            str(f), "# @verifies: STORY-1.AC-1",
        ) is True
        body = f.read_text()
        assert body.startswith("# @verifies: STORY-1.AC-1\n")
        assert "def test_x()" in body

    def test_prepend_marker_respects_shebang(self, tmp_path):
        from harness.test_generation import _prepend_verifies_marker
        f = tmp_path / "run.py"
        f.write_text("#!/usr/bin/env python3\ndef test_x(): pass\n")
        _prepend_verifies_marker(str(f), "# @verifies: STORY-1.AC-1")
        lines = f.read_text().splitlines()
        assert lines[0] == "#!/usr/bin/env python3"
        assert lines[1] == "# @verifies: STORY-1.AC-1"

    def test_prepend_marker_idempotent_when_already_present(self, tmp_path):
        # Autofix is called from a loop; running it twice must not
        # duplicate the marker (and must not write the file again).
        from harness.test_generation import _prepend_verifies_marker
        f = tmp_path / "test_x.py"
        f.write_text("# @verifies: STORY-1.AC-1\ndef test_x(): pass\n")
        assert _prepend_verifies_marker(
            str(f), "# @verifies: STORY-1.AC-1",
        ) is True
        body = f.read_text()
        # Only one marker line in the file
        assert body.count("@verifies:") == 1


# ---------------------------------------------------------------------------
# Fix 5a (2026-07-10): the test-gen user prompt must include the current
# on-disk bytes of every existing test file the LLM might edit. Without
# this, REPLACE_BLOCK anchors are built from the LLM's stale mental model
# (root cause behind iter 4 of session 44c5e194).
# ---------------------------------------------------------------------------

class TestPreflightInjectionInTestGenPrompt:

    @pytest.mark.asyncio
    async def test_existing_test_file_body_appears_in_user_prompt(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        # Source file being tested this round.
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n")
        # A pre-existing test file that shares the conventional name —
        # the harness must show its current body to the LLM.
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_calc.py").write_text(
            "# @verifies: STORY-1.AC-1\n"
            "def test_add_returns_sum():\n"
            "    from calc import add\n"
            "    assert add(2, 3) == 5\n"
        )
        gw = stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_calc_extra.py\n"
            "content:\n"
            "def test_extra(): pass\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(0, "1 passed")
        await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["calc.py", "tests/test_calc.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
        })
        sent = gw.dispatched[0]["messages"]
        joined = "\n".join(
            m.get("content", "") for m in sent if m.get("role") == "user"
        )
        # Preflight section header appears
        assert "Current Content of Files You Need to Edit" in joined
        # And carries the ACTUAL test file body (not the LLM's memory of it)
        assert "test_add_returns_sum" in joined
        assert "assert add(2, 3) == 5" in joined
        # Line-numbered rendering (the `  N| ` prefix from _render_file...)
        assert "1| " in joined or "1|" in joined


# ---------------------------------------------------------------------------
# Fix 2a (2026-07-10) — POSITIVE path: zero-emit retry succeeds on second
# response. Confirms the counter split: test_generation_zero_emit=1,
# test_generation=1 (only the successful attempt is counted).
# ---------------------------------------------------------------------------

class TestZeroEmitRetrySucceeds:

    @pytest.mark.asyncio
    async def test_reprompt_then_valid_patch_lands_and_counters_split(
        self, tmp_path, stub_sandbox, monkeypatch,
    ):
        # Custom stub gateway that returns different content on each call.
        # First call: zero patch blocks. Second call: a valid patch.
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "foo.py").write_text("def foo(): return 1\n")

        class _MultiResponseGateway(_StubGateway):
            def __init__(self):
                super().__init__("")
                self._responses = [
                    "no patch blocks here — just prose",
                    "<<<CREATE_FILE>>>\n"
                    "file: tests/test_foo.py\n"
                    "content:\n"
                    "# @verifies: STORY-1.AC-1\n"
                    "def test_foo(): assert True\n"
                    "<<<END_CREATE_FILE>>>\n",
                ]

            async def dispatch(self, *, messages, role, budget_remaining_usd, **kwargs):
                self.dispatched.append({"messages": list(messages), "role": role})
                idx = min(len(self.dispatched) - 1, len(self._responses) - 1)
                return _StubResponse(self._responses[idx]), (
                    budget_remaining_usd - 0.001
                )

        from harness import graph as graph_mod
        gw = _MultiResponseGateway()
        graph_mod.set_gateway(gw)
        try:
            stub_sandbox(0, "1 passed")
            result = await run_test_generation({
                "workspace_path": str(tmp_path),
                "modified_files": ["foo.py"],
                "messages": [],
                "budget_remaining_usd": 1.5,
                "token_tracker": {},
            })
        finally:
            graph_mod.set_gateway(None)

        # Two dispatches: the first was the zero-emit re-prompt, the
        # second landed a real patch.
        assert len(gw.dispatched) == 2
        # The zero-emit counter recorded the single retry.
        assert result["loop_counter"]["test_generation_zero_emit"] == 1
        # The REAL iteration counter only advanced for the successful
        # attempt — that's Fix 2a's whole point.
        assert result["loop_counter"]["test_generation"] == 1
        # Second dispatch's messages must include the stronger contract
        # system message pushed after the zero-emit response.
        second_msgs = gw.dispatched[1]["messages"]
        stronger_prompt_hits = [
            m for m in second_msgs
            if m.get("role") == "system"
            and "zero PATCH blocks" in m.get("content", "")
        ]
        assert stronger_prompt_hits, (
            "second dispatch must carry the stronger re-prompt system "
            "message pushed after the first zero-emit response"
        )
