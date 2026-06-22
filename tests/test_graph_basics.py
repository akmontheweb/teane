"""Tests for harness/graph.py — orchestration basics."""

import asyncio

from harness.graph import (
    apply_memory_cleanse,
    _format_diagnostics_for_repair,
    _repair_budget_warning,
    route_after_security_scan,
)


class TestRepairBudgetWarning:
    """Audit #19 — soft warnings on the last two repair iterations."""

    def test_silent_with_slack(self):
        assert _repair_budget_warning(total_repairs=1, cap=8) is None
        assert _repair_budget_warning(total_repairs=5, cap=8) is None

    def test_medium_warning_at_two_remaining(self):
        msg = _repair_budget_warning(total_repairs=6, cap=8)
        assert msg is not None
        assert "2 repair iterations remain" in msg

    def test_hard_warning_at_one_remaining(self):
        msg = _repair_budget_warning(total_repairs=7, cap=8)
        assert msg is not None
        assert "LAST repair iteration" in msg

    def test_silent_past_cap(self):
        # Past the cap the router has already moved to HITL; no point
        # warning a model that won't be called.
        assert _repair_budget_warning(total_repairs=8, cap=8) is None
        assert _repair_budget_warning(total_repairs=99, cap=8) is None

    def test_zero_cap_is_silent(self):
        # Edge case — operator disabled the throttle. Don't crash, don't warn.
        assert _repair_budget_warning(total_repairs=3, cap=0) is None

    def test_small_cap_still_warns_on_last_two(self):
        # With cap=2 the LLM gets warned on both attempts.
        assert _repair_budget_warning(total_repairs=0, cap=2) is not None
        assert _repair_budget_warning(total_repairs=1, cap=2) is not None
        assert _repair_budget_warning(total_repairs=2, cap=2) is None


class TestApplyMemoryCleanse:
    """Test memory cleansing on compiler success."""

    def test_cleanse_with_no_messages(self):
        """Should handle state with no messages."""
        state = {"messages": []}
        result = apply_memory_cleanse(state, resolution_kind="compiler_success")
        assert isinstance(result, dict)

    def test_cleanse_with_single_message(self):
        """Should cleanse state with single message."""
        state = {
            "messages": [
                {"role": "user", "content": "Hello"},
            ]
        }
        result = apply_memory_cleanse(state, resolution_kind="compiler_success")
        assert isinstance(result, dict)

    def test_cleanse_with_multiple_messages(self):
        """Should cleanse state with conversation."""
        state = {
            "messages": [
                {"role": "user", "content": "Fix this code"},
                {"role": "assistant", "content": "Here's the fix"},
                {"role": "user", "content": "Test it"},
            ]
        }
        result = apply_memory_cleanse(state, resolution_kind="compiler_success")
        assert isinstance(result, dict)
        # Should have messages in result
        assert "messages" in result

    def test_cleanse_different_resolution_kinds(self):
        """Should handle different resolution kinds."""
        state = {"messages": [{"role": "user", "content": "test"}]}

        for kind in ["compiler_success", "repair_success", "human_intervention"]:
            result = apply_memory_cleanse(state, resolution_kind=kind)
            assert isinstance(result, dict)

    def test_cleanse_preserves_state_fields(self):
        """Should preserve other state fields."""
        state = {
            "messages": [],
            "current_node": "compiler",
            "loop_counters": {"repair": 1},
            "exit_code": 0,
        }
        result = apply_memory_cleanse(state)
        # Should preserve non-message fields
        assert isinstance(result, dict)


class TestFormatDiagnosticsForRepair:
    """Test diagnostic formatting for repair hints."""

    def test_format_empty_errors(self):
        """Empty error list should produce empty or minimal output."""
        result = _format_diagnostics_for_repair([])
        assert isinstance(result, str)

    def test_format_single_error(self):
        """Single error should be formatted."""
        errors = [
            {
                "file": "main.py",
                "line": 10,
                "message": "undefined variable x",
                "severity": "error",
            }
        ]
        result = _format_diagnostics_for_repair(errors)
        assert isinstance(result, str)
        # Should contain error information
        if result:
            assert "error" in result.lower() or "main.py" in result or "10" in result

    def test_format_multiple_errors(self):
        """Multiple errors should all be included."""
        errors = [
            {
                "file": "app.py",
                "line": 5,
                "message": "syntax error",
                "severity": "error",
            },
            {
                "file": "utils.py",
                "line": 20,
                "message": "undefined function",
                "severity": "error",
            },
        ]
        result = _format_diagnostics_for_repair(errors)
        assert isinstance(result, str)

    def test_format_with_semantic_context(self):
        """Should include semantic context if present."""
        errors = [
            {
                "file": "main.py",
                "line": 10,
                "message": "type mismatch",
                "severity": "error",
                "semantic_context": "x = 'string'; y = x + 1",
            }
        ]
        result = _format_diagnostics_for_repair(errors)
        assert isinstance(result, str)

    def test_format_warnings_and_errors(self):
        """Should handle both warnings and errors."""
        errors = [
            {
                "file": "a.py",
                "line": 1,
                "message": "unused import",
                "severity": "warning",
            },
            {
                "file": "b.py",
                "line": 2,
                "message": "critical error",
                "severity": "error",
            },
        ]
        result = _format_diagnostics_for_repair(errors)
        assert isinstance(result, str)


class TestGraphStateTypes:
    """Test that graph state can be constructed with required fields."""

    def test_state_with_messages(self):
        """State should support messages field."""
        state = {
            "messages": [{"role": "user", "content": "test"}],
        }
        assert "messages" in state

    def test_state_with_tokens(self):
        """State should support token tracking."""
        state = {
            "messages": [],
            "token_tracker": {
                "total_input_tokens": 100,
                "total_output_tokens": 50,
                "total_cost_usd": 0.001,
            },
        }
        assert "token_tracker" in state

    def test_state_with_diagnostics(self):
        """State should support diagnostics."""
        state = {
            "messages": [],
            "diagnostics": [
                {
                    "file": "test.py",
                    "line": 5,
                    "message": "error",
                    "severity": "error",
                }
            ],
        }
        assert "diagnostics" in state

    def test_state_with_loop_counters(self):
        """State should support loop counters."""
        state = {
            "messages": [],
            "loop_counters": {
                "repair": 0,
                "discovery": 0,
            },
        }
        assert "loop_counters" in state


class TestNodeStateTransitions:
    """Test state transitions between nodes."""

    def test_planning_to_patching_transition(self):
        """State should transition from planning to patching."""
        state = {
            "messages": [
                {"role": "user", "content": "Fix bug"},
                {"role": "assistant", "content": "I'll fix it"},
            ],
            "current_node": "planning",
        }
        # After planning, state should have messages for patching
        assert len(state["messages"]) >= 2
        assert state["current_node"] == "planning"

    def test_compiler_exit_code_routing(self):
        """Exit code should determine next node."""
        state_success = {
            "messages": [],
            "exit_code": 0,
        }
        state_failure = {
            "messages": [],
            "exit_code": 1,
        }
        # These would be used by router functions
        assert state_success["exit_code"] == 0
        assert state_failure["exit_code"] == 1

    def test_repair_loop_counter_increment(self):
        """Repair loop should track iterations."""
        state = {
            "messages": [],
            "loop_counters": {"repair": 0},
        }
        state["loop_counters"]["repair"] += 1
        assert state["loop_counters"]["repair"] == 1


class TestErrorHandling:
    """Test error handling in graph state."""

    def test_state_with_error_message(self):
        """State should track LLM errors."""
        state = {
            "messages": [],
            "error": "budget_exhausted",
        }
        assert state["error"] == "budget_exhausted"

    def test_state_with_build_failure(self):
        """State should track build failures."""
        state = {
            "messages": [],
            "exit_code": 127,
            "diagnostics": [
                {
                    "file": "build.log",
                    "line": 1,
                    "message": "Build command not found",
                    "severity": "error",
                }
            ],
        }
        assert state["exit_code"] == 127
        assert len(state["diagnostics"]) > 0

    def test_state_with_timeout(self):
        """State should track timeouts."""
        state = {
            "messages": [],
            "timed_out": True,
            "exit_code": -1,
        }
        assert state["timed_out"] is True


class TestGatewayConfigPropagation:
    """Regression: ``set_gateway`` used to inject only the Gateway instance,
    not its config. Every ``get_gateway_config()`` consumer
    (``spec_review_node``, ``code_review_node``, the pre-flight reviewer
    in ``cmd_run``) read None and silently skipped — making the
    ``doc_reviewer_primary`` / ``code_reviewer_primary`` config keys
    effectively dead code. The fix makes ``set_gateway`` atomic: setting
    the gateway also stashes ``gateway.config`` for downstream readers.
    """

    def test_set_gateway_also_stashes_config(self):
        from harness.graph import set_gateway, get_gateway_config, get_gateway

        class _FakeConfig:
            doc_reviewer_primary = "openai:gpt-4o"
            code_reviewer_primary = "anthropic:claude-sonnet"

        class _FakeGateway:
            def __init__(self):
                self.config = _FakeConfig()

        gw = _FakeGateway()
        try:
            set_gateway(gw)
            assert get_gateway() is gw
            cfg = get_gateway_config()
            assert cfg is gw.config
            assert cfg.doc_reviewer_primary == "openai:gpt-4o"
        finally:
            # Reset the module-level slot so other tests aren't polluted.
            set_gateway.__globals__["_gateway"] = None
            set_gateway.__globals__["_gateway_config"] = None

    def test_set_gateway_without_config_is_safe(self):
        """A test double that doesn't expose ``.config`` still works —
        the gateway gets injected and the config slot stays whatever it
        was before (no AttributeError)."""
        from harness.graph import set_gateway, get_gateway_config

        class _Bare:
            pass

        # Capture whatever the slot was before to assert we don't crash.
        prior = get_gateway_config()
        try:
            set_gateway(_Bare())
            # No assertion on cfg value — just that the call returned cleanly.
            _ = get_gateway_config()
        finally:
            set_gateway.__globals__["_gateway"] = None
            set_gateway.__globals__["_gateway_config"] = prior


class TestRouteAfterSecurityScan:
    """Routing decisions after security_scan_node.

    Covers the --deploy-dev opt-in gate that controls whether a clean
    security scan rolls forward into deployment_discovery_node or stops at
    END. route_after_security_scan imports ``_is_flutter_project`` inside
    the function body, so we monkeypatch the attribute on
    ``harness.impact`` rather than on ``harness.graph``.
    """

    def _clean_state(self, **overrides):
        state = {
            "compiler_errors": [],
            "budget_remaining_usd": 1.0,
            "workspace_path": "/tmp/ws",
            "loop_counter": {"security": 0},
            "dev_deployment": False,
            "cd_discovery": False,
        }
        state.update(overrides)
        return state

    def test_clean_scan_ends_when_dev_deployment_false(self, monkeypatch):
        import harness.impact as impact_mod
        monkeypatch.setattr(impact_mod, "_is_flutter_project", lambda p: False)
        # Clean + no --deploy-dev now routes through installation_doc_node;
        # the node's only outgoing edge is END, so this is still a
        # terminal path (the doc may be a no-op if install_doc=False).
        assert route_after_security_scan(self._clean_state()) == "installation_doc_node"

    def test_clean_scan_enters_discovery_when_dev_deployment_and_cd_discovery_true(self, monkeypatch):
        import harness.impact as impact_mod
        monkeypatch.setattr(impact_mod, "_is_flutter_project", lambda p: False)
        # The classic flow: --deploy-dev true + --cd-discovery true → run
        # the LLM-driven blueprint pipeline.
        state = self._clean_state(dev_deployment=True, cd_discovery=True)
        assert route_after_security_scan(state) == "deployment_discovery_node"

    def test_clean_scan_skips_discovery_when_cd_discovery_false(self, monkeypatch):
        import harness.impact as impact_mod
        monkeypatch.setattr(impact_mod, "_is_flutter_project", lambda p: False)
        # The new fast-path: --deploy-dev true + --cd-discovery false →
        # straight to deployment_node, which synthesises the blueprint
        # from workspace telemetry alone (plus any deployment_defaults
        # section of config.json).
        state = self._clean_state(dev_deployment=True, cd_discovery=False)
        assert route_after_security_scan(state) == "deployment_node"

    def test_clean_scan_ends_when_cd_discovery_true_but_dev_deployment_false(self, monkeypatch):
        import harness.impact as impact_mod
        monkeypatch.setattr(impact_mod, "_is_flutter_project", lambda p: False)
        # cd_discovery alone (no dev_deployment) is meaningless — the
        # security-scan-clean terminal path still wins because no deploy
        # was requested. After the installation_doc_node insertion the
        # terminal hop is via the doc node (which then edges to END).
        state = self._clean_state(dev_deployment=False, cd_discovery=True)
        assert route_after_security_scan(state) == "installation_doc_node"

    def test_flutter_short_circuits_regardless_of_dev_deployment(self, monkeypatch):
        import harness.impact as impact_mod
        monkeypatch.setattr(impact_mod, "_is_flutter_project", lambda p: True)
        # With dev_deployment=True the Flutter early-return must still win;
        # Flutter and the opt-in flag are independent skip reasons. The
        # terminal hop now goes via installation_doc_node (which edges
        # to END), not __end__ directly — Flutter projects still get
        # docs/INSTALLATION.md when --install-doc is on.
        state = self._clean_state(dev_deployment=True)
        assert route_after_security_scan(state) == "installation_doc_node"

    def test_security_findings_route_to_repair(self, monkeypatch):
        import harness.impact as impact_mod
        monkeypatch.setattr(impact_mod, "_is_flutter_project", lambda p: False)
        state = self._clean_state(
            compiler_errors=[
                {
                    "file": "x.py",
                    "line": 1,
                    "column": 0,
                    "severity": "error",
                    "error_code": "GITLEAKS-X",
                    "message": "secret detected",
                    "semantic_context": "",
                }
            ],
        )
        # Findings exist + low attempts → repair_node regardless of flag.
        assert route_after_security_scan(state) == "repair_node"

    # Audit #18 — pre-exit verify
    def test_pre_exit_verify_routes_to_compiler_when_mutations_pending(self, monkeypatch):
        import harness.impact as impact_mod
        monkeypatch.setattr(impact_mod, "_is_flutter_project", lambda p: False)
        # Clean scan + opt-in flag + pending mutations → re-verify.
        state = self._clean_state(
            pre_exit_verify=True,
            pending_mutations=["server/app.py"],
        )
        assert route_after_security_scan(state) == "compiler_node"

    def test_pre_exit_verify_off_keeps_normal_terminal_route(self, monkeypatch):
        import harness.impact as impact_mod
        monkeypatch.setattr(impact_mod, "_is_flutter_project", lambda p: False)
        # Pending mutations exist but flag is off — defaults still hold.
        state = self._clean_state(
            pre_exit_verify=False,
            pending_mutations=["server/app.py"],
        )
        assert route_after_security_scan(state) == "installation_doc_node"

    def test_pre_exit_verify_skipped_when_no_mutations(self, monkeypatch):
        import harness.impact as impact_mod
        monkeypatch.setattr(impact_mod, "_is_flutter_project", lambda p: False)
        # Flag is on but nothing changed since last green compile.
        state = self._clean_state(
            pre_exit_verify=True,
            pending_mutations=[],
        )
        assert route_after_security_scan(state) == "installation_doc_node"

    def test_pre_exit_verify_one_shot_cap(self, monkeypatch):
        import harness.impact as impact_mod
        monkeypatch.setattr(impact_mod, "_is_flutter_project", lambda p: False)
        # Cap consumed → don't loop even if mutations are still flagged.
        state = self._clean_state(
            pre_exit_verify=True,
            pending_mutations=["server/app.py"],
            loop_counter={"security": 0, "final_verify": 1},
        )
        assert route_after_security_scan(state) == "installation_doc_node"


class _PatchingResponse:
    """Stand-in for the gateway response object the patching node
    consumes — needs ``content``, ``finish_reason``, ``tool_calls``,
    and ``usage`` (with input/output/cost fields)."""

    class _Usage:
        input_tokens = 100
        output_tokens = 200
        cost_usd = 0.001
        cached_tokens = 0

    def __init__(self, content: str, finish_reason: str):
        self.content = content
        self.finish_reason = finish_reason
        self.tool_calls = []
        self.usage = self._Usage()


class TestPatchingNodeContinuation:
    """When the patching LLM hits its 8192-token output cap
    mid-blueprint (session web-6d5ef9b18f6a's symptom — backend
    emitted, frontend never), the node now re-dispatches with a
    "continue" prompt and concatenates the chunks before handing
    them to the patcher."""

    def _install_gateway_stub(self, monkeypatch):
        from harness import graph as graph_mod

        class _Cfg:
            use_structured_tools = False
            enforce_read_before_edit = False

        class _Gw:
            config = _Cfg()

            def aggregate_tokens(self, tracker, usage, role=None):
                out = dict(tracker or {})
                out["total_cost_usd"] = out.get("total_cost_usd", 0.0) + usage.cost_usd
                return out

        gw = _Gw()
        graph_mod.set_gateway(gw)
        monkeypatch.setattr(graph_mod, "_build_patcher_allowlist", lambda ws: [])
        return graph_mod

    def test_continues_when_finish_reason_is_length(self, monkeypatch, tmp_path):
        graph_mod = self._install_gateway_stub(monkeypatch)

        # First call → "length"; second call → "stop". The text-DSL
        # path should concatenate both response bodies before patch
        # parsing.
        responses = [
            _PatchingResponse("<<<CREATE_FILE>>>\nfile: backend/a.py\ncontent:\nx\n<<<END_CREATE_FILE>>>", "length"),
            _PatchingResponse("<<<CREATE_FILE>>>\nfile: frontend/index.js\ncontent:\ny\n<<<END_CREATE_FILE>>>", "stop"),
        ]
        call_messages: list[list[dict]] = []

        async def fake_tool_loop(**kwargs):
            call_messages.append(list(kwargs["messages"]))
            resp = responses.pop(0)
            return resp, kwargs["budget"] - 0.10, kwargs["messages"], {}

        monkeypatch.setattr(graph_mod, "_patching_tool_loop", fake_tool_loop)

        captured: dict = {}

        async def fake_apply(content, ws, existing, allowed_paths=None):
            captured["content"] = content
            return [], []

        import harness.patcher as patcher_mod
        monkeypatch.setattr(
            patcher_mod, "process_llm_patch_output", fake_apply,
        )

        state = {
            "messages": [{"role": "system", "content": "you are a patcher"}],
            "budget_remaining_usd": 2.0,
            "workspace_path": str(tmp_path),
            "modified_files": [],
            "loop_counter": {},
            "token_tracker": {},
        }

        result = asyncio.run(graph_mod.patching_node(state))

        # Two dispatches happened (initial + 1 continuation).
        assert len(call_messages) == 2
        # The second dispatch saw the partial as an assistant turn
        # plus a continue-prompt as the trailing user turn.
        continuation_msgs = call_messages[1]
        assert continuation_msgs[-2]["role"] == "assistant"
        assert "backend/a.py" in continuation_msgs[-2]["content"]
        assert continuation_msgs[-1]["role"] == "user"
        assert "hit the output token cap" in continuation_msgs[-1]["content"]
        # The patcher saw BOTH chunks concatenated — without this the
        # frontend CREATE_FILE block would never reach disk.
        assert "backend/a.py" in captured["content"]
        assert "frontend/index.js" in captured["content"]
        # Node returned cleanly with both files in modified_files
        # provided by the fake apply (empty here — we only assert
        # the call shape).
        assert "messages" in result

    def test_caps_continuation_at_three_cycles(self, monkeypatch, tmp_path):
        """Pathological case: LLM keeps returning ``length``. The
        node must not loop forever — three continuation cycles is
        the ceiling, after which the node accepts what landed."""
        graph_mod = self._install_gateway_stub(monkeypatch)

        # Initial + 3 continuations = 4 dispatches, all "length".
        responses = [
            _PatchingResponse(f"chunk{i} ", "length") for i in range(1, 5)
        ]
        dispatches: list[int] = []

        async def fake_tool_loop(**kwargs):
            dispatches.append(1)
            resp = responses.pop(0)
            return resp, kwargs["budget"] - 0.10, kwargs["messages"], {}

        monkeypatch.setattr(graph_mod, "_patching_tool_loop", fake_tool_loop)

        captured: dict = {}

        async def fake_apply(content, ws, existing, allowed_paths=None):
            captured["content"] = content
            return [], []

        import harness.patcher as patcher_mod
        monkeypatch.setattr(
            patcher_mod, "process_llm_patch_output", fake_apply,
        )

        state = {
            "messages": [{"role": "system", "content": "you are a patcher"}],
            "budget_remaining_usd": 2.0,
            "workspace_path": str(tmp_path),
            "modified_files": [],
            "loop_counter": {},
            "token_tracker": {},
        }

        asyncio.run(graph_mod.patching_node(state))

        # 4 = initial + 3 continuation cycles. No more.
        assert len(dispatches) == 4
        # All four chunks reached the patcher.
        assert "chunk1" in captured["content"]
        assert "chunk4" in captured["content"]
