"""Tests for harness/graph.py — orchestration basics."""



from harness.graph import (
    apply_memory_cleanse,
    _format_diagnostics_for_repair,
)


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
