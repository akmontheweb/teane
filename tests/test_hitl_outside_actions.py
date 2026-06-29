"""Tests for the per-trigger outside-the-harness action checklist
surfaced in the HITL menu (``cli._build_outside_harness_actions``).

The escalation summary diagnoses; these actions prescribe. Each trigger
in the taxonomy should produce concrete steps the operator can take
outside the harness before resuming."""

from __future__ import annotations

from harness.cli import _build_outside_harness_actions


def _base_state(**overrides):
    state = {
        "workspace_path": "/tmp/ws",
        "build_command": "uv pip install --system -r requirements.txt && pytest",
        "sandbox_config": {"docker_image": "python:3.12-slim"},
        "session_id": "abc-123",
        "compiler_errors": [],
        "modified_files": [],
        "node_state": {},
    }
    state.update(overrides)
    return state


class TestEnvMisconfig:
    def test_env_misconfig_with_symbol_names_the_symbol(self):
        state = _base_state(
            node_state={"env_misconfig_symbol": "pytest"},
        )
        actions = _build_outside_harness_actions(state, "env_misconfig:pytest")
        text = "\n".join(actions).lower()
        assert "pytest" in text
        # Should mention the two operator-fixable places.
        assert "sandbox.docker_image" in "\n".join(actions)
        assert "build_command" in "\n".join(actions)

    def test_env_misconfig_without_symbol_still_gives_guidance(self):
        state = _base_state()
        actions = _build_outside_harness_actions(state, "env_misconfig")
        assert any("sandbox.docker_image" in a for a in actions)


class TestBudgetTriggers:
    def test_budget_exhausted_mentions_hard_cap_and_cheaper_model(self):
        state = _base_state()
        actions = _build_outside_harness_actions(state, "budget_exhausted")
        text = "\n".join(actions)
        assert "hard_cap_usd" in text
        assert "model_assignments" in text

    def test_budget_preflight_same_advice(self):
        state = _base_state()
        actions = _build_outside_harness_actions(state, "budget_preflight")
        assert any("hard_cap_usd" in a for a in actions)


class TestLlmSilent:
    def test_mentions_api_keys_and_model_id(self):
        state = _base_state()
        actions = _build_outside_harness_actions(state, "llm_silent")
        text = "\n".join(actions).lower()
        assert "api_key" in text or "api key" in text
        assert "model" in text


class TestSecurityFixLimit:
    def test_mentions_suppression_and_manual_rewrite(self):
        state = _base_state()
        actions = _build_outside_harness_actions(state, "security_fix_limit:2/2")
        text = "\n".join(actions).lower()
        assert "suppress" in text or "noqa" in text or "nosec" in text
        assert "manual" in text or "ide" in text


class TestZeroPatchLoop:
    def test_mentions_manual_edit_or_hint_options(self):
        state = _base_state()
        actions = _build_outside_harness_actions(state, "zero_patch_loop:3")
        text = "\n".join(actions)
        assert "[m]" in text
        assert "[e]" in text


class TestRepairLoopLimit:
    def test_surfaces_allowlist_rejections(self):
        state = _base_state(
            node_state={
                "allowlist_rejections": [
                    {"file": ".env.example"},
                    {"file": "prompts/foo.txt"},
                ],
            },
        )
        actions = _build_outside_harness_actions(state, "repair_loop_limit")
        text = "\n".join(actions)
        assert ".env.example" in text or "prompts/foo.txt" in text
        assert "patcher.root_files" in text or "patcher.allowed_paths" in text

    def test_surfaces_files_from_errors(self):
        state = _base_state(
            compiler_errors=[
                {"file": "server/main.py", "message": "boom"},
                {"file": "server/db.py", "message": "boom"},
            ],
        )
        actions = _build_outside_harness_actions(state, "repair_loop_limit")
        text = "\n".join(actions)
        assert "server/main.py" in text or "server/db.py" in text

    def test_surfaces_docker_image_when_present(self):
        state = _base_state(
            sandbox_config={"docker_image": "harness-builder:latest"},
        )
        actions = _build_outside_harness_actions(state, "persistent_build_failure")
        text = "\n".join(actions)
        assert "harness-builder:latest" in text

    def test_patch_failures_surface_manual_edit_path(self):
        state = _base_state(
            node_state={"patch_failures": [{"operation": "replace_block",
                                            "file": "x.py",
                                            "reason": "search miss"}]},
        )
        actions = _build_outside_harness_actions(state, "repair_loop_limit")
        text = "\n".join(actions).lower()
        assert "manual" in text or "ide" in text

    def test_no_specific_signal_falls_back_to_workspace_inspect(self):
        # No errors, no rejections, no patch failures, no docker image —
        # the catch-all should suggest inspecting the workspace.
        state = _base_state(sandbox_config={})
        actions = _build_outside_harness_actions(state, "repair_loop_limit")
        text = "\n".join(actions)
        assert "/tmp/ws" in text


class TestUniversalTail:
    def test_every_trigger_lists_resume_options(self):
        for trigger in [
            "env_misconfig:pytest",
            "budget_exhausted",
            "llm_silent",
            "security_fix_limit:2/2",
            "zero_patch_loop:3",
            "repair_loop_limit",
            "persistent_build_failure",
            "no_progress_failsafe",
            "unknown",
        ]:
            state = _base_state()
            actions = _build_outside_harness_actions(state, trigger)
            # The universal tail names every menu choice the operator
            # might use after fixing.
            text = "\n".join(actions)
            assert "[r]" in text, f"missing [r] for {trigger}"
            assert "[m]" in text, f"missing [m] for {trigger}"
            assert "[e]" in text, f"missing [e] for {trigger}"
            assert "[s]" in text, f"missing [s] for {trigger}"
            assert state["session_id"] in text, f"missing session_id for {trigger}"

    def test_modified_files_listed_when_small(self):
        state = _base_state(modified_files=["a.py", "b.py"])
        actions = _build_outside_harness_actions(state, "repair_loop_limit")
        text = "\n".join(actions)
        assert "a.py" in text and "b.py" in text

    def test_modified_files_omitted_when_huge(self):
        state = _base_state(modified_files=[f"f{i}.py" for i in range(50)])
        actions = _build_outside_harness_actions(state, "repair_loop_limit")
        text = "\n".join(actions)
        # 50 > 30 cap → list is suppressed to keep the menu readable.
        assert "f0.py" not in text


class TestUnknownTrigger:
    def test_unknown_falls_through_to_workspace_inspect(self):
        state = _base_state()
        actions = _build_outside_harness_actions(state, "unknown")
        text = "\n".join(actions)
        assert "/tmp/ws" in text
