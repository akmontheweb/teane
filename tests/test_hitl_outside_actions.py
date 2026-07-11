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


class TestPhase7TraceabilityAndNonToolchainSymbols:
    """Phase 7 BUG #6 regressions: traceability_block trigger surfaces
    coverage-gap advice (not 'open failing files'); non-toolchain
    env_misconfig symbols get sensible advice (not 'install the
    missing package `test_generation_max_iterations`')."""

    def test_traceability_block_lists_audit_advice(self):
        state = _base_state()
        actions = _build_outside_harness_actions(state, "traceability_block")
        text = "\n".join(actions)
        assert "TRACEABILITY BLOCK" in text
        assert "requirement_keys" in text
        assert "@verifies" in text
        # Emergency bypass must be mentioned.
        assert "traceability.enforce" in text
        # Must NOT advise "open failing files in IDE".
        assert "open failing files" not in text.lower()

    def test_traceability_block_waterfall_workspace_uses_iso_vocabulary(self):
        """No decomposition_enabled flag → waterfall path. Remediation
        hint should cite the FR / NFR / US family, not the agile one."""
        state = _base_state()
        actions = _build_outside_harness_actions(state, "traceability_block")
        text = "\n".join(actions)
        assert "FR-NNN" in text
        assert "EPIC-NNN" not in text and "FEAT-NNN" not in text

    def test_traceability_block_agile_workspace_uses_safe_vocabulary(self):
        """decomposition_enabled=True → agile path. Remediation hint
        must surface EPIC/FEAT/STORY vocabulary, not waterfall FRs.
        This is the regression guard for the HITL where the operator
        was told to cite FR-NNN keys in an agile workspace whose spec
        only contained EPIC/FEAT/STORY identifiers."""
        state = _base_state(decomposition_enabled=True)
        actions = _build_outside_harness_actions(state, "traceability_block")
        text = "\n".join(actions)
        assert "EPIC-NNN" in text
        assert "FEAT-NNN" in text
        assert "STORY-NNN" in text
        # The waterfall remediation prose is "FR-NNN / NFR-XXX-NNN /
        # US-NN-NN" — checking for the full pattern avoids a false
        # positive on "STORY-NFR-NNN" which legitimately contains the
        # "FR-NNN" substring.
        assert "FR-NNN / NFR-XXX-NNN" not in text
        assert "US-NN-NN" not in text

    def test_llm_behavior_max_iterations_does_not_suggest_pip_install(self):
        state = _base_state(node_state={"llm_behavior_symbol": "test_generation_max_iterations"})
        actions = _build_outside_harness_actions(
            state, "llm_behavior:test_generation_max_iterations",
        )
        text = "\n".join(actions)
        # Must NOT tell the operator to pip install a counter name.
        assert "Install the missing tool/package" not in text
        assert "pip install test_generation_max_iterations" not in text
        # Must mention max_iterations as the config knob.
        assert "max_iterations" in text

    def test_env_misconfig_llm_api_key_suggests_env_var(self):
        state = _base_state(node_state={"env_misconfig_symbol": "llm_api_key"})
        actions = _build_outside_harness_actions(
            state, "env_misconfig:llm_api_key",
        )
        text = "\n".join(actions)
        assert "ANTHROPIC_API_KEY" in text or "OPENAI_API_KEY" in text
        assert "Install the missing tool/package" not in text

    def test_env_misconfig_no_source_files_suggests_workspace_inspection(self):
        state = _base_state(node_state={"env_misconfig_symbol": "no_source_files"})
        actions = _build_outside_harness_actions(
            state, "env_misconfig:no_source_files",
        )
        text = "\n".join(actions)
        assert "source" in text.lower()
        assert "Install the missing tool/package" not in text

    def test_env_misconfig_real_package_still_suggests_install(self):
        """Real package names (pytest, jest, mvn) should still get
        the install-package advice. Non-toolchain branch must not
        cannibalize the legitimate cases."""
        state = _base_state(node_state={"env_misconfig_symbol": "pytest"})
        actions = _build_outside_harness_actions(
            state, "env_misconfig:pytest",
        )
        text = "\n".join(actions)
        assert "Install the missing tool/package" in text
        assert "pytest" in text

    def test_llm_behavior_zero_emit_does_not_suggest_pip_install(self):
        """Bug A (2026-07-10, reclassified 2026-07-10 evening): the
        `test_generation_zero_emit` symbol first went through the
        toolchain branch and produced "pip install
        test_generation_zero_emit" — Fix 2a moved it under the
        _NON_TOOLCHAIN_ENV_SYMBOLS gate, but that was a hack (the code
        itself apologized for grouping LLM refusal with missing tools).
        Now routed under a dedicated ``llm_behavior:`` trigger family."""
        state = _base_state(
            node_state={"llm_behavior_symbol": "test_generation_zero_emit"},
        )
        actions = _build_outside_harness_actions(
            state, "llm_behavior:test_generation_zero_emit",
        )
        text = "\n".join(actions)
        # No nonsensical package-install text
        assert "Install the missing tool/package" not in text
        assert "pip install test_generation_zero_emit" not in text
        # The correct recovery hints are surfaced
        assert "max_zero_emit_reprompts" in text
        # And the root-cause pointer to check upstream patcher output
        assert "patches=N succeed=0" in text or "patcher" in text.lower()
