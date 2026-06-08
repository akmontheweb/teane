"""
CLI entry point, subcommand routing, HITL interactive menu loop, and configuration discovery.

Provides the following commands:
    harness run     — Primary execution entry point. Runs the full agent graph.
    harness resume  — Resume a crashed/interrupted session from its checkpoint.
    harness status  — Read-only inspection of a checkpointed session.
    harness purge   — Manually wipe all checkpoint data.

Use `harness -h` or `harness <command> -h` for detailed help on each subcommand.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Optional

# Configure logging for the CLI
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("harness.cli")


# ---------------------------------------------------------------------------
# 1. Configuration Discovery
# ---------------------------------------------------------------------------

def discover_config(workspace_path: str) -> dict[str, Any]:
    """
    Hierarchical configuration discovery.

    Priority order:
        1. .harness_config.json in the workspace root (if it exists)
        2. ~/.harness/config.json (user-global defaults)
        3. harness/cli.json (shipped fallback defaults)

    Returns a merged configuration dictionary.
    """
    config: dict[str, Any] = _get_default_config()

    # Layer 2: User-global config
    global_config_path = os.path.expanduser("~/.harness/config.json")
    if os.path.isfile(global_config_path):
        try:
            with open(global_config_path, "r", encoding="utf-8") as f:
                global_config = json.load(f)
            _validate_config_keys(global_config, global_config_path)
            _deep_merge(config, global_config)
            logger.debug("[cli] Merged global config from %s", global_config_path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(
                "[cli] Failed to parse global config at %s: %s\n"
                "  This file must contain valid JSON. The harness cannot proceed without it.\n"
                "  Fix the JSON syntax in this file and re-run.",
                global_config_path, exc,
            )

    # Layer 1: Workspace-local config (highest priority)
    workspace_config_path = os.path.join(workspace_path, ".harness_config.json")
    if not os.path.isfile(workspace_config_path):
        # Auto-generate from global config + fallback defaults.
        fallback = _get_default_config()
        _generate_workspace_config(workspace_config_path, config, fallback)

    if os.path.isfile(workspace_config_path):
        try:
            with open(workspace_config_path, "r", encoding="utf-8") as f:
                workspace_config = json.load(f)
            _validate_config_keys(workspace_config, workspace_config_path)
            _deep_merge(config, workspace_config)
            logger.debug("[cli] Merged workspace config from %s", workspace_config_path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[cli] Failed to read workspace config: %s", exc)

    return config


def _get_default_config() -> dict[str, Any]:
    """Return fallback configuration loaded from harness/cli.json.

    Models are defined in ~/.harness/config.json (global, shared across projects).
    Per-project model routing lives in .harness_config.json.
    Fallback defaults live in harness/cli.json (shipped with the package).
    Users can edit cli.json instead of modifying Python source code.
    """
    cli_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cli.json")
    if os.path.isfile(cli_json_path):
        try:
            with open(cli_json_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            logger.debug("[cli] Loaded fallback defaults from %s", cli_json_path)
            # Remove comment keys (keys starting with _)
            return {k: v for k, v in config.items() if not k.startswith("_")}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[cli] Failed to read cli.json fallback defaults: %s. Using hardcoded defaults.", exc)
    else:
        logger.warning("[cli] cli.json not found at %s. Using hardcoded defaults.", cli_json_path)

    # Absolute fallback (should never be reached if cli.json is shipped correctly)
    return {
        "build_command": "make build",
        "allow_network": False,
        "sandbox": {
            "backend": "auto",
            "docker_image": "ubuntu:22.04",
            "docker_memory_limit": "512m",
            "docker_cpu_limit": "1.0",
            "docker_pids_limit": 100,
            "readonly_cache_mounts": [
                "~/.cache/pip",
                "~/.npm",
                "~/.cache/go-build",
                "~/.cargo/registry",
            ],
            "timeout_seconds": 300,
            "pgid_kill_on_timeout": True,
        },
        "token_budget": {
            "hard_cap_usd": 2.00,
            "context_window_threshold_pct": 0.85,
        },
        "node_throttle": {
            "max_patch_repair_iterations": 3,
        },
        "models": {},
        "model_routing": {
            "planning_primary": "",
            "planning_mode": "thinking_max",
            "planning_fallback": "",
            "patching_primary": "",
            "patching_mode": "non_thinking",
            "repair_primary": "",
            "repair_fallback": "",
            "repair_mode": "thinking",
            "ollama_local_model": "",
            "ollama_local_backup": "",
            "force_local_only": False,
        },
        "persistence": {
            "db_path": "~/.harness/checkpoints.db",
            "ttl_days": 30,
        },
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively merge ``override`` into ``base`` in-place.

    Merge rules:
      - dict + dict  → recursive merge (existing behavior)
      - list + list  → concatenate + dedupe (preserves base order, then
        appends any new items from override). This means a partial
        workspace override (e.g. one extra cache mount) doesn't wipe
        the global defaults. To explicitly clear a base list, pass
        ``null`` (None) in the override.
      - any + None   → clears the base value (lets users opt out of
        defaults without having to know their values).
      - otherwise    → override wins.
    """
    for key, value in override.items():
        if value is None:
            base[key] = None
            continue
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        elif key in base and isinstance(base[key], list) and isinstance(value, list):
            # Concatenate + dedupe, preserving first-seen order.
            seen: set[Any] = set()
            merged: list[Any] = []
            for item in list(base[key]) + list(value):
                key_item = item if isinstance(item, (str, int, float, bool, tuple)) else repr(item)
                if key_item in seen:
                    continue
                seen.add(key_item)
                merged.append(item)
            base[key] = merged
        else:
            base[key] = value


# Top-level keys the harness knows about. Anything outside this set in a
# user-provided config is almost certainly a typo (e.g. "model_routin").
# Add new keys here when wiring new config sections.
_KNOWN_TOP_LEVEL_KEYS = frozenset({
    "build_command", "allow_network", "sandbox", "token_budget",
    "node_throttle", "models", "model_routing", "persistence",
    "manifest_file", "redaction", "security", "skills", "deployment",
    "speculative", "impact", "lintgate", "logging",
})


def _validate_config_keys(config: dict[str, Any], source_label: str) -> None:
    """
    Warn on top-level config keys that aren't in the known set. Catches
    typos like 'model_routin' silently no-op-ing. Heuristic-only: doesn't
    block load, just logs an actionable warning with the closest match.
    """
    import difflib
    for key in config.keys():
        if key.startswith("_"):
            continue  # comment keys
        if key not in _KNOWN_TOP_LEVEL_KEYS:
            suggestion = difflib.get_close_matches(key, _KNOWN_TOP_LEVEL_KEYS, n=1, cutoff=0.6)
            hint = f" (did you mean '{suggestion[0]}'?)" if suggestion else ""
            logger.warning(
                "[cli] Unknown config key '%s' in %s%s — this entry will be ignored. "
                "Known top-level keys: %s",
                key, source_label, hint, ", ".join(sorted(_KNOWN_TOP_LEVEL_KEYS)),
            )


def _generate_workspace_config(
    workspace_config_path: str,
    global_config: dict[str, Any],
    fallback_config: dict[str, Any],
) -> None:
    """
    Auto-generate a .harness_config.json in the workspace when one is missing.

    Builds the workspace config by extracting relevant fields from the
    global config (~/.harness/config.json) and filling in remaining fields
    from the built-in fallback defaults (harness/cli.json).
    """
    # Fields to pull from global config
    workspace_config: dict[str, Any] = {
        "_comment": (
            "Auto-generated .harness_config.json. Generated because no per-project "
            "configuration was found in this workspace. Values sourced from "
            "~/.harness/config.json (global) and harness/cli.json (built-in defaults). "
            "Modify this file to customize per-project settings."
        ),
        "build_command": fallback_config.get("build_command", "make build"),
        "allow_network": fallback_config.get("allow_network", False),
        "sandbox": fallback_config.get("sandbox", {}),
        "token_budget": global_config.get("token_budget", fallback_config.get("token_budget", {})),
        "node_throttle": fallback_config.get("node_throttle", {}),
        "models": global_config.get("models", fallback_config.get("models", {})),
        "model_routing": global_config.get("model_routing", fallback_config.get("model_routing", {})),
        "persistence": global_config.get("persistence", fallback_config.get("persistence", {})),
        "languages": fallback_config.get("languages", {}),
    }

    # Strip _comment keys from nested dicts inherited from global config.
    for key in ("token_budget", "models", "model_routing", "persistence"):
        if isinstance(workspace_config.get(key), dict):
            workspace_config[key] = {
                k: v for k, v in workspace_config[key].items() if not k.startswith("_")
            }

    try:
        with open(workspace_config_path, "w", encoding="utf-8") as f:
            json.dump(workspace_config, f, indent=4)
        logger.warning(
            "[cli] .harness_config.json not found in workspace. "
            "Auto-generated one from global configuration + defaults. "
            "Written to: %s",
            workspace_config_path,
        )
    except OSError as exc:
        logger.error(
            "[cli] Failed to auto-generate .harness_config.json at %s: %s",
            workspace_config_path, exc,
        )


def resolve_build_command(cli_build_cmd: Optional[str], config: dict[str, Any]) -> str:
    """
    Resolve the build command using hierarchical discovery:
        1. CLI flag --build-cmd (if provided)
        2. .harness_config.json 'build_command' key
        3. Default from cli.json: 'make build'
    """
    if cli_build_cmd:
        logger.info("[cli] Using build command from CLI flag: %s", cli_build_cmd)
        return cli_build_cmd
    config_cmd = config.get("build_command", "")
    if config_cmd:
        logger.info("[cli] Using build command from config: %s", config_cmd)
        return config_cmd
    fallback = "make build"
    logger.info("[cli] No build command configured. Using default: %s", fallback)
    return fallback


# ---------------------------------------------------------------------------
# 2. HITL Interactive Menu Loop
# ---------------------------------------------------------------------------

def _gatekeeper_auto_approves() -> bool:
    """
    True when the gatekeeper should skip interactive approval — set in CI
    or when the user opted in via HARNESS_AUTO_APPROVE, or when stdin is
    not a TTY (a piped invocation has no way to answer the prompt).

    Unlike the deploy preview gate (which fails closed on non-TTY because
    LLM-generated containers are about to launch), the spec/architecture
    gatekeeper has lower blast radius — a non-TTY here just means CI, so
    auto-approve is safe.
    """
    return (
        os.environ.get("CI", "").lower() == "true"
        or os.environ.get("HARNESS_AUTO_APPROVE", "").lower() == "true"
        or not sys.stdin.isatty()
    )


def human_gatekeeper_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Adaptive three-phase HITL gatekeeper node.
    
    Inspects state["current_gate"] and presents a phase-specific review menu:
        - REQUIREMENTS: Review SPEC_REQUIREMENTS.md
        - ARCHITECTURE: Review SPEC_ARCHITECTURE.md
        - DEPLOYMENT: Review DEPLOYMENT_BLUEPRINT.md
    
    Options per phase:
        [a] Approve → Proceed to next phase
        [e] Refine → Capture feedback, append to messages, loop back to generator
        [m] Manual → Pause for IDE edits, read updated file from disk
    
    Returns state update with routing signal in node_state.gatekeeper_action.
    """
    gate = state.get("current_gate", "")
    workspace = state.get("workspace_path", os.getcwd())
    messages = list(state.get("messages", []))
    loop_counter = state.get("loop_counter", {})
    loop_counter = dict(loop_counter)
    gate_attempts_key = f"gate_{gate.lower()}"
    attempt = loop_counter.get(gate_attempts_key, 0) + 1
    loop_counter[gate_attempts_key] = attempt

    # Determine which file to show
    if gate == "REQUIREMENTS":
        spec_path = state.get("spec_requirements_path", os.path.join(workspace, "docs", "SPEC_REQUIREMENTS.md"))
        gate_label = "REQUIREMENTS"
        gate_desc = "Requirements Specification"
        file_label = "SPEC_REQUIREMENTS.md"
        next_phase = "Architecture Specification"
    elif gate == "ARCHITECTURE":
        spec_path = state.get("spec_architecture_path", os.path.join(workspace, "docs", "SPEC_ARCHITECTURE.md"))
        gate_label = "ARCHITECTURE"
        gate_desc = "Architecture Specification"
        file_label = "SPEC_ARCHITECTURE.md"
        next_phase = "Code Generation & Patching"
    elif gate == "DEPLOYMENT":
        spec_path = state.get("deployment_blueprint_path", os.path.join(workspace, "docs", "DEPLOYMENT_BLUEPRINT.md"))
        gate_label = "DEPLOYMENT"
        gate_desc = "Deployment Blueprint"
        file_label = "DEPLOYMENT_BLUEPRINT.md"
        next_phase = "Container Deployment"
    else:
        logger.warning("[gatekeeper] Unknown gate: %s. Proceeding.", gate)
        return {"node_state": {"gatekeeper_action": "approve", "current_gate": gate}}

    # Non-interactive auto-approval. The spec lists CI / HARNESS_AUTO_APPROVE
    # as supported, but the gatekeeper was previously blocking on input()
    # even when those were set — making CI runs hang forever waiting on
    # stdin. Honor the env vars here as well as a non-TTY stdin.
    if _gatekeeper_auto_approves():
        logger.info(
            "[gatekeeper] %s auto-approved (non-interactive: CI / HARNESS_AUTO_APPROVE / no TTY).",
            gate_label,
        )
        return {
            "messages": messages,
            "loop_counter": loop_counter,
            "node_state": {"gatekeeper_action": "approve", "current_gate": gate},
        }

    while True:
        spec_content = ""
        spec_size = 0
        if os.path.isfile(spec_path):
            try:
                with open(spec_path, "r", encoding="utf-8") as f:
                    spec_content = f.read()
                spec_size = len(spec_content)
            except OSError:
                pass

        print()
        print("=" * 72)
        print(f"[HITL GATE: {gate_label}] — {gate_desc}")
        print(f"  File: {spec_path}")
        print(f"  Size: {spec_size:,} characters")
        print(f"  Attempt: {attempt}")
        print("=" * 72)
        print()

        if gate == "REQUIREMENTS":
            print(f"Requirements written to {file_label}. Please review the specification.")
            print("Options:")
            print(f"  [a] Approve & Proceed to {next_phase}")
            print("  [e] Refine via text feedback")
            print("  [m] Pause for manual local edits in IDE")
            print("  [s] Save & Quit (resume later)")
        elif gate == "ARCHITECTURE":
            print(f"Technical layout blueprints written to {file_label}. Please review module boundaries.")
            print("Options:")
            print("  [a] Approve & Begin Coding/Patching")
            print("  [e] Refine layout parameters")
            print("  [m] Pause for manual edits")
            print("  [s] Save & Quit (resume later)")
        elif gate == "DEPLOYMENT":
            print(f"Application fully compiled. Docker Composition written to {file_label}.")
            print("Please review container network bridges and volumes before firing.")
            print("Options:")
            print(f"  [a] Approve & Execute Infrastructure {next_phase}")
            print("  [e] Refine variables")
            print("  [m] Pause for manual edits")
            print("  [s] Save & Quit (resume later)")
        print()

        try:
            choice = input(f"[HITL:{gate_label}] Select action [a/e/m/s]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[Gatekeeper] Input interrupted. Aborting.")
            sys.exit(1)

        if choice == "a":
            logger.info("[gatekeeper] %s approved by developer.", gate_label)
            return {
                "messages": messages,
                "loop_counter": loop_counter,
                "node_state": {"gatekeeper_action": "approve", "current_gate": gate},
            }

        elif choice == "e":
            try:
                notes = input(f"[Refine:{gate_label}] Enter additional notes/feedback:\n").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Refine] Input interrupted.")
                continue

            if not notes:
                print("[Refine] No notes provided. Returning to menu.")
                continue

            # Append feedback to messages as a user instruction
            messages.append({"role": "user", "content": f"[HITL Feedback - {gate_label}]: {notes}"})

            # Reset loop counter to give the generator a fresh attempt
            loop_counter["patching"] = 0
            loop_counter["repair"] = 0
            loop_counter["compiler"] = 0
            loop_counter["total_repairs"] = 0

            logger.info("[gatekeeper] %s refine requested: %d chars of feedback.", gate_label, len(notes))
            return {
                "messages": messages,
                "loop_counter": loop_counter,
                "node_state": {"gatekeeper_action": "refine", "current_gate": gate},
            }

        elif choice == "m":
            print(f"[Manual] Edit the file at: {spec_path}")
            print("[Manual] Make your changes in your editor (VS Code, Cursor, etc.).")
            try:
                input("[Manual] Press Enter when you are done editing... ")
            except (EOFError, KeyboardInterrupt):
                print("\n[Manual] Input interrupted. Reading current file state.")

            # Reload the manually edited file into messages[0] (system prompt)
            if os.path.isfile(spec_path):
                try:
                    with open(spec_path, "r", encoding="utf-8") as f:
                        updated_spec = f.read()
                    # Update messages[0] with the manually edited spec
                    if messages:
                        messages[0] = {"role": "system", "content": updated_spec}
                    logger.info("[gatekeeper] %s manual edits confirmed (%d chars).", gate_label, len(updated_spec))
                except OSError:
                    logger.warning("[gatekeeper] Failed to read manually edited file.")

            return {
                "messages": messages,
                "loop_counter": loop_counter,
                "node_state": {"gatekeeper_action": "manual", "current_gate": gate},
            }

        elif choice == "s":
            session_id = state.get("session_id", "")
            print()
            print("=" * 60)
            print("Session saved to checkpoint.")
            print(f"Resume later with:")
            print(f"  harness resume --session-id {session_id}")
            if workspace and workspace != os.getcwd():
                print(f"  harness resume --session-id {session_id} -r {workspace}")
            print("=" * 60)
            print()
            logger.info("[gatekeeper] %s suspended by developer. Session: %s", gate_label, session_id)
            return {
                "messages": messages,
                "loop_counter": loop_counter,
                "node_state": {"gatekeeper_action": "suspend", "current_gate": gate},
            }

        else:
            print(f"[Gatekeeper] Unknown option: '{choice}'. Please choose a, e, m, or s.")


def discovery_interview_loop(state: dict[str, Any]) -> dict[str, Any]:
    """
    Multi-question streaming interface for exhaustive discovery phases.
    
    Reads state["discovery_questions"] (JSON from requirements/architecture discovery node),
    displays questions grouped by engineering modules, collects answers,
    and routes back to the discovery node for evaluation.
    
    Type 'DONE' to attempt finalization. If critical unknowns remain, the loop
    refuses to exit and displays [CRITICAL UNKNOWN DETECTED].
    """

    gate = state.get("current_gate", "REQUIREMENTS")
    discovery_data = state.get("discovery_questions", {})
    modules = discovery_data.get("modules", [])
    messages = list(state.get("messages", []))
    node_state = state.get("node_state", {})
    critical_remaining = node_state.get("discovery_critical_remaining", 0)
    complete = node_state.get("discovery_complete", False)
    round_num = node_state.get("discovery_question_count", 0)

    phase_label = "REQUIREMENTS" if gate == "REQUIREMENTS" else "ARCHITECTURE"

    if complete:
        logger.info("[discovery] %s discovery complete. Proceeding.", phase_label)
        return {"messages": messages, "node_state": node_state}

    # Display the header
    print()
    print("=" * 80)
    print(f"[HARNESS ARCHITECT SYSTEM AUDIT: {phase_label} PHASE] — Round {round_num}")
    print("=" * 80)
    print("The Architect has compiled a list of critical structural questions to eliminate all unknowns:")
    print()

    for module in modules:
        mod_name = module.get("name", "Module")
        questions = module.get("questions", [])
        if not questions:
            continue
        print(f"[MODULE: {mod_name}]")
        for q in questions:
            qid = q.get("id", "?")
            text = q.get("text", "")
            critical_marker = " **CRITICAL**" if q.get("critical") else ""
            print(f"  - {qid}:{critical_marker} {text}")
        print()

    if critical_remaining > 0:
        print(f"[CRITICAL]: {critical_remaining} critical question(s) remain unanswered.")
    print("-" * 80)
    print("Type your answers (referencing question numbers if preferred), 'DONE' to finalize, or 'SUSPEND' to save & quit.")
    print("-" * 80)

    try:
        response = input("User Response > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[Discovery] Input interrupted. Saving current state.")
        return {"messages": messages, "node_state": node_state}

    if response.upper() == "SUSPEND":
        session_id = state.get("session_id", "")
        workspace = state.get("workspace_path", "")
        print()
        print("=" * 60)
        print("Session saved to checkpoint.")
        print(f"Resume later with:")
        print(f"  harness resume --session-id {session_id}")
        if workspace and workspace != os.getcwd():
            print(f"  harness resume --session-id {session_id} -r {workspace}")
        print("=" * 60)
        print()
        logger.info("[discovery] %s phase suspended by developer. Session: %s", phase_label, session_id)
        node_state["hitl_suspend"] = True
        return {"messages": messages, "node_state": node_state}

    if response.upper() == "DONE":
        if critical_remaining > 0:
            # Refuse to exit with critical unknowns
            print()
            print("=" * 60)
            print(f"[CRITICAL UNKNOWN DETECTED]: {critical_remaining} critical question(s) still require answers.")
            print("You must specify the remaining variables before this phase can be finalized.")
            print("=" * 60)
            print()
            # Set flag so the router knows user tried to skip
            node_state["user_done_with_critical"] = True
            return {
                "messages": messages,
                "node_state": node_state,
            }
        else:
            node_state["discovery_complete"] = True
            logger.info("[discovery] User finalized %s phase. All questions resolved.", phase_label)
            print("[Discovery] All questions resolved. Finalizing specification...")
            return {
                "messages": messages,
                "node_state": node_state,
            }

    if not response:
        # Empty input, loop again
        return {"messages": messages, "node_state": node_state}

    # Append user's answers to conversation
    messages.append({"role": "user", "content": f"[Discovery Response - {phase_label}]: {response}"})
    node_state["discovery_complete"] = False  # Will be re-evaluated by discovery node
    logger.info("[discovery] Received user response (%d chars). Routing back for evaluation.", len(response))

    return {
        "messages": messages,
        "node_state": node_state,
    }


def hitl_menu_loop(state: dict[str, Any]) -> dict[str, Any]:
    """
    Interactive stdin menu for the human_intervention_node.

    Presents the developer with structured options:
        [v] View active file diffs
        [r] Resume graph execution (re-run compilation node)
        [e] Inject manual hint instruction string for the repair node
        [m] Pause for manual edits (notifies harness to wait while you fix files in your IDE)
        [b] Increase session budget limit (+ $2.00)
        [q] Abandon session and execute Git rollback

    Returns updated state dict reflecting the developer's chosen action.
    """
    node_state = state.get("node_state", {})
    trigger = node_state.get("hitl_trigger", "unknown")
    budget_remaining = state.get("budget_remaining_usd", 0.0)
    loop_counter = state.get("loop_counter", {})
    errors = state.get("compiler_errors", [])
    exit_code = state.get("exit_code", -1)
    modified_files = state.get("modified_files", [])
    workspace_path = state.get("workspace_path", os.getcwd())

    # Format error display
    error_text = "No compiler errors captured."
    if errors:
        error_lines = []
        for i, err in enumerate(errors[:5], 1):  # Show first 5 errors max
            error_lines.append(
                f"  [{i}] {err.get('file', '?')}:{err.get('line', 0)}:{err.get('column', 0)} "
                f"- {err.get('message', 'Unknown error')[:120]}"
            )
        error_text = "\n".join(error_lines)
        if len(errors) > 5:
            error_text += f"\n  ... and {len(errors) - 5} more errors."
    else:
        # No structured diagnostics — show raw build output instead
        raw_output = node_state.get("last_build_output", "")
        if raw_output:
            error_text = f"[No structured diagnostics. Raw build output (last 2000 chars):]\n{raw_output[-2000:]}"

    # Format diffs summary
    diffs_text = "No files modified."
    if modified_files:
        diffs_text = "Modified files:\n" + "\n".join(f"  - {f}" for f in modified_files)

    while True:
        print()
        print("=" * 80)
        print(f"[HUMAN-IN-THE-LOOP INTERVENTION] Trigger: {trigger}")
        print(f"  Budget: ${budget_remaining:.4f} / $2.00 | Loop Counter: {loop_counter.get('total_repairs', 0)}")
        print(f"  Exit Code: {exit_code}")
        print(f"  Modified Files: {len(modified_files)}")
        print("=" * 80)
        print()
        print("CRITICAL INFORMATION:")
        print(error_text)
        print()
        print("Options:")
        print("  [v] View active file diffs")
        print("  [r] Resume graph execution (re-run compilation node)")
        print("  [e] Inject manual hint instruction string for the repair node")
        print("  [m] Pause for manual edits (notifies harness to wait while you fix files in your IDE)")
        print("  [b] Increase session budget limit (+ $2.00)")
        print("  [s] Save & Quit (resume later)")
        print("  [q] Abandon session and execute Git rollback")
        print()

        try:
            choice = input("[HITL] Select action: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[HITL] Input interrupted. Aborting session.")
            node_state["hitl_abandon"] = True
            node_state["hitl_active"] = False
            state["node_state"] = node_state
            return state

        if choice == "v":
            print()
            print("--- Active File Diffs ---")
            print(diffs_text)
            print("-------------------------")

        elif choice == "r":
            # Resume: clear HITL flags, reset loop counter to allow one more repair attempt
            node_state["hitl_active"] = False
            node_state["hitl_awaiting_input"] = False
            node_state["hitl_resolved"] = True
            state["node_state"] = node_state
            # Reset total_repairs to 2 so route_after_compiler allows one more repair_node pass
            state["loop_counter"] = {"patching": 0, "repair": 0, "compiler": 0, "total_repairs": 2}
            logger.info("[HITL] Developer chose to resume. Loop counter reset to 2. Routing to compiler_node.")
            return state

        elif choice == "e":
            # Inject hint: append user string as a user message, reset loop counter to 1
            try:
                hint = input("[HITL] Enter hint/instruction for the repair node: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[HITL] Input interrupted.")
                continue
            if hint:
                messages = state.get("messages", [])
                messages.append({"role": "user", "content": f"[HITL Hint]: {hint}"})
                state["messages"] = messages
                # Reset loop counter to give AI another fresh attempt
                state["loop_counter"] = {"patching": 0, "repair": 0, "compiler": 0, "total_repairs": 1}
                node_state["hitl_active"] = False
                node_state["hitl_awaiting_input"] = False
                node_state["hitl_resolved"] = True
                state["node_state"] = node_state
                logger.info("[HITL] Hint injected. Loop counter reset to 1. Resuming.")
                return state

        elif choice == "m":
            # Manual edits: wait for developer to fix files in IDE
            print("[HITL] Pausing for manual IDE edits...")
            print(f"[HITL] Workspace: {workspace_path}")
            print("[HITL] Make your changes in your editor, then press Enter to continue.")
            try:
                input("[HITL] Press Enter when you are done editing... ")
            except (EOFError, KeyboardInterrupt):
                print("\n[HITL] Input interrupted.")
                continue
            # Reset loop counter and clear compiler errors since manual fix was applied
            state["loop_counter"] = {"patching": 0, "repair": 0, "compiler": 0, "total_repairs": 0}
            state["compiler_errors"] = []
            node_state["hitl_active"] = False
            node_state["hitl_awaiting_input"] = False
            node_state["hitl_resolved"] = True
            state["node_state"] = node_state
            logger.info("[HITL] Manual edits confirmed. Compiler errors cleared. Resuming to compiler_node.")
            return state

        elif choice == "b":
            # Increase budget by $2.00 and reset loop counter for a fresh attempt
            budget_remaining += 2.00
            state["budget_remaining_usd"] = budget_remaining
            # Reset loop counter to give the repair loop a full fresh cycle
            state["loop_counter"] = {"patching": 0, "repair": 0, "compiler": 0, "total_repairs": 0}
            print(f"[HITL] Budget increased by $2.00. New budget: ${budget_remaining:.2f}. Loop counter reset.")
            continue  # Stay in the menu loop

        elif choice == "s":
            session_id = state.get("session_id", "")
            print()
            print("=" * 60)
            print("Session saved to checkpoint.")
            print(f"Resume later with:")
            print(f"  harness resume --session-id {session_id}")
            if workspace_path and workspace_path != os.getcwd():
                print(f"  harness resume --session-id {session_id} -r {workspace_path}")
            print("=" * 60)
            print()
            logger.info("[HITL] Session suspended by developer. Session: %s", session_id)
            node_state["hitl_suspend"] = True
            node_state["hitl_active"] = False
            node_state["hitl_awaiting_input"] = False
            state["node_state"] = node_state
            return state

        elif choice == "q":
            # Abandon: set abandon flag, route to END
            print("[HITL] Abandoning session...")
            confirm = input("[HITL] Confirm abandon? This will attempt a git rollback. (y/N): ").strip().lower()
            if confirm == "y":
                node_state["hitl_abandon"] = True
                node_state["hitl_active"] = False
                node_state["hitl_awaiting_input"] = False
                state["node_state"] = node_state
                _attempt_git_rollback(workspace_path)
                logger.info("[HITL] Session abandoned. Git rollback attempted.")
                return state
            else:
                print("[HITL] Abandon cancelled.")
                continue

        else:
            print(f"[HITL] Unknown option: '{choice}'. Please choose from [v/r/e/m/b/s/q].")


# ---------------------------------------------------------------------------
# 2b. Requirement Refinement Layer (Pre-Flight Specification Gate)
# ---------------------------------------------------------------------------

_REQUIREMENTS_SYNTHESIS_PROMPT = """You are a Principal Systems Architect and Technical Product Manager.
Transform the following raw notes into a comprehensive, professionally structured
SPEC_REQUIREMENTS.md document.

## Output Sections

### 1. Executive Summary
- One paragraph describing the system's purpose and business value.

### 2. Functional Requirements (FR)
- **FR-XXX**: Title
  - Description: What the system must do.
  - Priority: Must Have / Should Have / Could Have.
  - Acceptance Criteria: Given/When/Then format.

### 3. System Scope
- In-scope features and modules.
- Out-of-scope items explicitly excluded.

### 4. Technical Constraints
- Language, framework, database, and infrastructure requirements.
- Performance targets (latency, throughput).
- Security requirements.

### 5. Explicit Edge Cases
- Error states: what happens when things go wrong.
- Boundary conditions: maximum/minimum values, concurrency limits.
- Recovery scenarios: retry logic, fallback behavior.

### 6. Non-Functional Requirements
- Reliability, scalability, observability.

## Raw Notes
{raw_notes}

## Formatting
Output as clean, well-structured Markdown. Use proper headings, bullet points,
and code blocks where appropriate. Do not include any text outside the document."""


async def synthesize_requirements(
    manifest_path: str,
    output_dir: str,
    gateway: Any,
) -> str:
    """
    Read raw notes from a manifest file, route to LLM for synthesis,
    and write SPEC_REQUIREMENTS.md to the output directory.

    Args:
        manifest_path: Path to the raw notes/text file.
        output_dir: Directory to write SPEC_REQUIREMENTS.md.
        gateway: Initialized LLM Gateway instance.

    Returns:
        Absolute path to the generated SPEC_REQUIREMENTS.md file.

    Raises:
        FileNotFoundError: If manifest_path does not exist.
        RuntimeError: If LLM synthesis fails.
    """
    manifest_full = os.path.abspath(manifest_path)
    if not os.path.isfile(manifest_full):
        raise FileNotFoundError(f"Manifest file not found: {manifest_full}")

    logger.info("[requirements] Reading manifest: %s", manifest_full)
    try:
        import aiofiles
        async with aiofiles.open(manifest_full, "r", encoding="utf-8", errors="replace") as f:
            raw_notes = await f.read()
    except ImportError:
        with open(manifest_full, "r", encoding="utf-8", errors="replace") as f:
            raw_notes = f.read()

    if not raw_notes.strip():
        raise RuntimeError("Manifest file is empty.")

    logger.info("[requirements] Synthesizing SPEC_REQUIREMENTS.md from %d chars of raw notes...", len(raw_notes))

    from harness.gateway import NodeRole
    prompt = _REQUIREMENTS_SYNTHESIS_PROMPT.format(raw_notes=raw_notes)
    messages = [
        {"role": "system", "content": "You are a technical documentation expert. Output clean, structured Markdown."},
        {"role": "user", "content": prompt},
    ]

    try:
        response, budget = await gateway.dispatch(
            messages=messages,
            role=NodeRole.PLANNING,
            budget_remaining_usd=2.00,
        )
    except Exception as exc:
        raise RuntimeError(f"LLM synthesis failed: {exc}") from exc

    content = response.content.strip()
    if not content:
        raise RuntimeError("LLM returned empty content for specification synthesis.")

    # Write the file
    os.makedirs(output_dir, exist_ok=True)
    spec_path = os.path.join(output_dir, "SPEC_REQUIREMENTS.md")
    try:
        import aiofiles
        async with aiofiles.open(spec_path, "w", encoding="utf-8") as f:
            await f.write(content)
    except ImportError:
        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(content)

    logger.info("[requirements] SPEC_REQUIREMENTS.md written to %s (%d chars, cost=$%.6f).",
                 spec_path, len(content), response.usage.cost_usd)
    return spec_path


def _read_spec_file(spec_path: str) -> str:
    """Read a specification file from disk."""
    if not os.path.isfile(spec_path):
        return ""
    try:
        with open(spec_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


async def _refine_requirements(
    spec_path: str,
    additional_notes: str,
    gateway: Any,
) -> str:
    """
    Refine an existing SPEC_REQUIREMENTS.md with additional user notes.
    Overwrites the file with the updated version.

    Returns the updated spec content.
    """
    current_spec = _read_spec_file(spec_path)
    if not current_spec:
        raise RuntimeError(f"Cannot read spec file for refinement: {spec_path}")

    refine_prompt = f"""You are reviewing and improving a software requirements specification.

## Current SPEC_REQUIREMENTS.md
{current_spec}

## User's Additional Notes / Feedback
{additional_notes}

## Task
Integrate the user's feedback into the specification. Keep the same structure
(Functional Requirements, System Scope, Technical Constraints, Edge Cases, NFRs).
Add, modify, or clarify sections as directed by the feedback. Output the complete
updated SPEC_REQUIREMENTS.md document."""

    from harness.gateway import NodeRole
    messages = [
        {"role": "system", "content": "You are a technical documentation expert. Output clean, structured Markdown."},
        {"role": "user", "content": refine_prompt},
    ]

    response, budget = await gateway.dispatch(
        messages=messages,
        role=NodeRole.PLANNING,
        budget_remaining_usd=2.00,
    )

    content = response.content.strip()
    if not content:
        raise RuntimeError("LLM returned empty content for specification refinement.")

    try:
        import aiofiles
        async with aiofiles.open(spec_path, "w", encoding="utf-8") as f:
            await f.write(content)
    except ImportError:
        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(content)

    logger.info("[requirements] SPEC_REQUIREMENTS.md refined (%d chars).", len(content))
    return content


def interactive_review_loop(spec_path: str, gateway: Any) -> str:
    """
    Interactive terminal review loop for SPEC_REQUIREMENTS.md.

    Options:
        [A] Approve — Accept the specification as-is and proceed.
        [B] Refine — Provide additional notes to improve the spec (loops).
        [C] Manual — Open the file in your IDE, edit, press Enter to continue.

    Args:
        spec_path: Absolute path to the SPEC_REQUIREMENTS.md file.
        gateway: Initialized LLM Gateway instance for refinement.

    Returns:
        The final approved specification content (to be used as messages[0]).
    """
    while True:
        spec_content = _read_spec_file(spec_path)
        spec_size = len(spec_content) if spec_content else 0

        print()
        print("=" * 72)
        print("[REQUIREMENT REFINEMENT GATE]")
        print(f"  Specification: {spec_path}")
        print(f"  Size: {spec_size:,} characters")
        print("=" * 72)
        print()
        print("[A] Approve — Lock this specification and proceed to graph execution.")
        print("[B] Refine — Provide additional notes to improve the specification.")
        print("[C] Manual — Edit the file in your IDE, then press Enter to continue.")
        print()

        try:
            choice = input("[Requirements] Select action [A/B/C]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[Requirements] Input interrupted. Aborting.")
            sys.exit(1)

        if choice == "a":
            # Approve: return the current content as the locked spec
            logger.info("[requirements] Specification approved (%d chars).", spec_size)
            return spec_content

        elif choice == "b":
            # Refine: get feedback, send to LLM, overwrite, loop
            try:
                notes = input("[Refine] Enter additional notes/feedback for the specification:\n").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Refine] Input interrupted.")
                continue

            if not notes:
                print("[Refine] No notes provided. Returning to menu.")
                continue

            print("[Refine] Updating specification with your feedback...")
            try:
                updated = asyncio.get_event_loop().run_until_complete(
                    _refine_requirements(spec_path, notes, gateway)
                )
                print(f"[Refine] Specification updated ({len(updated):,} chars).")
            except Exception as exc:
                print(f"[Refine] Error: {exc}")
            # Loop back to menu

        elif choice == "c":
            # Manual: pause for IDE edits, then read from disk
            print(f"[Manual] Edit the file at: {spec_path}")
            print("[Manual] Make your changes in your editor (VS Code, Cursor, etc.).")
            try:
                input("[Manual] Press Enter when you are done editing... ")
            except (EOFError, KeyboardInterrupt):
                print("\n[Manual] Input interrupted. Reading current file state.")

            spec_content = _read_spec_file(spec_path)
            if spec_content:
                logger.info("[requirements] Manual edits confirmed (%d chars).", len(spec_content))
                return spec_content
            else:
                print("[Manual] Warning: Could not read the file. Returning to menu.")
                continue

        else:
            print(f"[Requirements] Unknown option: '{choice}'. Please choose A, B, or C.")


def _attempt_git_rollback(workspace_path: str) -> None:
    """Attempt a git checkout to restore modified files to their original state."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "-C", workspace_path, "checkout", "--", "."],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info("[HITL] Git rollback successful.")
        else:
            logger.warning("[HITL] Git rollback failed: %s", result.stderr.strip())
    except Exception as exc:
        logger.warning("[HITL] Git rollback error: %s", exc)


# ---------------------------------------------------------------------------
# 3. Subcommand Handlers
# ---------------------------------------------------------------------------

async def cmd_run(args: argparse.Namespace) -> int:
    """
    Execute the `harness run` subcommand.

    Steps:
        1. Resolve workspace path.
        2. Discover configuration.
        3. Resolve build command.
        4. Initialize checkpointer.
        5. Compile the graph.
        6. Execute the graph with the provided prompt.
        7. Handle HITL breakpoints if triggered.

    Examples:
        harness run -r /path/to/repo -p "Add JWT authentication"
        harness run -r ./myproject -p "Refactor the auth module" --manifest notes.txt
    """
    workspace_path = os.path.abspath(args.workspace)
    if not os.path.isdir(workspace_path):
        logger.error("Workspace path does not exist: %s", workspace_path)
        return 1

    config = discover_config(workspace_path)
    build_command = resolve_build_command(args.build_cmd, config)

    # Extract persistence settings
    persistence_cfg = config.get("persistence", {})
    db_path = persistence_cfg.get("db_path", "~/.harness/checkpoints.db")
    ttl_days = persistence_cfg.get("ttl_days", 30)

    # Initialize checkpointer
    from harness.storage import HarnessAsyncSqliteSaver, generate_session_id
    checkpointer = await HarnessAsyncSqliteSaver.from_db_path(db_path=db_path, ttl_days=ttl_days)

    session_id = generate_session_id(args.session_id)

    # Extract budget and sandbox settings
    token_budget = config.get("token_budget", {})
    budget_usd = token_budget.get("hard_cap_usd", 2.00)
    allow_network = args.allow_network or config.get("allow_network", False)

    # Initialize the LLM Gateway and inject it for graph nodes
    from harness.gateway import create_gateway_from_config
    from harness.graph import set_gateway, run_graph

    gateway = create_gateway_from_config(config)
    set_gateway(gateway)

    # Initialize the secret redactor
    from harness.redactor import create_redactor_from_config
    create_redactor_from_config(config)

    # Initialize GitGuardian for branch lifecycle management
    from harness.security import GitGuardian
    git_guardian = GitGuardian(workspace_path)
    git_guardian.stash_if_dirty()
    git_guardian.create_patch_branch(session_id)

    # --- Requirement Refinement Layer (product_spec.txt auto-discovery or --manifest override) ---
    spec_override: Optional[str] = None

    # Resolve the manifest file path:
    #   1. --manifest flag (explicit override, highest priority)
    #   2. Auto-discovered product_spec.txt in workspace root (convention)
    #   3. None — proceed with prompt-only execution
    manifest_path: Optional[str] = None
    if args.manifest:
        manifest_path = os.path.abspath(args.manifest)
        if not os.path.isfile(manifest_path):
            logger.error("[requirements] Explicit manifest file not found: %s", manifest_path)
            return 1
        logger.info("[requirements] Using explicit manifest: %s", manifest_path)
    else:
        # Auto-discovery: look for product_spec.txt in workspace root
        manifest_file = config.get("manifest_file", "product_spec.txt")
        auto_manifest = os.path.join(workspace_path, manifest_file)
        if os.path.isfile(auto_manifest):
            manifest_path = auto_manifest
            logger.info("[requirements] Auto-discovered product spec: %s", manifest_path)

    if manifest_path:
        logger.info("[requirements] Synthesizing specification from %s", manifest_path)
        try:
            # Resolve output_dir relative to the workspace, not the CWD where harness was invoked
            output_dir = args.output_dir
            if not os.path.isabs(output_dir):
                output_dir = os.path.join(workspace_path, output_dir)
            spec_path = await synthesize_requirements(
                manifest_path=manifest_path,
                output_dir=output_dir,
                gateway=gateway,
            )
            logger.info("[requirements] Specification synthesized. Entering review loop.")
            spec_override = interactive_review_loop(spec_path, gateway)
            logger.info("[requirements] Specification locked. %d characters approved.", len(spec_override))
        except Exception as exc:
            logger.error("[requirements] Requirement refinement failed: %s", exc)
            return 1
    else:
        logger.info("[requirements] No product spec file found. Place '%s' at the workspace root with your product requirements, or use --manifest to specify an alternate file.",
                     config.get("manifest_file", "product_spec.txt"))

    thread_id = args.thread_id if args.thread_id else session_id

    logger.info("=" * 60)
    logger.info("AI Agent Harness — Starting Graph Execution")
    logger.info("  Workspace:  %s", workspace_path)
    logger.info("  Build Cmd:  %s", build_command)
    logger.info("  Session ID: %s", session_id)
    logger.info("  Thread ID:  %s", thread_id)
    logger.info("  Budget:     $%.2f", budget_usd)
    logger.info("  Network:    %s", "enabled" if allow_network else "blocked")
    logger.info("  Prompt:     %s", args.prompt[:100] + ("..." if len(args.prompt) > 100 else ""))
    if spec_override:
        logger.info("  Spec:       SPEC_REQUIREMENTS.md (%d chars)", len(spec_override))
    logger.info("=" * 60)

    try:
        final_state = await run_graph(
            workspace_path=workspace_path,
            prompt=args.prompt,
            build_command=build_command,
            spec_override=spec_override,
            allow_network=allow_network,
            budget_usd=budget_usd,
            session_id=session_id,
            checkpointer=checkpointer,
            thread_id=thread_id,
            skip_discovery=args.skip_discovery,
        )
    except Exception:
        logger.exception("Graph execution failed with unhandled exception.")
        git_guardian.rollback()
        git_guardian.pop_stash()
        await checkpointer.conn.close()
        return 1

    exit_code = final_state.get("exit_code", -1)
    modified_files = final_state.get("modified_files", [])
    token_tracker = final_state.get("token_tracker", {})
    total_cost = token_tracker.get("total_cost_usd", 0.0)

    # Git lifecycle: commit on success, rollback on failure
    if exit_code == 0:
        git_guardian.commit_all_changes(session_id, modified_files, exit_code)
        git_guardian.restore_original_branch()
    else:
        git_guardian.rollback(modified_files)

    git_guardian.pop_stash()

    logger.info("=" * 60)
    logger.info("Graph Execution Complete")
    logger.info("  Exit Code:      %d", exit_code)
    logger.info("  Modified Files: %d", len(modified_files))
    for f in modified_files:
        logger.info("    - %s", f)
    logger.info("  Token Cost:     $%.6f", total_cost)
    logger.info("  Session ID:     %s", session_id)
    logger.info("=" * 60)

    await checkpointer.conn.close()

    return 0 if exit_code == 0 else 1


async def cmd_resume(args: argparse.Namespace) -> int:
    """
    Execute the `harness resume` subcommand.

    Restores a previously checkpointed session from SQLite and resumes
    graph execution from the exact checkpoint boundary.

    Example:
        harness resume --session-id my-session-abc123
        harness resume --session-id my-session -r /path/to/repo
    """
    from harness.storage import HarnessAsyncSqliteSaver

    workspace_path = os.path.abspath(args.workspace) if args.workspace else os.getcwd()
    config = discover_config(workspace_path)
    persistence_cfg = config.get("persistence", {})
    db_path = persistence_cfg.get("db_path", "~/.harness/checkpoints.db")
    ttl_days = persistence_cfg.get("ttl_days", 30)

    checkpointer = await HarnessAsyncSqliteSaver.from_db_path(db_path=db_path, ttl_days=ttl_days)

    # Verify that the thread exists
    config_for_get = {"configurable": {"thread_id": args.session_id}}
    existing = await checkpointer.aget(config_for_get)
    if existing is None:
        logger.error("No checkpoint found for session '%s'.", args.session_id)
        await checkpointer.conn.close()
        return 1

    build_command = resolve_build_command(args.build_cmd, config)
    token_budget = config.get("token_budget", {})
    budget_usd = token_budget.get("hard_cap_usd", 2.00)
    allow_network = args.allow_network or config.get("allow_network", False)

    # Initialize the LLM Gateway and inject it for graph nodes
    from harness.gateway import create_gateway_from_config
    from harness.graph import set_gateway, run_graph

    gateway = create_gateway_from_config(config)
    set_gateway(gateway)

    # Initialize the secret redactor
    from harness.redactor import create_redactor_from_config
    create_redactor_from_config(config)

    logger.info("[resume] Restoring session '%s' from checkpoint.", args.session_id)

    try:
        final_state = await run_graph(
            workspace_path=workspace_path,
            prompt=args.prompt or "(resumed session)",
            build_command=build_command,
            allow_network=allow_network,
            budget_usd=budget_usd,
            session_id=args.session_id,
            checkpointer=checkpointer,
            thread_id=args.session_id,
        )
    except Exception:
        logger.exception("Resume execution failed.")
        await checkpointer.conn.close()
        return 1

    exit_code = final_state.get("exit_code", -1)
    logger.info("[resume] Session '%s' completed with exit code %d.", args.session_id, exit_code)

    await checkpointer.conn.close()
    return 0 if exit_code == 0 else 1


async def cmd_status(args: argparse.Namespace) -> int:
    """
    Execute the `harness status` subcommand.

    Reads the SQLite checkpoint database read-only and prints a clean
    text snapshot of the specified session's state without triggering
    any graph execution.

    Examples:
        harness status --session-id my-session
        harness status --all
    """
    from harness.storage import HarnessAsyncSqliteSaver, inspect_session, list_all_sessions

    workspace_path = os.path.abspath(args.workspace) if args.workspace else os.getcwd()
    config = discover_config(workspace_path)
    persistence_cfg = config.get("persistence", {})
    db_path = persistence_cfg.get("db_path", "~/.harness/checkpoints.db")
    ttl_days = persistence_cfg.get("ttl_days", 30)

    # Run GC on startup
    checkpointer = await HarnessAsyncSqliteSaver.from_db_path(db_path=db_path, ttl_days=ttl_days)

    if args.all:
        # List all sessions
        sessions = await list_all_sessions(db_path)
        if not sessions:
            print("No checkpointed sessions found.")
        else:
            print(f"{'SESSION ID':<40} {'UPDATED':<20} {'CREATED':<20} {'WORKSPACE':<40}")
            print("-" * 100)
            for s in sessions:
                print(f"{s.thread_id:<40} {s.updated_at:<20} {s.created_at:<20} {s.workspace_path:<40}")
        await checkpointer.conn.close()
        return 0

    if not args.session_id:
        logger.error("Please provide --session-id or use --all to list all sessions.")
        await checkpointer.conn.close()
        return 1

    summary = await inspect_session(db_path, args.session_id)
    if summary is None:
        print(f"No checkpoint found for session '{args.session_id}'.")
        await checkpointer.conn.close()
        return 1

    print("=" * 60)
    print("Session Status")
    print("=" * 60)
    print(f"  Thread ID:          {summary.thread_id}")
    print(f"  Session ID:         {summary.session_id}")
    print(f"  Current Node:       {summary.current_node or '(unknown)'}")
    print(f"  Exit Code:          {summary.exit_code}")
    print(f"  Budget Remaining:   ${summary.budget_remaining_usd:.4f}")
    print(f"  Total Token Cost:   ${summary.total_cost_usd:.6f}")
    print(f"  Modified Files:     {len(summary.modified_files)}")
    for f in summary.modified_files[:10]:
        print(f"    - {f}")
    if len(summary.modified_files) > 10:
        print(f"    ... and {len(summary.modified_files) - 10} more")
    print(f"  Loop Counters:      {summary.loop_counters}")
    print(f"  Is Active:          {summary.is_active}")
    print(f"  Created:            {summary.created_at}")
    print(f"  Updated:            {summary.updated_at}")
    print(f"  Workspace:          {summary.workspace_path}")
    print("=" * 60)

    await checkpointer.conn.close()
    return 0


async def cmd_purge(args: argparse.Namespace) -> int:
    """
    Execute the `harness purge` subcommand.

    Wipes all checkpoint data from the SQLite database.

    Examples:
        harness purge --session-id my-session
        harness purge --all
    """
    workspace_path = os.path.abspath(args.workspace) if args.workspace else os.getcwd()
    config = discover_config(workspace_path)
    persistence_cfg = config.get("persistence", {})
    db_path = persistence_cfg.get("db_path", "~/.harness/checkpoints.db")
    ttl_days = persistence_cfg.get("ttl_days", 30)

    from harness.storage import HarnessAsyncSqliteSaver, purge_checkpoints

    if args.all:
        print("WARNING: This will delete ALL checkpoint data permanently.")
        confirm = input("Type 'yes' to confirm: ").strip()
        if confirm.lower() != "yes":
            print("Purge cancelled.")
            return 0
        deleted = await purge_checkpoints(db_path)
        print(f"Purged {deleted} rows from the checkpoint database.")
    elif args.session_id:
        checkpointer = await HarnessAsyncSqliteSaver.from_db_path(db_path=db_path, ttl_days=ttl_days)
        await checkpointer.adelete_thread(args.session_id)
        print(f"Purged all checkpoints for session '{args.session_id}'.")
        await checkpointer.conn.close()
    else:
        logger.error("Please specify --all to purge everything or --session-id to purge a specific session.")
        return 1

    return 0


# ---------------------------------------------------------------------------
# 4. Argument Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct the full CLI argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="harness",
        description=(
            "AI Agent Harness — Production-grade, model-agnostic LangGraph agent\n"
            "for autonomous code generation, sandboxed builds, and bulletproof persistence.\n\n"
            "Quick Start:\n"
            "  harness run -r /path/to/repo -p \"Your engineering task description\"\n"
            "  harness -h                     Show this help\n"
            "  harness run -h                 Show run subcommand help\n"
            "  harness status --all           List all checkpointed sessions\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  harness run -r ./myproject -p \"Add JWT authentication\"\n"
            "  harness run -r /path/to/repo -p \"Refactor logging\" --manifest notes.txt\n"
            "  harness resume --session-id abc123\n"
            "  harness status --session-id abc123\n"
            "  harness purge --all\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- `harness run` ---
    run_parser = subparsers.add_parser("run", help="Execute the agent graph on a workspace")
    run_parser.add_argument(
        "--manifest", "-m",
        default=None,
        help="Path to a raw notes/text file to synthesize into SPEC_REQUIREMENTS.md before execution.",
    )
    run_parser.add_argument(
        "--output-dir", "-o",
        default="./docs",
        help="Directory to write SPEC_REQUIREMENTS.md (default: ./docs).",
    )
    run_parser.add_argument(
        "--workspace", "-w", "-r",
        required=True,
        help="Absolute or relative path to the target repository root.",
    )
    run_parser.add_argument(
        "--prompt", "-p",
        required=True,
        help="The engineering task description (e.g., 'Refactor the auth module to use JWT').",
    )
    run_parser.add_argument(
        "--build-cmd",
        default=None,
        help="Override the build command (e.g., 'make build'). Falls back to .harness_config.json or 'make build'.",
    )
    run_parser.add_argument(
        "--session-id",
        default=None,
        help="Human-readable session identifier. Auto-generated UUIDv4 if not provided.",
    )
    run_parser.add_argument(
        "--thread-id",
        default=None,
        help="LangGraph thread ID for checkpoint lookups. Defaults to session-id.",
    )
    run_parser.add_argument(
        "--allow-network",
        action="store_true",
        default=False,
        help="Permit outbound network traffic in the sandbox.",
    )
    run_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable debug-level logging.",
    )
    run_parser.add_argument(
        "--skip-discovery", "-s",
        action="store_true",
        default=False,
        help="Skip requirements/architecture discovery phases and go directly to code generation.",
    )

    # --- `harness resume` ---
    resume_parser = subparsers.add_parser("resume", help="Resume a crashed or interrupted session from its checkpoint")
    resume_parser.add_argument(
        "--session-id",
        required=True,
        help="The session/thread ID to resume.",
    )
    resume_parser.add_argument(
        "--workspace", "-w", "-r",
        default=None,
        help="Workspace path (auto-detected from checkpoint if omitted).",
    )
    resume_parser.add_argument(
        "--prompt", "-p",
        default=None,
        help="Optional additional prompt to append to the resumed session.",
    )
    resume_parser.add_argument(
        "--build-cmd",
        default=None,
        help="Override the build command.",
    )
    resume_parser.add_argument(
        "--allow-network",
        action="store_true",
        default=False,
        help="Permit outbound network traffic in the sandbox.",
    )
    resume_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable debug-level logging.",
    )

    # --- `harness status` ---
    status_parser = subparsers.add_parser("status", help="Read-only inspection of a checkpointed session")
    status_parser.add_argument(
        "--session-id",
        default=None,
        help="The session/thread ID to inspect.",
    )
    status_parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="List all checkpointed sessions.",
    )
    status_parser.add_argument(
        "--workspace", "-w", "-r",
        default=None,
        help="Workspace path (for config discovery). Defaults to current directory.",
    )

    # --- `harness purge` ---
    purge_parser = subparsers.add_parser("purge", help="Manually wipe checkpoint data")
    purge_parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Delete ALL checkpoint data permanently.",
    )
    purge_parser.add_argument(
        "--session-id",
        default=None,
        help="Purge checkpoints for a specific session only.",
    )
    purge_parser.add_argument(
        "--workspace", "-w", "-r",
        default=None,
        help="Workspace path (for config discovery). Defaults to current directory.",
    )

    return parser


# ---------------------------------------------------------------------------
# 5. Main Entry Point
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Primary CLI entry point. Dispatches to the correct subcommand handler.

    Returns:
        0 on success, non-zero on failure.
    """
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Set logging level
    if getattr(args, "verbose", False):
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("harness").setLevel(logging.DEBUG)

    # Dispatch to subcommand
    if args.command == "run":
        return asyncio.run(cmd_run(args))
    elif args.command == "resume":
        return asyncio.run(cmd_resume(args))
    elif args.command == "status":
        return asyncio.run(cmd_status(args))
    elif args.command == "purge":
        return asyncio.run(cmd_purge(args))
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())