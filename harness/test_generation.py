"""Auto-generate technology-specific unit tests after patching_node and run them
deterministically in the sandbox.

Topology:
    patching_node → speculative_node → test_generation_node → lintgate_node → compiler_node

The node:
    1. Reads state["modified_files"], filters out anything that's already a test.
    2. Detects the workspace stack via harness.impact._detect_workspace_stack.
    3. Loads the per-stack test_guides/<lang>.md guidance into the LLM prompt.
    4. Dispatches one LLM call (gateway, NodeRole.PATCHING) asking for patch
       blocks that create stack-canonical test files for the modified sources.
       **The prompt forbids mocks** — tests call the real implementation; only
       the test runner's built-in fakes (monkeypatch / tmp_path / httptest /
       @TempDir / etc.) are allowed.
    5. Applies the patches via the existing patcher pipeline. Path traversal +
       absolute-path attempts are rejected by harness.trust.safe_resolve.
    6. If zero tests were generated → return passed status, skip sandbox.
    7. Otherwise runs a stack-canonical test command in the sandbox (separate
       from the user's build_command) and surfaces failures via the standard
       compiler_errors path so repair_node can fix them.

Guardrails:
    - Requires a configured LLM gateway. When `get_gateway() is None`, the node
      synthesises an env-misconfig diagnostic ("installer must provide a valid
      LLM API key") and short-circuits to HITL — never silently no-ops.
    - Config-gated: `test_generation.enabled = false` in cli.json / .harness_config.json
      disables the node entirely.
    - Workspace boundary: every generated file is post-validated to live under
      state["workspace_path"]; anything that escaped (would only happen on a
      patcher bug) is dropped from generated_tests with a warning.
    - Loop guard: `test_generation` loop counter + `max_iterations` config cap.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-stack canonical test invocations
# ---------------------------------------------------------------------------

# Maps a primary stack tag → the deterministic command we run after generating
# tests. Tokens like "pip install" / "npm install" trigger the existing
# _build_command_needs_network heuristic in harness.graph, so the sandbox
# auto-enables network for these calls without any extra config.
_STACK_TEST_COMMANDS: dict[str, str] = {
    "python": "pip install -q pytest && python3 -m pytest -q",
    "node": "npm install --no-save --silent jest && npx jest --silent",
    "javascript": "npm install --no-save --silent jest && npx jest --silent",
    "typescript": "npm install --no-save --silent jest ts-jest typescript && npx jest --silent",
    "go": "go test ./...",
    "java": "mvn -q test",
    "rust": "cargo test --quiet",
    "dart": "dart test",
    "flutter": "flutter test",
}


# Stack-tag priority: when _detect_workspace_stack returns multiple tags, pick
# the first hit in this list as the primary language for prompt + test runner
# selection. Frontend frameworks (react/vue/angular) imply javascript/typescript,
# so they don't appear here directly.
_PRIMARY_STACK_PRIORITY: tuple[str, ...] = (
    "flutter", "dart", "rust", "go", "java", "typescript",
    "node", "javascript", "python",
)


# File-extension → stack hint for the "is this modified file a source file
# worth testing?" check. Anything not in this table is skipped (markdown,
# JSON config, lockfiles, etc.).
_SOURCE_EXTENSIONS: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".dart": "dart",
}


# Filename / path patterns identifying files that ARE tests (skip these — we
# don't write tests for tests).
_TEST_FILE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(^|/)test_[^/]+\.py$"),
    re.compile(r"(^|/)tests?(/|$)"),
    re.compile(r"_test\.go$"),
    re.compile(r"\.test\.(js|jsx|ts|tsx|mjs)$"),
    re.compile(r"\.spec\.(js|jsx|ts|tsx|mjs)$"),
    re.compile(r"__tests__/"),
    re.compile(r"src/test/"),
    re.compile(r"Test\.java$"),
    re.compile(r"_test\.dart$"),
)


_PYTEST_IMPORTLIB_INI = (
    "[pytest]\n"
    "# Auto-written by harness.test_generation. Uses importlib import mode so\n"
    "# same-named test files in different packages (e.g. tests/models/test_job.py\n"
    "# and tests/schemas/test_job.py) coexist as distinct dotted names instead\n"
    "# of colliding with 'import file mismatch' under the default prepend mode.\n"
    "addopts = --import-mode=importlib\n"
)


def _ensure_pytest_importlib_config(workspace_path: str) -> Optional[str]:
    """Write a minimal ``pytest.ini`` with ``--import-mode=importlib`` if the
    workspace has no pytest configuration of any kind.

    Recognises every shape pytest itself looks at:
      - ``pytest.ini``
      - ``pyproject.toml`` with a ``[tool.pytest.ini_options]`` table
      - ``setup.cfg`` with a ``[tool:pytest]`` section
      - ``tox.ini`` with a ``[pytest]`` section

    Returns the workspace-relative path of the newly-written file, or
    ``None`` if any existing config was found (no-op).
    """
    if not workspace_path or not os.path.isdir(workspace_path):
        return None

    # Hard signals — file existence wins. pyproject.toml / setup.cfg / tox.ini
    # only count if they actually contain a pytest section.
    if os.path.isfile(os.path.join(workspace_path, "pytest.ini")):
        return None

    section_signals: tuple[tuple[str, str], ...] = (
        ("pyproject.toml", "[tool.pytest.ini_options]"),
        ("setup.cfg", "[tool:pytest]"),
        ("tox.ini", "[pytest]"),
    )
    for fname, marker in section_signals:
        path = os.path.join(workspace_path, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                # Manifests are small; cap reads at 256 KB.
                content = f.read(256 * 1024)
        except OSError:
            continue
        if marker in content:
            return None

    target = os.path.join(workspace_path, "pytest.ini")
    try:
        with open(target, "w", encoding="utf-8") as f:
            f.write(_PYTEST_IMPORTLIB_INI)
    except OSError as exc:
        logger.warning(
            "[test_generation_node] Failed to write pytest.ini: %s. "
            "Same-basename test collisions may still occur.", exc,
        )
        return None
    logger.info(
        "[test_generation_node] Wrote default pytest.ini with "
        "--import-mode=importlib so duplicate test basenames coexist."
    )
    return "pytest.ini"


def _is_test_file(rel_path: str) -> bool:
    """True when ``rel_path`` looks like an existing test file."""
    norm = rel_path.replace("\\", "/")
    return any(p.search(norm) for p in _TEST_FILE_PATTERNS)


# Directory names that should never be walked for test_generation candidates.
# Mirrors harness.impact._NEVER_SOURCE_DIRS but kept local to avoid an import
# cycle (impact also imports test-related state types in some configurations).
_SCAN_SKIP_DIRS: frozenset[str] = frozenset({
    "__pycache__", "node_modules", "vendor", "target", "build", "dist",
    "out", ".venv", "venv", "env", ".git", ".tox", ".nox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "docs", "doc", "migrations", "fixtures",
    "tests", "test", "__tests__",
})


def _scan_workspace_for_source(workspace_path: str, limit: int = 200) -> list[str]:
    """Return workspace-relative paths of testable source files anywhere
    under ``workspace_path``.

    Used by the no_tests_collected fallback path: when the router sent us
    here because pytest had no tests to run but `state["modified_files"]`
    happened to be test/manifest-only, we still need to know which source
    files exist on disk so we can write tests for them.
    """
    found: list[str] = []
    if not workspace_path or not os.path.isdir(workspace_path):
        return found
    workspace_path = os.path.abspath(workspace_path)
    try:
        for sub_root, sub_dirs, sub_files in os.walk(workspace_path):
            sub_dirs[:] = [
                d for d in sub_dirs
                if not d.startswith(".") and d not in _SCAN_SKIP_DIRS
            ]
            for fname in sub_files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in _SOURCE_EXTENSIONS:
                    continue
                abspath = os.path.join(sub_root, fname)
                relpath = os.path.relpath(abspath, workspace_path)
                if _is_test_file(relpath):
                    continue
                found.append(relpath)
                if len(found) >= limit:
                    return found
    except OSError:
        return found
    return found


def _pick_primary_stack(tags: set[str]) -> Optional[str]:
    """Pick the single stack tag to drive test generation."""
    for tag in _PRIMARY_STACK_PRIORITY:
        if tag in tags:
            return tag
    return None


def _stack_test_command(primary: str) -> Optional[str]:
    """Return the deterministic test runner command for a primary stack."""
    return _STACK_TEST_COMMANDS.get(primary)


# ---------------------------------------------------------------------------
# Workspace boundary check (defense in depth)
# ---------------------------------------------------------------------------

def _inside_workspace(file_rel: str, workspace_path: str) -> bool:
    """True when ``file_rel`` (workspace-relative) resolves to a path inside
    ``workspace_path``.

    Defence in depth — the patcher's trust.safe_resolve already enforces this
    hard, so this only fires if a patcher bug ever lets something through.
    """
    if not file_rel:
        return False
    if os.path.isabs(file_rel):
        return False
    workspace_real = os.path.realpath(workspace_path)
    file_real = os.path.realpath(os.path.join(workspace_real, file_rel))
    try:
        return os.path.commonpath([workspace_real, file_real]) == workspace_real
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

_PROMPT_FORMAT_REMINDER = """[CRITICAL FORMAT INSTRUCTION]
You MUST respond using ONLY the patch block syntax below. No prose, no markdown
code fences, no commentary. Your entire response must be parseable as patch
blocks.

<<<CREATE_FILE>>>
file: <workspace-relative path>
content:
<complete file contents>
<<<END_CREATE_FILE>>>

<<<INSERT_AT_BLOCK>>>
file: <workspace-relative path>
anchor: <function or class name>
placement: before|after
content:
<lines to insert>
<<<END_INSERT_AT_BLOCK>>>

RULES — absolute:
  1. File paths MUST be workspace-relative. Anything starting with '/', '~',
     or '..' will be rejected.
  2. Do NOT generate mocks. No unittest.mock, Mockito, jest.mock, mockall,
     mockito (Dart), gomock, sinon, nock. Tests must call the real
     implementation with realistic inputs. Use only the test runner's
     built-in fakes (pytest monkeypatch / tmp_path, httptest.Server,
     @TempDir, etc.) when a side effect cannot be invoked directly.
  3. Cover the typical paths AND the edge cases (empty input, zero/negative
     values, error branches). Skip cases that would require mocking external
     services.
  4. Match the stack-canonical layout and naming convention.

Generate test patches NOW. Only the blocks above. No other text."""


def _build_test_gen_prompt(
    workspace_path: str,
    modified_source_files: list[str],
    primary_stack: str,
    max_per_file_chars: int = 6000,
) -> str:
    """Build the user-prompt body listing modified source files and asking for tests."""
    lines: list[str] = [
        f"Generate unit tests for the following source files (stack: {primary_stack}).",
        "Each test file should follow the conventions in the test-generation guide "
        "already loaded in the system prompt.",
        "",
        "## Source files to test",
        "",
    ]
    for rel in modified_source_files:
        abs_path = os.path.join(workspace_path, rel)
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                body = fh.read(max_per_file_chars)
        except OSError:
            body = "<unreadable>"
        lines.append(f"### `{rel}`")
        lines.append("```")
        lines.append(body)
        lines.append("```")
        lines.append("")
    lines.append(_PROMPT_FORMAT_REMINDER)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthetic diagnostics
# ---------------------------------------------------------------------------

def _synth_diag(
    file: str,
    message: str,
    error_code: str = "TEST_FAILURE",
    line: int = 0,
) -> dict[str, Any]:
    """Synthesize a DiagnosticObject-dict so the existing compiler_errors /
    repair_node pipeline can consume it without changes."""
    return {
        "file": file,
        "line": line,
        "column": 0,
        "severity": "error",
        "error_code": error_code,
        "message": message,
        "semantic_context": "",
    }


# ---------------------------------------------------------------------------
# Public node
# ---------------------------------------------------------------------------

async def test_generation_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: generate tests for the just-patched source files and
    verify them deterministically in the sandbox.

    See module docstring for the contract. Returns a state-update dict.
    """
    cfg = state.get("test_generation_config", {}) or {}
    if not cfg.get("enabled", True):
        logger.info("[test_generation_node] Disabled in config. Skipping.")
        return {}

    workspace_path: str = state.get("workspace_path", os.getcwd())
    modified_files: list[str] = list(state.get("modified_files", []) or [])
    no_tests_collected: bool = bool(
        state.get("node_state", {}).get("no_tests_collected")
    )

    # --- Skip when nothing testable ---
    if not modified_files and not no_tests_collected:
        logger.info("[test_generation_node] No modified files. Skipping.")
        return {}

    source_files = [
        rel for rel in modified_files
        if not _is_test_file(rel)
        and os.path.splitext(rel)[1].lower() in _SOURCE_EXTENSIONS
    ]

    # When the compiler routed us here because pytest exit=5 (no tests
    # collected), modified_files may legitimately contain only test scaffolds
    # or manifest files — yet the workspace has source code that needs tests.
    # Fall back to a workspace scan for testable source so we don't return
    # empty and bounce back through compiler→test_gen→repair forever.
    if not source_files and no_tests_collected:
        scanned = _scan_workspace_for_source(workspace_path)
        if scanned:
            logger.info(
                "[test_generation_node] no_tests_collected: modified_files had "
                "no testable source, scanned workspace and found %d source "
                "file(s): %s",
                len(scanned), scanned[:10],
            )
            source_files = scanned
        else:
            logger.warning(
                "[test_generation_node] no_tests_collected but workspace scan "
                "found no source files either. Routing to HITL."
            )
            loop_counter = dict(state.get("loop_counter", {}))
            return {
                "loop_counter": loop_counter,
                "node_state": {
                    "current_node": "test_generation",
                    "env_misconfig": True,
                    "env_misconfig_symbol": "no_source_files",
                    "test_generation": {
                        "status": "skipped",
                        "reason": "no_source_in_workspace",
                    },
                },
            }

    if not source_files:
        logger.info(
            "[test_generation_node] No testable source files in the %d modified file(s). Skipping.",
            len(modified_files),
        )
        return {}

    # --- Stack detection ---
    from harness.impact import _detect_workspace_stack
    tags = _detect_workspace_stack(workspace_path) or set()
    # Fall back to source-extension heuristic when manifest detection finds
    # nothing (greenfield workspaces).
    if not any(t in _PRIMARY_STACK_PRIORITY for t in tags):
        for rel in source_files:
            ext_stack = _SOURCE_EXTENSIONS.get(os.path.splitext(rel)[1].lower())
            if ext_stack:
                tags.add(ext_stack)
                break
    primary = _pick_primary_stack(tags)
    if primary is None:
        logger.info(
            "[test_generation_node] No supported stack detected (tags=%s). Skipping.",
            sorted(tags),
        )
        return {}

    # --- LLM-key gate ---
    from harness.graph import get_gateway
    gateway = get_gateway()
    if gateway is None:
        logger.error(
            "[test_generation_node] No LLM gateway configured. test_generation "
            "requires a valid LLM API key. Routing to HITL."
        )
        loop_counter = dict(state.get("loop_counter", {}))
        diagnostic = _synth_diag(
            file="<test_generation>",
            message=(
                "test_generation requires a valid LLM API key, but no gateway is "
                "configured. The installer must set ANTHROPIC_API_KEY / "
                "OPENAI_API_KEY / DEEPSEEK_API_KEY (matching the provider in "
                "model_routing) and re-run, OR disable test_generation by setting "
                "test_generation.enabled = false in .harness_config.json."
            ),
            error_code="ENV_MISCONFIG",
        )
        return {
            "exit_code": 1,
            "compiler_errors": [diagnostic],
            "loop_counter": loop_counter,
            "node_state": {
                "current_node": "test_generation",
                "env_misconfig": True,
                "env_misconfig_symbol": "llm_api_key",
                "test_generation": {
                    "status": "skipped",
                    "reason": "no_gateway",
                    "primary_stack": primary,
                },
            },
        }

    loop_counter = dict(state.get("loop_counter", {}))
    loop_counter["test_generation"] = loop_counter.get("test_generation", 0) + 1
    max_iterations = int(cfg.get("max_iterations", 2))
    if loop_counter["test_generation"] > max_iterations:
        logger.warning(
            "[test_generation_node] Max iterations (%d) reached. Routing to HITL.",
            max_iterations,
        )
        return {
            "loop_counter": loop_counter,
            "exit_code": 1,
            "compiler_errors": [_synth_diag(
                file="<test_generation>",
                message=(
                    f"test_generation_node exceeded max_iterations={max_iterations}. "
                    "The last attempt is reflected in the workspace; manual review needed."
                ),
                error_code="ENV_MISCONFIG",
            )],
            "node_state": {
                "current_node": "test_generation",
                "env_misconfig": True,
                "env_misconfig_symbol": "test_generation_max_iterations",
            },
        }

    # --- Build the LLM prompt ---
    from harness.gateway import NodeRole
    from harness.patcher import process_llm_patch_output

    test_guides_dir = os.path.join(os.path.dirname(__file__), "test_guides")
    guides_body = ""
    if os.path.isdir(test_guides_dir):
        # Reuse the existing style-guides loader directly — same frontmatter
        # filtering, same byte caps.
        # We override the search directory by calling the lower-level loader
        # explicitly: pass tags={primary, *tags} so applies_to: [<primary>]
        # files match.
        from harness.style_guides import _load_style_guides_markdown
        guides_body, _ = _load_style_guides_markdown(
            test_guides_dir,
            workspace_tags=(tags | {primary}),
        )

    messages = list(state.get("messages", []) or [])
    if guides_body:
        messages.append({
            "role": "system",
            "content": "## Test-generation guidance\n\n" + guides_body,
        })
    user_prompt = _build_test_gen_prompt(workspace_path, source_files, primary)
    messages.append({"role": "user", "content": user_prompt})

    budget = float(state.get("budget_remaining_usd", 2.00))

    try:
        response, new_budget = await gateway.dispatch(
            messages=list(messages),
            role=NodeRole.PATCHING,
            budget_remaining_usd=budget,
        )
    except RuntimeError as exc:
        logger.warning("[test_generation_node] Gateway refused: %s", exc)
        return {
            "loop_counter": loop_counter,
            "node_state": {
                "current_node": "test_generation",
                "test_generation": {"status": "gateway_error", "error": str(exc)},
            },
        }

    token_tracker = state.get("token_tracker", {})
    token_tracker = gateway.aggregate_tokens(token_tracker, response.usage)

    # --- Apply the patches ---
    # Constrain test placement to the workspace's source root + the
    # conventional test directories (mirrors the patching_node enforcement).
    # When _detect_source_root returns None (flat workspace), allowed_paths
    # is None and the pre-fix permissive behaviour applies.
    from harness.graph import _build_patcher_allowlist
    existing_modified = list(modified_files)
    allowed_paths = _build_patcher_allowlist(workspace_path)
    patch_results, new_modified = await process_llm_patch_output(
        response.content,
        workspace_path,
        existing_modified,
        allowed_paths=allowed_paths,
    )

    # Identify just the newly-applied test files (delta from existing_modified).
    generated_tests: list[str] = []
    for rel in new_modified:
        if rel in existing_modified:
            continue
        if not _inside_workspace(rel, workspace_path):
            # Defence in depth — patcher.safe_resolve should have already
            # rejected this. Log so an audit trail exists.
            logger.error(
                "[test_generation_node] Dropping out-of-workspace file from "
                "generated_tests: %r", rel,
            )
            continue
        generated_tests.append(rel)

    success_count = sum(1 for r in patch_results if r.success)
    fail_count = len(patch_results) - success_count
    logger.info(
        "[test_generation_node] LLM produced %d patch block(s), %d applied, "
        "%d failed. %d new test file(s).",
        len(patch_results), success_count, fail_count, len(generated_tests),
    )

    messages.append({"role": "assistant", "content": response.content})
    messages.append({
        "role": "system",
        "content": (
            f"[test_generation] Generated {len(generated_tests)} new test file(s): "
            f"{', '.join(generated_tests) if generated_tests else '(none)'}."
        ),
    })

    # --- Skip deterministic run when no tests landed ---
    if not generated_tests:
        logger.info(
            "[test_generation_node] No tests generated → skipping deterministic execution."
        )
        return {
            "messages": messages,
            "modified_files": new_modified,
            "generated_tests": list(state.get("generated_tests", [])),  # unchanged
            "token_tracker": token_tracker,
            "budget_remaining_usd": new_budget,
            "loop_counter": loop_counter,
            "node_state": {
                "current_node": "test_generation",
                "test_generation": {
                    "status": "passed",
                    "primary_stack": primary,
                    "tests_generated": 0,
                    "reason": "no_tests_generated",
                },
            },
        }

    # --- Deterministic test run ---
    test_cmd = _stack_test_command(primary)
    if test_cmd is None:
        logger.info(
            "[test_generation_node] No deterministic test command for stack=%s. "
            "Tests written but unverified.", primary,
        )
        return {
            "messages": messages,
            "modified_files": new_modified,
            "generated_tests": list(state.get("generated_tests", [])) + generated_tests,
            "token_tracker": token_tracker,
            "budget_remaining_usd": new_budget,
            "loop_counter": loop_counter,
            "node_state": {
                "current_node": "test_generation",
                "test_generation": {
                    "status": "passed",
                    "primary_stack": primary,
                    "tests_generated": len(generated_tests),
                    "reason": "no_runner_command_for_stack",
                },
            },
        }

    # When the stack is Python, ensure pytest has a config that uses the
    # importlib import mode. Without this, two same-named test files in
    # different directories (e.g. `tests/app/models/test_job.py` and
    # `tests/app/schemas/test_job.py`, both arising from a `job.py` source
    # in each package) collide on collection with the well-known
    # "import file mismatch: imported module 'test_job' has this __file__
    # attribute" error. importlib mode uses Python's package resolution
    # so the two coexist as distinct dotted names. Idempotent — leaves any
    # existing pytest config (pytest.ini / pyproject.toml / setup.cfg)
    # alone.
    if primary == "python":
        ensured = _ensure_pytest_importlib_config(workspace_path)
        if ensured:
            new_modified.append(ensured)

    from harness.sandbox import SandboxExecutor
    sandbox_cfg = dict(state.get("sandbox_config", {}) or {})
    allow_network = bool(state.get("allow_network", False))

    # The test command always contains a package-install token for stacks
    # that need one, so the sandbox auto-network heuristic kicks in. We
    # also lift it explicitly here so the SandboxExecutor sees it.
    if any(tok in test_cmd for tok in ("pip install", "npm install", "go get", "cargo")):
        allow_network = True

    # Adapt the sandbox image and root-FS writability to match the test
    # command's toolchain. Without this, `pip install pytest && pytest -q`
    # runs in the default ubuntu:22.04 base image, which has no pip
    # installed → exit 127 in 0.2s, and the LLM gets routed to a wasted
    # repair iteration with a spurious "test failure". compiler_node does
    # the same adaptation via _toolchain_image_for; reuse it here.
    from harness.graph import _toolchain_image_for, _build_command_needs_network
    desired_image = _toolchain_image_for(test_cmd)
    if desired_image and sandbox_cfg.get("docker_image") != desired_image:
        logger.info(
            "[test_generation_node] Adapting sandbox docker_image to '%s' "
            "to match toolchain implied by test command: %s",
            desired_image, test_cmd,
        )
        sandbox_cfg["docker_image"] = desired_image
    # Pip / npm / cargo / go install steps write into system locations the
    # read-only root FS would block; flip the flag when the test command
    # has an install step.
    if _build_command_needs_network(test_cmd) and sandbox_cfg.get("read_only_root", True):
        logger.info(
            "[test_generation_node] Adapting sandbox.read_only_root to False "
            "because test command installs packages into system locations: %s",
            test_cmd,
        )
        sandbox_cfg["read_only_root"] = False

    executor = SandboxExecutor(
        workspace_path=workspace_path,
        allow_network=allow_network,
        sandbox_config=sandbox_cfg,
    )

    logger.info(
        "[test_generation_node] Running deterministic test command for %s: %s",
        primary, test_cmd,
    )
    build_result = await executor.run(test_cmd)

    if build_result.exit_code == 0:
        logger.info(
            "[test_generation_node] Tests passed (%d test file(s) executed).",
            len(generated_tests),
        )
        return {
            "messages": messages,
            "modified_files": new_modified,
            "generated_tests": list(state.get("generated_tests", [])) + generated_tests,
            "token_tracker": token_tracker,
            "budget_remaining_usd": new_budget,
            "loop_counter": loop_counter,
            "node_state": {
                "current_node": "test_generation",
                "test_generation": {
                    "status": "passed",
                    "primary_stack": primary,
                    "tests_generated": len(generated_tests),
                    "test_command": test_cmd,
                },
            },
        }

    # --- Failures → flow into the standard repair path ---
    raw_diags = [d.to_dict() for d in build_result.diagnostics]
    if not raw_diags:
        # No structured parser hit — synthesize one from the tail.
        raw_diags = [_synth_diag(
            file="<test_runner>",
            message=(
                f"Generated tests failed (exit={build_result.exit_code}). "
                f"Command: {test_cmd}. "
                f"Tail: ...{(build_result.raw_output or '')[-1500:]}"
            ),
            error_code="TEST_FAILURE",
        )]
    else:
        # Tag every structured diagnostic with the TEST_FAILURE prefix so
        # repair_node's framing tweak knows these came from the test runner.
        for d in raw_diags:
            d["error_code"] = f"TEST_FAILURE:{d.get('error_code', 'unknown')}"

    logger.warning(
        "[test_generation_node] Tests failed (exit=%d, %d diagnostic(s)). "
        "Routing to repair.",
        build_result.exit_code, len(raw_diags),
    )

    return {
        "messages": messages,
        "modified_files": new_modified,
        "generated_tests": list(state.get("generated_tests", [])) + generated_tests,
        "exit_code": build_result.exit_code,
        "compiler_errors": raw_diags,
        "token_tracker": token_tracker,
        "budget_remaining_usd": new_budget,
        "loop_counter": loop_counter,
        "node_state": {
            "current_node": "test_generation",
            "last_build_output": build_result.raw_output,
            "test_generation": {
                "status": "failed",
                "primary_stack": primary,
                "tests_generated": len(generated_tests),
                "test_command": test_cmd,
                "test_failures": len(raw_diags),
            },
        },
    }


# ---------------------------------------------------------------------------
# Router (post-test_generation conditional edge)
# ---------------------------------------------------------------------------

def route_after_test_generation(state: dict[str, Any]) -> str:
    """Conditional edge router executed after test_generation_node.

    Decision matrix:
        env_misconfig flag set                 → human_intervention_node
            (covers the "no LLM gateway" gate and the "max iterations" cap)
        compiler_errors populated (TEST_FAILURE) → repair_node
        otherwise                              → lintgate_node
    """
    node_state = state.get("node_state", {}) or {}
    if node_state.get("env_misconfig"):
        return "human_intervention_node"
    if state.get("compiler_errors"):
        return "repair_node"
    return "lintgate_node"
