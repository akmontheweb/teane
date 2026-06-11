"""
LangGraph StateGraph topology, typed state schema, and all node implementations.

This module defines the complete graph execution pipeline:
    planning_node → patching_node → compiler_node
                                       │
                             ┌─────────┼──────────┐
                             │ exit 0  │ exit ≠ 0 │
                             ▼         ▼           │
                            END    loop < 3?       │
                                   │      │        │
                                yes      no        │
                                   │      │        │
                                   ▼      ▼        │
                              repair_node  human_intervention_node
                                   │              │
                                   └──────┬───────┘
                                          │
                                    compiler_node (re-validation)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Literal, Optional

from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Strongly-Typed State Schema (TypedDict + Pydantic)
# ---------------------------------------------------------------------------

class TokenTrackerDict(TypedDict, total=False):
    """Cumulative token cost tracking aggregated across all LLM calls."""
    total_input_tokens: int
    total_output_tokens: int
    total_cached_tokens: int
    total_cost_usd: float
    per_model: dict[str, dict[str, int | float]]


class DiagnosticObjectDict(TypedDict):
    """Structured compiler diagnostic parsed from build output."""
    file: str
    line: int
    column: int
    severity: Literal["error", "warning"]
    error_code: str
    message: str
    semantic_context: str


class MessageDict(TypedDict, total=False):
    """A single conversation turn in the messages array."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str]
    tool_calls: Optional[list[dict[str, Any]]]
    tool_call_id: Optional[str]


class AgentState(TypedDict, total=False):
    """
    The complete LangGraph agent state.

    Fields:
        workspace_path: Absolute path to the target repository root.
        messages: The conversation history array (system prompt anchored at index 0).
        modified_files: List of file paths edited during the active session.
        compiler_errors: Structured diagnostic objects from the last compiler run.
        token_tracker: Cumulative token usage and cost across all LLM calls.
        loop_counter: Tracks how many times each file/node has been visited
                      to prevent endless repair loops. Keys: 'patching', 'repair', 'compiler'.
        allow_network: Whether the sandbox namespace permits outbound network traffic.
        build_command: The resolved build command string (e.g., 'make build').
        budget_remaining_usd: Remaining dollar budget for LLM calls.
        session_id: Unique identifier for this graph invocation.
        exit_code: Last compiler exit code (0 = success, non-zero = failure).
        node_state: Internal routing metadata set by nodes.
    """
    workspace_path: str
    messages: list[MessageDict]
    modified_files: list[str]
    compiler_errors: list[DiagnosticObjectDict]
    token_tracker: TokenTrackerDict
    loop_counter: dict[str, int]
    allow_network: bool
    build_command: str
    budget_remaining_usd: float
    session_id: str
    exit_code: int
    node_state: dict[str, Any]
    current_gate: str  # "REQUIREMENTS" | "ARCHITECTURE" | "DEPLOYMENT" | ""
    spec_requirements_path: str
    spec_architecture_path: str
    deployment_blueprint_path: str
    skip_discovery: bool
    # Paths of test files the test_generation_node has written this session.
    # Telemetry / status display only; the patcher continues to use
    # `modified_files` as the source of truth for git staging.
    generated_tests: list[str]
    # Config carried in state so test_generation_node can read it without
    # round-tripping through cli.py's config loader. Loaded from the
    # "test_generation" section of cli.json / .harness_config.json.
    test_generation_config: dict[str, Any]
    # Reviewer LLM artifacts. Each is independently populated; either may be
    # absent if the corresponding *_reviewer_primary slot is unset.
    reviewer_comments_requirements: str
    reviewer_comments_code: str
    # Discovery-shaped follow-up questions the doc reviewer wants the user to
    # answer in a second pass of the interview loop.
    reviewer_followups: list[dict[str, Any]]

# ---------------------------------------------------------------------------
# 2. Default State Factory
# ---------------------------------------------------------------------------

def create_initial_state(
    *,
    workspace_path: str,
    initial_prompt: str,
    build_command: str,
    allow_network: bool = False,
    budget_usd: float = 2.00,
    session_id: str = "",
    spec_override: Optional[str] = None,
    skip_discovery: bool = False,
) -> AgentState:
    """
    Construct the initial graph state with anchored system prompt at messages[0]
    for maximum downstream prefix-caching discounts.

    If spec_override is provided (from --manifest requirement refinement),
    the SPEC_REQUIREMENTS.md content becomes the system prompt, replacing
    the default snapshot-based prompt. This makes the approved specification
    the immutable root context for all downstream nodes.
    """
    if spec_override:
        system_prompt = spec_override
        # When a user-approved spec already exists (from pre-flight --manifest
        # refinement), skip the graph's discovery pipeline completely. Otherwise
        # write_spec_node would overwrite the approved SPEC_REQUIREMENTS.md with
        # a minimal conversation-history compilation.
        skip_discovery = True
    else:
        system_prompt = _build_system_prompt(workspace_path, build_command)
    return AgentState(
        workspace_path=workspace_path,
        messages=[
            MessageDict(role="system", content=system_prompt),
            MessageDict(role="user", content=initial_prompt),
        ],
        modified_files=[],
        compiler_errors=[],
        token_tracker=TokenTrackerDict(
            total_input_tokens=0,
            total_output_tokens=0,
            total_cached_tokens=0,
            total_cost_usd=0.0,
            per_model={},
        ),
        loop_counter={
            "patching": 0, "repair": 0, "compiler": 0, "total_repairs": 0,
            "review_spec": 0, "review_code": 0,
        },
        allow_network=allow_network,
        build_command=build_command,
        budget_remaining_usd=budget_usd,
        session_id=session_id,
        exit_code=-1,
        node_state={},
        skip_discovery=skip_discovery,
    )


# Files that conventionally live at workspace root and are exempt from the
# "all source under <root>/" enforcement when a source root is detected.
# patching_node / repair_node / test_generation_node compose the patcher
# allowlist as [<source_root>/, tests/, test/, __tests__/, *_ROOT_ALLOWLIST_FILES].
_ROOT_ALLOWLIST_FILES: frozenset[str] = frozenset({
    "setup.py", "setup.cfg", "pyproject.toml",
    "conftest.py", "manage.py", "__init__.py",
    "wsgi.py", "asgi.py", "main.py",
    "tox.ini", "pytest.ini", "MANIFEST.in", ".gitignore",
    # Dependency manifests — must be in the static set so the LLM can
    # CREATE them (e.g. add pytest after env_misconfig HITL), not just
    # amend existing ones via the requirements*.txt scan below.
    "requirements.txt",
})


def _build_patcher_allowlist(workspace_path: str) -> Optional[list[str]]:
    """Return the patcher allowed_paths list for ``workspace_path``.

    When a source root is detected, returns the focused allowlist:
      - the source root itself as a directory prefix (e.g. ``"app/"``),
      - the conventional test trees,
      - the conventionally-root files in :data:`_ROOT_ALLOWLIST_FILES`,
      - any ``requirements*.txt`` actually present at the workspace root.

    When detection fails (flat / ambiguous workspaces) we used to return
    ``None`` which the patcher reads as "allow ANY path under the
    workspace tree". That gave the LLM unconstrained write access to
    ``.git/`` config, dotfiles, and anywhere else in the tree — well
    outside the intent of "edit the source." Instead we return a
    conservative best-guess allowlist covering the common source layouts
    (``src/``, ``lib/``, ``app/``, ``pkg/``, ``cmd/``) plus the standard
    test trees and root manifest files. If the LLM genuinely needs to
    write somewhere outside that, the patcher will reject the write and
    the operator can either fix the layout heuristic in ``harness.impact``
    or add a top-level entry to the allowlist explicitly.

    Mirrors the language used in the system prompt's "Workspace Layout"
    section, so the LLM sees the same rules as the patcher applies.
    """
    from harness.impact import (
        _detect_source_root,
        _is_greenfield_workspace,
        _workspace_basename_variants,
    )
    root = _detect_source_root(workspace_path)

    if root:
        allowlist: list[str] = [
            f"{root}/",
            "tests/", "test/", "__tests__/",
            *_ROOT_ALLOWLIST_FILES,
        ]
    elif _is_greenfield_workspace(workspace_path):
        # Greenfield project (no source files yet) — the LLM IS the one
        # defining the layout, so constraining its package directory name
        # to "must match the workspace basename" is the wrong rule. The
        # LLM often picks a descriptive name (e.g. `job_queue/` for a
        # workspace called `TaskDispatcher`) — a reasonable engineering
        # choice that the basename-only allowlist used to reject.
        #
        # Return ``None`` so the patcher skips the prefix check entirely.
        # The patcher's :func:`harness.trust.safe_resolve` still blocks
        # path traversal (``../``) and absolute paths, which is the only
        # real safety concern for a fresh workspace. Subsequent runs will
        # detect the source root the LLM picked and lock the allowlist
        # to it (regular ``if root:`` branch above).
        logger.info(
            "[allowlist] Greenfield workspace %s — no allowlist; the LLM "
            "defines the layout. Path-traversal still blocked by the "
            "patcher's safe_resolve guard.",
            workspace_path,
        )
        return None
    else:
        # Conservative fallback when no source root is detected but the
        # workspace also isn't truly greenfield (e.g. leftover files from a
        # previous abandoned session). Still include the workspace's
        # basename variants — without them, the LLM's natural choice of a
        # package directory matching the project name (e.g. task_dispatcher/
        # for a workspace called TaskDispatcher) gets rejected on every
        # patch attempt and the repair loop burns through to HITL.
        basename_dirs = [
            f"{name}/" for name in _workspace_basename_variants(workspace_path)
        ]
        logger.warning(
            "[allowlist] No source root detected for %s — falling back to "
            "permissive allowlist (src/, lib/, app/, pkg/, cmd/, tests/ + "
            "basename variants %s). Workspace not greenfield, so this "
            "likely contains stale files; the LLM still needs a place to "
            "put new code matching the project name.",
            workspace_path, basename_dirs,
        )
        allowlist = [
            "src/", "lib/", "app/", "pkg/", "cmd/", "internal/",
            "tests/", "test/", "__tests__/",
            *basename_dirs,
            *_ROOT_ALLOWLIST_FILES,
        ]

    # Pick up any requirements*.txt actually present so the LLM can amend
    # them without the patcher rejecting the write.
    try:
        for entry in os.listdir(workspace_path):
            if entry.startswith("requirements") and entry.endswith(".txt"):
                allowlist.append(entry)
    except OSError:
        pass

    return allowlist


def _build_system_prompt(workspace_path: str, build_command: str) -> str:
    """
    Construct the static, immutable system prompt anchored at messages[0].
    This prompt is never mutated or truncated — it maximizes prefix caching
    across all downstream LLM calls because its position and content are fixed.
    """
    tree = _snapshot_directory_tree(workspace_path)

    # Redact secrets from the directory tree snapshot if redactor is active
    try:
        from harness.redactor import redact_text
        tree = redact_text(tree)
    except ImportError:
        pass

    # --- Two-Tier Skills System (language-aware filtering) ---
    # Detect the workspace's stack so we only inject skills the LLM will
    # actually use. A pure FastAPI project doesn't need the Angular skill
    # in its system prompt; trimming saves ~2-3 KB per call and reduces
    # noise in the prompt.
    from harness.impact import _detect_workspace_stack, _detect_source_root
    workspace_tags = _detect_workspace_stack(workspace_path)
    source_root = _detect_source_root(workspace_path)

    # Tier 1: Harness skills (harness/skills/*.md) — agent standards +
    # stack-specific skills filtered by `applies_to:` frontmatter.
    harness_skills = _load_skills_markdown(
        os.path.join(os.path.dirname(__file__), "skills"),
        max_file_chars=4000,
        workspace_tags=workspace_tags,
    )
    if harness_skills:
        harness_skills = f"## Agent Skills & Standards\n{harness_skills}\n"

    # Tier 2: Project skills ({workspace_path}/skills/*.md) — per-project
    # conventions. Same filter applies so user-supplied skills can also
    # opt-in via frontmatter; skills without frontmatter always load.
    project_skills = _load_skills_markdown(
        os.path.join(workspace_path, "skills"),
        max_file_chars=3000,
        workspace_tags=workspace_tags,
    )
    if project_skills:
        project_skills = f"## Project Skills & Conventions\n{project_skills}\n"

    # --- Technology-Specific Style Guides ---
    # Two-tier like skills: shipped defaults under harness/style_guides/
    # + per-project overrides under {workspace}/style_guides/. Both are
    # filtered by `applies_to:` frontmatter against the detected stack
    # so e.g. a pure Python project never sees React style content.
    from harness.style_guides import load_style_guides
    style_guides = load_style_guides(workspace_path, workspace_tags=workspace_tags)
    if style_guides:
        style_guides = f"## Coding Style Guides\n{style_guides}\n"

    # --- Workspace Layout Constraint ---
    # When the workspace has a clear source root (e.g. `app/`, `src/`, `lib/`),
    # tell the LLM that new source files MUST land there. Paired with the
    # `allowed_paths` enforcement in patching_node / repair_node — files
    # outside the allowlist are rejected with a clear error so the LLM
    # tries again with the constraint in mind.
    layout_block = ""
    if source_root:
        layout_block = (
            f"## Workspace Layout (mandatory)\n"
            f"The workspace organizes its source under `{source_root}/`. "
            f"**All new source files MUST be created under `{source_root}/`.** "
            f"Do NOT place new modules at workspace root.\n\n"
            f"The only files that may live at workspace root are: "
            f"`setup.py`, `setup.cfg`, `pyproject.toml`, `conftest.py`, "
            f"`manage.py`, `__init__.py`, `wsgi.py`, `asgi.py`, `main.py`, "
            f"`requirements*.txt`, `tox.ini`, `pytest.ini`, `MANIFEST.in`, "
            f"`.gitignore`. Test files live under `tests/`, `test/`, or "
            f"`__tests__/` per the language convention. CREATE_FILE blocks "
            f"that target other root paths will be rejected by the patcher.\n"
        )

    return f"""You are an expert software engineer with deep knowledge of the codebase below.

## Repository Root
{workspace_path}

## Directory Structure (snapshot at invocation)
{tree}
{layout_block}{harness_skills if harness_skills else ""}{project_skills if project_skills else ""}{style_guides if style_guides else ""}
## Build Command
{build_command}

## Dependency Manifest Coherence (mandatory)
The build command above is exactly what the sandbox will execute. Every CLI
the build invokes (test runner, linter, type checker, formatter) MUST be
declared in the workspace's dependency manifest — otherwise the install
step succeeds but the next step fails with "No module named X" /
"command not found", and the run wastes a repair iteration.

Audit the build command before writing any manifest:
  - `pip install -r requirements.txt && pytest`  → `requirements.txt`
    must list `pytest` (and `pytest-asyncio`, `ruff`, `mypy`, etc. if the
    command invokes them).
  - `pip install -e '.[dev]' && pytest`  → `pytest` must live under
    `[project.optional-dependencies].dev` in `pyproject.toml`.
  - `npm install && npm test`  → the test runner referenced by the `test`
    script must be in `package.json` `devDependencies`.
  - `cargo build && cargo test`  → cargo bundles the runner; no extra dep.
  - `go build ./... && go test ./...`  → go bundles the runner; no extra dep.

When you generate or amend a manifest, include every tool the build needs.
A clean separation between runtime and dev dependencies is fine, but BOTH
must be installable by the build command.

## Your Role
You are an autonomous coding agent. You will receive tasks and must:
1. Plan the implementation strategy before writing code.
2. Generate precise code patches using a strict SEARCH/REPLACE syntax.
3. Only modify files that need changes — never touch unrelated code.

## Patch Syntax
When applying patches, use these exact formats:

### REPLACE_BLOCK
```
<<<REPLACE_BLOCK>>>
file: path/to/file.ext
search:
<exact lines to find>
replace:
<exact replacement lines>
<<<END_REPLACE_BLOCK>>>
```

### CREATE_FILE
```
<<<CREATE_FILE>>>
file: path/to/new/file.ext
content:
<complete file contents>
<<<END_CREATE_FILE>>>
```

### DELETE_BLOCK
```
<<<DELETE_BLOCK>>>
file: path/to/file.ext
search:
<exact lines to delete>
<<<END_DELETE_BLOCK>>>
```

### INSERT_AT_BLOCK
```
<<<INSERT_AT_BLOCK>>>
file: path/to/file.ext
anchor: <function or class name to insert relative to>
placement: before|after
content:
<lines to insert>
<<<END_INSERT_AT_BLOCK>>>
```

## Code Quality Standards
- Write modular, self-contained functions/classes with single responsibility.
- Include proper error handling: try/except, input validation, graceful fallbacks.
- Add type hints (Python), type annotations (TypeScript), or equivalent in the target language.
- Use meaningful variable/function names; include docstrings and inline comments.
- Follow the principle of least surprise — never modify unrelated code or files.
- Handle edge cases: empty inputs, None/null values, network failures, timeouts.
- Write production-ready code: no debug print statements, no hardcoded secrets or credentials.
- Prefer composition over inheritance; keep coupling loose and interfaces clean.

## Rules
- Never remove or alter existing comments unless instructed.
- Preserve existing indentation and code style.
- If you are unsure about a change, ask for clarification rather than guessing.
"""


def _snapshot_directory_tree(path: str, max_depth: int = 4, max_files_per_dir: int = 50) -> str:
    """
    Generate a lightweight directory tree snapshot for the system prompt.
    Limits depth and file count to avoid bloating the prompt.
    """
    lines: list[str] = []
    try:
        for root, dirs, files in os.walk(path):
            depth = root[len(path):].count(os.sep)
            if depth > max_depth:
                dirs.clear()
                continue
            # Skip hidden and common noise directories
            dirs[:] = [
                d
                for d in sorted(dirs)
                if not d.startswith(".")
                and d not in ("node_modules", "__pycache__", "target", "build", "dist", ".git")
            ]
            indent = "  " * (depth + 1)
            rel = os.path.relpath(root, path)
            if rel == ".":
                lines.append(f"{os.path.basename(path)}/")
            else:
                lines.append(f"{indent[:-2]}{os.path.basename(root)}/")
            shown = sorted(files)[:max_files_per_dir]
            for f in shown:
                lines.append(f"{indent}{f}")
            if len(files) > max_files_per_dir:
                lines.append(f"{indent}... ({len(files) - max_files_per_dir} more files)")
    except (OSError, PermissionError) as exc:
        # Surface as a WARNING so operators see the failure in logs — the
        # previous silent return injected the error string straight into the
        # LLM system prompt with no other signal, and the LLM would then
        # hallucinate file paths against a workspace it cannot see.
        logger.warning(
            "[graph] Could not snapshot directory tree at %s: %s", path, exc,
        )
        lines.append(f"[Error reading directory: {exc}]")
    if not lines:
        return (
            f"[Unable to read directory structure at {path!s}. "
            "The workspace appears empty or inaccessible — patches generated "
            "against this snapshot will likely target non-existent files.]"
        )
    return "\n".join(lines)


_APPLIES_TO_RE = re.compile(
    r'^---\s*\n\s*applies_to\s*:\s*\[([^\]]*)\]\s*\n---\s*\n',
    re.MULTILINE,
)


def _parse_skill_frontmatter(content: str) -> tuple[Optional[set[str]], str]:
    """Extract the ``applies_to:`` tag list from a skill file's frontmatter.

    Recognises the minimal form::

        ---
        applies_to: [tag1, tag2]
        ---

        ... rest of the skill ...

    Returns ``(tags, body)`` where ``tags`` is ``None`` when no frontmatter
    is present (skill loads unconditionally) or a set of tag strings when
    the frontmatter declares them. ``body`` is the markdown content with
    the frontmatter stripped.

    This is a deliberately tiny hand-rolled parser — we don't want a YAML
    dependency for a one-field schema.
    """
    m = _APPLIES_TO_RE.match(content)
    if not m:
        return None, content
    tag_blob = m.group(1)
    tags = {t.strip() for t in tag_blob.split(",") if t.strip()}
    body = content[m.end():]
    return tags, body


def _load_skills_markdown(
    skills_dir: str,
    max_file_chars: int = 4000,
    workspace_tags: Optional[set[str]] = None,
) -> str:
    """
    Scan a skills/ directory for .md files and return their concatenated content.

    Each file's content is truncated to ``max_file_chars`` to prevent system
    prompt bloat. Files are sorted alphabetically for deterministic ordering.

    When ``workspace_tags`` is provided, skills with an ``applies_to:``
    frontmatter are loaded only if at least one of their declared tags
    appears in ``workspace_tags``. Skills with no frontmatter (e.g. the
    universal ``agent-standards.md``) always load.

    Args:
        skills_dir: Absolute path to the skills directory.
        max_file_chars: Maximum characters to read per skill file.
        workspace_tags: Tags returned by ``impact._detect_workspace_stack``.
                        ``None`` disables filtering (legacy behavior).

    Returns:
        Concatenated markdown content (frontmatter stripped), or empty
        string if no skills match.
    """
    if not os.path.isdir(skills_dir):
        return ""
    parts: list[str] = []
    try:
        for fname in sorted(os.listdir(skills_dir)):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(skills_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as sf:
                    content = sf.read(max_file_chars)
            except OSError:
                logger.warning("[graph] Could not read skills file: %s", fpath)
                continue

            applies_to, body = _parse_skill_frontmatter(content)

            # Filter by workspace tags when both sides have something to say.
            # Skills with no frontmatter (applies_to is None) load
            # unconditionally — that's the "universal skill" pattern used by
            # agent-standards.md and any user-supplied project skill.
            if workspace_tags is not None and applies_to is not None:
                if not (applies_to & workspace_tags):
                    logger.debug(
                        "[graph] Skipping skill %s (applies_to=%s, workspace=%s)",
                        fname, sorted(applies_to), sorted(workspace_tags),
                    )
                    continue

            if body.strip():
                parts.append(body)
    except OSError:
        return ""
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 2b. Gateway Injection Mechanism
# ---------------------------------------------------------------------------

_gateway: Optional[Any] = None
_gateway_config: Optional[Any] = None


def set_gateway(gateway: Any) -> None:
    """
    Inject the LLM Gateway instance for use by graph nodes.
    Called by the CLI layer before running the graph.

    Also stashes ``gateway.config`` via :func:`set_gateway_config` so the
    reviewer nodes (``spec_review_node``, ``code_review_node``) and the
    pre-flight reviewer block in ``cmd_run`` see a non-None config. Prior
    to this, ``set_gateway_config`` had no caller anywhere — every
    ``get_gateway_config()`` consumer read None and silently skipped,
    which made every configured reviewer LLM effectively dead code.
    """
    global _gateway
    _gateway = gateway
    config = getattr(gateway, "config", None)
    if config is not None:
        set_gateway_config(config)
    logger.info("[graph] Gateway instance injected.")


def get_gateway() -> Optional[Any]:
    """Retrieve the injected Gateway instance, or None if not set."""
    return _gateway


def set_gateway_config(config: Any) -> None:
    """Inject GatewayConfig for node model selection."""
    global _gateway_config
    _gateway_config = config


def get_gateway_config() -> Optional[Any]:
    """Retrieve the injected GatewayConfig."""
    return _gateway_config


# ---------------------------------------------------------------------------
# 3. Memory Cleanse Utility (Module 4)
# ---------------------------------------------------------------------------

def apply_memory_cleanse(state: AgentState, resolution_kind: str = "compiler_success") -> dict[str, Any]:
    """
    Purge verbose intermediate repair-loop messages from the conversation history
    and compress the debugging session into a single structured summary line.

    Triggered on:
        - Compiler exit code 0 (resolution_kind = "compiler_success")
        - Human intervention resolution (resolution_kind = "human_intervention")

    Retains:
        - messages[0] (system prompt — never truncated)
        - The original planning/user message
        - The final successful patching message (last assistant message before the cleanse)
        - The compression summary injected as a system-tagged message

    Returns:
        A state update dict with cleansed messages that should be merged into the state.
    """
    messages: list[MessageDict] = state.get("messages", [])
    if len(messages) < 3:
        return {}  # Nothing to cleanse

    # Locate the original planning exchange: first user message after system prompt
    system_prompt = messages[0] if messages else None
    planning_message: Optional[MessageDict] = None
    for m in messages[1:]:
        if m.get("role") == "user":
            planning_message = m
            break

    # The final assistant message (the patch that succeeded)
    final_patch: Optional[MessageDict] = None
    for m in reversed(messages):
        if m.get("role") == "assistant":
            final_patch = m
            break

    # Collect metrics for the compression summary
    repair_iterations = state.get("loop_counter", {}).get("total_repairs", 0)
    token_tracker: TokenTrackerDict = state.get("token_tracker", {})
    debug_cost = token_tracker.get("total_cost_usd", 0.0)
    debug_input = token_tracker.get("total_input_tokens", 0)
    debug_output = token_tracker.get("total_output_tokens", 0)

    # Determine which file was fixed
    target_file = "unknown"
    modified: list[str] = state.get("modified_files", [])
    if modified:
        target_file = modified[-1]
    elif state.get("compiler_errors"):
        errs: list[DiagnosticObjectDict] = state.get("compiler_errors", [])
        if errs:
            target_file = errs[0].get("file", "unknown")

    # Build the compression summary
    if resolution_kind == "human_intervention":
        summary = (
            f"[System]: Resolved via manual human intervention.\n"
            f"  Target File: {target_file}\n"
            f"  Iterations Before Intervention: {repair_iterations}\n"
            f"  Debug Token Cost: ${debug_cost:.6f} (In: {debug_input}, Out: {debug_output})"
        )
    else:
        summary = (
            f"[System]: Successfully resolved compilation fault in file {target_file}. "
            f"({repair_iterations} repair iteration(s), "
            f"${debug_cost:.6f} debug token cost, "
            f"In: {debug_input}, Out: {debug_output})"
        )

    # Reconstruct the cleansed message list
    cleansed: list[MessageDict] = []
    if system_prompt is not None:
        cleansed.append(system_prompt)
    if planning_message is not None:
        cleansed.append(planning_message)
    if final_patch is not None:
        cleansed.append(final_patch)
    cleansed.append(MessageDict(role="system", content=summary))

    logger.info(
        "[memory_cleanse] Compressed %d messages → %d messages. File: %s, Iterations: %d, Cost: $%.6f",
        len(messages), len(cleansed), target_file, repair_iterations, debug_cost,
    )

    return {"messages": cleansed}


# ---------------------------------------------------------------------------
# 4. Node Implementations (Gateway-Integrated)
# ---------------------------------------------------------------------------

async def planning_node(state: AgentState) -> dict[str, Any]:
    """
    Node 1: The Architect.

    Uses the configured reasoning model (planning_primary in .harness_config.json)
    via the model-agnostic gateway to generate an implementation blueprint.
    The blueprint is a structured plan that the patching_node will execute.

    This node reads the user's task from the last user message and produces
    a detailed step-by-step implementation plan.
    """
    logger.info("[planning_node] Generating implementation blueprint...")

    messages: list[MessageDict] = list(state.get("messages", []))
    loop_counter = state.get("loop_counter", {})
    loop_counter = dict(loop_counter)
    loop_counter["planning"] = loop_counter.get("planning", 0) + 1

    gateway = get_gateway()
    if gateway is None:
        logger.error("[planning_node] No gateway configured. Cannot call LLM.")
        return {
            "node_state": {"current_node": "planning", "error": "No gateway configured"},
            "loop_counter": loop_counter,
        }

    try:
        from harness.gateway import NodeRole

        budget = state.get("budget_remaining_usd", 2.00)
        response, new_budget = await gateway.dispatch(
            messages=list(messages),
            role=NodeRole.PLANNING,
            budget_remaining_usd=budget,
        )

        # Update token tracker
        token_tracker = state.get("token_tracker", {})
        token_tracker = gateway.aggregate_tokens(token_tracker, response.usage)

        # Append the planning response to messages
        messages.append(MessageDict(role="assistant", content=response.content))

        logger.info(
            "[planning_node] Blueprint generated. tokens_in=%d tokens_out=%d cost=$%.6f budget_left=$%.4f",
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.cost_usd,
            new_budget,
        )

        return {
            "messages": messages,
            "token_tracker": token_tracker,
            "budget_remaining_usd": new_budget,
            "loop_counter": loop_counter,
            "node_state": {"current_node": "planning", "plan_complete": True},
        }
    except RuntimeError as exc:
        # Budget exhausted or other guardrail triggered
        logger.warning("[planning_node] Gateway refused: %s", exc)
        return {
            "node_state": {"current_node": "planning", "error": str(exc), "budget_exhausted": True},
            "loop_counter": loop_counter,
        }
    except Exception as exc:
        logger.exception("[planning_node] Unexpected error during LLM call.")
        return {
            "node_state": {"current_node": "planning", "error": str(exc)},
            "loop_counter": loop_counter,
        }


async def patching_node(state: AgentState) -> dict[str, Any]:
    """
    Node 2: The Builder.

    Uses the configured fast code model (patching_primary in .harness_config.json)
    via the gateway to parse the planning blueprint and produce precise
    SEARCH/REPLACE blocks that are written to disk by the hybrid patcher engine.

    Output: Updates `modified_files` with paths of files that were changed.
    """
    logger.info("[patching_node] Generating and applying code patches...")

    loop_counter = state.get("loop_counter", {})
    loop_counter = dict(loop_counter)
    loop_counter["patching"] = loop_counter.get("patching", 0) + 1

    gateway = get_gateway()
    if gateway is None:
        logger.error("[patching_node] No gateway configured. Cannot call LLM.")
        return {
            "node_state": {"current_node": "patching", "error": "No gateway configured"},
            "loop_counter": loop_counter,
        }

    try:
        from harness.gateway import NodeRole
        from harness.patcher import process_llm_patch_output

        messages = list(state.get("messages", []))
        budget = state.get("budget_remaining_usd", 2.00)

        # Inject a format reminder to ensure the LLM outputs patch blocks
        _FORMAT_REMINDER = """[CRITICAL FORMAT INSTRUCTION]
You MUST respond using ONLY the patch block syntax below. Do NOT include any explanations,
markdown code fences, or text outside the blocks. Your entire response must be parseable
as one or more patch blocks.

Valid blocks:
<<<CREATE_FILE>>>
file: path/to/file.ext
content:
<complete file contents>
<<<END_CREATE_FILE>>>

<<<REPLACE_BLOCK>>>
file: path/to/file.ext
search:
<exact lines to find>
replace:
<exact replacement lines>
<<<END_REPLACE_BLOCK>>>

<<<DELETE_BLOCK>>>
file: path/to/file.ext
search:
<exact lines to delete>
<<<END_DELETE_BLOCK>>>

<<<INSERT_AT_BLOCK>>>
file: path/to/file.ext
anchor: <function or class name>
placement: before|after
content:
<lines to insert>
<<<END_INSERT_AT_BLOCK>>>

Quality: Write modular, production-ready code with proper error handling, type hints, and docstrings. Handle edge cases.
Generate your patches NOW. Only the blocks above. No other text."""
        messages.append({"role": "user", "content": _FORMAT_REMINDER})

        response, new_budget = await gateway.dispatch(
            messages=list(messages),
            role=NodeRole.PATCHING,
            budget_remaining_usd=budget,
        )

        # Update token tracker
        token_tracker = state.get("token_tracker", {})
        token_tracker = gateway.aggregate_tokens(token_tracker, response.usage)

        # Apply patches to disk. Constrain new source files to the detected
        # source root (e.g. `app/`, `src/`) when one exists, so the LLM
        # can't accidentally place new modules at workspace root.
        workspace = state.get("workspace_path", os.getcwd())
        existing_modified = list(state.get("modified_files", []))
        allowed_paths = _build_patcher_allowlist(workspace)
        patch_results, modified_files = await process_llm_patch_output(
            response.content,
            workspace,
            existing_modified,
            allowed_paths=allowed_paths,
        )

        # Append the LLM response to messages
        messages.append(MessageDict(role="assistant", content=response.content))

        # Report patch application results
        success_count = sum(1 for r in patch_results if r.success)
        fail_count = len(patch_results) - success_count
        # Carve out allowlist rejections from generic failures so the next
        # repair iteration sees the exact paths and reason — without this,
        # the LLM keeps re-proposing the same blocked paths.
        allowlist_rejections = [
            {"file": r.file, "operation": r.operation, "reason": r.error}
            for r in patch_results
            if not r.success and isinstance(r.error, str)
            and "not in skill allowlist" in r.error
        ]
        if success_count > 0:
            status_msg = f"[System]: Applied {success_count}/{len(patch_results)} patches successfully."
            if fail_count > 0:
                failed_files = [r.file for r in patch_results if not r.success]
                status_msg += f" Failed on: {', '.join(failed_files)}."
        else:
            status_msg = f"[System]: Failed to apply {fail_count} patch(es)."
        if allowlist_rejections:
            rejected_paths = ", ".join(sorted({r["file"] for r in allowlist_rejections}))
            status_msg += (
                f"\n[Allowlist] Rejected paths outside the configured layout: "
                f"{rejected_paths}. Allowed roots: {allowed_paths}."
            )
        messages.append(MessageDict(role="system", content=status_msg))

        logger.info(
            "[patching_node] Patches applied. tokens_in=%d tokens_out=%d cost=$%.6f budget_left=$%.4f "
            "patches=%d succeed=%d fail=%d",
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.cost_usd,
            new_budget,
            len(patch_results), success_count, fail_count,
        )

        return {
            "messages": messages,
            "modified_files": modified_files,
            "token_tracker": token_tracker,
            "budget_remaining_usd": new_budget,
            "loop_counter": loop_counter,
            "node_state": {
                "current_node": "patching",
                "patch_complete": True,
                "patch_success": success_count,
                "patch_fail": fail_count,
                "allowlist_rejections": allowlist_rejections,
                "allowed_paths": allowed_paths,
            },
        }
    except RuntimeError as exc:
        logger.warning("[patching_node] Gateway refused: %s", exc)
        return {
            "node_state": {"current_node": "patching", "error": str(exc), "budget_exhausted": True},
            "loop_counter": loop_counter,
        }
    except Exception as exc:
        logger.exception("[patching_node] Unexpected error during patching.")
        return {
            "node_state": {"current_node": "patching", "error": str(exc)},
            "loop_counter": loop_counter,
        }


def _toolchain_image_for(build_command: str) -> Optional[str]:
    """Pick a sandbox image that ships with the toolchain implied by the
    given build command. Returns None when we have no opinion (so caller
    keeps whatever is configured)."""
    cmd = build_command.lower()
    if "python3" in cmd or "pip " in cmd or "pytest" in cmd or "poetry" in cmd:
        return "python:3.12-slim"
    if "npm " in cmd or "yarn " in cmd or "pnpm " in cmd or "node " in cmd:
        return "node:20-slim"
    if "cargo " in cmd:
        return "rust:1.79-slim"
    if cmd.strip().startswith("go ") or " go build" in cmd or " go test" in cmd:
        return "golang:1.22"
    return None


def _build_command_needs_network(build_command: str) -> bool:
    """True when the build command performs a package install that needs
    to reach a registry (pip/npm/yarn/pnpm/cargo/go)."""
    cmd = build_command.lower()
    return any(token in cmd for token in (
        "pip install", "pip3 install", "npm install", "yarn install",
        "pnpm install", "cargo build", "cargo test", "go mod",
        "go get", "poetry install",
    ))


# Patterns that mean the build container is missing a required runtime
# (interpreter, test framework, package manager) — NOT a code bug. When
# one of these fires we route to HITL immediately instead of burning the
# whole 3-iteration repair budget on a problem the LLM cannot fix from
# inside the sandbox.
#
# Split by source kind because the repairability heuristic differs:
#   - Python ModuleNotFoundError → ALWAYS repairable. The fix is "add the
#     missing distribution to requirements.txt / pyproject deps", which
#     the autofix / repair LLM can handle for any single-segment module
#     name (httpx, fastapi, pydantic, sqlalchemy — every regular pip
#     package). The earlier whitelist-only policy meant any framework
#     dep outside the test/lint tool set short-circuited to HITL.
#   - Shell `command not found` → repairable only when the command is a
#     known pip-installable Python tool (pytest, ruff, mypy, ...). For
#     anything else (npm, cargo, go, docker) the container needs a
#     different base image, which only the operator can change.
_PYTHON_MODULE_MISS_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Python: "/usr/local/bin/python3: No module named pytest"
    # Dotted names (e.g. 'api.database') are excluded — those signal a
    # local-import bug in the user's code, which the repair loop can fix.
    re.compile(r"(?m)^/[^:\s]+/python3?: No module named (?P<sym>[^.\s]+)\s*$"),
    # Python: "ModuleNotFoundError: No module named 'pytest'"
    # Same dotted-name exclusion as above.
    re.compile(r"ModuleNotFoundError: No module named ['\"](?P<sym>[^'\".]+)['\"]"),
)

_SHELL_COMMAND_MISS_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Shell-style "<cmd>: command not found" (covers npm, cargo, go, etc.)
    re.compile(r"(?m)^(?:/bin/sh: \d+: )?(?P<sym>[\w.\-+]+): command not found\s*$"),
    re.compile(r"(?m)^(?:bash: )?(?P<sym>[\w.\-+]+): command not found\s*$"),
    # Dash / busybox shells say "X: not found" (no "command"). Often
    # prefixed with "/bin/sh: 1: " in the alpine/slim images.
    re.compile(r"(?m)^/bin/sh: \d+: (?P<sym>\S+): not found\s*$"),
    # Docker entrypoint missing (exec format / OCI runtime error)
    re.compile(r"executable file not found in \$?PATH: (?P<sym>\S+)"),
    re.compile(r'exec: "(?P<sym>[^"]+)": executable file not found'),
    # Node: "node: not found" or "npm: not found"
    re.compile(r"(?m)^(?P<sym>node|npm|yarn|pnpm): not found\s*$"),
)

# Composite — preserved for callers that don't care about the source kind.
_ENV_MISCONFIG_PATTERNS: tuple[re.Pattern[str], ...] = (
    *_PYTHON_MODULE_MISS_PATTERNS,
    *_SHELL_COMMAND_MISS_PATTERNS,
)


def _is_env_misconfig(
    raw_output: str,
    workspace_path: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """Return ``(symbol, kind)`` where ``kind`` is ``"python"`` for a
    Python ``ModuleNotFoundError`` or ``"shell"`` for a shell
    ``command not found`` — or ``None`` if the build output doesn't
    look like an environment miss.

    The caller uses ``kind`` to decide repairability:
      - ``"python"`` → always repairable as MISSING_DEP. Any single-segment
        Python module name is a pip-installable distribution; autofix R4
        (``_try_missing_dep``) lands the requirements.txt edit without an
        LLM call. The earlier whitelist-only policy meant common app
        deps like ``httpx`` / ``fastapi`` / ``pydantic`` short-circuited
        to HITL for no good reason — they're as fixable as ``pytest``.
      - ``"shell"`` → repairable only when the symbol matches
        ``_PIP_INSTALLABLE_SYMBOLS`` (pytest, ruff, ...). Everything else
        (npm, cargo, go, docker) needs a different base image and the
        repair LLM can't fix it from inside the sandbox.

    When ``workspace_path`` is supplied and the matched symbol corresponds
    to a directory or top-level Python module in the workspace (e.g.
    ``task_dispatcher`` is the project's own package under ``src/``), we
    return ``None`` — the import error is a layout / sys.path bug in the
    user's own code, which the repair LLM can fix differently
    (conftest.py, pyproject src-layout, etc.).
    """
    if not raw_output:
        return None
    # Scan only the tail — these errors land at the end of the log when
    # the missing-binary line is the last thing the container prints.
    tail = raw_output[-4000:]
    for pattern in _PYTHON_MODULE_MISS_PATTERNS:
        match = pattern.search(tail)
        if match:
            sym = match.group("sym").strip("'\"")
            if workspace_path and _symbol_exists_in_workspace(sym, workspace_path):
                logger.info(
                    "[env_misconfig] '%s' looks like a missing module but a "
                    "directory or file by that name exists in %s — treating "
                    "as a user-code import bug, not env misconfig.",
                    sym, workspace_path,
                )
                return None
            return (sym, "python")
    for pattern in _SHELL_COMMAND_MISS_PATTERNS:
        match = pattern.search(tail)
        if match:
            sym = match.group("sym").strip("'\"")
            if workspace_path and _symbol_exists_in_workspace(sym, workspace_path):
                logger.info(
                    "[env_misconfig] '%s' looks like a missing command but a "
                    "directory or file by that name exists in %s — treating "
                    "as a user-code import bug, not env misconfig.",
                    sym, workspace_path,
                )
                return None
            return (sym, "shell")
    return None


def _symbol_exists_in_workspace(symbol: str, workspace_path: str) -> bool:
    """True when ``symbol`` matches a directory or Python module file in the
    workspace tree (anywhere under root, excluding never-source dirs).

    Used by :func:`_is_env_misconfig` to distinguish a missing-system-binary
    failure (``pytest`` not installed → genuine env misconfig) from a
    missing-own-package failure (``task_dispatcher`` not on sys.path → fixable
    user-code bug).
    """
    if not symbol or not workspace_path or not os.path.isdir(workspace_path):
        return False
    from harness.impact import _NEVER_SOURCE_DIRS
    sym = symbol.strip().strip("'\"")
    if not sym or "/" in sym or os.sep in sym:
        return False
    try:
        for sub_root, sub_dirs, sub_files in os.walk(workspace_path):
            sub_dirs[:] = [
                d for d in sub_dirs
                if not d.startswith(".") and d not in _NEVER_SOURCE_DIRS
            ]
            if sym in sub_dirs:
                return True
            if f"{sym}.py" in sub_files:
                return True
    except OSError:
        return False
    return False


# pytest's exit code 5 means "no tests collected" — distinct from a genuine
# compile/test failure. Confirm via the literal "no tests ran" line so we
# don't misclassify a config error that happens to produce exit 5.
_NO_TESTS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?m)^=+\s*no tests ran in [\d.]+s\s*=*$"),
    re.compile(r"(?m)^no tests ran in [\d.]+s\s*$"),
)


# pip's resolver emits one of these lines when version pins in the build's
# requirements file (or its transitive deps) can't be satisfied together.
# The error message rarely names BOTH sides of the conflict, so the repair
# LLM is forced to guess — autofix strips the pins entirely instead.
_PIP_RESOLUTION_CONFLICT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?m)^ERROR: ResolutionImpossible\b"),
    re.compile(
        r"(?m)^ERROR: Cannot install .+ because these package versions have "
        r"conflicting dependencies"
    ),
    re.compile(
        r"(?m)^ERROR: pip's dependency resolver does not currently take into "
        r"account all the packages that are installed"
    ),
)


def _is_pip_resolution_conflict(raw_output: str, build_command: str) -> bool:
    """True when the build failed because pip's resolver couldn't satisfy
    the requested version pins together.

    Used by compiler_node to emit a distinct ``DEP_RESOLUTION_CONFLICT``
    diagnostic. Autofix R5 then drops every version specifier from
    ``requirements.txt`` — for greenfield runs where every pin is a
    fresh LLM guess, that resolves the conflict 95% of the time without
    spending a repair iteration.
    """
    if not raw_output:
        return False
    if "pip" not in build_command.lower():
        return False
    tail = raw_output[-4000:]
    return any(p.search(tail) for p in _PIP_RESOLUTION_CONFLICT_PATTERNS)


def _is_no_tests_collected(exit_code: int, raw_output: str, build_command: str) -> bool:
    """True when the build's failure is pytest's exit-5 'no tests collected'
    rather than an actual test/compile failure.

    Detected by matching the literal 'no tests ran' line in the output AND
    requiring the build to actually exercise pytest. Treating this as a
    generic build failure burns repair iterations on a problem the repair
    LLM cannot fix (there's nothing to repair — the test runner found
    nothing). The router uses this to route to test_generation_node or
    HITL with a precise message instead.
    """
    if exit_code != 5 or not raw_output:
        return False
    if "pytest" not in build_command.lower():
        return False
    tail = raw_output[-4000:]
    return any(p.search(tail) for p in _NO_TESTS_PATTERNS)


# Lines matching these patterns are stripped from build output before the
# repair LLM sees them. They're informational warnings the toolchain emits
# regardless of whether the build succeeded, and feeding them to the LLM
# wastes context AND tempts the model into "fixing" warnings that don't
# actually block the build. Real errors (build failures, assertion fails,
# stack traces) are unaffected — those don't match.
_BUILD_OUTPUT_NOISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Python deprecation / future / pending warnings emitted as a single
    # warning summary line OR the multi-line stack-style emission.
    re.compile(r"^.*\b(?:Deprecation|PendingDeprecation|Future|ResourceWarning)Warning\b.*$"),
    # pytest's terminal "Warnings summary" header — followed by per-test
    # warning blocks. Drop the header so the model doesn't infer a warnings
    # section was important.
    re.compile(r"^=+\s*warnings summary\s*=+\s*$", re.IGNORECASE),
    # pip's chatty noise: "[notice] A new release of pip is available…",
    # the root-user warning, build wheel progress bars, and so on.
    re.compile(r"^\s*\[notice\]\s+.*$"),
    re.compile(r"^WARNING: Running pip as the 'root' user.*$"),
    re.compile(r"^WARNING: .*?Skipping.*?already satisfied.*$"),
    # Setuptools / pkg_resources transition warnings that have spammed
    # every Python build for two years and aren't actionable.
    re.compile(r"^.*\bpkg_resources is deprecated\b.*$"),
    re.compile(r"^.*\bSetuptoolsDeprecationWarning\b.*$"),
)


def _strip_build_output_noise(raw_output: str) -> str:
    """Drop deprecation warnings and other non-actionable noise from
    ``raw_output`` before it reaches the repair LLM.

    Real errors are untouched (none of the patterns match). The filter
    runs line-by-line so a stack trace mixed with a deprecation line
    only loses the deprecation line itself, not the surrounding context.
    """
    if not raw_output:
        return raw_output
    kept: list[str] = []
    for line in raw_output.splitlines():
        if any(p.search(line) for p in _BUILD_OUTPUT_NOISE_PATTERNS):
            continue
        kept.append(line)
    # Preserve trailing-newline shape so the slicer's char-budget math
    # downstream isn't off by one for files that ended in \n.
    out = "\n".join(kept)
    if raw_output.endswith("\n"):
        out += "\n"
    return out


def _slice_build_output_for_repair(
    raw_output: str,
    head_chars: int = 1500,
    tail_chars: int = 2000,
) -> str:
    """Return a head+tail slice of ``raw_output`` for the repair LLM.

    Long build logs (C++ template explosions, Java dependency stacks, Cargo
    compilation walls) tend to put the root-cause error near the START and
    cascading downstream errors near the END. A pure tail slice (the old
    behaviour) shows the repair LLM only the cascade, hiding the underlying
    cause. Slicing both ends gives the model both the original failure and
    the final state, separated by an explicit truncation marker so it knows
    chars were dropped.

    Also strips deprecation / pip-notice noise (see
    :func:`_strip_build_output_noise`) so the LLM doesn't waste a repair
    iteration trying to "fix" a DeprecationWarning that doesn't block the
    build.

    For outputs shorter than ``head_chars + tail_chars + 200`` we return
    the whole thing unchanged — the split adds noise without saving space.
    """
    if not raw_output:
        return ""
    cleaned = _strip_build_output_noise(raw_output)
    total = len(cleaned)
    if total <= head_chars + tail_chars + 200:
        return cleaned
    head = cleaned[:head_chars]
    tail = cleaned[-tail_chars:]
    dropped = total - head_chars - tail_chars
    return (
        f"{head}\n"
        f"... [truncated {dropped} chars from the middle of {total}-char build log] ...\n"
        f"{tail}"
    )


_DEP_MANIFEST_CANDIDATES: tuple[str, ...] = (
    "requirements.txt",
    "requirements-dev.txt",
    "requirements/base.txt",
    "requirements/dev.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "Pipfile",
)


def _collect_manifest_snippets_for_repair(
    errors: list[Any], workspace_path: str,
) -> str:
    """Build a Markdown block embedding the current contents of any dep
    manifest the build command references.

    Fires only when at least one diagnostic is shaped as a MISSING_DEP.
    Without the actual file contents in the repair prompt, the LLM resorts
    to ``CREATE_FILE`` (fails when the file exists) or ``REPLACE_BLOCK``
    with a search string it guessed (fails when the guess doesn't match).
    Returns the empty string when no MISSING_DEP diagnostic is present or
    no manifest is found, so callers can string-append unconditionally.
    """
    if not errors or not workspace_path:
        return ""
    missing_deps = [
        e for e in errors
        if str(e.get("error_code", "")) in ("MISSING_DEP", "DEP_RESOLUTION_CONFLICT")
    ]
    if not missing_deps:
        return ""
    if not os.path.isdir(workspace_path):
        return ""

    # Pick the manifests the build command actually references when we can
    # tell — otherwise fall back to whatever exists at workspace root.
    build_cmd = str(missing_deps[0].get("build_command", "") or "").lower()
    preferred: list[str] = []
    if "pyproject" in build_cmd or "pip install -e" in build_cmd:
        preferred = ["pyproject.toml", "setup.py", "setup.cfg"]
    elif "requirements" in build_cmd:
        preferred = ["requirements.txt", "requirements-dev.txt"]
    elif "npm install" in build_cmd or "yarn" in build_cmd or "pnpm" in build_cmd:
        preferred = ["package.json"]
    candidates = preferred or list(_DEP_MANIFEST_CANDIDATES)

    found: list[tuple[str, str]] = []
    for rel in candidates:
        abs_path = os.path.join(workspace_path, rel)
        if not os.path.isfile(abs_path):
            continue
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                # Cap at 8KB per manifest — anything bigger is almost
                # certainly the wrong file.
                content = f.read(8192)
        except OSError:
            continue
        found.append((rel, content))
        if len(found) >= 3:
            break

    if not found:
        return ""
    missing = sorted({
        str(e.get("missing_symbol", "") or "")
        for e in missing_deps if e.get("missing_symbol")
    })
    lines = [
        "\n## Dependency Manifests (current workspace contents)",
        (
            f"The build is missing {missing!r}. Use REPLACE_BLOCK / "
            f"INSERT_AT_BLOCK against the exact bytes below — do NOT "
            f"emit CREATE_FILE for a path that already exists, and do "
            f"NOT invent search strings that aren't in the file."
        ),
    ]
    for rel, content in found:
        lang = "toml" if rel.endswith(".toml") else (
            "json" if rel.endswith(".json") else "text"
        )
        lines.append(f"\n### `{rel}`\n```{lang}\n{content.rstrip()}\n```")
    return "\n".join(lines) + "\n"


def _workspace_has_source_files(workspace_path: str) -> bool:
    """True when the workspace contains at least one source file under a
    non-ignored directory. Used by the router to decide whether 'no tests
    collected' should route to test_generation_node (source exists, needs
    tests) or HITL (truly empty workspace — the LLM failed to scaffold)."""
    from harness.impact import (
        _SOURCE_FILE_EXTENSIONS,
        _NEVER_SOURCE_DIRS,
    )
    if not workspace_path or not os.path.isdir(workspace_path):
        return False
    try:
        for sub_root, sub_dirs, sub_files in os.walk(workspace_path):
            sub_dirs[:] = [
                d for d in sub_dirs
                if not d.startswith(".") and d not in _NEVER_SOURCE_DIRS
                and d not in {"tests", "test", "__tests__"}
            ]
            for fname in sub_files:
                if os.path.splitext(fname)[1].lower() in _SOURCE_FILE_EXTENSIONS:
                    return True
    except OSError:
        return False
    return False


# Pip-installable test / lint tools. When `_is_env_misconfig` flags one
# of these as missing, the LLM repair loop CAN fix it by appending the
# package to the workspace's dependency manifest — no image swap needed.
# Contrast with non-installable symbols (npm, node, cargo, go, rustc,
# docker) where the base image itself is wrong and only a config change
# can unblock the run; those still short-circuit to HITL.
_PIP_INSTALLABLE_SYMBOLS: frozenset[str] = frozenset({
    "pytest", "pytest-asyncio", "pytest-cov", "pytest-mock", "pytest-xdist",
    "ruff", "mypy", "black", "isort", "flake8", "pylint",
    "coverage", "tox", "nox", "poetry",
})


def _repairable_dep_hint(symbol: str, build_command: str) -> str:
    """Repair-friendly diagnostic for a missing pip-installable test / lint
    tool. Reaches the repair LLM via compiler_errors and points it at the
    smallest possible patch (add the dep to requirements.txt or pyproject
    dev extras). Distinct from :func:`_env_misconfig_hint`, which is sent
    to HITL because no in-container patch can fix it.
    """
    return (
        f"Build failed: '{symbol}' is required by the build command but is "
        f"not declared as a dependency. The sandbox runs "
        f"`{build_command.strip()}`, which invokes `{symbol}` after the "
        f"install step — but '{symbol}' isn't in the workspace's "
        f"dependency manifest, so pip never installs it.\n\n"
        f"Fix in ONE place:\n"
        f"  - If the install step is `pip install -r requirements.txt`: "
        f"add `{symbol}` to `requirements.txt`. If the file does not "
        f"exist yet, CREATE it with one dependency per line "
        f"(including `{symbol}`).\n"
        f"  - If the install step is `pip install -e '.[dev]'`: add "
        f"`{symbol}` to `[project.optional-dependencies].dev` in "
        f"`pyproject.toml`. If the section does not exist, CREATE it.\n"
        f"Do not change the build_command or docker_image — the package "
        f"is pip-installable and the current image is correct."
    )


def _env_misconfig_hint(symbol: str, build_command: str) -> str:
    """Build the actionable HITL message for an env-misconfig hit."""
    # Pick the most likely installer for the missing symbol.
    py_symbols = {"pytest", "ruff", "mypy", "black", "poetry", "tox", "nox"}
    if symbol.lower() in py_symbols or "python" in symbol.lower():
        installer = f"pip install {symbol}"
    elif symbol in {"npm", "node", "yarn", "pnpm"}:
        installer = (
            f"use a node-bearing docker_image (e.g. node:20-slim) — "
            f"'{symbol}' is not installable from inside the container"
        )
    elif symbol in {"cargo", "rustc"}:
        installer = (
            f"use a rust-bearing docker_image (e.g. rust:1.79-slim) — "
            f"'{symbol}' is not installable from inside the container"
        )
    elif symbol in {"go"}:
        installer = (
            f"use a go-bearing docker_image (e.g. golang:1.22) — "
            f"'{symbol}' is not installable from inside the container"
        )
    else:
        installer = f"install {symbol} before running the build"
    return (
        f"Build container is missing '{symbol}'. The repair LLM cannot fix this — "
        f"it's a sandbox/dependency setup issue. "
        f"Fix: update build_command to prepend the install step "
        f"(e.g. '{installer} && {build_command.strip()}') "
        f"or switch sandbox.docker_image to one that ships '{symbol}'."
    )


# Bare base images that ship none of the language toolchains. When the
# resolved build_command implies a specific toolchain, swap one of these
# out for a matching toolchain image so the very first build doesn't
# exit 127 with "python3: not found" / "node: not found".
_BARE_IMAGE_DEFAULTS = frozenset({
    "ubuntu:22.04", "ubuntu:latest", "debian:12", "debian:latest",
})


def _apply_toolchain_adaptation(
    build_command: str,
    sandbox_config: Optional[dict[str, Any]],
    allow_network: bool,
    *,
    command_is_adapter_synthesised: bool = False,
) -> tuple[dict[str, Any], bool, bool, bool, bool]:
    """Idempotently adapt sandbox_config/allow_network to match the build
    command's implied toolchain.

    Returns
    -------
    (new_sandbox_config, allow_network, image_was_adapted,
     network_was_adapted, ro_root_was_adapted).

    Calling twice with the same inputs returns *_was_adapted=False on
    the second call — each conditional only fires when its precondition is
    still true (image is a known-bare default, allow_network still False,
    read_only_root unset / still True).

    P1.3: the network auto-enable on a pip/npm install heuristic is gated
    by ``sandbox.auto_enable_network_for_install`` (default ``false``).
    When the heuristic would fire on a user-typed build command but the
    opt-in is off, the function declines to flip ``allow_network`` and
    logs a warning so the operator sees the divergence.

    ``command_is_adapter_synthesised`` (default False) marks the case
    where the install step was introduced by the harness's own late-bind
    detection (``_detect_default_build_command`` fell through to a
    bootstrapping branch), NOT by the operator. The user's opt-in
    governs auto-flips on commands the OPERATOR wrote; it doesn't make
    sense to apply it to commands the harness invented to make a
    greenfield build work at all. When ``True``, network is auto-enabled
    for the install step regardless of the opt-in (the operator can
    still hard-pin ``allow_network=False`` at the workspace config level
    if they really want it).
    """
    cfg = dict(sandbox_config or {})
    image_was_adapted = False
    network_was_adapted = False
    ro_root_was_adapted = False

    cur_image = cfg.get("docker_image", "ubuntu:22.04")
    toolchain_image = _toolchain_image_for(build_command)
    if (
        toolchain_image
        and cur_image in _BARE_IMAGE_DEFAULTS
        and cur_image != toolchain_image
    ):
        cfg["docker_image"] = toolchain_image
        image_was_adapted = True

    new_allow_network = allow_network
    needs_install = _build_command_needs_network(build_command)
    if not allow_network and needs_install:
        # When the harness itself produced the install step, the opt-in
        # doesn't apply — the operator never typed `pip install`, the
        # adapter inserted it to bootstrap a greenfield workspace. The
        # user's network policy still applies at the workspace config
        # level (sandbox config) but the opt-in flag is about user-typed
        # commands, not adapter-synthesised ones.
        if command_is_adapter_synthesised:
            new_allow_network = True
            network_was_adapted = True
        elif cfg.get("auto_enable_network_for_install", False):
            new_allow_network = True
            network_was_adapted = True
        else:
            logger.warning(
                "[sandbox] Build command requires network access for install "
                "but sandbox.auto_enable_network_for_install is false. "
                "Build will run offline and likely fail; either pre-install "
                "deps in the sandbox image, run `harness run --allow-network`, "
                "or set sandbox.auto_enable_network_for_install=true in "
                ".harness_config.json. Build command: %s",
                build_command,
            )

    # Install commands (pip install -e ., npm install -g, cargo install)
    # write to system locations the --read-only root FS makes unreachable.
    # Pip's `--user` fallback also fails because /root sits on the RO root.
    # Auto-flip read_only_root → False so the install can land, unless the
    # user has explicitly set it to True (respect explicit opt-in to hard
    # isolation even if it breaks the build — they'll need a baked image).
    if needs_install and "read_only_root" not in cfg:
        cfg["read_only_root"] = False
        ro_root_was_adapted = True

    return (
        cfg,
        new_allow_network,
        image_was_adapted,
        network_was_adapted,
        ro_root_was_adapted,
    )


async def compiler_node(state: AgentState) -> dict[str, Any]:
    """
    Node 3: The Verifier.

    A purely deterministic local node. Executes the project's build command
    inside an isolated sandbox (Linux namespace jail) via asyncio subprocess.

    Reads `build_command` and `workspace_path` from state.
    Writes `exit_code` and structured `compiler_errors` back to state.

    If exit_code == 0, triggers the memory cleanse utility to compress
    verbose debugging turns before the next graph transition.
    """
    logger.info("[compiler_node] Running build command in sandbox...")

    loop_counter = state.get("loop_counter", {})
    loop_counter = dict(loop_counter)
    loop_counter["compiler"] = loop_counter.get("compiler", 0) + 1

    workspace = state.get("workspace_path", os.getcwd())
    build_cmd = state.get("build_command", "make build")
    allow_network = state.get("allow_network", False)
    sandbox_cfg = dict(state.get("sandbox_config", {}) or {})

    # Late-bound adaptive detection: if the resolved build_cmd is the
    # historical default and the workspace genuinely has no Makefile,
    # re-sniff now that codegen has likely populated the tree. This
    # rescues greenfield runs where workspace detection at cmd_run start
    # had nothing to go on but the spec file.
    adapted_build_cmd: Optional[str] = None
    if build_cmd.strip() == "make build" and not any(
        os.path.exists(os.path.join(workspace, name))
        for name in ("Makefile", "makefile", "GNUmakefile")
    ):
        from harness.cli import _detect_default_build_command
        late = _detect_default_build_command(workspace)
        if late and late != "make build":
            logger.info(
                "[compiler_node] Workspace has no Makefile; adapting build command "
                "from default 'make build' to detected: %s", late,
            )
            adapted_build_cmd = late
            build_cmd = late

    # Late-bound sandbox image / network adaptation. With the pre-flight
    # adaptation in run_graph this is now a safety net — it only fires when
    # the build_command was just adapted above (greenfield rescue), or on
    # resume from a pre-fix checkpoint whose sandbox_config wasn't yet
    # adapted. The helper is idempotent: if the image already matches the
    # toolchain, image_was_adapted is False and no extra log line appears.
    # When the build_command was just adapter-synthesised, tell the
    # toolchain adapter so it can bypass the user-opt-in network gate —
    # the operator never typed this command, the harness invented it.
    prev_image = sandbox_cfg.get("docker_image", "ubuntu:22.04")
    (
        sandbox_cfg,
        allow_network,
        image_was_adapted,
        network_was_adapted,
        ro_root_was_adapted,
    ) = _apply_toolchain_adaptation(
        build_cmd,
        sandbox_cfg,
        allow_network,
        command_is_adapter_synthesised=adapted_build_cmd is not None,
    )
    if image_was_adapted:
        logger.info(
            "[compiler_node] Adapting sandbox docker_image from %r to %r "
            "to match build toolchain implied by command: %s",
            prev_image, sandbox_cfg["docker_image"], build_cmd,
        )
    if network_was_adapted:
        logger.info(
            "[compiler_node] Adapting allow_network from False to True "
            "because build command requires package install: %s",
            build_cmd,
        )
    if ro_root_was_adapted:
        logger.info(
            "[compiler_node] Adapting sandbox.read_only_root from True to False "
            "because build command installs packages (pip/npm/cargo/go) into "
            "system locations the read-only root FS would block: %s",
            build_cmd,
        )

    # Delegate to the sandbox module for actual execution.
    from harness.sandbox import SandboxExecutor

    executor = SandboxExecutor(
        workspace_path=workspace,
        allow_network=allow_network,
        sandbox_config=sandbox_cfg,
    )
    result = await executor.run(build_cmd)

    exit_code: int = result.exit_code
    compiler_errors: list[Any] = [d.to_dict() for d in result.diagnostics]
    raw_log: str = result.raw_output

    logger.info(
        "[compiler_node] Build finished with exit code %d. %d diagnostic(s) extracted.",
        exit_code,
        len(compiler_errors),
    )

    # Detect "sandbox is missing a required runtime" failures and split into
    # two routes:
    #   - Pip-installable test/lint tools (pytest, ruff, mypy, ...): the
    #     repair LLM CAN fix it by amending the workspace's dep manifest.
    #     Emit a MISSING_DEP diagnostic and let normal routing take over so
    #     repair_node gets the diagnostic as context.
    #   - Everything else (npm/node/cargo/go/docker, single-segment local
    #     modules): the image itself is wrong / the LLM cannot help from
    #     inside the sandbox. Short-circuit to HITL as before.
    # pip ResolutionImpossible — emit a distinct diagnostic so autofix R5
    # (`_try_dep_resolution_conflict`) can strip the version pins from
    # requirements.txt instead of forcing the repair LLM to guess which
    # side of the conflict to relax. Without this we burn the entire
    # 3-iteration repair budget on conflicts the LLM doesn't have enough
    # information to resolve (pip's error doesn't name both sides).
    if (
        exit_code != 0
        and not compiler_errors
        and _is_pip_resolution_conflict(raw_log, build_cmd)
    ):
        logger.warning(
            "[compiler_node] pip resolution conflict detected — routing "
            "through repair loop so autofix can strip pins from requirements."
        )
        compiler_errors = [{
            "file": "<sandbox>",
            "line": 0,
            "column": 0,
            "severity": "error",
            "error_code": "DEP_RESOLUTION_CONFLICT",
            "message": (
                "pip's resolver couldn't satisfy the version pins in the "
                "dependency manifest together. The repair-loop fix is to "
                "loosen or drop the version specifiers (>=, ==, ~=, <, !=) "
                "from the manifest so pip can pick a self-consistent set."
            ),
            "semantic_context": f"Build command: {build_cmd}.",
            "missing_symbol": "",
            "build_command": build_cmd,
            "miss_kind": "resolution",
        }]

    env_misconfig_symbol: Optional[str] = None
    env_misconfig_is_repairable: bool = False
    if exit_code != 0 and not compiler_errors:
        env_match = _is_env_misconfig(raw_log, workspace)
        if env_match:
            env_misconfig_symbol, miss_kind = env_match
            # Python ModuleNotFoundError is ALWAYS repairable — any
            # single-segment module name is a pip-installable distribution,
            # and the autofix R4 (`_try_missing_dep`) + the repair LLM can
            # land the requirements.txt edit. Shell `command not found` is
            # only repairable when the symbol is a pip-installable Python
            # tool listed in `_PIP_INSTALLABLE_SYMBOLS`; everything else
            # (npm, cargo, go, docker) needs an operator-side image swap.
            env_misconfig_is_repairable = (
                miss_kind == "python"
                or env_misconfig_symbol.lower() in _PIP_INSTALLABLE_SYMBOLS
            )
            if env_misconfig_is_repairable:
                logger.info(
                    "[compiler_node] Missing '%s' (kind=%s) is pip-installable. "
                    "Routing through repair loop so autofix / LLM can amend the "
                    "dep manifest.",
                    env_misconfig_symbol, miss_kind,
                )
                msg = _repairable_dep_hint(env_misconfig_symbol, build_cmd)
                code = "MISSING_DEP"
            else:
                logger.warning(
                    "[compiler_node] Environment misconfig detected: missing '%s' "
                    "(kind=%s). Short-circuiting repair loop — this needs a config "
                    "fix, not an LLM patch.",
                    env_misconfig_symbol, miss_kind,
                )
                msg = _env_misconfig_hint(env_misconfig_symbol, build_cmd)
                code = "ENV_MISCONFIG"
            compiler_errors = [{
                "file": "<sandbox>",
                "line": 0,
                "column": 0,
                "severity": "error",
                "error_code": code,
                "message": msg,
                "semantic_context": (
                    f"Missing runtime: {env_misconfig_symbol}. "
                    f"Build command: {build_cmd}. "
                    f"docker_image: {sandbox_cfg.get('docker_image', 'ubuntu:22.04')}."
                ),
                # Structured fields the autofix + repair-prompt builders read
                # without re-parsing the human-readable message.
                "missing_symbol": env_misconfig_symbol,
                "build_command": build_cmd,
                "miss_kind": miss_kind,
            }]

    # Build the return dictionary
    node_state: dict[str, Any] = {
        "current_node": "compiler",
        "last_build_output": raw_log,
    }
    # Only set the short-circuit flag for symbols the LLM truly can't fix.
    # Repairable symbols carry their diagnostic into repair_node normally.
    if env_misconfig_symbol and not env_misconfig_is_repairable:
        node_state["env_misconfig"] = True
        node_state["env_misconfig_symbol"] = env_misconfig_symbol

    # Pytest exit-5 (no tests collected) is NOT a build failure — the test
    # runner just had nothing to run. Surface as a distinct condition so the
    # router can fan out to test_generation (when source exists) or HITL
    # (when the workspace is empty), instead of burning the repair budget
    # on a problem the repair LLM cannot fix from inside the loop.
    #
    # Deliberately do NOT populate `compiler_errors` here. If we did, the
    # downstream `route_after_test_generation` would see a non-empty errors
    # list (even after test_generation_node skips or finishes cleanly) and
    # spin into a repair loop trying to "fix" a non-error.
    if exit_code != 0 and not compiler_errors and _is_no_tests_collected(
        exit_code, raw_log, build_cmd,
    ):
        has_source = _workspace_has_source_files(workspace)
        node_state["no_tests_collected"] = True
        node_state["no_tests_has_source"] = has_source
        logger.warning(
            "[compiler_node] pytest exit=5: no tests collected. "
            "Source files present: %s. Router will %s.",
            has_source,
            "route to test_generation" if has_source else "route to HITL",
        )

    return_dict: dict[str, Any] = {
        "exit_code": exit_code,
        "compiler_errors": compiler_errors,
        "loop_counter": loop_counter,
        "node_state": node_state,
    }

    # Persist the adapted command so repair_node / patching_node prompts
    # and subsequent compiler invocations see the updated value.
    if adapted_build_cmd is not None:
        return_dict["build_command"] = adapted_build_cmd
    if image_was_adapted:
        return_dict["sandbox_config"] = sandbox_cfg
    if network_was_adapted:
        return_dict["allow_network"] = allow_network

    # Trigger memory cleanse on successful build
    if exit_code == 0:
        logger.info("[compiler_node] Build succeeded. Applying memory cleanse.")
        cleanse_update = apply_memory_cleanse(state, resolution_kind="compiler_success")
        return_dict.update(cleanse_update)

    return return_dict


async def repair_node(state: AgentState) -> dict[str, Any]:
    """
    Node 4: The Fixer.

    Invoked when the compiler_node returns a non-zero exit code and the
    loop_counter is under the throttle limit (3).

    Uses the configured repair model (repair_primary in .harness_config.json)
    with thinking mode enabled via the gateway to analyze structured compiler
    diagnostics and produce a targeted fix patch. Applies the resulting patches
    to disk immediately.

    Increments the repair loop counter. If the counter hits the threshold,
    the conditional edge will route to human_intervention_node instead.
    """
    logger.info("[repair_node] Analyzing compiler errors and preparing fix...")

    loop_counter = state.get("loop_counter", {})
    loop_counter = dict(loop_counter)
    loop_counter["repair"] = loop_counter.get("repair", 0) + 1
    loop_counter["total_repairs"] = loop_counter.get("total_repairs", 0) + 1

    # Build a concise repair prompt from structured diagnostics. Drop any
    # diagnostic the parsers flagged as warning-severity — DeprecationWarning,
    # PendingDeprecationWarning, ruff style nits, pip notices — none of these
    # block the build, and surfacing them to the repair LLM tempts it to
    # spend an iteration "fixing" something that wasn't broken. The unfiltered
    # list stays in state for observability; only the LLM input is trimmed.
    raw_errors: list[DiagnosticObjectDict] = state.get("compiler_errors", [])
    errors: list[DiagnosticObjectDict] = [
        e for e in raw_errors
        if str(e.get("severity", "error")).lower() != "warning"
    ]
    if len(errors) < len(raw_errors):
        logger.info(
            "[repair_node] Filtered %d warning-severity diagnostic(s); "
            "%d error(s) passed through to repair.",
            len(raw_errors) - len(errors), len(errors),
        )

    # --- Deterministic autofix pass (R1+R2+R3) ---
    # Try to resolve diagnostics with compiler-suggested fixes,
    # missing-import insertion, or known-safe security autofixes BEFORE
    # spending an LLM call. Anything still unhandled falls through to
    # the LLM exactly as before.
    from harness.autofix import apply_autofixes, autofix_system_message
    workspace_path = state.get("workspace_path", os.getcwd())
    unhandled, applied_fixes = await apply_autofixes(list(errors), workspace_path)
    autofix_modified_files = list(state.get("modified_files", []))
    autofix_messages = list(state.get("messages", []))
    if applied_fixes:
        for r in applied_fixes:
            if r.file not in autofix_modified_files:
                autofix_modified_files.append(r.file)
        sys_msg = autofix_system_message(applied_fixes)
        if sys_msg:
            autofix_messages.append({"role": "system", "content": sys_msg})
        logger.info(
            "[repair_node] autofix resolved %d of %d diagnostic(s) without LLM.",
            len(applied_fixes), len(errors),
        )

    # Short-circuit: every diagnostic was resolved deterministically →
    # skip the LLM call entirely. Hand the routing back to compiler_node
    # for re-verification with the modified files in state.
    if applied_fixes and not unhandled:
        return {
            "messages": autofix_messages,
            "modified_files": autofix_modified_files,
            "loop_counter": loop_counter,
            "node_state": {
                "current_node": "repair",
                "repair_context": "all diagnostics resolved by autofix",
                "repair_success": len(applied_fixes),
                "repair_fail": 0,
                "autofix": {
                    "applied": len(applied_fixes),
                    "fix_kinds": sorted({r.fix_kind for r in applied_fixes}),
                },
            },
        }

    # The LLM only sees the unhandled tail.
    errors = unhandled
    error_summary = _format_diagnostics_for_repair(errors)

    gateway = get_gateway()
    if gateway is None:
        logger.error("[repair_node] No gateway configured. Cannot call LLM.")
        return {
            "node_state": {"current_node": "repair", "error": "No gateway configured"},
            "loop_counter": loop_counter,
        }

    # Detect security-finding repair: every diagnostic carries an
    # error_code prefixed by the scanner name (BANDIT:, SEMGREP:, TRIVY:,
    # GITLEAKS:, GITLEAKS-FALLBACK:) because _findings_to_diagnostics in
    # harness/security.py builds them that way. When ALL diagnostics
    # carry such a prefix we know the repair is being driven by the
    # security gate, not a compile failure, and we swap in framing that
    # tells the LLM to fix the vulnerability root cause rather than
    # patch over a build error.
    _SECURITY_PREFIXES = ("BANDIT:", "SEMGREP:", "TRIVY:", "GITLEAKS:", "GITLEAKS-FALLBACK:")
    is_security_repair = bool(errors) and all(
        str(e.get("error_code", "")).upper().startswith(_SECURITY_PREFIXES) for e in errors
    )

    # Detect repair driven by harness-generated test failures. The
    # test_generation_node tags each diagnostic with an error_code starting
    # with "TEST_FAILURE" so we can swap in framing that tells the LLM these
    # are unit-test failures (not compile errors) and that fixing the
    # implementation is preferred over weakening the test assertion.
    is_test_failure_repair = bool(errors) and all(
        str(e.get("error_code", "")).upper().startswith("TEST_FAILURE") for e in errors
    )

    # Include lintgate errors from the previous compiler run
    lintgate_state = state.get("node_state", {}).get("lintgate", {})
    lint_errors_list: list[str] = lintgate_state.get("lint_errors", [])
    if lint_errors_list:
        lint_summary = "\n## Lint Gate Errors\n" + "\n".join(f"  - {e}" for e in lint_errors_list)
        error_summary += lint_summary

    # If the previous patching/repair attempt had any allowlist rejections,
    # surface them so the LLM stops re-proposing the same blocked paths. The
    # repair LLM otherwise has no way to know its patches keep being thrown
    # away by the path filter.
    prior_rejections = state.get("node_state", {}).get("allowlist_rejections") or []
    if prior_rejections:
        prior_allowed = state.get("node_state", {}).get("allowed_paths") or []
        rejected_paths = sorted({r.get("file", "") for r in prior_rejections if r.get("file")})
        rejection_block = (
            "\n## Allowlist Rejections (PREVIOUS attempt)\n"
            "Your last attempt produced patches targeting paths the patcher's "
            "skill allowlist rejected. These patches did NOT land on disk. "
            "Do NOT re-propose the same paths verbatim — relocate the file or "
            "use one of the allowed roots.\n"
            f"Rejected: {rejected_paths}\n"
            f"Allowed roots: {prior_allowed}\n"
        )
        error_summary += rejection_block

    # If no structured diagnostics, include the raw build output so the LLM can see the actual error
    if not errors:
        raw_output = state.get("node_state", {}).get("last_build_output", "")
        if raw_output:
            error_summary += f"\n## Raw Build Output\n```\n{_slice_build_output_for_repair(raw_output)}\n```"

    # When any diagnostic is MISSING_DEP, attach the current contents of the
    # dependency manifest the build command references. Without this the LLM
    # tries CREATE_FILE (which fails — "already exists with different content")
    # then REPLACE_BLOCK with a guessed search string that doesn't match.
    # Autofix R4 (`_try_missing_dep`) handles requirements.txt automatically;
    # this block covers the pyproject / package.json fallback path where
    # autofix deliberately defers to the LLM.
    workspace_for_manifest = state.get("workspace_path", os.getcwd())
    manifest_attachments = _collect_manifest_snippets_for_repair(
        errors, workspace_for_manifest,
    )
    if manifest_attachments:
        error_summary += manifest_attachments

    try:
        from harness.gateway import NodeRole
        from harness.patcher import process_llm_patch_output

        # Use the autofix-augmented messages list so the LLM sees the
        # "we already fixed X" system message and doesn't re-fix.
        messages = list(autofix_messages)
        budget = state.get("budget_remaining_usd", 2.00)

        # --- Cross-Model Speculative Execution ---
        # Repair #1 and #2: use the cheap model (repair_primary)
        # Repair #3 (final attempt): escalate to the heavy reasoning model (repair_fallback)
        # This saves tokens by only using the expensive model when the cheap one has failed twice.
        total_repairs = loop_counter["total_repairs"]
        use_escalation = total_repairs >= 2  # Escalate on 2nd+ failure (3rd attempt is the last)

        if use_escalation:
            escalation_model = gateway.config.repair_fallback or gateway.config.planning_fallback
            if escalation_model:
                logger.warning(
                    "[repair_node] Cheap model failed %d time(s). Escalating to reasoning model: %s",
                    total_repairs - 1,
                    escalation_model,
                )
                # Escalated repair will use NodeRole.REPAIR with thinking mode enabled
            else:
                logger.warning(
                    "[repair_node] Cheap model failed %d time(s), but no escalation model configured. "
                    "Continuing with primary repair model.",
                    total_repairs - 1,
                )
                use_escalation = False
        else:
            escalation_model = None

        # Inject the error summary as a user message for context. The
        # framing sentence differs across three cases:
        #   1. Security gate flagged vulnerabilities → tell the LLM these
        #      are post-build security findings (build passes!) so it
        #      doesn't try to "fix a broken build" by removing tests or
        #      catching exceptions broadly. Emphasise minimum diff and
        #      not weakening other controls.
        #   2. Build failed and the cheap model has already missed twice
        #      → escalate to the reasoning model with reasoning framing.
        #   3. Build failed, first or second attempt → standard framing.
        if is_security_repair:
            repair_prompt = (
                "The deterministic security gate flagged the following vulnerabilities "
                "in code that has already passed the build. Generate precise SEARCH/REPLACE "
                "patches that REMOVE the root cause without regressing existing tests. "
                "Prefer the minimum diff: do not refactor unrelated code, do not weaken "
                "the security control elsewhere, and if a finding requires a dependency "
                "upgrade, write the new version into the manifest rather than vendoring "
                f"a patched copy.\n\n{error_summary}"
            )
        elif is_test_failure_repair:
            repair_prompt = (
                "The harness-generated unit tests just failed when executed in the "
                "sandbox. These are NOT compile errors — the code builds. For each "
                "failure, decide whether the implementation is wrong or the test "
                "expectation is wrong. Default to fixing the implementation when the "
                "behaviour was specified in the requirements; only adjust the test "
                "when the expectation itself contradicts the spec. Do NOT add mocks "
                "to make a test pass — if a test cannot be exercised without external "
                "dependencies, rewrite it to use the test runner's built-in fakes "
                "(monkeypatch / tmp_path / httptest / @TempDir) instead."
                f"\n\n{error_summary}"
            )
        elif use_escalation:
            repair_prompt = (
                f"The build has failed {total_repairs} time(s) despite previous fix attempts. "
                f"The simpler model could not resolve these errors. You are a senior reasoning model. "
                f"Carefully analyze the errors and produce a definitive fix.\n\n{error_summary}"
            )
        else:
            repair_prompt = (
                f"The build failed with the following errors. Generate precise SEARCH/REPLACE "
                f"patches to fix them.\n\n{error_summary}"
            )
        # Append the repair prompt first
        messages.append(MessageDict(role="user", content=repair_prompt))
        # Then append the strict format reminder (same as patching_node)
        _REPAIR_FORMAT_REMINDER = """[CRITICAL FORMAT INSTRUCTION]
You MUST respond using ONLY the patch block syntax below. Do NOT include any explanations,
markdown code fences, or text outside the blocks. Your entire response must be parseable
as one or more patch blocks.

<<<REPLACE_BLOCK>>>
file: path/to/file.ext
search:
<exact lines to find>
replace:
<exact replacement lines>
<<<END_REPLACE_BLOCK>>>

<<<CREATE_FILE>>>
file: path/to/file.ext
content:
<complete file contents>
<<<END_CREATE_FILE>>>

Quality: Write modular, production-ready code with proper error handling, type hints, and docstrings. Handle edge cases.
Generate your fix patches NOW. Only the blocks above. No other text."""
        messages.append({"role": "user", "content": _REPAIR_FORMAT_REMINDER})

        # Use the non-mutating model_override path so concurrent dispatches
        # don't see each other's transient config mutations and exceptions
        # don't leave gateway.config in an inconsistent state.
        if use_escalation and escalation_model:
            response, new_budget = await gateway.dispatch(
                messages=list(messages),
                role=NodeRole.REPAIR,
                budget_remaining_usd=budget,
                model_override=escalation_model,
            )
        else:
            response, new_budget = await gateway.dispatch(
                messages=list(messages),
                role=NodeRole.REPAIR,
                budget_remaining_usd=budget,
            )

        # Update token tracker
        token_tracker = state.get("token_tracker", {})
        token_tracker = gateway.aggregate_tokens(token_tracker, response.usage)

        # Apply the fix patches to disk. Seed the modified-files list with
        # files the autofix pass already touched so they survive the LLM
        # round-trip into state. Same source-root allowlist as patching_node
        # so the repair LLM can't widen the surface area by writing new
        # modules outside the configured layout.
        workspace = state.get("workspace_path", os.getcwd())
        existing_modified = list(autofix_modified_files)
        allowed_paths = _build_patcher_allowlist(workspace)
        patch_results, modified_files = await process_llm_patch_output(
            response.content,
            workspace,
            existing_modified,
            allowed_paths=allowed_paths,
        )

        # Append the LLM response to messages
        messages.append(MessageDict(role="assistant", content=response.content))

        # Report results
        success_count = sum(1 for r in patch_results if r.success)
        fail_count = len(patch_results) - success_count
        # Track allowlist rejections so the *next* repair iteration sees the
        # exact paths and reason and stops re-proposing them. Without this,
        # the LLM has no signal that its patches keep vanishing.
        allowlist_rejections = [
            {"file": r.file, "operation": r.operation, "reason": r.error}
            for r in patch_results
            if not r.success and isinstance(r.error, str)
            and "not in skill allowlist" in r.error
        ]
        status_msg = f"[System]: Repair attempt {loop_counter['total_repairs']}: applied {success_count}/{len(patch_results)} patches."
        if fail_count > 0:
            failed_files = [r.file for r in patch_results if not r.success]
            status_msg += f" Failed: {', '.join(failed_files)}."
        if allowlist_rejections:
            rejected_paths = ", ".join(sorted({r["file"] for r in allowlist_rejections}))
            status_msg += (
                f"\n[Allowlist] Rejected paths outside the configured layout: "
                f"{rejected_paths}. Allowed roots: {allowed_paths}."
            )
        messages.append(MessageDict(role="system", content=status_msg))

        logger.info(
            "[repair_node] Repair #%d complete. tokens_in=%d tokens_out=%d cost=$%.6f budget_left=$%.4f "
            "patches=%d succeed=%d fail=%d",
            loop_counter["total_repairs"],
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.cost_usd,
            new_budget,
            len(patch_results), success_count, fail_count,
        )

        return {
            "messages": messages,
            "modified_files": modified_files,
            "token_tracker": token_tracker,
            "budget_remaining_usd": new_budget,
            "loop_counter": loop_counter,
            "node_state": {
                "current_node": "repair",
                "repair_context": error_summary,
                "repair_success": success_count,
                "repair_fail": fail_count,
                "allowlist_rejections": allowlist_rejections,
                "allowed_paths": allowed_paths,
            },
        }
    except RuntimeError as exc:
        # Distinguish empty-response (P1.5) and budget-too-low (P1.4) from
        # the generic budget-exhausted case so the HITL router can surface
        # a precise message instead of pretending budget was depleted.
        from harness.gateway import EmptyLLMResponseError, BudgetTooLowError
        if isinstance(exc, EmptyLLMResponseError):
            logger.warning("[repair_node] LLM returned empty content: %s", exc)
            return {
                "node_state": {
                    "current_node": "repair",
                    "error": str(exc),
                    "llm_silent": True,
                    "hitl_trigger": "llm_silent",
                },
                "loop_counter": loop_counter,
            }
        if isinstance(exc, BudgetTooLowError):
            logger.warning("[repair_node] Pre-flight budget refusal: %s", exc)
            return {
                "node_state": {
                    "current_node": "repair",
                    "error": str(exc),
                    "budget_exhausted": True,
                    "hitl_trigger": "budget_preflight",
                },
                "loop_counter": loop_counter,
            }
        logger.warning("[repair_node] Gateway refused during repair: %s", exc)
        return {
            "node_state": {"current_node": "repair", "error": str(exc), "budget_exhausted": True},
            "loop_counter": loop_counter,
        }
    except Exception as exc:
        logger.exception("[repair_node] Unexpected error during repair.")
        return {
            "node_state": {"current_node": "repair", "error": str(exc)},
            "loop_counter": loop_counter,
        }


async def human_intervention_node(state: AgentState) -> dict[str, Any]:
    """
    Node 5: The Breakpoint.

    Terminal HITL node invoked when:
        - Budget is exhausted ($2.00 cap breached)
        - Repair loop counter hits the throttle limit (3 iterations)
        - Any other guardrail violation

    Presents an interactive stdin menu to the developer:
        [v] View diffs
        [r] Resume (re-run compiler)
        [e] Inject hint for the repair node
        [m] Pause for manual IDE edits
        [b] Increase budget (+$2.00)
        [q] Abandon and git rollback

    Delegates to the CLI layer's hitl_menu_loop for actual user interaction.
    The menu blocks until the developer makes a choice, then returns an
    updated state dict with routing signals.
    """
    logger.info("[human_intervention_node] Triggering HITL breakpoint...")

    # Determine why we were invoked
    loop_counter = state.get("loop_counter", {})
    budget_remaining = state.get("budget_remaining_usd", 0.0)

    trigger_reason = "unknown"
    if state.get("node_state", {}).get("env_misconfig"):
        sym = state.get("node_state", {}).get("env_misconfig_symbol", "")
        trigger_reason = f"env_misconfig:{sym}" if sym else "env_misconfig"
    elif budget_remaining <= 0.0:
        trigger_reason = "budget_exhausted"
    elif loop_counter.get("total_repairs", 0) >= 3:
        trigger_reason = "repair_loop_limit"
    elif state.get("exit_code", -1) != 0:
        trigger_reason = "persistent_build_failure"

    # Inject trigger reason into state so the menu can display it
    state_dict = dict(state)
    state_dict["node_state"] = dict(state_dict.get("node_state", {}))
    state_dict["node_state"]["current_node"] = "human_intervention"
    state_dict["node_state"]["hitl_trigger"] = trigger_reason
    state_dict["node_state"]["hitl_active"] = True
    state_dict["node_state"]["hitl_awaiting_input"] = True

    # Delegate to the CLI layer's interactive menu loop.
    # This blocks on stdin until the developer makes a choice.
    from harness.cli import hitl_menu_loop

    updated_state = hitl_menu_loop(state_dict)

    # Extract the node_state back — hitl_menu_loop returns a full state dict
    return updated_state


# ---------------------------------------------------------------------------
# 5. Helper Utilities
# ---------------------------------------------------------------------------

def _format_diagnostics_for_repair(errors: list[DiagnosticObjectDict]) -> str:
    """Format structured diagnostics into a concise repair prompt."""
    if not errors:
        return "No structured diagnostics available. Check raw build output."

    lines: list[str] = ["## Compiler Diagnostics\n"]
    for i, err in enumerate(errors, 1):
        lines.append(
            f"**Error {i}:** `{err.get('error_code', 'UNKNOWN')}` "
            f"in `{err.get('file', '?')}:{err.get('line', 0)}:{err.get('column', 0)}` "
            f"[{err.get('severity', 'error')}]"
        )
        lines.append(f"  Message: {err.get('message', 'No message')}")
        context = err.get("semantic_context", "")
        if context:
            lines.append(f"  Context:\n```\n{context}\n```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Conditional Edge: Compiler → Next Node
# ---------------------------------------------------------------------------

def route_after_compiler(state: AgentState) -> Literal["repair_node", "human_intervention_node", "security_scan_node", "test_generation_node"]:
    """
    Conditional edge router executed after compiler_node completes.

    Decision matrix:
        exit_code == 0                     → END (memory cleanse already applied in compiler_node)
        exit_code != 0 AND repairs < 3     → repair_node
        exit_code != 0 AND repairs >= 3    → human_intervention_node
        budget_remaining <= 0              → human_intervention_node
        no_tests_collected AND has_source  → test_generation_node
        no_tests_collected AND empty repo  → human_intervention_node
    """
    exit_code: int = state.get("exit_code", -1)
    loop_counter: dict[str, int] = state.get("loop_counter", {})
    budget_remaining: float = state.get("budget_remaining_usd", 0.0)
    total_repairs: int = loop_counter.get("total_repairs", 0)
    max_iterations: int = 3  # Spec: route to HITL after 3 failed repair attempts

    def _transition(dest: str) -> str:
        try:
            from harness.observability import emit_event
            emit_event("node_transition",
                       from_node="compiler_node", to_node=dest,
                       exit_code=exit_code, total_repairs=total_repairs,
                       budget_remaining_usd=budget_remaining)
        except Exception:  # noqa: BLE001
            pass
        return dest

    # Check budget first — financial guardrail takes priority
    if budget_remaining <= 0.0:
        logger.warning("[router] Budget exhausted ($%.4f remaining). Routing to HITL.", budget_remaining)
        return _transition("human_intervention_node")

    if exit_code == 0:
        logger.info("[router] Build succeeded (exit 0). Routing to security scan.")
        return _transition("security_scan_node")

    # Environment misconfig (missing pytest, npm, etc.) — LLM repair cannot
    # fix this. Skip the budget burn and surface the actionable error.
    if state.get("node_state", {}).get("env_misconfig"):
        symbol = state.get("node_state", {}).get("env_misconfig_symbol", "")
        logger.warning(
            "[router] Sandbox env misconfig (missing '%s'). Skipping repair loop, routing to HITL.",
            symbol,
        )
        return _transition("human_intervention_node")

    # P1.5: empty LLM response detected. Three rounds of an empty repair LLM
    # would burn three compile cycles with no chance of success; short-circuit
    # to HITL with the precise trigger so the operator sees "llm_silent"
    # instead of the generic repair-limit message.
    if state.get("node_state", {}).get("llm_silent"):
        logger.warning("[router] LLM returned empty content. Routing to HITL immediately.")
        return _transition("human_intervention_node")

    # Pytest exit=5 (no tests collected) is not a build failure.
    #   * Source files exist → route to test_generation_node, which will
    #     generate unit tests targeting them. Don't spend a repair on a
    #     non-error.
    #   * No source files either → the prior patching pass produced nothing
    #     usable (e.g. allowlist rejected every patch). Route to HITL with
    #     the precise diagnostic so the operator can fix the layout/config
    #     instead of letting the repair loop spin.
    if state.get("node_state", {}).get("no_tests_collected"):
        if state.get("node_state", {}).get("no_tests_has_source"):
            logger.info(
                "[router] No tests collected but source files exist. "
                "Routing to test_generation_node."
            )
            return _transition("test_generation_node")
        logger.warning(
            "[router] No tests collected and no source files in workspace. "
            "Routing to HITL — the patching pass did not produce code."
        )
        return _transition("human_intervention_node")

    if total_repairs >= max_iterations:
        logger.warning(
            "[router] Repair limit reached (%d/%d). Routing to HITL.",
            total_repairs,
            max_iterations,
        )
        return _transition("human_intervention_node")

    logger.info("[router] Build failed (exit %d). Repair attempt %d/%d.", exit_code, total_repairs + 1, max_iterations)
    return _transition("repair_node")


# ---------------------------------------------------------------------------
# 6c. Generator Nodes for Three-Phase HITL Gates
# ---------------------------------------------------------------------------

async def requirements_discovery_node(state: AgentState) -> dict[str, Any]:
    """
    Exhaustive requirements discovery: calls the planning LLM with a structured
    cross-examination prompt across 8 sectors. Returns grouped questions as JSON.
    If the user has already answered previous questions, evaluates answers for
    remaining critical unknowns and generates follow-ups.
    """
    gateway = get_gateway()
    if gateway is None:
        logger.error("[reqs_disc] No gateway configured.")
        return {"node_state": {"discovery_complete": True, "error": "No gateway"}}

    messages = list(state.get("messages", []))
    question_count = state.get("node_state", {}).get("discovery_question_count", 0)

    # Determine if this is the first pass or a follow-up
    is_followup = question_count > 0

    if is_followup:
        prompt = f"""You are a Lead Systems Auditor. This is a FOLLOW-UP round (#{question_count + 1}).
Review the conversation above where the user answered your previous questions.
Your task:
1. Cross-reference the user's answers against all 8 sectors below.
2. Identify any REMAINING unknowns, gaps, or contradictions.
3. For each critical gap, generate a targeted follow-up question.
4. If ALL sectors are fully resolved across all 8 areas, output: {{"complete": true}}

Output JSON:
{{
  "modules": [
    {{"name": "INPUT VALIDATION", "questions": [{{"id": "Q1.1", "text": "...", "critical": true/false}}]}},
    ...
  ],
  "complete": false,
  "summary": "Brief status of what's resolved vs remaining"
}}
Return ONLY valid JSON. No markdown or explanation."""

    else:
        prompt = """You are a Lead Systems Auditor. Perform EXHAUSTIVE requirements discovery across ALL 8 sectors below.
For each sector, ask every question needed to eliminate unknowns. Be extremely thorough.

## Required Sectors

### 1. INPUT DATA VALIDATION
- Data types, value ranges, required vs optional fields, format constraints, nesting limits, encoding rules.

### 2. PAYLOAD FORMATTING
- JSON schema/XML/binary format, field naming conventions, pagination structure, error envelope format, versioning.

### 3. ERROR HANDLING BEHAVIORS
- HTTP status codes per scenario, retry logic (exponential backoff, max retries), circuit breaker thresholds, graceful degradation, dead-letter queues.

### 4. MULTI-USER EDGE CASES
- Concurrency model (optimistic/pessimistic locking), race conditions, idempotency requirements, distributed transaction boundaries, eventual consistency windows.

### 5. SECURITY CONTROLS
- Authentication method (JWT, OAuth2, API keys), token storage/rotation, RBAC/permission model, CORS policy, rate limiting, input sanitization, CSRF protection.

### 6. STRICT BUSINESS LOGIC RULES
- State machines and transitions, invariants that must hold, validation constraints, business rule precedence, conflict resolution.

### 7. DATA RETENTION BOUNDARIES
- TTLs for cached data, archival policies, GDPR/compliance requirements, audit log retention, backup schedules.

### 8. HIDDEN ASSUMPTIONS
- Platform requirements (OS, architecture), network topology assumptions, timezone/locale handling, third-party service availability, expected load profiles.

Output the EXACT JSON shape below — the key must be literally "modules"
(not "sectors", not "questions", not the section titles above). The harness
parses this shape strictly; any other top-level key yields zero questions
and the operator sees an empty interview screen.

{
  "modules": [
    {"name": "INPUT VALIDATION", "questions": [
      {"id": "Q1.1", "text": "...", "critical": true},
      {"id": "Q1.2", "text": "...", "critical": false}
    ]},
    {"name": "PAYLOAD FORMATTING", "questions": [...]},
    {"name": "ERROR HANDLING BEHAVIORS", "questions": [...]},
    {"name": "MULTI-USER EDGE CASES", "questions": [...]},
    {"name": "SECURITY CONTROLS", "questions": [...]},
    {"name": "STRICT BUSINESS LOGIC RULES", "questions": [...]},
    {"name": "DATA RETENTION BOUNDARIES", "questions": [...]},
    {"name": "HIDDEN ASSUMPTIONS", "questions": [...]}
  ],
  "complete": false,
  "summary": "Brief status of what's covered vs still unknown"
}

Mark critical items with "critical": true. Return ONLY valid JSON. No
markdown, no explanation, no code blocks."""

    messages.append({"role": "user", "content": prompt})

    from harness.gateway import NodeRole

    current_budget = state.get("budget_remaining_usd", 0.0)
    if current_budget <= 0:
        logger.warning("[reqs_disc] Budget exhausted ($%.4f); skipping discovery.", current_budget)
        return {
            "messages": messages,
            "node_state": {"discovery_complete": True, "error": "budget exhausted"},
            "budget_remaining_usd": current_budget,
        }

    try:
        response, budget = await gateway.dispatch(
            messages=list(messages), role=NodeRole.PLANNING,
            budget_remaining_usd=current_budget,
        )

        from harness.trust import validate_discovery_json
        discovery_data, trust_errors = validate_discovery_json(response.content)
        if trust_errors:
            logger.error("[reqs_disc] Discovery response failed trust validation: %s", trust_errors)
            return {
                "messages": messages,
                "node_state": {"discovery_complete": True, "error": f"trust validation: {trust_errors}"},
                "budget_remaining_usd": budget,
            }

        complete = discovery_data.get("complete", False)
        modules = discovery_data.get("modules", [])
        total_questions = sum(len(m.get("questions", [])) for m in modules)
        critical_count = sum(1 for m in modules for q in m.get("questions", []) if q.get("critical"))

        # Catch schema drift: a discovery response that parsed but had no
        # modules + complete=False means the LLM used a different top-level
        # key (e.g. "sectors", "components"). Surface it loudly so the
        # operator doesn't see an empty interview screen and silently
        # answer "DONE" to no questions — which is the exact symptom that
        # produced the empty Round 1 in session c371c744-….
        if not modules and not complete:
            top_keys = sorted(discovery_data.keys()) if isinstance(discovery_data, dict) else []
            logger.warning(
                "[reqs_disc] LLM response parsed cleanly but has ZERO modules "
                "and complete=False — likely the model used a non-canonical "
                "top-level key (expected 'modules', saw keys=%s). The "
                "interview will display no questions. Re-issuing as a "
                "follow-up round will retry with the canonical schema "
                "embedded in the prompt.",
                top_keys,
            )

        messages.append({"role": "assistant", "content": json.dumps(discovery_data)})

        logger.info("[reqs_disc] Round %d: %d questions across %d modules (%d critical). Complete=%s budget=$%.4f",
                     question_count + 1, total_questions, len(modules), critical_count, complete, budget)

        return {
            "messages": messages,
            "discovery_questions": discovery_data,
            "current_gate": "REQUIREMENTS",
            "budget_remaining_usd": budget,
            "node_state": {
                "current_node": "requirements_discovery",
                "discovery_complete": complete,
                "discovery_question_count": question_count + 1,
                "discovery_critical_remaining": critical_count if not complete else 0,
            },
        }
    except Exception as exc:
        logger.exception("[reqs_disc] Discovery failed: %s", exc)
        return {
            "messages": messages,
            "node_state": {"discovery_complete": True, "error": str(exc)},
            "budget_remaining_usd": current_budget,
        }


async def architecture_discovery_node(state: AgentState) -> dict[str, Any]:
    """
    Exhaustive architecture discovery: queries for deep technical variables
    across 8 sectors. Returns grouped questions as JSON, same format as requirements.
    """
    gateway = get_gateway()
    if gateway is None:
        logger.error("[arch_disc] No gateway configured.")
        return {"node_state": {"discovery_complete": True, "error": "No gateway"}}

    messages = list(state.get("messages", []))
    question_count = state.get("node_state", {}).get("discovery_question_count", 0)
    is_followup = question_count > 0

    if is_followup:
        prompt = f"""You are a Principal Infrastructure Architect. FOLLOW-UP round #{question_count + 1}.
Review the conversation above. Cross-reference answers. Find remaining gaps.
If all 8 architectural sectors are fully resolved, output {{"complete": true}} and nothing else.

Otherwise, output the EXACT JSON shape below — top-level key MUST be
literally "modules". The harness parses this shape strictly; any other
top-level key yields zero questions and the operator sees an empty
interview screen.

{{
  "modules": [
    {{"name": "STORAGE TOPOLOGY", "questions": [
      {{"id": "A1.1", "text": "...", "critical": true}}
    ]}}
  ],
  "complete": false,
  "summary": "Brief status of what's resolved vs remaining"
}}

Return ONLY valid JSON. No markdown, no explanation, no code fences."""

    else:
        prompt = """You are a Principal Infrastructure Architect. Perform EXHAUSTIVE architecture discovery across ALL 8 sectors.

## Required Sectors

### 1. DATABASE SCHEMA DEFINITIONS
- Tables, columns, primary keys, foreign keys, unique constraints, indexes (B-tree/hash/GIN), partitioning strategy, replication (master-slave/multi-master), connection pooling.

### 2. MICROSERVICE COMMUNICATION
- Protocol per service boundary (REST, WebSocket, gRPC, GraphQL, message queue), serialization format, service discovery, load balancing, circuit breaking, retry/backoff.

### 3. EXTERNAL API RATE-LIMITING
- Provider rate limits per API, token bucket/sliding window algorithm, throttling tiers, quota management, retry-after handling.

### 4. CONTAINER VOLUME STORAGE
- Named volumes vs bind mounts, persistent storage paths per service, backup strategy, tmpfs requirements for ephemeral data, NFS/EFS for shared state.

### 5. ENVIRONMENTAL SECRECY
- Secrets manager (HashiCorp Vault, AWS Secrets Manager, Doppler), env var injection, .env file handling, CI/CD secret masking, rotation policy.

### 6. SCALING PARAMETERS
- Horizontal vs vertical scaling, auto-scaling triggers (CPU > 70%, memory, request queue depth), min/max replicas, cold start mitigation.

### 7. OBSERVABILITY
- Structured logging format (JSON), log aggregation, metrics (Prometheus/Datadog), distributed tracing (OpenTelemetry), alerting thresholds, health check endpoints.

### 8. CI/CD PIPELINE HOOKS
- Build triggers (push, PR, tag), deployment gates (approval, test pass), rollback strategy, canary/blue-green deployment, environment promotion path.

Output the EXACT JSON shape below — the top-level key MUST be literally
"modules" (not "sectors", not "components"). Any other key yields zero
questions and the operator sees an empty interview screen.

{
  "modules": [
    {"name": "STORAGE TOPOLOGY", "questions": [
      {"id": "A1.1", "text": "...", "critical": true}
    ]},
    ... one entry per architectural sector above ...
  ],
  "complete": false,
  "summary": "Brief status of what's resolved vs remaining"
}

Mark critical items with "critical": true. Return ONLY valid JSON. No
markdown, no explanation, no code blocks."""
    messages.append({"role": "user", "content": prompt})

    from harness.gateway import NodeRole

    current_budget = state.get("budget_remaining_usd", 0.0)
    if current_budget <= 0:
        logger.warning("[arch_disc] Budget exhausted ($%.4f); skipping discovery.", current_budget)
        return {
            "messages": messages,
            "node_state": {"discovery_complete": True, "error": "budget exhausted"},
            "budget_remaining_usd": current_budget,
        }

    try:
        response, budget = await gateway.dispatch(
            messages=list(messages), role=NodeRole.PLANNING,
            budget_remaining_usd=current_budget,
        )

        from harness.trust import validate_discovery_json
        discovery_data, trust_errors = validate_discovery_json(response.content)
        if trust_errors:
            logger.error("[arch_disc] Discovery response failed trust validation: %s", trust_errors)
            return {
                "messages": messages,
                "node_state": {"discovery_complete": True, "error": f"trust validation: {trust_errors}"},
                "budget_remaining_usd": budget,
            }

        complete = discovery_data.get("complete", False)
        modules = discovery_data.get("modules", [])
        total_q = sum(len(m.get("questions", [])) for m in modules)
        critical_count = sum(1 for m in modules for q in m.get("questions", []) if q.get("critical"))

        messages.append({"role": "assistant", "content": json.dumps(discovery_data)})

        logger.info("[arch_disc] Round %d: %d questions (%d critical). Complete=%s budget=$%.4f",
                     question_count + 1, total_q, critical_count, complete, budget)

        return {
            "messages": messages,
            "discovery_questions": discovery_data,
            "current_gate": "ARCHITECTURE",
            "budget_remaining_usd": budget,
            "node_state": {
                "current_node": "architecture_discovery",
                "discovery_complete": complete,
                "discovery_question_count": question_count + 1,
                "discovery_critical_remaining": critical_count if not complete else 0,
            },
        }
    except Exception as exc:
        logger.exception("[arch_disc] Discovery failed: %s", exc)
        return {
            "messages": messages,
            "node_state": {"discovery_complete": True, "error": str(exc)},
            "budget_remaining_usd": current_budget,
        }


async def deployment_discovery_node(state: AgentState) -> dict[str, Any]:
    """
    Exhaustive deployment infrastructure discovery: calls the planning LLM
    to cross-examine the user across 4 deployment-specific sectors.
    Returns grouped questions as JSON. Follow-ups on subsequent passes.
    """
    gateway = get_gateway()
    if gateway is None:
        logger.error("[deploy_disc] No gateway configured.")
        return {"node_state": {"discovery_complete": True, "error": "No gateway"}}

    messages = list(state.get("messages", []))
    question_count = state.get("node_state", {}).get("discovery_question_count", 0)
    is_followup = question_count > 0

    if is_followup:
        prompt = f"""You are a Principal DevSecOps Engineer. FOLLOW-UP round #{question_count + 1}.
Review the conversation above. Cross-reference deployment answers. Find remaining gaps.
If all 4 deployment sectors are fully resolved, output {{"complete": true}} and nothing else.

Otherwise, output the EXACT JSON shape below — top-level key MUST be
literally "modules". Any other key yields zero questions on the
interview screen.

{{
  "modules": [
    {{"name": "NETWORK TOPOLOGY", "questions": [
      {{"id": "D1.1", "text": "...", "critical": true}}
    ]}}
  ],
  "complete": false,
  "summary": "Brief status of what's resolved vs remaining"
}}

Return ONLY valid JSON. No markdown, no explanation, no code fences."""

    else:
        prompt = """You are a Principal DevSecOps Systems Engineer and Lead SRE. Perform EXHAUSTIVE deployment infrastructure discovery across ALL 4 sectors below.

## Required Sectors

### 1. NETWORK TOPOLOGY
- External routing ports (HTTP 80, HTTPS 443, custom ports), reverse proxy paths (Caddy/Nginx rules), internal Docker network bridge mappings, DNS configuration (host resolution, custom domains), port collision parameters on host.

### 2. DATA & STORAGE PERSISTENCE
- Absolute path volume mounting points, shared database cluster parameters, write permissions (UID/GID mapping), backup paths, tmpfs vs bind mount decisions, NFS/EFS for multi-host setups.

### 3. SECRETS & IDENTITY MANAGEMENT
- Vault integrations (HashiCorp Vault, Doppler, AWS Secrets Manager), decryption methods, Keycloak realm secrets, database passwords, runtime environment variable configurations, .env file handling, CI/CD secret masking.

### 4. PARTIAL INFRASTRUCTURE SYNC
- Verify any pre-existing containers running on host, existing databases or shared network clusters, active ports already bound, avoid destruction or duplication of production resources, sidecar dependencies, legacy service compatibility.

Output the EXACT JSON shape below — top-level key MUST be literally
"modules". Any other key yields zero questions on the interview screen.

{
  "modules": [
    {"name": "RUNTIME PLATFORM", "questions": [
      {"id": "D1.1", "text": "...", "critical": true}
    ]},
    ... one entry per deployment sector above ...
  ],
  "complete": false,
  "summary": "Brief status of what's resolved vs remaining"
}

Mark critical items with "critical": true.
Return ONLY valid JSON. No markdown, no explanation, no code blocks."""

    messages.append({"role": "user", "content": prompt})

    from harness.gateway import NodeRole

    current_budget = state.get("budget_remaining_usd", 0.0)
    if current_budget <= 0:
        logger.warning("[deploy_disc] Budget exhausted ($%.4f); skipping discovery.", current_budget)
        return {
            "messages": messages,
            "node_state": {"discovery_complete": True, "error": "budget exhausted"},
            "budget_remaining_usd": current_budget,
        }

    try:
        response, budget = await gateway.dispatch(
            messages=list(messages), role=NodeRole.PLANNING,
            budget_remaining_usd=current_budget,
        )

        from harness.trust import validate_discovery_json
        discovery_data, trust_errors = validate_discovery_json(response.content)
        if trust_errors:
            logger.error("[deploy_disc] Discovery response failed trust validation: %s", trust_errors)
            return {
                "messages": messages,
                "node_state": {"discovery_complete": True, "error": f"trust validation: {trust_errors}"},
                "budget_remaining_usd": budget,
            }

        complete = discovery_data.get("complete", False)
        modules = discovery_data.get("modules", [])
        total_q = sum(len(m.get("questions", [])) for m in modules)
        critical_count = sum(1 for m in modules for q in m.get("questions", []) if q.get("critical"))

        messages.append({"role": "assistant", "content": json.dumps(discovery_data)})

        logger.info("[deploy_disc] Round %d: %d questions (%d critical). Complete=%s budget=$%.4f",
                     question_count + 1, total_q, critical_count, complete, budget)

        return {
            "messages": messages,
            "discovery_questions": discovery_data,
            "current_gate": "DEPLOYMENT",
            "budget_remaining_usd": budget,
            "node_state": {
                "current_node": "deployment_discovery",
                "discovery_complete": complete,
                "discovery_question_count": question_count + 1,
                "discovery_critical_remaining": critical_count if not complete else 0,
            },
        }
    except Exception as exc:
        logger.exception("[deploy_disc] Discovery failed: %s", exc)
        return {
            "messages": messages,
            "node_state": {"discovery_complete": True, "error": str(exc)},
            "budget_remaining_usd": current_budget,
        }


async def write_spec_node(state: AgentState) -> dict[str, Any]:
    """
    Serializes the full discovery transcript into SPEC_REQUIREMENTS.md,
    SPEC_ARCHITECTURE.md, or DEPLOYMENT_BLUEPRINT.md based on current_gate.
    Compiles all Q&A from the conversation history into a comprehensive Markdown document.
    """
    gate = state.get("current_gate", "REQUIREMENTS")
    workspace = state.get("workspace_path", os.getcwd())
    output_dir = os.path.join(workspace, "docs")

    if gate == "REQUIREMENTS":
        path_key = "spec_requirements_path"
    elif gate == "ARCHITECTURE":
        path_key = "spec_architecture_path"
    else:
        path_key = "deployment_blueprint_path"

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        # docs/ exists but is not a directory, or perms blocked creation.
        logger.error("[write_spec] Cannot create %s: %s", output_dir, exc)
        return {
            path_key: "",
            "current_gate": gate,
            "node_state": {
                "current_node": "write_spec",
                "spec_written": False,
                "spec_write_error": f"makedirs failed: {exc}",
            },
        }

    messages = state.get("messages", [])

    # Build the specification from the full conversation
    sections: list[str] = []
    sections.append(f"# {gate.capitalize()} Specification\n")
    sections.append("*Auto-generated from exhaustive discovery process.*\n")

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user" and ("discovery" in content[:50].lower() or "You are a" in content[:100]):
            continue  # Skip the discovery prompts themselves

        if role == "assistant" and content.strip().startswith("{"):
            try:
                data = json.loads(content)
                if "modules" in data:
                    sections.append("\n## Discovery Questions\n")
                    for mod in data.get("modules", []):
                        sections.append(f"### {mod.get('name', 'Module')}\n")
                        for q in mod.get("questions", []):
                            marker = " **[CRITICAL]**" if q.get("critical") else ""
                            sections.append(f"- {q.get('id', '?')}:{marker} {q.get('text', '')}")
                    sections.append("")
            except json.JSONDecodeError:
                pass
        elif role in ("user", "assistant"):
            sections.append(f"### {role.upper()}\n{content[:3000]}\n")

    spec_content = "\n".join(sections)

    if gate == "REQUIREMENTS":
        spec_path = os.path.join(output_dir, "SPEC_REQUIREMENTS.md")
    elif gate == "ARCHITECTURE":
        spec_path = os.path.join(output_dir, "SPEC_ARCHITECTURE.md")
    else:
        spec_path = os.path.join(output_dir, "DEPLOYMENT_BLUEPRINT.md")

    try:
        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(spec_content)
        logger.info("[write_spec] %s written (%d chars).", spec_path, len(spec_content))
    except OSError as exc:
        # Don't silently claim success — the gatekeeper that follows will try
        # to read this path. Propagate the failure so routing can react.
        logger.error("[write_spec] Failed to write %s: %s", spec_path, exc)
        return {
            path_key: "",
            "current_gate": gate,
            "node_state": {
                "current_node": "write_spec",
                "spec_written": False,
                "spec_write_error": str(exc),
            },
        }

    return {
        path_key: spec_path,
        "current_gate": gate,
        "node_state": {"current_node": "write_spec", "spec_written": True},
    }


# ---------------------------------------------------------------------------
# Reviewer LLM nodes (DOC_REVIEWER + CODE_REVIEWER)
# ---------------------------------------------------------------------------

_SPEC_REVIEW_SYSTEM_PROMPT = """You are an independent reviewer of a software requirements/architecture specification. \
A different LLM drafted the spec; your job is to critique it adversarially.

Return STRICT JSON with this exact top-level shape and NOTHING else (no prose, no markdown fences):
{
  "completeness": ["string description of what is missing"],
  "contradictions": ["..."],
  "ambiguity": ["..."],
  "missing_edge_cases": ["..."],
  "security_gaps": ["..."],
  "testability": ["..."],
  "followup_questions": [
    {"id": "R1", "text": "question for the human author", "critical": true}
  ]
}

Each array may be empty if no issues are found in that category. \
Follow-up questions should be precise and answerable by the human in one or two sentences. \
Mark a question critical=true only if its answer would change the architecture or invalidate the spec."""


_SPEC_REVISE_INSTRUCTION_TEMPLATE = """The original specification draft is below, followed by an independent reviewer's critique JSON. \
Produce a fully revised specification document that addresses every actionable item in the critique. \
Output ONLY the revised Markdown — no preamble, no postscript, no code fences.

## Original Spec ({gate})
{original_spec}

## Reviewer Critique JSON
{critique_json}
"""


def _review_followups_to_discovery_shape(critique: dict[str, Any], gate: str) -> list[dict[str, Any]]:
    """Convert reviewer follow-up questions into the modules-and-questions shape
    that the existing discovery_interview_loop renders natively. All follow-ups
    are marked critical=False so the user can finalize with DONE — the original
    discovery phase already enforced critical-unknowns; this second pass is for
    refinement, not blocking."""
    raw = critique.get("followup_questions", []) or []
    if not isinstance(raw, list):
        return []

    questions: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        qid = str(item.get("id", "")).strip() or f"R{idx + 1}"
        questions.append({"id": qid, "text": text, "critical": False})

    if not questions:
        return []

    module_name = f"Reviewer Follow-ups ({gate.capitalize()})"
    return [{"name": module_name, "questions": questions}]


async def review_and_revise_spec(
    spec_path: str,
    gate: str,
    *,
    gateway: Any,
    budget_remaining_usd: float,
    user_goal: str,
) -> dict[str, Any]:
    """Run the independent doc-reviewer critique + revise pass on a spec
    file. Writes ``SPEC_{REQUIREMENTS,ARCHITECTURE}_REVIEW.md`` alongside
    the spec and overwrites ``spec_path`` with the revised version.

    Used by:
      - ``spec_review_node`` (graph path, after discovery).
      - ``harness.cli.cmd_run`` (pre-flight path, after
        ``synthesize_requirements``) — the reviewer fires whenever
        ``doc_reviewer_primary`` is configured, independent of
        ``--discover``.

    Returns a dict with: ``review_path``, ``critique`` (parsed dict),
    ``new_budget_usd``, ``token_usage_list`` (list of provider Usage
    objects for the caller to aggregate), ``ok`` (bool). On any
    non-fatal failure ``ok=False`` and the caller falls through.
    """
    from harness.gateway import NodeRole

    result: dict[str, Any] = {
        "ok": False,
        "review_path": None,
        "critique": None,
        "new_budget_usd": budget_remaining_usd,
        "token_usage_list": [],
    }

    if gate == "REQUIREMENTS":
        review_filename = "SPEC_REQUIREMENTS_REVIEW.md"
    elif gate == "ARCHITECTURE":
        review_filename = "SPEC_ARCHITECTURE_REVIEW.md"
    else:
        return result

    try:
        with open(spec_path, "r", encoding="utf-8") as f:
            original_spec = f.read()
    except OSError as exc:
        logger.warning("[spec_review] Cannot read %s: %s", spec_path, exc)
        return result

    critique_user_prompt = (
        f"## User Goal\n{user_goal}\n\n"
        f"## Specification Under Review ({gate})\n{original_spec}\n\n"
        "Produce the JSON critique now."
    )
    critique_messages = [
        {"role": "system", "content": _SPEC_REVIEW_SYSTEM_PROMPT},
        {"role": "user", "content": critique_user_prompt},
    ]

    try:
        critique_response, new_budget = await gateway.dispatch(
            messages=critique_messages,
            role=NodeRole.DOC_REVIEWER,
            budget_remaining_usd=budget_remaining_usd,
        )
    except Exception as exc:
        logger.warning("[spec_review] Reviewer dispatch failed: %s — passing through.", exc)
        return result
    result["new_budget_usd"] = new_budget
    result["token_usage_list"].append(critique_response.usage)

    try:
        from harness.trust import _strip_code_fences  # type: ignore
        critique_text = _strip_code_fences(critique_response.content)
    except Exception:
        critique_text = critique_response.content.strip()

    try:
        critique = json.loads(critique_text)
        if not isinstance(critique, dict):
            raise ValueError("critique JSON must be an object")
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("[spec_review] Critique was not valid JSON (%s) — passing through.", exc)
        return result
    result["critique"] = critique

    review_path = os.path.join(os.path.dirname(spec_path), review_filename)
    try:
        with open(review_path, "w", encoding="utf-8") as f:
            f.write(f"# {gate.capitalize()} Spec Review\n\n")
            f.write("*Generated by the independent doc-reviewer LLM.*\n\n")
            f.write("```json\n")
            f.write(json.dumps(critique, indent=2))
            f.write("\n```\n")
        result["review_path"] = review_path
    except OSError as exc:
        logger.warning("[spec_review] Failed to write %s: %s", review_path, exc)

    revise_prompt = _SPEC_REVISE_INSTRUCTION_TEMPLATE.format(
        gate=gate,
        original_spec=original_spec,
        critique_json=json.dumps(critique, indent=2),
    )
    revise_messages = [
        {"role": "system", "content": "You are a senior specification author. Output clean Markdown only."},
        {"role": "user", "content": revise_prompt},
    ]
    try:
        revised_response, new_budget = await gateway.dispatch(
            messages=revise_messages,
            role=NodeRole.PLANNING,
            budget_remaining_usd=new_budget,
        )
        result["new_budget_usd"] = new_budget
        result["token_usage_list"].append(revised_response.usage)
    except Exception as exc:
        logger.warning("[spec_review] Revise dispatch failed: %s — leaving original spec.", exc)
        revised_response = None

    if revised_response is not None:
        revised_text = revised_response.content.strip()
        if revised_text:
            try:
                with open(spec_path, "w", encoding="utf-8") as f:
                    f.write(revised_text)
                logger.info("[spec_review] Revised spec written to %s.", spec_path)
            except OSError as exc:
                logger.warning("[spec_review] Failed to overwrite %s: %s", spec_path, exc)

    result["ok"] = True
    return result


async def spec_review_node(state: AgentState) -> dict[str, Any]:
    """
    Independent LLM critiques the freshly written spec, then the primary
    planning model revises it. Only fires for REQUIREMENTS and ARCHITECTURE
    gates; DEPLOYMENT short-circuits.

    Activation is purely by configuration: if `doc_reviewer_primary` is empty
    in .harness_config.json, this node is a no-op and the graph follows the
    existing path (write_spec_node → human_gatekeeper_node).
    """
    gate = state.get("current_gate", "REQUIREMENTS")
    loop_counter = dict(state.get("loop_counter", {}))
    counter = loop_counter.get("review_spec", 0)

    gateway = get_gateway()
    gateway_config = get_gateway_config()

    doc_reviewer_primary = ""
    max_cycles = 1
    if gateway_config is not None:
        doc_reviewer_primary = getattr(gateway_config, "doc_reviewer_primary", "") or ""
        max_cycles = int(getattr(gateway_config, "max_doc_review_cycles", 1) or 0)

    # Determine which spec path to operate on.
    if gate == "REQUIREMENTS":
        path_key = "spec_requirements_path"
    elif gate == "ARCHITECTURE":
        path_key = "spec_architecture_path"
    else:
        logger.info("[spec_review] gate=%s — out of reviewer scope, passing through.", gate)
        return {"node_state": {"current_node": "spec_review", "skipped": True}}

    spec_path = state.get(path_key, "")
    budget = state.get("budget_remaining_usd", 0.0)

    if not doc_reviewer_primary:
        logger.info("[spec_review] doc_reviewer_primary not configured — skipping.")
        return {"node_state": {"current_node": "spec_review", "skipped": True}}
    if gateway is None:
        logger.info("[spec_review] No gateway available — skipping.")
        return {"node_state": {"current_node": "spec_review", "skipped": True}}
    if counter >= max_cycles:
        logger.info("[spec_review] cycle cap reached (%d/%d) — passing through.", counter, max_cycles)
        return {"node_state": {"current_node": "spec_review", "skipped": True}}
    if budget < 0.10:
        logger.info("[spec_review] budget too low ($%.4f) — skipping.", budget)
        return {"node_state": {"current_node": "spec_review", "skipped": True}}
    if not spec_path or not os.path.isfile(spec_path):
        logger.info("[spec_review] spec file missing (%s) — skipping.", spec_path)
        return {"node_state": {"current_node": "spec_review", "skipped": True}}

    # Short user-goal summary — keeps the reviewer prompt cheap and prevents
    # it from re-litigating the discovery process.
    messages = state.get("messages", [])
    user_goal = ""
    if len(messages) >= 2:
        user_goal = str(messages[1].get("content", ""))[:1000]

    # Run the shared critique + revise pass.
    review_result = await review_and_revise_spec(
        spec_path,
        gate,
        gateway=gateway,
        budget_remaining_usd=budget,
        user_goal=user_goal,
    )
    if not review_result["ok"]:
        # Helper logged the reason; pass through.
        return {"node_state": {"current_node": "spec_review", "skipped": True}}

    new_budget = review_result["new_budget_usd"]
    review_path = review_result["review_path"]
    critique = review_result["critique"] or {}

    # Aggregate token tracker across whichever dispatches actually fired.
    token_tracker = state.get("token_tracker", {})
    for usage in review_result["token_usage_list"]:
        token_tracker = gateway.aggregate_tokens(token_tracker, usage)

    loop_counter["review_spec"] = counter + 1

    # Shape follow-up questions for the discovery interview loop. The loop
    # reads state["discovery_questions"], so populate that. We also reset the
    # discovery counters so the loop renders these as a fresh round.
    followups = _review_followups_to_discovery_shape(critique, gate)
    discovery_payload: dict[str, Any] = {
        "modules": followups,
        "complete": False,
        "summary": (
            f"Reviewer raised {sum(len(m['questions']) for m in followups)} "
            "follow-up question(s)."
        ) if followups else "Reviewer found no critical issues.",
    }
    delta: dict[str, Any] = {
        "messages": list(messages),
        "token_tracker": token_tracker,
        "budget_remaining_usd": new_budget,
        "loop_counter": loop_counter,
        "reviewer_followups": followups,
        "current_gate": gate,
        "node_state": {
            "current_node": "spec_review",
            "skipped": False,
            "review_path": review_path,
            "followup_count": sum(len(m["questions"]) for m in followups),
            # critical=False for every reviewer question (see helper) so the
            # user can DONE through after the second pass.
            "discovery_critical_remaining": 0,
            "discovery_complete": False,
            "discovery_question_count": 0,
        },
    }
    if gate == "REQUIREMENTS":
        delta["reviewer_comments_requirements"] = json.dumps(critique)
    if followups:
        delta["discovery_questions"] = discovery_payload
    return delta


_CODE_REVIEW_SYSTEM_PROMPT = """You are an independent reviewer of code generated by another LLM. \
Critique it adversarially for correctness, security, performance, idiomatic style, and missing tests.

Return STRICT JSON with this exact shape and NOTHING else (no prose, no markdown fences):
{
  "findings": [
    {
      "file": "path/to/file.ext",
      "line": 42,
      "severity": "high|medium|low",
      "category": "correctness|security|performance|idiomatic|missing_tests",
      "suggestion": "concrete description of the change to make"
    }
  ]
}

`findings` may be empty if the code is clean. Be specific: vague suggestions like "improve error handling" are useless. \
Each finding must name a file and (where applicable) a line number."""


async def code_review_node(state: AgentState) -> dict[str, Any]:
    """
    Independent LLM critiques freshly compiled code, then the patcher model
    incorporates the feedback. Activation is purely by configuration: empty
    code_reviewer_primary == no-op, falling through to security_scan_node.
    """
    loop_counter = dict(state.get("loop_counter", {}))
    counter = loop_counter.get("review_code", 0)
    modified_files = list(state.get("modified_files", []))
    workspace = state.get("workspace_path", os.getcwd())
    budget = state.get("budget_remaining_usd", 0.0)

    gateway = get_gateway()
    gateway_config = get_gateway_config()

    code_reviewer_primary = ""
    max_cycles = 1
    if gateway_config is not None:
        code_reviewer_primary = getattr(gateway_config, "code_reviewer_primary", "") or ""
        max_cycles = int(getattr(gateway_config, "max_code_review_cycles", 1) or 0)

    if not code_reviewer_primary:
        logger.info("[code_review] code_reviewer_primary not configured — skipping.")
        return {"node_state": {"current_node": "code_review", "skipped": True, "repatched": False}}
    if gateway is None:
        return {"node_state": {"current_node": "code_review", "skipped": True, "repatched": False}}
    if counter >= max_cycles:
        logger.info("[code_review] cycle cap reached (%d/%d) — passing through.", counter, max_cycles)
        return {"node_state": {"current_node": "code_review", "skipped": True, "repatched": False}}
    if budget < 0.10:
        logger.info("[code_review] budget too low ($%.4f) — skipping.", budget)
        return {"node_state": {"current_node": "code_review", "skipped": True, "repatched": False}}
    if not modified_files:
        logger.info("[code_review] no modified_files — skipping.")
        return {"node_state": {"current_node": "code_review", "skipped": True, "repatched": False}}

    # Snapshot up to 20 files, 2000 lines each, to bound token cost.
    snapshot_chunks: list[str] = []
    file_cap = 20
    line_cap = 2000
    for path in modified_files[:file_cap]:
        abs_path = path if os.path.isabs(path) else os.path.join(workspace, path)
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()[:line_cap]
        except OSError as exc:
            snapshot_chunks.append(f"### {path}\n(could not read: {exc})\n")
            continue
        snapshot_chunks.append(
            f"### {path}\n```\n{''.join(lines)}\n```\n"
        )

    from harness.gateway import NodeRole

    critique_user_prompt = (
        "## Modified Files\n" + "\n".join(snapshot_chunks) +
        "\n\nProduce the JSON critique now."
    )
    critique_messages = [
        {"role": "system", "content": _CODE_REVIEW_SYSTEM_PROMPT},
        {"role": "user", "content": critique_user_prompt},
    ]

    try:
        critique_response, new_budget = await gateway.dispatch(
            messages=critique_messages,
            role=NodeRole.CODE_REVIEWER,
            budget_remaining_usd=budget,
        )
    except Exception as exc:
        logger.warning("[code_review] Reviewer dispatch failed: %s — passing through.", exc)
        return {"node_state": {"current_node": "code_review", "skipped": True, "repatched": False}}

    token_tracker = gateway.aggregate_tokens(state.get("token_tracker", {}), critique_response.usage)

    try:
        from harness.trust import _strip_code_fences  # type: ignore
        critique_text = _strip_code_fences(critique_response.content)
    except Exception:
        critique_text = critique_response.content.strip()

    try:
        critique = json.loads(critique_text)
        findings = critique.get("findings", []) if isinstance(critique, dict) else []
        if not isinstance(findings, list):
            findings = []
        # Schema-drift sentinel: response parsed cleanly, but the canonical
        # "findings" key is missing AND the LLM produced something else at
        # the top level (e.g. "issues" / "comments"). The previous behaviour
        # silently treated this as "code is clean" and skipped the security
        # gate — same silent-drift shape as the discovery 0-modules bug.
        if isinstance(critique, dict) and "findings" not in critique:
            other_keys = sorted(k for k in critique.keys() if not k.startswith("_"))
            if other_keys:
                logger.warning(
                    "[code_review] Response parsed cleanly but the canonical "
                    "'findings' key is missing — the LLM used non-canonical "
                    "top-level keys=%s. Treating as no-findings (code passes "
                    "review) BUT the security gate would skip; verify the "
                    "review JSON in docs/CODE_REVIEW.md before trusting the "
                    "result.", other_keys,
                )
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("[code_review] Critique was not valid JSON (%s) — passing through.", exc)
        findings = []
        critique = {"findings": []}

    # Persist the critique regardless of whether findings exist.
    docs_dir = os.path.join(workspace, "docs")
    try:
        os.makedirs(docs_dir, exist_ok=True)
        review_path = os.path.join(docs_dir, "CODE_REVIEW.md")
        with open(review_path, "w", encoding="utf-8") as f:
            f.write("# Code Review\n\n")
            f.write("*Generated by the independent code-reviewer LLM.*\n\n")
            if not findings:
                f.write("No findings.\n")
            else:
                f.write("```json\n")
                f.write(json.dumps(critique, indent=2))
                f.write("\n```\n")
    except OSError as exc:
        logger.warning("[code_review] Failed to write CODE_REVIEW.md: %s", exc)

    loop_counter["review_code"] = counter + 1

    if not findings:
        logger.info("[code_review] No findings — passing through to security scan.")
        return {
            "token_tracker": token_tracker,
            "budget_remaining_usd": new_budget,
            "loop_counter": loop_counter,
            "reviewer_comments_code": json.dumps(critique),
            "node_state": {
                "current_node": "code_review",
                "skipped": False,
                "repatched": False,
                "findings_count": 0,
            },
        }

    # Re-patch call. Use the patcher format-reminder so the patcher model
    # returns the same SEARCH/REPLACE block syntax patching_node uses.
    from harness.patcher import process_llm_patch_output

    _CODE_REVIEW_FORMAT_REMINDER = """[CRITICAL FORMAT INSTRUCTION]
You MUST respond using ONLY the patch block syntax below. No explanations, no markdown fences, no text outside the blocks.

Valid blocks:
<<<CREATE_FILE>>>
file: path/to/file.ext
content:
<complete file contents>
<<<END_CREATE_FILE>>>

<<<REPLACE_BLOCK>>>
file: path/to/file.ext
search:
<exact lines to find>
replace:
<exact replacement lines>
<<<END_REPLACE_BLOCK>>>

<<<DELETE_BLOCK>>>
file: path/to/file.ext
search:
<exact lines to delete>
<<<END_DELETE_BLOCK>>>

<<<INSERT_AT_BLOCK>>>
file: path/to/file.ext
anchor: <function or class name>
placement: before|after
content:
<lines to insert>
<<<END_INSERT_AT_BLOCK>>>

Generate the patches that address every finding below. Output only the blocks — no other text."""

    repatch_user_prompt = (
        f"## Reviewer Findings (JSON)\n```json\n{json.dumps(critique, indent=2)}\n```\n\n"
        f"## Modified Files Snapshot\n" + "\n".join(snapshot_chunks) +
        f"\n\n{_CODE_REVIEW_FORMAT_REMINDER}"
    )
    repatch_messages = [
        {"role": "system", "content": "You are a senior software engineer. Apply the reviewer's feedback as patch blocks only."},
        {"role": "user", "content": repatch_user_prompt},
    ]

    try:
        repatch_response, new_budget = await gateway.dispatch(
            messages=repatch_messages,
            role=NodeRole.PATCHING,
            budget_remaining_usd=new_budget,
        )
    except Exception as exc:
        logger.warning("[code_review] Re-patch dispatch failed: %s — proceeding without re-patch.", exc)
        return {
            "token_tracker": token_tracker,
            "budget_remaining_usd": new_budget,
            "loop_counter": loop_counter,
            "reviewer_comments_code": json.dumps(critique),
            "node_state": {
                "current_node": "code_review",
                "skipped": False,
                "repatched": False,
                "findings_count": len(findings),
            },
        }

    token_tracker = gateway.aggregate_tokens(token_tracker, repatch_response.usage)

    allowed_paths = _build_patcher_allowlist(workspace)
    patch_results, new_modified_files = await process_llm_patch_output(
        repatch_response.content,
        workspace,
        modified_files,
        allowed_paths=allowed_paths,
    )
    success_count = sum(1 for r in patch_results if r.success)
    repatched = success_count > 0

    logger.info(
        "[code_review] Findings=%d, re-patched=%d/%d files (success=%s).",
        len(findings), success_count, len(patch_results), repatched,
    )

    return {
        "modified_files": new_modified_files,
        "token_tracker": token_tracker,
        "budget_remaining_usd": new_budget,
        "loop_counter": loop_counter,
        "reviewer_comments_code": json.dumps(critique),
        "node_state": {
            "current_node": "code_review",
            "skipped": False,
            "repatched": repatched,
            "findings_count": len(findings),
            "repatch_success": success_count,
            "repatch_total": len(patch_results),
        },
    }


async def generate_deployment_spec_node(state: AgentState) -> dict[str, Any]:
    """
    Runs workspace telemetry + architecture spec through the deployment
    synthesizer to produce DEPLOYMENT_BLUEPRINT.md. Sets current_gate = "DEPLOYMENT".
    """
    workspace = state.get("workspace_path", os.getcwd())
    output_dir = os.path.join(workspace, "docs")
    os.makedirs(output_dir, exist_ok=True)

    blueprint_path = os.path.join(output_dir, "DEPLOYMENT_BLUEPRINT.md")

    # Use the deploy module's telemetry + synthesis
    try:
        from harness.deploy import scan_workspace_telemetry

        telemetry = scan_workspace_telemetry(workspace)

        gateway = get_gateway()
        if gateway:
            arch_path = state.get("spec_architecture_path", os.path.join(output_dir, "SPEC_ARCHITECTURE.md"))
            arch_content = ""
            if os.path.isfile(arch_path):
                with open(arch_path, "r", encoding="utf-8") as f:
                    arch_content = f.read()[:5000]

            from harness.gateway import NodeRole
            prompt = f"""You are a DevOps Architect. Produce a DEPLOYMENT_BLUEPRINT.md based on the telemetry and architecture below.

## Workspace Telemetry
```json
{json.dumps(telemetry, indent=2, default=str)}
```

## Architecture Spec
{arch_content if arch_content else "(no SPEC_ARCHITECTURE.md)"}

## Output Sections
1. Container Inventory (services, base images, build contexts)
2. Network Topology (bridge networks, exposed ports)
3. Volume Configuration (named volumes, bind mounts)
4. Environment Variables (per service)
5. Health Check Configuration
6. Scaling & Resource Limits
7. Deployment Sequence (order of container startup)

Output as clean Markdown. Do NOT wrap the document in an outer
```markdown … ``` fence — emit the body directly, starting with the
first heading. Fences are reserved for code blocks INSIDE the document."""

            messages = [
                {"role": "system", "content": "You are a DevOps architect. Output clean Markdown — no outer code fences around the document."},
                {"role": "user", "content": prompt},
            ]
            current_budget = state.get("budget_remaining_usd", 0.0)
            if current_budget <= 0:
                logger.warning("[deployment_spec] Budget exhausted; falling back to deterministic blueprint.")
                content = f"# DEPLOYMENT_BLUEPRINT.md\n\nWorkspace: {workspace}\n\nTelemetry: {json.dumps(telemetry, indent=2, default=str)}\n\n(LLM blueprint skipped — budget exhausted.)"
                budget = current_budget
            else:
                response, budget = await gateway.dispatch(
                    messages=messages, role=NodeRole.PLANNING,
                    budget_remaining_usd=current_budget,
                )
                content = response.content.strip()
        else:
            content = f"# DEPLOYMENT_BLUEPRINT.md\n\nWorkspace: {workspace}\n\nTelemetry: {json.dumps(telemetry, indent=2, default=str)}"
            budget = state.get("budget_remaining_usd", 0.0)

        with open(blueprint_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("[deployment_spec] DEPLOYMENT_BLUEPRINT.md written (%d chars).", len(content))
    except Exception as exc:
        logger.warning("[deployment_spec] Failed: %s", exc)
        budget = state.get("budget_remaining_usd", 0.0)
        with open(blueprint_path, "w", encoding="utf-8") as f:
            f.write(f"# DEPLOYMENT_BLUEPRINT.md\n\nError: {exc}")

    return {
        "deployment_blueprint_path": blueprint_path,
        "current_gate": "DEPLOYMENT",
        "budget_remaining_usd": budget,
        "node_state": {"current_node": "generate_deployment_spec"},
    }


# ---------------------------------------------------------------------------
# 6d. Route After Discovery Interview
# ---------------------------------------------------------------------------

def route_after_discovery(state: AgentState) -> str:
    """
    Routes after discovery_interview_loop completes.

    Decision matrix:
        discovery_complete == true  → write_spec_node (serialize to .md)
        discovery_complete == false AND critical > 0 → loop back to discovery node
        discovery_complete == false AND no critical → write_spec_node (non-critical only)
        user typed 'DONE' with critical remaining → alert + loop back
        user typed 'SUSPEND' → save & quit (route to END)

    Loop ceiling: the per-gate discovery_question_count is compared against
    GatewayConfig.max_discovery_iterations (default 10). When the cap is
    reached we forward to write_spec_node with the current draft instead of
    looping again — prevents a runaway loop from a confused user or a
    misbehaving LLM that keeps emitting follow-ups.
    """
    node_state = state.get("node_state", {})
    complete = node_state.get("discovery_complete", False)
    critical = node_state.get("discovery_critical_remaining", 0)
    gate = state.get("current_gate", "REQUIREMENTS")
    rounds = int(node_state.get("discovery_question_count", 0) or 0)

    if node_state.get("hitl_suspend"):
        logger.info("[router] Discovery: developer chose to suspend. Routing to END.")
        return "__end__"

    # Loop-cap short-circuit. Honoured before the user_done_with_critical
    # branch so a user who keeps typing DONE doesn't trap us either.
    cap = 10
    try:
        from harness.gateway import get_gateway_config
        gw_config = get_gateway_config()
        if gw_config is not None:
            cap = int(getattr(gw_config, "max_discovery_iterations", 10) or 10)
    except Exception:  # noqa: BLE001
        pass
    if rounds >= cap:
        logger.warning(
            "[router] Discovery hit max_discovery_iterations (%d). "
            "Forwarding to write_spec_node with the current draft.",
            cap,
        )
        return "write_spec_node"

    if node_state.get("user_done_with_critical"):
        # User tried to exit with critical unknowns — route back to the correct discovery node
        if gate == "DEPLOYMENT":
            return "deployment_discovery_node"
        elif gate == "ARCHITECTURE":
            return "architecture_discovery_node"
        else:
            return "discovery_interview_loop"

    if complete or (not complete and critical == 0):
        return "write_spec_node"

    # Still incomplete with critical unknowns → loop back to the right discovery node
    if gate == "REQUIREMENTS":
        return "requirements_discovery_node"
    elif gate == "ARCHITECTURE":
        return "architecture_discovery_node"
    else:
        return "deployment_discovery_node"


# ---------------------------------------------------------------------------
# 6e. Route After Human Gatekeeper
# ---------------------------------------------------------------------------

def route_after_gatekeeper(state: AgentState) -> str:
    """
    Routes based on current_gate and the gatekeeper's decision in node_state.
    
    The human_gatekeeper_node sets node_state.gatekeeper_action to:
        - "approve" → proceed to next phase
        - "refine"  → loop back to the current phase's generator
        - "manual"  → proceed (manual edits done)
        - "suspend" → save & quit (route to END)
    """
    gate = state.get("current_gate", "")
    action = state.get("node_state", {}).get("gatekeeper_action", "approve")

    if action == "suspend":
        logger.info("[router] Gatekeeper: developer chose to suspend. Routing to END.")
        return "__end__"

    if action == "refine":
        # Route back to the corresponding discovery node for re-generation
        if gate == "REQUIREMENTS":
            return "requirements_discovery_node"
        elif gate == "ARCHITECTURE":
            return "architecture_discovery_node"
        elif gate == "DEPLOYMENT":
            return "generate_deployment_spec_node"

    # approve or manual: proceed forward
    if gate == "REQUIREMENTS":
        return "architecture_discovery_node"
    elif gate == "ARCHITECTURE":
        return "patching_node"
    elif gate == "DEPLOYMENT":
        return "deployment_node"

    return "__end__"


# ---------------------------------------------------------------------------
# 7. Route After HITL: Always Back to Compiler
# ---------------------------------------------------------------------------

def route_after_security_scan(state: AgentState) -> Literal["repair_node", "human_intervention_node", "deployment_discovery_node", "__end__"]:
    """
    Conditional edge router executed after security_scan_node completes.

    Decision matrix:
        No security findings AND Flutter project → END (mobile builds don't
                                                  fit the docker-compose deploy
                                                  pipeline; user picks up the
                                                  artifact from build/app/outputs/)
        No security findings                    → deployment_discovery_node
        Security findings AND sec_attempts < 2  → repair_node (fix the vulnerability)
        Security findings AND sec_attempts >= 2 → human_intervention_node
        budget_remaining <= 0                   → human_intervention_node

    Routes to repair_node so security findings travel through the same
    formatter (``_format_diagnostics_for_repair``) and escalation logic
    (cheap → reasoning model on round 3) that compile errors use. The
    repair_node prompt detects the scanner-prefixed error_codes
    populated by security_scan_node and switches its framing sentence
    to make the security context explicit to the LLM. After repair, the
    compiler verifies the fix and security_scan_node re-verifies clean.
    """
    budget_remaining: float = state.get("budget_remaining_usd", 0.0)
    loop_counter: dict[str, int] = state.get("loop_counter", {})
    sec_attempts: int = loop_counter.get("security", 0)
    max_sec_attempts: int = 2
    compiler_errors = state.get("compiler_errors", [])

    # Check budget first
    if budget_remaining <= 0.0:
        logger.warning("[router] Budget exhausted ($%.4f). Routing to HITL.", budget_remaining)
        return "human_intervention_node"

    # If no compiler_errors populated, security scan passed
    if not compiler_errors:
        # Mobile short-circuit (M-1): Flutter projects don't fit the
        # docker-compose deployment model. Skip deployment_* and end after
        # the security scan passes. iOS builds need macOS anyway and would
        # fail in the Linux sandbox; Android artifacts live in
        # build/app/outputs/ for the user to pick up.
        from harness.impact import _is_flutter_project
        workspace_path = state.get("workspace_path", "")
        if workspace_path and _is_flutter_project(workspace_path):
            logger.info("[router] Flutter project detected. Skipping deploy pipeline (M-1). Routing to END.")
            return "__end__"
        logger.info("[router] Security scan clean. Routing to deployment discovery.")
        return "deployment_discovery_node"

    # Security findings exist — check attempt limit
    if sec_attempts >= max_sec_attempts:
        logger.warning(
            "[router] Security fix limit reached (%d/%d). %d finding(s) remain. Routing to HITL.",
            sec_attempts, max_sec_attempts, len(compiler_errors),
        )
        return "human_intervention_node"

    logger.info(
        "[router] %d security finding(s) detected. Routing to repair_node for fix (attempt %d/%d).",
        len(compiler_errors), sec_attempts, max_sec_attempts,
    )
    return "repair_node"


# ---------------------------------------------------------------------------
# 7. Route After HITL: Always Back to Compiler
# ---------------------------------------------------------------------------

def route_after_hitl(state: AgentState) -> Literal["compiler_node", "__end__"]:
    """
    After human intervention, always route back to compiler_node for re-validation.
    Exceptions:
        - Developer chose to abandon ([q]) → END with git rollback
        - Developer chose to suspend ([s]) → END without rollback

    Note: Memory cleanse for HITL resolution is handled inside
    compiler_node when exit_code == 0 after the re-validation build passes.
    """
    node_state: dict[str, Any] = state.get("node_state", {})
    if node_state.get("hitl_suspend", False):
        logger.info("[router] HITL: Developer chose to suspend. Routing to END.")
        return "__end__"
    if node_state.get("hitl_abandon", False):
        logger.info("[router] HITL: Developer chose to abandon. Routing to END.")
        return "__end__"
    logger.info("[router] HITL resolved. Routing to compiler_node for re-validation.")
    return "compiler_node"


# ---------------------------------------------------------------------------
# 8. Route After Planning: Always to Patching
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 9. Graph Builder — Assembles the Full StateGraph
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """
    Construct and return the complete LangGraph StateGraph with all nodes,
    edges, and conditional routing logic.

    Graph topology:

        START
          │
          ▼
    planning_node
          │
          ▼
    patching_node
          │
          ▼
    compiler_node ────── exit 0 ──────────► END
          │
          │ exit ≠ 0
          ▼
    ┌── route_after_compiler ──┐
    │                          │
    │ repairs < 3    repairs >= 3
    ▼                          ▼
    repair_node         human_intervention_node
    │                          │
    └──────► compiler_node ◄───┘
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        raise ImportError(
            "langgraph is required. Install with: pip install langgraph>=0.4.0"
        )

    # Create the graph bound to our AgentState type
    graph = StateGraph(AgentState)

    # Register core graph nodes
    graph.add_node("planning_node", planning_node)
    graph.add_node("patching_node", patching_node)
    graph.add_node("compiler_node", compiler_node)
    graph.add_node("repair_node", repair_node)
    graph.add_node("human_intervention_node", human_intervention_node)

    # Register security scan node for SAST + secret auditing (post-compile gatekeeper)
    from harness.security import security_scan_node as _security_scan_node
    graph.add_node("security_scan_node", _security_scan_node)

    # Register lintgate node for deterministic format verification
    from harness.lintgate import lintgate_node as _lintgate_node
    graph.add_node("lintgate_node", _lintgate_node)

    # Register speculative node for multi-variant branching
    from harness.speculative import speculate_node as _speculate_node
    graph.add_node("speculative_node", _speculate_node)

    # Register test-generation node — runs after speculative branching, before
    # lintgate, so the deterministic lint pass formats generated tests too.
    from harness.test_generation import (
        test_generation_node as _test_generation_node,
        route_after_test_generation as _route_after_test_generation,
    )
    graph.add_node("test_generation_node", _test_generation_node)

    # Register exhaustive discovery nodes
    graph.add_node("requirements_discovery_node", requirements_discovery_node)
    graph.add_node("architecture_discovery_node", architecture_discovery_node)
    graph.add_node("write_spec_node", write_spec_node)
    from harness.cli import discovery_interview_loop as _discovery_interview_loop
    graph.add_node("discovery_interview_loop", _discovery_interview_loop)

    # Register human gatekeeper node for final review
    from harness.cli import human_gatekeeper_node as _human_gatekeeper_node
    graph.add_node("human_gatekeeper_node", _human_gatekeeper_node)

    # Register reviewer LLM nodes (DOC_REVIEWER + CODE_REVIEWER). Each is a
    # no-op when its corresponding *_reviewer_primary slot is unset in config.
    graph.add_node("spec_review_node", spec_review_node)
    graph.add_node("code_review_node", code_review_node)

    # Register deployment spec node
    graph.add_node("generate_deployment_spec_node", generate_deployment_spec_node)

    # =====================================================================
    # Discovery pipeline (exhaustive zero-unknowns protocol):
    # START → requirements_discovery_node → discovery_interview_loop
    #   loop: discovery_interview ← route_after_discovery → requirements_discovery (if incomplete)
    #   done: discovery_interview → route_after_discovery → write_spec_node
    #   write_spec_node → human_gatekeeper_node (REQUIREMENTS final review)
    #   gatekeeper: approve → architecture_discovery_node
    #   architecture_discovery_node → discovery_interview_loop
    #   loop: discovery_interview ← route_after_discovery → architecture_discovery (if incomplete)
    #   done: discovery_interview → route_after_discovery → write_spec_node
    #   write_spec_node → human_gatekeeper_node (ARCHITECTURE final review)
    #   gatekeeper: approve → patching_node
    # =====================================================================

    # =====================================================================
    # START routing: if skip_discovery, jump directly to patching_node
    # =====================================================================
    def route_after_start(state: AgentState) -> Literal["requirements_discovery_node", "patching_node"]:
        if state.get("skip_discovery", False):
            logger.info("[router] --skip-discovery active. Routing START → patching_node.")
            return "patching_node"
        return "requirements_discovery_node"

    graph.add_conditional_edges(
        START,
        route_after_start,
        {
            "requirements_discovery_node": "requirements_discovery_node",
            "patching_node": "patching_node",
        },
    )

    # Requirements discovery loop
    graph.add_edge("requirements_discovery_node", "discovery_interview_loop")
    graph.add_conditional_edges(
        "discovery_interview_loop",
        route_after_discovery,
        {
            "requirements_discovery_node": "requirements_discovery_node",
            "architecture_discovery_node": "architecture_discovery_node",
            "deployment_discovery_node": "deployment_discovery_node",
            "write_spec_node": "write_spec_node",
            # Self-loop: user typed DONE with critical unknowns → re-display menu
            "discovery_interview_loop": "discovery_interview_loop",
            "__end__": END,
        },
    )

    # After write_spec, route through the doc-reviewer LLM (no-op if reviewer
    # is not configured). The reviewer either revises the spec and asks the
    # user follow-up questions, or short-circuits to the human gatekeeper.
    graph.add_edge("write_spec_node", "spec_review_node")

    def route_after_spec_review(state: AgentState) -> Literal[
        "discovery_interview_loop", "human_gatekeeper_node"
    ]:
        """The human gatekeeper is the FINAL review gate. The LLM reviewer
        either bails (no model configured / cycle cap / no follow-ups) and we
        route straight to the gatekeeper, or it produces follow-up questions
        and we route through one more pass of the discovery interview loop
        before the gatekeeper sees the upgraded spec."""
        if state.get("skip_discovery", False):
            return "human_gatekeeper_node"
        node_state = state.get("node_state", {}) or {}
        if node_state.get("skipped", False):
            return "human_gatekeeper_node"
        followups = state.get("reviewer_followups", []) or []
        if followups:
            logger.info(
                "[router] spec_review produced %d follow-up question(s) — routing to discovery interview.",
                sum(len(m.get("questions", [])) for m in followups),
            )
            return "discovery_interview_loop"
        return "human_gatekeeper_node"

    graph.add_conditional_edges(
        "spec_review_node",
        route_after_spec_review,
        {
            "discovery_interview_loop": "discovery_interview_loop",
            "human_gatekeeper_node": "human_gatekeeper_node",
        },
    )

    # Gatekeeper routes: approve → next phase, refine → loop back
    graph.add_conditional_edges(
        "human_gatekeeper_node",
        route_after_gatekeeper,
        {
            "requirements_discovery_node": "requirements_discovery_node",
            "architecture_discovery_node": "architecture_discovery_node",
            "generate_deployment_spec_node": "generate_deployment_spec_node",
            "patching_node": "patching_node",
            "deployment_node": "deployment_node",
            "__end__": END,
        },
    )

    # Architecture discovery loop (entered from gatekeeper approve)
    graph.add_edge("architecture_discovery_node", "discovery_interview_loop")

    # =====================================================================
    # Code generation pipeline (after ARCHITECTURE gate approved):
    # patching → speculative → lintgate → compiler
    # =====================================================================
    graph.add_edge("patching_node", "speculative_node")
    # speculative_node → test_generation_node → conditional edge:
    #   - tests passed (or skipped) → lintgate_node
    #   - tests failed → repair_node (TEST_FAILURE diagnostics surfaced)
    #   - env_misconfig (no LLM gateway / max iterations) → human_intervention_node
    graph.add_edge("speculative_node", "test_generation_node")
    graph.add_conditional_edges(
        "test_generation_node",
        _route_after_test_generation,
        {
            "lintgate_node": "lintgate_node",
            "repair_node": "repair_node",
            "human_intervention_node": "human_intervention_node",
        },
    )
    graph.add_edge("lintgate_node", "compiler_node")

    # =====================================================================
    # Compiler → repair loop + security gate
    #
    # On a clean compile, divert through code_review_node first. The reviewer
    # may apply a re-patch that needs re-validation, so it routes back to
    # compiler_node if it re-patched; otherwise it proceeds to security_scan.
    # The reviewer is a no-op when code_reviewer_primary is unset in config,
    # so this path is behavior-compatible with the pre-reviewer flow.
    # =====================================================================
    graph.add_conditional_edges(
        "compiler_node",
        route_after_compiler,
        {
            # Clean compile → code reviewer (no-op pass-through if unconfigured).
            "security_scan_node": "code_review_node",
            "repair_node": "repair_node",
            "human_intervention_node": "human_intervention_node",
            # pytest exit=5 with source present → generate tests instead of
            # burning a repair iteration on a non-error.
            "test_generation_node": "test_generation_node",
        },
    )

    def route_after_code_review(state: AgentState) -> Literal[
        "compiler_node", "security_scan_node"
    ]:
        """If the reviewer re-patched, re-validate by going back through the
        compiler. If not (no findings, cycle cap, or unconfigured), proceed
        straight to security_scan_node — exactly today's behavior."""
        node_state = state.get("node_state", {}) or {}
        if node_state.get("repatched", False):
            logger.info("[router] code_review re-patched — re-running compiler.")
            return "compiler_node"
        return "security_scan_node"

    graph.add_conditional_edges(
        "code_review_node",
        route_after_code_review,
        {
            "compiler_node": "compiler_node",
            "security_scan_node": "security_scan_node",
        },
    )

    # After repair, go directly to compiler (skip lintgate to avoid reformatting
    # the file between repair attempts, which would break SEARCH/REPLACE matching).
    graph.add_edge("repair_node", "compiler_node")

    # =====================================================================
    # Deployment discovery pipeline (after security scan clean):
    # security_scan → deployment_discovery_node → discovery_interview_loop
    #   loop: discovery_interview ← route_after_discovery → deployment_discovery (if incomplete)
    #   done: discovery_interview → route_after_discovery → write_spec_node
    #   write_spec_node → human_gatekeeper_node (DEPLOYMENT final review)
    #   gatekeeper: approve → deployment_node → END
    #   gatekeeper: refine → deployment_discovery_node
    # =====================================================================

    # Register deployment discovery node
    graph.add_node("deployment_discovery_node", deployment_discovery_node)

    # Route security_scan clean → deployment discovery (or END for Flutter);
    # findings → repair_node so security fixes go through the same
    # _format_diagnostics_for_repair + escalation path as compile errors.
    graph.add_conditional_edges(
        "security_scan_node",
        route_after_security_scan,
        {
            "deployment_discovery_node": "deployment_discovery_node",
            "repair_node": "repair_node",
            "human_intervention_node": "human_intervention_node",
            "__end__": END,
        },
    )

    # Deployment discovery → interview loop
    graph.add_edge("deployment_discovery_node", "discovery_interview_loop")

    # Register deployment node
    from harness.deploy import deployment_node as _deployment_node
    graph.add_node("deployment_node", _deployment_node)

    # Deployment conditional edges — after deployment, either end (success) or route to repair/HITL
    graph.add_conditional_edges(
        "deployment_node",
        route_after_compiler,
        {
            "security_scan_node": "security_scan_node",
            "repair_node": "repair_node",
            "human_intervention_node": "human_intervention_node",
            "test_generation_node": "test_generation_node",
        },
    )

    # After HITL resolution, go back to compiler (or END if abandoned)
    graph.add_conditional_edges(
        "human_intervention_node",
        route_after_hitl,
        {
            "compiler_node": "compiler_node",
            "__end__": END,
        },
    )

    logger.info("[graph] Full StateGraph topology assembled successfully.")
    return graph


# ---------------------------------------------------------------------------
# 12. Compile Helper — Builds the Graph with Checkpointer
# ---------------------------------------------------------------------------

def compile_graph(checkpointer: Any = None) -> Any:
    """
    Compile the StateGraph with an optional checkpointer for persistence.

    Args:
        checkpointer: A LangGraph-compatible checkpointer (MemorySaver, AsyncSqliteSaver, etc.).
                       If None, the graph runs without persistence.

    Returns:
        A compiled LangGraph runnable ready for .ainvoke() or .astream().
    """
    graph = build_graph()
    if checkpointer is not None:
        compiled = graph.compile(checkpointer=checkpointer)
    else:
        compiled = graph.compile()
    return compiled


# ---------------------------------------------------------------------------
# 13. Graph Execution Entry Point
# ---------------------------------------------------------------------------

async def run_graph(
    *,
    workspace_path: str,
    prompt: str,
    build_command: str,
    allow_network: bool = False,
    budget_usd: float = 2.00,
    session_id: str = "",
    checkpointer: Any = None,
    thread_id: Optional[str] = None,
    spec_override: Optional[str] = None,
    skip_discovery: bool = False,
    is_resume: bool = False,
    lintgate_config: Optional[dict[str, Any]] = None,
    deployment_config: Optional[dict[str, Any]] = None,
    sandbox_config: Optional[dict[str, Any]] = None,
    test_generation_config: Optional[dict[str, Any]] = None,
) -> AgentState:
    """
    Execute the full agent graph from start to finish.

    This is the primary async entry point called by the CLI layer.

    Args:
        workspace_path: Absolute path to the target repository.
        prompt: The engineering task description.
        build_command: The shell command to build/verify the project.
        allow_network: Whether the sandbox permits outbound network.
        budget_usd: Hard dollar cap for LLM calls.
        session_id: Human-readable session identifier (falls back to UUIDv4).
        checkpointer: LangGraph checkpointer for persistence. If None, runs ephemerally.
        thread_id: LangGraph thread ID for checkpoint lookups. Auto-generated if None.

    Returns:
        The final AgentState after graph completion.
    """
    import uuid

    # Generate session identifiers
    if not session_id:
        session_id = str(uuid.uuid4())
    if thread_id is None:
        thread_id = session_id  # Use session_id as thread_id for simplicity

    # Build initial state
    initial_state = create_initial_state(
        workspace_path=workspace_path,
        initial_prompt=prompt,
        build_command=build_command,
        allow_network=allow_network,
        budget_usd=budget_usd,
        session_id=session_id,
        spec_override=spec_override,
        skip_discovery=skip_discovery,
    )

    # Per-node config sections — read by lintgate_node and deployment_node
    # respectively. These are free-form dicts on the state; nodes consult
    # them via state.get("lintgate_config", {}) etc.
    if lintgate_config is not None:
        initial_state["lintgate_config"] = lintgate_config  # type: ignore[typeddict-unknown-key]
    if deployment_config is not None:
        initial_state["deployment_config"] = deployment_config  # type: ignore[typeddict-unknown-key]
    if test_generation_config is not None:
        initial_state["test_generation_config"] = test_generation_config  # type: ignore[typeddict-unknown-key]

    # Pre-flight toolchain adaptation: pick the right docker image (and
    # network bit) NOW so the very first compile lands on, e.g.,
    # python:3.12-slim instead of wasting a build cycle on ubuntu:22.04
    # exiting 127 with "python3: not found". compiler_node's own call to
    # the same helper is idempotent — it becomes a no-op once we've
    # pre-adapted here.
    (
        adapted_cfg,
        adapted_allow_network,
        image_was_adapted,
        network_was_adapted,
        ro_root_was_adapted,
    ) = _apply_toolchain_adaptation(build_command, sandbox_config, allow_network)
    if image_was_adapted:
        logger.info(
            "[run_graph] Pre-flight sandbox docker_image set to %r "
            "to match build toolchain implied by: %s",
            adapted_cfg["docker_image"], build_command,
        )
    if network_was_adapted:
        logger.info(
            "[run_graph] Pre-flight allow_network=True because build command "
            "requires registry access: %s", build_command,
        )
        initial_state["allow_network"] = adapted_allow_network
    if ro_root_was_adapted:
        logger.info(
            "[run_graph] Pre-flight sandbox.read_only_root=False because build "
            "command installs packages into system locations: %s", build_command,
        )
    if sandbox_config is not None or image_was_adapted or ro_root_was_adapted:
        initial_state["sandbox_config"] = adapted_cfg  # type: ignore[typeddict-unknown-key]

    # Compile graph with checkpointer
    compiled_graph = compile_graph(checkpointer=checkpointer)

    # Runtime configuration for LangGraph
    config: dict[str, Any] = {
        "configurable": {
            "thread_id": thread_id,
            "session_id": session_id,
        }
    }

    logger.info(
        "[run_graph] Starting graph execution. thread_id=%s session_id=%s is_resume=%s",
        thread_id, session_id, is_resume,
    )

    # Resume mode: do NOT pass a full initial_state — every key in the
    # input dict overwrites the corresponding channel in the checkpointed
    # state, so a fresh AgentState would reset messages, loop_counter,
    # current_gate, node_state, etc. back to their zero values and the
    # graph would re-enter at START → requirements_discovery_node even
    # though the prior session was deep in architecture_discovery.
    # Passing None tells LangGraph "no input update, just continue from
    # the checkpoint" — the graph picks up at exactly the node that was
    # in-flight when the session was saved.
    if is_resume:
        invoke_input: Optional[AgentState] = None
        logger.info(
            "[run_graph] Resume mode: invoking with no input update; "
            "the graph will continue from the last checkpointed node."
        )
    else:
        invoke_input = initial_state

    # Execute the graph — ainvoke streams all state updates and returns final state
    final_state: AgentState = await compiled_graph.ainvoke(invoke_input, config)  # type: ignore[arg-type,return-value]

    logger.info("[run_graph] Graph execution complete. Final exit_code=%d", final_state.get("exit_code", -1))
    return final_state
