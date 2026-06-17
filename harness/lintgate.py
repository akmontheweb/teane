"""
Deterministic Patch Verification Layer — Lint & Format Lock.

This module implements:
    - FormatRegistry: Maps file extensions to deterministic auto-formatter commands.
      Runs hyper-fast local tools (gofmt, black, ruff, prettier, rustfmt, clang-format)
      to clean up whitespace, missing brackets, and minor syntax issues automatically.
    - lintgate_node: LangGraph node that runs AFTER patches are applied but BEFORE
      the heavy compiler pipeline. Never calls an LLM — purely deterministic subprocess.
    - Linter support: Optionally runs lightweight linters (ruff check, eslint, clippy)
      to catch deeper issues cheaply before the build.

Integration:
    - Placed between patching_node → lintgate_node → compiler_node
    - Also after repair_node → lintgate_node → compiler_node
    - Avoids paid LLM loops for trivial formatting issues.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Common PEP 484 names that LLM-generated code routinely uses without
# remembering to import. Kept narrow to names that are essentially never
# bound as local identifiers, so a name-only check (no full scope analysis)
# is safe.
_TYPING_AUTO_IMPORT_NAMES = frozenset({
    "Optional", "List", "Dict", "Tuple", "Set", "Union", "Any",
    "Callable", "Type", "Iterator", "Iterable", "Generator",
    "Mapping", "Sequence", "FrozenSet", "DefaultDict", "Awaitable",
    "Coroutine", "AsyncIterator", "AsyncIterable", "AsyncGenerator",
    "TypeVar", "Literal", "Final",
})


def _inject_missing_typing_imports(filepath: str) -> int:
    """Pre-lint sweep: add `from typing import X` for typing names that
    appear in the file but aren't imported. Returns count of names injected.

    Skips files with syntax errors (ruff will report those upstream).
    Idempotent — running twice is a no-op.
    """
    import ast

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            src = f.read()
    except (OSError, UnicodeDecodeError):
        return 0
    if not src.strip():
        return 0
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0

    imported: set[str] = set()
    first_typing_import: Optional[ast.ImportFrom] = None
    all_typing_aliases: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in (
            "typing", "typing_extensions",
        ):
            for alias in node.names:
                imported.add(alias.asname or alias.name)
            if node.module == "typing":
                if first_typing_import is None:
                    first_typing_import = node
                all_typing_aliases.extend(
                    alias.name + (f" as {alias.asname}" if alias.asname else "")
                    for alias in node.names
                )

    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _TYPING_AUTO_IMPORT_NAMES:
            referenced.add(node.id)

    needed = referenced - imported
    if not needed:
        return 0

    lines = src.splitlines(keepends=True)
    if first_typing_import is not None:
        start = first_typing_import.lineno - 1
        end = getattr(first_typing_import, "end_lineno", first_typing_import.lineno)
        merged_names = sorted(set(all_typing_aliases) | needed)
        lines[start:end] = [f"from typing import {', '.join(merged_names)}\n"]
    else:
        insert_at = _typing_import_insertion_point(tree, lines)
        lines.insert(insert_at, f"from typing import {', '.join(sorted(needed))}\n")

    new_src = "".join(lines)
    if new_src == src:
        return 0
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_src)
    except OSError:
        return 0
    return len(needed)


def _typing_import_insertion_point(tree: Any, lines: list[str]) -> int:
    """Return 0-based line index for inserting a new `from typing import`
    line. Goes AFTER shebang, encoding cookie, module docstring, and
    `from __future__` imports so future-annotations stays at file head.
    """
    import ast as _ast
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    if (
        insert_at < len(lines)
        and lines[insert_at].lstrip().startswith("#")
        and "coding" in lines[insert_at]
    ):
        insert_at += 1
    if (
        tree.body
        and isinstance(tree.body[0], _ast.Expr)
        and isinstance(tree.body[0].value, _ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        insert_at = max(insert_at, getattr(tree.body[0], "end_lineno", tree.body[0].lineno))
    for node in tree.body:
        if isinstance(node, _ast.ImportFrom) and node.module == "__future__":
            insert_at = max(insert_at, getattr(node, "end_lineno", node.lineno))
    return insert_at


def _classify_files_by_git_status(
    files: list[str], workspace_path: str
) -> tuple[set[str], set[str]]:
    """
    Partition ``files`` into (created_in_session, pre_existing) sets using
    ``git status --porcelain``.

    A file is considered "created in this session" if git reports it as
    untracked (``??``) or staged-as-new (``A``). Anything else — modified
    (``M``), renamed, or absent from the status output — is treated as
    pre-existing: the formatter must not rewrite the whole file because
    the user owns the style outside the patch region.

    Falls back to ``(set(), set(files))`` if git status fails or the
    workspace isn't a git repo — when in doubt, treat every file as
    pre-existing and skip aggressive formatting.
    """
    created: set[str] = set()
    pre_existing: set[str] = set()
    try:
        result = subprocess.run(
            ["git", "-C", workspace_path, "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            logger.debug("[lintgate] git status failed; treating all files as pre-existing.")
            return set(), set(files)

        # Parse `XY <path>` lines from porcelain output.
        # X = staged status, Y = working-tree status. `??` = untracked.
        # New file in this session shows as `?? path` (untracked) or `A path`
        # (already added). Anything else (M, R, D...) means it existed before.
        new_paths: set[str] = set()
        for line in result.stdout.splitlines():
            if len(line) < 4:
                continue
            xy = line[:2]
            path = line[3:].strip()
            # Handle renames "R old -> new"
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if xy.strip() in ("??", "A", "AM", "A "):
                new_paths.add(path)

        # Normalise both sides relative to workspace
        for f in files:
            rel = os.path.relpath(f, workspace_path) if os.path.isabs(f) else f
            rel = rel.replace(os.sep, "/")
            if rel in new_paths or f in new_paths:
                created.add(f)
            else:
                pre_existing.add(f)
        return created, pre_existing

    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        logger.debug("[lintgate] git status unavailable (%s); treating all files as pre-existing.", e)
        return set(), set(files)


# ---------------------------------------------------------------------------
# 1. Format Registry — Extension → Formatter Command
# ---------------------------------------------------------------------------

@dataclass
class FormatterSpec:
    """Specification for an auto-formatter tool."""
    command: str
    args: list[str]
    linter_command: str = ""
    linter_args: list[str] = field(default_factory=list)
    install_hint: str = ""


# Default formatter registry. Users can override via .harness_config.json.
_DEFAULT_FORMATTERS: dict[str, FormatterSpec] = {
    ".py": FormatterSpec(
        command="ruff",
        args=["format", "--quiet"],
        linter_command="ruff",
        linter_args=["check", "--fix", "--quiet"],
        install_hint="pip install ruff",
    ),
    ".pyi": FormatterSpec(
        command="ruff",
        args=["format", "--quiet"],
        install_hint="pip install ruff",
    ),
    ".go": FormatterSpec(
        command="gofmt",
        args=["-w"],
        linter_command="go",
        linter_args=["vet", "./..."],
        install_hint="Install Go from https://go.dev/dl/",
    ),
    ".rs": FormatterSpec(
        command="rustfmt",
        args=["--edition", "2021"],
        linter_command="cargo",
        linter_args=["clippy", "--fix", "--allow-dirty", "--allow-staged"],
        install_hint="rustup component add rustfmt clippy",
    ),
    ".ts": FormatterSpec(
        command="prettier",
        args=["--write"],
        linter_command="eslint",
        linter_args=["--fix", "--quiet"],
        install_hint="npm install -g prettier eslint",
    ),
    ".tsx": FormatterSpec(
        command="prettier",
        args=["--write"],
        install_hint="npm install -g prettier",
    ),
    ".js": FormatterSpec(
        command="prettier",
        args=["--write"],
        install_hint="npm install -g prettier",
    ),
    ".jsx": FormatterSpec(
        command="prettier",
        args=["--write"],
        install_hint="npm install -g prettier",
    ),
    ".css": FormatterSpec(
        command="prettier",
        args=["--write"],
        install_hint="npm install -g prettier",
    ),
    ".html": FormatterSpec(
        command="prettier",
        args=["--write"],
        install_hint="npm install -g prettier",
    ),
    ".json": FormatterSpec(
        command="prettier",
        args=["--write"],
        install_hint="npm install -g prettier",
    ),
    ".yaml": FormatterSpec(
        command="prettier",
        args=["--write"],
        install_hint="npm install -g prettier",
    ),
    ".yml": FormatterSpec(
        command="prettier",
        args=["--write"],
        install_hint="npm install -g prettier",
    ),
    ".md": FormatterSpec(
        command="prettier",
        args=["--write", "--prose-wrap", "always"],
        install_hint="npm install -g prettier",
    ),
    ".c": FormatterSpec(
        command="clang-format",
        args=["-i"],
        install_hint="apt install clang-format  # or brew install clang-format",
    ),
    ".h": FormatterSpec(
        command="clang-format",
        args=["-i"],
        install_hint="apt install clang-format",
    ),
    ".cpp": FormatterSpec(
        command="clang-format",
        args=["-i"],
        install_hint="apt install clang-format",
    ),
    ".cc": FormatterSpec(
        command="clang-format",
        args=["-i"],
        install_hint="apt install clang-format",
    ),
    ".cxx": FormatterSpec(
        command="clang-format",
        args=["-i"],
        install_hint="apt install clang-format",
    ),
    ".hpp": FormatterSpec(
        command="clang-format",
        args=["-i"],
        install_hint="apt install clang-format",
    ),
    ".java": FormatterSpec(
        command="google-java-format",
        args=["-i"],
        install_hint="Download from https://github.com/google/google-java-format/releases",
    ),
    ".dart": FormatterSpec(
        command="dart",
        args=["format"],
        linter_command="dart",
        linter_args=["analyze"],
        install_hint="Install Dart/Flutter from https://flutter.dev/docs/get-started/install",
    ),
    ".sh": FormatterSpec(
        command="shfmt",
        args=["-w"],
        linter_command="shellcheck",
        linter_args=["--severity=error"],
        install_hint="apt install shfmt shellcheck  # or brew install shfmt shellcheck",
    ),
    ".bash": FormatterSpec(
        command="shfmt",
        args=["-w"],
        install_hint="apt install shfmt",
    ),
    ".sql": FormatterSpec(
        command="sql-formatter",
        args=["--fix"],
        install_hint="npm install -g sql-formatter",
    ),
}


def get_formatter_for_file(filepath: str) -> Optional[FormatterSpec]:
    """Look up the formatter spec for a given file by extension."""
    ext = os.path.splitext(filepath)[1].lower()
    return _DEFAULT_FORMATTERS.get(ext)


def register_formatter(extension: str, spec: FormatterSpec) -> None:
    """Register or override a formatter for a given file extension."""
    _DEFAULT_FORMATTERS[extension] = spec
    logger.info("[lintgate] Registered formatter for '%s': %s", extension, spec.command)


def is_tool_available(command: str) -> bool:
    """Check if a command-line tool is available on the system PATH."""
    return shutil.which(command) is not None


# ---------------------------------------------------------------------------
# 2. LintGate Node — Deterministic Format + Lint
# ---------------------------------------------------------------------------

@dataclass
class LintGateResult:
    """Result of running the lint gate on modified files."""
    files_formatted: list[str]
    files_linted: list[str]
    format_errors: list[str]
    lint_errors: list[str]
    total_files_checked: int
    had_errors: bool


async def lintgate_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Deterministic node that runs auto-formatters on modified files.

    Executed AFTER patches are applied but BEFORE the compiler pipeline.
    Catches trivial syntax/formatting issues that would cause noisy build
    failures, avoiding expensive LLM repair loops.

    Workflow:
        1. Reads `modified_files` from state
        2. Groups files by language/extension
        3. For each group, runs the auto-formatter in-place
        4. Optionally runs linter if configured and available
        5. Records results in node_state

    Returns:
        State update dict with lintgate results in node_state.
    """
    logger.info("[lintgate_node] Starting deterministic format verification...")

    modified_files: list[str] = list(state.get("modified_files", []))
    workspace_path: str = state.get("workspace_path", os.getcwd())

    # Config: whether to format pre-existing files (default off — clobbers
    # user style in untouched regions of large files). Linting still runs
    # on all files since it's read-only.
    lintgate_cfg = state.get("lintgate_config", {}) or {}
    format_modified = bool(lintgate_cfg.get("format_modified_files", False))
    # When true, a configured formatter that isn't on PATH is surfaced as
    # a format error rather than a silent warning — so a build can't quietly
    # ship unformatted code because the operator forgot to install ruff /
    # prettier. Default off to preserve compatibility with CI environments
    # that haven't installed every formatter.
    strict_missing_formatter = bool(
        lintgate_cfg.get("strict_missing_formatter", False)
    )

    if not modified_files:
        logger.info("[lintgate_node] No modified files to check.")
        return {
            "node_state": {
                "lintgate": {
                    "checked": 0,
                    "formatted": 0,
                    "linted": 0,
                    "errors": 0,
                }
            }
        }

    # Classify files as created-this-session vs pre-existing. The formatter
    # only runs on created files by default; pre-existing files get the
    # linter (read-only) but not the rewriter.
    created_files, preexisting_files = _classify_files_by_git_status(
        modified_files, workspace_path
    )
    if preexisting_files and not format_modified:
        logger.info(
            "[lintgate_node] Skipping formatter on %d pre-existing file(s) "
            "(set lintgate.format_modified_files=true to override): %s",
            len(preexisting_files),
            sorted(preexisting_files)[:5],
        )

    # Files eligible for the rewriter
    files_to_format = (
        modified_files if format_modified else sorted(created_files)
    )

    # Group files by extension → formatter
    grouped: dict[str, list[str]] = {}
    for filepath in files_to_format:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in _DEFAULT_FORMATTERS:
            grouped.setdefault(ext, []).append(filepath)

    # Lint-only grouping for files we won't format
    lint_only_grouped: dict[str, list[str]] = {}
    for filepath in modified_files:
        if filepath in files_to_format:
            continue  # already in grouped
        ext = os.path.splitext(filepath)[1].lower()
        if ext in _DEFAULT_FORMATTERS:
            lint_only_grouped.setdefault(ext, []).append(filepath)

    # Run the web-asset scan AND the architecture inventory post-patch check
    # early so their diagnostics surface even when there are no formatters to
    # apply (e.g. a pure-HTML static site with no formatter installed).
    web_asset_diagnostics = await _run_web_asset_scan(
        workspace_path, modified_files
    )
    inventory_diagnostics = await _run_inventory_post_patch_check(
        workspace_path, state
    )

    if not grouped:
        logger.info("[lintgate_node] No registered formatters for modified file types.")
        if not web_asset_diagnostics and not inventory_diagnostics:
            return {
                "node_state": {
                    "lintgate": {
                        "checked": len(modified_files),
                        "formatted": 0,
                        "linted": 0,
                        "errors": 0,
                        "web_asset_errors": [],
                        "lint_errors": [],
                    }
                }
            }
        # Fall through with empty format/lint lists so the asset diagnostics
        # surface through the normal return path below.

    files_formatted: list[str] = []
    files_linted: list[str] = []
    format_errors: list[str] = []
    lint_errors: list[str] = []

    for ext, files in grouped.items():
        spec = _DEFAULT_FORMATTERS[ext]

        # --- Python typing-import auto-injection (pre-format) ---
        # LLM-generated code frequently uses Optional/List/Dict/etc. without
        # importing them, then trips ruff F821. Fix it deterministically
        # before format/lint runs so a stylistic miss doesn't trigger an
        # expensive LLM repair loop.
        if ext == ".py":
            for filepath in files:
                full_path = _resolve_path(filepath, workspace_path)
                if not full_path or not os.path.isfile(full_path):
                    continue
                injected = _inject_missing_typing_imports(full_path)
                if injected:
                    logger.info(
                        "[lintgate_node] Injected %d missing typing import(s) into %s",
                        injected, filepath,
                    )

        # --- Run Formatter ---
        if is_tool_available(spec.command):
            for filepath in files:
                full_path = _resolve_path(filepath, workspace_path)
                if not full_path or not os.path.isfile(full_path):
                    continue

                logger.info("[lintgate_node] Formatting %s with %s", filepath, spec.command)
                try:
                    # Use the shared kill-on-timeout helper so a wedged
                    # formatter doesn't leak a Python interpreter per file
                    # per repair iteration (audit §2.6).
                    from harness.sandbox import run_subprocess_kill_on_timeout
                    rc, stdout, stderr, _timed_out = await run_subprocess_kill_on_timeout(
                        [spec.command, *spec.args, full_path],
                        timeout=30.0,
                    )
                    if rc == 0:
                        files_formatted.append(filepath)
                        logger.debug("[lintgate_node] Formatted %s successfully.", filepath)
                    else:
                        err_msg = stderr.decode("utf-8", errors="replace").strip()
                        if err_msg:
                            format_errors.append(f"{filepath}: {err_msg[:200]}")
                            logger.warning("[lintgate_node] Format failed for %s: %s", filepath, err_msg[:200])
                except Exception as exc:
                    format_errors.append(f"{filepath}: {exc}")
                    logger.warning("[lintgate_node] Format error for %s: %s", filepath, exc)
        else:
            if strict_missing_formatter:
                # Surface as a format error so the existing diagnostic channel
                # carries it into the repair / HITL flow rather than the build
                # silently proceeding with unformatted code.
                msg = (
                    f"formatter '{spec.command}' not installed for {ext} "
                    f"files — set lintgate.strict_missing_formatter=false to "
                    f"downgrade to a warning. Install hint: {spec.install_hint}"
                )
                for filepath in files:
                    format_errors.append(f"{filepath}: {msg}")
                logger.error(
                    "[lintgate_node] strict mode: formatter '%s' missing for %s "
                    "(%d file(s)). %s",
                    spec.command, ext, len(files), spec.install_hint,
                )
            else:
                logger.warning(
                    "[lintgate_node] Skipping formatter '%s' for %s extension: "
                    "tool not installed. Patches will not be auto-formatted. "
                    "Set lintgate.strict_missing_formatter=true to fail the "
                    "build instead. %s",
                    spec.command, ext, spec.install_hint,
                )

        # --- Run Linter (optional) ---
        if spec.linter_command and is_tool_available(spec.linter_command):
            for filepath in files:
                full_path = _resolve_path(filepath, workspace_path)
                if not full_path or not os.path.isfile(full_path):
                    continue

                logger.info("[lintgate_node] Linting %s with %s", filepath, spec.linter_command)
                try:
                    # Shared kill-on-timeout helper, audit §2.6.
                    from harness.sandbox import run_subprocess_kill_on_timeout
                    rc, stdout, stderr, _timed_out = await run_subprocess_kill_on_timeout(
                        [spec.linter_command, *spec.linter_args, full_path],
                        timeout=60.0,
                    )
                    if rc == 0:
                        files_linted.append(filepath)
                        logger.debug("[lintgate_node] Linted %s successfully.", filepath)
                    else:
                        err_msg = stderr.decode("utf-8", errors="replace").strip() or stdout.decode("utf-8", errors="replace").strip()
                        if err_msg:
                            lint_errors.append(f"{filepath}: {err_msg[:500]}")
                            logger.warning("[lintgate_node] Lint failed for %s: %s", filepath, err_msg[:500])
                except Exception as exc:
                    lint_errors.append(f"{filepath}: {exc}")
                    logger.warning("[lintgate_node] Lint error for %s: %s", filepath, exc)

    # --- Lint-only pass for pre-existing files (no formatting, read-only) ---
    for ext, files in lint_only_grouped.items():
        spec = _DEFAULT_FORMATTERS[ext]
        if not (spec.linter_command and is_tool_available(spec.linter_command)):
            continue
        for filepath in files:
            full_path = _resolve_path(filepath, workspace_path)
            if not full_path or not os.path.isfile(full_path):
                continue
            logger.info("[lintgate_node] Lint-only check on pre-existing %s (no format).", filepath)
            try:
                # Shared kill-on-timeout helper, audit §2.6.
                from harness.sandbox import run_subprocess_kill_on_timeout
                rc, stdout, stderr, _timed_out = await run_subprocess_kill_on_timeout(
                    [spec.linter_command, *spec.linter_args, full_path],
                    timeout=60.0,
                )
                if rc == 0:
                    files_linted.append(filepath)
                else:
                    err_msg = stderr.decode("utf-8", errors="replace").strip() or stdout.decode("utf-8", errors="replace").strip()
                    if err_msg:
                        lint_errors.append(f"{filepath}: {err_msg[:500]}")
                        logger.warning("[lintgate_node] Lint-only failed for %s: %s", filepath, err_msg[:500])
            except Exception as exc:
                lint_errors.append(f"{filepath}: {exc}")

    # Web-asset diagnostics from the early scan flow into lint_errors so
    # autofix R6 + the existing repair loop pick them up. Catches the class
    # of bug where the LLM emits `<link href="src/styles.css">` but never
    # writes the CSS file.
    for diag in web_asset_diagnostics:
        lint_errors.append(diag.format_compiler_style())

    # Layer 1 follow-on: post-patch existence check against the manifest
    # declared in SPEC_ARCHITECTURE.md. Catches "plan said style.css,
    # patcher didn't write it" without any new graph node. The diagnostics
    # were collected by the early-pass; flatten them into lint_errors here
    # so the existing repair loop picks them up.
    for diag in inventory_diagnostics:
        lint_errors.append(diag.format_compiler_style())

    total_checked = len(modified_files)
    total_formatted = len(files_formatted)
    total_linted = len(files_linted)
    total_errors = len(format_errors) + len(lint_errors)

    logger.info(
        "[lintgate_node] Complete: %d files checked, %d formatted, %d linted, %d errors.",
        total_checked, total_formatted, total_linted, total_errors,
    )

    # Build a status message for the conversation
    messages = list(state.get("messages", []))
    status_parts: list[str] = ["[LintGate] Pre-build verification complete:"]
    if total_formatted > 0:
        status_parts.append(f"  - Formatted {total_formatted} file(s): {', '.join(os.path.basename(f) for f in files_formatted[:5])}")
        if len(files_formatted) > 5:
            status_parts.append(f"    ... and {len(files_formatted) - 5} more")
    if total_linted > 0:
        status_parts.append(f"  - Linted {total_linted} file(s)")
    if format_errors:
        status_parts.append(f"  - Format errors ({len(format_errors)}):")
        for err in format_errors[:3]:
            status_parts.append(f"    {err}")
        if len(format_errors) > 3:
            status_parts.append(f"    ... and {len(format_errors) - 3} more")
    if web_asset_diagnostics:
        status_parts.append(
            f"  - Unresolved asset references ({len(web_asset_diagnostics)}):"
        )
        for diag in web_asset_diagnostics[:5]:
            status_parts.append(f"    {diag.format_compiler_style()}")
        if len(web_asset_diagnostics) > 5:
            status_parts.append(
                f"    ... and {len(web_asset_diagnostics) - 5} more"
            )
    if total_formatted == 0 and total_linted == 0:
        status_parts.append("  No formatters triggered (tools not installed or no matching file types).")
    messages.append({"role": "system", "content": "\n".join(status_parts)})

    # Audit §6.12: lintgate's auto-formatters mutated `files_formatted`
    # on disk. The patcher's idempotency check compares against the
    # hashes recorded when the LLM last read each file — so without a
    # refresh, the next REPLACE_BLOCK on a formatted file would mis-
    # match. Re-hash the formatted files and update files_seen_by_llm
    # in node_state so the downstream patch node sees current content.
    refreshed_hashes: dict[str, str] = {}
    if files_formatted:
        try:
            import hashlib as _hashlib
            for filepath in files_formatted:
                full_path = _resolve_path(filepath, workspace_path)
                if not full_path or not os.path.isfile(full_path):
                    continue
                try:
                    with open(full_path, "rb") as _fh:
                        refreshed_hashes[filepath] = _hashlib.sha256(
                            _fh.read()
                        ).hexdigest()
                except OSError:
                    continue
        except Exception as exc:  # noqa: BLE001 — best-effort hash refresh
            logger.debug("[lintgate] hash refresh failed: %s", exc)
    # Merge into the existing files_seen_by_llm dict so the patcher's
    # idempotency check sees the post-format hashes on the next pass.
    prior_seen = (state.get("node_state", {}) or {}).get("files_seen_by_llm") or {}
    merged_seen = {**prior_seen, **refreshed_hashes}

    return {
        "messages": messages,
        "node_state": {
            "lintgate": {
                "checked": total_checked,
                "formatted": total_formatted,
                "linted": total_linted,
                "errors": total_errors,
                "files_formatted": files_formatted,
                "files_linted": files_linted,
                "format_errors": format_errors,
                "lint_errors": lint_errors,
                "web_asset_errors": [
                    {
                        "referring_file": d.referring_file,
                        "line": d.line,
                        "column": d.column,
                        "raw_reference": d.raw_reference,
                        "resolved_path": d.resolved_path,
                        "suggested_path": d.suggested_path,
                    }
                    for d in web_asset_diagnostics
                ],
            },
            "files_seen_by_llm": merged_seen,
        }
    }


async def _run_inventory_post_patch_check(
    workspace_path: str,
    state: dict[str, Any],
) -> list[Any]:
    """Post-patch existence check: every file in the architecture inventory
    must exist on disk.

    Reads the fenced JSON inventory block from ``docs/SPEC_ARCHITECTURE.md``
    (or wherever ``state["spec_architecture_path"]`` points) and asserts
    every listed path was actually written. Catches the upstream class of
    bug where planning forgot to generate a file declared in the
    architecture — e.g. ticktaktoe's missing style.css.

    Silently no-ops when:
      * No architecture spec exists yet (first-pass project).
      * The spec contains no JSON inventory block (legacy spec format).
      * Workspace is backend-only (no html tag).

    Returns a list of InventoryDiagnostic (MISSING_FROM_DISK).
    """
    try:
        from harness.architecture_inventory import (
            check_files_on_disk,
            parse_inventory,
        )
        from harness.impact import _detect_workspace_stack
    except ImportError as exc:
        logger.debug("[lintgate_node] inventory check unavailable: %s", exc)
        return []

    try:
        tags = _detect_workspace_stack(workspace_path) or set()
    except Exception:
        return []
    if "html" not in tags:
        return []

    arch_path = state.get("spec_architecture_path") or os.path.join(
        workspace_path, "docs", "SPEC_ARCHITECTURE.md"
    )
    if not arch_path or not os.path.isfile(arch_path):
        return []
    try:
        with open(arch_path, "r", encoding="utf-8") as fh:
            spec_md = fh.read()
    except OSError as exc:
        logger.debug("[lintgate_node] inventory read failed: %s", exc)
        return []

    parsed = parse_inventory(spec_md)
    if not parsed.ok:
        logger.debug(
            "[lintgate_node] inventory block absent or malformed: %s",
            parsed.error,
        )
        return []
    return check_files_on_disk(parsed.files, workspace_path)


async def _run_web_asset_scan(
    workspace_path: str,
    modified_files: list[str],
) -> list[Any]:
    """Run the static asset reference scanner if the workspace ships HTML.

    Returns a list of AssetRefDiagnostic. Empty for non-web workspaces or
    when no asset references are broken. Gated on the same `html` workspace
    tag the skill loader uses, so backend-only projects pay nothing.
    """
    try:
        from harness.impact import _detect_workspace_stack
        from harness.web_asset_scan import scan_web_asset_references
    except ImportError as exc:
        logger.debug("[lintgate_node] web asset scan unavailable: %s", exc)
        return []

    try:
        tags = _detect_workspace_stack(workspace_path) or set()
    except Exception as exc:
        logger.debug("[lintgate_node] workspace stack detection failed: %s", exc)
        return []

    if "html" not in tags:
        return []

    try:
        return scan_web_asset_references(workspace_path, modified_files)
    except Exception as exc:
        logger.warning("[lintgate_node] web asset scan errored: %s", exc)
        return []


def _resolve_path(filepath: str, workspace_path: str) -> Optional[str]:
    """
    Resolve a filepath against the workspace with strict boundary enforcement.

    Rejects absolute paths that escape the workspace and any relative path
    that would traverse out via ``..`` or symlinks. Absolute paths that point
    inside the workspace are accepted (converted to workspace-relative form
    before delegating to ``trust.safe_resolve``).
    """
    from harness.trust import safe_resolve

    if not filepath:
        return None

    if os.path.isabs(filepath):
        # Convert an absolute path that lives inside the workspace into a
        # workspace-relative form. Anything else is rejected as a boundary
        # escape (previous behaviour silently accepted /etc/passwd).
        try:
            workspace_real = os.path.realpath(workspace_path)
            filepath_real = os.path.realpath(filepath)
            rel = os.path.relpath(filepath_real, workspace_real)
        except (OSError, ValueError):
            return None
        if rel == ".." or rel.startswith(".." + os.sep):
            return None
        filepath = rel

    try:
        resolved = safe_resolve(workspace_path, filepath)
    except ValueError:
        return None
    return resolved if os.path.exists(resolved) else None


# ---------------------------------------------------------------------------
# 3. Format Registry Factory from Config
# ---------------------------------------------------------------------------

def register_formatters_from_config(config_dict: dict[str, Any]) -> int:
    """
    Register custom formatters from the 'lintgate.formatters' section.

    Expected format:
        {
          "lintgate": {
            "formatters": {
              ".py": {
                "command": "ruff",
                "args": ["format", "--quiet"],
                "linter_command": "ruff",
                "linter_args": ["check", "--fix", "--quiet"],
                "install_hint": "pip install ruff"
              }
            }
          }
        }

    Args:
        config_dict: Merged configuration dictionary.

    Returns:
        Number of formatters registered.
    """
    lg_cfg = config_dict.get("lintgate", {})
    custom_formatters = lg_cfg.get("formatters", {})
    count = 0
    for ext, spec_dict in custom_formatters.items():
        if not isinstance(spec_dict, dict):
            continue
        try:
            spec = FormatterSpec(
                command=spec_dict.get("command", ""),
                args=spec_dict.get("args", []),
                linter_command=spec_dict.get("linter_command", ""),
                linter_args=spec_dict.get("linter_args", []),
                install_hint=spec_dict.get("install_hint", ""),
            )
            if spec.command:
                register_formatter(ext, spec)
                count += 1
        except Exception as exc:
            logger.warning("[lintgate] Failed to register formatter for '%s': %s", ext, exc)
    if count > 0:
        logger.info("[lintgate] Registered %d custom formatter(s) from config.", count)
    return count