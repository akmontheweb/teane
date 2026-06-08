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

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


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

    # Group files by extension → formatter
    grouped: dict[str, list[str]] = {}
    for filepath in modified_files:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in _DEFAULT_FORMATTERS:
            grouped.setdefault(ext, []).append(filepath)

    if not grouped:
        logger.info("[lintgate_node] No registered formatters for modified file types.")
        return {
            "node_state": {
                "lintgate": {
                    "checked": len(modified_files),
                    "formatted": 0,
                    "linted": 0,
                    "errors": 0,
                }
            }
        }

    files_formatted: list[str] = []
    files_linted: list[str] = []
    format_errors: list[str] = []
    lint_errors: list[str] = []

    for ext, files in grouped.items():
        spec = _DEFAULT_FORMATTERS[ext]

        # --- Run Formatter ---
        if is_tool_available(spec.command):
            for filepath in files:
                full_path = _resolve_path(filepath, workspace_path)
                if not full_path or not os.path.isfile(full_path):
                    continue

                logger.info("[lintgate_node] Formatting %s with %s", filepath, spec.command)
                try:
                    proc = await asyncio.create_subprocess_exec(
                        spec.command,
                        *spec.args,
                        full_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=30.0
                    )
                    if proc.returncode == 0:
                        files_formatted.append(filepath)
                        logger.debug("[lintgate_node] Formatted %s successfully.", filepath)
                    else:
                        err_msg = stderr.decode("utf-8", errors="replace").strip()
                        if err_msg:
                            format_errors.append(f"{filepath}: {err_msg[:200]}")
                            logger.warning("[lintgate_node] Format failed for %s: %s", filepath, err_msg[:200])
                except asyncio.TimeoutError:
                    format_errors.append(f"{filepath}: Formatter timed out")
                    logger.warning("[lintgate_node] Formatter timed out for %s", filepath)
                except Exception as exc:
                    format_errors.append(f"{filepath}: {exc}")
                    logger.warning("[lintgate_node] Format error for %s: %s", filepath, exc)
        else:
            logger.warning(
                "[lintgate_node] Skipping formatter '%s' for %s extension: "
                "tool not installed. Patches will not be auto-formatted. %s",
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
                    proc = await asyncio.create_subprocess_exec(
                        spec.linter_command,
                        *spec.linter_args,
                        full_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=60.0
                    )
                    if proc.returncode == 0:
                        files_linted.append(filepath)
                        logger.debug("[lintgate_node] Linted %s successfully.", filepath)
                    else:
                        err_msg = stderr.decode("utf-8", errors="replace").strip() or stdout.decode("utf-8", errors="replace").strip()
                        if err_msg:
                            lint_errors.append(f"{filepath}: {err_msg[:500]}")
                            logger.warning("[lintgate_node] Lint failed for %s: %s", filepath, err_msg[:500])
                except asyncio.TimeoutError:
                    lint_errors.append(f"{filepath}: Linter timed out")
                    logger.warning("[lintgate_node] Linter timed out for %s", filepath)
                except Exception as exc:
                    lint_errors.append(f"{filepath}: {exc}")
                    logger.warning("[lintgate_node] Lint error for %s: %s", filepath, exc)

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
    if total_formatted == 0 and total_linted == 0:
        status_parts.append("  No formatters triggered (tools not installed or no matching file types).")
    messages.append({"role": "system", "content": "\n".join(status_parts)})

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
            }
        }
    }


def _resolve_path(filepath: str, workspace_path: str) -> Optional[str]:
    """Resolve a filepath against the workspace."""
    if os.path.isabs(filepath):
        return filepath if os.path.exists(filepath) else None
    full = os.path.join(workspace_path, filepath)
    return full if os.path.exists(full) else None


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