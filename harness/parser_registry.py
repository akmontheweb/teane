"""
Language-specific diagnostic parser plugin registry.

This module provides the canonical registry for compiler diagnostic parsers
that extend the sandbox execution engine. New language parsers can be registered
without modifying the sandbox or graph engine.

The parsers in this module complement the built-in parsers in sandbox.py.
Callers can use register_parser() / get_parser() from either module
to register and look up diagnostic parsers at runtime.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from harness.sandbox import BaseLanguageParser, DiagnosticObject

logger = logging.getLogger(__name__)


# CSI / SGR (color/style) and OSC escape sequences emitted by modern
# compilers when stdout is a TTY (cargo, rustc, gcc, clang, go test).
# Sandbox builds capture pipe output so these usually don't appear, but
# some toolchains force-color via env vars (CARGO_TERM_COLOR=always,
# CLICOLOR_FORCE=1) and break our diagnostic regexes if not stripped.
_ANSI_ESCAPE_RE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\))"
)


def _strip_ansi(text: str) -> str:
    """Remove ANSI color/style escape sequences from compiler output."""
    return _ANSI_ESCAPE_RE.sub("", text)


# ---------------------------------------------------------------------------
# Built-in Language Parsers
# ---------------------------------------------------------------------------

class RustParser(BaseLanguageParser):
    """
    Parses Rust compiler JSON diagnostic output (--error-format=json).

    Extracts one JSON object per line; filters to compiler-message diagnostic
    entries containing spans with file/line/column/message/code information.
    """

    @staticmethod
    def parse_diagnostics(raw_output: str) -> list[DiagnosticObject]:
        diagnostics: list[DiagnosticObject] = []
        for line in raw_output.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            reason = obj.get("reason", "")
            if reason != "compiler-message":
                continue
            msg_data = obj.get("message", {})
            spans = msg_data.get("spans", [])
            primary_span = spans[0] if spans else {}
            diagnostics.append(DiagnosticObject(
                file=primary_span.get("file_name", ""),
                line=primary_span.get("line_start", 0),
                column=primary_span.get("column_start", 0),
                severity=msg_data.get("level", "error"),
                error_code=msg_data.get("code", ""),
                message=msg_data.get("message", ""),
                semantic_context=msg_data.get("rendered", ""),
            ))
        return diagnostics


class GccClangParser(BaseLanguageParser):
    """
    Parses GCC/Clang JSON diagnostic output (-fdiagnostics-format=json).

    Expects one JSON array per diagnostic line. Each array contains diagnostic
    items with location (caret) and message fields.
    """

    @staticmethod
    def parse_diagnostics(raw_output: str) -> list[DiagnosticObject]:
        diagnostics: list[DiagnosticObject] = []
        for line in raw_output.splitlines():
            line = line.strip()
            if not line.startswith("[") or not line.endswith("]"):
                continue
            try:
                items = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        loc = item.get("locations", [{}])[0]
                        diagnostics.append(DiagnosticObject(
                            file=loc.get("caret", {}).get("file", ""),
                            line=loc.get("caret", {}).get("line", 0),
                            column=loc.get("caret", {}).get("column", 0),
                            severity="error" if item.get("kind") == "error" else "warning",
                            error_code=str(item.get("option", "")),
                            message=item.get("message", ""),
                            semantic_context="",
                        ))
        return diagnostics


class GoParser(BaseLanguageParser):
    """
    Parses Go compiler output in the standard format:
        path/to/file.go:line:column: message
    """

    _PATTERN = re.compile(r'^(.+\.go):(\d+):(\d+):\s+(.+)$')

    @staticmethod
    def parse_diagnostics(raw_output: str) -> list[DiagnosticObject]:
        diagnostics: list[DiagnosticObject] = []
        for line in raw_output.splitlines():
            match = GoParser._PATTERN.match(line.strip())
            if match:
                diagnostics.append(DiagnosticObject(
                    file=match.group(1),
                    line=int(match.group(2)),
                    column=int(match.group(3)),
                    severity="error",
                    error_code="",
                    message=match.group(4),
                    semantic_context="",
                ))
        return diagnostics


class PythonParser(BaseLanguageParser):
    """
    Parses Python traceback and error output.

    Extracts file/line/file information from standard Python exception output
    including SyntaxError, ImportError, ModuleNotFoundError, etc.
    """

    _TRACEBACK_PATTERN = re.compile(
        r'^\s*File\s+"(.+?)",\s+line\s+(\d+)(?:,\s+in\s+(.+))?\s*$',
        re.MULTILINE,
    )
    _ERROR_PATTERN = re.compile(
        r'^(\w+(?:Error|Warning|Exception)):\s*(.+)$',
        re.MULTILINE,
    )

    @staticmethod
    def parse_diagnostics(raw_output: str) -> list[DiagnosticObject]:
        diagnostics: list[DiagnosticObject] = []
        lines = raw_output.splitlines()

        # Find the last traceback entry (closest to the actual error)
        last_file = ""
        last_line = 0
        last_function = ""
        for line in lines:
            match = PythonParser._TRACEBACK_PATTERN.match(line.strip())
            if match:
                last_file = match.group(1)
                last_line = int(match.group(2))
                last_function = match.group(3) or ""

        # Find the error message
        error_type = "Exception"
        error_msg = ""
        for line in lines:
            match = PythonParser._ERROR_PATTERN.match(line.strip())
            if match:
                error_type = match.group(1)
                error_msg = match.group(2)
                break

        if last_file:
            diagnostics.append(DiagnosticObject(
                file=last_file,
                line=last_line,
                column=0,
                severity="error",
                error_code=error_type,
                message=error_msg or last_function,
                semantic_context="",
            ))

        return diagnostics


class GenericParser(BaseLanguageParser):
    """
    Fallback generic diagnostic parser for compilers without structured output.
    Matches the common compiler format:
        /path/to/file.ext:line:column: error: message
    """

    _PATTERN = re.compile(
        r'^(.+?):(\d+):(\d+):\s+(error|warning):\s+(.+)$',
        re.IGNORECASE,
    )

    @staticmethod
    def parse_diagnostics(raw_output: str) -> list[DiagnosticObject]:
        diagnostics: list[DiagnosticObject] = []
        for line in raw_output.splitlines():
            match = GenericParser._PATTERN.match(line.strip())
            if match:
                filepath = match.group(1)
                diagnostics.append(DiagnosticObject(
                    file=filepath,
                    line=int(match.group(2)),
                    column=int(match.group(3)),
                    severity=match.group(4).lower(),
                    error_code="",
                    message=match.group(5),
                    semantic_context="",
                ))
        return diagnostics


# ---------------------------------------------------------------------------
# Parser Registry
# ---------------------------------------------------------------------------

# Maps compiler names to their parser classes
_PARSER_REGISTRY: dict[str, type[BaseLanguageParser]] = {
    "rustc": RustParser,
    "cargo": RustParser,
    "gcc": GccClangParser,
    "g++": GccClangParser,
    "clang": GccClangParser,
    "clang++": GccClangParser,
    "go": GoParser,
    "python": PythonParser,
    "pytest": PythonParser,
}

# Maps file extensions to parser classes
_EXTENSION_PARSER_MAP: dict[str, type[BaseLanguageParser]] = {
    ".rs": RustParser,
    ".c": GccClangParser,
    ".cpp": GccClangParser,
    ".cc": GccClangParser,
    ".cxx": GccClangParser,
    ".h": GccClangParser,
    ".hpp": GccClangParser,
    ".go": GoParser,
    ".py": PythonParser,
    ".pyi": PythonParser,
}


def register_parser(compiler_name: str, parser_cls: type[BaseLanguageParser]) -> None:
    """
    Register a new language parser plugin for a given compiler name.

    Args:
        compiler_name: The compiler tool name (e.g., 'rustc', 'gcc', 'swiftc').
        parser_cls: A BaseLanguageParser subclass implementing parse_diagnostics.
    """
    _PARSER_REGISTRY[compiler_name] = parser_cls
    logger.info("[parser_registry] Registered parser for '%s': %s", compiler_name, parser_cls.__name__)


def register_extension_parser(extension: str, parser_cls: type[BaseLanguageParser]) -> None:
    """
    Register a parser to be used for files with a given extension.

    Args:
        extension: File extension including dot (e.g., '.swift', '.kt').
        parser_cls: A BaseLanguageParser subclass.
    """
    _EXTENSION_PARSER_MAP[extension] = parser_cls
    logger.info("[parser_registry] Registered extension parser for '%s': %s", extension, parser_cls.__name__)


def get_parser(compiler_name: str) -> Optional[type[BaseLanguageParser]]:
    """
    Look up a registered parser by compiler name.

    Args:
        compiler_name: The compiler name (e.g., 'rustc', 'cargo').

    Returns:
        The parser class, or None if no parser is registered.
    """
    return _PARSER_REGISTRY.get(compiler_name)


def get_parser_for_extension(extension: str) -> Optional[type[BaseLanguageParser]]:
    """
    Look up a registered parser by file extension.

    Args:
        extension: File extension including dot (e.g., '.rs', '.go').

    Returns:
        The parser class, or None if no parser is registered.
    """
    return _EXTENSION_PARSER_MAP.get(extension)


def list_registered_parsers() -> dict[str, list[str]]:
    """
    Return a summary of all registered parsers.

    Returns:
        A dict with 'compiler' and 'extension' keys containing sorted lists of names.
    """
    return {
        "compiler": sorted(_PARSER_REGISTRY.keys()),
        "extension": sorted(_EXTENSION_PARSER_MAP.keys()),
    }


# ---------------------------------------------------------------------------
# Convenience: Detect compiler and return best parser
# ---------------------------------------------------------------------------

def detect_and_parse(
    raw_output: str,
    build_command: str = "",
    workspace_path: str = "",
    file_path: str = "",
) -> list[DiagnosticObject]:
    """
    Auto-detect the appropriate parser and extract structured diagnostics.

    Detection order:
        1. By compiler name inferred from build_command (e.g., 'cargo', 'gcc')
        2. By file extension (e.g., '.rs' → rustc, '.py' → python)
        3. Falls back to GenericParser

    Args:
        raw_output: The complete stdout+stderr from the build tool.
        build_command: The build command string (for compiler detection).
        workspace_path: Absolute workspace root (for resolving relative paths).
        file_path: Optional specific file path (for extension-based detection).

    Returns:
        A list of DiagnosticObject instances.
    """
    # Strip ANSI color escape sequences once at the entry point so every
    # downstream regex sees clean text. Modern compilers emit \x1b[31m...
    # when CARGO_TERM_COLOR=always or CLICOLOR_FORCE=1 is set, which
    # would otherwise silently drop every diagnostic.
    raw_output = _strip_ansi(raw_output)

    # Try compiler detection first
    if build_command:
        cmd_lower = build_command.lower()
        for compiler_name in _PARSER_REGISTRY:
            if compiler_name in cmd_lower:
                parser_cls = _PARSER_REGISTRY[compiler_name]
                logger.debug("[parser_registry] Detected compiler '%s' from build command.", compiler_name)
                return parser_cls.parse_diagnostics(raw_output)

    # Try file extension detection
    if file_path:
        ext = os.path.splitext(file_path)[1].lower()
        parser_cls = _EXTENSION_PARSER_MAP.get(ext)
        if parser_cls is not None:
            logger.debug("[parser_registry] Using extension-based parser for '%s'.", ext)
            return parser_cls.parse_diagnostics(raw_output)

    # Fall back to generic parser
    logger.debug("[parser_registry] No specific parser detected. Using GenericParser.")
    return GenericParser.parse_diagnostics(raw_output)
