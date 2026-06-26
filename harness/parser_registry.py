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

import logging
import os
import re
from typing import Optional

from harness.sandbox import BaseLanguageParser, DiagnosticObject

logger = logging.getLogger(__name__)


# Lines that look like "diagnostic context" attached to a previous error.
# Matched after we've already located a primary diagnostic; whatever follows
# until the next primary line / blank break / unrelated stanza is folded
# into the primary's semantic_context field so downstream repair prompts
# see the full annotated source snippet alongside the headline message.
#
# Patterns covered:
#   - Indented continuation lines (compilers emit leading whitespace for
#     source-code carets, snippets, and tree-style hint output)
#   - `note: ...`, `help: ...`, `info:`, `hint:` — javac/tsc sub-notes
#   - Bare caret / tilde underline rows
_CONTEXT_LINE_RE = re.compile(
    r"^("
    r"\s+\S"                               # any indented continuation
    r"|note:|help:|info:|hint:"            # named sub-notes (no leading ws)
    r"|\s*[\^~]+\s*$"                      # bare caret / tilde underline
    r")"
)


def _collect_context_lines(
    lines: list[str],
    start_idx: int,
    primary_re: re.Pattern[str],
    max_context_lines: int = 12,
) -> tuple[str, int]:
    """
    Walk forward from ``start_idx`` collecting attached context lines until
    we hit either (a) the next primary-diagnostic line, (b) a fully blank
    line followed by un-indented unrelated content, or (c) ``max_context_lines``
    accumulated lines. Returns ``(joined_context, next_idx_to_resume_from)``.
    """
    collected: list[str] = []
    i = start_idx
    seen_blank = False
    while i < len(lines) and len(collected) < max_context_lines:
        raw = lines[i]
        line = raw.rstrip()

        # Another primary diagnostic terminates context — let the outer loop
        # pick it up.
        if primary_re.match(line.strip()):
            break

        if not line.strip():
            # One blank line is fine (separates note: blocks from the
            # caret), two in a row ends context.
            if seen_blank:
                break
            seen_blank = True
            i += 1
            continue
        seen_blank = False

        if _CONTEXT_LINE_RE.match(line):
            collected.append(line)
            i += 1
            continue
        # Un-indented, non-matching line → end of this diagnostic's context.
        break
    return "\n".join(collected), i


# CSI / SGR (color/style) and OSC escape sequences emitted by modern
# compilers when stdout is a TTY. Sandbox builds capture pipe output so
# these usually don't appear, but some toolchains force-color via env vars
# (e.g. CLICOLOR_FORCE=1) and break our diagnostic regexes if not stripped.
_ANSI_ESCAPE_RE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\))"
)


_ANSI_INPUT_CAP_BYTES = 2 * 1024 * 1024  # 2 MB


def _strip_ansi(text: str) -> str:
    """Remove ANSI color/style escape sequences from compiler output.

    Caps input at 2 MB before stripping. The OSC branch in the regex
    matches ``\\x1b]<body>(\\x07|\\x1b\\\\)``; on a multi-megabyte stdout
    with an unterminated OSC, backtracking can run pathologically.
    Compiler diagnostics never need more than a few hundred KB of
    output to be parsed, so capping is safe.
    """
    if len(text) > _ANSI_INPUT_CAP_BYTES:
        # Keep the tail (where the most recent diagnostics live) rather
        # than the head (build banners).
        text = text[-_ANSI_INPUT_CAP_BYTES:]
    return _ANSI_ESCAPE_RE.sub("", text)


# ---------------------------------------------------------------------------
# Built-in Language Parsers
# ---------------------------------------------------------------------------

class PythonParser(BaseLanguageParser):
    """
    Parses Python traceback and error output.

    Handles three distinct shapes:

    1. Standard CPython traceback (any unhandled exception):
           Traceback (most recent call last):
             File "/path/x.py", line 42, in func
           ModuleNotFoundError: No module named 'foo'

    2. Pytest collection error (exit code 4) — pytest swaps the standard
       traceback for its own short form and prefixes error lines with ``E``:
           ImportError while loading conftest '/path/tests/conftest.py'.
           tests/conftest.py:1: in <module>
               import foo
           E   ModuleNotFoundError: No module named 'foo'

    3. Pytest test failure (exit code 1) — short summary at the end:
           tests/test_x.py:3: AssertionError
           FAILED tests/test_x.py::test_one - assert 1 == 2
           ERROR  tests/conftest.py - ModuleNotFoundError: No module named 'foo'

    Without shapes 2 and 3, every pytest failure feeds the repair loop
    zero diagnostics, which causes the LLM to repair blind.
    """

    _TRACEBACK_PATTERN = re.compile(
        r'^\s*File\s+"(.+?)",\s+line\s+(\d+)(?:,\s+in\s+(.+))?\s*$',
        re.MULTILINE,
    )
    _ERROR_PATTERN = re.compile(
        r'^(\w+(?:Error|Warning|Exception)):\s*(.+)$',
        re.MULTILINE,
    )

    # Pytest frame: "tests/conftest.py:1: in <module>"
    # Distinct from generic "file:line:col: error: msg" because the tail is
    # "in <name>" (no column, no "error:"/"warning:" keyword).
    _PYTEST_FRAME_PATTERN = re.compile(
        r'^(?P<file>[^\s:][^:]*?\.py):(?P<line>\d+):\s+in\s+(?P<func>.+)$'
    )
    # Pytest terminal line: "tests/test_x.py:3: AssertionError"
    # File:line followed by the bare exception type — emitted at the bottom
    # of pytest's per-failure block.
    _PYTEST_FAIL_LINE_PATTERN = re.compile(
        r'^(?P<file>[^\s:][^:]*?\.py):(?P<line>\d+):\s+'
        r'(?P<type>\w+(?:Error|Warning|Exception))\s*$'
    )
    # Pytest "E   " line — error name and message at the deepest level.
    _PYTEST_E_PATTERN = re.compile(
        r'^E\s+(?P<type>\w+(?:Error|Warning|Exception)):\s*(?P<msg>.+)$'
    )
    # Pytest "short test summary info" lines at run end:
    #   FAILED tests/foo.py::test_x - AssertionError: assert ...
    #   ERROR  tests/conftest.py - ModuleNotFoundError: No module named 'x'
    # ERROR rows omit the "::test" suffix (collection-time failures).
    _PYTEST_SUMMARY_PATTERN = re.compile(
        r'^(?P<kind>FAILED|ERROR)\s+(?P<file>[^\s:]+\.py)(?:::\S+)?\s+-\s+'
        r'(?P<rest>.+)$'
    )
    # Conftest header pytest prints before the short frame:
    #   "ImportError while loading conftest '/.../tests/conftest.py'."
    _CONFTEST_HEADER_PATTERN = re.compile(
        r"^(?P<type>\w+(?:Error|Exception))\s+while loading conftest\s+'(?P<file>.+?)'\.\s*$"
    )

    # Stdlib and virtual-env path fragments that should be deprioritised
    # when searching for the most useful traceback frame.
    _STDLIB_FRAGMENTS = (
        os.sep + "lib" + os.sep + "python",
        os.sep + "site-packages" + os.sep,
        "<frozen",
        "<string>",
    )

    @classmethod
    def _is_user_frame(cls, filepath: str) -> bool:
        """True when the traceback file looks like user/project code."""
        return not any(frag in filepath for frag in cls._STDLIB_FRAGMENTS)

    @staticmethod
    def _parse_standard_traceback(lines: list[str]) -> Optional[DiagnosticObject]:
        """Extract a diagnostic from a CPython ``Traceback ...`` block."""
        frames: list[tuple[str, int, str]] = []
        for line in lines:
            match = PythonParser._TRACEBACK_PATTERN.match(line.strip())
            if match:
                frames.append((match.group(1), int(match.group(2)), match.group(3) or ""))
        if not frames:
            return None

        # Prefer the innermost user-code frame; fall back to the last frame.
        best_file, best_line, best_func = "", 0, ""
        for filepath, lineno, func in reversed(frames):
            if PythonParser._is_user_frame(filepath):
                best_file, best_line, best_func = filepath, lineno, func
                break
        if not best_file:
            best_file, best_line, best_func = frames[-1]

        error_type = "Exception"
        error_msg = ""
        for line in lines:
            match = PythonParser._ERROR_PATTERN.match(line.strip())
            if match:
                error_type = match.group(1)
                error_msg = match.group(2)
                break

        return DiagnosticObject(
            file=best_file,
            line=best_line,
            column=0,
            severity="error",
            error_code=error_type,
            message=error_msg or best_func,
            semantic_context="",
        )

    @staticmethod
    def _parse_pytest_failure_block(lines: list[str]) -> list[DiagnosticObject]:
        """Walk pytest's per-failure short traceback layout.

        Pytest emits, per failure:
            <file>:<line>: in <func>           (one or more frames)
            E   <ErrorType>: <message>          (the actual error)
            ...
            <file>:<line>: <ErrorType>          (terminal "fail line", optional)

        We pair each ``E   ...`` line with the *nearest preceding* frame line
        in the same stanza — that's the user frame the assertion came from.
        Also catches the conftest-load header so the conftest path is
        attached even when pytest's frame is below the header.

        For plain ``assert`` failures pytest emits no ``file:line: in <func>``
        frame line at all (it inlines the source instead), so the E-line ends
        up unattached. We then fall back to the terminal ``file:line:
        ErrorType`` line and attach the last-seen E-message + the
        accompanying ``>`` marker line (the failing source). Without this
        fallback the diagnostic message collapses to just ``AssertionError``
        and the repair LLM is asked to fix a failure whose body it has never
        seen (session 2d0164f0).
        """
        diags: list[DiagnosticObject] = []
        last_frame: Optional[tuple[str, int]] = None
        conftest_path: Optional[str] = None
        # Buffer the most recent E-line message and the failing ``>`` source
        # line so a terminal fail-line that lacks its own message body can
        # attach them.
        pending_e: Optional[tuple[str, str]] = None  # (error_type, message)
        pending_failing_src: Optional[str] = None

        for line in lines:
            stripped = line.rstrip()
            bare = stripped.strip()

            header = PythonParser._CONFTEST_HEADER_PATTERN.match(bare)
            if header:
                conftest_path = header.group("file")
                continue

            # Pytest prefixes the failing source line with "> " in the short
            # traceback. Remember it so we can put it in semantic_context.
            if stripped.startswith(">"):
                pending_failing_src = stripped[1:].strip()
                continue

            frame = PythonParser._PYTEST_FRAME_PATTERN.match(bare)
            if frame:
                last_frame = (frame.group("file"), int(frame.group("line")))
                continue

            # Terminal "file:line: ErrorType" (no message, no E-line follow-up)
            term = PythonParser._PYTEST_FAIL_LINE_PATTERN.match(bare)
            if term:
                term_type = term.group("type")
                # Prefer the pending E-line body when it matches the terminal
                # error type — that's the actual assertion text the LLM needs.
                msg = term_type
                ctx = ""
                if pending_e is not None and pending_e[0] == term_type:
                    msg = f"{term_type}: {pending_e[1]}"
                if pending_failing_src:
                    ctx = f"failing source: {pending_failing_src}"
                diags.append(DiagnosticObject(
                    file=term.group("file"),
                    line=int(term.group("line")),
                    column=0,
                    severity="error",
                    error_code=term_type,
                    message=msg,
                    semantic_context=ctx,
                ))
                last_frame = None
                pending_e = None
                pending_failing_src = None
                continue

            e_line = PythonParser._PYTEST_E_PATTERN.match(stripped)
            if e_line:
                # Always remember it for the terminal-line fallback above.
                pending_e = (e_line.group("type"), e_line.group("msg"))
                if last_frame is not None:
                    file_path, lineno = last_frame
                elif conftest_path is not None:
                    file_path, lineno = conftest_path, 0
                else:
                    # No frame yet — wait for the terminal fail line to pair us.
                    continue
                ctx = (
                    f"failing source: {pending_failing_src}"
                    if pending_failing_src else ""
                )
                diags.append(DiagnosticObject(
                    file=file_path,
                    line=lineno,
                    column=0,
                    severity="error",
                    error_code=e_line.group("type"),
                    message=f"{e_line.group('type')}: {e_line.group('msg')}",
                    semantic_context=ctx,
                ))
                last_frame = None
                conftest_path = None
                pending_e = None
                pending_failing_src = None
        return diags

    @staticmethod
    def _parse_pytest_summary(lines: list[str]) -> list[DiagnosticObject]:
        """Extract one diagnostic per "FAILED ..." / "ERROR ..." summary row."""
        diags: list[DiagnosticObject] = []
        for line in lines:
            m = PythonParser._PYTEST_SUMMARY_PATTERN.match(line.strip())
            if not m:
                continue
            rest = m.group("rest").strip()
            err_match = PythonParser._ERROR_PATTERN.match(rest)
            if err_match:
                error_type = err_match.group(1)
                message = err_match.group(2)
            else:
                # Bare-assert form ("assert 1 == 2") with no ErrorType prefix.
                error_type = "AssertionError" if m.group("kind") == "FAILED" else "Error"
                message = rest
            diags.append(DiagnosticObject(
                file=m.group("file"),
                line=0,
                column=0,
                severity="error",
                error_code=error_type,
                message=message,
                semantic_context="",
            ))
        return diags

    @staticmethod
    def _dedup(diags: list[DiagnosticObject]) -> list[DiagnosticObject]:
        """Merge duplicate (file, line, error_code) entries; keep richer ones.

        Summary lines lack a line number (line=0) while per-failure blocks
        carry the exact line. When both surface the same failure, we keep
        the one with the line number and prefer the message that isn't just
        the error type repeated.
        """
        by_key: dict[tuple[str, str], DiagnosticObject] = {}
        for d in diags:
            key = (d.file, d.error_code)
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = d
                continue
            # Prefer the entry with a concrete line number.
            if d.line and not existing.line:
                by_key[key] = d
            elif d.line == existing.line and len(d.message) > len(existing.message):
                by_key[key] = d
        return list(by_key.values())

    @staticmethod
    def parse_diagnostics(raw_output: str) -> list[DiagnosticObject]:
        lines = raw_output.splitlines()
        diagnostics: list[DiagnosticObject] = []

        standard = PythonParser._parse_standard_traceback(lines)
        if standard is not None:
            diagnostics.append(standard)

        diagnostics.extend(PythonParser._parse_pytest_failure_block(lines))
        diagnostics.extend(PythonParser._parse_pytest_summary(lines))

        return PythonParser._dedup(diagnostics)


class JavaParser(BaseLanguageParser):
    """
    Parses Java compiler output from javac, Gradle, and Maven.

    Three input formats are supported (whichever ``mvn``/``gradle``/``javac``
    happens to emit):

        Maven:    [ERROR] /path/File.java:[42,17] cannot find symbol
        Gradle:   /path/File.java:42: error: cannot find symbol
        javac:    /path/File.java:42: error: cannot find symbol

    Maven and the others differ only by the optional ``[ERROR]`` prefix and
    the ``[L,C]`` bracketed coordinates.
    """

    _MAVEN_PATTERN = re.compile(
        r'^\s*(?:\[(?P<sev>ERROR|WARNING)\]\s+)?(?P<file>.+?\.java):\[(?P<line>\d+),(?P<col>\d+)\]\s+(?P<msg>.+)$'
    )
    _JAVAC_PATTERN = re.compile(
        r'^\s*(?P<file>.+?\.java):(?P<line>\d+):\s+(?P<sev>error|warning|note):\s+(?P<msg>.+)$'
    )

    @staticmethod
    def parse_diagnostics(raw_output: str) -> list[DiagnosticObject]:
        diagnostics: list[DiagnosticObject] = []
        lines = raw_output.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            m = JavaParser._MAVEN_PATTERN.match(line)
            if m:
                severity = "warning" if (m.group("sev") or "").upper() == "WARNING" else "error"
                context, next_i = _collect_context_lines(
                    lines, i + 1, JavaParser._MAVEN_PATTERN,
                )
                diagnostics.append(DiagnosticObject(
                    file=m.group("file"),
                    line=int(m.group("line")),
                    column=int(m.group("col")),
                    severity=severity,
                    error_code="",
                    message=m.group("msg"),
                    semantic_context=context,
                ))
                i = next_i
                continue

            m = JavaParser._JAVAC_PATTERN.match(line)
            if m:
                sev_raw = (m.group("sev") or "error").lower()
                severity = "warning" if sev_raw == "warning" else "error"
                context, next_i = _collect_context_lines(
                    lines, i + 1, JavaParser._JAVAC_PATTERN,
                )
                diagnostics.append(DiagnosticObject(
                    file=m.group("file"),
                    line=int(m.group("line")),
                    column=0,  # javac/Gradle omit column in this short form
                    severity=severity,
                    error_code="",
                    message=m.group("msg"),
                    semantic_context=context,
                ))
                i = next_i
                continue

            i += 1
        return diagnostics


class TypeScriptParser(BaseLanguageParser):
    """
    Parses TypeScript compiler (``tsc``) output.

    tsc uses a parens-coordinate format that no other parser handles:
        src/foo.ts(42,17): error TS2304: Cannot find name 'bar'.

    ESLint and other JS/TS linters use the ``file:L:C:`` form that the
    GenericParser already handles, so this parser focuses on the tsc
    variant specifically.
    """

    _PATTERN = re.compile(
        r'^\s*(?P<file>.+?\.(?:ts|tsx|d\.ts))\((?P<line>\d+),(?P<col>\d+)\):\s+'
        r'(?P<sev>error|warning)\s+(?P<code>TS\d+):\s+(?P<msg>.+)$'
    )

    @staticmethod
    def parse_diagnostics(raw_output: str) -> list[DiagnosticObject]:
        diagnostics: list[DiagnosticObject] = []
        lines = raw_output.splitlines()
        i = 0
        while i < len(lines):
            m = TypeScriptParser._PATTERN.match(lines[i])
            if not m:
                i += 1
                continue
            context, next_i = _collect_context_lines(
                lines, i + 1, TypeScriptParser._PATTERN,
            )
            diagnostics.append(DiagnosticObject(
                file=m.group("file"),
                line=int(m.group("line")),
                column=int(m.group("col")),
                severity=m.group("sev"),
                error_code=m.group("code"),
                message=m.group("msg"),
                semantic_context=context,
            ))
            i = next_i
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
        lines = raw_output.splitlines()
        i = 0
        while i < len(lines):
            match = GenericParser._PATTERN.match(lines[i].strip())
            if not match:
                i += 1
                continue
            # Collect attached context: javac/tsc `note:` follow-ons and
            # caret lines. Without this the LLM repair prompt only sees the
            # headline error and misses what the compiler explained
            # underneath.
            context, next_i = _collect_context_lines(
                lines, i + 1, GenericParser._PATTERN,
            )
            diagnostics.append(DiagnosticObject(
                file=match.group(1),
                line=int(match.group(2)),
                column=int(match.group(3)),
                severity=match.group(4).lower(),
                error_code="",
                message=match.group(5),
                semantic_context=context,
            ))
            i = next_i
        return diagnostics


# ---------------------------------------------------------------------------
# Parser Registry
# ---------------------------------------------------------------------------

# Maps compiler names to their parser classes. Detection uses substring
# matching against the lowercased build_command, so order doesn't matter
# but longer/more specific names should win over shorter ones (e.g.
# "ts-node" before "ts" — handled by exclusive prefixes here).
_PARSER_REGISTRY: dict[str, type[BaseLanguageParser]] = {
    "python": PythonParser,
    "pytest": PythonParser,
    # Java toolchain — Maven, Gradle, javac all share JavaParser since it
    # handles both bracketed-Maven and javac short-form on the same input.
    "mvn": JavaParser,
    "maven": JavaParser,
    "gradle": JavaParser,
    "gradlew": JavaParser,
    "javac": JavaParser,
    # TypeScript toolchain — tsc and frameworks that ultimately run tsc.
    "tsc": TypeScriptParser,
    "ts-node": TypeScriptParser,
    "vite": TypeScriptParser,
    "tsx": TypeScriptParser,
}

# Maps file extensions to parser classes
_EXTENSION_PARSER_MAP: dict[str, type[BaseLanguageParser]] = {
    ".py": PythonParser,
    ".pyi": PythonParser,
    ".java": JavaParser,
    ".ts": TypeScriptParser,
    ".tsx": TypeScriptParser,
}


def register_parser(compiler_name: str, parser_cls: type[BaseLanguageParser]) -> None:
    """
    Register a new language parser plugin for a given compiler name.

    Args:
        compiler_name: The compiler tool name (e.g., 'tsc', 'javac', 'python').
        parser_cls: A BaseLanguageParser subclass implementing parse_diagnostics.
    """
    _PARSER_REGISTRY[compiler_name] = parser_cls
    logger.info("[parser_registry] Registered parser for '%s': %s", compiler_name, parser_cls.__name__)


def register_extension_parser(extension: str, parser_cls: type[BaseLanguageParser]) -> None:
    """
    Register a parser to be used for files with a given extension.

    Args:
        extension: File extension including dot (e.g., '.ts', '.tsx').
        parser_cls: A BaseLanguageParser subclass.
    """
    _EXTENSION_PARSER_MAP[extension] = parser_cls
    logger.info("[parser_registry] Registered extension parser for '%s': %s", extension, parser_cls.__name__)


def get_parser(compiler_name: str) -> Optional[type[BaseLanguageParser]]:
    """
    Look up a registered parser by compiler name.

    Args:
        compiler_name: The compiler name (e.g., 'tsc', 'javac').

    Returns:
        The parser class, or None if no parser is registered.
    """
    return _PARSER_REGISTRY.get(compiler_name)


def get_parser_for_extension(extension: str) -> Optional[type[BaseLanguageParser]]:
    """
    Look up a registered parser by file extension.

    Args:
        extension: File extension including dot (e.g., '.py', '.ts').

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
        1. By compiler name inferred from build_command (e.g., 'tsc', 'javac')
        2. By file extension (e.g., '.py' → python, '.ts' → tsc)
        3. By output signature — every registered parser is run against
           the raw output and the one extracting the most diagnostics
           wins. Catches the common case where the build command is a
           generic wrapper (``npm run build``, ``make``)
           whose name doesn't appear in :data:`_PARSER_REGISTRY` but
           whose internals call ``tsc`` / ``jest`` / etc.
           and emit recognisable diagnostic formats.
        4. Falls back to GenericParser.

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
    # when CLICOLOR_FORCE=1 is set, which would otherwise silently drop
    # every diagnostic.
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
        ext_parser_cls = _EXTENSION_PARSER_MAP.get(ext)
        if ext_parser_cls is not None:
            logger.debug("[parser_registry] Using extension-based parser for '%s'.", ext)
            return ext_parser_cls.parse_diagnostics(raw_output)

    # Output-signature detection — when neither the command nor the file
    # extension identifies the compiler (e.g. ``npm run build``,
    # ``make``, ``yarn test``, ``pnpm tsc``), try every registered
    # parser on the raw output and pick the one with the most matches.
    # The per-language diagnostic formats are distinctive enough that
    # cross-matches are rare; ties resolve to the parser registered
    # first (dict iteration order is insertion order).
    candidates: list[tuple[type[BaseLanguageParser], list[DiagnosticObject]]] = []
    seen_classes: set[type[BaseLanguageParser]] = set()
    for parser_cls in _PARSER_REGISTRY.values():
        if parser_cls in seen_classes:
            continue
        seen_classes.add(parser_cls)
        try:
            diags = parser_cls.parse_diagnostics(raw_output)
        except Exception as exc:  # noqa: BLE001 — one bad parser shouldn't kill detection
            logger.debug("[parser_registry] %s raised during sniff: %s", parser_cls.__name__, exc)
            continue
        if diags:
            candidates.append((parser_cls, diags))
    if candidates:
        candidates.sort(key=lambda c: len(c[1]), reverse=True)
        winner_cls, winner_diags = candidates[0]
        logger.debug(
            "[parser_registry] Output-signature match: %s (%d diag(s)).",
            winner_cls.__name__, len(winner_diags),
        )
        return winner_diags

    # Fall back to generic parser
    logger.debug("[parser_registry] No specific parser detected. Using GenericParser.")
    return GenericParser.parse_diagnostics(raw_output)
