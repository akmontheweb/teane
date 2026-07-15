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
    # Long-traceback frame boundary: ``--tb=long`` (teane's build flag)
    # ends each frame's section with a BARE "path.py:NN: " line — no
    # ``in <func>`` suffix (that's the short-tb style matched above) and
    # no error type (that's the terminal fail line). Session 22471c0c:
    # every workspace frame in the real traceback rendered this way, so
    # the short-tb pattern captured nothing and the "no such table"
    # diagnostic stayed anchored on sqlalchemy venv internals.
    _PYTEST_LONG_FRAME_PATTERN = re.compile(
        r'^(?P<file>[^\s:][^:]*?\.py):(?P<line>\d+):\s*$'
    )
    # Source ``def`` line inside a long-tb frame section — pytest renders
    # the frame's source snippet above the boundary line, so the last
    # ``def`` seen names the function the frame executes in.
    _PYTEST_DEF_LINE_PATTERN = re.compile(
        r'^(?:async\s+)?def\s+(?P<name>\w+)\s*\('
    )
    # Pytest "E   " line — error name and message at the deepest level.
    _PYTEST_E_PATTERN = re.compile(
        r'^E\s+(?P<type>\w+(?:Error|Warning|Exception)):\s*(?P<msg>.+)$'
    )
    # Bare pytest "E   " continuation line without an ``ErrorType:`` prefix —
    # emitted by pytest's assertion-rewrite for plain ``assert`` failures
    # (``E   assert True is False``, ``E    +  where True = <obj>.attr``).
    # These lines carry the ACTUAL value the assertion evaluated to and are
    # the single most useful piece of context for the repair LLM on a bare
    # ``AssertionError``; without them the diagnostic collapses to just the
    # exception type and the LLM has to guess what the value was.
    _PYTEST_E_BARE_PATTERN = re.compile(r'^E\s+(?P<body>\S.*)$')
    # Pytest per-failure block header (``--tb=long``):
    #   ____________ TestClass.test_method ____________
    # We use this to reset the ``pending_locals`` buffer so a stray
    # ``x = 1`` line from an earlier test can't leak into the next
    # failure's context.
    _PYTEST_FAILURE_HEADER_PATTERN = re.compile(
        r'^_{3,}\s+\S.*?\S\s+_{3,}$'
    )
    # Pytest ``--showlocals`` dump line at column 0:
    #   ``resp       = <Response [500 Internal Server Error]>``
    #   ``client     = <httpx.AsyncClient object at 0x7f...>``
    # These are frame-local values at the failing frame — for HTTP tests
    # they surface the response object, for dict comparisons they dump
    # the full expected/actual structures. Constrained to column 0 (no
    # leading whitespace) so this doesn't accidentally swallow source
    # code lines from the ``--tb=long`` snippet (those are indented by 4).
    # The name must be a valid Python identifier; disqualifies path-like
    # tokens (``tests/foo.py``) and pytest summary rows.
    _PYTEST_LOCAL_PATTERN = re.compile(
        r'^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s{1,}=\s(?P<value>.+)$'
    )
    # Cap on ``--showlocals`` lines captured per failure block. Test
    # frames rarely have more than 6-8 real locals; a cap keeps the
    # semantic_context bounded on pathological cases (a debug fixture
    # that stashes 50 items into ``self``).
    _MAX_LOCAL_LINES = 10
    # Embedded ``File "path", line N`` inside a bare-E body — emitted by
    # pytest when the exception is a SyntaxError (or similar compile-time
    # error) whose real location is the user file being parsed, not the
    # stdlib frame that raised. Without this recovery, the diagnostic anchors
    # to ``/usr/lib/python3.11/ast.py:50`` and the repair LLM cannot ground.
    _E_BARE_FILE_LINE_PATTERN = re.compile(
        r'^File\s+"(?P<file>[^"]+\.py)",\s+line\s+(?P<line>\d+)\s*$'
    )
    # Pytest "short test summary info" lines at run end:
    #   FAILED tests/foo.py::test_x - AssertionError: assert ...
    #   ERROR  tests/conftest.py - ModuleNotFoundError: No module named 'x'
    # ERROR rows omit the "::test" suffix (collection-time failures).
    # ``nodeid_tail`` captures the ``::TestClass::test_method`` suffix so
    # ``_parse_pytest_summary`` can preserve the full pytest node id for
    # downstream isolation re-runs (compiler_node uses it to detect
    # test-ordering pollution).
    _PYTEST_SUMMARY_PATTERN = re.compile(
        r'^(?P<kind>FAILED|ERROR)\s+(?P<file>[^\s:]+\.py)(?P<nodeid_tail>::\S+)?\s+-\s+'
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
        # Buffer bare ``E   ...`` lines from pytest's assertion-rewrite output
        # (``E   assert True is False``, ``E    +  where True = <obj>.attr``).
        # These are the resolved values the assertion actually evaluated to —
        # they turn "AssertionError at line 38" (unactionable) into
        # "session.is_active was True where the test asserted False"
        # (immediately actionable). Cap at 8 lines to bound context growth
        # on tests that print large diffs (``assert big_dict == other_dict``).
        pending_e_bare: list[str] = []
        _MAX_E_BARE_LINES = 8
        # Buffer ``--showlocals`` dumps between the failure-block header and
        # the terminal fail-line. Pytest emits these as unindented
        # ``name = <repr>`` lines both before the code snippet (parent-
        # frame locals like ``self``) and after the ``E`` lines (failing-
        # frame locals). Both variants are useful; we merge them by name.
        pending_locals: dict[str, str] = {}
        # Buffer ``[teane] ...`` lines emitted by the Layer 2 pytest plugin
        # (``harness/pytest_plugins/teane_diagnostics.py``). The plugin
        # renders shape-aware detail for known types (httpx.Response body,
        # subprocess.CompletedProcess streams, exception chains) which no
        # ``--showlocals`` output would surface on its own. Bounded by
        # ``_MAX_TEANE_LINES`` so a runaway renderer can't overwhelm the
        # diagnostic prompt.
        pending_teane: list[str] = []
        _MAX_TEANE_LINES = 60
        # Workspace-owned frames seen in the current stanza, outermost →
        # innermost. When the error is raised inside a library (the
        # sqlalchemy/starlette case), the terminal fail-line and
        # ``last_frame`` both point at a venv path — actionless for the
        # repair LLM and outside the patch allowlist. The frames that
        # matter (the app-side call chain: api → repository → session
        # wiring) were walked right past. We keep them so (1) the emitted
        # diagnostic can be re-anchored on the innermost user frame and
        # (2) the full chain lands in semantic_context as breadcrumbs.
        # Session 22471c0c: five repair rounds re-edited test fixtures
        # because "no such table: companies" surfaced ONLY as
        # site-packages/sqlalchemy/engine/default.py:952 with starlette
        # middleware locals — every app frame had been discarded.
        stanza_user_frames: list[tuple[str, int, str]] = []
        _MAX_USER_FRAMES = 10
        # Function name for the NEXT long-tb frame boundary — tracked from
        # the ``def`` line in the frame's source snippet.
        last_def_name: Optional[str] = None
        # Index into ``diags`` where the current failure block began.
        # Chained exceptions (``raise X from Y``) render the cause's
        # traceback first — often library-frames-only — then the outer
        # chain with the workspace frames, and BOTH end in a terminal
        # fail line carrying near-identical messages. The collapse logic
        # at emission compares only against diags from the same block.
        block_start_idx = 0

        def _chain_dup_key(code: str, msg: str) -> tuple[str, str]:
            # Chained duplicates differ only in exception-wrapper noise
            # ("sqlite3.OperationalError: X" vs
            # "sqlalchemy.exc.OperationalError: (sqlite3.OperationalError)
            # X"), so key on the punctuation-stripped tail of the message.
            normalized = re.sub(r"[^a-z0-9]", "", (msg or "").lower())
            return (code, normalized[-48:])
        # The Layer 2 plugin emits ``[teane] ...`` lines (runtime object
        # detail) and the Layer 3b debug mode emits ``[teane-debug] ...``
        # lines (pre-call fixture snapshot, unhandled asyncio task
        # exceptions). We accept both markers here — the ``[teane`` prefix
        # tightly identifies plugin output while leaving room for future
        # buckets (``[teane-lsp]``, etc.) without another parser round.
        _TEANE_LINE_PREFIX = "[teane"
        _TEANE_SECTION_HEADER = "runtime object detail (harness enrichment):"
        # Pytest's ``--tb=long`` renders the terminal ``file:line: ErrorType``
        # BEFORE the plugin's ``addsection`` output — so the diagnostic gets
        # emitted (and buffers reset) while the ``[teane]`` lines are still
        # incoming. We keep a reference to the most-recent diagnostic within
        # the current failure block and graft late-arriving ``[teane]`` lines
        # onto its ``semantic_context`` rather than dropping them.
        last_emitted: Optional[DiagnosticObject] = None

        def _attach_teane_line(diag: DiagnosticObject, raw_line: str) -> None:
            # ``raw_line`` is the marker-stripped body (caller passes the
            # slice past ``] ``). Header is added once per diagnostic so
            # multiple ``[teane...]`` buckets land under the same section.
            current = diag.semantic_context or ""
            if _TEANE_SECTION_HEADER in current:
                diag.semantic_context = current + "\n  " + raw_line
                return
            sep = "\n" if current else ""
            diag.semantic_context = (
                current + sep + _TEANE_SECTION_HEADER + "\n  " + raw_line
            )

        def _fmt_context(
            bare_lines: list[str],
            src: Optional[str],
            locals_map: dict[str, str],
            teane_lines: list[str],
            user_frames: Optional[list[tuple[str, int, str]]] = None,
        ) -> str:
            parts: list[str] = []
            if src:
                parts.append(f"failing source: {src}")
            if user_frames and len(user_frames) > 1:
                # Render the workspace call chain so the repair LLM can
                # follow the failure into the app-side files instead of
                # fixating on the single file:line the diagnostic anchors
                # on. Only worth showing with ≥2 frames — a single frame
                # is already the diagnostic's own location.
                chain = "\n  ".join(
                    f"{f}:{ln} in {fn}" if fn else f"{f}:{ln}"
                    for f, ln, fn in user_frames
                )
                parts.append(
                    f"workspace call chain (outermost → innermost):\n  {chain}"
                )
            if bare_lines:
                joined = "\n  ".join(bare_lines)
                parts.append(f"assertion-rewrite:\n  {joined}")
            if locals_map:
                # Deterministic order — sort so successive rounds produce
                # byte-identical context, which lets the cross-round
                # fingerprint deduper and cache_control prefix stay stable.
                joined = "\n  ".join(
                    f"{name} = {value}"
                    for name, value in sorted(locals_map.items())
                )
                parts.append(f"locals at failure frame:\n  {joined}")
            if teane_lines:
                # Strip the ``[teane...] `` marker from each line before
                # rendering — the marker was for the parser, not the LLM.
                # Match ``[teane] `` and ``[teane-<bucket>] `` variants.
                def _strip_marker(ln: str) -> str:
                    if not ln.startswith(_TEANE_LINE_PREFIX):
                        return ln
                    close = ln.find("] ")
                    return ln[close + 2:] if close != -1 else ln
                stripped_teane = [_strip_marker(ln) for ln in teane_lines]
                joined = "\n  ".join(stripped_teane)
                parts.append(f"{_TEANE_SECTION_HEADER}\n  {joined}")
            return "\n".join(parts)

        def _recover_user_frame_from_e_bare(
            frame: Optional[tuple[str, int]],
            bare_lines: list[str],
        ) -> Optional[tuple[str, int]]:
            # Trust the recorded frame when it already points at user code.
            if frame is not None and PythonParser._is_user_frame(frame[0]):
                return frame
            # Pytest tags SyntaxError bodies with ``E   File "<path>", line N``.
            # Prefer the last user-code hit — the deepest user frame wins the
            # same way ``_parse_standard_traceback`` picks innermost user code.
            recovered: Optional[tuple[str, int]] = None
            for body in bare_lines:
                m = PythonParser._E_BARE_FILE_LINE_PATTERN.match(body.strip())
                if m and PythonParser._is_user_frame(m.group("file")):
                    recovered = (m.group("file"), int(m.group("line")))
            if recovered is not None:
                return recovered
            # Last resort before giving up: the innermost workspace frame
            # walked earlier in this stanza. Beats anchoring the diagnostic
            # on a site-packages path the LLM can neither read usefully nor
            # patch.
            if stanza_user_frames:
                f, ln, _fn = stanza_user_frames[-1]
                return (f, ln)
            return frame

        for line in lines:
            stripped = line.rstrip()
            bare = stripped.strip()

            # Failure-block header — a new failure begins here; reset the
            # buffers that must not leak from the previous block. Locals
            # dumped BEFORE the code snippet (e.g. ``self = <...>``) still
            # belong to this failure, so we reset here rather than on the
            # first local line.
            if PythonParser._PYTEST_FAILURE_HEADER_PATTERN.match(bare):
                pending_locals = {}
                pending_teane = []
                pending_e = None
                pending_e_bare = []
                pending_failing_src = None
                last_frame = None
                stanza_user_frames = []
                last_def_name = None
                block_start_idx = len(diags)
                conftest_path = None
                last_emitted = None
                continue

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
                if PythonParser._is_user_frame(last_frame[0]):
                    entry = (
                        last_frame[0], last_frame[1],
                        frame.group("func").strip(),
                    )
                    if (
                        len(stanza_user_frames) < _MAX_USER_FRAMES
                        and (not stanza_user_frames
                             or stanza_user_frames[-1] != entry)
                    ):
                        stanza_user_frames.append(entry)
                continue

            # ``--tb=long`` frame boundary: bare "path.py:NN:" with nothing
            # after the colon. No clash with the terminal fail line below —
            # that one carries an error-type suffix this pattern rejects.
            long_frame = PythonParser._PYTEST_LONG_FRAME_PATTERN.match(bare)
            if long_frame:
                last_frame = (
                    long_frame.group("file"), int(long_frame.group("line")),
                )
                if PythonParser._is_user_frame(last_frame[0]):
                    entry = (last_frame[0], last_frame[1], last_def_name or "")
                    if (
                        len(stanza_user_frames) < _MAX_USER_FRAMES
                        and (not stanza_user_frames
                             or stanza_user_frames[-1] != entry)
                    ):
                        stanza_user_frames.append(entry)
                last_def_name = None
                continue

            # ``def`` line from a long-tb source snippet — names the
            # function for the NEXT frame boundary.
            def_line = PythonParser._PYTEST_DEF_LINE_PATTERN.match(bare)
            if def_line:
                last_def_name = def_line.group("name")
                # Fall through: a def line carries no other signal.
                continue

            # Terminal "file:line: ErrorType" (no message, no E-line follow-up)
            term = PythonParser._PYTEST_FAIL_LINE_PATTERN.match(bare)
            if term:
                term_type = term.group("type")
                # Prefer the pending E-line body when it matches the terminal
                # error type — that's the actual assertion text the LLM needs.
                msg = term_type
                if pending_e is not None and pending_e[0] == term_type:
                    msg = f"{term_type}: {pending_e[1]}"
                elif pending_e_bare:
                    # No typed E-line, but assertion-rewrite gave us the
                    # resolved expression (``assert True is False``).
                    # Promote the first bare line into the message so the
                    # judge/repair prompts don't show a bare exception type.
                    msg = f"{term_type}: {pending_e_bare[0]}"
                ctx = _fmt_context(
                    pending_e_bare, pending_failing_src, pending_locals,
                    pending_teane, stanza_user_frames,
                )
                # Re-anchor on the innermost workspace frame when the
                # terminal line points into a library. The venv path is
                # where the exception was *raised*; the workspace frame is
                # where the fix lives — and it's what drives downstream
                # file auto-injection into the repair prompt. The original
                # library location is preserved in semantic_context.
                term_file, term_line = term.group("file"), int(term.group("line"))
                if (
                    not PythonParser._is_user_frame(term_file)
                    and stanza_user_frames
                ):
                    lib_note = f"raised in library frame: {term_file}:{term_line}"
                    ctx = f"{ctx}\n{lib_note}" if ctx else lib_note
                    term_file, term_line, _fn = stanza_user_frames[-1]
                diag = DiagnosticObject(
                    file=term_file,
                    line=term_line,
                    column=0,
                    severity="error",
                    error_code=term_type,
                    message=msg,
                    semantic_context=ctx,
                )
                # Chained-exception collapse: ``raise X from Y`` renders
                # the cause's traceback first (library frames only, ending
                # in its own terminal line), then the outer chain with the
                # workspace frames. Both terminals carry the same
                # underlying error. Keep ONE diagnostic per block,
                # preferring the workspace-anchored emission — otherwise
                # the cause's venv-anchored duplicate can win downstream
                # dedupe and hide the app-side call chain again.
                _dup_idx: Optional[int] = None
                _new_key = _chain_dup_key(term_type, msg)
                for _i in range(block_start_idx, len(diags)):
                    if _chain_dup_key(
                        diags[_i].error_code, diags[_i].message,
                    ) == _new_key:
                        _dup_idx = _i
                        break
                if _dup_idx is not None:
                    _old_user = PythonParser._is_user_frame(diags[_dup_idx].file)
                    _new_user = PythonParser._is_user_frame(term_file)
                    if _new_user and not _old_user:
                        diags[_dup_idx] = diag
                        last_emitted = diag
                    # else: keep the earlier emission, drop the duplicate.
                else:
                    diags.append(diag)
                    last_emitted = diag
                last_frame = None
                stanza_user_frames = []
                last_def_name = None
                pending_e = None
                pending_e_bare = []
                pending_failing_src = None
                pending_locals = {}
                pending_teane = []
                continue

            e_line = PythonParser._PYTEST_E_PATTERN.match(stripped)
            if e_line:
                # Always remember it for the terminal-line fallback above.
                pending_e = (e_line.group("type"), e_line.group("msg"))
                recovered = _recover_user_frame_from_e_bare(
                    last_frame, pending_e_bare,
                )
                if recovered is not None:
                    file_path, lineno = recovered
                elif conftest_path is not None:
                    file_path, lineno = conftest_path, 0
                else:
                    # No frame yet — wait for the terminal fail line to pair us.
                    continue
                ctx = _fmt_context(
                    pending_e_bare, pending_failing_src, pending_locals,
                    pending_teane, stanza_user_frames,
                )
                # Same raised-at preservation as the terminal branch: when
                # the walked frame was a library path and we re-anchored on
                # a workspace frame, keep the library location visible.
                if (
                    last_frame is not None
                    and not PythonParser._is_user_frame(last_frame[0])
                    and (file_path, lineno) != last_frame
                ):
                    lib_note = (
                        f"raised in library frame: "
                        f"{last_frame[0]}:{last_frame[1]}"
                    )
                    ctx = f"{ctx}\n{lib_note}" if ctx else lib_note
                diag = DiagnosticObject(
                    file=file_path,
                    line=lineno,
                    column=0,
                    severity="error",
                    error_code=e_line.group("type"),
                    message=f"{e_line.group('type')}: {e_line.group('msg')}",
                    semantic_context=ctx,
                )
                diags.append(diag)
                last_emitted = diag
                last_frame = None
                stanza_user_frames = []
                last_def_name = None
                conftest_path = None
                pending_e = None
                pending_e_bare = []
                pending_failing_src = None
                pending_locals = {}
                pending_teane = []
                continue

            # ``[teane...] ...`` marker line from the Layer 2 plugin
            # (``[teane]``) or Layer 3b debug mode (``[teane-debug]``).
            # Two arrival patterns:
            #  1. BEFORE the diagnostic is emitted (rare — plugin runs at
            #     ``call`` phase; --tb=long normally emits the terminal
            #     fail-line first). Buffered so the emission grafts them in.
            #  2. AFTER the terminal fail-line (the common case — pytest's
            #     ``addsection`` output sits UNDER the fail-line). Grafted
            #     onto ``last_emitted`` in-place so the LLM sees the runtime
            #     detail on the same diagnostic that carries the assertion.
            if bare.startswith(_TEANE_LINE_PREFIX):
                close = bare.find("] ")
                if close != -1:
                    body = bare[close + 2:]
                    if last_emitted is not None:
                        _attach_teane_line(last_emitted, body)
                        continue
                # Buffered fallback (no diagnostic yet or malformed marker):
                # keep the raw line so the emission-time _fmt_context can
                # strip the marker uniformly.
                if len(pending_teane) < _MAX_TEANE_LINES:
                    pending_teane.append(bare)
                continue
            # Bare ``E   ...`` line without an ``ErrorType:`` prefix —
            # assertion-rewrite output. Buffer it; it will be attached to
            # whichever diagnostic emits next (terminal fail-line or typed
            # E-line). Skip if it duplicates the ``>`` failing source line
            # (pytest sometimes echoes the raw assertion back on the E-line
            # verbatim, adding noise without new info).
            e_bare = PythonParser._PYTEST_E_BARE_PATTERN.match(stripped)
            if e_bare:
                body = e_bare.group("body").rstrip()
                if pending_failing_src and body == pending_failing_src:
                    continue
                if len(pending_e_bare) < _MAX_E_BARE_LINES:
                    pending_e_bare.append(body)
                continue

            # ``--showlocals`` dump — column-0 ``name = <repr>`` line.
            # Match strictly on ``stripped`` (unindented). Test-source code
            # lines from the ``--tb=long`` snippet are indented by ≥4 spaces
            # and get filtered out here; assertion-rewrite lines start with
            # ``E`` and are caught above; summary rows contain file paths
            # that fail the identifier pattern. Cap the map at
            # ``_MAX_LOCAL_LINES`` so a debug fixture stashing 50 items on
            # ``self`` can't blow the diagnostic prompt.
            local = PythonParser._PYTEST_LOCAL_PATTERN.match(stripped)
            if local and len(pending_locals) < PythonParser._MAX_LOCAL_LINES:
                pending_locals[local.group("name")] = local.group("value").rstrip()
                continue
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
            # Preserve the full pytest node id when present. Collection-time
            # errors ("ERROR tests/conftest.py - ...") lack the ``::test``
            # tail — those aren't re-runnable in isolation and stay blank.
            nodeid_tail = m.group("nodeid_tail") or ""
            full_nodeid = m.group("file") + nodeid_tail if nodeid_tail else ""
            diags.append(DiagnosticObject(
                file=m.group("file"),
                line=0,
                column=0,
                severity="error",
                error_code=error_type,
                message=message,
                semantic_context="",
                pytest_nodeid=full_nodeid,
            ))
        return diags

    @staticmethod
    def _dedup(diags: list[DiagnosticObject]) -> list[DiagnosticObject]:
        """Merge duplicate (file, line, error_code) entries; keep richer ones.

        Summary lines lack a line number (line=0) while per-failure blocks
        carry the exact line. When both surface the same failure, we keep
        the one with the line number and prefer the message that isn't just
        the error type repeated.

        ``pytest_nodeid`` is only populated by ``_parse_pytest_summary``
        (the failure-block form uses a bare ``file:line`` frame with no
        ``::test`` tail), so whenever we discard a summary entry in favour
        of the richer block entry we transfer the nodeid across — losing
        it would kill the isolation re-run downstream.
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
                if existing.pytest_nodeid and not d.pytest_nodeid:
                    d.pytest_nodeid = existing.pytest_nodeid
                by_key[key] = d
            elif d.line == existing.line and len(d.message) > len(existing.message):
                if existing.pytest_nodeid and not d.pytest_nodeid:
                    d.pytest_nodeid = existing.pytest_nodeid
                by_key[key] = d
            elif not existing.pytest_nodeid and d.pytest_nodeid:
                # The kept entry loses out on the nodeid otherwise; graft it.
                existing.pytest_nodeid = d.pytest_nodeid
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


class PyrightJSONParser(BaseLanguageParser):
    """
    Parses ``pyright --outputjson`` output.

    Pyright emits a single JSON document with a ``generalDiagnostics`` array;
    positions are 0-indexed (converted to the 1-indexed convention every
    other parser uses). ``information``-severity entries are dropped —
    they are hints, not actionable diagnostics. Malformed or truncated
    JSON yields ``[]`` (fail-open: the diagnostics gate must never block
    a run on parser trouble).
    """

    @staticmethod
    def parse_diagnostics(raw_output: str) -> list[DiagnosticObject]:
        try:
            payload = json.loads(raw_output)
        except (TypeError, ValueError):
            return []
        entries = payload.get("generalDiagnostics") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            return []
        diagnostics: list[DiagnosticObject] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            severity = str(entry.get("severity", "error")).lower()
            if severity not in ("error", "warning"):
                continue
            start = (entry.get("range") or {}).get("start") or {}
            try:
                line = int(start.get("line", 0)) + 1
                column = int(start.get("character", 0)) + 1
            except (TypeError, ValueError):
                line, column = 1, 1
            # Pyright message bodies are often multi-line: headline first,
            # indented elaboration below. Keep the headline as the message
            # and fold the rest into semantic_context like the text parsers.
            raw_msg = str(entry.get("message", "")).strip()
            first, _, rest = raw_msg.partition("\n")
            diagnostics.append(DiagnosticObject(
                file=str(entry.get("file", "")),
                line=line,
                column=column,
                severity=severity,
                error_code=str(entry.get("rule") or "pyright"),
                message=first.strip(),
                semantic_context=rest.strip(),
            ))
        return diagnostics


class MypyParser(BaseLanguageParser):
    """
    Parses mypy text output.

    Expects mypy run with ``--show-column-numbers --show-error-codes
    --no-error-summary --no-color-output``; the optional-column group keeps
    the regex tolerant of older mypy that omits columns. ``note:`` lines are
    folded into the preceding diagnostic's semantic_context (mypy uses them
    for "possible overload variants" style elaboration); a note with no
    preceding diagnostic is dropped.
    """

    _PATTERN = re.compile(
        r'^(?P<file>[^:\n]+\.pyi?):(?P<line>\d+):(?:(?P<col>\d+):)?\s*'
        r'(?P<sev>error|warning|note):\s*(?P<msg>.*?)(?:\s+\[(?P<code>[a-z0-9-]+)\])?$'
    )

    @staticmethod
    def parse_diagnostics(raw_output: str) -> list[DiagnosticObject]:
        diagnostics: list[DiagnosticObject] = []
        for raw_line in raw_output.splitlines():
            m = MypyParser._PATTERN.match(raw_line.strip())
            if not m:
                continue
            severity = m.group("sev")
            if severity == "note":
                if diagnostics:
                    prev = diagnostics[-1]
                    note = m.group("msg")
                    prev.semantic_context = (
                        f"{prev.semantic_context}\nnote: {note}".strip()
                        if prev.semantic_context else f"note: {note}"
                    )
                continue
            diagnostics.append(DiagnosticObject(
                file=m.group("file"),
                line=int(m.group("line")),
                column=int(m.group("col")) if m.group("col") else 0,
                severity=severity,
                error_code=m.group("code") or "mypy",
                message=m.group("msg"),
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
    # Python type-checkers (diagnostics gate; also matched when a user's
    # build_command invokes them directly).
    "pyright": PyrightJSONParser,
    "mypy": MypyParser,
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
