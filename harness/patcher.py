"""
Hybrid file modification engine: tree-sitter AST-aware rewriting with
pure text SEARCH/REPLACE fallback. Uses aiofiles for async file I/O.

This module implements:
    - BasePatcher ABC defining the canonical OperationType set:
        CREATE_FILE, REPLACE_BLOCK, DELETE_BLOCK, INSERT_AT_BLOCK,
        INSERT_AT_LINE, REPLACE_LINE_RANGE
    - TreeSitterPatcher: AST-aware rewriting using tree-sitter grammars.
      Locates target nodes by structural signature, rewrites, and pretty-prints.
      Preserves surrounding codebase formatting entirely.
    - TextPatcher: Pure exact-match SEARCH/REPLACE engine.
      Works on any text file; no AST dependency. Uses aiofiles for async I/O.
      Sensitive to whitespace/formatting drift but universally applicable.
    - HybridPatcher: Auto-selects TreeSitterPatcher for files with registered grammars
      (based on file extension), falls back to TextPatcher for everything else or when
      tree-sitter parsing fails.
    - PatchBlockParser: Extracts SEARCH/REPLACE blocks from LLM-generated text using
      the strict syntax defined in the system prompt.
    - Operation result tracking that populates the AgentState's modified_files list.
"""

from __future__ import annotations

import ast
import asyncio
import difflib
import errno
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Post-patch syntactic validation
# ---------------------------------------------------------------------------
#
# When the repair LLM emits a REPLACE_BLOCK / DELETE_BLOCK that surgically
# lands but leaves the file syntactically broken (mismatched braces, orphan
# dict tails, mid-scope indent jumps), the diagnostic surface on the next
# compile is dominated by that same file's SyntaxError. The judge then
# points at line N again, and the repair loop cycles on a self-inflicted
# blocker instead of the original one — observed session 674bfdbd, where
# a REPLACE_BLOCK left a `}}}}}` tail at test_edgar.py:187 and the next
# 6 rounds kept nibbling around it without ever landing a fix.
#
# The guard here re-parses every ``.py`` (and ``.json``) file after a
# successful patch. If the file was CLEAN before and is BROKEN after,
# the patch is rolled back and reported as a failure to the LLM. If the
# file was ALREADY broken pre-patch, we keep the change — the LLM may be
# iteratively fixing it and rolling back would regress partial progress.
_VALIDATED_SUFFIXES = (".py", ".json")


def _validate_syntax(filepath: str, content: str) -> Optional[str]:
    """Return an error-message string when ``content`` doesn't parse for
    the language implied by ``filepath``, otherwise None.

    Deliberately narrow: only Python, JSON, and requirements.txt —
    formats where a bad patch reliably wedges the whole build
    (SyntaxError at collect time; malformed manifest at install time;
    unparseable version specifier at pip resolve). Extending to TS/JS
    would require ``tsc``/``node`` in the harness runtime, which we
    don't guarantee.
    """
    if filepath.endswith(".py"):
        try:
            ast.parse(content)
            return None
        except SyntaxError as e:
            loc = f"line {e.lineno}" if e.lineno else "unknown line"
            return f"SyntaxError at {loc}: {e.msg}"
        except ValueError as e:
            # Empty ``\x00`` bytes and similar make ast.parse raise ValueError
            # rather than SyntaxError. Treat them the same for rollback.
            return f"ValueError during Python parse: {e}"
    if filepath.endswith(".json"):
        try:
            json.loads(content)
            return None
        except json.JSONDecodeError as e:
            return f"JSONDecodeError at line {e.lineno}: {e.msg}"
    # Finsearch session 44c5e194 root cause E4: a repair patch corrupted
    # requirements.txt to ``lxml==6.1.0.`` (trailing dot). Reflection
    # caught it 90 minutes later, past the HITL budget. Rolling back
    # invalid requirement lines at patch time forces the LLM to emit a
    # valid version specifier on the next turn instead of burning
    # rounds on the install failure.
    base = os.path.basename(filepath).lower()
    if base in ("requirements.txt", "requirements-dev.txt", "requirements-test.txt"):
        try:
            from packaging.requirements import Requirement, InvalidRequirement
        except ImportError:
            # packaging isn't guaranteed in the harness runtime.
            # Skip validation rather than block patches on our own
            # missing dep.
            return None
        for lineno, raw in enumerate(content.splitlines(), start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith(("#", "-r ", "--", "-e ", "-c ")):
                # Blank, comment, or a pip flag like -r reqs-base.txt / -e .
                continue
            # Environment markers like `pkg==1; python_version>="3.10"`
            # are handled by Requirement itself.
            try:
                Requirement(stripped)
            except InvalidRequirement as e:
                return (
                    f"Invalid requirement at line {lineno}: {stripped!r} "
                    f"({e}). Every non-comment line must be a valid PEP 508 "
                    "requirement — no trailing dots on versions, no "
                    "half-typed specifiers."
                )
    return None


# ---------------------------------------------------------------------------
# 1. Types & Enums
# ---------------------------------------------------------------------------

class OperationType(Enum):
    """File modification operations.

    The first four are anchor-based: the caller supplies a search string
    or anchor symbol and the patcher locates it in the file. The last two
    (added for scanner-driven autofix, Layer 1 of the security pipeline)
    are line-coordinate-based: the caller supplies a 1-based line number
    or range and the patcher splices directly. Line-based ops sidestep
    the failure mode where a scanner's "missing X" finding has no anchor
    string for the LLM to latch onto.
    """
    CREATE_FILE = "create_file"
    REPLACE_BLOCK = "replace_block"
    DELETE_BLOCK = "delete_block"
    INSERT_AT_BLOCK = "insert_at_block"
    INSERT_AT_LINE = "insert_at_line"
    REPLACE_LINE_RANGE = "replace_line_range"
    # Full-file overwrite. Emitted as an escape hatch when surgical
    # patches on a single file have failed to converge across multiple
    # repair rounds — the LLM's mental model of the file has drifted so
    # far that fresh regeneration is cheaper than another round of
    # REPLACE_BLOCK guessing. Behaves like CREATE_FILE but tolerates a
    # pre-existing file (which CREATE_FILE deliberately does not).
    # Post-patch parse validation still applies, so a REWRITE_FILE that
    # produces broken syntax is rolled back the same as any other op.
    REWRITE_FILE = "rewrite_file"


class Placement(Enum):
    BEFORE = "before"
    AFTER = "after"


@dataclass
class PatchBlock:
    """
    A single parsed patch instruction extracted from LLM output.

    Matches the strict formatting defined in the system prompt:

    REPLACE_BLOCK, CREATE_FILE, DELETE_BLOCK, INSERT_AT_BLOCK
    with tagged file:, search:, replace:, content:, anchor:, placement: fields.
    """
    operation: OperationType
    file: str
    search: str = ""
    replace: str = ""
    content: str = ""
    anchor: str = ""         # For INSERT_AT_BLOCK: function/class name to anchor to
    placement: Placement = Placement.AFTER  # For INSERT_AT_BLOCK
    # Match-count policy for REPLACE_BLOCK / DELETE_BLOCK:
    #   "unique" — fail when the search matches more than once (default,
    #              matches historical strict behaviour).
    #   "all"    — apply to every match in the file.
    #   "first"  — apply only to the first match.
    # Borrows the spirit of Claude Code's Edit ``replace_all`` flag: gives
    # the LLM an explicit escape from the "matched N times. Must be unique"
    # dead-end without forcing it to pad the search with extra context.
    count: str = "unique"
    # Line coordinates for INSERT_AT_LINE / REPLACE_LINE_RANGE.
    # ``line`` is the 1-based line number the content lands at (or starts
    # at, for the range op). ``end_line`` is the inclusive end for the
    # range op; left at 0 for the insert op. ``expected_file_hash`` is an
    # optional sha256 of the file's bytes at the time the caller chose
    # the line number — when supplied the patcher rejects the patch if
    # the file has drifted, reusing the B5 drift sensor. Empty string
    # means "trust the line number unconditionally" (callers that always
    # re-extract diagnostics fresh per round can leave it blank).
    line: int = 0
    end_line: int = 0
    expected_file_hash: str = ""

    raw_block: str = ""  # The full matched block text for debugging


@dataclass
class PatchResult:
    """Result of applying a single patch block."""
    success: bool
    file: str
    operation: OperationType
    message: str = ""
    lines_changed: int = 0
    error: Optional[str] = None
    # True when the operation succeeded as a resume-safe idempotency no-op
    # (file already at the target state). Distinguishes real progress from
    # "nothing to do" so the repair loop's consecutive-zero tripwire is not
    # masked by the LLM repeatedly emitting already-applied patches.
    no_op: bool = False


# ---------------------------------------------------------------------------
# 2. Patch Block Parser — Extracts SEARCH/REPLACE from LLM text
# ---------------------------------------------------------------------------

# Regex patterns to match each block type in LLM output
# READ_FILE is not a patch operation — it's an LLM-side request for current
# file bytes that the host resolves inline before re-dispatching. Borrows
# the spirit of Claude Code's Read tool: instead of guessing what a file
# contains, the LLM asks and gets the line-numbered current content back
# in the same turn (no repair-loop slot consumed).
#
# Syntax:
#   <<<READ_FILE>>>
#   file: path/to/foo.py
#   range: 1-200       # optional; default is whole file capped at the
#                      # patcher's line/char limits.
#   <<<END_READ_FILE>>>
# Bracket count is lenient (1-3) on purpose: models under long reasoning
# traces slip to ``<READ_FILE>`` / ``<<READ_FILE>>``, and a strict match
# silently drops the request — the round then lands zero patches AND zero
# reads, with no corrective feedback. Session 22471c0c's post-resume run
# stalled to HITL exactly this way: a well-formed investigation request
# for server/app/models/*.py written with single angle brackets was
# ignored, and two such rounds tripped the zero-patch gate. The interior
# shape (READ_FILE marker + ``file:`` line + END marker) is distinctive
# enough that bracket leniency cannot misfire on real code content.
_READ_FILE_PATTERN = re.compile(
    r'<{1,3}READ_FILE>{1,3}\s*\n'
    r'file:\s*(?P<file>.+?)\s*\n'
    r'(?:range:\s*(?P<range>\d+\s*-\s*\d+|\d+)\s*\n)?'
    r'<{1,3}END_READ_FILE>{1,3}',
    re.DOTALL,
)


def parse_read_blocks(llm_output: str) -> list[tuple[str, Optional[tuple[int, int]]]]:
    """Extract READ_FILE blocks from ``llm_output``.

    Returns a list of ``(rel_path, optional_range)`` where ``optional_range``
    is ``(start_line, end_line)`` or ``None``. Best-effort: malformed range
    fields produce ``None`` rather than raising — the resolver will fall back
    to whole-file output.
    """
    out: list[tuple[str, Optional[tuple[int, int]]]] = []
    for match in _READ_FILE_PATTERN.finditer(llm_output):
        file_ref = match.group("file").strip()
        if not file_ref:
            continue
        range_str = match.group("range")
        if range_str is None:
            out.append((file_ref, None))
            continue
        try:
            if "-" in range_str:
                a, b = range_str.split("-", 1)
                lo, hi = int(a.strip()), int(b.strip())
                if lo > hi or lo < 1:
                    out.append((file_ref, None))
                else:
                    out.append((file_ref, (lo, hi)))
            else:
                line = int(range_str.strip())
                if line < 1:
                    out.append((file_ref, None))
                else:
                    out.append((file_ref, (line, line)))
        except ValueError:
            out.append((file_ref, None))
    return out


def strip_read_blocks(llm_output: str) -> str:
    """Return ``llm_output`` with every READ_FILE block removed.

    Used after the harness has resolved READ_FILE blocks and is about to
    parse the rest of the response as patch blocks. We do NOT want READ_FILE
    blocks to feed into ``parse_patch_blocks`` (they're not patches), nor do
    we want them to leak into commit messages or transcripts.
    """
    return _READ_FILE_PATTERN.sub("", llm_output)


# PROMOTE_DEFERRED is the LLM's escape hatch against the cascade-ranking
# heuristic. When the diagnostic formatter defers a group to the "for
# awareness" tail, the LLM can emit:
#
#   <<<PROMOTE_DEFERRED>>>
#   codes: TS2769, TS2353
#   <<<END_PROMOTE_DEFERRED>>>
#
# to force the named error codes into top-N (full ``semantic_context``)
# on the NEXT repair iteration. Mirrors READ_FILE's pattern: it's not a
# patch operation, the harness consumes it inline and acts on it during
# prompt assembly. The model knows when it disagrees with the harness's
# prioritisation; this is how it says so.
_PROMOTE_DEFERRED_PATTERN = re.compile(
    r'<<<PROMOTE_DEFERRED>>>\s*\n'
    r'codes:\s*(?P<codes>[^\n]+)\s*\n'
    r'<<<END_PROMOTE_DEFERRED>>>',
    re.DOTALL,
)


def parse_promote_deferred_blocks(llm_output: str) -> list[str]:
    """Extract error codes from PROMOTE_DEFERRED blocks in ``llm_output``.

    Returns a deduped, order-preserving list of code strings (e.g.
    ``["TS2769", "F401"]``). Whitespace around individual codes is
    stripped; empty entries are dropped. Multiple PROMOTE_DEFERRED blocks
    in one response are combined. Best-effort: malformed blocks
    contribute nothing rather than raising.
    """
    seen: set[str] = set()
    out: list[str] = []
    for match in _PROMOTE_DEFERRED_PATTERN.finditer(llm_output):
        codes_field = match.group("codes") or ""
        for raw in codes_field.split(","):
            code = raw.strip()
            if not code or code in seen:
                continue
            seen.add(code)
            out.append(code)
    return out


def strip_promote_deferred_blocks(llm_output: str) -> str:
    """Return ``llm_output`` with every PROMOTE_DEFERRED block removed.

    Called after the harness has captured the promoted codes so the
    blocks don't leak into ``parse_patch_blocks`` (they're not patches),
    commit messages, or transcripts.
    """
    return _PROMOTE_DEFERRED_PATTERN.sub("", llm_output)


# Tempered greedy token: matches any character (DOTALL) UNLESS that
# character is the start of another block opener. This prevents a
# non-greedy ``.*?`` body from spanning a forbidden boundary when the
# expected END marker is missing — e.g. when ``_continue_on_length``
# fires mid-CREATE_FILE and the LLM re-emits the whole block on the
# continuation, the concatenated text has two ``<<<CREATE_FILE>>>``
# openers but only one ``<<<END_CREATE_FILE>>>``. Without this guard
# the first block's content captures the second block's directive
# lines (``<<<CREATE_FILE>>>\nfile: …\ncontent:\n…``) verbatim and the
# patcher writes that DSL to disk as file content — a real bug we hit
# on the ciod build (server/src/db/seed.ts corruption). With the
# guard, the malformed first block simply fails to match → the patcher
# logs "no patches parsed" → repair sees the failure cleanly.
_TEMPERED_CONTENT = (
    r'(?:(?!<<<(?:CREATE_FILE|REWRITE_FILE|REPLACE_BLOCK|DELETE_BLOCK|INSERT_AT_BLOCK)>>>).)*?'
)


# Header fields (``file:``, ``anchor:``) are always one line. The
# original ``.+?\s*\n`` capture was too loose: under DOTALL, ``.``
# matches newlines, so when the END marker was missing the regex engine
# would backtrack-extend the ``file:`` capture across newlines until it
# found a valid trailer somewhere downstream. Restricting these to
# ``[^\n]+?`` confines them to the header line where they belong.
_FIELD_VALUE = r'[^\n]+?'


# Body-field separator. The canonical documented shape is
# ``content:\n<body>`` (newline immediately after the label), and the
# LLM produces that whenever it has slack. Under output-token pressure
# it drops the newline and inlines: ``content:<body>``. Historically
# the regex was ``content:\s*\n`` — that silently dropped every inline
# block. Finsearch session 156032347 died from this: LLM emitted 600
# inline ``content:<body>`` blocks under a 32k output cap, the parser
# returned zero blocks, and stories 014–019 all carried as defects.
#
# ``[ \t]*\n?`` accepts both shapes without eating leading whitespace
# of the body itself (``\s*`` would greedily consume indentation of
# the file's real content, breaking any file that starts with
# leading blank lines or indented python/js/ts).
_BODY_SEP = r'[ \t]*\n?'

_BLOCK_PATTERNS = {
    OperationType.REPLACE_BLOCK: re.compile(
        r'<<<REPLACE_BLOCK>>>\s*\n'
        r'file:\s*(?P<file>' + _FIELD_VALUE + r')\s*\n'
        r'(?:count:\s*(?P<count>unique|all|first)\s*\n)?'
        r'search:' + _BODY_SEP + r'(?P<search>' + _TEMPERED_CONTENT + r')\n'
        r'replace:' + _BODY_SEP + r'(?P<replace>' + _TEMPERED_CONTENT + r')'
        r'<<<END_REPLACE_BLOCK>>>',
        re.DOTALL,
    ),
    OperationType.CREATE_FILE: re.compile(
        r'<<<CREATE_FILE>>>\s*\n'
        r'file:\s*(?P<file>' + _FIELD_VALUE + r')\s*\n'
        r'content:' + _BODY_SEP + r'(?P<content>' + _TEMPERED_CONTENT + r')'
        r'<<<END_CREATE_FILE>>>',
        re.DOTALL,
    ),
    OperationType.REWRITE_FILE: re.compile(
        r'<<<REWRITE_FILE>>>\s*\n'
        r'file:\s*(?P<file>' + _FIELD_VALUE + r')\s*\n'
        r'content:' + _BODY_SEP + r'(?P<content>' + _TEMPERED_CONTENT + r')'
        r'<<<END_REWRITE_FILE>>>',
        re.DOTALL,
    ),
    OperationType.DELETE_BLOCK: re.compile(
        r'<<<DELETE_BLOCK>>>\s*\n'
        r'file:\s*(?P<file>' + _FIELD_VALUE + r')\s*\n'
        r'(?:count:\s*(?P<count>unique|all|first)\s*\n)?'
        r'search:' + _BODY_SEP + r'(?P<search>' + _TEMPERED_CONTENT + r')'
        r'<<<END_DELETE_BLOCK>>>',
        re.DOTALL,
    ),
    OperationType.INSERT_AT_BLOCK: re.compile(
        r'<<<INSERT_AT_BLOCK>>>\s*\n'
        r'file:\s*(?P<file>' + _FIELD_VALUE + r')\s*\n'
        r'anchor:\s*(?P<anchor>' + _FIELD_VALUE + r')\s*\n'
        r'placement:\s*(?P<placement>before|after)\s*\n'
        r'content:' + _BODY_SEP + r'(?P<content>' + _TEMPERED_CONTENT + r')'
        r'<<<END_INSERT_AT_BLOCK>>>',
        re.DOTALL,
    ),
    # Line-coordinate operations — used by scanner-driven autofix paths
    # whose findings carry start.line / end.line metadata (semgrep
    # extra.fix, ESLint rules, etc.). The LLM may also emit these
    # directly when it sees a line-numbered file view in its prompt and
    # decides a coordinate is more reliable than picking an anchor.
    #
    # Both ops support an optional ``hash:`` field carrying the sha256
    # hex digest of the file the caller saw. The patcher rejects the
    # patch when the on-disk hash differs — line numbers go stale fast
    # if a sibling patch in the same round mutated the file first.
    OperationType.INSERT_AT_LINE: re.compile(
        r'<<<INSERT_AT_LINE>>>\s*\n'
        r'file:\s*(?P<file>' + _FIELD_VALUE + r')\s*\n'
        r'line:\s*(?P<line>\d+)\s*\n'
        r'(?:hash:\s*(?P<hash>[0-9a-fA-F]+)\s*\n)?'
        r'content:' + _BODY_SEP + r'(?P<content>' + _TEMPERED_CONTENT + r')'
        r'<<<END_INSERT_AT_LINE>>>',
        re.DOTALL,
    ),
    OperationType.REPLACE_LINE_RANGE: re.compile(
        r'<<<REPLACE_LINE_RANGE>>>\s*\n'
        r'file:\s*(?P<file>' + _FIELD_VALUE + r')\s*\n'
        r'start_line:\s*(?P<line>\d+)\s*\n'
        r'end_line:\s*(?P<end_line>\d+)\s*\n'
        r'(?:hash:\s*(?P<hash>[0-9a-fA-F]+)\s*\n)?'
        r'content:' + _BODY_SEP + r'(?P<content>' + _TEMPERED_CONTENT + r')'
        r'<<<END_REPLACE_LINE_RANGE>>>',
        re.DOTALL,
    ),
}


# Kinds the parse-miss diagnostic looks for. Kept in sync with
# ``_BLOCK_PATTERNS`` keys — if a new op type is added above, add its
# short name here so ``summarize_parse_miss`` can report on it.
_PARSE_MISS_MARKERS: tuple[str, ...] = (
    "REPLACE_BLOCK",
    "CREATE_FILE",
    "REWRITE_FILE",
    "DELETE_BLOCK",
    "INSERT_AT_BLOCK",
    "INSERT_AT_LINE",
    "REPLACE_LINE_RANGE",
)


def summarize_parse_miss(llm_output: str) -> str:
    """Explain a ``parse_patch_blocks(...) == []`` result when patch
    markers were actually present in the raw output.

    Called by patching / repair paths when ``parse_patch_blocks``
    returns zero blocks. If the output contains no known ``<<<XYZ>>>``
    openers at all, returns ``""`` — the LLM legitimately emitted no
    patches and the caller should log the usual "no patches" line. If
    openers ARE present, returns a short, LLM-facing string naming
    what was seen and hinting at the most common cause so the retry
    prompt can echo it as a directive.

    The two shapes we specifically call out because the LLM has been
    observed to emit them under output-token pressure and both cause
    silent parser drops:

      * ``content:<body>`` on the same line (the finsearch 156032347
        signature — 600/610 blocks dropped this way in one round).
      * A block opener with no matching closer, cut off by the model's
        output-token cap (``finish: length``). The tail block's opener
        counts but its closer never appears.

    The returned string is short by design — it goes into a system
    message the LLM sees next round, and long context there hurts more
    than it helps.
    """
    counts: dict[str, tuple[int, int]] = {}
    for name in _PARSE_MISS_MARKERS:
        opens = llm_output.count(f"<<<{name}>>>")
        closes = llm_output.count(f"<<<END_{name}>>>")
        if opens or closes:
            counts[name] = (opens, closes)
    if not counts:
        return ""

    total_opens = sum(o for o, _ in counts.values())
    total_closes = sum(c for _, c in counts.values())
    kinds = ", ".join(
        f"{k}={o}/{c}" for k, (o, c) in sorted(counts.items())
    )
    hints: list[str] = []
    # Unclosed tail block: model hit output-token cap mid-block.
    if total_opens > total_closes:
        hints.append(
            f"{total_opens - total_closes} opener(s) with no matching "
            "closer — likely truncated by the output-token cap "
            "(finish=length). Split the batch or use smaller blocks."
        )
    # Inline body sniff: closers are all paired but zero blocks parsed.
    # Almost always the ``content:<body>`` / ``search:<body>`` shape;
    # sample the first such occurrence for the LLM.
    inline_sample = ""
    if total_opens == total_closes and total_opens > 0:
        # Find first "content:<non-newline>" or "search:<non-newline>"
        # to give the LLM a concrete example of what to fix.
        m = re.search(r'(content|search|replace):([^\n]{1,60})', llm_output)
        if m and m.group(2).strip():
            inline_sample = f"{m.group(1)}:{m.group(2).strip()[:40]}"
            hints.append(
                "Body field starts on the same line as its label — e.g. "
                f"`{inline_sample}`. Put the body on the LINE AFTER the "
                "`content:` / `search:` / `replace:` label."
            )
    hint_str = " ".join(hints) if hints else (
        "Markers present but no block parsed — check field order and "
        "that each block has file/content pairs before its END marker."
    )
    return f"Markers seen: {kinds}. {hint_str}"


def parse_patch_blocks(llm_output: str) -> list[PatchBlock]:
    """
    Extract all SEARCH/REPLACE patch blocks from an LLM's output text.

    Parses the strict tagged syntax:
        <<<REPLACE_BLOCK>>>
        file: path/to/file.ext
        search:
        <exact lines to find>
        replace:
        <exact replacement lines>
        <<<END_REPLACE_BLOCK>>>

    Supports all six operation types: REPLACE_BLOCK, CREATE_FILE,
    DELETE_BLOCK, INSERT_AT_BLOCK, INSERT_AT_LINE, REPLACE_LINE_RANGE.

    Args:
        llm_output: The full text response from the LLM.

    Returns:
        List of PatchBlock objects in the order they appear in the text.
    """
    blocks: list[PatchBlock] = []

    for op_type, pattern in _BLOCK_PATTERNS.items():
        for match in pattern.finditer(llm_output):
            raw = match.group(0)
            gd = match.groupdict()

            if op_type == OperationType.REPLACE_BLOCK:
                blocks.append(PatchBlock(
                    operation=OperationType.REPLACE_BLOCK,
                    file=gd["file"].strip(),
                    search=gd["search"].rstrip(),
                    replace=gd["replace"].rstrip(),
                    count=(gd.get("count") or "unique").strip().lower(),
                    raw_block=raw,
                ))
            elif op_type == OperationType.CREATE_FILE:
                blocks.append(PatchBlock(
                    operation=OperationType.CREATE_FILE,
                    file=gd["file"].strip(),
                    content=gd["content"].rstrip(),
                    raw_block=raw,
                ))
            elif op_type == OperationType.REWRITE_FILE:
                blocks.append(PatchBlock(
                    operation=OperationType.REWRITE_FILE,
                    file=gd["file"].strip(),
                    content=gd["content"].rstrip(),
                    raw_block=raw,
                ))
            elif op_type == OperationType.DELETE_BLOCK:
                blocks.append(PatchBlock(
                    operation=OperationType.DELETE_BLOCK,
                    file=gd["file"].strip(),
                    search=gd["search"].rstrip(),
                    count=(gd.get("count") or "unique").strip().lower(),
                    raw_block=raw,
                ))
            elif op_type == OperationType.INSERT_AT_BLOCK:
                placement = (
                    Placement.BEFORE
                    if gd["placement"].strip().lower() == "before"
                    else Placement.AFTER
                )
                blocks.append(PatchBlock(
                    operation=OperationType.INSERT_AT_BLOCK,
                    file=gd["file"].strip(),
                    anchor=gd["anchor"].strip(),
                    placement=placement,
                    content=gd["content"].rstrip(),
                    raw_block=raw,
                ))
            elif op_type == OperationType.INSERT_AT_LINE:
                blocks.append(PatchBlock(
                    operation=OperationType.INSERT_AT_LINE,
                    file=gd["file"].strip(),
                    line=int(gd["line"]),
                    expected_file_hash=(gd.get("hash") or "").strip().lower(),
                    content=gd["content"].rstrip(),
                    raw_block=raw,
                ))
            elif op_type == OperationType.REPLACE_LINE_RANGE:
                blocks.append(PatchBlock(
                    operation=OperationType.REPLACE_LINE_RANGE,
                    file=gd["file"].strip(),
                    line=int(gd["line"]),
                    end_line=int(gd["end_line"]),
                    expected_file_hash=(gd.get("hash") or "").strip().lower(),
                    content=gd["content"].rstrip(),
                    raw_block=raw,
                ))

    # Sort blocks by their position in the original text to preserve ordering
    blocks.sort(key=lambda b: llm_output.find(b.raw_block) if b.raw_block else 0)
    return blocks


# ---------------------------------------------------------------------------
# 3. File Extension → Language Mapping for Tree-Sitter
# ---------------------------------------------------------------------------

# Maps file extensions to tree-sitter language names.
# Tree-sitter grammars are loaded lazily on first use.
_EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".java": "java",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".sql": "sql",
    ".css": "css",
    ".html": "html",
    ".sh": "bash",
    ".bash": "bash",
}


def get_language_for_file(filepath: str) -> Optional[str]:
    """Determine the tree-sitter language for a given file by its extension."""
    ext = os.path.splitext(filepath)[1].lower()
    return _EXTENSION_LANGUAGE_MAP.get(ext)


# ---------------------------------------------------------------------------
# 4. Async File I/O Helpers (aiofiles)
# ---------------------------------------------------------------------------

def _is_reparse_or_symlink(filepath: str) -> bool:
    """True when ``filepath`` exists and is a symlink (POSIX or Windows) OR
    a Windows reparse point (directory junction / mount point).

    ``os.path.islink`` returns False for Windows directory junctions created
    with ``mklink /J`` — they're reparse points but not symlinks. An LLM
    that staged a junction inside the workspace pointing at, e.g.,
    ``C:\\Users\\<user>\\AppData`` would be able to write outside the
    workspace through it, since the patcher uses ``os.replace`` which
    follows reparse points. Checking the file-attribute reparse-point bit
    via ``stat.FILE_ATTRIBUTE_REPARSE_POINT`` (0x0400) closes that gap.
    """
    try:
        if not os.path.lexists(filepath):
            return False
    except OSError:
        return False
    if os.path.islink(filepath):
        return True
    if os.name == "nt":
        try:
            st = os.lstat(filepath)
        except OSError:
            return False
        # FILE_ATTRIBUTE_REPARSE_POINT = 0x0400. Present on the stat result
        # as st_file_attributes on Windows (Python 3.5+). Anything with the
        # reparse-point bit set — junction, mount point, symlink — is
        # blocked. Standard files / dirs have the bit clear.
        attrs = getattr(st, "st_file_attributes", 0)
        if attrs & 0x0400:
            return True
    return False


async def _aread(filepath: str) -> str:
    """Read a file asynchronously using aiofiles."""
    try:
        import aiofiles
        async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
            return await f.read()
    except ImportError:
        logger.debug("[patcher] aiofiles not installed. Falling back to sync open().")
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()


async def _awrite(filepath: str, content: str) -> None:
    """
    Atomically write ``content`` to ``filepath``.

    Strategy: write to a sibling temp file in the same directory, fsync, then
    `os.replace` (atomic on POSIX and Windows). A crash or kill mid-write
    leaves the original file unchanged; partial writes never become visible.
    Previously this used a plain ``open(filepath, "w")`` which truncated the
    file *before* writing — a crash between truncate and final flush left
    the user's file empty.

    P1.2: refuse to write through a symlink at ``filepath`` (Linux/macOS).
    ``safe_resolve`` already rejects symlinks that escape the workspace,
    but an attacker could pre-stage a symlink ``pyproject.toml ->
    ~/.ssh/authorized_keys`` (or any in-workspace dotfile) and ride the
    LLM's first patch onto a sensitive target. ``lstat`` + ``O_NOFOLLOW``
    closes that path before the atomic rename takes effect. On Windows
    we additionally check for directory junctions (``mklink /J``) via
    the reparse-tag bit — ``os.path.islink`` only flags true symlinks
    and would miss junctions, leaving a workspace-escape primitive open.
    """
    import tempfile

    # P1.2: reject symlinks AND Windows reparse points (junctions, mount
    # points) at the destination BEFORE we set up the temp file. On
    # Linux/macOS the islink + O_NOFOLLOW pair is enough. On Windows
    # os.path.islink returns False for directory junctions; check the
    # reparse-point bit explicitly so we don't traverse them.
    if _is_reparse_or_symlink(filepath):
        raise PermissionError(
            f"[patcher] Refusing to write through symlink / reparse point: "
            f"{filepath!r}. This usually indicates a path-traversal attempt "
            f"or a stale dev symlink/junction — delete it and retry."
        )
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is not None and os.path.lexists(filepath):
        try:
            check_fd = os.open(filepath, os.O_RDONLY | nofollow)
        except OSError as exc:
            # ELOOP: target is a symlink. Anything else (permissions,
            # missing file) is fine to let through to the existing write
            # path. Using ``errno.ELOOP`` covers Linux (40), BSD/macOS
            # (62), and any future platform the hard-coded tuple missed.
            if getattr(exc, "errno", None) == errno.ELOOP:
                raise PermissionError(
                    f"[patcher] O_NOFOLLOW check tripped on {filepath!r}: "
                    f"target resolves through a symlink ({exc})."
                ) from exc
        else:
            os.close(check_fd)

    directory = os.path.dirname(os.path.abspath(filepath)) or "."
    # delete=False so we own cleanup; suffix keeps editor file-watchers happy.
    # Cap suffix length so an unusually long filename (e.g. a 250-char
    # generated test name) can't push the tmpfile past NAME_MAX (255 on
    # ext4/POSIX). The suffix is purely informational; truncating it
    # keeps editor watchers happy without risking ENAMETOOLONG.
    base = os.path.basename(filepath)
    safe_suffix = base[-32:] if len(base) > 32 else base
    fd, tmp_path = tempfile.mkstemp(
        prefix=".harness.tmp.",
        suffix=safe_suffix,
        dir=directory,
    )
    try:
        try:
            import aiofiles
            # Close the os-level fd we don't need; aiofiles will open by path.
            os.close(fd)
            async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
                await f.write(content)
                await f.flush()
        except ImportError:
            logger.debug("[patcher] aiofiles not installed. Falling back to sync write.")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass  # fsync not supported on some filesystems
        os.replace(tmp_path, filepath)
    except Exception:
        # Clean up the temp file on any failure so we don't leak it.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# 5. BasePatcher ABC
# ---------------------------------------------------------------------------

# Delegate path-traversal checking to the central trust boundary module.
# The local name is preserved so that existing callers (tests, internal
# code) continue to work without changes.
from harness.trust import safe_resolve as _safe_resolve  # noqa: E402


# ---------------------------------------------------------------------------
# Duplicate-test-root detector
# ---------------------------------------------------------------------------
#
# Motivating failure: finsearch STORY-038 batch had the LLM create two copies
# of ``test_config.py`` — one at ``tests/unit/config/test_config.py`` and
# a second at ``server/tests/unit/config/test_config.py`` — with contradictory
# assertions on ``server.config.database_url``. The repair loop oscillated
# ``server/config.py`` between the two expected values for 7 rounds before
# the stuck-file HITL fired.
#
# This detector fires at the patcher's CREATE_FILE / REWRITE_FILE entry.
# When the LLM tries to land a file whose test-scoped path suffix already
# exists at a different tests root, the patcher rejects with a
# NEXT-ROUND DIRECTIVE naming both paths so the LLM sees the collision
# in its next repair round.

# Only fire on files that look like test suites. Everything else can
# legitimately duplicate across roots (`__init__.py`, `conftest.py`,
# `README.md`, `.gitignore`).
_TEST_BASENAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^test_[^/\\]+\.py$"),
    re.compile(r"^[^/\\]+_test\.py$"),
    re.compile(r"^[^/\\]+\.test\.[tj]sx?$"),
    re.compile(r"^[^/\\]+\.spec\.[tj]sx?$"),
)

# Directory segments that mark a "tests root" — the ancestor whose
# subtree carries a mirrored test tree.
_TESTS_ROOT_SEGMENTS: frozenset[str] = frozenset({
    "tests", "test", "__tests__",
})

# Directories the workspace walk should NOT descend into. Mirrors
# ``harness/graph.py``'s ``_HITL_SKIP_DIRS`` so this walk stays fast on
# realistic monorepos.
_DUP_CHECK_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", "dist", "build", ".next", ".nuxt", "target", ".cache",
    "htmlcov", "site-packages",
})


def _is_test_basename(basename: str) -> bool:
    """True when ``basename`` matches one of the test-file naming patterns
    the duplicate check applies to. Non-test files skip the check
    entirely."""
    return any(p.match(basename) for p in _TEST_BASENAME_PATTERNS)


def _extract_test_scoped_suffix(rel_path: str) -> Optional[str]:
    """Return the substring of ``rel_path`` starting at the leftmost
    ``tests`` / ``test`` / ``__tests__`` segment. When no such segment
    is present, returns ``None`` — the path isn't under a tests root
    and the duplicate check is skipped. Comparison uses ``/`` as the
    separator regardless of platform so Windows paths match POSIX ones."""
    parts = re.split(r"[/\\]", rel_path)
    for i, seg in enumerate(parts):
        if seg in _TESTS_ROOT_SEGMENTS:
            return "/".join(parts[i:])
    return None


def _detect_duplicate_test_root(
    new_rel_path: str, workspace_root: str,
) -> Optional[str]:
    """Return a rejection message when ``new_rel_path`` would create a
    test file that duplicates one already under a DIFFERENT tests root
    in the same workspace. Returns ``None`` when the check doesn't
    apply (non-test basename, no tests root in the path) or no match is
    found.

    Matching rule: the test-scoped path suffix — everything from the
    leftmost ``tests``/``test``/``__tests__`` segment onward — must be
    byte-identical between the new path and the existing path, and the
    two paths must differ only in what comes BEFORE that segment. That's
    the exact fingerprint of "duplicate test tree under a subdir" (root
    ``tests/`` vs ``server/tests/``) — the kind that produces
    contradictory assertions on the same symbol.

    Legitimate duplication across multiple language stacks (e.g. Python
    ``tests/foo.py`` and Node ``client/tests/foo.spec.ts``) is NOT
    caught: the basename patterns are language-specific, and the
    suffixes wouldn't match anyway. Legitimate duplication of non-test
    files (``__init__.py``, ``conftest.py``, ``README.md``) is skipped
    at the ``_is_test_basename`` gate.
    """
    basename = os.path.basename(new_rel_path)
    if not _is_test_basename(basename):
        return None
    new_suffix = _extract_test_scoped_suffix(new_rel_path)
    if new_suffix is None:
        return None
    if not os.path.isdir(workspace_root):
        return None
    # Normalise for suffix comparison — POSIX separators throughout.
    new_rel_norm = re.sub(r"[/\\]", "/", new_rel_path).lstrip("/")
    duplicates: list[str] = []
    for root, dirs, files in os.walk(workspace_root, followlinks=False):
        # Prune skip dirs in-place so the walk stays cheap.
        dirs[:] = [d for d in dirs if d not in _DUP_CHECK_SKIP_DIRS]
        if basename not in files:
            continue
        candidate_abs = os.path.join(root, basename)
        candidate_rel = os.path.relpath(candidate_abs, workspace_root)
        candidate_norm = re.sub(r"[/\\]", "/", candidate_rel).lstrip("/")
        if candidate_norm == new_rel_norm:
            continue  # same path — that's the ``already exists`` case, not a duplicate root
        candidate_suffix = _extract_test_scoped_suffix(candidate_norm)
        if candidate_suffix == new_suffix:
            duplicates.append(candidate_norm)
        if len(duplicates) >= 3:
            break  # bound the message payload
    if not duplicates:
        return None
    dup_list = ", ".join(f"`{p}`" for p in duplicates)
    return (
        f"DUPLICATE_TEST_ROOT: refusing to land `{new_rel_norm}` because "
        f"another test file with the same test-scoped path already exists: "
        f"{dup_list}. The workspace has two mirrored tests trees for the "
        "same target module — this is a topology bug that produces "
        "contradictory test expectations and traps the repair loop in a "
        "REPLACE_BLOCK oscillation on the shared implementation file "
        "(observed in finsearch STORY-038: 7 rounds toggling "
        "`server/config.py::database_url` before HITL). "
        "NEXT-ROUND DIRECTIVE — pick ONE of these three:\n"
        "  (a) COVERED BY EXISTING FILE. If the existing file at "
        f"{dup_list} already covers the assertions you wanted this new "
        "file for, do nothing — no CREATE_FILE, no REWRITE_FILE. Move "
        "on to the next diagnostic.\n"
        "  (b) GENUINELY DIFFERENT TESTS — use a different basename or "
        "path. If your new file tests something different (different "
        "AC IDs, different scenarios, different subsystem), rename it "
        "so pytest collects both without the topology collision. "
        f"Examples: `{new_rel_norm.replace('test_', 'test_env_', 1) if '/test_' in new_rel_norm.replace(os.sep, '/') or new_rel_norm.startswith('test_') else new_rel_norm}` "
        "(prefix), or move it under a different scope directory that "
        "makes its purpose obvious. Do NOT re-emit CREATE_FILE with the "
        "same path — the patcher will reject again.\n"
        "  (c) INTENTIONAL REPLACEMENT of the older file. If the file "
        f"at {dup_list} is stale / wrong and you want this new file to "
        "supersede it, first DELETE_BLOCK the entire contents of the "
        "existing file (or REWRITE_FILE it to just a comment noting the "
        "consolidation), then re-emit this CREATE_FILE. Only pick this "
        "path if you're confident the OLDER file is wrong."
    )


SEARCH_BLOCK_COPY_RULES = (
    "IMPORTANT — how to copy from a `  N| ` line-numbered view into a "
    "SEARCH block:\n"
    "  - Strip ONLY the `  N| ` line-number prefix. Every other "
    "character on the line — INCLUDING leading spaces, leading tabs, "
    "trailing punctuation, and blank lines — is part of the file and "
    "MUST appear VERBATIM in your SEARCH.\n"
    "  - Do NOT start a SEARCH mid-line or mid-word. Copy WHOLE lines "
    "end-to-end; a search that begins after a real line's leading "
    "whitespace (or after a few words on that line) will not match "
    "and the patch will be REJECTED.\n"
    "  - Worked example. If the view shows this line exactly:\n"
    "        21|         All outgoing requests carry the required "
    "User-Agent header and are\n"
    "    then a SEARCH containing `User-Agent header and are` will "
    "FAIL — it drops the eight leading spaces plus five words. The "
    "correct SEARCH content for that line is:\n"
    "        `        All outgoing requests carry the required "
    "User-Agent header and are`\n"
    "    (eight leading spaces, full sentence, no `21|` prefix).\n"
    "  - If a whole-line copy of what you want to replace is more "
    "than ~10 lines, prefer a single `REWRITE_FILE` over a "
    "REPLACE_BLOCK — the copy-fidelity risk grows with SEARCH size."
)
"""Tightened copy rules for LLM SEARCH-block emission. The single
source of truth — shared by every path that shows the LLM a
line-numbered file view: ``graph._resolve_read_blocks``,
``graph._format_preflight_file_content``,
``graph._format_current_file_content``, the auto-inject recovery
wrappers, and the patcher's own SEARCH-miss / CREATE_FILE-collision
rejection messages.

Empirical trigger: finsearch session finsearch-optB-1783830081
(2026-07-12) rejected multiple REPLACE_BLOCK edits at ~55% search
similarity because the LLM emitted mid-line fragments after
stripping the ``  N| `` prefix AND the file's real leading
whitespace. The prior intros (``"WITHOUT the `  N| ` prefix"``) did
not distinguish "strip the prefix" from "strip the leading
whitespace"; these tightened rules make that boundary explicit and
give the LLM one worked example against a concrete rejected
pattern. Owned by patcher.py so ``graph.py`` can import it without
a circular dependency."""


class BasePatcher(ABC):
    """
    Abstract base for all file modification engines.

    The four canonical operations are:
        - create_file(path, content) → PatchResult
        - replace_block(path, search, replace) → PatchResult
        - delete_block(path, search) → PatchResult
        - insert_at_block(path, anchor, placement, content) → PatchResult
    """

    workspace_root: str  # set by concrete subclasses

    def _resolve_safe(
        self, filepath: str, op: "OperationType"
    ) -> Union[str, "PatchResult"]:
        """
        Resolve ``filepath`` against ``self.workspace_root`` with traversal
        protection. Returns the resolved absolute path on success, or a
        ``PatchResult`` carrying an error to propagate. Callers should
        ``isinstance``-check the return to narrow the union.
        """
        try:
            return _safe_resolve(self.workspace_root, filepath)
        except ValueError as exc:
            return PatchResult(
                success=False,
                file=filepath,
                operation=op,
                error=f"path traversal rejected: {exc}",
            )

    @abstractmethod
    async def create_file(self, filepath: str, content: str) -> PatchResult:
        """
        Create a new file with the given content.

        Idempotent: if the file already exists with byte-identical content,
        returns success with a no-op message (resume-safe). If the file
        exists with different content, returns an error.
        """
        ...

    @abstractmethod
    async def replace_block(
        self, filepath: str, search: str, replace: str,
        *, count: str = "unique",
    ) -> PatchResult:
        """Replace an exact-match block of text within an existing file.

        ``count`` is the match-count policy: ``"unique"`` errors on >1 match
        (historical default), ``"all"`` replaces every occurrence, ``"first"``
        replaces only the first.
        """
        ...

    @abstractmethod
    async def delete_block(
        self, filepath: str, search: str, *, count: str = "unique",
    ) -> PatchResult:
        """Delete an exact-match block of text from an existing file.

        ``count`` has the same semantics as ``replace_block``.
        """
        ...

    @abstractmethod
    async def insert_at_block(
        self, filepath: str, anchor: str, placement: Placement, content: str
    ) -> PatchResult:
        """
        Insert content before or after a named structural block (function, class, etc.)
        identified by anchor string.
        """
        ...

    @abstractmethod
    async def insert_at_line(
        self, filepath: str, line: int, content: str,
        *, expected_file_hash: str = "",
    ) -> PatchResult:
        """Insert content as new line(s) at 1-based line ``line``.

        After the patch, ``content`` becomes line ``line`` and the
        previous line ``line`` is shifted to ``line + 1``. ``line == 1``
        prepends; ``line == len(file_lines) + 1`` appends. When
        ``expected_file_hash`` is supplied and does not match the file's
        current sha256, the patch is rejected (the line number is stale).
        """
        ...

    @abstractmethod
    async def replace_line_range(
        self, filepath: str, start_line: int, end_line: int, content: str,
        *, expected_file_hash: str = "",
    ) -> PatchResult:
        """Replace lines ``[start_line, end_line]`` (inclusive, 1-based)
        with ``content``. Hash-guarded the same way as ``insert_at_line``.
        Idempotent: when the target range already equals ``content``
        verbatim the patch returns a no-op success.
        """
        ...


# ---------------------------------------------------------------------------
# 6. TextPatcher — Pure Exact-Match Search & Replace (aiofiles-backed)
# ---------------------------------------------------------------------------

class TextPatcher(BasePatcher):
    """
    Pure text-based SEARCH/REPLACE engine with exact-match semantics.

    Works on any text file. Operations are atomic:
      - search string must match exactly once (ambiguity is an error).
      - The file is read and written using aiofiles for async I/O.

    Falls back to sync open() if aiofiles is not installed.
    """

    def __init__(self, workspace_root: str):
        self.workspace_root = os.path.abspath(workspace_root)

    async def create_file(self, filepath: str, content: str) -> PatchResult:
        full_path = self._resolve_safe(filepath, OperationType.CREATE_FILE)
        if isinstance(full_path, PatchResult):
            return full_path

        # Duplicate-test-root check — only fires when the file is a
        # test-like basename AND its test-scoped suffix already exists
        # under a DIFFERENT tests root in the workspace. Returns None
        # (falls through) for every non-test file, every path outside a
        # tests root, and every non-duplicate. See
        # ``_detect_duplicate_test_root`` for the full rationale.
        dup_err = _detect_duplicate_test_root(filepath, self.workspace_root)
        if dup_err:
            return PatchResult(
                success=False,
                file=filepath,
                operation=OperationType.CREATE_FILE,
                error=dup_err,
            )

        # Idempotency: if the file already exists with byte-identical content,
        # treat as a successful no-op so a crash-then-resume of the same patch
        # batch doesn't fail with "File already exists". Different content
        # remains a hard error — that's genuinely unsafe to overwrite blindly.
        expected = content + "\n"
        if os.path.exists(full_path):
            try:
                actual = await _aread(full_path)
            except OSError as exc:
                return PatchResult(
                    success=False,
                    file=filepath,
                    operation=OperationType.CREATE_FILE,
                    error=f"File exists but unreadable: {exc}",
                )
            if actual == expected:
                logger.info(
                    "[patcher:text] CREATE_FILE no-op: %s already at target state (resume-safe).",
                    filepath,
                )
                return PatchResult(
                    success=True,
                    file=filepath,
                    operation=OperationType.CREATE_FILE,
                    message=f"already at target state (no-op on resume): {filepath}",
                    lines_changed=0,
                    no_op=True,
                )
            # Empty / whitespace-only file: treat CREATE_FILE as a
            # first-write, not a conflict. This is the DELETE_BLOCK +
            # CREATE_FILE trap the LLM falls into after failing REPLACE_
            # BLOCK twice: the harness's own directive tells it to use
            # DELETE_BLOCK to clear the file, then CREATE_FILE the new
            # content. The DELETE lands (file is now ``\n``), then
            # CREATE_FILE rejects on "different content" because empty
            # != new — net result is an empty file and a wasted round.
            # Session b61f48a7 spent 3+ HITL cycles on
            # ``backend/api/search.py`` in this exact loop.
            # An empty file has no operator-authored content to preserve,
            # so overwriting it is safe and the correct end-state anyway.
            if not actual.strip():
                logger.info(
                    "[patcher:text] CREATE_FILE overwriting empty/whitespace-only "
                    "file %s (no content to preserve).", filepath,
                )
                try:
                    await _awrite(full_path, expected)
                    lines_added = content.count("\n") + 1
                    return PatchResult(
                        success=True,
                        file=filepath,
                        operation=OperationType.CREATE_FILE,
                        message=(
                            f"Created {filepath} ({lines_added} lines; "
                            "overwrote empty file)"
                        ),
                        lines_changed=lines_added,
                    )
                except OSError as exc:
                    return PatchResult(
                        success=False,
                        file=filepath,
                        operation=OperationType.CREATE_FILE,
                        error=str(exc),
                    )
            # File exists with different-but-similar content — promote
            # to REWRITE_FILE. In headless story-mode the LLM re-emits
            # CREATE_FILE for the same test file across repair rounds
            # (finsearch session 156032347: server/app/tests/test_edgar_
            # client.py rejected 4 times, test_rate_limiter.py 3 times,
            # test_company_service.py 3 times — 32 total rejections in
            # one run). Each rejection burned a repair round while the
            # LLM's intent was clearly "overwrite this file I already
            # created earlier." The hard-reject was originally there to
            # prevent blindly clobbering unrelated operator content;
            # that safety concern only applies when the on-disk content
            # is *unrelated* to what the LLM is emitting. A high
            # similarity ratio (>= 0.85) means the LLM is targeting the
            # same conceptual file with modifications, and REWRITE_FILE
            # is the correct escape hatch. Post-patch parse validation
            # (HybridPatcher._validate_and_maybe_rollback) still fires
            # on the promoted write, so a syntactically-broken promotion
            # gets rolled back to `actual` — the safety net doesn't move.
            #
            # Threshold 0.85 keeps the existing safety tests rejecting
            # (max ratio 0.80 among unrelated content in the test suite)
            # while catching intra-session repairs (typically 0.9+).
            similarity = difflib.SequenceMatcher(None, actual, expected).ratio()
            if similarity >= 0.85:
                logger.info(
                    "[patcher:text] CREATE_FILE auto-promoted to REWRITE_FILE "
                    "for %s (content similarity=%.3f >= 0.85). LLM re-emitted "
                    "same-file content across rounds; post-patch parse "
                    "validation still applies to catch broken promotions.",
                    filepath, similarity,
                )
                promoted = await self.rewrite_file(filepath, content)
                if promoted.success:
                    # Preserve the CREATE_FILE operation identity in the
                    # result so upstream accounting (patch counters,
                    # traceability markers) doesn't see this as a
                    # separately-emitted REWRITE_FILE — from the LLM's
                    # perspective it still asked for CREATE_FILE.
                    return PatchResult(
                        success=True,
                        file=filepath,
                        operation=OperationType.CREATE_FILE,
                        message=(
                            f"Promoted CREATE_FILE to REWRITE_FILE "
                            f"({similarity:.2f} similarity): "
                            f"{promoted.message or filepath}"
                        ),
                        lines_changed=promoted.lines_changed,
                    )
                # Fall through to hard-reject on rewrite failure so the
                # LLM still gets the recovery directive it needs.
            # File exists with different content — surface the FULL
            # current content (line-numbered, whole-file mode when small)
            # so the LLM's next round can emit a REPLACE_BLOCK against
            # the real starting point instead of guessing. Without this,
            # a 200-char snippet is useless for anything but the shortest
            # files and the repair loop burns rounds hallucinating
            # search-blocks. Reuses _find_closest_match's whole-file /
            # window logic by passing `actual` as its own search anchor —
            # that returns the same line-numbered rendering used by the
            # REPLACE_BLOCK-not-found error, keeping the two error shapes
            # symmetric so the LLM's parsing prompt handles both.
            annotated = _find_closest_match(actual, actual)
            return PatchResult(
                success=False,
                file=filepath,
                operation=OperationType.CREATE_FILE,
                error=(
                    f"File already exists with different content: "
                    f"{filepath}. The patcher will NOT overwrite blindly. "
                    f"NEXT-ROUND DIRECTIVE: switch operations. Do NOT "
                    f"emit another CREATE_FILE for this path — the "
                    f"patcher will reject the same shape again. Emit a "
                    f"REPLACE_BLOCK against the current content shown "
                    f"below. If you intended to add to it, use "
                    f"REPLACE_BLOCK to insert around an existing anchor "
                    f"line. If your intended content actually matches "
                    f"what's here, no patch is needed — move on to the "
                    f"next diagnostic.\n\n"
                    f"{SEARCH_BLOCK_COPY_RULES}\n\n"
                    f"Current file content:\n{annotated}"
                ),
            )

        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            await _awrite(full_path, expected)
            lines_added = content.count("\n") + 1
            logger.info("[patcher:text] Created file: %s (%d lines)", filepath, lines_added)
            return PatchResult(
                success=True,
                file=filepath,
                operation=OperationType.CREATE_FILE,
                message=f"Created {filepath} ({lines_added} lines)",
                lines_changed=lines_added,
            )
        except OSError as exc:
            return PatchResult(
                success=False,
                file=filepath,
                operation=OperationType.CREATE_FILE,
                error=str(exc),
            )

    async def rewrite_file(self, filepath: str, content: str) -> PatchResult:
        """Overwrite ``filepath`` with ``content``. Unlike ``create_file``,
        this DOES clobber an existing file — that's the whole point of the
        escape hatch. Post-patch parse validation still applies (see
        ``HybridPatcher._validate_and_maybe_rollback``), so a REWRITE_FILE
        that produces broken syntax is rolled back to the pre-patch state
        just like any other op.

        Also idempotent: writing byte-identical content is a successful
        no-op (matches ``create_file``'s resume semantics).
        """
        full_path = self._resolve_safe(filepath, OperationType.REWRITE_FILE)
        if isinstance(full_path, PatchResult):
            return full_path
        # Duplicate-test-root check — only applies when this REWRITE_FILE
        # would create a NEW file (target doesn't yet exist). If the file
        # already exists, REWRITE_FILE is legitimately modifying it in
        # place and duplicate-across-roots isn't a concern. See
        # ``_detect_duplicate_test_root`` for the full rationale.
        if not os.path.exists(full_path):
            dup_err = _detect_duplicate_test_root(filepath, self.workspace_root)
            if dup_err:
                return PatchResult(
                    success=False,
                    file=filepath,
                    operation=OperationType.REWRITE_FILE,
                    error=dup_err,
                )
        expected = content + "\n"
        if os.path.exists(full_path):
            try:
                actual = await _aread(full_path)
            except OSError as exc:
                return PatchResult(
                    success=False,
                    file=filepath,
                    operation=OperationType.REWRITE_FILE,
                    error=f"File exists but unreadable: {exc}",
                )
            if actual == expected:
                # Report as failure with an actionable message so the
                # LLM sees it in the patch-failure surface next round.
                # Unlike ``CREATE_FILE`` no-op (which happens on
                # resume/idempotency and IS success), a ``REWRITE_FILE``
                # no-op means the LLM deliberately generated content
                # that matched disk byte-for-byte — it thought it was
                # fixing something and did nothing. Silently marking it
                # success hid the signal: session b9369w5uu (ciod) had
                # the LLM emit the SAME wrong content for
                # ``server/models/__init__.py`` two rounds running while
                # the judge kept flagging a missing symbol. Surfacing
                # the no-op as an actionable failure gives the LLM a
                # concrete "you're stuck in a loop" nudge.
                logger.info(
                    "[patcher:text] REWRITE_FILE no-op signaled as failure: "
                    "%s content byte-identical to disk. LLM will see the "
                    "'you emitted what's already there' hint next round.",
                    filepath,
                )
                return PatchResult(
                    success=False,
                    file=filepath,
                    operation=OperationType.REWRITE_FILE,
                    error=(
                        f"REWRITE_FILE no-op: the content you emitted for "
                        f"`{filepath}` is byte-identical to what's already "
                        "on disk. Your patch changed nothing. This usually "
                        "means one of two things:\n"
                        "  1) The file is already correct — the bug is "
                        "somewhere ELSE (a caller, an import site, a test "
                        "expectation). Re-read the judge's real_blocker "
                        "and look for a different target.\n"
                        "  2) You are stuck rewriting the same wrong "
                        "content — your mental model of what this file "
                        "should contain is out of date. Emit a READ_FILE "
                        "block on this file AND on any callers/importers "
                        "before your next patch attempt.\n"
                        "Do NOT emit REWRITE_FILE again with the exact "
                        "same content — that will be rejected as a no-op "
                        "again. Either change target, change content, or "
                        "READ_FILE first."
                    ),
                    lines_changed=0,
                    no_op=True,
                )
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            await _awrite(full_path, expected)
            new_lines = content.count("\n") + 1
            logger.info(
                "[patcher:text] Rewrote file: %s (%d lines)",
                filepath, new_lines,
            )
            return PatchResult(
                success=True,
                file=filepath,
                operation=OperationType.REWRITE_FILE,
                message=f"Rewrote {filepath} ({new_lines} lines)",
                lines_changed=new_lines,
            )
        except OSError as exc:
            return PatchResult(
                success=False,
                file=filepath,
                operation=OperationType.REWRITE_FILE,
                error=str(exc),
            )

    async def replace_block(
        self, filepath: str, search: str, replace: str,
        *, count: str = "unique",
    ) -> PatchResult:
        full_path = self._resolve_safe(filepath, OperationType.REPLACE_BLOCK)
        if isinstance(full_path, PatchResult):
            return full_path
        if not os.path.isfile(full_path):
            if not search.strip():
                logger.info(
                    "[patcher:text] %s missing and search is empty — "
                    "treating REPLACE_BLOCK as CREATE_FILE.", filepath,
                )
                return await self.create_file(filepath, replace)
            # List existing siblings with the same stem so the LLM
            # can tell whether it meant e.g. `jest.config.cjs` when
            # `jest.config.js` is what's on disk. Session 4d1f9e1c
            # bounced between `.js` and `.cjs` for 3+ rounds because
            # neither message named the actual file that exists.
            siblings = _list_stem_siblings(full_path)
            hint = ""
            if siblings:
                hint = (
                    f" Existing files with the same stem in that "
                    f"directory: {', '.join(siblings)}. If you meant "
                    f"one of those, REPLACE_BLOCK it directly (the "
                    f"file is not this exact extension)."
                )
            return PatchResult(
                success=False,
                file=filepath,
                operation=OperationType.REPLACE_BLOCK,
                error=(
                    f"File not found: {filepath}. Use CREATE_FILE for "
                    f"new files, NOT another REPLACE_BLOCK.{hint}"
                ),
            )

        # Structural-file guard: JSON/YAML/TOML edits via REPLACE_BLOCK
        # are inherently fragile — one misplaced brace, trailing comma,
        # or indent shift breaks the whole file, and the LLM's copy
        # fidelity degrades fast with line count. Finsearch session
        # 156032347 shipped 2 broken JSON patches (client/tsconfig.test
        # .json, config/health_score_benchmarks.json) via multi-line
        # REPLACE_BLOCK — each rolled back correctly but burned a
        # repair round. Route multi-line structural edits to REWRITE_
        # FILE instead, where the LLM emits the whole file and the
        # parser validates it in one shot.
        #
        # The 4-line threshold matches the patcher's own SEARCH_BLOCK_
        # COPY_RULES guidance ("prefer a single REWRITE_FILE over a
        # REPLACE_BLOCK — the copy-fidelity risk grows with SEARCH
        # size"): single-value edits (1-3 lines) stay on the fast path,
        # anything larger goes through the escape hatch.
        _lower_ext = filepath.lower()
        if (
            _lower_ext.endswith((".json", ".yaml", ".yml", ".toml"))
            and (search.count("\n") >= 4 or replace.count("\n") >= 4)
        ):
            return PatchResult(
                success=False,
                file=filepath,
                operation=OperationType.REPLACE_BLOCK,
                error=(
                    f"REPLACE_BLOCK on {filepath} rejected: multi-line "
                    f"REPLACE_BLOCK on structural files (JSON/YAML/TOML) "
                    f"is fragile — a misplaced brace, trailing comma, or "
                    f"indent shift breaks the whole file and the parser "
                    f"rolls back the whole patch. Use REWRITE_FILE with "
                    f"the complete intended file contents instead. Search "
                    f"had {search.count(chr(10)) + 1} lines; replace had "
                    f"{replace.count(chr(10)) + 1} lines; the threshold "
                    f"for structural files is 4 lines. Single-value edits "
                    f"(e.g. one line changed) still work via REPLACE_BLOCK."
                ),
            )

        try:
            original = await _aread(full_path)
        except OSError as exc:
            return PatchResult(success=False, file=filepath, operation=OperationType.REPLACE_BLOCK, error=str(exc))

        policy = (count or "unique").strip().lower()
        if policy not in {"unique", "all", "first"}:
            return PatchResult(
                success=False, file=filepath,
                operation=OperationType.REPLACE_BLOCK,
                error=(
                    f"Unknown count policy {count!r} on REPLACE_BLOCK for "
                    f"{filepath}. Use one of: unique, all, first."
                ),
            )

        # Exact match search — count occurrences
        n_matches = original.count(search)
        if n_matches == 0:
            # Idempotency: if the replacement text is already present in
            # the file (and the search text is gone), this REPLACE_BLOCK
            # was already applied — likely by an earlier run of the same
            # patch batch before the process crashed. Report success so
            # the resume continues cleanly. We require the replacement
            # to appear exactly once to avoid false positives where the
            # text happens to appear elsewhere.
            if replace and (
                original.count(replace) == 1
                or len(_whitespace_tolerant_match(original, replace)) == 1
                or len(_whitespace_tolerant_match(
                    original, replace, normalize=str.strip,
                )) == 1
            ):
                # Three tiers count as "already at target state":
                #   1. Exact-byte match (original behavior)
                #   2. Trailing-whitespace tolerant (CRLF / EOF newline drift)
                #   3. Full-strip tolerant (tab vs space leading-indent drift)
                # The full-strip tier covers the most common idempotent
                # re-emit failure: the LLM re-issues a Makefile / YAML /
                # Python patch whose ``replace`` content is logically
                # identical but uses different leading whitespace than the
                # file. Without tier 3, that re-emit looks like a search
                # miss and trips the repair loop on a non-bug.
                logger.info(
                    "[patcher:text] REPLACE_BLOCK no-op: %s already at target state (resume-safe).",
                    filepath,
                )
                return PatchResult(
                    success=True,
                    file=filepath,
                    operation=OperationType.REPLACE_BLOCK,
                    message=f"region already at target state (no-op on resume): {filepath}",
                    lines_changed=0,
                    no_op=True,
                )
            # Whitespace-tolerant fallback: try matching after rstrip-per-line
            # before declaring failure. Catches trailing-whitespace drift,
            # trailing-newline-only diff, and CRLF/LF mismatch — common LLM
            # mistakes on small files where exact-byte matching is brittle.
            ws_matches = _whitespace_tolerant_match(original, search)
            if len(ws_matches) == 1:
                modified = _whitespace_tolerant_replace(
                    original, search, replace, ws_matches[0],
                )
                lines_changed = _count_diff_lines(original, modified)
                try:
                    await _awrite(full_path, modified)
                    logger.info(
                        "[patcher:text] Replaced block in %s via whitespace-tolerant match "
                        "(%d lines changed). Exact byte search did not match — the LLM's "
                        "search block had whitespace/newline drift.",
                        filepath, lines_changed,
                    )
                    return PatchResult(
                        success=True,
                        file=filepath,
                        operation=OperationType.REPLACE_BLOCK,
                        message=f"Replaced block in {filepath} (whitespace-tolerant match)",
                        lines_changed=lines_changed,
                    )
                except OSError as exc:
                    return PatchResult(
                        success=False, file=filepath,
                        operation=OperationType.REPLACE_BLOCK, error=str(exc),
                    )
            if len(ws_matches) > 1:
                return PatchResult(
                    success=False,
                    file=filepath,
                    operation=OperationType.REPLACE_BLOCK,
                    error=(
                        f"Search block matched {len(ws_matches)} regions in "
                        f"{filepath} under whitespace-tolerant comparison. "
                        f"Add more context lines to make the search unique."
                    ),
                )

            # Quote-style fallback. JS/TS files mix single- and double-quote
            # conventions and LLMs frequently emit one style in REPLACE_BLOCK
            # search while the file has the other (e.g. search for
            # `from 'react-router-dom'` against a file that has
            # `from "react-router-dom"`). Quote chars aren't whitespace so
            # ws-tolerant doesn't catch this. Normalize both sides to a
            # single canonical quote char and retry.
            q_matches = _whitespace_tolerant_match(
                original, search, normalize=_quote_normalize,
            )
            if len(q_matches) == 1:
                modified = _whitespace_tolerant_replace(
                    original, search, replace, q_matches[0],
                )
                lines_changed = _count_diff_lines(original, modified)
                try:
                    await _awrite(full_path, modified)
                    logger.info(
                        "[patcher:text] Replaced block in %s via quote-tolerant match "
                        "(%d lines changed). The LLM's search block had "
                        "single/double-quote drift relative to the file.",
                        filepath, lines_changed,
                    )
                    return PatchResult(
                        success=True,
                        file=filepath,
                        operation=OperationType.REPLACE_BLOCK,
                        message=f"Replaced block in {filepath} (quote-tolerant match)",
                        lines_changed=lines_changed,
                    )
                except OSError as exc:
                    return PatchResult(
                        success=False, file=filepath,
                        operation=OperationType.REPLACE_BLOCK, error=str(exc),
                    )
            if len(q_matches) > 1:
                return PatchResult(
                    success=False,
                    file=filepath,
                    operation=OperationType.REPLACE_BLOCK,
                    error=(
                        f"Search block matched {len(q_matches)} regions in "
                        f"{filepath} under quote-tolerant comparison. "
                        f"Add more context lines to make the search unique."
                    ),
                )

            # Indent-style fallback. Makefile/Python/YAML are sensitive to
            # leading whitespace, and LLM responses regularly drift tabs ↔
            # spaces at line starts (especially Makefile recipes where the
            # tokenizer prefers spaces). ws-tolerant only rstrips; this tier
            # normalizes both leading AND trailing whitespace via full
            # ``str.strip``. Last resort before giving up — strictly less
            # safe than the rstrip tier because it can match across indent
            # boundaries, so it's gated on producing a UNIQUE match.
            i_matches = _whitespace_tolerant_match(
                original, search, normalize=str.strip,
            )
            if len(i_matches) == 1:
                # Re-anchor the replacement's indentation to the file's
                # matched region: this tier fires precisely when the
                # LLM's block is uniformly out/indented relative to the
                # file, and writing its replace text verbatim would land
                # the new code at the WRONG column (fatal in Python /
                # Makefile / YAML). Falls back to verbatim when no
                # consistent delta exists.
                modified = _whitespace_tolerant_replace(
                    original, search,
                    _reindent_replace_for_match(
                        original, search, replace, i_matches[0],
                    ),
                    i_matches[0],
                )
                lines_changed = _count_diff_lines(original, modified)
                try:
                    await _awrite(full_path, modified)
                    logger.info(
                        "[patcher:text] Replaced block in %s via indent-tolerant match "
                        "(%d lines changed). The LLM's search block had "
                        "leading-whitespace drift (tabs vs spaces).",
                        filepath, lines_changed,
                    )
                    return PatchResult(
                        success=True,
                        file=filepath,
                        operation=OperationType.REPLACE_BLOCK,
                        message=f"Replaced block in {filepath} (indent-tolerant match)",
                        lines_changed=lines_changed,
                    )
                except OSError as exc:
                    return PatchResult(
                        success=False, file=filepath,
                        operation=OperationType.REPLACE_BLOCK, error=str(exc),
                    )
            if len(i_matches) > 1:
                # Don't error out on ambiguous indent-tolerant — fall through
                # to the line-number-prefix fallback so the LLM gets the
                # closest-match window. Indent-stripped ambiguity is mostly
                # noise (every indented block looks similar) so we'd rather
                # not waste the round on an error.
                pass

            # Line-number-prefix fallback. The patcher's "Search block not
            # found" error includes a line-numbered view of the current
            # file content (see _find_closest_match). LLMs sometimes copy
            # the prefix verbatim into their next REPLACE_BLOCK search —
            # e.g. they emit "  1| import asyncio" instead of
            # "import asyncio". That's invisible to whitespace-tolerant
            # matching because the prefix is non-whitespace characters.
            # If every non-blank line of the search starts with a
            # uniform ``\\s*\\d+\\|\\s?`` prefix, strip it and retry.
            stripped_search = _strip_line_number_prefixes(search)
            if stripped_search is not None and stripped_search != search:
                n_matches = original.count(stripped_search)
                if n_matches == 1:
                    modified = original.replace(stripped_search, replace, 1)
                    lines_changed = _count_diff_lines(original, modified)
                    try:
                        await _awrite(full_path, modified)
                        logger.info(
                            "[patcher:text] Replaced block in %s via "
                            "line-number-stripped match (%d lines changed). "
                            "The LLM's search block had `  N| ` line-number "
                            "prefixes — likely copied verbatim from a prior "
                            "patcher error's wider-context window.",
                            filepath, lines_changed,
                        )
                        return PatchResult(
                            success=True,
                            file=filepath,
                            operation=OperationType.REPLACE_BLOCK,
                            message=(
                                f"Replaced block in {filepath} "
                                f"(line-number-prefix stripped)"
                            ),
                            lines_changed=lines_changed,
                        )
                    except OSError as exc:
                        return PatchResult(
                            success=False, file=filepath,
                            operation=OperationType.REPLACE_BLOCK, error=str(exc),
                        )
                if n_matches == 0:
                    # Try whitespace-tolerant on the stripped search too.
                    ws_matches_stripped = _whitespace_tolerant_match(
                        original, stripped_search,
                    )
                    if len(ws_matches_stripped) == 1:
                        modified = _whitespace_tolerant_replace(
                            original, stripped_search, replace,
                            ws_matches_stripped[0],
                        )
                        lines_changed = _count_diff_lines(original, modified)
                        try:
                            await _awrite(full_path, modified)
                            logger.info(
                                "[patcher:text] Replaced block in %s via "
                                "stripped + whitespace-tolerant match "
                                "(%d lines changed).", filepath, lines_changed,
                            )
                            return PatchResult(
                                success=True,
                                file=filepath,
                                operation=OperationType.REPLACE_BLOCK,
                                message=(
                                    f"Replaced block in {filepath} "
                                    f"(line-number-prefix stripped + ws-tolerant)"
                                ),
                                lines_changed=lines_changed,
                            )
                        except OSError as exc:
                            return PatchResult(
                                success=False, file=filepath,
                                operation=OperationType.REPLACE_BLOCK, error=str(exc),
                            )

            # Elided-middle tier (Aider-style): the LLM wrote head/tail
            # anchor lines with a whole-line "..." marker for the
            # unchanged middle. Match the anchors uniquely and in order,
            # keep the file's middle bytes verbatim, and require the
            # replace block to carry the same marker count. This is the
            # lazy-edit shape long-reasoning models fall into; without
            # this tier it's an unconditional search miss.
            elided = _elided_match_replace(original, search, replace)
            if elided is not None:
                lines_changed = _count_diff_lines(original, elided)
                try:
                    await _awrite(full_path, elided)
                    logger.info(
                        "[patcher:text] Replaced block in %s via "
                        "elided-middle match (%d lines changed). The "
                        "search block used '...' markers; the file's "
                        "elided middle was preserved verbatim.",
                        filepath, lines_changed,
                    )
                    return PatchResult(
                        success=True,
                        file=filepath,
                        operation=OperationType.REPLACE_BLOCK,
                        message=(
                            f"Replaced block in {filepath} "
                            f"(elided-middle match)"
                        ),
                        lines_changed=lines_changed,
                    )
                except OSError as exc:
                    return PatchResult(
                        success=False, file=filepath,
                        operation=OperationType.REPLACE_BLOCK, error=str(exc),
                    )

            # Log the first ~600 chars of the failed search at DEBUG so
            # future debugging has the LLM's actual output to look at
            # without enabling full transcript logging.
            logger.debug(
                "[patcher:text] REPLACE_BLOCK search miss on %s. Search "
                "block excerpt:\n%s",
                filepath, search[:600],
            )
            # No match even after normalization — fall through to closest-match suggestion.
            suggestion = _find_closest_match(original, search)
            # Show the LLM the exact delta between what it searched for
            # and the closest region in the file. When present, this is
            # the single most actionable signal — it explains WHY the
            # search missed instead of leaving the LLM to re-derive it
            # from the raw file window. Empty when the diff would be
            # misleading (no close match / search too large).
            diff_prefix = _render_search_miss_diff(original, search)
            # Cross-file grep: if the LLM's search text lives in another
            # file (usually a same-basename sibling), tell it so instead
            # of letting it spin against the wrong target for N rounds.
            sibling_hits = _find_search_in_other_files(
                self.workspace_root, filepath, search,
            )
            sibling_tail = _format_sibling_hits(sibling_hits, filepath)
            if sibling_hits:
                logger.info(
                    "[patcher:text] REPLACE_BLOCK search miss on %s — but "
                    "distinctive line found in %d other file(s): %s",
                    filepath, len(sibling_hits),
                    ", ".join(f"{p}:{ln}" for p, ln in sibling_hits),
                )
            diff_section = f"\n{diff_prefix}\n" if diff_prefix else ""
            # Fix #3 — for SHORT files that missed the REPLACE_BLOCK anchor,
            # append a hint that suggests REWRITE_FILE. Motivating case:
            # finsearch STORY-038 round 23. LLM was told
            # ``server/main.py`` has only 3 defined symbols
            # (``app, health_check, on_startup``), yet emitted a
            # REPLACE_BLOCK whose ``search`` contained
            # ``app.include_router(company_search_router)`` — a line that
            # doesn't exist in a 20-line file. The generic error message
            # showed the file content and suggested REPLACE_BLOCK, but
            # the right op here is REWRITE_FILE: the file is small, the
            # LLM has the full context, and the fix requires ADDING a
            # line (not replacing one). Threshold at 80 lines — larger
            # files should stay on REPLACE_BLOCK to avoid clobbering
            # unrelated content.
            _file_line_count = original.count("\n") + (1 if original and not original.endswith("\n") else 0)
            _rewrite_hint = ""
            if _file_line_count <= 80:
                _rewrite_hint = (
                    f"\n\nREWRITE_FILE HINT: this file is short "
                    f"({_file_line_count} lines). If your intended change "
                    "is to ADD content that doesn't exist in the file yet "
                    "(new import, new function, new decorator call), "
                    "REPLACE_BLOCK is the wrong op — its search anchor "
                    "cannot match content that doesn't exist. Emit a "
                    "REWRITE_FILE for this path with the FULL corrected "
                    "file body. Format: same as CREATE_FILE, but the "
                    "block name is REWRITE_FILE / END_REWRITE_FILE. "
                    "Post-patch parse validation still applies (broken "
                    "syntax rolls back). If you're modifying an existing "
                    "line, copy the EXACT current content shown above "
                    "into your REPLACE_BLOCK's search — do NOT paste the "
                    "expected-future content as both search and replace."
                )
            return PatchResult(
                success=False,
                file=filepath,
                operation=OperationType.REPLACE_BLOCK,
                error=(
                    f"Search block not found in {filepath}. Your search did "
                    f"not match the current file content — the file may have "
                    f"drifted since your mental model of it. Below is the "
                    f"line-numbered current content of the file (either the "
                    f"entire file, or a window around the closest match for "
                    f"larger files).\n\n"
                    f"{SEARCH_BLOCK_COPY_RULES}"
                    f"{diff_section}\n"
                    f"Current file content (around closest match):\n{suggestion}"
                    f"{sibling_tail}"
                    f"{_rewrite_hint}"
                ),
            )
        if n_matches > 1:
            if policy == "all":
                modified = original.replace(search, replace)
                replaced_n = n_matches
            elif policy == "first":
                modified = original.replace(search, replace, 1)
                replaced_n = 1
            else:
                return PatchResult(
                    success=False,
                    file=filepath,
                    operation=OperationType.REPLACE_BLOCK,
                    error=(
                        f"Search block matched {n_matches} times in {filepath}. "
                        f"Must be unique. To replace every occurrence add "
                        f"`count: all` to the REPLACE_BLOCK; to replace only "
                        f"the first add `count: first`."
                    ),
                )
            lines_changed = _count_diff_lines(original, modified)
            try:
                await _awrite(full_path, modified)
                logger.info(
                    "[patcher:text] Replaced block in %s (%d lines changed, "
                    "policy=%s, %d occurrence(s))",
                    filepath, lines_changed, policy, replaced_n,
                )
                return PatchResult(
                    success=True,
                    file=filepath,
                    operation=OperationType.REPLACE_BLOCK,
                    message=(
                        f"Replaced block in {filepath} "
                        f"(count={policy}, {replaced_n} occurrence(s))"
                    ),
                    lines_changed=lines_changed,
                )
            except OSError as exc:
                return PatchResult(
                    success=False, file=filepath,
                    operation=OperationType.REPLACE_BLOCK, error=str(exc),
                )

        modified = original.replace(search, replace, 1)
        lines_changed = _count_diff_lines(original, modified)

        try:
            await _awrite(full_path, modified)
            logger.info("[patcher:text] Replaced block in %s (%d lines changed)", filepath, lines_changed)
            return PatchResult(
                success=True,
                file=filepath,
                operation=OperationType.REPLACE_BLOCK,
                message=f"Replaced block in {filepath}",
                lines_changed=lines_changed,
            )
        except OSError as exc:
            return PatchResult(success=False, file=filepath, operation=OperationType.REPLACE_BLOCK, error=str(exc))

    async def delete_block(
        self, filepath: str, search: str, *, count: str = "unique",
    ) -> PatchResult:
        full_path = self._resolve_safe(filepath, OperationType.DELETE_BLOCK)
        if isinstance(full_path, PatchResult):
            return full_path
        if not os.path.isfile(full_path):
            return PatchResult(
                success=False,
                file=filepath,
                operation=OperationType.DELETE_BLOCK,
                error=f"File not found: {full_path}",
            )

        try:
            original = await _aread(full_path)
        except OSError as exc:
            return PatchResult(success=False, file=filepath, operation=OperationType.DELETE_BLOCK, error=str(exc))

        policy = (count or "unique").strip().lower()
        if policy not in {"unique", "all", "first"}:
            return PatchResult(
                success=False, file=filepath,
                operation=OperationType.DELETE_BLOCK,
                error=(
                    f"Unknown count policy {count!r} on DELETE_BLOCK for "
                    f"{filepath}. Use one of: unique, all, first."
                ),
            )

        n_matches = original.count(search)
        if n_matches == 0:
            # Idempotency: DELETE_BLOCK's post-condition is exactly "this
            # text is not in the file". If it's already gone, we are
            # done — a resume of a partially-applied patch batch picks
            # up cleanly instead of erroring.
            logger.info(
                "[patcher:text] DELETE_BLOCK no-op: %s already deleted (resume-safe).",
                filepath,
            )
            return PatchResult(
                success=True,
                file=filepath,
                operation=OperationType.DELETE_BLOCK,
                message=f"already deleted (no-op on resume): {filepath}",
                lines_changed=0,
                no_op=True,
            )
        if n_matches > 1:
            if policy == "all":
                modified = original.replace(search, "")
                removed_n = n_matches
            elif policy == "first":
                modified = original.replace(search, "", 1)
                removed_n = 1
            else:
                return PatchResult(
                    success=False,
                    file=filepath,
                    operation=OperationType.DELETE_BLOCK,
                    error=(
                        f"Delete block matched {n_matches} times in {filepath}. "
                        f"Must be unique. To delete every occurrence add "
                        f"`count: all` to the DELETE_BLOCK; to delete only "
                        f"the first add `count: first`."
                    ),
                )
            lines_changed = _count_diff_lines(original, modified)
            try:
                await _awrite(full_path, modified)
                logger.info(
                    "[patcher:text] Deleted block from %s "
                    "(policy=%s, %d occurrence(s))",
                    filepath, policy, removed_n,
                )
                return PatchResult(
                    success=True,
                    file=filepath,
                    operation=OperationType.DELETE_BLOCK,
                    message=(
                        f"Deleted block from {filepath} "
                        f"(count={policy}, {removed_n} occurrence(s))"
                    ),
                    lines_changed=lines_changed,
                )
            except OSError as exc:
                return PatchResult(
                    success=False, file=filepath,
                    operation=OperationType.DELETE_BLOCK, error=str(exc),
                )

        modified = original.replace(search, "", 1)
        lines_changed = _count_diff_lines(original, modified)

        try:
            await _awrite(full_path, modified)
            logger.info("[patcher:text] Deleted block from %s", filepath)
            return PatchResult(
                success=True,
                file=filepath,
                operation=OperationType.DELETE_BLOCK,
                message=f"Deleted block from {filepath}",
                lines_changed=lines_changed,
            )
        except OSError as exc:
            return PatchResult(success=False, file=filepath, operation=OperationType.DELETE_BLOCK, error=str(exc))

    async def insert_at_block(
        self, filepath: str, anchor: str, placement: Placement, content: str
    ) -> PatchResult:
        full_path = self._resolve_safe(filepath, OperationType.INSERT_AT_BLOCK)
        if isinstance(full_path, PatchResult):
            return full_path
        if not os.path.isfile(full_path):
            return PatchResult(
                success=False,
                file=filepath,
                operation=OperationType.INSERT_AT_BLOCK,
                error=f"File not found: {full_path}",
            )

        try:
            original = await _aread(full_path)
        except OSError as exc:
            return PatchResult(success=False, file=filepath, operation=OperationType.INSERT_AT_BLOCK, error=str(exc))

        # For text patcher, anchor is used as a plain substring search
        anchor_idx = original.find(anchor)
        if anchor_idx == -1:
            sibling_hits = _find_search_in_other_files(
                self.workspace_root, filepath, anchor,
            )
            sibling_tail = _format_sibling_hits(sibling_hits, filepath)
            return PatchResult(
                success=False,
                file=filepath,
                operation=OperationType.INSERT_AT_BLOCK,
                error=(
                    f"Anchor '{anchor[:60]}...' not found in {filepath}."
                    f"{sibling_tail}"
                ),
            )

        if placement == Placement.BEFORE:
            # Find the start of the line containing the anchor
            line_start = original.rfind("\n", 0, anchor_idx) + 1
            insert_point = line_start
            content_with_newline = content.rstrip("\n") + "\n"
            # Idempotency: if content_with_newline is already immediately
            # before the anchor line, this INSERT_AT_BLOCK already ran.
            # Re-running would duplicate. Detect and no-op.
            existing_before = original[max(0, line_start - len(content_with_newline)):line_start]
            if existing_before == content_with_newline:
                logger.info(
                    "[patcher:text] INSERT_AT_BLOCK no-op (BEFORE): %s already inserted (resume-safe).",
                    filepath,
                )
                return PatchResult(
                    success=True,
                    file=filepath,
                    operation=OperationType.INSERT_AT_BLOCK,
                    message=f"already inserted (no-op on resume): {filepath}",
                    lines_changed=0,
                    no_op=True,
                )
            modified = original[:insert_point] + content_with_newline + original[insert_point:]
        else:  # Placement.AFTER
            # Find the end of the line containing the anchor
            line_end = original.find("\n", anchor_idx)
            if line_end == -1:
                line_end = len(original)
            insert_point = line_end + 1
            content_with_newline = "\n" + content.rstrip("\n")
            # Idempotency: if content_with_newline is already at the insert
            # point, this INSERT_AT_BLOCK already ran. Re-running would
            # duplicate.
            existing_after = original[insert_point:insert_point + len(content_with_newline)]
            if existing_after == content_with_newline:
                logger.info(
                    "[patcher:text] INSERT_AT_BLOCK no-op (AFTER): %s already inserted (resume-safe).",
                    filepath,
                )
                return PatchResult(
                    success=True,
                    file=filepath,
                    operation=OperationType.INSERT_AT_BLOCK,
                    message=f"already inserted (no-op on resume): {filepath}",
                    lines_changed=0,
                    no_op=True,
                )
            modified = original[:insert_point] + content_with_newline + original[insert_point:]

        lines_changed = content.count("\n") + 1

        try:
            await _awrite(full_path, modified)
            logger.info("[patcher:text] Inserted %s anchor '%s' in %s", placement.value, anchor[:40], filepath)
            return PatchResult(
                success=True,
                file=filepath,
                operation=OperationType.INSERT_AT_BLOCK,
                message=f"Inserted {placement.value} '{anchor[:40]}...' in {filepath}",
                lines_changed=lines_changed,
            )
        except OSError as exc:
            return PatchResult(success=False, file=filepath, operation=OperationType.INSERT_AT_BLOCK, error=str(exc))

    async def insert_at_line(
        self, filepath: str, line: int, content: str,
        *, expected_file_hash: str = "",
    ) -> PatchResult:
        full_path = self._resolve_safe(filepath, OperationType.INSERT_AT_LINE)
        if isinstance(full_path, PatchResult):
            return full_path
        if not os.path.isfile(full_path):
            return PatchResult(
                success=False, file=filepath,
                operation=OperationType.INSERT_AT_LINE,
                error=f"File not found: {filepath}. Use CREATE_FILE for new files.",
            )
        if line < 1:
            return PatchResult(
                success=False, file=filepath,
                operation=OperationType.INSERT_AT_LINE,
                error=f"line must be >= 1 (got {line}).",
            )

        hash_err = _check_file_hash(full_path, expected_file_hash, OperationType.INSERT_AT_LINE, filepath)
        if hash_err is not None:
            return hash_err
        try:
            original = await _aread(full_path)
        except OSError as exc:
            return PatchResult(
                success=False, file=filepath,
                operation=OperationType.INSERT_AT_LINE, error=str(exc),
            )

        file_lines = original.splitlines(keepends=True)
        n_lines = len(file_lines)
        if line > n_lines + 1:
            return PatchResult(
                success=False, file=filepath,
                operation=OperationType.INSERT_AT_LINE,
                error=(
                    f"line={line} is past end of file ({n_lines} lines). "
                    f"Use line={n_lines + 1} to append."
                ),
            )

        # Normalise the content to a list of lines, each terminated with
        # \n. The user supplies content without a trailing newline; we
        # always add one so the inserted block sits on its own line(s).
        # ``rstrip("\r\n")`` first to avoid CRLF double-up when the source
        # of ``content`` is a YAML file authored on Windows.
        normalised = content.rstrip("\r\n")
        insert_lines = (normalised + "\n").splitlines(keepends=True) if normalised else []

        # Idempotency: if file_lines[line-1 : line-1+len(insert_lines)]
        # already equals insert_lines, this patch already landed (or the
        # caller hit a no-op race). Don't duplicate.
        existing_slice = file_lines[line - 1 : line - 1 + len(insert_lines)]
        if existing_slice == insert_lines:
            logger.info(
                "[patcher:text] INSERT_AT_LINE no-op: %s line %d already contains the target content.",
                filepath, line,
            )
            return PatchResult(
                success=True, file=filepath,
                operation=OperationType.INSERT_AT_LINE,
                message=f"already inserted (no-op on resume): {filepath}:{line}",
                lines_changed=0, no_op=True,
            )

        new_lines = file_lines[: line - 1] + insert_lines + file_lines[line - 1 :]
        modified = "".join(new_lines)
        try:
            await _awrite(full_path, modified)
            lines_changed = len(insert_lines)
            logger.info(
                "[patcher:text] INSERT_AT_LINE landed: %s line=%d (+%d line(s)).",
                filepath, line, lines_changed,
            )
            return PatchResult(
                success=True, file=filepath,
                operation=OperationType.INSERT_AT_LINE,
                message=f"Inserted {lines_changed} line(s) at {filepath}:{line}",
                lines_changed=lines_changed,
            )
        except OSError as exc:
            return PatchResult(
                success=False, file=filepath,
                operation=OperationType.INSERT_AT_LINE, error=str(exc),
            )

    async def replace_line_range(
        self, filepath: str, start_line: int, end_line: int, content: str,
        *, expected_file_hash: str = "",
    ) -> PatchResult:
        full_path = self._resolve_safe(filepath, OperationType.REPLACE_LINE_RANGE)
        if isinstance(full_path, PatchResult):
            return full_path
        if not os.path.isfile(full_path):
            return PatchResult(
                success=False, file=filepath,
                operation=OperationType.REPLACE_LINE_RANGE,
                error=f"File not found: {filepath}.",
            )
        if start_line < 1 or end_line < start_line:
            return PatchResult(
                success=False, file=filepath,
                operation=OperationType.REPLACE_LINE_RANGE,
                error=(
                    f"invalid range: start_line={start_line}, "
                    f"end_line={end_line}. Require 1 <= start_line <= end_line."
                ),
            )

        hash_err = _check_file_hash(
            full_path, expected_file_hash, OperationType.REPLACE_LINE_RANGE, filepath,
        )
        if hash_err is not None:
            return hash_err
        try:
            original = await _aread(full_path)
        except OSError as exc:
            return PatchResult(
                success=False, file=filepath,
                operation=OperationType.REPLACE_LINE_RANGE, error=str(exc),
            )

        file_lines = original.splitlines(keepends=True)
        n_lines = len(file_lines)
        if end_line > n_lines:
            return PatchResult(
                success=False, file=filepath,
                operation=OperationType.REPLACE_LINE_RANGE,
                error=(
                    f"end_line={end_line} is past end of file ({n_lines} lines). "
                    "Re-extract diagnostics or use INSERT_AT_LINE to append."
                ),
            )

        # Empty content means "delete this range" — don't synthesise a stray
        # blank line. (The naive `(content + "\n").splitlines(keepends=True)`
        # of "" returns ["\n"], which would leave a trailing newline behind.)
        if content == "":
            replacement_lines: list[str] = []
        else:
            # Normalise: strip any caller-supplied trailing newlines and
            # CRLFs so we don't double-up EOLs when ``content`` was loaded
            # from a CRLF source (YAML on Windows, etc.).
            normalised = content.rstrip("\r\n")
            replacement_lines = (normalised + "\n").splitlines(keepends=True)

        # Idempotency: target range already equals the replacement.
        existing_slice = file_lines[start_line - 1 : end_line]
        if existing_slice == replacement_lines:
            logger.info(
                "[patcher:text] REPLACE_LINE_RANGE no-op: %s lines %d-%d already at target state.",
                filepath, start_line, end_line,
            )
            return PatchResult(
                success=True, file=filepath,
                operation=OperationType.REPLACE_LINE_RANGE,
                message=f"already at target state (no-op): {filepath}:{start_line}-{end_line}",
                lines_changed=0, no_op=True,
            )

        new_lines = (
            file_lines[: start_line - 1]
            + replacement_lines
            + file_lines[end_line :]
        )
        modified = "".join(new_lines)
        try:
            await _awrite(full_path, modified)
            lines_changed = _count_diff_lines(original, modified)
            logger.info(
                "[patcher:text] REPLACE_LINE_RANGE landed: %s lines %d-%d (%d line(s) changed).",
                filepath, start_line, end_line, lines_changed,
            )
            return PatchResult(
                success=True, file=filepath,
                operation=OperationType.REPLACE_LINE_RANGE,
                message=f"Replaced lines {start_line}-{end_line} in {filepath}",
                lines_changed=lines_changed,
            )
        except OSError as exc:
            return PatchResult(
                success=False, file=filepath,
                operation=OperationType.REPLACE_LINE_RANGE, error=str(exc),
            )


def _check_file_hash(
    abs_path: str,
    expected_hash: str,
    op: OperationType,
    rel_path: str,
) -> Optional[PatchResult]:
    """Return a PatchResult error when the file's current sha256 disagrees
    with ``expected_hash``, or None when the check passes (or is skipped
    because ``expected_hash`` is empty).

    Reuses ``sha256_file_bytes`` — the same drift sensor B5 uses for the
    anchor-based ops. Keeping line-based ops on the same hashing primitive
    means the "file drifted under the LLM's mental model" failure mode is
    surfaced with one consistent error shape regardless of which op type
    the LLM (or autofix) chose.
    """
    if not expected_hash:
        return None
    actual = sha256_file_bytes(abs_path) or ""
    if actual.lower() != expected_hash.lower():
        # Show the current bytes rather than an opaque hash prefix.
        # Some models attempt to "match the hash" by editing content
        # (a hallucinatory response to a machine-readable signal), and
        # "re-extract diagnostics" isn't something the LLM can do
        # anyway — it's the harness's job. Point at the actionable
        # move: emit READ_FILE, then rewrite the line range against
        # the current content shown below.
        try:
            with open(abs_path, encoding="utf-8", errors="replace") as fh:
                current = fh.read()
        except OSError:
            current = ""
        if current:
            annotated = _find_closest_match(current, current)
            content_block = f"\n\nCurrent file content:\n{annotated}"
        else:
            content_block = " (current file unreadable)."
        return PatchResult(
            success=False, file=rel_path, operation=op,
            error=(
                f"File hash drift on {rel_path}: the file changed since "
                f"the line numbers in your patch were chosen (a prior "
                f"patch in this batch, or an external editor, rewrote "
                f"it). Do NOT try to \"match\" the sha256 by editing "
                f"content — the hash is a machine signal, not a value "
                f"you can produce. Emit READ_FILE for the up-to-date "
                f"bytes, then re-issue your line-range patch against "
                f"the numbering shown below.{content_block}"
            ),
        )
    return None


# ---------------------------------------------------------------------------
# 7. TreeSitterPatcher — AST-Aware Rewriting
# ---------------------------------------------------------------------------

class TreeSitterPatcher(BasePatcher):
    """
    Structural, AST-aware patching engine using tree-sitter.

    For registered languages, files are parsed into their concrete syntax tree.
    Nodes are located by structural signature (function/class name), rewritten,
    and pretty-printed back. This completely bypasses whitespace/indentation
    mismatch failures that plague text-only search-and-replace engines.

    If tree-sitter is not installed or the grammar for a language is unavailable,
    this patcher raises ImportError and the HybridPatcher falls back to TextPatcher.
    """

    def __init__(self, workspace_root: str):
        self.workspace_root = os.path.abspath(workspace_root)
        self._parsers: dict[str, Any] = {}  # Language → tree-sitter Parser
        self._languages: dict[str, Any] = {}  # Language → tree-sitter Language

    def _get_parser(self, language_name: str) -> Any:
        """
        Lazily load a tree-sitter parser for the given language.
        Returns the parser or raises ImportError if tree-sitter/grammar unavailable.
        """
        if language_name in self._parsers:
            return self._parsers[language_name]

        try:
            import tree_sitter_python as tspython
            import tree_sitter
        except ImportError:
            raise ImportError(
                "tree-sitter is not installed. Install with: pip install tree-sitter tree-sitter-python "
                "and the relevant language packages (e.g., tree-sitter-typescript, tree-sitter-java). "
                "Falling back to TextPatcher."
            )

        # Load the language grammar
        language: Any = None
        try:
            # Try to import the language-specific grammar package
            if language_name == "python":
                language = tree_sitter.Language(tspython.language())
            elif language_name in ("typescript", "tsx", "javascript", "java"):
                # For these languages, try dynamic import
                grammar_module = __import__(f"tree_sitter_{language_name}", fromlist=["language"])
                language = tree_sitter.Language(grammar_module.language())
            else:
                raise ImportError(f"No tree-sitter grammar registered for language: {language_name}")
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                f"tree-sitter grammar for '{language_name}' is not installed. "
                f"Install with: pip install tree-sitter-{language_name}. "
                f"Falling back to TextPatcher. Error: {exc}"
            )

        parser = tree_sitter.Parser()
        parser.language = language
        self._languages[language_name] = language
        self._parsers[language_name] = parser
        return parser

    def _find_node_by_name(self, tree: Any, name: str, node_types: tuple[str, ...]) -> list[Any]:
        """
        Recursively search the tree-sitter CST for nodes matching the given name
        and of specified node types (e.g., 'function_definition', 'class_definition').
        """
        results: list[Any] = []
        cursor = tree.walk()
        stack: list[Any] = [cursor.node]

        while stack:
            node = stack.pop()
            if node.type in node_types:
                # Check if the node's name child matches
                for child in node.children:
                    if child.type == "identifier" or child.type == "name":
                        text = child.text.decode("utf-8") if hasattr(child, "text") else str(child)
                        if text == name:
                            results.append(node)
                            break
            # Add children to stack (DFS)
            for child in reversed(node.children):
                stack.append(child)

        return results

    async def create_file(self, filepath: str, content: str) -> PatchResult:
        # CREATE_FILE is purely text-based regardless — delegate to TextPatcher
        # so the idempotency logic lives in exactly one place.
        text_patcher = TextPatcher(self.workspace_root)
        return await text_patcher.create_file(filepath, content)

    async def replace_block(
        self, filepath: str, search: str, replace: str,
        *, count: str = "unique",
    ) -> PatchResult:
        """
        AST-aware replacement: locate the target node by structural signature
        and replace only that node's text, preserving all surrounding formatting.
        Falls back to text search if AST parsing fails.

        ``count`` other than ``"unique"`` is delegated to the text patcher,
        whose byte-level repetition logic already handles multi-match cases.
        """
        # count != unique exits the AST path immediately — repetition
        # semantics are naturally a byte-level concern.
        if (count or "unique").strip().lower() != "unique":
            text_patcher = TextPatcher(self.workspace_root)
            return await text_patcher.replace_block(
                filepath, search, replace, count=count,
            )
        full_path = self._resolve_safe(filepath, OperationType.REPLACE_BLOCK)
        if isinstance(full_path, PatchResult):
            return full_path
        if not os.path.isfile(full_path):
            # Common LLM mistake: emit REPLACE_BLOCK with an empty search
            # against a file that doesn't exist yet, when CREATE_FILE was
            # intended. Degrade quietly so the repair round still lands
            # the file rather than failing with a raw ENOENT.
            if not search.strip():
                logger.info(
                    "[patcher:ast] %s missing and search is empty — "
                    "treating REPLACE_BLOCK as CREATE_FILE.", filepath,
                )
                text_patcher = TextPatcher(self.workspace_root)
                return await text_patcher.create_file(filepath, replace)
            return PatchResult(
                success=False,
                file=filepath,
                operation=OperationType.REPLACE_BLOCK,
                error=f"File not found: {filepath}. Use CREATE_FILE for new files.",
            )
        lang = get_language_for_file(filepath)
        if lang is None:
            text_patcher = TextPatcher(self.workspace_root)
            return await text_patcher.replace_block(filepath, search, replace)

        try:
            source = await _aread(full_path)
        except OSError as exc:
            return PatchResult(success=False, file=filepath, operation=OperationType.REPLACE_BLOCK, error=str(exc))

        try:
            parser = self._get_parser(lang)
            tree = parser.parse(source.encode("utf-8"))

            # Find nodes whose text matches the search block
            search_bytes = search.encode("utf-8")
            root_node = tree.root_node
            matching_nodes = self._find_text_nodes(root_node, search_bytes)

            if len(matching_nodes) == 0:
                logger.debug("[patcher:ast] No AST node matches search. Falling back to text replace.")
                text_patcher = TextPatcher(self.workspace_root)
                return await text_patcher.replace_block(filepath, search, replace)
            if len(matching_nodes) > 1:
                return PatchResult(
                    success=False,
                    file=filepath,
                    operation=OperationType.REPLACE_BLOCK,
                    error=(
                        f"Search block matched {len(matching_nodes)} AST "
                        f"nodes in {filepath}. Must be unique. To replace "
                        f"every occurrence add `count: all` to the "
                        f"REPLACE_BLOCK; to replace only the first add "
                        f"`count: first`."
                    ),
                )

            target = matching_nodes[0]
            # Replace only the target node's bytes, preserving everything
            # else. Encode the source once (the earlier code encoded it
            # three times in a single expression, allocating 3x source
            # size for every patch on a 1MB file).
            start_byte = target.start_byte
            end_byte = target.end_byte
            source_bytes = source.encode("utf-8")
            modified_bytes = (
                source_bytes[:start_byte]
                + replace.encode("utf-8")
                + source_bytes[end_byte:]
            )
            modified = modified_bytes.decode("utf-8")

            # Verify the result can still be parsed (sanity check)
            try:
                parser.parse(modified.encode("utf-8"))
            except Exception:
                logger.warning("[patcher:ast] Modified source fails to parse. Applying changes anyway.")

            lines_changed = _count_diff_lines(source, modified)
            await _awrite(full_path, modified)
            logger.info("[patcher:ast] AST-aware replace in %s (%d lines changed)", filepath, lines_changed)
            return PatchResult(
                success=True,
                file=filepath,
                operation=OperationType.REPLACE_BLOCK,
                message=f"AST-aware replace in {filepath}",
                lines_changed=lines_changed,
            )
        except ImportError:
            text_patcher = TextPatcher(self.workspace_root)
            return await text_patcher.replace_block(filepath, search, replace)
        except Exception as exc:
            logger.warning("[patcher:ast] AST replace failed: %s. Falling back to text.", exc)
            text_patcher = TextPatcher(self.workspace_root)
            return await text_patcher.replace_block(filepath, search, replace)

    async def delete_block(
        self, filepath: str, search: str, *, count: str = "unique",
    ) -> PatchResult:
        """AST-aware delete: locate and remove the target node.

        ``count`` other than ``"unique"`` delegates to the text patcher.
        """
        if (count or "unique").strip().lower() != "unique":
            text_patcher = TextPatcher(self.workspace_root)
            return await text_patcher.delete_block(filepath, search, count=count)
        full_path = self._resolve_safe(filepath, OperationType.DELETE_BLOCK)
        if isinstance(full_path, PatchResult):
            return full_path
        if not os.path.isfile(full_path):
            return PatchResult(
                success=False,
                file=filepath,
                operation=OperationType.DELETE_BLOCK,
                error=f"File not found: {filepath}. Cannot delete from a file that does not exist.",
            )
        lang = get_language_for_file(filepath)
        if lang is None:
            text_patcher = TextPatcher(self.workspace_root)
            return await text_patcher.delete_block(filepath, search)

        try:
            source = await _aread(full_path)
        except OSError as exc:
            return PatchResult(success=False, file=filepath, operation=OperationType.DELETE_BLOCK, error=str(exc))

        try:
            parser = self._get_parser(lang)
            tree = parser.parse(source.encode("utf-8"))
            matching = self._find_text_nodes(tree.root_node, search.encode("utf-8"))

            if len(matching) == 0:
                text_patcher = TextPatcher(self.workspace_root)
                return await text_patcher.delete_block(filepath, search)
            if len(matching) > 1:
                return PatchResult(
                    success=False,
                    file=filepath,
                    operation=OperationType.DELETE_BLOCK,
                    error=(
                        f"Search block matched {len(matching)} nodes. "
                        f"Must be unique. To delete every occurrence add "
                        f"`count: all` to the DELETE_BLOCK; to delete only "
                        f"the first add `count: first`."
                    ),
                )

            target = matching[0]
            start, end = target.start_byte, target.end_byte
            modified = source[:start] + source[end:]
            lines_changed = _count_diff_lines(source, modified)

            await _awrite(full_path, modified)
            logger.info("[patcher:ast] AST-aware delete from %s", filepath)
            return PatchResult(
                success=True,
                file=filepath,
                operation=OperationType.DELETE_BLOCK,
                message=f"AST-aware delete from {filepath}",
                lines_changed=lines_changed,
            )
        except ImportError:
            text_patcher = TextPatcher(self.workspace_root)
            return await text_patcher.delete_block(filepath, search)
        except Exception as exc:
            logger.warning("[patcher:ast] AST delete failed: %s. Falling back to text.", exc)
            text_patcher = TextPatcher(self.workspace_root)
            return await text_patcher.delete_block(filepath, search)

    async def insert_at_block(
        self, filepath: str, anchor: str, placement: Placement, content: str
    ) -> PatchResult:
        """
        AST-aware insertion: find the named function/class block and insert
        before or after it, using tree-sitter to identify block boundaries
        rather than fragile line numbers or substring matching.
        """
        full_path = self._resolve_safe(filepath, OperationType.INSERT_AT_BLOCK)
        if isinstance(full_path, PatchResult):
            return full_path
        if not os.path.isfile(full_path):
            return PatchResult(
                success=False,
                file=filepath,
                operation=OperationType.INSERT_AT_BLOCK,
                error=f"File not found: {filepath}. Use CREATE_FILE for new files.",
            )
        lang = get_language_for_file(filepath)
        if lang is None:
            text_patcher = TextPatcher(self.workspace_root)
            return await text_patcher.insert_at_block(filepath, anchor, placement, content)

        try:
            source = await _aread(full_path)
        except OSError as exc:
            return PatchResult(success=False, file=filepath, operation=OperationType.INSERT_AT_BLOCK, error=str(exc))

        try:
            parser = self._get_parser(lang)
            tree = parser.parse(source.encode("utf-8"))

            # Search for function/class definitions matching the anchor name
            node_types: tuple[str, ...]
            if lang in ("python",):
                node_types = ("function_definition", "class_definition")
            elif lang in ("javascript", "typescript", "tsx"):
                node_types = ("function_declaration", "class_declaration", "method_definition")
            else:
                node_types = ("function_definition", "class_definition")

            candidates = self._find_node_by_name(tree, anchor, node_types)
            if not candidates:
                text_patcher = TextPatcher(self.workspace_root)
                return await text_patcher.insert_at_block(filepath, anchor, placement, content)

            target_node = candidates[0]

            if placement == Placement.BEFORE:
                insert_byte = target_node.start_byte
            else:
                # For AFTER: advance past any trailing comment + whitespace
                # on the SAME line as ``end_byte``. Tree-sitter's end_byte
                # for a function/class node lands right after the last
                # statement byte; a trailing inline comment ("def f(): pass
                # # note") is a SIBLING node that lives past end_byte. The
                # earlier shape spliced the insert between ``pass`` and the
                # comment, breaking the file. Advancing to the next newline
                # (inclusive) keeps the trailing comment attached and puts
                # the inserted block on the line below.
                insert_byte = target_node.end_byte
                nl_pos = source.find("\n", insert_byte)
                if nl_pos >= 0:
                    insert_byte = nl_pos + 1

            prefix = source[:insert_byte]
            suffix = source[insert_byte:]
            normalised_content = content.rstrip("\r\n")

            if placement == Placement.AFTER:
                # prefix already ends with "\n" when we found a newline
                # above; if it does not (last line of file with no EOL),
                # add one. ``content`` gets its own terminating newline so
                # the next existing line starts cleanly.
                if not prefix.endswith("\n"):
                    prefix = prefix + "\n"
                modified = prefix + normalised_content + "\n" + suffix.lstrip("\n")
            else:
                # BEFORE: insert between any leading code and the anchor.
                # Strip whatever EOLs precede the anchor and put exactly
                # one newline before/after the inserted block.
                modified = (
                    prefix.rstrip("\n") + ("\n" if prefix else "")
                    + normalised_content + "\n"
                    + suffix.lstrip("\n")
                )

            lines_changed = content.count("\n") + 1

            await _awrite(full_path, modified)
            logger.info("[patcher:ast] AST-aware insert %s '%s' in %s", placement.value, anchor, filepath)
            return PatchResult(
                success=True,
                file=filepath,
                operation=OperationType.INSERT_AT_BLOCK,
                message=f"AST-aware insert {placement.value} '{anchor}' in {filepath}",
                lines_changed=lines_changed,
            )
        except ImportError:
            text_patcher = TextPatcher(self.workspace_root)
            return await text_patcher.insert_at_block(filepath, anchor, placement, content)
        except Exception as exc:
            logger.warning("[patcher:ast] AST insert failed: %s. Falling back to text.", exc)
            text_patcher = TextPatcher(self.workspace_root)
            return await text_patcher.insert_at_block(filepath, anchor, placement, content)

    @staticmethod
    def _find_text_nodes(root_node: Any, search_bytes: bytes) -> list[Any]:
        """Find all nodes in the tree whose text matches the given search bytes.

        Dedups by ``(start_byte, end_byte)``: when a search string is the
        entire content of a single-statement file, tree-sitter's parse
        chain (``module`` → ``expression_statement`` → ``assignment``)
        produces multiple ancestor nodes with byte-identical text at the
        same range. Semantically that is ONE occurrence, not many —
        counting each layer inflated ``len(matching)`` and made the
        uniqueness check emit a false "matched N nodes" rejection that
        the LLM could not work around (``count: all`` / ``count: first``
        are meaningless when the "duplicates" are AST ancestors of the
        same leaf). Two distinct real occurrences always have distinct
        byte ranges, so range-dedup is the correct invariant.
        """
        results: list[Any] = []
        seen_ranges: set[tuple[int, int]] = set()
        cursor = root_node.walk()
        stack: list[Any] = [cursor.node]

        while stack:
            node = stack.pop()
            node_bytes = node.text if hasattr(node, "text") else b""
            if node_bytes == search_bytes:
                node_range = (node.start_byte, node.end_byte)
                if node_range not in seen_ranges:
                    seen_ranges.add(node_range)
                    results.append(node)
            for child in reversed(node.children):
                stack.append(child)

        return results

    async def insert_at_line(
        self, filepath: str, line: int, content: str,
        *, expected_file_hash: str = "",
    ) -> PatchResult:
        # Line ops are file-bytes-level — there is no AST advantage to be
        # had. Delegate so the idempotency / hash-drift / boundary checks
        # live in exactly one place.
        return await TextPatcher(self.workspace_root).insert_at_line(
            filepath, line, content, expected_file_hash=expected_file_hash,
        )

    async def replace_line_range(
        self, filepath: str, start_line: int, end_line: int, content: str,
        *, expected_file_hash: str = "",
    ) -> PatchResult:
        return await TextPatcher(self.workspace_root).replace_line_range(
            filepath, start_line, end_line, content,
            expected_file_hash=expected_file_hash,
        )


# ---------------------------------------------------------------------------
# 8. HybridPatcher — Auto-Selects Best Strategy
# ---------------------------------------------------------------------------

class HybridPatcher:
    """
    Orchestrator that selects the best patching strategy per file at runtime.

    Decision logic:
        1. If the file extension maps to a registered tree-sitter language,
           attempt TreeSitterPatcher.
        2. If tree-sitter is unavailable, the import fails, or any AST operation
           encounters an unrecoverable error, fall back to TextPatcher.
        3. CREATE_FILE operations always use the text path (no AST needed).

    All file I/O uses aiofiles with sync fallback.
    """

    def __init__(self, workspace_root: str):
        self.workspace_root = os.path.abspath(workspace_root)
        self._text_patcher = TextPatcher(workspace_root)
        self._ast_patcher: Optional[TreeSitterPatcher] = None

        # Try to initialize the AST patcher
        try:
            self._ast_patcher = TreeSitterPatcher(workspace_root)
            logger.info("[patcher:hybrid] Tree-sitter AST patcher initialized.")
        except ImportError as exc:
            logger.info("[patcher:hybrid] Tree-sitter not available: %s. Using text-only patching.", exc)
            self._ast_patcher = None

    def _select_patcher(self, filepath: str) -> BasePatcher:
        """Select the appropriate patcher for the given file."""
        if self._ast_patcher is not None:
            lang = get_language_for_file(filepath)
            if lang is not None:
                return self._ast_patcher
        return self._text_patcher

    async def apply_patch(self, block: PatchBlock) -> PatchResult:
        """
        Apply a single parsed PatchBlock using the best available strategy.

        Args:
            block: A parsed PatchBlock from parse_patch_blocks().

        Returns:
            PatchResult indicating success/failure and details.
        """
        patcher = self._select_patcher(block.file)

        if block.operation == OperationType.CREATE_FILE:
            return await patcher.create_file(block.file, block.content)
        elif block.operation == OperationType.REWRITE_FILE:
            # REWRITE_FILE always uses the text path — we're overwriting
            # the whole file, not doing an AST-level edit, so tree-sitter
            # buys nothing and the text patcher's atomic write is the
            # right primitive.
            return await self._text_patcher.rewrite_file(
                block.file, block.content,
            )
        elif block.operation == OperationType.REPLACE_BLOCK:
            return await patcher.replace_block(
                block.file, block.search, block.replace, count=block.count,
            )
        elif block.operation == OperationType.DELETE_BLOCK:
            return await patcher.delete_block(
                block.file, block.search, count=block.count,
            )
        elif block.operation == OperationType.INSERT_AT_BLOCK:
            return await patcher.insert_at_block(block.file, block.anchor, block.placement, block.content)
        elif block.operation == OperationType.INSERT_AT_LINE:
            return await patcher.insert_at_line(
                block.file, block.line, block.content,
                expected_file_hash=block.expected_file_hash,
            )
        elif block.operation == OperationType.REPLACE_LINE_RANGE:
            return await patcher.replace_line_range(
                block.file, block.line, block.end_line, block.content,
                expected_file_hash=block.expected_file_hash,
            )
        else:
            return PatchResult(
                success=False,
                file=block.file,
                operation=block.operation,
                error=f"Unknown operation: {block.operation}",
            )

    async def apply_all(self, blocks: list[PatchBlock]) -> list[PatchResult]:
        """
        Apply a sequence of patch blocks in order and return results for ALL
        blocks (success or failure).

        Historical behaviour was to stop on the first failure to avoid
        cascading damage — but in practice this discarded 80-90% of the
        LLM's output every repair iteration: a single bad block (typically
        a CREATE_FILE against an already-existing file) would nuke 5-11
        unrelated patches on other files. The repair LLM then saw the same
        build errors next round and re-emitted the same blocks, looping.

        We now apply every block independently. Failures are surfaced via
        the returned PatchResult list (and ultimately ``patch_failures`` in
        node_state — see ``_format_prior_patch_failures`` in graph.py),
        which the repair LLM reads on the next turn. True cross-file
        dependencies are rare in our patch sets; when they appear, the
        dependent patch will fail with a clear error rather than be
        silently skipped — same information surface, much higher throughput.

        Returns:
            List of PatchResult objects, one per block. Length always
            equals ``len(blocks)``.
        """
        results: list[PatchResult] = []
        for block in blocks:
            # Snapshot pre-patch state for post-patch validation rollback.
            # Skip validation entirely for extensions we don't check —
            # keeps the fast path allocation-free for non-Python files.
            pre_snapshot: Optional[tuple[Optional[str], bool]] = None
            if block.file.endswith(_VALIDATED_SUFFIXES):
                abs_path = _safe_resolve(self.workspace_root, block.file)
                if abs_path is not None and os.path.isfile(abs_path):
                    try:
                        with open(abs_path, encoding="utf-8") as f:
                            pre_content = f.read()
                        pre_snapshot = (pre_content, True)
                    except (OSError, UnicodeDecodeError):
                        # Unreadable / binary — skip validation for this block.
                        pre_snapshot = None
                else:
                    pre_snapshot = (None, False)

            result = await self.apply_patch(block)
            result = self._validate_and_maybe_rollback(block, result, pre_snapshot)
            results.append(result)
            if not result.success:
                logger.error(
                    "[patcher:hybrid] Patch failed at %s (%s): %s",
                    block.file,
                    block.operation.value,
                    result.error,
                )
        return results

    def _validate_and_maybe_rollback(
        self,
        block: PatchBlock,
        result: PatchResult,
        pre_snapshot: Optional[tuple[Optional[str], bool]],
    ) -> PatchResult:
        """Re-parse the patched file; roll back and return a failure result
        if the patch turned a clean file into a broken one.

        Contract:
            - No-op when the patch already failed (nothing to validate).
            - No-op when ``pre_snapshot`` is None (extension not in
              ``_VALIDATED_SUFFIXES`` or the pre-read errored).
            - No-op when the pre-patch content ALSO failed to parse —
              the LLM might be iteratively repairing an already-broken
              file and rolling back would regress partial progress.
            - Rollback + report failure when pre-patch parsed and
              post-patch does not.
        """
        if not result.success or pre_snapshot is None:
            return result
        abs_path = _safe_resolve(self.workspace_root, block.file)
        if abs_path is None:
            return result
        try:
            if os.path.isfile(abs_path):
                with open(abs_path, encoding="utf-8") as f:
                    new_content = f.read()
            else:
                # File does not exist post-patch (e.g. a hypothetical delete
                # op). Nothing to validate.
                return result
        except (OSError, UnicodeDecodeError):
            return result

        post_err = _validate_syntax(block.file, new_content)
        if post_err is None:
            return result

        pre_content, pre_existed = pre_snapshot
        pre_err = (
            _validate_syntax(block.file, pre_content) if pre_content is not None else None
        )
        if pre_content is not None and pre_err is not None:
            # File was already broken — keep the change; the LLM may be
            # iteratively fixing it. Log at debug so a run trace can
            # confirm no rollback fired.
            logger.debug(
                "[patcher:validate] %s: pre-patch already unparseable (%s); "
                "not rolling back post-patch state (%s).",
                block.file, pre_err, post_err,
            )
            return result

        # Pre-patch was clean; post-patch is broken. Roll back and report.
        try:
            if pre_existed and pre_content is not None:
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(pre_content)
            elif not pre_existed:
                os.remove(abs_path)
        except OSError as exc:
            logger.warning(
                "[patcher:validate] Rollback write failed for %s: %s. "
                "File is left in broken state; patch reported as failed anyway.",
                block.file, exc,
            )
        logger.info(
            "[patcher:validate] Rolled back %s: patch left the file "
            "unparseable (%s). Reporting as patch failure so the LLM "
            "sees the corruption on the next round.",
            block.file, post_err,
        )
        return PatchResult(
            success=False,
            file=block.file,
            operation=block.operation,
            error=(
                f"Post-patch validation failed: {post_err}. "
                "The file was syntactically clean before your patch and is "
                "broken after. The patch was ROLLED BACK — the on-disk file "
                "is unchanged. Emit a corrected patch that preserves valid "
                "syntax (balanced braces/brackets, consistent indent, "
                "matching quotes). If your intent was to remove code, use "
                "DELETE_BLOCK on the exact lines rather than a REPLACE_BLOCK "
                "that leaves orphan syntax."
            ),
        )


# ---------------------------------------------------------------------------
# 9. Utility Functions
# ---------------------------------------------------------------------------

def _count_diff_lines(original: str, modified: str) -> int:
    """Count the number of lines changed between original and modified text."""
    orig_lines = original.splitlines()
    mod_lines = modified.splitlines()
    differ = difflib.unified_diff(orig_lines, mod_lines, lineterm="")
    count = 0
    for line in differ:
        if line.startswith("+") or line.startswith("-"):
            count += 1
    return count // 2  # Each change appears as both + and -


# Line-number prefix regex: matches the patcher's own annotation format
# (see _find_closest_match): optional leading whitespace, one or more
# digits, a pipe character, then a single space. Used to strip prefixes
# the LLM may have accidentally copied from a prior failure's
# wider-context window into its REPLACE_BLOCK search.
_LINE_NUMBER_PREFIX_RE = re.compile(r"^\s*\d+\|\s?")


def _strip_line_number_prefixes(search: str) -> Optional[str]:
    """Strip a uniform ``\\s*\\d+\\|\\s?`` prefix from every non-blank line
    of ``search`` and return the result. Returns ``None`` when the prefix
    doesn't appear on every non-blank line OR when the numeric prefixes
    don't form a contiguous monotone run starting from some line N
    (i.e. the search doesn't look like a copy-paste from the patcher's
    annotated wider-context window).

    The conservative "every non-blank line must match AND numbers form a
    contiguous run" rule prevents the stripper from mangling user content
    where multiple lines happen to start with a digit and pipe by
    coincidence (audit §6.11) — e.g. markdown documenting the patcher's
    own annotation format, code that arrays integers into pipe-separated
    output, etc.
    """
    lines = search.splitlines(keepends=True)
    non_blank = [line for line in lines if line.strip()]
    if not non_blank:
        return None
    matches = [_LINE_NUMBER_PREFIX_RE.match(line) for line in non_blank]
    if not all(matches):
        return None
    # Audit §6.11: require the numeric prefixes to form a contiguous
    # monotonically-increasing run. The patcher's annotated context
    # window emits ``  N| line`` for consecutive N — coincidental matches
    # in arbitrary content rarely have that shape.
    try:
        nums = [int(m.group(0).strip().rstrip("|").strip()) for m in matches if m]
    except (AttributeError, ValueError):
        return None
    if len(nums) >= 2:
        for prev, curr in zip(nums, nums[1:]):
            if curr != prev + 1:
                return None
    stripped = [
        _LINE_NUMBER_PREFIX_RE.sub("", line) if line.strip() else line
        for line in lines
    ]
    return "".join(stripped)


def _whitespace_tolerant_match(
    original: str, search: str, *, normalize: Optional[Callable[[str], str]] = None,
) -> list[int]:
    """Find line-aligned regions of ``original`` that match ``search`` after
    per-line normalization.

    Returns a list of *byte* offsets where each match begins in ``original``.
    Empty list means no normalized match. Multiple entries means ambiguous —
    the caller should refuse rather than guess.

    Default normalizer is ``str.rstrip`` — tolerates trailing whitespace per
    line, trailing-newline mismatch on the final line, and CRLF/LF drift.
    Callers can pass a different normalizer (e.g. ``str.strip`` for full
    indent tolerance, or ``_quote_normalize`` for single/double-quote drift)
    to layer additional tolerance tiers. Does NOT tolerate inserted/deleted
    blank lines mid-block — that is a structural change the LLM should re-emit.
    """
    if not search:
        return []
    if normalize is None:
        normalize = str.rstrip
    orig_lines_keep = original.splitlines(keepends=True)
    search_lines = search.splitlines()
    if not search_lines:
        return []
    orig_lines_stripped = [normalize(ln) for ln in orig_lines_keep]
    search_lines_stripped = [normalize(ln) for ln in search_lines]
    n_search = len(search_lines_stripped)
    if n_search > len(orig_lines_stripped):
        return []

    # Build cumulative byte offsets so we can map line index → byte offset
    # without re-scanning the source for every candidate window.
    line_offsets = [0]
    for line in orig_lines_keep:
        line_offsets.append(line_offsets[-1] + len(line))

    matches: list[int] = []
    for i in range(len(orig_lines_stripped) - n_search + 1):
        if orig_lines_stripped[i:i + n_search] == search_lines_stripped:
            matches.append(line_offsets[i])
    return matches


_QUOTE_NORMALIZE_RE = re.compile(r"['\"]")


def _quote_normalize(line: str) -> str:
    """Per-line normalizer that collapses single/double quotes to a single
    canonical char so the matcher tolerates JS/TS quote-style drift.

    The LLM frequently emits ``'react-router-dom'`` in REPLACE_BLOCK search
    while the file has ``"react-router-dom"`` (or vice versa). Neither
    whitespace nor line-number-prefix fallback covers this — quote chars are
    plain content. Normalizing both sides to the same quote char lets the
    matcher find the line; the caller still applies the LLM's replace text
    as written (mixed quote style is benign and a linter will fix it).

    Also rstrips trailing whitespace so this composes the existing
    whitespace-tolerant behavior in one pass.
    """
    return _QUOTE_NORMALIZE_RE.sub('"', line).rstrip()


def _whitespace_tolerant_replace(
    original: str, search: str, replace: str, start_byte: int,
) -> str:
    """Apply ``replace`` to ``original`` at the given byte offset, replacing
    the same number of lines as ``search`` had (post-rstrip). Preserves the
    original final-line newline behavior so we don't silently drop or add a
    trailing newline."""
    orig_lines_keep = original.splitlines(keepends=True)
    line_offsets = [0]
    for line in orig_lines_keep:
        line_offsets.append(line_offsets[-1] + len(line))
    start_line = line_offsets.index(start_byte)
    n_search = len(search.splitlines())
    end_line = start_line + n_search
    replaced_bytes = sum(
        len(orig_lines_keep[j]) for j in range(start_line, end_line)
    )
    last_orig_line = orig_lines_keep[end_line - 1]
    last_has_newline = last_orig_line.endswith(("\n", "\r\n", "\r"))
    replace_text = replace
    replace_has_newline = replace_text.endswith(("\n", "\r\n", "\r"))
    if last_has_newline and not replace_has_newline:
        replace_text += "\n"
    elif not last_has_newline and replace_has_newline:
        replace_text = replace_text.rstrip("\r\n")
    return original[:start_byte] + replace_text + original[start_byte + replaced_bytes:]


def _reindent_replace_for_match(
    original: str, search: str, replace: str, start_byte: int,
) -> str:
    """Re-anchor ``replace``'s indentation to the file's matched region.

    The full-strip (indent-tolerant) tier matches when the LLM's search
    block is uniformly outdented/indented relative to the file — e.g. a
    Python method emitted at column 0 while the file has it at column 4.
    Writing the LLM's ``replace`` verbatim would then land the new code
    at the WRONG indentation and corrupt indentation-sensitive files.

    Detect a CONSISTENT delta between the file's matched lines and the
    search lines (either the file adds a common prefix to every non-blank
    search line, or vice versa) and apply the same transformation to every
    non-blank ``replace`` line. When the delta is inconsistent (mixed
    tabs/spaces drift, ragged edits) return ``replace`` unchanged — the
    caller keeps today's verbatim behavior rather than guessing.
    """
    orig_lines_keep = original.splitlines(keepends=True)
    line_offsets = [0]
    for line in orig_lines_keep:
        line_offsets.append(line_offsets[-1] + len(line))
    try:
        start_line = line_offsets.index(start_byte)
    except ValueError:
        return replace
    search_lines = search.splitlines()
    file_lines = [
        ln.rstrip("\r\n")
        for ln in orig_lines_keep[start_line:start_line + len(search_lines)]
    ]
    if len(file_lines) != len(search_lines):
        return replace

    def _leading_ws(s: str) -> str:
        return s[: len(s) - len(s.lstrip())]

    # Determine the delta from the first non-blank pair, then verify it
    # holds for every other non-blank pair.
    file_adds: Optional[str] = None   # file = delta + search
    search_adds: Optional[str] = None  # search = delta + file
    verified_any = False
    for f_ln, s_ln in zip(file_lines, search_lines):
        if not f_ln.strip() or not s_ln.strip():
            continue
        f_ws, s_ws = _leading_ws(f_ln), _leading_ws(s_ln)
        if file_adds is None and search_adds is None:
            if f_ws == s_ws:
                file_adds = ""  # no delta — nothing to re-anchor
            elif f_ws.endswith(s_ws):
                file_adds = f_ws[: len(f_ws) - len(s_ws)]
            elif s_ws.endswith(f_ws):
                search_adds = s_ws[: len(s_ws) - len(f_ws)]
            else:
                return replace
        if file_adds is not None:
            if f_ws != file_adds + s_ws:
                return replace
        elif search_adds is not None:
            if s_ws != search_adds + f_ws:
                return replace
        verified_any = True
    if not verified_any or (file_adds == "" or (file_adds is None and search_adds is None)):
        return replace

    ends_nl = replace.endswith("\n")
    out_lines: list[str] = []
    for r_ln in replace.splitlines():
        if not r_ln.strip():
            out_lines.append(r_ln)
        elif file_adds is not None:
            out_lines.append(file_adds + r_ln)
        else:
            assert search_adds is not None
            out_lines.append(
                r_ln[len(search_adds):]
                if r_ln.startswith(search_adds) else r_ln
            )
    return "\n".join(out_lines) + ("\n" if ends_nl else "")


# Whole-line elision markers the LLM uses to mean "the middle of this
# region is unchanged". Aider-style: a search block segmented by these
# markers matches head/tail anchors and preserves the file's middle
# bytes verbatim; the replace block must carry the SAME number of
# markers, in order.
_ELISION_MARKERS = frozenset({"...", "# ...", "// ...", "/* ... */", "…"})


def _split_on_elision(text: str) -> Optional[list[str]]:
    """Split ``text`` into segments on whole-line elision markers.

    Returns None when the text contains no marker. Segments keep their
    trailing newlines; a marker at the very start/end produces an empty
    boundary segment (rejected by the caller — anchors must be real)."""
    lines = text.splitlines(keepends=True)
    if not any(ln.strip() in _ELISION_MARKERS for ln in lines):
        return None
    segments: list[str] = []
    current: list[str] = []
    for ln in lines:
        if ln.strip() in _ELISION_MARKERS:
            segments.append("".join(current))
            current = []
        else:
            current.append(ln)
    segments.append("".join(current))
    return segments


def _elided_match_replace(
    original: str, search: str, replace: str,
) -> Optional[str]:
    """Aider-style elided-middle matching.

    When the search block contains whole-line ``...`` markers, match each
    non-elided segment as an ordered, UNIQUE anchor in ``original`` and
    rebuild the region as: replace-segment + preserved-middle + … . The
    replace block must contain exactly as many markers as the search.
    Returns the modified file text, or None when the shape doesn't apply
    or any anchor is missing/ambiguous (caller falls through to the
    closest-match error)."""
    search_segments = _split_on_elision(search)
    if search_segments is None or len(search_segments) < 2:
        return None
    replace_segments = _split_on_elision(replace)
    if replace_segments is None or len(replace_segments) != len(search_segments):
        return None
    if any(not seg.strip() for seg in search_segments):
        return None  # markers must be BETWEEN real anchor lines

    # Locate each anchor uniquely, in order.
    spans: list[tuple[int, int]] = []
    cursor = 0
    for seg in search_segments:
        first = original.find(seg, cursor)
        if first == -1:
            return None
        if original.find(seg, first + 1) != -1:
            return None  # ambiguous anchor — refuse rather than guess
        spans.append((first, first + len(seg)))
        cursor = first + len(seg)

    out: list[str] = [original[: spans[0][0]]]
    for i, seg in enumerate(replace_segments):
        out.append(seg)
        if i < len(spans) - 1:
            # Preserve the file's elided middle verbatim.
            out.append(original[spans[i][1]: spans[i + 1][0]])
    out.append(original[spans[-1][1]:])
    return "".join(out)


def _list_stem_siblings(missing_path: str, limit: int = 6) -> list[str]:
    """Return workspace-relative filenames in ``missing_path``'s directory
    that share its stem (basename minus the last extension) — the LLM's
    ``jest.config.cjs`` vs the on-disk ``jest.config.js`` case.

    Used by REPLACE_BLOCK's file-not-found error to point the LLM at
    the actual file variant that exists, so it stops REPLACE_BLOCK-ing
    a phantom .cjs/.mjs/.ts extension across rounds (session 4d1f9e1c).
    Returns at most ``limit`` names, sorted, to keep the error string
    bounded.
    """
    try:
        parent = os.path.dirname(missing_path) or "."
        if not os.path.isdir(parent):
            return []
        base = os.path.basename(missing_path)
        stem = os.path.splitext(base)[0]
        # Guard against a bare-extension file (``.gitignore``) whose
        # splitext stem is empty — every dotfile in the dir would match.
        if not stem:
            return []
        out: list[str] = []
        for entry in sorted(os.listdir(parent)):
            if entry == base:
                continue
            if os.path.splitext(entry)[0] == stem:
                out.append(entry)
            if len(out) >= limit:
                break
        return out
    except OSError:
        return []


# Directories and file extensions used by the cross-file search-block finder.
# Skip build artifacts, VCS metadata, and vendored deps — they inflate scan
# time and never contain the LLM's intended edit target.
_CROSS_FILE_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", "dist", "build", ".next", ".nuxt", "target", ".cache",
    "htmlcov", ".coverage", "site-packages",
})
_CROSS_FILE_TEXT_EXTS = frozenset({
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".go", ".rs", ".java", ".kt", ".rb", ".php", ".c", ".h", ".cc",
    ".cpp", ".hpp", ".cs", ".swift", ".sh", ".bash", ".zsh",
    ".md", ".txt", ".rst", ".yaml", ".yml", ".json", ".toml",
    ".ini", ".cfg", ".conf", ".env", ".html", ".css", ".scss",
    ".sql", ".graphql", ".proto", ".dockerfile",
})


def _pick_distinctive_line(search: str) -> Optional[str]:
    """Pick a distinctive single line from ``search`` for a workspace grep.

    Prefers the longest non-boilerplate line so the sibling-file hit list
    stays specific (grep on ``}`` or ``return`` produces useless noise).
    Returns ``None`` if the search block has no line worth searching for.
    """
    trivial = {
        "{", "}", "(", ")", "[", "]", ":", ",", ";",
        "pass", "return", "break", "continue", "else:", "try:", "finally:",
        "});", "};", "});", "});",
    }
    candidates: list[str] = []
    for raw in search.splitlines():
        line = raw.strip()
        if len(line) < 12:
            continue
        if line in trivial:
            continue
        if line.startswith("#") or line.startswith("//"):
            continue
        candidates.append(line)
    if not candidates:
        return None
    candidates.sort(key=len, reverse=True)
    # Cap at 200 chars — the needle just has to be distinctive enough to
    # identify a candidate line, not literally reproduce the search block.
    return candidates[0][:200]


def _find_search_in_other_files(
    workspace_root: str,
    target_file: str,
    search: str,
    *,
    max_hits: int = 3,
    max_files_scanned: int = 3000,
) -> list[tuple[str, int]]:
    """When a REPLACE_BLOCK search misses, grep the workspace for a
    distinctive line from ``search`` and return hits in files OTHER than
    ``target_file``.

    This catches the common "LLM confused two same-basename files" failure
    (e.g. ``tests/test_edgar.py`` vs ``tests/unit/backend/test_edgar.py``)
    where every retry rejects because the search text lives in the sibling.
    """
    needle = _pick_distinctive_line(search)
    if needle is None:
        return []

    try:
        target_abs = os.path.abspath(os.path.join(workspace_root, target_file))
    except (TypeError, ValueError):
        return []

    hits: list[tuple[str, int]] = []
    files_scanned = 0

    for dirpath, dirnames, filenames in os.walk(workspace_root):
        dirnames[:] = [d for d in dirnames if d not in _CROSS_FILE_SKIP_DIRS]
        for fname in filenames:
            if files_scanned >= max_files_scanned:
                return hits
            ext = os.path.splitext(fname)[1].lower()
            if ext and ext not in _CROSS_FILE_TEXT_EXTS:
                continue
            path = os.path.join(dirpath, fname)
            try:
                if os.path.abspath(path) == target_abs:
                    continue
            except (TypeError, ValueError):
                continue
            files_scanned += 1
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    for lineno, line in enumerate(fh, start=1):
                        if needle in line:
                            try:
                                rel = os.path.relpath(path, workspace_root)
                            except ValueError:
                                rel = path
                            hits.append((rel, lineno))
                            if len(hits) >= max_hits:
                                return hits
                            break
            except OSError:
                continue
    return hits


def _format_sibling_hits(
    hits: list[tuple[str, int]], target_file: str,
) -> str:
    """Format cross-file grep hits into a REPLACE_BLOCK error tail.

    Ranks same-basename siblings first — those are almost always the
    intended target when the LLM has confused two similarly-named files.
    """
    if not hits:
        return ""
    target_base = os.path.basename(target_file)
    same_basename = [h for h in hits if os.path.basename(h[0]) == target_base]
    other = [h for h in hits if h not in same_basename]
    lines = [f"  - {path}:{lineno}" for path, lineno in (same_basename + other)]
    intro = (
        "\n\nHOWEVER, your distinctive search text WAS found in other file(s):\n"
        + "\n".join(lines)
    )
    if same_basename:
        intro += (
            f"\nNote: {os.path.basename(same_basename[0][0])} matches the "
            f"basename of your target ({target_base}) — you may have picked "
            f"the wrong path. Re-emit REPLACE_BLOCK against the correct file."
        )
    else:
        intro += (
            "\nIf you meant to edit one of those, re-emit REPLACE_BLOCK "
            "against the correct path."
        )
    return intro


def _render_search_miss_diff(
    text: str, search: str,
    *, max_search_lines: int = 30,
) -> str:
    """Return a unified-diff view of the LLM's search block vs. the
    closest matching region of the file, or an empty string when the
    diff would be uninformative.

    The classic patcher rejection error tells the LLM its search "didn't
    match" and dumps the current file content. In practice the LLM often
    already has the file bytes from a prior round but still can't spot
    the delta between what it typed and what's on disk (session
    cec4d124: 3 rounds of REPLACE_BLOCK misses on ``tests/test_edgar.py``
    before HITL escalation). This diff gives it the delta directly —
    "your search says ``company: Company``, file says ``company_id``" —
    so it doesn't have to re-derive it from the raw window.

    Skips the diff when:
      * the search is too large to diff meaningfully (>``max_search_lines``);
      * no line in the file scores above 0.4 similarity — the "closest"
        match would be misleading noise, and the caller's line-numbered
        file window is more useful on its own.
    """
    search_lines = search.strip().splitlines()
    if not search_lines or len(search_lines) > max_search_lines:
        return ""

    text_lines = text.splitlines()
    if not text_lines:
        return ""

    # Anchor on the best-ratio match for the FIRST search line — same
    # heuristic ``_find_closest_match`` uses. Kept separate so tweaking
    # one doesn't accidentally rotate the other.
    first_line = search_lines[0].strip()
    best_start = -1
    best_ratio = 0.0
    for i, line in enumerate(text_lines):
        ratio = difflib.SequenceMatcher(None, first_line, line.strip()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i

    if best_ratio < 0.4 or best_start < 0:
        return ""

    # Span the same number of lines as the search block, clamped to the
    # file. Longer/shorter regions distort the diff — matched line
    # counts make the delta obvious.
    span = len(search_lines)
    region_start = best_start
    region_end = min(len(text_lines), best_start + span)
    file_region = text_lines[region_start:region_end]

    # ``ndiff`` produces a per-line comparison with ``- `` / ``+ `` /
    # ``  `` prefixes that reads well inline — better than unified diff
    # for tiny regions where hunk headers add noise without value.
    diff_lines = list(difflib.ndiff(search_lines, file_region))
    # If every line matches (shouldn't happen since the exact search
    # didn't match, but paranoid): don't emit noise.
    if not any(line.startswith(("- ", "+ ")) for line in diff_lines):
        return ""

    header = (
        f"Delta between your search block and the closest region in the "
        f"file (lines {region_start + 1}-{region_end}, similarity "
        f"{best_ratio:.0%}). ``-`` = your search, ``+`` = current file "
        f"content; unprefixed lines matched exactly. Fix your search "
        f"block to reflect the ``+`` lines EXACTLY (including whitespace) "
        f"before your next REPLACE_BLOCK."
    )
    return header + "\n" + "\n".join(diff_lines)


def _find_closest_match(text: str, search: str, context: int = 20) -> str:
    """Return a line-numbered view of the current file content for the
    LLM's next REPLACE_BLOCK attempt.

    Two modes:

    - **Whole-file mode** (file ≤ 300 lines AND ≤ 6000 chars): return the
      entire file content with line numbers. Eliminates the "LLM's first
      line didn't match anything close to the actual edit site" failure —
      the LLM sees the full file and can pick any region for its next
      search block. Covers nearly every config file (pytest.ini,
      requirements.txt, small .py modules) we patch in practice.

    - **Window mode** (anything larger): use the legacy closest-match
      heuristic to find the best-ratio line and return a ±20-line window
      around it. Capped at 6000 chars so the prompt doesn't blow up on
      huge files.

    Earlier sessions (8b7f7d52, 62084672) failed the repair loop because
    the LLM kept emitting REPLACE_BLOCK searches that didn't match. With
    a 6-line file like pytest.ini, the LLM was somehow still missing the
    target — confirming that even a 20-line window isn't enough if the
    LLM's mental model of the file is wrong. Whole-file mode removes
    that whole class of failure for small files.
    """
    search_lines = search.strip().splitlines()
    if not search_lines:
        return "(empty search block)"

    text_lines = text.splitlines()
    width = max(2, len(str(max(1, len(text_lines)))))

    def _annotate(start: int, end: int) -> str:
        return "\n".join(
            f"{(i + 1):>{width}}| {text_lines[i]}" for i in range(start, end)
        )

    WHOLE_FILE_LINE_CAP = 1000
    WHOLE_FILE_CHAR_CAP = 20000

    # Whole-file mode for small files: no closest-match logic at all,
    # just the full file with line numbers. The LLM sees everything and
    # can construct any REPLACE_BLOCK search it needs.
    if len(text_lines) <= WHOLE_FILE_LINE_CAP and len(text) <= WHOLE_FILE_CHAR_CAP:
        return _annotate(0, len(text_lines))

    # Window mode for larger files: anchor on the best-ratio match for
    # the LLM's first search line and pad with ``context`` lines either
    # side.
    first_line = search_lines[0].strip()
    best_start = -1
    best_end = -1
    best_ratio = 0.0
    for i, line in enumerate(text_lines):
        ratio = difflib.SequenceMatcher(None, first_line, line.strip()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = max(0, i - context)
            best_end = min(len(text_lines), i + len(search_lines) + context)

    if best_ratio < 0.3 or best_start < 0:
        return "(No similar lines found in current file content)"

    rendered = _annotate(best_start, best_end)
    if len(rendered) > WHOLE_FILE_CHAR_CAP:
        rendered = rendered[:WHOLE_FILE_CHAR_CAP] + "\n...(truncated)"
    return rendered


# ---------------------------------------------------------------------------
# 10. Primary Integration Point
# ---------------------------------------------------------------------------

def _classify_patch_failure(error: str) -> str:
    """Short tag for a PatchResult.error string, used in the rollup log so
    operators see the dominant failure mode without digging into per-attempt
    lines. Pure string matching against the canned error messages the
    operation helpers above produce — kept in sync with those messages by
    convention. Unknown errors fall back to ``error``."""
    if not error:
        return "unknown"
    e = error.lower()
    if "file not found" in e and "use create_file" in e:
        return "file missing"
    if "search block not found" in e:
        return "search miss"
    if "matched" in e and "regions" in e and "tolerant" in e:
        return "ambiguous match"
    if "matched" in e and "times in" in e:
        return "ambiguous match"
    if "rejected" in e and "create_file" in e:
        return "rejected: file already exists"
    if "not in skill allowlist" in e:
        return "allowlist denied"
    if "outside workspace" in e or "permission" in e:
        return "path denied"
    if "no patch blocks" in e:
        return "no blocks parsed"
    if "duplicate" in e and "in this batch" in e:
        return "duplicate op in batch"
    if "multi-line replace_block on structural files" in e:
        return "structural file multiline"
    if "[test-protected]" in e:
        return "test file protected"
    return "error"


def sha256_file_bytes(abs_path: str) -> Optional[str]:
    """Return the SHA-256 hex digest of ``abs_path``'s bytes, or ``None``
    when the file is missing or unreadable.

    Cheap drift sensor for B5: the host records the hash of each file at
    the moment its content is shown to the LLM and compares against the
    current on-disk hash before applying a patch. Mismatch means a
    prior patch in the same batch (or an external editor) changed the
    file out from under the LLM's mental model.
    """
    import hashlib
    try:
        h = hashlib.sha256()
        with open(abs_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _serialise_blocks_for_approval(
    blocks: list[PatchBlock], workspace_root: str,
) -> list[dict[str, Any]]:
    """Turn a list of :class:`PatchBlock` instances into the payload
    the Phase-5 diff-approval HITL card renders.

    Each entry carries ``{path, operation, before, after, is_binary,
    size_before, size_after}``. ``before``/``after`` are derived from
    the block's declared intent (search/replace/content), NOT from
    predicting the exact final file state — that would duplicate the
    patcher's job. Consumers can render a small diff of intent, which
    is what the operator needs to decide "approve/reject."

    Each entry is capped at 200 KB per side; anything larger gets a
    truncation marker so the SSE payload stays sane.
    """
    _MAX_SIDE_BYTES = 200 * 1024
    out: list[dict[str, Any]] = []
    for b in blocks:
        op = b.operation.value if hasattr(b.operation, "value") else str(b.operation)
        before = ""
        after = ""
        if op == "create_file":
            after = b.content
        elif op == "delete_block":
            before = b.search
        elif op == "replace_block":
            before = b.search
            after = b.replace
        elif op == "insert_at_block":
            after = b.content or b.replace
        elif op == "insert_at_line":
            after = b.content or b.replace
        elif op == "replace_line_range":
            before = f"(current lines {b.line}-{b.end_line})"
            after = b.content or b.replace
        else:
            after = b.content or b.replace

        def _cap(s: str) -> str:
            if len(s) <= _MAX_SIDE_BYTES:
                return s
            return s[:_MAX_SIDE_BYTES] + f"\n… (+{len(s) - _MAX_SIDE_BYTES} bytes truncated)"

        out.append({
            "path": b.file,
            "operation": op,
            "before": _cap(before),
            "after": _cap(after),
            "is_binary": False,  # patch blocks are text; binary edits go through the file-copy path
            "size_before": len(before),
            "size_after": len(after),
        })
    return out


def _apply_approval_decision(
    blocks: list[PatchBlock], raw_answer: str,
) -> tuple[list[PatchBlock], list[PatchBlock]]:
    """Interpret an operator answer against a proposed batch of blocks.

    Returns ``(approved, rejected)``.

    - ``"approve"`` (or any string that starts with ``a``): approve all.
    - ``"reject"``  (or any string that starts with ``r``): reject all.
    - ``"edit"``    (or any string that starts with ``e``): approve
      everything (the ``extra_notes`` field would carry the subset
      selection when the wire contract is refined; today we fall back
      to approve-all to avoid a silent reject).
    - Anything else: approve all (safe default; matches the ``default``
      passed to ``channel.prompt``).
    """
    a = (raw_answer or "").strip().lower()
    if a.startswith("r"):
        return [], list(blocks)
    # approve / edit / unknown → approve all
    return list(blocks), []


async def process_llm_patch_output(
    llm_output: str,
    workspace_root: str,
    existing_modified_files: Optional[list[str]] = None,
    allowed_paths: Optional["Iterable[str]"] = None,
    *,
    files_seen_by_llm: Optional[dict[str, str]] = None,
    enforce_read_before_edit: bool = False,
    require_approval: bool = False,
) -> tuple[list[PatchResult], list[str]]:
    """
    Parse LLM output for patch blocks, apply them using the hybrid patcher,
    and return results + updated modified_files list.

    This is the primary integration function called by the patching_node
    and repair_node when an LLM response containing code patches is received.

    Args:
        llm_output: The full text response from the LLM containing patch blocks.
        workspace_root: Absolute path to the repository root.
        existing_modified_files: List of files already modified in this session.
        allowed_paths: Optional iterable of workspace-relative paths or
            directory prefixes. When supplied, any patch block targeting a
            file outside this allowlist is rejected with a PatchResult
            error. When None (default), there is no restriction beyond the
            standard traversal guard — preserves backward compatibility for
            top-level patching_node / repair_node callers.
        files_seen_by_llm: Optional dict of ``{rel_path: sha256_hex}``
            tracking the bytes of each file as last shown to the LLM
            (via pre-flight injection, READ_FILE resolution, or the
            patcher's closest-match window). Used for B5 drift detection
            and (when ``enforce_read_before_edit=True``) read-before-edit
            enforcement.
        enforce_read_before_edit: When True, any REPLACE_BLOCK /
            DELETE_BLOCK / INSERT_AT_BLOCK targeting a file NOT in
            ``files_seen_by_llm`` is rejected with a "you have not been
            shown this file" error directing the LLM to emit READ_FILE
            first. Default False — drift detection still runs.

    Returns:
        A tuple of (list of PatchResult, updated modified_files list).
    """
    # Parse the LLM output into structured patch blocks
    blocks = parse_patch_blocks(llm_output)
    logger.info("[patcher] Parsed %d patch blocks from LLM output.", len(blocks))

    return await apply_patch_blocks(
        blocks,
        workspace_root,
        existing_modified_files,
        allowed_paths,
        files_seen_by_llm=files_seen_by_llm,
        enforce_read_before_edit=enforce_read_before_edit,
        require_approval=require_approval,
    )


async def apply_patch_blocks(
    blocks: list[PatchBlock],
    workspace_root: str,
    existing_modified_files: Optional[list[str]] = None,
    allowed_paths: Optional["Iterable[str]"] = None,
    *,
    files_seen_by_llm: Optional[dict[str, str]] = None,
    enforce_read_before_edit: bool = False,
    require_approval: bool = False,
) -> tuple[list[PatchResult], list[str]]:
    """Apply pre-parsed :class:`PatchBlock` instances using the same
    allowlist / B5 / hybrid-patcher pipeline as
    :func:`process_llm_patch_output`.

    Exists so callers that already have structured patches in hand —
    notably the B6 native tool-use path, where
    ``tool_calls_to_patch_blocks`` returns ``PatchBlock`` objects
    directly — can reuse the apply pipeline without round-tripping
    through the text DSL.
    """
    if not blocks:
        logger.warning("[patcher] No patch blocks to apply.")
        return [], existing_modified_files or []

    # If an allowlist is configured, partition blocks into allowed/rejected
    # *before* applying. Rejected blocks become failure PatchResults so the
    # caller can see what the LLM tried to do.
    results: list[PatchResult] = []
    blocks_to_apply: list[PatchBlock] = []

    # Harness-internal files the LLM must NEVER touch, regardless of what
    # the skill allowlist permits. Subdir copies (e.g.
    # ``tests/.harness_config.json``) are also dead weight — the runtime
    # only reads the root file — so reject them too with a precise
    # diagnostic that explains the file is harness-internal rather than
    # workspace code. Without this, the LLM keeps proposing patches to
    # this file when the only signal it gets from the validator is
    # ``Add it to security.allowed_commands in .harness_config.json``.
    _HARNESS_INTERNAL_BASENAMES: frozenset[str] = frozenset({
        ".harness_config.json",
    })
    harness_internal_blocks: list[PatchBlock] = []
    other_blocks: list[PatchBlock] = []
    for block in blocks:
        basename = os.path.basename(block.file)
        if basename in _HARNESS_INTERNAL_BASENAMES:
            harness_internal_blocks.append(block)
        else:
            other_blocks.append(block)
    for block in harness_internal_blocks:
        results.append(PatchResult(
            success=False,
            file=block.file,
            operation=block.operation,
            error=(
                f"refusing to patch harness-internal file {block.file!r}: "
                "this is the harness's own configuration (validator "
                "allowlist, model routing, etc.) and lives outside the "
                "workspace's code. Subdirectory copies of this file are "
                "never read at runtime. If the harness's behaviour needs "
                "to change to unblock the build (e.g. an allowed_commands "
                "entry), surface that requirement in the build output and "
                "let the operator adjust the global config — do NOT "
                "propose patches to this file."
            ),
        ))
        logger.warning(
            "[patcher] Refused harness-internal patch to %s", block.file
        )

    # In-batch dedup for CREATE_FILE / REWRITE_FILE — whole-file ops that are
    # semantically unique per path. If the LLM emits two such ops for the same
    # file in one response, only the first can meaningfully land; the second's
    # bytes are either identical (waste) or contradictory (which one wins is
    # arbitrary). Reject the duplicates up-front with a directive so the LLM
    # knows to consolidate on the next round. REPLACE_BLOCK / DELETE_BLOCK /
    # INSERT_AT_BLOCK are intentionally excluded — multiple such ops per file
    # are legitimate (edit different regions).
    _UNIQUE_PER_FILE_OPS = {OperationType.CREATE_FILE, OperationType.REWRITE_FILE}
    _first_unique_seen: dict[tuple[str, OperationType], int] = {}
    _deduped: list[PatchBlock] = []
    for block in other_blocks:
        if block.operation in _UNIQUE_PER_FILE_OPS:
            key = (block.file, block.operation)
            if key in _first_unique_seen:
                results.append(PatchResult(
                    success=False,
                    file=block.file,
                    operation=block.operation,
                    error=(
                        f"duplicate {block.operation.value.upper()} for "
                        f"{block.file!r} in this batch — the first "
                        f"{block.operation.value.upper()} for this path "
                        f"was applied; this one was dropped. Whole-file "
                        f"ops (CREATE_FILE, REWRITE_FILE) can only land "
                        f"once per path per batch. If you meant to edit "
                        f"the file further after creating/rewriting it, "
                        f"emit a REPLACE_BLOCK / INSERT_AT_BLOCK / "
                        f"DELETE_BLOCK instead."
                    ),
                ))
                logger.warning(
                    "[patcher] Dropped duplicate %s for %s in one batch",
                    block.operation.value.upper(), block.file,
                )
                continue
            _first_unique_seen[key] = 1
        _deduped.append(block)
    other_blocks = _deduped

    if allowed_paths is not None:
        from harness.trust import is_path_allowed
        allowed_list = list(allowed_paths)
        for block in other_blocks:
            if is_path_allowed(block.file, workspace_root, allowed_list):
                blocks_to_apply.append(block)
            else:
                # Two things the LLM commonly misreads about this message:
                # (a) the list is an ALLOWLIST (paths the patcher will
                # write to), not "the only paths that exist in the
                # workspace" — files on disk outside the list are real,
                # they're just off-limits for editing.
                # (b) the fix is not to guess a listed root that "looks
                # closest" — it's to name a listed root or manifest that
                # actually contains the file the LLM was trying to
                # touch. Session 4d1f9e1c pinged among listed roots
                # until it landed on `server/.harness_config.json`,
                # which was rejected as harness-internal anyway.
                results.append(PatchResult(
                    success=False,
                    file=block.file,
                    operation=block.operation,
                    error=(
                        f"path not in skill allowlist: {block.file!r}. "
                        f"The list below is the set of paths this run is "
                        f"PERMITTED to edit — NOT a list of files that "
                        f"exist. Do NOT pick a listed path just because "
                        f"it looks like the closest match to {block.file!r}: "
                        f"only patch a path if it's the actual file you "
                        f"meant to change. If your intended file is a "
                        f"real workspace file that isn't listed, either "
                        f"(1) target it via a listed parent directory, "
                        f"or (2) surface the requirement in the next "
                        f"turn's diagnostic — do NOT keep retrying "
                        f"variants of the same path. Allowlist: {allowed_list}"
                    ),
                ))
                logger.warning(
                    "[patcher] Skill allowlist rejected patch to %s", block.file
                )
    else:
        blocks_to_apply = list(other_blocks)

    # B5 — read-before-edit + drift detection. Both run against the same
    # files_seen_by_llm dict the host maintains in node_state. Drift detection
    # is always on when the dict is provided; the harder "must have been read"
    # rejection is gated by enforce_read_before_edit so existing callers stay
    # backwards-compatible until the operator opts in.
    seen_hashes: dict[str, str] = dict(files_seen_by_llm or {})
    if seen_hashes or enforce_read_before_edit:
        edit_ops = {
            OperationType.REPLACE_BLOCK,
            OperationType.DELETE_BLOCK,
            OperationType.INSERT_AT_BLOCK,
        }
        kept: list[PatchBlock] = []
        for block in blocks_to_apply:
            if block.operation not in edit_ops:
                kept.append(block)
                continue
            abs_path = os.path.join(workspace_root, block.file)
            recorded = seen_hashes.get(block.file)
            if recorded is None:
                if enforce_read_before_edit:
                    results.append(PatchResult(
                        success=False, file=block.file,
                        operation=block.operation,
                        error=(
                            f"You have not been shown the current bytes of "
                            f"`{block.file}`. Emit a READ_FILE block for it "
                            f"first; the harness will resolve it inline and "
                            f"re-dispatch you in the same iteration."
                        ),
                    ))
                    continue
                # Drift detection only runs when we have a recorded hash to
                # compare against. With no record, fall through and let the
                # patcher do exact-byte matching as before.
                kept.append(block)
                continue
            current = sha256_file_bytes(abs_path)
            if current is None:
                # File does not exist on disk — patcher will report a clean
                # "File not found" error. Don't pre-reject here.
                kept.append(block)
                continue
            if current != recorded:
                results.append(PatchResult(
                    success=False, file=block.file,
                    operation=block.operation,
                    error=(
                        f"`{block.file}` has drifted since you last read "
                        f"it (recorded sha256={recorded[:12]}…, current "
                        f"sha256={current[:12]}…). A prior patch in this "
                        f"batch (or an external editor) changed the file. "
                        f"Emit READ_FILE for the current bytes before "
                        f"patching."
                    ),
                ))
                continue
            kept.append(block)
        blocks_to_apply = kept

    # Phase 5: pre-write diff-approval gate. When enabled (opt-in via
    # ``require_approval=True`` from the caller — typically driven off
    # ``security.diff_approval_required`` in config), we hand the
    # blocks-in-hand to the operator BEFORE any bytes land on disk.
    # The webhook payload rides the same HITL channel as every other
    # gate; the dashboard's ``_render_pending_hitl_rows`` renders a
    # split-diff card via ``harness.diff_render.render_patch_list``.
    if require_approval and blocks_to_apply:
        try:
            from harness.hitl import get_channel
            channel = get_channel()
            approval_payload = _serialise_blocks_for_approval(
                blocks_to_apply, workspace_root,
            )
            raw_answer = await asyncio.to_thread(
                channel.prompt,
                "Review proposed file changes before they are written.",
                ["approve", "reject", "edit"],
                default="approve",
                option_labels={
                    "approve": "Approve all",
                    "reject": "Reject all — skip these writes",
                    "edit": "Approve subset — list paths to keep in the notes",
                },
                metadata={
                    "kind": "patch_approval",
                    "headline": "Review file changes before writing",
                    "patches": approval_payload,
                    "timeout_hint_seconds": 1800,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[patcher] diff-approval gate failed (%s); "
                "falling through to write without approval.", exc,
            )
            raw_answer = "approve"
        approved, rejected = _apply_approval_decision(blocks_to_apply, raw_answer)
        for reject_block in rejected:
            results.append(PatchResult(
                success=False, file=reject_block.file,
                operation=reject_block.operation,
                error="rejected by operator via diff-approval gate",
            ))
        try:
            from harness.observability import emit_event
            emit_event(
                "patch_approval_decision",
                answer=str(raw_answer)[:32],
                approved_count=len(approved),
                rejected_count=len(rejected),
            )
        except Exception:  # noqa: BLE001
            pass
        blocks_to_apply = approved

    # Apply the allowed blocks using the hybrid patcher
    if blocks_to_apply:
        patcher = HybridPatcher(workspace_root)
        results.extend(await patcher.apply_all(blocks_to_apply))

    # Track modified files (only successful ones). Keep this-call's new
    # modifications separate from the accumulated list passed in by the
    # caller so the log line below describes what THIS call did, not what
    # any earlier call had already done.
    modified_files = list(existing_modified_files or [])
    newly_modified: list[str] = []
    for result in results:
        if result.success and result.file not in modified_files:
            modified_files.append(result.file)
            newly_modified.append(result.file)

    success_count = sum(1 for r in results if r.success)
    total = len(results)

    # Every file this call wrote to (regardless of whether it was already
    # tracked in existing_modified_files). REPLACE_BLOCK / INSERT_AT_BLOCK
    # on a file that the prior patching pass already touched would show
    # up here, but NOT in `newly_modified` — so the summary log used to
    # say "Files: []" after successful in-place edits, which misled
    # operators into thinking the patch landed nowhere. Show the actual
    # touched files in this call instead.
    touched_this_call: list[str] = []
    for r in results:
        if r.success and r.file not in touched_this_call:
            touched_this_call.append(r.file)

    if total == 0:
        logger.info("[patcher] No patch blocks to apply.")
    elif success_count == total:
        logger.info(
            "[patcher] Applied %d/%d patches. Files: %s",
            success_count, total, touched_this_call,
        )
    else:
        # Partial- or full-failure path: surface what the LLM tried and why
        # each block was rejected so the log isn't just "0/N modified=[]".
        rejected_paths = sorted({
            r.file for r in results
            if not r.success and isinstance(r.error, str)
            and "not in skill allowlist" in r.error
        })
        # Per-file reason for non-allowlist failures. The previous rollup
        # only listed the file paths, which forced the operator to dig into
        # the per-attempt logs to learn whether each miss was a search-not-
        # found / ambiguous / missing-file. Surfacing a short reason tag
        # makes post-mortems 10× faster and helps the repair-loop tuning
        # pass see the dominant failure mode at a glance.
        other_failure_pairs: list[tuple[str, str]] = []
        seen_failure_files: set[str] = set()
        for r in results:
            if r.success or not r.file:
                continue
            err = r.error if isinstance(r.error, str) else ""
            if "not in skill allowlist" in err:
                continue
            if r.file in seen_failure_files:
                continue
            seen_failure_files.add(r.file)
            other_failure_pairs.append((r.file, _classify_patch_failure(err)))
        other_failure_pairs.sort(key=lambda p: p[0])
        parts = [f"[patcher] Applied {success_count}/{total} patches."]
        if touched_this_call:
            parts.append(f"Succeeded on: {touched_this_call}.")
        if rejected_paths:
            parts.append(f"Rejected by allowlist: {rejected_paths}.")
        if other_failure_pairs:
            formatted = ", ".join(
                f"{path} ({reason})" for path, reason in other_failure_pairs
            )
            parts.append(f"Other failures: {formatted}.")
        logger.info(" ".join(parts))

    return results, modified_files