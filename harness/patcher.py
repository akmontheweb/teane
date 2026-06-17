"""
Hybrid file modification engine: tree-sitter AST-aware rewriting with
pure text SEARCH/REPLACE fallback. Uses aiofiles for async file I/O.

This module implements:
    - BasePatcher ABC defining the four canonical operations:
        CREATE_FILE, REPLACE_BLOCK, DELETE_BLOCK, INSERT_AT_BLOCK
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

import difflib
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Types & Enums
# ---------------------------------------------------------------------------

class OperationType(Enum):
    """The four canonical file modification operations."""
    CREATE_FILE = "create_file"
    REPLACE_BLOCK = "replace_block"
    DELETE_BLOCK = "delete_block"
    INSERT_AT_BLOCK = "insert_at_block"


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
_READ_FILE_PATTERN = re.compile(
    r'<<<READ_FILE>>>\s*\n'
    r'file:\s*(?P<file>.+?)\s*\n'
    r'(?:range:\s*(?P<range>\d+\s*-\s*\d+|\d+)\s*\n)?'
    r'<<<END_READ_FILE>>>',
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


_BLOCK_PATTERNS = {
    OperationType.REPLACE_BLOCK: re.compile(
        r'<<<REPLACE_BLOCK>>>\s*\n'
        r'file:\s*(?P<file>.+?)\s*\n'
        r'(?:count:\s*(?P<count>unique|all|first)\s*\n)?'
        r'search:\s*\n(?P<search>.*?)\n'
        r'replace:\s*\n(?P<replace>.*?)'
        r'<<<END_REPLACE_BLOCK>>>',
        re.DOTALL,
    ),
    OperationType.CREATE_FILE: re.compile(
        r'<<<CREATE_FILE>>>\s*\n'
        r'file:\s*(?P<file>.+?)\s*\n'
        r'content:\s*\n(?P<content>.*?)'
        r'<<<END_CREATE_FILE>>>',
        re.DOTALL,
    ),
    OperationType.DELETE_BLOCK: re.compile(
        r'<<<DELETE_BLOCK>>>\s*\n'
        r'file:\s*(?P<file>.+?)\s*\n'
        r'(?:count:\s*(?P<count>unique|all|first)\s*\n)?'
        r'search:\s*\n(?P<search>.*?)'
        r'<<<END_DELETE_BLOCK>>>',
        re.DOTALL,
    ),
    OperationType.INSERT_AT_BLOCK: re.compile(
        r'<<<INSERT_AT_BLOCK>>>\s*\n'
        r'file:\s*(?P<file>.+?)\s*\n'
        r'anchor:\s*(?P<anchor>.+?)\s*\n'
        r'placement:\s*(?P<placement>before|after)\s*\n'
        r'content:\s*\n(?P<content>.*?)'
        r'<<<END_INSERT_AT_BLOCK>>>',
        re.DOTALL,
    ),
}


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

    Supports all four operation types: REPLACE_BLOCK, CREATE_FILE,
    DELETE_BLOCK, and INSERT_AT_BLOCK.

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
    ".rs": "rust",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".java": "java",
    ".rb": "ruby",
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
            # missing file) is fine to let through to the existing write path.
            if getattr(exc, "errno", None) in (40, 62):  # ELOOP across Linux/BSD
                raise PermissionError(
                    f"[patcher] O_NOFOLLOW check tripped on {filepath!r}: "
                    f"target resolves through a symlink ({exc})."
                ) from exc
        else:
            os.close(check_fd)

    directory = os.path.dirname(os.path.abspath(filepath)) or "."
    # delete=False so we own cleanup; suffix keeps editor file-watchers happy.
    fd, tmp_path = tempfile.mkstemp(
        prefix=".harness.tmp.",
        suffix=os.path.basename(filepath),
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
    ) -> tuple[Optional[str], Optional["PatchResult"]]:
        """
        Resolve ``filepath`` against ``self.workspace_root`` with traversal
        protection. Returns ``(absolute_path, None)`` on success or
        ``(None, PatchResult)`` carrying an error to propagate.
        """
        try:
            return _safe_resolve(self.workspace_root, filepath), None
        except ValueError as exc:
            return None, PatchResult(
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
        full_path, _err = self._resolve_safe(filepath, OperationType.CREATE_FILE)
        if _err is not None:
            return _err

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
            snippet = actual[:200].replace("\n", "\\n")
            return PatchResult(
                success=False,
                file=filepath,
                operation=OperationType.CREATE_FILE,
                error=(
                    f"File already exists with different content: {full_path}. "
                    f"Existing first 200 chars: {snippet!r}"
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

    async def replace_block(
        self, filepath: str, search: str, replace: str,
        *, count: str = "unique",
    ) -> PatchResult:
        full_path, _err = self._resolve_safe(filepath, OperationType.REPLACE_BLOCK)
        if _err is not None:
            return _err
        if not os.path.isfile(full_path):
            if not search.strip():
                logger.info(
                    "[patcher:text] %s missing and search is empty — "
                    "treating REPLACE_BLOCK as CREATE_FILE.", filepath,
                )
                return await self.create_file(filepath, replace)
            return PatchResult(
                success=False,
                file=filepath,
                operation=OperationType.REPLACE_BLOCK,
                error=f"File not found: {filepath}. Use CREATE_FILE for new files.",
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
        count = original.count(search)
        if count == 0:
            # Idempotency: if the replacement text is already present in
            # the file (and the search text is gone), this REPLACE_BLOCK
            # was already applied — likely by an earlier run of the same
            # patch batch before the process crashed. Report success so
            # the resume continues cleanly. We require the replacement
            # to appear exactly once to avoid false positives where the
            # text happens to appear elsewhere.
            if replace and original.count(replace) == 1:
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
                count = original.count(stripped_search)
                if count == 1:
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
                if count == 0:
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
                    f"larger files). Copy the EXACT lines you want to "
                    f"replace (WITHOUT the line-number prefix `  N| `) into "
                    f"your next REPLACE_BLOCK's `search:`.\n"
                    f"Current file content (around closest match):\n{suggestion}"
                ),
            )
        if count > 1:
            if policy == "all":
                modified = original.replace(search, replace)
                replaced_n = count
            elif policy == "first":
                modified = original.replace(search, replace, 1)
                replaced_n = 1
            else:
                return PatchResult(
                    success=False,
                    file=filepath,
                    operation=OperationType.REPLACE_BLOCK,
                    error=(
                        f"Search block matched {count} times in {filepath}. "
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
        full_path, _err = self._resolve_safe(filepath, OperationType.DELETE_BLOCK)
        if _err is not None:
            return _err
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

        count = original.count(search)
        if count == 0:
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
        if count > 1:
            if policy == "all":
                modified = original.replace(search, "")
                removed_n = count
            elif policy == "first":
                modified = original.replace(search, "", 1)
                removed_n = 1
            else:
                return PatchResult(
                    success=False,
                    file=filepath,
                    operation=OperationType.DELETE_BLOCK,
                    error=(
                        f"Delete block matched {count} times in {filepath}. "
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
        full_path, _err = self._resolve_safe(filepath, OperationType.INSERT_AT_BLOCK)
        if _err is not None:
            return _err
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
            return PatchResult(
                success=False,
                file=filepath,
                operation=OperationType.INSERT_AT_BLOCK,
                error=f"Anchor '{anchor[:60]}...' not found in {filepath}.",
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
                "and the relevant language packages (e.g., tree-sitter-rust, tree-sitter-typescript). "
                "Falling back to TextPatcher."
            )

        # Load the language grammar
        language: Any = None
        try:
            # Try to import the language-specific grammar package
            if language_name == "python":
                language = tree_sitter.Language(tspython.language())
            elif language_name in ("rust", "typescript", "tsx", "javascript", "go", "c", "cpp"):
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
        full_path, _err = self._resolve_safe(filepath, OperationType.REPLACE_BLOCK)
        if _err is not None:
            return _err
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
            # Replace only the target node's bytes, preserving everything else
            start_byte = target.start_byte
            end_byte = target.end_byte
            modified_bytes = source.encode("utf-8")[:start_byte] + replace.encode("utf-8") + source.encode("utf-8")[end_byte:]
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
        full_path, _err = self._resolve_safe(filepath, OperationType.DELETE_BLOCK)
        if _err is not None:
            return _err
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
        full_path, _err = self._resolve_safe(filepath, OperationType.INSERT_AT_BLOCK)
        if _err is not None:
            return _err
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
            elif lang == "rust":
                node_types = ("function_item", "struct_item", "impl_item")
            elif lang in ("c", "cpp"):
                node_types = ("function_definition", "class_specifier")
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
                insert_byte = target_node.end_byte

            # Ensure clean newline separation
            prefix = source[:insert_byte]
            suffix = source[insert_byte:]

            if placement == Placement.AFTER:
                modified = prefix.rstrip("\n") + "\n" + content.rstrip("\n") + "\n" + suffix.lstrip("\n")
            else:
                modified = prefix.rstrip("\n") + "\n" + content.rstrip("\n") + "\n" + suffix.lstrip("\n")

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
        """Find all nodes in the tree whose text matches the given search bytes."""
        results: list[Any] = []
        cursor = root_node.walk()
        stack: list[Any] = [cursor.node]

        while stack:
            node = stack.pop()
            node_bytes = node.text if hasattr(node, "text") else b""
            if node_bytes == search_bytes:
                results.append(node)
            for child in reversed(node.children):
                stack.append(child)

        return results


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
            result = await self.apply_patch(block)
            results.append(result)
            if not result.success:
                logger.error(
                    "[patcher:hybrid] Patch failed at %s (%s): %s",
                    block.file,
                    block.operation.value,
                    result.error,
                )
        return results


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
        nums = [int(m.group(0).strip().rstrip("|").strip()) for m in matches]
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


def _whitespace_tolerant_match(original: str, search: str) -> list[int]:
    """Find line-aligned regions of ``original`` that match ``search`` after
    rstrip-per-line normalization.

    Returns a list of *byte* offsets where each match begins in ``original``.
    Empty list means no normalized match. Multiple entries means ambiguous —
    the caller should refuse rather than guess.

    Tolerates: trailing whitespace per line, trailing-newline mismatch on the
    final line, and CRLF/LF drift. Does NOT tolerate inserted/deleted blank
    lines mid-block — that is a structural change the LLM should re-emit.
    """
    if not search:
        return []
    orig_lines_keep = original.splitlines(keepends=True)
    search_lines = search.splitlines()
    if not search_lines:
        return []
    orig_lines_stripped = [ln.rstrip() for ln in orig_lines_keep]
    search_lines_stripped = [ln.rstrip() for ln in search_lines]
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

    WHOLE_FILE_LINE_CAP = 300
    WHOLE_FILE_CHAR_CAP = 6000

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


async def process_llm_patch_output(
    llm_output: str,
    workspace_root: str,
    existing_modified_files: Optional[list[str]] = None,
    allowed_paths: Optional["Iterable[str]"] = None,
    *,
    files_seen_by_llm: Optional[dict[str, str]] = None,
    enforce_read_before_edit: bool = False,
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
    )


async def apply_patch_blocks(
    blocks: list[PatchBlock],
    workspace_root: str,
    existing_modified_files: Optional[list[str]] = None,
    allowed_paths: Optional["Iterable[str]"] = None,
    *,
    files_seen_by_llm: Optional[dict[str, str]] = None,
    enforce_read_before_edit: bool = False,
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

    if allowed_paths is not None:
        from harness.trust import is_path_allowed
        allowed_list = list(allowed_paths)
        for block in blocks:
            if is_path_allowed(block.file, workspace_root, allowed_list):
                blocks_to_apply.append(block)
            else:
                results.append(PatchResult(
                    success=False,
                    file=block.file,
                    operation=block.operation,
                    error=(
                        f"path not in skill allowlist: {block.file!r} "
                        f"(allowed: {allowed_list})"
                    ),
                ))
                logger.warning(
                    "[patcher] Skill allowlist rejected patch to %s", block.file
                )
    else:
        blocks_to_apply = list(blocks)

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
        other_failures = sorted({
            r.file for r in results
            if not r.success and (
                not isinstance(r.error, str)
                or "not in skill allowlist" not in r.error
            )
        })
        parts = [f"[patcher] Applied {success_count}/{total} patches."]
        if touched_this_call:
            parts.append(f"Succeeded on: {touched_this_call}.")
        if rejected_paths:
            parts.append(f"Rejected by allowlist: {rejected_paths}.")
        if other_failures:
            parts.append(f"Other failures: {other_failures}.")
        logger.info(" ".join(parts))

    return results, modified_files