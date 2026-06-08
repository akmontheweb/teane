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


# ---------------------------------------------------------------------------
# 2. Patch Block Parser — Extracts SEARCH/REPLACE from LLM text
# ---------------------------------------------------------------------------

# Regex patterns to match each block type in LLM output
_BLOCK_PATTERNS = {
    OperationType.REPLACE_BLOCK: re.compile(
        r'<<<REPLACE_BLOCK>>>\s*\n'
        r'file:\s*(.+?)\s*\n'
        r'search:\s*\n(.*?)\n'
        r'replace:\s*\n(.*?)\n'
        r'<<<END_REPLACE_BLOCK>>>',
        re.DOTALL,
    ),
    OperationType.CREATE_FILE: re.compile(
        r'<<<CREATE_FILE>>>\s*\n'
        r'file:\s*(.+?)\s*\n'
        r'content:\s*\n(.*?)\n'
        r'<<<END_CREATE_FILE>>>',
        re.DOTALL,
    ),
    OperationType.DELETE_BLOCK: re.compile(
        r'<<<DELETE_BLOCK>>>\s*\n'
        r'file:\s*(.+?)\s*\n'
        r'search:\s*\n(.*?)\n'
        r'<<<END_DELETE_BLOCK>>>',
        re.DOTALL,
    ),
    OperationType.INSERT_AT_BLOCK: re.compile(
        r'<<<INSERT_AT_BLOCK>>>\s*\n'
        r'file:\s*(.+?)\s*\n'
        r'anchor:\s*(.+?)\s*\n'
        r'placement:\s*(before|after)\s*\n'
        r'content:\s*\n(.*?)\n'
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

            if op_type == OperationType.REPLACE_BLOCK:
                blocks.append(PatchBlock(
                    operation=OperationType.REPLACE_BLOCK,
                    file=match.group(1).strip(),
                    search=match.group(2).rstrip(),
                    replace=match.group(3).rstrip(),
                    raw_block=raw,
                ))
            elif op_type == OperationType.CREATE_FILE:
                blocks.append(PatchBlock(
                    operation=OperationType.CREATE_FILE,
                    file=match.group(1).strip(),
                    content=match.group(2).rstrip(),
                    raw_block=raw,
                ))
            elif op_type == OperationType.DELETE_BLOCK:
                blocks.append(PatchBlock(
                    operation=OperationType.DELETE_BLOCK,
                    file=match.group(1).strip(),
                    search=match.group(2).rstrip(),
                    raw_block=raw,
                ))
            elif op_type == OperationType.INSERT_AT_BLOCK:
                placement = Placement.BEFORE if match.group(3).strip().lower() == "before" else Placement.AFTER
                blocks.append(PatchBlock(
                    operation=OperationType.INSERT_AT_BLOCK,
                    file=match.group(1).strip(),
                    anchor=match.group(2).strip(),
                    placement=placement,
                    content=match.group(4).rstrip(),
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
    """
    import tempfile

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
    async def replace_block(self, filepath: str, search: str, replace: str) -> PatchResult:
        """Replace an exact-match block of text within an existing file."""
        ...

    @abstractmethod
    async def delete_block(self, filepath: str, search: str) -> PatchResult:
        """Delete an exact-match block of text from an existing file."""
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

    async def replace_block(self, filepath: str, search: str, replace: str) -> PatchResult:
        full_path, _err = self._resolve_safe(filepath, OperationType.REPLACE_BLOCK)
        if _err is not None:
            return _err
        if not os.path.isfile(full_path):
            return PatchResult(
                success=False,
                file=filepath,
                operation=OperationType.REPLACE_BLOCK,
                error=f"File not found: {full_path}",
            )

        try:
            original = await _aread(full_path)
        except OSError as exc:
            return PatchResult(success=False, file=filepath, operation=OperationType.REPLACE_BLOCK, error=str(exc))

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
                )
            # Try fuzzy matching to produce a helpful error
            suggestion = _find_closest_match(original, search)
            return PatchResult(
                success=False,
                file=filepath,
                operation=OperationType.REPLACE_BLOCK,
                error=f"Search block not found in {filepath}. Closest match:\n{suggestion}",
            )
        if count > 1:
            return PatchResult(
                success=False,
                file=filepath,
                operation=OperationType.REPLACE_BLOCK,
                error=f"Search block matched {count} times in {filepath}. Must be unique.",
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

    async def delete_block(self, filepath: str, search: str) -> PatchResult:
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
            )
        if count > 1:
            return PatchResult(
                success=False,
                file=filepath,
                operation=OperationType.DELETE_BLOCK,
                error=f"Delete block matched {count} times in {filepath}. Must be unique.",
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

    async def replace_block(self, filepath: str, search: str, replace: str) -> PatchResult:
        """
        AST-aware replacement: locate the target node by structural signature
        and replace only that node's text, preserving all surrounding formatting.
        Falls back to text search if AST parsing fails.
        """
        full_path, _err = self._resolve_safe(filepath, OperationType.REPLACE_BLOCK)
        if _err is not None:
            return _err
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
                    error=f"Search block matched {len(matching_nodes)} AST nodes in {filepath}. Must be unique.",
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

    async def delete_block(self, filepath: str, search: str) -> PatchResult:
        """AST-aware delete: locate and remove the target node."""
        full_path, _err = self._resolve_safe(filepath, OperationType.DELETE_BLOCK)
        if _err is not None:
            return _err
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
                    error=f"Search block matched {len(matching)} nodes. Must be unique.",
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
            return await patcher.replace_block(block.file, block.search, block.replace)
        elif block.operation == OperationType.DELETE_BLOCK:
            return await patcher.delete_block(block.file, block.search)
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
        Apply a sequence of patch blocks in order and return results.

        Stops on the first failure to avoid cascading errors (a failed
        search on file X followed by a valid patch on file Y that depends
        on X's changes would be unsafe).

        Returns:
            List of PatchResult objects, one per block up to the failure point.
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
                break
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


def _find_closest_match(text: str, search: str, context: int = 3) -> str:
    """
    When an exact search block is not found, try to find the closest matching
    substring to provide a helpful error message.
    """
    search_lines = search.strip().splitlines()
    if not search_lines:
        return "(empty search block)"

    first_line = search_lines[0].strip()
    text_lines = text.splitlines()

    best_match = ""
    best_ratio = 0.0

    for i, line in enumerate(text_lines):
        ratio = difflib.SequenceMatcher(None, first_line, line.strip()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            start = max(0, i - context)
            end = min(len(text_lines), i + len(search_lines) + context)
            best_match = "\n".join(text_lines[start:end])

    if best_ratio < 0.3:
        return "(No similar lines found)"

    return best_match[:500]


# ---------------------------------------------------------------------------
# 10. Primary Integration Point
# ---------------------------------------------------------------------------

async def process_llm_patch_output(
    llm_output: str,
    workspace_root: str,
    existing_modified_files: Optional[list[str]] = None,
    allowed_paths: Optional["Iterable[str]"] = None,
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

    Returns:
        A tuple of (list of PatchResult, updated modified_files list).
    """
    # Parse the LLM output into structured patch blocks
    blocks = parse_patch_blocks(llm_output)
    logger.info("[patcher] Parsed %d patch blocks from LLM output.", len(blocks))

    if not blocks:
        logger.warning("[patcher] No patch blocks found in LLM output.")
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

    # Apply the allowed blocks using the hybrid patcher
    if blocks_to_apply:
        patcher = HybridPatcher(workspace_root)
        results.extend(await patcher.apply_all(blocks_to_apply))

    # Track modified files (only successful ones)
    modified_files = list(existing_modified_files or [])
    for result in results:
        if result.success and result.file not in modified_files:
            modified_files.append(result.file)

    success_count = sum(1 for r in results if r.success)
    logger.info(
        "[patcher] Applied %d/%d patches successfully. Modified files: %s",
        success_count, len(results), modified_files,
    )

    return results, modified_files