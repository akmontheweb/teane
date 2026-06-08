"""
Semantic Code Graph Regression — AST Impact Analysis.

This module implements:
    - DependencyGraph: Cross-file dependency scanner using tree-sitter AST parsing
      and text-pattern fallback for non-AST languages. Builds a reverse dependency
      index: "if file X is modified, which other files are at risk?"

    - ImpactAnalyzer: Pre-patch impact checker. Before patches are applied,
      queries the dependency graph to warn the agent about downstream breakage:
      "Warning: Your edit to core/auth.go modifies an interface used by 14 endpoints."

Integration point:
    - Called inside patching_node and repair_node after process_llm_patch_output()
      succeeds. The impact warning is injected as a system message into the
      conversation context before routing to compiler_node.

Data structures:
    - DependencyGraph holds two indices:
        forward:  {file → {symbol → [files that define it]}}
        reverse:  {file → {symbol → [files that depend on it]}}
    - The "symbol" is a qualified name: function name, class name, interface, export.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Types
# ---------------------------------------------------------------------------

@dataclass
class ImpactResult:
    """Result of an impact analysis for a set of modified files."""
    modified_files: list[str]
    impacted_files: list[str] = field(default_factory=list)
    symbol_impact: dict[str, list[str]] = field(default_factory=dict)
    warning: str = ""
    total_impacted: int = 0
    # True when the underlying dependency graph hit max_scan_files and
    # the workspace was only partially indexed. impacted_files only
    # reflects dependents the scanner actually saw — there may be more
    # downstream files that reference the modified files but lived
    # beyond the scan cap. Callers should treat impact results as a
    # *floor*, not a complete enumeration, when this is True.
    graph_incomplete: bool = False
    # Number of source files the graph builder actually scanned. Useful
    # for surfacing "scanned 500 of ~2300 files" to the user.
    files_scanned: int = 0

    def has_impact(self) -> bool:
        return self.total_impacted > 0


# ---------------------------------------------------------------------------
# 2. DependencyGraph — Cross-File Dependency Scanner
# ---------------------------------------------------------------------------

class DependencyGraph:
    """
    Builds a cross-file dependency graph for the workspace.

    Uses tree-sitter for AST-level accuracy on supported languages,
    and regex/text patterns for unsupported files (JSON, YAML, Markdown, configs).

    Two indices:
        - _exports:   {file_path → set of symbol names defined in that file}
        - _dependents: {symbol → set of file_paths that import/use that symbol}
    """

    # Language-agnostic import/reference patterns for text-based fallback
    _IMPORT_PATTERNS: list[re.Pattern[str]] = [
        # Python: import X, from X import Y
        re.compile(r'(?:from\s+(\S+)\s+import\s|\bimport\s+(\S+))', re.MULTILINE),
        # JavaScript/TypeScript: import ... from 'X', require('X')
        re.compile(r"""(?:import\s+.*?from\s+['"](\S+?)['"]|require\s*\(\s*['"](\S+?)['"])""", re.MULTILINE),
        # Go: import "X"
        re.compile(r'import\s+(?:\w+\s+)?["\x60](\S+?)["\x60]', re.MULTILINE),
        # Rust: use X, extern crate X, mod X
        re.compile(r'(?:^\s*(?:use|extern\s+crate|mod)\s+(\S+?)\s*;)', re.MULTILINE),
        # C/C++: #include "X", #include <X>
        re.compile(r'#include\s+[<"](\S+?)[>"]', re.MULTILINE),
        # Java: import X;
        re.compile(r'import\s+(\S+?);', re.MULTILINE),
    ]

    def __init__(
        self,
        workspace_path: str,
        max_scan_files: int = 500,
        ignore_patterns: Optional[list[str]] = None,
    ):
        self.workspace_path = os.path.abspath(workspace_path)
        self.max_scan_files = max_scan_files
        self.ignore_patterns = ignore_patterns or [
            "tests/", "test/", "__pycache__/", "node_modules/",
            "vendor/", "target/", "build/", "dist/", ".git/",
            ".tox/", ".nox/", "venv/", ".venv/",
        ]
        self._exports: dict[str, set[str]] = {}
        self._dependents: dict[str, set[str]] = {}
        self._built = False
        # True when build() bailed out at max_scan_files. Surfaced via
        # ImpactResult.graph_incomplete so downstream warnings can say
        # "this list is a lower bound" instead of treating it as complete.
        self.incomplete: bool = False
        self.files_scanned: int = 0

    def _is_ignored(self, filepath: str) -> bool:
        """Check if a file should be excluded from scanning."""
        rel = os.path.relpath(filepath, self.workspace_path)
        for pattern in self.ignore_patterns:
            if pattern in rel:
                return True
        return False

    def build(self) -> int:
        """
        Scan all source files in the workspace and build the dependency graph.

        Returns:
            Number of files scanned.
        """
        if self._built:
            return len(self._exports)

        scanned = 0
        for root, dirs, files in os.walk(self.workspace_path):
            # Filter ignored directories in-place
            dirs[:] = [d for d in dirs if not self._is_ignored(os.path.join(root, d))]
            for filename in files:
                filepath = os.path.join(root, filename)
                if self._is_ignored(filepath):
                    continue
                if scanned >= self.max_scan_files:
                    logger.warning(
                        "[impact] Maximum scan limit reached (%d files). Dependency graph is INCOMPLETE.",
                        self.max_scan_files,
                    )
                    # Mark the result as incomplete so ImpactResult can warn
                    # the caller that impacted_files is a lower bound.
                    self.incomplete = True
                    self.files_scanned = scanned
                    self._built = True
                    # Still build the reverse index over the partial set so
                    # impact lookups remain useful (just not exhaustive).
                    self._build_reverse_index()
                    return scanned
                try:
                    self._scan_file(filepath)
                except Exception as exc:
                    logger.debug("[impact] Failed to scan %s: %s", filepath, exc)
                scanned += 1

        # Build reverse index: for each symbol, find all files that reference it
        self._build_reverse_index()

        self.files_scanned = scanned
        self._built = True
        logger.info("[impact] Dependency graph built: %d files, %d symbols.",
                     scanned, sum(len(syms) for syms in self._exports.values()))
        return scanned

    def _scan_file(self, filepath: str) -> None:
        """
        Scan a single file for symbols and dependencies.

        Strategy:
            1. Try tree-sitter AST parsing for known languages
            2. Fall back to regex text patterns for all files
        """
        ext = os.path.splitext(filepath)[1].lower()
        lang = _EXTENSION_TO_TREE_SITTER.get(ext, "")

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except (OSError, UnicodeDecodeError):
            return

        symbols: set[str] = set()
        imports: set[str] = set()

        # Try tree-sitter for structural extraction
        if lang and self._try_tree_sitter_extract(filepath, source, lang, symbols):
            pass
        else:
            # Fallback: text-based extraction
            self._text_extract_symbols(filepath, ext, source, symbols)
            self._text_extract_imports(filepath, source, imports)

        self._exports[filepath] = symbols

        # Register imports: each imported package/module references external symbols
        for imp in imports:
            self._dependents.setdefault(imp, set()).add(filepath)

    def _try_tree_sitter_extract(
        self,
        filepath: str,
        source: str,
        lang: str,
        symbols: set[str],
    ) -> bool:
        """Try to extract symbols from a file using tree-sitter AST. Returns True on success."""
        try:
            # Lazy import to avoid hard dependency
            import tree_sitter_python

            grammar_map: dict[str, Any] = {
                "python": tree_sitter_python,
            }
            try:
                import tree_sitter
            except ImportError:
                return False

            grammar_module = grammar_map.get(lang)
            if grammar_module is None:
                return False

            ts_lang = tree_sitter.Language(grammar_module.language())
            parser = tree_sitter.Parser()
            parser.language = ts_lang

            tree = parser.parse(source.encode("utf-8"))
            self._extract_symbols_from_ast(tree.root_node, lang, symbols)
            return True
        except (ImportError, Exception) as exc:
            logger.debug("[impact] Tree-sitter extraction failed for %s: %s", filepath, exc)
            return False

    def _extract_symbols_from_ast(self, node: Any, lang: str, symbols: set[str]) -> None:
        """Recursively walk a tree-sitter AST and extract symbol definitions."""
        if not hasattr(node, "children") or node.children is None:
            return

        for child in node.children:
            if lang == "python":
                # Function definitions
                if child.type == "function_definition":
                    for sub in child.children:
                        if sub.type == "identifier":
                            symbols.add(sub.text.decode("utf-8"))
                            break
                # Class definitions
                elif child.type == "class_definition":
                    for sub in child.children:
                        if sub.type == "identifier":
                            symbols.add(sub.text.decode("utf-8"))
                            break
                # Module-level assignments (top-level vars, constants)
                elif child.type == "assignment" and node.type == "module":
                    for sub in child.children:
                        if sub.type == "identifier":
                            symbols.add(sub.text.decode("utf-8"))
                            break
            elif lang in ("typescript", "tsx", "javascript", "jsx"):
                if child.type in ("function_declaration", "class_declaration", "method_definition"):
                    for sub in child.children:
                        if sub.type == "identifier":
                            symbols.add(sub.text.decode("utf-8"))
                            break
            elif lang == "rust":
                if child.type in ("function_item", "struct_item", "trait_item", "impl_item"):
                    for sub in child.children:
                        if sub.type == "identifier":
                            symbols.add(sub.text.decode("utf-8"))
                            break

            self._extract_symbols_from_ast(child, lang, symbols)

    def _text_extract_symbols(self, filepath: str, ext: str, source: str, symbols: set[str]) -> None:
        """Extract symbols using regex patterns when tree-sitter is unavailable."""
        if ext in (".py", ".pyi"):
            # Python: def foo, class Foo
            for m in re.finditer(r'(?:^\s*(?:def|class)\s+)([A-Za-z_]\w*)', source, re.MULTILINE):
                symbols.add(m.group(1))
            # Top-level assignments: FOO = ...
            for m in re.finditer(r'^([A-Z_][A-Z0-9_]*)\s*=', source, re.MULTILINE):
                symbols.add(m.group(1))
        elif ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"):
            # function foo, class Foo, const foo =, export const foo
            for m in re.finditer(r'(?:function|class|const|let|var)\s+([A-Za-z_$]\w*)', source, re.MULTILINE):
                symbols.add(m.group(1))
        elif ext == ".go":
            for m in re.finditer(r'func\s+(?:\([^)]*\)\s+)?([A-Za-z_]\w*)', source, re.MULTILINE):
                symbols.add(m.group(1))
            for m in re.finditer(r'type\s+([A-Za-z_]\w*)\s+(?:struct|interface)', source, re.MULTILINE):
                symbols.add(m.group(1))
        elif ext == ".rs":
            for m in re.finditer(r'(?:^\s*(?:pub\s+)?(?:fn|struct|trait|enum|impl)\s+)([A-Za-z_]\w*)', source, re.MULTILINE):
                symbols.add(m.group(1))
        elif ext in (".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hxx"):
            for m in re.finditer(r'(?:^\s*(?:void|int|char|float|double|bool|auto|static|const)\s+\*?\s*)([A-Za-z_]\w*)\s*\(', source, re.MULTILINE):
                symbols.add(m.group(1))
        elif ext == ".java":
            for m in re.finditer(r'(?:public|private|protected)?\s*(?:static\s+)?(?:class|interface|void|int|String|boolean)\s+([A-Za-z_]\w*)', source, re.MULTILINE):
                symbols.add(m.group(1))

    def _text_extract_imports(self, filepath: str, source: str, imports: set[str]) -> None:
        """Extract imports using regex patterns."""
        for pattern in self._IMPORT_PATTERNS:
            for m in pattern.finditer(source):
                # Each pattern has capture groups; take the first non-None
                for group in m.groups():
                    if group:
                        imports.add(group)
                        break

    def _build_reverse_index(self) -> None:
        """
        Build the reverse dependency index:
        For each symbol defined in a file, find all other files that import
        or reference that symbol's source file or package.

        This is an approximation — true cross-file symbol resolution would
        require full language-specific import resolution. Instead we use
        file-to-file and package-to-file matching.
        """
        # Build a set of all symbol names → files that define them
        symbol_files: dict[str, set[str]] = {}
        for filepath, symbols in self._exports.items():
            for sym in symbols:
                symbol_files.setdefault(sym, set()).add(filepath)

        # For each dependent (file that imports X), if X is a file path
        # or package name defined by a scanned file, link them
        module_to_file: dict[str, str] = {}
        for filepath in self._exports:
            rel = os.path.relpath(filepath, self.workspace_path)
            # Map module paths to files
            module_to_file[rel.replace("/", ".").replace(".py", "").replace(".ts", "").replace(".js", "").replace(".rs", "").replace(".go", "")] = filepath
            module_to_file[rel.replace("/", ".")] = filepath

        # For each import in each file, try to resolve to a known file
        for imp, files in list(self._dependents.items()):
            for filepath in files:
                resolved = self._resolve_import_to_file(imp, module_to_file)
                if resolved:
                    # This file depends on symbols from the resolved file
                    for symbol in self._exports.get(resolved, set()):
                        self._dependents.setdefault(symbol, set()).add(filepath)

    def _resolve_import_to_file(self, imp: str, module_to_file: dict[str, str]) -> Optional[str]:
        """Try to resolve an import string to a known file path."""
        # Direct module path match
        if imp in module_to_file:
            return module_to_file[imp]

        # Try with different prefixes and separators
        imp_normalized = imp.replace("/", ".").replace("\\", ".")
        for key, val in module_to_file.items():
            if key.endswith("." + imp_normalized) or key.endswith("/" + imp_normalized.replace(".", "/")):
                return val
            if imp_normalized.endswith(os.path.splitext(os.path.basename(val))[0]):
                return val

        return None

    def get_impacted_files(self, modified_files: list[str]) -> list[tuple[str, list[str]]]:
        """
        For a list of modified files, find which other files reference their symbols.

        Args:
            modified_files: List of file paths that are being modified.

        Returns:
            List of (impacted_file, [symbols_affected]) tuples.
            Empty list if no impact detected.
        """
        if not self._built:
            self.build()

        impacted: dict[str, set[str]] = {}
        for mf in modified_files:
            resolved = self._resolve_path(mf)
            if resolved is None:
                continue
            symbols = self._exports.get(resolved, set())
            if not symbols:
                # No known symbols — check if this file is directly imported
                file_rel = os.path.relpath(resolved, self.workspace_path)
                # Find any file that imports this file's relative path or module name
                file_module = file_rel.replace("/", ".").replace("\\", ".")
                for imp_key, dep_files in self._dependents.items():
                    if file_rel in imp_key or file_module in imp_key or imp_key.endswith(file_module):
                        for df in dep_files:
                            if df != resolved:
                                impacted.setdefault(df, set()).add(os.path.basename(resolved))
                continue

            for symbol in symbols:
                dep_files = self._dependents.get(symbol, set())
                for df in dep_files:
                    if df != resolved:
                        impacted.setdefault(df, set()).add(symbol)

        return [(f, sorted(syms)) for f, syms in sorted(impacted.items())]

    def _resolve_path(self, filepath: str) -> Optional[str]:
        """Resolve a filepath against the workspace and known exports."""
        # Try as-is
        if filepath in self._exports:
            return filepath

        # Try absolute
        abs_path = os.path.join(self.workspace_path, filepath) if not os.path.isabs(filepath) else filepath
        if abs_path in self._exports:
            return abs_path

        # Try relative
        rel_path = os.path.relpath(abs_path, self.workspace_path)
        for key in self._exports:
            if key.endswith(rel_path) or key.endswith(os.path.basename(filepath)):
                return key

        return None


# ---------------------------------------------------------------------------
# 3. ImpactAnalyzer — Pre-Patch Impact Checker
# ---------------------------------------------------------------------------

class ImpactAnalyzer:
    """
    Checks the impact of patches before they are applied.

    Queries the DependencyGraph for downstream files that may be affected
    by the modifications, and generates human-readable warning messages.
    """

    def __init__(
        self,
        workspace_path: str,
        max_scan_files: int = 500,
        ignore_patterns: Optional[list[str]] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._graph = DependencyGraph(
            workspace_path=workspace_path,
            max_scan_files=max_scan_files,
            ignore_patterns=ignore_patterns,
        )
        self._built = False

    def _ensure_built(self) -> DependencyGraph:
        """Lazily build the dependency graph on first use."""
        if not self._built:
            self._graph.build()
            self._built = True
        return self._graph

    def analyze(self, modified_files: list[str]) -> ImpactResult:
        """
        Analyze the impact of modifying the given files.

        Args:
            modified_files: List of file paths that were or will be modified.

        Returns:
            ImpactResult with impacted files, symbols, and a warning message.
        """
        if not self.enabled or not modified_files:
            return ImpactResult(modified_files=modified_files)

        graph = self._ensure_built()
        impacted = graph.get_impacted_files(modified_files)

        result = ImpactResult(modified_files=modified_files)
        result.total_impacted = len(impacted)
        result.impacted_files = [f for f, _ in impacted]
        # Propagate scan-completeness so callers know whether impacted_files
        # is a complete enumeration or a lower bound.
        result.graph_incomplete = graph.incomplete
        result.files_scanned = graph.files_scanned

        # Build symbol impact map
        symbol_impact: dict[str, list[str]] = {}
        for impacted_file, symbols in impacted:
            for sym in symbols:
                symbol_impact.setdefault(sym, []).append(impacted_file)
        result.symbol_impact = symbol_impact

        # Build warning message
        if impacted:
            warning_parts: list[str] = ["[Impact Analysis] Downstream files may be affected:"]
            for filepath, symbols in impacted:
                rel = os.path.relpath(filepath, self._graph.workspace_path)
                sym_list = ", ".join(symbols[:5])
                if len(symbols) > 5:
                    sym_list += f" ... and {len(symbols) - 5} more"
                warning_parts.append(f"  - {rel} (references: {sym_list})")

            warning_parts.append(
                f"\n  You must now verify these {len(impacted)} file(s) to ensure no "
                f"downstream breakage from the changes to {', '.join(os.path.basename(m) for m in modified_files)}."
            )
            if result.graph_incomplete:
                warning_parts.append(
                    f"  ⚠ Note: dependency graph is INCOMPLETE (scanned {result.files_scanned} of "
                    f"~{graph.max_scan_files}+ files). The list above is a *lower bound* — more "
                    f"files may be affected. Raise `impact.max_scan_files` in config to widen the scan."
                )
            result.warning = "\n".join(warning_parts)

            logger.warning(
                "[impact] Modified %d file(s) → %d downstream file(s) potentially impacted%s.",
                len(modified_files), len(impacted),
                " (graph incomplete — lower bound)" if result.graph_incomplete else "",
            )
        elif result.graph_incomplete:
            # No matches found, but the scan was truncated — caller deserves to know.
            result.warning = (
                f"[Impact Analysis] No downstream impact detected, BUT the dependency "
                f"graph is INCOMPLETE (scanned {result.files_scanned} of "
                f"~{graph.max_scan_files}+ files). Some impacted files may have been "
                f"missed. Raise `impact.max_scan_files` to widen the scan."
            )
            logger.warning(
                "[impact] Scan incomplete (%d/%d files) — 'no impact' result is unreliable.",
                result.files_scanned, graph.max_scan_files,
            )
        else:
            logger.info("[impact] No downstream impact detected for %d modified file(s).", len(modified_files))

        return result

    def analyze_and_warn(self, modified_files: list[str], messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Analyze impact and append a warning system message to the conversation.

        Args:
            modified_files: List of file paths that were modified.
            messages: The current conversation messages list (to append warning to).

        Returns:
            The modified messages list with impact warning appended (if any impact).
        """
        result = self.analyze(modified_files)
        if result.has_impact():
            messages.append({"role": "system", "content": result.warning})
        return messages


# ---------------------------------------------------------------------------
# 4. Language Extension Mapping for Tree-Sitter
# ---------------------------------------------------------------------------

_EXTENSION_TO_TREE_SITTER: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".rs": "rust",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".go": "go",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".java": "java",
}


# ---------------------------------------------------------------------------
# 5. Factory from Config
# ---------------------------------------------------------------------------

def create_impact_analyzer_from_config(
    workspace_path: str,
    config_dict: dict[str, Any],
) -> ImpactAnalyzer:
    """
    Build an ImpactAnalyzer from the 'impact_analysis' section of .harness_config.json.

    Args:
        workspace_path: Absolute path to the workspace.
        config_dict: Merged configuration dictionary.

    Returns:
        Configured ImpactAnalyzer instance.
    """
    ia_cfg = config_dict.get("impact_analysis", {})

    return ImpactAnalyzer(
        workspace_path=workspace_path,
        max_scan_files=ia_cfg.get("max_scan_files", 500),
        ignore_patterns=ia_cfg.get("ignore_patterns", None),
        enabled=ia_cfg.get("enabled", True),
    )