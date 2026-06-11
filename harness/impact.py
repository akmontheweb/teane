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

import json
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

    # Map harness language tag → tree_sitter_language_pack grammar name.
    # Anything not listed here falls back to regex extraction.
    _GRAMMAR_NAMES: dict[str, str] = {
        "python": "python",
        "javascript": "javascript",
        "jsx": "javascript",
        "typescript": "typescript",
        "tsx": "tsx",
        "java": "java",
        "go": "go",
        "rust": "rust",
        "dart": "dart",
    }

    def _try_tree_sitter_extract(
        self,
        filepath: str,
        source: str,
        lang: str,
        symbols: set[str],
    ) -> bool:
        """Try to extract symbols from a file using tree-sitter AST.

        Uses tree_sitter_language_pack to resolve grammars at runtime so a
        single dependency covers every language in the stack. If the pack
        isn't installed or the requested grammar isn't bundled, the caller
        falls back to the regex extractor.
        """
        grammar_name = self._GRAMMAR_NAMES.get(lang)
        if grammar_name is None:
            return False

        try:
            import tree_sitter
            from tree_sitter_language_pack import get_language
        except ImportError as exc:
            logger.debug(
                "[impact] tree_sitter_language_pack not installed for %s: %s",
                filepath, exc,
            )
            return False

        try:
            ts_lang = get_language(grammar_name)  # type: ignore[arg-type]
            parser = tree_sitter.Parser()
            parser.language = ts_lang
            tree = parser.parse(source.encode("utf-8"))
            self._extract_symbols_from_ast(tree.root_node, lang, symbols)
            return True
        except Exception as exc:  # noqa: BLE001
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
                if child.type in (
                    "function_declaration", "class_declaration", "method_definition",
                    "interface_declaration", "type_alias_declaration", "enum_declaration",
                    "generator_function_declaration",
                ):
                    for sub in child.children:
                        # JS uses `identifier`, TS uses `type_identifier` for
                        # class/interface/type-alias names.
                        if sub.type in ("identifier", "type_identifier"):
                            symbols.add(sub.text.decode("utf-8"))
                            break
            elif lang == "rust":
                if child.type in ("function_item", "struct_item", "trait_item", "impl_item", "enum_item"):
                    for sub in child.children:
                        if sub.type in ("identifier", "type_identifier"):
                            symbols.add(sub.text.decode("utf-8"))
                            break
            elif lang == "go":
                if child.type in ("function_declaration", "method_declaration", "type_declaration"):
                    for sub in child.children:
                        if sub.type in ("identifier", "field_identifier", "type_identifier", "type_spec"):
                            # type_spec wraps an identifier — drill one level
                            if sub.type == "type_spec":
                                for inner in sub.children:
                                    if inner.type == "type_identifier":
                                        symbols.add(inner.text.decode("utf-8"))
                                        break
                            else:
                                symbols.add(sub.text.decode("utf-8"))
                            break
            elif lang == "java":
                if child.type in ("class_declaration", "interface_declaration", "method_declaration",
                                   "enum_declaration", "record_declaration"):
                    for sub in child.children:
                        if sub.type == "identifier":
                            symbols.add(sub.text.decode("utf-8"))
                            break
            elif lang == "dart":
                if child.type in ("class_definition", "function_signature", "method_signature",
                                   "enum_declaration", "mixin_declaration", "extension_declaration"):
                    for sub in child.children:
                        if sub.type in ("identifier", "type_identifier"):
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
# Project-type detection — used by the graph router to decide whether
# the workspace should run through the docker-compose deployment pipeline
# or skip it (mobile / pure library projects).
# ---------------------------------------------------------------------------

def _is_flutter_project(workspace_path: str) -> bool:
    """Return True when the workspace looks like a Flutter project.

    Heuristic: ``pubspec.yaml`` at the root AND a ``lib/`` directory. The
    Flutter scaffolding always produces both. Pure Dart server projects
    also have these, which is fine — the deploy pipeline doesn't fit
    them either.

    Used by ``harness.graph.route_after_compiler`` to send Flutter
    builds straight to END after a successful test run instead of
    routing them through ``security_scan_node`` → ``deployment_node``,
    which would try to ``docker compose up`` a mobile artifact and fail.
    """
    try:
        return (
            os.path.isfile(os.path.join(workspace_path, "pubspec.yaml"))
            and os.path.isdir(os.path.join(workspace_path, "lib"))
        )
    except (OSError, TypeError):
        return False


def _detect_workspace_stack(workspace_path: str) -> set[str]:
    """Return the set of stack tags applicable to this workspace.

    Used by ``harness.graph._load_skills_markdown`` to filter shipped
    skill files by their ``applies_to:`` frontmatter, so the LLM
    system prompt only includes stack-relevant guidance.

    Detection scans manifest files and their declared dependencies. The
    returned tag taxonomy:
        Languages:           python, node, typescript, java, dart, go, rust
        Backend frameworks:  fastapi, django, spring, express, nest, fastify
        Frontend frameworks: react, vue, angular
        Markup / style:      html, css, tailwind
        Mobile platforms:    ios, android   (target-platform tags — set
                             whenever the workspace builds for that
                             platform via Flutter, React Native, or
                             native Swift/Kotlin)
        Mobile frameworks:   flutter
        Databases:           postgres, redis, mysql
        Build tools:         maven, gradle

    Detection is best-effort and silent — unparseable manifests just
    don't contribute tags. Returns an empty set for non-directory
    inputs so callers can chain ``_detect_workspace_stack(...) or set()``.
    """
    tags: set[str] = set()
    if not workspace_path or not os.path.isdir(workspace_path):
        return tags

    def _read(relpath: str) -> str:
        try:
            with open(os.path.join(workspace_path, relpath), encoding="utf-8", errors="replace") as f:
                # Cap reads at 256 KB — manifests are tiny, anything bigger
                # is probably a wrong file or an attack vector.
                return f.read(256 * 1024)
        except OSError:
            return ""

    def _exists(relpath: str) -> bool:
        return os.path.isfile(os.path.join(workspace_path, relpath))

    # Flutter (pubspec.yaml + lib/) — also tagged dart for the Dart skill.
    if _is_flutter_project(workspace_path):
        tags.update({"flutter", "dart"})
    elif _exists("pubspec.yaml"):
        # Pure Dart project (server-side, CLI tool, etc.)
        tags.add("dart")

    # Django — manage.py is the canonical signal.
    if _exists("manage.py"):
        tags.update({"python", "django"})

    # Go / Rust / Java build tools.
    if _exists("go.mod"):
        tags.add("go")
    if _exists("Cargo.toml"):
        tags.add("rust")
    if _exists("pom.xml"):
        tags.update({"java", "maven"})
    if _exists("build.gradle") or _exists("build.gradle.kts"):
        tags.update({"java", "gradle"})

    # Python manifests — scan content for frameworks and DB drivers.
    py_content_parts: list[str] = []
    for fname in ("requirements.txt", "pyproject.toml", "setup.py", "Pipfile", "Pipfile.lock"):
        if _exists(fname):
            tags.add("python")
            py_content_parts.append(_read(fname).lower())
    py_blob = "\n".join(py_content_parts)
    if py_blob:
        if "fastapi" in py_blob:
            tags.add("fastapi")
        if "django" in py_blob:
            tags.add("django")
        if any(pkg in py_blob for pkg in ("psycopg", "asyncpg")):
            tags.add("postgres")
        if "pgvector" in py_blob:
            tags.add("postgres")
        # Match "redis" as a word boundary so we don't get fooled by
        # "redirect" or "rediscover" in unrelated package names.
        if re.search(r'\bredis\b', py_blob):
            tags.add("redis")
        if any(pkg in py_blob for pkg in ("pymysql", "mysql-connector", "mysqlclient", "aiomysql")):
            tags.add("mysql")

    # Node.js — parse package.json declared deps.
    pkg_content = _read("package.json")
    if pkg_content:
        tags.add("node")
        try:
            data = json.loads(pkg_content)
            deps: dict[str, Any] = {}
            for key in ("dependencies", "devDependencies", "peerDependencies"):
                if isinstance(data.get(key), dict):
                    deps.update(data[key])
            if "react" in deps or "next" in deps:
                tags.add("react")
            if "vue" in deps or "nuxt" in deps:
                tags.add("vue")
            if "@angular/core" in deps:
                tags.add("angular")
            if "express" in deps:
                tags.add("express")
            if "@nestjs/core" in deps:
                tags.add("nest")
            if "fastify" in deps:
                tags.add("fastify")
            # TypeScript — declared dep is the strongest signal; the
            # tsconfig.json fallback below catches projects that consume
            # TS only via a build tool (vite, esbuild) without a direct dep.
            if "typescript" in deps:
                tags.add("typescript")
            # Tailwind — declared dep covers the common case; the
            # tailwind.config.* fallback below catches projects that
            # vendor the CLI or pull it transitively.
            if "tailwindcss" in deps:
                tags.add("tailwind")
            # React Native targets BOTH iOS and Android by default.
            # Expo apps too — they pull react-native transitively.
            if "react-native" in deps or "expo" in deps:
                tags.update({"ios", "android"})
            # DB clients
            if "pg" in deps or "postgres" in deps or "@vercel/postgres" in deps:
                tags.add("postgres")
            if "redis" in deps or "ioredis" in deps:
                tags.add("redis")
            if "mysql" in deps or "mysql2" in deps or "mariadb" in deps:
                tags.add("mysql")
        except (json.JSONDecodeError, ValueError):
            pass

    # TypeScript fallback — tsconfig at root is the canonical signal even
    # when package.json doesn't declare TS as a direct dep (vite/esbuild
    # toolchains often pull it transitively).
    if _exists("tsconfig.json"):
        tags.add("typescript")

    # Tailwind fallback — config file at root.
    if (
        _exists("tailwind.config.js")
        or _exists("tailwind.config.ts")
        or _exists("tailwind.config.cjs")
        or _exists("tailwind.config.mjs")
    ):
        tags.add("tailwind")

    # HTML / CSS — any frontend framework workspace produces and consumes
    # both, so style guidance for HTML semantics and CSS layout always
    # applies. Standalone .html / .css at the workspace root catches plain
    # static sites with no JS framework.
    if tags & {"react", "vue", "angular"}:
        tags.update({"html", "css"})
    else:
        try:
            for entry in os.listdir(workspace_path):
                lower = entry.lower()
                if lower.endswith(".html") or lower.endswith(".htm"):
                    tags.add("html")
                if lower.endswith(".css") or lower.endswith(".scss") or lower.endswith(".sass"):
                    tags.add("css")
        except OSError:
            pass

    # Mobile target platforms — set whenever the workspace builds for
    # iOS and/or Android. Sources:
    #   * Flutter projects always create ios/ and android/ sub-projects
    #     unless the team explicitly removed one (rare).
    #   * Native iOS: Podfile + *.xcodeproj / *.xcworkspace at the root.
    #     Swift Package Manager projects with iOS deployment targets are
    #     also iOS, but distinguishing those from server-side Swift
    #     packages requires reading Package.swift content — defer for now.
    #   * Native Android: settings.gradle with an `:app` include, or a
    #     root build.gradle whose content references com.android.application.
    #     AndroidManifest.xml in the tree is the irrefutable signal.
    def _isdir(relpath: str) -> bool:
        return os.path.isdir(os.path.join(workspace_path, relpath))

    if "flutter" in tags:
        # Flutter project — check which platform folders were kept.
        if _isdir("ios"):
            tags.add("ios")
        if _isdir("android"):
            tags.add("android")

    # Native iOS markers (works for non-Flutter SwiftUI / UIKit apps too).
    if _exists("Podfile") or _exists("ios/Podfile"):
        tags.add("ios")
    try:
        for entry in os.listdir(workspace_path):
            if entry.endswith(".xcodeproj") or entry.endswith(".xcworkspace"):
                tags.add("ios")
                break
    except OSError:
        pass

    # Native Android markers.
    if _exists("AndroidManifest.xml") or _exists("app/AndroidManifest.xml"):
        tags.add("android")
    if _exists("app/build.gradle") or _exists("app/build.gradle.kts"):
        tags.add("android")
    settings_gradle = _read("settings.gradle") + _read("settings.gradle.kts")
    if "':app'" in settings_gradle or '":app"' in settings_gradle:
        tags.add("android")
    root_gradle_blob = "\n".join(
        _read(f) for f in ("build.gradle", "build.gradle.kts", "app/build.gradle", "app/build.gradle.kts")
    )
    if (
        "com.android.application" in root_gradle_blob
        or "com.android.library" in root_gradle_blob
    ):
        tags.add("android")

    # Java/Spring — scan pom.xml or build.gradle for spring-boot deps.
    java_blob = "\n".join(_read(f) for f in ("pom.xml", "build.gradle", "build.gradle.kts"))
    if "spring-boot" in java_blob.lower():
        tags.update({"java", "spring"})

    # docker-compose.yml service hints for databases not declared in any
    # manifest (common when the app talks to a sidecar by URL only).
    compose_blob = "\n".join(
        _read(f) for f in ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml")
    ).lower()
    if compose_blob:
        if "postgres" in compose_blob or "postgis" in compose_blob:
            tags.add("postgres")
        if "redis:" in compose_blob or "redis/" in compose_blob:
            tags.add("redis")
        if "mysql" in compose_blob or "mariadb" in compose_blob:
            tags.add("mysql")

    return tags


# ---------------------------------------------------------------------------
# 3a. Source-Root Detection
# ---------------------------------------------------------------------------

# Source-file extensions across the supported stacks. Used to find the
# dominant top-level folder containing project code.
_SOURCE_FILE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".pyi",
    ".js", ".mjs", ".cjs", ".jsx",
    ".ts", ".tsx",
    ".go",
    ".java",
    ".rs",
    ".dart",
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp",
})

# Top-level directories that should never be considered a source root even
# when they're full of code. Generated artefacts, vendored deps, docs, etc.
_NEVER_SOURCE_DIRS: frozenset[str] = frozenset({
    "tests", "test", "__tests__", "spec", "specs",
    "__pycache__", "node_modules", "vendor", "target",
    "build", "dist", "out",
    ".venv", "venv", "env", ".env",
    ".git", ".svn", ".hg",
    ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "docs", "doc", "examples", "example", "scripts", "tools",
    "migrations", "fixtures", "deployment", "infra",
    "ios", "android",     # mobile platforms
    "public", "static",   # frontend assets
})

# Directory names we BIAS TOWARD when picking a source root. A folder named
# `app/` or `src/` beats a folder named `random_thing/` even when the latter
# has marginally more source files — to avoid locking onto an unrelated
# vendored tree.
_PREFERRED_SOURCE_NAMES: frozenset[str] = frozenset({
    "app", "src", "lib", "pkg", "source", "sources", "internal",
})

_MAX_FILES_PER_SCAN = 5000


def _detect_source_root(workspace_path: str) -> Optional[str]:
    """Return the dominant top-level directory containing source files.

    Used by graph._build_system_prompt and patching_node/repair_node to
    constrain LLM-generated code to the workspace's existing layout —
    when this returns `"app"`, the LLM is told (and the patcher
    enforces) that all new source modules go under `app/` rather than
    landing at workspace root.

    Detection logic:
      1. Skip directories in ``_NEVER_SOURCE_DIRS`` and any starting
         with ``.``.
      2. For every remaining top-level dir AND the workspace root
         itself, count source files (extensions in
         ``_SOURCE_FILE_EXTENSIONS``), recursing up to
         ``_MAX_FILES_PER_SCAN`` files.
      3. Bias toward a preferred name (``app`` / ``src`` / ``lib`` /
         ``pkg`` / ``source`` / ``sources`` / ``internal``) when it has
         at least one source file: it wins over any other candidate.
      4. Otherwise pick the directory with the most source files,
         provided it dominates (≥ 80% of all non-root source files OR
         > 3 files vs. 0 in every other candidate).
      5. A top-level directory containing ``__init__.py`` is treated
         as a Python package and qualifies as a source root in its own
         right (even with no other .py files yet) — this catches the
         standard "package_name/" layout that LLMs produce for greenfield
         projects.
      6. Return ``None`` when the workspace is flat (all source at
         root), empty (no source files anywhere), or genuinely
         ambiguous (no clear leader).

    Returns the **directory name** (no trailing slash, no path prefix),
    so callers compose ``f"{root}/"`` themselves for patcher allowlists.
    """
    if not workspace_path or not os.path.isdir(workspace_path):
        return None

    try:
        entries = os.listdir(workspace_path)
    except OSError:
        return None

    counts: dict[str, int] = {}
    package_dirs: set[str] = set()
    files_scanned = 0
    root_source_count = 0

    for entry in entries:
        full = os.path.join(workspace_path, entry)
        if os.path.isfile(full):
            if os.path.splitext(entry)[1].lower() in _SOURCE_FILE_EXTENSIONS:
                root_source_count += 1
                files_scanned += 1
                if files_scanned >= _MAX_FILES_PER_SCAN:
                    break
            continue
        if not os.path.isdir(full):
            continue
        if entry.startswith(".") or entry in _NEVER_SOURCE_DIRS:
            continue

        # Python-package signal: a directory carrying __init__.py is a
        # source root candidate even before it contains user code.
        if os.path.isfile(os.path.join(full, "__init__.py")):
            package_dirs.add(entry)

        dir_count = 0
        for sub_root, sub_dirs, sub_files in os.walk(full):
            # Prune ignored subdirectories in-place so os.walk doesn't recurse.
            sub_dirs[:] = [
                d for d in sub_dirs
                if not d.startswith(".") and d not in _NEVER_SOURCE_DIRS
            ]
            for fname in sub_files:
                if os.path.splitext(fname)[1].lower() in _SOURCE_FILE_EXTENSIONS:
                    dir_count += 1
                    files_scanned += 1
                    if files_scanned >= _MAX_FILES_PER_SCAN:
                        break
            if files_scanned >= _MAX_FILES_PER_SCAN:
                break
        counts[entry] = dir_count
        if files_scanned >= _MAX_FILES_PER_SCAN:
            break

    # Preferred-name bias: if any preferred name has ≥1 source file, it wins
    # over any non-preferred candidate. Among preferred names, the one with
    # the most files wins.
    preferred_candidates = {
        name: cnt for name, cnt in counts.items()
        if name in _PREFERRED_SOURCE_NAMES and cnt > 0
    }
    if preferred_candidates:
        best_preferred = max(preferred_candidates.items(), key=lambda kv: kv[1])
        return best_preferred[0]

    # Python-package signal — if exactly one top-level dir has __init__.py,
    # that's the source root even if it currently has zero scanned source
    # files (e.g. a freshly created package with only sub-packages so far).
    # When multiple top-level packages exist, pick the one with the most
    # source files so we don't lock onto a tests/ shim or a scratch dir.
    if package_dirs:
        if len(package_dirs) == 1:
            return next(iter(package_dirs))
        best_pkg = max(package_dirs, key=lambda name: counts.get(name, 0))
        if counts.get(best_pkg, 0) > 0:
            return best_pkg

    # No source files at all → no opinion.
    total_non_root = sum(counts.values())
    if total_non_root == 0 and root_source_count == 0:
        return None

    # No preferred-name match. Fall back to "the dominant non-preferred dir,
    # if it dominates clearly." Used for workspaces with stack-specific
    # roots like `cmd/` for Go or unusual project layouts.
    if not counts:
        return None
    best_name, best_count = max(counts.items(), key=lambda kv: kv[1])
    if best_count == 0:
        return None
    # Domination test: best is ≥ 80% of all non-root, OR best is > 3 and the
    # runner-up is 0.
    second_best = sorted(counts.values(), reverse=True)[1] if len(counts) > 1 else 0
    dominates = (
        (total_non_root > 0 and best_count >= 0.8 * total_non_root and best_count > 0)
        or (best_count > 3 and second_best == 0)
    )
    if not dominates:
        return None
    return best_name


def _is_greenfield_workspace(workspace_path: str) -> bool:
    """True when the workspace has no source files anywhere — a true
    greenfield project where the LLM is about to scaffold from scratch.

    Used by the patcher allowlist builder to widen the allowed write set:
    on greenfield, we let the LLM pick any reasonable top-level layout
    (e.g. ``task_dispatcher/`` for a project named TaskDispatcher) rather
    than forcing it into the conservative ``src/``/``lib/``/``app/`` set
    that would reject the standard Python package-at-root layout.
    """
    if not workspace_path or not os.path.isdir(workspace_path):
        return False
    files_seen = 0
    try:
        for sub_root, sub_dirs, sub_files in os.walk(workspace_path):
            sub_dirs[:] = [
                d for d in sub_dirs
                if not d.startswith(".") and d not in _NEVER_SOURCE_DIRS
            ]
            for fname in sub_files:
                if os.path.splitext(fname)[1].lower() in _SOURCE_FILE_EXTENSIONS:
                    return False
                files_seen += 1
                if files_seen >= _MAX_FILES_PER_SCAN:
                    return True
    except OSError:
        return False
    return True


def _workspace_basename_variants(workspace_path: str) -> list[str]:
    """Return plausible package-directory names derived from the workspace
    folder basename.

    For ``/path/to/TaskDispatcher`` returns
    ``["TaskDispatcher", "taskdispatcher", "task_dispatcher"]`` — the
    natural Python-package and JS-module shapes an LLM is likely to pick
    when scaffolding code into a project named TaskDispatcher.

    Used by the patcher allowlist builder so a greenfield run doesn't
    fail closed when the LLM picks the obvious snake_case name.
    """
    if not workspace_path:
        return []
    base = os.path.basename(os.path.abspath(workspace_path))
    if not base or base in {"/", "."}:
        return []
    variants: list[str] = [base]
    lowered = base.lower()
    if lowered != base:
        variants.append(lowered)
    # CamelCase → snake_case
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", base).lower()
    if snake not in variants:
        variants.append(snake)
    # Dash to underscore (e.g. "task-dispatcher" → "task_dispatcher")
    if "-" in base:
        dashless = base.replace("-", "_").lower()
        if dashless not in variants:
            variants.append(dashless)
    return variants


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
    ".dart": "dart",
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