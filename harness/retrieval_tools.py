"""Read-only repository retrieval tools for the native tool-use loop.

Adds *agentic navigation* to the patching / repair loop. Until now the only
way the model could look at the repo was ``read_file`` on a path it already
knew — it could not discover anything. These tools let it search by content
(``grep``), by filename (``glob``), list directories (``list_dir``), jump to
a symbol by name (``find_symbol``, LSP ``workspace/symbol``), outline a file
(``file_outline``, LSP ``documentSymbol``), query the semantic index
(``semantic_search``, :mod:`harness.repo_index`), and inspect git history
(``git_blame`` / ``git_log``).

Every tool is READ-ONLY and best-effort: a missing backend (no ripgrep, no
LSP pool, no built index, a non-git workspace) returns a short, actionable
"unavailable" string rather than raising, so the model learns and moves on —
the same contract the web / MCP / LSP text-DSL skills already use.

Integration (:func:`harness.graph._patching_tool_loop`): these calls are
resolved exactly like ``read_file`` — execute, feed the result back as a
``tool_result``, re-dispatch — until the model stops navigating and emits
patches. They never become :class:`~harness.patcher.PatchBlock` objects
(``tool_calls_to_patch_blocks`` drops unknown names), so the patcher is
unchanged.
"""

from __future__ import annotations

import glob as _glob_mod
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Optional

from harness.observability import emit_event, log_failure

logger = __import__("logging").getLogger("harness.retrieval_tools")

# Directories never worth walking / searching. ripgrep already honours
# .gitignore, but the pure-Python fallbacks and list_dir/glob need this.
_NOISE_DIRS = frozenset({
    ".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "node_modules", "dist", "build", ".venv", "venv", "env", ".tox",
    ".next", ".turbo", "coverage", "htmlcov", ".idea", ".vscode",
})

# LSP SymbolKind → short label (LSP spec §3.17.4). Only the useful ones.
_SYMBOL_KIND = {
    5: "class", 6: "method", 9: "constructor", 11: "interface",
    12: "function", 13: "variable", 14: "constant", 8: "field",
    10: "enum", 7: "property", 23: "struct", 26: "type",
}


@dataclass
class RetrievalToolsConfig:
    """Caps + master switch for the retrieval tools. Read from the
    ``retrieval_tools`` config section; every field has a safe default so a
    missing section Just Works."""

    enabled: bool = True
    max_results: int = 80          # grep / find_symbol match cap
    max_files: int = 200           # glob / list_dir entry cap
    max_bytes: int = 12_000        # hard cap on any single result string
    grep_timeout_s: int = 15
    git_timeout_s: int = 15
    list_dir_depth: int = 2
    semantic_top_k: int = 8
    git_log_max: int = 15

    @classmethod
    def from_config(cls, config: Optional[dict[str, Any]]) -> "RetrievalToolsConfig":
        sec: dict[str, Any] = {}
        if isinstance(config, dict):
            raw = config.get("retrieval_tools")
            if isinstance(raw, dict):
                sec = raw

        def _int(key: str, default: int) -> int:
            try:
                return int(sec.get(key, default))
            except (TypeError, ValueError):
                return default

        return cls(
            enabled=bool(sec.get("enabled", True)),
            max_results=_int("max_results", 80),
            max_files=_int("max_files", 200),
            max_bytes=_int("max_bytes", 12_000),
            grep_timeout_s=_int("grep_timeout_s", 15),
            git_timeout_s=_int("git_timeout_s", 15),
            list_dir_depth=_int("list_dir_depth", 2),
            semantic_top_k=_int("semantic_top_k", 8),
            git_log_max=_int("git_log_max", 15),
        )


# ---------------------------------------------------------------------------
# Tool schemas (same {name, description, input_schema} shape as READ_FILE)
# ---------------------------------------------------------------------------

GREP_SCHEMA: dict[str, Any] = {
    "name": "grep",
    "description": (
        "Search file *contents* across the workspace with a regular "
        "expression and get back matching `path:line: text` lines. Use this "
        "to find where a symbol is defined or used, or any literal/regex "
        "text — it is the fastest way to orient in an unfamiliar repo. "
        "Read-only."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression to search for."},
            "path": {"type": "string", "description": "Optional workspace-relative subdirectory to scope the search (default: whole workspace)."},
            "glob": {"type": "string", "description": "Optional filename filter, e.g. '*.py' or '*.tsx', to restrict which files are searched."},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive match (default false)."},
        },
        "required": ["pattern"],
    },
}

GLOB_SCHEMA: dict[str, Any] = {
    "name": "glob",
    "description": (
        "Find files by name/path pattern (e.g. '**/*.tsx', 'src/**/models/*.py'). "
        "Returns workspace-relative paths. Use this to discover which files "
        "exist before reading them. Read-only."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern; '**' matches directories recursively."},
            "path": {"type": "string", "description": "Optional workspace-relative base directory (default: workspace root)."},
        },
        "required": ["pattern"],
    },
}

LIST_DIR_SCHEMA: dict[str, Any] = {
    "name": "list_dir",
    "description": (
        "List the contents of a directory as an indented tree (bounded "
        "depth). Use this to understand a repo's layout. Read-only."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative directory (default: workspace root)."},
            "depth": {"type": "integer", "minimum": 1, "description": "How many levels deep to descend (default 2)."},
        },
        "required": [],
    },
}

FIND_SYMBOL_SCHEMA: dict[str, Any] = {
    "name": "find_symbol",
    "description": (
        "Find where a symbol (function / class / method / variable) is "
        "DEFINED, by name, across the whole workspace — without needing to "
        "know its file. Uses the language server. Falls back to a hint to "
        "use grep when no language server is running. Read-only."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Symbol name to locate (exact or prefix)."},
        },
        "required": ["name"],
    },
}

FILE_OUTLINE_SCHEMA: dict[str, Any] = {
    "name": "file_outline",
    "description": (
        "Get a file's structural outline — its top-level classes, functions "
        "and methods with line numbers — without reading the whole file. "
        "Token-efficient way to triage a large file before read_file. Uses "
        "the language server. Read-only."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Workspace-relative path to the file to outline."},
        },
        "required": ["file_path"],
    },
}

SEMANTIC_SEARCH_SCHEMA: dict[str, Any] = {
    "name": "semantic_search",
    "description": (
        "Search the repository's semantic index for the code chunks most "
        "relevant to a natural-language query. Complements grep (which is "
        "exact/regex) when you don't know the exact identifier. Requires a "
        "prebuilt index (`teane index build`); returns a hint if none "
        "exists. Read-only."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language description of what you're looking for."},
            "k": {"type": "integer", "minimum": 1, "description": "How many chunks to return (default 8)."},
        },
        "required": ["query"],
    },
}

GIT_BLAME_SCHEMA: dict[str, Any] = {
    "name": "git_blame",
    "description": (
        "Show git blame for a file (optionally a line range): the commit and "
        "author that last touched each line. Use during repair to understand "
        "why code is the way it is. Returns a hint if the workspace is not a "
        "git repo. Read-only."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Workspace-relative file path."},
            "start_line": {"type": "integer", "minimum": 1, "description": "Optional 1-indexed first line."},
            "end_line": {"type": "integer", "minimum": 1, "description": "Optional 1-indexed last line."},
        },
        "required": ["file_path"],
    },
}

GIT_LOG_SCHEMA: dict[str, Any] = {
    "name": "git_log",
    "description": (
        "Show recent git history. With `file_path`, the commits that touched "
        "that file; with `file_path` AND `symbol`, the change history of that "
        "one function/method (git log -L). Read-only."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Optional workspace-relative file path to scope history to."},
            "symbol": {"type": "string", "description": "Optional function/method name; with file_path, shows that symbol's evolution."},
            "max_count": {"type": "integer", "minimum": 1, "description": "Max commits to return (default 15)."},
        },
        "required": [],
    },
}

RETRIEVAL_TOOLS: list[dict[str, Any]] = [
    GREP_SCHEMA, GLOB_SCHEMA, LIST_DIR_SCHEMA, FIND_SYMBOL_SCHEMA,
    FILE_OUTLINE_SCHEMA, SEMANTIC_SEARCH_SCHEMA, GIT_BLAME_SCHEMA, GIT_LOG_SCHEMA,
]

# Names the native tool loop must resolve-and-continue on (like read_file),
# rather than treat as a terminal patch/text response.
RETRIEVAL_TOOL_NAMES: frozenset[str] = frozenset(t["name"] for t in RETRIEVAL_TOOLS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, cfg: RetrievalToolsConfig) -> str:
    if len(text) <= cfg.max_bytes:
        return text
    return text[: cfg.max_bytes] + f"\n... [truncated at {cfg.max_bytes} bytes]"


def _resolve_in_ws(workspace: str, rel: Optional[str]) -> Optional[str]:
    """Resolve ``rel`` under ``workspace``; None if it escapes the root."""
    base = os.path.abspath(workspace)
    target = os.path.abspath(os.path.join(base, (rel or ".").strip() or "."))
    if target != base and not target.startswith(base + os.sep):
        return None
    return target


def _rel(workspace: str, abs_path: str) -> str:
    try:
        return os.path.relpath(abs_path, os.path.abspath(workspace))
    except ValueError:
        return abs_path


def _have_rg() -> bool:
    from shutil import which
    return which("rg") is not None


# ---------------------------------------------------------------------------
# Deterministic resolvers
# ---------------------------------------------------------------------------

def _grep(args: dict[str, Any], workspace: str, cfg: RetrievalToolsConfig) -> str:
    pattern = str(args.get("pattern") or "").strip()
    if not pattern:
        return "Error: grep requires a non-empty 'pattern'."
    sub = args.get("path")
    scope_abs = _resolve_in_ws(workspace, sub if isinstance(sub, str) else ".")
    if scope_abs is None:
        return f"Error: path '{sub}' escapes the workspace."
    ignore_case = bool(args.get("ignore_case"))
    glob_filter = args.get("glob") if isinstance(args.get("glob"), str) else None

    if _have_rg():
        cmd = ["rg", "--line-number", "--no-heading", "--color", "never",
               "--max-columns", "300", "-g", "!.git"]
        if ignore_case:
            cmd.append("-i")
        if glob_filter:
            cmd += ["-g", glob_filter]
        cmd += ["-e", pattern, os.path.relpath(scope_abs, os.path.abspath(workspace))]
        try:
            proc = subprocess.run(
                cmd, cwd=os.path.abspath(workspace), capture_output=True,
                text=True, timeout=cfg.grep_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return f"Error: grep timed out after {cfg.grep_timeout_s}s (narrow the pattern or scope with 'path'/'glob')."
        except Exception as exc:  # noqa: BLE001
            return f"Error: grep failed: {exc}"
        if proc.returncode not in (0, 1):  # 1 = no matches (not an error)
            return f"Error: grep: {(proc.stderr or '').strip()[:300]}"
        lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
        if not lines:
            return f"No matches for /{pattern}/."
        shown = lines[: cfg.max_results]
        header = f"{len(lines)} match(es)" + (f", showing first {cfg.max_results}" if len(lines) > cfg.max_results else "")
        return _truncate(f"[grep /{pattern}/ — {header}]\n" + "\n".join(shown), cfg)

    # Pure-Python fallback (no ripgrep).
    import re as _re
    from fnmatch import fnmatch
    try:
        rx = _re.compile(pattern, _re.IGNORECASE if ignore_case else 0)
    except _re.error as exc:
        return f"Error: invalid regex: {exc}"
    hits: list[str] = []
    for root, dirs, files in os.walk(scope_abs):
        dirs[:] = [d for d in dirs if d not in _NOISE_DIRS]
        for fn in files:
            if glob_filter and not fnmatch(fn, glob_filter):
                continue
            fp = os.path.join(root, fn)
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            hits.append(f"{_rel(workspace, fp)}:{i}: {line.rstrip()[:300]}")
                            if len(hits) >= cfg.max_results:
                                break
            except OSError:
                continue
            if len(hits) >= cfg.max_results:
                break
        if len(hits) >= cfg.max_results:
            break
    if not hits:
        return f"No matches for /{pattern}/."
    return _truncate(f"[grep /{pattern}/ — {len(hits)} match(es) (python fallback)]\n" + "\n".join(hits), cfg)


def _glob(args: dict[str, Any], workspace: str, cfg: RetrievalToolsConfig) -> str:
    pattern = str(args.get("pattern") or "").strip()
    if not pattern:
        return "Error: glob requires a non-empty 'pattern'."
    base = args.get("path")
    base_abs = _resolve_in_ws(workspace, base if isinstance(base, str) else ".")
    if base_abs is None:
        return f"Error: path '{base}' escapes the workspace."
    try:
        matches = _glob_mod.glob(pattern, root_dir=base_abs, recursive=True)
    except Exception as exc:  # noqa: BLE001
        return f"Error: glob failed: {exc}"
    out: list[str] = []
    for m in sorted(matches):
        if any(part in _NOISE_DIRS for part in m.split(os.sep)):
            continue
        abs_m = os.path.join(base_abs, m)
        if os.path.isdir(abs_m):
            continue
        out.append(_rel(workspace, abs_m))
        if len(out) >= cfg.max_files:
            break
    if not out:
        return f"No files match '{pattern}'."
    header = f"{len(out)} file(s)" + (" (capped)" if len(out) >= cfg.max_files else "")
    return _truncate(f"[glob '{pattern}' — {header}]\n" + "\n".join(out), cfg)


def _list_dir(args: dict[str, Any], workspace: str, cfg: RetrievalToolsConfig) -> str:
    rel = args.get("path")
    root_abs = _resolve_in_ws(workspace, rel if isinstance(rel, str) else ".")
    if root_abs is None:
        return f"Error: path '{rel}' escapes the workspace."
    if not os.path.isdir(root_abs):
        return f"Error: '{rel or '.'}' is not a directory."
    try:
        depth = int(args.get("depth", cfg.list_dir_depth))
    except (TypeError, ValueError):
        depth = cfg.list_dir_depth
    depth = max(1, min(depth, 6))
    lines: list[str] = [f"[list_dir '{_rel(workspace, root_abs)}' depth={depth}]"]
    count = 0
    base_level = root_abs.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(root_abs):
        level = root.rstrip(os.sep).count(os.sep) - base_level
        if level >= depth:
            dirs[:] = []
        dirs[:] = sorted(d for d in dirs if d not in _NOISE_DIRS)
        indent = "  " * level
        if root != root_abs:
            lines.append(f"{indent}{os.path.basename(root)}/")
            count += 1
        for fn in sorted(files):
            lines.append(f"{indent}{'  ' if root != root_abs else ''}{fn}")
            count += 1
            if count >= cfg.max_files:
                lines.append(f"... [truncated at {cfg.max_files} entries]")
                return _truncate("\n".join(lines), cfg)
    return _truncate("\n".join(lines), cfg)


def _git(args_list: list[str], workspace: str, cfg: RetrievalToolsConfig) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", "-C", os.path.abspath(workspace), *args_list],
        capture_output=True, text=True, timeout=cfg.git_timeout_s,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _is_git_repo(workspace: str, cfg: RetrievalToolsConfig) -> bool:
    try:
        rc, out, _ = _git(["rev-parse", "--is-inside-work-tree"], workspace, cfg)
        return rc == 0 and out.strip() == "true"
    except Exception:  # noqa: BLE001
        return False


def _git_blame(args: dict[str, Any], workspace: str, cfg: RetrievalToolsConfig) -> str:
    fp = str(args.get("file_path") or "").strip()
    if not fp:
        return "Error: git_blame requires 'file_path'."
    if _resolve_in_ws(workspace, fp) is None:
        return f"Error: path '{fp}' escapes the workspace."
    if not _is_git_repo(workspace, cfg):
        return "git_blame unavailable: the workspace is not a git repository."
    cmd = ["blame", "--date=short"]
    s, e = args.get("start_line"), args.get("end_line")
    if isinstance(s, int) and isinstance(e, int) and s >= 1 and e >= s:
        cmd += ["-L", f"{s},{e}"]
    cmd += ["--", fp]
    try:
        rc, out, err = _git(cmd, workspace, cfg)
    except subprocess.TimeoutExpired:
        return f"Error: git_blame timed out after {cfg.git_timeout_s}s."
    except Exception as exc:  # noqa: BLE001
        return f"Error: git_blame failed: {exc}"
    if rc != 0:
        return f"Error: git_blame: {err.strip()[:300]}"
    lines = out.splitlines()[: cfg.max_results]
    return _truncate(f"[git blame {fp}]\n" + "\n".join(lines), cfg)


def _git_log(args: dict[str, Any], workspace: str, cfg: RetrievalToolsConfig) -> str:
    if not _is_git_repo(workspace, cfg):
        return "git_log unavailable: the workspace is not a git repository."
    fp = args.get("file_path")
    fp = fp.strip() if isinstance(fp, str) else ""
    symbol = args.get("symbol")
    symbol = symbol.strip() if isinstance(symbol, str) else ""
    if fp and _resolve_in_ws(workspace, fp) is None:
        return f"Error: path '{fp}' escapes the workspace."
    try:
        n = int(args.get("max_count", cfg.git_log_max))
    except (TypeError, ValueError):
        n = cfg.git_log_max
    n = max(1, min(n, 100))
    try:
        if symbol and fp:
            rc, out, err = _git(
                ["log", f"-L:{symbol}:{fp}", "--no-patch", f"-n{n}", "--date=short",
                 "--pretty=format:%h %ad %an: %s"], workspace, cfg)
            if rc != 0:  # -L can fail if the funcname regex doesn't match; degrade
                rc, out, err = _git(
                    ["log", f"-n{n}", "--date=short", "--pretty=format:%h %ad %an: %s",
                     "-S", symbol, "--", fp], workspace, cfg)
        elif fp:
            rc, out, err = _git(
                ["log", f"-n{n}", "--date=short", "--pretty=format:%h %ad %an: %s",
                 "--", fp], workspace, cfg)
        else:
            rc, out, err = _git(
                ["log", f"-n{n}", "--date=short", "--pretty=format:%h %ad %an: %s"],
                workspace, cfg)
    except subprocess.TimeoutExpired:
        return f"Error: git_log timed out after {cfg.git_timeout_s}s."
    except Exception as exc:  # noqa: BLE001
        return f"Error: git_log failed: {exc}"
    if rc != 0:
        return f"Error: git_log: {err.strip()[:300]}"
    if not out.strip():
        return "No matching history."
    scope = f" {fp}" + (f":{symbol}" if symbol else "") if fp else ""
    return _truncate(f"[git log{scope}]\n" + out.strip(), cfg)


# ---------------------------------------------------------------------------
# Semantic + LSP resolvers (async)
# ---------------------------------------------------------------------------

async def _semantic_search(args: dict[str, Any], workspace: str, cfg: RetrievalToolsConfig) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return "Error: semantic_search requires a non-empty 'query'."
    try:
        k = int(args.get("k", cfg.semantic_top_k))
    except (TypeError, ValueError):
        k = cfg.semantic_top_k
    k = max(1, min(k, 25))
    try:
        from harness.repo_index import async_query_top_chunks
    except Exception as exc:  # noqa: BLE001
        return f"semantic_search unavailable: repo_index not importable ({exc})."
    try:
        results = await async_query_top_chunks(workspace, query, top_k=k)
    except Exception as exc:  # noqa: BLE001
        return f"Error: semantic_search failed: {exc}"
    if not results:
        return ("No semantic-index results (the index may not be built for this "
                "workspace — run `teane index build`, or use grep for exact text).")
    rendered = []
    for r in results:
        try:
            rendered.append(r.render(content_max_lines=25))
        except Exception:  # noqa: BLE001
            rendered.append(f"### {getattr(r, 'file_path', '?')}\n{getattr(r, 'content', '')[:400]}")
    return _truncate(f"[semantic_search '{query}' — {len(results)} chunk(s)]\n" + "\n".join(rendered), cfg)


async def _find_symbol(args: dict[str, Any], workspace: str, cfg: RetrievalToolsConfig) -> str:
    name = str(args.get("name") or "").strip()
    if not name:
        return "Error: find_symbol requires a 'name'."
    try:
        from harness.lsp_client import get_active_pool, find_definition_by_symbol
    except Exception as exc:  # noqa: BLE001
        return f"find_symbol unavailable: LSP client not importable ({exc}). Use grep instead."
    pool = get_active_pool()
    if pool is None or not pool.healthy():
        return ("find_symbol unavailable: no language server is running for this "
                "workspace (brownfield patch/test flows need a venv, or "
                "tsconfig + node_modules). Use grep to search for the name instead.")
    try:
        locs = await find_definition_by_symbol(pool, name)
    except Exception as exc:  # noqa: BLE001
        return f"Error: find_symbol failed: {exc}. Use grep instead."
    if not locs:
        return f"No definition found for symbol '{name}'. Try grep, or check the spelling/case."
    lines = [f"{loc.get('file')}:{int(loc.get('line', 0)) + 1}" for loc in locs[: cfg.max_results]]
    return _truncate(f"[find_symbol '{name}' — {len(locs)} definition site(s)]\n" + "\n".join(lines), cfg)


async def _file_outline(args: dict[str, Any], workspace: str, cfg: RetrievalToolsConfig) -> str:
    fp = str(args.get("file_path") or "").strip()
    if not fp:
        return "Error: file_outline requires 'file_path'."
    if _resolve_in_ws(workspace, fp) is None:
        return f"Error: path '{fp}' escapes the workspace."
    try:
        from harness.lsp_client import get_active_pool, _flatten_document_symbols
    except Exception as exc:  # noqa: BLE001
        return f"file_outline unavailable: LSP client not importable ({exc}). Use read_file instead."
    pool = get_active_pool()
    if pool is None or not pool.healthy():
        return ("file_outline unavailable: no language server is running. Use "
                "read_file to inspect the file directly.")
    client = pool.client_for_file(fp)
    if client is None:
        return (f"file_outline unavailable for '{fp}': no language server handles this "
                f"file type. Use read_file instead.")
    try:
        raw = await client.document_symbols(fp)
    except Exception as exc:  # noqa: BLE001
        return f"Error: file_outline failed: {exc}. Use read_file instead."
    flat = _flatten_document_symbols(raw)
    if not flat:
        return f"No symbols reported for '{fp}' (empty file, or the server had no outline)."
    lines = []
    for s in flat[: cfg.max_results]:
        kind = _SYMBOL_KIND.get(int(s.get("kind") or 0), "symbol")
        lines.append(f"  L{int(s.get('line', 0)) + 1}: {kind} {s.get('name')}")
    return _truncate(f"[file_outline {fp} — {len(flat)} top-level symbol(s)]\n" + "\n".join(lines), cfg)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_SYNC = {"grep": _grep, "glob": _glob, "list_dir": _list_dir,
         "git_blame": _git_blame, "git_log": _git_log}
_ASYNC = {"semantic_search": _semantic_search, "find_symbol": _find_symbol,
          "file_outline": _file_outline}


async def resolve_retrieval_call(
    call: dict[str, Any],
    workspace: str,
    *,
    config: Optional[dict[str, Any]] = None,
) -> str:
    """Execute one retrieval tool call and return an LLM-facing result string.

    Never raises: any backend failure becomes an ``Error:`` / "unavailable"
    string so the tool loop keeps going. Emits ``tool_call_succeeded`` /
    ``tool_call_failed`` telemetry mirroring the read_file path.
    """
    name = str(call.get("name") or "")
    args = call.get("input")
    if not isinstance(args, dict):
        args = {}
    cfg = RetrievalToolsConfig.from_config(config)
    try:
        if name in _SYNC:
            result = _SYNC[name](args, workspace, cfg)
        elif name in _ASYNC:
            result = await _ASYNC[name](args, workspace, cfg)
        else:
            result = f"Error: unknown retrieval tool '{name}'."
    except Exception as exc:  # noqa: BLE001 — resolver must never break the loop
        result = f"Error: {name} raised {type(exc).__name__}: {exc}"

    try:
        if result.startswith("Error:"):
            log_failure("tool_call_failed", tool_name=name, reason=result[:200])
        else:
            emit_event("tool_call_succeeded", tool_name=name)
    except Exception:  # noqa: BLE001 — telemetry must never block
        pass
    return result
