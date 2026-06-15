"""
Deterministic autofix pass: apply compiler suggestions, missing-import imports,
and known-safe security finding fixes WITHOUT spending an LLM call.

Three dispatchers, dispatched per-diagnostic by apply_autofixes():

    R1 (compiler)  — _try_compiler_suggestion
        Consumes the DiagnosticObject.suggested_fix populated by the
        rustc / gcc / clang parsers (only when applicability is
        "machine-applicable"). Returns a REPLACE_BLOCK PatchBlock built
        from (file, span, replacement).

    R2 (import)    — _try_missing_import
        Recognises common "name undefined" / "cannot find symbol" / "TS2304"
        / "E0425" / "go: undefined: X" / "cannot find symbol" (java)
        diagnostics. Grep-walks the workspace for top-level definitions
        of the missing symbol. If EXACTLY ONE definition exists outside
        the offending file, emits an INSERT_AT_BLOCK at the top of the
        offending file with the language-appropriate import statement.

    R3 (security)  — _try_security_autofix
        Fires only when the diagnostic's error_code carries a scanner
        prefix (BANDIT:, GITLEAKS:, TRIVY:). Dispatched data-driven via
        the _SECURITY_FIX_TABLE registry. Initial coverage:
            - bandit B201 (Flask debug=True) → flip to False
            - bandit B602 (shell=True with list args) → flip to False
            - gitleaks any rule → DELETE the offending line + add
              <RULE_ID>=<placeholder> to .env.example
            - trivy dep-vuln with FixedVersion → bump pin in the
              manifest file (requirements.txt / package.json / go.mod /
              Cargo.toml)

    R6 (web asset) — _try_asset_reference_fix
        Fires when error_code == "WEB_ASSET_REF" (produced by the
        web-asset reference scanner in lintgate). Rewrites the broken
        local reference to the suggested path if (a) a suggestion exists
        and (b) the raw reference appears exactly once in the referring
        file. Otherwise returns None — the diagnostic escalates to the
        LLM, which has full context to choose between fixing the path
        and creating the missing asset.

Each dispatcher returns Optional[PatchBlock]. apply_autofixes() funnels every
emitted PatchBlock through the existing HybridPatcher so AST safety,
allowlist gating, and idempotency-on-resume are all inherited unchanged.

The deliberate exclusions (R3):
    - Bandit B608 (SQL injection): parameterised-query rewrites require
      context the autofixer can't capture.
    - Trivy findings with no FixedVersion: nothing to bump to.
    - shell=True where args are a string-concatenation: skipping shell
      escaping needs human judgment.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from harness.patcher import (
    OperationType,
    PatchBlock,
    PatchResult,
    Placement,
    TextPatcher,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class AutofixResult:
    """One successful application of an autofix to disk.

    Returned by apply_autofixes for telemetry and to let callers append
    a "we auto-fixed X" system message into the conversation so the LLM
    (if it still gets called) does not try to fix the same issue again.
    """
    diagnostic_index: int          # which input diagnostic was resolved
    fix_kind: str                  # "compiler" | "import" | "security"
    rule_id: str                   # for telemetry / system message
    file: str
    patch_block: PatchBlock
    apply_status: PatchResult


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def apply_autofixes(
    diagnostics: list[dict[str, Any]],
    workspace_path: str,
) -> tuple[list[dict[str, Any]], list[AutofixResult]]:
    """Walk each diagnostic, attempt every dispatcher, apply any success.

    Args:
        diagnostics: A list of DiagnosticObjectDict dicts (the shape the
            graph carries in state["compiler_errors"]). May include the
            optional ``suggested_fix`` field hoisted by the rustc / gcc
            parsers.
        workspace_path: Absolute path to the workspace root. Used both for
            resolving relative file paths in patches and for the
            R2 symbol grep.

    Returns:
        Tuple of (unhandled_diagnostics, applied_fixes). ``unhandled`` is
        the diagnostics list with every successfully-fixed entry removed,
        ready to feed into the LLM repair prompt.
    """
    if not diagnostics:
        return [], []

    # Use the TextPatcher directly rather than the HybridPatcher: autofix
    # already knows exactly which bytes to replace (compiler-suggested
    # span, surgical rule-driven swap, or single-line delete), so the
    # AST overlay's structural-uniqueness check would flag overlapping
    # AST nodes (e.g. expression_statement vs. call_expression with the
    # same text) and reject otherwise-valid patches. The atomic-write
    # and idempotency guarantees of TextPatcher are sufficient here.
    patcher = TextPatcher(workspace_path)

    unhandled: list[dict[str, Any]] = []
    applied: list[AutofixResult] = []

    for idx, diag in enumerate(diagnostics):
        candidate = _try_compiler_suggestion(diag, workspace_path)
        fix_kind = "compiler"
        rule_id = str(diag.get("error_code", ""))
        if candidate is None:
            candidate = _try_missing_import(diag, workspace_path)
            if candidate is not None:
                fix_kind = "import"
        if candidate is None:
            candidate = _try_security_autofix(diag, workspace_path)
            if candidate is not None:
                fix_kind = "security"
        if candidate is None:
            candidate = _try_missing_dep(diag, workspace_path)
            if candidate is not None:
                fix_kind = "dep"
        if candidate is None:
            candidate = _try_dep_resolution_conflict(diag, workspace_path)
            if candidate is not None:
                fix_kind = "dep"
        if candidate is None:
            candidate = _try_asset_reference_fix(diag, workspace_path)
            if candidate is not None:
                fix_kind = "web_asset"

        if candidate is None:
            unhandled.append(diag)
            continue

        result = await _apply_block(patcher, candidate)
        if result.success:
            applied.append(AutofixResult(
                diagnostic_index=idx,
                fix_kind=fix_kind,
                rule_id=rule_id,
                file=candidate.file,
                patch_block=candidate,
                apply_status=result,
            ))
            logger.info(
                "[autofix] %s fix landed for %s (rule=%s): %s",
                fix_kind, candidate.file, rule_id, result.message,
            )
        else:
            # Apply failed (ambiguity, file moved, idempotency miss).
            # Hand the diagnostic to the LLM untouched so it can still try.
            logger.debug(
                "[autofix] %s fix did not apply for %s (rule=%s): %s",
                fix_kind, candidate.file, rule_id, result.error,
            )
            unhandled.append(diag)

    return unhandled, applied


async def _apply_block(patcher: TextPatcher, block: PatchBlock) -> PatchResult:
    """Dispatch a PatchBlock through the TextPatcher by operation type."""
    if block.operation == OperationType.CREATE_FILE:
        return await patcher.create_file(block.file, block.content)
    if block.operation == OperationType.REPLACE_BLOCK:
        return await patcher.replace_block(block.file, block.search, block.replace)
    if block.operation == OperationType.DELETE_BLOCK:
        return await patcher.delete_block(block.file, block.search)
    if block.operation == OperationType.INSERT_AT_BLOCK:
        return await patcher.insert_at_block(
            block.file, block.anchor, block.placement, block.content,
        )
    return PatchResult(
        success=False,
        file=block.file,
        operation=block.operation,
        error=f"unknown operation: {block.operation}",
    )


def autofix_system_message(applied: list[AutofixResult]) -> str:
    """Build a one-shot system message describing what was auto-fixed.

    Appended to the conversation by repair_node / security_scan_node when
    one or more autofixes landed, so the LLM (if it still gets called)
    does not try to re-fix what the deterministic pass already fixed.
    """
    if not applied:
        return ""
    lines = [
        f"[autofix] {len(applied)} finding(s) resolved deterministically — "
        "do not re-attempt these:",
    ]
    for r in applied:
        lines.append(
            f"  - [{r.fix_kind}] {r.rule_id or '(no rule_id)'} "
            f"@ {r.file}: {r.apply_status.message or 'applied'}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# R1 — Compiler suggestion dispatcher
# ---------------------------------------------------------------------------

def _relative_to_workspace(file: str, workspace_path: str) -> Optional[str]:
    """Convert an absolute path inside the workspace to a relative form.

    The patcher's safe_resolve rejects absolute paths outright — every
    PatchBlock we emit must use the relative form. Returns None when the
    file is outside the workspace (rejected).
    """
    if not file:
        return None
    if not os.path.isabs(file):
        # Already relative — verify it stays inside the workspace.
        candidate_abs = os.path.realpath(os.path.join(workspace_path, file))
    else:
        candidate_abs = os.path.realpath(file)
    workspace_real = os.path.realpath(workspace_path)
    try:
        common = os.path.commonpath([candidate_abs, workspace_real])
    except ValueError:
        return None
    if common != workspace_real:
        return None
    rel = os.path.relpath(candidate_abs, workspace_real)
    if rel.startswith("..") or rel == ".":
        return None
    return rel


def _try_compiler_suggestion(
    diag: dict[str, Any],
    workspace_path: str,
) -> Optional[PatchBlock]:
    """Lift a machine-applicable suggested_fix into a REPLACE_BLOCK.

    Maybe-incorrect / unspecified suggestions are explicitly passed through
    to the LLM — they need judgement the autofixer doesn't have.
    """
    sf = diag.get("suggested_fix")
    if not isinstance(sf, dict):
        return None
    applicability = str(sf.get("applicability", "")).lower().replace("_", "-")
    if applicability != "machine-applicable":
        return None

    file = str(diag.get("file", "") or "")
    if not file:
        return None
    replacement = sf.get("replacement", "")
    if replacement is None:
        return None

    rel_file = _relative_to_workspace(file, workspace_path)
    if rel_file is None:
        return None
    file_path = os.path.join(workspace_path, rel_file)
    if not os.path.isfile(file_path):
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            source = fh.read()
    except OSError:
        return None

    span = _slice_span(
        source,
        int(sf.get("span_start_line", 0) or 0),
        int(sf.get("span_start_col", 0) or 0),
        int(sf.get("span_end_line", 0) or 0),
        int(sf.get("span_end_col", 0) or 0),
    )
    if span is None:
        return None
    if not span.strip() and not str(replacement).strip():
        # Replacing whitespace with whitespace is a no-op
        return None

    return PatchBlock(
        operation=OperationType.REPLACE_BLOCK,
        file=rel_file,
        search=span,
        replace=str(replacement),
    )


def _slice_span(
    source: str,
    start_line: int,
    start_col: int,
    end_line: int,
    end_col: int,
) -> Optional[str]:
    """Extract the substring of ``source`` covered by a 1-indexed span.

    Returns None when any coordinate is out of range. The replacement
    will then bypass autofix and fall through to the LLM.
    """
    if start_line <= 0 or end_line <= 0 or start_col <= 0 or end_col <= 0:
        return None
    lines = source.splitlines(keepends=True)
    if start_line > len(lines) or end_line > len(lines):
        return None
    if start_line == end_line:
        line = lines[start_line - 1]
        # Strip the trailing newline before column-slicing
        no_eol = line.rstrip("\n").rstrip("\r")
        if start_col - 1 > len(no_eol) or end_col - 1 > len(no_eol):
            return None
        return no_eol[start_col - 1:end_col - 1]
    # Multi-line span
    parts: list[str] = []
    first = lines[start_line - 1]
    first_no_eol = first.rstrip("\n").rstrip("\r")
    if start_col - 1 > len(first_no_eol):
        return None
    parts.append(first_no_eol[start_col - 1:])
    # Preserve newline that terminated the first line
    if first.endswith("\r\n"):
        parts.append("\r\n")
    elif first.endswith("\n"):
        parts.append("\n")
    for i in range(start_line, end_line - 1):
        parts.append(lines[i])
    last = lines[end_line - 1]
    last_no_eol = last.rstrip("\n").rstrip("\r")
    if end_col - 1 > len(last_no_eol):
        return None
    parts.append(last_no_eol[:end_col - 1])
    return "".join(parts)


# ---------------------------------------------------------------------------
# R2 — Missing-symbol auto-import
# ---------------------------------------------------------------------------

# Per-language matcher table keyed on (parser-hint, error pattern).
# Each entry yields the missing symbol name when the diagnostic matches.
# Hints are matched against (file extension, error_code, message) — see
# _detect_language_for_import for the dispatch.

_PY_IMPORT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"NameError:\s*name\s*'(?P<sym>\w+)'\s*is\s*not\s*defined"),
    re.compile(r"name\s*'(?P<sym>\w+)'\s*is\s*not\s*defined"),
]
_TS_IMPORT_PATTERN = re.compile(r"Cannot find name\s*'(?P<sym>\w+)'")
_RUST_IMPORT_PATTERN = re.compile(r"cannot find (?:value|function|type|trait)\s*`(?P<sym>\w+)`")
_GO_IMPORT_PATTERN = re.compile(r"undefined:\s*(?P<sym>\w+)")
_JAVA_IMPORT_PATTERN = re.compile(
    r"cannot find symbol[\s\S]*?symbol:\s*(?:class|method|variable)\s+(?P<sym>\w+)"
)


def _try_missing_import(
    diag: dict[str, Any],
    workspace_path: str,
) -> Optional[PatchBlock]:
    """Emit an import statement for an undefined symbol with exactly one
    workspace definition outside the offending file.

    Ambiguity (zero or multiple matches) is the failure mode — those go
    to the LLM. The "exactly one" rule is strict on purpose.
    """
    file = str(diag.get("file", "") or "")
    if not file:
        return None
    ext = os.path.splitext(file)[1].lower()
    message = str(diag.get("message", "") or "")
    error_code = str(diag.get("error_code", "") or "").upper()

    language: Optional[str]
    symbol: Optional[str]
    language, symbol = _detect_missing_symbol(ext, error_code, message)
    if language is None or symbol is None:
        return None

    rel_file = _relative_to_workspace(file, workspace_path)
    if rel_file is None:
        return None
    file_abs = os.path.join(workspace_path, rel_file)

    candidates = _find_definitions(workspace_path, language, symbol, file_abs)
    if len(candidates) != 1:
        return None

    def_path = candidates[0]
    import_stmt = _build_import_statement(language, symbol, def_path, file_abs, workspace_path)
    if not import_stmt:
        return None

    # Defend against re-running on a file that already has the import.
    try:
        with open(file_abs, "r", encoding="utf-8") as fh:
            head = fh.read()
    except OSError:
        return None
    if import_stmt.strip() in head:
        return None

    # Emit CREATE_FILE for "insert at start" semantics? No — we want to
    # INSERT_AT_BLOCK but there is no named anchor at the file top. The
    # text-patcher INSERT_AT_BLOCK uses a substring anchor, so we anchor
    # on the first non-empty line and place BEFORE it. We fall back to a
    # REPLACE_BLOCK against the first line when present so the patcher
    # works in both text and AST modes.
    first_line = _first_meaningful_line(head)
    if not first_line:
        # Empty / comment-only file — just write the import as a brand new file.
        return PatchBlock(
            operation=OperationType.CREATE_FILE,
            file=rel_file,
            content=import_stmt + "\n" + head,
        )

    return PatchBlock(
        operation=OperationType.INSERT_AT_BLOCK,
        file=rel_file,
        anchor=first_line,
        placement=Placement.BEFORE,
        content=import_stmt,
    )


def _detect_missing_symbol(
    ext: str,
    error_code: str,
    message: str,
) -> tuple[Optional[str], Optional[str]]:
    """Map (ext, error_code, message) → (language, missing_symbol)."""
    if ext in (".py", ".pyi"):
        for pat in _PY_IMPORT_PATTERNS:
            m = pat.search(message)
            if m:
                return "python", m.group("sym")
        # Python parser may have stashed the exception type in error_code
        if "NAMEERROR" in error_code:
            m = re.search(r"name\s*'(\w+)'", message)
            if m:
                return "python", m.group(1)
    if ext in (".ts", ".tsx", ".d.ts"):
        if "TS2304" in error_code or "Cannot find name" in message:
            m = _TS_IMPORT_PATTERN.search(message)
            if m:
                return "typescript", m.group("sym")
    if ext == ".rs":
        if "E0425" in error_code or "E0412" in error_code or "E0422" in error_code or "cannot find" in message:
            m = _RUST_IMPORT_PATTERN.search(message)
            if m:
                return "rust", m.group("sym")
    if ext == ".go":
        m = _GO_IMPORT_PATTERN.search(message)
        if m:
            return "go", m.group("sym")
    if ext == ".java":
        m = _JAVA_IMPORT_PATTERN.search(message)
        if m:
            return "java", m.group("sym")
    return None, None


# Definition-line patterns per language. These look for top-level symbol
# declarations — not method/parameter shadowing — by leading-of-line plus
# a small whitelist of keyword prefixes.
_DEFINITION_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "python": [
        re.compile(r"^(?:def|class|async\s+def)\s+{name}\b"),
    ],
    "typescript": [
        re.compile(r"^export\s+(?:async\s+)?(?:function|class|const|let|interface|type|enum)\s+{name}\b"),
        re.compile(r"^(?:async\s+)?function\s+{name}\b"),
        re.compile(r"^class\s+{name}\b"),
    ],
    "rust": [
        re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+{name}\b"),
        re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|type|const|static)\s+{name}\b"),
    ],
    "go": [
        re.compile(r"^func\s+{name}\b"),
        re.compile(r"^type\s+{name}\b"),
        re.compile(r"^var\s+{name}\b"),
        re.compile(r"^const\s+{name}\b"),
    ],
    "java": [
        re.compile(r"^(?:public|private|protected)?\s*(?:abstract\s+|final\s+|static\s+)*(?:class|interface|enum)\s+{name}\b"),
    ],
}

_EXTENSION_FOR_LANG: dict[str, tuple[str, ...]] = {
    "python": (".py",),
    "typescript": (".ts", ".tsx"),
    "rust": (".rs",),
    "go": (".go",),
    "java": (".java",),
}

_IGNORE_DIRS_FOR_GREP: frozenset[str] = frozenset({
    ".git", "__pycache__", "node_modules", "vendor", "target",
    "build", "dist", ".tox", ".venv", "venv", ".mypy_cache",
    ".pytest_cache", ".ruff_cache",
})


def _find_definitions(
    workspace_path: str,
    language: str,
    symbol: str,
    skip_file: str,
) -> list[str]:
    """Grep the workspace for top-level definitions of ``symbol``.

    Returns absolute paths to any file that has exactly one matching
    definition pattern at the start of a line (after optional whitespace).
    Caps at 2 matches — anything ambiguous goes back to the LLM.
    """
    exts = _EXTENSION_FOR_LANG.get(language)
    pats_template = _DEFINITION_PATTERNS.get(language)
    if not exts or not pats_template:
        return []
    # Materialise the templates against the actual symbol
    pats = [re.compile(p.pattern.replace("{name}", re.escape(symbol))) for p in pats_template]

    skip_real = os.path.realpath(skip_file)
    matches: list[str] = []
    try:
        for root, dirs, files in os.walk(workspace_path):
            dirs[:] = [
                d for d in dirs
                if d not in _IGNORE_DIRS_FOR_GREP and not d.startswith(".")
            ]
            for fname in files:
                if not fname.endswith(exts):
                    continue
                full = os.path.join(root, fname)
                if os.path.realpath(full) == skip_real:
                    continue
                try:
                    with open(full, "r", encoding="utf-8", errors="replace") as fh:
                        text = fh.read()
                except OSError:
                    continue
                for line in text.splitlines():
                    stripped = line.lstrip()
                    if any(p.search(stripped) for p in pats):
                        matches.append(full)
                        break
                if len(matches) >= 2:
                    return matches
    except OSError:
        pass
    return matches


def _build_import_statement(
    language: str,
    symbol: str,
    def_path: str,
    offending_file: str,
    workspace_path: str,
) -> str:
    """Construct a language-canonical import statement."""
    if language == "python":
        module_path = _python_module_path(def_path, workspace_path)
        if not module_path:
            return ""
        return f"from {module_path} import {symbol}"
    if language == "typescript":
        rel = _ts_relative_import(def_path, offending_file)
        if not rel:
            return ""
        return f"import {{ {symbol} }} from '{rel}';"
    if language == "rust":
        # Use a crate-relative path. We can't reliably compute it without
        # parsing Cargo.toml, so fall back to ``crate::<filename>::<symbol>``
        # which the compiler will at least flag clearly if wrong.
        module = os.path.splitext(os.path.basename(def_path))[0]
        return f"use crate::{module}::{symbol};"
    if language == "go":
        # Go imports are at package level — the simplest correct form is
        # an import path derived from the workspace location.
        pkg_dir = os.path.dirname(os.path.relpath(def_path, workspace_path))
        if not pkg_dir or pkg_dir == ".":
            return ""
        return f'import "{pkg_dir}"'
    if language == "java":
        # Best-effort: read a `package X.Y.Z;` declaration from the def file.
        package = _java_package(def_path)
        if not package:
            return ""
        return f"import {package}.{symbol};"
    return ""


def _python_module_path(def_path: str, workspace_path: str) -> str:
    """Convert ``workspace/foo/bar/baz.py`` → ``foo.bar.baz``."""
    rel = os.path.relpath(def_path, workspace_path)
    if rel.endswith(".py"):
        rel = rel[:-3]
    elif rel.endswith(".pyi"):
        rel = rel[:-4]
    rel = rel.replace(os.sep, ".")
    if rel.endswith(".__init__"):
        rel = rel[: -len(".__init__")]
    return rel


def _ts_relative_import(def_path: str, offending_file: str) -> str:
    """Compute a relative import path for TypeScript."""
    dest = os.path.splitext(def_path)[0]
    src_dir = os.path.dirname(offending_file)
    rel = os.path.relpath(dest, src_dir)
    rel = rel.replace(os.sep, "/")
    if not rel.startswith("."):
        rel = "./" + rel
    return rel


def _java_package(def_path: str) -> str:
    """Read the top-of-file ``package X;`` line from a Java source file."""
    try:
        with open(def_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("//"):
                    continue
                m = re.match(r"^package\s+([\w.]+)\s*;", stripped)
                if m:
                    return m.group(1)
                # First non-comment, non-blank line that wasn't a package
                # declaration → no package
                return ""
    except OSError:
        return ""
    return ""


def _first_meaningful_line(source: str) -> str:
    """First non-blank, non-comment line of ``source`` (for INSERT_AT_BLOCK anchor)."""
    for raw in source.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("//"):
            continue
        return raw
    return ""


# ---------------------------------------------------------------------------
# R4 — Missing pip-installable dep autofix
# ---------------------------------------------------------------------------

# Symbols whose package name on PyPI differs from the import name. The
# compiler_node sets `missing_symbol` to whatever pytest / python printed —
# usually the import name — but we have to write the *install* name into
# the manifest. Only list symbols whose distribution name truly differs;
# the common case (`pytest` → `pytest`, `pydantic` → `pydantic`) needs no entry.
_DEP_INSTALL_NAMES: dict[str, str] = {
    "yaml": "PyYAML",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "bs4": "beautifulsoup4",
}


def _try_missing_dep(
    diag: dict[str, Any],
    workspace_path: str,
) -> Optional[PatchBlock]:
    """Append a missing pip-installable dep to the workspace's requirements
    manifest. Triggered by compiler_node's MISSING_DEP diagnostic.

    The compiler_node already validated that the missing symbol is in
    ``_PIP_INSTALLABLE_SYMBOLS`` (pytest, ruff, mypy, etc.), so we don't
    need to second-guess the package name. We only handle the
    ``requirements.txt`` path here — pyproject.toml `[project.optional-
    dependencies].dev` needs structural TOML edits that the LLM can do
    more accurately than a regex. If the manifest doesn't exist yet, we
    CREATE_FILE it with the single line.

    Idempotent: if the symbol is already present anywhere in the manifest
    (with or without a version pin), we return None so the LLM gets the
    diagnostic and can investigate further.
    """
    if str(diag.get("error_code", "")) != "MISSING_DEP":
        return None
    symbol = str(diag.get("missing_symbol", "") or "").strip()
    if not symbol:
        return None
    install_name = _DEP_INSTALL_NAMES.get(symbol, symbol)
    build_cmd = str(diag.get("build_command", "") or "").lower()

    # Defer to the LLM only when the build command points at pyproject
    # (`pip install -e .`) — autofixing TOML structure is too error-prone.
    # Everything else (bare `pip install pytest && pytest -q`,
    # `pip install -r requirements.txt && pytest`, any pip-prefixed flow)
    # gets a requirements.txt edit; the compiler_node's adapter will
    # upgrade the build command to consume it on the next compile cycle.
    if "pip install -e" in build_cmd or "pyproject" in build_cmd:
        return None

    manifest_rel = "requirements.txt"
    manifest_abs = os.path.join(workspace_path, manifest_rel)

    # If the manifest doesn't exist yet, create it with just this dep.
    if not os.path.isfile(manifest_abs):
        return PatchBlock(
            operation=OperationType.CREATE_FILE,
            file=manifest_rel,
            content=f"{install_name}\n",
        )

    # Read existing content. Treat any line that starts with the install
    # name (case-insensitive, ignoring extras like `package[extra]>=1.0`)
    # as "already present" → idempotent no-op.
    try:
        with open(manifest_abs, "r", encoding="utf-8", errors="replace") as f:
            existing = f.read()
    except OSError:
        return None

    pin_pattern = re.compile(
        rf"^\s*{re.escape(install_name)}(?:\[|[<>=!~]|\s|$)",
        re.MULTILINE | re.IGNORECASE,
    )
    if pin_pattern.search(existing):
        return None

    # Append the dep on a new line. Use INSERT_AT_BLOCK with no anchor
    # would require the AST patcher; instead emit a REPLACE_BLOCK that
    # rewrites the last existing line into "<last_line>\n<dep>", which the
    # TextPatcher applies as a straight substring swap.
    stripped = existing.rstrip("\n")
    if not stripped:
        # File exists but is empty / whitespace — CREATE_FILE will fail
        # ("already exists"), so emit a REPLACE_BLOCK that rewrites the
        # whitespace into the single dep line.
        return PatchBlock(
            operation=OperationType.REPLACE_BLOCK,
            file=manifest_rel,
            search=existing,
            replace=f"{install_name}\n",
        )
    last_line = stripped.split("\n")[-1]
    return PatchBlock(
        operation=OperationType.REPLACE_BLOCK,
        file=manifest_rel,
        search=last_line,
        replace=f"{last_line}\n{install_name}",
    )


# ---------------------------------------------------------------------------
# R5 — pip ResolutionImpossible autofix
# ---------------------------------------------------------------------------

# Matches a requirement-spec line and captures the package name + optional
# extras (e.g. "uvicorn[standard]"). Anything trailing — the version
# specifier — is dropped. Line shapes covered:
#   fastapi
#   fastapi>=0.100.0
#   fastapi >= 0.100.0, < 0.120
#   uvicorn[standard]>=0.23.0
#   pydantic~=2.0
#   pkg-name==1.2.3 ; python_version >= "3.10"
# We deliberately do NOT touch:
#   - URLs / VCS refs (anything starting with git+, http://, etc.)
#   - editable installs (-e .)
#   - local paths (./, /path/to)
#   - constraint / index flags (-c, -r, --index-url)
#   - comments and blanks
_REQ_LINE_RE = re.compile(
    r"^(?P<pkg>[A-Za-z0-9][A-Za-z0-9_.\-]*(?:\[[A-Za-z0-9_,\-]+\])?)"
    r"\s*(?:[<>=!~][^;]*)?(?P<marker>\s*;[^\n]*)?$"
)


def _try_dep_resolution_conflict(
    diag: dict[str, Any],
    workspace_path: str,
) -> Optional[PatchBlock]:
    """Drop every version specifier from ``requirements.txt`` so pip's
    resolver can pick a self-consistent set on its own.

    The repair LLM is otherwise forced to guess which pin to relax —
    pip's "ResolutionImpossible" message rarely names both sides of the
    conflict, so guesses make conflicts worse and burn the repair budget.

    Stripping pins is the right move on greenfield runs where every pin
    is itself an LLM guess. Lines that aren't simple version specs
    (editable installs, VCS refs, URLs, comments, blanks, pip flags) are
    preserved verbatim.

    Returns None when:
      - the diagnostic isn't a DEP_RESOLUTION_CONFLICT,
      - requirements.txt doesn't exist,
      - the file already has no pins anywhere (so stripping would be a no-op).
    """
    if str(diag.get("error_code", "")) != "DEP_RESOLUTION_CONFLICT":
        return None

    manifest_rel = "requirements.txt"
    manifest_abs = os.path.join(workspace_path, manifest_rel)
    if not os.path.isfile(manifest_abs):
        return None

    try:
        with open(manifest_abs, "r", encoding="utf-8", errors="replace") as f:
            existing = f.read()
    except OSError:
        return None

    if not existing.strip():
        return None

    stripped_lines: list[str] = []
    changed = False
    for raw_line in existing.splitlines():
        line = raw_line.rstrip("\r")
        stripped = line.strip()
        # Preserve blanks, comments, and pip flags verbatim.
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            stripped_lines.append(line)
            continue
        # Preserve URLs / VCS refs / local paths.
        if any(stripped.startswith(prefix) for prefix in (
            "git+", "hg+", "svn+", "bzr+",
            "http://", "https://", "file://",
            "./", "../", "/",
        )):
            stripped_lines.append(line)
            continue
        m = _REQ_LINE_RE.match(stripped)
        if m is None:
            # Unrecognised shape — leave it untouched.
            stripped_lines.append(line)
            continue
        pkg = m.group("pkg")
        marker = (m.group("marker") or "").strip()
        new_line = pkg if not marker else f"{pkg} {marker}"
        if new_line != stripped:
            changed = True
        stripped_lines.append(new_line)

    if not changed:
        return None

    # Preserve trailing newline if the original had one.
    new_content = "\n".join(stripped_lines)
    if existing.endswith("\n"):
        new_content += "\n"

    return PatchBlock(
        operation=OperationType.REPLACE_BLOCK,
        file=manifest_rel,
        search=existing,
        replace=new_content,
    )


# ---------------------------------------------------------------------------
# R3 — Security finding autofix
# ---------------------------------------------------------------------------

def _try_security_autofix(
    diag: dict[str, Any],
    workspace_path: str,
) -> Optional[PatchBlock]:
    """Dispatch security-finding diagnostics through the rule registry.

    The diagnostic ``error_code`` is shaped ``SCANNER:RULE_ID`` (built by
    harness/security.py::_findings_to_diagnostics). Only diagnostics with
    such a prefix are eligible.
    """
    error_code = str(diag.get("error_code", "") or "")
    if ":" not in error_code:
        return None
    scanner, _, rule_id = error_code.partition(":")
    scanner = scanner.upper()
    rule_id = rule_id.upper()

    fix_fn = _SECURITY_FIX_TABLE.get(scanner)
    if fix_fn is None:
        return None
    return fix_fn(rule_id, diag, workspace_path)


def _fix_bandit(
    rule_id: str,
    diag: dict[str, Any],
    workspace_path: str,
) -> Optional[PatchBlock]:
    """Bandit-specific autofix dispatch."""
    file = str(diag.get("file", "") or "")
    if not file:
        return None
    rel_file = _relative_to_workspace(file, workspace_path)
    if rel_file is None:
        return None
    file_abs = os.path.join(workspace_path, rel_file)
    if not os.path.isfile(file_abs):
        return None

    try:
        with open(file_abs, "r", encoding="utf-8") as fh:
            source = fh.read()
    except OSError:
        return None

    line_no = int(diag.get("line", 0) or 0)
    if line_no <= 0:
        return None
    lines = source.splitlines(keepends=True)
    if line_no > len(lines):
        return None
    target = lines[line_no - 1]
    naked = target.rstrip("\n").rstrip("\r")

    if rule_id == "B201":
        # Flask app.run(debug=True) — flip to False
        if "debug=True" not in naked:
            return None
        new = naked.replace("debug=True", "debug=False")
        return PatchBlock(
            operation=OperationType.REPLACE_BLOCK,
            file=rel_file,
            search=naked,
            replace=new,
        )

    if rule_id == "B602":
        # subprocess(..., shell=True) — only safe to flip when the first
        # positional arg is already a list literal. String args need
        # shell escaping that we cannot synthesise.
        if "shell=True" not in naked:
            return None
        # Heuristic: look for "[ ... ]" before the shell=True keyword.
        # We don't try to fully parse — false negatives are fine, false
        # positives (string args getting flipped) are NOT.
        if not re.search(r"\[[^\]]*\]\s*,", naked):
            return None
        new = naked.replace("shell=True", "shell=False")
        return PatchBlock(
            operation=OperationType.REPLACE_BLOCK,
            file=rel_file,
            search=naked,
            replace=new,
        )

    return None


def _fix_gitleaks(
    rule_id: str,
    diag: dict[str, Any],
    workspace_path: str,
) -> Optional[PatchBlock]:
    """Gitleaks autofix: delete the offending line.

    NOTE: the plan also calls for adding ``<RULE_ID>=<placeholder>`` to
    .env.example. The patch-application API only supports one PatchBlock
    per dispatcher return, so we delete the line here; a follow-up
    enhancement could chain a second patch via apply_autofixes itself.
    Deleting is the security-critical move; the .env.example placeholder
    is a developer-experience nicety.
    """
    file = str(diag.get("file", "") or "")
    if not file:
        return None
    rel_file = _relative_to_workspace(file, workspace_path)
    if rel_file is None:
        return None
    file_abs = os.path.join(workspace_path, rel_file)
    if not os.path.isfile(file_abs):
        return None
    try:
        with open(file_abs, "r", encoding="utf-8") as fh:
            source = fh.read()
    except OSError:
        return None

    line_no = int(diag.get("line", 0) or 0)
    if line_no <= 0:
        return None
    lines = source.splitlines(keepends=True)
    if line_no > len(lines):
        return None

    target = lines[line_no - 1]
    naked = target.rstrip("\n").rstrip("\r")
    # The TextPatcher.delete_block requires the search string to appear
    # exactly once. If the same secret line appears twice we abort —
    # ambiguity is the LLM's job.
    if source.count(naked) != 1 or not naked.strip():
        return None
    return PatchBlock(
        operation=OperationType.DELETE_BLOCK,
        file=rel_file,
        search=naked,
    )


_TRIVY_FIXED_VERSION_RE = re.compile(r"upgrade to\s+([0-9][\w.\-+]*)")
_PKG_FROM_TRIVY_MESSAGE_RE = re.compile(r"^(?P<pkg>[\w@./-]+)\s+(?P<ver>[\w.\-+]+):")


def _fix_trivy(
    rule_id: str,
    diag: dict[str, Any],
    workspace_path: str,
) -> Optional[PatchBlock]:
    """Trivy autofix: bump a version pin when a FixedVersion is known.

    The trivy parser packs ``"upgrade to X"`` into the message string
    when ``FixedVersion`` is non-empty. We extract that, parse the
    package + installed version out of the message prefix, locate the
    pin in the target manifest, and emit a REPLACE_BLOCK.
    """
    file = str(diag.get("file", "") or "")
    if not file:
        return None
    rel_file = _relative_to_workspace(file, workspace_path)
    if rel_file is None:
        return None
    file_abs = os.path.join(workspace_path, rel_file)
    if not os.path.isfile(file_abs):
        return None

    message = str(diag.get("message", "") or "")
    fix_match = _TRIVY_FIXED_VERSION_RE.search(message)
    if not fix_match:
        return None
    fixed_version = fix_match.group(1)

    pkg_match = _PKG_FROM_TRIVY_MESSAGE_RE.search(message)
    if not pkg_match:
        return None
    package = pkg_match.group("pkg")
    installed = pkg_match.group("ver")

    try:
        with open(file_abs, "r", encoding="utf-8") as fh:
            source = fh.read()
    except OSError:
        return None

    candidate = _find_version_pin(file_abs, source, package, installed, fixed_version)
    if candidate is None:
        return None
    search, replace = candidate
    if source.count(search) != 1:
        return None
    return PatchBlock(
        operation=OperationType.REPLACE_BLOCK,
        file=rel_file,
        search=search,
        replace=replace,
    )


def _find_version_pin(
    file_abs: str,
    source: str,
    package: str,
    installed: str,
    fixed: str,
) -> Optional[tuple[str, str]]:
    """Return (search, replace) for bumping ``package`` in this manifest."""
    name = os.path.basename(file_abs).lower()
    if name == "package.json":
        # JSON: "package": "installed" → "package": "fixed"
        try:
            data = json.loads(source)
        except (json.JSONDecodeError, ValueError):
            return None
        # Look for the package in any of the dep buckets to confirm it's present.
        if not _package_in_npm_manifest(data, package, installed):
            return None
        # Build search using the literal JSON spelling. We use the
        # human-friendly form (with quotes) so the patcher's exact-match
        # against the file content lands. Handle both "1.0.0" and "^1.0.0".
        # Try the bare version first, then with common prefix sigils.
        for prefix in ("", "^", "~", ">=", ">"):
            search = f'"{package}": "{prefix}{installed}"'
            if search in source:
                replace = f'"{package}": "{prefix}{fixed}"'
                return search, replace
        return None
    if name == "requirements.txt":
        # pip: package==installed → package==fixed
        for sep in ("==", "~=", ">="):
            search = f"{package}{sep}{installed}"
            if search in source:
                return search, f"{package}{sep}{fixed}"
        return None
    if name == "go.mod":
        # go.mod: require <pkg> v<installed> → v<fixed>
        search_a = f"{package} v{installed}"
        if search_a in source:
            return search_a, f"{package} v{fixed}"
        search_b = f"{package} {installed}"
        if search_b in source:
            return search_b, f"{package} {fixed}"
        return None
    if name == "cargo.toml":
        # Cargo: pkg = "installed" or pkg = "^installed"
        for prefix in ("", "^", "~"):
            search = f'{package} = "{prefix}{installed}"'
            if search in source:
                return search, f'{package} = "{prefix}{fixed}"'
        return None
    return None


def _package_in_npm_manifest(
    data: dict[str, Any],
    package: str,
    installed: str,
) -> bool:
    for bucket in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
        deps = data.get(bucket)
        if not isinstance(deps, dict):
            continue
        if package not in deps:
            continue
        # The pin may be "1.0.0" or "^1.0.0" or "~1.0.0" — accept any
        # that contains the installed substring.
        pin = str(deps[package])
        if installed in pin:
            return True
    return False


_SECURITY_FIX_TABLE: dict[str, Any] = {
    "BANDIT": _fix_bandit,
    "GITLEAKS": _fix_gitleaks,
    "GITLEAKS-FALLBACK": _fix_gitleaks,
    "TRIVY": _fix_trivy,
}


# ---------------------------------------------------------------------------
# R6 — Web asset reference fix dispatcher
# ---------------------------------------------------------------------------

WEB_ASSET_ERROR_CODE = "WEB_ASSET_REF"


def _try_asset_reference_fix(
    diag: dict[str, Any],
    workspace_path: str,
) -> Optional[PatchBlock]:
    """Rewrite a broken local asset reference to its suggested path.

    Triggered by lintgate's web-asset scanner via error_code WEB_ASSET_REF.
    Two-step safety:
      1. A suggested_path must exist (the scanner only attaches one when
         the basename match is unique in the workspace).
      2. The raw_reference must appear exactly once in the referring file,
         so REPLACE_BLOCK has an unambiguous anchor.

    Returns None when either condition fails — the LLM repair loop then
    gets the diagnostic with full file context and can choose between
    fixing the path or creating the missing asset.
    """
    if str(diag.get("error_code", "")) != WEB_ASSET_ERROR_CODE:
        return None

    raw_ref = diag.get("raw_reference") or ""
    suggested = diag.get("suggested_path") or ""
    if not raw_ref or not suggested:
        return None
    if raw_ref == suggested:
        # Scanner shouldn't suggest itself, but guard against pathological
        # inputs that would produce a no-op patch.
        return None

    file = str(diag.get("file") or "")
    if not file:
        return None
    rel_file = _relative_to_workspace(file, workspace_path)
    if rel_file is None:
        return None
    file_path = os.path.join(workspace_path, rel_file)
    if not os.path.isfile(file_path):
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            source = fh.read()
    except OSError:
        return None

    # Demand a unique occurrence of raw_ref so the rewrite is unambiguous.
    # If the string appears multiple times (e.g. one HTML referencing the
    # same broken path in two <link> tags) we punt to the LLM rather than
    # risk rewriting the wrong one.
    if source.count(raw_ref) != 1:
        return None

    return PatchBlock(
        operation=OperationType.REPLACE_BLOCK,
        file=rel_file,
        search=raw_ref,
        replace=suggested,
    )


def web_asset_diagnostics_to_standard(
    web_asset_errors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert lintgate's web_asset_errors dicts to the standard diagnostic
    shape consumed by apply_autofixes.

    Called from the bridge in repair_node so R6 sees these alongside compiler
    diagnostics in the same dispatcher loop.
    """
    out: list[dict[str, Any]] = []
    for err in web_asset_errors or []:
        out.append({
            "file": err.get("referring_file", ""),
            "line": err.get("line", 0),
            "column": err.get("column", 0),
            "error_code": WEB_ASSET_ERROR_CODE,
            "message": (
                f"unresolved asset reference '{err.get('raw_reference', '')}'"
                + (f" (did you mean '{err['suggested_path']}'?)"
                   if err.get("suggested_path") else "")
            ),
            "severity": "error",
            "raw_reference": err.get("raw_reference", ""),
            "suggested_path": err.get("suggested_path"),
        })
    return out
