"""Static asset reference scanner for web apps.

Walks HTML/CSS/JS files and reports local references that don't resolve to
files on disk. Catches the class of bug where `index.html` says
`<link href="src/styles.css">` but no such file was ever written.

Two consumers:
  * In-process: `scan_web_asset_references()` called from `lintgate_node`.
    Diagnostics flow into `lint_errors` and trigger autofix R6.
  * CLI: `python -m harness.web_asset_scan <workspace>` invoked from the
    Makefile `build:` target so the sandbox loop catches the same bugs.

Scope:
  * HTML: <link href>, <script src>, <img src>, <source src>,
          <video src>, <audio src>, <a href>
  * CSS:  url(...) and @import "..."
  * JS:   relative-path import statements (skips bare module specifiers
          like `react`)

Skips: http(s)://, protocol-relative //, mailto:, tel:, data:, javascript:,
       and #anchor-only references.
"""

from __future__ import annotations

import html.parser
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

_HTML_EXTS = frozenset({".html", ".htm"})
_CSS_EXTS = frozenset({".css"})
_JS_EXTS = frozenset({".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"})

_SKIP_PROTOCOL_PREFIXES = ("http://", "https://", "//", "mailto:", "tel:",
                           "data:", "javascript:", "ftp://", "ws://", "wss://")

_HTML_REF_ATTRS = {
    "link": "href",
    "script": "src",
    "img": "src",
    "source": "src",
    "video": "src",
    "audio": "src",
    "a": "href",
    "iframe": "src",
    "embed": "src",
    "object": "data",
}


@dataclass
class AssetRefDiagnostic:
    """One unresolved local reference."""
    referring_file: str        # workspace-relative path
    line: int
    column: int
    raw_reference: str         # exact string from the source
    resolved_path: str         # what we tried to resolve to (workspace-relative)
    suggested_path: Optional[str] = None   # unique basename match elsewhere

    def format_compiler_style(self) -> str:
        """Format as `file:line:col: error: message` for sandbox regex match."""
        msg = f"unresolved asset reference '{self.raw_reference}'"
        if self.suggested_path:
            msg += f" (did you mean '{self.suggested_path}'?)"
        return f"{self.referring_file}:{self.line}:{self.column}: error: {msg}"


def _is_skippable_ref(ref: str) -> bool:
    if not ref:
        return True
    ref = ref.strip()
    if not ref:
        return True
    if ref.startswith("#"):
        return True
    lower = ref.lower()
    return any(lower.startswith(p) for p in _SKIP_PROTOCOL_PREFIXES)


def _strip_url_fragment(ref: str) -> str:
    """Drop ?query and #fragment from a URL path."""
    for sep in ("?", "#"):
        idx = ref.find(sep)
        if idx != -1:
            ref = ref[:idx]
    return ref


class _HtmlRefExtractor(html.parser.HTMLParser):
    """Collect (tag, attr, ref, line, col) for each known asset-bearing tag."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.refs: list[tuple[str, str, int, int]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attr_name = _HTML_REF_ATTRS.get(tag.lower())
        if not attr_name:
            return
        # Special case: a <link> without rel="stylesheet"/"icon"/"preload" etc
        # may be metadata-only (e.g. canonical URL). We still check it — if the
        # ref is local it should resolve; if it's an external URL it gets
        # skipped by _is_skippable_ref.
        for name, value in attrs:
            if name.lower() == attr_name and value:
                line, col = self.getpos()
                self.refs.append((tag.lower(), value, line, col))
                return

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        # <link rel="stylesheet" href="..." /> is parsed as startendtag.
        self.handle_starttag(tag, attrs)


def _scan_html(text: str) -> list[tuple[str, int, int]]:
    """Return (ref, line, col) for each asset reference in an HTML file."""
    parser = _HtmlRefExtractor()
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        # Malformed HTML: return what we got. The formatters/linters will
        # flag the parse error separately.
        pass
    return [(ref, line, col) for _tag, ref, line, col in parser.refs]


_CSS_URL_RE = re.compile(
    r"""url\(\s*(?P<q>['"]?)(?P<ref>[^'")]+)(?P=q)\s*\)""",
    re.IGNORECASE,
)
_CSS_IMPORT_RE = re.compile(
    r"""@import\s+(?:url\(\s*)?['"](?P<ref>[^'"]+)['"]""",
    re.IGNORECASE,
)


def _scan_css(text: str) -> list[tuple[str, int, int]]:
    refs: list[tuple[str, int, int]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for m in _CSS_URL_RE.finditer(line):
            refs.append((m.group("ref"), line_no, m.start("ref") + 1))
        for m in _CSS_IMPORT_RE.finditer(line):
            refs.append((m.group("ref"), line_no, m.start("ref") + 1))
    return refs


# Match `import ... from "path"` and `import("path")` and side-effect `import "path"`.
# We're intentionally loose: false positives (e.g. a string literal that
# happens to look like an import) get filtered out by the "is it a relative
# path?" check below.
_JS_IMPORT_RE = re.compile(
    r"""(?:^|[\s;])import\s*(?:[^'";]*?\sfrom\s*)?['"](?P<ref>[^'"]+)['"]""",
)
_JS_DYNAMIC_IMPORT_RE = re.compile(
    r"""\bimport\s*\(\s*['"](?P<ref>[^'"]+)['"]\s*\)""",
)


def _is_relative_js_specifier(ref: str) -> bool:
    """Relative JS imports start with `./`, `../`, or `/`. Everything else is
    a bare module specifier resolved by the JS package manager."""
    return ref.startswith(("./", "../", "/"))


def _scan_js(text: str) -> list[tuple[str, int, int]]:
    refs: list[tuple[str, int, int]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for m in _JS_IMPORT_RE.finditer(line):
            ref = m.group("ref")
            if _is_relative_js_specifier(ref):
                refs.append((ref, line_no, m.start("ref") + 1))
        for m in _JS_DYNAMIC_IMPORT_RE.finditer(line):
            ref = m.group("ref")
            if _is_relative_js_specifier(ref):
                refs.append((ref, line_no, m.start("ref") + 1))
    return refs


def _build_basename_index(workspace_path: str) -> dict[str, list[str]]:
    """Walk the workspace, return {basename: [relpath, ...]}.

    Used to suggest corrections: `src/styles.css` is unresolved but
    `style.css` exists at root → suggest `style.css`. We also try the
    misspelling cousin: same stem (`styles` ~ `style`).

    Skips common churn dirs to keep this fast on large repos.
    """
    skip_dirs = {"node_modules", ".git", "dist", "build", "coverage",
                 "__pycache__", ".venv", "venv", "target", ".next", ".nuxt"}
    index: dict[str, list[str]] = {}
    for root, dirs, files in os.walk(workspace_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in files:
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, workspace_path)
            index.setdefault(fname.lower(), []).append(rel)
    return index


def _suggest_path(
    raw_ref: str,
    basename_index: dict[str, list[str]],
) -> Optional[str]:
    """If the basename uniquely exists elsewhere, suggest it."""
    bare = os.path.basename(_strip_url_fragment(raw_ref)).lower()
    if not bare:
        return None
    matches = basename_index.get(bare)
    if matches and len(matches) == 1:
        return matches[0]
    # Try stem-only match for typos like `styles.css` vs `style.css`.
    stem, ext = os.path.splitext(bare)
    if not ext or not stem:
        return None
    # Look for files with same extension and a stem differing by at most 1 char.
    candidates: list[str] = []
    for fname, paths in basename_index.items():
        f_stem, f_ext = os.path.splitext(fname)
        if f_ext != ext or not f_stem:
            continue
        if abs(len(f_stem) - len(stem)) > 1:
            continue
        # Either prefix-match or single edit distance check (cheap form).
        if f_stem.startswith(stem) or stem.startswith(f_stem):
            candidates.extend(paths)
    if len(candidates) == 1:
        return candidates[0]
    return None


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except (OSError, UnicodeDecodeError):
        return None


def scan_web_asset_references(
    workspace_path: str,
    changed_files: Optional[list[str]] = None,
) -> list[AssetRefDiagnostic]:
    """Scan HTML/CSS/JS files for local references that don't resolve.

    Args:
        workspace_path: absolute workspace root.
        changed_files: optional subset of workspace-relative paths to scan.
            If None, scans every eligible file in the workspace.

    Returns:
        One diagnostic per unresolved reference.
    """
    workspace_path = os.path.abspath(workspace_path)
    if not os.path.isdir(workspace_path):
        return []

    files_to_scan: list[str] = []
    if changed_files is None:
        for root, dirs, names in os.walk(workspace_path):
            dirs[:] = [d for d in dirs if d not in {
                "node_modules", ".git", "dist", "build", "coverage",
                "__pycache__", ".venv", "venv", "target", ".next", ".nuxt"
            }]
            for name in names:
                ext = os.path.splitext(name)[1].lower()
                if ext in _HTML_EXTS or ext in _CSS_EXTS or ext in _JS_EXTS:
                    full = os.path.join(root, name)
                    files_to_scan.append(os.path.relpath(full, workspace_path))
    else:
        for rel in changed_files:
            ext = os.path.splitext(rel)[1].lower()
            if ext in _HTML_EXTS or ext in _CSS_EXTS or ext in _JS_EXTS:
                files_to_scan.append(rel)

    if not files_to_scan:
        return []

    basename_index = _build_basename_index(workspace_path)
    diagnostics: list[AssetRefDiagnostic] = []

    for rel_path in files_to_scan:
        abs_path = os.path.join(workspace_path, rel_path)
        text = _read_text(abs_path)
        if text is None:
            continue
        ext = os.path.splitext(rel_path)[1].lower()
        if ext in _HTML_EXTS:
            refs = _scan_html(text)
        elif ext in _CSS_EXTS:
            refs = _scan_css(text)
        else:
            refs = _scan_js(text)

        referring_dir = os.path.dirname(abs_path)
        for raw_ref, line, col in refs:
            if _is_skippable_ref(raw_ref):
                continue
            clean_ref = _strip_url_fragment(raw_ref).strip()
            if not clean_ref:
                continue
            # Resolve relative to the referring file's directory.
            target_abs = os.path.normpath(os.path.join(referring_dir, clean_ref))
            # Stay inside workspace.
            try:
                target_rel = os.path.relpath(target_abs, workspace_path)
            except ValueError:
                continue
            if target_rel.startswith(".."):
                continue
            if os.path.exists(target_abs):
                continue
            suggested = _suggest_path(clean_ref, basename_index)
            diagnostics.append(AssetRefDiagnostic(
                referring_file=rel_path,
                line=line,
                column=col,
                raw_reference=raw_ref,
                resolved_path=target_rel,
                suggested_path=suggested,
            ))

    return diagnostics


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry: `python -m harness.web_asset_scan <workspace>`.

    Exits 0 on no findings, 1 on any unresolved reference. Prints each
    diagnostic in `file:line:col: error: message` form so the harness
    sandbox's generic diagnostic extractor picks them up.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    workspace = argv[0] if argv else os.getcwd()
    diagnostics = scan_web_asset_references(workspace)
    if not diagnostics:
        return 0
    for d in diagnostics:
        print(d.format_compiler_style(), file=sys.stderr)
    print(f"\n{len(diagnostics)} unresolved asset reference(s).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
