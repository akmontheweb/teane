"""Structured file inventory: parse and cross-check.

The architecture step (`SPEC_ARCHITECTURE.md`) and the planning step both
embed a JSON file-inventory block. The cross-check function compares the
two and surfaces three classes of diagnostic:

  * MISSING_FROM_PLAN  — architecture listed a file the plan forgot
  * EXTRA_IN_PLAN      — plan listed a file the architecture didn't
                         (warning — plan may legitimately split a file)
  * PATH_MISMATCH      — same basename, different path (`style.css` vs
                         `src/styles.css`) — high confidence hint

A fourth diagnostic class fires after patching:

  * MISSING_FROM_DISK  — plan listed it, patcher didn't write it

The inventory block is a fenced JSON code block inside the markdown:

    ```json
    {
      "files": [
        {"path": "index.html", "purpose": "entry", "kind": "html"},
        {"path": "style.css",  "purpose": "styling", "kind": "css"}
      ]
    }
    ```

The block must contain a top-level "files" array of objects with at
least a "path" field. "purpose" and "kind" are optional but encouraged.
Multiple fenced blocks may exist in the document — the first one whose
parsed JSON has a "files" array wins.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FileEntry:
    """One file in a structured inventory."""
    path: str
    purpose: str = ""
    kind: str = ""

    @classmethod
    def from_dict(cls, raw: Any) -> Optional["FileEntry"]:
        if not isinstance(raw, dict):
            return None
        path = raw.get("path")
        if not isinstance(path, str) or not path.strip():
            return None
        # Normalize path: strip leading "./" and trailing slashes; keep the
        # workspace-relative form intact.
        norm = path.strip().lstrip("./").rstrip("/")
        return cls(
            path=norm,
            purpose=str(raw.get("purpose", "") or ""),
            kind=str(raw.get("kind", "") or ""),
        )


@dataclass
class InventoryParseResult:
    """Outcome of trying to parse a fenced JSON inventory block."""
    files: list[FileEntry] = field(default_factory=list)
    error: Optional[str] = None
    raw_block: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class InventoryDiagnostic:
    """A discrepancy between architecture and planning, or between plan
    and disk after patching."""
    kind: str           # MISSING_FROM_PLAN | EXTRA_IN_PLAN | PATH_MISMATCH | MISSING_FROM_DISK
    file: str           # workspace-relative path the diagnostic concerns
    message: str
    suggested_path: Optional[str] = None

    @property
    def is_error(self) -> bool:
        """EXTRA_IN_PLAN is advisory; the others are errors that should
        block progression."""
        return self.kind != "EXTRA_IN_PLAN"

    def format_compiler_style(self) -> str:
        msg = self.message
        if self.suggested_path:
            msg += f" (did you mean '{self.suggested_path}'?)"
        return f"{self.file}:1:1: error: {msg}"


# Match ```json ... ``` fenced blocks. Captures the inner JSON text.
_FENCED_JSON_RE = re.compile(
    r"```(?:json|JSON)\s*\n(.*?)\n```",
    re.DOTALL,
)

# A bare fenced block (no language tag) — fallback. Some LLMs forget the
# language hint. Only used after the tagged-block search fails.
_FENCED_ANY_RE = re.compile(r"```\s*\n(\{.*?\})\n```", re.DOTALL)


def parse_inventory(spec_md: str) -> InventoryParseResult:
    """Extract a fenced JSON inventory block from a markdown document.

    Returns InventoryParseResult with files populated on success or an
    error message on failure. Callers should fail loud on errors — a
    missing or malformed inventory is the bug we're trying to surface.
    """
    if not spec_md or not spec_md.strip():
        return InventoryParseResult(error="empty document")

    # Pass 1: prefer ```json-tagged blocks.
    candidates: list[tuple[str, dict[str, Any]]] = []
    for match in _FENCED_JSON_RE.finditer(spec_md):
        raw = match.group(1)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "files" in parsed:
            candidates.append((raw, parsed))

    # Pass 2: untagged fence as fallback.
    if not candidates:
        for match in _FENCED_ANY_RE.finditer(spec_md):
            raw = match.group(1)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "files" in parsed:
                candidates.append((raw, parsed))

    if not candidates:
        return InventoryParseResult(
            error="no fenced JSON block with a 'files' array found",
        )

    raw, parsed = candidates[0]
    files_raw = parsed.get("files")
    if not isinstance(files_raw, list):
        return InventoryParseResult(
            error="'files' is not an array",
            raw_block=raw,
        )

    files: list[FileEntry] = []
    for idx, item in enumerate(files_raw):
        entry = FileEntry.from_dict(item)
        if entry is None:
            return InventoryParseResult(
                error=f"files[{idx}] is missing a non-empty 'path' string",
                raw_block=raw,
            )
        files.append(entry)

    return InventoryParseResult(files=files, raw_block=raw)


# ---------------------------------------------------------------------------
# Workspace layout block — the second structured block in SPEC_ARCHITECTURE.md
#
# The architecture step emits a `workspace_layout` JSON block alongside the
# `files` block. The patcher's allowlist is derived from this layout's
# `roots[].path` list (plus the test directories and `root_files`), so the
# spec — not a filesystem heuristic — is authoritative about which top-level
# directories the LLM is allowed to write into.
#
# Schema (inside the same fenced ```json block style as the inventory):
#   {
#     "workspace_layout": {
#       "roots": [
#         {"path": "client", "purpose": "React frontend", "stack": "react"},
#         {"path": "server", "purpose": "Express backend", "stack": "express"}
#       ],
#       "test_placement": "co-located",        # or "centralized" / "mixed"
#       "root_files": ["package.json", "tsconfig.json", "vite.config.ts"]
#     }
#   }
#
# Backwards compat: when a spec written before this contract existed lacks
# the `workspace_layout` key, parse_layout falls back to deriving roots from
# the inventory's top-level path components — every distinct first-segment
# of a path in the `files` block IS a source root. The semantic metadata
# (purpose, stack, test_placement) is empty in that case, but the allowlist
# the patcher needs is still spec-driven.
# ---------------------------------------------------------------------------

_VALID_TEST_PLACEMENTS: frozenset[str] = frozenset({
    "co-located", "centralized", "mixed",
})


@dataclass
class LayoutRoot:
    """One source root in the workspace_layout block."""
    path: str          # workspace-relative directory name, no trailing slash
    purpose: str = ""  # short human description: "React frontend", "Express API"
    stack: str = ""    # tech stack hint: "react", "express", "fastapi", ...

    @classmethod
    def from_dict(cls, raw: Any) -> Optional["LayoutRoot"]:
        if not isinstance(raw, dict):
            return None
        path = raw.get("path")
        if not isinstance(path, str) or not path.strip():
            return None
        norm = path.strip()
        # Strip a literal "./" prefix without using lstrip("./") — the
        # latter strips dots and slashes character-by-character and
        # would silently turn ".git" into "git", letting a hidden-dir
        # entry slip past the dotfile guard below.
        if norm.startswith("./"):
            norm = norm[2:]
        norm = norm.rstrip("/")
        if not norm or "/" in norm or norm.startswith("."):
            # Roots must be a single top-level directory name, not a
            # nested path or hidden dir.
            return None
        return cls(
            path=norm,
            purpose=str(raw.get("purpose", "") or ""),
            stack=str(raw.get("stack", "") or ""),
        )


@dataclass
class LayoutParseResult:
    """Outcome of trying to parse a workspace_layout block."""
    roots: list[LayoutRoot] = field(default_factory=list)
    test_placement: str = ""
    root_files: list[str] = field(default_factory=list)
    error: Optional[str] = None
    raw_block: Optional[str] = None
    derived_from_inventory: bool = False  # True when block absent and we
                                          # synthesised roots from the
                                          # `files` inventory instead.

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def has_layout(self) -> bool:
        """True when at least one root path was determined (either parsed
        from a workspace_layout block or derived from the inventory)."""
        return bool(self.roots)


def parse_layout(spec_md: str) -> LayoutParseResult:
    """Extract the workspace_layout fenced JSON block from a spec doc.

    Two-pass extraction mirrors :func:`parse_inventory` — try
    ``json``-tagged fenced blocks first, then bare fences. The first block
    whose parsed JSON has a top-level ``workspace_layout`` object wins.

    When no such block is found but the document still has a parseable
    ``files`` inventory, ``roots`` is synthesised by collecting the
    distinct first-segment of each ``files[].path`` that lives in a
    subdirectory. This keeps spec-driven allowlists working for documents
    written before the layout contract existed — the operator gets the
    benefit without having to re-run the architecture phase.
    """
    if not spec_md or not spec_md.strip():
        return LayoutParseResult(error="empty document")

    # Pass 1: ```json-tagged blocks holding a `workspace_layout` key.
    candidates: list[tuple[str, dict[str, Any]]] = []
    for match in _FENCED_JSON_RE.finditer(spec_md):
        raw = match.group(1)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "workspace_layout" in parsed:
            candidates.append((raw, parsed))

    # Pass 2: bare-fence fallback.
    if not candidates:
        for match in _FENCED_ANY_RE.finditer(spec_md):
            raw = match.group(1)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "workspace_layout" in parsed:
                candidates.append((raw, parsed))

    if candidates:
        raw, parsed = candidates[0]
        layout = parsed.get("workspace_layout")
        if not isinstance(layout, dict):
            return LayoutParseResult(
                error="'workspace_layout' is not an object",
                raw_block=raw,
            )

        roots_raw = layout.get("roots")
        if roots_raw is not None and not isinstance(roots_raw, list):
            return LayoutParseResult(
                error="'workspace_layout.roots' is not an array",
                raw_block=raw,
            )

        roots: list[LayoutRoot] = []
        for idx, item in enumerate(roots_raw or []):
            entry = LayoutRoot.from_dict(item)
            if entry is None:
                return LayoutParseResult(
                    error=(
                        f"workspace_layout.roots[{idx}] must be an object "
                        "with a non-empty single-segment 'path' string"
                    ),
                    raw_block=raw,
                )
            roots.append(entry)

        # De-dup by path while preserving order; LLMs sometimes list the
        # same root twice when the spec went through a refinement pass.
        seen_paths: set[str] = set()
        dedup_roots: list[LayoutRoot] = []
        for r in roots:
            if r.path in seen_paths:
                continue
            seen_paths.add(r.path)
            dedup_roots.append(r)

        test_placement_raw = layout.get("test_placement", "")
        test_placement = (
            str(test_placement_raw).strip().lower()
            if isinstance(test_placement_raw, str) else ""
        )
        if test_placement and test_placement not in _VALID_TEST_PLACEMENTS:
            # Tolerant: drop the bad value, keep the rest of the layout.
            # The system prompt will fall back to neutral guidance.
            test_placement = ""

        root_files_raw = layout.get("root_files", [])
        root_files: list[str] = []
        if isinstance(root_files_raw, list):
            for item in root_files_raw:
                if isinstance(item, str) and item.strip():
                    norm = item.strip().lstrip("./")
                    if norm and "/" not in norm:
                        root_files.append(norm)

        return LayoutParseResult(
            roots=dedup_roots,
            test_placement=test_placement,
            root_files=root_files,
            raw_block=raw,
        )

    # Block absent — try the inventory-derivation fallback so specs that
    # predate the layout contract still produce a usable allowlist.
    inv = parse_inventory(spec_md)
    if inv.ok and inv.files:
        derived_paths: list[str] = []
        seen: set[str] = set()
        for fentry in inv.files:
            if "/" not in fentry.path:
                continue  # root-level file, not a directory
            first = fentry.path.split("/", 1)[0]
            if not first or first.startswith(".") or first in seen:
                continue
            seen.add(first)
            derived_paths.append(first)
        if derived_paths:
            return LayoutParseResult(
                roots=[LayoutRoot(path=p) for p in derived_paths],
                derived_from_inventory=True,
            )

    return LayoutParseResult(
        error="no fenced JSON block with 'workspace_layout' found",
    )


def cross_check_inventories(
    architecture: list[FileEntry],
    planning: list[FileEntry],
) -> list[InventoryDiagnostic]:
    """Compare architecture and planning inventories.

    Returns one diagnostic per discrepancy:
      * Files in architecture but not in plan → MISSING_FROM_PLAN.
        If a file with the same basename exists in plan at a different
        path, emits a PATH_MISMATCH instead with the plan path as
        suggested correction.
      * Files in plan but not in architecture → EXTRA_IN_PLAN (advisory).
    """
    arch_paths = {f.path for f in architecture}
    plan_paths = {f.path for f in planning}

    # Build basename → list of paths indexes for both sides so we can spot
    # `style.css` (arch) vs `src/styles.css` (plan) drift.
    arch_by_basename: dict[str, list[str]] = {}
    for f in architecture:
        arch_by_basename.setdefault(os.path.basename(f.path).lower(), []).append(f.path)
    plan_by_basename: dict[str, list[str]] = {}
    for f in planning:
        plan_by_basename.setdefault(os.path.basename(f.path).lower(), []).append(f.path)

    diagnostics: list[InventoryDiagnostic] = []

    for arch_path in sorted(arch_paths - plan_paths):
        basename = os.path.basename(arch_path).lower()
        plan_candidates = plan_by_basename.get(basename, [])
        # Stem-cousin search: `style.css` arch vs `styles.css` plan.
        if not plan_candidates:
            stem, ext = os.path.splitext(basename)
            if stem and ext:
                for plan_base, plan_paths_for in plan_by_basename.items():
                    p_stem, p_ext = os.path.splitext(plan_base)
                    if p_ext != ext or not p_stem:
                        continue
                    if (p_stem.startswith(stem) or stem.startswith(p_stem)) \
                            and abs(len(p_stem) - len(stem)) <= 1:
                        plan_candidates.extend(plan_paths_for)
        # Only treat as PATH_MISMATCH when exactly one plan candidate matches
        # the basename / stem — ambiguous cases stay as MISSING_FROM_PLAN.
        unique_candidates = [p for p in plan_candidates if p != arch_path]
        if len(unique_candidates) == 1:
            diagnostics.append(InventoryDiagnostic(
                kind="PATH_MISMATCH",
                file=arch_path,
                message=(
                    f"architecture lists '{arch_path}' but planning "
                    f"references '{unique_candidates[0]}' instead"
                ),
                suggested_path=unique_candidates[0],
            ))
        else:
            diagnostics.append(InventoryDiagnostic(
                kind="MISSING_FROM_PLAN",
                file=arch_path,
                message=(
                    f"architecture lists '{arch_path}' but planning "
                    "manifest does not include it"
                ),
            ))

    # EXTRA_IN_PLAN: only fire when the plan path's basename does NOT match
    # any architecture entry (otherwise it's already covered by the
    # PATH_MISMATCH emitted above and would be a duplicate complaint).
    accounted: set[str] = set()
    for d in diagnostics:
        if d.kind == "PATH_MISMATCH" and d.suggested_path:
            accounted.add(d.suggested_path)
    for plan_path in sorted(plan_paths - arch_paths):
        if plan_path in accounted:
            continue
        diagnostics.append(InventoryDiagnostic(
            kind="EXTRA_IN_PLAN",
            file=plan_path,
            message=(
                f"planning manifest includes '{plan_path}' but architecture "
                "inventory does not list it"
            ),
        ))

    return diagnostics


def check_files_on_disk(
    manifest: list[FileEntry],
    workspace_path: str,
) -> list[InventoryDiagnostic]:
    """Post-patch check: every file in the manifest must exist on disk.

    Returns one MISSING_FROM_DISK diagnostic per missing file.
    """
    diagnostics: list[InventoryDiagnostic] = []
    for entry in manifest:
        abs_path = os.path.join(workspace_path, entry.path)
        if not os.path.exists(abs_path):
            diagnostics.append(InventoryDiagnostic(
                kind="MISSING_FROM_DISK",
                file=entry.path,
                message=(
                    f"planning manifest lists '{entry.path}' but no file "
                    "was written to disk"
                ),
            ))
    return diagnostics


# Prompt fragments — exposed for graph.py to splice into architecture and
# planning prompts. Keeping them here means the two callers can never
# drift out of sync with what parse_inventory expects.

ARCHITECTURE_INVENTORY_INSTRUCTION = """\
**Required: file inventory block.** At the end of SPEC_ARCHITECTURE.md, \
include a fenced ```json block listing every file the implementation must \
create. Schema:

```json
{
  "files": [
    {"path": "index.html", "purpose": "entry HTML", "kind": "html"},
    {"path": "style.css",  "purpose": "global styling", "kind": "css"},
    {"path": "src/main.js", "purpose": "app bootstrap", "kind": "js"}
  ]
}
```

This block is parsed by the harness and cross-checked against the \
planning manifest. Use workspace-relative paths. Include every artifact \
the operator will see — HTML, CSS, JS, configuration, README, Makefile. \
Do NOT list `node_modules/`, generated build output, or test fixtures.

**Required: workspace layout block.** Immediately after the file \
inventory, include a SECOND fenced ```json block declaring the workspace \
layout. The harness derives the patcher's directory allowlist from this \
block — any CREATE_FILE targeting a top-level directory not listed in \
`roots[].path` (and not in `root_files`) will be rejected. Schema:

```json
{
  "workspace_layout": {
    "roots": [
      {"path": "client", "purpose": "React frontend SPA", "stack": "react"},
      {"path": "server", "purpose": "Express REST API",   "stack": "express"}
    ],
    "test_placement": "co-located",
    "root_files": ["package.json", "tsconfig.json", "vite.config.ts", ".eslintrc.json"]
  }
}
```

Rules for this block:
  * `roots` lists every top-level directory the implementation will \
create source code under. Use a **single-segment** directory name (e.g. \
`"client"`, not `"client/src"`). Source files inside `client/src/...` are \
implicitly covered by the `client` root.
  * `purpose` is a short human description (≤8 words). `stack` is a tech \
hint the system prompt uses to guide new-file placement in later turns \
(`react`, `vue`, `express`, `fastapi`, `flask`, `next`, etc.).
  * `test_placement` is one of `"co-located"` (tests next to source, e.g. \
`Foo.test.jsx` beside `Foo.jsx`), `"centralized"` (tests in a top-level \
`tests/` / `test/` / `__tests__/` directory), or `"mixed"`. The harness \
uses this to write the test-placement guidance into the system prompt.
  * `root_files` enumerates workspace-root files that must live OUTSIDE \
any `roots[].path` — manifests, tool configs, CI files. Other root-level \
files are rejected unless they match the harness's built-in safe list \
(`README.md`, `Makefile`, `pyproject.toml`, `requirements*.txt`, …).
  * For single-language workspaces with one source root, emit a single \
`roots` entry (e.g. `{"path": "src", "purpose": "library code", \
"stack": "python"}`).
"""

PLANNING_INVENTORY_INSTRUCTION = """\
**Required: file manifest block.** At the end of your blueprint, include \
a fenced ```json block listing every file you intend to CREATE_FILE. The \
schema MUST match the architecture inventory's schema:

```json
{
  "files": [
    {"path": "index.html", "purpose": "entry HTML", "kind": "html"},
    {"path": "style.css",  "purpose": "global styling", "kind": "css"}
  ]
}
```

The harness cross-checks this manifest against the architecture \
inventory in SPEC_ARCHITECTURE.md. Mismatches (missing files, renamed \
paths) block progression to patching. Use the EXACT paths from the \
architecture inventory — do not rename `style.css` to `src/styles.css` \
or vice versa.
"""
