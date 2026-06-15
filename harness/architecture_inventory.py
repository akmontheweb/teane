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
