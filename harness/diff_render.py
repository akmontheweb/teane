"""Server-side unified-diff renderer for the Phase-5 patch-approval
HITL card (and Phase-6 config-preset preview).

Uses stdlib :mod:`difflib` — no extra dependencies, no client-side
diff library, no LLM call. The rendered HTML uses the ``.diff*``
primitives from :file:`harness/static/css/tokens.css`.

Kept out of :file:`harness/dashboard.py` because:

* The renderer is pure — no HTTP concerns — so it's naturally reusable
  by CLI callers, tests, and future non-dashboard surfaces.
* It's small; growing dashboard.py past 6k lines has already become
  a maintenance smell (see plan file's Phase-0 notes).
"""

from __future__ import annotations

import difflib
import html
from typing import Iterable


_DIFF_MAX_LINES = 600
_BYTE_NUL = "\x00"


def looks_binary(payload: bytes | str) -> bool:
    """Cheap heuristic: any NUL byte in the first 1024 units means
    we don't want to render this as text. Mirrors what git uses
    before running the diff engine."""
    if isinstance(payload, bytes):
        window = payload[:1024]
        return b"\x00" in window
    return _BYTE_NUL in payload[:1024]


def _summary_line(a_len: int, b_len: int) -> str:
    add = max(0, b_len - a_len)
    sub = max(0, a_len - b_len)
    parts: list[str] = []
    if add:
        parts.append(f"+{add}")
    if sub:
        parts.append(f"-{sub}")
    if not parts:
        parts.append("=")
    return " · ".join(parts)


def render_unified_diff(
    before: str,
    after: str,
    *,
    file_path: str = "",
    context_lines: int = 3,
    max_lines: int = _DIFF_MAX_LINES,
) -> str:
    """Return an HTML string with the unified diff between ``before``
    and ``after``.

    Class names:
      * ``.diff``          — outermost container
      * ``.diff__file``    — header row with the path + summary
      * ``.diff__hunk``    — each ``@@`` hunk header
      * ``.diff-add`` / ``.diff-del`` / ``.diff-ctx`` — line classes

    ``max_lines`` caps the emitted body so a 10 KB diff renders but a
    500 KB one collapses to a "diff truncated" marker instead of
    tanking the page render time.
    """
    before_lines = before.splitlines() or [""]
    after_lines = after.splitlines() or [""]
    file_label = file_path or "(no path)"
    summary = _summary_line(len(before_lines), len(after_lines))
    head = (
        f"<div class='diff__file'>"
        f"<span>{html.escape(file_label)}</span>"
        f"<span class='muted small tabular'>{html.escape(summary)}</span>"
        f"</div>"
    )
    if looks_binary(before) or looks_binary(after):
        return (
            "<div class='diff'>"
            + head +
            "<div class='diff-binary'>Binary file — diff not shown</div>"
            "</div>"
        )
    body_parts: list[str] = []
    emitted = 0
    for line in difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile="before",
        tofile="after",
        n=context_lines,
        lineterm="",
    ):
        emitted += 1
        if emitted > max_lines:
            body_parts.append(
                "<span class='diff-ctx muted'>… diff truncated "
                f"(&gt; {max_lines} lines)</span>"
            )
            break
        # Skip the two synthetic file-header rows difflib emits — the
        # `.diff__file` block above already conveys the file.
        if line.startswith("--- ") or line.startswith("+++ "):
            continue
        if line.startswith("@@"):
            body_parts.append(f"<span class='diff__hunk'>{html.escape(line)}</span>")
        elif line.startswith("+"):
            body_parts.append(f"<span class='diff-add'>{html.escape(line)}</span>")
        elif line.startswith("-"):
            body_parts.append(f"<span class='diff-del'>{html.escape(line)}</span>")
        else:
            body_parts.append(f"<span class='diff-ctx'>{html.escape(line)}</span>")
    if not body_parts:
        body_parts.append(
            "<span class='diff-ctx muted'>Files are identical.</span>"
        )
    return (
        "<div class='diff'>"
        + head
        + "<div class='diff__body mono'>"
        + "".join(body_parts)
        + "</div>"
        "</div>"
    )


def render_patch_list(patches: Iterable[dict]) -> str:
    """Render every patch entry from a Phase-5 approval payload as a
    stacked series of ``render_unified_diff`` blocks (Linear PR-style,
    no tabs). Unknown / malformed entries are skipped silently rather
    than throwing — the HITL card already exists to let the operator
    react."""
    blocks: list[str] = []
    for p in patches or []:
        if not isinstance(p, dict):
            continue
        if p.get("is_binary"):
            path = str(p.get("path") or "(no path)")
            op = str(p.get("operation") or "?")
            size = p.get("size_after") or p.get("size_before") or 0
            blocks.append(
                "<div class='diff'>"
                f"<div class='diff__file'><span>{html.escape(path)}</span>"
                f"<span class='muted small'>{html.escape(op)} · {int(size)} B</span>"
                "</div>"
                "<div class='diff-binary'>Binary file — diff not shown</div>"
                "</div>"
            )
            continue
        before = str(p.get("before") or "")
        after = str(p.get("after") or "")
        path = str(p.get("path") or "(no path)")
        blocks.append(render_unified_diff(before, after, file_path=path))
    if not blocks:
        return "<p class='muted small'>No patches to review.</p>"
    return "<div class='stack--sm'>" + "".join(blocks) + "</div>"
