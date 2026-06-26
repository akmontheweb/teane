"""
Architecture-summary extractor.

The ``arch_doc_generator`` skill (``harness/skills/docgen/arch_doc.md``)
writes ``docs/SPEC_ARCHITECTURE.md`` and embeds a machine-readable
summary as a fenced ``jsonc`` block inside its §11 ("Machine-readable
summary") section. That block is the structured handoff to downstream
nodes: ``decomposition_node`` uses it to align feature / story
generation with the resolved endpoint + component map, and
``patching_node`` surfaces it as a deterministic planning preamble so
the patching LLM does not re-derive endpoint paths, schema names, or
contract locations on every turn.

This module owns the read+parse path. It is intentionally lenient —
every failure mode returns ``None`` with a debug log so a malformed or
missing summary degrades to "prose-only handoff" (the prior behaviour)
instead of crashing the graph.

Schema gate: ``schema_version`` must equal ``EXPECTED_SCHEMA_VERSION``.
Bumping the version is a deliberate cross-cut: edit the skill prompt
(§ "Outputs produced") AND this constant in the same commit; tests
verify they stay aligned.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

EXPECTED_SCHEMA_VERSION = 1
SPEC_ARCHITECTURE_REL_PATH = os.path.join("docs", "SPEC_ARCHITECTURE.md")

# Match fenced ``jsonc`` (or plain ``json``) blocks. The skill prompt
# emits ``jsonc`` but we accept ``json`` too — the LLM occasionally
# elides the ``c`` and the parse path is identical.
_FENCE_RE = re.compile(
    r"```(?:jsonc|json)\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)


def _strip_jsonc_comments(text: str) -> str:
    """Remove ``//`` line comments so :mod:`json` can parse a jsonc block.

    Block comments (``/* … */``) are not emitted by the skill prompt
    (only ``//`` line comments appear in the schema example) so we do
    not bother stripping them. If the LLM ever adds block comments,
    the parse will fail cleanly and ``load_arch_summary`` returns
    ``None`` — the consumer falls back to prose handoff.

    Quoted strings containing ``//`` are preserved by walking the
    text character-by-character and only entering "comment" mode
    when outside string state. The simpler regex-only approach
    (``re.sub(r'//.*$', '', flags=MULTILINE)``) corrupts URLs like
    ``"http://localhost"`` inside the JSON payload.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    string_quote = ""
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == string_quote:
                in_string = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_string = True
            string_quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            # Skip to end-of-line (preserve the newline so downstream
            # line counts in error messages remain useful).
            j = text.find("\n", i)
            if j < 0:
                break
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _extract_last_jsonc_block(markdown: str) -> Optional[str]:
    """Return the body of the LAST fenced ``jsonc``/``json`` block.

    The arch_doc skill emits a single summary in §11 near the end of
    the document. Using the LAST match (not the first) tolerates
    earlier illustrative ``json`` snippets in §10 (the error response
    shape example) or in ADRs without misreading them as the summary.
    """
    matches = list(_FENCE_RE.finditer(markdown))
    if not matches:
        return None
    return matches[-1].group(1)


def load_arch_summary(workspace_path: str) -> Optional[dict[str, Any]]:
    """Read ``<workspace>/docs/SPEC_ARCHITECTURE.md`` and return the
    parsed §11 summary, or ``None`` on any failure.

    Failure modes (each returns ``None`` + a debug log):
      - workspace_path is empty / file missing / not readable
      - no fenced ``jsonc`` block found
      - JSON parse error after stripping ``//`` comments
      - ``schema_version`` missing or != :data:`EXPECTED_SCHEMA_VERSION`

    Callers MUST treat ``None`` as "fall back to prose handoff" —
    never raise on absence. The prose document remains the contract
    for human readers and the patching preamble's free-text section.
    """
    if not workspace_path:
        logger.debug("[arch_summary] empty workspace_path — skipping load")
        return None
    path = os.path.join(workspace_path, SPEC_ARCHITECTURE_REL_PATH)
    if not os.path.isfile(path):
        logger.debug("[arch_summary] %s not found — skipping load", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as fp:
            markdown = fp.read()
    except OSError as exc:
        logger.debug("[arch_summary] cannot read %s: %s", path, exc)
        return None

    raw = _extract_last_jsonc_block(markdown)
    if raw is None:
        logger.debug("[arch_summary] no jsonc fence in %s — prose-only handoff", path)
        return None

    try:
        data = json.loads(_strip_jsonc_comments(raw))
    except json.JSONDecodeError as exc:
        logger.debug("[arch_summary] malformed jsonc block in %s: %s", path, exc)
        return None

    if not isinstance(data, dict):
        logger.debug("[arch_summary] jsonc block in %s is not an object", path)
        return None

    version = data.get("schema_version")
    if version != EXPECTED_SCHEMA_VERSION:
        logger.debug(
            "[arch_summary] schema_version mismatch in %s: got %r, expected %d",
            path, version, EXPECTED_SCHEMA_VERSION,
        )
        return None

    logger.info(
        "[arch_summary] loaded summary from %s (backend=%s frontend=%s endpoints=%d)",
        path,
        data.get("backend_language", "?"),
        data.get("frontend", "?"),
        len(((data.get("backend") or {}).get("endpoints")) or []),
    )
    return data


_CONSUMER_GUIDANCE: dict[str, str] = {
    "patcher": (
        "These decisions are RESOLVED — do not re-derive paths, "
        "schema names, or contract locations. If a decision you "
        "need is NOT listed here, emit `<<<NO_PROGRESS reason=\"ARCH_GAP: "
        "<what is missing>\">>>` instead of guessing."
    ),
    "reviewer": (
        "These tables are the resolved contract. When the modified "
        "code drifts from a listed endpoint path, schema name, "
        "contract location, or component path, raise it as a "
        "finding (severity: high). Endpoints / components not yet "
        "implemented are NOT findings — only contradictions are."
    ),
    "test_generator": (
        "Use these tables as your coverage target. Every endpoint "
        "below should have at least one test that hits the listed "
        "method+path and asserts against the listed response "
        "schema; every component listed should have at least one "
        "render test. Schema names map directly to the request / "
        "response classes the patcher generated."
    ),
    "security": (
        "Security fixes must STAY CONSISTENT with the resolved "
        "stack below. Do NOT swap in a different secrets store, "
        "auth library, ORM, or contract path — the architecture "
        "doc has already decided those. Pull credentials from the "
        "declared config source (env / Pydantic Settings / Spring "
        "@Value), use the declared auth strategy for any new "
        "guards, and reference the same schema names listed in the "
        "endpoint map. Findings against files outside this map are "
        "still real bugs — fix them in place; just do not "
        "re-architect."
    ),
}


def render_arch_preamble(
    summary: Optional[dict[str, Any]],
    *,
    consumer: str = "patcher",
) -> str:
    """Render the architecture-summary preamble.

    Returns the empty string when ``summary`` is ``None`` or carries no
    actionable structured fields — keeps the planning prompt byte-
    identical for projects whose arch doc has no §11 block (legacy or
    third-party docs).

    The block is deliberately compact: endpoint table, component table
    (only when ``frontend != "none"``), contract path. Free-text design
    rationale stays in the prose document the system prompt already
    prepends; this preamble is the structural index the LLM should not
    have to re-derive.

    Args:
        summary: parsed §11 jsonc, or ``None``.
        consumer: which downstream node will read this — selects the
            one-paragraph guidance block at the top. One of
            ``"patcher"`` (default; tells the LLM to emit
            ``NO_PROGRESS`` on missing decisions), ``"reviewer"``
            (tells the reviewer to flag *drift* — endpoints / schemas
            implemented inconsistently with the tables), or
            ``"test_generator"`` (tells the test author to treat the
            tables as a coverage target). Unknown values fall back to
            the patcher block.
    """
    if not summary or not isinstance(summary, dict):
        return ""

    backend_lang = summary.get("backend_language") or "?"
    frontend = summary.get("frontend") or "none"
    db_engine = summary.get("db_engine") or "none"
    auth = summary.get("auth_strategy") or "none"

    backend = summary.get("backend") or {}
    endpoints = backend.get("endpoints") or []
    contract = summary.get("contract") or {}
    contract_path = contract.get("openapi_spec_path") or ""

    lines: list[str] = []
    lines.append(
        "## Architecture summary (from docs/SPEC_ARCHITECTURE.md §11)\n"
    )
    guidance = _CONSUMER_GUIDANCE.get(consumer, _CONSUMER_GUIDANCE["patcher"])
    lines.append(guidance + "\n")
    lines.append(
        f"- **Backend stack:** `{backend_lang}` · DB `{db_engine}` · auth `{auth}`"
    )
    lines.append(f"- **Frontend stack:** `{frontend}`")
    if contract_path:
        lines.append(
            f"- **OpenAPI contract path (fixed):** `{contract_path}`"
        )
    extraction = contract.get("extraction_command")
    if extraction:
        # Show the command on one short line — verbose multi-line
        # commands stay in the prose doc; the LLM only needs to know
        # the canonical path here.
        head = extraction.strip().splitlines()[0][:120]
        lines.append(f"- **OpenAPI extraction (after scaffold):** `{head}`")

    if endpoints:
        lines.append("\n### Endpoint map\n")
        lines.append(
            "| EP | Method | Path | Request | Response | Auth | RSD IDs |"
        )
        lines.append("|----|--------|------|---------|----------|------|---------|")
        for ep in endpoints:
            if not isinstance(ep, dict):
                continue
            rsd_ids = _format_rsd_ids(ep)
            lines.append(
                "| {ep_id} | {method} | `{path}` | {req} | {res} | {auth} | {ids} |".format(
                    ep_id=ep.get("id", "?"),
                    method=(ep.get("method") or "").upper(),
                    path=ep.get("path", ""),
                    req=ep.get("request_schema") or "—",
                    res=ep.get("response_schema") or "—",
                    auth=("Yes" if ep.get("auth_required") else "No"),
                    ids=rsd_ids,
                )
            )

    if frontend and frontend != "none":
        # The §11 schema separates the enum (``frontend``) from the
        # object (``frontend_spec``). An LLM that emits ``frontend`` as
        # an object instead of the enum (early-draft prompt regression)
        # should still render — fall back to the legacy field if
        # ``frontend_spec`` is absent.
        spec = summary.get("frontend_spec")
        if not isinstance(spec, dict):
            legacy = summary.get("frontend")
            spec = legacy if isinstance(legacy, dict) else {}
        components = spec.get("components") or []
        if components:
            lines.append("\n### Component map\n")
            lines.append("| Component | Path | RSD IDs | Radix primitives |")
            lines.append("|-----------|------|---------|-------------------|")
            for cp in components:
                if not isinstance(cp, dict):
                    continue
                primitives = cp.get("radix_primitives") or []
                lines.append(
                    "| {name} | `{path}` | {ids} | {prims} |".format(
                        name=cp.get("name", "?"),
                        path=cp.get("path", ""),
                        ids=_format_rsd_ids(cp),
                        prims=(", ".join(primitives) if primitives else "—"),
                    )
                )

    lines.append("\n---\n")
    return "\n".join(lines) + "\n"


def _format_rsd_ids(entry: dict[str, Any]) -> str:
    """Join STORY-N / FEAT-N / FR-N IDs into a single human-readable cell."""
    bits: list[str] = []
    for key in ("rsd_story_ids", "rsd_feature_ids", "rsd_fr_ids"):
        for v in entry.get(key) or []:
            if isinstance(v, str) and v not in bits:
                bits.append(v)
    return " · ".join(bits) if bits else "—"
