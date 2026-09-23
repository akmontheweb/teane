"""Measure whether an LLM call actually USED the anchored spec region.

ADR-0008 action item 2. The ADR proposes replacing the whole-spec anchor in
``messages[0]`` with a per-story slice, on the premise that the loop nodes
(repair, patching, test-gen, regeneration) do not need the requirement bodies
they are handed on every call. That premise was established by reading code.
This module measures it on a real run instead, BEFORE anything changes.

Nothing here alters a prompt or a response: it observes dispatches and emits
``spec_region_usage`` events. It is gated by ``debug.measure_spec_usage``
(default false).

What "used" can and cannot mean
-------------------------------
A model's attention is not observable from outside. What IS observable is
whether the response reproduces content that exists ONLY in the spec region —
not in the rest of the prompt, and not in the generic vocabulary shared by
both. Two independent signals, reported separately because they fail
differently:

``cited``
    The response names a requirement id (``STORY-003``, ``FR-007``,
    ``NFR-002``) that appears in the spec region and nowhere else in the
    prompt. Precise, and an undercount: a node can be steered by a
    requirement without ever naming it.

``echoed``
    The response reproduces a distinctive word-shingle from a requirement
    body — one that appears nowhere else in the prompt. Catches silent use,
    and is the noisier of the two: shared domain vocabulary
    ("the employees table remains intact") can be echoed from the diagnostics
    rather than from the spec, which is exactly why a shingle that occurs
    outside the spec region is discarded before matching.

A call with neither signal is evidence the spec region was carried and not
used. Absence of evidence is weak for ONE call and strong across a run: if a
role never cites and never echoes across hundreds of dispatches, the region is
freight.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

#: Words per shingle. Long enough that a match is distinctive rather than a
#: stock phrase ("the user must be able to"), short enough to survive the
#: light rewording a model applies when it paraphrases a requirement.
_SHINGLE_WORDS = 8

#: Distinct matching shingles a requirement needs before it counts as echoed.
#: One is not enough: a single punctuation or line-break difference at a
#: window boundary makes one window unique to the response even when the
#: surrounding sentence is quoted from the prompt's own diagnostics, and that
#: lone artifact would score the call as "used". Since a false positive here
#: argues FOR carrying the spec region, the instrument must not manufacture
#: one. A genuine reproduction yields many overlapping windows.
_MIN_ECHO_SHINGLES = 2

#: Cap on shingles kept per requirement body. A 17 KB epic would otherwise
#: dominate the index; the cap keeps the measurement's cost flat per call.
_MAX_SHINGLES_PER_REQ = 400

#: The separator ``create_initial_state`` puts between the spec override and
#: the harness system prompt (graph.py:503-535).
_SPEC_SEPARATOR = "\n\n---\n\n"

#: How the harness system prompt opens. Used to confirm a split landed on the
#: real seam rather than on a ``---`` inside the spec's own markdown.
_SYSTEM_PROMPT_OPENER = "You are an expert software engineer"

_WORD_RE = re.compile(r"[a-z0-9_]+")

#: Module-level index cache. The spec region is byte-identical across every
#: call in a run (that is the point of the anchor), so it is parsed once.
_INDEX_CACHE: dict[str, "SpecIndex"] = {}


class SpecIndex:
    """Requirement blocks of one spec region, plus their distinctive shingles."""

    __slots__ = ("chars", "req_keys", "shingles", "req_of_shingle")

    def __init__(self, chars: int, req_keys: list[str],
                 shingles: dict[str, frozenset[str]]) -> None:
        self.chars = chars
        self.req_keys = req_keys
        self.shingles = shingles
        # Reverse map for O(1) attribution of a matched shingle to its owner.
        self.req_of_shingle: dict[str, str] = {}
        for key, shs in shingles.items():
            for sh in shs:
                self.req_of_shingle.setdefault(sh, key)


def split_spec_region(system_content: str) -> str:
    """Return the spec region of an anchored system prompt, or ``""``.

    The anchor is ``spec_override + "\\n\\n---\\n\\n" + system_prompt``, but the
    seam is not reliably that separator: a spec document whose own last line
    is a ``---`` rule produces a DOUBLED separator
    (``\\n\\n---\\n\\n\\n---\\n\\n``), and spec markdown contains horizontal rules
    throughout. So the harness prompt is located by its opening words and
    everything above them is the region, with any trailing rule/whitespace
    trimmed.

    An earlier version split on the separator and required the tail to start
    with the opener. That held in lumina-run12-20260923-0914 and silently
    failed in run 13, where the architecture document ended with a rule: the
    tail then began ``---\\n\\nYou are an expert…``, no candidate qualified,
    and the function reported "no spec region" for every repair call —
    disabling both the measurement and the slice without a word. Anchoring on
    content that is actually invariant, rather than on punctuation, is the
    point.
    """
    if not system_content:
        return ""
    idx = system_content.rfind(_SYSTEM_PROMPT_OPENER)
    if idx <= 0:
        return ""
    head = system_content[:idx]
    # Trim the seam: trailing blank lines and any run of `---` rules.
    head = re.sub(r"(?:\s*\n-{3,}[ \t]*)+\s*$", "", head)
    return head if head.strip() else ""


def _shingles(text: str) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < _SHINGLE_WORDS:
        return set()
    return {
        " ".join(words[i:i + _SHINGLE_WORDS])
        for i in range(len(words) - _SHINGLE_WORDS + 1)
    }


def build_index(spec_region: str) -> SpecIndex:
    """Parse the spec region into requirement blocks with distinctive shingles.

    Reuses ``req_ids.parse_spec_requirements`` — the same heading walk the
    ingest path uses — so a block here is the same unit as a ``requirements``
    row. A region that parses to nothing still yields a usable index: the
    whole region becomes one pseudo-block, so the measurement degrades to
    "was any spec-only text echoed" rather than failing shut.
    """
    key = hashlib.md5(spec_region.encode("utf-8", "replace")).hexdigest()
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    blocks: list[tuple[str, str]] = []
    try:
        from harness.req_ids import parse_spec_requirements
        for parsed in parse_spec_requirements(spec_region) or []:
            req_key = str(getattr(parsed, "req_key", "") or "")
            body = str(getattr(parsed, "body", "") or "")
            if req_key and body:
                blocks.append((req_key, body))
    except Exception as exc:  # noqa: BLE001 — measurement must never break a run
        logger.debug("[spec_usage] requirement parse failed: %s", exc)
    if not blocks:
        blocks = [("(unparsed-spec-region)", spec_region)]

    shingles: dict[str, frozenset[str]] = {}
    for req_key, body in blocks:
        shs = _shingles(body)
        if len(shs) > _MAX_SHINGLES_PER_REQ:
            shs = set(sorted(shs)[:_MAX_SHINGLES_PER_REQ])
        if shs:
            shingles[req_key] = frozenset(shs)

    index = SpecIndex(len(spec_region), [b[0] for b in blocks], shingles)
    _INDEX_CACHE[key] = index
    return index


def _req_ids_in(text: str) -> set[str]:
    try:
        from harness.req_ids import _ID_ALTERNATION  # type: ignore[attr-defined]
        return {m.group(0) for m in re.finditer(_ID_ALTERNATION, text)}
    except Exception:  # noqa: BLE001 — fall back to the common shapes
        return {
            m.group(0) for m in re.finditer(
                r"\b(?:STORY-NFR-\d{1,4}|STORY-\d{1,4}|FEAT-\d{1,4}|"
                r"EPIC-\d{1,4}|NFR(?:-[A-Z]+)?-\d{1,4}|FR-\d{1,4})\b",
                text,
            )
        }


def measure_call(
    *,
    messages: Iterable[dict[str, Any]],
    response_text: str,
) -> Optional[dict[str, Any]]:
    """Measure one dispatch. ``None`` when there is no anchored spec region.

    Returns a payload describing whether the response reproduced content
    unique to the spec region, and which requirements it came from.
    """
    msgs = list(messages or [])
    if not msgs:
        return None
    first = msgs[0] or {}
    if str(first.get("role", "")) != "system":
        return None
    spec_region = split_spec_region(str(first.get("content", "") or ""))
    if not spec_region:
        return None

    index = build_index(spec_region)

    # Everything in the prompt that is NOT the spec region. A signal must be
    # absent here to count — otherwise the response may have taken it from the
    # diagnostics, the story preamble or the source code.
    rest_parts = [str(first.get("content", "") or "")[len(spec_region):]]
    for m in msgs[1:]:
        if isinstance(m, dict):
            rest_parts.append(str(m.get("content", "") or ""))
    rest = "\n".join(rest_parts)

    text = str(response_text or "")
    payload: dict[str, Any] = {
        "spec_chars": index.chars,
        "req_blocks": len(index.req_keys),
        "response_chars": len(text),
    }
    if not text:
        payload.update({"cited": [], "echoed": [], "used": False,
                        "empty_response": True})
        return payload

    spec_ids = _req_ids_in(spec_region)
    rest_ids = _req_ids_in(rest)
    cited = sorted(
        (_req_ids_in(text) & spec_ids) - rest_ids
    )

    rest_shingles = _shingles(rest)
    resp_shingles = _shingles(text)
    hits: dict[str, int] = {}
    for sh in resp_shingles - rest_shingles:
        owner = index.req_of_shingle.get(sh)
        if owner:
            hits[owner] = hits.get(owner, 0) + 1
    echoed = sorted(k for k, n in hits.items() if n >= _MIN_ECHO_SHINGLES)

    payload["cited"] = cited
    payload["echoed"] = echoed
    payload["echo_shingles"] = sum(
        n for k, n in hits.items() if n >= _MIN_ECHO_SHINGLES
    )
    payload["used"] = bool(cited or echoed)
    return payload


def maybe_emit(
    *,
    messages: Iterable[dict[str, Any]],
    response_text: str,
    role: Any,
    cache_family: Optional[str] = None,
) -> None:
    """Measure and emit ``spec_region_usage``. Never raises."""
    try:
        payload = measure_call(messages=messages, response_text=response_text)
        if payload is None:
            return
        role_str = getattr(role, "value", None) or str(role)
        from harness.observability import emit_event
        emit_event(
            "spec_region_usage",
            role=role_str,
            cache_family=str(cache_family or role_str),
            **payload,
        )
    except Exception as exc:  # noqa: BLE001 — measurement must never break dispatch
        logger.debug("[spec_usage] measurement skipped: %s", exc)
