"""Shared regexes + parser for requirement identifiers.

teane recognises three identifier families in ``docs/SPEC_REQUIREMENTS.md``:

- ``FR-NNN`` — functional requirements (``FR-007``)
- ``NFR-XXX-NNN`` — non-functional requirements grouped by category
  (``NFR-SEC-001``, ``NFR-PERF-014``)
- ``US-NN-NN`` — user stories from the discovery doc
  (``US-03-02``)

Both the v5 ``requirements_ingest`` (parses headings into rows in the
``requirements`` table) and the v5 SQL traceability audit
(``harness/traceability.py``) share these regexes — they used to live
only in ``traceability.py`` for the text-grep audit; lifting them
here keeps a single source of truth as the audit migrates to
DB-backed queries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Identifier patterns. Anchored on word boundaries so plain text
# containing the token gets matched without picking up sub-strings
# inside identifiers like ``USER-1234``.
FR_ID_RE = re.compile(r"\bFR-\d{1,4}\b")
US_ID_RE = re.compile(r"\bUS-\d{1,3}-\d{1,3}\b")
NFR_ID_RE = re.compile(r"\bNFR-[A-Z]+-\d{1,4}\b")

# Heading patterns the ingest parser looks for. Convention matches what
# the existing decomposition LLM emits and what `docs/SPEC_REQUIREMENTS.md`
# in the teane workspace uses today:
#
#   ### FR-007: One-line title
#   #### NFR-SEC-001: Encrypt session tokens at rest
#
# Two or more ``#``, then the id token, then ``:`` + title. Anything
# after the title goes into ``body`` (captured until the next heading
# by ``parse_spec_requirements`` — see below).
_HEADING_RE = re.compile(
    r"^\s*#{2,}\s+(?P<id>"
    r"FR-\d{1,4}"
    r"|NFR-[A-Z]+-\d{1,4}"
    r"|US-\d{1,3}-\d{1,3}"
    r")\s*[:\-]\s*(?P<title>.+?)\s*$"
)


# Terminators that close out the body of the current requirement. Any
# ``#``-prefixed heading qualifies (a new requirement OR a section
# header), as does a horizontal rule.
_BODY_TERMINATOR_RE = re.compile(r"^\s*(?:#{1,}\s|---\s*$)")


def kind_for(req_key: str) -> Optional[str]:
    """Return the ``kind`` string (``fr``/``nfr``/``us``) for a given
    requirement id, or ``None`` when the token doesn't match any
    known family. Used by ``requirements_ingest`` to set the
    ``requirements.kind`` column without re-running every regex.
    """
    if FR_ID_RE.fullmatch(req_key):
        return "fr"
    if NFR_ID_RE.fullmatch(req_key):
        return "nfr"
    if US_ID_RE.fullmatch(req_key):
        return "us"
    return None


@dataclass(frozen=True)
class ParsedRequirement:
    """One requirement row scraped from a spec file.

    ``source_line`` is 1-indexed (matches editor line numbers and the
    convention git/grep use). ``body`` may be empty when the heading
    has no following prose before the next terminator.
    """
    req_key: str
    kind: str
    title: str
    body: str
    source_line: int


def parse_spec_requirements(
    text: str,
) -> list[ParsedRequirement]:
    """Walk ``text`` and yield one :class:`ParsedRequirement` per
    heading that matches the FR/NFR/US convention.

    The body is the lines between the current heading and the next
    heading or horizontal rule (``---``). Leading/trailing blank
    lines are trimmed; internal whitespace is preserved verbatim so
    snippets like fenced code blocks survive intact.

    Duplicate ``req_key`` headings are NOT deduplicated here — caller
    (``requirements_ingest``) relies on the DB's ``ON CONFLICT
    DO UPDATE`` to UPSERT, so a late heading wins. This matches the
    "spec edits propagate" contract documented in
    ``harness/story_state.py:create_requirements``.
    """
    out: list[ParsedRequirement] = []
    lines = text.splitlines()
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        m = _HEADING_RE.match(line)
        if not m:
            i += 1
            continue
        req_key = m.group("id")
        kind = kind_for(req_key)
        if kind is None:
            # Shouldn't happen given the regex above, but defensive:
            # an id that matches no kind is skipped rather than crashing.
            i += 1
            continue
        title = m.group("title").strip()
        body_start = i + 1
        j = body_start
        while j < n and not _BODY_TERMINATOR_RE.match(lines[j]):
            j += 1
        body_lines = lines[body_start:j]
        # Strip leading/trailing blank lines but keep internal layout.
        while body_lines and not body_lines[0].strip():
            body_lines.pop(0)
        while body_lines and not body_lines[-1].strip():
            body_lines.pop()
        out.append(ParsedRequirement(
            req_key=req_key,
            kind=kind,
            title=title,
            body="\n".join(body_lines),
            source_line=i + 1,  # 1-indexed
        ))
        i = j
    return out
