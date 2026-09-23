"""Assemble the spec context one call actually needs (ADR-0008, item 3).

The anchored ``messages[0]`` carries the whole requirements + architecture
specification to every node. ``lumina-run12-20260923-0914`` measured what that
buys: ``repair`` reproduced nothing unique to the region across 24 dispatches,
while ``patching`` used three stories' criteria in a single call. So the
region is freight for repair and load-bearing for patching, and this module
replaces it — for one role at a time — with the stories in scope plus the
cross-cutting preamble.

Two halves, matching the ADR's two tiers:

``tier1_preamble``
    The leading cross-cutting sections of the requirements document (product
    decisions, assumptions, data model, API conventions, test conventions) —
    everything before the first requirement-bearing heading. Needed
    everywhere, stable across a run, and therefore still cache-friendly.

``scoped_slice``
    The requirement bodies for the stories in scope, their parent feature and
    epic, and their acceptance criteria — read from the ``requirements`` and
    ``acceptance_criteria`` rows that ``_ingest_requirements`` already wrote,
    not re-parsed from the file.

Safety rule, from the ADR amendment: **an unresolved scope keeps the full
region.** A node whose stories cannot be determined is a bug to fix, and the
symptom must be a large prompt, never a silently missing requirement.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Requirement-bearing headings. The first one ends the cross-cutting
#: preamble: everything above it is context every node shares, everything
#: below is one story's business. Level is deliberately loose — the spec
#: writes stories at ``####`` and epics at ``##``, and spec_reconciler
#: tolerates any depth, so the ID token is what identifies the heading.
_REQ_HEADING_RE = re.compile(
    r"^#{2,6}\s+.*?\b(?:EPIC|FEAT|STORY)[-‐-―−]",
    re.IGNORECASE | re.MULTILINE,
)

#: ``**Parent feature:** FEAT-001`` / ``**Parent epic:** EPIC-001`` — the
#: hierarchy markers decomposition writes into requirement bodies and
#: story_state already reads (story_state.py:1361).
_PARENT_RE = re.compile(
    r"\*\*Parent (?:feature|epic):\*\*\s*([A-Za-z]+[-‐-―−]\d+)",
    re.IGNORECASE,
)


def tier1_preamble(spec_region: str) -> str:
    """The cross-cutting head of the requirements document.

    Returns everything before the first EPIC/FEAT/STORY heading. When no such
    heading exists the whole region is cross-cutting by definition and is
    returned unchanged — trimming to nothing would be the silent
    under-injection this module exists to avoid.
    """
    if not spec_region:
        return ""
    match = _REQ_HEADING_RE.search(spec_region)
    if not match:
        return spec_region
    return spec_region[:match.start()].rstrip() + "\n"


def _parents_of(body: str) -> list[str]:
    return [m.group(1) for m in _PARENT_RE.finditer(body or "")]


def scoped_slice(
    workspace: str, story_keys: list[str], *, max_chars: int = 24000,
) -> str:
    """Render the requirement context for ``story_keys``, or ``""``.

    Walks each story to its parent feature and epic so a story whose contract
    lives one level up (a feature-level AC naming the response envelope) is
    not cut off from it. Returns ``""`` when nothing resolves — the caller
    treats that as an unresolved scope and keeps the full region.
    """
    keys = [k for k in dict.fromkeys(story_keys or []) if k]
    if not workspace or not keys:
        return ""

    rows: list[tuple[str, str]] = []
    acs: dict[str, list[str]] = {}
    try:
        from harness import story_state as sst
        app_name = sst.app_name_for_workspace(workspace)
        conn = sst.open_story_db(workspace_path=workspace)
        try:
            seen: set[str] = set()
            queue = list(keys)
            while queue:
                key = queue.pop(0)
                if key in seen:
                    continue
                seen.add(key)
                req = sst.get_requirement_by_key(conn, app_name, key)
                if req:
                    body = str(req.get("body") or "")
                    if body.strip():
                        rows.append((key, body))
                    # Parents last: a feature's own body may name further
                    # parents, and the walk must terminate on `seen`.
                    queue.extend(_parents_of(body))
                story = sst.get_story(conn, app_name, key)
                if story:
                    crit = [str(a) for a in (story.get("acceptance_criteria") or [])]
                    if crit:
                        acs[key] = crit
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001 — never break a dispatch
        logger.debug("[spec_slice] DB read failed: %s", exc)
        return ""

    if not rows and not acs:
        return ""

    parts: list[str] = [
        "## Specification extract for the work in hand\n"
        "(the requirements this call is scoped to, taken from the plan's own "
        "rows — the rest of the specification is not reproduced)\n"
    ]
    for key, body in rows:
        parts.append(f"### {key}\n{body.strip()}\n")
    for key, crit in acs.items():
        if key in {r[0] for r in rows}:
            # The body already carries them; listing twice invites the model
            # to read a stale copy against a fresh one.
            continue
        parts.append(
            f"### {key} — acceptance criteria\n"
            + "\n".join(f"- {c}" for c in crit) + "\n"
        )
    out = "\n".join(parts)
    if len(out) > max_chars:
        out = out[:max_chars].rstrip() + "\n\n[extract truncated]\n"
    return out


def resolve_scope(state: dict[str, Any]) -> list[str]:
    """Story keys in scope for this call, most specific first.

    A single active story wins; otherwise the batch's stories, which is what
    run 12 measured a patching call using three of at once. Empty means
    unresolved — the caller must then keep the full region.
    """
    story = str(state.get("current_story_id", "") or "")
    if story:
        return [story]
    batch = [str(k) for k in (state.get("batch_patched_story_keys") or []) if k]
    if batch:
        return batch
    workspace = str(state.get("workspace_path", "") or "")
    batch_id = int(state.get("current_batch_id") or 0)
    if workspace and batch_id:
        try:
            from harness import story_state as sst
            app_name = sst.app_name_for_workspace(workspace)
            conn = sst.open_story_db(workspace_path=workspace)
            try:
                return [str(k) for k in sst.story_keys_for_batch(
                    conn, app_name, batch_id) or []]
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("[spec_slice] batch lookup failed: %s", exc)
    return []


def apply_to_messages(
    messages: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    consumer: str,
    position: str = "before_last",
) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
    """Return ``(messages, telemetry)`` with the spec region narrowed.

    ``messages[0]`` keeps the Tier-1 preamble and the harness system prompt,
    so the prefix stays stable for the whole run rather than changing per
    story; the story extract is appended as a LATER message, which is the
    invariant ``graph.py:1721`` ("never mutated — it maximizes prefix
    caching") depends on.

    ``position`` decides where the extract goes. The default,
    ``"before_last"``, keeps whatever the caller put last in that position —
    for repair that is ``_REPAIR_FORMAT_REMINDER``, which restates the patch
    DSL and carries the round's blocking correction, and whose whole design
    is to have the last word (see the comment at its append site; burying it
    once cost lumina-run7 five byte-identical rounds).

    lumina-run14-20260923-2352 appended the extract at the very end instead,
    displacing that reminder behind 6,876 chars of specification. Ten repair
    rounds came back as ``UNPARSED`` — the model answering in
    ``anthropic_xml`` ``edit_file`` tool calls rather than the harness DSL —
    and the run ended on persistent_build_failure. A slice that changes what
    the model is told to DO is not a context optimisation.

    The input list is not mutated. Returns the original list unchanged, and
    ``None`` telemetry, whenever anything is missing: no anchored region, no
    resolvable scope, or nothing in the DB for the stories named.
    """
    from harness.spec_usage import split_spec_region

    msgs = list(messages or [])
    if not msgs or str((msgs[0] or {}).get("role", "")) != "system":
        return messages, None
    system_content = str((msgs[0] or {}).get("content", "") or "")
    region = split_spec_region(system_content)
    if not region:
        return messages, None

    scope = resolve_scope(state)
    if not scope:
        logger.info(
            "[spec_slice] %s: no story scope resolved — keeping the full "
            "spec region rather than guessing.", consumer,
        )
        return messages, None

    extract = scoped_slice(str(state.get("workspace_path", "") or ""), scope)
    if not extract:
        logger.info(
            "[spec_slice] %s: scope %s resolved to no requirement rows — "
            "keeping the full spec region.", consumer, scope,
        )
        return messages, None

    preamble = tier1_preamble(region)
    msgs[0] = dict(msgs[0])
    msgs[0]["content"] = preamble + system_content[len(region):]
    entry = {"role": "user", "content": extract}
    if position == "before_last" and len(msgs) >= 2:
        msgs.insert(len(msgs) - 1, entry)
    else:
        msgs.append(entry)

    telemetry = {
        "consumer": consumer,
        "stories": list(scope),
        "region_chars": len(region),
        "preamble_chars": len(preamble),
        "extract_chars": len(extract),
        "saved_chars": max(0, len(region) - len(preamble) - len(extract)),
    }
    return msgs, telemetry
