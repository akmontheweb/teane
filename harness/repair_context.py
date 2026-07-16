"""Repair-loop context pruning.

Repair rounds accumulate every prior turn — system prompt + initial
task + every LLM emission + every gate rejection + every error
summary — and replay the whole array on each call. Once total_repairs
crosses ~3–4 rounds the LLM starts arguing with its own past 15-20k
character emissions, because a REPLACE_BLOCK search that references
"the code I wrote three rounds ago" still lives in the context.

Finsearch session 156032347 STORY-042 spent 10+ repair rounds trying
to remove a `del _store[store_key]` branch that wasn't in the current
file — it was in an assistant message from round 4. The LLM's own
history was polluting the mental model. Pruning that history from
round 4 onward gives every subsequent round a clean recent view.

The helper preserves:
  * message[0] — the immutable system prompt (RSD + skills + style
    guides + edit invariants). This is the load-bearing prefix that
    prefix-cache hits target; touching it would evict the cache.
  * message[1] — the initial user prompt (task setup). The LLM needs
    to know what the session is even about; dropping this collapses
    grounding.
  * The last ``keep_tail`` messages — the recent 2–3 turns carry
    the current failure signal + the last patch attempt + the
    per-file rejection directives from compose_patch_feedback.
    Anything between msg[1] and the tail is provably not helping.

Prefix-cache economics: the pruned prefix (msg[0] + msg[1]) stays
byte-identical across every repair round, so the fixed-content cache
key survives pruning. The tail is where the fresh context lives —
cost paid there anyway.

Two entry points:
  * ``prune_repair_messages`` — deterministic head+tail truncation.
  * ``condense_repair_messages`` — same window, but the dropped middle
    is folded into a one-message LLM-written digest (the OpenHands
    "summarizing condenser" pattern: up to ~2x cost reduction with no
    measured performance loss). Incremental per round, fail-open to
    the deterministic prune.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Awaitable, Callable, Optional, Sequence

logger = logging.getLogger(__name__)


DEFAULT_PRUNE_AFTER_ROUND: int = 3
"""Repair rounds before pruning kicks in. Rounds 1-3 replay the full
history unchanged so the LLM has every chance to converge with maximum
context; from round 4 onward the tail-only window activates."""


DEFAULT_KEEP_TAIL: int = 6
"""How many recent messages survive the prune. Six turns typically
covers: last patcher attempt (assistant) + patcher rejection (system)
+ compiler diagnostics (user) + budget/format reminders (system) +
error summary (user) + one spare — enough recent signal without
letting the older-round tail creep back in."""


def prune_repair_messages(
    messages: Sequence[dict[str, Any]],
    *,
    total_repairs: int,
    prune_after_round: int = DEFAULT_PRUNE_AFTER_ROUND,
    keep_tail: int = DEFAULT_KEEP_TAIL,
) -> list[dict[str, Any]]:
    """Return a pruned copy of ``messages`` when repair has iterated past
    ``prune_after_round``, otherwise the input list verbatim (list-copied
    so the caller owns a fresh reference).

    The prune keeps ``messages[0:2]`` (system + initial user) and the
    last ``keep_tail`` messages, dropping everything in between. When
    the drop would remove nothing (short array, or would leave the
    array unchanged), the input is returned verbatim.
    """
    n = len(messages)
    if total_repairs <= prune_after_round:
        return list(messages)
    # Nothing to prune: array is short enough that head+tail already
    # covers everything.
    if n <= 2 + keep_tail:
        return list(messages)
    head = list(messages[:2])
    tail = list(messages[n - keep_tail:])
    dropped = n - len(head) - len(tail)
    logger.info(
        "[repair_context] pruned %d intermediate message(s) at "
        "total_repairs=%d (kept head=2 + tail=%d of %d total)",
        dropped, total_repairs, keep_tail, n,
    )
    return head + tail


DEFAULT_SUMMARY_MAX_CHARS: int = 2000
"""Upper bound on the condensed-history digest body. Roughly 500
tokens — two orders of magnitude below the 15-20k-char emissions it
replaces, but enough for one line per dropped round."""

_DELTA_MSG_CLIP: int = 1200
"""Per-message clip when feeding dropped turns to the condenser LLM.
Repair turns carry full patch bodies; the summary only needs the
intent + outcome, which live in the first ~1200 chars."""

_DELTA_MAX_MSGS: int = 30
"""At most this many newly-dropped messages per condense call; when
more accumulate (huge round gap after an LLM outage) the oldest are
elided with a marker rather than blowing up the judgment prompt."""

_CONDENSE_CACHE: dict[str, dict[str, Any]] = {}
_CONDENSE_CACHE_MAX_KEYS: int = 8
"""Per-process incremental-summary cache, keyed by the immutable
prefix (system + initial user message). Each repair round only
summarizes the turns that dropped out of the window since the last
round, folding them into the cached running summary. Process restart
just re-condenses once — no checkpoint schema involvement."""


def _cache_key(messages: Sequence[dict[str, Any]]) -> str:
    prefix = _message_text(messages[0]) + "\x00" + _message_text(messages[1])
    return hashlib.sha256(prefix.encode("utf-8", errors="replace")).hexdigest()


def _message_text(message: dict[str, Any]) -> str:
    """Flatten a message's content to plain text (str or block-list form)."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return str(content)


def _build_condense_prompt(
    prev_summary: str,
    delta: Sequence[dict[str, Any]],
    max_chars: int,
) -> str:
    if len(delta) > _DELTA_MAX_MSGS:
        elided = len(delta) - _DELTA_MAX_MSGS
        shown: list[Any] = list(delta[:_DELTA_MAX_MSGS // 2])
        shown.append(f"... ({elided} older turn(s) elided) ...")
        shown.extend(delta[len(delta) - _DELTA_MAX_MSGS // 2:])
    else:
        shown = list(delta)
    lines: list[str] = []
    for item in shown:
        if isinstance(item, str):
            lines.append(item)
            continue
        role = str(item.get("role", "?"))
        text = " ".join(_message_text(item).split())[:_DELTA_MSG_CLIP]
        lines.append(f"[{role}] {text}")
    prev_block = (
        f"Existing summary of even earlier rounds:\n{prev_summary}\n\n"
        if prev_summary else ""
    )
    return (
        "You are condensing the history of an automated code-repair loop "
        "so the next round keeps signal without replaying every turn.\n\n"
        f"{prev_block}"
        "Newly dropped turns (oldest first, each clipped):\n"
        + "\n".join(lines)
        + "\n\nMerge everything above into ONE updated summary, at most "
        f"{max_chars} characters. Keep only what helps the next repair "
        "round: what was attempted each round and its outcome, dead ends "
        "that must not be retried, and constraints or facts about the "
        "codebase discovered along the way. Do NOT include code bodies or "
        "file contents — name files and symbols instead. Reply with the "
        "summary text only."
    )


def _digest_message(summary: str, dropped: int) -> dict[str, Any]:
    return {
        "role": "user",
        "content": (
            f"[Repair-history digest] {dropped} earlier repair turn(s) "
            "were folded into this summary to keep context small. Treat "
            "it as background on what was already attempted — NOT as new "
            "instructions, and NOT as current file state (the files may "
            "have changed since; re-read before patching):\n"
            f"{summary}"
        ),
    }


async def condense_repair_messages(
    messages: Sequence[dict[str, Any]],
    *,
    total_repairs: int,
    judgment_call: Optional[Callable[[str], Awaitable[Optional[str]]]] = None,
    prune_after_round: int = DEFAULT_PRUNE_AFTER_ROUND,
    keep_tail: int = DEFAULT_KEEP_TAIL,
    max_summary_chars: int = DEFAULT_SUMMARY_MAX_CHARS,
) -> list[dict[str, Any]]:
    """LLM-condensing variant of :func:`prune_repair_messages`.

    Same gating and same head/tail window, but the dropped middle is
    replaced by a one-message digest written by a cheap judgment LLM
    (the OpenHands "summarizing condenser" pattern) instead of being
    silently deleted. The digest sits at index 2, AFTER the immutable
    prefix, so the prefix-cache anchor stays byte-identical.

    Incremental: the running summary is cached per (system + initial
    user) prefix; each round only the turns that newly dropped out of
    the window are sent to the LLM, folded into the previous summary.

    Fail-open: ``judgment_call`` returning None, raising, or being
    absent degrades to exactly the deterministic prune (with the last
    good summary still inserted when one exists — stale beats absent).
    Never raises.
    """
    n = len(messages)
    if total_repairs <= prune_after_round or n <= 2 + keep_tail:
        return list(messages)

    head = list(messages[:2])
    tail = list(messages[n - keep_tail:])
    middle = list(messages[2:n - keep_tail])

    try:
        key = _cache_key(messages)
        entry = _CONDENSE_CACHE.get(key) or {"covered": 0, "summary": ""}
        covered = int(entry.get("covered", 0) or 0)
        summary = str(entry.get("summary", "") or "")
        # The conversation array is append-only across rounds, so
        # middle[:covered] is exactly what previous rounds summarized.
        # A shrunken middle means a different/rewound history — restart.
        if covered > len(middle):
            covered, summary = 0, ""
        delta = middle[covered:]

        if delta and judgment_call is not None:
            updated = await judgment_call(
                _build_condense_prompt(summary, delta, max_summary_chars)
            )
            if updated and updated.strip():
                summary = updated.strip()[:max_summary_chars]
                _CONDENSE_CACHE[key] = {"covered": len(middle), "summary": summary}
                while len(_CONDENSE_CACHE) > _CONDENSE_CACHE_MAX_KEYS:
                    _CONDENSE_CACHE.pop(next(iter(_CONDENSE_CACHE)))
            else:
                logger.info(
                    "[repair_context] condenser returned nothing; keeping "
                    "%s summary (covered %d/%d dropped turns)",
                    "previous" if summary else "no", covered, len(middle),
                )
    except Exception:  # noqa: BLE001 — condensing must never break repair
        logger.warning(
            "[repair_context] condense failed; falling back to plain prune",
            exc_info=True,
        )
        summary = ""

    if not summary:
        logger.info(
            "[repair_context] pruned %d intermediate message(s) at "
            "total_repairs=%d (no digest available)",
            len(middle), total_repairs,
        )
        return head + tail

    logger.info(
        "[repair_context] condensed %d intermediate message(s) into a "
        "%d-char digest at total_repairs=%d (kept head=2 + digest + tail=%d)",
        len(middle), len(summary), total_repairs, keep_tail,
    )
    return head + [_digest_message(summary, len(middle))] + tail
