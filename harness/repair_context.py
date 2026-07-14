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
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

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
