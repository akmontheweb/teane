"""LLM-driven batch grouping for the per-batch verification pipeline.

After ``decomposition_node`` writes stories into ``.teane/state.db``, the
graph asks **one** question: how should these stories be grouped into
batches so the per-batch verification pipeline (compile / review / test /
security / regression) runs against coherent units of work?

The grouping is bounded by the existing dependency graph: a story may
only land in a batch whose number is >= the highest batch that contains
one of its dependencies. Within that constraint there's room for taste —
small projects can run as a single batch; large ones split by feature
seam, complexity, or planner intuition. That's where the LLM helps.

The module exposes:

- :func:`propose_or_fallback` — high-level entry point. Tries the LLM
  call; on dispatch failure, invalid JSON, or schema violation falls back
  to the deterministic topological-layer batching.
- :func:`deterministic_batches` — dependency-frontier batching. Each
  batch is the next layer of dep-ready stories.
- :func:`validate_batches` — pure validator: returns a list of human-
  readable errors. Empty list means the batch list is internally
  consistent against the story set.

This module does NOT touch the DB or graph state — it works on plain
dicts. The caller (``batch_planner_node`` after Phase E) is responsible
for ``story_state.start_batch()`` per batch and stamping
``current_batch_id`` onto the LangGraph state.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Hard cap matching the decomposition module — beyond this, the operator
# should re-decompose rather than try to grind through one huge session.
MAX_STORIES = 20

# Soft cap on stories-per-batch in the deterministic fallback. The LLM
# may suggest larger batches when the dep graph allows.
DETERMINISTIC_BATCH_SIZE = 5


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_batch_sizing_prompt(stories: list[dict[str, Any]]) -> str:
    """The contract the batch-sizing LLM must follow.

    The prompt is deliberately short. The LLM only sees keys, titles,
    and the dependency edges — it does not need acceptance criteria or
    scope_files to decide grouping."""
    lines = ["You are a delivery planner deciding how to group user stories"
             " into execution batches for a code-generation harness.\n"]
    lines.append(
        "Each batch will run the full verification pipeline (compile, "
        "review, test, security scan, regression) **once** against the "
        "code from all of its stories. Batching well means:\n"
        "- Stories whose code likely interacts go into the SAME batch "
        "(they need to compile together anyway).\n"
        "- Stories whose dep edges cross go into LATER batches than "
        "their dependencies (deps must be 'done' before consumers run).\n"
        "- A small project (<= 5 stories with no cross-dependencies) is "
        "ONE batch. Don't over-split.\n"
        "- A large project breaks into multiple batches by dependency "
        "layer and feature seam.\n"
    )
    lines.append("Stories to batch:\n")
    for s in stories:
        deps = ", ".join(s.get("depends_on") or []) or "-"
        lines.append(
            f"- {s['story_key']}: {s['title']}  (depends_on: {deps})"
        )
    lines.append("")
    lines.append(
        "Output STRICT JSON in this exact shape — no markdown, no code "
        "fence, no commentary:\n"
    )
    lines.append('{"batches": [{"batch_id": 1, "story_keys": ["STORY-001", "STORY-002"]}, ...]}')
    lines.append("")
    lines.append(
        "Constraints:\n"
        "- batch_id integers start at 1 and increment by 1.\n"
        "- Every story_key in the input MUST appear in exactly one batch.\n"
        "- A story's depends_on entries MUST appear in an earlier OR THE "
        "SAME batch (orphan deps that aren't in the input are tolerated).\n"
        "- When a dep is in the SAME batch as its dependent, list the "
        "dep FIRST in story_keys (topological order within the batch). "
        "The patcher walks story_keys in list order, so out-of-order "
        "entries would let the dependent patch before its dep lands.\n"
    )
    return "\n".join(lines)


# Re-exported from harness.decomposition so the two JSON-mode LLM
# callers share one implementation.
from harness.decomposition import strip_json_fence as _strip_json_fence  # noqa: E402


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_batches(
    stories: list[dict[str, Any]],
    batches: list[dict[str, Any]],
) -> list[str]:
    """Return a list of human-readable errors. Empty list = valid.

    Validates:
    - batches is a list of dicts
    - batch_ids start at 1 and increment by 1 with no gaps
    - every input story appears in exactly one batch (no missing, no dup)
    - no story's depends_on entry lives in a LATER batch
      (orphan deps — referenced but not in the input — are ignored)
    - Phase I: same-batch dependencies are ALLOWED, but the story_keys
      list order must put each dep BEFORE its dependents within the
      batch (so ``_next_story_in_batch`` picks them up in topological
      order). The patcher operates per-story sequentially within a
      batch, so an out-of-order list would let the dependent story
      patch before its dep's code lands.
    """
    errs: list[str] = []
    if not isinstance(batches, list):
        return [f"batches must be a list, got {type(batches).__name__}"]
    if not batches:
        if stories:
            return ["batches is empty but stories were provided"]
        return errs

    story_keys = {s["story_key"] for s in stories}
    deps_by_key: dict[str, list[str]] = {
        s["story_key"]: list(s.get("depends_on") or []) for s in stories
    }

    seen_in_batch: dict[str, int] = {}
    # Phase I: per-key index within its containing batch's story_keys
    # list so we can enforce topological order WITHIN a batch.
    position_in_batch: dict[str, int] = {}
    expected_id = 1
    for batch in batches:
        if not isinstance(batch, dict):
            errs.append(f"batch entry must be a dict, got {type(batch).__name__}")
            continue
        bid = batch.get("batch_id")
        if bid != expected_id:
            errs.append(f"batch_id sequence broken: expected {expected_id}, got {bid!r}")
        keys = batch.get("story_keys")
        if not isinstance(keys, list) or not keys:
            errs.append(f"batch {bid}: story_keys must be a non-empty list")
            expected_id += 1
            continue
        for idx, key in enumerate(keys):
            if not isinstance(key, str):
                errs.append(f"batch {bid}: story_key must be a string, got {key!r}")
                continue
            if key in seen_in_batch:
                errs.append(
                    f"story {key} appears in batch {seen_in_batch[key]} and "
                    f"batch {bid}"
                )
                continue
            if key not in story_keys:
                errs.append(
                    f"batch {bid} references unknown story_key {key!r}"
                )
                continue
            seen_in_batch[key] = bid
            position_in_batch[key] = idx
        expected_id += 1

    # All stories must be assigned somewhere.
    unassigned = sorted(story_keys - set(seen_in_batch))
    for key in unassigned:
        errs.append(f"story {key} is not assigned to any batch")

    # Dependency edges:
    # - cross-batch deps must point at an EARLIER batch
    # - same-batch deps must appear earlier in the story_keys list (topo)
    for key, deps in deps_by_key.items():
        if key not in seen_in_batch:
            continue
        my_batch = seen_in_batch[key]
        my_pos = position_in_batch.get(key, -1)
        for d in deps:
            if d not in seen_in_batch:
                # orphan / external dep — tolerated, matches v1 behavior
                continue
            dep_batch = seen_in_batch[d]
            if dep_batch > my_batch:
                errs.append(
                    f"story {key} (batch {my_batch}) depends on {d} "
                    f"(batch {dep_batch}) — dep must be in an earlier or "
                    f"same batch"
                )
            elif dep_batch == my_batch:
                dep_pos = position_in_batch.get(d, -1)
                if dep_pos >= my_pos:
                    errs.append(
                        f"story {key} (batch {my_batch}, pos {my_pos}) "
                        f"depends on {d} (same batch, pos {dep_pos}) — "
                        f"dep must come BEFORE its dependent in story_keys"
                    )
    return errs


# ---------------------------------------------------------------------------
# Deterministic fallback — dependency-frontier batching
# ---------------------------------------------------------------------------

def deterministic_batches(
    stories: list[dict[str, Any]],
    batch_size_hint: int = DETERMINISTIC_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Group stories by dependency-ready layer.

    Iteratively pull the next "ready" set — stories whose dependencies
    are all satisfied by earlier-emitted batches (or are orphans against
    the input) — and emit them as a batch. Within a layer, slice into
    chunks of at most ``batch_size_hint`` to keep batches under a sane
    size for the verification pipeline.

    Returns ``[{batch_id, story_keys}, ...]`` with batch_ids starting at 1.
    Returns ``[]`` for an empty input.
    Returns a single batch if every story is independent and the total
    fits in one slice. Tolerates orphan dependencies (deps that reference
    a story_key not in the input) — they're treated as satisfied.
    """
    if not stories:
        return []
    if batch_size_hint < 1:
        batch_size_hint = 1

    pending: dict[str, dict[str, Any]] = {s["story_key"]: s for s in stories}
    deps_by_key: dict[str, set[str]] = {
        k: set(v.get("depends_on") or []) & pending.keys()
        for k, v in pending.items()
    }
    emitted: set[str] = set()
    batches: list[dict[str, Any]] = []
    batch_id = 1

    while pending:
        ready = [
            key for key, deps in deps_by_key.items()
            if key in pending and deps <= emitted
        ]
        ready.sort(key=lambda k: int(k.split("-")[-1]) if k.split("-")[-1].isdigit() else 0)
        if not ready:
            # Cycle or unresolvable dep — emit the rest in one batch so we
            # don't infinite-loop. The validator will catch it.
            ready = sorted(pending.keys())
            logger.warning(
                "[batch_sizing] dependency cycle detected among %s; emitting "
                "remainder as one batch.", ready
            )
        for i in range(0, len(ready), batch_size_hint):
            slice_keys = ready[i:i + batch_size_hint]
            # Phase I: topo-sort within the slice. The dep-frontier
            # algorithm above already guarantees no intra-slice deps
            # (every story in ``ready`` has its deps in ``emitted``,
            # not in the slice), so this is a no-op for the current
            # algorithm — but it's the explicit invariant the
            # ``_next_story_in_batch`` patcher relies on, and a future
            # batcher (LLM-driven post-validation, manual operator
            # input, etc.) may produce slices that do have intra-batch
            # deps. Defensive ordering keeps that contract intact.
            slice_keys = _topo_sort_within_batch(slice_keys, deps_by_key)
            batches.append({"batch_id": batch_id, "story_keys": slice_keys})
            batch_id += 1
            for k in slice_keys:
                emitted.add(k)
                pending.pop(k, None)
    return batches


def _topo_sort_within_batch(
    keys: list[str], deps_by_key: dict[str, set[str]],
) -> list[str]:
    """Stable topological sort over a single batch's story_keys.

    Edges considered are intra-batch only — ``deps_by_key[k] & set(keys)``.
    Ties are broken by the input order so a slice already in topo order
    is returned unchanged. On an unresolvable cycle (shouldn't happen
    for a validated batch but possible from a malformed LLM proposal),
    falls back to the input order and lets ``validate_batches`` flag
    the issue."""
    key_set = set(keys)
    intra_deps: dict[str, set[str]] = {
        k: set(deps_by_key.get(k, set())) & key_set for k in keys
    }
    placed: set[str] = set()
    out: list[str] = []
    remaining = list(keys)
    safety_passes = len(keys) + 1  # cycle guard
    while remaining and safety_passes > 0:
        safety_passes -= 1
        progress = False
        next_remaining: list[str] = []
        for k in remaining:
            if intra_deps[k] <= placed:
                out.append(k)
                placed.add(k)
                progress = True
            else:
                next_remaining.append(k)
        remaining = next_remaining
        if not progress:
            break
    if remaining:
        # Cycle / unresolvable — preserve input order for the tail.
        out.extend(remaining)
    return out


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

async def _llm_propose(
    stories: list[dict[str, Any]],
    gateway: Any,
    budget_remaining_usd: float,
    system_message: Optional[dict[str, Any]] = None,
) -> tuple[Optional[list[dict[str, Any]]], float]:
    """Dispatch the batch-sizing prompt to the planning LLM.

    Returns ``(batches_or_None, new_budget)``. ``None`` on dispatch /
    JSON / schema failure so the caller can fall back. The budget is
    decremented by the call regardless of success."""
    from harness.gateway import NodeRole

    prompt = build_batch_sizing_prompt(stories)
    messages: list[dict[str, Any]] = []
    if system_message:
        messages.append(system_message)
    messages.append({"role": "user", "content": prompt})

    try:
        response, budget_remaining_usd = await gateway.dispatch(
            messages=messages,
            role=NodeRole.PLANNING,
            budget_remaining_usd=budget_remaining_usd,
            cache_family="planning:batch_sizing",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[batch_sizing] gateway dispatch failed: %s", exc)
        return None, budget_remaining_usd

    raw = _strip_json_fence(getattr(response, "content", "") or "")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("[batch_sizing] invalid JSON from LLM: %s", exc)
        return None, budget_remaining_usd

    batches = data.get("batches") if isinstance(data, dict) else None
    if not isinstance(batches, list):
        logger.warning("[batch_sizing] payload missing 'batches' list")
        return None, budget_remaining_usd
    return batches, budget_remaining_usd


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

async def propose_or_fallback(
    stories: list[dict[str, Any]],
    gateway: Any,
    budget_remaining_usd: float,
    *,
    batch_size_hint: int = DETERMINISTIC_BATCH_SIZE,
    system_message: Optional[dict[str, Any]] = None,
) -> tuple[list[dict[str, Any]], bool, float]:
    """Return ``(batches, used_llm, new_budget)``.

    Tries the LLM call first; on any failure (no gateway, dispatch
    error, invalid JSON, schema violation) falls back to the
    deterministic topological-layer batcher. ``used_llm`` is True only
    when the returned batches came from the LLM and passed validation.

    Hard caps:
    - Empty input → ``([], False, budget)``.
    - More than :data:`MAX_STORIES` stories → falls back deterministically
      without calling the LLM (matches the decomposition cap).
    """
    if not stories:
        return [], False, budget_remaining_usd

    if len(stories) > MAX_STORIES:
        logger.info(
            "[batch_sizing] %d stories exceeds MAX_STORIES=%d — using "
            "deterministic batching to avoid an oversize LLM call.",
            len(stories), MAX_STORIES,
        )
        return (
            deterministic_batches(stories, batch_size_hint=batch_size_hint),
            False,
            budget_remaining_usd,
        )

    if gateway is None or budget_remaining_usd <= 0.0:
        logger.info(
            "[batch_sizing] no gateway or zero budget — deterministic batching."
        )
        return (
            deterministic_batches(stories, batch_size_hint=batch_size_hint),
            False,
            budget_remaining_usd,
        )

    proposed, budget_remaining_usd = await _llm_propose(
        stories, gateway, budget_remaining_usd, system_message
    )
    if proposed is None:
        return (
            deterministic_batches(stories, batch_size_hint=batch_size_hint),
            False,
            budget_remaining_usd,
        )

    errs = validate_batches(stories, proposed)
    if errs:
        for e in errs[:5]:
            logger.warning("[batch_sizing] LLM proposal rejected: %s", e)
        return (
            deterministic_batches(stories, batch_size_hint=batch_size_hint),
            False,
            budget_remaining_usd,
        )

    # Coerce the LLM's shape into a stable list[dict] with sorted keys
    # within each batch so downstream is deterministic.
    cleaned: list[dict[str, Any]] = []
    for batch in proposed:
        cleaned.append({
            "batch_id": int(batch["batch_id"]),
            "story_keys": list(batch["story_keys"]),
        })
    return cleaned, True, budget_remaining_usd
