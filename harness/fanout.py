"""Graph-level multi-agent fan-out (#11).

Why this exists
===============
Today the harness's only fan-out is :mod:`harness.speculative`
(parallel patch variants) and :class:`harness.skills.SubAgentSkill`
(one sub-agent at a time). Many real tasks benefit from "ask N
agents the same question, or N variants of the question, and merge":

  - Discovery: one agent per sector (auth / persistence / observability
    / ...) running in parallel, then merge.
  - Test generation: one agent per modified file (or per language
    bucket) instead of one giant call.
  - Adversarial review: one finder + N independent skeptics that each
    try to refute the finding.

This module is the *primitive* for those patterns. The planner can
invoke a fan-out via the existing tool-DSL — see
:class:`SubAgentFanoutSkill` — and graph nodes (or skills) can call
:func:`run_parallel_agents` directly when they want to fan out work
deterministically.

Concurrency model
=================
``asyncio.Semaphore``-bounded fan-out. Default cap is 8 (matching the
``Workflow`` default in Claude Code), tunable per call. The shared
budget is *reserved* per task before the dispatch and refunded
afterwards based on actual spend, so two parallel agents never
double-spend the cap. When the shared budget would go negative the
remaining tasks are rejected with a structured error instead of
silently overspending.

Failure mode
============
A task that raises is captured in :class:`AgentResult.error`; we
never propagate exceptions out of :func:`run_parallel_agents` because
"task 3 of 8 crashed" should not kill the other seven. Callers filter
results by ``r.success`` to decide what to do with the survivors.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from harness.gateway import NodeRole

logger = logging.getLogger(__name__)


_DEFAULT_MAX_CONCURRENCY = 8
_DEFAULT_TIMEOUT_SECONDS = 180.0
_MIN_BUDGET_RESERVATION = 0.005  # per-task floor when budget hint absent


# ---------------------------------------------------------------------------
# 1. Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AgentSpec:
    """One parallel agent invocation.

    ``system_prompt`` + ``user_prompt`` are the bare minimum; provide
    ``messages`` directly when you need fine-grained control (e.g.
    multi-turn priming). Either ``messages`` is non-empty *or* the
    prompt fields produce a valid dispatch — :func:`_messages_for` does
    the resolution.

    ``role`` controls model routing through the gateway (planning vs
    patching vs repair). Default :attr:`NodeRole.PLANNING` because
    fan-out is overwhelmingly used during exploratory phases.

    ``budget_hint`` is the caller's *estimate* of what this agent will
    cost. The runner uses it for fairness when partitioning the shared
    budget across agents; the actual cost is taken from the gateway
    response. Leave 0 to fall back to the floor.
    """

    name: str
    system_prompt: str = ""
    user_prompt: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    role: NodeRole = NodeRole.PLANNING
    model_override: Optional[str] = None
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    budget_hint: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Outcome of a single fan-out task. Always returned in input order."""

    name: str
    success: bool
    content: str = ""
    cost_usd: float = 0.0
    elapsed_ms: int = 0
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 2. Core runner
# ---------------------------------------------------------------------------

def _messages_for(spec: AgentSpec) -> list[dict[str, Any]]:
    if spec.messages:
        return list(spec.messages)
    msgs: list[dict[str, Any]] = []
    if spec.system_prompt:
        msgs.append({"role": "system", "content": spec.system_prompt})
    msgs.append({"role": "user", "content": spec.user_prompt or ""})
    return msgs


async def run_parallel_agents(
    specs: list[AgentSpec],
    gateway: Any,
    *,
    budget_remaining_usd: float,
    max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
) -> tuple[list[AgentResult], float]:
    """Dispatch ``specs`` concurrently against ``gateway`` with a
    bounded semaphore + a shared budget reservation.

    Returns ``(results, new_budget)`` where ``results`` is in input
    order (gaps filled with ``success=False`` AgentResults so the
    caller can ``zip(specs, results)`` safely).
    """
    if not specs:
        return [], budget_remaining_usd
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    budget_lock = asyncio.Lock()
    state = {"budget": float(budget_remaining_usd)}

    async def _reserve(amount: float) -> bool:
        async with budget_lock:
            if state["budget"] >= amount:
                state["budget"] -= amount
                return True
            return False

    async def _refund(amount: float) -> None:
        async with budget_lock:
            state["budget"] += amount

    async def _run_one(spec: AgentSpec) -> AgentResult:
        # Reserve the agent's hinted cost (or the floor) so a swarm of
        # cheap agents can't starve the budget for in-flight expensive
        # ones; the actual cost is reconciled after dispatch.
        reservation = max(_MIN_BUDGET_RESERVATION, float(spec.budget_hint))
        async with semaphore:
            if not await _reserve(reservation):
                return AgentResult(
                    name=spec.name, success=False,
                    error=(
                        "shared budget exhausted before this agent could "
                        "reserve its hint"
                    ),
                )
            t0 = time.monotonic()
            # ``reconciled`` flips True once we have either dispatched
            # the spend (delta-refund inside the success path) or
            # refunded the reservation due to an error. The finally
            # block uses it as a fail-safe so cancellation, a missing
            # ``response.usage``, or any unexpected exception cannot
            # leak the reservation (audit §1.8).
            reconciled = False
            try:
                try:
                    # Pass a *per-call* cap that's the larger of the
                    # reservation and the shared budget so the gateway's
                    # internal pre-flight guard doesn't reject the call
                    # against a tiny per-task reservation. The shared
                    # accounting happens at our layer using the response's
                    # usage.cost_usd.
                    async with budget_lock:
                        per_call_cap = max(reservation, state["budget"] + reservation)
                    response, _ignored_dispatch_budget = await asyncio.wait_for(
                        gateway.dispatch(
                            messages=_messages_for(spec),
                            role=spec.role,
                            budget_remaining_usd=per_call_cap,
                            model_override=spec.model_override,
                        ),
                        timeout=spec.timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    await _refund(reservation)
                    reconciled = True
                    return AgentResult(
                        name=spec.name, success=False,
                        error=f"timeout after {spec.timeout_seconds}s",
                        elapsed_ms=int((time.monotonic() - t0) * 1000),
                    )
                except Exception as exc:  # noqa: BLE001
                    await _refund(reservation)
                    reconciled = True
                    logger.exception("[fanout] agent %s crashed", spec.name)
                    return AgentResult(
                        name=spec.name, success=False,
                        error=f"{type(exc).__name__}: {exc}",
                        elapsed_ms=int((time.monotonic() - t0) * 1000),
                    )
                # Defensive cost extraction — a non-numeric or missing
                # cost_usd previously raised here, jumped to the outer
                # CancelledError window, and leaked the reservation.
                try:
                    actual_cost = float(getattr(response.usage, "cost_usd", 0.0) or 0.0)
                except (TypeError, ValueError):
                    actual_cost = 0.0
                # Reconcile the reservation against the real spend. When the
                # agent came in under the reservation, refund the difference;
                # when it overshot, deduct the extra (may push the shared
                # budget negative — caller's responsibility to clamp).
                delta = reservation - actual_cost
                if delta != 0:
                    await _refund(delta)
                reconciled = True
                elapsed = int((time.monotonic() - t0) * 1000)
                return AgentResult(
                    name=spec.name, success=True,
                    content=response.content or "",
                    cost_usd=actual_cost,
                    elapsed_ms=elapsed,
                    metadata=dict(spec.metadata),
                )
            finally:
                # Fail-SAFE: if we exit without reconciling (CancelledError
                # bubbling, sibling-cancellation under gather, an unexpected
                # exception type), refund the full reservation so the shared
                # budget doesn't slowly bleed across fanouts. Audit §1.8.
                if not reconciled:
                    try:
                        await _refund(reservation)
                    except Exception:  # noqa: BLE001
                        pass

    raw = await asyncio.gather(
        *(_run_one(s) for s in specs), return_exceptions=False,
    )
    return list(raw), state["budget"]


# ---------------------------------------------------------------------------
# 3. Adversarial verifier pattern
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    is_real: bool
    confidence: float  # fraction of votes agreeing with finder
    votes: list[AgentResult]


async def run_with_verification(
    finder_spec: AgentSpec,
    *,
    gateway: Any,
    budget_remaining_usd: float,
    n_verifiers: int = 3,
    verifier_role: NodeRole = NodeRole.CODE_REVIEWER,
    refute_prompt_template: str = (
        "You are a skeptical reviewer. The following finding was emitted "
        "by another agent. Your job is to refute it. Default to "
        "refuted=true when uncertain. Respond with a one-line JSON "
        "object with keys `refuted` (bool) and `reason` (string).\n\n"
        "Finding:\n__FINDING__"
    ),
) -> tuple[AgentResult, Verdict, float]:
    """Run a finder agent, then dispatch N skeptics in parallel to try
    to refute it. Returns the finder's :class:`AgentResult`, the
    aggregated :class:`Verdict`, and the budget after all calls.

    A verdict is ``is_real=True`` iff the *majority* of verifiers
    accepted the finding (did not refute). ``confidence`` is the
    fraction of accepting votes.
    """
    finder_result, budget = await run_parallel_agents(
        [finder_spec], gateway, budget_remaining_usd=budget_remaining_usd,
    )
    finder = finder_result[0]
    if not finder.success or not finder.content.strip():
        return finder, Verdict(is_real=False, confidence=0.0, votes=[]), budget

    verifier_specs = [
        AgentSpec(
            name=f"verifier-{i}",
            system_prompt=(
                "You are reviewing a finding from another agent. "
                "Be adversarial: try to find a reason it might be wrong."
            ),
            user_prompt=refute_prompt_template.replace("__FINDING__", finder.content),
            role=verifier_role,
        )
        for i in range(max(1, n_verifiers))
    ]
    votes, budget = await run_parallel_agents(
        verifier_specs, gateway, budget_remaining_usd=budget,
    )
    accept = 0
    for v in votes:
        if not v.success:
            continue
        verdict_json = _parse_first_json(v.content)
        refuted = bool((verdict_json or {}).get("refuted", True))
        if not refuted:
            accept += 1
    total = sum(1 for v in votes if v.success) or 1
    confidence = accept / total
    is_real = confidence > 0.5
    return finder, Verdict(is_real=is_real, confidence=confidence, votes=votes), budget


def _parse_first_json(text: str) -> Optional[dict[str, Any]]:
    """Find and parse the first JSON object embedded in ``text``. Returns
    None if none decodes."""
    if not isinstance(text, str):
        return None
    start = text.find("{")
    while start != -1:
        # Find the matching close brace by walking balanced.
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


# ---------------------------------------------------------------------------
# 4. SubAgentFanoutSkill — text-DSL surface for the planner
# ---------------------------------------------------------------------------

def make_fanout_skill():
    """Construct the :class:`harness.skills.ToolSkill` the planner can
    invoke via ``<<<FANOUT_QUERY prompts='[...]'>>>``. Each entry in
    ``prompts`` is either a bare string (used as the user prompt) or
    a JSON object with ``{name, system_prompt, user_prompt}``.

    The skill returns ``{"results": [{"name": ..., "success": ...,
    "content": ..., "cost_usd": ..., "elapsed_ms": ...}, ...]}`` so the
    planner can decide how to merge.
    """
    from harness.skills import (
        SkillParameter, SkillSchema, SkillType, ToolSkill,
    )

    async def _call(**kwargs: Any) -> dict[str, Any]:
        from harness.graph import get_gateway
        gateway = get_gateway()
        if gateway is None:
            return {"error": "no gateway configured"}
        raw_prompts = kwargs.get("prompts") or "[]"
        if isinstance(raw_prompts, str):
            try:
                items = json.loads(raw_prompts)
            except json.JSONDecodeError as exc:
                return {"error": f"prompts must be JSON: {exc}"}
        elif isinstance(raw_prompts, list):
            items = raw_prompts
        else:
            return {"error": "prompts must be a JSON array or list"}
        if not isinstance(items, list) or not items:
            return {"error": "prompts must be a non-empty array"}

        specs: list[AgentSpec] = []
        for i, item in enumerate(items):
            if isinstance(item, str):
                specs.append(AgentSpec(
                    name=f"agent-{i}",
                    user_prompt=item,
                ))
            elif isinstance(item, dict):
                specs.append(AgentSpec(
                    name=str(item.get("name") or f"agent-{i}"),
                    system_prompt=str(item.get("system_prompt") or ""),
                    user_prompt=str(item.get("user_prompt") or ""),
                ))
            else:
                return {"error": f"prompts[{i}] must be string or object"}

        budget = float(kwargs.get("budget_usd", 1.00))
        max_concurrency = int(kwargs.get("max_concurrency", _DEFAULT_MAX_CONCURRENCY))
        results, remaining = await run_parallel_agents(
            specs, gateway,
            budget_remaining_usd=budget,
            max_concurrency=max_concurrency,
        )
        return {
            "results": [
                {
                    "name": r.name,
                    "success": r.success,
                    "content": r.content,
                    "cost_usd": r.cost_usd,
                    "elapsed_ms": r.elapsed_ms,
                    "error": r.error,
                }
                for r in results
            ],
            "budget_remaining_usd": remaining,
        }

    schema = SkillSchema(
        name="fanout_query",
        description=(
            "Dispatch multiple planner prompts in parallel and return "
            "every response. Use when you want to explore N independent "
            "questions concurrently (e.g. parallel discovery per sector, "
            "diverse implementation sketches). Input ``prompts`` is a "
            "JSON array of strings or objects with `name`, "
            "`system_prompt`, `user_prompt`."
        ),
        skill_type=SkillType.TOOL,
        parameters=[
            SkillParameter("prompts", "string", "JSON array of agent specs."),
            SkillParameter(
                "budget_usd", "number",
                "Shared USD budget across all agents (default 1.00).",
                required=False,
            ),
            SkillParameter(
                "max_concurrency", "integer",
                "Max agents running simultaneously (default 8).",
                required=False,
            ),
        ],
        returns_description=(
            "Object with `results` (per-agent outcomes) and "
            "`budget_remaining_usd`."
        ),
        tags=["fanout", "parallel"],
    )
    return ToolSkill(schema, fn=_call)


def register_fanout_skill() -> int:
    """Register the fan-out skill in the global SkillRegistry. Returns
    the number registered (1, or 0 on failure)."""
    try:
        from harness.skills import register
        register(make_fanout_skill())
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("[fanout] skill registration failed: %s", exc)
        return 0
