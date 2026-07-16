"""
Speculative Patch Branching — Multi-Variant Compilation.

This module implements:
    - speculate_node: Replaces single-patch flow with 3 parallel variants.
      Each variant gets an isolated git worktree, is compiled simultaneously,
      and the first passing variant is merged back. Reduces debugging cycles
      and increases first-pass build success rates.

    - Selector strategies: "first_success", "fewest_changes", "all_pass".

Integration:
    - Placed as speculative_node between patching_node and lintgate_node.
    - If enabled, patching_node routes to speculative_node instead of lintgate.
    - Falls back to sequential single-patch flow if all variants fail.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from harness import _platform
from harness.gateway import NodeRole
from harness.patcher import process_llm_patch_output, PatchResult
from harness.sandbox import BUILDER_IMAGE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Types
# ---------------------------------------------------------------------------

@dataclass
class VariantResult:
    """Result of a single speculative variant."""
    index: int
    variant_id: str
    worktree_path: str
    llm_response: Optional[Any] = None
    patch_results: list[PatchResult] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    exit_code: int = -1
    raw_output: str = ""
    timed_out: bool = False
    error: str = ""

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.error

    @property
    def total_lines_changed(self) -> int:
        return sum(r.lines_changed for r in self.patch_results if r.success)


@dataclass
class SpeculativeResult:
    """Aggregate result of speculative branching."""
    total_variants: int = 0
    passed_variants: int = 0
    winner_index: int = -1
    variant_results: list[VariantResult] = field(default_factory=list)
    strategy: str = "first_pass"
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# 2. Strategy enums + config (rebuild, #12)
# ---------------------------------------------------------------------------

# Enums-as-string-constants. We use plain strings (not Enum subclasses) so
# they round-trip through JSON / state-dict serialisation without
# additional type adapters; the validation in :func:`SpeculativeConfig.
# normalize` covers typos.

DIVERSITY_TEMPERATURE = "temperature"   # all variants same model, varied temp
DIVERSITY_PROMPT      = "prompt"        # all variants same model, varied system prompt
DIVERSITY_MODEL       = "model"         # different models per variant
DIVERSITY_MIXED       = "mixed"         # different models AND different prompt styles
DIVERSITY_MODES = frozenset({
    DIVERSITY_TEMPERATURE, DIVERSITY_PROMPT, DIVERSITY_MODEL, DIVERSITY_MIXED,
})

COST_EQUAL = "equal_cost"                          # current behaviour
COST_CHEAP_FIRST_SEQUENTIAL = "cheap_first_sequential"  # cheap one-by-one, expensive last
COST_CHEAP_PARALLEL_THEN_EXPENSIVE = "cheap_parallel_then_expensive"
COST_ALL_CHEAP = "all_cheap"
# Multi-LLM gradient — variants consume ``variant_models`` in declared
# cost order (variant 0 → variant_models[0], ..., variant N → variant_models[N]).
# Unlike the two-tier strategies above, this honours the full list as a
# monotonic cheap→expensive sequence. The last position is marked
# expensive so the runner can prioritise / report cost the same way as
# the cheap-first strategies.
COST_GRADIENT_LOW_TO_HIGH = "gradient_low_to_high"
COST_STRATEGIES = frozenset({
    COST_EQUAL,
    COST_CHEAP_FIRST_SEQUENTIAL,
    COST_CHEAP_PARALLEL_THEN_EXPENSIVE,
    COST_ALL_CHEAP,
    COST_GRADIENT_LOW_TO_HIGH,
})

SELECT_FIRST_PASS = "first_pass"          # canonical name
SELECT_FIRST_SUCCESS = "first_success"    # legacy alias for first_pass
SELECT_FEWEST_CHANGES = "fewest_changes"
SELECT_VOTED = "voted"                    # adversarial judges (uses fanout #11)
SELECT_ALL_PASS = "all_pass"
SELECTION_STRATEGIES = frozenset({
    SELECT_FIRST_PASS, SELECT_FIRST_SUCCESS, SELECT_FEWEST_CHANGES,
    SELECT_VOTED, SELECT_ALL_PASS,
})

SALVAGE_NONE = "none"      # fall back to sequential repair against untouched workspace
SALVAGE_FEWEST_ERRORS = "fewest_errors"
SALVAGE_VOTED_PARTIAL = "voted_partial"
SALVAGE_MERGE = "merge"    # legacy behaviour (often produces incoherent workspaces)
SALVAGE_STRATEGIES = frozenset({
    SALVAGE_NONE, SALVAGE_FEWEST_ERRORS, SALVAGE_VOTED_PARTIAL, SALVAGE_MERGE,
})

TRIGGER_ALWAYS = "always"
TRIGGER_FIRST_ATTEMPT_ONLY = "first_attempt_only"
TRIGGER_AFTER_N_REPAIR_FAILURES = "after_n_repair_failures"
TRIGGER_MANUAL = "manual"  # only when state["force_speculative"] is truthy
TRIGGERS = frozenset({
    TRIGGER_ALWAYS, TRIGGER_FIRST_ATTEMPT_ONLY,
    TRIGGER_AFTER_N_REPAIR_FAILURES, TRIGGER_MANUAL,
})


@dataclass
class VotingConfig:
    """Settings for ``selection_strategy=voted`` (adversarial judges)."""

    n_judges: int = 3
    judge_role: str = "code_reviewer"  # one of the NodeRole values


@dataclass
class SpeculativeConfig:
    """Full strategy surface for speculative execution (#12 rebuild).

    Each axis (``diversity_mode``, ``cost_strategy``,
    ``selection_strategy``, ``salvage_strategy``, ``trigger``) is
    independently selectable from config. The defaults below were chosen
    to deliver positive ROI for typical workloads:

    - ``trigger=after_n_repair_failures`` (threshold 2): sequential is
      cheaper and more focused on the happy path. Speculative is held
      back for the moment when sequential repair is stuck — that's when
      diversity actually buys recovery.
    - ``cost_strategy=cheap_first_sequential``: try a cheap model first;
      only spawn the expensive baseline when cheap fails. Expected cost
      is ~1.1× sequential rather than the old 3×.
    - ``diversity_mode=model``: use *different* models for variants when
      we do fan out. Different architectures genuinely fail differently;
      temperature noise on one model does not.
    - ``selection_strategy=first_pass``: take the first variant that
      compiles cleanly. Cheap and effective.
    - ``salvage_strategy=none``: when all variants fail, throw the
      worktrees away and fall back to sequential repair against the
      untouched workspace. Pareto-better than the legacy ``merge`` path
      which often produced incoherent workspaces.
    """

    enabled: bool = False  # opt-in; legacy default preserved
    trigger: str = TRIGGER_AFTER_N_REPAIR_FAILURES
    n_repair_failures_threshold: int = 2
    diversity_mode: str = DIVERSITY_MODEL
    cost_strategy: str = COST_CHEAP_FIRST_SEQUENTIAL
    selection_strategy: str = SELECT_FIRST_PASS
    salvage_strategy: str = SALVAGE_NONE
    num_variants: int = 3
    max_concurrency: int = 3
    temperature: float = 0.3
    # Repair-level fanout: when the sequential repair loop has burned
    # ``repair_fanout_after_rounds`` consecutive no-progress rounds,
    # sample ``repair_fanout_variants`` repair responses (one dispatch
    # each, distinct strategy framings), test-compile each in a worktree
    # seeded with the CURRENT dirty workspace, and hand the best response
    # back to repair_node's normal apply path. Off by default — N extra
    # dispatches + N sandbox compiles per engagement.
    repair_fanout: bool = False
    repair_fanout_variants: int = 3
    repair_fanout_after_rounds: int = 2
    # Diversity vectors — used when the mode references them.
    variant_models: list[str] = field(default_factory=list)
    variant_prompt_styles: list[str] = field(default_factory=list)
    expensive_model: str = ""  # primary model for cheap_first / cheap_parallel
    cheap_model: str = ""      # fallback / cheap variants
    voting: VotingConfig = field(default_factory=VotingConfig)
    # Resolved via factory so the platform check happens at instance-creation
    # time, not import time. POSIX → "/tmp/.harness/speculative" (unchanged);
    # Windows → "%TEMP%\.harness\speculative" (avoids the FileNotFoundError
    # that os.makedirs would raise on a literal "/tmp" path).
    worktree_base_dir: str = field(
        default_factory=lambda: _platform.harness_temp_dir(".harness/speculative")
    )

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "SpeculativeConfig":
        raw = dict(state.get("speculative_config") or {})
        raw = _upgrade_legacy_config(raw)
        return cls.normalize(raw)

    @classmethod
    def normalize(cls, raw: dict[str, Any]) -> "SpeculativeConfig":
        """Build a :class:`SpeculativeConfig` from a raw dict, clamping
        out-of-range values and replacing unknown enum values with the
        documented defaults (with a warning).
        """
        def _pick(value: Any, allowed: frozenset[str], default: str) -> str:
            if not isinstance(value, str) or value not in allowed:
                if value not in (None, ""):
                    logger.warning(
                        "[speculative] unknown value %r; using %r.",
                        value, default,
                    )
                return default
            return value

        def _clamp_int(value: Any, default: int, lo: int, hi: int) -> int:
            try:
                v = int(value) if value is not None else default
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        def _clamp_float(value: Any, default: float, lo: float, hi: float) -> float:
            try:
                v = float(value) if value is not None else default
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        voting_raw = raw.get("voting") or {}
        voting = VotingConfig(
            n_judges=_clamp_int(voting_raw.get("n_judges"), 3, 1, 7),
            judge_role=str(voting_raw.get("judge_role", "code_reviewer") or "code_reviewer"),
        )
        return cls(
            enabled=bool(raw.get("enabled", False)),
            trigger=_pick(raw.get("trigger"), TRIGGERS, TRIGGER_AFTER_N_REPAIR_FAILURES),
            n_repair_failures_threshold=_clamp_int(
                raw.get("n_repair_failures_threshold"), 2, 1, 10,
            ),
            diversity_mode=_pick(raw.get("diversity_mode"), DIVERSITY_MODES, DIVERSITY_MODEL),
            cost_strategy=_pick(raw.get("cost_strategy"), COST_STRATEGIES, COST_CHEAP_FIRST_SEQUENTIAL),
            selection_strategy=_pick(
                raw.get("selection_strategy"), SELECTION_STRATEGIES, SELECT_FIRST_PASS,
            ),
            salvage_strategy=_pick(raw.get("salvage_strategy"), SALVAGE_STRATEGIES, SALVAGE_NONE),
            num_variants=_clamp_int(raw.get("num_variants"), 3, 1, 10),
            max_concurrency=_clamp_int(
                raw.get("max_concurrency", raw.get("num_variants")), 3, 1, 10,
            ),
            temperature=_clamp_float(raw.get("temperature"), 0.3, 0.0, 1.5),
            variant_models=[
                str(m) for m in (raw.get("variant_models") or [])
                if isinstance(m, str) and m
            ],
            variant_prompt_styles=[
                str(s) for s in (raw.get("variant_prompt_styles") or [])
                if isinstance(s, str) and s
            ],
            expensive_model=str(raw.get("expensive_model") or ""),
            cheap_model=str(raw.get("cheap_model") or ""),
            voting=voting,
            worktree_base_dir=str(raw.get("worktree_base_dir") or "/tmp/.harness/speculative"),
            repair_fanout=bool(raw.get("repair_fanout", False)),
            repair_fanout_variants=_clamp_int(
                raw.get("repair_fanout_variants"), 3, 2, 5,
            ),
            repair_fanout_after_rounds=_clamp_int(
                raw.get("repair_fanout_after_rounds"), 2, 1, 10,
            ),
        )


_LEGACY_KEY_ALIASES = {
    "first_success": SELECT_FIRST_PASS,  # selection_strategy alias
}


def _upgrade_legacy_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Map old-shape speculative config to the new schema.

    The legacy shape was::

        {"enabled": bool, "num_variants": int, "temperature": float,
         "selection_strategy": "first_success" | "fewest_changes" |
                                "all_pass",
         "worktree_base_dir": str}

    When the operator hasn't migrated to the new keys, we infer the
    backwards-compatible defaults so their flow keeps working:

      - ``diversity_mode=temperature``  (matches old "all variants, vary
        temp" behaviour)
      - ``cost_strategy=equal_cost``    (all variants used same model)
      - ``salvage_strategy=merge``      (legacy merge-on-fail path)
      - ``trigger=first_attempt_only``  (engages on the first patching
        call, mirroring the old wiring)

    Logs a one-time deprecation warning when any legacy-only key is
    detected so operators see they should migrate.
    """
    if not isinstance(raw, dict):
        return {}
    out = dict(raw)
    has_legacy_only = (
        "trigger" not in raw
        and "diversity_mode" not in raw
        and "cost_strategy" not in raw
        and "salvage_strategy" not in raw
        and raw.get("enabled", False)
    )
    if "selection_strategy" in out and out["selection_strategy"] in _LEGACY_KEY_ALIASES:
        out["selection_strategy"] = _LEGACY_KEY_ALIASES[out["selection_strategy"]]
    if has_legacy_only:
        out.setdefault("diversity_mode", DIVERSITY_TEMPERATURE)
        out.setdefault("cost_strategy", COST_EQUAL)
        out.setdefault("salvage_strategy", SALVAGE_MERGE)
        out.setdefault("trigger", TRIGGER_FIRST_ATTEMPT_ONLY)
        logger.warning(
            "[speculative] legacy config detected; mapping to "
            "diversity_mode=%r cost_strategy=%r salvage_strategy=%r "
            "trigger=%r. Migrate to the new keys to silence this "
            "warning — see config/config.json.",
            DIVERSITY_TEMPERATURE, COST_EQUAL, SALVAGE_MERGE,
            TRIGGER_FIRST_ATTEMPT_ONLY,
        )
    return out


# ---------------------------------------------------------------------------
# 3. Trigger evaluation
# ---------------------------------------------------------------------------

def _trigger_met(cfg: "SpeculativeConfig", state: dict[str, Any]) -> tuple[bool, str]:
    """Return ``(engage, reason)``.

    The reason is a human-readable string used in the no-op log so the
    operator can tell why a speculative round was skipped.
    """
    loop = state.get("loop_counter") or {}
    if cfg.trigger == TRIGGER_ALWAYS:
        return True, "trigger=always"
    if cfg.trigger == TRIGGER_FIRST_ATTEMPT_ONLY:
        patching_count = int(loop.get("patching", 0) or 0)
        if patching_count <= 1:
            return True, "first patching attempt"
        return False, f"patching_count={patching_count} > 1"
    if cfg.trigger == TRIGGER_AFTER_N_REPAIR_FAILURES:
        repair_count = int(loop.get("repair", 0) or 0)
        if repair_count >= cfg.n_repair_failures_threshold:
            return True, f"repair_count={repair_count} >= {cfg.n_repair_failures_threshold}"
        return False, (
            f"repair_count={repair_count} < threshold "
            f"{cfg.n_repair_failures_threshold}"
        )
    if cfg.trigger == TRIGGER_MANUAL:
        if state.get("force_speculative"):
            return True, "state.force_speculative=true"
        return False, "manual trigger; state.force_speculative not set"
    # Defensive fallback (should not happen — normalize() catches typos).
    return False, f"unknown trigger {cfg.trigger!r}"


# ---------------------------------------------------------------------------
# 4. Variant spec builder
# ---------------------------------------------------------------------------

@dataclass
class _VariantSpec:
    """Per-variant LLM dispatch spec — internal, fed to fanout or
    serially executed depending on the cost strategy."""

    index: int
    model_override: Optional[str]
    temperature: float
    system_prompt_suffix: str  # extra style hint appended to messages[0]
    is_expensive: bool = False  # used by cheap_first / cheap_parallel strategies


_PROMPT_STYLE_LIBRARY = {
    "minimal-diff": (
        "Style override: produce the smallest possible diff that solves the "
        "task. Avoid speculative refactors, comment churn, or imports you "
        "don't need."
    ),
    "balanced": (
        "Style override: balance correctness and minimal change. Refactor "
        "only what's required by the task."
    ),
    "thorough": (
        "Style override: prefer thorough, defensive code with explicit error "
        "handling and tests where appropriate."
    ),
    "conservative": (
        "Style override: prefer adding small wrappers over modifying existing "
        "behaviour. When in doubt, leave existing code untouched."
    ),
    "bold": (
        "Style override: don't hesitate to refactor when the current "
        "structure is the root cause of the problem."
    ),
}


def _build_variant_specs(cfg: "SpeculativeConfig") -> list[_VariantSpec]:
    """Build the per-variant dispatch specs based on diversity_mode +
    cost_strategy. Returns a list of length ``cfg.num_variants``.
    """
    specs: list[_VariantSpec] = []

    # Resolve the model lists used by each strategy.
    cheap = cfg.cheap_model or ""
    expensive = cfg.expensive_model or ""
    models_pool = list(cfg.variant_models)
    prompts_pool = list(cfg.variant_prompt_styles) or list(_PROMPT_STYLE_LIBRARY.keys())

    for i in range(cfg.num_variants):
        # --- Diversity axis ---
        if cfg.diversity_mode == DIVERSITY_TEMPERATURE:
            # Same model (gateway default), spread temperatures.
            temp = max(0.0, min(1.5, cfg.temperature + i * 0.15))
            model = None
            style = ""
        elif cfg.diversity_mode == DIVERSITY_PROMPT:
            model = None
            temp = cfg.temperature
            style = prompts_pool[i % len(prompts_pool)] if prompts_pool else ""
        elif cfg.diversity_mode == DIVERSITY_MODEL:
            model = models_pool[i % len(models_pool)] if models_pool else None
            temp = cfg.temperature
            style = ""
        else:  # DIVERSITY_MIXED
            model = models_pool[i % len(models_pool)] if models_pool else None
            temp = max(0.0, min(1.5, cfg.temperature + (i // max(1, len(models_pool))) * 0.15))
            style = prompts_pool[i % len(prompts_pool)] if prompts_pool else ""

        # --- Cost axis: override model assignment when strategy demands ---
        is_expensive = False
        if cfg.cost_strategy == COST_EQUAL:
            pass  # diversity already chose the model
        elif cfg.cost_strategy == COST_ALL_CHEAP and cheap:
            model = cheap
        elif cfg.cost_strategy == COST_CHEAP_FIRST_SEQUENTIAL:
            # First N-1 cheap; last expensive. Sequential execution
            # below short-circuits as soon as one passes.
            if i < cfg.num_variants - 1 and cheap:
                model = cheap
            elif expensive:
                model = expensive
                is_expensive = True
        elif cfg.cost_strategy == COST_CHEAP_PARALLEL_THEN_EXPENSIVE:
            # All-but-one cheap, one expensive. Parallel; expensive marked
            # so the runner can prioritise / report cost.
            if i == 0 and expensive:
                model = expensive
                is_expensive = True
            elif cheap:
                model = cheap
        elif cfg.cost_strategy == COST_GRADIENT_LOW_TO_HIGH:
            # Multi-LLM gradient — variants consume ``variant_models`` in
            # declared order without modulo cycling, so the sequence stays
            # monotonic cheap → expensive. Falls back to the diversity
            # choice when the gradient runs out (i >= len(variant_models)).
            if models_pool and i < len(models_pool):
                model = models_pool[i]
                is_expensive = (i == len(models_pool) - 1)

        # --- Resolve style suffix from the library when name is a key ---
        style_suffix = _PROMPT_STYLE_LIBRARY.get(style, style)

        specs.append(_VariantSpec(
            index=i,
            model_override=model,
            temperature=temp,
            system_prompt_suffix=style_suffix,
            is_expensive=is_expensive,
        ))
    return specs


def _seed_messages_with_style(
    messages: list[dict[str, Any]], spec: "_VariantSpec",
) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` with the variant's style suffix
    appended to the first system message. Used by diversity_mode=prompt
    / mixed to actually steer the LLM through a *different* angle than
    its peers — without this the spec's prompt_style does nothing.
    """
    if not spec.system_prompt_suffix:
        return list(messages)
    out = [dict(m) for m in messages]
    inserted = False
    for m in out:
        if m.get("role") == "system":
            base = str(m.get("content") or "")
            m["content"] = (base + "\n\n" + spec.system_prompt_suffix).strip()
            inserted = True
            break
    if not inserted:
        out.insert(0, {"role": "system", "content": spec.system_prompt_suffix})
    return out


# ---------------------------------------------------------------------------
# 5. Speculative Node (entry point — preserved name + state contract)
# ---------------------------------------------------------------------------

async def speculate_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Speculative execution node: generates N variants, compiles them in parallel,
    and selects the best passing variant.

    Workflow:
        1. Call the LLM N times with temperature > 0 for diverse solutions
        2. Create isolated git worktrees for each variant
        3. Apply patches to each worktree
        4. Run lintgate + compiler on each worktree in parallel
        5. Select the first passing variant (or best by strategy)
        6. Copy winning files back to main workspace
        7. Clean up temporary worktrees

    Configuration via the `speculative` section of config/config.json
    (threaded onto state via cli.py → run_graph(speculative_config=...)).
    Strict validation lives in cli.py:validate_config_strict; ranges:
    num_variants ∈ [1, 10], temperature ∈ [0.0, 1.5], selection_strategy
    ∈ {first_success, fewest_changes, all_pass}.
        {
          "speculative": {
            "enabled": true,
            "num_variants": 3,
            "temperature": 0.3,
            "selection_strategy": "first_success",
            "worktree_base_dir": "/tmp/.harness/speculative"
          }
        }

    Returns:
        State update dict with winning variant data.
    """
    import time as time_module

    # --- Config (#12 rebuild) ---
    cfg = SpeculativeConfig.from_state(state)
    if not cfg.enabled:
        logger.info(
            "[speculative] Disabled (speculative.enabled is false). "
            "Passing through to standard patching flow."
        )
        return _fallback_result()
    # Trigger gate. Even when enabled, we only engage on the workloads the
    # operator opted into (e.g. trigger=after_n_repair_failures means we
    # stay out of the way until sequential repair has stalled).
    engage, reason = _trigger_met(cfg, state)
    if not engage:
        logger.info(
            "[speculative] Trigger %r not met (%s). Passing through.",
            cfg.trigger, reason,
        )
        return _fallback_result()
    num_variants = cfg.num_variants
    temperature = cfg.temperature
    strategy = cfg.selection_strategy
    worktree_base = cfg.worktree_base_dir
    variant_specs = _build_variant_specs(cfg)

    workspace_path = state.get("workspace_path", os.getcwd())
    build_command = state.get("build_command", "make build")
    sandbox_config = dict(state.get("sandbox_config", {}) or {})
    allow_network = state.get("allow_network", False)
    messages = state.get("messages", [])
    budget = state.get("budget_remaining_usd", 2.00)

    # Late-bind the build command + toolchain image the same way
    # ``compiler_node`` does. Without this, speculative variants run with
    # the historical ``make build`` default in ``ubuntu:22.04`` against
    # workspaces the LLM just populated (e.g. Python sources with no
    # Makefile), every variant exits 127, and the whole speculative round
    # is guaranteed budget waste. Keeping this inline rather than
    # importing ``compiler_node``'s block verbatim because we don't need
    # the loop-counter / token-tracker plumbing — just the resolved
    # build_command + sandbox_config + allow_network.
    # is_greenfield is recovered from flow so an LLM-scaffolded Makefile
    # can't hijack the build command away from the per-stack baseline.
    is_greenfield_spec = bool(state.get("flow") == "build")
    adapted_build_cmd: Optional[str] = None
    if build_command.strip() == "make build" and not any(
        os.path.exists(os.path.join(workspace_path, name))
        for name in ("Makefile", "makefile", "GNUmakefile")
    ):
        try:
            from harness.cli import _detect_default_build_command
            late = _detect_default_build_command(
                workspace_path, is_greenfield=is_greenfield_spec,
            )
            if late and late != "make build":
                logger.info(
                    "[speculative] Workspace has no Makefile; adapting build command "
                    "from default 'make build' to detected: %s", late,
                )
                adapted_build_cmd = late
                build_command = late
        except Exception as exc:  # noqa: BLE001
            logger.debug("[speculative] build-command late-bind failed: %s", exc)
    try:
        from harness.graph import _apply_toolchain_adaptation
        (
            sandbox_config,
            allow_network,
            image_was_adapted,
            network_was_adapted,
            _ro_was_adapted,
        ) = _apply_toolchain_adaptation(
            build_command,
            sandbox_config,
            allow_network,
            command_is_adapter_synthesised=adapted_build_cmd is not None,
        )
        if image_was_adapted:
            logger.info(
                "[speculative] Adapting sandbox docker_image to %r to match toolchain implied by: %s",
                sandbox_config.get("docker_image"), build_command,
            )
        if network_was_adapted:
            logger.info(
                "[speculative] Auto-enabling network for adapter-synthesised install step: %s",
                build_command,
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("[speculative] toolchain adaptation failed: %s", exc)

    start_time = time_module.monotonic()

    logger.info("[speculative] Starting speculative branching: %d variants, temp=%.2f, strategy=%s",
                 num_variants, temperature, strategy)

    # --- Get gateway ---
    from harness.graph import get_gateway
    gateway = get_gateway()
    if gateway is None:
        logger.error("[speculative] No gateway configured. Falling back to single patch.")
        return _fallback_result()

    # --- Step 1: Generate N variants ---
    # The cost strategy decides parallel vs sequential. cheap_first_sequential
    # dispatches one variant at a time so that, downstream, we can short-circuit
    # the remaining LLM calls once a cheap variant proves it can compile.
    variant_responses: list[Any] = []
    if cfg.cost_strategy == COST_CHEAP_FIRST_SEQUENTIAL:
        for spec in variant_specs:
            try:
                seeded_messages = _seed_messages_with_style(messages, spec)
                response, _ = await gateway.dispatch(
                    messages=seeded_messages,
                    role=NodeRole.PATCHING,
                    budget_remaining_usd=budget,
                    temperature=spec.temperature,
                    model_override=spec.model_override,
                )
                variant_responses.append(response)
                logger.info(
                    "[speculative] Variant %d (seq, model=%s, temp=%.2f): "
                    "%d tokens (in=%d out=%d)",
                    spec.index,
                    spec.model_override or "(routed)",
                    spec.temperature,
                    response.usage.input_tokens + response.usage.output_tokens,
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[speculative] Variant %d sequential dispatch failed: %s",
                    spec.index, exc,
                )
                variant_responses.append(None)
    else:
        try:
            tasks = [
                gateway.dispatch(
                    messages=_seed_messages_with_style(messages, spec),
                    role=NodeRole.PATCHING,
                    budget_remaining_usd=budget,
                    temperature=spec.temperature,
                    model_override=spec.model_override,
                )
                for spec in variant_specs
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for spec, result in zip(variant_specs, results):
                if isinstance(result, BaseException):
                    logger.warning(
                        "[speculative] Variant %d LLM call failed: %s",
                        spec.index, result,
                    )
                    variant_responses.append(None)
                else:
                    response, new_budget = result  # (LLMResponse, new_budget)
                    variant_responses.append(response)
                    logger.info(
                        "[speculative] Variant %d (parallel, model=%s, temp=%.2f): "
                        "%d tokens (in=%d out=%d)",
                        spec.index,
                        spec.model_override or "(routed)",
                        spec.temperature,
                        response.usage.input_tokens + response.usage.output_tokens,
                        response.usage.input_tokens,
                        response.usage.output_tokens,
                    )
        except Exception as exc:
            logger.exception("[speculative] Variant generation failed: %s", exc)
            return _fallback_result()

    # Count successful LLM calls
    valid_variants = [r for r in variant_responses if r is not None]
    if not valid_variants:
        logger.error("[speculative] All variant LLM calls failed.")
        return _fallback_result()

    # Speculative needs HEAD to exist (worktree add uses HEAD as the source ref).
    # On a freshly `git init`'d repo with zero commits, skip cleanly instead
    # of letting `git worktree add HEAD` fail N times with a cryptic error.
    if not _repo_has_resolvable_head(workspace_path):
        logger.warning(
            "[speculative] Skipping speculative branching: workspace %s has no commits yet "
            "(unborn HEAD). Make an initial commit to enable speculative repair. "
            "Falling back to sequential repair.",
            workspace_path,
        )
        return _fallback_result()

    # --- Step 2: Create isolated worktrees and apply patches ---
    variant_results: list[VariantResult] = []

    for i, response in enumerate(variant_responses):
        if response is None:
            variant_results.append(VariantResult(index=i, variant_id="failed", worktree_path="", error="LLM call failed"))
            continue

        variant_id = str(uuid.uuid4())[:8]
        worktree_path = os.path.join(worktree_base, f"variant-{i}-{variant_id}")

        vr = VariantResult(index=i, variant_id=variant_id, worktree_path=worktree_path)
        vr.llm_response = response

        # Create git worktree
        if not _create_worktree(workspace_path, worktree_path):
            vr.error = "Failed to create git worktree"
            variant_results.append(vr)
            continue

        # Apply patches to the worktree
        try:
            patch_results, modified_files = await process_llm_patch_output(
                response.content,
                worktree_path,
                existing_modified_files=[],
            )
            vr.patch_results = patch_results
            vr.modified_files = modified_files

            success_count = sum(1 for r in patch_results if r.success)
            if success_count == 0:
                vr.error = f"No patches applied ({len(patch_results)} attempted)"
                variant_results.append(vr)
                continue

            logger.info("[speculative] Variant %d: %d/%d patches applied to %s",
                         i, success_count, len(patch_results), worktree_path)

        except Exception as exc:
            vr.error = f"Patch application failed: {exc}"
            variant_results.append(vr)
            continue

        variant_results.append(vr)

    # --- Step 3: Run lintgate on all variants ---
    try:
        from harness.lintgate import lintgate_node
        for vr in variant_results:
            if vr.error or not vr.worktree_path:
                continue
            lint_state = {
                "modified_files": vr.modified_files,
                "workspace_path": vr.worktree_path,
                "messages": [],
            }
            await lintgate_node(lint_state)
    except ImportError:
        pass  # lintgate not required

    # --- Step 4: Compile all variants in parallel ---
    from harness.sandbox import SandboxExecutor

    async def _compile_variant(vr: VariantResult) -> VariantResult:
        if vr.error or not vr.worktree_path:
            return vr
        try:
            # Per-variant late-bind. The workspace-time resolution above
            # sniffed an empty/greenfield workspace before any LLM call.
            # Now that this variant's patches are on disk in its worktree,
            # re-sniff against THAT tree so the toolchain reflects what
            # the variant actually produced. Without this, a greenfield
            # run keeps build_command="make build" against the bare
            # ubuntu:22.04 default and every variant exits 127 even when
            # one wrote a perfectly good Python (or Node) project.
            # detect_default_build_command + _apply_toolchain_adaptation
            # are idempotent — re-calling them produces the same answer
            # the workspace-time pass picked when the worktree matches.
            per_variant_build = build_command
            per_variant_sandbox = sandbox_config
            per_variant_network = allow_network
            per_variant_adapted: Optional[str] = None
            try:
                from harness.cli import _detect_default_build_command
                late = _detect_default_build_command(
                    vr.worktree_path, is_greenfield=is_greenfield_spec,
                )
                if late and late != per_variant_build:
                    logger.info(
                        "[speculative] Variant %d: build command resolved to %r "
                        "(workspace-time default was %r).",
                        vr.index, late, per_variant_build,
                    )
                    per_variant_adapted = late
                    per_variant_build = late
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "[speculative] Variant %d: per-variant build-command late-bind failed: %s",
                    vr.index, exc,
                )
            try:
                from harness.graph import _apply_toolchain_adaptation
                (
                    per_variant_sandbox,
                    per_variant_network,
                    v_image_adapted,
                    v_net_adapted,
                    _,
                ) = _apply_toolchain_adaptation(
                    per_variant_build,
                    per_variant_sandbox,
                    per_variant_network,
                    command_is_adapter_synthesised=per_variant_adapted is not None,
                )
                if v_image_adapted:
                    logger.info(
                        "[speculative] Variant %d: adapting docker_image to %r to match toolchain implied by: %s",
                        vr.index, per_variant_sandbox.get("docker_image"), per_variant_build,
                    )
                if v_net_adapted:
                    logger.info(
                        "[speculative] Variant %d: auto-enabling network for adapter-synthesised install step: %s",
                        vr.index, per_variant_build,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "[speculative] Variant %d: per-variant toolchain adaptation failed: %s",
                    vr.index, exc,
                )

            # Give each variant a private writable cache directory tree.
            # Multiple variants running in parallel would otherwise corrupt
            # each other's pip / npm / mypy / pytest caches —
            # those tools assume single-writer access to their cache dirs.
            #
            # Read-only host cache mounts (~/.cache/pip etc. via the unshare
            # backend's --bind -o ro) still serve as warm sources; the env
            # vars below redirect *writes* to per-variant locations.
            # When sandbox.cache_volumes is on, route package caches through
            # the shared named volume (warm-up has already populated it);
            # keep build-output caches variant-private.
            use_shared = bool((per_variant_sandbox or {}).get("cache_volumes"))
            variant_env = _build_variant_cache_env(
                vr.worktree_path, use_shared_package_cache=use_shared,
            )
            executor = SandboxExecutor(
                workspace_path=vr.worktree_path,
                extra_env=variant_env,
                sandbox_config=per_variant_sandbox,
                allow_network=per_variant_network,
                session_id=state.get("session_id"),
            )
            result = await executor.run(per_variant_build)
            vr.exit_code = result.exit_code
            vr.raw_output = result.raw_output
            vr.timed_out = result.timed_out
            logger.info("[speculative] Variant %d compiled: exit=%d timed_out=%s",
                         vr.index, vr.exit_code, vr.timed_out)
            # Exit 127 across variants is almost always a missing-toolchain
            # signal (the shell could not find the build binary). Append a
            # one-line hint to the raw_output so the operator — and the
            # downstream repair loop — see WHY rather than staring at a
            # bare "exit=127".
            if vr.exit_code == 127:
                hint = (
                    f"\n[speculative-hint] exit 127 typically means the shell could not "
                    f"find a binary in the build command. Resolved build_command="
                    f"{per_variant_build!r}, docker_image="
                    f"{per_variant_sandbox.get('docker_image', BUILDER_IMAGE)!r}. "
                    f"Either the toolchain isn't installed in that image or the worktree "
                    f"has no source markers the harness can detect."
                )
                vr.raw_output = (vr.raw_output or "") + hint
        except Exception as exc:
            vr.error = f"Compile failed: {exc}"
            logger.warning("[speculative] Variant %d compile error: %s", vr.index, exc)
        return vr

    # --- Cache warm-up pass ---
    # When sandbox.cache_volumes is on, the variants share a writable named
    # volume per tool (pip/npm). Without warm-up, all variants race to
    # cold-fill the cache in parallel — N× the network downloads. Warm-up
    # runs the install step once against the workspace (single-writer) so
    # the variants then fan out against an already-populated cache.
    #
    # Skip warm-up when:
    #   - cache_volumes is off (no shared cache to fill — variants have
    #     their own per-variant write dirs and host caches are read-only).
    #   - The build command does no install work (`make build`, etc.).
    #   - The workspace has no install markers (greenfield run — the
    #     workspace-time late-bind returned None, so build_command is still
    #     the unresolved default and warming up with it would just exit 127).
    cache_volumes_on = bool((sandbox_config or {}).get("cache_volumes"))
    if cache_volumes_on:
        try:
            from harness.graph import _build_command_needs_network
            from harness.cli import _detect_default_build_command as _detect_workspace_marker
            from harness.sandbox import SandboxExecutor
            session_id = state.get("session_id")
            needs_install = _build_command_needs_network(build_command)
            workspace_has_markers = _detect_workspace_marker(
                workspace_path, is_greenfield=is_greenfield_spec,
            ) is not None
            if needs_install and workspace_has_markers:
                logger.info(
                    "[speculative] Warm-up: priming shared cache volume(s) with "
                    "a single install pass before %d variants fan out.",
                    len(variant_results),
                )
                warmup_exec = SandboxExecutor(
                    workspace_path=workspace_path,
                    sandbox_config=sandbox_config,
                    allow_network=allow_network,
                    session_id=session_id,
                )
                warmup_result = await warmup_exec.run(build_command)
                logger.info(
                    "[speculative] Warm-up complete: exit=%d elapsed=%.2fs.",
                    warmup_result.exit_code, warmup_result.elapsed_seconds,
                )
            else:
                logger.debug(
                    "[speculative] Warm-up skipped: needs_install=%s "
                    "workspace_has_markers=%s.", needs_install, workspace_has_markers,
                )
        except Exception as exc:  # noqa: BLE001 — warm-up is best-effort
            logger.debug("[speculative] Warm-up failed: %s", exc)

    # Gate variant compilation behind ``cfg.max_concurrency`` so we don't
    # spawn N parallel docker containers (or N parallel native compilers)
    # regardless of operator config. Previously the semaphore was only
    # consulted in ``run_parallel_agents`` for voting; variant compilation
    # ignored it and could saturate the host. The cap is normalized to
    # [1, 10] by the config loader.
    _compile_sem = asyncio.Semaphore(max(1, cfg.max_concurrency))

    async def _compile_variant_gated(vr: "VariantResult") -> "VariantResult":
        async with _compile_sem:
            return await _compile_variant(vr)

    variant_results = list(await asyncio.gather(*[
        _compile_variant_gated(vr) for vr in variant_results
    ]))

    # --- Step 5: Select the winning variant ---
    winner = await _select_winner_async(
        variant_results,
        strategy,
        cfg=cfg,
        gateway=gateway,
        budget_remaining_usd=state.get("budget_remaining_usd", 0.0),
    )
    elapsed = time_module.monotonic() - start_time

    spec_result = SpeculativeResult(
        total_variants=len(variant_results),
        passed_variants=sum(1 for vr in variant_results if vr.passed),
        winner_index=winner.index if winner else -1,
        variant_results=variant_results,
        strategy=strategy,
        elapsed_seconds=elapsed,
    )

    # --- Step 6: Merge winning variant back ---
    if winner and winner.passed and winner.worktree_path:
        logger.info("[speculative] Selected Variant %d (exit=%d, files=%d). Merging back.",
                     winner.index, winner.exit_code, len(winner.modified_files))

        # Copy winning-variant files back to main workspace.
        # Use temp files + atomic rename so a crash mid-copy doesn't leave
        # the workspace in a half-merged state.
        import tempfile as _tempfile
        from harness.trust import safe_resolve as _safe_resolve
        merge_errors: list[str] = []
        for filepath in winner.modified_files:
            # Defense: the patcher already validates paths but the winner
            # comes from a worktree — re-validate against workspace_path.
            try:
                _safe_resolve(workspace_path, filepath)
            except ValueError:
                logger.warning("[speculative] Skipping out-of-workspace path: %s", filepath)
                continue

            src = os.path.join(winner.worktree_path, filepath)
            dst = os.path.join(workspace_path, filepath)
            if not os.path.isfile(src):
                continue
            dst_dir = os.path.dirname(dst)
            try:
                os.makedirs(dst_dir, exist_ok=True)
                fd, tmp = _tempfile.mkstemp(dir=dst_dir)
                try:
                    os.close(fd)
                    shutil.copy2(src, tmp)
                    os.replace(tmp, dst)
                except Exception:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
                    raise
            except OSError as copy_err:
                logger.error("[speculative] Failed to merge %s: %s", filepath, copy_err)
                merge_errors.append(filepath)

        if merge_errors:
            logger.warning("[speculative] %d file(s) could not be merged: %s",
                           len(merge_errors), merge_errors)

        # --- Step 7: Cleanup worktrees ---
        _cleanup_worktrees(workspace_path, worktree_base, variant_results)

        # Build status message
        status_parts = [
            f"[Speculative] {spec_result.passed_variants}/{spec_result.total_variants} variants passed.",
            f"  Selected Variant {winner.index} (strategy: {strategy}).",
            f"  Winner: {len(winner.patch_results)} patches, {len(winner.modified_files)} files, exit {winner.exit_code}.",
        ]
        for vr in variant_results:
            if vr is not winner:
                status = "PASS" if vr.passed else f"FAIL (exit={vr.exit_code})"
                status_parts.append(f"  Variant {vr.index}: {status}")

        messages_out = list(state.get("messages", []))
        messages_out.append({"role": "system", "content": "\n".join(status_parts)})

        # Update token tracker with the winner's LLM usage. Per-stage
        # attribution: speculative variants are NodeRole.PATCHING dispatches.
        token_tracker = state.get("token_tracker", {})
        if winner.llm_response is not None:
            token_tracker = gateway.aggregate_tokens(
                token_tracker, winner.llm_response.usage, role=NodeRole.PATCHING,
            )

        logger.info("[speculative] Complete: %.2fs, winner=Variant %d.", elapsed, winner.index)

        # Merge winner.modified_files into the prior accumulated list
        # (audit §6.2). When speculative runs as ``after_n_repair_failures``
        # the patching_node already produced files; replacing here would
        # drop those from state and downstream nodes (lintgate /
        # test_generation / repair) would lose visibility of them.
        prior_modified_winner: list[str] = list(state.get("modified_files", []) or [])
        merged_winner_modified = list(prior_modified_winner)
        for f in winner.modified_files:
            if f not in merged_winner_modified:
                merged_winner_modified.append(f)

        return {
            "modified_files": merged_winner_modified,
            "messages": messages_out,
            "token_tracker": token_tracker,
            "node_state": {
                "speculative": {
                    "winner_index": winner.index,
                    "total_variants": spec_result.total_variants,
                    "passed_variants": spec_result.passed_variants,
                },
            },
        }

    # --- Fallback: all variants failed ---
    # Before throwing away every variant's work, try to salvage the best
    # failing one and merge its patches back to the real workspace. Variants
    # often fail their compile not because their generated code is wrong
    # but because the sandbox is missing a pip dep or pytest had no tests
    # to collect — both of which the repair loop can resolve once the code
    # actually lives on disk. Without salvage, the repair loop starts from
    # an empty workspace and spins out on "no source to fix".
    # Gate salvage on the strategy. Default is SALVAGE_NONE — fall back
    # to sequential repair against the untouched workspace rather than
    # risk an incoherent merge.
    if cfg.salvage_strategy == SALVAGE_NONE:
        salvage = None
    else:
        salvage = _pick_salvage_variant(variant_results)
    if salvage is not None:
        logger.warning(
            "[speculative] All %d variants failed, but Variant %d applied "
            "%d patch(es) with a recoverable failure (exit=%d). Salvaging "
            "its patches to the workspace so the repair loop has real code "
            "to work with.",
            len(variant_results), salvage.index,
            sum(1 for r in salvage.patch_results if r.success),
            salvage.exit_code,
        )
        merge_errors = _merge_variant_into_workspace(salvage, workspace_path)
        _cleanup_worktrees(workspace_path, worktree_base, variant_results)

        # Merge salvaged files into the accumulated modified_files list rather
        # than replacing it — the downstream nodes (lintgate, test_generation,
        # repair) all read modified_files as the source of truth for "what
        # changed this session." If we replaced instead of merged we'd drop
        # any files an earlier patching pass produced.
        prior_modified: list[str] = list(state.get("modified_files", []) or [])
        merged_modified = list(prior_modified)
        for f in salvage.modified_files:
            if f not in merged_modified:
                merged_modified.append(f)

        logger.info(
            "[speculative:salvage] Merged Variant %d → workspace: %d new file(s) "
            "(prior modified=%d, merge_errors=%d). modified_files now=%d.",
            salvage.index, len(salvage.modified_files),
            len(prior_modified), len(merge_errors), len(merged_modified),
        )

        messages_out = list(state.get("messages", []))
        status_parts = [
            f"[Speculative] All {len(variant_results)} variants failed.",
            (
                f"  Salvaged Variant {salvage.index}: "
                f"{len(salvage.modified_files)} file(s) merged back. "
                f"Build failure (exit={salvage.exit_code}) appears recoverable; "
                f"routing to repair_node for follow-up fix."
            ),
        ]
        for vr in variant_results:
            if vr is not salvage:
                status_parts.append(f"  Variant {vr.index}: {vr.error or f'exit={vr.exit_code}'}")
        if merge_errors:
            status_parts.append(
                f"  Note: {len(merge_errors)} file(s) could not be merged back: {merge_errors}"
            )
        messages_out.append({"role": "system", "content": "\n".join(status_parts)})

        token_tracker = state.get("token_tracker", {})
        if salvage.llm_response is not None:
            token_tracker = gateway.aggregate_tokens(
                token_tracker, salvage.llm_response.usage, role=NodeRole.PATCHING,
            )

        return {
            "modified_files": merged_modified,
            "messages": messages_out,
            "token_tracker": token_tracker,
            "node_state": {
                "speculative": {
                    "all_failed": True,
                    "salvaged_index": salvage.index,
                    "salvaged_files": len(salvage.modified_files),
                    "total_variants": spec_result.total_variants,
                },
            },
        }

    _cleanup_worktrees(workspace_path, worktree_base, variant_results)

    logger.warning("[speculative] All %d variants failed. Falling back to sequential repair.",
                   len(variant_results))

    messages_out = list(state.get("messages", []))
    status_parts = [f"[Speculative] All {len(variant_results)} variants failed. Falling back to standard repair."]
    for vr in variant_results:
        status_parts.append(f"  Variant {vr.index}: {vr.error or f'exit={vr.exit_code}'}")
    messages_out.append({"role": "system", "content": "\n".join(status_parts)})

    return {
        "messages": messages_out,
        "node_state": {
            "speculative": {
                "all_failed": True,
                "total_variants": spec_result.total_variants,
            },
        },
    }


# ---------------------------------------------------------------------------
# 2b. Salvage helpers (rescue the best failing variant on full-fleet failure)
# ---------------------------------------------------------------------------

# Exit codes / output patterns that signal a recoverable build failure —
# the variant's patches are likely fine, but the sandbox couldn't run them
# end to end. The repair loop on the real workspace can resolve these.
_RECOVERABLE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # pip-installable test runner missing
    re.compile(r"ModuleNotFoundError: No module named ['\"](?:pytest|pytest_\w+|ruff|mypy|black|isort|coverage)['\"]"),
    re.compile(r"^/[^:\s]+/python3?: No module named (pytest|pytest_\w+|ruff|mypy|black|isort)\s*$", re.MULTILINE),
    # pytest exit-5: no tests collected — handled downstream by test_generation
    re.compile(r"(?m)^=*\s*no tests ran in [\d.]+s\s*=*$"),
    re.compile(r"(?m)^no tests ran in [\d.]+s\s*$"),
    # Missing application dep (e.g. fastapi, uvicorn, sqlalchemy)
    re.compile(r"ModuleNotFoundError: No module named ['\"][^'\"]+['\"]"),
)

_SALVAGE_PYTEST_EXIT_CODES: frozenset[int] = frozenset({1, 2, 4, 5})


def _is_recoverable_failure(vr: "VariantResult") -> bool:
    """True when the variant's compile failure looks like something the
    sequential repair loop can fix once the patches live on the real
    workspace (missing deps, no tests collected, generic test failures).

    Excludes timeouts, sandbox errors, and exit codes that suggest the
    container itself is misconfigured (which the repair loop can't help with).
    """
    if vr.timed_out:
        return False
    if vr.error and not vr.patch_results:
        return False
    # Hard NO when no patches actually landed on the worktree — there's
    # nothing to merge back.
    if not any(r.success for r in vr.patch_results):
        return False
    # Permissive: any non-zero pytest-shaped exit code can be salvaged if
    # the tail of the output contains a recoverable signature.
    if vr.exit_code in _SALVAGE_PYTEST_EXIT_CODES:
        tail = (vr.raw_output or "")[-4000:]
        if any(p.search(tail) for p in _RECOVERABLE_PATTERNS):
            return True
        # Even without a signature, exit codes 1-5 from a test runner are
        # fixable by the repair LLM in most cases (assertion failures,
        # import errors in the user's own code).
        return True
    return False


def _pick_salvage_variant(variant_results: list["VariantResult"]) -> Optional["VariantResult"]:
    """Among the failed variants, pick the most-promising one to merge back.

    Ranking: most successful patches first, then fewest lines changed
    (Occam-ish — smaller diffs are less likely to drag in hallucinated code).
    Returns None when no variant qualifies for salvage.

    Quality gate (C1): even the best candidate is refused if it doesn't
    meet a minimum coherence bar. A variant where only a tiny fraction of
    its patches landed produces a half-built workspace that the repair
    loop then tries — and fails — to fix. Better to refuse salvage and let
    repair work from the pre-speculative workspace, which is internally
    coherent even if it's missing features.

    Bar (conservative, all must hold):
      - applied_patches >= 3 (variants with 1-2 successful patches are too
        thin to be worth merging — they add scaffolding without substance).
      - applied_patches / total_patches >= 0.50 (at least half of what the
        variant tried actually landed — fewer than that suggests fundamental
        confusion about workspace state).
    """
    candidates = [vr for vr in variant_results if _is_recoverable_failure(vr)]
    if not candidates:
        return None
    candidates.sort(
        key=lambda vr: (
            -sum(1 for r in vr.patch_results if r.success),
            vr.total_lines_changed,
        )
    )
    best = candidates[0]
    applied = sum(1 for r in best.patch_results if r.success)
    total = len(best.patch_results) or 1
    pct = applied / total
    MIN_APPLIED = 3
    MIN_PCT = 0.50
    if applied < MIN_APPLIED or pct < MIN_PCT:
        logger.warning(
            "[speculative:salvage] Best candidate Variant %d does not meet "
            "the quality gate (applied=%d/%d=%.0f%%, need >=%d patches and "
            ">=%.0f%%). Refusing salvage; repair will start from the "
            "pre-speculative workspace.",
            best.index, applied, total, pct * 100, MIN_APPLIED, MIN_PCT * 100,
        )
        return None
    return best


def _merge_variant_into_workspace(
    vr: "VariantResult", workspace_path: str,
) -> list[str]:
    """Copy a variant's successful patch files back into the workspace.

    Mirrors the merge step used for the winner path, but operates on a
    failing-but-salvageable variant. Returns the list of files that could
    not be merged (empty on full success).
    """
    import tempfile as _tempfile
    from harness.trust import safe_resolve as _safe_resolve

    merge_errors: list[str] = []
    for filepath in vr.modified_files:
        try:
            _safe_resolve(workspace_path, filepath)
        except ValueError:
            logger.warning(
                "[speculative:salvage] Skipping out-of-workspace path: %s", filepath
            )
            continue

        src = os.path.join(vr.worktree_path, filepath)
        dst = os.path.join(workspace_path, filepath)
        if not os.path.isfile(src):
            continue
        dst_dir = os.path.dirname(dst)
        try:
            os.makedirs(dst_dir, exist_ok=True)
            fd, tmp = _tempfile.mkstemp(dir=dst_dir)
            try:
                os.close(fd)
                shutil.copy2(src, tmp)
                os.replace(tmp, dst)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError as copy_err:
            logger.error(
                "[speculative:salvage] Failed to merge %s: %s", filepath, copy_err
            )
            merge_errors.append(filepath)
    return merge_errors


# ---------------------------------------------------------------------------
# 3. Selection Strategies
# ---------------------------------------------------------------------------

def _select_winner(
    variant_results: list[VariantResult],
    strategy: str = SELECT_FIRST_PASS,
) -> Optional[VariantResult]:
    """
    Select the winning variant based on the configured strategy.

    Strategies (string constants exported at module top):
        - first_pass / first_success: first variant with exit_code 0
        - fewest_changes: passing variant with the smallest diff
        - all_pass: only return a winner when every variant passes
        - voted: see :func:`_select_winner_async` (requires gateway)
    """
    passing = [vr for vr in variant_results if vr.passed]

    if not passing:
        return None

    if strategy == SELECT_ALL_PASS:
        if len(passing) == len(variant_results):
            return passing[0]
        logger.warning("[speculative] all_pass strategy: %d/%d passed. No winner selected.",
                        len(passing), len(variant_results))
        return None

    if strategy == SELECT_FEWEST_CHANGES:
        return min(passing, key=lambda vr: vr.total_lines_changed)

    # voted falls back here when the async selector wasn't used —
    # treat as first_pass so callers always get a deterministic answer.
    return passing[0]


async def _select_winner_async(
    variant_results: list[VariantResult],
    strategy: str,
    *,
    cfg: "SpeculativeConfig",
    gateway: Any,
    budget_remaining_usd: float,
) -> Optional[VariantResult]:
    """Async wrapper that adds the ``voted`` strategy on top of the
    synchronous :func:`_select_winner`.

    For non-voted strategies, returns the same answer as the sync path.
    For ``voted``: keeps the passing variants, dispatches
    ``cfg.voting.n_judges`` adversarial reviewers per candidate to
    score them, then returns the variant with the highest accept-rate
    (ties broken by fewest_changes — Occam keeps the smaller diff).
    """
    passing = [vr for vr in variant_results if vr.passed]
    if strategy != SELECT_VOTED or len(passing) <= 1 or gateway is None:
        return _select_winner(variant_results, strategy)
    try:
        from harness.fanout import AgentSpec, run_parallel_agents
        from harness.gateway import NodeRole as _NR
    except Exception as exc:  # noqa: BLE001
        logger.warning("[speculative] voted fallback to first_pass (%s)", exc)
        return passing[0]
    role_map = {
        "code_reviewer": _NR.CODE_REVIEWER,
        "doc_reviewer": _NR.DOC_REVIEWER,
        "planning": _NR.PLANNING,
    }
    judge_role = role_map.get(cfg.voting.judge_role, _NR.CODE_REVIEWER)

    scores: dict[int, int] = {}
    for vr in passing:
        snippet = (vr.raw_output or "")[-1500:] if vr.raw_output else ""
        files = ", ".join(vr.modified_files[:6])
        prompt = (
            "You are scoring a candidate code patch variant. The variant "
            "compiled cleanly. Decide whether you would accept this "
            "variant as the winner among several passing candidates. "
            "Respond with a one-line JSON object with keys "
            "`accept` (bool) and `reason` (short string).\n\n"
            f"Variant index: {vr.index}\n"
            f"Files modified: {files}\n"
            f"Build output tail:\n{snippet}\n"
        )
        judge_specs = [
            AgentSpec(
                name=f"judge-v{vr.index}-{i}",
                system_prompt=(
                    "You are an adversarial code reviewer. Be skeptical; "
                    "default to accept=false when uncertain."
                ),
                user_prompt=prompt,
                role=judge_role,
            )
            for i in range(max(1, cfg.voting.n_judges))
        ]
        votes, budget_remaining_usd = await run_parallel_agents(
            judge_specs, gateway,
            budget_remaining_usd=budget_remaining_usd,
            max_concurrency=cfg.max_concurrency,
        )
        accept = 0
        for v in votes:
            if not v.success:
                continue
            from harness.fanout import _parse_first_json
            verdict = _parse_first_json(v.content) or {}
            if bool(verdict.get("accept", False)):
                accept += 1
        scores[vr.index] = accept

    # Pick max accept; ties broken by fewest lines changed.
    ranked = sorted(
        passing,
        key=lambda vr: (-scores.get(vr.index, 0), vr.total_lines_changed),
    )
    return ranked[0]


# ---------------------------------------------------------------------------
# 4. Worktree Management
# ---------------------------------------------------------------------------

def _build_variant_cache_env(
    worktree_path: str,
    *,
    use_shared_package_cache: bool = False,
) -> dict[str, str]:
    """
    Build environment variables that redirect every common build tool's
    *writable* cache to a variant-local directory tree.

    Without this, parallel variants run concurrent `pip install`,
    `npm install`, `pytest`, `mypy`, etc. against the same shared
    per-user cache directories — pip's lock file races, mypy's
    incremental cache gets mixed across branches, and pytest's
    `.pytest_cache` becomes meaningless.

    Each variant gets ``<worktree>/.harness-cache/<tool>/`` so writes are
    isolated. The host-level read-only cache mounts (configured via
    ``sandbox.readonly_cache_mounts``) still seed warm dependencies —
    these env vars only affect where writes land.

    When ``use_shared_package_cache`` is True (``sandbox.cache_volumes`` is
    on), the **package** caches (pip, npm, maven repo) are NOT overridden
    — they fall through to the tool's default paths inside the container,
    which the docker backend has bind-mounted to a writable named volume.
    Build-output / incremental tool caches (__pycache__, mypy, pytest,
    ruff, gradle) stay per-variant since different variants produce
    different code and must not share build artifacts.

    Returned env-var keys (each pointing to a per-variant subdirectory):
      - PIP_CACHE_DIR          (Python pip; omitted when shared cache is on)
      - npm_config_cache       (npm — lowercase is canonical; omitted when shared)
      - GRADLE_USER_HOME       (Gradle — always per-variant)
      - MAVEN_OPTS             (-Dmaven.repo.local override; omitted when shared)
      - PYTHONPYCACHEPREFIX    (Python __pycache__ — always per-variant)
      - MYPY_CACHE_DIR         (mypy incremental — always per-variant)
      - RUFF_CACHE_DIR         (ruff — always per-variant)
      - PYTEST_ADDOPTS         (forces -o cache_dir=... — always per-variant)
      - XDG_CACHE_HOME         (generic XDG fallback — always per-variant)
    """
    base = os.path.join(worktree_path, ".harness-cache")
    os.makedirs(base, exist_ok=True)

    def _sub(name: str) -> str:
        p = os.path.join(base, name)
        os.makedirs(p, exist_ok=True)
        return p

    env: dict[str, str] = {
        # Build outputs + tool incremental state — ALWAYS per-variant,
        # regardless of cache_volumes. Different variants produce different
        # code; sharing __pycache__, mypy incremental, ruff, pytest, or
        # gradle home would mix branches and corrupt verdicts.
        "PYTHONPYCACHEPREFIX": _sub("pycache"),
        "MYPY_CACHE_DIR": _sub("mypy"),
        "RUFF_CACHE_DIR": _sub("ruff"),
        "PYTEST_ADDOPTS": f"-o cache_dir={_sub('pytest')}",
        "GRADLE_USER_HOME": _sub("gradle"),
        "XDG_CACHE_HOME": _sub("xdg"),
    }
    if not use_shared_package_cache:
        # Default: per-variant package caches so concurrent writes don't
        # race. With cache_volumes on, the docker backend bind-mounts a
        # writable named volume at the container's default tool paths,
        # and the speculative warm-up pass primes the volume before
        # fan-out — leaving these env vars unset lets the tools pick up
        # the shared cache.
        maven_repo = _sub("maven-repo")
        env.update({
            "PIP_CACHE_DIR": _sub("pip"),
            "npm_config_cache": _sub("npm"),
            "MAVEN_OPTS": f"-Dmaven.repo.local={maven_repo}",
        })
    return env


def _repo_has_resolvable_head(repo_path: str) -> bool:
    """True iff the repo at repo_path has at least one commit (HEAD resolves).

    Speculative branching depends on `git worktree add ... HEAD`, which fails
    on an empty `git init`'d repo with `fatal: invalid reference: HEAD`.
    """
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "--verify", "--quiet", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except Exception:
        return False
    return result.returncode == 0


def _create_worktree(repo_path: str, worktree_path: str) -> bool:
    """Create a git worktree at the given path."""
    os.makedirs(os.path.dirname(worktree_path), exist_ok=True)

    try:
        # Remove if exists from a previous run
        if os.path.exists(worktree_path):
            _remove_worktree(repo_path, worktree_path)

        result = subprocess.run(
            ["git", "-C", repo_path, "worktree", "add", "--detach", worktree_path, "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("[speculative] Failed to create worktree at %s: %s",
                           worktree_path, result.stderr.strip())
            return False

        logger.debug("[speculative] Created worktree at %s", worktree_path)
        return True
    except Exception as exc:
        logger.warning("[speculative] Worktree creation error: %s", exc)
        return False


def _remove_worktree(repo_path: str, worktree_path: str) -> None:
    """Remove a git worktree."""
    try:
        subprocess.run(
            ["git", "-C", repo_path, "worktree", "remove", "--force", worktree_path],
            capture_output=True,
            timeout=30,
        )
    except Exception:
        pass

    # Fallback: manual cleanup
    if os.path.isdir(worktree_path):
        try:
            shutil.rmtree(worktree_path, ignore_errors=True)
        except Exception:
            pass


def _cleanup_worktrees(
    repo_path: str,
    worktree_base: str,
    variant_results: list[VariantResult],
) -> None:
    """Remove all temporary worktrees.

    Do NOT clear ``vr.modified_files`` / ``vr.patch_results`` here. Both the
    winner-merge path and the salvage path read those fields after cleanup
    (to populate the LangGraph state return) — clearing them dropped the
    list of merged files on the floor, so downstream nodes saw
    ``modified_files=[]`` even though files HAD been copied to the
    workspace. Only ``worktree_path`` is reset so callers don't try to
    touch a directory that's no longer there.
    """
    for vr in variant_results:
        if vr.worktree_path and os.path.isdir(vr.worktree_path):
            _remove_worktree(repo_path, vr.worktree_path)
            logger.debug("[speculative] Removed worktree %s", vr.worktree_path)
        vr.worktree_path = ""


# ---------------------------------------------------------------------------
# 5. Fallback
# ---------------------------------------------------------------------------

def _fallback_result() -> dict[str, Any]:
    """Return a state update that passes through to normal patching."""
    logger.info("[speculative] Passing through to standard patching flow.")
    return {
        "node_state": {
            "speculative": {
                "fallback": True,
                "reason": "speculative execution unavailable",
            },
        },
    }

# ---------------------------------------------------------------------------
# 6. Repair-level fanout
# ---------------------------------------------------------------------------
# The graph-level speculative_node fans out at PATCH generation (per story /
# batch entry); the inner compiler<->repair cycle stays sequential and can
# burn its whole round cap on one stuck trajectory. Repair fanout is the
# missing variant primitive for that loop: when K consecutive no-progress
# rounds have accrued, sample N repair responses (distinct strategy
# framings), test-compile each in a worktree seeded with the CURRENT dirty
# workspace, and return the best response. The caller (repair_node)
# substitutes it for the single dispatch it was about to make and runs its
# ENTIRE normal path unchanged — READ_FILE cycle, patcher, bookkeeping —
# so the workspace is only ever mutated by the standard apply machinery.

_REPAIR_FANOUT_STRATEGIES: tuple[str, ...] = (
    "STRATEGY DIRECTIVE: Make the minimal, surgical fix for the single "
    "highest-priority diagnostic. Touch as few lines as possible; do not "
    "refactor.",
    "STRATEGY DIRECTIVE: Re-derive the root cause from scratch. Previous "
    "rounds may have misdiagnosed the failure — consider that the real "
    "defect is upstream of the reported symptom (wrong file, wrong layer, "
    "wrong assumption in a caller or fixture).",
    "STRATEGY DIRECTIVE: Abandon the approach taken in previous rounds. "
    "Rewrite the affected block(s) cleanly rather than patching the prior "
    "attempt; prefer replacing a whole function body over another "
    "incremental edit.",
    "STRATEGY DIRECTIVE: Suspect the tests and environment as much as the "
    "code: check for stale imports, fixture drift, and assertions that "
    "no longer match intended behaviour, and fix whichever side is wrong.",
    "STRATEGY DIRECTIVE: Fix the top diagnostic AND audit the same pattern "
    "elsewhere in the touched files — apply the corrected pattern "
    "consistently in one pass.",
)


@dataclass
class RepairFanoutOutcome:
    """What ``maybe_run_repair_fanout`` hands back to repair_node."""
    response: Any                    # winning (or best-effort) LLMResponse
    budget: float                    # budget after all variant dispatches
    won: bool                        # True when a variant compiled clean
    extra_usages: list[Any] = field(default_factory=list)
    """Usage payloads from the non-returned variant dispatches, so the
    caller can aggregate the FULL token spend, not just the winner's."""


def repair_fanout_should_engage(
    cfg: "SpeculativeConfig", loop_counter: dict[str, Any],
) -> bool:
    """Engage exactly when the no-progress streak reaches the configured
    round count. The == (not >=) comparison makes engagement self-limiting:
    one fanout per climb toward the HITL cap — if the fanout round fails,
    the streak keeps climbing past the threshold without re-engaging; if
    it succeeds, the streak resets and the trigger re-arms for the next
    stall.
    """
    if not cfg.repair_fanout:
        return False
    streak = int(loop_counter.get("no_progress_repairs", 0) or 0)
    return streak == cfg.repair_fanout_after_rounds


def _seed_worktree_from_workspace(workspace_path: str, worktree_path: str) -> int:
    """Overlay the workspace's uncommitted state onto a fresh HEAD worktree.

    ``git worktree add --detach HEAD`` reproduces the last commit, but the
    repair loop runs mid-batch with many uncommitted modifications — a
    variant compiled against bare HEAD would be judged against stale code.
    Copy every path ``git status`` reports (modified, added, untracked;
    gitignore respected) and mirror deletions. Returns the number of paths
    synced. NUL-separated output (-z) so quoting/renames can't corrupt
    paths.
    """
    result = subprocess.run(
        ["git", "-C", workspace_path, "status", "--porcelain=v1", "-z", "-uall"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git status failed: {result.stderr.strip()[:200]}")
    synced = 0
    entries = [e for e in result.stdout.split("\0") if e]
    i = 0
    while i < len(entries):
        entry = entries[i]
        i += 1
        status, path = entry[:2], entry[3:]
        if status.startswith(("R", "C")):
            # Rename/copy entries carry the ORIGIN path as the next
            # NUL-separated record; the destination is in this one.
            i += 1
        src = os.path.join(workspace_path, path)
        dst = os.path.join(worktree_path, path)
        if os.path.isfile(src):
            os.makedirs(os.path.dirname(dst) or worktree_path, exist_ok=True)
            shutil.copy2(src, dst)
            synced += 1
        elif not os.path.exists(src) and os.path.isfile(dst):
            os.remove(dst)
            synced += 1
    return synced


async def maybe_run_repair_fanout(
    *,
    state: dict[str, Any],
    messages: list[dict[str, Any]],
    dispatch: Any,
    workspace_path: str,
    budget: float,
    loop_counter: dict[str, Any],
    compile_variant: Any = None,
) -> Optional[RepairFanoutOutcome]:
    """Run the repair fanout when the trigger is met; None means "caller
    proceeds with its normal single dispatch" (disabled, trigger not met,
    unborn HEAD, or every variant dispatch failed). Never raises.

    ``dispatch`` is repair_node's own ``_dispatch_repair`` (escalation
    model + cache family included) called as ``await dispatch(msgs,
    budget) -> (response, new_budget)``. Variant diversity comes from
    per-variant strategy directives appended as the final user message.
    ``compile_variant`` is a test seam: ``await compile_variant(vr) -> vr``
    with ``vr.exit_code`` set; the default runs the state's build command
    in a sandbox against the variant's worktree.
    """
    try:
        cfg = SpeculativeConfig.from_state(state)
        if not repair_fanout_should_engage(cfg, loop_counter):
            return None
        if not _repo_has_resolvable_head(workspace_path):
            logger.info(
                "[repair_fanout] Skipping: workspace has no commits "
                "(unborn HEAD) — worktrees need a HEAD to branch from.",
            )
            return None

        n = cfg.repair_fanout_variants
        logger.info(
            "[repair_fanout] Engaging: %d consecutive no-progress repair "
            "round(s) reached the configured threshold — sampling %d "
            "repair variant(s).",
            loop_counter.get("no_progress_repairs", 0), n,
        )

        # --- Dispatch N variants (sequential: budget threads through) ---
        responses: list[Any] = []
        for i in range(n):
            strategy = _REPAIR_FANOUT_STRATEGIES[i % len(_REPAIR_FANOUT_STRATEGIES)]
            variant_messages = list(messages) + [
                {"role": "user", "content": strategy},
            ]
            try:
                response, budget = await dispatch(variant_messages, budget)
                responses.append(response)
            except Exception as exc:  # noqa: BLE001 — one bad variant is fine
                logger.warning(
                    "[repair_fanout] Variant %d dispatch failed: %s", i, exc,
                )
                responses.append(None)
        live = [r for r in responses if r is not None
                and (getattr(r, "content", "") or "").strip()]
        if not live:
            logger.warning("[repair_fanout] Every variant dispatch failed.")
            return None

        # --- Apply each variant in a seeded worktree and compile ---
        build_command = str(state.get("build_command") or "")
        worktree_base = os.path.join(cfg.worktree_base_dir, "repair")
        variant_results: list[VariantResult] = []

        async def _default_compile(vr: "VariantResult") -> "VariantResult":
            from harness.sandbox import SandboxExecutor
            sandbox_config = dict(state.get("sandbox_config") or {})
            variant_env = _build_variant_cache_env(
                vr.worktree_path,
                use_shared_package_cache=bool(sandbox_config.get("cache_volumes")),
            )
            executor = SandboxExecutor(
                workspace_path=vr.worktree_path,
                extra_env=variant_env,
                sandbox_config=sandbox_config,
                allow_network=bool(state.get("allow_network", False)),
                session_id=state.get("session_id"),
            )
            result = await executor.run(build_command)
            vr.exit_code = result.exit_code
            vr.timed_out = result.timed_out
            return vr

        compile_fn = compile_variant or _default_compile

        for i, response in enumerate(responses):
            if response is None or not (getattr(response, "content", "") or "").strip():
                continue
            variant_id = str(uuid.uuid4())[:8]
            worktree_path = os.path.join(worktree_base, f"repair-{i}-{variant_id}")
            vr = VariantResult(index=i, variant_id=variant_id, worktree_path=worktree_path)
            vr.llm_response = response
            if not _create_worktree(workspace_path, worktree_path):
                vr.error = "worktree creation failed"
                vr.worktree_path = ""
                variant_results.append(vr)
                continue
            try:
                synced = _seed_worktree_from_workspace(workspace_path, worktree_path)
                logger.info(
                    "[repair_fanout] Variant %d worktree seeded with %d "
                    "dirty path(s) from the workspace.", i, synced,
                )
                patch_results, modified = await process_llm_patch_output(
                    response.content, worktree_path, existing_modified_files=[],
                )
                vr.patch_results = patch_results
                vr.modified_files = modified
                if not any(r.success for r in patch_results):
                    vr.error = f"no patches applied ({len(patch_results)} attempted)"
                    variant_results.append(vr)
                    continue
            except Exception as exc:  # noqa: BLE001
                vr.error = f"apply failed: {exc}"
                variant_results.append(vr)
                continue
            variant_results.append(vr)

        compile_sem = asyncio.Semaphore(max(1, cfg.max_concurrency))

        async def _compile_gated(vr: "VariantResult") -> "VariantResult":
            if vr.error or not vr.worktree_path or not build_command:
                return vr
            async with compile_sem:
                try:
                    return await compile_fn(vr)
                except Exception as exc:  # noqa: BLE001
                    vr.error = f"compile failed: {exc}"
                    return vr

        variant_results = list(await asyncio.gather(
            *[_compile_gated(vr) for vr in variant_results]
        ))

        # --- Select: first clean compile wins; else best-effort ---
        winner = next(
            (vr for vr in variant_results
             if not vr.error and vr.exit_code == 0 and not vr.timed_out),
            None,
        )
        for vr in variant_results:
            logger.info(
                "[repair_fanout] Variant %d: exit=%s patches=%d error=%s%s",
                vr.index, vr.exit_code, len(vr.patch_results),
                vr.error or "-",
                " <- WINNER" if winner is vr else "",
            )
        _cleanup_worktrees(workspace_path, worktree_base, variant_results)

        if winner is not None:
            chosen = winner.llm_response
            won = True
        else:
            # No variant compiled clean: hand back the response whose
            # patches at least APPLIED (most patcher successes), so the
            # round costs no additional dispatch and the loop's normal
            # feedback machinery sees a concrete attempt.
            applied = [vr for vr in variant_results if not vr.error]
            best = max(
                applied,
                key=lambda vr: sum(1 for r in vr.patch_results if r.success),
                default=None,
            )
            chosen = best.llm_response if best is not None else live[0]
            won = False
            logger.info(
                "[repair_fanout] No variant compiled clean — continuing "
                "the sequential round with the best-applying variant.",
            )
        extra = [
            getattr(r, "usage", None) for r in live
            if r is not chosen and getattr(r, "usage", None)
        ]
        return RepairFanoutOutcome(
            response=chosen, budget=budget, won=won, extra_usages=extra,
        )
    except Exception:  # noqa: BLE001 — fanout must never break the repair loop
        logger.warning(
            "[repair_fanout] Fanout failed; falling back to the normal "
            "sequential dispatch.", exc_info=True,
        )
        return None
