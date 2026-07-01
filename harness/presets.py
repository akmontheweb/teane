"""Named config presets — Frugal / Balanced / Maximum Quality.

Phase 6 of the consumer-grade UI overhaul. Each preset is a set of
dotted-key overlays that get merged into the currently-loaded config,
re-validated with :func:`harness.cli.validate_config_strict`, and
persisted atomically.

Kept as a plain library so the Phase 3 wizard (whose "no config yet"
path builds a fresh config from scratch) can consume the same table
and the /config-ui preset page (this phase) can render it. No HTTP
concerns here.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("harness.presets")


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------
# Each preset is a dict-of-dotted-keys → value. Applying a preset walks
# every key, drills into the config, and sets the leaf value. Dotted-key
# overlay was chosen over a nested dict merge because it makes the diff
# preview trivially computable: for each key, "old value" vs "new value"
# is a one-liner.

FRUGAL_PRESET: dict[str, Any] = {
    # Cheapest models everywhere (DeepSeek is currently the cheapest
    # remote choice; Ollama is free but slower/less capable — pick per
    # user's key setup). We keep the SAME model for every role so
    # there's no fallback burn.
    "model_routing.planning_primary":  "deepseek:deepseek-v4-flash",
    "model_routing.planning_mode":     "non_thinking",
    "model_routing.patching_primary":  "deepseek:deepseek-v4-flash",
    "model_routing.patching_mode":     "non_thinking",
    "model_routing.repair_primary":    "deepseek:deepseek-v4-flash",
    "model_routing.repair_mode":       "non_thinking",
    "model_routing.doc_reviewer_primary":  "",
    "model_routing.code_reviewer_primary": "",
    # Tight budget so a runaway loop can't burn a hobbyist's month
    # of credits.
    "token_budget.hard_cap_usd": 2.0,
    # Never spend on speculative sibling attempts.
    "speculative.enabled":       False,
    "speculative.trigger":       "manual",
    "speculative.cost_strategy": "all_cheap",
}

BALANCED_PRESET: dict[str, Any] = {
    # Balanced is a "reset to conservative defaults" preset: sensible
    # cost + safety without extras. Numbers mirror the shipped
    # config/config.json defaults so applying Balanced feels like
    # "start over from a clean slate."
    "token_budget.hard_cap_usd": 3.0,
    "speculative.enabled":       True,
    "speculative.trigger":       "after_n_repair_failures",
    "speculative.cost_strategy": "cheap_first_sequential",
    "security.diff_approval_required": False,
}

MAXIMUM_QUALITY_PRESET: dict[str, Any] = {
    # Best-in-class remote models across the board. Requires the
    # relevant API keys to be set — the wizard's setup step is the
    # right place to arrange that.
    "model_routing.planning_primary":  "anthropic:claude-sonnet-4",
    "model_routing.planning_mode":     "thinking_max",
    "model_routing.patching_primary":  "anthropic:claude-sonnet-4",
    "model_routing.patching_mode":     "thinking",
    "model_routing.repair_primary":    "anthropic:claude-sonnet-4",
    "model_routing.repair_mode":       "thinking_max",
    "model_routing.doc_reviewer_primary":  "anthropic:claude-sonnet-4",
    "model_routing.doc_reviewer_mode":     "thinking",
    "model_routing.code_reviewer_primary":  "anthropic:claude-sonnet-4",
    "model_routing.code_reviewer_mode":     "thinking",
    # Aggressive speculative parallelism — burn compute to save cycles.
    "speculative.enabled":       True,
    "speculative.trigger":       "always",
    "speculative.cost_strategy": "gradient_low_to_high",
    # High budget headroom.
    "token_budget.hard_cap_usd": 50.0,
    # Safety: require operator approval before any write lands.
    "security.diff_approval_required": True,
}

PRESETS: dict[str, dict[str, Any]] = {
    "frugal":  FRUGAL_PRESET,
    "balanced": BALANCED_PRESET,
    "maximum_quality": MAXIMUM_QUALITY_PRESET,
}

# Human labels for the UI. Kept alongside the definitions so a new
# preset can't ship without a rendered name.
PRESET_LABELS: dict[str, dict[str, str]] = {
    "frugal": {
        "title": "Frugal",
        "tagline": "For $2 per session. Cheap remote model, tight budget, no speculative burn.",
    },
    "balanced": {
        "title": "Balanced",
        "tagline": "The shipped defaults. Reasonable cost, sensible speculation, no approval gates.",
    },
    "maximum_quality": {
        "title": "Maximum quality",
        "tagline": "Best model everywhere, always-on speculation, diff approval before every write.",
    },
}


# ---------------------------------------------------------------------------
# Overlay + diff helpers
# ---------------------------------------------------------------------------

def _walk_dotted(config: dict[str, Any], dotted: str):
    """Traverse ``dotted`` (e.g. ``token_budget.hard_cap_usd``) through
    ``config``. Returns ``(present, value)``. Mirrors the private
    helper in cli.py — duplicated here to avoid pulling in the whole
    cli module for a 6-line traversal."""
    parts = dotted.split(".")
    node: Any = config
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            return False, None
        node = node[p]
    return True, node


def _set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``dotted`` in ``config``, creating intermediate dicts along
    the way. Overwrites any non-dict intermediate (a config that
    has ``token_budget = "old-string"`` will get replaced with a
    dict when we set ``token_budget.hard_cap_usd``)."""
    parts = dotted.split(".")
    node = config
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


@dataclass
class PresetChange:
    """One dotted key that a preset would change."""

    key: str
    before: Any
    after: Any
    changed: bool = field(init=False)

    def __post_init__(self) -> None:
        self.changed = self.before != self.after


def preview_preset(
    config: dict[str, Any], preset_name: str,
) -> list[PresetChange]:
    """Return every dotted key the preset would touch, showing before
    vs after. The UI renders this as a small preview so the operator
    sees what's about to change before clicking Apply."""
    preset = PRESETS.get(preset_name)
    if preset is None:
        return []
    out: list[PresetChange] = []
    for key, new_value in preset.items():
        present, current = _walk_dotted(config, key)
        before = current if present else None
        out.append(PresetChange(key=key, before=before, after=new_value))
    return out


@dataclass
class ApplyPresetResult:
    """Outcome of :func:`apply_preset`."""

    ok: bool
    preset_name: str
    changed_keys: tuple[str, ...] = ()
    written_path: str = ""
    error: str = ""


def apply_preset(
    preset_name: str,
    *,
    config_path: str,
    validate: bool = True,
) -> ApplyPresetResult:
    """Load ``config_path``, overlay the named preset's dotted keys,
    (optionally) re-validate, and write it back atomically.

    The overlay is additive: only the preset's keys are touched;
    every other section stays exactly as-is. This makes the preset
    predictable — operators can layer Frugal on top of a heavily-
    customised config without losing their tuning."""
    if preset_name not in PRESETS:
        return ApplyPresetResult(
            ok=False, preset_name=preset_name,
            error=f"unknown preset {preset_name!r}; "
                  f"pick one of {sorted(PRESETS)}",
        )
    if not os.path.isfile(config_path):
        return ApplyPresetResult(
            ok=False, preset_name=preset_name,
            error=f"config not found at {config_path}",
        )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return ApplyPresetResult(
            ok=False, preset_name=preset_name,
            error=f"could not read config: {exc}",
        )
    if not isinstance(raw, dict):
        return ApplyPresetResult(
            ok=False, preset_name=preset_name,
            error="config root is not an object",
        )

    proposed = copy.deepcopy(raw)
    preview = preview_preset(proposed, preset_name)
    changed = [c.key for c in preview if c.changed]
    for change in preview:
        _set_dotted(proposed, change.key, change.after)

    if validate:
        try:
            # Strict validator rejects _comment keys — the shipped
            # config has many for operator documentation. Strip them
            # in the copy we send to validate_config_strict, then
            # write the un-stripped ``proposed`` so the comments
            # survive.
            from harness.cli import _strip_comments, validate_config_strict
            validate_config_strict(_strip_comments(proposed), source=config_path)
        except Exception as exc:  # noqa: BLE001
            return ApplyPresetResult(
                ok=False, preset_name=preset_name,
                changed_keys=tuple(changed),
                error=f"post-preset config failed validation: {exc}",
            )

    dirpath = os.path.dirname(os.path.abspath(config_path)) or "."
    fd, tmp = tempfile.mkstemp(dir=dirpath, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(proposed, f, indent=2, sort_keys=False)
            f.write("\n")
        os.replace(tmp, config_path)
    except OSError as exc:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return ApplyPresetResult(
            ok=False, preset_name=preset_name,
            changed_keys=tuple(changed),
            error=f"config write failed: {exc}",
        )

    logger.info(
        "[presets] Applied %r: %d key(s) changed → %s",
        preset_name, len(changed), config_path,
    )
    try:
        from harness.observability import emit_event
        emit_event(
            "preset_applied",
            preset=preset_name,
            changed_count=len(changed),
        )
    except Exception:  # noqa: BLE001
        pass
    return ApplyPresetResult(
        ok=True, preset_name=preset_name,
        changed_keys=tuple(changed),
        written_path=config_path,
    )
