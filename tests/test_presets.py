"""Phase 6 regression: named config presets + preview + apply."""

from __future__ import annotations

import json
import os
import shutil


from harness.presets import (
    BALANCED_PRESET,
    FRUGAL_PRESET,
    MAXIMUM_QUALITY_PRESET,
    PRESET_LABELS,
    PRESETS,
    apply_preset,
    preview_preset,
    _set_dotted,
    _walk_dotted,
)


def _shipped_config_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "config.json",
    )


# ---------------------------------------------------------------------------
# Preset catalogue completeness
# ---------------------------------------------------------------------------

def test_every_preset_has_labels():
    for slug in PRESETS:
        assert slug in PRESET_LABELS, f"missing label for {slug!r}"
        meta = PRESET_LABELS[slug]
        assert meta["title"]
        assert meta["tagline"]


def test_frugal_caps_hard_cap_under_balanced():
    """The frugal preset must always leave the user with a lower
    hard cap than balanced — otherwise the name is a lie."""
    frugal = FRUGAL_PRESET["token_budget.hard_cap_usd"]
    balanced = BALANCED_PRESET["token_budget.hard_cap_usd"]
    assert frugal < balanced


def test_maximum_quality_enables_diff_approval():
    """Diff approval is the flagship safety feature — must be on
    under Maximum Quality."""
    assert MAXIMUM_QUALITY_PRESET["security.diff_approval_required"] is True


def test_balanced_leaves_diff_approval_off():
    """Balanced is the "shipped default" preset; the shipped config
    does not force diff approval, so balanced must not flip it on."""
    assert BALANCED_PRESET["security.diff_approval_required"] is False


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def test_set_dotted_creates_intermediate_dicts():
    cfg = {}
    _set_dotted(cfg, "a.b.c", 42)
    assert cfg == {"a": {"b": {"c": 42}}}


def test_walk_dotted_finds_and_reports_missing():
    cfg = {"a": {"b": 1}}
    assert _walk_dotted(cfg, "a.b") == (True, 1)
    assert _walk_dotted(cfg, "a.c") == (False, None)
    assert _walk_dotted(cfg, "x.y.z") == (False, None)


# ---------------------------------------------------------------------------
# preview_preset
# ---------------------------------------------------------------------------

def test_preview_returns_before_and_after_for_every_touched_key():
    cfg = {"token_budget": {"hard_cap_usd": 10.0}, "speculative": {"enabled": True}}
    preview = preview_preset(cfg, "frugal")
    keys = {c.key for c in preview}
    assert "token_budget.hard_cap_usd" in keys
    # Every entry has a `changed` marker.
    hard_cap = next(c for c in preview if c.key == "token_budget.hard_cap_usd")
    assert hard_cap.before == 10.0
    assert hard_cap.after == FRUGAL_PRESET["token_budget.hard_cap_usd"]
    assert hard_cap.changed is True


def test_preview_marks_unchanged_when_value_already_matches():
    cfg = {"token_budget": {"hard_cap_usd": BALANCED_PRESET["token_budget.hard_cap_usd"]}}
    preview = preview_preset(cfg, "balanced")
    hard_cap = next(c for c in preview if c.key == "token_budget.hard_cap_usd")
    assert hard_cap.changed is False


def test_preview_unknown_preset_returns_empty():
    assert preview_preset({}, "nonexistent") == []


# ---------------------------------------------------------------------------
# apply_preset — end-to-end, writing a real config file
# ---------------------------------------------------------------------------

def test_apply_writes_only_named_keys_leaves_rest_intact(tmp_path, monkeypatch):
    """Overlay must be additive — sections not touched by the preset
    (models registry, mcp block, dashboard block, etc.) survive."""
    cfg_path = tmp_path / "config.json"
    shutil.copy(_shipped_config_path(), cfg_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")

    before = json.loads(cfg_path.read_text())
    before_models = before["models"]  # unrelated section

    result = apply_preset("balanced", config_path=str(cfg_path))
    assert result.ok, result.error
    after = json.loads(cfg_path.read_text())
    # The models registry is preserved byte-for-byte.
    assert after["models"] == before_models


def test_apply_flips_diff_approval_under_maximum_quality(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    shutil.copy(_shipped_config_path(), cfg_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")

    result = apply_preset("maximum_quality", config_path=str(cfg_path))
    assert result.ok, result.error
    written = json.loads(cfg_path.read_text())
    assert written["security"]["diff_approval_required"] is True


def test_apply_rejects_unknown_preset(tmp_path):
    cfg_path = tmp_path / "config.json"
    shutil.copy(_shipped_config_path(), cfg_path)
    result = apply_preset("does-not-exist", config_path=str(cfg_path))
    assert not result.ok
    assert "unknown preset" in result.error


def test_apply_missing_config_file_reports_gracefully(tmp_path):
    result = apply_preset("frugal", config_path=str(tmp_path / "nope.json"))
    assert not result.ok
    assert "config not found" in result.error


def test_apply_reports_changed_keys(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    shutil.copy(_shipped_config_path(), cfg_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    result = apply_preset("frugal", config_path=str(cfg_path))
    assert result.ok, result.error
    assert "token_budget.hard_cap_usd" in result.changed_keys
    assert result.written_path == str(cfg_path)
