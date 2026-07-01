"""Phase 6 regression: dashboard preset routes."""

from __future__ import annotations

from harness.dashboard import DashboardConfig, dispatch


def _cfg(tmp_path):
    return DashboardConfig.from_config(
        {
            "dashboard": {
                "log_dir": str(tmp_path / "logs"),
                "metrics_dir": str(tmp_path / "metrics"),
                "memory_dir": str(tmp_path / "memory"),
                "repo_index_dir": str(tmp_path / "idx"),
                "schedule_db": str(tmp_path / "schedule.db"),
                "enabled": True,
                "writes_enabled": True,
            }
        }
    )


def test_config_ui_redirects_to_presets(tmp_path):
    status, ctype, body = dispatch(_cfg(tmp_path), "/config-ui")
    assert status == 302
    assert body.endswith("/config-ui/presets")


def test_presets_page_renders_three_cards(tmp_path):
    status, ctype, body = dispatch(_cfg(tmp_path), "/config-ui/presets")
    assert status == 200
    # The three preset titles from PRESET_LABELS.
    assert "Frugal" in body
    assert "Balanced" in body
    assert "Maximum quality" in body
    # Preview table cells.
    assert "would change" in body
    # Apply form + preset hidden field wired to Alpine state.
    assert 'action="/config-ui/preset/apply"' in body or \
           "action='/config-ui/preset/apply'" in body
    assert "name='preset'" in body or 'name="preset"' in body


def test_advanced_page_still_reachable(tmp_path):
    """The tree editor moved to /config-ui/advanced but must still
    render — power users depend on it."""
    status, ctype, body = dispatch(_cfg(tmp_path), "/config-ui/advanced")
    assert status == 200
    # Tree editor renders as the classic Configure Harness body.
    assert "Configure Harness" in body


def test_presets_page_alpine_component_uses_selected_state(tmp_path):
    """The chooser mutates one Alpine variable (`selected`). Ensures
    the diff-preview panels are gated on that same variable so only
    one preview shows at a time."""
    _, _, body = dispatch(_cfg(tmp_path), "/config-ui/presets")
    assert 'x-data="{ selected: ' in body
    assert "selected === 'frugal'" in body
    assert "selected === 'balanced'" in body
    assert "selected === 'maximum_quality'" in body
