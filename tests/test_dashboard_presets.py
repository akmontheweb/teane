"""Tests for the run-preset CRUD UI (Phase 5.1).

Confirms the read API the renderer consumes, the round-trip through
the dispatch path, and the deep-link convention (subcommand token
prepended to ``harness_args`` so the "Use" button can route back to
the right /run/<subcommand> page).
"""

from __future__ import annotations


def test_run_presets_page_empty_state(tmp_path):
    from harness.dashboard import DashboardConfig, dispatch
    cfg = DashboardConfig(
        token_env="",
        csrf_token_env="FAKE_CSRF",
        web_db_path=str(tmp_path / "web.db"),
    )
    _, _, body = dispatch(cfg, "/run/presets")
    assert "No presets saved yet" in body
    # Still a real page — nav chrome present.
    assert "Saved presets" in body


def test_run_presets_page_lists_saved_preset(tmp_path):
    """A preset saved through the storage layer shows up in the UI
    table with its subcommand badge and the correct deep-link URL on
    the Use button."""
    from harness.dashboard import DashboardConfig, dispatch
    from harness.web_state import save_run_preset

    db_path = str(tmp_path / "web.db")
    save_run_preset(
        db_path=db_path,
        name="nightly-rebuild",
        workspace="/srv/teane/nightly",
        prompt="rebuild from latest spec",
        harness_args=["build", "--git=true", "--cd-discovery=true"],
    )

    cfg = DashboardConfig(
        token_env="",
        csrf_token_env="FAKE_CSRF",
        web_db_path=db_path,
    )
    _, _, body = dispatch(cfg, "/run/presets")
    assert "nightly-rebuild" in body
    # Subcommand badge derived from harness_args[0].
    assert ">build<" in body  # tag content
    # Use button links to /run/build with the preset query param.
    assert "/run/build?preset=nightly-rebuild" in body
    # Delete form posts to /run/presets/delete.
    assert "/run/presets/delete" in body


def test_run_presets_page_handles_legacy_preset_without_subcommand(tmp_path):
    """A preset saved with empty harness_args (pre-convention) should
    render the row but render the Use button as a muted placeholder
    rather than a broken /run/ link."""
    from harness.dashboard import DashboardConfig, dispatch
    from harness.web_state import save_run_preset

    db_path = str(tmp_path / "web.db")
    save_run_preset(
        db_path=db_path,
        name="legacy",
        workspace="/srv/x",
        prompt="x",
        harness_args=[],
    )

    cfg = DashboardConfig(
        token_env="",
        csrf_token_env="FAKE_CSRF",
        web_db_path=db_path,
    )
    _, _, body = dispatch(cfg, "/run/presets")
    assert "legacy" in body
    # No invalid /run/ link generated.
    assert "/run/?preset=" not in body
