"""Phase 1 regression: root now lands on /home (wizard placeholder)
instead of /status.

Home nav item is prepended so it shows first in the side nav; the
Home route resolves to a 200 with the placeholder card that Phase 3
replaces with the real wizard.
"""

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
            }
        }
    )


def test_root_redirects_to_home_not_status(tmp_path):
    status, ctype, body = dispatch(_cfg(tmp_path), "/")
    assert status == 302
    assert body.endswith("/home"), (
        f"Root must land on the wizard-first /home page. Body was: {body!r}"
    )


def test_home_route_renders_wizard(tmp_path):
    """Phase 3 replaced the Phase 1 stub with the real wizard. The
    wizard branches on config state; without a valid config it lands
    on the setup step, which is what happens by default on a bare
    tmp_path with no config."""
    status, ctype, body = dispatch(_cfg(tmp_path), "/home")
    assert status == 200
    assert ctype.startswith("text/html")
    # Either wizard step is fine; both are Phase-3 wizard content.
    assert (
        "Let's pick a provider" in body
        or "What do you want to build?" in body
    )


def test_home_nav_item_is_first_and_highlighted(tmp_path):
    """When active='home' the side nav highlights the Home item, and
    Home renders first in the nav order."""
    _, _, body = dispatch(_cfg(tmp_path), "/home")
    home_idx = body.find(">Home<")
    status_idx = body.find(">View Status<")
    assert home_idx != -1, "Home nav entry missing"
    assert status_idx != -1
    assert home_idx < status_idx, (
        "Home must appear before View Status in the side nav"
    )
    # The current-page marker class must be on the Home link.
    home_link_start = body.rfind("<a", 0, home_idx)
    home_link_open = body[home_link_start:home_idx]
    assert "bx--side-nav__link--current" in home_link_open


def test_home_route_uses_home_icon(tmp_path):
    _, _, body = dispatch(_cfg(tmp_path), "/home")
    # The sprite icon reference — sprite.svg id="i-home" is used via
    # `<use href="#i-home">`; that's the exact string emitted by _icon.
    assert "#i-home" in body


def test_status_route_still_reachable_directly(tmp_path):
    """Muscle memory: operators can still type /status to reach the
    old landing page even though / no longer redirects there."""
    status, _, body = dispatch(_cfg(tmp_path), "/status")
    assert status == 200
    assert "View Status" in body
