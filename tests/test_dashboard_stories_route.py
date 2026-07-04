"""Route + view coverage for the /stories browser.

The stories browser is a workspace-scoped read view of
``~/.harness/state.db``: an index listing every workspace with story
data, and per-workspace pages showing features (with per-status
rollups), recent batches, and open + resolved defects.

The autouse ``isolated_state_db`` fixture in conftest.py points
``TEANE_STATE_DB`` at a per-test file so the real
``~/.harness/state.db`` is never touched.
"""

from __future__ import annotations

from pathlib import Path

from harness.dashboard import DashboardConfig, dispatch
from harness.story_state import (
    app_name_for_workspace,
    create_features,
    create_stories,
    mark_done,
    mark_in_progress,
    open_story_db,
    record_defect,
    start_batch,
)


def _cfg(tmp_path: Path) -> DashboardConfig:
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


def _seed_workspace(tmp_path: Path, name: str = "ws-alpha") -> str:
    """Create a workspace folder and populate the isolated state.db
    with two features, three stories, one batch and two defects.
    Returns the workspace basename (which is also the /stories/<ws>
    URL segment)."""
    workspace = tmp_path / name
    workspace.mkdir()
    app = app_name_for_workspace(str(workspace))
    conn = open_story_db(str(workspace))
    try:
        create_features(conn, app, [
            {"feature_key": "auth", "name": "Authentication"},
            {"feature_key": "billing", "name": "Billing"},
        ])
        create_stories(conn, app, [
            {"title": "Login form",  "feature": "auth"},
            {"title": "Logout link", "feature": "auth"},
            {"title": "Invoice PDF", "feature": "billing"},
        ])
        mark_done(conn, app, "STORY-001")
        mark_in_progress(conn, app, "STORY-002")
        # STORY-003 stays planned.
        start_batch(
            conn, app, "sess-abc", ["STORY-001", "STORY-002"],
        )
        record_defect(
            conn, workspace=app, story_key="STORY-002",
            session_id="sess-abc", severity="high",
            summary="logout link 404s in Safari",
        )
        record_defect(
            conn, workspace=app, story_key=None,
            session_id="sess-abc", severity="low",
            summary="cosmetic: header alignment",
        )
    finally:
        conn.close()
    return app


# ---------------------------------------------------------------------------
# Route wiring — regex table + dispatch()
# ---------------------------------------------------------------------------

def test_stories_index_renders_empty_state_when_no_state_db(tmp_path):
    """Fresh install: no state.db, no story data. The page must still
    render 200 with a friendly empty-state — not 500."""
    status, ctype, body = dispatch(_cfg(tmp_path), "/stories")
    assert status == 200
    assert ctype.startswith("text/html")
    assert "Stories" in body
    assert "No story data yet" in body


def test_stories_index_lists_workspaces_with_data(tmp_path):
    app = _seed_workspace(tmp_path)
    status, _, body = dispatch(_cfg(tmp_path), "/stories")
    assert status == 200
    # Workspace surfaces as a link into the detail page.
    assert f"/stories/{app}" in body
    # Rollup: 1 done out of 3 stories (STORY-001 done, STORY-002 in flight,
    # STORY-003 planned). Two features, one batch, one open defect
    # (STORY-002 tagged; the orphan defect also counts as open).
    assert "1 / 3" in body  # done / total
    assert "Open defects" in body


def test_stories_workspace_page_shows_features_batches_defects(tmp_path):
    app = _seed_workspace(tmp_path)
    status, ctype, body = dispatch(_cfg(tmp_path), f"/stories/{app}")
    assert status == 200
    assert ctype.startswith("text/html")
    # Features card: both features listed with names.
    assert "Authentication" in body
    assert "Billing" in body
    # Feature rollup badges reflect statuses.
    assert "1 done" in body
    assert "1 in flight" in body
    # Stories card content: story keys and titles show inside expanded
    # feature <details> blocks.
    assert "STORY-001" in body
    assert "Login form" in body
    # Batches card: session id linked back to /sessions.
    assert "/sessions/sess-abc" in body
    # Defects card: open defects surfaced up top.
    assert "logout link 404s in Safari" in body
    # Breadcrumb ties back to the index.
    assert "href='/stories'" in body or 'href="/stories"' in body


def test_stories_workspace_page_unknown_slug_renders_not_found_card(tmp_path):
    """A URL for a workspace that has never produced story data must
    render a friendly not-found card and preserve the breadcrumb, not
    500 out.
    """
    status, _, body = dispatch(_cfg(tmp_path), "/stories/does-not-exist")
    assert status == 200
    assert "Workspace not found" in body
    assert "/stories" in body  # back-to-index link


def test_stories_workspace_page_still_renders_when_seed_workspace_exists(tmp_path):
    """Requesting a *different* workspace slug when other workspaces
    have data should surface the not-found card for the requested slug
    — not silently fall back to a random workspace's data."""
    seeded = _seed_workspace(tmp_path)
    status, _, body = dispatch(_cfg(tmp_path), "/stories/some-other-ws")
    assert status == 200
    assert "Workspace not found" in body
    # Guard against leaking the seeded workspace's data into the
    # not-found response.
    assert "Authentication" not in body
    assert seeded not in body or seeded == "some-other-ws"


def test_stories_route_rejects_path_traversal(tmp_path):
    """The /stories/<ws> regex must not match slashes or dots; a
    traversal-shaped URL falls through to the router's 404."""
    status, _, body = dispatch(_cfg(tmp_path), "/stories/../etc/passwd")
    assert status == 404


def test_dashboards_landing_advertises_stories_tile(tmp_path):
    """The /dashboards landing page should surface a tile pointing at
    /stories so operators can discover the browser without knowing
    the URL."""
    status, _, body = dispatch(_cfg(tmp_path), "/dashboards")
    assert status == 200
    assert "/stories" in body
    assert "Stories" in body
