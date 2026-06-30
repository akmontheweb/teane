"""Tests for the workspace-status + CR-DEFECT file-serving module.

The path-traversal defenses on ``workspace_file_payload`` are the
load-bearing piece here: a buggy filter would let an authed operator
read arbitrary files under the workspace, or worse, anywhere on the
filesystem via ``..`` escapes. The tests below confirm both layers
of the defense (realpath containment + basename allowlist) reject
the classic attacks while the happy-path bytes still flow through.
"""

from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# CR-DEFECT file serving
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace_with_defect(tmp_path):
    """Build a workspace layout that mimics what ``teane test`` emits."""
    ws = tmp_path / "ws"
    cr_dir = ws / "change_requests" / "CR-DEFECT-20260601-login-deadb1"
    cr_dir.mkdir(parents=True)
    (cr_dir / "narrative.txt").write_text("login redirect failed\n")
    (cr_dir / "screenshot.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (cr_dir / "cluster_evidence.json").write_text(json.dumps({"a": 1}))
    # Decoy that lives in change_requests/ but is NOT under a
    # CR-DEFECT-* subdirectory. Even with the realpath check passing
    # (it's still under change_requests/), the directory-prefix check
    # must reject it.
    decoy_cr = ws / "change_requests" / "CR-7" / "narrative.txt"
    decoy_cr.parent.mkdir(parents=True)
    decoy_cr.write_text("real CR, not a defect")
    # Sensitive file outside change_requests/ to confirm the realpath
    # check catches a `..` escape.
    secret = ws / ".env"
    secret.write_text("API_KEY=hunter2")
    return ws


def test_workspace_file_payload_serves_allowlisted_basename(workspace_with_defect):
    from harness.dashboard_workspace import workspace_file_payload
    relpath = (
        "change_requests/CR-DEFECT-20260601-login-deadb1/narrative.txt"
    )
    status, ctype, body = workspace_file_payload(
        str(workspace_with_defect), relpath,
    )
    assert status == 200
    assert ctype.startswith("text/plain")
    assert b"login redirect failed" in body


def test_workspace_file_payload_serves_binary_attachment(workspace_with_defect):
    from harness.dashboard_workspace import workspace_file_payload
    relpath = (
        "change_requests/CR-DEFECT-20260601-login-deadb1/screenshot.png"
    )
    status, ctype, body = workspace_file_payload(
        str(workspace_with_defect), relpath,
    )
    assert status == 200
    assert ctype.startswith("image/")
    assert body.startswith(b"\x89PNG")


def test_workspace_file_payload_rejects_non_allowlisted_basename(
    workspace_with_defect,
):
    """A basename that isn't in the allowlist (random file inside a
    CR-DEFECT directory) is rejected even though everything else looks
    legitimate. Defense in depth: the directory layout filter alone
    wouldn't catch a malicious in-place rename."""
    bogus = workspace_with_defect / "change_requests" / "CR-DEFECT-20260601-login-deadb1" / "evil.sh"
    bogus.write_text("rm -rf /\n")
    from harness.dashboard_workspace import workspace_file_payload
    relpath = "change_requests/CR-DEFECT-20260601-login-deadb1/evil.sh"
    status, ctype, body = workspace_file_payload(
        str(workspace_with_defect), relpath,
    )
    assert status == 404


def test_workspace_file_payload_rejects_traversal_with_dotdot(
    workspace_with_defect,
):
    """The classic ``../../etc/passwd`` escape attempt — the realpath
    check must catch this even before the basename allowlist."""
    from harness.dashboard_workspace import workspace_file_payload
    status, ctype, body = workspace_file_payload(
        str(workspace_with_defect),
        "../../../etc/passwd",
    )
    assert status == 404


def test_workspace_file_payload_rejects_non_defect_cr(workspace_with_defect):
    """A change_requests/ sibling that isn't a CR-DEFECT-* directory
    is rejected. The dashboard's defect-serving endpoint must NOT
    leak operator-authored CR narratives or other artefacts."""
    from harness.dashboard_workspace import workspace_file_payload
    status, ctype, body = workspace_file_payload(
        str(workspace_with_defect),
        "change_requests/CR-7/narrative.txt",
    )
    assert status == 404


def test_workspace_file_payload_rejects_absolute_relpath(
    workspace_with_defect,
):
    """Operator query parameter must be relative — absolute paths land
    outside the workspace and are rejected outright."""
    from harness.dashboard_workspace import workspace_file_payload
    status, ctype, body = workspace_file_payload(
        str(workspace_with_defect),
        "/etc/passwd",
    )
    assert status == 404


def test_workspace_file_payload_rejects_missing_workspace(tmp_path):
    from harness.dashboard_workspace import workspace_file_payload
    status, ctype, body = workspace_file_payload(
        str(tmp_path / "does-not-exist"),
        "change_requests/CR-DEFECT-x/narrative.txt",
    )
    assert status == 404


# ---------------------------------------------------------------------------
# Workspace status widget
# ---------------------------------------------------------------------------


def test_render_workspace_status_widget_lists_three_flow_tiles(tmp_path):
    """The widget renders one tile per tracked flow + the test-gate
    tile. With no markers yet they all render the "no clean
    completion" empty-state but the structure stays consistent."""
    from harness.dashboard_workspace import render_workspace_status_widget
    ws = tmp_path / "ws"
    ws.mkdir()
    html = render_workspace_status_widget(str(ws))
    assert "Workspace status" in html
    assert "Build" in html and "Patch" in html and "Deploy" in html
    assert "Test" in html
    # No markers → muted state, no run-test button.
    assert "Run test" not in html


def test_render_workspace_status_widget_renders_completed_marker(tmp_path):
    """When a marker file exists, the corresponding tile shows the
    session id + completion timestamp instead of the empty-state."""
    from harness.dashboard_workspace import render_workspace_status_widget
    ws = tmp_path / "ws"
    teane_dir = ws / ".teane"
    teane_dir.mkdir(parents=True)
    (teane_dir / "last_build.json").write_text(json.dumps({
        "flow": "build",
        "session_id": "abc123",
        "completed_at": "2026-06-30T12:00:00Z",
        "exit_code": 0,
    }))
    html = render_workspace_status_widget(str(ws))
    assert "abc123" in html
    assert "2026-06-30" in html


# ---------------------------------------------------------------------------
# list_cr_defects
# ---------------------------------------------------------------------------


def test_list_cr_defects_finds_only_defect_dirs(workspace_with_defect):
    """Only ``CR-DEFECT-*`` subdirectories of ``change_requests/`` are
    returned. The CR-7 directory in the fixture must NOT appear."""
    from harness.dashboard_workspace import list_cr_defects
    names = list_cr_defects(str(workspace_with_defect))
    assert names == ["CR-DEFECT-20260601-login-deadb1"]
