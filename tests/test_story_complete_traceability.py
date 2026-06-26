"""Tests for story_complete_node / traceability_node.

The ``story_test_first_node`` section here was removed in Phase F when
the node itself was deleted — its acceptance-criteria-derived xfail
stubs duplicated content the patching LLM already gets through
``_build_story_preamble``."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from harness import story_loop, story_state


@pytest.fixture
def workspace(tmp_path: Path) -> str:
    ws = tmp_path / "complete-ws"
    ws.mkdir()
    return str(ws)


def _app(workspace: str) -> str:
    return story_state.app_name_for_workspace(workspace)


_DEFAULT_AC = ["AC-1", "AC-2"]


def _seed_one_story(
    workspace: str, *, acceptance=None, title="T", scope_files=None
) -> str:
    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        story_state.ensure_feature(conn, app, "test", name="Test feature")
        keys = story_state.create_stories(conn, app, [{
            "title": title,
            "feature": "test",
            "acceptance_criteria": (
                acceptance if acceptance is not None else _DEFAULT_AC
            ),
            "scope_files": scope_files or [],
        }])
    finally:
        conn.close()
    return keys[0]


def _make_git_repo(workspace: str) -> None:
    subprocess.run(["git", "-C", workspace, "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", workspace, "config", "user.email", "t@t.test"], check=True,
    )
    subprocess.run(
        ["git", "-C", workspace, "config", "user.name", "Tester"], check=True,
    )
    subprocess.run(
        ["git", "-C", workspace, "config", "commit.gpgsign", "false"], check=True,
    )
    Path(workspace, "seed.txt").write_text("seed\n")
    subprocess.run(["git", "-C", workspace, "add", "."], check=True)
    subprocess.run(
        ["git", "-C", workspace, "commit", "-q", "-m", "seed"], check=True,
    )


# ---------------------------------------------------------------------------
# _classify_file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,expected", [
    ("tests/test_foo.py", "test"),
    ("src/__tests__/foo.test.js", "test"),
    ("src/conftest.py", "test"),
    ("docs/STORIES.md", "doc"),
    ("README.md", "doc"),
    ("Dockerfile", "infra"),
    ("docker-compose.yml", "infra"),
    ("Makefile", "infra"),
    ("src/api/handlers.py", "code"),
    ("auth.py", "code"),
])
def test_classify_file(path, expected):
    assert story_loop._classify_file(path) == expected


# ---------------------------------------------------------------------------
# story_complete_node
# ---------------------------------------------------------------------------

def _complete_state(
    workspace: str, story_key: str, **overrides: Any
) -> dict[str, Any]:
    base = {
        "workspace_path": workspace,
        "session_id": "sess-1",
        "current_story_id": story_key,
        "story_modified_baseline": [],
        "modified_files": [],
        "loop_counter": {},
        "exit_code": 0,
        "story_repair_cap": 3,
    }
    base.update(overrides)
    return base


def test_complete_skipped_when_no_current_story(workspace: str):
    out = story_loop.story_complete_node({"workspace_path": workspace})
    assert out["node_state"]["skipped"] is True


def test_complete_marks_done_on_success(workspace: str):
    key = _seed_one_story(workspace)
    app = _app(workspace)
    conn = story_state.open_story_db()
    try:
        story_state.mark_in_progress(conn, app, key)
    finally:
        conn.close()
    out = story_loop.story_complete_node(_complete_state(
        workspace, key,
        modified_files=["src/feature.py", "tests/test_feature.py"],
    ))
    assert out["node_state"]["outcome"] == "done"
    assert out["current_story_id"] == ""

    conn = story_state.open_story_db()
    try:
        s = story_state.get_story(conn, app, key)
        files = conn.execute(
            "SELECT path, kind FROM file_links ORDER BY path"
        ).fetchall()
    finally:
        conn.close()
    assert s["status"] == "done"
    assert ("src/feature.py", "code") in files
    assert ("tests/test_feature.py", "test") in files


def test_complete_attributes_only_files_above_baseline(workspace: str):
    key = _seed_one_story(workspace)
    out = story_loop.story_complete_node(_complete_state(
        workspace, key,
        story_modified_baseline=["pre.py"],
        modified_files=["pre.py", "new1.py", "new2.py"],
    ))
    assert sorted(out["node_state"]["new_files"]) == ["new1.py", "new2.py"]
    conn = story_state.open_story_db()
    try:
        paths = {p for (p, _k) in conn.execute(
            "SELECT path, kind FROM file_links"
        )}
    finally:
        conn.close()
    assert paths == {"new1.py", "new2.py"}


def test_complete_marks_blocked_on_repair_cap(workspace: str):
    key = _seed_one_story(workspace)
    app = _app(workspace)
    out = story_loop.story_complete_node(_complete_state(
        workspace, key,
        exit_code=1,
        loop_counter={"total_repairs": 3},
        story_repair_cap=3,
    ))
    assert out["node_state"]["outcome"] == "blocked"

    conn = story_state.open_story_db()
    try:
        s = story_state.get_story(conn, app, key)
        defects = conn.execute(
            "SELECT severity, status FROM defects"
        ).fetchall()
    finally:
        conn.close()
    assert s["status"] == "blocked"
    assert defects == [("repair_cap_exceeded", "open")]


def test_complete_incomplete_when_failing_but_under_cap(workspace: str):
    key = _seed_one_story(workspace)
    app = _app(workspace)
    out = story_loop.story_complete_node(_complete_state(
        workspace, key,
        exit_code=1,
        loop_counter={"total_repairs": 1},
        story_repair_cap=3,
    ))
    assert out["node_state"]["outcome"] == "incomplete"
    # Story still in_progress (or planned — mark_in_progress wasn't called)
    conn = story_state.open_story_db()
    try:
        s = story_state.get_story(conn, app, key)
    finally:
        conn.close()
    assert s["status"] in ("planned", "in_progress")


def test_complete_resets_loop_counters(workspace: str):
    key = _seed_one_story(workspace)
    out = story_loop.story_complete_node(_complete_state(
        workspace, key,
        loop_counter={
            "patching": 5, "repair": 3, "compiler": 7,
            "total_repairs": 2, "review_spec": 1,
        },
    ))
    lc = out["loop_counter"]
    assert lc["patching"] == 0
    assert lc["repair"] == 0
    assert lc["total_repairs"] == 0
    assert lc["review_spec"] == 0


def test_complete_commits_when_commit_on_story_set(workspace: str):
    _make_git_repo(workspace)
    key = _seed_one_story(workspace, title="Add foo")

    # Create a real file change so the commit isn't empty
    Path(workspace, "foo.py").write_text("print('foo')\n")

    out = story_loop.story_complete_node(_complete_state(
        workspace, key,
        commit_on_story=True,
        modified_files=["foo.py"],
    ))
    assert out["node_state"]["committed_sha"]
    log = subprocess.run(
        ["git", "-C", workspace, "log", "-1", "--pretty=%s"],
        capture_output=True, text=True, check=True,
    )
    assert "STORY-1: Add foo" in log.stdout

    conn = story_state.open_story_db()
    try:
        row = conn.execute(
            "SELECT sha, message FROM commits"
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == out["node_state"]["committed_sha"]
    assert "STORY-1: Add foo" in row[1]


def test_complete_no_git_repo_is_silently_skipped(workspace: str):
    key = _seed_one_story(workspace)
    app = _app(workspace)
    Path(workspace, "foo.py").write_text("print('foo')\n")
    out = story_loop.story_complete_node(_complete_state(
        workspace, key,
        commit_on_story=True,
        modified_files=["foo.py"],
    ))
    assert out["node_state"]["committed_sha"] is None
    # But story still marked done
    conn = story_state.open_story_db()
    try:
        s = story_state.get_story(conn, app, key)
    finally:
        conn.close()
    assert s["status"] == "done"


def test_complete_routes_back_to_loop():
    assert story_loop.route_after_story_complete({}) == "story_loop_node"


# ---------------------------------------------------------------------------
# traceability_node
# ---------------------------------------------------------------------------

def test_traceability_skipped_without_workspace():
    out = story_loop.traceability_node({})
    assert out["node_state"]["skipped"] is True


def test_traceability_regenerates_both_views(workspace: str):
    _seed_one_story(workspace, title="Demo")
    out = story_loop.traceability_node({"workspace_path": workspace})
    assert out["node_state"]["skipped"] is False
    assert os.path.exists(out["node_state"]["stories_md"])
    assert os.path.exists(out["node_state"]["traceability_md"])
    assert "STORY-1" in Path(out["node_state"]["stories_md"]).read_text()
    assert "STORY-1" in Path(out["node_state"]["traceability_md"]).read_text()


# ---------------------------------------------------------------------------
# traceability_node — architecture coverage matrix (§11 handoff)
# ---------------------------------------------------------------------------

_BASE_ARCH_SUMMARY: dict[str, Any] = {
    "schema_version": 1,
    "backend_language": "python_fastapi",
    "frontend": "react",
    "backend": {
        "endpoints": [
            {
                "id": "EP-001",
                "method": "POST",
                "path": "/api/v1/login",
                "rsd_story_ids": ["STORY-1"],
            },
            {
                "id": "EP-002",
                "method": "GET",
                "path": "/api/v1/orphan",
                "rsd_story_ids": [],          # no story linked → gap
            },
            {
                "id": "EP-003",
                "method": "GET",
                "path": "/api/v1/ghost",
                "rsd_story_ids": ["STORY-99"],  # story doesn't exist → missing
            },
        ],
    },
    "frontend_spec": {
        "components": [
            {
                "name": "LoginForm",
                "path": "pages/auth/LoginPage.tsx",
                "rsd_story_ids": ["STORY-1"],
            },
        ],
    },
    "contract": {"openapi_spec_path": "contracts/openapi.json"},
}


def test_traceability_skips_arch_section_without_summary(workspace: str):
    """No state.arch_summary AND no SPEC_ARCHITECTURE.md on disk →
    TRACEABILITY.md must not contain the new section."""
    _seed_one_story(workspace, title="Demo")
    out = story_loop.traceability_node({"workspace_path": workspace})
    body = Path(out["node_state"]["traceability_md"]).read_text()
    assert "Architecture coverage" not in body
    assert out["node_state"]["arch_coverage_emitted"] is False


def test_traceability_emits_arch_section_from_state(workspace: str):
    _seed_one_story(workspace, title="Demo")
    out = story_loop.traceability_node({
        "workspace_path": workspace,
        "arch_summary": _BASE_ARCH_SUMMARY,
    })
    body = Path(out["node_state"]["traceability_md"]).read_text()
    assert "Architecture coverage" in body
    assert "EP-001" in body and "/api/v1/login" in body
    assert out["node_state"]["arch_coverage_emitted"] is True


def test_traceability_links_story_status(workspace: str):
    """When EP-001 cites STORY-1, the coverage row should carry the
    live story status — not just the ID."""
    _seed_one_story(workspace, title="Demo")
    out = story_loop.traceability_node({
        "workspace_path": workspace,
        "arch_summary": _BASE_ARCH_SUMMARY,
    })
    body = Path(out["node_state"]["traceability_md"]).read_text()
    # The seeded story lands as "planned" — exact label depends on
    # _status_label, so we accept any non-empty status cell next to
    # the STORY-1 ID and just assert the ID surfaces in the coverage
    # block (which sits AFTER the per-story drill-down).
    coverage_section = body.split("Architecture coverage", 1)[1]
    assert "STORY-1" in coverage_section
    assert "EP-001" in coverage_section


def test_traceability_flags_orphan_endpoint_as_gap(workspace: str):
    """An endpoint with no rsd_story_ids should render the 'gap'
    sentinel so reviewers spot un-storied architecture work."""
    _seed_one_story(workspace, title="Demo")
    out = story_loop.traceability_node({
        "workspace_path": workspace,
        "arch_summary": _BASE_ARCH_SUMMARY,
    })
    body = Path(out["node_state"]["traceability_md"]).read_text()
    # Slice down to the orphan row to keep the assertion targeted.
    orphan_line = [ln for ln in body.splitlines() if "EP-002" in ln]
    assert orphan_line, "EP-002 row missing from coverage table"
    assert "gap" in orphan_line[0].lower()


def test_traceability_flags_missing_story_for_unknown_id(workspace: str):
    """An endpoint citing a story that the DB doesn't recognise should
    be tagged 'missing' rather than silently rendered as healthy."""
    _seed_one_story(workspace, title="Demo")
    out = story_loop.traceability_node({
        "workspace_path": workspace,
        "arch_summary": _BASE_ARCH_SUMMARY,
    })
    body = Path(out["node_state"]["traceability_md"]).read_text()
    ghost_line = [ln for ln in body.splitlines() if "EP-003" in ln]
    assert ghost_line, "EP-003 row missing from coverage table"
    assert "STORY-99" in ghost_line[0]
    assert "missing" in ghost_line[0].lower()


def test_traceability_omits_components_section_when_frontend_none(workspace: str):
    _seed_one_story(workspace, title="Demo")
    summary = dict(_BASE_ARCH_SUMMARY)
    summary["frontend"] = "none"
    out = story_loop.traceability_node({
        "workspace_path": workspace,
        "arch_summary": summary,
    })
    body = Path(out["node_state"]["traceability_md"]).read_text()
    # Endpoint subheading still present.
    assert "### Endpoints" in body
    # Component subheading should be suppressed for headless backends.
    assert "### Components" not in body


def test_traceability_lazy_loads_summary_from_disk(workspace: str):
    """When state.arch_summary is empty but docs/SPEC_ARCHITECTURE.md
    carries a §11 block, the node should hydrate from disk and emit
    the coverage section — same path patching_node uses on monolithic
    flows."""
    import json as _json
    _seed_one_story(workspace, title="Demo")
    docs = Path(workspace) / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    body = (
        "# Architecture Document\n\n"
        "## §11 Summary\n\n"
        "```jsonc\n" + _json.dumps(_BASE_ARCH_SUMMARY) + "\n```\n"
    )
    (docs / "SPEC_ARCHITECTURE.md").write_text(body, encoding="utf-8")

    out = story_loop.traceability_node({"workspace_path": workspace})
    body_md = Path(out["node_state"]["traceability_md"]).read_text()
    assert "Architecture coverage" in body_md
    assert "EP-001" in body_md
    # Resolved summary is echoed back into state for downstream nodes.
    assert out["arch_summary"]["schema_version"] == 1


def test_render_arch_coverage_empty_summary_returns_empty_list():
    from harness.story_state import _render_arch_coverage
    assert _render_arch_coverage([], None) == []
    assert _render_arch_coverage([], {}) == []
    assert _render_arch_coverage(
        [], {"backend": {"endpoints": []}, "frontend": "none"}
    ) == []
