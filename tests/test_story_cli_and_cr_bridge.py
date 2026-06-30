"""Tests for the CLI story-mode flags + CR→STORY bridge (step 8)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from harness import story_state


# ---------------------------------------------------------------------------
# CLI argparse — flags exist and parse correctly
# ---------------------------------------------------------------------------

def _parse_run_args(extra: list[str]) -> Any:
    from harness.cli import build_parser
    parser = build_parser()
    # `teane run` is gone; the agile-mode flag now lives on `teane build`
    # and `teane patch`. The story-tuning flags (--story-batch-size,
    # --commit-on-story, --story-repair-cap) were dropped from the CLI
    # and migrated to config.json's agile_defaults block.
    return parser.parse_args(["build", "-w", "/tmp/x", "-p", "do a thing", *extra])


def test_build_parser_exposes_agile_flag():
    args = _parse_run_args(["--agile", "true"])
    assert args.agile is True


def test_build_parser_agile_defaults_to_none_sentinel():
    """Default is the None sentinel — the cmd_build handler resolves it
    against config["agile"] (and for `patch`, against the workspace's
    .teane/state.db) before passing it through to the graph."""
    args = _parse_run_args([])
    assert args.agile is None


def test_build_parser_rejects_legacy_stories_flag():
    """`--stories` is gone — argparse must surface the rename loudly so
    operators with scripts using the old flag notice."""
    import pytest as _pytest
    with _pytest.raises(SystemExit):
        _parse_run_args(["--stories", "true"])


def test_build_parser_rejects_legacy_story_tuning_flags():
    """The three per-knob flags moved into config.json's agile_defaults
    block. Argparse must reject the old CLI forms."""
    import pytest as _pytest
    for flag in ("--story-batch-size", "--commit-on-story", "--story-repair-cap"):
        with _pytest.raises(SystemExit):
            _parse_run_args([flag, "5"])


def test_resolve_agile_args_pulls_defaults_from_config():
    """_resolve_agile_args reads agile_defaults from config.json and pins
    them onto the args namespace so cmd_run threads them into the graph."""
    from harness.cli import _resolve_agile_args
    import argparse as _argparse
    args = _argparse.Namespace(agile=True)
    _resolve_agile_args(args, config={
        "agile_defaults": {"batch_size": 9, "commit_on_story": True, "repair_cap": 6},
    }, workspace_path="/tmp/x", flow="build")
    assert args.decomposition_enabled is True
    assert args.story_batch_size == 9
    assert args.commit_on_story is True
    assert args.story_repair_cap == 6


def test_resolve_agile_args_falls_back_to_hard_defaults():
    """When agile_defaults is absent, fall through to (5, False, 3)."""
    from harness.cli import _resolve_agile_args
    import argparse as _argparse
    args = _argparse.Namespace(agile=True)
    _resolve_agile_args(args, config={}, workspace_path="/tmp/x", flow="build")
    assert args.story_batch_size == 5
    assert args.commit_on_story is False
    assert args.story_repair_cap == 3


def test_cmd_patch_agile_forces_install_doc_true(tmp_path, monkeypatch):
    """Phase 6c: agile patches must enable installation_doc_node so the
    end-of-session traceability audit fires. Non-agile patches keep
    install_doc=False (legacy behavior); an explicit --install-doc=false
    is respected (operator override)."""
    import argparse as _argparse
    import asyncio
    from harness import cli as cli_mod

    # Stub cmd_run so we just observe the args namespace cmd_patch
    # hands off without actually executing the graph.
    captured: dict[str, Any] = {}

    async def _fake_run(a):
        captured["args"] = a
        return 0

    monkeypatch.setattr(cli_mod, "cmd_run", _fake_run)
    monkeypatch.chdir(tmp_path)

    # Agile patch — install_doc was unset; cmd_patch should force True.
    args = _argparse.Namespace(
        workspace=str(tmp_path), agile=True, generate_specs=None,
        install_doc=None,
    )
    asyncio.run(cli_mod.cmd_patch(args))
    assert captured["args"].decomposition_enabled is True
    assert captured["args"].install_doc is True

    # Non-agile patch — install_doc stays None/False.
    captured.clear()
    args = _argparse.Namespace(
        workspace=str(tmp_path), agile=False, generate_specs=None,
        install_doc=None,
    )
    asyncio.run(cli_mod.cmd_patch(args))
    assert captured["args"].decomposition_enabled is False
    assert captured["args"].install_doc is None

    # Explicit operator override — agile but install_doc=False must
    # be respected (no silent upgrade).
    captured.clear()
    args = _argparse.Namespace(
        workspace=str(tmp_path), agile=True, generate_specs=None,
        install_doc=False,
    )
    asyncio.run(cli_mod.cmd_patch(args))
    assert captured["args"].decomposition_enabled is True
    assert captured["args"].install_doc is False


def test_cmd_patch_install_doc_default_is_none_through_argparse(tmp_path):
    """Regression for the original Phase 6c BUG: when --install-doc was
    omitted on the actual CLI, argparse defaulted to False (not None),
    which silently skipped the agile→install_doc=True upgrade in
    cmd_patch. This test drives the real parser, NOT a hand-rolled
    Namespace, so the bug would resurface immediately if anyone
    flipped the default back to False.
    """
    from harness.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["patch", "-w", str(tmp_path), "-p", "do x"])
    # MUST be None — Phase 6c relies on this for the cmd_patch upgrade.
    assert args.install_doc is None

    # Explicit values still survive.
    args = parser.parse_args(
        ["patch", "-w", str(tmp_path), "-p", "do x", "--install-doc", "true"],
    )
    assert args.install_doc is True
    args = parser.parse_args(
        ["patch", "-w", str(tmp_path), "-p", "do x", "--install-doc", "false"],
    )
    assert args.install_doc is False


# ---------------------------------------------------------------------------
# create_initial_state — accepts the new kwargs
# ---------------------------------------------------------------------------

def test_create_initial_state_accepts_story_kwargs(tmp_path: Path):
    from harness.graph import create_initial_state
    s = create_initial_state(
        workspace_path=str(tmp_path),
        initial_prompt="p",
        build_command="make",
        decomposition_enabled=True,
        commit_on_story=True,
        story_batch_size=7,
        story_repair_cap=4,
        stories_db_path=str(tmp_path / "state.db"),
    )
    assert s["decomposition_enabled"] is True
    assert s["commit_on_story"] is True
    assert s["story_batch_size"] == 7
    assert s["story_repair_cap"] == 4
    assert s["stories_db_path"].endswith("state.db")


def test_create_initial_state_safe_defaults(tmp_path: Path):
    """All story fields default to safe no-ops — the monolithic flow
    must remain bit-for-bit identical when the caller doesn't opt in."""
    from harness.graph import create_initial_state
    s = create_initial_state(
        workspace_path=str(tmp_path),
        initial_prompt="p",
        build_command="make",
    )
    assert s["decomposition_enabled"] is False
    assert s["commit_on_story"] is False
    assert s["current_story_id"] == ""
    assert s["current_batch_id"] == 0
    assert s["story_scope_files"] == []
    assert s["story_modified_baseline"] == []
    assert s["stories_db_path"] == ""


# ---------------------------------------------------------------------------
# CR → STORY bridge
# ---------------------------------------------------------------------------

@pytest.fixture
def cr_workspace(tmp_path: Path) -> str:
    ws = tmp_path / "cr-ws"
    ws.mkdir()
    cr_dir = ws / "change_requests"
    cr_dir.mkdir()
    (cr_dir / "CR_001_add_login.txt").write_text("Add a /login endpoint.")
    (cr_dir / "CR_002_add_logout.txt").write_text("Add a /logout endpoint.")
    return str(ws)


def _ingest_state(workspace: str, **extra: Any) -> dict[str, Any]:
    base = {
        "workspace_path": workspace,
        "change_request_mode": True,
        "change_requests_dir_abs": os.path.join(workspace, "change_requests"),
        "archive_target_dir": os.path.join(workspace, "change_requests", "applied"),
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "seed"},
        ],
        "session_id": "sess-1",
        "loop_counter": {},
    }
    base.update(extra)
    return base


def test_cr_bridge_creates_one_story_per_cr_when_decomp_enabled(cr_workspace: str):
    from harness.graph import ingest_change_requests_node
    state = _ingest_state(cr_workspace, decomposition_enabled=True)
    out = asyncio.run(ingest_change_requests_node(state))

    assert out.get("stories_db_path", "").endswith("state.db")
    app = story_state.app_name_for_workspace(cr_workspace)
    conn = story_state.open_story_db()
    try:
        stories = story_state.list_stories(conn, app)
    finally:
        conn.close()
    assert len(stories) == 2
    refs = {s["external_ref"] for s in stories}
    assert refs == {"CR-1", "CR-2"}
    assert all(s["feature_key"] == story_state.CR_FEATURE_KEY for s in stories)
    # Each story is tagged as a CR-kind row stamped with its CR id.
    assert all(s["build_kind"] == story_state.BUILD_KIND_CR for s in stories)
    cr_ids_seen = {s["cr_ids"][0] for s in stories if s["cr_ids"]}
    assert cr_ids_seen == {1, 2}


def test_cr_bridge_is_idempotent(cr_workspace: str):
    """Running ingest twice (resume) must not create duplicate rows —
    existing external_refs are skipped."""
    from harness.graph import ingest_change_requests_node
    state = _ingest_state(cr_workspace, decomposition_enabled=True)

    asyncio.run(ingest_change_requests_node(state))
    # Re-create archived files so the second ingest finds them again
    cr_dir = Path(cr_workspace, "change_requests")
    if not (cr_dir / "CR_001_add_login.txt").exists():
        (cr_dir / "CR_001_add_login.txt").write_text("Add a /login endpoint.")
    if not (cr_dir / "CR_002_add_logout.txt").exists():
        (cr_dir / "CR_002_add_logout.txt").write_text("Add a /logout endpoint.")
    asyncio.run(ingest_change_requests_node(state))

    app = story_state.app_name_for_workspace(cr_workspace)
    conn = story_state.open_story_db()
    try:
        stories = story_state.list_stories(conn, app)
    finally:
        conn.close()
    assert len(stories) == 2


def test_cr_bridge_skipped_when_decomp_disabled(cr_workspace: str):
    """Default flow — no story rows for this app, monolithic CR mode runs unchanged."""
    from harness.graph import ingest_change_requests_node
    state = _ingest_state(cr_workspace, decomposition_enabled=False)
    out = asyncio.run(ingest_change_requests_node(state))

    assert "stories_db_path" not in out
    # The global state.db may or may not exist; the guarantee is that
    # no rows landed for this app.
    app = story_state.app_name_for_workspace(cr_workspace)
    if os.path.isfile(story_state.state_db_path()):
        conn = story_state.open_story_db()
        try:
            assert story_state.list_stories(conn, app) == []
        finally:
            conn.close()


def test_cr_bridge_creates_synthetic_cr_requirements_and_links(
    cr_workspace: str,
):
    """v5 traceability: each CR-bridged story must satisfy a
    synthetic ``CR-N`` requirement so the audit gate treats CR work
    uniformly with spec work (no ``OR build_kind='cr'`` special case
    needed in the SQL audit)."""
    from harness.graph import ingest_change_requests_node
    state = _ingest_state(cr_workspace, decomposition_enabled=True)
    asyncio.run(ingest_change_requests_node(state))

    app = story_state.app_name_for_workspace(cr_workspace)
    conn = story_state.open_story_db()
    try:
        # Two CRs ingested → two synthetic requirements created.
        reqs = story_state.list_requirements(conn, app, kind="cr_synthetic")
        req_keys = {r["req_key"] for r in reqs}
        assert req_keys == {"CR-1", "CR-2"}
        # Audit must show zero untraced requirements — every CR-N
        # requirement is satisfied by its bridged story.
        untraced = story_state.requirements_without_satisfying_story(
            conn, app,
        )
        assert untraced == []
    finally:
        conn.close()
