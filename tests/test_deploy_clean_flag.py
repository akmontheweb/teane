"""`teane deploy --clean` — tear down this workspace's stack (containers,
orphans, and named volumes) before re-deploying.

Covers:
  - teardown_containers(remove_volumes=...) builds the right compose argv
    (`--volumes` present only when requested; `--remove-orphans` always).
  - The clean_deploy flag is threaded into graph state.
"""
from __future__ import annotations

import os

import pytest


@pytest.mark.asyncio
async def test_teardown_adds_volumes_flag_when_requested(tmp_path, monkeypatch):
    from harness import deploy, sandbox

    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    captured: dict[str, list[str]] = {}

    async def _fake_run(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return (0, b"", b"", False)

    monkeypatch.setattr(sandbox, "run_subprocess_kill_on_timeout", _fake_run)

    ok = await deploy.teardown_containers(str(tmp_path), remove_volumes=True)
    assert ok is True
    assert "down" in captured["cmd"]
    assert "--remove-orphans" in captured["cmd"]
    assert "--volumes" in captured["cmd"]


@pytest.mark.asyncio
async def test_teardown_omits_volumes_by_default(tmp_path, monkeypatch):
    from harness import deploy, sandbox

    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    captured: dict[str, list[str]] = {}

    async def _fake_run(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return (0, b"", b"", False)

    monkeypatch.setattr(sandbox, "run_subprocess_kill_on_timeout", _fake_run)

    await deploy.teardown_containers(str(tmp_path))  # remove_volumes defaults False
    assert "--remove-orphans" in captured["cmd"]
    assert "--volumes" not in captured["cmd"]


@pytest.mark.asyncio
async def test_teardown_noops_without_compose_file(tmp_path):
    from harness import deploy

    # No docker-compose.yml present → nothing to tear down, returns False
    # without shelling out.
    assert await deploy.teardown_containers(str(tmp_path), remove_volumes=True) is False


def test_create_initial_state_threads_clean_deploy(tmp_path):
    from harness.graph import create_initial_state

    common = dict(
        workspace_path=str(tmp_path),
        initial_prompt="x",
        build_command="true",
    )
    on = create_initial_state(clean_deploy=True, **common)
    off = create_initial_state(**common)
    assert on["clean_deploy"] is True
    assert off["clean_deploy"] is False  # default
