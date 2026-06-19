"""Tests for CLI helper audit hardening (batches 5, 7, 9).

Covers:
  - _get_global_config_path resolves symlinks                        (§5.19)
  - _archive_consumed_change_requests cross-FS fallback              (§5.9)
  - _archive_consumed_change_requests atomic manifest                (§5.10)
  - continue_on_length role typo validation                          (§5.13)
"""

from __future__ import annotations

import errno
import json
import os
import shutil

import pytest

from harness import cli as cli_mod


# ---------------------------------------------------------------------------
# _get_global_config_path realpath (audit §5.19)
# ---------------------------------------------------------------------------


def test_get_global_config_path_resolves_symlinks(tmp_path, monkeypatch):
    """The function uses ``os.path.realpath(__file__)`` so a symlinked
    harness module still locates the source repo's config dir."""
    # Set up a symlinked harness module structure.
    real_repo = tmp_path / "src_repo"
    (real_repo / "harness").mkdir(parents=True)
    (real_repo / "config").mkdir()
    (real_repo / "config" / "config.json").write_text("{}")
    real_module = real_repo / "harness" / "cli.py"
    real_module.write_text("# stub")

    symlinked = tmp_path / "venv" / "site-packages" / "harness"
    symlinked.parent.mkdir(parents=True)
    # Windows refuses symlink creation without admin or developer mode;
    # skip cleanly rather than fail the assertion below.
    try:
        os.symlink(str(real_repo / "harness"), str(symlinked))
    except OSError:
        pytest.skip("symlinks unavailable on this platform")

    # Point __file__ at the symlinked path.
    monkeypatch.setattr(
        "harness.cli.__file__",
        str(symlinked / "cli.py"),
    )
    path = cli_mod._get_global_config_path()
    # realpath resolved through the symlink → real_repo/config/config.json.
    assert os.path.realpath(path) == os.path.realpath(
        str(real_repo / "config" / "config.json")
    )


# ---------------------------------------------------------------------------
# Archive consumed CRs: EXDEV fallback to shutil.move (audit §5.9)
# ---------------------------------------------------------------------------


def test_archive_consumed_falls_back_on_cross_filesystem_rename(tmp_path, monkeypatch):
    """os.replace across filesystems raises EXDEV; the archive helper
    should fall back to shutil.move so the CR still gets archived (and
    isn't left in change_requests/ to be re-applied next run)."""
    src = tmp_path / "change_requests"
    src.mkdir()
    cr_path = src / "CR-1-test.txt"
    cr_path.write_text("body")
    archive_dir = tmp_path / "applied" / "sess-1"

    # Make os.replace raise EXDEV; shutil.move actually performs the move.
    real_move = shutil.move
    calls = {"replace": 0, "move": 0}

    def _exdev_replace(s, d):
        calls["replace"] += 1
        raise OSError(errno.EXDEV, "cross-device link")

    def _spy_move(s, d):
        calls["move"] += 1
        return real_move(s, d)

    monkeypatch.setattr(os, "replace", _exdev_replace)
    monkeypatch.setattr(shutil, "move", _spy_move)

    cli_mod._archive_consumed_change_requests(
        [{
            "abs_path": str(cr_path),
            "cr_id": "1",
            "original_name": "CR-1-test.txt",
        }],
        str(archive_dir),
        session_id="sess-1",
        status="completed",
        modified_files=[],
    )
    # os.replace was attempted at least once (for the CR move). The
    # write_atomic manifest write may also call replace, which our
    # mock will also reject — that's fine; the archive itself is the
    # critical path we're testing.
    assert calls["replace"] >= 1
    assert calls["move"] >= 1
    # Archived file is in the target dir; source is gone.
    assert (archive_dir / "CR-1-test.txt").exists()
    assert not cr_path.exists()


def test_archive_consumed_writes_atomic_manifest(tmp_path):
    """The manifest write must be atomic — uses metrics.write_atomic so
    a SIGKILL mid-write can't leave a truncated JSON file."""
    src = tmp_path / "change_requests"
    src.mkdir()
    cr_path = src / "CR-2-x.txt"
    cr_path.write_text("body")
    archive_dir = tmp_path / "applied" / "sess-2"

    cli_mod._archive_consumed_change_requests(
        [{
            "abs_path": str(cr_path),
            "cr_id": "2",
            "original_name": "CR-2-x.txt",
        }],
        str(archive_dir),
        session_id="sess-2",
        status="completed",
        modified_files=["foo.py"],
    )
    manifest_path = archive_dir / "manifest.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["session_id"] == "sess-2"
    assert data["status"] == "completed"
    assert data["modified_files"] == ["foo.py"]


# ---------------------------------------------------------------------------
# continue_on_length role typo validation (audit §5.13)
# ---------------------------------------------------------------------------


def test_continue_on_length_rejects_unknown_role():
    """A typo in the role name produces a validation error with a
    difflib suggestion — earlier this was a silent no-op."""

    bad_config = {
        # Minimal required fields to get past the rest of validation.
        "models": {"any": {"provider": "deepseek", "model_id": "x", "api_key": ""}},
        "model_routing": {
            "planning_primary": "any",
            "patching_primary": "any",
            "repair_primary": "any",
        },
        "persistence": {"db_path": "~/.harness/x.db"},
        "token_budget": {"hard_cap_usd": 1.0},
        "sandbox": {"backend": "bare"},
        "product_spec_dir": "product_spec",
        "llm_dispatch": {
            "continue_on_length": {"planing": True},  # typo of 'planning'
        },
    }
    with pytest.raises(cli_mod.ConfigError) as ex:
        cli_mod.validate_config_strict(bad_config, source="<test>")
    msg = str(ex.value)
    assert "continue_on_length.planing" in msg
    # difflib suggestion guides the operator.
    assert "did you mean" in msg
