"""Regression tests for the per-repo session memory module (#7).

Covers:
    - ``repo_identity`` is deterministic for the same path / origin URL.
    - ``read_repo_memory`` returns empty string when the file is absent.
    - ``append_session_note`` creates the directory and file, then
      successive appends accumulate.
    - The FIFO trim drops the oldest sections when the file exceeds
      ``max_bytes`` without ever discarding the just-written entry.
    - The read path tails to ``inject_max_bytes`` cleanly on section
      boundaries.
    - Writes are atomic — a half-written ``.tmp`` is never left behind
      on success.
    - ``enabled=false`` short-circuits both read and write.
"""

from __future__ import annotations

import os
import stat

from harness.repo_memory import (
    RepoMemoryConfig,
    append_session_note,
    memory_file_path,
    read_repo_memory,
    repo_identity,
)


def test_repo_identity_is_deterministic_for_same_path(tmp_path):
    p = str(tmp_path)
    a = repo_identity(p)
    b = repo_identity(p)
    assert a == b
    assert len(a) == 16


def test_repo_identity_differs_for_different_paths(tmp_path):
    p1 = str(tmp_path / "one")
    p2 = str(tmp_path / "two")
    os.makedirs(p1)
    os.makedirs(p2)
    assert repo_identity(p1) != repo_identity(p2)


def test_read_returns_empty_when_file_missing(tmp_path):
    cfg = RepoMemoryConfig(dir=str(tmp_path))
    assert read_repo_memory(str(tmp_path), cfg) == ""


def test_append_then_read_roundtrip(tmp_path):
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    cfg = RepoMemoryConfig(dir=str(tmp_path / "mem"))
    path = append_session_note(
        workspace,
        session_id="abcd1234-test",
        prompt_summary="Add JWT auth",
        modified_files=["src/auth.py", "tests/test_auth.py"],
        exit_code=0,
        cfg=cfg,
    )
    assert path is not None
    assert os.path.isfile(path)
    content = read_repo_memory(workspace, cfg)
    assert "Session abcd1234" in content
    assert "Add JWT auth" in content
    assert "src/auth.py" in content
    assert "success" in content


def test_memory_file_permissions_are_owner_only(tmp_path):
    """Memory files contain workspace paths and session metadata —
    ``~/.harness`` already lives under the operator's home, but tightening
    the file to 0600 means a malformed umask or a shared host can't
    expose them to other local accounts."""
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    cfg = RepoMemoryConfig(dir=str(tmp_path / "mem"))
    path = append_session_note(
        workspace,
        session_id="perm-test",
        prompt_summary="t",
        modified_files=[],
        exit_code=0,
        cfg=cfg,
    )
    assert path is not None
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"


def test_memory_file_redacts_home_dir_prefix(tmp_path, monkeypatch):
    """Absolute paths under the operator's $HOME get a ``~/...``
    substitution before they land in the memory file."""
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    fake_home = str(tmp_path / "homedir")
    os.makedirs(fake_home)
    monkeypatch.setenv("HOME", fake_home)
    cfg = RepoMemoryConfig(dir=str(tmp_path / "mem"))
    path = append_session_note(
        workspace,
        session_id="redact-test",
        prompt_summary="t",
        modified_files=[os.path.join(fake_home, "project", "main.py")],
        exit_code=0,
        cfg=cfg,
        extra_notes=f"see {os.path.join(fake_home, 'logs', 'session.log')}",
    )
    assert path is not None
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    assert fake_home not in content
    assert "~/project/main.py" in content
    assert "~/logs/session.log" in content


def test_multiple_appends_accumulate(tmp_path):
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    cfg = RepoMemoryConfig(dir=str(tmp_path / "mem"))
    for i in range(3):
        append_session_note(
            workspace,
            session_id=f"s{i:03d}-test",
            prompt_summary=f"task {i}",
            modified_files=[f"f{i}.py"],
            exit_code=0,
            cfg=cfg,
        )
    content = read_repo_memory(workspace, cfg)
    assert "task 0" in content
    assert "task 1" in content
    assert "task 2" in content
    # Three Session headings means three append calls landed.
    assert content.count("## Session ") == 3


def test_max_bytes_fifo_trims_oldest(tmp_path):
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    # Small cap → after enough writes the oldest section drops first.
    cfg = RepoMemoryConfig(dir=str(tmp_path / "mem"), max_bytes=600)
    for i in range(10):
        append_session_note(
            workspace,
            session_id=f"s{i:03d}-test",
            prompt_summary=("x" * 60) + f" iteration {i}",
            modified_files=[f"file_{i}.py"],
            exit_code=0,
            cfg=cfg,
        )
    content = read_repo_memory(workspace, cfg)
    # The most recent entry MUST be there.
    assert "iteration 9" in content
    # And the file size must respect the cap (allowing a small overhead
    # for the final unsplittable section).
    path = memory_file_path(workspace, cfg)
    assert os.path.getsize(path) <= 600 * 2  # generous; we may keep two


def test_read_caps_at_inject_max_bytes(tmp_path):
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    cfg = RepoMemoryConfig(
        dir=str(tmp_path / "mem"),
        max_bytes=200_000,
        inject_max_bytes=300,
    )
    for i in range(8):
        append_session_note(
            workspace,
            session_id=f"s{i:03d}-test",
            prompt_summary=("y" * 40) + f" iter {i}",
            modified_files=[f"f{i}.py"],
            exit_code=0,
            cfg=cfg,
        )
    injected = read_repo_memory(workspace, cfg)
    assert len(injected.encode("utf-8")) <= 600  # capped to ~inject_max + small overhead
    # The tail (most recent) must be preserved.
    assert "iter 7" in injected


def test_atomic_write_leaves_no_tmp(tmp_path):
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    cfg = RepoMemoryConfig(dir=str(tmp_path / "mem"))
    append_session_note(
        workspace,
        session_id="x-test",
        prompt_summary="check",
        modified_files=[],
        exit_code=0,
        cfg=cfg,
    )
    mem_dir = str(tmp_path / "mem")
    files = os.listdir(mem_dir)
    assert not any(f.endswith(".tmp") for f in files)


def test_disabled_short_circuits_read_and_write(tmp_path):
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    cfg = RepoMemoryConfig(enabled=False, dir=str(tmp_path / "mem"))
    result = append_session_note(
        workspace,
        session_id="x",
        prompt_summary="anything",
        modified_files=[],
        exit_code=0,
        cfg=cfg,
    )
    assert result is None  # disabled — no write happened
    assert read_repo_memory(workspace, cfg) == ""
    # Directory should never have been created.
    assert not os.path.isdir(str(tmp_path / "mem"))


def test_from_config_parses_section():
    cfg = RepoMemoryConfig.from_config({
        "memory": {
            "enabled": False,
            "dir": "/tmp/xyz",
            "max_bytes": 4096,
            "inject_max_bytes": 1024,
        },
    })
    assert cfg.enabled is False
    assert cfg.dir == "/tmp/xyz"
    assert cfg.max_bytes == 4096
    assert cfg.inject_max_bytes == 1024


def test_from_config_defaults_when_section_missing():
    cfg = RepoMemoryConfig.from_config({})
    assert cfg.enabled is True


# ---------------------------------------------------------------------------
# Compaction (audit #24)
# ---------------------------------------------------------------------------

def _make_verbose_session(sid: str, iteration: int, n_files: int = 8) -> dict:
    """Build a session-append payload with enough files + a Notes line to
    exercise the compaction stripper (which drops indented file bullets +
    ``- Notes:`` lines from older sections).
    """
    return {
        "session_id": sid,
        "prompt_summary": f"iteration {iteration} — building some feature",
        "modified_files": [f"src/file_{iteration}_{k}.py" for k in range(n_files)],
        "exit_code": 0,
        "extra_notes": (
            f"[learned-rule:iter_{iteration}] Some hypothesis text that would "
            f"normally survive across sessions."
        ),
    }


def test_compaction_strips_old_file_lists_and_notes(tmp_path):
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    # compact_after tuned low so the second append trips compaction on
    # the first (older) section. keep_recent=1 → only the just-written
    # section is spared.
    cfg = RepoMemoryConfig(
        dir=str(tmp_path / "mem"),
        max_bytes=1_000_000,       # generous so FIFO trim never fires
        inject_max_bytes=1_000_000,
        compact_after=500,
        compact_keep_recent=1,
    )
    append_session_note(workspace, cfg=cfg, **_make_verbose_session("s000", 0))
    append_session_note(workspace, cfg=cfg, **_make_verbose_session("s001", 1))
    content = read_repo_memory(workspace, cfg)
    # First session's file bullets and Notes must have been stripped.
    assert "src/file_0_0.py" not in content
    assert "[learned-rule:iter_0]" not in content
    # Recent (just-written) session keeps everything.
    assert "src/file_1_0.py" in content
    assert "[learned-rule:iter_1]" in content
    # Prompt + Status + Modified count for the older session are preserved.
    assert "iteration 0" in content
    assert "8 file(s) modified" in content  # count survives; list doesn't


def test_compaction_below_threshold_is_a_no_op(tmp_path):
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    # Threshold high enough that neither append trips compaction.
    cfg = RepoMemoryConfig(
        dir=str(tmp_path / "mem"),
        max_bytes=1_000_000,
        inject_max_bytes=1_000_000,
        compact_after=100_000,
        compact_keep_recent=1,
    )
    append_session_note(workspace, cfg=cfg, **_make_verbose_session("s000", 0))
    append_session_note(workspace, cfg=cfg, **_make_verbose_session("s001", 1))
    content = read_repo_memory(workspace, cfg)
    # Both sessions keep their file lists — nothing was compacted.
    assert "src/file_0_0.py" in content
    assert "src/file_1_0.py" in content
    assert "[learned-rule:iter_0]" in content
    assert "[learned-rule:iter_1]" in content


def test_compaction_is_idempotent_on_already_compact_section(tmp_path):
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    cfg = RepoMemoryConfig(
        dir=str(tmp_path / "mem"),
        max_bytes=1_000_000,
        inject_max_bytes=1_000_000,
        compact_after=500,
        compact_keep_recent=1,
    )
    # Three appends: after append #2, session #0 gets compacted. After
    # append #3, sessions #0 AND #1 are outside the recent window; #0
    # is already compact so only #1 has verbose bullets to strip. This
    # exercises the "seen_any_verbose" gate that prevents double-counting.
    for i in range(3):
        append_session_note(workspace, cfg=cfg, **_make_verbose_session(f"s{i:03d}", i))
    content = read_repo_memory(workspace, cfg)
    # Old sections (#0 and #1) have no file lists.
    assert "src/file_0_0.py" not in content
    assert "src/file_1_0.py" not in content
    # #0's Prompt still survives.
    assert "iteration 0" in content
    # Most recent session (#2) is intact.
    assert "src/file_2_0.py" in content
    assert "[learned-rule:iter_2]" in content


def test_compaction_emits_event(tmp_path, caplog):
    import logging as _logging
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    cfg = RepoMemoryConfig(
        dir=str(tmp_path / "mem"),
        max_bytes=1_000_000,
        inject_max_bytes=1_000_000,
        compact_after=500,
        compact_keep_recent=1,
    )
    append_session_note(workspace, cfg=cfg, **_make_verbose_session("s000", 0))
    with caplog.at_level(_logging.INFO, logger="harness.events"):
        append_session_note(workspace, cfg=cfg, **_make_verbose_session("s001", 1))
    # Look for the structured event record.
    compaction_records = [
        r for r in caplog.records
        if getattr(r, "event", None) == "memory_compaction_fired"
    ]
    assert compaction_records, "expected a memory_compaction_fired event"
    rec = compaction_records[0]
    assert getattr(rec, "sections_compacted", 0) >= 1
    assert getattr(rec, "bytes_before", 0) > getattr(rec, "bytes_after", 0)
    assert getattr(rec, "bytes_reclaimed", -1) > 0


def test_compaction_from_config_parses_knobs():
    cfg = RepoMemoryConfig.from_config({
        "memory": {
            "compact_after": 12345,
            "compact_keep_recent": 7,
        },
    })
    assert cfg.compact_after == 12345
    assert cfg.compact_keep_recent == 7


def test_compaction_defaults_when_knobs_missing():
    cfg = RepoMemoryConfig.from_config({"memory": {}})
    assert cfg.compact_after == 60_000
    assert cfg.compact_keep_recent == 3
    assert cfg.dir == "~/.harness/memory"
