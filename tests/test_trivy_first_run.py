"""Regression tests for the trivy first-run DB init behaviour.

Ciod session 523e86a7 hit ``FATAL --skip-db-update cannot be specified
on the first run`` because the stamp file
``~/.harness/cache/trivy/.db_refreshed_at`` existed from an earlier
session but the actual DB (``<cache_dir>/db/trivy.db``) was missing —
cleared by an operator, evicted by disk pressure, or wiped by a
trivy schema upgrade.

Fix (2026-07-04): gate ``--skip-db-update`` on BOTH the stamp file
AND the physical DB file, so a first-run invocation always downloads
without crashing.
"""

from __future__ import annotations

import os
import time

import pytest


@pytest.fixture
def isolated_trivy_cache(monkeypatch, tmp_path):
    """Redirect ``~/.harness/cache/trivy`` to a per-test temp dir so
    the test doesn't touch the operator's real trivy cache."""
    cache = tmp_path / "harness-cache-trivy"
    cache.mkdir(parents=True)

    orig_expand = os.path.expanduser

    def _fake_expand(path: str) -> str:
        if path == "~/.harness/cache/trivy":
            return str(cache)
        return orig_expand(path)

    monkeypatch.setattr(os.path, "expanduser", _fake_expand)
    return cache


@pytest.fixture
def fake_trivy_binary(monkeypatch, tmp_path):
    """Pretend trivy is on PATH by making ``shutil.which("trivy")``
    return a fake path — the subprocess call is stubbed out
    separately, so the "binary" never actually runs."""
    fake_path = tmp_path / "trivy-stub"
    fake_path.write_text("#!/bin/sh\nexit 0\n")
    fake_path.chmod(0o755)
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: (
        str(fake_path) if name == "trivy" else None
    ))
    return fake_path


class TestTrivyFirstRun:
    def _captured_cmd(self, monkeypatch):
        """Stub ``_run_subprocess_scanner`` to capture the command list
        and return a trivial "no vulns" JSON result."""
        captured: list[list[str]] = []

        async def _fake_run(cmd, *, timeout_seconds, label):
            captured.append(list(cmd))
            # Empty findings, exit 0 — matches a clean trivy scan.
            return (0, '{"Results":[]}', "")

        from harness import security
        monkeypatch.setattr(
            security, "_run_subprocess_scanner", _fake_run,
        )
        return captured

    @pytest.mark.asyncio
    async def test_no_skip_db_update_when_db_missing(
        self, isolated_trivy_cache, fake_trivy_binary, monkeypatch,
    ):
        # Simulate the ciod 523e86a7 pathology: stamp exists (fresh)
        # but the ``db/trivy.db`` file does NOT — a cleared cache or
        # trivy upgrade. Skip must NOT be set: run_trivy_scan must
        # let trivy download.
        stamp = isolated_trivy_cache / ".db_refreshed_at"
        stamp.write_text(str(time.time()))
        # deliberately do NOT create db/trivy.db
        assert not (isolated_trivy_cache / "db" / "trivy.db").exists()

        captured = self._captured_cmd(monkeypatch)
        from harness.security import run_trivy_scan
        await run_trivy_scan("/tmp/workspace-placeholder")
        assert captured, "trivy command was never dispatched"
        assert "--skip-db-update" not in captured[0], (
            "First-run guard failed: --skip-db-update was passed to "
            "trivy despite the DB file being absent. This is the "
            "ciod 523e86a7 FATAL 'cannot be specified on the first "
            "run' regression."
        )

    @pytest.mark.asyncio
    async def test_skip_db_update_set_when_stamp_and_db_both_present(
        self, isolated_trivy_cache, fake_trivy_binary, monkeypatch,
    ):
        # The happy path: stamp is fresh AND the DB file exists.
        # Trivy has done the download; the cache is valid; we should
        # skip the update on this run to save the ~150 MB pull.
        stamp = isolated_trivy_cache / ".db_refreshed_at"
        stamp.write_text(str(time.time()))
        db_dir = isolated_trivy_cache / "db"
        db_dir.mkdir()
        (db_dir / "trivy.db").write_text("fake-db-blob")

        captured = self._captured_cmd(monkeypatch)
        from harness.security import run_trivy_scan
        await run_trivy_scan("/tmp/workspace-placeholder")
        assert captured
        assert "--skip-db-update" in captured[0], (
            "Happy-path regression: with both stamp and DB present, "
            "--skip-db-update should be set to avoid the 150 MB pull."
        )

    @pytest.mark.asyncio
    async def test_no_skip_when_stamp_stale_even_with_db(
        self, isolated_trivy_cache, fake_trivy_binary, monkeypatch,
    ):
        # Stamp older than 24h → force a refresh even when the DB
        # file is present. Preserves the pre-existing "at most one
        # DB refresh per 24h" contract.
        stamp = isolated_trivy_cache / ".db_refreshed_at"
        old = time.time() - (25 * 60 * 60)
        stamp.write_text(str(old))
        os.utime(str(stamp), (old, old))
        db_dir = isolated_trivy_cache / "db"
        db_dir.mkdir()
        (db_dir / "trivy.db").write_text("fake-db-blob")

        captured = self._captured_cmd(monkeypatch)
        from harness.security import run_trivy_scan
        await run_trivy_scan("/tmp/workspace-placeholder")
        assert captured
        assert "--skip-db-update" not in captured[0]

    @pytest.mark.asyncio
    async def test_no_skip_on_true_first_run(
        self, isolated_trivy_cache, fake_trivy_binary, monkeypatch,
    ):
        # Genuinely first ever run: neither stamp nor DB exists.
        # Must NOT pass --skip-db-update.
        assert not (isolated_trivy_cache / ".db_refreshed_at").exists()
        assert not (isolated_trivy_cache / "db" / "trivy.db").exists()

        captured = self._captured_cmd(monkeypatch)
        from harness.security import run_trivy_scan
        await run_trivy_scan("/tmp/workspace-placeholder")
        assert captured
        assert "--skip-db-update" not in captured[0]
