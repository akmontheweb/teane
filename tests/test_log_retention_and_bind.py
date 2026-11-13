"""Session-log retention, and the bind address the operator configured.

Two gaps found while reviewing the web UI on 2026-11-13.

**Retention.** Nothing pruned ``~/.harness/logs``. The harness wrote one
JSONL per session and kept every one forever, so the directory was either
unbounded or — as actually happened — cleared by hand. Five of six lumina
runs vanished that way, leaving the dashboard a single log to show, which
is precisely why the missing ``session_start`` producer went unnoticed for
so long: there was nothing in the UI to notice it with.

Pruning a log directory is easy to get subtly wrong, and the tests here are
mostly about the ways it bites:

- deleting the log the current run is writing
- deleting a rotation backup without its parent, leaving ``teane metrics``
  (which globs ``<sid>.jsonl.*``) reading a truncated run and calling it
  complete
- following ``latest.jsonl`` and taking the live session's real file with it
- ageing out a long run on the mtime of its own first backup

**Bind address.** ``--port`` carried ``default=9000`` in argparse, which
beat ``dashboard.port`` on every start. The shipped config said 8729 and
the server had been serving 9000 the whole time, with nothing reporting the
discrepancy. An omitted flag has to stay distinguishable from a typed one
or config can never be consulted.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from harness import observability as obs
from harness.observability import prune_session_logs


def _log(d, name, *, age_days=0.0, body='{"event":"llm_call"}\n'):
    p = os.path.join(str(d), name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(body)
    if age_days:
        t = time.time() - age_days * 86400
        os.utime(p, (t, t))
    return p


def _sessions(d):
    return sorted(
        n.split(".jsonl", 1)[0]
        for n in os.listdir(str(d))
        if n.endswith(".jsonl") and not os.path.islink(os.path.join(str(d), n))
    )


# ---------------------------------------------------------------------------
# 1. Age
# ---------------------------------------------------------------------------

class TestRetentionByAge:
    def test_an_old_session_is_removed(self, tmp_path):
        _log(tmp_path, "old.jsonl", age_days=120)
        _log(tmp_path, "new.jsonl", age_days=1)
        assert prune_session_logs(
            str(tmp_path), retention_days=90, max_sessions=0) == 1
        assert _sessions(tmp_path) == ["new"]

    def test_a_session_inside_the_window_survives(self, tmp_path):
        _log(tmp_path, "recent.jsonl", age_days=89)
        prune_session_logs(str(tmp_path), retention_days=90, max_sessions=0)
        assert _sessions(tmp_path) == ["recent"]

    def test_zero_days_disables_the_age_axis(self, tmp_path):
        _log(tmp_path, "ancient.jsonl", age_days=10_000)
        assert prune_session_logs(
            str(tmp_path), retention_days=0, max_sessions=0) == 0
        assert _sessions(tmp_path) == ["ancient"]

    def test_age_is_the_newest_file_in_the_session(self, tmp_path):
        """A long run rotates: its FIRST backup can be hours older than the
        run itself. Ageing on the oldest file would delete a session that
        was still being written an hour ago.

        The backup is asserted present, not just the parent. That is the
        half that matters and the half a weaker assertion misses: ``teane
        metrics`` globs ``<sid>.jsonl.*`` to re-assemble a rotated run, so
        silently dropping one backup leaves it reading a fragment and
        reporting it as the complete run. Grouping by session is what stops
        the age axis reaching inside a session and taking only part of it.
        """
        _log(tmp_path, "long.jsonl.1", age_days=120)   # early backup
        _log(tmp_path, "long.jsonl", age_days=1)       # live tail
        assert prune_session_logs(
            str(tmp_path), retention_days=90, max_sessions=0) == 0
        assert _sessions(tmp_path) == ["long"]
        assert os.path.exists(os.path.join(str(tmp_path), "long.jsonl.1")), (
            "the old backup was pruned away from its parent — metrics would "
            "now read a truncated run and call it complete"
        )


# ---------------------------------------------------------------------------
# 2. Count
# ---------------------------------------------------------------------------

class TestRetentionByCount:
    def test_the_newest_n_survive(self, tmp_path):
        for i in range(5):
            _log(tmp_path, f"s{i}.jsonl", age_days=5 - i)  # s4 newest
        assert prune_session_logs(
            str(tmp_path), retention_days=0, max_sessions=2) == 3
        assert _sessions(tmp_path) == ["s3", "s4"]

    def test_zero_disables_the_count_axis(self, tmp_path):
        for i in range(5):
            _log(tmp_path, f"s{i}.jsonl", age_days=i)
        assert prune_session_logs(
            str(tmp_path), retention_days=0, max_sessions=0) == 0
        assert len(_sessions(tmp_path)) == 5

    def test_it_counts_sessions_not_files(self, tmp_path):
        """A rotated session is many files but one session — counting files
        would let three rotations evict two whole other runs."""
        for i in range(2):
            _log(tmp_path, f"s{i}.jsonl", age_days=2 - i)
            _log(tmp_path, f"s{i}.jsonl.1", age_days=2 - i)
            _log(tmp_path, f"s{i}.jsonl.2", age_days=2 - i)
        assert prune_session_logs(
            str(tmp_path), retention_days=0, max_sessions=2) == 0
        assert _sessions(tmp_path) == ["s0", "s1"]

    def test_both_axes_apply_together(self, tmp_path):
        _log(tmp_path, "old.jsonl", age_days=200)
        for i in range(3):
            _log(tmp_path, f"s{i}.jsonl", age_days=3 - i)
        prune_session_logs(str(tmp_path), retention_days=90, max_sessions=2)
        assert _sessions(tmp_path) == ["s1", "s2"]


# ---------------------------------------------------------------------------
# 3. What must never be deleted
# ---------------------------------------------------------------------------

class TestWhatSurvivesRegardless:
    def test_the_current_session_is_never_pruned(self, tmp_path):
        """The reason ``keep`` exists: a tight cap would otherwise delete
        the file the running build is in the middle of writing."""
        _log(tmp_path, "live.jsonl", age_days=500)
        for i in range(3):
            _log(tmp_path, f"s{i}.jsonl", age_days=0)
        prune_session_logs(
            str(tmp_path), retention_days=1, max_sessions=1, keep="live")
        assert "live" in _sessions(tmp_path)

    def test_a_rotated_session_is_pruned_as_one_unit(self, tmp_path):
        """``teane metrics`` globs ``<sid>.jsonl.*`` to re-assemble a
        rotated run. Removing the parent but leaving a backup would have it
        read a fragment and report it as the whole run."""
        _log(tmp_path, "old.jsonl", age_days=200)
        _log(tmp_path, "old.jsonl.1", age_days=200)
        _log(tmp_path, "old.jsonl.2", age_days=200)
        prune_session_logs(str(tmp_path), retention_days=90, max_sessions=0)
        assert os.listdir(str(tmp_path)) == []

    def test_the_latest_symlink_is_not_followed(self, tmp_path):
        """Deleting through the link would take the live session's real log
        with it — the opposite of what retention is for."""
        _log(tmp_path, "live.jsonl", age_days=0)
        os.symlink("live.jsonl", os.path.join(str(tmp_path), "latest.jsonl"))
        prune_session_logs(
            str(tmp_path), retention_days=1, max_sessions=1, keep="live")
        assert os.path.exists(os.path.join(str(tmp_path), "live.jsonl"))

    def test_a_symlink_is_never_counted_as_a_session(self, tmp_path):
        _log(tmp_path, "a.jsonl", age_days=2)
        _log(tmp_path, "b.jsonl", age_days=1)
        os.symlink("b.jsonl", os.path.join(str(tmp_path), "latest.jsonl"))
        prune_session_logs(str(tmp_path), retention_days=0, max_sessions=2)
        assert _sessions(tmp_path) == ["a", "b"]

    def test_a_dangling_latest_link_is_cleared(self, tmp_path):
        """Harmless to the dashboard, which skips symlinks, but confusing to
        a human running `tail -f`."""
        _log(tmp_path, "old.jsonl", age_days=200)
        os.symlink("old.jsonl", os.path.join(str(tmp_path), "latest.jsonl"))
        prune_session_logs(str(tmp_path), retention_days=90, max_sessions=0)
        assert not os.path.lexists(os.path.join(str(tmp_path), "latest.jsonl"))

    def test_a_live_latest_link_is_kept(self, tmp_path):
        _log(tmp_path, "live.jsonl", age_days=0)
        link = os.path.join(str(tmp_path), "latest.jsonl")
        os.symlink("live.jsonl", link)
        prune_session_logs(str(tmp_path), retention_days=90, max_sessions=0)
        assert os.path.islink(link)

    def test_a_missing_directory_is_not_an_error(self, tmp_path):
        assert prune_session_logs(str(tmp_path / "nope")) == 0

    def test_non_log_files_are_left_alone(self, tmp_path):
        _log(tmp_path, "notes.txt", age_days=500)
        prune_session_logs(str(tmp_path), retention_days=1, max_sessions=0)
        assert os.path.exists(os.path.join(str(tmp_path), "notes.txt"))


# ---------------------------------------------------------------------------
# 4. Through the real entry point
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_logging():
    import logging

    root = logging.getLogger()
    saved, lvl = list(root.handlers), root.level
    was_open = obs._session_open
    obs._session_open = False
    try:
        yield
    finally:
        for h in list(root.handlers):
            try:
                h.close()
            except Exception:  # noqa: BLE001
                pass
        root.handlers = saved
        root.setLevel(lvl)
        obs._session_open = was_open


class TestConfigureLoggingPrunes:
    def test_starting_a_run_drops_an_expired_log(self, tmp_path, clean_logging):
        _log(tmp_path, "ancient.jsonl", age_days=400)
        obs.configure_logging(
            session_id="fresh", log_dir=str(tmp_path), console_level="OFF",
            retention_days=90, max_session_logs=0,
        )
        assert "ancient" not in _sessions(tmp_path)
        assert "fresh" in _sessions(tmp_path)

    def test_the_log_being_opened_survives_a_cap_of_one(self, tmp_path, clean_logging):
        for i in range(4):
            _log(tmp_path, f"s{i}.jsonl", age_days=0)
        obs.configure_logging(
            session_id="fresh", log_dir=str(tmp_path), console_level="OFF",
            retention_days=0, max_session_logs=1,
        )
        assert "fresh" in _sessions(tmp_path)

    def test_retention_off_keeps_everything(self, tmp_path, clean_logging):
        _log(tmp_path, "ancient.jsonl", age_days=9999)
        obs.configure_logging(
            session_id="fresh", log_dir=str(tmp_path), console_level="OFF",
            retention_days=0, max_session_logs=0,
        )
        assert "ancient" in _sessions(tmp_path)

    def test_the_new_session_is_still_usable_afterwards(self, tmp_path, clean_logging):
        """Pruning runs after the handler is attached; a bug there could
        leave the run with no log at all."""
        for i in range(5):
            _log(tmp_path, f"s{i}.jsonl", age_days=i + 1)
        p = obs.configure_logging(
            session_id="fresh", log_dir=str(tmp_path), console_level="OFF",
            retention_days=0, max_session_logs=1, workspace_path="/tmp/ws",
        )
        obs.emit_event("llm_call", cost_usd=0.01)
        obs.emit_session_end(0)
        with open(p, encoding="utf-8") as fh:
            names = [json.loads(line).get("event") for line in fh if line.strip()]
        assert "session_start" in names and "session_end" in names


class TestTheShippedDefaults:
    def test_config_declares_both_axes(self):
        import pathlib
        cfg = json.loads(
            (pathlib.Path(__file__).resolve().parent.parent
             / "config" / "config.json").read_text())["logging"]
        assert cfg["retention_days"] == 90
        assert cfg["max_session_logs"] == 500

    def test_both_keys_are_registered_with_the_validator(self):
        """An unregistered key is dropped silently — the failure mode the
        nested-key whitelist exists to prevent."""
        from harness.cli import _KNOWN_NESTED_KEYS, _TYPE_SCHEMA

        assert "retention_days" in _KNOWN_NESTED_KEYS["logging"]
        assert "max_session_logs" in _KNOWN_NESTED_KEYS["logging"]
        assert _TYPE_SCHEMA["logging.retention_days"] == (int,)
        assert _TYPE_SCHEMA["logging.max_session_logs"] == (int,)

    def test_both_are_documented(self):
        import pathlib
        txt = (pathlib.Path(__file__).resolve().parent.parent
               / "config" / "config_help.md").read_text()
        assert "retention_days" in txt and "max_session_logs" in txt


# ---------------------------------------------------------------------------
# 5. The bind address
# ---------------------------------------------------------------------------

class TestWebBindResolution:
    """Flag > config > built-in default."""

    def _cfg(self, monkeypatch, **dash):
        from harness import cli

        monkeypatch.setattr(
            cli, "discover_config", lambda _p: {"dashboard": dash}, raising=True)

    def test_the_configured_port_is_used_when_no_flag_is_typed(self, monkeypatch):
        """The original bug: config said 8729, the server served 9000."""
        from harness.cli import _resolve_web_bind

        self._cfg(monkeypatch, port=8729)
        assert _resolve_web_bind(None, None)[1] == 8729

    def test_a_typed_port_beats_config(self, monkeypatch):
        from harness.cli import _resolve_web_bind

        self._cfg(monkeypatch, port=8729)
        assert _resolve_web_bind(None, 8080)[1] == 8080

    def test_the_configured_host_is_used(self, monkeypatch):
        from harness.cli import _resolve_web_bind

        self._cfg(monkeypatch, host="0.0.0.0")
        assert _resolve_web_bind(None, None)[0] == "0.0.0.0"

    def test_a_typed_host_beats_config(self, monkeypatch):
        from harness.cli import _resolve_web_bind

        self._cfg(monkeypatch, host="0.0.0.0")
        assert _resolve_web_bind("127.0.0.1", None)[0] == "127.0.0.1"

    def test_an_empty_config_falls_back_to_dashboard_configs_own_default(
        self, monkeypatch,
    ):
        """8729, not the 9000 the CLI help used to claim. ``_DEFAULT_PORT``
        has been 8729 all along; the flag default was invented."""
        from harness.cli import _resolve_web_bind

        self._cfg(monkeypatch)
        assert _resolve_web_bind(None, None) == ("127.0.0.1", 8729)

    def test_a_broken_config_still_starts_the_server(self, monkeypatch):
        """Retention and bind resolution are both housekeeping — neither may
        stop an operator bringing the UI up."""
        from harness import cli

        def _boom(_p):
            raise RuntimeError("config is a mess")

        monkeypatch.setattr(cli, "discover_config", _boom, raising=True)
        assert cli._resolve_web_bind(None, None) == ("127.0.0.1", 8729)

    def test_the_flags_have_no_argparse_default(self):
        """The load-bearing part: with ``default=9000`` baked in, an omitted
        flag is indistinguishable from a typed one and config can never
        win."""
        from harness.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["web", "start"])
        assert args.host is None
        assert args.port is None

    def test_a_typed_flag_still_parses(self):
        from harness.cli import build_parser

        args = build_parser().parse_args(
            ["web", "start", "--host", "0.0.0.0", "--port", "8080"])
        assert args.host == "0.0.0.0" and args.port == 8080
