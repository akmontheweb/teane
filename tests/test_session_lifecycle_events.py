"""The dashboard's session lifecycle, end to end through the real producer.

Two defects, found 2026-11-13 by opening the web UI and reading what it
showed rather than what its tests asserted.

**1. ``session_start`` / ``session_end`` were consumed in ten places and
emitted nowhere.** Grepping all of ``harness/`` for either name turned up
only consumers plus one docstring. Everything downstream was dead: the
sessions table's Started / Ended / Status / Mode / Workspace columns were
blank, all three status windows (today / 7d / 30d) read zero because a
session with no timestamp cannot be placed in a window, and "is it still
running?" -- decided by whether the tail event is ``session_end`` -- could
never be answered.

The existing tests passed throughout, because they *synthesise* the events
in their fixtures (``test_dashboard.py`` writes ``{"event":
"session_start", ...}`` by hand). The consumer was verified against a
contract the producer never met. So every test here that covers the
lifecycle drives ``configure_logging`` / ``emit_session_end`` -- the real
producer -- and reads the file they actually wrote.

A second, quieter mismatch sat underneath: the dashboard read
``event["timestamp"]``, while ``JSONFormatter`` stamps every line with
``ts``. That is why even the fallback ("no session_start? use the first
event's time") produced nothing.

**2. ``latest.jsonl`` is a symlink and was globbed as a second session.**
``configure_logging`` maintains it so ``tail -f`` works without knowing the
session id; ``list_sessions`` matched ``*.jsonl`` with no symlink check. The
newest run was parsed twice -- a phantom row named "latest" in the sessions
table, double-counted in all three status windows, every point of its cost
burn plotted twice, and a duplicate offered in the resume picker. Confirmed
live before the fix: two rows, identical 23 calls / $0.0572 / 1,547,505
tokens.
"""

from __future__ import annotations

import json
import os

import pytest

from harness import observability as obs
from harness.dashboard import (
    DashboardConfig,
    _event_timestamp,
    list_running_sessions,
    list_sessions,
)


def _cfg(tmp_path, **over):
    base = {
        "log_dir": str(tmp_path / "logs"),
        "metrics_dir": str(tmp_path / "metrics"),
        "memory_dir": str(tmp_path / "memory"),
        "repo_index_dir": str(tmp_path / "idx"),
        "schedule_db": str(tmp_path / "schedule.db"),
        "static_dir": str(tmp_path / "static"),
        "enabled": True,
    }
    base.update(over)
    return DashboardConfig.from_config({"dashboard": base})


@pytest.fixture
def clean_logging():
    """Save/restore the root logger and the module's session flag.

    ``configure_logging`` takes over the root logger, so a test that calls
    it must hand the process back as it found it or every later test's
    captured output changes.
    """
    import logging

    root = logging.getLogger()
    saved, saved_level = list(root.handlers), root.level
    saved_open = obs._session_open
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
        root.setLevel(saved_level)
        obs._session_open = saved_open


def _events(path):
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _named(path, name):
    return [e for e in _events(path) if e.get("event") == name]


# ---------------------------------------------------------------------------
# 1. The producer
# ---------------------------------------------------------------------------

class TestSessionStartIsActuallyEmitted:
    def test_a_run_writes_session_start(self, tmp_path, clean_logging):
        p = obs.configure_logging(
            session_id="sess-real", log_dir=str(tmp_path), console_level="OFF",
            workspace_path="/tmp/my-repo",
        )
        assert p, "configure_logging returned no log path"
        starts = _named(p, "session_start")
        assert len(starts) == 1, (
            "this is the original defect: ten consumers, zero producers"
        )

    def test_it_carries_the_fields_the_dashboard_reads(self, tmp_path, clean_logging):
        p = obs.configure_logging(
            session_id="sess-real", log_dir=str(tmp_path), console_level="OFF",
            workspace_path="/tmp/my-repo",
        )
        evt = _named(p, "session_start")[0]
        assert evt["workspace_path"] == "/tmp/my-repo"
        assert evt["session_id"] == "sess-real"
        assert evt["timestamp"]

    def test_it_is_the_first_event_in_the_log(self, tmp_path, clean_logging):
        """``list_running_sessions`` stops scanning at the first
        ``session_start`` it finds and treats its absence as "not a run", so
        the event has to lead the file."""
        p = obs.configure_logging(
            session_id="sess-real", log_dir=str(tmp_path), console_level="OFF",
            workspace_path="/tmp/my-repo",
        )
        obs.emit_event("llm_call", cost_usd=0.01)
        evts = [e for e in _events(p) if e.get("event")]
        assert evts[0]["event"] == "session_start"

    def test_a_process_that_is_not_a_run_emits_nothing(self, tmp_path, clean_logging):
        """``teane web start`` configures logging too. Emitting the event
        there would put the dashboard's own boot log in its session list --
        the phantom-row failure its running-session scan already guards
        against."""
        p = obs.configure_logging(
            session_id="web-boot", log_dir=str(tmp_path), console_level="OFF",
        )
        assert _named(p, "session_start") == []
        assert obs._session_open is False


class TestSessionEnd:
    def _start(self, tmp_path):
        return obs.configure_logging(
            session_id="sess-real", log_dir=str(tmp_path), console_level="OFF",
            workspace_path="/tmp/my-repo",
        )

    def test_it_records_the_exit_code(self, tmp_path, clean_logging):
        p = self._start(tmp_path)
        obs.emit_session_end(0)
        ends = _named(p, "session_end")
        assert len(ends) == 1 and ends[0]["exit_code"] == 0

    def test_a_failure_code_survives(self, tmp_path, clean_logging):
        """The sessions table shows WHY a run ended; a crashed run that
        reports 0 is worse than one that reports nothing."""
        p = self._start(tmp_path)
        obs.emit_session_end(2)
        assert _named(p, "session_end")[0]["exit_code"] == 2

    def test_it_is_the_last_event(self, tmp_path, clean_logging):
        """Running-detection reads the TAIL event name."""
        p = self._start(tmp_path)
        obs.emit_event("llm_call", cost_usd=0.01)
        obs.emit_session_end(0)
        evts = [e for e in _events(p) if e.get("event")]
        assert evts[-1]["event"] == "session_end"

    def test_it_is_a_no_op_without_a_session(self, tmp_path, clean_logging):
        """``teane doctor`` must not write a dangling tail event."""
        p = obs.configure_logging(
            session_id="web-boot", log_dir=str(tmp_path), console_level="OFF",
        )
        obs.emit_session_end(0)
        assert _named(p, "session_end") == []

    def test_calling_it_twice_emits_once(self, tmp_path, clean_logging):
        p = self._start(tmp_path)
        obs.emit_session_end(0)
        obs.emit_session_end(1)
        assert len(_named(p, "session_end")) == 1


class TestTheCliClosesTheSessionHoweverItExits:
    """``main`` wraps the dispatch in a finally.

    A run that dies on a provider 402, a traceback or a Ctrl-C is the run an
    operator most needs to see the end of — and a crash is precisely when a
    ``return``-based emit would be skipped.
    """

    def _run_main(self, monkeypatch, outcome):
        from harness import cli

        seen: list[int] = []
        monkeypatch.setattr(
            cli, "_main_dispatch", outcome, raising=True,
        )
        monkeypatch.setattr(
            "harness.observability.emit_session_end",
            lambda code: seen.append(code), raising=True,
        )
        return seen

    def test_a_clean_return_closes_with_its_code(self, monkeypatch):
        from harness import cli

        seen = self._run_main(monkeypatch, lambda: 0)
        assert cli.main() == 0
        assert seen == [0]

    def test_a_nonzero_return_closes_with_that_code(self, monkeypatch):
        from harness import cli

        seen = self._run_main(monkeypatch, lambda: 2)
        assert cli.main() == 2
        assert seen == [2]

    def test_a_crash_still_closes_the_session(self, monkeypatch):
        """The 402 case: before this, the run simply stopped writing and
        looked identical to one still in flight."""
        from harness import cli

        def _boom():
            raise RuntimeError("provider 402")

        seen = self._run_main(monkeypatch, _boom)
        with pytest.raises(RuntimeError):
            cli.main()
        assert seen == [1]

    def test_a_keyboard_interrupt_still_closes_the_session(self, monkeypatch):
        from harness import cli

        def _ctrl_c():
            raise KeyboardInterrupt

        seen = self._run_main(monkeypatch, _ctrl_c)
        with pytest.raises(KeyboardInterrupt):
            cli.main()
        assert seen == [1]

    def test_an_explicit_sys_exit_carries_its_code(self, monkeypatch):
        from harness import cli

        def _exit():
            raise SystemExit(3)

        seen = self._run_main(monkeypatch, _exit)
        with pytest.raises(SystemExit):
            cli.main()
        assert seen == [3]


# ---------------------------------------------------------------------------
# 2. Producer → consumer, with nothing faked in between
# ---------------------------------------------------------------------------

class TestTheDashboardReadsWhatTheHarnessWrote:
    """The join the old tests could not make, because they wrote the
    producer's output by hand."""

    def _real_log(self, tmp_path):
        logs = tmp_path / "logs"
        obs.configure_logging(
            session_id="sess-real", log_dir=str(logs), console_level="OFF",
            workspace_path="/tmp/my-repo",
        )
        obs.emit_event("llm_call", cost_usd=0.02, tokens_in=100,
                       prompt_tokens=400, tokens_out=50)
        obs.emit_session_end(0)
        return _cfg(tmp_path, log_dir=str(logs))

    def test_the_session_is_listed(self, tmp_path, clean_logging):
        cfg = self._real_log(tmp_path)
        assert [s.session_id for s in list_sessions(cfg)] == ["sess-real"]

    def test_started_and_ended_are_populated(self, tmp_path, clean_logging):
        """Both columns were blank in the live UI."""
        cfg = self._real_log(tmp_path)
        s = list_sessions(cfg)[0]
        assert s.started_at and s.ended_at

    def test_the_exit_code_comes_through(self, tmp_path, clean_logging):
        cfg = self._real_log(tmp_path)
        assert list_sessions(cfg)[0].exit_code == 0

    def test_the_workspace_comes_through(self, tmp_path, clean_logging):
        cfg = self._real_log(tmp_path)
        assert list_sessions(cfg)[0].workspace_path == "/tmp/my-repo"

    def test_a_finished_run_is_not_reported_as_running(self, tmp_path, clean_logging):
        cfg = self._real_log(tmp_path)
        assert list_running_sessions(cfg) == []

    def test_an_unfinished_run_IS_reported_as_running(self, tmp_path, clean_logging):
        logs = tmp_path / "logs"
        obs.configure_logging(
            session_id="sess-live", log_dir=str(logs), console_level="OFF",
            workspace_path="/tmp/my-repo",
        )
        obs.emit_event("llm_call", cost_usd=0.01)
        rows = list_running_sessions(_cfg(tmp_path, log_dir=str(logs)))
        assert [r["session_id"] for r in rows] == ["sess-live"]
        assert rows[0]["workspace_path"] == "/tmp/my-repo"

    def test_a_dashboard_boot_log_is_not_a_session(self, tmp_path, clean_logging):
        logs = tmp_path / "logs"
        obs.configure_logging(
            session_id="web-boot", log_dir=str(logs), console_level="OFF",
        )
        obs.emit_event("llm_call", cost_usd=0.0)
        assert list_running_sessions(_cfg(tmp_path, log_dir=str(logs))) == []


# ---------------------------------------------------------------------------
# 3. timestamp vs ts
# ---------------------------------------------------------------------------

class TestEventTimestamp:
    def test_an_explicit_timestamp_wins(self):
        assert _event_timestamp({"timestamp": "A", "ts": "B"}) == "A"

    def test_the_formatter_field_is_accepted(self):
        """Every ordinary log line carries ``ts`` and no ``timestamp``.
        Reading only the latter is why the no-session_start fallback always
        produced an empty string."""
        assert _event_timestamp({"ts": "B"}) == "B"

    def test_neither_is_empty_not_none(self):
        assert _event_timestamp({}) == ""

    def test_a_log_with_only_ts_still_gets_a_start_and_end(self, tmp_path):
        """Repairs sessions ALREADY on disk, which predate the lifecycle
        events entirely."""
        logs = tmp_path / "logs"
        os.makedirs(logs)
        with open(logs / "old.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"ts": "2026-11-04T12:21:00+00:00",
                                "level": "INFO", "msg": "start"}) + "\n")
            f.write(json.dumps({"ts": "2026-11-04T12:24:00+00:00",
                                "event": "llm_call", "cost_usd": 0.05}) + "\n")
        s = list_sessions(_cfg(tmp_path, log_dir=str(logs)))[0]
        assert s.started_at == "2026-11-04T12:21:00+00:00"
        assert s.ended_at == "2026-11-04T12:24:00+00:00"


# ---------------------------------------------------------------------------
# 4. The latest.jsonl symlink
# ---------------------------------------------------------------------------

class TestTheLatestSymlinkIsNotASession:
    def _with_symlink(self, tmp_path):
        logs = tmp_path / "logs"
        os.makedirs(logs, exist_ok=True)
        real = logs / "sess-001.jsonl"
        with open(real, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "event": "session_start", "timestamp": "2026-11-04T10:00:00+00:00",
                "workspace_path": "/tmp/repo"}) + "\n")
            f.write(json.dumps({
                "event": "llm_call", "timestamp": "2026-11-04T10:00:01+00:00",
                "tokens_in": 100, "tokens_out": 50, "cost_usd": 0.0572}) + "\n")
        os.symlink(real.name, logs / "latest.jsonl")
        return _cfg(tmp_path, log_dir=str(logs))

    def test_only_the_real_session_is_listed(self, tmp_path):
        """Live before the fix: a phantom row named "latest" beside
        ``01a1a67e-…``, with identical numbers."""
        cfg = self._with_symlink(tmp_path)
        assert [s.session_id for s in list_sessions(cfg)] == ["sess-001"]

    def test_the_cost_is_not_doubled(self, tmp_path):
        """Four consumers summed this list; each counted the newest run
        twice."""
        cfg = self._with_symlink(tmp_path)
        assert sum(s.total_cost_usd for s in list_sessions(cfg)) == pytest.approx(
            0.0572)

    def test_the_tokens_are_not_doubled(self, tmp_path):
        cfg = self._with_symlink(tmp_path)
        assert sum(s.total_input_tokens for s in list_sessions(cfg)) == 100

    def test_the_symlink_is_not_a_running_session_either(self, tmp_path):
        """``list_running_sessions`` has its own glob and the same hole."""
        cfg = self._with_symlink(tmp_path)
        assert [r["session_id"] for r in list_running_sessions(cfg)] == ["sess-001"]

    def test_any_symlink_is_skipped_not_just_that_name(self, tmp_path):
        """Resolve-and-skip rather than name-matching, so a convenience link
        an operator drops in the log dir is handled too."""
        cfg = self._with_symlink(tmp_path)
        logs = tmp_path / "logs"
        os.symlink("sess-001.jsonl", logs / "yesterday.jsonl")
        assert [s.session_id for s in list_sessions(cfg)] == ["sess-001"]

    def test_the_real_log_is_still_found_when_no_symlink_exists(self, tmp_path):
        """The skip must not be so eager that it drops ordinary files."""
        logs = tmp_path / "logs"
        os.makedirs(logs)
        with open(logs / "sess-002.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"event": "llm_call", "cost_usd": 0.01}) + "\n")
        assert len(list_sessions(_cfg(tmp_path, log_dir=str(logs)))) == 1

    def test_the_writer_really_does_create_a_symlink(self, tmp_path, clean_logging):
        """Pins the premise. If configure_logging ever stops linking, this
        whole class is guarding nothing and should say so."""
        obs.configure_logging(
            session_id="sess-real", log_dir=str(tmp_path), console_level="OFF",
            workspace_path="/tmp/repo",
        )
        link = tmp_path / "latest.jsonl"
        assert link.is_symlink()
        assert os.readlink(link) == "sess-real.jsonl"
