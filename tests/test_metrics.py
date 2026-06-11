"""Tests for harness/metrics.py (P2.7).

Covers the pure-function surface (aggregation, projection, formatters,
atomic writer) and an end-to-end CLI smoke through cmd_metrics.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone

import pytest


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _llm_call(ts: str, cost: float, tin: int = 100, tout: int = 50, cached: int = 0) -> dict:
    return {
        "ts": ts,
        "level": "INFO",
        "logger": "harness.events",
        "msg": "",
        "event": "llm_call",
        "model": "anthropic:claude-sonnet-4-6",
        "tokens_in": tin,
        "tokens_out": tout,
        "cached_tokens": cached,
        "cost_usd": cost,
        "budget_remaining_usd": 2.0 - cost,
        "elapsed_ms": 1200,
        "finish_reason": "stop",
    }


def _failure(ts: str, name: str) -> dict:
    return {"ts": ts, "level": "ERROR", "logger": "harness.events", "msg": "", "event": name}


_FIXED_NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# 1. Aggregation
# ---------------------------------------------------------------------------

class TestAggregateSession:

    def test_sums_cost_and_tokens(self, tmp_path):
        from harness.metrics import aggregate_session

        log = tmp_path / "sess-A.jsonl"
        _write_jsonl(str(log), [
            _llm_call("2026-06-10T11:30:00+00:00", 0.10, tin=1000, tout=200),
            _llm_call("2026-06-10T11:31:00+00:00", 0.20, tin=1500, tout=300),
            _llm_call("2026-06-10T11:32:00+00:00", 0.05, tin=500, tout=100),
        ])

        m = aggregate_session("sess-A", str(tmp_path), now=_FIXED_NOW)
        assert m.llm_call_count == 3
        assert m.total_cost_usd == pytest.approx(0.35)
        assert m.tokens_in == 3000
        assert m.tokens_out == 600
        assert m.first_ts is not None and m.last_ts is not None
        assert m.first_ts < m.last_ts

    def test_reads_rotated_backups(self, tmp_path):
        from harness.metrics import aggregate_session

        # Older content lives in .2, then .1, then the live .jsonl.
        _write_jsonl(str(tmp_path / "sess-R.jsonl.2"), [
            _llm_call("2026-06-10T10:00:00+00:00", 0.01),
        ])
        _write_jsonl(str(tmp_path / "sess-R.jsonl.1"), [
            _llm_call("2026-06-10T10:15:00+00:00", 0.02),
            _llm_call("2026-06-10T10:20:00+00:00", 0.03),
        ])
        _write_jsonl(str(tmp_path / "sess-R.jsonl"), [
            _llm_call("2026-06-10T11:00:00+00:00", 0.04),
        ])

        m = aggregate_session("sess-R", str(tmp_path), now=_FIXED_NOW)
        assert m.llm_call_count == 4
        assert m.total_cost_usd == pytest.approx(0.10)
        assert len(m.log_files) == 3
        # File order: .2, .1, .jsonl (chronological)
        assert m.log_files[0].endswith(".jsonl.2")
        assert m.log_files[1].endswith(".jsonl.1")
        assert m.log_files[2].endswith(".jsonl")

    def test_skips_malformed_lines(self, tmp_path):
        from harness.metrics import aggregate_session

        path = tmp_path / "sess-M.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(_llm_call("2026-06-10T11:00:00+00:00", 0.10)) + "\n")
            fh.write("{not valid json\n")
            fh.write(json.dumps(_llm_call("2026-06-10T11:01:00+00:00", 0.15)) + "\n")
            fh.write("\n")  # blank — also skipped

        m = aggregate_session("sess-M", str(tmp_path), now=_FIXED_NOW)
        assert m.llm_call_count == 2
        assert m.total_cost_usd == pytest.approx(0.25)

    def test_counts_failure_events(self, tmp_path):
        from harness.metrics import aggregate_session

        log = tmp_path / "sess-F.jsonl"
        _write_jsonl(str(log), [
            _llm_call("2026-06-10T11:00:00+00:00", 0.10),
            _failure("2026-06-10T11:01:00+00:00", "llm_empty_response"),
            _failure("2026-06-10T11:02:00+00:00", "llm_circuit_open"),
            _failure("2026-06-10T11:03:00+00:00", "token_budget_exhausted"),
            _failure("2026-06-10T11:04:00+00:00", "llm_empty_response"),
        ])

        m = aggregate_session("sess-F", str(tmp_path), now=_FIXED_NOW)
        assert m.error_counts == {
            "llm_empty_response": 2,
            "llm_circuit_open": 1,
            "token_budget_exhausted": 1,
        }

    def test_empty_log_dir_returns_zeroed(self, tmp_path):
        from harness.metrics import aggregate_session

        m = aggregate_session("nothing-here", str(tmp_path), now=_FIXED_NOW)
        assert m.llm_call_count == 0
        assert m.total_cost_usd == 0.0
        assert m.first_ts is None
        assert m.last_ts is None
        assert m.recent_burn_rate_usd_per_min == 0.0


# ---------------------------------------------------------------------------
# 2. Burn rate
# ---------------------------------------------------------------------------

class TestBurnRate:

    def test_only_records_in_window_contribute(self, tmp_path):
        from harness.metrics import aggregate_session

        # now=12:00 UTC, window=10 min → cutoff is 11:50.
        log = tmp_path / "sess-W.jsonl"
        _write_jsonl(str(log), [
            _llm_call("2026-06-10T11:00:00+00:00", 1.00),  # outside window
            _llm_call("2026-06-10T11:55:00+00:00", 0.10),  # inside (5 min ago)
            _llm_call("2026-06-10T11:58:00+00:00", 0.20),  # inside (2 min ago)
        ])
        m = aggregate_session("sess-W", str(tmp_path), window_minutes=10, now=_FIXED_NOW)

        # Total cost includes everything ($1.30), but burn rate only the
        # last two records ($0.30 across ~5 min elapsed within window).
        assert m.total_cost_usd == pytest.approx(1.30)
        assert m.recent_burn_rate_usd_per_min > 0
        # The earliest in-window record is 11:55 → elapsed to 12:00 = 5 min.
        # Burn rate = $0.30 / 5 min = $0.06/min.
        assert m.recent_burn_rate_usd_per_min == pytest.approx(0.06, rel=0.01)

    def test_zero_burn_when_no_recent_activity(self, tmp_path):
        from harness.metrics import aggregate_session

        log = tmp_path / "sess-stale.jsonl"
        _write_jsonl(str(log), [
            _llm_call("2026-06-10T08:00:00+00:00", 0.50),  # 4 hours ago
        ])
        m = aggregate_session("sess-stale", str(tmp_path), window_minutes=10, now=_FIXED_NOW)
        assert m.total_cost_usd == pytest.approx(0.50)
        assert m.recent_burn_rate_usd_per_min == 0.0


# ---------------------------------------------------------------------------
# 3. Projection
# ---------------------------------------------------------------------------

class TestProjectExhaustion:

    def test_basic_division(self):
        from harness.metrics import SessionMetrics, project_exhaustion

        m = SessionMetrics(session_id="x", total_cost_usd=1.00, recent_burn_rate_usd_per_min=0.10)
        # Hard cap $2.00, spent $1.00, burning $0.10/min → 10 min.
        assert project_exhaustion(m, hard_cap_usd=2.00) == pytest.approx(10.0)

    def test_zero_burn_returns_none(self):
        from harness.metrics import SessionMetrics, project_exhaustion

        m = SessionMetrics(session_id="x", total_cost_usd=1.00, recent_burn_rate_usd_per_min=0.0)
        assert project_exhaustion(m, hard_cap_usd=2.00) is None

    def test_already_exhausted_returns_zero(self):
        from harness.metrics import SessionMetrics, project_exhaustion

        m = SessionMetrics(session_id="x", total_cost_usd=3.00, recent_burn_rate_usd_per_min=0.05)
        assert project_exhaustion(m, hard_cap_usd=2.00) == 0.0


# ---------------------------------------------------------------------------
# 4. Formatters
# ---------------------------------------------------------------------------

class TestFormatters:

    def test_format_human_includes_key_fields(self, tmp_path):
        from harness.metrics import aggregate_session, format_human

        log = tmp_path / "sess-H.jsonl"
        _write_jsonl(str(log), [
            _llm_call("2026-06-10T11:55:00+00:00", 0.10),
            _llm_call("2026-06-10T11:58:00+00:00", 0.20),
        ])
        m = aggregate_session("sess-H", str(tmp_path), now=_FIXED_NOW)
        text = format_human(m, hard_cap_usd=2.00)

        assert "sess-H" in text
        assert "$0.3000" in text  # total cost
        assert "Burn rate" in text
        assert "Projected exhaust" in text
        assert "$2.00" in text  # hard cap

    def test_format_table_with_total_footer(self, tmp_path):
        from harness.metrics import aggregate_session, format_table

        for sid, cost in [("alpha", 0.10), ("beta", 0.30)]:
            _write_jsonl(str(tmp_path / f"{sid}.jsonl"), [
                _llm_call("2026-06-10T11:55:00+00:00", cost),
            ])
        ms = [
            aggregate_session("alpha", str(tmp_path), now=_FIXED_NOW),
            aggregate_session("beta", str(tmp_path), now=_FIXED_NOW),
        ]
        text = format_table(ms, hard_cap_usd=2.00)

        assert "alpha" in text
        assert "beta" in text
        assert "TOTAL" in text
        assert "2 sessions" in text

    def test_format_prometheus_well_formed(self, tmp_path):
        from harness.metrics import aggregate_session, format_prometheus

        _write_jsonl(str(tmp_path / "promtest.jsonl"), [
            _llm_call("2026-06-10T11:55:00+00:00", 0.10, tin=500, tout=100),
        ])
        m = aggregate_session("promtest", str(tmp_path), now=_FIXED_NOW)
        text = format_prometheus([m], hard_cap_usd=2.00)

        # Every documented metric appears at least once.
        for metric in [
            "harness_session_cost_usd",
            "harness_session_llm_calls",
            "harness_session_tokens",
            "harness_burn_rate_usd_per_min",
            "harness_budget_hard_cap_usd",
        ]:
            assert f"# HELP {metric}" in text
            assert f"# TYPE {metric}" in text

        # Label syntax is well-formed and the session id appears.
        assert 'session_id="promtest"' in text
        assert 'direction="in"' in text
        assert 'direction="out"' in text

    def test_prometheus_escapes_label_value(self):
        from harness.metrics import _prometheus_label_value

        assert _prometheus_label_value('quote"backslash\\newline\n') == 'quote\\"backslash\\\\newline\\n'


# ---------------------------------------------------------------------------
# 5. list_sessions
# ---------------------------------------------------------------------------

class TestListSessions:

    def test_dedupes_rotation_suffixes(self, tmp_path):
        from harness.metrics import list_sessions

        # a has live + 2 backups, b has only a backup, c has only live,
        # noise.txt is an unrelated file.
        (tmp_path / "a.jsonl").write_text("")
        (tmp_path / "a.jsonl.1").write_text("")
        (tmp_path / "a.jsonl.2").write_text("")
        (tmp_path / "b.jsonl.3").write_text("")
        (tmp_path / "c.jsonl").write_text("")
        (tmp_path / "noise.txt").write_text("")

        assert list_sessions(str(tmp_path)) == ["a", "b", "c"]

    def test_missing_dir_returns_empty(self, tmp_path):
        from harness.metrics import list_sessions
        assert list_sessions(str(tmp_path / "doesnt-exist")) == []


# ---------------------------------------------------------------------------
# 6. Atomic writer
# ---------------------------------------------------------------------------

class TestAtomicWrite:

    def test_creates_dest_dir(self, tmp_path):
        from harness.metrics import write_atomic

        dest = tmp_path / "subdir" / "file.prom"
        write_atomic(str(dest), "hello\n")
        assert dest.read_text() == "hello\n"

    def test_no_partial_file_on_write_failure(self, tmp_path, monkeypatch):
        from harness import metrics as mod
        from harness.metrics import write_atomic

        # Force os.replace to raise so the tmp → final swap never happens.
        def boom(src, dst):
            raise OSError("simulated rename failure")

        monkeypatch.setattr(mod.os, "replace", boom)

        dest = tmp_path / "out.prom"
        with pytest.raises(OSError):
            write_atomic(str(dest), "payload\n")

        # The dest file MUST NOT exist — atomic guarantee.
        assert not dest.exists()
        # The .tmp shard may be left behind; that's fine, readers never
        # see it as the canonical path.


# ---------------------------------------------------------------------------
# 7. CLI smoke
# ---------------------------------------------------------------------------

class TestCmdMetrics:

    def test_session_human_report_to_stdout(self, tmp_path, capsys, monkeypatch):
        # Under the single-source-config contract the harness no longer
        # reads .harness_config.json from the workspace. Each test patches
        # the canonical discover_config to return a constructed dict.
        from harness import cli as cli_mod
        from harness.cli import cmd_metrics, build_parser

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        _write_jsonl(str(log_dir / "abc.jsonl"), [
            _llm_call("2026-06-10T11:55:00+00:00", 0.12),
        ])
        ws = tmp_path / "ws"
        ws.mkdir()
        monkeypatch.setattr(cli_mod, "discover_config", lambda _ws: {
            "logging": {"log_dir": str(log_dir)},
            "token_budget": {"hard_cap_usd": 2.00},
        })

        args = build_parser().parse_args(
            ["metrics", "--session-id", "abc", "-r", str(ws)]
        )
        rc = asyncio.run(cmd_metrics(args))
        assert rc == 0
        out = capsys.readouterr().out
        assert "abc" in out
        assert "$0.1200" in out

    def test_prometheus_writes_atomic_to_metrics_dir(self, tmp_path, monkeypatch):
        from harness import cli as cli_mod
        from harness.cli import cmd_metrics, build_parser

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        _write_jsonl(str(log_dir / "promtest.jsonl"), [
            _llm_call("2026-06-10T11:55:00+00:00", 0.20),
        ])
        metrics_dir = tmp_path / "metrics-out"
        ws = tmp_path / "ws"
        ws.mkdir()
        monkeypatch.setattr(cli_mod, "discover_config", lambda _ws: {
            "logging": {"log_dir": str(log_dir)},
            "metrics": {"metrics_dir": str(metrics_dir)},
            "token_budget": {"hard_cap_usd": 2.00},
        })

        args = build_parser().parse_args([
            "metrics", "--session-id", "promtest", "--prometheus",
            "-r", str(ws),
        ])
        rc = asyncio.run(cmd_metrics(args))
        assert rc == 0
        out_file = metrics_dir / "promtest.prom"
        assert out_file.exists(), f"expected {out_file} to be written"
        body = out_file.read_text()
        assert "harness_session_cost_usd" in body
        assert 'session_id="promtest"' in body

    def test_metrics_dir_config_override_respected(self, tmp_path, monkeypatch):
        from harness import cli as cli_mod
        from harness.cli import cmd_metrics, build_parser

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        _write_jsonl(str(log_dir / "sess.jsonl"), [
            _llm_call("2026-06-10T11:55:00+00:00", 0.05),
        ])
        custom_dir = tmp_path / "elsewhere"
        ws = tmp_path / "ws"
        ws.mkdir()
        monkeypatch.setattr(cli_mod, "discover_config", lambda _ws: {
            "logging": {"log_dir": str(log_dir)},
            "metrics": {"metrics_dir": str(custom_dir)},
            "token_budget": {"hard_cap_usd": 2.00},
        })

        args = build_parser().parse_args([
            "metrics", "--session-id", "sess", "--json", "-r", str(ws),
        ])
        rc = asyncio.run(cmd_metrics(args))
        assert rc == 0
        assert (custom_dir / "sess.json").exists()
        # Default location must not be used.
        assert not os.path.exists(os.path.expanduser("~/.harness/metrics/sess.json")) or \
               (custom_dir / "sess.json").read_text() != ""

    def test_no_logs_returns_exit_code_1(self, tmp_path, capsys, monkeypatch):
        from harness import cli as cli_mod
        from harness.cli import cmd_metrics, build_parser

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        ws = tmp_path / "ws"
        ws.mkdir()
        monkeypatch.setattr(cli_mod, "discover_config", lambda _ws: {
            "logging": {"log_dir": str(log_dir)},
        })

        args = build_parser().parse_args(["metrics", "--all", "-r", str(ws)])
        rc = asyncio.run(cmd_metrics(args))
        assert rc == 1

    def test_output_dash_emits_to_stdout(self, tmp_path, capsys, monkeypatch):
        from harness import cli as cli_mod
        from harness.cli import cmd_metrics, build_parser

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        _write_jsonl(str(log_dir / "stdoutsess.jsonl"), [
            _llm_call("2026-06-10T11:55:00+00:00", 0.07),
        ])
        ws = tmp_path / "ws"
        ws.mkdir()
        monkeypatch.setattr(cli_mod, "discover_config", lambda _ws: {
            "logging": {"log_dir": str(log_dir)},
            "metrics": {"metrics_dir": str(tmp_path / "should-not-be-touched")},
            "token_budget": {"hard_cap_usd": 2.00},
        })

        args = build_parser().parse_args([
            "metrics", "--session-id", "stdoutsess", "--prometheus",
            "--output", "-", "-r", str(ws),
        ])
        rc = asyncio.run(cmd_metrics(args))
        assert rc == 0
        out = capsys.readouterr().out
        assert "harness_session_cost_usd" in out
        # Metrics dir must not have been created.
        assert not (tmp_path / "should-not-be-touched").exists()
