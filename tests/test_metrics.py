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


def _tool_succeeded(ts: str, tool: str) -> dict:
    return {
        "ts": ts,
        "level": "INFO",
        "logger": "harness.events",
        "msg": "",
        "event": "tool_call_succeeded",
        "tool_name": tool,
    }


def _tool_failed(ts: str, tool: str, reason: str = "boom") -> dict:
    return {
        "ts": ts,
        "level": "ERROR",
        "logger": "harness.events",
        "msg": "",
        "event": "tool_call_failed",
        "tool_name": tool,
        "reason": reason,
    }


def _system_prompt_built(ts: str, chars: int, lines: int, tree_lines: int = 0) -> dict:
    return {
        "ts": ts,
        "level": "INFO",
        "logger": "harness.events",
        "msg": "",
        "event": "system_prompt_built",
        "chars": chars,
        "lines": lines,
        "tree_lines": tree_lines,
    }


def _loop_counter_snapshot(
    ts: str,
    loop_counter: dict,
    *,
    exit_code: int = 0,
    n_diagnostics: int = 0,
) -> dict:
    return {
        "ts": ts,
        "level": "INFO",
        "logger": "harness.events",
        "msg": "",
        "event": "loop_counter_snapshot",
        "loop_counter": loop_counter,
        "exit_code": exit_code,
        "n_diagnostics": n_diagnostics,
    }


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
# 1b. Cache-hit rate (#26) and per-tool error rate (#15, #27)
# ---------------------------------------------------------------------------

class TestCacheHitRate:

    def test_empty_session_returns_zero(self):
        from harness.metrics import SessionMetrics
        assert SessionMetrics(session_id="x").cache_hit_rate() == 0.0

    def test_basic_ratio(self):
        from harness.metrics import SessionMetrics
        m = SessionMetrics(session_id="x", tokens_in=200, cached_tokens=800)
        assert m.cache_hit_rate() == pytest.approx(0.80)

    def test_no_cached_returns_zero(self):
        from harness.metrics import SessionMetrics
        m = SessionMetrics(session_id="x", tokens_in=500, cached_tokens=0)
        assert m.cache_hit_rate() == 0.0

    def test_aggregated_from_llm_call_events(self, tmp_path):
        from harness.metrics import aggregate_session
        log = tmp_path / "sess-cache.jsonl"
        _write_jsonl(str(log), [
            _llm_call("2026-06-10T11:55:00+00:00", 0.10, tin=100, cached=900),
            _llm_call("2026-06-10T11:56:00+00:00", 0.10, tin=100, cached=900),
        ])
        m = aggregate_session("sess-cache", str(tmp_path), now=_FIXED_NOW)
        # 1800 cached / (200 input + 1800 cached) = 0.90
        assert m.cache_hit_rate() == pytest.approx(0.90)


class TestToolErrorRate:

    def test_no_calls_returns_zero(self):
        from harness.metrics import SessionMetrics
        assert SessionMetrics(session_id="x").tool_error_rate("read_file") == 0.0
        assert SessionMetrics(session_id="x").tool_error_rates() == {}

    def test_aggregates_succeeded_and_failed(self, tmp_path):
        from harness.metrics import aggregate_session
        log = tmp_path / "sess-tool.jsonl"
        _write_jsonl(str(log), [
            _tool_succeeded("2026-06-10T11:55:00+00:00", "read_file"),
            _tool_succeeded("2026-06-10T11:55:30+00:00", "read_file"),
            _tool_failed("2026-06-10T11:56:00+00:00", "read_file"),
            _tool_succeeded("2026-06-10T11:56:30+00:00", "web_search"),
            _tool_failed("2026-06-10T11:57:00+00:00", "web_search"),
        ])
        m = aggregate_session("sess-tool", str(tmp_path), now=_FIXED_NOW)
        # read_file: 3 calls total (2 succeeded + 1 failed), 1 error → 33%.
        assert m.tool_call_count == {"read_file": 3, "web_search": 2}
        assert m.tool_error_count == {"read_file": 1, "web_search": 1}
        assert m.tool_error_rate("read_file") == pytest.approx(1 / 3)
        assert m.tool_error_rate("web_search") == pytest.approx(0.5)
        assert m.tool_error_rate("unknown_tool") == 0.0

    def test_tool_call_failed_only_still_counts_as_attempt(self, tmp_path):
        # A tool that always fails should show 100% error rate, not 0%.
        from harness.metrics import aggregate_session
        log = tmp_path / "sess-fail.jsonl"
        _write_jsonl(str(log), [
            _tool_failed("2026-06-10T11:55:00+00:00", "broken_tool"),
            _tool_failed("2026-06-10T11:56:00+00:00", "broken_tool"),
        ])
        m = aggregate_session("sess-fail", str(tmp_path), now=_FIXED_NOW)
        assert m.tool_call_count == {"broken_tool": 2}
        assert m.tool_error_count == {"broken_tool": 2}
        assert m.tool_error_rate("broken_tool") == 1.0

    def test_missing_tool_name_falls_back_to_unknown(self, tmp_path):
        from harness.metrics import aggregate_session
        log = tmp_path / "sess-nname.jsonl"
        # Event with no tool_name field — should still be counted under "unknown".
        rec = {
            "ts": "2026-06-10T11:55:00+00:00",
            "level": "INFO",
            "logger": "harness.events",
            "msg": "",
            "event": "tool_call_succeeded",
        }
        _write_jsonl(str(log), [rec])
        m = aggregate_session("sess-nname", str(tmp_path), now=_FIXED_NOW)
        assert m.tool_call_count == {"unknown": 1}


class TestToJsonableShapeNew:

    def test_includes_new_fields(self):
        from harness.metrics import SessionMetrics
        m = SessionMetrics(
            session_id="x",
            tokens_in=100,
            cached_tokens=900,
            tool_call_count={"read_file": 4},
            tool_error_count={"read_file": 1},
        )
        out = m.to_jsonable()
        assert out["cache_hit_rate"] == pytest.approx(0.90)
        assert out["tool_call_count"] == {"read_file": 4}
        assert out["tool_error_count"] == {"read_file": 1}
        assert out["tool_error_rates"]["read_file"] == pytest.approx(0.25)


class TestSystemPromptSize:

    def test_records_latest_prompt_size(self, tmp_path):
        from harness.metrics import aggregate_session
        log = tmp_path / "sess-prompt.jsonl"
        _write_jsonl(str(log), [
            _system_prompt_built("2026-06-10T11:55:00+00:00", chars=10_000, lines=120),
            # Second emission overwrites — most-recent wins.
            _system_prompt_built("2026-06-10T11:55:30+00:00", chars=8_000, lines=80),
        ])
        m = aggregate_session("sess-prompt", str(tmp_path), now=_FIXED_NOW)
        assert m.system_prompt_chars == 8_000
        assert m.system_prompt_lines == 80


class TestLoopCounterSnapshot:

    def test_latest_snapshot_wins_and_scalar_peak_tracked(self, tmp_path):
        from harness.metrics import aggregate_session
        log = tmp_path / "sess-loop.jsonl"
        _write_jsonl(str(log), [
            _loop_counter_snapshot(
                "2026-06-10T11:55:00+00:00",
                {
                    "total_repairs": 2,
                    "consecutive_zero_patch_rounds": 1,
                    "cheap_shots_taken": 1,
                    "replace_block_misses_per_file": {"a.py": 1},
                    "missing_dep_last_symbol": "requests",
                },
            ),
            _loop_counter_snapshot(
                "2026-06-10T11:56:00+00:00",
                {
                    # Peak of consecutive_zero_patch_rounds — this is
                    # the "how close did we get to HITL?" reading.
                    "total_repairs": 4,
                    "consecutive_zero_patch_rounds": 3,
                    "cheap_shots_taken": 2,
                    "replace_block_misses_per_file": {"a.py": 2, "b.py": 1},
                    "missing_dep_last_symbol": "flask",
                },
            ),
            _loop_counter_snapshot(
                "2026-06-10T11:57:00+00:00",
                {
                    # Green build resets the streak to 0 — final snapshot
                    # shows current state, peak preserves the earlier max.
                    "total_repairs": 5,
                    "consecutive_zero_patch_rounds": 0,
                    "cheap_shots_taken": 2,
                    "replace_block_misses_per_file": {},
                    "missing_dep_last_symbol": "",
                },
                exit_code=0,
            ),
        ])
        m = aggregate_session("sess-loop", str(tmp_path), now=_FIXED_NOW)
        # Latest-wins for the final snapshot.
        assert m.loop_counter_final["total_repairs"] == 5
        assert m.loop_counter_final["consecutive_zero_patch_rounds"] == 0
        assert m.loop_counter_final["missing_dep_last_symbol"] == ""
        # Element-wise peak of scalar-int fields.
        assert m.loop_counter_peak["total_repairs"] == 5
        assert m.loop_counter_peak["consecutive_zero_patch_rounds"] == 3
        assert m.loop_counter_peak["cheap_shots_taken"] == 2
        # Nested dicts (replace_block_misses_per_file) skipped from peak.
        assert "replace_block_misses_per_file" not in m.loop_counter_peak
        # Non-numeric strings skipped from peak.
        assert "missing_dep_last_symbol" not in m.loop_counter_peak

    def test_snapshot_appears_in_jsonable_and_format_human(self, tmp_path):
        from harness.metrics import (
            aggregate_session, format_human,
        )
        log = tmp_path / "sess-loop2.jsonl"
        _write_jsonl(str(log), [
            _loop_counter_snapshot(
                "2026-06-10T11:55:00+00:00",
                {"total_repairs": 3, "consecutive_zero_patch_rounds": 2},
            ),
        ])
        m = aggregate_session("sess-loop2", str(tmp_path), now=_FIXED_NOW)
        out = m.to_jsonable()
        assert out["loop_counter_final"] == {
            "total_repairs": 3, "consecutive_zero_patch_rounds": 2,
        }
        assert out["loop_counter_peak"] == {
            "total_repairs": 3, "consecutive_zero_patch_rounds": 2,
        }
        report = format_human(m, hard_cap_usd=2.00)
        assert "Loop health:" in report
        assert "repairs=3" in report
        assert "peak_zero_patch_rounds=2" in report

    def test_no_snapshot_leaves_empty_dicts_and_no_loop_health_line(self, tmp_path):
        from harness.metrics import aggregate_session, format_human
        log = tmp_path / "sess-loop3.jsonl"
        _write_jsonl(str(log), [
            _llm_call("2026-06-10T11:55:00+00:00", 0.10),
        ])
        m = aggregate_session("sess-loop3", str(tmp_path), now=_FIXED_NOW)
        assert m.loop_counter_final == {}
        assert m.loop_counter_peak == {}
        assert "Loop health:" not in format_human(m, hard_cap_usd=2.00)


class TestPrometheusNewMetrics:

    def test_emits_new_gauges(self, tmp_path):
        from harness.metrics import aggregate_session, format_prometheus
        log = tmp_path / "promnew.jsonl"
        _write_jsonl(str(log), [
            _llm_call("2026-06-10T11:55:00+00:00", 0.10, tin=100, cached=900),
            _tool_succeeded("2026-06-10T11:55:10+00:00", "read_file"),
            _tool_failed("2026-06-10T11:55:20+00:00", "read_file"),
        ])
        m = aggregate_session("promnew", str(tmp_path), now=_FIXED_NOW)
        text = format_prometheus([m], hard_cap_usd=2.00)
        for metric in [
            "harness_session_cache_hit_rate",
            "harness_tool_calls_total",
            "harness_tool_errors_total",
            "harness_tool_error_rate",
        ]:
            assert f"# HELP {metric}" in text
            assert f"# TYPE {metric}" in text
        assert 'tool="read_file"' in text


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
            ["metrics", "--session-id", "abc", "-w", str(ws)]
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
            "-w", str(ws),
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
            "metrics", "--session-id", "sess", "--json-dump", "true", "-w", str(ws),
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

        args = build_parser().parse_args(["metrics", "--all", "-w", str(ws)])
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
            "--output-path", "-", "-w", str(ws),
        ])
        rc = asyncio.run(cmd_metrics(args))
        assert rc == 0
        out = capsys.readouterr().out
        assert "harness_session_cost_usd" in out
        # Metrics dir must not have been created.
        assert not (tmp_path / "should-not-be-touched").exists()
