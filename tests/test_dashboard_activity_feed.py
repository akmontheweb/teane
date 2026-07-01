"""Phase 2 regression: consumer-facing activity feed + live cost meter.

Scope of assertions:
1. `_initial_cost_snapshot` reads the on-disk session log and returns
   a zeroed dict when the log is missing or empty, and the correct
   running totals when llm_call events are present. This is the seed
   that keeps a mid-run page reload's cost meter from resetting.
2. `_render_activity_feed_and_cost` embeds that snapshot as the
   card's initial state and wires the Alpine `x-data` + SSE bus
   listener (`teane:sse:event`) so the meter and feed hydrate without
   opening a second EventSource.
3. `_render_session_with_hitl` sequences the activity feed *between*
   the HITL sticky banner (top) and the technical event stream
   (bottom) — the intended IA of Phase 2.
4. `_ACTIVITY_EVENT_LABELS` covers the event types the harness
   already emits, so real runs don't drop rows into a mystery
   fallback bucket.
"""

from __future__ import annotations

import json

from harness.dashboard import (
    _ACTIVITY_EVENT_LABELS,
    _initial_cost_snapshot,
    _render_activity_feed_and_cost,
    _render_session_with_hitl,
    DashboardConfig,
)


def _cfg(tmp_path):
    return DashboardConfig.from_config(
        {
            "dashboard": {
                "log_dir": str(tmp_path / "logs"),
                "metrics_dir": str(tmp_path / "metrics"),
                "memory_dir": str(tmp_path / "memory"),
                "repo_index_dir": str(tmp_path / "idx"),
                "schedule_db": str(tmp_path / "schedule.db"),
                "enabled": True,
            }
        }
    )


def _write_events(tmp_path, session_id, events):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / f"{session_id}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for evt in events:
            f.write(json.dumps(evt) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# _initial_cost_snapshot
# ---------------------------------------------------------------------------

def test_initial_cost_zero_when_log_missing(tmp_path):
    """A fresh session with no log file → all-zero snapshot; must not
    raise (fatal on the first render otherwise)."""
    snap = _initial_cost_snapshot(_cfg(tmp_path), "sess-never-run")
    assert snap == {
        "cost": 0.0,
        "calls": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "cached_tokens": 0,
    }


def test_initial_cost_aggregates_llm_events(tmp_path):
    """Server-side seeding must sum every historic ``llm_call`` so a
    mid-run refresh keeps the cumulative total. The client only adds
    events observed AFTER the SSE connects — the seed is what closes
    the gap."""
    _write_events(
        tmp_path,
        "sess-1",
        [
            {"event": "session_start", "ts": "2026-06-30T10:00:00Z"},
            {"event": "llm_call", "ts": "2026-06-30T10:00:01Z",
             "cost_usd": 0.001, "tokens_in": 100, "tokens_out": 50,
             "cached_tokens": 20},
            {"event": "llm_call", "ts": "2026-06-30T10:00:02Z",
             "cost_usd": 0.0025, "tokens_in": 200, "tokens_out": 80},
            {"event": "tool_call_succeeded", "ts": "2026-06-30T10:00:03Z"},
        ],
    )
    snap = _initial_cost_snapshot(_cfg(tmp_path), "sess-1")
    assert snap["calls"] == 2
    assert snap["cost"] == 0.0035
    assert snap["tokens_in"] == 300
    assert snap["tokens_out"] == 130
    assert snap["cached_tokens"] == 20


# ---------------------------------------------------------------------------
# _render_activity_feed_and_cost
# ---------------------------------------------------------------------------

def test_activity_feed_embeds_seeded_snapshot(tmp_path):
    _write_events(
        tmp_path,
        "sess-render",
        [{"event": "llm_call", "ts": "t", "cost_usd": 0.123,
          "tokens_in": 100, "tokens_out": 50}],
    )
    html_body = _render_activity_feed_and_cost(_cfg(tmp_path), "sess-render")
    # The formatted printed cost sits inside the meter as a fallback so
    # users see something before Alpine hydrates.
    assert "$0.1230" in html_body
    # The Alpine bindings are present.
    assert "x-data=\"teaneCostMeter" in html_body
    assert "x-data=\"teaneActivityFeed" in html_body
    # Snapshot is passed via data-initial so JS boot picks it up.
    assert "data-initial=" in html_body
    # Both components subscribe to the shared window bus rather than
    # opening their own EventSource.
    assert "teane:sse:event.window" in html_body


def test_activity_feed_embeds_label_table_for_client_rendering(tmp_path):
    html_body = _render_activity_feed_and_cost(_cfg(tmp_path), "sess-x")
    # The full label table gets serialised into the feed's data-labels
    # so client code can look up icon + plain-English text without a
    # round-trip. Assert at least one entry is present.
    assert "data-labels=" in html_body
    assert "hitl_pending" in html_body
    assert "tool_call_succeeded" in html_body


def test_activity_feed_targets_the_existing_sse_endpoint(tmp_path):
    html_body = _render_activity_feed_and_cost(_cfg(tmp_path), "sess-x")
    # The SSE URL exposed on the card matches the existing endpoint
    # `_route_session_events_sse_marker` serves (`/api/sessions/<sid>/events`).
    # No new server route is required for Phase 2.
    assert "data-sse-url='/api/sessions/sess-x/events'" in html_body


# ---------------------------------------------------------------------------
# Placement in _render_session_with_hitl
# ---------------------------------------------------------------------------

def test_session_detail_places_activity_feed_between_hitl_and_event_stream(tmp_path):
    """Consumer IA: sticky HITL at top → live plain-English activity
    → technical raw event stream at the bottom. The order matters for
    at-a-glance comprehension."""
    _write_events(
        tmp_path,
        "sess-order",
        [{"event": "session_start", "ts": "t"}],
    )
    html_body = _render_session_with_hitl(_cfg(tmp_path), "sess-order")
    hitl_idx = html_body.find("hitl-pending-slot")
    feed_idx = html_body.find("teaneActivityFeed")
    raw_idx = html_body.find("id='event-stream'")
    assert hitl_idx != -1
    assert feed_idx != -1
    assert raw_idx != -1
    assert hitl_idx < feed_idx < raw_idx, (
        "Expected order: HITL banner (top), activity feed (middle), "
        "raw event stream (bottom). "
        f"Got hitl={hitl_idx}, feed={feed_idx}, raw={raw_idx}."
    )


# ---------------------------------------------------------------------------
# Label table completeness
# ---------------------------------------------------------------------------

def test_label_table_covers_events_the_harness_actually_emits():
    """The event names below are the ones observability.emit_event
    produces at the sites the Phase-1 exploration identified. If a new
    event lands and isn't in the table, the feed still renders it
    (falls through to a generic row), but the label loses its
    plain-English polish — so this test is the reminder to update the
    table when a new event name ships.
    """
    required = {
        "llm_call",
        "session_start",
        "session_end",
        "build_start",
        "build_end",
        "node_transition",
        "tool_call_succeeded",
        "tool_call_failed",
        "hitl_pending",
        "hitl_resolved",
        "deployment_outcome",
    }
    missing = required - set(_ACTIVITY_EVENT_LABELS.keys())
    assert not missing, f"Add labels for these harness event names: {sorted(missing)}"


def test_label_severity_values_are_within_the_documented_set():
    """The `severity` field maps to a CSS modifier (`activity-row--<sev>`).
    Restrict values to the ones tokens.css actually styles so we don't
    drop rows into an unstyled bucket."""
    allowed = {"", "ok", "hitl", "fail"}
    for name, meta in _ACTIVITY_EVENT_LABELS.items():
        assert meta.get("severity", "") in allowed, (
            f"{name!r} uses severity {meta.get('severity')!r} which "
            f"isn't in tokens.css. Add a style or pick from {allowed}."
        )
