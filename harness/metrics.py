"""
Cost-metrics aggregation for the harness (P2.7).

Reads per-session JSONL logs at ``~/.harness/logs/<id>.jsonl`` (plus
rotated backups ``<id>.jsonl.1``, ``<id>.jsonl.2``, ...) and reconstructs
per-session cost, token, error, and burn-rate metrics. Pure functions
only — the CLI surface lives in ``harness/cli.py::cmd_metrics`` and calls
into here.

Source of truth for event field names is ``harness/observability.py``:
``JSONFormatter`` injects ``ts``/``level``/``logger``/``msg``/``event``
and merges every ``extra=`` field (cost_usd, tokens_in, tokens_out,
cached_tokens, etc.) at the top level.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# Events that count as "this session burned money/tokens". Currently only
# llm_call, but the list is here so a future provider that emits a
# differently-named event (e.g. embedding_call) can be added with one
# line.
_COST_EVENTS = frozenset({"llm_call"})

# Events that count as "something went wrong worth surfacing in the
# metrics view". Their per-session counts land in
# ``SessionMetrics.error_counts``.
_TRACKED_FAILURE_EVENTS = frozenset({
    "token_budget_exhausted",
    "llm_empty_response",
    "llm_circuit_open",
    "sandbox_start_failed",
    "hitl_gate_blocked",
})

# Default sliding window used by the recent-burn-rate calculation. Ten
# minutes is short enough to see a runaway session in near-real-time but
# long enough to smooth across the spiky per-call cost pattern.
_DEFAULT_WINDOW_MINUTES = 10

# Floor on the elapsed-minutes denominator for the burn-rate division.
# With a window of one record (or a tight cluster), the natural elapsed
# can be zero or sub-second; clamping to one minute avoids absurd
# extrapolated rates while keeping the math honest at higher densities.
_BURN_RATE_FLOOR_MINUTES = 1.0


# ---------------------------------------------------------------------------
# 1. Data shape
# ---------------------------------------------------------------------------

@dataclass
class SessionMetrics:
    """Aggregated cost/usage state for a single harness session."""

    session_id: str
    total_cost_usd: float = 0.0
    llm_call_count: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cached_tokens: int = 0
    error_counts: dict[str, int] = field(default_factory=dict)
    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None
    recent_burn_rate_usd_per_min: float = 0.0
    recent_window_minutes: int = _DEFAULT_WINDOW_MINUTES
    log_files: list[str] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        """Render the dataclass to a JSON-serialisable dict."""
        return {
            "session_id": self.session_id,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "llm_call_count": self.llm_call_count,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cached_tokens": self.cached_tokens,
            "error_counts": dict(self.error_counts),
            "first_ts": self.first_ts.isoformat() if self.first_ts else None,
            "last_ts": self.last_ts.isoformat() if self.last_ts else None,
            "recent_burn_rate_usd_per_min": round(self.recent_burn_rate_usd_per_min, 6),
            "recent_window_minutes": self.recent_window_minutes,
            "log_files": list(self.log_files),
        }


# ---------------------------------------------------------------------------
# 2. JSONL reader  (tolerant)
# ---------------------------------------------------------------------------

def parse_jsonl_file(path: str) -> Iterator[dict[str, Any]]:
    """Yield each JSON object in a ``.jsonl`` file.

    Tolerant: a malformed line (truncated rotation tail, partially
    flushed buffer at process kill) is logged at WARNING and skipped
    instead of aborting the iterator. Non-dict top-level objects are
    skipped too — we only care about records.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "[metrics] %s:%d malformed JSON, skipping: %s",
                        path, lineno, exc,
                    )
                    continue
                if isinstance(obj, dict):
                    yield obj
    except FileNotFoundError:
        # Race: file was rotated/purged between glob and open. Caller
        # already logged the path list, so we silently skip.
        return


def _parse_ts(value: Any) -> Optional[datetime]:
    """Parse an ISO 8601 ``ts`` field into a UTC datetime, or None."""
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# 3. Log file discovery
# ---------------------------------------------------------------------------

_ROTATION_SUFFIX_RE = re.compile(r"\.jsonl\.(\d+)$")


def _sorted_session_log_files(session_id: str, log_dir: str) -> list[str]:
    """Return ``<log_dir>/<id>.jsonl*`` sorted chronologically.

    RotatingFileHandler rolls the live file ``foo.jsonl`` into
    ``foo.jsonl.1`` on rotation, shifting the previous .1 → .2, etc.
    Older content has a higher suffix; the live file (no suffix) has the
    newest content. We return [.N, .N-1, ..., .1, .jsonl] so the caller
    can append-iterate in time order.
    """
    expanded = os.path.expanduser(log_dir)
    primary = os.path.join(expanded, f"{session_id}.jsonl")
    rotated = sorted(
        glob.glob(os.path.join(expanded, f"{session_id}.jsonl.*")),
        key=lambda p: int(_ROTATION_SUFFIX_RE.search(p).group(1)) if _ROTATION_SUFFIX_RE.search(p) else 0,
        reverse=True,
    )
    files: list[str] = []
    files.extend(rotated)
    if os.path.exists(primary):
        files.append(primary)
    return files


_SESSION_FILENAME_RE = re.compile(r"\.jsonl(\.\d+)?$")


def list_sessions(log_dir: str) -> list[str]:
    """Return distinct session IDs discovered from filenames in log_dir.

    Matches both ``<id>.jsonl`` (live) and ``<id>.jsonl.N`` (rotated
    backup). Returns a sorted list (deterministic for snapshot tests /
    table output).
    """
    expanded = os.path.expanduser(log_dir)
    if not os.path.isdir(expanded):
        return []
    seen: set[str] = set()
    for entry in os.listdir(expanded):
        m = _SESSION_FILENAME_RE.search(entry)
        if m:
            seen.add(entry[: m.start()])
    return sorted(seen)


# ---------------------------------------------------------------------------
# 4. Aggregation
# ---------------------------------------------------------------------------

def aggregate_session(
    session_id: str,
    log_dir: str,
    *,
    window_minutes: int = _DEFAULT_WINDOW_MINUTES,
    now: Optional[datetime] = None,
) -> SessionMetrics:
    """Aggregate all on-disk log records for a session into a SessionMetrics.

    ``now`` is injectable so tests can pin the clock for the burn-rate
    window. Production callers should leave it as None (defaults to
    ``datetime.now(timezone.utc)``).
    """
    files = _sorted_session_log_files(session_id, log_dir)
    metrics = SessionMetrics(
        session_id=session_id,
        recent_window_minutes=window_minutes,
        log_files=files,
    )

    # Track per-record contributions to the recent-window burn-rate.
    # We collect (ts, cost) tuples then filter at the end so the
    # window math is one pass.
    window_records: list[tuple[datetime, float]] = []

    for path in files:
        for rec in parse_jsonl_file(path):
            event = rec.get("event")
            if event in _COST_EVENTS:
                cost = _coerce_float(rec.get("cost_usd"))
                metrics.total_cost_usd += cost
                metrics.llm_call_count += 1
                metrics.tokens_in += _coerce_int(rec.get("tokens_in"))
                metrics.tokens_out += _coerce_int(rec.get("tokens_out"))
                metrics.cached_tokens += _coerce_int(rec.get("cached_tokens"))
                ts = _parse_ts(rec.get("ts"))
                if ts is not None:
                    _update_ts_range(metrics, ts)
                    window_records.append((ts, cost))
            elif event in _TRACKED_FAILURE_EVENTS:
                metrics.error_counts[event] = metrics.error_counts.get(event, 0) + 1
                ts = _parse_ts(rec.get("ts"))
                if ts is not None:
                    _update_ts_range(metrics, ts)

    metrics.recent_burn_rate_usd_per_min = _compute_burn_rate(
        window_records,
        window_minutes=window_minutes,
        now=now or datetime.now(timezone.utc),
    )
    return metrics


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _update_ts_range(metrics: SessionMetrics, ts: datetime) -> None:
    if metrics.first_ts is None or ts < metrics.first_ts:
        metrics.first_ts = ts
    if metrics.last_ts is None or ts > metrics.last_ts:
        metrics.last_ts = ts


def _compute_burn_rate(
    records: list[tuple[datetime, float]],
    *,
    window_minutes: int,
    now: datetime,
) -> float:
    """Compute $/min over the trailing ``window_minutes`` of records.

    Burn rate = sum(cost in window) / max(elapsed_minutes_in_window,
    _BURN_RATE_FLOOR_MINUTES). The floor avoids div-by-zero and absurd
    extrapolations when a window contains one record or a tight cluster.
    """
    if not records:
        return 0.0
    window_start = now.timestamp() - (window_minutes * 60)
    in_window = [(ts, cost) for ts, cost in records if ts.timestamp() >= window_start]
    if not in_window:
        return 0.0
    total_cost = sum(cost for _, cost in in_window)
    earliest = min(ts for ts, _ in in_window)
    elapsed_minutes = max(
        (now.timestamp() - earliest.timestamp()) / 60.0,
        _BURN_RATE_FLOOR_MINUTES,
    )
    return total_cost / elapsed_minutes


# ---------------------------------------------------------------------------
# 5. Projection
# ---------------------------------------------------------------------------

def project_exhaustion(metrics: SessionMetrics, hard_cap_usd: float) -> Optional[float]:
    """Estimate minutes until the hard-cap is hit at the recent burn rate.

    Returns None when the burn rate is zero (no recent activity → no
    projection possible) or when the session has already exceeded the
    cap. Otherwise returns ``(hard_cap - total_cost) / burn_rate``.
    """
    if metrics.recent_burn_rate_usd_per_min <= 0:
        return None
    remaining = hard_cap_usd - metrics.total_cost_usd
    if remaining <= 0:
        return 0.0
    return remaining / metrics.recent_burn_rate_usd_per_min


# ---------------------------------------------------------------------------
# 6. Formatters
# ---------------------------------------------------------------------------

def format_human(metrics: SessionMetrics, hard_cap_usd: float) -> str:
    """Render a single-session report in the `harness status` style."""
    proj = project_exhaustion(metrics, hard_cap_usd)
    if proj is None:
        proj_label = "n/a (no recent activity)"
    elif proj == 0.0:
        proj_label = "already exhausted"
    else:
        proj_label = f"~{proj:.1f} min at current rate"

    remaining = hard_cap_usd - metrics.total_cost_usd
    span = _format_ts_span(metrics.first_ts, metrics.last_ts)

    errs = ", ".join(
        f"{k}={v}" for k, v in sorted(metrics.error_counts.items())
    ) or "none"

    lines = [
        "=" * 60,
        f"Session: {metrics.session_id}",
        f"  Log files:           {len(metrics.log_files)}",
        f"  Total LLM calls:     {metrics.llm_call_count}",
        f"  Total cost:          ${metrics.total_cost_usd:.4f}",
        f"  Tokens (in/out):     {metrics.tokens_in:,} / {metrics.tokens_out:,}",
        f"  Cached tokens:       {metrics.cached_tokens:,}",
        f"  Errors:              {errs}",
        f"  Wall-clock span:     {span}",
        f"  Burn rate ({metrics.recent_window_minutes}m): ${metrics.recent_burn_rate_usd_per_min:.4f}/min",
        f"  Budget (hard cap):   ${hard_cap_usd:.2f} — ${remaining:.4f} remaining",
        f"  Projected exhaust:   {proj_label}",
        "=" * 60,
    ]
    return "\n".join(lines)


def format_table(metrics_list: list[SessionMetrics], hard_cap_usd: float) -> str:
    """Render a multi-session table with a TOTAL footer."""
    if not metrics_list:
        return "(no sessions found)"

    header = f"{'Session':<24} {'Cost':>10} {'Calls':>7} {'Burn $/min':>12} {'Last activity':<24}"
    sep = "-" * len(header)

    rows: list[str] = [header, sep]
    total_cost = 0.0
    total_calls = 0
    for m in metrics_list:
        last = m.last_ts.strftime("%Y-%m-%d %H:%M:%S UTC") if m.last_ts else "(none)"
        sid = m.session_id[:24]
        rows.append(
            f"{sid:<24} "
            f"${m.total_cost_usd:>8.4f} "
            f"{m.llm_call_count:>7d} "
            f"${m.recent_burn_rate_usd_per_min:>10.4f} "
            f"{last:<24}"
        )
        total_cost += m.total_cost_usd
        total_calls += m.llm_call_count
    rows.append(sep)
    rows.append(
        f"{'TOTAL':<24} "
        f"${total_cost:>8.4f} "
        f"{total_calls:>7d} "
        f"{'':>12} "
        f"({len(metrics_list)} sessions, cap ${hard_cap_usd:.2f})"
    )
    return "\n".join(rows)


def format_prometheus(metrics_list: list[SessionMetrics], hard_cap_usd: float) -> str:
    """Render a Prometheus text-exposition document.

    Metric names follow the convention ``harness_<noun>[_unit]``. Cost
    and burn rate are gauges (point-in-time reconstruction from disk,
    not monotonic counters). The hard cap is exposed once as a separate
    gauge so dashboards can compute remaining-budget client-side.
    """
    lines: list[str] = []

    def _emit(metric: str, mtype: str, helptext: str, samples: list[str]) -> None:
        lines.append(f"# HELP {metric} {helptext}")
        lines.append(f"# TYPE {metric} {mtype}")
        lines.extend(samples)

    cost_samples: list[str] = []
    calls_samples: list[str] = []
    tokens_in_samples: list[str] = []
    tokens_out_samples: list[str] = []
    burn_samples: list[str] = []
    proj_samples: list[str] = []

    for m in metrics_list:
        sid = _prometheus_label_value(m.session_id)
        cost_samples.append(f'harness_session_cost_usd{{session_id="{sid}"}} {m.total_cost_usd:.6f}')
        calls_samples.append(f'harness_session_llm_calls{{session_id="{sid}"}} {m.llm_call_count}')
        tokens_in_samples.append(
            f'harness_session_tokens{{session_id="{sid}",direction="in"}} {m.tokens_in}'
        )
        tokens_out_samples.append(
            f'harness_session_tokens{{session_id="{sid}",direction="out"}} {m.tokens_out}'
        )
        burn_samples.append(
            f'harness_burn_rate_usd_per_min{{session_id="{sid}"}} {m.recent_burn_rate_usd_per_min:.6f}'
        )
        proj = project_exhaustion(m, hard_cap_usd)
        if proj is not None:
            proj_samples.append(
                f'harness_projected_exhaustion_minutes{{session_id="{sid}"}} {proj:.4f}'
            )

    _emit(
        "harness_session_cost_usd",
        "gauge",
        "Total LLM cost per session in USD (reconstructed from logs).",
        cost_samples,
    )
    _emit(
        "harness_session_llm_calls",
        "gauge",
        "Number of LLM calls per session (reconstructed from logs).",
        calls_samples,
    )
    _emit(
        "harness_session_tokens",
        "gauge",
        "Tokens consumed per session by direction (in/out).",
        tokens_in_samples + tokens_out_samples,
    )
    _emit(
        "harness_burn_rate_usd_per_min",
        "gauge",
        "Recent USD/minute burn rate per session over the configured window.",
        burn_samples,
    )
    _emit(
        "harness_projected_exhaustion_minutes",
        "gauge",
        "Estimated minutes until the configured hard-cap is reached at the current burn rate.",
        proj_samples,
    )
    lines.append("# HELP harness_budget_hard_cap_usd Configured hard-cap budget in USD.")
    lines.append("# TYPE harness_budget_hard_cap_usd gauge")
    lines.append(f"harness_budget_hard_cap_usd {hard_cap_usd:.4f}")

    return "\n".join(lines) + "\n"


def _prometheus_label_value(value: str) -> str:
    """Escape a string for use inside a Prometheus label value.

    Prometheus exposition format requires \\ → \\\\, " → \\", LF → \\n.
    Anything else is fine as-is.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_ts_span(first: Optional[datetime], last: Optional[datetime]) -> str:
    if first is None or last is None:
        return "(none)"
    duration_s = max(0, int((last - first).total_seconds()))
    hours, rem = divmod(duration_s, 3600)
    minutes, _ = divmod(rem, 60)
    duration = f"{hours}h {minutes}m" if hours else f"{minutes}m"
    return (
        f"{first.strftime('%Y-%m-%d %H:%M:%S UTC')} → "
        f"{last.strftime('%Y-%m-%d %H:%M:%S UTC')} ({duration})"
    )


# ---------------------------------------------------------------------------
# 7. Atomic writer
# ---------------------------------------------------------------------------

def write_atomic(dest_path: str, content: str) -> None:
    """Write ``content`` to ``dest_path`` atomically.

    Strategy: write into ``<dest>.tmp`` in the same directory, fsync,
    then ``os.replace`` over the destination. A reader that opens the
    final path always sees either the previous version or the new one,
    never a half-written file (matters for node_exporter textfile
    collector and similar scrapers).
    """
    dest_dir = os.path.dirname(dest_path) or "."
    os.makedirs(dest_dir, exist_ok=True)
    tmp_path = dest_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(content)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            # fsync can fail on some filesystems (tmpfs in restricted
            # containers); the durability guarantee weakens but the
            # atomicity from rename still holds.
            pass
    os.replace(tmp_path, dest_path)
