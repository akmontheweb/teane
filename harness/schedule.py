"""``harness schedule`` — cron-driven background daemon (#13).

Why this exists
===============
``harness run`` is a workstation tool — operator runs it, watches it,
walks away when it's done. Many of the highest-value harness workloads
are recurring: "every night, ingest open Renovate PRs and regenerate
failing tests", "every Monday, run the security review on main".
Standing those up today means wrapping ``harness run`` in cron or
systemd timers and reinventing per-job state, retry, and notification
for each. The schedule daemon turns that into a config-driven
primitive.

Scope of v1
===========
- **Hand-rolled cron syntax subset.** ``every 15m`` / ``every 6h`` /
  ``every 3d`` / ``hourly :MM`` / ``daily HH:MM`` / ``weekly mon HH:MM``.
  Covers >90% of real scheduled-job use cases without depending on
  ``croniter``. Full POSIX cron is a follow-up if the demand
  materialises (clean upgrade path: drop in an opt-in backend the
  same way ``repo_index.backend`` flips ``tfidf`` → ``openai_embeddings``).
- **Generic shell hooks** for ``on_success`` and ``on_failure``. The
  operator wires Slack / Discord / PagerDuty / email via curl in one
  config line. No built-in notifiers in v1 — keeps the daemon small
  and the security review trivial.
- **Subprocess job execution.** Each job runs as ``harness run`` in
  its own subprocess so a crash never takes the daemon down. Stdout
  / stderr are streamed to per-job log files under
  ``~/.harness/schedule_logs/<job>/<iso8601>.log``.
- **SQLite-backed history** so ``harness schedule list`` /
  ``harness schedule history`` survive restarts.

Not in scope
============
- Filesystem-watch mode (``harness watch``) — separate slice if anyone
  asks for it.
- Built-in Slack/Discord notifiers — follow-up.
- Distributed scheduling across hosts — out of scope; operators run
  the daemon under their orchestrator of choice.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import re
import shlex
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_DEFAULT_HISTORY_DB = "~/.harness/schedule.db"
_DEFAULT_LOG_DIR = "~/.harness/schedule_logs"
_DEFAULT_TICK_SECONDS = 60


# ---------------------------------------------------------------------------
# 1. Schedule expression — hand-rolled cron subset
# ---------------------------------------------------------------------------

SCHEDULE_KIND_INTERVAL = "interval"
SCHEDULE_KIND_HOURLY = "hourly"
SCHEDULE_KIND_DAILY = "daily"
SCHEDULE_KIND_WEEKLY = "weekly"

_INTERVAL_RE = re.compile(r"^every\s+(\d+)\s*([mhd])\s*$", re.IGNORECASE)
_HOURLY_RE = re.compile(r"^hourly\s*:\s*(\d{1,2})\s*$", re.IGNORECASE)
_DAILY_RE = re.compile(r"^daily\s+(\d{1,2})\s*:\s*(\d{2})\s*$", re.IGNORECASE)
_WEEKLY_RE = re.compile(
    r"^weekly\s+(mon|tue|wed|thu|fri|sat|sun)\s+(\d{1,2})\s*:\s*(\d{2})\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Schedule:
    """Parsed schedule expression. Use :func:`parse_schedule` to build
    one and :func:`next_run` to compute when it next fires."""

    raw: str
    kind: str
    minutes: int = 0    # for interval schedules: total minutes between runs
    hour: int = 0       # for hourly/daily/weekly: target HH (0-23)
    minute: int = 0     # for hourly/daily/weekly: target MM (0-59)
    weekday: int = -1   # for weekly: 0=Mon ... 6=Sun


def parse_schedule(raw: str) -> Schedule:
    """Parse a schedule string.

    Accepted forms (case-insensitive)::

        every 15m            # every 15 minutes
        every 6h             # every 6 hours
        every 3d             # every 3 days
        hourly :30           # at 30 minutes past every hour
        daily 02:30          # every day at 02:30 UTC
        weekly mon 03:00     # every Monday at 03:00 UTC

    Raises ``ValueError`` with a guidance-shaped error message when the
    input doesn't match any form. The error always names the supported
    forms so the operator never has to guess.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("schedule must be a non-empty string")
    expr = raw.strip()

    m = _INTERVAL_RE.match(expr)
    if m:
        amount, unit = int(m.group(1)), m.group(2).lower()
        if amount < 1:
            raise ValueError(f"schedule {raw!r}: interval must be >= 1")
        minutes = {"m": amount, "h": amount * 60, "d": amount * 60 * 24}[unit]
        return Schedule(raw=expr, kind=SCHEDULE_KIND_INTERVAL, minutes=minutes)

    m = _HOURLY_RE.match(expr)
    if m:
        minute = int(m.group(1))
        if not 0 <= minute <= 59:
            raise ValueError(f"schedule {raw!r}: minute must be 0-59")
        return Schedule(raw=expr, kind=SCHEDULE_KIND_HOURLY, minute=minute)

    m = _DAILY_RE.match(expr)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(f"schedule {raw!r}: HH:MM must be 00:00-23:59")
        return Schedule(raw=expr, kind=SCHEDULE_KIND_DAILY, hour=hour, minute=minute)

    m = _WEEKLY_RE.match(expr)
    if m:
        wd_name = m.group(1).lower()
        hour, minute = int(m.group(2)), int(m.group(3))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(f"schedule {raw!r}: HH:MM must be 00:00-23:59")
        weekday = _WEEKDAYS.index(wd_name)
        return Schedule(
            raw=expr, kind=SCHEDULE_KIND_WEEKLY,
            hour=hour, minute=minute, weekday=weekday,
        )

    raise ValueError(
        f"schedule {raw!r} is not recognised. Accepted forms: "
        "'every Nm' / 'every Nh' / 'every Nd' / 'hourly :MM' / "
        "'daily HH:MM' / 'weekly DAY HH:MM' (DAY ∈ mon,tue,wed,"
        "thu,fri,sat,sun). All times are UTC."
    )


def next_run(
    schedule: Schedule,
    *,
    after: datetime,
    last_started: Optional[datetime] = None,
) -> datetime:
    """Compute the next firing time strictly after ``after``.

    For interval schedules, ``last_started`` is the timestamp the job
    last fired at — the next run is ``last_started + interval`` (clamped
    to be after ``after`` so a long-stopped daemon catches up instead
    of firing thousands of missed runs in a row). For absolute
    schedules (hourly/daily/weekly), ``last_started`` is ignored — we
    just find the next clock instant that matches.

    All datetimes are expected to be tz-aware UTC; the function asserts
    this rather than silently converting from naive local time, because
    "the daemon fired at the wrong time because we forgot the timezone"
    is a confusing failure mode.
    """
    if after.tzinfo is None:
        raise ValueError("next_run requires tz-aware datetimes (use UTC)")

    if schedule.kind == SCHEDULE_KIND_INTERVAL:
        delta = timedelta(minutes=schedule.minutes)
        if last_started is None:
            return after + delta
        candidate = last_started + delta
        while candidate <= after:
            candidate = candidate + delta
        return candidate

    if schedule.kind == SCHEDULE_KIND_HOURLY:
        candidate = after.replace(
            minute=schedule.minute, second=0, microsecond=0,
        )
        if candidate <= after:
            candidate = candidate + timedelta(hours=1)
        return candidate

    if schedule.kind == SCHEDULE_KIND_DAILY:
        candidate = after.replace(
            hour=schedule.hour, minute=schedule.minute,
            second=0, microsecond=0,
        )
        if candidate <= after:
            candidate = candidate + timedelta(days=1)
        return candidate

    if schedule.kind == SCHEDULE_KIND_WEEKLY:
        candidate = after.replace(
            hour=schedule.hour, minute=schedule.minute,
            second=0, microsecond=0,
        )
        days_ahead = (schedule.weekday - candidate.weekday()) % 7
        candidate = candidate + timedelta(days=days_ahead)
        if candidate <= after:
            candidate = candidate + timedelta(days=7)
        return candidate

    raise ValueError(f"unknown schedule kind: {schedule.kind!r}")


# ---------------------------------------------------------------------------
# 2. Job + config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Job:
    """One scheduled job. The daemon spawns ``harness run`` in a
    subprocess per fire and shells the hooks afterwards.

    ``on_success`` / ``on_failure`` are arbitrary shell commands. Two
    environment variables are exported to the hook:

        HARNESS_JOB_NAME          — the job's ``name``.
        HARNESS_JOB_EXIT_CODE     — the exit code of ``harness run``.
        HARNESS_JOB_DURATION_SEC  — wall-clock seconds.
        HARNESS_JOB_LOG_PATH      — path to the run's captured log file.
    """

    name: str
    schedule: Schedule
    workspace: str
    prompt: str = ""
    on_success: str = ""
    on_failure: str = ""
    enabled: bool = True
    harness_args: list[str] = field(default_factory=list)


@dataclass
class ScheduleConfig:
    enabled: bool = False
    jobs: list[Job] = field(default_factory=list)
    history_db: str = _DEFAULT_HISTORY_DB
    log_dir: str = _DEFAULT_LOG_DIR
    tick_seconds: int = _DEFAULT_TICK_SECONDS
    harness_binary: str = "harness"  # operators using a venv set "/path/to/venv/bin/harness"
    # Path to the dashboard's web.db. When set (default
    # ``~/.harness/web.db``), the daemon polls ``web_oneshot_jobs`` for
    # one-shot runs the dashboard enqueued via ``POST /run/schedule`` and
    # fires + marks them consumed alongside the config-driven jobs.
    # Empty string disables this integration (useful for headless CI
    # daemons that never run the dashboard).
    web_db_path: str = "~/.harness/web.db"

    @classmethod
    def from_config(cls, config: Optional[dict[str, Any]]) -> "ScheduleConfig":
        section = ((config or {}).get("schedule") or {})
        jobs: list[Job] = []
        for raw in (section.get("jobs") or []):
            if not isinstance(raw, dict):
                logger.warning("[schedule] dropping malformed job entry: %r", raw)
                continue
            try:
                job = cls._build_job(raw)
            except ValueError as exc:
                logger.warning(
                    "[schedule] dropping job %r: %s",
                    raw.get("name") or "(unnamed)", exc,
                )
                continue
            jobs.append(job)
        tick_seconds = max(1, min(3600, int(section.get("tick_seconds", _DEFAULT_TICK_SECONDS))))
        return cls(
            enabled=bool(section.get("enabled", False)),
            jobs=jobs,
            history_db=str(section.get("history_db", _DEFAULT_HISTORY_DB)),
            log_dir=str(section.get("log_dir", _DEFAULT_LOG_DIR)),
            tick_seconds=tick_seconds,
            harness_binary=str(section.get("harness_binary", "harness")),
            web_db_path=str(section.get("web_db_path", "~/.harness/web.db")),
        )

    @staticmethod
    def _build_job(raw: dict[str, Any]) -> Job:
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ValueError("job is missing 'name'")
        sched_raw = str(raw.get("schedule") or "")
        if not sched_raw:
            raise ValueError("job is missing 'schedule'")
        workspace = str(raw.get("workspace") or "").strip()
        if not workspace:
            raise ValueError("job is missing 'workspace'")
        sched = parse_schedule(sched_raw)
        return Job(
            name=name,
            schedule=sched,
            workspace=workspace,
            prompt=str(raw.get("prompt") or ""),
            on_success=str(raw.get("on_success") or ""),
            on_failure=str(raw.get("on_failure") or ""),
            enabled=bool(raw.get("enabled", True)),
            harness_args=[str(a) for a in (raw.get("harness_args") or []) if isinstance(a, str)],
        )


# ---------------------------------------------------------------------------
# 3. History store (SQLite next to checkpoints)
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schedule_runs (
    job_name        TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    exit_code       INTEGER,
    duration_sec    REAL,
    log_path        TEXT,
    pid             INTEGER,
    PRIMARY KEY (job_name, started_at)
);
CREATE INDEX IF NOT EXISTS idx_schedule_job_time
    ON schedule_runs (job_name, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_schedule_in_flight
    ON schedule_runs (ended_at, pid);
"""


def _ensure_schedule_pid_column(conn: sqlite3.Connection) -> None:
    """Migrate older schedule_runs tables to include the `pid` column.

    Originally the schema had no pid column; in-flight tracking was
    purely in-memory (`ScheduleDaemon._in_flight`). On daemon crash the
    set was lost and the job could re-fire on restart even while the
    original was still running (audit §1.5). New rows now carry the
    subprocess pid; existing rows get NULL via ADD COLUMN.
    """
    try:
        cur = conn.execute("PRAGMA table_info(schedule_runs)")
        cols = {row[1] for row in cur.fetchall()}
        if "pid" not in cols:
            conn.execute("ALTER TABLE schedule_runs ADD COLUMN pid INTEGER")
            conn.commit()
    except sqlite3.DatabaseError:  # pragma: no cover
        pass


def _open_history(cfg: ScheduleConfig) -> sqlite3.Connection:
    """Open schedule history DB with WAL + busy_timeout, leak-safe.

    Audit §1.11 (WAL needed for multi-writer coexistence between the
    daemon and dashboard process) and §2.14 (close the connection if
    schema migration raises).
    """
    path = os.path.expanduser(cfg.history_db)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
        except sqlite3.DatabaseError:  # pragma: no cover — best-effort
            pass
        conn.executescript(_SCHEMA_SQL)
        _ensure_schedule_pid_column(conn)
    except Exception:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
        raise
    return conn


def record_run_started(
    cfg: ScheduleConfig, *, job_name: str, started_at: datetime, log_path: str,
    pid: Optional[int] = None,
) -> None:
    """Insert (or replace) a row marking this run started.

    ``pid`` is the running subprocess pid so an orphan-detection pass on
    daemon restart can flip dead in-flight rows to ``ended_at`` (audit
    §1.5). Older callers pass nothing → pid stays NULL → orphan-detection
    skips them (backwards-compatible).

    Audit §1.16: if a row already exists for the (job_name, started_at)
    pair (two runs colliding in the same second — leap-second tick, a
    manual ``harness schedule once`` racing a daemon tick), bump the
    timestamp by 1 µs until the insert succeeds rather than replacing
    the prior row's log_path / pid.
    """
    conn = _open_history(cfg)
    try:
        with conn:
            iso = started_at.isoformat()
            attempt = 0
            while attempt < 1000:
                try:
                    conn.execute(
                        "INSERT INTO schedule_runs "
                        "(job_name, started_at, log_path, pid) VALUES (?, ?, ?, ?)",
                        (job_name, iso, log_path,
                         int(pid) if pid is not None else None),
                    )
                    return
                except sqlite3.IntegrityError:
                    # PK collision (same job_name+started_at). The most
                    # common path is record_run_started being called twice
                    # for the SAME run (once at spawn, once with the pid)
                    # — that's an intentional REPLACE so keep that
                    # behaviour for true duplicates by detecting pid
                    # presence on the existing row.
                    existing = conn.execute(
                        "SELECT pid FROM schedule_runs "
                        "WHERE job_name = ? AND started_at = ?",
                        (job_name, iso),
                    ).fetchone()
                    if existing is None or (existing[0] is None and pid is not None):
                        # Genuine re-stamp of the same run (pid landed
                        # after the original insert) — replace, not bump.
                        conn.execute(
                            "INSERT OR REPLACE INTO schedule_runs "
                            "(job_name, started_at, log_path, pid) VALUES (?, ?, ?, ?)",
                            (job_name, iso, log_path,
                             int(pid) if pid is not None else None),
                        )
                        return
                    # Different run colliding in the same second: bump
                    # the iso timestamp by 1 µs and try again.
                    started_at = started_at + timedelta(microseconds=1)
                    iso = started_at.isoformat()
                    attempt += 1
    finally:
        conn.close()


def record_run_finished(
    cfg: ScheduleConfig, *,
    job_name: str, started_at: datetime,
    exit_code: int, duration_sec: float,
) -> None:
    conn = _open_history(cfg)
    try:
        with conn:
            conn.execute(
                "UPDATE schedule_runs SET ended_at = ?, exit_code = ?, "
                "duration_sec = ?, pid = NULL "
                "WHERE job_name = ? AND started_at = ?",
                (
                    _utcnow().isoformat(), exit_code, duration_sec,
                    job_name, started_at.isoformat(),
                ),
            )
    finally:
        conn.close()


def find_inflight_runs(cfg: ScheduleConfig) -> list[dict[str, Any]]:
    """Return rows that started but never finished (``ended_at IS NULL``).

    Returned dicts include ``pid`` so the daemon can probe each on
    boot via ``os.kill(pid, 0)`` and decide whether the process is
    still alive (re-attach / skip-this-tick) or truly orphaned (mark
    as exit_code=-1 so the row falls out of the in-flight set).
    """
    conn = _open_history(cfg)
    try:
        rows = conn.execute(
            "SELECT job_name, started_at, log_path, pid FROM schedule_runs "
            "WHERE ended_at IS NULL"
        ).fetchall()
    finally:
        conn.close()
    return [
        {"job_name": r[0], "started_at": r[1], "log_path": r[2],
         "pid": (int(r[3]) if r[3] is not None else None)}
        for r in rows
    ]


def reap_orphan_run(
    cfg: ScheduleConfig, *, job_name: str, started_at: str,
) -> None:
    """Mark an in-flight row as terminated with exit_code=-1.

    Used by the daemon on boot when ``_pid_alive`` reports the row's
    pid is dead — the subprocess died before recording its exit, so
    we close the row defensively so it doesn't keep the slot ``busy``
    forever (audit §1.5).
    """
    conn = _open_history(cfg)
    try:
        with conn:
            conn.execute(
                "UPDATE schedule_runs SET ended_at = ?, exit_code = -1, "
                "pid = NULL WHERE job_name = ? AND started_at = ?",
                (_utcnow().isoformat(), job_name, started_at),
            )
    finally:
        conn.close()


def last_run_for_job(
    cfg: ScheduleConfig, job_name: str,
) -> Optional[dict[str, Any]]:
    conn = _open_history(cfg)
    try:
        row = conn.execute(
            "SELECT started_at, ended_at, exit_code, duration_sec, log_path "
            "FROM schedule_runs WHERE job_name = ? "
            "ORDER BY started_at DESC LIMIT 1",
            (job_name,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "started_at": row[0],
        "ended_at": row[1],
        "exit_code": row[2],
        "duration_sec": row[3],
        "log_path": row[4],
    }


def history_for_job(
    cfg: ScheduleConfig, job_name: str, *, limit: int = 20,
) -> list[dict[str, Any]]:
    conn = _open_history(cfg)
    try:
        rows = conn.execute(
            "SELECT started_at, ended_at, exit_code, duration_sec, log_path "
            "FROM schedule_runs WHERE job_name = ? "
            "ORDER BY started_at DESC LIMIT ?",
            (job_name, max(1, min(1000, int(limit)))),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "started_at": r[0], "ended_at": r[1], "exit_code": r[2],
            "duration_sec": r[3], "log_path": r[4],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# 4. Job execution — subprocess + hooks
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _pid_alive_int(pid: int) -> bool:
    """True if a process with this pid currently exists on the system.

    Used by the daemon's boot-time orphan reconciliation to decide
    whether an in-flight row from a prior daemon corresponds to a
    still-running process (adopt) or a dead one (reap). Signal 0 is
    the POSIX existence probe; it doesn't actually deliver a signal.
    """
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by someone else — still alive.
        return True
    except OSError:
        return False


def _log_path_for(cfg: ScheduleConfig, job_name: str, started_at: datetime) -> str:
    base = os.path.expanduser(cfg.log_dir)
    job_dir = os.path.join(base, _safe_filename(job_name))
    os.makedirs(job_dir, exist_ok=True)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    return os.path.join(job_dir, f"{stamp}.log")


def _safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "job"


def build_run_command(cfg: ScheduleConfig, job: Job) -> list[str]:
    """Build the argv for ``harness run`` for this job. Pure function;
    no I/O. Used by ``harness schedule list`` for transparency too.
    """
    cmd = [cfg.harness_binary, "run", "-w", job.workspace]
    if job.prompt:
        cmd += ["-p", job.prompt]
    cmd += list(job.harness_args)
    return cmd


async def execute_job_once(
    cfg: ScheduleConfig, job: Job,
    *,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Spawn ``harness run`` for this job, await completion, fire the
    success/failure hook. Returns a dict summary so the caller can
    print or log it (``harness schedule once`` does both).
    """
    started_at = now or _utcnow()
    log_path = _log_path_for(cfg, job.name, started_at)
    # Record start without pid first; the pid only exists after spawn.
    record_run_started(cfg, job_name=job.name, started_at=started_at, log_path=log_path)

    argv = build_run_command(cfg, job)
    logger.info("[schedule:%s] launching: %s", job.name, " ".join(shlex.quote(a) for a in argv))
    monotonic_start = time.monotonic()
    exit_code = -1
    try:
        with open(log_path, "wb") as log_fh:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=log_fh, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            # Update the row with the live pid so orphan-detection on
            # daemon restart can decide whether to reap or re-attach
            # (audit §1.5). Best-effort: if the update fails the row
            # just stays in the "no pid" state and is skipped.
            try:
                record_run_started(
                    cfg, job_name=job.name, started_at=started_at,
                    log_path=log_path, pid=proc.pid,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "[schedule:%s] pid stamp failed (%s); orphan-detection "
                    "will skip this row.", job.name, exc,
                )
            exit_code = await proc.wait()
    except Exception as exc:  # noqa: BLE001 — subprocess crashes shouldn't kill the daemon
        logger.exception("[schedule:%s] subprocess error: %s", job.name, exc)
        with open(log_path, "ab") as log_fh:
            log_fh.write(f"\n[schedule] subprocess error: {exc}\n".encode("utf-8"))
        exit_code = -1
    duration = time.monotonic() - monotonic_start

    record_run_finished(
        cfg, job_name=job.name, started_at=started_at,
        exit_code=exit_code, duration_sec=duration,
    )

    hook = job.on_success if exit_code == 0 else job.on_failure
    if hook.strip():
        await _run_hook(
            hook,
            job_name=job.name, exit_code=exit_code,
            duration_sec=duration, log_path=log_path,
        )

    return {
        "job_name": job.name,
        "started_at": started_at.isoformat(),
        "exit_code": exit_code,
        "duration_sec": duration,
        "log_path": log_path,
    }


async def _run_hook(
    hook: str, *,
    job_name: str, exit_code: int,
    duration_sec: float, log_path: str,
) -> None:
    """Invoke the operator-supplied shell hook. Hooks run via ``/bin/sh
    -c <hook>`` so curl + redirection + pipes Just Work without
    operators having to argv-tokenise. The harness's
    :func:`harness.trust.safe_subprocess_env` is intentionally NOT
    applied — the hook may want to read tokens the harness scrubs
    (SLACK_WEBHOOK, GH_TOKEN, etc.) directly from the operator's env.
    Operators who want stricter scrubbing pass env through ``env -i``
    themselves.
    """
    env = dict(os.environ)
    env["HARNESS_JOB_NAME"] = job_name
    env["HARNESS_JOB_EXIT_CODE"] = str(exit_code)
    env["HARNESS_JOB_DURATION_SEC"] = f"{duration_sec:.2f}"
    env["HARNESS_JOB_LOG_PATH"] = log_path
    proc = None
    pgid: Optional[int] = None
    try:
        proc = await asyncio.create_subprocess_shell(
            hook, env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        if hasattr(os, "getpgid"):
            try:
                pgid = os.getpgid(proc.pid)
            except (ProcessLookupError, OSError):
                pgid = proc.pid
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode != 0:
            logger.warning(
                "[schedule:%s] hook exit=%d: %s",
                job_name, proc.returncode,
                (stdout or b"").decode("utf-8", errors="replace")[:500],
            )
    except asyncio.TimeoutError:
        logger.warning("[schedule:%s] hook timed out after 30s; killing process group.", job_name)
        if proc is not None:
            # Without this kill the /bin/sh -c tree (curl, wget, etc.)
            # keeps running forever — one leak per job fire, every tick.
            # Audit §2.5.
            try:
                from harness.sandbox import _kill_process_group_async
                await _kill_process_group_async(pgid, proc)
            except Exception:  # noqa: BLE001
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("[schedule:%s] hook error: %s", job_name, exc)


# ---------------------------------------------------------------------------
# 5. Daemon
# ---------------------------------------------------------------------------

class ScheduleDaemon:
    """The cron loop. Wakes every ``cfg.tick_seconds``, checks each
    enabled job, fires it if its ``next_run`` has elapsed. A job that
    is still running from a prior tick will NOT be fired again — the
    daemon tracks in-flight jobs in memory.
    """

    def __init__(self, cfg: ScheduleConfig):
        self.cfg = cfg
        self._in_flight: set[str] = set()
        self._next_due: dict[str, datetime] = {}

    def initialise_due_times(self) -> None:
        """Seed ``_next_due`` from history (or the current time if the
        job has never run). Called once at daemon start.

        Also reconciles stale in-flight rows in the history DB: a prior
        daemon may have crashed/SIGKILL'd mid-run, leaving a row with
        ``ended_at IS NULL`` but a dead pid. Audit §1.5.
        """
        self._reconcile_inflight_history()
        now = _utcnow()
        for job in self.cfg.jobs:
            if not job.enabled:
                continue
            last = last_run_for_job(self.cfg, job.name)
            last_started = None
            if last and last.get("started_at"):
                try:
                    last_started = datetime.fromisoformat(last["started_at"])
                    if last_started.tzinfo is None:
                        last_started = last_started.replace(tzinfo=timezone.utc)
                except ValueError:
                    last_started = None
            self._next_due[job.name] = next_run(
                job.schedule, after=now, last_started=last_started,
            )

    def _reconcile_inflight_history(self) -> None:
        """Inspect schedule_runs rows with ``ended_at IS NULL`` on daemon
        boot. Two cases:

          * pid is dead (or missing)   → mark as orphaned (exit_code=-1)
          * pid is alive               → assume the prior process is still
                                         working; add to ``_in_flight`` so
                                         this daemon won't fire a duplicate

        This closes the audit §1.5 gap where an in-memory ``_in_flight``
        set was lost across daemon restarts and a long-running job could
        be fired twice.
        """
        try:
            rows = find_inflight_runs(self.cfg)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[schedule] in-flight reconcile read failed (%s); proceeding "
                "without orphan check.", exc,
            )
            return
        for row in rows:
            pid = row.get("pid")
            job_name = str(row.get("job_name", ""))
            started_at = str(row.get("started_at", ""))
            if pid is None or not _pid_alive_int(pid):
                logger.info(
                    "[schedule] reaping orphan run job=%s started_at=%s "
                    "(pid=%s no longer alive).", job_name, started_at, pid,
                )
                try:
                    reap_orphan_run(
                        self.cfg, job_name=job_name, started_at=started_at,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[schedule] reap_orphan_run failed for %s: %s",
                        job_name, exc,
                    )
                continue
            logger.info(
                "[schedule] adopting still-running prior job: %s (pid=%s)",
                job_name, pid,
            )
            self._in_flight.add(job_name)

    def jobs_due(self, *, now: Optional[datetime] = None) -> list[Job]:
        """Return the subset of enabled jobs whose ``next_due`` time
        is at or before ``now`` and which are not currently in flight."""
        now = now or _utcnow()
        out: list[Job] = []
        for job in self.cfg.jobs:
            if not job.enabled:
                continue
            if job.name in self._in_flight:
                continue
            due = self._next_due.get(job.name)
            if due is None or due <= now:
                out.append(job)
        return out

    async def fire_job(self, job: Job) -> dict[str, Any]:
        """Run a single job and update its next_due. Never raises;
        records the outcome through ``execute_job_once``."""
        if job.name in self._in_flight:
            return {"skipped": True, "reason": "job already in flight"}
        self._in_flight.add(job.name)
        try:
            now = _utcnow()
            result = await execute_job_once(self.cfg, job, now=now)
            self._next_due[job.name] = next_run(
                job.schedule, after=_utcnow(), last_started=now,
            )
            return result
        finally:
            self._in_flight.discard(job.name)

    async def tick_once(self) -> list[dict[str, Any]]:
        """Run one tick of the daemon loop: fire all due config-driven
        jobs *and* any web one-shot jobs the dashboard enqueued whose
        ``fire_at_utc`` has elapsed. Returns the union of summaries.
        Public so unit tests can drive the daemon without sleeping."""
        due = self.jobs_due()
        oneshots = self._due_oneshots() if self.cfg.web_db_path else []
        coroutines: list[Any] = []
        for j in due:
            coroutines.append(self.fire_job(j))
        for row in oneshots:
            coroutines.append(self._fire_oneshot(row))
        if not coroutines:
            return []
        results = await asyncio.gather(*coroutines, return_exceptions=False)
        return list(results)

    def _due_oneshots(self) -> list[dict[str, Any]]:
        """Read all pending web one-shot jobs whose fire time has
        elapsed. Best-effort: failures (missing DB, schema mismatch)
        log and return ``[]`` so the daemon's main loop continues."""
        try:
            from harness.web_state import list_pending_oneshot_jobs
        except Exception as exc:  # noqa: BLE001
            logger.debug("[schedule] web_state import failed: %s", exc)
            return []
        try:
            return list_pending_oneshot_jobs(db_path=self.cfg.web_db_path)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[schedule] web_oneshot poll failed for %s: %s",
                self.cfg.web_db_path, exc,
            )
            return []

    async def _fire_oneshot(self, row: dict[str, Any]) -> dict[str, Any]:
        """Build a transient :class:`Job` from a web one-shot row and
        run it through :func:`execute_job_once`. Mark consumed
        regardless of exit code so a failed run doesn't re-fire next
        tick (operators re-enqueue from the UI if they want a retry).

        Claims the row atomically BEFORE firing so a second daemon /
        manual ``harness schedule once`` invocation can't race ahead
        and run the same job twice. Audit §1.1.
        """
        # Atomic claim: any concurrent firer trying to grab the same
        # row gets `False` here and skips, while we proceed exclusively.
        try:
            from harness.web_state import claim_oneshot_job
            claimed = claim_oneshot_job(
                db_path=self.cfg.web_db_path, job_id=int(row["id"]),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[schedule] failed to atomic-claim oneshot %s (%s); "
                "skipping to avoid double-fire.", row.get("id"), exc,
            )
            return {"oneshot_id": row.get("id"), "skipped": "claim_failed"}
        if not claimed:
            logger.info(
                "[schedule] oneshot %s already consumed by another firer; "
                "skipping.", row.get("id"),
            )
            return {"oneshot_id": row.get("id"), "skipped": "already_consumed"}

        # The schedule field is required by Job but only used by
        # next_run; execute_job_once ignores it. Pick a no-op shape.
        try:
            sched = parse_schedule("daily 00:00")
        except ValueError:
            sched = Schedule(raw="daily 00:00", kind=SCHEDULE_KIND_DAILY)
        job = Job(
            name=f"web-oneshot-{row['id']}-{row['name']}",
            schedule=sched,
            workspace=str(row["workspace"]),
            prompt=str(row.get("prompt") or ""),
            harness_args=list(row.get("harness_args") or []),
        )
        result = await execute_job_once(self.cfg, job)
        result["oneshot_id"] = row["id"]
        return result

    async def run_forever(self) -> int:
        """Main loop. Returns when the asyncio task is cancelled
        (Ctrl-C / SIGTERM).

        On cancellation, signals SIGTERM to every still-running
        ``schedule_runs`` row's pid so the spawned ``harness run``
        subprocesses don't continue as orphans after the daemon exits
        (audit §2.1). Without this drain, repeated daemon stop/start
        cycles accumulated stray harness processes.
        """
        self.initialise_due_times()
        logger.info(
            "[schedule] starting with %d enabled job(s); tick=%ds",
            sum(1 for j in self.cfg.jobs if j.enabled),
            self.cfg.tick_seconds,
        )
        try:
            while True:
                try:
                    await self.tick_once()
                except Exception as exc:  # noqa: BLE001 — never crash the loop
                    logger.exception("[schedule] tick error: %s", exc)
                await asyncio.sleep(self.cfg.tick_seconds)
        except asyncio.CancelledError:
            logger.info("[schedule] cancellation received; shutting down.")
            self._drain_inflight_subprocesses()
            return 0

    def _drain_inflight_subprocesses(self) -> None:
        """Send SIGTERM (then SIGKILL after 5 s) to every in-flight pid.

        On a clean daemon shutdown we don't want spawned ``harness run``
        children to continue as orphans (audit §2.1). Best-effort: any
        process we can't signal (gone, EPERM) is just left alone.
        """
        try:
            rows = find_inflight_runs(self.cfg)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[schedule] shutdown drain read failed: %s", exc)
            return
        signalled: list[int] = []
        for row in rows:
            pid = row.get("pid")
            if pid is None or not _pid_alive_int(pid):
                continue
            try:
                if hasattr(os, "killpg"):
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGTERM)
                    except (ProcessLookupError, OSError):
                        os.kill(pid, signal.SIGTERM)
                else:
                    os.kill(pid, signal.SIGTERM)
                signalled.append(pid)
            except (ProcessLookupError, PermissionError, OSError):
                continue
        if not signalled:
            return
        logger.info(
            "[schedule] shutdown drain: SIGTERMed %d in-flight subprocess(es); "
            "waiting up to 5s before SIGKILL.", len(signalled),
        )
        deadline = time.monotonic() + 5.0
        while signalled and time.monotonic() < deadline:
            signalled = [pid for pid in signalled if _pid_alive_int(pid)]
            if signalled:
                time.sleep(0.2)
        # Anything still alive after the grace gets SIGKILL.
        for pid in signalled:
            try:
                if hasattr(os, "killpg"):
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        os.kill(pid, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
