"""Regression tests for the scheduled-job daemon (#13).

Covers:
    - The hand-rolled cron parser accepts every documented form and
      rejects everything else with a guidance-shaped error.
    - ``next_run`` produces correct firing times for each schedule
      kind, including the edge cases (Sunday wrap, day rollover,
      interval catch-up).
    - ``ScheduleConfig.from_config`` parses jobs cleanly, drops the
      malformed ones with a warning, and clamps ``tick_seconds``.
    - ``execute_job_once`` records start + end rows in the history
      DB, captures subprocess stdout to the per-job log file, and
      shells the success/failure hook with the documented env vars.
    - ``ScheduleDaemon.tick_once`` fires due jobs, skips
      already-in-flight jobs, and advances ``_next_due`` after a fire.
    - The daemon never raises out of ``tick_once`` when a single job
      crashes — the rest of the fleet stays alive.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

from harness.schedule import (
    SCHEDULE_KIND_DAILY,
    SCHEDULE_KIND_HOURLY,
    SCHEDULE_KIND_INTERVAL,
    SCHEDULE_KIND_WEEKLY,
    Job,
    ScheduleConfig,
    ScheduleDaemon,
    build_run_command,
    execute_job_once,
    history_for_job,
    last_run_for_job,
    next_run,
    parse_schedule,
)


# ---------------------------------------------------------------------------
# 1. Parser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,kind,expected_minutes", [
    ("every 15m", SCHEDULE_KIND_INTERVAL, 15),
    ("every 6h",  SCHEDULE_KIND_INTERVAL, 6 * 60),
    ("every 2d",  SCHEDULE_KIND_INTERVAL, 2 * 60 * 24),
    ("EVERY 1H",  SCHEDULE_KIND_INTERVAL, 60),       # case-insensitive
    ("every  3  d", SCHEDULE_KIND_INTERVAL, 3 * 60 * 24),  # extra whitespace
])
def test_parse_interval(raw, kind, expected_minutes):
    s = parse_schedule(raw)
    assert s.kind == kind
    assert s.minutes == expected_minutes


@pytest.mark.parametrize("raw,minute", [
    ("hourly :00", 0),
    ("hourly :30", 30),
    ("HOURLY :59", 59),
])
def test_parse_hourly(raw, minute):
    s = parse_schedule(raw)
    assert s.kind == SCHEDULE_KIND_HOURLY
    assert s.minute == minute


@pytest.mark.parametrize("raw,hour,minute", [
    ("daily 02:30", 2, 30),
    ("daily 00:00", 0, 0),
    ("daily 23:59", 23, 59),
])
def test_parse_daily(raw, hour, minute):
    s = parse_schedule(raw)
    assert s.kind == SCHEDULE_KIND_DAILY
    assert s.hour == hour and s.minute == minute


@pytest.mark.parametrize("raw,weekday,hour,minute", [
    ("weekly mon 03:00", 0, 3, 0),
    ("weekly tue 14:15", 1, 14, 15),
    ("weekly sun 23:59", 6, 23, 59),
])
def test_parse_weekly(raw, weekday, hour, minute):
    s = parse_schedule(raw)
    assert s.kind == SCHEDULE_KIND_WEEKLY
    assert s.weekday == weekday
    assert s.hour == hour and s.minute == minute


@pytest.mark.parametrize("raw", [
    "", "   ",
    "every 0m",            # zero interval
    "every 5x",            # bad unit
    "hourly :60",          # minute out of range
    "daily 24:00",         # hour out of range
    "daily 02:60",         # minute out of range
    "weekly funday 02:00", # bad weekday
    "30 2 * * mon",        # full cron — not supported in v1
])
def test_parse_rejects_malformed(raw):
    with pytest.raises(ValueError):
        parse_schedule(raw)


def test_parse_error_message_lists_supported_forms():
    with pytest.raises(ValueError) as exc:
        parse_schedule("noon every other day")
    msg = str(exc.value)
    assert "every Nm" in msg
    assert "daily HH:MM" in msg


# ---------------------------------------------------------------------------
# 2. next_run
# ---------------------------------------------------------------------------

UTC = timezone.utc


def _at(year=2026, month=6, day=15, hour=10, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def test_next_run_requires_tz_aware():
    with pytest.raises(ValueError):
        next_run(parse_schedule("daily 02:00"), after=datetime(2026, 1, 1))


def test_next_run_interval_first_time():
    s = parse_schedule("every 30m")
    now = _at(hour=10, minute=0)
    # No prior run → fire `interval` from now.
    assert next_run(s, after=now) == now + timedelta(minutes=30)


def test_next_run_interval_with_recent_last_started():
    s = parse_schedule("every 1h")
    last = _at(hour=10, minute=0)
    now = _at(hour=10, minute=15)
    # Next fire is last + 1h regardless of now.
    assert next_run(s, after=now, last_started=last) == last + timedelta(hours=1)


def test_next_run_interval_catches_up_after_long_downtime():
    s = parse_schedule("every 1h")
    last = _at(hour=10)
    # Daemon was down 3 hours; should fire from the NEXT-after-now slot.
    now = _at(hour=13, minute=15)
    nxt = next_run(s, after=now, last_started=last)
    assert nxt > now
    assert (nxt - last).total_seconds() % 3600 == 0  # still on the hourly cadence


def test_next_run_daily_fires_today_when_clock_not_yet_past():
    s = parse_schedule("daily 14:00")
    now = _at(hour=10, minute=0)
    assert next_run(s, after=now).hour == 14
    assert next_run(s, after=now).day == 15


def test_next_run_daily_rolls_to_tomorrow_when_clock_past():
    s = parse_schedule("daily 09:00")
    now = _at(hour=10, minute=0)
    nxt = next_run(s, after=now)
    assert nxt.day == 16
    assert nxt.hour == 9


def test_next_run_hourly_fires_this_hour_when_minute_in_future():
    s = parse_schedule("hourly :30")
    now = _at(hour=10, minute=10)
    nxt = next_run(s, after=now)
    assert nxt.hour == 10 and nxt.minute == 30


def test_next_run_hourly_rolls_to_next_hour_when_minute_past():
    s = parse_schedule("hourly :15")
    now = _at(hour=10, minute=20)
    nxt = next_run(s, after=now)
    assert nxt.hour == 11 and nxt.minute == 15


def test_next_run_weekly_wraps_correctly():
    # 2026-06-15 is a Monday.
    monday = _at(year=2026, month=6, day=15, hour=10, minute=0)
    s = parse_schedule("weekly mon 03:00")
    # Already past 03:00 on Monday → fires next Monday.
    nxt = next_run(s, after=monday)
    assert nxt.weekday() == 0
    # Fire time is next Mon 03:00 UTC (this Mon is 10:00 UTC → already past).
    assert nxt.day == 22 and nxt.hour == 3 and nxt.minute == 0

    # Wednesday 10:00 → next Mon 03:00 is 5 days + back-up = ~5d earlier.
    wed = _at(year=2026, month=6, day=17, hour=10, minute=0)
    s2 = parse_schedule("weekly sun 23:30")
    nxt2 = next_run(s2, after=wed)
    assert nxt2.weekday() == 6
    assert wed < nxt2 < wed + timedelta(days=8)


# ---------------------------------------------------------------------------
# 3. ScheduleConfig
# ---------------------------------------------------------------------------

def test_from_config_parses_valid_jobs():
    raw = {
        "schedule": {
            "enabled": True,
            "tick_seconds": 30,
            "jobs": [
                {
                    "name": "nightly",
                    "schedule": "daily 02:30",
                    "workspace": "/tmp/ws",
                    "prompt": "Regenerate tests",
                    "on_success": "echo ok",
                    "on_failure": "echo fail",
                },
                {
                    "name": "every-hour",
                    "schedule": "hourly :15",
                    "workspace": "/tmp/ws",
                    "enabled": False,
                },
            ],
        },
    }
    cfg = ScheduleConfig.from_config(raw)
    assert cfg.enabled is True
    assert cfg.tick_seconds == 30
    assert len(cfg.jobs) == 2
    nightly = cfg.jobs[0]
    assert nightly.name == "nightly"
    assert nightly.schedule.kind == SCHEDULE_KIND_DAILY
    assert nightly.on_success == "echo ok"
    assert cfg.jobs[1].enabled is False


def test_from_config_drops_malformed_jobs(caplog):
    raw = {
        "schedule": {
            "enabled": True,
            "jobs": [
                {"name": "ok", "schedule": "every 5m", "workspace": "/tmp/ws"},
                {"schedule": "every 5m", "workspace": "/tmp/ws"},  # missing name
                {"name": "x"},  # missing schedule + workspace
                {"name": "y", "schedule": "noon today", "workspace": "/tmp/ws"},  # bad sched
                "string-not-dict",
            ],
        },
    }
    with caplog.at_level("WARNING"):
        cfg = ScheduleConfig.from_config(raw)
    # Only the well-formed entry survives.
    assert [j.name for j in cfg.jobs] == ["ok"]
    # The malformed-list entry logs.
    assert any("malformed job" in r.message or "dropping job" in r.message for r in caplog.records)


def test_from_config_clamps_tick_seconds():
    cfg = ScheduleConfig.from_config({"schedule": {"tick_seconds": 0}})
    assert cfg.tick_seconds == 1
    cfg2 = ScheduleConfig.from_config({"schedule": {"tick_seconds": 99999}})
    assert cfg2.tick_seconds == 3600


def test_build_run_command_includes_prompt_and_extra_args():
    job = Job(
        name="x",
        schedule=parse_schedule("daily 02:00"),
        workspace="/tmp/ws",
        prompt="do the thing",
        harness_args=["--new_build=false", "-v"],
    )
    cfg = ScheduleConfig()
    argv = build_run_command(cfg, job)
    assert "run" in argv
    assert "-r" in argv and "/tmp/ws" in argv
    assert "-p" in argv and "do the thing" in argv
    assert "--new_build=false" in argv and "-v" in argv


# ---------------------------------------------------------------------------
# 4. execute_job_once — runs harness binary in subprocess, records history
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_job_once_records_history_and_runs_hook(tmp_path):
    # Use the real Python interpreter as a stand-in for the harness binary;
    # it exits 0 immediately, which is plenty to exercise the history +
    # hook plumbing.
    log_dir = tmp_path / "logs"
    history_db = tmp_path / "history.db"
    hook_marker = tmp_path / "hook_ran.txt"
    cfg = ScheduleConfig(
        enabled=True,
        history_db=str(history_db),
        log_dir=str(log_dir),
        tick_seconds=1,
        harness_binary=sys.executable,
    )
    job = Job(
        name="stub-success",
        schedule=parse_schedule("every 5m"),
        workspace=str(tmp_path),
        prompt="ignored",
        on_success=f"echo $HARNESS_JOB_NAME-$HARNESS_JOB_EXIT_CODE > {hook_marker}",
        # Replace the default "run -r W -p P" argv with our own so we
        # actually exit 0 via python -c. The Job's command construction
        # is `[binary, 'run', '-r', W, '-p', P, *harness_args]`, so we
        # have to override it with a custom build below.
    )
    # Monkeypatch build_run_command to return a sane argv for the test.
    import harness.schedule as sched_mod
    original = sched_mod.build_run_command

    def _stub_argv(_cfg, _job):
        return [sys.executable, "-c", "import sys; sys.exit(0)"]

    sched_mod.build_run_command = _stub_argv
    try:
        result = await execute_job_once(cfg, job)
    finally:
        sched_mod.build_run_command = original

    assert result["job_name"] == "stub-success"
    assert result["exit_code"] == 0
    assert os.path.isfile(result["log_path"])

    rows = history_for_job(cfg, "stub-success")
    assert len(rows) == 1
    assert rows[0]["exit_code"] == 0
    assert rows[0]["ended_at"] is not None

    # Give the hook subprocess a moment to flush its file.
    for _ in range(20):
        if hook_marker.is_file():
            break
        await asyncio.sleep(0.05)
    assert hook_marker.is_file(), "on_success hook did not run"
    assert "stub-success-0" in hook_marker.read_text()


@pytest.mark.asyncio
async def test_execute_job_once_records_failure_and_runs_failure_hook(tmp_path):
    cfg = ScheduleConfig(
        enabled=True,
        history_db=str(tmp_path / "h.db"),
        log_dir=str(tmp_path / "logs"),
        harness_binary=sys.executable,
    )
    hook_marker = tmp_path / "fail.txt"
    job = Job(
        name="stub-failure",
        schedule=parse_schedule("every 5m"),
        workspace=str(tmp_path),
        on_failure=f"echo failed-$HARNESS_JOB_EXIT_CODE > {hook_marker}",
    )
    import harness.schedule as sched_mod
    original = sched_mod.build_run_command

    def _stub_argv(_cfg, _job):
        return [sys.executable, "-c", "import sys; sys.exit(7)"]

    sched_mod.build_run_command = _stub_argv
    try:
        result = await execute_job_once(cfg, job)
    finally:
        sched_mod.build_run_command = original

    assert result["exit_code"] == 7
    for _ in range(20):
        if hook_marker.is_file():
            break
        await asyncio.sleep(0.05)
    assert hook_marker.is_file(), "on_failure hook did not run"
    assert "failed-7" in hook_marker.read_text()


# ---------------------------------------------------------------------------
# 5. ScheduleDaemon
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_daemon_tick_fires_due_jobs_only(tmp_path):
    cfg = ScheduleConfig(
        history_db=str(tmp_path / "h.db"),
        log_dir=str(tmp_path / "logs"),
        harness_binary=sys.executable,
    )
    sched = parse_schedule("every 5m")
    # Two jobs; one is "due" (next_due in the past), one isn't.
    job_due = Job(name="due", schedule=sched, workspace=str(tmp_path))
    job_not = Job(name="not-due", schedule=sched, workspace=str(tmp_path))
    cfg.jobs = [job_due, job_not]

    daemon = ScheduleDaemon(cfg)
    far_future = datetime.now(UTC) + timedelta(hours=24)
    long_ago = datetime.now(UTC) - timedelta(minutes=10)
    daemon._next_due = {"due": long_ago, "not-due": far_future}

    import harness.schedule as sched_mod
    original = sched_mod.build_run_command

    def _fast_argv(_cfg, _job):
        return [sys.executable, "-c", "import sys; sys.exit(0)"]

    sched_mod.build_run_command = _fast_argv
    try:
        results = await daemon.tick_once()
    finally:
        sched_mod.build_run_command = original
    # Only the due job ran.
    assert len(results) == 1
    assert results[0]["job_name"] == "due"
    # ``due`` had its next_due advanced; ``not-due`` is unchanged.
    assert daemon._next_due["due"] > long_ago
    assert daemon._next_due["not-due"] == far_future


@pytest.mark.asyncio
async def test_daemon_in_flight_job_does_not_double_fire(tmp_path):
    cfg = ScheduleConfig(
        history_db=str(tmp_path / "h.db"),
        log_dir=str(tmp_path / "logs"),
        harness_binary=sys.executable,
    )
    sched = parse_schedule("every 5m")
    job = Job(name="x", schedule=sched, workspace=str(tmp_path))
    cfg.jobs = [job]
    daemon = ScheduleDaemon(cfg)
    daemon._next_due = {"x": datetime.now(UTC) - timedelta(minutes=10)}
    # Pretend job is currently running.
    daemon._in_flight.add("x")
    results = await daemon.tick_once()
    assert results == []


def test_daemon_jobs_due_skips_disabled_and_in_flight():
    cfg = ScheduleConfig(
        jobs=[
            Job(name="on", schedule=parse_schedule("every 5m"),
                workspace="/tmp", enabled=True),
            Job(name="off", schedule=parse_schedule("every 5m"),
                workspace="/tmp", enabled=False),
        ],
    )
    daemon = ScheduleDaemon(cfg)
    daemon._next_due = {
        "on": datetime.now(UTC) - timedelta(minutes=1),
        "off": datetime.now(UTC) - timedelta(minutes=1),
    }
    due = [j.name for j in daemon.jobs_due()]
    assert due == ["on"]


@pytest.mark.asyncio
async def test_daemon_one_crashing_job_does_not_kill_loop(tmp_path):
    """A subprocess that explodes (or just exits non-zero) must not
    propagate an exception out of tick_once — the rest of the fleet
    keeps running."""
    cfg = ScheduleConfig(
        history_db=str(tmp_path / "h.db"),
        log_dir=str(tmp_path / "logs"),
        harness_binary="/this/binary/does/not/exist",
    )
    sched = parse_schedule("every 5m")
    cfg.jobs = [Job(name="impossible", schedule=sched, workspace=str(tmp_path))]
    daemon = ScheduleDaemon(cfg)
    daemon._next_due = {"impossible": datetime.now(UTC) - timedelta(minutes=1)}
    # The subprocess will fail to launch (binary missing); the function
    # must absorb that and return a result.
    results = await daemon.tick_once()
    assert len(results) == 1
    # exit_code will be -1 (subprocess error path).
    assert results[0]["exit_code"] == -1


# ---------------------------------------------------------------------------
# 6. History helpers
# ---------------------------------------------------------------------------

def test_last_run_for_job_returns_none_when_no_history(tmp_path):
    cfg = ScheduleConfig(history_db=str(tmp_path / "empty.db"))
    assert last_run_for_job(cfg, "anything") is None


@pytest.mark.asyncio
async def test_daemon_tick_fires_due_web_oneshot_jobs(tmp_path):
    """Tier-C integration smoke: the schedule daemon picks up rows
    enqueued in web.db's ``web_oneshot_jobs`` whose ``fire_at_utc``
    has elapsed, fires them via the same execute_job_once path, and
    marks them consumed afterwards."""
    from datetime import datetime as _dt, timedelta as _td
    from harness.web_state import (
        add_oneshot_job, list_all_oneshot_jobs,
    )

    web_db = str(tmp_path / "web.db")
    cfg = ScheduleConfig(
        history_db=str(tmp_path / "h.db"),
        log_dir=str(tmp_path / "logs"),
        harness_binary=sys.executable,
        web_db_path=web_db,
    )
    add_oneshot_job(
        db_path=web_db,
        name="urgent-fix",
        fire_at_utc=_dt.now(UTC) - _td(seconds=10),  # already due
        workspace=str(tmp_path),
        prompt="fix the regression",
    )
    add_oneshot_job(
        db_path=web_db,
        name="much-later",
        fire_at_utc=_dt.now(UTC) + _td(hours=2),  # not due
        workspace=str(tmp_path),
    )

    daemon = ScheduleDaemon(cfg)

    import harness.schedule as sched_mod
    original = sched_mod.build_run_command

    def _fast_argv(_cfg, _job):
        return [sys.executable, "-c", "import sys; sys.exit(0)"]

    sched_mod.build_run_command = _fast_argv
    try:
        results = await daemon.tick_once()
    finally:
        sched_mod.build_run_command = original

    assert len(results) == 1
    assert results[0]["job_name"].startswith("web-oneshot-")
    assert "oneshot_id" in results[0]

    all_jobs = list_all_oneshot_jobs(db_path=web_db)
    # Both rows still present; only the due one is consumed.
    consumed = [j for j in all_jobs if j["consumed_at"] is not None]
    pending = [j for j in all_jobs if j["consumed_at"] is None]
    assert len(consumed) == 1
    assert consumed[0]["name"] == "urgent-fix"
    assert len(pending) == 1
    assert pending[0]["name"] == "much-later"


@pytest.mark.asyncio
async def test_daemon_tick_skips_oneshots_when_web_db_path_empty(tmp_path):
    """Headless deployments that disable the web.db integration
    shouldn't try to read it."""
    cfg = ScheduleConfig(
        history_db=str(tmp_path / "h.db"),
        log_dir=str(tmp_path / "logs"),
        harness_binary=sys.executable,
        web_db_path="",  # disabled
    )
    daemon = ScheduleDaemon(cfg)
    results = await daemon.tick_once()
    assert results == []


def test_history_for_job_respects_limit(tmp_path):
    cfg = ScheduleConfig(history_db=str(tmp_path / "h.db"))
    # Manually insert 5 rows; verify limit=2 returns 2 newest.
    import harness.schedule as sched_mod
    for i in range(5):
        ts = datetime(2026, 6, 1, 10, i, tzinfo=UTC)
        sched_mod.record_run_started(
            cfg, job_name="t", started_at=ts, log_path=f"/tmp/{i}.log",
        )
        sched_mod.record_run_finished(
            cfg, job_name="t", started_at=ts,
            exit_code=0, duration_sec=1.0,
        )
    rows = history_for_job(cfg, "t", limit=2)
    assert len(rows) == 2
    # Newest first.
    assert rows[0]["started_at"] > rows[1]["started_at"]
