#!/usr/bin/env python3
"""Thin eval harness for teane (audit #29).

Walks every task in ``evals/golden_set.yaml``, drives the harness through
each in an isolated temp workspace, and writes a per-task results record
to ``evals/results.json``. Pair with ``compare.py`` to print a delta vs.
``baseline.json``.

This is intentionally minimal: a subprocess shells out to ``teane build``
(greenfield) or ``teane patch`` (brownfield fixtures) per task, the runner reconstructs metrics from the on-disk JSONL logs via
``harness.metrics.aggregate_session``, and a success_check shell command
decides pass/fail per task.

Usage:
    python -m evals.run_eval                  # run every task → results.json
    python -m evals.run_eval --task fix_off_by_one   # one task
    python -m evals.run_eval --output baseline.json  # snapshot a new baseline
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

try:
    import yaml  # PyYAML — already pulled in by the harness's deps.
except ImportError:  # pragma: no cover — surfaced at CLI time
    print(
        "error: PyYAML is required. Install with `pip install pyyaml`.",
        file=sys.stderr,
    )
    raise

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent

# Test seam: the watchdog tests swap this for a fake child (e.g.
# ``[sys.executable, "-c", "import time; time.sleep(60)"]``) so timeout
# and process-group-kill behaviour is testable without LLM spend.
_HARNESS_CMD_PREFIX: list[str] = [sys.executable, "-m", "harness.cli"]
DEFAULT_GOLDEN_SET = EVAL_DIR / "golden_set.yaml"
DEFAULT_OUTPUT = EVAL_DIR / "results.json"
DEFAULT_LOG_DIR = Path(os.path.expanduser("~/.harness/logs"))


@dataclasses.dataclass
class TaskRecord:
    name: str
    success: bool
    harness_exit_code: int
    check_exit_code: Optional[int]
    wall_clock_s: float
    session_id: str
    workspace: str
    error: Optional[str] = None
    # Last ~600 chars of the harness subprocess's stderr on non-zero exit —
    # without this, an instant contract failure (wrong flags, missing
    # product_spec/) reads as a bare exit code and needs manual repro.
    stderr_tail: str = ""
    total_cost_usd: float = 0.0
    llm_call_count: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cached_tokens: int = 0
    cache_hit_rate: float = 0.0
    tool_call_count: dict[str, int] = dataclasses.field(default_factory=dict)
    tool_error_rates: dict[str, float] = dataclasses.field(default_factory=dict)
    system_prompt_lines: int = 0

    def to_jsonable(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    tasks = data.get("tasks") or []
    if not isinstance(tasks, list):
        raise ValueError(f"{path}: top-level `tasks:` must be a list.")
    return tasks


def _materialise_workspace(
    task: dict[str, Any], dest: Path, *, new_build: bool, description: str,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    fixture_dir = task.get("fixture_dir")
    if fixture_dir:
        src = (EVAL_DIR / fixture_dir).resolve()
        if not src.is_dir():
            raise FileNotFoundError(f"fixture_dir {src} does not exist for task {task.get('name')}")
        for entry in src.iterdir():
            target = dest / entry.name
            if entry.is_dir():
                shutil.copytree(entry, target, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, target)
    # The spec files are the authoritative task source since `teane run`
    # split into build/patch: build refuses to start without product_spec/,
    # patch without change_requests/*.txt. Write the task description into
    # the mode's authoritative location (the -p prompt is secondary).
    if new_build:
        spec_dir = dest / "product_spec"
        spec_dir.mkdir(exist_ok=True)
        (spec_dir / "task.md").write_text(description + "\n", encoding="utf-8")
    else:
        cr_dir = dest / "change_requests"
        cr_dir.mkdir(exist_ok=True)
        (cr_dir / "eval_task.txt").write_text(description + "\n", encoding="utf-8")


def _run_harness(
    *,
    workspace: Path,
    session_id: str,
    prompt: str,
    new_build: bool,
    timeout_s: int,
) -> tuple[int, Optional[str], str]:
    """Invoke the harness in a subprocess.

    Returns ``(exit_code, error_str, stderr_tail)`` — the tail is the last
    ~600 chars of the child's output on non-zero exit ("" otherwise), so an
    instant contract failure doesn't read as a bare exit code.

    ``teane run`` was split into the four-target CLI: greenfield tasks
    (``new_build: true``) drive ``teane build`` (workspace reset implied,
    ``--yes`` confirms it); brownfield fixture tasks drive ``teane patch``
    (which has no ``--yes``/reset semantics).
    """
    cmd = [
        *_HARNESS_CMD_PREFIX,
        "build" if new_build else "patch",
        "--workspace", str(workspace),
        "--prompt", prompt,
        "--session-id", session_id,
        # Eval profile: skip the interactive discovery interview rounds —
        # the task description in the spec file IS the whole requirement,
        # and discovery is the dominant wall-clock cost of the modern
        # pipeline (a greenfield hello-world spent 70+ min in synthesis
        # with discovery on).
        "--spec-discovery", "false",
        "--cd-discovery", "false",
    ]
    if new_build:
        cmd.append("--yes")
    env = os.environ.copy()
    env.setdefault("TEANE_EVAL", "1")

    # Manual watchdog instead of subprocess.run(timeout=...): a 600s cap
    # was observed NOT firing while the harness child ran 68+ minutes
    # (runner parked in poll()), and run()'s kill-on-timeout only signals
    # the direct child — MCP-server / sandbox grandchildren leak (an
    # orphaned `harness.cli` was found reparented to init). Popen with
    # start_new_session=True + killpg bounds the WHOLE tree, and the
    # drain threads + poll loop cannot miss the deadline.
    import signal
    import threading

    bufs: dict[str, list[str]] = {"out": [], "err": []}

    def _drain(stream: Any, key: str) -> None:
        try:
            for line in stream:
                bufs[key].append(line)
        except Exception:  # noqa: BLE001 — stream closed on kill
            pass

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except Exception as exc:  # noqa: BLE001 — surface every reason
        return 1, f"{type(exc).__name__}: {exc}", ""

    threads = [
        threading.Thread(target=_drain, args=(proc.stdout, "out"), daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, "err"), daemon=True),
    ]
    for t in threads:
        t.start()

    # Belt and suspenders on the deadline: monotonic AND wall clock (a
    # 600s subprocess.run timeout was once observed not firing while the
    # child ran 68+ minutes), plus a per-minute heartbeat so a stalled
    # watchdog is visible in the output stream instead of silent.
    deadline = time.monotonic() + timeout_s
    wall_deadline = time.time() + timeout_s
    next_beat = time.monotonic() + 60.0
    while proc.poll() is None:
        now = time.monotonic()
        if now >= deadline or time.time() >= wall_deadline:
            break
        if now >= next_beat:
            print(
                f"[eval] watchdog: {int(max(deadline - now, 0))}s remaining",
                flush=True,
            )
            next_beat = now + 60.0
        time.sleep(1.0)

    timed_out = proc.poll() is None
    if timed_out:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=15)
        except Exception:  # noqa: BLE001
            pass
    for t in threads:
        t.join(timeout=5)

    stderr_text = "".join(bufs["err"])
    stdout_text = "".join(bufs["out"])
    if timed_out:
        return 124, f"timeout after {timeout_s}s (process group killed)", (
            (stderr_text + stdout_text)[-600:]
        )
    tail = ""
    if proc.returncode != 0:
        tail = (stderr_text + stdout_text)[-600:]
    return proc.returncode, None, tail


def _run_success_check(workspace: Path, command: str) -> int:
    """Run the task's success_check shell command inside the workspace.

    Returns the exit code. A missing command short-circuits to 0 — the
    harness exit code carries the verdict in that case.
    """
    if not command:
        return 0
    proc = subprocess.run(
        command, shell=True, cwd=str(workspace), capture_output=True, text=True,
    )
    return proc.returncode


def _collect_metrics(session_id: str, log_dir: Path) -> dict[str, Any]:
    from harness.metrics import aggregate_session
    metrics = aggregate_session(session_id, str(log_dir))
    return metrics.to_jsonable()


def run_task(task: dict[str, Any], *, log_dir: Path) -> TaskRecord:
    name = str(task.get("name") or "").strip()
    if not name:
        raise ValueError("task missing required `name` field")
    description = str(task.get("description") or "").strip()
    if not description:
        raise ValueError(f"task {name}: missing required `description` field")
    new_build_default = "fixture_dir" not in task
    new_build = bool(task.get("new_build", new_build_default))
    timeout_s = int(task.get("timeout_s", 600))
    success_check = str(task.get("success_check") or "").strip()

    session_id = f"eval-{name}-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory(prefix=f"teane-eval-{name}-") as tmp:
        workspace = Path(tmp)
        try:
            _materialise_workspace(
                task, workspace, new_build=new_build, description=description,
            )
        except Exception as exc:  # noqa: BLE001
            return TaskRecord(
                name=name, success=False, harness_exit_code=-1, check_exit_code=None,
                wall_clock_s=0.0, session_id=session_id, workspace=str(workspace),
                error=f"workspace setup failed: {exc}",
            )

        t0 = time.monotonic()
        exit_code, error, stderr_tail = _run_harness(
            workspace=workspace, session_id=session_id, prompt=description,
            new_build=new_build, timeout_s=timeout_s,
        )
        wall_clock_s = round(time.monotonic() - t0, 3)

        check_code: Optional[int] = None
        if error is None and success_check:
            try:
                check_code = _run_success_check(workspace, success_check)
            except Exception as exc:  # noqa: BLE001
                error = f"success_check failed to run: {exc}"
                check_code = -1
        success = (exit_code == 0) and (check_code in (0, None)) and (error is None)

        try:
            agg = _collect_metrics(session_id, log_dir)
        except Exception as exc:  # noqa: BLE001
            agg = {}
            error = (error + " | " if error else "") + f"metrics collection failed: {exc}"

        return TaskRecord(
            name=name,
            success=success,
            harness_exit_code=exit_code,
            check_exit_code=check_code,
            wall_clock_s=wall_clock_s,
            session_id=session_id,
            workspace=str(workspace),
            error=error,
            stderr_tail=stderr_tail,
            total_cost_usd=float(agg.get("total_cost_usd") or 0.0),
            llm_call_count=int(agg.get("llm_call_count") or 0),
            tokens_in=int(agg.get("tokens_in") or 0),
            tokens_out=int(agg.get("tokens_out") or 0),
            cached_tokens=int(agg.get("cached_tokens") or 0),
            cache_hit_rate=float(agg.get("cache_hit_rate") or 0.0),
            tool_call_count=dict(agg.get("tool_call_count") or {}),
            tool_error_rates=dict(agg.get("tool_error_rates") or {}),
            system_prompt_lines=int(agg.get("system_prompt_lines") or 0),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the teane eval harness.")
    parser.add_argument(
        "--golden-set", default=str(DEFAULT_GOLDEN_SET),
        help="Path to golden_set.yaml (default: evals/golden_set.yaml).",
    )
    parser.add_argument(
        "--task", default=None,
        help="Name of a single task to run (default: every task).",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help="Path to write the results JSON (default: evals/results.json).",
    )
    parser.add_argument(
        "--log-dir", default=str(DEFAULT_LOG_DIR),
        help="Harness session log directory (default: ~/.harness/logs).",
    )
    args = parser.parse_args()

    tasks = _load_tasks(Path(args.golden_set))
    if args.task:
        tasks = [t for t in tasks if t.get("name") == args.task]
        if not tasks:
            print(f"error: task {args.task!r} not found in golden set.", file=sys.stderr)
            return 2

    log_dir = Path(os.path.expanduser(args.log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)

    results: list[TaskRecord] = []
    for task in tasks:
        name = task.get("name", "<unnamed>")
        print(f"[eval] {name}: starting", flush=True)
        rec = run_task(task, log_dir=log_dir)
        results.append(rec)
        verdict = "PASS" if rec.success else "FAIL"
        print(
            f"[eval] {name}: {verdict} "
            f"(harness_exit={rec.harness_exit_code}, "
            f"check_exit={rec.check_exit_code}, "
            f"wall={rec.wall_clock_s}s, "
            f"cost=${rec.total_cost_usd:.4f})",
            flush=True,
        )

    output = {
        "schema_version": 1,
        "tasks": [r.to_jsonable() for r in results],
        "summary": {
            "task_count": len(results),
            "pass_count": sum(1 for r in results if r.success),
            "fail_count": sum(1 for r in results if not r.success),
            "total_cost_usd": round(sum(r.total_cost_usd for r in results), 6),
            "total_wall_clock_s": round(sum(r.wall_clock_s for r in results), 3),
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, sort_keys=True)
    print(f"[eval] wrote {out_path}", flush=True)
    return 0 if all(r.success for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
