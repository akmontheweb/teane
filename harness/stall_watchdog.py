"""Make a wedged harness diagnose itself.

Two lumina runs died to a stall that nothing in the harness could explain.
Run 01a16872 sat in one HTTP read for 68 minutes after its 200 header; run
01a1790b stopped writing anything at all -- no log lines, no debug dumps, no
state.db updates -- for the 21 minutes before it was killed. Both times the
post-mortem had only timestamps to reason from, and both times that was
enough to localise the window and not the line.

``asyncio.timeout`` cannot close that gap. It is scheduled ON the event loop,
so it is useless in precisely the case where the loop is the thing that is
stuck, and when it does fire it cancels the await without ever saying what
the await was.

So the watchdog here runs OFF the loop, in a daemon thread, and its output is
a stack -- not a verdict:

* every thread's stack, via :mod:`faulthandler`, which writes to the fd in C
  and does not need the GIL (the one instrument that still works when a
  C extension is holding it);
* every asyncio task's stack, which is where a never-completing ``await``
  actually shows up -- a blocked coroutine's frames live on no thread stack
  at all, so faulthandler alone would only ever report the loop sitting in
  ``select()``;
* the in-flight register, so the dump names the role, the model and the
  phase rather than leaving them to be inferred from a log timestamp.

Dumps land in ``~/.harness/stalls/`` and NOT in ``~/.harness/debug/``, which
``_prune_debug_dumps`` trims by age. A stall dump is the rarest artifact the
harness produces; it must not be deleted to make room for an ordinary LLM
dump.

Nothing here aborts, cancels or recovers anything. A watchdog that also acted
would have to be trusted; this one only has to be read.
"""

from __future__ import annotations

import asyncio
import faulthandler
import logging
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

#: The scope enclosing the current task, so code deep inside a watched block
#: can mark its phase without the scope being threaded through every call.
#: A ContextVar and not an attribute on the caller: one Gateway instance
#: serves every concurrent dispatch in the run, and an attribute would have
#: each one overwriting the others' phase. asyncio copies the context per
#: task, so each dispatch sees only its own.
_current: ContextVar[Optional["_Scope"]] = ContextVar(
    "stall_watchdog_scope", default=None,
)


def mark_phase(phase: str) -> None:
    """Record the phase of the watched block enclosing this task.

    A no-op when nothing is watching, so callers need no guard.
    """
    scope = _current.get()
    if scope is not None:
        scope.mark(phase)

#: Where stall dumps are written. Deliberately outside ``~/.harness/debug``.
STALL_DIR = os.path.expanduser("~/.harness/stalls")

#: How often the monitor thread re-reads the register. Not a cap on anything
#: -- it only sets how precisely a breach time is reported -- so it stays a
#: module constant rather than another config key.
_POLL_SECONDS = 5.0

_lock = threading.Lock()
_inflight: dict[int, "_Scope"] = {}
_monitor: Optional[threading.Thread] = None
_seq = 0

#: ``faulthandler.dump_traceback_later`` writes straight to a file
#: descriptor, so the file object has to outlive the call that armed it or
#: the fd is closed before the timer ever fires.
_hard_file: Optional[Any] = None
_hard_armed_for: Optional[int] = None


class _Scope:
    """One piece of work being watched, and what the watchdog knows of it."""

    __slots__ = ("sid", "label", "phase", "started", "loop", "soft", "repeat",
                 "hard", "dumps", "next_dump")

    def __init__(
        self,
        sid: int,
        label: str,
        *,
        soft: float,
        repeat: float,
        hard: float,
        loop: Optional[asyncio.AbstractEventLoop],
    ) -> None:
        self.sid = sid
        self.label = label
        self.phase = "start"
        self.started = time.monotonic()
        self.loop = loop
        self.soft = soft
        self.repeat = repeat
        self.hard = hard
        self.dumps = 0
        self.next_dump = self.started + soft

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def mark(self, phase: str) -> None:
        """Record which part of the work is running.

        Cheap enough to call on the hot path, and it is the difference
        between a dump that says "a dispatch is stuck" and one that says the
        provider call already returned.
        """
        self.phase = phase


@contextmanager
def watch(
    label: str,
    *,
    soft_seconds: float,
    repeat_seconds: float = 0.0,
    hard_seconds: float = 0.0,
) -> Iterator[_Scope]:
    """Watch the enclosing block; dump every stack if it outlives its budget.

    ``soft_seconds <= 0`` disables the watch entirely and yields a scope whose
    :meth:`_Scope.mark` is a no-op in effect -- callers need no second code
    path for the disabled case.

    Synchronous by design: registering and unregistering must not await, or
    the watchdog would depend on the scheduler it exists to observe.
    """
    global _seq
    if soft_seconds <= 0:
        # Not registered and not published to the ContextVar: an unwatched
        # block's phase marks go nowhere, which is the point of "off".
        yield _Scope(-1, label, soft=0.0, repeat=0.0, hard=0.0, loop=None)
        return

    try:
        loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    with _lock:
        _seq += 1
        sid = _seq
        scope = _Scope(
            sid, label,
            soft=soft_seconds,
            repeat=repeat_seconds if repeat_seconds > 0 else soft_seconds,
            hard=hard_seconds,
            loop=loop,
        )
        _inflight[sid] = scope
    _ensure_monitor()
    token = _current.set(scope)
    try:
        yield scope
    finally:
        _current.reset(token)
        with _lock:
            _inflight.pop(sid, None)
            _disarm_hard_locked(sid)


def _ensure_monitor() -> None:
    """Start the monitor thread once, lazily.

    Daemon: a diagnostic must never be the reason a run fails to exit. That
    matters here specifically -- an earlier shutdown wedge in this harness
    was a non-daemon aiosqlite worker thread that CPython joined before
    atexit ran.
    """
    global _monitor
    if _monitor is not None and _monitor.is_alive():
        return
    with _lock:
        if _monitor is not None and _monitor.is_alive():
            return
        _monitor = threading.Thread(
            target=_monitor_loop, name="stall-watchdog", daemon=True,
        )
        _monitor.start()


def _monitor_loop() -> None:
    while True:
        time.sleep(_POLL_SECONDS)
        try:
            _tick()
        except Exception:  # noqa: BLE001 — a broken watchdog must stay quiet
            logger.debug("[stall-watchdog] tick failed", exc_info=True)


def _tick() -> None:
    now = time.monotonic()
    due: list[_Scope] = []
    with _lock:
        for scope in list(_inflight.values()):
            if now >= scope.next_dump:
                scope.dumps += 1
                scope.next_dump = now + scope.repeat
                due.append(scope)
    for scope in due:
        _dump(scope)
        # Arm the GIL-proof backstop the moment a scope becomes suspect,
        # rather than keeping a timer running through healthy dispatches.
        # Reaching this line proves the GIL was available NOW; it says
        # nothing about the next minute, which is the case the C timer
        # covers.
        if scope.hard > 0:
            _arm_hard(scope)


def _dump(scope: _Scope) -> None:
    """Write one dump for ``scope``. Best-effort, start to finish."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(
        STALL_DIR, f"stall_{stamp}_{scope.label}_{scope.dumps:02d}.txt",
    )
    logger.error(
        "[stall-watchdog] %s has been in phase %r for %.0fs (budget %.0fs) — "
        "dumping every thread and task stack to %s. Dump #%d; compare "
        "consecutive dumps: identical frames mean wedged, moving frames mean "
        "slow.",
        scope.label, scope.phase, scope.elapsed, scope.soft, path, scope.dumps,
    )
    try:
        os.makedirs(STALL_DIR, exist_ok=True)
        with open(path, "w") as fh:
            fh.write(
                f"=== STALL DUMP #{scope.dumps} ===\n"
                f"written     : {datetime.now(timezone.utc).isoformat()}\n"
                f"label       : {scope.label}\n"
                f"phase       : {scope.phase}\n"
                f"elapsed     : {scope.elapsed:.1f}s\n"
                f"soft budget : {scope.soft:.0f}s\n"
                f"pid         : {os.getpid()}\n"
            )
            _write_inflight(fh, scope.sid)
            _write_tasks(fh, scope.loop)
            fh.write("\n--- all thread stacks (faulthandler) ---\n")
            # faulthandler writes to the fd in C, bypassing this file
            # object's buffer: without the flush its output lands BEFORE
            # everything written above.
            fh.flush()
            faulthandler.dump_traceback(file=fh, all_threads=True)
    except Exception:  # noqa: BLE001
        logger.debug("[stall-watchdog] could not write %s", path, exc_info=True)


def _write_inflight(fh: Any, sid: int) -> None:
    fh.write("\n--- in flight ---\n")
    with _lock:
        scopes = sorted(_inflight.values(), key=lambda s: s.started)
    for other in scopes:
        marker = ">>" if other.sid == sid else "  "
        fh.write(
            f"{marker} {other.label} phase={other.phase} "
            f"elapsed={other.elapsed:.1f}s\n"
        )
    if not scopes:
        fh.write("  (empty — the scope ended while this dump was being "
                 "written)\n")


def _write_tasks(fh: Any, loop: Optional[asyncio.AbstractEventLoop]) -> None:
    """Dump every pending asyncio task's stack.

    The load-bearing half of the dump. A coroutine parked on an ``await``
    has its frames held by the task, not by any thread, so the thread
    stacks below would show only the loop in ``select()`` -- true, and
    useless for naming the await that never came back.

    ``asyncio.all_tasks`` iterates a WeakSet that the loop thread may be
    mutating, so a RuntimeError here is expected rather than exceptional.
    """
    fh.write("\n--- pending asyncio tasks ---\n")
    if loop is None:
        fh.write("  (no running loop was captured for this scope)\n")
        return
    tasks: list[Any] = []
    for attempt in range(3):
        try:
            tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
            break
        except RuntimeError:
            if attempt == 2:
                fh.write("  (task set kept mutating; could not snapshot it)\n")
                return
            time.sleep(0.05)
    fh.write(f"  {len(tasks)} pending\n")
    for task in tasks:
        try:
            fh.write(f"\n  * {task!r}\n")
            chain = _await_chain(task)
            if chain:
                fh.write("    await chain (outermost first):\n")
                for line in chain:
                    fh.write(f"      {line}\n")
            else:
                fh.write("    (no await chain; task may have just resumed)\n")
                fh.flush()
                task.print_stack(file=fh)
        except Exception as exc:  # noqa: BLE001 — one bad task must not end the dump
            fh.write(f"    (stack unavailable: {exc!r})\n")


def _await_chain(task: Any) -> list[str]:
    """Walk ``cr_await`` down a suspended task to the frame actually stuck.

    ``Task.get_stack()`` is not enough, and this is the subtle part. A
    suspended coroutine's frame has ``f_back is None`` -- the chain of awaits
    is not a call stack, so walking ``f_back`` the way ``get_stack`` does
    yields exactly one frame: the task's own outermost coroutine. For a
    stalled dispatch that reports ``dispatch`` and stops, which names the
    task we already knew about and not the call it is parked in.

    ``cr_await`` is the link ``f_back`` is missing: each awaiting coroutine
    points at what it awaits, so following it reaches the provider call, the
    socket read, or whichever frame is genuinely wedged. The walk ends at a
    Future (the usual terminus -- a sleep, a lock, a socket), at something
    with no further link, or at a repeat, since a cycle here would otherwise
    spin the watchdog forever.
    """
    out: list[str] = []
    seen: set[int] = set()
    node: Any = task.get_coro()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        frame = getattr(node, "cr_frame", None) or getattr(node, "gi_frame", None)
        if frame is not None:
            code = frame.f_code
            out.append(
                f'File "{code.co_filename}", line {frame.f_lineno}, '
                f'in {code.co_name}'
            )
        else:
            # A Future, a lock, a C-level awaitable: no frame, but its repr
            # usually says what is being waited on.
            out.append(f"awaiting {node!r}")
        node = (
            getattr(node, "cr_await", None)
            or getattr(node, "ag_await", None)
            or getattr(node, "gi_yieldfrom", None)
        )
    return out


def _arm_hard(scope: _Scope) -> None:
    """Arm ``faulthandler.dump_traceback_later`` as a dead-man's switch.

    One timer exists process-wide, so the first suspect scope claims it and
    keeps it until it ends. The timer lives in C and waits on a lock rather
    than on the event loop or the GIL, which makes it the only part of this
    module that still fires when Python itself cannot run.
    """
    global _hard_file, _hard_armed_for
    with _lock:
        if _hard_armed_for is not None:
            return
        try:
            os.makedirs(STALL_DIR, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = os.path.join(STALL_DIR, f"stall_hard_{stamp}.txt")
            # Unbuffered: the process this is meant for may never get the
            # chance to flush.
            _hard_file = open(path, "wb", buffering=0)
            faulthandler.dump_traceback_later(
                scope.hard, repeat=True, file=_hard_file,
            )
            _hard_armed_for = scope.sid
            logger.error(
                "[stall-watchdog] armed the GIL-proof backstop: every %.0fs "
                "from now, all thread stacks go to %s until %s ends. This one "
                "fires even if the interpreter is wedged — if it is the only "
                "dump present, suspect a C extension holding the GIL.",
                scope.hard, path, scope.label,
            )
        except Exception:  # noqa: BLE001
            logger.debug("[stall-watchdog] could not arm the backstop",
                         exc_info=True)
            _hard_file = None


def _disarm_hard_locked(sid: int) -> None:
    """Cancel the backstop when the scope that armed it ends. Caller holds
    ``_lock``."""
    global _hard_file, _hard_armed_for
    if _hard_armed_for != sid:
        return
    try:
        faulthandler.cancel_dump_traceback_later()
    except Exception:  # noqa: BLE001
        pass
    if _hard_file is not None:
        try:
            _hard_file.close()
        except Exception:  # noqa: BLE001
            pass
    _hard_file = None
    _hard_armed_for = None


def inflight_snapshot() -> list[tuple[str, str, float]]:
    """``(label, phase, elapsed)`` per watched scope. For tests and for
    anything that wants to report what the harness is waiting on."""
    with _lock:
        return [
            (s.label, s.phase, s.elapsed)
            for s in sorted(_inflight.values(), key=lambda s: s.started)
        ]


def reset_for_tests() -> None:
    """Clear module state between tests. Not for production use."""
    global _seq
    with _lock:
        for sid in list(_inflight):
            _disarm_hard_locked(sid)
        _inflight.clear()
        _seq = 0
