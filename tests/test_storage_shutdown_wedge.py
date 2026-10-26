"""An unclosed checkpointer must never park interpreter shutdown.

Regression cover for lumina 01a16c1e, where a 402 propagated out of
``gateway.dispatch`` and the process then printed:

    [harness] shutdown did not complete within 60s (a non-daemon worker
    thread is wedged) -- forcing exit(1).

``aiosqlite.Connection`` runs queries on a worker ``Thread`` built WITHOUT
``daemon=True``, which exits only when ``close()``/``stop()`` puts the stop
sentinel on its queue. CPython joins non-daemon threads BEFORE atexit hooks,
so one unclosed connection stalls the interpreter and the atexit hook that
reaps stdio MCP servers never runs. ``storage.py`` opened such a connection
on two factory paths and closed it on neither -- the file contained no
``close`` call at all.

The decisive test runs a REAL subprocess that leaks a connection and asserts
it exits promptly, because that is the actual failure: not a wrong value, a
process that will not die.

A trap worth knowing, which cost me a wrong conclusion: the leak only wedges
while something still REFERENCES the connection. Drop it and the garbage
collector retires the worker, so a snippet that discards the connection
exits cleanly even against the broken code. Verified against the pre-fix
shape: reference held -> wedged (killed at 15s); reference dropped -> 0s.
These tests therefore keep the saver alive past ``asyncio.run``.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

from harness.storage import (
    _OPEN_SQLITE_CONNECTIONS,
    _release_sqlite_workers,
    HarnessAsyncSqliteSaver,
)

#: Comfortably under the 60s watchdog, comfortably over a healthy exit.
_EXIT_BUDGET_S = 25.0


def _run_snippet(body: str) -> tuple[int, float, str]:
    """Run `body` in a fresh interpreter; return (rc, wall seconds, stderr)."""
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "snippet.py"
        script.write_text(textwrap.dedent(body).replace("<<DB>>", f"{td}/cp.db"))
        started = time.monotonic()
        proc = subprocess.run(
            [sys.executable, "-I", str(script)],
            capture_output=True, text=True, timeout=_EXIT_BUDGET_S + 35,
            cwd=str(Path(__file__).parent.parent),
        )
        return proc.returncode, time.monotonic() - started, proc.stderr


class TestTheProcessActuallyExits:
    """The bug as reported: a leaked connection holds the process open."""

    def test_a_leaked_checkpointer_does_not_stall_shutdown(self):
        rc, elapsed, stderr = _run_snippet(
            """
            import asyncio
            from harness.storage import HarnessAsyncSqliteSaver

            async def main():
                # Opened, NEVER closed, and the reference RETURNED so it
                # outlives the loop -- the shape every command path had.
                # Returning it matters: a discarded connection gets garbage
                # collected and the worker retires on its own, so a snippet
                # that drops it exits cleanly even against the broken code
                # and proves nothing. The harness holds its checkpointer for
                # the whole run.
                return await HarnessAsyncSqliteSaver.from_db_path(
                    db_path="<<DB>>"
                )

            saver = asyncio.run(main())
            print("main returned", flush=True)
            """
        )
        assert rc == 0, f"snippet failed (rc={rc}):\n{stderr[-2000:]}"
        assert elapsed < _EXIT_BUDGET_S, (
            f"interpreter took {elapsed:.1f}s to exit with a leaked "
            f"connection — shutdown is still wedged.\n{stderr[-2000:]}"
        )

    def test_it_also_exits_when_unwinding_from_an_exception(self):
        """The real trigger was a HarnessConfigError escaping dispatch, so
        the exception path matters more than the happy one."""
        rc, elapsed, stderr = _run_snippet(
            """
            import asyncio, sys
            from harness.storage import HarnessAsyncSqliteSaver

            holder = {}

            async def main():
                # Stash it outside the frame so unwinding does not drop the
                # only reference -- the checkpointer really does outlive the
                # exception in cmd_run.
                holder["saver"] = (
                    await HarnessAsyncSqliteSaver.from_db_path(
                        db_path="<<DB>>"
                    )
                )
                raise RuntimeError("402 Payment Required (simulated)")

            try:
                asyncio.run(main())
            except RuntimeError:
                sys.exit(1)
            """
        )
        assert rc == 1, f"expected the simulated failure's code; {stderr[-800:]}"
        assert elapsed < _EXIT_BUDGET_S, (
            f"took {elapsed:.1f}s to exit while unwinding — this is the "
            f"lumina 01a16c1e path.\n{stderr[-2000:]}"
        )

    def test_an_explicitly_closed_checkpointer_exits_too(self):
        """aclose() must not itself hang or raise on the clean path."""
        rc, elapsed, stderr = _run_snippet(
            """
            import asyncio
            from harness.storage import HarnessAsyncSqliteSaver

            async def main():
                saver = await HarnessAsyncSqliteSaver.from_db_path(
                    db_path="<<DB>>"
                )
                await saver.aclose()
                await saver.aclose()   # idempotent

            asyncio.run(main())
            """
        )
        assert rc == 0, stderr[-2000:]
        assert elapsed < _EXIT_BUDGET_S, stderr[-2000:]


class TestTheWorkerThread:
    @pytest.mark.asyncio
    async def test_the_worker_is_a_daemon(self):
        """The safety net: a thread we forgot to close cannot block exit."""
        with tempfile.TemporaryDirectory() as td:
            saver = await HarnessAsyncSqliteSaver.from_db_path(
                db_path=f"{td}/cp.db"
            )
            try:
                worker = saver.conn._thread
                assert worker.daemon is True, (
                    "aiosqlite's worker is non-daemon again — an unclosed "
                    "connection will park shutdown for 60s"
                )
            finally:
                await saver.aclose()

    @pytest.mark.asyncio
    async def test_aclose_retires_the_worker(self):
        """The real fix: deterministic close, not just a daemon flag."""
        with tempfile.TemporaryDirectory() as td:
            saver = await HarnessAsyncSqliteSaver.from_db_path(
                db_path=f"{td}/cp.db"
            )
            worker = saver.conn._thread
            assert worker.is_alive()
            await saver.aclose()
            worker.join(timeout=10.0)
            assert not worker.is_alive(), "aclose() left the worker running"

    @pytest.mark.asyncio
    async def test_the_connection_is_tracked_then_untracked(self):
        with tempfile.TemporaryDirectory() as td:
            saver = await HarnessAsyncSqliteSaver.from_db_path(
                db_path=f"{td}/cp.db"
            )
            assert saver.conn in _OPEN_SQLITE_CONNECTIONS
            await saver.aclose()
            assert saver.conn not in _OPEN_SQLITE_CONNECTIONS

    @pytest.mark.asyncio
    async def test_the_saver_still_works_through_the_tracked_opener(self):
        """The opener wraps the awaitable; prove it returns a usable saver
        and not the wrapper."""
        with tempfile.TemporaryDirectory() as td:
            saver = await HarnessAsyncSqliteSaver.from_db_path(
                db_path=f"{td}/cp.db"
            )
            try:
                import aiosqlite

                assert isinstance(saver.conn, aiosqlite.Connection)
                async with saver.conn.execute("select 1") as cur:
                    assert (await cur.fetchone())[0] == 1
            finally:
                await saver.aclose()


class TestTheBackstop:
    @pytest.mark.asyncio
    async def test_release_stops_an_unclosed_worker(self):
        """_release_sqlite_workers runs via threading._register_atexit, i.e.
        BEFORE the join that would otherwise hang."""
        with tempfile.TemporaryDirectory() as td:
            saver = await HarnessAsyncSqliteSaver.from_db_path(
                db_path=f"{td}/cp.db"
            )
            worker = saver.conn._thread
            assert worker.is_alive()
            _release_sqlite_workers()          # no await, no running close
            worker.join(timeout=10.0)
            assert not worker.is_alive()

    def test_release_is_safe_with_nothing_open(self):
        _release_sqlite_workers()  # must not raise

    def test_the_hook_is_registered_before_thread_join(self):
        """atexit would run too late; only _register_atexit helps."""
        import harness.storage as st

        src = Path(st.__file__).read_text()
        assert "threading._register_atexit(_release_sqlite_workers)" in src
        assert "atexit.register(_release_sqlite_workers)" not in src

    @pytest.mark.asyncio
    async def test_release_tolerates_an_already_closed_connection(self):
        with tempfile.TemporaryDirectory() as td:
            saver = await HarnessAsyncSqliteSaver.from_db_path(
                db_path=f"{td}/cp.db"
            )
            await saver.aclose()
            _release_sqlite_workers()  # must not raise


class TestBothFactoriesAreCovered:
    def test_neither_factory_calls_aiosqlite_connect_directly(self):
        """Both from_db_path and from_conn_string_with_gc leaked
        identically; fixing one would leave the other wedging."""
        import harness.storage as st

        src = Path(st.__file__).read_text()
        body = src[src.index("class HarnessAsyncSqliteSaver"):]
        assert "await aiosqlite.connect(" not in body, (
            "a saver factory still opens an untracked, non-daemon connection"
        )
        assert body.count("_open_tracked_connection(") == 2
