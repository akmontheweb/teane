"""A wedged harness must diagnose itself.

Two lumina runs died to a stall nothing could explain. 01a16872 sat in one
HTTP read for 68 minutes after its 200 header; 01a1790b stopped writing
anything at all for the 21 minutes before it was killed. Both post-mortems
had timestamps and nothing else, and both ended at a window rather than a
line.

``total_request_timeout_seconds`` does not close that gap even when it fires,
because cancelling an await records nothing about what the await was -- and in
01a1790b it did not fire at all.

The decisive test here is
:meth:`TestItNamesTheStuckAwait.test_the_dump_names_the_coroutine`: a
coroutine parked on an Event keeps its frames on the TASK, not on any thread,
so a thread-stack dump alone would show the loop sitting in ``select()`` --
accurate, and no more useful than the timestamps were.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import threading
import time

import pytest

import harness.stall_watchdog as w


@pytest.fixture(scope="session", autouse=True)
def _fast_polling():
    """Poll fast enough to test in milliseconds, for the WHOLE session.

    Deliberately not ``monkeypatch``: the monitor thread is a process-wide
    singleton that re-reads this global on every iteration, so a per-test
    patch hands it back the 5s production value between tests and any test
    unlucky enough to start in that window waits a full 5s for its dump.
    That raced three runs in four before this became session-scoped.
    """
    w._POLL_SECONDS = 0.02
    yield


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Point dumps at tmp_path and leave no scope registered behind."""
    monkeypatch.setattr(w, "STALL_DIR", str(tmp_path / "stalls"))
    w.reset_for_tests()
    yield
    w.reset_for_tests()


def _dumps(tmp_path) -> list[pathlib.Path]:
    d = tmp_path / "stalls"
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.name.startswith("stall_"))


def _wait_for_dump(tmp_path, *, count=1, timeout=5.0) -> list[pathlib.Path]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = [p for p in _dumps(tmp_path) if p.stat().st_size > 0]
        if len(found) >= count:
            return found
        time.sleep(0.02)
    return [p for p in _dumps(tmp_path) if p.stat().st_size > 0]


class TestItFiresOnlyWhenItShould:
    def test_a_block_that_finishes_in_time_dumps_nothing(self, tmp_path):
        with w.watch("quick", soft_seconds=5.0):
            pass
        assert _dumps(tmp_path) == []

    def test_zero_disables_it_entirely(self, tmp_path):
        """0 is the documented off switch. It must not register, so a
        disabled watch cannot dump however long the block runs."""
        with w.watch("off", soft_seconds=0.0):
            assert w.inflight_snapshot() == []
            time.sleep(0.2)
        assert _dumps(tmp_path) == []

    def test_an_overrun_dumps(self, tmp_path):
        with w.watch("slow", soft_seconds=0.05):
            found = _wait_for_dump(tmp_path)
        assert found, "a block past its budget must produce a dump"

    def test_the_dump_is_named_for_the_label(self, tmp_path):
        with w.watch("dispatch-repair", soft_seconds=0.05):
            found = _wait_for_dump(tmp_path)
        assert any("dispatch-repair" in p.name for p in found)

    def test_a_stuck_scope_keeps_dumping(self, tmp_path):
        """One dump says something is stuck; the second says whether it is
        wedged or merely slow. That comparison is the whole diagnostic."""
        with w.watch("repeating", soft_seconds=0.05, repeat_seconds=0.05):
            found = _wait_for_dump(tmp_path, count=2)
        assert len(found) >= 2

    def test_dumps_are_numbered_in_order(self, tmp_path):
        with w.watch("counted", soft_seconds=0.05, repeat_seconds=0.05):
            found = _wait_for_dump(tmp_path, count=2)
        bodies = [p.read_text() for p in found[:2]]
        assert "STALL DUMP #1" in bodies[0]
        assert "STALL DUMP #2" in bodies[1]


class TestWhatTheDumpContains:
    def test_it_records_the_phase_and_the_elapsed_time(self, tmp_path):
        with w.watch("phased", soft_seconds=0.05) as scope:
            scope.mark("post_response")
            found = _wait_for_dump(tmp_path)
        body = found[0].read_text()
        assert "phase       : post_response" in body
        assert "elapsed     :" in body
        assert f"pid         : {os.getpid()}" in body

    def test_it_lists_every_thread_stack(self, tmp_path):
        with w.watch("threads", soft_seconds=0.05):
            found = _wait_for_dump(tmp_path)
        body = found[0].read_text()
        assert "all thread stacks (faulthandler)" in body
        assert "Thread 0x" in body or "Current thread" in body

    def test_the_sections_are_in_order(self, tmp_path):
        """faulthandler writes to the raw fd, bypassing the file object's
        buffer. Without the flush before it, its output lands BEFORE the
        header and the dump reads back inside out."""
        with w.watch("ordering", soft_seconds=0.05):
            found = _wait_for_dump(tmp_path)
        body = found[0].read_text()
        assert body.index("=== STALL DUMP") < body.index("--- in flight ---")
        assert body.index("--- in flight ---") < body.index(
            "--- pending asyncio tasks ---"
        )
        assert body.index("--- pending asyncio tasks ---") < body.index(
            "all thread stacks (faulthandler)"
        )

    def test_it_marks_which_scope_breached(self, tmp_path):
        """With several dispatches in flight the register is only useful if
        it says which one the dump is about."""
        with w.watch("other-scope", soft_seconds=30.0):
            with w.watch("breaching-scope", soft_seconds=0.05):
                found = _wait_for_dump(tmp_path)
        body = found[0].read_text()
        assert ">> breaching-scope" in body
        assert "   other-scope" in body or "  other-scope" in body


class TestItNamesTheStuckAwait:
    def test_the_dump_names_the_coroutine(self, tmp_path):
        """The reason this module exists.

        A coroutine parked on an await holds its frames on the task. They
        appear on no thread stack, so faulthandler alone reports the loop in
        ``select()`` and the stall stays anonymous -- exactly the dead end
        both lumina post-mortems hit.
        """
        async def scenario():
            release = asyncio.Event()

            async def a_coroutine_that_never_returns():
                await release.wait()

            with w.watch("await-naming", soft_seconds=0.05):
                task = asyncio.create_task(a_coroutine_that_never_returns())
                await asyncio.sleep(0.01)
                found = await asyncio.get_running_loop().run_in_executor(
                    None, _wait_for_dump, tmp_path,
                )
            release.set()
            await task
            return found

        found = asyncio.run(scenario())
        assert found, "no dump was written"
        body = found[0].read_text()
        assert "a_coroutine_that_never_returns" in body, (
            "the dump must name the parked coroutine; without the asyncio "
            "task stacks it would only show the loop in select() and the "
            "stall would stay as anonymous as it was in run 01a1790b\n\n"
            + body
        )

    def test_it_counts_the_pending_tasks(self, tmp_path):
        async def scenario():
            release = asyncio.Event()
            tasks = [
                asyncio.create_task(release.wait()) for _ in range(3)
            ]
            with w.watch("counting", soft_seconds=0.05):
                found = await asyncio.get_running_loop().run_in_executor(
                    None, _wait_for_dump, tmp_path,
                )
            release.set()
            await asyncio.gather(*tasks)
            return found

        body = asyncio.run(scenario())[0].read_text()
        assert "pending asyncio tasks" in body
        # 3 waiters + the scenario task itself + the executor await.
        assert " pending\n" in body
        count = int(body.split(" pending\n")[0].rsplit("\n", 1)[-1].strip())
        assert count >= 4, f"only {count} tasks captured"

    def test_no_loop_is_said_so_rather_than_guessed(self, tmp_path):
        """A watch from a plain thread has no loop to enumerate. The dump
        must say that, not silently omit the section and read as 'no tasks
        were pending'."""
        with w.watch("threadland", soft_seconds=0.05):
            found = _wait_for_dump(tmp_path)
        body = found[0].read_text()
        assert "no running loop was captured" in body


class TestTheGilProofBackstop:
    def test_it_is_not_armed_before_a_breach(self, tmp_path):
        """Armed on every dispatch it would dump through healthy runs. It is
        a dead-man's switch for a scope already suspect."""
        with w.watch("healthy", soft_seconds=30.0, hard_seconds=0.05):
            time.sleep(0.2)
            assert w._hard_armed_for is None
        assert _dumps(tmp_path) == []

    def test_a_breach_arms_it(self, tmp_path):
        with w.watch("suspect", soft_seconds=0.05, hard_seconds=120.0):
            _wait_for_dump(tmp_path)
            deadline = time.monotonic() + 2.0
            while w._hard_armed_for is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert w._hard_armed_for is not None
            assert any(p.name.startswith("stall_hard_")
                       for p in (tmp_path / "stalls").iterdir())

    def test_leaving_the_scope_disarms_and_closes_it(self, tmp_path):
        """A timer left armed would dump into a closed fd for the rest of
        the run; a file left open leaks a descriptor per stall."""
        with w.watch("disarm", soft_seconds=0.05, hard_seconds=120.0):
            _wait_for_dump(tmp_path)
            deadline = time.monotonic() + 2.0
            while w._hard_armed_for is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert w._hard_armed_for is not None
            fh = w._hard_file
        assert w._hard_armed_for is None
        assert w._hard_file is None
        assert fh is not None and fh.closed

    def test_zero_disables_it(self, tmp_path):
        with w.watch("nohard", soft_seconds=0.05, hard_seconds=0.0):
            _wait_for_dump(tmp_path)
            time.sleep(0.1)
            assert w._hard_armed_for is None


class TestTheRegister:
    def test_it_reports_label_phase_and_elapsed(self):
        with w.watch("reported", soft_seconds=30.0) as scope:
            scope.mark("provider_call(deepseek-v4-pro)")
            snap = w.inflight_snapshot()
        assert len(snap) == 1
        label, phase, elapsed = snap[0]
        assert label == "reported"
        assert phase == "provider_call(deepseek-v4-pro)"
        assert elapsed >= 0.0

    def test_it_empties_on_exit(self):
        with w.watch("transient", soft_seconds=30.0):
            assert w.inflight_snapshot()
        assert w.inflight_snapshot() == []

    def test_it_empties_when_the_block_raises(self):
        with pytest.raises(ValueError):
            with w.watch("raising", soft_seconds=30.0):
                raise ValueError("boom")
        assert w.inflight_snapshot() == []

    def test_concurrent_scopes_do_not_share_a_phase(self):
        """One Gateway instance serves every concurrent dispatch in a run.
        Holding the scope on the instance had each dispatch overwriting the
        others' phase, which is why this goes through a ContextVar."""
        async def scenario():
            seen: dict[str, str] = {}

            async def leg(name: str, phase: str) -> None:
                with w.watch(name, soft_seconds=30.0):
                    w.mark_phase(phase)
                    await asyncio.sleep(0.05)
                    found = {
                        lbl: ph for lbl, ph, _ in w.inflight_snapshot()
                    }
                    seen[name] = found[name]

            await asyncio.gather(
                leg("leg-a", "provider_call(a)"),
                leg("leg-b", "post_response"),
            )
            return seen

        seen = asyncio.run(scenario())
        assert seen == {"leg-a": "provider_call(a)", "leg-b": "post_response"}

    def test_mark_phase_outside_a_watch_is_a_no_op(self):
        w.mark_phase("nowhere")  # must not raise


class TestItCannotHarmTheRun:
    def test_the_monitor_thread_is_a_daemon(self):
        """An earlier shutdown wedge in this harness was a non-daemon
        aiosqlite worker that CPython joined before atexit ran. A diagnostic
        must never be the reason a run fails to exit."""
        with w.watch("daemoncheck", soft_seconds=30.0):
            pass
        thread = w._monitor
        assert thread is not None
        assert thread.daemon

    def test_only_one_monitor_thread_is_ever_started(self):
        for i in range(5):
            with w.watch(f"n{i}", soft_seconds=30.0):
                pass
        live = [t for t in threading.enumerate() if t.name == "stall-watchdog"]
        assert len(live) == 1

    def test_an_unwritable_dump_dir_does_not_break_the_block(self, monkeypatch):
        monkeypatch.setattr(w, "STALL_DIR", "/proc/definitely/not/writable")
        with w.watch("unwritable", soft_seconds=0.05):
            time.sleep(0.2)  # the tick must swallow the I/O error
        assert w.inflight_snapshot() == []

    def test_a_dump_never_propagates_into_the_watched_block(self, tmp_path):
        """The watchdog runs on its own thread by construction, so a failure
        inside it cannot surface in the dispatch it is watching."""
        monitor_failed = []
        original = w._dump

        def exploding(scope):
            monitor_failed.append(scope.label)
            raise RuntimeError("dump blew up")

        w._dump = exploding
        try:
            with w.watch("exploding", soft_seconds=0.05):
                deadline = time.monotonic() + 2.0
                while not monitor_failed and time.monotonic() < deadline:
                    time.sleep(0.02)
            assert monitor_failed, "the monitor never reached the dump"
        finally:
            w._dump = original

    def test_dumps_are_kept_away_from_the_pruned_debug_directory(self):
        """``_prune_debug_dumps`` deletes the oldest .txt in ~/.harness/debug
        to stay under dump_max_files. A stall dump is the rarest artifact the
        harness produces and must not be evicted to make room for an
        ordinary LLM dump."""
        import harness.stall_watchdog as fresh

        # The fixture repoints STALL_DIR for the rest of the suite, and
        # reloading the module to recover the real value would reset its
        # registry and leak a second monitor thread. Read the assignment.
        src = pathlib.Path(fresh.__file__).read_text()
        assert 'STALL_DIR = os.path.expanduser("~/.harness/stalls")' in src
        assert os.path.expanduser("~/.harness/stalls") != os.path.expanduser(
            "~/.harness/debug"
        )


class TestTheGatewayWiring:
    def test_dispatch_watches_the_whole_call(self):
        """Not just the provider call. Run 01a1790b stopped with
        ``chat_completion`` already returned, so a watch ending where the
        HTTP read ends would have had nothing to say about it."""
        import inspect

        from harness.gateway import Gateway

        src = inspect.getsource(Gateway.dispatch)
        assert "watch(" in src
        assert "_dispatch_guarded" in src

    def test_the_label_carries_the_role(self):
        import inspect

        from harness.gateway import Gateway

        src = inspect.getsource(Gateway.dispatch)
        assert 'f"dispatch-{role_str}"' in src

    def test_both_phases_are_marked(self):
        """provider_call vs post_response is the distinction the log could
        not make: both sides of ``chat_completion`` look identical from
        outside, and the 21-minute gap was never attributed to either."""
        import inspect

        from harness.gateway import Gateway

        src = inspect.getsource(Gateway._dispatch_guarded)
        assert "provider_call" in src
        assert '_mark_phase("post_response")' in src

    def test_the_old_dispatch_signature_still_works(self):
        """Every caller in the graph passes these by keyword; the wrapper
        must forward them untouched."""
        import inspect

        from harness.gateway import Gateway

        guarded = inspect.signature(Gateway._dispatch_guarded).parameters
        for name in (
            "messages", "role", "budget_remaining_usd", "force_local",
            "model_override", "tools", "tool_choice", "cache_family",
        ):
            assert name in guarded, name
        wrapper = inspect.signature(Gateway.dispatch).parameters
        assert "kwargs" in wrapper


class TestTheConfig:
    def _loaded(self, overrides: dict):
        """Config as the REAL loader builds it, clamps and all."""
        from harness.gateway import create_gateway_from_config

        return create_gateway_from_config(
            {"llm_dispatch": overrides}
        ).config

    def test_the_defaults_are_on(self):
        cfg = self._loaded({})
        assert cfg.stall_watchdog_seconds == 300.0
        assert cfg.stall_watchdog_repeat_seconds == 300.0
        assert cfg.stall_watchdog_hard_seconds == 600.0

    def test_the_soft_budget_sits_below_the_total_deadline(self):
        """A watchdog that fires only at the deadline tells us nothing the
        deadline would not already have said."""
        cfg = self._loaded({})
        assert cfg.stall_watchdog_seconds < cfg.total_request_timeout_seconds

    def test_an_operator_can_override_each_one(self):
        cfg = self._loaded({
            "stall_watchdog_seconds": 60.0,
            "stall_watchdog_repeat_seconds": 30.0,
            "stall_watchdog_hard_seconds": 90.0,
        })
        assert cfg.stall_watchdog_seconds == 60.0
        assert cfg.stall_watchdog_repeat_seconds == 30.0
        assert cfg.stall_watchdog_hard_seconds == 90.0

    def test_zero_survives_the_clamp(self):
        """floor=0.0 on all three, or clamping 0 up to 1.0 would silently
        re-enable a watchdog an operator turned off -- and at 1s it would
        dump continuously."""
        cfg = self._loaded({
            "stall_watchdog_seconds": 0,
            "stall_watchdog_hard_seconds": 0,
        })
        assert cfg.stall_watchdog_seconds == 0.0
        assert cfg.stall_watchdog_hard_seconds == 0.0

    def test_the_keys_are_accepted_by_config_validation(self):
        from harness.cli import _KNOWN_NESTED_KEYS

        allowed = _KNOWN_NESTED_KEYS["llm_dispatch"]
        for key in (
            "stall_watchdog_seconds", "stall_watchdog_repeat_seconds",
            "stall_watchdog_hard_seconds",
        ):
            assert key in allowed, key

    def test_the_keys_are_type_checked(self):
        from harness.cli import _TYPE_SCHEMA

        for key in (
            "llm_dispatch.stall_watchdog_seconds",
            "llm_dispatch.stall_watchdog_repeat_seconds",
            "llm_dispatch.stall_watchdog_hard_seconds",
        ):
            assert _TYPE_SCHEMA[key] == (int, float), key

    def test_the_shipped_config_enables_them(self):
        import json

        cfg = json.loads(
            (pathlib.Path(__file__).parent.parent
             / "config" / "config.json").read_text()
        )["llm_dispatch"]
        assert cfg["stall_watchdog_seconds"] > 0
        assert cfg["stall_watchdog_hard_seconds"] > 0

    def test_they_are_documented(self):
        help_md = (
            pathlib.Path(__file__).parent.parent
            / "config" / "config_help.md"
        ).read_text()
        for key in (
            "stall_watchdog_seconds", "stall_watchdog_repeat_seconds",
            "stall_watchdog_hard_seconds",
        ):
            assert key in help_md, key
        assert "~/.harness/stalls" in help_md


class TestTheRealDispatchPathDumps:
    """Drives the shipped ``Gateway.dispatch`` against a provider that hangs.

    The unit tests above prove the watchdog works; this proves the shipped
    dispatch actually arms it. Both are needed -- 33 tests once passed
    against an in-turn retry that raised a TypeError on every firing,
    because not one of them executed the real code.
    """

    @staticmethod
    def _hanging_gateway(**cfg_overrides):
        from harness.gateway import (
            Gateway,
            GatewayConfig,
            ModelSpec,
            register_model,
        )

        key = "stub:hangs-watchdog"
        spec = ModelSpec(
            provider="stub", model_id="hangs-watchdog",
            context_window=128_000, input_cost_per_1m=0.5,
            output_cost_per_1m=1.0, api_base_url="", api_key="x",
        )
        register_model(key, spec)

        class _HangsInTheBodyRead:
            def __init__(self) -> None:
                self.spec = spec
                self.api_key = "x"

            async def a_provider_call_that_never_comes_back(self):
                await asyncio.sleep(3600)

            async def chat_completion(self, **_kw):
                # The lumina 01a16872 shape: the 200 header arrived and the
                # body never did.
                await self.a_provider_call_that_never_comes_back()

            async def close(self):
                return None

        provider = _HangsInTheBodyRead()
        gw = Gateway(GatewayConfig(
            planning_primary=key, patching_primary=key, repair_primary=key,
            total_request_timeout_seconds=30.0,
            max_retries=0,
            retry_network_cooloff_seconds=0.0,
            network_cooloff_attempts=0,
            fallback_on_timeout=False,
            **cfg_overrides,
        ))
        gw._providers[key] = provider  # type: ignore[index]
        return gw

    @pytest.mark.asyncio
    async def test_a_stalled_dispatch_dumps_and_names_the_provider_call(
        self, tmp_path,
    ):
        from harness.gateway import NodeRole

        gw = self._hanging_gateway(
            stall_watchdog_seconds=0.05,
            stall_watchdog_repeat_seconds=30.0,
            stall_watchdog_hard_seconds=0.0,
        )
        task = asyncio.create_task(gw.dispatch(
            messages=[
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
            ],
            role=NodeRole.REPAIR,
            budget_remaining_usd=10.0,
        ))
        found = await asyncio.get_running_loop().run_in_executor(
            None, _wait_for_dump, tmp_path,
        )
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: B014
            pass

        assert found, "a stalled real dispatch produced no dump"
        body = found[0].read_text()
        assert "dispatch-repair" in found[0].name, found[0].name
        assert "phase       : provider_call(" in body, (
            "the dump must say the provider call is still open — telling "
            "that apart from a stall in the harness's own post-response "
            "work is the question run 01a1790b could not answer\n\n" + body
        )
        assert "a_provider_call_that_never_comes_back" in body, (
            "the dump must name the frame that is actually stuck\n\n" + body
        )

    @pytest.mark.asyncio
    async def test_a_healthy_dispatch_dumps_nothing(self, tmp_path):
        """Vacuity guard: if the watchdog fired on ordinary dispatches the
        test above would pass for the wrong reason, and a real run would
        bury itself in dumps."""
        from harness.gateway import NodeRole

        gw = self._hanging_gateway(stall_watchdog_seconds=600.0)
        task = asyncio.create_task(gw.dispatch(
            messages=[
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
            ],
            role=NodeRole.REPAIR,
            budget_remaining_usd=10.0,
        ))
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: B014
            pass
        assert _dumps(tmp_path) == []

    @pytest.mark.asyncio
    async def test_the_scope_is_released_when_dispatch_is_cancelled(self):
        """A leaked scope would keep dumping for the rest of the run and
        report a dispatch that ended minutes ago as still in flight."""
        from harness.gateway import NodeRole

        gw = self._hanging_gateway(stall_watchdog_seconds=600.0)
        task = asyncio.create_task(gw.dispatch(
            messages=[
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
            ],
            role=NodeRole.REPAIR,
            budget_remaining_usd=10.0,
        ))
        await asyncio.sleep(0.2)
        assert [lbl for lbl, _, _ in w.inflight_snapshot()] == [
            "dispatch-repair"
        ]
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: B014
            pass
        assert w.inflight_snapshot() == []
