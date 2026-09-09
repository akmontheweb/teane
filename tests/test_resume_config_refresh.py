"""Config changes must take effect on ``teane resume``.

Resume deliberately invokes the graph with ``None`` so LangGraph continues
from the checkpoint rather than resetting messages / loop_counter /
node_state to their zero values (see ``run_graph``'s ``is_resume`` branch).
The side effect nobody accounted for: ``run_graph`` assembles a full
``initial_state`` — including every freshly-read ``*_config`` channel — and
then throws it away with everything else. A resumed session therefore ran on
whatever config.json said when the session FIRST started.

lumina 01a079dc: ``llm_dispatch.continue_on_length.code_reviewer`` was set to
``false`` between the original run and the resume, and the resumed run still
took the ``true`` branch out of its checkpoint.

``_refresh_config_channels_on_resume`` writes just the config channels back
onto the checkpointed state, leaving every work channel untouched.
"""

from __future__ import annotations

import asyncio
import re

from harness.graph import (
    _CONFIG_STATE_CHANNELS,
    _refresh_config_channels_on_resume,
)


class _State:
    def __init__(self, values):
        self.values = values


class _Graph:
    """Minimal compiled-graph stand-in: records aupdate_state payloads."""

    def __init__(self, values):
        self._values = values
        self.updates: list[dict] = []

    async def aget_state(self, config):
        return _State(self._values)

    async def aupdate_state(self, config, updates, **kw):
        self.updates.append(updates)
        self._values.update(updates)


def _run(graph, fresh):
    asyncio.run(_refresh_config_channels_on_resume(graph, {}, fresh))


def test_changed_config_channel_is_written_back():
    graph = _Graph({
        "llm_dispatch_config": {"continue_on_length": {"code_reviewer": True}},
        "loop_counter": {"total_repairs": 9},
        "messages": ["keep me"],
    })
    fresh = {
        "llm_dispatch_config": {"continue_on_length": {"code_reviewer": False}},
    }
    _run(graph, fresh)
    assert len(graph.updates) == 1
    assert graph.updates[0]["llm_dispatch_config"] == fresh["llm_dispatch_config"]


def test_work_channels_are_never_written():
    graph = _Graph({
        "llm_dispatch_config": {"a": 1},
        "loop_counter": {"total_repairs": 9},
        "messages": ["keep me"],
        "node_state": {"hitl_suspend": True},
    })
    # A caller handing over work channels alongside config must not be able to
    # clobber the checkpoint's in-flight state.
    fresh = {
        "llm_dispatch_config": {"a": 2},
        "loop_counter": {"total_repairs": 0},
        "messages": [],
        "node_state": {},
    }
    _run(graph, fresh)
    written = graph.updates[0]
    assert set(written) == {"llm_dispatch_config"}
    assert graph._values["loop_counter"] == {"total_repairs": 9}
    assert graph._values["messages"] == ["keep me"]
    assert graph._values["node_state"] == {"hitl_suspend": True}


def test_no_update_when_config_is_unchanged():
    cfg = {"continue_on_length": {"code_reviewer": False}}
    graph = _Graph({"llm_dispatch_config": cfg})
    _run(graph, {"llm_dispatch_config": dict(cfg)})
    assert graph.updates == []


def test_missing_channels_are_skipped_not_nulled():
    # A config section the caller did not resolve must not overwrite a
    # checkpointed value with None.
    graph = _Graph({"sandbox_config": {"backend": "docker"}})
    _run(graph, {"sandbox_config": None, "llm_dispatch_config": {"x": 1}})
    assert "sandbox_config" not in graph.updates[0]
    assert graph._values["sandbox_config"] == {"backend": "docker"}


def test_unreadable_state_does_not_block_resume():
    class _Broken(_Graph):
        async def aget_state(self, config):
            raise RuntimeError("checkpoint unreadable")

    graph = _Broken({})
    _run(graph, {"llm_dispatch_config": {"x": 1}})   # must not raise
    assert graph.updates == []


def test_update_failure_does_not_block_resume():
    class _Broken(_Graph):
        async def aupdate_state(self, config, updates, **kw):
            raise RuntimeError("write failed")

    graph = _Broken({"llm_dispatch_config": {"x": 0}})
    _run(graph, {"llm_dispatch_config": {"x": 1}})   # must not raise


def test_channel_list_matches_run_graph_assignments():
    """Drift guard: a config channel added to ``run_graph`` but not to
    ``_CONFIG_STATE_CHANNELS`` would silently keep its stale checkpoint value
    on every resume — the exact bug this module exists to prevent."""
    src = open("harness/graph.py", encoding="utf-8").read()
    seg = src[src.rindex("    initial_state = create_initial_state("):][:9000]
    assigned = set(re.findall(r'initial_state\["(\w+)"\]\s*=', seg))
    config_like = {
        a for a in assigned
        if a.endswith("_config") or a.endswith("_defaults")
    }
    missing = config_like - set(_CONFIG_STATE_CHANNELS)
    assert not missing, (
        f"config channels assigned in run_graph but absent from "
        f"_CONFIG_STATE_CHANNELS (they would stay stale on resume): {sorted(missing)}"
    )


def test_compiler_config_reaches_the_state():
    """``compiler_node`` reads ``state['compiler_config']`` for
    ``targeted_tests_first`` and ``advisory_exit_codes``; nothing ever put the
    section there, so both settings were inert and both read sites silently
    fell back to ``{}``."""
    src = open("harness/graph.py", encoding="utf-8").read()
    assert 'initial_state["compiler_config"] = compiler_config' in src
