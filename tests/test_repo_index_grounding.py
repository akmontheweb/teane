"""Tests for repo_index grounding injection into the generation path.

Wave-1 grounding: patching_node / repair_node retrieve neighbour code and inject
it so the generator writes consistent with the codebase. Here we pin the shared
helper ``_inject_repo_index_grounding`` (retrieval is mocked).
"""

from __future__ import annotations

import pytest

from harness import graph as g
from harness import repo_index as ri


@pytest.fixture
def _mock_retrieval(monkeypatch):
    """Patch repo_index retrieval so no real index/DB is needed."""
    async def _fake_query(workspace_path, query, *, cfg=None):
        return [("chunk-obj",)]  # opaque; render is also mocked

    calls = {"render": 0}

    def _fake_render(results, *, max_bytes):
        calls["render"] += 1
        return "server/app/db.py:1\n```\ndef get_conn(): ...\n```" if results else ""

    monkeypatch.setattr(ri, "async_query_top_chunks", _fake_query)
    monkeypatch.setattr(ri, "render_results_for_injection", _fake_render)
    return calls


def _msgs():
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "build the thing"},
    ]


@pytest.mark.asyncio
class TestGroundingHelper:
    async def test_disabled_injects_nothing(self, _mock_retrieval):
        msgs = _msgs()
        state = {"workspace_path": "/ws", "repo_index_config": {"enabled": False}}
        ok = await g._inject_repo_index_grounding(state, msgs, query="q", consumer="test")
        assert ok is False
        assert len(msgs) == 2  # unchanged

    async def test_no_config_injects_nothing(self, _mock_retrieval):
        msgs = _msgs()
        ok = await g._inject_repo_index_grounding(
            {"workspace_path": "/ws"}, msgs, query="q", consumer="test")
        assert ok is False
        assert len(msgs) == 2

    async def test_empty_query_injects_nothing(self, _mock_retrieval):
        msgs = _msgs()
        state = {"workspace_path": "/ws", "repo_index_config": {"enabled": True}}
        ok = await g._inject_repo_index_grounding(state, msgs, query="   ", consumer="test")
        assert ok is False

    async def test_enabled_injects_after_system(self, _mock_retrieval):
        msgs = _msgs()
        state = {"workspace_path": "/ws", "repo_index_config": {"enabled": True}}
        ok = await g._inject_repo_index_grounding(state, msgs, query="add contact endpoint", consumer="test")
        assert ok is True
        assert len(msgs) == 3
        # inserted right after the leading system message, before the user turn
        assert msgs[0]["role"] == "system" and msgs[0]["content"] == "system prompt"
        assert msgs[1]["role"] == "system"
        assert "Repository context (semantic retrieval)" in msgs[1]["content"]
        assert "get_conn" in msgs[1]["content"]
        assert msgs[2]["role"] == "user"

    async def test_empty_block_injects_nothing(self, monkeypatch):
        async def _empty_query(workspace_path, query, *, cfg=None):
            return []  # no results
        monkeypatch.setattr(ri, "async_query_top_chunks", _empty_query)
        monkeypatch.setattr(ri, "render_results_for_injection", lambda r, *, max_bytes: "")
        msgs = _msgs()
        state = {"workspace_path": "/ws", "repo_index_config": {"enabled": True}}
        ok = await g._inject_repo_index_grounding(state, msgs, query="q", consumer="test")
        assert ok is False
        assert len(msgs) == 2

    async def test_retrieval_error_is_fail_open(self, monkeypatch):
        async def _boom(workspace_path, query, *, cfg=None):
            raise RuntimeError("index corrupt")
        monkeypatch.setattr(ri, "async_query_top_chunks", _boom)
        msgs = _msgs()
        state = {"workspace_path": "/ws", "repo_index_config": {"enabled": True}}
        ok = await g._inject_repo_index_grounding(state, msgs, query="q", consumer="test")
        assert ok is False
        assert len(msgs) == 2  # generation never breaks on grounding failure
