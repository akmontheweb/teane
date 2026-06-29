"""Regression tests for embeddings cost tracking.

The opt-in OpenAI embeddings backend in ``harness/repo_index.py`` used
to bypass ``gateway.dispatch`` entirely, so any spend on
``/v1/embeddings`` was invisible to every cost surface (end-of-run
summary, ``teane status``, ``teane metrics``, dashboard ``/cost``).

These tests confirm the new path:

  * ``Gateway.track_embedding_call`` accounts the cost into
    ``gateway.session_tracker`` AND emits an ``embedding_call``
    observability event.
  * ``OpenAIEmbeddingsBackend.fit_chunks`` and
    ``OpenAIEmbeddingsBackend.vectorize_query`` both call into the
    tracker via the lazy ``get_gateway()`` lookup.
  * ``metrics.aggregate_session`` rolls ``embedding_call`` events into
    ``total_cost_usd`` so the JSONL replay matches the in-memory tracker.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any
from unittest.mock import patch

import pytest

from harness.gateway import (
    Gateway,
    GatewayConfig,
    ModelSpec,
    register_model,
)
from harness.repo_index import Chunk, OpenAIEmbeddingsBackend, RepoIndexConfig


_EMBED_MODEL_KEY = "openai:text-embedding-3-small"
_EMBED_RATE_PER_1M = 0.02


def _make_gateway() -> Gateway:
    register_model(_EMBED_MODEL_KEY, ModelSpec(
        provider="openai",
        model_id="text-embedding-3-small",
        context_window=8191,
        input_cost_per_1m=_EMBED_RATE_PER_1M,
        output_cost_per_1m=0.0,
        api_base_url="https://api.openai.com/v1",
        api_key="x",
    ))
    return Gateway(GatewayConfig(
        planning_primary=_EMBED_MODEL_KEY,
        patching_primary=_EMBED_MODEL_KEY,
        repair_primary=_EMBED_MODEL_KEY,
    ))


def test_track_embedding_call_updates_session_tracker():
    gateway = _make_gateway()
    assert gateway.session_tracker.get("total_cost_usd", 0.0) == 0.0
    cost = gateway.track_embedding_call(_EMBED_MODEL_KEY, prompt_tokens=1_000_000)
    expected = _EMBED_RATE_PER_1M
    assert cost == pytest.approx(expected, rel=1e-9)
    assert gateway.session_tracker["total_cost_usd"] == pytest.approx(expected, rel=1e-9)
    assert gateway.session_tracker["total_input_tokens"] == 1_000_000
    assert gateway.session_tracker["total_output_tokens"] == 0
    # Per-model rollup recorded.
    assert _EMBED_MODEL_KEY in gateway.session_tracker.get("per_model", {})
    assert gateway.session_tracker["per_model"][_EMBED_MODEL_KEY]["cost_usd"] == pytest.approx(
        expected, rel=1e-9,
    )


def test_track_embedding_call_unknown_model_accounts_at_zero():
    gateway = _make_gateway()
    cost = gateway.track_embedding_call("openai:nonexistent-embed-model", prompt_tokens=5_000)
    assert cost == 0.0
    # The call still updates the tracker's token totals so the operator
    # can see something happened, but cost is $0.
    assert gateway.session_tracker.get("total_cost_usd", 0.0) == 0.0
    assert gateway.session_tracker.get("total_input_tokens", 0) == 5_000


def test_track_embedding_call_emits_observability_event():
    gateway = _make_gateway()
    from harness.observability import configure_logging
    with tempfile.TemporaryDirectory() as log_dir:
        path = configure_logging(
            session_id="embed-emit-test", log_dir=log_dir, level="DEBUG",
        )
        gateway.track_embedding_call(_EMBED_MODEL_KEY, prompt_tokens=500_000)
        for handler in logging.getLogger().handlers:
            handler.flush()
        with open(path) as f:
            events = [json.loads(ln) for ln in f if ln.strip()]
    embed_events = [e for e in events if e.get("event") == "embedding_call"]
    assert len(embed_events) == 1
    evt = embed_events[0]
    assert evt["model"] == _EMBED_MODEL_KEY
    assert evt["tokens_in"] == 500_000
    assert evt["tokens_out"] == 0
    assert evt["cost_usd"] == pytest.approx(0.01, rel=1e-6)


class _StubHTTPResponse:
    """Minimal httpx response double for the embeddings backend tests."""

    def __init__(self, payload: dict[str, Any]):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _StubHTTPClient:
    """Drop-in for ``httpx.Client`` used as a context manager. Records
    each POST and returns a scripted response."""

    def __init__(self, scripted_payload: dict[str, Any]):
        self._payload = scripted_payload
        self.posts: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self) -> "_StubHTTPClient":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def post(self, url: str, json: dict[str, Any] | None = None, **_kwargs):
        self.posts.append((url, json or {}))
        return _StubHTTPResponse(self._payload)


def _fake_embedding_payload(num_inputs: int, prompt_tokens: int) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": i}
            for i in range(num_inputs)
        ],
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    }


def test_fit_chunks_tracks_embedding_cost(monkeypatch):
    os.environ["OPENAI_API_KEY"] = "sk-test"  # required by backend
    gateway = _make_gateway()
    from harness.graph import set_gateway
    set_gateway(gateway)

    cfg = RepoIndexConfig(openai_model="text-embedding-3-small")
    backend = OpenAIEmbeddingsBackend(cfg)
    chunks = [
        Chunk(file_path="a.py", chunk_index=0, content="def foo(): pass"),
        Chunk(file_path="b.py", chunk_index=0, content="def bar(): pass"),
    ]
    stub = _StubHTTPClient(_fake_embedding_payload(num_inputs=2, prompt_tokens=1_000_000))
    # The backend imports httpx lazily; patch the symbol on the module.
    with patch("httpx.Client", return_value=stub):
        vectors = backend.fit_chunks(chunks)

    assert len(vectors) == 2
    # Tracker reflects the single batch's prompt_tokens charge.
    assert gateway.session_tracker["total_cost_usd"] == pytest.approx(
        _EMBED_RATE_PER_1M, rel=1e-9,
    )
    assert gateway.session_tracker["total_input_tokens"] == 1_000_000


def test_vectorize_query_tracks_embedding_cost(monkeypatch):
    os.environ["OPENAI_API_KEY"] = "sk-test"
    gateway = _make_gateway()
    from harness.graph import set_gateway
    set_gateway(gateway)

    cfg = RepoIndexConfig(openai_model="text-embedding-3-small")
    backend = OpenAIEmbeddingsBackend(cfg)
    stub = _StubHTTPClient(_fake_embedding_payload(num_inputs=1, prompt_tokens=250_000))
    with patch("httpx.Client", return_value=stub):
        vec_json = backend.vectorize_query("how does the patcher node decide rollbacks?")
    assert isinstance(vec_json, str) and vec_json.startswith("[")
    expected = 0.25 * _EMBED_RATE_PER_1M  # 250k / 1M of the rate
    assert gateway.session_tracker["total_cost_usd"] == pytest.approx(expected, rel=1e-9)


def test_metrics_aggregate_session_includes_embedding_cost():
    """End-to-end: an emitted embedding_call event must be summed into
    metrics.SessionMetrics.total_cost_usd by aggregate_session."""
    gateway = _make_gateway()
    from harness.observability import configure_logging
    from harness.metrics import aggregate_session

    with tempfile.TemporaryDirectory() as log_dir:
        configure_logging(
            session_id="embed-metrics-test", log_dir=log_dir, level="DEBUG",
        )
        gateway.track_embedding_call(_EMBED_MODEL_KEY, prompt_tokens=200_000)
        gateway.track_embedding_call(_EMBED_MODEL_KEY, prompt_tokens=300_000)
        for handler in logging.getLogger().handlers:
            handler.flush()
        metrics = aggregate_session("embed-metrics-test", log_dir)
    expected = 0.5 * _EMBED_RATE_PER_1M  # 500k tokens total
    assert metrics.total_cost_usd == pytest.approx(expected, rel=1e-6)
    assert metrics.llm_call_count == 2  # embedding_call counts in the same bucket
    assert metrics.tokens_in == 500_000
