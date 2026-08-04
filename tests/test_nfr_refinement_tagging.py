"""ADR-0004 #1 — the requirements-refinement NFR classifier: deterministic
keyword scoring first, one batched LLM call only for the ambiguous band, then
a ``**Class:**`` marker written into SPEC_REQUIREMENTS.md. Exercises the cli
glue (``_classify_and_tag_nfrs_in_spec``) with a stub gateway; the pure cascade
+ marker helpers are covered in test_decomposition.py.
"""

from __future__ import annotations

import pytest

from harness import cli
from harness.decomposition import _parse_nfr_class_markers


class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content

        class _U:
            cost_usd = 0.0

        self.usage = _U()


class _StubGateway:
    """Async gateway double: records dispatches and returns a canned body."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    async def dispatch(self, *, messages, role, budget_remaining_usd, **kw):
        self.calls += 1
        return _Resp(self.content), budget_remaining_usd - 0.001


def _spec(*nfr_blocks: str) -> str:
    return "## Requirements\n\n" + "\n\n---\n\n".join(nfr_blocks) + "\n"


_NFR_VALIDATION = (
    "#### Enabler Story: STORY-NFR-002 — Input Validation\n"
    "**Type:** Architecture\n"
    "**Description:** sanitize all input; reject SQL injection with a 422 and "
    "validate required fields.\n"
    "**Linked features:** FEAT-001"
)
_NFR_PERF = (
    "#### Enabler Story: STORY-NFR-001 — Dashboard Performance\n"
    "**Type:** Architecture\n"
    "**Description:** the endpoint must respond within 200 ms at p95 latency.\n"
    "**Linked features:** FEAT-002"
)
_NFR_LEAP = (  # no constraint/capability keyword → ambiguous → LLM band
    "#### Enabler Story: STORY-NFR-004 — Leap Day Handling\n"
    "**Type:** Architecture\n"
    "**Description:** birthdays on Feb 29 are handled correctly in non-leap years.\n"
    "**Linked features:** FEAT-003"
)


def _write(tmp_path, text: str) -> str:
    p = tmp_path / "SPEC_REQUIREMENTS.md"
    p.write_text(text)
    return str(p)


@pytest.mark.asyncio
async def test_flag_off_is_noop(tmp_path):
    path = _write(tmp_path, _spec(_NFR_VALIDATION))
    before = open(path).read()
    gw = _StubGateway("{}")
    await cli._classify_and_tag_nfrs_in_spec(
        path, gw, {"planning": {"embed_constraint_nfrs": False}}, 1.0,
    )
    assert open(path).read() == before
    assert gw.calls == 0


@pytest.mark.asyncio
async def test_unambiguous_only_writes_markers_without_llm(tmp_path):
    path = _write(tmp_path, _spec(_NFR_PERF, _NFR_VALIDATION))
    gw = _StubGateway("{}")
    await cli._classify_and_tag_nfrs_in_spec(
        path, gw, {"planning": {"embed_constraint_nfrs": True}}, 1.0,
    )
    markers = _parse_nfr_class_markers(open(path).read())
    assert markers == {"NFR-001": "capability", "NFR-002": "constraint"}
    assert gw.calls == 0  # both were deterministically confident


@pytest.mark.asyncio
async def test_ambiguous_triggers_llm_and_applies_override(tmp_path):
    path = _write(tmp_path, _spec(_NFR_VALIDATION, _NFR_LEAP))
    gw = _StubGateway('{"STORY-NFR-004": "constraint"}')
    await cli._classify_and_tag_nfrs_in_spec(
        path, gw, {"planning": {"embed_constraint_nfrs": True}}, 1.0,
    )
    markers = _parse_nfr_class_markers(open(path).read())
    assert gw.calls == 1                        # exactly one batched call
    assert markers["NFR-002"] == "constraint"   # deterministic, untouched
    assert markers["NFR-004"] == "constraint"   # LLM override applied


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_deterministic(tmp_path):
    class _BadGateway:
        calls = 0

        async def dispatch(self, **kw):
            _BadGateway.calls += 1
            raise RuntimeError("provider down")

    path = _write(tmp_path, _spec(_NFR_LEAP))
    await cli._classify_and_tag_nfrs_in_spec(
        path, _BadGateway(), {"planning": {"embed_constraint_nfrs": True}}, 1.0,
    )
    markers = _parse_nfr_class_markers(open(path).read())
    assert markers["NFR-004"] == "capability"   # fail-safe stands


@pytest.mark.asyncio
async def test_no_nfr_blocks_is_noop(tmp_path):
    path = _write(tmp_path, "## Requirements\n\nNo enabler stories here.\n")
    before = open(path).read()
    gw = _StubGateway("{}")
    await cli._classify_and_tag_nfrs_in_spec(
        path, gw, {"planning": {"embed_constraint_nfrs": True}}, 1.0,
    )
    assert open(path).read() == before
    assert gw.calls == 0
