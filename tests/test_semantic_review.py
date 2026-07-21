"""Semantic coverage review (P3) — harness.semantic_review.

Covers feature/story coverage gathering, the reviewer filter, and the node's
advisory-vs-enforce behavior. The LLM is always mocked.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from harness import semantic_review, story_state
from harness.gateway import NodeRole


_SPEC = (
    "# Spec\n\n"
    "## Epic: EPIC-001 — Root\n\n"
    "### Feature: FEAT-001 — Contact management\n**Parent epic:** EPIC-001\n\n"
    "#### Story: STORY-001 — Add contact\n**Parent feature:** FEAT-001\n\n"
    "**As a** user\n**I want** to add a contact\n**So that** it is stored.\n\n"
    "```gherkin\nScenario: valid contact is added\n  Given a name\n"
    "  When I submit\n  Then it is saved\n```\n\n"
    "### Feature: FEAT-002 — Empty feature\n**Parent epic:** EPIC-001\n\n"
    "**Description:** has no story under it.\n"
)


@pytest.fixture
def isolated_state_db(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    monkeypatch.setattr(story_state, "state_db_path", lambda: tmp.name)
    yield tmp.name
    os.unlink(tmp.name)


def _seed(tmp_path) -> tuple[str, str]:
    from harness.decomposition import _ingest_requirements
    from harness.spec_reconciler import reconcile_workspace_from_spec

    ws = str(tmp_path)
    os.makedirs(os.path.join(ws, "docs"), exist_ok=True)
    spec_path = os.path.join(ws, "docs", "SPEC_REQUIREMENTS.md")
    with open(spec_path, "w") as f:
        f.write(_SPEC)
    app = story_state.app_name_for_workspace(ws)
    _ingest_requirements(app, app, _SPEC)
    conn = story_state.open_story_db()
    reconcile_workspace_from_spec(conn, app, spec_path)
    conn.close()
    return ws, app


def test_gather_feature_coverage(isolated_state_db, tmp_path):
    _ws, app = _seed(tmp_path)
    conn = story_state.open_story_db()
    cov = semantic_review._gather_feature_coverage(conn, app)
    conn.close()
    # FEAT-001 has a covering story; FEAT-002 (no story) is excluded — that's a
    # deterministic gap, handled upstream, not a semantic one.
    keys = {c["req_key"] for c in cov}
    assert keys == {"FEAT-001"}
    feat = cov[0]
    assert feat["stories"][0]["story_key"] == "STORY-001"
    assert feat["stories"][0]["acceptance_criteria"]  # ACs captured


class _Resp:
    def __init__(self, content):
        self.content = content


class _FakeGateway:
    def __init__(self, content, reviewer="fake:reviewer"):
        self._content = content
        self._reviewer = reviewer

    def select_model(self, role):
        return self._reviewer if role == NodeRole.DECOMPOSITION_REVIEWER else "fake:model"

    async def dispatch(self, *, messages, role, budget_remaining_usd, cache_family):
        return _Resp(self._content), budget_remaining_usd


_COV = [{"req_key": "FEAT-001", "title": "F", "intent": "do things",
         "stories": [{"story_key": "STORY-001", "title": "s",
                      "acceptance_criteria": ["a"]}]}]


@pytest.mark.asyncio
async def test_review_returns_only_nonsatisfied():
    gw = _FakeGateway('[{"req_key":"FEAT-001","verdict":"partial","gap":"no delete"}]')
    findings, _ = await semantic_review.review_semantic_coverage(gw, _COV, 5.0)
    assert findings == [{"req_key": "FEAT-001", "verdict": "partial", "gap": "no delete"}]


@pytest.mark.asyncio
async def test_review_satisfied_yields_no_findings():
    gw = _FakeGateway('[{"req_key":"FEAT-001","verdict":"satisfied","gap":""}]')
    findings, _ = await semantic_review.review_semantic_coverage(gw, _COV, 5.0)
    assert findings == []


@pytest.mark.asyncio
async def test_review_skips_when_no_reviewer_configured():
    gw = _FakeGateway("[]", reviewer="")   # reviewer disabled
    findings, _ = await semantic_review.review_semantic_coverage(gw, _COV, 5.0)
    assert findings == []


@pytest.mark.asyncio
async def test_review_fail_open_on_bad_json():
    gw = _FakeGateway("not json at all")
    findings, _ = await semantic_review.review_semantic_coverage(gw, _COV, 5.0)
    assert findings == []


@pytest.mark.asyncio
async def test_node_advisory_does_not_block(isolated_state_db, tmp_path, monkeypatch):
    ws, _app = _seed(tmp_path)
    gw = _FakeGateway('[{"req_key":"FEAT-001","verdict":"partial","gap":"no delete"}]')
    monkeypatch.setattr("harness.graph.get_gateway", lambda: gw)
    out = await semantic_review.semantic_coverage_review_node(
        {"workspace_path": ws, "budget_remaining_usd": 5.0}
    )  # semantic_review_enforce defaults False
    assert out["node_state"]["semantic_coverage_findings"][0]["req_key"] == "FEAT-001"
    assert "semantic_coverage_gap" not in out["node_state"]
    assert "exit_code" not in out


@pytest.mark.asyncio
async def test_node_enforce_blocks(isolated_state_db, tmp_path, monkeypatch):
    ws, _app = _seed(tmp_path)
    gw = _FakeGateway('[{"req_key":"FEAT-001","verdict":"unsatisfied","gap":"misses point"}]')
    monkeypatch.setattr("harness.graph.get_gateway", lambda: gw)
    out = await semantic_review.semantic_coverage_review_node({
        "workspace_path": ws, "budget_remaining_usd": 5.0,
        "harness_config": {"traceability": {"semantic_review_enforce": True}},
    })
    assert out["node_state"]["semantic_coverage_gap"] is True
    assert out["exit_code"] == 1
