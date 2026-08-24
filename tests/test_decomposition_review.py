"""Tests for the ADR-0007 decomposition-quality review (Phase 1).

Deterministic tier is pure; the LLM tier uses a fake gateway.
"""

from __future__ import annotations

import json

import pytest

from harness import decomposition_review as dr


def _story(key, title="T", *, deps=None, scope=None, acs=("does a thing",), desc=""):
    return {
        "story_key": key, "title": title, "description": desc,
        "feature_key": "FEAT-001",
        "depends_on": list(deps or []), "scope_files": list(scope or []),
        "acceptance_criteria": list(acs),
    }


# ---------------------------------------------------------------------------
# Deterministic tier
# ---------------------------------------------------------------------------


class TestDeterministic:
    def test_clean_decomposition_no_findings(self):
        stories = [_story("STORY-1"), _story("STORY-2", deps=["STORY-1"])]
        assert dr.deterministic_findings(stories) == []

    def test_dangling_dependency(self):
        f = dr.deterministic_findings([_story("STORY-1", deps=["STORY-9"])])
        assert len(f) == 1
        assert f[0]["dimension"] == "dependency" and f[0]["severity"] == "high"
        assert "STORY-9" in f[0]["problem"]

    def test_circular_dependency(self):
        stories = [
            _story("STORY-1", deps=["STORY-2"]),
            _story("STORY-2", deps=["STORY-1"]),
        ]
        f = dr.deterministic_findings(stories)
        cyc = [x for x in f if "circular" in x["problem"]]
        assert len(cyc) == 1

    def test_zero_ac_story(self):
        f = dr.deterministic_findings([_story("STORY-1", acs=())])
        assert any(x["dimension"] == "ac_quality" and "no acceptance" in x["problem"] for x in f)

    def test_empty_ac(self):
        f = dr.deterministic_findings([_story("STORY-1", acs=("real one", "   "))])
        assert any("empty/blank" in x["problem"] for x in f)

    def test_duplicate_story_same_title_and_scope(self):
        stories = [
            _story("STORY-1", title="Add a contact", scope=["api/contacts.py"]),
            _story("STORY-2", title="Add a contact!", scope=["api/contacts.py"]),
        ]
        f = dr.deterministic_findings(stories)
        dup = [x for x in f if x["dimension"] == "overlap"]
        assert len(dup) == 1 and dup[0]["suggested_action"] == "merge"

    def test_same_title_disjoint_scope_not_duplicate(self):
        stories = [
            _story("STORY-1", title="Setup", scope=["a.py"]),
            _story("STORY-2", title="Setup", scope=["b.py"]),
        ]
        # different scope → not flagged (avoid false positives)
        assert not [x for x in dr.deterministic_findings(stories) if x["dimension"] == "overlap"]

    def test_cycle_detection_no_false_positive_on_dag(self):
        stories = [
            _story("STORY-1"),
            _story("STORY-2", deps=["STORY-1"]),
            _story("STORY-3", deps=["STORY-1", "STORY-2"]),
        ]
        assert not [x for x in dr.deterministic_findings(stories) if "circular" in x["problem"]]


# ---------------------------------------------------------------------------
# LLM tier
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, content):
        self.content = content
        self.usage = {}


class _FakeGateway:
    def __init__(self, content="", *, has_model=True, raise_exc=None):
        self._content, self._has, self._raise = content, has_model, raise_exc

    def select_model(self, role):
        return "some:model" if self._has else ""

    async def dispatch(self, *, messages, role, budget_remaining_usd, **kw):
        if self._raise:
            raise self._raise
        return _Resp(self._content), budget_remaining_usd - 0.01


@pytest.mark.asyncio
class TestLLMReview:
    async def test_parses_and_filters_findings(self):
        stories = [_story("STORY-1"), _story("STORY-2")]
        payload = json.dumps([
            {"story_key": "STORY-1", "dimension": "right_sizing", "severity": "high",
             "problem": "too big", "suggested_action": "split"},
            {"story_key": "STORY-9", "dimension": "overlap", "severity": "low",
             "problem": "bogus key", "suggested_action": "merge"},  # dropped: unknown key
            {"story_key": "STORY-2", "dimension": "nonsense", "severity": "high",
             "problem": "bad dim", "suggested_action": "split"},  # dropped: bad dimension
        ])
        gw = _FakeGateway(payload)
        findings, budget = await dr.review_decomposition_quality(gw, stories, 1.0)
        assert [f["story_key"] for f in findings] == ["STORY-1"]
        assert findings[0]["source"] == "llm"
        assert budget == pytest.approx(0.99)

    async def test_no_reviewer_model_skips(self):
        gw = _FakeGateway(has_model=False)
        findings, budget = await dr.review_decomposition_quality(gw, [_story("STORY-1")], 1.0)
        assert findings == [] and budget == 1.0

    async def test_non_json_is_fail_open(self):
        gw = _FakeGateway("not json at all")
        findings, _ = await dr.review_decomposition_quality(gw, [_story("STORY-1")], 1.0)
        assert findings == []

    async def test_dispatch_error_is_fail_open(self):
        gw = _FakeGateway(raise_exc=RuntimeError("boom"))
        findings, budget = await dr.review_decomposition_quality(gw, [_story("STORY-1")], 1.0)
        assert findings == [] and budget == 1.0

    async def test_bad_severity_and_action_defaulted(self):
        payload = json.dumps([{"story_key": "STORY-1", "dimension": "ac_quality",
                               "severity": "catastrophic", "problem": "x",
                               "suggested_action": "explode"}])
        findings, _ = await dr.review_decomposition_quality(_FakeGateway(payload), [_story("STORY-1")], 1.0)
        assert findings[0]["severity"] == "medium"  # invalid -> medium
        assert findings[0]["suggested_action"] == "resize"  # invalid -> resize


# ---------------------------------------------------------------------------
# Node guard + doc rendering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestNode:
    async def test_disabled_is_passthrough(self):
        out = await dr.decomposition_quality_review_node(
            {"decomposition_config": {"quality_review": False}})
        assert out["node_state"]["skipped"] is True
        assert "exit_code" not in out


class TestRouting:
    def test_gap_routes_to_end(self):
        from langgraph.graph import END
        assert dr.route_after_decomposition_quality(
            {"node_state": {"decomposition_quality_gap": True}}) == END

    def test_clean_routes_to_gate(self):
        assert dr.route_after_decomposition_quality({"node_state": {}}) == "human_gatekeeper_node"


class TestDoc:
    def test_renders_table_with_findings(self):
        findings = [dr._finding("STORY-1", "right_sizing", "high", "too big", "split", source="llm")]
        doc = dr.render_review_doc(findings, 3)
        assert "Decomposition Quality Review" in doc
        assert "STORY-1" in doc and "right_sizing" in doc and "split" in doc

    def test_renders_clean(self):
        assert "well-formed" in dr.render_review_doc([], 5)
