"""Requirement gap-fill (P2) — harness.gap_fill.

Covers the deterministic spec surgery (ordinal allocation, block rendering,
append) and the node end-to-end: draft (mocked LLM) → append to spec →
re-reconcile closes the gap. The LLM is always mocked — these assert the
plumbing, not model quality.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from harness import gap_fill, story_state
from harness.gateway import NodeRole


# --------------------------------------------------------------------------
# deterministic helpers
# --------------------------------------------------------------------------
def test_next_story_ordinal_ignores_nfr_keys():
    spec = (
        "#### Story: STORY-001 — a\n"
        "#### Story: STORY-007 — b\n"
        "#### Story: STORY-NFR-099 — enabler\n"
    )
    assert gap_fill.next_story_ordinal(spec) == 8
    assert gap_fill.next_story_ordinal("no stories here") == 1


def test_render_story_block_is_parseable_by_the_reconciler():
    from harness.spec_reconciler import _PARENT_FEAT_RE

    blk = gap_fill.render_story_block(
        story_key="STORY-010", title="Do the thing", parent_feature="FEAT-002",
        as_a="user", i_want="a capability", so_that="value is delivered",
        acceptance_criteria=["it works"],
    )
    assert "#### Story: STORY-010 — Do the thing" in blk
    assert _PARENT_FEAT_RE.search(blk).group(1) == "FEAT-002"
    assert "```gherkin" in blk


def test_append_stories_to_spec_assigns_fresh_keys(tmp_path):
    spec = tmp_path / "SPEC_REQUIREMENTS.md"
    spec.write_text("#### Story: STORY-003 — existing\n")
    keys = gap_fill.append_stories_to_spec(str(spec), [
        {"parent_feature": "FEAT-002", "title": "New A"},
        {"parent_feature": "FEAT-003", "title": "New B"},
    ])
    assert keys == ["STORY-004", "STORY-005"]
    text = spec.read_text()
    assert "STORY-004 — New A" in text and "STORY-005 — New B" in text
    assert "**Parent feature:** FEAT-002" in text


# --------------------------------------------------------------------------
# node end-to-end (mocked LLM)
# --------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeGateway:
    """Returns a scripted content string per cache_family; reviewer disabled
    unless a reviewer_model is given."""

    def __init__(self, by_family, reviewer_model=""):
        self._by_family = by_family
        self._reviewer_model = reviewer_model

    def select_model(self, role):
        if role == NodeRole.DECOMPOSITION_REVIEWER:
            return self._reviewer_model
        return "fake:model"

    async def dispatch(self, *, messages, role, budget_remaining_usd, cache_family):
        return _FakeResponse(self._by_family.get(cache_family, "[]")), budget_remaining_usd


@pytest.fixture
def isolated_state_db(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    monkeypatch.setattr(story_state, "state_db_path", lambda: tmp.name)
    yield tmp.name
    os.unlink(tmp.name)


_SPEC_WITH_GAP = (
    "# Spec\n\n"
    "## Epic: EPIC-001 — Root\n\n"
    "### Feature: FEAT-001 — Covered feature\n**Parent epic:** EPIC-001\n\n"
    "#### Story: STORY-001 — A story\n**Parent feature:** FEAT-001\n\n"
    "**As a** user\n**I want** X\n**So that** Y.\n\n"
    "```gherkin\nScenario: s\n  Given a\n  When b\n  Then c\n```\n\n"
    "### Feature: FEAT-002 — Uncovered feature\n**Parent epic:** EPIC-001\n\n"
    "**Description:** A feature with NO story under it.\n"
)


def _seed_workspace_with_gap(tmp_path) -> tuple[str, str]:
    from harness.decomposition import _ingest_requirements
    from harness.spec_reconciler import reconcile_workspace_from_spec

    ws = str(tmp_path)
    os.makedirs(os.path.join(ws, "docs"), exist_ok=True)
    spec_path = os.path.join(ws, "docs", "SPEC_REQUIREMENTS.md")
    with open(spec_path, "w") as f:
        f.write(_SPEC_WITH_GAP)
    app = story_state.app_name_for_workspace(ws)
    _ingest_requirements(app, app, _SPEC_WITH_GAP)
    conn = story_state.open_story_db()
    reconcile_workspace_from_spec(conn, app, spec_path)
    uncovered = [r["req_key"] for r in
                 story_state.requirements_without_satisfying_story(conn, app)]
    conn.close()
    return ws, app, uncovered


@pytest.mark.asyncio
async def test_node_fills_feature_gap_and_reconcile_closes_it(
    isolated_state_db, tmp_path, monkeypatch,
):
    ws, app, uncovered = _seed_workspace_with_gap(tmp_path)
    assert uncovered == ["FEAT-002"]  # precondition

    draft_json = (
        '[{"parent_feature": "FEAT-002", "title": "Handle FEAT-002", '
        '"as_a": "user", "i_want": "the FEAT-002 capability", '
        '"so_that": "it is covered", '
        '"acceptance_criteria": ["it works end to end"]}]'
    )
    gw = _FakeGateway({"decomposition:gap_fill": draft_json})
    monkeypatch.setattr("harness.graph.get_gateway", lambda: gw)

    out = await gap_fill.requirement_gap_fill_node(
        {"workspace_path": ws, "budget_remaining_usd": 5.0}
    )
    assert out["node_state"]["gap_fill_story_keys"]  # a story was appended
    assert out["loop_counter"]["requirement_gap_fill_cycles"] == 1

    # Re-ingest + reconcile from the now-augmented spec; the gap must close.
    from harness.decomposition import _ingest_requirements
    from harness.spec_reconciler import reconcile_workspace_from_spec
    spec_path = os.path.join(ws, "docs", "SPEC_REQUIREMENTS.md")
    _ingest_requirements(app, app, open(spec_path).read())
    conn = story_state.open_story_db()
    reconcile_workspace_from_spec(conn, app, spec_path)
    still = [r["req_key"] for r in
             story_state.requirements_without_satisfying_story(conn, app)]
    conn.close()
    assert still == []


@pytest.mark.asyncio
async def test_reviewer_rejection_drops_draft(isolated_state_db, tmp_path, monkeypatch):
    ws, app, _ = _seed_workspace_with_gap(tmp_path)
    draft_json = (
        '[{"parent_feature": "FEAT-002", "title": "Weak story", '
        '"acceptance_criteria": []}]'
    )
    # Reviewer configured, and it accepts nothing (empty array).
    gw = _FakeGateway(
        {"decomposition:gap_fill": draft_json,
         "decomposition_reviewer:gap_fill": "[]"},
        reviewer_model="fake:reviewer",
    )
    monkeypatch.setattr("harness.graph.get_gateway", lambda: gw)
    out = await gap_fill.requirement_gap_fill_node(
        {"workspace_path": ws, "budget_remaining_usd": 5.0}
    )
    assert out["node_state"]["gap_fill_story_keys"] == []  # nothing appended


@pytest.mark.asyncio
async def test_no_feature_gap_is_noop(isolated_state_db, tmp_path, monkeypatch):
    # Fully covered spec — the node has nothing to fill.
    from harness.decomposition import _ingest_requirements
    from harness.spec_reconciler import reconcile_workspace_from_spec
    ws = str(tmp_path)
    os.makedirs(os.path.join(ws, "docs"), exist_ok=True)
    spec = (
        "# Spec\n\n## Epic: EPIC-001 — Root\n\n"
        "### Feature: FEAT-001 — F\n**Parent epic:** EPIC-001\n\n"
        "#### Story: STORY-001 — S\n**Parent feature:** FEAT-001\n\n"
        "**As a** user\n**I want** X\n**So that** Y.\n\n"
        "```gherkin\nScenario: s\n  Given a\n  When b\n  Then c\n```\n"
    )
    spec_path = os.path.join(ws, "docs", "SPEC_REQUIREMENTS.md")
    open(spec_path, "w").write(spec)
    app = story_state.app_name_for_workspace(ws)
    _ingest_requirements(app, app, spec)
    conn = story_state.open_story_db()
    reconcile_workspace_from_spec(conn, app, spec_path)
    conn.close()

    called = {"dispatch": False}

    class _NoCallGateway(_FakeGateway):
        async def dispatch(self, **kw):
            called["dispatch"] = True
            return _FakeResponse("[]"), kw["budget_remaining_usd"]

    monkeypatch.setattr("harness.graph.get_gateway", lambda: _NoCallGateway({}))
    out = await gap_fill.requirement_gap_fill_node(
        {"workspace_path": ws, "budget_remaining_usd": 5.0}
    )
    assert out["node_state"]["gap_fill_story_keys"] == []
    assert called["dispatch"] is False  # no LLM call when nothing to fill
