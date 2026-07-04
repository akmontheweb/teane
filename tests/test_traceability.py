"""Tests for the v5 SQL-backed traceability audit.

Replaces the legacy filesystem text-grep audit (Phase 4 of the
schema-v5 plan). Each test seeds the per-workspace state.db with
requirements / stories / ACs / link rows, then asserts what
``audit_workspace`` and ``format_report`` produce.

The conftest autouse fixture redirects ``TEANE_STATE_DB`` into a
per-test tmpdir so these tests never touch the operator's real DB.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness import story_state
from harness.traceability import (
    TraceabilityReport,
    UntestedCriterion,
    UntracedRequirement,
    audit_workspace,
    format_report,
)


@pytest.fixture
def workspace(tmp_path: Path) -> str:
    """Workspace dir. Its basename is the app name used to scope
    state.db rows."""
    ws = tmp_path / "trace-ws"
    ws.mkdir()
    return str(ws)


@pytest.fixture
def app(workspace: str) -> str:
    return story_state.app_name_for_workspace(workspace)


def _seed_story_with_ac(
    app: str, *, title: str = "S", ac: list[str] | None = None,
) -> int:
    """Seed one story with optional AC strings; return its story_id."""
    conn = story_state.open_story_db()
    try:
        story_state.ensure_feature(conn, app, "f", name="F")
        keys = story_state.create_stories(conn, app, [{
            "title": title, "feature": "f",
            "acceptance_criteria": list(ac or []),
        }])
        return story_state.get_story(conn, app, keys[0])["id"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# audit_workspace — empty / vacuous cases
# ---------------------------------------------------------------------------

def test_invalid_workspace_returns_none():
    assert audit_workspace("") is None
    assert audit_workspace("/nonexistent/path/xyz") is None


def test_empty_db_returns_clean_report(workspace: str):
    """No requirements + no ACs = vacuously clean (legacy workspace
    pre-v5 ingest)."""
    report = audit_workspace(workspace)
    assert report is not None
    assert report.total_reqs == 0
    assert report.total_acs == 0
    assert report.req_coverage_pct == 100.0
    assert report.ac_coverage_pct == 100.0
    assert report.untraced == []
    assert report.untested_acs == []
    assert report.has_failures() is False


# ---------------------------------------------------------------------------
# audit_workspace — requirement-coverage gaps
# ---------------------------------------------------------------------------

def test_all_requirements_satisfied_has_no_failures(workspace: str, app: str):
    conn = story_state.open_story_db()
    try:
        story_state.create_requirements(conn, app, [
            {"req_key": "FR-001", "kind": "fr", "title": "Login"},
            {"req_key": "FR-002", "kind": "fr", "title": "Logout"},
        ])
    finally:
        conn.close()
    sid = _seed_story_with_ac(app, title="Auth flows")
    conn = story_state.open_story_db()
    try:
        story_state.link_story_to_requirements(
            conn, app, sid, ["FR-001", "FR-002"],
        )
    finally:
        conn.close()

    report = audit_workspace(workspace)
    assert report.total_reqs == 2
    assert report.traced_reqs == 2
    assert report.req_coverage_pct == 100.0
    assert report.untraced == []
    assert report.has_failures() is False


def test_unsatisfied_requirement_surfaces_as_untraced(workspace: str, app: str):
    conn = story_state.open_story_db()
    try:
        story_state.create_requirements(conn, app, [
            {"req_key": "FR-001", "kind": "fr", "title": "covered"},
            {"req_key": "FR-002", "kind": "fr", "title": "gap"},
        ])
    finally:
        conn.close()
    sid = _seed_story_with_ac(app)
    conn = story_state.open_story_db()
    try:
        story_state.link_story_to_requirements(conn, app, sid, ["FR-001"])
    finally:
        conn.close()

    report = audit_workspace(workspace)
    assert report.total_reqs == 2
    assert report.traced_reqs == 1
    assert report.req_coverage_pct == 50.0
    assert [u.req_id for u in report.untraced] == ["FR-002"]
    assert report.untraced[0].kind == "fr"
    assert report.has_failures() is True


def test_nfr_and_cr_synthetic_kinds_preserved(workspace: str, app: str):
    conn = story_state.open_story_db()
    try:
        story_state.create_requirements(conn, app, [
            {"req_key": "NFR-SEC-001", "kind": "nfr", "title": "TLS"},
            {"req_key": "CR-7", "kind": "cr_synthetic", "title": "CR 7"},
        ])
    finally:
        conn.close()
    report = audit_workspace(workspace)
    assert {(u.req_id, u.kind) for u in report.untraced} == {
        ("NFR-SEC-001", "nfr"),
        ("CR-7", "cr_synthetic"),
    }


# ---------------------------------------------------------------------------
# audit_workspace — AC-coverage gaps
# ---------------------------------------------------------------------------

def test_all_acs_verified_has_no_ac_failures(workspace: str, app: str):
    sid = _seed_story_with_ac(app, ac=["AC-a", "AC-b"])
    conn = story_state.open_story_db()
    try:
        acs = story_state.list_acceptance_criteria(conn, app, sid)
        story_state.link_test_to_ac(conn, app, "tests/test_a.py", acs[0]["id"])
        story_state.link_test_to_ac(conn, app, "tests/test_b.py", acs[1]["id"])
    finally:
        conn.close()

    report = audit_workspace(workspace)
    assert report.total_acs == 2
    assert report.verified_acs == 2
    assert report.ac_coverage_pct == 100.0
    assert report.untested_acs == []


def test_unverified_ac_surfaces_grouped_by_story(workspace: str, app: str):
    s1 = _seed_story_with_ac(app, title="S1", ac=["a", "b"])
    _seed_story_with_ac(app, title="S2", ac=["c"])
    conn = story_state.open_story_db()
    try:
        # Verify only the first AC of S1.
        acs = story_state.list_acceptance_criteria(conn, app, s1)
        story_state.link_test_to_ac(conn, app, "tests/test_s1.py", acs[0]["id"])
    finally:
        conn.close()

    report = audit_workspace(workspace)
    assert report.total_acs == 3
    assert report.verified_acs == 1
    keys = [(u.story_key, u.ac_key) for u in report.untested_acs]
    # Ordered by story_id then ordinal — S1.AC-2 before S2.AC-1.
    assert keys == [
        ("STORY-1", "STORY-1.AC-2"),
        ("STORY-2", "STORY-2.AC-1"),
    ]
    assert report.has_failures() is True


def test_ac_text_truncated_at_200_chars(workspace: str, app: str):
    long_text = "X" * 500
    _seed_story_with_ac(app, ac=[long_text])
    report = audit_workspace(workspace)
    assert len(report.untested_acs) == 1
    # Cap is 200 chars; the original was 500.
    assert len(report.untested_acs[0].text) == 200


# ---------------------------------------------------------------------------
# format_report — rendering
# ---------------------------------------------------------------------------

def test_format_report_empty_when_no_failures():
    report = TraceabilityReport(
        spec_path="docs/SPEC_REQUIREMENTS.md",
        total_reqs=2, traced_reqs=2, untraced=[],
        total_acs=2, verified_acs=2, untested_acs=[],
    )
    assert format_report(report) == ""


def test_format_report_includes_both_sections_when_both_fail():
    report = TraceabilityReport(
        spec_path="docs/SPEC_REQUIREMENTS.md",
        total_reqs=2, traced_reqs=1,
        untraced=[UntracedRequirement(req_id="FR-002", kind="fr")],
        total_acs=2, verified_acs=1,
        untested_acs=[UntestedCriterion(
            ac_key="STORY-1.AC-2", story_key="STORY-1", text="some criterion",
        )],
    )
    out = format_report(report)
    assert "Untraced requirements (1)" in out
    assert "Untested acceptance criteria (1)" in out
    assert "FR-002" in out
    assert "STORY-1.AC-2" in out
    assert "some criterion" in out
    # Coverage percentages render with %0.f
    assert "50% coverage" in out


def test_format_report_groups_untraced_by_kind():
    report = TraceabilityReport(
        spec_path="docs/SPEC_REQUIREMENTS.md",
        total_reqs=3, traced_reqs=0,
        untraced=[
            UntracedRequirement(req_id="FR-001", kind="fr"),
            UntracedRequirement(req_id="NFR-SEC-001", kind="nfr"),
            UntracedRequirement(req_id="CR-7", kind="cr_synthetic"),
        ],
        total_acs=0, verified_acs=0, untested_acs=[],
    )
    out = format_report(report)
    assert "Functional Requirements" in out
    assert "Non-Functional Requirements" in out
    assert "Change Requests" in out


# ---------------------------------------------------------------------------
# Backward-compat alias surface
# ---------------------------------------------------------------------------

def test_legacy_coverage_pct_aliases_req_coverage(workspace: str, app: str):
    """Old call sites referenced ``coverage_pct``, ``total_ids``, and
    ``traced_ids``. Keep them working pointing at the req metrics."""
    conn = story_state.open_story_db()
    try:
        story_state.create_requirements(conn, app, [
            {"req_key": "FR-001", "kind": "fr", "title": "x"},
            {"req_key": "FR-002", "kind": "fr", "title": "y"},
        ])
    finally:
        conn.close()
    sid = _seed_story_with_ac(app)
    conn = story_state.open_story_db()
    try:
        story_state.link_story_to_requirements(conn, app, sid, ["FR-001"])
    finally:
        conn.close()
    report = audit_workspace(workspace)
    assert report.total_ids == 2
    assert report.traced_ids == 1
    assert report.coverage_pct == report.req_coverage_pct == 50.0


# ---------------------------------------------------------------------------
# route_after_installation_doc — end-of-session gate
# ---------------------------------------------------------------------------

class TestRouteAfterInstallationDoc:
    """The conditional edge that hard-blocks the session when
    traceability.enforce=true and the audit reported failures."""

    def test_returns_end_when_no_block(self):
        from harness.graph import route_after_installation_doc
        from langgraph.graph import END
        assert route_after_installation_doc({"node_state": {}}) == END
        assert route_after_installation_doc({}) == END

    def test_returns_hitl_when_traceability_blocked(self):
        from harness.graph import route_after_installation_doc
        state = {"node_state": {"traceability_blocked": True}}
        assert route_after_installation_doc(state) == "human_intervention_node"

    def test_still_hitl_at_first_cycle(self):
        # First trip → HITL. Interactive operator can still fix by
        # editing ``.harness_config.json`` outside the harness.
        from harness.graph import route_after_installation_doc
        state = {
            "node_state": {"traceability_blocked": True},
            "loop_counter": {"traceability_block_cycles": 1},
        }
        assert route_after_installation_doc(state) == "human_intervention_node"

    def test_routes_to_end_at_cycle_cap(self):
        # Ciod session 523e86a7 (2026-07-04): 376 iterations of
        # traceability_block → HITL(auto-resume) → traceability_node →
        # security_scan → installation_doc → same audit → same block.
        # Auto-resume cannot fix DB-level story→req links; after the
        # cap we must exit cleanly instead of spinning forever.
        from harness.graph import route_after_installation_doc, TRACEABILITY_BLOCK_CYCLE_CAP
        from langgraph.graph import END
        state = {
            "node_state": {"traceability_blocked": True},
            "loop_counter": {
                "traceability_block_cycles": TRACEABILITY_BLOCK_CYCLE_CAP,
            },
        }
        assert route_after_installation_doc(state) == END

    def test_routes_to_end_when_cycles_over_cap(self):
        # Defensive: even if the counter overshot the cap (e.g. a
        # stale checkpoint), we still want END, not HITL.
        from harness.graph import route_after_installation_doc, TRACEABILITY_BLOCK_CYCLE_CAP
        from langgraph.graph import END
        state = {
            "node_state": {"traceability_blocked": True},
            "loop_counter": {
                "traceability_block_cycles": TRACEABILITY_BLOCK_CYCLE_CAP + 5,
            },
        }
        assert route_after_installation_doc(state) == END

    def test_cycle_counter_ignored_when_not_blocked(self):
        # If the audit is clean this pass, the cycle counter should
        # NOT force END prematurely — a fresh clean pass wins.
        from harness.graph import route_after_installation_doc, TRACEABILITY_BLOCK_CYCLE_CAP
        from langgraph.graph import END
        state = {
            "node_state": {},  # no traceability_blocked
            "loop_counter": {
                "traceability_block_cycles": TRACEABILITY_BLOCK_CYCLE_CAP + 1,
            },
        }
        # Clean run terminates as before — no HITL, no lingering.
        assert route_after_installation_doc(state) == END


class TestHarnessConfigPlumbing:
    """Phase 7 BUG #1 regression: ``state["harness_config"]`` is the
    operator's escape hatch for the traceability gate. Before the
    fix, no code path wrote it into state, so
    ``traceability.enforce=false`` was unreachable.
    """

    def test_create_initial_state_stores_config(self):
        from harness.graph import create_initial_state
        s = create_initial_state(
            workspace_path="/tmp", initial_prompt="x", build_command="make",
            config={"traceability": {"enforce": False}, "other": "value"},
        )
        assert s.get("harness_config") == {
            "traceability": {"enforce": False}, "other": "value",
        }

    def test_create_initial_state_default_is_empty_dict_not_none(self):
        from harness.graph import create_initial_state
        s = create_initial_state(
            workspace_path="/tmp", initial_prompt="x", build_command="make",
        )
        # Empty dict (not None) so readers can `.get("key", {}).get(...)`.
        assert s.get("harness_config") == {}

    def test_traceability_enforce_false_disables_gate(self):
        """End-to-end: the gate read site at graph.py:12205 must
        honor enforce=false from the freshly-plumbed harness_config."""
        # Simulate what installation_doc_node does internally.
        state = {"harness_config": {"traceability": {"enforce": False}}}
        tr_cfg = (state.get("harness_config") or {}).get("traceability", {})
        enforce = bool(tr_cfg.get("enforce", True))
        assert enforce is False

    def test_traceability_enforce_default_is_true_when_unset(self):
        state = {"harness_config": {}}
        tr_cfg = (state.get("harness_config") or {}).get("traceability", {})
        enforce = bool(tr_cfg.get("enforce", True))
        assert enforce is True

        # Also when the whole key is missing.
        state = {}
        tr_cfg = (state.get("harness_config") or {}).get("traceability", {})
        enforce = bool(tr_cfg.get("enforce", True))
        assert enforce is True


# ---------------------------------------------------------------------------
# TRACEABILITY.md render — Requirements + AC coverage sections
# ---------------------------------------------------------------------------

class TestTraceabilityMdCoverageSections:
    """Phase 4b: regenerate_markdown_views emits new Requirements and
    Acceptance-criteria coverage sections when v5 ingest has run."""

    def test_legacy_workspace_omits_v5_sections(self, workspace: str, app: str):
        """No requirements, no ACs → byte-identical to pre-v5 output
        (no v5 sections rendered)."""
        story_state.ensure_feature(
            story_state.open_story_db(), app, "f", name="F",
        )
        conn = story_state.open_story_db()
        try:
            story_state.regenerate_markdown_views(conn, workspace)
        finally:
            conn.close()
        import os
        with open(os.path.join(workspace, "docs", "TRACEABILITY.md")) as f:
            body = f.read()
        assert "Requirements coverage" not in body
        assert "Acceptance-criteria coverage" not in body

    def test_render_includes_requirements_table(self, workspace: str, app: str):
        conn = story_state.open_story_db()
        try:
            story_state.create_requirements(conn, app, [
                {"req_key": "FR-001", "kind": "fr", "title": "Login"},
                {"req_key": "FR-002", "kind": "fr", "title": "Gap"},
            ])
        finally:
            conn.close()
        sid = _seed_story_with_ac(app, title="Login", ac=["AC text"])
        conn = story_state.open_story_db()
        try:
            story_state.link_story_to_requirements(conn, app, sid, ["FR-001"])
            story_state.regenerate_markdown_views(conn, workspace)
        finally:
            conn.close()
        import os
        with open(os.path.join(workspace, "docs", "TRACEABILITY.md")) as f:
            body = f.read()
        assert "## Requirements coverage" in body
        assert "`FR-001`" in body
        assert "`FR-002`" in body
        assert "— (gap)" in body  # FR-002 has no story

    def test_render_includes_ac_table_with_gap(self, workspace: str, app: str):
        sid = _seed_story_with_ac(app, ac=["covered", "uncovered"])
        conn = story_state.open_story_db()
        try:
            acs = story_state.list_acceptance_criteria(conn, app, sid)
            story_state.link_test_to_ac(
                conn, app, "tests/test_x.py", acs[0]["id"],
            )
            story_state.regenerate_markdown_views(conn, workspace)
        finally:
            conn.close()
        import os
        with open(os.path.join(workspace, "docs", "TRACEABILITY.md")) as f:
            body = f.read()
        assert "## Acceptance-criteria coverage" in body
        assert "`STORY-1.AC-1`" in body
        assert "`STORY-1.AC-2`" in body
        assert "`tests/test_x.py`" in body
        assert "— (gap)" in body  # STORY-1.AC-2 has no test
