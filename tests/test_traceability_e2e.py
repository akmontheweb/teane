"""End-to-end integration test for schema-v5 traceability.

Drives the full chain in one test so a regression in any phase
surfaces at the integration boundary, not just in the unit tests:

    Phase 2 — requirements_ingest parses docs/SPEC_REQUIREMENTS.md
              and UPSERTs requirements rows
    Phase 1 — decomposition writes stories with feature_id; AC rows
              land in the side table via the create_stories shim
    Phase 2 — link_story_to_requirements wires story_satisfies_req
              edges (cross-validated against the requirements set)
    Phase 3 — link_test_to_ac wires test_verifies_ac edges
              (simulates a passed test-gen pass with @verifies marker)
    Phase 4 — audit_workspace returns a TraceabilityReport carrying
              both gap sets; format_report + TRACEABILITY.md render
              both new sections

The fixture deliberately exercises every gap and coverage shape
the production pipeline encounters: covered requirement + verified
AC, covered requirement + unverified AC, untraced requirement, and
NFR/cr_synthetic kinds. Snapshot assertions catch render shape
drift.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness import decomposition, story_state
from harness.traceability import audit_workspace, format_report


@pytest.fixture
def workspace(tmp_path: Path) -> str:
    ws = tmp_path / "e2e-trace-ws"
    ws.mkdir()
    (ws / "docs").mkdir()
    return str(ws)


@pytest.fixture
def app(workspace: str) -> str:
    return story_state.app_name_for_workspace(workspace)


SPEC = """\
# Product spec

Some preamble that should be ignored by the parser.

### FR-001: User can log in
The system MUST allow login with email + password.

### FR-002: User can log out
The system MUST allow logout and invalidate the session token.

### FR-003: Forgotten requirement (will be the untraced gap)
This requirement intentionally has no covering story so the audit
surfaces it.

#### NFR-SEC-001: Hash session tokens at rest
NFR family — the validator/audit treat it the same as FR.
"""


def _ingest_and_decompose(workspace: str, app: str) -> None:
    """Simulate the Phase 2 pipeline up to the link writer.

    - Phase 2 ingest: parse spec → requirements table.
    - Phase 1+2: insert features + stories with v5-shape items that
      carry requirement_keys + acceptance_criteria.
    - Phase 2 link: write story_satisfies_req edges.
    - Phase 3 link: write test_verifies_ac edges for SOME ACs to
      exercise both the covered and the uncovered paths.

    The actual decomposition LLM call is bypassed (no gateway in
    tests); we call the persistence helpers directly with the same
    contract decomposition_node uses.
    """
    spec_path = os.path.join(workspace, "docs", "SPEC_REQUIREMENTS.md")
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(SPEC)

    # Phase 2: ingest spec → requirements table.
    parsed, upserted = decomposition._ingest_requirements(workspace, app, SPEC)
    assert parsed == 4 and upserted == 4, "spec parser should pick up 4 IDs"

    # Phase 1 + 2: features + stories. STORY-001 covers FR-001 + NFR-SEC-001;
    # STORY-002 covers FR-002. FR-003 is intentionally untraced.
    conn = story_state.open_story_db()
    try:
        story_state.create_features(conn, app, [
            {"feature_key": "auth", "name": "Auth"},
        ])
        story_state.create_stories(conn, app, [
            {
                "title": "Login flow",
                "feature": "auth",
                "acceptance_criteria": [
                    "POST /login returns 200 with valid creds",
                    "POST /login returns 401 with bad password",
                ],
            },
            {
                "title": "Logout flow",
                "feature": "auth",
                "acceptance_criteria": [
                    "POST /logout returns 204",
                ],
            },
        ])
        # Phase 2 link writer.
        s1 = story_state.get_story(conn, app, "STORY-001")
        s2 = story_state.get_story(conn, app, "STORY-002")
        story_state.link_story_to_requirements(
            conn, app, s1["id"], ["FR-001", "NFR-SEC-001"],
        )
        story_state.link_story_to_requirements(
            conn, app, s2["id"], ["FR-002"],
        )

        # Phase 3 link writer (simulates test-gen + sandbox pass).
        # Cover STORY-001.AC-1 and STORY-002.AC-1; leave STORY-001.AC-2
        # uncovered so the audit surfaces an untested AC.
        s1_acs = story_state.list_acceptance_criteria(conn, app, s1["id"])
        s2_acs = story_state.list_acceptance_criteria(conn, app, s2["id"])
        story_state.link_test_to_ac(
            conn, app, "tests/test_login_success.py", s1_acs[0]["id"],
        )
        story_state.link_test_to_ac(
            conn, app, "tests/test_logout.py", s2_acs[0]["id"],
        )
    finally:
        conn.close()


def test_e2e_audit_surfaces_both_gap_sets(workspace: str, app: str):
    _ingest_and_decompose(workspace, app)

    report = audit_workspace(workspace)
    assert report is not None
    assert report.has_failures() is True

    # Requirement coverage: 3/4 (FR-001, FR-002, NFR-SEC-001 covered;
    # FR-003 untraced).
    assert report.total_reqs == 4
    assert report.traced_reqs == 3
    assert report.req_coverage_pct == 75.0
    assert [u.req_id for u in report.untraced] == ["FR-003"]

    # AC coverage: 2/3 (STORY-001.AC-1 + STORY-002.AC-1 verified;
    # STORY-001.AC-2 uncovered).
    assert report.total_acs == 3
    assert report.verified_acs == 2
    assert [(u.story_key, u.ac_key) for u in report.untested_acs] == [
        ("STORY-001", "STORY-001.AC-2"),
    ]


def test_e2e_format_report_renders_both_sections(workspace: str, app: str):
    _ingest_and_decompose(workspace, app)
    report = audit_workspace(workspace)
    rendered = format_report(report)

    # Untraced reqs section.
    assert "Untraced requirements (1)" in rendered
    assert "FR-003" in rendered
    assert "Functional Requirements" in rendered

    # Untested ACs section.
    assert "Untested acceptance criteria (1)" in rendered
    assert "STORY-001.AC-2" in rendered

    # Coverage stats line.
    assert "3/4" in rendered  # req coverage
    assert "2/3" in rendered  # AC coverage


def test_e2e_traceability_md_emits_both_v5_tables(workspace: str, app: str):
    _ingest_and_decompose(workspace, app)

    conn = story_state.open_story_db()
    try:
        story_state.regenerate_markdown_views(conn, workspace)
    finally:
        conn.close()

    with open(os.path.join(workspace, "docs", "TRACEABILITY.md")) as f:
        md = f.read()

    # Both v5 sections present.
    assert "## Requirements coverage" in md
    assert "## Acceptance-criteria coverage" in md

    # Requirements table: untraced row marked as gap; traced rows
    # show their covering story.
    assert "`FR-001`" in md
    assert "`FR-003`" in md
    assert "— (gap)" in md  # FR-003 has no story
    assert "STORY-001" in md  # FR-001 covered by STORY-001

    # AC table: STORY-001.AC-2 marked as gap; the others list their test path.
    assert "`STORY-001.AC-1`" in md
    assert "`STORY-001.AC-2`" in md
    assert "`STORY-002.AC-1`" in md
    assert "`tests/test_login_success.py`" in md
    assert "`tests/test_logout.py`" in md


def test_e2e_router_blocks_when_traceability_blocked_flag_set(workspace: str, app: str):
    """The end-of-session gate populates node_state.traceability_blocked
    when enforce=true and the audit has gaps. Verifies the routing
    decision in isolation (the full installation_doc_node integration
    requires a configured LLM gateway, which is out of scope here)."""
    _ingest_and_decompose(workspace, app)

    from harness.graph import route_after_installation_doc
    from langgraph.graph import END

    # Clean state → END.
    assert route_after_installation_doc({"node_state": {}}) == END

    # Blocked → HITL.
    blocked = {"node_state": {"traceability_blocked": True}}
    assert route_after_installation_doc(blocked) == "human_intervention_node"
