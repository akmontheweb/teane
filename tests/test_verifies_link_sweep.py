"""Regression tests for the batch-commit ``@verifies`` link sweep.

Ciod session 523e86a7 sealed 5 batches with tests passing, but the
end-of-session traceability audit reported ``test_verifies_ac`` empty
for every acceptance criterion. Root cause: ``_persist_verifies_links``
only runs inside ``test_generation_node`` on the same call where the
sandbox test run passes. In practice, the LLM's initial output fails
the marker gate or the sandbox run, and repair fixes it via
``compiler_node`` — which never re-enters ``test_generation_node``.
Result: markers are on disk but the DB link table stays empty.

The fix is a workspace-wide sweep called from ``batch_commit_node``:
walk every test file, parse the marker, INSERT OR IGNORE into
``test_verifies_ac``. Idempotent, best-effort, never fails the seal.
"""

from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture
def isolated_state_db(monkeypatch):
    """Point ``state_db_path`` at a fresh temp file so ``open_story_db``
    creates its schema there. Avoids touching the shared
    ``~/.harness/state.db``."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    from harness import story_state
    monkeypatch.setattr(story_state, "state_db_path", lambda: tmp.name)
    yield tmp.name
    os.unlink(tmp.name)


def _seed_workspace_with_stories(workspace_path: str, workspace_key: str) -> None:
    """Populate features/stories/acceptance_criteria so the sweep has
    something to link against. Mirrors what ``decomposition_node``
    produces on an agile run."""
    from harness import story_state
    conn = story_state.open_story_db()
    try:
        story_state.create_features(
            conn, workspace_key,
            [{"feature_key": "FEAT-001", "name": "Auth", "description": ""}],
        )
        story_state.create_stories(
            conn, workspace_key,
            [
                {
                    "title": "Log in",
                    "feature": "FEAT-001",
                    "acceptance_criteria": [
                        "Valid credentials return a JWT",
                        "Invalid credentials return 401",
                    ],
                    "requirement_keys": [],
                    "depends_on": [], "scope_files": [],
                },
                {
                    "title": "Log out",
                    "feature": "FEAT-001",
                    "acceptance_criteria": ["Blacklists the JWT"],
                    "requirement_keys": [],
                    "depends_on": [], "scope_files": [],
                },
            ],
        )
    finally:
        conn.close()


class TestVerifiesLinkSweep:
    def test_writes_python_verifies_link(self, isolated_state_db, tmp_path, monkeypatch):
        # app_name_for_workspace derives from the workspace basename;
        # match it to our seeded key.
        ws = tmp_path / "app-A"
        (ws / "tests").mkdir(parents=True)
        _seed_workspace_with_stories(str(ws), "app-A")
        (ws / "tests" / "test_login.py").write_text(
            "# @verifies: STORY-1.AC-1, STORY-1.AC-2\n"
            "def test_valid():\n"
            "    assert True\n"
        )
        from harness.test_generation import sweep_verifies_links
        scanned, inserted, dropped = sweep_verifies_links(str(ws))
        assert scanned >= 1
        assert inserted == 2
        assert dropped == 0

    def test_writes_ts_verifies_link(self, isolated_state_db, tmp_path):
        ws = tmp_path / "app-B"
        (ws / "src").mkdir(parents=True)
        _seed_workspace_with_stories(str(ws), "app-B")
        (ws / "src" / "login.test.tsx").write_text(
            "// @verifies: STORY-2.AC-1\n"
            "describe('login', () => { it('works', () => {}); });\n"
        )
        from harness.test_generation import sweep_verifies_links
        _, inserted, _ = sweep_verifies_links(str(ws))
        assert inserted == 1

    def test_ignores_source_files_even_with_marker_comment(
        self, isolated_state_db, tmp_path,
    ):
        # A source file (not test_*.py) that mentions "@verifies" in a
        # docstring/comment must NOT create a link. The sweep is only
        # for test files by convention.
        ws = tmp_path / "app-C"
        (ws / "src").mkdir(parents=True)
        _seed_workspace_with_stories(str(ws), "app-C")
        (ws / "src" / "login.py").write_text(
            "# @verifies: STORY-1.AC-1\n"
            "def login(): pass\n"
        )
        from harness.test_generation import sweep_verifies_links
        _, inserted, _ = sweep_verifies_links(str(ws))
        assert inserted == 0

    def test_ignores_ignored_directories(
        self, isolated_state_db, tmp_path,
    ):
        # Files inside node_modules / __pycache__ etc. must not be
        # scanned — otherwise the sweep would find bogus tests from
        # bundled deps on large workspaces.
        ws = tmp_path / "app-D"
        (ws / "node_modules" / "some-lib" / "src").mkdir(parents=True)
        (ws / "__pycache__").mkdir(parents=True)
        _seed_workspace_with_stories(str(ws), "app-D")
        (ws / "node_modules" / "some-lib" / "src" / "test_x.py").write_text(
            "# @verifies: STORY-1.AC-1\n"
        )
        (ws / "__pycache__" / "test_y.py").write_text(
            "# @verifies: STORY-1.AC-1\n"
        )
        from harness.test_generation import sweep_verifies_links
        scanned, inserted, _ = sweep_verifies_links(str(ws))
        assert scanned == 0
        assert inserted == 0

    def test_drops_unknown_ac_key(self, isolated_state_db, tmp_path):
        ws = tmp_path / "app-E"
        (ws / "tests").mkdir(parents=True)
        _seed_workspace_with_stories(str(ws), "app-E")
        (ws / "tests" / "test_x.py").write_text(
            "# @verifies: STORY-99.AC-42\n"
            "def test_x(): pass\n"
        )
        from harness.test_generation import sweep_verifies_links
        _, inserted, dropped = sweep_verifies_links(str(ws))
        assert inserted == 0
        assert dropped == 1

    def test_idempotent_across_repeated_sweeps(
        self, isolated_state_db, tmp_path,
    ):
        # Same batch may re-fire the sweep on an operator re-run
        # (or on batch_planner retry). The link_test_to_ac contract
        # is INSERT OR IGNORE — running the sweep a second time must
        # not double-count rows.
        ws = tmp_path / "app-F"
        (ws / "tests").mkdir(parents=True)
        _seed_workspace_with_stories(str(ws), "app-F")
        (ws / "tests" / "test_x.py").write_text(
            "# @verifies: STORY-1.AC-1\n"
            "def test_x(): pass\n"
        )
        from harness.test_generation import sweep_verifies_links
        _, first_run, _ = sweep_verifies_links(str(ws))
        assert first_run == 1
        _, second_run, _ = sweep_verifies_links(str(ws))
        assert second_run == 0, (
            "Second sweep must be a no-op — INSERT OR IGNORE prevents "
            "duplicate rows; the returned count is INSERTED not scanned."
        )

    def test_no_workspace_no_op(self, isolated_state_db):
        from harness.test_generation import sweep_verifies_links
        assert sweep_verifies_links("") == (0, 0, 0)
        assert sweep_verifies_links("/nonexistent/path") == (0, 0, 0)

    def test_traceability_audit_reports_ac_coverage_after_sweep(
        self, isolated_state_db, tmp_path,
    ):
        # End-to-end: after the sweep populates test_verifies_ac,
        # audit_workspace should report the AC as verified (i.e.
        # NOT in the untested_acs list). This is the ciod 523e86a7
        # regression — the audit reported 0/47 ACs verified because
        # this table stayed empty.
        ws = tmp_path / "app-G"
        (ws / "tests").mkdir(parents=True)
        _seed_workspace_with_stories(str(ws), "app-G")
        (ws / "tests" / "test_login.py").write_text(
            "# @verifies: STORY-1.AC-1, STORY-1.AC-2, STORY-2.AC-1\n"
            "def test_all(): pass\n"
        )
        from harness.test_generation import sweep_verifies_links
        from harness import traceability
        _, inserted, _ = sweep_verifies_links(str(ws))
        assert inserted == 3

        report = traceability.audit_workspace(str(ws))
        assert report is not None
        # Every AC in the seed was cited in the marker → zero
        # untested_acs remain.
        assert report.untested_acs == [], (
            f"Expected zero untested ACs after sweep; got "
            f"{[u.ac_key for u in report.untested_acs]}. "
            f"This is the ciod 523e86a7 regression."
        )
