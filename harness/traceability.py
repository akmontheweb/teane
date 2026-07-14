"""SQL-backed requirement and acceptance-criterion traceability audit.

Phase 4 of the schema-v5 traceability plan: replaces the legacy
text-grep audit that walked the workspace looking for ``FR-NNN``
tokens in source/test/doc files. With the v5 schema, requirements
and acceptance criteria are first-class rows in ``state.db`` linked
to stories (``story_satisfies_req``) and tests
(``test_verifies_ac``), so the audit collapses to two SQL queries:

  1. Requirements with no satisfying story — the planner forgot
     to cover the spec requirement (or rejected it without operator
     consent).
  2. Acceptance criteria with no verifying test — the test
     generator didn't emit a ``@verifies`` marker for this AC, so
     it ships uncovered by automated checks.

The audit returns a :class:`TraceabilityReport` carrying both gap
sets plus per-bucket coverage percentages. Render via
:func:`format_report` for the end-of-session console block.

Configuration toggle: the end-of-session caller can disable
hard-blocking via ``traceability.enforce = false`` in
``cli.json`` / ``.harness_config.json``; that switch is read in
``harness/graph.py``, not here — this module is a pure query layer.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UntracedRequirement:
    """A requirement row with no satisfying story.

    Attributes:
        req_id: The literal identifier (``FR-007``, ``US-03-02``,
            ``NFR-SEC-001``, ``CR-7``).
        kind: One of ``fr``, ``us``, ``nfr``, ``cr_synthetic`` —
            mirrors ``requirements.kind`` and lets the report
            group findings by family.
    """
    req_id: str
    kind: str


@dataclass(frozen=True)
class UntestedCriterion:
    """An acceptance criterion with no verifying test.

    Attributes:
        ac_key: The criterion identifier (``STORY-003.AC-2``) — the
            same string the test-gen ``@verifies:`` marker is
            expected to cite.
        story_key: The owning story (``STORY-003``).
        text: The criterion's text body, capped at 200 chars in
            the report so the console block stays readable.
    """
    ac_key: str
    story_key: str
    text: str


@dataclass(frozen=True)
class TraceabilityReport:
    """Aggregate result of one SQL traceability audit pass.

    Attributes:
        spec_path: Workspace-relative path of the spec the
            requirements_ingest parsed (informational; the audit
            itself queries the DB, not the file).
        total_reqs: Count of ``requirements`` rows for this workspace.
        traced_reqs: Subset that has at least one row in
            ``story_satisfies_req`` (i.e. at least one story
            satisfies the requirement).
        untraced: List of :class:`UntracedRequirement` records —
            the requirement-side gaps to surface to the operator.
        total_acs: Count of ``acceptance_criteria`` rows for this
            workspace.
        verified_acs: Subset with at least one row in
            ``test_verifies_ac`` (a generated test claims to verify
            this AC and that test passed its sandbox run).
        untested_acs: List of :class:`UntestedCriterion` records —
            the AC-side gaps.
    """
    spec_path: str
    total_reqs: int
    traced_reqs: int
    untraced: list[UntracedRequirement]
    total_acs: int = 0
    verified_acs: int = 0
    untested_acs: list[UntestedCriterion] = field(default_factory=list)

    @property
    def req_coverage_pct(self) -> float:
        """Percentage of declared requirements with at least one
        satisfying story. ``100.0`` when no requirements exist
        (vacuously complete — pre-v5 workspace or empty spec)."""
        if self.total_reqs == 0:
            return 100.0
        return 100.0 * self.traced_reqs / self.total_reqs

    @property
    def ac_coverage_pct(self) -> float:
        """Percentage of acceptance criteria with at least one
        verifying test. ``100.0`` when no ACs exist."""
        if self.total_acs == 0:
            return 100.0
        return 100.0 * self.verified_acs / self.total_acs

    # Backward-compat alias for the pre-v5 single coverage metric.
    # Old call sites that referenced ``coverage_pct`` (and the
    # ``total_ids`` / ``traced_ids`` fields) get the requirement-
    # coverage view, which is the closest 1:1 match for what the
    # text-grep audit measured.
    @property
    def coverage_pct(self) -> float:
        return self.req_coverage_pct

    @property
    def total_ids(self) -> int:
        return self.total_reqs

    @property
    def traced_ids(self) -> int:
        return self.traced_reqs

    def has_failures(self) -> bool:
        """True when either gap set is non-empty. Retained for callers
        that report both gap classes together; the end-of-session gate
        splits them via :meth:`has_req_gap` / :meth:`has_ac_gap`.
        """
        return bool(self.untraced or self.untested_acs)

    def has_req_gap(self) -> bool:
        """True when at least one requirement lacks a satisfying story.

        Gated in every flow (build / patch / test) — an untraced
        requirement means the story-planner never covered it, and no
        downstream ``teane test`` pass can close the gap.
        """
        return bool(self.untraced)

    def has_ac_gap(self) -> bool:
        """True when at least one acceptance criterion lacks a linked
        test in ``test_verifies_ac``.

        AC coverage is closed by acceptance / Playwright tests written
        by ``teane test`` — build / patch generate unit tests (linked
        to code modules), not AC-scoped e2e tests. So this gap is only
        enforced when ``flow == "test"``. Blocking build / patch on it
        creates an unfixable auto-resume loop (finsearch session
        156032347 hit this: 25/124 ACs untested at end-of-build,
        headless resume ping-pongs through ``traceability_node`` with
        no way to add markers).
        """
        return bool(self.untested_acs)


def audit_workspace(
    workspace_path: str,
    *,
    spec_relpath: str = "docs/SPEC_REQUIREMENTS.md",
) -> Optional[TraceabilityReport]:
    """Run the SQL traceability audit for ``workspace_path``.

    Args:
        workspace_path: Project root. The workspace's basename is
            used to scope ``state.db`` rows (the standard teane
            convention — see ``story_state.app_name_for_workspace``).
        spec_relpath: Recorded in the report for display; the
            audit itself joins on the ``requirements`` table and
            does NOT re-read the spec file.

    Returns:
        A :class:`TraceabilityReport`, or ``None`` when the
        workspace path is invalid or the audit could not open the
        state DB. An empty workspace (no requirements, no ACs)
        returns a report with both ``has_failures()`` and the gap
        lists empty — vacuously clean.
    """
    if not workspace_path or not os.path.isdir(workspace_path):
        return None
    try:
        from harness import story_state
        app_name = story_state.app_name_for_workspace(workspace_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[traceability] audit skipped (workspace=%r): %s",
                       workspace_path, exc)
        return None

    try:
        conn = story_state.open_story_db()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[traceability] audit skipped (open_story_db failed): %s", exc,
        )
        return None
    try:
        all_reqs = story_state.list_requirements(conn, app_name)
        untraced_rows = story_state.requirements_without_satisfying_story(
            conn, app_name,
        )
        untraced = [
            UntracedRequirement(req_id=r["req_key"], kind=r["kind"])
            for r in untraced_rows
        ]
        total_acs = conn.execute(
            "SELECT COUNT(*) FROM acceptance_criteria WHERE workspace = ?",
            (app_name,),
        ).fetchone()[0]
        untested_rows = story_state.acs_without_verifying_test(conn, app_name)
        untested = [
            UntestedCriterion(
                ac_key=r["ac_key"],
                story_key=r["story_key"],
                text=(r["text"] or "")[:200],
            )
            for r in untested_rows
        ]
    finally:
        conn.close()

    total_reqs = len(all_reqs)
    return TraceabilityReport(
        spec_path=spec_relpath,
        total_reqs=total_reqs,
        traced_reqs=total_reqs - len(untraced),
        untraced=untraced,
        total_acs=int(total_acs),
        verified_acs=int(total_acs) - len(untested),
        untested_acs=untested,
    )


def format_report(report: TraceabilityReport) -> str:
    """Render the report as a human-readable Markdown block.

    Returns the empty string when both gap sets are empty (saves
    the operator from a noisy "everything is fine" line in the
    end-of-session output).
    """
    if not report.has_failures():
        return ""

    lines: list[str] = [
        "## Requirement & Acceptance-Criterion Traceability Audit",
        f"_Spec: {report.spec_path}_",
        "",
        (
            f"- **Requirements**: {report.traced_reqs}/{report.total_reqs} "
            f"with a satisfying story ({report.req_coverage_pct:.0f}% coverage)."
        ),
        (
            f"- **Acceptance criteria**: {report.verified_acs}/{report.total_acs} "
            f"with a verifying test ({report.ac_coverage_pct:.0f}% coverage)."
        ),
        "",
    ]

    if report.untraced:
        lines.append(
            f"### Untraced requirements ({len(report.untraced)})"
        )
        lines.append(
            "These requirement IDs are declared in the spec but no "
            "story satisfies them — codegen has not covered the "
            "intent. Add a story that cites the requirement_key "
            "or revise the spec."
        )
        by_kind: dict[str, list[str]] = {}
        for item in report.untraced:
            by_kind.setdefault(item.kind, []).append(item.req_id)
        kind_labels = {
            "fr": "Functional Requirements",
            "us": "User Stories",
            "nfr": "Non-Functional Requirements",
            "cr_synthetic": "Change Requests",
        }
        for kind in ("fr", "us", "nfr", "cr_synthetic"):
            ids = by_kind.get(kind)
            if not ids:
                continue
            lines.append(f"\n#### {kind_labels.get(kind, kind)}")
            for req_id in sorted(ids):
                lines.append(f"- `{req_id}`")

    if report.untested_acs:
        lines.append("")
        lines.append(
            f"### Untested acceptance criteria ({len(report.untested_acs)})"
        )
        lines.append(
            "These acceptance criteria have no test that cites them "
            "via a `# @verifies: STORY-N.AC-N` marker. Either the "
            "test was never generated, the marker is missing/malformed, "
            "or the generated test failed and its link was dropped."
        )
        by_story: dict[str, list[UntestedCriterion]] = {}
        for ac in report.untested_acs:
            by_story.setdefault(ac.story_key, []).append(ac)
        for story_key in sorted(by_story):
            lines.append(f"\n#### {story_key}")
            for ac in by_story[story_key]:
                snippet = ac.text or "(no text)"
                lines.append(f"- `{ac.ac_key}` — {snippet}")

    return "\n".join(lines) + "\n"
