"""V5-traceability and test-result views for the operator dashboard.

These renderers consume read-only data from the global
``~/.harness/state.db`` (v4/v5 schema in :mod:`harness.story_state`)
and the per-session JSONL log to surface:

- **Feature/story panel**: the stories the run worked on, grouped by
  feature, with build_kind / cr_ids badges per row.
- **Batch panel**: per-session batches with their commit SHA and
  build_kind, tying a ``teane build``/``patch`` session to the
  feature-first decomposition rows it produced.
- **Test-results panel**: the verdict (``passed``/``failed``/
  ``cluster_count``) plus links to each CR-DEFECT-* directory's
  attachments, served via the ``/api/workspace-file`` route in
  :mod:`harness.dashboard_workspace`.
- **Traceability panel**: per-workspace coverage gauges plus the
  untraced / untested lists from :func:`harness.traceability.audit_workspace`.

Nothing here writes; the dashboard's POST handlers stay in
:mod:`harness.dashboard`. The renderers take a workspace path or a
session_id and return an HTML fragment ready to drop inside a card.
"""

from __future__ import annotations

import html
import json
import logging
import os
import sqlite3
from typing import Any, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _safe_json_line(line: str) -> Optional[dict[str, Any]]:
    try:
        return json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None


def resolve_workspace_from_log(log_path: str) -> str:
    """Read the first ``session_start`` event from ``log_path`` and
    return its ``workspace_path`` field. Empty string when the log
    doesn't exist, has no session_start, or isn't a valid JSONL stream.

    The dashboard's per-session detail view receives a session_id, not a
    workspace path; this helper bridges the two using the same event
    the sessions-list renderer already inspects.
    """
    if not log_path or not os.path.isfile(log_path):
        return ""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                evt = _safe_json_line(line)
                if not evt:
                    continue
                if evt.get("event") == "session_start":
                    return str(evt.get("workspace_path") or "")
    except OSError:
        return ""
    return ""


def _build_kind_badge(build_kind: str) -> str:
    """One-line badge for ``stories.build_kind`` / ``batches.build_kind``.

    Green for greenfield (the canonical first run), blue for CR-driven
    (incremental patch addressing a change_request).
    """
    kind = (build_kind or "").strip().lower()
    if kind == "greenfield":
        return "<span class='bx--tag bx--tag--green'>greenfield</span>"
    if kind == "cr":
        return "<span class='bx--tag bx--tag--blue'>cr</span>"
    return f"<span class='bx--tag'>{_esc(kind or '—')}</span>"


def _cr_ids_badge(cr_ids: Any) -> str:
    """Render a ``[CR-3, CR-7]`` chip row for the cr_ids JSON list.

    Empty list → empty string (no chip clutter on greenfield rows).
    """
    if not cr_ids:
        return ""
    try:
        items = [int(c) for c in cr_ids]
    except (TypeError, ValueError):
        return ""
    if not items:
        return ""
    return " ".join(
        f"<span class='bx--tag bx--tag--purple'>CR-{c}</span>" for c in items
    )


def _open_state_db_readonly() -> Optional[sqlite3.Connection]:
    """Open ``~/.harness/state.db`` read-only for dashboard queries.

    Returns None when the file doesn't exist (e.g. waterfall workspace
    that never engaged Agile mode). The dashboard handles that by
    rendering an empty-state card rather than 500-ing.
    """
    try:
        from harness.story_state import state_db_path  # local import; story_state is heavy
    except ImportError:
        return None
    db_path = state_db_path()
    if not os.path.isfile(db_path):
        return None
    try:
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
            timeout=2.0,
            isolation_level=None,
        )
        conn.execute("PRAGMA busy_timeout = 1000")
        return conn
    except sqlite3.Error as exc:
        logger.warning("[v5views] state.db open failed: %s", exc)
        return None


def _list_batches_for_session(
    conn: sqlite3.Connection, workspace: str, session_id: str,
) -> list[dict[str, Any]]:
    """All batches in ``workspace`` whose ``session_id`` matches.

    The ``batches`` table doesn't have a dedicated helper for this in
    story_state.py — every existing helper filters by cr_id or
    batch_id. We pull what we need via a small inline SELECT against
    the documented column layout.
    """
    rows = conn.execute(
        "SELECT id, session_id, feature_id, started_at, completed_at, "
        "status, committed_sha, build_kind, cr_ids "
        "FROM batches WHERE workspace = ? AND session_id = ? "
        "ORDER BY id",
        (workspace, session_id),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            cr_list = json.loads(r[8]) if r[8] else []
        except (json.JSONDecodeError, TypeError):
            cr_list = []
        out.append({
            "id": r[0],
            "session_id": r[1],
            "feature_id": r[2],
            "started_at": r[3],
            "completed_at": r[4],
            "status": r[5],
            "committed_sha": r[6],
            "build_kind": r[7] or "",
            "cr_ids": cr_list,
        })
    return out


def render_session_features_card(workspace_path: str) -> str:
    """Render the "Features & stories" card for the session detail page.

    Groups stories by feature; each feature lists its stories with
    status / build_kind / cr_ids badges. Empty state when the
    workspace has no rows in state.db (waterfall workspaces, or
    sessions older than the v4 decomposition rollout).
    """
    if not workspace_path:
        return ""
    try:
        from harness.story_state import (
            app_name_for_workspace, list_features, list_stories,
        )
    except ImportError:
        return ""
    conn = _open_state_db_readonly()
    if conn is None:
        return ""
    try:
        app = app_name_for_workspace(workspace_path)
        features = list_features(conn, app)
        stories = list_stories(conn, app)
    except sqlite3.Error as exc:
        conn.close()
        logger.warning("[v5views] features/stories read failed: %s", exc)
        return ""
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    if not stories:
        return (
            "<div class='card'>"
            "<h2>Features &amp; stories</h2>"
            "<p class='muted'>This workspace has no v4/v5 decomposition rows. "
            "Either it's a waterfall run, or the build pre-dates the "
            "feature-first decomposition rollout.</p>"
            "</div>"
        )

    # Bucket stories by feature_id; preserve feature insertion order.
    by_feature: dict[Optional[int], list[dict[str, Any]]] = {}
    for s in stories:
        by_feature.setdefault(s.get("feature_id"), []).append(s)

    body_parts: list[str] = []
    seen_features: set[Optional[int]] = set()
    for feat in features:
        fid = feat["id"]
        feat_stories = by_feature.get(fid, [])
        if not feat_stories:
            continue
        seen_features.add(fid)
        body_parts.append(_render_feature_block(feat, feat_stories))
    # Orphan stories (feature_id NULL or feature row missing) — render
    # under a sentinel "Unassigned" group so they don't silently vanish.
    orphans: list[dict[str, Any]] = []
    for fid, batch in by_feature.items():
        if fid in seen_features:
            continue
        orphans.extend(batch)
    if orphans:
        body_parts.append(_render_feature_block(
            {"feature_key": "(unassigned)", "name": "Unassigned"}, orphans,
        ))

    return (
        "<div class='card'>"
        "<h2>Features &amp; stories</h2>"
        + "".join(body_parts)
        + "</div>"
    )


def _render_feature_block(
    feature: dict[str, Any], stories: list[dict[str, Any]],
) -> str:
    feature_key = feature.get("feature_key") or "(unknown)"
    feature_name = feature.get("name") or feature_key
    header = (
        f"<h3>{_esc(feature_name)} "
        f"<code class='muted fs-sm'>{_esc(feature_key)}</code></h3>"
    )
    rows = []
    for s in stories:
        status = s.get("status") or ""
        rows.append(
            "<tr>"
            f"<td><code>{_esc(s.get('story_key'))}</code></td>"
            f"<td>{_esc(s.get('title') or '')}</td>"
            f"<td><span class='bx--tag'>{_esc(status)}</span></td>"
            f"<td>{_build_kind_badge(str(s.get('build_kind') or ''))}</td>"
            f"<td>{_cr_ids_badge(s.get('cr_ids'))}</td>"
            "</tr>"
        )
    table = (
        "<div class='table-wrap'><table class='w-100'>"
        "<thead><tr><th>Story</th><th>Title</th><th>Status</th>"
        "<th>Build kind</th><th>CRs</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )
    return header + table


def render_session_batches_card(
    workspace_path: str, session_id: str,
) -> str:
    """Render the "Batches" card scoped to the given session.

    Each row carries the batch id, status, build_kind, commit SHA, and
    cr_ids — so an operator can trace a single ``teane build``/``patch``
    session to the rows of the feature-first decomposition.
    """
    if not workspace_path or not session_id:
        return ""
    try:
        from harness.story_state import app_name_for_workspace
    except ImportError:
        return ""
    conn = _open_state_db_readonly()
    if conn is None:
        return ""
    try:
        app = app_name_for_workspace(workspace_path)
        batches = _list_batches_for_session(conn, app, session_id)
    except sqlite3.Error as exc:
        conn.close()
        logger.warning("[v5views] batches read failed: %s", exc)
        return ""
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    if not batches:
        # Waterfall sessions don't emit batches; render nothing so the
        # session-detail page doesn't accumulate empty cards.
        return ""

    rows = []
    for b in batches:
        sha = (b.get("committed_sha") or "")[:10]
        rows.append(
            "<tr>"
            f"<td>{int(b['id'])}</td>"
            f"<td>{_esc(b.get('status') or '')}</td>"
            f"<td>{_build_kind_badge(str(b.get('build_kind') or ''))}</td>"
            f"<td><code class='muted'>{_esc(sha) if sha else '—'}</code></td>"
            f"<td>{_cr_ids_badge(b.get('cr_ids'))}</td>"
            f"<td>{_esc(b.get('started_at') or '')}</td>"
            f"<td>{_esc(b.get('completed_at') or '—')}</td>"
            "</tr>"
        )
    return (
        "<div class='card'><h2>Batches</h2>"
        "<div class='table-wrap'><table class='w-100'>"
        "<thead><tr><th>ID</th><th>Status</th><th>Build kind</th>"
        "<th>Commit</th><th>CRs</th><th>Started</th><th>Completed</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div></div>"
    )


def _read_last_node_state_test(log_path: str) -> Optional[dict[str, Any]]:
    """Walk ``log_path`` for the most recent event whose ``node_state.test``
    is non-empty. Returns the test dict (per ``test_target.py`` shape) or
    None when no test event is present.
    """
    if not log_path or not os.path.isfile(log_path):
        return None
    found: Optional[dict[str, Any]] = None
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                evt = _safe_json_line(line)
                if not evt:
                    continue
                ns = evt.get("node_state") or {}
                if not isinstance(ns, dict):
                    continue
                test_dict = ns.get("test")
                if isinstance(test_dict, dict) and test_dict:
                    found = test_dict
    except OSError:
        return None
    return found


def render_session_test_results_card(
    workspace_path: str, log_path: str,
) -> str:
    """Render the "Test results" card from the run's last
    ``node_state.test`` event.

    For ``skipped=True`` + ``reason=prereq_failed`` we render an
    empty-state with a link to ``/run/deploy`` — that's the typical
    case where ``teane test`` exited because no clean prior deploy
    exists.

    Successful runs render the verdict plus per-CR-DEFECT rows
    linking to the screenshot / trace / DOM / cluster-evidence
    attachments via the ``/api/workspace-file`` route.
    """
    test = _read_last_node_state_test(log_path)
    if not test:
        return ""

    skipped = bool(test.get("skipped"))
    reason = str(test.get("reason") or "")
    if skipped:
        if reason == "prereq_failed":
            detail = _esc(test.get("detail") or "test prerequisites not met")
            return (
                "<div class='card'><h2>Test results</h2>"
                f"<p>Test run was skipped: {detail}.</p>"
                "<p><a href='/run/deploy' class='bx--btn bx--btn--tertiary'>"
                "Go to deploy</a></p></div>"
            )
        return (
            "<div class='card'><h2>Test results</h2>"
            f"<p class='muted'>Skipped: {_esc(reason or 'unknown')}</p>"
            "</div>"
        )

    passed = int(test.get("passed") or 0)
    failed = int(test.get("failed") or 0)
    clusters = int(test.get("cluster_count") or 0)
    scope = str(test.get("scope") or "—")
    base_url = str(test.get("base_url") or "")
    cr_paths = test.get("cr_paths") or []
    verdict_tag = (
        "<span class='bx--tag bx--tag--green'>green</span>"
        if failed == 0 else
        "<span class='bx--tag bx--tag--red'>red</span>"
    )
    summary = (
        f"<p>{verdict_tag} "
        f"Passed: <strong>{passed}</strong>, "
        f"Failed: <strong>{failed}</strong>, "
        f"Clusters: <strong>{clusters}</strong> "
        f"(scope: <code>{_esc(scope)}</code>"
        + (f", base: <code>{_esc(base_url)}</code>" if base_url else "")
        + ")</p>"
    )

    defects_block = ""
    if cr_paths and workspace_path:
        rows = []
        for cr_path in cr_paths:
            cr_path_str = str(cr_path)
            try:
                relpath = os.path.relpath(cr_path_str, workspace_path)
            except ValueError:
                # Different drive letters on Windows etc — fall back to
                # the basename so the link still works.
                relpath = os.path.basename(cr_path_str)
            ws_q = quote(workspace_path, safe="")
            # Render one row per CR-DEFECT directory with quick links
            # to each attachment basename. The dashboard_workspace
            # /api/workspace-file route enforces the basename allowlist.
            links = []
            for name in (
                "narrative.txt", "screenshot.png", "trace.zip",
                "cluster_evidence.json", "dom.html", "source_spec.md",
            ):
                file_relpath = quote(
                    os.path.join(relpath, name), safe="/",
                )
                links.append(
                    f"<a href='/api/workspace-file?workspace={ws_q}"
                    f"&relpath={file_relpath}'>{name}</a>"
                )
            rows.append(
                "<tr>"
                f"<td><code>{_esc(os.path.basename(cr_path_str.rstrip('/')))}</code></td>"
                f"<td>{' · '.join(links)}</td>"
                "</tr>"
            )
        defects_block = (
            "<h3>Defect bundles</h3>"
            "<div class='table-wrap'><table class='w-100'>"
            "<thead><tr><th>CR-DEFECT</th><th>Attachments</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table></div>"
        )

    return (
        "<div class='card'><h2>Test results</h2>"
        + summary + defects_block +
        "</div>"
    )


def render_traceability_card(workspace_path: str) -> str:
    """Render the "Traceability" card for ``workspace_path``.

    Calls :func:`harness.traceability.audit_workspace` and surfaces the
    two coverage percentages + the untraced-requirements /
    untested-acceptance-criteria lists. Empty state when:

    - The workspace path is missing / invalid.
    - The audit can't run (no state.db, no SPEC_REQUIREMENTS.md).
    - ``total_reqs`` is 0 (rendering 100% would mislead — we say "no
      requirements declared" instead).
    """
    if not workspace_path:
        return ""
    try:
        from harness.traceability import audit_workspace
    except ImportError:
        return ""
    try:
        report = audit_workspace(workspace_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[v5views] traceability audit failed: %s", exc)
        return ""
    if report is None:
        return (
            "<div class='card'><h2>Traceability</h2>"
            "<p class='muted'>Traceability audit unavailable — workspace "
            "missing, pre-v5, or SPEC_REQUIREMENTS.md not yet generated.</p>"
            "</div>"
        )

    if report.total_reqs == 0:
        return (
            "<div class='card'><h2>Traceability</h2>"
            "<p class='muted'>No requirements declared for this workspace yet.</p>"
            "</div>"
        )

    req_pct = report.req_coverage_pct
    ac_pct = report.ac_coverage_pct
    gauges = (
        "<div class='trace-gauges'>"
        f"<div><strong>Requirements</strong>: {report.traced_reqs}"
        f"/{report.total_reqs} <span class='bx--tag bx--tag--green'>"
        f"{req_pct:.0f}%</span></div>"
        f"<div><strong>Acceptance criteria</strong>: {report.verified_acs}"
        f"/{report.total_acs} <span class='bx--tag bx--tag--green'>"
        f"{ac_pct:.0f}%</span></div>"
        "</div>"
    )

    untraced_html = ""
    if report.untraced:
        items = "".join(
            f"<li><code>{_esc(u.req_id)}</code> "
            f"<span class='muted'>({_esc(u.kind)})</span></li>"
            for u in report.untraced
        )
        untraced_html = (
            "<details><summary><strong>Untraced requirements "
            f"({len(report.untraced)})</strong></summary>"
            f"<ul>{items}</ul></details>"
        )

    untested_html = ""
    if report.untested_acs:
        items = "".join(
            f"<li><code>{_esc(u.ac_key)}</code> "
            f"<span class='muted'>({_esc(u.story_key)})</span>"
            f": {_esc(u.text)}</li>"
            for u in report.untested_acs
        )
        untested_html = (
            "<details><summary><strong>Untested acceptance criteria "
            f"({len(report.untested_acs)})</strong></summary>"
            f"<ul>{items}</ul></details>"
        )

    return (
        "<div class='card'><h2>Traceability</h2>"
        + gauges + untraced_html + untested_html +
        "</div>"
    )


# ---------------------------------------------------------------------------
# Stories browser — cross-session workspace-scoped views
# ---------------------------------------------------------------------------
# Two entry points:
#
# - :func:`render_stories_index_page` — one row per workspace with story
#   data, rolling up feature/story/defect counts and last activity.
# - :func:`render_stories_workspace_page` — for one workspace: features
#   with per-status rollups, recent batches, open+closed defects.
#
# These read the same ``~/.harness/state.db`` the session-detail cards
# already query; the difference is scope. The session cards answer
# "what did THIS run touch"; the stories page answers "what does this
# workspace look like, across every run".


_STORY_KEY_URL_RE = "[A-Za-z0-9_.\\-]+"
"""Regex fragment reused in the dashboard route table so the stories
routes accept the same workspace-basename shape ``app_name_for_workspace``
already produces."""


def _list_workspaces_with_story_data(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """One row per workspace that has any features/stories/batches/defects.

    Rollup counts come from a single UNION-ALL scan so the query works
    even when a workspace has features but no stories yet (or a defect
    orphaned by a deleted story). Empty state.db → empty list.
    """
    sql = """
    WITH ws AS (
        SELECT workspace FROM features
        UNION SELECT workspace FROM stories
        UNION SELECT workspace FROM batches
        UNION SELECT workspace FROM defects
    )
    SELECT
        ws.workspace,
        (SELECT COUNT(*) FROM features f WHERE f.workspace = ws.workspace)   AS features,
        (SELECT COUNT(*) FROM stories  s WHERE s.workspace = ws.workspace)   AS stories,
        (SELECT COUNT(*) FROM stories  s WHERE s.workspace = ws.workspace AND s.status = 'done')        AS done,
        (SELECT COUNT(*) FROM stories  s WHERE s.workspace = ws.workspace AND s.status = 'in_progress') AS in_progress,
        (SELECT COUNT(*) FROM stories  s WHERE s.workspace = ws.workspace AND s.status = 'planned')     AS planned,
        (SELECT COUNT(*) FROM stories  s WHERE s.workspace = ws.workspace AND s.status = 'blocked')     AS blocked,
        (SELECT COUNT(*) FROM defects  d WHERE d.workspace = ws.workspace AND d.status = 'open')        AS open_defects,
        (SELECT COUNT(*) FROM batches  b WHERE b.workspace = ws.workspace)   AS batches,
        (SELECT MAX(COALESCE(s.completed_at, s.started_at, s.created_at))
           FROM stories s WHERE s.workspace = ws.workspace)                  AS last_story_activity,
        (SELECT MAX(COALESCE(b.completed_at, b.started_at))
           FROM batches b WHERE b.workspace = ws.workspace)                  AS last_batch_activity
    FROM ws
    ORDER BY COALESCE(last_batch_activity, last_story_activity, '') DESC, ws.workspace
    """
    rows = conn.execute(sql).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({
            "workspace": r[0],
            "features": int(r[1] or 0),
            "stories": int(r[2] or 0),
            "done": int(r[3] or 0),
            "in_progress": int(r[4] or 0),
            "planned": int(r[5] or 0),
            "blocked": int(r[6] or 0),
            "open_defects": int(r[7] or 0),
            "batches": int(r[8] or 0),
            "last_activity": r[9] or r[10] or "",
        })
    return out


def _feature_rollups(
    conn: sqlite3.Connection, workspace: str,
) -> list[dict[str, Any]]:
    """One row per feature with per-status story counts.

    A LEFT JOIN keeps features that have no stories yet — rendering them
    as empty rather than dropping them, so the operator can see the
    planned decomposition even before any batch has run.
    """
    sql = """
    SELECT
        f.id, f.feature_key, f.name,
        COUNT(s.id)                                    AS total,
        SUM(CASE WHEN s.status='done'        THEN 1 ELSE 0 END) AS done,
        SUM(CASE WHEN s.status='in_progress' THEN 1 ELSE 0 END) AS in_progress,
        SUM(CASE WHEN s.status='planned'     THEN 1 ELSE 0 END) AS planned,
        SUM(CASE WHEN s.status='blocked'     THEN 1 ELSE 0 END) AS blocked
    FROM features f
    LEFT JOIN stories s
        ON s.workspace = f.workspace AND s.feature_id = f.id
    WHERE f.workspace = ?
    GROUP BY f.id, f.feature_key, f.name
    ORDER BY f.id
    """
    rows = conn.execute(sql, (workspace,)).fetchall()
    return [
        {
            "id": r[0], "feature_key": r[1], "name": r[2],
            "total": int(r[3] or 0),
            "done": int(r[4] or 0),
            "in_progress": int(r[5] or 0),
            "planned": int(r[6] or 0),
            "blocked": int(r[7] or 0),
        }
        for r in rows
    ]


def _list_recent_batches(
    conn: sqlite3.Connection, workspace: str, limit: int = 50,
) -> list[dict[str, Any]]:
    """Last ``limit`` batches for the workspace, newest first, joined
    with feature_key so the row is readable without a second lookup."""
    rows = conn.execute(
        "SELECT b.id, b.session_id, b.started_at, b.completed_at, b.status, "
        "b.committed_sha, b.build_kind, b.cr_ids, "
        "f.feature_key, "
        "(SELECT COUNT(*) FROM batch_stories bs WHERE bs.batch_id = b.id) AS story_count "
        "FROM batches b "
        "LEFT JOIN features f ON f.id = b.feature_id "
        "WHERE b.workspace = ? "
        "ORDER BY b.id DESC LIMIT ?",
        (workspace, int(limit)),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            cr_list = json.loads(r[7]) if r[7] else []
        except (json.JSONDecodeError, TypeError):
            cr_list = []
        out.append({
            "id": r[0],
            "session_id": r[1],
            "started_at": r[2],
            "completed_at": r[3],
            "status": r[4],
            "committed_sha": r[5],
            "build_kind": r[6] or "",
            "cr_ids": cr_list,
            "feature_key": r[8],
            "story_count": int(r[9] or 0),
        })
    return out


def _list_defects(
    conn: sqlite3.Connection, workspace: str, *, only_open: bool,
) -> list[dict[str, Any]]:
    """Defects joined with the offending story's key/title.

    ``only_open=True`` scopes to ``status='open'`` — the default view.
    ``False`` returns every defect, resolved rows included, so operators
    can review closures in the expanded panel.
    """
    where = "d.workspace = ?"
    params: list[Any] = [workspace]
    if only_open:
        where += " AND d.status = 'open'"
    rows = conn.execute(
        "SELECT d.id, d.session_id, d.severity, d.summary, d.status, "
        "d.created_at, d.resolved_at, s.story_key, s.title "
        "FROM defects d "
        "LEFT JOIN stories s ON s.id = d.story_id "
        f"WHERE {where} "
        "ORDER BY (d.status='open') DESC, d.created_at DESC",
        tuple(params),
    ).fetchall()
    return [
        {
            "id": r[0], "session_id": r[1], "severity": r[2], "summary": r[3],
            "status": r[4], "created_at": r[5], "resolved_at": r[6],
            "story_key": r[7], "title": r[8],
        }
        for r in rows
    ]


def render_stories_index_page(cfg: Any = None) -> str:
    """Full-page HTML for ``GET /stories``.

    Lists every workspace that has story data, with rollup counts and a
    click-through to the workspace detail page. Empty state.db renders
    as an "no story data yet" card so the route never 500s.
    """
    conn = _open_state_db_readonly()
    if conn is None:
        return _stories_empty_state_index()
    try:
        rows = _list_workspaces_with_story_data(conn)
    except sqlite3.Error as exc:
        logger.warning("[v5views] stories index read failed: %s", exc)
        rows = []
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    if not rows:
        return _stories_empty_state_index()

    body_rows = []
    for r in rows:
        ws = r["workspace"]
        ws_link = f"<a href='/stories/{_esc(ws)}'><code>{_esc(ws)}</code></a>"
        defect_cell = (
            f"<span class='bx--tag bx--tag--red'>{r['open_defects']}</span>"
            if r["open_defects"] else
            "<span class='muted'>0</span>"
        )
        body_rows.append(
            "<tr>"
            f"<td>{ws_link}</td>"
            f"<td>{r['features']}</td>"
            f"<td>{r['done']} / {r['stories']}</td>"
            f"<td>{r['in_progress']}</td>"
            f"<td>{r['planned']}</td>"
            f"<td>{r['blocked']}</td>"
            f"<td>{defect_cell}</td>"
            f"<td>{r['batches']}</td>"
            f"<td><span class='muted'>{_esc(r['last_activity'] or '—')}</span></td>"
            "</tr>"
        )
    table = (
        "<div class='table-wrap'><table class='w-100'>"
        "<thead><tr>"
        "<th>Workspace</th><th>Features</th><th>Done / total</th>"
        "<th>In progress</th><th>Planned</th><th>Blocked</th>"
        "<th>Open defects</th><th>Batches</th><th>Last activity</th>"
        "</tr></thead>"
        "<tbody>" + "".join(body_rows) + "</tbody></table></div>"
    )
    return (
        "<div class='card'><h2>Stories &amp; coverage</h2>"
        "<p class='muted'>Every workspace with feature/story/batch/defect "
        "rows in <code>~/.harness/state.db</code>. Click a workspace to "
        "see its features, batches and defect list.</p>"
        + table +
        "</div>"
    )


def _stories_empty_state_index() -> str:
    return (
        "<div class='card'><h2>Stories &amp; coverage</h2>"
        "<p class='muted'>No story data yet. This page populates as "
        "<code>teane build</code> / <code>teane patch</code> / "
        "<code>teane test</code> runs create features, stories, batches "
        "and defects in <code>~/.harness/state.db</code>.</p>"
        "</div>"
    )


def render_stories_workspace_page(cfg: Any, workspace_basename: str) -> str:
    """Full-page HTML for ``GET /stories/<workspace_basename>``.

    Renders four cards: Features (with per-feature story rollups),
    Batches (recent 50), Defects (open + closed collapsed), and a
    breadcrumb back to /stories. Unknown workspace → empty-state card
    listing what a valid workspace basename looks like.
    """
    ws = (workspace_basename or "").strip()
    breadcrumb = (
        "<ol class='breadcrumb' aria-label='Breadcrumb'>"
        "<li><a href='/stories'>Stories</a></li>"
        f"<li class='breadcrumb__current' aria-current='page'>{_esc(ws)}</li>"
        "</ol>"
    )
    if not ws:
        return breadcrumb + _stories_workspace_missing_card(ws)

    conn = _open_state_db_readonly()
    if conn is None:
        return breadcrumb + _stories_workspace_missing_card(ws)
    try:
        features = _feature_rollups(conn, ws)
        stories = []
        try:
            from harness.story_state import list_stories  # local import
            stories = list_stories(conn, ws)
        except (ImportError, sqlite3.Error) as exc:
            logger.warning("[v5views] list_stories failed for %s: %s", ws, exc)
        batches = _list_recent_batches(conn, ws)
        open_defects = _list_defects(conn, ws, only_open=True)
        all_defects = _list_defects(conn, ws, only_open=False)
    except sqlite3.Error as exc:
        logger.warning("[v5views] stories workspace read failed: %s", exc)
        return breadcrumb + _stories_workspace_missing_card(ws)
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    # If nothing exists for this workspace basename, treat it as unknown
    # rather than rendering four empty cards.
    if not features and not stories and not batches and not all_defects:
        return breadcrumb + _stories_workspace_missing_card(ws)

    stories_by_feature: dict[Optional[int], list[dict[str, Any]]] = {}
    for s in stories:
        stories_by_feature.setdefault(s.get("feature_id"), []).append(s)

    parts: list[str] = [breadcrumb]
    parts.append(_render_features_rollup_card(features, stories_by_feature))
    parts.append(_render_batches_card(batches))
    parts.append(_render_defects_card(open_defects, all_defects))
    return "".join(parts)


def _stories_workspace_missing_card(ws: str) -> str:
    return (
        "<div class='card'><h2>Workspace not found</h2>"
        f"<p class='muted'>No story data for workspace "
        f"<code>{_esc(ws) if ws else '(empty)'}</code>. The "
        "identifier is the workspace folder's basename (see "
        "<code>app_name_for_workspace</code>).</p>"
        "<p><a href='/stories' class='bx--btn bx--btn--tertiary'>"
        "Back to Stories index</a></p></div>"
    )


def _render_features_rollup_card(
    features: list[dict[str, Any]],
    stories_by_feature: dict[Optional[int], list[dict[str, Any]]],
) -> str:
    if not features:
        return (
            "<div class='card'><h2>Features</h2>"
            "<p class='muted'>No feature rows yet for this workspace.</p>"
            "</div>"
        )
    blocks: list[str] = []
    for f in features:
        fid = f["id"]
        f_key = f.get("feature_key") or "(unknown)"
        f_name = f.get("name") or f_key
        rollup = (
            f"<span class='bx--tag bx--tag--green'>{f['done']} done</span> "
            f"<span class='bx--tag bx--tag--blue'>{f['in_progress']} in flight</span> "
            f"<span class='bx--tag'>{f['planned']} planned</span> "
            + (
                f"<span class='bx--tag bx--tag--red'>{f['blocked']} blocked</span>"
                if f["blocked"] else ""
            )
        )
        summary = (
            f"<summary><strong>{_esc(f_name)}</strong> "
            f"<code class='muted fs-sm'>{_esc(f_key)}</code> "
            f"<span class='muted'>· {f['total']} stories</span> "
            f"{rollup}</summary>"
        )
        story_rows = stories_by_feature.get(fid, [])
        if story_rows:
            rows = []
            for s in story_rows:
                rows.append(
                    "<tr>"
                    f"<td><code>{_esc(s.get('story_key'))}</code></td>"
                    f"<td>{_esc(s.get('title') or '')}</td>"
                    f"<td><span class='bx--tag'>{_esc(s.get('status') or '')}</span></td>"
                    f"<td>{_build_kind_badge(str(s.get('build_kind') or ''))}</td>"
                    f"<td>{_cr_ids_badge(s.get('cr_ids'))}</td>"
                    "</tr>"
                )
            table = (
                "<div class='table-wrap'><table class='w-100'>"
                "<thead><tr><th>Story</th><th>Title</th><th>Status</th>"
                "<th>Build kind</th><th>CRs</th></tr></thead>"
                "<tbody>" + "".join(rows) + "</tbody></table></div>"
            )
        else:
            table = "<p class='muted'>No stories yet.</p>"
        blocks.append(f"<details open>{summary}{table}</details>")
    return "<div class='card'><h2>Features</h2>" + "".join(blocks) + "</div>"


def _render_batches_card(batches: list[dict[str, Any]]) -> str:
    if not batches:
        return (
            "<div class='card'><h2>Batches</h2>"
            "<p class='muted'>No batches recorded for this workspace yet.</p>"
            "</div>"
        )
    rows = []
    for b in batches:
        sha = (b.get("committed_sha") or "")[:10]
        sid = b.get("session_id") or ""
        session_cell = (
            f"<a href='/sessions/{_esc(sid)}'><code>{_esc(sid)}</code></a>"
            if sid else "<span class='muted'>—</span>"
        )
        rows.append(
            "<tr>"
            f"<td>{int(b['id'])}</td>"
            f"<td>{session_cell}</td>"
            f"<td>{_esc(b.get('feature_key') or '—')}</td>"
            f"<td>{b.get('story_count', 0)}</td>"
            f"<td>{_esc(b.get('status') or '')}</td>"
            f"<td>{_build_kind_badge(str(b.get('build_kind') or ''))}</td>"
            f"<td><code class='muted'>{_esc(sha) if sha else '—'}</code></td>"
            f"<td>{_cr_ids_badge(b.get('cr_ids'))}</td>"
            f"<td>{_esc(b.get('started_at') or '')}</td>"
            f"<td>{_esc(b.get('completed_at') or '—')}</td>"
            "</tr>"
        )
    return (
        "<div class='card'><h2>Batches</h2>"
        "<div class='table-wrap'><table class='w-100'>"
        "<thead><tr><th>ID</th><th>Session</th><th>Feature</th>"
        "<th>Stories</th><th>Status</th><th>Build kind</th>"
        "<th>Commit</th><th>CRs</th><th>Started</th><th>Completed</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div></div>"
    )


def _render_defects_card(
    open_defects: list[dict[str, Any]],
    all_defects: list[dict[str, Any]],
) -> str:
    if not all_defects:
        return (
            "<div class='card'><h2>Defects</h2>"
            "<p class='muted'>No defects recorded for this workspace.</p>"
            "</div>"
        )
    open_table = _defects_table(open_defects, empty_msg="No open defects.")
    resolved = [d for d in all_defects if d.get("status") != "open"]
    resolved_html = ""
    if resolved:
        resolved_html = (
            f"<details><summary><strong>Resolved defects "
            f"({len(resolved)})</strong></summary>"
            + _defects_table(resolved, empty_msg="")
            + "</details>"
        )
    return (
        "<div class='card'><h2>Defects</h2>"
        f"<h3>Open ({len(open_defects)})</h3>"
        + open_table + resolved_html +
        "</div>"
    )


def _defects_table(defects: list[dict[str, Any]], *, empty_msg: str) -> str:
    if not defects:
        return f"<p class='muted'>{_esc(empty_msg)}</p>" if empty_msg else ""
    rows = []
    for d in defects:
        sid = d.get("session_id") or ""
        session_cell = (
            f"<a href='/sessions/{_esc(sid)}'><code>{_esc(sid)}</code></a>"
            if sid else "<span class='muted'>—</span>"
        )
        story_cell = (
            f"<code>{_esc(d.get('story_key'))}</code>"
            if d.get("story_key") else "<span class='muted'>—</span>"
        )
        severity = str(d.get("severity") or "").lower()
        severity_class = (
            "bx--tag--red" if severity in ("critical", "high", "error")
            else "bx--tag--purple" if severity == "medium"
            else "bx--tag"
        )
        rows.append(
            "<tr>"
            f"<td>{int(d['id'])}</td>"
            f"<td><span class='bx--tag {severity_class}'>{_esc(d.get('severity') or '')}</span></td>"
            f"<td>{_esc(d.get('summary') or '')}</td>"
            f"<td>{story_cell}</td>"
            f"<td>{session_cell}</td>"
            f"<td>{_esc(d.get('created_at') or '')}</td>"
            f"<td>{_esc(d.get('resolved_at') or '—')}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap'><table class='w-100'>"
        "<thead><tr><th>ID</th><th>Severity</th><th>Summary</th>"
        "<th>Story</th><th>Session</th><th>Created</th><th>Resolved</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )
