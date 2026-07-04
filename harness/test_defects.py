"""Defect ingestion + clustering + CR emission for ``teane test``.

Phase 2 boundary: this module takes a Playwright JSON reporter blob, normalises
its failures, clusters them (so 30 scenarios failing because one backend is
wedged become 1 CR not 30), and writes each cluster to
``change_requests/CR-DEFECT-<yyyymmdd>-<slug>/`` in a layout that
``teane patch`` can ingest directly:

    change_requests/CR-DEFECT-20260630-login-button-noop/
      narrative.txt          — patcher-consumable description
      source_spec.md         — spec excerpt the failing scenario was generated from
      trace.zip              — Playwright trace (if attached)
      screenshot.png         — last screenshot (if attached)
      dom.html               — DOM at failure (if attached)
      cluster_evidence.json  — other failures in the same cluster

Scope intentionally narrow: parsing, clustering, and IO. Live execution
(Phase 5) wires this together with the playwright subprocess.

Agile vs waterfall awareness — Phases 3-4 will populate
``FailureRecord.source_spec_id`` from either ``acceptance_criteria.ac_key``
(agile) or a section anchor in SPEC_REQUIREMENTS.md (waterfall) so the
narrative can cite the right artefact. This module accepts either string
form without caring; the source-of-truth resolution lives in the writers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


# Playwright JSON-reporter top-stack-frame heuristic. Stack lines look like
# ``    at SomeFn (/path/to/login.spec.ts:42:18)`` — the first user-code
# frame (skipping node_modules / playwright internals) is the cluster key.
_STACK_LINE = re.compile(
    r"^\s*at\s+[^(]*\(?(?P<path>[^):]+):(?P<line>\d+):(?P<col>\d+)\)?\s*$"
)
_NETCALL_HINT = re.compile(
    r"(?P<method>GET|POST|PUT|DELETE|PATCH)\s+(?P<url>https?://[^\s\"']+)"
)
_INTERNAL_PATH_HINTS = ("node_modules", "playwright-core", "playwright/lib", "internal/")


# ---------------------------------------------------------------------------
# Normalised failure record
# ---------------------------------------------------------------------------


@dataclass
class FailureRecord:
    """One failing Playwright result, normalised across reporter shapes."""

    title: str
    spec_file: str
    error_message: str
    stack: str = ""
    duration_ms: int = 0
    attachments: dict[str, str] = field(default_factory=dict)
    """``{"trace": "/abs/path/trace.zip", "screenshot": "/abs/...", "dom": "/abs/..."}``"""

    source_spec_id: Optional[str] = None
    """``STORY-003.AC-2`` (agile) or section anchor (waterfall). Populated by
    Phase 4 scenario generator via inline comments in the .spec.ts file —
    parsed at execution time, not here. Optional in Phase 2."""

    def cluster_key(self) -> tuple[str, str]:
        """Conservative cluster key: (top user-code frame, failing net call).

        - Top frame: first stack line whose path is NOT in node_modules /
          Playwright internals. Falls back to error-message SHA-1 prefix
          when there's no usable frame (common for assertion-only failures).
        - Net call: any ``GET /api/foo`` or ``POST https://...`` mention in
          the error string. Empty when the failure isn't network-mediated.
        """
        frame = self._top_user_frame()
        if frame is None:
            digest = hashlib.sha1(self.error_message.encode("utf-8", "replace")).hexdigest()
            frame = f"msg:{digest[:12]}"
        netcall = self._net_call_hint()
        return frame, netcall

    def _top_user_frame(self) -> Optional[str]:
        for line in self.stack.splitlines():
            m = _STACK_LINE.match(line)
            if not m:
                continue
            path = m.group("path")
            if any(hint in path for hint in _INTERNAL_PATH_HINTS):
                continue
            base = os.path.basename(path)
            return f"{base}:{m.group('line')}"
        return None

    def _net_call_hint(self) -> str:
        m = _NETCALL_HINT.search(self.error_message) or _NETCALL_HINT.search(self.stack)
        if not m:
            return ""
        return f"{m.group('method')} {m.group('url')}"


# ---------------------------------------------------------------------------
# Playwright JSON parser
# ---------------------------------------------------------------------------


def parse_playwright_json(blob: dict[str, Any]) -> list[FailureRecord]:
    """Walk the Playwright JSON-reporter shape and return only failures.

    Tolerant of partial inputs — missing keys yield empty strings rather
    than raising, since we'd rather emit a degraded CR than swallow a real
    defect. Same forgiveness applies to attachments without paths.
    """
    failures: list[FailureRecord] = []
    suites = blob.get("suites") or []
    for suite in suites:
        for spec in _walk_specs(suite):
            for test in spec.get("tests") or []:
                for result in test.get("results") or []:
                    if result.get("status") != "failed":
                        continue
                    failures.append(_failure_from_result(spec, test, result))
    return failures


def _walk_specs(suite: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Playwright nests suites inside suites; specs live at any depth."""
    for spec in suite.get("specs") or []:
        yield spec
    for child in suite.get("suites") or []:
        yield from _walk_specs(child)


def _failure_from_result(
    spec: dict[str, Any],
    test: dict[str, Any],
    result: dict[str, Any],
) -> FailureRecord:
    err = result.get("error") or {}
    attachments: dict[str, str] = {}
    for att in result.get("attachments") or []:
        name = (att.get("name") or "").lower()
        path = att.get("path")
        if not path:
            continue
        if name == "trace":
            attachments["trace"] = path
        elif name in ("screenshot", "screenshot-fail"):
            attachments.setdefault("screenshot", path)
        elif name in ("dom", "page", "snapshot"):
            attachments.setdefault("dom", path)
    title = test.get("title") or spec.get("title") or "(untitled scenario)"
    return FailureRecord(
        title=title,
        spec_file=spec.get("file") or test.get("file") or "",
        error_message=str(err.get("message") or ""),
        stack=str(err.get("stack") or ""),
        duration_ms=int(result.get("duration") or 0),
        attachments=attachments,
        source_spec_id=test.get("annotations", [{}])[0].get("description")
            if isinstance(test.get("annotations"), list) and test.get("annotations")
            else None,
    )


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


@dataclass
class Cluster:
    """A group of failures sharing a cluster key.

    ``primary`` is the representative failure whose artefacts get
    promoted into the CR root; ``evidence`` lists the others (title +
    spec file + first stack line — enough for the patcher LLM to see
    the spread without copying every trace.zip).
    """

    key: tuple[str, str]
    primary: FailureRecord
    evidence: list[FailureRecord] = field(default_factory=list)

    def size(self) -> int:
        return 1 + len(self.evidence)


def cluster_failures(failures: list[FailureRecord]) -> list[Cluster]:
    """Group failures by ``cluster_key``.

    Within a cluster, the primary is the failure with the most attachments
    (so the CR has the richest debugging surface possible) — ties broken
    by alphabetical scenario title for determinism.
    """
    buckets: dict[tuple[str, str], list[FailureRecord]] = {}
    for f in failures:
        buckets.setdefault(f.cluster_key(), []).append(f)

    clusters: list[Cluster] = []
    for key, members in buckets.items():
        members.sort(key=lambda r: (-len(r.attachments), r.title))
        primary, *evidence = members
        clusters.append(Cluster(key=key, primary=primary, evidence=evidence))
    # Largest clusters first — patcher should see the broadest impact at the top.
    clusters.sort(key=lambda c: (-c.size(), c.primary.title))
    return clusters


# ---------------------------------------------------------------------------
# CR emission
# ---------------------------------------------------------------------------


def emit_defect_cr(
    cluster: Cluster,
    workspace_path: str,
    *,
    change_requests_dir: Optional[str] = None,
    now: Optional[datetime] = None,
) -> str:
    """Write a CR-DEFECT-* directory for ``cluster``. Returns the directory path.

    ``change_requests_dir`` defaults to ``<workspace>/change_requests``;
    callers can override for tests. ``now`` is injectable for
    deterministic slugs in tests (defaults to ``datetime.now(UTC)``).

    Side effects:
      - Creates ``<change_requests_dir>/CR-DEFECT-<yyyymmdd>-<slug>/``
      - Copies attachments from the primary failure into the directory
      - Writes narrative.txt, source_spec.md, cluster_evidence.json
    """
    if change_requests_dir is None:
        change_requests_dir = os.path.join(workspace_path, "change_requests")
    if now is None:
        now = datetime.now(timezone.utc)

    slug = _slugify(cluster.primary.title)
    base = f"CR-DEFECT-{now.strftime('%Y%m%d')}-{slug}"
    # Disambiguate same-day collisions with the cluster-key hash so two
    # distinct defects with similar titles don't overwrite each other.
    digest = hashlib.sha1(
        f"{cluster.key[0]}|{cluster.key[1]}".encode("utf-8")
    ).hexdigest()[:6]
    cr_dir = os.path.join(change_requests_dir, f"{base}-{digest}")
    os.makedirs(cr_dir, exist_ok=True)

    _write_narrative(cr_dir, cluster)
    _write_source_spec(cr_dir, cluster)
    _write_evidence(cr_dir, cluster)
    _copy_attachments(cr_dir, cluster)
    logger.info("[test_defects] Emitted %s (cluster size=%d)", cr_dir, cluster.size())
    return cr_dir


def _slugify(text: str, max_len: int = 48) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not cleaned:
        cleaned = "defect"
    return cleaned[:max_len].rstrip("-")


def _write_narrative(cr_dir: str, cluster: Cluster) -> None:
    p = cluster.primary
    frame, netcall = cluster.key
    lines = [
        "# Defect surfaced by `teane test`",
        "",
        f"**Scenario**: {p.title}",
        f"**Spec file**: {p.spec_file or '(unknown)'}",
        f"**Cluster size**: {cluster.size()} failure(s) sharing this root cause",
        f"**Top frame**: {frame}",
    ]
    if netcall:
        lines.append(f"**Failing network call**: {netcall}")
    lines.extend([
        "",
        "## Error",
        "```",
        p.error_message.strip() or "(no error message captured)",
        "```",
    ])
    if p.stack:
        lines.extend(["", "## Stack (top frames)", "```"])
        for line in p.stack.splitlines()[:12]:
            lines.append(line)
        lines.append("```")
    lines.extend([
        "",
        "## What `teane patch` should do",
        "",
        "Investigate the failing scenario above and the spec it was generated from",
        "(see `source_spec.md` in this directory). Apply the smallest change that",
        "makes the scenario pass without regressing other tests. If the spec is",
        "ambiguous or under-specified, surface that finding rather than guessing.",
        "",
    ])
    with open(os.path.join(cr_dir, "narrative.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _write_source_spec(cr_dir: str, cluster: Cluster) -> None:
    """Phase 2 ships a placeholder; Phase 4 will populate this from the
    scenario's ``// @verifies: STORY-003.AC-2`` annotation (agile) or
    spec-section anchor (waterfall). Until then, surface the field so
    the layout is stable for ``teane patch``."""
    p = cluster.primary
    if p.source_spec_id:
        body = (
            f"# Source spec reference\n\n"
            f"Scenario `{p.title}` verifies `{p.source_spec_id}`.\n\n"
            f"_Phase 4 will inline the spec excerpt here._\n"
        )
    else:
        body = (
            f"# Source spec reference\n\n"
            f"_No `@verifies` annotation found on scenario `{p.title}`._\n"
            f"_Phase 4 will require the scenario generator to emit this._\n"
        )
    with open(os.path.join(cr_dir, "source_spec.md"), "w", encoding="utf-8") as fh:
        fh.write(body)


def _write_evidence(cr_dir: str, cluster: Cluster) -> None:
    payload = {
        "cluster_key": list(cluster.key),
        "size": cluster.size(),
        "primary": _failure_to_dict(cluster.primary, with_paths=True),
        "evidence": [_failure_to_dict(f, with_paths=False) for f in cluster.evidence],
    }
    with open(os.path.join(cr_dir, "cluster_evidence.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _failure_to_dict(f: FailureRecord, *, with_paths: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "title": f.title,
        "spec_file": f.spec_file,
        "error_message": f.error_message,
        "duration_ms": f.duration_ms,
    }
    if f.source_spec_id:
        out["source_spec_id"] = f.source_spec_id
    if with_paths and f.attachments:
        out["attachments"] = sorted(f.attachments.keys())
    # First stack line is enough evidence for grouped-failure context;
    # avoids ballooning the JSON with 30 full stacks.
    first_stack = next(iter(f.stack.splitlines()), "")
    if first_stack:
        out["stack_top"] = first_stack
    return out


def _copy_attachments(cr_dir: str, cluster: Cluster) -> None:
    """Promote the primary's attachments into the CR root with stable names."""
    name_map = {
        "trace": "trace.zip",
        "screenshot": "screenshot.png",
        "dom": "dom.html",
    }
    for key, dest_name in name_map.items():
        src = cluster.primary.attachments.get(key)
        if not src or not os.path.isfile(src):
            continue
        dest = os.path.join(cr_dir, dest_name)
        try:
            shutil.copyfile(src, dest)
        except OSError as exc:
            logger.warning("[test_defects] Could not copy %s → %s: %s", src, dest, exc)
