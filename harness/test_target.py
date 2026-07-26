"""``teane test`` — E2E verification pack runner.

LangGraph entry node for the ``test`` flow. The node:

  1. Reads the prerequisite markers (deploy + build|patch) via
     ``flow_state.check_test_prereqs``. Refuses with exit_code=1 if
     not satisfied.
  2. Delegates to :func:`harness.test_runtime.run_test_pipeline` for
     scenario generation, seed data, Playwright execution, and CR
     emission.
  3. Sets ``node_state.test`` with structured outcome data and
     propagates ``exit_code`` so cmd_run returns the right shell code:
       0 — all scenarios passed
       1 — prereq failed or infra problem (chromium missing, no
           scenarios, runner exited non-zero with no JSON)
       2 — defects emitted (CI-friendly "tests ran but failed" signal)

The target intentionally does NOT auto-invoke ``teane patch``. Operators
or CI scripts chain ``teane test && teane patch`` themselves (or with
``||`` to enable an auto-fix loop) — keeps the primitives composable.
"""

from __future__ import annotations

import logging
from typing import Any

from harness import flow_state
from harness import test_runtime

logger = logging.getLogger(__name__)


REASON_PREREQ_FAILED = "prereq_failed"
REASON_NOT_IMPLEMENTED = "not_implemented"  # legacy from Phase 1; kept for tests
REASON_OK = "ok"
REASON_INFRA = "infra_failure"
REASON_DEFECTS = "defects_emitted"


async def test_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph entry node for ``flow == "test"``.

    State keys consumed:
      - ``workspace_path``        — required
      - ``test_scope``            — "touched" (default) or "full"
      - ``test_retries``          — Playwright per-scenario retry count
      - ``test_no_cleanup``       — skip teardown of seeded test DB
      - ``test_overrides``        — :class:`test_runtime.PipelineOverrides`
                                    instance (tests inject runners here)
    """
    workspace_path = state.get("workspace_path") or ""
    logger.info("[test_node] entry. Workspace=%s", workspace_path)

    ok, prereq_reason = flow_state.check_test_prereqs(workspace_path)
    if not ok:
        logger.error("[test_node] prerequisite check failed: %s", prereq_reason)
        return {
            "exit_code": 1,
            "node_state": {
                "test": {
                    "skipped": True,
                    "reason": REASON_PREREQ_FAILED,
                    "detail": prereq_reason,
                },
            },
        }

    scope = state.get("test_scope", "touched")
    retries = int(state.get("test_retries", 2))
    no_cleanup = bool(state.get("test_no_cleanup", False))
    overrides = state.get("test_overrides")

    result = await test_runtime.run_test_pipeline(
        workspace_path,
        scope=scope,
        retries=retries,
        no_cleanup=no_cleanup,
        overrides=overrides,
    )

    reason: str
    if result.exit_code == test_runtime.EXIT_OK:
        reason = REASON_OK
    elif result.exit_code == test_runtime.EXIT_DEFECTS:
        reason = REASON_DEFECTS
    else:
        reason = REASON_INFRA

    logger.info(
        "[test_node] complete. passed=%d failed=%d clusters=%d crs=%d exit=%d",
        result.passed, result.failed, result.cluster_count,
        len(result.cr_paths), result.exit_code,
    )

    # Self-report acceptance-criteria completeness. The end-of-run
    # traceability gate is what BLOCKS on uncovered ACs (flow == "test"),
    # but surfacing coverage here — right after generation + execution —
    # gives the operator an immediate, per-run signal of how many ACs this
    # pass actually verified, and flags when a `--scope touched` run left
    # criteria outside the changed set unverified.
    ac_coverage_pct = 100.0
    ac_verified = ac_total = 0
    uncovered_ac_keys: list[str] = []
    try:
        from harness.traceability import audit_workspace
        report = audit_workspace(workspace_path)
        if report is not None:
            ac_coverage_pct = report.ac_coverage_pct
            ac_verified, ac_total = report.verified_acs, report.total_acs
            uncovered_ac_keys = [u.ac_key for u in report.untested_acs]
            if uncovered_ac_keys:
                logger.warning(
                    "[test_node] AC coverage %d/%d (%.0f%%) after --scope=%s. "
                    "%d criteria still unverified: %s%s",
                    ac_verified, ac_total, ac_coverage_pct, scope,
                    len(uncovered_ac_keys), ", ".join(uncovered_ac_keys[:8]),
                    " …" if len(uncovered_ac_keys) > 8 else "",
                )
                if scope != "full":
                    logger.warning(
                        "[test_node] Run `teane test --scope full` to generate "
                        "and run a scenario for EVERY acceptance criterion."
                    )
            else:
                logger.info(
                    "[test_node] AC coverage %d/%d (100%%) — every acceptance "
                    "criterion has a verifying test.", ac_verified, ac_total,
                )
    except Exception as exc:  # noqa: BLE001 — coverage report must never break the run
        logger.debug("[test_node] AC coverage self-report skipped: %s", exc)

    return {
        "exit_code": result.exit_code,
        "node_state": {
            "test": {
                "skipped": False,
                "reason": reason,
                "passed": result.passed,
                "failed": result.failed,
                "cluster_count": result.cluster_count,
                "cr_paths": list(result.cr_paths),
                "base_url": result.base_url,
                "scope": result.scope,
                "infra_reason": result.reason if reason == REASON_INFRA else "",
                "ac_coverage_pct": ac_coverage_pct,
                "ac_verified": ac_verified,
                "ac_total": ac_total,
                "uncovered_acs": uncovered_ac_keys,
            },
        },
    }
