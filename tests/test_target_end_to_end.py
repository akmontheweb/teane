"""Phase 6 — end-to-end coverage for ``teane test``.

Exercises the LangGraph entry node (``harness.test_target.test_node``)
with state dicts that match what ``cmd_run`` would assemble at runtime,
plus the Phase 6 edge cases that don't fit cleanly into earlier phases:

- The reachability probe blocks Playwright when the app is down between
  deploy and test (the silent-failure mode the probe was designed for).
- An empty workspace with no SPEC files still produces a smoke spec
  via the fallback generator and runs cleanly.
- The full state-dict shape (``test_scope``, ``test_retries``,
  ``test_no_cleanup``) is honoured by ``test_node``.

CLI-parser-level integration is covered by ``tests/test_cli_basics.py``
and ``tests/test_test_target.py``; this file focuses on the graph-node
behaviour end-to-end.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from harness import flow_state, test_runtime
from harness.test_runtime import PipelineOverrides
from harness.test_target import test_node as _test_node_impl


def _seed_workspace(tmp_path: Path, *, with_spec: bool = True) -> Path:
    """Build a workspace whose flow_state markers + INSTALLATION.md +
    docs/ folder mirror what a prior `teane build && teane deploy` would
    have left behind."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    flow_state.record_flow_completion(
        workspace_path=str(workspace), flow="build",
        session_id="b1", exit_code=0,
    )
    flow_state.record_flow_completion(
        workspace_path=str(workspace), flow="deploy",
        session_id="d1", exit_code=0,
    )
    (workspace / "INSTALLATION.md").write_text(
        "Open http://localhost:7777/ to use the app.\n"
    )
    if with_spec:
        docs = workspace / "docs"
        docs.mkdir()
        (docs / "SPEC_REQUIREMENTS.md").write_text(
            "# FR-001 user login\n\nUser can log in.\n"
        )
    return workspace


def _ok_playwright_blob() -> str:
    return json.dumps({
        "suites": [{"specs": [{
            "file": "smoke.spec.ts",
            "tests": [{"title": "smoke ok",
                       "results": [{"status": "passed", "duration": 1}]}],
        }]}]
    })


# ---------------------------------------------------------------------------
# Happy path through the LangGraph node
# ---------------------------------------------------------------------------


def test_e2e_clean_run_node_returns_exit_0(tmp_path: Path) -> None:
    workspace = _seed_workspace(tmp_path)
    overrides = PipelineOverrides(
        chromium_runner=lambda cmd: 0,
        skip_reachability=True,
        playwright_runner=lambda cmd, cwd: (0, _ok_playwright_blob(), ""),
    )
    state = {
        "workspace_path": str(workspace),
        "test_scope": "full",
        "test_retries": 0,
        "test_no_cleanup": False,
        "test_overrides": overrides,
    }
    result = asyncio.run(_test_node_impl(state))
    assert result["exit_code"] == 0
    test_state = result["node_state"]["test"]
    assert test_state["reason"] == "ok"
    assert test_state["scope"] == "full"
    assert test_state["passed"] == 1
    assert test_state["failed"] == 0
    # AC-coverage self-report present with safe defaults when the workspace
    # has no registered acceptance criteria (no state.db).
    assert test_state["ac_coverage_pct"] == 100.0
    assert test_state["ac_total"] == 0
    assert test_state["uncovered_acs"] == []


# ---------------------------------------------------------------------------
# Reachability probe — compose-down between deploy and test
# ---------------------------------------------------------------------------


def test_e2e_unreachable_app_emits_infra_failure(tmp_path: Path) -> None:
    workspace = _seed_workspace(tmp_path)
    overrides = PipelineOverrides(
        chromium_runner=lambda cmd: 0,
        reachability_probe=lambda url: False,  # simulate compose down
        playwright_runner=lambda cmd, cwd: (0, _ok_playwright_blob(), ""),
    )
    state = {
        "workspace_path": str(workspace),
        "test_overrides": overrides,
    }
    result = asyncio.run(_test_node_impl(state))
    assert result["exit_code"] == test_runtime.EXIT_INFRA
    test_state = result["node_state"]["test"]
    assert test_state["reason"] == "infra_failure"
    assert test_state["infra_reason"] == "app_unreachable"
    # No CRs emitted when infra failed.
    assert test_state["cr_paths"] == []


def test_e2e_reachability_probe_passes_real_responses(tmp_path: Path) -> None:
    """A reachability probe returning True lets the rest of the pipeline run."""
    workspace = _seed_workspace(tmp_path)
    probe_calls: list[str] = []

    def probe(url: str) -> bool:
        probe_calls.append(url)
        return True

    overrides = PipelineOverrides(
        chromium_runner=lambda cmd: 0,
        reachability_probe=probe,
        playwright_runner=lambda cmd, cwd: (0, _ok_playwright_blob(), ""),
    )
    state = {
        "workspace_path": str(workspace),
        "test_overrides": overrides,
    }
    result = asyncio.run(_test_node_impl(state))
    assert result["exit_code"] == 0
    assert probe_calls == ["http://localhost:7777"]


# ---------------------------------------------------------------------------
# Empty workspace — fallback generator + no spec headings
# ---------------------------------------------------------------------------


def test_e2e_empty_workspace_still_runs(tmp_path: Path) -> None:
    workspace = _seed_workspace(tmp_path, with_spec=False)
    overrides = PipelineOverrides(
        chromium_runner=lambda cmd: 0,
        skip_reachability=True,
        playwright_runner=lambda cmd, cwd: (0, _ok_playwright_blob(), ""),
    )
    state = {
        "workspace_path": str(workspace),
        "test_overrides": overrides,
    }
    result = asyncio.run(_test_node_impl(state))
    # Empty workspace → smoke.spec.ts is auto-emitted → pipeline still completes.
    assert result["exit_code"] == 0
    assert (workspace / "tests" / "e2e" / "smoke.spec.ts").is_file()


# ---------------------------------------------------------------------------
# Defects path — exit 2 + CRs on disk + last_test marker
# ---------------------------------------------------------------------------


def test_e2e_defects_emit_exit_2_and_last_test_marker(tmp_path: Path) -> None:
    workspace = _seed_workspace(tmp_path)
    failing_blob = json.dumps({
        "suites": [{"specs": [{
            "file": "smoke.spec.ts",
            "tests": [{"title": "broken",
                       "results": [{"status": "failed", "duration": 10,
                                    "error": {"message": "fail",
                                              "stack": "  at f (/x/y.spec.ts:1:1)\n"},
                                    "attachments": []}]}],
        }]}]
    })
    overrides = PipelineOverrides(
        chromium_runner=lambda cmd: 0,
        skip_reachability=True,
        playwright_runner=lambda cmd, cwd: (0, failing_blob, ""),
    )
    state = {
        "workspace_path": str(workspace),
        "test_overrides": overrides,
    }
    result = asyncio.run(_test_node_impl(state))
    assert result["exit_code"] == test_runtime.EXIT_DEFECTS
    test_state = result["node_state"]["test"]
    assert test_state["reason"] == "defects_emitted"
    assert len(test_state["cr_paths"]) == 1

    marker = workspace / ".teane" / "last_test.json"
    assert marker.is_file()
    payload = json.loads(marker.read_text())
    assert payload["exit_code"] == test_runtime.EXIT_DEFECTS
    assert payload["failed"] == 1
    assert payload["cluster_count"] == 1
