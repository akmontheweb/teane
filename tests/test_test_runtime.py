"""Phase 5 tests for ``harness.test_runtime``.

Covers the end-to-end pipeline with all subprocess + LLM calls stubbed via
:class:`PipelineOverrides`. Asserts exit-code semantics, CR emission, the
last_test.json marker, the cache short-circuit on rerun, and infra failure
modes.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from harness.test_runtime import (
    EXIT_DEFECTS,
    EXIT_INFRA,
    EXIT_OK,
    PipelineOverrides,
    discover_base_url,
    run_test_pipeline,
)
from harness.playwright_gen import Scenario, SpecFile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workspace(tmp_path: Path) -> Path:
    """Minimal workspace with one SPEC heading so the fallback generator
    produces a scenario."""
    workspace = tmp_path / "ws"
    docs = workspace / "docs"
    docs.mkdir(parents=True)
    (docs / "SPEC_REQUIREMENTS.md").write_text("# FR-001 login\n\nlog in works.\n")
    return workspace


def _ok_playwright_json(passed_count: int = 1) -> dict:
    return {
        "suites": [{
            "specs": [{
                "file": "smoke.spec.ts",
                "tests": [
                    {"title": f"t{i}", "results": [{"status": "passed", "duration": 5}]}
                    for i in range(passed_count)
                ],
            }],
        }],
    }


def _failing_playwright_json() -> dict:
    return {
        "suites": [{
            "specs": [{
                "file": "smoke.spec.ts",
                "tests": [
                    {"title": "passed scenario", "results": [{"status": "passed", "duration": 5}]},
                    {"title": "broken", "results": [{
                        "status": "failed",
                        "duration": 100,
                        "error": {
                            "message": "POST /api/login expected 200 got 500",
                            "stack": "    at Login (/work/tests/e2e/smoke.spec.ts:10:5)\n",
                        },
                        "attachments": [],
                    }]},
                ],
            }],
        }],
    }


def _stub_runner(payload: dict, rc: int = 0):
    body = json.dumps(payload)

    def runner(cmd, cwd):
        return rc, body, ""

    return runner


# ---------------------------------------------------------------------------
# discover_base_url
# ---------------------------------------------------------------------------


def test_discover_base_url_falls_back(tmp_path: Path) -> None:
    url = asyncio.run(discover_base_url(str(tmp_path)))
    assert url.startswith("http://localhost")


def test_discover_base_url_extracts_from_installation(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "INSTALLATION.md").write_text(
        "Visit http://localhost:8080/ to use the app.\n"
    )
    url = asyncio.run(discover_base_url(str(workspace)))
    assert url == "http://localhost:8080"


# ---------------------------------------------------------------------------
# Pipeline — happy path (all passed)
# ---------------------------------------------------------------------------


def test_pipeline_clean_run_returns_exit_ok(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path)
    overrides = PipelineOverrides(
        base_url="http://app",
        chromium_runner=lambda cmd: 0,
        playwright_runner=_stub_runner(_ok_playwright_json(passed_count=3)),
        skip_reachability=True,
    )
    result = asyncio.run(run_test_pipeline(str(workspace), overrides=overrides))
    assert result.exit_code == EXIT_OK
    assert result.passed == 3
    assert result.failed == 0
    assert result.cr_paths == []


def test_pipeline_writes_last_test_marker(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path)
    overrides = PipelineOverrides(
        base_url="http://app",
        chromium_runner=lambda cmd: 0,
        playwright_runner=_stub_runner(_ok_playwright_json()),
        skip_reachability=True,
    )
    asyncio.run(run_test_pipeline(str(workspace), overrides=overrides))
    marker = workspace / ".teane" / "last_test.json"
    assert marker.is_file()
    data = json.loads(marker.read_text())
    assert data["exit_code"] == EXIT_OK
    assert data["base_url"] == "http://app"
    assert data["scenario_cache_key"]


def test_pipeline_seeds_test_db(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path)
    overrides = PipelineOverrides(
        base_url="http://app",
        chromium_runner=lambda cmd: 0,
        playwright_runner=_stub_runner(_ok_playwright_json()),
        no_cleanup=False,  # default — DB is truncated post-run
    ) if False else PipelineOverrides(
        base_url="http://app",
        chromium_runner=lambda cmd: 0,
        playwright_runner=_stub_runner(_ok_playwright_json()),
    )
    asyncio.run(run_test_pipeline(str(workspace), overrides=overrides, no_cleanup=True))
    # no_cleanup=True → DB tables remain
    db_path = workspace / ".teane" / "test_app.db"
    assert db_path.is_file()


# ---------------------------------------------------------------------------
# Pipeline — defect path (failures → CRs → exit 2)
# ---------------------------------------------------------------------------


def test_pipeline_emits_cr_on_failure(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path)
    overrides = PipelineOverrides(
        base_url="http://app",
        chromium_runner=lambda cmd: 0,
        playwright_runner=_stub_runner(_failing_playwright_json()),
        skip_reachability=True,
    )
    result = asyncio.run(run_test_pipeline(str(workspace), overrides=overrides))
    assert result.exit_code == EXIT_DEFECTS
    assert result.passed == 1
    assert result.failed == 1
    assert len(result.cr_paths) == 1
    assert os.path.isdir(result.cr_paths[0])
    # Narrative includes the failing scenario title.
    narrative = (Path(result.cr_paths[0]) / "narrative.txt").read_text()
    assert "broken" in narrative


# ---------------------------------------------------------------------------
# Pipeline — infra failures
# ---------------------------------------------------------------------------


def test_pipeline_infra_no_scenarios(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path)
    overrides = PipelineOverrides(
        base_url="http://app",
        scenario_generator=lambda ctx: [SpecFile(filename="empty.spec.ts", scenarios=[])],
        chromium_runner=lambda cmd: 0,
        playwright_runner=_stub_runner(_ok_playwright_json()),
    )
    result = asyncio.run(run_test_pipeline(str(workspace), overrides=overrides))
    assert result.exit_code == EXIT_INFRA
    assert result.reason == "no_scenarios"


def test_pipeline_infra_chromium_failed(monkeypatch, tmp_path: Path) -> None:
    # Point HOME at an empty dir (and clear the Windows equivalent) so the
    # host's real playwright cache can't satisfy `_chromium_cache_present()`.
    # Without this, `ensure_chromium_installed` short-circuits before ever
    # calling the injected failing runner, the pipeline legitimately returns
    # EXIT_OK, and the test fails on any machine that has run Playwright.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    workspace = _make_workspace(tmp_path)
    overrides = PipelineOverrides(
        base_url="http://app",
        chromium_runner=lambda cmd: 17,  # nonzero rc
        playwright_runner=_stub_runner(_ok_playwright_json()),
        skip_reachability=True,
    )
    result = asyncio.run(run_test_pipeline(str(workspace), overrides=overrides))
    assert result.exit_code == EXIT_INFRA
    assert result.reason == "chromium_missing"


def test_pipeline_infra_runner_returned_no_json(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path)

    def runner(cmd, cwd):
        return 1, "junk that is not JSON", "boom"

    overrides = PipelineOverrides(
        base_url="http://app",
        chromium_runner=lambda cmd: 0,
        playwright_runner=runner,
        skip_reachability=True,
    )
    result = asyncio.run(run_test_pipeline(str(workspace), overrides=overrides))
    assert result.exit_code == EXIT_INFRA
    assert result.reason == "runner_failed"


# ---------------------------------------------------------------------------
# Caching — second run reuses scenarios + seed without regenerating
# ---------------------------------------------------------------------------


def test_pipeline_cache_short_circuits_second_run(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path)
    gen_calls: list[int] = []

    def counting_scenario_gen(ctx):
        gen_calls.append(1)
        return [SpecFile(
            filename="smoke.spec.ts",
            scenarios=[Scenario(name="t1", verifies="STORY-1.AC-1",
                                body="  await expect(page).toHaveTitle(/.+/);\n")],
        )]

    overrides = PipelineOverrides(
        base_url="http://app",
        scenario_generator=counting_scenario_gen,
        chromium_runner=lambda cmd: 0,
        playwright_runner=_stub_runner(_ok_playwright_json()),
    )
    asyncio.run(run_test_pipeline(str(workspace), overrides=overrides))
    asyncio.run(run_test_pipeline(str(workspace), overrides=overrides))
    assert gen_calls == [1], "second run should hit the scenario cache"


# ---------------------------------------------------------------------------
# Scope flag — touched degrades to full with log (no hard error)
# ---------------------------------------------------------------------------


def test_pipeline_scope_touched_runs_without_crash(tmp_path: Path) -> None:
    """Phase 5 doesn't filter on cr_attribution yet; the call must still
    succeed and return the result rather than raising."""
    workspace = _make_workspace(tmp_path)
    # Lay down a fake deploy marker so the warn branch fires (which is
    # the *only* difference from scope=full at Phase 5).
    teane = workspace / ".teane"
    teane.mkdir(parents=True, exist_ok=True)
    (teane / "last_deploy.json").write_text(json.dumps({
        "flow": "deploy", "session_id": "abc", "exit_code": 0,
        "completed_at": "2026-06-30T00:00:00+00:00",
    }))
    overrides = PipelineOverrides(
        base_url="http://app",
        chromium_runner=lambda cmd: 0,
        playwright_runner=_stub_runner(_ok_playwright_json()),
        skip_reachability=True,
    )
    result = asyncio.run(run_test_pipeline(str(workspace), scope="touched", overrides=overrides))
    assert result.scope == "touched"
    assert result.exit_code == EXIT_OK
