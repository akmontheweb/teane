"""Phase 4 tests for ``harness.playwright_gen``.

Covers:
- gather_scenario_context: bundles SchemaContext + base_url; flow_kind threaded
- compute_scenario_cache_key: stable / sensitive in the right places
- fallback_scenarios (waterfall): one spec, one scenario per heading anchor
- fallback_scenarios (agile): one spec per story, one scenario per AC
- generate_scenarios pluggability + validation
- write_scenarios: file layout, @verifies annotation present, cache marker
- cached_scenarios_dir: hit / miss
- ensure_chromium_installed: skipped when cache present, runner invoked otherwise
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.playwright_gen import (
    FLOW_AGILE,
    FLOW_WATERFALL,
    Scenario,
    ScenarioContext,
    SpecFile,
    cached_scenarios_dir,
    compute_scenario_cache_key,
    ensure_chromium_installed,
    fallback_scenarios,
    gather_scenario_context,
    generate_scenarios,
    write_scenarios,
)
from harness.test_data_gen import SchemaContext


# ---------------------------------------------------------------------------
# Context gathering + cache key
# ---------------------------------------------------------------------------


def test_gather_scenario_context_threads_base_url(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    ctx = gather_scenario_context(str(workspace), base_url="http://app:8080")
    assert ctx.base_url == "http://app:8080"
    assert ctx.schema.flow_kind == FLOW_WATERFALL


def test_cache_key_stable_for_same_context() -> None:
    s = SchemaContext(workspace_path="/x", flow_kind=FLOW_WATERFALL,
                      spec_excerpts={"a.md": "hi"})
    ctx_a = ScenarioContext(schema=s, base_url="http://x")
    ctx_b = ScenarioContext(schema=s, base_url="http://x")
    assert compute_scenario_cache_key(ctx_a) == compute_scenario_cache_key(ctx_b)


def test_cache_key_changes_with_base_url() -> None:
    s = SchemaContext(workspace_path="/x", flow_kind=FLOW_WATERFALL)
    ctx_a = ScenarioContext(schema=s, base_url="http://a")
    ctx_b = ScenarioContext(schema=s, base_url="http://b")
    assert compute_scenario_cache_key(ctx_a) != compute_scenario_cache_key(ctx_b)


# ---------------------------------------------------------------------------
# Fallback generator — waterfall
# ---------------------------------------------------------------------------


def test_fallback_waterfall_extracts_headings() -> None:
    schema = SchemaContext(
        workspace_path="/x",
        flow_kind=FLOW_WATERFALL,
        spec_excerpts={
            "docs/SPEC_REQUIREMENTS.md": (
                "# FR-001 user login\n\nblah\n\n## FR-002 logout\n\nblah\n"
            ),
        },
    )
    ctx = ScenarioContext(schema=schema, base_url="http://x")
    [spec] = fallback_scenarios(ctx)
    assert spec.filename == "smoke.spec.ts"
    anchors = [sc.verifies for sc in spec.scenarios]
    assert anchors == [
        "SPEC_REQUIREMENTS.md#fr-001-user-login",
        "SPEC_REQUIREMENTS.md#fr-002-logout",
    ]


def test_fallback_waterfall_handles_no_specs() -> None:
    ctx = ScenarioContext(
        schema=SchemaContext(workspace_path="/x", flow_kind=FLOW_WATERFALL),
        base_url="http://x",
    )
    [spec] = fallback_scenarios(ctx)
    assert spec.filename == "smoke.spec.ts"
    assert len(spec.scenarios) == 1
    assert spec.scenarios[0].verifies.startswith("SPEC_REQUIREMENTS.md#")


# ---------------------------------------------------------------------------
# Fallback generator — agile
# ---------------------------------------------------------------------------


def test_fallback_agile_one_spec_per_story() -> None:
    schema = SchemaContext(
        workspace_path="/x",
        flow_kind=FLOW_AGILE,
        stories=[
            {
                "story_key": "STORY-1",
                "title": "User Login",
                "acceptance_criteria_keys": ["STORY-1.AC-1", "STORY-1.AC-2"],
            },
            {
                "story_key": "STORY-2",
                "title": "Cart Checkout",
                "acceptance_criteria_keys": ["STORY-2.AC-1"],
            },
        ],
    )
    ctx = ScenarioContext(schema=schema, base_url="http://app")
    specs = fallback_scenarios(ctx)
    assert {s.filename for s in specs} == {"user-login.spec.ts", "cart-checkout.spec.ts"}
    login = next(s for s in specs if s.filename == "user-login.spec.ts")
    assert [sc.verifies for sc in login.scenarios] == ["STORY-1.AC-1", "STORY-1.AC-2"]


def test_fallback_agile_falls_back_when_no_stories() -> None:
    schema = SchemaContext(workspace_path="/x", flow_kind=FLOW_AGILE, stories=[])
    ctx = ScenarioContext(schema=schema, base_url="http://app")
    specs = fallback_scenarios(ctx)
    assert specs[0].filename == "smoke.spec.ts"


# ---------------------------------------------------------------------------
# generate_scenarios + validation
# ---------------------------------------------------------------------------


def test_generate_scenarios_accepts_custom_generator() -> None:
    schema = SchemaContext(workspace_path="/x", flow_kind=FLOW_WATERFALL)
    ctx = ScenarioContext(schema=schema, base_url="http://x")

    custom = SpecFile(
        filename="auth.spec.ts",
        scenarios=[Scenario(name="logs in", verifies="STORY-1.AC-1", body="// body")],
    )
    out = generate_scenarios(ctx, generator=lambda c: [custom])
    assert out == [custom]


def test_generate_scenarios_rejects_bad_shape() -> None:
    ctx = ScenarioContext(
        schema=SchemaContext(workspace_path="/x", flow_kind=FLOW_WATERFALL),
        base_url="http://x",
    )
    with pytest.raises(ValueError):
        generate_scenarios(ctx, generator=lambda c: "not a list")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        generate_scenarios(ctx, generator=lambda c: [
            SpecFile(filename="missing-extension.ts", scenarios=[]),
        ])
    with pytest.raises(ValueError):
        generate_scenarios(ctx, generator=lambda c: [SpecFile(
            filename="x.spec.ts",
            scenarios=[Scenario(name="", verifies="STORY-1.AC-1", body="// b")],
        )])


# ---------------------------------------------------------------------------
# write_scenarios + cache marker
# ---------------------------------------------------------------------------


def test_write_scenarios_emits_files_with_verifies(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    spec = SpecFile(
        filename="login.spec.ts",
        scenarios=[
            Scenario(name="rejects empty password", verifies="STORY-1.AC-1",
                     body="  await page.goto('http://x');\n  await expect(page).toHaveTitle(/.+/);\n"),
        ],
    )
    paths = write_scenarios(str(workspace), [spec], cache_key="k1")
    assert len(paths) == 1
    body = (Path(paths[0])).read_text(encoding="utf-8")
    assert "@verifies: STORY-1.AC-1" in body
    assert "test(\"rejects empty password\"" in body
    assert (Path(workspace) / "tests" / "e2e" / ".cache_key").read_text() == "k1"


def test_cached_scenarios_dir_hits_and_misses(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    spec = SpecFile(filename="x.spec.ts", scenarios=[
        Scenario(name="t", verifies="STORY-1.AC-1", body="  // body\n"),
    ])
    write_scenarios(str(workspace), [spec], cache_key="hash-v1")
    assert cached_scenarios_dir(str(workspace), "hash-v1") is not None
    assert cached_scenarios_dir(str(workspace), "hash-v2") is None


# ---------------------------------------------------------------------------
# ensure_chromium_installed
# ---------------------------------------------------------------------------


def test_ensure_chromium_skips_when_cache_present(monkeypatch, tmp_path) -> None:
    fake_cache = tmp_path / ".cache" / "ms-playwright" / "chromium-1234"
    fake_cache.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))

    called = []

    def fake_runner(cmd):
        called.append(cmd)
        return 0

    assert ensure_chromium_installed(runner=fake_runner) is True
    assert called == []  # cache hit short-circuited


def test_ensure_chromium_invokes_runner_when_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))  # no cache dir
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    # Pin shutil.which so the assertion doesn't depend on whether the
    # host has Node installed. The resolved path is what the fix passes
    # to subprocess.run so the .cmd/.exe shim is found on Windows.
    monkeypatch.setattr("harness.playwright_gen.shutil.which", lambda name: "/fake/bin/npx" if name == "npx" else None)
    called: list[list[str]] = []

    def fake_runner(cmd):
        called.append(cmd)
        return 0

    assert ensure_chromium_installed(runner=fake_runner) is True
    assert called == [["/fake/bin/npx", "playwright", "install", "chromium"]]


def test_ensure_chromium_reports_runner_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert ensure_chromium_installed(runner=lambda cmd: 17) is False
