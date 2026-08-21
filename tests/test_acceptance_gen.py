"""Tests for the ADR-0006 Phase-0 dual-altitude acceptance generator.

All offline — the LLM path is exercised with a fake gateway so no key or
network is needed.
"""

from __future__ import annotations

import json

import pytest

from harness import acceptance_gen as ag
from harness.acceptance_gen import (
    ALTITUDE_E2E,
    ALTITUDE_INTEGRATION,
    CLASS_BACKEND,
    CLASS_UI,
    AcceptanceScenario,
    StoryAcceptanceContext,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _ctx() -> StoryAcceptanceContext:
    return StoryAcceptanceContext(
        story_key="STORY-001",
        title="Add a new contact",
        description="As a keeper I want to add a contact.",
        acceptance_criteria=[
            {"ac_key": "STORY-001.AC-1", "text": "Add a valid contact through API"},
            {"ac_key": "STORY-001.AC-2", "text": "Add Contact modal UI opens"},
            {"ac_key": "STORY-001.AC-3", "text": "Reject a future date of birth returns 422"},
        ],
        routes=[{"method": "POST", "path": "/contacts"}],
        stack={"backend": "fastapi", "db": "sqlite"},
    )


class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content
        self.usage = {}


class _FakeGateway:
    """Minimal async gateway stub: returns a canned response, deducts budget."""

    def __init__(self, content: str = "", *, raise_exc: Exception | None = None) -> None:
        self._content = content
        self._raise = raise_exc
        self.calls: list[dict] = []

    async def dispatch(self, *, messages, role, budget_remaining_usd, **kw):
        self.calls.append({"messages": messages, "role": role, "kw": kw})
        if self._raise is not None:
            raise self._raise
        return _Resp(self._content), budget_remaining_usd - 0.01


def _good_llm_json() -> str:
    return json.dumps({
        "scenarios": [
            {
                "verifies": "STORY-001.AC-1",
                "classification": "backend-verifiable",
                "rationale": "posts a valid contact",
                "integration": {
                    "name": "test_add_valid_contact",
                    "body": "    response = client.post('/contacts', json={'first_name': 'A', 'date_of_birth': '2000-01-01'})\n    assert response.status_code == 201\n    assert response.json()['first_name'] == 'A'",
                },
                "e2e": {
                    "name": "adds a valid contact",
                    "body": "  await page.goto('/');\n  await page.getByRole('button', {name: 'Add'}).click();\n  await expect(page.getByText('A')).toBeVisible();",
                },
            },
            {
                "verifies": "STORY-001.AC-2",
                "classification": "ui-only",
                "rationale": "modal is pure UI",
                "integration": None,
                "e2e": {
                    "name": "opens the add modal",
                    "body": "  await page.goto('/');\n  await page.getByRole('button', {name: 'Add'}).click();\n  await expect(page.getByRole('dialog')).toBeVisible();",
                },
            },
        ]
    })


# ---------------------------------------------------------------------------
# validate_scenarios — the anti-rubber-stamp gate
# ---------------------------------------------------------------------------


class TestValidateScenarios:
    def test_keeps_substantive_integration_and_e2e(self):
        ctx = _ctx()
        cands = [
            AcceptanceScenario("STORY-001.AC-1", "t", ALTITUDE_INTEGRATION, CLASS_BACKEND,
                               "    response = client.post('/contacts', json={})\n    assert response.status_code == 422"),
            AcceptanceScenario("STORY-001.AC-1", "t", ALTITUDE_E2E, CLASS_BACKEND,
                               "  await expect(page.getByText('x')).toBeVisible();"),
        ]
        kept, dropped = ag.validate_scenarios(cands, ctx)
        assert len(kept) == 2
        assert dropped == []

    def test_drops_tautology(self):
        ctx = _ctx()
        cands = [
            AcceptanceScenario("STORY-001.AC-1", "t", ALTITUDE_INTEGRATION, CLASS_BACKEND,
                               "    assert True"),
        ]
        kept, dropped = ag.validate_scenarios(cands, ctx)
        assert kept == []
        assert dropped[0]["reason"] == "tautological assertion"

    def test_drops_assert_x_equals_x(self):
        ctx = _ctx()
        cands = [
            AcceptanceScenario("STORY-001.AC-1", "t", ALTITUDE_INTEGRATION, CLASS_BACKEND,
                               "    response = client.get('/')\n    assert foo == foo"),
        ]
        kept, dropped = ag.validate_scenarios(cands, ctx)
        assert kept == []
        assert dropped[0]["reason"] == "tautological assertion"

    def test_drops_e2e_placeholder_title_check(self):
        ctx = _ctx()
        cands = [
            AcceptanceScenario("STORY-001.AC-1", "t", ALTITUDE_E2E, CLASS_BACKEND,
                               "  await expect(page).toHaveTitle(/.+/);"),
        ]
        kept, dropped = ag.validate_scenarios(cands, ctx)
        assert kept == []
        assert dropped[0]["reason"] == "placeholder toHaveTitle(/.+/)"

    def test_drops_syntactically_broken_integration_body(self):
        ctx = _ctx()
        cands = [
            AcceptanceScenario("STORY-001.AC-1", "t", ALTITUDE_INTEGRATION, CLASS_BACKEND,
                               "resp = client.get('/'\nassert resp.status_code == 200"),  # unbalanced paren
        ]
        kept, dropped = ag.validate_scenarios(cands, ctx)
        assert kept == []
        assert dropped[0]["reason"] == "integration body is not valid python"

    def test_drops_integration_without_assert(self):
        ctx = _ctx()
        cands = [
            AcceptanceScenario("STORY-001.AC-1", "t", ALTITUDE_INTEGRATION, CLASS_BACKEND,
                               "    response = client.get('/')"),
        ]
        kept, dropped = ag.validate_scenarios(cands, ctx)
        assert kept == []
        assert dropped[0]["reason"] == "integration body has no assert"

    def test_drops_integration_that_never_calls_client(self):
        ctx = _ctx()
        cands = [
            AcceptanceScenario("STORY-001.AC-1", "t", ALTITUDE_INTEGRATION, CLASS_BACKEND,
                               "    assert 2 + 2 == 4"),
        ]
        kept, dropped = ag.validate_scenarios(cands, ctx)
        assert kept == []
        assert dropped[0]["reason"] == "integration body never calls the client"

    def test_drops_unknown_verifies(self):
        ctx = _ctx()
        cands = [
            AcceptanceScenario("STORY-999.AC-9", "t", ALTITUDE_INTEGRATION, CLASS_BACKEND,
                               "    response = client.get('/')\n    assert response.status_code == 200"),
        ]
        kept, dropped = ag.validate_scenarios(cands, ctx)
        assert kept == []
        assert "not an AC" in dropped[0]["reason"]

    def test_drops_ui_only_integration(self):
        ctx = _ctx()
        cands = [
            AcceptanceScenario("STORY-001.AC-2", "t", ALTITUDE_INTEGRATION, CLASS_UI,
                               "    response = client.get('/')\n    assert response.status_code == 200"),
        ]
        kept, dropped = ag.validate_scenarios(cands, ctx)
        assert kept == []
        assert "ui-only" in dropped[0]["reason"]

    def test_drops_bad_enums(self):
        ctx = _ctx()
        cands = [
            AcceptanceScenario("STORY-001.AC-1", "t", "browser", CLASS_BACKEND, "assert x"),
            AcceptanceScenario("STORY-001.AC-1", "t", ALTITUDE_E2E, "maybe", "expect(x)"),
        ]
        kept, dropped = ag.validate_scenarios(cands, ctx)
        assert kept == []
        assert {d["reason"].split()[0] for d in dropped} == {"unknown"}


# ---------------------------------------------------------------------------
# _parse_candidates — JSON flattening + altitude filtering
# ---------------------------------------------------------------------------


class TestParseCandidates:
    def test_flattens_both_altitudes(self):
        data = json.loads(_good_llm_json())
        out = ag._parse_candidates(data, frozenset({ALTITUDE_INTEGRATION, ALTITUDE_E2E}))
        # AC-1 → integration + e2e; AC-2 (ui-only) → e2e only
        assert len(out) == 3
        assert sorted({s.altitude for s in out}) == [ALTITUDE_E2E, ALTITUDE_INTEGRATION]

    def test_altitude_filter_drops_e2e(self):
        data = json.loads(_good_llm_json())
        out = ag._parse_candidates(data, frozenset({ALTITUDE_INTEGRATION}))
        assert all(s.altitude == ALTITUDE_INTEGRATION for s in out)
        assert len(out) == 1  # only AC-1 has an integration block

    def test_null_integration_skipped(self):
        data = json.loads(_good_llm_json())
        out = ag._parse_candidates(data, frozenset({ALTITUDE_INTEGRATION, ALTITUDE_E2E}))
        ui = [s for s in out if s.verifies == "STORY-001.AC-2"]
        assert len(ui) == 1 and ui[0].altitude == ALTITUDE_E2E

    def test_garbage_shape_returns_empty(self):
        assert ag._parse_candidates({}, frozenset({ALTITUDE_E2E})) == []
        assert ag._parse_candidates({"scenarios": "nope"}, frozenset({ALTITUDE_E2E})) == []


# ---------------------------------------------------------------------------
# generate_acceptance_scenarios — the async LLM core
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGenerateAcceptanceScenarios:
    async def test_happy_path_parses_and_validates(self):
        gw = _FakeGateway(_good_llm_json())
        res = await ag.generate_acceptance_scenarios(_ctx(), gateway=gw, budget_remaining_usd=1.0)
        assert res.source == "llm"
        assert res.budget_remaining_usd == pytest.approx(0.99)
        # 3 valid scenarios survive (AC-1 integration+e2e, AC-2 e2e)
        assert len(res.scenarios) == 3
        assert len(res.integration()) == 1
        assert len(res.e2e()) == 2
        # the planning role was used
        from harness.gateway import NodeRole
        assert gw.calls[0]["role"] == NodeRole.PLANNING

    async def test_dispatch_exception_is_fail_soft(self):
        gw = _FakeGateway(raise_exc=RuntimeError("boom"))
        res = await ag.generate_acceptance_scenarios(_ctx(), gateway=gw, budget_remaining_usd=1.0)
        assert res.scenarios == []
        assert res.budget_remaining_usd == 1.0  # unchanged

    async def test_non_json_is_fail_soft(self):
        gw = _FakeGateway("here are your tests: <not json>")
        res = await ag.generate_acceptance_scenarios(_ctx(), gateway=gw, budget_remaining_usd=1.0)
        assert res.scenarios == []

    async def test_fenced_json_is_stripped(self):
        gw = _FakeGateway("```json\n" + _good_llm_json() + "\n```")
        res = await ag.generate_acceptance_scenarios(_ctx(), gateway=gw, budget_remaining_usd=1.0)
        assert len(res.scenarios) == 3

    async def test_weak_scenarios_are_dropped_not_kept(self):
        weak = json.dumps({"scenarios": [{
            "verifies": "STORY-001.AC-1", "classification": "backend-verifiable",
            "integration": {"name": "t", "body": "    assert True"},
            "e2e": {"name": "t", "body": "  await expect(page).toHaveTitle(/.+/);"},
        }]})
        gw = _FakeGateway(weak)
        res = await ag.generate_acceptance_scenarios(_ctx(), gateway=gw, budget_remaining_usd=1.0)
        assert res.scenarios == []
        assert len(res.dropped) == 2

    async def test_altitudes_config_limits_output(self):
        gw = _FakeGateway(_good_llm_json())
        res = await ag.generate_acceptance_scenarios(
            _ctx(), gateway=gw, budget_remaining_usd=1.0,
            config={"altitudes": ["integration"]},
        )
        assert all(s.altitude == ALTITUDE_INTEGRATION for s in res.scenarios)


# ---------------------------------------------------------------------------
# Fallback — honest, designed-to-fail scaffolds
# ---------------------------------------------------------------------------


class TestFallback:
    def test_classifies_ui_vs_backend(self):
        res = ag.fallback_acceptance_scenarios(_ctx())
        assert res.source == "fallback"
        by_ac = {}
        for s in res.scenarios:
            by_ac.setdefault(s.verifies, set()).add((s.altitude, s.classification))
        # AC-2 is "modal UI" → ui-only, e2e only (no integration)
        assert (ALTITUDE_INTEGRATION, CLASS_UI) not in by_ac["STORY-001.AC-2"]
        assert (ALTITUDE_E2E, CLASS_UI) in by_ac["STORY-001.AC-2"]
        # AC-1 / AC-3 are backend → integration present
        assert (ALTITUDE_INTEGRATION, CLASS_BACKEND) in by_ac["STORY-001.AC-1"]

    def test_bodies_are_designed_to_fail(self):
        res = ag.fallback_acceptance_scenarios(_ctx())
        for s in res.scenarios:
            if s.altitude == ALTITUDE_INTEGRATION:
                assert "pytest.fail" in s.body
            else:
                assert "expect(false)" in s.body


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRendering:
    def test_integration_file_has_markers_and_fixture(self):
        scen = [AcceptanceScenario("STORY-001.AC-1", "test_add", ALTITUDE_INTEGRATION,
                                   CLASS_BACKEND, "    assert client.get('/').status_code == 200")]
        out = ag.render_integration_file("STORY-001", scen)
        assert "# @verifies: STORY-001.AC-1" in out
        assert "def test_add(client):" in out
        assert "import pytest" in out

    def test_integration_file_renames_non_test_prefixed(self):
        scen = [AcceptanceScenario("STORY-001.AC-1", "add a contact", ALTITUDE_INTEGRATION,
                                   CLASS_BACKEND, "    assert client.get('/').status_code == 200")]
        out = ag.render_integration_file("STORY-001", scen)
        assert "def test_add_a_contact(client):" in out

    def test_renders_valid_python_from_column0_body(self):
        # LLM bodies often come at column 0 with their own nested blocks; the
        # renderer must re-indent so the file parses.
        import ast
        body = (
            "resp = client.post('/contacts', json={})\n"
            "assert resp.status_code == 201\n"
            "data = resp.json()\n"
            "if isinstance(data, dict):\n"
            "    data = data.get('contacts', [])\n"
            "assert data is not None"
        )
        scen = [AcceptanceScenario("STORY-1.AC-1", "test_add", ALTITUDE_INTEGRATION,
                                   CLASS_BACKEND, body)]
        out = ag.render_integration_file("STORY-1", scen)
        ast.parse(out)  # must be valid Python
        assert "    resp = client.post" in out  # re-indented to one level
        assert "        data = data.get" in out  # nested block preserved (+4)

    def test_renders_valid_python_from_preindented_body(self):
        import ast
        body = "    resp = client.get('/')\n    assert resp.status_code == 200"
        scen = [AcceptanceScenario("STORY-1.AC-1", "test_g", ALTITUDE_INTEGRATION,
                                   CLASS_BACKEND, body)]
        ast.parse(ag.render_integration_file("STORY-1", scen))

    def test_e2e_spec_has_verifies_and_import(self):
        scen = [AcceptanceScenario("STORY-001.AC-2", "opens modal", ALTITUDE_E2E,
                                   CLASS_UI, "  await expect(page.getByRole('dialog')).toBeVisible();")]
        out = ag.render_e2e_spec("STORY-001", scen)
        assert "// @verifies: STORY-001.AC-2" in out
        assert "import { test, expect } from '@playwright/test';" in out
        assert 'test("opens modal"' in out


# ---------------------------------------------------------------------------
# Route discovery + prompt
# ---------------------------------------------------------------------------


class TestRouteDiscovery:
    def test_discovers_prefixed_fastapi_routes(self, tmp_path):
        api = tmp_path / "server" / "app" / "api"
        api.mkdir(parents=True)
        (api / "contacts.py").write_text(
            "from fastapi import APIRouter\n"
            "router = APIRouter(prefix='/contacts', tags=['contacts'])\n"
            "@router.get('')\n"
            "def list_contacts(): ...\n"
            "@router.post('')\n"
            "def add(): ...\n"
            "@router.delete('/{contact_id}')\n"
            "def rm(): ...\n"
        )
        routes = ag.discover_routes(str(tmp_path))
        paths = {(r["method"], r["path"]) for r in routes}
        assert ("GET", "/contacts") in paths
        assert ("POST", "/contacts") in paths
        assert ("DELETE", "/contacts/{contact_id}") in paths

    def test_skips_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules" / "x"
        nm.mkdir(parents=True)
        (nm / "junk.py").write_text("@app.get('/leak')\ndef f(): ...\n")
        routes = ag.discover_routes(str(tmp_path))
        assert routes == []


class TestAppDiscoveryAndConftest:
    def test_discovers_create_app_factory(self, tmp_path):
        app_dir = tmp_path / "server" / "app"
        app_dir.mkdir(parents=True)
        (app_dir / "main.py").write_text(
            "from fastapi import FastAPI\n"
            "def create_app() -> FastAPI:\n    return FastAPI()\n"
        )
        d = ag.discover_app_factory(str(tmp_path))
        assert d == {"module": "server.app.main", "symbol": "create_app", "kind": "factory"}

    def test_discovers_singleton_when_no_factory(self, tmp_path):
        (tmp_path / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n")
        d = ag.discover_app_factory(str(tmp_path))
        assert d == {"module": "main", "symbol": "app", "kind": "singleton"}

    def test_factory_preferred_over_singleton(self, tmp_path):
        (tmp_path / "a.py").write_text("app = FastAPI()\n")
        (tmp_path / "b.py").write_text("def create_app():\n    return 1\n")
        d = ag.discover_app_factory(str(tmp_path))
        assert d["kind"] == "factory"

    def test_skips_tests_dir(self, tmp_path):
        t = tmp_path / "tests"
        t.mkdir()
        (t / "conftest.py").write_text("app = FastAPI()\n")
        assert ag.discover_app_factory(str(tmp_path)) is None

    def test_conftest_renders_factory_call(self):
        out = ag.render_acceptance_conftest({"module": "server.app.main", "symbol": "create_app", "kind": "factory"})
        assert "from server.app.main import create_app" in out
        assert "app = create_app()" in out
        assert "_Client(app, base_url=_ACCEPTANCE_BASE_URL, raise_server_exceptions=False)" in out
        assert "http://127.0.0.1:8000" in out  # loopback+port base URL
        assert "class _Client(TestClient):" in out  # strips Host/Origin
        assert "('host', 'origin')" in out
        assert "def client():" in out

    def test_conftest_renders_singleton_without_call(self):
        out = ag.render_acceptance_conftest({"module": "main", "symbol": "app", "kind": "singleton"})
        assert "app = app\n" in out or "app = app " in out

    def test_discovers_pydantic_settings_db_env_var(self, tmp_path):
        app_dir = tmp_path / "server" / "app"
        app_dir.mkdir(parents=True)
        (app_dir / "config.py").write_text(
            "from pydantic_settings import BaseSettings, SettingsConfigDict\n"
            "class Settings(BaseSettings):\n"
            "    model_config = SettingsConfigDict(env_prefix='LUMINA_', extra='ignore')\n"
            "    db_path: str = './data/lumina.db'\n"
        )
        assert ag.discover_db_env_var(str(tmp_path)) == "LUMINA_DB_PATH"

    def test_db_env_var_none_when_absent(self, tmp_path):
        (tmp_path / "x.py").write_text("db_path = './x.db'\n")  # no env_prefix
        assert ag.discover_db_env_var(str(tmp_path)) is None

    def test_conftest_isolation_block_when_db_env_var(self):
        out = ag.render_acceptance_conftest(
            {"module": "server.app.main", "symbol": "create_app", "kind": "factory"},
            db_env_var="LUMINA_DB_PATH",
        )
        assert "tempfile.mkdtemp" in out  # a fresh temp DIR (apps harden the parent)
        assert "os.environ['LUMINA_DB_PATH'] = _db" in out
        assert "app = create_app()" in out
        # restores prior value + cleans the temp dir
        assert "shutil.rmtree(_dir" in out
        assert "_prev" in out

    def test_conftest_no_isolation_without_db_env_var(self):
        out = ag.render_acceptance_conftest(
            {"module": "m", "symbol": "create_app", "kind": "factory"})
        assert "tempfile.mkdtemp" not in out

    def test_conftest_with_seed_is_valid_and_applies(self):
        import ast
        out = ag.render_acceptance_conftest(
            {"module": "server.app.main", "symbol": "create_app", "kind": "factory"},
            db_env_var="LUMINA_DB_PATH", seed=True,
        )
        ast.parse(out)  # the inlined stdlib seed applier must be valid Python
        assert "def _apply_seed(db_path):" in out
        assert "_apply_seed(_db)" in out
        assert "sqlite3" in out
        # applier must NOT import teane's harness (unavailable in the sandbox)
        assert "harness" not in out

    def test_seed_ignored_without_isolation(self):
        # seed=True but no db_env_var → no known DB path → no seed applier
        out = ag.render_acceptance_conftest(
            {"module": "m", "symbol": "create_app", "kind": "factory"}, seed=True)
        assert "_apply_seed" not in out


class TestPrompt:
    def test_user_prompt_includes_ac_text_and_routes(self):
        p = ag.build_user_prompt(_ctx(), max_scenarios=10)
        assert "Reject a future date of birth returns 422" in p
        assert "POST /contacts" in p
        assert "STORY-001" in p
        assert "at most 10" in p


# ---------------------------------------------------------------------------
# Seed generator
# ---------------------------------------------------------------------------

from harness.acceptance_gen import SeedContext  # noqa: E402


def _seed_ctx() -> SeedContext:
    return SeedContext(
        flow_kind="agile",
        table_schemas=[{
            "table": "contacts",
            "ddl": "CREATE TABLE contacts (id INTEGER PRIMARY KEY, first_name TEXT NOT NULL, date_of_birth TEXT NOT NULL);",
        }],
        stories=[{
            "story_key": "STORY-004",
            "title": "Countdown",
            "acceptance_criteria": [{"ac_key": "STORY-004.AC-1", "text": "days until next birthday"}],
        }],
    )


def _good_seed_json() -> str:
    return json.dumps({"tables": {
        "contacts": [
            {"first_name": "Ada", "date_of_birth": "1990-06-01", "_verifies": "STORY-004.AC-1"},
            {"first_name": "Bo", "date_of_birth": "1985-12-25"},
        ],
    }})


class TestSchemaDiscovery:
    def test_extracts_create_table_from_python(self, tmp_path):
        (tmp_path / "migrations.py").write_text(
            'SQL = """\nCREATE TABLE IF NOT EXISTS contacts (\n'
            '    id INTEGER PRIMARY KEY AUTOINCREMENT,\n'
            '    first_name TEXT NOT NULL,\n'
            '    date_of_birth TEXT NOT NULL\n'
            ');\n"""\n'
        )
        schemas = ag.discover_table_schemas(str(tmp_path))
        assert len(schemas) == 1
        assert schemas[0]["table"] == "contacts"
        assert "first_name TEXT NOT NULL" in schemas[0]["ddl"]


class TestValidateSeed:
    def test_keeps_valid_rows_and_caps(self):
        ctx = _seed_ctx()
        data = {"tables": {"contacts": [{"first_name": f"n{i}"} for i in range(50)]}}
        seed, dropped = ag.validate_seed(data, ctx, max_rows=5)
        assert len(seed["tables"]["contacts"]) == 5

    def test_drops_unknown_table(self):
        ctx = _seed_ctx()
        data = {"tables": {"orders": [{"x": 1}]}}
        seed, dropped = ag.validate_seed(data, ctx, max_rows=20)
        assert seed["tables"] == {}
        assert dropped[0]["reason"] == "not in schema"

    def test_drops_row_with_only_internal_keys(self):
        ctx = _seed_ctx()
        data = {"tables": {"contacts": [{"_verifies": "STORY-004.AC-1"}]}}
        seed, dropped = ag.validate_seed(data, ctx, max_rows=20)
        assert seed["tables"] == {}
        assert any(d["reason"] == "row has no real column" for d in dropped)

    def test_no_tables_object(self):
        ctx = _seed_ctx()
        seed, dropped = ag.validate_seed({"nope": 1}, ctx, max_rows=20)
        assert seed == {"tables": {}}


@pytest.mark.asyncio
class TestGenerateSeedLLM:
    async def test_happy_path(self):
        gw = _FakeGateway(_good_seed_json())
        res = await ag.generate_seed_data_llm(_seed_ctx(), gateway=gw, budget_remaining_usd=1.0)
        assert res.source == "llm"
        assert res.row_count() == 2
        assert res.seed["tables"]["contacts"][0]["_verifies"] == "STORY-004.AC-1"

    async def test_dispatch_exception_fail_soft(self):
        gw = _FakeGateway(raise_exc=RuntimeError("x"))
        res = await ag.generate_seed_data_llm(_seed_ctx(), gateway=gw, budget_remaining_usd=1.0)
        assert res.seed == {"tables": {}}
        assert res.budget_remaining_usd == 1.0

    async def test_seed_prompt_carries_schema_and_acs(self):
        gw = _FakeGateway(_good_seed_json())
        await ag.generate_seed_data_llm(_seed_ctx(), gateway=gw, budget_remaining_usd=1.0)
        sent = gw.calls[0]["messages"][1]["content"]
        assert "CREATE TABLE contacts" in sent
        assert "STORY-004.AC-1" in sent
