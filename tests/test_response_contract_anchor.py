"""Acceptance scenarios must assert the response keys the API really returns.

``gather_story_acceptance_context`` gave the generator ``METHOD /path`` and
prose — never the response body's shape. So it invented envelope keys from the
criterion wording while ``test_generation`` read the real schema source, and
the two independently-invented shapes could not both be satisfied.

lumina 969f8e1c: ``tests/acceptance/test_story_001_acceptance.py`` asserted
``body['items']``; ``server/tests/test_main.py`` asserted ``"birthdays" in
body``. Renaming the response key to satisfy one broke the other, the
diagnostics oscillated, and the run died on its distraction budget.

Discovery then had to stop filtering by class NAME. lumina-run11-20260923-0040
returned to the same failure with the anchor in place: ``GET /api/birthdays``
declares ``response_model=BirthdayPage`` and that name matches no suffix in
``_RESPONSE_NAME_RE``, so the envelope was dropped, the generator saw only the
ITEM shape and asserted ``body[0]['first_name']`` against
``{"birthdays": [...], "total": ...}``. All 10 acceptance criteria failed and
were reported as production defects.
"""

from __future__ import annotations

from harness.acceptance_gen import (
    StoryAcceptanceContext,
    build_user_prompt,
    discover_response_models,
    discover_route_response_literals,
)


def _write(tmp_path, rel, body):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


class TestDiscovery:
    def test_finds_response_models_and_their_fields(self, tmp_path):
        _write(tmp_path, "app/schemas.py", (
            "from pydantic import BaseModel\n"
            "class BirthdayResponse(BaseModel):\n"
            "    id: int\n"
            "    first_name: str\n"
            "class BirthdayListResponse(BaseModel):\n"
            "    items: list[BirthdayResponse]\n"
        ))
        got = discover_response_models(str(tmp_path))
        assert got["BirthdayResponse"] == ["id", "first_name"]
        assert got["BirthdayListResponse"] == ["items"]

    def test_resolves_inherited_fields(self, tmp_path):
        # The real shape of `UpcomingBirthdayResponse(BirthdayResponse)`:
        # one declared field, four inherited.
        _write(tmp_path, "app/schemas.py", (
            "from pydantic import BaseModel\n"
            "class BirthdayResponse(BaseModel):\n"
            "    id: int\n"
            "    first_name: str\n"
            "class UpcomingBirthdayResponse(BirthdayResponse):\n"
            "    days_left: int\n"
        ))
        got = discover_response_models(str(tmp_path))
        assert got["UpcomingBirthdayResponse"] == ["id", "first_name", "days_left"]

    def test_module_level_code_is_not_mistaken_for_fields(self, tmp_path):
        # A regex slice running to the next class swept up trailing module
        # code and reported `try` as a response field.
        _write(tmp_path, "app/schemas.py", (
            "from pydantic import BaseModel\n"
            "class ThingResponse(BaseModel):\n"
            "    name: str\n"
            "\n"
            "def validate(value: str) -> str:\n"
            "    try:\n"
            "        return value\n"
            "    except ValueError:\n"
            "        raise\n"
        ))
        assert discover_response_models(str(tmp_path))["ThingResponse"] == ["name"]

    def test_non_response_models_are_excluded(self, tmp_path):
        _write(tmp_path, "app/schemas.py", (
            "from pydantic import BaseModel\n"
            "class BirthdayCreate(BaseModel):\n"
            "    first_name: str\n"
            "class BirthdayResponse(BaseModel):\n"
            "    id: int\n"
        ))
        got = discover_response_models(str(tmp_path))
        assert "BirthdayResponse" in got and "BirthdayCreate" not in got

    def test_test_dirs_and_unparseable_files_are_skipped(self, tmp_path):
        _write(tmp_path, "tests/schemas.py", (
            "from pydantic import BaseModel\n"
            "class LeakedResponse(BaseModel):\n    x: int\n"
        ))
        _write(tmp_path, "app/broken.py", "from pydantic import BaseModel\nclass (:\n")
        assert discover_response_models(str(tmp_path)) == {}

    def test_model_config_is_not_a_field(self, tmp_path):
        _write(tmp_path, "app/schemas.py", (
            "from pydantic import BaseModel, ConfigDict\n"
            "class ThingResponse(BaseModel):\n"
            "    model_config: ConfigDict = ConfigDict()\n"
            "    name: str\n"
        ))
        assert discover_response_models(str(tmp_path))["ThingResponse"] == ["name"]


class TestPromptCarriesTheContract:
    @staticmethod
    def _ctx(**over):
        base = dict(
            story_key="STORY-001", title="Directory", description="",
            acceptance_criteria=[{"ac_key": "STORY-001.AC-1", "text": "List all"}],
            routes=[{"method": "GET", "path": "/api/birthdays"}],
        )
        base.update(over)
        return StoryAcceptanceContext(**base)

    def test_real_keys_reach_the_prompt(self):
        p = build_user_prompt(
            self._ctx(response_models={"BirthdayListResponse": ["items"]}),
            max_scenarios=5,
        )
        assert "Response schemas (read from the code" in p
        assert "BirthdayListResponse: items" in p

    def test_prompt_forbids_inventing_an_envelope_key(self):
        p = build_user_prompt(
            self._ctx(response_models={"BirthdayListResponse": ["items"]}),
            max_scenarios=5,
        )
        assert "Do NOT invent a different envelope key" in p
        assert "contradict" in p

    def test_section_absent_when_nothing_discovered(self):
        p = build_user_prompt(self._ctx(), max_scenarios=5)
        assert "Response schemas" not in p


class TestRouteDeclaredModelsAreAlwaysIncluded:
    """What a route DECLARES beats what a class is called. The name pattern
    is the fallback for apps that declare nothing, not the test."""

    def test_the_run11_envelope_is_discovered(self, tmp_path):
        _write(tmp_path, "app/schemas.py", (
            "from pydantic import BaseModel\n"
            "class BirthdayResponse(BaseModel):\n"
            "    id: int\n"
            "    first_name: str\n"
            "class BirthdayPage(BaseModel):\n"
            "    birthdays: list[BirthdayResponse]\n"
            "    total: int\n"
            "    limit: int\n"
            "    offset: int\n"
        ))
        _write(tmp_path, "app/api/birthdays.py", (
            "from fastapi import APIRouter\n"
            "from app.schemas import BirthdayPage, BirthdayResponse\n"
            "router = APIRouter()\n"
            "@router.get('', response_model=BirthdayPage)\n"
            "def list_birthdays():\n"
            "    ...\n"
        ))
        got = discover_response_models(str(tmp_path))
        assert got["BirthdayPage"] == ["birthdays", "total", "limit", "offset"], (
            "the envelope the endpoint declares must reach the generator"
        )
        assert got["BirthdayResponse"] == ["id", "first_name"]

    def test_a_dotted_declaration_is_resolved(self, tmp_path):
        _write(tmp_path, "app/schemas.py", (
            "from pydantic import BaseModel\n"
            "class Page(BaseModel):\n    rows: list[str]\n"
        ))
        _write(tmp_path, "app/api.py", (
            "from fastapi import APIRouter\n"
            "from app import schemas\n"
            "router = APIRouter()\n"
            "@router.get('/x', response_model=schemas.Page)\n"
            "def x():\n    ...\n"
        ))
        assert discover_response_models(str(tmp_path))["Page"] == ["rows"]

    def test_a_container_declaration_is_unwrapped(self, tmp_path):
        _write(tmp_path, "app/schemas.py", (
            "from pydantic import BaseModel\n"
            "class Row(BaseModel):\n    name: str\n"
        ))
        _write(tmp_path, "app/api.py", (
            "from fastapi import APIRouter\n"
            "from app.schemas import Row\n"
            "router = APIRouter()\n"
            "@router.get('/x', response_model=list[Row])\n"
            "def x():\n    ...\n"
        ))
        assert discover_response_models(str(tmp_path))["Row"] == ["name"]

    def test_response_model_dict_declares_no_shape(self, tmp_path):
        # `response_model=dict` is a real pattern in the wild and names no
        # model; it must not invent one.
        _write(tmp_path, "app/schemas.py", (
            "from pydantic import BaseModel\n"
            "class Internal(BaseModel):\n    x: int\n"
        ))
        _write(tmp_path, "app/api.py", (
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            "@router.get('/x', response_model=dict)\n"
            "def x():\n    ...\n"
        ))
        assert discover_response_models(str(tmp_path)) == {}

    def test_undeclared_request_models_are_still_excluded(self, tmp_path):
        # The name filter still earns its keep: nothing declares these.
        _write(tmp_path, "app/schemas.py", (
            "from pydantic import BaseModel\n"
            "class BirthdayCreate(BaseModel):\n    first_name: str\n"
            "class ThingResponse(BaseModel):\n    id: int\n"
        ))
        got = discover_response_models(str(tmp_path))
        assert "ThingResponse" in got and "BirthdayCreate" not in got


class TestEndpointsThatDeclareNoShape:
    """``response_model=dict`` declares nothing, so there is no class to read
    — but the handler usually spells the envelope out in its return.

    lumina-run15-20260924-1423: `GET /api/birthdays/upcoming` declares
    ``dict`` and returns ``{"birthdays": [...]}``. With no shape for it the
    generator guessed a bare list and wrote
    ``[item['first_name'] for item in data]``, which iterates a dict's KEYS.
    Eight of ten acceptance criteria failed on it — while the sibling
    endpoint, which declares ``response_model=BirthdayPage``, passed.
    """

    ROUTER = (
        "from fastapi import APIRouter\n"
        "router = APIRouter(prefix='/api/birthdays')\n"
        "@router.get('/upcoming', response_model=dict)\n"
        "def get_upcoming():\n"
        "    return {'birthdays': [u.model_dump() for u in upcoming]}\n"
    )

    def test_the_run15_envelope_is_discovered(self, tmp_path):
        _write(tmp_path, "app/api.py", self.ROUTER)
        got = discover_route_response_literals(str(tmp_path))
        assert got == {"GET /api/birthdays/upcoming (returned body)": ["birthdays"]}

    def test_an_async_handler_counts(self, tmp_path):
        _write(tmp_path, "app/api.py", (
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            "@router.get('/x')\n"
            "async def x():\n"
            "    return {'items': [], 'total': 0}\n"
        ))
        got = discover_route_response_literals(str(tmp_path))
        assert got["GET /x (returned body)"] == ["items", "total"]

    def test_a_declared_model_is_left_to_the_schema_reader(self, tmp_path):
        # The declared model is the authority; reading the handler too could
        # contradict it.
        _write(tmp_path, "app/api.py", (
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            "@router.get('/x', response_model=BirthdayPage)\n"
            "def x():\n"
            "    return {'whatever': 1}\n"
        ))
        assert discover_route_response_literals(str(tmp_path)) == {}

    def test_a_handler_returning_a_model_instance_yields_nothing(self, tmp_path):
        _write(tmp_path, "app/api.py", (
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            "@router.get('/x')\n"
            "def x():\n"
            "    return BirthdayPage(birthdays=[], total=0)\n"
        ))
        assert discover_route_response_literals(str(tmp_path)) == {}

    def test_non_route_functions_are_ignored(self, tmp_path):
        _write(tmp_path, "app/util.py", (
            "def helper():\n"
            "    return {'not': 'a response'}\n"
        ))
        assert discover_route_response_literals(str(tmp_path)) == {}

    def test_tests_and_broken_files_are_skipped(self, tmp_path):
        _write(tmp_path, "tests/api.py", self.ROUTER)
        _write(tmp_path, "app/broken.py", "@router.get('/x')\ndef (:\n")
        assert discover_route_response_literals(str(tmp_path)) == {}

    def test_both_sources_reach_the_context(self, tmp_path):
        # The merge must not let a literal shadow a declared model, nor drop
        # the endpoints only a literal describes.
        _write(tmp_path, "app/schemas.py", (
            "from pydantic import BaseModel\n"
            "class BirthdayPage(BaseModel):\n"
            "    birthdays: list\n    total: int\n"
        ))
        _write(tmp_path, "app/api.py", self.ROUTER + (
            "@router.get('', response_model=BirthdayPage)\n"
            "def listing():\n"
            "    return page\n"
        ))
        merged = {
            **discover_route_response_literals(str(tmp_path)),
            **discover_response_models(str(tmp_path)),
        }
        assert merged["BirthdayPage"] == ["birthdays", "total"]
        assert merged["GET /api/birthdays/upcoming (returned body)"] == ["birthdays"]

    def test_the_prompt_shows_the_endpoint_shape(self, tmp_path):
        ctx = StoryAcceptanceContext(
            story_key="STORY-001", title="t", description="d",
            acceptance_criteria=[{"ac_key": "AC-1", "text": "shows names"}],
            routes=[{"method": "GET", "path": "/api/birthdays/upcoming"}],
            response_models={
                "GET /api/birthdays/upcoming (returned body)": ["birthdays"]},
            data_model_excerpt="", architecture_excerpt="", stack={},
        )
        prompt = build_user_prompt(ctx, max_scenarios=5)
        assert "GET /api/birthdays/upcoming (returned body): birthdays" in prompt


class TestCrossCuttingRulesReachTheGenerator:
    """lumina-run17-20260924-2351. The generator verified a leap-day criterion
    through `GET /api/birthdays/upcoming`, an endpoint with a 30-DAY window.
    A February birthday is correctly absent in September, `next(...)` raised
    StopIteration, and repair spent the run chasing a defect that did not
    exist. The rule — "Only birthdays with days_left from 0 through 30
    inclusive are shown" — is IN the spec and was not in the prompt.
    """

    SPEC = (
        "# Software Requirements Specification\n\n"
        "## API conventions\n\n"
        "Collections come back wrapped in an envelope.\n\n"
        "## Time and date semantics\n\n"
        "- `days_left` runs from 0 through 30 inclusive.\n\n"
        "#### Story: STORY-001 - Dashboard\n\n"
        "**Acceptance Criteria:**\n\n- Alice is listed first.\n"
    )

    def test_the_head_is_read_and_the_stories_are_not(self, tmp_path):
        from harness.acceptance_gen import _cross_cutting_rules
        _write(tmp_path, "docs/SPEC_REQUIREMENTS.md", self.SPEC)
        rules = _cross_cutting_rules(str(tmp_path))
        assert "days_left` runs from 0 through 30" in rules
        assert "Collections come back wrapped" in rules
        assert "STORY-001" not in rules, "story bodies belong to the story slice"

    def test_a_missing_spec_is_not_fatal(self, tmp_path):
        from harness.acceptance_gen import _cross_cutting_rules
        assert _cross_cutting_rules(str(tmp_path)) == ""

    def test_trimming_stops_at_a_section_boundary(self, tmp_path):
        # A mid-section cut silently drops the tail of a rule the generator
        # is being told to obey.
        from harness.acceptance_gen import _cross_cutting_rules
        _write(tmp_path, "docs/SPEC_REQUIREMENTS.md", self.SPEC)
        rules = _cross_cutting_rules(str(tmp_path), max_chars=80)
        assert rules.endswith("\n") or rules.strip()
        assert "## Time and date semantics" not in rules or "0 through 30" in rules

    def test_the_prompt_presents_them_as_binding(self, tmp_path):
        ctx = StoryAcceptanceContext(
            story_key="STORY-001", title="t", description="d",
            acceptance_criteria=[{"ac_key": "AC-1", "text": "leap day"}],
            routes=[], response_models={},
            cross_cutting_rules="- `days_left` runs from 0 through 30 inclusive.",
            data_model_excerpt="", architecture_excerpt="", stack={},
        )
        prompt = build_user_prompt(ctx, max_scenarios=5)
        assert "0 through 30 inclusive" in prompt
        assert "cannot pass" in prompt


class TestClockFreeze:
    """A criterion about dates is otherwise unverifiable except by accident of
    the calendar."""

    def _ws(self, tmp_path):
        _write(tmp_path, "app/services/clock.py", (
            "from datetime import date, datetime, timezone\n"
            "class Clock:\n"
            "    @staticmethod\n"
            "    def today_utc() -> date:\n"
            "        return datetime.now(timezone.utc).date()\n"
            "    @staticmethod\n"
            "    def now_utc() -> datetime:\n"
            "        return datetime.now(timezone.utc)\n"
        ))
        _write(tmp_path, "app/deps.py", (
            "from app.services.clock import Clock\n"
            "def get_clock() -> Clock:\n"
            "    return Clock()\n"
        ))
        return str(tmp_path)

    def test_discovers_the_dependency_and_its_accessors(self, tmp_path):
        from harness.acceptance_gen import discover_clock_override
        got = discover_clock_override(self._ws(tmp_path))
        assert got["symbol"] == "get_clock" and got["module"] == "app.deps"
        assert got["date_methods"] == ["today_utc"]
        assert got["datetime_methods"] == ["now_utc"]

    def test_no_clock_is_not_an_error(self, tmp_path):
        from harness.acceptance_gen import discover_clock_override
        _write(tmp_path, "app/main.py", "app = 1\n")
        assert discover_clock_override(str(tmp_path)) is None

    def test_the_conftest_exposes_freeze_today(self, tmp_path):
        import ast
        from harness.acceptance_gen import (
            discover_clock_override, render_acceptance_conftest)
        src = render_acceptance_conftest(
            {"module": "app.main", "symbol": "create_app", "kind": "factory"},
            clock=discover_clock_override(self._ws(tmp_path)))
        ast.parse(src)
        assert "def freeze_today(self, d):" in src
        assert "from app.deps import get_clock as _CLOCK_DEP" in src
        assert "dependency_overrides[_CLOCK_DEP]" in src
        # every discovered accessor is overridden, or the app reads real time
        assert "def today_utc(self):" in src and "def now_utc(self):" in src

    def test_no_clock_renders_no_freeze(self, tmp_path):
        import ast
        from harness.acceptance_gen import render_acceptance_conftest
        src = render_acceptance_conftest(
            {"module": "app.main", "symbol": "app", "kind": "singleton"},
            clock=None)
        ast.parse(src)
        assert "freeze_today" not in src and "_CLOCK_DEP" not in src

    def test_the_prompt_matches_what_the_conftest_provides(self):
        from harness.acceptance_gen import build_system_prompt
        with_clock = build_system_prompt(db_isolated=True, clock_available=True)
        without = build_system_prompt(db_isolated=True, clock_available=False)
        assert "client.freeze_today(" in with_clock
        assert "freeze_today" not in without
        assert "no injectable clock" in without
        # The slot must never leak into a prompt either way.
        assert "{clock_rule}" not in with_clock and "{clock_rule}" not in without
