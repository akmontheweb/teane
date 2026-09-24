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
