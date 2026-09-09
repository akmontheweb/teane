"""Acceptance scenarios must assert the response keys the API really returns.

``gather_story_acceptance_context`` gave the generator ``METHOD /path`` and
prose — never the response body's shape. So it invented envelope keys from the
criterion wording while ``test_generation`` read the real schema source, and
the two independently-invented shapes could not both be satisfied.

lumina 969f8e1c: ``tests/acceptance/test_story_001_acceptance.py`` asserted
``body['items']``; ``server/tests/test_main.py`` asserted ``"birthdays" in
body``. Renaming the response key to satisfy one broke the other, the
diagnostics oscillated, and the run died on its distraction budget.
"""

from __future__ import annotations

from harness.acceptance_gen import (
    StoryAcceptanceContext,
    build_user_prompt,
    discover_response_models,
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
