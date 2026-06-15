"""Tests for language-aware skill filtering in harness/graph.py.

Covers `_parse_skill_frontmatter`, the filter branch in
`_load_skills_markdown`, and end-to-end skill selection via
`_build_system_prompt` for representative workspaces.
"""

import json
import os
import tempfile


from harness.graph import (
    _build_system_prompt,
    _load_skills_markdown,
    _parse_skill_frontmatter,
)


class TestParseSkillFrontmatter:
    def test_no_frontmatter_returns_none_and_full_body(self):
        body = "## Some skill\n\nContent here.\n"
        tags, parsed = _parse_skill_frontmatter(body)
        assert tags is None
        assert parsed == body

    def test_applies_to_list_parsed(self):
        content = "---\napplies_to: [fastapi, django]\n---\n\n## Python\nBody.\n"
        tags, body = _parse_skill_frontmatter(content)
        assert tags == {"fastapi", "django"}
        # Regex consumes the blank line separator; body starts at content.
        assert body == "## Python\nBody.\n"

    def test_single_tag_parsed(self):
        content = "---\napplies_to: [react]\n---\n\n## React\n"
        tags, body = _parse_skill_frontmatter(content)
        assert tags == {"react"}
        assert "## React" in body

    def test_empty_brackets_yields_empty_set(self):
        content = "---\napplies_to: []\n---\n\nbody"
        tags, body = _parse_skill_frontmatter(content)
        assert tags == set()
        assert body == "body"

    def test_non_frontmatter_dashes_not_consumed(self):
        # Real markdown sometimes starts with a horizontal rule. The
        # parser must NOT treat that as frontmatter.
        body = "---\n\nNo `applies_to:` field here.\n"
        tags, parsed = _parse_skill_frontmatter(body)
        assert tags is None
        assert parsed == body


class TestSkillsLoaderFilter:
    def _write_skill(self, dir_path, fname, applies_to, body_marker):
        if applies_to is None:
            content = f"## {body_marker}\nBody here.\n"
        else:
            tag_list = ", ".join(applies_to)
            content = f"---\napplies_to: [{tag_list}]\n---\n\n## {body_marker}\nBody here.\n"
        with open(os.path.join(dir_path, fname), "w") as f:
            f.write(content)

    def test_no_workspace_tags_loads_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_skill(tmp, "a.md", ["fastapi"], "A")
            self._write_skill(tmp, "b.md", ["vue"], "B")
            self._write_skill(tmp, "c.md", None, "C")
            out = _load_skills_markdown(tmp, workspace_tags=None)
            assert "## A" in out
            assert "## B" in out
            assert "## C" in out

    def test_matching_tag_loads_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_skill(tmp, "fastapi.md", ["fastapi"], "FastAPI")
            self._write_skill(tmp, "vue.md", ["vue"], "Vue")
            out = _load_skills_markdown(tmp, workspace_tags={"fastapi", "python"})
            assert "## FastAPI" in out
            assert "## Vue" not in out

    def test_no_match_excludes_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_skill(tmp, "angular.md", ["angular"], "Angular")
            out = _load_skills_markdown(tmp, workspace_tags={"fastapi"})
            assert out.strip() == ""

    def test_universal_skill_always_loads(self):
        # A skill with no frontmatter (like agent-standards.md) must load
        # even when workspace_tags is narrow.
        with tempfile.TemporaryDirectory() as tmp:
            self._write_skill(tmp, "standards.md", None, "Standards")
            self._write_skill(tmp, "vue.md", ["vue"], "Vue")
            out = _load_skills_markdown(tmp, workspace_tags={"fastapi"})
            assert "## Standards" in out
            assert "## Vue" not in out

    def test_multi_tag_skill_loads_if_any_matches(self):
        # postgresql_redis_mysql.md declares [postgres, redis, mysql]
        # — any one match should be enough.
        with tempfile.TemporaryDirectory() as tmp:
            self._write_skill(tmp, "db.md", ["postgres", "redis", "mysql"], "DB")
            out = _load_skills_markdown(tmp, workspace_tags={"node", "redis"})
            assert "## DB" in out

    def test_frontmatter_stripped_from_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_skill(tmp, "x.md", ["fastapi"], "X")
            out = _load_skills_markdown(tmp, workspace_tags={"fastapi"})
            assert "applies_to" not in out
            assert "---" not in out.split("## X")[0] or out.split("## X")[0].strip() == ""


class TestBuildSystemPromptIntegration:
    """The end-to-end test: confirm that a real workspace shape gets
    the right subset of shipped skills in its system prompt."""

    def test_fastapi_workspace_excludes_irrelevant_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "pyproject.toml"), "w") as f:
                f.write('dependencies = ["fastapi>=0.115", "psycopg[binary]"]\n')
            os.makedirs(os.path.join(tmp, "app"))
            with open(os.path.join(tmp, "app", "main.py"), "w") as f:
                f.write("from fastapi import FastAPI\napp = FastAPI()\n")

            prompt = _build_system_prompt(tmp, "pytest")
            # Should include FastAPI + DB skills (postgres detected)
            assert "Python — FastAPI" in prompt
            assert "Databases — PostgreSQL" in prompt
            # Should EXCLUDE every other framework skill
            assert "Python — Django" not in prompt
            assert "Java — Spring Boot" not in prompt
            assert "Node.js — Express" not in prompt
            assert "Frontend — React" not in prompt
            assert "Frontend — Vue" not in prompt
            assert "Frontend — Angular" not in prompt
            assert "Mobile — Flutter" not in prompt
            # Universal skill always loads
            assert "Agent Standards" in prompt

    def test_flutter_workspace_only_loads_flutter_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "pubspec.yaml"), "w") as f:
                f.write("name: my_app\n")
            os.makedirs(os.path.join(tmp, "lib"))

            prompt = _build_system_prompt(tmp, "flutter test")
            assert "Mobile — Flutter" in prompt
            # No backend frameworks should appear
            assert "Python — FastAPI" not in prompt
            assert "Java — Spring Boot" not in prompt
            assert "Node.js — Express" not in prompt
            # Universal still loads
            assert "Agent Standards" in prompt

    def test_react_workspace_only_loads_react_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "package.json"), "w") as f:
                json.dump({"dependencies": {"react": "^18", "react-dom": "^18"}}, f)

            prompt = _build_system_prompt(tmp, "npm test")
            assert "Frontend — React" in prompt
            assert "Frontend — Vue" not in prompt
            assert "Frontend — Angular" not in prompt
            assert "Java — Spring Boot" not in prompt

    def test_empty_workspace_loads_only_universal_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt = _build_system_prompt(tmp, "true")
            assert "Agent Standards" in prompt
            # No stack-specific skill should appear
            assert "Python — FastAPI" not in prompt
            assert "Mobile — Flutter" not in prompt
            assert "Frontend — React" not in prompt

    def test_static_web_workspace_loads_web_skills(self):
        """Regression: the ticktaktoe shape (index.html + jest tests, no
        package.json or framework) must pull in both the web-app asset
        contract and the static-web Makefile skill. The empty Makefile +
        missing CSS bug came from neither of these skills being loaded."""
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "index.html"), "w") as f:
                f.write(
                    '<!DOCTYPE html><html><head>'
                    '<link rel="stylesheet" href="style.css">'
                    '</head><body></body></html>'
                )
            os.makedirs(os.path.join(tmp, "tests"))
            with open(os.path.join(tmp, "tests", "x.test.js"), "w") as f:
                f.write("test('x', () => expect(1).toBe(1));\n")

            prompt = _build_system_prompt(tmp, "make test")
            # Both Layer 2 (preventive contract) and Layer 3 (Makefile)
            # skills must load for this shape — the upstream root cause of
            # the empty `build: @echo` Makefile was that neither did.
            assert "Web App — Asset Reference Contract" in prompt
            assert "Build — Static Web App Makefile" in prompt
            # Should NOT load framework-specific skills.
            assert "Frontend — React" not in prompt
            assert "Frontend — Vue" not in prompt
            # Should NOT load the node Makefile skill (no package.json).
            assert "Build — Node.js / TypeScript Makefile" not in prompt
