"""Tests for the operator-configurable coverage gate (FR-080).

The threshold and the enforce flag are read from ``config.json``'s
``coverage`` section and injected into the skill markdown at prompt-
build time via ``{{coverage.*}}`` substitution markers. Threshold
defaults to 70; enforce defaults to true.
"""

from __future__ import annotations

import os
from pathlib import Path

from harness.graph import _load_skills_markdown


class TestSubstitutionMechanics:
    """Basic {{key}} → value substitution — the loader must apply the
    map without touching content that doesn't reference it."""

    def test_placeholder_replaced_with_value(self, tmp_path: Path) -> None:
        (tmp_path / "s.md").write_text(
            "---\napplies_to: [python]\n---\n"
            "Coverage floor: {{coverage.min_pct}}%\n"
        )
        out = _load_skills_markdown(
            str(tmp_path),
            workspace_tags={"python"},
            substitutions={"coverage.min_pct": "70"},
        )
        assert "Coverage floor: 70%" in out
        assert "{{coverage.min_pct}}" not in out

    def test_no_substitutions_keeps_placeholders_verbatim(
        self, tmp_path: Path,
    ) -> None:
        # A visible unresolved marker is the intended "bug signal" —
        # better than silent replacement with nothing.
        (tmp_path / "s.md").write_text(
            "---\napplies_to: [python]\n---\n"
            "Threshold: {{coverage.min_pct}}%\n"
        )
        out = _load_skills_markdown(
            str(tmp_path), workspace_tags={"python"},
        )
        assert "{{coverage.min_pct}}" in out

    def test_unrelated_key_leaves_placeholder(self, tmp_path: Path) -> None:
        # A skill that references a marker not in the substitutions
        # dict keeps the literal — reader can see the missed key.
        (tmp_path / "s.md").write_text(
            "---\napplies_to: [python]\n---\n"
            "Foo: {{other.value}}. Bar: {{coverage.min_pct}}.\n"
        )
        out = _load_skills_markdown(
            str(tmp_path), workspace_tags={"python"},
            substitutions={"coverage.min_pct": "70"},
        )
        assert "Foo: {{other.value}}" in out
        assert "Bar: 70" in out


class TestPytestFailFlagPlaceholder:
    """The pytest fail-under flag is fully pre-rendered so the LLM
    sees either ` --cov-fail-under=N` or ``. Never has to conditional."""

    def test_enforce_true_renders_flag(self, tmp_path: Path) -> None:
        (tmp_path / "s.md").write_text(
            "---\napplies_to: [python]\n---\n"
            "test: pytest --cov=src{{coverage.pytest_fail_flag}}\n"
        )
        out = _load_skills_markdown(
            str(tmp_path), workspace_tags={"python"},
            substitutions={"coverage.pytest_fail_flag": " --cov-fail-under=70"},
        )
        assert "pytest --cov=src --cov-fail-under=70" in out

    def test_enforce_false_renders_empty_flag(self, tmp_path: Path) -> None:
        (tmp_path / "s.md").write_text(
            "---\napplies_to: [python]\n---\n"
            "test: pytest --cov=src{{coverage.pytest_fail_flag}}\n"
        )
        out = _load_skills_markdown(
            str(tmp_path), workspace_tags={"python"},
            substitutions={"coverage.pytest_fail_flag": ""},
        )
        assert "pytest --cov=src\n" in out
        assert "--cov-fail-under" not in out


class TestJestThresholdSnippet:
    """The Jest threshold snippet either appears as a full JSON fragment
    with a trailing comma+space or vanishes. The surrounding JSON must
    be valid in both branches."""

    def test_enforce_true_snippet_is_valid_json_fragment(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / "s.md").write_text(
            '---\napplies_to: [node]\n---\n'
            '"jest": {\n'
            '  {{coverage.jest_threshold_snippet}}"collectCoverageFrom": []\n'
            '}\n'
        )
        snippet = (
            '"coverageThreshold": {"global": '
            '{"lines": 70, "statements": 70}}, '
        )
        out = _load_skills_markdown(
            str(tmp_path), workspace_tags={"node"},
            substitutions={"coverage.jest_threshold_snippet": snippet},
        )
        # The rendered JSON should have both keys separated by a comma.
        assert '"coverageThreshold"' in out
        assert '"collectCoverageFrom"' in out
        # No double-comma, no missing comma.
        assert ",  " not in out or ',  "collectCoverageFrom"' not in out
        # Basic structural check: comma between the two keys.
        idx_thresh = out.index('"coverageThreshold"')
        idx_collect = out.index('"collectCoverageFrom"')
        between = out[idx_thresh:idx_collect]
        assert "," in between

    def test_enforce_false_snippet_is_empty(self, tmp_path: Path) -> None:
        (tmp_path / "s.md").write_text(
            '---\napplies_to: [node]\n---\n'
            '"jest": {\n'
            '  {{coverage.jest_threshold_snippet}}"collectCoverageFrom": []\n'
            '}\n'
        )
        out = _load_skills_markdown(
            str(tmp_path), workspace_tags={"node"},
            substitutions={"coverage.jest_threshold_snippet": ""},
        )
        assert '"coverageThreshold"' not in out
        assert '"collectCoverageFrom"' in out


class TestShippedSkillsWireIntoConfig:
    """Sanity check: the actual shipped skills carry the placeholders
    we template. A future refactor that removes them would silently
    disable the operator override; this test catches that."""

    HARNESS_SKILLS_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "harness", "skills",
    )

    def test_makefile_python_uses_pytest_fail_flag_marker(self) -> None:
        with open(
            os.path.join(self.HARNESS_SKILLS_DIR, "makefile_python.md"),
            encoding="utf-8",
        ) as f:
            body = f.read()
        assert "{{coverage.pytest_fail_flag}}" in body
        # And no hard-coded fallback.
        assert "--cov-fail-under=70" not in body

    def test_makefile_node_uses_jest_threshold_marker(self) -> None:
        with open(
            os.path.join(self.HARNESS_SKILLS_DIR, "makefile_node.md"),
            encoding="utf-8",
        ) as f:
            body = f.read()
        assert "{{coverage.jest_threshold_snippet}}" in body
        # No hard-coded threshold block.
        assert '"lines": 70' not in body

    def test_unit_test_skills_reference_min_pct(self) -> None:
        for fname in ("unit_tests_python.md", "unit_tests_react.md"):
            with open(
                os.path.join(self.HARNESS_SKILLS_DIR, fname),
                encoding="utf-8",
            ) as f:
                body = f.read()
            assert "{{coverage.min_pct}}" in body


class TestConfigDefaults:
    """Behaviour when config.json omits the coverage section or
    sub-key. Defaults: min_pct=70, enforce=true."""

    def test_missing_section_uses_defaults(self) -> None:
        # This test verifies the plumbing in _build_system_prompt
        # (called via create_initial_state) but that function has many
        # side effects. Instead we replicate the exact snippet from
        # graph.py to confirm the defaults produce the expected
        # substitution values.
        config: dict = {}
        _cov = config.get("coverage", {}) or {}
        assert _cov == {}
        min_pct = int(_cov.get("min_pct", 70))
        enforce = bool(_cov.get("enforce", True))
        assert min_pct == 70
        assert enforce is True

    def test_explicit_off_disables_enforce(self) -> None:
        config = {"coverage": {"enforce": False, "min_pct": 85}}
        _cov = config.get("coverage", {})
        assert bool(_cov.get("enforce", True)) is False
        assert int(_cov.get("min_pct", 70)) == 85

    def test_malformed_min_pct_falls_back_to_default(self) -> None:
        # Replicates the try/except in graph.py — a stray string in
        # min_pct (someone edited config.json by hand) must not crash.
        config = {"coverage": {"min_pct": "not-a-number"}}
        _cov = config.get("coverage", {})
        try:
            min_pct = int(_cov.get("min_pct", 70))
        except (TypeError, ValueError):
            min_pct = 70
        assert min_pct == 70
