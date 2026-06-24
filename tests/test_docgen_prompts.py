"""Tests for the docgen prompt loader and shipped prompt content.

The discovery nodes in ``harness.graph`` and the standalone CLI doc-gen
skills in ``harness.skills`` both read their system prompts from
``harness/skills/docgen/*.md`` via ``harness.docgen_prompts.load``. These
tests guard:

  - The loader returns each shipped prompt non-empty.
  - The loader raises FileNotFoundError on unknown names (no silent
    fallback to a truncated default — a missing discovery prompt would
    produce an empty interview screen).
  - The discovery prompts preserve the canonical JSON output schema
    (top-level ``modules`` key) the harness's discovery parser
    hard-requires.
  - The follow-up prompts contain the ``{ROUND_NUMBER}`` placeholder the
    callers substitute via ``str.replace``.
  - The per-workspace override directory wins when a file with the same
    name exists there.
"""
from __future__ import annotations

import pytest

from harness import docgen_prompts


SHIPPED_PROMPTS = [
    "requirements_discovery",
    "requirements_discovery_followup",
    "architecture_discovery",
    "architecture_discovery_followup",
    "requirements_doc",
    "arch_doc",
]


@pytest.fixture(autouse=True)
def _clear_cache():
    docgen_prompts.clear_cache()
    yield
    docgen_prompts.clear_cache()


@pytest.mark.parametrize("name", SHIPPED_PROMPTS)
def test_shipped_prompt_loads_non_empty(name):
    body = docgen_prompts.load(name)
    assert body.strip(), f"Shipped docgen prompt '{name}' is empty."
    # A few hundred bytes minimum — these are deliberately verbose
    # specifications. A tiny body almost certainly means truncation.
    assert len(body) > 500, f"Shipped docgen prompt '{name}' suspiciously short ({len(body)} chars)."


def test_unknown_name_raises():
    with pytest.raises(FileNotFoundError):
        docgen_prompts.load("does_not_exist")


def test_loader_caches_result(tmp_path):
    # Use a per-workspace override so we can mutate the file underneath
    # and confirm the cache returns the original.
    override_dir = tmp_path / "skills" / "docgen"
    override_dir.mkdir(parents=True)
    f = override_dir / "fixture_prompt.md"
    f.write_text("initial body")

    first = docgen_prompts.load("fixture_prompt", workspace_path=str(tmp_path))
    assert first == "initial body"

    f.write_text("mutated body")
    cached = docgen_prompts.load("fixture_prompt", workspace_path=str(tmp_path))
    assert cached == "initial body", "Loader did not cache the prompt body."

    docgen_prompts.clear_cache()
    fresh = docgen_prompts.load("fixture_prompt", workspace_path=str(tmp_path))
    assert fresh == "mutated body", "clear_cache() did not invalidate the entry."


def test_workspace_override_wins_over_shipped(tmp_path):
    # If a workspace ships its own requirements_discovery.md, it must
    # replace the harness default — that's the per-project override
    # contract the skills system already uses.
    override_dir = tmp_path / "skills" / "docgen"
    override_dir.mkdir(parents=True)
    custom = "WORKSPACE OVERRIDE BODY"
    (override_dir / "requirements_discovery.md").write_text(custom)

    body = docgen_prompts.load("requirements_discovery", workspace_path=str(tmp_path))
    assert body == custom


@pytest.mark.parametrize(
    "name",
    [
        "requirements_discovery",
        "requirements_discovery_followup",
        "architecture_discovery",
        "architecture_discovery_followup",
    ],
)
def test_discovery_prompt_specifies_canonical_modules_key(name):
    # The discovery JSON parser in harness/trust.py only accepts a
    # top-level "modules" key. A drift to "sectors" / "components" would
    # silently yield an empty interview screen — guard against that by
    # checking the shipped prompts still spell out the canonical key.
    body = docgen_prompts.load(name)
    assert '"modules"' in body, (
        f"Discovery prompt '{name}' no longer instructs the LLM to use the "
        f'canonical "modules" top-level key. The discovery parser will '
        f"return zero questions."
    )


@pytest.mark.parametrize(
    "name",
    ["requirements_discovery_followup", "architecture_discovery_followup"],
)
def test_followup_prompt_has_round_number_placeholder(name):
    # Callers in harness/graph.py substitute {ROUND_NUMBER} via
    # str.replace. If the placeholder is removed by mistake the prompt
    # still works, but if it's misspelled to something other than what
    # the caller substitutes, the literal placeholder leaks into the
    # LLM's input — guard against that drift.
    body = docgen_prompts.load(name)
    assert "{ROUND_NUMBER}" in body, (
        f"Follow-up prompt '{name}' lost its {{ROUND_NUMBER}} placeholder; "
        f"callers in harness/graph.py expect it for substitution."
    )


@pytest.mark.parametrize(
    "name",
    ["requirements_discovery_followup", "architecture_discovery_followup"],
)
def test_followup_prompt_has_focus_block_placeholder(name):
    # The discovery nodes substitute {FOCUS_SECTORS_BLOCK} with either
    # an LLM-picked focus block or an empty string. Drift in the
    # placeholder name (case, underscore vs dash) would cause the
    # literal placeholder to leak into the LLM's input — guard against
    # that.
    body = docgen_prompts.load(name)
    assert "{FOCUS_SECTORS_BLOCK}" in body, (
        f"Follow-up prompt '{name}' lost its {{FOCUS_SECTORS_BLOCK}} "
        f"placeholder; the discovery nodes in harness/graph.py expect "
        f"it for focus-block substitution."
    )


def test_skills_get_docgen_prompt_routes_externalized_types():
    # The skills.py helper must route requirements + arch_doc through
    # the loader (disk content), and fall back to the inline dict for
    # the others (e.g. readme).
    from harness import skills

    req_prompt = skills._get_docgen_prompt("requirements")
    arch_prompt = skills._get_docgen_prompt("arch_doc")
    readme_prompt = skills._get_docgen_prompt("readme")

    assert "Requirements Specification" in req_prompt
    assert "Architecture Document" in arch_prompt or "Architecture" in arch_prompt
    # readme still comes from the inline dict
    assert "README.md" in readme_prompt
