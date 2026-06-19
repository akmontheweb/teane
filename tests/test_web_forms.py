"""Regression tests for the form-schema derivation (harness/web_forms.py).

Covers:
    - ``kind_for_type_tuple`` maps the validator's type tuples to the
      right form widget kinds (bool→checkbox, int/float→number, etc.).
    - ``build_section`` pulls field names + types from the live
      validator tables (no hardcoding).
    - ``parse_value`` coerces form-string input to the typed values
      the validator expects and raises FormParseError on bad input.
    - ``parse_section_post`` round-trips a section through POST.
    - Coverage gate: every dotted key in ``cli._TYPE_SCHEMA`` is
      renderable, OR is intentionally omitted (top-level scalars
      without a section).
"""

from __future__ import annotations

import pytest

from harness.web_forms import (
    FORM_KIND_CHECKBOX,
    FORM_KIND_JSON_DICT,
    FORM_KIND_JSON_LIST,
    FORM_KIND_NUMBER_FLOAT,
    FORM_KIND_NUMBER_INT,
    FORM_KIND_TEXT,
    FormField,
    FormParseError,
    all_sections,
    build_section,
    kind_for_type_tuple,
    parse_section_post,
    parse_value,
    renderable_dotted_keys,
)


# ---------------------------------------------------------------------------
# 1. kind_for_type_tuple
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("type_tuple,expected_kind", [
    ((bool,), FORM_KIND_CHECKBOX),
    ((int,), FORM_KIND_NUMBER_INT),
    ((int, float), FORM_KIND_NUMBER_FLOAT),
    ((float,), FORM_KIND_NUMBER_FLOAT),
    ((str,), FORM_KIND_TEXT),
    ((list,), FORM_KIND_JSON_LIST),
    ((dict,), FORM_KIND_JSON_DICT),
    ((), FORM_KIND_TEXT),
])
def test_kind_for_type_tuple(type_tuple, expected_kind):
    assert kind_for_type_tuple(type_tuple) == expected_kind


def test_kind_bool_wins_over_int():
    # Python's bool IS-A int — make sure we don't mis-render bools as
    # numeric inputs.
    assert kind_for_type_tuple((bool, int)) == FORM_KIND_CHECKBOX


# ---------------------------------------------------------------------------
# 2. build_section
# ---------------------------------------------------------------------------

def test_build_section_pulls_from_validator():
    section = build_section("token_budget", current_config={
        "token_budget": {"hard_cap_usd": 3.0, "context_window_threshold_pct": 0.85},
    })
    assert section.section == "token_budget"
    names = {f.name for f in section.fields}
    assert "hard_cap_usd" in names
    assert "context_window_threshold_pct" in names
    hard_cap = next(f for f in section.fields if f.name == "hard_cap_usd")
    assert hard_cap.kind == FORM_KIND_NUMBER_FLOAT
    assert hard_cap.current_value == 3.0


def test_build_section_renders_checkboxes_for_bools():
    section = build_section("debug", current_config={
        "debug": {"dump_llm_calls": True, "dump_max_files": 5000},
    })
    bool_field = next(f for f in section.fields if f.name == "dump_llm_calls")
    int_field = next(f for f in section.fields if f.name == "dump_max_files")
    assert bool_field.kind == FORM_KIND_CHECKBOX
    assert bool_field.current_value is True
    assert int_field.kind == FORM_KIND_NUMBER_INT
    assert int_field.current_value == 5000


def test_build_section_unknown_section_returns_empty():
    section = build_section("definitely-not-a-section")
    assert section.section == "definitely-not-a-section"
    assert section.fields == []


def test_build_section_missing_current_config_uses_none_values():
    section = build_section("sandbox")
    assert section.fields  # has fields
    assert all(f.current_value is None for f in section.fields)


# ---------------------------------------------------------------------------
# 3. parse_value — happy paths + error paths
# ---------------------------------------------------------------------------

def _f(name: str, kind: str) -> FormField:
    type_map = {
        FORM_KIND_CHECKBOX: (bool,),
        FORM_KIND_NUMBER_INT: (int,),
        FORM_KIND_NUMBER_FLOAT: (int, float),
        FORM_KIND_TEXT: (str,),
        FORM_KIND_JSON_LIST: (list,),
        FORM_KIND_JSON_DICT: (dict,),
    }
    return FormField(section="x", name=name, kind=kind, type_tuple=type_map[kind])


def test_parse_value_checkbox_absent_is_false():
    assert parse_value(_f("x", FORM_KIND_CHECKBOX), None) is False


def test_parse_value_checkbox_truthy_strings():
    assert parse_value(_f("x", FORM_KIND_CHECKBOX), "on") is True
    assert parse_value(_f("x", FORM_KIND_CHECKBOX), "true") is True
    assert parse_value(_f("x", FORM_KIND_CHECKBOX), "1") is True
    assert parse_value(_f("x", FORM_KIND_CHECKBOX), "no") is False


def test_parse_value_number_int_valid():
    assert parse_value(_f("x", FORM_KIND_NUMBER_INT), "42") == 42
    assert parse_value(_f("x", FORM_KIND_NUMBER_INT), "  -7 ") == -7


def test_parse_value_number_int_rejects_garbage():
    with pytest.raises(FormParseError):
        parse_value(_f("x", FORM_KIND_NUMBER_INT), "abc")
    with pytest.raises(FormParseError):
        parse_value(_f("x", FORM_KIND_NUMBER_INT), "")


def test_parse_value_number_float_valid():
    assert parse_value(_f("x", FORM_KIND_NUMBER_FLOAT), "3.14") == pytest.approx(3.14)
    # Whole-number string still parses as float for the float field.
    assert parse_value(_f("x", FORM_KIND_NUMBER_FLOAT), "5") == 5.0


def test_parse_value_text_passes_through():
    assert parse_value(_f("x", FORM_KIND_TEXT), "some text") == "some text"
    assert parse_value(_f("x", FORM_KIND_TEXT), None) == ""


def test_parse_value_json_list_round_trips():
    assert parse_value(_f("x", FORM_KIND_JSON_LIST), '["a", "b"]') == ["a", "b"]
    assert parse_value(_f("x", FORM_KIND_JSON_LIST), "") == []
    assert parse_value(_f("x", FORM_KIND_JSON_LIST), "   ") == []


def test_parse_value_json_list_rejects_non_list_json():
    with pytest.raises(FormParseError, match="must be a list"):
        parse_value(_f("x", FORM_KIND_JSON_LIST), '{"not": "a list"}')


def test_parse_value_json_dict_round_trips():
    assert parse_value(_f("x", FORM_KIND_JSON_DICT), '{"a": 1}') == {"a": 1}
    assert parse_value(_f("x", FORM_KIND_JSON_DICT), "") == {}


def test_parse_value_json_dict_rejects_non_object_json():
    with pytest.raises(FormParseError, match="must be an object"):
        parse_value(_f("x", FORM_KIND_JSON_DICT), '[1, 2]')


# ---------------------------------------------------------------------------
# 4. parse_section_post — multi-field round trip
# ---------------------------------------------------------------------------

def test_parse_section_post_round_trip():
    section = build_section("token_budget", current_config={
        "token_budget": {"hard_cap_usd": 3.0, "context_window_threshold_pct": 0.85},
    })
    parsed, errors = parse_section_post(section, {
        "token_budget.hard_cap_usd": "5.0",
        "token_budget.context_window_threshold_pct": "0.9",
        "token_budget.stages": '{"planning": 0.2, "patching": 0.3}',
    })
    assert errors == []
    assert parsed["hard_cap_usd"] == pytest.approx(5.0)
    assert parsed["context_window_threshold_pct"] == pytest.approx(0.9)
    assert parsed["stages"] == {"planning": 0.2, "patching": 0.3}


def test_parse_section_post_collects_errors_but_returns_dict():
    section = build_section("token_budget")
    parsed, errors = parse_section_post(section, {
        "token_budget.hard_cap_usd": "not a number",
        "token_budget.context_window_threshold_pct": "0.85",
    })
    assert len(errors) == 1
    assert errors[0].dotted_key == "token_budget.hard_cap_usd"
    # The good field is still in the dict so the operator doesn't lose it.
    assert parsed["context_window_threshold_pct"] == pytest.approx(0.85)


def test_parse_section_post_missing_checkbox_yields_false():
    section = build_section("debug")
    # POST body omits the checkbox entirely — that's how HTML behaves
    # when the checkbox is unchecked.
    parsed, errors = parse_section_post(section, {
        "debug.dump_max_files": "3000",
    })
    assert errors == []
    assert parsed["dump_llm_calls"] is False
    assert parsed["dump_max_files"] == 3000


# ---------------------------------------------------------------------------
# 5. Coverage gate — every dotted key in cli._TYPE_SCHEMA is renderable
# ---------------------------------------------------------------------------

def test_every_validator_key_is_renderable():
    """Drift detector: if someone lands a new key in cli._TYPE_SCHEMA
    without also covering it in the form schema (almost always:
    forgetting to add the key to _KNOWN_NESTED_KEYS), this test fires."""
    from harness.cli import _TYPE_SCHEMA, _KNOWN_TOP_LEVEL_KEYS
    typed = set(_TYPE_SCHEMA.keys())
    renderable = renderable_dotted_keys()

    # The TYPE_SCHEMA contains some keys that are intentionally not
    # rendered through the generic editor — typically top-level scalar
    # keys with a separate dedicated UI (models, model_routing) or
    # deprecated aliases. Allowlist those explicitly so the rest of the
    # schema is still gated.
    intentional_omissions: set[str] = set()
    # Reserved slot for top-level scalars rendered by a dedicated UI.
    missing = typed - renderable - intentional_omissions

    # Top-level scalars (e.g. "build_command", "allow_network",
    # "product_spec_dir") DO appear in _TYPE_SCHEMA but their section
    # name is empty. The form schema renders them under a single-field
    # section keyed by the top-level name; their renderable form is
    # the bare name without a dot. Filter those out before asserting.
    top_level_scalars = {
        key for key in missing
        if "." not in key and key in _KNOWN_TOP_LEVEL_KEYS
    }
    # And these scalars must be renderable under their bare name:
    for scalar in top_level_scalars:
        assert scalar in renderable, (
            f"top-level scalar {scalar!r} is in _TYPE_SCHEMA but isn't "
            f"renderable through the form schema"
        )
    missing -= top_level_scalars

    assert not missing, (
        f"Validator keys without form-schema coverage: {sorted(missing)}. "
        f"Add them to _KNOWN_NESTED_KEYS[<section>] in harness/cli.py "
        f"so the dashboard's config editor can render them."
    )


def test_all_sections_iterates_in_sorted_order():
    sections = all_sections()
    names = [s.section for s in sections]
    assert names == sorted(names)


def test_all_sections_returns_known_sections():
    from harness.cli import _KNOWN_TOP_LEVEL_KEYS
    sections = all_sections()
    section_names = {s.section for s in sections}
    # Every top-level section we know about appears (even if empty).
    assert _KNOWN_TOP_LEVEL_KEYS <= section_names


# ---------------------------------------------------------------------------
# 6. Run-harness CLI flag form
# ---------------------------------------------------------------------------

def test_run_flags_covers_run_parser_bools():
    """The Run page surfaces every bool-choice flag the `harness run`
    argparse surface accepts (besides workspace + prompt, which have
    dedicated inputs). Text/integer flags stay on the terminal."""
    from harness.web_forms import run_flags
    names = {f.name for f in run_flags()}
    assert names == {
        "git",
        "new_build",
        "spec_discovery",
        "deploy_dev",
        "cd_discovery",
        "hitl_requirement",
        "hitl_architecture",
        "hitl_repair",
        "hitl_deployment",
    }


def test_build_run_argv_empty_form_produces_no_args():
    """A wholly-default form should not emit any CLI flag, so the
    harness's own defaults take over."""
    from harness.web_forms import build_run_argv_from_form
    argv, errors = build_run_argv_from_form({})
    assert errors == []
    assert argv == []


def test_build_run_argv_spec_discovery_true_emits_flag():
    from harness.web_forms import build_run_argv_from_form
    argv, errors = build_run_argv_from_form({"flag.spec_discovery": "true"})
    assert errors == []
    assert "--spec-discovery=true" in argv


def test_build_run_argv_spec_discovery_false_omits_flag():
    from harness.web_forms import build_run_argv_from_form
    argv, errors = build_run_argv_from_form({"flag.spec_discovery": "false"})
    assert errors == []
    # The CLI already defaults to false, so emitting the token would be noise.
    assert not any(a.startswith("--spec-discovery") for a in argv)


def test_build_run_argv_select_emits_flag_equals_value():
    from harness.web_forms import build_run_argv_from_form
    argv, errors = build_run_argv_from_form({
        "flag.new_build": "true",
        "flag.git": "true",
    })
    assert errors == []
    assert "--new-build=true" in argv
    assert "--git=true" in argv


def test_build_run_argv_hitl_flags_at_default_emit_nothing():
    # Default is now "true" for every --hitl-* flag; an operator who
    # leaves them alone should NOT emit the token. The CLI argparse
    # default (None) then defers to config.json's hitl.* block, falling
    # back to the in-code default of True.
    from harness.web_forms import build_run_argv_from_form
    argv, errors = build_run_argv_from_form({
        "flag.hitl_requirement": "true",
        "flag.hitl_architecture": "true",
        "flag.hitl_repair": "true",
        "flag.hitl_deployment": "true",
    })
    assert errors == []
    assert not any(a.startswith("--hitl-") for a in argv)


def test_build_run_argv_hitl_flags_emit_when_operator_opts_out():
    # Operator explicitly chooses false → emit the flag so the resolver
    # routes it as an explicit CLI override (highest precedence).
    from harness.web_forms import build_run_argv_from_form
    argv, errors = build_run_argv_from_form({
        "flag.hitl_requirement": "false",
        "flag.hitl_architecture": "false",
        "flag.hitl_repair": "false",
        "flag.hitl_deployment": "false",
    })
    assert errors == []
    assert "--hitl-requirement=false" in argv
    assert "--hitl-architecture=false" in argv
    assert "--hitl-repair=false" in argv
    assert "--hitl-deployment=false" in argv


def test_build_run_argv_deploy_dev_and_cd_discovery():
    from harness.web_forms import build_run_argv_from_form
    argv, errors = build_run_argv_from_form({
        "flag.deploy_dev": "true",
        "flag.cd_discovery": "true",
    })
    assert errors == []
    assert "--deploy-dev=true" in argv
    assert "--cd-discovery=true" in argv


def test_build_run_argv_new_build_true_also_emits_yes():
    """The CLI wizard auto-sets --yes when the operator picks
    --new-build=true (otherwise cmd_run would block on a confirmation
    prompt). The web form mirrors that because the spawned subprocess
    has no TTY."""
    from harness.web_forms import build_run_argv_from_form
    argv, errors = build_run_argv_from_form({"flag.new_build": "true"})
    assert errors == []
    assert "--new-build=true" in argv
    assert "--yes" in argv


def test_build_run_argv_new_build_false_does_not_emit_yes():
    from harness.web_forms import build_run_argv_from_form
    argv, errors = build_run_argv_from_form({"flag.new_build": "false"})
    assert errors == []
    # new_build=false is the default → emit nothing at all (skip the
    # noisy `--new-build=false` token and the irrelevant `--yes`).
    assert argv == []


def test_build_run_argv_select_rejects_unknown_choice():
    from harness.web_forms import build_run_argv_from_form
    argv, errors = build_run_argv_from_form({"flag.git": "bogus"})
    assert any("git" in e for e in errors)
    # Even on error the argv list is well-formed (no partial flags).
    assert "--git" not in " ".join(argv)
    # And the new choice set is true/false, not enable/disable.
    assert any("['false', 'true']" in e for e in errors)


# ---------------------------------------------------------------------------
# 7. Config-page grouping
# ---------------------------------------------------------------------------

def test_grouped_sections_covers_every_known_section():
    """Every section all_sections() returns must land in a group. The
    catch-all "Other" group is the safety net — but the named groups
    should be exhaustive."""
    from harness.web_forms import grouped_sections, all_sections
    groups = grouped_sections()
    grouped_names = set()
    for g in groups:
        for s in g.sections:
            grouped_names.add(s.section)
    all_names = {s.section for s in all_sections()}
    assert all_names == grouped_names, (
        f"sections not placed in any group: {all_names - grouped_names}"
    )


def test_grouped_sections_render_order_is_stable():
    from harness.web_forms import grouped_sections
    groups = grouped_sections()
    titles = [g.title for g in groups]
    # First and last group titles are operator-visible nav landmarks —
    # pin them so a reorder shows up as a test failure rather than silent
    # UI churn.
    assert titles[0] == "General"
    assert "LLM Registry" in titles
    assert "LLM Routing" in titles


def test_grouped_sections_each_section_appears_exactly_once():
    """A section listed in two groups would render twice and confuse the
    save-button targeting. Catch the regression here."""
    from harness.web_forms import grouped_sections
    seen: list[str] = []
    for g in grouped_sections():
        for s in g.sections:
            seen.append(s.section)
    assert len(seen) == len(set(seen)), (
        f"duplicate sections across groups: {seen}"
    )
