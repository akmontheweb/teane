"""Tests for harness.config_tree — the JSON-tree form editor that
powers the Configure Harness page's per-section editors.

Round-trips are the central guarantee: any value the renderer emits
must come back identical after parse_tree() walks the form payload.
The validator in harness.cli runs at save time and catches anything
the round-trip itself would miss.
"""

from __future__ import annotations

import re

import pytest

from harness.config_tree import (
    COLLECTION_DICT_OF_RECORDS,
    COLLECTION_DICT_OF_SCALARS,
    COLLECTION_LIST_OF_RECORDS,
    COLLECTION_LIST_OF_SCALARS,
    COLLECTION_NONE,
    TYPE_BOOL,
    TYPE_FLOAT,
    TYPE_INT,
    TYPE_STR,
    TreeParseError,
    infer_collection_kind,
    parse_tree,
    render_tree,
)


# ---------------------------------------------------------------------------
# 1. infer_collection_kind — basis for the renderer's + button choice
# ---------------------------------------------------------------------------

def test_infer_dict_of_records_for_llm_registry_shape():
    models = {
        "openai:gpt-4o": {"provider": "openai", "context_window": 128000},
        "anthropic:claude-sonnet": {"provider": "anthropic", "context_window": 200000},
    }
    assert infer_collection_kind(models) == COLLECTION_DICT_OF_RECORDS


def test_infer_list_of_records_for_mcp_servers_shape():
    servers = [
        {"name": "fetch", "transport": "stdio"},
        {"name": "fs", "transport": "stdio"},
    ]
    assert infer_collection_kind(servers) == COLLECTION_LIST_OF_RECORDS


def test_infer_list_of_scalars_for_string_list():
    assert infer_collection_kind(["~/.cache/pip", "~/.npm"]) == COLLECTION_LIST_OF_SCALARS


def test_infer_dict_of_scalars_for_max_tokens_per_role():
    per_role = {"planning": 4096, "patching": 8192}
    assert infer_collection_kind(per_role) == COLLECTION_DICT_OF_SCALARS


def test_infer_empty_dict_defaults_to_dict_of_scalars():
    # Empty containers default to the simpler/of-scalars flavour so
    # the operator can add their first entry without first picking a
    # shape; the shape pins on first add.
    assert infer_collection_kind({}) == COLLECTION_DICT_OF_SCALARS


def test_infer_empty_list_defaults_to_list_of_scalars():
    assert infer_collection_kind([]) == COLLECTION_LIST_OF_SCALARS


def test_infer_mixed_dict_falls_back_to_none():
    assert infer_collection_kind({"a": {"x": 1}, "b": 42}) == COLLECTION_NONE


# ---------------------------------------------------------------------------
# 2. render_tree — basic shape checks
# ---------------------------------------------------------------------------

def test_render_scalar_emits_path_type_value_hidden_fields():
    html = render_tree("hello", path="root/foo")
    assert "name='__path[]' value='root/foo'" in html
    assert "name='__type[]' value='str'" in html
    assert "value='hello'" in html


def test_render_int_uses_number_input():
    html = render_tree(42, path="x")
    assert "type='number'" in html
    assert "step='1'" in html
    assert "name='__type[]' value='int'" in html


def test_render_bool_uses_checkbox_backed_by_hidden_value():
    html_true = render_tree(True, path="flag")
    html_false = render_tree(False, path="flag")
    # Exactly ONE __value[] per bool — the visible checkbox is unnamed
    # and JS syncs it into the sibling hidden input.
    assert html_true.count("name='__value[]'") == 1
    assert "name='__value[]' value='true'" in html_true
    assert "type='checkbox'" in html_true
    assert "data-ct-bool-checkbox" in html_true
    assert "checked" in html_true
    assert html_false.count("name='__value[]'") == 1
    assert "name='__value[]' value='false'" in html_false
    assert "type='checkbox'" in html_false


def test_render_long_string_uses_textarea():
    long = "x" * 100
    html = render_tree(long, path="note")
    assert "<textarea" in html


def test_render_secret_path_uses_password_input():
    """Keys named *api_key*, *token*, *secret*, *password* mask in the
    UI so they don't leak across a shared screen."""
    html = render_tree("sk-abc123", path="models/openai/api_key")
    assert "type='password'" in html


# ---------------------------------------------------------------------------
# 3. render_tree — collections
# ---------------------------------------------------------------------------

def test_render_dict_of_records_emits_add_button_with_dict_record_kind():
    """The LLM registry case — a + Add button keyed to the dict_record
    collection so the JS knows it must prompt for a key name."""
    models = {"openai:gpt-4o": {"provider": "openai", "context_window": 128000}}
    html = render_tree(models, path="models")
    assert "data-collection='dict_record'" in html
    assert "data-path='models'" in html
    # The template element exists so JS can clone fresh records.
    assert "<template class='ct-template'>" in html
    # The existing entry is wrapped in a <details>.
    assert "data-dict-key='openai:gpt-4o'" in html


def test_render_dict_of_records_template_has_placeholder_key():
    """The template's placeholder gets `__NEW_KEY__` swapped to the
    operator's input on Add. Round-trip check: the placeholder is
    discoverable in the template region."""
    models = {"openai:gpt-4o": {"provider": "openai"}}
    html = render_tree(models, path="models")
    template_match = re.search(
        r"<template class='ct-template'>(.*?)</template>", html, re.DOTALL,
    )
    assert template_match is not None
    template_html = template_match.group(1)
    assert "__NEW_KEY__" in template_html


def test_render_list_of_records_emits_add_button_with_list_record_kind():
    servers = [{"name": "fetch", "transport": "stdio"}]
    html = render_tree(servers, path="mcp/servers")
    assert "data-collection='list_record'" in html
    assert "data-path='mcp/servers'" in html


def test_render_list_of_scalars_emits_add_entry_button():
    html = render_tree(["a", "b", "c"], path="sandbox/readonly_cache_mounts")
    assert "data-collection='list_scalar'" in html
    assert "data-path='sandbox/readonly_cache_mounts'" in html
    # Each existing entry has a × remove button.
    assert html.count("ct-remove") >= 3


def test_render_dict_of_scalars_emits_add_row_button():
    html = render_tree({"planning": 4096, "patching": 8192}, path="llm_dispatch/max_tokens_per_role")
    assert "data-collection='dict_scalar'" in html
    assert "ct-new-key" in html  # input for the new key


def test_render_empty_list_includes_container_marker():
    """Empty containers must round-trip back as empty containers so the
    validator sees the same shape."""
    html = render_tree([], path="mcp/command_allowlist")
    assert "name='__container[]' value='mcp/command_allowlist'" in html


# ---------------------------------------------------------------------------
# 4. parse_tree — type coercion + path reconstruction
# ---------------------------------------------------------------------------

def _form_payload(triples, containers=None):
    """Helper: build a dict[str, list[str]] mimicking what
    urllib.parse_qs produces from a multi-valued POST body."""
    paths, types, values = [], [], []
    for p, t, v in triples:
        paths.append(p)
        types.append(t)
        values.append(v)
    out = {"__path[]": paths, "__type[]": types, "__value[]": values}
    if containers:
        out["__container[]"] = list(containers)
    return out


def test_parse_flat_dict_round_trip():
    form = _form_payload([
        ("hard_cap_usd", TYPE_FLOAT, "3.0"),
        ("context_window_threshold_pct", TYPE_FLOAT, "0.85"),
    ])
    out = parse_tree(form)
    assert out == {"hard_cap_usd": 3.0, "context_window_threshold_pct": 0.85}


def test_parse_nested_dict_round_trip():
    form = _form_payload([
        ("openai:gpt-4o/provider", TYPE_STR, "openai"),
        ("openai:gpt-4o/input_cost_per_1m", TYPE_FLOAT, "2.5"),
        ("openai:gpt-4o/supports_thinking", TYPE_BOOL, "false"),
        ("openai:gpt-4o/supports_thinking", TYPE_BOOL, "true"),  # checkbox on
        ("openai:gpt-4o/context_window", TYPE_INT, "128000"),
    ])
    out = parse_tree(form)
    assert out == {
        "openai:gpt-4o": {
            "provider": "openai",
            "input_cost_per_1m": 2.5,
            "supports_thinking": True,
            "context_window": 128000,
        },
    }


def test_parse_bool_off_when_only_sentinel_present():
    """If the operator unchecks the box, only the hidden 'false'
    sentinel reaches us — the parser must collapse to False."""
    form = _form_payload([("flag", TYPE_BOOL, "false")])
    out = parse_tree(form)
    assert out == {"flag": False}


def test_parse_list_indices_via_numeric_segments():
    form = _form_payload([
        ("0/name", TYPE_STR, "fetch"),
        ("0/transport", TYPE_STR, "stdio"),
        ("1/name", TYPE_STR, "fs"),
        ("1/transport", TYPE_STR, "stdio"),
    ])
    out = parse_tree(form)
    assert out == [
        {"name": "fetch", "transport": "stdio"},
        {"name": "fs", "transport": "stdio"},
    ]


def test_parse_preserves_empty_container_via_container_marker():
    form = _form_payload(
        triples=[],
        containers=["mcp/command_allowlist"],
    )
    out = parse_tree(form)
    assert "mcp" in out and out["mcp"].get("command_allowlist") == {}


def test_parse_raises_on_unknown_type_tag():
    form = {
        "__path[]": ["foo"],
        "__type[]": ["bogus"],
        "__value[]": ["bar"],
    }
    with pytest.raises(TreeParseError):
        parse_tree(form)


def test_parse_raises_on_mismatched_array_lengths():
    form = {
        "__path[]": ["a", "b"],
        "__type[]": ["str"],
        "__value[]": ["x", "y"],
    }
    with pytest.raises(TreeParseError):
        parse_tree(form)


def test_parse_invalid_number_raises():
    form = _form_payload([("count", TYPE_INT, "notanumber")])
    with pytest.raises(TreeParseError):
        parse_tree(form)


# ---------------------------------------------------------------------------
# 5. Full round-trip: render → parse on a realistic config slice
# ---------------------------------------------------------------------------

def _harvest_form_payload_from_html(html):
    """A tiny extractor that walks the rendered HTML and reconstructs
    the same dict[str, list[str]] payload a browser would POST when the
    operator clicks Save without changing anything.

    Pair up the parallel hidden __path[] / __type[] entries with the
    following __value[] entry. Order matters: render_tree emits them
    tightly grouped per scalar. Bool fields emit a single hidden
    __value[] (the visible checkbox has no name). Strip
    <template>...</template> blocks first so we don't harvest the
    placeholder rows.
    """
    cleaned = re.sub(r"<template[^>]*>.*?</template>", "", html, flags=re.DOTALL)
    # Walk per leaf — scan forward from each __path occurrence to the
    # subsequent __value entries.
    leaves: list[tuple[str, str, list[str]]] = []
    cursor = 0
    pat = re.compile(
        r"name='__path\[\]'\s+value='([^']*)'.*?"
        r"name='__type\[\]'\s+value='([^']*)'",
        re.DOTALL,
    )
    while True:
        m = pat.search(cleaned, cursor)
        if not m:
            break
        leaf_path = m.group(1)
        leaf_type = m.group(2)
        # Look forward to the next __path or end-of-string and pull
        # any __value attributes in that window. Bool leaves have two;
        # everything else has one.
        next_m = re.search(
            r"name='__path\[\]'", cleaned[m.end():],
        )
        end = m.end() + (next_m.start() if next_m else len(cleaned) - m.end())
        window = cleaned[m.end():end]
        # textarea content
        ta = re.search(
            r"<textarea[^>]*name='__value\[\]'[^>]*>(.*?)</textarea>", window,
            re.DOTALL,
        )
        if ta:
            leaves.append((leaf_path, leaf_type, [ta.group(1)]))
            cursor = end
            continue
        # <select> element: pull the value of the selected <option> (or
        # empty string when no option is marked selected).
        sel_m = re.search(
            r"<select[^>]*name='__value\[\]'[^>]*>(.*?)</select>",
            window, re.DOTALL,
        )
        if sel_m:
            sel_body = sel_m.group(1)
            selected = re.search(
                r"<option\s+value='([^']*)'[^>]*selected[^>]*>", sel_body,
            )
            picked = selected.group(1) if selected else ""
            leaves.append((leaf_path, leaf_type, [picked]))
            cursor = end
            continue
        # input value='...' attrs for __value[]. The checkbox itself
        # has no name attribute, so this captures only the hidden
        # input that actually carries the value.
        ivals = re.findall(
            r"name='__value\[\]'\s+value='([^']*)'", window,
        )
        leaves.append((leaf_path, leaf_type, ivals or [""]))
        cursor = end
    # Flatten in order.
    out_paths: list[str] = []
    out_types: list[str] = []
    out_values: list[str] = []
    for p, t, vs in leaves:
        for v in vs:
            out_paths.append(p)
            out_types.append(t)
            out_values.append(v)
    return {
        "__path[]": out_paths,
        "__type[]": out_types,
        "__value[]": out_values,
    }


def test_round_trip_llm_registry_through_render_and_parse():
    """The LLM registry — render the live shape, harvest the would-be
    POST payload, parse it, and verify we got the original back."""
    original = {
        "openai:gpt-4o": {
            "provider": "openai",
            "model_id": "gpt-4o",
            "context_window": 128000,
            "input_cost_per_1m": 2.50,
            "output_cost_per_1m": 10.00,
            "supports_thinking": True,
            "supports_cache": True,
            "api_key": "",
        },
        "anthropic:claude-sonnet-4": {
            "provider": "anthropic",
            "model_id": "claude-sonnet-4-20250514",
            "context_window": 200000,
            "input_cost_per_1m": 3.00,
            "output_cost_per_1m": 15.00,
            "supports_thinking": False,
            "supports_cache": True,
            "api_key": "",
        },
    }
    html = render_tree(original, path="models")
    payload = _harvest_form_payload_from_html(html)
    rebuilt = parse_tree(payload)
    # The renderer emits per-model leaves under 'models/<key>/...' so
    # parsing yields {'models': {...}} — the section save handler will
    # extract the right top-level value.
    assert "models" in rebuilt
    assert rebuilt["models"] == original


def test_round_trip_list_of_scalars():
    original = ["~/.cache/pip", "~/.npm", "~/.cache/yarn"]
    html = render_tree(original, path="sandbox/readonly_cache_mounts")
    payload = _harvest_form_payload_from_html(html)
    rebuilt = parse_tree(payload)
    assert rebuilt == {
        "sandbox": {"readonly_cache_mounts": original},
    }


# ---------------------------------------------------------------------------
# 6. Record-inner schemas are LOCKED: no add/delete on dependent fields
# ---------------------------------------------------------------------------

def _record_body_html(full_html, record_marker):
    """Extract the body HTML for a single <details class='ct-record'>
    matching ``record_marker`` (e.g. data-dict-key='openai:gpt-4o')."""
    import re as _re
    m = _re.search(
        r"<details[^>]*" + _re.escape(record_marker)
        + r"[^>]*>(.*?)(?=<details class='ct-record'|<template|</div>$)",
        full_html, _re.DOTALL,
    )
    return m.group(1) if m else ""


def test_llm_record_body_has_no_inner_add_button():
    """A model record (e.g. openai:gpt-4o) has fixed schema fields —
    provider, model_id, api_key etc. The operator cannot invent new
    keys per model, so the record body must NOT show a + Add control.

    Regression check for the user-reported issue: 'LLM API KEY by
    itself has no meaning — only in the context of a model.'"""
    models = {
        "openai:gpt-4o": {
            "provider": "openai",
            "model_id": "gpt-4o",
            "api_key": "",
        },
    }
    out = render_tree(models, path="models")
    body = _record_body_html(out, "data-dict-key='openai:gpt-4o'")
    assert body, "could not extract openai:gpt-4o record body"
    # No add affordances on the record's own dict-of-scalars level.
    assert "data-collection='dict_scalar'" not in body
    assert "data-collection='dict_record'" not in body


def test_llm_record_body_has_no_inner_remove_per_field():
    """Same regression: individual model fields (api_key, provider,
    model_id, ...) have no × delete button — they're part of the
    record's fixed schema. The × on the record's HEADER deletes the
    whole record; that's the only valid delete at this level."""
    models = {
        "openai:gpt-4o": {
            "provider": "openai",
            "api_key": "",
            "model_id": "gpt-4o",
        },
    }
    out = render_tree(models, path="models")
    body = _record_body_html(out, "data-dict-key='openai:gpt-4o'")
    assert body
    # No row/item delete buttons inside the body.
    assert "data-target='row'" not in body
    assert "data-target='item'" not in body


def test_mcp_server_record_body_has_no_inner_add_button():
    """Same rule for list-of-records: a server entry has a fixed
    schema (name, transport, command) — no + Add inside."""
    servers = [
        {"name": "fetch", "transport": "stdio", "command": ["uvx", "mcp-server-fetch"]},
    ]
    out = render_tree(servers, path="mcp/servers")
    body = _record_body_html(out, "data-list-index='0'")
    assert body
    assert "data-collection='dict_scalar'" not in body


def test_mcp_server_record_inner_list_still_extensible():
    """The record's body is locked, but a nested list-of-scalars INSIDE
    a record (like server.command = ['uvx', '...']) is still
    extensible — operator can add args to the command without
    re-defining the record's schema."""
    servers = [
        {"name": "fetch", "transport": "stdio", "command": ["uvx", "mcp-server-fetch"]},
    ]
    out = render_tree(servers, path="mcp/servers")
    body = _record_body_html(out, "data-list-index='0'")
    assert body
    # The nested command list still has its own + Add entry button.
    assert "data-collection='list_scalar'" in body


def test_record_can_still_be_deleted_whole():
    """The × on the record HEADER deletes the whole record. That's
    the only delete affordance for a record."""
    models = {"openai:gpt-4o": {"provider": "openai"}}
    out = render_tree(models, path="models")
    # The record's summary carries data-target='record' so JS removes it.
    assert "ct-remove ct-remove--record" in out
    assert "data-target='record'" in out


def test_extensible_dict_of_scalars_still_has_per_row_delete():
    """Counterpoint: a *truly* extensible dict-of-scalars (like
    max_tokens_per_role, where keys are operator-defined roles) keeps
    the × delete on each row. The fix must not over-correct."""
    per_role = {"planning": 4096, "patching": 8192}
    out = render_tree(per_role, path="llm_dispatch/max_tokens_per_role")
    assert "data-target='row'" in out
    assert "data-collection='dict_scalar'" in out


# ---------------------------------------------------------------------------
# 7. + Add on dict-of-records: editable inline key, no upfront prompt
# ---------------------------------------------------------------------------

def test_dict_of_records_add_button_says_singular_noun():
    """The + Add button labels itself with the singular form of the
    collection — '+ Add model' for `models`, '+ Add server' for
    `mcp/servers`, '+ Add job' for `schedule/jobs`. Clearer than the
    generic '+ Add record'."""
    models = {"openai:gpt-4o": {"provider": "openai"}}
    assert "+ Add model" in render_tree(models, path="models")
    servers = [{"name": "fetch", "transport": "stdio"}]
    assert "+ Add server" in render_tree(servers, path="mcp/servers")
    # And empty collections still get the right noun (default schema kicks in).
    assert "+ Add job" in render_tree([], path="schedule/jobs")


def test_dict_of_records_template_uses_editable_key_input():
    """The template a + Add click clones must carry an editable
    <input class='ct-record__key-input'> in the record header — NOT
    a static title span. That's what lets the operator name the record
    inline after clicking + Add, without filling a separate prompt."""
    models = {"openai:gpt-4o": {"provider": "openai"}}
    out = render_tree(models, path="models")
    template_match = re.search(
        r"<template class='ct-template'>(.*?)</template>", out, re.DOTALL,
    )
    assert template_match
    template = template_match.group(1)
    # Editable key input present.
    assert "class='ct-record__key-input'" in template
    assert "value='__NEW_KEY__'" in template
    # No legacy static title span inside the template.
    assert "<span class='ct-record__title'>__NEW_KEY__</span>" not in template


def test_dict_of_records_no_longer_renders_upfront_key_input():
    """After the UX fix, the bottom "New key + + Add" row is gone.
    Only the + Add button remains — clicking it inserts a record
    instantly with an inline editable key."""
    models = {"openai:gpt-4o": {"provider": "openai"}}
    out = render_tree(models, path="models")
    # ct-add-row carried the old upfront key prompt for dict_record.
    # It's only used by dict_scalar now (max_tokens_per_role, etc.).
    # Confirm no ct-new-key INSIDE the dict-of-records (template aside).
    template_match = re.search(
        r"<template class='ct-template'>.*?</template>", out, re.DOTALL,
    )
    cleaned = out
    if template_match:
        cleaned = cleaned.replace(template_match.group(0), "")
    assert "ct-new-key" not in cleaned, "dict-of-records should not render an upfront key prompt"


# ---------------------------------------------------------------------------
# 8. Default record schemas — + Add works on empty collections
# ---------------------------------------------------------------------------

def test_empty_models_uses_default_llm_schema_for_template():
    """When `models` is empty in config.json (fresh install / cleared
    registry), + Add still needs a usable template. The default schema
    registered for 'models' kicks in so the cloned record has all the
    expected fields ready to fill (provider, model_id, costs, ...)."""
    out = render_tree({}, path="models")
    template_match = re.search(
        r"<template class='ct-template'>(.*?)</template>", out, re.DOTALL,
    )
    assert template_match
    template = template_match.group(1)
    # All canonical model fields are present in the template.
    for required in ("provider", "model_id", "context_window",
                     "input_cost_per_1m", "output_cost_per_1m",
                     "supports_thinking", "api_key"):
        assert "models/__NEW_KEY__/" + required in template, \
            f"default model template missing {required!r}"


def test_empty_schedule_jobs_uses_default_job_schema_for_template():
    """`schedule.jobs` ships empty by default. + Add must still produce
    a usable job record (name + schedule + workspace + prompt)."""
    out = render_tree([], path="schedule/jobs")
    template_match = re.search(
        r"<template class='ct-template'>(.*?)</template>", out, re.DOTALL,
    )
    assert template_match
    template = template_match.group(1)
    for required in ("name", "schedule", "workspace", "prompt"):
        assert "schedule/jobs/__INDEX__/" + required in template, \
            f"default job template missing {required!r}"


def test_empty_mcp_servers_uses_default_server_schema_for_template_dummy():
    pass  # placeholder so the next test keeps its docstring


# ---------------------------------------------------------------------------
# 9. model_routing custom renderer — Role → Primary/Fallback → Field
# ---------------------------------------------------------------------------

def test_model_routing_renders_all_five_role_groups():
    """The model_routing renderer groups by ROLE so each (Primary,
    Fallback) pair lives together. All five role groups must appear:
    Planning, Patching, Repair, Doc Review, Code Review."""
    from harness.config_tree import render_model_routing
    routing = {
        "planning_primary": "openai:gpt-4o",
        "planning_mode": "thinking_max",
        "planning_fallback": "openai:gpt-4o-mini",
        "patching_primary": "openai:gpt-4o",
        "patching_mode": "no_thinking",
    }
    html_out = render_model_routing(
        routing, path="model_routing",
        available_models=["openai:gpt-4o", "openai:gpt-4o-mini"],
    )
    for label in ("Planning", "Patching", "Repair", "Doc Review", "Code Review"):
        # Each role title appears in a <summary> head exactly once
        assert f">{label}<" in html_out, f"missing role group {label!r}"


def test_model_routing_planning_has_primary_and_fallback_subgroups():
    """Planning has both primary and fallback in the validator schema,
    so its body must contain both subgroups."""
    from harness.config_tree import render_model_routing
    html_out = render_model_routing(
        {"planning_primary": "x", "planning_mode": "thinking", "planning_fallback": "y"},
        path="model_routing", available_models=[],
    )
    assert ">Planning Primary<" in html_out
    assert ">Planning Fallback<" in html_out


def test_model_routing_patching_has_fallback_subgroup():
    """Patching now supports a fallback model (added in the
    configure-page overhaul) — the renderer must emit a Patching
    Fallback subgroup with its own thinking-mode selector."""
    from harness.config_tree import render_model_routing
    html_out = render_model_routing(
        {"patching_primary": "x", "patching_mode": "no_thinking"},
        path="model_routing", available_models=[],
    )
    assert ">Patching Primary<" in html_out
    assert ">Patching Fallback<" in html_out
    assert "value='model_routing/patching_fallback_mode'" in html_out


def test_model_routing_thinking_mode_on_primary_and_fallback():
    """Primary writes <role>_mode, fallback writes <role>_fallback_mode.
    Both subgroups now render their own thinking-mode dropdown so
    operators can diverge the two paths."""
    from harness.config_tree import render_model_routing
    html_out = render_model_routing(
        {
            "planning_primary": "x", "planning_mode": "thinking",
            "planning_fallback": "y",
        },
        path="model_routing", available_models=[],
    )
    # Exactly one entry per mode key.
    assert html_out.count("value='model_routing/planning_mode'") == 1
    assert html_out.count("value='model_routing/planning_fallback_mode'") == 1
    # Fallback explains the inheritance default (apostrophe is escaped
    # for HTML — match on the unambiguous prefix).
    assert "Leave blank to inherit the primary" in html_out


def test_model_routing_model_dropdown_populates_from_available_models():
    """Model name fields are <select> elements populated from the live
    config's models registry, so operators pick from registered keys
    instead of typing model names from memory."""
    from harness.config_tree import render_model_routing
    html_out = render_model_routing(
        {"planning_primary": "openai:gpt-4o"},
        path="model_routing",
        available_models=["openai:gpt-4o", "anthropic:claude-sonnet-4"],
    )
    # Every available model appears as an <option> at least once
    # (multiple times because every role's select has them).
    assert html_out.count(">openai:gpt-4o<") >= 1
    assert html_out.count(">anthropic:claude-sonnet-4<") >= 1
    # Current value is pre-selected.
    assert "value='openai:gpt-4o' selected" in html_out


def test_model_routing_preserves_unregistered_value():
    """If the current routing references a model NOT in the registry
    (e.g. operator removed it), keep the value visible + selected as
    an extra option marked '(unregistered)' so the operator notices."""
    from harness.config_tree import render_model_routing
    html_out = render_model_routing(
        {"planning_primary": "openai:gpt-5-orphan"},
        path="model_routing",
        available_models=["openai:gpt-4o"],
    )
    assert "openai:gpt-5-orphan" in html_out
    assert "(unregistered)" in html_out


def test_model_routing_local_ollama_group_includes_force_local_only():
    """The Local Ollama group at the bottom carries the non-role
    ollama_local_model + ollama_local_backup + force_local_only knobs."""
    from harness.config_tree import render_model_routing
    html_out = render_model_routing(
        {
            "ollama_local_model": "ollama:qwen2.5-coder:14b",
            "ollama_local_backup": "",
            "force_local_only": False,
        },
        path="model_routing", available_models=["ollama:qwen2.5-coder:14b"],
    )
    assert ">Local Ollama<" in html_out
    # All three fields rendered with the expected paths.
    assert "value='model_routing/ollama_local_model'" in html_out
    assert "value='model_routing/ollama_local_backup'" in html_out
    assert "value='model_routing/force_local_only'" in html_out


def test_model_routing_round_trips_through_parse_tree():
    """The custom renderer must emit the SAME flat __path[]/__type[]/
    __value[] schema as the generic tree so parse_tree round-trips
    identically. Critical for the save handler."""
    from harness.config_tree import render_model_routing
    original = {
        "planning_primary": "openai:gpt-4o",
        "planning_mode": "thinking_max",
        "planning_fallback": "openai:gpt-4o-mini",
        "patching_primary": "openai:gpt-4o",
        "patching_mode": "no_thinking",
        "repair_primary": "openai:gpt-4o",
        "repair_mode": "no_thinking",
        "repair_fallback": "openai:gpt-4o-mini",
        "doc_reviewer_primary": "openai:gpt-4o-mini",
        "doc_reviewer_mode": "thinking",
        "doc_reviewer_fallback": "openai:gpt-4o",
        "code_reviewer_primary": "openai:gpt-4o-mini",
        "code_reviewer_mode": "thinking",
        "code_reviewer_fallback": "openai:gpt-4o",
        "ollama_local_model": "ollama:qwen2.5-coder:14b",
        "ollama_local_backup": "",
        "force_local_only": False,
    }
    html_out = render_model_routing(
        original, path="model_routing",
        available_models=["openai:gpt-4o", "openai:gpt-4o-mini",
                          "ollama:qwen2.5-coder:14b"],
    )
    # Reconstruct the form payload a default-submit would POST.
    payload = _harvest_form_payload_from_html(html_out)
    rebuilt = parse_tree(payload)
    assert "model_routing" in rebuilt
    # The rebuilt section equals the original (every key round-trips).
    for k, v in original.items():
        assert rebuilt["model_routing"].get(k) == v, \
            f"round-trip mismatch on {k}: got {rebuilt['model_routing'].get(k)!r} != {v!r}"


def test_empty_mcp_servers_uses_default_server_schema_for_template():
    """If `mcp.servers` were ever empty, + Add still produces a server
    record with the right fields (name, transport, command)."""
    out = render_tree([], path="mcp/servers")
    template_match = re.search(
        r"<template class='ct-template'>(.*?)</template>", out, re.DOTALL,
    )
    assert template_match
    template = template_match.group(1)
    for required in ("name", "transport"):
        assert "mcp/servers/__INDEX__/" + required in template, \
            f"default server template missing {required!r}"
