"""Generic JSON-tree form editor for ``config.json``.

The Configure Harness page in :mod:`harness.dashboard` needs to expose
*every* key in ``config.json`` for inline editing, mirror the JSON's
nesting in the UI, and offer ``+`` controls for collections that can
grow (the LLM registry, MCP servers, etc.).

This module is the type-aware part of that:

- :func:`render_tree` walks a Python value and emits nested form HTML.
- :func:`parse_tree` reconstructs the original nested value from the
  flat ``__path[]`` / ``__type[]`` / ``__value[]`` form payload that
  comes back on POST.
- :func:`infer_collection_kind` lets the renderer decide where to add
  ``+ Add`` buttons.

No HTML lives in ``dashboard.py``'s legacy curated-form path; this
module replaces it. The strict validator in :mod:`harness.cli` runs at
save time, so type-coercion errors here still get caught before the
write lands on disk.
"""

from __future__ import annotations

import html
import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Type tags carried in __type[] alongside __path[] / __value[]
# ---------------------------------------------------------------------------

TYPE_STR = "str"
TYPE_INT = "int"
TYPE_FLOAT = "float"
TYPE_BOOL = "bool"
TYPE_NULL = "null"
TYPE_JSON = "json"  # for opaque/heterogeneous list scalars; raw JSON value

_KNOWN_TYPES: frozenset[str] = frozenset(
    {TYPE_STR, TYPE_INT, TYPE_FLOAT, TYPE_BOOL, TYPE_NULL, TYPE_JSON},
)


def _python_type_tag(value: Any) -> str:
    """Map a live Python value to a type-tag string the form will carry.

    Booleans come before ints because ``isinstance(True, int)`` is True
    in Python and we want the checkbox kind, not the number kind.
    """
    if value is None:
        return TYPE_NULL
    if isinstance(value, bool):
        return TYPE_BOOL
    if isinstance(value, int):
        return TYPE_INT
    if isinstance(value, float):
        return TYPE_FLOAT
    if isinstance(value, str):
        return TYPE_STR
    # list / dict / anything else: opaque JSON.
    return TYPE_JSON


# ---------------------------------------------------------------------------
# 2. Inferring "collectability" so the renderer can decide where to add +
# ---------------------------------------------------------------------------

COLLECTION_NONE = "none"
COLLECTION_DICT_OF_RECORDS = "dict_of_records"   # dict[str, dict]; +Add by key
COLLECTION_LIST_OF_RECORDS = "list_of_records"   # list[dict];      +Add appends
COLLECTION_LIST_OF_SCALARS = "list_of_scalars"   # list[scalar];    +Add row
COLLECTION_DICT_OF_SCALARS = "dict_of_scalars"   # dict[str, scalar]; +Add k:v


def infer_collection_kind(value: Any) -> str:
    """Classify a container so the renderer knows what + control to add.

    Empty containers are classified as their respective "of scalars"
    flavour (the operator can extend them; their first entry is what
    pins the shape later). A ``list`` or ``dict`` whose contents are
    a mix of dicts and scalars is treated as opaque (``COLLECTION_NONE``)
    — the renderer falls back to a JSON textarea for those.
    """
    if isinstance(value, dict):
        if not value:
            return COLLECTION_DICT_OF_SCALARS
        sample = next(iter(value.values()))
        all_dicts = all(isinstance(v, dict) for v in value.values())
        all_scalar = all(not isinstance(v, (dict, list)) for v in value.values())
        if all_dicts:
            return COLLECTION_DICT_OF_RECORDS
        if all_scalar:
            return COLLECTION_DICT_OF_SCALARS
        # Mixed — fall back to opaque JSON. Real configs rarely mix.
        _ = sample  # for the linter; intentionally inspected
        return COLLECTION_NONE
    if isinstance(value, list):
        if not value:
            return COLLECTION_LIST_OF_SCALARS
        all_dicts = all(isinstance(v, dict) for v in value)
        all_scalar = all(not isinstance(v, (dict, list)) for v in value)
        if all_dicts:
            return COLLECTION_LIST_OF_RECORDS
        if all_scalar:
            return COLLECTION_LIST_OF_SCALARS
        return COLLECTION_NONE
    return COLLECTION_NONE


# ---------------------------------------------------------------------------
# 3. Path encoding — JSON-pointer-ish, slash-separated
# ---------------------------------------------------------------------------

def _join_path(parent: str, segment: str) -> str:
    if not parent:
        return segment
    return f"{parent}/{segment}"


_INT_SEGMENT = re.compile(r"^\d+$")


def _parse_path(path: str) -> list[Any]:
    """Split a slash-joined path into segments. Numeric segments stay
    as strings here; the walker decides whether to treat them as list
    indices based on the existing container shape."""
    if not path:
        return []
    return path.split("/")


# ---------------------------------------------------------------------------
# 3b. Default record schemas for known collection paths
# ---------------------------------------------------------------------------
#
# When an extensible collection is EMPTY in config.json there's no existing
# entry to use as a "template" for + Add. Hard-code the canonical record
# schema for the well-known collections so + Add produces a usable record
# even when nothing has been added yet.
#
# The blank values double as type tags (str/int/float/bool/list) so the
# renderer picks the right input kind. Keep these in sync with the
# validator schema in harness/cli.py if it grows new required fields.

_DEFAULT_RECORD_SCHEMAS: dict[str, dict[str, Any]] = {
    # LLM registry. Mirrors the shape of every entry in config.json:models.
    "models": {
        "provider": "",
        "model_id": "",
        "context_window": 0,
        "input_cost_per_1m": 0.0,
        "output_cost_per_1m": 0.0,
        "cached_input_cost_per_1m": 0.0,
        "api_base_url": "",
        "supports_thinking": False,
        "supports_cache": False,
        "api_key": "",
    },
    # MCP server pool. Mirrors mcp.servers[i].
    "mcp/servers": {
        "name": "",
        "transport": "stdio",
        "command": [],
    },
    # Cron-driven scheduled jobs. Mirrors the documented schedule.jobs shape.
    "schedule/jobs": {
        "name": "",
        "schedule": "",
        "workspace": "",
        "prompt": "",
        "extra_args": [],
    },
    # Additional web-tool backends (the configure-page overhaul exposes
    # this as a list under web_tools so operators can declare more than
    # one search backend at once). The primary backend stays at
    # web_tools.search_backend; this list is consulted afterwards.
    "web_tools/backends": {
        "name": "",
        "enabled": True,
        "search_backend": "",
        "api_key_env": "",
    },
}


def default_record_schema(path: str) -> Optional[dict[str, Any]]:
    """Return the canonical record schema for the collection at ``path``,
    or None if no default is registered. Used by the renderer when a
    collection is empty so + Add can still produce a populated template."""
    return _DEFAULT_RECORD_SCHEMAS.get(path)


# ---------------------------------------------------------------------------
# 3c. Custom renderer: model_routing as a Role → Primary/Fallback tree
# ---------------------------------------------------------------------------
#
# The flat ``model_routing`` dict has 16 keys (planning_primary,
# planning_mode, planning_fallback, patching_primary, ...) — the
# generic tree editor renders these as a long alphabetical row stack
# that makes it hard to see "which model serves which role". This
# renderer groups them by ROLE and SUBGROUP so each (Primary, Fallback)
# pair lives together with its applicable thinking-mode dropdown.
#
# The emitted form fields still carry the SAME flat ``__path[]``
# entries (``model_routing/planning_primary`` etc.), so parse_tree
# round-trips identically and the validator sees no shape change —
# this is purely a presentation upgrade.

# (role_key, display_label, has_fallback) for every role the
# validator's _KNOWN_NESTED_KEYS["model_routing"] understands.
_ROUTING_ROLES: tuple[tuple[str, str, bool], ...] = (
    ("planning", "Planning", True),
    ("patching", "Patching", True),
    ("repair", "Repair", True),
    ("doc_reviewer", "Doc Review", True),
    ("code_reviewer", "Code Review", True),
)

# Thinking-mode option set. The gateway treats "thinking"/"thinking_max"
# as thinking-on and anything else as thinking-off, so we keep the
# canonical four values as the dropdown choices and preserve any
# operator-set value we don't recognise.
_THINKING_MODES: tuple[str, ...] = ("thinking_max", "thinking", "no_thinking")


def _render_routing_select(
    *,
    path: str, value: str, options: list[str],
    label: str, sublabel: str = "",
    extra_note: str = "",
) -> str:
    """Render one labelled row: hidden __path/__type + a <select>
    populated from ``options``. Preserves ``value`` even when it's
    not in ``options`` (operator-set custom value) by appending an
    extra <option marked "(unregistered)"."""
    seen = False
    opt_html: list[str] = ["<option value=''></option>"]
    for opt in options:
        sel = " selected" if opt == value else ""
        if opt == value:
            seen = True
        opt_html.append(
            f"<option value='{html.escape(opt)}'{sel}>{html.escape(opt)}</option>"
        )
    if value and not seen:
        opt_html.append(
            f"<option value='{html.escape(value)}' selected>"
            f"{html.escape(value)} (unregistered)</option>"
        )
    note_html = (
        f"<div class='muted fs-sm mt-3'>{html.escape(extra_note)}</div>"
        if extra_note else ""
    )
    sublabel_html = (
        f"<code class='ct-row__key'>{html.escape(sublabel)}</code>"
        if sublabel else ""
    )
    return (
        f"<div class='ct-row'>"
        f"<label class='bx--label ct-row__label'>"
        f"{html.escape(label)}"
        f"{sublabel_html}"
        f"</label>"
        f"<div class='ct-row__value'>"
        f"{_hidden('__path[]', path)}"
        f"{_hidden('__type[]', TYPE_STR)}"
        f"<select class='bx--select-input ct-input' name='__value[]'>"
        f"{''.join(opt_html)}</select>"
        f"{note_html}"
        f"</div>"
        f"</div>"
    )


def _render_routing_subgroup(
    *,
    title: str,
    model_path: str,
    model_value: str,
    available_models: list[str],
    mode_path: Optional[str] = None,
    mode_value: str = "",
    is_primary: bool = True,
) -> str:
    """Render a Primary or Fallback subgroup: Model select + Thinking-mode
    select.

    Both primary and fallback subgroups now expose their own thinking
    selector — primary writes ``<role>_mode``, fallback writes
    ``<role>_fallback_mode``. When the fallback key is unset the
    gateway resolves it to the primary's mode, preserving legacy
    behaviour.
    """
    body: list[str] = []
    body.append(_render_routing_select(
        path=model_path, value=model_value,
        options=available_models,
        label="Model name",
        sublabel=model_path.rsplit("/", 1)[-1],
    ))
    if mode_path is not None:
        body.append(_render_routing_select(
            path=mode_path, value=mode_value,
            options=list(_THINKING_MODES),
            label="Thinking mode",
            sublabel=mode_path.rsplit("/", 1)[-1],
            extra_note=(
                "" if is_primary
                else "Leave blank to inherit the primary's mode."
            ),
        ))
    return (
        f"<details class='ct-routing__sub' open>"
        f"<summary class='ct-routing__sub-head'>{html.escape(title)}</summary>"
        f"<div class='ct-routing__sub-body'>{''.join(body)}</div>"
        f"</details>"
    )


def render_model_routing(
    value: dict[str, Any], path: str, *, available_models: list[str],
) -> str:
    """Tree-shaped renderer for the ``model_routing`` config section.

    Three-level layout (Group → Subgroup → Field) so each (Primary,
    Fallback) pair reads together, with the thinking mode attached to
    the Primary. The Local Ollama group at the bottom handles the
    non-role ``ollama_*`` and ``force_local_only`` knobs that don't
    fit the (primary, fallback) pattern.

    ``available_models`` is the sorted list of keys from the live
    ``models`` registry; we use it to populate the Model <select>
    dropdowns so operators pick from the registered set instead of
    typing model names from memory.
    """
    routing = value or {}
    parts: list[str] = [f"<div class='ct-routing' data-path='{html.escape(path)}'>"]

    for role_key, role_label, has_fallback in _ROUTING_ROLES:
        primary_path = f"{path}/{role_key}_primary"
        fallback_path = f"{path}/{role_key}_fallback"
        mode_path = f"{path}/{role_key}_mode"
        fallback_mode_path = f"{path}/{role_key}_fallback_mode"
        primary_val = str(routing.get(f"{role_key}_primary", "") or "")
        fallback_val = str(routing.get(f"{role_key}_fallback", "") or "")
        mode_val = str(routing.get(f"{role_key}_mode", "") or "")
        fallback_mode_val = str(routing.get(f"{role_key}_fallback_mode", "") or "")

        # Group header
        parts.append(
            f"<details class='ct-routing__role' open>"
            f"<summary class='ct-routing__role-head'>"
            f"{html.escape(role_label)}"
            f"</summary>"
            f"<div class='ct-routing__role-body'>"
        )
        # Primary subgroup — always present
        parts.append(_render_routing_subgroup(
            title=f"{role_label} Primary",
            model_path=primary_path, model_value=primary_val,
            available_models=available_models,
            mode_path=mode_path, mode_value=mode_val,
            is_primary=True,
        ))
        # Fallback subgroup — every role supports one now, including
        # patching (added in the configure-page overhaul).
        if has_fallback:
            parts.append(_render_routing_subgroup(
                title=f"{role_label} Fallback",
                model_path=fallback_path, model_value=fallback_val,
                available_models=available_models,
                mode_path=fallback_mode_path, mode_value=fallback_mode_val,
                is_primary=False,
            ))
        parts.append("</div></details>")

    # Local Ollama group — non-role keys that still belong to
    # model_routing. Operators flip force_local_only=true to route
    # every dispatch to the configured Ollama model.
    ollama_model = str(routing.get("ollama_local_model", "") or "")
    ollama_backup = str(routing.get("ollama_local_backup", "") or "")
    force_local = bool(routing.get("force_local_only", False))
    parts.append(
        "<details class='ct-routing__role'>"
        "<summary class='ct-routing__role-head'>Local Ollama</summary>"
        "<div class='ct-routing__role-body'>"
        "<details class='ct-routing__sub' open>"
        "<summary class='ct-routing__sub-head'>Models</summary>"
        "<div class='ct-routing__sub-body'>"
    )
    parts.append(_render_routing_select(
        path=f"{path}/ollama_local_model", value=ollama_model,
        options=available_models,
        label="Local model",
        sublabel="ollama_local_model",
    ))
    parts.append(_render_routing_select(
        path=f"{path}/ollama_local_backup", value=ollama_backup,
        options=available_models,
        label="Local backup",
        sublabel="ollama_local_backup",
    ))
    # Boolean toggle for force_local_only — render as a scalar row.
    parts.append(_render_scalar_row(
        f"{path}/force_local_only", "force_local_only", force_local,
    ))
    parts.append("</div></details></div></details>")

    # Preserve any unknown keys the operator may have added (or that
    # land here in a future schema bump) by rendering them via the
    # generic tree at the bottom. Defensive — keeps round-trip safe.
    known: set[str] = set()
    for role_key, _, _ in _ROUTING_ROLES:
        known.update({
            f"{role_key}_primary", f"{role_key}_mode", f"{role_key}_fallback",
            f"{role_key}_fallback_mode",
        })
    known.update({"ollama_local_model", "ollama_local_backup", "force_local_only"})
    unknown = {k: v for k, v in routing.items() if k not in known}
    if unknown:
        parts.append(
            "<details class='ct-routing__role'>"
            "<summary class='ct-routing__role-head'>Other / unknown keys</summary>"
            "<div class='ct-routing__role-body'>"
            + render_tree(unknown, path=path, depth=1, allow_add_keys=False)
            + "</div></details>"
        )

    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# 4. Rendering — walk a Python value, emit nested HTML form fields
# ---------------------------------------------------------------------------

# Form name conventions:
#   __path[]      — JSON-pointer-ish path to the field
#   __type[]      — type tag from _KNOWN_TYPES, parallel to __path
#   __value[]     — raw value, parallel to __path / __type
#   __container[] — paths of empty containers we need to preserve
#                   ("models" must remain as {} even if zero entries)
#
# These names use the PHP-style [] suffix; stdlib urllib.parse_qs handles
# them as repeated keys ("__path[]": ["a", "b", ...]).


# Renderer config — controls indent depth + which top-level paths get
# special treatment (e.g. password masking for "*api_key*").

_SECRET_KEY_PAT = re.compile(r"(api_key|token|secret|password)", re.IGNORECASE)


def _is_secret_path(path: str) -> bool:
    last = path.rsplit("/", 1)[-1]
    return bool(_SECRET_KEY_PAT.search(last))


def _hidden(name: str, value: str) -> str:
    return (
        f"<input type='hidden' name='{html.escape(name)}' "
        f"value='{html.escape(value, quote=True)}'>"
    )


def _input_for_scalar(path: str, value: Any) -> str:
    """Render the form input + parallel __path/__type hidden fields for
    one scalar leaf. ``value`` can be str/int/float/bool/None."""
    tag = _python_type_tag(value)
    hidden = (
        _hidden("__path[]", path)
        + _hidden("__type[]", tag)
    )
    safe_val = "" if value is None else str(value)
    if tag == TYPE_BOOL:
        # Hidden "off" sentinel + checkbox; if the checkbox is unchecked
        # only the "off" value reaches us. The parser strips the sentinel.
        checked = "checked" if value else ""
        return (
            hidden
            + _hidden("__value[]", "false")
            + f"<input type='checkbox' class='ct-bool' name='__value[]' "
              f"value='true' {checked}>"
        )
    if tag == TYPE_INT:
        return (
            hidden
            + f"<input type='number' step='1' class='bx--text-input ct-input' "
              f"name='__value[]' value='{html.escape(safe_val)}'>"
        )
    if tag == TYPE_FLOAT:
        return (
            hidden
            + f"<input type='number' step='any' class='bx--text-input ct-input' "
              f"name='__value[]' value='{html.escape(safe_val)}'>"
        )
    if tag == TYPE_NULL:
        # Render as an empty text input so the operator can fill it in;
        # if they leave it blank we keep it null. The PARSER reads the
        # text and re-tags as TYPE_STR if filled.
        return (
            hidden
            + "<input type='text' class='bx--text-input ct-input' "
              "name='__value[]' value='' placeholder='(null)'>"
        )
    # TYPE_STR — secrets get masked; long strings get textareas.
    if _is_secret_path(path):
        return (
            hidden
            + f"<input type='password' autocomplete='off' "
              f"class='bx--text-input ct-input' name='__value[]' "
              f"value='{html.escape(safe_val)}'>"
        )
    if len(safe_val) > 80 or "\n" in safe_val:
        return (
            hidden
            + f"<textarea class='bx--text-area ct-textarea' "
              f"name='__value[]' rows='3'>{html.escape(safe_val)}</textarea>"
        )
    return (
        hidden
        + f"<input type='text' class='bx--text-input ct-input' "
          f"name='__value[]' value='{html.escape(safe_val)}'>"
    )


def _render_key_label(key_segment: str) -> str:
    """Make a JSON key visually friendly. Underscores → spaces,
    first-letter cap. Preserve `:` because model keys look like
    ``openai:gpt-4o`` and that colon is meaningful."""
    if not key_segment:
        return ""
    label = key_segment.replace("_", " ")
    return label[0].upper() + label[1:]


def _render_scalar_row(path: str, key_segment: str, value: Any) -> str:
    """One <label> + input row for a scalar inside an object."""
    return (
        f"<div class='ct-row'>"
        f"<label class='bx--label ct-row__label'>"
        f"{html.escape(_render_key_label(key_segment))}"
        f"<code class='ct-row__key' aria-hidden='true'>{html.escape(key_segment)}</code>"
        f"</label>"
        f"<div class='ct-row__value'>{_input_for_scalar(path, value)}</div>"
        f"</div>"
    )


def _render_empty_marker(path: str) -> str:
    """Hidden marker that says "this container exists but is empty so
    the parser must rebuild it as {} / [] even though no leaves point
    inside". Important for sections like ``mcp.command_allowlist: []``
    where losing the empty list would change validation."""
    return _hidden("__container[]", path)


def _render_list_of_scalars(path: str, items: list[Any]) -> str:
    """Render a list whose elements are all strings/ints/etc."""
    out: list[str] = []
    out.append(f"<div class='ct-list ct-list--scalars' data-path='{html.escape(path)}'>")
    if not items:
        out.append(_render_empty_marker(path))
    for i, item in enumerate(items):
        item_path = _join_path(path, str(i))
        out.append(
            f"<div class='ct-list__item'>"
            f"<span class='ct-list__index'>{i}</span>"
            f"<div class='ct-list__value'>{_input_for_scalar(item_path, item)}</div>"
            f"<button type='button' class='ct-remove' "
            f"data-target='item' aria-label='Remove entry'>&times;</button>"
            f"</div>"
        )
    sample = items[0] if items else ""
    template_tag = _python_type_tag(sample)
    out.append(
        f"<button type='button' class='bx--btn bx--btn--tertiary ct-add' "
        f"data-collection='list_scalar' data-path='{html.escape(path)}' "
        f"data-type='{template_tag}'>+ Add entry</button>"
    )
    out.append("</div>")
    return "".join(out)


def _render_dict_of_scalars(path: str, mapping: dict[str, Any]) -> str:
    """Render a dict whose values are all scalars (e.g. max_tokens_per_role)."""
    out: list[str] = []
    out.append(f"<div class='ct-dict ct-dict--scalars' data-path='{html.escape(path)}'>")
    if not mapping:
        out.append(_render_empty_marker(path))
    for key in sorted(mapping.keys()):
        val = mapping[key]
        child_path = _join_path(path, key)
        out.append(
            f"<div class='ct-row ct-row--dict-entry'>"
            f"<label class='bx--label ct-row__label' for='ct-{html.escape(child_path)}'>"
            f"{html.escape(_render_key_label(key))}"
            f"<code class='ct-row__key' aria-hidden='true'>{html.escape(key)}</code>"
            f"</label>"
            f"<div class='ct-row__value'>{_input_for_scalar(child_path, val)}</div>"
            f"<button type='button' class='ct-remove' data-target='row' "
            f"aria-label='Remove entry'>&times;</button>"
            f"</div>"
        )
    sample = next(iter(mapping.values())) if mapping else ""
    template_tag = _python_type_tag(sample)
    out.append(
        f"<div class='ct-add-row'>"
        f"<input type='text' class='bx--text-input ct-new-key' "
        f"placeholder='New key'>"
        f"<button type='button' class='bx--btn bx--btn--tertiary ct-add' "
        f"data-collection='dict_scalar' data-path='{html.escape(path)}' "
        f"data-type='{template_tag}'>+ Add</button>"
        f"</div>"
    )
    out.append("</div>")
    return "".join(out)


def _render_list_of_records(path: str, items: list[dict[str, Any]]) -> str:
    """Render a list of records (e.g. mcp.servers).

    The list itself is extensible (+ Add / × remove record) but each
    record body has a FIXED schema — its inner fields are "dependent"
    on the record's identity (per the JSON shape) so we recurse with
    ``allow_add_keys=False`` to suppress inner add/remove affordances.

    + Add: clicking the button instantly inserts a new record card
    populated with the same schema as existing entries (or, if the
    collection is empty, the default schema registered for ``path``).
    No key prompt — list-of-records entries are index-addressed.
    """
    out: list[str] = []
    out.append(f"<div class='ct-list ct-list--records' data-path='{html.escape(path)}'>")
    if not items:
        out.append(_render_empty_marker(path))
    template_record = items[0] if items else (default_record_schema(path) or {})
    for i, record in enumerate(items):
        item_path = _join_path(path, str(i))
        title = html.escape(str(record.get("name") or record.get("label") or f"#{i}"))
        out.append(
            f"<details class='ct-record' data-list-index='{i}' open>"
            f"<summary class='ct-record__head'>"
            f"<span class='ct-record__title'>{title}</span>"
            f"<span class='ct-record__sub muted'>#{i}</span>"
            f"<button type='button' class='ct-remove ct-remove--record' "
            f"data-target='record' aria-label='Remove record'>&times;</button>"
            f"</summary>"
            f"<div class='ct-record__body'>"
            + render_tree(record, item_path, depth=1, allow_add_keys=False)
            + "</div></details>"
        )
    # Template (hidden) used by JS when + is clicked. Placeholders:
    #   __INDEX__ → the next list index
    if template_record:
        next_index = "__INDEX__"
        template_path = _join_path(path, next_index)
        template_record_blank = {k: _blank_for(v) for k, v in template_record.items()}
        template_html = (
            "<details class='ct-record' data-list-index='__INDEX__' open>"
            "<summary class='ct-record__head'>"
            "<span class='ct-record__title'>New entry</span>"
            "<span class='ct-record__sub muted'>#__INDEX__</span>"
            "<button type='button' class='ct-remove ct-remove--record' "
            "data-target='record' aria-label='Remove record'>&times;</button>"
            "</summary>"
            "<div class='ct-record__body'>"
            + render_tree(template_record_blank, template_path, depth=1, allow_add_keys=False)
            + "</div></details>"
        )
    else:
        template_html = (
            "<details class='ct-record' data-list-index='__INDEX__' open>"
            "<summary class='ct-record__head'>"
            "<span class='ct-record__title'>New entry</span>"
            "<span class='ct-record__sub muted'>#__INDEX__</span>"
            "<button type='button' class='ct-remove ct-remove--record' "
            "data-target='record'>&times;</button>"
            "</summary>"
            "<div class='ct-record__body'>"
            + _render_scalar_row(_join_path(path, "__INDEX__/value"), "value", "")
            + "</div></details>"
        )
    noun = _record_noun(path)
    out.append(
        f"<template class='ct-template'>{template_html}</template>"
        f"<button type='button' class='bx--btn bx--btn--tertiary ct-add' "
        f"data-collection='list_record' data-path='{html.escape(path)}'>"
        f"+ Add {html.escape(noun)}</button>"
    )
    out.append("</div>")
    return "".join(out)


def _record_noun(path: str) -> str:
    """Operator-facing singular noun for "+ Add <noun>" buttons. Derived
    from the last segment of the collection path with a few hand-tuned
    plurals demunged. Falls back to "entry"."""
    last = path.rsplit("/", 1)[-1] if path else ""
    irregular = {"models": "model", "servers": "server", "jobs": "job"}
    if last in irregular:
        return irregular[last]
    if last.endswith("ies"):
        return last[:-3] + "y"
    if last.endswith("s") and not last.endswith("ss"):
        return last[:-1]
    return last or "entry"


def _render_dict_of_records(path: str, mapping: dict[str, Any]) -> str:
    """Render a dict whose values are all records — the canonical case
    is the LLM registry (``models``).

    Each record's INNER fields are a fixed schema bound to that record
    type (an LLM has a provider, model_id, costs, etc. — not an
    operator-extensible vocabulary). We recurse with
    ``allow_add_keys=False`` so the record body shows labeled rows
    only — no inner + Add, no per-row × delete. The record AS A WHOLE
    can still be removed via the × on its header.

    + Add: clicking the button instantly inserts a new record card
    populated with the same schema as existing entries (or, if the
    collection is empty, the default schema registered for ``path``).
    The new record's KEY appears as an editable text input in the
    header; JS updates the child ``__path[]`` inputs in lockstep when
    the operator renames the key.
    """
    out: list[str] = []
    out.append(f"<div class='ct-dict ct-dict--records' data-path='{html.escape(path)}'>")
    if not mapping:
        out.append(_render_empty_marker(path))
    keys = sorted(mapping.keys())
    # Pick the template shape: first existing entry, else the registered
    # default schema for this collection path, else empty.
    if keys:
        template_record = mapping[keys[0]]
    else:
        template_record = default_record_schema(path) or {}

    for key in keys:
        record = mapping[key]
        child_path = _join_path(path, key)
        subtitle = html.escape(str(
            record.get("provider") or record.get("model_id") or record.get("backend") or ""
        ))
        out.append(
            f"<details class='ct-record' data-dict-key='{html.escape(key)}' open>"
            f"<summary class='ct-record__head'>"
            f"<span class='ct-record__title'>{html.escape(key)}</span>"
            f"<span class='ct-record__sub muted'>{subtitle}</span>"
            f"<button type='button' class='ct-remove ct-remove--record' "
            f"data-target='record' aria-label='Remove record'>&times;</button>"
            f"</summary>"
            f"<div class='ct-record__body'>"
            + render_tree(record, child_path, depth=1, allow_add_keys=False)
            + "</div></details>"
        )

    # Template for + Add. Always rendered (uses default schema when the
    # collection is empty) so the button always produces a usable record.
    if template_record:
        template_key = "__NEW_KEY__"
        template_path = _join_path(path, template_key)
        template_record_blank = {k: _blank_for(v) for k, v in template_record.items()}
        # In the template the key is an EDITABLE input so the operator
        # can name the new entry inline. JS rewrites child __path[]
        # entries whenever the input changes.
        template_html = (
            "<details class='ct-record' data-dict-key='__NEW_KEY__' open>"
            "<summary class='ct-record__head'>"
            "<input class='ct-record__key-input' type='text' "
            "value='__NEW_KEY__' aria-label='Record key' "
            "data-key-editor='1' onclick='event.stopPropagation()'>"
            "<span class='ct-record__sub muted'>new</span>"
            "<button type='button' class='ct-remove ct-remove--record' "
            "data-target='record'>&times;</button>"
            "</summary>"
            "<div class='ct-record__body'>"
            + render_tree(template_record_blank, template_path, depth=1, allow_add_keys=False)
            + "</div></details>"
        )
    else:
        template_html = ""

    noun = _record_noun(path)
    out.append(
        f"<template class='ct-template'>{template_html}</template>"
        f"<button type='button' class='bx--btn bx--btn--tertiary ct-add' "
        f"data-collection='dict_record' data-path='{html.escape(path)}'>"
        f"+ Add {html.escape(noun)}</button>"
    )
    out.append("</div>")
    return "".join(out)


def _blank_for(value: Any) -> Any:
    """A "same-shape but empty" copy of a value, used to seed a new
    record's defaults so a freshly-added LLM has all the expected
    sub-fields ready to fill in."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 0
    if isinstance(value, float):
        return 0.0
    if isinstance(value, str):
        return ""
    if isinstance(value, list):
        return []
    if isinstance(value, dict):
        return {k: _blank_for(v) for k, v in value.items()}
    return None


def _render_object(path: str, mapping: dict[str, Any], depth: int) -> str:
    """Render a dict with mixed scalar / nested children as labeled
    rows — no + Add button. Used for known-schema sections (the
    top-level config sections like ``sandbox``) where the operator
    shouldn't add arbitrary new keys."""
    out: list[str] = []
    out.append(f"<div class='ct-object' data-path='{html.escape(path)}'>")
    if not mapping:
        out.append(_render_empty_marker(path))
    for key in sorted(mapping.keys()):
        child = mapping[key]
        child_path = _join_path(path, key)
        if isinstance(child, (dict, list)):
            out.append(
                f"<details class='ct-nested' open>"
                f"<summary>{html.escape(_render_key_label(key))} "
                f"<code class='muted ct-row__key'>{html.escape(key)}</code>"
                f"</summary>"
                f"<div class='ct-nested__body'>"
                + render_tree(child, child_path, depth + 1) +
                "</div></details>"
            )
        else:
            out.append(_render_scalar_row(child_path, key, child))
    out.append("</div>")
    return "".join(out)


def render_tree(
    value: Any, path: str = "", depth: int = 0,
    *, allow_add_keys: bool = True,
) -> str:
    """Render any JSON-like value as a nested HTML form tree.

    ``path`` is the JSON-pointer-ish path to ``value`` from the section
    root (e.g. ``"models/openai:gpt-4o"``). The leaves embed their
    ``path`` in hidden ``__path[]`` fields so the POST handler can
    rebuild the original nested shape.

    ``allow_add_keys`` controls whether dicts get a "+ Add" affordance
    for new keys. The dashboard sets this to False at the *top level*
    of known-schema sections (``sandbox``, ``dashboard``, etc.) where
    the validator only accepts a closed set of keys; nested dicts deeper
    in the tree always allow +.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        # Scalar leaf at the very root — bare input.
        return _input_for_scalar(path, value)

    # If the value is empty BUT this path has a registered default
    # record schema, route to the records renderer so + Add still
    # produces a usable template (canonical case: schedule.jobs ships
    # empty, but + Add a job should yield a full job record).
    is_empty_container = (
        isinstance(value, (list, dict)) and not value
    )
    if is_empty_container and default_record_schema(path) is not None:
        if isinstance(value, list):
            return _render_list_of_records(path, [])
        return _render_dict_of_records(path, {})

    kind = infer_collection_kind(value)
    if kind == COLLECTION_DICT_OF_RECORDS:
        return _render_dict_of_records(path, value)
    if kind == COLLECTION_LIST_OF_RECORDS and isinstance(value, list):
        return _render_list_of_records(path, value)
    if kind == COLLECTION_LIST_OF_SCALARS and isinstance(value, list):
        return _render_list_of_scalars(path, value)
    if kind == COLLECTION_DICT_OF_SCALARS and isinstance(value, dict):
        if allow_add_keys:
            return _render_dict_of_scalars(path, value)
        return _render_object(path, value, depth)
    # Mixed dict — labeled object rows, nested dicts recurse with
    # allow_add_keys=True so the operator can grow them.
    if isinstance(value, dict):
        return _render_object(path, value, depth)
    # Opaque value (shouldn't happen often) — JSON textarea fallback.
    return (
        _hidden("__path[]", path)
        + _hidden("__type[]", TYPE_JSON)
        + f"<textarea class='bx--text-area ct-textarea' "
          f"name='__value[]' rows='4'>"
          f"{html.escape(json.dumps(value, indent=2))}</textarea>"
    )


# ---------------------------------------------------------------------------
# 5. Parsing — turn the form payload back into a nested Python value
# ---------------------------------------------------------------------------

class TreeParseError(ValueError):
    """Raised when a tree submission can't be parsed back into a
    nested dict (mismatched array lengths, unknown type tags,
    malformed numeric segments, etc.)."""


def _normalize_form_field(raw: Any) -> list[str]:
    """The form parser delivers single-valued keys as strings and
    multi-valued keys as lists. Normalize to a list."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return [str(raw)]


def _coerce_value(tag: str, raw: str) -> Any:
    """Coerce a raw form string into the declared Python type."""
    if tag == TYPE_NULL:
        # Operator left it null → keep null. If they filled it in we
        # promote to string here; the validator will reject mismatches.
        return None if raw == "" else raw
    if tag == TYPE_STR:
        return raw
    if tag == TYPE_BOOL:
        return raw.lower() in ("1", "true", "on", "yes")
    if tag == TYPE_INT:
        if raw == "":
            return 0
        try:
            return int(raw)
        except ValueError as exc:
            raise TreeParseError(f"not an int: {raw!r}") from exc
    if tag == TYPE_FLOAT:
        if raw == "":
            return 0.0
        try:
            return float(raw)
        except ValueError as exc:
            raise TreeParseError(f"not a number: {raw!r}") from exc
    if tag == TYPE_JSON:
        if raw == "":
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TreeParseError(f"not valid JSON: {exc}") from exc
    raise TreeParseError(f"unknown type tag: {tag!r}")


def _ensure_container_for(root_kind: str) -> Any:
    return [] if root_kind == "list" else {}


def _set_at_path(root: Any, segments: list[str], value: Any) -> Any:
    """Walk ``segments`` from ``root``, creating intermediate dicts /
    lists as needed; set the leaf to ``value``. Numeric segments are
    treated as list indices ONLY when the current container is already
    a list (or empty/missing — in which case we materialise a list).

    Returns the (possibly new) root.
    """
    if not segments:
        return value
    # Decide root shape.
    if root is None:
        root = [] if _INT_SEGMENT.match(segments[0]) else {}
    cur: Any = root
    for i, seg in enumerate(segments[:-1]):
        next_seg = segments[i + 1]
        next_is_int = bool(_INT_SEGMENT.match(next_seg))
        if isinstance(cur, list):
            idx = int(seg)
            while len(cur) <= idx:
                cur.append({} if not next_is_int else [])
            if cur[idx] is None or (isinstance(cur[idx], list) and not next_is_int) \
                    or (isinstance(cur[idx], dict) and next_is_int):
                cur[idx] = {} if not next_is_int else []
            cur = cur[idx]
        else:
            # dict
            if seg not in cur or cur[seg] is None \
                    or (isinstance(cur.get(seg), list) and not next_is_int) \
                    or (isinstance(cur.get(seg), dict) and next_is_int):
                cur[seg] = {} if not next_is_int else []
            cur = cur[seg]
    last = segments[-1]
    if isinstance(cur, list):
        idx = int(last)
        while len(cur) <= idx:
            cur.append(None)
        cur[idx] = value
    else:
        cur[last] = value
    return root


def _materialize_container(root: Any, segments: list[str]) -> Any:
    """Ensure the container at ``segments`` exists (as {} or []) even
    if no leaves point inside it. Used so empty lists like
    ``mcp.command_allowlist: []`` survive a round-trip."""
    if not segments:
        # Whole section is empty — caller decides root shape.
        return root if root is not None else {}
    next_is_int = bool(_INT_SEGMENT.match(segments[-1]))
    return _set_at_path(
        root, segments,
        [] if next_is_int else {},
    ) if False else _set_at_path(  # ensure key exists, even if value is empty
        root, segments[:-1],
        _ensure_empty_at(root, segments),
    )


def _ensure_empty_at(root: Any, segments: list[str]) -> Any:
    """Pre-existing intermediate value at the path, or a fresh empty
    container shaped by the last segment."""
    cur: Any = root
    for seg in segments[:-1]:
        if isinstance(cur, list):
            try:
                cur = cur[int(seg)]
            except (IndexError, ValueError):
                return {}
        elif isinstance(cur, dict):
            cur = cur.get(seg)
        else:
            return {}
    if isinstance(cur, dict):
        existing = cur.get(segments[-1])
    elif isinstance(cur, list):
        try:
            existing = cur[int(segments[-1])]
        except (IndexError, ValueError):
            existing = None
    else:
        existing = None
    if isinstance(existing, (dict, list)):
        return existing
    return {}


def parse_tree(form_data: dict[str, Any]) -> Any:
    """Reconstruct the nested value from a POSTed tree form.

    The form payload carries three parallel arrays:
      - ``__path[]``  — slash-joined paths
      - ``__type[]``  — type tags
      - ``__value[]`` — raw string values

    Plus an optional ``__container[]`` array of paths whose containers
    must exist even when empty (so deleting the last entry of a list
    doesn't drop the empty list from the saved config).
    """
    paths = _normalize_form_field(form_data.get("__path[]"))
    types = _normalize_form_field(form_data.get("__type[]"))
    values = _normalize_form_field(form_data.get("__value[]"))
    if not (len(paths) == len(types) == len(values)):
        raise TreeParseError(
            f"length mismatch: __path={len(paths)} __type={len(types)} "
            f"__value={len(values)}"
        )

    # Bool fields emit TWO entries each: a hidden "false" sentinel plus
    # the checkbox's "true" only when checked. Collapse consecutive
    # entries with the same path: a "true" wins, "false" loses.
    collapsed: list[tuple[list[str], str, str]] = []
    seen_index_by_path: dict[str, int] = {}
    for path, tag, val in zip(paths, types, values):
        if tag not in _KNOWN_TYPES:
            raise TreeParseError(f"unknown __type {tag!r} for path {path!r}")
        segments = _parse_path(path)
        if path in seen_index_by_path and tag == TYPE_BOOL:
            # Two bool entries for the same path — keep the "true" one.
            idx = seen_index_by_path[path]
            prev_val = collapsed[idx][2].lower()
            new_val = val.lower()
            # "true" wins; anything else stays as previously stored.
            if new_val in ("1", "true", "on", "yes"):
                collapsed[idx] = (segments, tag, val)
            elif prev_val not in ("1", "true", "on", "yes"):
                collapsed[idx] = (segments, tag, val)
            continue
        seen_index_by_path[path] = len(collapsed)
        collapsed.append((segments, tag, val))

    # Decide root shape — list if every top-level segment is an int.
    if all(seg and _INT_SEGMENT.match(seg[0]) for seg, _t, _v in collapsed) \
            and any(seg for seg, _t, _v in collapsed):
        root: Any = []
    else:
        root = {}

    for segments, tag, raw in collapsed:
        if not segments:
            # Whole-section scalar (rare).
            return _coerce_value(tag, raw)
        coerced = _coerce_value(tag, raw)
        root = _set_at_path(root, segments, coerced)

    # Preserve empty containers.
    for cpath in _normalize_form_field(form_data.get("__container[]")):
        segments = _parse_path(cpath)
        if not segments:
            continue
        # Skip if the path already has any leaves under it — those win.
        already_set = _ensure_empty_at(root, segments)
        if isinstance(already_set, (list, dict)) and (already_set or _has_value(root, segments)):
            continue
        next_is_int = False  # default empty containers to dict
        root = _set_at_path(root, segments, [] if next_is_int else {})
    return root


def _has_value(root: Any, segments: list[str]) -> bool:
    """True if the path resolves to a non-empty existing value."""
    cur: Any = root
    for seg in segments:
        if isinstance(cur, dict):
            if seg not in cur:
                return False
            cur = cur[seg]
        elif isinstance(cur, list):
            try:
                cur = cur[int(seg)]
            except (IndexError, ValueError):
                return False
        else:
            return False
    if isinstance(cur, (list, dict)):
        return bool(cur)
    return cur is not None
