"""Form-schema derivation from the strict config validator.

The dashboard's config editor is form-based. Hand-curating a form per
config section would drift the moment a new key landed in
``harness/cli.py``'s validator tables, so we **derive** the form
schema from the same source the validator uses:

- ``_KNOWN_TOP_LEVEL_KEYS`` — the section names.
- ``_KNOWN_NESTED_KEYS[section]`` — the per-section field names.
- ``_TYPE_SCHEMA[\"section.field\"]`` — the runtime type tuple.

A round-trip test asserts every nested key in ``_TYPE_SCHEMA`` is
renderable + persistable through this layer; if someone lands a new
key without a type entry the form refuses to render the field (and
the round-trip test fails CI), so the validator stays the source of
truth.

The HTML renderer in :mod:`harness.dashboard` consumes the descriptors
this module produces. No HTML lives here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Form field descriptor
# ---------------------------------------------------------------------------

FORM_KIND_CHECKBOX = "checkbox"   # bool
FORM_KIND_NUMBER_INT = "number_int"
FORM_KIND_NUMBER_FLOAT = "number_float"
FORM_KIND_TEXT = "text"
FORM_KIND_TEXTAREA = "textarea"
FORM_KIND_JSON_LIST = "json_list"
FORM_KIND_JSON_DICT = "json_dict"


@dataclass
class FormField:
    """One renderable field in a config section.

    ``kind`` is one of the ``FORM_KIND_*`` constants. ``type_tuple`` is
    the validator's runtime-type tuple (preserved so the form's POST
    handler can route the parsed value through the same gate the
    validator uses).
    """

    section: str
    name: str
    kind: str
    type_tuple: tuple[type, ...]
    current_value: Any = None
    required: bool = False
    secret: bool = False  # render as <input type=password>; never echo on errors
    placeholder: str = ""

    @property
    def dotted_key(self) -> str:
        return f"{self.section}.{self.name}" if self.section else self.name


@dataclass
class FormSection:
    """All renderable fields in one section.

    ``section`` is the top-level config key. ``fields`` is in
    sorted-name order. Sections whose every type entry is missing fall
    out as empty, which is fine — the renderer skips them.
    """

    section: str
    fields: list[FormField] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. Type → form kind mapping
# ---------------------------------------------------------------------------

def kind_for_type_tuple(type_tuple: tuple[type, ...]) -> str:
    """Pick the render kind from the validator's type tuple.

    Order of preference (longest-match first so ``(int, float)`` picks
    float, not int):

    - ``(bool,)`` → checkbox
    - ``(int, float)`` or ``(float, ...)`` → number with step=any
    - ``(int,)`` → number with step=1
    - ``(list,)`` → JSON list textarea
    - ``(dict,)`` → JSON dict textarea
    - ``(str,)`` → text input
    - anything mixed or unknown → text input as the safest fallback
    """
    if not type_tuple:
        return FORM_KIND_TEXT
    s = set(type_tuple)
    if bool in s:
        return FORM_KIND_CHECKBOX
    if float in s:
        return FORM_KIND_NUMBER_FLOAT
    if int in s and float not in s:
        return FORM_KIND_NUMBER_INT
    if list in s:
        return FORM_KIND_JSON_LIST
    if dict in s:
        return FORM_KIND_JSON_DICT
    return FORM_KIND_TEXT


# ---------------------------------------------------------------------------
# 3. Building descriptors from the live validator tables
# ---------------------------------------------------------------------------

def build_section(
    section: str,
    *,
    current_config: Optional[dict[str, Any]] = None,
) -> FormSection:
    """Build a :class:`FormSection` for one top-level config section.

    Pulls field names from :data:`harness.cli._KNOWN_NESTED_KEYS` and
    types from :data:`harness.cli._TYPE_SCHEMA`. Fields without a type
    entry are quietly dropped (the round-trip test guards against this).
    """
    from harness.cli import _KNOWN_NESTED_KEYS, _TYPE_SCHEMA
    field_names = sorted(_KNOWN_NESTED_KEYS.get(section, frozenset()))
    out = FormSection(section=section)
    section_data = (current_config or {}).get(section) or {}
    for name in field_names:
        dotted = f"{section}.{name}"
        type_tuple = _TYPE_SCHEMA.get(dotted)
        if type_tuple is None:
            logger.debug("[web_forms] no type entry for %s; skipping", dotted)
            continue
        out.fields.append(FormField(
            section=section, name=name,
            kind=kind_for_type_tuple(type_tuple),
            type_tuple=type_tuple,
            current_value=section_data.get(name),
        ))
    return out


def all_sections(
    *, current_config: Optional[dict[str, Any]] = None,
) -> list[FormSection]:
    """Build form schemas for every known top-level section. Sections
    without any renderable fields are still returned (empty) so the UI
    can render a placeholder.
    """
    from harness.cli import _KNOWN_TOP_LEVEL_KEYS, _KNOWN_NESTED_KEYS
    out: list[FormSection] = []
    for section in sorted(_KNOWN_TOP_LEVEL_KEYS):
        # Top-level scalar keys (e.g. "build_command") aren't in
        # _KNOWN_NESTED_KEYS; render them as a single-field section
        # carrying the section name as the field name.
        if section not in _KNOWN_NESTED_KEYS:
            from harness.cli import _TYPE_SCHEMA
            type_tuple = _TYPE_SCHEMA.get(section)
            if type_tuple is None:
                # Scalar section without a typed entry (e.g. "models",
                # "model_routing" — render via per-section editors,
                # not the generic form). Skip from the generic editor.
                out.append(FormSection(section=section, fields=[]))
                continue
            section_data = (current_config or {})
            out.append(FormSection(
                section=section,
                fields=[FormField(
                    section="", name=section,
                    kind=kind_for_type_tuple(type_tuple),
                    type_tuple=type_tuple,
                    current_value=section_data.get(section),
                )],
            ))
            continue
        out.append(build_section(section, current_config=current_config))
    return out


# ---------------------------------------------------------------------------
# 4. Parsing form POST data back into typed Python values
# ---------------------------------------------------------------------------

class FormParseError(ValueError):
    """Raised when a form field value can't be parsed into its declared
    type. Carries the offending dotted key + the operator-facing error
    message so the renderer can show a per-field error."""

    def __init__(self, dotted_key: str, message: str):
        super().__init__(f"{dotted_key}: {message}")
        self.dotted_key = dotted_key
        self.message = message


def parse_value(field_: FormField, raw: Any) -> Any:
    """Parse a single field's raw POST value into its declared type.

    HTML form values arrive as strings (or lists for multi-valued
    fields); checkboxes are present-or-absent. We coerce to the
    validator's expected type and raise :class:`FormParseError` on
    failure so the renderer can surface a per-field error.
    """
    kind = field_.kind
    if kind == FORM_KIND_CHECKBOX:
        if isinstance(raw, bool):
            return raw
        if raw is None or raw == "":
            return False
        if isinstance(raw, str):
            return raw.lower() in ("1", "true", "on", "yes")
        return bool(raw)
    if kind == FORM_KIND_NUMBER_INT:
        if raw is None or raw == "":
            raise FormParseError(field_.dotted_key, "value required (integer)")
        try:
            return int(str(raw).strip())
        except (TypeError, ValueError) as exc:
            raise FormParseError(field_.dotted_key, f"not a valid integer: {exc}") from exc
    if kind == FORM_KIND_NUMBER_FLOAT:
        if raw is None or raw == "":
            raise FormParseError(field_.dotted_key, "value required (number)")
        try:
            return float(str(raw).strip())
        except (TypeError, ValueError) as exc:
            raise FormParseError(field_.dotted_key, f"not a valid number: {exc}") from exc
    if kind == FORM_KIND_JSON_LIST:
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return []
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            raise FormParseError(field_.dotted_key, f"not valid JSON: {exc}") from exc
        if not isinstance(parsed, list):
            raise FormParseError(field_.dotted_key, "JSON must be a list")
        return parsed
    if kind == FORM_KIND_JSON_DICT:
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return {}
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            raise FormParseError(field_.dotted_key, f"not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise FormParseError(field_.dotted_key, "JSON must be an object")
        return parsed
    # text / unknown — pass through stringified.
    if raw is None:
        return ""
    return str(raw)


def parse_section_post(
    section: FormSection, post_data: dict[str, Any],
) -> tuple[dict[str, Any], list[FormParseError]]:
    """Parse a posted form payload back into a section dict.

    ``post_data`` is a mapping of dotted-key → raw form value (string
    or list, as multipart/x-www-form-urlencoded delivers). Missing
    checkboxes are interpreted as ``False`` (HTML omits absent
    checkboxes from the POST body).

    Returns ``(section_dict, errors)`` so the caller can re-render the
    form with per-field errors when validation fails. The ``section_dict``
    is **always** present even when ``errors`` is non-empty so the
    operator's other typed values aren't lost.
    """
    out: dict[str, Any] = {}
    errors: list[FormParseError] = []
    for f in section.fields:
        raw = post_data.get(f.dotted_key, None)
        # Lists from form parsing arrive as the first/last; pick last
        # to honour "value last write wins".
        if isinstance(raw, list):
            raw = raw[-1] if raw else None
        try:
            value = parse_value(f, raw)
        except FormParseError as exc:
            errors.append(exc)
            continue
        out[f.name] = value
    return out, errors


# ---------------------------------------------------------------------------
# 5. Coverage check — round-trip validator vs. form schema
# ---------------------------------------------------------------------------

def renderable_dotted_keys() -> set[str]:
    """Every dotted key the form schema can render. The round-trip test
    asserts this equals (or is a near-equal of) the validator's
    ``_TYPE_SCHEMA`` keys. New config keys land in both lists or fail
    the test."""
    sections = all_sections(current_config=None)
    keys: set[str] = set()
    for s in sections:
        for f in s.fields:
            keys.add(f.dotted_key)
    return keys
