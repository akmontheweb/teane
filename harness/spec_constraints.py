"""Carry the spec's documented field constraints into the generated models.

The specification's data-model table states them plainly:

    | `first_name` | TEXT | NOT NULL, length 1-100 after trimming |

and the generated model does not:

    first_name: str = Field(..., description="Employee first name")

Nothing downstream can recover the rule from that. Each tier then invents its
own answer and they disagree — which is how lumina-run20-20260925-1342 ended.
Its property test read the unconstrained annotation and asserted that ANY
string up to 200 characters must construct:

    @given(first_name=st.text(max_size=200), ...)
    def test_birthday_create_roundtrip(...)

while other tests demanded invalid names be rejected. Repair could satisfy
either, never both, and the judge flipped every round — "relax the last_name
validator", "remove the _validate_name function", "add field_validators that
reject..." — until three distraction-loop trips ended the run with the
validators deleted and the 422 tests about to fail again.

With ``min_length=1, max_length=100`` on the field, the contradiction cannot
form: the strategy generator derives its domain from the constraint, the
validation tests assert the same rule, and both trace to the spec.

Scope, deliberately narrow (see :mod:`harness.static_preflight` on
false-negative bias):

  * only columns the spec gives a NUMERIC length range for;
  * only ``str`` fields of REQUEST-shaped models — a response model echoes
    what is already stored, and demanding validators on it is noise;
  * only fields declaring no length constraint at all. A model that declares
    something different from the spec is a judgement call, not an omission,
    and is left alone.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: ``| `first_name` | TEXT | NOT NULL, length 1-100 after trimming |``
#: The column cell may or may not be backticked; the type cell is ignored.
_TABLE_ROW_RE = re.compile(
    r"^\|\s*`?(?P<col>[A-Za-z_][A-Za-z0-9_]*)`?\s*\|"
    r"(?P<type>[^|]*)\|(?P<rules>[^|]*)\|",
    re.MULTILINE,
)

#: ``length 1-100``, ``length 1–100`` (en dash), ``between 1 and 100``,
#: ``1..100 characters``. Unicode dashes are folded before matching.
_RANGE_RES = (
    re.compile(r"length\s+(?P<lo>\d+)\s*-\s*(?P<hi>\d+)", re.IGNORECASE),
    re.compile(r"between\s+(?P<lo>\d+)\s+and\s+(?P<hi>\d+)", re.IGNORECASE),
    re.compile(r"(?P<lo>\d+)\s*\.\.\s*(?P<hi>\d+)\s*char", re.IGNORECASE),
    re.compile(r"max(?:imum)?\s+length\s+(?P<hi>\d+)", re.IGNORECASE),
)

_DASHES = "‐‑‒–—―−"

#: Class-name suffixes that mark a model as a RESPONSE shape. Mirrors
#: acceptance_gen's list; a response echoes stored data, so a missing
#: constraint there is not a defect worth a diagnostic.
_RESPONSE_NAME_RE = re.compile(r"(Response|Payload|Out|Result|Envelope|Page)\w*$")

_SPEC_REL = os.path.join("docs", "SPEC_REQUIREMENTS.md")


def _fold(text: str) -> str:
    for d in _DASHES:
        text = text.replace(d, "-")
    return text


def parse_documented_lengths(spec_text: str) -> dict[str, dict[str, int]]:
    """Column name → ``{"min_length": n, "max_length": m}`` from the spec.

    Only rows stating a numeric range are returned; prose without numbers is
    not guessed at.
    """
    out: dict[str, dict[str, int]] = {}
    for m in _TABLE_ROW_RE.finditer(_fold(spec_text or "")):
        col = m.group("col")
        rules = m.group("rules")
        # No name-based skipping: a column really called `name` is common,
        # and excluding it by name would silently drop the very rule this
        # module exists to carry. A header row ("| Column | Type |
        # Constraints |") states no numeric range, so it falls out below.
        if not col:
            continue
        for rx in _RANGE_RES:
            hit = rx.search(rules)
            if not hit:
                continue
            groups = hit.groupdict()
            spec: dict[str, int] = {}
            if groups.get("lo") is not None:
                spec["min_length"] = int(groups["lo"])
            if groups.get("hi") is not None:
                spec["max_length"] = int(groups["hi"])
            if spec:
                out[col] = spec
            break
    return out


def _request_models(source: str, rel: str) -> list[Any]:
    from harness.contract_tests import parse_pydantic_models

    models = []
    for m in parse_pydantic_models(source, rel_path=rel) or []:
        if _RESPONSE_NAME_RE.search(m.name):
            continue
        models.append(m)
    return models


def unconstrained_fields(
    workspace: str, *, max_files: int = 300,
) -> list[dict[str, Any]]:
    """Fields the spec bounds and the model does not.

    Each item: ``{"file", "line", "model", "field", "min_length",
    "max_length"}``. Empty when the spec documents nothing, which is the
    common case for a spec without a data-model table.
    """
    spec_path = os.path.join(workspace, _SPEC_REL)
    if not os.path.isfile(spec_path):
        return []
    try:
        with open(spec_path, "r", encoding="utf-8", errors="replace") as fh:
            documented = parse_documented_lengths(fh.read())
    except OSError:
        return []
    if not documented:
        return []

    findings: list[dict[str, Any]] = []
    scanned = 0
    _skip = {"node_modules", ".venv", ".git", "__pycache__", "tests", "test"}
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in _skip and not d.startswith(".")]
        for name in files:
            if not name.endswith(".py") or scanned >= max_files:
                continue
            scanned += 1
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    source = fh.read()
            except OSError:
                continue
            if "BaseModel" not in source:
                continue
            rel = os.path.relpath(path, workspace)
            for model in _request_models(source, rel):
                for fs in model.fields:
                    rule = documented.get(fs.name)
                    if not rule:
                        continue
                    if _base_str(fs.type_str) is False:
                        continue
                    if fs.constraints.get("min_length") is not None or \
                            fs.constraints.get("max_length") is not None:
                        continue  # declares something; not an omission
                    findings.append({
                        "file": rel, "line": 0, "model": model.name,
                        "field": fs.name, **rule,
                    })
    return findings


def _base_str(type_str: str) -> bool:
    t = (type_str or "").replace("Optional[", "").replace("]", "").strip()
    return t == "str"


def constraint_diagnostics(
    workspace: str, *, limit: int = 10,
) -> list[dict[str, Any]]:
    """``compiler_errors``-shaped diagnostics for undeclared constraints."""
    out: list[dict[str, Any]] = []
    for f in unconstrained_fields(workspace)[:limit]:
        bits = []
        if f.get("min_length") is not None:
            bits.append(f"min_length={f['min_length']}")
        if f.get("max_length") is not None:
            bits.append(f"max_length={f['max_length']}")
        declared = ", ".join(bits)
        out.append({
            "file": f["file"],
            "line": 0,
            "column": 0,
            "severity": "error",
            "error_code": "SPEC_CONSTRAINT_NOT_DECLARED",
            "message": (
                f"{f['model']}.{f['field']} declares no length constraint, "
                f"but the specification's data model bounds it: "
                f"{declared}. Declare it on the field — "
                f"`{f['field']}: str = Field(..., {declared})` — so every "
                f"tier derives the same rule. Left undeclared, the "
                f"property-test generator reads the bare annotation and "
                f"asserts that ANY string is valid, while the validation "
                f"tests assert the opposite; no production change can "
                f"satisfy both and the repair loop oscillates."
            ),
            "semantic_context": (
                f"Specification data model: {f['field']} {declared}."
            ),
        })
    return out
