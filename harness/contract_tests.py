"""Deterministic contract-test emitter — ADR-0003 Tier 1 (schema-declarative).

Generates provably-correct unit tests for a model's *declarative* constraints
directly from the AST, with no LLM judgment. A Pydantic ``Field(max_length=N)``
is a machine-readable spec of its own validation contract; re-deriving it in
prose is what the LLM does inconsistently (session 019f803f). Here we read the
contract and emit the test.

Scope (Tier 1): Pydantic v2 models. Declarative constraints only —
``max_length`` / ``min_length`` / ``ge`` / ``le`` / ``gt`` / ``lt``, and
required-vs-optional. Custom ``@field_validator`` / ``@model_validator``
bodies are the imperative gray zone and are left to the LLM (Tier 4); this
module never guesses their semantics.

Two hard invariants:

1. **Never import the model.** The harness runs untrusted generated code;
   introspection is AST-only. No ``importlib``, no ``exec``.
2. **Conservative — skip what you can't prove.** If a field's type can't be
   given a synthesised valid value, the whole model is skipped rather than
   emit a test that might construct an invalid instance. A missing test is
   safe; a wrong one reintroduces the 019f803f failure. Same bias as
   ``harness.test_contradiction``.

Pure module: AST + string rendering only, no harness imports, no I/O beyond
the caller handing in source text. Fully unit-testable in isolation. The
node-side writer (``emit_contract_tests``) is the only function that touches
the filesystem and mirrors ``test_generation._emit_nfr_stubs``' shape.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "FieldSpec",
    "ModelSpec",
    "parse_pydantic_models",
    "render_contract_test",
    "emit_contract_tests",
]

# Pydantic base classes we recognise as "this is a validated model". Kept
# narrow on purpose — a class inheriting something else entirely may not have
# Pydantic construction semantics, so we don't guess.
_PYDANTIC_BASES = frozenset({"BaseModel"})

# Field(...) keyword constraints we can turn into a deterministic assertion.
_LEN_CONSTRAINTS = ("max_length", "min_length")
_NUM_CONSTRAINTS = ("ge", "le", "gt", "lt")


@dataclass(frozen=True)
class FieldSpec:
    """One model field's declaratively-checkable contract."""
    name: str
    type_str: str                       # normalised annotation, e.g. "str", "Optional[int]"
    required: bool                       # non-Optional AND no default
    constraints: dict[str, object] = field(default_factory=dict)
    has_custom_validator: bool = False   # a @field_validator targets this field


@dataclass(frozen=True)
class ModelSpec:
    name: str
    fields: tuple[FieldSpec, ...]
    module_import: str                   # dotted path guess for the import line
    has_model_validator: bool = False    # a @model_validator can reject any instance


def _ann_to_str(node: Optional[ast.expr]) -> str:
    """Best-effort annotation text via ast.unparse; '' when absent."""
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 — unparse is best-effort
        return ""


def _is_optional(type_str: str) -> bool:
    t = type_str.replace(" ", "")
    return (
        t.startswith("Optional[")
        or "|None" in t
        or t.startswith("None|")
        or t.endswith("|None")
    )


def _base_type(type_str: str) -> str:
    """Strip Optional[...] / '| None' to the inner scalar type name."""
    t = type_str.strip()
    if t.startswith("Optional[") and t.endswith("]"):
        t = t[len("Optional["):-1]
    t = t.replace(" ", "")
    for suffix in ("|None",):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
    for prefix in ("None|",):
        if t.startswith(prefix):
            t = t[len(prefix):]
    return t.strip()


def _extract_field_call_constraints(value: ast.expr) -> tuple[dict[str, object], bool]:
    """Return (constraints, has_default) from a field's RHS.

    Recognises ``Field(default, max_length=.., ge=.., ...)``. ``has_default``
    is True when a default value is expressed (first positional arg that isn't
    ``...``, or a ``default=``/``default_factory=`` kw, or a bare literal RHS).
    """
    constraints: dict[str, object] = {}
    has_default = False
    if isinstance(value, ast.Call) and _call_name(value) == "Field":
        # positional default: Field(default, ...) — Ellipsis means "required"
        if value.args:
            first = value.args[0]
            if not (isinstance(first, ast.Constant) and first.value is Ellipsis):
                has_default = True
        for kw in value.keywords:
            if kw.arg in ("default", "default_factory"):
                has_default = True
            elif kw.arg in _LEN_CONSTRAINTS + _NUM_CONSTRAINTS:
                lit = _literal(kw.value)
                if lit is not None:
                    constraints[kw.arg] = lit
    elif value is not None and not (
        isinstance(value, ast.Constant) and value.value is Ellipsis
    ):
        # Bare default: ``x: int = 5`` — a real default value present.
        has_default = True
    return constraints, has_default


def _call_name(node: ast.Call) -> Optional[str]:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _literal(node: ast.expr) -> Optional[object]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    # unary minus (ge=-1)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -node.operand.value
    return None


def _module_dotted(rel_path: str) -> str:
    """server/app/schemas/contact.py -> server.app.schemas.contact."""
    p = rel_path
    if p.endswith(".py"):
        p = p[:-3]
    return p.replace("/", ".").replace("\\", ".").strip(".")


def _validator_targets(cls: ast.ClassDef) -> tuple[set[str], bool]:
    """Return ``(field_validator_targets, has_model_validator)``.

    - ``field_validator_targets``: field names named by a
      ``@field_validator('x', ...)`` — those fields carry imperative
      semantics we do NOT model deterministically.
    - ``has_model_validator``: any ``@model_validator`` (or bare
      ``@validator``/``@root_validator``) is present. Such a validator can
      reject *any* constructed instance for reasons opaque to the AST, so a
      model that has one is skipped entirely — we cannot prove a valid
      instance exists to build the happy-path / required-omission tests.
    """
    targets: set[str] = set()
    has_model_validator = False
    for item in cls.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in item.decorator_list:
            name = None
            if isinstance(dec, ast.Call):
                name = _call_name(dec)
            elif isinstance(dec, ast.Name):
                name = dec.id
            elif isinstance(dec, ast.Attribute):
                name = dec.attr
            if name in ("model_validator", "root_validator", "validator"):
                has_model_validator = True
            if name == "field_validator" and isinstance(dec, ast.Call):
                for a in dec.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        targets.add(a.value)
    return targets, has_model_validator


def parse_pydantic_models(source: str, *, rel_path: str = "") -> list[ModelSpec]:
    """Parse ``source`` and return a ModelSpec per Pydantic model found.

    Empty on parse failure (the file's syntax is the LLM's problem, surfaced
    elsewhere) or when no recognised model is present.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    module_dotted = _module_dotted(rel_path) if rel_path else ""
    out: list[ModelSpec] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = {
            (b.id if isinstance(b, ast.Name) else
             b.attr if isinstance(b, ast.Attribute) else "")
            for b in node.bases
        }
        if not (base_names & _PYDANTIC_BASES):
            continue
        validator_fields, has_model_validator = _validator_targets(node)
        fields: list[FieldSpec] = []
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
                continue
            fname = item.target.id
            if fname.startswith("_") or fname == "model_config":
                continue
            type_str = _ann_to_str(item.annotation)
            constraints, has_default = _extract_field_call_constraints(item.value) if item.value is not None else ({}, False)
            required = not _is_optional(type_str) and not has_default
            fields.append(FieldSpec(
                name=fname,
                type_str=type_str,
                required=required,
                constraints=constraints,
                has_custom_validator=fname in validator_fields,
            ))
        if fields:
            out.append(ModelSpec(
                name=node.name,
                fields=tuple(fields),
                module_import=module_dotted,
                has_model_validator=has_model_validator,
            ))
    return out


# --- valid-value synthesis (conservative) ----------------------------------

def _valid_value(fs: FieldSpec) -> Optional[str]:
    """A Python source literal that is a VALID value for this field's
    declarative constraints, or None when we can't prove one (→ skip model).

    Deliberately narrow: str / int / float / bool and their Optional wrappers.
    Anything else (nested models, containers, date, enums, custom types)
    returns None so the caller skips the whole model rather than risk an
    invalid instance.
    """
    base = _base_type(fs.type_str)
    c = fs.constraints
    if base == "str":
        max_len = c.get("max_length")
        min_len = c.get("min_length")
        n = 3
        if isinstance(min_len, (int, float)):
            n = max(n, int(min_len))
        if isinstance(max_len, (int, float)):
            n = min(n, int(max_len))
            if n < (int(min_len) if isinstance(min_len, (int, float)) else 0):
                return None  # unsatisfiable declared range — don't guess
        return repr("a" * max(n, 0))
    if base in ("int", "float"):
        lo = c.get("ge")
        if "gt" in c and isinstance(c["gt"], (int, float)):
            lo = c["gt"] + 1 if base == "int" else c["gt"] + 1
        hi = c.get("le")
        if "lt" in c and isinstance(c["lt"], (int, float)):
            hi = c["lt"] - 1 if base == "int" else c["lt"] - 1
        val: float
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            if lo > hi:
                return None
            val = lo
        elif isinstance(lo, (int, float)):
            val = lo
        elif isinstance(hi, (int, float)):
            val = hi
        else:
            val = 1
        return repr(int(val) if base == "int" else float(val))
    if base == "bool":
        return "True"
    return None


def _valid_kwargs(model: ModelSpec) -> Optional[dict[str, str]]:
    """Minimal valid kwargs (source literals) covering every REQUIRED field,
    or None when a provably-valid instance can't be synthesised.

    Returns None (→ skip the whole model) when:

    - the model has a ``@model_validator`` — it may reject any instance for
      reasons opaque to the AST (lumina ``ContactUpdate`` has one; lumina
      ``ContactCreate`` requires a real ISO ``date_of_birth`` via a field
      validator, so ``'aaa'`` — a declaratively-valid str — is actually
      rejected). Emitting a happy-path test there asserts success on an
      instance that raises: the exact tautology this tier exists to avoid.
    - any REQUIRED field carries a ``@field_validator`` (imperative
      semantics beyond its declared type), or
    - any required field's type can't be given a synthesised valid value.
    """
    if model.has_model_validator:
        return None
    kwargs: dict[str, str] = {}
    for fs in model.fields:
        if not fs.required:
            continue
        if fs.has_custom_validator:
            return None
        v = _valid_value(fs)
        if v is None:
            return None
        kwargs[fs.name] = v
    return kwargs


def _kwargs_src(kwargs: dict[str, str]) -> str:
    return ", ".join(f"{k}={v}" for k, v in kwargs.items())


_HEADER = (
    "# @tests: {source}\n"
    '"""Deterministic contract tests for {source} (ADR-0003 Tier 1).\n\n'
    "Generated from declarative Pydantic constraints — do not hand-edit;\n"
    "re-emitted from the model on each generation pass. Business-logic and\n"
    "custom-validator behaviour are covered by the LLM tier, not here.\n"
    '"""\n'
    "import pytest\n"
    "from pydantic import ValidationError\n"
    "from {module} import {names}\n\n\n"
)


def render_contract_test(
    models: list[ModelSpec], *, source_rel: str,
) -> Optional[str]:
    """Render a full deterministic contract-test file for ``models`` (all from
    one source file), or None when nothing testable could be derived."""
    testable: list[tuple[ModelSpec, dict[str, str]]] = []
    for m in models:
        kw = _valid_kwargs(m)
        if kw is None:
            continue  # conservative skip — a required field we can't synthesise
        testable.append((m, kw))
    if not testable:
        return None

    module = testable[0][0].module_import or _module_dotted(source_rel)
    names = ", ".join(sorted({m.name for m, _ in testable}))
    parts = [_HEADER.format(source=source_rel, module=module, names=names)]

    for m, valid in testable:
        base_kwargs = _kwargs_src(valid)
        # 1. Happy path — a fully-valid instance constructs.
        parts.append(
            f"def test_{_snake(m.name)}_valid_construction():\n"
            f"    obj = {m.name}({base_kwargs})\n"
            f"    assert obj is not None\n\n"
        )
        # 2. Required-field omission raises (one test per required field).
        for fs in m.fields:
            if not fs.required:
                continue
            reduced = {k: v for k, v in valid.items() if k != fs.name}
            parts.append(
                f"def test_{_snake(m.name)}_requires_{fs.name}():\n"
                f"    with pytest.raises(ValidationError):\n"
                f"        {m.name}({_kwargs_src(reduced)})\n\n"
            )
        # 3. max_length boundary — N ok, N+1 raises. Skip fields with a
        #    custom validator (imperative semantics may reject the boundary
        #    value for other reasons — that's the LLM's to assert).
        for fs in m.fields:
            if fs.has_custom_validator:
                continue
            max_len = fs.constraints.get("max_length")
            if not isinstance(max_len, (int, float)) or _base_type(fs.type_str) != "str":
                continue
            n = int(max_len)
            over = {**valid, fs.name: repr("a" * (n + 1))}
            if fs.name not in over:
                over[fs.name] = repr("a" * (n + 1))
            parts.append(
                f"def test_{_snake(m.name)}_{fs.name}_max_length():\n"
                f"    with pytest.raises(ValidationError):\n"
                f"        {m.name}({_kwargs_src(over)})\n\n"
            )
        # 4. Numeric range — below ge / above le raises.
        for fs in m.fields:
            if fs.has_custom_validator or _base_type(fs.type_str) not in ("int", "float"):
                continue
            ge = fs.constraints.get("ge")
            if isinstance(ge, (int, float)):
                under = {**valid, fs.name: repr(int(ge) - 1)}
                under.setdefault(fs.name, repr(int(ge) - 1))
                parts.append(
                    f"def test_{_snake(m.name)}_{fs.name}_below_min():\n"
                    f"    with pytest.raises(ValidationError):\n"
                    f"        {m.name}({_kwargs_src(under)})\n\n"
                )
            le = fs.constraints.get("le")
            if isinstance(le, (int, float)):
                over_n = {**valid, fs.name: repr(int(le) + 1)}
                over_n.setdefault(fs.name, repr(int(le) + 1))
                parts.append(
                    f"def test_{_snake(m.name)}_{fs.name}_above_max():\n"
                    f"    with pytest.raises(ValidationError):\n"
                    f"        {m.name}({_kwargs_src(over_n)})\n\n"
                )
    return "".join(parts)


def _snake(name: str) -> str:
    out: list[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _contract_test_rel_path(source_rel: str) -> str:
    """tests/contract/test_<module>_contract.py for a Python source file."""
    stem = os.path.splitext(os.path.basename(source_rel))[0]
    return os.path.join("tests", "contract", f"test_{stem}_contract.py")


def emit_contract_tests(
    workspace_path: str,
    source_files: list[str],
    primary_stack: str,
) -> tuple[list[str], dict[str, list[str]]]:
    """Write deterministic contract-test files for the Pydantic models in
    ``source_files``. Returns ``(rel_paths_written, tests_markers_by_file)``,
    mirroring ``test_generation._emit_nfr_stubs`` so the caller reuses the
    same marker-persistence path.

    Python-only (Tier 1). Idempotent: an existing contract-test file is NOT
    overwritten (respects operator edits). Best-effort — IO / parse errors on
    one file skip it without failing the batch.
    """
    if primary_stack != "python":
        return [], {}
    written: list[str] = []
    markers: dict[str, list[str]] = {}
    for rel in source_files:
        if not rel.endswith(".py") or _looks_like_test(rel):
            continue
        abs_src = os.path.join(workspace_path, rel)
        try:
            with open(abs_src, "r", encoding="utf-8", errors="replace") as fh:
                source = fh.read()
        except OSError:
            continue
        models = parse_pydantic_models(source, rel_path=rel)
        if not models:
            continue
        body = render_contract_test(models, source_rel=rel)
        if not body:
            continue
        out_rel = _contract_test_rel_path(rel)
        out_abs = os.path.join(workspace_path, out_rel)
        if os.path.exists(out_abs):
            # Idempotent — record the marker edge but don't clobber.
            markers[out_rel] = [rel]
            continue
        try:
            os.makedirs(os.path.dirname(out_abs), exist_ok=True)
            with open(out_abs, "w", encoding="utf-8") as fh:
                fh.write(body)
        except OSError:
            continue
        written.append(out_rel)
        markers[out_rel] = [rel]
    return written, markers


def _looks_like_test(rel_path: str) -> bool:
    base = os.path.basename(rel_path)
    parts = rel_path.replace("\\", "/").split("/")
    if any(seg in ("tests", "test", "__tests__") for seg in parts):
        return True
    return base.startswith("test_") or base.endswith("_test.py")
