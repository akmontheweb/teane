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
import re
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "FieldSpec",
    "ModelSpec",
    "parse_pydantic_models",
    "render_contract_test",
    "emit_contract_tests",
    # Tier 2 — API status-code contracts
    "RouteSpec",
    "find_fastapi_app",
    "parse_fastapi_routes",
    "render_api_contract_test",
    "emit_api_contract_tests",
    # Tier 3 — property-based structural invariants
    "render_property_test",
    "emit_property_tests",
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
    has_alias: bool = False              # Field(alias=...) — breaks field-name round-trip


@dataclass(frozen=True)
class ModelSpec:
    name: str
    fields: tuple[FieldSpec, ...]
    module_import: str                   # dotted path guess for the import line
    has_model_validator: bool = False    # a @model_validator can reject any instance


def _read_text(abs_path: str) -> Optional[str]:
    """Read a file as UTF-8 (errors replaced), or None if unreadable."""
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


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


def _extract_field_call_constraints(
    value: ast.expr,
) -> tuple[dict[str, object], bool, bool]:
    """Return ``(constraints, has_default, has_alias)`` from a field's RHS.

    Recognises ``Field(default, max_length=.., ge=.., alias=..)``.
    ``has_default`` is True when a default is expressed (first positional arg
    that isn't ``...``, a ``default=``/``default_factory=`` kw, or a bare
    literal RHS). ``has_alias`` is True for any ``alias``/``validation_alias``/
    ``serialization_alias`` kw — those break the field-name round-trip Tier 3
    relies on, so a model with one is skipped there.
    """
    constraints: dict[str, object] = {}
    has_default = False
    has_alias = False
    if isinstance(value, ast.Call) and _call_name(value) == "Field":
        # positional default: Field(default, ...) — Ellipsis means "required"
        if value.args:
            first = value.args[0]
            if not (isinstance(first, ast.Constant) and first.value is Ellipsis):
                has_default = True
        for kw in value.keywords:
            if kw.arg in ("default", "default_factory"):
                has_default = True
            elif kw.arg in ("alias", "validation_alias", "serialization_alias"):
                has_alias = True
            elif kw.arg in _LEN_CONSTRAINTS + _NUM_CONSTRAINTS:
                lit = _literal(kw.value)
                if lit is not None:
                    constraints[kw.arg] = lit
    elif value is not None and not (
        isinstance(value, ast.Constant) and value.value is Ellipsis
    ):
        # Bare default: ``x: int = 5`` — a real default value present.
        has_default = True
    return constraints, has_default, has_alias


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
            if item.value is not None:
                constraints, has_default, has_alias = _extract_field_call_constraints(item.value)
            else:
                constraints, has_default, has_alias = {}, False, False
            required = not _is_optional(type_str) and not has_default
            fields.append(FieldSpec(
                name=fname,
                type_str=type_str,
                required=required,
                constraints=constraints,
                has_custom_validator=fname in validator_fields,
                has_alias=has_alias,
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


# ===========================================================================
# Tier 2 — API status-code contracts (ADR-0003)
#
# Deterministic tests for FastAPI's *framework-guaranteed* error contracts:
# a request that fails validation returns 422 BEFORE the handler runs, so no
# database or business state is needed — the exact robustness sweet spot.
#
#   - POST/PUT/PATCH with a Pydantic body that has >=1 required field:
#       an empty JSON body {} → 422.
#   - a route with an int/float path param: a non-numeric value there → 422.
#
# Still AST-only at emit time (never imports the app); the *generated test*
# uses fastapi.testclient.TestClient at run time in the sandbox. Success-path
# status codes (201/200) are deliberately NOT asserted — they need valid data
# and live state, which is business-logic-coupled (Tier 4's job).
# ===========================================================================

_HTTP_METHODS = ("get", "post", "put", "delete", "patch")
_NUMERIC_PATH_TYPES = frozenset({"int", "float"})
# Param defaults that mark a function arg as NOT a request body.
_NON_BODY_DEPENDENCY_CALLS = frozenset({
    "Depends", "Query", "Path", "Header", "Cookie", "Form", "Security",
})


@dataclass(frozen=True)
class RouteSpec:
    method: str                          # 'post', 'put', ...
    path: str                            # full path incl. router prefix
    body_model: Optional[str]            # a known-model body param, or None
    body_required: bool                  # body model has >=1 declared-required field
    int_path_params: tuple[str, ...]     # path params annotated int/float
    func_name: str


def find_fastapi_app(source: str, *, rel_path: str = "") -> Optional[tuple[str, str]]:
    """Return ``(module_dotted, app_var)`` for the FastAPI app instance in
    ``source``, or None. Recognises two module-level patterns:

    - ``app = FastAPI(...)`` — direct instantiation.
    - ``app = create_app()`` where a ``def create_app(...) -> FastAPI:`` with
      that return annotation exists in the same module (lumina's factory).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    module = _module_dotted(rel_path) if rel_path else ""
    # functions annotated `-> FastAPI`
    fastapi_factories = {
        n.name for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _ann_to_str(n.returns) == "FastAPI"
    }
    for node in tree.body:  # module-level only
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        val = node.value
        if isinstance(val, ast.Call):
            fn = _call_name(val)
            if fn == "FastAPI" or fn in fastapi_factories:
                return module, target.id
    return None


def _router_prefixes(tree: ast.Module) -> dict[str, str]:
    """Map ``APIRouter`` / ``FastAPI`` variable name → path prefix.

    ``router = APIRouter(prefix="/api/contacts")`` → {"router": "/api/contacts"}.
    A bare ``FastAPI()`` / ``APIRouter()`` maps to "" (no prefix).
    """
    prefixes: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(node.value, ast.Call):
            continue
        if _call_name(node.value) not in ("APIRouter", "FastAPI"):
            continue
        prefix = ""
        for kw in node.value.keywords:
            if kw.arg == "prefix" and isinstance(kw.value, ast.Constant):
                if isinstance(kw.value.value, str):
                    prefix = kw.value.value
        prefixes[target.id] = prefix
    return prefixes


def _route_from_decorator(
    dec: ast.expr,
) -> Optional[tuple[str, str, str]]:
    """``@router.post("/x", status_code=201)`` → ("router", "post", "/x").
    None if the decorator isn't an HTTP-method route decorator."""
    if not isinstance(dec, ast.Call) or not isinstance(dec.func, ast.Attribute):
        return None
    method = dec.func.attr
    if method not in _HTTP_METHODS:
        return None
    var = dec.func.value
    if not isinstance(var, ast.Name):
        return None
    path = ""
    if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
        path = dec.args[0].value
    return var.id, method, path


def _join_path(prefix: str, route: str) -> str:
    full = (prefix or "") + (route or "")
    while "//" in full:
        full = full.replace("//", "/")
    return full or "/"


def _is_body_param(arg: ast.arg, default: Optional[ast.expr], known_models: set[str]) -> bool:
    """A function arg is a request body when its annotation is a known
    Pydantic model AND its default isn't a FastAPI dependency marker."""
    ann = _ann_to_str(arg.annotation)
    base = _base_type(ann)
    if base not in known_models:
        return False
    if isinstance(default, ast.Call) and _call_name(default) in _NON_BODY_DEPENDENCY_CALLS:
        return False
    return True


def parse_fastapi_routes(
    source: str,
    *,
    rel_path: str = "",
    model_required: Optional[dict[str, bool]] = None,
) -> list[RouteSpec]:
    """Extract testable ``RouteSpec``s from a FastAPI route module.

    ``model_required`` maps a Pydantic model name → whether it has >=1
    declared-required field (from :func:`parse_pydantic_models` over the
    workspace); used to decide the empty-body 422 test and to recognise body
    params. Empty when no models are known → body detection is skipped.
    """
    model_required = model_required or {}
    known_models = set(model_required)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    prefixes = _router_prefixes(tree)
    out: list[RouteSpec] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            parsed = _route_from_decorator(dec)
            if parsed is None:
                continue
            var, method, route_path = parsed
            prefix = prefixes.get(var, "")
            full_path = _join_path(prefix, route_path)
            # Body param + path params from the signature.
            body_model: Optional[str] = None
            path_param_types: dict[str, str] = {}
            args = node.args
            defaults = list(args.defaults)
            # align defaults to the tail of positional args
            pos = args.posonlyargs + args.args
            pad = [None] * (len(pos) - len(defaults))
            paired = list(zip(pos, pad + defaults))
            for a, d in paired:
                base = _base_type(_ann_to_str(a.annotation))
                if body_model is None and _is_body_param(a, d, known_models):
                    body_model = base
                if base in _NUMERIC_PATH_TYPES:
                    path_param_types[a.arg] = base
            # int path params that actually appear in the path template
            path_names = set(re.findall(r"\{(\w+)\}", full_path))
            int_params = tuple(
                p for p in sorted(path_param_types) if p in path_names
            )
            out.append(RouteSpec(
                method=method,
                path=full_path,
                body_model=body_model,
                body_required=bool(body_model and model_required.get(body_model, False)),
                int_path_params=int_params,
                func_name=node.name,
            ))
    return out


def _concrete_path(path: str, *, bad_param: Optional[str] = None) -> str:
    """Fill a path template with placeholder values. ``bad_param`` gets a
    non-numeric sentinel (to trigger a 422); every other ``{param}`` gets 1."""
    def _sub(m: "re.Match[str]") -> str:
        name = m.group(1)
        return "not-a-number" if name == bad_param else "1"
    return re.sub(r"\{(\w+)\}", _sub, path)


_API_HEADER = (
    "# @tests: {source}\n"
    '"""Deterministic API contract tests for {source} (ADR-0003 Tier 2).\n\n'
    "FastAPI framework guarantees only — a request that fails validation\n"
    "returns 422 before the handler runs, so no DB/business state is needed.\n"
    "Success-path codes and business behaviour are the LLM tier's job.\n"
    '"""\n'
    "from fastapi.testclient import TestClient\n"
    "from {app_module} import {app_var}\n\n"
    "client = TestClient({app_var})\n\n\n"
)


def render_api_contract_test(
    routes: list[RouteSpec],
    *,
    app_module: str,
    app_var: str,
    source_rel: str,
) -> Optional[str]:
    """Render the API-contract test file, or None if no route yields a
    deterministic assertion."""
    body: list[str] = []
    for r in routes:
        # 1. Empty-body → 422 (only when the body model has a required field;
        #    an all-optional body accepts {} and would NOT 422).
        if r.method in ("post", "put", "patch") and r.body_required:
            url = _concrete_path(r.path)
            body.append(
                f"def test_{r.method}_{_path_slug(r.path)}_empty_body_422():\n"
                f'    resp = client.{r.method}("{url}", json={{}})\n'
                f"    assert resp.status_code == 422\n\n"
            )
        # 2. Non-numeric value in an int path param → 422.
        slug = _path_slug(r.path)
        for p in r.int_path_params:
            url = _concrete_path(r.path, bad_param=p)
            call = (
                f'client.{r.method}("{url}", json={{}})'
                if r.method in ("post", "put", "patch")
                else f'client.{r.method}("{url}")'
            )
            # Avoid a name stutter when the slug already ends with the param.
            suffix = "bad_type" if slug.endswith(p) else f"{p}_bad_type"
            body.append(
                f"def test_{r.method}_{slug}_{suffix}_422():\n"
                f"    resp = {call}\n"
                f"    assert resp.status_code == 422\n\n"
            )
    if not body:
        return None
    return _API_HEADER.format(
        source=source_rel, app_module=app_module, app_var=app_var,
    ) + "".join(body)


def _path_slug(path: str) -> str:
    """/api/contacts/{contact_id} → api_contacts_contact_id."""
    s = re.sub(r"[{}]", "", path)
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_")
    return s.lower() or "root"


def _api_contract_test_rel_path(source_rel: str) -> str:
    stem = os.path.splitext(os.path.basename(source_rel))[0]
    return os.path.join("tests", "contract", f"test_{stem}_api_contract.py")


def emit_api_contract_tests(
    workspace_path: str,
    source_files: list[str],
    primary_stack: str,
) -> tuple[list[str], dict[str, list[str]]]:
    """Write deterministic API-contract test files for FastAPI routes in
    ``source_files``. Returns ``(rel_paths_written, tests_markers_by_file)``,
    mirroring :func:`emit_contract_tests`.

    Python-only. Needs a discoverable FastAPI app instance among the source
    files (for the TestClient import); returns ``([], {})`` if none is found.
    Idempotent; best-effort per file.
    """
    if primary_stack != "python":
        return [], {}
    py_files = [
        r for r in source_files if r.endswith(".py") and not _looks_like_test(r)
    ]
    # Locate the app instance (usually main.py) across all source files.
    app_ref: Optional[tuple[str, str]] = None
    model_required: dict[str, bool] = {}
    file_source: dict[str, str] = {}
    for rel in py_files:
        src = _read_text(os.path.join(workspace_path, rel))
        if src is None:
            continue
        file_source[rel] = src
        if app_ref is None:
            app_ref = find_fastapi_app(src, rel_path=rel)
        for m in parse_pydantic_models(src, rel_path=rel):
            model_required[m.name] = any(f.required for f in m.fields)
    if app_ref is None:
        return [], {}
    app_module, app_var = app_ref

    written: list[str] = []
    markers: dict[str, list[str]] = {}
    for rel, src in file_source.items():
        routes = parse_fastapi_routes(
            src, rel_path=rel, model_required=model_required,
        )
        if not routes:
            continue
        body = render_api_contract_test(
            routes, app_module=app_module, app_var=app_var, source_rel=rel,
        )
        if not body:
            continue
        out_rel = _api_contract_test_rel_path(rel)
        out_abs = os.path.join(workspace_path, out_rel)
        if os.path.exists(out_abs):
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


# ===========================================================================
# Tier 3 — property-based structural invariants (ADR-0003)
#
# Derives Hypothesis strategies from field types + declarative constraints and
# asserts a STRUCTURAL invariant that holds regardless of business logic: a
# validated instance survives a model_dump() → reconstruct round-trip
# unchanged. Deliberately NOT value-correctness (Tier 4's job).
#
# The known-fiddly tier — false positives (a legitimate model quirk failing
# the property) would block a build, so this ships config-gated, default OFF
# (test_generation.contract_tests.property_based), and is aggressively
# conservative: a model is skipped unless EVERY field is a plain scalar
# (str/int/float/bool or Optional thereof) with a mappable strategy, no
# custom/model validator, and no alias. The generated file does
# ``pytest.importorskip("hypothesis")`` so it skips (not errors) where the
# dependency is absent.
# ===========================================================================

def _hypothesis_strategy(fs: "FieldSpec") -> Optional[str]:
    """A Hypothesis strategy expression for a field, or None when its type
    isn't safely mappable (→ skip the whole model)."""
    base = _base_type(fs.type_str)
    c = fs.constraints
    inner: Optional[str] = None
    if base == "str":
        kw: list[str] = []
        if isinstance(c.get("min_length"), (int, float)):
            kw.append(f"min_size={int(c['min_length'])}")
        max_l = c.get("max_length")
        kw.append(f"max_size={int(max_l)}" if isinstance(max_l, (int, float)) else "max_size=200")
        inner = f"st.text({', '.join(kw)})"
    elif base == "int":
        kw = []
        if isinstance(c.get("ge"), (int, float)):
            kw.append(f"min_value={int(c['ge'])}")
        elif isinstance(c.get("gt"), (int, float)):
            kw.append(f"min_value={int(c['gt']) + 1}")
        if isinstance(c.get("le"), (int, float)):
            kw.append(f"max_value={int(c['le'])}")
        elif isinstance(c.get("lt"), (int, float)):
            kw.append(f"max_value={int(c['lt']) - 1}")
        inner = f"st.integers({', '.join(kw)})"
    elif base == "float":
        kw = ["allow_nan=False", "allow_infinity=False"]
        if isinstance(c.get("ge"), (int, float)):
            kw.append(f"min_value={float(c['ge'])}")
        if isinstance(c.get("le"), (int, float)):
            kw.append(f"max_value={float(c['le'])}")
        inner = f"st.floats({', '.join(kw)})"
    elif base == "bool":
        inner = "st.booleans()"
    else:
        return None
    if _is_optional(fs.type_str):
        inner = f"st.none() | {inner}"
    return inner


def _property_testable(model: "ModelSpec") -> Optional[dict[str, str]]:
    """Field-name → strategy expression for a model whose round-trip is a safe
    structural invariant, or None when the model must be skipped."""
    if model.has_model_validator:
        return None
    strategies: dict[str, str] = {}
    for fs in model.fields:
        if fs.has_custom_validator or fs.has_alias:
            return None
        strat = _hypothesis_strategy(fs)
        if strat is None:
            return None
        strategies[fs.name] = strat
    return strategies or None


_PROP_HEADER = (
    "# @tests: {source}\n"
    '"""Property-based contract tests for {source} (ADR-0003 Tier 3).\n\n'
    "Structural invariant only — a valid instance survives a model_dump() →\n"
    "reconstruct round-trip unchanged, for any generated input. Value-\n"
    "correctness and business rules are the LLM tier's job.\n"
    '"""\n'
    "import pytest\n"
    'pytest.importorskip("hypothesis")\n'
    "from hypothesis import given, strategies as st\n"
    "from {module} import {names}\n\n\n"
)


def render_property_test(
    models: list["ModelSpec"], *, source_rel: str,
) -> Optional[str]:
    """Render the property-test file for ``models``, or None if none qualify."""
    testable: list[tuple[ModelSpec, dict[str, str]]] = []
    for m in models:
        strategies = _property_testable(m)
        if strategies is None:
            continue
        testable.append((m, strategies))
    if not testable:
        return None
    module = testable[0][0].module_import or _module_dotted(source_rel)
    names = ", ".join(sorted({m.name for m, _ in testable}))
    parts = [_PROP_HEADER.format(source=source_rel, module=module, names=names)]
    for m, strategies in testable:
        given_kwargs = ", ".join(f"{k}={v}" for k, v in strategies.items())
        params = ", ".join(strategies)
        ctor_kwargs = ", ".join(f"{k}={k}" for k in strategies)
        parts.append(
            f"@given({given_kwargs})\n"
            f"def test_{_snake(m.name)}_roundtrip({params}):\n"
            f"    obj = {m.name}({ctor_kwargs})\n"
            f"    assert {m.name}(**obj.model_dump()) == obj\n\n"
        )
    return "".join(parts)


def _property_test_rel_path(source_rel: str) -> str:
    stem = os.path.splitext(os.path.basename(source_rel))[0]
    return os.path.join("tests", "contract", f"test_{stem}_property.py")


def emit_property_tests(
    workspace_path: str,
    source_files: list[str],
    primary_stack: str,
) -> tuple[list[str], dict[str, list[str]]]:
    """Write property-based round-trip test files for the Pydantic models in
    ``source_files``. Returns ``(rel_paths_written, tests_markers_by_file)``.

    Python-only; idempotent; best-effort. Gated by the caller (default off);
    this function itself just emits when asked. Conservative model selection
    lives in :func:`_property_testable`.
    """
    if primary_stack != "python":
        return [], {}
    written: list[str] = []
    markers: dict[str, list[str]] = {}
    for rel in source_files:
        if not rel.endswith(".py") or _looks_like_test(rel):
            continue
        src = _read_text(os.path.join(workspace_path, rel))
        if src is None:
            continue
        models = parse_pydantic_models(src, rel_path=rel)
        if not models:
            continue
        body = render_property_test(models, source_rel=rel)
        if not body:
            continue
        out_rel = _property_test_rel_path(rel)
        out_abs = os.path.join(workspace_path, out_rel)
        if os.path.exists(out_abs):
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
