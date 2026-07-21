"""Deterministic React/TS render smoke tests — ADR-0003 Tier (React).

The frontend analog of the Python schema tier: where "construct the Pydantic
model, assert no error" proved the backend contract, "render the component
with valid props, assert it doesn't crash" proves the frontend one. That
single class catches missing-required-prop crashes, undefined access on
mount, broken imports, and bad default handling.

A component's ``Props`` interface is the contract (like ``Field(...)`` was).
This module heuristically parses that interface, synthesises a minimal valid
prop set from the declared types, and emits a ``@testing-library/react``
render test.

Same two hard invariants as the Python tiers:

1. **Never execute the component at emit time.** Parsing only, via the
   ``tsx`` tree-sitter grammar (already bundled in ``tree-sitter-language-pack``
   and used by the patcher) — a real AST, not regex. jest runs the render
   later, in the sandbox.
2. **Conservative — skip what you can't prove.** A component is emitted only
   when every REQUIRED prop is a safely-synthesisable type
   (string/number/boolean, any array → ``[]``, any function → ``() => {}``)
   AND the component uses no provider-dependent hook (context/router/query),
   which would make an isolated render throw. Anything else is skipped and
   left to the LLM. A missing test is safe; a false render failure would
   block the build.

Coverage is therefore leaf/presentational components (buttons, cards, lists,
dialogs, banners). Container components that need providers or custom-object
props are the LLM's job (custom-object prop synthesis is a future extension —
resolve the referenced interface and build an object).

Pure module: regex + string rendering only. The writer
(``emit_react_contract_tests``) is the sole filesystem toucher.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "PropSpec",
    "ComponentSpec",
    "parse_react_component",
    "render_smoke_test",
    "emit_react_contract_tests",
]

# Hooks whose presence means the component reads from a provider/router/query
# context — an isolated render() without that provider throws. Skip such
# components (conservative). Matched as `useX(` to avoid false hits on names.
_PROVIDER_HOOKS = (
    "useContext", "useNavigate", "useParams", "useLocation",
    "useSearchParams", "useOutletContext", "useQuery", "useMutation",
    "useInfiniteQuery", "useSelector", "useDispatch", "useStore",
    "useFormContext", "useTheme",
)


@dataclass(frozen=True)
class PropSpec:
    name: str
    ts_type: str
    optional: bool


@dataclass(frozen=True)
class ComponentSpec:
    name: str                     # the exported component identifier
    props: tuple[PropSpec, ...]
    rel_path: str                 # source file, workspace-relative


def _tsx_parser():
    """The bundled ``tsx`` tree-sitter parser, or None if unavailable."""
    try:
        from tree_sitter_language_pack import get_parser
        return get_parser("tsx")
    except Exception:  # noqa: BLE001 — missing grammar → skip the tier
        return None


def _find_all(node, node_type: str, out: list) -> None:
    if node.type == node_type:
        out.append(node)
    for c in node.children:
        _find_all(c, node_type, out)


def _text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _props_from_interface(src: bytes, iface) -> Optional[list[PropSpec]]:
    """Extract PropSpecs from an ``interface_declaration`` node, or None if a
    member isn't a plain ``property_signature`` we can read."""
    body = iface.child_by_field_name("body")
    if body is None:
        return None
    props: list[PropSpec] = []
    for m in body.named_children:
        if m.type in ("comment",):
            continue
        if m.type != "property_signature":
            return None  # index signature / method / call signature → bail
        name_n = m.child_by_field_name("name")
        type_n = m.child_by_field_name("type")
        if name_n is None or type_n is None:
            return None
        name = _text(src, name_n)
        # type field text includes the leading ": " — strip it.
        ts_type = _text(src, type_n).lstrip(": ").strip()
        # optional marker: a '?' token sits between the name and the ':'.
        optional = "?" in _text(src, m).split(":", 1)[0]
        props.append(PropSpec(name=name, ts_type=ts_type, optional=optional))
    return props


def parse_react_component(
    source: str, *, rel_path: str = "",
) -> Optional[ComponentSpec]:
    """Return a ComponentSpec for a default-exported function component with a
    parseable props interface, or None when the file doesn't match the
    conservative shape, uses a provider hook, or can't be parsed.

    Uses the ``tsx`` tree-sitter grammar (real AST). Only the common
    ``export default function Name(props: NamedInterface)`` shape is handled;
    inline object-type props, arrow-function components, and generics are
    conservatively skipped in this tier.
    """
    # Provider-dependent hook → an isolated render throws. Skip.
    if any(re.search(r"\b" + h + r"\s*\(", source) for h in _PROVIDER_HOOKS):
        return None
    parser = _tsx_parser()
    if parser is None:
        return None
    src = source.encode("utf-8")
    tree = parser.parse(src)
    if tree.root_node.has_error:
        return None  # malformed source — don't risk a misparse

    # Find `export default function <Name>(...)`.
    exports: list = []
    _find_all(tree.root_node, "export_statement", exports)
    func = None
    for ex in exports:
        if "default" not in {c.type for c in ex.children}:
            continue
        fdecls: list = []
        _find_all(ex, "function_declaration", fdecls)
        if fdecls:
            func = fdecls[0]
            break
    if func is None:
        return None
    name_n = func.child_by_field_name("name")
    if name_n is None:
        return None
    name = _text(src, name_n)

    # First parameter's type annotation → the props interface name.
    params = func.child_by_field_name("parameters")
    if params is None:
        return None
    first = next(
        (c for c in params.named_children
         if c.type in ("required_parameter", "optional_parameter")),
        None,
    )
    if first is None:
        # Paramless component — almost always a self-managing container/page
        # (own state, data fetch on mount). A bare render is unreliable
        # (fetch-on-mount, missing route/query context). Skip — the LLM tier,
        # which can mock those, owns containers.
        return None
    type_ann = first.child_by_field_name("type")
    if type_ann is None:
        return None
    # type_ann text is ": PropsType"; resolve only a bare named interface.
    props_type = _text(src, type_ann).lstrip(": ").strip()
    if not re.fullmatch(r"\w+", props_type):
        return None  # inline / generic / union props type → skip

    ifaces: list = []
    _find_all(tree.root_node, "interface_declaration", ifaces)
    target = next(
        (i for i in ifaces
         if i.child_by_field_name("name") is not None
         and _text(src, i.child_by_field_name("name")) == props_type),
        None,
    )
    if target is None:
        return None
    props = _props_from_interface(src, target)
    if props is None:
        return None
    return ComponentSpec(name=name, props=tuple(props), rel_path=rel_path)


def _synth_prop_value(ts_type: str) -> Optional[str]:
    """A TS literal that is a valid value for ``ts_type``, or None when the
    type isn't safely synthesisable (→ skip the component)."""
    t = ts_type.strip()
    # Arrays: `T[]`, `Array<T>`, `ReadonlyArray<T>` → empty array is always valid.
    if t.endswith("[]") or re.match(r"^(Readonly)?Array\s*<", t):
        return "[]"
    # Function props → a no-op.
    if "=>" in t:
        return "() => {}"
    if t == "string":
        return "''"
    if t == "number":
        return "0"
    if t == "boolean":
        return "false"
    return None


def _minimal_props(spec: ComponentSpec) -> Optional[dict[str, str]]:
    """JSX attribute → value for every REQUIRED prop, or None if any required
    prop can't be synthesised. Optional props are omitted."""
    out: dict[str, str] = {}
    for p in spec.props:
        if p.optional:
            continue
        v = _synth_prop_value(p.ts_type)
        if v is None:
            return None
        out[p.name] = v
    return out


def _jsx_attrs(props: dict[str, str]) -> str:
    return "".join(f" {k}={{{v}}}" for k, v in props.items())


def render_smoke_test(spec: ComponentSpec) -> Optional[str]:
    """Render the @testing-library/react smoke-test source, or None when the
    component's required props aren't fully synthesisable."""
    props = _minimal_props(spec)
    if props is None:
        return None
    attrs = _jsx_attrs(props)
    return (
        f"// @tests: {spec.rel_path}\n"
        f"// Deterministic render smoke test (ADR-0003 React tier).\n"
        f"// Generated from the component's prop types — asserts it mounts\n"
        f"// without crashing on minimal valid props. Behaviour is the LLM's job.\n"
        f"import {{ render }} from '@testing-library/react';\n"
        f"import {spec.name} from '../{spec.name}';\n\n"
        f"describe('{spec.name} contract', () => {{\n"
        f"  it('renders without crashing on minimal valid props', () => {{\n"
        f"    const {{ container }} = render(<{spec.name}{attrs} />);\n"
        f"    expect(container).toBeTruthy();\n"
        f"  }});\n"
        f"}});\n"
    )


_TSX_EXTS = (".tsx", ".jsx")


def _react_test_rel_path(source_rel: str) -> str:
    """client/src/components/AlertBanner.tsx →
    client/src/components/__tests__/AlertBanner.contract.test.tsx."""
    d = os.path.dirname(source_rel)
    stem = os.path.splitext(os.path.basename(source_rel))[0]
    return os.path.join(d, "__tests__", f"{stem}.contract.test.tsx")


def _read_text(abs_path: str) -> Optional[str]:
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def _looks_like_test(rel_path: str) -> bool:
    parts = rel_path.replace("\\", "/").split("/")
    if any(seg in ("__tests__", "tests", "test") for seg in parts):
        return True
    base = os.path.basename(rel_path)
    return ".test." in base or ".spec." in base


def emit_react_contract_tests(
    workspace_path: str,
    source_files: list[str],
    primary_stack: str,
) -> tuple[list[str], dict[str, list[str]]]:
    """Write render smoke tests for React components in ``source_files``.
    Returns ``(rel_paths_written, tests_markers_by_file)``.

    Fires whenever there is a ``.tsx``/``.jsx`` component present (independent
    of the workspace's primary stack). Idempotent; best-effort. The generated
    test needs ``@testing-library/react`` in the jest env — ensuring that dep
    is the node-side integration's job (see emit wiring), not this pure
    emitter's.
    """
    del primary_stack  # gate is component-presence, not primary stack
    written: list[str] = []
    markers: dict[str, list[str]] = {}
    for rel in source_files:
        if os.path.splitext(rel)[1].lower() not in _TSX_EXTS:
            continue
        if _looks_like_test(rel):
            continue
        src = _read_text(os.path.join(workspace_path, rel))
        if src is None:
            continue
        spec = parse_react_component(src, rel_path=rel)
        if spec is None:
            continue
        body = render_smoke_test(spec)
        if not body:
            continue
        out_rel = _react_test_rel_path(rel)
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
