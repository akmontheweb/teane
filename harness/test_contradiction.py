"""Machine-checkable detection of *unsatisfiable* generated tests.

This is the deterministic ("code rung") bottom of the autonomy ladder in
ADR-0001: before the repair loop's ``UNSATISFIABLE_TEST`` escape is allowed
to halt for a human, we first try to *prove* the test is defective without
any model judgement. Two classes are provable purely from the AST:

1. **Internal contradiction** — the same call expression (identical callable
   + identical arguments) is asserted to BOTH raise (inside ``pytest.raises``)
   and succeed (as a bare expression / assignment) across the file. Since the
   call is deterministic for identical input, no production-code change can
   satisfy both assertions simultaneously. This is the lumina
   ``test_contact_models.py`` case: ``ContactUpdate(first_name=None)`` is
   required to raise by ``test_all_none_raises`` and to succeed by
   ``test_none_fields_allowed``.

2. **Unparseable** — the test file does not parse at all, so no production
   change can make it collect.

Detection is deliberately CONSERVATIVE: it only reports a contradiction when
the *exact* same normalized call expression carries both classifications.
Differing arguments (``X(a)`` vs ``X(b)``) are never flagged, because a
production change could legitimately make one raise and the other not. A
false negative (missing a subtler contradiction) just falls through to the
next rung of the ladder; a false positive would wrongly regenerate a valid
test, so we bias hard against it.

Pure module — no harness imports, no I/O beyond an optional file read. Fully
unit-testable in isolation.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "Contradiction",
    "find_contradictions",
    "find_contradictions_across",
    "unparseable_reason",
    "machine_unsatisfiable_reason",
]

# Calls whose *names* are assertion/harness plumbing, never the "subject under
# test". Excluded when collecting subject-call signatures so we don't compare
# ``pytest.raises(...)`` or ``str(exc.value)`` against anything.
_HELPER_CALL_NAMES = frozenset({
    "raises", "warns", "approx", "deprecated_call",
    "str", "repr", "len", "isinstance", "print", "type",
    "assertRaises", "assertRaisesRegex", "assertWarns",
})

# A test that reconfigures ambient state (environment variables, patched
# attributes, working dir, mocked globals) does NOT expose its real input in
# the call signature: ``get_settings()`` reads ``os.environ`` and behaves
# differently under ``monkeypatch.setenv("PORT", "bad")`` vs
# ``monkeypatch.delenv("PORT")``. Two such tests with opposite outcomes are
# COMPLEMENTARY, not a proven same-input contradiction. Since this module only
# ever reports *proven* contradictions (see ``machine_unsatisfiable_reason``),
# it must decline to claim "identical input" whenever ambient state is in play
# and let the case fall through to normal repair/HITL.
_AMBIENT_METHOD_NAMES = frozenset({
    # pytest monkeypatch mutators
    "setenv", "delenv", "setattr", "delattr", "chdir",
    "syspath_prepend", "syspath_append",
})
_AMBIENT_FIXTURE_NAMES = frozenset({"monkeypatch", "mocker"})


def _mutates_ambient(node: ast.Call) -> bool:
    """True if ``node`` is a call that reconfigures ambient state the subject
    under test might read: a monkeypatch env/attr mutator, ``mock.patch(...)``,
    ``patch.object/dict(...)``, or the ``setattr``/``delattr`` builtins."""
    func = node.func
    if isinstance(func, ast.Attribute):
        if func.attr in _AMBIENT_METHOD_NAMES:
            return True
        if func.attr == "patch":  # mock.patch(...) / unittest.mock.patch(...)
            return True
        if func.attr in ("object", "dict", "multiple"):
            # patch.object(...) / patch.dict(...) — root callable is ``patch``.
            root = func.value
            if isinstance(root, ast.Name) and root.id == "patch":
                return True
            if isinstance(root, ast.Attribute) and root.attr == "patch":
                return True
        return False
    if isinstance(func, ast.Name):
        return func.id in ("patch", "setattr", "delattr")
    return False


def _is_environ_subscript(target: ast.expr) -> bool:
    """True for an ``os.environ[...]`` / ``environ[...]`` assignment target."""
    if not isinstance(target, ast.Subscript):
        return False
    base = target.value
    if isinstance(base, ast.Attribute) and base.attr == "environ":
        return True
    if isinstance(base, ast.Name) and base.id == "environ":
        return True
    return False


def _has_ambient_fixture(node: ast.AST) -> bool:
    """True if a test function declares a ``monkeypatch``/``mocker`` fixture
    parameter — a strong signal the test drives the subject via ambient state."""
    args = getattr(node, "args", None)
    if args is None:
        return False
    params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    return any(a.arg in _AMBIENT_FIXTURE_NAMES for a in params)


def _is_raises_call(node: ast.expr) -> bool:
    """True if ``node`` is a ``pytest.raises(...)`` / bare ``raises(...)`` /
    ``self.assertRaises(...)`` call — the markers that open a "must raise"
    context."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr in ("raises", "assertRaises", "assertRaisesRegex")
    if isinstance(func, ast.Name):
        return func.id in ("raises", "assertRaises", "assertRaisesRegex")
    return False


def _call_name(node: ast.Call) -> Optional[str]:
    """The simple callable name of a Call (``X`` for ``X(...)`` /
    ``pkg.X(...)``), or None if it isn't a plain name/attribute call."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _normalize_call(node: ast.Call) -> Optional[str]:
    """Canonical text of a *subject* call, or None if it should be ignored.

    Uses ``ast.unparse`` so ``X(first_name=None)`` normalizes identically
    regardless of source whitespace. Helper/assertion calls are dropped.
    """
    name = _call_name(node)
    if name is None or name in _HELPER_CALL_NAMES:
        return None
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 — unparse is best-effort
        return None


@dataclass(frozen=True)
class Contradiction:
    """One proven same-input / opposite-expectation contradiction.

    ``expect_raise_file`` / ``expect_success_file`` are populated by the
    cross-file detector (:func:`find_contradictions_across`) and name the
    file each side lives in; the single-file :func:`find_contradictions`
    leaves them blank (both sides share ``filename``).
    """
    call: str                 # normalized call expression, e.g. "ContactUpdate(first_name=None)"
    expect_raise_test: str    # test function asserting it raises
    expect_success_test: str  # test function asserting it succeeds
    filename: str = ""
    expect_raise_file: str = ""
    expect_success_file: str = ""

    def describe(self) -> str:
        loc = f"{self.filename}: " if self.filename else ""
        base = (
            f"{loc}`{self.call}` is required to RAISE by "
            f"`{self.expect_raise_test}` and to SUCCEED by "
            f"`{self.expect_success_test}` — identical input, opposite "
            f"expected outcomes, so no production change can satisfy both."
        )
        if (
            self.expect_raise_file
            and self.expect_success_file
            and self.expect_raise_file != self.expect_success_file
        ):
            base += (
                f" (RAISE asserted in {self.expect_raise_file}, "
                f"SUCCEED in {self.expect_success_file}.)"
            )
        return base


class _TestFnScanner(ast.NodeVisitor):
    """Collect, per test function, the subject-call signatures classified as
    expect-raise vs expect-success.

    Expect-raise: any subject call inside a ``pytest.raises`` context (the
    body of a ``with pytest.raises(...)`` block, or the functional-form first
    positional argument ``raises(Exc, call, ...)``).

    Expect-success: a subject call that is the RHS of an assignment or a bare
    expression statement AND is not inside a raises context and not inside a
    ``try`` (which could catch the exception, making success ambiguous).
    """

    def __init__(self) -> None:
        # test function name -> {"raise": set[str], "success": set[str]}
        self.by_fn: dict[str, dict[str, set[str]]] = {}
        # test function names whose body/params reconfigure ambient state —
        # their call signatures do not capture the real input, so they must
        # not participate in a "proven same-input contradiction".
        self.ambient_fns: set[str] = set()
        self._fn_stack: list[str] = []
        self._in_raises = 0
        self._in_try = 0

    # --- function boundary -------------------------------------------------
    def _visit_fn(self, node) -> None:
        is_test = node.name.startswith("test")
        if is_test:
            self._fn_stack.append(node.name)
            self.by_fn.setdefault(node.name, {"raise": set(), "success": set()})
            if _has_ambient_fixture(node):
                self.ambient_fns.add(node.name)
        for child in node.body:
            self.visit(child)
        if is_test:
            self._fn_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_fn(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_fn(node)

    # --- raises / try context ---------------------------------------------
    def visit_With(self, node: ast.With) -> None:  # noqa: N802
        opens_raises = any(_is_raises_call(item.context_expr) for item in node.items)
        if opens_raises:
            self._in_raises += 1
            for stmt in node.body:
                self.visit(stmt)
            self._in_raises -= 1
        else:
            self.generic_visit(node)

    visit_AsyncWith = visit_With  # type: ignore[assignment]

    def visit_Try(self, node: ast.Try) -> None:  # noqa: N802
        # Calls in the try body may have their exception swallowed — ambiguous
        # for "success". Mark the body; handlers/else/finally scan normally.
        self._in_try += 1
        for stmt in node.body:
            self.visit(stmt)
        self._in_try -= 1
        for stmt in (*node.handlers, *node.orelse, *node.finalbody):
            self.visit(stmt)

    # --- ambient-state detection ------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if self._fn_stack and _mutates_ambient(node):
            self.ambient_fns.add(self._fn_stack[-1])
        self.generic_visit(node)

    # --- statement classification -----------------------------------------
    def _record(self, kind: str, call: ast.Call) -> None:
        if not self._fn_stack:
            return
        sig = _normalize_call(call)
        if sig is None:
            return
        self.by_fn[self._fn_stack[-1]][kind].add(sig)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        if isinstance(node.value, ast.Call) and not self._in_try:
            if self._in_raises:
                self._record("raise", node.value)
            else:
                self._record("success", node.value)
        # Direct ``os.environ["X"] = ...`` mutation is ambient setup too.
        if self._fn_stack and any(_is_environ_subscript(t) for t in node.targets):
            self.ambient_fns.add(self._fn_stack[-1])
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if isinstance(node.value, ast.Call) and not self._in_try:
            self._record("raise" if self._in_raises else "success", node.value)
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:  # noqa: N802
        if isinstance(node.value, ast.Call):
            if self._is_raises_functional(node.value):
                # raises(Exc, call, ...) — the 2nd positional is the subject.
                if len(node.value.args) >= 2 and isinstance(node.value.args[1], ast.Call):
                    self._record("raise", node.value.args[1])
            elif not self._in_try:
                self._record("raise" if self._in_raises else "success", node.value)
        self.generic_visit(node)

    @staticmethod
    def _is_raises_functional(node: ast.Call) -> bool:
        return _is_raises_call(node) and len(node.args) >= 2


def _scan(source: str) -> Optional[_TestFnScanner]:
    """Parse ``source`` and return the populated scanner (per-test-function
    raise/success signature sets plus the ``ambient_fns`` set), or None if it
    doesn't parse. Shared by the single-file and cross-file detectors so both
    classify identically."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    scanner = _TestFnScanner()
    for node in tree.body:
        scanner.visit(node)
    # class-scoped test methods
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                scanner.visit(child)
    return scanner


def find_contradictions(
    source: str, *, filename: str = "",
) -> list[Contradiction]:
    """Return proven same-input / opposite-expectation contradictions in
    ``source``. Empty on parse failure (see :func:`unparseable_reason`)."""
    scanner = _scan(source)
    if scanner is None:
        return []
    by_fn = scanner.by_fn
    ambient = scanner.ambient_fns

    # Aggregate per-signature the set of tests that expect raise vs success.
    raise_tests: dict[str, list[str]] = {}
    success_tests: dict[str, list[str]] = {}
    for fn, kinds in by_fn.items():
        for sig in kinds["raise"]:
            raise_tests.setdefault(sig, []).append(fn)
        for sig in kinds["success"]:
            success_tests.setdefault(sig, []).append(fn)

    out: list[Contradiction] = []
    for sig in sorted(set(raise_tests) & set(success_tests)):
        rt = sorted(raise_tests[sig])
        st = sorted(success_tests[sig])
        # Require the opposite classifications to come from *different* test
        # functions — a single function that both raises and succeeds on the
        # same call is usually a copy-paste inside one assertion group, not a
        # spec contradiction. Cross-function is the strong, provable signal.
        # Also drop pairs where either side reconfigures ambient state: the
        # call signature doesn't capture env/patched-global input, so the two
        # outcomes are complementary, not a proven same-input contradiction.
        distinct = [
            (r, s) for r in rt for s in st
            if r != s and r not in ambient and s not in ambient
        ]
        if not distinct:
            continue
        r, s = distinct[0]
        out.append(Contradiction(
            call=sig, expect_raise_test=r, expect_success_test=s,
            filename=filename,
        ))
    return out


def find_contradictions_across(
    files: Mapping[str, str],
) -> list[Contradiction]:
    """Proven same-input / opposite-expectation contradictions across a
    *batch* of test files. ``files`` maps a display path to its source.

    This is the generalisation of :func:`find_contradictions` to the
    multi-file test batch a single ``test_generation_node`` call emits.
    The lumina ``019f803f`` deadlock is exactly this shape and invisible to
    the single-file detector: ``ContactUpdate(first_name='   ')`` is required
    to RAISE by a test in ``test_contact_schemas.py`` (schema-layer
    rejection) and to SUCCEED by a test in ``test_contact_service.py``
    (which constructs the object to hand to the service) — a real
    contradiction split across two files, so neither file is self-
    contradictory.

    A signature required to RAISE in one ``(file, test)`` and to SUCCEED in
    a *different* ``(file, test)`` is reported. Same conservative bias as the
    single-file detector: identical normalised call only, distinct locations
    only. Unparseable files are skipped here (that class is reported
    separately via :func:`unparseable_reason`)."""
    # sig -> list of (file, test_fn) locations
    raise_loc: dict[str, list[tuple[str, str]]] = {}
    success_loc: dict[str, list[tuple[str, str]]] = {}
    # (file, test_fn) locations that reconfigure ambient state — excluded from
    # contradiction pairing for the same reason as the single-file detector.
    ambient_loc: set[tuple[str, str]] = set()
    for fname in sorted(files):
        scanner = _scan(files[fname])
        if scanner is None:
            continue
        for fn, kinds in scanner.by_fn.items():
            for sig in kinds["raise"]:
                raise_loc.setdefault(sig, []).append((fname, fn))
            for sig in kinds["success"]:
                success_loc.setdefault(sig, []).append((fname, fn))
        for fn in scanner.ambient_fns:
            ambient_loc.add((fname, fn))

    out: list[Contradiction] = []
    for sig in sorted(set(raise_loc) & set(success_loc)):
        rl = sorted(raise_loc[sig])
        sl = sorted(success_loc[sig])
        # Distinct (file, fn) locations only — the same call both raising and
        # succeeding inside ONE test fn is an intra-assertion artefact, not a
        # spec contradiction (mirrors find_contradictions' cross-fn rule).
        # Exclude ambient-mutating locations: their input isn't in the call
        # signature, so the opposite outcomes aren't a proven contradiction.
        distinct = [
            (r, s) for r in rl for s in sl
            if r != s and r not in ambient_loc and s not in ambient_loc
        ]
        if not distinct:
            continue
        # Prefer a cross-FILE pair for the report — that's the signal this
        # detector adds over the single-file one; fall back to any distinct
        # pair (an intra-file contradiction it also legitimately covers).
        cross = [(r, s) for r, s in distinct if r[0] != s[0]]
        (rf, rfn), (sf, sfn) = (cross or distinct)[0]
        out.append(Contradiction(
            call=sig,
            expect_raise_test=rfn,
            expect_success_test=sfn,
            filename=rf if rf == sf else "",
            expect_raise_file=rf,
            expect_success_file=sf,
        ))
    return out


def unparseable_reason(source: str, *, filename: str = "") -> Optional[str]:
    """If ``source`` doesn't parse as Python, a one-line reason; else None."""
    try:
        ast.parse(source)
        return None
    except SyntaxError as exc:
        loc = f"{filename}:{exc.lineno}" if filename else f"line {exc.lineno}"
        return f"test file does not parse ({loc}: {exc.msg})"


def machine_unsatisfiable_reason(
    source: str, *, filename: str = "",
) -> Optional[str]:
    """Tier-A classifier: a human-readable reason if ``source`` is *provably*
    unsatisfiable (unparseable, or internally contradictory), else None.

    This is the deterministic rung the router consults before honouring a
    model-declared ``UNSATISFIABLE_TEST`` — a None result means "not provable
    here; fall through to the next rung / HITL", never "the test is fine".
    """
    # Python test files only — a non-.py path (jest/vitest) isn't handled by
    # this AST detector; fall through rather than guess.
    if filename and not filename.endswith(".py"):
        return None
    unparse = unparseable_reason(source, filename=filename)
    if unparse is not None:
        return unparse
    contradictions = find_contradictions(source, filename=filename)
    if contradictions:
        return contradictions[0].describe()
    return None
