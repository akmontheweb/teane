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
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "Contradiction",
    "find_contradictions",
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
    """One proven same-input / opposite-expectation contradiction."""
    call: str                 # normalized call expression, e.g. "ContactUpdate(first_name=None)"
    expect_raise_test: str    # test function asserting it raises
    expect_success_test: str  # test function asserting it succeeds
    filename: str = ""

    def describe(self) -> str:
        loc = f"{self.filename}: " if self.filename else ""
        return (
            f"{loc}`{self.call}` is required to RAISE by "
            f"`{self.expect_raise_test}` and to SUCCEED by "
            f"`{self.expect_success_test}` — identical input, opposite "
            f"expected outcomes, so no production change can satisfy both."
        )


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
        self._fn_stack: list[str] = []
        self._in_raises = 0
        self._in_try = 0

    # --- function boundary -------------------------------------------------
    def _visit_fn(self, node) -> None:
        is_test = node.name.startswith("test")
        if is_test:
            self._fn_stack.append(node.name)
            self.by_fn.setdefault(node.name, {"raise": set(), "success": set()})
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


def find_contradictions(
    source: str, *, filename: str = "",
) -> list[Contradiction]:
    """Return proven same-input / opposite-expectation contradictions in
    ``source``. Empty on parse failure (see :func:`unparseable_reason`)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    scanner = _TestFnScanner()
    for node in tree.body:
        scanner.visit(node)
    # class-scoped test methods
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                scanner.visit(child)

    # Aggregate per-signature the set of tests that expect raise vs success.
    raise_tests: dict[str, list[str]] = {}
    success_tests: dict[str, list[str]] = {}
    for fn, kinds in scanner.by_fn.items():
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
        distinct = [(r, s) for r in rt for s in st if r != s]
        if not distinct:
            continue
        r, s = distinct[0]
        out.append(Contradiction(
            call=sig, expect_raise_test=r, expect_success_test=s,
            filename=filename,
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
