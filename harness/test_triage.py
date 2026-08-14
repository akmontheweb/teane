"""ADR-0005: pre-repair test-failure triage.

Classify a compiler/pytest diagnostic as a **test-bug** (a defect in the
generated TEST's own authoring — cheaply regenerate the test before the repair
loop ever sees it) or a **code-gap** (a genuine code-vs-spec mismatch that the
repair loop should fix).

**Conservative by construction.** Only a small set of high-confidence,
test-authoring fingerprints yield ``TEST_BUG``. Every ambiguous or
behaviour-level failure — ``AssertionError``, ``Failed: DID NOT RAISE``, any
failure whose innermost frame is in source, anything unrecognised — defaults to
``CODE_GAP``. Triage therefore can never *mask* a real code defect: the worst
case is exactly today's behaviour (the failure flows to repair).

The classifier keys off the parsed ``DiagnosticObject`` shape (``harness.sandbox``
/ ``DiagnosticObjectDict`` in ``harness.graph``): ``file`` is the innermost
user-code frame, ``error_code`` is the exception type, ``message`` is its text.
Grounded in the real diagnostics from lumina run 019ff418 — e.g. a
``NameError: name 'patch' is not defined`` or a ``TypeError: tuple indices must
be integers or slices, not str`` located *in a test file* is a test-authoring
bug no production change can fix, whereas the same run's ``async ... context
manager protocol`` in ``database.py`` is a code-gap repair must own.

See ``docs/adr/ADR-0005-shift-left-test-quality-to-minimize-repair.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional

__all__ = [
    "FailureClass",
    "TriageResult",
    "classify_diagnostic",
    "summarize_failures",
]


class FailureClass(str, Enum):
    TEST_BUG = "test_bug"
    CODE_GAP = "code_gap"


@dataclass(frozen=True)
class TriageResult:
    """Outcome of classifying one diagnostic.

    ``fingerprint`` is a stable kebab-case label of the matched test-defect
    pattern (empty for ``CODE_GAP``). ``confidence`` is only meaningful for
    ``TEST_BUG`` ("high" | "medium"); ``CODE_GAP`` always carries "n/a".
    """
    fclass: FailureClass
    fingerprint: str
    confidence: str
    reason: str

    @property
    def is_test_bug(self) -> bool:
        return self.fclass is FailureClass.TEST_BUG


def _code_gap(reason: str) -> TriageResult:
    return TriageResult(FailureClass.CODE_GAP, "", "n/a", reason)


def _field(diag: Any, name: str) -> Any:
    """Read ``name`` from a diagnostic that may be a ``DiagnosticObject`` or a
    ``DiagnosticObjectDict``."""
    if isinstance(diag, dict):
        return diag.get(name)
    return getattr(diag, name, None)


def _is_test_artifact(path: str) -> bool:
    """Canonical test-file predicate — the SAME union the repair-loop tamper
    guard uses, so "the failing frame is in a file repair may not edit" and
    "this is a candidate test-bug" are the one decision. Imported lazily to
    avoid a circular import with ``harness.graph``."""
    if not path:
        return False
    from harness.graph import _is_test_artifact as _graph_is_test_artifact
    return _graph_is_test_artifact(path)


# --- test-authoring defect fingerprints (message/error_code based) ----------
# Each matches ONLY when the innermost frame is a test file; a source-file
# frame with the same exception is a code-gap.

_UNDEFINED_NAME_RE = re.compile(r"name '([^']+)' is not defined")
_TUPLE_SUBSCRIPT_RE = re.compile(
    r"tuple indices must be integers or slices, not str"
)
# unittest.mock raises this exact phrasing when ``patch("mod.attr")`` names an
# attribute the module does not define — distinctive to a bad patch target.
_PATCH_TARGET_MISS_RE = re.compile(r"does not have the attribute")
# ``ImportError: cannot import name 'X' from 'Y'`` — Y was found but does not
# export X (a renamed/nonexistent symbol the test imported). Distinct from
# ``No module named`` (a missing module/dependency), which stays a code-gap.
_BAD_SYMBOL_IMPORT_RE = re.compile(r"cannot import name ['\"]([^'\"]+)['\"]")


def _classify_test_frame(
    error_code: str, message: str
) -> Optional[TriageResult]:
    """High-confidence test-authoring fingerprints for a failure whose
    innermost frame is already known to be a test file. Returns None when no
    fingerprint matches (caller then defaults to CODE_GAP)."""
    code = (error_code or "").strip()
    msg = message or ""

    # NameError: name 'patch' is not defined — the test references a symbol it
    # never imported (lumina 019ff418: test_contact_service.py:83 used
    # ``patch`` with no ``from unittest.mock import patch``). Pure test bug.
    if code == "NameError":
        m = _UNDEFINED_NAME_RE.search(msg)
        if m:
            return TriageResult(
                FailureClass.TEST_BUG, "test-undefined-name", "high",
                f"test references undefined name {m.group(1)!r} "
                f"— missing import in the test itself",
            )

    # TypeError: tuple indices must be integers or slices, not str — the test
    # subscripts a DB row by name (``row['x']``) off a connection whose
    # ``row_factory`` it never set (lumina 019ff418: test_database.py:41). No
    # production change can satisfy it; the test's own setup is wrong.
    if code == "TypeError" and _TUPLE_SUBSCRIPT_RE.search(msg):
        return TriageResult(
            FailureClass.TEST_BUG, "test-raw-row-subscript", "high",
            "test subscripts a tuple row by name — its own connection never "
            "set row_factory (raw sqlite3/aiosqlite rows are tuples)",
        )

    # AttributeError: <module 'x'> does not have the attribute 'Y' — a
    # ``patch("x.Y")`` aimed at a target the module never defines. The phrasing
    # is unittest.mock-specific, so within a test frame it is a test-side
    # mistake (wrong/renamed patch target), not a production gap.
    if code == "AttributeError" and _PATCH_TARGET_MISS_RE.search(msg):
        return TriageResult(
            FailureClass.TEST_BUG, "test-unresolved-patch-target", "high",
            "test patches a target the module does not define "
            "(unittest.mock 'does not have the attribute')",
        )

    # ImportError: cannot import name 'X' from 'Y' — the module Y resolved but
    # does not export X: the test imports a renamed/nonexistent symbol. Only the
    # ``cannot import name`` shape qualifies; a bare ``No module named`` is a
    # missing module/dependency and stays a code-gap (repair/env owns it).
    if code in ("ImportError", "ModuleNotFoundError"):
        m = _BAD_SYMBOL_IMPORT_RE.search(msg)
        if m:
            return TriageResult(
                FailureClass.TEST_BUG, "test-bad-symbol-import", "high",
                f"test imports name {m.group(1)!r} that its target module "
                f"does not export",
            )

    return None


def classify_diagnostic(
    diag: Any,
    *,
    is_test_file: Optional[bool] = None,
) -> TriageResult:
    """Classify one diagnostic as ``TEST_BUG`` or ``CODE_GAP``.

    ``diag`` is a ``DiagnosticObject`` or ``DiagnosticObjectDict``. Pass
    ``is_test_file`` to override the file-based check (e.g. in tests, or when
    the caller already resolved it); otherwise it is derived from ``diag.file``
    via the canonical test-artifact predicate.
    """
    file = _field(diag, "file") or ""
    error_code = _field(diag, "error_code") or ""
    message = _field(diag, "message") or ""

    # Diagnostics routed from test_generation_node carry a ``TEST_FAILURE:``
    # prefix on error_code (the graph tags them so repair's framing knows they
    # came from the test runner). Strip it before fingerprint matching so the
    # exact-match fingerprints fire on both the raw and the tagged form.
    if error_code.startswith("TEST_FAILURE:"):
        error_code = error_code[len("TEST_FAILURE:"):]

    in_test = _is_test_artifact(file) if is_test_file is None else is_test_file

    # A failure whose innermost user frame is in SOURCE is, by definition, a
    # code-level fault repair should own — even if a test triggered it. Only
    # test-frame failures can be test-authoring bugs.
    if not in_test:
        return _code_gap(
            f"innermost frame in source ({file or 'unknown'}) — code-gap"
        )

    hit = _classify_test_frame(error_code, message)
    if hit is not None:
        return hit

    # In a test frame but no test-authoring fingerprint matched. This is the
    # deliberately conservative default: behaviour assertions
    # (``AssertionError``, ``Failed: DID NOT RAISE``) are frequently DOWNSTREAM
    # symptoms of a real code bug (lumina 019ff418: an async-connection fault
    # in database.py cascaded into ``assert 500 == 404`` and ``DID NOT RAISE``
    # across many service tests). Routing them to repair is correct; only an
    # unambiguous test-authoring fingerprint may divert to regeneration.
    return _code_gap(
        f"test-frame {error_code or 'failure'} with no test-authoring "
        f"fingerprint — default to repair"
    )


def summarize_failures(
    diags: Iterable[Any],
) -> dict[str, Any]:
    """Classify a batch of diagnostics and return a telemetry-ready summary.

    Shape: ``{"test_bug": int, "code_gap": int, "fingerprints": {label: int},
    "results": [TriageResult, ...]}``. Used both by the repair-round telemetry
    (ADR-0005 item #1) and the pre-repair triage gate (item #2)."""
    results = [classify_diagnostic(d) for d in diags]
    test_bug = sum(1 for r in results if r.is_test_bug)
    fingerprints: dict[str, int] = {}
    for r in results:
        if r.is_test_bug:
            fingerprints[r.fingerprint] = fingerprints.get(r.fingerprint, 0) + 1
    return {
        "test_bug": test_bug,
        "code_gap": len(results) - test_bug,
        "fingerprints": fingerprints,
        "results": results,
    }
