"""Collection-error carve-out for the repair-loop test-tamper guard.

The tamper guard refuses repair edits to test files (reward-hacking defense).
But when a pytest *collection* failure's only fix is in test infrastructure —
a broken ``conftest.py``, a missing/duplicate ``__init__.py``, a pytest config
— refusing those edits leaves the repair loop with zero legal moves. lumina
019f82af deadlocked exactly this way: two ``tests`` packages collided
(ImportPathMismatchError), the only fix was editing conftest/``__init__.py``,
the guard refused it every round, and the run ping-ponged through every HITL
auto-resume while the production code was already green.

``_syntax_broken_test_files`` now opens a carve-out for collection/import
errors — scoped to test-INFRASTRUCTURE files (conftest, test-tree
``__init__.py``, pytest config), never test-CASE files, so assertions still
can't be weakened.
"""

from __future__ import annotations

import os

import pytest

from harness.graph import (
    _has_collection_error,
    _is_test_infra_file,
    _reject_test_patch_blocks,
    _syntax_broken_test_files,
)
from harness.patcher import OperationType


class _Block:
    def __init__(self, file: str, operation=OperationType.REPLACE_BLOCK) -> None:
        self.file = file
        self.operation = operation


def _mk_workspace(tmp_path) -> str:
    ws = str(tmp_path)
    for rel in (
        "tests/conftest.py",
        "tests/__init__.py",
        "tests/test_contacts.py",
        "server/tests/conftest.py",
        "server/tests/__init__.py",
        "server/app/__init__.py",  # source pkg marker — must NOT be opened
        "pytest.ini",
    ):
        p = os.path.join(ws, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write("# scaffold\n")
    return ws


def test_has_collection_error_matches_code_and_message():
    assert _has_collection_error([{"error_code": "ImportPathMismatchError"}])
    assert _has_collection_error([{"message": "ERROR collecting tests/foo.py"}])
    assert _has_collection_error([{"message": "import file mismatch"}])
    assert not _has_collection_error([{"error_code": "AssertionError"}])
    assert not _has_collection_error([])


def test_is_test_infra_file_scope():
    assert _is_test_infra_file("tests/conftest.py")
    assert _is_test_infra_file("server/tests/__init__.py")
    assert _is_test_infra_file("pytest.ini")
    # test-CASE files and source-package __init__.py are NOT infra
    assert not _is_test_infra_file("tests/test_contacts.py")
    assert not _is_test_infra_file("server/app/__init__.py")
    # a nested (subpackage) pyproject is not the root config
    assert not _is_test_infra_file("server/pyproject.toml")


def test_collection_error_opens_infra_files_only(tmp_path):
    ws = _mk_workspace(tmp_path)
    errors = [{"error_code": "ImportPathMismatchError", "file": "conftest.py"}]
    allowed = _syntax_broken_test_files(errors, ws)
    assert "tests/conftest.py" in allowed
    assert "tests/__init__.py" in allowed
    assert os.path.join("server", "tests", "conftest.py") in allowed
    # the nested second `tests` package marker — the actual lumina collider
    assert os.path.join("server", "tests", "__init__.py") in allowed
    assert "pytest.ini" in allowed
    # test-case file stays protected even during a collection error
    assert "tests/test_contacts.py" not in allowed
    # source-package __init__.py is never opened
    assert os.path.join("server", "app", "__init__.py") not in allowed


def test_no_carveout_without_collection_error(tmp_path):
    ws = _mk_workspace(tmp_path)
    # A plain assertion failure opens nothing (no parse error, no collection).
    allowed = _syntax_broken_test_files(
        [{"error_code": "AssertionError", "file": "tests/test_contacts.py"}], ws,
    )
    assert allowed == frozenset()


def test_reject_permits_conftest_edit_under_collection_error(tmp_path):
    ws = _mk_workspace(tmp_path)
    errors = [{"error_code": "ImportPathMismatchError", "file": "conftest.py"}]
    allowed = _syntax_broken_test_files(errors, ws)
    blocks = [
        _Block("server/app/routes.py"),      # production — always kept
        _Block("tests/conftest.py"),          # infra — kept under carve-out
        _Block("tests/test_contacts.py"),     # test case — still refused
    ]
    kept, rejections = _reject_test_patch_blocks(
        blocks, allow_parse_broken=allowed, workspace_path=ws,
    )
    kept_files = {b.file for b in kept}
    assert "server/app/routes.py" in kept_files
    assert "tests/conftest.py" in kept_files
    assert "tests/test_contacts.py" not in kept_files
    assert [r.file for r in rejections] == ["tests/test_contacts.py"]


# ---------------------------------------------------------------------------
# Carve-out 3 — unimportable test modules (lumina-verify-20260914-1151).
#
# The generated `server/tests/test_errors.py` used `@pytest.fixture` without
# importing pytest, so it failed at collection with a NameError. Carve-out 1
# missed it (NameError, not SyntaxError) and carve-out 2 missed it (a
# test-CASE file, not infra), so the repair LLM emitted the correct one-line
# import fix and the guard refused it:
#
#   [repair_node:test-guard] Refused 1 repair edit(s) to test file(s):
#   server/tests/test_errors.py
#
# No production change can make `pytest` defined in that module, so the loop
# had no move left. The safety argument is carve-out 1's verbatim: a module
# that cannot be imported has no assertions being evaluated, so making it
# importable cannot weaken one.
# ---------------------------------------------------------------------------

_COLLECTING = "ERROR collecting server/tests/test_errors.py"


def test_unimportable_test_case_file_is_opened(tmp_path):
    ws = _mk_workspace(tmp_path)
    errors = [
        {"error_code": "NameError", "file": "tests/test_contacts.py",
         "message": "name 'pytest' is not defined"},
        {"message": _COLLECTING},
    ]
    assert "tests/test_contacts.py" in _syntax_broken_test_files(errors, ws)


@pytest.mark.parametrize("code", ["NameError", "ImportError", "ModuleNotFoundError"])
def test_each_import_fatal_code_opens_the_file(tmp_path, code):
    ws = _mk_workspace(tmp_path)
    errors = [
        {"error_code": code, "file": "tests/test_contacts.py", "message": "x"},
        {"message": _COLLECTING},
    ]
    assert "tests/test_contacts.py" in _syntax_broken_test_files(errors, ws)


def test_runtime_nameerror_does_not_open_the_file(tmp_path):
    """The load-bearing guard. A NameError raised INSIDE a running test is an
    assertion-phase failure — the file's assertions are live and must stay
    protected, or the model could edit them to make a red test green."""
    ws = _mk_workspace(tmp_path)
    errors = [{
        "error_code": "NameError", "file": "tests/test_contacts.py",
        "message": "name 'undefined_helper' is not defined",
    }]
    assert _syntax_broken_test_files(errors, ws) == frozenset()


def test_assertion_failure_during_a_collection_run_stays_protected(tmp_path):
    """Collection failed somewhere, but THIS file failed on an assertion.
    Carve-out 3 is per-file and keyed on the file's own diagnostic."""
    ws = _mk_workspace(tmp_path)
    errors = [
        {"error_code": "AssertionError", "file": "tests/test_contacts.py",
         "message": "assert 0 == 2"},
        {"message": _COLLECTING},
    ]
    assert "tests/test_contacts.py" not in _syntax_broken_test_files(errors, ws)


def test_production_file_is_never_opened(tmp_path):
    ws = _mk_workspace(tmp_path)
    errors = [
        {"error_code": "NameError", "file": "server/app/main.py",
         "message": "name 'pytest' is not defined"},
        {"message": _COLLECTING},
    ]
    assert _syntax_broken_test_files(errors, ws) == frozenset({
        f for f in _syntax_broken_test_files(errors, ws)
        if not f.startswith("server/app")
    })
    assert os.path.join("server", "app", "main.py") not in \
        _syntax_broken_test_files(errors, ws)


def test_carveout_one_still_works_without_a_collection_error(tmp_path):
    """A SyntaxError opens its file whether or not the run also reported a
    collection abort — carve-out 3 must not have narrowed carve-out 1."""
    ws = _mk_workspace(tmp_path)
    errors = [{"error_code": "SyntaxError", "file": "tests/test_contacts.py"}]
    assert "tests/test_contacts.py" in _syntax_broken_test_files(errors, ws)


# ---------------------------------------------------------------------------
# Phase beats error code.
#
# The code allowlist cannot enumerate every way a module can die on import —
# a decorator evaluating to AttributeError, a module-level TypeError, a
# plugin that raises on import. What they share is that NOTHING in the file
# was evaluated, so editing it cannot weaken an assertion. A diagnostic whose
# own message reports a collection failure is the signal.
# ---------------------------------------------------------------------------

def test_collection_scoped_diagnostic_opens_any_error_code(tmp_path):
    ws = _mk_workspace(tmp_path)
    errors = [{
        "error_code": "AttributeError", "file": "tests/test_contacts.py",
        "message": "ERROR collecting tests/test_contacts.py - module 'pytest' "
                   "has no attribute 'fixtrue'",
    }]
    assert "tests/test_contacts.py" in _syntax_broken_test_files(errors, ws)


def test_assertion_error_elsewhere_stays_protected_during_collection_abort(
    tmp_path,
):
    """The load-bearing distinction: the RUN had a collection failure, but
    THIS file's diagnostic is an assertion that actually ran. Its assertions
    are live and must not become editable."""
    ws = _mk_workspace(tmp_path)
    errors = [
        {"message": "ERROR collecting tests/test_other.py"},
        {"error_code": "AssertionError", "file": "tests/test_contacts.py",
         "message": "assert 0 == 2"},
    ]
    assert "tests/test_contacts.py" not in _syntax_broken_test_files(errors, ws)


def test_a_collected_test_cannot_carry_a_collection_marker(tmp_path):
    """Sanity-check the invariant the safety argument rests on: an
    AssertionError diagnostic has no collection marker, so it can never
    reach the phase-scoped branch."""
    from harness.graph import _diagnostic_is_collection_scoped
    assert not _diagnostic_is_collection_scoped(
        {"error_code": "AssertionError", "message": "assert 0 == 2"}
    )
    assert not _diagnostic_is_collection_scoped({"message": ""})
    assert not _diagnostic_is_collection_scoped({})


def test_production_file_with_a_collection_marker_is_still_never_opened(tmp_path):
    ws = _mk_workspace(tmp_path)
    errors = [{
        "error_code": "ImportError", "file": "server/app/main.py",
        "message": "ERROR collecting server/app/main.py",
    }]
    assert os.path.join("server", "app", "main.py") not in \
        _syntax_broken_test_files(errors, ws)
