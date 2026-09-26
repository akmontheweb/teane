"""Static checks on GENERATED TESTS that the build finds the expensive way.

Both checks here were predicted from the shape of earlier failures rather
than from a run that had already lost to them, and both were then confirmed
present in a real workspace. They cost no container and no LLM call.

The failures they catch share a property: the test is wrong in a way that
makes the PRODUCTION code look wrong, so the repair loop spends its rounds
editing correct code. That is the most expensive kind of generated-test
defect this harness has, and it has ended several lumina runs.

Design constraints are :mod:`harness.static_preflight`'s: false-negative
biased, never fatal. Each check fires only when the workspace itself shows
the precondition — an env-configured database, a UTC clock — so a project
that does not have the hazard never sees the diagnostic.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Env var names that mean "the database location is chosen at runtime".
_DB_ENV_RE = re.compile(
    r"\b(DATABASE_PATH|DATABASE_URL|DB_PATH|DB_URL|DB_DSN|SQLALCHEMY_DATABASE_URI)\b"
)

_TEST_DIR_PARTS = (os.sep + "tests" + os.sep, os.sep + "test" + os.sep)
_SKIP_DIRS = {"node_modules", ".venv", ".git", "__pycache__"}


def _walk_python(workspace: str, *, max_files: int = 400) -> list[str]:
    out: list[str] = []
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        for name in files:
            if name.endswith(".py"):
                out.append(os.path.relpath(os.path.join(root, name), workspace))
                if len(out) >= max_files:
                    return out
    return out


def _is_test_file(rel: str) -> bool:
    norm = os.sep + rel
    base = os.path.basename(rel)
    return (any(p in norm for p in _TEST_DIR_PARTS)
            or base.startswith("test_") or base.endswith("_test.py"))


def _read(workspace: str, rel: str) -> Optional[str]:
    try:
        with open(os.path.join(workspace, rel), "r",
                  encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def _app_reads_db_env(workspace: str, rels: list[str]) -> bool:
    """True when production code chooses its database from the environment."""
    for rel in rels:
        if _is_test_file(rel):
            continue
        src = _read(workspace, rel)
        if src and ("environ" in src or "getenv" in src) and _DB_ENV_RE.search(src):
            return True
    return False


def _app_is_utc(workspace: str, rels: list[str]) -> bool:
    """True when production code declares a UTC clock."""
    for rel in rels:
        if _is_test_file(rel):
            continue
        src = _read(workspace, rel)
        if src and "timezone.utc" in src and (
                "clock" in rel.lower() or "class Clock" in src):
            return True
    return False


def module_level_client_diagnostics(
    workspace: str, rels: list[str],
) -> list[dict[str, Any]]:
    """A test client built at IMPORT time, in an env-configured app.

    ``client = TestClient(app)`` at module scope constructs the application
    — and binds whatever database path the environment held at import —
    before any fixture can redirect it. Every test in the file then shares
    that one application, so per-test isolation silently does not apply.

    lumina-run17-20260924-2351 lost two HITL trips to this: the contract
    tests returned 500 instead of 422 because the app's connection was
    opened against an unwritable path fixed at import, and the judge spent
    its rounds guessing at row factories and config defaults.
    """
    if not _app_reads_db_env(workspace, rels):
        return []
    out: list[dict[str, Any]] = []
    for rel in rels:
        if not _is_test_file(rel):
            continue
        src = _read(workspace, rel)
        if not src or "TestClient" not in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:  # module scope only
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if not isinstance(value, ast.Call):
                continue
            fn = value.func
            fname = getattr(fn, "id", "") or getattr(fn, "attr", "")
            if fname != "TestClient":
                continue
            out.append({
                "file": rel,
                "line": getattr(node, "lineno", 0),
                "column": 0,
                "severity": "error",
                "error_code": "TEST_CLIENT_BOUND_AT_IMPORT",
                "message": (
                    "This module builds its TestClient at import time, but "
                    "the application chooses its database from the "
                    "environment. The app — and its database connection — "
                    "is therefore bound before any fixture can point it at "
                    "a per-test location, so every test in this file shares "
                    "one application and the isolation the suite assumes "
                    "does not apply. Build the client inside a fixture "
                    "(after the environment is set) and request it per "
                    "test. Left as is, failures depend on which test ran "
                    "first, and a database error surfaces as a wrong status "
                    "code rather than as itself."
                ),
                "semantic_context": (
                    "Module-level TestClient(...) in an env-configured app."
                ),
            })
    return out


def naive_date_diagnostics(
    workspace: str, rels: list[str],
) -> list[dict[str, Any]]:
    """Tests computing LOCAL dates against a UTC application.

    ``date.today()`` and ``datetime.now()`` without a timezone return the
    machine's local date. An application on UTC disagrees with them for
    part of every day — the whole of it, at large offsets — so an expected
    value computed from a local date is off by one for those hours. The
    test then passes or fails according to the clock on the wall, which is
    the hardest kind of failure to read from one run.
    """
    if not _app_is_utc(workspace, rels):
        return []
    out: list[dict[str, Any]] = []
    for rel in rels:
        if not _is_test_file(rel):
            continue
        src = _read(workspace, rel)
        if not src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = getattr(fn, "attr", "") or getattr(fn, "id", "")
            if name not in ("today", "now", "utcnow"):
                continue
            if name == "utcnow":
                pass  # deprecated and naive: still a mismatch
            elif node.args or node.keywords:
                continue  # now(timezone.utc) etc. is explicit; leave it
            out.append({
                "file": rel,
                "line": getattr(node, "lineno", 0),
                "column": 0,
                "severity": "error",
                "error_code": "NAIVE_LOCAL_DATE_IN_TEST",
                "message": (
                    f"`{name}()` here returns the machine's LOCAL date/time, "
                    "while the application runs on UTC. The two differ for "
                    "part of every day, so any expected value computed from "
                    "this — a day count, a window boundary, a next "
                    "occurrence — is off by one during those hours, and the "
                    "test passes or fails by the time of day it runs. Use "
                    "the application's own clock (or pin it, where the "
                    "suite exposes a freeze), or compute the expectation in "
                    "UTC: datetime.now(timezone.utc).date()."
                ),
                "semantic_context": (
                    "Local-time call in a test against a UTC application."
                ),
            })
            break  # one finding per file is enough to act on
    return out


def run_test_hygiene_checks(
    workspace: str, *, limit: int = 12,
) -> list[dict[str, Any]]:
    """All hygiene diagnostics for ``workspace``. Never raises."""
    try:
        rels = _walk_python(workspace)
        if not rels:
            return []
        found = (module_level_client_diagnostics(workspace, rels)
                 + naive_date_diagnostics(workspace, rels))
    except Exception as exc:  # noqa: BLE001 — a checker must never block
        logger.debug("[test-hygiene] checks skipped: %s", exc)
        return []
    return found[:limit]
