"""Cheap static checks over the generated workspace, before the sandbox runs.

The harness has been finding statically-visible defects by dynamic means.
lumina-run8-20260916-2043 is the worked example: ``server/tests/test_employee.py``
was generated complete and well-formed -- 44 lines, docstring, closing
assertion -- calling ``pytest.raises`` while importing only ``Employee``.
Finding that cost a Docker build, a reflection round, three 32,768-token
regenerations (~$0.088), three ``persistent_build_failure`` auto-resumes,
and the run. Ruff reports it in milliseconds.

This module runs the checks that need no container, no install step and no
test execution, and emits ``compiler_errors``-shaped diagnostics so findings
flow into the existing repair loop exactly like ``PROD_IMPORT_SMOKE`` or
``ROUTE_UNRESOLVED`` do.

Design constraints, inherited from :mod:`harness.route_check`:

  * **False-negative-biased.** Only high-confidence findings are emitted. A
    false diagnostic poisons the repair loop -- it sends the model to edit a
    file that is correct -- which is strictly worse than missing a real one
    that the build would have caught anyway. Every heuristic that could not
    be made precise was left out rather than shipped noisy.
  * **Never fatal.** Any internal failure returns no diagnostics. A broken
    checker must not be able to block a build.

Deliberately NOT checked here: date-dependent tests without clock control.
Detecting those reliably needs to distinguish "asserts on a hardcoded date"
from "uses a date incidentally", and the heuristic version would reject good
tests. It stays a known gap rather than a noisy check.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)

# Directories that never contain first-party source worth checking.
_SKIP_DIRS = {
    ".git", ".teane", ".harness", "__pycache__", ".pytest_cache",
    ".hypothesis", "node_modules", ".venv", "venv", "dist", "build",
    ".mypy_cache", ".ruff_cache", "htmlcov", ".tox",
}

# Same opt-outs as the patcher's write gate: a file doing anything dynamic
# with names cannot be judged statically.
_DYNAMIC_NAME_MARKERS = (
    "globals()", "locals()", "exec(", "eval(",
    "import *", "# noqa: F821", "# type: ignore",
)

_F821_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+):(?P<col>\d+):\s+F821\s+"
                      r"Undefined name `(?P<name>[^`]+)`")


def _python_files(workspace: str) -> list[str]:
    """Every first-party ``.py`` file in ``workspace``, workspace-relative."""
    out: list[str] = []
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fn in files:
            if fn.endswith(".py"):
                out.append(os.path.relpath(os.path.join(root, fn), workspace))
    return sorted(out)


def _undefined_name_diagnostics(
    workspace: str, rel_paths: list[str],
) -> list[dict[str, Any]]:
    """Run ruff F821 over ``rel_paths`` and map hits to diagnostics.

    One ruff invocation for the whole set rather than one per file: ruff is
    fast, but process spawn is not, and this runs on every compile.
    """
    checkable: list[str] = []
    for rel in rel_paths:
        try:
            with open(os.path.join(workspace, rel), encoding="utf-8") as fh:
                src = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        if any(m in src for m in _DYNAMIC_NAME_MARKERS):
            continue
        checkable.append(rel)
    if not checkable:
        return []
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ruff", "check",
             "--isolated", "--select", "F821",
             "--output-format", "concise", *checkable],
            capture_output=True, text=True, timeout=60, cwd=workspace,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("[static-preflight] ruff unavailable (%s); skipping.", exc)
        return []
    diags: list[dict[str, Any]] = []
    for line in (proc.stdout or "").splitlines():
        m = _F821_RE.match(line.strip())
        if not m:
            continue
        rel = os.path.relpath(m.group("path"), workspace) \
            if os.path.isabs(m.group("path")) else m.group("path")
        name = m.group("name")
        diags.append({
            "file": rel,
            "line": int(m.group("line")),
            "column": int(m.group("col")),
            "severity": "error",
            "error_code": "STATIC_UNDEFINED_NAME",
            "message": (
                f"NameError at runtime: `{name}` is used here but never "
                "imported, defined, or received as a parameter. The file "
                "parses, so this is invisible until the line executes."
            ),
            "semantic_context": (
                "Found by static preflight before the build ran. Add the "
                f"missing import for `{name}`, or define it. If `{name}` is "
                "meant to be a pytest fixture, it must be declared as a "
                "parameter of the test function."
            ),
        })
    return diags


def run_static_preflight(
    workspace: str, *, limit: int = 25,
) -> list[dict[str, Any]]:
    """Return ``compiler_errors``-shaped diagnostics for ``workspace``.

    Empty list means "nothing high-confidence found" -- which is NOT a claim
    that the build will pass, only that these particular checks are clean.
    Never raises: an internal failure yields no diagnostics.
    """
    try:
        rels = _python_files(workspace)
        if not rels:
            return []
        diags = _undefined_name_diagnostics(workspace, rels)
        # A constraint the spec states and the model omits is statically
        # visible too, and costs a whole run when it is not surfaced: the
        # property-test generator reads the bare annotation, asserts that any
        # string is valid, and deadlocks against the validation tests
        # (lumina-run20-20260925-1342). See harness/spec_constraints.
        try:
            from harness.spec_constraints import constraint_diagnostics
            diags = diags + constraint_diagnostics(workspace)
        except Exception as exc:  # noqa: BLE001 — a checker must never block
            logger.debug("[static-preflight] constraint check skipped: %s", exc)
    except Exception as exc:  # noqa: BLE001 — a checker must never block a build
        logger.warning(
            "[static-preflight] check failed (%s); continuing without it.",
            exc,
        )
        return []
    if not diags:
        logger.info(
            "[static-preflight] %d Python file(s) clean on name resolution.",
            len(rels),
        )
        return []
    _undef = [d for d in diags
              if d.get("error_code") != "SPEC_CONSTRAINT_NOT_DECLARED"]
    _constraints = len(diags) - len(_undef)
    logger.warning(
        "[static-preflight] %d undefined-name defect(s) and %d undeclared "
        "spec constraint(s) across %d file(s), found without running the "
        "build: %s",
        len(_undef), _constraints, len({d["file"] for d in diags}),
        ", ".join(sorted({d["file"] for d in diags})[:5]),
    )
    return diags[:limit]
