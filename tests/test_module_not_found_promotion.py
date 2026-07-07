"""Regression tests for the 2026-07-07 ModuleNotFoundError → MISSING_DEP
promotion in ``compiler_node``.

User-reported symptom: pytest hit
``ModuleNotFoundError: No module named 'sqlalchemy'`` during test
collection of ``backend/core/db.py:9``. The parser stamped the diag as
``error_code=ModuleNotFoundError`` with no ``missing_symbol`` — which
made autofix's ``_try_missing_dep`` skip the diagnostic entirely
(autofix.py:824 requires ``error_code == "MISSING_DEP"``). The
env-misconfig synthesis at compiler_node's other MISSING_DEP path is
gated on ``not compiler_errors``, so a pytest-produced diagnostic
bypassed both branches and the LLM had to figure out a mechanical
requirements.txt fix on its own — usually badly.

The promoter closes the gap by walking the extracted diagnostics
after ``compiler_node`` runs the build and rewriting Python module-
miss shapes into ``MISSING_DEP`` so autofix / the router can consume
them. Broad by design — any top-level module name that is not a
workspace source directory gets promoted (no ``_PIP_INSTALLABLE_
SYMBOLS`` whitelist gate), because PyPI resolution failure at
``pip install`` is a strictly better UX than the LLM burning rounds
on the wrong file.
"""

from __future__ import annotations

from harness.graph import _promote_module_not_found_diagnostics


class TestModuleNotFoundPromotion:
    def test_sqlalchemy_shape_gets_promoted(self, tmp_path):
        # The exact shape the parser emits for the user's session.
        # workspace has no ``sqlalchemy`` directory → treat as dep.
        (tmp_path / "backend").mkdir()
        (tmp_path / "backend" / "core").mkdir()
        (tmp_path / "backend" / "core" / "db.py").write_text("import sqlalchemy\n")
        diags = [{
            "file": "backend/core/db.py",
            "line": 9,
            "column": 0,
            "severity": "error",
            "error_code": "ModuleNotFoundError",
            "message": "ModuleNotFoundError: No module named 'sqlalchemy'",
        }]
        promoted = _promote_module_not_found_diagnostics(
            diags, str(tmp_path), "python3 -m pytest -q",
        )
        assert promoted == 1
        assert diags[0]["error_code"] == "MISSING_DEP"
        assert diags[0]["missing_symbol"] == "sqlalchemy"
        assert diags[0]["miss_kind"] == "python"
        assert diags[0]["build_command"] == "python3 -m pytest -q"
        # File/line/message preserved — the LLM prompt still shows the
        # import site, which is useful context for the reflection judge.
        assert diags[0]["file"] == "backend/core/db.py"
        assert diags[0]["line"] == 9
        # Original exception type recorded in semantic_context so the
        # judge / prompt builders don't lose the signal.
        assert "ModuleNotFoundError" in diags[0]["semantic_context"]

    def test_workspace_source_name_not_promoted(self, tmp_path):
        # Session 3193a24f: pytest reports "No module named 'server'"
        # when ``server/`` is a workspace source directory (PATH/CWD
        # bug, not a dep). Must NOT be promoted or _try_missing_dep
        # will append ``server`` to requirements.txt and pip install
        # will fail with a nonsense package resolution error.
        (tmp_path / "server").mkdir()
        (tmp_path / "server" / "__init__.py").write_text("")
        diags = [{
            "file": "tests/test_x.py",
            "line": 1,
            "error_code": "ModuleNotFoundError",
            "message": "ModuleNotFoundError: No module named 'server'",
        }]
        promoted = _promote_module_not_found_diagnostics(
            diags, str(tmp_path), "pytest",
        )
        assert promoted == 0
        # Diag untouched — router keeps steering repair via the
        # workspace-source signal (see
        # _missing_module_matches_workspace_source at graph.py:7047).
        assert diags[0]["error_code"] == "ModuleNotFoundError"
        assert "missing_symbol" not in diags[0]

    def test_dotted_module_uses_top_level_for_source_check(self, tmp_path):
        # ``sqlalchemy.orm`` resolves against sys.path via the leading
        # segment ``sqlalchemy``. The workspace check must use the
        # top-level too so ``backend.core.db`` (source) doesn't get
        # promoted just because the fully-qualified name isn't a
        # directory tree at the root.
        (tmp_path / "backend").mkdir()
        # Case A: dotted external dep — promote.
        diag_ext = {
            "error_code": "ModuleNotFoundError",
            "message": "ModuleNotFoundError: No module named 'sqlalchemy.orm'",
        }
        # Case B: dotted internal module — do NOT promote.
        diag_int = {
            "error_code": "ModuleNotFoundError",
            "message": "ModuleNotFoundError: No module named 'backend.core.db'",
        }
        diags = [diag_ext, diag_int]
        promoted = _promote_module_not_found_diagnostics(
            diags, str(tmp_path), "pytest",
        )
        assert promoted == 1
        assert diag_ext["error_code"] == "MISSING_DEP"
        assert diag_ext["missing_symbol"] == "sqlalchemy"
        assert diag_int["error_code"] == "ModuleNotFoundError"

    def test_import_error_shape_also_promoted(self, tmp_path):
        # Pytest sometimes surfaces the miss as ``ImportError`` (via a
        # broader ``try/except ImportError`` around the offending
        # import). Same fix — treat both as promotable when the module
        # name is extractable.
        (tmp_path / "app").mkdir()
        diags = [{
            "error_code": "ImportError",
            "message": (
                "ImportError: No module named 'httpx'"
            ),
        }]
        promoted = _promote_module_not_found_diagnostics(
            diags, str(tmp_path), "pytest",
        )
        assert promoted == 1
        assert diags[0]["error_code"] == "MISSING_DEP"
        assert diags[0]["missing_symbol"] == "httpx"

    def test_non_python_error_codes_untouched(self, tmp_path):
        # TypeScript ``TS2307`` has its own handler; JS ``Cannot find
        # module`` goes through autofix R7 (_try_missing_npm_dep). We
        # don't reroute those through the Python autofix. Only Python
        # exception types promote.
        diags = [{
            "error_code": "TS2307",
            "message": "Cannot find module 'react'",
        }, {
            "error_code": "SyntaxError",  # not a module-miss shape
            "message": "invalid syntax",
        }]
        promoted = _promote_module_not_found_diagnostics(
            diags, str(tmp_path), "npm test",
        )
        assert promoted == 0
        assert diags[0]["error_code"] == "TS2307"
        assert diags[1]["error_code"] == "SyntaxError"

    def test_broad_promotion_no_whitelist_gate(self, tmp_path):
        # The whole point of the "broad" design: even for a package
        # not in ``_PIP_INSTALLABLE_SYMBOLS`` (e.g. ``httpx-oauth``,
        # ``pytest-postgresql``), we still promote. PyPI resolution
        # failure downstream is strictly better UX than the LLM burning
        # 3 rounds on the wrong file because autofix silently skipped.
        (tmp_path / "backend").mkdir()
        diags = [{
            "error_code": "ModuleNotFoundError",
            "message": (
                "ModuleNotFoundError: No module named "
                "'some_uncommon_pypi_package_2026'"
            ),
        }]
        promoted = _promote_module_not_found_diagnostics(
            diags, str(tmp_path), "pytest",
        )
        assert promoted == 1
        assert diags[0]["missing_symbol"] == "some_uncommon_pypi_package_2026"

    def test_empty_diagnostics_short_circuits(self, tmp_path):
        # No diags → returns 0 without touching disk (via
        # _project_top_level_names). Cheap fast path for the common
        # success case where compiler_node has nothing to promote.
        assert _promote_module_not_found_diagnostics(
            [], str(tmp_path), "pytest",
        ) == 0

    def test_semantic_context_preserved_when_promoting(self, tmp_path):
        # If the original diag already had semantic_context (from the
        # pytest parser's failing-source / assertion-rewrite capture),
        # the promoter must preserve it — the repair prompt reads
        # semantic_context, and dropping it would lose the wider frame
        # even though the promoter's own note gets appended.
        (tmp_path / "backend").mkdir()
        original_ctx = "failing source: from sqlalchemy import Column"
        diags = [{
            "error_code": "ModuleNotFoundError",
            "message": "ModuleNotFoundError: No module named 'sqlalchemy'",
            "semantic_context": original_ctx,
        }]
        _promote_module_not_found_diagnostics(
            diags, str(tmp_path), "pytest",
        )
        assert original_ctx in diags[0]["semantic_context"]
        # And the promoter's own note is appended.
        assert "Promoted from ModuleNotFoundError" in diags[0]["semantic_context"]
