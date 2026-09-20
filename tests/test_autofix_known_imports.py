"""R2 knew the symbol and still declined to act.

lumina-run8-20260916-2043 died on a missing `import pytest`.
_detect_missing_symbol correctly extracted ('python', 'pytest') from the
diagnostic — the detector was never the problem. _try_missing_import then
returned None, because it resolves a symbol by grep-walking the workspace
for its definition and requiring EXACTLY ONE match:

    pytest    -> 0 workspace definition(s)
    datetime  -> 0 workspace definition(s)
    Employee  -> 1 workspace definition(s)   <- the only shape R2 handled

Zero definitions is correct for first-party code and structurally blind to
the stdlib and installed packages, where zero is not ambiguity but "you
looked in the wrong place". A one-line import went to the repair loop, and
the run died on it.
"""

from __future__ import annotations


from harness.autofix import (
    _detect_missing_symbol,
    _known_import_statement,
    _module_available_to_workspace,
    _try_missing_import,
)

RUNTIME_MSG = "NameError: name 'pytest' is not defined"
PREFLIGHT_MSG = (
    "NameError at runtime: `pytest` is used here but never imported, "
    "defined, or received as a parameter."
)


def _ws(tmp_path, files=None):
    (tmp_path / "server" / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "server" / "tests" / "test_a.py").write_text(
        "class T:\n"
        "    def test_x(self):\n"
        "        with pytest.raises(ValueError):\n"
        "            pass\n"
    )
    for rel, src in (files or {}).items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
    return str(tmp_path)


class TestDetectorCoversBothDiagnosticShapes:
    def test_runtime_nameerror(self) -> None:
        assert _detect_missing_symbol(".py", "", RUNTIME_MSG) == ("python", "pytest")

    def test_static_preflight_diagnostic(self) -> None:
        """Without this the preflight finds it in 0.02s and still pays an LLM."""
        assert _detect_missing_symbol(
            ".py", "STATIC_UNDEFINED_NAME", PREFLIGHT_MSG
        ) == ("python", "pytest")


class TestKnownImportTable:
    def test_stdlib_needs_no_dependency_check(self, tmp_path) -> None:
        ws = _ws(tmp_path)
        assert _known_import_statement("datetime", ws) == "import datetime"

    def test_from_import_form_is_used_where_correct(self, tmp_path) -> None:
        """`Path` must not become `import Path`."""
        ws = _ws(tmp_path)
        assert _known_import_statement("Path", ws) == "from pathlib import Path"

    def test_sandbox_guaranteed_package(self, tmp_path) -> None:
        """freezegun is installed by the sandbox preamble, not the manifest."""
        ws = _ws(tmp_path)
        assert _known_import_statement("freeze_time", ws) == (
            "from freezegun import freeze_time"
        )

    def test_third_party_absent_from_manifests_is_refused(self, tmp_path) -> None:
        """Trading NameError for ModuleNotFoundError is not a fix."""
        ws = _ws(tmp_path)
        assert _known_import_statement("pandas", ws) is None

    def test_third_party_declared_in_manifest_is_allowed(self, tmp_path) -> None:
        ws = _ws(tmp_path, {"requirements.txt": "numpy==2.1.0\n"})
        assert _known_import_statement("numpy", ws) == "import numpy"

    def test_unknown_symbol_is_refused(self, tmp_path) -> None:
        """A closed allowlist: absent names behave exactly as before."""
        ws = _ws(tmp_path)
        assert _known_import_statement("Employee", ws) is None
        assert _known_import_statement("some_helper", ws) is None


class TestAvailabilityCheck:
    def test_server_subdir_manifest_is_consulted(self, tmp_path) -> None:
        ws = _ws(tmp_path, {"server/requirements.txt": "httpx>=0.27\n"})
        assert _module_available_to_workspace("httpx", ws) is True

    def test_missing_manifests_do_not_raise(self, tmp_path) -> None:
        ws = _ws(tmp_path)
        assert _module_available_to_workspace("definitely_not_a_pkg", ws) is False


class TestEndToEnd:
    def test_emits_a_patch_for_the_run8_defect(self, tmp_path) -> None:
        ws = _ws(tmp_path)
        block = _try_missing_import(
            {"file": "server/tests/test_a.py", "line": 3,
             "message": RUNTIME_MSG, "error_code": ""}, ws,
        )
        assert block is not None
        assert block.file == "server/tests/test_a.py"

    def test_emits_a_patch_from_the_preflight_diagnostic(self, tmp_path) -> None:
        ws = _ws(tmp_path)
        block = _try_missing_import(
            {"file": "server/tests/test_a.py", "line": 3,
             "message": PREFLIGHT_MSG,
             "error_code": "STATIC_UNDEFINED_NAME"}, ws,
        )
        assert block is not None

    def test_ambiguous_workspace_symbol_still_defers_to_the_llm(self, tmp_path) -> None:
        """len(candidates) > 1 is untouched — the LLM has the context."""
        ws = _ws(tmp_path, {
            "server/app/a.py": "class Widget:\n    pass\n",
            "server/app/b.py": "class Widget:\n    pass\n",
            "server/tests/test_w.py": "def test_w():\n    assert Widget\n",
        })
        block = _try_missing_import(
            {"file": "server/tests/test_w.py", "line": 2,
             "message": "NameError: name 'Widget' is not defined",
             "error_code": ""}, ws,
        )
        assert block is None

    def test_already_imported_file_is_left_alone(self, tmp_path) -> None:
        ws = _ws(tmp_path, {
            "server/tests/test_b.py": "import pytest\n\ndef test_b():\n    assert pytest\n",
        })
        block = _try_missing_import(
            {"file": "server/tests/test_b.py", "line": 4,
             "message": RUNTIME_MSG, "error_code": ""}, ws,
        )
        assert block is None
