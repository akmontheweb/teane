"""teane doctor — runtime-dependency healthcheck + preflight guard.

A missing or ABI-incompatible runtime dependency used to surface as a raw
ImportError traceback partway through startup. `_doctor_check_dependencies`
turns that into an upfront doctor FAIL, and `_preflight_dependency_guard`
aborts `teane run` / `teane resume` cleanly with a fix. These tests pin the
pass/fail behaviour (env-independent, via monkeypatched imports) and assert
the dependency table stays in sync with pyproject.toml so it can't drift.
"""

from __future__ import annotations

import importlib
import pathlib

import pytest

from harness.cli import (
    _RUNTIME_DEPENDENCIES,
    _doctor_check_dependencies,
    _missing_runtime_dependencies,
    _preflight_dependency_guard,
)

_REAL_IMPORT = importlib.import_module


def _fake_import(*, fail: set[str]):
    def _imp(name, package=None):
        if name in fail:
            raise ImportError(f"No module named {name!r} (simulated)")
        return _REAL_IMPORT(name, package)
    return _imp


class TestDoctorCheck:
    def test_pass_when_all_import(self, monkeypatch):
        monkeypatch.setattr(importlib, "import_module", _fake_import(fail=set()))
        status, detail = _doctor_check_dependencies()
        assert status == "pass"
        assert str(len(_RUNTIME_DEPENDENCIES)) in detail

    def test_fail_names_missing_and_fix(self, monkeypatch):
        monkeypatch.setattr(importlib, "import_module", _fake_import(fail={"langgraph", "httpx"}))
        status, detail = _doctor_check_dependencies()
        assert status == "fail"
        # pip names, not import names, and the fix command
        assert "langgraph" in detail and "httpx" in detail
        assert "pip install -e ." in detail

    def test_broken_import_counts_as_missing(self, monkeypatch):
        # A dep present but broken on the ABI raises something other than
        # ImportError on import; it must still be reported.
        def _imp(name, package=None):
            if name == "tree_sitter":
                raise RuntimeError("bad ABI")
            return _REAL_IMPORT(name, package)
        monkeypatch.setattr(importlib, "import_module", _imp)
        assert "tree-sitter" in _missing_runtime_dependencies()


class TestPreflightGuard:
    def test_none_when_complete(self, monkeypatch):
        monkeypatch.setattr(importlib, "import_module", _fake_import(fail=set()))
        assert _preflight_dependency_guard() is None

    def test_returns_exit_and_prints(self, monkeypatch, capsys):
        monkeypatch.setattr(importlib, "import_module", _fake_import(fail={"uuid7"}))
        rc = _preflight_dependency_guard()
        assert rc == 1
        err = capsys.readouterr().err
        assert "uuid7" in err and "pip install -e ." in err


class TestNoDrift:
    def test_table_matches_pyproject(self):
        """The pip names in _RUNTIME_DEPENDENCIES must exactly equal the
        distributions in pyproject.toml [project.dependencies]. If someone
        adds/removes a dependency without updating the table, this fails."""
        try:
            import tomllib
        except ModuleNotFoundError:  # py<3.11
            pytest.skip("tomllib requires Python 3.11+")
        root = pathlib.Path(__file__).resolve().parents[1]
        data = tomllib.loads((root / "pyproject.toml").read_text())
        # strip version specifiers, normalize case
        declared = {
            _split_dist(d) for d in data["project"]["dependencies"]
        }
        table = {pip for _imp, pip in _RUNTIME_DEPENDENCIES}
        assert table == declared, (
            f"dependency table out of sync with pyproject.toml.\n"
            f"  only in table:     {sorted(table - declared)}\n"
            f"  only in pyproject: {sorted(declared - table)}"
        )

    def test_import_names_actually_import_in_this_env(self):
        """In a properly-installed env every declared import name resolves.
        Skips gracefully where the deps aren't installed (e.g. a minimal
        sandbox) so it never gives a false failure."""
        for import_name, pip in _RUNTIME_DEPENDENCIES:
            try:
                importlib.import_module(import_name)
            except Exception:
                pytest.skip(f"{pip} not installed in this environment")


def _split_dist(requirement: str) -> str:
    import re
    return re.split(r"[<>=!~;\[ ]", requirement.strip(), 1)[0].lower()
