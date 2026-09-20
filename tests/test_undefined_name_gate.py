"""Parsing is necessary but not sufficient.

lumina-run8-20260916-2043 died on a missing import. ``server/tests/test_employee.py``
was generated complete and well-formed -- 44 lines, docstring, closing
assertion -- calling ``pytest.raises`` at line 32 while importing only
``Employee``. ``ast.parse`` accepts that file, so the write gate passed it,
and the defect only surfaced when pytest ran it.

Discovering it dynamically cost a Docker build, a reflection round, three
32,768-token regenerations (~$0.088), three persistent_build_failure
auto-resumes, and the run. Ruff reports it in ~26ms before the write.

The gate is deliberately conservative: a false positive here rolls back a
patch the model got RIGHT, which is strictly worse than missing a bad one.
Most of these tests pin the cases it must stay SILENT on.
"""

from __future__ import annotations

from harness.patcher import _validate_syntax


def _check(src: str, path: str = "server/tests/test_x.py"):
    return _validate_syntax(path, src)


class TestCatchesTheRealDefect:
    def test_run8_missing_pytest_import(self) -> None:
        src = (
            "class TestEmployee:\n"
            "    def test_immutable(self):\n"
            "        with pytest.raises(AttributeError):\n"
            "            pass\n"
        )
        err = _check(src)
        assert err is not None
        assert "pytest" in err

    def test_message_explains_the_consequence(self) -> None:
        """The model needs to know why, not just that."""
        err = _check("def f():\n    return missing_helper()\n")
        assert err is not None
        assert "NameError" in err

    def test_reports_several_names(self) -> None:
        src = "def f():\n    return alpha() + beta() + gamma()\n"
        err = _check(src)
        assert err is not None
        for n in ("alpha", "beta", "gamma"):
            assert n in err


class TestStaysSilentOnLegitimateCode:
    """Every one of these is a patch that must NOT be rolled back."""

    def test_correctly_imported(self) -> None:
        assert _check(
            "import pytest\n\n"
            "def test_a():\n"
            "    with pytest.raises(ValueError):\n"
            "        pass\n"
        ) is None

    def test_pytest_fixture_parameters(self) -> None:
        """Fixtures arrive as function parameters, not imports."""
        assert _check(
            "def test_a(tmp_path, monkeypatch, capsys):\n"
            "    assert tmp_path and monkeypatch and capsys\n"
        ) is None

    def test_star_import(self) -> None:
        """With a star-import the names genuinely are not statically visible."""
        assert _check("from mod import *\n\ndef test_a():\n    assert thing()\n") is None

    def test_dynamic_name_injection(self) -> None:
        assert _check('def f():\n    globals()["x"] = 1\n    return x\n') is None

    def test_explicit_noqa_is_respected(self) -> None:
        assert _check("def f():\n    return magic  # noqa: F821\n") is None

    def test_type_checking_string_annotation(self) -> None:
        assert _check(
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    from x import Y\n\n"
            'def f(a: "Y") -> None:\n'
            "    pass\n"
        ) is None

    def test_class_and_method_self_reference(self) -> None:
        assert _check(
            "class A:\n"
            "    def a(self):\n"
            "        return self.b()\n"
            "    def b(self):\n"
            "        return A\n"
        ) is None

    def test_comprehension_and_walrus_scoping(self) -> None:
        assert _check(
            "def f(items):\n"
            "    squares = [x * x for x in items]\n"
            "    if (n := len(squares)) > 0:\n"
            "        return n\n"
            "    return 0\n"
        ) is None


class TestGateOrdering:
    def test_syntax_error_still_wins(self) -> None:
        """A file that does not parse reports the SyntaxError, not F821."""
        err = _validate_syntax("m.py", "def f(:\n")
        assert err is not None and "SyntaxError" in err

    def test_non_python_files_untouched(self) -> None:
        assert _validate_syntax("a.json", '{"a": 1}') is None
        assert _validate_syntax("notes.md", "pytest.raises everywhere") is None
