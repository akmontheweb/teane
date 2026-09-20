"""Find statically-visible defects statically.

lumina-run8-20260916-2043 spent an entire run discovering one missing
import. server/tests/test_employee.py was generated complete and
well-formed -- 44 lines, docstring, closing assertion -- calling
pytest.raises at line 32 while importing only Employee. ast.parse accepts
that file, so the write gate passed it and pytest was the first thing to
notice.

The cost of finding it dynamically: a Docker build, a reflection round,
three 32,768-token regenerations (~$0.088), three persistent_build_failure
auto-resumes, the run. Measured cost of finding it statically against that
same workspace: 1 diagnostic, exact line, 0.02s.

The checker is false-negative-biased on purpose. A false diagnostic sends
the repair loop to edit a file that is correct, which is worse than missing
one the build would have caught anyway -- so most of these tests pin the
cases it must stay SILENT on.
"""

from __future__ import annotations

from harness.static_preflight import run_static_preflight


def _ws(tmp_path, files: dict[str, str]) -> str:
    for rel, src in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
    return str(tmp_path)


class TestCatchesTheRun8Defect:
    def test_missing_import_is_found(self, tmp_path) -> None:
        ws = _ws(tmp_path, {"server/tests/test_employee.py": (
            '"""Tests for the employee record model."""\n\n'
            "from server.app.models.employee import Employee\n\n\n"
            "class TestEmployee:\n"
            "    def test_immutable(self):\n"
            "        with pytest.raises(AttributeError):\n"
            "            pass\n"
        )})
        diags = run_static_preflight(ws)
        assert len(diags) == 1
        d = diags[0]
        assert d["file"] == "server/tests/test_employee.py"
        assert d["error_code"] == "STATIC_UNDEFINED_NAME"
        assert d["severity"] == "error"
        assert "pytest" in d["message"]

    def test_diagnostic_carries_a_usable_line_number(self, tmp_path) -> None:
        ws = _ws(tmp_path, {"a.py": "x = 1\ny = 2\nz = undefined_thing\n"})
        diags = run_static_preflight(ws)
        assert diags and diags[0]["line"] == 3

    def test_context_tells_the_model_about_fixtures(self, tmp_path) -> None:
        """The most likely honest cause in a test file."""
        ws = _ws(tmp_path, {"t.py": "def test_a():\n    assert some_fixture\n"})
        diags = run_static_preflight(ws)
        assert diags and "fixture" in diags[0]["semantic_context"]


class TestStaysSilent:
    def test_correct_workspace(self, tmp_path) -> None:
        ws = _ws(tmp_path, {
            "server/app/m.py": "def f():\n    return 1\n",
            "server/tests/test_m.py": (
                "import pytest\n"
                "from server.app.m import f\n\n"
                "def test_f():\n"
                "    with pytest.raises(TypeError):\n"
                "        f(1)\n"
            ),
        })
        assert run_static_preflight(ws) == []

    def test_fixture_parameters(self, tmp_path) -> None:
        ws = _ws(tmp_path, {"t.py": (
            "def test_a(tmp_path, monkeypatch):\n"
            "    assert tmp_path and monkeypatch\n"
        )})
        assert run_static_preflight(ws) == []

    def test_star_import_file_is_skipped(self, tmp_path) -> None:
        ws = _ws(tmp_path, {"t.py": "from m import *\n\ndef f():\n    return thing()\n"})
        assert run_static_preflight(ws) == []

    def test_noqa_is_respected(self, tmp_path) -> None:
        ws = _ws(tmp_path, {"t.py": "def f():\n    return magic  # noqa: F821\n"})
        assert run_static_preflight(ws) == []

    def test_vendored_and_cache_dirs_are_not_walked(self, tmp_path) -> None:
        ws = _ws(tmp_path, {
            "node_modules/pkg/bad.py": "x = undefined_name\n",
            ".venv/lib/bad.py": "y = also_undefined\n",
            "__pycache__/bad.py": "z = nope\n",
            "good.py": "a = 1\n",
        })
        assert run_static_preflight(ws) == []

    def test_empty_workspace(self, tmp_path) -> None:
        assert run_static_preflight(str(tmp_path)) == []

    def test_non_python_files_ignored(self, tmp_path) -> None:
        ws = _ws(tmp_path, {"notes.md": "pytest.raises everywhere\n",
                            "data.json": '{"a": 1}'})
        assert run_static_preflight(ws) == []


class TestNeverBlocksABuild:
    def test_missing_workspace_returns_empty(self) -> None:
        assert run_static_preflight("/nonexistent/path/xyz") == []

    def test_unreadable_file_is_skipped_not_fatal(self, tmp_path) -> None:
        ws = _ws(tmp_path, {"good.py": "a = 1\n"})
        (tmp_path / "binary.py").write_bytes(b"\xff\xfe\x00bad")
        assert run_static_preflight(ws) == []

    def test_limit_caps_the_diagnostic_count(self, tmp_path) -> None:
        src = "\n".join(f"v{i} = undefined_{i}" for i in range(40))
        ws = _ws(tmp_path, {"many.py": src})
        assert len(run_static_preflight(ws, limit=5)) == 5
