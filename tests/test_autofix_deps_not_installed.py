"""Regression tests for the DEPS_NOT_INSTALLED autofix path.

Motivation: ciod session 523e86a7 spent 3+ hours in a REPLACE_BLOCK /
DELETE_BLOCK LLM loop on ``requirements.txt`` because ``flask``,
``flask_cors``, and ``flask_limiter`` were bundled into ONE
DEPS_NOT_INSTALLED diagnostic from ``_run_prod_import_smoke_check``
and the per-symbol ``_try_missing_dep`` handler couldn't consume it.
The new ``_try_deps_not_installed`` handler batch-appends every
missing package to the manifest in a single atomic PatchBlock so the
next compile picks them all up at once.
"""

from __future__ import annotations



from harness.autofix import _try_deps_not_installed, _try_multiple_deps_pyproject
from harness.patcher import OperationType


def _diag(packages, *, message=None, error_code="DEPS_NOT_INSTALLED"):
    """Shape a minimal DEPS_NOT_INSTALLED diagnostic matching
    ``_run_prod_import_smoke_check``'s emission format."""
    if message is None:
        message = (
            f"{len(packages)} third-party Python package(s) failed to "
            f"import. Missing: {', '.join(packages)}. "
            "Fix: ensure each is declared in `requirements.txt` ..."
        )
    return {
        "error_code": error_code,
        "message": message,
        "file": "requirements.txt",
        "line": 0,
        "column": 0,
        "severity": "error",
        "missing_packages": list(packages),
    }


class TestDepsNotInstalledStructuredField:
    """The 2026-07-04 fix path — diagnostic carries a
    ``missing_packages`` list that the autofix consumes structurally
    (no message re-parsing)."""

    def test_creates_manifest_when_absent(self, tmp_path):
        block = _try_deps_not_installed(
            _diag(["flask", "flask-cors"]), str(tmp_path),
        )
        assert block is not None
        assert block.operation == OperationType.CREATE_FILE
        assert block.file == "requirements.txt"
        assert "flask" in block.content
        assert "flask-cors" in block.content

    def test_appends_missing_to_existing_manifest(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("gunicorn\npytest\n")
        block = _try_deps_not_installed(
            _diag(["flask", "flask-cors"]), str(tmp_path),
        )
        assert block is not None
        assert block.operation == OperationType.REPLACE_BLOCK
        assert block.search == "pytest"
        # Both deps appended after the last line.
        assert "flask" in block.replace
        assert "flask-cors" in block.replace

    def test_skips_packages_already_declared(self, tmp_path):
        (tmp_path / "requirements.txt").write_text(
            "flask==2.3.0\ngunicorn\n"
        )
        block = _try_deps_not_installed(
            _diag(["flask", "flask-cors", "flask-limiter"]),
            str(tmp_path),
        )
        # ``flask`` already pinned so it should be dropped; only
        # ``flask-cors`` and ``flask-limiter`` land in the append.
        assert block is not None
        assert "flask-cors" in block.replace
        assert "flask-limiter" in block.replace
        # Case-insensitive pin match ensures ``Flask==...`` variants
        # wouldn't get a duplicate.

    def test_returns_none_when_every_package_present(self, tmp_path):
        (tmp_path / "requirements.txt").write_text(
            "flask==2.3.0\nflask-cors\nflask-limiter\n"
        )
        assert _try_deps_not_installed(
            _diag(["flask", "flask-cors", "flask-limiter"]),
            str(tmp_path),
        ) is None

    def test_empty_manifest_gets_all_deps(self, tmp_path):
        # A whitespace-only file — CREATE_FILE would fail on "exists",
        # so the handler must emit a REPLACE_BLOCK that swaps the
        # whitespace for the dep list.
        (tmp_path / "requirements.txt").write_text("\n")
        block = _try_deps_not_installed(
            _diag(["flask", "flask-cors"]), str(tmp_path),
        )
        assert block is not None
        assert block.operation == OperationType.REPLACE_BLOCK
        assert "flask" in block.replace
        assert "flask-cors" in block.replace

    def test_canonical_install_name_used(self, tmp_path):
        # ``bs4`` is a common miss but installs as ``beautifulsoup4``.
        block = _try_deps_not_installed(
            _diag(["bs4"]), str(tmp_path),
        )
        assert block is not None
        assert "beautifulsoup4" in block.content
        assert "bs4\n" not in block.content


class TestDepsNotInstalledMessageFallback:
    """When the diagnostic is replayed from a pre-2026-07-04
    checkpoint the ``missing_packages`` field is absent — the handler
    must fall back to parsing the ``Missing: pkg1, pkg2, ...`` string
    from ``message`` so old sessions get the fix too."""

    def test_parses_message_when_no_structured_field(self, tmp_path):
        diag = {
            "error_code": "DEPS_NOT_INSTALLED",
            "message": (
                "Missing: flask, flask-cors, flask-limiter. "
                "Add them to requirements.txt."
            ),
            "file": "requirements.txt",
        }
        block = _try_deps_not_installed(diag, str(tmp_path))
        assert block is not None
        assert block.operation == OperationType.CREATE_FILE
        assert "flask" in block.content
        assert "flask-cors" in block.content
        assert "flask-limiter" in block.content


class TestDepsNotInstalledIgnoresOtherDiagnostics:
    """The handler must not fire on any diagnostic that isn't
    DEPS_NOT_INSTALLED — otherwise it would swallow unrelated
    diagnostics whose message happens to mention "Missing:"."""

    def test_ignores_missing_dep_diag(self, tmp_path):
        # Single-symbol MISSING_DEP has its own handler
        # (``_try_missing_dep``); this handler must decline.
        assert _try_deps_not_installed({
            "error_code": "MISSING_DEP",
            "missing_symbol": "flask",
        }, str(tmp_path)) is None

    def test_ignores_unrelated_error_code(self, tmp_path):
        assert _try_deps_not_installed({
            "error_code": "SYNTAX_ERROR",
            "message": "Missing: closing paren",
        }, str(tmp_path)) is None


class TestPyprojectBatchAppend:
    """The pyproject.toml path — same batch-append semantics but the
    handler edits ``[project].dependencies`` instead of writing to
    ``requirements.txt``."""

    def _write_pyproject(self, tmp_path, deps_body="") -> None:
        (tmp_path / "pyproject.toml").write_text(
            "[project]\n"
            'name = "app"\n'
            "dependencies = [\n"
            f"{deps_body}"
            "]\n"
        )

    def test_appends_to_pyproject_dependencies(self, tmp_path):
        self._write_pyproject(tmp_path, deps_body='    "gunicorn",\n')
        block = _try_multiple_deps_pyproject(
            ["flask", "flask-cors"], str(tmp_path),
        )
        assert block is not None
        assert block.file == "pyproject.toml"
        assert "flask" in block.replace
        assert "flask-cors" in block.replace
        # Existing entry is preserved.
        assert "gunicorn" in block.replace

    def test_returns_none_when_no_pyproject(self, tmp_path):
        # No pyproject.toml on disk → None so the caller falls back
        # to requirements.txt.
        assert _try_multiple_deps_pyproject(
            ["flask"], str(tmp_path),
        ) is None

    def test_skips_packages_already_present_in_pyproject(self, tmp_path):
        self._write_pyproject(
            tmp_path,
            deps_body='    "flask",\n    "gunicorn",\n',
        )
        # ``flask`` already there — should be filtered out; only
        # ``flask-cors`` should land in the replace body.
        block = _try_multiple_deps_pyproject(
            ["flask", "flask-cors"], str(tmp_path),
        )
        assert block is not None
        assert "flask-cors" in block.replace
        # Duplicate check — ``flask`` still present exactly once.
        assert block.replace.count('"flask"') == 1


class TestFileFieldRouting:
    """Route on the diag's ``file`` field so autofix writes to the SAME
    manifest the install step reads. Prior routing sniffed a
    ``build_command`` field the DEPS_NOT_INSTALLED producer never set,
    so pyproject-based workspaces stalled: autofix would append celery
    to a root ``requirements.txt`` while the install step ran ``uv pip
    install -e .`` against pyproject.toml. Multi-round celery-missing
    stall observed 2026-07-06."""

    def _write_pyproject(self, path, deps_body="") -> None:
        path.write_text(
            "[project]\n"
            'name = "app"\n'
            "dependencies = [\n"
            f"{deps_body}"
            "]\n"
        )

    def test_pyproject_hint_routes_to_pyproject(self, tmp_path):
        # Workspace has BOTH pyproject.toml and requirements.txt. The
        # diag's ``file`` says pyproject, so that's where celery lands.
        self._write_pyproject(tmp_path / "pyproject.toml")
        (tmp_path / "requirements.txt").write_text("gunicorn\n")
        diag = _diag(["celery"])
        diag["file"] = "pyproject.toml"
        block = _try_deps_not_installed(diag, str(tmp_path))
        assert block is not None
        assert block.file == "pyproject.toml"
        assert "celery" in block.replace

    def test_subdir_pyproject_hint_routes_to_subdir(self, tmp_path):
        # Monorepo shape: server/pyproject.toml is the install target.
        (tmp_path / "server").mkdir()
        self._write_pyproject(tmp_path / "server" / "pyproject.toml")
        diag = _diag(["celery"])
        diag["file"] = "server/pyproject.toml"
        block = _try_deps_not_installed(diag, str(tmp_path))
        assert block is not None
        assert block.file == "server/pyproject.toml"
        assert "celery" in block.replace

    def test_subdir_requirements_hint_routes_to_subdir(self, tmp_path):
        (tmp_path / "server").mkdir()
        (tmp_path / "server" / "requirements.txt").write_text("gunicorn\n")
        diag = _diag(["celery"])
        diag["file"] = "server/requirements.txt"
        block = _try_deps_not_installed(diag, str(tmp_path))
        assert block is not None
        assert block.file == "server/requirements.txt"
        assert "celery" in block.replace

    def test_pyproject_hint_falls_through_when_file_absent(self, tmp_path):
        # Hint says pyproject.toml but the file isn't on disk (stale
        # checkpoint, deleted between diag emission and autofix). Fall
        # through to requirements.txt at the workspace root.
        diag = _diag(["celery"])
        diag["file"] = "pyproject.toml"
        block = _try_deps_not_installed(diag, str(tmp_path))
        assert block is not None
        assert block.file == "requirements.txt"

    def test_missing_file_field_defaults_to_root_requirements(self, tmp_path):
        # Legacy diag shape without the ``file`` hint — default to
        # writing a fresh requirements.txt at workspace root.
        diag = _diag(["celery"])
        diag.pop("file", None)
        block = _try_deps_not_installed(diag, str(tmp_path))
        assert block is not None
        assert block.file == "requirements.txt"
