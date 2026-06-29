"""Tests for the prod-import smoke check's workspace-aware install step
composer and the third-party / project-side ModuleNotFoundError
classifier. Reproduces the FinancialResearch monorepo failure mode
(session aa76d684) where prod-smoke ran `pip install pytest` only and
27 third-party imports cascaded into the repair loop."""

import pytest

from harness.graph import (
    _classify_smoke_failure,
    _compose_prod_smoke_install_step,
    _detect_python_manifest,
    _is_critical_config_path,
    _project_top_level_names,
)


class TestComposeProdSmokeInstallStep:
    def test_returns_none_on_empty_workspace(self, tmp_path):
        assert _compose_prod_smoke_install_step(str(tmp_path)) is None

    def test_root_requirements_txt(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("fastapi\n")
        step = _compose_prod_smoke_install_step(str(tmp_path))
        assert step is not None
        # Install lands in the writable /tmp venv, not /usr/local/dist-packages.
        assert "uv venv" in step
        assert "/tmp/teane-venv/bin/activate" in step
        assert "uv pip install -r requirements.txt" in step
        assert "uv pip install pytest" in step
        # `--system` on `uv pip install` would route writes to /usr/local —
        # non-root sandboxes can't write there. (`uv venv --system-site-
        # packages` is fine — it only widens import resolution, not writes.)
        assert "uv pip install --system" not in step

    def test_root_pyproject_takes_precedence_over_requirements(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname="x"\n')
        (tmp_path / "requirements.txt").write_text("fastapi\n")
        step = _compose_prod_smoke_install_step(str(tmp_path))
        assert "uv pip install -e ." in step
        # Root requirements should NOT also be installed when pyproject is present.
        assert "-r requirements.txt" not in step

    def test_monorepo_server_subdir(self, tmp_path):
        """The FinancialResearch failure case: server/requirements.txt with
        no root manifest. Compose step MUST install server/requirements.txt."""
        srv = tmp_path / "server"
        srv.mkdir()
        (srv / "requirements.txt").write_text("fastapi\nsqlalchemy\n")
        (tmp_path / "client").mkdir()
        (tmp_path / "client" / "package.json").write_text("{}")
        step = _compose_prod_smoke_install_step(str(tmp_path))
        assert step is not None
        assert "uv pip install -r server/requirements.txt" in step
        # Skipped subdirs (client/) must not be probed for Python deps.
        assert "client/" not in step

    def test_monorepo_pyproject_subdir(self, tmp_path):
        srv = tmp_path / "backend"
        srv.mkdir()
        (srv / "pyproject.toml").write_text('[project]\nname="b"\n')
        step = _compose_prod_smoke_install_step(str(tmp_path))
        assert "uv pip install -e backend" in step

    def test_includes_dev_requirements_alongside(self, tmp_path):
        srv = tmp_path / "server"
        srv.mkdir()
        (srv / "requirements.txt").write_text("fastapi\n")
        (srv / "requirements-dev.txt").write_text("pytest-asyncio\n")
        step = _compose_prod_smoke_install_step(str(tmp_path))
        assert "server/requirements.txt" in step
        assert "server/requirements-dev.txt" in step

    def test_skips_node_only_subdirs(self, tmp_path):
        # A workspace with only a frontend subdir is not a Python workspace;
        # the composer should return None so the caller skips prod-smoke.
        (tmp_path / "client").mkdir()
        (tmp_path / "client" / "package.json").write_text("{}")
        assert _compose_prod_smoke_install_step(str(tmp_path)) is None


class TestProjectTopLevelNames:
    def test_picks_up_root_dirs_and_modules(self, tmp_path):
        (tmp_path / "server").mkdir()
        (tmp_path / "core").mkdir()
        (tmp_path / "main.py").write_text("")
        (tmp_path / ".hidden").mkdir()
        (tmp_path / "tests").mkdir()  # in skip set
        names = _project_top_level_names(str(tmp_path))
        assert "server" in names
        assert "core" in names
        assert "main" in names
        assert ".hidden" not in names
        assert "tests" not in names


class TestClassifySmokeFailure:
    def test_third_party_module_classified_as_deps_not_installed(self):
        code, hint = _classify_smoke_failure(
            module="server.main",
            exc_type="ModuleNotFoundError",
            message="No module named 'fastapi'",
            project_top_names={"server", "client"},
        )
        assert code == "DEPS_NOT_INSTALLED:fastapi"
        assert "third-party" in hint
        assert "requirements.txt" in hint

    def test_project_module_keeps_smoke_tag(self):
        code, hint = _classify_smoke_failure(
            module="server.main",
            exc_type="ModuleNotFoundError",
            message="No module named 'server.missing_helper'",
            project_top_names={"server"},
        )
        assert code == "PROD_IMPORT_SMOKE:ModuleNotFoundError"
        assert hint == ""

    def test_non_modulenotfound_left_alone(self):
        code, hint = _classify_smoke_failure(
            module="server.main",
            exc_type="SyntaxError",
            message="invalid syntax (server/main.py, line 12)",
            project_top_names={"server"},
        )
        assert code == "PROD_IMPORT_SMOKE:SyntaxError"
        assert hint == ""

    def test_dotted_third_party_uses_top_segment(self):
        code, hint = _classify_smoke_failure(
            module="server.db",
            exc_type="ModuleNotFoundError",
            message="No module named 'sqlalchemy.ext'",
            project_top_names={"server"},
        )
        assert code == "DEPS_NOT_INSTALLED:sqlalchemy"


class TestDetectPythonManifest:
    def test_root_pyproject_wins(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "requirements.txt").write_text("")
        assert _detect_python_manifest(str(tmp_path)) == "pyproject.toml"

    def test_falls_through_to_subdir(self, tmp_path):
        srv = tmp_path / "server"
        srv.mkdir()
        (srv / "requirements.txt").write_text("")
        assert _detect_python_manifest(str(tmp_path)) == "server/requirements.txt"

    def test_default_when_nothing_exists(self, tmp_path):
        assert _detect_python_manifest(str(tmp_path)) == "requirements.txt"


class TestCriticalConfigPath:
    @pytest.mark.parametrize("path", [
        "requirements.txt",
        "server/requirements.txt",
        "pyproject.toml",
        "client/package.json",
        "server/Dockerfile",
        "Makefile",
        ".env",
        "tailwind.config.js",
        "pom.xml",
    ])
    def test_critical_paths_recognised(self, path):
        assert _is_critical_config_path(path) is True

    @pytest.mark.parametrize("path", [
        "server/main.py",
        "client/src/App.tsx",
        "docs/README.md",
        "",
    ])
    def test_non_critical_paths_rejected(self, path):
        assert _is_critical_config_path(path) is False
