"""Tests for the ADR-0006 Phase-1 acceptance_node (guards, routing, wiring).

The DB/sandbox-heavy paths are covered by the engine tests
(``test_acceptance_run``) and the story_state tests; here we pin the node's
guard behaviour, routing, diagnostic synthesis, and suite writing.
"""

from __future__ import annotations

import pytest

from harness import acceptance_node as an
from harness import acceptance_run as ar
from harness.acceptance_gen import ALTITUDE_INTEGRATION, CLASS_BACKEND, AcceptanceScenario


def _int_scen(ac: str, name: str) -> AcceptanceScenario:
    return AcceptanceScenario(ac, name, ALTITUDE_INTEGRATION, CLASS_BACKEND,
                              "    assert client.get('/').status_code == 200")


@pytest.mark.asyncio
class TestGuards:
    async def test_disabled_is_passthrough(self):
        out = await an.acceptance_node({"acceptance_config": {"enabled": False}})
        ns = out["node_state"]
        assert ns["skipped"] is True
        assert ns["acceptance_attributable"] is False
        # a pass-through must not touch compiler_errors / loop_counter
        assert "compiler_errors" not in out

    async def test_missing_config_is_passthrough(self):
        out = await an.acceptance_node({})
        assert out["node_state"]["skipped"] is True

    async def test_passthrough_preserves_prior_node_state(self):
        out = await an.acceptance_node({
            "acceptance_config": {"enabled": False},
            "node_state": {"repatched": True, "foo": 1},
        })
        # prior flags survive the merge (spread form)
        assert out["node_state"]["repatched"] is True
        assert out["node_state"]["foo"] == 1


class TestRouting:
    def test_attributable_routes_to_repair(self):
        assert an.route_after_acceptance(
            {"node_state": {"acceptance_attributable": True}}) == "repair_node"

    def test_clean_routes_to_code_review(self):
        assert an.route_after_acceptance(
            {"node_state": {"acceptance_attributable": False}}) == "code_review_node"

    def test_missing_flag_routes_to_code_review(self):
        assert an.route_after_acceptance({}) == "code_review_node"


class TestDiagnostics:
    def test_acceptance_diagnostics_shape(self):
        outs = [ar.ACOutcome("STORY-1.AC-2", ar.STATUS_ATTRIBUTABLE, ALTITUDE_INTEGRATION,
                             "assert 500 == 201")]
        diags = an._acceptance_diagnostics(outs, ["tests/acceptance/test_story_1_acceptance.py"])
        assert len(diags) == 1
        d = diags[0]
        assert d["error_code"] == "ACCEPTANCE_GAP"
        assert d["file"] == "tests/acceptance/test_story_1_acceptance.py"
        assert "STORY-1.AC-2" in d["message"]
        assert "do not weaken the test" in d["message"]

    def test_diagnostics_empty_written_falls_back_path(self):
        diags = an._acceptance_diagnostics(
            [ar.ACOutcome("STORY-1.AC-1", ar.STATUS_ATTRIBUTABLE)], [])
        assert diags[0]["file"] == "tests/acceptance"


class TestWriteSuite:
    def test_writes_conftest_and_per_story_files(self, tmp_path):
        # a discoverable FastAPI factory
        app_dir = tmp_path / "server" / "app"
        app_dir.mkdir(parents=True)
        (app_dir / "main.py").write_text(
            "from fastapi import FastAPI\ndef create_app():\n    return FastAPI()\n"
        )
        scen = [_int_scen("STORY-1.AC-1", "test_a"), _int_scen("STORY-2.AC-1", "test_b")]
        written = an._write_integration_suite(str(tmp_path), "tests/acceptance", scen, {})
        assert len(written) == 2  # one file per story
        conftest = (tmp_path / "tests" / "acceptance" / "conftest.py").read_text()
        assert "from server.app.main import create_app" in conftest
        assert "def client():" in conftest
        # verifies markers present
        body = (tmp_path / "tests" / "acceptance" / "test_story_1_acceptance.py").read_text()
        assert "# @verifies: STORY-1.AC-1" in body

    def test_suite_isolates_db_when_pydantic_settings_present(self, tmp_path):
        app_dir = tmp_path / "server" / "app"
        app_dir.mkdir(parents=True)
        (app_dir / "main.py").write_text(
            "from fastapi import FastAPI\ndef create_app():\n    return FastAPI()\n"
        )
        (app_dir / "config.py").write_text(
            "from pydantic_settings import BaseSettings, SettingsConfigDict\n"
            "class Settings(BaseSettings):\n"
            "    model_config = SettingsConfigDict(env_prefix='LUMINA_')\n"
            "    db_path: str = './data/lumina.db'\n"
        )
        an._write_integration_suite(str(tmp_path), "tests/acceptance",
                                    [_int_scen("STORY-1.AC-1", "test_a")], {})
        conftest = (tmp_path / "tests" / "acceptance" / "conftest.py").read_text()
        assert "os.environ['LUMINA_DB_PATH'] = _db" in conftest

    def test_db_path_env_config_override_wins(self, tmp_path):
        app_dir = tmp_path / "app"
        app_dir.mkdir(parents=True)
        (app_dir / "main.py").write_text(
            "from fastapi import FastAPI\ndef create_app():\n    return FastAPI()\n"
        )
        an._write_integration_suite(str(tmp_path), "tests/acceptance",
                                    [_int_scen("STORY-1.AC-1", "test_a")],
                                    {"db_path_env": "MYAPP_DB"})
        conftest = (tmp_path / "tests" / "acceptance" / "conftest.py").read_text()
        assert "os.environ['MYAPP_DB'] = _db" in conftest

    def test_seed_json_written_when_isolated_and_rows(self, tmp_path):
        app_dir = tmp_path / "server" / "app"
        app_dir.mkdir(parents=True)
        (app_dir / "main.py").write_text(
            "from fastapi import FastAPI\ndef create_app():\n    return FastAPI()\n")
        (app_dir / "config.py").write_text(
            "from pydantic_settings import BaseSettings, SettingsConfigDict\n"
            "class Settings(BaseSettings):\n"
            "    model_config = SettingsConfigDict(env_prefix='LUMINA_')\n"
            "    db_path: str = './data/lumina.db'\n")
        seed = {"tables": {"contacts": [{"first_name": "Ada", "_verifies": "STORY-1.AC-1"}]}}
        an._write_integration_suite(str(tmp_path), "tests/acceptance",
                                    [_int_scen("STORY-1.AC-1", "test_a")], {}, seed=seed)
        acc = tmp_path / "tests" / "acceptance"
        assert (acc / "seed.json").is_file()
        assert "_apply_seed(_db)" in (acc / "conftest.py").read_text()

    def test_seed_json_skipped_without_isolation(self, tmp_path):
        # app discoverable but NO pydantic settings → no DB isolation → no seed.json
        (tmp_path / "main.py").write_text(
            "from fastapi import FastAPI\ndef create_app():\n    return FastAPI()\n")
        seed = {"tables": {"contacts": [{"first_name": "Ada"}]}}
        an._write_integration_suite(str(tmp_path), "tests/acceptance",
                                    [_int_scen("STORY-1.AC-1", "test_a")], {}, seed=seed)
        assert not (tmp_path / "tests" / "acceptance" / "seed.json").exists()

    def test_no_app_discovered_returns_empty(self, tmp_path):
        (tmp_path / "readme.md").write_text("no app here")
        assert an._write_integration_suite(str(tmp_path), "tests/acceptance",
                                           [_int_scen("STORY-1.AC-1", "test_a")], {}) == []


# ---------------------------------------------------------------------------
# Sandbox runner — Fix B provisioning (B1) + collection-error signal (B3)
# ---------------------------------------------------------------------------

class _CaptureExecutor:
    """Fake SandboxExecutor that records the command + network policy and
    returns a preset pytest output."""

    last: dict = {}
    output: str = ""

    def __init__(self, *, workspace_path, allow_network, sandbox_config):
        _CaptureExecutor.last = {
            "allow_network": allow_network,
            "workspace": workspace_path,
        }

    async def run(self, cmd):
        _CaptureExecutor.last["cmd"] = cmd

        class _R:
            full_output = _CaptureExecutor.output
            raw_output = _CaptureExecutor.output

        return _R()


_INSTALL_OK = an._INSTALL_OK_MARKER
_IMPORT_OK = an._IMPORT_OK_MARKER


def _factory_ws(tmp_path) -> str:
    """A workspace whose discovery resolves to server.app.main:create_app."""
    d = tmp_path / "server" / "app"
    d.mkdir(parents=True)
    (d / "main.py").write_text("def create_app():\n    return object()\n")
    (tmp_path / "server" / "__init__.py").write_text("")
    (d / "__init__.py").write_text("")
    return str(tmp_path)


class TestSandboxRunnerProvisioning:
    def _patch(self, monkeypatch, *, install_step, output):
        _CaptureExecutor.output = output
        monkeypatch.setattr("harness.sandbox.SandboxExecutor", _CaptureExecutor)
        monkeypatch.setattr(
            "harness.graph._compose_prod_smoke_install_step",
            lambda ws: install_step,
        )

    def test_install_step_prepended_and_network_forced(self, monkeypatch):
        # B1: the acceptance run installs the app's deps (like the compiler
        # smoke check) and forces network on, so the ephemeral sandbox has
        # fastapi/etc. and the conftest import doesn't collection-error.
        self._patch(
            monkeypatch,
            install_step="uv pip install -r server/requirements.txt",
            output=f"{_INSTALL_OK}\nPASSED tests/acceptance/test_s1.py::test_add",
        )
        runner = an._make_sandbox_runner(
            {"sandbox_config": {}, "allow_network": False}, "tests/acceptance")
        outcomes = runner(["tests/acceptance/test_s1.py"], "/ws")

        cmd = _CaptureExecutor.last["cmd"]
        assert cmd.startswith("uv pip install -r server/requirements.txt && ")
        assert f"echo {_INSTALL_OK}" in cmd
        assert "python -m pytest tests/acceptance" in cmd
        assert _CaptureExecutor.last["allow_network"] is True
        assert len(outcomes) == 1 and outcomes[0].passed

    def test_no_manifest_falls_back_to_bare_pytest(self, monkeypatch):
        # No Python manifest → composer returns None → bare pytest command
        # with the build's own network policy (here False).
        self._patch(
            monkeypatch, install_step=None,
            output="PASSED tests/acceptance/test_s1.py::test_add",
        )
        runner = an._make_sandbox_runner(
            {"sandbox_config": {}, "allow_network": False}, "tests/acceptance")
        runner(["tests/acceptance/test_s1.py"], "/ws")
        cmd = _CaptureExecutor.last["cmd"]
        assert cmd.startswith("python -m pytest tests/acceptance")
        assert "uv pip install" not in cmd
        assert _CaptureExecutor.last["allow_network"] is False

    def test_import_preflight_probes_discovered_entrypoint(self, monkeypatch, tmp_path):
        # B4/B2: the command includes an import preflight that replicates the
        # conftest's app import EXACTLY, gating the (seeded) pytest run.
        ws = _factory_ws(tmp_path)
        self._patch(
            monkeypatch, install_step="uv pip install -e .",
            output=f"{_INSTALL_OK}\n{_IMPORT_OK}\n"
                   "PASSED tests/acceptance/test_s1.py::test_add",
        )
        runner = an._make_sandbox_runner({"sandbox_config": {}}, "tests/acceptance")
        outcomes = runner(["tests/acceptance/test_s1.py"], ws)
        cmd = _CaptureExecutor.last["cmd"]
        assert 'python -c "from server.app.main import create_app"' in cmd
        # preflight sits before pytest so a bad import short-circuits the run.
        assert cmd.index("from server.app.main") < cmd.index("python -m pytest")
        assert len(outcomes) == 1 and outcomes[0].passed

    def test_app_entrypoint_import_failure_raises_with_message(
        self, monkeypatch, tmp_path,
    ):
        # B4: install succeeded (marker present) but the app import preflight
        # did not (no import marker) → precise entrypoint error, not a vague
        # blocked-by-dependency.
        ws = _factory_ws(tmp_path)
        self._patch(
            monkeypatch, install_step="uv pip install -e .",
            output=f"{_INSTALL_OK}\n"
                   "Traceback (most recent call last):\n"
                   "ModuleNotFoundError: No module named 'server.app.db'",
        )
        runner = an._make_sandbox_runner({"sandbox_config": {}}, "tests/acceptance")
        with pytest.raises(ar.AcceptanceCollectionError) as ei:
            runner(["tests/acceptance/test_s1.py"], ws)
        assert "server.app.main:create_app" in str(ei.value)

    def test_install_failure_raises_collection_error(self, monkeypatch, tmp_path):
        # B1: the install step failed and short-circuited before the preflight
        # / pytest (no install marker) → a provisioning collection-error.
        ws = _factory_ws(tmp_path)
        self._patch(
            monkeypatch, install_step="uv pip install -e .",
            output="ERROR: Could not find a version that satisfies fastapi",
        )
        runner = an._make_sandbox_runner({"sandbox_config": {}}, "tests/acceptance")
        with pytest.raises(ar.AcceptanceCollectionError) as ei:
            runner(["tests/acceptance/test_s1.py"], ws)
        assert "install failed" in str(ei.value)

    def test_collection_banner_after_clean_import_raises(self, monkeypatch):
        # B3: install + app import both fine, but pytest still hit a collection
        # banner (a generated test file's own import) → collection-error.
        self._patch(
            monkeypatch, install_step="uv pip install -e .",
            output=f"{_INSTALL_OK}\nERROR collecting tests/acceptance/test_s1.py\n"
                   "ImportError: cannot import name helper",
        )
        # workspace "/ws" → no discovery → import preflight skipped.
        runner = an._make_sandbox_runner({"sandbox_config": {}}, "tests/acceptance")
        with pytest.raises(ar.AcceptanceCollectionError):
            runner(["tests/acceptance/test_s1.py"], "/ws")

    def test_empty_output_raises_collection_error(self, monkeypatch):
        # B3: a run that produced no pytest summary at all is a "suite did not
        # run" case, not a per-test deferral.
        self._patch(monkeypatch, install_step=None, output="")
        runner = an._make_sandbox_runner(
            {"sandbox_config": {}, "allow_network": False}, "tests/acceptance")
        with pytest.raises(ar.AcceptanceCollectionError):
            runner(["tests/acceptance/test_s1.py"], "/ws")
