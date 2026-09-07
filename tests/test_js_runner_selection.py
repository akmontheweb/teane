"""The JS/TS test runner must be read off the workspace, not assumed.

lumina session 01a079dc. ``_STACK_TEST_COMMANDS`` hardcoded
``npx --no-install jest --silent`` for every JS/TS workspace, and
``_stack_test_command`` took only the stack tag — no workspace path — so it
structurally could not consult package.json. Two consequences, both of which
had to be fixed together for a non-root JS layout to run at all:

  * Wrong runner. The project was a Vitest project: the spec said so in three
    places, ``vite.config.ts`` carried the ``test`` block, ``scripts.test``
    was ``vitest run --coverage``, and every generated test opened with
    ``import { describe, expect, it, vi } from 'vitest'``. The harness
    installed jest anyway — four devDependencies, a jest.config.cjs, a
    jest.setup.ts and a "jest" tsconfig type — and nine of twelve suites
    then failed to load with "Vitest cannot be imported in a CommonJS
    module".

  * Wrong directory. The command ran bare at the workspace root while the
    scaffolding landed in ``client/``. lumina has no root package.json, so
    the run died in 0.61s on "Could not find a config file based on provided
    values: path: /workspace" having executed zero tests. The sandbox parsed
    no diagnostics, the node synthesised a ``<test_runner>`` placeholder from
    the tail, and that placeholder is what routed the session into the eight
    round repair loop.

``harness.cli.detect_node_test_runner`` is now the single source of truth,
shared with the build-command builder so the two cannot drift apart again.
"""

from __future__ import annotations

import json
import os

from harness.cli import _compose_node_build_command, detect_node_test_runner
from harness.test_generation import (
    _ensure_js_test_env,
    _js_package_roots_for_tests,
    _stack_test_command,
)

VITE_CONFIG_WITH_TEST = """\
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/setupTests.ts'],
  },
});
"""

VITE_CONFIG_NO_TEST = """\
import { defineConfig } from 'vite';
export default defineConfig({ plugins: [] });
"""


def _pkg(path, **kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(kw, indent=2) + "\n")


def _lumina_layout(root):
    """The real lumina shape: no root package.json, one client/ package whose
    vitest config lives in vite.config.ts."""
    client = os.path.join(root, "client")
    os.makedirs(os.path.join(client, "src", "pages"), exist_ok=True)
    _pkg(
        os.path.join(client, "package.json"),
        name="birthday-manager-client",
        scripts={"build": "tsc --noEmit && vite build",
                 "test": "vitest run --coverage"},
        devDependencies={"vite": "^5.4.0", "vitest": "^2.0.5"},
    )
    with open(os.path.join(client, "vite.config.ts"), "w") as fh:
        fh.write(VITE_CONFIG_WITH_TEST)
    test_rel = "client/src/pages/DirectoryPage.test.tsx"
    with open(os.path.join(root, test_rel), "w") as fh:
        fh.write("import { describe, it } from 'vitest';\n")
    return [test_rel]


class TestDetectNodeTestRunner:
    def test_vitest_from_dependency(self, tmp_path):
        d = str(tmp_path)
        assert detect_node_test_runner(
            {"devDependencies": {"vitest": "^2.0.5"}}, d,
        ) == ("vitest", False)

    def test_vitest_from_vite_config_test_block(self, tmp_path):
        (tmp_path / "vite.config.ts").write_text(VITE_CONFIG_WITH_TEST)
        # No vitest in deps at all — this is the form lumina used.
        assert detect_node_test_runner({}, str(tmp_path))[0] == "vitest"

    def test_vite_config_without_test_block_is_not_vitest(self, tmp_path):
        (tmp_path / "vite.config.ts").write_text(VITE_CONFIG_NO_TEST)
        assert detect_node_test_runner({}, str(tmp_path))[0] == "none"

    def test_vitest_from_config_file(self, tmp_path):
        (tmp_path / "vitest.config.ts").write_text("export default {};\n")
        assert detect_node_test_runner({}, str(tmp_path))[0] == "vitest"

    def test_vitest_from_test_script(self, tmp_path):
        assert detect_node_test_runner(
            {"scripts": {"test": "vitest run"}}, str(tmp_path),
        ) == ("vitest", True)

    def test_jest_from_dependency_and_config(self, tmp_path):
        assert detect_node_test_runner(
            {"devDependencies": {"jest": "^29"}}, str(tmp_path),
        )[0] == "jest"
        (tmp_path / "jest.config.cjs").write_text("module.exports = {};\n")
        assert detect_node_test_runner({}, str(tmp_path))[0] == "jest"

    def test_jest_from_package_json_key(self, tmp_path):
        assert detect_node_test_runner(
            {"jest": {"testEnvironment": "node"}}, str(tmp_path),
        )[0] == "jest"

    def test_vitest_wins_when_both_present(self, tmp_path):
        """Exactly the state a previous harness run leaves behind: jest
        scaffolded on top of a vitest project. The tests are written against
        vitest, so vitest must win."""
        (tmp_path / "jest.config.cjs").write_text("module.exports = {};\n")
        assert detect_node_test_runner(
            {"devDependencies": {"vitest": "^2.0.5", "jest": "^29.7.0"}},
            str(tmp_path),
        )[0] == "vitest"

    def test_nothing_configured(self, tmp_path):
        assert detect_node_test_runner({}, str(tmp_path)) == ("none", False)

    def test_has_test_script_reported_independently(self, tmp_path):
        assert detect_node_test_runner(
            {"scripts": {"test": "mocha"}}, str(tmp_path),
        ) == ("none", True)


class TestStackTestCommand:
    def test_lumina_gets_scoped_vitest(self, tmp_path):
        root = str(tmp_path)
        tests = _lumina_layout(root)
        cmd = _stack_test_command("typescript", root, tests)
        # Both halves of the fix: right runner, right directory.
        assert cmd == "(cd client && npx --no-install vitest run)"

    def test_root_package_gets_no_cd(self, tmp_path):
        root = str(tmp_path)
        _pkg(os.path.join(root, "package.json"),
             devDependencies={"vitest": "^2.0.5"})
        (tmp_path / "a.test.ts").write_text("")
        cmd = _stack_test_command("typescript", root, ["a.test.ts"])
        assert cmd == "npx --no-install vitest run"

    def test_jest_package_still_gets_jest(self, tmp_path):
        root = str(tmp_path)
        _pkg(os.path.join(root, "package.json"),
             devDependencies={"jest": "^29.7.0"})
        (tmp_path / "a.test.ts").write_text("")
        assert _stack_test_command("typescript", root, ["a.test.ts"]) == (
            "npx --no-install jest --silent"
        )

    def test_unconfigured_package_defaults_to_jest(self, tmp_path):
        """_ensure_js_test_env provisions jest for these, so jest is right."""
        root = str(tmp_path)
        _pkg(os.path.join(root, "package.json"))
        (tmp_path / "a.test.ts").write_text("")
        assert _stack_test_command("typescript", root, ["a.test.ts"]) == (
            "npx --no-install jest --silent"
        )

    def test_unknown_runner_with_test_script_uses_npm_test(self, tmp_path):
        root = str(tmp_path)
        _pkg(os.path.join(root, "package.json"),
             scripts={"test": "mocha --recursive"})
        (tmp_path / "a.test.js").write_text("")
        assert _stack_test_command("javascript", root, ["a.test.js"]) == (
            "npm test"
        )

    def test_multiple_packages_are_chained(self, tmp_path):
        root = str(tmp_path)
        for name, dev in (("web", {"vitest": "^2"}), ("api", {"jest": "^29"})):
            _pkg(os.path.join(root, name, "package.json"), devDependencies=dev)
            with open(os.path.join(root, name, "a.test.ts"), "w") as fh:
                fh.write("")
        cmd = _stack_test_command(
            "typescript", root, ["web/a.test.ts", "api/a.test.ts"],
        )
        assert cmd == (
            "(cd api && npx --no-install jest --silent) && "
            "(cd web && npx --no-install vitest run)"
        )

    def test_no_js_package_falls_back_to_the_static_entry(self, tmp_path):
        # A .test.ts with no package.json anywhere — nothing to scope to.
        (tmp_path / "a.test.ts").write_text("")
        assert _stack_test_command(
            "typescript", str(tmp_path), ["a.test.ts"],
        ) == "npx --no-install jest --silent"

    def test_non_js_stacks_are_untouched(self, tmp_path):
        assert _stack_test_command("java", str(tmp_path), []) == "mvn -q test"
        assert "pytest" in _stack_test_command("python", str(tmp_path), [])

    def test_workspaceless_call_still_works(self):
        """Back-compat for callers that only care about python/java."""
        assert _stack_test_command("java") == "mvn -q test"
        assert _stack_test_command("typescript") == (
            "npx --no-install jest --silent"
        )


class TestPackageRootResolution:
    def test_finds_the_owning_package(self, tmp_path):
        root = str(tmp_path)
        tests = _lumina_layout(root)
        roots = _js_package_roots_for_tests(root, tests)
        assert roots == [os.path.join(root, "client")]

    def test_workspace_root_sorts_first(self, tmp_path):
        root = str(tmp_path)
        _pkg(os.path.join(root, "package.json"))
        _pkg(os.path.join(root, "web", "package.json"))
        for rel in ("a.test.ts", "web/b.test.ts"):
            with open(os.path.join(root, rel), "w") as fh:
                fh.write("")
        roots = _js_package_roots_for_tests(root, ["web/b.test.ts", "a.test.ts"])
        assert roots[0] == os.path.abspath(root)

    def test_non_js_tests_are_ignored(self, tmp_path):
        root = str(tmp_path)
        _pkg(os.path.join(root, "package.json"))
        assert _js_package_roots_for_tests(root, ["tests/test_x.py"]) == []


class TestScaffolderRespectsVitest:
    def test_no_jest_scaffolding_on_a_vitest_package(self, tmp_path):
        root = str(tmp_path)
        tests = _lumina_layout(root)
        changed = _ensure_js_test_env(root, tests)

        client = tmp_path / "client"
        assert not (client / "jest.config.cjs").exists()
        assert not (client / "jest.setup.ts").exists()
        pkg = json.loads((client / "package.json").read_text())
        dev = pkg["devDependencies"]
        for jest_dep in ("jest", "ts-jest", "@types/jest",
                         "jest-environment-jsdom"):
            assert jest_dep not in dev, jest_dep
        # The project's own pins survive untouched.
        assert dev["vitest"] == "^2.0.5"
        assert "jest.config.cjs" not in " ".join(changed)

    def test_runner_agnostic_component_deps_are_still_added(self, tmp_path):
        root = str(tmp_path)
        tests = _lumina_layout(root)
        _ensure_js_test_env(root, tests)
        dev = json.loads(
            (tmp_path / "client" / "package.json").read_text(),
        )["devDependencies"]
        # A generated component test needs these under either runner.
        assert "@testing-library/react" in dev
        assert "@testing-library/jest-dom" in dev
        assert "jsdom" in dev

    def test_vitest_globals_typed_not_jest(self, tmp_path):
        root = str(tmp_path)
        tests = _lumina_layout(root)
        ts_path = tmp_path / "client" / "tsconfig.json"
        ts_path.write_text(json.dumps({"compilerOptions": {"types": []}}))
        _ensure_js_test_env(root, tests)
        types = json.loads(ts_path.read_text())["compilerOptions"]["types"]
        assert types == ["vitest/globals"]

    def test_jest_package_scaffolding_is_unchanged(self, tmp_path):
        """The pre-existing path must keep working exactly as before."""
        root = str(tmp_path)
        os.makedirs(os.path.join(root, "src"))
        _pkg(os.path.join(root, "package.json"), name="app")
        ts_path = tmp_path / "tsconfig.json"
        ts_path.write_text(json.dumps({"compilerOptions": {"types": []}}))
        with open(os.path.join(root, "src", "a.test.tsx"), "w") as fh:
            fh.write("")
        _ensure_js_test_env(root, ["src/a.test.tsx"])

        assert (tmp_path / "jest.config.cjs").exists()
        dev = json.loads((tmp_path / "package.json").read_text())["devDependencies"]
        assert "jest" in dev and "ts-jest" in dev
        types = json.loads(ts_path.read_text())["compilerOptions"]["types"]
        assert types == ["jest"]


class TestBuildCommandSharesTheDetector:
    def test_scripts_test_still_wins(self, tmp_path):
        pkg = os.path.join(str(tmp_path), "package.json")
        _pkg(pkg, scripts={"test": "vitest run"})
        assert _compose_node_build_command(pkg).endswith("npm test")

    def test_vite_config_test_block_now_yields_vitest_run(self, tmp_path):
        """The detector is broader than the old inline ``"vitest" in deps``
        probe, so this shape is no longer sent to the silent no-op tail."""
        (tmp_path / "vite.config.ts").write_text(VITE_CONFIG_WITH_TEST)
        pkg = os.path.join(str(tmp_path), "package.json")
        _pkg(pkg, scripts={"build": "vite build"})
        assert _compose_node_build_command(pkg).endswith("npx vitest run")

    def test_bare_scaffold_still_gets_the_safe_no_op_tail(self, tmp_path):
        pkg = os.path.join(str(tmp_path), "package.json")
        _pkg(pkg, scripts={"build": "vite build"})
        assert _compose_node_build_command(pkg).endswith(
            "npm test --if-present"
        )
