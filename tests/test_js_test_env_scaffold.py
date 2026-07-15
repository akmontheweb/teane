"""Deterministic JS/TS test-environment scaffolding.

A generated ``.test.tsx`` without its jest/type environment produces
hundreds of TS2304/TS2307 type-noise diagnostics and a test run that
cannot collect (session 22471c0c: 456 of them). The TypeScript guide
instructs the LLM to patch the config in the same response;
``_ensure_js_test_env`` is the harness-side guarantee for when it
doesn't — the JS counterpart of ``_ensure_pytest_importlib_config``.
"""

from __future__ import annotations

import json

from harness.test_generation import _ensure_js_test_env


def _write(tmp_path, rel, content):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


class TestComponentTestScaffold:
    def _setup_ws(self, tmp_path):
        _write(tmp_path, "client/package.json", json.dumps({
            "name": "client", "dependencies": {"react": "^18.2.0"},
        }))
        _write(tmp_path, "client/tsconfig.json", json.dumps({
            "compilerOptions": {"strict": True, "types": ["node"]},
        }))
        _write(
            tmp_path, "client/src/__tests__/Panel.test.tsx",
            "import { render } from '@testing-library/react';\n",
        )
        return str(tmp_path)

    def test_full_scaffold_for_tsx_component_test(self, tmp_path):
        ws = self._setup_ws(tmp_path)
        changed = _ensure_js_test_env(
            ws, ["client/src/__tests__/Panel.test.tsx"],
        )
        assert set(changed) == {
            "client/package.json", "client/jest.config.cjs",
            "client/jest.setup.ts", "client/tsconfig.json",
        }
        pkg = json.loads((tmp_path / "client/package.json").read_text())
        dev = pkg["devDependencies"]
        for dep in ("jest", "ts-jest", "@types/jest",
                    "@testing-library/react", "@testing-library/jest-dom",
                    "jest-environment-jsdom"):
            assert dep in dev, f"missing devDependency {dep}"
        # Existing dependencies untouched.
        assert pkg["dependencies"] == {"react": "^18.2.0"}
        cfg = (tmp_path / "client/jest.config.cjs").read_text()
        assert "ts-jest" in cfg
        assert "testEnvironment: 'jsdom'" in cfg
        assert "setupFilesAfterEnv: ['<rootDir>/jest.setup.ts']" in cfg
        assert (tmp_path / "client/jest.setup.ts").read_text() == (
            "import '@testing-library/jest-dom';\n"
        )
        ts = json.loads((tmp_path / "client/tsconfig.json").read_text())
        assert ts["compilerOptions"]["types"] == ["node", "jest"]

    def test_idempotent_second_run_is_a_no_op(self, tmp_path):
        ws = self._setup_ws(tmp_path)
        _ensure_js_test_env(ws, ["client/src/__tests__/Panel.test.tsx"])
        assert _ensure_js_test_env(
            ws, ["client/src/__tests__/Panel.test.tsx"],
        ) == []

    def test_existing_pins_are_never_modified(self, tmp_path):
        ws = self._setup_ws(tmp_path)
        pkg_path = tmp_path / "client/package.json"
        pkg = json.loads(pkg_path.read_text())
        pkg["devDependencies"] = {"jest": "30.0.0"}
        pkg_path.write_text(json.dumps(pkg))
        _ensure_js_test_env(ws, ["client/src/__tests__/Panel.test.tsx"])
        dev = json.loads(pkg_path.read_text())["devDependencies"]
        assert dev["jest"] == "30.0.0"
        assert "ts-jest" in dev  # missing ones still added


class TestExistingConfigRespected:
    def test_existing_jest_config_not_overwritten(self, tmp_path):
        _write(tmp_path, "client/package.json", json.dumps({"name": "c"}))
        _write(tmp_path, "client/jest.config.js",
               "module.exports = { testEnvironment: 'node' };\n")
        _write(tmp_path, "client/src/x.test.tsx", "test('x', () => {});\n")
        changed = _ensure_js_test_env(
            str(tmp_path), ["client/src/x.test.tsx"],
        )
        assert "client/jest.config.cjs" not in changed
        assert not (tmp_path / "client/jest.setup.ts").exists()

    def test_jest_key_in_package_json_counts_as_config(self, tmp_path):
        _write(tmp_path, "client/package.json", json.dumps({
            "name": "c", "jest": {"testEnvironment": "node"},
        }))
        _write(tmp_path, "client/src/x.test.ts", "test('x', () => {});\n")
        changed = _ensure_js_test_env(
            str(tmp_path), ["client/src/x.test.ts"],
        )
        assert "client/jest.config.cjs" not in changed

    def test_tsconfig_without_types_key_left_alone(self, tmp_path):
        # Absent `types` means ALL @types packages auto-include — adding
        # the key would EXCLUDE everything not listed. Must not touch.
        _write(tmp_path, "client/package.json", json.dumps({"name": "c"}))
        _write(tmp_path, "client/tsconfig.json", json.dumps({
            "compilerOptions": {"strict": True},
        }))
        _write(tmp_path, "client/src/x.test.ts", "test('x', () => {});\n")
        changed = _ensure_js_test_env(
            str(tmp_path), ["client/src/x.test.ts"],
        )
        assert "client/tsconfig.json" not in changed
        ts = json.loads((tmp_path / "client/tsconfig.json").read_text())
        assert "types" not in ts["compilerOptions"]

    def test_jsonc_tsconfig_skipped_gracefully(self, tmp_path):
        _write(tmp_path, "client/package.json", json.dumps({"name": "c"}))
        _write(tmp_path, "client/tsconfig.json",
               '{\n  // comment makes this JSONC\n  "compilerOptions": {}\n}\n')
        _write(tmp_path, "client/src/x.test.ts", "test('x', () => {});\n")
        changed = _ensure_js_test_env(
            str(tmp_path), ["client/src/x.test.ts"],
        )
        assert "client/tsconfig.json" not in changed


class TestNonComponentAndEdgeCases:
    def test_plain_js_test_gets_node_env_and_no_ts_deps(self, tmp_path):
        _write(tmp_path, "package.json", json.dumps({"name": "app"}))
        _write(tmp_path, "src/util.test.js", "test('u', () => {});\n")
        changed = _ensure_js_test_env(str(tmp_path), ["src/util.test.js"])
        assert "package.json" in changed
        dev = json.loads((tmp_path / "package.json").read_text())[
            "devDependencies"
        ]
        assert "jest" in dev
        assert "ts-jest" not in dev
        assert "@testing-library/react" not in dev
        cfg = (tmp_path / "jest.config.cjs").read_text()
        assert "testEnvironment: 'node'" in cfg
        assert "setupFilesAfterEnv" not in cfg

    def test_no_package_json_is_a_no_op(self, tmp_path):
        _write(tmp_path, "src/x.test.ts", "test('x', () => {});\n")
        assert _ensure_js_test_env(str(tmp_path), ["src/x.test.ts"]) == []

    def test_non_js_tests_ignored(self, tmp_path):
        _write(tmp_path, "package.json", json.dumps({"name": "app"}))
        assert _ensure_js_test_env(
            str(tmp_path), ["tests/test_x.py"],
        ) == []

    def test_unparseable_package_json_fails_open(self, tmp_path):
        _write(tmp_path, "package.json", "{ not json ")
        _write(tmp_path, "src/x.test.ts", "test('x', () => {});\n")
        assert _ensure_js_test_env(str(tmp_path), ["src/x.test.ts"]) == []
