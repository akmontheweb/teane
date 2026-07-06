"""Tests for the npm missing-dep autofix (R7) and the related
graph-side detection helpers.
"""
from __future__ import annotations

import json
import os

import pytest

from harness.autofix import (
    _find_package_json,
    _is_npm_dev_package,
    _try_missing_npm_dep,
)
from harness.graph import (
    _is_env_misconfig,
    _NODE_MODULE_MISS_PATTERNS,
    _node_module_exists_in_workspace,
)


# ---------------------------------------------------------------------------
# Pattern matching — wrapped + plain "Cannot find module" variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected_sym", [
    # Plain Node
    ("Error: Cannot find module 'foo'\n", "foo"),
    ("Cannot find module \"@scope/pkg\"", "@scope/pkg"),
    # The exact CIOD HITL session log line:
    (
        "[vite:css] [postcss] Cannot find module '@tailwindcss/forms'\n"
        "Require stack:\n- /workspace/client/tailwind.config.js\n",
        "@tailwindcss/forms",
    ),
    # Webpack
    ("Module not found: Error: Can't resolve 'lodash/throttle'",
     "lodash/throttle"),
])
def test_node_miss_patterns_capture_symbol(raw, expected_sym):
    found = None
    for pattern in _NODE_MODULE_MISS_PATTERNS:
        m = pattern.search(raw)
        if m:
            found = m.group("sym")
            break
    assert found == expected_sym


def test_is_env_misconfig_returns_node_kind_for_vite_postcss():
    raw = (
        "x Build failed in 204ms\n"
        "error during build:\n"
        "[vite:css] [postcss] Cannot find module '@tailwindcss/forms'\n"
    )
    result = _is_env_misconfig(raw)
    assert result == ("@tailwindcss/forms", "node")


def test_is_env_misconfig_skips_relative_paths():
    # Cannot find module './components/Dashboard' is a LINK_BROKEN case,
    # NOT a missing npm package. The regex must reject it.
    raw = "Error: Cannot find module './components/Dashboard'\n"
    assert _is_env_misconfig(raw) is None


def test_node_module_exists_in_workspace_recognises_local(tmp_path):
    os.makedirs(os.path.join(str(tmp_path), "src", "components"))
    with open(os.path.join(str(tmp_path), "src", "components", "Dashboard.jsx"), "w") as f:
        f.write("export default 1;\n")
    assert _node_module_exists_in_workspace("Dashboard", str(tmp_path)) is True


def test_node_module_exists_in_workspace_rejects_scoped_names(tmp_path):
    # Scoped names are always third-party — never matched as local.
    assert _node_module_exists_in_workspace("@tailwindcss/forms", str(tmp_path)) is False


# ---------------------------------------------------------------------------
# Autofix R7 — package.json mutation
# ---------------------------------------------------------------------------

def _seed_pkg_json(tmp_path, body: dict) -> str:
    abs_path = os.path.join(str(tmp_path), "package.json")
    with open(abs_path, "w") as f:
        json.dump(body, f, indent=2)
    return abs_path


def test_try_missing_npm_dep_adds_devdependency(tmp_path):
    _seed_pkg_json(tmp_path, {
        "name": "x", "version": "0.0.0",
        "dependencies": {"react": "^18.0.0"},
    })
    diag = {
        "error_code": "MISSING_DEP",
        "miss_kind": "node",
        "missing_symbol": "@tailwindcss/forms",
    }
    patch = _try_missing_npm_dep(diag, str(tmp_path))
    assert patch is not None
    assert patch.file == "package.json"
    # The replacement body must parse and contain devDependencies entry.
    new = json.loads(patch.replace)
    assert new["devDependencies"]["@tailwindcss/forms"] == "*"
    # Pre-existing deps preserved.
    assert new["dependencies"]["react"] == "^18.0.0"


def test_try_missing_npm_dep_classifies_runtime_dep(tmp_path):
    _seed_pkg_json(tmp_path, {"name": "x", "version": "0.0.0"})
    diag = {
        "error_code": "MISSING_DEP",
        "miss_kind": "node",
        "missing_symbol": "axios",
    }
    patch = _try_missing_npm_dep(diag, str(tmp_path))
    assert patch is not None
    new = json.loads(patch.replace)
    assert new.get("dependencies", {}).get("axios") == "*"
    assert "axios" not in new.get("devDependencies", {})


def test_try_missing_npm_dep_idempotent_when_present(tmp_path):
    _seed_pkg_json(tmp_path, {
        "name": "x",
        "devDependencies": {"@tailwindcss/forms": "^0.5.0"},
    })
    diag = {
        "error_code": "MISSING_DEP",
        "miss_kind": "node",
        "missing_symbol": "@tailwindcss/forms",
    }
    assert _try_missing_npm_dep(diag, str(tmp_path)) is None


def test_try_missing_npm_dep_strips_subpath(tmp_path):
    # "next/router" → install "next"; "@scope/pkg/sub" → "@scope/pkg".
    _seed_pkg_json(tmp_path, {"name": "x"})
    diag = {
        "error_code": "MISSING_DEP",
        "miss_kind": "node",
        "missing_symbol": "next/router",
    }
    patch = _try_missing_npm_dep(diag, str(tmp_path))
    assert patch is not None
    new = json.loads(patch.replace)
    deps = new.get("dependencies") or {}
    assert "next" in deps
    assert "next/router" not in deps


def test_try_missing_npm_dep_creates_package_json_when_missing(tmp_path):
    diag = {
        "error_code": "MISSING_DEP",
        "miss_kind": "node",
        "missing_symbol": "@tailwindcss/forms",
    }
    patch = _try_missing_npm_dep(diag, str(tmp_path))
    assert patch is not None
    assert patch.file == "package.json"
    assert patch.operation.value in ("create_file", "createfile", "CREATE_FILE")
    body = json.loads(patch.content)
    assert body["devDependencies"]["@tailwindcss/forms"] == "*"


def test_try_missing_npm_dep_finds_client_subdir_pkg_json(tmp_path):
    # When package.json lives at client/package.json (CIOD-style layout),
    # autofix must locate it instead of creating a new one at the root.
    os.makedirs(os.path.join(str(tmp_path), "client"))
    pkg_path = os.path.join(str(tmp_path), "client", "package.json")
    with open(pkg_path, "w") as f:
        json.dump({"name": "client"}, f)
    diag = {
        "error_code": "MISSING_DEP",
        "miss_kind": "node",
        "missing_symbol": "@tailwindcss/forms",
    }
    patch = _try_missing_npm_dep(diag, str(tmp_path))
    assert patch is not None
    assert patch.file == os.path.join("client", "package.json")


def test_try_missing_npm_dep_ignores_non_node_kind(tmp_path):
    _seed_pkg_json(tmp_path, {"name": "x"})
    diag = {
        "error_code": "MISSING_DEP",
        "miss_kind": "python",  # NOT node
        "missing_symbol": "fastapi",
    }
    assert _try_missing_npm_dep(diag, str(tmp_path)) is None


def test_try_missing_npm_dep_ignores_other_error_codes(tmp_path):
    _seed_pkg_json(tmp_path, {"name": "x"})
    diag = {
        "error_code": "LINK_BROKEN",
        "miss_kind": "node",
        "missing_symbol": "@tailwindcss/forms",
    }
    assert _try_missing_npm_dep(diag, str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Dev/runtime classification heuristic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, expected", [
    ("@tailwindcss/forms", True),
    ("@types/node", True),
    ("vite", True),
    ("vitest", True),
    ("postcss-import", True),
    ("eslint-plugin-react", True),
    ("tailwindcss", True),
    ("axios", False),
    ("react", False),
    ("@scope/some-runtime", False),
    # Fix J — bundlers / TS runners / formatting plugins / test mocks
    # that were misclassifying as runtime and landing in ``dependencies``.
    ("parcel", True),
    ("tsx", True),
    ("tsup", True),
    ("swc", True),
    ("msw", True),
    ("prettier-plugin-tailwindcss", True),
    # Regression guard: react-adjacent runtime packages must NOT get
    # swept into devDependencies by an accidental overreach of the new
    # entries (`tsx` is a common name that could clash with .tsx files,
    # but the classifier is name-exact — no substring matching).
    ("next", False),
    ("express", False),
    ("@nestjs/core", False),
])
def test_npm_dev_classification(name, expected):
    assert _is_npm_dev_package(name) is expected


def test_find_package_json_root(tmp_path):
    with open(os.path.join(str(tmp_path), "package.json"), "w") as f:
        f.write("{}")
    assert _find_package_json(str(tmp_path)) == "package.json"


def test_find_package_json_returns_none_when_absent(tmp_path):
    assert _find_package_json(str(tmp_path)) is None
