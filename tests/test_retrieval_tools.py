"""Read-only repository retrieval tools (harness/retrieval_tools.py).

These are the agentic-navigation tools the native tool loop exposes alongside
read_file: grep / glob / list_dir / find_symbol / file_outline /
semantic_search / git_blame / git_log. The tests exercise the deterministic
resolvers against a scratch workspace, the safety guards (path escape, empty
input), and the best-effort "unavailable" degradation for the tools whose
backend (LSP pool / semantic index) isn't running in a unit test.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess

import pytest

from harness.retrieval_tools import (
    RETRIEVAL_TOOLS,
    RETRIEVAL_TOOL_NAMES,
    RetrievalToolsConfig,
    resolve_retrieval_call,
)
from harness.tool_schemas import to_anthropic_tools, to_openai_tools


def _call(name, inp, workspace, config=None) -> str:
    return asyncio.run(
        resolve_retrieval_call(
            {"name": name, "input": inp, "id": "t"}, workspace, config=config,
        )
    )


@pytest.fixture()
def ws(tmp_path):
    root = tmp_path / "repo"
    (root / "src" / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "app" / "service.py").write_text(
        "def compute_total(items):\n"
        "    return sum(i.price for i in items)\n\n"
        "class OrderService:\n"
        "    def place(self, order):\n"
        "        return compute_total(order.items)\n"
    )
    (root / "src" / "app" / "util.py").write_text(
        "def helper():\n    return compute_total([])\n"
    )
    (root / "tests" / "test_service.py").write_text(
        "def test_compute_total():\n    assert True\n"
    )
    return str(root)


@pytest.fixture()
def git_ws(ws):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t.co",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t.co"}
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True, env=env)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=ws, check=True, env=env)
    return ws


# ---------------------------------------------------------------------------
# Deterministic resolvers
# ---------------------------------------------------------------------------

class TestGrep:
    def test_finds_matches_across_files(self, ws):
        out = _call("grep", {"pattern": "compute_total"}, ws)
        assert "service.py" in out and "util.py" in out
        assert "match(es)" in out

    def test_glob_and_path_scoping(self, ws):
        out = _call("grep", {"pattern": "compute_total", "glob": "*.py", "path": "src"}, ws)
        assert "src/app/service.py" in out
        assert "tests/test_service.py" not in out  # scoped out of src/

    def test_no_matches_is_not_an_error(self, ws):
        out = _call("grep", {"pattern": "zzz_never_here"}, ws)
        assert "No matches" in out
        assert not out.startswith("Error:")

    def test_empty_pattern_rejected(self, ws):
        assert _call("grep", {"pattern": ""}, ws).startswith("Error:")

    def test_path_escape_rejected(self, ws):
        assert "escapes the workspace" in _call(
            "grep", {"pattern": "x", "path": "../../etc"}, ws)


class TestGlobAndListDir:
    def test_glob_lists_files(self, ws):
        out = _call("glob", {"pattern": "**/*.py"}, ws)
        assert "src/app/service.py" in out and "tests/test_service.py" in out

    def test_glob_no_match(self, ws):
        assert "No files match" in _call("glob", {"pattern": "**/*.rs"}, ws)

    def test_list_dir_tree(self, ws):
        out = _call("list_dir", {"path": ".", "depth": 2}, ws)
        assert "src/" in out and "app/" in out and "service.py" in out


class TestGit:
    def test_blame(self, git_ws):
        out = _call("git_blame", {"file_path": "src/app/service.py", "start_line": 1, "end_line": 2}, git_ws)
        assert "compute_total" in out and not out.startswith("Error:")

    def test_log_for_file(self, git_ws):
        out = _call("git_log", {"file_path": "src/app/service.py"}, git_ws)
        assert "init" in out

    def test_log_for_symbol(self, git_ws):
        out = _call("git_log", {"file_path": "src/app/service.py", "symbol": "compute_total"}, git_ws)
        assert not out.startswith("Error:")

    def test_blame_non_git_is_graceful(self, ws):
        out = _call("git_blame", {"file_path": "src/app/service.py"}, ws)
        assert "not a git repository" in out


# ---------------------------------------------------------------------------
# Best-effort degradation (no LSP pool / no semantic index in a unit test)
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_find_symbol_without_pool(self, ws):
        out = _call("find_symbol", {"name": "compute_total"}, ws)
        # Either LSP not importable or no active pool — both are actionable,
        # non-raising, and steer the model to grep.
        assert not out.startswith("[find_symbol")  # no fake success header
        assert "grep" in out or "unavailable" in out

    def test_file_outline_without_pool(self, ws):
        out = _call("file_outline", {"file_path": "src/app/service.py"}, ws)
        assert "read_file" in out or "unavailable" in out

    def test_semantic_search_without_index(self, ws):
        out = _call("semantic_search", {"query": "order total"}, ws)
        assert not out.startswith("Error:")  # missing index is not an error


# ---------------------------------------------------------------------------
# Schema wiring
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_names_and_shape(self):
        names = {t["name"] for t in RETRIEVAL_TOOLS}
        assert names == set(RETRIEVAL_TOOL_NAMES)
        assert names == {
            "grep", "glob", "list_dir", "find_symbol",
            "file_outline", "semantic_search", "git_blame", "git_log",
        }
        for t in RETRIEVAL_TOOLS:
            assert t["name"] and t["description"].strip()
            assert t["input_schema"]["type"] == "object"

    def test_provider_conversions_accept_retrieval_tools(self):
        anth = to_anthropic_tools(RETRIEVAL_TOOLS)
        assert {t["name"] for t in anth} == set(RETRIEVAL_TOOL_NAMES)
        oai = to_openai_tools(RETRIEVAL_TOOLS)
        assert all(t["type"] == "function" for t in oai)
        assert {t["function"]["name"] for t in oai} == set(RETRIEVAL_TOOL_NAMES)

    def test_config_defaults_and_disable(self):
        assert RetrievalToolsConfig.from_config(None).enabled is True
        assert RetrievalToolsConfig.from_config(
            {"retrieval_tools": {"enabled": False}}).enabled is False
        # bad types fall back to defaults, never raise
        assert RetrievalToolsConfig.from_config(
            {"retrieval_tools": {"max_results": "oops"}}).max_results == 80
