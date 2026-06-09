"""Integration tests for the source-root allowlist enforcement.

When a workspace has a clear source root (`app/`, `src/`, ...), the
patching_node and repair_node must reject CREATE_FILE blocks targeting
workspace root (with the documented allowlist of conventionally-root
files as the only exception). This stops the LLM from accidentally
sprinkling new `.py` modules at workspace root when the project's
existing layout is `app/<module>.py`.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from harness import graph as graph_mod
from harness.graph import _build_patcher_allowlist


# ---------------------------------------------------------------------------
# Stubs — minimal gateway + helpers so patching_node runs without LLM access.
# ---------------------------------------------------------------------------

class _StubUsage:
    input_tokens = 50
    output_tokens = 40
    cached_tokens = 0
    cost_usd = 0.0005
    model = "stub"


class _StubResponse:
    def __init__(self, content: str):
        self.content = content
        self.usage = _StubUsage()


class _StubGateway:
    """Returns a pre-canned patch-block string. Records the dispatch."""

    class config:
        repair_fallback = ""
        planning_fallback = ""

    def __init__(self, content: str):
        self._content = content
        self.dispatched: list[dict[str, Any]] = []

    async def dispatch(self, *, messages, role, budget_remaining_usd, **kwargs):
        self.dispatched.append({"messages": list(messages), "role": role})
        return _StubResponse(self._content), budget_remaining_usd - 0.001

    def aggregate_tokens(self, tracker, usage):
        out = dict(tracker or {})
        out["total_cost_usd"] = out.get("total_cost_usd", 0.0) + float(usage.cost_usd)
        return out


@pytest.fixture
def stub_gateway():
    """Install/uninstall a stub LLM gateway via the graph module's setter."""

    def _set(content: str) -> _StubGateway:
        gw = _StubGateway(content)
        graph_mod.set_gateway(gw)
        return gw

    yield _set
    graph_mod.set_gateway(None)


def _seed_app_workspace(tmp_path) -> None:
    """Create a workspace with a clear `app/` source root."""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "calculator.py").write_text(
        "def divide(a, b):\n    return a // b\n"
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")


# ---------------------------------------------------------------------------
# _build_patcher_allowlist
# ---------------------------------------------------------------------------

class TestBuildPatcherAllowlist:

    def test_returns_none_for_flat_workspace(self, tmp_path):
        # Flat workspace → no source root → no enforcement.
        (tmp_path / "foo.py").write_text("")
        (tmp_path / "bar.py").write_text("")
        assert _build_patcher_allowlist(str(tmp_path)) is None

    def test_returns_allowlist_for_app_workspace(self, tmp_path):
        _seed_app_workspace(tmp_path)
        allowlist = _build_patcher_allowlist(str(tmp_path))
        assert allowlist is not None
        # The source root prefix is the first entry.
        assert "app/" in allowlist
        # Test trees are always included.
        assert "tests/" in allowlist
        # Conventional root files are included.
        assert "pyproject.toml" in allowlist
        assert "conftest.py" in allowlist
        assert "setup.py" in allowlist

    def test_picks_up_requirements_files(self, tmp_path):
        _seed_app_workspace(tmp_path)
        (tmp_path / "requirements.txt").write_text("x==1.0\n")
        (tmp_path / "requirements-dev.txt").write_text("pytest\n")
        allowlist = _build_patcher_allowlist(str(tmp_path))
        assert allowlist is not None
        assert "requirements.txt" in allowlist
        assert "requirements-dev.txt" in allowlist

    def test_requirements_txt_allowed_even_when_absent(self, tmp_path):
        # Greenfield workspaces won't have a requirements.txt yet — the LLM
        # must still be allowed to CREATE one in response to env_misconfig.
        # Pre-fix, the scan loop only included files already on disk, so a
        # CREATE_FILE for `requirements.txt` was rejected with
        # "path not in skill allowlist".
        _seed_app_workspace(tmp_path)
        allowlist = _build_patcher_allowlist(str(tmp_path))
        assert allowlist is not None
        assert "requirements.txt" in allowlist


# ---------------------------------------------------------------------------
# patching_node — out-of-root rejection
# ---------------------------------------------------------------------------

class TestPatchingNodeAllowlist:

    @pytest.mark.asyncio
    async def test_out_of_root_create_file_is_rejected(self, tmp_path, stub_gateway):
        # Workspace has a clear `app/` root. LLM emits a CREATE_FILE block
        # at workspace root (`new_module.py`). The patcher MUST reject it
        # because the allowlist only permits `app/`, `tests/`, and the
        # conventional root files.
        _seed_app_workspace(tmp_path)
        stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: new_module.py\n"
            "content:\n"
            "def hello(): pass\n"
            "<<<END_CREATE_FILE>>>\n"
        )

        from harness.graph import patching_node
        result = await patching_node({
            "workspace_path": str(tmp_path),
            "messages": [{"role": "system", "content": "test"}],
            "modified_files": [],
            "budget_remaining_usd": 1.0,
            "token_tracker": {},
        })

        # The file MUST NOT exist on disk.
        assert not (tmp_path / "new_module.py").exists()
        # And it MUST NOT appear in modified_files.
        assert "new_module.py" not in result.get("modified_files", [])
        # A system breadcrumb explaining the failure is in the messages.
        messages = result.get("messages", [])
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        joined = "\n".join(m.get("content", "") for m in sys_msgs)
        assert "Failed" in joined or "failed" in joined.lower()

    @pytest.mark.asyncio
    async def test_under_root_create_file_is_accepted(self, tmp_path, stub_gateway):
        # Same workspace, but the LLM puts the file under `app/` — patch lands.
        _seed_app_workspace(tmp_path)
        stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: app/new_module.py\n"
            "content:\n"
            "def hello(): return 1\n"
            "<<<END_CREATE_FILE>>>\n"
        )

        from harness.graph import patching_node
        result = await patching_node({
            "workspace_path": str(tmp_path),
            "messages": [{"role": "system", "content": "test"}],
            "modified_files": [],
            "budget_remaining_usd": 1.0,
            "token_tracker": {},
        })

        assert (tmp_path / "app" / "new_module.py").exists()
        assert "app/new_module.py" in result.get("modified_files", [])

    @pytest.mark.asyncio
    async def test_conftest_at_root_is_accepted(self, tmp_path, stub_gateway):
        # conftest.py is in the root-file allowlist — must land at root
        # even with a non-flat workspace.
        _seed_app_workspace(tmp_path)
        stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: conftest.py\n"
            "content:\n"
            "import pytest\n"
            "<<<END_CREATE_FILE>>>\n"
        )

        from harness.graph import patching_node
        result = await patching_node({
            "workspace_path": str(tmp_path),
            "messages": [{"role": "system", "content": "test"}],
            "modified_files": [],
            "budget_remaining_usd": 1.0,
            "token_tracker": {},
        })

        assert (tmp_path / "conftest.py").exists()
        assert "conftest.py" in result.get("modified_files", [])

    @pytest.mark.asyncio
    async def test_flat_workspace_has_no_enforcement(self, tmp_path, stub_gateway):
        # Flat workspace (no clear source root). Out-of-root file MUST land,
        # because allowed_paths is None and the patcher imposes no
        # additional restriction.
        (tmp_path / "foo.py").write_text("def foo(): pass\n")
        (tmp_path / "bar.py").write_text("def bar(): pass\n")

        stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: new_module.py\n"
            "content:\n"
            "def hello(): pass\n"
            "<<<END_CREATE_FILE>>>\n"
        )

        from harness.graph import patching_node
        result = await patching_node({
            "workspace_path": str(tmp_path),
            "messages": [{"role": "system", "content": "test"}],
            "modified_files": [],
            "budget_remaining_usd": 1.0,
            "token_tracker": {},
        })

        # Pre-fix behaviour preserved: file landed.
        assert (tmp_path / "new_module.py").exists()
        assert "new_module.py" in result.get("modified_files", [])


# ---------------------------------------------------------------------------
# System prompt includes the workspace-layout sentence
# ---------------------------------------------------------------------------

class TestSystemPromptLayoutInjection:

    def test_layout_block_present_when_source_root_detected(self, tmp_path):
        _seed_app_workspace(tmp_path)
        from harness.graph import _build_system_prompt
        prompt = _build_system_prompt(str(tmp_path), "make build")
        assert "Workspace Layout" in prompt
        assert "`app/`" in prompt
        # Must mention the conventional exceptions so the LLM doesn't
        # over-constrain.
        assert "pyproject.toml" in prompt
        assert "conftest.py" in prompt

    def test_layout_block_absent_for_flat_workspace(self, tmp_path):
        (tmp_path / "foo.py").write_text("")
        (tmp_path / "bar.py").write_text("")
        from harness.graph import _build_system_prompt
        prompt = _build_system_prompt(str(tmp_path), "make build")
        # No clear root → no layout instruction → no false constraint
        # for a flat-layout workspace.
        assert "Workspace Layout" not in prompt
