"""Integration tests for the source-root allowlist enforcement.

When a workspace has a clear source root (`app/`, `src/`, ...), the
patching_node and repair_node must reject CREATE_FILE blocks targeting
workspace root (with the documented allowlist of conventionally-root
files as the only exception). This stops the LLM from accidentally
sprinkling new `.py` modules at workspace root when the project's
existing layout is `app/<module>.py`.
"""

from __future__ import annotations

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

    def aggregate_tokens(self, tracker, usage, role=None):
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


def _seed_node_workspace(tmp_path) -> None:
    """Create a Node workspace with a `src/` source root and JS manifests."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "index.ts").write_text("export const x = 1;\n")
    (tmp_path / "package.json").write_text('{"name":"x","version":"0.1.0"}\n')


# ---------------------------------------------------------------------------
# _build_patcher_allowlist
# ---------------------------------------------------------------------------

class TestBuildPatcherAllowlist:

    def test_returns_conservative_fallback_for_flat_workspace(self, tmp_path):
        # P1.1 closeout: when no source root is detected, the patcher used to
        # see allowlist=None (permissive — anything goes under workspace).
        # We now hand back a conservative fallback covering common layouts
        # so the LLM is still constrained.
        (tmp_path / "foo.py").write_text("")
        (tmp_path / "bar.py").write_text("")
        allowlist = _build_patcher_allowlist(str(tmp_path))
        assert allowlist is not None
        # Must NOT be the permissive None default any more.
        # Must include the common source-layout prefixes.
        for expected in ("src/", "lib/", "app/", "pkg/", "cmd/"):
            assert expected in allowlist
        # Test trees and root manifest files still present.
        assert "tests/" in allowlist
        assert "pyproject.toml" in allowlist

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

    def test_node_root_manifests_in_allowlist(self, tmp_path):
        # Regression: Node workspaces used to fail because the static set
        # was Python-only — patches to package.json / tsconfig.json got
        # rejected at workspace root even though the kitchen-sink builder
        # supports the locked React + TypeScript + TailwindCSS stack. The
        # static set must cover the canonical JS/TS manifests for that stack.
        _seed_node_workspace(tmp_path)
        allowlist = _build_patcher_allowlist(str(tmp_path))
        assert allowlist is not None
        for expected in (
            "package.json", "package-lock.json",
            "tsconfig.json", "tsconfig.base.json",
            ".npmrc", ".nvmrc",
        ):
            assert expected in allowlist, (
                f"{expected} missing — Node patches at workspace root will be rejected"
            )

    def test_node_config_files_picked_up_by_runtime_scan(self, tmp_path):
        # The proliferating *.config.* and dotrc variants are not in the
        # static set — they're caught by the runtime scan. Any file actually
        # on disk that matches must land in the allowlist so the LLM can
        # amend it.
        _seed_node_workspace(tmp_path)
        (tmp_path / "jest.config.cjs").write_text("module.exports = {};\n")
        (tmp_path / "vite.config.ts").write_text("export default {};\n")
        (tmp_path / "tailwind.config.js").write_text("module.exports = {};\n")
        (tmp_path / ".eslintrc.json").write_text("{}\n")
        (tmp_path / ".prettierrc").write_text("{}\n")
        allowlist = _build_patcher_allowlist(str(tmp_path))
        assert allowlist is not None
        for expected in (
            "jest.config.cjs", "vite.config.ts", "tailwind.config.js",
            ".eslintrc.json", ".prettierrc",
        ):
            assert expected in allowlist, (
                f"{expected} missing — runtime scan should have picked it up"
            )

    def test_unseeded_exotic_node_configs_not_added_when_absent(self, tmp_path):
        # The runtime scan must not invent allowlist entries for configs
        # that aren't on disk AND aren't in the static seed set. The
        # static set covers the canonical config filenames for the
        # supported React+TS+Tailwind+Vite stack (jest / vite / tailwind
        # / postcss / vitest / playwright / cypress / next / rollup /
        # webpack / svelte / astro / nuxt configs in all four extension
        # variants), so those DO greenfield-CREATE without needing the
        # runtime scan — session 6177bcec hit the `jest.config.cjs`
        # rejection because the pattern-scan-only path was greenfield-
        # blind. Files that neither match the static seed nor exist on
        # disk still stay out; guard doesn't broaden to arbitrary
        # ``*.config.*`` writes at workspace root.
        _seed_node_workspace(tmp_path)
        allowlist = _build_patcher_allowlist(str(tmp_path))
        assert allowlist is not None
        # Exotic / project-specific tool configs — pattern-matched by
        # ``_is_node_config_file`` but NOT in the static seed. Absent
        # from disk, they must stay out.
        for absent in (
            "mystery-tool.config.js",
            "custom-runner.config.cjs",
            "internal-linter.config.ts",
        ):
            assert absent not in allowlist, (
                f"{absent} snuck in without being on disk — runtime scan "
                f"must remain disk-gated for non-canonical configs"
            )
        # Canonical stack configs, on the other hand, MUST be present in
        # the static seed so greenfield CREATE succeeds (this is the
        # bug 6177bcec exercised).
        for canonical in (
            "jest.config.cjs", "vite.config.ts", "tailwind.config.js",
            "postcss.config.js", "playwright.config.ts",
        ):
            assert canonical in allowlist, (
                f"{canonical} missing from static seed — greenfield "
                f"CREATE_FILE will be rejected on the LLM's first turn"
            )


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
    async def test_conftest_at_root_is_dropped_in_phase1(self, tmp_path, stub_gateway):
        # Fix #49 (two-phase split): patching_node is phase 1 —
        # production code only. conftest.py is test infrastructure and
        # is dropped from the LLM response BEFORE the patcher sees it,
        # so the file does NOT land on disk during this phase. The
        # follow-up test_generation_node creates conftest in phase 2,
        # after prod imports cleanly.
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

        # Phase 1 filter dropped the block — nothing on disk, nothing in
        # modified_files.
        assert not (tmp_path / "conftest.py").exists()
        assert "conftest.py" not in result.get("modified_files", [])

    @pytest.mark.asyncio
    async def test_flat_workspace_conservative_fallback_blocks_root_writes(
        self, tmp_path, stub_gateway,
    ):
        # P1.1 closeout: flat workspace no longer means "anything goes".
        # The conservative fallback restricts writes to common source-layout
        # prefixes, test trees, and root manifest files. A bare module at
        # workspace root (not in src/lib/app/pkg/cmd/ and not in
        # _ROOT_ALLOWLIST_FILES) must now be rejected.
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

        # Conservative fallback in effect: bare-root new_module.py is denied.
        assert not (tmp_path / "new_module.py").exists()
        assert "new_module.py" not in result.get("modified_files", [])

    @pytest.mark.asyncio
    async def test_flat_workspace_conservative_fallback_allows_src_writes(
        self, tmp_path, stub_gateway,
    ):
        # Conservative fallback still permits the conventional source-layout
        # prefixes (src/, lib/, app/, pkg/, cmd/) so the LLM can land a new
        # file under one of those without a detected source root.
        (tmp_path / "foo.py").write_text("def foo(): pass\n")

        stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: src/new_module.py\n"
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

        assert (tmp_path / "src" / "new_module.py").exists()
        assert "src/new_module.py" in result.get("modified_files", [])


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
