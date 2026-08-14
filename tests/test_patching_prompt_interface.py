"""Prompt defect #3 — the patching system prompt must not teach a text DSL and
then tell the model to use native tools instead, must document the read-only
navigation tools, and must reconcile "build the software" with per-turn story
scoping.
"""

from __future__ import annotations

from pathlib import Path

from harness.graph import _build_system_prompt


def _prompt(tmp_path: Path, use_tools: bool) -> str:
    ws = tmp_path / f"ws-{use_tools}"
    ws.mkdir()
    return _build_system_prompt(
        str(ws), "make test",
        config={"patcher": {"use_structured_tools": use_tools}},
    )


class TestNativeToolBridge:
    def test_tool_mode_injects_bridge(self, tmp_path: Path):
        p = _prompt(tmp_path, True)
        assert "NATIVE TOOL-USE MODE (authoritative)" in p
        # Maps each tool to the DSL operation it replaces.
        assert "= REPLACE_BLOCK" in p and "edit_file" in p
        assert "= CREATE_FILE" in p and "create_file" in p
        # Documents the read-only navigation tools the prompt never named.
        for tool in ("read_file", "list_dir", "glob", "grep", "find_symbol"):
            assert tool in p, tool
        # Tells the model to call tools, not write the <<<>>> markers.
        assert "DO NOT reproduce the `<<<>>>` markers" in p

    def test_text_mode_omits_bridge(self, tmp_path: Path):
        p = _prompt(tmp_path, False)
        assert "NATIVE TOOL-USE MODE (authoritative)" not in p

    def test_default_config_is_tool_mode(self, tmp_path: Path):
        ws = tmp_path / "ws-default"
        ws.mkdir()
        # No patcher.use_structured_tools key → defaults to True (gateway default).
        p = _build_system_prompt(str(ws), "make test", config={})
        assert "NATIVE TOOL-USE MODE (authoritative)" in p

    def test_edit_invariants_present_in_both_modes(self, tmp_path: Path):
        # The exact-byte contract still governs edit_file, so it must survive.
        assert "Edit Invariants" in _prompt(tmp_path, True)
        assert "Edit Invariants" in _prompt(tmp_path, False)
