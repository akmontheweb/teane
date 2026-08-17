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

    def test_role_enforces_code_not_commentary(self, tmp_path: Path):
        """The harness applies change operations, not prose. The Your Role
        section must tell the model to emit changes directly and NOT narrate a
        plan/reasoning — a planning preamble lands zero patches and, on
        reasoning-by-default models, exhausts the output budget before any
        change is emitted (lumina 01a00fdc)."""
        for mode in (True, False):
            p = _prompt(tmp_path, mode)
            assert "does\nNOT read prose" in p or "does NOT read prose" in p
            # It must NOT still instruct the model to plan out loud first.
            assert "Plan the implementation strategy before writing code" not in p
            # And it must name the failure mode it's preventing.
            assert "FAILED turn" in p

    def test_native_tool_list_matches_patch_tools(self, tmp_path: Path):
        """Every tool advertised in PATCH_TOOLS must be enumerated by name in
        the prompt's authoritative native-tool list. Guards against schema↔prompt
        drift like the ``rewrite_file`` gap (session 01a00e33): a tool the model
        is told to use but never offered, or offered but never explained.
        """
        from harness.tool_schemas import PATCH_TOOLS

        p = _prompt(tmp_path, True)
        for tool in PATCH_TOOLS:
            name = tool["name"]
            assert f"`{name}`" in p, (
                f"{name} is advertised in PATCH_TOOLS but not named in the "
                f"native-tool system prompt"
            )

    def test_rewrite_file_is_a_callable_tool_not_just_prose(self, tmp_path: Path):
        """The prompt tells the model to use REWRITE_FILE for whole-file
        overwrites; that only works if ``rewrite_file`` is a real advertised
        tool (native mode forbids emitting the raw ``<<<>>>`` markers).
        """
        from harness.tool_schemas import PATCH_TOOLS, tool_call_to_patch_block

        assert any(t["name"] == "rewrite_file" for t in PATCH_TOOLS), (
            "rewrite_file missing from PATCH_TOOLS"
        )
        # The name-dispatch must translate a rewrite_file call into a patch.
        block = tool_call_to_patch_block(
            {"name": "rewrite_file", "input": {"file_path": "a.py", "content": "x = 1\n"}}
        )
        assert block is not None and block.file == "a.py"

        p = _prompt(tmp_path, True)
        assert "`rewrite_file`" in p and "= REWRITE_FILE" in p
