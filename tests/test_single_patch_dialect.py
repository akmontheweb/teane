"""One patch dialect per call, chosen by provider capability — never both.

The system prompt used to carry the native-tool bridge AND the full text-DSL
reference at the same time: the bridge said "DO NOT reproduce the `<<<>>>`
markers" while 10,696 chars of `## Patch Syntax` / `## Edit Invariants`
taught that exact syntax below it. A model shown a syntax uses it, which is
the whole explanation for

  * `<<<EDIT_FILE>>>` emissions the patcher parsed as zero blocks (1e08364),
  * non-canonical `REWRITE_FILE` output needing salvage (5cb41d7),
  * the zero-patch rounds that cascade into the distraction loop.

Native mode now gets the tool list plus the invariants that still apply to a
tool call; the per-operation FIELD semantics live in each tool's JSON-Schema
`description`, where the provider renders them beside the call site. DSL mode
— a provider that cannot do tool use — gets the full text reference and no
tool talk. Tool calls are still folded to DSL text and parsed by the shared
`PatchBlock` backend, so the parser is NOT dead code: it is the convergence
point for both front-ends.
"""

import harness.tool_schemas as ts
from harness.graph import (
    _DSL_PATCH_SYNTAX,
    _NATIVE_EDIT_INVARIANTS,
    _NATIVE_TOOL_MODE_BRIDGE,
    _build_patch_interface_block,
)

#: Patch operations that exist as both a tool and a DSL marker. Control
#: markers with no tool form (READ_FILE is a tool, but PROMOTE_DEFERRED and
#: the web/MCP blocks are text-only by design) are excluded.
_PATCH_OPS = (
    "REPLACE_BLOCK", "CREATE_FILE", "REWRITE_FILE", "DELETE_BLOCK",
    "INSERT_AT_BLOCK", "INSERT_AT_LINE", "REPLACE_LINE_RANGE",
)


class TestExactlyOneDialectIsTaught:
    def test_native_mode_teaches_no_dsl_markers(self):
        block = _build_patch_interface_block(True)
        assert "<<<" not in block

    def test_native_mode_names_no_dsl_operations(self):
        """Even without the markers, naming REPLACE_BLOCK invites the
        syntax — the old reminder said "call the tool of the same meaning"
        precisely because the prompt still named blocks."""
        block = _build_patch_interface_block(True)
        for op in _PATCH_OPS:
            assert op not in block, f"native prompt still names {op}"

    def test_native_mode_names_the_tools(self):
        block = _build_patch_interface_block(True)
        for tool in ("create_file", "edit_file", "rewrite_file",
                     "delete_block", "insert_at_block", "insert_at_line",
                     "replace_line_range", "read_file"):
            assert f"`{tool}`" in block, f"native prompt omits {tool}"

    def test_dsl_mode_keeps_every_operation_it_documented(self):
        block = _build_patch_interface_block(False)
        # NOTE: the DSL reference documents 5 of the 7 operations — it never
        # taught INSERT_AT_LINE or REPLACE_LINE_RANGE, even though the parser
        # accepts both and the native tool list offers them. That gap predates
        # this change; asserted as-is so the split is provably lossless rather
        # than quietly "fixed" here.
        for op in ("REPLACE_BLOCK", "CREATE_FILE", "REWRITE_FILE",
                   "DELETE_BLOCK", "INSERT_AT_BLOCK"):
            assert f"<<<{op}>>>" in block, f"DSL prompt lost {op}"
        assert "<<<READ_FILE>>>" in block
        assert "Edit Invariants" in block

    def test_dsl_mode_does_not_offer_tools(self):
        """A provider that cannot call tools must not be told to."""
        block = _build_patch_interface_block(False)
        assert "NATIVE TOOL-USE" not in block
        assert "CALLING a tool" not in block

    def test_the_two_modes_are_disjoint_not_additive(self):
        native = _build_patch_interface_block(True)
        dsl = _build_patch_interface_block(False)
        assert native != dsl
        # The old bug was concatenation: bridge + DSL in one prompt.
        assert _NATIVE_TOOL_MODE_BRIDGE not in dsl
        assert _DSL_PATCH_SYNTAX not in native

    def test_dsl_text_is_preserved_byte_for_byte(self):
        """A DSL-mode run must behave exactly as it did before the split."""
        assert _DSL_PATCH_SYNTAX.lstrip("\n").startswith("## Edit Invariants")
        assert "## Patch Syntax" in _DSL_PATCH_SYNTAX
        # Size guard: the extracted reference was 10,696 chars.
        assert len(_DSL_PATCH_SYNTAX) > 10_000


class TestNativeInvariantsKeepTheSemantics:
    """The field FORMATS moved to the schemas; the RULES stay in the prompt
    because they are semantics, not syntax."""

    def test_exact_byte_matching_survives(self):
        assert "EXACT-BYTE" in _NATIVE_EDIT_INVARIANTS

    def test_line_number_prefix_rule_survives(self):
        assert "N| " in _NATIVE_EDIT_INVARIANTS

    def test_read_before_edit_survives(self):
        low = _NATIVE_EDIT_INVARIANTS.lower()
        assert "read before you edit" in low
        assert "read_file" in _NATIVE_EDIT_INVARIANTS

    def test_count_rule_survives_without_dsl_syntax(self):
        assert "count" in _NATIVE_EDIT_INVARIANTS
        assert "count:" not in _NATIVE_EDIT_INVARIANTS.replace('count: "', "")

    def test_rewrite_is_framed_as_an_escape_hatch(self):
        assert "escape hatch" in _NATIVE_EDIT_INVARIANTS

    def test_it_is_far_smaller_than_the_dsl_reference(self):
        assert len(_NATIVE_EDIT_INVARIANTS) < len(_DSL_PATCH_SYNTAX) / 3


class TestToolSchemasDoNotTeachTheDsl:
    """The schemas are model-facing: a description saying "Equivalent to
    <<<REPLACE_BLOCK>>> in the text DSL" teaches the DSL at the call site,
    which defeats removing it from the prompt."""

    def _schemas(self):
        out = []
        for name in dir(ts):
            if not name.endswith("_SCHEMA"):
                continue
            val = getattr(ts, name)
            if isinstance(val, dict) and val.get("name"):
                out.append(val)
        assert out, "no schemas discovered"
        return out

    def test_no_schema_mentions_dsl_markers(self):
        for s in self._schemas():
            assert "<<<" not in repr(s), f"{s['name']} still shows a marker"

    def test_no_schema_cross_references_the_text_dsl(self):
        for s in self._schemas():
            assert "text DSL" not in repr(s), f"{s['name']} cites the DSL"

    def test_edit_file_carries_the_invariants_itself(self):
        desc = ts.EDIT_FILE_SCHEMA["description"]
        # The rules that used to be reachable only via the prompt's DSL block.
        assert "verbatim substring" in desc
        assert "N| " in desc
        assert "read_file" in desc

    def test_edit_file_still_documents_count(self):
        props = ts.EDIT_FILE_SCHEMA["input_schema"]["properties"]
        assert set(props["count"]["enum"]) == {"unique", "all", "first"}

    def test_rewrite_file_still_warns_about_partial_content(self):
        desc = ts.REWRITE_FILE_SCHEMA["description"]
        assert "COMPLETE" in desc
        assert "escape hatch" in desc


class TestTheSharedBackendIsStillReachable:
    """Tool calls are rendered to DSL and parsed by the same backend, so the
    parser is the convergence point for both dialects — not dead code."""

    def test_tool_calls_still_render_to_dsl(self):
        dsl = ts.tool_call_to_dsl({
            "name": "edit_file",
            "input": {
                "file_path": "a/b.py",
                "old_string": "x = 1",
                "new_string": "x = 2",
            },
        })
        assert dsl and "<<<REPLACE_BLOCK>>>" in dsl
        assert "file: a/b.py" in dsl

    def test_rendered_dsl_parses_back_into_a_patch_block(self):
        from harness.patcher import parse_patch_blocks

        dsl = ts.tool_call_to_dsl({
            "name": "edit_file",
            "input": {
                "file_path": "a/b.py",
                "old_string": "x = 1",
                "new_string": "x = 2",
            },
        })
        blocks = parse_patch_blocks(dsl)
        assert len(blocks) == 1
        assert blocks[0].file == "a/b.py"
