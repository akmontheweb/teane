"""A patch expressed in another dialect's envelope is still a patch.

``edit_file(file_path, old_string, new_string)`` is teane's OWN tool schema
(harness/tool_schemas.py). When a model writes that call in Anthropic's XML
instead of emitting it natively, the name and the arguments are exactly what
the harness asked for — only the wrapper is foreign, and ``tool_dialects``
already recognises it. Recognising it and then discarding it wastes a round
the model did its part in.

lumina-run14-20260923-2352 lost TEN repair rounds that way: deepseek-v4-pro
answered in ``anthropic_xml`` naming ``edit_file``, the detector reported it
every time, nothing consumed it, and the run ended on
persistent_build_failure with the same three selectors failing throughout.
"""

from __future__ import annotations

from harness.tool_dialects import (
    invocations_to_patch_blocks,
    unmapped_invocations,
)
from harness.patcher import OperationType


ANTHROPIC_EDIT = '''I'll fix the directory ordering.

<invoke name="edit_file">
<parameter name="file_path">server/app/repositories/birthday_repository.py</parameter>
<parameter name="old_string">ORDER BY first_name ASC</parameter>
<parameter name="new_string">ORDER BY last_name ASC, first_name ASC</parameter>
</invoke>
'''


class TestTheRun14Round:
    def test_an_anthropic_xml_edit_becomes_a_replace_block(self):
        blocks, untranslated = invocations_to_patch_blocks(
            unmapped_invocations(ANTHROPIC_EDIT))
        assert untranslated == []
        assert len(blocks) == 1
        b = blocks[0]
        assert b.operation is OperationType.REPLACE_BLOCK
        assert b.file == "server/app/repositories/birthday_repository.py"
        assert b.search == "ORDER BY first_name ASC"
        assert b.replace == "ORDER BY last_name ASC, first_name ASC"

    def test_the_arguments_survive_intact(self):
        # A translation that mangles the search string produces a patch that
        # cannot match — worse than reporting the round, because it looks
        # like the model got it wrong.
        blocks, _ = invocations_to_patch_blocks(
            unmapped_invocations(ANTHROPIC_EDIT))
        assert "first_name" in blocks[0].search
        assert blocks[0].search.strip() == blocks[0].search


class TestEveryPatchShapedTool:
    def _one(self, name: str, params: dict) -> object:
        body = "".join(
            f'<parameter name="{k}">{v}</parameter>' for k, v in params.items())
        text = f'<invoke name="{name}">{body}</invoke>'
        blocks, _ = invocations_to_patch_blocks(unmapped_invocations(text))
        return blocks[0] if blocks else None

    def test_create_file(self):
        b = self._one("create_file",
                      {"file_path": "server/app/new.py", "content": "x = 1"})
        assert b.operation is OperationType.CREATE_FILE
        assert b.file == "server/app/new.py" and b.content == "x = 1"

    def test_rewrite_file(self):
        b = self._one("rewrite_file",
                      {"file_path": "a.py", "content": "print(1)"})
        assert b.operation is OperationType.REWRITE_FILE

    def test_delete_block(self):
        b = self._one("delete_block", {"file_path": "a.py", "search": "dead()"})
        assert b.operation is OperationType.DELETE_BLOCK
        assert b.search == "dead()"


class TestWhatMustNotBeTranslated:
    def test_a_read_is_a_host_round_trip_not_a_patch(self):
        # read_file has its own resolver path; turning it into a patch block
        # would apply an edit the model never asked for.
        text = ('<invoke name="read_file">'
                '<parameter name="file_path">a.py</parameter></invoke>')
        blocks, untranslated = invocations_to_patch_blocks(
            # unmapped_invocations already routes read_file away; pass the
            # raw invocation to prove the translator refuses it too.
            [{"name": "read_file", "args": {"file_path": "a.py"},
              "dialect": "anthropic_xml"}])
        assert blocks == [] and len(untranslated) == 1

    def test_an_unknown_tool_is_reported_not_guessed(self):
        blocks, untranslated = invocations_to_patch_blocks(
            [{"name": "run_terminal_cmd", "args": {"command": "rm -rf /"},
              "dialect": "anthropic_xml"}])
        assert blocks == [] and len(untranslated) == 1

    def test_an_edit_with_no_path_cannot_be_applied(self):
        blocks, untranslated = invocations_to_patch_blocks(
            [{"name": "edit_file",
              "args": {"old_string": "a", "new_string": "b"},
              "dialect": "anthropic_xml"}])
        assert blocks == [] and len(untranslated) == 1

    def test_junk_is_tolerated(self):
        assert invocations_to_patch_blocks([]) == ([], [])
        assert invocations_to_patch_blocks(
            [None, "nope", {"name": "edit_file", "args": "not-a-dict"}],
        ) == ([], [{"name": "edit_file", "args": "not-a-dict"}])

    def test_harness_dsl_text_is_not_double_counted(self):
        # A proper DSL block parses upstream; the rescue only runs when
        # nothing parsed, and must find nothing to translate here.
        dsl = ("<<<REPLACE_BLOCK>>>\nfile: a.py\nsearch:\nx\nreplace:\ny\n"
               "<<<END_REPLACE_BLOCK>>>")
        blocks, _ = invocations_to_patch_blocks(unmapped_invocations(dsl))
        assert blocks == []


class TestPipelineParity:
    """Rescued blocks are ordinary PatchBlocks by the time the pipeline sees
    them, so the tamper guard, allowlist and read-before-edit all still
    apply. This pins the shape that guarantees it."""

    def test_a_rescued_block_is_the_same_type_the_parser_produces(self):
        from harness.patcher import parse_patch_blocks
        parsed = parse_patch_blocks(
            "<<<REPLACE_BLOCK>>>\nfile: a.py\nsearch:\nx\nreplace:\ny\n"
            "<<<END_REPLACE_BLOCK>>>")
        rescued, _ = invocations_to_patch_blocks(
            [{"name": "edit_file",
              "args": {"file_path": "a.py", "old_string": "x",
                       "new_string": "y"},
              "dialect": "anthropic_xml"}])
        assert parsed and rescued
        assert type(rescued[0]) is type(parsed[0])
        assert rescued[0].operation == parsed[0].operation
        assert rescued[0].file == parsed[0].file

    def test_a_rescued_test_file_edit_is_still_a_test_file_edit(self):
        # The repair tamper guard keys on the block's path, so a rescued
        # block targeting a test is caught by the same check.
        from harness.graph import _is_test_artifact
        rescued, _ = invocations_to_patch_blocks(
            [{"name": "edit_file",
              "args": {"file_path": "server/tests/test_x.py",
                       "old_string": "a", "new_string": "b"},
              "dialect": "anthropic_xml"}])
        assert _is_test_artifact(rescued[0].file)
