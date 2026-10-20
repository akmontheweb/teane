

class TestEditFileAliasAndUnknownMarkers:
    """`edit_file` is the one operation whose tool name and DSL marker differ.

    tool_schemas names it ``edit_file`` and its own description says
    "Equivalent to <<<REPLACE_BLOCK>>> in the text DSL"; every other operation
    (create_file, rewrite_file, delete_block, insert_at_block, insert_at_line,
    replace_line_range) is spelled identically in both vocabularies. So
    EDIT_FILE is the only name a model can carry from the tool surface into DSL
    syntax and have silently discarded.

    lumina 01a15562 resume 3 (2026-10-20): the repair model diagnosed a real
    IDOR — traced test -> resolve_owner -> service.get -> repository.get and
    found a None owner skipping the ownership filter — emitted the correct fix
    as <<<EDIT_FILE>>>, and the patcher parsed zero blocks. The round counted
    as zero-patch, the stall counters climbed, and the resume capped out with
    the fix never reaching disk.
    """

    # Verbatim shape of the dropped patch.
    REAL = (
        "The root cause is confirmed.\n\n"
        "<<<EDIT_FILE>>>\n"
        "file: server/app/services/birthday_service.py\n"
        "search:\n"
        "        record = self._repository.get(record_id, owner_id)\n"
        "replace:\n"
        "        if owner_id is None:\n"
        "            raise NotFoundError()\n"
        "        record = self._repository.get(record_id, owner_id)\n"
        "<<<END_EDIT_FILE>>>\n"
    )

    def test_the_dropped_patch_now_parses(self):
        from harness.patcher import parse_patch_blocks
        blocks = parse_patch_blocks(self.REAL)
        assert len(blocks) == 1

    def test_it_parses_as_replace_block_with_fields_intact(self):
        from harness.patcher import OperationType, parse_patch_blocks
        b = parse_patch_blocks(self.REAL)[0]
        assert b.operation is OperationType.REPLACE_BLOCK
        assert b.file == "server/app/services/birthday_service.py"
        assert "self._repository.get(record_id, owner_id)" in b.search
        assert "if owner_id is None:" in b.replace

    def test_canonical_replace_block_is_unaffected(self):
        from harness.patcher import OperationType, parse_patch_blocks
        canonical = self.REAL.replace("EDIT_FILE", "REPLACE_BLOCK")
        b = parse_patch_blocks(canonical)
        assert len(b) == 1 and b[0].operation is OperationType.REPLACE_BLOCK

    def test_normaliser_leaves_output_without_aliases_untouched(self):
        from harness.patcher import normalize_dsl_markers
        text = "<<<CREATE_FILE>>>\nfile: a.py\ncontent:\nx\n<<<END_CREATE_FILE>>>"
        assert normalize_dsl_markers(text) == text

    def test_every_other_tool_name_already_matches_its_marker(self):
        """If a second divergence is ever introduced, this fails loudly."""
        from harness import tool_schemas
        from harness.patcher import _DSL_MARKER_ALIASES, _PARSE_MISS_MARKERS
        import re as _re
        names = set(
            _re.findall(r'"name": "([a-z_]+)"', open(tool_schemas.__file__).read())
        )
        patch_ops = {
            n.upper() for n in names
            if n not in ("read_file", "submit_decomposition")
        }
        for op in patch_ops:
            assert op in _PARSE_MISS_MARKERS or op in _DSL_MARKER_ALIASES, (
                f"tool '{op.lower()}' has no DSL marker and no alias — a model "
                "using that name in text form would be silently dropped"
            )

    def test_unknown_marker_is_reported_not_swallowed(self):
        from harness.patcher import summarize_parse_miss
        msg = summarize_parse_miss(
            "<<<APPLY_PATCH>>>\nfile: x\nsearch:\na\nreplace:\nb\n"
            "<<<END_APPLY_PATCH>>>"
        )
        assert msg, "an unrecognised marker must not look like an empty round"
        assert "APPLY_PATCH" in msg
        assert "Nothing was applied" in msg

    def test_no_markers_at_all_still_returns_empty(self):
        # A genuinely patch-free response must stay distinguishable from a
        # dropped one.
        from harness.patcher import summarize_parse_miss
        assert summarize_parse_miss("I could not determine a fix.") == ""
