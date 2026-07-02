"""Verify parse_patch_blocks recognises the REWRITE_FILE block shape
introduced by fix #4."""

from harness.patcher import OperationType, parse_patch_blocks


def test_rewrite_file_block_parses():
    llm = (
        "<<<REWRITE_FILE>>>\n"
        "file: tests/test_bar.py\n"
        "content:\n"
        "def test_ok():\n"
        "    assert 1 == 1\n"
        "<<<END_REWRITE_FILE>>>\n"
    )
    blocks = parse_patch_blocks(llm)
    assert len(blocks) == 1
    b = blocks[0]
    assert b.operation is OperationType.REWRITE_FILE
    assert b.file == "tests/test_bar.py"
    assert "def test_ok" in b.content
    assert "assert 1 == 1" in b.content


def test_rewrite_file_boundary_isolates_from_create_file():
    # Two adjacent blocks — CREATE_FILE followed by REWRITE_FILE. The
    # tempered regex must not swallow the boundary between them.
    llm = (
        "<<<CREATE_FILE>>>\n"
        "file: a.py\n"
        "content:\n"
        "x = 1\n"
        "<<<END_CREATE_FILE>>>\n"
        "<<<REWRITE_FILE>>>\n"
        "file: b.py\n"
        "content:\n"
        "y = 2\n"
        "<<<END_REWRITE_FILE>>>\n"
    )
    blocks = parse_patch_blocks(llm)
    ops = sorted(b.operation.value for b in blocks)
    assert ops == ["create_file", "rewrite_file"]
    by_file = {b.file: b for b in blocks}
    assert by_file["a.py"].content.strip() == "x = 1"
    assert by_file["b.py"].content.strip() == "y = 2"


def test_rewrite_file_can_contain_dsl_looking_text_in_content():
    # As with CREATE_FILE, REWRITE_FILE content is captured verbatim
    # up to the matching END marker. Content that *looks* like another
    # DSL keyword (e.g. a Python string mentioning REPLACE_BLOCK) must
    # not confuse the parser.
    llm = (
        "<<<REWRITE_FILE>>>\n"
        "file: doc.py\n"
        "content:\n"
        "TEXT = 'used REPLACE_BLOCK earlier'\n"
        "<<<END_REWRITE_FILE>>>\n"
    )
    blocks = parse_patch_blocks(llm)
    assert len(blocks) == 1
    assert "REPLACE_BLOCK" in blocks[0].content
