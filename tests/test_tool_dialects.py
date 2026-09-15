"""Recognise tool invocations written as TEXT, across every major dialect.

lumina-run5-20260914-2120 terminated on zero_patch_loop because
deepseek-v4-pro asked to read a file in Anthropic's XML syntax and the
harness saw prose. Four rounds lost, one of them spending $0.010 to produce
29 output tokens.

Enumerating dialects shrinks how often that happens but cannot make it
never happen — the next model ships a syntax nobody catalogued. So the
module has three tiers: literal dialects, a permissive JSON tier for the
uncatalogued, and honest reporting of whatever still cannot be translated
so the caller can tell the model rather than mistake it for refusal.
"""

from __future__ import annotations

import pytest

from harness.tool_dialects import (
    extract_tool_invocations,
    strip_reasoning,
    to_read_file_dsl,
    unmapped_invocations,
)

# The literal shapes each family emits, verbatim from their chat templates.
DIALECTS = {
    "anthropic_xml":
        '<invoke name="read_file">\n'
        '<parameter name="file">a/b.py</parameter>\n</invoke>',
    "hermes_qwen":
        '<tool_call>{"name":"read_file","arguments":{"file":"a/b.py"}}</tool_call>',
    "mistral":
        '[TOOL_CALLS] [{"name":"read_file","arguments":{"path":"a/b.py"}}]',
    "llama_python_tag":
        '<|python_tag|>{"name":"read_file","parameters":{"file_path":"a/b.py"}}',
    "deepseek":
        "<｜tool▁call▁begin｜>read_file<｜tool▁sep｜>"
        '{"file":"a/b.py"}<｜tool▁call▁end｜>',
    "nemotron":
        "<function=read_file><parameter=file>a/b.py</parameter></function>",
    "generic_json":
        'I need to look: {"name": "read_file", "arguments": {"file": "a/b.py"}}',
}


@pytest.mark.parametrize("dialect,raw", sorted(DIALECTS.items()))
def test_every_dialect_resolves_to_the_same_read(dialect, raw):
    invs = extract_tool_invocations(raw)
    assert invs, f"{dialect} produced no invocation"
    assert invs[0]["dialect"] == dialect
    assert to_read_file_dsl(invs[0]) == (
        "<<<READ_FILE>>>\nfile: a/b.py\n<<<END_READ_FILE>>>"
    )


@pytest.mark.parametrize("dialect,raw", sorted(DIALECTS.items()))
def test_every_dialect_round_trips_through_the_patcher(dialect, raw):
    """The real contract: the existing parser must see a read, and the
    stripper must remove the foreign syntax so it is not parsed as prose."""
    from harness.patcher import parse_read_blocks, strip_read_blocks
    assert parse_read_blocks(raw) == [("a/b.py", None)]
    assert "a/b.py" not in strip_read_blocks(raw)


class TestArgumentKeyAliases:
    """Llama uses `parameters` where everyone else uses `arguments`. A
    parser keying on one silently misses the other."""

    @pytest.mark.parametrize("key", ["arguments", "parameters", "args", "input"])
    def test_all_argument_keys(self, key):
        raw = '<tool_call>{"name":"read_file","%s":{"file":"a/b.py"}}</tool_call>' % key
        assert to_read_file_dsl(extract_tool_invocations(raw)[0])

    @pytest.mark.parametrize("key", ["file", "path", "file_path", "filename"])
    def test_all_path_aliases(self, key):
        raw = '<tool_call>{"name":"read_file","arguments":{"%s":"a/b.py"}}</tool_call>' % key
        assert "a/b.py" in to_read_file_dsl(extract_tool_invocations(raw)[0])

    def test_stringified_arguments(self):
        """OpenAI-shaped payloads carry arguments as a JSON STRING."""
        raw = (
            '<tool_call>{"name":"read_file",'
            '"arguments":"{\\"file\\": \\"a/b.py\\"}"}</tool_call>'
        )
        assert "a/b.py" in to_read_file_dsl(extract_tool_invocations(raw)[0])


class TestReasoningTags:
    """DeepSeek-R1, Qwen3 and GLM emit <think>...</think>. A call the model
    only CONSIDERED must not be executed."""

    def test_call_inside_reasoning_is_ignored(self):
        raw = '<think>I could {"name":"read_file","arguments":{"file":"x.py"}}</think>'
        assert extract_tool_invocations(raw) == []

    def test_call_after_reasoning_is_honoured(self):
        raw = (
            "<think>let me look at it</think>"
            '<tool_call>{"name":"read_file","arguments":{"file":"a/b.py"}}</tool_call>'
        )
        assert len(extract_tool_invocations(raw)) == 1

    def test_strip_is_case_insensitive_and_multiline(self):
        assert strip_reasoning("<THINK>\na\nb\n</THINK>x").strip() == "x"


class TestNegativeControls:
    """Over-matching is worse than under-matching: a false read wastes a
    round AND injects a file the model never asked for."""

    @pytest.mark.parametrize("raw", [
        "just prose about reading a file",
        '```json\n{"config": {"debug": true}}\n```',
        '{"name": "widget"}',                      # name but no arguments
        '{"arguments": {"file": "a.py"}}',         # arguments but no name
        "def read_file(path):\n    return open(path).read()",
    ])
    def test_no_false_positives(self, raw):
        assert extract_tool_invocations(raw) == []


class TestUnmappedReporting:
    """Tier 3. What cannot be translated must be REPORTED, never guessed at
    — silently turning an unknown call into a file operation would be far
    worse than telling the model its syntax was not understood."""

    def test_unknown_tool_is_reported_not_executed(self):
        raw = '<tool_call>{"name":"run_tests","arguments":{"suite":"all"}}</tool_call>'
        assert to_read_file_dsl(extract_tool_invocations(raw)[0]) is None
        assert len(unmapped_invocations(raw)) == 1

    def test_read_without_a_path_is_reported(self):
        raw = '<tool_call>{"name":"read_file","arguments":{"mode":"r"}}</tool_call>'
        assert unmapped_invocations(raw)

    def test_a_translatable_call_is_not_reported(self):
        assert unmapped_invocations(DIALECTS["hermes_qwen"]) == []

    def test_dialect_is_named_for_the_feedback_message(self):
        """The corrective message tells the model which syntax it used, so
        the dialect label has to survive to the caller."""
        raw = '<invoke name="run_tests"><parameter name="x">1</parameter></invoke>'
        assert unmapped_invocations(raw)[0]["dialect"] == "anthropic_xml"


class TestMultipleAndMixed:

    def test_several_calls_in_one_response(self):
        raw = DIALECTS["hermes_qwen"] + "\n" + DIALECTS["anthropic_xml"]
        assert len(extract_tool_invocations(raw)) == 2

    def test_generic_tier_does_not_double_count_a_known_dialect(self):
        """A well-formed <tool_call> also contains a bare JSON object; it
        must be counted once, not twice."""
        assert len(extract_tool_invocations(DIALECTS["hermes_qwen"])) == 1

    def test_range_is_carried_when_well_formed(self):
        raw = (
            '<tool_call>{"name":"read_file","arguments":'
            '{"file":"a/b.py","range":"10-40"}}</tool_call>'
        )
        assert "range: 10-40" in to_read_file_dsl(extract_tool_invocations(raw)[0])

    def test_malformed_range_is_dropped_not_propagated(self):
        raw = (
            '<tool_call>{"name":"read_file","arguments":'
            '{"file":"a/b.py","range":"the whole thing"}}</tool_call>'
        )
        assert "range:" not in to_read_file_dsl(extract_tool_invocations(raw)[0])

    def test_empty_and_none_are_safe(self):
        assert extract_tool_invocations("") == []
        assert extract_tool_invocations(None) == []  # type: ignore[arg-type]
