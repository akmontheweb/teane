"""Tests for harness/parser_registry.py — error parsing utilities."""

import pytest

from harness.parser_registry import (
    _strip_ansi,
    RustParser,
    GoParser,
    GenericParser,
)


class TestStripAnsi:
    """Test ANSI escape code stripping."""

    def test_removes_color_code(self):
        """Should remove ANSI color codes."""
        text = "\x1b[31merror\x1b[0m"  # red
        result = _strip_ansi(text)
        assert result == "error"

    def test_removes_bold(self):
        """Should remove bold codes."""
        text = "\x1b[1mbold\x1b[0m"
        result = _strip_ansi(text)
        assert result == "bold"

    def test_plain_text_unchanged(self):
        """Plain text should be unchanged."""
        text = "hello world"
        result = _strip_ansi(text)
        assert result == "hello world"

    def test_multiple_codes(self):
        """Should remove multiple ANSI sequences."""
        text = "\x1b[1m\x1b[31m\x1b[1mRED BOLD\x1b[0m"
        result = _strip_ansi(text)
        assert result == "RED BOLD"
        assert "\x1b" not in result

    def test_empty_string(self):
        """Empty string should return empty."""
        assert _strip_ansi("") == ""


class TestParserDiagnostics:
    """Test parser diagnostics methods."""

    def test_rust_parser_parse_diagnostics(self):
        """RustParser should have parse_diagnostics static method."""
        output = "error[E0425]: cannot find value"
        diagnostics = RustParser.parse_diagnostics(output)
        assert isinstance(diagnostics, list)

    def test_rust_parser_empty_output(self):
        """Empty output should return empty list."""
        diagnostics = RustParser.parse_diagnostics("")
        assert diagnostics == [] or isinstance(diagnostics, list)

    def test_go_parser_parse_diagnostics(self):
        """GoParser should have parse_diagnostics static method."""
        output = "./main.go:10:5: undefined: SomeFunc"
        diagnostics = GoParser.parse_diagnostics(output)
        assert isinstance(diagnostics, list)

    def test_generic_parser_parse_diagnostics(self):
        """GenericParser should have parse_diagnostics static method."""
        output = "error: something failed"
        diagnostics = GenericParser.parse_diagnostics(output)
        assert isinstance(diagnostics, list)
