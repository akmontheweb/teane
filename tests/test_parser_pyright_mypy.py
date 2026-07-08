"""Tests for the pyright / mypy diagnostic parsers in harness/parser_registry.py.

Covers:
    - PyrightJSONParser: 0→1 index shift, information-severity filter,
      multi-line message folding into semantic_context, malformed JSON → []
    - MypyParser: column/code extraction, columnless (older mypy) lines,
      note-folding into the preceding diagnostic, orphan notes dropped
    - Registry: both tools resolve via _PARSER_REGISTRY; a tsc fixture
      still parses through TypeScriptParser with --pretty false output
"""

from __future__ import annotations

import json

from harness.parser_registry import (
    MypyParser,
    PyrightJSONParser,
    TypeScriptParser,
    _PARSER_REGISTRY,
)


# ---------------------------------------------------------------------------
# PyrightJSONParser
# ---------------------------------------------------------------------------

def _pyright_payload(*diags: dict) -> str:
    return json.dumps({"generalDiagnostics": list(diags)})


def test_pyright_index_shift_and_rule():
    out = _pyright_payload({
        "file": "/w/app/models.py",
        "severity": "error",
        "message": 'Type "int" is not assignable to "str"',
        "rule": "reportAssignmentType",
        "range": {"start": {"line": 41, "character": 4},
                  "end": {"line": 41, "character": 9}},
    })
    diags = PyrightJSONParser.parse_diagnostics(out)
    assert len(diags) == 1
    d = diags[0]
    assert d.line == 42 and d.column == 5           # 0-indexed → 1-indexed
    assert d.error_code == "reportAssignmentType"
    assert d.severity == "error"


def test_pyright_information_severity_dropped():
    out = _pyright_payload(
        {"file": "a.py", "severity": "information", "message": "hint",
         "range": {"start": {"line": 0, "character": 0}}},
        {"file": "a.py", "severity": "warning", "message": "unused",
         "range": {"start": {"line": 1, "character": 0}}},
    )
    diags = PyrightJSONParser.parse_diagnostics(out)
    assert [d.severity for d in diags] == ["warning"]


def test_pyright_multiline_message_folds_to_context():
    out = _pyright_payload({
        "file": "a.py", "severity": "error",
        "message": "No overloads match\n  Overload 1: (x: int) -> int\n  Overload 2: (x: str) -> str",
        "range": {"start": {"line": 3, "character": 0}},
    })
    d = PyrightJSONParser.parse_diagnostics(out)[0]
    assert d.message == "No overloads match"
    assert "Overload 1" in d.semantic_context


def test_pyright_missing_rule_defaults_to_pyright():
    out = _pyright_payload({
        "file": "a.py", "severity": "error", "message": "boom",
        "range": {"start": {"line": 0, "character": 0}},
    })
    assert PyrightJSONParser.parse_diagnostics(out)[0].error_code == "pyright"


def test_pyright_malformed_json_fails_open():
    assert PyrightJSONParser.parse_diagnostics("not json at all") == []
    assert PyrightJSONParser.parse_diagnostics("") == []
    assert PyrightJSONParser.parse_diagnostics(json.dumps({"summary": {}})) == []
    assert PyrightJSONParser.parse_diagnostics(json.dumps([1, 2])) == []


# ---------------------------------------------------------------------------
# MypyParser
# ---------------------------------------------------------------------------

def test_mypy_column_code_and_note_folding():
    out = (
        'app/models.py:12:5: error: Incompatible return value type '
        '(got "int", expected "str")  [return-value]\n'
        "app/models.py:12:5: note: Maybe you meant str(x)?\n"
        "app/old.py:3: warning: unused ignore comment  [unused-ignore]\n"
    )
    diags = MypyParser.parse_diagnostics(out)
    assert len(diags) == 2
    first, second = diags
    assert first.line == 12 and first.column == 5
    assert first.error_code == "return-value"
    assert "note: Maybe you meant str(x)?" in first.semantic_context
    assert second.column == 0                       # columnless older-mypy form
    assert second.error_code == "unused-ignore"
    assert second.severity == "warning"


def test_mypy_orphan_note_dropped():
    out = "app/a.py:1:1: note: See https://mypy.readthedocs.io\n"
    assert MypyParser.parse_diagnostics(out) == []


def test_mypy_line_without_code_defaults_to_mypy():
    out = "app/a.py:7:1: error: Name 'x' is not defined\n"
    d = MypyParser.parse_diagnostics(out)[0]
    assert d.error_code == "mypy"


# ---------------------------------------------------------------------------
# Registry / tsc fixture
# ---------------------------------------------------------------------------

def test_registry_contains_new_parsers():
    assert _PARSER_REGISTRY["pyright"] is PyrightJSONParser
    assert _PARSER_REGISTRY["mypy"] is MypyParser


def test_tsc_pretty_false_fixture_parses():
    # Exact shape of `tsc --noEmit --pretty false` output.
    out = (
        "src/components/App.tsx(17,23): error TS2339: "
        "Property 'userName' does not exist on type 'Props'.\n"
    )
    diags = TypeScriptParser.parse_diagnostics(out)
    assert len(diags) == 1
    assert diags[0].error_code == "TS2339"
    assert diags[0].line == 17 and diags[0].column == 23
