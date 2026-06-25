"""Tests for the semgrep Layer-1 autofix path.

Covers two surfaces:
  - harness/security.py: _parse_semgrep_json preserves extra.fix and
    end.line on the SecurityFinding; _findings_to_diagnostics propagates
    those fields into the diagnostic dict so autofix can read them.
  - harness/autofix.py: _fix_semgrep consumes a diagnostic carrying
    fix metadata and returns a REPLACE_LINE_RANGE PatchBlock pinned to
    the current file's sha256. Falls through to None when the fix is
    absent so the LLM repair loop still runs.
"""
from __future__ import annotations

import json
import os

from harness.autofix import _fix_semgrep
from harness.patcher import OperationType, sha256_file_bytes
from harness.security import (
    SecurityFinding,
    _findings_to_diagnostics,
    _parse_semgrep_json,
)


def _seed(tmp_path, rel: str, body: str) -> str:
    abs_path = os.path.join(str(tmp_path), rel)
    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(body)
    return abs_path


# ---------------------------------------------------------------------------
# _parse_semgrep_json — capture autofix metadata
# ---------------------------------------------------------------------------

def test_parse_semgrep_captures_rendered_fix_and_end_line():
    raw = json.dumps({"results": [{
        "check_id": "python.lang.security.audit.sql-injection",
        "path": "app.py",
        "start": {"line": 42, "col": 1},
        "end": {"line": 44, "col": 30},
        "extra": {
            "message": "Possible SQL injection",
            "severity": "WARNING",
            "rendered_fix": "    cursor.execute(query, (user_id,))",
            "metadata": {"confidence": "HIGH"},
        },
    }]})
    findings = _parse_semgrep_json(raw)
    assert len(findings) == 1
    f = findings[0]
    assert f.scanner == "semgrep"
    assert f.line == 42
    assert f.end_line == 44
    assert "cursor.execute" in f.fix


def test_parse_semgrep_falls_back_to_fix_when_no_rendered_fix():
    # Older semgrep versions / some rule packs only emit extra.fix
    # without the pre-rendered form.
    raw = json.dumps({"results": [{
        "check_id": "x.y",
        "path": "f.py",
        "start": {"line": 1, "col": 1},
        "end": {"line": 1, "col": 5},
        "extra": {
            "message": "...",
            "severity": "INFO",
            "fix": "ok()",
            "metadata": {},
        },
    }]})
    findings = _parse_semgrep_json(raw)
    assert findings[0].fix == "ok()"


def test_parse_semgrep_no_fix_yields_empty_string():
    raw = json.dumps({"results": [{
        "check_id": "x.y",
        "path": "f.py",
        "start": {"line": 1, "col": 1},
        "end": {"line": 1, "col": 5},
        "extra": {"message": "...", "severity": "INFO", "metadata": {}},
    }]})
    findings = _parse_semgrep_json(raw)
    assert findings[0].fix == ""
    assert findings[0].end_line == 1


# ---------------------------------------------------------------------------
# _findings_to_diagnostics — fix metadata flows to diagnostic dict
# ---------------------------------------------------------------------------

def test_findings_to_diagnostics_propagates_fix_and_end_line():
    finding = SecurityFinding(
        scanner="semgrep",
        rule_id="some.rule",
        severity="high",
        file="src/api.py",
        line=10,
        message="fix me",
        end_line=11,
        fix="safe_call()",
    )
    diags = _findings_to_diagnostics([finding])
    assert len(diags) == 1
    d = diags[0]
    assert d["error_code"] == "SEMGREP:some.rule"
    assert d["fix"] == "safe_call()"
    assert d["end_line"] == 11


def test_findings_to_diagnostics_omits_fix_when_none():
    finding = SecurityFinding(
        scanner="semgrep",
        rule_id="absence.rule",
        severity="high",
        file="Dockerfile",
        line=19,
        message="missing-user",
    )
    diags = _findings_to_diagnostics([finding])
    # No fix on the diagnostic — falls through to LLM as before.
    assert "fix" not in diags[0]


# ---------------------------------------------------------------------------
# _fix_semgrep — end-to-end PatchBlock production
# ---------------------------------------------------------------------------

def test_fix_semgrep_emits_replace_line_range_with_hash(tmp_path):
    abs_path = _seed(tmp_path, "app.py",
                     "line1\nline2\nline3\nline4\n")
    expected_hash = sha256_file_bytes(abs_path)
    diag = {
        "error_code": "SEMGREP:python.lang.security.audit.x",
        "file": abs_path,
        "line": 2,
        "end_line": 3,
        "fix": "REPLACED-A\nREPLACED-B",
    }
    patch = _fix_semgrep("python.lang.security.audit.x", diag, str(tmp_path))
    assert patch is not None
    assert patch.operation == OperationType.REPLACE_LINE_RANGE
    assert patch.file == "app.py"
    assert patch.line == 2
    assert patch.end_line == 3
    assert patch.content == "REPLACED-A\nREPLACED-B"
    assert patch.expected_file_hash == expected_hash


def test_fix_semgrep_returns_none_when_no_fix_and_no_layer2_rule(tmp_path):
    # Diagnostic has no scanner-suggested fix AND the rule isn't in
    # Layer 2's YAML table — Layer 1 misses, Layer 2 misses, the
    # diagnostic flows to the LLM repair loop unchanged.
    _seed(tmp_path, "app.py", "import os\nos.system('rm -rf /')\n")
    diag = {
        "error_code": "SEMGREP:python.lang.audit.no-autofix-for-this-one",
        "file": os.path.join(str(tmp_path), "app.py"),
        "line": 2,
        "end_line": 2,
        # No "fix" key, and no matching rule in security_fix_rules.yaml.
    }
    assert _fix_semgrep("no-autofix-for-this-one", diag, str(tmp_path)) is None


def test_fix_semgrep_returns_none_when_file_missing(tmp_path):
    diag = {
        "error_code": "SEMGREP:rule",
        "file": "does-not-exist.py",
        "line": 1,
        "end_line": 1,
        "fix": "ok()",
    }
    assert _fix_semgrep("rule", diag, str(tmp_path)) is None


def test_fix_semgrep_rejects_invalid_range(tmp_path):
    _seed(tmp_path, "x.py", "a\nb\n")
    diag = {
        "error_code": "SEMGREP:rule",
        "file": "x.py",
        "line": 5,
        "end_line": 3,  # end < start
        "fix": "ok()",
    }
    assert _fix_semgrep("rule", diag, str(tmp_path)) is None


def test_fix_semgrep_dispatcher_registered():
    # The security autofix table must route SEMGREP findings to
    # _fix_semgrep — without that the autofix is dark code.
    from harness.autofix import _SECURITY_FIX_TABLE
    assert "SEMGREP" in _SECURITY_FIX_TABLE
    assert _SECURITY_FIX_TABLE["SEMGREP"] is _fix_semgrep
