"""Tests for harness/autofix.py — deterministic R1+R2+R3 autofix pass.

Covers:
    - R1 compiler suggestions (Rust / GCC, plus the applicability gate)
    - R2 missing-symbol imports across Python / TypeScript / Rust / Go / Java,
      plus the "exactly one definition" strictness rule
    - R3 security autofixes for Bandit B201/B602, gitleaks, and trivy
      manifest bumps
    - Integration: repair_node short-circuits when autofix clears every
      diagnostic, falls through when ambiguous
    - Integration: security_scan_node reports passed=True when autofix
      resolves every blocking finding
"""

from __future__ import annotations

import json
import os

import pytest

from harness.autofix import (
    apply_autofixes,
    autofix_system_message,
    _try_compiler_suggestion,
    _try_missing_import,
    _try_security_autofix,
)
from harness.patcher import OperationType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


# ---------------------------------------------------------------------------
# R1 — Compiler suggestion dispatcher
# ---------------------------------------------------------------------------

class TestCompilerSuggestion:

    def test_machine_applicable_rust_emits_replace_block(self, tmp_path):
        src = tmp_path / "src" / "main.rs"
        _write(str(src), "fn main() {\n    let x = foo;\n}\n")
        diag = {
            "file": str(src),
            "line": 2,
            "column": 13,
            "error_code": "E0425",
            "message": "cannot find value `foo` in this scope",
            "suggested_fix": {
                "replacement": "bar",
                "span_start_line": 2,
                "span_start_col": 13,
                "span_end_line": 2,
                "span_end_col": 16,
                "applicability": "machine-applicable",
            },
        }
        block = _try_compiler_suggestion(diag, str(tmp_path))
        assert block is not None, "machine-applicable suggestion should produce a patch"
        assert block.operation == OperationType.REPLACE_BLOCK
        assert block.search == "foo"
        assert block.replace == "bar"

    def test_maybe_incorrect_falls_through(self, tmp_path):
        # "maybe-incorrect" suggestions must NOT auto-apply — they need
        # human / LLM judgement. Returning None passes the diagnostic
        # through to the LLM untouched.
        src = tmp_path / "src" / "main.rs"
        _write(str(src), "fn main() { let x = foo; }\n")
        diag = {
            "file": str(src),
            "line": 1,
            "column": 21,
            "error_code": "E0425",
            "message": "cannot find value `foo`",
            "suggested_fix": {
                "replacement": "bar",
                "span_start_line": 1,
                "span_start_col": 21,
                "span_end_line": 1,
                "span_end_col": 24,
                "applicability": "maybe-incorrect",
            },
        }
        assert _try_compiler_suggestion(diag, str(tmp_path)) is None

    def test_unspecified_falls_through(self, tmp_path):
        src = tmp_path / "x.c"
        _write(str(src), "int x = y;\n")
        diag = {
            "file": str(src),
            "line": 1,
            "suggested_fix": {
                "replacement": "z",
                "span_start_line": 1,
                "span_start_col": 9,
                "span_end_line": 1,
                "span_end_col": 10,
                "applicability": "unspecified",
            },
        }
        assert _try_compiler_suggestion(diag, str(tmp_path)) is None

    def test_gcc_fixit_machine_applicable(self, tmp_path):
        # GCC parser always tags fixits as machine-applicable, so a
        # diagnostic with a populated fixit triggers an autofix.
        src = tmp_path / "x.c"
        _write(str(src), 'int main() { printf("hi"); }\n')
        diag = {
            "file": str(src),
            "line": 1,
            "column": 14,
            "error_code": "-Wimplicit-function-declaration",
            "message": "implicit declaration of function 'printf'",
            "suggested_fix": {
                "replacement": '#include <stdio.h>\n',
                "span_start_line": 1,
                "span_start_col": 1,
                "span_end_line": 1,
                "span_end_col": 1,
                "applicability": "machine-applicable",
            },
        }
        block = _try_compiler_suggestion(diag, str(tmp_path))
        assert block is not None
        assert block.operation == OperationType.REPLACE_BLOCK
        assert block.replace == '#include <stdio.h>\n'


# ---------------------------------------------------------------------------
# R2 — Missing-symbol auto-import
# ---------------------------------------------------------------------------

class TestMissingImportPython:

    def test_exactly_one_definition_emits_import(self, tmp_path):
        _write(str(tmp_path / "foo" / "bar.py"), "def baz():\n    return 1\n")
        offending = tmp_path / "main.py"
        _write(str(offending), "print(baz())\n")
        diag = {
            "file": str(offending),
            "line": 1,
            "error_code": "NameError",
            "message": "NameError: name 'baz' is not defined",
        }
        block = _try_missing_import(diag, str(tmp_path))
        assert block is not None, "single-def workspace should auto-import"
        assert block.operation in (OperationType.INSERT_AT_BLOCK, OperationType.CREATE_FILE)
        if block.operation == OperationType.INSERT_AT_BLOCK:
            assert "from foo.bar import baz" in block.content
        else:
            assert "from foo.bar import baz" in block.content

    def test_two_definitions_skips(self, tmp_path):
        # Ambiguity → LLM territory.
        _write(str(tmp_path / "a.py"), "def baz():\n    return 1\n")
        _write(str(tmp_path / "b.py"), "def baz():\n    return 2\n")
        offending = tmp_path / "main.py"
        _write(str(offending), "print(baz())\n")
        diag = {
            "file": str(offending),
            "line": 1,
            "error_code": "NameError",
            "message": "NameError: name 'baz' is not defined",
        }
        assert _try_missing_import(diag, str(tmp_path)) is None

    def test_zero_definitions_skips(self, tmp_path):
        offending = tmp_path / "main.py"
        _write(str(offending), "print(baz())\n")
        diag = {
            "file": str(offending),
            "line": 1,
            "message": "NameError: name 'baz' is not defined",
        }
        assert _try_missing_import(diag, str(tmp_path)) is None


class TestMissingImportTypeScript:

    def test_ts2304_emits_import(self, tmp_path):
        _write(str(tmp_path / "utils.ts"), "export function helper() { return 1; }\n")
        offending = tmp_path / "main.ts"
        _write(str(offending), "console.log(helper());\n")
        diag = {
            "file": str(offending),
            "line": 1,
            "error_code": "TS2304",
            "message": "Cannot find name 'helper'.",
        }
        block = _try_missing_import(diag, str(tmp_path))
        assert block is not None
        content = block.content
        assert "helper" in content and "from " in content


class TestMissingImportRust:

    def test_e0425_emits_use(self, tmp_path):
        _write(str(tmp_path / "lib.rs"), "pub fn helper() {}\n")
        offending = tmp_path / "main.rs"
        _write(str(offending), "fn main() { helper(); }\n")
        diag = {
            "file": str(offending),
            "line": 1,
            "error_code": "E0425",
            "message": "cannot find function `helper` in this scope",
        }
        block = _try_missing_import(diag, str(tmp_path))
        assert block is not None
        assert "use crate::lib::helper" in block.content


class TestMissingImportGo:

    def test_undefined_emits_import_path(self, tmp_path):
        _write(str(tmp_path / "pkg" / "util" / "u.go"), "package util\n\nfunc Helper() {}\n")
        offending = tmp_path / "main.go"
        _write(str(offending), "package main\n\nfunc main() { Helper() }\n")
        diag = {
            "file": str(offending),
            "line": 3,
            "message": "undefined: Helper",
        }
        block = _try_missing_import(diag, str(tmp_path))
        assert block is not None
        assert 'import "pkg/util"' in block.content


class TestMissingImportJava:

    def test_cannot_find_symbol_emits_import(self, tmp_path):
        _write(
            str(tmp_path / "com" / "ex" / "Helper.java"),
            "package com.ex;\n\npublic class Helper {}\n",
        )
        offending = tmp_path / "Main.java"
        _write(str(offending), "public class Main {}\n")
        diag = {
            "file": str(offending),
            "line": 1,
            "message": "cannot find symbol\n  symbol:   class Helper\n  location: class Main",
        }
        block = _try_missing_import(diag, str(tmp_path))
        assert block is not None
        assert "import com.ex.Helper;" in block.content


# ---------------------------------------------------------------------------
# R3 — Security finding autofixes
# ---------------------------------------------------------------------------

class TestSecurityBanditB201:

    def test_flips_debug_true_to_false(self, tmp_path):
        src = tmp_path / "app.py"
        _write(str(src), "from flask import Flask\napp = Flask(__name__)\napp.run(debug=True)\n")
        diag = {
            "file": str(src),
            "line": 3,
            "error_code": "BANDIT:B201",
            "message": "[SECURITY HIGH] bandit/B201: Flask debug=True",
        }
        block = _try_security_autofix(diag, str(tmp_path))
        assert block is not None
        assert block.operation == OperationType.REPLACE_BLOCK
        assert "debug=False" in block.replace
        assert "debug=True" in block.search


class TestSecurityBanditB602:

    def test_flips_shell_true_when_args_are_list(self, tmp_path):
        src = tmp_path / "app.py"
        _write(
            str(src),
            "import subprocess\nsubprocess.run(['ls', '-l'], shell=True)\n",
        )
        diag = {
            "file": str(src),
            "line": 2,
            "error_code": "BANDIT:B602",
            "message": "[SECURITY HIGH] bandit/B602: subprocess with shell=True",
        }
        block = _try_security_autofix(diag, str(tmp_path))
        assert block is not None
        assert "shell=False" in block.replace

    def test_skips_string_args(self, tmp_path):
        # When args are a string, flipping shell=True would break behaviour
        # — autofix MUST stand down.
        src = tmp_path / "app.py"
        _write(
            str(src),
            "import subprocess\nsubprocess.run('ls -l', shell=True)\n",
        )
        diag = {
            "file": str(src),
            "line": 2,
            "error_code": "BANDIT:B602",
            "message": "[SECURITY HIGH] bandit/B602: subprocess shell=True",
        }
        assert _try_security_autofix(diag, str(tmp_path)) is None


class TestSecurityGitleaks:

    def test_deletes_offending_line(self, tmp_path):
        src = tmp_path / "config.py"
        _write(
            str(src),
            "DEBUG = True\nAWS_KEY = 'AKIAIOSFODNN7EXAMPLE'\nFOO = 1\n",
        )
        diag = {
            "file": str(src),
            "line": 2,
            "error_code": "GITLEAKS:aws-access-key",
            "message": "[SECURITY HIGH] gitleaks/aws-access-key",
        }
        block = _try_security_autofix(diag, str(tmp_path))
        assert block is not None
        assert block.operation == OperationType.DELETE_BLOCK
        assert "AKIAIOSFODNN7EXAMPLE" in block.search


class TestSecurityTrivy:

    def test_bumps_package_json_pin(self, tmp_path):
        manifest = tmp_path / "package.json"
        manifest_text = json.dumps({
            "name": "x",
            "dependencies": {"lodash": "4.17.20"},
        }, indent=2)
        _write(str(manifest), manifest_text + "\n")
        diag = {
            "file": str(manifest),
            "line": 0,
            "error_code": "TRIVY:CVE-2021-23337",
            "message": "lodash 4.17.20: Command injection. Fix available: upgrade to 4.17.21.",
        }
        block = _try_security_autofix(diag, str(tmp_path))
        assert block is not None
        assert block.operation == OperationType.REPLACE_BLOCK
        assert "4.17.20" in block.search
        assert "4.17.21" in block.replace

    def test_skips_when_no_fixed_version(self, tmp_path):
        manifest = tmp_path / "package.json"
        _write(str(manifest), json.dumps({"dependencies": {"x": "1.0.0"}}) + "\n")
        diag = {
            "file": str(manifest),
            "line": 0,
            "error_code": "TRIVY:CVE-X",
            "message": "x 1.0.0: vuln. No fix released — dependency-vuln may require workaround.",
        }
        assert _try_security_autofix(diag, str(tmp_path)) is None


# ---------------------------------------------------------------------------
# apply_autofixes — orchestrator
# ---------------------------------------------------------------------------

class TestApplyAutofixes:

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self, tmp_path):
        unhandled, applied = await apply_autofixes([], str(tmp_path))
        assert unhandled == []
        assert applied == []

    @pytest.mark.asyncio
    async def test_handles_only_resolvable_diagnostic(self, tmp_path):
        # A bandit B201 that the autofixer CAN resolve is removed from
        # the unhandled list and added to the applied list.
        src = tmp_path / "app.py"
        _write(str(src), "app.run(debug=True)\n")
        diag = {
            "file": str(src),
            "line": 1,
            "error_code": "BANDIT:B201",
            "message": "Flask debug=True",
        }
        unhandled, applied = await apply_autofixes([diag], str(tmp_path))
        assert unhandled == []
        assert len(applied) == 1
        assert applied[0].fix_kind == "security"
        with open(src) as fh:
            text = fh.read()
        assert "debug=False" in text

    @pytest.mark.asyncio
    async def test_layer2_insert_at_line_actually_lands(self, tmp_path):
        """Regression test for the 2026-06-25 security-gate HITL loop.

        The Layer-2 YAML rule for ``missing-user`` emits an
        ``INSERT_AT_LINE`` PatchBlock, but ``_apply_block`` never had
        a branch for that op — so the autofix ran, the block was
        produced, and the dispatcher returned
        ``success=False, error="unknown operation"``. Every security
        scan loop ended at the LLM repair stage and eventually
        escalated to HITL even though the deterministic fix was
        ready and correct.

        This test produces a real semgrep-shaped diagnostic for
        ``missing-user``, runs it through ``apply_autofixes``, and
        asserts (a) the autofix is reported applied and (b) the
        ``USER`` directive actually exists in the file on disk.
        """
        dockerfile = tmp_path / "Dockerfile"
        _write(
            str(dockerfile),
            "FROM alpine:3.18\n"
            "COPY app /app\n"
            "RUN apk add --no-cache curl\n"
            "EXPOSE 8080\n"
            "CMD [\"./app\"]\n",
        )
        diag = {
            "file": str(dockerfile),
            "line": 5,
            "error_code": "SEMGREP:dockerfile.security.missing-user.missing-user",
            "message": "Missing USER directive",
        }
        unhandled, applied = await apply_autofixes([diag], str(tmp_path))
        assert unhandled == [], f"Layer-2 patch did not apply: {unhandled}"
        assert len(applied) == 1
        assert applied[0].fix_kind == "security"
        with open(dockerfile) as fh:
            text = fh.read()
        assert "USER 1000:1000" in text
        # And the new USER landed BEFORE the CMD line.
        assert text.index("USER 1000:1000") < text.index("CMD")

    @pytest.mark.asyncio
    async def test_layer1_replace_line_range_actually_lands(self, tmp_path):
        """Mirror of the INSERT_AT_LINE regression test for Layer 1's
        ``REPLACE_LINE_RANGE`` op (scanner-suggested ``extra.fix``).
        Same dispatcher bug, same impact — the patch is produced and
        then silently dropped by ``_apply_block``."""
        target = tmp_path / "Dockerfile"
        _write(
            str(target),
            "FROM python:3.11\n"
            "RUN pip install flask==1.0\n"
            "CMD [\"python\", \"-m\", \"flask\", \"run\"]\n",
        )
        diag = {
            "file": str(target),
            "line": 2,
            "end_line": 2,
            "error_code": "SEMGREP:dockerfile.security.pinned-dep-via-pip-install",
            "message": "Pin Flask to a patched version",
            "fix": "RUN pip install flask==3.0.3",
        }
        _, applied = await apply_autofixes([diag], str(tmp_path))
        assert len(applied) == 1
        with open(target) as fh:
            text = fh.read()
        assert "flask==3.0.3" in text
        assert "flask==1.0" not in text

    @pytest.mark.asyncio
    async def test_mixed_only_fix_resolvable(self, tmp_path):
        # One resolvable + one ambiguous diagnostic → applied has the
        # B201, unhandled has the unknown one.
        src = tmp_path / "app.py"
        _write(str(src), "app.run(debug=True)\n")
        resolvable = {
            "file": str(src),
            "line": 1,
            "error_code": "BANDIT:B201",
            "message": "Flask debug=True",
        }
        ambiguous = {
            "file": "other.py",
            "line": 1,
            "error_code": "BANDIT:B999",
            "message": "[SECURITY MEDIUM] bandit/B999: unknown",
        }
        unhandled, applied = await apply_autofixes(
            [resolvable, ambiguous], str(tmp_path)
        )
        assert len(applied) == 1
        assert len(unhandled) == 1
        assert unhandled[0]["error_code"] == "BANDIT:B999"


# ---------------------------------------------------------------------------
# Integration — repair_node short-circuits when autofix clears the queue
# ---------------------------------------------------------------------------

class TestRepairNodeAutofixShortCircuit:

    @pytest.mark.asyncio
    async def test_skips_llm_when_all_resolved(self, tmp_path):
        from harness import graph

        # A B201 — autofix can resolve this.
        src = tmp_path / "app.py"
        _write(str(src), "app.run(debug=True)\n")
        diag = {
            "file": str(src),
            "line": 1,
            "column": 0,
            "severity": "error",
            "error_code": "BANDIT:B201",
            "message": "[SECURITY HIGH] bandit/B201 in app.py:1: Flask debug=True",
            "semantic_context": "",
        }

        sentinel = {"called": False}

        class StubGateway:
            class config:
                repair_fallback = ""
                planning_fallback = ""

            async def dispatch(self, *args, **kwargs):  # noqa: D401
                sentinel["called"] = True
                raise AssertionError("LLM must not be called when autofix resolves everything")

        graph.set_gateway(StubGateway())
        try:
            result = await graph.repair_node({
                "workspace_path": str(tmp_path),
                "compiler_errors": [diag],
                "loop_counter": {},
                "messages": [],
                "modified_files": [],
                "budget_remaining_usd": 1.0,
            })
        finally:
            graph.set_gateway(None)

        assert sentinel["called"] is False, "LLM must not run when autofix handled everything"
        assert any(
            str(src) in mf or "app.py" in mf for mf in result["modified_files"]
        ), "modified_files should record the autofixed file"
        assert result["node_state"]["repair_success"] == 1
        assert result["node_state"]["repair_fail"] == 0

    @pytest.mark.asyncio
    async def test_calls_llm_when_unhandled_remains(self, tmp_path):
        # Same setup, but with an ambiguous diagnostic the autofixer
        # cannot handle. The LLM MUST be called for the remainder.
        from harness import graph

        src = tmp_path / "app.py"
        _write(str(src), "app.run(debug=True)\n")
        resolvable = {
            "file": str(src),
            "line": 1,
            "column": 0,
            "severity": "error",
            "error_code": "BANDIT:B201",
            "message": "[SECURITY HIGH] bandit/B201 in app.py:1: Flask debug=True",
            "semantic_context": "",
        }
        ambiguous = {
            "file": "unknown.py",
            "line": 0,
            "column": 0,
            "severity": "error",
            "error_code": "BANDIT:B999",
            "message": "[SECURITY MEDIUM] bandit/B999: arbitrary thing",
            "semantic_context": "",
        }

        sentinel = {"called": False}

        class StubResp:
            content = ""

            class usage:
                input_tokens = 0
                output_tokens = 0
                cached_tokens = 0
                cost_usd = 0.0
                model = "stub"

        class StubGateway:
            class config:
                repair_fallback = ""
                planning_fallback = ""

            async def dispatch(self, *, messages, role, budget_remaining_usd, **kwargs):
                sentinel["called"] = True
                sentinel["messages"] = list(messages)
                return StubResp(), budget_remaining_usd

            def aggregate_tokens(self, tracker, usage, role=None):
                return tracker or {}

        graph.set_gateway(StubGateway())
        try:
            await graph.repair_node({
                "workspace_path": str(tmp_path),
                "compiler_errors": [resolvable, ambiguous],
                "loop_counter": {},
                "messages": [],
                "modified_files": [],
                "budget_remaining_usd": 1.0,
            })
        finally:
            graph.set_gateway(None)

        assert sentinel["called"] is True, "LLM must be called for the unhandled diagnostic"
        # The autofix system message must appear in the messages sent to LLM
        # so it doesn't try to re-fix B201.
        msgs = sentinel["messages"]
        autofix_sys = [m for m in msgs if m.get("role") == "system" and "[autofix]" in m.get("content", "")]
        assert autofix_sys, "LLM prompt should include the autofix system message"


# ---------------------------------------------------------------------------
# Integration — security_scan_node passes when autofix resolves all blockers
# ---------------------------------------------------------------------------

class TestSecurityScanNodeAutofixShortCircuit:

    @pytest.mark.asyncio
    async def test_passes_when_autofix_resolves_all_blockers(self, tmp_path, monkeypatch):
        from harness import security as sec
        from harness.security import SecurityFinding, ScannerOutcome, ScannerStatus

        src = tmp_path / "app.py"
        _write(str(src), "app.run(debug=True)\n")

        async def bandit_with_b201(*args, **kwargs):
            return ScannerOutcome(
                scanner="bandit", status=ScannerStatus.FOUND,
                findings=[SecurityFinding(
                    scanner="bandit", rule_id="B201", severity="high",
                    file=str(src), line=1,
                    message="Flask debug=True",
                    cwe="CWE-94", confidence="high",
                )],
            )

        async def empty(name):
            return ScannerOutcome(scanner=name, status=ScannerStatus.OK)

        monkeypatch.setattr(sec, "run_gitleaks_scan", lambda *a, **k: empty("gitleaks"))
        monkeypatch.setattr(sec, "run_bandit_scan",   bandit_with_b201)
        monkeypatch.setattr(sec, "run_semgrep_scan",  lambda *a, **k: empty("semgrep"))
        monkeypatch.setattr(sec, "run_trivy_scan",    lambda *a, **k: empty("trivy"))

        result = await sec.security_scan_node({
            "workspace_path": str(tmp_path),
            "security_scan_config": {"enabled": True},
            "messages": [],
            "modified_files": [],
        })

        assert result["node_state"]["security_scan"]["passed"] is True, (
            "autofix landed the B201 — gate should report passed=True"
        )
        assert result["node_state"]["security_scan"]["autofix_applied"] == 1
        with open(src) as fh:
            assert "debug=False" in fh.read()


# ---------------------------------------------------------------------------
# autofix_system_message
# ---------------------------------------------------------------------------

class TestAutofixSystemMessage:

    def test_empty_applied_returns_empty(self):
        assert autofix_system_message([]) == ""

    @pytest.mark.asyncio
    async def test_message_lists_each_fix(self, tmp_path):
        src = tmp_path / "app.py"
        _write(str(src), "app.run(debug=True)\n")
        diag = {
            "file": str(src),
            "line": 1,
            "error_code": "BANDIT:B201",
            "message": "Flask debug=True",
        }
        _, applied = await apply_autofixes([diag], str(tmp_path))
        msg = autofix_system_message(applied)
        assert "[autofix]" in msg
        assert "BANDIT:B201" in msg
        assert "do not re-attempt" in msg.lower()
