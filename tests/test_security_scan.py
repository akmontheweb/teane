"""Tests for the deterministic security scanner gate (harness/security.py).

Covers:
    - SecurityFinding dataclass round-trip
    - SecurityScanPolicy load from config + defaults
    - apply_policy: block / warn / ignore / allowlist / cap / dedup
    - JSON parsers for gitleaks, bandit, semgrep, trivy
    - ScannerOutcome status distinguishes ok / found / crashed / timeout / not_installed
    - security_scan_node end-to-end behaviour with stubbed scanner adapters
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from harness.security import (
    SecurityFinding,
    SecurityScanPolicy,
    ScannerOutcome,
    ScannerStatus,
    apply_policy,
    _findings_to_diagnostics,
    _normalize_severity,
    _parse_bandit_json,
    _parse_gitleaks_json,
    _parse_semgrep_json,
    _parse_trivy_json,
    _severity_at_or_above,
)


# ---------------------------------------------------------------------------
# SecurityFinding
# ---------------------------------------------------------------------------

class TestSecurityFinding:

    def test_dedupe_key_collides_on_same_rule_file_line_message(self):
        a = SecurityFinding(
            scanner="semgrep", rule_id="python.lang.security.audit.sql-injection",
            severity="high", file="app/db.py", line=42,
            message="Possible SQL injection",
            cwe="CWE-89", confidence="high",
        )
        b = SecurityFinding(
            scanner="bandit", rule_id="python.lang.security.audit.sql-injection",
            severity="high", file="app/db.py", line=42,
            message="Possible SQL injection",
            cwe="CWE-89", confidence="medium",
        )
        # Different scanners — the dedupe key intentionally ignores
        # scanner identity so cross-scanner duplicates collapse.
        assert a.dedupe_key() == b.dedupe_key()

    def test_distinct_lines_have_distinct_keys(self):
        a = SecurityFinding(
            scanner="bandit", rule_id="B201", severity="medium",
            file="x.py", line=10, message="hi",
        )
        b = SecurityFinding(
            scanner="bandit", rule_id="B201", severity="medium",
            file="x.py", line=11, message="hi",
        )
        assert a.dedupe_key() != b.dedupe_key()

    def test_findings_are_frozen_hashable(self):
        f = SecurityFinding(
            scanner="bandit", rule_id="B201", severity="medium",
            file="x.py", line=1, message="hi",
        )
        # Frozen dataclasses are hashable — required for set-based dedup.
        s = {f, f}
        assert len(s) == 1


# ---------------------------------------------------------------------------
# Severity normalization
# ---------------------------------------------------------------------------

class TestSeverityNormalization:

    def test_bandit_high_medium_low_normalize(self):
        assert _normalize_severity("bandit", "HIGH") == "high"
        assert _normalize_severity("bandit", "MEDIUM") == "medium"
        assert _normalize_severity("bandit", "LOW") == "low"

    def test_semgrep_error_warning_info_normalize(self):
        # Semgrep emits ERROR / WARNING / INFO — these map onto our
        # high/medium/low/info scale so policy thresholds compare apples
        # to apples across scanners.
        assert _normalize_severity("semgrep", "ERROR") == "high"
        assert _normalize_severity("semgrep", "WARNING") == "medium"
        assert _normalize_severity("semgrep", "INFO") == "low"

    def test_trivy_critical_normalizes(self):
        assert _normalize_severity("trivy", "CRITICAL") == "critical"
        assert _normalize_severity("trivy", "HIGH") == "high"
        assert _normalize_severity("trivy", "UNKNOWN") == "info"

    def test_gitleaks_always_high(self):
        # Gitleaks doesn't emit a per-finding severity — every secret
        # is treated as a high-severity breach regardless of which key
        # it was. This is the right default; teams can downgrade via
        # allowlist if they have a specific reason.
        assert _normalize_severity("gitleaks", "") == "high"
        assert _normalize_severity("gitleaks", "anything") == "high"

    def test_severity_at_or_above_ordering(self):
        # critical > high > medium > low > info; at_or_above answers
        # "is sev as serious as threshold?".
        assert _severity_at_or_above("critical", "high")
        assert _severity_at_or_above("high", "high")
        assert not _severity_at_or_above("medium", "high")
        assert _severity_at_or_above("low", "low")
        assert not _severity_at_or_above("info", "low")


# ---------------------------------------------------------------------------
# SecurityScanPolicy
# ---------------------------------------------------------------------------

class TestSecurityScanPolicy:

    def test_defaults_block_critical_high_warn_medium(self):
        p = SecurityScanPolicy()
        assert p.block_on == frozenset({"critical", "high"})
        assert p.warn_on == frozenset({"medium"})
        assert p.ignore_below == "low"
        assert "gitleaks" in p.scanners
        assert "trivy" in p.scanners

    def test_from_config_overrides_defaults(self):
        p = SecurityScanPolicy.from_config({
            "block_on": ["CRITICAL"],
            "warn_on": ["high", "medium"],
            "ignore_below": "medium",
            "scanners": ["semgrep"],
            "allowlist_rules": ["python.lang.security.audit.formatted-sql-query"],
            "max_findings_to_route_to_repair": 5,
        })
        # Case-folded; severities are stored lowercase.
        assert p.block_on == frozenset({"critical"})
        assert p.warn_on == frozenset({"high", "medium"})
        assert p.ignore_below == "medium"
        assert p.scanners == ("semgrep",)
        assert "python.lang.security.audit.formatted-sql-query" in p.allowlist_rules
        assert p.max_findings_to_route_to_repair == 5

    def test_from_config_missing_keys_use_defaults(self):
        # Legacy config that only sets enabled / scanner paths — every
        # policy field falls back to its default, no KeyError.
        p = SecurityScanPolicy.from_config({"enabled": True})
        assert p.block_on == frozenset({"critical", "high"})
        assert p.max_findings_to_route_to_repair == 10

    def test_from_config_empty_block_list_uses_defaults(self):
        # An explicitly empty block_on would be a footgun (no findings
        # ever block) — fall back to safe defaults rather than honor it.
        p = SecurityScanPolicy.from_config({"block_on": []})
        assert p.block_on == frozenset({"critical", "high"})


# ---------------------------------------------------------------------------
# apply_policy: partitioning, allowlist, cap, dedup
# ---------------------------------------------------------------------------

class TestApplyPolicy:

    def _f(self, **kwargs: Any) -> SecurityFinding:
        defaults = dict(
            scanner="semgrep", rule_id="rule-x", severity="high",
            file="src/a.py", line=10, message="m",
        )
        defaults.update(kwargs)
        return SecurityFinding(**defaults)

    def test_critical_and_high_block_under_default_policy(self):
        policy = SecurityScanPolicy()
        block, warn = apply_policy([
            self._f(severity="critical"),
            self._f(severity="high", line=11),
        ], policy)
        assert len(block) == 2
        assert not warn

    def test_medium_warns_low_drops_under_default_policy(self):
        policy = SecurityScanPolicy()
        block, warn = apply_policy([
            self._f(severity="medium"),
            self._f(severity="low", line=11),
        ], policy)
        assert not block
        assert len(warn) == 1

    def test_info_dropped_under_default_policy(self):
        # info is strictly below ignore_below (low) so it never even
        # appears in warn output.
        policy = SecurityScanPolicy()
        block, warn = apply_policy([self._f(severity="info")], policy)
        assert not block and not warn

    def test_allowlist_drops_specific_rule_id(self):
        policy = SecurityScanPolicy(
            allowlist_rules=frozenset({"python.lang.security.audit.formatted-sql-query"}),
        )
        block, warn = apply_policy([
            self._f(rule_id="python.lang.security.audit.formatted-sql-query", severity="high"),
            self._f(rule_id="other-rule", severity="high"),
        ], policy)
        assert len(block) == 1
        assert block[0].rule_id == "other-rule"
        assert not warn

    def test_block_list_capped(self):
        # 50 highs in, cap=3 → only 3 routed to repair.
        policy = SecurityScanPolicy(max_findings_to_route_to_repair=3)
        block, _ = apply_policy(
            [self._f(severity="high", line=i, file=f"f{i}.py") for i in range(50)],
            policy,
        )
        assert len(block) == 3

    def test_block_list_sorted_critical_first(self):
        # Critical findings must rank ahead of high — if the cap forces
        # us to drop some, we drop the less severe ones.
        policy = SecurityScanPolicy(max_findings_to_route_to_repair=2)
        block, _ = apply_policy([
            self._f(severity="high", line=1, file="a.py"),
            self._f(severity="critical", line=2, file="b.py"),
            self._f(severity="high", line=3, file="c.py"),
        ], policy)
        assert len(block) == 2
        assert block[0].severity == "critical"

    def test_dedupes_cross_scanner_duplicates(self):
        # Bandit and semgrep both flag the same SQLi on the same line —
        # we only want it in the block list once. The dedupe key is
        # (rule_id, file, line, message) regardless of scanner.
        policy = SecurityScanPolicy()
        block, _ = apply_policy([
            self._f(scanner="bandit",  rule_id="r1", severity="high",
                    file="x.py", line=5, message="SQLi"),
            self._f(scanner="semgrep", rule_id="r1", severity="high",
                    file="x.py", line=5, message="SQLi"),
        ], policy)
        assert len(block) == 1

    def test_severity_not_in_block_or_warn_dropped(self):
        # Policy says only critical blocks and only high warns.
        # A medium finding falls through both lists and is dropped.
        policy = SecurityScanPolicy(
            block_on=frozenset({"critical"}),
            warn_on=frozenset({"high"}),
            ignore_below="low",
        )
        block, warn = apply_policy([self._f(severity="medium")], policy)
        assert not block and not warn


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

class TestGitleaksParser:

    def test_empty_input_returns_empty(self):
        assert _parse_gitleaks_json("") == []
        assert _parse_gitleaks_json("[]") == []

    def test_extracts_findings_with_cwe_798(self):
        raw = json.dumps([
            {"RuleID": "aws-access-key", "File": "src/aws.py", "StartLine": 12,
             "Description": "AWS Access Key detected", "Secret": "AKIAIOSFODNN7EXAMPLE"},
        ])
        findings = _parse_gitleaks_json(raw)
        assert len(findings) == 1
        f = findings[0]
        assert f.scanner == "gitleaks"
        assert f.rule_id == "aws-access-key"
        assert f.severity == "high"
        assert f.file == "src/aws.py"
        assert f.line == 12
        assert f.cwe == "CWE-798"

    def test_malformed_json_returns_empty(self):
        # Gitleaks crashing mid-write → don't blow up; just yield no findings.
        assert _parse_gitleaks_json("not-json{{{") == []


class TestBanditParser:

    def test_extracts_test_id_and_cwe(self):
        raw = json.dumps({
            "results": [{
                "filename": "/abs/path/app.py",
                "line_number": 7,
                "test_id": "B201",
                "issue_text": "Flask app deployed with debug=True",
                "issue_severity": "HIGH",
                "issue_confidence": "MEDIUM",
                "issue_cwe": {"id": 94},
            }],
        })
        findings = _parse_bandit_json(raw)
        assert len(findings) == 1
        f = findings[0]
        assert f.scanner == "bandit"
        assert f.rule_id == "B201"
        assert f.severity == "high"
        assert f.cwe == "CWE-94"
        assert f.confidence == "medium"

    def test_missing_cwe_yields_none(self):
        raw = json.dumps({"results": [{
            "filename": "x.py", "line_number": 1, "test_id": "B999",
            "issue_text": "x", "issue_severity": "LOW",
        }]})
        f = _parse_bandit_json(raw)[0]
        assert f.cwe is None


class TestSemgrepParser:

    def test_extracts_check_id_severity_and_cwe(self):
        raw = json.dumps({"results": [{
            "check_id": "python.lang.security.audit.sql-injection",
            "path": "app/db.py",
            "start": {"line": 42},
            "extra": {
                "severity": "ERROR",
                "message": "Possible SQL injection",
                "metadata": {"cwe": ["CWE-89: SQL Injection"], "confidence": "HIGH"},
            },
        }]})
        f = _parse_semgrep_json(raw)[0]
        assert f.scanner == "semgrep"
        assert f.severity == "high"
        assert f.rule_id == "python.lang.security.audit.sql-injection"
        assert f.cwe == "CWE-89"
        assert f.confidence == "high"

    def test_cwe_as_bare_number_normalized(self):
        raw = json.dumps({"results": [{
            "check_id": "r", "path": "x.py", "start": {"line": 1},
            "extra": {"severity": "WARNING", "message": "x",
                      "metadata": {"cwe": "79"}},
        }]})
        f = _parse_semgrep_json(raw)[0]
        assert f.cwe == "CWE-79"


class TestTrivyParser:

    def test_flattens_results_into_findings(self):
        # Trivy nests vulns under per-target Results arrays — we flatten
        # so the gate sees one finding per vuln regardless of which
        # lockfile / image layer they came from.
        raw = json.dumps({
            "Results": [{
                "Target": "package-lock.json",
                "Vulnerabilities": [
                    {"VulnerabilityID": "CVE-2024-0001", "PkgName": "lodash",
                     "InstalledVersion": "4.17.20", "FixedVersion": "4.17.21",
                     "Severity": "HIGH",
                     "CweIDs": ["CWE-1321"],
                     "Title": "Prototype Pollution"},
                    {"VulnerabilityID": "CVE-2024-0002", "PkgName": "axios",
                     "InstalledVersion": "0.20.0", "FixedVersion": "",
                     "Severity": "MEDIUM",
                     "Title": "ReDoS"},
                ],
            }],
        })
        findings = _parse_trivy_json(raw)
        assert len(findings) == 2
        first = findings[0]
        assert first.scanner == "trivy"
        assert first.rule_id == "CVE-2024-0001"
        assert first.severity == "high"
        assert first.cwe == "CWE-1321"
        assert "4.17.21" in first.message
        # The unfixed vuln must surface the workaround hint so a repair
        # LLM doesn't loop trying to "upgrade to version " (empty).
        unfixed = findings[1]
        assert unfixed.confidence == "medium"
        assert "No fix" in unfixed.message


# ---------------------------------------------------------------------------
# _findings_to_diagnostics
# ---------------------------------------------------------------------------

class TestFindingsToDiagnostics:

    def test_critical_and_high_emit_error_severity(self):
        diags = _findings_to_diagnostics([
            SecurityFinding(scanner="trivy", rule_id="CVE-X", severity="critical",
                            file="x.json", line=0, message="bad", cwe="CWE-89"),
            SecurityFinding(scanner="bandit", rule_id="B201", severity="high",
                            file="y.py", line=10, message="m", cwe="CWE-94"),
        ])
        assert all(d["severity"] == "error" for d in diags)

    def test_medium_emits_warning_severity(self):
        diags = _findings_to_diagnostics([
            SecurityFinding(scanner="semgrep", rule_id="r", severity="medium",
                            file="x.py", line=1, message="m"),
        ])
        assert diags[0]["severity"] == "warning"

    def test_message_contains_cwe_when_set(self):
        d = _findings_to_diagnostics([SecurityFinding(
            scanner="bandit", rule_id="B201", severity="high",
            file="x.py", line=1, message="bad", cwe="CWE-94",
        )])[0]
        assert "CWE-94" in d["message"]
        assert "BANDIT:B201" == d["error_code"]


# ---------------------------------------------------------------------------
# security_scan_node — end to end with stubbed scanner adapters
# ---------------------------------------------------------------------------

class TestSecurityScanNode:

    @pytest.fixture
    def workspace(self, tmp_path) -> str:
        return str(tmp_path)

    @pytest.mark.asyncio
    async def test_disabled_short_circuits(self, workspace):
        from harness.security import security_scan_node
        result = await security_scan_node({
            "workspace_path": workspace,
            "security_scan_config": {"enabled": False},
        })
        assert result == {}

    @pytest.mark.asyncio
    async def test_all_clean_returns_passed_true(self, workspace, monkeypatch):
        # Stub every scanner with an OK outcome — node must report
        # passed=True and not populate compiler_errors.
        from harness import security as sec

        async def ok(name):
            return ScannerOutcome(scanner=name, status=ScannerStatus.OK, findings=[])

        monkeypatch.setattr(sec, "run_gitleaks_scan", lambda *a, **k: ok("gitleaks"))
        monkeypatch.setattr(sec, "run_bandit_scan",   lambda *a, **k: ok("bandit"))
        monkeypatch.setattr(sec, "run_semgrep_scan",  lambda *a, **k: ok("semgrep"))
        monkeypatch.setattr(sec, "run_trivy_scan",    lambda *a, **k: ok("trivy"))

        result = await sec.security_scan_node({
            "workspace_path": workspace,
            "security_scan_config": {"enabled": True},
        })
        assert result["node_state"]["security_scan"]["passed"] is True
        assert "compiler_errors" not in result

    @pytest.mark.asyncio
    async def test_high_finding_blocks_and_routes_to_patching(self, workspace, monkeypatch):
        # One high-severity SAST finding → compiler_errors populated +
        # loop_counter bumped → router will send this to patching_node.
        from harness import security as sec

        async def bandit_with_high(*a, **k):
            return ScannerOutcome(
                scanner="bandit", status=ScannerStatus.FOUND,
                findings=[SecurityFinding(
                    scanner="bandit", rule_id="B201", severity="high",
                    file="app.py", line=42, message="Flask debug=True",
                    cwe="CWE-94", confidence="high",
                )],
            )

        async def empty(name):
            return ScannerOutcome(scanner=name, status=ScannerStatus.OK)

        monkeypatch.setattr(sec, "run_gitleaks_scan", lambda *a, **k: empty("gitleaks"))
        monkeypatch.setattr(sec, "run_bandit_scan",   bandit_with_high)
        monkeypatch.setattr(sec, "run_semgrep_scan",  lambda *a, **k: empty("semgrep"))
        monkeypatch.setattr(sec, "run_trivy_scan",    lambda *a, **k: empty("trivy"))

        result = await sec.security_scan_node({
            "workspace_path": workspace,
            "security_scan_config": {"enabled": True},
            "messages": [],
        })
        assert result["node_state"]["security_scan"]["passed"] is False
        assert result["node_state"]["security_scan"]["block_count"] == 1
        assert result["loop_counter"]["security"] == 1
        assert len(result["compiler_errors"]) == 1
        assert "BANDIT:B201" in result["compiler_errors"][0]["error_code"]
        assert "CWE-94" in result["compiler_errors"][0]["message"]

    @pytest.mark.asyncio
    async def test_medium_warns_does_not_route(self, workspace, monkeypatch):
        # Medium severity → warn list → status remains passed=True,
        # no compiler_errors, but the warning is recorded in node_state.
        from harness import security as sec

        async def semgrep_with_med(*a, **k):
            return ScannerOutcome(
                scanner="semgrep", status=ScannerStatus.FOUND,
                findings=[SecurityFinding(
                    scanner="semgrep", rule_id="r-med", severity="medium",
                    file="x.py", line=1, message="meh",
                )],
            )

        async def empty(name):
            return ScannerOutcome(scanner=name, status=ScannerStatus.OK)

        monkeypatch.setattr(sec, "run_gitleaks_scan", lambda *a, **k: empty("gitleaks"))
        monkeypatch.setattr(sec, "run_bandit_scan",   lambda *a, **k: empty("bandit"))
        monkeypatch.setattr(sec, "run_semgrep_scan",  semgrep_with_med)
        monkeypatch.setattr(sec, "run_trivy_scan",    lambda *a, **k: empty("trivy"))

        result = await sec.security_scan_node({
            "workspace_path": workspace,
            "security_scan_config": {"enabled": True},
        })
        assert result["node_state"]["security_scan"]["passed"] is True
        assert result["node_state"]["security_scan"]["block_count"] == 0
        assert result["node_state"]["security_scan"]["warn_count"] == 1
        assert "compiler_errors" not in result

    @pytest.mark.asyncio
    async def test_allowlist_drops_finding(self, workspace, monkeypatch):
        # The high-severity finding is allowlisted by rule_id → node
        # behaves as if the scanner returned clean.
        from harness import security as sec

        async def semgrep_with_allowed(*a, **k):
            return ScannerOutcome(
                scanner="semgrep", status=ScannerStatus.FOUND,
                findings=[SecurityFinding(
                    scanner="semgrep",
                    rule_id="python.lang.security.audit.formatted-sql-query",
                    severity="high", file="x.py", line=1, message="m",
                )],
            )

        async def empty(name):
            return ScannerOutcome(scanner=name, status=ScannerStatus.OK)

        monkeypatch.setattr(sec, "run_gitleaks_scan", lambda *a, **k: empty("gitleaks"))
        monkeypatch.setattr(sec, "run_bandit_scan",   lambda *a, **k: empty("bandit"))
        monkeypatch.setattr(sec, "run_semgrep_scan",  semgrep_with_allowed)
        monkeypatch.setattr(sec, "run_trivy_scan",    lambda *a, **k: empty("trivy"))

        result = await sec.security_scan_node({
            "workspace_path": workspace,
            "security_scan_config": {
                "enabled": True,
                "allowlist_rules": ["python.lang.security.audit.formatted-sql-query"],
            },
        })
        assert result["node_state"]["security_scan"]["passed"] is True

    @pytest.mark.asyncio
    async def test_crashed_scanner_surfaces_but_does_not_block(self, workspace, monkeypatch):
        # When a scanner crashes (e.g. semgrep can't parse a rule pack),
        # we DON'T treat the absence of its findings as "clean" — we
        # report it as incomplete coverage. The build still continues
        # because no blocking finding was confirmed; the user sees the
        # crash in node_state and can rerun with stricter config.
        from harness import security as sec

        async def semgrep_crashed(*a, **k):
            return ScannerOutcome(
                scanner="semgrep", status=ScannerStatus.CRASHED,
                error="rule pack download failed",
            )

        async def empty(name):
            return ScannerOutcome(scanner=name, status=ScannerStatus.OK)

        monkeypatch.setattr(sec, "run_gitleaks_scan", lambda *a, **k: empty("gitleaks"))
        monkeypatch.setattr(sec, "run_bandit_scan",   lambda *a, **k: empty("bandit"))
        monkeypatch.setattr(sec, "run_semgrep_scan",  semgrep_crashed)
        monkeypatch.setattr(sec, "run_trivy_scan",    lambda *a, **k: empty("trivy"))

        result = await sec.security_scan_node({
            "workspace_path": workspace,
            "security_scan_config": {"enabled": True},
        })
        assert result["node_state"]["security_scan"]["passed"] is True
        assert "semgrep" in result["node_state"]["security_scan"]["scanners_crashed"]

    @pytest.mark.asyncio
    async def test_cap_limits_compiler_errors(self, workspace, monkeypatch):
        # 50 high-severity findings + cap=3 → only 3 reach compiler_errors.
        # The remaining are silently dropped so the repair LLM doesn't
        # drown in identical findings.
        from harness import security as sec

        async def semgrep_many(*a, **k):
            return ScannerOutcome(
                scanner="semgrep", status=ScannerStatus.FOUND,
                findings=[
                    SecurityFinding(
                        scanner="semgrep", rule_id="r", severity="high",
                        file=f"f{i}.py", line=i, message=f"finding {i}",
                    )
                    for i in range(50)
                ],
            )

        async def empty(name):
            return ScannerOutcome(scanner=name, status=ScannerStatus.OK)

        monkeypatch.setattr(sec, "run_gitleaks_scan", lambda *a, **k: empty("gitleaks"))
        monkeypatch.setattr(sec, "run_bandit_scan",   lambda *a, **k: empty("bandit"))
        monkeypatch.setattr(sec, "run_semgrep_scan",  semgrep_many)
        monkeypatch.setattr(sec, "run_trivy_scan",    lambda *a, **k: empty("trivy"))

        result = await sec.security_scan_node({
            "workspace_path": workspace,
            "security_scan_config": {
                "enabled": True,
                "max_findings_to_route_to_repair": 3,
            },
        })
        assert len(result["compiler_errors"]) == 3
        assert result["node_state"]["security_scan"]["raw_findings_total"] == 50

    @pytest.mark.asyncio
    async def test_repair_node_uses_security_framing_for_scanner_diagnostics(
        self, workspace, monkeypatch,
    ):
        # When every compiler_errors entry carries a scanner prefix
        # (BANDIT:, SEMGREP:, TRIVY:, GITLEAKS:, GITLEAKS-FALLBACK:),
        # repair_node must use the security framing sentence — not the
        # generic "build failed" framing. Compile errors must still use
        # the generic framing.
        from harness import graph

        captured: dict[str, Any] = {}

        class StubResponse:
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
                # Capture the assembled prompt for inspection. Return a
                # benign empty response so repair_node short-circuits
                # without trying to parse / apply patches.
                captured["messages"] = list(messages)
                return StubResponse(), budget_remaining_usd

        graph.set_gateway(StubGateway())
        try:
            security_diag = {
                "file": "app/db.py", "line": 42, "column": 0,
                "severity": "error",
                "error_code": "BANDIT:B608",
                "message": "[SECURITY HIGH] bandit/B608 in app/db.py:42: SQLi (CWE-89)",
                "semantic_context": "Scanner: bandit | Rule: B608 | Severity: high | Confidence: high | CWE-89",
            }
            state = {
                "workspace_path": workspace,
                "compiler_errors": [security_diag],
                "loop_counter": {"repair": 0, "total_repairs": 0, "security": 1},
                "messages": [],
                "budget_remaining_usd": 1.0,
            }
            await graph.repair_node(state)
        finally:
            graph.set_gateway(None)

        # The repair prompt is the first user message after the existing
        # message history. Find it and check the framing.
        user_msgs = [m for m in captured["messages"] if m["role"] == "user"]
        assert user_msgs, "repair_node should have appended a user prompt"
        prompt_text = user_msgs[0]["content"]
        assert "deterministic security gate" in prompt_text, (
            "security framing sentence missing — repair prompt would not "
            "tell the LLM that the build is green and these are post-build "
            "vulnerabilities, not compile errors"
        )
        # The structured diagnostics block still appears — that's what
        # _format_diagnostics_for_repair produces and what we're
        # specifically preserving by routing through repair_node.
        assert "Compiler Diagnostics" in prompt_text
        assert "BANDIT:B608" in prompt_text
        assert "CWE-89" in prompt_text

    @pytest.mark.asyncio
    async def test_repair_node_uses_generic_framing_for_compile_errors(
        self, workspace, monkeypatch,
    ):
        # The mirror test: a non-security diagnostic (e.g. a Rust compile
        # error) must NOT trigger the security framing. The detection is
        # all-or-nothing — any non-security diagnostic in the batch falls
        # through to the generic / escalation framing.
        from harness import graph

        captured: dict[str, Any] = {}

        class StubResponse:
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
                captured["messages"] = list(messages)
                return StubResponse(), budget_remaining_usd

        graph.set_gateway(StubGateway())
        try:
            compile_diag = {
                "file": "src/main.rs", "line": 12, "column": 5,
                "severity": "error",
                "error_code": "E0308",
                "message": "mismatched types",
                "semantic_context": "expected `i32`, found `&str`",
            }
            state = {
                "workspace_path": workspace,
                "compiler_errors": [compile_diag],
                "loop_counter": {"repair": 0, "total_repairs": 0},
                "messages": [],
                "budget_remaining_usd": 1.0,
            }
            await graph.repair_node(state)
        finally:
            graph.set_gateway(None)

        user_msgs = [m for m in captured["messages"] if m["role"] == "user"]
        prompt_text = user_msgs[0]["content"]
        assert "deterministic security gate" not in prompt_text, (
            "security framing leaked into a compile-error repair — the "
            "detection should fire only when ALL diagnostics carry a "
            "scanner prefix"
        )
        # Generic framing must still appear.
        assert "build failed" in prompt_text.lower()

    @pytest.mark.asyncio
    async def test_findings_route_to_repair_node(self, workspace, monkeypatch):
        # Full path: security_scan_node populates compiler_errors with
        # scanner-prefixed error codes, then route_after_security_scan
        # ships the state to repair_node (NOT patching_node). This is
        # the wiring that brings security fixes onto the same
        # _format_diagnostics_for_repair path compile errors use.
        from harness import security as sec
        from harness.graph import route_after_security_scan

        async def bandit_with_high(*a, **k):
            return ScannerOutcome(
                scanner="bandit", status=ScannerStatus.FOUND,
                findings=[SecurityFinding(
                    scanner="bandit", rule_id="B201", severity="high",
                    file="app.py", line=42, message="Flask debug=True",
                    cwe="CWE-94",
                )],
            )

        async def empty(name):
            return ScannerOutcome(scanner=name, status=ScannerStatus.OK)

        monkeypatch.setattr(sec, "run_gitleaks_scan", lambda *a, **k: empty("gitleaks"))
        monkeypatch.setattr(sec, "run_bandit_scan",   bandit_with_high)
        monkeypatch.setattr(sec, "run_semgrep_scan",  lambda *a, **k: empty("semgrep"))
        monkeypatch.setattr(sec, "run_trivy_scan",    lambda *a, **k: empty("trivy"))

        state: dict[str, Any] = {
            "workspace_path": workspace,
            "security_scan_config": {"enabled": True},
            "messages": [],
            "budget_remaining_usd": 1.0,
        }
        update = await sec.security_scan_node(state)
        state.update(update)
        # The gate's update carries the diagnostics; merge into state
        # before consulting the router (LangGraph does this for us in
        # production).
        assert state["compiler_errors"]
        assert state["compiler_errors"][0]["error_code"].startswith("BANDIT:")

        assert route_after_security_scan(state) == "repair_node"

    @pytest.mark.asyncio
    async def test_disabled_scanner_skips_invocation(self, workspace, monkeypatch):
        # Policy lists only "gitleaks" → trivy / bandit / semgrep
        # adapters must not even be called. We verify this by stubbing
        # them with a sentinel that would blow up the test if called.
        from harness import security as sec

        async def empty_gitleaks(*a, **k):
            return ScannerOutcome(scanner="gitleaks", status=ScannerStatus.OK)

        async def boom(*a, **k):
            raise AssertionError("scanner should not have run")

        monkeypatch.setattr(sec, "run_gitleaks_scan", empty_gitleaks)
        monkeypatch.setattr(sec, "run_bandit_scan",   boom)
        monkeypatch.setattr(sec, "run_semgrep_scan",  boom)
        monkeypatch.setattr(sec, "run_trivy_scan",    boom)

        result = await sec.security_scan_node({
            "workspace_path": workspace,
            "security_scan_config": {
                "enabled": True,
                "scanners": ["gitleaks"],
            },
        })
        assert result["node_state"]["security_scan"]["passed"] is True
        assert result["node_state"]["security_scan"]["scanners_clean"] == ["gitleaks"]

    # -----------------------------------------------------------------
    # Architecture-summary handoff (§11 jsonc) into the repair LLM
    # -----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_arch_summary_preamble_injected_before_findings(
        self, workspace, monkeypatch,
    ):
        """When state carries a §11 summary AND there are blocking
        findings, the messages array must contain an architecture
        preamble (system role) immediately BEFORE the findings
        breadcrumb. The repair LLM should read the resolved stack
        before the findings list."""
        from harness import security as sec

        async def bandit_with_high(*a, **k):
            return ScannerOutcome(
                scanner="bandit", status=ScannerStatus.FOUND,
                findings=[SecurityFinding(
                    scanner="bandit", rule_id="B201", severity="high",
                    file="app.py", line=42, message="Flask debug=True",
                    cwe="CWE-94", confidence="high",
                )],
            )

        async def empty(name):
            return ScannerOutcome(scanner=name, status=ScannerStatus.OK)

        monkeypatch.setattr(sec, "run_gitleaks_scan", lambda *a, **k: empty("gitleaks"))
        monkeypatch.setattr(sec, "run_bandit_scan",   bandit_with_high)
        monkeypatch.setattr(sec, "run_semgrep_scan",  lambda *a, **k: empty("semgrep"))
        monkeypatch.setattr(sec, "run_trivy_scan",    lambda *a, **k: empty("trivy"))

        arch = {
            "schema_version": 1,
            "backend_language": "python_fastapi",
            "frontend": "none",
            "db_engine": "postgres",
            "auth_strategy": "jwt",
            "backend": {"endpoints": [{
                "id": "EP-001", "method": "POST", "path": "/api/v1/login",
                "rsd_story_ids": ["STORY-1"],
            }]},
            "contract": {"openapi_spec_path": "contracts/openapi.json"},
        }

        result = await sec.security_scan_node({
            "workspace_path": workspace,
            "security_scan_config": {"enabled": True},
            "messages": [],
            "arch_summary": arch,
        })

        assert result["node_state"]["security_scan"]["passed"] is False
        # System messages added by the node, in order.
        added = [m for m in result["messages"] if m["role"] == "system"]
        assert len(added) >= 2, "expected arch preamble + findings breadcrumb"
        # Penultimate is the arch preamble; last is the findings.
        arch_msg = added[-2]["content"]
        findings_msg = added[-1]["content"]
        assert "Architecture summary" in arch_msg
        assert "EP-001" in arch_msg
        assert "Security Scan" in findings_msg
        assert "BANDIT" in findings_msg or "bandit" in findings_msg
        # Resolved summary echoed back into state so a downstream
        # patching_node turn doesn't re-load from disk.
        assert result["arch_summary"]["schema_version"] == 1

    @pytest.mark.asyncio
    async def test_no_arch_preamble_when_summary_absent(
        self, workspace, monkeypatch,
    ):
        """No state.arch_summary AND no SPEC_ARCHITECTURE.md on disk
        → the findings breadcrumb stands alone (pre-existing
        behaviour, no regression)."""
        from harness import security as sec

        async def bandit_with_high(*a, **k):
            return ScannerOutcome(
                scanner="bandit", status=ScannerStatus.FOUND,
                findings=[SecurityFinding(
                    scanner="bandit", rule_id="B201", severity="high",
                    file="app.py", line=42, message="Flask debug=True",
                )],
            )

        async def empty(name):
            return ScannerOutcome(scanner=name, status=ScannerStatus.OK)

        monkeypatch.setattr(sec, "run_gitleaks_scan", lambda *a, **k: empty("gitleaks"))
        monkeypatch.setattr(sec, "run_bandit_scan",   bandit_with_high)
        monkeypatch.setattr(sec, "run_semgrep_scan",  lambda *a, **k: empty("semgrep"))
        monkeypatch.setattr(sec, "run_trivy_scan",    lambda *a, **k: empty("trivy"))

        result = await sec.security_scan_node({
            "workspace_path": workspace,
            "security_scan_config": {"enabled": True},
            "messages": [],
        })

        added = [m for m in result["messages"] if m["role"] == "system"]
        # Exactly one system message — the findings breadcrumb. No
        # arch preamble means none of the system messages should
        # mention the canonical heading.
        joined = "\n".join(m["content"] for m in added)
        assert "Architecture summary" not in joined
        assert "Security Scan" in joined
        # Resolved summary in the return delta is the empty dict —
        # caching contract: "we looked, there's nothing".
        assert result["arch_summary"] == {}

    @pytest.mark.asyncio
    async def test_no_arch_preamble_on_clean_pass(
        self, workspace, monkeypatch,
    ):
        """Even with state.arch_summary set, a clean scan must NOT
        inject an arch preamble — there's no LLM hand-off on the
        clean path."""
        from harness import security as sec

        async def empty(name):
            return ScannerOutcome(scanner=name, status=ScannerStatus.OK)

        monkeypatch.setattr(sec, "run_gitleaks_scan", lambda *a, **k: empty("gitleaks"))
        monkeypatch.setattr(sec, "run_bandit_scan",   lambda *a, **k: empty("bandit"))
        monkeypatch.setattr(sec, "run_semgrep_scan",  lambda *a, **k: empty("semgrep"))
        monkeypatch.setattr(sec, "run_trivy_scan",    lambda *a, **k: empty("trivy"))

        arch = {
            "schema_version": 1,
            "backend_language": "python_fastapi",
            "frontend": "none",
            "backend": {"endpoints": []},
            "contract": {"openapi_spec_path": "contracts/openapi.json"},
        }
        result = await sec.security_scan_node({
            "workspace_path": workspace,
            "security_scan_config": {"enabled": True},
            "messages": [],
            "arch_summary": arch,
        })
        assert result["node_state"]["security_scan"]["passed"] is True
        # Clean path returns only node_state — no messages mutation.
        assert "messages" not in result


# ---------------------------------------------------------------------------
# exclude_paths: scanners must skip docs/ by default so the spec-driven
# patcher allowlist (which already rejects writes to docs/) never sees a
# fix request it has to reject — that was the cause of session 0ee7807d's
# 9-iteration HITL ping-pong loop where semgrep flagged code snippets in
# docs/SPEC_ARCHITECTURE.md and the LLM's patches kept bouncing.
# ---------------------------------------------------------------------------


class TestExcludePathsPolicy:

    def test_default_excludes_docs_and_product_spec(self):
        p = SecurityScanPolicy()
        assert "docs" in p.exclude_paths
        assert "product_spec" in p.exclude_paths

    def test_from_config_overrides_default_excludes(self):
        p = SecurityScanPolicy.from_config({"exclude_paths": ["fixtures", "examples"]})
        assert p.exclude_paths == ("fixtures", "examples")
        # The defaults are NOT merged in — explicit list is authoritative.
        assert "docs" not in p.exclude_paths

    def test_from_config_empty_list_disables_excludes(self):
        # Operator who genuinely wants to scan docs/ can opt in.
        p = SecurityScanPolicy.from_config({"exclude_paths": []})
        assert p.exclude_paths == ()

    def test_from_config_missing_key_uses_default(self):
        # Legacy config: untouched policies still get the safe default.
        p = SecurityScanPolicy.from_config({"enabled": True})
        assert "docs" in p.exclude_paths

    def test_normalize_strips_leading_slashes_and_dot(self):
        from harness.security import _normalize_exclude_path
        assert _normalize_exclude_path("./docs") == "docs"
        assert _normalize_exclude_path("/docs/") == "docs"
        assert _normalize_exclude_path("docs/sub/") == "docs/sub"

    def test_normalize_rejects_parent_traversal(self):
        # ``../etc/passwd`` as an "exclude path" would let an operator
        # silently mark anything outside the workspace as out-of-scope.
        # The normalizer drops these entirely.
        from harness.security import _normalize_exclude_path
        assert _normalize_exclude_path("..") == ""
        assert _normalize_exclude_path("../etc") == ""
        assert _normalize_exclude_path("a/../etc") == ""
        assert _normalize_exclude_path(".") == ""
        assert _normalize_exclude_path("") == ""


class TestPathUnderExcludes:

    def test_direct_match(self):
        from harness.security import _path_under_excludes
        assert _path_under_excludes("docs/x.md", ["docs"]) is True

    def test_nested_match(self):
        from harness.security import _path_under_excludes
        assert _path_under_excludes("docs/a/b/c.md", ["docs"]) is True

    def test_prefix_collision_not_excluded(self):
        # ``docs`` must not exclude ``docs2/``. The check is path-component
        # based, not raw startswith.
        from harness.security import _path_under_excludes
        assert _path_under_excludes("docs2/x.md", ["docs"]) is False

    def test_empty_excludes_list_excludes_nothing(self):
        from harness.security import _path_under_excludes
        assert _path_under_excludes("docs/x.md", []) is False

    def test_handles_backslash_input(self):
        # Defensive — even on POSIX a finding could carry a Windows-style
        # path from a vendored scanner. Normalize before comparing.
        from harness.security import _path_under_excludes
        assert _path_under_excludes("docs\\x.md", ["docs"]) is True


# ---------------------------------------------------------------------------
# Scanner command construction — confirm exclude_paths reaches the wire.
# We monkeypatch _run_subprocess_scanner to capture the argv each scanner
# would actually execute, then assert the right flags landed.
# ---------------------------------------------------------------------------


class TestScannerExcludeFlagWiring:

    @pytest.fixture
    def capture_cmd(self, monkeypatch):
        from harness import security as sec
        captured: dict[str, list[str]] = {}

        async def fake_run(cmd, timeout_seconds=15, label="scanner"):
            captured["cmd"] = list(cmd)
            # Return an "empty findings" successful exit so the caller's
            # parsing path runs but emits no findings. Each parser is
            # already covered by its own tests; we're only checking argv.
            return 0, "{}", ""

        monkeypatch.setattr(sec, "_run_subprocess_scanner", fake_run)
        monkeypatch.setattr(sec, "shutil", _StubShutil())
        return captured

    @pytest.mark.asyncio
    async def test_semgrep_passes_exclude_per_flag(self, capture_cmd):
        from harness.security import run_semgrep_scan
        await run_semgrep_scan(
            "/ws", semgrep_path="semgrep",
            exclude_paths=("docs", "product_spec"),
        )
        cmd = capture_cmd["cmd"]
        # Each excluded dir gets its own ``--exclude <dir>`` pair; the
        # workspace path is the LAST positional.
        assert cmd.count("--exclude") == 2
        assert "docs" in cmd
        assert "product_spec" in cmd
        assert cmd[-1] == "/ws"

    @pytest.mark.asyncio
    async def test_semgrep_no_excludes_omits_flag(self, capture_cmd):
        from harness.security import run_semgrep_scan
        await run_semgrep_scan(
            "/ws", semgrep_path="semgrep", exclude_paths=(),
        )
        assert "--exclude" not in capture_cmd["cmd"]

    @pytest.mark.asyncio
    async def test_bandit_passes_one_exclude_per_path(
        self, capture_cmd, monkeypatch,
    ):
        # Bandit only runs when .py files are detected; stub that out so
        # the run_bandit_scan body actually issues the subprocess call.
        from harness import security as sec
        monkeypatch.setattr(
            sec, "_scan_workspace_languages", lambda _p: (True, False, False),
        )
        from harness.security import run_bandit_scan
        await run_bandit_scan(
            "/ws", bandit_path="bandit",
            exclude_paths=("docs", "product_spec"),
        )
        cmd = capture_cmd["cmd"]
        # Bandit now receives one ``--exclude <path>`` per excluded
        # directory. The previous "comma-joined absolute paths into a
        # single -x" shape silently dropped paths containing commas and
        # corrupted Windows ``C:\…`` paths.
        excludes = [
            cmd[i + 1] for i, v in enumerate(cmd) if v == "--exclude"
        ]
        assert "/ws/docs" in excludes
        assert "/ws/product_spec" in excludes
        assert len(excludes) == 2

    @pytest.mark.asyncio
    async def test_bandit_no_excludes_omits_exclude_flag(
        self, capture_cmd, monkeypatch,
    ):
        from harness import security as sec
        monkeypatch.setattr(
            sec, "_scan_workspace_languages", lambda _p: (True, False, False),
        )
        from harness.security import run_bandit_scan
        await run_bandit_scan(
            "/ws", bandit_path="bandit", exclude_paths=(),
        )
        assert "--exclude" not in capture_cmd["cmd"]
        assert "-x" not in capture_cmd["cmd"]

    @pytest.mark.asyncio
    async def test_trivy_passes_skip_dirs_per_flag(self, capture_cmd):
        from harness.security import run_trivy_scan
        await run_trivy_scan(
            "/ws", trivy_path="trivy",
            exclude_paths=("docs", "product_spec"),
        )
        cmd = capture_cmd["cmd"]
        # Trivy wants ``--skip-dirs <dir>`` repeated.
        assert cmd.count("--skip-dirs") == 2
        assert "docs" in cmd
        assert "product_spec" in cmd

    @pytest.mark.asyncio
    async def test_gitleaks_post_filters_excluded_findings(self, monkeypatch):
        # gitleaks doesn't accept ad-hoc --exclude paths cleanly, so the
        # excludes are applied as a post-filter on the parsed findings.
        from harness import security as sec
        gitleaks_json = json.dumps([
            {"RuleID": "generic-api-key", "File": "docs/SPEC_ARCHITECTURE.md",
             "StartLine": 5, "Description": "JWT example", "Match": "x",
             "Secret": "y", "Tags": []},
            {"RuleID": "generic-api-key", "File": "src/auth.py",
             "StartLine": 9, "Description": "real key", "Match": "x",
             "Secret": "y", "Tags": []},
        ])

        async def fake_run(cmd, timeout_seconds=15, label="scanner"):
            return 0, gitleaks_json, ""

        monkeypatch.setattr(sec, "_run_subprocess_scanner", fake_run)
        monkeypatch.setattr(sec, "shutil", _StubShutil())

        outcome = await sec.run_gitleaks_scan(
            "/ws", gitleaks_path="gitleaks", exclude_paths=("docs",),
        )
        # Only the src/auth.py finding survives — the docs/ one was filtered.
        files = [f.file for f in outcome.findings]
        assert files == ["src/auth.py"]


class _StubShutil:
    """Make ``shutil.which`` succeed so the scanner adapter doesn't
    short-circuit on NOT_INSTALLED before reaching its subprocess call."""
    @staticmethod
    def which(name):
        return f"/usr/bin/{name}" if name else None


# ---------------------------------------------------------------------------
# HITL ping-pong hard ceiling. After 3 × max_security_fix_attempts the
# router gives up and ends the run rather than keep auto-resuming a
# loop that can't make progress (cf. session 0ee7807d, where the LLM
# kept proposing patches to docs/SPEC_ARCHITECTURE.md that the spec
# allowlist refused, and the loop only broke when semgrep timed out).
# ---------------------------------------------------------------------------


class TestRouterHardCeiling:

    def _make_state(self, *, attempts: int, max_attempts: int = 2,
                    hard_ceiling: int = None, errors: int = 1) -> dict[str, Any]:
        cfg: dict[str, Any] = {"max_security_fix_attempts": max_attempts}
        if hard_ceiling is not None:
            cfg["hard_security_loop_ceiling"] = hard_ceiling
        return {
            "budget_remaining_usd": 5.0,
            "loop_counter": {"security": attempts},
            "security_scan_config": cfg,
            "compiler_errors": [
                {"file": "x.py", "line": 1, "message": "stub"}
            ] * errors,
        }

    def test_below_soft_cap_routes_to_repair(self):
        from harness.graph import route_after_security_scan
        # 1 attempt, cap is 2 → still within the repair budget.
        state = self._make_state(attempts=1, max_attempts=2)
        assert route_after_security_scan(state) == "repair_node"

    def test_at_soft_cap_routes_to_hitl(self):
        from harness.graph import route_after_security_scan
        # At the soft cap (2/2) we go to HITL — operator decides.
        state = self._make_state(attempts=2, max_attempts=2)
        assert route_after_security_scan(state) == "human_intervention_node"

    def test_at_hard_ceiling_routes_to_end(self):
        from harness.graph import route_after_security_scan
        # Default ceiling = 3 * max_attempts = 6. At 6/2 the auto-resume
        # loop is provably non-productive; terminate.
        state = self._make_state(attempts=6, max_attempts=2)
        assert route_after_security_scan(state) == "__end__"

    def test_past_hard_ceiling_routes_to_end(self):
        from harness.graph import route_after_security_scan
        state = self._make_state(attempts=9, max_attempts=2)
        assert route_after_security_scan(state) == "__end__"

    def test_explicit_hard_ceiling_override(self):
        from harness.graph import route_after_security_scan
        # Operator can lower the ceiling to catch the loop sooner.
        state = self._make_state(
            attempts=3, max_attempts=2, hard_ceiling=3,
        )
        assert route_after_security_scan(state) == "__end__"

    def test_ceiling_does_not_fire_when_no_findings(self):
        from harness.graph import route_after_security_scan
        # No compiler_errors → the security gate passed; ceiling check
        # shouldn't trigger even with sec_attempts past the hard cap.
        # (This guards against routing a clean build straight to END.)
        state = self._make_state(
            attempts=99, max_attempts=2, errors=0,
        )
        state["dev_deployment"] = False  # → installation_doc_node
        # Phase G: simulate "EoS regression already ran" so the router
        # skips the EoS intercept and falls through to the
        # no-deployment installation_doc destination.
        state["loop_counter"]["end_of_session_regression_repair"] = 1
        assert route_after_security_scan(state) == "installation_doc_node"
