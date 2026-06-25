"""Tests for the Layer-2 YAML rule table.

Layer 2 is the bounded follow-up to Layer 1's ``extra.fix`` consumer:
it catches absence-based scanner findings (``missing-user``,
``missing-workdir``, …) the scanner ships no autofix template for.
Rules live in ``harness/security_fix_rules.yaml`` with optional
per-workspace overrides at ``{workspace}/security_fix_rules.yaml``.

These tests cover both the loader and the rule-application logic
end-to-end, and the integration with ``_fix_semgrep`` so Layer 1 wins
when both are applicable.
"""
from __future__ import annotations

import os

from harness.autofix import _fix_semgrep
from harness.patcher import OperationType
from harness.security_fix_rules import (
    HARNESS_RULES_FILE,
    apply_rule,
    load_rules,
    try_rule_table_fix,
)


def _w(tmp_path, rel: str, body: str) -> str:
    abs_path = os.path.join(str(tmp_path), rel)
    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(body)
    return abs_path


# ---------------------------------------------------------------------------
# Shipped YAML loads cleanly
# ---------------------------------------------------------------------------

def test_shipped_yaml_loads():
    assert os.path.isfile(HARNESS_RULES_FILE)
    rules = load_rules()
    # We ship 4 seed rules (missing-user, last-user-is-root,
    # missing-workdir, missing-healthcheck).
    assert len(rules) >= 4
    missing_user = rules.get((
        "SEMGREP",
        "dockerfile.security.missing-user.missing-user",
    ))
    assert missing_user is not None
    assert "USER 1000:1000" in missing_user.content


def test_workspace_override_wins(tmp_path):
    # Operator-supplied YAML at workspace root replaces the shipped
    # rule with the same (scanner, rule).
    _w(tmp_path, "security_fix_rules.yaml", """
rules:
  - scanner: SEMGREP
    rule: dockerfile.security.missing-user.missing-user
    description: Custom non-root UID for this project
    operation: insert_at_line
    content: "USER 4242:4242"
    before_regex: '^(CMD|ENTRYPOINT)\\b'
    fallback: at_end
    idempotency_regex: '^USER\\s+\\S+'
""".strip())
    table = load_rules(str(tmp_path))
    rule = table[("SEMGREP", "dockerfile.security.missing-user.missing-user")]
    assert "USER 4242:4242" in rule.content


# ---------------------------------------------------------------------------
# apply_rule — happy path (the CIOD HITL case)
# ---------------------------------------------------------------------------

def test_apply_missing_user_inserts_before_cmd(tmp_path):
    abs_path = _w(tmp_path, "Dockerfile",
        "FROM alpine:3.18\n"
        "COPY app /app\n"
        "RUN apk add --no-cache curl\n"
        "EXPOSE 8080\n"
        "CMD [\"./app\"]\n"
    )
    table = load_rules()
    rule = table[("SEMGREP", "dockerfile.security.missing-user.missing-user")]
    diag = {
        "error_code": "SEMGREP:dockerfile.security.missing-user.missing-user",
        "file": abs_path,
        "line": 5,
    }
    patch = apply_rule(rule, diag, str(tmp_path))
    assert patch is not None
    assert patch.operation == OperationType.INSERT_AT_LINE
    assert patch.file == "Dockerfile"
    # CMD is on line 5 — the patch must insert AT line 5 so the new
    # USER lands BEFORE it.
    assert patch.line == 5
    assert patch.content == "USER 1000:1000"
    assert patch.expected_file_hash  # snapshot taken


def test_apply_missing_workdir_no_match_no_patch(tmp_path):
    # missing-workdir uses fallback: none. If no COPY/RUN/ADD lines
    # exist there's no safe place to insert WORKDIR, so we punt to LLM.
    abs_path = _w(tmp_path, "Dockerfile",
        "FROM alpine:3.18\nCMD [\"./app\"]\n"
    )
    table = load_rules()
    rule = table[("SEMGREP", "dockerfile.audit.missing-workdir.missing-workdir")]
    diag = {
        "error_code": "SEMGREP:dockerfile.audit.missing-workdir.missing-workdir",
        "file": abs_path,
        "line": 1,
    }
    assert apply_rule(rule, diag, str(tmp_path)) is None


def test_apply_with_at_end_fallback(tmp_path):
    # missing-user with no CMD/ENTRYPOINT in the file — should append.
    abs_path = _w(tmp_path, "Dockerfile",
        "FROM alpine:3.18\nCOPY x /x\n"
    )
    table = load_rules()
    rule = table[("SEMGREP", "dockerfile.security.missing-user.missing-user")]
    diag = {
        "error_code": "SEMGREP:dockerfile.security.missing-user.missing-user",
        "file": abs_path,
        "line": 2,
    }
    patch = apply_rule(rule, diag, str(tmp_path))
    assert patch is not None
    # File has 2 lines; "append" is line 3 (len+1).
    assert patch.line == 3


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_idempotency_check_skips_when_already_satisfied(tmp_path):
    abs_path = _w(tmp_path, "Dockerfile",
        "FROM alpine:3.18\n"
        "USER 1000:1000\n"  # already in place
        "CMD [\"./app\"]\n"
    )
    table = load_rules()
    rule = table[("SEMGREP", "dockerfile.security.missing-user.missing-user")]
    diag = {
        "error_code": "SEMGREP:dockerfile.security.missing-user.missing-user",
        "file": abs_path,
        "line": 3,
    }
    # USER already present — apply_rule must return None so the
    # repair loop doesn't try to insert a duplicate.
    assert apply_rule(rule, diag, str(tmp_path)) is None


def test_last_user_is_root_idempotency_excludes_root(tmp_path):
    # The idempotency_regex on last-user-is-root deliberately matches
    # only NON-root USER lines, so a "USER root" file doesn't satisfy
    # the check and the rule still fires.
    abs_path = _w(tmp_path, "Dockerfile",
        "FROM alpine\nUSER root\nCMD [\"./app\"]\n"
    )
    table = load_rules()
    rule = table[("SEMGREP", "dockerfile.security.last-user-is-root.last-user-is-root")]
    diag = {
        "error_code": "SEMGREP:dockerfile.security.last-user-is-root.last-user-is-root",
        "file": abs_path,
        "line": 3,
    }
    patch = apply_rule(rule, diag, str(tmp_path))
    assert patch is not None
    assert patch.content == "USER 1000:1000"


# ---------------------------------------------------------------------------
# try_rule_table_fix — dispatcher entry point
# ---------------------------------------------------------------------------

def test_try_rule_table_fix_dispatches(tmp_path):
    abs_path = _w(tmp_path, "Dockerfile",
        "FROM alpine\nCMD [\"./app\"]\n"
    )
    diag = {
        "error_code": "SEMGREP:dockerfile.security.missing-user.missing-user",
        "file": abs_path,
        "line": 2,
    }
    patch = try_rule_table_fix(diag, str(tmp_path), scanner="SEMGREP")
    assert patch is not None
    assert patch.operation == OperationType.INSERT_AT_LINE


def test_try_rule_table_fix_returns_none_for_unknown_rule(tmp_path):
    diag = {
        "error_code": "SEMGREP:unknown.rule.name",
        "file": "Dockerfile",
    }
    assert try_rule_table_fix(diag, str(tmp_path), scanner="SEMGREP") is None


def test_try_rule_table_fix_ignores_other_scanners(tmp_path):
    # When the scanner doesn't match, the rule table doesn't fire even
    # if the rule_id is identical.
    abs_path = _w(tmp_path, "Dockerfile",
        "FROM alpine\nCMD [\"./app\"]\n"
    )
    diag = {
        "error_code": "BANDIT:dockerfile.security.missing-user.missing-user",
        "file": abs_path,
    }
    assert try_rule_table_fix(diag, str(tmp_path), scanner="BANDIT") is None


# ---------------------------------------------------------------------------
# Integration with _fix_semgrep — Layer 1 wins over Layer 2
# ---------------------------------------------------------------------------

def test_fix_semgrep_layer1_wins_over_layer2(tmp_path):
    # When both extra.fix is present AND the rule is in the YAML
    # table, the scanner's own suggestion wins (it's authored
    # per-finding, the rule table is the bounded fallback).
    abs_path = _w(tmp_path, "Dockerfile",
        "FROM alpine\nCMD [\"./app\"]\n"
    )
    diag = {
        "error_code": "SEMGREP:dockerfile.security.missing-user.missing-user",
        "file": abs_path,
        "line": 2,
        "end_line": 2,
        "fix": "USER 9999:9999  # custom UID",
    }
    patch = _fix_semgrep("missing-user", diag, str(tmp_path))
    assert patch is not None
    # Layer 1 yields REPLACE_LINE_RANGE, Layer 2 would have yielded
    # INSERT_AT_LINE — the fact that REPLACE_LINE_RANGE wins proves
    # the order.
    assert patch.operation == OperationType.REPLACE_LINE_RANGE
    assert "9999" in patch.content


def test_fix_semgrep_falls_through_to_layer2_when_no_extra_fix(tmp_path):
    # No fix metadata on the diagnostic — Layer 2 takes over and
    # emits an INSERT_AT_LINE using the YAML rule.
    abs_path = _w(tmp_path, "Dockerfile",
        "FROM alpine\nCMD [\"./app\"]\n"
    )
    diag = {
        "error_code": "SEMGREP:dockerfile.security.missing-user.missing-user",
        "file": abs_path,
        "line": 2,
    }
    patch = _fix_semgrep("missing-user", diag, str(tmp_path))
    assert patch is not None
    assert patch.operation == OperationType.INSERT_AT_LINE
    assert patch.content == "USER 1000:1000"


def test_fix_semgrep_returns_none_for_unknown_absence_rule(tmp_path):
    # No extra.fix, no rule in the YAML table — fall through to LLM.
    abs_path = _w(tmp_path, "Dockerfile", "FROM alpine\n")
    diag = {
        "error_code": "SEMGREP:some.totally.unknown.rule",
        "file": abs_path,
    }
    assert _fix_semgrep("rule", diag, str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Loader resilience
# ---------------------------------------------------------------------------

def test_malformed_yaml_does_not_crash_loader(tmp_path):
    # Garbage YAML at workspace root → loader logs + skips it,
    # shipped rules still load.
    _w(tmp_path, "security_fix_rules.yaml", "this: is\n  : not [valid")
    rules = load_rules(str(tmp_path))
    # Shipped defaults still present.
    assert ("SEMGREP", "dockerfile.security.missing-user.missing-user") in rules


def test_invalid_regex_skips_just_that_rule(tmp_path):
    _w(tmp_path, "security_fix_rules.yaml", """
rules:
  - scanner: SEMGREP
    rule: test.rule.broken
    description: bad regex
    operation: insert_at_line
    content: "X"
    before_regex: '['   # malformed
    fallback: none
  - scanner: SEMGREP
    rule: test.rule.good
    description: ok
    operation: insert_at_line
    content: "Y"
    before_regex: '^Z'
    fallback: none
""".strip())
    rules = load_rules(str(tmp_path))
    assert ("SEMGREP", "test.rule.broken") not in rules
    assert ("SEMGREP", "test.rule.good") in rules


def test_unsupported_operation_skipped(tmp_path):
    _w(tmp_path, "security_fix_rules.yaml", """
rules:
  - scanner: SEMGREP
    rule: test.rule.future_op
    description: not yet supported
    operation: delete_block
    content: "X"
    before_regex: '^Z'
""".strip())
    rules = load_rules(str(tmp_path))
    assert ("SEMGREP", "test.rule.future_op") not in rules


def test_pyyaml_missing_returns_empty_dict(monkeypatch):
    # Simulate pyyaml absent by making the import fail. load_rules
    # must return {} so the LLM path still works.
    import builtins
    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("forced for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    assert load_rules() == {}
