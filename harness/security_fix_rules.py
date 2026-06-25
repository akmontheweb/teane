"""
Layer-2 security autofix rule table.

Reads ``harness/security_fix_rules.yaml`` (shipped default) plus an
optional ``{workspace}/security_fix_rules.yaml`` override and translates
scanner findings whose rule_id is on file into deterministic PatchBlocks
— without ever entering the LLM repair loop.

This is the bounded follow-up to the line-coordinate patcher primitives
landed earlier. Layer 1 (semgrep ``extra.fix`` consumption) covers any
rule whose author shipped an autofix template; Layer 2 covers the
"absence-based" rules that don't have one — e.g. ``missing-user``,
``missing-workdir``, ``missing-healthcheck`` — where the fix is
well-known but the scanner has no anchor string to give.

Architecture:

  1. ``load_rules(workspace_path)`` walks two YAML files (workspace
     override then shipped default), validates each rule's shape, and
     returns a ``dict[(scanner, rule_id_lower), Rule]``. Workspace
     rules win on collision.
  2. ``apply_rule(rule, diag, workspace_path)`` reads the target file,
     respects ``idempotency_regex``, locates the insertion point from
     ``before_regex`` (with ``fallback``), hash-snapshots the file, and
     returns an ``INSERT_AT_LINE`` PatchBlock pinned to that hash.
  3. ``try_rule_table_fix(diag, workspace_path)`` is the convenience
     entry point used by ``harness.autofix._fix_semgrep`` —
     ``Optional[PatchBlock]``, fail-open on any I/O / parse error.

Fail-open contract: every failure path returns ``None`` (or an empty
rule set) so a malformed YAML, a missing pyyaml installation, or a
broken regex never breaks the repair loop. The diagnostic still flows
to the LLM exactly as it did before this module existed.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from harness.patcher import OperationType, PatchBlock, sha256_file_bytes

logger = logging.getLogger(__name__)


HARNESS_RULES_FILE = os.path.join(
    os.path.dirname(__file__), "security_fix_rules.yaml",
)

# Whitelist of fallback modes the loader accepts. Anything else falls
# back to "none" so a misconfigured YAML can't smuggle in a surprise
# at_end append on a rule that shouldn't have one.
_VALID_FALLBACKS: frozenset[str] = frozenset({"none", "at_end", "at_start"})


@dataclass(frozen=True)
class Rule:
    """One parsed rule from the YAML table.

    Attributes:
        scanner: Uppercase scanner identifier (e.g. ``"SEMGREP"``).
        rule_id: The scanner's rule identifier, lowercased for
            case-insensitive matching.
        description: One-line note logged when the rule fires.
        content: Lines to splice in. No trailing newline expected;
            ``apply_rule`` adds one.
        before_regex: Compiled regex. The patch inserts the content on
            the line BEFORE the first match.
        fallback: ``"at_end"``, ``"at_start"``, or ``"none"`` (default)
            — what to do when ``before_regex`` finds no match.
        idempotency_regex: Compiled regex. When any line of the target
            file matches, the rule is considered already satisfied and
            no patch is emitted.
    """
    scanner: str
    rule_id: str
    description: str
    content: str
    before_regex: re.Pattern[str]
    fallback: str
    idempotency_regex: Optional[re.Pattern[str]]


def load_rules(workspace_path: Optional[str] = None) -> dict[tuple[str, str], Rule]:
    """Return the merged rule table (workspace override wins over shipped).

    The result is keyed by ``(scanner_upper, rule_id_lower)`` so
    ``apply_rule`` can look up by the same normalisation the
    diagnostic carries. Returns an empty dict when YAML loading fails
    or pyyaml is unavailable — Layer 2 then short-circuits and the
    diagnostic continues to the LLM.
    """
    try:
        import yaml
    except ImportError:
        logger.warning(
            "[security_fix_rules] pyyaml not installed — Layer-2 rule "
            "table disabled. Falling back to LLM repair loop for "
            "absence-based scanner findings."
        )
        return {}

    candidates: list[str] = []
    if workspace_path:
        candidates.append(os.path.join(workspace_path, "security_fix_rules.yaml"))
    candidates.append(HARNESS_RULES_FILE)

    merged: dict[tuple[str, str], Rule] = {}
    # Iterate in REVERSE so the workspace override gets to overwrite
    # the shipped default. (The shipped default is loaded first, then
    # the workspace one stomps on it.)
    for path in reversed(candidates):
        rules = _load_one_yaml(path, yaml)
        for r in rules:
            merged[(r.scanner, r.rule_id)] = r
    return merged


def _load_one_yaml(path: str, yaml_mod) -> list[Rule]:
    """Read and validate a single YAML file. Returns an empty list on
    any error so a malformed override never kills the rule pipeline.
    """
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml_mod.safe_load(f)
    except (OSError, yaml_mod.YAMLError) as exc:
        logger.warning(
            "[security_fix_rules] Could not load %s: %s. Skipping.", path, exc,
        )
        return []
    if not isinstance(raw, dict):
        return []
    rules_raw = raw.get("rules")
    if not isinstance(rules_raw, list):
        return []

    out: list[Rule] = []
    for idx, item in enumerate(rules_raw):
        rule = _parse_rule(item, path, idx)
        if rule is not None:
            out.append(rule)
    return out


def _parse_rule(item: object, path: str, idx: int) -> Optional[Rule]:
    """Validate a single dict against the rule schema. Returns ``None``
    when any required field is missing, malformed, or the regex doesn't
    compile — logs the problem and continues so one bad rule doesn't
    sink the rest of the table.
    """
    if not isinstance(item, dict):
        logger.warning("[security_fix_rules] %s[rule #%d]: not a dict; skipped.", path, idx)
        return None
    scanner = str(item.get("scanner", "") or "").strip().upper()
    rule_id = str(item.get("rule", "") or "").strip().lower()
    operation = str(item.get("operation", "") or "").strip().lower()
    if not scanner or not rule_id:
        logger.warning(
            "[security_fix_rules] %s[rule #%d]: missing scanner or rule; skipped.",
            path, idx,
        )
        return None
    if operation != "insert_at_line":
        # Forward-compat slot: a future revision can add
        # replace_line_range support, but for now we accept only the
        # one shape that covers absence-based findings.
        logger.warning(
            "[security_fix_rules] %s[rule #%d]: unsupported operation %r; skipped.",
            path, idx, operation,
        )
        return None
    content = str(item.get("content", "") or "").rstrip("\n")
    if not content:
        logger.warning(
            "[security_fix_rules] %s[rule #%d]: empty content; skipped.",
            path, idx,
        )
        return None
    before_raw = str(item.get("before_regex", "") or "")
    if not before_raw:
        logger.warning(
            "[security_fix_rules] %s[rule #%d]: insert_at_line requires before_regex; skipped.",
            path, idx,
        )
        return None
    try:
        before_regex = re.compile(before_raw, re.MULTILINE)
    except re.error as exc:
        logger.warning(
            "[security_fix_rules] %s[rule #%d]: before_regex %r failed to "
            "compile (%s); skipped.", path, idx, before_raw, exc,
        )
        return None
    fallback = str(item.get("fallback", "none") or "none").strip().lower()
    if fallback not in _VALID_FALLBACKS:
        logger.warning(
            "[security_fix_rules] %s[rule #%d]: invalid fallback %r; coerced to 'none'.",
            path, idx, fallback,
        )
        fallback = "none"
    idem_raw = str(item.get("idempotency_regex", "") or "")
    idem_regex: Optional[re.Pattern[str]] = None
    if idem_raw:
        try:
            idem_regex = re.compile(idem_raw, re.MULTILINE)
        except re.error as exc:
            logger.warning(
                "[security_fix_rules] %s[rule #%d]: idempotency_regex %r "
                "failed to compile (%s); falling back to no idempotency check.",
                path, idx, idem_raw, exc,
            )
    description = str(item.get("description", "") or "").strip()

    return Rule(
        scanner=scanner,
        rule_id=rule_id,
        description=description,
        content=content,
        before_regex=before_regex,
        fallback=fallback,
        idempotency_regex=idem_regex,
    )


def apply_rule(
    rule: Rule,
    diag: dict[str, object],
    workspace_path: str,
) -> Optional[PatchBlock]:
    """Translate a matching rule + diagnostic into a hash-pinned PatchBlock.

    Returns ``None`` when the rule cannot deterministically produce a
    patch — file missing, idempotency check passes, ``before_regex``
    misses with ``fallback: none``, or the diagnostic's file path
    escapes the workspace. The caller (``_fix_semgrep``) then falls
    through to the LLM repair loop.
    """
    from harness.autofix import _relative_to_workspace  # local — avoid cycle

    file_raw = str(diag.get("file", "") or "")
    if not file_raw:
        return None
    rel_file = _relative_to_workspace(file_raw, workspace_path)
    if rel_file is None:
        return None
    abs_file = os.path.join(workspace_path, rel_file)
    if not os.path.isfile(abs_file):
        return None

    try:
        with open(abs_file, "r", encoding="utf-8") as f:
            body = f.read()
    except OSError:
        return None

    # Idempotency — if the rule's "fix already in place" pattern matches
    # ANY line, do nothing. Lets one rule cover both freshly-built
    # Dockerfiles and ones a prior round already patched.
    if rule.idempotency_regex is not None and rule.idempotency_regex.search(body):
        logger.info(
            "[security_fix_rules] %s/%s: idempotency check matched %s; "
            "no patch emitted.",
            rule.scanner, rule.rule_id, rel_file,
        )
        return None

    insert_line = _resolve_insert_line(body, rule)
    if insert_line is None:
        return None

    file_hash = sha256_file_bytes(abs_file) or ""
    logger.info(
        "[security_fix_rules] %s/%s firing on %s at line %d.",
        rule.scanner, rule.rule_id, rel_file, insert_line,
    )
    return PatchBlock(
        operation=OperationType.INSERT_AT_LINE,
        file=rel_file,
        line=insert_line,
        content=rule.content,
        expected_file_hash=file_hash,
    )


def _resolve_insert_line(body: str, rule: Rule) -> Optional[int]:
    """Compute the 1-based line number at which the rule should insert.

    Tries ``before_regex`` first; falls back per ``rule.fallback``.
    Returns ``None`` when the fallback is ``"none"`` and the regex
    found no match — in which case the LLM repair loop is the right
    next step.
    """
    lines = body.splitlines()
    for idx, line in enumerate(lines):
        if rule.before_regex.search(line):
            return idx + 1  # convert to 1-based; INSERT_AT_LINE inserts BEFORE this line

    if rule.fallback == "at_end":
        # len(lines) + 1 means "append" per the patcher's semantics.
        return len(lines) + 1
    if rule.fallback == "at_start":
        return 1
    return None


def try_rule_table_fix(
    diag: dict[str, object],
    workspace_path: str,
    *,
    scanner: str,
) -> Optional[PatchBlock]:
    """Look up ``(scanner, rule_id)`` in the rule table and dispatch.

    Convenience entry point for the scanner-specific autofix
    dispatchers in ``harness.autofix``. ``scanner`` is supplied by the
    caller so each dispatcher (``_fix_semgrep``, future ``_fix_bandit``
    Layer-2 extension, …) only sees its own rules.
    """
    error_code = str(diag.get("error_code", "") or "")
    if ":" not in error_code:
        return None
    _, _, rule_raw = error_code.partition(":")
    rule_id = rule_raw.strip().lower()
    if not rule_id:
        return None

    table = load_rules(workspace_path)
    rule = table.get((scanner.upper(), rule_id))
    if rule is None:
        return None
    return apply_rule(rule, diag, workspace_path)
