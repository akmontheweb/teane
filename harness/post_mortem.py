"""
Automated failure post-mortems — the HITL → learning loop.

When a HITL breakpoint fires (or a run ends non-zero without ever reaching
HITL), the harness distills the failure into a one-line durable rule and
appends it to the per-repo memory file via ``append_session_note``'s
``extra_notes`` field. Repo memory is auto-injected into the next run's
planner prompt, so the same failure class stops needing a human:

    [learned-rule:repair_loop_limit] fp=3fa9c02d1e — Hypothesis from failed
    run 7e4cba32: pytest collection fails until FastAPI test fixtures pin
    httpx<0.28; pin it in requirements.txt in the first patch round.

Design invariants:
    - NEVER blocks or breaks the run. Every entry point is fail-open; the
      deterministic template guarantees a rule even with no LLM available.
    - Rules are HYPOTHESES, framed as such, single-line, length-capped, and
      heading-stripped so LLM output cannot forge a ``## Session`` section
      in the memory file (the FIFO trimmer and injector are section-based).
    - Bounded blast radius: repo memory is FIFO-trimmed (100KB) and
      injection-capped (8KB); duplicates are fingerprint-deduped; and any
      clean (exit 0) run retires every active rule — a green run is proof
      the failure class no longer bites, and a stale rule that outlives its
      failure is pure poison risk.
    - Bounded cost: one cheap JUDGMENT-role call per session, allowed to
      run on a small synthetic budget floor even when the session budget is
      exhausted (``budget_exhausted`` is a key trigger — the lesson must
      still be recorded).
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

RULE_TAG_PREFIX = "[learned-rule:"
RETIRED_TAG_PREFIX = "[learned-rule(retired):"

_RULE_MAX_CHARS = 600
_WS_RE = re.compile(r"\s+")
_DIGITS_RE = re.compile(r"\d+")

# Forward-looking advice per HITL trigger prefix (the taxonomy from
# graph._infer_hitl_trigger; parametrised triggers like
# ``env_misconfig:pytest`` match on the prefix before ':').
_TRIGGER_ADVICE: dict[str, str] = {
    "repair_loop_limit": (
        "the repair loop hit its iteration cap without converging on these "
        "errors. Address these error classes in the initial patch plan "
        "instead of relying on repair iterations"
    ),
    "persistent_build_failure": (
        "the build/test command kept failing to the end of the session. "
        "Verify the build command and fix these error classes first"
    ),
    "budget_exhausted": (
        "the session spent its full budget before converging. Plan smaller "
        "batches and fix the dominant error class before anything else"
    ),
    "no_progress_failsafe": (
        "multiple rounds burned budget without changing the diagnostic set. "
        "Re-read the failing files before patching instead of retrying "
        "near-identical patches"
    ),
    "zero_patch_loop": (
        "consecutive rounds produced no applicable patch blocks. Read the "
        "current file contents before emitting SEARCH/REPLACE blocks"
    ),
    "all_allowlist_rejected": (
        "every emitted patch targeted files outside the patcher allowlist. "
        "Keep changes inside the detected source root and root-allowlisted "
        "files"
    ),
    "env_misconfig": (
        "a required tool or dependency was missing from the environment. "
        "Declare it in the dependency manifest in the first patch round"
    ),
    "llm_behavior": (
        "the test-generation LLM refused to emit valid patch blocks (either "
        "exhausted its real-iteration cap or its zero-emit re-prompt sub-cap). "
        "Ensure the story ships enough source context that the model has "
        "something concrete to test, or narrow the batch so the eligible "
        "source set is unambiguous"
    ),
    "build_command_cd_missing": (
        "the build command referenced a directory that does not exist. "
        "Align the build command with the actual workspace layout"
    ),
    "security_fix_limit": (
        "security-scan findings persisted past the fix cap. Address the "
        "flagged patterns at design time rather than post-hoc"
    ),
    "reflection_distraction_loop": (
        "repair rounds kept drifting to unrelated edits. Fix only the "
        "diagnostics listed in the error summary"
    ),
    "low_signal_verdict_loop": (
        "repeated rounds produced low-signal verdicts. Prefer decisive, "
        "minimal patches over exploratory edits"
    ),
    "traceability_block": (
        "patches could not be traced to spec requirements. Keep changes "
        "mapped to SPEC_REQUIREMENTS.md IDs"
    ),
    "decomposition_validation_failed": (
        "story decomposition failed validation. Simplify the story split "
        "or fix SPEC_REQUIREMENTS.md structure first"
    ),
    "decomposition_missing": (
        "story mode ran without a usable decomposition. Generate or repair "
        "the decomposition before patching"
    ),
    # Post-finsearch-156032347 additions. See the "review teane web for
    # updates" thread — before these entries the 5 new HITL prefixes fell
    # through to _GENERIC_ADVICE, defeating the labeling work in
    # _infer_hitl_trigger.
    "replace_block_stuck": (
        "a single file racked up three consecutive REPLACE_BLOCK misses "
        "even after the automatic REWRITE_FILE recovery round. The LLM's "
        "mental model of that file has drifted beyond surgical repair. "
        "Emit a fresh READ_FILE against the file before any subsequent "
        "edit — the drift signals the previous SEARCH windows have gone "
        "stale"
    ),
    "no_progress_repairs": (
        "the repair loop hit its per-cap non-progress budget — enough "
        "consecutive rounds shrunk neither the fingerprint set nor the "
        "raw diagnostic count that the harness declared the loop stalled. "
        "Fix the dominant error class in a single early patch rather than "
        "iterating; when a diagnostic won't shrink on plausible edits, "
        "the LLM is patching the wrong file or the wrong layer"
    ),
    "hard_iteration_ceiling": (
        "the repair loop ran to the hard total-iteration ceiling "
        "(max_patch_repair_iterations × total_hard_cap_multiplier) while "
        "still showing per-round progress signals. The batch is too broad "
        "for one repair loop to finish — split the story or narrow the "
        "batch so each verification chain has fewer failing fingerprints "
        "to converge on"
    ),
    "same_missing_dep": (
        "the same missing dependency recurred past the autofix bypass "
        "cap. If the symbol is a bootstrap tool (pip / make / a system "
        "package) it belongs in sandbox.docker_image; if it's a regular "
        "pip / npm package the workspace-manifest topology is likely "
        "mismatched — the build_command should install from the manifest "
        "that actually contains the package"
    ),
    "build_command_blocked": (
        "the sandbox CommandValidator refused the build command because "
        "a leading token (cd / bash / etc.) is not in security."
        "allowed_commands. Repair rounds cannot amend the global "
        "validator config; adjust the policy or rewrite the build "
        "command to use only whitelisted primitives"
    ),
}

_GENERIC_ADVICE = (
    "the run failed in a way the harness could not classify. Review the "
    "session log before rerunning with the same prompt"
)


def _trigger_prefix(trigger: str) -> str:
    return (trigger or "unknown").split(":", 1)[0].strip() or "unknown"


def _top_errors(state: Mapping[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    errors = [
        e for e in (state.get("compiler_errors") or [])
        if isinstance(e, dict)
        and str(e.get("severity", "error")).lower() != "warning"
    ]
    return errors[:limit]


# Signature-specific rules. Each entry pairs a regex that matches a
# well-known assertion-shape or diagnostic-body against a canned
# forward-looking rule. When ``deterministic_rule`` sees a top-diagnostic
# message that matches one of these signatures, it prepends the canned
# rule to the trigger-taxonomy fallback so the learned-rules memory
# carries the SPECIFIC advice for the observed failure class, not just
# the generic trigger advice.
#
# The finsearch session 156032347 400-vs-422 oscillation (5+ repair
# rounds, 1 HITL trip on ``server/app/tests/test_company_api.py::test_
# search_returns_400_on_empty_query``) is the canonical case: FastAPI /
# Pydantic returns 422 for schema-invalid input, the test asserted 400,
# and neither side would give. The signature rule tells the next run's
# planner to either (a) install a global exception handler that maps
# ValidationError → 400 in the endpoint contract, or (b) update the
# test to assert 422 — whichever matches the SPEC_REQUIREMENTS.md
# contract for that endpoint.
_SIGNATURE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"assert\s+(?:400\s*==\s*422|422\s*==\s*400)|"
            r"AssertionError.*(?:400\s*==\s*422|422\s*==\s*400)",
            re.IGNORECASE,
        ),
        "For FastAPI endpoints, Pydantic ValidationError yields HTTP 422 "
        "by default, not 400 — a repair loop that oscillates on `assert "
        "400 == 422` (or the reverse) is fighting this default. Pick "
        "ONE contract at planning time: either (a) install a global "
        "exception_handler(RequestValidationError) that returns 400 on "
        "the endpoint(s) whose SPEC_REQUIREMENTS.md contract mandates "
        "400 for invalid input, OR (b) change the test to assert 422 "
        "when the spec doesn't mandate 400. Do NOT loop the repair "
        "node on the same assertion — it can't converge without the "
        "contract decision.",
    ),
)


def _match_signature_rule(state: Mapping[str, Any]) -> Optional[str]:
    """When any of the top diagnostics matches a well-known assertion
    signature, return the canned forward-looking rule for that class.
    None otherwise — caller falls through to the trigger-taxonomy
    advice.
    """
    for err in _top_errors(state, limit=5):
        msg = str(err.get("message") or "")
        if not msg:
            continue
        for pattern, rule in _SIGNATURE_RULES:
            if pattern.search(msg):
                return rule
    return None


def deterministic_rule(trigger: str, state: Mapping[str, Any]) -> str:
    """Template rule from the trigger taxonomy + top diagnostics.

    The no-LLM floor of the learning loop: always returns a non-empty,
    single-line, forward-looking rule. When a well-known assertion-shape
    signature matches one of the top diagnostics (e.g. FastAPI 400-vs-
    422), the signature-specific rule leads and the trigger-taxonomy
    advice becomes context.
    """
    signature_rule = _match_signature_rule(state)
    prefix = _trigger_prefix(trigger)
    advice = _TRIGGER_ADVICE.get(prefix, _GENERIC_ADVICE)
    detail = ""
    if ":" in (trigger or "") and prefix in (
        "env_misconfig", "llm_behavior", "build_command_cd_missing",
        # Post-finsearch-156032347: the new label taxonomy carries
        # actionable detail in the suffix — file path, cap ratio,
        # missing symbol, validator rule. Dropping it here would leave
        # the learned rule generic exactly where it should be specific.
        "replace_block_stuck", "no_progress_repairs",
        "hard_iteration_ceiling", "same_missing_dep",
        "build_command_blocked",
    ):
        detail = f" ({trigger.split(':', 1)[1]})"
    errs = _top_errors(state)
    err_part = ""
    if errs:
        codes = ", ".join(
            f"{e.get('error_code') or 'error'}: "
            f"{_WS_RE.sub(' ', str(e.get('message') or '')).strip()[:80]}"
            for e in errs
        )
        err_part = f" Top errors: {codes}."
    build_cmd = str(state.get("build_command") or "").strip()
    cmd_part = f" Build command: `{build_cmd}`." if build_cmd else ""
    if signature_rule:
        return sanitize_rule(
            f"{signature_rule} Context: {advice}{detail}."
            f"{err_part}{cmd_part}"
        )
    return sanitize_rule(
        f"In this repo, {advice}{detail}.{err_part}{cmd_part}"
    )


def sanitize_rule(text: str, *, max_chars: int = _RULE_MAX_CHARS) -> str:
    """Make LLM output safe to embed in the memory file.

    Strips code fences, drops markdown-heading lines (a rule must not be
    able to forge a ``## Session`` section — the trimmer and injector are
    section-based), collapses to a single line, and caps length.
    """
    from harness.trust import strip_code_fences
    text = strip_code_fences(text or "")
    lines = [
        ln for ln in text.splitlines()
        if not ln.lstrip().startswith("#")
    ]
    flat = _WS_RE.sub(" ", " ".join(lines)).strip()
    if len(flat) > max_chars:
        flat = flat[: max_chars - 1].rstrip() + "…"
    return flat


def rule_fingerprint(trigger: str, compiler_errors: list[dict[str, Any]]) -> str:
    """Stable identity of a failure class: trigger prefix + top error codes
    + the digit-stripped first message. Used for dedupe across sessions."""
    errors = [e for e in (compiler_errors or []) if isinstance(e, dict)]
    codes = sorted({str(e.get("error_code") or "") for e in errors[:5]})
    first_msg = ""
    if errors:
        first_msg = _WS_RE.sub(
            " ", _DIGITS_RE.sub("", str(errors[0].get("message") or ""))
        ).strip().lower()[:120]
    payload = "|".join([_trigger_prefix(trigger), *codes, first_msg])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def format_rule_note(
    trigger: str, rule_text: str, fingerprint: str, session_id: str,
) -> str:
    short_id = session_id.split("-")[0] if "-" in session_id else session_id
    return (
        f"{RULE_TAG_PREFIX}{trigger}] fp={fingerprint} — "
        f"Hypothesis from failed run {short_id}: {rule_text}"
    )


_NOTE_RE = re.compile(
    re.escape(RULE_TAG_PREFIX) + r"(?P<trigger>[^\]]+)\] fp=(?P<fp>[0-9a-f]+)"
)


def parse_rule_note(note: str) -> Optional[tuple[str, str]]:
    """Extract ``(trigger, fingerprint)`` from a formatted rule note, or
    None when the note doesn't carry the tag (defensive — the cli dedupe
    must not crash on a malformed staged note)."""
    m = _NOTE_RE.search(note or "")
    if not m:
        return None
    return m.group("trigger"), m.group("fp")


def already_recorded(memory_text: str, trigger: str, fingerprint: str) -> bool:
    """True when an ACTIVE rule with this trigger+fingerprint already exists.

    Retired rules don't count: if the same failure class recurs after a
    clean run retired its rule, the lesson is evidently still needed and
    gets re-recorded.
    """
    if not memory_text:
        return False
    needle = f"{RULE_TAG_PREFIX}{trigger}] fp={fingerprint}"
    return needle in memory_text


def _build_post_mortem_prompt(
    state: Mapping[str, Any],
    trigger: str,
    escalation_summary: Optional[str],
) -> str:
    parts: list[str] = [
        "A fully-autonomous coding-agent run on this repository failed and "
        "escalated to a human. Distill ONE forward-looking rule (max 2 "
        "sentences, plain prose, no markdown, no headings) that a future "
        "run on this repo should follow to avoid the same failure. State "
        "the rule as actionable guidance, not a description of what "
        "happened.",
        f"Failure trigger: {trigger}",
    ]
    build_cmd = str(state.get("build_command") or "").strip()
    if build_cmd:
        parts.append(f"Build command: {build_cmd}")
    errs = _top_errors(state, limit=5)
    if errs:
        lines = [
            f"- {e.get('file')}:{e.get('line')} "
            f"[{e.get('error_code') or 'error'}] {str(e.get('message') or '')[:200]}"
            for e in errs
        ]
        parts.append("Top diagnostics:\n" + "\n".join(lines))
    node_state = state.get("node_state") or {}
    rejections = node_state.get("allowlist_rejections") or []
    if rejections:
        parts.append(f"Allowlist rejections this session: {len(rejections)}")
    patch_failures = node_state.get("patch_failures") or []
    if patch_failures:
        parts.append(f"Patch application failures this session: {len(patch_failures)}")
    if escalation_summary:
        parts.append(
            "Grounded escalation summary (already verified against the "
            f"workspace):\n{escalation_summary[:1500]}"
        )
    parts.append("Respond with the rule only.")
    return "\n\n".join(parts)


async def generate_post_mortem(
    state: Mapping[str, Any],
    *,
    trigger: str,
    escalation_summary: Optional[str],
    config: dict[str, Any],
) -> tuple[str, float]:
    """Produce a formatted ``[learned-rule:...]`` note for this failure.

    Returns ``(note, llm_cost_usd)``. The note is never empty: when the
    JUDGMENT call is unavailable or fails, the deterministic template rule
    is used. The LLM call runs on ``max(actual_budget, max_cost_usd)`` — a
    synthetic floor (default $0.10) so the lesson still gets distilled when
    the session budget is exhausted; the gateway's own tracker records the
    real spend either way.
    """
    session_id = str(state.get("session_id") or "unknown")
    fingerprint = rule_fingerprint(trigger, state.get("compiler_errors") or [])
    rule_text = ""
    cost = 0.0
    try:
        from harness.graph import _maybe_judgment_llm  # deferred: avoids import cycle
        actual_budget = float(state.get("budget_remaining_usd") or 0.0)
        floor = float(config.get("max_cost_usd", 0.10))
        synthetic_budget = max(actual_budget, floor)
        raw, new_budget = await _maybe_judgment_llm(
            prompt=_build_post_mortem_prompt(state, trigger, escalation_summary),
            budget_remaining_usd=synthetic_budget,
            purpose="post_mortem",
            enabled=bool(config.get("enabled", True)),
        )
        cost = max(0.0, synthetic_budget - new_budget)
        if raw:
            rule_text = sanitize_rule(raw)
    except Exception as exc:  # noqa: BLE001 — the loop must never no-op on infra trouble
        logger.warning("[post_mortem] LLM distillation failed (%s); using template.", exc)
    if not rule_text:
        rule_text = deterministic_rule(trigger, state)
    return format_rule_note(trigger, rule_text, fingerprint, session_id), cost


def retire_learned_rules(workspace_path: str, cfg: Any = None) -> int:
    """Mark every active learned rule in this repo's memory as retired.

    Called after a clean (exit 0) run: a green run is proof the recorded
    failure classes no longer bite, and keeping their rules active would
    poison future planner prompts. The text stays in the file
    (retired-tagged, greppable) for forensics. Returns the retire count.
    """
    from harness.repo_memory import (
        RepoMemoryConfig,
        _atomic_write_text,
        _memory_file_lock,
        memory_file_path,
    )
    import os

    cfg = cfg or RepoMemoryConfig()
    path = memory_file_path(workspace_path, cfg)
    if not os.path.isfile(path):
        return 0
    try:
        with _memory_file_lock(path):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            count = text.count(RULE_TAG_PREFIX)
            if not count:
                return 0
            _atomic_write_text(
                path, text.replace(RULE_TAG_PREFIX, RETIRED_TAG_PREFIX)
            )
    except OSError as exc:
        logger.warning("[post_mortem] retire failed for %s: %s", path, exc)
        return 0
    logger.info("[post_mortem] retired %d learned rule(s) in %s", count, path)
    return count
