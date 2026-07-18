"""Test-author regeneration for declared-unsatisfiable tests (ADR-0001).

When the repair loop declares a test ``UNSATISFIABLE_TEST`` and the autonomy
ladder (``route_after_unsatisfiable``) elects regeneration over HITL, this
node rewrites the *one* defective test file **from the spec** — the
test-author phase, not the code-fixer — then hands control back to
``compiler_node`` to re-verify.

Safety (why this is not a reward-hack backdoor):
  * Authority is the SPEC, never "make the build green". The prompt asks for
    a faithful encoding of the acceptance criteria; the regenerated test is
    then run. If it now fails because the *production* code is wrong, that
    failure flows back to the repair loop (a real fix), not another rewrite.
  * The regeneration may only touch the declared test path — blocks targeting
    any other file are rejected before applying.
  * A mechanical **coverage-non-regression** gate rejects a rewrite that
    guts assertions to pass, and a **spec-citation** gate rejects one that
    doesn't name the requirement it aligned to.

The node clears ``node_state["unsatisfiable_test"]`` by simply not re-emitting
it (LangGraph replaces ``node_state`` wholesale), and records the attempt in
``loop_counter["test_regen_attempts"]`` (the persistent store) so the router's
per-test cap converges to HITL if regeneration doesn't resolve it.
"""

from __future__ import annotations

import ast
import os
import re
from typing import Any, Optional

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "test_regeneration_node",
    "count_test_functions",
    "count_assertion_sites",
    "coverage_nonregression_ok",
    "has_spec_citation",
    "patch_target_paths",
]

_REGEN_ATTEMPTS_KEY = "test_regen_attempts"

# A spec/requirement reference the regeneration must cite — story/feature/AC
# ids from the RSD model, or an explicit acceptance-criteria mention.
_SPEC_ID_RE = re.compile(
    r"\b(?:EPIC|FEAT|STORY|FR|NFR|AC)[-_ ]?\d+|acceptance\s+criteri|requirement|\bspec\b",
    re.IGNORECASE,
)

# `file:` line inside a patch DSL block (CREATE_FILE / REWRITE_FILE /
# REPLACE_BLOCK). Used to confirm the regeneration only touches the declared
# test path.
_BLOCK_FILE_RE = re.compile(r"^\s*file:\s*(?P<path>.+?)\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Pure gates (unit-testable without a gateway / graph)
# ---------------------------------------------------------------------------

def count_test_functions(source: str) -> int:
    """Number of ``test*`` functions/methods in ``source`` (0 on parse error)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                node.name.startswith("test"):
            n += 1
    return n


def count_assertion_sites(source: str) -> int:
    """Number of assertion sites — ``assert`` statements plus
    ``pytest.raises`` / ``assertRaises`` / ``assertEqual``-family calls
    (0 on parse error)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            n += 1
        elif isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name)
                else ""
            )
            if name in ("raises", "assertRaises", "assertRaisesRegex", "warns") \
                    or name.startswith("assert"):
                n += 1
    return n


def coverage_nonregression_ok(
    old_source: str, new_source: str,
) -> tuple[bool, str]:
    """Reject a regeneration that weakens coverage (the anti-reward-hack gate).

    A legitimate contradiction fix keeps both test functions and adjusts their
    assertions to be consistent, so we allow at most one test function to be
    dropped (the redundant half of a contradictory pair) and require the
    rewrite to keep the bulk of the assertion sites. A rewrite that guts the
    file to pass — no assertions, or a collapse below half the original — is
    rejected.
    """
    if not new_source.strip():
        return False, "regenerated test is empty"
    try:
        ast.parse(new_source)
    except SyntaxError as exc:
        return False, f"regenerated test does not parse ({exc.msg})"

    of, nf = count_test_functions(old_source), count_test_functions(new_source)
    oa, na = count_assertion_sites(old_source), count_assertion_sites(new_source)

    if na == 0:
        return False, "regenerated test has no assertions (gutted)"
    if nf < max(1, of - 1):
        return False, f"test-function count dropped {of}->{nf} (>1 removed)"
    if oa and na * 2 < oa:
        return False, f"assertion sites dropped {oa}->{na} (>50% weakened)"
    return True, f"functions {of}->{nf}, assertions {oa}->{na}"


def has_spec_citation(text: str) -> bool:
    """True if ``text`` references a requirement/story/AC — the regeneration
    must justify its interpretation against the spec, not the build result."""
    return bool(_SPEC_ID_RE.search(text or ""))


def patch_target_paths(patch_text: str) -> set[str]:
    """Workspace-relative paths named by ``file:`` lines in the patch DSL."""
    return {m.group("path") for m in _BLOCK_FILE_RE.finditer(patch_text or "")}


def _norm(path: str, workspace: str) -> str:
    from harness.graph import _normalize_ws_path
    return _normalize_ws_path(path, workspace)


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

_REGEN_SYSTEM = (
    "You are the TEST AUTHOR repairing ONE defective generated test file. The "
    "repair loop proved this test is unsatisfiable — no production change can "
    "make it pass as written. Your job is NOT to make a build green; it is to "
    "rewrite the test so it faithfully encodes the SPECIFICATION's intended "
    "behaviour.\n\n"
    "RULES:\n"
    "1. Emit exactly one <<<REWRITE_FILE>>> block for the named test file and "
    "nothing else. Touch no other file.\n"
    "2. Anchor every assertion to the spec. Cite the governing requirement / "
    "story / acceptance-criterion id (e.g. STORY-002, FR-014) in a comment.\n"
    "3. Do NOT weaken coverage to pass: keep the test functions and their "
    "assertions; only correct the ones that contradict the spec. Removing all "
    "assertions or deleting tests is forbidden.\n"
    "4. Resolve contradictions by the spec: if two tests demand opposite "
    "outcomes for the same input, keep the spec-consistent one and correct the "
    "other to match the spec.\n"
    "5. Write the test against intended behaviour. If the current production "
    "code then fails the corrected assertion, that is expected — the repair "
    "loop will fix the production code."
)


def build_regeneration_messages(
    *,
    system_spec: str,
    test_rel_path: str,
    test_source: str,
    unsat_reason: str,
    failing_output: str,
) -> list[dict[str, str]]:
    """Assemble the regeneration prompt. Kept pure for testability."""
    user = (
        f"## Defective test file: {test_rel_path}\n\n"
        f"### Why the repair loop declared it unsatisfiable\n{unsat_reason}\n\n"
        f"### Failing test output\n{failing_output[:4000]}\n\n"
        f"### Current test file content\n```\n{test_source}\n```\n\n"
        "Rewrite this ONE file so it encodes the specification's intended "
        "behaviour, citing the governing requirement id(s). Emit only the "
        "<<<REWRITE_FILE>>> block."
    )
    messages: list[dict[str, str]] = []
    if system_spec.strip():
        messages.append({"role": "system", "content": system_spec})
    messages.append({"role": "system", "content": _REGEN_SYSTEM})
    messages.append({"role": "user", "content": user})
    return messages


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

def _bump_attempt(loop_counter: dict[str, Any], rel: str) -> None:
    attempts = dict(loop_counter.get(_REGEN_ATTEMPTS_KEY, {}) or {})
    attempts[rel] = int(attempts.get(rel, 0)) + 1
    loop_counter[_REGEN_ATTEMPTS_KEY] = attempts


def _failing_output(state: dict[str, Any], rel: str, workspace: str) -> str:
    """Best-effort pytest tail for the declared test, from compiler_errors."""
    lines: list[str] = []
    for d in state.get("compiler_errors", []) or []:
        if not isinstance(d, dict):
            continue
        df = _norm(str(d.get("file", "") or ""), workspace)
        if not df or df == _norm(rel, workspace) or df.endswith(rel) or "<" in df:
            msg = str(d.get("message", "") or "")
            if msg:
                lines.append(msg)
    return "\n".join(lines)


async def test_regeneration_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: regenerate one declared-unsatisfiable test from spec,
    apply gates, write it, and route back to ``compiler_node`` to re-verify."""
    node_state = state.get("node_state", {}) or {}
    rel = str(node_state.get("unsatisfiable_test", "") or "")
    reason = str(node_state.get("unsatisfiable_test_reason", "") or "")
    workspace = str(state.get("workspace_path", "") or "")
    loop_counter = dict(state.get("loop_counter", {}) or {})
    cfg = state.get("test_regeneration_config", {}) or {}

    # Defensive: the router gates entry, but never trust that alone.
    if not rel:
        logger.warning("[test_regeneration_node] No unsatisfiable_test in state; no-op.")
        return {"loop_counter": loop_counter,
                "node_state": {"current_node": "test_regeneration"}}

    _bump_attempt(loop_counter, rel)

    def _give_up(status: str, detail: str) -> dict[str, Any]:
        # Clear the unsatisfiable flag (wholesale node_state replace) and let
        # compiler → route_after_compiler re-evaluate; the per-test attempt
        # cap converges the loop to HITL.
        logger.warning("[test_regeneration_node] %s: %s — deferring to ladder.",
                       status, detail)
        return {
            "loop_counter": loop_counter,
            "node_state": {
                "current_node": "test_regeneration",
                "test_regeneration": {"status": status, "detail": detail,
                                      "file": rel},
            },
        }

    abs_path = rel if os.path.isabs(rel) else os.path.join(workspace, rel)
    try:
        with open(abs_path, "r", encoding="utf-8") as fh:
            old_source = fh.read()
    except OSError as exc:
        return _give_up("read_failed", f"could not read {abs_path}: {exc}")

    from harness.graph import get_gateway
    gateway = get_gateway()
    if gateway is None:
        return _give_up("no_gateway", "no LLM gateway configured")

    from harness.gateway import NodeRole

    # Anchor the regeneration in the spec: the anchored system prompt
    # (messages[0]) carries the SRS for spec-driven builds.
    system_spec = ""
    for m in state.get("messages", []) or []:
        if isinstance(m, dict) and m.get("role") == "system" and \
                isinstance(m.get("content"), str):
            system_spec = m["content"]
            break

    messages = build_regeneration_messages(
        system_spec=system_spec,
        test_rel_path=rel,
        test_source=old_source,
        unsat_reason=reason,
        failing_output=_failing_output(state, rel, workspace),
    )

    budget = float(state.get("budget_remaining_usd", 2.00))
    token_tracker = state.get("token_tracker", {})
    try:
        response, budget = await gateway.dispatch(
            messages=messages, role=NodeRole.PATCHING,
            budget_remaining_usd=budget,
        )
    except Exception as exc:  # noqa: BLE001 — a gateway error must not crash the graph
        return _give_up("gateway_error", str(exc))
    token_tracker = gateway.aggregate_tokens(token_tracker, response.usage)
    content = response.content or ""

    # --- Gate 1: only the declared test path may be touched ---
    targets = {_norm(t, workspace) for t in patch_target_paths(content)}
    stray = {t for t in targets if t != _norm(rel, workspace)}
    if stray:
        return _give_up("targeted_other_files",
                        f"regeneration tried to touch {sorted(stray)}")
    if not targets:
        return _give_up("no_patch", "regeneration emitted no REWRITE_FILE block")

    # --- Gate 2: spec citation ---
    if cfg.get("require_spec_reference", True) and not has_spec_citation(content):
        return _give_up("no_spec_citation",
                        "regeneration did not cite a requirement/story/AC")

    # --- Gate 3: coverage non-regression (parse the proposed new content) ---
    from harness.patcher import process_llm_patch_output
    from harness.graph import _build_patcher_allowlist
    # Restrict the allowlist to the single declared test file — defence in
    # depth against a REWRITE that slipped a second block past Gate 1.
    allowed_paths = _build_patcher_allowlist(workspace)
    existing_modified = list(state.get("modified_files", []) or [])
    patch_results, new_modified = await process_llm_patch_output(
        content, workspace, existing_modified, allowed_paths=allowed_paths,
    )
    if cfg.get("coverage_nonregression", True):
        try:
            with open(abs_path, "r", encoding="utf-8") as fh:
                written = fh.read()
        except OSError as exc:
            return _give_up("reread_failed", str(exc))
        ok, detail = coverage_nonregression_ok(old_source, written)
        if not ok:
            # Roll back to the original — the gate rejected this rewrite.
            try:
                with open(abs_path, "w", encoding="utf-8") as fh:
                    fh.write(old_source)
            except OSError:
                logger.error("[test_regeneration_node] rollback write failed for %s", abs_path)
            return _give_up("coverage_regression", detail)

    applied = sum(1 for r in patch_results if getattr(r, "success", False))
    logger.warning(
        "[test_regeneration_node] Regenerated %s (attempt %d): %d block(s) "
        "applied. Routing to compiler to re-verify.",
        rel, loop_counter[_REGEN_ATTEMPTS_KEY][rel], applied,
    )

    messages_out = list(state.get("messages", []) or [])
    messages_out.append({"role": "assistant", "content": content})
    return {
        "messages": messages_out,
        "modified_files": new_modified,
        "token_tracker": token_tracker,
        "budget_remaining_usd": budget,
        "loop_counter": loop_counter,
        # unsatisfiable_test intentionally omitted → cleared on wholesale
        # node_state replace, so the recompile is judged on its own merits.
        "node_state": {
            "current_node": "test_regeneration",
            "test_regeneration": {
                "status": "regenerated", "file": rel,
                "attempt": loop_counter[_REGEN_ATTEMPTS_KEY][rel],
            },
        },
    }
