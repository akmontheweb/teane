"""Test-author regeneration for declared-unsatisfiable tests (ADR-0001).

When the repair loop declares a test ``UNSATISFIABLE_TEST`` and the autonomy
ladder (``route_after_unsatisfiable``) elects regeneration over HITL, this
node rewrites the *one* defective test file as a comprehensive unit suite for
the code module it maps to (1:1), **anchored on that module's contract** — the
test-author phase, not the code-fixer — then hands control back to
``compiler_node`` to re-verify.

Source of truth (teane's model: unit tests link to CODE, never to stories/ACs):
  1. the code module under test (found via the ``# @tests: <source>`` marker) —
     its signatures / docstrings / validators define intended behaviour;
  2. the sibling passing tests in the file;
  3. the SRS as a *tiebreaker only* for genuinely ambiguous intent — never
     cited in the test.

Safety (why this is not a reward-hack backdoor):
  * Authority is the code CONTRACT, never "make the build green". If the
    corrected assertion then fails because the *production* code is wrong, that
    failure flows back to the repair loop (a real fix), not another rewrite.
  * The regeneration may only touch the declared test path — blocks targeting
    any other file are rejected.
  * A mechanical **coverage-non-regression** gate rejects a rewrite that guts
    assertions to pass, and a **code-linkage** gate rejects one that drops its
    ``# @tests:`` marker. A public-symbol coverage advisory drives toward an
    exhaustive per-module suite.

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
    "has_code_linkage",
    "public_symbols",
    "symbol_coverage",
    "patch_target_paths",
]

_REGEN_ATTEMPTS_KEY = "test_regen_attempts"

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


def has_code_linkage(test_source: str) -> bool:
    """True if the regenerated test carries a ``# @tests: <source>`` marker.

    Build/patch unit tests link to the CODE under test, never to stories/ACs
    (teane's traceability model). The regeneration must preserve that 1:1
    code linkage — it is the anchor, not an SRS citation.
    """
    from harness.test_generation import _parse_tests_marker
    return bool(_parse_tests_marker(test_source))


def public_symbols(source: str) -> list[str]:
    """Public top-level functions/classes defined in a code module — the
    surface a comprehensive unit suite should exercise (empty on parse error)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and not node.name.startswith("_"):
            out.append(node.name)
    return out


def symbol_coverage(
    test_source: str, symbols: list[str],
) -> tuple[list[str], list[str]]:
    """Split ``symbols`` into (referenced, unreferenced) by the test source —
    a coarse completeness signal toward exhaustive per-module coverage."""
    covered, uncovered = [], []
    for sym in symbols:
        if re.search(rf"\b{re.escape(sym)}\b", test_source or ""):
            covered.append(sym)
        else:
            uncovered.append(sym)
    return covered, uncovered


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
    "You are the TEST AUTHOR rewriting ONE defective unit-test file. The repair "
    "loop proved it unsatisfiable — no production change can make it pass as "
    "written. Your job is NOT to make a build green; it is to regenerate a "
    "correct, COMPREHENSIVE unit-test suite for the ONE code module this file "
    "maps to (1:1), anchored on that module's CONTRACT.\n\n"
    "SOURCE OF TRUTH (in order):\n"
    "  1. The CODE UNDER TEST below — its signatures, docstrings, validators "
    "and type hints define the intended behaviour. This is a UNIT test: anchor "
    "on the code, NOT on requirements documents.\n"
    "  2. The sibling passing tests in the file — they show the established "
    "contract to preserve.\n"
    "  3. The specification is a TIEBREAKER only, used to reason about intent "
    "when the code contract is genuinely ambiguous. Never CITE a story/FR/AC "
    "id in the test — unit tests link to code, not to acceptance criteria.\n\n"
    "RULES:\n"
    "1. Emit exactly one <<<REWRITE_FILE>>> block for the named test file and "
    "nothing else. Touch no other file.\n"
    "2. Keep the `# @tests: <source path>` marker at the top — it records the "
    "1:1 code linkage. Do NOT add @verifies / STORY / AC references.\n"
    "3. Cover the module comprehensively: exercise every public function and "
    "class listed below across the meaningful permutations of their inputs "
    "(valid, boundary, and error cases). Aim for an exhaustive suite, not a "
    "token test.\n"
    "4. Do NOT weaken coverage to pass. Removing assertions or deleting tests "
    "to go green is forbidden.\n"
    "5. Resolve contradictions by the code contract: if two tests demand "
    "opposite outcomes for the identical call, keep the one consistent with the "
    "code's contract and correct the other to match it.\n"
    "6. If the corrected assertion then fails against the current production "
    "code, that is EXPECTED — the repair loop will fix the production code."
)


def build_regeneration_messages(
    *,
    test_rel_path: str,
    test_source: str,
    code_module_path: str,
    code_module_source: str,
    module_symbols: list[str],
    unsat_reason: str,
    failing_output: str,
    spec_tiebreaker: str = "",
) -> list[dict[str, str]]:
    """Assemble the regeneration prompt, code-contract first. Pure/testable."""
    symbols = ", ".join(module_symbols) if module_symbols else "(none detected)"
    parts = [
        f"## Regenerate the unit tests for module: {code_module_path}\n",
        f"### Code under test ({code_module_path})\n```\n{code_module_source[:12000]}\n```\n",
        f"### Public symbols to cover comprehensively\n{symbols}\n",
        f"### Defective test file: {test_rel_path}\n"
        f"(keep its `# @tests:` marker; rewrite its body)\n"
        f"```\n{test_source[:8000]}\n```\n",
        f"### Why the repair loop declared it unsatisfiable\n{unsat_reason}\n",
        f"### Failing test output\n{failing_output[:3000]}\n",
    ]
    if spec_tiebreaker.strip():
        parts.append(
            "### Specification (TIEBREAKER ONLY — do not cite in the test)\n"
            f"{spec_tiebreaker[:4000]}\n"
        )
    parts.append(
        "Rewrite the ONE test file as a comprehensive unit suite for the module "
        "above, anchored on its contract. Emit only the <<<REWRITE_FILE>>> block."
    )
    return [
        {"role": "system", "content": _REGEN_SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


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
    """LangGraph node: regenerate one declared-unsatisfiable unit-test file as
    a comprehensive suite for its 1:1-mapped code module, apply gates, write
    it, and route back to ``compiler_node`` to re-verify."""
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
    from harness.test_generation import _parse_tests_marker

    # Source of truth is the CODE this test maps to (1:1), located via the
    # `# @tests: <source>` marker — unit tests anchor on code, not the SRS.
    code_module_path = ""
    code_module_source = ""
    marker_paths = _parse_tests_marker(old_source)
    if marker_paths:
        code_module_path = marker_paths[0]
        code_abs = code_module_path if os.path.isabs(code_module_path) \
            else os.path.join(workspace, code_module_path)
        try:
            with open(code_abs, "r", encoding="utf-8") as fh:
                code_module_source = fh.read()
        except OSError as exc:
            logger.warning(
                "[test_regeneration_node] @tests source %s unreadable (%s); "
                "regenerating from the test + failing output only.",
                code_abs, exc,
            )
    module_symbols = public_symbols(code_module_source)

    # SRS is a TIEBREAKER only (anchored system prompt for spec-driven builds).
    spec_tiebreaker = ""
    for m in state.get("messages", []) or []:
        if isinstance(m, dict) and m.get("role") == "system" and \
                isinstance(m.get("content"), str):
            spec_tiebreaker = m["content"]
            break

    messages = build_regeneration_messages(
        test_rel_path=rel,
        test_source=old_source,
        code_module_path=code_module_path or "(unknown — no @tests marker)",
        code_module_source=code_module_source,
        module_symbols=module_symbols,
        unsat_reason=reason,
        failing_output=_failing_output(state, rel, workspace),
        spec_tiebreaker=spec_tiebreaker,
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

    # --- Pre-apply gate: only the declared test path may be touched ---
    targets = {_norm(t, workspace) for t in patch_target_paths(content)}
    stray = {t for t in targets if t != _norm(rel, workspace)}
    if stray:
        return _give_up("targeted_other_files",
                        f"regeneration tried to touch {sorted(stray)}")
    if not targets:
        return _give_up("no_patch", "regeneration emitted no REWRITE_FILE block")

    # Apply the rewrite (test-author phase — writing tests is permitted). The
    # allowlist still constrains where writes may land as defence in depth.
    from harness.patcher import process_llm_patch_output
    from harness.graph import _build_patcher_allowlist
    allowed_paths = _build_patcher_allowlist(workspace)
    existing_modified = list(state.get("modified_files", []) or [])
    patch_results, new_modified = await process_llm_patch_output(
        content, workspace, existing_modified, allowed_paths=allowed_paths,
    )

    # Re-read the written file and run the post-apply gates on it. Any gate
    # failure rolls the file back to the original — a rejected regeneration
    # must not leave a half-baked test on disk.
    try:
        with open(abs_path, "r", encoding="utf-8") as fh:
            written = fh.read()
    except OSError as exc:
        return _give_up("reread_failed", str(exc))

    def _rollback() -> None:
        try:
            with open(abs_path, "w", encoding="utf-8") as fh:
                fh.write(old_source)
        except OSError:
            logger.error("[test_regeneration_node] rollback write failed for %s", abs_path)

    # --- Gate: code linkage (@tests marker) — unit tests link to code ---
    if cfg.get("require_code_linkage", True) and not has_code_linkage(written):
        _rollback()
        return _give_up("no_code_linkage",
                        "regenerated test dropped its `# @tests:` marker")

    # --- Gate: coverage non-regression (anti-reward-hack floor) ---
    if cfg.get("coverage_nonregression", True):
        ok, detail = coverage_nonregression_ok(old_source, written)
        if not ok:
            _rollback()
            return _give_up("coverage_regression", detail)

    # --- Advisory: comprehensiveness against the module's public surface ---
    covered, uncovered = symbol_coverage(written, module_symbols)
    if uncovered:
        logger.warning(
            "[test_regeneration_node] Regenerated %s covers %d/%d public "
            "symbol(s); not exercised: %s.",
            rel, len(covered), len(module_symbols), ", ".join(uncovered),
        )

    applied = sum(1 for r in patch_results if getattr(r, "success", False))
    logger.warning(
        "[test_regeneration_node] Regenerated %s (attempt %d): %d block(s) "
        "applied, %d/%d public symbol(s) covered. Routing to compiler to "
        "re-verify.",
        rel, loop_counter[_REGEN_ATTEMPTS_KEY][rel], applied,
        len(covered), len(module_symbols),
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
                "code_module": code_module_path,
                "attempt": loop_counter[_REGEN_ATTEMPTS_KEY][rel],
                "symbols_covered": len(covered),
                "symbols_total": len(module_symbols),
            },
        },
    }
