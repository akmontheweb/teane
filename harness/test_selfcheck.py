"""Check a generated test's own expectations before the build sees them.

Four consecutive lumina runs (6, 7-blocked, 8, 9) and then 10 died on a
generated test that contradicts the spec or itself. The known instances:

* ``_relative_label`` asserted "This Saturday" for +3d and "In 2 days" for
  +2d -- no single rule satisfies both. Reproduced in TWO independent
  decompositions.
* ``test_api_birthdays.py:103`` asserted an unconditional 400 against
  ASM-009's stored-profile fallback.
* ``test_quick_action_service.py:41`` demanded ``wa.me/+15551234567`` where
  the spec fixes the URI without a leading ``+``.
* ``body["error"]`` asserted as a string where the SRS mandates an object
  with code/message/error_ref_id.
* ``upcoming(30, Jan 1)`` asserted 2 results in a window holding 1.
* ASM-007 fixes directory order as last_name, first_name, id ascending;
  ``test_returns_records_in_directory_order`` asserted the Jones pair
  backwards while a sibling test required first_name ascending.

Every one was diagnosed CORRECTLY by the model, usually within a round. The
recovery ladder kept improving and the runs kept ending the same way, because
no production change satisfies a false assertion and repair may not edit
tests. The defect is upstream: the tests are authored wrong.

Two gates already stand here and neither can see these. The prompt states
the spec's cross-cutting rules and declares them binding -- necessary, and
demonstrably not sufficient. ``find_contradictions_across`` proves one shape
only: the same normalised call required to both raise and succeed. None of
the six above has that shape; a sort order and a label are not a raise.

So this pass asks the question an AST cannot: for every expected value the
author COMPUTED, which rule fixes it, and does the assertion follow? It is a
review-and-remediate pair on the ADR-0007 pattern -- findings from a
separately routable reviewer, corrections re-prompted to the author.

The dangerous failure mode, and the one most of the guarding here is for: a
reviewer that flags a test merely because today's production code does not
satisfy it yet. That test is CORRECT and the build is supposed to go red
until the code catches up. Remediating it would delete working requirements
to make a suite green, so the prompt forbids it, the parser drops any finding
that reads that way, and :func:`actionable` acts on nothing else.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: A computed expectation that the spec's own rules rule out.
VERDICT_CONTRADICTS = "contradicts_spec"
#: Two or more expectations in scope that no single rule satisfies. The
#: ``_relative_label`` and ASM-007 shapes: individually plausible, jointly
#: impossible, and invisible to a raise/success prover.
VERDICT_INCONSISTENT = "self_inconsistent"
#: Computed from nothing the reviewer could find. Advisory only -- the spec is
#: silent on plenty of legitimate internal detail, and acting here would churn
#: correct tests.
VERDICT_UNJUSTIFIED = "unjustified"
VERDICT_OK = "follows"

#: Verdicts that earn a correction. Deliberately short.
ACTIONABLE_VERDICTS = (VERDICT_CONTRADICTS, VERDICT_INCONSISTENT)

_VALID_VERDICTS = (
    VERDICT_CONTRADICTS, VERDICT_INCONSISTENT, VERDICT_UNJUSTIFIED, VERDICT_OK,
)

#: Phrases that betray the one judgement this pass must never act on: a test
#: being blamed for outrunning the production code. Rule 6 of the author's own
#: system prompt keeps such a test, and the build is expected to be red until
#: the code catches up.
_NOT_YET_IMPLEMENTED_MARKERS = (
    "not implemented",
    "not yet implemented",
    "does not exist",
    "doesn't exist",
    "no such attribute",
    "missing from the code",
    "missing in the code",
    "not present in the code",
    "the code does not",
    "the code doesn't",
    "current implementation does not",
    "current implementation doesn't",
    "production code does not",
    "production code doesn't",
    "would fail against",
    "fails against the current",
)


@dataclass(frozen=True)
class Finding:
    """One reviewed expectation."""

    file: str
    test: str
    expectation: str
    verdict: str
    rule: str
    correction: str = ""

    def describe(self) -> str:
        head = f"{self.file}::{self.test} — {self.expectation}"
        if self.verdict == VERDICT_INCONSISTENT:
            head += " [no single rule satisfies this and its sibling]"
        parts = [head]
        if self.rule:
            parts.append(f"    rule: {self.rule}")
        if self.correction:
            parts.append(f"    should be: {self.correction}")
        return "\n".join(parts)


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated at {limit} chars]"


def build_selfcheck_messages(
    *,
    test_files: dict[str, str],
    module_sources: dict[str, str],
    cross_cutting_rules: str,
    max_test_chars: int = 12000,
    max_module_chars: int = 6000,
    max_rules_chars: int = 12000,
) -> list[dict[str, Any]]:
    """Build the reviewer prompt.

    The module sources are included for ONE reason, and it is not so that the
    code can arbitrate: it is so the reviewer can tell a spec conflict from a
    test that merely outruns the implementation, and leave the latter alone.
    """
    rules_block = (
        _truncate(cross_cutting_rules.strip(), max_rules_chars)
        if cross_cutting_rules.strip()
        else "(none found — judge self-consistency only, see below)"
    )

    lines: list[str] = [
        "You are reviewing freshly generated unit tests for ONE question: "
        "is every expected value the author COMPUTED actually correct?",
        "",
        "## The rules the specification fixes (AUTHORITATIVE)",
        rules_block,
        "",
        "## What counts as a COMPUTED expectation",
        "A value the author had to derive rather than copy: a sort order, a "
        "count, a date window, a label or phrase, an error shape or envelope, "
        "a URI or its formatting, a status code, a rounding, a pagination "
        "boundary. These are where a test goes false.",
        "",
        "## What you must NOT report",
        "1. **A test that is correct per the rules but that today's code "
        "does not satisfy.** That test is RIGHT and the build is SUPPOSED to "
        "be red until the code catches up. Never report it. The code is here "
        "only so you can recognise this case and skip it — the code is NOT "
        "an authority on what is correct.",
        "2. Style, naming, coverage gaps, missing tests, duplication, "
        "fixtures, imports, or anything about how a test is written.",
        "3. A value you merely find surprising. Report it only if you can "
        "name the rule it breaks, or name the sibling assertion it cannot "
        "coexist with.",
        "",
        "## Verdicts",
        f"- `{VERDICT_CONTRADICTS}` — the rules fix this value and the "
        "assertion states a different one. Quote the rule.",
        f"- `{VERDICT_INCONSISTENT}` — two or more assertions in these files "
        "cannot all hold under any single rule (e.g. one test requires "
        "ordering by last name, another by first name, for the same query). "
        "Name both sides in `rule`.",
        f"- `{VERDICT_UNJUSTIFIED}` — a computed value you can trace to "
        "nothing. Advisory; say so and move on.",
        f"- `{VERDICT_OK}` — omit these entirely. Do not list passing "
        "assertions.",
        "",
        "## Output",
        "A JSON array, nothing else. No prose, no fence commentary.",
        "```json",
        "[",
        '  {"file": "server/tests/test_x.py", "test": "test_order",',
        '   "expectation": "asserts [\\"Carol\\", \\"Bob\\", \\"Alice\\"]",',
        f'   "verdict": "{VERDICT_CONTRADICTS}",',
        '   "rule": "ASM-007: sorted by last_name asc, then first_name asc",',
        '   "correction": "[\\"Alice\\", \\"Bob\\", \\"Carol\\"]"}',
        "]",
        "```",
        "An empty array `[]` is the correct answer when every computed value "
        "checks out. Say that rather than inventing a finding.",
        "",
        "## Tests under review",
        "",
    ]
    for rel, src in sorted(test_files.items()):
        lines += [f"### `{rel}`", "```python",
                  _truncate(src, max_test_chars), "```", ""]

    if module_sources:
        lines += [
            "## Code under test — for recognising case 1 above, NOT as an "
            "authority",
            "",
        ]
        for rel, src in sorted(module_sources.items()):
            lines += [f"### `{rel}`", "```python",
                      _truncate(src, max_module_chars), "```", ""]

    return [{"role": "user", "content": "\n".join(lines)}]


def _looks_like_a_not_implemented_complaint(*fields: str) -> bool:
    blob = " ".join(f or "" for f in fields).lower()
    return any(m in blob for m in _NOT_YET_IMPLEMENTED_MARKERS)


def parse_selfcheck_findings(
    content: str, *, known_files: Optional[set[str]] = None,
) -> list[Finding]:
    """Parse the reviewer's JSON into findings, dropping what must not act.

    Defensive by design: a malformed review must degrade to "no findings" and
    let the build proceed, never raise into the generation node. The one piece
    of judgement here is the not-implemented filter — a reviewer that ignores
    the prompt and blames a test for outrunning the code gets its finding
    dropped rather than acted on, because acting on it deletes a correct
    requirement to make a suite green.
    """
    from harness.decomposition import strip_json_fence

    raw = strip_json_fence(content or "")
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "[test_selfcheck] reviewer returned non-JSON; treating as no "
            "findings and proceeding to the build.",
        )
        return []
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                data = value
                break
    if not isinstance(data, list):
        return []

    findings: list[Finding] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict") or "").strip().lower()
        if verdict not in _VALID_VERDICTS or verdict == VERDICT_OK:
            continue
        rel = str(item.get("file") or "").strip()
        if not rel:
            continue
        if known_files is not None and rel not in known_files:
            # A finding against a file this round did not generate cannot be
            # remediated here, and re-prompting for it would aim the author at
            # someone else's file.
            logger.info(
                "[test_selfcheck] dropping a finding against %s — not a file "
                "this round generated.", rel,
            )
            continue
        rule = str(item.get("rule") or "").strip()
        expectation = str(item.get("expectation") or "").strip()
        correction = str(item.get("correction") or "").strip()
        if verdict in ACTIONABLE_VERDICTS and not rule:
            # The prompt requires a cited rule or a named sibling. Without one
            # there is nothing to check the claim against, and an uncited
            # "this looks wrong" is how a correct test gets rewritten.
            logger.info(
                "[test_selfcheck] dropping an uncited %s finding on %s::%s.",
                verdict, rel, item.get("test"),
            )
            continue
        if _looks_like_a_not_implemented_complaint(rule, expectation, correction):
            logger.warning(
                "[test_selfcheck] dropping a finding on %s::%s that blames "
                "the test for outrunning the code — that test is correct and "
                "the build is meant to be red until the code catches up "
                "(author system rule 6). Rule given: %.120s",
                rel, item.get("test"), rule,
            )
            continue
        findings.append(Finding(
            file=rel,
            test=str(item.get("test") or "").strip(),
            expectation=expectation,
            verdict=verdict,
            rule=rule,
            correction=correction,
        ))
    return findings


def actionable(findings: list[Finding]) -> list[Finding]:
    """The findings worth a correction round."""
    return [f for f in findings if f.verdict in ACTIONABLE_VERDICTS]


def build_remediation_message(findings: list[Finding]) -> dict[str, Any]:
    """The author-facing correction prompt.

    Addressed to the author in its existing conversation, so it inherits the
    modules and the rules it was given rather than restating them -- the same
    shape as the cross-file contradiction re-prompt it sits beside.
    """
    contradicts = [f for f in findings if f.verdict == VERDICT_CONTRADICTS]
    inconsistent = [f for f in findings if f.verdict == VERDICT_INCONSISTENT]
    body: list[str] = [
        "STOP — a review of the tests you just generated found expected "
        "values that cannot be correct. No production change satisfies a "
        "false assertion, and the repair loop may NOT edit test files, so "
        "these must be fixed here.",
        "",
    ]
    if contradicts:
        body += [
            "## Values the specification rules out",
            *[f.describe() for f in contradicts],
            "",
        ]
    if inconsistent:
        body += [
            "## Expectations that cannot all hold",
            "No single rule satisfies these. Production can satisfy one or "
            "the other, never both, and the repair loop will oscillate "
            "between them until the run dies.",
            *[f.describe() for f in inconsistent],
            "",
        ]
    body += [
        "Fix the EXPECTED VALUES. Keep every test function and every "
        "assertion count — do not delete a test to resolve a finding, and do "
        "not weaken an assertion to `assert True` or a bare `is not None`.",
        "",
        "If an assertion is correct per the specification and today's "
        "production code simply does not satisfy it yet, LEAVE IT — a red "
        "build is the correct outcome there and the repair loop will fix the "
        "code. Change only what the rules actually rule out.",
        "",
        "Re-emit ONLY the corrected test file(s) as REWRITE_FILE blocks, "
        "keeping each file's `@tests` marker. Only patch blocks, no prose.",
    ]
    return {"role": "system", "content": "\n".join(body)}


async def review_generated_tests(
    *,
    gateway: Any,
    test_files: dict[str, str],
    module_sources: dict[str, str],
    cross_cutting_rules: str,
    budget: float,
    token_tracker: Any = None,
) -> tuple[list[Finding], float, Any]:
    """Run one review pass. Returns ``(findings, budget, token_tracker)``.

    Never raises: a review that cannot run returns no findings and the build
    proceeds exactly as it did before this pass existed.
    """
    from harness.gateway import NodeRole

    if not test_files:
        return [], budget, token_tracker
    try:
        if not gateway.select_model(NodeRole.CODE_REVIEWER):
            logger.info(
                "[test_selfcheck] no code_reviewer model configured; "
                "skipping the generated-test spec review.",
            )
            return [], budget, token_tracker
    except Exception:  # noqa: BLE001 — an unexpected gateway shape must not block
        pass

    messages = build_selfcheck_messages(
        test_files=test_files,
        module_sources=module_sources,
        cross_cutting_rules=cross_cutting_rules,
    )
    try:
        response, budget = await gateway.dispatch(
            messages=messages,
            role=NodeRole.CODE_REVIEWER,
            budget_remaining_usd=budget,
            cache_family="code_reviewer:test_selfcheck",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[test_selfcheck] review dispatch failed: %s — proceeding to the "
            "build without it.", exc,
        )
        return [], budget, token_tracker
    if token_tracker is not None:
        try:
            token_tracker = gateway.aggregate_tokens(
                token_tracker, response.usage,
            )
        except Exception:  # noqa: BLE001
            pass

    findings = parse_selfcheck_findings(
        getattr(response, "content", "") or "",
        known_files=set(test_files),
    )
    advisory = [f for f in findings if f.verdict == VERDICT_UNJUSTIFIED]
    if advisory:
        logger.info(
            "[test_selfcheck] %d computed value(s) the reviewer could not "
            "trace to a rule (advisory, not corrected): %s",
            len(advisory),
            "; ".join(f"{f.file}::{f.test}" for f in advisory[:5]),
        )
    return findings, budget, token_tracker
