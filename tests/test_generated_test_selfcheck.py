"""A generated test's expectations must be checked before the build.

Four consecutive lumina runs (6, 8, 9, 10) died on a generated test that
contradicts the spec or itself. The recovery ladder kept improving and the
runs kept ending the same way: no production change satisfies a false
assertion, and repair may not edit tests.

The six real instances are the fixtures here, because the gate that already
exists cannot see any of them. ``find_contradictions_across`` proves exactly
one shape -- the same normalised call required to both raise and succeed --
and a sort order is not a raise, nor is a label, a window count, an error
envelope or a URI. :meth:`TestTheDeterministicGateCannotSeeThese` pins that,
so if the AST prover ever grows to cover one of them this suite says so
instead of quietly duplicating work.

The most important tests here are the NEGATIVE ones. A reviewer that flags a
test merely because today's production code does not satisfy it yet would
have remediation delete a correct requirement to make a suite green -- worse
than the bug this pass exists for, because it is silent. See
:class:`TestItNeverPunishesATestForOutrunningTheCode`.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from harness.test_selfcheck import (
    VERDICT_CONTRADICTS,
    VERDICT_INCONSISTENT,
    VERDICT_OK,
    VERDICT_UNJUSTIFIED,
    Finding,
    actionable,
    build_remediation_message,
    build_selfcheck_messages,
    parse_selfcheck_findings,
)

#: ASM-007 fixes directory order; the test asserted the Jones pair backwards
#: while a sibling required first_name ascending. Run 6's killer.
ASM_007_TESTS = {
    "server/tests/test_directory.py": '''
# @tests: server/app/services/directory.py

def test_returns_records_in_directory_order(client):
    rows = client.get("/directory").json()
    assert [r["first_name"] for r in rows] == ["Carol", "Bob", "Alice"]

def test_directory_sorts_by_first_name(client):
    rows = client.get("/directory").json()
    assert [r["first_name"] for r in rows] == ["Alice", "Bob", "Carol"]
'''
}

#: The _relative_label case: no single rule satisfies both. Reproduced in two
#: independent decompositions.
RELATIVE_LABEL_TESTS = {
    "server/tests/test_dashboard_service.py": '''
# @tests: server/app/services/dashboard.py

def test_label_three_days_out():
    assert _relative_label(days=3) == "This Saturday"

def test_label_two_days_out():
    assert _relative_label(days=2) == "In 2 days"
'''
}

CROSS_CUTTING = """
## Cross-cutting rules
ASM-007: Directory records are sorted by last_name ascending, then
first_name ascending, then id ascending.
ASM-009: When no timezone is supplied the stored device-profile timezone is
resolved first; only its absence yields 400 TIMEZONE_REQUIRED.
REQ-041: The WhatsApp message URI is `wa.me/<digits>` with no leading `+`.
REQ-050: Every error response body is an object with code, message and
error_ref_id.
"""


def _review(items: list[dict]) -> str:
    return "```json\n" + json.dumps(items) + "\n```"


class TestTheDeterministicGateCannotSeeThese:
    """The premise. If this starts failing, the AST prover has grown and this
    pass may be duplicating it for that shape."""

    def test_the_prover_is_blind_to_a_sort_order_contradiction(self):
        from harness.test_contradiction import find_contradictions_across

        assert find_contradictions_across(ASM_007_TESTS) == [], (
            "find_contradictions proves raise-vs-success on one call; a sort "
            "order is neither"
        )

    def test_the_prover_is_blind_to_an_inconsistent_label_rule(self):
        from harness.test_contradiction import find_contradictions_across

        assert find_contradictions_across(RELATIVE_LABEL_TESTS) == []


class TestTheReviewerPrompt:
    def _body(self, **kw) -> str:
        return build_selfcheck_messages(
            test_files=kw.get("tests", ASM_007_TESTS),
            module_sources=kw.get("modules", {}),
            cross_cutting_rules=kw.get("rules", CROSS_CUTTING),
        )[-1]["content"]

    def test_the_rules_are_stated_as_authoritative(self):
        body = self._body()
        assert "AUTHORITATIVE" in body
        assert "ASM-007" in body

    def test_the_tests_under_review_are_included(self):
        assert "test_returns_records_in_directory_order" in self._body()

    def test_it_names_what_counts_as_a_computed_expectation(self):
        """Scoping the question is what keeps the reviewer off style and
        coverage, where it would generate churn instead of findings."""
        body = self._body()
        for kind in ("sort order", "count", "window", "label",
                     "error shape", "URI", "status code"):
            assert kind in body, kind

    def test_it_forbids_blaming_a_test_for_outrunning_the_code(self):
        """The load-bearing instruction. Without it the reviewer reports every
        red test and remediation deletes correct requirements."""
        body = self._body()
        assert "does not satisfy" in body
        assert "SUPPOSED to be red" in body or "supposed to be red" in body

    def test_the_code_is_labelled_as_not_an_authority(self):
        """The code was written to satisfy the test under review, so letting
        it arbitrate reproduces the falsehood -- the same trap the repair-side
        anchor fix exists for."""
        body = self._body(modules={"server/app/services/directory.py": "x=1"})
        assert "NOT as an authority" in body
        assert "NOT\nan authority" in body or "NOT an authority" in body

    def test_the_modules_section_is_absent_when_there_are_none(self):
        assert "Code under test" not in self._body(modules={})

    def test_it_rules_out_style_and_coverage_findings(self):
        body = self._body()
        assert "Style, naming, coverage gaps" in body

    def test_it_asks_for_json_only(self):
        body = self._body()
        assert "A JSON array, nothing else" in body

    def test_an_empty_array_is_named_as_a_valid_answer(self):
        """A reviewer that believes it must find something will."""
        assert "empty array" in self._body()

    def test_it_survives_a_missing_spec(self):
        body = self._body(rules="")
        assert "none found" in body
        assert "test_returns_records_in_directory_order" in body


class TestParsingTheReview:
    def test_a_real_asm_007_finding_parses(self):
        found = parse_selfcheck_findings(_review([{
            "file": "server/tests/test_directory.py",
            "test": "test_returns_records_in_directory_order",
            "expectation": 'asserts ["Carol", "Bob", "Alice"]',
            "verdict": VERDICT_CONTRADICTS,
            "rule": "ASM-007: last_name asc, then first_name asc",
            "correction": '["Alice", "Bob", "Carol"]',
        }]))
        assert len(found) == 1
        assert found[0].verdict == VERDICT_CONTRADICTS
        assert "ASM-007" in found[0].rule

    def test_passing_assertions_are_dropped(self):
        assert parse_selfcheck_findings(_review([{
            "file": "server/tests/test_directory.py", "test": "t",
            "verdict": VERDICT_OK, "rule": "ASM-007",
        }])) == []

    def test_an_unknown_verdict_is_dropped(self):
        assert parse_selfcheck_findings(_review([{
            "file": "a.py", "test": "t", "verdict": "looks_odd", "rule": "r",
        }])) == []

    def test_a_finding_with_no_file_is_dropped(self):
        assert parse_selfcheck_findings(_review([{
            "test": "t", "verdict": VERDICT_CONTRADICTS, "rule": "r",
        }])) == []

    def test_a_finding_against_a_file_we_did_not_generate_is_dropped(self):
        """Remediation re-prompts the author for ITS files; a finding
        elsewhere would aim a REWRITE_FILE at someone else's code."""
        assert parse_selfcheck_findings(
            _review([{
                "file": "server/app/services/directory.py", "test": "t",
                "verdict": VERDICT_CONTRADICTS, "rule": "ASM-007",
            }]),
            known_files={"server/tests/test_directory.py"},
        ) == []

    def test_an_actionable_finding_with_no_cited_rule_is_dropped(self):
        """An uncited "this looks wrong" is how a correct test gets
        rewritten. The prompt demands a rule; the parser enforces it."""
        assert parse_selfcheck_findings(_review([{
            "file": "a.py", "test": "t", "verdict": VERDICT_CONTRADICTS,
            "expectation": "asserts 3", "rule": "",
        }])) == []

    def test_an_advisory_finding_needs_no_rule(self):
        """`unjustified` means precisely that no rule was found, so requiring
        one would delete the whole category."""
        found = parse_selfcheck_findings(_review([{
            "file": "a.py", "test": "t", "verdict": VERDICT_UNJUSTIFIED,
            "expectation": "asserts 7", "rule": "",
        }]))
        assert len(found) == 1

    def test_non_json_degrades_to_no_findings(self):
        """A malformed review must let the build proceed exactly as it did
        before this pass existed, never raise into the node."""
        assert parse_selfcheck_findings("I reviewed them and they look fine") == []

    def test_an_empty_body_is_no_findings(self):
        assert parse_selfcheck_findings("") == []

    def test_a_wrapped_object_is_unwrapped(self):
        found = parse_selfcheck_findings(_review({"findings": [{  # type: ignore[arg-type]
            "file": "a.py", "test": "t", "verdict": VERDICT_CONTRADICTS,
            "rule": "REQ-041",
        }]}))
        assert len(found) == 1

    def test_garbage_list_entries_are_skipped(self):
        found = parse_selfcheck_findings(_review([
            "nonsense",  # type: ignore[list-item]
            {"file": "a.py", "test": "t", "verdict": VERDICT_CONTRADICTS,
             "rule": "REQ-050"},
        ]))
        assert len(found) == 1


class TestItNeverPunishesATestForOutrunningTheCode:
    """The failure mode worse than the bug.

    A test that is correct per the spec but fails against today's production
    code is RIGHT, and the build is meant to be red until repair fixes the
    code (the author's own system rule 6). Remediating it would delete a
    correct requirement to make a suite green, and silently.
    """

    @pytest.mark.parametrize("rule", [
        "the function is not implemented yet",
        "resolve_caller_timezone does not exist in the module",
        "NameError — missing from the code",
        "the code does not handle this branch",
        "current implementation doesn't return an envelope",
        "this would fail against the current production code",
    ])
    def test_a_not_implemented_complaint_is_dropped(self, rule):
        assert parse_selfcheck_findings(_review([{
            "file": "a.py", "test": "t", "verdict": VERDICT_CONTRADICTS,
            "expectation": "asserts an envelope object", "rule": rule,
        }])) == [], f"acted on: {rule}"

    def test_the_filter_reads_the_correction_field_too(self):
        """A reviewer can smuggle the same judgement into any field, and the
        correction is the natural place for it — a plausible rule up front,
        then "so delete it" where the fix should be."""
        assert parse_selfcheck_findings(_review([{
            "file": "a.py", "test": "t", "verdict": VERDICT_CONTRADICTS,
            "rule": "REQ-050 mandates an object",
            "correction": "remove the assertion — resolve_caller_timezone "
                          "does not exist in the module yet",
        }])) == []

    def test_a_genuine_spec_finding_still_survives_the_filter(self):
        """Vacuity guard: a filter that dropped everything would make this
        whole pass a no-op and every other test here pass for free."""
        found = parse_selfcheck_findings(_review([{
            "file": "a.py", "test": "t",
            "verdict": VERDICT_CONTRADICTS,
            "expectation": "asserts wa.me/+15551234567",
            "rule": "REQ-041 fixes the URI as wa.me/<digits> with no "
                    "leading +",
            "correction": "wa.me/15551234567",
        }]))
        assert len(found) == 1


class TestWhatEarnsACorrection:
    def test_a_spec_contradiction_is_actionable(self):
        f = Finding("a.py", "t", "e", VERDICT_CONTRADICTS, "ASM-007")
        assert actionable([f]) == [f]

    def test_a_self_inconsistency_is_actionable(self):
        """The _relative_label and ASM-007 shapes. Individually plausible,
        jointly impossible, and invisible to the AST prover."""
        f = Finding("a.py", "t", "e", VERDICT_INCONSISTENT, "both sides")
        assert actionable([f]) == [f]

    def test_an_unjustified_value_is_only_advisory(self):
        """The spec is silent on plenty of legitimate internal detail;
        correcting here would churn correct tests."""
        f = Finding("a.py", "t", "e", VERDICT_UNJUSTIFIED, "")
        assert actionable([f]) == []


class TestTheCorrectionPrompt:
    def _msg(self, findings) -> str:
        m = build_remediation_message(findings)
        assert m["role"] == "system"
        return m["content"]

    def test_it_names_the_rule_and_the_fix(self):
        body = self._msg([Finding(
            "server/tests/test_directory.py",
            "test_returns_records_in_directory_order",
            'asserts ["Carol", "Bob", "Alice"]',
            VERDICT_CONTRADICTS,
            "ASM-007: last_name asc, then first_name asc",
            '["Alice", "Bob", "Carol"]',
        )])
        assert "ASM-007" in body
        assert "test_returns_records_in_directory_order" in body
        assert '["Alice", "Bob", "Carol"]' in body

    def test_it_explains_why_repair_cannot_do_this(self):
        body = self._msg([Finding("a.py", "t", "e", VERDICT_CONTRADICTS, "r")])
        assert "may NOT edit test files" in body

    def test_it_forbids_deleting_tests_to_resolve_a_finding(self):
        """The reward-hacking escape. The coverage gate catches it later on
        the repair side; cheaper to forbid it here."""
        body = self._msg([Finding("a.py", "t", "e", VERDICT_CONTRADICTS, "r")])
        assert "do not delete a test" in body
        assert "assert True" in body

    def test_it_repeats_the_leave_red_tests_alone_rule(self):
        """The correction round is a second chance to get this wrong."""
        body = self._msg([Finding("a.py", "t", "e", VERDICT_CONTRADICTS, "r")])
        assert "LEAVE IT" in body

    def test_it_asks_for_rewrite_blocks_keeping_the_marker(self):
        body = self._msg([Finding("a.py", "t", "e", VERDICT_CONTRADICTS, "r")])
        assert "REWRITE_FILE" in body
        assert "@tests" in body

    def test_an_inconsistency_is_explained_as_jointly_impossible(self):
        body = self._msg([Finding(
            "server/tests/test_dashboard_service.py", "test_label_three_days_out",
            'asserts "This Saturday" at +3d',
            VERDICT_INCONSISTENT,
            'the sibling requires "In 2 days" at +2d',
        )])
        assert "cannot all hold" in body
        assert "oscillate" in body

    def test_the_two_kinds_get_separate_sections(self):
        body = self._msg([
            Finding("a.py", "t1", "e1", VERDICT_CONTRADICTS, "ASM-007"),
            Finding("a.py", "t2", "e2", VERDICT_INCONSISTENT, "sibling"),
        ])
        assert body.index("specification rules out") < body.index(
            "cannot all hold"
        )


class TestTheWiring:
    def _src(self) -> str:
        return (
            pathlib.Path(__file__).parent.parent
            / "harness" / "test_generation.py"
        ).read_text()

    def test_the_node_runs_the_review(self):
        src = self._src()
        assert "review_generated_tests" in src
        assert "build_remediation_message" in src

    def test_it_runs_after_the_deterministic_gate(self):
        """The AST prover is free; spend a dispatch only on what it cannot
        decide.

        Both gates are now extracted helpers, so the order is the order of
        the two calls in the node — ``find_contradictions_across`` itself
        lives inside ``_run_contradiction_gate``.
        """
        import inspect

        import harness.test_generation as tg

        node = inspect.getsource(tg.test_generation_node)
        assert node.index("_run_contradiction_gate(") < node.index(
            "_run_spec_consistency_gate("
        )
        assert "find_contradictions_across(_py_tests)" in inspect.getsource(
            tg._run_contradiction_gate
        )

    def test_it_feeds_the_reviewer_the_mapped_modules(self):
        """So a test outrunning its implementation is recognisable."""
        src = self._src()
        assert "module_sources=_sc_modules" in src
        assert "_parse_tests_marker" in src

    def test_it_feeds_the_reviewer_the_same_spec_rules_the_author_got(self):
        src = self._src()
        assert "cross_cutting_rules=_cross_cutting_rules_for_tests" in src

    def test_the_correction_is_billed_and_tracked(self):
        """An untracked dispatch breaks the budget telemetry that every cap
        downstream reads."""
        src = self._src()
        at = src.index("patching:test_spec_review_reprompt")
        window = src[at:at + 900]
        assert "aggregate_tokens" in window
        assert "budget = new_budget" in window

    def test_the_review_is_bounded(self):
        src = self._src()
        assert "max_spec_review_rounds" in src
        assert "_sc_rounds >= _sc_cap" in src

    def test_the_cap_bounds_corrections_not_the_review(self):
        """0 must still review and log — the mode to watch a run in before
        letting it rewrite anything, which is what config_help promises."""
        src = self._src()
        assert "while _sc_cfg and generated_tests:" in src

    def test_it_emits_telemetry(self):
        src = self._src()
        assert "test_generation_spec_review" in src


class TestTheConfig:
    def _tg(self) -> dict:
        from harness.gateway import create_gateway_from_config  # noqa: F401
        return json.loads(
            (pathlib.Path(__file__).parent.parent
             / "config" / "config.json").read_text()
        )["test_generation"]

    def test_the_shipped_config_enables_it(self):
        cfg = self._tg()
        assert cfg["spec_consistency_review"] is True
        assert cfg["max_spec_review_rounds"] >= 1

    def test_the_keys_pass_config_validation(self):
        from harness.cli import _KNOWN_NESTED_KEYS

        allowed = _KNOWN_NESTED_KEYS["test_generation"]
        assert "spec_consistency_review" in allowed
        assert "max_spec_review_rounds" in allowed

    def test_the_keys_are_type_checked(self):
        from harness.cli import _TYPE_SCHEMA

        assert _TYPE_SCHEMA["test_generation.spec_consistency_review"] == (bool,)
        assert _TYPE_SCHEMA["test_generation.max_spec_review_rounds"] == (int,)

    def test_they_are_documented(self):
        help_md = (
            pathlib.Path(__file__).parent.parent
            / "config" / "config_help.md"
        ).read_text()
        assert "spec_consistency_review" in help_md
        assert "max_spec_review_rounds" in help_md


class TestTheReviewPassEndToEnd:
    @pytest.mark.asyncio
    async def test_a_reviewer_finding_comes_back_parsed(self):
        from harness.test_selfcheck import review_generated_tests

        class _Resp:
            content = _review([{
                "file": "server/tests/test_directory.py",
                "test": "test_returns_records_in_directory_order",
                "expectation": 'asserts ["Carol", "Bob", "Alice"]',
                "verdict": VERDICT_CONTRADICTS,
                "rule": "ASM-007: last_name asc, then first_name asc",
                "correction": '["Alice", "Bob", "Carol"]',
            }])
            usage = None

        class _GW:
            def select_model(self, _role):
                return "stub:reviewer"

            async def dispatch(self, **_kw):
                return _Resp(), 9.0

            def aggregate_tokens(self, t, _u):
                return t

        findings, budget, _ = await review_generated_tests(
            gateway=_GW(), test_files=ASM_007_TESTS, module_sources={},
            cross_cutting_rules=CROSS_CUTTING, budget=10.0,
        )
        assert len(actionable(findings)) == 1
        assert budget == 9.0

    @pytest.mark.asyncio
    async def test_no_reviewer_model_skips_cleanly(self):
        from harness.test_selfcheck import review_generated_tests

        class _GW:
            def select_model(self, _role):
                return ""

            async def dispatch(self, **_kw):  # pragma: no cover
                raise AssertionError("must not dispatch without a model")

        findings, budget, _ = await review_generated_tests(
            gateway=_GW(), test_files=ASM_007_TESTS, module_sources={},
            cross_cutting_rules=CROSS_CUTTING, budget=10.0,
        )
        assert findings == []
        assert budget == 10.0

    @pytest.mark.asyncio
    async def test_a_dispatch_failure_does_not_block_the_build(self):
        from harness.test_selfcheck import review_generated_tests

        class _GW:
            def select_model(self, _role):
                return "stub:reviewer"

            async def dispatch(self, **_kw):
                raise RuntimeError("provider down")

        findings, budget, _ = await review_generated_tests(
            gateway=_GW(), test_files=ASM_007_TESTS, module_sources={},
            cross_cutting_rules=CROSS_CUTTING, budget=10.0,
        )
        assert findings == []
        assert budget == 10.0

    @pytest.mark.asyncio
    async def test_no_tests_means_no_dispatch(self):
        from harness.test_selfcheck import review_generated_tests

        class _GW:
            async def dispatch(self, **_kw):  # pragma: no cover
                raise AssertionError("must not dispatch with nothing to review")

        findings, budget, _ = await review_generated_tests(
            gateway=_GW(), test_files={}, module_sources={},
            cross_cutting_rules=CROSS_CUTTING, budget=10.0,
        )
        assert findings == []
        assert budget == 10.0


class TestTheGateActuallyRuns:
    """Drives the REAL loop, not a source grep.

    This is the class that matters. 33 tests once passed against an in-turn
    retry that raised TypeError on every firing, because not one of them
    executed it. Everything in TestTheWiring above is a grep; this runs
    ``_run_spec_consistency_gate`` with injected fakes and checks the
    corrected bytes reach disk.
    """

    FALSE_TEST = '''# @tests: server/app/services/directory.py


def test_returns_records_in_directory_order(client):
    rows = client.get("/directory").json()
    assert [r["first_name"] for r in rows] == ["Carol", "Bob", "Alice"]
'''

    CORRECTED = '''# @tests: server/app/services/directory.py


def test_returns_records_in_directory_order(client):
    rows = client.get("/directory").json()
    assert [r["first_name"] for r in rows] == ["Alice", "Bob", "Carol"]
'''

    REVIEW = _review([{
        "file": "server/tests/test_directory.py",
        "test": "test_returns_records_in_directory_order",
        "expectation": 'asserts ["Carol", "Bob", "Alice"]',
        "verdict": VERDICT_CONTRADICTS,
        "rule": "ASM-007: sorted by last_name asc, then first_name asc",
        "correction": '["Alice", "Bob", "Carol"]',
    }])

    CLEAN_REVIEW = "[]"

    @staticmethod
    def _rewrite(body: str) -> str:
        """A real REWRITE_FILE block — `file:`/`content:` fields and
        `<<<END_REWRITE_FILE>>>`, per harness.patcher._BLOCK_PATTERNS.

        Written out longhand on purpose: the first version of this test
        invented `<<<REWRITE_FILE: path>>> ... <<<END>>>`, the patcher logged
        "No patch blocks to apply", and the gate looked like it had corrected
        the file when nothing had been written.
        """
        return (
            "<<<REWRITE_FILE>>>\n"
            "file: server/tests/test_directory.py\n"
            "content:\n"
            + body +
            "<<<END_REWRITE_FILE>>>\n"
        )

    def _workspace(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "server" / "tests").mkdir(parents=True)
        (ws / "server" / "app" / "services").mkdir(parents=True)
        (ws / "docs").mkdir()
        (ws / "server" / "tests" / "test_directory.py").write_text(
            self.FALSE_TEST
        )
        (ws / "server" / "app" / "services" / "directory.py").write_text(
            "def list_records():\n    return []\n"
        )
        (ws / "docs" / "SPEC_REQUIREMENTS.md").write_text(
            "# Requirements\n\n## Cross-cutting rules\n\n"
            "ASM-007: Directory records are sorted by last_name ascending, "
            "then first_name ascending, then id ascending.\n\n"
            "# EPIC-001: Directory\n"
        )
        return ws

    class _GW:
        """Scripts the reviewer turn and the author's correction turn."""

        class config:
            repair_fallback = ""
            planning_fallback = ""

        def __init__(self, scripted: list[str]):
            self._scripted = list(scripted)
            self.roles: list[str] = []

        def select_model(self, _role):
            return "stub:model"

        async def dispatch(self, *, messages, role, budget_remaining_usd, **kw):
            self.roles.append(getattr(role, "value", str(role)))
            content = self._scripted.pop(0) if self._scripted else "[]"

            class _R:
                def __init__(self, c):
                    self.content = c

                    class _U:
                        cost_usd = 0.001
                        input_tokens = 10
                        output_tokens = 10
                        cached_tokens = 0
                        model = "stub"
                    self.usage = _U()

            return _R(content), budget_remaining_usd - 0.001

        def aggregate_tokens(self, tracker, usage, role=None):
            out = dict(tracker or {})
            out["total_cost_usd"] = out.get("total_cost_usd", 0.0) + 0.001
            return out

    async def _run(self, tmp_path, scripted, *, cfg=None):
        import harness.test_generation as tg

        ws = self._workspace(tmp_path)
        gw = self._GW(scripted)
        out = await tg._run_spec_consistency_gate(
            gateway=gw,
            workspace_path=str(ws),
            generated_tests=["server/tests/test_directory.py"],
            messages=[{"role": "user", "content": "generate tests"}],
            existing_modified=[],
            allowed_paths=None,
            budget=10.0,
            token_tracker={},
            cfg=cfg if cfg is not None else {},
        )
        return ws, gw, out

    @pytest.mark.asyncio
    async def test_a_false_expectation_is_corrected_on_disk(self, tmp_path):
        """End to end: ASM-007 violated, reviewer flags it, author corrects
        it, the corrected bytes are on disk before any build runs."""
        rewrite = self._rewrite(self.CORRECTED)
        ws, gw, out = await self._run(
            tmp_path, [self.REVIEW, rewrite, self.CLEAN_REVIEW],
        )
        on_disk = (ws / "server" / "tests" / "test_directory.py").read_text()
        assert '["Alice", "Bob", "Carol"]' in on_disk, (
            "the corrected expectation never reached disk:\n" + on_disk
        )
        assert out["rounds"] == 1

    @pytest.mark.asyncio
    async def test_the_reviewer_and_the_author_are_separate_roles(self, tmp_path):
        """The review is routable independently (code_reviewer_*) so an
        operator can point it at a stronger model than the author."""
        rewrite = self._rewrite(self.CORRECTED)
        _, gw, _ = await self._run(
            tmp_path, [self.REVIEW, rewrite, self.CLEAN_REVIEW],
        )
        assert gw.roles[0] == "code_reviewer"
        assert gw.roles[1] == "patching"

    @pytest.mark.asyncio
    async def test_a_clean_review_costs_one_dispatch_and_changes_nothing(
        self, tmp_path,
    ):
        ws, gw, out = await self._run(tmp_path, [self.CLEAN_REVIEW])
        assert len(gw.roles) == 1
        assert out["rounds"] == 0
        on_disk = (ws / "server" / "tests" / "test_directory.py").read_text()
        assert on_disk == self.FALSE_TEST, "a clean review must not rewrite"

    @pytest.mark.asyncio
    async def test_it_is_disabled_by_config(self, tmp_path):
        _, gw, out = await self._run(
            tmp_path, [self.REVIEW], cfg={"spec_consistency_review": False},
        )
        assert gw.roles == []
        assert out["rounds"] == 0

    @pytest.mark.asyncio
    async def test_zero_rounds_reviews_and_logs_without_correcting(
        self, tmp_path,
    ):
        """What config_help promises for max_spec_review_rounds=0: the mode
        to watch a run in before letting it rewrite anything."""
        ws, gw, out = await self._run(
            tmp_path, [self.REVIEW], cfg={"max_spec_review_rounds": 0},
        )
        assert gw.roles == ["code_reviewer"], "the review must still run"
        assert out["rounds"] == 0
        on_disk = (ws / "server" / "tests" / "test_directory.py").read_text()
        assert on_disk == self.FALSE_TEST, "nothing may be rewritten at cap 0"

    @pytest.mark.asyncio
    async def test_a_model_that_never_fixes_it_is_bounded(self, tmp_path):
        """The cap is what keeps an unreconcilable finding from spending the
        whole budget here instead of escalating to the repair-side ladder.

        Each round emits DIFFERENT still-wrong content, because a
        byte-identical re-emission is rejected by the patcher as a no-op and
        exits on the no-write path below instead of reaching the cap.
        """
        wrongs = [
            self.FALSE_TEST.replace("Carol", f"Carol{i}") for i in range(3)
        ]
        _, gw, out = await self._run(
            tmp_path,
            [self.REVIEW, self._rewrite(wrongs[0]),
             self.REVIEW, self._rewrite(wrongs[1]),
             self.REVIEW, self._rewrite(wrongs[2])],
            cfg={"max_spec_review_rounds": 2},
        )
        assert out["rounds"] == 2
        assert gw.roles.count("patching") == 2

    @pytest.mark.asyncio
    async def test_a_correction_that_writes_nothing_keeps_the_test_set(
        self, tmp_path,
    ):
        """A real defect this test found, and one the sibling contradiction
        loop still has.

        The patcher rejects a byte-identical re-emission as a no-op, so the
        correction lands no patch and the recomputed "newly modified" list is
        EMPTY. Assigning that through would wipe the generated-test set that
        downstream work keys off — marker persistence, the runner's selection
        — silently dropping tests that were generated perfectly well.
        """
        _, gw, out = await self._run(
            tmp_path,
            [self.REVIEW, self._rewrite(self.FALSE_TEST), self.REVIEW],
        )
        assert out["generated_tests"] == ["server/tests/test_directory.py"], (
            "the generated-test set was wiped by a no-op correction"
        )
        assert out["rounds"] == 1
        assert gw.roles.count("patching") == 1, (
            "a correction that wrote nothing will not write anything next "
            "round either — it must stop, not retry to the cap"
        )

    @pytest.mark.asyncio
    async def test_the_budget_is_spent_and_returned(self, tmp_path):
        rewrite = self._rewrite(self.CORRECTED)
        _, _, out = await self._run(
            tmp_path, [self.REVIEW, rewrite, self.CLEAN_REVIEW],
        )
        assert out["budget"] < 10.0, "the dispatches must be billed"
        assert out["token_tracker"]["total_cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_the_reviewer_is_given_the_spec_and_the_module(
        self, tmp_path,
    ):
        """The two inputs the findings depend on: the rule to check against,
        and the module needed to recognise a test that merely outruns it."""
        captured: dict = {}

        class _Capture(self._GW):
            async def dispatch(self, *, messages, role, budget_remaining_usd, **kw):
                if getattr(role, "value", "") == "code_reviewer":
                    captured["prompt"] = messages[-1]["content"]
                return await super().dispatch(
                    messages=messages, role=role,
                    budget_remaining_usd=budget_remaining_usd, **kw,
                )

        import harness.test_generation as tg

        ws = self._workspace(tmp_path)
        await tg._run_spec_consistency_gate(
            gateway=_Capture([self.CLEAN_REVIEW]),
            workspace_path=str(ws),
            generated_tests=["server/tests/test_directory.py"],
            messages=[{"role": "user", "content": "go"}],
            existing_modified=[], allowed_paths=None,
            budget=10.0, token_tracker={}, cfg={},
        )
        prompt = captured["prompt"]
        assert "ASM-007" in prompt, "the spec rule never reached the reviewer"
        assert "def list_records" in prompt, (
            "the mapped module never reached the reviewer, so it cannot tell "
            "a spec conflict from a test outrunning the code"
        )

    @pytest.mark.asyncio
    async def test_a_reviewer_failure_leaves_the_build_untouched(self, tmp_path):
        import harness.test_generation as tg

        class _Broken(self._GW):
            async def dispatch(self, **_kw):
                raise RuntimeError("provider down")

        ws = self._workspace(tmp_path)
        out = await tg._run_spec_consistency_gate(
            gateway=_Broken([]),
            workspace_path=str(ws),
            generated_tests=["server/tests/test_directory.py"],
            messages=[{"role": "user", "content": "go"}],
            existing_modified=[], allowed_paths=None,
            budget=10.0, token_tracker={}, cfg={},
        )
        assert out["rounds"] == 0
        assert out["generated_tests"] == ["server/tests/test_directory.py"]
        assert (ws / "server" / "tests" / "test_directory.py").read_text() == (
            self.FALSE_TEST
        )

    @pytest.mark.asyncio
    async def test_a_non_python_test_set_is_skipped(self, tmp_path):
        import harness.test_generation as tg

        ws = self._workspace(tmp_path)
        (ws / "web").mkdir()
        (ws / "web" / "a.spec.ts").write_text("it('x', () => {})\n")
        gw = self._GW([self.REVIEW])
        out = await tg._run_spec_consistency_gate(
            gateway=gw,
            workspace_path=str(ws),
            generated_tests=["web/a.spec.ts"],
            messages=[{"role": "user", "content": "go"}],
            existing_modified=[], allowed_paths=None,
            budget=10.0, token_tracker={}, cfg={},
        )
        assert gw.roles == [], "a .ts suite is not this reviewer's business"
        assert out["rounds"] == 0

    @pytest.mark.asyncio
    async def test_the_node_calls_the_gate(self):
        """The helper is only useful if the node actually invokes it."""
        import inspect

        import harness.test_generation as tg

        src = inspect.getsource(tg.test_generation_node)
        assert "_run_spec_consistency_gate(" in src
        assert 'generated_tests = _sc_out["generated_tests"]' in src
        assert 'budget = _sc_out["budget"]' in src
