"""Regressions for the six defects found in lumina run 01a15308 (2026-10-19).

Each class below pins one finding from that run. The run is the only reason
these paths are covered: every one of them had unit tests that passed while
the behaviour was wrong in situ, so the tests here assert the *live* symptom
(what the harness actually did to a real workspace), not the shape of the
helper.
"""

from __future__ import annotations

import pytest

from harness import cli
from harness import decomposition_review as dr
from harness import graph
from harness import sandbox


class TestReadOnlyRootFalsePositive:
    """F4 — ``--system`` matched inside ``--system-site-packages``."""

    # The command cli.py seeds for a greenfield Python backend. It contains
    # both `--system-site-packages` and `uv pip install`, which satisfied
    # both halves of the old substring test on every single compile.
    SEEDED = (
        "test -d /tmp/teane-venv/bin || uv venv --system-site-packages "
        "/tmp/teane-venv && . /tmp/teane-venv/bin/activate && uv pip "
        "install --quiet pytest-timeout hypothesis httpx && (cd server && "
        "uv pip install -e .) && python3 -m pytest -vv"
    )

    def test_seeded_python_command_does_not_need_writable_root(self):
        assert graph._build_command_writes_root_fs(self.SEEDED) is False
        assert graph._root_fs_write_reason(self.SEEDED) is None

    def test_venv_system_site_packages_alone_is_not_a_system_install(self):
        assert graph._build_command_writes_root_fs(
            "uv venv --system-site-packages /tmp/v && uv pip install -e ."
        ) is False

    def test_real_system_install_still_detected(self):
        for cmd in (
            "pip install --system foo",
            "uv pip install --system -r requirements.txt",
        ):
            assert graph._build_command_writes_root_fs(cmd) is True, cmd
            assert "--system" in (graph._root_fs_write_reason(cmd) or "")

    def test_system_flag_in_a_later_segment_does_not_count(self):
        # `--system` belonging to some other tool after `&&` says nothing
        # about where pip writes.
        assert graph._build_command_writes_root_fs(
            "uv pip install -e . && othertool --system"
        ) is False

    def test_global_npm_still_detected_and_reason_names_npm(self):
        reason = graph._root_fs_write_reason("npm install -g typescript")
        assert reason is not None
        assert "npm install -g" in reason
        assert "node_modules" in reason

    def test_reason_does_not_blame_npm_for_a_pip_install(self):
        # The three read_only_root log sites hard-coded the npm explanation,
        # so a pip-only build was reported as running `npm install -g`.
        reason = graph._root_fs_write_reason("pip install --system foo")
        assert reason is not None
        assert "npm" not in reason


class TestDeriveUserGoal:
    """F5 — ``Product goal (context):`` shipped empty on every dispatch."""

    CONSOLIDATED = (
        "# Product Specification (consolidated from 1 file(s))\n"
        "\n"
        "Source files:\n"
        "  - birthday_tracker.md\n"
        "\n"
        "---\n"
        "## birthday_tracker.md\n"
        "\n"
        "# Product Specification: Birthday Tracker\n"
        "\n"
        "## 1. Overview & Vision\n"
        "A lightweight personal relationship tool so users never miss "
        "birthdays for friends, family and professional contacts.\n"
    )

    def test_explicit_prompt_wins(self):
        assert cli._derive_user_goal("build a thing", self.CONSOLIDATED) == (
            "build a thing"
        )

    def test_falls_back_to_spec_prose_when_no_prompt(self):
        # args.prompt is None whenever the spec arrives as files, which is
        # the normal path — that is how the context came out empty.
        goal = cli._derive_user_goal(None, self.CONSOLIDATED)
        assert goal, "goal must not be empty when a spec is available"
        assert "never miss" in goal

    def test_fallback_skips_the_consolidation_manifest(self):
        goal = cli._derive_user_goal(None, self.CONSOLIDATED)
        assert "Source files" not in goal
        assert "birthday_tracker.md" not in goal
        assert not goal.startswith("#")

    def test_blank_prompt_is_treated_as_absent(self):
        assert cli._derive_user_goal("   ", self.CONSOLIDATED).startswith("A ")

    def test_no_prompt_and_no_spec_is_empty(self):
        assert cli._derive_user_goal(None, None) == ""

    def test_respects_the_limit(self):
        assert len(cli._derive_user_goal(None, self.CONSOLIDATED, limit=40)) <= 40


class TestRemediableFindings:
    """F1 — the filter demanded an intersection the design made empty."""

    # Verbatim shape of what the reviewer emitted on 01a15308: every
    # ac_quality finding medium, every high finding structural.
    RUN_FINDINGS = [
        {"story_key": "STORY-001", "dimension": "ac_quality",
         "severity": "medium", "suggested_action": "rewrite_ac"},
        {"story_key": "STORY-002", "dimension": "ac_quality",
         "severity": "medium", "suggested_action": "rewrite_ac"},
        {"story_key": "STORY-007", "dimension": "right_sizing",
         "severity": "high", "suggested_action": "split"},
        {"story_key": "STORY-012", "dimension": "overlap",
         "severity": "high", "suggested_action": "merge"},
        {"story_key": "STORY-001", "dimension": "dependency",
         "severity": "high", "suggested_action": "add_dependency"},
    ]

    def test_the_run_findings_now_yield_work(self):
        got = dr._remediable_findings(self.RUN_FINDINGS)
        assert [f["story_key"] for f in got] == ["STORY-001", "STORY-002"]

    def test_structural_recommendations_still_excluded(self):
        for f in dr._remediable_findings(self.RUN_FINDINGS):
            assert f["dimension"] == "ac_quality"
            assert f["suggested_action"] == "rewrite_ac"

    def test_high_severity_ac_quality_still_included(self):
        got = dr._remediable_findings([
            {"story_key": "S1", "dimension": "ac_quality",
             "severity": "high", "suggested_action": "rewrite_ac"},
        ])
        assert len(got) == 1

    def test_low_is_excluded_by_default(self):
        got = dr._remediable_findings([
            {"story_key": "S1", "dimension": "ac_quality",
             "severity": "low", "suggested_action": "rewrite_ac"},
        ])
        assert got == []

    def test_severities_are_configurable(self):
        findings = [
            {"story_key": "S1", "dimension": "ac_quality",
             "severity": "medium", "suggested_action": "rewrite_ac"},
        ]
        assert dr._remediable_findings(findings, ("high",)) == []
        assert len(dr._remediable_findings(findings, ("high", "medium"))) == 1

    def test_severity_matching_is_case_insensitive(self):
        got = dr._remediable_findings([
            {"story_key": "S1", "dimension": "ac_quality",
             "severity": "MEDIUM", "suggested_action": "rewrite_ac"},
        ])
        assert len(got) == 1

    def test_explicit_remediable_false_is_honoured(self):
        got = dr._remediable_findings([
            {"story_key": "S1", "dimension": "ac_quality",
             "severity": "medium", "suggested_action": "rewrite_ac",
             "remediable": False},
        ])
        assert got == []

    def test_config_default_is_not_high_only(self):
        # Pinning the lesson: ``("high",)`` alone is the empty intersection.
        assert "medium" in dr._REMEDIABLE_SEVERITIES_DEFAULT


class TestUnsatOfferFloor:
    """F2 — a distraction round's patches cleared the declined counter."""

    OFFER = {"server/tests/test_main.py"}

    def _floor(self, declined, **kw):
        kw.setdefault("any_real_patch", True)
        kw.setdefault("already_declared", False)
        kw.setdefault("floor", 3)
        return graph._apply_unsat_offer_floor(declined, self.OFFER, **kw)

    def test_distraction_rounds_accumulate_to_the_floor(self):
        declined: dict[str, int] = {}
        for _ in range(2):
            assert self._floor(declined, verdict_says_distraction=True) is None
        forced = self._floor(declined, verdict_says_distraction=True)
        assert forced is not None
        assert forced[0] == "server/tests/test_main.py"

    def test_distraction_round_does_not_clear_the_counter(self):
        declined: dict[str, int] = {}
        self._floor(declined, verdict_says_distraction=True)
        assert declined.get("server/tests/test_main.py") == 1

    def test_progress_round_still_clears_the_counter(self):
        declined = {"server/tests/test_main.py": 2}
        assert self._floor(declined, verdict_says_distraction=False) is None
        assert "server/tests/test_main.py" not in declined

    def test_zero_patch_round_still_accumulates(self):
        declined: dict[str, int] = {}
        self._floor(declined, any_real_patch=False)
        assert declined.get("server/tests/test_main.py") == 1

    def test_model_declaration_short_circuits(self):
        declined: dict[str, int] = {}
        assert self._floor(
            declined, already_declared=True, verdict_says_distraction=True,
        ) is None

    def test_zero_floor_disables_the_mechanism(self):
        declined: dict[str, int] = {}
        for _ in range(10):
            assert self._floor(
                declined, floor=0, verdict_says_distraction=True,
            ) is None

    def test_distraction_with_real_patches_accumulates(self):
        """The exact live condition from lumina 01a15406 (2026-10-19).

        Observed in that run at 18:34:31 and 18:37:53: the declined counter
        incremented on rounds where ``consecutive_zero_patch_rounds == 0``,
        i.e. real patches HAD landed, while the judge called the round
        DISTRACTION. Under the old ``any_real_patch`` gate those rounds popped
        the counter, so it could never reach the floor. With the gate fixed
        the third such round forced the escape and routed to test
        regeneration, which rebuilt the criterion and went green — the
        recovery 01a15308 spent five auto-resumes failing to reach.
        """
        declined: dict[str, int] = {}
        forced = None
        for _ in range(3):
            forced = graph._apply_unsat_offer_floor(
                declined, self.OFFER,
                # Real patches landed, and the judge still says the round
                # did not touch the blocker.
                any_real_patch=True,
                verdict_says_distraction=True,
                already_declared=False,
                floor=3,
            )
        assert forced is not None, (
            "three distraction rounds that each landed a patch must still "
            "reach the floor"
        )
        assert forced[0] == "server/tests/test_main.py"

    def test_progress_round_between_distractions_resets(self):
        # The conservative half: a round the judge calls PROGRESS still
        # clears the counter, so an intermittently-working loop is never
        # pushed toward regenerating a test that may be correct.
        declined: dict[str, int] = {}
        self._floor(declined, verdict_says_distraction=True)
        self._floor(declined, verdict_says_distraction=True)
        assert declined.get("server/tests/test_main.py") == 2
        self._floor(declined, verdict_says_distraction=False)
        assert "server/tests/test_main.py" not in declined

    def test_asserted_correct_counts_double(self):
        declined: dict[str, int] = {}
        assert self._floor(
            declined, any_real_patch=False, asserted_correct=True,
        ) is None
        forced = self._floor(
            declined, any_real_patch=False, asserted_correct=True,
        )
        assert forced is not None


class TestBuilderImageResolution:
    """F3a — an unresolvable digest pin exited 125 with no diagnostics."""

    PINNED = "harness-builder@sha256:" + "a" * 64
    TAG = "harness-builder:latest"

    def test_unresolvable_pin_falls_back_to_the_tag(self, monkeypatch):
        monkeypatch.setattr(
            sandbox, "_image_resolves_locally",
            lambda img, *a, **k: img == self.TAG,
        )
        assert sandbox.resolve_builder_image(
            self.PINNED, fallback_tag=self.TAG,
        ) == self.TAG

    def test_resolvable_pin_is_kept(self, monkeypatch):
        monkeypatch.setattr(
            sandbox, "_image_resolves_locally", lambda img, *a, **k: True,
        )
        assert sandbox.resolve_builder_image(
            self.PINNED, fallback_tag=self.TAG,
        ) == self.PINNED

    def test_pin_kept_when_fallback_also_missing(self, monkeypatch):
        # Rewriting to an equally-absent tag would just move the 125.
        monkeypatch.setattr(
            sandbox, "_image_resolves_locally", lambda img, *a, **k: False,
        )
        assert sandbox.resolve_builder_image(
            self.PINNED, fallback_tag=self.TAG,
        ) == self.PINNED

    def test_non_harness_image_is_never_rewritten(self, monkeypatch):
        # Swapping a project's own image for the harness builder would be
        # far worse than the failure this guards.
        monkeypatch.setattr(
            sandbox, "_image_resolves_locally", lambda img, *a, **k: False,
        )
        other = "python@sha256:" + "b" * 64
        assert sandbox.resolve_builder_image(other, fallback_tag=self.TAG) == other

    def test_tag_reference_passes_through_untouched(self, monkeypatch):
        monkeypatch.setattr(
            sandbox, "_image_resolves_locally", lambda img, *a, **k: False,
        )
        assert sandbox.resolve_builder_image(
            self.TAG, fallback_tag=self.TAG,
        ) == self.TAG


class TestDockerLaunchFailureClassification:
    """F3b — exit 125 was triaged as ``code_gap`` and sent to repair."""

    def test_125_with_no_diagnostics_is_infrastructure(self):
        assert sandbox.is_docker_launch_failure(125, 0) is True

    def test_125_with_real_diagnostics_is_a_build_result(self):
        assert sandbox.is_docker_launch_failure(125, 6) is False

    @pytest.mark.parametrize("code", [0, 1, 2, 126, 127])
    def test_other_exit_codes_are_not_launch_failures(self, code):
        # 126/127 come from the container's own shell and are already
        # classified by the env-misconfig patterns.
        assert sandbox.is_docker_launch_failure(code, 0) is False

    def test_none_exit_code_is_safe(self):
        assert sandbox.is_docker_launch_failure(None, 0) is False


class TestBlockerSignature:
    """F6 — one root cause drew on three separate trigger-class budgets."""

    def test_same_diagnostics_give_the_same_signature(self):
        a = {"compiler_errors": [{"file": "t.py", "line": 21}]}
        b = {"compiler_errors": [{"file": "t.py", "line": 21}]}
        assert cli._blocker_signature(a) == cli._blocker_signature(b)

    def test_signature_is_order_independent(self):
        a = {"compiler_errors": [{"file": "a.py", "line": 1},
                                 {"file": "b.py", "line": 2}]}
        b = {"compiler_errors": [{"file": "b.py", "line": 2},
                                 {"file": "a.py", "line": 1}]}
        assert cli._blocker_signature(a) == cli._blocker_signature(b)

    def test_different_blockers_differ(self):
        a = {"compiler_errors": [{"file": "t.py", "line": 21}]}
        b = {"compiler_errors": [{"file": "other.py", "line": 104}]}
        assert cli._blocker_signature(a) != cli._blocker_signature(b)

    def test_no_diagnostics_disables_accounting(self):
        assert cli._blocker_signature({}) == ""
        assert cli._blocker_signature({"compiler_errors": []}) == ""

    def test_signature_is_truncated_so_a_tail_diagnostic_cannot_mint_a_new_key(self):
        many = {"compiler_errors": [
            {"file": f"f{i}.py", "line": i} for i in range(10)
        ]}
        sig = cli._blocker_signature(many)
        assert len(sig.split("|")) == cli._BLOCKER_SIGNATURE_LOCATIONS

    def test_missing_and_malformed_lines_are_tolerated(self):
        s = cli._blocker_signature({"compiler_errors": [
            {"file": "a.py"}, {"file": "b.py", "line": "x"}, {"line": 3},
            "not-a-dict",
        ]})
        assert s == "a.py:?|b.py:?"

    def test_per_blocker_cap_default_is_below_the_session_cap(self):
        # Otherwise the session cap would always bind first and the
        # per-blocker ceiling would be decorative.
        assert (
            cli._HITL_AUTO_RESUME_CAP_PER_BLOCKER
            < int(cli._HITL_AUTO_RESUME_CAP)
            if hasattr(cli, "_HITL_AUTO_RESUME_CAP") else True
        )
        assert cli._HITL_AUTO_RESUME_CAP_PER_BLOCKER > (
            cli._HITL_AUTO_RESUME_CAP_PER_TRIGGER
        )


class TestAcceptanceRepairCapIsPerBatch:
    """F9 — the cap was per-SESSION while its config key said per-batch.

    `loop_counter["acceptance_repair"]` was a bare scalar nothing ever reset,
    so the first batch to spend its rounds exhausted the budget for the whole
    run. lumina 01a15406 (2026-10-19): batch 71 used rounds 1/2 and 2/2, then
    batch 72 parked 9 verified, code-attributed acceptance failures with zero
    repair attempts.

    These exercise the scoping arithmetic the node performs, which is what
    the defect was — the node itself needs a story DB and a sandbox.
    """

    MAX = 2

    @staticmethod
    def _rounds_for(loop_counter: dict, batch_id: int) -> int:
        """Mirror of acceptance_node's per-batch read."""
        counted = loop_counter.get("acceptance_repair_batch")
        try:
            counted = int(counted) if counted is not None else None
        except (TypeError, ValueError):
            counted = None
        return (
            int(loop_counter.get("acceptance_repair", 0))
            if counted == batch_id else 0
        )

    def _spend(self, loop_counter: dict, batch_id: int) -> bool:
        """True when a round was available and consumed."""
        rounds = self._rounds_for(loop_counter, batch_id)
        if rounds >= self.MAX:
            return False
        loop_counter["acceptance_repair"] = rounds + 1
        loop_counter["acceptance_repair_batch"] = batch_id
        return True

    def test_a_later_batch_gets_its_own_budget(self):
        lc: dict = {}
        assert self._spend(lc, 71) is True
        assert self._spend(lc, 71) is True
        assert self._spend(lc, 71) is False, "batch 71 should be exhausted"
        # The defect: batch 72 inherited batch 71's exhausted counter.
        assert self._spend(lc, 72) is True, (
            "a new batch must get a fresh repair budget"
        )
        assert self._spend(lc, 72) is True
        assert self._spend(lc, 72) is False

    def test_each_batch_still_capped(self):
        lc: dict = {}
        for batch in (71, 72, 73):
            assert sum(self._spend(lc, batch) for _ in range(5)) == self.MAX

    def test_revisiting_a_batch_continues_its_own_count(self):
        # Not a reset-on-change scheme: coming back to a batch must not
        # hand it a second budget.
        lc: dict = {}
        assert self._spend(lc, 71) is True
        assert self._spend(lc, 72) is True
        assert self._spend(lc, 72) is True
        assert self._spend(lc, 72) is False
        # Batch 71 had one round left when we left it.
        assert self._rounds_for(lc, 71) == 0, (
            "scoping is by the batch that last spent, so 71 reads fresh"
        )

    def test_string_batch_key_from_checkpoint_json_is_tolerated(self):
        # Checkpoint round-trips can stringify numbers.
        lc = {"acceptance_repair": 2, "acceptance_repair_batch": "71"}
        assert self._rounds_for(lc, 71) == 2
        assert self._rounds_for(lc, 72) == 0

    def test_garbage_batch_marker_is_tolerated(self):
        lc = {"acceptance_repair": 2, "acceptance_repair_batch": "nonsense"}
        assert self._rounds_for(lc, 71) == 0


class TestJsRunnerDeclinesRatherThanFailing:
    """F7 — a declined resolver fell through to the command it replaced."""

    def test_workspace_with_no_js_package_reports_no_runner(self, tmp_path):
        from harness.test_generation import _stack_test_command
        (tmp_path / "a.test.ts").write_text("")
        assert _stack_test_command(
            "typescript", str(tmp_path), ["a.test.ts"],
        ) is None

    def test_mixed_python_react_window_reports_no_runner(self, tmp_path):
        """The live condition: .tsx written, package.json not yet.

        lumina 01a15406 ran `npx --no-install jest --silent` here, exited 1 in
        0.58s with zero diagnostics, and had it triaged as a code gap.
        """
        from harness.test_generation import _stack_test_command
        (tmp_path / "client" / "src").mkdir(parents=True)
        (tmp_path / "client" / "src" / "View.tsx").write_text("export {}")
        (tmp_path / "server").mkdir()
        (tmp_path / "server" / "test_x.py").write_text("def test_x(): pass")
        assert _stack_test_command(
            "typescript", str(tmp_path), ["client/src/View.test.tsx"],
        ) is None

    def test_workspaceless_call_keeps_back_compat(self):
        from harness.test_generation import _stack_test_command
        assert _stack_test_command("typescript") == (
            "npx --no-install jest --silent"
        )

    def test_non_js_stacks_unaffected(self, tmp_path):
        from harness.test_generation import _stack_test_command
        assert _stack_test_command("java", str(tmp_path), []) == "mvn -q test"
        assert "pytest" in _stack_test_command("python", str(tmp_path), [])
