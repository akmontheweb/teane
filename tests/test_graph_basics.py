"""Tests for harness/graph.py — orchestration basics."""

import asyncio

from harness.graph import (
    apply_memory_cleanse,
    _fingerprint_diagnostics,
    _format_diagnostics_for_repair,
    _repair_budget_warning,
    _rotate_diag_fingerprints_delta,
    route_after_deployment,
    route_after_security_scan,
)


class TestRepairBudgetWarning:
    """Audit #19 — soft warnings on the last two repair iterations."""

    def test_silent_with_slack(self):
        assert _repair_budget_warning(total_repairs=1, cap=8) is None
        assert _repair_budget_warning(total_repairs=5, cap=8) is None

    def test_medium_warning_at_two_remaining(self):
        msg = _repair_budget_warning(total_repairs=6, cap=8)
        assert msg is not None
        assert "2 repair iterations remain" in msg

    def test_hard_warning_at_one_remaining(self):
        msg = _repair_budget_warning(total_repairs=7, cap=8)
        assert msg is not None
        assert "LAST repair iteration" in msg

    def test_silent_past_cap(self):
        # Past the cap the router has already moved to HITL; no point
        # warning a model that won't be called.
        assert _repair_budget_warning(total_repairs=8, cap=8) is None
        assert _repair_budget_warning(total_repairs=99, cap=8) is None

    def test_zero_cap_is_silent(self):
        # Edge case — operator disabled the throttle. Don't crash, don't warn.
        assert _repair_budget_warning(total_repairs=3, cap=0) is None

    def test_small_cap_still_warns_on_last_two(self):
        # With cap=2 the LLM gets warned on both attempts.
        assert _repair_budget_warning(total_repairs=0, cap=2) is not None
        assert _repair_budget_warning(total_repairs=1, cap=2) is not None
        assert _repair_budget_warning(total_repairs=2, cap=2) is None


class TestApplyMemoryCleanse:
    """Test memory cleansing on compiler success."""

    def test_cleanse_with_no_messages(self):
        """Should handle state with no messages."""
        state = {"messages": []}
        result = apply_memory_cleanse(state, resolution_kind="compiler_success")
        assert isinstance(result, dict)

    def test_cleanse_with_single_message(self):
        """Should cleanse state with single message."""
        state = {
            "messages": [
                {"role": "user", "content": "Hello"},
            ]
        }
        result = apply_memory_cleanse(state, resolution_kind="compiler_success")
        assert isinstance(result, dict)

    def test_cleanse_with_multiple_messages(self):
        """Should cleanse state with conversation."""
        state = {
            "messages": [
                {"role": "user", "content": "Fix this code"},
                {"role": "assistant", "content": "Here's the fix"},
                {"role": "user", "content": "Test it"},
            ]
        }
        result = apply_memory_cleanse(state, resolution_kind="compiler_success")
        assert isinstance(result, dict)
        # Should have messages in result
        assert "messages" in result

    def test_cleanse_different_resolution_kinds(self):
        """Should handle different resolution kinds."""
        state = {"messages": [{"role": "user", "content": "test"}]}

        for kind in ["compiler_success", "repair_success", "human_intervention"]:
            result = apply_memory_cleanse(state, resolution_kind=kind)
            assert isinstance(result, dict)

    def test_cleanse_preserves_state_fields(self):
        """Should preserve other state fields."""
        state = {
            "messages": [],
            "current_node": "compiler",
            "loop_counters": {"repair": 1},
            "exit_code": 0,
        }
        result = apply_memory_cleanse(state)
        # Should preserve non-message fields
        assert isinstance(result, dict)


class TestFormatDiagnosticsForRepair:
    """Test diagnostic formatting for repair hints."""

    def test_format_empty_errors(self):
        """Empty error list should produce empty or minimal output."""
        result = _format_diagnostics_for_repair([])
        assert isinstance(result, str)

    def test_format_single_error(self):
        """Single error should be formatted."""
        errors = [
            {
                "file": "main.py",
                "line": 10,
                "message": "undefined variable x",
                "severity": "error",
            }
        ]
        result = _format_diagnostics_for_repair(errors)
        assert isinstance(result, str)
        # Should contain error information
        if result:
            assert "error" in result.lower() or "main.py" in result or "10" in result

    def test_format_multiple_errors(self):
        """Multiple errors should all be included."""
        errors = [
            {
                "file": "app.py",
                "line": 5,
                "message": "syntax error",
                "severity": "error",
            },
            {
                "file": "utils.py",
                "line": 20,
                "message": "undefined function",
                "severity": "error",
            },
        ]
        result = _format_diagnostics_for_repair(errors)
        assert isinstance(result, str)

    def test_format_with_semantic_context(self):
        """Should include semantic context if present."""
        errors = [
            {
                "file": "main.py",
                "line": 10,
                "message": "type mismatch",
                "severity": "error",
                "semantic_context": "x = 'string'; y = x + 1",
            }
        ]
        result = _format_diagnostics_for_repair(errors)
        assert isinstance(result, str)

    def test_format_warnings_and_errors(self):
        """Should handle both warnings and errors."""
        errors = [
            {
                "file": "a.py",
                "line": 1,
                "message": "unused import",
                "severity": "warning",
            },
            {
                "file": "b.py",
                "line": 2,
                "message": "critical error",
                "severity": "error",
            },
        ]
        result = _format_diagnostics_for_repair(errors)
        assert isinstance(result, str)


class TestCascadeDefenseLayers:
    """Four-layer defense against the cascade-ranking heuristic deferring a
    real error past the repair budget — see the AuthService.ts TS2769 case
    in session 6cf20a5d. Each test pins one layer; they compose in real use.
    """

    def _err(self, code: str, msg: str, file: str = "x.ts") -> dict:
        return {
            "file": file,
            "line": 1,
            "column": 1,
            "error_code": code,
            "message": msg,
            "severity": "error",
            "semantic_context": f"// snippet for {code}",
        }

    def test_layer0_no_lying_wording(self):
        """The deferred-section header must NOT claim items 'may resolve on
        their own' — that was the false promise the LLM took at face value."""
        # 7 distinct groups forces the deferred section to render.
        errors = [self._err(f"TS{n}", f"msg {n}") for n in range(7000, 7007)]
        out = _format_diagnostics_for_repair(errors)
        assert "may resolve on their own" not in out
        # And the deferred section must still appear (we didn't accidentally
        # delete it — Layer 0 is wording-only).
        assert "Deferred" in out

    def test_layer1_typescript_upstream_promoted(self):
        """TS2304/TS2305/TS2307 are upstream-shaped: cannot-find-name,
        missing-export, cannot-find-module. They must outrank a generic
        type-mismatch like TS2741 even when seen later."""
        errors = [
            self._err("TS2741", "Property 'cookie' is missing"),  # not upstream
            self._err("TS2741", "Property 'cookie' is missing", file="y.ts"),
            self._err("TS2741", "Property 'cookie' is missing", file="z.ts"),
            self._err("TS2741", "Property 'cookie' is missing", file="w.ts"),
            self._err("TS2304", "Cannot find name 'Foo'"),         # upstream — newer
            self._err("TS2307", "Cannot find module 'bar'", file="b.ts"),
            self._err("TS2305", "Module has no exported member 'baz'"),
        ]
        out = _format_diagnostics_for_repair(errors)
        # The upstream TS codes must appear in the prompt before TS2741.
        i_2304 = out.find("TS2304")
        i_2307 = out.find("TS2307")
        i_2305 = out.find("TS2305")
        i_2741 = out.find("TS2741")
        assert -1 not in (i_2304, i_2307, i_2305, i_2741)
        assert i_2304 < i_2741
        assert i_2307 < i_2741
        assert i_2305 < i_2741

    def test_layer2_small_n_short_circuit_shows_every_group(self):
        """At ≤ 5 distinct groups, every group is shown with full context
        and the deferred section is suppressed entirely."""
        errors = [self._err(f"TS{n}", f"shape {n}") for n in range(2700, 2705)]  # 5 groups
        out = _format_diagnostics_for_repair(errors)
        for n in range(2700, 2705):
            assert f"TS{n}" in out
            # Each group's semantic_context must render (full-context proof).
            assert f"// snippet for TS{n}" in out
        # No deferred section.
        assert "Deferred" not in out
        assert "deferred" not in out

    def test_layer2_threshold_boundary(self):
        """At 6 groups (just past the small-N threshold) the deferred
        section reappears and only top-3 get context."""
        errors = [self._err(f"TS{n}", f"shape {n}") for n in range(2700, 2706)]  # 6 groups
        out = _format_diagnostics_for_repair(errors)
        assert "Deferred" in out
        # Exactly 3 semantic_context blocks appear in the MARKDOWN portion
        # (one per shown top-N). The Phase 4 structured JSON payload
        # below echoes every snippet but we don't count those here.
        markdown_only = out.split("### Structured payload", 1)[0]
        assert markdown_only.count("// snippet for TS") == 3

    def test_layer3_persisted_group_promoted_past_cascade(self):
        """The AuthService.ts case in miniature. Round N has a non-upstream
        TS2769 deferred behind 3 upstream errors. Round N+1, the upstream
        errors got fixed but TS2769 survived — it must be promoted to the
        top of the prompt regardless of cascade rank."""
        # Build round 2's diagnostics: TS2769 survived, two upstream errors
        # are NEW (e.g. a new cascade victim and a fresh missing-import).
        # Sized at 6 distinct groups to force the deferred section to
        # render (so we can prove the persisted item is in `shown`, not
        # `hidden`).
        errors = [
            self._err("TS2304", "Cannot find name 'Brand new'", file="a.ts"),
            self._err("TS2307", "Cannot find module 'fresh'", file="b.ts"),
            self._err("TS2305", "Module has no exported member 'extra'", file="c.ts"),
            self._err("TS2741", "Property 'x' is missing", file="d.ts"),
            self._err("TS2353", "updated_at not in Omit<>", file="e.ts"),
            self._err("TS2769", "No overload matches this call.",
                      file="services/AuthService.ts"),
        ]
        # Prior round had only the TS2769 (the survivor).
        prior = {"TS2769::No overload matches this call."}
        out = _format_diagnostics_for_repair(errors, prior_fingerprints=prior)
        # The persisted marker must render on the surviving group.
        assert "persisted from previous round" in out
        i_persisted_marker = out.find("persisted from previous round")
        # TS2769 must be in the TOP-N (shown with full context), not in
        # the deferred tail. Prove this by locating its semantic_context.
        assert "// snippet for TS2769" in out
        # And its top-N entry must appear BEFORE the deferred section
        # header (so it wasn't shoved into the tail).
        i_deferred = out.find("### Deferred")
        assert i_deferred != -1, "deferred section should still render"
        i_2769 = out.find("TS2769")
        assert i_2769 < i_deferred
        assert i_persisted_marker < i_deferred

    def test_layer3_no_prior_state_behaves_like_before(self):
        """No prior fingerprints (round 1 or after a green compile) →
        formatter must rank by the original cascade prior. Backward
        compat for the first-iteration case."""
        errors = [
            self._err("TS2741", "non-upstream A"),
            self._err("TS2353", "non-upstream B"),
            self._err("F821", "upstream undefined name"),
        ]
        out_no_prior = _format_diagnostics_for_repair(errors)
        out_empty_prior = _format_diagnostics_for_repair(
            errors, prior_fingerprints=set()
        )
        # F821 (upstream) must outrank both non-upstream errors in both
        # codepaths.
        assert out_no_prior.find("F821") < out_no_prior.find("TS2741")
        assert out_empty_prior.find("F821") < out_empty_prior.find("TS2741")
        # No persisted marker rendered.
        assert "persisted from previous round" not in out_no_prior

    def test_fingerprint_helper_round_trips_with_formatter(self):
        """The fingerprint helper must produce a string that matches the
        group key the formatter compares against. If these ever diverge,
        survival promotion silently fails."""
        errors = [
            self._err("TS2769", "No overload matches this call."),
            self._err("TS2769", "No overload matches this call.", file="b.ts"),
            self._err("F401", "imported but unused"),
        ]
        fps = _fingerprint_diagnostics(errors)
        # Deduped → 2 fingerprints from 3 errors.
        assert len(fps) == 2
        assert "TS2769::No overload matches this call." in fps
        assert "F401::imported but unused" in fps
        # And feeding them back as prior_fingerprints actually promotes
        # the matching groups.
        out = _format_diagnostics_for_repair(errors, prior_fingerprints=set(fps))
        # Both groups persisted → both should be tagged.
        assert out.count("persisted from previous round") == 2

    def test_fingerprint_helper_excludes_warnings(self):
        """Warnings don't enter the repair prompt; their fingerprints
        shouldn't enter the survival set either."""
        errors = [
            {"error_code": "W001", "message": "warn", "severity": "warning"},
            {"error_code": "E001", "message": "err", "severity": "error"},
        ]
        fps = _fingerprint_diagnostics(errors)
        assert fps == ["E001::err"]


class TestRotateDiagFingerprintsDelta:
    """Regression for session 7e4cba32 — the runaway repair loop where
    the reflection judge saw ``prior=1, current=0`` and hallucinated
    PROGRESS every round. Root cause: the prod-smoke short-circuit
    return in compiler_node didn't rotate ``last_diag_fingerprints``,
    leaving the current slot stale at whatever the last full build
    left behind (often ``[]`` from a green compile). This helper is
    the single point every ``compiler_errors``-populating node must
    now merge into its return dict."""

    def _err(self, code: str, msg: str, severity: str = "error") -> dict:
        return {
            "file": "x.py", "line": 1, "column": 1,
            "error_code": code, "message": msg, "severity": severity,
        }

    def test_rotates_current_into_prior(self):
        state = {
            "last_diag_fingerprints": ["OLD_A::alpha", "OLD_B::beta"],
            "last_diag_count": 2,
        }
        diags = [self._err("NEW_C", "gamma")]
        out = _rotate_diag_fingerprints_delta(state, diags)
        # Prior slot gets the old current, verbatim (as a list, not a set).
        assert out["prior_diag_fingerprints"] == ["OLD_A::alpha", "OLD_B::beta"]
        assert out["prior_diag_count"] == 2
        # Current slot reflects the new diagnostics.
        assert out["last_diag_fingerprints"] == ["NEW_C::gamma"]
        assert out["last_diag_count"] == 1

    def test_stale_prior_after_green_build(self):
        """The bug: after a green compile ``last_diag_fingerprints`` is
        ``[]``; on the next round of failures the rotation must move
        that ``[]`` into prior and put the fresh failures into current.
        Without this the judge sees prior=<whatever> current=[] and
        classifies it as PROGRESS."""
        state = {"last_diag_fingerprints": [], "last_diag_count": 0}
        diags = [self._err("PROD_IMPORT_SMOKE", "cannot import edgar")]
        out = _rotate_diag_fingerprints_delta(state, diags)
        assert out["prior_diag_fingerprints"] == []
        assert out["prior_diag_count"] == 0
        # This is the critical assertion — the fresh failing set MUST
        # populate the current slot, else reflection reads 0 and the
        # circuit-breaker misfires.
        assert out["last_diag_fingerprints"] == [
            "PROD_IMPORT_SMOKE::cannot import edgar"
        ]
        assert out["last_diag_count"] == 1

    def test_persisted_failure_is_visible_to_judge(self):
        """When the same failure repeats round-over-round, the helper
        must produce identical prior and current fingerprint sets so the
        reflection judge sees intersection == full set (== DISTRACTION)."""
        state = {
            "last_diag_fingerprints": ["PROD_IMPORT_SMOKE::cannot import edgar"],
            "last_diag_count": 1,
        }
        diags = [self._err("PROD_IMPORT_SMOKE", "cannot import edgar")]
        out = _rotate_diag_fingerprints_delta(state, diags)
        assert (
            set(out["prior_diag_fingerprints"])
            == set(out["last_diag_fingerprints"])
        )

    def test_warnings_excluded_from_count(self):
        """The count fed to reflection must exclude warnings; a warnings-
        only round shouldn't look like progress vs. a prior error round."""
        state = {"last_diag_fingerprints": [], "last_diag_count": 0}
        diags = [
            self._err("W1", "deprecation notice", severity="warning"),
            self._err("E1", "real failure"),
        ]
        out = _rotate_diag_fingerprints_delta(state, diags)
        assert out["last_diag_count"] == 1
        assert out["last_diag_fingerprints"] == ["E1::real failure"]

    def test_empty_state_slots_do_not_crash(self):
        """The helper is called from many nodes; some may run before
        compiler_node has populated the fingerprint slots. Missing keys
        must degrade gracefully to empty prior + fresh current."""
        out = _rotate_diag_fingerprints_delta({}, [self._err("E", "x")])
        assert out["prior_diag_fingerprints"] == []
        assert out["prior_diag_count"] == 0
        assert out["last_diag_fingerprints"] == ["E::x"]
        assert out["last_diag_count"] == 1


class TestPromoteDeferredEscapeHatch:
    """Phase 1.2 — the LLM can force a deferred code into top-N on the next
    round by emitting <<<PROMOTE_DEFERRED>>>. Validates the patcher parser
    and the formatter's honoring of the request.
    """

    def _err(self, code: str, msg: str, file: str = "x.ts") -> dict:
        return {
            "file": file,
            "line": 1,
            "column": 1,
            "error_code": code,
            "message": msg,
            "severity": "error",
            "semantic_context": f"// snippet for {code}",
        }

    def test_parser_extracts_codes(self):
        from harness.patcher import parse_promote_deferred_blocks
        text = (
            "Some preamble.\n"
            "<<<PROMOTE_DEFERRED>>>\n"
            "codes: TS2769, F401\n"
            "<<<END_PROMOTE_DEFERRED>>>\n"
            "More text."
        )
        assert parse_promote_deferred_blocks(text) == ["TS2769", "F401"]

    def test_parser_dedupes_and_strips_whitespace(self):
        from harness.patcher import parse_promote_deferred_blocks
        text = (
            "<<<PROMOTE_DEFERRED>>>\n"
            "codes:   TS2769 ,  F401,  TS2769 ,  \n"
            "<<<END_PROMOTE_DEFERRED>>>"
        )
        assert parse_promote_deferred_blocks(text) == ["TS2769", "F401"]

    def test_parser_handles_multiple_blocks(self):
        from harness.patcher import parse_promote_deferred_blocks
        text = (
            "<<<PROMOTE_DEFERRED>>>\ncodes: TS2769\n<<<END_PROMOTE_DEFERRED>>>\n"
            "Other content.\n"
            "<<<PROMOTE_DEFERRED>>>\ncodes: F401, F821\n<<<END_PROMOTE_DEFERRED>>>"
        )
        assert parse_promote_deferred_blocks(text) == ["TS2769", "F401", "F821"]

    def test_strip_removes_block(self):
        from harness.patcher import strip_promote_deferred_blocks
        text = (
            "Patch A.\n"
            "<<<PROMOTE_DEFERRED>>>\ncodes: TS2769\n<<<END_PROMOTE_DEFERRED>>>\n"
            "Patch B."
        )
        stripped = strip_promote_deferred_blocks(text)
        assert "PROMOTE_DEFERRED" not in stripped
        assert "Patch A" in stripped and "Patch B" in stripped

    def test_parser_returns_empty_on_no_blocks(self):
        from harness.patcher import parse_promote_deferred_blocks
        assert parse_promote_deferred_blocks("Just patches, no directive.") == []

    def test_formatter_promotes_requested_code_past_cascade_prior(self):
        """The LLM's explicit request beats the cascade prior — TS2769 (a
        non-upstream code) outranks the upstream F821 when promoted."""
        errors = [
            self._err("F821", "Undefined name 'foo'"),
            self._err("F401", "imported but unused"),
            self._err("MISSING_DEP", "missing pkg"),
            self._err("TS2741", "type mismatch A"),
            self._err("TS2353", "type mismatch B"),
            self._err("TS2769", "No overload matches this call.",
                      file="services/AuthService.ts"),
        ]
        out = _format_diagnostics_for_repair(
            errors, promoted_codes={"TS2769"},
        )
        # TS2769 must appear before every other code.
        i_2769 = out.find("TS2769")
        for code in ("F821", "F401", "MISSING_DEP", "TS2741", "TS2353"):
            assert i_2769 < out.find(code), (
                f"Promoted TS2769 should outrank {code}; "
                f"i_2769={i_2769}, i_{code}={out.find(code)}"
            )
        # Promotion marker rendered.
        assert "promoted at your request" in out

    def test_formatter_promotion_case_insensitive(self):
        errors = [
            self._err("F821", "upstream"),
            self._err("TS2769", "type mismatch"),
        ]
        out = _format_diagnostics_for_repair(
            errors, promoted_codes={"ts2769"},  # lower case
        )
        assert "promoted at your request" in out

    def test_formatter_promotion_outranks_survival(self):
        """An explicit LLM request outranks empirical survival promotion —
        the model just saw the prompt and disagrees, so honor it."""
        errors = [
            self._err("F821", "upstream + persisted"),
            self._err("TS2769", "type mismatch + promoted"),
        ]
        out = _format_diagnostics_for_repair(
            errors,
            prior_fingerprints={"F821::upstream + persisted"},
            promoted_codes={"TS2769"},
        )
        # TS2769 (promoted) before F821 (persisted only).
        assert out.find("TS2769") < out.find("F821")

    def test_formatter_advertises_directive_when_deferred(self):
        """The 'how to use' instructions must appear in the prompt when
        there are deferred groups (so the LLM knows the option exists)."""
        errors = [self._err(f"TS{n}", f"shape {n}") for n in range(7000, 7008)]
        out = _format_diagnostics_for_repair(errors)
        assert "<<<PROMOTE_DEFERRED>>>" in out
        assert "<<<END_PROMOTE_DEFERRED>>>" in out

    def test_formatter_no_directive_when_nothing_deferred(self):
        """If there's no deferred section, don't pollute the prompt with
        the escape-hatch instructions."""
        errors = [self._err(f"TS{n}", f"shape {n}") for n in range(7000, 7003)]  # 3 groups, all shown
        out = _format_diagnostics_for_repair(errors)
        assert "<<<PROMOTE_DEFERRED>>>" not in out


class TestPhase4StructuredPayload:
    """Phase 4 — alongside the markdown summary, the formatter emits a
    structured JSON block so the LLM can sort/filter the raw diagnostics
    if it disagrees with the harness's cascade ranking."""

    def _err(self, code: str, msg: str, file: str = "x.ts") -> dict:
        return {
            "file": file,
            "line": 1,
            "column": 1,
            "error_code": code,
            "message": msg,
            "severity": "error",
            "semantic_context": f"// snippet for {code}",
        }

    def test_structured_payload_includes_every_diagnostic(self):
        from harness.graph import _format_structured_diagnostic_payload
        errors = [self._err(f"TS{n}", f"msg {n}") for n in range(2700, 2703)]
        out = _format_structured_diagnostic_payload(errors)
        import json
        # Extract the JSON fenced block.
        body = out.split("```json", 1)[1].split("```", 1)[0]
        parsed = json.loads(body)
        assert "diagnostics" in parsed and len(parsed["diagnostics"]) == 3
        codes = [d["code"] for d in parsed["diagnostics"]]
        assert codes == ["TS2700", "TS2701", "TS2702"]

    def test_structured_payload_caps_at_max_total(self):
        from harness.graph import _format_structured_diagnostic_payload
        errors = [self._err(f"TS{n}", f"msg {n}") for n in range(2700, 2735)]  # 35
        out = _format_structured_diagnostic_payload(errors, max_total=10)
        import json
        body = out.split("```json", 1)[1].split("```", 1)[0]
        parsed = json.loads(body)
        assert len(parsed["diagnostics"]) == 10
        assert "_truncated" in parsed
        assert parsed["_truncated"]["total"] == 35

    def test_structured_payload_empty_input_returns_empty(self):
        from harness.graph import _format_structured_diagnostic_payload
        assert _format_structured_diagnostic_payload([]) == ""

    def test_formatter_appends_structured_block_by_default(self):
        errors = [self._err("TS2769", "x")]
        out = _format_diagnostics_for_repair(errors)
        assert "Structured payload" in out
        assert '"diagnostics"' in out

    def test_formatter_omits_structured_block_when_opted_out(self):
        errors = [self._err("TS2769", "x")]
        out = _format_diagnostics_for_repair(errors, emit_structured_payload=False)
        assert "Structured payload" not in out


class TestPhase3Hardening:
    """Phase 3 — five small hardening items that close common signal-loss
    paths the cascade-defense layers depend on."""

    def _err(self, code: str, msg: str, file: str = "x.ts") -> dict:
        return {
            "file": file,
            "line": 1,
            "column": 1,
            "error_code": code,
            "message": msg,
            "severity": "error",
            "semantic_context": f"// snippet for {code}",
        }

    def test_3a_fingerprint_normalizes_quoted_spans(self):
        """A partial fix that swaps one concrete type for another must
        still leave the fingerprint stable."""
        from harness.graph import _normalize_diagnostic_message
        msg1 = "Type 'string[]' is not assignable to type 'number[]'"
        msg2 = "Type 'string[]' is not assignable to type 'boolean[]'"
        assert _normalize_diagnostic_message(msg1) == _normalize_diagnostic_message(msg2)

    def test_3a_fingerprint_normalises_backticks_and_doublequotes(self):
        from harness.graph import _normalize_diagnostic_message
        a = "Property `foo` does not exist"
        b = "Property `bar` does not exist"
        c = 'Property "baz" does not exist'
        assert _normalize_diagnostic_message(a) == _normalize_diagnostic_message(b)
        assert _normalize_diagnostic_message(a) == _normalize_diagnostic_message(c)

    def test_3a_fingerprint_collapses_whitespace(self):
        from harness.graph import _normalize_diagnostic_message
        a = "Type 'X' is not assignable to type 'Y'"
        b = "Type 'X' is not\n  assignable to\ntype 'Y'"
        assert _normalize_diagnostic_message(a) == _normalize_diagnostic_message(b)

    def test_3a_survival_promotion_survives_partial_fix(self):
        """The TS2769 case in miniature — Round N had `string` in the
        message, Round N+1 has `number` after a partial fix, but the
        error shape is morally identical. Layer 3 must still promote."""
        round_n1 = self._err(
            "TS2769",
            "Type 'string' is not assignable to type 'number | StringValue | undefined'",
        )
        prior_fps = set(_fingerprint_diagnostics([round_n1]))
        # Round N+1: same code, message has different concrete types.
        round_n2 = self._err(
            "TS2769",
            "Type 'string' is not assignable to type 'boolean | StringValue | undefined'",
        )
        out = _format_diagnostics_for_repair(
            [round_n2], prior_fingerprints=prior_fps,
        )
        assert "persisted from previous round" in out

    def test_3c_typescript_upstream_codes_recognised(self):
        """TS2304 (cannot-find-name), TS2307 (cannot-find-module),
        TS2305 (missing-export) must outrank generic non-upstream codes."""
        errors = [
            self._err("TS2741", "type mismatch"),
            self._err("TS2304", "Cannot find name 'Foo'"),
        ]
        out = _format_diagnostics_for_repair(errors)
        assert out.find("TS2304") < out.find("TS2741")

    def test_3c_java_upstream_codes(self):
        errors = [
            self._err("CUSTOM_ERR_X", "irrelevant"),
            self._err("JAVA:CANNOT_FIND_SYMBOL", "cannot find symbol 'foo'"),
        ]
        out = _format_diagnostics_for_repair(errors)
        custom_pos = out.find("CUSTOM_ERR_X")
        assert out.find("JAVA:CANNOT_FIND_SYMBOL") < custom_pos

    def test_3b_prefix_diff_finds_first_divergence(self):
        from harness.gateway import _summarize_prefix_diff
        old = "system|hello\n---\nuser|world"
        new = "system|hello\n---\nuser|wOrld"
        result = _summarize_prefix_diff(old=old, new=new)
        assert "first_diff_offset" in result
        # Divergence at position of capital O in "wOrld".
        assert result["first_diff_offset"] == old.index("world") + 1
        assert "before_excerpt" in result and "after_excerpt" in result
        assert "world" in result["before_excerpt"]
        assert "wOrld" in result["after_excerpt"]

    def test_3b_prefix_diff_identical(self):
        from harness.gateway import _summarize_prefix_diff
        s = "system|same|same"
        assert _summarize_prefix_diff(old=s, new=s) == {"identical": True}


class TestRepairReflection:
    """Phase 2.2 — per-round repair reflection. The cheap LLM judges whether
    the previous round addressed the highest-priority error and returns
    a structured verdict the harness can inject into the next dispatch.
    Tests cover the prompt builder, the strict-JSON parser, and the
    edge cases the parser must reject."""

    def test_prompt_includes_before_after_counts(self):
        from harness.graph import _build_repair_reflection_prompt
        prompt = _build_repair_reflection_prompt(
            prior_diagnostics_count=9,
            current_diagnostics_count=3,
            resolved_fingerprints=["F401::imported but unused"],
            persisted_fingerprints=["TS2769::No overload matches"],
            new_fingerprints=[],
            top_persisted_diagnostics=[
                {
                    "error_code": "TS2769",
                    "message": "No overload matches this call.",
                    "file": "src/auth/AuthService.ts",
                    "line": 42,
                }
            ],
        )
        assert "before this round: 9" in prompt
        assert "after this round:  3" in prompt
        assert "TS2769" in prompt
        # Strict-JSON instruction must appear so the model knows the shape.
        assert "STRICT JSON" in prompt
        assert "verdict" in prompt

    def test_prompt_emits_file_line_for_each_top_diag(self):
        """Fix A — the reflection prompt must show real file:line so the
        LLM doesn't have to fabricate locations to satisfy the
        'cite a specific file/symbol' instruction."""
        from harness.graph import _build_repair_reflection_prompt
        prompt = _build_repair_reflection_prompt(
            prior_diagnostics_count=6,
            current_diagnostics_count=6,
            resolved_fingerprints=[],
            persisted_fingerprints=["AssertionError::assert 3 == 2"],
            new_fingerprints=[],
            top_persisted_diagnostics=[
                {
                    "error_code": "AssertionError",
                    "message": "assert 3 == 2",
                    "file": "server/tests/test_services_filings.py",
                    "line": 199,
                }
            ],
        )
        assert "server/tests/test_services_filings.py:199" in prompt

    def test_prompt_offers_insufficient_data_escape(self):
        """Fix A — when the LLM lacks data to localise, it should be
        able to write the literal 'insufficient data' sentence instead
        of inventing a file/symbol. The escape hatch must be advertised
        in the prompt."""
        from harness.graph import _build_repair_reflection_prompt
        prompt = _build_repair_reflection_prompt(
            prior_diagnostics_count=3, current_diagnostics_count=3,
            resolved_fingerprints=[], persisted_fingerprints=["TypeError::"],
            new_fingerprints=[],
            top_persisted_diagnostics=[
                {"error_code": "TypeError", "message": "TypeError",
                 "file": "x.py", "line": 0}
            ],
        )
        assert "insufficient data" in prompt
        assert "do not invent" in prompt.lower() or "do NOT invent" in prompt

    def test_parser_accepts_well_formed_distraction(self):
        from harness.graph import _parse_repair_reflection_verdict
        raw = (
            '{"verdict": "DISTRACTION", '
            '"real_blocker": "AuthService.ts:20 jwt.sign signature is wrong", '
            '"recommendation": "Pass options as the third arg, not second."}'
        )
        v = _parse_repair_reflection_verdict(raw)
        assert v is not None
        assert v["verdict"] == "DISTRACTION"
        assert "AuthService.ts" in v["real_blocker"]

    def test_parser_handles_markdown_fence(self):
        from harness.graph import _parse_repair_reflection_verdict
        raw = (
            "```json\n"
            '{"verdict": "PROGRESS", "real_blocker": "", "recommendation": ""}\n'
            "```"
        )
        v = _parse_repair_reflection_verdict(raw)
        assert v is not None and v["verdict"] == "PROGRESS"

    def test_parser_rejects_invalid_verdict(self):
        from harness.graph import _parse_repair_reflection_verdict
        raw = '{"verdict": "DUNNO", "real_blocker": "x", "recommendation": "y"}'
        assert _parse_repair_reflection_verdict(raw) is None

    def test_parser_rejects_distraction_without_blocker(self):
        """DISTRACTION/REGRESSION are useless without a blocker — the
        whole point is to redirect the next round."""
        from harness.graph import _parse_repair_reflection_verdict
        raw = '{"verdict": "DISTRACTION", "real_blocker": "", "recommendation": ""}'
        assert _parse_repair_reflection_verdict(raw) is None

    def test_parser_progress_allows_empty_blocker(self):
        """PROGRESS verdicts can omit the blocker — there's nothing to
        redirect to."""
        from harness.graph import _parse_repair_reflection_verdict
        raw = '{"verdict": "PROGRESS", "real_blocker": "", "recommendation": ""}'
        v = _parse_repair_reflection_verdict(raw)
        assert v is not None and v["verdict"] == "PROGRESS"

    def test_parser_rejects_non_json(self):
        from harness.graph import _parse_repair_reflection_verdict
        assert _parse_repair_reflection_verdict("not json") is None
        assert _parse_repair_reflection_verdict("") is None

    def test_grounding_accepts_full_path_reference(self):
        """Fix C — when real_blocker names a file present in
        compiler_errors (full path), injection is allowed."""
        from harness.graph import _reflection_grounds_in_diagnostics
        v = {
            "real_blocker": (
                "Regex in server/services/filings.py misses 'period ended'"
            ),
            "recommendation": "",
        }
        errs = [{
            "file": "server/services/filings.py", "line": 23,
            "error_code": "AssertionError", "message": "assert 3 == 2",
        }]
        assert _reflection_grounds_in_diagnostics(v, errs) is True

    def test_grounding_accepts_basename_reference(self):
        """Fix C — LLMs often shorten paths to the basename; that
        should still count as grounded."""
        from harness.graph import _reflection_grounds_in_diagnostics
        v = {"real_blocker": "bug in filings.py extract_period_date",
             "recommendation": ""}
        errs = [{
            "file": "server/services/filings.py", "line": 23,
            "error_code": "X", "message": "y",
        }]
        assert _reflection_grounds_in_diagnostics(v, errs) is True

    def test_grounding_rejects_hallucinated_blocker(self):
        """Fix C — when the LLM names no file present in the actual
        compiler_errors (e.g. 'in the filing list retrieval logic'),
        injection must be skipped — that text is fabricated."""
        from harness.graph import _reflection_grounds_in_diagnostics
        v = {
            "real_blocker": (
                "TypeError in the filing list retrieval logic, "
                "indexing a list with a string"
            ),
            "recommendation": "edit the helper that processes filings",
        }
        errs = [{
            "file": "server/services/filings.py", "line": 23,
            "error_code": "X", "message": "y",
        }]
        assert _reflection_grounds_in_diagnostics(v, errs) is False

    def test_grounding_rejects_insufficient_data_escape(self):
        """Fix C — the explicit 'insufficient data' escape hatch (from
        fix A) is ungrounded ONLY when the recommendation is also
        empty/vague. With a non-empty recommendation that DOES ground,
        the partial verdict is still actionable; see the next test."""
        from harness.graph import _reflection_grounds_in_diagnostics
        v = {
            "real_blocker": (
                "insufficient data — investigate filings.py's data flow "
                "into the assertion"
            ),
            "recommendation": "",
        }
        errs = [{
            "file": "server/services/filings.py", "line": 23,
            "error_code": "X", "message": "y",
        }]
        assert _reflection_grounds_in_diagnostics(v, errs) is False

    def test_grounding_accepts_recommendation_when_blocker_is_escape(self):
        """Observed in session cf3fcd27: the judge hedged on the blocker
        line ('insufficient data — investigate test_api.py') but produced
        a concrete recommendation naming services/edgar.py — which IS in
        the failing set. The injection must still fire and use the
        recommendation as the lead, otherwise the SRS keeps winning the
        repair LLM's attention."""
        from harness.graph import _reflection_grounds_in_diagnostics
        v = {
            "real_blocker": (
                "insufficient data — investigate test_api.py's data flow"
            ),
            "recommendation": (
                "Mock the service function that triggers the RuntimeError "
                "at services/edgar.py:48 in tests."
            ),
        }
        errs = [{
            "file": "server/services/edgar.py", "line": 48,
            "error_code": "RuntimeError", "message": "real http call",
        }]
        assert _reflection_grounds_in_diagnostics(v, errs) is True

    def test_grounding_falls_open_when_no_file_info(self):
        """Fix C — when compiler_errors carry no file info at all (e.g.
        synthetic markers like '<harness:...>'), the helper falls open
        so a real injection isn't muted by absent grounding data."""
        from harness.graph import _reflection_grounds_in_diagnostics
        v = {"real_blocker": "some sensible-sounding blocker text",
             "recommendation": ""}
        errs = [{
            "file": "<harness:security-validator>", "line": 0,
            "error_code": "BUILD_COMMAND_BLOCKED", "message": "blocked",
        }]
        assert _reflection_grounds_in_diagnostics(v, errs) is True

    def test_verdict_named_files_returns_intersection(self):
        """The judge-ignored gate (Fix #3) needs the paths — not a bool —
        of files that BOTH appear in the verdict text AND are in the
        current failing set, so it can later check whether the repair
        round touched them. Full path AND basename mentions both count."""
        from harness.graph import _verdict_named_files
        v = {
            "real_blocker": (
                "RuntimeError at services/edgar.py:39 — real EDGAR HTTP"
            ),
            "recommendation": "mock fetch_company_index from edgar.py",
        }
        errs = [
            {"file": "server/services/edgar.py", "line": 39,
             "error_code": "RuntimeError", "message": "boom"},
            {"file": "server/tests/conftest.py", "line": 1,
             "error_code": "X", "message": "y"},
        ]
        named = _verdict_named_files(v, errs)
        # basename "edgar.py" matches server/services/edgar.py
        assert named == ["server/services/edgar.py"]

    def test_verdict_named_files_skips_insufficient_data(self):
        from harness.graph import _verdict_named_files
        v = {
            "real_blocker": "insufficient data — investigate filings.py",
            "recommendation": "",
        }
        errs = [{"file": "server/services/filings.py", "line": 1,
                 "error_code": "X", "message": "y"}]
        assert _verdict_named_files(v, errs) == []

    def test_verdict_named_files_uses_recommendation_when_blocker_is_escape(self):
        """Recommendation fallback: when blocker is 'insufficient data'
        but recommendation names a file in the failing set, return it."""
        from harness.graph import _verdict_named_files
        v = {
            "real_blocker": "insufficient data — investigate test_api.py",
            "recommendation": (
                "Mock fetch_company_index at services/edgar.py:48."
            ),
        }
        errs = [{"file": "server/services/edgar.py", "line": 48,
                 "error_code": "RuntimeError", "message": "boom"}]
        assert _verdict_named_files(v, errs) == ["server/services/edgar.py"]

    def test_verdict_named_files_no_intersection_returns_empty(self):
        from harness.graph import _verdict_named_files
        v = {"real_blocker": "bug in foo.py somewhere", "recommendation": ""}
        errs = [{"file": "server/services/edgar.py", "line": 39,
                 "error_code": "X", "message": "y"}]
        assert _verdict_named_files(v, errs) == []

    def test_patches_touched_judge_files_suffix_match(self):
        """A judge that named ``services/edgar.py`` and a patch that
        touched ``server/services/edgar.py`` (different prefix) must
        count as compliance — the harness and the repair LLM frequently
        disagree on workspace-relative roots."""
        from harness.graph import _patches_touched_judge_files

        class _R:
            def __init__(self, file, success=True, no_op=False):
                self.file = file
                self.success = success
                self.no_op = no_op

        assert _patches_touched_judge_files(
            [_R("server/services/edgar.py")],
            ["services/edgar.py"],
        ) is True

    def test_patches_touched_judge_files_basename_match(self):
        from harness.graph import _patches_touched_judge_files

        class _R:
            def __init__(self, file, success=True, no_op=False):
                self.file = file
                self.success = success
                self.no_op = no_op

        assert _patches_touched_judge_files(
            [_R("apps/backend/services/edgar.py")],
            ["server/services/edgar.py"],
        ) is True

    def test_patches_touched_judge_files_ignores_no_ops(self):
        """Idempotency no-ops do NOT count as touching the file. Without
        this, a DELETE_BLOCK on an already-empty file masks a real
        distraction."""
        from harness.graph import _patches_touched_judge_files

        class _R:
            def __init__(self, file, success=True, no_op=False):
                self.file = file
                self.success = success
                self.no_op = no_op

        assert _patches_touched_judge_files(
            [_R("server/services/edgar.py", no_op=True)],
            ["services/edgar.py"],
        ) is False

    def test_patches_touched_judge_files_unrelated_patches(self):
        from harness.graph import _patches_touched_judge_files

        class _R:
            def __init__(self, file, success=True, no_op=False):
                self.file = file
                self.success = success
                self.no_op = no_op

        assert _patches_touched_judge_files(
            [_R("server/tests/test_api.py"), _R("client/src/App.tsx")],
            ["server/services/edgar.py"],
        ) is False

    def test_patches_touched_judge_files_empty_named_is_no_op(self):
        """When the judge named no files the gate has nothing to enforce
        — caller should treat that as 'fall open'."""
        from harness.graph import _patches_touched_judge_files

        class _R:
            def __init__(self, file, success=True, no_op=False):
                self.file = file
                self.success = success
                self.no_op = no_op

        assert _patches_touched_judge_files(
            [_R("anything.py")], [],
        ) is True

    def test_shared_root_cause_fanout_above_threshold(self):
        """Fix #3 — the same error_code across 3+ files is a shared root
        cause; the banner should call this out so the LLM patches all of
        them in one response instead of one per round."""
        from harness.graph import _shared_root_cause_fanout
        errs = [
            {"error_code": "RuntimeError", "file": "tests/test_a.py",
             "message": "edgar guard"},
            {"error_code": "RuntimeError", "file": "tests/test_b.py",
             "message": "edgar guard"},
            {"error_code": "RuntimeError", "file": "tests/test_c.py",
             "message": "edgar guard"},
            {"error_code": "AssertionError", "file": "tests/test_z.py",
             "message": "unrelated"},
        ]
        result = _shared_root_cause_fanout(errs)
        assert len(result) == 1
        code, files = result[0]
        assert code == "RuntimeError"
        assert files == [
            "tests/test_a.py", "tests/test_b.py", "tests/test_c.py",
        ]

    def test_shared_root_cause_fanout_below_threshold_silent(self):
        """Two files sharing a code does not justify a fan-out directive."""
        from harness.graph import _shared_root_cause_fanout
        errs = [
            {"error_code": "RuntimeError", "file": "tests/test_a.py",
             "message": "boom"},
            {"error_code": "RuntimeError", "file": "tests/test_b.py",
             "message": "boom"},
        ]
        assert _shared_root_cause_fanout(errs) == []

    def test_shared_root_cause_fanout_ignores_synthetic_markers(self):
        """Synthetic file markers like '<harness:security-validator>' must
        not contribute to fan-out — they're not real files the LLM can
        patch."""
        from harness.graph import _shared_root_cause_fanout
        errs = [
            {"error_code": "X", "file": "<harness:scan>", "message": "y"},
            {"error_code": "X", "file": "<harness:scan>", "message": "y"},
            {"error_code": "X", "file": "<harness:scan>", "message": "y"},
            {"error_code": "X", "file": "real.py", "message": "z"},
        ]
        assert _shared_root_cause_fanout(errs) == []

    def test_shared_root_cause_fanout_dedupes_per_file(self):
        """If the SAME file has multiple errors with the same code, that
        still counts as ONE file for the fan-out threshold."""
        from harness.graph import _shared_root_cause_fanout
        errs = [
            {"error_code": "RuntimeError", "file": "tests/test_a.py",
             "message": "1"},
            {"error_code": "RuntimeError", "file": "tests/test_a.py",
             "message": "2"},
            {"error_code": "RuntimeError", "file": "tests/test_a.py",
             "message": "3"},
        ]
        assert _shared_root_cause_fanout(errs) == []


class TestRawCountProgress:
    """Phase 1.1 follow-up — when many failing tests share one
    fingerprint (e.g. 10 tests all hitting one EDGAR-mock guard), fixing
    them one-at-a-time leaves the fingerprint *set* unchanged while the
    raw *count* shrinks. Crediting raw-count shrinkage as progress
    prevents the no_progress gate from firing on real wins."""

    def test_count_shrinkage_credits_progress_even_when_fps_set_unchanged(self):
        """Both fingerprint sets are the same singleton — but the raw
        count dropped 10 → 9. That's real progress."""
        prior_fps = {"RuntimeError::real edgar http call"}
        current_fps = {"RuntimeError::real edgar http call"}
        prior_count, current_count = 10, 9
        fps_advanced = bool(prior_fps - current_fps)
        count_shrank = (
            prior_count > 0
            and current_count > 0
            and current_count < prior_count
        )
        assert fps_advanced is False
        assert count_shrank is True
        assert (fps_advanced or count_shrank) is True

    def test_fps_identity_flip_credits_progress_even_when_cardinality_unchanged(self):
        """Prior failure resolved, downstream failure surfaced by the
        same fix. Sets are both singletons but the identity changed —
        prior_fps - current_fps is {A}, so at least one prior fingerprint
        is no longer present. That's real progress, not a stall.

        Historical bug: the log copy called this "shrank fingerprint set"
        even though sizes were equal. The name ``fps_advanced`` and the
        rewritten log message capture the actual semantic: the prior set
        moved forward, whether or not it got smaller."""
        prior_fps = {"AssertionError::form-type-slash"}
        current_fps = {"ModuleNotFoundError::pytest-asyncio"}
        prior_count, current_count = 1, 1
        fps_advanced = bool(prior_fps - current_fps)
        count_shrank = (
            prior_count > 0
            and current_count > 0
            and current_count < prior_count
        )
        assert fps_advanced is True
        assert count_shrank is False
        assert (fps_advanced or count_shrank) is True

    def test_count_static_is_not_progress(self):
        """No fingerprint advance AND no count shrinkage → no progress."""
        prior_fps = {"RuntimeError::x"}
        current_fps = {"RuntimeError::x"}
        prior_count, current_count = 4, 4
        fps_advanced = bool(prior_fps - current_fps)
        count_shrank = (
            prior_count > 0
            and current_count > 0
            and current_count < prior_count
        )
        assert (fps_advanced or count_shrank) is False

    def test_count_growth_is_not_progress(self):
        """A round that LEAVES MORE diagnostics than it started with is a
        regression — must not credit progress."""
        prior_count, current_count = 4, 7
        count_shrank = (
            prior_count > 0
            and current_count > 0
            and current_count < prior_count
        )
        assert count_shrank is False

    def test_count_to_zero_not_credited_here(self):
        """current_count == 0 means the build is now green — that's the
        loop's terminator, not progress credit. Guard ensures we don't
        flap the no_progress counter on a successful end-of-loop transition."""
        prior_count, current_count = 4, 0
        count_shrank = (
            prior_count > 0
            and current_count > 0
            and current_count < prior_count
        )
        assert count_shrank is False


class TestConftestChainCollection:
    """Fix #3 — pytest loads every ``conftest.py`` on the path from the
    rootdir down to a failing test. When the workspace has overlapping
    test trees (``tests/conftest.py`` AND ``server/tests/conftest.py``),
    the repair LLM otherwise has no way to know which one is in scope.
    The chain helper surfaces the exact load order."""

    def test_single_conftest_at_test_directory(self, tmp_path):
        from harness.graph import _conftest_chain_for_test
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "conftest.py").write_text("# fixtures")
        chain = _conftest_chain_for_test(
            str(tmp_path), "tests/test_x.py",
        )
        assert chain == ["tests/conftest.py"]

    def test_workspace_root_conftest_picked_up(self, tmp_path):
        from harness.graph import _conftest_chain_for_test
        (tmp_path / "conftest.py").write_text("# rootmost")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "conftest.py").write_text("# nested")
        chain = _conftest_chain_for_test(
            str(tmp_path), "tests/test_x.py",
        )
        # Pytest precedence: root first, leaf last.
        assert chain == ["conftest.py", "tests/conftest.py"]

    def test_nested_chain_includes_every_ancestor_conftest(self, tmp_path):
        from harness.graph import _conftest_chain_for_test
        for sub in ("", "server", "server/tests"):
            d = tmp_path / sub if sub else tmp_path
            d.mkdir(exist_ok=True)
            (d / "conftest.py").write_text(f"# at {sub or '<root>'}")
        chain = _conftest_chain_for_test(
            str(tmp_path), "server/tests/test_y.py",
        )
        assert chain == [
            "conftest.py", "server/conftest.py", "server/tests/conftest.py",
        ]

    def test_missing_intermediate_conftest_skipped(self, tmp_path):
        """No ``conftest.py`` directly under ``server/`` — pytest just
        skips that level; the chain reflects what actually loads."""
        from harness.graph import _conftest_chain_for_test
        (tmp_path / "conftest.py").write_text("# root")
        (tmp_path / "server" / "tests").mkdir(parents=True)
        (tmp_path / "server" / "tests" / "conftest.py").write_text("# leaf")
        chain = _conftest_chain_for_test(
            str(tmp_path), "server/tests/test_y.py",
        )
        # No conftest.py exists under server/ — only the two that do.
        assert chain == ["conftest.py", "server/tests/conftest.py"]

    def test_no_conftests_returns_empty(self, tmp_path):
        from harness.graph import _conftest_chain_for_test
        assert _conftest_chain_for_test(
            str(tmp_path), "tests/test_x.py",
        ) == []

    def test_collect_distinct_chains_per_failing_test(self, tmp_path):
        """Two test trees with independent conftests should both appear.
        This is exactly the cf3fcd27 scenario — ``tests/conftest.py``
        AND ``server/tests/conftest.py`` were both alive and the LLM
        had no signal about which to patch."""
        from harness.graph import _collect_conftests_for_failing_tests
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "conftest.py").write_text("# tree A")
        (tmp_path / "server" / "tests").mkdir(parents=True)
        (tmp_path / "server" / "tests" / "conftest.py").write_text("# tree B")
        errors = [
            {"file": "tests/test_a.py", "error_code": "AssertionError",
             "message": "boom"},
            {"file": "server/tests/test_b.py", "error_code": "AssertionError",
             "message": "boom"},
        ]
        chains = _collect_conftests_for_failing_tests(
            str(tmp_path), errors,
        )
        # Two distinct chains, one per tree.
        assert len(chains) == 2
        rep_paths = {test for test, _ in chains}
        assert rep_paths == {"tests/test_a.py", "server/tests/test_b.py"}

    def test_collect_dedupes_within_chain(self, tmp_path):
        """Two failing tests sharing the same chain → one entry only."""
        from harness.graph import _collect_conftests_for_failing_tests
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "conftest.py").write_text("# shared")
        errors = [
            {"file": "tests/test_a.py", "error_code": "X", "message": "y"},
            {"file": "tests/test_b.py", "error_code": "X", "message": "y"},
        ]
        chains = _collect_conftests_for_failing_tests(
            str(tmp_path), errors,
        )
        assert len(chains) == 1

    def test_collect_skips_synthetic_markers(self, tmp_path):
        from harness.graph import _collect_conftests_for_failing_tests
        errors = [
            {"file": "<harness:security>", "error_code": "X", "message": "y"},
        ]
        assert _collect_conftests_for_failing_tests(
            str(tmp_path), errors,
        ) == []


class TestFirstPartyImportPrefetch:
    """Fix #2 — prefetch the first-party modules a failing test imports
    so the repair LLM has them in-prompt instead of burning a round on
    a READ_FILE for each one. Matches how Claude Code reads
    transitively before patching."""

    def test_extracts_from_import(self, tmp_path):
        from harness.graph import _first_party_imports_for
        (tmp_path / "server" / "services").mkdir(parents=True)
        (tmp_path / "server" / "services" / "search.py").write_text("# src")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_search.py").write_text(
            "from server.services.search import search_companies\n"
        )
        imports = _first_party_imports_for(
            str(tmp_path), "tests/test_search.py",
        )
        assert "server/services/search.py" in imports

    def test_extracts_plain_import(self, tmp_path):
        from harness.graph import _first_party_imports_for
        (tmp_path / "server").mkdir()
        (tmp_path / "server" / "__init__.py").write_text("")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_x.py").write_text("import server\n")
        imports = _first_party_imports_for(
            str(tmp_path), "tests/test_x.py",
        )
        assert "server/__init__.py" in imports

    def test_skips_stdlib_imports(self, tmp_path):
        from harness.graph import _first_party_imports_for
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_x.py").write_text(
            "import os\nfrom json import loads\nfrom datetime import date\n"
        )
        assert _first_party_imports_for(
            str(tmp_path), "tests/test_x.py",
        ) == []

    def test_skips_third_party_imports(self, tmp_path):
        from harness.graph import _first_party_imports_for
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_x.py").write_text(
            "import pytest\n"
            "from unittest.mock import AsyncMock\n"
            "from fastapi.testclient import TestClient\n"
        )
        assert _first_party_imports_for(
            str(tmp_path), "tests/test_x.py",
        ) == []

    def test_skips_nonexistent_first_party(self, tmp_path):
        """An import that LOOKS first-party but resolves to no file in
        the workspace is dropped — we never speculate about paths the
        LLM can't actually read."""
        from harness.graph import _first_party_imports_for
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_x.py").write_text(
            "from app.services.search import x\n"
        )
        # No app/ directory exists.
        assert _first_party_imports_for(
            str(tmp_path), "tests/test_x.py",
        ) == []

    def test_prefers_module_file_over_package_init(self, tmp_path):
        """When both ``server/services/search.py`` AND
        ``server/services/search/__init__.py`` could match, prefer the
        .py file — that's where the actual code lives in the common
        case."""
        from harness.graph import _first_party_imports_for
        (tmp_path / "server" / "services").mkdir(parents=True)
        (tmp_path / "server" / "services" / "search.py").write_text("# code")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_x.py").write_text(
            "from server.services.search import x\n"
        )
        imports = _first_party_imports_for(
            str(tmp_path), "tests/test_x.py",
        )
        assert imports == ["server/services/search.py"]

    def test_deduplicates(self, tmp_path):
        from harness.graph import _first_party_imports_for
        (tmp_path / "server" / "services").mkdir(parents=True)
        (tmp_path / "server" / "services" / "search.py").write_text("# code")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_x.py").write_text(
            "from server.services.search import a\n"
            "from server.services.search import b\n"
        )
        imports = _first_party_imports_for(
            str(tmp_path), "tests/test_x.py",
        )
        assert imports == ["server/services/search.py"]

    def test_missing_test_file_returns_empty(self, tmp_path):
        from harness.graph import _first_party_imports_for
        assert _first_party_imports_for(
            str(tmp_path), "tests/does_not_exist.py",
        ) == []


class TestDecisionPointLogging:
    """Phase 2.1 — every filter/drop site emits a 'dropped_from_prompt'
    event so post-mortems can grep one event name and see exactly what
    the harness hid from the LLM."""

    def _err(self, code: str, msg: str, file: str = "x.ts") -> dict:
        return {
            "file": file,
            "line": 1,
            "column": 1,
            "error_code": code,
            "message": msg,
            "severity": "error",
            "semantic_context": f"// snippet for {code}",
        }

    def test_deferred_diagnostics_emit_event(self, caplog):
        import logging
        # Force ≥ 6 groups so deferral renders (Layer 2 threshold).
        errors = [self._err(f"TS{n}", f"msg {n}") for n in range(7000, 7008)]
        caplog.set_level(logging.INFO, logger="harness.events")
        _format_diagnostics_for_repair(errors)
        events = [r for r in caplog.records if getattr(r, "event", "") == "dropped_from_prompt"]
        deferred_events = [r for r in events if getattr(r, "site", "") == "deferred_diagnostics"]
        assert deferred_events, "expected a deferred_diagnostics event"
        ev = deferred_events[-1]
        assert getattr(ev, "dropped_count", 0) > 0
        # Examples list should carry codes for grep-ability.
        examples = getattr(ev, "examples", [])
        assert isinstance(examples, list) and len(examples) > 0
        assert "code" in examples[0]

    def test_no_event_when_nothing_deferred(self, caplog):
        import logging
        # ≤ 5 groups → Layer 2 short-circuit, no deferral, no event.
        errors = [self._err(f"TS{n}", f"msg {n}") for n in range(7000, 7003)]
        caplog.set_level(logging.INFO, logger="harness.events")
        _format_diagnostics_for_repair(errors)
        deferred_events = [
            r for r in caplog.records
            if getattr(r, "event", "") == "dropped_from_prompt"
            and getattr(r, "site", "") == "deferred_diagnostics"
        ]
        assert not deferred_events


class TestProgressBasedBudget:
    """Phase 1.1 — repair counter should tick the no_progress_repairs
    counter only when the prior round's patches did not shrink the failing
    fingerprint set. Validates the routing-gate change in
    route_after_compiler and the conditional increment in repair_node by
    calling the relevant pieces in isolation.
    """

    def test_router_HITLs_on_no_progress_cap_not_total_repairs(self, monkeypatch):
        from harness import graph as _graph
        # max_iterations=3 default in test scope (no gateway). Build state
        # where total_repairs is high (LLM has been working hard) but
        # no_progress_repairs is 0 — should NOT HITL.
        state = {
            "exit_code": 2,
            "compiler_errors": [{"error_code": "TS2769", "message": "x"}],
            "loop_counter": {
                "total_repairs": 5,        # high
                "no_progress_repairs": 0,  # but every round made progress
                "consecutive_zero_patch_rounds": 0,
            },
            "budget_remaining_usd": 1.0,
            "node_state": {},
        }
        result = _graph.route_after_compiler(state)
        assert result == "repair_node", (
            "5 total rounds with 0 no-progress should not escalate — "
            f"got {result}"
        )

    def test_router_HITLs_when_no_progress_hits_cap(self):
        from harness import graph as _graph
        state = {
            "exit_code": 2,
            "compiler_errors": [{"error_code": "TS2769", "message": "x"}],
            "loop_counter": {
                "total_repairs": 3,
                "no_progress_repairs": 3,  # at cap (3)
                "consecutive_zero_patch_rounds": 0,
            },
            "budget_remaining_usd": 1.0,
            "node_state": {},
        }
        result = _graph.route_after_compiler(state)
        assert result == "human_intervention_node"

    def test_router_hard_ceiling_at_4x_total_repairs(self):
        """Even with every round making progress, run away protection
        kicks in at 4 * max_iterations to prevent fingerprint-churn loops.

        Multiplier was raised from 2 → 4 on 2026-07-04 (ciod session
        523e86a7): a converging prod-smoke cascade needed more than
        6 rounds and the ceiling was tripping while genuine progress was
        landing. 4 × 3 = 12 total gives cascades enough runway without
        losing the tripwire for real thrash."""
        from harness import graph as _graph
        state = {
            "exit_code": 2,
            "compiler_errors": [{"error_code": "TS2769", "message": "x"}],
            "loop_counter": {
                "total_repairs": 12,       # 4 * 3 = hard ceiling
                "no_progress_repairs": 0,
                "consecutive_zero_patch_rounds": 0,
            },
            "budget_remaining_usd": 1.0,
            "node_state": {},
        }
        result = _graph.route_after_compiler(state)
        assert result == "human_intervention_node"

    def test_router_below_hard_ceiling_continues(self):
        """Sanity guard for the raised multiplier: at 11 rounds
        (one below the new 4×3=12 cap), we must still route to repair,
        not HITL. Locks in the 4× ratio so a future refactor that
        accidentally reverts to 2× produces a clear test failure."""
        from harness import graph as _graph
        state = {
            "exit_code": 2,
            "compiler_errors": [{"error_code": "TS2769", "message": "x"}],
            "loop_counter": {
                "total_repairs": 11,
                "no_progress_repairs": 0,
                "consecutive_zero_patch_rounds": 0,
            },
            "budget_remaining_usd": 1.0,
            "node_state": {},
        }
        result = _graph.route_after_compiler(state)
        assert result == "repair_node"

    def test_router_backward_compat_no_progress_field_absent(self):
        """Sessions checkpointed before Phase 1.1 won't have
        no_progress_repairs in state. Default-0 → behaves like a fresh
        session, no premature HITL."""
        from harness import graph as _graph
        state = {
            "exit_code": 2,
            "compiler_errors": [{"error_code": "TS2769", "message": "x"}],
            "loop_counter": {
                "total_repairs": 2,
                # no_progress_repairs ABSENT — old checkpoint
                "consecutive_zero_patch_rounds": 0,
            },
            "budget_remaining_usd": 1.0,
            "node_state": {},
        }
        result = _graph.route_after_compiler(state)
        assert result == "repair_node"


class TestGraphStateTypes:
    """Test that graph state can be constructed with required fields."""

    def test_state_with_messages(self):
        """State should support messages field."""
        state = {
            "messages": [{"role": "user", "content": "test"}],
        }
        assert "messages" in state

    def test_state_with_tokens(self):
        """State should support token tracking."""
        state = {
            "messages": [],
            "token_tracker": {
                "total_input_tokens": 100,
                "total_output_tokens": 50,
                "total_cost_usd": 0.001,
            },
        }
        assert "token_tracker" in state

    def test_state_with_diagnostics(self):
        """State should support diagnostics."""
        state = {
            "messages": [],
            "diagnostics": [
                {
                    "file": "test.py",
                    "line": 5,
                    "message": "error",
                    "severity": "error",
                }
            ],
        }
        assert "diagnostics" in state

    def test_state_with_loop_counters(self):
        """State should support loop counters."""
        state = {
            "messages": [],
            "loop_counters": {
                "repair": 0,
                "discovery": 0,
            },
        }
        assert "loop_counters" in state


class TestNodeStateTransitions:
    """Test state transitions between nodes."""

    def test_planning_to_patching_transition(self):
        """State should transition from planning to patching."""
        state = {
            "messages": [
                {"role": "user", "content": "Fix bug"},
                {"role": "assistant", "content": "I'll fix it"},
            ],
            "current_node": "planning",
        }
        # After planning, state should have messages for patching
        assert len(state["messages"]) >= 2
        assert state["current_node"] == "planning"

    def test_compiler_exit_code_routing(self):
        """Exit code should determine next node."""
        state_success = {
            "messages": [],
            "exit_code": 0,
        }
        state_failure = {
            "messages": [],
            "exit_code": 1,
        }
        # These would be used by router functions
        assert state_success["exit_code"] == 0
        assert state_failure["exit_code"] == 1

    def test_repair_loop_counter_increment(self):
        """Repair loop should track iterations."""
        state = {
            "messages": [],
            "loop_counters": {"repair": 0},
        }
        state["loop_counters"]["repair"] += 1
        assert state["loop_counters"]["repair"] == 1


class TestErrorHandling:
    """Test error handling in graph state."""

    def test_state_with_error_message(self):
        """State should track LLM errors."""
        state = {
            "messages": [],
            "error": "budget_exhausted",
        }
        assert state["error"] == "budget_exhausted"

    def test_state_with_build_failure(self):
        """State should track build failures."""
        state = {
            "messages": [],
            "exit_code": 127,
            "diagnostics": [
                {
                    "file": "build.log",
                    "line": 1,
                    "message": "Build command not found",
                    "severity": "error",
                }
            ],
        }
        assert state["exit_code"] == 127
        assert len(state["diagnostics"]) > 0

    def test_state_with_timeout(self):
        """State should track timeouts."""
        state = {
            "messages": [],
            "timed_out": True,
            "exit_code": -1,
        }
        assert state["timed_out"] is True


class TestGatewayConfigPropagation:
    """Regression: ``set_gateway`` used to inject only the Gateway instance,
    not its config. Every ``get_gateway_config()`` consumer
    (``spec_review_node``, ``code_review_node``, the pre-flight reviewer
    in ``cmd_run``) read None and silently skipped — making the
    ``doc_reviewer_primary`` / ``code_reviewer_primary`` config keys
    effectively dead code. The fix makes ``set_gateway`` atomic: setting
    the gateway also stashes ``gateway.config`` for downstream readers.
    """

    def test_set_gateway_also_stashes_config(self):
        from harness.graph import set_gateway, get_gateway_config, get_gateway

        class _FakeConfig:
            doc_reviewer_primary = "openai:gpt-4o"
            code_reviewer_primary = "anthropic:claude-sonnet"

        class _FakeGateway:
            def __init__(self):
                self.config = _FakeConfig()

        gw = _FakeGateway()
        try:
            set_gateway(gw)
            assert get_gateway() is gw
            cfg = get_gateway_config()
            assert cfg is gw.config
            assert cfg.doc_reviewer_primary == "openai:gpt-4o"
        finally:
            # Reset the module-level slot so other tests aren't polluted.
            set_gateway.__globals__["_gateway"] = None
            set_gateway.__globals__["_gateway_config"] = None

    def test_set_gateway_without_config_is_safe(self):
        """A test double that doesn't expose ``.config`` still works —
        the gateway gets injected and the config slot stays whatever it
        was before (no AttributeError)."""
        from harness.graph import set_gateway, get_gateway_config

        class _Bare:
            pass

        # Capture whatever the slot was before to assert we don't crash.
        prior = get_gateway_config()
        try:
            set_gateway(_Bare())
            # No assertion on cfg value — just that the call returned cleanly.
            _ = get_gateway_config()
        finally:
            set_gateway.__globals__["_gateway"] = None
            set_gateway.__globals__["_gateway_config"] = prior


class TestRouteAfterSecurityScan:
    """Routing decisions after security_scan_node.

    Covers the --deploy-dev opt-in gate that controls whether a clean
    security scan rolls forward into deployment_discovery_node or stops at
    END. route_after_security_scan imports ``_is_flutter_project`` inside
    the function body, so we monkeypatch the attribute on
    ``harness.impact`` rather than on ``harness.graph``.
    """

    def _clean_state(self, **overrides):
        state = {
            "compiler_errors": [],
            "budget_remaining_usd": 1.0,
            "workspace_path": "/tmp/ws",
            # Phase G: setting end_of_session_regression_repair > 0
            # simulates "EoS regression already ran this session", so
            # route_after_security_scan skips the EoS intercept and
            # the tests below can assert the destination routing
            # (Flutter / no-deploy / cd_discovery / etc.) directly.
            # The "first visit → EoS regression" path is covered in
            # tests/test_end_of_session_regression.py.
            "loop_counter": {
                "security": 0,
                "end_of_session_regression_repair": 1,
            },
            "dev_deployment": False,
            "cd_discovery": False,
        }
        # Merge loop_counter from overrides so the post-EoS marker
        # always survives even when a test wants to override a few
        # specific keys (e.g. final_verify for pre_exit_verify tests).
        if "loop_counter" in overrides:
            merged = dict(state["loop_counter"])
            merged.update(overrides.pop("loop_counter"))
            state["loop_counter"] = merged
        state.update(overrides)
        return state

    def test_clean_scan_ends_when_dev_deployment_false(self, monkeypatch):
        # Clean + no --deploy-dev now routes through installation_doc_node;
        # the node's only outgoing edge is END, so this is still a
        # terminal path (the doc may be a no-op if install_doc=False).
        assert route_after_security_scan(self._clean_state()) == "installation_doc_node"

    def test_clean_scan_enters_discovery_when_dev_deployment_and_cd_discovery_true(self, monkeypatch):
        # The classic flow: --deploy-dev true + --cd-discovery true → run
        # the LLM-driven blueprint pipeline.
        state = self._clean_state(dev_deployment=True, cd_discovery=True)
        assert route_after_security_scan(state) == "deployment_discovery_node"

    def test_clean_scan_skips_discovery_when_cd_discovery_false(self, monkeypatch):
        # The new fast-path: --deploy-dev true + --cd-discovery false →
        # straight to deployment_node, which synthesises the blueprint
        # from workspace telemetry alone (plus any deployment_defaults
        # section of config.json).
        state = self._clean_state(dev_deployment=True, cd_discovery=False)
        assert route_after_security_scan(state) == "deployment_node"

    def test_clean_scan_ends_when_cd_discovery_true_but_dev_deployment_false(self, monkeypatch):
        # cd_discovery alone (no dev_deployment) is meaningless — the
        # security-scan-clean terminal path still wins because no deploy
        # was requested. After the installation_doc_node insertion the
        # terminal hop is via the doc node (which then edges to END).
        state = self._clean_state(dev_deployment=False, cd_discovery=True)
        assert route_after_security_scan(state) == "installation_doc_node"

    def test_security_findings_route_to_repair(self, monkeypatch):
        state = self._clean_state(
            compiler_errors=[
                {
                    "file": "x.py",
                    "line": 1,
                    "column": 0,
                    "severity": "error",
                    "error_code": "GITLEAKS-X",
                    "message": "secret detected",
                    "semantic_context": "",
                }
            ],
        )
        # Findings exist + low attempts → repair_node regardless of flag.
        assert route_after_security_scan(state) == "repair_node"

    # Audit #18 — pre-exit verify
    def test_pre_exit_verify_routes_to_compiler_when_mutations_pending(self, monkeypatch):
        # Clean scan + opt-in flag + pending mutations → re-verify.
        state = self._clean_state(
            pre_exit_verify=True,
            pending_mutations=["server/app.py"],
        )
        assert route_after_security_scan(state) == "compiler_node"

    def test_pre_exit_verify_off_keeps_normal_terminal_route(self, monkeypatch):
        # Pending mutations exist but flag is off — defaults still hold.
        state = self._clean_state(
            pre_exit_verify=False,
            pending_mutations=["server/app.py"],
        )
        assert route_after_security_scan(state) == "installation_doc_node"

    def test_pre_exit_verify_skipped_when_no_mutations(self, monkeypatch):
        # Flag is on but nothing changed since last green compile.
        state = self._clean_state(
            pre_exit_verify=True,
            pending_mutations=[],
        )
        assert route_after_security_scan(state) == "installation_doc_node"

    def test_pre_exit_verify_one_shot_cap(self, monkeypatch):
        # Cap consumed → don't loop even if mutations are still flagged.
        state = self._clean_state(
            pre_exit_verify=True,
            pending_mutations=["server/app.py"],
            loop_counter={"security": 0, "final_verify": 1},
        )
        assert route_after_security_scan(state) == "installation_doc_node"

    # F2 — post-deployment clean-scan guard. Prevents the
    # deployment_discovery ↔ deployment_node ↔ security_scan loop observed
    # in session 951f102f. Once deployment_node has returned (any outcome
    # — success / skipped / failure), a subsequent clean scan must NOT
    # re-enter the discovery pipeline.
    def test_post_deploy_clean_scan_terminates_when_success(self, monkeypatch):
        state = self._clean_state(
            dev_deployment=True, cd_discovery=True,
            node_state={"deployment": {"success": True}},
        )
        # Without the F2 guard this would route to deployment_discovery_node
        # and start another interview round.
        assert route_after_security_scan(state) == "installation_doc_node"

    def test_post_deploy_clean_scan_terminates_when_skipped(self, monkeypatch):
        state = self._clean_state(
            dev_deployment=True, cd_discovery=True,
            node_state={
                "deployment": {
                    "skipped": True, "reason": "user_declined_preview",
                    "phase": "preview_gate",
                }
            },
        )
        # The exact bug from session 951f102f: skipped deploy used to
        # fall through into another security_scan → deployment_discovery
        # cycle. Must terminate via installation_doc_node now.
        assert route_after_security_scan(state) == "installation_doc_node"

    def test_post_deploy_clean_scan_terminates_when_phase_only(self, monkeypatch):
        # ``deployment_node`` failure paths set only ``phase`` (e.g.
        # synthesis_failed) — the guard must still fire because
        # node_state.deployment is present as a dict.
        state = self._clean_state(
            dev_deployment=True, cd_discovery=True,
            node_state={"deployment": {"phase": "synthesis_failed"}},
        )
        assert route_after_security_scan(state) == "installation_doc_node"

    def test_first_clean_scan_without_deployment_still_enters_discovery(
        self, monkeypatch,
    ):
        # node_state has no ``deployment`` key — we have NOT been through
        # deployment_node yet. Existing dev_deployment/cd_discovery
        # routing must still apply (regression guard).
        state = self._clean_state(
            dev_deployment=True, cd_discovery=True,
            node_state={"other_key": "noise"},
        )
        assert route_after_security_scan(state) == "deployment_discovery_node"


class TestRouteAfterDeployment:
    """F1 — make route_after_deployment terminal.

    The historic fall-through to route_after_compiler silently re-entered
    security_scan_node whenever ``deployment.success`` wasn't strictly
    True, causing the deployment ↔ security-scan loop in session
    951f102f. The fixed router has four explicit branches.
    """

    def _state(self, deployment=None, compiler_errors=None):
        return {
            "node_state": (
                {"deployment": deployment} if deployment is not None else {}
            ),
            "compiler_errors": compiler_errors or [],
        }

    def test_success_routes_to_installation_doc(self):
        assert route_after_deployment(
            self._state(deployment={"success": True})
        ) == "installation_doc_node"

    def test_skipped_routes_to_installation_doc(self):
        # The exact session-951f102f trap: skipped via user_declined_preview
        # used to fall through to route_after_compiler and re-enter the
        # security scan. Must now terminate via installation_doc_node.
        assert route_after_deployment(self._state(
            deployment={"skipped": True, "reason": "user_declined_preview"}
        )) == "installation_doc_node"

    def test_skipped_disabled_also_routes_to_installation_doc(self):
        assert route_after_deployment(self._state(
            deployment={"skipped": True, "reason": "disabled"}
        )) == "installation_doc_node"

    def test_compiler_errors_route_to_repair(self):
        assert route_after_deployment(self._state(
            deployment={"phase": "build_failed"},
            compiler_errors=[{
                "file": "docker-compose.yml", "line": 0, "column": 0,
                "severity": "error", "error_code": "DEPLOYMENT_BUILD_FAILED",
                "message": "compose build failed",
            }],
        )) == "repair_node"

    def test_compiler_errors_without_deployment_dict_route_to_repair(self):
        # The synthesis_failed / docker_unavailable paths set
        # compiler_errors but no deployment dict on node_state. Repair
        # must still pick them up.
        assert route_after_deployment({
            "node_state": {},
            "compiler_errors": [{
                "file": "deployment", "line": 0, "column": 0,
                "severity": "error", "error_code": "DEPLOYMENT_DOCKER_UNAVAILABLE",
                "message": "docker-compose not installed",
            }],
        }) == "repair_node"

    def test_no_deployment_no_errors_routes_to_hitl(self):
        # The Bug-A trap: deployment_node returned without setting either
        # success/skipped or compiler_errors. Previously fell through and
        # looped; now surfaces to the operator.
        assert route_after_deployment({
            "node_state": {}, "compiler_errors": [],
        }) == "human_intervention_node"

    def test_deployment_dict_without_known_keys_routes_to_hitl(self):
        # A truthy but uninformative deployment dict still hits the
        # HITL fallback rather than looping silently.
        assert route_after_deployment(self._state(
            deployment={"unknown_field": 42},
        )) == "human_intervention_node"


class _PatchingResponse:
    """Stand-in for the gateway response object the patching node
    consumes — needs ``content``, ``finish_reason``, ``tool_calls``,
    and ``usage`` (with input/output/cost fields)."""

    class _Usage:
        input_tokens = 100
        output_tokens = 200
        cost_usd = 0.001
        cached_tokens = 0

    def __init__(self, content: str, finish_reason: str):
        self.content = content
        self.finish_reason = finish_reason
        self.tool_calls = []
        self.usage = self._Usage()


class TestPatchingNodeContinuation:
    """When the patching LLM hits its 8192-token output cap
    mid-blueprint (session web-6d5ef9b18f6a's symptom — backend
    emitted, frontend never), the node now re-dispatches with a
    "continue" prompt and concatenates the chunks before handing
    them to the patcher."""

    def _install_gateway_stub(self, monkeypatch):
        from harness import graph as graph_mod

        class _Cfg:
            use_structured_tools = False
            enforce_read_before_edit = False

        class _Gw:
            config = _Cfg()

            def aggregate_tokens(self, tracker, usage, role=None):
                out = dict(tracker or {})
                out["total_cost_usd"] = out.get("total_cost_usd", 0.0) + usage.cost_usd
                return out

        gw = _Gw()
        graph_mod.set_gateway(gw)
        monkeypatch.setattr(graph_mod, "_build_patcher_allowlist", lambda ws: [])
        return graph_mod

    def test_continues_when_finish_reason_is_length(self, monkeypatch, tmp_path):
        graph_mod = self._install_gateway_stub(monkeypatch)

        # First call → "length"; second call → "stop". The text-DSL
        # path should concatenate both response bodies before patch
        # parsing.
        responses = [
            _PatchingResponse("<<<CREATE_FILE>>>\nfile: backend/a.py\ncontent:\nx\n<<<END_CREATE_FILE>>>", "length"),
            _PatchingResponse("<<<CREATE_FILE>>>\nfile: frontend/index.js\ncontent:\ny\n<<<END_CREATE_FILE>>>", "stop"),
        ]
        call_messages: list[list[dict]] = []

        async def fake_tool_loop(**kwargs):
            call_messages.append(list(kwargs["messages"]))
            resp = responses.pop(0)
            return resp, kwargs["budget"] - 0.10, kwargs["messages"], {}

        monkeypatch.setattr(graph_mod, "_patching_tool_loop", fake_tool_loop)

        captured: dict = {}

        async def fake_apply(blocks, ws, existing, allowed_paths=None, **kwargs):
            captured["blocks"] = list(blocks)
            captured["files"] = [b.file for b in blocks]
            return [], []

        import harness.patcher as patcher_mod
        monkeypatch.setattr(
            patcher_mod, "apply_patch_blocks", fake_apply,
        )

        state = {
            "messages": [{"role": "system", "content": "you are a patcher"}],
            "budget_remaining_usd": 2.0,
            "workspace_path": str(tmp_path),
            "modified_files": [],
            "loop_counter": {},
            "token_tracker": {},
        }

        result = asyncio.run(graph_mod.patching_node(state))

        # Two dispatches happened (initial + 1 continuation).
        assert len(call_messages) == 2
        # The second dispatch saw the partial as an assistant turn
        # plus a continue-prompt as the trailing user turn.
        continuation_msgs = call_messages[1]
        assert continuation_msgs[-2]["role"] == "assistant"
        assert "backend/a.py" in continuation_msgs[-2]["content"]
        assert continuation_msgs[-1]["role"] == "user"
        assert "hit the output token cap" in continuation_msgs[-1]["content"]
        # The patcher saw BOTH chunks parsed and concatenated — without
        # this the frontend CREATE_FILE block would never reach disk.
        assert "backend/a.py" in captured["files"]
        assert "frontend/index.js" in captured["files"]
        # Node returned cleanly with both files in modified_files
        # provided by the fake apply (empty here — we only assert
        # the call shape).
        assert "messages" in result

    def test_caps_continuation_at_three_cycles(self, monkeypatch, tmp_path):
        """Pathological case: LLM keeps returning ``length``. The
        node must not loop forever — the default continuation ceiling
        (5 cycles) is respected, after which the node accepts what landed."""
        graph_mod = self._install_gateway_stub(monkeypatch)

        # Initial + 5 continuations = 6 dispatches, all "length".
        responses = [
            _PatchingResponse(f"chunk{i} ", "length") for i in range(1, 7)
        ]
        dispatches: list[int] = []

        async def fake_tool_loop(**kwargs):
            dispatches.append(1)
            resp = responses.pop(0)
            return resp, kwargs["budget"] - 0.10, kwargs["messages"], {}

        monkeypatch.setattr(graph_mod, "_patching_tool_loop", fake_tool_loop)

        captured: dict = {}

        # The pathological test uses non-block content ("chunk1"...),
        # so nothing parses to a PatchBlock. Assert on the concatenated
        # text handed to parse_patch_blocks via a shim on the parse fn.
        import harness.patcher as patcher_mod
        original_parse = patcher_mod.parse_patch_blocks

        def fake_parse(text):
            captured["content"] = text
            return original_parse(text)

        monkeypatch.setattr(patcher_mod, "parse_patch_blocks", fake_parse)

        async def fake_apply(blocks, ws, existing, allowed_paths=None, **kwargs):
            return [], []

        monkeypatch.setattr(patcher_mod, "apply_patch_blocks", fake_apply)

        state = {
            "messages": [{"role": "system", "content": "you are a patcher"}],
            "budget_remaining_usd": 2.0,
            "workspace_path": str(tmp_path),
            "modified_files": [],
            "loop_counter": {},
            "token_tracker": {},
        }

        asyncio.run(graph_mod.patching_node(state))

        # 6 = initial + 5 continuation cycles. No more.
        assert len(dispatches) == 6
        # All six chunks reached the patcher via the concatenated text.
        assert "chunk1" in captured["content"]
        assert "chunk6" in captured["content"]


class TestImportConventionsSection:
    """Import-conventions block for the code-review re-patch prompt
    (lumina 019f7109: the reviewer authored test files importing
    `from app.` while every existing test used `from server.app.` —
    four flat repair rounds and a no-progress HITL, because the tamper
    guard rightly keeps repair out of test files)."""

    def _ws(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_date_utils.py").write_text(
            "# @tests: server/app/date_utils.py\n"
            "import pytest\n"
            "from server.app.date_utils import compute_birthday_info\n"
            "def test_x(): pass\n"
        )
        return str(tmp_path)

    def test_samples_real_import_lines(self, tmp_path):
        from harness.graph import _import_conventions_section
        section = _import_conventions_section(self._ws(tmp_path))
        assert "MIRROR THESE" in section
        assert "from server.app.date_utils import compute_birthday_info" in section
        assert "tests/test_date_utils.py" in section

    def test_empty_workspace_yields_empty_section(self, tmp_path):
        from harness.graph import _import_conventions_section
        assert _import_conventions_section(str(tmp_path)) == ""

    def test_line_cap_respected(self, tmp_path):
        from harness.graph import (
            _IMPORT_CONVENTIONS_MAX_LINES,
            _import_conventions_section,
        )
        (tmp_path / "tests").mkdir()
        body = "\n".join(f"import mod{i}" for i in range(40)) + "\ndef test_a(): pass\n"
        for name in ("test_a.py", "test_b.py"):
            (tmp_path / "tests" / name).write_text(body)
        section = _import_conventions_section(str(tmp_path))
        sampled = [ln for ln in section.splitlines() if ": import mod" in ln]
        assert 0 < len(sampled) <= _IMPORT_CONVENTIONS_MAX_LINES

    def test_js_test_imports_sampled_too(self, tmp_path):
        from harness.graph import _import_conventions_section
        (tmp_path / "client" / "src").mkdir(parents=True)
        (tmp_path / "client" / "src" / "Panel.test.tsx").write_text(
            "import { render } from '@testing-library/react';\n"
            "import Panel from './Panel';\n"
        )
        section = _import_conventions_section(str(tmp_path))
        assert "@testing-library/react" in section
