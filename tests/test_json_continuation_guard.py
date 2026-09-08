"""Continuing a JSON response is unsafe, and the critique that needed it was
unbounded to begin with.

lumina session 01a079dc. The architecture spec critique consumed 6 calls —
624,585 input / 196,608 output tokens, 24.3 minutes — and then failed to
parse, forcing a repair pass. 30.1 min and 43% of the session's spend on one
critique that produced nothing. Four compounding defects:

  1. ``_SPEC_REVIEW_SYSTEM_PROMPT`` requested eight open-ended arrays and told
     the reviewer to go "field by field ... every displayed value, computed
     output, form input, alert, and acceptance criterion" over a 62k-char
     spec, with no cap on item count or item length. Hitting the 32,768
     output cap was the requested behaviour.
  2. ``continue_on_length.doc_reviewer`` was true in config.json, overriding
     this module's own False default and the "JSON critique continuation is
     RISKY" comment beside it. Five cycles fired; every one hit the cap
     again, because defect 1 gave the model no reason to stop.
  3. Each cycle resends the accumulated transcript, so input grew
     22k → 55k → 88k → 120k → 153k → 186k. Cost is quadratic in cycles.
  4. The chunks were rejoined with ``"\\n".join``. The continue prompt says
     "Stay inside the same JSON object", so the chunks are one continuous
     token stream — a newline at a boundary that landed inside a string
     literal is a raw control character. lumina's error was "Invalid control
     character at: line 397 column 261".
"""

from __future__ import annotations

import asyncio
import json

from harness.graph import _SPEC_REVIEW_SYSTEM_PROMPT, _continue_on_length


class _Resp:
    def __init__(self, content, finish_reason="length"):
        self.content = content
        self.finish_reason = finish_reason
        self.tool_calls = None


def _run(**kw):
    return asyncio.run(_continue_on_length(**kw))


def _never_called(msgs, budget):  # pragma: no cover - must not run
    raise AssertionError("dispatch must not be called")


class TestJsonOutputRefusesContinuation:
    def test_json_output_short_circuits(self):
        resp, budget, chunks = _run(
            initial_response=_Resp('{"issues": ["a'),
            initial_budget=9.0,
            messages=[],
            dispatch=_never_called,
            continue_prompt="continue",
            enabled=True,          # config says yes …
            role_label="spec_review:critique",
            json_output=True,      # … the call site's shape says no
        )
        assert chunks == ['{"issues": ["a']
        assert budget == 9.0

    def test_shape_gate_beats_the_config_flag(self):
        """The flag is keyed by role NAME and cannot express "this call site
        parses JSON" — so the call site has to, and it must win."""
        calls = []

        async def _dispatch(msgs, budget):
            calls.append(1)
            return _Resp("more", finish_reason="stop"), budget - 1

        _run(
            initial_response=_Resp("partial"),
            initial_budget=9.0, messages=[], dispatch=_dispatch,
            continue_prompt="continue", enabled=True,
            role_label="spec_review:critique", json_output=True,
        )
        assert calls == []

    def test_text_output_still_continues(self):
        """Patch-DSL and prose callers are unaffected — they are the reason
        the loop exists."""
        seen = []

        async def _dispatch(msgs, budget):
            seen.append(len(msgs))
            return _Resp("tail", finish_reason="stop"), budget - 1

        resp, budget, chunks = _run(
            initial_response=_Resp("head"),
            initial_budget=9.0, messages=[], dispatch=_dispatch,
            continue_prompt="continue", enabled=True,
            role_label="patching_node",
        )
        assert chunks == ["head", "tail"]
        assert budget == 8.0
        assert len(seen) == 1

    def test_disabled_still_short_circuits(self):
        _, _, chunks = _run(
            initial_response=_Resp("head"),
            initial_budget=9.0, messages=[], dispatch=_never_called,
            continue_prompt="c", enabled=False, role_label="x",
        )
        assert chunks == ["head"]

    def test_cycles_are_capped(self):
        async def _always_truncated(msgs, budget):
            return _Resp("x"), budget - 1

        _, budget, chunks = _run(
            initial_response=_Resp("head"),
            initial_budget=9.0, messages=[], dispatch=_always_truncated,
            continue_prompt="c", enabled=True, role_label="patching_node",
            max_cycles=3,
        )
        assert len(chunks) == 4       # initial + 3
        assert budget == 6.0


class TestDegenerateRepeatRefusesContinuation:
    """A truncated turn that is mostly verbatim repeats is a stuck model, not
    work in progress — continuing it re-sends the whole degenerate turn as
    context and re-primes the loop.

    lumina 01a079dc, repair round 9: 163 copies of the same four paragraphs
    filled 32,768 output tokens over 4m19s ($0.039), and the harness then paid
    for a continuation on top of it.
    """

    # Four distinct paragraphs, repeated — the shape of the real failure.
    LOOP = "\n".join([
        "The problem is that the database contains records from previous runs.",
        "Since I can't edit the test conftest, I must modify production code.",
        "Let me look at the create_app function and the initialization code.",
        "The fix should be in server/app/main.py where create_app is defined.",
    ] * 40)

    def test_degenerate_initial_turn_is_not_continued(self):
        _, budget, chunks = _run(
            initial_response=_Resp(self.LOOP),
            initial_budget=9.0, messages=[], dispatch=_never_called,
            continue_prompt="c", enabled=True, role_label="repair",
        )
        # Whatever landed is kept — the caller's parser takes what it can.
        assert chunks == [self.LOOP]
        assert budget == 9.0

    def test_loop_that_starts_mid_continuation_stops_there(self):
        calls = []

        async def _dispatch(msgs, budget):
            calls.append(1)
            return _Resp(self.LOOP), budget - 1

        _, budget, chunks = _run(
            initial_response=_Resp("head"),
            initial_budget=9.0, messages=[], dispatch=_dispatch,
            continue_prompt="c", enabled=True, role_label="repair",
            max_cycles=5,
        )
        assert len(calls) == 1        # bailed instead of burning 5 cycles
        assert chunks == ["head", self.LOOP]
        assert budget == 8.0

    def test_real_patch_output_is_not_mistaken_for_a_loop(self):
        # REPLACE_BLOCK search/replace pairs repeat lines by construction; the
        # highest legitimate ratio measured in 01a079dc was 0.462.
        from harness.graph import _is_degenerate_repeat, _repetition_ratio
        patch = (
            "<<<REPLACE_BLOCK>>>\nfile: server/app/main.py\nsearch:\n"
            + "\n".join(f"    value_{i} = compute_something(argument_{i})"
                        for i in range(12))
            + "\nreplace:\n"
            + "\n".join(f"    value_{i} = compute_something(argument_{i})"
                        for i in range(12))
            + "\n<<<END_REPLACE_BLOCK>>>\n"
        )
        assert _repetition_ratio(patch) < 0.7
        assert not _is_degenerate_repeat(patch)

    def test_short_output_never_trips_the_detector(self):
        from harness.graph import _repetition_ratio
        assert _repetition_ratio("same line repeated\n" * 3) == 0.0


class TestJsonCritiqueRolesStayDisabledInConfig:
    """Both JSON-critique reviewers must stay off in the shipped config.

    ``continue_on_length.doc_reviewer: true`` in config.json is what overrode
    the module default and produced lumina 01a079dc's 4 wasted cycles (131k
    output tokens, 16 min, input growing 87k → 186k). ``code_reviewer`` sat at
    ``true`` beside it with a byte-identical JSON critique call site. The
    ``json_output`` shape gate now protects both at the call site, but config
    drifting back to ``true`` restores the cost even when the gate saves the
    parse — so pin it.
    """

    ROLES = ("doc_reviewer", "code_reviewer")

    def _config(self):
        from harness.cli import _strip_comments, load_raw_config
        return _strip_comments(load_raw_config())

    def test_shipped_config_disables_json_critique_continuation(self):
        cfg = self._config().get("llm_dispatch", {}).get("continue_on_length", {})
        for role in self.ROLES:
            assert cfg.get(role) is False, f"{role} must stay False in config.json"

    def test_config_agrees_with_module_defaults(self):
        from harness.graph import _CONTINUE_ON_LENGTH_DEFAULTS
        cfg = self._config().get("llm_dispatch", {}).get("continue_on_length", {})
        for role in self.ROLES:
            assert cfg.get(role) == _CONTINUE_ON_LENGTH_DEFAULTS[role]


class TestLosslessReassembly:
    """The concrete corruption, reproduced. A cycle boundary that lands inside
    a string literal is the case that matters; every other boundary is
    insignificant JSON whitespace, so "" is never worse than "\\n"."""

    SPLIT_MID_STRING = (
        '{"contradictions": [{"id": "C1", "detail": "The architecture says '
        'the audit logger',
        ' propagates to root, contradicting the requirements."}]}',
    )

    def test_newline_join_corrupts_a_mid_string_split(self):
        broken = "\n".join(self.SPLIT_MID_STRING)
        try:
            json.loads(broken)
        except ValueError as exc:
            assert "control character" in str(exc).lower()
        else:  # pragma: no cover
            raise AssertionError("expected the historical corruption")

    def test_empty_join_reassembles_cleanly(self):
        parsed = json.loads("".join(self.SPLIT_MID_STRING))
        assert parsed["contradictions"][0]["id"] == "C1"
        assert "propagates to root" in parsed["contradictions"][0]["detail"]

    def test_empty_join_is_safe_between_tokens_too(self):
        parts = ('{"ambiguity": ["a",', ' "b"]}')
        assert json.loads("".join(parts))["ambiguity"] == ["a", "b"]


class TestCritiquePromptIsBounded:
    def test_caps_are_stated(self):
        p = _SPEC_REVIEW_SYSTEM_PROMPT
        assert "10 items per array" in p
        assert "240 characters per item" in p
        assert "followup_questions" in p

    def test_overflow_instruction_is_drop_not_continue(self):
        """The failure mode is a critique that runs past the cap and is then
        discarded whole, so the instruction must be to drop the tail — never
        to keep going."""
        p = _SPEC_REVIEW_SYSTEM_PROMPT
        assert "DROP THE REST" in p
        assert "does not parse" in p

    def test_drop_detection_survives_the_bounding(self):
        """dropped_requirements is the category where exhaustiveness actually
        matters, so it keeps a higher cap and the source-of-truth framing."""
        p = _SPEC_REVIEW_SYSTEM_PROMPT
        assert "dropped_requirements" in p
        assert "up to 15" in p
        assert "SOURCE OF TRUTH" in p

    def test_saturated_critique_fits_the_output_cap(self):
        """A critique obeying every cap must land well inside
        max_tokens_per_role.doc_reviewer (32768), or the bound is decorative.
        ~4 chars/token, which is conservative for English prose."""
        caps = {
            "completeness": 10, "dropped_requirements": 15,
            "contradictions": 10, "ambiguity": 10, "missing_edge_cases": 10,
            "security_gaps": 10, "testability": 10,
        }
        worst_chars = sum(caps.values()) * 240 + 6 * 260
        assert worst_chars // 4 < 32768 // 2, worst_chars // 4
