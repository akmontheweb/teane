"""Truncation is the one failure that looks like success.

Ten of the eleven dispatch sites in this codebase never checked
``finish_reason``, and nine of those parse JSON — where a cut-off response
is not an error but silent corruption. The remedy for truncation is "emit
less", which is the opposite of the remedy for every other parse failure,
so naming the cause wrongly actively causes the repeat.

decomposition.py set ``parse_error = f"invalid_json: {exc}"`` on a payload
that was never invalid, only unfinished — and decomposition resolves
``continue_on_length`` to False (absent from both config.json's map and
_CONTINUE_ON_LENGTH_DEFAULTS), so there is no continuation to rescue it.
"""

from __future__ import annotations

from harness.gateway import LLMResponse, truncation_nudge


class TestTruncatedProperty:
    def test_length_is_truncated(self) -> None:
        r = LLMResponse(content="x", usage={}, model="m", finish_reason="length")
        assert r.truncated is True

    def test_stop_is_not(self) -> None:
        assert LLMResponse(
            content="x", usage={}, model="m", finish_reason="stop"
        ).truncated is False

    def test_default_is_not(self) -> None:
        assert LLMResponse(content="x", usage={}, model="m").truncated is False

    def test_tool_calls_finish_is_not(self) -> None:
        assert LLMResponse(
            content="", usage={}, model="m", finish_reason="tool_calls"
        ).truncated is False

    def test_empty_finish_reason_is_not(self) -> None:
        """Providers that omit the field must not read as truncated."""
        assert LLMResponse(
            content="x", usage={}, model="m", finish_reason=""
        ).truncated is False


class TestTruncationNudge:
    def test_states_the_response_was_not_rejected(self) -> None:
        """Naming the wrong cause is the bug this exists to prevent."""
        n = truncation_nudge("json")
        assert "cut off" in n
        assert "not rejected" in n

    def test_json_variant_asks_for_a_smaller_payload(self) -> None:
        assert "SMALLER payload" in truncation_nudge("json")

    def test_tests_variant_asks_for_fewer_test_functions(self) -> None:
        assert "SHORTER suite" in truncation_nudge("tests")

    def test_patch_variant_asks_for_fewer_blocks(self) -> None:
        assert "FEWER blocks" in truncation_nudge("patch")

    def test_unknown_kind_still_returns_usable_text(self) -> None:
        n = truncation_nudge("something-else")
        assert "SHORTER" in n and len(n) > 50

    def test_never_tells_the_model_its_content_was_wrong(self) -> None:
        for kind in ("json", "patch", "tests", "other"):
            n = truncation_nudge(kind).lower()
            assert "invalid" not in n, f"{kind} nudge misnames the cause"
