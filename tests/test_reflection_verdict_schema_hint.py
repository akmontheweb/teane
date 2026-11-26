"""The malformed-JSON repair hint must state the contract the parser enforces.

It had drifted from both the judge's own ``output_spec`` and
``_parse_repair_reflection_verdict``, in three ways that each lose signal:

  * ``WORKING_HYPOTHESIS`` was missing from the verdict set, so a malformed
    deferral was repaired into one of the other three and the "hold
    judgement, the LLM is testing an upstream theory" signal vanished.
  * the emptiness rule named only ``PROGRESS``, pushing the model to invent a
    ``real_blocker`` for a verdict allowed to omit one.
  * "exactly three keys ... No other keys" instructed the repair to DROP
    ``impl_file`` and ``wrong_test`` — the fields that send a round to
    PRODUCTION repair instead of rewriting the test.

The hint is now generated from the same constants the parser checks, so these
tests are drift detectors against the parser, not assertions about prose.
"""

import json

import pytest

from harness.graph import (
    _REFLECTION_OPTIONAL_KEYS,
    _REFLECTION_VERDICTS,
    _REFLECTION_VERDICTS_BLOCKER_OPTIONAL,
    _parse_repair_reflection_verdict,
    _reflection_verdict_schema_hint,
)


def _raw(**kw) -> str:
    return json.dumps(kw)


class TestTheParserIsTheContract:
    @pytest.mark.parametrize("verdict", _REFLECTION_VERDICTS)
    def test_every_declared_verdict_parses(self, verdict: str):
        out = _parse_repair_reflection_verdict(
            _raw(verdict=verdict, real_blocker="x fails at a.py:1",
                 recommendation="Edit a.py line 1."),
        )
        assert out is not None, f"{verdict} was rejected"
        assert out["verdict"] == verdict

    def test_an_undeclared_verdict_is_rejected(self):
        assert _parse_repair_reflection_verdict(
            _raw(verdict="MAYBE", real_blocker="b", recommendation="c"),
        ) is None

    @pytest.mark.parametrize("verdict", _REFLECTION_VERDICTS)
    def test_blocker_may_be_empty_exactly_where_declared(self, verdict: str):
        out = _parse_repair_reflection_verdict(
            _raw(verdict=verdict, real_blocker="", recommendation="Carry on."),
        )
        if verdict in _REFLECTION_VERDICTS_BLOCKER_OPTIONAL:
            assert out is not None, (
                f"{verdict} must tolerate an empty real_blocker"
            )
        else:
            assert out is None, (
                f"{verdict} must require a real_blocker"
            )

    @pytest.mark.parametrize("key", _REFLECTION_OPTIONAL_KEYS)
    def test_optional_keys_survive_parsing(self, key: str):
        """If the parser keeps them, the repair hint must not drop them."""
        out = _parse_repair_reflection_verdict(
            _raw(verdict="DISTRACTION", real_blocker="x fails at a.py:1",
                 recommendation="Edit server/app/svc.py.",
                 **{key: "server/app/svc.py"}),
        )
        assert out is not None
        assert out.get(key) == "server/app/svc.py"


class TestTheHintMatchesTheParser:
    @pytest.mark.parametrize("verdict", _REFLECTION_VERDICTS)
    def test_hint_names_every_accepted_verdict(self, verdict: str):
        assert verdict in _reflection_verdict_schema_hint(), (
            f"{verdict} parses but the repair hint never mentions it, so a "
            f"malformed {verdict} verdict gets repaired into another one"
        )

    @pytest.mark.parametrize("verdict", _REFLECTION_VERDICTS_BLOCKER_OPTIONAL)
    def test_hint_states_the_emptiness_rule(self, verdict: str):
        assert f"verdict='{verdict}'" in _reflection_verdict_schema_hint()

    @pytest.mark.parametrize("key", _REFLECTION_OPTIONAL_KEYS)
    def test_hint_tells_the_repair_to_keep_the_optional_keys(self, key: str):
        assert key in _reflection_verdict_schema_hint()

    def test_hint_does_not_forbid_the_optional_keys(self):
        """The original regression: a blanket 'No other keys' stripped
        impl_file and wrong_test, re-opening the misrouting that 5c48300
        and e64be1c exist to prevent."""
        hint = _reflection_verdict_schema_hint().lower()
        assert "no other keys" not in hint
        assert "exactly three keys" not in hint
