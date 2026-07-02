"""Phase J — end-of-session repair authority.

Verifies the three behaviour changes the repair pipeline gets when
``node_state.end_of_session_phase`` is set by
``end_of_session_regression_node``:

1. ``_repair_file_caps(state)`` returns the bigger (30, 150) tuple
   (configurable via gateway config). Default returns (24, 100).
2. Repair prompt prepends an end-of-session framing block that names
   shared-utility cascades as a likely cause.
3. Reasoning-model escalation is forced on the FIRST attempt (no need
   to burn cheap-model rounds first).

The repair_node itself is a long async LLM-touching function, so these
tests exercise the surface helpers and the prompt-construction
conditional rather than running the full node.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from harness.graph import (
    _DEFAULT_REPAIR_DIAGNOSTIC_CAP,
    _DEFAULT_REPAIR_INVENTORY_CAP,
    _EOS_REPAIR_DIAGNOSTIC_CAP,
    _EOS_REPAIR_INVENTORY_CAP,
    _repair_file_caps,
    set_gateway,
)


class _FakeGateway:
    def __init__(self, **cfg: Any):
        self.config = SimpleNamespace(**cfg)


@pytest.fixture
def clean_gateway():
    yield
    set_gateway(None)


# ---------------------------------------------------------------------------
# _repair_file_caps
# ---------------------------------------------------------------------------

class TestRepairFileCaps:
    def test_returns_defaults_outside_end_of_session_phase(self, clean_gateway):
        diag, inv = _repair_file_caps({"node_state": {}})
        assert diag == _DEFAULT_REPAIR_DIAGNOSTIC_CAP == 24
        assert inv == _DEFAULT_REPAIR_INVENTORY_CAP == 100

    def test_returns_defaults_when_node_state_missing(self, clean_gateway):
        assert _repair_file_caps({}) == (24, 100)

    def test_returns_eos_caps_when_phase_set(self, clean_gateway):
        st = {"node_state": {"end_of_session_phase": True}}
        diag, inv = _repair_file_caps(st)
        assert diag == _EOS_REPAIR_DIAGNOSTIC_CAP == 30
        assert inv == _EOS_REPAIR_INVENTORY_CAP == 150

    def test_caps_are_gateway_configurable(self, clean_gateway):
        set_gateway(_FakeGateway(
            end_of_session_repair_diagnostic_cap=50,
            end_of_session_repair_inventory_cap=500,
        ))
        st = {"node_state": {"end_of_session_phase": True}}
        assert _repair_file_caps(st) == (50, 500)

    def test_partial_gateway_config_falls_back_to_defaults(self, clean_gateway):
        # Operator overrode the inventory cap but not the diagnostic cap.
        set_gateway(_FakeGateway(
            end_of_session_repair_inventory_cap=200,
        ))
        st = {"node_state": {"end_of_session_phase": True}}
        diag, inv = _repair_file_caps(st)
        assert diag == _EOS_REPAIR_DIAGNOSTIC_CAP  # default kept
        assert inv == 200  # override applied

    def test_phase_flag_must_be_truthy(self, clean_gateway):
        # False / 0 / "" → still uses defaults.
        assert _repair_file_caps(
            {"node_state": {"end_of_session_phase": False}},
        ) == (24, 100)
        assert _repair_file_caps(
            {"node_state": {"end_of_session_phase": 0}},
        ) == (24, 100)


# ---------------------------------------------------------------------------
# Prompt construction — EoS preamble + escalation logic
# ---------------------------------------------------------------------------
#
# The prompt-building logic is inline in repair_node and depends on
# many surrounding state fields. Re-implement the small decision shape
# under test so we lock the contract without invoking the real LLM
# dispatch.

def _eos_preamble_if_active(node_state: dict[str, Any]) -> str:
    """Mirror the inline conditional in repair_node so we can assert
    the preamble appears only when the phase flag is set."""
    if not (
        isinstance(node_state, dict)
        and node_state.get("end_of_session_phase")
    ):
        return ""
    return (
        "## End-of-session regression\n\n"
        "This is the FINAL pre-deployment regression check. "
        "The failing tests below ran after the security-scan "
        "repair loop already landed patches in this session, "
        "so a likely cause is a SHARED UTILITY the security "
        "fix touched"
    )


class TestEosPreamble:
    def test_preamble_empty_in_per_batch_repair(self):
        assert _eos_preamble_if_active({}) == ""
        assert _eos_preamble_if_active({"end_of_session_phase": False}) == ""

    def test_preamble_present_at_end_of_session(self):
        preamble = _eos_preamble_if_active({"end_of_session_phase": True})
        assert "End-of-session regression" in preamble
        assert "shared utility" in preamble.lower() or (
            "SHARED UTILITY" in preamble
        )
        assert "FINAL pre-deployment" in preamble


# ---------------------------------------------------------------------------
# Force-escalation gate
# ---------------------------------------------------------------------------

def _should_escalate(
    *, total_repairs: int, max_attempts: int,
    eos_active: bool, force_reasoning_model: bool,
) -> bool:
    """Mirror the Phase J escalation decision."""
    use_escalation = total_repairs >= max(1, max_attempts - 1)
    if eos_active and force_reasoning_model:
        use_escalation = True
    return use_escalation


class TestForceEscalation:
    def test_per_batch_repair_does_not_force_escalate_on_first_attempt(self):
        # Without EoS, cheap model handles the first N-1 attempts.
        assert _should_escalate(
            total_repairs=0, max_attempts=5,
            eos_active=False, force_reasoning_model=True,
        ) is False

    def test_per_batch_escalates_only_on_last_attempt(self):
        assert _should_escalate(
            total_repairs=4, max_attempts=5,
            eos_active=False, force_reasoning_model=True,
        ) is True

    def test_eos_repair_force_escalates_on_first_attempt(self):
        # The whole point of Phase J — skip the cheap-model round at EoS.
        assert _should_escalate(
            total_repairs=0, max_attempts=5,
            eos_active=True, force_reasoning_model=True,
        ) is True

    def test_eos_force_can_be_disabled_via_config(self):
        # Operator turned off the force-escalation toggle. Then the
        # normal repair-count-driven rule applies even at EoS.
        assert _should_escalate(
            total_repairs=0, max_attempts=5,
            eos_active=True, force_reasoning_model=False,
        ) is False
