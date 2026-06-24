"""Tests for the LLM-judgment focus picker on discovery follow-up rounds.

``harness.graph._maybe_discovery_followup_focus`` runs at the top of each
follow-up round in ``requirements_discovery_node`` and
``architecture_discovery_node``. It asks a cheap LLM to pick 3-5 sectors
worth re-auditing this round, then the node splices those into the
``{FOCUS_SECTORS_BLOCK}`` slot.

These tests cover the fail-open contract end-to-end:

  - Disabled by config → no dispatch, returns None unchanged.
  - First round → no dispatch (focus only fires from round 2 onward).
  - No gateway → no dispatch.
  - Budget exhausted → no dispatch.
  - Garbage JSON → fail-open, return None, budget reflects the spend.
  - Unknown sector names in response → filtered out.
  - Below floor (<3 valid) → fail-open.
  - Valid JSON with known names → return the validated list.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

class _StubUsage:
    input_tokens = 30
    output_tokens = 20
    cached_tokens = 0
    cost_usd = 0.0005
    model = "stub-judgment"


class _StubResponse:
    def __init__(self, content: str):
        self.content = content
        self.usage = _StubUsage()


class _StubGateway:
    """Minimal gateway for judgment-helper tests.

    Records dispatches and returns whatever canned content was passed in.
    Each dispatch deducts a fixed $0.005 from the budget so we can assert
    cost-tracking propagates.
    """
    def __init__(self, content: str, *, followup_focus_enabled: bool = True,
                 repair_primary: str = "stub-repair-model"):
        self._content = content
        self.dispatched: list[dict] = []
        self.config = SimpleNamespace(
            repair_primary=repair_primary,
            llm_judgment_discovery_followup_focus=followup_focus_enabled,
        )

    async def dispatch(self, *, messages, role, budget_remaining_usd, **kwargs):
        self.dispatched.append({"messages": list(messages), "role": role})
        return _StubResponse(self._content), budget_remaining_usd - 0.005


@pytest.fixture
def install_gateway():
    from harness import graph as graph_mod
    installed: list[_StubGateway] = []

    def _install(content: str, **kwargs) -> _StubGateway:
        gw = _StubGateway(content, **kwargs)
        graph_mod.set_gateway(gw)
        installed.append(gw)
        return gw

    yield _install
    graph_mod.set_gateway(None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def test_first_round_skips_focus(install_gateway):
    # question_count=0 means round 1 is about to fire — there's nothing
    # to focus yet. The helper must NOT spend a dispatch.
    from harness.graph import _maybe_discovery_followup_focus, _REQUIREMENTS_SECTORS
    gw = install_gateway('{"focus": ["USER ROLES & PERSONAS"]}')

    focus, new_budget = _run(_maybe_discovery_followup_focus(
        gate="REQUIREMENTS", question_count=0,
        sectors=_REQUIREMENTS_SECTORS, messages=[], budget=1.0,
    ))

    assert focus is None
    assert new_budget == 1.0
    assert gw.dispatched == [], "Helper dispatched on round 1 — should have skipped."


def test_disabled_by_config_skips_dispatch(install_gateway):
    from harness.graph import _maybe_discovery_followup_focus, _REQUIREMENTS_SECTORS
    gw = install_gateway(
        '{"focus": ["USER ROLES & PERSONAS","EPICS & USER STORIES","INPUT VALIDATION & PAYLOAD FORMAT"]}',
        followup_focus_enabled=False,
    )

    focus, new_budget = _run(_maybe_discovery_followup_focus(
        gate="REQUIREMENTS", question_count=2,
        sectors=_REQUIREMENTS_SECTORS, messages=[], budget=1.0,
    ))

    assert focus is None
    assert new_budget == 1.0
    assert gw.dispatched == []


def test_no_repair_model_skips_dispatch(install_gateway):
    # The judgment helpers all reuse the repair role; without a repair
    # model routed there's nothing cheap to dispatch on.
    from harness.graph import _maybe_discovery_followup_focus, _REQUIREMENTS_SECTORS
    gw = install_gateway(
        '{"focus": ["USER ROLES & PERSONAS"]}',
        repair_primary="",
    )

    focus, new_budget = _run(_maybe_discovery_followup_focus(
        gate="REQUIREMENTS", question_count=2,
        sectors=_REQUIREMENTS_SECTORS, messages=[], budget=1.0,
    ))

    assert focus is None
    assert new_budget == 1.0
    assert gw.dispatched == []


def test_budget_under_floor_skips_dispatch(install_gateway):
    from harness.graph import _maybe_discovery_followup_focus, _REQUIREMENTS_SECTORS
    gw = install_gateway(
        '{"focus": ["USER ROLES & PERSONAS"]}',
    )

    focus, new_budget = _run(_maybe_discovery_followup_focus(
        gate="REQUIREMENTS", question_count=2,
        sectors=_REQUIREMENTS_SECTORS, messages=[], budget=0.001,
    ))

    assert focus is None
    assert new_budget == 0.001
    assert gw.dispatched == []


def test_garbage_json_fail_opens(install_gateway):
    # If the LLM returns prose instead of JSON the helper must return
    # None (caller renders the empty focus block and proceeds unfocused).
    from harness.graph import _maybe_discovery_followup_focus, _ARCHITECTURE_SECTORS
    gw = install_gateway("this is not JSON at all")

    focus, new_budget = _run(_maybe_discovery_followup_focus(
        gate="ARCHITECTURE", question_count=2,
        sectors=_ARCHITECTURE_SECTORS, messages=[], budget=1.0,
    ))

    assert focus is None
    # Budget reflects the dispatch cost — the LLM call did fire, it just
    # didn't produce parseable JSON.
    assert new_budget == pytest.approx(1.0 - 0.005)
    assert len(gw.dispatched) == 1


def test_unknown_sectors_filtered_below_floor_returns_none(install_gateway):
    # LLM hallucinated sector names that don't exist in the gate's
    # taxonomy. After filtering we're below the floor (=3), so fail
    # open — better an unfocused round than asking 1-2 questions in
    # the wrong scope.
    from harness.graph import _maybe_discovery_followup_focus, _REQUIREMENTS_SECTORS
    install_gateway(
        '{"focus": ["INVENTED SECTOR 1", "USER ROLES & PERSONAS", "MYSTERY ZONE"]}',
    )

    focus, _ = _run(_maybe_discovery_followup_focus(
        gate="REQUIREMENTS", question_count=2,
        sectors=_REQUIREMENTS_SECTORS, messages=[], budget=1.0,
    ))

    assert focus is None


def test_valid_focus_returns_validated_list(install_gateway):
    from harness.graph import _maybe_discovery_followup_focus, _REQUIREMENTS_SECTORS
    picked = [
        "SECURITY CONTROLS & THREAT MODEL",
        "ABUSE & MISUSE CASES",
        "COMPLIANCE & DATA CLASSIFICATION",
        "DATA RETENTION & LIFECYCLE",
    ]
    import json
    gw = install_gateway(json.dumps({"focus": picked}))

    focus, new_budget = _run(_maybe_discovery_followup_focus(
        gate="REQUIREMENTS", question_count=2,
        sectors=_REQUIREMENTS_SECTORS, messages=[], budget=1.0,
    ))

    assert focus == picked
    assert new_budget == pytest.approx(1.0 - 0.005)
    assert len(gw.dispatched) == 1


def test_max_cap_enforced(install_gateway):
    # The helper caps at 5 even if the LLM returns more.
    from harness.graph import _maybe_discovery_followup_focus, _ARCHITECTURE_SECTORS
    import json
    huge = list(_ARCHITECTURE_SECTORS[:8])  # 8 valid sectors
    install_gateway(json.dumps({"focus": huge}))

    focus, _ = _run(_maybe_discovery_followup_focus(
        gate="ARCHITECTURE", question_count=2,
        sectors=_ARCHITECTURE_SECTORS, messages=[], budget=1.0,
    ))

    assert focus is not None
    assert len(focus) == 5, f"Cap not enforced, got {len(focus)} entries."
    assert focus == huge[:5]


def test_dedupes_repeated_sector_names(install_gateway):
    # An LLM that repeats a sector name shouldn't pad the focus list.
    from harness.graph import _maybe_discovery_followup_focus, _ARCHITECTURE_SECTORS
    import json
    install_gateway(json.dumps({"focus": [
        "DATA MODEL & OWNERSHIP",
        "DATA MODEL & OWNERSHIP",  # duplicate
        "TRUST BOUNDARIES & SECURITY ZONES",
        "FAILURE DOMAINS & RESILIENCE PATTERNS",
    ]}))

    focus, _ = _run(_maybe_discovery_followup_focus(
        gate="ARCHITECTURE", question_count=2,
        sectors=_ARCHITECTURE_SECTORS, messages=[], budget=1.0,
    ))

    assert focus == [
        "DATA MODEL & OWNERSHIP",
        "TRUST BOUNDARIES & SECURITY ZONES",
        "FAILURE DOMAINS & RESILIENCE PATTERNS",
    ]


def test_render_focus_block_empty():
    from harness.graph import _render_focus_block
    assert _render_focus_block([]) == ""


def test_render_focus_block_populated():
    from harness.graph import _render_focus_block
    rendered = _render_focus_block(["A", "B", "C"])
    assert "Focus this round" in rendered
    assert "- A" in rendered
    assert "- B" in rendered
    assert "- C" in rendered


def test_sector_constants_match_markdown_file():
    # Guard against drift between the in-code sector tuples and the
    # human-edited prompt markdown. Every name in _REQUIREMENTS_SECTORS
    # must appear in the requirements_discovery.md file (and the same
    # for architecture).
    from harness import docgen_prompts
    from harness.graph import _REQUIREMENTS_SECTORS, _ARCHITECTURE_SECTORS

    req_md = docgen_prompts.load("requirements_discovery")
    for name in _REQUIREMENTS_SECTORS:
        assert name in req_md, (
            f"Sector '{name}' from _REQUIREMENTS_SECTORS missing from "
            f"requirements_discovery.md — either rename in code or "
            f"update the markdown."
        )

    arch_md = docgen_prompts.load("architecture_discovery")
    for name in _ARCHITECTURE_SECTORS:
        assert name in arch_md, (
            f"Sector '{name}' from _ARCHITECTURE_SECTORS missing from "
            f"architecture_discovery.md — either rename in code or "
            f"update the markdown."
        )
