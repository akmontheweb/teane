"""The planner must not receive the specification twice (ADR-0008).

Measured on lumina 01a1de20 call 1: ``# Software Requirements Specification``
appeared TWICE in the decomposition prompt — once anchored in ``messages[0]``
(a SLIMMED copy, with ``**Priority:** / **Estimated size:** / **Wave:**``
stripped as planner-only) and once in the planner's own user prompt, read
from disk in full. 82,415 input tokens, of which ~26k was the duplicate.

The duplicate is not merely waste: the two copies DISAGREE. The planner —
the one node that needs priority and sizing — saw every story twice, once
with those fields and once with them silently missing.

These tests EXECUTE the node against a recording gateway double and assert on
the messages actually dispatched, rather than inspecting source text.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytest

from harness import decomposition

SRS_HEADING = "# Software Requirements Specification"

_SPEC = (
    f"{SRS_HEADING}\n\n"
    "## Epic: EPIC-001 — Todos\n\n"
    "### FR-001: Create a TODO\n"
    "**Priority:** High\n"
    "**Estimated size:** M\n"
    "**Wave:** 1\n"
    "POST /todos creates an item.\n\n"
    "### FR-002: List TODOs\n"
    "**Priority:** Low\n"
    "GET /todos returns the list.\n"
)


@dataclass
class _FakeResponse:
    content: str = ""
    tool_calls: list = field(default_factory=list)
    reasoning_content: str = ""


class _RecordingGateway:
    """Records the messages each dispatch actually received."""

    def __init__(self, payload: dict[str, Any]):
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    async def dispatch(
        self, *, messages, role, budget_remaining_usd, **_kw
    ) -> tuple[_FakeResponse, float]:
        self.calls.append({"messages": list(messages), "role": role})
        return (
            _FakeResponse(
                content="",
                tool_calls=[{
                    "name": "submit_decomposition",
                    "input": self._payload,
                    "id": "call_1",
                }],
            ),
            1.50,
        )


def _payload() -> dict[str, Any]:
    return {
        "features": [
            {"feature_key": "core", "name": "Core TODOs",
             "description": "MVP create/list."}
        ],
        "stories": [
            {
                "story_key": "STORY-001", "feature": "core",
                "title": "Create a TODO",
                "description": "POST /todos creates an item.",
                "requirement_keys": ["FR-001"],
                "acceptance_criteria": [
                    "POST /todos with title returns 201",
                    "Created item appears in GET /todos",
                ],
                "depends_on": [], "scope_files": ["src/todos/create.py"],
            },
            {
                "story_key": "STORY-002", "feature": "core",
                "title": "List TODOs",
                "description": "GET /todos returns the list.",
                "requirement_keys": ["FR-002"],
                "acceptance_criteria": ["GET /todos returns 200"],
                "depends_on": [], "scope_files": ["src/todos/list.py"],
            },
        ],
    }


@pytest.fixture
def workspace(tmp_path: Path) -> str:
    ws = tmp_path / "dup-ws"
    (ws / "docs").mkdir(parents=True)
    (ws / "docs" / "SPEC_REQUIREMENTS.md").write_text(_SPEC)
    return str(ws)


@pytest.fixture(autouse=True)
def _clear_gateway():
    from harness.graph import set_gateway
    g = set_gateway.__globals__
    prior = g.get("_gateway")
    g["_gateway"] = None
    yield
    g["_gateway"] = prior


def _anchored_system_prompt() -> str:
    """What messages[0] looks like in production: spec, then contract."""
    return _SPEC + "\n\n---\n\n" + "PATCHER OPERATING CONTRACT: edit invariants."


def _state(workspace: str, *, with_contract: bool = True) -> dict[str, Any]:
    st: dict[str, Any] = {
        "workspace_path": workspace,
        "messages": [{"role": "system", "content": _anchored_system_prompt()}],
        "budget_remaining_usd": 2.00,
    }
    if with_contract:
        st["system_prompt_no_spec"] = (
            "PATCHER OPERATING CONTRACT: edit invariants."
        )
    return st


def _run(workspace: str, **kw) -> _RecordingGateway:
    from harness.graph import set_gateway
    gw = _RecordingGateway(_payload())
    set_gateway(gw)
    asyncio.run(decomposition.decomposition_node(_state(workspace, **kw)))
    assert gw.calls, "the node never dispatched"
    return gw


class TestTheSpecArrivesExactlyOnce:
    def test_system_message_carries_no_spec(self, workspace: str):
        gw = _run(workspace)
        system_msg = gw.calls[0]["messages"][0]
        assert system_msg["role"] == "system"
        assert SRS_HEADING not in system_msg["content"]
        # ...but the contract it DOES need is still there.
        assert "PATCHER OPERATING CONTRACT" in system_msg["content"]

    def test_spec_reaches_the_planner_exactly_once(self, workspace: str):
        gw = _run(workspace)
        whole_prompt = "\n".join(
            m.get("content") or "" for m in gw.calls[0]["messages"]
        )
        assert whole_prompt.count(SRS_HEADING) == 1, (
            "the planner saw the spec "
            f"{whole_prompt.count(SRS_HEADING)} times"
        )

    def test_the_user_prompt_is_the_copy_that_survives(self, workspace: str):
        gw = _run(workspace)
        user_msgs = [
            m for m in gw.calls[0]["messages"] if m.get("role") == "user"
        ]
        assert user_msgs, "no user message dispatched"
        assert any(SRS_HEADING in (m.get("content") or "") for m in user_msgs)


class TestThePlannerKeepsWhatItNeeds:
    """The raw copy exists BECAUSE _slim_spec_for_prompt drops these."""

    @pytest.mark.parametrize(
        "planner_field",
        ["**Priority:** High", "**Estimated size:** M", "**Wave:** 1"],
    )
    def test_planner_only_fields_still_reach_the_planner(
        self, workspace: str, planner_field: str,
    ):
        gw = _run(workspace)
        whole_prompt = "\n".join(
            m.get("content") or "" for m in gw.calls[0]["messages"]
        )
        assert planner_field in whole_prompt


class TestFallback:
    def test_absent_contract_degrades_to_messages_zero(self, workspace: str):
        """An older checkpoint must keep working — degrade to the previous
        behaviour, never to an empty system message."""
        gw = _run(workspace, with_contract=False)
        system_msg = gw.calls[0]["messages"][0]
        assert system_msg["role"] == "system"
        assert "PATCHER OPERATING CONTRACT" in system_msg["content"]
        # The duplicate returns, which is the old behaviour by design.
        assert SRS_HEADING in system_msg["content"]


class TestCreateInitialStateStashesIt:
    def test_contract_is_stashed_and_spec_free(self, tmp_path: Path):
        from harness.graph import create_initial_state
        st = create_initial_state(
            workspace_path=str(tmp_path),
            initial_prompt="build it",
            build_command="true",
            spec_override=_SPEC,
        )
        contract = st.get("system_prompt_no_spec")
        assert contract, "create_initial_state did not stash the contract"
        assert SRS_HEADING not in contract
        # messages[0] still anchors the spec for every other role.
        assert SRS_HEADING in st["messages"][0]["content"]
        # And the anchor is exactly spec + separator + contract.
        assert st["messages"][0]["content"].endswith(contract)
