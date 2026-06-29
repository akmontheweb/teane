"""Unit tests for harness/decomposition.py — the spec → stories node."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pytest

from harness import decomposition, story_state


# ---------------------------------------------------------------------------
# Test gateway double
# ---------------------------------------------------------------------------

@dataclass
class _FakeResponse:
    content: str


class _FakeGateway:
    """Replaces ``harness.gateway.Gateway`` for tests.

    Behavior:
      - ``responses`` queue: pop one per ``dispatch`` call.
      - If the queue is empty, raises so tests don't hang on a missing
        stub.
      - ``raise_on_call``: when set, raise the given exception instead
        of returning a response (covers the gateway-error branch).
    """

    def __init__(
        self,
        responses: list[str],
        *,
        raise_on_call: Optional[Exception] = None,
        budget_after: float = 1.50,
    ):
        self._responses = list(responses)
        self._raise = raise_on_call
        self._budget = budget_after
        self.calls: list[dict[str, Any]] = []

    async def dispatch(
        self, *, messages, role, budget_remaining_usd, **_kw
    ) -> tuple[_FakeResponse, float]:
        self.calls.append({
            "messages": list(messages),
            "role": role,
            "budget_in": budget_remaining_usd,
        })
        if self._raise is not None:
            raise self._raise
        if not self._responses:
            raise AssertionError("fake gateway out of responses")
        content = self._responses.pop(0)
        return _FakeResponse(content=content), self._budget


@pytest.fixture
def workspace(tmp_path: Path) -> str:
    ws = tmp_path / "decomp-ws"
    ws.mkdir()
    return str(ws)


@pytest.fixture(autouse=True)
def _clear_gateway():
    """Each test starts with no gateway registered."""
    from harness.graph import set_gateway
    prior_g = set_gateway.__globals__.get("_gateway")
    prior_c = set_gateway.__globals__.get("_gateway_config")
    set_gateway.__globals__["_gateway"] = None
    set_gateway.__globals__["_gateway_config"] = None
    yield
    set_gateway.__globals__["_gateway"] = prior_g
    set_gateway.__globals__["_gateway_config"] = prior_c


def _write_spec(workspace: str, body: str = "Build a TODO API.") -> None:
    docs = os.path.join(workspace, "docs")
    os.makedirs(docs, exist_ok=True)
    Path(os.path.join(docs, "SPEC_REQUIREMENTS.md")).write_text(body)


def _build_state(workspace: str, budget: float = 2.00) -> dict[str, Any]:
    return {
        "workspace_path": workspace,
        "messages": [{"role": "system", "content": "system"}],
        "budget_remaining_usd": budget,
    }


def _valid_payload() -> str:
    return json.dumps({
        "features": [
            {"feature_key": "core", "name": "Core TODOs",
             "description": "MVP create/list."}
        ],
        "stories": [
            {
                "story_key": "STORY-1",
                "feature": "core",
                "title": "Create a TODO",
                "description": "POST /todos creates an item.",
                "requirement_keys": ["FR-001"],
                "acceptance_criteria": [
                    "POST /todos with title returns 201",
                    "Created item appears in GET /todos",
                ],
                "depends_on": [],
                "scope_files": ["src/todos/create.py"],
            },
            {
                "story_key": "STORY-2",
                "feature": "core",
                "title": "List TODOs",
                "description": "GET /todos returns the list.",
                "requirement_keys": ["FR-002"],
                "acceptance_criteria": ["GET /todos returns JSON array"],
                "depends_on": ["STORY-1"],
                "scope_files": ["src/todos/list.py"],
            },
        ],
        "summary": "Two stories: create and list.",
    })


def _seed_story(workspace_app: str, title: str, **extra) -> str:
    """Test helper: create the ``test`` feature if missing, then insert
    a single story under it. Returns the assigned story_key."""
    conn = story_state.open_story_db()
    try:
        story_state.ensure_feature(
            conn, workspace_app, "test", name="Test feature",
            description="Seed for tests.",
        )
        keys = story_state.create_stories(
            conn, workspace_app,
            [{
                "title": title,
                "feature": extra.pop("feature", "test"),
                "acceptance_criteria": extra.pop(
                    "acceptance_criteria", ["x"],
                ),
                "depends_on": extra.pop("depends_on", []),
                "scope_files": extra.pop("scope_files", []),
                **extra,
            }],
        )
    finally:
        conn.close()
    return keys[0]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _payload_with_one_feature(stories: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap a story list in the minimal valid features+stories envelope.

    Every story gets ``feature: "core"`` if it doesn't already specify
    one, plus a default ``requirement_keys: ["FR-001"]`` if it doesn't
    already specify one (v5 contract: every story must cite ≥1 req
    key). Validator tests that exercise the req_keys rule itself
    override the field. Used by validator tests that focus on
    story-level rules rather than the feature-validation logic.
    """
    for s in stories:
        s.setdefault("feature", "core")
        s.setdefault("requirement_keys", ["FR-001"])
    return {
        "features": [{"feature_key": "core", "name": "Core"}],
        "stories": stories,
    }


def test_validate_accepts_minimal_valid_payload():
    payload = json.loads(_valid_payload())
    features, stories = decomposition._validate_stories_payload(payload)
    assert len(features) == 1
    assert features[0]["feature_key"] == "core"
    assert len(stories) == 2
    assert stories[0]["title"] == "Create a TODO"
    assert stories[0]["feature"] == "core"
    assert stories[1]["depends_on"] == ["STORY-1"]


def test_validate_rejects_non_dict():
    with pytest.raises(ValueError, match="top-level"):
        decomposition._validate_stories_payload(["not", "an", "object"])


def test_validate_rejects_empty_stories():
    payload = {
        "features": [{"feature_key": "core", "name": "Core"}],
        "stories": [],
    }
    with pytest.raises(ValueError, match="non-empty"):
        decomposition._validate_stories_payload(payload)


def test_validate_rejects_missing_features():
    """v4 schema makes features mandatory in initial decomposition."""
    payload = {"stories": [{
        "story_key": "STORY-1", "feature": "core",
        "title": "t", "acceptance_criteria": ["x"],
    }]}
    with pytest.raises(ValueError, match="features"):
        decomposition._validate_stories_payload(payload)


def test_validate_rejects_story_referencing_undeclared_feature():
    payload = {
        "features": [{"feature_key": "core", "name": "Core"}],
        "stories": [{
            "story_key": "STORY-1", "feature": "ghost",
            "title": "t", "acceptance_criteria": ["x"],
            "requirement_keys": ["FR-001"],
        }],
    }
    with pytest.raises(ValueError, match="ghost"):
        decomposition._validate_stories_payload(payload)


def test_validate_rejects_orphan_feature():
    """Every declared feature must own at least one story."""
    payload = {
        "features": [
            {"feature_key": "core", "name": "Core"},
            {"feature_key": "orphan", "name": "Nobody"},
        ],
        "stories": [{
            "story_key": "STORY-1", "feature": "core",
            "title": "t", "acceptance_criteria": ["x"],
            "requirement_keys": ["FR-001"],
        }],
    }
    with pytest.raises(ValueError, match="orphan"):
        decomposition._validate_stories_payload(payload)


def test_validate_rejects_over_cap():
    payload = _payload_with_one_feature([
        {
            "story_key": f"STORY-{i}",
            "title": f"t{i}",
            "acceptance_criteria": ["x"],
        }
        for i in range(1, decomposition.MAX_STORIES_PER_PASS + 2)
    ])
    with pytest.raises(ValueError, match="too many"):
        decomposition._validate_stories_payload(payload)


def test_validate_rejects_bad_story_key():
    payload = _payload_with_one_feature([{
        "story_key": "ABC-1", "title": "t", "acceptance_criteria": ["x"]
    }])
    with pytest.raises(ValueError, match="invalid story_key"):
        decomposition._validate_stories_payload(payload)


def test_validate_rejects_missing_acceptance():
    payload = _payload_with_one_feature([{
        "story_key": "STORY-1", "title": "t", "acceptance_criteria": []
    }])
    with pytest.raises(ValueError, match="acceptance"):
        decomposition._validate_stories_payload(payload)


def test_validate_rejects_forward_dependency():
    payload = _payload_with_one_feature([
        {"story_key": "STORY-1", "title": "a",
         "acceptance_criteria": ["x"], "depends_on": ["STORY-2"]},
        {"story_key": "STORY-2", "title": "b",
         "acceptance_criteria": ["y"]},
    ])
    with pytest.raises(ValueError, match="not declared earlier"):
        decomposition._validate_stories_payload(payload)


def test_validate_rejects_duplicate_keys():
    payload = _payload_with_one_feature([
        {"story_key": "STORY-1", "title": "a", "acceptance_criteria": ["x"]},
        {"story_key": "STORY-1", "title": "b", "acceptance_criteria": ["y"]},
    ])
    with pytest.raises(ValueError, match="duplicate"):
        decomposition._validate_stories_payload(payload)


# ---------------------------------------------------------------------------
# Fence stripping
# ---------------------------------------------------------------------------

def test_strip_json_fence_handles_fenced():
    raw = "```json\n{\"a\": 1}\n```"
    assert decomposition.strip_json_fence(raw) == '{"a": 1}'


def test_strip_json_fence_passes_through_clean():
    assert decomposition.strip_json_fence('{"a": 1}') == '{"a": 1}'


# ---------------------------------------------------------------------------
# Node — happy path
# ---------------------------------------------------------------------------

def test_decomposition_node_happy_path(workspace: str, monkeypatch):
    from harness.graph import set_gateway
    _write_spec(workspace)
    gw = _FakeGateway([_valid_payload()])
    set_gateway(gw)

    state = _build_state(workspace)
    out = asyncio.run(decomposition.decomposition_node(state))

    assert out["current_gate"] == "STORIES"
    assert out["node_state"]["decomposition_complete"] is True
    assert out["node_state"]["story_count"] == 2
    assert out["node_state"]["story_keys"] == ["STORY-1", "STORY-2"]
    assert out["stories_db_path"].endswith("state.db")
    assert out["budget_remaining_usd"] == 1.50

    # DB has the stories
    app = story_state.app_name_for_workspace(workspace)
    conn = story_state.open_story_db()
    try:
        stories = story_state.list_stories(conn, app)
    finally:
        conn.close()
    assert [s["story_key"] for s in stories] == ["STORY-1", "STORY-2"]
    assert stories[1]["depends_on"] == ["STORY-1"]

    # Markdown view regenerated
    assert os.path.exists(os.path.join(workspace, "docs", "STORIES.md"))


def test_decomposition_node_uses_planning_role(workspace: str):
    from harness.graph import set_gateway
    from harness.gateway import NodeRole
    _write_spec(workspace)
    gw = _FakeGateway([_valid_payload()])
    set_gateway(gw)

    asyncio.run(decomposition.decomposition_node(_build_state(workspace)))
    assert gw.calls[0]["role"] == NodeRole.PLANNING


# ---------------------------------------------------------------------------
# Node — error paths
# ---------------------------------------------------------------------------

def test_decomposition_node_no_spec(workspace: str):
    """Spec missing → graceful error, no DB write for this app."""
    from harness.graph import set_gateway
    set_gateway(_FakeGateway([]))  # should not be called
    out = asyncio.run(decomposition.decomposition_node(_build_state(workspace)))
    assert out["node_state"]["error"] == "spec_requirements_missing"
    assert out["node_state"]["story_count"] == 0
    # Global state.db may or may not exist depending on test ordering;
    # the important guarantee is that no rows landed for this app.
    app = story_state.app_name_for_workspace(workspace)
    if os.path.isfile(story_state.state_db_path()):
        conn = story_state.open_story_db()
        try:
            assert story_state.list_stories(conn, app) == []
        finally:
            conn.close()


def test_decomposition_node_no_gateway(workspace: str):
    _write_spec(workspace)
    out = asyncio.run(decomposition.decomposition_node(_build_state(workspace)))
    assert out["node_state"]["error"] == "no_gateway"


def test_decomposition_node_budget_exhausted(workspace: str):
    from harness.graph import set_gateway
    _write_spec(workspace)
    set_gateway(_FakeGateway([]))
    out = asyncio.run(
        decomposition.decomposition_node(_build_state(workspace, budget=0.0))
    )
    assert out["node_state"]["error"] == "budget_exhausted"


def test_decomposition_node_invalid_json(workspace: str):
    from harness.graph import set_gateway
    _write_spec(workspace)
    set_gateway(_FakeGateway(["not actually json"]))
    out = asyncio.run(decomposition.decomposition_node(_build_state(workspace)))
    assert out["node_state"]["error"].startswith("invalid_json")
    # DB should not have been populated for this app
    app = story_state.app_name_for_workspace(workspace)
    conn = story_state.open_story_db()
    try:
        assert story_state.list_stories(conn, app) == []
    finally:
        conn.close()


def test_decomposition_node_validation_failure(workspace: str):
    from harness.graph import set_gateway
    _write_spec(workspace)
    bad = json.dumps({"stories": [{
        "story_key": "BAD-1", "title": "x", "acceptance_criteria": ["y"]
    }]})
    set_gateway(_FakeGateway([bad]))
    out = asyncio.run(decomposition.decomposition_node(_build_state(workspace)))
    assert out["node_state"]["error"].startswith("validation")


def test_decomposition_node_dispatch_exception(workspace: str):
    from harness.graph import set_gateway
    _write_spec(workspace)
    set_gateway(_FakeGateway([], raise_on_call=RuntimeError("upstream 503")))
    out = asyncio.run(decomposition.decomposition_node(_build_state(workspace)))
    assert out["node_state"]["error"].startswith("dispatch_failed")
    assert "upstream 503" in out["node_state"]["error"]


# ---------------------------------------------------------------------------
# Augment mode — delta-only decomposition on workspaces with existing stories
# ---------------------------------------------------------------------------

def test_augment_prompt_includes_existing_stories():
    p = decomposition._build_decomposition_augment_prompt(
        existing_features=[
            {"feature_key": "auth", "name": "Auth"},
        ],
        existing_stories=[
            {"story_key": "STORY-1", "status": "done",
             "feature_key": "auth",
             "title": "Login", "acceptance_criteria": ["POST /login returns 200"]},
            {"story_key": "STORY-2", "status": "in_progress",
             "feature_key": "auth",
             "title": "Logout", "acceptance_criteria": ["POST /logout returns 204"]},
        ],
        spec_requirements="req", spec_architecture="arch",
        workspace_path="/tmp/ws",
    )
    assert "STORY-1" in p and "STORY-2" in p
    assert "[done]" in p and "[in_progress]" in p
    assert "Login" in p
    assert "augment mode" in p.lower()
    assert "auth" in p


def test_augment_prompt_handles_empty_existing():
    p = decomposition._build_decomposition_augment_prompt(
        existing_features=[], existing_stories=[],
        spec_requirements="req",
        spec_architecture="arch", workspace_path="/tmp/ws",
    )
    assert "_(none)_" in p


def test_augment_validator_accepts_empty_stories():
    """No new stories AND no new features = valid no-op answer."""
    features, stories = decomposition._validate_augment_payload(
        {"features": [], "stories": []},
    )
    assert features == [] and stories == []
    features, stories = decomposition._validate_augment_payload(
        {"features": [], "stories": [], "summary": "no-op"},
    )
    assert features == [] and stories == []


def test_augment_validator_accepts_story_new_placeholders():
    features, stories = decomposition._validate_augment_payload({
        "features": [{"feature_key": "metrics", "name": "Metrics"}],
        "stories": [{
            "story_key": "STORY-NEW-1", "feature": "metrics",
            "title": "Add metrics endpoint",
            "requirement_keys": ["FR-010"],
            "acceptance_criteria": ["GET /metrics returns 200"],
            "depends_on": [], "scope_files": ["src/metrics.py"],
        }],
    })
    assert len(stories) == 1
    assert stories[0]["title"] == "Add metrics endpoint"
    assert stories[0]["feature"] == "metrics"
    assert features[0]["feature_key"] == "metrics"


def test_augment_validator_accepts_story_referencing_existing_feature():
    """Augment mode lets stories reference features already on file."""
    features, stories = decomposition._validate_augment_payload(
        {
            "features": [],
            "stories": [{
                "story_key": "STORY-NEW-1", "feature": "auth",
                "title": "Add MFA",
                "requirement_keys": ["FR-010"],
                "acceptance_criteria": ["MFA enrolment works"],
                "depends_on": [], "scope_files": [],
            }],
        },
        existing_feature_keys={"auth"},
    )
    assert features == []
    assert stories[0]["feature"] == "auth"


def test_augment_validator_rejects_cross_response_forward_dep():
    """Depends-on can only reference placeholders earlier in the same response."""
    with pytest.raises(ValueError, match="depends_on"):
        decomposition._validate_augment_payload({
            "features": [{"feature_key": "x", "name": "X"}],
            "stories": [{
                "story_key": "STORY-NEW-1", "feature": "x", "title": "x",
                "requirement_keys": ["FR-001"],
                "acceptance_criteria": ["x"],
                "depends_on": ["STORY-NEW-2"],  # forward reference
                "scope_files": [],
            }],
        })


def test_decomposition_node_augment_mode_appends_new_story(workspace: str):
    """Workspace already has STORY-1 from a prior agile run. New
    decomposition pass detects the existing row and runs in augment
    mode, appending only the genuinely new story (STORY-2)."""
    from harness.graph import set_gateway

    # Seed an existing feature + story in the DB for this workspace.
    app = story_state.app_name_for_workspace(workspace)
    _seed_story(app, "Original feature",
                acceptance_criteria=["GET /orig returns 200"])

    _write_spec(workspace, "Revised: adds a /new endpoint")
    augment_response = json.dumps({
        "features": [],
        "stories": [{
            "story_key": "STORY-NEW-1",
            "feature": "test",
            "title": "Add /new endpoint",
            "requirement_keys": ["FR-001"],
            "acceptance_criteria": ["GET /new returns 200"],
            "depends_on": [],
            "scope_files": ["src/new.py"],
        }],
        "summary": "one new story",
    })
    gw = _FakeGateway([augment_response])
    set_gateway(gw)

    out = asyncio.run(decomposition.decomposition_node(_build_state(workspace)))

    # Augment mode marker surfaced in node_state
    assert out["node_state"]["augment_mode"] is True
    assert out["node_state"]["augment_existing_count"] == 1
    assert out["node_state"]["story_count"] == 1

    # DB now has both the original AND the new story; the DB allocator
    # assigned the placeholder key to the next-available STORY-N.
    conn = story_state.open_story_db()
    try:
        rows = story_state.list_stories(conn, app)
    finally:
        conn.close()
    assert len(rows) == 2
    titles = sorted(r["title"] for r in rows)
    assert titles == ["Add /new endpoint", "Original feature"]

    # Augment prompt actually went to the LLM
    sent_prompt = gw.calls[0]["messages"][1]["content"]
    assert "augment mode" in sent_prompt.lower()
    assert "Original feature" in sent_prompt


def test_decomposition_node_augment_mode_handles_no_new_stories(workspace: str):
    """LLM returns an empty stories+features list = 'existing set
    already covers everything'. Node skips the DB insert and returns
    cleanly."""
    from harness.graph import set_gateway

    app = story_state.app_name_for_workspace(workspace)
    _seed_story(app, "Already covers it")

    _write_spec(workspace)
    gw = _FakeGateway([json.dumps({
        "features": [], "stories": [], "summary": "no-op",
    })])
    set_gateway(gw)

    out = asyncio.run(decomposition.decomposition_node(_build_state(workspace)))

    assert out["node_state"]["augment_mode"] is True
    assert out["node_state"]["story_count"] == 0
    assert out["node_state"]["story_keys"] == []
    assert out["node_state"]["augment_existing_count"] == 1

    # DB still has just the one pre-existing story.
    conn = story_state.open_story_db()
    try:
        rows = story_state.list_stories(conn, app)
    finally:
        conn.close()
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# Routing — story_reopen_node insertion in route_after_gatekeeper
# ---------------------------------------------------------------------------

def test_route_after_gatekeeper_patch_agile_with_done_stories_goes_to_reopen(workspace: str):
    """ARCHITECTURE gate approval in PATCH flow with an existing DONE
    story routes through story_reopen_node BEFORE decomposition."""
    from harness.graph import (
        route_after_gatekeeper, AgentState, FLOW_PATCH,
    )

    # Seed a DONE story in the DB
    app = story_state.app_name_for_workspace(workspace)
    key = _seed_story(
        app, "Login",
        acceptance_criteria=["POST /login returns 200"],
    )
    conn = story_state.open_story_db()
    try:
        story_state.mark_done(conn, app, key)
    finally:
        conn.close()

    state = AgentState(
        flow=FLOW_PATCH,
        workspace_path=workspace,
        decomposition_enabled=True,
        current_gate="ARCHITECTURE",
        node_state={"gatekeeper_action": "approve"},
    )
    assert route_after_gatekeeper(state) == "story_reopen_node"


def test_route_after_gatekeeper_build_agile_skips_reopen(workspace: str):
    """BUILD flow never routes through story_reopen_node even when
    DONE stories exist (build always starts fresh-ish)."""
    from harness.graph import (
        route_after_gatekeeper, AgentState, FLOW_BUILD,
    )

    app = story_state.app_name_for_workspace(workspace)
    key = _seed_story(app, "Whatever")
    conn = story_state.open_story_db()
    try:
        story_state.mark_done(conn, app, key)
    finally:
        conn.close()

    state = AgentState(
        flow=FLOW_BUILD,
        workspace_path=workspace,
        decomposition_enabled=True,
        current_gate="ARCHITECTURE",
        node_state={"gatekeeper_action": "approve"},
    )
    assert route_after_gatekeeper(state) == "decomposition_node"


def test_route_after_gatekeeper_patch_agile_no_done_skips_reopen(workspace: str):
    """PATCH flow with planned-but-not-DONE stories does NOT trigger reopen."""
    from harness.graph import (
        route_after_gatekeeper, AgentState, FLOW_PATCH,
    )

    app = story_state.app_name_for_workspace(workspace)
    _seed_story(
        app, "Planned-only", acceptance_criteria=["x"],
    )
    # No mark_done — story stays in 'planned' state.

    state = AgentState(
        flow=FLOW_PATCH,
        workspace_path=workspace,
        decomposition_enabled=True,
        current_gate="ARCHITECTURE",
        node_state={"gatekeeper_action": "approve"},
    )
    assert route_after_gatekeeper(state) == "decomposition_node"


# ---------------------------------------------------------------------------
# v5 requirement_keys validation
# ---------------------------------------------------------------------------

class TestRequirementKeysValidation:
    """Phase 2 contract: every story must cite >=1 valid req_key.

    Shape checks (missing / non-list / empty) fire regardless of
    whether ``known_req_keys`` is passed; cross-validation (key
    exists in the spec) only fires when the caller threads the
    known-set in. decomposition_node always does — these tests
    exercise both modes.
    """

    def test_missing_requirement_keys_rejected(self):
        payload = _payload_with_one_feature([{
            "story_key": "STORY-1", "title": "t",
            "acceptance_criteria": ["x"],
        }])
        del payload["stories"][0]["requirement_keys"]
        with pytest.raises(ValueError, match="requirement_keys"):
            decomposition._validate_stories_payload(payload)

    def test_empty_requirement_keys_rejected(self):
        payload = _payload_with_one_feature([{
            "story_key": "STORY-1", "title": "t",
            "acceptance_criteria": ["x"],
        }])
        payload["stories"][0]["requirement_keys"] = []
        with pytest.raises(ValueError, match="at least one"):
            decomposition._validate_stories_payload(payload)

    def test_non_list_requirement_keys_rejected(self):
        payload = _payload_with_one_feature([{
            "story_key": "STORY-1", "title": "t",
            "acceptance_criteria": ["x"],
        }])
        payload["stories"][0]["requirement_keys"] = "FR-001"  # str not list
        with pytest.raises(ValueError, match="requirement_keys"):
            decomposition._validate_stories_payload(payload)

    def test_unknown_key_rejected_with_alternative_listing(self):
        """When known_req_keys is provided, an unknown key fails with
        a message that lists valid alternatives so the operator sees
        the universe of choices without re-opening the spec."""
        payload = _payload_with_one_feature([{
            "story_key": "STORY-1", "title": "t",
            "acceptance_criteria": ["x"],
            "requirement_keys": ["FR-099"],
        }])
        with pytest.raises(ValueError) as excinfo:
            decomposition._validate_stories_payload(
                payload, known_req_keys={"FR-001", "FR-002", "FR-003"},
            )
        msg = str(excinfo.value)
        assert "FR-099" in msg
        assert "FR-001" in msg
        assert "FR-003" in msg

    def test_known_key_accepted(self):
        payload = _payload_with_one_feature([{
            "story_key": "STORY-1", "title": "t",
            "acceptance_criteria": ["x"],
            "requirement_keys": ["FR-001"],
        }])
        features, stories = decomposition._validate_stories_payload(
            payload, known_req_keys={"FR-001"},
        )
        assert stories[0]["requirement_keys"] == ["FR-001"]

    def test_known_keys_capped_at_40_in_error_message(self):
        payload = _payload_with_one_feature([{
            "story_key": "STORY-1", "title": "t",
            "acceptance_criteria": ["x"],
            "requirement_keys": ["FR-9999"],
        }])
        known = {f"FR-{i:04d}" for i in range(100)}
        with pytest.raises(ValueError) as excinfo:
            decomposition._validate_stories_payload(
                payload, known_req_keys=known,
            )
        msg = str(excinfo.value)
        assert "first 40" in msg
        # Sort puts FR-0000 first; FR-0099 lives past index 39 so is excluded.
        assert "FR-0000" in msg
        assert "FR-0099" not in msg

    def test_augment_validator_enforces_requirement_keys(self):
        with pytest.raises(ValueError, match="requirement_keys"):
            decomposition._validate_augment_payload({
                "features": [{"feature_key": "x", "name": "X"}],
                "stories": [{
                    "story_key": "STORY-NEW-1", "feature": "x", "title": "t",
                    "acceptance_criteria": ["x"],
                    "depends_on": [], "scope_files": [],
                }],
            })


# ---------------------------------------------------------------------------
# v5 requirements ingest (parse SPEC_REQUIREMENTS.md -> requirements table)
# ---------------------------------------------------------------------------

class TestRequirementsIngest:
    """Phase 2 helper: ``_ingest_requirements`` UPSERTs spec rows into
    the requirements table before decomposition asks the LLM to cite
    requirement_keys."""

    SPEC = (
        "# Spec\n\n"
        "### FR-001: Login\n"
        "User can log in.\n\n"
        "### FR-002: Logout\n"
        "User can log out.\n\n"
        "#### NFR-SEC-001: Token hashing\n"
        "Tokens MUST be hashed.\n\n"
        "### US-03-02: Reset screen\n"
        "User sees a reset confirmation page.\n"
    )

    def test_ingest_inserts_fr_nfr_us(self, workspace: str):
        app = story_state.app_name_for_workspace(workspace)
        parsed, upserted = decomposition._ingest_requirements(
            workspace, app, self.SPEC,
        )
        assert parsed == 4
        assert upserted == 4
        conn = story_state.open_story_db()
        try:
            keys = {r["req_key"] for r in story_state.list_requirements(conn, app)}
        finally:
            conn.close()
        assert keys == {"FR-001", "FR-002", "NFR-SEC-001", "US-03-02"}

    def test_ingest_idempotent_on_rerun(self, workspace: str):
        app = story_state.app_name_for_workspace(workspace)
        decomposition._ingest_requirements(workspace, app, self.SPEC)
        decomposition._ingest_requirements(workspace, app, self.SPEC)
        conn = story_state.open_story_db()
        try:
            rows = story_state.list_requirements(conn, app)
        finally:
            conn.close()
        assert len(rows) == 4

    def test_ingest_upserts_changed_title(self, workspace: str):
        app = story_state.app_name_for_workspace(workspace)
        decomposition._ingest_requirements(workspace, app, self.SPEC)
        revised = self.SPEC.replace(
            "### FR-001: Login", "### FR-001: Login (revised)",
        )
        decomposition._ingest_requirements(workspace, app, revised)
        conn = story_state.open_story_db()
        try:
            row = story_state.get_requirement_by_key(conn, app, "FR-001")
        finally:
            conn.close()
        assert row["title"] == "Login (revised)"

    def test_ingest_captures_source_line(self, workspace: str):
        app = story_state.app_name_for_workspace(workspace)
        decomposition._ingest_requirements(workspace, app, self.SPEC)
        conn = story_state.open_story_db()
        try:
            row = story_state.get_requirement_by_key(conn, app, "FR-001")
        finally:
            conn.close()
        # 1-indexed line of "### FR-001: Login" within SPEC
        assert row["source_line"] == 3

    def test_ingest_empty_spec_is_noop(self, workspace: str):
        app = story_state.app_name_for_workspace(workspace)
        parsed, upserted = decomposition._ingest_requirements(workspace, app, "")
        assert parsed == 0 and upserted == 0
        conn = story_state.open_story_db()
        try:
            assert story_state.list_requirements(conn, app) == []
        finally:
            conn.close()


