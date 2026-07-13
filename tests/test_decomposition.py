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


_DEFAULT_SPEC_BODY = (
    "# Build a TODO API\n\n"
    "### FR-001: Create a TODO\n"
    "POST /todos creates an item.\n\n"
    "### FR-002: List TODOs\n"
    "GET /todos returns the list.\n"
)


def _write_spec(workspace: str, body: str = _DEFAULT_SPEC_BODY) -> None:
    """Write a SPEC_REQUIREMENTS.md whose FR-NNN headings match the
    requirement_keys used in the canonical ``_valid_payload`` /
    augment fixtures. Without these headings the v5 requirements
    ingest leaves the table empty and the validator rejects the
    fixtures' bogus keys (post-BUG #5 — pre-fix it silently passed)."""
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
                "story_key": "STORY-001",
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
                "story_key": "STORY-002",
                "feature": "core",
                "title": "List TODOs",
                "description": "GET /todos returns the list.",
                "requirement_keys": ["FR-002"],
                "acceptance_criteria": ["GET /todos returns JSON array"],
                "depends_on": ["STORY-001"],
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
    assert stories[1]["depends_on"] == ["STORY-001"]


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
        "story_key": "STORY-001", "feature": "core",
        "title": "t", "acceptance_criteria": ["x"],
    }]}
    with pytest.raises(ValueError, match="features"):
        decomposition._validate_stories_payload(payload)


def test_validate_rejects_story_referencing_undeclared_feature():
    payload = {
        "features": [{"feature_key": "core", "name": "Core"}],
        "stories": [{
            "story_key": "STORY-001", "feature": "ghost",
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
            "story_key": "STORY-001", "feature": "core",
            "title": "t", "acceptance_criteria": ["x"],
            "requirement_keys": ["FR-001"],
        }],
    }
    with pytest.raises(ValueError, match="orphan"):
        decomposition._validate_stories_payload(payload)


def test_validate_rejects_bad_story_key():
    payload = _payload_with_one_feature([{
        "story_key": "ABC-1", "title": "t", "acceptance_criteria": ["x"]
    }])
    with pytest.raises(ValueError, match="invalid story_key"):
        decomposition._validate_stories_payload(payload)


def test_validate_rejects_missing_acceptance():
    payload = _payload_with_one_feature([{
        "story_key": "STORY-001", "title": "t", "acceptance_criteria": []
    }])
    with pytest.raises(ValueError, match="acceptance"):
        decomposition._validate_stories_payload(payload)


def test_validate_accepts_forward_dependency():
    """LLMs commonly emit stories in feature-grouping order, not topological
    order. A forward depends_on is acyclic and therefore safe; the validator
    must accept it (the runtime planner gates on deps being 'done')."""
    payload = _payload_with_one_feature([
        {"story_key": "STORY-001", "title": "a", "requirement_keys": ["FR-001"],
         "acceptance_criteria": ["x"], "depends_on": ["STORY-002"]},
        {"story_key": "STORY-002", "title": "b", "requirement_keys": ["FR-001"],
         "acceptance_criteria": ["y"]},
    ])
    _, stories = decomposition._validate_stories_payload(
        payload, known_req_keys={"FR-001"},
    )
    assert stories[0]["depends_on"] == ["STORY-002"]


def test_validate_rejects_unknown_dependency_target():
    payload = _payload_with_one_feature([
        {"story_key": "STORY-001", "title": "a", "requirement_keys": ["FR-001"],
         "acceptance_criteria": ["x"], "depends_on": ["STORY-099"]},
    ])
    with pytest.raises(ValueError, match="STORY-099"):
        decomposition._validate_stories_payload(
            payload, known_req_keys={"FR-001"},
        )


def test_validate_rejects_dependency_cycle():
    payload = _payload_with_one_feature([
        {"story_key": "STORY-001", "title": "a", "requirement_keys": ["FR-001"],
         "acceptance_criteria": ["x"], "depends_on": ["STORY-002"]},
        {"story_key": "STORY-002", "title": "b", "requirement_keys": ["FR-001"],
         "acceptance_criteria": ["y"], "depends_on": ["STORY-001"]},
    ])
    with pytest.raises(ValueError, match="cycle"):
        decomposition._validate_stories_payload(
            payload, known_req_keys={"FR-001"},
        )


def test_validate_rejects_duplicate_keys():
    payload = _payload_with_one_feature([
        {"story_key": "STORY-001", "title": "a", "acceptance_criteria": ["x"]},
        {"story_key": "STORY-001", "title": "b", "acceptance_criteria": ["y"]},
    ])
    with pytest.raises(ValueError, match="duplicate"):
        decomposition._validate_stories_payload(payload)


# ---------------------------------------------------------------------------
# Fence stripping
# ---------------------------------------------------------------------------

def test_scope_files_js_under_frontend_root_rewritten_to_tsx(caplog):
    payload = _payload_with_one_feature([
        {
            "story_key": "STORY-001",
            # Title / ACs cover every scope_files entry's domain word so
            # the B2 cross-domain guard keeps them all — this test is
            # about the JS→TSX rewrite, not the domain check.
            "title": "Add SearchBar and Home page",
            "acceptance_criteria": [
                "SearchBar renders", "Home page renders",
                "search API returns results",
                "jest and webpack config compile",
            ],
            "scope_files": [
                "client/src/components/SearchBar.js",
                "frontend/src/pages/Home.jsx",
                "server/api/search.py",
                "jest.config.js",
                "webpack.config.js",
            ],
        },
    ])
    caplog.set_level("WARNING", logger="harness.decomposition")
    _, stories = decomposition._validate_stories_payload(payload)
    assert stories[0]["scope_files"] == [
        "client/src/components/SearchBar.tsx",
        "frontend/src/pages/Home.tsx",
        "server/api/search.py",
        "jest.config.js",
        "webpack.config.js",
    ]
    warned = [r for r in caplog.records if "stack-enforce" in r.getMessage()]
    assert len(warned) == 2, warned


def test_scope_files_tsx_untouched():
    payload = _payload_with_one_feature([
        {
            "story_key": "STORY-001",
            "title": "SearchBar in TypeScript",
            "acceptance_criteria": ["SearchBar renders"],
            "scope_files": ["client/src/components/SearchBar.tsx"],
        },
    ])
    _, stories = decomposition._validate_stories_payload(payload)
    assert stories[0]["scope_files"] == ["client/src/components/SearchBar.tsx"]


def test_scope_files_monorepo_marker_rewritten():
    payload = _payload_with_one_feature([
        {
            "story_key": "STORY-001",
            "title": "Foo component in monorepo",
            "acceptance_criteria": ["Foo renders"],
            "scope_files": ["packages/web/src/components/Foo.js"],
        },
    ])
    _, stories = decomposition._validate_stories_payload(payload)
    assert stories[0]["scope_files"] == ["packages/web/src/components/Foo.tsx"]


# ---------------------------------------------------------------------------
# B2 — cross-domain scope_files guard (finsearch session 44c5e194)
# ---------------------------------------------------------------------------

class TestCrossDomainScopeGuard:
    """Finsearch session 44c5e194 root cause B2: planner assigned
    STORY-032 "Source Traceability" (feature "PDF & CSV Export") to
    ``server/services/forecast.py`` — a file that neither exists nor
    shares any domain word with the story or feature. The deterministic
    guard filters cross-domain entries at planning time so the patcher
    never operates on hallucinated scope."""

    def test_drops_cross_domain_entry(self, caplog):
        # The finsearch pattern, exactly: story about traceability,
        # scope pointing at forecast.
        payload = _payload_with_one_feature([{
            "story_key": "STORY-001",
            "title": "Source Traceability",
            "acceptance_criteria": [
                "Chart footnotes cite the source filing",
                "CSV export includes source_url column",
            ],
            "scope_files": [
                "server/services/forecast.py",
                "server/exporters/csv.py",
            ],
        }])
        caplog.set_level("WARNING", logger="harness.decomposition")
        _, stories = decomposition._validate_stories_payload(payload)
        # forecast.py dropped (no shared domain word); csv.py kept
        # (shares "csv" with the acceptance criterion).
        assert stories[0]["scope_files"] == ["server/exporters/csv.py"]
        warned = [
            r for r in caplog.records if "cross-domain drop" in r.getMessage()
        ]
        assert any("forecast.py" in r.getMessage() for r in warned)


class TestCrossTestRootScopeGuard:
    """Finsearch session 156032347 root cause: LLM emitted
    ``server/app/services/tests/test_filing_service.py`` on one story
    while ``server/app/tests/test_filing_service.py`` was already in
    another story's scope_files. Two mirrored test trees for the same
    module trap the repair loop in a REPLACE_BLOCK oscillation on the
    shared implementation file. The patcher's DUPLICATE_TEST_ROOT guard
    catches this at land time — but by then a codegen round is spent.
    The decomposition guard pre-filters, saving the round."""

    def test_drops_deeper_test_root_when_shallower_exists(self, caplog):
        # Both stories name a test file with the same test-scoped
        # suffix (`tests/test_filing_service.py`) under DIFFERENT test
        # roots. The deeper (services/tests/) one is dropped; the
        # shallower (tests/) one is kept.
        payload = _payload_with_one_feature([
            {
                "story_key": "STORY-001",
                "title": "Filing service canonical",
                "acceptance_criteria": [
                    "Filing service parses EDGAR filings",
                ],
                "scope_files": [
                    "server/app/services/filing_service.py",
                    "server/app/tests/test_filing_service.py",
                ],
            },
            {
                "story_key": "STORY-002",
                "title": "Filing edge cases",
                "acceptance_criteria": [
                    "Filing service handles restatements",
                ],
                "scope_files": [
                    "server/app/services/tests/test_filing_service.py",
                ],
            },
        ])
        caplog.set_level("WARNING", logger="harness.decomposition")
        _, stories = decomposition._validate_stories_payload(payload)
        by_key = {s["title"]: s for s in stories}
        # STORY-001 (shallower root) keeps its test file.
        assert (
            "server/app/tests/test_filing_service.py"
            in by_key["Filing service canonical"]["scope_files"]
        )
        # STORY-002 (deeper root) lost its cross-root duplicate.
        assert (
            "server/app/services/tests/test_filing_service.py"
            not in by_key["Filing edge cases"]["scope_files"]
        )
        warned = [
            r for r in caplog.records
            if "cross-test-root drop" in r.getMessage()
        ]
        assert any(
            "server/app/services/tests/test_filing_service.py" in
            r.getMessage() for r in warned
        )

    def test_does_not_touch_stories_with_different_test_suffixes(self):
        # Two stories, both with test files, but DIFFERENT basenames.
        # Neither collides — both scope_files entries survive.
        payload = _payload_with_one_feature([
            {
                "story_key": "STORY-001",
                "title": "Filing service tests",
                "acceptance_criteria": ["Filings parsed"],
                "scope_files": [
                    "server/app/tests/test_filing_service.py",
                ],
            },
            {
                "story_key": "STORY-002",
                "title": "Company service tests",
                "acceptance_criteria": ["Companies searched"],
                "scope_files": [
                    "server/app/tests/test_company_service.py",
                ],
            },
        ])
        _, stories = decomposition._validate_stories_payload(payload)
        assert stories[0]["scope_files"] == [
            "server/app/tests/test_filing_service.py",
        ]
        assert stories[1]["scope_files"] == [
            "server/app/tests/test_company_service.py",
        ]

    def test_non_test_basenames_are_ignored(self):
        # Two stories, same non-test basename under different roots.
        # Guard MUST NOT fire — this is legitimate (e.g. two __init__.py
        # files across sibling packages).
        payload = _payload_with_one_feature([
            {
                "story_key": "STORY-001",
                "title": "Auth package init",
                "acceptance_criteria": ["Auth exports registered"],
                "scope_files": ["server/app/auth/__init__.py"],
            },
            {
                "story_key": "STORY-002",
                "title": "Billing package init",
                "acceptance_criteria": ["Billing exports registered"],
                "scope_files": ["server/app/billing/__init__.py"],
            },
        ])
        _, stories = decomposition._validate_stories_payload(payload)
        assert stories[0]["scope_files"] == [
            "server/app/auth/__init__.py",
        ]
        assert stories[1]["scope_files"] == [
            "server/app/billing/__init__.py",
        ]

    def test_keeps_entry_matching_story_title(self):
        payload = _payload_with_one_feature([{
            "story_key": "STORY-001",
            "title": "User can register with email",
            "acceptance_criteria": ["POST /register returns 201"],
            "scope_files": [
                "src/auth/register.py",
                "tests/test_register.py",
            ],
        }])
        _, stories = decomposition._validate_stories_payload(payload)
        # Both entries share "register" with title/ACs.
        assert stories[0]["scope_files"] == [
            "src/auth/register.py", "tests/test_register.py",
        ]

    def test_keeps_entry_matching_feature_name(self):
        # Story title / ACs don't mention "billing" but the feature
        # name does — that's enough context.
        payload = {
            "features": [{
                "feature_key": "billing",
                "name": "Subscription billing",
                "description": "Stripe integration",
            }],
            "stories": [{
                "story_key": "STORY-001",
                "feature": "billing",
                "title": "Charge card",
                "requirement_keys": ["FR-001"],
                "acceptance_criteria": ["stripe.Charge.create returns id"],
                "scope_files": ["src/billing/charge.py"],
            }],
        }
        _, stories = decomposition._validate_stories_payload(payload)
        assert stories[0]["scope_files"] == ["src/billing/charge.py"]

    def test_keeps_entry_matching_camelcase_component(self):
        # Path has `SearchBar.tsx` (camelCase); title says
        # "SearchBar" — both should tokenize to {search, bar}.
        payload = _payload_with_one_feature([{
            "story_key": "STORY-001",
            "title": "Add SearchBar component",
            "acceptance_criteria": ["SearchBar renders"],
            "scope_files": ["client/src/components/SearchBar.tsx"],
        }])
        _, stories = decomposition._validate_stories_payload(payload)
        assert stories[0]["scope_files"] == [
            "client/src/components/SearchBar.tsx",
        ]

    def test_short_title_falls_through(self):
        # A very short story title with no non-generic words
        # (e.g. "Login") means we can't build a context — trust the
        # planner rather than nuke every entry.
        payload = _payload_with_one_feature([{
            "story_key": "STORY-001",
            "title": "GO",  # 2-char, filtered as too short
            "acceptance_criteria": ["ok"],
            "scope_files": ["src/anything/at_all.py"],
        }])
        _, stories = decomposition._validate_stories_payload(payload)
        # Fall-through — nothing to compare against, keep everything.
        assert stories[0]["scope_files"] == ["src/anything/at_all.py"]


class TestScopePathTokens:
    def test_extracts_domain_word_and_filters_generic(self):
        # server/services/*/py: all generic; "forecast" is the domain word.
        assert decomposition._scope_path_tokens(
            "server/services/forecast.py"
        ) == {"forecast"}

    def test_splits_camel_case(self):
        assert decomposition._scope_path_tokens(
            "client/src/components/SearchBar.tsx"
        ) == {"search", "bar"}

    def test_splits_snake_and_hyphen(self):
        assert decomposition._scope_path_tokens(
            "tests/test_source_traceability.py"
        ) == {"source", "traceability"}

    def test_drops_short_tokens(self):
        # "db", "id", "js" are all < 3 chars → dropped.
        assert decomposition._scope_path_tokens(
            "src/db/id.js"
        ) == set()


class TestContextTokens:
    def test_extracts_from_title_and_camel(self):
        assert decomposition._context_tokens(
            "Add SearchBar",
        ) == {"add", "search", "bar"}

    def test_flattens_acceptance_criteria_list(self):
        assert decomposition._context_tokens(
            "T", ["metric returns 200", "metric is TTM"],
        ) == {"metric", "returns", "200", "ttm"}


class TestWorkspaceFileTreeHint:
    def test_returns_empty_for_missing_dir(self, tmp_path):
        assert decomposition._build_workspace_file_tree_hint(
            str(tmp_path / "does-not-exist"),
        ) == ""

    def test_returns_empty_for_empty_workspace(self, tmp_path):
        assert decomposition._build_workspace_file_tree_hint(str(tmp_path)) == ""

    def test_lists_real_files_alphabetically(self, tmp_path):
        (tmp_path / "server").mkdir()
        (tmp_path / "server" / "main.py").write_text("x\n")
        (tmp_path / "client").mkdir()
        (tmp_path / "client" / "App.tsx").write_text("x\n")
        out = decomposition._build_workspace_file_tree_hint(str(tmp_path))
        assert "## Current workspace file tree" in out
        assert "- client/App.tsx" in out
        assert "- server/main.py" in out
        # Sort order: client < server
        assert out.index("client/App.tsx") < out.index("server/main.py")

    def test_prunes_noise_directories(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "keep.py").write_text("x\n")
        for noise in ("node_modules", "__pycache__", ".git", "dist"):
            (tmp_path / noise).mkdir()
            (tmp_path / noise / "junk.py").write_text("y\n")
        out = decomposition._build_workspace_file_tree_hint(str(tmp_path))
        assert "keep.py" in out
        assert "node_modules" not in out
        assert "__pycache__" not in out
        assert ".git" not in out
        assert "dist" not in out

    def test_caps_at_max_files_with_footer(self, tmp_path):
        for i in range(15):
            (tmp_path / f"f{i:02}.py").write_text("x\n")
        out = decomposition._build_workspace_file_tree_hint(
            str(tmp_path), max_files=5,
        )
        # 10 hidden → footer says (+10 more not shown)
        assert "10 more not shown" in out
        # Only the first 5 alphabetical entries appear.
        assert "- f00.py" in out
        assert "- f04.py" in out
        assert "- f05.py" not in out


class TestB2PromptEdits:
    """The prompt edits themselves (#1, #2, #3): verify the shipped
    prompt strings carry the new language, so a doc-review or prompt
    regression would catch a re-introduction of the old wording."""

    def test_initial_prompt_says_empty_is_default(self):
        p = decomposition._build_decomposition_prompt(
            "# spec", "", "/tmp/ws", known_req_keys={"FR-001"},
        )
        assert "Empty ``scope_files`` (``[]``) is the right default" in p

    def test_initial_prompt_names_domain_consistency_rule(self):
        p = decomposition._build_decomposition_prompt(
            "# spec", "", "/tmp/ws", known_req_keys={"FR-001"},
        )
        assert "MUST share a domain word" in p
        # The finsearch example is called out by name so future edits
        # to the prompt keep the concrete illustration.
        assert "Source Traceability" in p
        assert "forecast.py" in p

    def test_initial_prompt_includes_empty_example(self):
        p = decomposition._build_decomposition_prompt(
            "# spec", "", "/tmp/ws", known_req_keys={"FR-001"},
        )
        assert '"scope_files": []' in p

    def test_augment_prompt_says_empty_is_default(self):
        p = decomposition._build_decomposition_augment_prompt(
            existing_features=[], existing_stories=[],
            spec_requirements="# spec", spec_architecture="",
            workspace_path="/tmp/ws-empty",
            known_req_keys={"FR-001"},
        )
        assert "empty ``[]`` is the right default" in p

    def test_augment_prompt_includes_workspace_tree_when_present(
        self, tmp_path,
    ):
        (tmp_path / "server").mkdir()
        (tmp_path / "server" / "auth.py").write_text("x\n")
        p = decomposition._build_decomposition_augment_prompt(
            existing_features=[], existing_stories=[],
            spec_requirements="# spec", spec_architecture="",
            workspace_path=str(tmp_path),
            known_req_keys={"FR-001"},
        )
        assert "## Current workspace file tree" in p
        assert "server/auth.py" in p


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
    assert out["node_state"]["story_keys"] == ["STORY-001", "STORY-002"]
    assert out["stories_db_path"].endswith("state.db")
    assert out["budget_remaining_usd"] == 1.50

    # DB has the stories
    app = story_state.app_name_for_workspace(workspace)
    conn = story_state.open_story_db()
    try:
        stories = story_state.list_stories(conn, app)
    finally:
        conn.close()
    assert [s["story_key"] for s in stories] == ["STORY-001", "STORY-002"]
    assert stories[1]["depends_on"] == ["STORY-001"]

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


def _cyclic_payload() -> str:
    """Same shape as _valid_payload but with a STORY-001 ↔ STORY-002 cycle."""
    p = json.loads(_valid_payload())
    p["stories"][0]["depends_on"] = ["STORY-002"]
    p["stories"][1]["depends_on"] = ["STORY-001"]
    return json.dumps(p)


def test_decomposition_node_cycle_auto_repairs(workspace: str):
    """Cycle in the first response → 1-shot repair → commit succeeds."""
    from harness.graph import set_gateway
    _write_spec(workspace)
    gw = _FakeGateway([_cyclic_payload(), _valid_payload()])
    set_gateway(gw)

    out = asyncio.run(decomposition.decomposition_node(_build_state(workspace)))

    assert out["current_gate"] == "STORIES"
    assert out["node_state"]["decomposition_complete"] is True
    assert out["node_state"]["story_count"] == 2
    # Two LLM calls: original + one repair attempt
    assert len(gw.calls) == 2
    # Repair prompt must reference the cycle path
    repair_msg = gw.calls[1]["messages"][-1]["content"]
    assert "depends_on cycle detected" in repair_msg
    assert "STORY-001" in repair_msg and "STORY-002" in repair_msg


def test_decomposition_node_cycle_repair_failure_routes_to_hitl(workspace: str):
    """Repair attempt still cyclic → HITL with both errors in message."""
    from harness.graph import set_gateway
    _write_spec(workspace)
    gw = _FakeGateway([_cyclic_payload(), _cyclic_payload()])
    set_gateway(gw)

    out = asyncio.run(decomposition.decomposition_node(_build_state(workspace)))

    assert out["node_state"]["decomposition_failed"] is True
    err = out["node_state"]["error"]
    assert err.startswith("validation:")
    assert "repair_failed" in err
    assert len(gw.calls) == 2


def test_decomposition_node_cycle_repair_skipped_when_budget_zero(workspace: str):
    """Budget exhausted after the first call → no repair attempted."""
    from harness.graph import set_gateway
    _write_spec(workspace)
    # First call drains budget to 0.0 — repair branch must skip.
    gw = _FakeGateway([_cyclic_payload()], budget_after=0.0)
    set_gateway(gw)

    out = asyncio.run(decomposition.decomposition_node(_build_state(workspace)))

    assert out["node_state"]["decomposition_failed"] is True
    assert out["node_state"]["error"].startswith("validation:")
    assert "repair_failed" not in out["node_state"]["error"]
    assert len(gw.calls) == 1


def test_decomposition_node_dispatch_exception(workspace: str):
    from harness.graph import set_gateway
    _write_spec(workspace)
    set_gateway(_FakeGateway([], raise_on_call=RuntimeError("upstream 503")))
    out = asyncio.run(decomposition.decomposition_node(_build_state(workspace)))
    assert out["node_state"]["error"].startswith("dispatch_failed")
    assert "upstream 503" in out["node_state"]["error"]


# ---------------------------------------------------------------------------
# Unknown-requirement_key auto-repair — same 1-shot contract as cycles
# ---------------------------------------------------------------------------

def _payload_with_bogus_req_key() -> str:
    """A payload whose STORY-001 cites a suffix-extrapolated key (``FR-001B``)
    that the workspace spec does not declare. Matches the failure mode
    observed in session 5e0552bc where the LLM emitted ``STORY-011B``.
    """
    p = json.loads(_valid_payload())
    p["stories"][0]["requirement_keys"] = ["FR-001B"]
    return json.dumps(p)


def test_decomposition_node_unknown_req_key_auto_repairs(workspace: str):
    """Bogus req_key in the first response → 1-shot repair → commit."""
    from harness.graph import set_gateway
    _write_spec(workspace)
    gw = _FakeGateway([_payload_with_bogus_req_key(), _valid_payload()])
    set_gateway(gw)

    out = asyncio.run(decomposition.decomposition_node(_build_state(workspace)))

    assert out["current_gate"] == "STORIES"
    assert out["node_state"]["decomposition_complete"] is True
    assert out["node_state"]["story_count"] == 2
    assert len(gw.calls) == 2
    repair_msg = gw.calls[1]["messages"][-1]["content"]
    assert "cites unknown requirement_keys" in repair_msg
    assert "FR-001B" in repair_msg
    # Repair prompt must list the workspace's valid alternatives so the
    # planner can swap in-vocabulary in one turn.
    assert "FR-001" in repair_msg and "FR-002" in repair_msg


def test_decomposition_node_unknown_req_key_repair_failure_routes_to_hitl(
    workspace: str,
):
    """Repair attempt still cites unknown key → HITL with both errors."""
    from harness.graph import set_gateway
    _write_spec(workspace)
    gw = _FakeGateway([
        _payload_with_bogus_req_key(),
        _payload_with_bogus_req_key(),
    ])
    set_gateway(gw)

    out = asyncio.run(decomposition.decomposition_node(_build_state(workspace)))

    assert out["node_state"]["decomposition_failed"] is True
    err = out["node_state"]["error"]
    assert err.startswith("validation:")
    assert "repair_failed" in err
    assert len(gw.calls) == 2


def test_decomposition_node_unknown_req_key_repair_skipped_when_budget_zero(
    workspace: str,
):
    """Budget exhausted after first call → no repair attempted."""
    from harness.graph import set_gateway
    _write_spec(workspace)
    gw = _FakeGateway([_payload_with_bogus_req_key()], budget_after=0.0)
    set_gateway(gw)

    out = asyncio.run(decomposition.decomposition_node(_build_state(workspace)))

    assert out["node_state"]["decomposition_failed"] is True
    assert out["node_state"]["error"].startswith("validation:")
    assert "repair_failed" not in out["node_state"]["error"]
    assert len(gw.calls) == 1


# ---------------------------------------------------------------------------
# Too-many-features auto-repair — same 1-shot contract as cycles/unknown-keys
# ---------------------------------------------------------------------------

def _payload_with_too_many_features() -> str:
    """A payload that trips MAX_FEATURES_PER_PASS with one-line-item
    features (finsearch session finsearch-agile-1783819612 hit exactly
    this: 23 features against the 8-cap)."""
    p = json.loads(_valid_payload())
    # Emit MAX_FEATURES_PER_PASS + 1 features so we're strictly over cap.
    overflow_count = decomposition.MAX_FEATURES_PER_PASS + 1
    p["features"] = [
        {"feature_key": f"feat-{i:02d}", "name": f"Feature {i:02d}",
         "description": "line item"}
        for i in range(overflow_count)
    ]
    return json.dumps(p)


def test_decomposition_node_too_many_features_auto_repairs(workspace: str):
    """Too many features in the first response → 1-shot repair → commit."""
    from harness.graph import set_gateway
    _write_spec(workspace)
    gw = _FakeGateway([_payload_with_too_many_features(), _valid_payload()])
    set_gateway(gw)

    out = asyncio.run(decomposition.decomposition_node(_build_state(workspace)))

    assert out["current_gate"] == "STORIES"
    assert out["node_state"]["decomposition_complete"] is True
    assert out["node_state"]["story_count"] == 2
    assert len(gw.calls) == 2
    repair_msg = gw.calls[1]["messages"][-1]["content"]
    assert "too many features" in repair_msg
    # The repair prompt must name the cap so the LLM knows the target.
    assert str(decomposition.MAX_FEATURES_PER_PASS) in repair_msg
    # And it must instruct feature-level merging (not story dropping).
    assert "merging" in repair_msg or "merge" in repair_msg


def test_decomposition_node_too_many_features_repair_failure_routes_to_hitl(
    workspace: str,
):
    """Repair attempt still over-cap → HITL with both errors in message."""
    from harness.graph import set_gateway
    _write_spec(workspace)
    gw = _FakeGateway([
        _payload_with_too_many_features(),
        _payload_with_too_many_features(),
    ])
    set_gateway(gw)

    out = asyncio.run(decomposition.decomposition_node(_build_state(workspace)))

    assert out["node_state"]["decomposition_failed"] is True
    err = out["node_state"]["error"]
    assert err.startswith("validation:")
    assert "repair_failed" in err
    assert len(gw.calls) == 2


def test_decomposition_node_too_many_features_repair_skipped_when_budget_zero(
    workspace: str,
):
    """Budget exhausted after first call → no repair attempted."""
    from harness.graph import set_gateway
    _write_spec(workspace)
    gw = _FakeGateway([_payload_with_too_many_features()], budget_after=0.0)
    set_gateway(gw)

    out = asyncio.run(decomposition.decomposition_node(_build_state(workspace)))

    assert out["node_state"]["decomposition_failed"] is True
    assert out["node_state"]["error"].startswith("validation:")
    assert "repair_failed" not in out["node_state"]["error"]
    assert len(gw.calls) == 1


def test_prompt_constraint_forbids_suffix_extrapolation():
    """Part 2 defense-in-depth: the planner prompt's Constraints block
    must explicitly warn against extrapolating listed keys with
    suffixes/decimals. Cheap prompt-level guard that complements the
    validator's after-the-fact rejection."""
    prompt = decomposition._build_decomposition_prompt(
        spec_requirements="body", spec_architecture="",
        workspace_path="/tmp/ws",
        known_req_keys={"FR-001", "FR-002"},
    )
    assert "Do NOT append suffixes" in prompt
    # Sanity: both the sample keys embedded in the constraint block.
    assert "FR-001" in prompt and "FR-002" in prompt


# ---------------------------------------------------------------------------
# Augment mode — delta-only decomposition on workspaces with existing stories
# ---------------------------------------------------------------------------

def test_augment_prompt_includes_existing_stories():
    p = decomposition._build_decomposition_augment_prompt(
        existing_features=[
            {"feature_key": "auth", "name": "Auth"},
        ],
        existing_stories=[
            {"story_key": "STORY-001", "status": "done",
             "feature_key": "auth",
             "title": "Login", "acceptance_criteria": ["POST /login returns 200"]},
            {"story_key": "STORY-002", "status": "in_progress",
             "feature_key": "auth",
             "title": "Logout", "acceptance_criteria": ["POST /logout returns 204"]},
        ],
        spec_requirements="req", spec_architecture="arch",
        workspace_path="/tmp/ws",
    )
    assert "STORY-001" in p and "STORY-002" in p
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
    """Workspace already has STORY-001 from a prior agile run. New
    decomposition pass detects the existing row and runs in augment
    mode, appending only the genuinely new story (STORY-002)."""
    from harness.graph import set_gateway

    # Seed an existing feature + story in the DB for this workspace.
    app = story_state.app_name_for_workspace(workspace)
    _seed_story(app, "Original feature",
                acceptance_criteria=["GET /orig returns 200"])

    # Spec must declare FR-001 so the v5 ingest seeds the requirements
    # table; otherwise the validator rejects the augment-stories'
    # requirement_keys cite as "unknown" (BUG #5 contract).
    _write_spec(
        workspace,
        "# Spec\n\n### FR-001: New endpoint\nGET /new returns 200.\n",
    )
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

    def test_empty_known_set_still_rejects_bogus_key(self):
        """Phase 7 BUG #5 regression: an empty set of known req_keys
        must STILL cause unknown-key validation to fire. Pre-fix,
        ``known_req_keys or None`` collapsed an empty set to None,
        dropping the validator into shape-only mode and accepting any
        string — so a workspace whose spec had no FR/NFR/US headings
        silently let bogus keys through and the end-of-session audit
        passed vacuously."""
        payload = _payload_with_one_feature([{
            "story_key": "STORY-001", "title": "t",
            "acceptance_criteria": ["x"],
            "requirement_keys": ["FR-001"],  # bogus — empty known set
        }])
        with pytest.raises(ValueError, match="unknown requirement_keys"):
            decomposition._validate_stories_payload(
                payload, known_req_keys=set(),
            )

    def test_missing_requirement_keys_rejected(self):
        payload = _payload_with_one_feature([{
            "story_key": "STORY-001", "title": "t",
            "acceptance_criteria": ["x"],
        }])
        del payload["stories"][0]["requirement_keys"]
        with pytest.raises(ValueError, match="requirement_keys"):
            decomposition._validate_stories_payload(payload)

    def test_empty_requirement_keys_rejected(self):
        payload = _payload_with_one_feature([{
            "story_key": "STORY-001", "title": "t",
            "acceptance_criteria": ["x"],
        }])
        payload["stories"][0]["requirement_keys"] = []
        with pytest.raises(ValueError, match="at least one"):
            decomposition._validate_stories_payload(payload)

    def test_non_list_requirement_keys_rejected(self):
        payload = _payload_with_one_feature([{
            "story_key": "STORY-001", "title": "t",
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
            "story_key": "STORY-001", "title": "t",
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
            "story_key": "STORY-001", "title": "t",
            "acceptance_criteria": ["x"],
            "requirement_keys": ["FR-001"],
        }])
        features, stories = decomposition._validate_stories_payload(
            payload, known_req_keys={"FR-001"},
        )
        assert stories[0]["requirement_keys"] == ["FR-001"]

    def test_known_keys_capped_at_40_in_error_message(self):
        payload = _payload_with_one_feature([{
            "story_key": "STORY-001", "title": "t",
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
        # Sort puts FR-000 first; FR-099 lives past index 39 so is excluded.
        assert "FR-000" in msg
        assert "FR-099" not in msg

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

    def test_unicode_hyphen_in_llm_emitted_key_matches_ascii_known_set(self):
        # LLM sometimes echoes back the spec's non-breaking hyphen in
        # its requirement_keys payload. Validator normalises before
        # set-membership comparison so ``STORY‑001`` (U+2011) matches
        # the canonical ``STORY-001`` sitting in ``known_req_keys``.
        known = {"STORY-001", "STORY-002"}
        keys = decomposition._validate_story_requirement_keys(
            "STORY-007", ["STORY‑001"], known_req_keys=known,
        )
        assert keys == ["STORY-001"]

    def test_short_digit_llm_key_canonicalises_to_padded_known_key(self):
        # LLM emits ``STORY-001`` when the spec (and DB) hold the padded
        # ``STORY-001``. Validator canonicalises both sides so the
        # citation matches without a repair round-trip.
        known = {"STORY-001", "FR-007", "EPIC-002"}
        assert decomposition._validate_story_requirement_keys(
            "STORY-007", ["STORY-001"], known_req_keys=known,
        ) == ["STORY-001"]
        assert decomposition._validate_story_requirement_keys(
            "STORY-008", ["FR-007", "EPIC-002"], known_req_keys=known,
        ) == ["FR-007", "EPIC-002"]


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


# ---------------------------------------------------------------------------
# Agile-vs-waterfall vocabulary branching in the planner prompt.
#
# Regression guard for a HITL where the planner prompt hardcoded
# waterfall identifiers (FR-NNN / NFR-XXX-NNN / US-NN-NN) even on agile
# workspaces, causing the LLM to invent TEST-NNN keys when the spec
# actually used EPIC/FEAT/STORY/STORY-NFR. The fix routes the actual
# ``known_req_keys`` snapshot into the prompt so the LLM sees ground
# truth instead of a vocabulary hint that may not match.
# ---------------------------------------------------------------------------


def test_format_guidance_agile_workspace_embeds_safe_keys_only():
    agile_keys = {
        "EPIC-001", "FEAT-001", "FEAT-002",
        "STORY-001", "STORY-002", "STORY-NFR-001",
    }
    example, constraint = decomposition._format_requirement_keys_guidance(
        agile_keys
    )
    # Example uses a real agile key the LLM can copy.
    assert "EPIC-001" in example
    assert "FR-" not in example and "NFR-" not in example and "US-" not in example
    # Constraint embeds the workspace's actual identifier list.
    for key in agile_keys:
        assert f"``{key}``" in constraint
    # No waterfall vocabulary leaks through.
    assert "FR-NNN" not in constraint
    assert "NFR-XXX-NNN" not in constraint
    assert "US-NN-NN" not in constraint


def test_format_guidance_waterfall_workspace_embeds_iso_keys_only():
    wf_keys = {"FR-001", "FR-002", "NFR-SEC-001", "US-03-02"}
    example, constraint = decomposition._format_requirement_keys_guidance(
        wf_keys
    )
    assert "FR-001" in example
    assert "EPIC-" not in example and "FEAT-" not in example
    for key in wf_keys:
        assert f"``{key}``" in constraint
    # No agile vocabulary leaks through.
    assert "EPIC-NNN" not in constraint
    assert "FEAT-NNN" not in constraint
    assert "STORY-NFR-NNN" not in constraint


def test_format_guidance_empty_falls_back_to_generic_pointer():
    """When the requirements ingest produced no headings (empty spec,
    parser miss), the prompt must still be valid — fall back to a
    generic pointer at SPEC_REQUIREMENTS.md rather than dictating
    either vocabulary."""
    for empty in (None, set()):
        example, constraint = decomposition._format_requirement_keys_guidance(
            empty
        )
        assert "<one valid req_key>" in example
        assert "docs/SPEC_REQUIREMENTS.md" in constraint
        # No hardcoded family hint in either direction.
        assert "FR-NNN" not in constraint and "EPIC-NNN" not in constraint


def test_format_guidance_caps_embedded_list_for_large_workspaces():
    """Token-budget guard: very large specs cap the embedded list. The
    validator still knows every key, so this is purely a prompt-size
    safeguard."""
    big = {f"FR-{i:03d}" for i in range(200)}
    _, constraint = decomposition._format_requirement_keys_guidance(big)
    embedded_count = constraint.count("``FR-")
    assert embedded_count <= decomposition._REQ_KEY_LIST_CAP
    assert "the validator knows all of them" in constraint


def test_build_decomposition_prompt_no_fr_leak_on_agile_workspace():
    """End-to-end: the full planner prompt rendered for an agile
    workspace must not contain the literal waterfall family hint
    anywhere — neither in the example block nor in the constraints."""
    agile_keys = {"EPIC-001", "FEAT-001", "STORY-001", "STORY-NFR-001"}
    prompt = decomposition._build_decomposition_prompt(
        "## EPIC-001: Auth\n", "", "/tmp/ws", known_req_keys=agile_keys,
    )
    assert "FR-007" not in prompt
    assert "FR-008" not in prompt
    assert "FR-NNN" not in prompt
    assert "NFR-XXX-NNN" not in prompt
    assert "US-NN-NN" not in prompt
    # And the agile vocabulary IS present.
    assert "EPIC-001" in prompt


def test_build_decomposition_augment_prompt_no_fr_leak_on_agile_workspace():
    agile_keys = {"EPIC-001", "STORY-001", "STORY-NFR-001"}
    prompt = decomposition._build_decomposition_augment_prompt(
        existing_features=[{"feature_key": "auth", "name": "Auth"}],
        existing_stories=[{
            "story_key": "STORY-001", "feature_key": "auth",
            "title": "Login", "status": "done", "acceptance_criteria": [],
        }],
        spec_requirements="## EPIC-001: Auth\n",
        spec_architecture="",
        workspace_path="/tmp/ws",
        known_req_keys=agile_keys,
    )
    assert "FR-007" not in prompt
    assert "FR-NNN" not in prompt
    assert "STORY-001" in prompt


def test_validator_error_no_longer_hardcodes_waterfall_hint():
    """The 'must cite at least one' error used to embed FR-NNN /
    NFR-XXX-NNN / US-NN-NN — wrong on agile workspaces. The full known
    set is reported by the separate 'unknown key' branch; this branch
    just needs to point at the spec."""
    with pytest.raises(ValueError) as exc_info:
        decomposition._validate_story_requirement_keys(
            "STORY-001", raw=[], known_req_keys={"EPIC-001"},
        )
    msg = str(exc_info.value)
    assert "FR-NNN" not in msg
    assert "NFR-XXX-NNN" not in msg
    assert "US-NN-NN" not in msg
    assert "docs/SPEC_REQUIREMENTS.md" in msg


