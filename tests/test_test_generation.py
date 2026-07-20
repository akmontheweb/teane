"""Tests for harness/test_generation.py — the auto test-generation + deterministic
sandbox execution node wired between speculative_node and lintgate_node.
"""

from __future__ import annotations

from typing import Any

import pytest

from harness.test_generation import (
    _PRIMARY_STACK_PRIORITY,
    _STACK_TEST_COMMANDS,
    _build_format_reminder,
    _is_test_file,
    _parse_verifies_marker,
    _persist_verifies_links,
    _pick_primary_stack,
    _stack_test_command,
    _inside_workspace,
    autofix_markers_by_body_reference,
    backfill_untested_nfr_acs,
    route_after_test_generation,
)
# Imported under a non-test_ alias so pytest's auto-collection doesn't try
# to invoke this graph node as if it were a test function.
from harness.test_generation import test_generation_node as run_test_generation


# ---------------------------------------------------------------------------
# Helpers — stubs for the gateway + sandbox so the node runs without touching
# the network or spinning up docker.
# ---------------------------------------------------------------------------

class _StubUsage:
    input_tokens = 100
    output_tokens = 80
    cached_tokens = 0
    cost_usd = 0.001
    model = "stub"


class _StubResponse:
    def __init__(self, content: str):
        self.content = content
        self.usage = _StubUsage()


class _StubGateway:
    """Records every dispatch + returns canned patch content.

    ``content`` may be a single string (returned for every dispatch) or a
    list of strings consumed one per dispatch, the last repeating — for
    tests that script a multi-turn exchange (e.g. zero-emit retry)."""

    def __init__(self, content):
        self._contents = list(content) if isinstance(content, list) else [content]
        self.dispatched = []

    class config:
        repair_fallback = ""
        planning_fallback = ""

    async def dispatch(self, *, messages, role, budget_remaining_usd, **kwargs):
        self.dispatched.append({"messages": list(messages), "role": role})
        idx = min(len(self.dispatched) - 1, len(self._contents) - 1)
        return _StubResponse(self._contents[idx]), budget_remaining_usd - 0.001

    def aggregate_tokens(self, tracker, usage, role=None):
        out = dict(tracker or {})
        out["total_cost_usd"] = out.get("total_cost_usd", 0.0) + float(usage.cost_usd)
        return out


class _StubBuildResult:
    def __init__(self, exit_code: int, raw_output: str):
        self.exit_code = exit_code
        self.raw_output = raw_output
        self.diagnostics = []
        self.elapsed_seconds = 0.01
        self.timed_out = False
        self.log_truncated = False


class _StubSandboxExecutor:
    """Records every run() invocation and returns a pre-canned BuildResult."""
    last_command: str = ""
    canned: _StubBuildResult = _StubBuildResult(0, "")

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def run(self, build_command: str):
        _StubSandboxExecutor.last_command = build_command
        return _StubSandboxExecutor.canned


@pytest.fixture
def stub_sandbox(monkeypatch):
    """Replace harness.sandbox.SandboxExecutor with the stub above and return
    a setter the test can use to pre-can the build result."""
    import harness.sandbox as sandbox_mod
    monkeypatch.setattr(sandbox_mod, "SandboxExecutor", _StubSandboxExecutor)

    def _set(exit_code: int, raw_output: str = "") -> None:
        _StubSandboxExecutor.canned = _StubBuildResult(exit_code, raw_output)
        _StubSandboxExecutor.last_command = ""

    _set(0, "")  # default to passing sandbox
    return _set


@pytest.fixture
def stub_gateway(monkeypatch):
    """Install a stub LLM gateway for the duration of the test."""
    from harness import graph as graph_mod

    holder: dict[str, _StubGateway] = {}

    def _set(content: str) -> _StubGateway:
        gw = _StubGateway(content)
        graph_mod.set_gateway(gw)
        holder["gw"] = gw
        return gw

    yield _set

    graph_mod.set_gateway(None)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_format_reminder_carries_contradiction_rules(self):
        # Rules 6-7 (generation-side contradiction prevention) must be in
        # the author's prompt so the model reconciles layers up front.
        reminder = _build_format_reminder()
        assert "ONE enforcement layer per validation" in reminder
        assert "CONSTRUCTIBILITY before use" in reminder
        # And the pre-existing rules are still present (not clobbered).
        assert "Do NOT generate mocks" in reminder
        assert "@tests" in reminder

    def test_is_test_file_python(self):
        assert _is_test_file("tests/test_foo.py") is True
        assert _is_test_file("src/foo.py") is False

    def test_is_test_file_javascript(self):
        assert _is_test_file("src/foo.test.ts") is True
        assert _is_test_file("__tests__/foo.spec.js") is True
        assert _is_test_file("src/foo.ts") is False

    def test_is_test_file_java(self):
        assert _is_test_file("src/test/java/com/x/FooTest.java") is True
        assert _is_test_file("src/main/java/com/x/Foo.java") is False

    def test_pick_primary_stack_prefers_specific_over_generic(self):
        # typescript should win over node when both are present
        assert _pick_primary_stack({"node", "typescript"}) == "typescript"
        assert _pick_primary_stack({"python", "node"}) in ("node", "python")

    def test_pick_primary_stack_none_when_unknown(self):
        assert _pick_primary_stack({"cobol"}) is None
        assert _pick_primary_stack(set()) is None

    def test_stack_test_command_runs_pytest_for_python(self):
        # pytest is pre-baked into the builder image so the per-run install
        # step is gone — the command is now just the pytest invocation.
        cmd = _stack_test_command("python")
        assert cmd is not None
        assert "pytest" in cmd
        # Regression: the legacy `pip install -q pytest && ...` prefix must
        # NOT come back; that round-tripped to PyPI on every single test run.
        assert "pip install" not in cmd

    def test_stack_test_command_runs_jest_for_javascript(self):
        # jest is also pre-baked; `npx --no-install` resolves it from PATH.
        cmd = _stack_test_command("javascript")
        assert cmd is not None
        assert "jest" in cmd
        assert "npm install" not in cmd

    def test_every_priority_stack_has_a_test_command(self):
        # If we add a new stack to the priority list we must also add its
        # test command, otherwise the node silently skips the deterministic
        # run. Catch the omission in CI.
        for tag in _PRIMARY_STACK_PRIORITY:
            assert tag in _STACK_TEST_COMMANDS, (
                f"stack {tag!r} in _PRIMARY_STACK_PRIORITY but missing from "
                f"_STACK_TEST_COMMANDS"
            )

    def test_inside_workspace_accepts_relative(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_x.py").write_text("")
        assert _inside_workspace("tests/test_x.py", str(tmp_path)) is True

    def test_inside_workspace_rejects_absolute(self, tmp_path):
        assert _inside_workspace("/etc/passwd", str(tmp_path)) is False

    def test_inside_workspace_rejects_traversal(self, tmp_path):
        assert _inside_workspace("../outside.py", str(tmp_path)) is False


# ---------------------------------------------------------------------------
# Skip / gate behaviours
# ---------------------------------------------------------------------------

class TestSkipBehaviour:

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self, tmp_path, stub_sandbox, stub_gateway):
        # enabled: false → no work, no LLM call
        gw = stub_gateway("")
        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["foo.py"],
            "test_generation_config": {"enabled": False},
        })
        assert result == {}
        assert gw.dispatched == []

    @pytest.mark.asyncio
    async def test_skips_when_modified_files_empty(self, tmp_path, stub_sandbox, stub_gateway):
        gw = stub_gateway("")
        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": [],
        })
        assert result == {}
        assert gw.dispatched == []

    @pytest.mark.asyncio
    async def test_skips_when_only_test_files_modified(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        # modified files are themselves tests → nothing to do
        gw = stub_gateway("")
        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["tests/test_foo.py", "src/foo.test.ts"],
        })
        assert result == {}
        assert gw.dispatched == []

    @pytest.mark.asyncio
    async def test_skips_when_no_supported_stack(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        gw = stub_gateway("")
        # Unknown extension → no stack tag inferred
        (tmp_path / "x.cobol").write_text("DISPLAY 'hi'.\n")
        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["x.cobol"],
        })
        assert result == {}
        assert gw.dispatched == []

    @pytest.mark.asyncio
    async def test_routes_to_hitl_when_no_gateway(self, tmp_path, stub_sandbox):
        # gateway is None → env_misconfig diagnostic + route to HITL
        from harness import graph as graph_mod
        graph_mod.set_gateway(None)

        (tmp_path / "foo.py").write_text("def x(): return 1\n")
        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["foo.py"],
        })
        assert result["node_state"]["env_misconfig"] is True
        assert result["node_state"]["env_misconfig_symbol"] == "llm_api_key"
        # The synthetic diagnostic must spell out the fix the operator needs.
        msg = result["compiler_errors"][0]["message"]
        assert "LLM API key" in msg
        assert "ANTHROPIC_API_KEY" in msg
        # And the router must send it to HITL.
        assert route_after_test_generation(result) == "human_intervention_node"

    @staticmethod
    def _seed_nfr_story(workspace_path: str, story_key: str, acs: list[str]) -> None:
        """Insert an NFR story (with the explicit STORY-NFR-N key) plus its
        acceptance criteria into the per-test state.db. create_stories()
        allocates keys sequentially and can't be told to use an NFR key,
        so we go through raw SQL for the story row and reuse the public
        AC helper for the criteria."""
        from harness import story_state
        app = story_state.app_name_for_workspace(workspace_path)
        conn = story_state.open_story_db()
        try:
            story_state.ensure_feature(conn, app, "nfr-f", name="NFR feature")
            feat = story_state.get_feature_by_key(conn, app, "nfr-f")
            now = "2026-07-11T00:00:00+00:00"
            cur = conn.execute(
                "INSERT INTO stories(workspace, story_key, feature_id, title, "
                "depends_on, scope_files, status, build_kind, created_at) "
                "VALUES(?, ?, ?, ?, '[]', '[]', 'planned', 'greenfield', ?)",
                (app, story_key, int(feat["id"]), f"{story_key} title", now),
            )
            story_id = int(cur.lastrowid)
            story_state.create_acceptance_criteria(
                conn, app, story_id,
                [
                    {"ac_key": f"{story_key}.AC-{i + 1}", "text": t, "ordinal": i + 1}
                    for i, t in enumerate(acs)
                ],
            )
            conn.commit()
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_nfr_only_batch_skips_without_stubs_or_links(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        """Unit-test model: NFR-only batches skip cleanly — no LLM
        dispatch (NFRs aren't unit-testable and the model burns its
        zero-emit sub-cap trying), no `@verifies` skip-stubs, and no
        ``test_verifies_ac`` edges. NFR verification is owned by the
        `teane test` functional pack; the AC-coverage gate only fires
        in that flow (traceability.has_ac_gap)."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "foo.py").write_text("def x(): return 1\n")
        self._seed_nfr_story(
            str(tmp_path), "STORY-NFR-001",
            ["Non-December fiscal year", "Current incomplete fiscal year"],
        )
        gw = stub_gateway("should never be dispatched")
        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["foo.py"],
            "batch_patched_story_keys": ["STORY-NFR-001"],
        })
        # No LLM dispatch — the whole point.
        assert gw.dispatched == []
        ns = result["node_state"]["test_generation"]
        assert ns["status"] == "skipped"
        assert ns["reason"] == "nfr_only_batch"
        assert ns["story_keys"] == ["STORY-NFR-001"]
        # No placeholder stubs on disk, no AC edges in state.db —
        # build/patch never writes AC linkage.
        assert not (tmp_path / "tests" / "nfr").exists()
        from harness import story_state
        app = story_state.app_name_for_workspace(str(tmp_path))
        conn = story_state.open_story_db()
        try:
            rows = conn.execute(
                "SELECT test_path FROM test_verifies_ac "
                "WHERE workspace = ?", (app,),
            ).fetchall()
        finally:
            conn.close()
        assert rows == []

    @pytest.mark.asyncio
    async def test_nfr_only_batch_skips_even_without_story_rows(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        """The skip is keyed on the story-key SHAPE alone — no state.db
        rows required (decomposition drift, wrong workspace path)."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "foo.py").write_text("def x(): return 1\n")
        # NOTE: no _seed_nfr_story call — state.db has no matching rows
        gw = stub_gateway("...")
        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["foo.py"],
            "batch_patched_story_keys": ["STORY-NFR-999"],
        })
        assert gw.dispatched == []
        ns = result["node_state"]["test_generation"]
        assert ns["status"] == "skipped"
        assert ns["reason"] == "nfr_only_batch"
        assert ns["story_keys"] == ["STORY-NFR-999"]

    @pytest.mark.asyncio
    async def test_nfr_only_batch_skips_on_any_stack(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        """The skip is stack-independent — no stub templates exist any
        more, so there is no python/typescript special case."""
        (tmp_path / "Main.java").write_text("class Main {}\n")
        gw = stub_gateway("...")
        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["Main.java"],
            "batch_patched_story_keys": ["STORY-NFR-001"],
        })
        assert gw.dispatched == []
        ns = result["node_state"]["test_generation"]
        assert ns["status"] == "skipped"
        assert ns["reason"] == "nfr_only_batch"

    @pytest.mark.asyncio
    async def test_mixed_nfr_and_regular_batch_still_dispatches(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        """Guardrail is intentionally narrow: mixed batches (NFR alongside
        a regular story) MUST still run test-gen — the regular story
        anchors the tests and the NFR ACs can ride along as ``@verifies:``
        citations."""
        gw = stub_gateway("<<<CREATE_FILE:tests/test_foo.py>>>\ndef test_foo():\n    from foo import x\n    assert x() == 1\n<<<END>>>\n")
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "foo.py").write_text("def x(): return 1\n")
        stub_sandbox(0, "1 passed in 0.01s\n")
        await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["foo.py"],
            "batch_patched_story_keys": ["STORY-005", "STORY-NFR-002"],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
        })
        # LLM was actually dispatched — the guardrail did NOT skip.
        assert len(gw.dispatched) >= 1


# ---------------------------------------------------------------------------
# Happy path: LLM emits a CREATE_FILE block, sandbox passes
# ---------------------------------------------------------------------------

class TestHappyPath:

    @pytest.mark.asyncio
    async def test_python_writes_test_runs_pytest_routes_to_lintgate(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        # Workspace with a Python source file
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "calculator.py").write_text(
            "def divide(a, b):\n"
            "    if b == 0:\n"
            "        raise ZeroDivisionError('cannot divide by zero')\n"
            "    return a // b\n"
        )

        # Stub the LLM to return one CREATE_FILE block for a test file.
        # v5 Phase 3 contract: every generated test MUST carry a
        # `# @verifies: STORY-N.AC-N` marker. This canned response
        # includes one so the marker gate passes; the ac_key references
        # an AC that doesn't exist in this test's state.db, which
        # produces a "dropped unknown ac_key" log but doesn't fail the
        # gate (Phase 3 warn-and-drop).
        gw = stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_calculator.py\n"
            "content:\n"
            "# @verifies: STORY-001.AC-1\n"
            "from calculator import divide\n"
            "def test_divide():\n"
            "    assert divide(10, 2) == 5\n"
            "<<<END_CREATE_FILE>>>\n"
        )

        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["calculator.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
        })

        # 1. LLM was called exactly once
        assert len(gw.dispatched) == 1
        sent = gw.dispatched[0]["messages"]
        # 2. The Python test guide was injected into the system messages
        guide_sys = [m for m in sent if m.get("role") == "system" and "monkeypatch" in m.get("content", "")]
        assert guide_sys, "system prompt should include the python test guide"
        # 3. The prompt explicitly forbids mocks
        user_msgs = [m for m in sent if m.get("role") == "user"]
        joined_user = "\n".join(m.get("content", "") for m in user_msgs)
        assert "Do NOT generate mocks" in joined_user or "do not generate mocks" in joined_user.lower()
        # 4. The deterministic test command ran in the sandbox. pytest is
        # pre-baked into the builder image so there's no install prefix —
        # just the bare pytest invocation.
        assert "pytest" in _StubSandboxExecutor.last_command
        assert "pip install" not in _StubSandboxExecutor.last_command
        # 5. The result reports a pass and lists the generated test
        assert result["node_state"]["test_generation"]["status"] == "passed"
        assert result["generated_tests"] == ["tests/test_calculator.py"]
        # 6. The router would proceed to lintgate
        assert route_after_test_generation(result) == "lintgate_node"
        # 7. The test file landed inside the workspace, not anywhere else
        assert (tmp_path / "tests" / "test_calculator.py").is_file()

    @pytest.mark.asyncio
    async def test_zero_patch_emission_retries_then_trips_hitl(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        # Fix 2a (2026-07-10): when the LLM returns zero patch blocks
        # that's a prompt-comprehension miss, not a benign "no tests
        # needed" pass-through. The node retries inline with a stronger
        # contract (max_zero_emit_reprompts=3 by default). If the LLM
        # STILL emits nothing, HITL fires with a distinct
        # llm_behavior symbol so the operator can distinguish this
        # failure class from the generic max_iterations one.
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "foo.py").write_text("def foo(): pass\n")
        stub_gateway("no patch blocks here, just prose")
        stub_sandbox(99, "this should not be observed because sandbox should not run")

        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["foo.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
        })
        # No deterministic run happened — we never got past the
        # zero-emit retry loop.
        assert _StubSandboxExecutor.last_command == ""
        # LLM-behavior HITL fires with the distinct zero_emit symbol
        # so a post-mortem can see WHICH failure class ate the budget.
        assert result["node_state"]["llm_behavior"] is True
        assert (
            result["node_state"]["llm_behavior_symbol"]
            == "test_generation_zero_emit"
        )
        assert result["exit_code"] == 1
        # The zero-emit budget is fully consumed but the real
        # test_generation iteration counter DID NOT advance — that's
        # the whole point of the sub-counter split.
        assert result["loop_counter"]["test_generation_zero_emit"] == 3
        assert "test_generation" not in result["loop_counter"] or \
            result["loop_counter"]["test_generation"] == 0
        # Router sends this to human_intervention on llm_behavior.
        assert route_after_test_generation(result) == "human_intervention_node"


# ---------------------------------------------------------------------------
# Cross-file contradiction gate (generation-side prevention, lumina 019f803f)
# ---------------------------------------------------------------------------

class TestContradictionGate:

    _SRC_SCHEMA = (
        "from pydantic import BaseModel\n"
        "class ContactUpdate(BaseModel):\n"
        "    first_name: str | None = None\n"
    )
    _SRC_SERVICE = (
        "def update_contact(db, cid, payload):\n"
        "    return payload\n"
    )

    # Round 1: the author emits a same-input / opposite-expectation pair
    # split across two files — the schema test requires the ctor to RAISE,
    # the service test constructs the same value expecting it to SUCCEED.
    _CONTRADICTING = (
        "<<<CREATE_FILE>>>\n"
        "file: server/tests/test_contact_schemas.py\n"
        "content:\n"
        "# @tests: server/app/schemas/contact.py\n"
        "import pytest\n"
        "from pydantic import ValidationError\n"
        "from server.app.schemas.contact import ContactUpdate\n"
        "def test_empty_first_name_rejected():\n"
        "    with pytest.raises(ValidationError):\n"
        "        ContactUpdate(first_name='   ')\n"
        "<<<END_CREATE_FILE>>>\n"
        "<<<CREATE_FILE>>>\n"
        "file: server/tests/test_contact_service.py\n"
        "content:\n"
        "# @tests: server/app/services/contact_service.py\n"
        "from server.app.schemas.contact import ContactUpdate\n"
        "def test_service_path():\n"
        "    payload = ContactUpdate(first_name='   ')\n"
        "    assert payload is not None\n"
        "<<<END_CREATE_FILE>>>\n"
    )

    # Round 2: the corrected service test no longer constructs the invalid
    # value, so the batch is consistent.
    _FIXED = (
        "<<<REWRITE_FILE>>>\n"
        "file: server/tests/test_contact_service.py\n"
        "content:\n"
        "# @tests: server/app/services/contact_service.py\n"
        "from server.app.schemas.contact import ContactUpdate\n"
        "def test_service_path():\n"
        "    payload = ContactUpdate(first_name='Alice')\n"
        "    assert payload.first_name == 'Alice'\n"
        "<<<END_REWRITE_FILE>>>\n"
    )

    def _workspace(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        for rel, body in (
            ("server/app/schemas/contact.py", self._SRC_SCHEMA),
            ("server/app/services/contact_service.py", self._SRC_SERVICE),
        ):
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body)

    @pytest.mark.asyncio
    async def test_contradiction_bounces_back_to_author_and_resolves(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        self._workspace(tmp_path)
        gw = stub_gateway([self._CONTRADICTING, self._FIXED])
        stub_sandbox(0, "2 passed in 0.01s\n")

        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": [
                "server/app/schemas/contact.py",
                "server/app/services/contact_service.py",
            ],
            "messages": [],
            "budget_remaining_usd": 2.0,
            "token_tracker": {},
        })

        # The author was re-prompted: 1 initial + 1 contradiction bounce.
        assert len(gw.dispatched) == 2
        # The re-prompt named the exact unsatisfiable call.
        bounce = gw.dispatched[1]["messages"]
        joined = "\n".join(m.get("content", "") for m in bounce)
        assert "UNSATISFIABLE" in joined
        assert "ContactUpdate(first_name='   ')" in joined
        # The corrected batch reached the sandbox and passed.
        assert "pytest" in _StubSandboxExecutor.last_command
        assert result["node_state"]["test_generation"]["status"] == "passed"
        # The on-disk service test is the fixed version (no invalid ctor).
        svc = (tmp_path / "server/tests/test_contact_service.py").read_text()
        assert "first_name='Alice'" in svc
        assert "first_name='   '" not in svc

    @pytest.mark.asyncio
    async def test_no_contradiction_does_not_reprompt(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        # A clean batch dispatches exactly once — the gate is a no-op.
        self._workspace(tmp_path)
        clean = (
            "<<<CREATE_FILE>>>\n"
            "file: server/tests/test_contact_schemas.py\n"
            "content:\n"
            "# @tests: server/app/schemas/contact.py\n"
            "import pytest\n"
            "from pydantic import ValidationError\n"
            "from server.app.schemas.contact import ContactUpdate\n"
            "def test_empty_first_name_rejected():\n"
            "    with pytest.raises(ValidationError):\n"
            "        ContactUpdate(first_name='   ')\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        gw = stub_gateway([clean, "SHOULD NOT BE DISPATCHED"])
        stub_sandbox(0, "1 passed in 0.01s\n")

        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["server/app/schemas/contact.py"],
            "messages": [],
            "budget_remaining_usd": 2.0,
            "token_tracker": {},
        })
        assert len(gw.dispatched) == 1
        assert result["node_state"]["test_generation"]["status"] == "passed"


# ---------------------------------------------------------------------------
# Failure path: sandbox exits non-zero → repair_node
# ---------------------------------------------------------------------------

class TestFailurePath:

    @pytest.mark.asyncio
    async def test_test_failure_routes_to_repair_with_test_failure_code(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "foo.py").write_text("def foo(): return 1\n")
        stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_foo.py\n"
            "content:\n"
            "# @verifies: STORY-001.AC-1\n"
            "from foo import foo\n"
            "def test_foo(): assert foo() == 2  # wrong on purpose\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(1, "tests/test_foo.py:2: assert 1 == 2\nFAILED tests/test_foo.py::test_foo")

        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["foo.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
        })

        assert result["node_state"]["test_generation"]["status"] == "failed"
        assert result["compiler_errors"], "must populate compiler_errors on failure"
        # Every diagnostic must carry the TEST_FAILURE prefix so repair_node's
        # framing tweak knows these came from the test runner.
        codes = [d["error_code"] for d in result["compiler_errors"]]
        assert all(c.upper().startswith("TEST_FAILURE") for c in codes), codes
        # Router sends to repair.
        assert route_after_test_generation(result) == "repair_node"


# ---------------------------------------------------------------------------
# Graph wiring smoke test
# ---------------------------------------------------------------------------

class TestGraphWiring:

    def test_graph_includes_test_generation_node(self):
        # The build_graph must register the new node and the edge from
        # speculative_node to it. We don't execute the graph here, just
        # confirm the wiring via the graph's compiled structure.
        from harness.graph import build_graph
        try:
            g = build_graph(checkpointer=None)
        except Exception as exc:
            pytest.skip(f"build_graph requires extra deps in this env: {exc}")
        # LangGraph compiled graph exposes node names via .nodes
        node_names = set(g.nodes.keys()) if hasattr(g, "nodes") else set()
        # build_graph returns a compiled graph; the source StateGraph has
        # different introspection. Either path is acceptable — just confirm
        # the node name appears somewhere in repr.
        graph_repr = repr(g)
        assert (
            "test_generation_node" in node_names
            or "test_generation_node" in graph_repr
        )


# ---------------------------------------------------------------------------
# repair_node framing for TEST_FAILURE diagnostics
# ---------------------------------------------------------------------------

class TestRepairFraming:

    @pytest.mark.asyncio
    async def test_repair_node_uses_test_failure_framing(self, tmp_path):
        # Feed repair_node a state whose only diagnostic carries TEST_FAILURE.
        # The prompt sent to the LLM must contain the new framing sentence,
        # NOT the generic "build failed" framing or the security framing.
        from harness import graph as graph_mod

        captured: dict[str, Any] = {}

        class StubResp:
            content = ""

            class usage:
                input_tokens = 0
                output_tokens = 0
                cached_tokens = 0
                cost_usd = 0.0
                model = "stub"

        class StubGW:
            class config:
                repair_fallback = ""
                planning_fallback = ""

            async def dispatch(self, *, messages, role, budget_remaining_usd, **kwargs):
                captured["messages"] = list(messages)
                return StubResp(), budget_remaining_usd

            def aggregate_tokens(self, tracker, usage, role=None):
                return tracker or {}

        graph_mod.set_gateway(StubGW())
        try:
            await graph_mod.repair_node({
                "workspace_path": str(tmp_path),
                "compiler_errors": [{
                    "file": "tests/test_x.py",
                    "line": 5,
                    "column": 0,
                    "severity": "error",
                    "error_code": "TEST_FAILURE:assertion",
                    "message": "assert 1 == 2",
                    "semantic_context": "",
                }],
                "loop_counter": {"total_repairs": 0, "repair": 0},
                "messages": [],
                "modified_files": [],
                "budget_remaining_usd": 1.0,
            })
        finally:
            graph_mod.set_gateway(None)

        user_msgs = [m for m in captured["messages"] if m.get("role") == "user"]
        joined = "\n".join(m.get("content", "") for m in user_msgs)
        assert "harness-generated unit tests" in joined, (
            "repair_node must use the TEST_FAILURE framing for these diagnostics"
        )
        assert "Do NOT add mocks" in joined or "do not add mocks" in joined.lower()


# ---------------------------------------------------------------------------
# v5 @verifies marker — parser + gate + link writer
# ---------------------------------------------------------------------------

class TestVerifiesMarkerParser:
    """Unit tests for _parse_verifies_marker. Permissive on whitespace,
    strict on key shape (STORY-N.AC-N anchored)."""

    def test_python_comment_style(self):
        assert _parse_verifies_marker(
            "# @verifies: STORY-003.AC-2\nimport pytest"
        ) == ["STORY-003.AC-2"]

    def test_js_java_comment_style(self):
        assert _parse_verifies_marker(
            "// @verifies: STORY-001.AC-1\nimport foo;"
        ) == ["STORY-001.AC-1"]

    def test_multi_ac_comma_separated(self):
        assert _parse_verifies_marker(
            "// @verifies: STORY-001.AC-1, STORY-001.AC-2, STORY-002.AC-3\n"
        ) == ["STORY-001.AC-1", "STORY-001.AC-2", "STORY-002.AC-3"]

    def test_permissive_whitespace(self):
        assert _parse_verifies_marker(
            "#  @verifies:   STORY-005.AC-7\n"
        ) == ["STORY-005.AC-7"]

    def test_missing_marker_returns_empty(self):
        assert _parse_verifies_marker(
            "import pytest\ndef test_foo(): pass\n"
        ) == []

    def test_malformed_key_filtered_out(self):
        # NOT-A-KEY doesn't match STORY-N.AC-N; the cleaning step drops it.
        assert _parse_verifies_marker("# @verifies: NOT-A-KEY\n") == []

    def test_mixed_valid_and_invalid_keeps_valid(self):
        assert _parse_verifies_marker(
            "# @verifies: STORY-001.AC-1, BOGUS, STORY-002.AC-3\n"
        ) == ["STORY-001.AC-1", "STORY-002.AC-3"]

    def test_marker_buried_past_scan_window_ignored(self):
        body = "\n".join(
            ["# preamble"] * 60 + ["# @verifies: STORY-001.AC-1"]
        )
        assert _parse_verifies_marker(body) == []

    def test_empty_body_returns_empty(self):
        assert _parse_verifies_marker("") == []


class TestTestsMarkerGate:
    """Integration tests for the ``@tests`` code-linkage gate inside
    test_generation_node. Unit tests link to the CODE under test —
    a missing marker is autofixed deterministically from the source
    files the generation call was asked to cover, so no LLM turn (and
    no repair round-trip) is spent on it."""

    @pytest.mark.asyncio
    async def test_markerless_test_gets_tests_marker_autofixed(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "calculator.py").write_text("def divide(a, b): return a // b\n")
        stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_calculator.py\n"
            "content:\n"
            "from calculator import divide\n"
            "def test_divide(): assert divide(10, 2) == 5\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(0, "1 passed in 0.01s")

        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["calculator.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
            "decomposition_enabled": True,
        })

        assert result["node_state"]["test_generation"]["status"] == "passed"
        body = (tmp_path / "tests" / "test_calculator.py").read_text()
        # Basename heuristic maps test_calculator.py → calculator.py.
        assert "# @tests: calculator.py" in body
        assert "@verifies" not in body
        assert route_after_test_generation(result) == "lintgate_node"

    @pytest.mark.asyncio
    async def test_marker_comment_style_follows_file_not_primary_stack(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        """Regression (lumina 019f7054): in a mixed py+ts workspace,
        _PRIMARY_STACK_PRIORITY resolves primary to "typescript", and the
        @tests autofix used the PRIMARY stack's comment lead — stamping a
        JS-style ``// @tests:`` onto a PYTHON file. That's a SyntaxError
        on line 1 of the test file, which killed pytest collection for
        the whole package and dead-ended in a zero-patch HITL (the
        repair loop's test-guard refused to touch the file). The lead
        must come from the file being stamped, exactly as the @verifies
        autofix already does."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        # Root package.json + tsconfig → typescript tag, which OUTRANKS
        # python in _PRIMARY_STACK_PRIORITY.
        (tmp_path / "package.json").write_text('{"name": "client"}\n')
        (tmp_path / "tsconfig.json").write_text("{}\n")
        (tmp_path / "calculator.py").write_text(
            "def divide(a, b): return a // b\n"
        )
        stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_calculator.py\n"
            "content:\n"
            "from calculator import divide\n"
            "def test_divide(): assert divide(10, 2) == 5\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(0, "1 passed in 0.01s")

        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["calculator.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
            "decomposition_enabled": True,
        })

        body = (tmp_path / "tests" / "test_calculator.py").read_text()
        assert "# @tests: calculator.py" in body
        assert "// @tests" not in body
        # The autofixed file must still be valid Python.
        import ast as _ast
        _ast.parse(body)

    @pytest.mark.asyncio
    async def test_stack_framing_follows_sources_not_workspace_primary(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        """Regression (lumina 019f7109): in a mixed py+ts workspace the
        test-gen prompt framed PYTHON sources as "(stack: typescript)" —
        the workspace-priority pick ranks typescript above python — and
        the model floundered into the zero-emit HITL. Homogeneous
        sources must drive the stack for the generation call."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "package.json").write_text('{"name": "client"}\n')
        (tmp_path / "tsconfig.json").write_text("{}\n")
        (tmp_path / "calculator.py").write_text(
            "def divide(a, b): return a // b\n"
        )
        gw = stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_calculator.py\n"
            "content:\n"
            "from calculator import divide\n"
            "def test_divide(): assert divide(10, 2) == 5\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(0, "1 passed in 0.01s")

        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["calculator.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
            "decomposition_enabled": True,
        })

        assert result["node_state"]["test_generation"]["status"] == "passed"
        joined = "\n".join(
            str(m.get("content", ""))
            for d in gw.dispatched for m in d["messages"]
        )
        assert "(stack: python)" in joined
        assert "(stack: typescript)" not in joined

    @pytest.mark.asyncio
    async def test_zero_emit_mimicry_gets_targeted_feedback(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        """Regression (lumina 019f7109): the model adopted the flattened
        tool-history notation as a tool syntax and zero-emitted three
        responses in it. The retry system message must name that exact
        mistake, not just repeat the generic emit-patch-blocks nudge."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "f.py").write_text("def f(): return 1\n")
        responses = [
            # First response: bracket-mimicry, zero parseable blocks.
            '[called tool create_file with arguments: {"file_path": '
            '"tests/test_f.py", "content": "def test_f(): pass"}]',
            # After the targeted retry: a real patch block.
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_f.py\n"
            "content:\n"
            "from f import f\n"
            "def test_f(): assert f() == 1\n"
            "<<<END_CREATE_FILE>>>\n",
        ]
        stub_gateway(responses)
        stub_sandbox(0, "1 passed")

        state = {
            "workspace_path": str(tmp_path),
            "modified_files": ["f.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
            "decomposition_enabled": True,
        }
        result = await run_test_generation(state)

        assert result["node_state"]["test_generation"]["status"] == "passed"
        retry_msgs = [
            m for m in result["messages"]
            if m.get("role") == "system"
            and "zero PATCH blocks" in str(m.get("content", ""))
        ]
        assert retry_msgs, "expected a zero-emit retry system message"
        assert any(
            "NOT a tool interface" in str(m["content"]) for m in retry_msgs
        )

    @pytest.mark.asyncio
    async def test_verifies_marker_alone_does_not_satisfy_gate(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        """An AC marker is NOT code linkage — a test carrying only
        ``@verifies`` still gets the ``@tests`` autofix."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "f.py").write_text("def f(): return 1\n")
        stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_f.py\n"
            "content:\n"
            "# @verifies: STORY-001.AC-1\n"
            "from f import f\n"
            "def test_f(): assert f() == 1\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(0, "1 passed")

        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["f.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
            "decomposition_enabled": True,
        })

        assert result["node_state"]["test_generation"]["status"] == "passed"
        body = (tmp_path / "tests" / "test_f.py").read_text()
        assert "# @tests: f.py" in body

    @pytest.mark.asyncio
    async def test_present_tests_marker_is_left_alone(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "g.py").write_text("def g(): return 2\n")
        stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_g.py\n"
            "content:\n"
            "# @tests: g.py\n"
            "from g import g\n"
            "def test_g(): assert g() == 2\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(0, "1 passed")

        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["g.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
        })

        assert result["node_state"]["test_generation"]["status"] == "passed"
        body = (tmp_path / "tests" / "test_g.py").read_text()
        assert body.count("@tests:") == 1


class TestVerifiesLinkPersistence:
    """_persist_verifies_links writes (test_path, ac_key) edges into
    the test_verifies_ac table after the sandbox passes."""

    def test_empty_input_no_op(self, tmp_path):
        assert _persist_verifies_links(str(tmp_path), {}) == (0, 0)

    def test_inserts_known_ac_keys_and_drops_unknown(self, tmp_path):
        from harness import story_state
        # Seed: one feature + one story + one AC on a fresh state.db
        # scoped to this tmp_path's basename (which is what
        # app_name_for_workspace derives).
        ws = tmp_path / "verify-link-ws"
        ws.mkdir()
        app = story_state.app_name_for_workspace(str(ws))
        conn = story_state.open_story_db()
        try:
            story_state.ensure_feature(conn, app, "f", name="F")
            keys = story_state.create_stories(conn, app, [{
                "title": "S", "feature": "f",
                "acceptance_criteria": ["only AC"],
            }])
            sid = story_state.get_story(conn, app, keys[0])["id"]
            ac = story_state.list_acceptance_criteria(conn, app, sid)[0]
            known_ac_key = ac["ac_key"]
        finally:
            conn.close()

        inserted, dropped = _persist_verifies_links(
            str(ws),
            {
                "tests/test_real.py": [known_ac_key, "STORY-099.AC-99"],
                "tests/test_other.py": [known_ac_key],
            },
        )
        assert inserted == 2  # one per file pointing at the known AC
        assert dropped == 1  # the STORY-099 key

        # Round-trip: link rows present
        conn = story_state.open_story_db()
        try:
            rows = conn.execute(
                "SELECT test_path FROM test_verifies_ac "
                "WHERE workspace = ? ORDER BY test_path", (app,),
            ).fetchall()
        finally:
            conn.close()
        assert [r[0] for r in rows] == [
            "tests/test_other.py", "tests/test_real.py",
        ]

    def test_idempotent_on_rerun(self, tmp_path):
        from harness import story_state
        ws = tmp_path / "idempotent-ws"
        ws.mkdir()
        app = story_state.app_name_for_workspace(str(ws))
        conn = story_state.open_story_db()
        try:
            story_state.ensure_feature(conn, app, "f", name="F")
            keys = story_state.create_stories(conn, app, [{
                "title": "S", "feature": "f",
                "acceptance_criteria": ["AC"],
            }])
            sid = story_state.get_story(conn, app, keys[0])["id"]
            ac_key = story_state.list_acceptance_criteria(conn, app, sid)[0]["ac_key"]
        finally:
            conn.close()
        # First call: 1 insert
        a, _ = _persist_verifies_links(str(ws), {"tests/t.py": [ac_key]})
        # Second call: composite PK → no new insert
        b, _ = _persist_verifies_links(str(ws), {"tests/t.py": [ac_key]})
        assert a == 1 and b == 0


# ---------------------------------------------------------------------------
# NFR AC backfill — called by traceability_node before the end-of-batch audit
# ---------------------------------------------------------------------------

class TestBackfillUntestedNfrAcs:
    """2026-07-11 fix — the in-node NFR guard only fires during
    test_generation for a fresh NFR batch. Sessions whose NFR batches
    sealed before the guard existed leave AC edges empty and the
    end-of-session traceability audit blocks. The backfill sweep runs
    inside traceability_node and closes these historical gaps."""

    @staticmethod
    def _seed_nfr_story(workspace_path: str, story_key: str, acs: list[str]) -> None:
        from harness import story_state
        app = story_state.app_name_for_workspace(workspace_path)
        conn = story_state.open_story_db()
        try:
            story_state.ensure_feature(conn, app, "nfr-f", name="NFR feature")
            feat = story_state.get_feature_by_key(conn, app, "nfr-f")
            cur = conn.execute(
                "INSERT INTO stories(workspace, story_key, feature_id, title, "
                "depends_on, scope_files, status, build_kind, created_at) "
                "VALUES(?, ?, ?, ?, '[]', '[]', 'planned', 'greenfield', ?)",
                (app, story_key, int(feat["id"]), f"{story_key}",
                 "2026-07-11T00:00:00+00:00"),
            )
            story_state.create_acceptance_criteria(
                conn, app, int(cur.lastrowid),
                [
                    {"ac_key": f"{story_key}.AC-{i + 1}", "text": t, "ordinal": i + 1}
                    for i, t in enumerate(acs)
                ],
            )
            conn.commit()
        finally:
            conn.close()

    def test_backfill_emits_stubs_for_orphan_nfr_stories(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "foo.py").write_text("def x(): return 1\n")
        # Two NFR stories, both fully orphan; and one regular story with
        # ACs to prove the backfill does NOT touch it.
        self._seed_nfr_story(str(tmp_path), "STORY-NFR-001", ["AC one", "AC two"])
        self._seed_nfr_story(str(tmp_path), "STORY-NFR-004", ["Secrets AC"])

        stub_paths, inserted, dropped = backfill_untested_nfr_acs(str(tmp_path))
        assert set(stub_paths) == {
            "tests/nfr/test_story_nfr_001.py",
            "tests/nfr/test_story_nfr_004.py",
        }
        assert inserted == 3
        assert dropped == 0
        # Files exist and carry markers.
        assert (tmp_path / "tests" / "nfr" / "test_story_nfr_001.py").exists()
        body = (tmp_path / "tests" / "nfr" / "test_story_nfr_001.py").read_text()
        assert "@verifies: STORY-NFR-001.AC-1" in body
        assert "@verifies: STORY-NFR-001.AC-2" in body

    def test_backfill_is_idempotent_and_respects_existing_files(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        self._seed_nfr_story(str(tmp_path), "STORY-NFR-004", ["Secrets AC"])
        # First call writes stub + persists the link.
        first_paths, first_ins, _ = backfill_untested_nfr_acs(str(tmp_path))
        assert first_paths == ["tests/nfr/test_story_nfr_004.py"]
        assert first_ins == 1
        # Operator replaces the stub with a real integration test.
        stub = tmp_path / "tests" / "nfr" / "test_story_nfr_004.py"
        stub.write_text(
            "# @verifies: STORY-NFR-004.AC-1\n"
            "def test_real_integration():\n    assert True\n"
        )
        # Second call: the AC is now covered by the linked test row, so
        # the orphan-AC query returns nothing and the sweep no-ops. The
        # operator's file survives untouched — that's the "respects
        # existing files" contract even without another persist round.
        second_paths, second_ins, _ = backfill_untested_nfr_acs(str(tmp_path))
        assert second_paths == []
        assert second_ins == 0
        assert stub.read_text().startswith("# @verifies: STORY-NFR-004.AC-1")

    def test_backfill_survives_partially_linked_story(self, tmp_path):
        """One AC of a multi-AC NFR story is already linked; the sweep
        must STILL fire (the story is orphan for the other ACs) and the
        existing stub file must be respected (not overwritten)."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        self._seed_nfr_story(
            str(tmp_path), "STORY-NFR-004",
            ["Secrets AC", "CSRF AC", "SQLi AC"],
        )
        # Link only AC-2 via an unrelated test file.
        _persist_verifies_links(
            str(tmp_path),
            {"tests/prior.py": ["STORY-NFR-004.AC-2"]},
        )
        # Pre-existing stub file (operator's manual edit).
        stub = tmp_path / "tests" / "nfr" / "test_story_nfr_004.py"
        stub.parent.mkdir(parents=True)
        stub.write_text("# @verifies: STORY-NFR-004.AC-1\n# hand-edited\n")

        paths, inserted, _ = backfill_untested_nfr_acs(str(tmp_path))
        assert paths == ["tests/nfr/test_story_nfr_004.py"]
        # Three link rows land — the composite PK is
        # (workspace, test_path, ac_id), and this stub is a NEW test_path,
        # so all three ACs are fresh links even though AC-2 already had
        # a link from tests/prior.py.
        assert inserted == 3
        # The operator's hand-edited stub is preserved verbatim — the
        # backfill did NOT overwrite the file.
        assert stub.read_text() == (
            "# @verifies: STORY-NFR-004.AC-1\n# hand-edited\n"
        )

    def test_backfill_skips_when_no_orphan_nfr_stories(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        # NFR story exists but its AC already has a link.
        self._seed_nfr_story(str(tmp_path), "STORY-NFR-002", ["Existing AC"])
        _persist_verifies_links(
            str(tmp_path),
            {"tests/prior.py": ["STORY-NFR-002.AC-1"]},
        )
        stub_paths, inserted, _ = backfill_untested_nfr_acs(str(tmp_path))
        assert stub_paths == []
        assert inserted == 0


# ---------------------------------------------------------------------------
# @verifies marker autofix by body reference — rescues patcher-emitted tests
# ---------------------------------------------------------------------------

class TestAutofixMarkersByBodyReference:
    """2026-07-11 fix — patching_node emits test files as part of a
    story's scope; those tests bypass test_generation_node's marker
    gate entirely. The finsearch run left 20/26 untested ACs on tests
    that DID reference the story in a docstring but forgot the
    ``@verifies:`` line syntax. This helper rescues them at
    batch_commit_node before the sweep."""

    @staticmethod
    def _seed_story(workspace_path: str, story_key: str, acs: list[str]) -> None:
        from harness import story_state
        app = story_state.app_name_for_workspace(workspace_path)
        conn = story_state.open_story_db()
        try:
            story_state.ensure_feature(conn, app, "f", name="F")
            feat = story_state.get_feature_by_key(conn, app, "f")
            cur = conn.execute(
                "INSERT INTO stories(workspace, story_key, feature_id, title, "
                "depends_on, scope_files, status, build_kind, created_at) "
                "VALUES(?, ?, ?, ?, '[]', '[]', 'planned', 'greenfield', ?)",
                (app, story_key, int(feat["id"]), story_key,
                 "2026-07-11T00:00:00+00:00"),
            )
            story_state.create_acceptance_criteria(
                conn, app, int(cur.lastrowid),
                [
                    {"ac_key": f"{story_key}.AC-{i + 1}", "text": t, "ordinal": i + 1}
                    for i, t in enumerate(acs)
                ],
            )
            conn.commit()
        finally:
            conn.close()

    def test_prepends_python_marker_when_docstring_references_story(self, tmp_path):
        self._seed_story(str(tmp_path), "STORY-019", ["AC one", "AC two", "AC three"])
        test_file = tmp_path / "server" / "tests" / "test_ai_service.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text(
            '"""\n'
            "Tests for AI service.\n"
            "\n"
            "STORY-019: Management Guidance Extraction\n"
            '"""\n'
            "def test_x():\n    pass\n"
        )
        scanned, patched = autofix_markers_by_body_reference(str(tmp_path))
        assert scanned == 1
        assert patched == 1
        body = test_file.read_text()
        assert body.splitlines()[0].startswith("# @verifies: STORY-019.AC-1")
        # All three ACs of the referenced story land.
        assert "STORY-019.AC-2" in body.splitlines()[0]
        assert "STORY-019.AC-3" in body.splitlines()[0]

    def test_prepends_typescript_marker_when_jsdoc_references_multiple_stories(
        self, tmp_path,
    ):
        self._seed_story(str(tmp_path), "STORY-002", ["a", "b"])
        self._seed_story(str(tmp_path), "STORY-003", ["c"])
        test_file = tmp_path / "client" / "src" / "__tests__" / "FilingList.test.tsx"
        test_file.parent.mkdir(parents=True)
        test_file.write_text(
            "/** Unit tests.\n"
            " *\n"
            " * # STORY-002: Filing Index\n"
            " * # STORY-003: Date Range\n"
            " */\n"
            "it('does something', () => {});\n"
        )
        _, patched = autofix_markers_by_body_reference(str(tmp_path))
        assert patched == 1
        first_line = test_file.read_text().splitlines()[0]
        assert first_line.startswith("// @verifies: STORY-002.AC-1")
        # Both stories' ACs get merged into one marker.
        assert "STORY-003.AC-1" in first_line

    def test_respects_specific_ac_mentions_over_full_story_expansion(self, tmp_path):
        self._seed_story(str(tmp_path), "STORY-005", ["a", "b", "c"])
        test_file = tmp_path / "tests" / "test_scoped.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text(
            "# This only covers STORY-005.AC-2 specifically\n"
            "def test_y():\n    pass\n"
        )
        _, patched = autofix_markers_by_body_reference(str(tmp_path))
        assert patched == 1
        first_line = test_file.read_text().splitlines()[0]
        # Only AC-2 is cited — the specific mention wins over full-story
        # expansion. AC-1 and AC-3 are NOT attached.
        assert first_line == "# @verifies: STORY-005.AC-2"

    def test_skips_files_already_carrying_markers(self, tmp_path):
        self._seed_story(str(tmp_path), "STORY-010", ["a"])
        test_file = tmp_path / "tests" / "test_ok.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text(
            "# @verifies: STORY-010.AC-1\n# Also references STORY-010\n"
            "def test_z():\n    pass\n"
        )
        _, patched = autofix_markers_by_body_reference(str(tmp_path))
        assert patched == 0
        # File content untouched.
        assert test_file.read_text().splitlines()[0] == "# @verifies: STORY-010.AC-1"

    def test_skips_files_with_no_story_reference(self, tmp_path):
        self._seed_story(str(tmp_path), "STORY-020", ["a"])
        test_file = tmp_path / "tests" / "test_unrelated.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("def test_a(): assert 1 == 1\n")
        _, patched = autofix_markers_by_body_reference(str(tmp_path))
        assert patched == 0

    def test_ignores_story_mentions_below_scan_window(self, tmp_path):
        """A stray STORY-N mention deep in the file body (e.g. a mocked
        fixture name that happens to match the pattern) shouldn't drive
        the inference — the marker convention is "top of file"."""
        self._seed_story(str(tmp_path), "STORY-030", ["a"])
        test_file = tmp_path / "tests" / "test_deep.py"
        test_file.parent.mkdir(parents=True)
        # 60 blank lines, then a story mention. Scan window is 50 lines.
        test_file.write_text("\n" * 60 + "# STORY-030 in a comment far below.\n")
        _, patched = autofix_markers_by_body_reference(str(tmp_path))
        assert patched == 0


# ---------------------------------------------------------------------------
# v5 Phase 6 — non-agile mode skips the @verifies machinery
# ---------------------------------------------------------------------------

class TestTestsMarkerRuleInPrompt:
    """Unit-test contract: RULE 5 is the ``@tests`` code-linkage marker
    and is emitted in EVERY flow — linking a unit test to the source
    file it exercises needs no story/AC rows, so the old agile /
    non-agile split is gone. The prompt must never instruct the LLM to
    cite AC keys."""

    @pytest.mark.asyncio
    async def test_tests_rule_present_in_non_agile_prompt(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n")
        gw = stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_calc.py\n"
            "content:\n"
            "from calc import add\n"
            "def test_add(): assert add(1, 2) == 3\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(0, "1 passed")
        await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["calc.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
            # decomposition_enabled deliberately False / unset.
        })
        sent = gw.dispatched[0]["messages"]
        joined = "\n".join(m.get("content", "") for m in sent if m.get("role") == "user")
        assert "@tests:" in joined
        # The prompt must not demand AC citations anywhere.
        assert "# @verifies: STORY" not in joined
        assert "STORY-003.AC-2" not in joined

    @pytest.mark.asyncio
    async def test_markerless_test_accepted_in_non_agile(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        """A markerless test still lands as ``status=passed`` in
        non-agile runs (the @tests autofix supplies the marker), and no
        AC-link fields ever appear in node_state."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n")
        stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_calc.py\n"
            "content:\n"
            "from calc import add\n"
            "def test_add(): assert add(1, 2) == 3\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(0, "1 passed in 0.01s")
        result = await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["calc.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
        })
        assert "compiler_errors" not in result
        assert result["node_state"]["test_generation"]["status"] == "passed"
        # Agile-only fields are absent in non-agile node_state.
        assert "verifies_links_inserted" not in result["node_state"]["test_generation"]
        assert "verifies_links_dropped" not in result["node_state"]["test_generation"]
        assert route_after_test_generation(result) == "lintgate_node"

    @pytest.mark.asyncio
    async def test_tests_rule_present_in_agile_prompt(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        """Agile runs get the same code-linkage rule — stories change
        nothing about how unit tests are linked."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n")
        gw = stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_calc.py\n"
            "content:\n"
            "# @tests: calc.py\n"
            "from calc import add\n"
            "def test_add(): assert add(1, 2) == 3\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(0, "1 passed")
        await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["calc.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
            "decomposition_enabled": True,
        })
        sent = gw.dispatched[0]["messages"]
        joined = "\n".join(m.get("content", "") for m in sent if m.get("role") == "user")
        assert "@tests:" in joined
        assert "# @verifies: STORY" not in joined
        assert "STORY-003.AC-2" not in joined


class TestStoryPreambleInjectedIntoTestGenPrompt:
    """Phase 6b: _build_story_preamble is now prepended to the test-gen
    user prompt alongside the change-request and arch-summary preambles
    so the LLM actually sees the AC keys it's expected to cite.

    In non-agile / no-current-story mode the preamble is empty, so the
    prompt picks up zero extra content (the bytes match the pre-Phase-6
    non-preamble form).
    """

    @pytest.mark.asyncio
    async def test_preamble_present_when_current_story_active(
        self, tmp_path, stub_sandbox, stub_gateway, monkeypatch,
    ):
        from harness import story_state
        # Seed an agile workspace with a real STORY-001 + AC.
        ws = tmp_path / "agile-preamble-ws"
        ws.mkdir()
        (ws / "pyproject.toml").write_text("[project]\nname='x'\n")
        (ws / "calc.py").write_text("def add(a, b): return a + b\n")
        db = tmp_path / "state.db"
        monkeypatch.setenv("TEANE_STATE_DB", str(db))
        app = story_state.app_name_for_workspace(str(ws))
        conn = story_state.open_story_db()
        try:
            story_state.ensure_feature(conn, app, "core", name="Core")
            story_state.create_stories(conn, app, [{
                "title": "Add two numbers", "feature": "core",
                "acceptance_criteria": ["add(1, 2) returns 3"],
            }])
        finally:
            conn.close()

        gw = stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_calc.py\n"
            "content:\n"
            "# @verifies: STORY-001.AC-1\n"
            "from calc import add\n"
            "def test_add(): assert add(1, 2) == 3\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(0, "1 passed")
        await run_test_generation({
            "workspace_path": str(ws),
            "modified_files": ["calc.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
            "decomposition_enabled": True,
            "current_story_id": "STORY-001",
        })
        sent = gw.dispatched[0]["messages"]
        joined = "\n".join(m.get("content", "") for m in sent if m.get("role") == "user")
        # Story preamble rendered with the AC key so the LLM can cite it.
        assert "STORY-001.AC-1" in joined
        assert "add(1, 2) returns 3" in joined

    @pytest.mark.asyncio
    async def test_no_batch_scope_preamble_in_batch_verification(
        self, tmp_path, stub_sandbox, stub_gateway, monkeypatch,
    ):
        """Unit-test model: the batch-scope AC preamble is gone. In the
        per-batch verification phase (current_story_id cleared) the
        test-gen prompt carries the modified source files and the
        @tests contract — no story/AC keys are injected, because unit
        tests never cite them."""
        from harness import story_state
        ws = tmp_path / "batch-preamble-ws"
        ws.mkdir()
        (ws / "pyproject.toml").write_text("[project]\nname='x'\n")
        (ws / "calc.py").write_text("def add(a, b): return a + b\n")
        db = tmp_path / "state.db"
        monkeypatch.setenv("TEANE_STATE_DB", str(db))
        app = story_state.app_name_for_workspace(str(ws))
        conn = story_state.open_story_db()
        try:
            story_state.ensure_feature(conn, app, "core", name="Core")
            story_state.create_stories(conn, app, [
                {
                    "title": "Add",
                    "feature": "core",
                    "acceptance_criteria": ["add(1, 2) returns 3"],
                },
                {
                    "title": "Subtract",
                    "feature": "core",
                    "acceptance_criteria": ["sub(2, 1) returns 1"],
                },
            ])
        finally:
            conn.close()

        gw = stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_calc.py\n"
            "content:\n"
            "# @tests: calc.py\n"
            "from calc import add\n"
            "def test_add(): assert add(1, 2) == 3\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(0, "1 passed")
        await run_test_generation({
            "workspace_path": str(ws),
            "modified_files": ["calc.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
            "decomposition_enabled": True,
            # Mid-verification state: story_loop already cleared this.
            "current_story_id": "",
            "current_batch_id": 1,
            "batch_patched_story_keys": ["STORY-001", "STORY-002"],
        })
        sent = gw.dispatched[0]["messages"]
        joined = "\n".join(m.get("content", "") for m in sent if m.get("role") == "user")
        # No AC keys injected — unit tests never cite them.
        assert "Batch Scope:" not in joined
        assert "STORY-001.AC-1" not in joined
        assert "STORY-002.AC-1" not in joined
        # The code-linkage contract is still there.
        assert "@tests:" in joined

    @pytest.mark.asyncio
    async def test_preamble_empty_when_no_current_story(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n")
        gw = stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_calc.py\n"
            "content:\n"
            "from calc import add\n"
            "def test_add(): assert add(1, 2) == 3\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(0, "1 passed")
        await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["calc.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
            # no current_story_id, no decomposition_enabled
        })
        sent = gw.dispatched[0]["messages"]
        joined = "\n".join(m.get("content", "") for m in sent if m.get("role") == "user")
        # Story preamble renders the empty string when no story is set —
        # the prompt has no "Story Scope:" header.
        assert "Story Scope:" not in joined


# ---------------------------------------------------------------------------
# Fix 2c (2026-07-10): the test-gen format reminder must document all four
# patch block types. Prior to this fix only CREATE_FILE and INSERT_AT_BLOCK
# were listed, but the LLM sees REPLACE_BLOCK examples in the messages
# history from patching_node — mismatch caused iter 4 of session 44c5e194
# to emit 5 REPLACE_BLOCKs that all rejected as "unknown format".
# ---------------------------------------------------------------------------

class TestFormatReminderDocumentsAllBlockTypes:

    def test_reminder_lists_all_four_patch_ops(self):
        from harness.test_generation import _PROMPT_FORMAT_REMINDER_BASE
        assert "<<<CREATE_FILE>>>" in _PROMPT_FORMAT_REMINDER_BASE
        assert "<<<REPLACE_BLOCK>>>" in _PROMPT_FORMAT_REMINDER_BASE
        assert "<<<REWRITE_FILE>>>" in _PROMPT_FORMAT_REMINDER_BASE
        assert "<<<INSERT_AT_BLOCK>>>" in _PROMPT_FORMAT_REMINDER_BASE

    def test_reminder_explains_when_to_use_each_op(self):
        # The "CHOOSING THE RIGHT BLOCK:" section is what steers the LLM
        # away from REPLACE_BLOCK-when-file-is-empty (Fix 4 bait).
        from harness.test_generation import _PROMPT_FORMAT_REMINDER_BASE
        assert "CHOOSING THE RIGHT BLOCK" in _PROMPT_FORMAT_REMINDER_BASE
        assert "REWRITE_FILE" in _PROMPT_FORMAT_REMINDER_BASE
        assert "small" in _PROMPT_FORMAT_REMINDER_BASE.lower()


# ---------------------------------------------------------------------------
# Fix 3 (2026-07-10): missing @verifies marker is autofixed deterministically
# from the current story's AC keys rather than routed to LLM repair. Only
# markerless files WITHOUT usable story context still route to repair.
# ---------------------------------------------------------------------------

class TestVerifiesMarkerAutofix:

    def test_marker_line_for_python_uses_hash_lead(self):
        from harness.test_generation import _marker_line_for
        line = _marker_line_for("python", ["STORY-3.AC-1", "STORY-3.AC-2"])
        assert line == "# @verifies: STORY-3.AC-1, STORY-3.AC-2"

    def test_marker_line_for_typescript_uses_slash_lead(self):
        from harness.test_generation import _marker_line_for
        line = _marker_line_for("typescript", ["STORY-3.AC-1"])
        assert line == "// @verifies: STORY-3.AC-1"

    def test_marker_line_for_no_keys_returns_none(self):
        from harness.test_generation import _marker_line_for
        assert _marker_line_for("python", []) is None

    def test_marker_line_for_drops_malformed_keys(self):
        # Bad keys are silently filtered — the persist gate would drop
        # them downstream anyway, so autofix shouldn't waste I/O
        # writing them.
        from harness.test_generation import _marker_line_for
        assert _marker_line_for("python", ["bogus", "STORY-1.AC-2"]) == (
            "# @verifies: STORY-1.AC-2"
        )
        assert _marker_line_for("python", ["bogus", "also-bad"]) is None

    def test_prepend_marker_writes_at_top_of_file(self, tmp_path):
        from harness.test_generation import _prepend_verifies_marker
        f = tmp_path / "test_x.py"
        f.write_text("def test_x(): pass\n")
        assert _prepend_verifies_marker(
            str(f), "# @verifies: STORY-1.AC-1",
        ) is True
        body = f.read_text()
        assert body.startswith("# @verifies: STORY-1.AC-1\n")
        assert "def test_x()" in body

    def test_prepend_marker_respects_shebang(self, tmp_path):
        from harness.test_generation import _prepend_verifies_marker
        f = tmp_path / "run.py"
        f.write_text("#!/usr/bin/env python3\ndef test_x(): pass\n")
        _prepend_verifies_marker(str(f), "# @verifies: STORY-1.AC-1")
        lines = f.read_text().splitlines()
        assert lines[0] == "#!/usr/bin/env python3"
        assert lines[1] == "# @verifies: STORY-1.AC-1"

    def test_prepend_marker_idempotent_when_already_present(self, tmp_path):
        # Autofix is called from a loop; running it twice must not
        # duplicate the marker (and must not write the file again).
        from harness.test_generation import _prepend_verifies_marker
        f = tmp_path / "test_x.py"
        f.write_text("# @verifies: STORY-1.AC-1\ndef test_x(): pass\n")
        assert _prepend_verifies_marker(
            str(f), "# @verifies: STORY-1.AC-1",
        ) is True
        body = f.read_text()
        # Only one marker line in the file
        assert body.count("@verifies:") == 1


# ---------------------------------------------------------------------------
# Fix 5a (2026-07-10): the test-gen user prompt must include the current
# on-disk bytes of every existing test file the LLM might edit. Without
# this, REPLACE_BLOCK anchors are built from the LLM's stale mental model
# (root cause behind iter 4 of session 44c5e194).
# ---------------------------------------------------------------------------

class TestPreflightInjectionInTestGenPrompt:

    @pytest.mark.asyncio
    async def test_existing_test_file_body_appears_in_user_prompt(
        self, tmp_path, stub_sandbox, stub_gateway,
    ):
        # Source file being tested this round.
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n")
        # A pre-existing test file that shares the conventional name —
        # the harness must show its current body to the LLM.
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_calc.py").write_text(
            "# @verifies: STORY-1.AC-1\n"
            "def test_add_returns_sum():\n"
            "    from calc import add\n"
            "    assert add(2, 3) == 5\n"
        )
        gw = stub_gateway(
            "<<<CREATE_FILE>>>\n"
            "file: tests/test_calc_extra.py\n"
            "content:\n"
            "def test_extra(): pass\n"
            "<<<END_CREATE_FILE>>>\n"
        )
        stub_sandbox(0, "1 passed")
        await run_test_generation({
            "workspace_path": str(tmp_path),
            "modified_files": ["calc.py", "tests/test_calc.py"],
            "messages": [],
            "budget_remaining_usd": 1.5,
            "token_tracker": {},
        })
        sent = gw.dispatched[0]["messages"]
        joined = "\n".join(
            m.get("content", "") for m in sent if m.get("role") == "user"
        )
        # Preflight section header appears
        assert "Current Content of Files You Need to Edit" in joined
        # And carries the ACTUAL test file body (not the LLM's memory of it)
        assert "test_add_returns_sum" in joined
        assert "assert add(2, 3) == 5" in joined
        # Line-numbered rendering (the `  N| ` prefix from _render_file...)
        assert "1| " in joined or "1|" in joined


# ---------------------------------------------------------------------------
# Fix 2a (2026-07-10) — POSITIVE path: zero-emit retry succeeds on second
# response. Confirms the counter split: test_generation_zero_emit=1,
# test_generation=1 (only the successful attempt is counted).
# ---------------------------------------------------------------------------

class TestZeroEmitRetrySucceeds:

    @pytest.mark.asyncio
    async def test_reprompt_then_valid_patch_lands_and_counters_split(
        self, tmp_path, stub_sandbox, monkeypatch,
    ):
        # Custom stub gateway that returns different content on each call.
        # First call: zero patch blocks. Second call: a valid patch.
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "foo.py").write_text("def foo(): return 1\n")

        class _MultiResponseGateway(_StubGateway):
            def __init__(self):
                super().__init__("")
                self._responses = [
                    "no patch blocks here — just prose",
                    "<<<CREATE_FILE>>>\n"
                    "file: tests/test_foo.py\n"
                    "content:\n"
                    "# @verifies: STORY-1.AC-1\n"
                    "def test_foo(): assert True\n"
                    "<<<END_CREATE_FILE>>>\n",
                ]

            async def dispatch(self, *, messages, role, budget_remaining_usd, **kwargs):
                self.dispatched.append({"messages": list(messages), "role": role})
                idx = min(len(self.dispatched) - 1, len(self._responses) - 1)
                return _StubResponse(self._responses[idx]), (
                    budget_remaining_usd - 0.001
                )

        from harness import graph as graph_mod
        gw = _MultiResponseGateway()
        graph_mod.set_gateway(gw)
        try:
            stub_sandbox(0, "1 passed")
            result = await run_test_generation({
                "workspace_path": str(tmp_path),
                "modified_files": ["foo.py"],
                "messages": [],
                "budget_remaining_usd": 1.5,
                "token_tracker": {},
            })
        finally:
            graph_mod.set_gateway(None)

        # Two dispatches: the first was the zero-emit re-prompt, the
        # second landed a real patch.
        assert len(gw.dispatched) == 2
        # The zero-emit counter recorded the single retry.
        assert result["loop_counter"]["test_generation_zero_emit"] == 1
        # The REAL iteration counter only advanced for the successful
        # attempt — that's Fix 2a's whole point.
        assert result["loop_counter"]["test_generation"] == 1
        # Second dispatch's messages must include the stronger contract
        # system message pushed after the zero-emit response.
        second_msgs = gw.dispatched[1]["messages"]
        stronger_prompt_hits = [
            m for m in second_msgs
            if m.get("role") == "system"
            and "zero PATCH blocks" in m.get("content", "")
        ]
        assert stronger_prompt_hits, (
            "second dispatch must carry the stronger re-prompt system "
            "message pushed after the first zero-emit response"
        )


class TestTestsMarkerHelpers:
    """Direct unit coverage for the @tests code-linkage helpers."""

    def test_parse_tests_marker_python_style(self):
        from harness.test_generation import _parse_tests_marker
        body = '"""Docstring."""\n# @tests: server/app/billing.py, server/app/tax.py\nimport x\n'
        assert _parse_tests_marker(body) == [
            "server/app/billing.py", "server/app/tax.py",
        ]

    def test_parse_tests_marker_js_style(self):
        from harness.test_generation import _parse_tests_marker
        body = "// @tests: client/src/utils/fmt.ts\nimport { fmt } from '../fmt';\n"
        assert _parse_tests_marker(body) == ["client/src/utils/fmt.ts"]

    def test_parse_tests_marker_absent(self):
        from harness.test_generation import _parse_tests_marker
        assert _parse_tests_marker("def test_x(): pass\n") == []
        assert _parse_tests_marker("") == []

    def test_parse_tests_marker_outside_scan_window_ignored(self):
        from harness.test_generation import _parse_tests_marker
        body = "\n" * 60 + "# @tests: a.py\n"
        assert _parse_tests_marker(body) == []

    def test_marker_line_uses_stack_comment_lead(self):
        from harness.test_generation import _tests_marker_line_for
        assert _tests_marker_line_for("python", ["a.py"]) == "# @tests: a.py"
        assert _tests_marker_line_for("typescript", ["b.ts"]) == "// @tests: b.ts"
        assert _tests_marker_line_for("python", []) is None

    def test_guess_sources_prefers_basename_match(self):
        from harness.test_generation import _guess_sources_for_test
        sources = ["server/app/billing.py", "server/app/tax.py"]
        assert _guess_sources_for_test(
            "tests/test_billing.py", sources,
        ) == ["server/app/billing.py"]
        assert _guess_sources_for_test(
            "client/src/fmt.test.ts", ["client/src/fmt.ts", "client/src/other.ts"],
        ) == ["client/src/fmt.ts"]

    def test_guess_sources_falls_back_to_generation_scope(self):
        from harness.test_generation import _guess_sources_for_test
        sources = ["a.py", "b.py", "c.py", "d.py"]
        assert _guess_sources_for_test("tests/test_misc.py", sources) == [
            "a.py", "b.py", "c.py",
        ]
