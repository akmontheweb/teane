"""Emission continuation + scope-coverage guard in patching_node.

Regression suite for lumina 019f71b0: the tool-use patching path
applied only the FIRST response's patch calls and declared the story
patched, so a model that emits a few files per response (as the
prompt's ≤3-files-per-response rule mandates) landed nothing but its
opening scaffold — three stories in a row "succeeded" with zero scope
files on disk, and the batch reached verification with no feature
code.

Covers:
  * the apply → tool_result → re-dispatch emission loop and its exit
    conditions (model stops, round cap, two-round futility brake);
  * write-credits letting a round-2 ``edit_file`` land on a round-1
    ``create_file`` despite ``enforce_read_before_edit``;
  * the scope-coverage guard demoting an all-peripheral round to
    zero-patch (and staying silent on stem-matched or on-disk scope);
  * the ``_build_patch_tool_results`` per-call mapping and the
    ``_resolve_patching_emission_rounds`` config clamp.
"""

import asyncio

import pytest

from harness import graph as graph_mod
from harness.graph import (
    _build_patch_tool_results,
    _resolve_patching_emission_rounds,
    _story_scope_files_uncovered,
)
from harness.patcher import OperationType, PatchResult


@pytest.fixture(autouse=True)
def _restore_gateway():
    """Save/restore the module-global gateway around every test.

    ``_install_gateway`` swaps in a stub via ``set_gateway``; without
    this teardown the stub leaks into whichever test file pytest
    collects next (alphabetically test_presets / test_repair_loop_* /
    test_setup_script / test_web_wizard), and any code there that
    calls ``get_gateway()`` sees a config-less stub and fails in ways
    that vanish under isolated runs.
    """
    prior = graph_mod.get_gateway()
    yield
    graph_mod.set_gateway(prior)


class _Usage:
    input_tokens = 100
    output_tokens = 200
    cost_usd = 0.001
    cached_tokens = 0


class _Response:
    """Gateway-response stand-in: content, finish_reason, tool_calls, usage."""

    def __init__(self, content="", tool_calls=None, finish_reason="stop"):
        self.content = content
        self.finish_reason = finish_reason
        self.tool_calls = list(tool_calls or [])
        self.usage = _Usage()


def _create_call(call_id, path, content="x = 1\n"):
    return {
        "id": call_id,
        "name": "create_file",
        "input": {"file_path": path, "content": content},
    }


def _edit_call(call_id, path, old, new):
    return {
        "id": call_id,
        "name": "edit_file",
        "input": {"file_path": path, "old_string": old, "new_string": new},
    }


def _install_gateway(monkeypatch, *, enforce_read_before_edit=True):
    class _Cfg:
        use_structured_tools = True

    _Cfg.enforce_read_before_edit = enforce_read_before_edit

    class _Gw:
        config = _Cfg()

        def aggregate_tokens(self, tracker, usage, role=None):
            out = dict(tracker or {})
            out["total_cost_usd"] = (
                out.get("total_cost_usd", 0.0) + usage.cost_usd
            )
            return out

    graph_mod.set_gateway(_Gw())
    monkeypatch.setattr(graph_mod, "_build_patcher_allowlist", lambda ws: [])
    return graph_mod


def _install_tool_loop(monkeypatch, responses):
    """Monkeypatch ``_patching_tool_loop`` to pop scripted responses.

    Returns the list that records one entry per dispatch (the messages
    snapshot each call saw).
    """
    seen: list[list] = []

    async def fake_tool_loop(**kwargs):
        seen.append(list(kwargs["messages"]))
        resp = responses.pop(0)
        return resp, kwargs["budget"] - 0.01, kwargs["messages"], {}

    monkeypatch.setattr(graph_mod, "_patching_tool_loop", fake_tool_loop)
    return seen


def _state(tmp_path, **extra):
    state = {
        "messages": [{"role": "system", "content": "you are a patcher"}],
        "budget_remaining_usd": 2.0,
        "workspace_path": str(tmp_path),
        "modified_files": [],
        "loop_counter": {},
        "token_tracker": {},
    }
    state.update(extra)
    return state


class TestEmissionContinuation:
    def test_multi_round_emission_lands_every_file(self, monkeypatch, tmp_path):
        """Round 1 emits two files, round 2 one more, round 3 is a pure
        text sign-off — all three files must reach disk (pre-fix, only
        round 1's landed and rounds 2-3 never happened)."""
        _install_gateway(monkeypatch)
        responses = [
            _Response(
                "scaffolding first",
                tool_calls=[
                    _create_call("c1", "server/a.py"),
                    _create_call("c2", "server/b.py"),
                ],
                finish_reason="tool_calls",
            ),
            _Response(
                "one more",
                tool_calls=[_create_call("c3", "server/c.py")],
                finish_reason="tool_calls",
            ),
            _Response("Story complete.", finish_reason="stop"),
        ]
        dispatches = _install_tool_loop(monkeypatch, responses)

        result = asyncio.run(graph_mod.patching_node(_state(tmp_path)))

        assert (tmp_path / "server" / "a.py").is_file()
        assert (tmp_path / "server" / "b.py").is_file()
        assert (tmp_path / "server" / "c.py").is_file()
        assert len(dispatches) == 3
        assert result["node_state"]["patch_success"] == 3
        assert sorted(result["modified_files"]) == [
            "server/a.py", "server/b.py", "server/c.py",
        ]

    def test_tool_results_answer_every_call_id(self, monkeypatch, tmp_path):
        """Each continuation turn must carry one tool_result per
        tool_use id (OpenAI-compatible providers 400 on any gap), with
        the harness continuation note on the last one."""
        _install_gateway(monkeypatch)
        responses = [
            _Response(
                tool_calls=[
                    _create_call("c1", "server/a.py"),
                    _create_call("c2", "server/b.py"),
                ],
                finish_reason="tool_calls",
            ),
            _Response("done", finish_reason="stop"),
        ]
        _install_tool_loop(monkeypatch, responses)

        result = asyncio.run(graph_mod.patching_node(_state(tmp_path)))

        tool_result_turns = [
            m for m in result["messages"]
            if isinstance(m, dict)
            and m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in m["content"]
            )
        ]
        assert len(tool_result_turns) == 1
        blocks = tool_result_turns[0]["content"]
        assert [b["tool_use_id"] for b in blocks] == ["c1", "c2"]
        assert all("Applied" in b["content"] for b in blocks)
        assert "emission round 1" in blocks[-1]["content"]

    def test_round_two_edit_lands_on_round_one_create(
        self, monkeypatch, tmp_path,
    ):
        """Write-credits: with ``enforce_read_before_edit`` on, an
        ``edit_file`` against a file the model created one round
        earlier must pass B5 — the model wrote those exact bytes."""
        _install_gateway(monkeypatch, enforce_read_before_edit=True)
        responses = [
            _Response(
                tool_calls=[_create_call("c1", "server/a.py", "x = 1\n")],
                finish_reason="tool_calls",
            ),
            _Response(
                tool_calls=[_edit_call("c2", "server/a.py", "x = 1", "x = 2")],
                finish_reason="tool_calls",
            ),
            _Response("done", finish_reason="stop"),
        ]
        _install_tool_loop(monkeypatch, responses)

        result = asyncio.run(graph_mod.patching_node(_state(tmp_path)))

        assert (tmp_path / "server" / "a.py").read_text().rstrip() == "x = 2"
        assert result["node_state"]["patch_success"] == 2
        assert result["node_state"]["patch_fail"] == 0

    def test_emission_cap_bounds_the_loop(self, monkeypatch, tmp_path):
        """A model that never stops emitting patch calls is cut off at
        ``llm_dispatch.patching_emission_rounds`` applied rounds."""
        _install_gateway(monkeypatch)
        counter = {"n": 0}

        async def endless_tool_loop(**kwargs):
            counter["n"] += 1
            return (
                _Response(
                    tool_calls=[
                        _create_call(
                            f"c{counter['n']}",
                            f"server/f{counter['n']}.py",
                        ),
                    ],
                    finish_reason="tool_calls",
                ),
                kwargs["budget"] - 0.01,
                kwargs["messages"],
                {},
            )

        monkeypatch.setattr(
            graph_mod, "_patching_tool_loop", endless_tool_loop,
        )

        state = _state(
            tmp_path, llm_dispatch_config={"patching_emission_rounds": 3},
        )
        result = asyncio.run(graph_mod.patching_node(state))

        # Cap of 3 applied rounds → exactly 3 dispatches, 3 files.
        assert counter["n"] == 3
        assert result["node_state"]["patch_success"] == 3

    def test_two_futile_rounds_stop_continuation(self, monkeypatch, tmp_path):
        """Two consecutive rounds with zero real successes end the loop
        (here: edits against a file the model was never shown, rejected
        by read-before-edit both times)."""
        _install_gateway(monkeypatch, enforce_read_before_edit=True)
        counter = {"n": 0}

        async def futile_tool_loop(**kwargs):
            counter["n"] += 1
            return (
                _Response(
                    tool_calls=[
                        _edit_call(
                            f"c{counter['n']}", "server/ghost.py", "a", "b",
                        ),
                    ],
                    finish_reason="tool_calls",
                ),
                kwargs["budget"] - 0.01,
                kwargs["messages"],
                {},
            )

        monkeypatch.setattr(
            graph_mod, "_patching_tool_loop", futile_tool_loop,
        )

        result = asyncio.run(graph_mod.patching_node(_state(tmp_path)))

        assert counter["n"] == 2
        assert result["node_state"]["patch_success"] == 0
        assert result["loop_counter"]["consecutive_zero_patch_rounds"] == 1

    def test_continuation_turns_survive_openai_normalizer(
        self, monkeypatch, tmp_path,
    ):
        """The emission loop's assistant/tool_result turns must satisfy
        the OpenAI-shape pairing invariant after gateway normalization —
        every assistant tool_call id answered by exactly one role=tool
        message, no orphans (the 400 class from commit 63f2b50). This
        pins the real provider boundary, not just the canonical shape."""
        from harness.gateway import _normalize_messages_for_openai_tools

        _install_gateway(monkeypatch)
        responses = [
            _Response(
                "part one",
                tool_calls=[
                    _create_call("c1", "server/a.py"),
                    _create_call("c2", "server/b.py"),
                ],
                finish_reason="tool_calls",
            ),
            _Response(
                tool_calls=[_create_call("c3", "server/c.py")],
                finish_reason="tool_calls",
            ),
            _Response("done", finish_reason="stop"),
        ]
        _install_tool_loop(monkeypatch, responses)

        result = asyncio.run(graph_mod.patching_node(_state(tmp_path)))

        normalized = _normalize_messages_for_openai_tools(
            [dict(m) for m in result["messages"]],
        )
        pending: set[str] = set()
        tool_turns_seen = 0
        for msg in normalized:
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                assert not pending, (
                    f"new assistant tool turn while ids {pending} unanswered"
                )
                pending = {c["id"] for c in msg["tool_calls"]}
                tool_turns_seen += 1
            elif role == "tool":
                call_id = msg.get("tool_call_id")
                assert call_id in pending, (
                    f"orphan role=tool message for id {call_id!r}"
                )
                pending.discard(call_id)
            else:
                assert not pending, (
                    f"{role} message interleaved while ids {pending} unanswered"
                )
        assert not pending, f"unanswered tool_call ids at tail: {pending}"
        # Guard against vacuous passes: rounds 1 and 2 each left an
        # assistant tool turn in history (round 3 was pure text).
        assert tool_turns_seen == 2

    def test_single_text_response_unchanged(self, monkeypatch, tmp_path):
        """A pure-text first response never enters the emission loop —
        the legacy text-DSL path handles it exactly as before."""
        _install_gateway(monkeypatch)
        responses = [
            _Response(
                "<<<CREATE_FILE>>>\nfile: server/a.py\ncontent:\nx\n"
                "<<<END_CREATE_FILE>>>",
                finish_reason="stop",
            ),
        ]
        dispatches = _install_tool_loop(monkeypatch, responses)

        result = asyncio.run(graph_mod.patching_node(_state(tmp_path)))

        assert len(dispatches) == 1
        assert (tmp_path / "server" / "a.py").is_file()
        assert result["node_state"]["patch_success"] == 1


class TestScopeCoverageGuard:
    def _story_state(self, tmp_path, **extra):
        return _state(
            tmp_path,
            current_story_id="STORY-002",
            current_batch_id=1,
            story_scope_files=[
                "server/app/api/contacts.py",
                "client/src/components/AddEditForm.jsx",
            ],
            **extra,
        )

    def test_all_peripheral_round_demoted_to_zero_patch(
        self, monkeypatch, tmp_path,
    ):
        """The lumina signature: patches landed, but none of the
        story's scope files — demote so story_loop re-picks, and leave
        the directive naming the missing files."""
        _install_gateway(monkeypatch)
        responses = [
            _Response(
                tool_calls=[
                    _create_call("c1", "requirements.txt", "fastapi\n"),
                    _create_call("c2", "server/app/__init__.py", ""),
                ],
                finish_reason="tool_calls",
            ),
            _Response("done", finish_reason="stop"),
        ]
        _install_tool_loop(monkeypatch, responses)

        result = asyncio.run(
            graph_mod.patching_node(self._story_state(tmp_path)),
        )

        assert result["node_state"]["patch_success"] == 0
        assert (
            result["loop_counter"]["story_zero_patch_rounds"]["STORY-002"]
            == 1
        )
        directives = [
            m for m in result["messages"]
            if isinstance(m, dict)
            and isinstance(m.get("content"), str)
            and "[Scope-coverage guard]" in m["content"]
        ]
        assert len(directives) == 1
        assert "server/app/api/contacts.py" in directives[0]["content"]

    def test_stem_match_counts_as_coverage(self, monkeypatch, tmp_path):
        """AddEditForm.tsx covers the decomposer's AddEditForm.jsx hint
        — a legitimate stack-driven rename must not demote."""
        _install_gateway(monkeypatch)
        responses = [
            _Response(
                tool_calls=[
                    _create_call(
                        "c1", "client/src/components/AddEditForm.tsx",
                    ),
                ],
                finish_reason="tool_calls",
            ),
            _Response("done", finish_reason="stop"),
        ]
        _install_tool_loop(monkeypatch, responses)

        result = asyncio.run(
            graph_mod.patching_node(self._story_state(tmp_path)),
        )

        assert result["node_state"]["patch_success"] == 1
        assert not any(
            isinstance(m, dict)
            and isinstance(m.get("content"), str)
            and "[Scope-coverage guard]" in m["content"]
            for m in result["messages"]
        )

    def test_scope_file_on_disk_counts_as_coverage(
        self, monkeypatch, tmp_path,
    ):
        """An edit-shaped story whose scope files already exist (built
        by an earlier story) is not demoted for touching other files."""
        scope_file = tmp_path / "server" / "app" / "api" / "contacts.py"
        scope_file.parent.mkdir(parents=True)
        scope_file.write_text("# existing\n")
        _install_gateway(monkeypatch)
        responses = [
            _Response(
                tool_calls=[_create_call("c1", "server/app/helper.py")],
                finish_reason="tool_calls",
            ),
            _Response("done", finish_reason="stop"),
        ]
        _install_tool_loop(monkeypatch, responses)

        result = asyncio.run(
            graph_mod.patching_node(self._story_state(tmp_path)),
        )

        assert result["node_state"]["patch_success"] == 1

    def test_monolithic_mode_never_demotes(self, monkeypatch, tmp_path):
        """No story cursor → no scope contract → guard stays out of
        the way entirely."""
        _install_gateway(monkeypatch)
        responses = [
            _Response(
                tool_calls=[_create_call("c1", "requirements.txt")],
                finish_reason="tool_calls",
            ),
            _Response("done", finish_reason="stop"),
        ]
        _install_tool_loop(monkeypatch, responses)

        result = asyncio.run(graph_mod.patching_node(_state(tmp_path)))

        assert result["node_state"]["patch_success"] == 1


class TestScopeUncoveredHelper:
    def test_returns_scope_only_when_nothing_covered(self, tmp_path):
        state = {
            "current_story_id": "STORY-005",
            "story_scope_files": ["server/x.py", "server/y.py"],
        }
        assert _story_scope_files_uncovered(
            state, str(tmp_path), ["requirements.txt"],
        ) == ["server/x.py", "server/y.py"]

    def test_partial_coverage_is_silent(self, tmp_path):
        state = {
            "current_story_id": "STORY-005",
            "story_scope_files": ["server/x.py", "server/y.py"],
        }
        assert _story_scope_files_uncovered(
            state, str(tmp_path), ["server/x.py"],
        ) == []

    def test_empty_scope_is_silent(self, tmp_path):
        state = {"current_story_id": "STORY-005", "story_scope_files": []}
        assert _story_scope_files_uncovered(state, str(tmp_path), []) == []

    def test_stem_match_is_case_insensitive(self, tmp_path):
        state = {
            "current_story_id": "STORY-005",
            "story_scope_files": ["client/src/AddEditForm.jsx"],
        }
        assert _story_scope_files_uncovered(
            state, str(tmp_path), ["client/src/addeditform.tsx"],
        ) == []


class TestBuildPatchToolResults:
    def test_maps_success_failure_and_dropped(self):
        calls = [
            _create_call("c1", "server/a.py"),
            _create_call("c2", "tests/test_a.py"),
            {"id": "c3", "name": "mystery_tool", "input": {}},
        ]
        from harness.tool_schemas import tool_call_to_patch_block
        call_blocks = [
            tool_call_to_patch_block(calls[0]),
            tool_call_to_patch_block(calls[1]),
            None,
        ]
        kept = frozenset([id(call_blocks[0])])
        results = [
            PatchResult(
                success=True, file="server/a.py",
                operation=OperationType.CREATE_FILE, lines_changed=1,
            ),
        ]
        out = _build_patch_tool_results(
            calls, call_blocks, kept, results, harness_note="[Harness]: go on.",
        )
        assert [b["tool_use_id"] for b in out] == ["c1", "c2", "c3"]
        assert "Applied" in out[0]["content"]
        assert "test-artifact" in out[1]["content"]
        assert "Ignored" in out[2]["content"]
        assert out[-1]["content"].endswith("[Harness]: go on.")

    def test_failure_carries_patcher_error(self):
        calls = [_create_call("c1", "server/a.py")]
        from harness.tool_schemas import tool_call_to_patch_block
        blocks = [tool_call_to_patch_block(calls[0])]
        results = [
            PatchResult(
                success=False, file="server/a.py",
                operation=OperationType.CREATE_FILE,
                error="File already exists with different content",
            ),
        ]
        out = _build_patch_tool_results(
            calls, blocks, frozenset([id(blocks[0])]), results,
        )
        assert "REJECTED" in out[0]["content"]
        assert "already exists" in out[0]["content"]


class TestEmissionRoundsResolver:
    def test_default_when_unconfigured(self):
        assert _resolve_patching_emission_rounds({}) == 10

    def test_operator_override(self):
        state = {"llm_dispatch_config": {"patching_emission_rounds": 4}}
        assert _resolve_patching_emission_rounds(state) == 4

    def test_clamps_and_survives_garbage(self):
        assert _resolve_patching_emission_rounds(
            {"llm_dispatch_config": {"patching_emission_rounds": 999}},
        ) == 30
        assert _resolve_patching_emission_rounds(
            {"llm_dispatch_config": {"patching_emission_rounds": 0}},
        ) == 1
        assert _resolve_patching_emission_rounds(
            {"llm_dispatch_config": {"patching_emission_rounds": "nope"}},
        ) == 10
