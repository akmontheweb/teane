"""READ_FILE budget fixes from session 22471c0c.

The repair model spent its two hard-coded READ_FILE rounds walking
service → api → repository and had its third request — for
server/app/database.py, the file containing the actual root cause —
stripped by the forced-patch valve. Five fixture-shaped no-op repairs
and a reflection_distraction_loop HITL followed. Three fixes:

  1. The resolve-round cap is config-driven (llm_dispatch.read_file_rounds,
     default 6) instead of hard-coded at 2.
  2. Past the cap, one bonus resolve round is granted when the stripped
     request names a workspace file the LLM has never been shown.
  3. ``files_seen_by_llm`` keeps last-read recency order (move-to-end on
     re-read) and the post-cleanse re-injection walks it most-recent-first,
     so the cap evicts stale reads instead of the active working set.
"""

from __future__ import annotations

import json
from pathlib import Path

from harness.graph import (
    _READ_FILE_ROUNDS_DEFAULT,
    _REREAD_INJECTION_CAP,
    _resolve_read_blocks,
    _resolve_read_file_rounds,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


class TestResolveReadFileRounds:
    def test_default_is_six(self):
        assert _READ_FILE_ROUNDS_DEFAULT == 6
        assert _resolve_read_file_rounds({}) == 6

    def test_config_override(self):
        state = {"llm_dispatch_config": {"read_file_rounds": 3}}
        assert _resolve_read_file_rounds(state) == 3

    def test_clamped_to_upper_bound(self):
        state = {"llm_dispatch_config": {"read_file_rounds": 99}}
        assert _resolve_read_file_rounds(state) == 20

    def test_clamped_to_lower_bound(self):
        state = {"llm_dispatch_config": {"read_file_rounds": 0}}
        assert _resolve_read_file_rounds(state) == 1

    def test_non_int_falls_back_to_default(self):
        state = {"llm_dispatch_config": {"read_file_rounds": "lots"}}
        assert _resolve_read_file_rounds(state) == _READ_FILE_ROUNDS_DEFAULT

    def test_none_config_falls_back_to_default(self):
        assert (
            _resolve_read_file_rounds({"llm_dispatch_config": None})
            == _READ_FILE_ROUNDS_DEFAULT
        )


class TestConfigWiring:
    """The key must exist in the shipped template AND pass the CLI's
    config validator — a template key the validator rejects would fail
    every fresh install at load time."""

    def test_template_ships_the_default(self):
        cfg = json.loads(
            (_REPO_ROOT / "config" / "config.json").read_text(encoding="utf-8")
        )
        assert cfg["llm_dispatch"]["read_file_rounds"] == (
            _READ_FILE_ROUNDS_DEFAULT
        )

    def test_validator_accepts_the_key(self):
        from harness.cli import _KNOWN_NESTED_KEYS, _TYPE_SCHEMA
        assert "read_file_rounds" in _KNOWN_NESTED_KEYS["llm_dispatch"]
        assert _TYPE_SCHEMA["llm_dispatch.read_file_rounds"] == (int,)


class TestSeenFilesRecencyOrder:
    """``files_seen_by_llm`` insertion order doubles as last-read recency
    (fix 3): a re-read must move the key to the end, or the post-cleanse
    re-injection cap evicts the CURRENT investigation's files while
    keeping stale ones (session 22471c0c re-injected five files from the
    previous story every round)."""

    @staticmethod
    def _workspace_with(tmp_path, names):
        for name in names:
            (tmp_path / name).write_text(f"content of {name}\n", encoding="utf-8")
        return str(tmp_path)

    def test_first_reads_append_in_order(self, tmp_path):
        ws = self._workspace_with(tmp_path, ["a.py", "b.py"])
        seen: dict[str, str] = {}
        _resolve_read_blocks([("a.py", None)], ws, record_hashes_into=seen)
        _resolve_read_blocks([("b.py", None)], ws, record_hashes_into=seen)
        assert list(seen.keys()) == ["a.py", "b.py"]

    def test_reread_moves_key_to_end(self, tmp_path):
        ws = self._workspace_with(tmp_path, ["a.py", "b.py", "c.py"])
        seen: dict[str, str] = {}
        for name in ("a.py", "b.py", "c.py"):
            _resolve_read_blocks([(name, None)], ws, record_hashes_into=seen)
        _resolve_read_blocks([("a.py", None)], ws, record_hashes_into=seen)
        assert list(seen.keys()) == ["b.py", "c.py", "a.py"]

    def test_range_reread_also_moves_to_end(self, tmp_path):
        ws = self._workspace_with(tmp_path, ["a.py", "b.py"])
        seen: dict[str, str] = {}
        _resolve_read_blocks([("a.py", None)], ws, record_hashes_into=seen)
        _resolve_read_blocks([("b.py", None)], ws, record_hashes_into=seen)
        _resolve_read_blocks([("a.py", (1, 1))], ws, record_hashes_into=seen)
        assert list(seen.keys()) == ["b.py", "a.py"]


class TestRereadInjectionCap:
    def test_cap_covers_a_full_read_budget(self):
        """With read_file_rounds=6 an investigation legitimately reads
        six-plus files; a re-injection cap below the default budget would
        evict part of the active working set between rounds."""
        assert _REREAD_INJECTION_CAP >= _READ_FILE_ROUNDS_DEFAULT


class TestSourceContracts:
    """Source-level guards (repo convention — see
    test_repair_loop_audit_fixes.py) for the inline node logic that has
    no seam to drive directly."""

    @staticmethod
    def _graph_src() -> str:
        return (_REPO_ROOT / "harness" / "graph.py").read_text(encoding="utf-8")

    def test_no_hardcoded_two_round_budget_in_prompts(self):
        src = self._graph_src()
        assert "at most 2 READ_FILE rounds" not in src
        assert "at most two READ_FILE" not in src

    def test_repair_and_patching_caps_use_the_resolver(self):
        src = self._graph_src()
        assert "READ_FILE_MAX_RESOLVES = 2" not in src, (
            "a READ_FILE resolve cap is hard-coded again; both nodes "
            "must use _resolve_read_file_rounds(state)"
        )
        assert src.count("_resolve_read_file_rounds(state)") >= 2

    def test_bonus_round_exists_in_both_nodes(self):
        # The bonus round lives in the SHARED _read_file_resolution_cycle
        # (event name assembled from event_prefix); both nodes must call
        # the cycle with their prefix so both emit their bonus event.
        src = self._graph_src()
        assert "_read_file_bonus_round" in src
        assert src.count("_read_file_resolution_cycle(") >= 4, (
            "expected the shared cycle definition plus call sites in "
            "patching_node (main + mop-up) and repair_node"
        )
        assert 'event_prefix="repair"' in src
        assert src.count('event_prefix="patching"') >= 2

    def test_reread_injection_walks_most_recent_first(self):
        src = self._graph_src()
        assert "reversed(list(files_seen_by_llm.keys()))" in src, (
            "post-cleanse re-injection no longer walks most-recent-first; "
            "the cap will evict the active working set again "
            "(session 22471c0c)"
        )


class TestReadBlockBracketLeniency:
    """Session 22471c0c post-resume: the repair model emitted
    ``<READ_FILE>`` (single angle brackets) for the two files containing
    the root cause; the strict triple-bracket pattern dropped the request
    silently and the round counted as zero patches — two of those tripped
    the zero-patch HITL. Bracket count is now lenient (1-3) on both parse
    and strip."""

    def test_triple_brackets_still_parse(self):
        from harness.patcher import parse_read_blocks
        out = parse_read_blocks(
            "<<<READ_FILE>>>\nfile: a.py\n<<<END_READ_FILE>>>"
        )
        assert out == [("a.py", None)]

    def test_single_brackets_parse(self):
        from harness.patcher import parse_read_blocks
        # Verbatim shape from debug dump 22471c0c_0012.
        out = parse_read_blocks(
            "<READ_FILE>\nfile: server/app/models/financial.py\n<END_READ_FILE>\n"
            "<READ_FILE>\nfile: server/app/models/company.py\n<END_READ_FILE>\n"
        )
        assert out == [
            ("server/app/models/financial.py", None),
            ("server/app/models/company.py", None),
        ]

    def test_double_brackets_parse_with_range(self):
        from harness.patcher import parse_read_blocks
        out = parse_read_blocks(
            "<<READ_FILE>>\nfile: b.py\nrange: 10-20\n<<END_READ_FILE>>"
        )
        assert out == [("b.py", (10, 20))]

    def test_strip_removes_lenient_variants(self):
        from harness.patcher import strip_read_blocks
        text = (
            "prefix\n"
            "<READ_FILE>\nfile: a.py\n<END_READ_FILE>\n"
            "suffix"
        )
        stripped = strip_read_blocks(text)
        assert "READ_FILE" not in stripped
        assert "prefix" in stripped and "suffix" in stripped


class _Resp:
    def __init__(self, content: str):
        self.content = content


class _CycleDispatch:
    """Stub dispatch: returns canned responses in order, records calls."""

    def __init__(self, scripted: list[str]):
        self.scripted = [_Resp(s) for s in scripted]
        self.calls = 0

    async def __call__(self, messages, budget):
        self.calls += 1
        return self.scripted.pop(0), budget - 0.01


class TestReadFileResolutionCycle:
    """The shared patching/repair READ_FILE component: resolve loop →
    never-seen-file bonus round → forced-patch retry."""

    @staticmethod
    def _run(response_text, scripted, ws, files_seen=None, **kw):
        import asyncio
        from harness.graph import _read_file_resolution_cycle
        dispatch = _CycleDispatch(scripted)
        messages: list = []
        out = asyncio.run(_read_file_resolution_cycle(
            response=_Resp(response_text),
            text=response_text,
            budget=1.0,
            messages=messages,
            dispatch=dispatch,
            state={},
            workspace=ws,
            files_seen=files_seen if files_seen is not None else {},
            node_label="repair_node",
            event_prefix="repair",
            iteration_noun="repair",
            **kw,
        ))
        return out, dispatch, messages

    def test_no_reads_is_a_no_op(self, tmp_path):
        (out_resp, out_text, out_budget), dispatch, messages = self._run(
            "<<<REPLACE_BLOCK>>>\nfile: a.py\nsearch:\nx\nreplace:\ny\n<<<END_REPLACE_BLOCK>>>",
            [], str(tmp_path),
        )
        assert dispatch.calls == 0
        assert messages == []
        assert out_budget == 1.0

    def test_resolve_loop_feeds_content_and_redispatches(self, tmp_path):
        (tmp_path / "a.py").write_text("def f(): return 1\n")
        patch_resp = (
            "<<<REPLACE_BLOCK>>>\nfile: a.py\nsearch:\ndef f(): return 1\n"
            "replace:\ndef f(): return 2\n<<<END_REPLACE_BLOCK>>>"
        )
        (out_resp, out_text, _), dispatch, messages = self._run(
            "<<<READ_FILE>>>\nfile: a.py\n<<<END_READ_FILE>>>",
            [patch_resp], str(tmp_path),
        )
        assert dispatch.calls == 1
        assert out_resp.content == patch_resp and out_text == patch_resp
        # The resolution round-trip landed in the transcript.
        assert any("READ_FILE results" in m["content"] for m in messages)

    def test_bonus_round_for_unseen_file_past_cap(self, tmp_path):
        (tmp_path / "root_cause.py").write_text("BUG = True\n")
        read_req = "<<<READ_FILE>>>\nfile: root_cause.py\n<<<END_READ_FILE>>>"
        patch_resp = (
            "<<<REPLACE_BLOCK>>>\nfile: root_cause.py\nsearch:\nBUG = True\n"
            "replace:\nBUG = False\n<<<END_REPLACE_BLOCK>>>"
        )
        # max_resolves=0 → straight past the cap; the unseen file earns
        # the bonus resolve instead of the forced-patch ultimatum.
        (out_resp, _, _), dispatch, messages = self._run(
            read_req, [patch_resp], str(tmp_path), max_resolves=0,
        )
        assert dispatch.calls == 1
        assert out_resp.content == patch_resp
        assert any("READ_FILE results" in m["content"] for m in messages)

    def test_forced_patch_for_already_seen_file_past_cap(self, tmp_path):
        (tmp_path / "seen.py").write_text("x = 1\n")
        read_req = "<<<READ_FILE>>>\nfile: seen.py\n<<<END_READ_FILE>>>"
        patch_resp = (
            "<<<REPLACE_BLOCK>>>\nfile: seen.py\nsearch:\nx = 1\n"
            "replace:\nx = 2\n<<<END_REPLACE_BLOCK>>>"
        )
        (out_resp, _, _), dispatch, messages = self._run(
            read_req, [patch_resp], str(tmp_path),
            files_seen={"seen.py": "somehash"},  # already shown → no bonus
            max_resolves=0,
        )
        assert dispatch.calls == 1
        assert any("[Forced-patch]" in m["content"] for m in messages)

    def test_stages_can_be_disabled_for_mop_up_reuse(self, tmp_path):
        read_req = "<<<READ_FILE>>>\nfile: never/there.py\n<<<END_READ_FILE>>>"
        (out_resp, out_text, _), dispatch, _ = self._run(
            read_req, [], str(tmp_path),
            max_resolves=0, bonus_round=False, forced_patch=False,
        )
        # Nothing fired: no bonus, no ultimatum, no dispatch.
        assert dispatch.calls == 0
        assert out_text == read_req
