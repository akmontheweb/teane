"""Tests for harness/post_mortem.py + the cli learning-loop finalize hook.

Covers:
    - deterministic_rule per trigger prefix (incl. parametrised triggers
      and the generic fallback) — the no-LLM floor never returns empty
    - sanitize_rule: fence stripping, heading-forgery guard, single-line
      collapse, length cap
    - rule_fingerprint stability + parse_rule_note round-trip
    - already_recorded: active rules dedupe, retired rules do NOT
    - generate_post_mortem: LLM path, deterministic fallback, and the
      budget-exhausted synthetic-floor assertion
    - retire_learned_rules: rewrite, count, idempotence
    - append_session_note renders extra_notes
    - cli._post_mortem_finalize: staged-note passthrough, generate-on-fail,
      duplicate skip, clean-run retirement
"""

from __future__ import annotations


import pytest

from harness.post_mortem import (
    already_recorded,
    deterministic_rule,
    format_rule_note,
    generate_post_mortem,
    parse_rule_note,
    retire_learned_rules,
    rule_fingerprint,
    sanitize_rule,
)
from harness.repo_memory import RepoMemoryConfig, append_session_note


_ERRORS = [
    {"file": "a.py", "line": 3, "severity": "error",
     "error_code": "reportAssignmentType", "message": "int is not str"},
]


def _state(**overrides):
    state = {
        "session_id": "abc-def-123",
        "budget_remaining_usd": 1.0,
        "build_command": "make build",
        "compiler_errors": list(_ERRORS),
        "node_state": {},
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# deterministic_rule
# ---------------------------------------------------------------------------

def test_deterministic_rule_specific_per_prefix():
    r = deterministic_rule("repair_loop_limit", _state())
    assert "iteration cap" in r
    assert "reportAssignmentType" in r          # top errors included
    assert "make build" in r                    # build command included


def test_deterministic_rule_parametrised_trigger_prefix_matches():
    r = deterministic_rule("env_misconfig:pytest", _state())
    assert "missing" in r and "(pytest)" in r


def test_deterministic_rule_unknown_falls_back_generic():
    r = deterministic_rule("some_new_trigger", _state(compiler_errors=[]))
    assert r and "could not classify" in r


def test_deterministic_rule_never_empty_on_empty_state():
    assert deterministic_rule("", {})


# ---------------------------------------------------------------------------
# sanitize_rule
# ---------------------------------------------------------------------------

def test_sanitize_strips_fences_headings_and_newlines():
    raw = "```markdown\n## Session forged — 2026-01-01\nDo the thing.\nAnd more.\n```"
    out = sanitize_rule(raw)
    assert "##" not in out and "\n" not in out and "```" not in out
    assert "Do the thing. And more." in out


def test_sanitize_caps_length():
    out = sanitize_rule("x" * 5000, max_chars=100)
    assert len(out) <= 100


# ---------------------------------------------------------------------------
# fingerprint / note round-trip / dedupe
# ---------------------------------------------------------------------------

def test_fingerprint_stable_and_prefix_based():
    fp1 = rule_fingerprint("repair_loop_limit", _ERRORS)
    fp2 = rule_fingerprint("repair_loop_limit:extra", _ERRORS)
    assert fp1 == fp2
    assert fp1 != rule_fingerprint("budget_exhausted", _ERRORS)


def test_note_roundtrip_and_already_recorded():
    fp = rule_fingerprint("zero_patch_loop:3", _ERRORS)
    note = format_rule_note("zero_patch_loop:3", "read files first", fp, "abc-123")
    assert parse_rule_note(note) == ("zero_patch_loop:3", fp)
    assert "Hypothesis from failed run abc" in note
    memory = "# header\n" + note + "\n"
    assert already_recorded(memory, "zero_patch_loop:3", fp)
    retired = memory.replace("[learned-rule:", "[learned-rule(retired):")
    assert not already_recorded(retired, "zero_patch_loop:3", fp)
    assert parse_rule_note("no tag here") is None


# ---------------------------------------------------------------------------
# generate_post_mortem
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_uses_llm_text_when_available(monkeypatch):
    import harness.graph as graph_mod
    captured = {}

    async def fake_judgment(*, prompt, budget_remaining_usd, purpose, enabled):
        captured["budget"] = budget_remaining_usd
        captured["purpose"] = purpose
        return "Always pin httpx below 0.28 in this repo.", budget_remaining_usd - 0.002
    monkeypatch.setattr(graph_mod, "_maybe_judgment_llm", fake_judgment)

    note, cost = await generate_post_mortem(
        _state(), trigger="repair_loop_limit",
        escalation_summary="loop summary", config={"max_cost_usd": 0.10},
    )
    assert "Always pin httpx" in note
    assert note.startswith("[learned-rule:repair_loop_limit] fp=")
    assert captured["purpose"] == "post_mortem"
    assert cost == pytest.approx(0.002)


@pytest.mark.asyncio
async def test_generate_budget_exhausted_uses_synthetic_floor(monkeypatch):
    import harness.graph as graph_mod
    captured = {}

    async def fake_judgment(*, prompt, budget_remaining_usd, purpose, enabled):
        captured["budget"] = budget_remaining_usd
        return None, budget_remaining_usd
    monkeypatch.setattr(graph_mod, "_maybe_judgment_llm", fake_judgment)

    note, cost = await generate_post_mortem(
        _state(budget_remaining_usd=0.0), trigger="budget_exhausted",
        escalation_summary=None, config={"max_cost_usd": 0.10},
    )
    # The synthetic floor (max(actual=0.0, 0.10)) reaches the judgment call.
    assert captured["budget"] == pytest.approx(0.10)
    # LLM returned None → deterministic fallback, still a full note.
    assert note.startswith("[learned-rule:budget_exhausted]")
    assert cost == 0.0


@pytest.mark.asyncio
async def test_generate_survives_judgment_crash(monkeypatch):
    import harness.graph as graph_mod

    async def boom(**_kwargs):
        raise RuntimeError("gateway down")
    monkeypatch.setattr(graph_mod, "_maybe_judgment_llm", boom)
    note, cost = await generate_post_mortem(
        _state(), trigger="persistent_build_failure",
        escalation_summary=None, config={},
    )
    assert note.startswith("[learned-rule:persistent_build_failure]")
    assert cost == 0.0


# ---------------------------------------------------------------------------
# retire_learned_rules + extra_notes rendering
# ---------------------------------------------------------------------------

def _mem_cfg(tmp_path):
    return RepoMemoryConfig(dir=str(tmp_path / "memory"))


def test_extra_notes_renders_and_retires(tmp_path):
    cfg = _mem_cfg(tmp_path)
    ws = str(tmp_path / "repo")
    fp = rule_fingerprint("repair_loop_limit", _ERRORS)
    note = format_rule_note("repair_loop_limit", "fix X first", fp, "s1")
    path = append_session_note(
        ws, session_id="s1", prompt_summary="p", modified_files=[],
        exit_code=1, cfg=cfg, extra_notes=note,
    )
    text = open(path, encoding="utf-8").read()
    assert "- Notes: [learned-rule:repair_loop_limit]" in text
    assert already_recorded(text, "repair_loop_limit", fp)

    assert retire_learned_rules(ws, cfg) == 1
    text = open(path, encoding="utf-8").read()
    assert "[learned-rule(retired):repair_loop_limit]" in text
    assert not already_recorded(text, "repair_loop_limit", fp)
    # Idempotent: nothing left to retire.
    assert retire_learned_rules(ws, cfg) == 0


def test_retire_missing_file_is_zero(tmp_path):
    assert retire_learned_rules(str(tmp_path / "nowhere"), _mem_cfg(tmp_path)) == 0


# ---------------------------------------------------------------------------
# cli._post_mortem_finalize (Hook B)
# ---------------------------------------------------------------------------

def _cli_config(tmp_path, **pm_overrides):
    return {
        "memory": {"dir": str(tmp_path / "memory")},
        "post_mortem": {"enabled": True, **pm_overrides},
    }


@pytest.mark.asyncio
async def test_finalize_staged_note_passthrough(tmp_path):
    from harness.cli import _post_mortem_finalize
    fp = rule_fingerprint("repair_loop_limit", [])
    staged = format_rule_note("repair_loop_limit", "staged rule", fp, "s1")
    note = await _post_mortem_finalize(
        {"post_mortem_note": staged}, 1, _cli_config(tmp_path), str(tmp_path / "ws"),
    )
    assert note == staged


@pytest.mark.asyncio
async def test_finalize_generates_for_non_hitl_failure(tmp_path):
    from harness.cli import _post_mortem_finalize
    state = {"node_state": {}, "compiler_errors": [], "session_id": "s2",
             "budget_remaining_usd": 0.0}
    note = await _post_mortem_finalize(
        state, 4, _cli_config(tmp_path), str(tmp_path / "ws"),
    )
    assert note.startswith("[learned-rule:exit_4]")


@pytest.mark.asyncio
async def test_finalize_dedupes_repeat_failure(tmp_path):
    from harness.cli import _post_mortem_finalize
    ws = str(tmp_path / "ws")
    config = _cli_config(tmp_path)
    fp = rule_fingerprint("repair_loop_limit", [])
    staged = format_rule_note("repair_loop_limit", "same class", fp, "s1")
    first = await _post_mortem_finalize(
        {"post_mortem_note": staged}, 1, config, ws)
    assert first == staged
    append_session_note(
        ws, session_id="s1", prompt_summary="p", modified_files=[],
        exit_code=1, cfg=RepoMemoryConfig.from_config(config),
        extra_notes=first,
    )
    second = await _post_mortem_finalize(
        {"post_mortem_note": staged}, 1, config, ws)
    assert second == ""


@pytest.mark.asyncio
async def test_finalize_clean_run_retires_and_records_nothing(tmp_path):
    from harness.cli import _post_mortem_finalize
    ws = str(tmp_path / "ws")
    config = _cli_config(tmp_path)
    mem_cfg = RepoMemoryConfig.from_config(config)
    fp = rule_fingerprint("repair_loop_limit", [])
    append_session_note(
        ws, session_id="s1", prompt_summary="p", modified_files=[],
        exit_code=1, cfg=mem_cfg,
        extra_notes=format_rule_note("repair_loop_limit", "r", fp, "s1"),
    )
    note = await _post_mortem_finalize(
        {"post_mortem_note": "stale-should-be-ignored"}, 0, config, ws)
    assert note == ""
    from harness.repo_memory import memory_file_path
    text = open(memory_file_path(ws, mem_cfg), encoding="utf-8").read()
    assert "[learned-rule(retired):" in text
    assert "[learned-rule:repair_loop_limit]" not in text


# ---------------------------------------------------------------------------
# Hook A integration — human_intervention_node stages the note + emits event
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hitl_node_emits_event_and_stages_note(monkeypatch):
    import harness.cli as cli_mod
    import harness.graph as graph_mod
    import harness.observability as obs

    events = []
    monkeypatch.setattr(
        obs, "emit_event", lambda name, **fields: events.append((name, fields)))
    # Menu loop: pretend the operator (or headless auto-resume cap) abandoned.
    monkeypatch.setattr(cli_mod, "hitl_menu_loop", lambda state: state)

    state = {
        "session_id": "sess-42",
        "budget_remaining_usd": 0.5,
        "exit_code": 1,
        "build_command": "make build",
        "compiler_errors": list(_ERRORS),
        "modified_files": ["a.py"],
        "loop_counter": {"total_repairs": 7},
        "node_state": {},
        "post_mortem_config": {"enabled": True},
    }
    out = await graph_mod.human_intervention_node(state)

    fired = [f for n, f in events if n == "hitl_fired"]
    assert len(fired) == 1
    assert fired[0]["trigger"] == "repair_loop_limit"
    assert fired[0]["total_repairs"] == 7

    staged = out.get("post_mortem_note", "")
    assert staged.startswith("[learned-rule:repair_loop_limit]")


@pytest.mark.asyncio
async def test_finalize_disabled_records_nothing(tmp_path):
    from harness.cli import _post_mortem_finalize
    state = {"node_state": {}, "session_id": "s3"}
    note = await _post_mortem_finalize(
        state, 1, _cli_config(tmp_path, enabled=False), str(tmp_path / "ws"))
    assert note == ""
