"""Tests for graph-level audit hardening (batches 8, 9).

Covers:
  - compiler_node honours compiler.advisory_exit_codes               (§6.8)
  - _strip_build_output_noise drops multi-line deprecation blocks    (§6.9)
  - route_after_compiler generic no-progress tripwire                (§6.1)
  - _PIP_RESOLUTION_CONFLICT_PATTERNS bounded                        (§6.10)
"""

from __future__ import annotations


import pytest

from harness import graph as gr


# ---------------------------------------------------------------------------
# compiler_node advisory exit codes (audit §6.8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compiler_advisory_exit_codes_folded_to_zero(monkeypatch, tmp_path):
    """An advisory exit code in the operator-configured list must be
    treated as success (folded to 0) rather than triggering repair."""

    class _FakeResult:
        exit_code = 2  # non-zero, but in the advisory list
        diagnostics = []
        raw_output = ""

    class _FakeExecutor:
        def __init__(self, *a, **kw):
            pass

        async def run(self, build_cmd):
            return _FakeResult()

    monkeypatch.setattr("harness.sandbox.SandboxExecutor", _FakeExecutor)
    state = {
        "workspace_path": str(tmp_path),
        "build_command": "terraform validate",
        "compiler_config": {"advisory_exit_codes": [2]},
        "loop_counter": {"compiler": 0, "patching": 0, "repair": 0, "total_repairs": 0},
        "modified_files": [],
        "messages": [],
        "node_state": {},
        "session_id": "test",
        "allow_network": True,
        "deployment_config": {},
    }
    out = await gr.compiler_node(state)
    # Advisory code 2 → folded to 0, no compiler errors emitted.
    assert out["exit_code"] == 0


@pytest.mark.asyncio
async def test_compiler_non_advisory_exit_code_preserved(monkeypatch, tmp_path):
    class _FakeResult:
        exit_code = 1
        diagnostics = []
        raw_output = ""

    class _FakeExecutor:
        def __init__(self, *a, **kw):
            pass

        async def run(self, build_cmd):
            return _FakeResult()

    monkeypatch.setattr("harness.sandbox.SandboxExecutor", _FakeExecutor)
    state = {
        "workspace_path": str(tmp_path),
        "build_command": "make build",
        "compiler_config": {"advisory_exit_codes": [2, 3]},  # NOT 1
        "loop_counter": {"compiler": 0, "patching": 0, "repair": 0, "total_repairs": 0},
        "modified_files": [],
        "messages": [],
        "node_state": {},
        "session_id": "test",
        "allow_network": True,
        "deployment_config": {},
    }
    out = await gr.compiler_node(state)
    # exit_code 1 stays — only 2 and 3 are advisory.
    assert out["exit_code"] == 1


# ---------------------------------------------------------------------------
# _strip_build_output_noise multi-line deprecation drop (audit §6.9)
# ---------------------------------------------------------------------------


def test_strip_noise_drops_multiline_deprecation_block():
    """A DeprecationWarning header followed by an indented source line
    must be dropped — both lines."""
    raw = (
        "actual error line\n"
        "/path/foo.py:42: DeprecationWarning: use bar instead\n"
        "    x = old_thing()\n"
        "\n"
        "more output\n"
    )
    out = gr._strip_build_output_noise(raw)
    assert "DeprecationWarning" not in out
    # The indented continuation line gets dropped too (audit §6.9).
    assert "x = old_thing()" not in out
    # Real errors and non-warning content survive.
    assert "actual error line" in out
    assert "more output" in out


def test_strip_noise_preserves_real_errors():
    """Real stack traces / errors are untouched even when adjacent to
    a deprecation block."""
    raw = (
        "Traceback (most recent call last):\n"
        "  File 'x.py', line 5, in <module>\n"
        "    raise RuntimeError('real bug')\n"
        "RuntimeError: real bug\n"
    )
    out = gr._strip_build_output_noise(raw)
    # Stack trace survives.
    assert "RuntimeError: real bug" in out
    assert "Traceback" in out


# ---------------------------------------------------------------------------
# route_after_compiler generic no-progress tripwire (audit §6.1)
# ---------------------------------------------------------------------------


def _make_router_state(consecutive_zero: int, total_repairs: int = 4) -> dict:
    """Build a minimal state for route_after_compiler tests with a
    healthy budget so the budget-exhaustion gate doesn't fire first."""
    return {
        "exit_code": 1,
        "budget_remaining_usd": 1.00,
        "loop_counter": {
            "total_repairs": total_repairs,
            "consecutive_zero_patch_rounds": consecutive_zero,
            "missing_dep_consecutive_same": 0,
        },
        "compiler_errors": [{"error_code": "MISSING_DEP"}],
        "node_state": {},
    }


def test_route_after_compiler_escalates_on_generic_no_progress():
    """5 consecutive zero-real-patch rounds escalate to HITL even with
    autofixable diagnostics in play — closes the alternating MISSING_DEP
    cycle bypass."""
    decision = gr.route_after_compiler(_make_router_state(consecutive_zero=5))
    assert "human_intervention_node" in str(decision)


def test_route_after_compiler_continues_under_threshold():
    """4 consecutive zero rounds with autofixable diagnostics still
    proceeds to repair_node — the new gate doesn't false-positive."""
    decision = gr.route_after_compiler(_make_router_state(consecutive_zero=4))
    # Should NOT escalate yet (< 5).
    assert "repair_node" in str(decision)


# ---------------------------------------------------------------------------
# Pip resolution regex bounded (audit §6.10)
# ---------------------------------------------------------------------------


def test_pip_resolution_conflict_regex_bounded():
    """The Cannot-install pattern must use a bounded [^\\n]{1,500}
    instead of unbounded .+ to avoid backtracking on long lines."""
    for pat in gr._PIP_RESOLUTION_CONFLICT_PATTERNS:
        pattern_src = pat.pattern
        # No unbounded ``.+`` in any of the audit-sensitive patterns.
        # ``Cannot install`` is the audit-flagged regex.
        if "Cannot install" in pattern_src:
            assert "[^\\n]" in pattern_src or "{1,500}" in pattern_src


def test_pip_resolution_conflict_matches_actual_log():
    """Regression: the bounded regex must still match real pip output."""
    raw_output = (
        "ERROR: Cannot install some-pkg==1.0 and other==2.0 because these package versions have "
        "conflicting dependencies\n"
        "tail\n"
    )
    assert gr._is_pip_resolution_conflict(raw_output, "pip install -r requirements.txt")


# ---------------------------------------------------------------------------
# _update_sticky_create_rejections — D1 (finsearch session 44c5e194)
# ---------------------------------------------------------------------------

class TestStickyCreateRejections:
    """Finsearch session 44c5e194 root cause D1: the LLM emitted
    CREATE_FILE for existing files 15+ times because the rejection
    memory lived only in ``node_state['patch_failures']`` (round-N-1
    only). The sticky accumulator persists across every round of the
    session until the LLM proves re-orientation by successfully
    modifying the file with a non-CREATE op."""

    def test_new_rejection_added_to_sticky_set(self):
        state = {"sticky_create_rejections": []}
        failures = [
            {
                "file": "server/models.py",
                "operation": "create_file",
                "error": "File already exists with different content: server/models.py",
            },
        ]
        out = gr._update_sticky_create_rejections(state, failures, [])
        assert out == ["server/models.py"]

    def test_pre_existing_rejections_preserved(self):
        state = {
            "sticky_create_rejections": ["server/database.py", "server/main.py"],
        }
        failures = [
            {
                "file": "server/models.py",
                "operation": "create_file",
                "error": "File already exists with different content: ...",
            },
        ]
        out = gr._update_sticky_create_rejections(state, failures, [])
        assert out == ["server/database.py", "server/main.py", "server/models.py"]

    def test_dedup_when_same_file_rejected_again(self):
        state = {"sticky_create_rejections": ["server/main.py"]}
        failures = [
            {
                "file": "server/main.py",
                "operation": "create_file",
                "error": "File already exists ...",
            },
        ]
        out = gr._update_sticky_create_rejections(state, failures, [])
        assert out == ["server/main.py"]

    def test_successful_modification_clears_sticky(self):
        # The LLM has now used replace_block / insert_at_block on
        # server/main.py — no more need to nag about it.
        state = {"sticky_create_rejections": ["server/main.py", "server/db.py"]}
        out = gr._update_sticky_create_rejections(
            state, [], this_round_modified=["server/main.py"],
        )
        assert out == ["server/db.py"]

    def test_ignores_non_create_failures(self):
        # A REPLACE_BLOCK "search not found" failure is a different
        # bug — don't add to the create-rejection set.
        state = {"sticky_create_rejections": []}
        failures = [
            {
                "file": "server/x.py",
                "operation": "replace_block",
                "error": "Search block not found ...",
            },
        ]
        out = gr._update_sticky_create_rejections(state, failures, [])
        assert out == []

    def test_ignores_create_failures_without_already_exists_message(self):
        # E.g. an allowlist rejection routed via patch_failures — has
        # operation=create_file but not the "already exists" phrase.
        state = {"sticky_create_rejections": []}
        failures = [
            {
                "file": "outside/root.py",
                "operation": "create_file",
                "error": "path not in allowlist",
            },
        ]
        out = gr._update_sticky_create_rejections(state, failures, [])
        assert out == []

    def test_tolerates_missing_state_key(self):
        # Fresh session: state has no sticky_create_rejections yet.
        out = gr._update_sticky_create_rejections({}, [], [])
        assert out == []

    def test_tolerates_malformed_failure_entries(self):
        state = {"sticky_create_rejections": []}
        failures = [
            "not a dict",
            {"file": "", "operation": "create_file", "error": "already exists"},
            {"operation": "create_file", "error": "already exists"},  # no file
            None,
        ]
        # None of these should be added.
        out = gr._update_sticky_create_rejections(state, failures, [])
        assert out == []


class TestDedupeRepeatedPreambles:
    """lumina 019f82af: patching/repair re-inject static preambles every round
    without deduping, so the conversation accumulates byte-identical copies of
    20-30k-char blocks (~30k tokens of a 143k prompt). The helper stubs the
    older copies, keeping the last full — cutting context-window pressure and
    the duplicate-file distraction without touching alternation."""

    def test_stubs_all_but_last_identical_copy(self):
        big = "P" * 3000
        msgs = [
            {"role": "system", "content": big},
            {"role": "user", "content": "small"},
            {"role": "system", "content": big},
            {"role": "system", "content": big},  # last → kept
        ]
        stubbed, reclaimed = gr._dedupe_repeated_preambles(msgs)
        assert stubbed == 2
        assert reclaimed == 6000
        assert msgs[0]["content"].startswith("[deduplicated")
        assert msgs[2]["content"].startswith("[deduplicated")
        assert msgs[3]["content"] == big          # last copy kept full
        assert msgs[1]["content"] == "small"       # small untouched

    def test_ignores_small_and_unique_messages(self):
        msgs = [
            {"role": "user", "content": "a" * 100},   # small
            {"role": "user", "content": "b" * 5000},  # unique large
        ]
        stubbed, _ = gr._dedupe_repeated_preambles(msgs)
        assert stubbed == 0
        assert len(msgs[1]["content"]) == 5000

    def test_preserves_message_count_and_roles(self):
        big = "Q" * 2500
        msgs = [
            {"role": "system", "content": big},
            {"role": "assistant", "content": "resp"},
            {"role": "system", "content": big},
        ]
        gr._dedupe_repeated_preambles(msgs)
        assert len(msgs) == 3
        assert [m["role"] for m in msgs] == ["system", "assistant", "system"]

    def test_handles_non_string_content_safely(self):
        # tool-block content (a list) must not crash the scan.
        msgs = [
            {"role": "user", "content": [{"type": "tool_result", "x": 1}]},
            {"role": "user", "content": "z" * 3000},
        ]
        stubbed, _ = gr._dedupe_repeated_preambles(msgs)
        assert stubbed == 0
