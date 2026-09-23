"""ADR-0008 item 3 — the spec region narrowed to the work in hand.

run12 (lumina-run12-20260923-0914) measured repair using the anchored region
on 0 of 24 dispatches and patching on 2 of 13, so repair is narrowed first.

The invariant that matters most here is the ADR's safety rule: an unresolved
scope keeps the FULL region. Narrowing is an optimisation; a silently missing
requirement is a correctness bug, and these tests pin which way the failure
must fall.
"""

from __future__ import annotations

import pytest

from harness import spec_slice
from harness.spec_slice import (
    apply_to_messages,
    resolve_scope,
    scoped_slice,
    tier1_preamble,
)


SPEC = """# Software Requirements Specification

## API conventions

Collections are returned wrapped in an envelope, never as a bare list.

## Test conventions

Unit tests live beside the module they cover.

## Epic: EPIC-001 — Birthday visibility

**Description:** Everything about seeing birthdays.

#### Story: STORY-001 — Display upcoming birthdays

**Parent feature:** FEAT-001

**Acceptance Criteria:**

- The dashboard lists Alice first with one day left.
"""

SYSTEM_TAIL = "You are an expert software engineer.\n\n## Rules\n- Be careful.\n"


def _anchored(spec: str = SPEC) -> str:
    return spec + "\n\n---\n\n" + SYSTEM_TAIL


def _msgs():
    return [
        {"role": "system", "content": _anchored()},
        {"role": "user", "content": "Build it."},
        {"role": "user", "content": "FAILED test_main.py::test_x"},
    ]


class TestTier1Preamble:
    def test_keeps_the_cross_cutting_head(self):
        got = tier1_preamble(SPEC)
        assert "Collections are returned wrapped in an envelope" in got
        assert "Unit tests live beside the module" in got

    def test_drops_the_requirement_bodies(self):
        got = tier1_preamble(SPEC)
        assert "STORY-001" not in got and "EPIC-001" not in got

    def test_a_spec_with_no_requirement_headings_is_all_preamble(self):
        # Trimming to nothing would be the silent under-injection this
        # module exists to prevent.
        text = "## Conventions\n\nEverything here is cross-cutting.\n"
        assert tier1_preamble(text) == text.rstrip() + "\n"

    def test_empty_input(self):
        assert tier1_preamble("") == ""


class TestScopeResolution:
    def test_an_active_story_wins(self):
        assert resolve_scope({"current_story_id": "STORY-003"}) == ["STORY-003"]

    def test_falls_back_to_the_batch(self):
        # run12: one patching call used STORY-001, STORY-002 and STORY-004
        # together, which is why batch scope exists at all.
        assert resolve_scope(
            {"batch_patched_story_keys": ["STORY-001", "STORY-002"]}
        ) == ["STORY-001", "STORY-002"]

    def test_nothing_resolvable_is_empty(self):
        assert resolve_scope({}) == []
        assert resolve_scope({"current_story_id": ""}) == []


class TestSliceAssembly:
    def test_no_workspace_or_keys_yields_nothing(self):
        assert scoped_slice("", ["STORY-001"]) == ""
        assert scoped_slice("/tmp", []) == ""

    def test_a_db_failure_is_not_fatal(self, tmp_path):
        # A workspace with no story DB must return "" (→ full region kept),
        # never raise into a dispatch.
        assert scoped_slice(str(tmp_path), ["STORY-001"]) == ""


class TestApplyToMessages:
    """The failure direction is the whole point: when anything is missing,
    the call keeps the full spec region."""

    def test_unresolved_scope_keeps_the_full_region(self):
        msgs = _msgs()
        out, tel = apply_to_messages(msgs, {"workspace_path": "/nope"},
                                     consumer="repair_node")
        assert out is msgs and tel is None
        assert "STORY-001" in out[0]["content"], "the region must be intact"

    def test_a_scope_with_no_rows_keeps_the_full_region(self, tmp_path):
        out, tel = apply_to_messages(
            _msgs(),
            {"workspace_path": str(tmp_path), "current_story_id": "STORY-001"},
            consumer="repair_node",
        )
        assert tel is None
        assert "Display upcoming birthdays" in out[0]["content"]

    def test_a_prompt_with_no_anchored_region_is_untouched(self):
        msgs = [{"role": "system", "content": SYSTEM_TAIL}]
        out, tel = apply_to_messages(msgs, {"current_story_id": "STORY-001"},
                                     consumer="repair_node")
        assert out is msgs and tel is None

    def test_narrows_the_prefix_and_appends_an_extract(self, monkeypatch):
        monkeypatch.setattr(
            spec_slice, "scoped_slice",
            lambda ws, keys, **kw: "### STORY-001\nthe story body\n")
        out, tel = apply_to_messages(
            _msgs(), {"workspace_path": "/ws", "current_story_id": "STORY-001"},
            consumer="repair_node",
        )
        head = out[0]["content"]
        # Cross-cutting context and the harness contract both survive.
        assert "Collections are returned wrapped in an envelope" in head
        assert "You are an expert software engineer" in head
        # The requirement bodies are gone from the cached prefix...
        assert "Display upcoming birthdays" not in head
        # ...and the extract rides as a LATER message, per graph.py:1721.
        assert out[-2]["role"] == "user"
        assert "the story body" in out[-2]["content"]
        assert tel["stories"] == ["STORY-001"] and tel["saved_chars"] > 0

    def test_the_input_list_is_not_mutated(self, monkeypatch):
        monkeypatch.setattr(
            spec_slice, "scoped_slice", lambda ws, keys, **kw: "### X\nbody\n")
        msgs = _msgs()
        before = [dict(m) for m in msgs]
        apply_to_messages(
            msgs, {"workspace_path": "/ws", "current_story_id": "STORY-001"},
            consumer="repair_node",
        )
        assert msgs == before, "state's own message list must be untouched"


class TestAgainstASeededPlan:
    """The measurement used a real plan; this seeds one with the same shape.

    ``tests/conftest.py`` redirects TEANE_STATE_DB into tmp_path so tests
    never read the operator's DB — so the plan is built here, through the
    same ``create_requirements`` helper decomposition's ingest uses.
    """

    @pytest.fixture
    def workspace(self, tmp_path):
        from harness import story_state as sst
        ws = tmp_path / "lumina"
        ws.mkdir()
        app = sst.app_name_for_workspace(str(ws))
        conn = sst.open_story_db(workspace_path=str(ws))
        sst.create_requirements(conn, app, [
            {"req_key": "EPIC-001", "kind": "epic", "title": "Visibility",
             "body": "**Description:** Everything about seeing birthdays."},
            {"req_key": "FEAT-001", "kind": "feat", "title": "Dashboard",
             "body": "**Parent epic:** EPIC-001\n\n**Feature-level AC:**\n"
                     "- Collections come back wrapped in an envelope."},
            {"req_key": "STORY-001", "kind": "safe_story", "title": "Display",
             "body": "**Parent feature:** FEAT-001\n\n**As a** employee\n"
                     "**Acceptance Criteria:**\n- Alice is listed first."},
            {"req_key": "STORY-005", "kind": "safe_story", "title": "Delete",
             "body": "**Parent feature:** FEAT-002\n\n"
                     "- Deleting a record leaves the others intact."},
        ])
        conn.commit()
        conn.close()
        return str(ws)

    def test_carries_the_story_and_walks_to_its_parents(self, workspace):
        out = scoped_slice(workspace, ["STORY-001"])
        assert "Alice is listed first" in out
        # A story's contract often lives one level up — the feature-level AC
        # naming the envelope is exactly the fact run 11 was missing.
        assert "wrapped in an envelope" in out
        assert "EPIC-001" in out, "the walk must not stop at the feature"

    def test_does_not_carry_unrelated_stories(self, workspace):
        out = scoped_slice(workspace, ["STORY-001"])
        assert "STORY-005" not in out
        assert "Deleting a record" not in out

    def test_a_batch_scope_carries_every_story_in_it(self, workspace):
        # run12's patching call used three stories at once; a batch slice
        # must not silently keep only the first.
        out = scoped_slice(workspace, ["STORY-001", "STORY-005"])
        assert "Alice is listed first" in out and "Deleting a record" in out

    def test_an_unknown_story_yields_nothing(self, workspace):
        # → caller keeps the full region rather than shipping an empty slice.
        assert scoped_slice(workspace, ["STORY-404"]) == ""

    def test_the_extract_is_bounded(self, workspace):
        out = scoped_slice(workspace, ["STORY-001"], max_chars=200)
        assert len(out) < 300 and "[extract truncated]" in out


class TestGatewayWiring:
    def test_off_by_default(self):
        from harness.gateway import create_gateway_from_config
        assert create_gateway_from_config({}).config.scoped_spec_context is False

    def test_honoured_from_planning_section(self):
        from harness.gateway import create_gateway_from_config
        gw = create_gateway_from_config(
            {"planning": {"scoped_spec_context": True}})
        assert gw.config.scoped_spec_context is True


class TestTheRun13Regressions:
    """Two bugs run 13 (lumina-run13-20260923-2235) exposed in this feature,
    both of which failed SILENTLY — the run completed and said nothing."""

    def test_a_spec_ending_in_a_rule_still_splits(self):
        # The seam is spec + "\n\n---\n\n" + prompt, but a spec whose own
        # last line is a `---` rule doubles it. Run 13's architecture
        # document did exactly that: region detection returned "" for every
        # repair call, disabling both the measurement and the slice without
        # a word.
        from harness.spec_usage import split_spec_region
        doubled = SPEC.rstrip() + "\n\n---\n" + "\n\n---\n\n" + SYSTEM_TAIL
        region = split_spec_region(doubled)
        assert region, "a trailing horizontal rule must not hide the region"
        assert "STORY-001" in region
        assert "You are an expert" not in region

    def test_several_trailing_rules_are_trimmed(self):
        from harness.spec_usage import split_spec_region
        region = split_spec_region(
            SPEC.rstrip() + "\n\n---\n\n---\n\n---\n\n" + SYSTEM_TAIL)
        assert region.rstrip().endswith("day left.")

    def test_narrowing_never_reaches_the_list_the_node_returns(self, monkeypatch):
        # repair_node returns its ``messages`` into state. Narrowing that
        # list made one repair round truncate the anchor permanently, for
        # every later call of every role — including patching, the role the
        # measurement showed using the requirement bodies. The dispatch copy
        # is narrowed; the caller's list is not.
        monkeypatch.setattr(
            spec_slice, "scoped_slice", lambda ws, keys, **kw: "### S\nbody\n")
        state_messages = _msgs()
        dispatched, tel = apply_to_messages(
            state_messages,
            {"workspace_path": "/ws", "current_story_id": "STORY-001"},
            consumer="repair_node",
        )
        assert tel is not None, "precondition: the slice must have applied"
        assert "Display upcoming birthdays" not in dispatched[0]["content"]
        assert "Display upcoming birthdays" in state_messages[0]["content"], (
            "the node's own list must still carry the full region"
        )
        assert len(state_messages) == 3 and len(dispatched) == 4


class TestExtractPlacement:
    """run14: the extract was appended at the very end, displacing
    _REPAIR_FORMAT_REMINDER — the message whose entire purpose is to have the
    last word (patch DSL + the round's blocking correction). Ten repair
    rounds came back UNPARSED, the model answering in anthropic_xml
    `edit_file` tool calls, and the run died on persistent_build_failure."""

    def _applied(self, monkeypatch, **kw):
        monkeypatch.setattr(
            spec_slice, "scoped_slice", lambda ws, keys, **k: "### S\nbody\n")
        msgs = _msgs() + [{"role": "user", "content": "Generate patches NOW."}]
        out, _ = apply_to_messages(
            msgs, {"workspace_path": "/ws", "current_story_id": "STORY-001"},
            consumer="repair_node", **kw)
        return out

    def test_the_callers_last_message_keeps_the_last_word(self, monkeypatch):
        out = self._applied(monkeypatch)
        assert out[-1]["content"] == "Generate patches NOW."
        assert "body" in out[-2]["content"]

    def test_end_placement_is_still_available(self, monkeypatch):
        out = self._applied(monkeypatch, position="end")
        assert "body" in out[-1]["content"]

    def test_a_single_message_prompt_degrades_to_append(self, monkeypatch):
        monkeypatch.setattr(
            spec_slice, "scoped_slice", lambda ws, keys, **k: "### S\nbody\n")
        out, tel = apply_to_messages(
            [{"role": "system", "content": _anchored()}],
            {"workspace_path": "/ws", "current_story_id": "STORY-001"},
            consumer="repair_node")
        assert tel is not None and "body" in out[-1]["content"]
