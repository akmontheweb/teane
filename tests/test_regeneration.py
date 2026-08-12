"""Test-author regeneration node + gates (ADR-0001, Tier A).

The gates are the anti-reward-hack boundary — a regeneration that guts
assertions, touches other files, or ignores the spec must be rejected. These
tests pin the gates and the node's ladder-deferral behaviour.
"""

import os
import tempfile

import pytest

from harness import test_regeneration as tr
from harness.test_regeneration import (
    coverage_nonregression_ok,
    count_assertion_sites,
    count_test_functions,
    has_code_linkage,
    public_symbols,
    symbol_coverage,
    patch_target_paths,
    salvage_canonical_rewrite,
    build_regeneration_messages,
)
from harness.patcher import _BLOCK_PATTERNS, OperationType

_REWRITE_RE = _BLOCK_PATTERNS[OperationType.REWRITE_FILE]

# The node itself is named ``test_regeneration_node`` — aliased so pytest
# doesn't collect it as a test case (and mis-read its ``state`` arg as a
# fixture).
regeneration_node = tr.test_regeneration_node


ORIGINAL = '''
class TestContactUpdate:
    def test_none_fields_allowed(self):
        obj = ContactUpdate(first_name=None)
        pass

    def test_all_none_raises(self):
        with pytest.raises(ValidationError):
            ContactUpdate(first_name=None)
        assert "at least one field" in str(exc.value).lower()
'''

# A legitimate fix: both tests kept, the contradictory one corrected.
GOOD_FIX = '''
class TestContactUpdate:
    # spec: STORY-002 — at least one field must be provided (else 422)
    def test_some_field_allowed(self):
        obj = ContactUpdate(first_name="Jo")
        assert obj.first_name == "Jo"

    def test_all_none_raises(self):
        with pytest.raises(ValidationError) as exc:
            ContactUpdate()
        assert "at least one field" in str(exc.value).lower()
'''

# Reward-hack: assertions gutted to pass.
GUTTED = '''
class TestContactUpdate:
    def test_none_fields_allowed(self):
        pass

    def test_all_none_raises(self):
        pass
'''


class TestCounters:
    def test_count_functions(self):
        assert count_test_functions(ORIGINAL) == 2
        assert count_test_functions("def helper(): pass") == 0

    def test_count_assertion_sites(self):
        assert count_assertion_sites(ORIGINAL) >= 2  # raises + assert
        assert count_assertion_sites(GUTTED) == 0

    def test_count_on_syntax_error(self):
        assert count_test_functions("def t(:") == 0
        assert count_assertion_sites("def t(:") == 0


class TestCoverageGate:
    def test_legitimate_fix_passes(self):
        ok, detail = coverage_nonregression_ok(ORIGINAL, GOOD_FIX)
        assert ok, detail

    def test_gutted_rejected(self):
        ok, detail = coverage_nonregression_ok(ORIGINAL, GUTTED)
        assert not ok and "no assertions" in detail

    def test_wholesale_deletion_rejected(self):
        empty = "class T:\n    def test_a(self):\n        assert True\n"  # 1 fn vs 2
        # dropping from 2 -> 1 is allowed (>=of-1); dropping 3 -> 1 is not
        three = (ORIGINAL + "\n    def test_c(self):\n        assert 1\n")
        ok, _ = coverage_nonregression_ok(three, empty)
        assert not ok

    def test_unparseable_regen_rejected(self):
        ok, detail = coverage_nonregression_ok(ORIGINAL, "def t(:\n x")
        assert not ok and "parse" in detail

    def test_empty_regen_rejected(self):
        ok, _ = coverage_nonregression_ok(ORIGINAL, "   ")
        assert not ok


class TestCodeLinkage:
    def test_tests_marker_present(self):
        assert has_code_linkage("# @tests: server/app/models/contact.py\ndef test_x(): pass")

    def test_no_marker(self):
        assert not has_code_linkage("def test_x():\n    assert True")
        assert not has_code_linkage("")

    def test_verifies_marker_is_not_code_linkage(self):
        # AC linkage must NOT count — unit tests link to code, not stories.
        assert not has_code_linkage("# @verifies: STORY-002.AC-1\ndef test_x(): pass")


class TestPublicSymbols:
    def test_extracts_public_functions_and_classes(self):
        src = (
            "class Foo:\n    pass\n"
            "def bar():\n    pass\n"
            "def _private():\n    pass\n"
            "class _Hidden:\n    pass\n"
        )
        assert set(public_symbols(src)) == {"Foo", "bar"}

    def test_empty_on_syntax_error(self):
        assert public_symbols("def f(:") == []


class TestSymbolCoverage:
    def test_covered_and_uncovered(self):
        test_src = "obj = ContactUpdate()\nassert ContactCreate\n"
        covered, uncovered = symbol_coverage(
            test_src, ["ContactCreate", "ContactUpdate", "ContactOut"],
        )
        assert set(covered) == {"ContactCreate", "ContactUpdate"}
        assert uncovered == ["ContactOut"]


class TestPatchTargets:
    def test_extracts_file_lines(self):
        patch = (
            "<<<REWRITE_FILE>>>\n"
            "file: tests/backend/test_x.py\n"
            "content:\n...\n"
            "<<<END_REWRITE_FILE>>>\n"
        )
        assert patch_target_paths(patch) == {"tests/backend/test_x.py"}

    def test_multiple_targets(self):
        patch = "file: a.py\nfile: b.py\n"
        assert patch_target_paths(patch) == {"a.py", "b.py"}


class TestSalvageCanonicalRewrite:
    """A reasoning model emits a correct body in a non-canonical dialect the
    strict patcher grammar rejects; salvage recovers it targeting the KNOWN
    file so the tier-B escape lands instead of dropping to HITL (lumina
    019ff418: both regen attempts emitted a valid body this way, parsed as
    zero targets, and stalled the build at reflection_distraction_loop)."""

    REL = "tests/unit/test_database.py"

    def _parses_to(self, salvaged, rel):
        assert salvaged is not None
        m = _REWRITE_RE.search(salvaged)
        assert m is not None, f"salvaged output does not parse:\n{salvaged}"
        assert m.group("file").strip() == rel
        return m.group("content")

    def test_dialect_fence_glued_to_marker(self):
        # `<<<REWRITE_FILE>>>```python` + path only in the @tests comment.
        raw = (
            "<<<REWRITE_FILE>>>```python\n"
            "# @tests: server/app/database.py\n"
            "import pytest\n\n"
            "def test_x():\n    assert True\n"
            "```"
        )
        assert patch_target_paths(raw) == set()          # reproduces the bug
        assert _REWRITE_RE.search(raw) is None
        body = self._parses_to(salvage_canonical_rewrite(raw, self.REL), self.REL)
        assert "@tests:" in body and "def test_x" in body
        assert "```" not in body                          # fence stripped

    def test_dialect_inline_path_on_marker(self):
        # `<<<REWRITE_FILE>>> path` with no file:/content: and no END marker.
        raw = (
            "<<<REWRITE_FILE>>> tests/unit/test_database.py\n"
            "# @tests: server/app/database.py\n"
            "def test_y():\n    assert 1 == 1\n"
        )
        assert patch_target_paths(raw) == set()
        body = self._parses_to(salvage_canonical_rewrite(raw, self.REL), self.REL)
        assert "def test_y" in body

    def test_canonical_input_still_parses_after_salvage(self):
        raw = (
            "<<<REWRITE_FILE>>>\n"
            "file: tests/unit/test_database.py\n"
            "content:\n"
            "# @tests: server/app/database.py\n"
            "def test_z():\n    assert True\n"
            "<<<END_REWRITE_FILE>>>"
        )
        # Canonical input already parses; salvage must not corrupt it.
        assert patch_target_paths(raw) == {self.REL}
        self._parses_to(salvage_canonical_rewrite(raw, self.REL), self.REL)

    def test_salvage_forces_known_target_over_wrong_inline_path(self):
        raw = (
            "<<<REWRITE_FILE>>> some/wrong/other_path.py\n"
            "# @tests: server/app/database.py\n"
            "def test_a():\n    assert True\n"
        )
        # Even if the model names the wrong path, salvage binds to rel.
        self._parses_to(salvage_canonical_rewrite(raw, self.REL), self.REL)

    def test_no_marker_returns_none(self):
        assert salvage_canonical_rewrite("just prose, no block", self.REL) is None

    def test_empty_body_returns_none(self):
        assert salvage_canonical_rewrite("<<<REWRITE_FILE>>>```python\n```", self.REL) is None
        assert salvage_canonical_rewrite("<<<REWRITE_FILE>>>\n\n\n", self.REL) is None


class TestMessageAssembly:
    def test_leads_with_code_contract(self):
        msgs = build_regeneration_messages(
            test_rel_path="tests/t.py",
            test_source="# @tests: app/m.py\ndef test_x(): pass",
            code_module_path="app/m.py",
            code_module_source="class Widget:\n    def go(self): ...",
            module_symbols=["Widget"],
            unsat_reason="contradiction",
            failing_output="AssertionError",
            spec_tiebreaker="SRS: at least one field required",
        )
        joined = " ".join(m["content"] for m in msgs)
        # code module + symbols present; test author framing; spec labelled tiebreaker
        assert "app/m.py" in joined
        assert "class Widget" in joined
        assert "Widget" in joined
        assert "TEST AUTHOR" in joined
        assert "TIEBREAKER ONLY" in joined  # the spec section header
        assert "contradiction" in joined

    def test_spec_section_omitted_when_absent(self):
        msgs = build_regeneration_messages(
            test_rel_path="tests/t.py", test_source="x",
            code_module_path="app/m.py", code_module_source="def f(): ...",
            module_symbols=["f"], unsat_reason="r", failing_output="o",
        )
        joined = " ".join(m["content"] for m in msgs)
        # the spec section is only injected when a tiebreaker is supplied
        assert "TIEBREAKER ONLY — do not cite" not in joined


# --- node-level: gate deferrals route back to the ladder (no crash) ---

class _FakeResp:
    def __init__(self, content):
        self.content = content
        self.usage = {}


class _FakeGateway:
    def __init__(self, content):
        self._content = content
    async def dispatch(self, **kw):
        return _FakeResp(self._content), kw.get("budget_remaining_usd", 1.0)
    def aggregate_tokens(self, tt, usage):
        return tt


def _state(ws, rel, content_reason="contradictory pair"):
    return {
        "workspace_path": ws,
        "node_state": {"unsatisfiable_test": rel,
                       "unsatisfiable_test_reason": content_reason},
        "loop_counter": {},
        "test_regeneration_config": {"enabled": True, "max_attempts_per_test": 1,
                                     "require_code_linkage": True,
                                     "coverage_nonregression": True},
        "messages": [{"role": "system", "content": "SRS spec"}],
        "budget_remaining_usd": 5.0,
        "compiler_errors": [],
        "modified_files": [],
    }


@pytest.mark.asyncio
async def test_node_rejects_stray_file(monkeypatch):
    with tempfile.TemporaryDirectory() as ws:
        rel = "tests/t.py"
        os.makedirs(os.path.join(ws, "tests"))
        open(os.path.join(ws, rel), "w").write(ORIGINAL)
        import harness.graph as g
        monkeypatch.setattr(g, "get_gateway",
                            lambda: _FakeGateway("file: server/app.py\ncontent: x"))
        out = await regeneration_node(_state(ws, rel))
        # stray file target → give up (no unsatisfiable_test re-emitted)
        assert out["node_state"]["test_regeneration"]["status"] == "targeted_other_files"
        assert out["loop_counter"]["test_regen_attempts"][rel] == 1


@pytest.mark.asyncio
async def test_node_rejects_missing_code_linkage(monkeypatch):
    with tempfile.TemporaryDirectory() as ws:
        rel = "tests/t.py"
        os.makedirs(os.path.join(ws, "tests"))
        open(os.path.join(ws, rel), "w").write("# @tests: app/m.py\n" + ORIGINAL)
        import harness.graph as g
        # targets the right file but drops the @tests marker
        patch = f"<<<REWRITE_FILE>>>\nfile: {rel}\ncontent:\ndef test_a():\n    assert True\n<<<END_REWRITE_FILE>>>"
        monkeypatch.setattr(g, "get_gateway", lambda: _FakeGateway(patch))
        out = await regeneration_node(_state(ws, rel))
        assert out["node_state"]["test_regeneration"]["status"] == "no_code_linkage"
        # rolled back to original (marker restored)
        assert "@tests: app/m.py" in open(os.path.join(ws, rel)).read()


@pytest.mark.asyncio
async def test_node_rejects_gutted_coverage(monkeypatch):
    with tempfile.TemporaryDirectory() as ws:
        rel = "tests/t.py"
        os.makedirs(os.path.join(ws, "tests"))
        open(os.path.join(ws, rel), "w").write("# @tests: app/m.py\n" + ORIGINAL)
        import harness.graph as g
        # keeps the marker but guts every assertion → coverage gate rejects
        gutted = "# @tests: app/m.py\nclass T:\n    def test_a(self):\n        pass\n"
        patch = f"<<<REWRITE_FILE>>>\nfile: {rel}\ncontent:\n{gutted}<<<END_REWRITE_FILE>>>"
        monkeypatch.setattr(g, "get_gateway", lambda: _FakeGateway(patch))
        out = await regeneration_node(_state(ws, rel))
        assert out["node_state"]["test_regeneration"]["status"] == "coverage_regression"
        assert ORIGINAL.strip() in open(os.path.join(ws, rel)).read()


@pytest.mark.asyncio
async def test_node_happy_path_regenerates(monkeypatch):
    with tempfile.TemporaryDirectory() as ws:
        rel = "tests/t.py"
        os.makedirs(os.path.join(ws, "tests"))
        os.makedirs(os.path.join(ws, "app"))
        open(os.path.join(ws, "app/m.py"), "w").write("class Widget:\n    def go(self):\n        return 1\n")
        open(os.path.join(ws, rel), "w").write("# @tests: app/m.py\n" + ORIGINAL)
        import harness.graph as g
        good = (
            "# @tests: app/m.py\n"
            "class TestWidget:\n"
            "    def test_go(self):\n"
            "        assert Widget().go() == 1\n"
            "    def test_type(self):\n"
            "        assert isinstance(Widget(), Widget)\n"
        )
        patch = f"<<<REWRITE_FILE>>>\nfile: {rel}\ncontent:\n{good}<<<END_REWRITE_FILE>>>"
        monkeypatch.setattr(g, "get_gateway", lambda: _FakeGateway(patch))
        out = await regeneration_node(_state(ws, rel))
        assert out["node_state"]["test_regeneration"]["status"] == "regenerated"
        assert out["node_state"]["test_regeneration"]["code_module"] == "app/m.py"
        # unsatisfiable flag cleared (not re-emitted)
        assert "unsatisfiable_test" not in out["node_state"]
        assert out["loop_counter"]["test_regen_attempts"][rel] == 1


@pytest.mark.asyncio
async def test_node_no_unsatisfiable_is_noop(monkeypatch):
    with tempfile.TemporaryDirectory() as ws:
        st = _state(ws, "")
        st["node_state"] = {}
        out = await regeneration_node(st)
        assert "unsatisfiable_test" not in out.get("node_state", {})


class _QueueResp:
    """Response stand-in exposing an empty/queued content + reasoning channel."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.reasoning_content = "planned the fix but emitted no block"
        self.usage = {}


class _QueueGateway:
    """Records each dispatch's messages; returns queued contents (then '').

    Distinct from the module's single-shot ``_FakeGateway`` — this one lets a
    test drive a sequence of responses (e.g. empty→empty) and count calls.
    """

    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = []

    async def dispatch(self, messages=None, role=None, budget_remaining_usd=1.0):
        self.calls.append(messages)
        c = self._contents.pop(0) if self._contents else ""
        return _QueueResp(c), budget_remaining_usd

    def aggregate_tokens(self, tt, usage):
        return tt or {}


def _regen_setup(ws, rel):
    """Write app/m.py + a mapped defective test at rel; return node state."""
    os.makedirs(os.path.join(ws, "tests"), exist_ok=True)
    os.makedirs(os.path.join(ws, "app"), exist_ok=True)
    with open(os.path.join(ws, "app/m.py"), "w") as fh:
        fh.write("class Widget:\n    def go(self):\n        return 1\n")
    with open(os.path.join(ws, rel), "w") as fh:
        fh.write("# @tests: app/m.py\n" + ORIGINAL)
    return _state(ws, rel)


class TestEmptyContentRetry:
    """lumina 019fd587: deepseek-v4-pro front-loaded its whole response into the
    reasoning channel and returned an EMPTY content — no REWRITE_FILE block —
    silently burning the one regeneration attempt. The node now retries once
    with a block-only nudge before giving up."""

    @pytest.mark.asyncio
    async def test_empty_first_response_triggers_one_retry(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws:
            rel = "tests/t.py"
            state = _regen_setup(ws, rel)
            fake = _QueueGateway(contents=["", ""])  # empty both times
            import harness.graph as g
            monkeypatch.setattr(g, "get_gateway", lambda: fake)
            result = await regeneration_node(state)
        # Retry fired — dispatched exactly twice (not the old single shot).
        assert len(fake.calls) == 2
        # The second dispatch carried the block-only nudge (which now restates
        # the exact canonical grammar).
        assert any(
            isinstance(m, dict)
            and "Output ONLY this" in str(m.get("content", ""))
            and "<<<END_REWRITE_FILE>>>" in str(m.get("content", ""))
            for m in fake.calls[1]
        )
        # Both empty ⇒ still gives up as no_patch (deferring to the ladder).
        assert result["node_state"]["test_regeneration"]["status"] == "no_patch"

    @pytest.mark.asyncio
    async def test_no_retry_when_first_response_has_block(self, monkeypatch):
        good = (
            "# @tests: app/m.py\n"
            "class TestWidget:\n"
            "    def test_go(self):\n"
            "        assert Widget().go() == 1\n"
            "    def test_type(self):\n"
            "        assert isinstance(Widget(), Widget)\n"
        )
        block = f"<<<REWRITE_FILE>>>\nfile: tests/t.py\ncontent:\n{good}<<<END_REWRITE_FILE>>>"
        with tempfile.TemporaryDirectory() as ws:
            rel = "tests/t.py"
            state = _regen_setup(ws, rel)
            fake = _QueueGateway(contents=[block])
            import harness.graph as g
            monkeypatch.setattr(g, "get_gateway", lambda: fake)
            await regeneration_node(state)
        # A good first response must NOT trigger a second dispatch.
        assert len(fake.calls) == 1
