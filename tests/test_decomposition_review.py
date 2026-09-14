"""Tests for the ADR-0007 decomposition-quality review (Phase 1).

Deterministic tier is pure; the LLM tier uses a fake gateway.
"""

from __future__ import annotations

import json

import pytest

from harness import decomposition_review as dr


def _story(key, title="T", *, deps=None, scope=None, acs=("does a thing",), desc=""):
    return {
        "story_key": key, "title": title, "description": desc,
        "feature_key": "FEAT-001",
        "depends_on": list(deps or []), "scope_files": list(scope or []),
        "acceptance_criteria": list(acs),
    }


# ---------------------------------------------------------------------------
# Deterministic tier
# ---------------------------------------------------------------------------


class TestDeterministic:
    def test_clean_decomposition_no_findings(self):
        stories = [_story("STORY-1"), _story("STORY-2", deps=["STORY-1"])]
        assert dr.deterministic_findings(stories) == []

    def test_dangling_dependency(self):
        f = dr.deterministic_findings([_story("STORY-1", deps=["STORY-9"])])
        assert len(f) == 1
        assert f[0]["dimension"] == "dependency" and f[0]["severity"] == "high"
        assert "STORY-9" in f[0]["problem"]

    def test_circular_dependency(self):
        stories = [
            _story("STORY-1", deps=["STORY-2"]),
            _story("STORY-2", deps=["STORY-1"]),
        ]
        f = dr.deterministic_findings(stories)
        cyc = [x for x in f if "circular" in x["problem"]]
        assert len(cyc) == 1

    def test_zero_ac_story(self):
        f = dr.deterministic_findings([_story("STORY-1", acs=())])
        assert any(x["dimension"] == "ac_quality" and "no acceptance" in x["problem"] for x in f)

    def test_empty_ac(self):
        f = dr.deterministic_findings([_story("STORY-1", acs=("real one", "   "))])
        assert any("empty/blank" in x["problem"] for x in f)

    def test_duplicate_story_same_title_and_scope(self):
        stories = [
            _story("STORY-1", title="Add a contact", scope=["api/contacts.py"]),
            _story("STORY-2", title="Add a contact!", scope=["api/contacts.py"]),
        ]
        f = dr.deterministic_findings(stories)
        dup = [x for x in f if x["dimension"] == "overlap"]
        assert len(dup) == 1 and dup[0]["suggested_action"] == "merge"

    def test_same_title_disjoint_scope_not_duplicate(self):
        stories = [
            _story("STORY-1", title="Setup", scope=["a.py"]),
            _story("STORY-2", title="Setup", scope=["b.py"]),
        ]
        # different scope → not flagged (avoid false positives)
        assert not [x for x in dr.deterministic_findings(stories) if x["dimension"] == "overlap"]

    def test_cycle_detection_no_false_positive_on_dag(self):
        stories = [
            _story("STORY-1"),
            _story("STORY-2", deps=["STORY-1"]),
            _story("STORY-3", deps=["STORY-1", "STORY-2"]),
        ]
        assert not [x for x in dr.deterministic_findings(stories) if "circular" in x["problem"]]


# ---------------------------------------------------------------------------
# LLM tier
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, content):
        self.content = content
        self.usage = {}


class _FakeGateway:
    def __init__(self, content="", *, has_model=True, raise_exc=None):
        self._content, self._has, self._raise = content, has_model, raise_exc

    def select_model(self, role):
        return "some:model" if self._has else ""

    async def dispatch(self, *, messages, role, budget_remaining_usd, **kw):
        if self._raise:
            raise self._raise
        return _Resp(self._content), budget_remaining_usd - 0.01


@pytest.mark.asyncio
class TestLLMReview:
    async def test_parses_and_filters_findings(self):
        stories = [_story("STORY-1"), _story("STORY-2")]
        payload = json.dumps([
            {"story_key": "STORY-1", "dimension": "right_sizing", "severity": "high",
             "problem": "too big", "suggested_action": "split"},
            {"story_key": "STORY-9", "dimension": "overlap", "severity": "low",
             "problem": "bogus key", "suggested_action": "merge"},  # dropped: unknown key
            {"story_key": "STORY-2", "dimension": "nonsense", "severity": "high",
             "problem": "bad dim", "suggested_action": "split"},  # dropped: bad dimension
        ])
        gw = _FakeGateway(payload)
        findings, budget = await dr.review_decomposition_quality(gw, stories, 1.0)
        assert [f["story_key"] for f in findings] == ["STORY-1"]
        assert findings[0]["source"] == "llm"
        assert budget == pytest.approx(0.99)

    async def test_no_reviewer_model_skips(self):
        gw = _FakeGateway(has_model=False)
        findings, budget = await dr.review_decomposition_quality(gw, [_story("STORY-1")], 1.0)
        assert findings == [] and budget == 1.0

    async def test_non_json_is_fail_open(self):
        gw = _FakeGateway("not json at all")
        findings, _ = await dr.review_decomposition_quality(gw, [_story("STORY-1")], 1.0)
        assert findings == []

    async def test_dispatch_error_is_fail_open(self):
        gw = _FakeGateway(raise_exc=RuntimeError("boom"))
        findings, budget = await dr.review_decomposition_quality(gw, [_story("STORY-1")], 1.0)
        assert findings == [] and budget == 1.0

    async def test_bad_severity_and_action_defaulted(self):
        payload = json.dumps([{"story_key": "STORY-1", "dimension": "ac_quality",
                               "severity": "catastrophic", "problem": "x",
                               "suggested_action": "explode"}])
        findings, _ = await dr.review_decomposition_quality(_FakeGateway(payload), [_story("STORY-1")], 1.0)
        assert findings[0]["severity"] == "medium"  # invalid -> medium
        assert findings[0]["suggested_action"] == "resize"  # invalid -> resize


# ---------------------------------------------------------------------------
# Node guard + doc rendering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestNode:
    async def test_disabled_is_passthrough(self):
        out = await dr.decomposition_quality_review_node(
            {"decomposition_config": {"quality_review": False}})
        assert out["node_state"]["skipped"] is True
        assert "exit_code" not in out


class TestRouting:
    def test_gap_routes_to_end(self):
        from langgraph.graph import END
        assert dr.route_after_decomposition_quality(
            {"node_state": {"decomposition_quality_gap": True}}) == END

    def test_clean_routes_to_gate(self):
        assert dr.route_after_decomposition_quality({"node_state": {}}) == "human_gatekeeper_node"


class TestDoc:
    def test_renders_table_with_findings(self):
        findings = [dr._finding("STORY-1", "right_sizing", "high", "too big", "split", source="llm")]
        doc = dr.render_review_doc(findings, 3)
        assert "Decomposition Quality Review" in doc
        assert "STORY-1" in doc and "right_sizing" in doc and "split" in doc

    def test_renders_clean(self):
        assert "well-formed" in dr.render_review_doc([], 5)


# ---------------------------------------------------------------------------
# ADR-0004 fan-out vs. the overlap dimension.
#
# The embedder attaches one non-functional policy to every story it
# constrains, tagging each copy ``[NFR:<policy>]``. A reviewer that has not
# internalised that reads the repeated criteria as duplication and
# recommends merging stories ADR-0004 requires to stay separate — two
# harness features contradicting each other with the operator left to
# arbitrate.
#
# lumina-fresh-20260911-1107 produced four of these in one pass:
#   STORY-001 [overlap/high]: STORY-001 and STORY-002 both contain
#   identical NFR-002 security ACs for POST /api/birthdays -> merge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestNfrFanoutIsNotOverlap:

    async def test_the_lumina_overlap_finding_is_dropped(self):
        stories = [_story("STORY-001"), _story("STORY-002")]
        payload = json.dumps([{
            "story_key": "STORY-001", "dimension": "overlap",
            "severity": "high",
            "problem": (
                "STORY-001 and STORY-002 both contain identical NFR-002 "
                "security ACs for POST /api/birthdays, duplicating the same "
                "behavior across stories."
            ),
            "suggested_action": "merge",
        }])
        findings, _ = await dr.review_decomposition_quality(
            _FakeGateway(payload), stories, 1.0,
        )
        assert findings == []

    async def test_bracket_tag_form_is_dropped(self):
        stories = [_story("STORY-001")]
        payload = json.dumps([{
            "story_key": "STORY-001", "dimension": "overlap",
            "severity": "high",
            "problem": "Shared [NFR:NFR-002] criterion appears in 5 stories.",
            "suggested_action": "merge",
        }])
        findings, _ = await dr.review_decomposition_quality(
            _FakeGateway(payload), stories, 1.0,
        )
        assert findings == []

    async def test_genuine_overlap_still_reported(self):
        """The filter is narrow — an overlap finding that does not rest on
        NFR fan-out must survive, or the dimension becomes useless."""
        stories = [_story("STORY-001"), _story("STORY-002")]
        payload = json.dumps([{
            "story_key": "STORY-001", "dimension": "overlap",
            "severity": "high",
            "problem": "Both stories implement the same delete endpoint.",
            "suggested_action": "merge",
        }])
        findings, _ = await dr.review_decomposition_quality(
            _FakeGateway(payload), stories, 1.0,
        )
        assert len(findings) == 1
        assert findings[0]["dimension"] == "overlap"

    async def test_other_dimensions_are_untouched_by_the_filter(self):
        """An ac_quality finding ABOUT an NFR criterion is legitimate — the
        criterion came from a real NFR story and may genuinely be vague.
        Only the overlap dimension is filtered."""
        stories = [_story("STORY-001")]
        payload = json.dumps([{
            "story_key": "STORY-001", "dimension": "ac_quality",
            "severity": "high",
            "problem": "NFR-002 criterion 'is secure' has no observable outcome.",
            "suggested_action": "rewrite_ac",
        }])
        findings, _ = await dr.review_decomposition_quality(
            _FakeGateway(payload), stories, 1.0,
        )
        assert len(findings) == 1
        assert findings[0]["dimension"] == "ac_quality"


def test_rubric_states_the_nfr_exception():
    """The deterministic filter is a safety net; the rubric is the fix.

    Without this line the reviewer has no way to tell deliberate ADR-0004
    fan-out from planner sloppiness.
    """
    assert "[NFR:<policy>]" in dr._QUALITY_RUBRIC
    assert "NEVER report overlap" in dr._QUALITY_RUBRIC


# ---------------------------------------------------------------------------
# Phase 2 — bounded AC remediation (ADR-0007).
#
# `quality_enforce` is binary: discard every finding, or fail the build with
# exit_code=1 and a "revise the spec by hand" message.
# lumina-fresh-20260911-1107 produced 13 ac_quality/high findings on an
# ordinary spec, so enforcing would have bricked the run at decomposition —
# strictly worse than ignoring them. They were ignored, and the vague
# criteria they named went on to feed test generation.
#
# The safety contract is the point of these tests: rewriting AC text must
# not disturb anything that currently works.
# ---------------------------------------------------------------------------

def _finding_rewrite(story_key: str, sev: str = "high") -> dict:
    return dr._finding(
        story_key, "ac_quality", sev,
        "AC 'Dashboard handles no upcoming birthdays' is not testable.",
        "rewrite_ac", source="llm",
    )


class TestRemediableSelection:
    """Only high-severity AC rewrites qualify. Structural recommendations
    would reshape the decomposition and invalidate story keys downstream
    nodes have already committed to."""

    def test_high_ac_quality_rewrite_qualifies(self):
        assert dr._remediable_findings([_finding_rewrite("STORY-001")])

    def test_medium_does_not_qualify(self):
        assert not dr._remediable_findings(
            [_finding_rewrite("STORY-001", sev="medium")]
        )

    @pytest.mark.parametrize("dim,action", [
        ("right_sizing", "split"),
        ("overlap", "merge"),
        ("dependency", "add_dependency"),
        ("balance", "resize"),
    ])
    def test_structural_dimensions_are_never_auto_applied(self, dim, action):
        f = dr._finding("STORY-001", dim, "high", "x", action, source="llm")
        assert not dr._remediable_findings([f])


class TestNfrCriteriaAreNeverRewritten:
    """ADR-0004 owns `[NFR:...]` text and fans identical copies across every
    story a policy constrains. Rewriting one copy desynchronises the rest
    and breaks the single-ownership the tag exists to express."""

    def test_tagged_criterion_is_excluded(self):
        assert not dr._ac_is_remediable(
            "[NFR:NFR-002] Input is validated before persistence"
        )

    def test_untagged_criterion_is_included(self):
        assert dr._ac_is_remediable("POST /api/birthdays returns 201")

    def test_tag_must_be_a_prefix_not_a_mention(self):
        """A criterion that merely mentions NFR-002 in passing is ordinary
        text and stays remediable."""
        assert dr._ac_is_remediable("Latency meets the NFR-002 budget")


@pytest.mark.asyncio
class TestRemediationWritePath:

    def _db(self, tmp_path):
        """A real story DB with one story and two ACs, one NFR-tagged."""
        from harness import story_state
        ws = str(tmp_path)
        app = story_state.app_name_for_workspace(ws)
        conn = story_state.open_story_db(workspace_path=ws)
        story_state.ensure_feature(conn, app, "core", name="Core",
                                   description="d")
        keys = story_state.create_stories(conn, app, [{
            "title": "View upcoming birthdays", "feature": "core",
            "acceptance_criteria": [
                "Dashboard handles no upcoming birthdays",
                "[NFR:NFR-002] Input is validated before persistence",
            ],
            "depends_on": [], "scope_files": [],
        }])
        return ws, app, conn, keys[0]

    async def test_rewrites_text_and_preserves_row_id(self, tmp_path):
        """The whole safety argument: `test_verifies_ac` edges point at
        `acceptance_criteria.id`, so the row id MUST survive the rewrite."""
        from harness import story_state
        ws, app, conn, skey = self._db(tmp_path)
        story = next(
            s for s in story_state.list_stories(conn, app)
            if s["story_key"] == skey
        )
        before = story_state.list_acceptance_criteria(conn, app, int(story["id"]))
        target = next(a for a in before if not a["text"].startswith("[NFR:"))
        payload = json.dumps([{
            "ac_key": target["ac_key"],
            "text": "GET /api/birthdays/upcoming returns 200 with an empty list",
        }])
        n, _ = await dr.remediate_acceptance_criteria(
            _FakeGateway(payload), conn, app,
            [{"story_key": skey}], [_finding_rewrite(skey)], 1.0,
        )
        assert n == 1
        after = story_state.list_acceptance_criteria(conn, app, int(story["id"]))
        rewritten = next(a for a in after if a["ac_key"] == target["ac_key"])
        assert rewritten["text"].startswith("GET /api/birthdays/upcoming")
        assert rewritten["id"] == target["id"]          # edges survive
        assert rewritten["ordinal"] == target["ordinal"]
        assert len(after) == len(before)                # coverage unchanged
        conn.close()

    async def test_nfr_criterion_is_left_alone(self, tmp_path):
        from harness import story_state
        ws, app, conn, skey = self._db(tmp_path)
        story = next(
            s for s in story_state.list_stories(conn, app)
            if s["story_key"] == skey
        )
        nfr = next(
            a for a in story_state.list_acceptance_criteria(
                conn, app, int(story["id"]))
            if a["text"].startswith("[NFR:")
        )
        # Even if the model tries, the key was never offered to it and the
        # write path re-checks.
        payload = json.dumps([{"ac_key": nfr["ac_key"], "text": "rewritten"}])
        n, _ = await dr.remediate_acceptance_criteria(
            _FakeGateway(payload), conn, app,
            [{"story_key": skey}], [_finding_rewrite(skey)], 1.0,
        )
        after = next(
            a for a in story_state.list_acceptance_criteria(
                conn, app, int(story["id"]))
            if a["ac_key"] == nfr["ac_key"]
        )
        assert after["text"].startswith("[NFR:NFR-002]")
        conn.close()

    async def test_a_rewrite_may_not_forge_an_nfr_tag(self, tmp_path):
        """A model that prefixes `[NFR:...]` onto a plain criterion would
        forge ADR-0004 ownership the embedder never granted."""
        from harness import story_state
        ws, app, conn, skey = self._db(tmp_path)
        story = next(
            s for s in story_state.list_stories(conn, app)
            if s["story_key"] == skey
        )
        target = next(
            a for a in story_state.list_acceptance_criteria(
                conn, app, int(story["id"]))
            if not a["text"].startswith("[NFR:")
        )
        payload = json.dumps([{
            "ac_key": target["ac_key"], "text": "[NFR:NFR-009] forged",
        }])
        n, _ = await dr.remediate_acceptance_criteria(
            _FakeGateway(payload), conn, app,
            [{"story_key": skey}], [_finding_rewrite(skey)], 1.0,
        )
        assert n == 0
        conn.close()

    async def test_no_findings_means_no_dispatch(self, tmp_path):
        ws, app, conn, skey = self._db(tmp_path)
        gw = _FakeGateway("[]")
        n, budget = await dr.remediate_acceptance_criteria(
            gw, conn, app, [{"story_key": skey}], [], 1.0,
        )
        assert (n, budget) == (0, 1.0)
        conn.close()

    async def test_non_json_response_leaves_criteria_untouched(self, tmp_path):
        from harness import story_state
        ws, app, conn, skey = self._db(tmp_path)
        story = next(
            s for s in story_state.list_stories(conn, app)
            if s["story_key"] == skey
        )
        before = story_state.list_acceptance_criteria(conn, app, int(story["id"]))
        n, _ = await dr.remediate_acceptance_criteria(
            _FakeGateway("I think these look fine actually"), conn, app,
            [{"story_key": skey}], [_finding_rewrite(skey)], 1.0,
        )
        assert n == 0
        after = story_state.list_acceptance_criteria(conn, app, int(story["id"]))
        assert [a["text"] for a in after] == [a["text"] for a in before]
        conn.close()

    async def test_dispatch_failure_is_fail_open(self, tmp_path):
        ws, app, conn, skey = self._db(tmp_path)
        gw = _FakeGateway("", raise_exc=RuntimeError("upstream 503"))
        n, budget = await dr.remediate_acceptance_criteria(
            gw, conn, app, [{"story_key": skey}], [_finding_rewrite(skey)], 1.0,
        )
        assert n == 0 and budget == 1.0
        conn.close()

    async def test_unchanged_text_is_not_counted_as_a_rewrite(self, tmp_path):
        """The prompt tells the model to return the original when it cannot
        improve a criterion. That is a no-op, not a fix."""
        from harness import story_state
        ws, app, conn, skey = self._db(tmp_path)
        story = next(
            s for s in story_state.list_stories(conn, app)
            if s["story_key"] == skey
        )
        target = next(
            a for a in story_state.list_acceptance_criteria(
                conn, app, int(story["id"]))
            if not a["text"].startswith("[NFR:")
        )
        payload = json.dumps([
            {"ac_key": target["ac_key"], "text": target["text"]},
        ])
        n, _ = await dr.remediate_acceptance_criteria(
            _FakeGateway(payload), conn, app,
            [{"story_key": skey}], [_finding_rewrite(skey)], 1.0,
        )
        assert n == 0
        conn.close()

    async def test_unknown_ac_key_is_ignored(self, tmp_path):
        ws, app, conn, skey = self._db(tmp_path)
        payload = json.dumps([{"ac_key": "STORY-999.AC-1", "text": "x"}])
        n, _ = await dr.remediate_acceptance_criteria(
            _FakeGateway(payload), conn, app,
            [{"story_key": skey}], [_finding_rewrite(skey)], 1.0,
        )
        assert n == 0
        conn.close()


@pytest.mark.asyncio
async def test_remediation_preserves_a_real_verification_edge(tmp_path):
    """The load-bearing safety claim, tested end to end rather than inferred.

    `test_verifies_ac` rows point at `acceptance_criteria.id`. If a rewrite
    deleted and recreated the row, the edge would cascade away and
    `teane audit` — which gates CI — would start reporting untested ACs
    that are in fact tested. Assert the edge itself survives, not just the
    row id.
    """
    from harness import story_state
    ws = str(tmp_path)
    app = story_state.app_name_for_workspace(ws)
    conn = story_state.open_story_db(workspace_path=ws)
    story_state.ensure_feature(conn, app, "core", name="Core", description="d")
    skey = story_state.create_stories(conn, app, [{
        "title": "View upcoming birthdays", "feature": "core",
        "acceptance_criteria": ["Dashboard handles no upcoming birthdays"],
        "depends_on": [], "scope_files": [],
    }])[0]
    story = next(
        s for s in story_state.list_stories(conn, app) if s["story_key"] == skey
    )
    ac = story_state.list_acceptance_criteria(conn, app, int(story["id"]))[0]

    # A test verifies this criterion, exactly as test_generation would record.
    assert story_state.link_test_to_ac(
        conn, app, "server/tests/test_dashboard.py", int(ac["id"]),
        test_function_name="test_empty_dashboard",
    )
    edges_before = conn.execute(
        "SELECT COUNT(*) FROM test_verifies_ac WHERE ac_id = ?", (int(ac["id"]),)
    ).fetchone()[0]
    assert edges_before == 1

    payload = json.dumps([{
        "ac_key": ac["ac_key"],
        "text": "GET /api/birthdays/upcoming returns 200 with an empty list",
    }])
    n, _ = await dr.remediate_acceptance_criteria(
        _FakeGateway(payload), conn, app,
        [{"story_key": skey}], [_finding_rewrite(skey)], 1.0,
    )
    assert n == 1

    after = story_state.list_acceptance_criteria(conn, app, int(story["id"]))[0]
    assert after["text"].startswith("GET /api/birthdays/upcoming")
    # The edge still resolves, still points at the same criterion, and the
    # criterion it names is the REWRITTEN one.
    edges_after = conn.execute(
        "SELECT test_path, test_function_name FROM test_verifies_ac "
        "WHERE ac_id = ?", (int(after["id"]),)
    ).fetchall()
    assert len(edges_after) == 1
    assert edges_after[0][0] == "server/tests/test_dashboard.py"
    assert edges_after[0][1] == "test_empty_dashboard"
    # And nothing was orphaned anywhere in the table.
    orphans = conn.execute(
        "SELECT COUNT(*) FROM test_verifies_ac t "
        "LEFT JOIN acceptance_criteria a ON a.id = t.ac_id "
        "WHERE a.id IS NULL"
    ).fetchone()[0]
    assert orphans == 0
    conn.close()
