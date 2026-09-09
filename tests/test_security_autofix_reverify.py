"""Security autofixes edit source on a green workspace and must be re-verified.

``security_scan_node``'s deterministic autofix pass (bandit B201/B602,
gitleaks line removal, trivy dependency bumps) mutates real source AFTER the
build went green. Any of those edits can break it — removing a secret-bearing
line, bumping a pinned dependency, rewriting a subprocess call.

Nothing re-compiled them. The only re-verify path in
``route_after_security_scan`` is the ``pre_exit_verify`` branch, which needs
BOTH an opt-in (``compiler.pre_exit_verify``, off by default) AND a non-empty
``pending_mutations`` — a list only ``code_review_node`` ever wrote to. So a
comment in security.py promising "routing then sends the build back through
compile + security_scan so we can confirm the fixes hold" described behaviour
that existed under no configuration at all.

The autofix path now sets ``autofix_needs_reverify``; the router honours it
unconditionally and ``compiler_node`` clears it, so one batch of autofixes
buys exactly one re-verify and the path cannot loop.
"""

from __future__ import annotations

from harness.graph import route_after_security_scan


def _state(**over):
    base = {
        "budget_remaining_usd": 5.0,
        "loop_counter": {"security": 0},
        "compiler_errors": [],
        "node_state": {},
        "pending_mutations": [],
    }
    base.update(over)
    return base


def test_autofix_routes_back_to_the_compiler():
    state = _state(node_state={
        "security_scan": {
            "passed": True,
            "autofix_applied": 2,
            "autofix_needs_reverify": True,
            "autofix_files": ["server/app/db.py", "requirements.txt"],
        },
    })
    assert route_after_security_scan(state) == "compiler_node"


def test_no_reverify_when_autofix_changed_nothing():
    # A clean scan with no autofixes must keep its existing terminal routing.
    state = _state(node_state={"security_scan": {"passed": True}})
    assert route_after_security_scan(state) != "compiler_node"


def test_reverify_does_not_depend_on_the_pre_exit_verify_opt_in():
    # The old path needed compiler.pre_exit_verify AND pending_mutations.
    # This one is unconditional.
    state = _state(
        pending_mutations=[],
        node_state={"security_scan": {"autofix_needs_reverify": True}},
    )
    assert route_after_security_scan(state) == "compiler_node"


def test_budget_exhaustion_still_wins_over_reverify():
    state = _state(
        budget_remaining_usd=0.0,
        node_state={"security_scan": {"autofix_needs_reverify": True}},
    )
    assert route_after_security_scan(state) == "human_intervention_node"


def test_malformed_security_node_state_is_ignored():
    state = _state(node_state={"security_scan": "not a dict"})
    assert route_after_security_scan(state) != "compiler_node"


class TestOneShotContract:
    """The flag must be cleared by the compile it triggers, or the router
    would send every subsequent pass back and loop compiler ⇄ security_scan."""

    def test_compiler_node_clears_the_flag(self):
        src = open("harness/graph.py", encoding="utf-8").read()
        assert '_sec_ns_clear.pop("autofix_needs_reverify", None)' in src

    def test_autofix_files_are_registered_as_pending_mutations(self):
        # So the opt-in pre-exit check can see them too — it never could
        # before, because only code_review populated that list.
        src = open("harness/security.py", encoding="utf-8").read()
        assert '"pending_mutations": _prior_pending + [' in src

    def test_flag_only_set_when_fixes_actually_applied(self):
        src = open("harness/security.py", encoding="utf-8").read()
        i = src.index('"autofix_needs_reverify": True')
        guard = src.rindex("if applied_fixes and not unhandled_diagnostics:", 0, i)
        assert guard < i
