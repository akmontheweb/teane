"""Paired producer+consumer decomposition (#3).

A cross-cutting client<->server requirement (an API the UI calls, an
auth/CSRF token it fetches) must be decomposed with BOTH sides in scope.
The lumina CSRF split-brain shipped a frontend `/api/csrf-token` fetch with
no backend endpoint, marked "done" — a read-only app. These guards steer
the planner to pair the sides and let the semantic reviewer flag one-sided
contracts.
"""
from __future__ import annotations

from harness.decomposition import _build_decomposition_prompt
from harness.semantic_review import _build_review_prompt


def test_decomposition_prompt_requires_paired_contracts(tmp_path):
    prompt = _build_decomposition_prompt(
        "FR-1 login with CSRF-protected form", "React client + FastAPI server",
        str(tmp_path),
    )
    assert "CROSS-CUTTING CONTRACTS" in prompt
    assert "PRODUCER" in prompt and "CONSUMER" in prompt
    # Names the concrete failure and the remedy (pair or depends_on).
    assert "404" in prompt
    assert "depends_on" in prompt


def test_semantic_review_flags_one_sided_contracts():
    prompt = _build_review_prompt([
        {"req_key": "FEAT-1", "title": "Auth", "intent": "login", "stories": []},
    ])
    low = prompt.lower()
    assert "client" in low and "contract" in low
    assert "csrf" in low
    assert "missing side" in low
