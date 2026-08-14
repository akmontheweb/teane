"""Requirements-refinement drop-detection (#2).

The doc-reviewer critique for the REQUIREMENTS gate is fed the operator's
original product spec as the source of truth, so it can flag (and the
revise pass restore) any user-visible requirement the refinement dropped —
the "next-birthday date vanished from the contract" class of bug.
"""
from __future__ import annotations

import asyncio
import json
import os

from harness.gateway import NodeRole
from harness.graph import _read_product_spec_text, review_and_revise_spec


class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content
        self.usage = None


class _StubGateway:
    """Returns a canned critique for the DOC_REVIEWER call and a canned
    revised spec for the PLANNING (revise) call; records every dispatch."""

    def __init__(self, critique_json: str, revised_md: str) -> None:
        self.calls: list[tuple[object, list]] = []
        self._critique = critique_json
        self._revised = revised_md

    async def dispatch(self, messages, role, budget_remaining_usd):
        self.calls.append((role, messages))
        content = self._critique if role == NodeRole.DOC_REVIEWER else self._revised
        return _Resp(content), budget_remaining_usd - 0.01


def _run(gw, spec_path, gate, product_spec=""):
    return asyncio.run(review_and_revise_spec(
        spec_path, gate, gateway=gw, budget_remaining_usd=1.0,
        user_goal="build the app", original_product_spec=product_spec,
    ))


def _doc_user_prompts(gw) -> list[str]:
    out = []
    for role, messages in gw.calls:
        if role == NodeRole.DOC_REVIEWER and len(messages) >= 2:
            out.append(str(messages[1].get("content", "")))
    return out


def test_reader_reads_product_spec(tmp_path):
    os.makedirs(tmp_path / "product_spec")
    (tmp_path / "product_spec" / "product_spec.txt").write_text(
        "Dashboard shows the next birthday date."
    )
    assert "next birthday date" in _read_product_spec_text(str(tmp_path))
    assert _read_product_spec_text(str(tmp_path / "missing")) == ""


def test_requirements_injects_product_spec_and_surfaces_drops(tmp_path):
    spec = tmp_path / "SPEC_REQUIREMENTS.md"
    spec.write_text("# Refined spec that only mentions days_until_next")
    critique = json.dumps({
        "dropped_requirements": ["formatted next-birthday DATE not preserved"],
        "completeness": [], "followup_questions": [],
    })
    gw = _StubGateway(critique, "# Revised spec that restores the next-birthday date")
    res = _run(gw, str(spec), "REQUIREMENTS",
               product_spec="AC-1.1: show the formatted date of the next birthday")
    assert res["ok"]
    # The original product spec is injected as source-of-truth into the critique.
    prompts = _doc_user_prompts(gw)
    assert any("Original Product Spec" in p and "next birthday" in p for p in prompts)
    # Drops are surfaced on the result for callers/logging.
    assert res["dropped_requirements"] == ["formatted next-birthday DATE not preserved"]
    # The revised spec was written back to disk.
    assert "restores the next-birthday date" in spec.read_text()


def test_architecture_gate_does_not_inject_product_spec(tmp_path):
    spec = tmp_path / "SPEC_ARCHITECTURE.md"
    spec.write_text("# Architecture")
    gw = _StubGateway(json.dumps({"dropped_requirements": [], "followup_questions": []}),
                      "# Revised architecture")
    _run(gw, str(spec), "ARCHITECTURE", product_spec="anything")
    assert all("Original Product Spec" not in p for p in _doc_user_prompts(gw))


def test_requirements_without_product_spec_no_injection(tmp_path):
    spec = tmp_path / "SPEC_REQUIREMENTS.md"
    spec.write_text("# Refined")
    gw = _StubGateway(json.dumps({"followup_questions": []}), "# Revised")
    _run(gw, str(spec), "REQUIREMENTS", product_spec="")
    assert all("Original Product Spec" not in p for p in _doc_user_prompts(gw))


def test_architecture_gate_injects_requirements_as_source_of_truth(tmp_path):
    # Prompt defect #4: the architecture reviewer must be fed SPEC_REQUIREMENTS.md
    # so it can flag a policy the architecture contradicts (Feb-29 → March 1 in
    # the RSD vs Feb 28 in the architecture).
    (tmp_path / "SPEC_REQUIREMENTS.md").write_text(
        "R2: February 29 birthdays are observed on March 1 in non-leap years."
    )
    arch = tmp_path / "SPEC_ARCHITECTURE.md"
    arch.write_text("Feb-29 birthdays are observed on February 28 in non-leap years.")
    critique = json.dumps({
        "contradictions": ["arch says Feb 28 but requirements say March 1"],
        "followup_questions": [],
    })
    gw = _StubGateway(critique, "# Revised architecture that observes March 1")
    res = _run(gw, str(arch), "ARCHITECTURE")
    assert res["ok"]
    prompts = _doc_user_prompts(gw)
    # The requirements doc is injected as the behavioural source of truth, with
    # the contradiction instruction and its actual content.
    assert any(
        "BEHAVIOURAL SOURCE OF TRUTH" in p and "March 1" in p for p in prompts
    )


def test_architecture_gate_without_requirements_file_no_injection(tmp_path):
    # No SPEC_REQUIREMENTS.md next to the arch doc → no block (graceful).
    arch = tmp_path / "SPEC_ARCHITECTURE.md"
    arch.write_text("# Architecture")
    gw = _StubGateway(json.dumps({"followup_questions": []}), "# Revised")
    _run(gw, str(arch), "ARCHITECTURE")
    assert all(
        "BEHAVIOURAL SOURCE OF TRUTH" not in p for p in _doc_user_prompts(gw)
    )


def test_arch_doc_skill_forbids_re_deciding_requirements_policy():
    # The generation-side root cause: the architecture skill must forbid
    # re-deciding a behavioural policy the requirements already fixed.
    import pathlib
    skill = pathlib.Path("harness/skills/docgen/arch_doc.md").read_text()
    assert "Requirements are authoritative on BEHAVIOUR" in skill
    assert "NEVER re-decide" in skill
    assert "February-29" in skill or "Feb-29" in skill
