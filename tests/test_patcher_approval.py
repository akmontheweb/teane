"""Phase 5 regression: PR-style diff approval before commit.

Covers three surfaces:
1. The serializer (`_serialise_blocks_for_approval`) — payload shape
   the dashboard renders.
2. The decision interpreter (`_apply_approval_decision`) — "approve"
   returns everything, "reject" returns nothing, anything else
   defaults to approve (safe default).
3. End-to-end: `apply_patch_blocks` with `require_approval=True`
   defers to a stubbed HITL channel; approve → writes land; reject
   → workspace unchanged, results record a rejection message.
"""

from __future__ import annotations

import asyncio


from harness import hitl as _hitl_mod
from harness.patcher import (
    OperationType,
    PatchBlock,
    _apply_approval_decision,
    _serialise_blocks_for_approval,
    apply_patch_blocks,
)


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------

def test_serialise_replace_block_carries_search_and_replace(tmp_path):
    b = PatchBlock(
        operation=OperationType.REPLACE_BLOCK,
        file="src/main.py",
        search="print('old')",
        replace="print('new')",
    )
    [payload] = _serialise_blocks_for_approval([b], str(tmp_path))
    assert payload["path"] == "src/main.py"
    assert payload["operation"] == "replace_block"
    assert payload["before"] == "print('old')"
    assert payload["after"] == "print('new')"
    assert payload["is_binary"] is False


def test_serialise_create_file_has_empty_before(tmp_path):
    b = PatchBlock(
        operation=OperationType.CREATE_FILE,
        file="src/new.py",
        content="def main():\n    pass\n",
    )
    [payload] = _serialise_blocks_for_approval([b], str(tmp_path))
    assert payload["operation"] == "create_file"
    assert payload["before"] == ""
    assert "def main" in payload["after"]


def test_serialise_delete_block_has_empty_after(tmp_path):
    b = PatchBlock(
        operation=OperationType.DELETE_BLOCK,
        file="src/dead.py",
        search="def dead_fn():\n    pass\n",
    )
    [payload] = _serialise_blocks_for_approval([b], str(tmp_path))
    assert payload["operation"] == "delete_block"
    assert payload["before"] == "def dead_fn():\n    pass\n"
    assert payload["after"] == ""


def test_serialise_truncates_large_payloads(tmp_path):
    huge = "x" * (300 * 1024)
    b = PatchBlock(
        operation=OperationType.CREATE_FILE, file="big.txt", content=huge,
    )
    [payload] = _serialise_blocks_for_approval([b], str(tmp_path))
    assert len(payload["after"]) < len(huge)
    assert "truncated" in payload["after"]
    # Original size preserved for context, even after truncation.
    assert payload["size_after"] == len(huge)


# ---------------------------------------------------------------------------
# Decision interpreter
# ---------------------------------------------------------------------------

def test_decision_approve_returns_all():
    blocks = [
        PatchBlock(operation=OperationType.CREATE_FILE, file="a.py"),
        PatchBlock(operation=OperationType.CREATE_FILE, file="b.py"),
    ]
    approved, rejected = _apply_approval_decision(blocks, "approve")
    assert approved == blocks
    assert rejected == []


def test_decision_reject_returns_nothing():
    blocks = [PatchBlock(operation=OperationType.CREATE_FILE, file="a.py")]
    approved, rejected = _apply_approval_decision(blocks, "reject")
    assert approved == []
    assert rejected == blocks


def test_decision_unknown_answer_defaults_to_approve():
    """Safe default: an unexpected answer must not silently reject
    patches — an operator whose network drops mid-answer should not
    also lose their work."""
    blocks = [PatchBlock(operation=OperationType.CREATE_FILE, file="a.py")]
    approved, rejected = _apply_approval_decision(blocks, "")
    assert approved == blocks
    assert rejected == []
    approved, rejected = _apply_approval_decision(blocks, "who knows")
    assert approved == blocks
    assert rejected == []


# ---------------------------------------------------------------------------
# End-to-end: apply_patch_blocks with require_approval=True
# ---------------------------------------------------------------------------

class _StubChannel:
    """Records the prompt call and returns whatever answer the test sets."""

    def __init__(self, answer: str):
        self.answer = answer
        self.calls: list[dict] = []

    def prompt(self, message, options, default=None, option_labels=None, metadata=None):
        self.calls.append({
            "message": message, "options": list(options), "default": default,
            "option_labels": dict(option_labels or {}),
            "metadata": dict(metadata or {}),
        })
        return self.answer

    def confirm(self, *args, **kwargs):
        return True

    def notes(self, *args, **kwargs):
        return ""

    def wait_for_manual_edit(self, *args, **kwargs):
        return None

    def is_interactive(self):
        return True


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_apply_with_approve_writes_the_file(tmp_path, monkeypatch):
    stub = _StubChannel("approve")
    monkeypatch.setattr(_hitl_mod, "get_channel", lambda: stub)
    block = PatchBlock(
        operation=OperationType.CREATE_FILE, file="hello.py",
        content="print('hi')\n",
    )
    results, modified = _run(apply_patch_blocks(
        [block], str(tmp_path), require_approval=True,
    ))
    assert (tmp_path / "hello.py").read_text().startswith("print('hi')")
    assert modified == ["hello.py"]
    # Approval prompt was actually made.
    assert stub.calls, "expected the HITL channel to be prompted"
    md = stub.calls[0]["metadata"]
    assert md["kind"] == "patch_approval"
    assert md["patches"][0]["path"] == "hello.py"


def test_apply_with_reject_skips_writes(tmp_path, monkeypatch):
    stub = _StubChannel("reject")
    monkeypatch.setattr(_hitl_mod, "get_channel", lambda: stub)
    block = PatchBlock(
        operation=OperationType.CREATE_FILE, file="hello.py",
        content="print('hi')\n",
    )
    results, modified = _run(apply_patch_blocks(
        [block], str(tmp_path), require_approval=True,
    ))
    assert not (tmp_path / "hello.py").exists()
    assert modified == []
    # A PatchResult marks the block as rejected so the caller (repair
    # loop, etc.) can see why nothing landed.
    assert any(
        not r.success and "rejected by operator" in (r.error or "")
        for r in results
    ), f"expected a rejected-by-operator PatchResult, got {results}"


def test_apply_without_approval_flag_is_byte_identical_to_pre_phase_5(tmp_path, monkeypatch):
    """require_approval defaults to False; existing callers see zero
    behaviour change. Prompt must NOT fire in that path."""
    stub = _StubChannel("reject")  # would reject if consulted
    monkeypatch.setattr(_hitl_mod, "get_channel", lambda: stub)
    block = PatchBlock(
        operation=OperationType.CREATE_FILE, file="baseline.py",
        content="OK\n",
    )
    results, modified = _run(apply_patch_blocks(
        [block], str(tmp_path),  # require_approval omitted
    ))
    assert (tmp_path / "baseline.py").read_text().startswith("OK")
    assert modified == ["baseline.py"]
    assert stub.calls == [], "approval channel must not fire when require_approval=False"


def test_apply_gate_channel_error_falls_through_to_approve(tmp_path, monkeypatch):
    """If the HITL channel raises (webhook down mid-run), fall
    through to writes instead of losing patches. Better to write than
    to silently discard the LLM's work."""
    class _BrokenChannel(_StubChannel):
        def prompt(self, *args, **kwargs):
            raise RuntimeError("webhook unreachable")
    monkeypatch.setattr(_hitl_mod, "get_channel", lambda: _BrokenChannel("reject"))
    block = PatchBlock(
        operation=OperationType.CREATE_FILE, file="fallback.py",
        content="OK\n",
    )
    results, modified = _run(apply_patch_blocks(
        [block], str(tmp_path), require_approval=True,
    ))
    assert (tmp_path / "fallback.py").exists()
