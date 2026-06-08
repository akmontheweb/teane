"""Tests for harness/storage.py — checkpoint persistence basics."""

import pytest

from harness.storage import (
    CheckpointSummary,
    generate_session_id,
)


class TestCheckpointSummary:
    """Test CheckpointSummary dataclass."""

    def test_construct_minimal(self):
        """Construct CheckpointSummary with required fields."""
        summary = CheckpointSummary(thread_id="thread-1")
        assert summary.thread_id == "thread-1"
        assert summary.session_id == ""
        assert summary.current_node == ""

    def test_construct_with_all_fields(self):
        """Construct with all fields."""
        summary = CheckpointSummary(
            thread_id="t1",
            session_id="s1",
            current_node="patching",
            exit_code=0,
            budget_remaining_usd=5.5,
            total_cost_usd=4.5,
            modified_files=["a.py", "b.py"],
            loop_counters={"repair": 2},
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T01:00:00Z",
            is_active=True,
            workspace_path="/workspace",
        )
        assert summary.thread_id == "t1"
        assert summary.session_id == "s1"
        assert summary.current_node == "patching"
        assert summary.modified_files == ["a.py", "b.py"]
        assert summary.budget_remaining_usd == 5.5
        assert summary.is_active is True


class TestGenerateSessionId:
    """Test session ID generation."""

    def test_generate_uuid_when_none_provided(self):
        """Should generate a UUID when no session ID provided."""
        sid = generate_session_id(user_provided=None)
        assert sid is not None
        assert len(sid) > 0
        # Should be a valid UUID format (36 chars with dashes)
        assert "-" in sid or len(sid) > 8

    def test_use_user_provided_session_id(self):
        """Should use user-provided session ID when given."""
        provided = "my-custom-session"
        sid = generate_session_id(user_provided=provided)
        assert sid == provided

    def test_user_provided_empty_string_generates_uuid(self):
        """Empty string should be treated as no user input."""
        sid = generate_session_id(user_provided="")
        assert sid is not None
        assert len(sid) > 0
        # Should be UUID, not empty
        assert sid != ""
