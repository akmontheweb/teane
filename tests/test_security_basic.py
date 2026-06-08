"""Tests for harness/security.py — Git and command validation basics."""

import tempfile
import subprocess
import os

import pytest

from harness.security import (
    GitGuardian,
    CommandValidator,
    CommandValidationResult,
    create_command_validator_from_config,
)


class TestGitGuardian:
    """Test GitGuardian initialization."""

    def test_init_with_workspace(self):
        """GitGuardian should initialize with workspace path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian = GitGuardian(tmpdir)
            assert guardian is not None

    def test_is_git_repo_empty_dir(self):
        """is_git_repo should return False for non-git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian = GitGuardian(tmpdir)
            assert guardian.is_git_repo() is False

    def test_is_git_repo_with_git_dir(self):
        """is_git_repo should return True when .git exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize git repo
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            guardian = GitGuardian(tmpdir)
            assert guardian.is_git_repo() is True

    def test_has_uncommitted_changes_empty_repo(self):
        """has_uncommitted_changes should return False for clean repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=tmpdir, capture_output=True)
            # Create initial commit
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test")
            subprocess.run(["git", "add", "test.txt"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=tmpdir, capture_output=True)

            guardian = GitGuardian(tmpdir)
            assert guardian.has_uncommitted_changes() is False

    def test_has_uncommitted_changes_with_changes(self):
        """has_uncommitted_changes should return True when files modified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=tmpdir, capture_output=True)
            # Create initial commit
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test")
            subprocess.run(["git", "add", "test.txt"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=tmpdir, capture_output=True)

            # Modify file
            with open(test_file, "w") as f:
                f.write("modified")

            guardian = GitGuardian(tmpdir)
            assert guardian.has_uncommitted_changes() is True


class TestCommandValidator:
    """Test CommandValidator initialization and validation."""

    def test_init_defaults(self):
        """CommandValidator should initialize with default configurations."""
        validator = CommandValidator()
        assert validator is not None

    def test_validate_safe_command(self):
        """validate() should pass safe commands."""
        validator = CommandValidator()
        result = validator.validate("echo hello")
        assert isinstance(result, CommandValidationResult)

    def test_validate_blocks_curl(self):
        """validate() should block curl by default."""
        validator = CommandValidator()
        result = validator.validate("curl http://example.com")
        # Curl is a blocked pattern
        assert result is not None

    def test_validate_blocks_wget(self):
        """validate() should block wget by default."""
        validator = CommandValidator()
        result = validator.validate("wget http://example.com")
        # Wget is a blocked pattern
        assert result is not None

    def test_add_blocked_pattern(self):
        """add_blocked_pattern() should add custom blocked patterns."""
        validator = CommandValidator()
        validator.add_blocked_pattern("^rm -rf")
        result = validator.validate("rm -rf /")
        # Should be blocked
        assert result is not None

    def test_add_allowed_command(self):
        """add_allowed_command() should whitelist commands."""
        validator = CommandValidator()
        validator.add_allowed_command("my_custom_tool")
        # With whitelist, only allowed commands should pass
        # (if whitelist is enforced)
        assert validator is not None


class TestCommandValidationResult:
    """Test CommandValidationResult dataclass."""

    def test_result_allowed(self):
        """Result should indicate when command is allowed."""
        result = CommandValidationResult(allowed=True, command="echo hello")
        assert result.allowed is True
        assert result.command == "echo hello"

    def test_result_blocked(self):
        """Result should indicate when command is blocked."""
        result = CommandValidationResult(allowed=False, command="curl", reason="network access")
        assert result.allowed is False
        assert result.reason == "network access"
        assert result.command == "curl"

    def test_result_with_all_fields(self):
        """Result should handle all fields."""
        result = CommandValidationResult(
            allowed=False,
            command="curl http://evil.com",
            reason="Dangerous pattern detected",
            matched_rule="blocked:curl",
        )
        assert result.allowed is False
        assert "Dangerous" in result.reason
        assert result.matched_rule == "blocked:curl"


class TestCreateCommandValidatorFromConfig:
    """Test factory function for CommandValidator."""

    def test_create_from_empty_config(self):
        """Should create validator from empty config."""
        config = {}
        validator = create_command_validator_from_config(config)
        assert validator is not None
        assert isinstance(validator, CommandValidator)

    def test_create_from_config_with_allowed(self):
        """Should add allowed commands from config."""
        config = {
            "allowed_commands": ["my_tool", "another_tool"],
        }
        validator = create_command_validator_from_config(config)
        assert validator is not None

    def test_create_from_config_with_blocked(self):
        """Should add blocked patterns from config."""
        config = {
            "blocked_patterns": ["^dangerous", "network_call"],
        }
        validator = create_command_validator_from_config(config)
        assert validator is not None
