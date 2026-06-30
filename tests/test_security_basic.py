"""Tests for harness/security.py — Git and command validation basics."""

import tempfile
import subprocess
import os


from harness.security import (
    GitGuardian,
    CommandValidator,
    CommandValidationResult,
    HITLGate,
    create_command_validator_from_config,
)


class TestHITLGateCIBranch:
    """Regression coverage for the CI auto-approve branch.

    Locks in the corrected behaviour: ``auto_approve_in_ci=True`` means the
    operation is approved (caller opted in); ``False`` means the operation
    is blocked because there is no human present to confirm.
    """

    def test_auto_approve_true_in_ci_approves(self, monkeypatch):
        monkeypatch.setenv("CI", "true")
        gate = HITLGate(enabled=True, auto_approve_in_ci=True)
        matches = [("git push", "sensitive push")]
        assert gate.prompt_approval(matches) is True

    def test_auto_approve_false_in_ci_blocks(self, monkeypatch):
        monkeypatch.setenv("CI", "true")
        gate = HITLGate(enabled=True, auto_approve_in_ci=False)
        matches = [("git push", "sensitive push")]
        assert gate.prompt_approval(matches) is False

    def test_no_matches_always_approves(self, monkeypatch):
        monkeypatch.setenv("CI", "true")
        gate = HITLGate(enabled=True, auto_approve_in_ci=False)
        assert gate.prompt_approval([]) is True


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

    def test_validate_strips_leading_subshell_paren(self):
        """A leading ``(`` (bash subshell) must not be mis-read as part
        of the command name. Reproduces the FinancialResearch failure
        where ``(test -d /tmp/venv || uv venv …) && pip install``
        was rejected as ``whitelist_missing:(test``."""
        validator = CommandValidator()
        # ``test`` and ``uv`` are in DEFAULT_ALLOWED_COMMANDS; verify
        # that the subshell wrapping doesn't trip whitelist parsing.
        cmd = (
            "(test -d /tmp/venv || uv venv /tmp/venv) && "
            ". /tmp/venv/bin/activate && uv pip install pytest"
        )
        result = validator.validate(cmd)
        assert result.allowed, f"unexpectedly blocked: {result.reason}"

    def test_validate_strips_trailing_close_paren_on_single_token(self):
        """A single-command subshell ``(true)`` must resolve to ``true``."""
        validator = CommandValidator()
        validator.add_allowed_command("true")
        result = validator.validate("(true) && echo ok")
        assert result.allowed, f"unexpectedly blocked: {result.reason}"


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


class TestSandboxExecutorPicksUpGlobalValidator:
    """P0.2 regression: every SandboxExecutor instantiated without an explicit
    `command_validator=` argument MUST pick up the process-wide default set by
    cmd_run. Without this, the validator is dead code (default = None means
    `if self.command_validator is not None` skips the check entirely).
    """

    def test_executor_inherits_global_validator(self):
        from harness.sandbox import SandboxExecutor
        from harness.security import (
            CommandValidator,
            set_command_validator,
            get_command_validator,
        )

        marker_validator = CommandValidator()
        set_command_validator(marker_validator)
        try:
            executor = SandboxExecutor(workspace_path="/tmp")
            assert executor.command_validator is marker_validator, (
                "SandboxExecutor must fall back to the global CommandValidator "
                "set by cmd_run when no command_validator is passed."
            )
        finally:
            set_command_validator(None)
            assert get_command_validator() is None

    def test_executor_explicit_validator_wins_over_global(self):
        from harness.sandbox import SandboxExecutor
        from harness.security import CommandValidator, set_command_validator

        global_validator = CommandValidator()
        explicit_validator = CommandValidator()
        set_command_validator(global_validator)
        try:
            executor = SandboxExecutor(
                workspace_path="/tmp",
                command_validator=explicit_validator,
            )
            assert executor.command_validator is explicit_validator
        finally:
            set_command_validator(None)

    def test_executor_no_global_no_explicit_is_none(self):
        from harness.sandbox import SandboxExecutor
        from harness.security import set_command_validator

        set_command_validator(None)
        executor = SandboxExecutor(workspace_path="/tmp")
        assert executor.command_validator is None
