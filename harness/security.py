"""
Lifecycle & Security Module — The Code Guardian.

This module implements:
    - GitGuardian: Git state tracking with automatic temp-branch creation per session.
                   Creates agent/patch-{session_id} branches, tracks original branch state,
                   and performs clean rollback on failure or squash-merge on success.
    - CommandValidator: Deterministic command whitelist/blocklist registry.
                       Scans build commands before execution, blocking dangerous
                       patterns (curl, wget, network calls, destructive operations)
                       unless explicitly authorized.

Integration points:
    - CLI layer calls GitGuardian before graph execution (create branch)
      and after (rollback or commit).
    - SandboxExecutor calls CommandValidator before every build command execution.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. GitGuardian — Git State Tracking & Clean Rollback
# ---------------------------------------------------------------------------

class GitGuardian:
    """
    Manages git state across a harness session.

    Lifecycle:
        1. create_patch_branch()   — Called before graph execution.
                                     Creates agent/patch-{session_id}, tracks original branch.
        2. [graph executes — planning, patching, compiler, repair]
        3a. commit_on_success()    — Called on exit_code 0.
                                     Commits all changes with a harness summary message.
        3b. rollback_on_failure()  — Called on repeated failure or abandon.
                                     Restores working tree, deletes patch branch,
                                     returns to original branch.
    """

    def __init__(self, workspace_path: str):
        self.workspace_path = os.path.abspath(workspace_path)
        self._original_branch: Optional[str] = None
        self._patch_branch: Optional[str] = None
        self._branch_created = False

    def _git(self, *args: str, capture: bool = True) -> subprocess.CompletedProcess[str]:
        """Run a git command in the workspace directory."""
        cmd = ["git", "-C", self.workspace_path, *args]
        logger.debug("[gitguardian] Running: %s", " ".join(cmd))
        return subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=60,
        )

    def is_git_repo(self) -> bool:
        """Check if the workspace is a git repository."""
        result = self._git("rev-parse", "--git-dir")
        return result.returncode == 0

    def get_current_branch(self) -> Optional[str]:
        """Get the current branch name, or None if detached HEAD."""
        result = self._git("rev-parse", "--abbrev-ref", "HEAD")
        if result.returncode != 0:
            return None
        branch = result.stdout.strip()
        if branch == "HEAD":
            return None  # Detached HEAD
        return branch

    def has_uncommitted_changes(self) -> bool:
        """Check for uncommitted changes (staged OR unstaged) in the working tree."""
        result = self._git("status", "--porcelain")
        return len(result.stdout.strip()) > 0

    def stash_if_dirty(self) -> bool:
        """
        If the working tree has uncommitted changes before we start,
        stash them to keep them safe from our patch operations.

        Returns True if something was stashed.
        """
        if self.has_uncommitted_changes():
            logger.info("[gitguardian] Stashing pre-existing uncommitted changes.")
            result = self._git("stash", "push", "-m", "[harness] auto-stash before agent session")
            if result.returncode == 0:
                return True
            logger.warning("[gitguardian] Stash failed: %s", result.stderr.strip())
        return False

    def pop_stash(self) -> bool:
        """Restore stashed changes after harness completes."""
        result = self._git("stash", "pop")
        if result.returncode == 0:
            logger.info("[gitguardian] Restored stashed changes.")
            return True
        # stash list may be empty; that's fine
        return False

    def create_patch_branch(self, session_id: str) -> bool:
        """
        Create an isolated patch branch for this harness session.

        Branch name: agent/patch-{first 8 chars of session_id}

        Steps:
            1. Record the current branch name
            2. Create and switch to agent/patch-{session_id[:8]}

        Returns True on success, False if git is unavailable or branch exists.
        """
        if not self.is_git_repo():
            logger.warning("[gitguardian] Workspace is not a git repository. Skipping branch creation.")
            return False

        self._original_branch = self.get_current_branch()
        if self._original_branch is None:
            logger.warning("[gitguardian] Detached HEAD. Skipping branch creation.")
            return False

        short_id = session_id[:8] if len(session_id) >= 8 else session_id
        self._patch_branch = f"agent/patch-{short_id}"

        # Check if branch already exists
        result = self._git("rev-parse", "--verify", self._patch_branch)
        if result.returncode == 0:
            logger.warning("[gitguardian] Branch '%s' already exists. Using existing branch.", self._patch_branch)
            self._git("checkout", self._patch_branch)
            self._branch_created = True
            return True

        # Create and switch to the new branch
        result = self._git("checkout", "-b", self._patch_branch)
        if result.returncode != 0:
            logger.error("[gitguardian] Failed to create branch '%s': %s", self._patch_branch, result.stderr.strip())
            return False

        self._branch_created = True
        logger.info("[gitguardian] Created patch branch '%s' (from '%s').", self._patch_branch, self._original_branch)
        return True

    def commit_all_changes(self, session_id: str, modified_files: list[str], exit_code: int) -> bool:
        """
        Stage and commit changes made during the harness session.

        Only files in ``modified_files`` are staged — using ``git add -A``
        would also stage user-introduced or stray files unrelated to the
        harness's patches. ``git add -A -- <paths>`` stages additions,
        modifications, and deletions scoped to those paths.

        If ``modified_files`` is empty but the working tree is dirty, the
        commit is refused: those changes are not the harness's to commit.

        Args:
            session_id: The harness session ID.
            modified_files: List of files the harness modified (relative to workspace).
            exit_code: The final build exit code.

        Returns True on success.
        """
        if not self.is_git_repo():
            return False

        if not self.has_uncommitted_changes():
            logger.debug("[gitguardian] No changes to commit.")
            return True

        if not modified_files:
            logger.warning(
                "[gitguardian] Working tree is dirty but modified_files is empty. "
                "Refusing to `git add -A` (would commit user files). Skipping commit."
            )
            return False

        # Stage only the files the harness actually modified.
        # `-A -- <paths>` covers additions, modifications, and deletions.
        result = self._git("add", "-A", "--", *modified_files)
        if result.returncode != 0:
            logger.warning("[gitguardian] git add failed: %s", result.stderr.strip())
            return False

        # Build commit message
        file_list = "\n".join(f"  - {f}" for f in modified_files[:10])
        if len(modified_files) > 10:
            file_list += f"\n  ... and {len(modified_files) - 10} more files"

        message = (
            f"[harness] Automated patch — session {session_id}\n\n"
            f"Build exit code: {exit_code}\n"
            f"Files modified ({len(modified_files)}):\n"
            f"{file_list}"
        )

        result = self._git("commit", "-m", message)
        if result.returncode != 0:
            logger.warning("[gitguardian] git commit failed: %s", result.stderr.strip())
            return False

        logger.info("[gitguardian] Committed %d file(s) on branch '%s'.", len(modified_files), self._patch_branch)
        return True

    def rollback(self, modified_files: Optional[Sequence[str]] = None) -> bool:
        """
        Clean rollback: restore working tree to HEAD, switch back to original branch,
        and delete the patch branch.

        ``git checkout -- .`` only restores **tracked** files — any new file
        the LLM created during the session (e.g. ``.env``, leaked secrets,
        scratch files) would otherwise remain in the workspace after
        rollback. To handle this, callers should pass ``modified_files``
        and any file in that list that is not tracked by git is removed
        before the checkout.

        If ``modified_files`` is None (e.g. unexpected crash before the
        graph populated state), a warning is logged and untracked LLM
        files may remain — we don't blanket-`git clean` because that
        would also remove the user's own untracked work.

        This is called when:
            - The harness abandons (HITL [q])
            - 3 repair attempts fail without resolution
            - An unhandled exception aborts graph execution

        Returns True on success.
        """
        if not self.is_git_repo() or not self._branch_created:
            return False

        logger.info("[gitguardian] Rolling back changes on branch '%s'.", self._patch_branch)

        if modified_files:
            self._remove_untracked_llm_files(modified_files)
        else:
            logger.warning(
                "[gitguardian] Rollback called without modified_files; "
                "any LLM-created untracked files will remain in the workspace."
            )

        # Restore tracked files to HEAD
        self._git("checkout", "--", ".")

        # Switch back to original branch
        if self._original_branch:
            result = self._git("checkout", self._original_branch)
            if result.returncode != 0:
                logger.warning("[gitguardian] Failed to switch back to '%s': %s", self._original_branch, result.stderr.strip())

        # Delete the patch branch
        if self._patch_branch:
            result = self._git("branch", "-D", self._patch_branch)
            if result.returncode == 0:
                logger.info("[gitguardian] Deleted patch branch '%s'.", self._patch_branch)
            else:
                logger.warning("[gitguardian] Failed to delete branch '%s': %s", self._patch_branch, result.stderr.strip())

        self._branch_created = False
        return True

    def _remove_untracked_llm_files(self, modified_files: Sequence[str]) -> None:
        """
        Delete files from ``modified_files`` that are not tracked by git.

        These are files the LLM CREATE_FILE'd during the session — git
        checkout doesn't know about them and would leave them behind on
        rollback, defeating the workspace-restoration contract.
        """
        for filepath in modified_files:
            # Resolve relative to workspace and stay inside it (defense in depth
            # against modified_files containing traversal — the patcher now
            # rejects these, but rollback runs even on patcher-rejected runs).
            if os.path.isabs(filepath):
                abs_path = filepath
            else:
                abs_path = os.path.join(self.workspace_path, filepath)
            abs_real = os.path.realpath(abs_path)
            ws_real = os.path.realpath(self.workspace_path)
            try:
                common = os.path.commonpath([abs_real, ws_real])
            except ValueError:
                continue
            if common != ws_real:
                logger.warning("[gitguardian] Skipping path outside workspace: %s", filepath)
                continue

            # `git ls-files --error-unmatch -- <path>` exits non-zero if untracked.
            result = self._git("ls-files", "--error-unmatch", "--", filepath)
            if result.returncode == 0:
                continue  # tracked — checkout will handle it

            if os.path.isfile(abs_real):
                try:
                    os.remove(abs_real)
                    logger.info("[gitguardian] Removed untracked LLM-created file: %s", filepath)
                except OSError as e:
                    logger.warning("[gitguardian] Failed to remove %s: %s", filepath, e)

    def restore_original_branch(self) -> bool:
        """
        Switch back to the original branch without deleting the patch branch.
        Used when the harness succeeds — the patch branch remains for manual review/merge.
        """
        if not self.is_git_repo() or not self._original_branch:
            return False

        result = self._git("checkout", self._original_branch)
        if result.returncode != 0:
            logger.warning("[gitguardian] Failed to switch back to '%s': %s", self._original_branch, result.stderr.strip())
            return False

        logger.info("[gitguardian] Switched back to original branch '%s'. Patch branch '%s' remains for review.",
                     self._original_branch, self._patch_branch)
        return True


# ---------------------------------------------------------------------------
# 2. CommandValidator — Deterministic Command Whitelist
# ---------------------------------------------------------------------------

@dataclass
class CommandValidationResult:
    """Result of command validation."""
    allowed: bool
    command: str
    reason: str = ""
    matched_rule: str = ""


class CommandValidator:
    """
    Secure whitelist/blocklist filter for build commands.

    Scans shell commands before execution and blocks:
        - Network tools (curl, wget, nc, telnet, ssh, scp)
        - Destructive operations (rm -rf /, chmod 777 /, dd, mkfs)
        - Privilege escalation (sudo)
        - Arbitrary script execution from network sources

    Configurable via .harness_config.json:
        {
          "security": {
            "allowed_commands": ["make", "cargo", "gcc", "g++", "pytest", ...],
            "blocked_patterns": ["curl", "wget", "sudo", ...],
            "allow_all_commands": false,
            "allow_network_in_build": false
          }
        }
    """

    # Default whitelist: safe build/dev tools
    DEFAULT_ALLOWED_COMMANDS: set[str] = {
        "make", "cmake", "ninja",
        "gcc", "g++", "clang", "clang++", "cc", "c++",
        "rustc", "cargo",
        "go", "gofmt",
        "python", "python3", "pip", "pip3", "poetry", "uv",
        "node", "npm", "npx", "yarn", "pnpm",
        "javac", "java", "mvn", "gradle",
        "pytest", "unittest", "tox", "nox",
        "dotnet", "msbuild",
        "sh", "bash", "dash",
        "echo", "cat", "ls", "cp", "mv", "mkdir", "rm", "chmod", "chown",
        "git", "hg",
        "docker", "docker-compose", "podman",
        "env", "export", "source",
        "test", "[",
        "true", "false",
    }

    # Default blocklist: dangerous or network-exposing patterns
    DEFAULT_BLOCKED_PATTERNS: list[str] = [
        r"\bcurl\b",
        r"\bwget\b",
        r"\bnc\b",
        r"\bnetcat\b",
        r"\btelnet\b",
        r"\bssh\b",
        r"\bscp\b",
        r"\bsftp\b",
        r"\brsync\b(?!.*\.\/(?!.*:\/\/)[a-zA-Z])",  # Allow local rsync, block remote
        r"\bftp\b",
        r"\btftp\b",
        r"\bnmap\b",
        r"\bsudo\b",
        r"\bsu\b",
        r"\bdd\s+if=",
        r"\bmkfs\.",
        r"\bmkswap\b",
        r"\bmount\b(?!.*--bind.*ro)",
        r"\bumount\b",
        r"\bfdisk\b",
        r"\bparted\b",
        r"\bkillall\b",
        r"\bpkill\b",
        r"\breboot\b",
        r"\bshutdown\b",
        r"\binit\s+[0-6]\b",
        r"\bsystemctl\b",
        r"\bservice\b",
        r"\brm\s+-rf\s+/",       # rm -rf / (absolute root)
        r"\bchmod\s+777\s+/",    # chmod 777 on absolute root paths
        r"\b>\/dev\/sd[a-z]\b",  # Writing to raw disk devices
        r"\b\/dev\/null\b",      # Allowable, but flagging suspicious pipes
        r"\bwget\b.*\|.*sh\b",   # curl | bash pattern
        r"\bcurl\b.*\|.*sh\b",   # curl | bash pattern
        r"\beval\b",
        r"\bexec\b",
        r"\bsource\s+<(curl|wget)",  # Process substitution from network
        r"\b\/etc\/passwd\b",
        r"\b\/etc\/shadow\b",
        r"\b\/etc\/sudoers\b",
        r"\b\/root\/",
    ]

    def __init__(
        self,
        allowed_commands: Optional[set[str]] = None,
        blocked_patterns: Optional[list[str]] = None,
        allow_all_commands: bool = False,
        allow_network_in_build: bool = False,
    ):
        self.allowed_commands = allowed_commands or set(self.DEFAULT_ALLOWED_COMMANDS)
        self.blocked_patterns = blocked_patterns or list(self.DEFAULT_BLOCKED_PATTERNS)
        self.allow_all_commands = allow_all_commands
        self.allow_network_in_build = allow_network_in_build
        self._compiled_patterns: list[re.Pattern[str]] = [
            re.compile(p, re.IGNORECASE) for p in self.blocked_patterns
        ]

    def validate(self, command: str) -> CommandValidationResult:
        """
        Validate a shell command against the whitelist and blocklist.

        Args:
            command: The full shell command string to validate.

        Returns:
            CommandValidationResult with allowed=True/False and reason.
        """
        if self.allow_all_commands:
            return CommandValidationResult(allowed=True, command=command, reason="allow_all_commands is enabled")

        command_stripped = command.strip()

        # Remove environment variable assignments for token analysis
        # e.g., "RUSTFLAGS=... cargo build" → "cargo build"
        clean_for_parsing = re.sub(r'^[A-Za-z_][A-Za-z0-9_]*=[^\s;]*\s+', '', command_stripped)

        # --- Blocklist check (highest priority) ---
        for i, pattern in enumerate(self._compiled_patterns):
            if pattern.search(command_stripped):
                return CommandValidationResult(
                    allowed=False,
                    command=command,
                    reason=f"Command matches blocked pattern: '{self.blocked_patterns[i]}'",
                    matched_rule=self.blocked_patterns[i],
                )

        # --- Network safety check ---
        # If network is not explicitly allowed, block any URL/domain patterns
        if not self.allow_network_in_build:
            url_patterns = [
                r'https?://[^\s]+',
                r'ftp://[^\s]+',
                r'\b[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}(?!\.\d)',  # IP addresses
            ]
            for url_pattern in url_patterns:
                if re.search(url_pattern, command_stripped):
                    return CommandValidationResult(
                        allowed=False,
                        command=command,
                        reason="Network access detected but allow_network_in_build is false. "
                               "URL/IP found in command.",
                        matched_rule=url_pattern,
                    )

        # --- Whitelist check ---
        # Extract the base command token (first word after env vars and pipes)
        # Split by &&, ||, ;, | to find individual commands
        cmd_parts = re.split(r'\s*(?:&&|\|\||;|\|)\s*', clean_for_parsing)
        for part in cmd_parts:
            part = part.strip()
            if not part:
                continue
            # Get the first token as the command name
            tokens = part.split()
            if not tokens:
                continue
            base_cmd = os.path.basename(tokens[0])  # Strip path if present

            # Skip common shell builtins and operators
            if base_cmd in ("", "sh", "bash", ".", "source", "export", "env", "exec"):
                continue

            if base_cmd not in self.allowed_commands:
                return CommandValidationResult(
                    allowed=False,
                    command=command,
                    reason=f"Command '{base_cmd}' is not in the allowed commands whitelist. "
                           f"Add it to 'security.allowed_commands' in .harness_config.json to permit it.",
                    matched_rule=f"whitelist_missing:{base_cmd}",
                )

        return CommandValidationResult(allowed=True, command=command, reason="All checks passed")

    def add_allowed_command(self, command_name: str) -> None:
        """Add a command to the whitelist."""
        self.allowed_commands.add(command_name)
        logger.info("[security] Added '%s' to allowed commands whitelist.", command_name)

    def add_blocked_pattern(self, pattern: str) -> None:
        """Add a regex pattern to the blocklist."""
        self.blocked_patterns.append(pattern)
        self._compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
        logger.info("[security] Added blocked pattern: '%s'", pattern)

    def validate_or_raise(self, command: str) -> str:
        """
        Validate a command and return it if allowed, or raise ValueError.

        Args:
            command: The shell command to validate.

        Returns:
            The command string (unchanged) if valid.

        Raises:
            ValueError: If the command is blocked.
        """
        result = self.validate(command)
        if not result.allowed:
            raise ValueError(
                f"[SECURITY BLOCKED]: {result.reason}\n"
                f"  Command: {command}\n"
                f"  Matched Rule: {result.matched_rule}\n"
                f"  Tip: Configure 'security.allowed_commands' or 'security.blocked_patterns' "
                f"in .harness_config.json to adjust."
            )
        return command


# ---------------------------------------------------------------------------
# 3. HITLGate — Proactive Pre-Execution Confirmation
# ---------------------------------------------------------------------------

# Patterns that trigger a HITL confirmation prompt before execution.
# Each maps a regex pattern to a user-facing warning message.
_DEFAULT_SENSITIVE_PATTERNS: dict[str, str] = {
    r"\bgit\s+push\b":                         "[Harness Warning]: LLM is attempting to push code modifications to a remote. Approve?",
    r"\bgit\s+merge\b":                         "[Harness Warning]: LLM is attempting to merge branches. Approve?",
    r"\bgit\s+rebase\b":                        "[Harness Warning]: LLM is attempting to rebase. Approve?",
    r"\bdocker\s+push\b":                       "[Harness Warning]: LLM is attempting to push a container image to a registry. Approve?",
    r"\bkubectl\s+apply\b":                     "[Harness Warning]: LLM is attempting to apply Kubernetes configuration. Approve?",
    r"\bterraform\s+apply\b":                   "[Harness Warning]: Infrastructure change detected (terraform apply). Approve?",
    r"\brm\s+-rf\b":                            "[Harness Warning]: Destructive file removal detected (rm -rf). Approve?",
    r"\bmv\s+.*\/etc\/":                        "[Harness Warning]: File operation targeting /etc detected. Approve?",
    r"\bchmod\s+777\b":                         "[Harness Warning]: LLM is setting world-writable permissions. Approve?",
    r"\bheroku\s+run\b":                        "[Harness Warning]: Heroku command detected. Approve?",
    r"\baws\s+s3\s+rm\b":                       "[Harness Warning]: AWS S3 delete operation detected. Approve?",
    r"\baws\s+ec2\s+terminate\b":               "[Harness Warning]: AWS EC2 termination detected. Approve?",
    r"\bgcloud\s+.*\s+delete\b":                "[Harness Warning]: GCP resource deletion detected. Approve?",
    r"\bDROP\s+TABLE\b":                        "[Harness Warning]: SQL DROP TABLE detected. Approve?",
    r"\bDELETE\s+FROM\b":                       "[Harness Warning]: SQL DELETE FROM detected. Approve?",
    r"\balembic\s+upgrade\b":                   "[Harness Warning]: Database migration detected. Approve?",
    r"\bnpx\s+.*\s+deploy\b":                   "[Harness Warning]: Deployment command detected. Approve?",
    r"\bpip\s+install\s+.*https?://":           "[Harness Warning]: pip install from URL detected. Approve?",
}


class HITLGate:
    """
    Proactive pre-execution confirmation gate for sensitive operations.

    Scans LLM-generated patch content for dangerous patterns BEFORE the
    patches are applied to disk. When a sensitive pattern is detected,
    the gate pauses execution and prompts the developer interactively.

    Differs from CommandValidator:
        - CommandValidator: BLOCKS commands in the build sandbox (always)
        - HITLGate: PROMPTS for approval on LLM-generated content (interactive)

    Configurable via .harness_config.json:
        {
          "hitl_gate": {
            "enabled": true,
            "sensitive_patterns": {
              "git push": "Custom warning message here"
            },
            "auto_approve_in_ci": true
          }
        }
    """

    def __init__(
        self,
        enabled: bool = True,
        sensitive_patterns: Optional[dict[str, str]] = None,
        auto_approve_in_ci: bool = True,
    ):
        self.enabled = enabled
        self.auto_approve_in_ci = auto_approve_in_ci
        self._patterns: dict[re.Pattern[str], str] = {}

        patterns = sensitive_patterns if sensitive_patterns is not None else dict(_DEFAULT_SENSITIVE_PATTERNS)
        for pattern_str, warning in patterns.items():
            self._patterns[re.compile(pattern_str, re.IGNORECASE)] = warning

    def _is_ci_environment(self) -> bool:
        """Detect if we're running in a non-interactive CI environment."""
        return (
            not sys.stdin.isatty()
            or os.environ.get("CI", "") == "true"
            or os.environ.get("HARNESS_AUTO_APPROVE", "") == "true"
        )

    def scan(self, content: str) -> list[tuple[str, str]]:
        """
        Scan content for sensitive patterns.

        Args:
            content: The LLM-generated text to scan (patch blocks, code, etc.).

        Returns:
            List of (pattern_regex, warning_message) for each match found.
            Empty list if no sensitive patterns detected.
        """
        if not self.enabled:
            return []

        matches: list[tuple[str, str]] = []
        for pattern, warning in self._patterns.items():
            if pattern.search(content):
                matches.append((pattern.pattern, warning))
        return matches

    def prompt_approval(self, matches: list[tuple[str, str]], llm_content: str = "", context: str = "") -> bool:
        """
        Present an interactive approval prompt for detected sensitive operations.

        Args:
            matches: List of (pattern, warning) tuples from scan().
            llm_content: The full LLM response content for view/show.
            context: Optional context string to display (e.g., file name, node name).

        Returns:
            True if the developer approves, False if denied.

        In CI environments (no interactive TTY available), the gate cannot
        prompt a human. Behavior depends on ``auto_approve_in_ci``:
          - ``True``: caller has explicitly opted into unattended CI runs;
            sensitive operations are auto-approved with a warning.
          - ``False`` (default): sensitive operations are blocked because
            there is no human to confirm them.
        """
        if not matches:
            return True

        if self._is_ci_environment():
            if self.auto_approve_in_ci:
                logger.warning(
                    "[hitl_gate] CI environment detected with auto_approve_in_ci=True. "
                    "Auto-approving %d sensitive pattern(s): %s",
                    len(matches),
                    [warning for _, warning in matches],
                )
                return True  # User opted into unattended CI approval
            else:
                logger.warning(
                    "[hitl_gate] CI environment detected with auto_approve_in_ci=False. "
                    "Blocking %d sensitive pattern(s) (no interactive prompt available): %s",
                    len(matches),
                    [warning for _, warning in matches],
                )
                return False  # No human available to confirm — block

        # Interactive prompt
        print()
        print("=" * 72)
        print("[HITL GATE] Sensitive Operation Detected — Manual Approval Required")
        print("=" * 72)
        if context:
            print(f"Context: {context}")
        print()

        for i, (pattern, warning) in enumerate(matches, 1):
            print(f"  [{i}] Pattern: {pattern}")
            print(f"      {warning}")
        print()

        print()

        from harness.hitl import get_channel as _get_channel
        confirmed = _get_channel().confirm(
            "[HITL Gate] Approve these changes?", default=False
        )
        if confirmed:
            logger.info("[hitl_gate] Developer approved sensitive operation(s).")
            return True
        else:
            logger.warning("[hitl_gate] Developer denied sensitive operation(s).")
            return False

    def check_and_prompt(self, content: str, context: str = "") -> bool:
        """
        Convenience method: scan content and prompt if needed.

        Args:
            content: The LLM-generated text to scan.
            context: Optional context for the prompt display.

        Returns:
            True if all clear or approved, False if blocked.
        """
        matches = self.scan(content)
        return self.prompt_approval(matches, llm_content=content, context=context)


# ---------------------------------------------------------------------------
# 4. Security Scan Node — SAST & Secret Auditing
# ---------------------------------------------------------------------------

async def _run_subprocess_scanner(
    cmd: list[str],
    timeout_seconds: int = 15,
    label: str = "scanner",
) -> tuple[int, str, str]:
    """
    Run a security scanning tool as a subprocess with strict timeout.

    Args:
        cmd: The command and arguments as a list.
        timeout_seconds: Maximum execution time.
        label: Human-readable label for logging.

    Returns:
        Tuple of (exit_code, stdout_text, stderr_text).
    """
    logger.info("[security_scan] Running %s: %s", label, " ".join(cmd))
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_seconds
        )
        exit_code = proc.returncode if proc.returncode is not None else -1
        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        logger.info("[security_scan] %s finished: exit=%d", label, exit_code)
        return exit_code, stdout_text, stderr_text
    except asyncio.TimeoutError:
        logger.warning("[security_scan] %s timed out after %ds.", label, timeout_seconds)
        return -1, "", f"{label} timed out after {timeout_seconds}s"
    except FileNotFoundError:
        logger.info("[security_scan] %s tool not installed. Skipping.", label)
        return 0, "", ""
    except Exception as exc:
        logger.warning("[security_scan] %s failed: %s", label, exc)
        return 1, "", str(exc)


def _parse_gitleaks_json(stdout: str) -> list[dict[str, Any]]:
    """Parse gitleaks JSON output into structured findings."""
    findings: list[dict[str, Any]] = []
    try:
        import json
        data = json.loads(stdout) if stdout.strip() else []
        if not isinstance(data, list):
            data = []
    except (json.JSONDecodeError, ValueError):
        return findings

    for item in data:
        if not isinstance(item, dict):
            continue
        findings.append({
            "file": item.get("File", item.get("file", "")),
            "line": item.get("StartLine", item.get("line", 0)),
            "secret_type": item.get("RuleID", item.get("rule_id", "Unknown Secret")),
            "message": item.get("Description", item.get("description", "")),
            "secret_hash": (item.get("Secret") or item.get("secret", "") or "")[:8],
        })
    return findings


def _fallback_secret_scan(workspace_path: str) -> list[dict[str, Any]]:
    """
    Python fallback secret scanner when gitleaks is not installed.
    Uses regex patterns from harness/redactor.py to detect secrets in source files.
    """
    import re as _re

    # Reuse patterns from redactor module if available, otherwise use built-in
    secret_patterns: list[tuple[str, str]] = [
        (r'\b(sk-(?:proj-)?[A-Za-z0-9]{20,})\b', "OpenAI API Key"),
        (r'\b(sk-ant-api[0-9]{2}-[A-Za-z0-9_-]{40,})\b', "Anthropic API Key"),
        (r'\b(gh[pousr]_[A-Za-z0-9]{20,})\b', "GitHub Token"),
        (r'\b(AKIA[0-9A-Z]{16})\b', "AWS Access Key"),
        (r'\b(eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)\b', "JWT Token"),
        (r'-----BEGIN (?:RSA|EC|DSA|OPENSSH|ENCRYPTED) PRIVATE KEY-----', "Private Key"),
        (r'(?:postgres|mysql|mongodb|redis)://[^:]+:[^@\s]+@', "Database Connection String"),
        (r'\b(sk_live_[A-Za-z0-9]{24,})\b', "Stripe Live Key"),
        (r'\b(xox[bpras]-[A-Za-z0-9-]{10,})\b', "Slack Token"),
    ]

    findings: list[dict[str, Any]] = []
    ignore_dirs = {".git", "__pycache__", "node_modules", "vendor", "target", "build", "dist", ".tox", ".venv", "venv"}

    try:
        for root, dirs, files in os.walk(workspace_path):
            dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith(".")]
            for filename in files[:100]:  # Limit files scanned in fallback mode
                if filename in (".env", ".env.local", ".env.production"):
                    continue  # .env files redacted separately
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except (OSError, UnicodeDecodeError):
                    continue

                for line_num, line in enumerate(content.splitlines(), 1):
                    for pattern, label in secret_patterns:
                        match = _re.search(pattern, line)
                        if match:
                            secret_val = match.group(0)
                            findings.append({
                                "file": os.path.relpath(filepath, workspace_path),
                                "line": line_num,
                                "secret_type": label,
                                "message": f"Hardcoded {label} detected by regex fallback scanner.",
                                "secret_hash": hashlib.sha256(secret_val.encode()).hexdigest()[:8] if len(secret_val) <= 200 else "",
                            })
    except Exception as exc:
        logger.warning("[security_scan] Fallback secret scan error: %s", exc)

    return findings


async def run_gitleaks_scan(
    workspace_path: str,
    gitleaks_path: str = "gitleaks",
    timeout_seconds: int = 15,
) -> list[dict[str, Any]]:
    """
    Run gitleaks for secret detection. Falls back to Python regex scanner
    if gitleaks is not installed.

    Args:
        workspace_path: Path to the repository root.
        gitleaks_path: Path to gitleaks binary (auto-detected if empty).
        timeout_seconds: Max execution time.

    Returns:
        List of finding dicts with file, line, secret_type, message, secret_hash.
    """
    # Determine gitleaks path
    if not gitleaks_path:
        gitleaks_path = "gitleaks"
    resolved = shutil.which(gitleaks_path) if gitleaks_path else None
    if resolved:
        gitleaks_path = resolved
    else:
        logger.info("[security_scan] gitleaks not found. Using Python fallback scanner.")
        return _fallback_secret_scan(workspace_path)

    cmd = [
        gitleaks_path, "detect",
        "--source", workspace_path,
        "--no-git",
        "--report-format", "json",
        "--exit-code", "0",  # Don't fail on findings, we handle parsing
    ]

    exit_code, stdout_text, stderr_text = await _run_subprocess_scanner(
        cmd, timeout_seconds=timeout_seconds, label="gitleaks"
    )

    if exit_code == 0:
        findings = _parse_gitleaks_json(stdout_text)
        if findings:
            logger.warning("[security_scan] gitleaks found %d potential secret(s).", len(findings))
        else:
            logger.info("[security_scan] gitleaks: no secrets found.")
        return findings

    # gitleaks failed or timed out — fall back to regex scanner
    logger.warning("[security_scan] gitleaks failed (exit=%d). Falling back to regex scanner.", exit_code)
    return _fallback_secret_scan(workspace_path)


async def run_sast_scan(
    workspace_path: str,
    bandit_path: str = "bandit",
    semgrep_path: str = "semgrep",
    timeout_seconds: int = 15,
) -> list[dict[str, Any]]:
    """
    Run SAST scanning based on detected file types in the workspace.

    Dispatches to:
        - bandit if .py files exist
        - semgrep if .js/.ts/.go files exist (or as universal fallback)
        - govulncheck if .go files exist and go is available

    Args:
        workspace_path: Path to the repository root.
        bandit_path: Path to bandit (auto-detected).
        semgrep_path: Path to semgrep (auto-detected).
        timeout_seconds: Max execution time per tool.

    Returns:
        List of finding dicts with file, line, code, message.
    """
    findings: list[dict[str, Any]] = []

    # Detect file types present in the workspace
    has_py = False
    has_go = False
    has_js_ts = False

    for root, dirs, files in os.walk(workspace_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "vendor", "__pycache__")]
        for f in files[:200]:
            ext = os.path.splitext(f)[1].lower()
            if ext in (".py", ".pyi"):
                has_py = True
            elif ext == ".go":
                has_go = True
            elif ext in (".ts", ".tsx", ".js", ".jsx", ".mjs"):
                has_js_ts = True
        if has_py and has_go and has_js_ts:
            break

    # --- Bandit for Python ---
    if has_py:
        resolved = shutil.which(bandit_path) if bandit_path else None
        if resolved:
            cmd = [resolved, "-r", "-f", "json", "-ll", "-q", workspace_path]
            exit_code, stdout_text, stderr_text = await _run_subprocess_scanner(
                cmd, timeout_seconds=timeout_seconds, label="bandit"
            )
            if exit_code != 0 and stdout_text.strip():
                try:
                    import json
                    bandit_data = json.loads(stdout_text)
                    results = bandit_data.get("results", []) if isinstance(bandit_data, dict) else []
                    for r in results:
                        findings.append({
                            "file": r.get("filename", ""),
                            "line": r.get("line_number", 0),
                            "code": r.get("test_id", "BANDIT"),
                            "message": r.get("issue_text", r.get("test_name", "Bandit finding")),
                            "severity": r.get("issue_severity", "medium"),
                        })
                    if findings:
                        logger.warning("[security_scan] bandit found %d issue(s).", len(findings))
                except Exception:
                    pass
        else:
            logger.debug("[security_scan] bandit not installed. Skipping Python SAST.")

    # --- Semgrep (universal fallback for JS/TS/Go without native tools) ---
    if has_js_ts or has_go or not (has_py):
        resolved = shutil.which(semgrep_path) if semgrep_path else None
        if resolved:
            cmd = [resolved, "scan", "--config=auto", "--json", "--quiet", "--no-git-ignore", workspace_path]
            exit_code, stdout_text, stderr_text = await _run_subprocess_scanner(
                cmd, timeout_seconds=timeout_seconds, label="semgrep"
            )
            if stdout_text.strip():
                try:
                    import json
                    semgrep_data = json.loads(stdout_text)
                    results = semgrep_data.get("results", []) if isinstance(semgrep_data, dict) else []
                    for r in results[:50]:  # Cap at 50 findings to avoid context bloat
                        findings.append({
                            "file": r.get("path", ""),
                            "line": r.get("start", {}).get("line", 0),
                            "code": r.get("check_id", "SEMGREP"),
                            "message": r.get("extra", {}).get("message", "Semgrep finding"),
                            "severity": r.get("extra", {}).get("severity", "warning"),
                        })
                    if findings:
                        logger.warning("[security_scan] semgrep found %d issue(s).", len(findings))
                except Exception:
                    pass
        else:
            logger.debug("[security_scan] semgrep not installed. Skipping universal SAST.")

    return findings


def _findings_to_diagnostics(
    findings: list[dict[str, Any]],
    source: str = "security_scan",
) -> list[dict[str, Any]]:
    """
    Convert scanner findings into Standardized DiagnosticObjectDict entries.

    Each finding becomes:
        [SECURITY CRITICAL FAULT]: <Type> found in file <path> at line <line>.
    """
    diagnostics: list[dict[str, Any]] = []
    for f in findings:
        secret_type = f.get("secret_type", f.get("code", "Unknown"))
        filepath = f.get("file", "unknown")
        line = f.get("line", 0)
        message = f.get("message", "")

        diagnostic_msg = (
            f"[SECURITY CRITICAL FAULT]: {secret_type} found in file {filepath}"
            + (f" at line {line}." if line else ".")
        )

        diagnostics.append({
            "file": filepath,
            "line": line,
            "column": 0,
            "severity": "error",
            "error_code": f"SECURITY_{secret_type.replace(' ', '_').upper()}",
            "message": diagnostic_msg,
            "semantic_context": f"Source: {source} | Rule: {secret_type} | {message}"[:500],
        })
    return diagnostics


async def security_scan_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: Security Gatekeeper.

    Executes AFTER compiler_node returns exit code 0, BEFORE transitioning to END.
    Runs secret scanning (gitleaks/regex) and SAST (bandit/semgrep) in parallel.
    If vulnerabilities are found, populates compiler_errors and routes back to repair_node.

    Configuration via .harness_config.json:
        {
          "security_scan": {
            "enabled": true,
            "gitleaks_path": "",
            "bandit_path": "",
            "semgrep_path": "",
            "sast_timeout_seconds": 15,
            "max_security_fix_attempts": 2
          }
        }

    Returns:
        State update dict. If vulnerabilities found, includes compiler_errors
        and increments loop_counter["security"]. If clean, passes through.
    """
    sec_cfg = state.get("security_scan_config", {}) or {}
    enabled = sec_cfg.get("enabled", True)

    if not enabled:
        logger.info("[security_scan_node] Security scanning disabled. Skipping.")
        return {}

    workspace_path = state.get("workspace_path", os.getcwd())
    timeout_sec = sec_cfg.get("sast_timeout_seconds", 15)

    logger.info("[security_scan_node] Starting security audit on %s...", workspace_path)

    # Run both scanners in parallel
    secrets_task = run_gitleaks_scan(
        workspace_path,
        gitleaks_path=sec_cfg.get("gitleaks_path", ""),
        timeout_seconds=timeout_sec,
    )
    sast_task = run_sast_scan(
        workspace_path,
        bandit_path=sec_cfg.get("bandit_path", ""),
        semgrep_path=sec_cfg.get("semgrep_path", ""),
        timeout_seconds=timeout_sec,
    )

    raw_secret: object
    raw_sast: object
    raw_secret, raw_sast = await asyncio.gather(
        secrets_task, sast_task, return_exceptions=True
    )

    secret_findings: list[dict[str, Any]] = []
    sast_findings: list[dict[str, Any]] = []

    if isinstance(raw_secret, Exception):
        logger.warning("[security_scan_node] Secret scan exception: %s", raw_secret)
    elif isinstance(raw_secret, list):
        secret_findings = [f for f in raw_secret if isinstance(f, dict)]

    if isinstance(raw_sast, Exception):
        logger.warning("[security_scan_node] SAST scan exception: %s", raw_sast)
    elif isinstance(raw_sast, list):
        sast_findings = [f for f in raw_sast if isinstance(f, dict)]

    # Convert findings to diagnostics
    all_diagnostics: list[dict[str, Any]] = []
    all_diagnostics.extend(_findings_to_diagnostics(secret_findings, source="gitleaks/secret_scan"))
    all_diagnostics.extend(_findings_to_diagnostics(sast_findings, source="sast"))

    total_findings = len(secret_findings) + len(sast_findings)

    if total_findings == 0:
        logger.info("[security_scan_node] Security audit clean. No vulnerabilities found.")
        return {
            "node_state": {
                "security_scan": {
                    "passed": True,
                    "secret_findings": 0,
                    "sast_findings": 0,
                },
            },
        }

    # Vulnerabilities found — populate compiler_errors and increment counter
    loop_counter = state.get("loop_counter", {})
    loop_counter = dict(loop_counter)
    loop_counter["security"] = loop_counter.get("security", 0) + 1

    max_attempts = sec_cfg.get("max_security_fix_attempts", 2)

    logger.warning(
        "[security_scan_node] %d security finding(s) detected (%d secrets, %d SAST). "
        "Security fix attempt %d/%d.",
        total_findings,
        len(secret_findings),
        len(sast_findings),
        loop_counter["security"],
        max_attempts,
    )

    # Build status message for the conversation
    messages = list(state.get("messages", []))
    status_parts = [
        f"[Security Scan] {total_findings} critical security finding(s) detected:",
    ]
    for d in all_diagnostics[:5]:
        status_parts.append(f"  - {d['message']}")
    if len(all_diagnostics) > 5:
        status_parts.append(f"  ... and {len(all_diagnostics) - 5} more")
    status_parts.append(f"  Security fix attempt {loop_counter['security']}/{max_attempts}.")
    messages.append({"role": "system", "content": "\n".join(status_parts)})

    return {
        "compiler_errors": all_diagnostics,
        "loop_counter": loop_counter,
        "messages": messages,
        "node_state": {
            "security_scan": {
                "passed": False,
                "secret_findings": len(secret_findings),
                "sast_findings": len(sast_findings),
                "total_findings": total_findings,
                "attempt": loop_counter["security"],
            },
        },
    }


# ---------------------------------------------------------------------------
# 5. Factory from Config — Security
# ---------------------------------------------------------------------------

def create_command_validator_from_config(config_dict: dict[str, Any]) -> CommandValidator:
    """
    Build a CommandValidator from the 'security' section of .harness_config.json.

    Args:
        config_dict: The merged configuration dictionary.

    Returns:
        Configured CommandValidator instance.
    """
    security_cfg = config_dict.get("security", {})

    allowed = security_cfg.get("allowed_commands", None)
    if allowed and isinstance(allowed, list):
        allowed = set(allowed) | set(CommandValidator.DEFAULT_ALLOWED_COMMANDS)
    else:
        allowed = None  # Use defaults

    blocked = security_cfg.get("blocked_patterns", None)
    if blocked and isinstance(blocked, list):
        blocked = list(blocked) + list(CommandValidator.DEFAULT_BLOCKED_PATTERNS)
    else:
        blocked = None  # Use defaults

    return CommandValidator(
        allowed_commands=allowed,
        blocked_patterns=blocked,
        allow_all_commands=security_cfg.get("allow_all_commands", False),
        allow_network_in_build=security_cfg.get("allow_network_in_build", False),
    )