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
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
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
            encoding="utf-8",
            errors="replace",
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

    def commit_repair_iteration(
        self,
        session_id: str,
        iteration: int,
        modified_files: list[str],
        success_count: int,
        fail_count: int,
        exit_code: int,
    ) -> bool:
        """Commit the working tree mid-loop with a structured per-iteration
        message. Enables ``git log`` / ``git bisect`` between iterations and
        clean rollback (the patch branch is deleted on session abandon, so
        these commits vanish along with the branch).

        Idempotent / best-effort: returns True when nothing needed committing
        (clean tree, not a git repo, not on an agent patch branch) so the
        caller never has to gate the call. Failures are logged + swallowed;
        the repair loop continues regardless.

        Detects the patch branch from ``get_current_branch()`` (rather than
        the in-process ``_branch_created`` flag) so the caller can construct
        a fresh ``GitGuardian(workspace)`` without re-running setup. This
        lets ``repair_node`` invoke it from inside the graph without
        threading the session-level GitGuardian through state.
        """
        if not self.is_git_repo():
            return True
        if not modified_files:
            return True
        if not self.has_uncommitted_changes():
            return True
        current = self.get_current_branch() or ""
        if not current.startswith("agent/patch-"):
            # Don't commit unless we're on a harness-managed patch branch —
            # we never want per-iteration commits to land on the operator's
            # own working branch.
            return True

        result = self._git("add", "-A", "--", *modified_files)
        if result.returncode != 0:
            logger.warning(
                "[gitguardian] Per-iteration `git add` failed: %s",
                result.stderr.strip(),
            )
            return False

        # Refuse to commit if staging produced nothing — happens when every
        # entry in modified_files was already at the latest committed state.
        staged = self._git("diff", "--cached", "--name-only")
        if staged.returncode == 0 and not staged.stdout.strip():
            return True

        message = (
            f"[harness] Repair iteration {iteration} — "
            f"session {session_id}\n\n"
            f"Patches: {success_count} succeeded, {fail_count} failed\n"
            f"Build exit code (after this iteration's patches): {exit_code}\n"
            f"Files touched: {len(modified_files)}"
        )
        result = self._git("commit", "-m", message)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            # "nothing to commit" is a benign race — treat as success.
            if "nothing to commit" in stderr.lower():
                return True
            logger.warning(
                "[gitguardian] Per-iteration commit failed: %s", stderr,
            )
            return False
        # Use the live branch name (current HEAD), not self._patch_branch —
        # callers that construct a fresh GitGuardian without running
        # create_patch_branch leave self._patch_branch as None, which
        # showed up as `'None'` in the log line. The current-branch probe
        # is cheap and gives the operator the actual branch name they
        # can `git checkout`.
        logger.info(
            "[gitguardian] Committed repair iteration %d on '%s' "
            "(%d patches, exit %d).",
            iteration, self.get_current_branch() or "(detached HEAD)",
            success_count, exit_code,
        )
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
            "allowed_commands": ["make", "pytest", "python", "npm", ...],
            "blocked_patterns": ["curl", "wget", "sudo", ...],
            "allow_all_commands": false,
            "allow_network_in_build": false
          }
        }
    """

    # Default whitelist: safe build/dev tools
    # NB: ``sh`` / ``bash`` / ``dash`` are intentionally NOT in the default
    # allowlist. Audit §3.3 — letting shells through and then short-
    # circuiting the per-segment check on ``base_cmd in ("sh","bash","...")``
    # was a full validator bypass: an LLM-emitted ``bash -c 'cat /etc/shadow'``
    # passes both the whitelist (one segment, basename=bash, skipped) and
    # the blocklist (no individual blocked token). Operators who really
    # need a shell wrapper in their build_command must opt in by adding
    # ``sh``/``bash`` to ``security.allowed_commands`` AND ensuring the
    # inner command is sanitised at the call site.
    DEFAULT_ALLOWED_COMMANDS: set[str] = {
        "make", "cmake", "ninja",
        "python", "python3", "pip", "pip3", "poetry", "uv",
        "node", "npm", "npx",
        "javac", "java", "mvn", "gradle",
        "pytest", "unittest", "tox", "nox",
        "echo", "cat", "ls", "cp", "mv", "mkdir", "rm", "chmod", "chown",
        "git", "hg",
        "docker", "docker-compose", "podman",
        "test", "[",
        "true", "false",
        # ``cd`` is a shell builtin used by planner-emitted build commands
        # for monorepo / non-flat layouts (``cd server && pytest``).
        # Allowlisting adds no real attack surface — the docker sandbox is
        # the actual isolation boundary, and ``cd`` cannot touch files
        # outside the working tree. Blocking it forced repair LLMs into
        # an unfixable loop (the global validator config is unreachable
        # via the per-workspace patcher allowlist).
        "cd",
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
        # NOTE: `/dev/null`, `eval`, and `exec` are NOT hard-blocked any
        # more — they were breaking ordinary builds (`mvn ... > /dev/null`,
        # `cmake -E execute_process`, `docker exec`, `kubectl exec`, even
        # `git exec-path`). The dangerous shapes — `eval $(curl …)`,
        # `bash -c 'exec …'`, redirecting build output into the network —
        # are covered by the `wget|sh`, `curl|sh`, and `source <(…)`
        # patterns below.
        r"\bwget\b.*\|.*sh\b",   # curl | bash pattern
        r"\bcurl\b.*\|.*sh\b",   # curl | bash pattern
        # Match `eval` / `exec` only at shell-builtin position: line
        # start, after a pipe/semicolon/&&, or directly after `bash -c '`.
        # This excludes `docker exec`, `kubectl exec`, `git exec-path`,
        # `cmake -E execute_process`, npm `exec`, etc.
        r"(?:^|[|;&]\s*|bash\s+-c\s*['\"])\s*eval\s+",
        r"(?:^|[|;&]\s*|bash\s+-c\s*['\"])\s*exec\s+[^/]",
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
        # e.g., "PYTHONPATH=... pytest" → "pytest"
        # The previous single-shot regex only stripped ONE leading
        # assignment, so multi-env commands like
        # "FOO=1 BAR=2 pytest" were rejected by the whitelist because the
        # basename of "BAR=2" isn't a known command. Strip ALL leading
        # "KEY=VALUE " pairs in one pass.
        clean_for_parsing = re.sub(
            r'^(?:[A-Za-z_][A-Za-z0-9_]*=[^\s;]*\s+)+',
            '',
            command_stripped,
        )

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
        # If network is not explicitly allowed, block any URL/domain patterns.
        # Loopback addresses (127.0.0.1, 0.0.0.0, ::1) are NOT treated as
        # "network": tests routinely bind to 127.0.0.1, and refusing them
        # broke many test commands. We require non-loopback IPs (or scheme
        # URLs that resolve off-host).
        if not self.allow_network_in_build:
            loopback_ip_re = re.compile(
                r'\b(?:127\.0\.0\.\d{1,3}|0\.0\.0\.0|::1)\b'
            )
            command_no_loopback = loopback_ip_re.sub('', command_stripped)
            url_patterns = [
                r'https?://(?!(?:localhost|127\.|0\.0\.0\.0))[^\s]+',
                r'ftp://[^\s]+',
                r'\b(?!127\.|0\.0\.0\.0)[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}(?!\.\d)',
            ]
            for url_pattern in url_patterns:
                if re.search(url_pattern, command_no_loopback):
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
            # Strip subshell / group / negation prefixes that bash treats
            # as syntax, not as a command name. Without this, a build line
            # like ``(test -d X || uv venv X) && pip install …`` was
            # rejected as ``whitelist_missing:(test`` — the validator saw
            # the literal ``(test`` token because of the subshell paren.
            # We also strip a trailing ``)`` from the same token so a
            # one-command subshell ``(true)`` resolves to ``true``.
            first = tokens[0].lstrip("(!{").rstrip(")}")
            if not first:
                # Token was entirely syntax (e.g. an opening ``(`` by
                # itself, which can legitimately occur as its own word).
                # Nothing actionable for the whitelist; move on to the
                # next split part.
                continue
            base_cmd = os.path.basename(first)  # Strip path if present

            # Skip common shell builtins and operators.
            #
            # NOTE: ``sh`` / ``bash`` / ``dash`` are intentionally NOT in
            # this skip-list (audit §3.3). Earlier they were skipped here,
            # but in combination with their presence in the default
            # allowlist that meant ``bash -c 'arbitrary'`` was a wholesale
            # validator bypass — the basename was ``bash``, the loop
            # continued, and the inner shell payload was never inspected.
            # If a build legitimately needs ``bash -c '...'``, the operator
            # must add ``bash`` to ``security.allowed_commands`` and accept
            # that the validator can no longer see inside the ``-c`` arg.
            if base_cmd in ("", ".", "source", "export", "env", "exec"):
                continue

            if base_cmd not in self.allowed_commands:
                return CommandValidationResult(
                    allowed=False,
                    command=command,
                    # NOTE for repair LLMs reading this string in build
                    # output: the validator config lives in the operator's
                    # global config, NOT in the workspace. Do NOT create
                    # or patch `.harness_config.json` — the patcher
                    # allowlist will refuse it and no round of edits will
                    # unblock the run. The fix on the LLM's side is to
                    # rewrite the failing command so `base_cmd` isn't the
                    # first token (e.g. drop stray leading parens, avoid
                    # invoking a binary that isn't on the whitelist,
                    # replace `sh -c '…'` with a direct invocation).
                    reason=(
                        f"Command '{base_cmd}' is not in the sandbox "
                        f"security-validator whitelist. This is an operator-"
                        f"side config the repair loop cannot reach: do NOT "
                        f"emit `CREATE_FILE .harness_config.json`. Instead, "
                        f"rewrite the offending command so its first token "
                        f"is a whitelisted binary (drop leading parens, "
                        f"avoid `sh -c`, use a direct binary invocation)."
                    ),
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
            # The "Tip" line is deliberately phrased for the OPERATOR
            # (who runs the CLI) and for a repair LLM reading this in
            # build output. The LLM cannot reach `.harness_config.json`
            # (the patcher allowlist blocks it and the file lives in
            # the operator's global config anyway), so we lead with the
            # LLM-side unblock — rewrite the command — before the
            # operator-side instruction. `_is_command_blocked_by_security`
            # keys off the `Matched Rule:` line so the exact format of
            # that line MUST stay stable across edits.
            raise ValueError(
                f"[SECURITY BLOCKED]: {result.reason}\n"
                f"  Command: {command}\n"
                f"  Matched Rule: {result.matched_rule}\n"
                f"  LLM: rewrite the command; do NOT create "
                f".harness_config.json.\n"
                f"  Operator: configure 'security.allowed_commands' / "
                f"'security.blocked_patterns' in the harness config to "
                f"adjust the whitelist."
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


# ---------------------------------------------------------------------------
# 4a. Uniform Scanner Output Shape
# ---------------------------------------------------------------------------
# Every scanner adapter returns a ScannerOutcome wrapping a list of
# SecurityFinding objects. This is the contract the gate and the routing
# logic in graph.py consume — adding a new scanner means writing one
# adapter that parses its native output into this shape.

# Canonical severity scale — all scanner-specific labels get normalized
# into one of these via _normalize_severity().
_SEVERITY_RANK: dict[str, int] = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
}


def _severity_at_or_above(severity: str, threshold: str) -> bool:
    """True when ``severity`` is at least as serious as ``threshold``."""
    s = _SEVERITY_RANK.get(severity.lower(), _SEVERITY_RANK["info"])
    t = _SEVERITY_RANK.get(threshold.lower(), _SEVERITY_RANK["info"])
    return s <= t


_BANDIT_SEVERITY_MAP = {
    "HIGH": "high", "MEDIUM": "medium", "LOW": "low",
    "high": "high", "medium": "medium", "low": "low",
}
_SEMGREP_SEVERITY_MAP = {
    "ERROR": "high",   "WARNING": "medium", "INFO": "low",
    "error": "high",   "warning": "medium", "info": "low",
}
_TRIVY_SEVERITY_MAP = {
    "CRITICAL": "critical", "HIGH": "high", "MEDIUM": "medium",
    "LOW": "low", "UNKNOWN": "info",
}


def _normalize_severity(scanner: str, raw: str) -> str:
    """Map a scanner's native severity label onto the canonical scale.

    Scanner-specific default rules come first so that gitleaks (which
    doesn't emit a per-finding severity) always lands at "high" even
    when ``raw`` is empty.
    """
    # gitleaks doesn't emit per-finding severity; secrets are always high
    # whether or not raw carries anything.
    if scanner == "gitleaks":
        return "high"
    if not raw:
        return "medium"
    if scanner == "bandit":
        return _BANDIT_SEVERITY_MAP.get(raw, "medium")
    if scanner == "semgrep":
        return _SEMGREP_SEVERITY_MAP.get(raw, "medium")
    if scanner == "trivy":
        return _TRIVY_SEVERITY_MAP.get(raw.upper(), "medium")
    return raw.lower() if raw.lower() in _SEVERITY_RANK else "medium"


@dataclass(frozen=True)
class SecurityFinding:
    """One vulnerability or secret finding from any scanner.

    The frozen=True flag makes findings hashable so callers can dedupe
    across scanners (semgrep + bandit will often double-report the
    same SQLi pattern in a Python file).
    """
    scanner: str                    # "bandit" | "semgrep" | "gitleaks" | "trivy"
    rule_id: str                    # e.g. "B201" or "python.lang.security.audit.sql-injection"
    severity: str                   # canonical: critical | high | medium | low | info
    file: str
    line: int
    message: str
    cwe: Optional[str] = None       # "CWE-89" etc. (None when scanner doesn't emit it)
    confidence: str = "medium"      # bandit-style high/medium/low — defaults to medium
    # Scanner-suggested autofix metadata. Semgrep, modern ESLint rules,
    # and some Trivy CVE fixes ship an exact replacement payload + the
    # line range it should overwrite. When ``fix`` is non-empty,
    # ``harness.autofix._fix_semgrep`` (Layer 1 of the security autofix)
    # consumes it directly via REPLACE_LINE_RANGE — no LLM round needed.
    # Empty when the scanner doesn't emit a fix; the diagnostic still
    # flows to the LLM repair loop as before.
    end_line: int = 0
    fix: str = ""

    def dedupe_key(self) -> tuple[str, str, int, str]:
        """Stable key for collapsing duplicates across scanners (same
        rule / file / line / message wins once)."""
        return (self.rule_id, self.file, self.line, self.message)


class ScannerStatus(str, Enum):
    """Why a scanner returned what it did. Use this — never the raw exit
    code — to decide whether the gate should react.

    OK and FOUND both mean the scanner ran successfully; FOUND just
    means it surfaced one or more findings. CRASHED / TIMEOUT /
    NOT_INSTALLED all mean the scanner did not give us a verdict and
    its result should be logged but never treated as "clean".
    """
    OK = "ok"
    FOUND = "found"
    CRASHED = "crashed"
    TIMEOUT = "timeout"
    NOT_INSTALLED = "not_installed"


@dataclass
class ScannerOutcome:
    scanner: str
    status: ScannerStatus
    findings: list[SecurityFinding] = field(default_factory=list)
    error: str = ""


_DEFAULT_BLOCK_ON: frozenset[str] = frozenset({"critical", "high"})
_DEFAULT_WARN_ON: frozenset[str] = frozenset({"medium"})
_DEFAULT_SCANNERS: tuple[str, ...] = ("gitleaks", "bandit", "semgrep", "trivy")

# Workspace-relative paths the scanners should NOT walk. `docs/` and
# `product_spec/` live in every greenfield workspace and almost always
# contain fenced code snippets (JWT examples, SQL placeholders, env
# stubs) that trip semgrep's `--config=auto` ruleset. The patcher's
# spec-driven allowlist refuses writes to those directories, so any
# finding the LLM tries to fix there is guaranteed to bounce — creating
# a HITL ping-pong loop. Excluding them up-front matches the patcher's
# scope: code is in roots[].path; everything else is documentation.
_DEFAULT_EXCLUDE_PATHS: tuple[str, ...] = ("docs", "product_spec")

# Multiplier on `max_security_fix_attempts`. When the same finding has
# survived (max_attempts × this) trips through compile → security_scan →
# HITL → resume, the loop is thrashing — terminate the run rather than
# loop until something times out and silently shadows the finding.
_HARD_SECURITY_CEILING_MULTIPLIER: int = 3

# Install hints surfaced by `teane doctor` when a scanner binary is not
# on PATH. Kept here (not in cli.py) so the runtime scanner code and the
# doctor share one source of truth.
SCANNER_INSTALL_HINTS: dict[str, str] = {
    "gitleaks": (
        "download from github.com/gitleaks/gitleaks/releases"
    ),
    "bandit": "pipx install bandit  # or pip install --user bandit",
    "semgrep": "pipx install semgrep  # or pip install --user semgrep",
    "trivy": (
        "see https://aquasecurity.github.io/trivy/latest/getting-started/installation/"
    ),
}


@dataclass
class SecurityScanPolicy:
    """Config-driven gate policy.

    Loaded from .harness_config.json under ``security_scan``:

        {
          "security_scan": {
            "block_on": ["critical", "high"],
            "warn_on": ["medium"],
            "ignore_below": "low",
            "scanners": ["semgrep", "gitleaks", "trivy"],
            "allowlist_rules": ["python.lang.security.audit.formatted-sql-query"],
            "max_findings_to_route_to_repair": 10
          }
        }
    """
    block_on: frozenset[str] = _DEFAULT_BLOCK_ON
    warn_on: frozenset[str] = _DEFAULT_WARN_ON
    ignore_below: str = "low"
    scanners: tuple[str, ...] = _DEFAULT_SCANNERS
    allowlist_rules: frozenset[str] = frozenset()
    max_findings_to_route_to_repair: int = 10
    exclude_paths: tuple[str, ...] = _DEFAULT_EXCLUDE_PATHS

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SecurityScanPolicy":
        """Build from a ``security_scan`` config block. Unknown keys are
        ignored so legacy configs (``enabled``, ``max_security_fix_attempts``,
        scanner paths) still pass through untouched."""
        block_on = cfg.get("block_on")
        warn_on = cfg.get("warn_on")
        scanners = cfg.get("scanners")
        allowlist = cfg.get("allowlist_rules")
        # `exclude_paths`: workspace-relative dirs the scanners skip. An
        # explicit empty list disables the default — operators who really
        # want to scan docs/ can set ``"exclude_paths": []``.
        excludes_raw = cfg.get("exclude_paths")
        if isinstance(excludes_raw, (list, tuple)):
            excludes = tuple(
                _normalize_exclude_path(p) for p in excludes_raw
                if isinstance(p, str) and _normalize_exclude_path(p)
            )
        else:
            excludes = _DEFAULT_EXCLUDE_PATHS
        return cls(
            block_on=(
                frozenset(s.lower() for s in block_on)
                if isinstance(block_on, (list, tuple)) and block_on
                else _DEFAULT_BLOCK_ON
            ),
            warn_on=(
                frozenset(s.lower() for s in warn_on)
                if isinstance(warn_on, (list, tuple))
                else _DEFAULT_WARN_ON
            ),
            ignore_below=str(cfg.get("ignore_below", "low")).lower(),
            scanners=(
                tuple(s.lower() for s in scanners)
                if isinstance(scanners, (list, tuple)) and scanners
                else _DEFAULT_SCANNERS
            ),
            allowlist_rules=(
                frozenset(allowlist)
                if isinstance(allowlist, (list, tuple))
                else frozenset()
            ),
            max_findings_to_route_to_repair=int(
                cfg.get("max_findings_to_route_to_repair", 10)
            ),
            exclude_paths=excludes,
        )


def _normalize_exclude_path(p: str) -> str:
    """Strip ``./`` and trailing slashes; drop absolute paths and ``..``
    segments (they'd let an operator point the exclusion outside the
    workspace and bypass the gate entirely)."""
    if not isinstance(p, str):
        return ""
    s = p.strip().lstrip("/").rstrip("/")
    while s.startswith("./"):
        s = s[2:]
    if not s or s.startswith("..") or "/.." in s or s == ".":
        return ""
    return s


def apply_policy(
    findings: Sequence[SecurityFinding],
    policy: SecurityScanPolicy,
) -> tuple[list[SecurityFinding], list[SecurityFinding]]:
    """Partition findings into (block, warn) per policy.

    Dropped silently:
        - Findings whose rule_id is in ``policy.allowlist_rules``
        - Findings whose severity is strictly below ``policy.ignore_below``
        - Findings whose severity falls into neither block_on nor warn_on
          (e.g. ``info`` when policy only mentions critical/high/medium)

    Returned block list is severity-sorted (critical first) and capped
    at ``policy.max_findings_to_route_to_repair`` so a 200-finding
    semgrep run doesn't drown the repair LLM. The warn list is not
    capped — it's only logged.

    Dedupe runs first: a finding seen by both bandit and semgrep on
    the same file:line for the same rule is counted once.
    """
    seen: set[tuple[str, str, int, str]] = set()
    deduped: list[SecurityFinding] = []
    for f in findings:
        key = f.dedupe_key()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)

    block: list[SecurityFinding] = []
    warn: list[SecurityFinding] = []
    for f in deduped:
        sev = f.severity.lower()
        if f.rule_id in policy.allowlist_rules:
            continue
        # Strict below the ignore_below floor → drop silently.
        # _severity_at_or_above returns False when sev is less severe.
        if not _severity_at_or_above(sev, policy.ignore_below):
            continue
        if sev in policy.block_on:
            block.append(f)
        elif sev in policy.warn_on:
            warn.append(f)
        # Severities listed in neither set are dropped (e.g. info when
        # only critical/high block and medium warns).

    block.sort(key=lambda f: _SEVERITY_RANK.get(f.severity.lower(), 99))
    return block[: policy.max_findings_to_route_to_repair], warn


# ---------------------------------------------------------------------------
# 4b. Scanner Adapters — Native Output → SecurityFinding
# ---------------------------------------------------------------------------

def _parse_gitleaks_json(stdout: str) -> list[SecurityFinding]:
    """Parse gitleaks JSON output. Gitleaks emits a top-level list.

    Secrets are uniformly high-severity — a leaked key in source is a
    breach class on its own regardless of which key it was.
    """
    try:
        data = json.loads(stdout) if stdout.strip() else []
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []

    findings: list[SecurityFinding] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        rule_id = str(item.get("RuleID", item.get("rule_id", "unknown-secret")))
        findings.append(SecurityFinding(
            scanner="gitleaks",
            rule_id=rule_id,
            severity="high",
            file=str(item.get("File", item.get("file", ""))),
            line=int(item.get("StartLine", item.get("line", 0)) or 0),
            message=str(item.get("Description", item.get("description", f"Secret detected: {rule_id}"))),
            cwe="CWE-798",  # Use of Hard-coded Credentials
            confidence="high",
        ))
    return findings


def _fallback_secret_scan(workspace_path: str) -> list[SecurityFinding]:
    """Python fallback secret scanner when gitleaks isn't on PATH.

    Uses the same regex set as ``harness/redactor.py`` so a leaked key
    that the redactor would have caught at output-time is also caught
    here at scan-time.
    """
    secret_patterns: list[tuple[str, str]] = [
        (r'\b(sk-(?:proj-)?[A-Za-z0-9]{20,})\b', "openai-api-key"),
        (r'\b(sk-ant-api[0-9]{2}-[A-Za-z0-9_-]{40,})\b', "anthropic-api-key"),
        (r'\b(gh[pousr]_[A-Za-z0-9]{20,})\b', "github-token"),
        (r'\b(AKIA[0-9A-Z]{16})\b', "aws-access-key"),
        (r'\b(eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)\b', "jwt-token"),
        (r'-----BEGIN (?:RSA|EC|DSA|OPENSSH|ENCRYPTED) PRIVATE KEY-----', "private-key"),
        (r'(?:postgres|mysql|mongodb|redis)://[^:]+:[^@\s]+@', "database-connection-string"),
        (r'\b(sk_live_[A-Za-z0-9]{24,})\b', "stripe-live-key"),
        (r'\b(xox[bpras]-[A-Za-z0-9-]{10,})\b', "slack-token"),
    ]

    findings: list[SecurityFinding] = []
    ignore_dirs = {".git", "__pycache__", "node_modules", "vendor", "target", "build", "dist", ".tox", ".venv", "venv"}
    # Files we never scan: .env-style secret files (these intentionally
    # carry secrets; flagging them creates noise) AND the harness's own
    # operational files (managed by _is_harness_owned_path so both the
    # fallback and real-gitleaks code paths share one definition).
    env_basenames = {".env", ".env.local", ".env.production"}

    try:
        for root, dirs, files in os.walk(workspace_path):
            dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith(".")]
            for filename in files[:100]:
                if filename in env_basenames or _is_harness_owned_path(filename):
                    continue
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except (OSError, UnicodeDecodeError):
                    continue

                for line_num, line in enumerate(content.splitlines(), 1):
                    for pattern, rule_id in secret_patterns:
                        if re.search(pattern, line):
                            findings.append(SecurityFinding(
                                scanner="gitleaks-fallback",
                                rule_id=rule_id,
                                severity="high",
                                file=os.path.relpath(filepath, workspace_path),
                                line=line_num,
                                message=f"Hardcoded {rule_id} detected by regex fallback scanner.",
                                cwe="CWE-798",
                                confidence="medium",  # regex matching is more conservative than gitleaks rules
                            ))
    except Exception as exc:
        logger.warning("[security_scan] Fallback secret scan error: %s", exc)

    return findings


async def run_gitleaks_scan(
    workspace_path: str,
    gitleaks_path: str = "gitleaks",
    timeout_seconds: int = 15,
    exclude_paths: Sequence[str] = (),
) -> ScannerOutcome:
    """Run gitleaks for secret detection.

    Returns a ScannerOutcome so the gate can distinguish "gitleaks said
    clean", "gitleaks found things", and "gitleaks isn't installed and
    we fell back to regex" from each other. Exit code is intentionally
    not the deciding signal — gitleaks returns 0 with our ``--exit-code 0``
    flag whether or not it finds anything, but a non-empty findings list
    is still meaningful.
    """
    resolved = shutil.which(gitleaks_path) if gitleaks_path else shutil.which("gitleaks")
    if not resolved:
        logger.info("[security_scan] gitleaks not on PATH. Using Python fallback.")
        findings = _fallback_secret_scan(workspace_path)
        status = ScannerStatus.FOUND if findings else ScannerStatus.OK
        return ScannerOutcome(scanner="gitleaks-fallback", status=status, findings=findings)

    cmd = [
        resolved, "detect",
        "--source", workspace_path,
        "--no-git",
        "--report-format", "json",
        "--exit-code", "0",  # don't fail on findings; we own the parsing
    ]
    exit_code, stdout, stderr = await _run_subprocess_scanner(
        cmd, timeout_seconds=timeout_seconds, label="gitleaks",
    )

    if exit_code == -1 and "timed out" in stderr:
        # _run_subprocess_scanner already logged the timeout.
        return ScannerOutcome(scanner="gitleaks", status=ScannerStatus.TIMEOUT, error=stderr)
    if exit_code != 0 and not stdout.strip():
        # No output AND non-zero exit = genuine crash, not findings.
        logger.warning(
            "[security_scan] gitleaks crashed (exit=%d). Falling back to regex.",
            exit_code,
        )
        findings = _fallback_secret_scan(workspace_path)
        status = ScannerStatus.FOUND if findings else ScannerStatus.OK
        return ScannerOutcome(
            scanner="gitleaks-fallback", status=status, findings=findings,
            error=f"gitleaks crashed: {stderr[:200]}",
        )

    findings = _parse_gitleaks_json(stdout)
    findings = [f for f in findings if not _is_harness_owned_path(f.file)]
    # gitleaks has no clean --exclude flag for ad-hoc paths (it wants a
    # full config file). Filter post-hoc so the same exclude_paths list
    # applies uniformly across all four scanners.
    if exclude_paths:
        findings = [f for f in findings if not _path_under_excludes(f.file, exclude_paths)]
    if findings:
        logger.warning("[security_scan] gitleaks found %d secret(s).", len(findings))
    else:
        logger.info("[security_scan] gitleaks: no secrets found.")
    return ScannerOutcome(
        scanner="gitleaks",
        status=ScannerStatus.FOUND if findings else ScannerStatus.OK,
        findings=findings,
    )


def _path_under_excludes(rel_path: str, exclude_paths: Sequence[str]) -> bool:
    """True when ``rel_path`` is inside one of the excluded directories.

    The path comparison is path-component-wise (so ``docs2/x.md`` is NOT
    excluded by ``docs``) and case-sensitive (workspaces are POSIX-style)."""
    if not rel_path:
        return False
    norm = rel_path.replace("\\", "/").lstrip("/")
    for ex in exclude_paths:
        if not ex:
            continue
        prefix = ex.rstrip("/") + "/"
        if norm == ex or norm.startswith(prefix):
            return True
    return False


# Files the harness owns / writes into the workspace that legitimately
# carry API keys and similar operator-side state. Centralised here so the
# fallback scanner (above) and the real-gitleaks post-filter (below) both
# use the same list. Match on basename so any directory the harness writes
# them into is covered.
_HARNESS_OWNED_BASENAMES: frozenset[str] = frozenset({
    ".harness_config.json",
    ".harness_session.lock",
})


def _is_harness_owned_path(rel_path: str) -> bool:
    """True when ``rel_path`` is one of the harness's own operational
    files inside the workspace and should NOT be reported by the
    security scan."""
    if not rel_path:
        return False
    return os.path.basename(rel_path) in _HARNESS_OWNED_BASENAMES


def _scan_workspace_languages(workspace_path: str) -> tuple[bool, bool, bool]:
    """Cheap one-pass walk: do any .py / .js|.ts files exist?

    Capped at 200 files so a huge tree doesn't slow the gate. Sufficient
    to decide whether bandit / semgrep are worth invoking at all. The
    middle tuple slot is retained as ``False`` for return-shape stability.
    """
    has_py = has_js_ts = False
    for root, dirs, files in os.walk(workspace_path):
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".")
            and d not in ("node_modules", "vendor", "__pycache__", "target", "build", "dist")
        ]
        for fname in files[:200]:
            ext = os.path.splitext(fname)[1].lower()
            if ext in (".py", ".pyi"):
                has_py = True
            elif ext in (".ts", ".tsx", ".js", ".jsx", ".mjs"):
                has_js_ts = True
        if has_py and has_js_ts:
            break
    return has_py, False, has_js_ts


def _parse_bandit_json(stdout: str) -> list[SecurityFinding]:
    """Parse bandit JSON output. CWE comes from ``issue_cwe.id``."""
    try:
        data = json.loads(stdout) if stdout.strip() else {}
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    _results_raw = data.get("results")
    results: list[Any] = _results_raw if isinstance(_results_raw, list) else []

    findings: list[SecurityFinding] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        _cwe_raw = r.get("issue_cwe")
        cwe_obj: dict[str, Any] = _cwe_raw if isinstance(_cwe_raw, dict) else {}
        cwe_id = cwe_obj.get("id")
        findings.append(SecurityFinding(
            scanner="bandit",
            rule_id=str(r.get("test_id", "BANDIT")),
            severity=_normalize_severity("bandit", r.get("issue_severity", "")),
            file=str(r.get("filename", "")),
            line=int(r.get("line_number", 0) or 0),
            message=str(r.get("issue_text", r.get("test_name", "Bandit finding"))),
            cwe=f"CWE-{cwe_id}" if cwe_id else None,
            confidence=str(r.get("issue_confidence", "medium")).lower(),
        ))
    return findings


def _parse_semgrep_json(stdout: str) -> list[SecurityFinding]:
    """Parse semgrep JSON. Semgrep nests metadata under ``extra``; CWE
    lives in ``extra.metadata.cwe`` (sometimes a list)."""
    try:
        data = json.loads(stdout) if stdout.strip() else {}
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    _results_raw = data.get("results")
    results: list[Any] = _results_raw if isinstance(_results_raw, list) else []

    findings: list[SecurityFinding] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        _extra_raw = r.get("extra")
        extra: dict[str, Any] = _extra_raw if isinstance(_extra_raw, dict) else {}
        _meta_raw = extra.get("metadata")
        meta: dict[str, Any] = _meta_raw if isinstance(_meta_raw, dict) else {}
        cwe_raw = meta.get("cwe")
        if isinstance(cwe_raw, list) and cwe_raw:
            cwe = str(cwe_raw[0])
        elif isinstance(cwe_raw, str):
            cwe = cwe_raw
        else:
            cwe = None
        if cwe:
            # Some rulesets emit "CWE-89: SQL Injection", others "CWE-89",
            # others just "89". Normalize all three to "CWE-<digits>" so
            # the dedupe key and the diagnostic message both stay clean.
            cleaned = cwe.upper()
            if cleaned.startswith("CWE-"):
                cleaned = cleaned[len("CWE-"):]
            cleaned = cleaned.split(":")[0].strip()
            cwe = f"CWE-{cleaned}" if cleaned else None
        confidence_raw = meta.get("confidence", "medium")
        # Capture autofix metadata when the rule ships one. Semgrep puts
        # the suggested replacement in ``extra.fix`` and the byte/line
        # range under top-level ``start``/``end``. We propagate both so
        # the autofix path can issue REPLACE_LINE_RANGE without ever
        # entering the LLM repair loop. ``rendered_fix`` is the rule's
        # rendered template result; ``fix`` is the raw template — the
        # rendered form is what we want to apply.
        fix_raw = extra.get("rendered_fix") or extra.get("fix") or ""
        fix_str = str(fix_raw) if isinstance(fix_raw, str) else ""
        end_line_raw = (r.get("end") or {}).get("line", 0)
        try:
            end_line_val = int(end_line_raw or 0)
        except (TypeError, ValueError):
            end_line_val = 0
        findings.append(SecurityFinding(
            scanner="semgrep",
            rule_id=str(r.get("check_id", "semgrep.unknown")),
            severity=_normalize_severity("semgrep", extra.get("severity", "")),
            file=str(r.get("path", "")),
            line=int((r.get("start") or {}).get("line", 0) or 0),
            message=str(extra.get("message", "Semgrep finding")),
            cwe=cwe,
            confidence=str(confidence_raw).lower() if isinstance(confidence_raw, str) else "medium",
            end_line=end_line_val,
            fix=fix_str,
        ))
    return findings


def _parse_trivy_json(stdout: str) -> list[SecurityFinding]:
    """Parse ``trivy fs --format json`` output.

    Trivy emits a top-level dict with a ``Results`` list, each result a
    target file (lockfile, image layer, etc.) with its own
    ``Vulnerabilities`` array. We flatten to one finding per vuln —
    the dedup pass handles duplicates across overlapping targets.
    """
    try:
        data = json.loads(stdout) if stdout.strip() else {}
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    _results_raw = data.get("Results")
    results: list[Any] = _results_raw if isinstance(_results_raw, list) else []

    findings: list[SecurityFinding] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        target = result.get("Target", "")
        _vulns_raw = result.get("Vulnerabilities")
        vulns: list[Any] = _vulns_raw if isinstance(_vulns_raw, list) else []
        for v in vulns:
            if not isinstance(v, dict):
                continue
            _cwe_raw = v.get("CweIDs")
            cwe_ids: list[Any] = _cwe_raw if isinstance(_cwe_raw, list) else []
            cwe = str(cwe_ids[0]) if cwe_ids else None
            if cwe and not cwe.upper().startswith("CWE-"):
                cwe = f"CWE-{cwe}"
            pkg = v.get("PkgName", "")
            installed = v.get("InstalledVersion", "")
            fixed = v.get("FixedVersion") or ""
            title = v.get("Title") or v.get("Description") or v.get("VulnerabilityID", "")
            message_parts = [f"{pkg} {installed}: {title}"] if pkg else [title]
            if fixed:
                message_parts.append(f"Fix available: upgrade to {fixed}.")
            else:
                message_parts.append("No fix released — dependency-vuln may require workaround.")
            findings.append(SecurityFinding(
                scanner="trivy",
                rule_id=v.get("VulnerabilityID", "TRIVY"),
                severity=_normalize_severity("trivy", v.get("Severity", "")),
                file=target,
                line=0,  # Trivy reports per-package, not per-line
                message=" ".join(message_parts),
                cwe=cwe,
                confidence="high" if fixed else "medium",
            ))
    return findings


async def run_bandit_scan(
    workspace_path: str,
    bandit_path: str = "bandit",
    timeout_seconds: int = 15,
    exclude_paths: Sequence[str] = (),
) -> ScannerOutcome:
    """Bandit (Python SAST). No-ops with status OK when no .py files
    are present so a polyglot repo doesn't waste a timer slot on it."""
    has_py, _, _ = _scan_workspace_languages(workspace_path)
    if not has_py:
        return ScannerOutcome(scanner="bandit", status=ScannerStatus.OK)

    resolved = shutil.which(bandit_path) if bandit_path else shutil.which("bandit")
    if not resolved:
        logger.debug("[security_scan] bandit not on PATH. Skipping Python SAST.")
        return ScannerOutcome(scanner="bandit", status=ScannerStatus.NOT_INSTALLED)

    cmd = [resolved, "-r", "-f", "json", "-ll", "-q"]
    # Bandit's -x accepts comma-separated paths AND can be passed
    # multiple times. Use one --exclude per path so a path containing a
    # literal comma can't silently corrupt the list, and Windows
    # ``C:\…`` paths aren't reinterpreted. Skip any path that contains
    # a comma defensively — bandit treats commas as delimiters even in
    # the single-arg form so embedded commas would still break.
    abs_excludes = [
        os.path.join(workspace_path, ex)
        for ex in exclude_paths
        if ex and "," not in ex
    ]
    for ex in abs_excludes:
        cmd.extend(["--exclude", ex])
    cmd.append(workspace_path)
    exit_code, stdout, stderr = await _run_subprocess_scanner(
        cmd, timeout_seconds=timeout_seconds, label="bandit",
    )

    # Bandit returns 1 when it finds anything, 0 when clean, and >1 on
    # actual crashes. Never trust the exit code alone — parse first,
    # crash only if JSON is empty AND exit_code > 1.
    if exit_code == -1 and "timed out" in stderr:
        return ScannerOutcome(scanner="bandit", status=ScannerStatus.TIMEOUT, error=stderr)
    if not stdout.strip() and exit_code not in (0, 1):
        logger.warning("[security_scan] bandit crashed (exit=%d): %s", exit_code, stderr[:200])
        return ScannerOutcome(
            scanner="bandit", status=ScannerStatus.CRASHED,
            error=f"bandit crashed: {stderr[:200]}",
        )

    findings = _parse_bandit_json(stdout)
    if findings:
        logger.warning("[security_scan] bandit found %d issue(s).", len(findings))
    return ScannerOutcome(
        scanner="bandit",
        status=ScannerStatus.FOUND if findings else ScannerStatus.OK,
        findings=findings,
    )


async def run_semgrep_scan(
    workspace_path: str,
    semgrep_path: str = "semgrep",
    timeout_seconds: int = 30,
    exclude_paths: Sequence[str] = (),
) -> ScannerOutcome:
    """Semgrep (universal SAST). Useful for JS/TS and as a
    cross-language second opinion alongside bandit on Python."""
    resolved = shutil.which(semgrep_path) if semgrep_path else shutil.which("semgrep")
    if not resolved:
        logger.debug("[security_scan] semgrep not on PATH. Skipping universal SAST.")
        return ScannerOutcome(scanner="semgrep", status=ScannerStatus.NOT_INSTALLED)

    cmd = [
        resolved, "scan", "--config=auto", "--json", "--quiet",
        "--no-git-ignore",
    ]
    # semgrep accepts repeated `--exclude <pattern>` flags. Pass each
    # excluded directory by name so the matcher skips them anywhere in
    # the workspace tree.
    for ex in exclude_paths:
        if ex:
            cmd.extend(["--exclude", ex])
    cmd.append(workspace_path)
    exit_code, stdout, stderr = await _run_subprocess_scanner(
        cmd, timeout_seconds=timeout_seconds, label="semgrep",
    )

    if exit_code == -1 and "timed out" in stderr:
        return ScannerOutcome(scanner="semgrep", status=ScannerStatus.TIMEOUT, error=stderr)
    # Semgrep: exit 0 = clean, 1 = findings, 2 = errors but partial output,
    # higher = crash. Treat anything with valid JSON as a successful parse.
    if not stdout.strip() and exit_code > 1:
        logger.warning("[security_scan] semgrep crashed (exit=%d): %s", exit_code, stderr[:200])
        return ScannerOutcome(
            scanner="semgrep", status=ScannerStatus.CRASHED,
            error=f"semgrep crashed: {stderr[:200]}",
        )

    findings = _parse_semgrep_json(stdout)
    if findings:
        logger.warning("[security_scan] semgrep found %d issue(s).", len(findings))
    return ScannerOutcome(
        scanner="semgrep",
        status=ScannerStatus.FOUND if findings else ScannerStatus.OK,
        findings=findings,
    )


async def run_trivy_scan(
    workspace_path: str,
    trivy_path: str = "trivy",
    timeout_seconds: int = 60,
    exclude_paths: Sequence[str] = (),
) -> ScannerOutcome:
    """Trivy filesystem scan for dependency / package vulnerabilities.

    Picks up vulnerable transitive deps (npm, pip, maven/gradle) that
    SAST scanners can't see.

    The vulnerability DB is cached under ``~/.harness/cache/trivy`` and
    refreshed at most once every 24 hours; after that, ``--skip-db-update``
    is added so repeated scans within a single session (or back-to-back
    builds in CI) don't re-pull the ~150 MB DB on every invocation.
    """
    resolved = shutil.which(trivy_path) if trivy_path else shutil.which("trivy")
    if not resolved:
        logger.debug("[security_scan] trivy not on PATH. Skipping dep-vuln scan.")
        return ScannerOutcome(scanner="trivy", status=ScannerStatus.NOT_INSTALLED)

    cache_dir = os.path.expanduser("~/.harness/cache/trivy")
    skip_db_update = False
    stamp_path = ""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        stamp_path = os.path.join(cache_dir, ".db_refreshed_at")
        now = time.time()
        # Trivy stores its vulnerability DB under ``<cache_dir>/db/``.
        # The stamp check alone is not enough: ciod session 523e86a7
        # hit ``FATAL --skip-db-update cannot be specified on the first
        # run`` because the stamp file existed from an earlier session
        # but the ``db/`` directory (or the trivy.db inside it) was
        # missing — cleared by an operator, trimmed by disk-pressure
        # eviction, or wiped by a trivy version upgrade that changed the
        # DB schema path. Gate ``--skip-db-update`` on BOTH the stamp
        # AND the actual presence of the DB file so a first-run trivy
        # invocation always downloads without crashing.
        trivy_db_present = os.path.isfile(os.path.join(cache_dir, "db", "trivy.db"))
        if os.path.isfile(stamp_path) and trivy_db_present:
            try:
                last = os.path.getmtime(stamp_path)
            except OSError:
                last = 0.0
            if now - last < 24 * 60 * 60:
                skip_db_update = True
        # Stamp write is DEFERRED until the scan exits cleanly (see
        # below). The earlier "touch stamp before scan" shape left a
        # fresh stamp on the disk even when the trivy DB download died
        # mid-flight; the next call would then see the stamp, pass
        # --skip-db-update, and quietly use the half-downloaded DB.
    except OSError as exc:
        logger.debug("[security_scan] trivy cache setup skipped: %s", exc)
        cache_dir = ""
        stamp_path = ""

    cmd = [
        resolved, "fs", "--format", "json", "--quiet", "--no-progress",
        "--exit-code", "0",
    ]
    if cache_dir:
        cmd.extend(["--cache-dir", cache_dir])
    if skip_db_update:
        cmd.append("--skip-db-update")
    # trivy fs supports `--skip-dirs <dir>` (repeatable). Skip the same
    # documentation directories as the SAST scanners so trivy doesn't
    # parse vendored lockfiles bundled inside example docs.
    for ex in exclude_paths:
        if ex:
            cmd.extend(["--skip-dirs", ex])
    cmd.append(workspace_path)
    exit_code, stdout, stderr = await _run_subprocess_scanner(
        cmd, timeout_seconds=timeout_seconds, label="trivy",
    )

    if exit_code == -1 and "timed out" in stderr:
        return ScannerOutcome(scanner="trivy", status=ScannerStatus.TIMEOUT, error=stderr)
    if not stdout.strip() and exit_code != 0:
        logger.warning("[security_scan] trivy crashed (exit=%d): %s", exit_code, stderr[:200])
        return ScannerOutcome(
            scanner="trivy", status=ScannerStatus.CRASHED,
            error=f"trivy crashed: {stderr[:200]}",
        )

    findings = _parse_trivy_json(stdout)
    if findings:
        logger.warning("[security_scan] trivy found %d dep-vuln(s).", len(findings))
    # Only stamp the cache after a successful scan; this prevents a
    # half-downloaded DB (interrupted mid-download by SIGINT or OOM)
    # from being treated as fresh on the next call.
    if stamp_path and not skip_db_update and exit_code == 0:
        try:
            with open(stamp_path, "w", encoding="utf-8") as _f:
                _f.write(str(time.time()))
        except OSError as stamp_err:
            logger.debug(
                "[security_scan] trivy stamp write failed (%s); next call will refresh DB.",
                stamp_err,
            )
    return ScannerOutcome(
        scanner="trivy",
        status=ScannerStatus.FOUND if findings else ScannerStatus.OK,
        findings=findings,
    )


# Kept for backwards compatibility with any caller that still expects
# the bundled SAST result. New code should call run_bandit_scan and
# run_semgrep_scan directly so individual scanner status surfaces in
# the gate.
async def run_sast_scan(
    workspace_path: str,
    bandit_path: str = "bandit",
    semgrep_path: str = "semgrep",
    timeout_seconds: int = 15,
) -> list[SecurityFinding]:
    """Legacy bundled SAST runner — runs bandit + semgrep and returns
    the combined finding list. Prefer the individual ``run_*_scan``
    functions for new code so individual scanner status is preserved.
    """
    bandit_outcome, semgrep_outcome = await asyncio.gather(
        run_bandit_scan(workspace_path, bandit_path, timeout_seconds),
        run_semgrep_scan(workspace_path, semgrep_path, timeout_seconds),
    )
    return [*bandit_outcome.findings, *semgrep_outcome.findings]


def _findings_to_diagnostics(
    findings: Sequence[SecurityFinding],
) -> list[dict[str, Any]]:
    """Convert SecurityFinding objects into DiagnosticObjectDict entries.

    Wraps the canonical fields (severity, rule_id, CWE, confidence) into
    one human-legible diagnostic message so the LLM repair prompt has
    everything it needs to fix the issue. The error_code carries the
    scanner + rule_id so the repair LLM can grep for the specific rule
    documentation.
    """
    diagnostics: list[dict[str, Any]] = []
    for f in findings:
        loc = f.file or "<unknown>"
        if f.line:
            loc = f"{loc}:{f.line}"
        msg_parts = [
            f"[SECURITY {f.severity.upper()}]",
            f"{f.scanner}/{f.rule_id}",
            f"in {loc}:",
            f.message,
        ]
        if f.cwe:
            msg_parts.append(f"({f.cwe})")
        diag: dict[str, Any] = {
            "file": f.file or "unknown",
            "line": f.line,
            "column": 0,
            # Critical / high security findings are hard errors for the
            # repair loop; medium becomes a warning so the LLM still
            # sees it without it bumping the build to "broken".
            "severity": "error" if f.severity in ("critical", "high") else "warning",
            "error_code": f"{f.scanner.upper()}:{f.rule_id}",
            "message": " ".join(msg_parts),
            "semantic_context": (
                f"Scanner: {f.scanner} | Rule: {f.rule_id} | "
                f"Severity: {f.severity} | Confidence: {f.confidence}"
                + (f" | {f.cwe}" if f.cwe else "")
            )[:500],
        }
        # Scanner-suggested autofix passthrough. _fix_semgrep reads these
        # fields directly and emits REPLACE_LINE_RANGE when ``fix`` is
        # populated, bypassing the LLM repair loop for any rule whose
        # upstream community shipped a fix template.
        if f.fix:
            diag["fix"] = f.fix
            diag["end_line"] = f.end_line or f.line
        diagnostics.append(diag)
    return diagnostics


async def security_scan_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: deterministic security gate.

    Runs AFTER ``compiler_node`` exits 0 (failing builds aren't worth
    scanning — the scanner's output would mostly be artifacts of broken
    code anyway). The configured scanners run in parallel inside the
    timeout budget. Their findings are normalized to ``SecurityFinding``,
    deduped, allowlisted, and partitioned by ``SecurityScanPolicy`` into:

        * **block** — populated into ``compiler_errors`` so
          ``route_after_security_scan`` ships them to ``patching_node``;
          capped at ``policy.max_findings_to_route_to_repair`` to keep
          the repair prompt focused.
        * **warn** — logged into ``node_state.security_scan.warnings``
          and surfaced in the system message, but the build keeps moving.

    Anything below ``policy.ignore_below`` or matching an allowlisted
    rule_id is dropped silently. Scanner crashes (CRASHED / TIMEOUT)
    surface in ``node_state.security_scan.crashed_scanners`` but do
    NOT count as a clean pass — log + continue, the user can rerun
    with stricter config when they understand why.

    Configuration via .harness_config.json::

        {
          "security_scan": {
            "enabled": true,
            "block_on": ["critical", "high"],
            "warn_on": ["medium"],
            "ignore_below": "low",
            "scanners": ["gitleaks", "bandit", "semgrep", "trivy"],
            "allowlist_rules": ["python.lang.security.audit.formatted-sql-query"],
            "max_findings_to_route_to_repair": 10,
            "gitleaks_path": "", "bandit_path": "", "semgrep_path": "", "trivy_path": "",
            "sast_timeout_seconds": 60,
            "trivy_timeout_seconds": 60,
            "max_security_fix_attempts": 2
          }
        }
    """
    sec_cfg = state.get("security_scan_config", {}) or {}
    if not sec_cfg.get("enabled", True):
        logger.info("[security_scan_node] Security scanning disabled. Skipping.")
        return {}

    workspace_path = state.get("workspace_path", os.getcwd())
    # Finsearch session 44c5e194 root cause E3: semgrep timed out at
    # 15s on a workspace with ~30 files, leaving the security scan
    # INCOMPLETE. Bumped default to 60s to give scanners a realistic
    # budget on non-trivial workspaces; operators can lower via
    # ``security.sast_timeout_seconds`` if latency matters more.
    timeout_sec = int(sec_cfg.get("sast_timeout_seconds", 60))
    trivy_timeout = int(sec_cfg.get("trivy_timeout_seconds", 60))
    policy = SecurityScanPolicy.from_config(sec_cfg)

    logger.info(
        "[security_scan_node] Starting audit on %s | scanners=%s | block=%s warn=%s "
        "ignore_below=%s exclude_paths=%s",
        workspace_path,
        list(policy.scanners),
        sorted(policy.block_on),
        sorted(policy.warn_on),
        policy.ignore_below,
        list(policy.exclude_paths),
    )

    # Build the task list dynamically — only enabled scanners run.
    tasks: list[Any] = []
    if "gitleaks" in policy.scanners:
        tasks.append(run_gitleaks_scan(
            workspace_path,
            gitleaks_path=sec_cfg.get("gitleaks_path", ""),
            timeout_seconds=timeout_sec,
            exclude_paths=policy.exclude_paths,
        ))
    if "bandit" in policy.scanners:
        tasks.append(run_bandit_scan(
            workspace_path,
            bandit_path=sec_cfg.get("bandit_path", ""),
            timeout_seconds=timeout_sec,
            exclude_paths=policy.exclude_paths,
        ))
    if "semgrep" in policy.scanners:
        tasks.append(run_semgrep_scan(
            workspace_path,
            semgrep_path=sec_cfg.get("semgrep_path", ""),
            timeout_seconds=timeout_sec,
            exclude_paths=policy.exclude_paths,
        ))
    if "trivy" in policy.scanners:
        tasks.append(run_trivy_scan(
            workspace_path,
            trivy_path=sec_cfg.get("trivy_path", ""),
            timeout_seconds=trivy_timeout,
            exclude_paths=policy.exclude_paths,
        ))

    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    all_findings: list[SecurityFinding] = []
    crashed: list[tuple[str, str]] = []
    timed_out: list[str] = []
    not_installed: list[str] = []
    ran_clean: list[str] = []

    for raw in outcomes:
        if isinstance(raw, Exception):
            logger.warning("[security_scan_node] Scanner exception: %s", raw)
            crashed.append(("?", str(raw)[:200]))
            continue
        if not isinstance(raw, ScannerOutcome):
            continue
        if raw.status == ScannerStatus.CRASHED:
            crashed.append((raw.scanner, raw.error))
            continue
        if raw.status == ScannerStatus.TIMEOUT:
            timed_out.append(raw.scanner)
            continue
        if raw.status == ScannerStatus.NOT_INSTALLED:
            not_installed.append(raw.scanner)
            continue
        ran_clean.append(raw.scanner)
        all_findings.extend(raw.findings)

    # Partition by policy: block (route to repair), warn (log), drop.
    block, warn = apply_policy(all_findings, policy)

    # Render audit summary that survives into node_state for telemetry.
    summary = {
        "policy": {
            "block_on": sorted(policy.block_on),
            "warn_on": sorted(policy.warn_on),
            "ignore_below": policy.ignore_below,
            "exclude_paths": list(policy.exclude_paths),
        },
        "scanners_clean": sorted(ran_clean),
        "scanners_not_installed": sorted(not_installed),
        "scanners_timed_out": sorted(timed_out),
        "scanners_crashed": [name for name, _ in crashed],
        "raw_findings_total": len(all_findings),
        "block_count": len(block),
        "warn_count": len(warn),
        "warnings": [
            {
                "scanner": w.scanner, "rule_id": w.rule_id, "severity": w.severity,
                "file": w.file, "line": w.line, "cwe": w.cwe,
            }
            for w in warn[:25]  # cap warn-list snapshot
        ],
    }

    if not block:
        if warn:
            logger.warning(
                "[security_scan_node] Clean (block=0). %d medium-severity finding(s) "
                "passed through as warnings. See node_state.security_scan.warnings.",
                len(warn),
            )
        else:
            logger.info("[security_scan_node] Security audit clean. No vulnerabilities found.")
        if crashed or timed_out:
            logger.warning(
                "[security_scan_node] Some scanners did not run cleanly: crashed=%s timed_out=%s. "
                "Treat as INCOMPLETE coverage, not as a pass.",
                [name for name, _ in crashed], timed_out,
            )
        return {
            "node_state": {
                "security_scan": {"passed": True, **summary},
            },
        }

    # Findings to block on. Route to patching via compiler_errors.
    loop_counter = dict(state.get("loop_counter") or {})
    loop_counter["security"] = loop_counter.get("security", 0) + 1
    max_attempts = int(sec_cfg.get("max_security_fix_attempts", 2))

    diagnostics = _findings_to_diagnostics(block)

    # --- Deterministic autofix pass (R3) ---
    # Attempt to resolve known-safe security findings (bandit B201/B602,
    # gitleaks line removal, trivy dep-vuln bumps with FixedVersion)
    # without spending an LLM call. Anything still unhandled falls
    # through to the LLM-driven repair loop exactly as before.
    from harness.autofix import apply_autofixes, autofix_system_message
    unhandled_diagnostics, applied_fixes = await apply_autofixes(
        list(diagnostics), workspace_path,
    )
    autofix_modified_files = list(state.get("modified_files", []))
    autofix_message: Optional[dict[str, Any]] = None
    if applied_fixes:
        for r in applied_fixes:
            if r.file not in autofix_modified_files:
                autofix_modified_files.append(r.file)
        msg_text = autofix_system_message(applied_fixes)
        if msg_text:
            autofix_message = {"role": "system", "content": msg_text}
        logger.info(
            "[security_scan_node] autofix resolved %d of %d blocking finding(s) without LLM.",
            len(applied_fixes), len(diagnostics),
        )

    # If the autofix pass cleared every blocking finding, the gate passes
    # for this round. Routing then sends the build back through compile +
    # security_scan so we can confirm the fixes hold.
    if applied_fixes and not unhandled_diagnostics:
        passed_state: dict[str, Any] = {
            "modified_files": autofix_modified_files,
            "loop_counter": loop_counter,
            "node_state": {
                "security_scan": {
                    "passed": True,
                    "autofix_applied": len(applied_fixes),
                    "autofix_kinds": sorted({r.fix_kind for r in applied_fixes}),
                    **summary,
                },
            },
        }
        if autofix_message is not None:
            passed_state["messages"] = list(state.get("messages", [])) + [autofix_message]
        return passed_state

    # Otherwise hand the unhandled tail to the LLM repair path.
    diagnostics = unhandled_diagnostics

    logger.warning(
        "[security_scan_node] %d blocking finding(s) (warn=%d, dropped=%d). "
        "Security fix attempt %d/%d.",
        len(block), len(warn),
        len(all_findings) - len(block) - len(warn),
        loop_counter["security"], max_attempts,
    )

    # Build a conversation breadcrumb for the LLM. Group by scanner so the
    # repair prompt can address whole classes at once instead of N-of-the-same.
    by_scanner: dict[str, list[SecurityFinding]] = {}
    for f in block:
        by_scanner.setdefault(f.scanner, []).append(f)
    status_lines = [
        f"[Security Scan] {len(block)} blocking finding(s) detected "
        f"(attempt {loop_counter['security']}/{max_attempts}):",
    ]
    for scanner, items in by_scanner.items():
        status_lines.append(f"  {scanner} ({len(items)}):")
        for f in items[:5]:
            loc = f"{f.file}:{f.line}" if f.line else f.file
            cwe = f" {f.cwe}" if f.cwe else ""
            status_lines.append(f"    - [{f.severity.upper()}] {f.rule_id} @ {loc}{cwe} — {f.message}")
        if len(items) > 5:
            status_lines.append(f"    ... and {len(items) - 5} more {scanner} findings")
    if warn:
        status_lines.append(
            f"  ({len(warn)} additional warn-level finding(s) — see audit summary.)"
        )

    messages = list(state.get("messages", []))
    if autofix_message is not None:
        messages.append(autofix_message)
    # Architecture-summary preamble — tells the repair LLM to fix
    # the findings *within* the resolved stack (auth strategy, DB
    # driver, contract path, schema names) instead of inventing a
    # different secrets store or ORM. Empty string when the arch
    # doc has no §11 block; in that case the security breadcrumb
    # below stands alone, matching pre-existing behaviour.
    from harness.graph import _build_arch_summary_preamble
    # state is dict[str, Any] here, but the helper only does .get()
    # on it (workspace_path / arch_summary) so an explicit AgentState
    # cast would be decorative — pass through verbatim.
    arch_preamble, resolved_arch = _build_arch_summary_preamble(
        state, consumer="security",  # type: ignore[arg-type]
    )
    if arch_preamble:
        messages.append({"role": "system", "content": arch_preamble})
    messages.append({"role": "system", "content": "\n".join(status_lines)})

    # Rotate diagnostic fingerprints — see
    # graph._rotate_diag_fingerprints_delta. Security findings become
    # the new failing set that repair_node's reflection judge compares
    # round-over-round; without the rotation the judge sees stale
    # compile fingerprints and mis-classifies verdicts.
    from harness.graph import _rotate_diag_fingerprints_delta
    return {
        "compiler_errors": diagnostics,
        "loop_counter": loop_counter,
        "messages": messages,
        "modified_files": autofix_modified_files,
        # Pass the resolved summary through so a downstream
        # patching_node turn skips the disk read if it was lazy-loaded
        # here. Empty dict = no §11 block on disk (same caching
        # contract as patching_node's return delta).
        "arch_summary": resolved_arch,
        "node_state": {
            "security_scan": {
                "passed": False,
                "attempt": loop_counter["security"],
                "max_attempts": max_attempts,
                "autofix_applied": len(applied_fixes),
                **summary,
            },
        },
        **_rotate_diag_fingerprints_delta(state, diagnostics),
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


# ---------------------------------------------------------------------------
# 6. Global validator accessor
#
# Mirrors the pattern used by harness/redactor.py's global SecretScanner.
# `cmd_run` calls set_command_validator() at startup so every SandboxExecutor
# instantiated during the session picks it up automatically — defense-in-depth
# without having to thread the validator through every call site.
# ---------------------------------------------------------------------------

_global_command_validator: Optional[CommandValidator] = None


def set_command_validator(validator: Optional[CommandValidator]) -> None:
    """Set the process-wide default CommandValidator. Pass None to clear."""
    global _global_command_validator
    _global_command_validator = validator


def get_command_validator() -> Optional[CommandValidator]:
    """Return the process-wide default CommandValidator, or None if unset."""
    return _global_command_validator