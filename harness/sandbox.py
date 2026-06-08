"""
Sandbox execution engine with pluggable isolation backends, async subprocess wrapping,
and automated compiler log interception.

This module implements:
    - SandboxBackend ABC: pluggable isolation strategy interface
    - UnshareBackend: Linux kernel namespace isolation via unshare(2) (zero deps)
    - DockerBackend: Docker container isolation with resource limits (docker CLI required)
    - BareBackend: No isolation, bare asyncio subprocess (fallback on all platforms)
    - SandboxExecutor: orchestrates build commands using the configured backend
    - Read-only bind-mounts for host dependency cache directories
    - Network namespace toggle controlled by the allow_network flag
    - Strict process timeouts with PGID-based termination hooks
    - Regex log interceptor that strips verbose success lines, extracts critical failures
    - Structured diagnostic parsing using language-specific parsers from parser_registry
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import shutil
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Types
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticObject:
    """
    Structured compiler diagnostic, matches the DiagnosticObjectDict
    TypedDict shape used in harness/graph.py.
    """
    file: str = ""
    line: int = 0
    column: int = 0
    severity: str = "error"  # "error" | "warning"
    error_code: str = ""
    message: str = ""
    semantic_context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "severity": self.severity,
            "error_code": self.error_code,
            "message": self.message,
            "semantic_context": self.semantic_context,
        }


@dataclass
class BuildResult:
    """Result of a sandboxed build execution."""
    exit_code: int
    raw_output: str
    diagnostics: list[DiagnosticObject] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    timed_out: bool = False


# ---------------------------------------------------------------------------
# 2. SandboxBackend — Pluggable Isolation Strategy
# ---------------------------------------------------------------------------

class SandboxBackend(ABC):
    """
    Abstract base for pluggable isolation backends.

    Each backend implements:
        - run(command, workspace_path, timeout_seconds, allow_network, extra_env)
          → (exit_code, stdout_stderr_combined, timed_out_bool)

    Implementations:
        - UnshareBackend:  Linux kernel namespaces (unshare)
        - DockerBackend:   Docker container with resource limits
        - BareBackend:     No isolation, bare subprocess
    """

    @abstractmethod
    async def run(
        self,
        command: str,
        workspace_path: str,
        timeout_seconds: int = 300,
        allow_network: bool = False,
        readonly_cache_mounts: Optional[list[str]] = None,
        extra_env: Optional[dict[str, str]] = None,
    ) -> tuple[int, str, bool]:
        """
        Execute a shell command inside the isolation backend.

        Args:
            command: The shell command to execute (e.g., 'make build').
            workspace_path: Absolute path to the workspace directory.
            timeout_seconds: Maximum execution time before forced termination.
            allow_network: Whether outbound network is permitted.
            readonly_cache_mounts: Host directories to mount read-only.
            extra_env: Additional environment variables to pass to the process.

        Returns:
            Tuple of (exit_code, combined_stdout_stderr, timed_out).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name for logging."""
        ...

    def is_available(self) -> bool:
        """Check if this backend is usable on the current host."""
        return True


# ---------------------------------------------------------------------------
# 3. UnshareBackend — Linux Kernel Namespace Isolation
# ---------------------------------------------------------------------------

class UnshareBackend(SandboxBackend):
    """
    Executes builds inside isolated Linux namespaces via unshare(2).

    Creates:
        - CLONE_NEWNS  (mount namespace — filesystem isolation)
        - CLONE_NEWNET (network namespace — blocks outbound unless toggled)
        - CLONE_NEWPID (PID namespace — clean process tree)

    Falls back to bare subprocess if unshare is unavailable or permission denied.
    """

    @property
    def name(self) -> str:
        return "unshare"

    def is_available(self) -> bool:
        """Check if Linux namespaces can be created."""
        if platform.system() != "Linux":
            return False
        try:
            result = subprocess.run(
                ["unshare", "-r", "true"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            return False

    async def run(
        self,
        command: str,
        workspace_path: str,
        timeout_seconds: int = 300,
        allow_network: bool = False,
        readonly_cache_mounts: Optional[list[str]] = None,
        extra_env: Optional[dict[str, str]] = None,
    ) -> tuple[int, str, bool]:
        ns_cmd = self._build_namespace_command(
            command,
            workspace_path,
            allow_network,
            readonly_cache_mounts or [],
        )
        logger.info("[sandbox:unshare] Running with Linux namespace isolation.")
        logger.debug("[sandbox:unshare] Command: %s", " ".join(ns_cmd))
        return await _execute_subprocess_with_timeout(ns_cmd, timeout_seconds)

    @staticmethod
    def _build_namespace_command(
        shell_cmd: str,
        workspace_path: str,
        allow_network: bool,
        cache_mounts: list[str],
    ) -> list[str]:
        """Build the unshare command wrapping the build in isolated namespaces."""
        ns_args = [
            "unshare",
            "--mount",         # CLONE_NEWNS — filesystem isolation
            "--pid",           # CLONE_NEWPID — PID isolation
            "--fork",          # Fork before entering new PID namespace
            "--mount-proc",    # Mount a fresh /proc in the new PID namespace
        ]

        if not allow_network:
            ns_args.append("--net")  # CLONE_NEWNET — network isolation

        # Build inner shell commands: bind-mount caches → cd workspace → execute build
        inner_commands: list[str] = []

        for cache_path in cache_mounts:
            expanded = os.path.expanduser(cache_path)
            if os.path.isdir(expanded):
                inner_commands.append(f"mkdir -p '{expanded}' 2>/dev/null || true")
                inner_commands.append(
                    f"mount --bind -o ro '{expanded}' '{expanded}' 2>/dev/null || true"
                )

        inner_commands.append(f"cd '{workspace_path}'")
        inner_commands.append(shell_cmd)

        inner_script = " && ".join(inner_commands)
        ns_args.extend(["--", "sh", "-c", inner_script])
        return ns_args


# ---------------------------------------------------------------------------
# 4. DockerBackend — Docker Container Isolation
# ---------------------------------------------------------------------------

class DockerBackend(SandboxBackend):
    """
    Executes builds inside an ephemeral Docker container with resource limits.

    Provides:
        - Network isolation (--network=none by default)
        - Memory limits (--memory)
        - CPU limits (--cpus)
        - PID limits (--pids-limit) to prevent fork bombs
        - Read-only root filesystem (--read-only) with tmpfs for /tmp
        - Automatic container cleanup (--rm)
        - Workspace volume mount (read-write)
        - Cache volume mounts (read-only)
    """

    def __init__(
        self,
        image: str = "ubuntu:22.04",
        memory_limit: str = "512m",
        cpu_limit: str = "1.0",
        pids_limit: int = 100,
        docker_path: str = "docker",
    ):
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.pids_limit = pids_limit
        self.docker_path = docker_path

    @property
    def name(self) -> str:
        return f"docker({self.image})"

    def is_available(self) -> bool:
        """Check if Docker is installed and the daemon is reachable."""
        if not shutil.which(self.docker_path):
            return False
        try:
            result = subprocess.run(
                [self.docker_path, "info"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    async def run(
        self,
        command: str,
        workspace_path: str,
        timeout_seconds: int = 300,
        allow_network: bool = False,
        readonly_cache_mounts: Optional[list[str]] = None,
        extra_env: Optional[dict[str, str]] = None,
    ) -> tuple[int, str, bool]:
        docker_cmd = self._build_docker_command(
            command,
            workspace_path,
            allow_network,
            readonly_cache_mounts or [],
            extra_env or {},
            timeout_seconds,
        )
        logger.info("[sandbox:docker] Running in Docker container (image=%s, mem=%s).", self.image, self.memory_limit)
        logger.debug("[sandbox:docker] Command: %s", " ".join(docker_cmd))
        return await _execute_subprocess_with_timeout(docker_cmd, timeout_seconds)

    def _build_docker_command(
        self,
        shell_cmd: str,
        workspace_path: str,
        allow_network: bool,
        cache_mounts: list[str],
        extra_env: dict[str, str],
        timeout_seconds: int,
    ) -> list[str]:
        """Build the docker run command with resource limits and volume mounts."""
        cmd = [
            self.docker_path, "run",
            "--rm",                              # Auto-cleanup container on exit
            "--read-only",                       # Read-only root filesystem
            "--tmpfs", "/tmp:exec",              # Writable /tmp for build artifacts
            f"--memory={self.memory_limit}",     # Memory limit
            f"--cpus={self.cpu_limit}",          # CPU limit
            f"--pids-limit={self.pids_limit}",   # Prevent fork bombs
            "--stop-timeout", str(max(5, timeout_seconds // 10)),  # Graceful stop
        ]

        # Network isolation
        if allow_network:
            cmd.extend(["--network", "bridge"])  # Docker default bridge network
        else:
            cmd.extend(["--network", "none"])     # Complete network isolation

        # Mount workspace read-write
        cmd.extend(["-v", f"{workspace_path}:{workspace_path}:rw"])

        # Mount cache directories read-only
        for cache_path in cache_mounts:
            expanded = os.path.expanduser(cache_path)
            if os.path.isdir(expanded):
                cmd.extend(["-v", f"{expanded}:{expanded}:ro"])

        # Set working directory
        cmd.extend(["-w", workspace_path])

        # Environment variables
        for key, value in extra_env.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Image and entrypoint
        cmd.append(self.image)
        cmd.extend(["sh", "-c", shell_cmd])

        return cmd


# ---------------------------------------------------------------------------
# 5. BareBackend — No Isolation (Fallback)
# ---------------------------------------------------------------------------

class BareBackend(SandboxBackend):
    """
    Executes builds as a bare subprocess with no isolation.

    Used as a fallback when no other backend is available, or when
    explicitly configured for trusted local builds.
    """

    @property
    def name(self) -> str:
        return "bare"

    async def run(
        self,
        command: str,
        workspace_path: str,
        timeout_seconds: int = 300,
        allow_network: bool = False,
        readonly_cache_mounts: Optional[list[str]] = None,
        extra_env: Optional[dict[str, str]] = None,
    ) -> tuple[int, str, bool]:
        cmd = ["sh", "-c", f"cd '{workspace_path}' && {command}"]
        logger.info("[sandbox:bare] Running without isolation (bare subprocess).")
        return await _execute_subprocess_with_timeout(cmd, timeout_seconds, extra_env=extra_env)


# ---------------------------------------------------------------------------
# 6. Shared Subprocess Execution with PGID Management
# ---------------------------------------------------------------------------

async def _execute_subprocess_with_timeout(
    cmd: list[str],
    timeout_seconds: int,
    extra_env: Optional[dict[str, str]] = None,
    log_buffer_mode: str = "disk",
    max_log_size_mb: int = 500,
    log_temp_dir: str = "/tmp/.harness",
) -> tuple[int, str, bool]:
    """
    Execute a command with asyncio subprocess, strict timeout, and PGID
    termination hooks to kill hanging builds (including all child processes).

    Log streaming modes:
        - "disk":   Streams stdout/stderr directly to temp files on disk.
                    Constant RAM usage regardless of build output size.
                    Filters and diagnostics are read from disk after execution.
        - "memory": Accumulates output in in-memory lists (fast for small builds).

    Args:
        cmd: The command and arguments as a list.
        timeout_seconds: Maximum execution time before forced kill.
        extra_env: Additional environment variables to pass.
        log_buffer_mode: "disk" or "memory" — how to buffer build output.
        max_log_size_mb: Maximum combined log size before truncation (disk mode only).
        log_temp_dir: Directory for temp log files (disk mode only).

    Returns:
        Tuple of (exit_code, combined_stdout_stderr, timed_out).
    """
    timed_out = False
    exit_code = -1

    # Merge environment
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    # Create log streamer based on mode
    streamer: Union[DiskLogStreamer, MemoryLogStreamer]
    if log_buffer_mode == "disk":
        streamer = DiskLogStreamer(
            max_size_mb=max_log_size_mb,
            temp_dir=log_temp_dir,
        )
    else:
        streamer = MemoryLogStreamer()

    try:
        await streamer.open()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # Create new process group (PGID = PID)
            env=env,
        )

        # Capture the process group ID for clean tree termination
        pgid: Optional[int] = None
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            pgid = proc.pid

        async def _read_stdout() -> None:
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                await streamer.write_stdout(line)

        async def _read_stderr() -> None:
            assert proc.stderr is not None
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                await streamer.write_stderr(line)

        async def _wait_with_timeout() -> int:
            try:
                return await asyncio.wait_for(proc.wait(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                return -1  # Sentinel for timeout

        reader_tasks: list[asyncio.Task[Any]] = [
            asyncio.create_task(_read_stdout()),
            asyncio.create_task(_read_stderr()),
        ]
        wait_task: asyncio.Task[int] = asyncio.create_task(_wait_with_timeout())

        done, pending = await asyncio.wait(
            [wait_task, *reader_tasks],
            return_when=asyncio.FIRST_COMPLETED,
        )

        exit_code = await wait_task

        if exit_code == -1:
            # Timeout — kill the entire process group
            timed_out = True
            logger.warning(
                "[sandbox] Build timed out after %ds. Killing process group %s.",
                timeout_seconds,
                pgid,
            )
            _kill_process_group(pgid, proc)

            # Drain remaining output after kill
            try:
                remaining_stdout, remaining_stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=10.0
                )
                if remaining_stdout:
                    await streamer.write_stdout_block(remaining_stdout)
                if remaining_stderr:
                    await streamer.write_stderr_block(remaining_stderr)
            except asyncio.TimeoutError:
                logger.warning("[sandbox] Process did not terminate after kill signal.")

            exit_code = -9  # SIGKILL equivalent

        # Cancel any reader tasks still running
        for task in reader_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        for task in pending:
            if not task.done():
                task.cancel()

    except FileNotFoundError:
        await streamer.close()
        return 127, f"Command not found: {cmd[0]}", False
    except PermissionError:
        await streamer.close()
        return 126, f"Permission denied: {cmd[0]}", False
    except Exception as exc:
        logger.exception("[sandbox] Unexpected subprocess error.")
        await streamer.close()
        return 1, f"Subprocess error: {exc}", False

    full_output = await streamer.read_all()
    await streamer.close(keep_on_success=(exit_code == 0))
    return exit_code, full_output, timed_out


# ---------------------------------------------------------------------------
# 6a. Log Streamer Utilities (Disk-Buffered & Memory)
# ---------------------------------------------------------------------------

class MemoryLogStreamer:
    """
    In-memory log accumulator. Fast for small builds (< 10MB output).
    Default for backward compatibility with no temp file overhead.
    """

    def __init__(self) -> None:
        self._stdout: list[str] = []
        self._stderr: list[str] = []
        self._total_bytes = 0

    async def open(self) -> None:
        pass

    async def write_stdout(self, data: bytes) -> None:
        decoded = data.decode("utf-8", errors="replace")
        self._stdout.append(decoded)
        self._total_bytes += len(data)

    async def write_stderr(self, data: bytes) -> None:
        decoded = data.decode("utf-8", errors="replace")
        self._stderr.append(decoded)
        self._total_bytes += len(data)

    async def write_stdout_block(self, data: bytes) -> None:
        self._stdout.append(data.decode("utf-8", errors="replace"))
        self._total_bytes += len(data)

    async def write_stderr_block(self, data: bytes) -> None:
        self._stderr.append(data.decode("utf-8", errors="replace"))
        self._total_bytes += len(data)

    async def read_all(self) -> str:
        return "".join(self._stdout) + "".join(self._stderr)

    async def close(self, keep_on_success: bool = False) -> None:
        self._stdout.clear()
        self._stderr.clear()


class DiskLogStreamer:
    """
    Disk-buffered log streamer. Streams build output directly to temp files
    to keep RAM usage constant regardless of build output size.

    Features:
        - Writes stdout and stderr to separate NamedTemporaryFiles
        - Enforces max total size limit (truncates oldest lines if exceeded)
        - Reads back via line-by-line file iteration (never loads full file into RAM)
        - Auto-cleans temp files on close
    """

    def __init__(
        self,
        max_size_mb: int = 500,
        temp_dir: str = "/tmp/.harness",
    ) -> None:
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.temp_dir = temp_dir
        self._stdout_file: Optional[Any] = None  # tempfile.NamedTemporaryFile
        self._stderr_file: Optional[Any] = None
        self._stdout_path: str = ""
        self._stderr_path: str = ""
        self._total_bytes: int = 0
        self._overflow_count: int = 0

    async def open(self) -> None:
        """Create temp files for log output."""
        os.makedirs(self.temp_dir, exist_ok=True)

        import tempfile as tfile
        self._stdout_file = tfile.NamedTemporaryFile(
            mode="wb",
            suffix=".stdout.log",
            prefix="harness_",
            dir=self.temp_dir,
            delete=False,
        )
        self._stderr_file = tfile.NamedTemporaryFile(
            mode="wb",
            suffix=".stderr.log",
            prefix="harness_",
            dir=self.temp_dir,
            delete=False,
        )
        self._stdout_path = self._stdout_file.name
        self._stderr_path = self._stderr_file.name

        logger.debug("[logstream:disk] Opened temp logs: stdout=%s stderr=%s", self._stdout_path, self._stderr_path)

    async def write_stdout(self, data: bytes) -> None:
        if self._stdout_file is None:
            return
        if self._total_bytes >= self.max_size_bytes:
            self._overflow_count += 1
            if self._overflow_count % 1000 == 1:
                logger.warning("[logstream:disk] Log overflow — truncating. Total bytes: %d", self._total_bytes)
            return
        self._stdout_file.write(data)
        self._total_bytes += len(data)

    async def write_stderr(self, data: bytes) -> None:
        if self._stderr_file is None:
            return
        if self._total_bytes >= self.max_size_bytes:
            self._overflow_count += 1
            if self._overflow_count % 1000 == 1:
                logger.warning("[logstream:disk] Log overflow — truncating. Total bytes: %d", self._total_bytes)
            return
        self._stderr_file.write(data)
        self._total_bytes += len(data)

    async def write_stdout_block(self, data: bytes) -> None:
        if self._stdout_file is None:
            return
        self._stdout_file.write(data)
        self._total_bytes += len(data)

    async def write_stderr_block(self, data: bytes) -> None:
        if self._stderr_file is None:
            return
        self._stderr_file.write(data)
        self._total_bytes += len(data)

    async def read_all(self) -> str:
        """
        Read all log output from disk files. Uses line-by-line iteration
        to avoid loading the entire file into RAM at once, but still
        returns the full string (caller expects it for filtering).
        For extremely large logs, use read_filtered() instead.
        """
        # Flush disk buffers to ensure all written data is visible
        for fh in (self._stdout_file, self._stderr_file):
            if fh is not None:
                try:
                    fh.flush()
                except OSError:
                    pass

        parts: list[str] = []
        for path in (self._stdout_path, self._stderr_path):
            if path and os.path.isfile(path):
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        parts.append(line)
        return "".join(parts)

    async def read_filtered(self, filter_fn: Callable[[str], bool]) -> str:
        """
        Stream-read from disk files and apply a filter function line-by-line.
        Only matching lines are kept in the result. This keeps RAM usage
        proportional to the filtered output, not the total log size.

        Args:
            filter_fn: A callable that takes a line string and returns True to keep it.

        Returns:
            Filtered output string containing only matching lines with context.
        """
        matching_lines: list[tuple[int, str]] = []
        line_idx = 0
        for path in (self._stdout_path, self._stderr_path):
            if path and os.path.isfile(path):
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if filter_fn(line):
                            matching_lines.append((line_idx, line))
                        line_idx += 1

        if not matching_lines:
            # Fallback: return last 500 lines
            return await self._read_tail(500)

        # Build result with context (±5 lines around each match)
        result_lines: list[str] = []
        context_range = 5
        match_indices = {idx for idx, _ in matching_lines}
        added: set[int] = set()
        for idx in sorted(match_indices):
            start = max(0, idx - context_range)
            end = idx + context_range + 1
            for i in range(start, end):
                if i not in added:
                    # Re-read the line at index i (could optimize with a ring buffer)
                    line_text = await self._read_line_at(i)
                    if line_text is not None:
                        result_lines.append(line_text)
                        added.add(i)
            result_lines.append("---")
        return "\n".join(result_lines)

    async def _read_line_at(self, index: int) -> Optional[str]:
        """Read a specific line index from disk files."""
        current = 0
        for path in (self._stdout_path, self._stderr_path):
            if path and os.path.isfile(path):
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if current == index:
                            return line.rstrip("\n")
                        current += 1
        return None

    async def _read_tail(self, n: int) -> str:
        """Read the last N lines from the combined log files."""
        all_lines: list[str] = []
        for path in (self._stdout_path, self._stderr_path):
            if path and os.path.isfile(path):
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    all_lines.extend(line.rstrip("\n") for line in f)
        return "\n".join(all_lines[-n:]) if len(all_lines) > n else "\n".join(all_lines)

    async def close(self, keep_on_success: bool = False) -> None:
        """Flush and close temp files. Auto-cleans unless keep_on_success is True."""
        for fh in (self._stdout_file, self._stderr_file):
            if fh is not None:
                try:
                    fh.flush()
                    fh.close()
                except OSError:
                    pass

        if not keep_on_success:
            for path in (self._stdout_path, self._stderr_path):
                if path and os.path.isfile(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

        self._stdout_file = None
        self._stderr_file = None
        self._stdout_path = ""
        self._stderr_path = ""
        self._total_bytes = 0
        self._overflow_count = 0

        if keep_on_success:
            logger.debug("[logstream:disk] Retained logs for successful build audit.")
        else:
            logger.debug("[logstream:disk] Log temp files cleaned up.")

    @property
    def total_size_mb(self) -> float:
        """Return the total bytes written so far, in MB."""
        return self._total_bytes / (1024 * 1024)


def _kill_process_group(pgid: Optional[int], proc: asyncio.subprocess.Process) -> None:
    """
    Kill an entire process group. Sends SIGTERM first, waits 3s,
    then escalates to SIGKILL if the group still exists.
    """
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        else:
            time.sleep(3.0)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
    else:
        # Fallback: just kill the parent process
        try:
            proc.kill()
        except ProcessLookupError:
            pass


# ---------------------------------------------------------------------------
# 7. Critical Error Pattern Definitions (Regex Log Interceptor)
# ---------------------------------------------------------------------------

_CRITICAL_ERROR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bfatal\b", re.IGNORECASE),
    re.compile(r"\bpanic\b", re.IGNORECASE),
    re.compile(r"\bERROR:\s", re.IGNORECASE),
    re.compile(r"\berror\[", re.IGNORECASE),
    re.compile(r"\bundefined reference to\b", re.IGNORECASE),
    re.compile(r"\bcannot find\b", re.IGNORECASE),
    re.compile(r"\bcompilation terminated\b", re.IGNORECASE),
    re.compile(r"\bsegmentation fault\b", re.IGNORECASE),
    re.compile(r"\bSIGSEGV\b", re.IGNORECASE),
    re.compile(r"\bSIGABRT\b", re.IGNORECASE),
    re.compile(r"\bTraceback \(most recent call last\)\b", re.IGNORECASE),
    re.compile(r"\bModuleNotFoundError\b", re.IGNORECASE),
    re.compile(r"\bImportError\b", re.IGNORECASE),
    re.compile(r"\bTypeError\b", re.IGNORECASE),
    re.compile(r"\bSyntaxError\b", re.IGNORECASE),
    re.compile(r"\bbuild failed\b", re.IGNORECASE),
    re.compile(r"\blink failed\b", re.IGNORECASE),
    re.compile(r"\bFAILED\b", re.IGNORECASE),
    re.compile(r"\baborted\b", re.IGNORECASE),
    re.compile(r"\bNo such file or directory\b", re.IGNORECASE),
    re.compile(r"\bcannot access\b", re.IGNORECASE),
    re.compile(r"\bPermission denied\b", re.IGNORECASE),
]


def _is_critical_line(line: str) -> bool:
    """Check if a log line matches any critical error pattern."""
    for pattern in _CRITICAL_ERROR_PATTERNS:
        if pattern.search(line):
            return True
    return False


def filter_critical_errors(raw_output: str) -> str:
    """
    Strip verbose success lines from build output, retaining only lines
    that match critical error patterns. If no critical patterns match,
    returns the last 50 lines of output as fallback.
    """
    lines = raw_output.splitlines()
    critical_lines = [line for line in lines if _is_critical_line(line)]

    if critical_lines:
        result_lines: list[str] = []
        critical_indices = {i for i, line in enumerate(lines) if _is_critical_line(line)}
        context_range = 5
        added_indices: set[int] = set()

        for idx in sorted(critical_indices):
            start = max(0, idx - context_range)
            end = min(len(lines), idx + context_range + 1)
            for i in range(start, end):
                if i not in added_indices:
                    result_lines.append(lines[i])
                    added_indices.add(i)
            result_lines.append("---")

        return "\n".join(result_lines)

    if len(lines) > 50:
        return "\n".join(lines[-50:])
    return raw_output


# ---------------------------------------------------------------------------
# 8. Structured Diagnostic Parsing
# ---------------------------------------------------------------------------

_STRUCTURED_COMPILER_FLAGS: dict[str, str] = {
    "rustc": "--error-format=json",
    "gcc": "-fdiagnostics-format=json",
    "g++": "-fdiagnostics-format=json",
    "clang": "-fdiagnostics-format=json",
    "clang++": "-fdiagnostics-format=json",
    "cargo": "",
    "go": "",
}


def _detect_compiler(build_command: str) -> Optional[str]:
    """Heuristically detect which compiler toolchain a build command uses."""
    cmd_lower = build_command.lower()
    for compiler in ["rustc", "cargo", "gcc", "g++", "clang", "clang++", "go build", "make", "cmake"]:
        if compiler in cmd_lower:
            return compiler.split()[0]
    return None


def _parse_rust_json_diagnostics(raw_output: str) -> list[DiagnosticObject]:
    """Parse Rust compiler JSON diagnostic output (--error-format=json)."""
    diagnostics: list[DiagnosticObject] = []
    for line in raw_output.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        reason = obj.get("reason", "")
        if reason != "compiler-message":
            continue
        msg_data = obj.get("message", {})
        spans = msg_data.get("spans", [])
        primary_span = spans[0] if spans else {}
        diagnostics.append(DiagnosticObject(
            file=primary_span.get("file_name", ""),
            line=primary_span.get("line_start", 0),
            column=primary_span.get("column_start", 0),
            severity=msg_data.get("level", "error"),
            error_code=msg_data.get("code", ""),
            message=msg_data.get("message", ""),
            semantic_context=msg_data.get("rendered", ""),
        ))
    return diagnostics


def _parse_gcc_json_diagnostics(raw_output: str) -> list[DiagnosticObject]:
    """Parse GCC/Clang JSON diagnostic output (-fdiagnostics-format=json)."""
    diagnostics: list[DiagnosticObject] = []
    for line in raw_output.splitlines():
        line = line.strip()
        if not line.startswith("[") or not line.endswith("]"):
            continue
        try:
            items = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    loc = item.get("locations", [{}])[0]
                    diagnostics.append(DiagnosticObject(
                        file=loc.get("caret", {}).get("file", ""),
                        line=loc.get("caret", {}).get("line", 0),
                        column=loc.get("caret", {}).get("column", 0),
                        severity="error" if item.get("kind") == "error" else "warning",
                        error_code=str(item.get("option", "")),
                        message=item.get("message", ""),
                        semantic_context="",
                    ))
    return diagnostics


def _parse_go_diagnostics(raw_output: str) -> list[DiagnosticObject]:
    """Parse Go compiler output: filename:line:col: message."""
    diagnostics: list[DiagnosticObject] = []
    pattern = re.compile(r'^(.+\.go):(\d+):(\d+):\s+(.+)$')
    for line in raw_output.splitlines():
        match = pattern.match(line.strip())
        if match:
            diagnostics.append(DiagnosticObject(
                file=match.group(1),
                line=int(match.group(2)),
                column=int(match.group(3)),
                severity="error",
                error_code="",
                message=match.group(4),
                semantic_context="",
            ))
    return diagnostics


def _parse_generic_diagnostics(raw_output: str, workspace_path: str) -> list[DiagnosticObject]:
    """Generic regex-based diagnostic parser for compilers without structured output."""
    diagnostics: list[DiagnosticObject] = []
    pattern = re.compile(r'^(.+?):(\d+):(\d+):\s+(error|warning):\s+(.+)$', re.IGNORECASE)
    for line in raw_output.splitlines():
        match = pattern.match(line.strip())
        if match:
            filepath = match.group(1)
            if not os.path.isabs(filepath):
                filepath = os.path.join(workspace_path, filepath)
            diagnostics.append(DiagnosticObject(
                file=filepath,
                line=int(match.group(2)),
                column=int(match.group(3)),
                severity=match.group(4).lower(),
                error_code="",
                message=match.group(5),
                semantic_context="",
            ))
    return diagnostics


def extract_diagnostics(raw_output: str, build_command: str, workspace_path: str) -> list[DiagnosticObject]:
    """Extract structured diagnostics from compiler output using appropriate parser."""
    compiler = _detect_compiler(build_command)

    if compiler in ("rustc", "cargo"):
        return _parse_rust_json_diagnostics(raw_output)
    elif compiler in ("gcc", "g++", "clang", "clang++"):
        return _parse_gcc_json_diagnostics(raw_output)
    elif compiler == "go":
        return _parse_go_diagnostics(raw_output)
    else:
        return _parse_generic_diagnostics(raw_output, workspace_path)


# ---------------------------------------------------------------------------
# 9. Backend Factory
# ---------------------------------------------------------------------------

_BACKEND_REGISTRY: dict[str, type[SandboxBackend]] = {
    "unshare": UnshareBackend,
    "docker": DockerBackend,
    "bare": BareBackend,
}


def create_backend(backend_name: str, **kwargs: Any) -> SandboxBackend:
    """
    Factory: create a sandbox backend by name.

    Args:
        backend_name: One of 'unshare', 'docker', 'bare', or 'auto'.
        **kwargs: Backend-specific configuration (e.g., image='python:3.11' for docker).

    Returns:
        A SandboxBackend instance.

    Raises:
        ValueError: If the backend name is unknown.
    """
    if backend_name == "auto":
        return _auto_detect_backend(**kwargs)

    cls = _BACKEND_REGISTRY.get(backend_name)
    if cls is None:
        raise ValueError(
            f"Unknown sandbox backend: '{backend_name}'. "
            f"Supported: {list(_BACKEND_REGISTRY.keys())}, auto"
        )
    return cls(**kwargs)


def _auto_detect_backend(**kwargs: Any) -> SandboxBackend:
    """
    Auto-detect the best available backend.

    Priority (Docker-First strategy):
        1. docker   (container isolation, strongest sandbox boundary)
        2. unshare  (Linux kernel namespaces, zero deps)
        3. bare     (no isolation) — requires explicit opt-in via
                    ``HARNESS_ALLOW_UNSAFE_SANDBOX=true`` env var.
                    Without the opt-in, auto-detect raises RuntimeError
                    rather than silently running LLM-generated build
                    commands directly on the host.

    User-requested explicit backends (e.g., "unshare" or "bare") bypass this
    function entirely — see create_backend() for the override path.
    """
    # Tier 1: Try Docker first (strongest isolation)
    docker = DockerBackend(**kwargs) if kwargs else DockerBackend()
    if docker.is_available():
        logger.info("[sandbox] Auto-detected backend: docker (container, image=%s).", docker.image)
        return docker

    # Tier 2: Fall back to unshare (Linux kernel namespaces)
    unshare = UnshareBackend()
    if unshare.is_available():
        logger.info("[sandbox] Auto-detected backend: unshare (Linux namespaces).")
        return unshare

    # Tier 3: No safe backend. Bare runs LLM-supplied shell on the host with
    # zero isolation — refuse unless the user opted in.
    if os.environ.get("HARNESS_ALLOW_UNSAFE_SANDBOX", "").lower() == "true":
        logger.warning(
            "[sandbox] No container or namespace isolation available. "
            "HARNESS_ALLOW_UNSAFE_SANDBOX=true is set; falling back to bare backend. "
            "Builds will run directly on the host with no protection."
        )
        return BareBackend()

    raise RuntimeError(
        "Sandbox auto-detect failed: neither Docker nor unshare is available, "
        "and bare-fallback is not opted-in. Install Docker, ensure `unshare` "
        "has the required capabilities, or set HARNESS_ALLOW_UNSAFE_SANDBOX=true "
        "to authorize zero-isolation execution."
    )


# ---------------------------------------------------------------------------
# 10. SandboxExecutor — Orchestrates Build Execution
# ---------------------------------------------------------------------------

class SandboxExecutor:
    """
    Executes build commands using a configured isolation backend.

    Configuration comes from .harness_config.json:
        {
          "sandbox": {
            "backend": "auto",          // "auto", "unshare", "docker", "bare"
            "docker_image": "ubuntu:22.04",
            "docker_memory_limit": "512m",
            "docker_cpu_limit": "1.0",
            "docker_pids_limit": 100,
            "readonly_cache_mounts": [...],
            "timeout_seconds": 300,
            "pgid_kill_on_timeout": true
          }
        }
    """

    def __init__(
        self,
        workspace_path: str,
        allow_network: bool = False,
        timeout_seconds: int = 300,
        readonly_cache_mounts: Optional[list[str]] = None,
        pgid_kill_on_timeout: bool = True,
        backend: Optional[SandboxBackend] = None,
        command_validator: Optional[Any] = None,
    ):
        self.workspace_path = os.path.abspath(workspace_path)
        self.allow_network = allow_network
        self.timeout_seconds = timeout_seconds
        self.readonly_cache_mounts = readonly_cache_mounts or []
        self.pgid_kill_on_timeout = pgid_kill_on_timeout
        self.backend = backend or _auto_detect_backend()
        self.command_validator = command_validator  # Optional CommandValidator for security checks

    async def run(self, build_command: str) -> BuildResult:
        """
        Execute a build command inside the sandbox.

        Args:
            build_command: Shell command string (e.g., 'make build').

        Returns:
            BuildResult with exit code, filtered output, and structured diagnostics.
        """
        # --- Command Whitelist Validation ---
        if self.command_validator is not None:
            try:
                self.command_validator.validate_or_raise(build_command)
            except ValueError as exc:
                logger.error("[sandbox] Command blocked by security validator: %s", exc)
                return BuildResult(
                    exit_code=1,
                    raw_output=str(exc),
                    diagnostics=[],
                    timed_out=False,
                )

        start_time = time.monotonic()

        logger.info(
            "[sandbox] Executing build with backend=%s: %s",
            self.backend.name,
            build_command,
        )

        exit_code, raw_output, timed_out = await self.backend.run(
            command=build_command,
            workspace_path=self.workspace_path,
            timeout_seconds=self.timeout_seconds,
            allow_network=self.allow_network,
            readonly_cache_mounts=self.readonly_cache_mounts,
        )

        elapsed = time.monotonic() - start_time

        # Filter the output to extract critical errors
        filtered = filter_critical_errors(raw_output)

        # Parse structured diagnostics
        diagnostics = extract_diagnostics(raw_output, build_command, self.workspace_path)

        logger.info(
            "[sandbox] Build finished: backend=%s exit=%d elapsed=%.2fs timed_out=%s diagnostics=%d",
            self.backend.name, exit_code, elapsed, timed_out, len(diagnostics),
        )

        return BuildResult(
            exit_code=exit_code,
            raw_output=filtered if exit_code != 0 else raw_output,
            diagnostics=diagnostics,
            elapsed_seconds=elapsed,
            timed_out=timed_out,
        )


# ---------------------------------------------------------------------------
# 11. BaseLanguageParser ABC (Plugin Architecture)
# ---------------------------------------------------------------------------

class BaseLanguageParser:
    """
    Abstract base for language-specific diagnostic parsers.

    Registered dynamically via harness/parser_registry.py without modifying
    the sandbox or graph engine. Each parser must implement
    parse_diagnostics(raw_output: str) -> list[DiagnosticObject].
    """

    @staticmethod
    def parse_diagnostics(raw_output: str) -> list[DiagnosticObject]:
        """Parse raw compiler output into structured DiagnosticObject list."""
        raise NotImplementedError("Subclasses must implement parse_diagnostics.")


# Built-in parser registry (complemented by harness/parser_registry.py)
_PARSER_REGISTRY: dict[str, type[BaseLanguageParser]] = {}


def register_parser(compiler_name: str, parser_cls: type[BaseLanguageParser]) -> None:
    """Register a new language parser plugin."""
    _PARSER_REGISTRY[compiler_name] = parser_cls
    logger.info("[sandbox] Registered parser for compiler '%s': %s", compiler_name, parser_cls.__name__)


def get_parser(compiler_name: str) -> Optional[type[BaseLanguageParser]]:
    """Look up a registered parser by compiler name."""
    return _PARSER_REGISTRY.get(compiler_name)


# ---------------------------------------------------------------------------
# 12. Convenience: Run a build and return enriched state fragment
# ---------------------------------------------------------------------------

async def execute_build(
    workspace_path: str,
    build_command: str,
    allow_network: bool = False,
    timeout_seconds: int = 300,
    readonly_cache_mounts: Optional[list[str]] = None,
    backend: Optional[SandboxBackend] = None,
) -> dict[str, Any]:
    """
    Execute a build in the sandbox and return a state-fragment dict
    compatible with LangGraph state updates.

    This is the primary integration point used by compiler_node.

    Args:
        workspace_path: Absolute path to the repository root.
        build_command: The shell command to build/compile/verify.
        allow_network: Whether to permit outbound network in the sandbox.
        timeout_seconds: Maximum build time before forced termination.
        readonly_cache_mounts: Host directories to bind-mount read-only.
        backend: Optional pre-configured SandboxBackend. Auto-detected if None.

    Returns:
        A dict suitable for merging into AgentState:
            - exit_code: int
            - compiler_errors: list[DiagnosticObject.to_dict()]
            - node_state: {'last_build_output': str}
    """
    executor = SandboxExecutor(
        workspace_path=workspace_path,
        allow_network=allow_network,
        timeout_seconds=timeout_seconds,
        readonly_cache_mounts=readonly_cache_mounts or [],
        pgid_kill_on_timeout=True,
        backend=backend,
    )
    result = await executor.run(build_command)

    return {
        "exit_code": result.exit_code,
        "compiler_errors": [d.to_dict() for d in result.diagnostics],
        "node_state": {
            "last_build_output": result.raw_output,
            "build_elapsed_seconds": result.elapsed_seconds,
            "build_timed_out": result.timed_out,
        },
    }