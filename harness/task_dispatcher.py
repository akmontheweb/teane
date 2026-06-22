"""
Task dispatcher for running build and test commands in sandboxed or local environments.

Provides asynchronous and synchronous execution of shell commands with timeout,
concurrency control, and structured error handling.
"""

import asyncio
import logging
import os
import subprocess
from typing import Dict, List, Optional, Union

from harness import _platform

logger = logging.getLogger(__name__)


class TaskError(Exception):
    """Raised when a dispatched task fails or times out.

    Attributes:
        message: Human-readable error description.
        returncode: Exit code of the failed command (if available).
        stdout: Captured standard output of the command.
        stderr: Captured standard error of the command.
    """

    def __init__(
        self,
        message: str,
        returncode: Optional[int] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TaskDispatcher:
    """Dispatches shell commands asynchronously with timeout and environment control.

    Supports both single task execution and parallel task execution with a
    configurable concurrency limit.

    Args:
        work_dir: Working directory for all dispatched commands.
        timeout: Default execution timeout in seconds.
        env: Environment variables to use. If not given, inherits the current
            environment.
    """

    def __init__(
        self,
        work_dir: Optional[str] = None,
        timeout: int = 300,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.work_dir = work_dir or os.getcwd()
        self.timeout = timeout
        self.env = env or os.environ.copy()

    async def run_task(
        self, task: Union[str, List[str]]
    ) -> subprocess.CompletedProcess[bytes]:
        """Run a single task asynchronously.

        Args:
            task: Command to execute, either a string (passed to a shell) or a
                list of arguments.

        Returns:
            CompletedProcess with stdout/stderr as bytes.

        Raises:
            TaskError: If the command exits with a non-zero code or times out.
        """
        cmd_str = task if isinstance(task, str) else " ".join(task)

        # Build the execution arguments: list form avoids shell injection when using list input.
        # String tasks go through a shell; _platform.shell_argv picks ``sh -c`` on POSIX
        # and ``sh -c`` (if Git Bash on PATH) or ``cmd /c`` on Windows.
        if isinstance(task, list):
            args = task
        else:
            args = _platform.shell_argv(task)

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=self.work_dir,
                env=self.env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise TaskError(
                    f"Task timed out after {self.timeout}s: {cmd_str}"
                )

            if proc.returncode != 0:
                raise TaskError(
                    f"Task failed (exit {proc.returncode}): {cmd_str}",
                    returncode=proc.returncode,
                    stdout=stdout.decode("utf-8", errors="replace") if stdout else "",
                    stderr=stderr.decode("utf-8", errors="replace") if stderr else "",
                )

            return subprocess.CompletedProcess(
                args=cmd_str,
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        except Exception as exc:
            if not isinstance(exc, TaskError):
                raise TaskError(str(exc)) from exc
            raise

    async def run_parallel(
        self, tasks: List[Union[str, List[str]]], max_concurrent: int = 3
    ) -> List[subprocess.CompletedProcess[bytes]]:
        """Run multiple tasks concurrently with a bounded parallelism.

        Args:
            tasks: Iterable of commands (strings or argument lists).
            max_concurrent: Maximum number of tasks that may run at the same time.

        Returns:
            List of CompletedProcess objects in the same order as the input tasks.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _run_with_semaphore(
            task: Union[str, List[str]],
        ) -> subprocess.CompletedProcess[bytes]:
            async with semaphore:
                return await self.run_task(task)

        results = await asyncio.gather(
            *(_run_with_semaphore(t) for t in tasks), return_exceptions=True
        )

        # Re-raise first encountered TaskError to preserve behaviour, while collecting successes.
        exceptions: list[tuple[int, BaseException]] = []
        for i, r in enumerate(results):
            if isinstance(r, BaseException):
                # Keep the original traceback by wrapping
                exceptions.append((i, r))
        if exceptions:
            # Raise the first exception; the caller can inspect others via exception chain if needed.
            first_idx, first_exc = exceptions[0]
            raise TaskError(
                f"Task {first_idx} failed: {first_exc}"
            ) from first_exc
        return [r for r in results if isinstance(r, subprocess.CompletedProcess)]

    def run_task_sync(
        self, task: Union[str, List[str]]
    ) -> subprocess.CompletedProcess[bytes]:
        """Synchronous wrapper for :meth:`run_task`.

        Suitable for simple scripts or situations where an event loop does not
        already exist.

        Args:
            task: Command string or list of arguments.

        Returns:
            CompletedProcess with captured output.
        """
        return asyncio.run(self.run_task(task))


__all__ = ["TaskDispatcher", "TaskError"]
