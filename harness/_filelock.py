"""Cross-platform file locking helper.

Dispatches to ``fcntl.flock`` on POSIX and ``msvcrt.locking`` on
Windows. Both are stdlib — no new dependency.

POSIX semantics are preserved byte-for-byte: ``lock_exclusive_*`` is a
direct passthrough to ``fcntl.flock`` with the matching blocking mode,
``unlock`` is ``fcntl.LOCK_UN``. Linux/macOS callers see no behavioural
change after the migration.

Windows behaviour matches the POSIX call-site contracts as closely as
the stdlib allows:

  - ``lock_exclusive_nonblocking(fh)`` raises ``BlockingIOError`` if the
    lock is already held — same exception class the POSIX call sites
    already catch.
  - ``lock_exclusive_blocking(fh)`` retries until acquired or until the
    call site decides it has waited long enough (msvcrt's per-call
    block is only ~10s; we loop). Raises ``OSError`` on irrecoverable
    failure, matching POSIX.
  - ``unlock(fh)`` is best-effort and swallows ``OSError`` to match the
    POSIX call sites that intentionally tolerate "already-released"
    races.

Notes on Windows:

  - ``msvcrt.locking`` locks a *byte range* relative to the file pointer.
    We always lock byte 0 (seeking there first, restoring the pointer
    after) so the lock is a whole-file mutex semantically equivalent to
    ``fcntl.flock``.
  - The lock is mandatory on Windows (Windows kernel enforces it),
    whereas POSIX ``flock`` is advisory. This is usually what callers
    want — concurrent harness processes can't accidentally bypass it.
"""

from __future__ import annotations

import logging
import os
import time
from typing import IO, Any


logger = logging.getLogger(__name__)


try:
    import fcntl  # type: ignore[import-not-found, unused-ignore]
    _HAVE_FCNTL = True
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]
    _HAVE_FCNTL = False

try:
    import msvcrt  # type: ignore[import-not-found, unused-ignore]
    _HAVE_MSVCRT = True
except ImportError:  # POSIX
    msvcrt = None  # type: ignore[assignment]
    _HAVE_MSVCRT = False


# Whether the host has at least one working lock backend.
LOCKING_AVAILABLE = _HAVE_FCNTL or _HAVE_MSVCRT


def _msvcrt_lock(fh: IO[Any], mode: int) -> None:
    """Lock byte 0 of the file via msvcrt, preserving the file pointer."""
    fd = fh.fileno()
    pos = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, mode, 1)  # type: ignore[attr-defined, unused-ignore]
    finally:
        os.lseek(fd, pos, os.SEEK_SET)


def lock_exclusive_nonblocking(fh: IO[Any]) -> None:
    """Acquire an exclusive whole-file lock; raise BlockingIOError if held.

    Mirrors ``fcntl.flock(fh, LOCK_EX | LOCK_NB)``.
    """
    if _HAVE_FCNTL:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    if _HAVE_MSVCRT:
        try:
            _msvcrt_lock(fh, msvcrt.LK_NBLCK)  # type: ignore[attr-defined, unused-ignore]
        except OSError as exc:
            # msvcrt raises OSError(EACCES) when the lock is held; the
            # POSIX call sites expect BlockingIOError.
            raise BlockingIOError(str(exc)) from exc
        return
    raise OSError("no file-locking backend available on this platform")


def lock_exclusive_blocking(fh: IO[Any], *, timeout_seconds: float = 30.0) -> None:
    """Acquire an exclusive whole-file lock, blocking until acquired.

    Mirrors ``fcntl.flock(fh, LOCK_EX)``. On Windows ``msvcrt.locking``
    blocks for ~10s per call before raising, so we loop until the
    overall timeout elapses. Raises ``OSError`` on irrecoverable
    failure, matching the POSIX call-site exception contract.
    """
    if _HAVE_FCNTL:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        return
    if _HAVE_MSVCRT:
        deadline = time.monotonic() + timeout_seconds
        last_exc: OSError | None = None
        while time.monotonic() < deadline:
            try:
                _msvcrt_lock(fh, msvcrt.LK_LOCK)  # type: ignore[attr-defined, unused-ignore]
                return
            except OSError as exc:
                last_exc = exc
                # Brief backoff before retry; LK_LOCK already blocked
                # internally so we don't need a long sleep.
                time.sleep(0.05)
        raise OSError(
            f"could not acquire lock within {timeout_seconds:.1f}s: {last_exc}"
        )
    raise OSError("no file-locking backend available on this platform")


def unlock(fh: IO[Any]) -> None:
    """Release the lock on ``fh``. Best-effort; swallows OSError to match
    the POSIX call sites that tolerate "already-released" races on shutdown.
    """
    if _HAVE_FCNTL:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        return
    if _HAVE_MSVCRT:
        try:
            _msvcrt_lock(fh, msvcrt.LK_UNLCK)  # type: ignore[attr-defined, unused-ignore]
        except OSError:
            pass
        return
    # No backend — nothing to release.
