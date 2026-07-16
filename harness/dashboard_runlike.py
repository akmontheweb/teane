"""Subcommand-aware spawn + form helpers for the Run Harness page.

The legacy ``spawn_harness_run`` in :mod:`harness.dashboard` hardcoded
``argv = [harness_binary, "run", ...]``. The ``run`` subparser has since
been removed from :mod:`harness.cli` (operators pick
``build``/``patch``/``deploy``/``test``/``audit`` based on intent), so
the historical spawner emits an argv the CLI will reject. This module
replaces it with a subcommand-aware spawner that:

- Takes a ``subcommand`` parameter ∈ {``build``, ``patch``, ``deploy``,
  ``test``, ``audit``}.
- Uses ``-w`` (the canonical workspace flag in
  ``_add_runlike_common`` at :mod:`harness.cli`), not the dead ``-r``.
- Drops ``--prompt`` for ``audit`` (the audit subparser only accepts
  ``--workspace``).
- Tags the audit-log row with the subcommand
  (``action=f"run_{subcommand}"``) so operators can filter the audit
  trail by what kind of run fired.

The HITL-webhook env + log-tail + process-registry wiring matches
``spawn_harness_resume`` so the dashboard tracks every subcommand the
same way it tracked the historical ``teane run``.

No HTML lives here — the Run page renderer in
:mod:`harness.dashboard` consumes the data layer this module exposes
once Phase 2.2 lands.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

from harness import _platform
from harness.web_state import (
    WebProcess,
    append_audit,
    get_process_registry,
)

logger = logging.getLogger(__name__)


_VALID_SUBCOMMANDS: frozenset[str] = frozenset({
    "build", "patch", "deploy", "test", "audit",
})


def is_valid_subcommand(subcommand: str) -> bool:
    """Whether ``subcommand`` is one this module's spawner accepts."""
    return subcommand in _VALID_SUBCOMMANDS


def spawn_harness_subcommand(
    cfg: Any,
    *,
    subcommand: str,
    workspace: str,
    prompt: str = "",
    extra_args: Optional[list[str]] = None,
    harness_binary: str = "harness",
) -> WebProcess:
    """Spawn ``teane <subcommand>`` as a subprocess, register it, and
    return the :class:`WebProcess` handle.

    Builds argv as::

        [harness_binary, subcommand, "-w", workspace, "-p", prompt,
         "--session-id", session_id, *extra_args]

    ``audit`` skips the ``-p`` pair (the audit subparser at
    ``cli.py:9080`` only accepts ``--workspace``); pass an empty
    ``prompt`` for audit.

    Sets ``HARNESS_HITL_WEBHOOK_URL`` so the harness's HttpChannel
    POSTs HITL prompts back to this dashboard, mirroring the legacy
    spawner.

    Resolves ``harness_binary`` via :func:`shutil.which` when not
    absolute so a same-host attacker who plants ``./harness`` in cwd
    or a writable PATH entry can't shadow the real binary (audit §3.6).
    """
    import shutil as _shutil
    import subprocess as _sub
    import uuid as _uuid

    if not is_valid_subcommand(subcommand):
        raise ValueError(
            f"unknown subcommand {subcommand!r}; "
            f"expected one of {sorted(_VALID_SUBCOMMANDS)}"
        )
    if not workspace:
        raise ValueError("workspace is required")

    if not os.path.isabs(harness_binary):
        resolved = _shutil.which(harness_binary)
        if resolved and os.path.isabs(resolved):
            harness_binary = resolved

    # uuid4, not uuid7: a truncated uuid7 is mostly timestamp bits, so
    # short ids minted close together would collide.
    session_id = f"web-{_uuid.uuid4().hex[:12]}"
    log_dir = os.path.expanduser(cfg.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{session_id}.jsonl")
    # Pre-create the log file so the SSE stream's tail can start before
    # the harness has written anything.
    open(log_path, "a", encoding="utf-8").close()

    argv: list[str] = [
        harness_binary, subcommand,
        "-w", workspace,
    ]
    if subcommand != "audit":
        # The audit subparser accepts only --workspace; passing -p
        # would argparse-error before the audit can run.
        argv.extend(["-p", prompt or ""])
    argv.extend(["--session-id", session_id])
    argv.extend(list(extra_args or []))

    env = dict(os.environ)
    env["HARNESS_HITL_WEBHOOK_URL"] = (
        f"http://{cfg.host}:{cfg.port}/hitl/webhook?session={session_id}"
    )
    if getattr(cfg, "hitl_webhook_secret", ""):
        env["HARNESS_HITL_WEBHOOK_SECRET"] = cfg.hitl_webhook_secret
    # Keep the harness's HttpChannel timeout >= the dashboard's
    # operator-wait ceiling; the +30s buffer lets the dashboard
    # return 504 first instead of the harness aborting mid-prompt.
    env.setdefault(
        "HARNESS_HITL_WEBHOOK_TIMEOUT",
        str(float(getattr(cfg, "hitl_webhook_timeout_seconds", 600.0) or 600.0) + 30.0),
    )

    # Open the stdout sink, hand the FD to Popen (which dup's it into
    # the child), then close the parent's copy so the dashboard process
    # doesn't leak one FD per spawned run.
    stdout_fh = open(log_path + ".stdout", "ab")
    proc_ok = False
    try:
        try:
            proc = _sub.Popen(
                argv,
                stdout=stdout_fh,
                stderr=_sub.STDOUT,
                env=env,
                **_platform.new_process_group_kwargs(),
            )
            proc_ok = True
        finally:
            stdout_fh.close()
    finally:
        if not proc_ok:
            # Popen raised — drop the empty stdout file we created so
            # the log dir doesn't accumulate zero-byte detritus
            # (audit §2.11).
            try:
                os.unlink(log_path + ".stdout")
            except OSError:
                pass

    # Capture pgid immediately after spawn so the cancel path can
    # target the original process group even if the kernel later
    # recycles the pid to an unrelated process (audit §1.2). Under
    # start_new_session the child is the leader of its own group so
    # pgid == pid here.
    spawn_pgid: Optional[int] = None
    if hasattr(os, "getpgid"):
        try:
            spawn_pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            spawn_pgid = proc.pid

    wp = WebProcess(
        session_id=session_id, pid=proc.pid, argv=argv,
        log_path=log_path, workspace_path=workspace, prompt=prompt,
        popen=proc, pgid=spawn_pgid,
        # The watcher thread below will record the real exit code via
        # mark_terminated; flag the entry so _prune_dead_locked doesn't
        # race to mark exit_code=-1 (audit §2.15).
        watcher_pending=True,
    )
    get_process_registry().register(wp)

    db_path = cfg.web_db_path
    audit_action = f"run_{subcommand}"

    def _watch() -> None:
        try:
            ec = proc.wait()
        except Exception:  # noqa: BLE001
            ec = -1
        get_process_registry().mark_terminated(session_id, int(ec or 0))
        try:
            append_audit(
                db_path=db_path, action="run_exit",
                target=session_id, detail=f"exit_code={ec}",
            )
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(
        target=_watch, daemon=True, name=f"web-{subcommand}-{session_id}",
    ).start()
    try:
        append_audit(
            db_path=db_path, action=audit_action,
            target=session_id, detail=f"argv={' '.join(argv)}",
        )
    except Exception:  # noqa: BLE001
        pass
    return wp
