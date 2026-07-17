"""Opt-in trajectory-level best-of-N for `teane build` / `teane patch`.

Wires the :mod:`harness.best_of_n` primitive to real solve trajectories. Each
variant runs as an **independent `teane` subprocess in its own git worktree** —
subprocess isolation sidesteps the process-wide singletons the in-process graph
relies on (command validator, redactor, LSP pool, active-session id), so N
trajectories can run without interfering. The winner's diff is then applied
back to the operator's workspace.

Enable via the ``best_of_n`` config section (``enabled: true``, ``n``,
``strategy``, ``max_concurrency``) or a ``--best-of N`` CLI flag. Off by
default — a single trajectory runs exactly as before.

Testability boundary: the config parsing, worktree lifecycle, trajectory-result
construction, and winner-diff application are unit-tested. The actual N-way LLM
solve is exercised only end-to-end (it needs a live provider), so the subprocess
runner and the git plumbing are injected as seams the tests substitute.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import subprocess
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from harness.best_of_n import (
    SELECT_FIRST_SUCCESS,
    TrajectoryResult,
    run_best_of_n,
)

logger = logging.getLogger("harness.best_of_n_runner")

_VALID_STRATEGIES = frozenset({"first_success", "fewest_changes", "voted"})


@dataclass
class BestOfNConfig:
    """The ``best_of_n`` config section (all optional; safe defaults)."""

    enabled: bool = False
    n: int = 1
    strategy: str = SELECT_FIRST_SUCCESS
    max_concurrency: int = 3
    # How variants are diversified. "temperature" bumps sampling temperature
    # per variant; "model" rotates through model_routing fallbacks; "none"
    # relies on nondeterminism alone.
    diversity_mode: str = "temperature"
    per_variant_budget_usd: Optional[float] = None

    @classmethod
    def from_config(cls, config: Optional[dict[str, Any]]) -> "BestOfNConfig":
        sec: dict[str, Any] = {}
        if isinstance(config, dict):
            raw = config.get("best_of_n")
            if isinstance(raw, dict):
                sec = raw

        def _int(k: str, d: int) -> int:
            try:
                return int(sec.get(k, d))
            except (TypeError, ValueError):
                return d

        strategy = str(sec.get("strategy", SELECT_FIRST_SUCCESS))
        if strategy not in _VALID_STRATEGIES:
            strategy = SELECT_FIRST_SUCCESS
        budget = sec.get("per_variant_budget_usd")
        try:
            budget = float(budget) if budget is not None else None
        except (TypeError, ValueError):
            budget = None
        return cls(
            enabled=bool(sec.get("enabled", False)),
            n=max(1, _int("n", 1)),
            strategy=strategy,
            max_concurrency=max(1, _int("max_concurrency", 3)),
            diversity_mode=str(sec.get("diversity_mode", "temperature")),
            per_variant_budget_usd=budget,
        )

    def is_active(self) -> bool:
        """Best-of-N only kicks in when explicitly enabled AND n > 1 —
        otherwise the caller runs its normal single trajectory."""
        return self.enabled and self.n > 1


# Seams (injected in tests). A runner executes one variant's solve and returns
# (exit_code, changed_files, lines_changed, cost_usd); a diff-applier moves the
# winning worktree's changes onto the main workspace.
VariantRunner = Callable[[int, str, "BestOfNConfig"], Awaitable[tuple[int, int, int, float]]]
DiffApplier = Callable[[str, str], bool]  # (winner_worktree, main_workspace) -> ok


def _git(args: list[str], cwd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", cwd, *args], capture_output=True,
                          text=True, timeout=timeout)


def _worktree_path(workspace: str, session: str, variant_id: int) -> str:
    base = os.path.join(os.path.dirname(os.path.abspath(workspace)),
                        f".teane_bon_{session[:8]}")
    return os.path.join(base, f"v{variant_id}")


def make_worktree(workspace: str, session: str, variant_id: int) -> str:
    """Create an isolated git worktree for a variant, branched from HEAD.
    Returns the worktree path. Raises on git failure."""
    wt = _worktree_path(workspace, session, variant_id)
    os.makedirs(os.path.dirname(wt), exist_ok=True)
    branch = f"agent/bon-{session[:8]}-v{variant_id}"
    # -B (not -b): reset the branch if it already exists. A branch leaked
    # by an earlier crashed run — or a reused session id — otherwise makes
    # ``worktree add -b`` fail outright.
    r = _git(["worktree", "add", "-f", "-B", branch, wt, "HEAD"], workspace)
    if r.returncode != 0:
        raise RuntimeError(f"worktree add failed for v{variant_id}: {r.stderr.strip()[:200]}")
    return wt


_BON_DIR_RE = re.compile(r"^\.teane_bon_([A-Za-z0-9_-]{1,8})$")
_BON_VARIANT_RE = re.compile(r"^v\d+$")


def remove_worktree(workspace: str, wt: str) -> None:
    """Best-effort worktree teardown (never raises).

    ``git worktree remove`` does NOT delete the branch the worktree was
    created with, and the ``.teane_bon_<sess>`` parent dir would otherwise
    accumulate — so both are swept here (branch name re-derived from the
    path shape ``.teane_bon_<sess8>/v<N>`` that :func:`_worktree_path`
    produces; unknown shapes just skip the sweep)."""
    try:
        _git(["worktree", "remove", "--force", wt], workspace)
        parent = os.path.dirname(wt)
        dir_m = _BON_DIR_RE.match(os.path.basename(parent))
        vid = os.path.basename(wt)
        if dir_m and _BON_VARIANT_RE.match(vid):
            _git(["branch", "-D", f"agent/bon-{dir_m.group(1)}-{vid}"],
                 workspace)
        try:
            os.rmdir(parent)  # only succeeds once the last variant is gone
        except OSError:
            pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("[best_of_n] worktree remove failed for %s: %s", wt, exc)


def _base_commit(workspace: str) -> Optional[str]:
    """The MAIN workspace's HEAD sha — the commit every variant worktree was
    branched from. Diffs must run against THIS, never against the worktree's
    own ``HEAD``: with ``agile_defaults.commit_on_story=true`` the child
    ``teane`` process commits inside the worktree after every green story,
    advancing the worktree branch — a winner whose work is fully committed
    then diffs EMPTY against its own HEAD and the run reports success while
    landing nothing. The main workspace's HEAD does not move during a
    best-of-N run (only the children commit, each in its own worktree), so it
    is the stable branch point. None on git failure — callers fail CLOSED.
    """
    try:
        r = _git(["rev-parse", "HEAD"], workspace)
    except Exception as exc:  # noqa: BLE001 — treated as resolution failure
        logger.warning("[best_of_n] rev-parse HEAD failed in %s: %s", workspace, exc)
        return None
    if r.returncode != 0:
        logger.warning("[best_of_n] rev-parse HEAD failed in %s: %s",
                       workspace, (r.stderr or "").strip()[:200])
        return None
    sha = (r.stdout or "").strip()
    return sha or None


def _diff_stat(wt: str, base: Optional[str]) -> tuple[int, int]:
    """(changed_files, lines_changed) for a worktree vs the branch-point
    commit ``base`` (see :func:`_base_commit`). (0,0) on error."""
    if not base:
        return (0, 0)
    r = _git(["diff", "--numstat", base], wt)
    if r.returncode != 0:
        return (0, 0)
    files = lines = 0
    for ln in (r.stdout or "").splitlines():
        parts = ln.split("\t")
        if len(parts) >= 2:
            files += 1
            for tok in parts[:2]:
                if tok.isdigit():
                    lines += int(tok)
    return (files, lines)


def apply_winner_diff(winner_wt: str, main_workspace: str) -> bool:
    """Apply the winning worktree's full delta — committed AND uncommitted —
    onto the main workspace. Returns True on clean apply.

    The delta is computed against the main workspace's HEAD (the commit the
    worktree was branched from), not the worktree's own HEAD, which
    ``commit_on_story`` advances. Base-resolution failure returns False
    (fail closed) rather than risking a silent no-op "success"."""
    base = _base_commit(main_workspace)
    if base is None:
        return False
    d = _git(["diff", base], winner_wt)
    if d.returncode != 0:
        return False
    patch = d.stdout
    if not patch.strip():
        return True  # winner made no changes — nothing to apply
    proc = subprocess.run(["git", "-C", main_workspace, "apply", "--whitespace=nowarn"],
                          input=patch, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        logger.warning("[best_of_n] winner diff did not apply cleanly: %s",
                       proc.stderr.strip()[:200])
        return False
    return True


def _save_winner_patch(winner_wt: str, main_workspace: str,
                       session: str) -> Optional[str]:
    """Write the winning worktree's full delta to a rescue file under
    ``~/.harness/best_of_n/`` and return its path (None on any failure —
    best-effort, never raises). Called only when the apply onto the main
    workspace failed and the worktrees are about to be force-removed."""
    try:
        base = _base_commit(main_workspace)
        if base is None:
            return None
        d = _git(["diff", base], winner_wt)
        if d.returncode != 0 or not (d.stdout or "").strip():
            return None
        rescue_dir = os.path.expanduser("~/.harness/best_of_n")
        os.makedirs(rescue_dir, exist_ok=True)
        path = os.path.join(rescue_dir, f"{session[:8]}-winner.patch")
        with open(path, "w", encoding="utf-8") as f:
            f.write(d.stdout)
        return path
    except Exception as exc:  # noqa: BLE001 — rescue is best-effort
        logger.debug("[best_of_n] winner-patch rescue failed: %s", exc)
        return None


def make_subprocess_variant_runner(
    base_argv: list[str],
    *,
    timeout_s: int = 3600,
    env: Optional[dict[str, str]] = None,
) -> VariantRunner:
    """Return a :data:`VariantRunner` that runs ``teane`` in the variant's
    worktree as a subprocess.

    ``base_argv`` is the base command minus the workspace (e.g.
    ``["teane", "build"]``); the runner points ``-w`` at the worktree.
    ``TEANE_BEST_OF_N_CHILD=1`` is exported so the child never re-enters
    best-of-N (no fork bomb). ``per_variant_budget_usd`` (when set) is
    passed to each child as ``--budget`` so the cap is enforced by the
    child's own budget gateway — a true per-variant ceiling. The runner
    itself returns 0 for cost (per-variant spend lives in each child's own
    session metrics) and lets the git diffstat supply the change counts.
    Meaningful diversity across variants relies on sampling nondeterminism
    today; per-variant model/temperature routing is a forward-looking
    ``diversity_mode`` seam.
    """

    async def runner(variant_id: int, wt: str, cfg: "BestOfNConfig") -> tuple[int, int, int, float]:
        cmd = [*base_argv, "-w", wt]
        if cfg.per_variant_budget_usd:
            cmd.extend(["--budget", str(cfg.per_variant_budget_usd)])
        child_env = dict(env if env is not None else os.environ)
        child_env["TEANE_BEST_OF_N_CHILD"] = "1"
        # Force every HITL gate to auto-approve in the child. The gates'
        # tty fallback checks sys.stdin.isatty(): children spawned from an
        # interactive terminal INHERIT that tty, so with default gate
        # config all N variants would block on input() reading the shared
        # terminal (menus written to the unread stdout pipe) and die at
        # the timeout — an hour of silence, then "no trajectory produced
        # an applicable green build". stdin=DEVNULL breaks the tty
        # inheritance; the env var pins auto-approve even if a future
        # gate forgets the isatty check.
        child_env["HARNESS_AUTO_APPROVE"] = "true"
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=wt, env=child_env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                # Own process group, so the timeout path can reap the WHOLE
                # tree — the child teane spawns sandbox/MCP grandchildren
                # that a bare proc.kill() leaves running to race the
                # worktree force-remove below. POSIX-only kwarg; a no-op
                # on Windows (where the killpg path falls back anyway).
                start_new_session=True,
            )
        except Exception as exc:  # noqa: BLE001 — spawn failure → failed variant
            logger.warning("[best_of_n] variant %d spawn failed: %s", variant_id, exc)
            return (1, 0, 0, 0.0)
        try:
            await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(proc.pid, signal.SIGKILL)
                else:  # Windows: no process groups — direct kill only
                    proc.kill()
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            # Reap so the kill is complete before the caller tears down
            # the worktree the (former) grandchildren were writing to.
            try:
                await asyncio.wait_for(proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                pass
            logger.warning("[best_of_n] variant %d timed out after %ds", variant_id, timeout_s)
            return (124, 0, 0, 0.0)
        return (proc.returncode if proc.returncode is not None else 1, 0, 0, 0.0)

    return runner


def is_best_of_n_child() -> bool:
    """True inside a best-of-N child subprocess — the guard that stops a run
    from recursively fanning out. Callers skip the best-of-N branch when set."""
    return os.environ.get("TEANE_BEST_OF_N_CHILD") == "1"


def reconstruct_child_argv(
    argv: list[str],
    *,
    strip_value_flags: tuple[str, ...] = ("-w", "--workspace", "--best-of"),
) -> list[str]:
    """Rebuild the base command each child variant should run, from the
    parent's ``argv``.

    Drops the workspace flag (the runner re-points ``-w`` at each worktree)
    and the ``--best-of`` flag (so a child never re-fans-out), each together
    with its value. Also drops ``--best-of=N`` / ``-w=…`` single-token forms.
    Pure and side-effect-free so it is unit-testable without spawning
    anything.
    """
    out: list[str] = []
    skip_next = False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        # single-token "--flag=value"
        if any(tok == f or tok.startswith(f + "=") for f in strip_value_flags):
            if "=" not in tok:
                skip_next = True  # value is the next token
            continue
        out.append(tok)
    return out


async def run_best_of_n_build(
    workspace: str,
    session: str,
    cfg: BestOfNConfig,
    *,
    variant_runner: VariantRunner,
    diff_applier: DiffApplier = apply_winner_diff,
) -> tuple[Optional[TrajectoryResult], list[TrajectoryResult]]:
    """Run N trajectories in worktrees, select a winner, apply it.

    ``variant_runner`` runs one variant's solve inside its worktree and returns
    ``(exit_code, changed_files, lines_changed, cost_usd)``; it is injected so
    tests can substitute the (LLM-driven, un-runnable-here) solve. On a clean
    run the winner's diff is applied to ``workspace`` via ``diff_applier``.

    Returns ``(winner, all_results)``; winner is None when no variant reached a
    green build or the winner's diff failed to apply.
    """
    worktrees: dict[int, str] = {}
    # Branch point every variant is created from — captured ONCE, up front,
    # so diffstats and the winner apply measure against the same commit even
    # if a child's commit_on_story advances its worktree branch.
    base = _base_commit(workspace)

    async def solve(variant_id: int) -> TrajectoryResult:
        wt = make_worktree(workspace, session, variant_id)
        worktrees[variant_id] = wt
        exit_code, changed, lines, cost = await variant_runner(variant_id, wt, cfg)
        # Prefer the runner's counts; fall back to a git diffstat of the worktree.
        if changed <= 0:
            changed, lines = _diff_stat(wt, base)
        return TrajectoryResult(
            variant_id=variant_id, label=f"v{variant_id}",
            compiled_ok=(exit_code == 0), changed_files=changed,
            lines_changed=lines, cost_usd=cost, payload=wt,
        )

    try:
        winner, results = await run_best_of_n(
            solve, cfg.n, strategy=cfg.strategy,
            max_concurrency=cfg.max_concurrency,
            budget_usd=(cfg.per_variant_budget_usd * cfg.n
                        if cfg.per_variant_budget_usd else None),
        )
        if winner is not None and isinstance(winner.payload, str):
            if not diff_applier(winner.payload, workspace):
                # The ``finally`` below force-removes every worktree —
                # including the green one — and with commit_on_story=false
                # the winning diff exists NOWHERE else. Save it to a rescue
                # file before teardown so a failed apply (typical cause: a
                # dirty operator workspace) degrades to "apply this patch
                # by hand", not to an unrecoverable loss of the run's work.
                rescue = _save_winner_patch(winner.payload, workspace, session)
                logger.warning(
                    "[best_of_n] winner v%d diff failed to apply — no "
                    "changes landed.%s", winner.variant_id,
                    f" Winner's diff saved to {rescue} — inspect and "
                    f"`git apply` it manually." if rescue else "",
                )
                winner = None
        return winner, results
    finally:
        for wt in worktrees.values():
            remove_worktree(workspace, wt)
