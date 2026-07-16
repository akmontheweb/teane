"""
CLI entry point, subcommand routing, HITL interactive menu loop, and configuration discovery.

Provides the following commands:
    teane run     — Primary execution entry point. Runs the full agent graph.
    teane resume  — Resume a crashed/interrupted session from its checkpoint.
    teane status  — Read-only inspection of a checkpointed session.
    teane doctor  — Run first-run healthchecks (git, API keys, sandbox, DB, config).
    teane purge   — Manually wipe all checkpoint data.

Use `teane -h` or `teane <command> -h` for detailed help on each subcommand.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Optional

from harness import _platform
from harness.loop_counter_keys import PER_BATCH_CAP_COUNTERS, STALL_TRIPWIRE_KEYS
from harness.spec_files import (
    SPEC_FILE_EXTS,
    list_spec_files,
    read_spec_file,
)

# Configure logging for the CLI
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("harness.cli")


# ---------------------------------------------------------------------------
# Workspace safety
# ---------------------------------------------------------------------------

def _refuse_if_workspace_is_harness_root(workspace_path: str) -> bool:
    """Refuse to run when the user has pointed the harness at its own repo.

    The harness writes patches, generated specs, branches, and state files
    into the workspace it's given. If that workspace is the harness checkout
    itself, every "fix" overwrites the harness's own source. Compare realpaths
    so a symlinked alias of either path still trips the check.

    Returns True when the caller should stop. The caller is expected to
    return a non-zero exit code immediately.
    """
    harness_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.realpath(workspace_path) != os.path.realpath(harness_root):
        return False
    print()
    print("=" * 72)
    print("Harness root cannot be the repo root")
    print("=" * 72)
    print(
        f"The path you provided ({workspace_path}) is the harness's own\n"
        f"installation directory. Running the harness against itself would\n"
        f"overwrite its own source as it generates patches.\n\n"
        f"Please re-run with --workspace / -w pointing at a different location\n"
        f"(your application's repository).\n"
    )
    print("=" * 72)
    logger.error(
        "[workspace] Refusing to run: workspace path resolves to the harness "
        "root itself (%s). Choose a different location.",
        harness_root,
    )
    return True


def _refuse_log_inside_workspace(
    *,
    workspace_path: str,
    log_file: Optional[str],
) -> Optional[str]:
    """Return an error message when the operator-supplied ``--log`` path
    (or, on Linux, a shell-redirected stdout target detected via
    ``/proc/self/fd/1``) resolves under ``workspace_path``.

    ``teane build`` wipes the workspace root at startup (preserving only
    product_spec/, .git/, optionally docs/). A log file staged inside
    that root — either via ``--log logs/build.txt`` or a shell redirect
    like ``teane build ... > logs/build.txt`` — gets deleted BEFORE it
    can capture the run. The finsearch session 156032347 operator hit
    this: the redirected log file was silently unlinked and the run
    became invisible.

    Refuse at CLI-parse time so the operator relocates BEFORE the
    destructive step. Returning ``None`` means the check passed.
    """
    workspace_root = os.path.realpath(os.path.abspath(workspace_path))

    def _resolves_under_workspace(target: str) -> bool:
        try:
            abs_target = os.path.realpath(os.path.abspath(target))
        except (OSError, ValueError):
            return False
        try:
            common = os.path.commonpath([workspace_root, abs_target])
        except ValueError:
            return False
        return common == workspace_root

    if log_file:
        if _resolves_under_workspace(log_file):
            return (
                f"Refusing to run: --log target {log_file!r} resolves "
                f"inside the workspace {workspace_path!r}, which `teane "
                f"build` wipes at startup. The log file would be deleted "
                f"before it could capture the run. Move the log to a "
                f"path outside the workspace (e.g. /tmp/teane.log or "
                f"~/logs/teane.log) and re-run."
            )

    # Best-effort stdout redirection detection on Linux only. On other
    # platforms /proc/self/fd is absent — we skip silently rather than
    # raise, so this guard doesn't fire spuriously off-Linux.
    stdout_target: Optional[str] = None
    try:
        stdout_target = os.readlink("/proc/self/fd/1")
    except (OSError, AttributeError):
        stdout_target = None
    if stdout_target and stdout_target.startswith("/"):
        # Only reject if it's a regular file (redirection target), not
        # a tty/pipe/socket — those are interactive runs the operator
        # sees on the terminal and can't accidentally lose to the wipe.
        try:
            is_regular = os.path.isfile(stdout_target)
        except OSError:
            is_regular = False
        if is_regular and _resolves_under_workspace(stdout_target):
            return (
                f"Refusing to run: stdout appears to be redirected to "
                f"{stdout_target!r}, which lives inside the workspace "
                f"{workspace_path!r} that `teane build` wipes at "
                f"startup. The log file would be deleted before it "
                f"could capture the run. Redirect to a path outside "
                f"the workspace (e.g. `> /tmp/teane.log`) or pass "
                f"`--log /tmp/teane.log` and re-run."
            )
    return None


# ---------------------------------------------------------------------------
# Version reporting (P2.5)
# ---------------------------------------------------------------------------

def _get_harness_version() -> str:
    """Return the installed harness package version, or '(unknown)' if not
    discoverable. Used by argparse's ``--version`` action.

    Falls through to '(unknown)' instead of raising so an uninstalled
    in-tree run (e.g. `python -m harness.cli ...`) still produces a usable
    help message.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
        return version("teane")
    except (PackageNotFoundError, ImportError, Exception):  # noqa: BLE001
        return "(unknown)"


# ---------------------------------------------------------------------------
# Git mode (--git true|false) — process-wide state
# ---------------------------------------------------------------------------

# Module-level pin: set once at the top of cmd_run / cmd_resume from
# args.git and read by every git-aware code path (GitGuardian init,
# _attempt_git_rollback, _perform_new_build_reset). Module-level instead
# of threaded as a parameter because the rollback path is nested several
# function calls deep inside the HITL gate node and the value never
# changes mid-run.
_GIT_ENABLED: bool = True


def _set_git_enabled(enabled: bool) -> None:
    global _GIT_ENABLED
    _GIT_ENABLED = bool(enabled)


def _git_enabled() -> bool:
    return _GIT_ENABLED


class _NullGitGuardian:
    """No-op stand-in for :class:`harness.security.GitGuardian` used when
    ``--git false``. Mirrors every public method GitGuardian exposes so
    the rest of cmd_run / cmd_resume don't have to gate each call site.
    Returns benign defaults: ``False`` from booleans, ``None`` from
    branch lookups. The harness treats those values the same way it would
    a real GitGuardian against a workspace with no patch branch yet.
    """

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path

    def is_git_repo(self) -> bool: return False
    def get_current_branch(self): return None
    def has_uncommitted_changes(self) -> bool: return False
    def stash_if_dirty(self) -> bool: return False
    def pop_stash(self) -> bool: return False
    def create_patch_branch(self, session_id: str) -> bool: return False
    def commit_repair_iteration(self, *args: Any, **kwargs: Any) -> bool: return True
    def commit_all_changes(self, *args: Any, **kwargs: Any) -> bool: return True
    def rollback(self, *args: Any, **kwargs: Any) -> bool: return False
    def restore_original_branch(self) -> bool: return False


def _make_git_guardian(workspace_path: str) -> Any:
    """Return a real ``GitGuardian`` when ``--git true`` is in effect,
    otherwise a no-op :class:`_NullGitGuardian`. One place to swap so the
    call sites stay clean."""
    if _git_enabled():
        from harness.security import GitGuardian
        return GitGuardian(workspace_path)
    logger.info("[git] --git false — using no-op GitGuardian stub.")
    return _NullGitGuardian(workspace_path)


# ---------------------------------------------------------------------------
# Workspace lock (P1.7) — single-writer guard
# ---------------------------------------------------------------------------

# Module-level pin: the lock-file handles MUST outlive cmd_run's locals so
# the OS holds the locks for the lifetime of the process. Keyed by workspace
# path so a re-entrant acquisition (cmd_chat after cmd_run, or two different
# workspaces in the same process) doesn't overwrite an earlier handle and
# silently release its file lock via GC. Audit §1.9 / §5.20.
_WORKSPACE_LOCK_HANDLES: dict[str, Any] = {}


def _clear_workspace_lock_pids_atexit() -> None:
    """Truncate every acquired ``.harness_session.lock`` on process exit.

    The OS-level ``fcntl.flock`` is released automatically when the file
    handle is GC'd or the process dies, but the ``pid=NNN`` diagnostic
    line stays behind. Any observer (dashboard, operator, next-run
    inspection) reading that file after we exit sees a phantom holder
    that no longer exists. Clearing the pid line here keeps the file's
    contents in sync with actual ownership.

    Best-effort: we swallow every error. This runs during interpreter
    shutdown where stdlib modules may already be partially torn down,
    and a lock-cleanup failure must never turn a clean exit into a
    traceback that overwrites the operator's real error.
    """
    for _, fh in list(_WORKSPACE_LOCK_HANDLES.items()):
        try:
            fh.seek(0)
            fh.truncate()
            fh.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            fh.close()
        except Exception:  # noqa: BLE001
            pass
    _WORKSPACE_LOCK_HANDLES.clear()


atexit.register(_clear_workspace_lock_pids_atexit)


def _acquire_workspace_lock(workspace_path: str, *, force: bool = False) -> Any:
    """Acquire an exclusive lock on the workspace.

    Returns the locked file handle on success, or ``False`` when another
    session holds the lock and ``force`` is False. POSIX uses
    ``fcntl.flock`` (advisory); Windows uses ``msvcrt.locking``
    (mandatory) — both via :mod:`harness._filelock`. Returns ``None``
    only when no locking backend is available at all (which shouldn't
    happen on any supported platform).

    Stash the handle in a per-workspace slot so:
      - the GC doesn't release the lock when cmd_run's local goes out of scope
      - a second acquisition for a different workspace doesn't accidentally
        evict the first lock's handle (audit §1.9 / §5.20)
    """
    from harness import _filelock
    if not _filelock.LOCKING_AVAILABLE:
        logger.debug(
            "[lock] No file-locking backend available; skipping workspace lock."
        )
        return None

    lock_path = os.path.join(workspace_path, ".harness_session.lock")
    # Open without truncation. The earlier ``mode="w"`` truncated the file
    # before flock acquired the lock — concurrent acquirers could blow away
    # the holder's diagnostic ``pid=NNN`` line. Open as O_RDWR|O_CREAT,
    # take the lock, THEN truncate + write the pid (audit §1.9).
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        fh = os.fdopen(fd, "r+", encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "[lock] Could not create lock file %s: %s. Proceeding without lock.",
            lock_path, exc,
        )
        return None

    try:
        _filelock.lock_exclusive_nonblocking(fh)
    except BlockingIOError:
        if force:
            logger.warning(
                "[lock] %s is held by another session, but --force-lock was "
                "passed — taking the lock anyway. Concurrent corruption is "
                "now possible; you own the risk.",
                lock_path,
            )
            try:
                _filelock.lock_exclusive_blocking(fh)
            except OSError as exc:
                logger.error("[lock] Force-lock failed too: %s", exc)
                fh.close()
                return False
        else:
            logger.error(
                "[lock] Workspace %s is locked by another live teane session "
                "session. Refusing to start so the two don't clobber each "
                "other's patches.\n"
                "  Wait for the other session to finish, or pass --force-lock "
                "if you're certain it's stuck (e.g. a previous crash left the "
                "lock stranded).",
                workspace_path,
            )
            fh.close()
            return False

    # Now that we hold the lock, truncate and write the holder pid so
    # operators inspecting the file see the live owner. No other process
    # can be mid-write because we hold the EX lock.
    try:
        fh.seek(0)
        fh.truncate()
        fh.write(f"pid={os.getpid()}\n")
        fh.flush()
    except OSError:
        pass

    # Pin in the per-workspace slot. Use realpath to merge symlinked
    # workspace aliases so two callers using different but equivalent
    # paths can't both think they hold the lock.
    try:
        key = os.path.realpath(workspace_path)
    except Exception:  # noqa: BLE001
        key = workspace_path
    _WORKSPACE_LOCK_HANDLES[key] = fh
    logger.info("[lock] Acquired workspace lock: %s (pid=%d)", lock_path, os.getpid())
    return fh


# ---------------------------------------------------------------------------
# 1. Configuration Discovery — single canonical source
# ---------------------------------------------------------------------------
#
# The harness reads ONE config file and only one: <teane_root>/config/config.json.
# There are no fallbacks, no per-workspace overrides, no auto-generated files.
# Per-project differences (docker image, network) are handled by the
# harness's auto-detection (graph._toolchain_image_for,
# graph._build_command_needs_network). The build command itself is
# auto-wired from workspace markers (cli._detect_default_build_command)
# under the locked core_languages stack — there is no CLI override.
# The only build-shape CLI flags are --budget and --allow-network.
#
# Validation is STRICT — see validate_config_strict() below. Unknown keys,
# missing required fields, wrong types, or cross-reference errors raise
# ConfigError. Every CLI subcommand catches ConfigError at the outermost
# layer, prints the message to stderr, and exits with code 2 BEFORE any
# other initialization (logging setup, storage init, lock acquisition,
# gateway construction). No LLM call ever happens with bad config.
#
# API keys live in env vars (ANTHROPIC_API_KEY, OPENAI_API_KEY,
# DEEPSEEK_API_KEY) and are resolved at dispatch time by gateway.py:331.
# Operators may NOT put live keys in this file; the schema slot is kept as
# empty string for documentation only.


class ConfigError(Exception):
    """Raised when the canonical config file is missing, malformed, has
    unknown keys, has wrong types, or fails cross-reference validation.

    Every CLI subcommand entry catches this at the outermost layer and
    exits 2 with the message printed to stderr — by design, the harness
    refuses to run with bad config rather than silently picking defaults.
    """


def _get_global_config_path() -> str:
    """Resolve the repo-root canonical config path:
    ``<teane_root>/config/config.json``.

    The harness package lives at ``<root>/harness/``, so the parent of
    this module's directory is the repo root.

    Audit §5.19: resolve symlinks via :func:`os.path.realpath` so a
    ``pip install -e`` from a symlinked checkout (or any deployment
    where the harness module is symlinked) still locates the real
    source repo's ``config/`` directory rather than pointing inside a
    venv site-packages copy.
    """
    package_dir = os.path.dirname(os.path.realpath(__file__))
    repo_root = os.path.dirname(package_dir)
    return os.path.join(repo_root, "config", "config.json")


def load_raw_config() -> dict[str, Any]:
    """Load the canonical config file and strip ``_comment`` keys, but
    DON'T validate. Used by the setup wizard so it can find missing API
    key env vars and prompt for them before strict validation runs.

    Same I/O errors as :func:`discover_config` (missing file, JSON syntax,
    OS) raise :class:`ConfigError`. Wrong shape (top-level not an object)
    also raises. Everything else — unknown keys, missing required fields,
    missing env vars — is left for :func:`validate_config_strict` to flag.
    """
    path = _get_global_config_path()
    if not os.path.isfile(path):
        raise ConfigError(
            f"Canonical config not found at {path}. "
            f"The harness reads exactly one config file and it must exist. "
            f"Create it (see <teane_root>/config/config.json in the repo "
            f"for the documented schema) before re-running the harness."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Invalid JSON in {path}: {exc}. "
            f"Fix the JSON syntax before re-running the harness."
        ) from exc
    except OSError as exc:
        raise ConfigError(
            f"Cannot read {path}: {exc}. "
            f"Fix the file permissions or path before re-running the harness."
        ) from exc

    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path} must contain a JSON object at the top level, got {type(raw).__name__}."
        )

    return _expand_env_placeholders(_strip_comments(raw))


def load_deployment_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Pull the optional ``deployment_defaults`` section out of the
    already-loaded canonical config.

    Returns ``{}`` when the section is absent. When populated, the four
    sub-sections (``network``, ``storage``, ``secrets``, ``infra_sync``)
    are threaded into ``deployment_discovery_node`` so the planning LLM
    treats every populated field as a resolved policy and skips emitting
    a question for it.

    Section shape (known sub-section keys, type checks) is enforced by
    :func:`validate_config_strict` at startup, so this function is a
    pure dict lookup + INFO log. Unknown LEAF keys inside the four
    sub-sections are passed through verbatim — operators may set
    organization-specific policies the harness has never heard of.
    """
    section = cfg.get("deployment_defaults") or {}
    if not isinstance(section, dict):
        return {}
    populated = [s for s in ("network", "storage", "secrets", "infra_sync")
                 if section.get(s)]
    logger.info(
        "[cli] Loaded deployment defaults from config.json "
        "(sections populated: %s).",
        populated or "none",
    )
    return section


def discover_config(workspace_path: Optional[str] = None) -> dict[str, Any]:
    """Load and strictly validate the canonical config.

    This MUST be the first deterministic check at every CLI entry. It
    performs no LLM calls, no network, no file writes — pure read +
    validate. On any error it raises :class:`ConfigError` with an
    actionable message and the harness exits before doing anything else.

    The ``workspace_path`` argument is accepted for call-site
    backwards-compatibility but is unused for config purposes — the
    harness no longer reads per-workspace ``.harness_config.json``
    files. If one is present we log a one-shot INFO line noting it
    will be ignored.

    Returns a fully-validated, comment-stripped config dictionary.
    """
    cfg = load_raw_config()
    validate_config_strict(cfg, source=_get_global_config_path())

    # Legacy detection: notify but do not act.
    if workspace_path:
        _warn_if_legacy_workspace_config(workspace_path)

    return cfg


def _strip_comments(cfg: dict[str, Any]) -> dict[str, Any]:
    """Recursively drop keys starting with ``_`` (used for inline JSON
    documentation that JSON-the-format doesn't support natively)."""
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        if isinstance(key, str) and key.startswith("_"):
            continue
        if isinstance(value, dict):
            out[key] = _strip_comments(value)
        else:
            out[key] = value
    return out


# Machine-local values in the committed config resolve from the
# environment, the same way API keys do: the repo commits the shape
# (`${TEANE_MCP_FS_ROOT:-~/projects/teane}`), each machine supplies its
# values via exported env vars, and the default after `:-` covers the
# common case so most machines need no setup at all. ONLY the
# TEANE_/HARNESS_ namespaces expand — `${POSTGRES_PASSWORD}`-style
# strings destined for docker-compose or shell templates pass through
# untouched.
_ENV_PLACEHOLDER_RE = re.compile(
    r"\$\{((?:TEANE|HARNESS)_[A-Z0-9_]+)(?::-([^}]*))?\}"
)


def _expand_env_placeholders(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``${TEANE_*}`` / ``${HARNESS_*}`` placeholders (with
    optional ``:-default``) in every string value, then expand a leading
    ``~``. Raises :class:`ConfigError` naming the variable and the config
    location when a placeholder has no default and the variable is unset,
    or when a namespace placeholder is malformed (e.g. unclosed brace).
    """
    def _expand_str(value: str, where: str) -> str:
        def _sub(match: "re.Match[str]") -> str:
            var, default = match.group(1), match.group(2)
            resolved = os.environ.get(var)
            if resolved is None:
                if default is None:
                    raise ConfigError(
                        f"Config value at '{where}' references ${{{var}}} "
                        f"but the environment variable is not set and no "
                        f"':-default' was given. Export {var} (like the "
                        f"API-key env vars) or add a default: "
                        f"${{{var}:-<value>}}."
                    )
                resolved = default
            return resolved

        expanded = _ENV_PLACEHOLDER_RE.sub(_sub, value)
        if "${TEANE_" in expanded or "${HARNESS_" in expanded:
            raise ConfigError(
                f"Config value at '{where}' contains a malformed "
                f"TEANE_/HARNESS_ env placeholder: {expanded!r}. Use the "
                f"form ${{TEANE_NAME}} or ${{TEANE_NAME:-default}}."
            )
        if expanded == "~" or expanded.startswith("~/"):
            expanded = os.path.expanduser(expanded)
        return expanded

    def _walk(node: Any, where: str) -> Any:
        if isinstance(node, str):
            return _expand_str(node, where)
        if isinstance(node, dict):
            # ``_``-prefixed keys are inline documentation. _strip_comments
            # removes them from dicts it reaches, but it doesn't recurse
            # into lists (e.g. mcp.servers[]) — leave any survivors
            # verbatim rather than expanding prose.
            return {
                k: v if (isinstance(k, str) and k.startswith("_"))
                else _walk(v, f"{where}.{k}" if where else str(k))
                for k, v in node.items()
            }
        if isinstance(node, list):
            return [_walk(v, f"{where}[{i}]") for i, v in enumerate(node)]
        return node

    return _walk(cfg, "")


def _warn_if_legacy_workspace_config(workspace_path: str) -> None:
    """Log one INFO line when a workspace still has a ``.harness_config.json``
    from the pre-consolidation era. The file is ignored; the operator can
    delete it (or leave it — we don't touch it)."""
    legacy = os.path.join(workspace_path, ".harness_config.json")
    if os.path.isfile(legacy):
        logger.info(
            "[cli] Legacy .harness_config.json detected at %s — ignored. "
            "The canonical config is %s; per-workspace overrides are no "
            "longer supported. You can delete the legacy file at your "
            "convenience.",
            legacy, _get_global_config_path(),
        )


# Top-level keys the harness knows about. Anything outside this set in a
# user-provided config is almost certainly a typo (e.g. "model_routin").
# Add new keys here when wiring new config sections.
_KNOWN_TOP_LEVEL_KEYS = frozenset({
    "allow_network", "sandbox", "token_budget",
    "node_throttle", "models", "model_routing", "persistence",
    "redaction", "security", "skills", "deployment",
    "speculative", "impact", "lintgate", "logging", "languages",
    "test_generation", "metrics", "llm_dispatch",
    # 2026-07-06 — parallel-agent fan-out defaults promoted from
    # harness/fanout.py constants. See GatewayConfig.fanout_max_concurrency
    # / fanout_timeout_seconds for semantics.
    "fanout",
    # Operator-configurable name of the folder at the workspace root that
    # holds the product spec files (.txt / .md / .pdf). Mandatory in
    # config.json — the harness refuses to start without it.
    # See _load_consolidated_product_spec.
    "product_spec_dir",
    # Operator-configurable name of the folder at the workspace root that
    # holds change-request files (.txt / .md / .pdf) for non-greenfield
    # runs. Optional; defaults to "change_requests". When --new-build
    # false the folder MUST contain at least one spec file —
    # see _load_consolidated_change_requests.
    "change_requests_dir",
    # Observability + debugging knobs. See _dump_repair_prompt_to_disk and
    # the compiler.run_prod_import_smoke_check flag.
    "debug", "compiler",
    # Patcher behaviour knobs (B5: enforce_read_before_edit).
    "patcher",
    # Change-request behaviour knobs (reverse_engineer_budget_usd etc.).
    "change_requests",
    # Default HITL gate switches. The matching --hitl-* CLI flag wins when
    # explicitly passed; this section is the second tier; the in-code
    # default (True) is the last-resort. See _resolve_hitl_flags.
    "hitl",
    # Web research tool skills (web_fetch / web_search). Default off — the
    # gateway path is unchanged when disabled. See harness/web_tools.py.
    "web_tools",
    # Per-repo session memory. Default enabled. Planner reads prior
    # session notes; cmd_run appends a session entry on exit. See
    # harness/repo_memory.py.
    "memory",
    # Model Context Protocol client pool. Default off. When enabled,
    # each declared server spawns as a subprocess (stdio transport),
    # advertises its tools, and the harness registers them as
    # `mcp__<server>__<tool>` skills the planner can invoke via
    # `<<<MCP_CALL>>>` blocks. See harness/mcp_client.py.
    "mcp",
    # GitHub integration. Pure config-side knobs (gh_path); subcommands
    # (`teane gh issue / pr-create / pr-comment`) work without a
    # config block when `gh` is on PATH.
    "github",
    # Semantic retrieval index. Built via `teane index build` and
    # injected into the planner context when `repo_index.enabled=true`.
    # Default off — the planner is unchanged when disabled. See
    # harness/repo_index.py.
    "repo_index",
    # Cron-driven scheduled job daemon. Started with `teane schedule
    # run`. Default off. See harness/schedule.py.
    "schedule",
    # Read-only web dashboard. Started with `teane web`. Default
    # bind 127.0.0.1; optional bearer-token auth. See harness/dashboard.py.
    "dashboard",
    # Optional org-wide deployment policy (reverse proxy, TLS strategy,
    # secret manager, UID/GID, backup destination, conflict policy, ...).
    # Populated fields are treated as RESOLVED by the planning LLM in
    # deployment_discovery_node — it skips emitting interview questions
    # for them. Empty / absent section keeps today's full-questionnaire
    # behaviour. See load_deployment_defaults() in this file.
    "deployment_defaults",
    # Top-level Agile-mode default. When true (and the operator does not
    # pass --agile explicitly), build/patch engage story decomposition +
    # per-story TDD. Defaults to false. See agile_defaults below for the
    # per-knob tuning that used to live as --story-* CLI flags.
    "agile",
    # Agile-mode tuning knobs. Replaces the --story-batch-size /
    # --commit-on-story / --story-repair-cap CLI flags which were
    # removed when `teane run` was split into `teane build`/`patch`.
    # Resolution precedence: hard-coded default < this section. Only
    # consulted when agile mode is engaged.
    "agile_defaults",
    # Locked core technology selection. backend_language must be
    # "Python" or "Java"; web_language must be exactly the list
    # ["React", "TypeScript", "TailwindCSS"]. Blank values are
    # auto-defaulted (Python / React+TS+Tailwind). Any other value
    # is a configuration error and the harness refuses to start.
    "core_languages",
    # Read-only type-checker gate between lintgate and compiler
    # (pyright/mypy for Python, tsc for TS/TSX). Default on, fail-open.
    # See harness/diagnostics_gate.py.
    "diagnostics",
    # LSP client pool for semantic navigation — brownfield flows only
    # (patch/test). pyright-langserver / typescript-language-server behind
    # an environment-health probe; DependencyGraph stays the fallback.
    # See harness/lsp_client.py.
    "lsp",
    # Automated failure post-mortems — the HITL learning loop. On HITL or
    # failed exit, a [learned-rule:<trigger>] note lands in per-repo
    # memory and reaches the next run's planner. See harness/post_mortem.py.
    "post_mortem",
    # Coverage gate for generated apps (FR-080). Operator override for the
    # minimum-line-coverage threshold the LLM writes into each generated
    # Makefile / package.json test target. Templated into the skills at
    # system-prompt-build time via {{coverage.min_pct}} substitution
    # (see graph._load_skills_markdown).
    "coverage",
    # Read-only agentic retrieval tools (grep / glob / list_dir / find_symbol
    # / file_outline / semantic_search / git_blame / git_log) exposed in the
    # native tool loop. Master switch + output caps. See
    # harness/retrieval_tools.py.
    "retrieval_tools",
    # Opt-in trajectory-level best-of-N: run N teane subprocesses in isolated
    # worktrees and apply the winner. Off by default. See
    # harness/best_of_n_runner.py.
    "best_of_n",
    # Kill switches for the cheap LLM-judgment touchpoints (HITL escalation
    # summary, patcher-rejection diagnosis, repair reflection, repair-history
    # condenser, ...). All default True; see GatewayConfig.llm_judgment_*.
    # Previously documented but rejected by this validator — the switches
    # were unusable.
    "llm_judgment",
    # Repair-node tuning (structured_diagnostic_payload). Read by the
    # gateway config factory; same previously-unregistered situation as
    # llm_judgment.
    "repair",
})

# Per-section known keys. Used to detect typos like
# `token_budget.hrad_cap_usd` that silently no-op'd before — a typoed
# budget cap meant agents ran without one. Keep these in sync with the
# consumers grep'd by section (sandbox.py, deploy.py, lintgate.py, etc.).
_KNOWN_NESTED_KEYS: dict[str, frozenset[str]] = {
    "sandbox": frozenset({
        "backend", "docker_image", "docker_memory_limit", "docker_cpu_limit",
        "docker_pids_limit", "readonly_cache_mounts", "timeout_seconds",
        "pgid_kill_on_timeout", "log_buffer_max_mb",
        # P1.3: opt-in for auto-enabling network on detected pip/npm install.
        "auto_enable_network_for_install",
        # Chown bind-mounted files back to the host user on docker container
        # exit — prevents root-owned __pycache__/ littering the workspace.
        "restore_workspace_ownership",
        # Toggled by compiler_node when the build command writes to system
        # locations the read-only root FS would block.
        "read_only_root",
        # Writable named Docker volumes mounted on the builder image's
        # fixed /cache/{pip,uv,npm} paths so pip / uv / npm downloads
        # persist across containers. Default ON, default scope "global"
        # (shared across sessions); set scope = "session" for per-session
        # isolation. See _cache_volume_name in sandbox.py.
        "cache_volumes", "cache_volumes_prefix", "cache_volumes_scope",
    }),
    "token_budget": frozenset({
        "hard_cap_usd", "context_window_threshold_pct",
        # Per-stage soft budget allocation. Optional dict mapping NodeRole
        # values to target fractions of hard_cap_usd. Observability-only
        # today (warning when a stage exceeds its share); hard enforcement
        # is a follow-up. Example: {"planning": 0.2, "patching": 0.2,
        # "repair": 0.5, "doc_reviewer": 0.05, "code_reviewer": 0.05}.
        "stages",
    }),
    "node_throttle": frozenset({
        "max_patch_repair_iterations",
        # Consecutive-DISTRACTION circuit breaker for the repair loop.
        "max_consecutive_distraction_rounds",
        # Bug B (2026-07-04) — consecutive-low-signal-verdict circuit
        # breaker for the repair loop.
        "max_consecutive_low_signal_rounds",
        # 2026-07-04 — router hard-ceiling multiplier for total repair
        # rounds per compile phase. Combined with
        # ``max_patch_repair_iterations`` to compute the absolute cap.
        "total_hard_cap_multiplier",
        "max_doc_review_cycles",
        "max_code_review_cycles",
        "max_discovery_iterations",
        # Phase G — end-of-session regression repair cap.
        "max_end_of_session_regression_cycles",
        # Phase J — end-of-session repair authority knobs.
        "end_of_session_repair_diagnostic_cap",
        "end_of_session_repair_inventory_cap",
        "end_of_session_force_reasoning_model",
        # 2026-07-06 — router tripwires promoted from hard-coded
        # constants in graph.route_after_repair. See GatewayConfig
        # fields of the same name for semantics.
        "stuck_target_limit",
        "generic_no_progress_limit",
        "same_missing_dep_limit",
    }),
    "persistence": frozenset({
        "db_path", "ttl_days", "redact_messages",
    }),
    "model_routing": frozenset({
        "planning_primary", "planning_mode", "planning_fallback",
        "planning_fallback_mode",
        "patching_primary", "patching_mode",
        "patching_fallback", "patching_fallback_mode",
        "repair_primary", "repair_fallback", "repair_mode",
        "repair_fallback_mode",
        "doc_reviewer_primary", "doc_reviewer_mode", "doc_reviewer_fallback",
        "doc_reviewer_fallback_mode",
        "code_reviewer_primary", "code_reviewer_mode", "code_reviewer_fallback",
        "code_reviewer_fallback_mode",
        "ollama_local_model", "ollama_local_backup", "force_local_only",
    }),
    "deployment": frozenset({
        "enabled", "compose_file",
        "health_check_interval_seconds", "health_check_timeout_seconds",
    }),
    # Org-wide deployment-policy defaults. The four sub-sections are
    # the only keys validated here; LEAF fields inside each sub-section
    # are intentionally NOT enumerated — operators may set
    # organization-specific policies the harness has never heard of and
    # they're passed verbatim to the planning LLM.
    "deployment_defaults": frozenset({
        "network", "storage", "secrets", "infra_sync",
    }),
    "lintgate": frozenset({
        "format_modified_files",
        "strict_missing_formatter",
    }),
    "diagnostics": frozenset({
        "enabled", "timeout_seconds", "max_rounds", "scope", "tools",
    }),
    "lsp": frozenset({
        "enabled", "enabled_flows", "request_timeout_seconds",
        "python_require_venv", "prefetch_budget_seconds",
    }),
    "post_mortem": frozenset({
        "enabled", "max_cost_usd", "retire_on_clean_run",
    }),
    "logging": frozenset({
        "level", "log_dir", "json_stderr", "langsmith",
        # P2.3: rotation knobs for the per-session JSONL file handler.
        "max_bytes", "backup_count",
        # Console (stderr) verbosity, independent of the file handler.
        # "WARNING" keeps the terminal quiet while the file captures the
        # full stream; "OFF"/"NONE" silences the console entirely; null
        # mirrors `level`. See observability.configure_logging.
        "console_level",
    }),
    "test_generation": frozenset({
        "enabled", "max_iterations",
    }),
    "fanout": frozenset({
        "max_concurrency",
        "timeout_seconds",
    }),
    # P2.7: cost-metrics aggregation (teane metrics subcommand).
    "metrics": frozenset({
        "burn_rate_window_minutes", "metrics_dir",
    }),
    # Speculative branching parameters consumed by harness/speculative.py.
    # enabled defaults to FALSE (the node short-circuits to the standard
    # patching flow). Flip to True per-config when running a workload where
    # parallel exploration is likely to find a passing variant — see the
    # speculative.enabled discussion in config/config.json. num_variants is
    # the fork count; temperature controls per-variant diversity (sweet spot
    # 0.2-0.4 for code); selection_strategy is the winner-pick rule.
    "security": frozenset({
        # Phase 5: PR-style diff-approval gate on writes. Off by
        # default; the "Maximum quality" preset (Phase 6) flips it on.
        "diff_approval_required",
        # Existing security-scan configuration (mirrors the shipped
        # config/config.json). Adding "security" to this dict at all
        # switches nested-key checking on for the section; every key
        # that was previously tolerated via fall-through must be
        # enumerated explicitly.
        "block_on", "warn_on", "ignore_below", "scanners",
        "allowlist_rules", "max_findings_to_route_to_repair",
        # 2026-07-06 — security-scan ceiling multiplier promoted from
        # harness/security.py::_HARD_SECURITY_CEILING_MULTIPLIER.
        "hard_ceiling_multiplier",
    }),
    "speculative": frozenset({
        # Original keys (legacy schema — preserved for backwards compat).
        "enabled", "num_variants", "temperature",
        "selection_strategy", "worktree_base_dir",
        # Rebuild keys (#12) — strategy axes and per-strategy parameters.
        "trigger", "n_repair_failures_threshold",
        "diversity_mode", "cost_strategy", "salvage_strategy",
        "max_concurrency", "variant_models", "variant_prompt_styles",
        "expensive_model", "cheap_model", "voting",
        # Repair-level fanout: after repair_fanout_after_rounds consecutive
        # no-progress repair rounds, sample repair_fanout_variants repair
        # responses, test-compile each in a seeded worktree, keep the best.
        # See harness/speculative.py:maybe_run_repair_fanout.
        "repair_fanout", "repair_fanout_variants",
        "repair_fanout_after_rounds",
    }),
    # LLM dispatch parameters consumed by harness/gateway.py.
    # max_tokens_per_role is a free-form dict (role -> int) — the
    # validator type-checks the wrapper but doesn't enumerate role
    # names, so future NodeRole additions don't break this list.
    "llm_dispatch": frozenset({
        "max_tokens_default", "max_tokens_per_role",
        # Per-role bool map controlling the finish_reason=="length"
        # continuation loop in graph._continue_on_length. Per-role
        # defaults live in graph._CONTINUE_ON_LENGTH_DEFAULTS; an
        # operator-supplied entry overrides the default for that role.
        # See the _llm_dispatch_comment block in config.json for the
        # per-role risk profile.
        "continue_on_length",
        # Cap on continuation cycles per dispatch. Clamped to [1, 10]
        # in graph._resolve_max_continuation_cycles; default 5.
        "max_continuation_cycles",
        # Cap on read_file tool-use rounds inside one patching turn.
        # Clamped to [1, 30] in graph._resolve_patching_read_file_cap;
        # default 10.
        "patching_read_file_cap",
        # Cap on <<<READ_FILE>>> DSL resolve rounds per repair/patching
        # iteration. Clamped to [1, 20] in
        # graph._resolve_read_file_rounds; default 6. One bonus round is
        # granted past the cap when the request names a workspace file
        # the LLM has never been shown (session 22471c0c).
        "read_file_rounds",
        # Prompt caching master switch. Default True. Falls back to the
        # legacy string-form Anthropic system payload when False, and
        # silences the prefix-stability drift events. Single-flag
        # rollback if a provider rejects the cache_control directives.
        "prompt_cache_enabled",
    }),
    # Observability knobs. dump_llm_calls captures every LLM dispatch
    # (across all roles) to ~/.harness/debug for post-mortem analysis;
    # dump_max_files caps the directory size. dump_repair_prompts is the
    # deprecated alias for dump_llm_calls — accepted with a warning.
    "debug": frozenset({
        "dump_llm_calls", "dump_max_files", "dump_repair_prompts",
    }),
    # Patcher behaviour knobs. enforce_read_before_edit gates the B5
    # read-before-edit invariant — when true the patcher rejects edits to
    # files the LLM has not yet been shown this turn. use_structured_tools
    # gates the B6 native tool-use migration — when true, providers that
    # support function/tool calling receive the PATCH_TOOLS schema in
    # their chat_completion call instead of (or alongside) the text DSL.
    "patcher": frozenset({
        "enforce_read_before_edit", "root_files", "use_structured_tools",
    }),
    # Pre-build smoke checks (see compiler_node prod-import step).
    "compiler": frozenset({
        "run_prod_import_smoke_check", "advisory_exit_codes",
        # Fail-to-pass fast path: re-run the previous round's failing
        # pytest selectors before the full suite (default True). See
        # compiler_node's targeted-tests-first block.
        "targeted_tests_first",
    }),
    # Coverage gate for generated apps (FR-080).
    # min_pct — integer 0-100, default 70. Substituted into every Makefile
    #   / package.json coverage rule the LLM emits (skills reference it via
    #   {{coverage.min_pct}}).
    # enforce — bool, default true. When true the LLM writes the fail-under
    #   flag (pytest --cov-fail-under, Jest coverageThreshold) so a build
    #   under threshold exits non-zero and repair_node re-enters to add
    #   more tests. When false coverage is still measured but the threshold
    #   flag is omitted — the build passes regardless of coverage%.
    "coverage": frozenset({"min_pct", "enforce"}),
    # Read-only agentic retrieval tools exposed in the native tool loop.
    # enabled=false removes them from the tool list (patch tools still ship).
    # The rest bound output/latency. See harness/retrieval_tools.py.
    "retrieval_tools": frozenset({
        "enabled", "max_results", "max_files", "max_bytes",
        "grep_timeout_s", "git_timeout_s", "list_dir_depth",
        "semantic_top_k", "git_log_max",
    }),
    # Opt-in trajectory-level best-of-N. enabled + n>1 activates it; strategy
    # ∈ {first_success, fewest_changes, voted}. See harness/best_of_n_runner.py.
    "best_of_n": frozenset({
        "enabled", "n", "strategy", "max_concurrency",
        "diversity_mode", "per_variant_budget_usd",
    }),
    # Web research tools. enabled toggles registration of web_fetch +
    # web_search in the SkillRegistry. max_bytes / max_results cap result
    # size. search_backend selects the search provider (only
    # ``duckduckgo_lite`` ships today). allow_private_ips opens the
    # SSRF guard for trusted internal targets. timeout_seconds bounds
    # each HTTP call. tool_call_cap_per_dispatch caps how many tool
    # rounds a single graph-node call may take before the interceptor
    # forces the LLM to proceed.
    "web_tools": frozenset({
        "enabled", "max_bytes", "max_results", "search_backend",
        "api_key_env", "allow_private_ips", "timeout_seconds",
        "tool_call_cap_per_dispatch",
        # Additional search-backend entries — each {name, enabled,
        # search_backend, api_key_env}. Surfaced as a + Add list on the
        # configure page so operators can register multiple backends
        # alongside the primary defined by the scalars above.
        "backends",
    }),
    # MCP client pool. enabled toggles registration of MCP servers as
    # subprocess clients + their tools as McpToolSkill entries.
    # tool_call_timeout_seconds bounds every tools/call. command_allowlist
    # extends the built-in safe allowlist (npx, node, python*, uvx,
    # docker) with operator-trusted binaries. allow_local_filesystem_servers
    # opens the gate for the official filesystem MCP server (which bypasses
    # the build sandbox — must be explicit). result_max_bytes caps every
    # tools/call response. servers is the per-server config list (each
    # entry validated separately — see McpServerConfig).
    "mcp": frozenset({
        "enabled", "tool_call_timeout_seconds", "command_allowlist",
        "allow_local_filesystem_servers", "result_max_bytes", "servers",
    }),
    # Skill loader. user_skills_dir is the path the SkillRegistry walks at
    # startup to import user-supplied *.py files (each can call
    # harness.skills.register to add tools / pipelines / sub-agents, or
    # harness.web_tools.register_backend to plug in an alternative
    # search backend). Defaults to ~/.harness/user_skills (the legacy
    # ~/.harness/skills path is still honoured with a deprecation log
    # so existing installs don't break).
    "skills": frozenset({"user_skills_dir"}),
    # Per-repo memory. enabled toggles the whole feature. dir is the
    # directory holding ``<repo_id>.md`` files. max_bytes is the FIFO
    # trim cap on the file itself. inject_max_bytes caps what the
    # planner sees (tail of file) so the system message stays small.
    "memory": frozenset({
        "enabled", "dir", "max_bytes", "inject_max_bytes",
        "compact_after", "compact_keep_recent",
    }),
    # Scheduled-job daemon. enabled toggles the entire feature; jobs is
    # the list of {name, schedule, workspace, prompt, on_success,
    # on_failure, enabled, harness_args}. history_db / log_dir /
    # tick_seconds / harness_binary configure where state lands and
    # how often the daemon polls. See harness/schedule.py.
    "schedule": frozenset({
        "enabled", "jobs", "history_db", "log_dir",
        "tick_seconds", "harness_binary",
    }),
    # Read-only web dashboard. Default bind 127.0.0.1; optional
    # bearer-token auth via the named env var. See harness/dashboard.py.
    "dashboard": frozenset({
        "enabled", "host", "port", "token_env",
        "log_dir", "metrics_dir", "memory_dir", "repo_index_dir",
        "schedule_db", "static_dir", "chart_js_url", "sessions_max",
        # Tier B/C extensions (web app).
        "writes_enabled", "csrf_token_env", "hitl_webhook_secret",
        "hitl_webhook_timeout_seconds", "audit_log_retention_days",
        "web_db_path", "config_path",
        # Carbon Design System shell + docs viewer.
        "carbon_css_url", "carbon_js_url", "docs_dir",
    }),
    # GitHub integration. gh_path lets ops point at a non-PATH `gh`.
    # default_owner / default_repo supply the fallback for PR creation
    # and issue ingest when the caller doesn't disambiguate (the
    # configure-page overhaul surfaces both as inputs on the GitHub card).
    "github": frozenset({"gh_path", "default_owner", "default_repo"}),
    # Semantic retrieval index. enabled gates planner injection +
    # auto-build behaviour. backend = "tfidf" (default) | "openai_embeddings".
    # chunk_lines / chunk_overlap shape the per-file slicing. top_k bounds
    # the retrieval set. inject_max_bytes caps what the planner sees.
    # index_dir is where the SQLite DB lives. exclude_globs / text_extensions
    # tune the file walker. openai_model / openai_api_base configure the
    # embeddings backend when selected.
    "repo_index": frozenset({
        "enabled", "backend", "top_k", "chunk_lines", "chunk_overlap",
        "inject_max_bytes", "index_dir", "exclude_globs", "text_extensions",
        "max_file_bytes", "openai_model", "openai_api_base",
    }),
    # Change-request mode knobs. reverse_engineer_budget_usd caps the
    # one-shot LLM walk in reverse_engineer_architecture_node so a large
    # codebase doesn't blow the session budget on first contact.
    "change_requests": frozenset({"reverse_engineer_budget_usd"}),
    # Default values for each --hitl-* gate switch. The CLI flag, when
    # explicitly passed, wins. See _resolve_hitl_flags.
    "hitl": frozenset({
        "requirement", "architecture", "repair", "deployment",
        "layout_divergence",
        # Fix 1 (2026-07-10): per-session and per-trigger auto-resume
        # caps for headless-mode HITL loop. See ``_HITL_AUTO_RESUME_CAP``
        # / ``_HITL_AUTO_RESUME_CAP_PER_TRIGGER`` at module top.
        "auto_resume_cap", "auto_resume_cap_per_trigger",
    }),
    # Agile tuning knobs. batch_size = max stories per dependency batch;
    # commit_on_story = git-commit after each green story; repair_cap =
    # max repair iterations before a story is parked as blocked. Only
    # consulted when agile mode is engaged. CLI flags were removed when
    # `teane run` was split into build/patch/deploy.
    "agile_defaults": frozenset({
        "batch_size", "commit_on_story", "repair_cap",
    }),
    # Locked core technology selection. Allowed values enforced by the
    # validator in this file — backend_language ∈ {"Python", "Java"};
    # web_language MUST equal {"React", "TypeScript", "TailwindCSS"}.
    "core_languages": frozenset({
        "backend_language", "web_language",
    }),
    # Per-touchpoint kill switches for the cheap LLM-judgment calls.
    # Names match the config-side keys read by the gateway factory
    # (create_gateway_from_config), NOT the GatewayConfig attribute
    # names. All default True when the section is absent.
    "llm_judgment": frozenset({
        "hitl_escalation_summary", "patcher_rejection_diagnosis",
        "preflight_autofix_judgment", "discovery_saturation_check",
        "repair_reflection", "discovery_followup_focus",
        "app_usage_guide", "repair_history_condense",
    }),
    # Repair-node tuning read by the gateway factory.
    "repair": frozenset({
        "structured_diagnostic_payload",
    }),
}


# Per-field type schema used by validate_config_strict. Keys are dotted paths
# matching the structure in config.json. A value's runtime type must be in
# the listed tuple; bool is excluded from int matches via an explicit check
# because Python's bool is a subclass of int.
_TYPE_SCHEMA: dict[str, tuple[type, ...]] = {
    "allow_network": (bool,),
    "product_spec_dir": (str,),
    "change_requests_dir": (str,),
    "debug.dump_llm_calls": (bool,),
    "debug.dump_max_files": (int,),
    "debug.dump_repair_prompts": (bool,),  # deprecated alias for dump_llm_calls
    "patcher.enforce_read_before_edit": (bool,),
    "patcher.root_files": (list,),
    "patcher.use_structured_tools": (bool,),
    # Phase 5: PR-style diff approval before writes. Off by default so
    # existing operators aren't blocked mid-run. The "Maximum quality"
    # preset flips it on; consumers who want the safety opt in via a
    # single-key config edit or the wizard.
    "security.diff_approval_required": (bool,),
    "compiler.run_prod_import_smoke_check": (bool,),
    "compiler.targeted_tests_first": (bool,),
    "change_requests.reverse_engineer_budget_usd": (int, float),
    "hitl.requirement": (bool,),
    "hitl.architecture": (bool,),
    "hitl.repair": (bool,),
    "hitl.deployment": (bool,),
    "hitl.layout_divergence": (bool,),
    "hitl.auto_resume_cap": (int,),
    "hitl.auto_resume_cap_per_trigger": (int,),
    "sandbox.backend": (str,),
    "sandbox.docker_image": (str,),
    "sandbox.docker_memory_limit": (str,),
    "sandbox.docker_cpu_limit": (str,),
    "sandbox.docker_pids_limit": (int,),
    "sandbox.readonly_cache_mounts": (list,),
    "sandbox.timeout_seconds": (int,),
    "sandbox.pgid_kill_on_timeout": (bool,),
    "sandbox.log_buffer_max_mb": (int,),
    "sandbox.auto_enable_network_for_install": (bool,),
    "sandbox.restore_workspace_ownership": (bool,),
    "sandbox.read_only_root": (bool,),
    "sandbox.cache_volumes": (bool,),
    "sandbox.cache_volumes_prefix": (str,),
    "sandbox.cache_volumes_scope": (str,),
    "token_budget.hard_cap_usd": (int, float),
    "token_budget.stages": (dict,),
    "token_budget.context_window_threshold_pct": (int, float),
    "node_throttle.max_patch_repair_iterations": (int,),
    "node_throttle.max_consecutive_distraction_rounds": (int,),
    "node_throttle.max_consecutive_low_signal_rounds": (int,),
    "node_throttle.total_hard_cap_multiplier": (int,),
    "node_throttle.max_doc_review_cycles": (int,),
    "node_throttle.max_code_review_cycles": (int,),
    "node_throttle.max_discovery_iterations": (int,),
    # Phase G + Phase J — new end-of-session knobs.
    "node_throttle.max_end_of_session_regression_cycles": (int,),
    "node_throttle.end_of_session_repair_diagnostic_cap": (int,),
    "node_throttle.end_of_session_repair_inventory_cap": (int,),
    "node_throttle.end_of_session_force_reasoning_model": (bool,),
    # 2026-07-06 promotions.
    "node_throttle.stuck_target_limit": (int,),
    "node_throttle.generic_no_progress_limit": (int,),
    "node_throttle.same_missing_dep_limit": (int,),
    "security.hard_ceiling_multiplier": (int,),
    "fanout.max_concurrency": (int,),
    "fanout.timeout_seconds": (int, float),
    "persistence.db_path": (str,),
    "persistence.ttl_days": (int,),
    "persistence.redact_messages": (bool,),
    "model_routing.planning_primary": (str,),
    "model_routing.planning_mode": (str,),
    "model_routing.planning_fallback": (str,),
    "model_routing.planning_fallback_mode": (str,),
    "model_routing.patching_primary": (str,),
    "model_routing.patching_mode": (str,),
    "model_routing.patching_fallback": (str,),
    "model_routing.patching_fallback_mode": (str,),
    "model_routing.repair_primary": (str,),
    "model_routing.repair_fallback": (str,),
    "model_routing.repair_mode": (str,),
    "model_routing.repair_fallback_mode": (str,),
    "model_routing.doc_reviewer_primary": (str,),
    "model_routing.doc_reviewer_mode": (str,),
    "model_routing.doc_reviewer_fallback": (str,),
    "model_routing.doc_reviewer_fallback_mode": (str,),
    "model_routing.code_reviewer_primary": (str,),
    "model_routing.code_reviewer_mode": (str,),
    "model_routing.code_reviewer_fallback": (str,),
    "model_routing.code_reviewer_fallback_mode": (str,),
    "model_routing.ollama_local_model": (str,),
    "model_routing.ollama_local_backup": (str,),
    "model_routing.force_local_only": (bool,),
    "lintgate.format_modified_files": (bool,),
    "lintgate.strict_missing_formatter": (bool,),
    "logging.level": (str,),
    "logging.log_dir": (str,),
    "logging.json_stderr": (bool,),
    "logging.langsmith": (bool,),
    "logging.max_bytes": (int,),
    "logging.backup_count": (int,),
    "test_generation.enabled": (bool,),
    "test_generation.max_iterations": (int,),
    # Coverage gate for generated apps (FR-080). See _KNOWN_NESTED_KEYS.
    "coverage.min_pct": (int,),
    "coverage.enforce": (bool,),
    "speculative.enabled": (bool,),
    "speculative.num_variants": (int,),
    "speculative.temperature": (int, float),
    "speculative.selection_strategy": (str,),
    "speculative.worktree_base_dir": (str,),
    "speculative.trigger": (str,),
    "speculative.n_repair_failures_threshold": (int,),
    "speculative.diversity_mode": (str,),
    "speculative.cost_strategy": (str,),
    "speculative.salvage_strategy": (str,),
    "speculative.max_concurrency": (int,),
    "speculative.variant_models": (list,),
    "speculative.variant_prompt_styles": (list,),
    "speculative.expensive_model": (str,),
    "speculative.cheap_model": (str,),
    "speculative.voting": (dict,),
    "llm_dispatch.max_tokens_default": (int, str, type(None)),
    "llm_dispatch.max_tokens_per_role": (dict,),
    "llm_dispatch.continue_on_length": (dict,),
    "llm_dispatch.max_continuation_cycles": (int,),
    "llm_dispatch.patching_read_file_cap": (int,),
    "llm_dispatch.read_file_rounds": (int,),
    "llm_dispatch.prompt_cache_enabled": (bool,),
    "web_tools.enabled": (bool,),
    "web_tools.max_bytes": (int,),
    "web_tools.max_results": (int,),
    "web_tools.search_backend": (str,),
    "web_tools.api_key_env": (str,),
    "web_tools.allow_private_ips": (bool,),
    "web_tools.timeout_seconds": (int, float),
    "web_tools.tool_call_cap_per_dispatch": (int,),
    "web_tools.backends": (list,),
    "mcp.enabled": (bool,),
    "mcp.tool_call_timeout_seconds": (int, float),
    "mcp.command_allowlist": (list,),
    "mcp.allow_local_filesystem_servers": (bool,),
    "mcp.result_max_bytes": (int,),
    "mcp.servers": (list,),
    "skills.user_skills_dir": (str,),
    "memory.enabled": (bool,),
    "memory.dir": (str,),
    "memory.max_bytes": (int,),
    "memory.inject_max_bytes": (int,),
    "memory.compact_after": (int,),
    "memory.compact_keep_recent": (int,),
    "schedule.enabled": (bool,),
    "schedule.jobs": (list,),
    "schedule.history_db": (str,),
    "schedule.log_dir": (str,),
    "schedule.tick_seconds": (int,),
    "schedule.harness_binary": (str,),
    "dashboard.enabled": (bool,),
    "dashboard.host": (str,),
    "dashboard.port": (int,),
    "dashboard.token_env": (str,),
    "dashboard.log_dir": (str,),
    "dashboard.metrics_dir": (str,),
    "dashboard.memory_dir": (str,),
    "dashboard.repo_index_dir": (str,),
    "dashboard.schedule_db": (str,),
    "dashboard.static_dir": (str,),
    "dashboard.chart_js_url": (str,),
    "dashboard.sessions_max": (int,),
    "dashboard.writes_enabled": (bool,),
    "dashboard.csrf_token_env": (str,),
    "dashboard.hitl_webhook_secret": (str,),
    "dashboard.hitl_webhook_timeout_seconds": (int, float),
    "dashboard.audit_log_retention_days": (int,),
    "dashboard.web_db_path": (str,),
    "dashboard.config_path": (str,),
    "dashboard.carbon_css_url": (str,),
    "dashboard.carbon_js_url": (str,),
    "dashboard.docs_dir": (str,),
    "github.gh_path": (str,),
    "github.default_owner": (str,),
    "github.default_repo": (str,),
    "repo_index.enabled": (bool,),
    "repo_index.backend": (str,),
    "repo_index.top_k": (int,),
    "repo_index.chunk_lines": (int,),
    "repo_index.chunk_overlap": (int,),
    "repo_index.inject_max_bytes": (int,),
    "repo_index.index_dir": (str,),
    "repo_index.exclude_globs": (list,),
    "repo_index.text_extensions": (list,),
    "repo_index.max_file_bytes": (int,),
    "repo_index.openai_model": (str,),
    "repo_index.openai_api_base": (str,),
    "metrics.burn_rate_window_minutes": (int,),
    "metrics.metrics_dir": (str,),
    "deployment.enabled": (bool,),
    "deployment.compose_file": (str,),
    "deployment.health_check_interval_seconds": (int,),
    "deployment.health_check_timeout_seconds": (int,),
    "deployment_defaults.network": (dict,),
    "deployment_defaults.storage": (dict,),
    "deployment_defaults.secrets": (dict,),
    "deployment_defaults.infra_sync": (dict,),
    "agile": (bool,),
    "agile_defaults.batch_size": (int,),
    "agile_defaults.commit_on_story": (bool,),
    "agile_defaults.repair_cap": (int,),
    "core_languages.backend_language": (str,),
    "core_languages.web_language": (list,),
}

# model_routing fields that must reference an entry in `models` when
# non-empty. doc_reviewer_*, code_reviewer_*, ollama_local_* are opt-in
# (empty disables that role). planning/patching/repair are REQUIRED.
_REQUIRED_ROUTING_FIELDS: tuple[str, ...] = (
    "planning_primary", "patching_primary", "repair_primary",
)
_OPTIONAL_ROUTING_FIELDS: tuple[str, ...] = (
    "planning_fallback", "patching_fallback", "repair_fallback",
    "doc_reviewer_primary", "doc_reviewer_fallback",
    "code_reviewer_primary", "code_reviewer_fallback",
    "ollama_local_model", "ollama_local_backup",
)

# Sandbox backend whitelist — outside this set the SandboxExecutor doesn't
# know how to construct anything.
_VALID_SANDBOX_BACKENDS: frozenset[str] = frozenset({
    "auto", "docker", "unshare", "bare",
})

# Speculative-branching winner-pick strategies. Must match the choices
# handled in harness/speculative.py:_select_winner.
_VALID_SELECTION_STRATEGIES: frozenset[str] = frozenset({
    "first_success", "fewest_changes", "all_pass",
})

# Providers that DON'T need an API key env var (run locally / on-host).
# Anything else is treated as remote and gated on {PROVIDER}_API_KEY.
_LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama"})


def _suggest_model_key(bad_value: str, known_keys) -> str:
    """Return a ' Did you mean X?' hint when the operator dropped the provider
    prefix on a model key. Model keys are 'provider:model_id'; a bare
    'model_id' with an unambiguous provider match is the common typo. Returns
    an empty string when no confident single match exists."""
    if not isinstance(bad_value, str) or ":" in bad_value:
        return ""
    matches = [k for k in known_keys if isinstance(k, str) and k.split(":", 1)[-1] == bad_value]
    if len(matches) == 1:
        return f" Did you mean '{matches[0]}'? (Model keys must include the provider prefix, e.g. 'deepseek:deepseek-v4-flash'.)"
    return ""


# Locked core technology selection. Backend may be Python or Java
# (Spring Boot). The web stack is anchored on the React + TypeScript +
# TailwindCSS trio: all three MUST be present in
# ``core_languages.web_language``, and entries OUTSIDE the trio + a
# small allowlist of permitted React-ecosystem libraries are rejected
# at config-load time with a polite refusal message. These constants
# are the single source of truth for what the harness will accept —
# the validator, prompt scaffolds, and stack detectors all read from
# here.
_ALLOWED_BACKEND_LANGUAGES: tuple[str, ...] = ("Python", "Java")

# The three web entries that are mandatory in every supported web
# stack. The validator rejects any web_language list that omits one
# of these.
_REQUIRED_WEB_LANGUAGES: tuple[str, ...] = ("React", "TypeScript", "TailwindCSS")

# Additional React-ecosystem libraries the operator may opt into by
# adding the string to ``core_languages.web_language`` alongside the
# trio above. Grow this list cautiously — each entry implies the
# harness ships skills, style guides, and prompt scaffolds for that
# library. radix-ui is the unstyled-primitive component library
# Tailwind UI / shadcn-style stacks layer on top of TailwindCSS; it
# composes cleanly with the locked trio.
_OPTIONAL_WEB_LANGUAGES: tuple[str, ...] = ("radix-ui",)

# Every web_language value the validator considers in-bounds — the
# required trio plus the optional extras. Operators trying to add
# something outside this set get an exit-code-2 with a polite
# refusal message.
_ALLOWED_WEB_LANGUAGES: tuple[str, ...] = tuple(
    list(_REQUIRED_WEB_LANGUAGES) + list(_OPTIONAL_WEB_LANGUAGES)
)

_DEFAULT_BACKEND_LANGUAGE: str = "Python"
_DEFAULT_WEB_LANGUAGES: tuple[str, ...] = _REQUIRED_WEB_LANGUAGES


def resolve_core_languages(config: dict[str, Any]) -> dict[str, Any]:
    """Return the resolved ``core_languages`` block.

    Blank / missing values are replaced with the locked defaults
    (Python backend; React+TypeScript+TailwindCSS web). Any other
    value is left as-is for :func:`validate_config_strict` to flag.
    Callers (graph.py prompt scaffolds, impact.py stack filter, etc.)
    should always go through this helper so behaviour is consistent.
    """
    raw = config.get("core_languages") if isinstance(config, dict) else None
    block: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
    backend = block.get("backend_language")
    if not isinstance(backend, str) or not backend.strip():
        backend = _DEFAULT_BACKEND_LANGUAGE
    web = block.get("web_language")
    if not isinstance(web, list) or not web:
        web = list(_DEFAULT_WEB_LANGUAGES)
    return {"backend_language": backend, "web_language": web}


def find_missing_api_keys(config: dict[str, Any]) -> dict[str, list[str]]:
    """Return ``{env_var: [model_keys needing it]}`` for every model that's
    actually referenced in ``model_routing`` and whose provider is remote
    (not in :data:`_LOCAL_PROVIDERS`) and whose ``{PROVIDER}_API_KEY`` env
    var is unset and which doesn't supply an inline ``api_key`` in config.

    Shared by :func:`validate_config_strict` (to fail fast on missing keys)
    and the bare-flag setup wizard (to prompt for the keys interactively).
    Keeping the scan in one place means the validator and the wizard can't
    drift on which providers are "local" or how env-var names are derived.
    """
    models = config.get("models") or {}
    routing = config.get("model_routing") or {}
    if not isinstance(models, dict) or not isinstance(routing, dict):
        return {}

    referenced: set[str] = set()
    for field in (*_REQUIRED_ROUTING_FIELDS, *_OPTIONAL_ROUTING_FIELDS):
        val = routing.get(field, "")
        if isinstance(val, str) and val.strip() and val in models:
            referenced.add(val)

    missing: dict[str, list[str]] = {}
    for model_key in sorted(referenced):
        spec = models.get(model_key)
        if not isinstance(spec, dict):
            continue
        provider = spec.get("provider", "")
        if not isinstance(provider, str) or not provider.strip():
            continue
        if provider.lower() in _LOCAL_PROVIDERS:
            continue
        inline_key = spec.get("api_key", "")
        if isinstance(inline_key, str) and inline_key.strip():
            continue
        env_var = f"{provider.upper()}_API_KEY"
        if not os.environ.get(env_var, "").strip():
            missing.setdefault(env_var, []).append(model_key)
    return missing


def validate_config_strict(config: dict[str, Any], source: str) -> None:
    """Validate ``config`` strictly. Raise :class:`ConfigError` with a
    consolidated message listing EVERY problem found in one pass.

    Caller (cli entry points) catches ConfigError and exits 2 before any
    further initialization. The harness never runs with bad config.

    Checks performed (collect all, fail once):
      1. Unknown top-level keys (with difflib suggestions).
      2. Unknown nested keys inside known sections (with suggestions).
      3. Wrong types per :data:`_TYPE_SCHEMA`.
      4. Required fields present and non-empty:
         - ``models`` is a non-empty dict.
         - ``model_routing.planning_primary``, ``patching_primary``,
           ``repair_primary`` non-empty AND reference keys in ``models``.
         - Optional routing fields, if non-empty, reference keys in ``models``.
         - ``persistence.db_path`` non-empty.
         - ``token_budget.hard_cap_usd`` positive.
         - ``sandbox.backend`` in the valid set.
      5. Every model that will actually be used (i.e. referenced in
         ``model_routing``) whose provider is NOT in ``_LOCAL_PROVIDERS``
         has its ``{PROVIDER}_API_KEY`` env var set. Without this the
         harness would crash mid-run when the gateway tries to dispatch.
    """
    import difflib

    errors: list[str] = []

    # --- 1 + 2. Key validation ---
    for key, value in config.items():
        if key not in _KNOWN_TOP_LEVEL_KEYS:
            suggestion = difflib.get_close_matches(key, _KNOWN_TOP_LEVEL_KEYS, n=1, cutoff=0.6)
            hint = f" (did you mean '{suggestion[0]}'?)" if suggestion else ""
            errors.append(f"Unknown top-level key '{key}'{hint}")
            continue
        known_nested = _KNOWN_NESTED_KEYS.get(key)
        if known_nested is None or not isinstance(value, dict):
            continue
        for nested_key in value.keys():
            if nested_key in known_nested:
                continue
            suggestion = difflib.get_close_matches(nested_key, known_nested, n=1, cutoff=0.6)
            hint = f" (did you mean '{suggestion[0]}'?)" if suggestion else ""
            errors.append(f"Unknown nested key '{key}.{nested_key}'{hint}")

    # --- 3. Type validation ---
    for dotted_path, expected_types in _TYPE_SCHEMA.items():
        present, actual_value = _walk_dotted(config, dotted_path)
        if not present:
            continue
        # bool is a subclass of int in Python — exclude bool from int matches
        # unless bool is itself in the expected set.
        if isinstance(actual_value, bool) and bool not in expected_types:
            errors.append(
                f"'{dotted_path}' must be of type "
                f"{'/'.join(t.__name__ for t in expected_types)}, "
                f"got bool ({actual_value!r})"
            )
            continue
        if not isinstance(actual_value, expected_types):
            errors.append(
                f"'{dotted_path}' must be of type "
                f"{'/'.join(t.__name__ for t in expected_types)}, "
                f"got {type(actual_value).__name__} ({actual_value!r})"
            )

    # --- 4. Required fields ---
    # product_spec_dir is mandatory: the harness mandates a folder of
    # spec files (.txt / .md / .pdf) describing the product, and that
    # folder MUST live at the workspace root. We enforce both presence
    # and the bare-folder-name rule here so the operator gets an exit-2
    # at config-load time (before lock acquisition, gateway init,
    # GitGuardian, etc.) instead of a softer runtime failure later.
    # The folder-exists + non-empty-spec-file check is separate (cmd_run
    # does it after workspace_path is known).
    spec_dir = config.get("product_spec_dir")
    if spec_dir is None:
        errors.append(
            "'product_spec_dir' is required. Set a top-level string key in "
            "config.json with the NAME of a folder at the workspace root "
            "that holds the product-specification files (.txt / .md / "
            ".pdf). The name must be a bare folder name — no path "
            "separators, no absolute paths, no `..`. Example: "
            "\"product_spec_dir\": \"product_spec\"."
        )
    else:
        name_error = _validate_product_spec_dir_name(spec_dir)
        if name_error is not None:
            errors.append(f"'product_spec_dir' {name_error}.")

    # change_requests_dir is OPTIONAL — defaults to "change_requests" — but
    # if the operator sets it explicitly, the value must obey the same
    # bare-folder-name rules as product_spec_dir.
    cr_dir = config.get("change_requests_dir")
    if cr_dir is not None:
        cr_name_error = _validate_product_spec_dir_name(cr_dir)
        if cr_name_error is not None:
            errors.append(f"'change_requests_dir' {cr_name_error}.")

    models = config.get("models")
    if not isinstance(models, dict) or not models:
        errors.append(
            "'models' must contain at least one entry. Declare every model "
            "the harness should know about (provider, model_id, costs, etc.)."
        )
        models = {}

    routing = config.get("model_routing")
    if not isinstance(routing, dict):
        errors.append("'model_routing' must be an object with role → model mappings.")
        routing = {}

    for field in _REQUIRED_ROUTING_FIELDS:
        val = routing.get(field, "")
        if not isinstance(val, str) or not val.strip():
            errors.append(
                f"'model_routing.{field}' is required and must reference "
                f"a key in 'models' (e.g. 'openai:gpt-4o')."
            )
            continue
        if val not in models:
            hint = _suggest_model_key(val, models.keys())
            errors.append(
                f"'model_routing.{field}' references unknown model "
                f"'{val}'.{hint} Declare it under 'models' or pick one of: "
                f"{sorted(models.keys())}"
            )

    for field in _OPTIONAL_ROUTING_FIELDS:
        val = routing.get(field, "")
        if isinstance(val, str) and val.strip() and val not in models:
            hint = _suggest_model_key(val, models.keys())
            errors.append(
                f"'model_routing.{field}' is set to '{val}' but no model "
                f"by that key exists in 'models'.{hint} Either declare it or "
                f"set the field to an empty string to disable."
            )

    persistence = config.get("persistence", {})
    if isinstance(persistence, dict):
        db_path = persistence.get("db_path", "")
        if not isinstance(db_path, str) or not db_path.strip():
            errors.append(
                "'persistence.db_path' is required and must be a non-empty path."
            )

    token_budget = config.get("token_budget", {})
    if isinstance(token_budget, dict):
        hard_cap = token_budget.get("hard_cap_usd")
        if isinstance(hard_cap, bool) or not isinstance(hard_cap, (int, float)) or hard_cap <= 0:
            errors.append(
                "'token_budget.hard_cap_usd' is required and must be a "
                "positive number (USD budget cap per session)."
            )

    sandbox = config.get("sandbox", {})
    if isinstance(sandbox, dict):
        backend = sandbox.get("backend", "")
        if not isinstance(backend, str) or backend not in _VALID_SANDBOX_BACKENDS:
            errors.append(
                f"'sandbox.backend' must be one of "
                f"{sorted(_VALID_SANDBOX_BACKENDS)}, got {backend!r}."
            )

    # Speculative branching: validate the strategy enum and sensible
    # ranges on the two numeric knobs. Section is optional — when absent
    # the harness uses the historical defaults (3 variants, 0.3 temp,
    # first_success); when present every key gets strict-checked.
    spec_cfg = config.get("speculative", {})
    if isinstance(spec_cfg, dict):
        strategy = spec_cfg.get("selection_strategy")
        if isinstance(strategy, str) and strategy not in _VALID_SELECTION_STRATEGIES:
            errors.append(
                f"'speculative.selection_strategy' must be one of "
                f"{sorted(_VALID_SELECTION_STRATEGIES)}, got {strategy!r}."
            )
        temp = spec_cfg.get("temperature")
        if isinstance(temp, (int, float)) and not isinstance(temp, bool):
            if temp < 0 or temp > 1.5:
                errors.append(
                    f"'speculative.temperature' must be in [0.0, 1.5] "
                    f"(0.2-0.4 recommended for code), got {temp}."
                )
        n_var = spec_cfg.get("num_variants")
        if isinstance(n_var, int) and not isinstance(n_var, bool):
            if n_var < 1 or n_var > 10:
                errors.append(
                    f"'speculative.num_variants' must be in [1, 10] "
                    f"(3 recommended), got {n_var}."
                )

    # LLM dispatch: a blank value (None / "" / 0 / missing key) means
    # "no per-call max_tokens" — the gateway omits the parameter and the
    # provider's own per-request output cap takes over. A value that IS
    # provided is clamped to [256, 32768]: below 256 → useless truncated
    # replies; above 32768 → blows past per-request output caps on every
    # supported model. Role names in max_tokens_per_role aren't enumerated
    # here — unknown roles get silently ignored at dispatch time so
    # adding a new NodeRole doesn't require a validator update.
    dispatch_cfg = config.get("llm_dispatch", {})
    if isinstance(dispatch_cfg, dict):
        _MIN_MAX_TOKENS = 256
        _MAX_MAX_TOKENS = 32768

        def _validate_max_tokens(dotted: str, raw: Any) -> None:
            if raw is None or raw == 0 or raw == "":
                return
            if isinstance(raw, str):
                errors.append(
                    f"'{dotted}' must be an int, null, or blank, "
                    f"got string {raw!r}."
                )
                return
            if not isinstance(raw, int) or isinstance(raw, bool):
                errors.append(
                    f"'{dotted}' must be an int, got {type(raw).__name__}."
                )
                return
            if raw < _MIN_MAX_TOKENS or raw > _MAX_MAX_TOKENS:
                errors.append(
                    f"'{dotted}' must be in "
                    f"[{_MIN_MAX_TOKENS}, {_MAX_MAX_TOKENS}], got {raw}."
                )

        _validate_max_tokens(
            "llm_dispatch.max_tokens_default",
            dispatch_cfg.get("max_tokens_default"),
        )
        per_role = dispatch_cfg.get("max_tokens_per_role", {})
        if isinstance(per_role, dict):
            for role_name, role_mt in per_role.items():
                if not isinstance(role_name, str) or not role_name.strip():
                    errors.append(
                        f"'llm_dispatch.max_tokens_per_role' keys must be "
                        f"non-empty role-name strings, got {role_name!r}."
                    )
                    continue
                _validate_max_tokens(
                    f"llm_dispatch.max_tokens_per_role.{role_name}",
                    role_mt,
                )

        continue_map = dispatch_cfg.get("continue_on_length", {})
        if isinstance(continue_map, dict):
            # Validate role names against the NodeRole enum so typos
            # like "planing" don't silently no-op (audit §5.13).
            try:
                from harness.gateway import NodeRole as _NodeRole
                valid_role_names = {r.value for r in _NodeRole}
            except Exception:  # noqa: BLE001 — fail open if import breaks
                valid_role_names = set()
            for role_name, role_flag in continue_map.items():
                if not isinstance(role_name, str) or not role_name.strip():
                    errors.append(
                        f"'llm_dispatch.continue_on_length' keys must be "
                        f"non-empty role-name strings, got {role_name!r}."
                    )
                    continue
                if not isinstance(role_flag, bool):
                    errors.append(
                        f"'llm_dispatch.continue_on_length.{role_name}' "
                        f"must be a bool, got {type(role_flag).__name__}."
                    )
                if valid_role_names and role_name not in valid_role_names:
                    import difflib as _difflib
                    suggestion = _difflib.get_close_matches(
                        role_name, valid_role_names, n=1, cutoff=0.6,
                    )
                    hint = f" (did you mean {suggestion[0]!r}?)" if suggestion else ""
                    errors.append(
                        f"'llm_dispatch.continue_on_length.{role_name}' "
                        f"is not a known NodeRole value{hint}. "
                        f"Valid: {sorted(valid_role_names)}"
                    )

    # --- 4b. Locked core-technology selection ---
    # Backend may be Python or Java (Spring Boot). Web is fixed to the
    # React + TypeScript + TailwindCSS trio. Blanks are auto-defaulted
    # via resolve_core_languages(); anything else is a hard error so the
    # operator can't silently smuggle in Go / Vue / Angular / Flutter
    # workloads the harness no longer ships skills for.
    raw_lang_block = config.get("core_languages")
    lang_block: dict[str, Any] = (
        raw_lang_block if isinstance(raw_lang_block, dict) else {}
    )
    backend_lang = lang_block.get("backend_language")
    if backend_lang is None or (isinstance(backend_lang, str) and not backend_lang.strip()):
        pass  # blank → resolve_core_languages() will default to Python
    elif not isinstance(backend_lang, str):
        errors.append(
            f"'core_languages.backend_language' must be a string, got "
            f"{type(backend_lang).__name__}. Please set it to one of: "
            f"{list(_ALLOWED_BACKEND_LANGUAGES)} — or leave it blank to "
            f"accept the default ({_DEFAULT_BACKEND_LANGUAGE!r})."
        )
    elif backend_lang not in _ALLOWED_BACKEND_LANGUAGES:
        errors.append(
            f"'core_languages.backend_language' is set to {backend_lang!r}, "
            f"which is not a supported backend language. This harness only "
            f"supports {list(_ALLOWED_BACKEND_LANGUAGES)} for backend work "
            f"(Java implies Spring Boot). Please update config.json to one "
            f"of the supported values, or leave the field blank to accept "
            f"the default ({_DEFAULT_BACKEND_LANGUAGE!r})."
        )

    web_lang = lang_block.get("web_language")
    if web_lang is None or (isinstance(web_lang, list) and not web_lang):
        pass  # blank → resolve_core_languages() will default to the trio
    elif not isinstance(web_lang, list):
        errors.append(
            f"'core_languages.web_language' must be a list, got "
            f"{type(web_lang).__name__}. Please set it to "
            f"{list(_ALLOWED_WEB_LANGUAGES)} — or leave it blank to accept "
            f"the default ({list(_DEFAULT_WEB_LANGUAGES)})."
        )
    else:
        # Order-insensitive set comparison. The React + TypeScript +
        # TailwindCSS trio MUST be present; additional entries are
        # allowed only when they appear in _OPTIONAL_WEB_LANGUAGES.
        # Anything else is rejected at load time so an operator can't
        # silently smuggle in Vue / Angular / Svelte / etc.
        try:
            web_set = {item for item in web_lang if isinstance(item, str)}
        except TypeError:
            web_set = set()
        required_set = set(_REQUIRED_WEB_LANGUAGES)
        allowed_set = set(_ALLOWED_WEB_LANGUAGES)
        non_string = [item for item in web_lang if not isinstance(item, str)]
        if non_string:
            errors.append(
                f"'core_languages.web_language' must contain only strings, "
                f"got non-string entries {non_string!r}. Required entries: "
                f"{list(_REQUIRED_WEB_LANGUAGES)}; optional extras: "
                f"{list(_OPTIONAL_WEB_LANGUAGES)}."
            )
        else:
            missing = sorted(required_set - web_set)
            extra = sorted(web_set - allowed_set)
            if missing:
                errors.append(
                    f"'core_languages.web_language' is missing required "
                    f"entries {missing}. The web stack must include the "
                    f"full trio {list(_REQUIRED_WEB_LANGUAGES)} — please "
                    f"add the missing entries, or leave the field blank "
                    f"to accept the trio as the default."
                )
            if extra:
                errors.append(
                    f"'core_languages.web_language' contains unsupported "
                    f"entries {extra}. This harness only supports the "
                    f"React + TypeScript + TailwindCSS trio plus the "
                    f"optional extras {list(_OPTIONAL_WEB_LANGUAGES)} for "
                    f"web work — please remove the unsupported entries, "
                    f"or leave the field blank to accept the default "
                    f"({list(_DEFAULT_WEB_LANGUAGES)})."
                )

    # --- 5. Env var presence for every model referenced by routing ---
    referenced_models: set[str] = set()
    for field in (*_REQUIRED_ROUTING_FIELDS, *_OPTIONAL_ROUTING_FIELDS):
        val = routing.get(field, "")
        if isinstance(val, str) and val.strip() and val in models:
            referenced_models.add(val)

    # Flag any referenced model with an empty/non-string provider — find_missing_api_keys
    # skips these silently (it can't compute an env var name), so the validator owns
    # this error message.
    for model_key in sorted(referenced_models):
        spec = models.get(model_key)
        if not isinstance(spec, dict):
            continue
        provider = spec.get("provider", "")
        if not isinstance(provider, str) or not provider.strip():
            errors.append(
                f"'models.{model_key}.provider' is missing or empty. Set "
                f"it to one of: openai, anthropic, deepseek, ollama, …"
            )

    missing_env = find_missing_api_keys(config)
    if missing_env:
        details = "; ".join(
            f"{env_var} (required by model{'s' if len(keys) > 1 else ''}: "
            f"{', '.join(keys)})"
            for env_var, keys in sorted(missing_env.items())
        )
        errors.append(
            f"Missing API key environment variable(s): {details}. "
            f"Export the env var(s) before re-running the harness — "
            f"the harness reads keys from the environment, never from "
            f"the config file. Example: `export {sorted(missing_env)[0]}=\"sk-...\"`."
        )

    # --- Raise if any error collected ---
    if errors:
        bullet = "\n  - "
        raise ConfigError(
            f"Configuration error{'s' if len(errors) > 1 else ''} in {source}:"
            f"{bullet}{bullet.join(errors)}\n\n"
            f"Fix the config file before re-running the harness. "
            f"The harness will not proceed with invalid configuration."
        )


def _walk_dotted(config: dict[str, Any], dotted_path: str) -> tuple[bool, Any]:
    """Return ``(present, value)`` for a dotted path inside ``config``.

    ``present=False`` means the path was absent (caller should skip type
    check — type schema only flags WRONG types, not missing optional
    fields; required-presence is enforced separately).
    """
    cur: Any = config
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def _makefile_has_target(workspace_path: str, target: str) -> bool:
    """Return True if any Makefile / makefile / GNUmakefile at the
    workspace root declares ``<target>:`` as a Make target.

    Used by :func:`_detect_default_build_command` to avoid the
    `make build` → exit 127 crash that hits when an LLM-generated
    Makefile is present but doesn't declare a ``build:`` target — the
    sandbox tries ``make build``, ``make`` reports "No rule to make
    target 'build'", and (on ``ubuntu:22.04``, the default bare image)
    `make` isn't even installed, producing a 0.21-second exit-127 with
    zero diagnostics for the repair LLM to act on. See session
    `b6fe5c6e` for the worked example.

    Target lines in Make grammar live at the start of a line: ``build:``
    (or ``build: deps``). Recipe lines (tab-indented) underneath
    aren't the target itself. We match ``^<target>:`` with the
    multiline flag. Reads files best-effort; on I/O error we treat the
    Makefile as absent and let the caller fall through.
    """
    import re as _re
    for name in ("Makefile", "makefile", "GNUmakefile"):
        path = os.path.join(workspace_path, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError:
            continue
        if _re.search(rf"^{_re.escape(target)}:", content, _re.MULTILINE):
            return True
    return False


# Canonical pytest invocation used by every build-command builder. Chosen
# so test failures carry enough context for the repair LLM (and the
# reflection judge) to ground without an exploration round. ``--tb=long``
# emits the full traceback instead of pytest's default ``auto`` (which
# collapses non-leaf frames); ``--showlocals`` prints the values of local
# variables at each frame — essential for compound assertions like
# ``assert any(c["ticker"] == "GOOGL" for c in results)`` where the
# assert-rewriter can't decompose the generator. ``-vv`` shows one test
# name per line so the per-failure parser can attribute frames correctly.
# The previous ``-q`` was capturing dots-only output that left the judge
# repeatedly emitting "insufficient data — investigate <file>" because
# the captured ``AssertionError`` had no traceback values.
#
# ``--timeout=30 --timeout-method=thread`` (pytest-timeout) is what turns
# LLM-generated infinite loops into an actionable traceback instead of a
# 5-minute sandbox SIGKILL that produces zero diagnostic. Session
# 6de334c3 lost 10+ minutes to a
# ``while True: ... await asyncio.sleep(sleep_time)`` in a rate limiter:
# pytest hung, sandbox timed out at 300s, exit code was -9, the repair
# LLM saw nothing and generated another hanging test. With
# ``--timeout=30`` pytest kills the offending test at 30s and prints a
# ``__________ Timeout __________`` block naming the exact file:line the
# thread was stuck on. ``method=thread`` (vs the default ``signal``) is
# necessary because ``signal`` fails on threads other than the main
# thread — irrelevant here since our sandbox pytest always runs main,
# but no downside and the thread method also works inside asyncio-run
# tests where signals would be intercepted by the loop.
#
# The plugin is baked into ``harness-builder:latest`` (see
# ``Dockerfile.builder``) so no runtime install is needed. Older/custom
# images that lack it will error out with "unrecognized argument
# --timeout" — operators can fix that with a one-line
# ``python3 -m pip install pytest-timeout`` in their build command.
_PYTEST_RUN = (
    "python3 -m pytest -vv --tb=long --showlocals "
    "--timeout=30 --timeout-method=thread"
)


_SUBDIR_BUILD_SKIP = frozenset({
    "node_modules", "__pycache__", ".git", "build", "dist", "venv", ".venv",
    "env", "target", "out", "docs", "product_spec", "change_requests",
})


# Matches `cd <SUBDIR> && npm ...` and `cd <SUBDIR> && npx ...` patterns
# in package.json `scripts` values. Used by _compose_node_build_command
# to detect root-delegating-to-subdir layouts so we can install the
# subdir's deps before the root build runs.
_CD_SUBDIR_NPM_RE = re.compile(
    r"\bcd\s+([A-Za-z0-9_\-./]+)\s*&&\s*(?:npm|npx|yarn|pnpm)\b"
)

# Bash metacharacters that disqualify a captured subdir name from being
# safely re-emitted as ``cd <name>``. The regex above already excludes
# spaces and obvious specials, but defense-in-depth so we never inject
# something like ``cd .;rm -rf .`` into the build command.
_UNSAFE_SUBDIR_CHARS = set(";|&`$<>(){}[]\\\"' \t\n")


def _extract_delegate_subdirs(scripts: dict) -> list[str]:
    """Return a stable-ordered list of first-level subdirectories that the
    root ``scripts.{build,test,dev,start}`` values delegate to via a
    ``cd <subdir> && npm/npx/yarn/pnpm ...`` shell idiom.

    Common with the ``client/`` + ``server/`` monorepo layout where the
    root package.json acts as an orchestrator (``"build": "cd client &&
    npm run build"``) and the actual deps live in the subdirs. Without
    a subdir-install step, the root ``npm install`` only materialises
    root-level devDeps (e.g. ``concurrently``) and the subdir build
    immediately crashes with 100+ "Cannot find module" errors — exactly
    the symptom from session 7c30bce2.

    Excludes ``.`` / ``..`` / absolute paths and anything containing
    bash metacharacters so the result is safe to inline into a ``cd``.
    """
    if not isinstance(scripts, dict):
        return []
    found: list[str] = []
    for key in ("build", "test", "dev", "start"):
        value = scripts.get(key)
        if not isinstance(value, str):
            continue
        for raw in _CD_SUBDIR_NPM_RE.findall(value):
            name = raw.strip().strip("/")
            if not name or name in {".", ".."}:
                continue
            if name.startswith("/") or name.startswith("~"):
                continue
            if any(ch in _UNSAFE_SUBDIR_CHARS for ch in name):
                continue
            # First-level subdirs only — nested paths (``packages/app``)
            # are also valid, but we only emit a single ``cd <name>`` so
            # capture the full relative path verbatim.
            if name not in found:
                found.append(name)
    return found


def _compose_node_build_command(package_json_path: str, *, prefix: str = "") -> str:
    """Build the Node command for a single ``package.json``.

    Default Vite scaffolds (``npm create vite@latest`` →
    ``dev``/``build``/``preview``/``lint``) have no ``test`` script.
    Emitting ``npm test`` against them exits 1 with
    ``Error: no test specified`` and traps the repair loop in a
    non-bug. To avoid that, peek into ``package.json`` and pick the
    right tail:

      - ``scripts.test`` defined → ``npm test``
      - ``vitest`` in deps / devDeps → ``npx vitest run``
      - otherwise → ``npm test --if-present`` (silent no-op, exit 0)

    For monorepos where the root ``scripts.build`` delegates via
    ``cd <subdir> && npm run build``, we ALSO emit ``cd <subdir> &&
    npm install`` for each referenced subdir before the root install.
    Without this, root ``npm install`` materialises only root devDeps;
    the subdir build crashes immediately with missing-module errors.
    npm workspaces (declared via ``"workspaces": [...]``) are handled
    by npm itself — root ``npm install`` installs workspace deps too —
    so we skip the explicit subdir installs in that case to avoid
    redundant work.

    ``prefix`` is used by the subdir probe to emit ``cd <dir> && ...``
    forms. Returns the full command string ready to drop into the
    detector.
    """
    has_test_script = False
    has_vitest = False
    has_workspaces = False
    delegate_subdirs: list[str] = []
    pkg_dir = os.path.dirname(package_json_path)
    try:
        import json as _json
        with open(package_json_path, "r", encoding="utf-8", errors="replace") as f:
            data = _json.load(f) or {}
        scripts = data.get("scripts") or {}
        if isinstance(scripts, dict) and "test" in scripts:
            has_test_script = True
        deps = {}
        for key in ("dependencies", "devDependencies"):
            section = data.get(key)
            if isinstance(section, dict):
                deps.update(section)
        if "vitest" in deps:
            has_vitest = True
        workspaces = data.get("workspaces")
        if isinstance(workspaces, (list, dict)) and workspaces:
            has_workspaces = True
        # Only collect delegations when not a workspaces project — npm
        # workspaces installs subdir deps automatically at root install.
        if not has_workspaces and isinstance(scripts, dict):
            for sub in _extract_delegate_subdirs(scripts):
                # Sanity-check: the subdir must actually exist under the
                # package's dir AND contain its own package.json. Without
                # this guard a stale script reference would emit a `cd`
                # that fails with "No such file or directory" and trap
                # the repair loop on a non-bug. ``pkg_dir`` may be empty
                # for relative paths — fall back to "." in that case.
                base = pkg_dir or "."
                sub_pkg = os.path.join(base, sub, "package.json")
                if os.path.isfile(sub_pkg):
                    delegate_subdirs.append(sub)
    except (OSError, ValueError):
        # Malformed JSON / unreadable file → fall through to the safe
        # `--if-present` tail. Better than crashing the detector.
        pass
    if has_test_script:
        tail = "npm test"
    elif has_vitest:
        tail = "npx vitest run"
    else:
        # Silent no-op: exits 0 when scripts.test is absent. Keeps the
        # build green for a freshly-scaffolded Vite app the LLM hasn't
        # wired vitest into yet, instead of trapping repair on a
        # non-bug.
        tail = "npm test --if-present"
    # Subdir installs go FIRST so the root build (which `cd`s into them)
    # finds populated node_modules. Each step pushes/pops the directory
    # with a subshell to avoid leaking ``cd`` state into the root step.
    subdir_install = "".join(
        f"(cd {sub} && npm install) && " for sub in delegate_subdirs
    )
    return f"{prefix}{subdir_install}npm install && npm run build && {tail}"


def _detect_subdir_build_command(workspace_path: str) -> Optional[str]:
    """Probe first-level subdirectories for a recognised manifest when
    the workspace root has none. Returns a ``cd <subdir> && ...`` form
    that installs deps and runs the test command, or ``None`` if no
    subdir manifest is found.

    Monorepo layouts (``server/requirements.txt`` + ``client/package.json``)
    are the canonical case — the LLM puts the backend in one subdir
    and the frontend in another. We prefer the FIRST subdir with a
    Python or Java manifest (backend tests are the smoke target);
    when no backend manifest is found we fall back to the FIRST Node
    subdir (pure-frontend monorepo). Probed alphabetically for
    determinism.
    """
    try:
        entries = sorted(os.listdir(workspace_path))
    except OSError:
        return None
    # Lazy import — `harness.graph` imports `harness.cli` at module load,
    # so importing graph at cli's top level would deadlock the bootstrap.
    from harness.graph import _uv_venv_prefix
    # Pass 1: backend-first (Python > Java). The smoke check imports
    # production Python modules, so a workspace with backend code gets
    # that path first.
    node_fallback: Optional[tuple[str, str]] = None
    for entry in entries:
        if entry.startswith(".") or entry in _SUBDIR_BUILD_SKIP:
            continue
        full = os.path.join(workspace_path, entry)
        if not os.path.isdir(full):
            continue
        # Probe order mirrors the root probe: pyproject > requirements
        # > Maven > Gradle. uv pip install is a drop-in for pip install
        # (same manifest semantics) but 10-30× faster on cold caches and
        # uses the harness-managed /cache/uv volume across runs. uv is
        # pre-baked into the sandbox builder image. Each install command
        # is prefixed with `_uv_venv_prefix()` so it writes to a user-
        # writable venv instead of /usr/local/lib/python3.11/dist-packages
        # — non-root sandbox users can't write there.
        if os.path.isfile(os.path.join(full, "pyproject.toml")):
            # Install steps run inside a subshell so `cd {entry}` does NOT leak
            # into the pytest invocation. Running pytest from `{entry}/` breaks
            # LLM-generated tests that import via the full subdir path (e.g.
            # `from server.app.config import ...`) — from that CWD `server`
            # isn't on sys.path and pytest exits 4 with `UsageError` when the
            # conftest fails at rootdir discovery. Keeping pytest at workspace
            # root means rootdir walking adds the workspace to sys.path and
            # `server.*` resolves as expected.
            dev_step = (
                f" && (cd {entry} && uv pip install -r requirements-dev.txt)"
                if os.path.isfile(os.path.join(full, "requirements-dev.txt"))
                else ""
            )
            return (
                f"{_uv_venv_prefix()} && (cd {entry} && uv pip install -e .)"
                f"{dev_step} && {_PYTEST_RUN}"
            )
        if os.path.isfile(os.path.join(full, "requirements.txt")):
            dev_step = (
                f" && (cd {entry} && uv pip install -r requirements-dev.txt)"
                if os.path.isfile(os.path.join(full, "requirements-dev.txt"))
                else ""
            )
            return (
                f"{_uv_venv_prefix()} && (cd {entry} && uv pip install -r requirements.txt)"
                f"{dev_step} && {_PYTEST_RUN}"
            )
        if os.path.isfile(os.path.join(full, "pom.xml")):
            return f"cd {entry} && mvn -B test"
        if os.path.isfile(os.path.join(full, "gradlew")):
            return f"cd {entry} && ./gradlew test"
        if (os.path.isfile(os.path.join(full, "build.gradle"))
                or os.path.isfile(os.path.join(full, "build.gradle.kts"))):
            return f"cd {entry} && gradle test"
        # Record (but don't return yet) the first Node subdir so a
        # pure-frontend monorepo can fall back to it after the backend
        # probe finishes empty.
        pkg = os.path.join(full, "package.json")
        if node_fallback is None and os.path.isfile(pkg):
            node_fallback = (entry, pkg)
    # Pass 2: frontend-only fallback. Pure-React/Vite repos with no
    # root manifest used to fall through to the bare-pytest fallback,
    # which made no sense for a Node project. Emit the same package.json
    # command we'd use at root, just prefixed with `cd <subdir> &&`.
    if node_fallback is not None:
        entry, pkg = node_fallback
        return _compose_node_build_command(pkg, prefix=f"cd {entry} && ")
    return None


def _detect_default_build_command(
    workspace_path: str,
    *,
    is_greenfield: bool = False,
) -> Optional[str]:
    """Pick a build command by sniffing workspace markers for the locked
    core stack.

    Only Python / Java / React+TypeScript+TailwindCSS+Vite are
    supported. Probed in priority order.

    **Greenfield vs brownfield Makefile handling**: in brownfield runs
    (``teane patch`` on an existing repo) we still respect a Makefile
    with a ``build:`` target — the operator's own build wiring may
    perform codegen, asset compilation, or other steps the harness
    can't infer. In greenfield runs (``teane build`` / ``--new-build``)
    the LLM is the sole author of the workspace; an LLM-scaffolded
    Makefile is informational, not load-bearing, so we ALWAYS emit the
    baselined per-stack command. This kills the bug class where the
    LLM emitted a Makefile whose ``build:`` target did the wrong thing
    (or no install at all) and the harness then deferred to it instead
    of running the project's actual ``pip install -r requirements.txt
    && pytest``.

    Returns ``None`` only when the workspace truly contains no marker
    for any supported stack — that case is treated as a brand-new
    Python scaffold (the harness's default) and the caller seeds a
    pip+pytest bootstrap command.
    """
    if not workspace_path or not os.path.isdir(workspace_path):
        return None
    # Lazy import — see _detect_subdir_build_command for the rationale.
    from harness.graph import _uv_venv_prefix

    def has(name: str) -> bool:
        return os.path.exists(os.path.join(workspace_path, name))

    # Brownfield only — see docstring. A Makefile in a greenfield
    # workspace was emitted by the patching LLM and must not be allowed
    # to override the deterministic per-stack baseline.
    if not is_greenfield and _makefile_has_target(workspace_path, "build"):
        return "make build"
    # Python — pyproject.toml > requirements.txt > any .py file.
    # uv pip install is preferred over plain pip — same manifest semantics,
    # 10-30× faster on cold caches, hits the harness-managed /cache/uv
    # volume across runs. uv is pre-baked into the sandbox builder image.
    #
    # Path A — install the UNION of every Python manifest in the workspace
    # (root pyproject/requirements + every first-level subdir's
    # requirements/pyproject/requirements-dev) rather than only the root
    # manifest. Structurally prevents the "manifest topology mismatch"
    # class of bug (finsearch STORY-038: sqlalchemy was in
    # root/requirements.txt but the build_command only installed
    # server/requirements.txt). Composed via
    # ``_compose_prod_smoke_install_step`` — the same helper the prod-
    # smoke check already uses, so install-time behaviour is aligned
    # across both entry points. The chain includes an ending
    # ``uv pip install pytest`` which harmlessly no-ops when pytest is
    # already installed by an earlier step.
    if has("pyproject.toml") or has("requirements.txt"):
        try:
            from harness.graph import _compose_prod_smoke_install_step
            _union_install = _compose_prod_smoke_install_step(workspace_path)
        except Exception:  # noqa: BLE001 — composer is best-effort; fall through on any failure
            _union_install = None
        if _union_install:
            return f"{_union_install} && {_PYTEST_RUN}"
        # Fallback: composer returned None despite root manifest existing
        # (transient FS error, symlink loop). Use the historical narrow
        # install so the build can at least attempt to run.
        if has("pyproject.toml"):
            dev_step = " && uv pip install -r requirements-dev.txt" if has("requirements-dev.txt") else ""
            return f"{_uv_venv_prefix()} && uv pip install -e .{dev_step} && {_PYTEST_RUN}"
        dev_step = " && uv pip install -r requirements-dev.txt" if has("requirements-dev.txt") else ""
        return f"{_uv_venv_prefix()} && uv pip install -r requirements.txt{dev_step} && {_PYTEST_RUN}"
    # Java — Maven first, then Gradle (wrapper if present).
    if has("pom.xml"):
        return "mvn -B test"
    if has("gradlew"):
        return "./gradlew test"
    if has("build.gradle") or has("build.gradle.kts"):
        return "gradle test"
    # Web (React + TypeScript + TailwindCSS, Vite-built). package.json
    # is the only allowed Node manifest — any other Node framework is
    # outside the supported stack. The compose helper peeks at
    # scripts.test / vitest in deps so a freshly-scaffolded Vite app
    # without a `test` script doesn't trap the repair loop on a
    # non-bug (`Error: no test specified`).
    if has("package.json"):
        return _compose_node_build_command(
            os.path.join(workspace_path, "package.json"),
        )
    # Monorepo layout: nothing at workspace root, but a first-level
    # subdirectory (e.g. ``server/``) carries the manifest. Common when
    # the LLM scaffolds a split backend/frontend layout — without this
    # probe the detector falls through to the bare-pytest fallback and
    # the project's actual deps are never installed.
    subdir_cmd = _detect_subdir_build_command(workspace_path)
    if subdir_cmd:
        return subdir_cmd
    # Last-chance Python heuristic: any .py file (top level OR one
    # level deep, e.g. ``app/__init__.py`` after LLM scaffolds a
    # package) → run pytest. Pytest is pre-baked into the sandbox
    # builder image so the legacy `pip install pytest &&` prefix is
    # unnecessary. The legacy substring is kept in the fallback below
    # only for the (rare) case where the harness runs outside the
    # builder image and the operator hasn't pre-installed pytest.
    fallback = f"{_uv_venv_prefix()} && uv pip install pytest && {_PYTEST_RUN}"
    try:
        for entry in os.listdir(workspace_path):
            if entry.endswith(".py"):
                return fallback
            full = os.path.join(workspace_path, entry)
            if os.path.isdir(full) and not entry.startswith("."):
                try:
                    if any(child.endswith(".py") for child in os.listdir(full)):
                        return fallback
                except OSError:
                    continue
    except OSError:
        pass
    return None


def resolve_build_command(
    config: dict[str, Any],
    workspace_path: Optional[str] = None,
    *,
    is_greenfield: bool = False,
) -> str:
    """Auto-wire the build command from the workspace and the locked
    core-language selection.

    Since the harness's supported stacks are fixed (Python / Java /
    React+TypeScript+TailwindCSS), the operator no longer chooses a
    build command — the workspace + ``core_languages.backend_language``
    fully determine it. Order of preference:

      1. Existing build markers — manifest sniff in
         :func:`_detect_default_build_command` (lets ``teane patch``
         reuse whatever build wiring the workspace already has).
      2. Greenfield seed — when no marker exists yet, fall back to a
         pip+pytest bootstrap for Python backends, ``mvn -B test`` for
         Java backends (so the very first compile in a fresh workspace
         doesn't exit-127 before the patcher writes the manifest).

    ``is_greenfield`` is threaded into the detector so an LLM-scaffolded
    ``Makefile`` in a ``--new-build`` run cannot hijack the build
    command. See :func:`_detect_default_build_command` for the
    greenfield/brownfield contract.
    """
    if workspace_path:
        detected = _detect_default_build_command(
            workspace_path, is_greenfield=is_greenfield,
        )
        if detected:
            logger.info(
                "[cli] Auto-wired build command from workspace markers: %s",
                detected,
            )
            return detected
    backend = resolve_core_languages(config)["backend_language"]
    seed = (
        "mvn -B test"
        if backend == "Java"
        # Install pytest-timeout alongside pytest so ``_PYTEST_RUN``'s
        # ``--timeout=30`` flag doesn't error out with "unrecognized
        # argument" on the very first compile — before any venv or
        # manifest is wired up.
        else f"python3 -m pip install pytest pytest-timeout && {_PYTEST_RUN}"
    )
    logger.info(
        "[cli] Workspace has no build markers yet; seeding build command "
        "for %s backend: %s",
        backend, seed,
    )
    return seed


# ---------------------------------------------------------------------------
# 2. HITL Interactive Menu Loop
# ---------------------------------------------------------------------------

# --hitl-* flag pin: cmd_run calls _set_hitl_flags(...) once at startup
# with the values returned by _resolve_hitl_flags — which threads the
# three-tier precedence (CLI > config.json > True) so each gate
# callsite below can ask "should I prompt?" without re-reading args.
# Tests / direct calls that never pin leave the map empty; the
# _hitl_gate_enabled accessor then defaults to True so legacy unit tests
# (which install a fake channel and expect prompts to fire) still work.
_HITL_FLAGS: dict[str, bool] = {}

# Per-gate refine cap (2026-06-26 loop-audit hardening). The operator can
# pick "refine" at each HITL gate (REQUIREMENTS / ARCHITECTURE / STORIES /
# DEPLOYMENT) to send the spec back for re-generation; each refine reruns
# the discovery + spec-writer + reviewer chain at ~$0.10–0.30 a pass.
# ``discovery_question_count`` (the inner discovery cap, default 10) is
# RESET on every entry to spec_review_node (graph.py:10937), so without a
# per-gate refine cap the operator can drive an unbounded loop simply by
# pressing "e" each time — the only brake is the session budget. This cap
# bounds the per-gate refines; once reached, the gatekeeper refuses
# further refines and asks the operator to approve / manual-edit /
# suspend instead. Generous default (5) so it never bites a normal
# review cycle but does catch a hung loop.
MAX_GATEKEEPER_REFINES = 5

# Per-trigger sub-cap. Bounds any SINGLE trigger's contribution so one
# exhausted failure class can't monopolize the recovery pool. Session
# 44c5e194-5715-451f-92c6-84362eeb7453 (2026-07-10) tripped the
# session cap via 1× test_generation_max_iterations + 2× zero_patch_loop:2,
# leaving the operator no slack for genuinely-different follow-on
# failures — this per-trigger cap prevents that shape. Operators can
# override via ``state['hitl_auto_resume_cap_per_trigger']``.
_HITL_AUTO_RESUME_CAP_PER_TRIGGER = 3

# Per-session cap on consecutive headless HITL auto-resumes for
# loop-stuck triggers (repair_loop_limit, persistent_build_failure,
# zero_patch_loop, no_progress_failsafe, low_signal_verdict_loop, etc.).
# The auto-resume path in ``hitl_menu_loop`` calls
# ``_reset_hitl_trip_counters`` so the same guards don't re-fire
# immediately — but the *underlying failure* is unchanged in headless
# mode (no operator did anything between iterations), so the guards
# accumulate to threshold again after a handful of rounds and HITL
# re-fires. Every trip also spends an LLM escalation-summary call, so
# an uncapped loop drains budget until it hits 0 (session cec4d124
# fired 19 HITLs in ~2 hours before the operator killed the process
# manually). This cap gives ``_reset_hitl_trip_counters`` a few shots
# to genuinely unstick the loop and then terminates cleanly. Operators
# can override via ``state['hitl_auto_resume_cap']`` when a long-tail
# recovery legitimately needs more headroom.
#
# Sized as 3× the per-trigger cap so up to three distinct failure
# classes can each spend their full per-trigger slack before the
# session-level kill-switch fires. Previously equal to the per-trigger
# cap, which defeated the point — with 3 distinct HITL triggers each
# taking 1 auto-resume the session cap tripped before ANY trigger came
# close to its own cap. Finsearch session 156032347 (2026-07-13)
# terminated at 2× repair_loop_limit + 1× test_generation_max_iterations
# = session cap 3/3, blocking a run that had per-trigger slack remaining
# on both triggers. Trades a slightly larger budget-drain ceiling in the
# genuinely-unrecoverable case for headroom in the multi-trigger case
# that the per-trigger cap already bounds.
_HITL_AUTO_RESUME_CAP = _HITL_AUTO_RESUME_CAP_PER_TRIGGER * 3

# The five HITL gates the resolver knows about. Tuples of
# (gate_name, args_attr, config_key) so the resolver, the CLI plumbing,
# and the config-tree validation stay aligned. New gates added here
# automatically pick up the same three-tier resolution. Names use full
# words (``requirement``, not ``req``; ``architecture``, not ``arch``)
# so the operator-facing CLI flag, the config.json key, and the in-code
# gate-name all match.
_HITL_GATES: tuple[tuple[str, str, str], ...] = (
    ("requirement",        "hitl_requirement",        "requirement"),
    ("architecture",       "hitl_architecture",       "architecture"),
    ("repair",             "hitl_repair",             "repair"),
    ("deployment",         "hitl_deployment",         "deployment"),
    ("layout_divergence",  "hitl_layout_divergence",  "layout_divergence"),
)


def _resolve_hitl_flags(
    args: Any,
    config: dict[str, Any],
) -> dict[str, bool]:
    """Resolve effective HITL gate values via three-tier precedence.

    For each gate:
      1. The matching CLI flag when it was explicitly passed
         (the argparse default is the ``None`` sentinel — anything
         other than ``None`` means the operator typed ``--hitl-foo
         true|false`` on the command line).
      2. The matching key under ``config.json``'s ``hitl`` block.
         Non-bool values are rejected by ``validate_config_strict``
         before we get here.
      3. ``True`` — the harness-level safe default: when neither the
         CLI nor the config has an opinion, the gate prompts.

    Returns the resolved map keyed by gate-name (``req``, ``arch``,
    ``repair``, ``deployment``, ``layout_divergence``) ready to hand
    to :func:`_set_hitl_flags` (which uses different parameter names).
    The auto-approve fallbacks (CI=true, HARNESS_AUTO_APPROVE,
    non-TTY stdin) still apply on top of this and are checked in
    :func:`_gatekeeper_auto_approves`.
    """
    cfg_block = config.get("hitl") if isinstance(config, dict) else None
    if not isinstance(cfg_block, dict):
        cfg_block = {}
    resolved: dict[str, bool] = {}
    for gate, args_attr, cfg_key in _HITL_GATES:
        cli_val = getattr(args, args_attr, None)
        if cli_val is not None:
            resolved[gate] = bool(cli_val)
            continue
        if cfg_key in cfg_block and isinstance(cfg_block[cfg_key], bool):
            resolved[gate] = cfg_block[cfg_key]
            continue
        resolved[gate] = True
    return resolved


def _set_hitl_flags(
    *,
    requirement: bool,
    architecture: bool,
    repair: bool,
    deployment: bool,
    layout_divergence: bool = False,
) -> None:
    """Pin the operator's --hitl-* choices for this run.

    Keys MUST match ``_HITL_GATES[0]`` so ``_hitl_gate_enabled`` lookups
    succeed at every gate callsite. The four mandatory kwargs use the
    same singular full-word naming as the CLI flags
    (``--hitl-requirement``, ``--hitl-architecture``, ...) — no
    abbreviations, no plural/singular mismatch."""
    _HITL_FLAGS.update({
        "requirement":       requirement,
        "architecture":      architecture,
        "repair":            repair,
        "deployment":        deployment,
        "layout_divergence": layout_divergence,
    })


def _hitl_gate_enabled(gate_name: str) -> bool:
    """True when the operator wants this gate to prompt. When the pin
    map is empty (tests / direct calls), defaults to True so legacy
    behaviour is preserved — only `_set_hitl_flags` activates the
    new opt-in semantics."""
    return _HITL_FLAGS.get(gate_name, True)


def _gatekeeper_auto_approves(gate_name: Optional[str] = None) -> bool:
    """
    True when the gatekeeper should skip interactive approval — when the
    operator opted out via --hitl-<gate>=false, when CI=true,
    HARNESS_AUTO_APPROVE=true, or when stdin isn't a TTY (a piped
    invocation has no way to answer the prompt).

    Unlike the deploy preview gate (which fails closed on non-TTY because
    LLM-generated containers are about to launch), the spec/architecture
    gatekeeper has lower blast radius — a non-TTY here just means CI, so
    auto-approve is safe.
    """
    if gate_name is not None and not _hitl_gate_enabled(gate_name):
        return True   # operator opted out via --hitl-<gate>=false
    # Mirror :func:`harness.hitl._is_truthy_env` — accept the same
    # canonical set ({true,1,yes,on,y}) so GitLab / Jenkins / CircleCI /
    # TeamCity (which commonly set CI=1) trigger the auto-approve path
    # rather than hanging on input(). Audit §5.5.
    _truthy = {"true", "1", "yes", "on", "y"}
    return (
        os.environ.get("CI", "").strip().lower() in _truthy
        or os.environ.get("HARNESS_AUTO_APPROVE", "").strip().lower() in _truthy
        or not sys.stdin.isatty()
    )


def human_gatekeeper_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Adaptive three-phase HITL gatekeeper node.
    
    Inspects state["current_gate"] and presents a phase-specific review menu:
        - REQUIREMENTS: Review SPEC_REQUIREMENTS.md
        - ARCHITECTURE: Review SPEC_ARCHITECTURE.md
        - DEPLOYMENT: Review DEPLOYMENT_BLUEPRINT.md
    
    Options per phase:
        [a] Approve → Proceed to next phase
        [e] Refine → Capture feedback, append to messages, loop back to generator
        [m] Manual → Pause for IDE edits, read updated file from disk
    
    Returns state update with routing signal in node_state.gatekeeper_action.
    """
    gate = state.get("current_gate", "")
    workspace = state.get("workspace_path", os.getcwd())
    messages = list(state.get("messages", []))
    loop_counter = state.get("loop_counter", {})
    loop_counter = dict(loop_counter)
    gate_attempts_key = f"gate_{gate.lower()}"
    attempt = loop_counter.get(gate_attempts_key, 0) + 1
    loop_counter[gate_attempts_key] = attempt

    # Determine which file to show
    if gate == "REQUIREMENTS":
        spec_path = state.get("spec_requirements_path", os.path.join(workspace, "docs", "SPEC_REQUIREMENTS.md"))
        gate_label = "REQUIREMENTS"
        gate_desc = "Requirements Specification"
        file_label = "SPEC_REQUIREMENTS.md"
        next_phase = "Architecture Specification"
    elif gate == "ARCHITECTURE":
        spec_path = state.get("spec_architecture_path", os.path.join(workspace, "docs", "SPEC_ARCHITECTURE.md"))
        gate_label = "ARCHITECTURE"
        gate_desc = "Architecture Specification"
        file_label = "SPEC_ARCHITECTURE.md"
        next_phase = "Code Generation & Patching"
    elif gate == "DEPLOYMENT":
        spec_path = state.get("deployment_blueprint_path", os.path.join(workspace, "docs", "DEPLOYMENT_BLUEPRINT.md"))
        gate_label = "DEPLOYMENT"
        gate_desc = "Deployment Blueprint"
        file_label = "DEPLOYMENT_BLUEPRINT.md"
        next_phase = "Container Deployment"
    elif gate == "STORIES":
        # Story decomposition gate. docs/STORIES.md is regenerated from
        # .teane/state.db; the operator approves the breakdown before the
        # per-story TDD loop starts. Refine re-runs decomposition with
        # appended feedback. Manual-edit is best-effort — STORIES.md is
        # a view, the DB is the source of truth, so the operator who
        # wants surgical edits should open .teane/state.db directly.
        spec_path = os.path.join(workspace, "docs", "STORIES.md")
        gate_label = "STORIES"
        gate_desc = "Story Decomposition"
        file_label = "STORIES.md"
        next_phase = "Per-Story TDD Loop"
    else:
        logger.warning("[gatekeeper] Unknown gate: %s. Proceeding.", gate)
        return {"node_state": {"gatekeeper_action": "approve", "current_gate": gate}}

    # Non-interactive auto-approval. The spec lists CI / HARNESS_AUTO_APPROVE
    # as supported, but the gatekeeper was previously blocking on input()
    # even when those were set — making CI runs hang forever waiting on
    # stdin. Honor the env vars here as well as a non-TTY stdin.
    # state["current_gate"] is the upper-case routing token
    # (REQUIREMENTS / ARCHITECTURE / DEPLOYMENT). _HITL_FLAGS keys use the
    # singular full-word form (requirement / architecture / deployment) so
    # they match --hitl-requirement / --hitl-architecture / --hitl-deployment
    # and the config.json hitl.* block. Normalise here so the gate_name
    # lookup hits the right entry.
    _gate_to_hitl_key = {
        "REQUIREMENTS": "requirement",
        "ARCHITECTURE": "architecture",
        "DEPLOYMENT":   "deployment",
        "STORIES":      "stories",
    }
    if _gatekeeper_auto_approves(
        gate_name=_gate_to_hitl_key.get(gate, gate.lower())
    ):
        logger.info(
            "[gatekeeper] %s auto-approved (--hitl-<gate>=false / CI / "
            "HARNESS_AUTO_APPROVE / no TTY).",
            gate_label,
        )
        return {
            "messages": messages,
            "loop_counter": loop_counter,
            "node_state": {"gatekeeper_action": "approve", "current_gate": gate},
        }

    while True:
        spec_content = ""
        spec_size = 0
        if os.path.isfile(spec_path):
            try:
                with open(spec_path, "r", encoding="utf-8") as f:
                    spec_content = f.read()
                spec_size = len(spec_content)
            except OSError:
                pass

        print()
        print("=" * 72)
        print(f"[HITL GATE: {gate_label}] — {gate_desc}")
        print(f"  File: {spec_path}")
        print(f"  Size: {spec_size:,} characters")
        print(f"  Attempt: {attempt}")
        print("=" * 72)
        print()

        # Build a single labels dict per gate. Used for both the stdin
        # menu printout AND the option_labels parameter to channel.prompt
        # — the dashboard renders the same dict as <option> text so the
        # operator sees the action description instead of a bare letter.
        if gate == "REQUIREMENTS":
            print(f"Requirements written to {file_label}. Please review the specification.")
            gate_options: dict[str, str] = {
                "a": f"Approve & proceed to {next_phase}",
                "e": "Refine via text feedback",
                "m": "Pause for manual local edits in IDE",
                "s": "Save & quit (resume later)",
            }
        elif gate == "ARCHITECTURE":
            print(f"Technical layout blueprints written to {file_label}. Please review module boundaries.")
            gate_options = {
                "a": "Approve & begin coding / patching",
                "e": "Refine layout parameters",
                "m": "Pause for manual edits",
                "s": "Save & quit (resume later)",
            }
        elif gate == "STORIES":
            print(f"Story decomposition written to {file_label}. Review the breakdown before TDD begins.")
            print("Source of truth is `.teane/state.db`; STORIES.md is a regenerated view.")
            gate_options = {
                "a": f"Approve & begin {next_phase}",
                "e": "Refine decomposition (re-run with feedback)",
                "m": "Pause for manual edits to .teane/state.db",
                "s": "Save & quit (resume later)",
            }
        else:  # DEPLOYMENT
            print(f"Application fully compiled. Docker Composition written to {file_label}.")
            print("Please review container network bridges and volumes before firing.")
            gate_options = {
                "a": f"Approve & execute infrastructure {next_phase}",
                "e": "Refine deployment variables",
                "m": "Pause for manual edits",
                "s": "Save & quit (resume later)",
            }
        print("Options:")
        for _key, _label in gate_options.items():
            print(f"  [{_key}] {_label}")
        print()

        from harness.hitl import get_channel as _get_channel
        choice = _get_channel().prompt(
            f"How would you like to proceed with the {gate_desc}?",
            list(gate_options.keys()),
            default="a",
            option_labels=gate_options,
        )

        if choice == "a":
            logger.info("[gatekeeper] %s approved by developer.", gate_label)
            return {
                "messages": messages,
                "loop_counter": loop_counter,
                "node_state": {"gatekeeper_action": "approve", "current_gate": gate},
            }

        elif choice == "e":
            # ``attempt`` is the per-gate visit counter (initial review
            # plus one per refine return). After MAX_GATEKEEPER_REFINES
            # refines the operator has burned 5 spec-regeneration passes
            # at this gate; refuse a further refine and force them to
            # approve, manual-edit, or suspend. Without this brake the
            # inner discovery cap (max_discovery_iterations) is
            # circumvented because spec_review_node resets the question
            # counter on every entry.
            refines_used = max(0, attempt - 1)
            if refines_used >= MAX_GATEKEEPER_REFINES:
                print()
                print(f"[{gate_label}] You've used {refines_used} refine(s) "
                      f"(cap: {MAX_GATEKEEPER_REFINES}). Further refines are "
                      f"disabled to prevent runaway spec-regeneration cost.")
                print("Choose [a] approve, [m] manual edit, or [s] suspend.")
                print()
                logger.warning(
                    "[gatekeeper] %s refine cap reached (used %d/%d). "
                    "Refusing further refines this session — operator must "
                    "approve, manual-edit, or suspend.",
                    gate_label, refines_used, MAX_GATEKEEPER_REFINES,
                )
                continue

            from harness.hitl import get_channel as _get_channel
            notes = _get_channel().notes(f"[Refine:{gate_label}] Enter additional notes/feedback")

            if not notes:
                print("[Refine] No notes provided. Returning to menu.")
                continue

            # Append feedback to messages as a user instruction
            messages.append({"role": "user", "content": f"[HITL Feedback - {gate_label}]: {notes}"})

            # Reset loop counter to give the generator a fresh attempt
            loop_counter["patching"] = 0
            loop_counter["repair"] = 0
            loop_counter["compiler"] = 0
            loop_counter["total_repairs"] = 0

            logger.info(
                "[gatekeeper] %s refine requested: %d chars of feedback "
                "(%d/%d refines used).",
                gate_label, len(notes), refines_used + 1, MAX_GATEKEEPER_REFINES,
            )
            return {
                "messages": messages,
                "loop_counter": loop_counter,
                "node_state": {"gatekeeper_action": "refine", "current_gate": gate},
            }

        elif choice == "m":
            print(f"[Manual] Edit the file at: {spec_path}")
            print("[Manual] Make your changes in your editor (VS Code, Cursor, etc.).")
            from harness.hitl import get_channel as _get_channel
            _get_channel().wait_for_manual_edit(spec_path)

            # Reload the manually edited file into messages[0] (system prompt)
            if os.path.isfile(spec_path):
                try:
                    with open(spec_path, "r", encoding="utf-8") as f:
                        updated_spec = f.read()
                    # Update messages[0] with the manually edited spec
                    if messages:
                        messages[0] = {"role": "system", "content": updated_spec}
                    logger.info("[gatekeeper] %s manual edits confirmed (%d chars).", gate_label, len(updated_spec))
                except OSError:
                    logger.warning("[gatekeeper] Failed to read manually edited file.")

            return {
                "messages": messages,
                "loop_counter": loop_counter,
                "node_state": {"gatekeeper_action": "manual", "current_gate": gate},
            }

        elif choice == "s":
            session_id = state.get("session_id", "")
            print()
            print("=" * 60)
            print("Session saved to checkpoint.")
            print("Resume later with:")
            print(f"  teane resume --session-id {session_id}")
            if workspace and workspace != os.getcwd():
                print(f"  teane resume --session-id {session_id} -w {workspace}")
            print("=" * 60)
            print()
            logger.info("[gatekeeper] %s suspended by developer. Session: %s", gate_label, session_id)
            return {
                "messages": messages,
                "loop_counter": loop_counter,
                "node_state": {
                    "gatekeeper_action": "suspend",
                    "current_gate": gate,
                    "hitl_suspend": True,
                    "suspended_from": "gatekeeper",
                },
            }

        else:
            print(f"[Gatekeeper] Unknown option: '{choice}'. Please choose a, e, m, or s.")


def discovery_interview_loop(state: dict[str, Any]) -> dict[str, Any]:
    """
    Sequential discovery interview for requirements/architecture/deployment phases.

    Walks the operator through one question at a time. Each question shows the
    LLM's recommended answer (from the discovery node's ``suggested_answer``
    field); pressing Enter accepts it, typing text overrides it. Commands:
        SUSPEND — save & quit (resumable via ``teane resume``).
        DONE    — finish the round now. If critical questions remain
                  unanswered, the loop refuses to finalize and routes back
                  to the discovery node for a follow-up round.
        SKIP    — leave a non-critical question blank. Critical questions
                  cannot be skipped — the loop re-asks once and, if skipped
                  again, marks the round incomplete so the LLM re-emits it.

    The collected answers are concatenated into one structured
    ``[Discovery Response - <phase>]`` message appended to the conversation
    history. ``route_after_discovery`` then either loops back to the
    discovery node (if critical unknowns remain) or proceeds to
    ``write_spec_node``.
    """

    gate = state.get("current_gate", "REQUIREMENTS")
    discovery_data = state.get("discovery_questions", {})
    modules = discovery_data.get("modules", [])
    messages = list(state.get("messages", []))
    node_state = state.get("node_state", {})
    complete = node_state.get("discovery_complete", False)
    round_num = node_state.get("discovery_question_count", 0)

    phase_label = gate if gate in ("REQUIREMENTS", "ARCHITECTURE", "DEPLOYMENT") else "ARCHITECTURE"

    if complete:
        logger.info("[discovery] %s discovery complete. Proceeding.", phase_label)
        return {"messages": messages, "node_state": node_state}

    # Flatten modules → ordered list of (module_name, question_dict).
    flat: list[tuple[str, dict[str, Any]]] = []
    for module in modules:
        mod_name = module.get("name", "Module")
        for q in module.get("questions", []) or []:
            if isinstance(q, dict):
                flat.append((mod_name, q))
    total = len(flat)
    critical_total = sum(1 for _, q in flat if q.get("critical"))

    from harness.hitl import get_channel as _get_channel
    channel = _get_channel()

    # Header
    print()
    print("=" * 80)
    print(f"[HARNESS ARCHITECT SYSTEM AUDIT: {phase_label} PHASE] — Round {round_num}")
    print("=" * 80)

    if total == 0:
        # Edge case: discovery node returned no questions (likely schema
        # drift). Let the operator either finalize or save & quit so they
        # aren't stuck staring at an empty interview.
        print("No questions returned this round.")
        print("Type 'DONE' to finalize this phase or 'SUSPEND' to save & quit.")
        print("-" * 80)
        response = (channel.notes("User Response") or "").strip()
        if response.upper() == "SUSPEND":
            return _discovery_suspend(state, node_state, messages, phase_label)
        if response.upper() == "DONE":
            node_state["discovery_complete"] = True
            print("[Discovery] Finalizing specification...")
            return {"messages": messages, "node_state": node_state}
        return {"messages": messages, "node_state": node_state}

    summary = discovery_data.get("summary", "")
    if summary:
        print(f"Summary: {summary}")
    print(
        f"{total} question(s) this round ({critical_total} critical). "
        "Answer one at a time."
    )
    print("Per question: [Enter] accept the recommendation, type to override,")
    print("              SUSPEND = save & quit · DONE = finish now · SKIP = skip non-critical.")
    print("-" * 80)
    print()

    collected: list[dict[str, Any]] = []
    current_module: Optional[str] = None
    suspended = False
    done_early = False

    for idx, (mod_name, q) in enumerate(flat, start=1):
        if mod_name != current_module:
            current_module = mod_name
            print(f"[MODULE: {mod_name}]")
            print()

        qid = q.get("id", f"Q{idx}")
        text = q.get("text", "(no question text)")
        is_critical = bool(q.get("critical"))
        critical_marker = " **CRITICAL**" if is_critical else ""
        suggested = str(q.get("suggested_answer", "") or "").strip()

        print(f"Q {idx}/{total} — {qid}{critical_marker}")
        print(f"  {text}")
        if suggested:
            print(f"  Recommended: {suggested}")

        answer, control = _ask_one_discovery_question(
            channel, qid, suggested, is_critical,
        )

        if control == "SUSPEND":
            suspended = True
            break
        if control == "DONE":
            done_early = True
            break
        if control == "SKIP":
            collected.append({
                "module": mod_name, "qid": qid, "text": text,
                "answer": "[SKIPPED]", "accepted_recommendation": False,
                "critical": is_critical, "skipped": True,
            })
            print("  → skipped")
            print()
            continue

        accepted_rec = (answer == suggested and bool(suggested) and control == "ACCEPT")
        collected.append({
            "module": mod_name, "qid": qid, "text": text,
            "answer": answer, "accepted_recommendation": accepted_rec,
            "critical": is_critical, "skipped": False,
        })
        if accepted_rec:
            print(f"  → accepted: {suggested}")
        else:
            print("  → recorded")
        print()

    if suspended:
        return _discovery_suspend(state, node_state, messages, phase_label)

    # Build the structured response back to the LLM.
    body_lines = [
        f"[Discovery Response - {phase_label}] "
        f"(Round {round_num}, answered {len(collected)} of {total} questions):"
    ]
    grouped: dict[str, list[str]] = {}
    for item in collected:
        line = f"  - {item['qid']}: {item['answer']}"
        if item["accepted_recommendation"]:
            line += "  [accepted recommendation]"
        if item["skipped"]:
            line += "  [skipped by operator]"
        grouped.setdefault(item["module"], []).append(line)
    for mod_name, lines in grouped.items():
        body_lines.append(f"[{mod_name}]")
        body_lines.extend(lines)
    response_text = "\n".join(body_lines)
    messages.append({"role": "user", "content": response_text})

    # Compute remaining critical questions: those marked critical that were
    # either skipped or never reached because of an early DONE.
    answered_ids = {
        item["qid"] for item in collected
        if not item["skipped"] and item["answer"]
    }
    critical_unresolved = sum(
        1 for _, q in flat
        if q.get("critical") and q.get("id") not in answered_ids
    )

    if done_early and critical_unresolved > 0:
        print()
        print("=" * 60)
        print(
            f"[CRITICAL UNKNOWN DETECTED]: {critical_unresolved} critical "
            "question(s) still require answers."
        )
        print("You must specify the remaining variables before this phase can be finalized.")
        print("=" * 60)
        print()
        node_state["user_done_with_critical"] = True
        node_state["discovery_complete"] = False
        return {"messages": messages, "node_state": node_state}

    if done_early:
        node_state["discovery_complete"] = True
        logger.info(
            "[discovery] User finalized %s phase early. %d answers, no critical remaining.",
            phase_label, len(collected),
        )
        print("[Discovery] Finalizing specification...")
        return {"messages": messages, "node_state": node_state}

    # Full round walked: hand back to the discovery LLM for follow-up
    # evaluation. If the LLM determines complete=True, route_after_discovery
    # routes to write_spec_node on the next pass. If critical_unresolved > 0
    # (operator SKIPped a critical question twice), force another round.
    node_state["discovery_complete"] = False
    if critical_unresolved > 0:
        node_state["discovery_critical_remaining"] = critical_unresolved
    logger.info(
        "[discovery] Received %d answers for %s phase (%d critical unresolved). "
        "Routing back for evaluation.",
        len(collected), phase_label, critical_unresolved,
    )
    return {"messages": messages, "node_state": node_state}


def _ask_one_discovery_question(
    channel: Any, qid: str, suggested: str, is_critical: bool,
) -> tuple[str, str]:
    """Prompt for a single discovery question and resolve special commands.

    Returns ``(answer, control)`` where control ∈ {"ACCEPT", "OVERRIDE",
    "SUSPEND", "DONE", "SKIP"}. For ACCEPT/OVERRIDE the ``answer`` field
    holds the final value to record. Empty input → ACCEPT (uses
    ``suggested`` if present, else empty string).

    Critical questions cannot be SKIPped — on a first SKIP we re-prompt
    with an explicit "no skip" message; on a second SKIP we return
    control=SKIP anyway so the caller can record the unresolved state
    and let route_after_discovery loop the round.
    """
    if suggested:
        prompt_label = f"Q {qid} (Enter = accept, type to override)"
    else:
        prompt_label = f"Q {qid} (type your answer)"

    for attempt in range(2):
        raw = (channel.notes(prompt_label) or "").strip()
        upper = raw.upper()
        if upper == "SUSPEND":
            return ("", "SUSPEND")
        if upper == "DONE":
            return ("", "DONE")
        if upper == "SKIP":
            if is_critical and attempt == 0:
                print(
                    "  [REJECTED] Cannot SKIP a critical question. "
                    "Type an answer or press Enter to accept the recommendation."
                )
                continue
            return ("", "SKIP")
        if raw == "":
            return (suggested, "ACCEPT")
        return (raw, "OVERRIDE")

    # Loop exhausted — shouldn't happen; treat as SKIP for safety.
    return ("", "SKIP")


def _discovery_suspend(
    state: dict[str, Any],
    node_state: dict[str, Any],
    messages: list[Any],
    phase_label: str,
) -> dict[str, Any]:
    """Stamp the suspend flag set + print the resume instructions banner."""
    session_id = state.get("session_id", "")
    workspace = state.get("workspace_path", "")
    print()
    print("=" * 60)
    print("Session saved to checkpoint.")
    print("Resume later with:")
    print(f"  teane resume --session-id {session_id}")
    if workspace and workspace != os.getcwd():
        print(f"  teane resume --session-id {session_id} -w {workspace}")
    print("=" * 60)
    print()
    logger.info(
        "[discovery] %s phase suspended by developer. Session: %s",
        phase_label, session_id,
    )
    node_state["hitl_suspend"] = True
    node_state["suspended_from"] = "discovery_interview"
    return {"messages": messages, "node_state": node_state}


def _reset_hitl_trip_counters(loop_counter: dict[str, Any]) -> None:
    """Zero the counters that gate ``route_after_compiler`` / ``route_after_patching``
    HITL escalation so a headless auto-resume can't ping-pong forever.

    In a TTY session the operator picks `[r]` after they've done something
    outside the harness (widened the allowlist, edited config, fixed a
    file); the preserved counters keep the directive-shaping context that
    LLMs need across the resume. In the no-TTY auto-resume path nothing
    outside the harness has changed — the counters are still at their
    trip values, so the next ``route_after_compiler`` immediately re-fires
    the same HITL trigger and the loop repeats within seconds until budget
    is drained (observed session 4754c913).

    Resets iff the caller explicitly opts in via ``hitl_menu_loop``'s
    auto-resume path. Human resume path stays byte-identical to preserve
    session 2d0164f0 / 19b28eff / 0a5c6fe8's fix (see
    ``_reset_iteration_counters``).
    """
    # STALL_TRIPWIRE_KEYS is the canonical registry of router stall
    # tripwires (see harness/loop_counter_keys.py). It includes
    # ``cheap_shots_taken`` (2026-07-04): reset on HITL auto-resume so
    # the cheap model gets a fresh ``max_repair_attempts - 1`` shots
    # per HITL cycle. Without this, the counter accumulates across the
    # whole session and every post-HITL round burns the reasoning
    # model. Ciod session 523e86a7 saw 5+ escalations per batch because
    # the cheap-shot budget was effectively exhausted after the first
    # HITL cycle. Present-only reset: a legacy checkpoint without a key
    # must not gain it here.
    for key in STALL_TRIPWIRE_KEYS:
        if key in loop_counter:
            loop_counter[key] = 0
    # PER_BATCH_CAP_COUNTERS is the canonical registry of caps that
    # trip HITL (see harness/loop_counter_keys.py). Auto-resume must
    # zero them or the very trigger that fired HITL trips again on
    # the next entry and the session ping-pongs to the session cap
    # (finsearch 156032347: test_generation counter carried 5 across
    # batch boundary, tripped max_iterations on batch 110's first
    # entry). Assignment (not ``if key in``) because a zero after
    # first-time exposure is still the correct state.
    for key in PER_BATCH_CAP_COUNTERS:
        loop_counter[key] = 0
    # judge_ignored bookkeeping resets on auto-resume — those flags
    # exist to make the NEXT round's banner escalate ("YOU IGNORED THE
    # JUDGE"), which only makes sense in a single unbroken repair
    # stretch. But we deliberately KEEP
    # ``judge_named_file_lines_last_round``,
    # ``judge_persistent_files_last_round``, and
    # ``persistent_blocker_streak_per_file``: in headless auto-resume
    # nothing outside the harness has changed, so a blocker that was
    # persistent BEFORE the HITL trip is still persistent AFTER it, and
    # the persistent-blocker banner should fire IMMEDIATELY on the next
    # round instead of waiting two more rounds to re-detect the streak
    # from zero. Session bs27lvfpl demonstrated the miss: hard-cap fired
    # every 6 rounds on ``edgar.py:172``, the wipe reset the streak, and
    # the ESCAPE HATCH banner never got to fire because streak >= 3
    # required three consecutive rounds without a HITL interruption.
    for key in (
        "judge_round_touched_files",
        "judge_ignored_last_round",
    ):
        loop_counter.pop(key, None)
    # Per-file miss counts drive the "use a different operation" LLM
    # directive at >=2 — we want to keep that signal alive but not let
    # the >=3 stuck-file HITL guard immediately re-trip. Cap at 2.
    per_file = loop_counter.get("replace_block_misses_per_file")
    capped_files: set[str] = set()
    if isinstance(per_file, dict):
        for f in list(per_file.keys()):
            v = per_file[f]
            if isinstance(v, int) and v > 2:
                per_file[f] = 2
                capped_files.add(str(f))
    # Sibling of the cap-to-2 above: the stuck-target REWRITE recovery
    # ledger vetoes a second recovery shot for any file already in it.
    # If we cap the miss counter to unstick the router but leave the
    # ledger populated, the very next stuck round (miss counter climbs
    # back to 3) will HITL again without giving repair the recovery
    # shot the cap was supposed to enable. Drop just the entries we
    # capped, so the operator's headless auto-resume gets the recovery
    # round the cap-to-2 was designed for.
    ledger_raw = loop_counter.get("stuck_rewrite_recovery_attempted")
    if isinstance(ledger_raw, list) and capped_files:
        loop_counter["stuck_rewrite_recovery_attempted"] = sorted(
            p for p in ledger_raw
            if isinstance(p, str) and p not in capped_files
        )


def _reset_iteration_counters(
    loop_counter: Optional[dict[str, Any]], *, total_repairs: int = 0,
) -> dict[str, Any]:
    """Reset only the iteration counters in ``loop_counter`` while preserving
    diagnostic trackers (``replace_block_misses_per_file``, etc.) that the
    repair loop relies on for prompt directives across HITL resume.

    Wiping the whole dict here (the original behavior) was the root cause
    behind sessions like 2d0164f0 ping-ponging through HITL: the
    ``_format_replace_block_miss_directive`` only fires at ≥2 consecutive
    misses per file, so resetting that counter to zero on every resume meant
    the LLM never received the "use a different operation" directive and went
    straight back to the same broken REPLACE_BLOCK pattern.

    The router stall tripwires (``STALL_TRIPWIRE_KEYS`` — distraction /
    low-signal / zero-patch streaks and friends) are stepped back by ONE
    instead of preserved verbatim or zeroed. Preserving them verbatim made
    ``[r]`` Resume a dead end whenever the HITL trigger WAS one of them:
    the counter sits at its cap when HITL fires, ``route_after_compiler``
    consults it again BEFORE repair_node can run, and the only resets
    (PROGRESS verdict in repair_node, green-build
    ``_reset_stall_tripwires_on_progress``) are unreachable — session
    22471c0c re-fired ``reflection_distraction_loop:3`` twenty seconds
    after every resume with zero repair turns in between. Decrementing by
    one admits exactly one more repair round through the gate that fired
    (mirroring the ``total_repairs`` seed contract: "one more attempt
    before HITL re-fires") while keeping the streak high enough that
    directive escalations still hold (e.g. the reflection judge keeps
    using the reasoning model at ``consecutive_distraction_rounds >= 2``).

    Two counters ARE reset alongside the iteration counters because they
    track the very condition the operator just intervened to address:

    - ``missing_dep_consecutive_same``: counts consecutive same-symbol
      MISSING_DEP recurrences. If we don't reset it, the next compiler
      pass after a `[r]` Resume immediately re-trips the bypass guard
      and routes straight back to HITL — exactly the loop session
      90d3a8d2 got stuck in after the operator changed the image.
    - ``missing_dep_last_symbol``: the symbol we were tracking. Cleared
      so a different MISSING_DEP after resume starts fresh.
    """
    base = dict(loop_counter or {})
    # PER_BATCH_CAP_COUNTERS is the canonical registry of caps that
    # trip HITL (see harness/loop_counter_keys.py). All three reset
    # sites — this function, ``_reset_hitl_trip_counters``, and
    # ``story_loop._batch_commit_node`` — must zero every key in it,
    # or a resumed batch trips its cap without any real iteration
    # (finsearch 156032347 batch 110). ``tests/test_reset_registry.py``
    # enforces the invariant.
    for key in PER_BATCH_CAP_COUNTERS:
        base[key] = 0
    # Step each stall tripwire one below its current (= trip) value so
    # the gate that fired HITL passes exactly once — see docstring.
    # Present-only, like ``_reset_hitl_trip_counters``: a checkpoint
    # without the key must not gain it here.
    for key in STALL_TRIPWIRE_KEYS:
        if key in base:
            base[key] = max(0, int(base.get(key, 0) or 0) - 1)
    base["total_repairs"] = total_repairs
    base["missing_dep_consecutive_same"] = 0
    base["missing_dep_last_symbol"] = ""
    return base


def _refresh_session_config_into_state(state: dict[str, Any]) -> None:
    """Re-read the on-disk config for ``state["workspace_path"]`` and
    propagate the keys that drive build behaviour into the live state.

    Called from the HITL ``[r]`` Resume branch (and any future branch where
    the operator might have edited config between triggers). Without this,
    the state's ``sandbox_config`` / ``build_command`` are frozen at the
    values checkpointed when ``run_graph`` was first invoked — operator
    edits to ``config.json`` between HITL triggers never reach the next
    iteration's compiler_node.

    Best-effort: any error during config rediscovery is logged at debug
    and the state is left untouched (better to retry the build with the
    stale image than to crash the HITL loop). The handful of keys we
    refresh are the ones an operator typically edits to resolve a build
    failure flagged by HITL — image, build command, network policy.
    """
    workspace_path = state.get("workspace_path")
    if not workspace_path:
        return
    try:
        fresh_config = discover_config(workspace_path)
    except Exception as exc:  # noqa: BLE001 — never block HITL on config error
        logger.debug(
            "[HITL] Could not refresh on-disk config for workspace %s: %s",
            workspace_path, exc,
        )
        return

    new_sandbox = dict(fresh_config.get("sandbox", {}) or {})
    old_sandbox = dict(state.get("sandbox_config") or {})
    if new_sandbox and new_sandbox != old_sandbox:
        state["sandbox_config"] = new_sandbox
        changed_keys = sorted(
            k for k in set(old_sandbox) | set(new_sandbox)
            if old_sandbox.get(k) != new_sandbox.get(k)
        )
        logger.info(
            "[HITL] sandbox_config refreshed from disk. Changed keys: %s",
            changed_keys,
        )

    # build_command is no longer config-driven; the workspace + locked
    # core_languages selection fully determine it. Re-auto-wire from
    # the current workspace so HITL pickups still adapt to new manifests
    # (e.g. operator dropped a pom.xml into a fresh workspace).
    #
    # is_greenfield is recovered from the session's flow ("build" = greenfield,
    # anything else = brownfield). Mid-session HITL refreshes must NOT let
    # an LLM-emitted Makefile flip a greenfield build's command to `make build`.
    is_greenfield_refresh = bool(state.get("flow") == "build")
    refreshed_build_cmd = resolve_build_command(
        fresh_config, workspace_path, is_greenfield=is_greenfield_refresh,
    )
    old_build_cmd = state.get("build_command")
    if refreshed_build_cmd and refreshed_build_cmd != old_build_cmd:
        state["build_command"] = refreshed_build_cmd
        logger.info(
            "[HITL] build_command auto-rewired from workspace: %r -> %r",
            old_build_cmd, refreshed_build_cmd,
        )

    new_allow_network = fresh_config.get("allow_network")
    if isinstance(new_allow_network, bool) and new_allow_network != state.get("allow_network"):
        state["allow_network"] = new_allow_network
        logger.info(
            "[HITL] allow_network refreshed from disk: %s",
            new_allow_network,
        )


def _build_outside_harness_actions(
    state: dict[str, Any], trigger: str,
) -> list[str]:
    """Per-trigger checklist of concrete steps the operator should take
    OUTSIDE the harness to unblock the run before resuming.

    The escalation summary diagnoses what went wrong; this list
    prescribes what to actually do about it. Strings are concrete
    actions (file paths, config keys, commands) — not generic advice
    like "investigate the error". Returned in priority order: try the
    first item first.

    Designed to be evidence-driven: when the state carries a missing
    symbol / rejected path / specific file, the action mentions it by
    name. Falls back to general guidance only when no specific signal
    is present.
    """
    node_state = state.get("node_state", {}) or {}
    workspace_path = state.get("workspace_path", os.getcwd())
    build_command = str(state.get("build_command", "") or "")
    sandbox_cfg = state.get("sandbox_config", {}) or {}
    docker_image = str(sandbox_cfg.get("docker_image", "") or "")
    errors = state.get("compiler_errors", []) or []
    rejections = node_state.get("allowlist_rejections") or []
    patch_failures = node_state.get("patch_failures") or []
    modified_files = state.get("modified_files", []) or []

    actions: list[str] = []

    # Symbols that test_generation_node emits via _synth_diag when the
    # node itself fails with an actual environment condition (missing
    # API key, no source files). They are NOT pip/npm packages, so the
    # toolchain branch below would emit nonsense like "install the
    # missing tool/package `llm_api_key`". Branch separately.
    #
    # Note: test_generation_max_iterations and test_generation_zero_emit
    # used to live here too. They're LLM-behavior failures (model won't
    # emit valid tests), not env-config problems, so they now flow
    # through the ``llm_behavior:<symbol>`` trigger family below.
    _NON_TOOLCHAIN_ENV_SYMBOLS = {
        "llm_api_key",
        "no_source_files",
    }

    # ---- traceability_block (v5 Phase 7 BUG #6) ---------------------
    # End-of-session audit found untraced FRs or untested ACs.
    # `compiler_errors` will be empty — the gate is a coverage gap,
    # not a build failure. The report text was printed to stdout right
    # before HITL fired; point the operator at it.
    if trigger == "traceability_block":
        actions.append(
            "The v5 traceability gate flagged a coverage gap. Scroll up "
            "in your terminal to the `==== TRACEABILITY BLOCK ====` "
            "banner — it lists the untraced requirements and untested "
            "acceptance criteria by ID."
        )
        if state.get("decomposition_enabled"):
            actions.append(
                "Untraced requirement: add a story (via `teane patch "
                "--agile true`) that cites the EPIC-NNN / FEAT-NNN / "
                "STORY-NNN / STORY-NFR-NNN key in its `requirement_keys`, "
                "OR revise `docs/SPEC_REQUIREMENTS.md` to remove the "
                "orphan requirement."
            )
        else:
            actions.append(
                "Untraced requirement: add a story (via `teane patch "
                "--agile true`) that cites the FR-NNN / NFR-XXX-NNN / "
                "US-NN-NN key in its `requirement_keys`, OR revise "
                "`docs/SPEC_REQUIREMENTS.md` to remove the orphan "
                "requirement."
            )
        actions.append(
            "Untested acceptance criterion: run `teane test` so the "
            "functional pack generates AC-linked coverage (it owns the "
            "`# @verifies: STORY-N.AC-N` markers), or write such a test "
            "by hand. Build/patch unit tests link to code via `@tests:` "
            "and never close AC gaps. Re-run `teane audit -w "
            "<workspace>` to verify."
        )
        actions.append(
            "Emergency bypass: set `traceability.enforce = false` in "
            "`.harness_config.json` and re-run. This is NOT recommended "
            "for ship-ready code — the audit exists to surface dropped "
            "requirements before they reach prod."
        )

    # ---- llm_behavior:<symbol> ---------------------------------------
    # LLM-behavior HITL: the test-generation model refused to emit valid
    # tests (either burned its real-iteration cap or its zero-emit
    # re-prompt sub-cap). Distinct from env_misconfig — there's nothing
    # to `pip install` here; the model's own output is the problem.
    elif trigger.startswith("llm_behavior"):
        symbol = ""
        if ":" in trigger:
            symbol = trigger.split(":", 1)[1].strip()
        symbol = symbol or str(node_state.get("llm_behavior_symbol", ""))
        if symbol == "test_generation_max_iterations":
            actions.append(
                "The test-generation node hit its max_iterations cap "
                "(default 2) trying to land valid tests. Inspect the "
                "last attempt under `tests/` — the LLM may have made "
                "the same mistake twice. Fix it by hand or raise "
                "`test_generation.max_iterations` in `config/config.json`."
            )
            actions.append(
                "Common cause: the `@tests:` marker contract — every "
                "generated test file must name the source file(s) it "
                "exercises (`# @tests: path/to/module.py`). Check the "
                "test files under review carry the marker and that the "
                "named paths exist in the workspace."
            )
        elif symbol == "test_generation_zero_emit":
            actions.append(
                "The test-generation LLM returned zero patch blocks "
                "for `test_generation.max_zero_emit_reprompts` "
                "consecutive re-prompts (default 3). This usually "
                "means the LLM can't tell what to test — check the "
                "prior patching_node output above: if the production "
                "code for this batch failed to apply (look for "
                "`patches=N succeed=0`), the LLM correctly has "
                "nothing to test. Fix the upstream patcher rejection "
                "first, then `[r]` Resume."
            )
            actions.append(
                "If the prod code DID land but the LLM is still "
                "silent, `[e]` inject a test hint like `write a "
                "test for <function_name> asserting <expected>` — "
                "the next re-prompt will see it."
            )
            actions.append(
                "Or raise the sub-cap for this session by setting "
                "`test_generation.max_zero_emit_reprompts` in "
                "`config/config.json` (default 3). Higher values "
                "trade budget for retries."
            )
        else:
            actions.append(
                "The LLM refused to emit a valid response and the "
                "harness exhausted its retry budget. Inject a manual "
                "hint via `[e]` describing what the model should "
                "produce, or `[m]` pause for manual edits."
            )

    # ---- env_misconfig:<symbol> --------------------------------------
    # Highest-precision trigger — we know exactly what's missing.
    elif trigger.startswith("env_misconfig"):
        symbol = ""
        if ":" in trigger:
            symbol = trigger.split(":", 1)[1].strip()
        symbol = symbol or str(node_state.get("env_misconfig_symbol", ""))
        if symbol in _NON_TOOLCHAIN_ENV_SYMBOLS:
            if symbol == "llm_api_key":
                actions.append(
                    "No LLM gateway available — the test-generation node "
                    "could not dispatch an LLM call. Set the appropriate "
                    "API key env var (`ANTHROPIC_API_KEY`, "
                    "`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`) in this shell "
                    "and re-run, or set `gateway.disable=true` in "
                    "`config/config.json` to skip test generation."
                )
            elif symbol == "no_source_files":
                actions.append(
                    "Test generation found zero candidate source files. "
                    "Verify the workspace's source tree (e.g. `src/`, "
                    "`app/`, `lib/`) actually contains files matching "
                    "the configured `_SOURCE_EXTENSIONS`. If the patcher "
                    "ran but wrote into an unexpected location, inspect "
                    "the recent diff with `git status` / `git diff`."
                )
        elif symbol:
            actions.append(
                f"Install the missing tool/package `{symbol}` into the build "
                f"sandbox. Easiest path: edit `config/config.json` → "
                f"`sandbox.docker_image` to a base image that ships "
                f"`{symbol}` (e.g. `python:3.12-slim` for pip tools, "
                f"`node:20-slim` for npm-side tooling)."
            )
            actions.append(
                f"Or bake `{symbol}` into your own builder image and point "
                f"`sandbox.docker_image` at it."
            )
            actions.append(
                f"Alternative: prepend an install step to `build_command` "
                f"in `config/config.json` so the sandbox installs `{symbol}` "
                f"before the build runs (e.g. `pip install {symbol} && "
                f"{build_command or '<your build cmd>'}`)."
            )
        else:
            actions.append(
                "The build sandbox is missing a required tool. Check the "
                "tail of the last build output above for the exact name, "
                "then either change `sandbox.docker_image` in "
                "`config/config.json` to an image that ships it, or "
                "prepend an install step to `build_command`."
            )

    # ---- budget exhausted / preflight --------------------------------
    elif trigger in {"budget_exhausted", "budget_preflight"}:
        actions.append(
            "Increase `token_budget.hard_cap_usd` in `config/config.json` "
            "(or just press `[b]` to add $2.00 to this session)."
        )
        actions.append(
            "If you're burning budget on retries, switch the expensive "
            "roles to a cheaper model in `config/config.json` → "
            "`model_assignments` (e.g. point `repair_primary` at "
            "`deepseek-v4-flash` instead of `deepseek-v4-pro`)."
        )

    # ---- llm_silent --------------------------------------------------
    elif trigger == "llm_silent":
        actions.append(
            "Provider returned no content. Check your API key env vars "
            "(`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `DEEPSEEK_API_KEY`) "
            "are exported in this shell and not expired."
        )
        actions.append(
            "Check the provider's status page; if degraded, switch the "
            "affected role in `config/config.json` → `model_assignments` "
            "to a different provider for now."
        )
        actions.append(
            "Confirm the model id in `gateway.models` actually exists "
            "(typos like `claude-3-5-sonnet` vs `claude-sonnet-3-5` "
            "produce silent empty responses on some providers)."
        )

    # ---- security_fix_limit:<n>/<m> ----------------------------------
    elif trigger.startswith("security_fix_limit"):
        actions.append(
            "Open the security findings the scanner kept flagging "
            "(look at `compiler_errors` above — entries tagged "
            "`SECURITY:` or `BANDIT:`) and decide if any are false "
            "positives in YOUR context."
        )
        actions.append(
            "For false positives, add the rule id to "
            "`config/config.json` → `security_scan.suppressed_rules` "
            "or `# nosec B<NNN>` / `# noqa: S<NNN>` inline in the "
            "offending file, then `[r]` Resume."
        )
        actions.append(
            "For real findings the LLM can't fix, manually rewrite the "
            "vulnerable code in your IDE, then `[m]` Pause for manual "
            "edits → Enter when done."
        )

    # ---- zero_patch_loop:<n> -----------------------------------------
    elif trigger.startswith("zero_patch_loop"):
        actions.append(
            "The repair LLM emitted zero patches for several rounds — "
            "it doesn't know what to change. Open the workspace at "
            f"`{workspace_path}` and read the failing file(s) listed "
            "in CRITICAL INFORMATION above; if the diagnostic is "
            "ambiguous (e.g. `KeyError` with no line number), the LLM "
            "can't recover without your help."
        )
        actions.append(
            "Pick the most likely fix yourself in your IDE — even a "
            "minimal change that converts the failure into a more "
            "specific error is enough. Then `[m]` Pause for manual "
            "edits → Enter when done."
        )
        actions.append(
            "If you'd rather steer the LLM than touch code, choose "
            "`[e]` and inject a sentence like 'the missing import is "
            "X in file Y' — the next repair turn will see it as a "
            "user hint."
        )

    # ---- replace_block_stuck:<file> ---------------------------------
    # Router bailed after a specific file racked up ≥3 REPLACE_BLOCK
    # misses AND the REWRITE-recovery round (repair_node's auto-promote)
    # also failed to unstick it. The suffix carries the file path.
    elif trigger.startswith("replace_block_stuck"):
        stuck_head = ""
        stuck_extra = 0
        if ":" in trigger:
            suffix = trigger.split(":", 1)[1].strip()
            # Format is either "path/to/file.py" or "path/to/file.py+N"
            # (N additional stuck files, per _infer_hitl_trigger's label
            # compression). Recover the head so we can name it.
            if "+" in suffix:
                stuck_head, _, extra_str = suffix.rpartition("+")
                try:
                    stuck_extra = int(extra_str)
                except ValueError:
                    stuck_head = suffix
            else:
                stuck_head = suffix
        also = (
            f" (and {stuck_extra} other file(s) in the same state)"
            if stuck_extra else ""
        )
        head_ref = f"`{stuck_head}`" if stuck_head else "the stuck file"
        actions.append(
            f"REPLACE_BLOCK on {head_ref}{also} has missed the on-disk "
            "content three times in a row AND the automatic REWRITE_FILE "
            "recovery round already spent its one shot. The LLM's mental "
            "model of the file has drifted beyond surgical or wholesale "
            "repair from inside the loop."
        )
        if stuck_head:
            actions.append(
                f"Open `{stuck_head}` in your IDE, apply the fix by hand, "
                "then `[m]` Pause for manual edits → Enter. That clears "
                "the LLM's stale mental model and re-runs the compiler."
            )
        actions.append(
            "Alternative: `[e]` inject a hint that names the concrete "
            "change the LLM keeps missing (e.g. 'the field was renamed "
            "from `foo` to `bar` in the previous batch'). A pointed hint "
            "often converges where mechanical retry couldn't."
        )

    # ---- no_progress_repairs:<n>/<cap> -------------------------------
    # Enough consecutive rounds shrunk neither the fingerprint set nor
    # the raw diagnostic count that the harness declared the loop
    # stalled. Distinct from repair_loop_limit — usually indicates the
    # LLM is patching the wrong file / layer.
    elif trigger.startswith("no_progress_repairs"):
        actions.append(
            "The repair loop hit its non-progress cap — enough "
            "consecutive rounds shrunk neither the fingerprint set nor "
            "the raw diagnostic count that the harness declared the "
            "loop stalled. Usually this means the LLM is patching the "
            "wrong file (a test that documents a bug in the production "
            "code, or vice-versa) or the wrong layer (patching an "
            "adapter when the interface contract is the actual issue)."
        )
        if errors:
            files_with_errors = sorted({
                str(e.get("file", "")) for e in errors[:8] if e.get("file")
            })[:5]
            if files_with_errors:
                actions.append(
                    "Open these files and trace the failure back to its "
                    f"root cause: {', '.join(files_with_errors)}. The "
                    "actual bug may live upstream of the file the "
                    "diagnostic names."
                )
        actions.append(
            "If you can identify the wrong-layer patching, `[e]` inject "
            "a hint like 'the bug is in file X, not the test that "
            "asserts it' — the next repair round will re-target."
        )

    # ---- hard_iteration_ceiling:<n>/<cap> ----------------------------
    # Total repairs hit the hard cap (max_iterations * multiplier, default
    # 12) while per-round progress signals kept the smaller no_progress
    # cap from tripping. Batch is likely too broad for one loop.
    elif trigger.startswith("hard_iteration_ceiling"):
        actions.append(
            "The repair loop ran to the hard total-iteration ceiling "
            "while STILL showing per-round progress. The batch is too "
            "broad for a single repair loop to finish — each round fixed "
            "one fingerprint but surfaced another of equal weight."
        )
        actions.append(
            "Best mitigation is prevention: split the story or narrow "
            "the batch so each verification chain has fewer failing "
            "fingerprints. For this session, `[q]` Abandon + re-plan is "
            "often cheaper than continuing."
        )
        actions.append(
            "If you want to try one more push, raise `node_throttle."
            "total_hard_cap_multiplier` in `config/config.json` (default "
            "4 → set to 6 or 8), then `[r]` Resume. Only do this if the "
            "diagnostic count has been trending DOWN across recent "
            "rounds — otherwise you're extending a lost cause."
        )

    # ---- same_missing_dep:<symbol> -----------------------------------
    # Same dep symbol recurred past the autofix bypass cap. Two very
    # different failure modes depending on symbol class (bootstrap tool
    # vs regular package) — mirror the router-side messaging at
    # graph.py::route_after_compiler L~17904 / L~17915.
    elif trigger.startswith("same_missing_dep"):
        symbol = ""
        if ":" in trigger:
            symbol = trigger.split(":", 1)[1].strip()
        symbol_ref = f"`{symbol}`" if symbol else "the missing dependency"
        # Deferred import — graph.py imports cli.py at module-level, so a
        # top-level `from harness.graph import ...` would risk a cycle.
        from harness.graph import _is_bootstrap_tool
        if symbol and _is_bootstrap_tool(symbol):
            actions.append(
                f"Missing dependency {symbol_ref} is a BOOTSTRAP TOOL — "
                "it belongs in the sandbox image, not the workspace "
                "manifest. The autofix bypass cannot install it from "
                "inside the loop."
            )
            actions.append(
                f"Edit `sandbox.docker_image` in `config/config.json` "
                f"(currently `{docker_image or '(unset)'}`) to an image "
                f"that includes {symbol_ref}, then `[r]` Resume."
            )
        else:
            actions.append(
                f"Missing package {symbol_ref} is a regular pip / npm "
                "package. Recurrence past the autofix cap usually means "
                "workspace-manifest topology mismatch: the workspace has "
                "multiple manifests (e.g. root `pyproject.toml` + "
                "`requirements.txt` + a subdir `requirements.txt`) but "
                "the `build_command` only installs from a subset."
            )
            actions.append(
                f"Grep the workspace at `{workspace_path}` for every "
                "`requirements.txt`, `pyproject.toml`, `package.json`, "
                f"`Pipfile`, `setup.py`. Confirm {symbol_ref} appears in "
                "at least one, then confirm the build_command installs "
                "from THAT manifest — either `uv pip install -e .` (root "
                "pyproject) or `uv pip install -r <path>` (specific "
                "requirements file)."
            )
            if build_command:
                actions.append(
                    f"Current build_command: `{build_command}`. Adjust "
                    "or add the missing install step, then `[r]` Resume."
                )

    # ---- build_command_blocked:<rule> --------------------------------
    # Sandbox CommandValidator refused the leading command primitive
    # (cd / bash / etc.) — the global validator config is what needs
    # to change; repair rounds cannot amend it.
    elif trigger.startswith("build_command_blocked"):
        rule = ""
        if ":" in trigger:
            rule = trigger.split(":", 1)[1].strip()
        rule_ref = f"`{rule}`" if rule else "the build command's leading token"
        actions.append(
            f"The sandbox CommandValidator refused {rule_ref}. The "
            "validator's allow-list lives in the global harness config "
            "(`~/.harness/config.json` under `security.allowed_commands`) — "
            "repair rounds cannot amend it because the patcher's "
            "allowlist protects the workspace tree, not the harness "
            "home directory."
        )
        actions.append(
            "Two mitigations: (a) add the refused primitive to "
            "`security.allowed_commands` in `~/.harness/config.json` if "
            "it's genuinely safe (bash / cd are common), OR (b) rewrite "
            f"the build_command (currently `{build_command or '(unset)'}`) "
            "to use only whitelisted primitives (e.g. `sh -c '...'` → "
            "invoke the interpreter directly)."
        )
        actions.append(
            "After fixing the policy or the command, `[r]` Resume."
        )

    # ---- repair_loop_limit / persistent_build_failure ----------------
    # The two most generic triggers — derive specifics from state.
    elif trigger in {"repair_loop_limit", "persistent_build_failure",
                     "no_progress_failsafe"}:
        # Patcher allowlist rejections are the single most operator-fixable
        # cause — the LLM kept trying to write to paths the harness banned.
        if rejections:
            rej_paths = sorted({
                str(r.get("file", "")) for r in rejections if r.get("file")
            })[:5]
            actions.append(
                "The patcher's path allowlist rejected: "
                f"{', '.join(rej_paths) or '(see logs)'}. Decide if "
                "those paths SHOULD be writable for this project — if "
                "yes, add their roots to `config/config.json` → "
                "`patcher.root_files` (matches by basename) or "
                "`patcher.allowed_paths` (prefix match)."
            )
        # READ_FILE failures / missing-context patterns — surface specific files.
        if errors:
            files_with_errors = sorted({
                str(e.get("file", "")) for e in errors[:8] if e.get("file")
            })[:5]
            if files_with_errors:
                actions.append(
                    "Open these files in your IDE and read the "
                    f"diagnostics top-to-bottom: {', '.join(files_with_errors)}. "
                    "Look for an obvious fix the LLM kept missing "
                    "(wrong import path, stale field name after a refactor, "
                    "etc.)."
                )
        # Patch-application failures (search-block misses) — the LLM's mental
        # model of the file is stale; manual edits or [m] resume helps.
        if patch_failures:
            actions.append(
                "The patcher kept failing to apply patches (likely "
                "REPLACE_BLOCK search misses). Make the fix manually "
                "in your IDE then `[m]` Pause → Enter, which clears "
                "the LLM's stale state and re-runs the compiler."
            )
        # Build-command shape mismatch (e.g. wrong Python version, missing
        # system lib in the docker image) — operator must change config.
        if docker_image:
            actions.append(
                "If the failure is environmental (e.g. wrong Python "
                f"version, missing system library), edit `sandbox.docker_image` "
                f"in `config/config.json` (currently `{docker_image}`) to "
                "an image that has the needed runtime, then `[r]` Resume."
            )
        if not actions:
            # Catch-all when no specific signal is present.
            actions.append(
                f"Open the workspace at `{workspace_path}` and inspect "
                "the diagnostics above. If you can spot the fix, make "
                "it in your IDE then `[m]` Pause → Enter. Otherwise "
                "`[e]` inject a hint that steers the next repair turn."
            )

    # ---- unknown / catch-all -----------------------------------------
    else:
        actions.append(
            f"Inspect the workspace at `{workspace_path}` and the "
            "diagnostics above. Manual fix in your IDE → `[m]` Pause → "
            "Enter to resume. To redirect the LLM instead, `[e]` and "
            "inject a hint."
        )

    # Universal tail — applies to every trigger.
    actions.append(
        "Once you've made the change, come back here and choose: "
        "`[r]` Resume (if you only edited config/docker_image), "
        "`[m]` Pause for manual edits → Enter (if you edited files in "
        "the workspace), `[e]` Inject hint (to steer the next repair "
        "turn), or `[s]` Save & Quit (resume later with `teane resume "
        f"--session-id {state.get('session_id', '<id>')}`)."
    )

    # Modified-files context — useful when the operator is about to
    # edit and wants to know what's already changed in this session.
    if modified_files and len(modified_files) <= 30:
        actions.append(
            "Files already modified this session (don't conflict with "
            "your manual edit): " + ", ".join(modified_files[:30])
        )

    return actions


def hitl_menu_loop(state: dict[str, Any]) -> dict[str, Any]:
    """
    Interactive stdin menu for the human_intervention_node.

    Presents the developer with structured options:
        [v] View active file diffs
        [r] Resume graph execution (re-run compilation node)
        [e] Inject manual hint instruction string for the repair node
        [m] Pause for manual edits (notifies harness to wait while you fix files in your IDE)
        [b] Increase session budget limit (+ $2.00)
        [q] Abandon session and execute Git rollback

    Returns updated state dict reflecting the developer's chosen action.
    """
    node_state = state.get("node_state", {})
    trigger = node_state.get("hitl_trigger", "unknown")
    budget_remaining = state.get("budget_remaining_usd", 0.0)
    # Show ``$remaining / $cap`` using the real session cap snapshot
    # (set by create_initial_state) rather than the previously
    # hardcoded $2.00 — the latter was misleading for runs configured
    # with a higher hard_cap_usd in .harness_config.json. Falls back
    # to $2.00 only for legacy states that don't carry the field.
    budget_initial = float(state.get("budget_initial_usd", 2.00) or 2.00)
    loop_counter = state.get("loop_counter", {})
    errors = state.get("compiler_errors", [])
    exit_code = state.get("exit_code", -1)
    modified_files = state.get("modified_files", [])
    workspace_path = state.get("workspace_path", os.getcwd())

    # Format error display
    error_text = "No compiler errors captured."
    if errors:
        error_lines = []
        for i, err in enumerate(errors[:5], 1):  # Show first 5 errors max
            error_lines.append(
                f"  [{i}] {err.get('file', '?')}:{err.get('line', 0)}:{err.get('column', 0)} "
                f"- {err.get('message', 'Unknown error')[:120]}"
            )
        error_text = "\n".join(error_lines)
        if len(errors) > 5:
            error_text += f"\n  ... and {len(errors) - 5} more errors."
    else:
        # No structured diagnostics — show raw build output instead
        raw_output = node_state.get("last_build_output", "")
        if raw_output:
            error_text = f"[No structured diagnostics. Raw build output (last 2000 chars):]\n{raw_output[-2000:]}"

    # Format diffs summary
    diffs_text = "No files modified."
    if modified_files:
        diffs_text = "Modified files:\n" + "\n".join(f"  - {f}" for f in modified_files)

    # Single source of truth for both the stdin menu and the rich
    # HTTP-webhook payload (option_labels). The web dashboard reads
    # option_labels to render a labeled dropdown instead of a free-text
    # input — keep these strings short and operator-friendly.
    menu_options: list[tuple[str, str]] = [
        ("v", "View active file diffs"),
        ("r", "Resume graph execution (re-run compilation node)"),
        ("e", "Inject manual hint instruction for the repair node"),
        ("m", "Pause for manual edits (wait while you fix files in your IDE)"),
        ("b", "Increase session budget limit (+ $2.00)"),
        ("s", "Save & Quit (resume later)"),
        ("q", "Abandon session and execute Git rollback"),
    ]

    # LLM-judgment escalation summary (#1). human_intervention_node attaches
    # this for loop-stuck triggers (repair_loop_limit, persistent_build_failure)
    # to replace the bare trigger string with a one-paragraph briefing.
    # Empty when the kill switch is off, no gateway, or the call failed.
    escalation_summary = str(node_state.get("hitl_escalation_summary", "") or "").strip()

    # Per-trigger checklist of what the operator should do OUTSIDE the
    # harness before resuming. Diagnoses (escalation_summary) tell the
    # operator what broke; this tells them what to actually do.
    outside_actions = _build_outside_harness_actions(state, trigger)

    while True:
        print()
        print("=" * 80)
        print(f"[HUMAN-IN-THE-LOOP INTERVENTION] Trigger: {trigger}")
        print(f"  Budget: ${budget_remaining:.4f} / ${budget_initial:.2f} | Loop Counter: {loop_counter.get('total_repairs', 0)}")
        print(f"  Exit Code: {exit_code}")
        print(f"  Modified Files: {len(modified_files)}")
        print("=" * 80)
        if escalation_summary:
            print()
            print("WHY THE LOOP STOPPED (LLM diagnosis):")
            print(escalation_summary)
        print()
        print("CRITICAL INFORMATION:")
        print(error_text)
        if outside_actions:
            print()
            print("-" * 80)
            print("FIX OUTSIDE THE HARNESS, THEN RESUME:")
            print("(the LLM can't make these changes itself — you need to)")
            print("-" * 80)
            for i, action in enumerate(outside_actions, 1):
                # Indent wrapped lines so the bullet stays readable in
                # narrow terminals. textwrap keeps long sentences from
                # running off the side of the screen.
                import textwrap as _textwrap
                wrapped = _textwrap.fill(
                    action, width=78, initial_indent=f"  {i}. ",
                    subsequent_indent="     ",
                )
                print(wrapped)
        print()
        print("Options:")
        for key, label in menu_options:
            print(f"  [{key}] {label}")
        print()

        from harness.hitl import get_channel as _get_channel
        # --hitl-repair=false / env auto-approve: take the [r] resume
        # default (matches the existing HARNESS_AUTO_APPROVE behaviour
        # — proceed when possible). Skips the prompt entirely.
        auto_resumed_this_round = False
        if not _hitl_gate_enabled("repair") or _gatekeeper_auto_approves():
            # 2026-07-04 fix — budget_exhausted trigger MUST NOT auto-
            # resume. The auto-resume path resets HITL trip counters
            # but leaves ``budget_remaining_usd`` at 0, so the next
            # dispatch immediately re-trips budget_exhausted and the
            # loop grinds until manual kill. Terminate cleanly instead:
            # choose ``q`` (abandon) so the graph exits with the
            # budget-terminated marker set on state, and cmd_build
            # returns the ``budget_exhausted`` exit code (3).
            # Budget-family triggers: BOTH ``budget_exhausted`` (raw
            # cap hit mid-dispatch) and ``budget_preflight`` (the
            # ``BudgetTooLowError`` gate that fires before repair_node
            # even dispatches — see graph.py where BudgetTooLowError
            # is raised) must terminate instead of auto-resuming. The
            # auto-resume path resets HITL trip counters but leaves
            # ``budget_remaining_usd`` at 0, so the next dispatch/
            # preflight check re-hits the same wall and the loop
            # grinds at ~10 events/sec until the process is killed
            # externally (session 4754c913 for budget_exhausted;
            # budget_preflight was the sibling gap flagged by the
            # 2026-07-07 repair-loop audit).
            if trigger in {"budget_exhausted", "budget_preflight"}:
                choice = "q"
                auto_resumed_this_round = False
                logger.warning(
                    "[HITL] %s in headless mode — terminating instead "
                    "of auto-resuming (spending more requires an "
                    "operator to raise token_budget.hard_cap_usd or "
                    "press [b]).",
                    trigger,
                )
                node_state["budget_terminated"] = True
                state["node_state"] = node_state
            else:
                # Per-session cap on consecutive headless auto-resumes.
                # See ``_HITL_AUTO_RESUME_CAP`` at module top for the
                # rationale. ``_reset_hitl_trip_counters`` (called
                # below when we DO auto-resume) doesn't touch this
                # counter, so it survives across resumes and eventually
                # forces a clean terminate instead of an unbounded
                # budget-drain loop when the underlying failure isn't
                # recoverable in headless mode.
                _lc_for_cap = state.get("loop_counter")
                if not isinstance(_lc_for_cap, dict):
                    _lc_for_cap = {}
                    state["loop_counter"] = _lc_for_cap
                _resumes_taken = int(
                    _lc_for_cap.get("hitl_auto_resumes_taken", 0) or 0
                )
                # Cap resolution: explicit state override wins, then
                # config.json's ``hitl.auto_resume_cap``, then the
                # module default. Both keys live on the config so the
                # termination banner's recovery-hint suggestion isn't
                # hollow.
                _hitl_config = (
                    (state.get("harness_config") or {}).get("hitl") or {}
                )
                _cap = int(
                    state.get("hitl_auto_resume_cap")
                    or _hitl_config.get("auto_resume_cap")
                    or _HITL_AUTO_RESUME_CAP
                )
                # Fix 1: per-trigger sub-cap. Session cap is the total
                # auto-resume budget across all triggers; the sub-cap
                # bounds any SINGLE trigger's contribution so one
                # exhausted failure class can't monopolize the pool
                # (session 44c5e194 termination: 1× test_gen +
                # 2× zero_patch_loop = 3 session cap, leaving no
                # slack for follow-on triggers).
                _cap_per_trigger = int(
                    state.get("hitl_auto_resume_cap_per_trigger")
                    or _hitl_config.get("auto_resume_cap_per_trigger")
                    or _HITL_AUTO_RESUME_CAP_PER_TRIGGER
                )
                # Per-trigger frequency so the termination banner (and
                # post-mortems) can point at WHICH failure classes ate
                # the auto-resume budget, not just the trigger that
                # happened to fire last. Dict lives on loop_counter so
                # it survives checkpoint round-trips.
                _per_trigger_raw = _lc_for_cap.get(
                    "hitl_auto_resumes_per_trigger"
                )
                _per_trigger: dict[str, int] = (
                    dict(_per_trigger_raw)
                    if isinstance(_per_trigger_raw, dict) else {}
                )
                _trigger_taken = int(_per_trigger.get(trigger, 0) or 0)
                _session_cap_hit = _resumes_taken >= _cap
                _trigger_cap_hit = _trigger_taken >= _cap_per_trigger
                if _session_cap_hit or _trigger_cap_hit:
                    # Direct-abandon: bypass the ``[q]`` handler's
                    # confirmation prompt. In headless mode the confirm
                    # channel returns False by default, so falling through
                    # to ``elif choice == "q":`` would just cancel the
                    # abandon and re-enter this cap-check on the next
                    # ``while True:`` iteration — a tight loop that spams
                    # this WARNING millions of times per second (session
                    # cec4d124 hit 18M repetitions in ~10 minutes).
                    _cap_reason = (
                        f"session cap {_resumes_taken}/{_cap}"
                        if _session_cap_hit else
                        f"per-trigger cap {_trigger_taken}/"
                        f"{_cap_per_trigger} for '{trigger}'"
                    )
                    logger.warning(
                        "[HITL] Auto-resume cap reached (%s) in headless "
                        "mode — terminating instead of looping. No "
                        "operator can act on this in a headless session; "
                        "further auto-resumes would just burn budget on "
                        "repeat escalation-summary calls without changing "
                        "the underlying failure.",
                        _cap_reason,
                    )
                    # Loud termination banner to stderr — the previous
                    # single-line WARNING was buried in verbose output
                    # and the finsearch session 5f65a887 operator had
                    # to reconstruct "why did the process exit" from
                    # log-tail archaeology. Give the exit reason a
                    # multi-line banner with the trigger frequency
                    # summary, session context, and copy-pasteable
                    # recovery hints.
                    _session_id = str(state.get("session_id") or "unknown")
                    _total_repairs = int(
                        loop_counter.get("total_repairs", 0) or 0
                    )
                    _budget_left = float(
                        state.get("budget_remaining_usd") or 0.0
                    )
                    _sorted_trigs = sorted(
                        _per_trigger.items(),
                        key=lambda kv: (-int(kv[1] or 0), kv[0]),
                    )
                    _trig_lines = [
                        f"  {name:<40s} {count:>3d} auto-resume(s)"
                        for name, count in _sorted_trigs
                    ] or ["  (no per-trigger accounting recorded)"]
                    _which_cap_line = (
                        f"Cap tripped:          {_cap_reason}\n"
                    )
                    if _session_cap_hit and _trigger_cap_hit:
                        _which_cap_line = (
                            f"Cap tripped:          {_cap_reason} "
                            "(both session AND per-trigger)\n"
                        )
                    _banner = (
                        "\n"
                        + "=" * 78 + "\n"
                        + "TERMINATED — HITL auto-resume cap exhausted "
                        "(headless mode)\n"
                        + "=" * 78 + "\n"
                        + f"Session:              {_session_id}\n"
                        + f"Last trigger:         {trigger}\n"
                        + _which_cap_line
                        + f"Total repairs:        {_total_repairs}\n"
                        + f"Budget remaining:     ${_budget_left:.4f}\n"
                        + "\n"
                        + "Auto-resumes by trigger:\n"
                        + "\n".join(_trig_lines) + "\n"
                        + "\n"
                        + "Recovery options:\n"
                        + "  1. Rerun with -y removed and a TTY so the "
                        "HITL menu is interactive.\n"
                        + "  2. Rerun with `--hitl-repair true` to opt "
                        "into interactive repair (still non-blocking\n"
                        + "     for other gates).\n"
                        + "  3. Raise the appropriate cap via config: "
                        f"{'`hitl.auto_resume_cap_per_trigger: 6`' if _trigger_cap_hit else '`hitl.auto_resume_cap: 10`'} "
                        "in config.json.\n"
                        + "     Higher caps trade budget for recovery "
                        "slack.\n"
                        + "  4. Inspect the workspace at the state "
                        f"above and manually address the '{trigger}'\n"
                        + "     failure, then rerun `teane patch`.\n"
                        + "=" * 78 + "\n"
                    )
                    try:
                        print(_banner, file=sys.stderr, flush=True)
                    except Exception:  # noqa: BLE001
                        pass
                    node_state["hitl_auto_resume_cap_hit"] = True
                    node_state["hitl_abandon"] = True
                    node_state["hitl_active"] = False
                    node_state["hitl_awaiting_input"] = False
                    state["node_state"] = node_state
                    try:
                        from harness.observability import log_failure as _lf
                        _lf(
                            "hitl_gate_blocked",
                            trigger=trigger,
                            session_id=state.get("session_id", ""),
                            loop_counter=loop_counter.get("total_repairs", 0),
                            modified_files=len(modified_files),
                            reason="auto_resume_cap_hit",
                            per_trigger_frequency=dict(_per_trigger),
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    return state
                else:
                    choice = "r"
                    auto_resumed_this_round = True
                    _lc_for_cap["hitl_auto_resumes_taken"] = (
                        _resumes_taken + 1
                    )
                    _per_trigger[trigger] = int(
                        _per_trigger.get(trigger, 0) or 0
                    ) + 1
                    _lc_for_cap["hitl_auto_resumes_per_trigger"] = _per_trigger
                    logger.info(
                        "[HITL] Repair menu auto-resumed "
                        "(--hitl-repair=false / CI / HARNESS_AUTO_APPROVE "
                        "/ no TTY). Auto-resume %d/%d for this session "
                        "(trigger '%s' cumulative: %d).",
                        _resumes_taken + 1, _cap, trigger,
                        _per_trigger[trigger],
                    )
        else:
            # Structured envelope for remote UIs (the dashboard's HITL
            # panel). The console path prints these above the prompt; the
            # webhook channel forwards them so the dashboard can render a
            # red trigger tag + markdown escalation summary + actionable
            # outside-harness fix list instead of falling through to the
            # generic JSON-dump renderer.
            hitl_metadata = {
                "hitl_trigger": trigger,
                "hitl_escalation_summary": escalation_summary,
                "outside_harness_actions": list(outside_actions or []),
            }
            choice = _get_channel().prompt(
                "[HITL] Select action",
                [k for k, _ in menu_options],
                default="r",
                option_labels={k: lbl for k, lbl in menu_options},
                metadata=hitl_metadata,
            )

        if choice == "v":
            print()
            print("--- Active File Diffs ---")
            print(diffs_text)
            print("-------------------------")

        elif choice == "r":
            # Resume: clear HITL flags, reset loop counter to allow one more repair attempt
            node_state["hitl_active"] = False
            node_state["hitl_awaiting_input"] = False
            node_state["hitl_resolved"] = True
            state["node_state"] = node_state
            # Reset iteration counters but preserve diagnostic trackers
            # (replace_block_misses_per_file, consecutive_zero_patch_rounds) so
            # the next repair iteration still sees the "use a different operation"
            # directive that broke the LLM out of REPLACE_BLOCK pattern-repetition.
            # Wiping them here re-opened the HITL ping-pong that this resume is
            # meant to escape — sessions 19b28eff, 0a5c6fe8, 2d0164f0.
            state["loop_counter"] = _reset_iteration_counters(
                state.get("loop_counter"), total_repairs=2,
            )
            # Auto-resume path only: also clear the counters that gate
            # ``route_after_compiler`` / ``route_after_patching`` HITL
            # escalation. In a TTY session the operator did something
            # outside the harness and those counters carry useful history;
            # in headless mode nothing changed, so preserving them
            # guarantees the very next router pass re-fires the same
            # trigger — a ~7s ping-pong that only stops when budget dies
            # (session 4754c913).
            if auto_resumed_this_round:
                _reset_hitl_trip_counters(state["loop_counter"])
                logger.info(
                    "[HITL] Auto-resume: HITL-trip counters cleared to "
                    "break the headless ping-pong."
                )
                # Clear env_misconfig state flags too. These are the
                # "compile / test-generation hit a condition the operator
                # is supposed to fix outside the harness" markers
                # (missing binary in the sandbox image, exhausted
                # test_generation_max_iterations, missing LLM API key,
                # etc.). ``route_after_compiler`` inspects them on
                # entry, so if we don't wipe them here the very next
                # router pass re-detects the same misconfig and
                # re-triggers HITL — which auto-resumes — which routes
                # back to compiler_node — which re-detects — infinite
                # loop at ~10 events/sec until the process is killed
                # externally. Session 21a638b4 demonstrated this on
                # ``test_generation_max_iterations`` (a per-batch soft
                # cap the batch already burned through; the correct
                # behaviour is to move on with what's been written, not
                # spin forever).
                _ns = state.get("node_state") or {}
                if isinstance(_ns, dict):
                    for _k in (
                        "env_misconfig",
                        "env_misconfig_symbol",
                        "llm_behavior",
                        "llm_behavior_symbol",
                        "build_command_blocked",
                        "build_command_blocked_rule",
                        "llm_silent",
                    ):
                        _ns.pop(_k, None)
                    state["node_state"] = _ns
            # Re-read on-disk config so operator edits between HITL triggers
            # (sandbox.docker_image, build_command, etc.) reach the in-memory
            # state. Without this the [r] Resume keeps using whatever
            # sandbox_config was checkpointed at the start of run_graph —
            # exactly the trap session 90d3a8d2 fell into: operator changed
            # docker_image from buildpack-deps:bookworm to python:3.12-slim
            # in config.json, but the state still pointed at the old image
            # so the build kept hitting "missing pip" and ping-ponging
            # straight back to HITL.
            _refresh_session_config_into_state(state)
            # The "2" refers to ``total_repairs`` — the seed value passed
            # to ``_reset_iteration_counters`` so the next compile has
            # one repair attempt remaining before the ``max_patch_repair_
            # iterations`` cap re-fires HITL. Per-file miss counts are
            # deliberately preserved so the repair LLM still gets the
            # "use a different operation" directive; router stall
            # tripwires are stepped one below their trip value so the
            # trigger that fired admits one repair round instead of
            # re-firing straight off the compile (session 22471c0c).
            # All of them reset to 0 on the next green compile or
            # successful code_review re-patch (see ``compiler_node`` /
            # ``code_review_node``).
            from harness.graph import hitl_next_node as _hitl_next_node
            _next_node = _hitl_next_node(str(trigger or ""))
            logger.info(
                "[HITL] Developer chose to resume. Iteration counters reset "
                "(total_repairs=2 — one more repair attempt before HITL "
                "re-fires). Routing to %s.",
                _next_node,
            )
            return state

        elif choice == "e":
            # Inject hint: append user string as a user message, reset loop counter to 1
            from harness.hitl import get_channel as _get_channel
            hint = _get_channel().notes("[HITL] Enter hint/instruction for the repair node")
            if hint:
                messages = state.get("messages", [])
                messages.append({"role": "user", "content": f"[HITL Hint]: {hint}"})
                state["messages"] = messages
                # Preserve diagnostic trackers — see comment in [r] branch.
                state["loop_counter"] = _reset_iteration_counters(
                    state.get("loop_counter"), total_repairs=1,
                )
                node_state["hitl_active"] = False
                node_state["hitl_awaiting_input"] = False
                node_state["hitl_resolved"] = True
                state["node_state"] = node_state
                logger.info(
                    "[HITL] Hint injected. Iteration counters reset "
                    "(total_repairs=1 — two repair attempts before HITL "
                    "re-fires). Resuming."
                )
                return state

        elif choice == "m":
            # Manual edits: wait for developer to fix files in IDE
            print("[HITL] Pausing for manual IDE edits...")
            print(f"[HITL] Workspace: {workspace_path}")
            print("[HITL] Make your changes in your editor, then press Enter to continue.")
            from harness.hitl import get_channel as _get_channel
            _get_channel().wait_for_manual_edit(workspace_path)
            # Manual IDE edits invalidate per-file miss history. Reset both
            # iteration counters and diagnostic trackers — the developer just
            # changed the file state under us, so the LLM's prior miss history
            # is no longer the right signal for the next iteration's prompt.
            state["loop_counter"] = {"patching": 0, "repair": 0, "compiler": 0, "total_repairs": 0}
            state["compiler_errors"] = []
            node_state["hitl_active"] = False
            node_state["hitl_awaiting_input"] = False
            node_state["hitl_resolved"] = True
            state["node_state"] = node_state
            from harness.graph import hitl_next_node as _hitl_next_node
            _next_node = _hitl_next_node(str(trigger or ""))
            logger.info(
                "[HITL] Manual edits confirmed. Compiler errors cleared. Resuming to %s.",
                _next_node,
            )
            return state

        elif choice == "b":
            # Increase budget by $2.00 and reset loop counter for a fresh attempt
            budget_remaining += 2.00
            state["budget_remaining_usd"] = budget_remaining
            # Preserve diagnostic trackers — see comment in [r] branch.
            state["loop_counter"] = _reset_iteration_counters(
                state.get("loop_counter"), total_repairs=0,
            )
            print(f"[HITL] Budget increased by $2.00. New budget: ${budget_remaining:.2f}. Loop counter reset.")
            continue  # Stay in the menu loop

        elif choice == "s":
            session_id = state.get("session_id", "")
            print()
            print("=" * 60)
            print("Session saved to checkpoint.")
            print("Resume later with:")
            print(f"  teane resume --session-id {session_id}")
            if workspace_path and workspace_path != os.getcwd():
                print(f"  teane resume --session-id {session_id} -w {workspace_path}")
            print("=" * 60)
            print()
            logger.info("[HITL] Session suspended by developer. Session: %s", session_id)
            node_state["hitl_suspend"] = True
            node_state["hitl_active"] = False
            node_state["hitl_awaiting_input"] = False
            node_state["suspended_from"] = "hitl_menu"
            state["node_state"] = node_state
            return state

        elif choice == "q":
            # Abandon: set abandon flag, route to END
            print("[HITL] Abandoning session...")
            from harness.hitl import get_channel as _get_channel
            confirmed = _get_channel().confirm(
                "[HITL] Confirm abandon? This will attempt a git rollback.", default=False
            )
            if confirmed:
                node_state["hitl_abandon"] = True
                node_state["hitl_active"] = False
                node_state["hitl_awaiting_input"] = False
                state["node_state"] = node_state
                _attempt_git_rollback(workspace_path)
                try:
                    from harness.observability import log_failure
                    log_failure(
                        "hitl_gate_blocked",
                        trigger=trigger,
                        session_id=state.get("session_id", ""),
                        loop_counter=loop_counter.get("total_repairs", 0),
                        modified_files=len(modified_files),
                    )
                except Exception:  # noqa: BLE001
                    pass
                logger.info("[HITL] Session abandoned. Git rollback attempted.")
                return state
            else:
                print("[HITL] Abandon cancelled.")
                continue

        else:
            print(f"[HITL] Unknown option: '{choice}'. Please choose from [v/r/e/m/b/s/q].")


# ---------------------------------------------------------------------------
# 2b. Requirement Refinement Layer (Pre-Flight Specification Gate)
# ---------------------------------------------------------------------------

# Trailer appended to the user message that carries the raw product notes.
# The skill file at ``harness/skills/docgen/requirements_doc.md`` is the
# system prompt; this trailer asks the LLM to apply that contract to the
# notes supplied below. Kept here (not in the skill file) so the skill
# remains a pure specification of the artifact rather than a chat turn.
_REQUIREMENTS_USER_TRAILER = (
    "\n\n## Instructions\n"
    "Apply the **Requirements Specification** skill instructions above to "
    "the raw product notes that follow. Emit the RSD as Markdown — no "
    "preamble, no postscript, no outer code fence. Begin with the "
    "machine-readable `<!-- RSD-META: ... -->` header described in §Output "
    "format rules.\n\n"
    "## Raw Product Notes\n{raw_notes}\n"
)


_REQUIREMENTS_OUTPUT_LANGUAGE_SUFFIX = (
    "\n\n---\n\n"
    "## MANDATORY OUTPUT LANGUAGE\n\n"
    "The synthesised SPEC_REQUIREMENTS.md MUST be written in English. "
    "Do not translate section headings, field labels, table headers, "
    "requirement text, acceptance criteria, or narrative into any other "
    "language regardless of the source notes' language or the model's own "
    "preference. Product names and code identifiers keep their original "
    "form; everything else is English prose. Non-English output is rejected "
    "by the trust boundary and aborts the run.\n"
)


def _load_requirements_doc_prompt(*, agile: bool, workspace_path: Optional[str] = None) -> str:
    """Resolve the requirements_doc system prompt with the agile-mode
    directive substituted in.

    The shipped prompt lives at ``harness/skills/docgen/requirements_doc.md``;
    a workspace can ship an override at ``{workspace_path}/skills/docgen/
    requirements_doc.md`` and the loader picks it first. The
    ``{AGILE_MODE_DIRECTIVE}`` placeholder is replaced based on the
    ``agile`` flag — see ``docgen_prompts.apply_agile_directive``.

    The English-output directive is appended unconditionally so that
    workspace prompt overrides still enforce it — the trust boundary
    (``validate_synthesized_spec``) rejects non-English output regardless,
    and this suffix keeps the prompt and the guard in sync.
    """
    from harness import docgen_prompts
    body = docgen_prompts.load("requirements_doc", workspace_path)
    body = docgen_prompts.apply_agile_directive(body, agile=agile)
    return body + _REQUIREMENTS_OUTPUT_LANGUAGE_SUFFIX


async def synthesize_requirements(
    manifest_path: str,
    output_dir: str,
    gateway: Any,
    *,
    agile: bool = False,
    workspace_path: Optional[str] = None,
) -> str:
    """
    Read raw notes from a manifest file, route to LLM for synthesis,
    and write SPEC_REQUIREMENTS.md to the output directory.

    Args:
        manifest_path: Path to the raw notes/text file.
        output_dir: Directory to write SPEC_REQUIREMENTS.md.
        gateway: Initialized LLM Gateway instance.
        agile: Resolved value of the ``--agile`` CLI flag (mirrored to
            ``args.decomposition_enabled`` by ``_resolve_agile_args``).
            True selects **Path A — Agile RSD** in the
            ``requirements_doc.md`` skill (SAFe Epic → Feature → Story
            hierarchy with Gherkin AC and INVEST validation). False
            selects **Path B — Default RSD** (ISO/IEC/IEEE 29148:2018).
        workspace_path: When supplied, a per-workspace prompt override
            at ``{workspace_path}/skills/docgen/requirements_doc.md`` is
            consulted first by the loader. Pass ``None`` (the default)
            for callers that have no workspace context yet.

    Returns:
        Absolute path to the generated SPEC_REQUIREMENTS.md file.

    Raises:
        FileNotFoundError: If manifest_path does not exist.
        RuntimeError: If LLM synthesis fails.
    """
    manifest_full = os.path.abspath(manifest_path)
    if not os.path.isfile(manifest_full):
        raise FileNotFoundError(f"Manifest file not found: {manifest_full}")

    logger.info("[requirements] Reading manifest: %s", manifest_full)
    try:
        import aiofiles
        async with aiofiles.open(manifest_full, "r", encoding="utf-8", errors="replace") as f:
            raw_notes = await f.read()
    except ImportError:
        with open(manifest_full, "r", encoding="utf-8", errors="replace") as f:
            raw_notes = f.read()

    if not raw_notes.strip():
        raise RuntimeError("Manifest file is empty.")

    logger.info(
        "[requirements] Synthesizing SPEC_REQUIREMENTS.md from %d chars of raw notes (agile=%s)...",
        len(raw_notes), agile,
    )

    from harness.gateway import NodeRole
    system_prompt = _load_requirements_doc_prompt(agile=agile, workspace_path=workspace_path)
    user_prompt = _REQUIREMENTS_USER_TRAILER.format(raw_notes=raw_notes)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        content_raw, cost_total = await _dispatch_with_continuation(
            gateway=gateway,
            messages=messages,
            role=NodeRole.PLANNING,
            budget_remaining_usd=2.00,
            log_label="requirements",
            cache_family="planning:requirements_synthesis",
        )
    except Exception as exc:
        raise RuntimeError(f"LLM synthesis failed: {exc}") from exc

    from harness.trust import validate_synthesized_spec
    content, trust_errors = validate_synthesized_spec(content_raw.strip())
    if trust_errors:
        raise RuntimeError(f"Synthesised spec failed trust validation: {trust_errors}")

    # Write the file
    os.makedirs(output_dir, exist_ok=True)
    spec_path = os.path.join(output_dir, "SPEC_REQUIREMENTS.md")
    try:
        import aiofiles
        async with aiofiles.open(spec_path, "w", encoding="utf-8") as f:
            await f.write(content)
    except ImportError:
        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(content)

    logger.info("[requirements] SPEC_REQUIREMENTS.md written to %s (%d chars, cost=$%.6f).",
                 spec_path, len(content), cost_total)
    return spec_path


def _read_spec_file(spec_path: str) -> str:
    """Read a specification file from disk.

    Relies on a single try/except instead of the previous isfile-then-open
    pattern, which had a microsecond TOCTOU window (Bug 7). open() raises
    FileNotFoundError (an OSError subclass) when the file doesn't exist,
    which the existing handler already catches — same observable behavior,
    no race.
    """
    try:
        with open(spec_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


# Planner-only section labels the code-emission LLM does not need — see
# _slim_spec_for_prompt below. Match anchor is the `**Label:**` bold
# marker at line start; the strip covers the label line PLUS any
# continuation lines up to the next `**Field:**` marker, section
# separator, or heading. Adding new labels here should be validated
# against a sample RSD before shipping.
_PLANNER_ONLY_SPEC_FIELDS = (
    "Business driver",
    "Success metrics",
    "Estimated size",
    "Priority",
    "Wave",
    "Iteration",
)


def _slim_spec_for_prompt(spec_text: str) -> str:
    """Remove planner-only sections from an agile RSD so the system prompt
    injection carries only the fields the code-emission LLM (patching /
    repair) actually needs.

    Kept: assumptions, epic / feature / story titles, vision, scope,
    out-of-scope, dependencies, acceptance criteria — everything that
    grounds *what* to build.

    Dropped: business drivers, success metrics, size, priority, wave —
    everything that grounds *whether or when* to build. Those are
    planner-only signals; carrying them into every LLM call inflates the
    system prompt (and its cache footprint) without changing what code
    the LLM writes.

    Finsearch session 156032347 shipped a 243 KB system prompt; this
    filter typically trims ~30% of that. Cache economics are preserved
    because the transform runs once at load time — the resulting text is
    still an immutable prefix across the session.

    No-op on empty input and on files that don't contain any of the
    stripped labels (waterfall SRS, plain markdown, ...); the ratio of
    removed bytes is logged upstream so a wildly-off filter is visible.
    """
    if not spec_text:
        return spec_text
    import re
    # One regex per field: match `**<Field>:** <anything until the next
    # bold-field marker OR a horizontal rule OR EOF>`. Non-greedy so a
    # long spec with many labels doesn't collapse into one match.
    fields_alt = "|".join(re.escape(f) for f in _PLANNER_ONLY_SPEC_FIELDS)
    pattern = re.compile(
        # Line-starting **Field:** capture the rest of that line PLUS any
        # continuation lines until we hit the next **Bold:** field, a
        # horizontal rule (`---`), a Markdown heading (`# ` at line start),
        # or the end of the string.
        rf"^\*\*(?:{fields_alt}):\*\*[^\n]*(?:\n(?!\*\*[A-Za-z][^:]*:\*\*|---|#).*)*\n?",
        flags=re.MULTILINE,
    )
    slimmed = pattern.sub("", spec_text)
    # Collapse runs of blank lines the strip leaves behind so the
    # rendered spec stays visually tidy for anyone dumping the prompt.
    slimmed = re.sub(r"\n{3,}", "\n\n", slimmed)
    return slimmed


async def _dispatch_with_continuation(
    *,
    gateway: Any,
    messages: list[dict[str, str]],
    role: Any,
    budget_remaining_usd: float,
    log_label: str,
    max_continuations: int = 3,
    cache_family: Optional[str] = None,
) -> tuple[str, float]:
    """Dispatch a planning-role completion that may exceed the model's
    per-call ``max_tokens``. When the LLM signals
    ``finish_reason == "length"`` we feed the partial back as an
    assistant turn and ask it to continue from where it stopped,
    repeating up to ``max_continuations`` extra cycles. Used by the
    spec-synthesis helpers so a 4096-token output cap doesn't truncate
    a multi-section Markdown document mid-sentence — that truncation
    is what produced session web-6d5ef9b18f6a's missing §4–§7 (and the
    downstream "backend code only, no frontend" build output).

    Returns ``(concatenated_content, total_cost_usd)``.
    """
    chunks: list[str] = []
    working = list(messages)
    total_cost = 0.0
    budget = budget_remaining_usd
    for cycle in range(max_continuations + 1):
        response, budget = await gateway.dispatch(
            messages=working,
            role=role,
            budget_remaining_usd=budget,
            cache_family=cache_family,
        )
        chunk = response.content or ""
        chunks.append(chunk)
        total_cost += response.usage.cost_usd
        # Stub gateways in tests historically omit finish_reason on
        # their canned response — treat missing field as a clean stop.
        if getattr(response, "finish_reason", "stop") != "length":
            break
        if cycle == max_continuations:
            logger.warning(
                "[%s] LLM still hit the output token cap after %d "
                "continuation cycle(s); accepting truncated output.",
                log_label, max_continuations,
            )
            break
        logger.info(
            "[%s] LLM hit output token cap (cycle %d/%d) — requesting "
            "continuation.",
            log_label, cycle + 1, max_continuations,
        )
        working = working + [
            {"role": "assistant", "content": chunk},
            {"role": "user", "content": (
                "You stopped mid-document because you hit the output "
                "token cap. Continue from EXACTLY where you left off — "
                "do not repeat any preceding heading, table, sentence, "
                "or partial word. Output only the remaining content."
            )},
        ]
    return "".join(chunks), total_cost


_ARCHITECTURE_SYNTHESIS_PROMPT = """You are a Principal Software Architect.
Read the approved SPEC_REQUIREMENTS.md below and produce a focused
SPEC_ARCHITECTURE.md that lays out the technical design the coding agent
will follow.

## Output Sections

### 1. Architecture Overview
- One paragraph naming the architectural style (monolith / modular monolith /
  layered / hex / event-driven / microservices) and the rationale tied to
  the requirements.

### 2. Component / Module Inventory
For each module the implementation will contain:
- **Module name** (use the file-system path the agent should create, e.g.
  `task_dispatcher/api.py`).
- Purpose (one sentence).
- Public surface — functions / classes / endpoints exposed.
- Dependencies on other modules (forward refs only — no cycles).

### 3. Data Model
- Entities, fields, types, relationships.
- Persistence mechanism (SQLite / Postgres / in-memory / file).
- Migration / schema-init strategy.

### 4. External Interfaces
- HTTP/gRPC endpoints with method, path, request/response shape.
- CLI commands / arguments.
- Message-queue topics / event payloads (where applicable).
- Third-party services consumed (with auth model).

### 5. Cross-Cutting Concerns
- Configuration (env vars, defaults, validation).
- Logging / observability hooks.
- Error handling strategy (where exceptions are raised vs. handled).
- Concurrency model (sync / async / threads / processes).
- Security boundaries (authn/authz placement).

### 6. Test Strategy
- Unit-test layout (directory, naming convention).
- Integration / E2E coverage targets.
- Fixtures / fakes / mocks the test suite will rely on.

### 7. Build & Run
- Dependency manifest file(s) — list every runtime AND dev dependency.
- Build command the harness will execute (matches the workspace setup).
- Run command for local development.

### 8. Required Inventory Blocks
{inventory_instruction}

## Approved Requirements Specification
{requirements}

## Formatting
Output clean, well-structured Markdown. Use proper headings, bullet points,
and fenced code blocks for file paths / endpoints / schemas. Do not include
prose outside the document. Do not restate the requirements verbatim —
reference them by FR-id when justifying a design decision.
Do NOT wrap the whole document in an outer ```markdown … ``` fence — emit
the Markdown body directly, starting with the first heading. Fences are
reserved for code blocks INSIDE the document."""


async def synthesize_architecture(
    requirements_path: str,
    output_dir: str,
    gateway: Any,
) -> str:
    """
    Read the approved SPEC_REQUIREMENTS.md and synthesize SPEC_ARCHITECTURE.md.

    Mirrors :func:`synthesize_requirements` but targets the architecture
    phase: takes a locked requirements spec and produces a technical-design
    document that the patching LLM will use as its blueprint. Without this,
    the harness skips straight from requirements to code generation with no
    explicit module/data-model/test-strategy guidance, and the LLM picks
    layouts ad-hoc — which is what produced the allowlist-rejected
    `task_dispatcher/...` patches in the TaskDispatcher run.

    Args:
        requirements_path: Absolute path to SPEC_REQUIREMENTS.md.
        output_dir: Directory to write SPEC_ARCHITECTURE.md.
        gateway: Initialized LLM Gateway instance.

    Returns:
        Absolute path to the generated SPEC_ARCHITECTURE.md file.

    Raises:
        FileNotFoundError: If requirements_path does not exist.
        RuntimeError: If LLM synthesis fails or produces empty content.
    """
    if not os.path.isfile(requirements_path):
        raise FileNotFoundError(f"Requirements spec not found: {requirements_path}")

    try:
        import aiofiles
        async with aiofiles.open(requirements_path, "r", encoding="utf-8", errors="replace") as f:
            requirements = await f.read()
    except ImportError:
        with open(requirements_path, "r", encoding="utf-8", errors="replace") as f:
            requirements = f.read()

    if not requirements.strip():
        raise RuntimeError("Requirements spec is empty.")

    logger.info(
        "[architecture] Synthesizing SPEC_ARCHITECTURE.md from %d chars of requirements...",
        len(requirements),
    )

    from harness.gateway import NodeRole
    from harness.architecture_inventory import ARCHITECTURE_INVENTORY_INSTRUCTION
    prompt = _ARCHITECTURE_SYNTHESIS_PROMPT.format(
        requirements=requirements,
        inventory_instruction=ARCHITECTURE_INVENTORY_INSTRUCTION,
    )
    messages = [
        {"role": "system", "content": "You are a technical architecture expert. Output clean, structured Markdown."},
        {"role": "user", "content": prompt},
    ]

    try:
        content_raw, cost_total = await _dispatch_with_continuation(
            gateway=gateway,
            messages=messages,
            role=NodeRole.PLANNING,
            budget_remaining_usd=2.00,
            log_label="architecture",
            cache_family="planning:architecture_synthesis",
        )
    except Exception as exc:
        raise RuntimeError(f"LLM architecture synthesis failed: {exc}") from exc

    from harness.trust import validate_synthesized_spec
    content, trust_errors = validate_synthesized_spec(content_raw.strip())
    if trust_errors:
        raise RuntimeError(f"Synthesised architecture spec failed trust validation: {trust_errors}")

    os.makedirs(output_dir, exist_ok=True)
    spec_path = os.path.join(output_dir, "SPEC_ARCHITECTURE.md")
    try:
        import aiofiles
        async with aiofiles.open(spec_path, "w", encoding="utf-8") as f:
            await f.write(content)
    except ImportError:
        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(content)

    logger.info(
        "[architecture] SPEC_ARCHITECTURE.md written to %s (%d chars, cost=$%.6f).",
        spec_path, len(content), cost_total,
    )
    return spec_path


# ---------------------------------------------------------------------------
# 2c. Installation Doc Synthesis (end-of-run, greenfield builds)
# ---------------------------------------------------------------------------

_INSTALLATION_SYNTHESIS_PROMPT = """You are a Principal DevOps Engineer producing
an INSTALLATION.md for a freshly generated application. The document must let a
new developer clone, install, configure, run locally, and (when applicable)
deploy the app — using ONLY commands and paths that exist in the inputs below.
Do NOT invent dependencies, scripts, env vars, or ports.

## Output Sections

### 1. Prerequisites
- Language runtimes and minimum versions (derive from telemetry / manifests).
- Required system tools (docker, docker-compose, make, etc.) ONLY if the
  inputs reference them.

### 2. Clone & Install
- The exact dependency-install commands implied by the manifests
  (`pip install -r requirements.txt`, `npm install`, `make install`, etc.).
- Where to run each command (e.g. `cd backend && ...`).

### 3. Configure
- Environment variables grouped by purpose (DB, auth, third-party).
- Default values when shown in `.env.example`; mark required vs. optional.
- Note: omit this section entirely if no env config is referenced anywhere.

### 4. Run Locally
- Run commands derived from the Build & Run section of SPEC_ARCHITECTURE.md
  and from manifests (Makefile targets, `npm run`, `uvicorn ...`).
- Default ports the app binds on (from telemetry port_hints or blueprint).

### 5. Run with Docker
- ONLY emit this section when a deployment blueprint is provided in the
  inputs. Commands: `docker compose up --build`. List the service names
  exposed and their host ports. Omit otherwise.

### 6. Verify
- Health endpoints (from blueprint healthchecks, or default `/health`
  when telemetry says a web framework is present).
- One smoke-test command per service (e.g. `curl http://localhost:PORT/`).

### 7. Troubleshooting
- 2–4 short entries for likely failures grounded in the actual stack
  (e.g. "Port X already in use", "DB connection refused — Postgres not
  ready", "Module not found — re-run install"). Avoid generic boilerplate.

## Inputs

### Workspace telemetry (JSON)
{telemetry_json}

### Architecture spec — Build & Run section
{architecture_build_run}

### Deployment blueprint (or "none" when --deploy-dev was not used)
{blueprint_json}

### Manifests (truncated to 4 KB each)
{manifests_block}

## Formatting
Output clean, well-structured Markdown starting with `# Installation`.
Use fenced code blocks for every command. Do NOT wrap the whole document
in an outer ```markdown … ``` fence — emit the body directly. Reference
files by their workspace-relative path. Do not restate sections the inputs
don't justify."""


_INSTALLATION_MANIFEST_GLOBS: tuple[str, ...] = (
    "requirements.txt", "requirements-dev.txt", "pyproject.toml",
    "package.json", "Makefile", "makefile", "GNUmakefile",
    ".env.example", "docker-compose.yml", "docker-compose.yaml",
)
_INSTALLATION_MANIFEST_MAX_BYTES = 4096
_INSTALLATION_ARCH_SECTION_MAX_BYTES = 6144


def _extract_arch_build_run(architecture_text: str) -> str:
    """Slice the '### 7. Build & Run' (or equivalent) section out of a
    SPEC_ARCHITECTURE.md body. Falls back to the trailing 6 KB when no
    section heading matches — the section is required by the synthesis
    prompt but the LLM may have used a slightly different heading."""
    if not architecture_text:
        return "(architecture spec not available)"
    lines = architecture_text.splitlines()
    start_idx: int = -1
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped.startswith("### ") and (
            "build & run" in stripped
            or "build and run" in stripped
            or "build/run" in stripped
        ):
            start_idx = i
            break
    if start_idx < 0:
        tail = architecture_text[-_INSTALLATION_ARCH_SECTION_MAX_BYTES:]
        return tail
    end_idx = len(lines)
    for j in range(start_idx + 1, len(lines)):
        if lines[j].startswith("### ") or lines[j].startswith("## "):
            end_idx = j
            break
    section = "\n".join(lines[start_idx:end_idx])
    return section[:_INSTALLATION_ARCH_SECTION_MAX_BYTES]


def _collect_installation_manifests(workspace_path: str) -> str:
    """Read up to ~5 manifest files from the workspace root and return them
    as a single annotated text block. Each entry is truncated to 4 KB so the
    prompt stays bounded regardless of repo size."""
    chunks: list[str] = []
    for name in _INSTALLATION_MANIFEST_GLOBS:
        path = os.path.join(workspace_path, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                body = f.read(_INSTALLATION_MANIFEST_MAX_BYTES + 1)
        except OSError:
            continue
        truncated = len(body) > _INSTALLATION_MANIFEST_MAX_BYTES
        body = body[:_INSTALLATION_MANIFEST_MAX_BYTES]
        suffix = "\n... (truncated)" if truncated else ""
        chunks.append(f"#### {name}\n```\n{body}{suffix}\n```")
        if len(chunks) >= 6:
            break
    if not chunks:
        return "(no manifest files found at workspace root)"
    return "\n\n".join(chunks)


def _slim_blueprint(blueprint: Optional[dict[str, Any]]) -> str:
    """Reduce a deployment blueprint to the install-doc-relevant fields
    (services with image/ports/healthchecks/env). Returns the literal
    string ``"none"`` when no blueprint exists so the prompt knows to
    skip §5."""
    if not blueprint or not isinstance(blueprint, dict):
        return "none"
    services_in = blueprint.get("services") or {}
    if not isinstance(services_in, dict) or not services_in:
        return "none"
    slim_services: dict[str, dict[str, Any]] = {}
    for name, svc in services_in.items():
        if not isinstance(svc, dict):
            continue
        slim_services[str(name)] = {
            k: svc[k] for k in ("image", "base_image", "ports", "healthcheck", "environment", "depends_on")
            if k in svc
        }
    if not slim_services:
        return "none"
    return json.dumps({"services": slim_services}, indent=2)[:6144]


async def synthesize_installation(
    workspace_path: str,
    architecture_path: str,
    output_dir: str,
    gateway: Any,
    *,
    blueprint: Optional[dict[str, Any]] = None,
) -> str:
    """
    Synthesize ``INSTALLATION.md`` for a freshly generated greenfield app.

    Reads workspace telemetry (deterministic scan), root manifests, the
    Build & Run section of SPEC_ARCHITECTURE.md, and the deployment
    blueprint (when --deploy-dev produced one). Routes the assembled
    inputs through the planning LLM and writes the rendered Markdown to
    ``<output_dir>/INSTALLATION.md``.

    Args:
        workspace_path: Absolute path to the generated project root.
        architecture_path: Absolute path to ``SPEC_ARCHITECTURE.md``.
        output_dir: Directory to write ``INSTALLATION.md`` (usually
            ``<workspace>/docs``).
        gateway: Initialized LLM Gateway instance.
        blueprint: Optional deployment blueprint dict (services / ports /
            healthchecks). Set when ``--deploy-dev true`` produced one;
            ``None`` for greenfield runs that skipped the deploy phase.

    Returns:
        Absolute path to the generated ``INSTALLATION.md`` file.

    Raises:
        FileNotFoundError: If ``workspace_path`` does not exist.
        RuntimeError: If LLM synthesis fails or trust validation rejects.
    """
    if not os.path.isdir(workspace_path):
        raise FileNotFoundError(f"Workspace not found: {workspace_path}")

    from harness.deploy import scan_workspace_telemetry
    telemetry = scan_workspace_telemetry(workspace_path)
    telemetry_json = json.dumps(telemetry, indent=2)[:6144]

    architecture_text = ""
    if architecture_path and os.path.isfile(architecture_path):
        try:
            with open(architecture_path, "r", encoding="utf-8", errors="replace") as f:
                architecture_text = f.read()
        except OSError:
            architecture_text = ""
    architecture_build_run = _extract_arch_build_run(architecture_text)

    manifests_block = _collect_installation_manifests(workspace_path)
    blueprint_json = _slim_blueprint(blueprint)

    logger.info(
        "[installation] Synthesizing INSTALLATION.md from telemetry "
        "(langs=%s, dbs=%s, ports=%s) + %d manifest block(s).",
        telemetry.get("languages"),
        telemetry.get("databases_detected"),
        telemetry.get("port_hints"),
        manifests_block.count("####"),
    )

    from harness.gateway import NodeRole
    prompt = _INSTALLATION_SYNTHESIS_PROMPT.format(
        telemetry_json=telemetry_json,
        architecture_build_run=architecture_build_run,
        blueprint_json=blueprint_json,
        manifests_block=manifests_block,
    )
    messages = [
        {"role": "system", "content": "You are a technical documentation expert. Output clean, structured Markdown."},
        {"role": "user", "content": prompt},
    ]

    try:
        content_raw, cost_total = await _dispatch_with_continuation(
            gateway=gateway,
            messages=messages,
            role=NodeRole.PLANNING,
            budget_remaining_usd=1.00,
            log_label="installation",
            cache_family="planning:installation_synthesis",
        )
    except Exception as exc:
        raise RuntimeError(f"LLM installation synthesis failed: {exc}") from exc

    from harness.trust import validate_synthesized_spec
    content, trust_errors = validate_synthesized_spec(content_raw.strip())
    if trust_errors:
        raise RuntimeError(f"Synthesised installation doc failed trust validation: {trust_errors}")

    os.makedirs(output_dir, exist_ok=True)
    install_path = os.path.join(output_dir, "INSTALLATION.md")
    try:
        import aiofiles
        async with aiofiles.open(install_path, "w", encoding="utf-8") as f:
            await f.write(content)
    except ImportError:
        with open(install_path, "w", encoding="utf-8") as f:
            f.write(content)

    logger.info(
        "[installation] INSTALLATION.md written to %s (%d chars, cost=$%.6f).",
        install_path, len(content), cost_total,
    )
    return install_path


async def _refine_requirements(
    spec_path: str,
    additional_notes: str,
    gateway: Any,
) -> str:
    """
    Refine an existing SPEC_REQUIREMENTS.md with additional user notes.
    Overwrites the file with the updated version.

    Returns the updated spec content.
    """
    current_spec = _read_spec_file(spec_path)
    if not current_spec:
        raise RuntimeError(f"Cannot read spec file for refinement: {spec_path}")

    refine_prompt = f"""You are reviewing and improving a software requirements specification.

## Current SPEC_REQUIREMENTS.md
{current_spec}

## User's Additional Notes / Feedback
{additional_notes}

## Task
Integrate the user's feedback into the specification. Keep the same structure
(Functional Requirements, System Scope, Technical Constraints, Edge Cases, NFRs).
Add, modify, or clarify sections as directed by the feedback. Output the complete
updated SPEC_REQUIREMENTS.md document."""

    from harness.gateway import NodeRole
    messages = [
        {"role": "system", "content": "You are a technical documentation expert. Output clean, structured Markdown."},
        {"role": "user", "content": refine_prompt},
    ]

    content_raw, _cost_total = await _dispatch_with_continuation(
        gateway=gateway,
        messages=messages,
        role=NodeRole.PLANNING,
        budget_remaining_usd=2.00,
        log_label="requirements:refine",
        cache_family="planning:requirements_refine",
    )

    content = content_raw.strip()
    if not content:
        raise RuntimeError("LLM returned empty content for specification refinement.")

    try:
        import aiofiles
        async with aiofiles.open(spec_path, "w", encoding="utf-8") as f:
            await f.write(content)
    except ImportError:
        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(content)

    logger.info("[requirements] SPEC_REQUIREMENTS.md refined (%d chars).", len(content))
    return content


async def interactive_review_loop(spec_path: str, gateway: Any) -> str:
    """
    Interactive terminal review loop for SPEC_REQUIREMENTS.md.

    Options:
        [A] Approve — Accept the specification as-is and proceed.
        [B] Refine — Provide additional notes to improve the spec (loops).
        [C] Manual — Open the file in your IDE, edit, press Enter to continue.

    Async because the refine branch awaits ``_refine_requirements``; the
    caller is already inside ``asyncio.run(cmd_run(...))`` so there's no
    new loop to start. Prior to this change the refine branch called
    ``asyncio.run`` inside the running loop and raised — the gate was
    effectively unusable.

    Args:
        spec_path: Absolute path to the SPEC_REQUIREMENTS.md file.
        gateway: Initialized LLM Gateway instance for refinement.

    Returns:
        The final approved specification content (to be used as messages[0]).
    """
    while True:
        spec_content = _read_spec_file(spec_path)
        spec_size = len(spec_content) if spec_content else 0

        # --hitl-requirement=false / env auto-approve: lock the
        # synthesised spec and return immediately, no operator
        # interaction. The in-graph REQUIREMENTS gatekeeper honours the
        # same flag, so both surfaces of the requirements gate share one
        # switch.
        if not _hitl_gate_enabled("requirement") or _gatekeeper_auto_approves():
            logger.info(
                "[requirements] Review auto-approved "
                "(--hitl-requirement=false / CI / HARNESS_AUTO_APPROVE / no TTY). "
                "Locking spec at %d chars.",
                spec_size,
            )
            return spec_content

        print()
        print("=" * 72)
        print("[REQUIREMENT REFINEMENT GATE]")
        print(f"  Specification: {spec_path}")
        print(f"  Size: {spec_size:,} characters")
        print("=" * 72)
        # Single labels dict drives both the stdin menu printout and the
        # dashboard dropdown option_labels — operators on the web UI saw
        # bare letters before this and had to memorise what each meant.
        review_options: dict[str, str] = {
            "a": "Approve — lock this specification and proceed to graph execution",
            "b": "Refine — provide additional notes to improve the specification",
            "c": "Manual — edit the file in your IDE, then press Enter to continue",
        }
        print()
        for _key, _label in review_options.items():
            print(f"[{_key.upper()}] {_label}")
        print()

        from harness.hitl import get_channel as _get_channel
        choice = _get_channel().prompt(
            "How would you like to handle the synthesized requirements specification?",
            list(review_options.keys()),
            default="a",
            option_labels=review_options,
        )

        if choice == "a":
            # Approve: return the current content as the locked spec
            logger.info("[requirements] Specification approved (%d chars).", spec_size)
            return spec_content

        elif choice == "b":
            # Refine: get feedback, send to LLM, overwrite, loop
            from harness.hitl import get_channel as _get_channel
            notes = _get_channel().notes("[Refine] Enter additional notes/feedback for the specification")

            if not notes:
                print("[Refine] No notes provided. Returning to menu.")
                continue

            print("[Refine] Updating specification with your feedback...")
            try:
                # Async path: the surrounding cmd_run loop owns the event
                # loop, so we await directly. Earlier code wrapped this in
                # asyncio.run() and tripped "loop already running".
                updated = await _refine_requirements(spec_path, notes, gateway)
                print(f"[Refine] Specification updated ({len(updated):,} chars).")
            except Exception as exc:
                print(f"[Refine] Error: {exc}")
            # Loop back to menu

        elif choice == "c":
            # Manual: pause for IDE edits, then read from disk
            print(f"[Manual] Edit the file at: {spec_path}")
            print("[Manual] Make your changes in your editor (VS Code, Cursor, etc.).")
            from harness.hitl import get_channel as _get_channel
            _get_channel().wait_for_manual_edit(spec_path)

            spec_content = _read_spec_file(spec_path)
            if spec_content:
                logger.info("[requirements] Manual edits confirmed (%d chars).", len(spec_content))
                return spec_content
            else:
                print("[Manual] Warning: Could not read the file. Returning to menu.")
                continue

        else:
            print(f"[Requirements] Unknown option: '{choice}'. Please choose A, B, or C.")


def _validate_product_spec_dir_name(config_value: str) -> Optional[str]:
    """Validate that ``config_value`` is a bare folder name suitable for
    use as a workspace-root subdirectory.

    Rules (the value must live inside the workspace root — no other
    locations are accepted):

    - Must be a non-empty string after stripping whitespace.
    - Must NOT be an absolute path.
    - Must NOT contain a path separator (``/`` or ``\\``).
    - Must NOT contain ``..`` (parent-directory traversal).
    - Must NOT start with ``~`` (home-directory expansion).
    - Must NOT be ``.`` or ``..``.

    Returns ``None`` on success, or a human-readable error string when
    the value violates one of the rules. Callers print + bail on a
    non-None return.
    """
    if not isinstance(config_value, str):
        return "must be a string"
    name = config_value.strip()
    if not name:
        return "must be a non-empty string"
    if os.path.isabs(name):
        return (
            f"must be a folder NAME at the workspace root (no leading "
            f"`/`). Got {config_value!r}; use something like \"product_spec\""
        )
    if "/" in name or "\\" in name:
        return (
            f"must be a folder NAME at the workspace root — no path "
            f"separators are allowed (`/` or `\\`). Got {config_value!r}; "
            f"use something like \"product_spec\""
        )
    if name.startswith("~"):
        return (
            f"must be a folder NAME at the workspace root (no `~` "
            f"home-directory shorthand). Got {config_value!r}"
        )
    if name in (".", ".."):
        return (
            f"must be a folder NAME (not `.` or `..`). Got {config_value!r}"
        )
    if ".." in name.split(os.sep):
        # Defensive: separators are already rejected, but catch any
        # platform-specific separator difference too.
        return (
            f"must not contain `..` components. Got {config_value!r}"
        )
    return None


def _resolve_product_spec_dir(workspace_path: str, config_value: str) -> str:
    """Resolve the ``product_spec_dir`` config value to an absolute path
    under the workspace root.

    The value is a bare folder name (validated separately by
    :func:`_validate_product_spec_dir_name`); this function joins it with
    ``workspace_path`` and normalises the result. Pure path arithmetic;
    no I/O.
    """
    return os.path.normpath(os.path.join(workspace_path, config_value.strip()))


def _load_consolidated_product_spec(
    workspace_path: str,
    resolved_spec_dir: str,
) -> Optional[str]:
    """Validate and consolidate the configured product-spec folder.

    ``resolved_spec_dir`` is the absolute path produced by
    :func:`_resolve_product_spec_dir` — it may live anywhere on disk, not
    necessarily inside ``workspace_path``. The folder must exist and
    contain one or more spec files (.txt / .md / .pdf —
    see :data:`harness.spec_files.SPEC_FILE_EXTS`). Reading order is
    alphabetical; each file's body is prefixed with a ``## <filename>``
    section header so the synthesis LLM can see file boundaries.

    Returns the consolidated content as a single string on success.
    Returns ``None`` and prints a clear, user-facing error to stderr on
    any of these failure modes:

    - configured folder missing.
    - configured folder exists but contains no spec files (``.txt`` /
      ``.md`` / ``.pdf`` — see :data:`harness.spec_files.SPEC_FILE_EXTS`).
    """
    product_spec_dir = resolved_spec_dir
    allowed_exts_str = ", ".join(SPEC_FILE_EXTS)

    def _fail(headline: str, body: str) -> None:
        print(file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        print(headline, file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        print(body, file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        logger.error("[product_spec] %s", headline)

    if not os.path.isdir(product_spec_dir):
        _fail(
            "Configured product_spec_dir does not exist",
            (
                f"The harness expected a directory at:\n\n"
                f"  {product_spec_dir}\n\n"
                "but it does not exist. `product_spec_dir` in config.json\n"
                "points there. Either create the directory and add one or\n"
                f"more spec files ({allowed_exts_str}) describing the\n"
                "product, OR update `product_spec_dir` to a directory that\n"
                "does exist. The config value can be an absolute path\n"
                "(anywhere on the filesystem) or a path relative to the\n"
                "workspace."
            ),
        )
        return None

    spec_files = list_spec_files(product_spec_dir)
    if not spec_files:
        _fail(
            f"Configured product_spec_dir contains no spec files ({allowed_exts_str})",
            (
                f"`{product_spec_dir}` exists but holds no spec files. Add\n"
                f"at least one {allowed_exts_str} file with the product\n"
                "specification and re-run."
            ),
        )
        return None

    sections: list[str] = [
        f"# Product Specification (consolidated from {len(spec_files)} file(s))",
        "",
        "Source files:",
        *(f"  - {f}" for f in spec_files),
        "",
    ]
    for fname in spec_files:
        fpath = os.path.join(product_spec_dir, fname)
        try:
            content = read_spec_file(fpath)
        except (OSError, ValueError) as exc:
            logger.warning(
                "[product_spec] Could not read %s: %s — skipping.", fpath, exc,
            )
            continue
        sections.append("---")
        sections.append(f"## {fname}")
        sections.append("")
        sections.append(content.rstrip())
        sections.append("")
    consolidated = "\n".join(sections)
    logger.info(
        "[product_spec] Consolidated %d file(s) from %s (%d chars).",
        len(spec_files), product_spec_dir, len(consolidated),
    )
    return consolidated


_DEFAULT_CHANGE_REQUESTS_DIR = "change_requests"
_CHANGE_REQUESTS_ARCHIVE_SUBDIR = "applied"
# Same pattern as harness.graph._CR_FILENAME_PREFIX. Duplicated here to
# avoid a cli → graph import dependency at module load (graph imports cli
# helpers in places). A single source of truth would require a small
# shared helpers module; the duplication is cheap and the regex is stable.
# Extension alternation matches harness.spec_files.SPEC_FILE_EXTS — keep
# the two in lockstep when adding a new spec extension.
_CR_FILENAME_PREFIX = re.compile(
    r"^CR-(\d+)(?:[-_].*)?\.(?:txt|md|pdf)$", re.IGNORECASE,
)


def _resolve_change_requests_dir(workspace_path: str, config_value: Optional[str]) -> str:
    """Resolve the ``change_requests_dir`` config value to an absolute path
    under the workspace root. Falls back to ``change_requests`` when the
    config key is absent. Same shape as :func:`_resolve_product_spec_dir`."""
    name = (config_value or _DEFAULT_CHANGE_REQUESTS_DIR).strip() or _DEFAULT_CHANGE_REQUESTS_DIR
    return os.path.normpath(os.path.join(workspace_path, name))


def _list_pending_change_request_files(change_requests_dir: str) -> list[str]:
    """Return the sorted list of spec filenames at the top of
    ``change_requests_dir`` (excluding the ``applied/`` archive). Returns
    an empty list when the directory is missing.

    Files are returned as basenames; callers join with ``change_requests_dir``
    to get absolute paths. Sorted alphabetically — the same order the
    ingest node uses to assign sequential CR-N IDs. Allowed extensions:
    see :data:`harness.spec_files.SPEC_FILE_EXTS`.
    """
    return list_spec_files(
        change_requests_dir,
        exclude=frozenset({_CHANGE_REQUESTS_ARCHIVE_SUBDIR}),
    )


def _archive_consumed_change_requests(
    change_request_files: list[dict[str, Any]],
    archive_target_dir: str,
    *,
    session_id: str,
    status: str,
    modified_files: list[str],
) -> None:
    """Move each consumed change-request file into the per-session archive
    and drop a ``manifest.json`` with run metadata.

    No-op when ``change_request_files`` is empty or ``archive_target_dir``
    is unset, so it can be called unconditionally at session end. The
    helper is intentionally tolerant — a file that has already been moved
    (re-run of the same session) is skipped silently; an unreadable
    source file is logged and skipped. The manifest is written even when
    every source is missing so the operator has a record of the run.
    """
    if not change_request_files or not archive_target_dir:
        return
    try:
        os.makedirs(archive_target_dir, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "[change_requests] Could not create archive %s: %s — skipping move.",
            archive_target_dir, exc,
        )
        return

    archived: list[dict[str, Any]] = []
    for rec in change_request_files:
        src = rec.get("abs_path", "")
        cr_id = rec.get("cr_id")
        original_name = rec.get("original_name", "")
        # Drop any existing CR-N prefix from the original filename so we
        # don't end up with CR-7-CR-7-foo.txt on operator-supplied IDs.
        # The regex requires a known spec extension, so `tail` always
        # keeps the original extension (.txt/.md/.pdf) — no rewriting.
        m = _CR_FILENAME_PREFIX.match(original_name) if original_name else None
        if m is not None:
            tail = original_name[m.end(1):].lstrip("-_")
            base_name = tail if tail else original_name
        else:
            base_name = original_name
        dst = os.path.join(archive_target_dir, f"CR-{cr_id}-{base_name}")
        if not src or not os.path.isfile(src):
            logger.info(
                "[change_requests] CR-%s source missing (%s) — already moved or "
                "deleted; skipping.", cr_id, src,
            )
            archived.append({"cr_id": cr_id, "archived_as": None, "source_missing": True})
            continue
        try:
            try:
                os.replace(src, dst)
            except OSError as exc:
                # Cross-filesystem rename (NFS, bind-mount, btrfs subvol)
                # produces EXDEV — os.replace can't span filesystems. Fall
                # back to shutil.move so the archive still works there.
                # Audit §5.9.
                import errno as _errno
                if getattr(exc, "errno", None) == _errno.EXDEV:
                    import shutil as _shutil
                    _shutil.move(src, dst)
                else:
                    raise
            archived.append({
                "cr_id": cr_id,
                "archived_as": os.path.basename(dst),
                "original_name": original_name,
            })
            logger.info("[change_requests] Archived CR-%s → %s", cr_id, dst)
        except OSError as exc:
            logger.warning(
                "[change_requests] Could not move %s → %s: %s", src, dst, exc,
            )
            archived.append({
                "cr_id": cr_id,
                "archived_as": None,
                "original_name": original_name,
                "error": str(exc),
            })

    manifest_path = os.path.join(archive_target_dir, "manifest.json")
    manifest = {
        "session_id": session_id,
        "status": status,
        "change_requests": archived,
        "modified_files": list(modified_files),
    }
    # Atomic write so a SIGKILL mid-write leaves the old manifest (or
    # nothing) — never a truncated JSON document that subsequent reads
    # fail to parse. Audit §5.10.
    try:
        from harness.metrics import write_atomic
        write_atomic(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[change_requests] Could not write manifest %s: %s",
            manifest_path, exc,
        )


def _list_workspace_entries_to_delete(
    workspace_path: str, spec_dirname: str,
) -> list[str]:
    """Enumerate the workspace-root entries that ``--new-build true``
    would delete. Mirrors the preserved-set logic in
    :func:`_perform_new_build_reset` so the preview shown to the
    operator matches what the destructive pass would actually touch.
    """
    if not os.path.isdir(workspace_path):
        return []
    preserved = frozenset({".git", spec_dirname})
    try:
        entries = sorted(os.listdir(workspace_path))
    except OSError:
        return []
    return [e for e in entries if e not in preserved]


def _list_orphan_patch_branches(workspace_path: str) -> list[str]:
    """Enumerate ``agent/patch-*`` branches in the workspace's git repo.
    Returns an empty list when the workspace is not a git repo or the
    git command fails. Matches what :func:`_perform_new_build_reset`
    would ``git branch -D``.
    """
    try:
        result = subprocess.run(
            ["git", "-C", workspace_path, "for-each-ref",
             "--format=%(refname:short)", "refs/heads/agent/patch-*"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []
    if result.returncode != 0:
        return []
    return sorted(b for b in result.stdout.split("\n") if b.strip())


async def _list_workspace_checkpoint_sessions(
    workspace_path: str, config: dict[str, Any],
) -> list[Any]:
    """Enumerate checkpoint sessions whose stored ``workspace_path``
    matches ``workspace_path`` (under ``os.path.realpath``). Mirrors the
    filter in :func:`_purge_workspace_checkpoints` so the preview shows
    exactly the sessions that would be deleted.
    """
    persistence_cfg = config.get("persistence", {}) or {}
    db_path = persistence_cfg.get("db_path", "~/.harness/checkpoints.db")
    if not os.path.isfile(os.path.expanduser(db_path)):
        return []
    try:
        from harness.storage import list_all_sessions
        sessions = await list_all_sessions(db_path, limit=10_000)
    except Exception:  # noqa: BLE001 — preview is best-effort
        return []
    ws_real = os.path.realpath(workspace_path)
    matches: list[Any] = []
    for s in sessions:
        s_ws = getattr(s, "workspace_path", "") or ""
        if not s_ws:
            continue
        try:
            if os.path.realpath(s_ws) == ws_real:
                matches.append(s)
        except OSError:
            continue
    return matches


def _describe_state_db_for_preview(workspace_path: str) -> Optional[str]:
    """If the global state.db has rows for this workspace's app name,
    return a short summary string (path + per-table row counts scoped
    to the app). Returns ``None`` when there's nothing to purge.
    """
    try:
        from harness import story_state as _story_state_mod
    except Exception:  # noqa: BLE001
        return None
    try:
        app_name = _story_state_mod.app_name_for_workspace(workspace_path)
    except ValueError:
        return None
    db_path = _story_state_mod.state_db_path()
    if not os.path.isfile(db_path):
        return None
    import sqlite3 as _sqlite3
    try:
        conn = _sqlite3.connect(db_path)
    except _sqlite3.DatabaseError:
        # Corrupt or unreadable — still announce that we'll touch it.
        return f"{db_path} (unreadable; purge will be attempted anyway)"
    try:
        counts: dict[str, int] = {}
        for table in ("stories", "batches", "defects", "commits"):
            try:
                row = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE workspace = ?",
                    (app_name,),
                ).fetchone()
                counts[table] = int(row[0]) if row else 0
            except _sqlite3.DatabaseError:
                counts[table] = 0
    finally:
        conn.close()
    if not any(counts.values()):
        return None
    summary = ", ".join(f"{k}={v}" for k, v in counts.items())
    return f"{db_path} for app {app_name!r} ({summary})"


def _describe_repo_index_for_preview(workspace_path: str) -> Optional[str]:
    """Return a one-line summary of repo_index.db rows tied to this
    workspace, or ``None`` when the index has nothing for it.
    """
    try:
        from harness.repo_index import get_stats
    except Exception:  # noqa: BLE001
        return None
    try:
        stats = get_stats(workspace_path)
    except Exception:  # noqa: BLE001
        return None
    if stats is None:
        return None
    return (
        f"repo_index.db ({stats.chunk_count} chunks across "
        f"{stats.file_count} file(s), built {stats.built_at})"
    )


def _print_new_build_preview(
    workspace_path: str,
    spec_dirname: str,
    files_to_delete: list[str],
    orphan_branches: list[str],
    checkpoint_sessions: list[Any],
) -> None:
    """Print a human-friendly preview of every destructive action
    ``--new-build true`` is about to take, so the operator can review
    before confirming."""
    print(file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print("--new-build true — REVIEW BEFORE PROCEEDING", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print(f"Workspace:           {workspace_path}", file=sys.stderr)
    print(f"Preserved at root:   `{spec_dirname}/`, `.git/`", file=sys.stderr)
    print(file=sys.stderr)

    if files_to_delete:
        print(
            f"Workspace files to DELETE from the base branch "
            f"({len(files_to_delete)} entries):",
            file=sys.stderr,
        )
        for entry in files_to_delete:
            print(f"  - {entry}", file=sys.stderr)
    else:
        print("Workspace files to delete: none.", file=sys.stderr)
    print(file=sys.stderr)

    if orphan_branches:
        print(
            f"Orphan agent/patch-* branches to DELETE "
            f"({len(orphan_branches)} branches):",
            file=sys.stderr,
        )
        for branch in orphan_branches:
            print(f"  - {branch}", file=sys.stderr)
    else:
        print("Orphan agent/patch-* branches: none.", file=sys.stderr)
    print(file=sys.stderr)

    if checkpoint_sessions:
        print(
            f"Checkpoint sessions + JSONL transcripts to PURGE "
            f"({len(checkpoint_sessions)} sessions):",
            file=sys.stderr,
        )
        for s in checkpoint_sessions:
            sid = getattr(s, "thread_id", "?")
            updated = getattr(s, "updated_at", "?")
            print(f"  - {sid}  (last updated {updated})", file=sys.stderr)
    else:
        print(
            "Checkpoint sessions for this workspace: none.",
            file=sys.stderr,
        )

    state_db_summary = _describe_state_db_for_preview(workspace_path)
    if state_db_summary:
        print(
            f"\nStory state DB to PURGE: {state_db_summary}",
            file=sys.stderr,
        )
    else:
        print(
            "\nStory state DB for this workspace: none.",
            file=sys.stderr,
        )

    repo_index_summary = _describe_repo_index_for_preview(workspace_path)
    if repo_index_summary:
        print(
            f"Repo index rows to PURGE: {repo_index_summary}",
            file=sys.stderr,
        )
    else:
        print(
            "Repo index rows for this workspace: none.",
            file=sys.stderr,
        )
    print("=" * 72, file=sys.stderr)


def _docker_wipe(workspace_path: str, entry: str) -> bool:
    """Delete ``<workspace_path>/<entry>`` from inside a throwaway
    ``alpine`` container so root-owned residue left by a prior
    ``teane deploy`` (a compose service that ran without ``user:``)
    can be cleared without escalating the harness itself.

    Runs ``docker run --rm -v <workspace>:/w alpine rm -rf /w/<entry>``.
    Times out at 60s so a wedged docker daemon can't stall reset. The
    bind mount is scoped to the workspace root, and the ``entry`` name
    is fixed by the parent's ``os.listdir`` result — no path traversal
    is possible because ``os.listdir`` never returns ``..`` or path
    separators.

    Returns True on success, False if docker is unavailable or the
    container exited non-zero. Best-effort — the caller logs and moves
    on either way.
    """
    if not shutil.which("docker"):
        return False
    if not entry or "/" in entry or entry in ("", ".", ".."):
        return False
    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{workspace_path}:/w",
                "alpine", "rm", "-rf", f"/w/{entry}",
            ],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning(
            "[new_build] docker-escalated wipe of %s raised: %s", entry, exc,
        )
        return False
    if result.returncode != 0:
        logger.warning(
            "[new_build] docker-escalated wipe of %s failed (rc=%d): %s",
            entry, result.returncode, (result.stderr or "").strip(),
        )
        return False
    logger.info(
        "[new_build] Escalated wipe of %s via alpine container "
        "(root-owned residue removed).", entry,
    )
    return True


def _perform_new_build_reset(
    workspace_path: str, spec_dirname: str,
    preserve_docs: bool = False,
) -> None:
    """When ``--new-build true`` fires, hard-reset the workspace.

    Four steps:

    1. Delete every row in the global ``~/.harness/state.db`` whose
       ``workspace`` column matches this workspace's app name
       (basename). Other apps' history is untouched. Runs FIRST,
       before any git operation, so a downstream failure can't leave
       stale stories/batches behind to be picked up by the next
       session.
    2. Checkout the base branch (``master`` if it exists, else ``main``).
    3. Delete every file / directory at the workspace root EXCEPT the
       preserved set (``.git/`` and the configured ``spec_dirname``,
       plus ``docs/`` when ``preserve_docs=True``) and commit the
       deletions on the base branch.
    4. Delete every orphaned ``agent/patch-*`` branch in the repo.

    ``preserve_docs`` is set by ``cmd_build`` when the operator opted
    to reuse the on-disk specs (see ``_resolve_reuse_docs``). The
    reset still purges state.db, but the ``docs/`` folder survives so
    the pre-graph synthesis step can read the preserved
    ``SPEC_REQUIREMENTS.md`` / ``SPEC_ARCHITECTURE.md`` instead of
    regenerating them via LLM.

    Runs BEFORE GitGuardian creates the new session's patch branch, so
    the new branch is forked from a now-clean base. Best-effort: any step
    that fails is logged but does not abort the harness — GitGuardian
    will still create the patch branch from whatever the working tree
    looks like after this function returns.

    When ``--git false`` (``_git_enabled()`` is False), steps 2 and 4
    are skipped and step 3 runs without a commit — the file deletion
    still happens so the workspace is cleaned for a fresh run, but no
    git subprocess calls are made. Step 1 (state.db row purge) runs
    unconditionally.
    """
    # Step 1: state.db row purge — runs unconditionally and BEFORE
    # anything else so even a hard failure later doesn't strand the
    # prior session's stories/batches in the global state.db.
    try:
        from harness import story_state as _story_state_mod
        _story_state_mod.purge_state_db(workspace_path)
    except Exception as exc:  # noqa: BLE001 — best-effort, log only
        logger.warning(
            "[new_build] state.db purge raised: %s — continuing with reset.",
            exc,
        )

    def _git(*args: str) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(
            ["git", "-C", workspace_path, *args],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=60,
        )

    git_mode = _git_enabled()

    if git_mode:
        if _git("rev-parse", "--git-dir").returncode != 0:
            logger.warning("[new_build] %s is not a git repo — skipping reset.", workspace_path)
            return

        base_branch: Optional[str] = None
        for candidate in ("master", "main"):
            if _git("rev-parse", "--verify", "--quiet", candidate).returncode == 0:
                base_branch = candidate
                break
        if base_branch is None:
            logger.error(
                "[new_build] Neither 'master' nor 'main' branch exists in %s — "
                "cannot perform reset. Skipping.", workspace_path,
            )
            return

        logger.info("[new_build] Resetting workspace on base branch '%s'.", base_branch)
        checkout = _git("checkout", base_branch)
        if checkout.returncode != 0:
            logger.error(
                "[new_build] Failed to checkout '%s': %s — aborting reset.",
                base_branch, (checkout.stderr or "").strip(),
            )
            return
    else:
        base_branch = None
        logger.info(
            "[new_build] --git false — clearing workspace files without "
            "git operations (no checkout, no commit, no branch cleanup)."
        )

    # Preserved at workspace root. .git/ can't be deleted without
    # destroying the repo; the configured product-spec folder is the
    # source of truth for the next run and must survive. `docs/` is
    # added when the operator opted to reuse prior specs — see
    # `_resolve_reuse_docs` and the preserve_docs param.
    preserved_set = {".git", spec_dirname}
    if preserve_docs:
        preserved_set.add("docs")
        logger.info(
            "[new_build] Preserving docs/ per operator choice — spec "
            "regeneration will be skipped."
        )
    preserved = frozenset(preserved_set)
    deleted = 0
    for entry in os.listdir(workspace_path):
        if entry in preserved:
            continue
        full = os.path.join(workspace_path, entry)
        try:
            if os.path.islink(full) or not os.path.isdir(full):
                os.remove(full)
            else:
                shutil.rmtree(full)
            deleted += 1
        except PermissionError as exc:
            # Root-owned residue from a prior `teane deploy` — the compose
            # stack ran a container without `user:` and its process left
            # files chowned to uid 0 on the host bind mount. The harness
            # runs unprivileged, so os.remove/rmtree cannot unlink them.
            # Escalate through a throwaway alpine container that has real
            # root inside its own userns and can rm anything on the mount.
            if _docker_wipe(workspace_path, entry):
                deleted += 1
            else:
                logger.warning(
                    "[new_build] Could not delete %s: %s "
                    "(docker escalation unavailable or failed — root-owned "
                    "residue may remain; run `sudo rm -rf %s` manually).",
                    entry, exc, full,
                )
        except OSError as exc:
            logger.warning("[new_build] Could not delete %s: %s", entry, exc)
    logger.info("[new_build] Deleted %d entry/entries from workspace root.", deleted)

    if not git_mode:
        # File deletion done; nothing else to do without git.
        return

    add = _git("add", "-A")
    if add.returncode != 0:
        logger.warning("[new_build] `git add -A` failed: %s", (add.stderr or "").strip())

    staged = _git("diff", "--cached", "--name-only")
    if staged.returncode == 0 and staged.stdout.strip():
        commit = _git("commit", "-m", "harness: --new-build reset")
        if commit.returncode == 0:
            logger.info(
                "[new_build] Committed reset on '%s' (deleted %d entry/entries).",
                base_branch, deleted,
            )
        else:
            logger.warning(
                "[new_build] git commit failed: %s",
                (commit.stderr or "").strip(),
            )
    else:
        logger.info("[new_build] No changes to commit on '%s'.", base_branch)

    branches = _git("for-each-ref", "--format=%(refname:short)", "refs/heads/agent/patch-*")
    if branches.returncode == 0 and branches.stdout.strip():
        deleted_branches = 0
        for branch in branches.stdout.split("\n"):
            branch = branch.strip()
            if not branch:
                continue
            result = _git("branch", "-D", branch)
            if result.returncode == 0:
                deleted_branches += 1
            else:
                logger.warning(
                    "[new_build] Could not delete branch %s: %s",
                    branch, (result.stderr or "").strip(),
                )
        if deleted_branches:
            logger.info(
                "[new_build] Deleted %d orphaned agent/patch-* branch(es).",
                deleted_branches,
            )


async def _purge_workspace_checkpoints(
    workspace_path: str, config: dict[str, Any],
) -> None:
    """Delete every checkpoint session (and per-session JSONL transcript)
    whose stored ``workspace_path`` matches the workspace being reset.

    Used by ``--new-build true`` cleanup so that "starting fresh" includes
    the persistence layer, not just the working tree. Session ↔ workspace
    association is indirect (the workspace path lives in the serialized
    LangGraph checkpoint blob under ``channel_values.workspace_path``),
    so we enumerate sessions via :func:`harness.storage.list_all_sessions`
    — the same canonical path ``teane status`` already uses — and match
    by ``os.path.realpath`` to absorb symlink aliases.

    Best-effort: failure to enumerate or delete is logged + swallowed so
    the harness still proceeds with the rest of session startup.
    """
    persistence_cfg = config.get("persistence", {}) or {}
    db_path = persistence_cfg.get("db_path", "~/.harness/checkpoints.db")
    ttl_days = persistence_cfg.get("ttl_days", 30)
    expanded_db = os.path.expanduser(db_path)
    if not os.path.isfile(expanded_db):
        logger.info(
            "[new_build] No checkpoint DB at %s — nothing to purge.", db_path,
        )
        return

    try:
        from harness.storage import HarnessAsyncSqliteSaver, list_all_sessions
        # Bump the limit well past anything a normal operator would ever
        # accumulate so a single pass picks up every match.
        sessions = await list_all_sessions(db_path, limit=10_000)
    except Exception as exc:  # noqa: BLE001 — best-effort cleanup
        logger.warning(
            "[new_build] Could not enumerate checkpoint sessions: %s — "
            "skipping checkpoint purge.", exc,
        )
        return

    ws_real = os.path.realpath(workspace_path)
    matches = []
    for s in sessions:
        s_ws = getattr(s, "workspace_path", "") or ""
        if not s_ws:
            continue
        try:
            if os.path.realpath(s_ws) == ws_real:
                matches.append(s)
        except OSError:
            continue
    if not matches:
        logger.info(
            "[new_build] No prior checkpoints for workspace %s.", workspace_path,
        )
        return

    # Delete from the SQLite store via the same async path cmd_purge uses.
    deleted_rows = 0
    try:
        checkpointer = await HarnessAsyncSqliteSaver.from_db_path(
            db_path=db_path, ttl_days=ttl_days,
        )
        try:
            for s in matches:
                try:
                    await checkpointer.adelete_thread(s.thread_id)
                    deleted_rows += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[new_build] Could not delete checkpoint thread %s: %s",
                        s.thread_id, exc,
                    )
        finally:
            try:
                await checkpointer.conn.close()
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[new_build] Checkpoint store open failed: %s — JSONL logs still "
            "cleaned below.", exc,
        )

    # Per-session JSONL transcripts. Same path + glob shape that
    # cmd_purge --session-id already handles. Best-effort per file.
    log_cfg = config.get("logging", {}) or {}
    log_dir = os.path.expanduser(log_cfg.get("log_dir", "~/.harness/logs"))
    removed_logs = 0
    if os.path.isdir(log_dir):
        import glob as _glob
        for s in matches:
            sid = s.thread_id
            for pat in (
                os.path.join(log_dir, f"{sid}.jsonl"),
                os.path.join(log_dir, f"{sid}.jsonl.*"),
            ):
                for path in _glob.glob(pat):
                    try:
                        os.remove(path)
                        removed_logs += 1
                    except OSError as exc:
                        logger.warning(
                            "[new_build] Could not remove log file %s: %s",
                            path, exc,
                        )

    logger.info(
        "[new_build] Purged %d checkpoint session(s) and %d JSONL log "
        "file(s) for workspace %s.",
        deleted_rows, removed_logs, workspace_path,
    )


def _purge_workspace_repo_index(workspace_path: str) -> None:
    """Drop every ``repo_meta`` / ``repo_chunks`` row tied to this
    workspace from ``~/.harness/repo_index.db``.

    Best-effort: import or open failure logs and returns. The repo
    index is rebuilt on demand by the planner anyway, so a missed
    purge here at worst leaves stale chunks that the next index build
    will overwrite — never a correctness issue, only a "lingering
    reference" we promised to clean.
    """
    try:
        from harness.repo_index import purge_workspace as _purge_idx
        _purge_idx(workspace_path)
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.warning(
            "[new_build] repo_index purge raised for %s: %s",
            workspace_path, exc,
        )


def _attempt_git_rollback(workspace_path: str) -> None:
    """Attempt a git checkout to restore modified files to their original state.

    No-op when ``--git false`` — without a repo there's no rollback target,
    so the workspace stays in whatever state the failure produced. The log
    line makes that explicit so the operator knows their files weren't
    silently restored.
    """
    if not _git_enabled():
        logger.info(
            "[HITL] Git rollback skipped: --git false. Workspace files "
            "remain in the state the failure left them in."
        )
        return
    import subprocess
    try:
        result = subprocess.run(
            ["git", "-C", workspace_path, "checkout", "--", "."],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if result.returncode == 0:
            logger.info("[HITL] Git rollback successful.")
        else:
            logger.warning("[HITL] Git rollback failed: %s", result.stderr.strip())
    except Exception as exc:
        logger.warning("[HITL] Git rollback error: %s", exc)


# ---------------------------------------------------------------------------
# MCP pool helper (shared by cmd_run / cmd_resume / cmd_doctor)
# ---------------------------------------------------------------------------

_mcp_pool_registry: list[Any] = []  # active pools — drained on shutdown


def _register_pool_for_shutdown(pool: Any) -> None:
    """Track ``pool`` so the atexit + signal handlers can drain it on
    process exit. Keeping the list global rather than threading through
    state mirrors how ``set_command_validator`` already does it."""
    _mcp_pool_registry.append(pool)


def _sync_kill_mcp_subprocesses() -> None:
    """Synchronous backstop for the asyncio drain path. Runs at
    ``atexit`` so subprocess leak survival of a hard crash / unhandled
    exception is bounded — every spawned MCP subprocess gets a SIGTERM
    on its process group.

    The clean shutdown path (``McpClientPool.shutdown``) handles SIGTERM
    + grace + SIGKILL on its own; this hook only kicks in when that path
    didn't get a chance to run. Best-effort; failures are silent.
    """
    import os as _os
    import signal as _signal
    if not hasattr(_os, "killpg"):
        # Windows has no process groups; the async drain path uses
        # taskkill /T /F via _platform.kill_process_tree. There's no
        # synchronous equivalent here, so just bail.
        return
    for pool in list(_mcp_pool_registry):
        clients = getattr(pool, "clients", None) or {}
        for client in clients.values():
            proc = getattr(client, "_proc", None)
            if proc is None or proc.returncode is not None:
                continue
            try:
                _os.killpg(_os.getpgid(proc.pid), _signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                continue


atexit.register(_sync_kill_mcp_subprocesses)


async def _drain_mcp_pools() -> None:
    """Shut down every registered pool. Idempotent."""
    pools = list(_mcp_pool_registry)
    _mcp_pool_registry.clear()
    for pool in pools:
        try:
            await pool.shutdown()
        except Exception as exc:  # noqa: BLE001 — shutdown is best-effort
            logger.debug("[cli:mcp] pool shutdown error: %s", exc)


def _append_repo_memory_safely(
    *,
    workspace_path: str,
    session_id: str,
    prompt_summary: str,
    modified_files: list[str],
    exit_code: int,
    config: dict[str, Any],
    extra_notes: Optional[str] = None,
) -> None:
    """Append a one-line session entry to the per-repo memory file.

    Wrapped: any failure (config disabled, write error, import error)
    logs and returns silently. The caller never sees an exception —
    memory is best-effort observability, not load-bearing for the run.
    """
    try:
        from harness.repo_memory import RepoMemoryConfig, append_session_note
        mem_cfg = RepoMemoryConfig.from_config(config)
        if not mem_cfg.enabled:
            return
        append_session_note(
            workspace_path,
            session_id=session_id,
            prompt_summary=prompt_summary,
            modified_files=modified_files,
            exit_code=exit_code,
            cfg=mem_cfg,
            extra_notes=extra_notes,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("[cli] repo memory append skipped: %s", exc)


async def _post_mortem_finalize(
    final_state: dict[str, Any],
    exit_code: int,
    config: dict[str, Any],
    workspace_path: str,
) -> str:
    """Learning-loop finalize (Hook B — the single write decision point).

    Returns the ``[learned-rule:...]`` note to append to repo memory via
    ``extra_notes`` ("" = nothing to record). Clean runs record nothing and
    retire every active rule instead — a green run is proof the recorded
    failure classes no longer bite. Failed runs use the note staged by
    human_intervention_node when present, else generate one here (covers
    failed runs that never reached HITL). Duplicate failure classes are
    fingerprint-deduped against the existing memory file.

    Fail-open like every other post-mortem surface: any exception logs
    and returns "" so finalization is never blocked.
    """
    try:
        from harness import post_mortem
        from harness.repo_memory import RepoMemoryConfig, read_repo_memory

        pm_cfg: dict[str, Any] = dict(config.get("post_mortem") or {})
        mem_cfg = RepoMemoryConfig.from_config(config)

        if exit_code == 0:
            if pm_cfg.get("retire_on_clean_run", True):
                retired = post_mortem.retire_learned_rules(workspace_path, mem_cfg)
                if retired:
                    from harness.observability import emit_event
                    emit_event("post_mortem_rules_retired", count=retired)
            return ""

        note = str(final_state.get("post_mortem_note") or "").strip()
        deterministic = False
        cost = 0.0
        if not note and pm_cfg.get("enabled", True):
            trigger = (
                str((final_state.get("node_state") or {}).get("hitl_trigger") or "")
                or f"exit_{exit_code}"
            )
            note, cost = await post_mortem.generate_post_mortem(
                final_state,
                trigger=trigger,
                escalation_summary=None,
                config=pm_cfg,
            )
            deterministic = cost == 0.0
        if not note:
            return ""

        parsed = post_mortem.parse_rule_note(note)
        if parsed is not None:
            trigger, fp = parsed
            existing = read_repo_memory(workspace_path, mem_cfg)
            if post_mortem.already_recorded(existing, trigger, fp):
                from harness.observability import emit_event
                emit_event("post_mortem_skipped", reason="duplicate",
                           trigger=trigger, fingerprint=fp)
                return ""
        from harness.observability import emit_event
        emit_event(
            "post_mortem_written",
            trigger=parsed[0] if parsed else "unknown",
            rule_chars=len(note),
            cost_usd=cost,
            deterministic=deterministic,
        )
        return note
    except Exception as exc:  # noqa: BLE001
        logger.debug("[cli] post-mortem finalize skipped: %s", exc)
        return ""


async def _maybe_start_mcp_pool(
    config: dict[str, Any],
    workspace_path: Optional[str] = None,
) -> Optional[Any]:
    """Build + start an :class:`McpClientPool` from ``config.mcp`` when
    ``mcp.enabled=true``. Registers every advertised tool into the
    SkillRegistry via ``register_mcp_skills``. Returns the pool (or
    ``None`` when disabled / failed) so the caller can hand it to
    later cleanup. Failures log and return ``None`` — MCP is additive,
    a bad config must not block the harness from running.

    When ``workspace_path`` is provided, the workspace stack is detected
    via :func:`impact._detect_workspace_stack` and passed to
    ``register_mcp_skills`` so servers declaring a ``tags`` list that
    doesn't overlap the workspace are skipped. Servers without declared
    tags are always registered. When ``workspace_path`` is None (legacy
    callers / tests), no filtering is applied.
    """
    try:
        from harness.mcp_client import (
            McpClientPool, McpPoolConfig, register_mcp_skills,
        )
    except Exception as exc:  # noqa: BLE001 — MCP optional
        logger.debug("[cli:mcp] mcp_client import skipped: %s", exc)
        return None
    pool_cfg = McpPoolConfig.from_config(config)
    if not pool_cfg.enabled:
        return None
    if not pool_cfg.servers:
        logger.info("[cli:mcp] enabled but no servers configured; skipping pool start.")
        return None
    pool = McpClientPool(pool_cfg)
    try:
        await pool.start()
    except ValueError as exc:
        # Filesystem-server safety gate, command-allowlist rejection, etc.
        logger.warning("[cli:mcp] pool refused to start: %s", exc)
        await pool.shutdown()
        return None
    except Exception as exc:  # noqa: BLE001
        logger.exception("[cli:mcp] pool failed to start: %s", exc)
        await pool.shutdown()
        return None
    workspace_tags: Optional[set[str]] = None
    if workspace_path:
        try:
            from harness.impact import _detect_workspace_stack
            workspace_tags = _detect_workspace_stack(workspace_path) or None
        except Exception as exc:  # noqa: BLE001 — workspace detection is best-effort
            logger.debug("[cli:mcp] workspace stack detection skipped: %s", exc)
    try:
        registered = register_mcp_skills(pool, workspace_tags=workspace_tags)
        logger.info("[cli:mcp] registered %d tool(s) from %d server(s) (workspace_tags=%s).",
                    registered, len(pool.clients),
                    sorted(workspace_tags) if workspace_tags else "<unfiltered>")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[cli:mcp] skill registration failed: %s", exc)
    _register_pool_for_shutdown(pool)
    return pool


async def _maybe_start_lsp_pool(
    config: dict[str, Any],
    workspace_path: str,
    flow: str,
) -> Optional[Any]:
    """Start the brownfield LSP navigation pool when ``lsp.enabled`` and
    the flow qualifies (never ``build`` — greenfield stays byte-identical).

    Registers the ``lsp__*`` navigation skills and publishes the pool via
    :func:`harness.lsp_client.set_active_pool` for graph-side prefetch.
    Reuses the MCP pool registry for drain + atexit cleanup (duck-typed:
    ``.shutdown()`` coroutine and ``.clients`` of objects carrying
    ``_proc``). Failures log and return ``None`` — LSP is additive; the
    DependencyGraph heuristics remain the fallback tier everywhere.
    """
    try:
        from harness.lsp_client import (
            LspClientPool, LspPoolConfig, register_lsp_skills, set_active_pool,
        )
    except Exception as exc:  # noqa: BLE001 — LSP optional
        logger.debug("[cli:lsp] import failed: %s", exc)
        return None
    pool_cfg = LspPoolConfig.from_config(config)
    if not pool_cfg.enabled:
        return None
    if flow == "build" or flow not in pool_cfg.enabled_flows:
        logger.debug("[cli:lsp] flow %r not lsp-enabled; pool not started.", flow)
        return None
    try:
        from harness.impact import _detect_workspace_stack
        workspace_tags = _detect_workspace_stack(workspace_path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[cli:lsp] workspace stack detection failed: %s", exc)
        workspace_tags = set()
    if not ({"python", "typescript", "node"} & workspace_tags):
        logger.info("[cli:lsp] no LSP-capable stack detected; pool not started.")
        return None
    pool = LspClientPool(pool_cfg, workspace_path)
    try:
        await pool.start(workspace_tags)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[cli:lsp] pool start failed: %s", exc)
        try:
            await pool.shutdown()
        except Exception:  # noqa: BLE001
            pass
        return None
    try:
        from harness.observability import emit_event
        emit_event(
            "lsp_pool_started",
            flow=flow,
            servers=sorted(pool.clients.keys()),
            skipped=pool.skipped,
        )
    except Exception:  # noqa: BLE001
        pass
    if not pool.clients:
        logger.info(
            "[cli:lsp] no servers started (skipped: %s) — navigation stays "
            "on heuristics.", pool.skipped,
        )
        return None
    try:
        register_lsp_skills(pool)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[cli:lsp] skill registration failed: %s", exc)
    set_active_pool(pool)
    _register_pool_for_shutdown(pool)
    logger.info(
        "[cli:lsp] pool up for flow %r: %s (skipped: %s)",
        flow, sorted(pool.clients.keys()), pool.skipped or "none",
    )
    return pool


# ---------------------------------------------------------------------------
# 3. Subcommand Handlers
# ---------------------------------------------------------------------------

async def _run_best_of_n_flow(
    args: argparse.Namespace, workspace_path: str, bon: Any,
) -> int:
    """Fan out ``bon.n`` teane subprocess trajectories in isolated worktrees
    and apply the winner. Returns 0 when a winner's diff landed, else 1.

    NOTE: this path drives full LLM solves across N subprocesses; its
    orchestration, worktree lifecycle, and winner-diff application are unit-
    tested (tests/test_best_of_n_runner.py), but the end-to-end multi-solve is
    only exercised against a live provider.
    """
    import sys
    import uuid
    from harness.best_of_n_runner import (
        make_subprocess_variant_runner,
        reconstruct_child_argv,
        run_best_of_n_build,
    )

    session = uuid.uuid7().hex
    base_argv = reconstruct_child_argv(list(sys.argv))
    runner = make_subprocess_variant_runner(base_argv)
    logger.info(
        "[best_of_n] fanning out %d trajectories (strategy=%s, concurrency=%d)...",
        bon.n, bon.strategy, bon.max_concurrency,
    )
    winner, results = await run_best_of_n_build(
        workspace_path, session, bon, variant_runner=runner,
    )
    n_ok = sum(1 for r in results if r.compiled_ok)
    if winner is None:
        logger.error(
            "[best_of_n] no trajectory produced an applicable green build "
            "(%d/%d reached green).", n_ok, len(results),
        )
        return 1
    logger.info(
        "[best_of_n] winner v%d applied (%d/%d green, %d files changed).",
        winner.variant_id, n_ok, len(results), winner.changed_files,
    )
    return 0


async def cmd_run(args: argparse.Namespace) -> int:
    """
    Execute the `teane run` subcommand.

    Steps:
        1. Resolve workspace path.
        2. Discover configuration.
        3. Resolve build command.
        4. Initialize checkpointer.
        5. Compile the graph.
        6. Execute the graph with the provided prompt.
        7. Handle HITL breakpoints if triggered.

    Examples:
        teane run -w /path/to/repo -p "Add JWT authentication"
        teane run -w ./myproject -p "Refactor the auth module" --new-build false
    """
    # Bare invocation: `teane run` with no --workspace and no --prompt.
    # Drop into the interactive setup wizard, which fills in args.workspace,
    # args.prompt, args.git, args.new_build, and args.spec_discovery before we
    # continue — OR, when the operator picks "resume existing session",
    # sets args.session_id and tells us to hand off to cmd_resume instead.
    # --prompt/-p is optional: product_spec/ (build) and change_requests/*.txt
    # (patch) are the authoritative source of the task. --workspace/-w
    # alone is a valid invocation. Only reject prompt-without-workspace,
    # since we can't infer where to run.
    workspace_given = getattr(args, "workspace", None) is not None
    prompt_given = getattr(args, "prompt", None) is not None
    if not workspace_given and not prompt_given:
        from harness.wizard import run_setup_wizard
        if run_setup_wizard(args) == "resume":
            return await cmd_resume(args)
    elif not workspace_given:
        print(
            "\nerror: --workspace/-w is required when --prompt/-p is given. "
            "To use the interactive setup, omit BOTH flags.\n",
            file=sys.stderr,
        )
        return 2

    # Normalize a missing prompt to empty string so downstream slicing /
    # logging paths don't need None-guards. product_spec/ or
    # change_requests/*.txt supplies the actual task description.
    if getattr(args, "prompt", None) is None:
        args.prompt = ""

    workspace_path = os.path.abspath(args.workspace)
    if not os.path.isdir(workspace_path):
        logger.error("Workspace path does not exist: %s", workspace_path)
        return 1
    if _refuse_if_workspace_is_harness_root(workspace_path):
        return 1

    # Fail fast on a broken environment (a missing/incompatible runtime
    # dependency) with a clean message + fix, rather than a raw ImportError
    # partway through startup once the graph/gateway build reaches it.
    _dep_exit = _preflight_dependency_guard()
    if _dep_exit is not None:
        return _dep_exit

    # Record git mode for every downstream code path that touches git
    # (GitGuardian init, _attempt_git_rollback, _perform_new_build_reset).
    # --git is now a bool (default False); the legacy str form
    # ("enable"/"disable") is gone. getattr default `True` covers tests
    # that construct args manually and expect git-on behaviour.
    _set_git_enabled(bool(getattr(args, "git", True)))

    # --yes only makes sense paired with --new-build. Reject the lone
    # use early so the operator sees a clean parser error rather than
    # a silent no-op far downstream.
    if getattr(args, "assume_yes", False) and not bool(getattr(args, "new_build", False)):
        print(
            "error: --yes can only be used with --new-build true",
            file=sys.stderr,
        )
        return 2

    # FIRST: deterministic config check. Reads + validates the canonical
    # config file with no side effects. Raises ConfigError (caught by
    # main()) on any problem — missing file, JSON syntax error, unknown
    # keys, missing required fields, wrong types, or missing API key env
    # vars for routed models. By running this before _acquire_workspace_lock
    # we avoid leaving a stale lock file when the operator's config is bad.
    config = discover_config(workspace_path)
    _preflight_config_env_report(config)

    # Opt-in trajectory-level best-of-N. Default off; a child variant sets
    # TEANE_BEST_OF_N_CHILD so it never re-enters (no fork bomb), and the
    # --best-of flag is stripped from the child argv as a second guard. When
    # active, fan out N teane subprocesses in worktrees and apply the winner.
    from harness.best_of_n_runner import BestOfNConfig, is_best_of_n_child
    _bon = BestOfNConfig.from_config(config)
    _best_of_flag = getattr(args, "best_of", None)
    if isinstance(_best_of_flag, int) and _best_of_flag > 1:
        _bon.enabled, _bon.n = True, _best_of_flag
    if _bon.is_active() and not is_best_of_n_child():
        return await _run_best_of_n_flow(args, workspace_path, _bon)

    # Pin the operator's --hitl-* choices for this run so the
    # human_gatekeeper_node / interactive_review_loop / repair-menu
    # callsites can short-circuit to auto-approve without re-reading
    # args. Three-tier precedence: CLI flag > config.json `hitl` > True.
    # See _resolve_hitl_flags + _hitl_gate_enabled in this module.
    _resolved_hitl = _resolve_hitl_flags(args, config)
    _set_hitl_flags(
        requirement=_resolved_hitl["requirement"],
        architecture=_resolved_hitl["architecture"],
        repair=_resolved_hitl["repair"],
        deployment=_resolved_hitl["deployment"],
        layout_divergence=_resolved_hitl["layout_divergence"],
    )
    logger.info(
        "[cli] HITL gates resolved: requirement=%s architecture=%s "
        "repair=%s deployment=%s layout_divergence=%s",
        _resolved_hitl["requirement"], _resolved_hitl["architecture"],
        _resolved_hitl["repair"], _resolved_hitl["deployment"],
        _resolved_hitl["layout_divergence"],
    )

    # P1.7: workspace-level advisory lock. Without this, two concurrent
    # `teane run -w <same workspace>` invocations both read and write
    # source files in interleaved order — silently corrupting each other's
    # patches. The lock holds for the lifetime of this process; the OS
    # releases it on exit. Pass --force-lock to override (e.g. recovering
    # from a crashed prior process that left the file stranded).
    workspace_lock_handle = _acquire_workspace_lock(
        workspace_path, force=getattr(args, "force_lock", False),
    )
    if workspace_lock_handle is False:
        # Another live session holds the lock and the operator didn't
        # opt into --force-lock. Refuse to proceed.
        return 1

    # NOTE: `build_command` is resolved LATER, after `_perform_new_build_reset`.
    # Resolving here would sniff the workspace as it exists BEFORE the reset —
    # so a leftover `backend/requirements.txt` from a previous run would seed
    # `cd backend && ...` into a build command that survives the reset, and
    # every subsequent build fails with `sh: cd: can't cd to backend`. See
    # _detect_default_build_command for the contract.

    # --- Change-request mode validation (before resource allocation) ---
    # Validate BEFORE initializing checkpointer/gateway/MCP, so early returns
    # don't leave resources hanging and causing cleanup exceptions.
    new_build_active = bool(getattr(args, "new_build", False))
    cr_dir_abs = _resolve_change_requests_dir(
        workspace_path, config.get("change_requests_dir"),
    )
    pending_change_requests = _list_pending_change_request_files(cr_dir_abs)
    if not new_build_active and not pending_change_requests:
        print(file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        print(
            "Existing-project run requires at least one change request",
            file=sys.stderr,
        )
        print("=" * 72, file=sys.stderr)
        print(
            "The harness needs at least one spec file (.txt / .md / .pdf) under:\n\n"
            f"  {cr_dir_abs}\n\n"
            "describing the bug to fix or feature to add. Each file becomes\n"
            "a numbered Change Request (CR-N) that flows through the\n"
            "gatekeeper review and is archived after the session terminates.\n\n"
            "To proceed:\n"
            "  1. Create the folder if it does not exist.\n"
            "  2. Add one or more .txt / .md / .pdf files describing the changes.\n"
            "  3. Re-run the same teane target.\n\n"
            "If you are starting a fresh build, pass --new-build true\n"
            "instead — that flow uses `product_spec_dir` and skips this\n"
            "check.\n",
            file=sys.stderr,
        )
        print("=" * 72, file=sys.stderr)
        return 1

    # Extract persistence settings
    persistence_cfg = config.get("persistence", {})
    db_path = persistence_cfg.get("db_path", "~/.harness/checkpoints.db")
    ttl_days = persistence_cfg.get("ttl_days", 30)
    redact_messages = bool(persistence_cfg.get("redact_messages", True))

    # Initialize checkpointer. With redact_messages=True (the default), the
    # checkpointer scrubs the `messages` channel through harness.redactor
    # before SQLite serialization, so secrets the user pasted into a prompt
    # never land at rest in checkpoints.db.
    from harness.storage import HarnessAsyncSqliteSaver, generate_session_id
    checkpointer = await HarnessAsyncSqliteSaver.from_db_path(
        db_path=db_path, ttl_days=ttl_days, redact_messages=redact_messages,
    )

    session_id = generate_session_id(args.session_id)
    # Bind the active session_id NOW so every LLM dispatch that happens
    # before the LangGraph runner is entered (spec synthesis,
    # architecture synthesis, doc review cycles) sees the real session
    # in Gateway._dump_llm_call_to_disk filenames instead of the default
    # "unknown" prefix. Without this the pre-graph dumps land at
    # ~/.harness/debug/unknown_NNNN_planning_*.txt and can't be grouped
    # with the in-graph dumps. ContextVar lasts until the process exits;
    # no explicit reset needed for a CLI command.
    from harness.observability import set_active_session_id
    set_active_session_id(session_id)

    # Configure structured logging / per-session log file
    from harness.observability import configure_logging
    log_cfg = config.get("logging", {})
    configure_logging(
        session_id=session_id,
        log_dir=log_cfg.get("log_dir", "~/.harness/logs"),
        level=log_cfg.get("level", "INFO"),
        langsmith_enabled=bool(log_cfg.get("langsmith", False)),
        json_stderr=bool(log_cfg.get("json_stderr", False)),
        max_bytes=int(log_cfg.get("max_bytes", 10_000_000)),
        backup_count=int(log_cfg.get("backup_count", 5)),
        console_level=log_cfg.get("console_level"),
    )

    # Extract budget and sandbox settings
    token_budget = config.get("token_budget", {})
    budget_usd = token_budget.get("hard_cap_usd", 2.00)
    # --allow-network is a real bool now (default true). Drop the
    # OR-truthy coalesce against config — the CLI default IS the
    # config-equivalent, and an explicit --allow-network false must
    # NOT be silently overridden by a stale `allow_network: true` in
    # config.json.
    allow_network = bool(getattr(args, "allow_network", True))

    # Reviewer cycle caps live in config.json now (the --spec-review-cycles
    # / --code-review-cycles CLI flags were removed). The gateway will
    # pick up config.node_throttle.max_doc_review_cycles /
    # max_code_review_cycles directly.

    # Initialize the LLM Gateway and inject it for graph nodes
    from harness.gateway import create_gateway_from_config
    from harness.graph import set_gateway, run_graph

    gateway = create_gateway_from_config(config)
    set_gateway(gateway)

    # Register built-in skills (pipeline + docgen + opt-in tool skills like
    # web_fetch / web_search). Wrapped: any failure inside the skill
    # registry must NOT block the harness from starting — the registry is
    # additive, not load-bearing for the core graph.
    try:
        from harness.skills import register_builtin_skills
        register_builtin_skills(config=config)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[cli] skill registration skipped: %s", exc)

    # Start the MCP client pool when ``mcp.enabled=true``. Each declared
    # MCP server spawns as a subprocess; their tools register into the
    # SkillRegistry under ``mcp__<server>__<tool>`` names so the graph's
    # tool-block interceptor can dispatch them. An atexit handler tears
    # the subprocesses down on a clean exit; Ctrl-C is handled by the
    # outer asyncio cancel path which also triggers the same shutdown.
    _mcp_pool = await _maybe_start_mcp_pool(config, workspace_path=workspace_path)
    # Brownfield-only LSP navigation pool (no-op for flow="build").
    _lsp_pool = await _maybe_start_lsp_pool(
        config, workspace_path, flow=getattr(args, "flow", "build"),
    )  # noqa: F841 — drained via the shared pool registry

    # Initialize the secret redactor
    from harness.redactor import create_redactor_from_config
    create_redactor_from_config(config)

    # Initialize the process-wide CommandValidator so every SandboxExecutor
    # spawned during this session inherits the configured allow/block lists.
    # Without this every executor falls back to validator=None (no check).
    from harness.security import (
        create_command_validator_from_config,
        set_command_validator,
    )
    set_command_validator(create_command_validator_from_config(config))

    # Determine change-request mode (validation already passed above)
    change_request_mode = bool(pending_change_requests) and not new_build_active
    archive_target_dir = (
        os.path.join(cr_dir_abs, "applied", session_id) if change_request_mode else ""
    )
    if change_request_mode:
        logger.info(
            "[change_requests] Change-request mode active. Pending: %s. "
            "Archive target: %s",
            pending_change_requests, archive_target_dir,
        )
        # Folder-driven runs are the source of truth — drop any -p prompt
        # so the operator can't accidentally double-source the run.
        prompt_arg = getattr(args, "prompt", "") or ""
        if prompt_arg.strip():
            logger.warning(
                "[change_requests] Both --prompt and %s are populated; "
                "the folder wins and --prompt is dropped.", cr_dir_abs,
            )
            args.prompt = ""

    # Read the operator-configured product-spec folder name once and
    # reuse for both the new_build cleanup (preserves the folder during
    # the workspace reset) and the requirement-refinement validation
    # below. The harness refuses to run without product_spec_dir in
    # config.json — this is the single source of truth for "where does
    # the product spec live." The value must be a bare folder name (no
    # separators, no absolute paths); the folder must exist at the
    # workspace root. validate_config_strict already rejects malformed
    # values at config-load time; this is the runtime defense-in-depth.
    spec_dirname_raw = config.get("product_spec_dir")
    name_error = _validate_product_spec_dir_name(spec_dirname_raw or "")
    if name_error is not None or spec_dirname_raw is None:
        print(file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        print("Invalid required config: product_spec_dir", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        print(
            "The harness requires a top-level `product_spec_dir` key in\n"
            "config.json with the NAME of a folder at the workspace root\n"
            "that holds the product-specification files (.txt / .md / .pdf).\n\n"
            f"Problem: {name_error or 'is required.'}\n\n"
            "Example:\n"
            "  \"product_spec_dir\": \"product_spec\"\n\n"
            "Then create the folder at the workspace root:\n"
            f"  mkdir {workspace_path}/product_spec\n",
            file=sys.stderr,
        )
        print("=" * 72, file=sys.stderr)
        logger.error("[config] product_spec_dir invalid or missing.")
        return 1
    spec_dirname = spec_dirname_raw.strip()
    resolved_spec_dir = _resolve_product_spec_dir(workspace_path, spec_dirname)

    # Validate the folder exists AND contains at least one spec file
    # (.txt / .md / .pdf).
    # product_spec_dir is the SOLE source for the product spec on greenfield
    # runs — the harness no longer accepts a --manifest override. We preload
    # the consolidated content here so the requirement-refinement step below
    # doesn't walk the folder a second time.
    #
    # In change-request mode (existing-project deltas) the product_spec
    # folder is not consulted — the change_requests/ folder drives the run
    # instead. The config value's NAME is still validated above so other
    # subsystems that reference spec_dirname (e.g. --new-build cleanup
    # preserves it) keep working.
    preloaded_consolidated_spec: Optional[str] = None
    if not change_request_mode:
        preloaded_consolidated_spec = _load_consolidated_product_spec(
            workspace_path, resolved_spec_dir,
        )
        if preloaded_consolidated_spec is None:
            # _load_consolidated_product_spec already printed a clear error to
            # stderr describing whether the folder is missing or empty.
            return 1

    # --new-build cleanup runs BEFORE GitGuardian creates the session's
    # patch branch, so the new branch forks from a clean base. The reset is
    # destructive (deletes most files at workspace root and commits the
    # deletions on master/main), but the operator opted in by passing the
    # flag — see _perform_new_build_reset's contract.
    if getattr(args, "new_build", False):
        # Build the deletion preview BEFORE touching anything so the
        # operator can review the exact list and bail if they hit the
        # flag by mistake or realise one of the files is still useful.
        files_to_delete = _list_workspace_entries_to_delete(
            workspace_path, spec_dirname,
        )
        orphan_branches = _list_orphan_patch_branches(workspace_path)
        checkpoint_sessions = await _list_workspace_checkpoint_sessions(
            workspace_path, config,
        )
        has_state_db = _describe_state_db_for_preview(workspace_path) is not None
        has_repo_index = _describe_repo_index_for_preview(workspace_path) is not None
        total_destructive = (
            len(files_to_delete)
            + len(orphan_branches)
            + len(checkpoint_sessions)
            + (1 if has_state_db else 0)
            + (1 if has_repo_index else 0)
        )
        if total_destructive == 0:
            # No destructive work — skip the prompt entirely. The
            # cleanup functions will be no-ops; we still call them so
            # the log line ("No prior checkpoints", etc.) appears.
            logger.info(
                "[new_build] --new-build true but nothing to clean "
                "(no extra files at workspace root, no orphan patch "
                "branches, no prior checkpoints, no state.db, no repo "
                "index rows for this workspace). Skipping the "
                "confirmation prompt."
            )
        else:
            _print_new_build_preview(
                workspace_path, spec_dirname,
                files_to_delete, orphan_branches, checkpoint_sessions,
            )
            if getattr(args, "assume_yes", False):
                logger.info(
                    "[new_build] --yes set — skipping the confirmation "
                    "prompt and proceeding with the reset."
                )
            else:
                from harness.hitl import _auto_approve as _hitl_auto_approve
                from harness.hitl import get_channel as _get_channel
                # Autonomy-first: auto-approve mode (CI=true,
                # HARNESS_AUTO_APPROVE=true, or non-TTY stdin) auto-declines
                # a `default=False` confirm(). Without this branch, a
                # non-interactive `teane build` silently exits after 10s of
                # subsystem init with no hint that `-y` was the missing flag.
                # Fail early instead, and point at the exact fix.
                if _hitl_auto_approve():
                    print(
                        "\nerror: `teane build` needs -y/--yes to auto-"
                        "approve the destructive workspace reset when "
                        "running non-interactively (CI=true, "
                        "HARNESS_AUTO_APPROVE=true, non-TTY stdin, or "
                        "--hitl-* false). Re-run with `-y` appended:\n"
                        "\n"
                        "  teane build -w <workspace> -p \"<prompt>\" ... -y\n"
                        "\n"
                        "The listing above shows exactly what -y would "
                        "delete. If you don't want those entries wiped, "
                        "back them up first or use `teane patch` instead.",
                        file=sys.stderr,
                    )
                    logger.warning(
                        "[new_build] Auto-approve declined the reset "
                        "(missing -y). Exiting."
                    )
                    return 1
                confirmed = _get_channel().confirm(
                    "Proceed with the destructive --new-build reset above?",
                    default=False,
                )
                if not confirmed:
                    print(
                        "\n--new-build reset cancelled. Re-run with `-y` to "
                        "auto-approve, or back up the entries above and try "
                        "again.",
                        file=sys.stderr,
                    )
                    logger.warning(
                        "[new_build] Operator declined the reset. Exiting."
                    )
                    return 1
        logger.warning(
            "[new_build] --new-build true — resetting workspace before "
            "starting the session. Files outside `%s/` and `.git/` will be "
            "deleted from the base branch.", spec_dirname,
        )
        _perform_new_build_reset(
            workspace_path, spec_dirname,
            preserve_docs=getattr(args, "reuse_docs", False),
        )
        # And purge every prior checkpoint + JSONL transcript that targeted
        # this workspace, so "fresh start" includes the persistence layer.
        # Runs BEFORE GitGuardian creates this session's patch branch and
        # before any checkpoint write for this session, so list_all_sessions
        # can't accidentally match (and delete) the run we're about to start.
        await _purge_workspace_checkpoints(workspace_path, config)
        # Repo index sits outside the workspace (in ~/.harness) so the
        # workspace rmtree above can't touch it — wipe its workspace-keyed
        # rows here so the next run starts with no stale chunks for this
        # workspace either.
        _purge_workspace_repo_index(workspace_path)

    # Resolve the build command AFTER the --new-build reset (if any) has
    # cleared out leftover files from the previous run. Doing this before
    # the reset would let `_detect_subdir_build_command` sniff a stale
    # `backend/requirements.txt`, seed `cd backend && ...` into the build
    # command, then have the reset delete `backend/` — leaving every
    # subsequent build stuck on `sh: cd: can't cd to backend`. Greenfield
    # runs (`teane build` / --new-build) get a deterministic baselined
    # command; an LLM-scaffolded Makefile cannot override it. See
    # _detect_default_build_command for the contract.
    build_command = resolve_build_command(
        config,
        workspace_path,
        is_greenfield=bool(getattr(args, "new_build", False)),
    )

    # Initialize GitGuardian for branch lifecycle management. When
    # --git false, _make_git_guardian returns a no-op stub so the
    # downstream rollback/pop_stash call sites don't need to gate
    # individually.
    git_guardian = _make_git_guardian(workspace_path)
    git_guardian.stash_if_dirty()
    git_guardian.create_patch_branch(session_id)

    # --- Requirement Refinement Layer ---
    # product_spec_dir is the SOLE source for the product spec on greenfield
    # runs. In change-request mode this layer is skipped — the change
    # requests drive the run instead, and ingest_change_requests_node
    # injects them as the LLM's task description in-graph.
    spec_override: Optional[str] = None
    manifest_path: Optional[str] = None
    reuse_docs = getattr(args, "reuse_docs", False)
    if not change_request_mode:
        if reuse_docs:
            # Operator opted to reuse the preserved docs/ (see
            # `_resolve_reuse_docs` and `preserve_docs=True` on the
            # reset above). Read the on-disk specs into spec_override
            # instead of paying the LLM cost of regeneration; leave
            # manifest_path=None so the synthesis block below is
            # naturally skipped.
            spec_req_path = os.path.join(
                workspace_path, "docs", "SPEC_REQUIREMENTS.md",
            )
            spec_arch_path = os.path.join(
                workspace_path, "docs", "SPEC_ARCHITECTURE.md",
            )
            _raw_spec = _read_spec_file(spec_req_path) or ""
            spec_override = _slim_spec_for_prompt(_raw_spec)
            logger.info(
                "[requirements] Reusing %s (%d chars raw → %d slimmed, "
                "%d%% smaller after planner-only sections stripped) — "
                "skipping spec synthesis + reviewer cycles.",
                spec_req_path, len(_raw_spec), len(spec_override),
                (100 * (len(_raw_spec) - len(spec_override))
                 // max(len(_raw_spec), 1)),
            )
            arch_content = (
                _read_spec_file(spec_arch_path)
                if os.path.exists(spec_arch_path) else ""
            )
            if arch_content:
                spec_override = (
                    f"{spec_override}\n\n"
                    f"# Architecture Specification\n"
                    f"_(reused from prior run)_\n\n"
                    f"{arch_content}"
                )
                logger.info(
                    "[architecture] Reusing %s (%d chars) — skipping "
                    "architecture synthesis + reviewer cycles.",
                    spec_arch_path, len(arch_content),
                )
            else:
                logger.info(
                    "[architecture] %s not present — nothing to reuse "
                    "for architecture.", spec_arch_path,
                )
        else:
            import tempfile as _tempfile
            fd, manifest_path = _tempfile.mkstemp(
                prefix=f"harness_spec_{session_id[:8]}_", suffix=".txt",
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(preloaded_consolidated_spec or "")
            logger.info(
                "[requirements] Product spec sourced from %s "
                "(consolidated → %s).", resolved_spec_dir, manifest_path,
            )
    else:
        logger.info(
            "[change_requests] Skipping greenfield spec refinement; "
            "in-graph ingest will compose the LLM task from the folder."
        )

    if manifest_path:
        logger.info("[requirements] Synthesizing specification from %s", manifest_path)
        try:
            # Specs always land under the workspace's `docs/` directory.
            # The standalone --output-dir flag was dropped; the only
            # writable root is the workspace itself.
            output_dir = os.path.join(workspace_path, "docs")
            spec_path = await synthesize_requirements(
                manifest_path=manifest_path,
                output_dir=output_dir,
                gateway=gateway,
                # `_resolve_agile_args` already collapsed the --agile
                # tri-state down to a single bool stored on
                # `decomposition_enabled` (CLI flag > workspace detect >
                # config["agile"] > False). Forward it so the
                # `requirements_doc.md` skill switches to Path A (SAFe +
                # Gherkin) when agile is active and stays on Path B
                # (ISO 29148) otherwise.
                agile=bool(getattr(args, "decomposition_enabled", False)),
                workspace_path=workspace_path,
            )
            # Pre-flight spec review: fire whenever doc_reviewer_primary is
            # configured, regardless of whether --spec-discovery was passed.
            # Previously the reviewer only ran inside the discovery flow
            # (write_spec_node → spec_review_node), so any run started from
            # a manifest skipped it silently.
            from harness.graph import get_gateway_config as _get_gateway_config
            gw_cfg = _get_gateway_config()
            doc_reviewer_primary = (
                getattr(gw_cfg, "doc_reviewer_primary", "") or ""
                if gw_cfg is not None else ""
            )
            # Honour node_throttle.max_doc_review_cycles. The graph-path
            # spec_review_node already loops via its own counter — the
            # pre-flight path was firing review_and_revise_spec exactly once
            # regardless of the config, which silently capped operators who'd
            # set the cycles to >1. Match the graph node's behaviour here.
            max_review_cycles = (
                int(getattr(gw_cfg, "max_doc_review_cycles", 1) or 0)
                if gw_cfg is not None else 1
            )
            if doc_reviewer_primary and max_review_cycles > 0:
                logger.info(
                    "[requirements] doc_reviewer_primary=%s configured — "
                    "running pre-flight spec review (up to %d cycle(s)).",
                    doc_reviewer_primary, max_review_cycles,
                )
                from harness.graph import review_and_revise_spec
                for cycle in range(1, max_review_cycles + 1):
                    # Budget gate matches spec_review_node's check at
                    # graph.py:3754 — stop revisiting the reviewer when the
                    # remaining session budget is below the reviewer's
                    # minimum useful cost.
                    if budget_usd < 0.10:
                        logger.info(
                            "[requirements] Budget too low ($%.4f) — "
                            "skipping remaining review cycles.", budget_usd,
                        )
                        break
                    logger.info(
                        "[requirements] Spec review cycle %d/%d.",
                        cycle, max_review_cycles,
                    )
                    review_result = await review_and_revise_spec(
                        spec_path,
                        "REQUIREMENTS",
                        gateway=gateway,
                        budget_remaining_usd=budget_usd,
                        user_goal=args.prompt or "",
                        llm_dispatch_config=config.get("llm_dispatch", {}),
                    )
                    if review_result["ok"] and review_result.get("review_path"):
                        logger.info(
                            "[requirements] Cycle %d/%d: review written to %s; "
                            "spec revised in place.",
                            cycle, max_review_cycles, review_result["review_path"],
                        )
                        budget_usd = review_result["new_budget_usd"]
                    else:
                        # Reviewer failed (bad JSON, dispatch error, etc.).
                        # Helper already logged the reason; abandon the
                        # remaining cycles rather than spinning on a
                        # broken reviewer.
                        logger.info(
                            "[requirements] Cycle %d/%d: reviewer did not "
                            "complete cleanly — aborting remaining cycles.",
                            cycle, max_review_cycles,
                        )
                        break
            elif not doc_reviewer_primary:
                logger.info(
                    "[requirements] doc_reviewer_primary not configured — skipping pre-flight spec review."
                )
            else:
                logger.info(
                    "[requirements] max_doc_review_cycles=0 — skipping pre-flight spec review."
                )
            logger.info("[requirements] Specification synthesized. Entering review loop.")
            _raw_reviewed_spec = await interactive_review_loop(spec_path, gateway)
            spec_override = _slim_spec_for_prompt(_raw_reviewed_spec)
            logger.info(
                "[requirements] Specification locked. %d characters "
                "approved, %d after planner-only strip (%d%% smaller).",
                len(_raw_reviewed_spec), len(spec_override),
                (100 * (len(_raw_reviewed_spec) - len(spec_override))
                 // max(len(_raw_reviewed_spec), 1)),
            )

            # Architecture synthesis runs whenever the architecture stage is not
            # explicitly disabled. The previous flow jumped straight from
            # locked requirements into code generation, leaving the patching
            # LLM with no explicit module-layout / data-model / test-strategy
            # guidance — which is what produced the allowlist-rejected
            # task_dispatcher/ patches in the TaskDispatcher run. The
            # synthesized SPEC_ARCHITECTURE.md is appended to the system
            # prompt so the patching LLM sees both documents.
            architecture_cfg = config.get("architecture", {}) or {}
            if architecture_cfg.get("enabled", True):
                try:
                    arch_path = await synthesize_architecture(
                        requirements_path=spec_path,
                        output_dir=output_dir,
                        gateway=gateway,
                    )
                    # Same adversarial doc-reviewer pass we run on
                    # requirements — the architecture spec drives every
                    # downstream patch, so skipping the critique was the
                    # difference between a layout the patching LLM follows
                    # and one it works around.
                    if doc_reviewer_primary and max_review_cycles > 0:
                        logger.info(
                            "[architecture] doc_reviewer_primary=%s configured — "
                            "running architecture spec review (up to %d cycle(s)).",
                            doc_reviewer_primary, max_review_cycles,
                        )
                        for cycle in range(1, max_review_cycles + 1):
                            if budget_usd < 0.10:
                                logger.info(
                                    "[architecture] Budget too low ($%.4f) — "
                                    "skipping remaining review cycles.", budget_usd,
                                )
                                break
                            logger.info(
                                "[architecture] Spec review cycle %d/%d.",
                                cycle, max_review_cycles,
                            )
                            arch_review_result = await review_and_revise_spec(
                                arch_path,
                                "ARCHITECTURE",
                                gateway=gateway,
                                budget_remaining_usd=budget_usd,
                                user_goal=args.prompt or "",
                                llm_dispatch_config=config.get("llm_dispatch", {}),
                            )
                            if arch_review_result["ok"] and arch_review_result.get("review_path"):
                                logger.info(
                                    "[architecture] Cycle %d/%d: review written to %s; "
                                    "spec revised in place.",
                                    cycle, max_review_cycles, arch_review_result["review_path"],
                                )
                                budget_usd = arch_review_result["new_budget_usd"]
                            else:
                                logger.info(
                                    "[architecture] Cycle %d/%d: reviewer did not "
                                    "complete cleanly — aborting remaining cycles.",
                                    cycle, max_review_cycles,
                                )
                                break
                    elif not doc_reviewer_primary:
                        logger.info(
                            "[architecture] doc_reviewer_primary not configured — skipping architecture spec review."
                        )
                    else:
                        logger.info(
                            "[architecture] max_doc_review_cycles=0 — skipping architecture spec review."
                        )
                    arch_content = _read_spec_file(arch_path)
                    if arch_content:
                        spec_override = (
                            f"{spec_override}\n\n"
                            f"# Architecture Specification\n"
                            f"_(synthesized from approved requirements)_\n\n"
                            f"{arch_content}"
                        )
                        logger.info(
                            "[architecture] SPEC_ARCHITECTURE.md (%d chars) appended to system prompt.",
                            len(arch_content),
                        )
                except Exception as exc:
                    logger.warning(
                        "[architecture] Architecture synthesis failed: %s. "
                        "Continuing without SPEC_ARCHITECTURE.md.",
                        exc,
                    )
            else:
                logger.info(
                    "[architecture] Disabled in config (architecture.enabled=false). "
                    "Skipping SPEC_ARCHITECTURE.md synthesis."
                )
        except Exception as exc:
            logger.error("[requirements] Requirement refinement failed: %s", exc)
            return 1
        finally:
            # Delete the temp manifest the moment the refinement block is
            # done with it — the consolidated spec lands under docs/ in the
            # workspace, so the /tmp copy serves no further purpose. Without
            # this cleanup every greenfield run leaked one /tmp/harness_spec_*
            # carrying the full product spec (audit §2.7 / §7.1).
            if manifest_path:
                try:
                    os.unlink(manifest_path)
                except OSError as cleanup_exc:
                    logger.debug(
                        "[requirements] manifest temp cleanup failed (%s): %s",
                        manifest_path, cleanup_exc,
                    )
    # If we got here, manifest_path is always set — _load_consolidated_product_spec
    # either succeeded or already returned 1 above. The old "fall through with
    # no spec" branch is gone with the product_spec/ folder mandate.

    thread_id = args.thread_id if args.thread_id else session_id

    logger.info("=" * 60)
    logger.info("Teane — Starting Graph Execution")
    logger.info("  Workspace:  %s", workspace_path)
    logger.info("  Build Cmd:  %s", build_command)
    logger.info("  Session ID: %s", session_id)
    logger.info("  Thread ID:  %s", thread_id)
    logger.info("  Budget:     $%.2f", budget_usd)
    logger.info("  Network:    %s", "enabled" if allow_network else "blocked")
    logger.info("  Prompt:     %s", args.prompt[:100] + ("..." if len(args.prompt) > 100 else ""))
    logger.info(
        "  Discovery:  %s",
        "enabled (--spec-discovery true)"
        if getattr(args, "spec_discovery", False)
        else "skipped (pass --spec-discovery true to enable)",
    )
    logger.info(
        "  Deployment: %s",
        "enabled (--deploy-dev true)"
        if getattr(args, "deploy_dev", False)
        else "skipped (pass --deploy-dev true to deploy locally)",
    )
    logger.info(
        "  CD discovery: %s",
        "enabled (--cd-discovery true)"
        if getattr(args, "cd_discovery", False)
        else "disabled (deployment skips the LLM interview)",
    )
    _install_doc_flag = getattr(args, "install_doc", None)
    _install_doc_effective = (
        _install_doc_flag if _install_doc_flag is not None
        else bool(getattr(args, "new_build", False))
    )
    logger.info(
        "  Install doc: %s",
        "enabled (--install-doc true)" if _install_doc_effective
        else "skipped (pass --install-doc true or --new-build true)",
    )
    if spec_override:
        logger.info("  Spec:       SPEC_REQUIREMENTS.md (+SPEC_ARCHITECTURE.md) (%d chars)", len(spec_override))
    logger.info("=" * 60)

    try:
        final_state = await run_graph(
            workspace_path=workspace_path,
            prompt=args.prompt,
            build_command=build_command,
            spec_override=spec_override,
            allow_network=allow_network,
            budget_usd=budget_usd,
            session_id=session_id,
            checkpointer=checkpointer,
            thread_id=thread_id,
            # Discovery (both requirements + architecture interviews)
            # runs only when --spec-discovery true is explicitly passed.
            skip_discovery=not getattr(args, "spec_discovery", False),
            # When False (default), the security-scan router short-circuits to
            # END instead of routing into deployment_discovery_node. See
            # route_after_security_scan in harness/graph.py.
            dev_deployment=getattr(args, "deploy_dev", False),
            # Container-deployment discovery toggle. When False AND
            # --deploy-dev is True, the deployment step skips the LLM
            # interview and synthesises the blueprint from workspace
            # telemetry alone (using deployment_defaults from config.json
            # where present). See route_after_security_scan in
            # harness/graph.py.
            cd_discovery=getattr(args, "cd_discovery", False),
            # End-of-run INSTALLATION.md synthesis. When --install-doc is
            # unset on the command line (default), follow --new-build —
            # greenfield runs get an install doc; incremental change
            # requests skip it. Explicit --install-doc true|false wins.
            install_doc=bool(
                getattr(args, "install_doc", None)
                if getattr(args, "install_doc", None) is not None
                else getattr(args, "new_build", False)
            ),
            # Agile mode (opt-in via --agile). When false, every
            # downstream node check on decomposition_enabled / current_story_id
            # short-circuits, so the byte-for-byte monolithic flow runs.
            # `_resolve_agile_args` collapses --agile → decomposition_enabled
            # plus the three agile_defaults knobs seeded below.
            decomposition_enabled=getattr(args, "decomposition_enabled", False),
            commit_on_story=getattr(args, "commit_on_story", False),
            story_batch_size=getattr(args, "story_batch_size", 5),
            story_repair_cap=getattr(args, "story_repair_cap", 3),
            lintgate_config=config.get("lintgate", {}),
            diagnostics_config=config.get("diagnostics", {}),
            post_mortem_config=config.get("post_mortem", {}),
            deployment_config=config.get("deployment", {}),
            deployment_defaults=load_deployment_defaults(config),
            sandbox_config=config.get("sandbox", {}),
            test_generation_config=config.get("test_generation", {}),
            speculative_config=config.get("speculative", {}),
            compiler_config=config.get("compiler", {}),
            change_request_mode=change_request_mode,
            change_requests_dir_abs=cr_dir_abs if change_request_mode else "",
            archive_target_dir=archive_target_dir,
            change_requests_config=config.get("change_requests", {}),
            repo_memory_config=config.get("memory", {}),
            repo_index_config=config.get("repo_index", {}),
            llm_dispatch_config=config.get("llm_dispatch", {}),
            # Which top-level command spawned this run. cmd_build /
            # cmd_patch / cmd_deploy each pin args.flow before delegating
            # here; legacy callers (tests, direct cmd_run invocations)
            # default to "build".
            flow=getattr(args, "flow", "build"),
            # In flow=patch, controls whether the reverse-spec node fires
            # before the planner. Resolved by cmd_patch from the
            # --generate-specs tri-state. False on every other flow.
            generate_specs=getattr(args, "generate_specs", False),
            # Full resolved config — threaded so the system-prompt builder
            # can pin the right backend block of the locked stack directive
            # based on core_languages.backend_language.
            full_config=config,
        )
    except Exception:
        logger.exception("Graph execution failed with unhandled exception.")
        git_guardian.rollback()
        git_guardian.pop_stash()
        # Commit any in-flight checkpoint write before close so that
        # `teane resume --session-id <id>` can recover the last state
        # the user saw on screen. Without the commit, aiosqlite drops
        # everything that wasn't already flushed (audit §5.6).
        try:
            await checkpointer.conn.commit()
        except Exception:  # noqa: BLE001 — best-effort; close still runs
            logger.exception("[cli] checkpoint commit on error path failed.")
        await checkpointer.conn.close()
        return 1

    exit_code = final_state.get("exit_code", -1)
    modified_files = final_state.get("modified_files", [])
    # Canonical cost comes from the Gateway's session tracker — every
    # dispatch lands there automatically, so the figure includes call
    # sites (cli synthesis, discovery nodes, speculative losers,
    # continuation loops, chat tool-loops, …) that historically skipped
    # the state-side ``aggregate_tokens`` step. Falls back to the state
    # mirror for the (defensive) case where the gateway tracker is
    # somehow empty but the state was populated by a legacy caller.
    gateway_total = gateway.session_tracker.get("total_cost_usd", 0.0)
    state_total = final_state.get("token_tracker", {}).get("total_cost_usd", 0.0)
    total_cost = gateway_total or state_total

    # Distinguish HITL Save & Quit (intentional pause; operator will
    # `teane resume`) from a hard failure. Previously both took the same
    # exit_code != 0 branch and the rollback wiped the LLM's in-flight
    # work — observed in session d880f762 where pressing [s] deleted 21
    # generated app/ + tests/ + requirements.txt files even though the
    # operator's intent was the opposite: keep the work, come back later.
    node_state = final_state.get("node_state", {}) or {}
    hitl_suspend = bool(node_state.get("hitl_suspend"))
    hitl_abandon = bool(node_state.get("hitl_abandon"))

    if hitl_suspend:
        # Suspend = "I'll come back to this." Leave the workspace EXACTLY
        # as the LLM left it on the agent/patch-<session> branch so a
        # subsequent `teane resume --session-id <id>` picks up against
        # the same files. The pre-session stash stays parked (the operator
        # can list it with `git stash list` and pop it manually if they
        # need the prior work); popping it here could merge-conflict with
        # the LLM's edits and surprise the operator.
        agent_branch = getattr(git_guardian, "_patch_branch", None) or "agent/patch-<unknown>"
        logger.info(
            "[cli] HITL suspend: leaving %d LLM-modified file(s) on branch "
            "'%s'. Resume with `teane resume --session-id %s` to continue "
            "from the same workspace state.",
            len(modified_files), agent_branch, session_id,
        )
    elif hitl_abandon:
        # Abandon = user explicitly confirmed "throw it away." The HITL
        # handler already ran _attempt_git_rollback(workspace_path); the
        # git_guardian-level rollback below would be redundant on a clean
        # tree but is the safe-to-rerun belt-and-suspenders.
        git_guardian.rollback(modified_files)
        git_guardian.pop_stash()
    elif exit_code == 0:
        # Git lifecycle: commit on success
        git_guardian.commit_all_changes(session_id, modified_files, exit_code)
        git_guardian.restore_original_branch()
        git_guardian.pop_stash()
        # When --deploy-dev was not passed, the harness ended right after
        # the security scan; surface the next step explicitly so operators
        # upgrading from the old auto-deploy default see why no Dockerfiles
        # / docker-compose run happened. Flutter projects always end here
        # regardless of the flag, so the hint is only useful when deployment
        # would otherwise have run.
        if not getattr(args, "deploy_dev", False):
            logger.info(
                "[cli] Code generated at %s. Deployment phase skipped. "
                "Re-run with --deploy-dev true to bring the app up "
                "locally via docker compose.",
                workspace_path,
            )
    else:
        # Real build failure with no operator intervention — rollback as before.
        git_guardian.rollback(modified_files)
        git_guardian.pop_stash()

    logger.info("=" * 60)
    logger.info("Graph Execution Complete")
    logger.info("  Exit Code:      %d", exit_code)
    logger.info("  Modified Files: %d", len(modified_files))
    for modified in modified_files:
        logger.info("    - %s", modified)
    logger.info("  Token Cost:     $%.6f", total_cost)
    logger.info("  Session ID:     %s", session_id)
    logger.info("=" * 60)

    # Archive consumed change-request files (.txt / .md / .pdf) into
    # <change_requests_dir>/applied/<session-id>/ along with a manifest.json.
    # Suspend (HITL Save & Quit) is exempted — the session will resume and
    # the files must still be readable from the original folder. Abandon
    # archives them with a "cancelled" status so the operator can tell
    # consumed-but-rolled-back runs apart from successful applies.
    if change_request_mode and not hitl_suspend:
        if exit_code == 0:
            cr_status = "success"
        elif hitl_abandon:
            cr_status = "cancelled"
        else:
            cr_status = "failed-build"
        _archive_consumed_change_requests(
            final_state.get("change_request_files", []),
            archive_target_dir,
            session_id=session_id,
            status=cr_status,
            modified_files=modified_files,
        )

    await checkpointer.conn.close()
    await _drain_mcp_pools()

    # Persist a one-line entry to the per-repo memory file so the next
    # `teane run` against this workspace sees the prior outcome in
    # the planner context. Wrapped: failures must not change the
    # exit code.
    pm_note = await _post_mortem_finalize(
        final_state, exit_code, config, workspace_path,
    )
    _append_repo_memory_safely(
        workspace_path=workspace_path,
        session_id=session_id,
        prompt_summary=getattr(args, "prompt", "") or "",
        modified_files=modified_files,
        exit_code=exit_code,
        config=config,
        extra_notes=pm_note or None,
    )

    # Record completion marker so `teane test`'s prereq gate can see this
    # clean run. Tracked flows: build, patch, deploy. No-op on failure.
    # Wrapped against exceptions inside flow_state itself.
    from harness import flow_state
    flow_state.record_flow_completion(
        workspace_path=workspace_path,
        flow=getattr(args, "flow", "build"),
        session_id=session_id,
        exit_code=exit_code,
        summary={
            "modified_files": len(modified_files),
            "loop_counters": final_state.get("loop_counter", {}),
        },
    )

    return _resolve_cli_exit_code(
        graph_exit_code=exit_code,
        final_state=final_state,
    )


# Deterministic CLI exit codes — reserved for automation / CI
# consumers. Prior to 2026-07-04 the CLI returned 0 or 1
# indiscriminately, which meant a ``teane build && teane deploy``
# script would happily deploy a run that finished with 15/26
# traceability coverage. These constants let a caller distinguish
# "everything clean" from every failure mode that surfaced during v10/
# v11/v12 investigation.
EXIT_CLEAN = 0
"""All requested work completed, all guards clean."""
EXIT_PARTIAL_SUCCESS = 1
"""Some batches sealed, but one or more guards fired (traceability
gap, HITL suspend/abandon, security warnings). Deploy-blocking."""
EXIT_CONFIG_ERROR = 2
"""Config or spec drift blocked the run before it could start.
Same as argparse conventions and the current preflight validators."""
EXIT_BUDGET_EXHAUSTED = 3
"""``token_budget.hard_cap_usd`` reached before all stories shipped.
Set on the ``budget_terminated`` node_state flag by the HITL
budget-exhausted auto-terminate path."""
EXIT_INFRASTRUCTURE_FAILURE = 4
"""Sandbox / gateway / provider outage prevented the run from
progressing. Distinguish from PARTIAL_SUCCESS so CI can retry
these vs. surface a spec/code problem to the operator."""


def _resolve_cli_exit_code(
    *, graph_exit_code: int, final_state: dict[str, Any],
) -> int:
    """Map the graph's final state to one of the reserved CLI exit
    codes. Runs at the very end of ``cmd_run``; consumes only
    read-only state so it can be called from resume / patch flows too.

    Precedence (most specific first):
        * ``budget_terminated`` on node_state → EXIT_BUDGET_EXHAUSTED
        * ``env_misconfig`` / ``build_command_cd_missing`` /
          ``llm_silent`` / connectivity flags → EXIT_INFRASTRUCTURE_FAILURE
        * ``traceability_blocked`` on node_state → EXIT_PARTIAL_SUCCESS
        * ``hitl_abandon`` or ``hitl_suspend`` on node_state →
          EXIT_PARTIAL_SUCCESS
        * graph exit_code != 0 → EXIT_PARTIAL_SUCCESS
        * else → EXIT_CLEAN

    Callers use these codes to distinguish "green build ready to
    deploy" from "shipped-with-warnings" from
    "infrastructure-please-retry".
    """
    node_state = (final_state.get("node_state") or {}) if isinstance(
        final_state, dict,
    ) else {}
    if node_state.get("budget_terminated"):
        return EXIT_BUDGET_EXHAUSTED
    # Infrastructure signals — the harness can't proceed until
    # someone outside fixes an environmental condition.
    # ``llm_behavior`` is grouped here too: an LLM-behavior HITL that
    # was never resolved (e.g. session ended with the model still
    # refusing to emit tests) is not "clean" and shares the same
    # exit-code semantics as an env condition — "someone (or something)
    # outside the harness needs to intervene before another run works."
    infra_flags = (
        "env_misconfig",
        "llm_behavior",
        "build_command_blocked",
        "build_command_cd_missing",
        "llm_silent",
    )
    if any(node_state.get(f) for f in infra_flags):
        return EXIT_INFRASTRUCTURE_FAILURE
    if node_state.get("traceability_blocked"):
        return EXIT_PARTIAL_SUCCESS
    if node_state.get("hitl_abandon") or node_state.get("hitl_suspend"):
        return EXIT_PARTIAL_SUCCESS
    if int(graph_exit_code or 0) != 0:
        return EXIT_PARTIAL_SUCCESS
    return EXIT_CLEAN


def _resolve_agile_args(args: argparse.Namespace, *, config: dict[str, Any], workspace_path: str, flow: str) -> None:
    """Resolve the `--agile` tri-state on build/patch and seed the legacy
    `decomposition_enabled` / `story_*` args that ``cmd_run`` reads.

    Resolution precedence:
      build: CLI flag > config["agile"] > False.
      patch: CLI flag > workspace_is_agile_managed(...) > config["agile"] > False.

    The agile_defaults block in config.json supplies the per-knob tuning
    (batch_size, commit_on_story, repair_cap) — moved out of the CLI when
    the three --story-* flags were removed.
    """
    cli_val = getattr(args, "agile", None)
    if cli_val is None:
        if flow == "patch":
            try:
                from harness.story_state import workspace_is_agile_managed
                detected = workspace_is_agile_managed(workspace_path)
            except Exception:  # noqa: BLE001
                detected = False
            if detected:
                cli_val = True
                logger.info("[cli] agile workspace detected (.teane/state.db non-empty).")
        if cli_val is None:
            cli_val = bool(config.get("agile", False))

    args.decomposition_enabled = bool(cli_val)
    defaults = config.get("agile_defaults") or {}
    args.story_batch_size = int(defaults.get("batch_size", 5))
    args.commit_on_story = bool(defaults.get("commit_on_story", False))
    args.story_repair_cap = int(defaults.get("repair_cap", 3))


def _resolve_generate_specs(args: argparse.Namespace, *, workspace_path: str) -> Optional[int]:
    """Resolve --generate-specs tri-state on patch. Sets args.generate_specs.

    Returns 2 (CLI error exit code) when the operator passed
    --generate-specs=false AND a spec file is missing, so the caller can
    fail-fast before any LLM call.
    """
    val = getattr(args, "generate_specs", None)
    spec_req = os.path.join(workspace_path, "docs", "SPEC_REQUIREMENTS.md")
    spec_arch = os.path.join(workspace_path, "docs", "SPEC_ARCHITECTURE.md")
    missing = [p for p in (spec_req, spec_arch) if not os.path.isfile(p)]

    if val is None:
        args.generate_specs = bool(missing)
        if args.generate_specs:
            logger.info("[cli] spec docs missing — auto-enabling reverse-spec generation.")
        return None
    if val is False and missing:
        print(
            "error: SPEC docs are missing but --generate-specs=false was "
            f"passed. Missing: {', '.join(os.path.basename(m) for m in missing)}. "
            "Pass --generate-specs true to reverse-engineer them from the "
            "codebase, or place the .md files under docs/ first.",
            file=sys.stderr,
        )
        return 2
    args.generate_specs = bool(val)
    if val is True:
        logger.info("[cli] --generate-specs=true: regenerating spec drafts.")
    return None


def _resolve_reuse_docs(
    workspace_path: str,
    reuse_specs_override: Optional[bool] = None,
) -> bool:
    """Decide whether `teane build` should preserve docs/ and skip spec regen.

    Semantics:
      - If ``reuse_specs_override`` is set (from ``--reuse-specs true|false``)
        → honor it verbatim; the CLI flag wins over both prompts and the
        non-TTY default. When the override is `true` but no spec marker
        exists, drop to False (nothing to reuse) with a clear log line
        so the operator isn't misled by the flag they passed.
      - Else if ``docs/SPEC_REQUIREMENTS.md`` is absent → False (nothing
        to reuse; the normal reset + synthesis flow runs).
      - Else if stdin is not a TTY → True (non-interactive default:
        reuse so eval sweeps stop paying the spec-generation LLM cost
        on every run).
      - Else prompt the operator [Y/n]; default True on empty input.

    The prompt is independent of ``-y/--yes`` — ``-y`` only suppresses
    the destructive-reset confirmation, not this reuse choice.
    """
    spec_marker = os.path.join(workspace_path, "docs", "SPEC_REQUIREMENTS.md")
    if reuse_specs_override is not None:
        if reuse_specs_override and not os.path.exists(spec_marker):
            logger.info(
                "[build] --reuse-specs true, but %s does not exist — "
                "specs will be generated from scratch.", spec_marker,
            )
            return False
        logger.info(
            "[build] --reuse-specs %s (CLI override).",
            str(reuse_specs_override).lower(),
        )
        return bool(reuse_specs_override)
    if not os.path.exists(spec_marker):
        return False
    if not sys.stdin.isatty():
        logger.info(
            "[build] Non-interactive run: %s exists — reusing docs/ "
            "(skipping spec regeneration). Pass `--reuse-specs false` to "
            "force regeneration.", spec_marker,
        )
        return True
    from harness.hitl import get_channel as _get_channel
    reuse = _get_channel().confirm(
        f"Existing specs found in {os.path.dirname(spec_marker)}. "
        "Reuse them (skip spec regeneration)?",
        default=True,
    )
    if reuse:
        logger.info(
            "[build] Operator chose to reuse docs/ — spec regeneration "
            "will be skipped."
        )
    else:
        logger.info(
            "[build] Operator chose to regenerate docs/ — the reset "
            "will wipe the folder and specs will be re-synthesised."
        )
    return reuse


async def cmd_build(args: argparse.Namespace) -> int:
    """`teane build` — destructive greenfield.

    Pins args.new_build=True (workspace reset always runs), args.flow="build",
    args.deploy_dev=False (deploy is its own command now), then delegates to
    cmd_run. The destructive-confirmation prompt + -y modifier come along
    from cmd_run's existing logic.
    """
    args.flow = "build"
    args.new_build = True
    args.deploy_dev = False
    args.install_doc = True  # build always writes initial INSTALLATION.md
    args.generate_specs = False
    # Resolve --agile from CLI/config; load config first so we can read it.
    workspace_path = os.path.abspath(getattr(args, "workspace", None) or os.getcwd())

    # Wipe-collision guard for --log target and shell-redirected stdout.
    # `teane build` wipes the workspace root at startup (preserving only
    # product_spec/, .git/, and optionally docs/). Anything else — a log
    # file the operator staged under the workspace, or a shell-redirected
    # stdout target inside the workspace — gets deleted before it can
    # capture the run. Finsearch session 156032347 (2026-07-13) hit this
    # exact shape: the operator redirected `>` to a path inside the
    # workspace, teane wiped it, and the run was invisible until the
    # operator moved the log outside the workspace and restarted. Refuse
    # here so the operator relocates BEFORE the destructive step.
    _wipe_target_guard_error = _refuse_log_inside_workspace(
        workspace_path=workspace_path,
        log_file=getattr(args, "log_file", None),
    )
    if _wipe_target_guard_error is not None:
        logger.error("[build] %s", _wipe_target_guard_error)
        return 1
    try:
        config = _strip_comments(load_raw_config())
    except Exception:  # noqa: BLE001 — cmd_run will re-load + error properly
        config = {}
    _resolve_agile_args(args, config=config, workspace_path=workspace_path, flow="build")
    # Ask (or infer, when non-interactive) whether to preserve docs/
    # and skip the pre-graph spec synthesis. This gates BOTH the reset's
    # preserved-set (via preserve_docs) AND the manifest/synthesis
    # block in cmd_run — the two are intentionally coupled since
    # preserving the folder without skipping regen would just overwrite
    # what we preserved, and skipping regen without preserving the
    # folder would leave nothing to read.
    args.reuse_docs = _resolve_reuse_docs(
        workspace_path,
        reuse_specs_override=getattr(args, "reuse_specs", None),
    )
    return await cmd_run(args)


async def cmd_patch(args: argparse.Namespace) -> int:
    """`teane patch` — brownfield reconcile.

    Resolves --generate-specs (auto/true/false) and --agile (auto/true/false),
    pins args.flow="patch" and args.new_build=False, then delegates to
    cmd_run.
    """
    args.flow = "patch"
    args.new_build = False
    args.deploy_dev = False
    args.assume_yes = False  # patch never resets, -y is irrelevant
    workspace_path = os.path.abspath(getattr(args, "workspace", None) or os.getcwd())
    try:
        config = _strip_comments(load_raw_config())
    except Exception:  # noqa: BLE001
        config = {}
    err = _resolve_generate_specs(args, workspace_path=workspace_path)
    if err is not None:
        return err
    _resolve_agile_args(args, config=config, workspace_path=workspace_path, flow="patch")
    # v5 Phase 6c: agile patches MUST run installation_doc_node so the
    # end-of-session traceability audit fires (the audit block lives
    # inside that node, see harness/graph.py:12111-12153). Non-agile
    # patch keeps install_doc=False so legacy operators see no change.
    # The user can still opt out explicitly with --install-doc false.
    if (
        getattr(args, "decomposition_enabled", False)
        and getattr(args, "install_doc", None) is None
    ):
        args.install_doc = True
    return await cmd_run(args)


async def cmd_deploy(args: argparse.Namespace) -> int:
    """`teane deploy` — artifacts + dev container + sign-off.

    Pins args.flow="deploy" and args.deploy_dev=True so cmd_run's existing
    deployment edges fire. install_doc is True so INSTALLATION.md is
    updated with the deployed port surface at the end.
    """
    args.flow = "deploy"
    args.new_build = False
    args.deploy_dev = True
    args.install_doc = True
    args.assume_yes = False
    args.spec_discovery = False
    args.generate_specs = False
    args.decomposition_enabled = False
    args.commit_on_story = False
    args.story_batch_size = 5
    args.story_repair_cap = 3
    # Deploy-specific HITL gates: only --hitl-deployment is exposed; the
    # build/patch gates are pinned to false here so cmd_run's resolver
    # doesn't surprise an operator who never sees them on the CLI.
    for dest in ("hitl_requirement", "hitl_architecture", "hitl_repair", "hitl_layout_divergence"):
        if not hasattr(args, dest):
            setattr(args, dest, False)
    return await cmd_run(args)


async def cmd_test(args: argparse.Namespace) -> int:
    """`teane test` — e2e verification pack against the dev compose stack.

    Pins args.flow="test" and disables every discovery / planning / build
    side-effect — this target ONLY runs the e2e pipeline against an
    already-deployed app and emits CR-DEFECT-* on failures. cmd_run's
    existing edge router (route_after_start) sees flow="test" and routes
    straight to harness.test_target.test_node, which gates on
    `<workspace>/.teane/last_{build,patch,deploy}.json` markers.
    """
    args.flow = "test"
    args.new_build = False
    args.deploy_dev = False
    args.install_doc = False
    args.assume_yes = False
    args.spec_discovery = False
    args.generate_specs = False
    args.decomposition_enabled = False
    args.commit_on_story = False
    args.story_batch_size = 5
    args.story_repair_cap = 3
    # Test-specific HITL gates: none are exposed; pin to false so cmd_run's
    # resolver doesn't surprise an operator. The test target is a pure
    # verification step — no interactive interview points belong here.
    for dest in (
        "hitl_requirement", "hitl_architecture", "hitl_repair",
        "hitl_layout_divergence", "hitl_deployment",
    ):
        if not hasattr(args, dest):
            setattr(args, dest, False)
    return await cmd_run(args)


async def cmd_resume(args: argparse.Namespace) -> int:
    """
    Execute the `teane resume` subcommand.

    Restores a previously checkpointed session from SQLite and resumes
    graph execution from the exact checkpoint boundary.

    Example:
        teane resume --session-id my-session-abc123
        teane resume --session-id my-session -w /path/to/repo
    """
    # Fail fast on a missing runtime dependency BEFORE the first heavy import
    # below (harness.storage pulls langgraph-checkpoint-sqlite) so the operator
    # gets a clean message + fix instead of a raw ImportError.
    _dep_exit = _preflight_dependency_guard()
    if _dep_exit is not None:
        return _dep_exit

    from harness.storage import HarnessAsyncSqliteSaver, inspect_session

    # Workspace auto-detect: --help promises "auto-detected from checkpoint if
    # omitted". Read the checkpoint's recorded workspace_path before falling
    # back to CWD, so the promise holds. Uses the default db_path — a
    # workspace-scoped persistence.db_path override still requires -w since
    # we can't discover config without the workspace.
    if args.workspace:
        workspace_path = os.path.abspath(args.workspace)
    else:
        default_db_path = "~/.harness/checkpoints.db"
        detected: Optional[str] = None
        try:
            summary = await inspect_session(default_db_path, args.session_id)
            if summary and summary.workspace_path:
                detected = summary.workspace_path
        except Exception as exc:  # noqa: BLE001 — best-effort auto-detect
            logger.debug("[resume] Workspace auto-detect failed: %s", exc)
        if detected:
            workspace_path = os.path.abspath(detected)
            logger.info(
                "[resume] Workspace auto-detected from checkpoint: %s",
                workspace_path,
            )
        else:
            workspace_path = os.getcwd()
    if _refuse_if_workspace_is_harness_root(workspace_path):
        return 1

    # Acquire the workspace lock (audit §5.1). cmd_run already takes the
    # lock; resume previously did not, so two concurrent
    # ``teane resume --session-id X`` invocations against the same
    # workspace could clobber each other's patches.
    force = bool(getattr(args, "force_lock", False))
    lock_handle = _acquire_workspace_lock(workspace_path, force=force)
    if lock_handle is False:
        return 1

    # Record git mode for the resumed session — same contract as cmd_run.
    # See the comment in cmd_run for why this is module-level state.
    _set_git_enabled(bool(getattr(args, "git", True)))

    config = discover_config(workspace_path)
    _preflight_config_env_report(config)

    # Bind the active session_id immediately so any pre-graph dispatches
    # (e.g. checkpoint health-check helpers, future hooks) and the in-graph
    # dispatches that follow all stamp the correct session into
    # ~/.harness/debug/<sid>_<seqno>_<role>_<model>.txt filenames. See the
    # matching call in cmd_run.
    from harness.observability import set_active_session_id
    set_active_session_id(args.session_id)

    # Configure structured logging / per-session log file. Without this,
    # the root logger only carries the module-import StreamHandler, so
    # every INFO/DEBUG/ERROR emitted during resume goes to console only
    # and never lands in ~/.harness/logs/<session>.jsonl — leaving
    # `teane metrics`, the dashboard, and any post-hoc log grep blind
    # to the entire resumed run. Mirror cmd_run's block verbatim.
    from harness.observability import configure_logging
    log_cfg = config.get("logging", {})
    configure_logging(
        session_id=args.session_id,
        log_dir=log_cfg.get("log_dir", "~/.harness/logs"),
        level=log_cfg.get("level", "INFO"),
        langsmith_enabled=bool(log_cfg.get("langsmith", False)),
        json_stderr=bool(log_cfg.get("json_stderr", False)),
        max_bytes=int(log_cfg.get("max_bytes", 10_000_000)),
        backup_count=int(log_cfg.get("backup_count", 5)),
        console_level=log_cfg.get("console_level"),
    )

    persistence_cfg = config.get("persistence", {})
    db_path = persistence_cfg.get("db_path", "~/.harness/checkpoints.db")
    ttl_days = persistence_cfg.get("ttl_days", 30)
    redact_messages = bool(persistence_cfg.get("redact_messages", True))

    checkpointer = await HarnessAsyncSqliteSaver.from_db_path(
        db_path=db_path, ttl_days=ttl_days, redact_messages=redact_messages,
    )

    # Verify that the thread exists
    from typing import cast as _cast
    from langchain_core.runnables import RunnableConfig
    config_for_get = {"configurable": {"thread_id": args.session_id}}
    existing = await checkpointer.aget(_cast(RunnableConfig, config_for_get))
    if existing is None:
        logger.error("No checkpoint found for session '%s'.", args.session_id)
        await checkpointer.conn.close()
        return 1

    build_command = resolve_build_command(config, workspace_path)
    token_budget = config.get("token_budget", {})
    budget_usd = token_budget.get("hard_cap_usd", 2.00)
    # --allow-network is a real bool (default true). Drop the
    # OR-truthy coalesce: an explicit --allow-network false must NOT be
    # silently overridden by a stale `allow_network: true` in config.
    allow_network = bool(getattr(args, "allow_network", True))

    # Initialize the LLM Gateway and inject it for graph nodes
    from harness.gateway import create_gateway_from_config
    from harness.graph import set_gateway, run_graph

    gateway = create_gateway_from_config(config)
    set_gateway(gateway)

    # Seed the gateway's session tracker from the JSONL log so the
    # end-of-run "Token Cost" line on a resumed session shows the
    # cumulative cost across all runs of this session — matching what
    # `teane metrics` and the dashboard already report. Without this
    # the resumed run's gateway tracker starts at $0 and the end-of-run
    # summary would silently drop the prior run's spend, re-introducing
    # the dual-display divergence we're fixing here.
    try:
        log_cfg_resume = config.get("observability", {}) or {}
        resume_log_dir = os.path.expanduser(
            log_cfg_resume.get("log_dir", "~/.harness/logs")
        )
        from harness.metrics import aggregate_session as _aggregate_session
        prior = _aggregate_session(args.session_id, resume_log_dir)
        if prior.total_cost_usd > 0:
            gateway.session_tracker["total_input_tokens"] = prior.tokens_in
            gateway.session_tracker["total_output_tokens"] = prior.tokens_out
            gateway.session_tracker["total_cached_tokens"] = prior.cached_tokens
            gateway.session_tracker["total_cost_usd"] = prior.total_cost_usd
            logger.info(
                "[cli] Seeded gateway session tracker from prior log: "
                "$%.6f across %d calls.",
                prior.total_cost_usd, prior.llm_call_count,
            )
    except Exception as exc:  # noqa: BLE001 — seeding is best-effort
        logger.debug("[cli] Could not seed session tracker on resume: %s", exc)

    # Register built-in skills (pipeline + docgen + opt-in tool skills like
    # web_fetch / web_search). Wrapped: any failure inside the skill
    # registry must NOT block the harness from starting — the registry is
    # additive, not load-bearing for the core graph.
    try:
        from harness.skills import register_builtin_skills
        register_builtin_skills(config=config)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[cli] skill registration skipped: %s", exc)

    # Start the MCP client pool when ``mcp.enabled=true``. Each declared
    # MCP server spawns as a subprocess; their tools register into the
    # SkillRegistry under ``mcp__<server>__<tool>`` names so the graph's
    # tool-block interceptor can dispatch them. An atexit handler tears
    # the subprocesses down on a clean exit; Ctrl-C is handled by the
    # outer asyncio cancel path which also triggers the same shutdown.
    _mcp_pool = await _maybe_start_mcp_pool(config, workspace_path=workspace_path)
    # Brownfield-only LSP navigation pool (no-op for flow="build").
    _lsp_pool = await _maybe_start_lsp_pool(
        config, workspace_path, flow=getattr(args, "flow", "build"),
    )  # noqa: F841 — drained via the shared pool registry

    # Initialize the secret redactor
    from harness.redactor import create_redactor_from_config
    create_redactor_from_config(config)

    # Wire the process-wide CommandValidator so resumed sessions get the same
    # defense-in-depth as fresh cmd_run sessions.
    from harness.security import (
        create_command_validator_from_config,
        set_command_validator,
    )
    set_command_validator(create_command_validator_from_config(config))

    logger.info("[resume] Restoring session '%s' from checkpoint.", args.session_id)

    # Pre-flight: confirm the most recent checkpoint blob actually
    # deserializes. Without this, a corrupted blob is silently restored as
    # an empty dict, and the graph restarts from scratch — likely clobbering
    # the workspace with a fresh first patch. Strict mode raises
    # CheckpointCorruptedError, which we surface as a clean operator message
    # instead of an opaque internal traceback.
    from harness.storage import (
        CheckpointCorruptedError,
        CheckpointSchemaMismatchError,
        _deserialize_checkpoint_blob,
        validate_checkpoint_schema,
    )
    import aiosqlite
    try:
        async with aiosqlite.connect(os.path.expanduser(db_path)) as conn:
            async with conn.execute(
                "SELECT checkpoint, metadata FROM checkpoints "
                "WHERE thread_id = ? ORDER BY checkpoint_id DESC LIMIT 1",
                (args.session_id,),
            ) as cur:
                row = await cur.fetchone()
        if row is None:
            logger.error(
                "[resume] No checkpoint found for session '%s'. "
                "Use `teane status --all` to list available sessions.",
                args.session_id,
            )
            return 1
        try:
            _deserialize_checkpoint_blob(row[0], strict=True)
        except CheckpointCorruptedError as exc:
            logger.error(
                "[resume] Checkpoint for session '%s' is corrupted: %s\n"
                "  Options:\n"
                "    - Start a fresh session with `teane build -w %s -p '<prompt>'` (or `teane patch` for brownfield).\n"
                "    - Restore checkpoints.db from a known-good backup.\n"
                "    - Run `teane purge --session-id %s` to drop only this session.",
                args.session_id, exc, workspace_path, args.session_id,
            )
            return 1
        # P2.4: refuse to resume a checkpoint stamped with an incompatible
        # schema version. Catches the "newer harness wrote this, older
        # harness is trying to restore" footgun before the graph touches
        # the workspace.
        try:
            validate_checkpoint_schema(row[1])
        except CheckpointSchemaMismatchError as exc:
            logger.error(
                "[resume] Checkpoint for session '%s' has an incompatible schema: %s\n"
                "  Options:\n"
                "    - Upgrade or downgrade the harness to match the checkpoint's version.\n"
                "    - Start a fresh session with `teane build -w %s -p '<prompt>'` (or `teane patch` for brownfield).\n"
                "    - Run `teane purge --session-id %s` to drop only this session.",
                args.session_id, exc, workspace_path, args.session_id,
            )
            return 1
    except aiosqlite.Error as exc:
        logger.error(
            "[resume] Could not read checkpoint DB at %s: %s",
            db_path, exc,
        )
        return 1

    # One-screen diagnostic: tell the user exactly what we're about to
    # resume so they don't fly blind. Read-only — re-uses the same
    # inspect_session() helper that powers `teane status`. Failures
    # here must never block resume itself, so wrap in a broad except.
    try:
        from harness.storage import inspect_session as _inspect_session
        summary = await _inspect_session(db_path, args.session_id)
        if summary is not None:
            file_preview = ""
            if summary.modified_files:
                shown = ", ".join(summary.modified_files[:5])
                more = f", +{len(summary.modified_files) - 5} more" if len(summary.modified_files) > 5 else ""
                file_preview = f" ({shown}{more})"
            last_exit_label = (
                "0 (clean)" if summary.exit_code == 0
                else "-1 (not yet built)" if summary.exit_code == -1
                else f"{summary.exit_code} (failed)"
            )
            logger.info("=" * 60)
            logger.info("Resuming session %s", args.session_id)
            logger.info("  Last node:        %s", summary.current_node or "(unknown)")
            logger.info("  Modified files:   %d%s", len(summary.modified_files), file_preview)
            logger.info("  Budget remaining: $%.4f", summary.budget_remaining_usd)
            logger.info("  Last exit code:   %s", last_exit_label)
            logger.info("  Loop counters:    %s", summary.loop_counters or "{}")
            logger.info("=" * 60)
    except Exception as _exc:  # noqa: BLE001 — diagnostic must never block resume
        logger.debug("[resume] Could not build pre-resume summary: %s", _exc)

    # Re-attach the GitGuardian to the same agent/patch-<id> branch that
    # the original cmd_run created. Without this, a resumed session that
    # ends in success (exit_code=0) leaves its fixes as uncommitted dirty
    # working-tree files — `git log` shows nothing, `git checkout main`
    # loses them, and the operator has to `git add . && git commit`
    # manually. A resumed session that ends in failure leaves the LLM's
    # bad patches in the workspace with no rollback. Mirroring cmd_run's
    # git lifecycle around the resumed graph fixes both.
    #
    # create_patch_branch is idempotent: it sees the existing
    # agent/patch-<id> branch (created by the original cmd_run that wrote
    # the checkpoint) and just checks it out — same primitive on both
    # paths. If the operator deleted the branch between suspend ↔ resume
    # we recreate it; if they switched to a different branch we re-attach
    # to the agent one. When --git false, _make_git_guardian returns a
    # no-op stub matching the same interface.
    git_guardian = _make_git_guardian(workspace_path)
    git_guardian.stash_if_dirty()
    git_guardian.create_patch_branch(args.session_id)

    try:
        final_state = await run_graph(
            workspace_path=workspace_path,
            prompt=args.prompt or "(resumed session)",
            build_command=build_command,
            allow_network=allow_network,
            budget_usd=budget_usd,
            session_id=args.session_id,
            checkpointer=checkpointer,
            thread_id=args.session_id,
            # Tell run_graph not to build a fresh initial_state that would
            # overwrite the saved channels (messages, loop_counter,
            # current_gate, node_state, etc.) and force the graph to
            # re-enter at requirements_discovery from round 1.
            is_resume=True,
            lintgate_config=config.get("lintgate", {}),
            diagnostics_config=config.get("diagnostics", {}),
            post_mortem_config=config.get("post_mortem", {}),
            deployment_config=config.get("deployment", {}),
            deployment_defaults=load_deployment_defaults(config),
            sandbox_config=config.get("sandbox", {}),
            test_generation_config=config.get("test_generation", {}),
            speculative_config=config.get("speculative", {}),
            compiler_config=config.get("compiler", {}),
            llm_dispatch_config=config.get("llm_dispatch", {}),
        )
    except Exception:
        logger.exception("Resume execution failed.")
        git_guardian.rollback()
        git_guardian.pop_stash()
        await checkpointer.conn.close()
        return 1

    exit_code = final_state.get("exit_code", -1)
    modified_files = final_state.get("modified_files", [])

    # Mirror cmd_run's post-graph dispatch — same branches, same flags,
    # same order. Keeps the two paths symmetric so behaviour after a
    # resume is identical to behaviour after a fresh run.
    node_state = final_state.get("node_state", {}) or {}
    hitl_suspend = bool(node_state.get("hitl_suspend"))
    hitl_abandon = bool(node_state.get("hitl_abandon"))

    if hitl_suspend:
        agent_branch = getattr(git_guardian, "_patch_branch", None) or "agent/patch-<unknown>"
        logger.info(
            "[cli] HITL suspend: leaving %d LLM-modified file(s) on branch "
            "'%s'. Resume with `teane resume --session-id %s` to continue "
            "from the same workspace state.",
            len(modified_files), agent_branch, args.session_id,
        )
    elif hitl_abandon:
        git_guardian.rollback(modified_files)
        git_guardian.pop_stash()
    elif exit_code == 0:
        git_guardian.commit_all_changes(args.session_id, modified_files, exit_code)
        git_guardian.restore_original_branch()
        git_guardian.pop_stash()
    else:
        git_guardian.rollback(modified_files)
        git_guardian.pop_stash()

    logger.info("[resume] Session '%s' completed with exit code %d.", args.session_id, exit_code)

    await checkpointer.conn.close()
    await _drain_mcp_pools()
    pm_note = await _post_mortem_finalize(
        final_state, exit_code, config, workspace_path,
    )
    _append_repo_memory_safely(
        workspace_path=workspace_path,
        session_id=args.session_id,
        prompt_summary=getattr(args, "prompt", "") or "",
        modified_files=modified_files,
        exit_code=exit_code,
        config=config,
        extra_notes=pm_note or None,
    )
    return _resolve_cli_exit_code(
        graph_exit_code=exit_code,
        final_state=final_state,
    )


async def cmd_status(args: argparse.Namespace) -> int:
    """
    Execute the `teane status` subcommand.

    Reads the SQLite checkpoint database read-only and prints a clean
    text snapshot of the specified session's state without triggering
    any graph execution.

    Examples:
        teane status --session-id my-session
        teane status --all
    """
    from harness.storage import HarnessAsyncSqliteSaver, inspect_session, list_all_sessions

    workspace_path = os.path.abspath(args.workspace) if args.workspace else os.getcwd()
    config = discover_config(workspace_path)
    persistence_cfg = config.get("persistence", {})
    db_path = persistence_cfg.get("db_path", "~/.harness/checkpoints.db")
    ttl_days = persistence_cfg.get("ttl_days", 30)
    # JSONL log dir feeds inspect_session's cost reconciliation so
    # `teane status` matches `teane metrics` and the dashboard.
    log_cfg = config.get("observability", {}) or {}
    status_log_dir = os.path.expanduser(log_cfg.get("log_dir", "~/.harness/logs"))

    # Run GC on startup
    checkpointer = await HarnessAsyncSqliteSaver.from_db_path(db_path=db_path, ttl_days=ttl_days)

    if args.all:
        # List all sessions
        sessions = await list_all_sessions(db_path)
        if not sessions:
            print("No checkpointed sessions found.")
        else:
            print(f"{'SESSION ID':<40} {'UPDATED':<20} {'CREATED':<20} {'WORKSPACE':<40}")
            print("-" * 100)
            for s in sessions:
                print(f"{s.thread_id:<40} {s.updated_at:<20} {s.created_at:<20} {s.workspace_path:<40}")
        await checkpointer.conn.close()
        return 0

    if not args.session_id:
        logger.error("Please provide --session-id or use --all to list all sessions.")
        await checkpointer.conn.close()
        return 1

    summary = await inspect_session(db_path, args.session_id, log_dir=status_log_dir)
    if summary is None:
        print(f"No checkpoint found for session '{args.session_id}'.")
        await checkpointer.conn.close()
        return 1

    print("=" * 60)
    print("Session Status")
    print("=" * 60)
    print(f"  Thread ID:          {summary.thread_id}")
    print(f"  Session ID:         {summary.session_id}")
    print(f"  Current Node:       {summary.current_node or '(unknown)'}")
    print(f"  Exit Code:          {summary.exit_code}")
    print(f"  Budget Remaining:   ${summary.budget_remaining_usd:.4f}")
    print(f"  Total Token Cost:   ${summary.total_cost_usd:.6f}")
    print(f"  Modified Files:     {len(summary.modified_files)}")
    for f in summary.modified_files[:10]:
        print(f"    - {f}")
    if len(summary.modified_files) > 10:
        print(f"    ... and {len(summary.modified_files) - 10} more")
    print(f"  Loop Counters:      {summary.loop_counters}")
    print(f"  Is Active:          {summary.is_active}")
    print(f"  Created:            {summary.created_at}")
    print(f"  Updated:            {summary.updated_at}")
    print(f"  Workspace:          {summary.workspace_path}")
    print("=" * 60)

    await checkpointer.conn.close()
    return 0


# ---------------------------------------------------------------------------
# 3b. `teane doctor` — first-run healthcheck
# ---------------------------------------------------------------------------

# ANSI color codes for the doctor report. Skipped when stdout is not a TTY
# so log scrapers and CI captures see plain text. Treat the constants as
# already-emitted-or-empty so callers don't have to branch on isatty.
def _doctor_colors() -> tuple[str, str, str, str]:
    if sys.stdout.isatty() and os.environ.get("NO_COLOR", "") == "":
        return ("\033[32m", "\033[33m", "\033[31m", "\033[0m")  # green, yellow, red, reset
    return ("", "", "", "")


def _format_doctor_line(status: str, label: str, detail: str) -> str:
    green, yellow, red, reset = _doctor_colors()
    if status == "pass":
        marker = f"{green}[ OK ]{reset}"
    elif status == "warn":
        marker = f"{yellow}[WARN]{reset}"
    elif status == "skip":
        marker = f"{yellow}[SKIP]{reset}"
    else:
        marker = f"{red}[FAIL]{reset}"
    return f"  {marker} {label:<32} {detail}"


def _doctor_check_git(workspace_path: str) -> tuple[str, str]:
    """Workspace is a git repo (rev-parse --git-dir) AND HEAD resolves
    AND ``user.name`` + ``user.email`` are configured.

    Identity fields matter because the harness commits on every batch
    when ``--git true`` is passed; without them, ``git commit`` fails
    with "Author identity unknown" and blocks the entire session.
    Silent failure at graph runtime is worse than a WARN here.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "-C", workspace_path, "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except FileNotFoundError:
        return "fail", "git binary not found on PATH"
    except subprocess.TimeoutExpired:
        return "fail", "git rev-parse timed out"
    if result.returncode != 0:
        return "fail", (
            f"{workspace_path} is not a git repo (run 'git init' to initialize)"
        )
    # Repo exists — also confirm HEAD resolves. An unborn HEAD breaks
    # speculative branching (worktree add needs HEAD as the source ref).
    try:
        head_result = subprocess.run(
            ["git", "-C", workspace_path, "rev-parse", "--verify", "--quiet", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        return "warn", "git repo detected, but HEAD verify timed out"
    if head_result.returncode != 0:
        return "warn", (
            f"git repo at {workspace_path} has no commits yet (unborn HEAD); "
            "make an initial commit before running teane to enable speculative repair"
        )
    # Identity fields — probe both user.name and user.email. Missing
    # either breaks `git commit` when --git true is passed. Falls back
    # to global config, so a globally-set identity satisfies this.
    identity_missing: list[str] = []
    for field in ("user.name", "user.email"):
        try:
            id_result = subprocess.run(
                ["git", "-C", workspace_path, "config", "--get", field],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=5,
            )
        except subprocess.TimeoutExpired:
            identity_missing.append(f"{field} (probe timed out)")
            continue
        if id_result.returncode != 0 or not (id_result.stdout or "").strip():
            identity_missing.append(field)
    if identity_missing:
        joined = " and ".join(identity_missing)
        return "warn", (
            f"git repo detected at {workspace_path}, but {joined} is not "
            f"set — `git commit` will fail during --git true runs. Set: "
            f"`git config --global user.name \"Your Name\"` and "
            f"`git config --global user.email you@example.com`"
        )
    return "pass", f"git repo detected at {workspace_path} (identity configured)"


def _doctor_check_builder_image(config: dict[str, Any]) -> tuple[str, str]:
    """Verify the configured ``sandbox.docker_image`` exists locally.

    Only meaningful when sandbox will actually use docker AND the image
    is not a well-known public one (python:*, node:*, eclipse-temurin:*
    all pull cleanly from Docker Hub). The default ``harness-builder:*``
    is a custom local build (harness/vendor/Dockerfile.builder) — not
    on any registry, so an operator on a fresh machine hits "image not
    found" on the first sandbox run.
    """
    sandbox_cfg = config.get("sandbox") or {}
    backend = (sandbox_cfg.get("backend", "auto") or "auto").lower()
    if backend not in ("docker", "auto"):
        return "skip", f"sandbox.backend={backend} (docker image not used)"
    image = (sandbox_cfg.get("docker_image") or "").strip()
    if not image:
        return "skip", "sandbox.docker_image not set"
    if shutil.which("docker") is None:
        return "skip", "docker not on PATH (probed elsewhere)"
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return "warn", f"probe failed: {type(exc).__name__}: {exc}"
    if proc.returncode == 0:
        return "pass", f"{image} present locally"
    # Distinguish "custom local image" (harness-builder) from "public
    # registry image" (python:3.12-slim) — the former needs a build
    # command, the latter will auto-pull on first `docker run`.
    if image.startswith("harness-builder"):
        return "fail", (
            f"{image} not built locally — this is a custom image "
            f"(harness/vendor/Dockerfile.builder), no registry pull "
            f"will succeed. Build: `docker build --pull -f "
            f"harness/vendor/Dockerfile.builder -t {image} "
            f"harness/vendor/`"
        )
    return "warn", (
        f"{image} not present locally — Docker will attempt registry "
        f"pull on first sandbox run (may fail on air-gapped hosts). "
        f"Pre-pull with: `docker pull {image}`"
    )


def _doctor_check_mcp_commands(config: dict[str, Any]) -> list[tuple[str, tuple[str, str]]]:
    """For each ``mcp.servers[*].command``, verify command[0] is on PATH.

    Complements ``_doctor_check_mcp`` which starts each server and
    checks tool advertisement — but that requires the binary to be
    resolvable first. When uvx / npx / a custom server binary is
    missing, the spawn fails with a vague error; this surfaces the
    root cause with an actionable install hint.
    """
    mcp_cfg = config.get("mcp") or {}
    if not mcp_cfg.get("enabled", False):
        return [(
            "mcp commands",
            ("skip", "mcp.enabled=false"),
        )]
    servers = mcp_cfg.get("servers") or []
    if not isinstance(servers, list) or not servers:
        return [(
            "mcp commands",
            ("skip", "mcp.enabled=true but no servers configured"),
        )]
    rows: list[tuple[str, tuple[str, str]]] = []
    seen: set[str] = set()
    for server in servers:
        if not isinstance(server, dict):
            continue
        name = str(server.get("name") or "<unnamed>")
        command = server.get("command") or []
        if not isinstance(command, list) or not command:
            rows.append((
                f"mcp {name} command",
                ("warn", f"malformed command (expected list, got {type(command).__name__})"),
            ))
            continue
        binary = str(command[0])
        if binary in seen:
            continue  # dedup common wrappers (npx, uvx)
        seen.add(binary)
        install_hint = _MCP_COMMAND_INSTALL_HINTS.get(binary, "")
        if shutil.which(binary) is None:
            detail = (
                f"`{binary}` not on PATH — mcp server '{name}' will "
                f"fail to spawn"
            )
            if install_hint:
                detail += f" (install: {install_hint})"
            rows.append((f"mcp: {binary}", ("fail", detail)))
        else:
            rows.append((
                f"mcp: {binary}",
                ("pass", f"on PATH (used by mcp server '{name}')"),
            ))
    return rows


# Install hints for the binaries commonly seen as ``mcp.servers[*].command[0]``.
_MCP_COMMAND_INSTALL_HINTS: dict[str, str] = {
    "uvx": "pip install uv",
    "npx": "install Node.js (bundles npx)",
    "python3": "install Python 3.11+",
    "python": "install Python 3.11+",
    "docker": "install Docker Engine",
}


def _doctor_check_config_paths(config: dict[str, Any]) -> list[tuple[str, tuple[str, str]]]:
    """Validate the path-like fields in the operator's config actually
    exist / are writable / aren't host-specific leftovers.

    Fields checked:
      - ``mcp.servers[fs].command[-1]`` — filesystem MCP server root arg
      - ``deployment_defaults.storage.volume_root``
      - ``deployment_defaults.storage.backup_path`` (when
        ``backup_destination`` is not none/s3/gcs/azure_blob)
      - ``sandbox.readonly_cache_mounts[*]`` — soft warn if missing

    Absolute paths that don't resolve on this host are flagged — this
    catches the specific class of bug where a colleague's config got
    committed with their machine's ``/mnt/*`` paths.
    """
    rows: list[tuple[str, tuple[str, str]]] = []

    def _looks_host_specific(p: str) -> bool:
        # Absolute path outside the operator's home dir that doesn't
        # exist — likely a leftover from another host. ~/... paths are
        # fine (they expand); /var, /opt, /srv are legitimate on some
        # hosts; but /mnt/<username>/ or /home/<other>/ almost always is
        # a leftover.
        if not p or not os.path.isabs(p):
            return False
        home = os.path.expanduser("~")
        return not p.startswith(home) and not os.path.exists(p)

    def _expand(p: str) -> str:
        return os.path.expanduser(os.path.expandvars(p or ""))

    # --- MCP filesystem server root -----------------------------------------
    mcp_cfg = config.get("mcp") or {}
    if mcp_cfg.get("enabled", False):
        for server in (mcp_cfg.get("servers") or []):
            if not isinstance(server, dict):
                continue
            command = server.get("command") or []
            if not (isinstance(command, list) and command
                    and str(command[0]) == "npx"
                    and any("server-filesystem" in str(c) for c in command)):
                continue
            root_arg = str(command[-1]) if len(command) > 1 else ""
            expanded = _expand(root_arg)
            name = str(server.get("name") or "fs")
            if not root_arg:
                rows.append((f"mcp: {name} root", ("warn", "no root arg given to filesystem server")))
                continue
            if not os.path.isdir(expanded):
                sev = "fail" if _looks_host_specific(root_arg) else "warn"
                rows.append((
                    f"mcp: {name} root",
                    (
                        sev,
                        f"filesystem MCP root `{root_arg}` "
                        f"(expanded: `{expanded}`) does not exist on "
                        f"this host — narrow it to your workspace or "
                        f"remove the fs server from mcp.servers",
                    ),
                ))
            else:
                rows.append((
                    f"mcp: {name} root",
                    ("pass", f"filesystem MCP root `{expanded}` exists"),
                ))

    # --- Deployment storage paths --------------------------------------------
    storage = ((config.get("deployment_defaults") or {}).get("storage")) or {}
    backup_dest = str(storage.get("backup_destination") or "none").lower()
    for field in ("volume_root", "backup_path"):
        raw = str(storage.get(field) or "")
        if not raw:
            continue
        # backup_path only relevant when backup_destination is local-ish.
        if field == "backup_path" and backup_dest in ("none", "s3", "gcs", "azure_blob"):
            rows.append((
                f"deploy: {field}",
                ("skip", f"backup_destination={backup_dest} (host path not used)"),
            ))
            continue
        expanded = _expand(raw)
        if _looks_host_specific(raw):
            rows.append((
                f"deploy: {field}",
                (
                    "fail",
                    f"`{raw}` looks like a host-specific path from "
                    f"another machine and doesn't exist here — set "
                    f"deployment_defaults.storage.{field} to a "
                    f"portable value like `~/.harness/deploy/*` or "
                    f"an absolute path that exists on this host",
                ),
            ))
        elif os.path.isabs(expanded) and not os.path.isdir(expanded):
            # Parent-directory writability check — the harness creates
            # these on first deploy, so missing dir is only a warning if
            # its parent isn't writable.
            parent = os.path.dirname(expanded) or "/"
            if os.path.isdir(parent) and os.access(parent, os.W_OK):
                rows.append((
                    f"deploy: {field}",
                    ("pass", f"`{expanded}` will be created on first deploy (parent writable)"),
                ))
            else:
                rows.append((
                    f"deploy: {field}",
                    (
                        "warn",
                        f"`{expanded}` does not exist and parent `{parent}` "
                        f"is not writable — first deploy will fail. "
                        f"Create the parent or pick a different path.",
                    ),
                ))
        else:
            rows.append((f"deploy: {field}", ("pass", f"`{expanded}`")))

    # --- Sandbox readonly cache mounts --------------------------------------
    sandbox_cfg = config.get("sandbox") or {}
    for mount in (sandbox_cfg.get("readonly_cache_mounts") or []):
        raw = str(mount)
        expanded = _expand(raw)
        if not os.path.isdir(expanded):
            # Cache mounts are optional performance boosts — the sandbox
            # skips missing ones cleanly. Log at info-ish severity.
            rows.append((
                f"sandbox: cache mount {raw}",
                ("skip", f"`{expanded}` does not exist (cache mount skipped, no impact)"),
            ))
    return rows


def _doctor_check_env_placeholders() -> list[tuple[str, tuple[str, str]]]:
    """Report every ``${TEANE_*}`` / ``${HARNESS_*}`` placeholder in the
    RAW config file and where its value came from (env override vs the
    ``:-default``). Env-resolved config is invisible state — this is the
    line that answers "why is it using that path?" without grepping
    shell profiles.
    """
    rows: list[tuple[str, tuple[str, str]]] = []
    path = _get_global_config_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except OSError:
        return rows  # the config check itself reports unreadable files
    seen: set[str] = set()
    for match in _ENV_PLACEHOLDER_RE.finditer(raw_text):
        var, default = match.group(1), match.group(2)
        if var in seen:
            continue
        seen.add(var)
        value = os.environ.get(var)
        if value is not None:
            rows.append((
                f"env override: {var}",
                ("pass", f"set in environment → `{value}`"),
            ))
        elif default is not None:
            rows.append((
                f"env override: {var}",
                ("pass", f"unset — default used → `{default}`"),
            ))
        else:
            rows.append((
                f"env override: {var}",
                ("fail", f"unset and no ':-default' in config — export {var}"),
            ))
    return rows


def _preflight_config_env_report(config: dict[str, Any]) -> None:
    """One startup log line per ``${TEANE_*}``/``${HARNESS_*}`` placeholder
    (env override vs default), plus a WARNING when the resolved MCP
    filesystem root doesn't exist on this host.

    Hard failures (unset var without default, malformed placeholder) are
    already raised by ``load_raw_config`` before this runs — this is the
    visibility half: env-resolved config is invisible state, and the
    session log should answer "which value fed that path?" without a
    trip to ``teane doctor``. Never raises; a nonexistent MCP root is a
    degradation (the server start fails and is skipped), not an abort.
    """
    try:
        for name, (status, detail) in _doctor_check_env_placeholders():
            logger.info("[preflight] %s: %s", name, detail)

        mcp_cfg = config.get("mcp") or {}
        if not mcp_cfg.get("enabled", False):
            return
        for server in (mcp_cfg.get("servers") or []):
            if not isinstance(server, dict):
                continue
            command = server.get("command") or []
            if not (isinstance(command, list) and len(command) > 1
                    and any("server-filesystem" in str(c) for c in command)):
                continue
            root = str(command[-1])
            if root and not os.path.isdir(root):
                logger.warning(
                    "[preflight] MCP filesystem root `%s` (server `%s`) "
                    "does not exist on this host — the fs server will "
                    "fail to start and its tools will be missing. Export "
                    "TEANE_MCP_FS_ROOT or fix mcp.servers in config.json.",
                    root, server.get("name") or "fs",
                )
    except Exception:  # noqa: BLE001 — reporting must never block startup
        logger.debug("[preflight] config env report failed", exc_info=True)


# Search backends the harness ships built-in, which never need an
# API key even when named as the ``web_tools.search_backend`` primary.
# Any name outside this set is user-supplied and expected to declare a
# working ``api_key_env``.
_BUILTIN_SEARCH_BACKENDS: frozenset[str] = frozenset({
    "duckduckgo_lite", "ddg", "duckduckgo",
})


def _doctor_check_env_vars_from_config(
    config: dict[str, Any],
) -> list[tuple[str, tuple[str, str]]]:
    """Scan the config for every ``*_env`` field naming an environment
    variable the operator must set, and verify each is present.

    LLM provider keys are handled separately by ``_doctor_check_api_keys``.
    This check covers the non-LLM surface:
      - ``web_tools.api_key_env`` (Tavily, Brave, etc.) — only when the
        primary backend is NOT one of the built-in no-key ones.
      - ``web_tools.backends[*].api_key_env`` — same rule per fallback.
      - ``mcp.servers[*].api_key_env`` — MCP servers that authenticate.
      - ``dashboard.token_env`` / ``dashboard.csrf_token_env`` — set
        only when the operator opts in to authenticated dashboard.

    Returns one row per required env var so the operator sees exactly
    which one is missing (not a lumped "N env vars missing" tail).
    """
    rows: list[tuple[str, tuple[str, str]]] = []
    # (env_var, feature_label, install_hint)
    required: list[tuple[str, str, str]] = []

    # --- web_tools primary + fallback chain ---------------------------------
    # Track whether a keyless / no-API-key fallback is enabled — if so,
    # missing keys on other web_tools entries downgrade from FAIL to
    # WARN because the chain will still serve queries (just from the
    # fallback instead of the preferred backend).
    web_tools = config.get("web_tools") or {}
    web_has_keyless_fallback = False
    if web_tools.get("enabled", False):
        # Detect keyless fallback first — informs severity below.
        for backend in (web_tools.get("backends") or []):
            if not isinstance(backend, dict):
                continue
            if not backend.get("enabled", True):
                continue
            fb_name = str(backend.get("search_backend") or "").strip().lower()
            if fb_name in _BUILTIN_SEARCH_BACKENDS:
                web_has_keyless_fallback = True
                break
        primary = str(web_tools.get("search_backend") or "").strip().lower()
        primary_env = str(web_tools.get("api_key_env") or "").strip()
        if primary and primary not in _BUILTIN_SEARCH_BACKENDS and primary_env:
            required.append((
                primary_env,
                f"web_tools primary backend `{primary}`",
                f'export {primary_env}="..."  # your {primary} API key',
            ))
        for backend in (web_tools.get("backends") or []):
            if not isinstance(backend, dict):
                continue
            if not backend.get("enabled", True):
                continue
            name = str(backend.get("search_backend") or "").strip().lower()
            env = str(backend.get("api_key_env") or "").strip()
            if name and name not in _BUILTIN_SEARCH_BACKENDS and env:
                label = str(backend.get("name") or name)
                required.append((
                    env,
                    f"web_tools fallback backend `{label}`",
                    f'export {env}="..."  # your {name} API key',
                ))

    # --- MCP servers --------------------------------------------------------
    mcp_cfg = config.get("mcp") or {}
    if mcp_cfg.get("enabled", False):
        for server in (mcp_cfg.get("servers") or []):
            if not isinstance(server, dict):
                continue
            env = str(server.get("api_key_env") or "").strip()
            if not env:
                continue
            name = str(server.get("name") or "<unnamed>")
            required.append((
                env,
                f"mcp server `{name}`",
                f'export {env}="..."  # required by mcp server {name!r}',
            ))

    # --- Dashboard auth -----------------------------------------------------
    dashboard = config.get("dashboard") or {}
    for field, purpose in (
        ("token_env", "dashboard bearer token"),
        ("csrf_token_env", "dashboard CSRF token"),
    ):
        env = str(dashboard.get(field) or "").strip()
        if not env:
            continue
        required.append((
            env,
            f"{purpose} (dashboard.{field})",
            f'export {env}="..."  # {purpose}',
        ))

    if not required:
        return [(
            "config env vars",
            ("skip", "no non-LLM env vars declared in config"),
        )]

    # De-duplicate on env-var name — one row per distinct variable,
    # accumulating the features that rely on it.
    grouped: dict[str, list[str]] = {}
    hints: dict[str, str] = {}
    for env, feature, hint in required:
        grouped.setdefault(env, []).append(feature)
        hints.setdefault(env, hint)
    for env in sorted(grouped):
        features = grouped[env]
        feature_str = "; ".join(features)
        value = (os.environ.get(env) or "").strip()
        if value:
            rows.append((
                f"env: {env}",
                ("pass", f"set ({feature_str})"),
            ))
        else:
            # Downgrade FAIL → WARN when the feature has a working
            # fallback that doesn't need this key. Currently only the
            # web_tools chain works this way — if a keyless built-in
            # backend (duckduckgo_lite / ddg / duckduckgo) is enabled,
            # a missing web_tools API key means "we'll use DDG instead"
            # not "we're broken". MCP + dashboard have no such fallback,
            # so they stay FAIL.
            web_only = all("web_tools" in f for f in features)
            if web_only and web_has_keyless_fallback:
                rows.append((
                    f"env: {env}",
                    (
                        "warn",
                        f"not set — {feature_str} will silently fall "
                        f"back to the keyless built-in backend "
                        f"(duckduckgo_lite). Set it to prefer the "
                        f"configured backend: {hints[env]}",
                    ),
                ))
            else:
                rows.append((
                    f"env: {env}",
                    (
                        "fail",
                        f"NOT set but required by: {feature_str}. "
                        f"Fix: {hints[env]}",
                    ),
                ))
    return rows


def _doctor_check_ollama_daemon(config: dict[str, Any]) -> tuple[str, str]:
    """When any routed model has provider=ollama OR any model in the
    registry is an ollama model, verify the ollama daemon is reachable
    at its configured (or default) HTTP endpoint.

    Only fires when ollama is actually referenced — if the config uses
    exclusively cloud providers, this returns skip and never touches
    the network.
    """
    models_cfg = config.get("models") or {}
    routing = config.get("model_routing") or {}
    routed_ollama: list[str] = []
    routing_keys = (
        "planning_primary", "planning_fallback",
        "patching_primary",
        "repair_primary", "repair_fallback",
        "doc_reviewer_primary", "doc_reviewer_fallback",
        "code_reviewer_primary", "code_reviewer_fallback",
    )
    for k in routing_keys:
        v = routing.get(k) or ""
        if isinstance(v, str) and v.startswith("ollama:"):
            routed_ollama.append(v)
    if not routed_ollama:
        return "skip", "no ollama models in model_routing"
    # Discover endpoint from the first routed ollama model, falling back
    # to the default localhost port.
    endpoint = "http://localhost:11434"
    for k in routed_ollama:
        entry = models_cfg.get(k) or {}
        if isinstance(entry, dict):
            candidate = str(entry.get("api_base_url") or "").rstrip("/")
            if candidate:
                endpoint = candidate.rstrip("/v1").rstrip("/")
                break
    # Cheap TCP probe — we only need to know the daemon is listening,
    # not that it can serve the specific model.
    import socket as _sock
    import urllib.parse
    try:
        parsed = urllib.parse.urlparse(endpoint)
        host = parsed.hostname or "localhost"
        port = parsed.port or 11434
    except Exception:  # noqa: BLE001
        host, port = "localhost", 11434
    try:
        with _sock.create_connection((host, port), timeout=2.0):
            pass
    except (OSError, _sock.timeout) as exc:
        return "fail", (
            f"ollama models routed ({', '.join(routed_ollama)}) but "
            f"daemon at {host}:{port} not reachable ({type(exc).__name__}). "
            f"Start with `ollama serve` or unset ollama routing"
        )
    return "pass", (
        f"ollama daemon reachable at {host}:{port} "
        f"({len(routed_ollama)} model(s) routed)"
    )


# Per-provider HTTP probe targets for the live api-keys check. Endpoint
# is the smallest possible chat call that exercises auth + the model id
# — we cap output to a single token so the cost-per-doctor-run stays
# well under a tenth of a cent across all providers combined.
_LIVE_PING_TIMEOUT_SECONDS = 8.0


async def _ping_provider_live(
    provider: str, model_id: str, api_key: str,
    *,
    timeout: float = _LIVE_PING_TIMEOUT_SECONDS,
) -> tuple[bool, str]:
    """Make the smallest possible chat call to confirm the key authenticates.

    Returns ``(ok, message)``. ``ok=True`` means the provider accepted
    the key for this model (HTTP 200). Anything else is a FAIL with a
    specific operator-actionable reason — 401 names the key as invalid,
    403 distinguishes "key valid but no access to model", 429 calls out
    that the key works but quota is exhausted, network errors point at
    reachability.
    """
    import httpx
    if not api_key:
        return False, "no API key resolved"

    try:
        if provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            body: dict[str, Any] = {
                "model": model_id,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "ping"}],
            }
        elif provider in {"openai", "deepseek"}:
            base = (
                "https://api.deepseek.com" if provider == "deepseek"
                else "https://api.openai.com"
            )
            url = f"{base}/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": model_id,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "ping"}],
            }
        else:
            return False, f"unknown provider '{provider}' (no live-ping probe registered)"

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=body)
    except httpx.TimeoutException:
        return False, (
            f"timeout after {timeout:.0f}s — provider unreachable or network blocked"
        )
    except httpx.ConnectError as exc:
        return False, f"connection failed ({exc})"
    except Exception as exc:  # noqa: BLE001 — surface anything unexpected
        return False, f"{type(exc).__name__}: {exc}"

    if response.status_code == 200:
        return True, "live"
    if response.status_code == 401:
        return False, "HTTP 401 — API key rejected (verify the key is correct and active)"
    if response.status_code == 403:
        return False, (
            f"HTTP 403 — key is valid but has no access to model '{model_id}'"
        )
    if response.status_code == 404:
        return False, (
            f"HTTP 404 — model '{model_id}' not found at provider; check spelling"
        )
    if response.status_code == 429:
        return False, (
            "HTTP 429 — rate limited (key works but quota exhausted right now)"
        )
    if 500 <= response.status_code < 600:
        return False, f"HTTP {response.status_code} — provider error"
    # Anything else: surface the status + truncated body so the operator
    # can paste it into a search engine.
    snippet = response.text[:200].replace("\n", " ").strip()
    return False, f"HTTP {response.status_code}: {snippet}"


async def _doctor_check_api_keys(config: dict[str, Any]) -> tuple[str, str]:
    """Every non-ollama model referenced in model_routing has a working API key.

    Two-phase check:
      1. Key resolution — same as before: env var OR per-model
         ``api_key`` config field, matching ``BaseProviderClient.__init__``.
      2. Live ping — for each provider that resolved a key, make the
         smallest possible chat call. Pings run in parallel via
         ``asyncio.gather`` so the doctor stays fast even with 3-4
         providers configured.

    Set ``HARNESS_DOCTOR_SKIP_LIVE=true`` to skip the live ping (key
    presence only). Useful in headless CI where outbound network may be
    blocked or where the per-check latency matters more than the
    correctness signal.
    """
    routing = config.get("model_routing", {}) or {}
    models_cfg = config.get("models", {}) or {}
    routing_keys = (
        "planning_primary", "planning_fallback",
        "patching_primary",
        "repair_primary", "repair_fallback",
        "doc_reviewer_primary", "doc_reviewer_fallback",
        "code_reviewer_primary", "code_reviewer_fallback",
    )
    needed_providers: dict[str, str] = {}  # provider -> first model that wants it
    for routing_key in routing_keys:
        model_key = routing.get(routing_key, "") or ""
        if not model_key or ":" not in model_key:
            continue
        provider = model_key.split(":", 1)[0]
        if provider == "ollama":
            continue
        needed_providers.setdefault(provider, model_key)

    if not needed_providers:
        return "warn", "no non-ollama models configured in model_routing"

    # Phase 1: resolve keys.
    resolved: list[tuple[str, str, str, str]] = []  # (provider, model_key, source, key)
    missing: list[str] = []
    for provider, model_key in needed_providers.items():
        env_var = f"{provider.upper()}_API_KEY"
        env_value = (os.environ.get(env_var, "") or "").strip()
        cfg_entry = models_cfg.get(model_key, {}) or {}
        cfg_value = (cfg_entry.get("api_key", "") or "").strip() if isinstance(cfg_entry, dict) else ""
        if env_value:
            resolved.append((provider, model_key, "env", env_value))
        elif cfg_value:
            resolved.append((provider, model_key, "config", cfg_value))
        else:
            missing.append(
                f"{model_key} (set {env_var} env var, or "
                f"models.\"{model_key}\".api_key in ~/.harness/config.json)"
            )

    if missing:
        return "fail", "missing: " + "; ".join(missing)

    skip_live = (os.environ.get("HARNESS_DOCTOR_SKIP_LIVE", "") or "").strip().lower() in (
        "1", "true", "yes",
    )
    if skip_live:
        return (
            "pass",
            "present: " + ", ".join(f"{m} ({s})" for _p, m, s, _k in resolved)
            + " (live ping skipped via HARNESS_DOCTOR_SKIP_LIVE)",
        )

    # Phase 2: live ping in parallel.
    import asyncio as _asyncio
    ping_results = await _asyncio.gather(
        *[
            _ping_provider_live(provider, model_key.split(":", 1)[1], key)
            for provider, model_key, _src, key in resolved
        ],
        return_exceptions=True,
    )

    live_failures: list[str] = []
    live_present: list[str] = []
    for (provider, model_key, source, _key), result in zip(resolved, ping_results):
        if isinstance(result, BaseException):
            live_failures.append(f"{model_key} ({source}): unexpected {type(result).__name__}: {result}")
            continue
        ok, detail = result
        if ok:
            live_present.append(f"{model_key} ({source}, live)")
        else:
            live_failures.append(f"{model_key} ({source}): {detail}")

    if live_failures:
        return "fail", "live ping failed — " + "; ".join(live_failures)
    return "pass", "live: " + ", ".join(live_present)


# Caps the file walk that detects which extensions are present in the
# workspace. Doctor must stay fast; this is enough to be representative
# without scanning huge generated/vendored trees.
_DOCTOR_EXTENSION_SCAN_FILE_LIMIT = 5000
_DOCTOR_EXTENSION_SCAN_PRUNED_DIRS = frozenset({
    ".git", "node_modules", ".venv", "venv", "__pycache__",
    "dist", "build", ".next", ".cache", "target",
})


def _doctor_workspace_extensions(workspace_path: str) -> set[str]:
    """Return the set of file extensions present in the workspace.

    Used to suppress formatter rows for languages the project doesn't
    use — surfacing a missing ``clang-format`` warning on a pure-Python
    project is noise, not signal. Bounded by file count and skips the
    usual large directories so doctor stays fast.
    """
    present: set[str] = set()
    if not os.path.isdir(workspace_path):
        return present
    seen = 0
    for root, dirs, files in os.walk(workspace_path):
        dirs[:] = [d for d in dirs if d not in _DOCTOR_EXTENSION_SCAN_PRUNED_DIRS]
        for name in files:
            seen += 1
            if seen > _DOCTOR_EXTENSION_SCAN_FILE_LIMIT:
                return present
            ext = os.path.splitext(name)[1].lower()
            if ext:
                present.add(ext)
    return present


# Minimum TypeScript compiler version. tsconfig.json syntax and the
# tsc CLI surface (--noEmit shape, project references) evolve; the
# diagnostics_gate contract assumes 5.x-era behaviour. Warn (not fail)
# below this floor — old tsc + modern config = silent parse failures.
_MIN_TSC_MAJOR = 5


def _probe_tsc_version() -> tuple[str, str, str]:
    """Return ``(status, version_str, install_hint)`` for the tsc CLI.

    ``status`` follows the doctor convention: ``pass`` when tsc is on
    PATH at or above :data:`_MIN_TSC_MAJOR`; ``warn`` when it's present
    but too old (silent tsconfig parse failures) or the version probe
    itself fails; ``fail`` when the binary is missing entirely.
    """
    path = shutil.which("tsc")
    if path is None:
        return "fail", "", "npm install -g typescript"
    try:
        proc = subprocess.run(
            ["tsc", "--version"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return "warn", f"probe failed: {type(exc).__name__}", ""
    version_raw = ((proc.stdout or "") + (proc.stderr or "")).strip()
    m = re.search(r"Version\s+(\d+)\.(\d+)", version_raw)
    if not m:
        return "warn", version_raw or "unknown", ""
    try:
        major = int(m.group(1))
    except ValueError:
        return "warn", version_raw, ""
    if major < _MIN_TSC_MAJOR:
        return (
            "warn",
            f"{version_raw} (below floor {_MIN_TSC_MAJOR}.x — modern "
            f"tsconfig.json syntax may fail to parse in diagnostics gate)",
            "npm install -g typescript@latest",
        )
    return "pass", version_raw, ""


def _doctor_check_lsp(config: dict[str, Any]) -> list[tuple[str, tuple[str, str]]]:
    """When ``lsp.enabled=true``, verify the language-server binaries
    the pool will spawn are actually on PATH. Silent fallback to regex
    extraction is the current failure mode when they're missing — the
    operator gets no signal that semantic navigation degraded to
    string matching.

    Returns per-server rows so each binary is called out individually
    (same UX as the security-scanner + formatter rows). When LSP is
    disabled, returns a single ``skip`` row for the group.
    """
    lsp_cfg = config.get("lsp") or {}
    if not lsp_cfg.get("enabled", False):
        return [(
            "external: lsp servers",
            ("skip", "lsp.enabled=false (LSP pool not started)"),
        )]
    servers = [
        (
            "pyright-langserver",
            "Python LSP navigation",
            "npm install -g pyright",
        ),
        (
            "typescript-language-server",
            "TS/TSX LSP navigation",
            "npm install -g typescript-language-server typescript",
        ),
    ]
    rows: list[tuple[str, tuple[str, str]]] = []
    for binary, feature, install_hint in servers:
        if shutil.which(binary) is None:
            rows.append((
                f"external: {binary}",
                (
                    "warn",
                    f"not on PATH ({feature}) — lsp.enabled=true but "
                    f"pool will silently fall back to regex extraction. "
                    f"Install: {install_hint}",
                ),
            ))
        else:
            rows.append((
                f"external: {binary}",
                ("pass", f"on PATH ({feature})"),
            ))
    return rows


def _doctor_check_external_tools(
    config: dict[str, Any],
    workspace_path: str,
) -> list[tuple[str, tuple[str, str]]]:
    """Probe every external binary the harness shells out to.

    Returns one ``(label, (status, detail))`` row per tool so each appears
    on its own line in the doctor report. Severity is config-aware: a tool
    is only marked ``fail`` when the operator's configuration actually
    relies on it (e.g. docker when ``sandbox.backend == "docker"``).
    Missing optional tools that have a working fallback become ``warn``.
    Formatter rows are suppressed for extensions that don't appear in the
    workspace.

    The install hint (from ``security.SCANNER_INSTALL_HINTS`` for the
    scanners and ``lintgate._DEFAULT_FORMATTERS[*].install_hint`` for the
    formatters) is appended to the detail line on warn/fail rows.
    """
    from harness import lintgate, security

    rows: list[tuple[str, tuple[str, str]]] = []

    def _append(name: str, status: str, detail: str, hint: str = "") -> None:
        if status in ("warn", "fail") and hint:
            detail = f"{detail} — install: {hint}"
        rows.append((f"external: {name}", (status, detail)))

    security_cfg = config.get("security_scan") or config.get("security") or {}
    enabled_scanners = tuple(security_cfg.get("scanners", security._DEFAULT_SCANNERS))

    # --- Security scanners ---------------------------------------------------
    scanners_present = 0
    scanners_enabled = 0
    for scanner in ("gitleaks", "bandit", "semgrep", "trivy"):
        hint = security.SCANNER_INSTALL_HINTS.get(scanner, "")
        if scanner not in enabled_scanners:
            _append(scanner, "skip", f"not in security.scanners ({list(enabled_scanners)})")
            continue
        scanners_enabled += 1
        if shutil.which(scanner) is None:
            if scanner == "gitleaks":
                # Real gitleaks falls back to the in-process regex scanner.
                # Count as "present" because coverage is preserved.
                scanners_present += 1
                _append(scanner, "warn", "not on PATH (Python fallback active)", hint)
            else:
                # bandit/semgrep/trivy have no in-process fallback — they
                # are simply skipped when missing, reducing scan coverage.
                _append(scanner, "warn", "not on PATH (scanner will be skipped)", hint)
        else:
            scanners_present += 1
            _append(scanner, "pass", "on PATH")
    # SAST-floor row: if the operator configured scanners but NONE are
    # actually available (no binary AND no fallback), security scans
    # will produce empty reports and the operator has no signal that
    # SAST coverage is zero. Surface this as an explicit row so it can't
    # be missed among the per-scanner rows.
    if scanners_enabled > 0 and scanners_present == 0:
        _append(
            "sast coverage",
            "fail",
            (
                f"{scanners_enabled} scanner(s) configured in "
                f"security.scanners but NONE runnable on this host — "
                f"security scans will produce empty reports"
            ),
            "install at least one of the scanners listed above",
        )
    elif scanners_enabled > 0:
        _append(
            "sast coverage",
            "pass",
            (
                f"{scanners_present}/{scanners_enabled} configured "
                f"scanner(s) available (incl. Python fallback for "
                f"gitleaks when missing)"
            ),
        )

    # --- Sandbox / deployment binaries --------------------------------------
    sandbox_backend = (config.get("sandbox", {}).get("backend", "auto") or "auto").lower()
    docker_present = shutil.which("docker") is not None
    unshare_present = shutil.which("unshare") is not None

    if sandbox_backend == "docker":
        if docker_present:
            _append("docker", "pass", "on PATH (sandbox.backend=docker)")
        else:
            _append(
                "docker", "fail",
                "not on PATH but sandbox.backend=docker",
                "install Docker Engine: https://docs.docker.com/engine/install/",
            )
    elif sandbox_backend == "unshare":
        _append(
            "docker", "skip",
            "sandbox.backend=unshare (docker not required for sandbox)",
        )
    elif sandbox_backend == "bare":
        _append("docker", "skip", "sandbox.backend=bare (no isolation requested)")
    else:  # auto
        if docker_present:
            _append("docker", "pass", "on PATH (sandbox.backend=auto)")
        elif unshare_present:
            _append(
                "docker", "warn",
                "not on PATH; sandbox.backend=auto will fall back to unshare",
                "install Docker Engine: https://docs.docker.com/engine/install/",
            )
        else:
            _append(
                "docker", "fail",
                "not on PATH and unshare also missing (sandbox.backend=auto)",
                "install Docker Engine: https://docs.docker.com/engine/install/",
            )

    deployment_enabled = bool(config.get("deployment", {}).get("enabled", False))
    compose_present = shutil.which("docker-compose") is not None or (
        docker_present and _has_docker_compose_subcommand()
    )
    if deployment_enabled:
        if compose_present:
            _append("docker-compose", "pass", "compose available (deployment.enabled)")
        else:
            _append(
                "docker-compose", "fail",
                "not available but deployment.enabled=true",
                "install Docker Compose: https://docs.docker.com/compose/install/",
            )
        # buildx is required for the multi-stage Dockerfiles teane deploy
        # synthesises. Older Docker installs may have the daemon but not
        # the buildx CLI plugin — deployment fails late without a good
        # signal at doctor time.
        if docker_present and _has_docker_buildx():
            _append("docker buildx", "pass", "buildx CLI plugin available")
        elif docker_present:
            _append(
                "docker buildx", "fail",
                "buildx CLI plugin missing but deployment.enabled=true "
                "(`teane deploy` uses multi-stage builds)",
                "install Docker Buildx: https://docs.docker.com/build/buildx/install/",
            )
        else:
            _append("docker buildx", "skip", "docker missing (see row above)")
    else:
        _append(
            "docker-compose", "skip",
            "deployment.enabled=false (compose not required)",
        )
        _append(
            "docker buildx", "skip",
            "deployment.enabled=false (buildx not required)",
        )

    # --- Type-checker version floor (diagnostics gate) -----------------------
    # diagnostics_gate.py invokes `tsc --noEmit` for TS files. When tsc
    # is present but below the version floor, modern tsconfig.json
    # syntax silently fails to parse — the gate reports zero diagnostics
    # instead of the real errors. Only surface when the diagnostics gate
    # is enabled AND tsc is listed in diagnostics.tools (or tools is
    # unset — treated as "all").
    diagnostics_cfg = config.get("diagnostics") or {}
    diagnostics_enabled = bool(diagnostics_cfg.get("enabled", True))
    diagnostics_tools = diagnostics_cfg.get("tools")
    tsc_in_tools = (
        diagnostics_tools is None
        or "tsc" in [str(t).lower() for t in (diagnostics_tools or [])]
    )
    if diagnostics_enabled and tsc_in_tools:
        tsc_status, tsc_detail, tsc_hint = _probe_tsc_version()
        if tsc_status == "fail":
            _append(
                "tsc version",
                "warn",
                "tsc not on PATH — TS files in the workspace will skip "
                "the diagnostics gate (Python-only projects can ignore)",
                tsc_hint,
            )
        elif tsc_status == "warn":
            _append("tsc version", "warn", tsc_detail, tsc_hint)
        else:
            _append("tsc version", "pass", tsc_detail)
    else:
        _append(
            "tsc version", "skip",
            "diagnostics gate disabled or tsc excluded from diagnostics.tools",
        )

    # --- Formatters / linters from lintgate (per-extension) -----------------
    present_exts = _doctor_workspace_extensions(workspace_path)
    seen_commands: set[str] = set()
    for ext, spec in lintgate._DEFAULT_FORMATTERS.items():
        if ext not in present_exts:
            continue
        for cmd in (spec.command, spec.linter_command):
            if not cmd or cmd in seen_commands:
                continue
            seen_commands.add(cmd)
            if shutil.which(cmd) is not None:
                _append(cmd, "pass", f"on PATH (used for {ext} files)")
            else:
                _append(
                    cmd, "warn",
                    f"not on PATH; {ext} files will skip auto-format",
                    spec.install_hint,
                )

    return rows


def _has_docker_compose_subcommand() -> bool:
    """Detect ``docker compose`` (v2 plugin) when the legacy ``docker-compose``
    binary is absent. Cheap probe; returns False on any error."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=3,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _has_docker_buildx() -> bool:
    """Detect the ``docker buildx`` CLI plugin. Required for the
    multi-stage Dockerfiles emitted by ``teane deploy``. Cheap probe;
    returns False on any error."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "buildx", "version"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=3,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _doctor_check_sandbox(config: dict[str, Any]) -> tuple[str, str]:
    """Sandbox backend is reachable (docker info / unshare echo)."""
    import shutil
    import subprocess
    sandbox_cfg = config.get("sandbox", {}) or {}
    backend = (sandbox_cfg.get("backend", "auto") or "auto").lower()

    def _probe_docker() -> tuple[str, str]:
        if shutil.which("docker") is None:
            return "fail", "docker binary not found on PATH"
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=5,
            )
        except subprocess.TimeoutExpired:
            return "fail", "docker info timed out (daemon unreachable?)"
        if result.returncode != 0:
            stderr = result.stderr.strip().splitlines()[-1] if result.stderr else "unknown error"
            return "fail", f"docker info failed: {stderr}"
        return "pass", "docker daemon reachable"

    def _probe_unshare() -> tuple[str, str]:
        if shutil.which("unshare") is None:
            return "fail", "unshare binary not found on PATH"
        try:
            result = subprocess.run(
                ["unshare", "--user", "echo", "ok"],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=5,
            )
        except subprocess.TimeoutExpired:
            return "fail", "unshare timed out"
        if result.returncode != 0 or result.stdout.strip() != "ok":
            return "fail", f"unshare --user failed (rc={result.returncode}); user namespaces may be disabled"
        return "pass", "unshare --user works"

    if backend == "docker":
        return _probe_docker()
    if backend == "unshare":
        return _probe_unshare()
    if backend == "bare":
        return "warn", "backend=bare: no isolation (host-mode execution)"
    # auto / unknown: prefer docker, fall back to unshare
    docker_status, docker_detail = _probe_docker()
    if docker_status == "pass":
        return "pass", f"auto: {docker_detail}"
    unshare_status, unshare_detail = _probe_unshare()
    if unshare_status == "pass":
        return "pass", f"auto: docker unavailable, fell back to unshare ({unshare_detail})"
    return "fail", f"auto: docker ({docker_detail}) AND unshare ({unshare_detail}) both unavailable"


def _doctor_check_checkpoint_db(config: dict[str, Any]) -> tuple[str, str]:
    """Checkpoint DB path is writable AND the latest few rows deserialize.

    A silent fall-back to ``{}`` on corrupted blobs used to mask data loss
    (P1.6) — verifying a deserialize cycle here surfaces the corruption as
    a doctor warning instead of a half-resumed session.
    """
    import sqlite3
    persistence_cfg = config.get("persistence", {}) or {}
    db_path = persistence_cfg.get("db_path", "~/.harness/checkpoints.db")
    expanded = os.path.expanduser(db_path)
    parent = os.path.dirname(expanded) or "."
    try:
        os.makedirs(parent, exist_ok=True)
    except OSError as exc:
        return "fail", f"cannot create parent dir {parent}: {exc}"
    try:
        conn = sqlite3.connect(expanded, timeout=2)
        conn.execute("PRAGMA user_version")
    except sqlite3.Error as exc:
        return "fail", f"sqlite3 open failed for {expanded}: {exc}"

    # Best-effort deserialize check on the 5 most recent checkpoints. If the
    # `checkpoints` table doesn't exist yet (fresh DB), skip silently.
    try:
        from harness.storage import (
            CheckpointCorruptedError,
            _deserialize_checkpoint_blob,
        )
        rows = conn.execute(
            "SELECT thread_id, checkpoint FROM checkpoints "
            "ORDER BY ROWID DESC LIMIT 5"
        ).fetchall()
        corrupted: list[str] = []
        for thread_id, blob in rows:
            try:
                _deserialize_checkpoint_blob(blob, strict=True)
            except CheckpointCorruptedError:
                corrupted.append(thread_id)
        if corrupted:
            conn.close()
            unique = sorted(set(corrupted))
            preview = ", ".join(unique[:3]) + ("…" if len(unique) > 3 else "")
            return (
                "warn",
                f"writable: {expanded} — but {len(corrupted)} recent checkpoint(s) "
                f"failed to deserialize (threads: {preview}). Run "
                f"`teane purge --session-id <id>` to drop them.",
            )
    except sqlite3.OperationalError:
        # `checkpoints` table not yet created — fresh DB, nothing to validate.
        pass
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    return "pass", f"writable: {expanded}"


def _doctor_check_patcher_mode(config: dict[str, Any]) -> tuple[str, str]:
    """Surface the patcher's two behaviour flags so operators know which
    mode the harness is running in.

    B5 (``patcher.enforce_read_before_edit``) — when on, REPLACE/DELETE/
    INSERT blocks against files the LLM has not been shown this turn are
    rejected. Default on.

    B6 (``patcher.use_structured_tools``) — when on, providers that
    support native tool-use receive PATCH_TOOLS (plus the read-only
    retrieval tools) as ``tools=...`` instead of relying on the text DSL.
    Default ON; set patcher.use_structured_tools=false to force the legacy
    text DSL for providers/models where native tool-use misbehaves.
    """
    patcher = (config.get("patcher") or {})
    b5 = bool(patcher.get("enforce_read_before_edit", True))
    b6 = bool(patcher.get("use_structured_tools", True))
    b5_label = "read-before-edit ON" if b5 else "read-before-edit OFF"
    b6_label = (
        "native tool-use ON (experimental)" if b6
        else "native tool-use OFF — text DSL active"
    )
    return "pass", f"{b5_label}; {b6_label}"


def _doctor_check_global_config() -> tuple[str, str]:
    """The in-repo global config file at <teane_root>/config/config.json exists.

    Without it, discover_config falls back to harness/cli.json's empty-routing
    defaults and the first LLM dispatch will fail with no model configured.
    """
    path = _get_global_config_path()
    if not os.path.isfile(path):
        return "fail", (
            f"missing {path} — run scripts/setup.py or restore the file from git"
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return "fail", f"cannot parse {path}: {exc}"
    return "pass", f"found at {path}"


def _doctor_check_config(workspace_path: str) -> tuple[str, str]:
    """Canonical config loads and strictly validates without errors.

    Under the single-source-config contract there is no "warn" outcome:
    discover_config either passes or raises ConfigError. The doctor
    surfaces the full multi-line error message so the operator can
    correct every problem in one pass.
    """
    try:
        config = discover_config(workspace_path)
    except ConfigError as exc:
        return "fail", f"strict validation failed: {exc}"
    except Exception as exc:  # noqa: BLE001
        return "fail", f"discover_config raised: {exc}"
    section_count = sum(1 for k in config if not k.startswith("_"))
    return "pass", f"config parsed cleanly ({section_count} top-level sections)"


def _doctor_check_product_spec(
    config: dict[str, Any], workspace_path: str,
) -> tuple[str, str]:
    """Mandatory: `product_spec_dir` is a valid workspace-root folder
    name AND the folder exists at the workspace root with at least one
    spec file (.txt / .md / .pdf —
    see :data:`harness.spec_files.SPEC_FILE_EXTS`).

    The value must be a bare folder name (no path separators, no
    absolute paths, no `..`). The harness mandates the spec folder lives
    inside the workspace so the operator's product description is
    versioned alongside the code that implements it. The operator should
    hear about a misconfiguration in `teane doctor` before they try
    `teane run`.
    """
    spec_dirname = config.get("product_spec_dir")
    if spec_dirname is None:
        return (
            "fail",
            "product_spec_dir not set in config.json — add a top-level "
            "string key with the NAME of a workspace-root folder, e.g. "
            "\"product_spec_dir\": \"product_spec\"",
        )
    name_error = _validate_product_spec_dir_name(spec_dirname)
    if name_error is not None:
        return ("fail", f"product_spec_dir {name_error}")
    resolved = _resolve_product_spec_dir(workspace_path, spec_dirname)
    allowed_exts_str = ", ".join(SPEC_FILE_EXTS)
    if not os.path.isdir(resolved):
        return (
            "fail",
            f"`{spec_dirname.strip()}/` folder not found at workspace "
            f"root — expected at {resolved!r}. Create it and add one or "
            f"more spec files ({allowed_exts_str})",
        )
    spec_files = list_spec_files(resolved)
    if not spec_files:
        return (
            "fail",
            f"`{spec_dirname.strip()}/` exists but contains no spec "
            f"files ({allowed_exts_str}) — add at least one with the "
            "product specification",
        )
    return (
        "pass",
        f"`{spec_dirname.strip()}/` at workspace root contains "
        f"{len(spec_files)} spec file(s)",
    )


# Runtime dependencies declared in pyproject.toml [project.dependencies],
# as (import_name, pip_name) pairs. KEEP IN SYNC with pyproject — the test
# tests/test_doctor_dependencies.py::test_table_matches_pyproject asserts the
# two agree so drift fails CI. import_name is what Python imports (e.g.
# `yaml`); pip_name is what an operator installs (e.g. `pyyaml`).
_RUNTIME_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("langgraph", "langgraph"),
    ("langgraph.checkpoint.sqlite", "langgraph-checkpoint-sqlite"),
    ("aiofiles", "aiofiles"),
    ("tree_sitter", "tree-sitter"),
    ("tree_sitter_language_pack", "tree-sitter-language-pack"),
    ("httpx", "httpx"),
    ("typing_extensions", "typing-extensions"),
    ("yaml", "pyyaml"),
    ("pypdf", "pypdf"),
)


def _missing_runtime_dependencies() -> list[str]:
    """Return the pip names of every runtime dependency that fails to import
    (missing, or present-but-broken on this Python ABI), sorted and
    de-duplicated. Cheap: import-only, no side effects."""
    import importlib
    missing: list[str] = []
    for import_name, pip_name in _RUNTIME_DEPENDENCIES:
        try:
            importlib.import_module(import_name)
        except Exception:  # noqa: BLE001 — ImportError or a broken transitive import
            missing.append(pip_name)
    return sorted(set(missing))


def _doctor_check_dependencies() -> tuple[str, str]:
    """Every runtime dependency declared in pyproject.toml is importable.

    Without this, a missing or ABI-incompatible dependency surfaces as a raw
    ImportError traceback the first time a run reaches the code that imports
    it — mid-startup, after config / git / API-key work — the single most
    basic environment failure, reported the least clearly. This turns it into
    an upfront FAIL naming the package(s) and the fix. (Tree-sitter also gets
    a deeper grammar-load check, `tree-sitter`; this one is import-only across
    the whole dependency set and runs even when the config check fails.)
    """
    missing = _missing_runtime_dependencies()
    if missing:
        return "fail", (
            f"unimportable: {', '.join(missing)}. Install the harness and its "
            f"dependencies from the repo root — `pip install -e .` — or "
            f"`pip install {' '.join(missing)}`."
        )
    return "pass", (
        f"all {len(_RUNTIME_DEPENDENCIES)} runtime dependencies importable"
    )


def _preflight_dependency_guard() -> "int | None":
    """Fail fast at the top of cmd_run / cmd_resume when a runtime dependency
    is missing, instead of letting a raw ImportError escape mid-startup.
    Returns an exit code to propagate, or None when the environment is
    complete."""
    # Interpreter floor first. pip enforces requires-python at install
    # time, but an editable install upgraded via `git pull` never
    # re-checks — a 3.11-3.13 venv would crash later at uuid.uuid7().
    if sys.version_info < (3, 14):
        print(
            "[teane] Cannot start — Python "
            f"{sys.version_info.major}.{sys.version_info.minor} is below "
            "the required 3.14 (session ids use the stdlib uuid.uuid7, "
            "added in 3.14).\n"
            "        Recreate the venv with Python 3.14+ and reinstall:\n"
            "            pip install -e .",
            file=sys.stderr,
        )
        return 1
    missing = _missing_runtime_dependencies()
    if not missing:
        return None
    print(
        "[teane] Cannot start — missing runtime dependencies: "
        + ", ".join(missing) + ".\n"
        "        Install the harness and its dependencies from the repo "
        "root:\n"
        "            pip install -e .\n"
        "        Then re-run. `teane doctor` gives a full environment report.",
        file=sys.stderr,
    )
    return 1


def _doctor_check_tree_sitter() -> tuple[str, str]:
    """Tree-sitter and the language-pack catalogue are importable and at
    least one grammar loads + parses.

    The harness uses tree-sitter in two places — :class:`HybridPatcher`
    for AST-aware code modifications (``harness/patcher.py``) and
    :class:`DependencyGraph` for cross-file symbol extraction
    (``harness/impact.py``). When the pip package
    ``tree-sitter-language-pack`` is missing or its bundled grammars stop
    loading on a new Python ABI, both subsystems silently fall back to
    regex extraction — every teane run continues to work but loses
    structural awareness, and no operator-facing signal warns about it.
    This check surfaces that degradation BEFORE the next teane run.

    Returns:
      - ``"pass"`` when every supported grammar loads and parses a tiny
        snippet.
      - ``"warn"`` when between 1 and 2 grammars are unhealthy (the rest
        still provide AST; only the degraded languages drop to regex).
      - ``"fail"`` when ImportError fires on either package, or when
        every grammar is unhealthy.
    """
    try:
        import tree_sitter  # noqa: F401
        from tree_sitter_language_pack import get_language
    except ImportError as exc:
        return "fail", (
            f"import failed: {exc}. Run `pip install -e .` from the repo "
            f"root (tree-sitter + tree-sitter-language-pack are declared "
            f"in pyproject.toml)."
        )

    # Iterate the same grammar map the harness's tree-sitter consumers
    # use (DependencyGraph._GRAMMAR_NAMES). Keeps this check in sync
    # with the actual supported-language set without duplicating the list.
    from harness.impact import DependencyGraph
    grammar_names_map = DependencyGraph._GRAMMAR_NAMES
    from tree_sitter import Parser

    # Minimal syntactically-valid snippet per language. The body doesn't
    # matter — we only need the parser to construct a tree without
    # raising. Anything more elaborate would risk false negatives on
    # grammar dialects.
    snippets: dict[str, str] = {
        "python": "x = 1\n",
        "javascript": "var x = 1;\n",
        "tsx": "const x: number = 1;\n",
        "typescript": "const x: number = 1;\n",
        "java": "class X { }\n",
        "go": "package main\n",
        "rust": "fn main() {}\n",
        "dart": "void main() {}\n",
    }

    healthy: list[str] = []
    degraded: list[str] = []
    # _GRAMMAR_NAMES maps harness language tags → grammar names; multiple
    # tags can share a grammar (jsx and javascript both → "javascript"),
    # so dedupe via set() to avoid double-probing.
    for grammar_name in sorted(set(grammar_names_map.values())):
        try:
            language = get_language(grammar_name)
            parser = Parser()
            parser.language = language
            sample = snippets.get(grammar_name, "x\n")
            parser.parse(sample.encode("utf-8"))
            healthy.append(grammar_name)
        except Exception:  # noqa: BLE001
            degraded.append(grammar_name)

    if not healthy:
        return "fail", (
            f"no grammars loadable. Degraded: {degraded}. The harness "
            f"will fall back to regex extraction across the board; "
            f"reinstall tree-sitter-language-pack."
        )
    if degraded:
        if len(degraded) <= 2:
            return "warn", (
                f"AST available for {len(healthy)} grammar(s); degraded: "
                f"{degraded} fall back to regex extraction."
            )
        return "fail", (
            f"AST degraded for {len(degraded)} grammar(s): {degraded}. "
            f"Only {healthy} remain healthy; reinstall "
            f"tree-sitter-language-pack."
        )
    return "pass", f"AST available for {len(healthy)} grammar(s): {healthy}"


# ---------------------------------------------------------------------------
# `teane web start` / `teane web stop` — marker-driven lifecycle
# ---------------------------------------------------------------------------
#
# A single marker file at ~/.harness/web.lock records the live web
# server's pid, host, port, mode (foreground / background), log path,
# and start timestamp. It's the single source of truth `teane web
# stop` reads to find and stop the process. The marker is written
# atomically (tmp + os.replace) so a crash mid-write doesn't leave a
# half-baked file behind, and is removed both during clean shutdown
# (signal handler / finally block in cmd_web_start) and at the top of
# `teane web stop`. A stale marker (pid no longer alive) is treated
# as no-server-running and silently cleaned up on the next start/stop.

_WEB_MARKER_FILENAME = "web.lock"
_WEB_LOG_FILENAME = "web.log"


def _web_marker_path() -> str:
    return os.path.join(os.path.expanduser("~/.harness"), _WEB_MARKER_FILENAME)


def _web_log_path() -> str:
    return os.path.join(os.path.expanduser("~/.harness"), _WEB_LOG_FILENAME)


def _read_web_marker() -> Optional[dict[str, Any]]:
    path = _web_marker_path()
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _write_web_marker(data: dict[str, Any]) -> None:
    path = _web_marker_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except OSError:
        # Best-effort cleanup of the half-written tmp file.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _delete_web_marker() -> None:
    """Idempotent: missing-file is success, not an error."""
    try:
        os.unlink(_web_marker_path())
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("[web] could not remove marker %s: %s", _web_marker_path(), exc)


def _is_pid_alive(pid: int) -> bool:
    """True if a process with this pid currently exists on the system.
    Signal 0 is the POSIX 'check existence' probe — it doesn't actually
    deliver a signal."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by someone else — still alive.
        return True
    except OSError:
        return False


def _utc_iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def cmd_web_start(args: argparse.Namespace) -> int:
    """``teane web start [--host H] [--port N] [--background yes/no]``

    Writes ``~/.harness/web.lock`` so ``teane web stop`` can find the
    process later. Refuses to start a second instance when the marker
    points at a live pid — the operator must stop the running one
    first. A stale marker (pid dead) is treated as no-server-running
    and silently overwritten.
    """
    # === 1. Single-instance gate ============================================
    existing = _read_web_marker()
    if existing is not None:
        ex_pid = existing.get("pid")
        if isinstance(ex_pid, int) and _is_pid_alive(ex_pid):
            host = existing.get("host", "?")
            port = existing.get("port", "?")
            mode = existing.get("mode", "?")
            log_path = existing.get("log_path") or "(none — foreground mode)"
            print(
                f"error: a teane web instance is already running "
                f"(pid {ex_pid}, http://{host}:{port}, mode={mode}).\n"
                f"  marker: {_web_marker_path()}\n"
                f"  log:    {log_path}\n"
                f"  Run 'teane web stop' first, then start a new instance.",
                file=sys.stderr,
            )
            return 1
        # Stale marker — the prior process crashed or was killed
        # without cleanup. Drop it and proceed with the new start.
        logger.info(
            "[web] stale marker found (pid %r not alive); removing.", ex_pid,
        )
        _delete_web_marker()

    # === 2. Background dispatch =============================================
    background = (str(getattr(args, "background", "no")).lower() == "yes")
    host = str(getattr(args, "host", "127.0.0.1") or "127.0.0.1")
    port = int(getattr(args, "port", 9000) or 9000)

    if background:
        return _web_start_background(host=host, port=port)
    return _web_start_foreground(
        host=host, port=port,
        print_mobile_url=bool(getattr(args, "print_mobile_url", False)),
    )


def _web_start_foreground(*, host: str, port: int, print_mobile_url: bool = False) -> int:
    """Run the server in the current process. Writes the marker with
    our own pid, installs a SIGTERM handler that triggers clean
    shutdown, blocks until the server exits, then releases the socket
    and removes the marker."""
    import signal as _signal
    import threading as _threading

    workspace_path = os.getcwd()
    try:
        config = discover_config(workspace_path)
    except ConfigError:
        config = {}

    from harness.dashboard import DashboardConfig, start_server
    dash_cfg = DashboardConfig.from_config(config)
    if not dash_cfg.enabled:
        logger.warning(
            "[web] dashboard.enabled is false in config; running anyway "
            "because the operator launched the subcommand directly."
        )
    # CLI flags always win over config — the operator typed them.
    dash_cfg.host = host
    dash_cfg.port = port

    server_handle = None
    try:
        server_handle = start_server(dash_cfg, blocking=False)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(
            f"error: cannot bind {dash_cfg.host}:{dash_cfg.port}: {exc}",
            file=sys.stderr,
        )
        return 1
    assert server_handle is not None

    # Write the marker AFTER the listening socket is bound — that way
    # if bind fails we don't leave a misleading marker on disk. From
    # this point on EVERYTHING is wrapped in a try/finally so a Ctrl-C
    # or any exception still releases the socket and removes the
    # marker — no stale lock files, no half-open ports.
    _write_web_marker({
        "pid": os.getpid(),
        "host": dash_cfg.host,
        "port": dash_cfg.port,
        "mode": "foreground",
        "log_path": "",
        "started_at": _utc_iso_now(),
    })

    shutdown_evt = _threading.Event()

    try:
        # Signal handler runs in a separate thread so it can call
        # server.shutdown() — calling shutdown() from the same thread
        # that owns serve_forever would deadlock. server_close() (run
        # in finally) releases the socket and the marker delete is
        # the last cleanup step.
        def _on_signal(signum, _frame):
            if shutdown_evt.is_set():
                return  # second Ctrl-C / SIGTERM is a no-op
            shutdown_evt.set()
            name = {2: "SIGINT (Ctrl-C)", 15: "SIGTERM"}.get(signum, f"signal {signum}")
            # Print to stderr so the operator sees feedback even when
            # the logger threshold hides INFO records.
            print(f"\n[web] {name} received; shutting down cleanly...",
                  file=sys.stderr, flush=True)
            _threading.Thread(
                target=server_handle.server.shutdown,
                name="harness-web-shutdown", daemon=True,
            ).start()

        # Catch every signal that's *catchable* and means "please stop":
        #   SIGTERM — `teane web stop`, `kill <pid>`, init shutdown
        #   SIGINT  — Ctrl-C in the terminal
        #   SIGHUP  — controlling terminal closed (parent shell exit)
        #   SIGQUIT — Ctrl-\ on most terminals
        # SIGKILL (kill -9) and SIGSTOP cannot be caught by any process —
        # the OS terminates immediately. The stale marker left behind
        # is auto-cleaned on the next `teane web start`.
        for sig_name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT"):
            sig = getattr(_signal, sig_name, None)
            if sig is None:
                continue   # Windows / platform doesn't define this one
            try:
                _signal.signal(sig, _on_signal)
            except (ValueError, OSError):
                # signal.signal() can raise inside non-main threads or
                # on signals the kernel doesn't allow handlers for.
                # Best-effort — at least SIGTERM/SIGINT will always work.
                pass

        # Belt-and-braces: register an atexit hook so even an uncaught
        # exception path (or interpreter shutdown without our signal
        # handler firing) still releases the socket + removes the
        # marker. atexit doesn't run on SIGKILL — nothing does.
        import atexit as _atexit
        _our_pid = os.getpid()
        def _atexit_cleanup() -> None:
            try:
                server_handle.server.shutdown()
            except Exception:  # noqa: BLE001
                pass
            try:
                server_handle.server.server_close()
            except Exception:  # noqa: BLE001
                pass
            # Only delete the marker if it still points at OUR pid —
            # don't clobber a marker a subsequent `web start` may have
            # written if our process lingered after the marker was
            # already cleaned.
            marker = _read_web_marker()
            if marker and marker.get("pid") == _our_pid:
                _delete_web_marker()
        _atexit.register(_atexit_cleanup)

        print(
            f"[web] listening on http://{dash_cfg.host}:{dash_cfg.port}/  "
            f"(pid {os.getpid()})\n"
            f"[web] marker: {_web_marker_path()}\n"
            f"[web] stop with: teane web stop  (or Ctrl-C)",
        )

        # Phase 7: --print-mobile-url. Emits a bookmarkable URL that a
        # phone can hit once; the server exchanges the token for a
        # cookie and strips the query on 302 so the token doesn't
        # linger in browser history. No-op when auth is disabled — the
        # mobile view is already reachable directly without a token.
        if print_mobile_url:
            token = None
            try:
                from harness.dashboard import resolve_expected_token
                token = resolve_expected_token(dash_cfg)
            except Exception:  # noqa: BLE001
                token = None
            if token:
                print(
                    f"[web] mobile URL (one-shot token): "
                    f"http://{dash_cfg.host}:{dash_cfg.port}/m/<session-id>?t={token}\n"
                    f"[web]   → open on your phone once; the token converts to "
                    f"a cookie and is stripped from the URL."
                )
            else:
                print(
                    "[web] mobile URL: auth is disabled, so "
                    f"http://{dash_cfg.host}:{dash_cfg.port}/m/<session-id> "
                    "works directly without a token."
                )

        try:
            # Block until the server thread exits. The signal handler
            # spawns the shutdown thread which causes serve_forever to
            # return, the server thread exits, and join() unblocks.
            server_handle.thread.join()
        except KeyboardInterrupt:
            # Belt-and-braces fallback for the unlikely case where a
            # Ctrl-C bypassed our SIGINT handler (e.g. delivered before
            # signal.signal() ran, or from a parent shell that re-raises).
            print("\n[web] Ctrl-C received; shutting down cleanly...",
                  file=sys.stderr, flush=True)
            try:
                server_handle.server.shutdown()
            except Exception:  # noqa: BLE001
                pass
            try:
                server_handle.thread.join(timeout=5.0)
            except Exception:  # noqa: BLE001
                pass
    finally:
        # Release every resource the server is holding so the process
        # exits clean. Order matters: shut down the server (in case
        # the signal handler didn't get to), close the listening
        # socket (so the next start can rebind immediately), and
        # remove the marker file last. The atexit hook above is a
        # safety net for paths that bypass this finally — but in the
        # normal flow this block runs first and atexit's reads find
        # the marker already gone (so it no-ops).
        try:
            server_handle.server.shutdown()
        except Exception:  # noqa: BLE001
            pass
        try:
            server_handle.server.server_close()
        except Exception:  # noqa: BLE001
            pass
        # Worker threads spawned by ThreadingMixIn are daemon threads
        # (daemon_threads=True on _ThreadingServer) so they die with
        # this process — no explicit join needed.
        _delete_web_marker()
        print("[web] stopped — marker removed, socket released.",
              file=sys.stderr, flush=True)
    return 0


def _web_start_background(*, host: str, port: int) -> int:
    """Re-exec ourselves in foreground mode, fully detached, with
    stdout/stderr redirected to ~/.harness/web.log. The child writes
    the marker with its own pid; we wait briefly to confirm startup
    succeeded and then rewrite the marker to flag mode='background'
    and record the log path.
    """
    import time as _time

    log_path = _web_log_path()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    try:
        log_file = open(log_path, "ab")
    except OSError as exc:
        print(f"error: cannot open log file {log_path}: {exc}", file=sys.stderr)
        return 1

    # Re-exec self in foreground mode. POSIX start_new_session / Windows
    # CREATE_NEW_PROCESS_GROUP both detach the child from this terminal's
    # session/console so it survives `exit`.
    argv = [
        sys.executable, "-m", "harness.cli", "web", "start",
        "--host", host, "--port", str(port), "--background", "no",
    ]
    try:
        proc = subprocess.Popen(
            argv,
            stdout=log_file, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            **_platform.new_process_group_kwargs(),
            cwd=os.getcwd(),
        )
    except OSError as exc:
        print(f"error: could not spawn background process: {exc}", file=sys.stderr)
        try:
            log_file.close()
        except OSError:
            pass
        return 1
    finally:
        # Parent doesn't need the log file handle once the child owns it.
        try:
            log_file.close()
        except OSError:
            pass

    # Wait for the child to write its marker (means the listener is
    # bound) — or for the child to exit prematurely (bind failure,
    # config error, etc.). Either way bail within 5s.
    deadline = _time.time() + 5.0
    while _time.time() < deadline:
        if proc.poll() is not None:
            # Child died early; surface what we can.
            print(
                f"error: background process exited early (code {proc.returncode}).\n"
                f"  see {log_path} for details.",
                file=sys.stderr,
            )
            return 1
        marker = _read_web_marker()
        if marker and marker.get("pid") == proc.pid:
            break
        _time.sleep(0.1)
    else:
        # No marker after the timeout but child still alive — odd, but
        # don't fail the start; print a warning.
        print(
            f"warning: background process started (pid {proc.pid}) but "
            f"didn't confirm via marker file within 5s. "
            f"See {log_path} for details.",
            file=sys.stderr,
        )

    # Annotate the marker with the background flag + log path so
    # operators (and `web stop`) can find the logs later.
    marker = _read_web_marker() or {}
    marker["mode"] = "background"
    marker["log_path"] = log_path
    try:
        _write_web_marker(marker)
    except OSError as exc:
        logger.warning("[web] could not annotate marker with background info: %s", exc)

    print(
        f"[web] started in background (pid {proc.pid}, http://{host}:{port}/).\n"
        f"[web] logs:   {log_path}\n"
        f"[web] marker: {_web_marker_path()}\n"
        f"[web] stop with: teane web stop",
    )
    return 0


def cmd_web_stop(args: argparse.Namespace) -> int:
    """``teane web stop`` — read the marker, delete it, signal the
    pid, and confirm the process exited. Idempotent: a missing marker
    is reported as "no server running" with exit code 0 so scripts can
    safely call it twice."""
    import signal as _signal
    import time as _time

    marker = _read_web_marker()
    if marker is None:
        print(
            f"[web] no server running (no marker at {_web_marker_path()}).",
        )
        return 0

    pid = marker.get("pid")
    host = marker.get("host", "?")
    port = marker.get("port", "?")
    if not isinstance(pid, int) or not _is_pid_alive(pid):
        print(
            f"[web] marker is stale (pid {pid!r} not alive); cleaning up.",
        )
        _delete_web_marker()
        return 0

    # Delete the marker FIRST so a concurrent `web start` doesn't
    # block on a marker that's about to disappear and so a second
    # `web stop` becomes a clean no-op.
    _delete_web_marker()

    # SIGTERM for graceful shutdown — the server's signal handler
    # calls server.shutdown() + server_close() so the listening socket
    # is released, then the process exits.
    try:
        os.kill(pid, _signal.SIGTERM)
    except ProcessLookupError:
        print(f"[web] pid {pid} already gone.")
        return 0
    except OSError as exc:
        print(f"error: could not signal pid {pid}: {exc}", file=sys.stderr)
        return 1

    print(f"[web] sent SIGTERM to pid {pid} (http://{host}:{port}/). waiting for clean exit...")

    deadline = _time.time() + 5.0
    while _time.time() < deadline:
        if not _is_pid_alive(pid):
            print(f"[web] stopped (pid {pid}).")
            return 0
        _time.sleep(0.1)

    # Stubborn — escalate to SIGKILL. Windows lacks SIGKILL so fall back
    # to SIGTERM, which Win32 maps to TerminateProcess (a hard kill).
    kill_sig = getattr(_signal, "SIGKILL", _signal.SIGTERM)
    print(
        f"warning: pid {pid} didn't exit within 5s; sending {kill_sig.name}.",
        file=sys.stderr,
    )
    try:
        os.kill(pid, kill_sig)
    except OSError:
        pass
    deadline = _time.time() + 2.0
    while _time.time() < deadline:
        if not _is_pid_alive(pid):
            print(f"[web] forcibly killed (pid {pid}).")
            return 0
        _time.sleep(0.1)

    print(
        f"error: pid {pid} still alive after {kill_sig.name} — manual intervention needed.",
        file=sys.stderr,
    )
    return 1


def _resolve_schedule_config(args: argparse.Namespace) -> "Any":
    """Load the schedule section from the canonical config. Returns a
    ScheduleConfig; raises ConfigError to the outer handler when the
    config itself doesn't load (subcommand exits with code 2)."""
    workspace_path = (
        os.path.abspath(args.workspace) if getattr(args, "workspace", None)
        else os.getcwd()
    )
    try:
        config = discover_config(workspace_path)
    except ConfigError:
        config = {}
    from harness.schedule import ScheduleConfig
    return ScheduleConfig.from_config(config)


async def cmd_schedule_run(args: argparse.Namespace) -> int:
    cfg = _resolve_schedule_config(args)
    if not cfg.enabled:
        print(
            "error: schedule.enabled is false. Flip it on in config.json "
            "before starting the daemon.",
            file=sys.stderr,
        )
        return 1
    if not cfg.jobs:
        print("error: schedule.jobs is empty — nothing to run.", file=sys.stderr)
        return 1
    from harness.schedule import ScheduleDaemon
    daemon = ScheduleDaemon(cfg)
    print(f"teane schedule — {len(cfg.jobs)} job(s), tick={cfg.tick_seconds}s. Ctrl-C to stop.")
    return await daemon.run_forever()


def cmd_schedule_list(args: argparse.Namespace) -> int:
    cfg = _resolve_schedule_config(args)
    from harness.schedule import (
        build_run_command, last_run_for_job, next_run,
    )
    from datetime import datetime as _dt, timezone as _tz
    now = _dt.now(_tz.utc)
    if not cfg.jobs:
        print("No jobs configured. Add entries under schedule.jobs in config.json.")
        return 0
    print(f"{'NAME':<24} {'ENABLED':<8} {'SCHEDULE':<24} {'NEXT (UTC)':<20} {'LAST EXIT':<10}")
    for job in cfg.jobs:
        last = last_run_for_job(cfg, job.name)
        last_started = None
        last_exit = "-"
        if last:
            if last.get("exit_code") is not None:
                last_exit = str(last["exit_code"])
            try:
                last_started = _dt.fromisoformat(last["started_at"])
                if last_started.tzinfo is None:
                    last_started = last_started.replace(tzinfo=_tz.utc)
            except (TypeError, ValueError):
                pass
        nxt = next_run(job.schedule, after=now, last_started=last_started)
        print(
            f"{job.name[:24]:<24} {('yes' if job.enabled else 'no'):<8} "
            f"{job.schedule.raw[:24]:<24} "
            f"{nxt.strftime('%Y-%m-%d %H:%M'):<20} {last_exit:<10}"
        )
        print(f"  cmd: {' '.join(build_run_command(cfg, job))}")
    return 0


def cmd_schedule_validate(args: argparse.Namespace) -> int:
    """Parse the section, print problems, exit non-zero on any."""
    workspace_path = (
        os.path.abspath(args.workspace) if getattr(args, "workspace", None)
        else os.getcwd()
    )
    try:
        config = discover_config(workspace_path)
    except ConfigError as exc:
        print(f"[harness] {exc}", file=sys.stderr)
        return 2
    raw_jobs = ((config.get("schedule") or {}).get("jobs") or [])
    problems = 0
    from harness.schedule import parse_schedule
    for i, raw in enumerate(raw_jobs):
        if not isinstance(raw, dict):
            print(f"  jobs[{i}]: not an object — skipped.")
            problems += 1
            continue
        name = raw.get("name") or f"(unnamed @ index {i})"
        if not raw.get("workspace"):
            print(f"  {name}: missing 'workspace'.")
            problems += 1
        if not raw.get("schedule"):
            print(f"  {name}: missing 'schedule'.")
            problems += 1
            continue
        try:
            parse_schedule(str(raw.get("schedule")))
            print(f"  {name}: OK — schedule {raw.get('schedule')!r}")
        except ValueError as exc:
            print(f"  {name}: {exc}")
            problems += 1
    if problems:
        print(f"\n{problems} problem(s) found.")
        return 1
    print(f"\nAll {len(raw_jobs)} job(s) validate cleanly.")
    return 0


async def cmd_schedule_once(args: argparse.Namespace) -> int:
    cfg = _resolve_schedule_config(args)
    job = next((j for j in cfg.jobs if j.name == args.name), None)
    if job is None:
        print(
            f"error: no job named {args.name!r}. Known jobs: "
            f"{[j.name for j in cfg.jobs]}",
            file=sys.stderr,
        )
        return 1
    from harness.schedule import execute_job_once
    result = await execute_job_once(cfg, job)
    print(
        f"{result['job_name']}: exit={result['exit_code']} "
        f"duration={result['duration_sec']:.1f}s log={result['log_path']}"
    )
    return 0 if result["exit_code"] == 0 else 1


def cmd_schedule_history(args: argparse.Namespace) -> int:
    cfg = _resolve_schedule_config(args)
    from harness.schedule import history_for_job
    job_names = [args.job] if args.job else [j.name for j in cfg.jobs]
    if not job_names:
        print("No jobs configured.")
        return 0
    for name in job_names:
        rows = history_for_job(cfg, name, limit=args.limit)
        if not rows:
            print(f"{name}: no recorded runs.")
            continue
        print(f"\n{name}:")
        print(f"  {'STARTED (UTC)':<20} {'EXIT':<5} {'DUR (s)':<8} LOG")
        for r in rows:
            started = r['started_at'] or '-'
            ec = r['exit_code']
            dur = r['duration_sec']
            print(
                f"  {started[:19]:<20} "
                f"{(str(ec) if ec is not None else '?'):<5} "
                f"{(f'{dur:.1f}' if dur is not None else '?'):<8} "
                f"{r['log_path'] or '-'}"
            )
    return 0


async def cmd_chat(args: argparse.Namespace) -> int:
    """``teane chat`` — interactive refinement REPL (#8).

    Builds the same gateway / redactor / skill registry that ``cmd_run``
    uses, then hands control to :func:`harness.chat.run_chat`. The
    workspace lock is acquired so a concurrent ``teane run`` against
    the same workspace can't corrupt patches.
    """
    workspace_path = (
        os.path.abspath(args.workspace) if getattr(args, "workspace", None)
        else os.getcwd()
    )
    if not os.path.isdir(workspace_path):
        print(f"error: workspace path does not exist: {workspace_path}", file=sys.stderr)
        return 1
    try:
        config = discover_config(workspace_path)
    except ConfigError as exc:
        print(f"[harness] {exc}", file=sys.stderr)
        return 2

    lock_handle = _acquire_workspace_lock(workspace_path, force=False)
    if lock_handle is False:
        return 1

    # Gateway + skill registry + (optional) MCP pool — same wiring as cmd_run.
    from harness.gateway import create_gateway_from_config
    from harness.graph import set_gateway
    gateway = create_gateway_from_config(config)
    set_gateway(gateway)
    try:
        from harness.skills import register_builtin_skills
        register_builtin_skills(config=config)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[cli:chat] skill registration skipped: %s", exc)
    try:
        from harness.redactor import create_redactor_from_config
        create_redactor_from_config(config)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[cli:chat] redactor init skipped: %s", exc)
    pool = await _maybe_start_mcp_pool(config, workspace_path=workspace_path)  # noqa: F841

    budget = (
        float(args.budget) if getattr(args, "budget", None) is not None
        else float((config.get("token_budget") or {}).get("hard_cap_usd", 2.00))
    )
    try:
        from harness.chat import run_chat
        return await run_chat(
            workspace_path=workspace_path,
            gateway=gateway,
            config=config,
            initial_budget_usd=budget,
        )
    finally:
        await gateway.close()
        await _drain_mcp_pools()


def _resolve_repo_index_config(workspace_path: str) -> "Any":
    """Build a RepoIndexConfig from the workspace's config, with safe
    defaults when the config file is missing or invalid."""
    try:
        config = discover_config(workspace_path)
    except ConfigError:
        config = {}
    from harness.repo_index import RepoIndexConfig
    return RepoIndexConfig.from_config(config)


def cmd_index_build(args: argparse.Namespace) -> int:
    """``teane index build`` — (re)build the workspace's repo index."""
    workspace_path = (
        os.path.abspath(args.workspace) if getattr(args, "workspace", None)
        else os.getcwd()
    )
    if not os.path.isdir(workspace_path):
        print(f"error: workspace path does not exist: {workspace_path}", file=sys.stderr)
        return 1
    cfg = _resolve_repo_index_config(workspace_path)
    from harness.repo_index import build_index
    print(f"Building repo index for {workspace_path} ...")
    print(f"  backend: {cfg.backend}")
    print(f"  chunk_lines={cfg.chunk_lines}, overlap={cfg.chunk_overlap}")
    try:
        stats = build_index(workspace_path, cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Indexed {stats.chunk_count} chunk(s) across {stats.file_count} file(s) "
        f"(backend={stats.backend}, built_at={stats.built_at})."
    )
    return 0


def cmd_index_status(args: argparse.Namespace) -> int:
    workspace_path = (
        os.path.abspath(args.workspace) if getattr(args, "workspace", None)
        else os.getcwd()
    )
    cfg = _resolve_repo_index_config(workspace_path)
    from harness.repo_index import get_stats
    stats = get_stats(workspace_path, cfg)
    if stats is None:
        print(f"No index built yet for {workspace_path}.")
        print("Run `teane index build` to create one.")
        return 0
    print(f"Workspace: {workspace_path}")
    print(f"Workspace ID: {stats.workspace_id}")
    print(f"Backend: {stats.backend}")
    print(f"Chunks: {stats.chunk_count}")
    print(f"Files: {stats.file_count}")
    print(f"Built: {stats.built_at}")
    return 0


def cmd_index_clear(args: argparse.Namespace) -> int:
    workspace_path = (
        os.path.abspath(args.workspace) if getattr(args, "workspace", None)
        else os.getcwd()
    )
    cfg = _resolve_repo_index_config(workspace_path)
    from harness.repo_index import clear_index
    deleted = clear_index(workspace_path, cfg)
    print(f"Removed {deleted} indexed chunk(s) for {workspace_path}.")
    return 0


def cmd_gh_issue(args: argparse.Namespace) -> int:
    """Pull a GitHub issue into ``change_requests/CR-N-<slug>.txt``.

    Example::

        teane gh issue --repo akmontheweb/teane --number 42

    Subsequent ``teane run`` against the workspace picks up the new
    CR file via the existing change-request flow.
    """
    workspace_path = (
        os.path.abspath(args.workspace) if getattr(args, "workspace", None)
        else os.getcwd()
    )
    if not os.path.isdir(workspace_path):
        print(f"error: workspace path does not exist: {workspace_path}", file=sys.stderr)
        return 1
    try:
        config = discover_config(workspace_path)
    except ConfigError:
        # The ingest subcommand only needs the github section + the
        # change-requests dir name; degrade gracefully when the full
        # config is broken so the operator can still ingest the issue
        # and fix the config separately.
        config = {}
    cr_dir_name = (config.get("change_requests_dir") or "change_requests")
    try:
        from harness.github_integration import ingest_issue_to_change_request
        path = ingest_issue_to_change_request(
            workspace_path, args.repo, args.number,
            change_requests_dir=str(cr_dir_name),
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {path}")
    print(
        "Next: run `teane run -w {} -p \"fix CR\" --new-build false` "
        "to process the new change request.".format(workspace_path)
    )
    return 0


def cmd_gh_pr_create(args: argparse.Namespace) -> int:
    """Open a PR from the workspace's current branch."""
    workspace_path = (
        os.path.abspath(args.workspace) if getattr(args, "workspace", None)
        else os.getcwd()
    )
    if not os.path.isdir(workspace_path):
        print(f"error: workspace path does not exist: {workspace_path}", file=sys.stderr)
        return 1
    try:
        config = discover_config(workspace_path)
    except ConfigError:
        config = {}
    try:
        from harness.github_integration import create_pr
        pr = create_pr(
            workspace_path,
            title=args.title,
            body=args.body or "",
            base=args.base,
            draft=getattr(args, "draft", False),
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(pr.url)
    return 0


def cmd_gh_pr_comment(args: argparse.Namespace) -> int:
    """Post a comment on an existing PR."""
    try:
        config = discover_config(os.getcwd())
    except ConfigError:
        config = {}
    try:
        from harness.github_integration import post_pr_comment
        post_pr_comment(args.repo, args.number, args.body, config=config)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Comment posted on {args.repo}#{args.number}")
    return 0


async def _doctor_check_mcp(
    config: dict[str, Any],
) -> list[tuple[str, tuple[str, str]]]:
    """Run a connectivity check against every configured MCP server.

    Returns one row per server. When ``mcp.enabled=false`` returns an
    empty list — the operator hasn't opted in, so there's nothing to
    report and the doctor table stays terse.

    Each server is started in isolation (so one bad server doesn't
    cascade) and the count of advertised tools is reported on success.
    Both the start AND a graceful shutdown happen here so the doctor
    leaves no zombie subprocesses.
    """
    try:
        from harness.mcp_client import (
            McpClientPool, McpPoolConfig, StdioMcpClient,
        )
    except Exception as exc:  # noqa: BLE001
        return [("mcp", ("warn", f"mcp_client import failed: {exc}"))]
    pool_cfg = McpPoolConfig.from_config(config)
    if not pool_cfg.enabled:
        return []
    if not pool_cfg.servers:
        return [(
            "mcp", ("warn", "enabled but no servers configured in mcp.servers"),
        )]
    rows: list[tuple[str, tuple[str, str]]] = []
    for server in pool_cfg.servers:
        if not server.name or not server.command:
            rows.append((
                f"mcp:{server.name or '?'}",
                ("fail", "malformed server entry (missing name or command)"),
            ))
            continue
        # Safety: refuse to start a filesystem server during doctor unless
        # the operator's already opted in. Doctor mirrors cmd_run.
        if (not pool_cfg.allow_local_filesystem_servers and
                McpClientPool._looks_like_filesystem_server(server)):
            rows.append((
                f"mcp:{server.name}",
                ("warn", (
                    "looks like a filesystem server — set "
                    "mcp.allow_local_filesystem_servers=true to enable."
                )),
            ))
            continue
        client = StdioMcpClient(
            server,
            timeout_seconds=pool_cfg.tool_call_timeout_seconds,
            extra_allowlist=pool_cfg.command_allowlist,
        )
        try:
            await client.start()
            tool_count = len(client.tools)
            tool_names = ", ".join(t.get("name", "?") for t in client.tools[:5])
            tail = " ..." if tool_count > 5 else ""
            rows.append((
                f"mcp:{server.name}",
                ("pass", f"{tool_count} tool(s): {tool_names}{tail}"),
            ))
        except ValueError as exc:
            # Command rejected by trust.validate_mcp_server_command.
            rows.append((
                f"mcp:{server.name}",
                ("fail", f"command rejected: {exc}"),
            ))
        except Exception as exc:  # noqa: BLE001
            rows.append((
                f"mcp:{server.name}", ("fail", f"start failed: {exc}"),
            ))
        finally:
            await client.shutdown()
    return rows


@dataclass(frozen=True)
class DoctorResult:
    """One row of the doctor checklist. Mirrors preflight.CheckResult so
    the web wizard (Phase 3) and any external caller (`teane doctor
    --json`) can render both check types with one schema."""

    name: str
    status: str  # pass / warn / fail / skip — same vocabulary as preflight
    detail: str


async def collect_doctor_results(
    workspace_path: str,
) -> list[DoctorResult]:
    """Run every doctor check and return the structured results.

    This is the shared entry point for both the human-readable CLI
    printer (:func:`render_doctor_human`) and the JSON emitter
    (:func:`render_doctor_json`) — as well as the web wizard's
    checklist card. All ordering, skipping, and status logic lives
    here so the two renderers stay in lockstep.
    """
    # Silence the chatty INFO logging from discover_config; we surface
    # the result via the explicit "config" check.
    logging.getLogger("harness.cli").setLevel(logging.ERROR)

    config_status, config_detail = _doctor_check_config(workspace_path)
    checks: list[tuple[str, tuple[str, str]]] = [
        ("config", (config_status, config_detail)),
        ("git repo", _doctor_check_git(workspace_path)),
        # Runtime-dependency import check runs regardless of config validity —
        # a missing dep is more fundamental than a config typo, and this is
        # the check that turns a raw mid-startup ImportError into a clean FAIL.
        ("dependencies", _doctor_check_dependencies()),
    ]

    if config_status == "pass":
        config = discover_config(workspace_path)
        api_keys_result = await _doctor_check_api_keys(config)
        checks.extend([
            ("product spec", _doctor_check_product_spec(config, workspace_path)),
            ("api keys (live)", api_keys_result),
            ("tree-sitter", _doctor_check_tree_sitter()),
            ("sandbox backend", _doctor_check_sandbox(config)),
            ("checkpoint db", _doctor_check_checkpoint_db(config)),
            ("patcher mode", _doctor_check_patcher_mode(config)),
        ])
        checks.append(("sandbox image", _doctor_check_builder_image(config)))
        checks.append(("ollama daemon", _doctor_check_ollama_daemon(config)))
        checks.extend(_doctor_check_external_tools(config, workspace_path))
        checks.extend(_doctor_check_lsp(config))
        checks.extend(_doctor_check_mcp_commands(config))
        checks.extend(_doctor_check_config_paths(config))
        checks.extend(_doctor_check_env_placeholders())
        checks.extend(_doctor_check_env_vars_from_config(config))
        checks.extend(await _doctor_check_mcp(config))
    else:
        skipped_detail = "skipped — fix the config check above first"
        checks.extend([
            ("product spec", ("skip", skipped_detail)),
            ("api keys (live)", ("skip", skipped_detail)),
            ("tree-sitter", ("skip", skipped_detail)),
            ("sandbox backend", ("skip", skipped_detail)),
            ("checkpoint db", ("skip", skipped_detail)),
            ("external tools", ("skip", skipped_detail)),
        ])

    return [DoctorResult(name=label, status=status, detail=detail)
            for label, (status, detail) in checks]


def render_doctor_human(
    results: list[DoctorResult],
    workspace_path: str,
    config_path: str,
) -> tuple[str, int]:
    """Render a doctor result list as the terminal-style report. Returns
    ``(text, exit_code)`` so the caller (:func:`cmd_doctor`) can print
    and exit without re-implementing the summary logic."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 72)
    lines.append(f"teane doctor — workspace: {workspace_path}")
    lines.append(f"canonical config: {config_path}")
    lines.append("=" * 72)
    for r in results:
        lines.append(_format_doctor_line(r.status, r.name, r.detail))
    lines.append("=" * 72)

    failures = [r.name for r in results if r.status == "fail"]
    warnings = [r.name for r in results if r.status == "warn"]
    skipped = [r.name for r in results if r.status == "skip"]
    exit_code = 0
    if failures:
        lines.append(f"FAIL: {len(failures)} check(s) failed: {', '.join(failures)}")
        if "config" in failures:
            lines.append(
                "Fix the config file at the path shown above and re-run "
                "`teane doctor` — the harness will not proceed with "
                "invalid configuration."
            )
        exit_code = 1
    elif warnings:
        lines.append(f"OK with warnings ({len(warnings)}): {', '.join(warnings)}")
    elif skipped:
        lines.append(f"PARTIAL: {len(skipped)} check(s) skipped.")
        exit_code = 1
    else:
        lines.append("OK: all checks passed.")
    return "\n".join(lines), exit_code


def render_doctor_json(
    results: list[DoctorResult],
    workspace_path: str,
    config_path: str,
) -> str:
    """Stable machine-readable shape mirroring preflight.render_json:
    ``{workspace, config_path, results: [...], summary: {...}}``."""
    counts = {"pass": 0, "warn": 0, "fail": 0, "skip": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    exit_code = 1 if counts["fail"] > 0 or counts["skip"] > 0 else 0
    payload = {
        "workspace": workspace_path,
        "config_path": config_path,
        "results": [asdict(r) for r in results],
        "summary": {
            "pass": counts["pass"],
            "warn": counts["warn"],
            "fail": counts["fail"],
            "skip": counts["skip"],
            "exit_code": exit_code,
        },
    }
    return json.dumps(payload, indent=2)


async def cmd_doctor(args: argparse.Namespace) -> int:
    """
    Execute the `teane doctor` subcommand.

    Runs healthchecks and prints a green/yellow/red summary — or, with
    ``--json``, emits the same result set as a JSON document for the
    web wizard (Phase 3) and any CI job. Under the single-source-config
    contract the very first check is "config" — if it fails, the
    harness can't load anything else, so every downstream check is
    marked "skipped" and the doctor returns non-zero.

    Exits 0 if every executed check passes (warn is non-blocking),
    non-zero on any failure or when config validation prevents the
    rest of the checks from running.

    Examples:
        teane doctor
        teane doctor -w /path/to/repo
        teane doctor --json          # machine-readable
    """
    workspace_path = os.path.abspath(args.workspace) if args.workspace else os.getcwd()
    results = await collect_doctor_results(workspace_path)
    config_path = _get_global_config_path()

    if getattr(args, "json", False):
        sys.stdout.write(render_doctor_json(results, workspace_path, config_path))
        sys.stdout.write("\n")
        summary_exit = 0
        if any(r.status == "fail" for r in results):
            summary_exit = 1
        elif any(r.status == "skip" for r in results):
            summary_exit = 1
        return summary_exit

    text, exit_code = render_doctor_human(results, workspace_path, config_path)
    print(text)
    return exit_code


def cmd_pre_flight(args: argparse.Namespace) -> int:
    """Execute the `teane pre-flight` subcommand.

    Standalone machine-readiness probe — no workspace, no config required.
    Auto-detects the host OS and runs an OS-appropriate probe set, then
    emits a coloured checklist or JSON. Exit 0 if no FAIL rows; exit 1
    if any required tool is missing.

    Examples:
        teane pre-flight
        teane pre-flight --quick           # skip live network probe
        teane pre-flight --json            # CI-friendly output
        teane pre-flight --platform windows  # verify Windows install docs from Linux
    """
    from harness import preflight as _preflight

    platform_override = (args.platform or "auto").lower()
    if platform_override == "auto":
        platform_override = None  # let _platform.is_*() decide

    results = _preflight.run_all(
        platform_override=platform_override,
        quick=bool(getattr(args, "quick", False)),
    )

    # The header should reflect the user's choice even though run_all
    # restored the real platform predicates after the probe pass.
    header_platform = platform_override

    if getattr(args, "json_dump", False):
        sys.stdout.write(
            _preflight.render_json(results, platform_name=header_platform) + "\n"
        )
    else:
        sys.stdout.write(_preflight.render_tty(
            results,
            no_color=bool(getattr(args, "no_color", False)),
            platform_name=header_platform,
        ))

    has_fail = any(r.status == _preflight.STATUS_FAIL for r in results)
    return 1 if has_fail else 0


async def cmd_purge(args: argparse.Namespace) -> int:
    """
    Execute the `teane purge` subcommand.

    Wipes all checkpoint data from the SQLite database.

    Examples:
        teane purge --session-id my-session
        teane purge --all
    """
    workspace_path = os.path.abspath(args.workspace) if args.workspace else os.getcwd()
    config = discover_config(workspace_path)
    persistence_cfg = config.get("persistence", {})
    db_path = persistence_cfg.get("db_path", "~/.harness/checkpoints.db")
    ttl_days = persistence_cfg.get("ttl_days", 30)

    from harness.storage import HarnessAsyncSqliteSaver, purge_checkpoints

    if args.all:
        print(
            "WARNING: This will delete ALL teane data permanently: "
            "checkpoints, story/feature/batch/defect state, repo index, "
            "and JSONL session logs — across every workspace."
        )
        from harness.hitl import get_channel as _get_channel
        confirmed = _get_channel().confirm("Type 'yes' to confirm full purge", default=False)
        if not confirmed:
            print("Purge cancelled.")
            return 0

        # 1) Checkpoint DB (writes + checkpoints).
        deleted = await purge_checkpoints(db_path)
        print(f"Purged {deleted} rows from the checkpoint database.")

        # 2) state.db — stories, features, batches, defects, requirements,
        #    ACs, test_runs, file_links, commits, link tables. Global.
        try:
            from harness import story_state as _story_state_mod
            state_counts = _story_state_mod.purge_state_db_all()
            state_total = sum(state_counts.values())
            print(f"Purged {state_total} rows from the story state database.")
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning("state.db purge failed: %s", exc)
            print(f"WARNING: state.db purge failed: {exc}")

        # 3) Repo index DB — meta + chunks across all workspaces. Honor
        #    the operator's ``repo_index.index_dir`` override so the
        #    purge hits the same DB the harness actually writes to.
        try:
            from harness.repo_index import (
                RepoIndexConfig as _RepoIndexConfig,
                purge_all as _purge_repo_all,
            )
            rcfg = _RepoIndexConfig.from_config(config)
            meta_n, chunk_n = _purge_repo_all(rcfg)
            print(
                f"Purged repo index ({meta_n} workspace entries, "
                f"{chunk_n} chunks)."
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning("repo_index purge failed: %s", exc)
            print(f"WARNING: repo_index purge failed: {exc}")

        # 4) JSONL session logs — every *.jsonl and rotated backup in
        #    the configured log dir. Directory itself is left in place.
        log_cfg = config.get("logging", {})
        log_dir = os.path.expanduser(log_cfg.get("log_dir", "~/.harness/logs"))
        removed_logs = 0
        if os.path.isdir(log_dir):
            import glob
            for pat in (
                os.path.join(log_dir, "*.jsonl"),
                os.path.join(log_dir, "*.jsonl.*"),
            ):
                for path in glob.glob(pat):
                    try:
                        os.remove(path)
                        removed_logs += 1
                    except OSError as exc:
                        logger.warning(
                            "Could not remove log file %s: %s", path, exc,
                        )
        print(f"Removed {removed_logs} JSONL log file(s) from {log_dir}.")
    elif args.session_id:
        checkpointer = await HarnessAsyncSqliteSaver.from_db_path(db_path=db_path, ttl_days=ttl_days)
        await checkpointer.adelete_thread(args.session_id)
        print(f"Purged all checkpoints for session '{args.session_id}'.")
        await checkpointer.conn.close()
        # P2.9: a GDPR / customer-deletion request needs the JSONL transcript
        # gone too — it may include redacted-but-not-eliminated prompt
        # excerpts. Best-effort: the live log file plus any rotated backups
        # (`<id>.jsonl.1`, `.jsonl.2`, ...).
        log_cfg = config.get("logging", {})
        log_dir = os.path.expanduser(log_cfg.get("log_dir", "~/.harness/logs"))
        removed = 0
        if os.path.isdir(log_dir):
            import glob
            patterns = [
                os.path.join(log_dir, f"{args.session_id}.jsonl"),
                os.path.join(log_dir, f"{args.session_id}.jsonl.*"),
            ]
            for pat in patterns:
                for path in glob.glob(pat):
                    try:
                        os.remove(path)
                        removed += 1
                    except OSError as exc:
                        logger.warning("Could not remove log file %s: %s", path, exc)
        if removed:
            print(f"Removed {removed} log file(s) for session '{args.session_id}'.")
    else:
        logger.error("Please specify --all to purge everything or --session-id to purge a specific session.")
        return 1

    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Execute ``teane audit`` — standalone v5 traceability audit.

    Runs the SQL-backed audit against the workspace's state.db rows
    (the same audit that fires at end-of-session) without spinning
    up a full graph run. Exit 0 when both gap sets are empty;
    exit 1 when ``has_failures()``.

    Useful as a CI check: after a build/patch lands, gate the
    follow-up step on whether traceability is clean. Honors
    ``--workspace`` to scope the audit to one project; defaults to
    the current directory's basename like every other subcommand.
    """
    workspace_path = (
        os.path.abspath(args.workspace) if args.workspace else os.getcwd()
    )
    if not os.path.isdir(workspace_path):
        print(
            f"[teane audit] Workspace path is not a directory: "
            f"{workspace_path}",
            file=sys.stderr,
        )
        return 2

    from harness.traceability import audit_workspace, format_report

    report = audit_workspace(workspace_path)
    if report is None:
        print(
            "[teane audit] No state.db rows for this workspace — "
            "nothing to audit. Run `teane build` or `teane patch` first.",
            file=sys.stderr,
        )
        return 0

    if not report.has_failures():
        print(
            f"[teane audit] Clean. "
            f"Requirements: {report.traced_reqs}/{report.total_reqs} "
            f"({report.req_coverage_pct:.0f}%). "
            f"Acceptance criteria: {report.verified_acs}/{report.total_acs} "
            f"({report.ac_coverage_pct:.0f}%)."
        )
        return 0

    rendered = format_report(report)
    if rendered:
        print(rendered)
    print(
        f"[teane audit] FAILED. "
        f"{len(report.untraced)} untraced requirement(s), "
        f"{len(report.untested_acs)} untested acceptance criteria.",
        file=sys.stderr,
    )
    return 1


async def cmd_cache_clear(args: argparse.Namespace) -> int:
    """Execute ``teane cache clear``.

    Enumerates harness-owned Docker volumes (those prefixed with
    ``harness-`` by default; configurable via ``sandbox.cache_volumes_prefix``)
    and removes them. With ``--session-id`` only the volumes scoped to that
    session are touched; otherwise every harness-owned volume is removed.

    Idempotent: a volume that has already been removed (or never existed)
    is treated as success, not an error.

    Examples:
        teane cache clear
        teane cache clear --session-id sess-abc123
        teane cache clear --yes  # skip the confirmation prompt
        teane cache clear --dry-run
    """
    workspace_path = os.path.abspath(args.workspace) if args.workspace else os.getcwd()
    try:
        config = discover_config(workspace_path)
    except Exception:  # noqa: BLE001 — `cache clear` must work without a workspace config
        config = {}

    sandbox_cfg = config.get("sandbox", {}) or {}
    prefix = sandbox_cfg.get("cache_volumes_prefix", "harness") + "-"
    docker_path = sandbox_cfg.get("docker_path", "docker")

    if not shutil.which(docker_path):
        print(
            f"[harness] `{docker_path}` not found on PATH. Nothing to clear "
            f"— cache volumes are a Docker-backend feature.",
            file=sys.stderr,
        )
        return 0

    try:
        listing = subprocess.run(
            [docker_path, "volume", "ls", "--format", "{{.Name}}"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        print(f"[harness] `docker volume ls` failed: {exc}", file=sys.stderr)
        return 1
    if listing.returncode != 0:
        print(
            f"[harness] `docker volume ls` exited {listing.returncode}: "
            f"{(listing.stderr or '').strip()[:200]}",
            file=sys.stderr,
        )
        return 1

    all_volumes = [v for v in (listing.stdout or "").splitlines() if v.strip()]
    candidates = [v for v in all_volumes if v.startswith(prefix)]
    if args.session_id:
        # session id is the trailing token of the volume name, separated by "-".
        sid_slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", args.session_id).strip("-")
        candidates = [v for v in candidates if v.endswith(f"-{sid_slug}")]

    if not candidates:
        scope = f"session '{args.session_id}'" if args.session_id else "all sessions"
        print(f"No teane cache volumes found for {scope}.")
        return 0

    print(f"Found {len(candidates)} teane cache volume(s):")
    for name in candidates:
        print(f"  {name}")
    if args.dry_run:
        print("(dry run — no volumes removed)")
        return 0
    if not args.yes:
        from harness.hitl import get_channel as _get_channel
        confirmed = _get_channel().confirm(
            f"Type 'yes' to remove {len(candidates)} volume(s)", default=False,
        )
        if not confirmed:
            print("Cache clear cancelled.")
            return 0

    removed, failed = 0, []
    for name in candidates:
        rm = subprocess.run(
            [docker_path, "volume", "rm", name],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30,
        )
        if rm.returncode == 0:
            removed += 1
        else:
            err = (rm.stderr or "").strip()
            # "in use" is a non-idempotent failure — surface it but keep going.
            failed.append((name, err[:160]))
    print(f"Removed {removed} volume(s).")
    if failed:
        print(f"Failed to remove {len(failed)} volume(s):", file=sys.stderr)
        for name, err in failed:
            print(f"  {name}: {err}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# 3b. Metrics Subcommand (P2.7)
# ---------------------------------------------------------------------------

# Default destination for machine-readable metrics outputs. Overridable
# via the metrics.metrics_dir config key at any layer (shipped defaults
# in cli.json, user-global ~/.harness/config.json, per-workspace
# .harness_config.json).
_DEFAULT_METRICS_DIR = "~/.harness/metrics"


async def cmd_metrics(args: argparse.Namespace) -> int:
    """Execute the `teane metrics` subcommand (P2.7).

    Aggregates per-session cost/usage from the JSONL logs and renders it
    as a human report (stdout), JSON dump, Prometheus exposition text,
    or roll-up table across every session in the log directory.

    Examples:
        teane metrics --session-id abc123
        teane metrics --all
        teane metrics --session-id abc123 --prometheus
        teane metrics --all --json --output -
    """
    from harness.metrics import (
        aggregate_session,
        format_human,
        format_prometheus,
        format_table,
        list_sessions,
        write_atomic,
    )

    workspace_path = os.path.abspath(args.workspace) if args.workspace else os.getcwd()
    config = discover_config(workspace_path)
    log_cfg = config.get("logging", {})
    metrics_cfg = config.get("metrics", {})
    token_budget_cfg = config.get("token_budget", {})

    log_dir = os.path.expanduser(log_cfg.get("log_dir", "~/.harness/logs"))
    metrics_dir = os.path.expanduser(
        metrics_cfg.get("metrics_dir", _DEFAULT_METRICS_DIR)
    )
    window_minutes = int(
        args.window_minutes
        if args.window_minutes is not None
        else metrics_cfg.get("burn_rate_window_minutes", 10)
    )
    window_minutes = max(1, min(1440, window_minutes))
    hard_cap_usd = float(token_budget_cfg.get("hard_cap_usd", 2.00))

    if args.all and args.session_id:
        logger.error("Pass --session-id OR --all, not both.")
        return 1
    if not args.all and not args.session_id:
        logger.error("Please specify --session-id <id> or --all.")
        return 1

    if args.session_id:
        target_sessions = [args.session_id]
    else:
        target_sessions = list_sessions(log_dir)
        if not target_sessions:
            logger.error(
                "No session logs found in %s. Run a session first, or set "
                "logging.log_dir if your logs live elsewhere.",
                log_dir,
            )
            return 1

    metrics_list = [
        aggregate_session(sid, log_dir, window_minutes=window_minutes)
        for sid in target_sessions
    ]

    # If we asked for a specific session and nothing was found on disk,
    # surface that as a non-zero exit so cron/automation catches it.
    if args.session_id and metrics_list[0].llm_call_count == 0 and not metrics_list[0].log_files:
        logger.error(
            "No log files found for session '%s' in %s.",
            args.session_id, log_dir,
        )
        return 1

    # Output routing. Human (default) → stdout. JSON / Prometheus go to
    # the configured metrics_dir unless --output-path overrides.
    if getattr(args, "json_dump", False):
        if args.session_id:
            body = json.dumps(metrics_list[0].to_jsonable(), indent=2) + "\n"
            default_name = f"{args.session_id}.json"
        else:
            body = json.dumps(
                {"sessions": [m.to_jsonable() for m in metrics_list]},
                indent=2,
            ) + "\n"
            default_name = "sessions.json"
        _emit_output(body, default_name, metrics_dir, getattr(args, "output_path", None), write_atomic)
    elif args.prometheus:
        body = format_prometheus(metrics_list, hard_cap_usd=hard_cap_usd)
        default_name = (
            f"{args.session_id}.prom" if args.session_id else "all.prom"
        )
        _emit_output(body, default_name, metrics_dir, getattr(args, "output_path", None), write_atomic)
    else:
        # Human-readable: always to stdout.
        if args.session_id:
            print(format_human(metrics_list[0], hard_cap_usd=hard_cap_usd))
        else:
            print(format_table(metrics_list, hard_cap_usd=hard_cap_usd))

    return 0


def _emit_output(
    body: str,
    default_name: str,
    metrics_dir: str,
    output_override: Optional[str],
    writer: Any,
) -> None:
    """Route a machine-readable payload to stdout or an atomic file.

    `output_override` semantics:
      - None: write to ``<metrics_dir>/<default_name>`` (the configured
        per-session/per-rollup file the operator can hand to a scraper).
      - "-": stream to stdout, no file written.
      - anything else: treat as an explicit absolute or relative file
        path and write there atomically.
    """
    if output_override == "-":
        sys.stdout.write(body)
        if not body.endswith("\n"):
            sys.stdout.write("\n")
        return
    dest = output_override or os.path.join(metrics_dir, default_name)
    writer(dest, body)
    print(f"Wrote {dest}")


# ---------------------------------------------------------------------------
# 4. Argument Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct the full CLI argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="teane",
        description=(
            "Teane — production-grade, model-agnostic LangGraph agent\n"
            "for autonomous code generation, sandboxed builds, and bulletproof persistence.\n\n"
            "Quick Start:\n"
            "  teane run -w /path/to/repo -p \"Your engineering task description\"\n"
            "  teane -h                       Show this help\n"
            "  teane --version                Print the installed teane version\n"
            "  teane run -h                   Show run subcommand help\n"
            "  teane status --all             List all checkpointed sessions\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  teane run -w ./myproject -p \"Add JWT authentication\"\n"
            "  teane run -w /path/to/repo -p \"Refactor logging\" --manifest notes.txt\n"
            "  teane resume --session-id abc123\n"
            "  teane status --session-id abc123\n"
            "  teane purge --all\n"
        ),
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"teane {_get_harness_version()}",
        help="Print the installed teane version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ----- shared parser helper -----
    # Used by every true|false flag on the run subcommand. Accepts the
    # common spellings ("true"/"false", "yes"/"no", "1"/"0", "on"/"off")
    # case-insensitively. Defined at module-scope inside build_parser so
    # every subparser that wants the same semantics can re-use it.
    def _bool_choice(value: str) -> bool:
        if isinstance(value, bool):
            return value
        v = str(value).strip().lower()
        if v in ("true", "1", "yes", "on"):
            return True
        if v in ("false", "0", "no", "off"):
            return False
        raise argparse.ArgumentTypeError(
            f"Expected true|false (or yes|no, 1|0), got {value!r}"
        )

    # ------------------------------------------------------------------
    # `teane build` / `teane patch` / `teane deploy` parsers share most
    # of their surface (workspace, prompt, verbose, allow-network, git,
    # HITL gates...). _add_runlike_common emits the shared block; the
    # caller then adds command-specific flags. The legacy `teane run`
    # subcommand has been removed — operators pick build/patch/deploy
    # based on intent.
    # ------------------------------------------------------------------
    def _add_runlike_common(p: argparse.ArgumentParser, *, want_prompt: bool = True) -> None:
        p.add_argument(
            "--workspace", "-w",
            default=None,
            help="Absolute or relative path to the target repository root.",
        )
        if want_prompt:
            p.add_argument(
                "--prompt", "-p",
                default=None,
                help=(
                    "Optional engineering task description. product_spec/ "
                    "(build) and change_requests/*.txt (patch) are the "
                    "authoritative source; -p is redundant when those exist."
                ),
            )
        p.add_argument(
            "--verbose", "-v",
            action="store_true",
            default=False,
            help="Enable debug-level logging.",
        )
        p.add_argument(
            "--allow-network",
            type=_bool_choice,
            default=True,
            metavar="true|false",
            dest="allow_network",
            help=(
                "Permit outbound network traffic in the sandbox. Defaults "
                "to true; pass --allow-network false to block."
            ),
        )
        p.add_argument(
            "--git",
            type=_bool_choice,
            default=False,
            metavar="true|false",
            help=(
                "Enable GitGuardian stash/patch-branch/rollback. True "
                "requires the workspace to be a git repo. Defaults to false."
            ),
        )
        p.add_argument(
            "--session-id",
            default=None,
            help="Human-readable session identifier. UUIDv4 if omitted.",
        )
        p.add_argument(
            "--thread-id",
            default=None,
            help="LangGraph thread ID for checkpoint lookups. Defaults to session-id.",
        )
        p.add_argument(
            "--force-lock",
            action="store_true",
            default=False,
            dest="force_lock",
            help=(
                "Bypass the workspace session lock. Use ONLY for stale-lock "
                "recovery after a crash; concurrent runs corrupt patches."
            ),
        )
        p.add_argument(
            "--log",
            default=None,
            dest="log_file",
            metavar="PATH",
            help=(
                "Optional file to receive a copy of the human-readable "
                "log stream (stderr format). Must resolve OUTSIDE the "
                "workspace — `teane build` wipes the workspace root at "
                "startup and would delete a log file placed inside it. "
                "Prefer this over shell redirection (`2> file.log`) when "
                "running build in a headless session so the harness can "
                "guard against the wipe collision."
            ),
        )

    # HITL gates shared by build / patch (deploy adds only --hitl-deployment).
    def _add_hitl_buildpatch(p: argparse.ArgumentParser) -> None:
        for flag, dest, help_text in (
            ("--hitl-requirement", "hitl_requirement",
             "Prompt the operator at the REQUIREMENTS gate. "
             "Falls back to config.json hitl.requirement, then to true."),
            ("--hitl-architecture", "hitl_architecture",
             "Prompt the operator at the ARCHITECTURE gate. "
             "Falls back to config.json hitl.architecture, then to true."),
            ("--hitl-repair", "hitl_repair",
             "Prompt at the repair-loop HITL menu when iteration limits trip. "
             "Falls back to config.json hitl.repair, then to true."),
            ("--hitl-layout-divergence", "hitl_layout_divergence",
             "Prompt when the on-disk layout drifts from SPEC_ARCHITECTURE.md "
             "workspace_layout. Falls back to config.json hitl.layout_divergence."),
        ):
            p.add_argument(
                flag, type=_bool_choice, default=None,
                metavar="true|false", dest=dest, help=help_text,
            )

    # --- `teane build` (greenfield, destructive) ---
    build_parser = subparsers.add_parser(
        "build",
        help=(
            "Greenfield build: wipe the workspace (preserving product_spec/ "
            "and .git/), then generate code from the spec."
        ),
    )
    _add_runlike_common(build_parser)
    build_parser.add_argument(
        "--spec-discovery",
        type=_bool_choice,
        default=False,
        metavar="true|false",
        dest="spec_discovery",
        help=(
            "When true, run BOTH the requirements and architecture "
            "discovery interviews before code generation. Defaults to false."
        ),
    )
    build_parser.add_argument(
        "--cd-discovery",
        type=_bool_choice,
        default=False,
        metavar="true|false",
        dest="cd_discovery",
        help=(
            "Run deployment discovery and write docs/DEPLOYMENT_DISCOVERY.md. "
            "Build does NOT deploy — that's `teane deploy`. Defaults to false."
        ),
    )
    build_parser.add_argument(
        "--agile",
        type=_bool_choice,
        default=None,
        metavar="true|false",
        dest="agile",
        help=(
            "Engage Agile-style story decomposition + per-story TDD. "
            "Falls back to config.json's top-level `agile` key, then to false. "
            "Per-knob tuning (batch_size, commit_on_story, repair_cap) lives "
            "in config.json's agile_defaults block."
        ),
    )
    build_parser.add_argument(
        "--best-of",
        type=int,
        default=None,
        metavar="N",
        dest="best_of",
        help=(
            "Run N independent solve trajectories in isolated git worktrees "
            "and apply the winner (trajectory-level best-of-N). Overrides "
            "best_of_n.enabled/n in config.json. Cost scales ~linearly with N."
        ),
    )
    build_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        dest="assume_yes",
        help=(
            "Skip the workspace-reset confirmation prompt. Build is always "
            "destructive; use -y in CI or when you're sure."
        ),
    )
    build_parser.add_argument(
        "--reuse-specs",
        type=_bool_choice,
        default=None,
        metavar="true|false",
        dest="reuse_specs",
        help=(
            "Override the docs/ reuse decision. `true` preserves an "
            "existing docs/SPEC_REQUIREMENTS.md across the --new-build "
            "reset (skip spec regeneration); `false` forces spec "
            "regeneration even when specs exist. Defaults to interactive "
            "prompt on a TTY, reuse-if-present when non-interactive."
        ),
    )
    _add_hitl_buildpatch(build_parser)

    # --- `teane patch` (brownfield, planner-reconciles) ---
    patch_parser = subparsers.add_parser(
        "patch",
        help=(
            "Incremental patch: read the existing code + specs + any "
            "change_requests/* files (.txt / .md / .pdf), reconcile against the spec."
        ),
    )
    _add_runlike_common(patch_parser)
    patch_parser.add_argument(
        "--spec-discovery",
        type=_bool_choice,
        default=False,
        metavar="true|false",
        dest="spec_discovery",
        help=(
            "When true, re-run the requirements + architecture discovery "
            "interviews to revise the specs before patching. Defaults to false."
        ),
    )
    patch_parser.add_argument(
        "--generate-specs",
        type=_bool_choice,
        default=None,
        metavar="true|false",
        dest="generate_specs",
        help=(
            "Reverse-engineer SPEC_REQUIREMENTS.md / SPEC_ARCHITECTURE.md "
            "from the existing codebase. Default (unset) = auto: generate "
            "only when both spec files are missing. true = always "
            "regenerate (overwrites after review). false = error if specs "
            "are missing."
        ),
    )
    patch_parser.add_argument(
        "--agile",
        type=_bool_choice,
        default=None,
        metavar="true|false",
        dest="agile",
        help=(
            "Engage Agile-style story decomposition + per-story TDD. "
            "Default (unset) = auto-detect from .teane/state.db (non-empty "
            "→ agile). true = force agile (decomposes into first story set "
            "on a flat workspace). false = force flat (logs a gap-marker "
            "row in state.db on agile workspaces)."
        ),
    )
    patch_parser.add_argument(
        "--cd-discovery",
        type=_bool_choice,
        default=False,
        metavar="true|false",
        dest="cd_discovery",
        help=(
            "Run deployment discovery and write docs/DEPLOYMENT_DISCOVERY.md. "
            "Patch does NOT deploy — that's `teane deploy`. Defaults to false."
        ),
    )
    patch_parser.add_argument(
        "--install-doc",
        type=_bool_choice,
        default=None,
        metavar="true|false",
        dest="install_doc",
        help=(
            "Update INSTALLATION.md at the end of a successful patch. "
            "Defaults: agile patches auto-enable so the end-of-session "
            "traceability audit runs; non-agile patches default to false "
            "(incremental changes rarely affect install steps). Pass "
            "true/false explicitly to override either default."
        ),
    )
    _add_hitl_buildpatch(patch_parser)

    # --- `teane deploy` (artifacts + dev container + sign-off) ---
    deploy_parser = subparsers.add_parser(
        "deploy",
        help=(
            "Synthesize deployment artifacts (Dockerfile + compose), bring "
            "up the dev environment, run health checks, update install docs."
        ),
    )
    _add_runlike_common(deploy_parser)
    deploy_parser.add_argument(
        "--cd-discovery",
        type=_bool_choice,
        default=False,
        metavar="true|false",
        dest="cd_discovery",
        help=(
            "Run the LLM-driven deployment discovery interview before "
            "synthesizing DEPLOYMENT_BLUEPRINT.md. When false (default), "
            "the blueprint is synthesized from workspace telemetry alone."
        ),
    )
    deploy_parser.add_argument(
        "--hitl-deployment",
        type=_bool_choice,
        default=None,
        metavar="true|false",
        dest="hitl_deployment",
        help=(
            "Prompt the operator at the DEPLOYMENT gate before the dev "
            "deploy fires. Falls back to config.json hitl.deployment, "
            "then to true."
        ),
    )

    # --- `teane test` (e2e verify pack on the dev container) ---
    test_parser = subparsers.add_parser(
        "test",
        help=(
            "Run the e2e verification pack against the dev compose stack. "
            "Requires a clean prior `teane deploy` and `teane build|patch`. "
            "Generates Playwright scenarios + synthetic data, executes, and "
            "emits failing scenarios as CR-DEFECT-* in change_requests/."
        ),
    )
    _add_runlike_common(test_parser)
    test_parser.add_argument(
        "--scope",
        choices=("touched", "full"),
        default="touched",
        help=(
            "Which scenarios to run. 'touched' (default) — only scenarios "
            "whose source spec touches a service in the last deploy's CR "
            "attribution. 'full' — run every scenario in tests/e2e/."
        ),
    )
    test_parser.add_argument(
        "--retries",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Playwright per-scenario retry count for flake suppression. "
            "Defaults to 2; pass 0 to disable retries."
        ),
    )
    test_parser.add_argument(
        "--no-cleanup",
        action="store_true",
        default=False,
        help=(
            "Skip teardown of generated synthetic data after the run. "
            "Useful when debugging a failed scenario against the live "
            "compose stack."
        ),
    )

    # --- `teane resume` ---
    resume_parser = subparsers.add_parser("resume", help="Resume a crashed or interrupted session from its checkpoint")
    resume_parser.add_argument(
        "--session-id",
        required=True,
        help="The session/thread ID to resume.",
    )
    resume_parser.add_argument(
        "--workspace", "-w",
        default=None,
        help="Workspace path (auto-detected from checkpoint if omitted).",
    )
    resume_parser.add_argument(
        "--prompt", "-p",
        default=None,
        help="Optional additional prompt to append to the resumed session.",
    )
    resume_parser.add_argument(
        "--allow-network",
        type=_bool_choice,
        default=True,
        metavar="true|false",
        dest="allow_network",
        help=(
            "Permit outbound network traffic in the sandbox. Defaults to "
            "true; pass --allow-network false to block."
        ),
    )
    resume_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable debug-level logging.",
    )
    resume_parser.add_argument(
        "--git",
        type=_bool_choice,
        default=False,
        metavar="true|false",
        help=(
            "Enable GitGuardian for the resumed session. Should match the "
            "value used when the session was originally started; passing a "
            "different value than the original run may corrupt state. "
            "Defaults to false (matches `teane run` default)."
        ),
    )

    # --- `teane status` ---
    status_parser = subparsers.add_parser("status", help="Read-only inspection of a checkpointed session")
    status_parser.add_argument(
        "--session-id",
        default=None,
        help="The session/thread ID to inspect.",
    )
    status_parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="List all checkpointed sessions.",
    )
    status_parser.add_argument(
        "--workspace", "-w",
        default=None,
        help="Workspace path (for config discovery). Defaults to current directory.",
    )

    # --- `teane doctor` ---
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run first-run healthchecks (git, api keys, sandbox, db, config)",
    )
    doctor_parser.add_argument(
        "--workspace", "-w",
        default=None,
        help="Workspace path to check (defaults to current directory).",
    )
    doctor_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable debug-level logging.",
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help=(
            "Emit results as JSON on stdout instead of the coloured "
            "terminal report. Consumed by the web wizard (Phase 3) and "
            "CI. Shape matches `teane pre-flight --json-dump`."
        ),
    )

    # --- `teane pre-flight` ---
    pre_flight_parser = subparsers.add_parser(
        "pre-flight",
        help="Probe this machine for tools and runtimes the harness needs.",
        description=(
            "Standalone readiness check. Does NOT need a workspace or config. "
            "Auto-detects your OS (Windows / macOS / Linux) and prints a "
            "coloured checklist of required and optional tools, with the "
            "install command for each missing item. Run BEFORE `teane doctor`."
        ),
    )
    pre_flight_parser.add_argument(
        "--platform",
        choices=["auto", "windows", "linux", "macos"],
        default="auto",
        help="Force a specific OS's check set (default: auto-detect).",
    )
    pre_flight_parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Skip live network probes (outbound HTTPS reachability).",
    )
    pre_flight_parser.add_argument(
        "--no-color",
        action="store_true",
        default=False,
        help="Plain text output for CI / log capture.",
    )
    pre_flight_parser.add_argument(
        "--json-dump",
        type=_bool_choice,
        default=False,
        metavar="true|false",
        dest="json_dump",
        help="Machine-readable JSON output.",
    )

    # --- `teane purge` ---
    audit_parser = subparsers.add_parser(
        "audit",
        help=(
            "Run the v5 traceability audit (untraced FRs + untested ACs) "
            "against state.db. Exits 1 on failures, useful as a CI gate."
        ),
    )
    audit_parser.add_argument(
        "--workspace", "-w",
        default=None,
        help="Workspace path to audit (defaults to current directory).",
    )

    purge_parser = subparsers.add_parser("purge", help="Manually wipe checkpoint data")
    purge_parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Delete ALL checkpoint data permanently.",
    )
    purge_parser.add_argument(
        "--session-id",
        default=None,
        help="Purge checkpoints for a specific session only.",
    )
    purge_parser.add_argument(
        "--workspace", "-w",
        default=None,
        help="Workspace path (for config discovery). Defaults to current directory.",
    )

    # --- `teane metrics` ---
    metrics_parser = subparsers.add_parser(
        "metrics",
        help="Per-session cost / burn-rate / Prometheus aggregation from logs",
    )
    metrics_parser.add_argument(
        "--session-id",
        default=None,
        help="Report on a single session.",
    )
    metrics_parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Roll-up table across every session in the log dir.",
    )
    metrics_parser.add_argument(
        "--json-dump",
        type=_bool_choice,
        default=False,
        metavar="true|false",
        dest="json_dump",
        help="Write machine-readable JSON to <metrics_dir>/ (or stdout with --output-path -).",
    )
    metrics_parser.add_argument(
        "--prometheus",
        action="store_true",
        default=False,
        help="Write Prometheus text-exposition output to <metrics_dir>/ (or stdout with --output-path -).",
    )
    metrics_parser.add_argument(
        "--output-path",
        default=None,
        dest="output_path",
        help="Override the destination path. Use '-' to emit to stdout.",
    )
    metrics_parser.add_argument(
        "--window-minutes",
        type=int,
        default=None,
        help="Burn-rate trailing window in minutes (default 10; clamped to [1, 1440]).",
    )
    metrics_parser.add_argument(
        "--workspace", "-w",
        default=None,
        help="Workspace path (for config discovery). Defaults to current directory.",
    )

    # --- `teane web` (#14) — start / stop subcommands ---
    dashboard_parser = subparsers.add_parser(
        "web",
        help="Carbon-styled web UI over the harness's on-disk state.",
        description=(
            "Manage the teane web UI.\n\n"
            "The dashboard is a single-instance server per user — its pid, "
            "host, and port are recorded in a marker file at ~/.harness/web.lock "
            "so 'teane web stop' can find and stop it cleanly. A second "
            "'web start' refuses to launch while a live instance is already "
            "registered; stop it first.\n\n"
            "Subcommands:\n"
            "  start    Launch the web UI (foreground by default).\n"
            "  stop     Stop the running web UI cleanly (SIGTERM, then SIGKILL after 5s).\n"
        ),
        epilog=(
            "Examples:\n"
            "  teane web start                          # foreground, http://127.0.0.1:9000\n"
            "  teane web start --port 8080              # foreground on port 8080\n"
            "  teane web start --host 0.0.0.0           # bind all interfaces\n"
            "  teane web start --background yes         # detach; logs to ~/.harness/web.log\n"
            "  teane web stop                           # graceful shutdown\n\n"
            "Files:\n"
            "  ~/.harness/web.lock     pid + host + port + mode (created at start, removed at stop)\n"
            "  ~/.harness/web.log      stdout/stderr (background mode only)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    dashboard_subparsers = dashboard_parser.add_subparsers(
        dest="web_action", help="Web action (start / stop).",
        metavar="{start,stop}",
    )
    # `teane web start`
    web_start_parser = dashboard_subparsers.add_parser(
        "start",
        help="Start the web UI server.",
        description=(
            "Start the teane web UI.\n\n"
            "Refuses to launch if another instance is already registered "
            "(via the marker file ~/.harness/web.lock). Run 'teane web stop' "
            "first to free the marker.\n\n"
            "Foreground mode (default): the server runs in this terminal. "
            "Ctrl-C triggers a clean shutdown — the listening socket is "
            "released and the marker is removed before the process exits.\n\n"
            "Background mode (--background yes): the server is re-spawned in "
            "a detached subprocess, stdout/stderr are redirected to "
            "~/.harness/web.log, and this command returns immediately."
        ),
        epilog=(
            "Examples:\n"
            "  teane web start\n"
            "      → http://127.0.0.1:9000 (foreground)\n\n"
            "  teane web start --port 8080\n"
            "      → http://127.0.0.1:8080 (foreground)\n\n"
            "  teane web start --host 0.0.0.0 --port 8080\n"
            "      → http://0.0.0.0:8080 (all interfaces, foreground)\n\n"
            "  teane web start --background yes\n"
            "      → detached on http://127.0.0.1:9000, logs to ~/.harness/web.log\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    web_start_parser.add_argument(
        "--host", default="127.0.0.1", metavar="HOST",
        help="Bind host. Default: 127.0.0.1 (localhost only). "
             "Pass 0.0.0.0 to bind every interface — when you do, set "
             "dashboard.token_env in config.json so the server requires "
             "a bearer token.",
    )
    web_start_parser.add_argument(
        "--port", type=int, default=9000, metavar="PORT",
        help="Bind port. Default: 9000.",
    )
    web_start_parser.add_argument(
        "--background", choices=["yes", "no"], default="no", metavar="{yes,no}",
        help="yes = detach (logs to ~/.harness/web.log), no = foreground. "
             "Default: no.",
    )
    web_start_parser.add_argument(
        "--print-mobile-url", action="store_true", default=False,
        help=(
            "Print a bookmarkable URL for the Phase-7 mobile view "
            "(includes a one-shot bearer token that the browser "
            "exchanges for a cookie on first hit, then strips)."
        ),
    )
    # `teane web stop`
    dashboard_subparsers.add_parser(
        "stop",
        help="Stop the running web UI cleanly.",
        description=(
            "Stop the running teane web UI.\n\n"
            "Reads ~/.harness/web.lock to find the pid, deletes the marker, "
            "sends SIGTERM, and waits up to 5s for a clean exit. If the "
            "process is still alive after 5s it escalates to SIGKILL.\n\n"
            "Idempotent: returns 0 even if no marker is present (so scripts "
            "can call 'web stop' unconditionally before 'web start')."
        ),
        epilog=(
            "Examples:\n"
            "  teane web stop\n"
            "      → reads ~/.harness/web.lock, SIGTERMs the pid, confirms\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # `teane schedule` was removed from the CLI. The ScheduleDaemon module
    # (harness/schedule.py) and its config section are preserved so the
    # web dashboard can drive scheduled runs in a future iteration. Don't
    # re-register a parser here.

    # --- `teane chat` (#8) ---
    chat_parser = subparsers.add_parser(
        "chat",
        help="Interactive refinement REPL — reuses the gateway, tools, and memory; no auto-apply.",
    )
    chat_parser.add_argument(
        "--workspace", "-w",
        default=None,
        help="Workspace path. Defaults to current directory.",
    )
    chat_parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Optional per-session budget cap in USD. Falls back to token_budget.hard_cap_usd.",
    )

    # --- `teane index <action>` ---
    index_parser = subparsers.add_parser(
        "index",
        help="Build / inspect / clear the per-workspace semantic retrieval index (repo_index.*).",
    )
    index_subparsers = index_parser.add_subparsers(dest="index_action", help="Index action")
    index_build_parser = index_subparsers.add_parser(
        "build",
        help="(Re)build the repo index for the given workspace.",
    )
    index_build_parser.add_argument(
        "--workspace", "-w",
        default=None,
        help="Workspace path. Defaults to current directory.",
    )
    index_status_parser = index_subparsers.add_parser(
        "status",
        help="Show the prior index summary for the given workspace.",
    )
    index_status_parser.add_argument(
        "--workspace", "-w",
        default=None,
        help="Workspace path. Defaults to current directory.",
    )
    index_clear_parser = index_subparsers.add_parser(
        "clear",
        help="Delete the prior index for the given workspace.",
    )
    index_clear_parser.add_argument(
        "--workspace", "-w",
        default=None,
        help="Workspace path. Defaults to current directory.",
    )

    # --- `teane gh <action>` ---
    gh_parser = subparsers.add_parser(
        "gh",
        help="GitHub integration (issue ingest, PR create, PR comment). Requires `gh` CLI on PATH.",
    )
    gh_subparsers = gh_parser.add_subparsers(dest="gh_action", help="GitHub action")

    gh_issue_parser = gh_subparsers.add_parser(
        "issue",
        help="Pull a GitHub issue into change_requests/CR-N-<slug>.txt for processing.",
    )
    gh_issue_parser.add_argument("--repo", required=True, help="owner/repo")
    gh_issue_parser.add_argument("--number", type=int, required=True, help="Issue number")
    gh_issue_parser.add_argument(
        "--workspace", "-w",
        default=None,
        help="Workspace path. Defaults to current directory.",
    )

    gh_pr_create_parser = gh_subparsers.add_parser(
        "pr-create",
        help="Open a PR from the current branch via `gh pr create`.",
    )
    gh_pr_create_parser.add_argument("--title", required=True, help="PR title")
    gh_pr_create_parser.add_argument("--body", default="", help="PR body (markdown)")
    gh_pr_create_parser.add_argument("--base", default="main", help="Base branch (default: main)")
    gh_pr_create_parser.add_argument("--draft", action="store_true", help="Open as draft PR")
    gh_pr_create_parser.add_argument(
        "--workspace", "-w",
        default=None,
        help="Workspace path. Defaults to current directory.",
    )

    gh_pr_comment_parser = gh_subparsers.add_parser(
        "pr-comment",
        help="Post a comment on a PR via `gh pr comment`.",
    )
    gh_pr_comment_parser.add_argument("--repo", required=True, help="owner/repo")
    gh_pr_comment_parser.add_argument("--number", type=int, required=True, help="PR number")
    gh_pr_comment_parser.add_argument("--body", required=True, help="Comment body (markdown)")

    # --- `teane cache <action>` ---
    cache_parser = subparsers.add_parser(
        "cache",
        help="Manage harness-owned Docker cache volumes (sandbox.cache_volumes).",
    )
    cache_subparsers = cache_parser.add_subparsers(
        dest="cache_action", help="Cache action",
    )
    cache_clear_parser = cache_subparsers.add_parser(
        "clear",
        help="Remove harness-owned cache volumes (idempotent).",
    )
    cache_clear_parser.add_argument(
        "--session-id",
        default=None,
        help="Limit removal to volumes scoped to a specific session.",
    )
    cache_clear_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="Skip the confirmation prompt.",
    )
    cache_clear_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="List the volumes that would be removed without removing them.",
    )
    cache_clear_parser.add_argument(
        "--workspace", "-w",
        default=None,
        help="Workspace path (for config discovery). Defaults to current directory.",
    )

    return parser


# ---------------------------------------------------------------------------
# 5. Main Entry Point
# ---------------------------------------------------------------------------

def _install_aiosqlite_shutdown_filter() -> None:
    """Swallow the aiosqlite background-thread ``Event loop is closed``
    noise that fires on harness exit.

    aiosqlite runs each connection on a worker thread that talks back
    to the asyncio event loop via ``call_soon_threadsafe``. When
    ``asyncio.run()`` returns and tears down the loop while the worker
    is mid-flight (common on early returns, declined-confirm exits,
    short subcommands), the worker raises a ``RuntimeError`` that
    Python's default ``threading.excepthook`` dumps as a multi-line
    traceback. The race is harmless — the connection is closing
    anyway — but the traceback looks like a crash to operators.

    The filter is narrowly scoped: it only swallows
    ``RuntimeError("Event loop is closed")`` whose traceback contains
    an ``aiosqlite`` frame. All other thread exceptions (including
    other ``RuntimeError`` instances and any user code raising the
    same message outside aiosqlite) flow through the prior excepthook
    unchanged.
    """
    import threading
    prior = threading.excepthook

    def _filter(args: "threading.ExceptHookArgs") -> None:
        exc_value = args.exc_value
        if (
            args.exc_type is RuntimeError
            and exc_value is not None
            and exc_value.args == ("Event loop is closed",)
        ):
            tb = args.exc_traceback
            while tb is not None:
                if "aiosqlite" in tb.tb_frame.f_code.co_filename:
                    return  # swallow — known harmless shutdown race
                tb = tb.tb_next
        prior(args)

    threading.excepthook = _filter


def main() -> int:
    """
    Primary CLI entry point. Dispatches to the correct subcommand handler.

    Catches :class:`ConfigError` at the outermost layer — any subcommand
    whose first step calls :func:`discover_config` and finds a bad config
    propagates the error up here. We print the consolidated message to
    stderr and return exit code 2 *before* any further side effects (no
    LLM call, no checkpointer init, no workspace lock leftover). This
    keeps the user's contract: the harness either runs with valid config
    or refuses to run at all.

    Returns:
        0 on success, 1 on subcommand-level failure, 2 on config failure.
    """
    _install_aiosqlite_shutdown_filter()
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Set logging level
    if getattr(args, "verbose", False):
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("harness").setLevel(logging.DEBUG)

    # Attach a FileHandler that mirrors the human-readable stderr stream
    # to the operator's --log path. Attached at dispatch time so it
    # captures every log line from cmd_* onward, including the wipe-
    # target guard (build) and the config-load path. The wipe-target
    # guard in cmd_build refuses to run when the log target is inside
    # the workspace, so this handler's file is always outside the
    # doomed root and survives the run.
    _log_file = getattr(args, "log_file", None)
    if _log_file:
        try:
            _log_dir = os.path.dirname(os.path.abspath(_log_file))
            if _log_dir:
                os.makedirs(_log_dir, exist_ok=True)
            _log_handler = logging.FileHandler(_log_file, mode="a", encoding="utf-8")
            _log_handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logging.getLogger().addHandler(_log_handler)
        except OSError as exc:
            # Don't halt the run for a log-write failure — the operator
            # can still see everything on stderr and the JSONL session
            # log. Just warn once so they know the file wasn't attached.
            logger.warning(
                "[cli] --log %r could not be opened for writing (%s); "
                "continuing without the extra file handler.",
                _log_file, exc,
            )

    try:
        if args.command == "build":
            return asyncio.run(cmd_build(args))
        elif args.command == "patch":
            return asyncio.run(cmd_patch(args))
        elif args.command == "deploy":
            return asyncio.run(cmd_deploy(args))
        elif args.command == "test":
            return asyncio.run(cmd_test(args))
        elif args.command == "resume":
            return asyncio.run(cmd_resume(args))
        elif args.command == "status":
            return asyncio.run(cmd_status(args))
        elif args.command == "doctor":
            return asyncio.run(cmd_doctor(args))
        elif args.command == "pre-flight":
            return cmd_pre_flight(args)
        elif args.command == "purge":
            return asyncio.run(cmd_purge(args))
        elif args.command == "audit":
            return cmd_audit(args)
        elif args.command == "metrics":
            return asyncio.run(cmd_metrics(args))
        elif args.command == "cache":
            if getattr(args, "cache_action", None) == "clear":
                return asyncio.run(cmd_cache_clear(args))
            parser.parse_args([args.command, "--help"])
            return 1
        elif args.command == "chat":
            return asyncio.run(cmd_chat(args))
        elif args.command == "web":
            action = getattr(args, "web_action", None)
            if action == "start":
                return cmd_web_start(args)
            if action == "stop":
                return cmd_web_stop(args)
            parser.parse_args([args.command, "--help"])
            return 1
        # `teane schedule` was removed from the CLI surface — the daemon
        # module remains importable for web-driven invocation later.
        elif args.command == "index":
            action = getattr(args, "index_action", None)
            if action == "build":
                return cmd_index_build(args)
            if action == "status":
                return cmd_index_status(args)
            if action == "clear":
                return cmd_index_clear(args)
            parser.parse_args([args.command, "--help"])
            return 1
        elif args.command == "gh":
            action = getattr(args, "gh_action", None)
            if action == "issue":
                return cmd_gh_issue(args)
            if action == "pr-create":
                return cmd_gh_pr_create(args)
            if action == "pr-comment":
                return cmd_gh_pr_comment(args)
            parser.parse_args([args.command, "--help"])
            return 1
        else:
            parser.print_help()
            return 1
    except ConfigError as exc:
        # Deterministic config-time failure. Print directly to stderr —
        # the logging subsystem may not even be configured yet (config-
        # check runs before any other init), so going through the
        # logger would swallow the message in early-startup contexts.
        print(f"\n[harness] {exc}\n", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())