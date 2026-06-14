"""
CLI entry point, subcommand routing, HITL interactive menu loop, and configuration discovery.

Provides the following commands:
    harness run     — Primary execution entry point. Runs the full agent graph.
    harness resume  — Resume a crashed/interrupted session from its checkpoint.
    harness status  — Read-only inspection of a checkpointed session.
    harness doctor  — Run first-run healthchecks (git, API keys, sandbox, DB, config).
    harness purge   — Manually wipe all checkpoint data.

Use `harness -h` or `harness <command> -h` for detailed help on each subcommand.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Optional

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
        f"Please re-run with --repo / -r pointing at a different location\n"
        f"(your application's repository).\n"
    )
    print("=" * 72)
    logger.error(
        "[workspace] Refusing to run: workspace path resolves to the harness "
        "root itself (%s). Choose a different location.",
        harness_root,
    )
    return True


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
        return version("ai-agent-harness")
    except (PackageNotFoundError, ImportError, Exception):  # noqa: BLE001
        return "(unknown)"


# ---------------------------------------------------------------------------
# Git mode (--git=enable|disable) — process-wide state
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
    ``--git=disable``. Mirrors every public method GitGuardian exposes so
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
    def commit_repair_iteration(self, *args, **kwargs) -> bool: return True
    def commit_all_changes(self, *args, **kwargs) -> bool: return True
    def rollback(self, *args, **kwargs) -> bool: return False
    def restore_original_branch(self) -> bool: return False


def _make_git_guardian(workspace_path: str):
    """Return a real ``GitGuardian`` when ``--git=enable`` is in effect,
    otherwise a no-op :class:`_NullGitGuardian`. One place to swap so the
    call sites stay clean."""
    if _git_enabled():
        from harness.security import GitGuardian
        return GitGuardian(workspace_path)
    logger.info("[git] --git=disable — using no-op GitGuardian stub.")
    return _NullGitGuardian(workspace_path)


# ---------------------------------------------------------------------------
# Workspace lock (P1.7) — single-writer guard
# ---------------------------------------------------------------------------

# Module-level pin: the lock-file handle MUST outlive cmd_run's locals so
# the OS holds the lock for the lifetime of the process. Releasing on exit
# is automatic.
_WORKSPACE_LOCK_HANDLE: Any = None


def _acquire_workspace_lock(workspace_path: str, *, force: bool = False) -> Any:
    """Acquire an advisory exclusive lock on the workspace.

    Returns the locked file handle on success, or ``False`` when another
    session holds the lock and ``force`` is False. On platforms without
    ``fcntl`` (Windows native), logs a debug message and returns ``None``
    — we trade hardening for compatibility there since the alternatives
    (msvcrt.locking, file deletion handshake) bring their own surprises.

    Stash the handle in a module-level slot so the GC doesn't release the
    lock the moment cmd_run's local goes out of scope.
    """
    global _WORKSPACE_LOCK_HANDLE
    try:
        import fcntl  # type: ignore[import-not-found]
    except ImportError:
        logger.debug(
            "[lock] fcntl unavailable (Windows native?); skipping workspace lock."
        )
        return None

    lock_path = os.path.join(workspace_path, ".harness_session.lock")
    try:
        fh = open(lock_path, "w", encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "[lock] Could not create lock file %s: %s. Proceeding without lock.",
            lock_path, exc,
        )
        return None

    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        if force:
            logger.warning(
                "[lock] %s is held by another session, but --force-lock was "
                "passed — taking the lock anyway. Concurrent corruption is "
                "now possible; you own the risk.",
                lock_path,
            )
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            except OSError as exc:
                logger.error("[lock] Force-lock failed too: %s", exc)
                fh.close()
                return False
        else:
            logger.error(
                "[lock] Workspace %s is locked by another live `harness run` "
                "session. Refusing to start so the two don't clobber each "
                "other's patches.\n"
                "  Wait for the other session to finish, or pass --force-lock "
                "if you're certain it's stuck (e.g. a previous crash left the "
                "lock stranded).",
                workspace_path,
            )
            fh.close()
            return False

    try:
        fh.write(f"pid={os.getpid()}\n")
        fh.flush()
    except OSError:
        pass

    _WORKSPACE_LOCK_HANDLE = fh
    logger.info("[lock] Acquired workspace lock: %s (pid=%d)", lock_path, os.getpid())
    return fh


# ---------------------------------------------------------------------------
# 1. Configuration Discovery — single canonical source
# ---------------------------------------------------------------------------
#
# The harness reads ONE config file and only one: <myharness_root>/config/config.json.
# There are no fallbacks, no per-workspace overrides, no auto-generated files.
# Per-project differences (build command, docker image, network) are handled
# by the harness's existing auto-detection (graph._toolchain_image_for,
# graph._build_command_needs_network, cli._detect_default_build_command) plus
# the existing CLI flags (--build-cmd, --budget, --allow-network).
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
    ``<myharness_root>/config/config.json``.

    The harness package lives at ``<root>/harness/``, so the parent of
    this module's directory is the repo root.
    """
    package_dir = os.path.dirname(os.path.abspath(__file__))
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
            f"Create it (see <myharness_root>/config/config.json in the repo "
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

    return _strip_comments(raw)


def _get_deployment_defaults_path() -> str:
    """Resolve the optional deployment-defaults file path:
    ``<myharness_root>/config/deployment.json``.

    Lives alongside the canonical ``config.json``. Loaded by
    :func:`load_deployment_defaults`.
    """
    package_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(package_dir)
    return os.path.join(repo_root, "config", "deployment.json")


def load_deployment_defaults() -> dict[str, Any]:
    """Load the optional deployment-policy defaults from
    ``<repo_root>/config/deployment.json``.

    Unlike :func:`load_raw_config`, this file is OPTIONAL. When absent the
    function returns ``{}`` and the deployment discovery questionnaire runs
    in its current full-questionnaire mode. When present, the parsed
    object is threaded into the deployment_discovery_node so the planning
    LLM treats every populated field as a resolved policy and skips
    emitting a question for it.

    The file is intended for tech-stack-agnostic deployment policy
    (reverse proxy choice, TLS strategy, secret manager, UID/GID,
    backup destination, conflict policy, ...) — the same answers
    operators would type into every run inside one organization.

    Validation is intentionally light:
    - Top-level must be a JSON object.
    - ``schema_version``, when present, must equal 1.
    - Known top-level section keys (``network``, ``storage``,
      ``secrets``, ``infra_sync``) must be objects when present.
    - Unknown leaf keys inside sections are passed through to the LLM
      verbatim. This is intentional — operators may set
      organization-specific policies the harness has never heard of.

    Malformed JSON, unreadable file, or wrong shapes raise
    :class:`ConfigError` so the harness exits with code 2 at startup
    (mirrors :func:`load_raw_config`'s contract — bad config never
    silently degrades).
    """
    path = _get_deployment_defaults_path()
    if not os.path.isfile(path):
        logger.info(
            "[cli] No deployment.json at %s — deployment discovery will run "
            "the full questionnaire. Drop a populated config/deployment.json "
            "to suppress org-wide policy questions.",
            path,
        )
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Invalid JSON in {path}: {exc}. "
            f"Fix the JSON syntax or delete the file to fall back to the "
            f"full deployment questionnaire."
        ) from exc
    except OSError as exc:
        raise ConfigError(
            f"Cannot read {path}: {exc}. "
            f"Fix the file permissions or delete the file."
        ) from exc

    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path} must contain a JSON object at the top level, got "
            f"{type(raw).__name__}."
        )

    stripped = _strip_comments(raw)

    schema_version = stripped.get("schema_version", 1)
    if schema_version != 1:
        raise ConfigError(
            f"{path}: unsupported schema_version {schema_version!r}. "
            f"Only schema_version 1 is recognised by this harness."
        )

    for section in ("network", "storage", "secrets", "infra_sync"):
        if section in stripped and not isinstance(stripped[section], dict):
            raise ConfigError(
                f"{path}: '{section}' must be a JSON object, got "
                f"{type(stripped[section]).__name__}."
            )

    populated = [s for s in ("network", "storage", "secrets", "infra_sync")
                 if stripped.get(s)]
    logger.info(
        "[cli] Loaded deployment defaults from %s (sections populated: %s).",
        path, populated or "none",
    )
    return stripped


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
    "build_command", "allow_network", "sandbox", "token_budget",
    "node_throttle", "models", "model_routing", "persistence",
    "redaction", "security", "skills", "deployment",
    "speculative", "impact", "lintgate", "logging", "languages",
    "test_generation", "metrics", "llm_dispatch",
    # Operator-configurable name of the folder at the workspace root that
    # holds the product spec .txt files. Mandatory in config.json — the
    # harness refuses to start without it. See _load_consolidated_product_spec.
    "product_spec_dir",
    # Operator-configurable name of the folder at the workspace root that
    # holds change-request .txt files for non-greenfield runs. Optional;
    # defaults to "change_requests". When --new_build=false the folder
    # MUST contain at least one .txt — see _load_consolidated_change_requests.
    "change_requests_dir",
    # Observability + debugging knobs. See _dump_repair_prompt_to_disk and
    # the compiler.run_prod_import_smoke_check flag.
    "debug", "compiler",
    # Patcher behaviour knobs (B5: enforce_read_before_edit).
    "patcher",
    # Change-request behaviour knobs (reverse_engineer_budget_usd etc.).
    "change_requests",
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
        # Writable named Docker volumes for the readonly_cache_mounts paths,
        # scoped to the session id. See _cache_volume_name in sandbox.py.
        "cache_volumes", "cache_volumes_prefix",
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
        "max_doc_review_cycles",
        "max_code_review_cycles",
        "max_discovery_iterations",
    }),
    "persistence": frozenset({
        "db_path", "ttl_days", "redact_messages",
    }),
    "model_routing": frozenset({
        "planning_primary", "planning_mode", "planning_fallback",
        "patching_primary", "patching_mode",
        "repair_primary", "repair_fallback", "repair_mode",
        "doc_reviewer_primary", "doc_reviewer_mode", "doc_reviewer_fallback",
        "code_reviewer_primary", "code_reviewer_mode", "code_reviewer_fallback",
        "ollama_local_model", "ollama_local_backup", "force_local_only",
    }),
    "deployment": frozenset({
        "enabled", "compose_file",
        "health_check_interval_seconds", "health_check_timeout_seconds",
    }),
    "lintgate": frozenset({
        "format_modified_files",
    }),
    "logging": frozenset({
        "level", "log_dir", "json_stderr", "langsmith",
        # P2.3: rotation knobs for the per-session JSONL file handler.
        "max_bytes", "backup_count",
    }),
    "test_generation": frozenset({
        "enabled", "max_iterations",
    }),
    # P2.7: cost-metrics aggregation (harness metrics subcommand).
    "metrics": frozenset({
        "burn_rate_window_minutes", "metrics_dir",
    }),
    # Speculative branching parameters consumed by harness/speculative.py.
    # enabled defaults to FALSE (the node short-circuits to the standard
    # patching flow). Flip to True per-config when running a workload where
    # parallel exploration is likely to find a passing variant — see the
    # speculative.enabled discussion in config.json.example. num_variants is
    # the fork count; temperature controls per-variant diversity (sweet spot
    # 0.2-0.4 for code); selection_strategy is the winner-pick rule.
    "speculative": frozenset({
        "enabled", "num_variants", "temperature",
        "selection_strategy", "worktree_base_dir",
    }),
    # LLM dispatch parameters consumed by harness/gateway.py.
    # max_tokens_per_role is a free-form dict (role -> int) — the
    # validator type-checks the wrapper but doesn't enumerate role
    # names, so future NodeRole additions don't break this list.
    "llm_dispatch": frozenset({
        "max_tokens_default", "max_tokens_per_role",
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
        "enforce_read_before_edit", "use_structured_tools",
    }),
    # Pre-build smoke checks (see compiler_node prod-import step).
    "compiler": frozenset({"run_prod_import_smoke_check"}),
    # Change-request mode knobs. reverse_engineer_budget_usd caps the
    # one-shot LLM walk in reverse_engineer_architecture_node so a large
    # codebase doesn't blow the session budget on first contact.
    "change_requests": frozenset({"reverse_engineer_budget_usd"}),
}


# Per-field type schema used by validate_config_strict. Keys are dotted paths
# matching the structure in config.json. A value's runtime type must be in
# the listed tuple; bool is excluded from int matches via an explicit check
# because Python's bool is a subclass of int.
_TYPE_SCHEMA: dict[str, tuple[type, ...]] = {
    "build_command": (str,),
    "allow_network": (bool,),
    "product_spec_dir": (str,),
    "change_requests_dir": (str,),
    "debug.dump_llm_calls": (bool,),
    "debug.dump_max_files": (int,),
    "debug.dump_repair_prompts": (bool,),  # deprecated alias for dump_llm_calls
    "patcher.enforce_read_before_edit": (bool,),
    "patcher.use_structured_tools": (bool,),
    "compiler.run_prod_import_smoke_check": (bool,),
    "change_requests.reverse_engineer_budget_usd": (int, float),
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
    "token_budget.hard_cap_usd": (int, float),
    "token_budget.stages": (dict,),
    "token_budget.context_window_threshold_pct": (int, float),
    "node_throttle.max_patch_repair_iterations": (int,),
    "node_throttle.max_doc_review_cycles": (int,),
    "node_throttle.max_code_review_cycles": (int,),
    "node_throttle.max_discovery_iterations": (int,),
    "persistence.db_path": (str,),
    "persistence.ttl_days": (int,),
    "persistence.redact_messages": (bool,),
    "model_routing.planning_primary": (str,),
    "model_routing.planning_mode": (str,),
    "model_routing.planning_fallback": (str,),
    "model_routing.patching_primary": (str,),
    "model_routing.patching_mode": (str,),
    "model_routing.repair_primary": (str,),
    "model_routing.repair_fallback": (str,),
    "model_routing.repair_mode": (str,),
    "model_routing.doc_reviewer_primary": (str,),
    "model_routing.doc_reviewer_mode": (str,),
    "model_routing.doc_reviewer_fallback": (str,),
    "model_routing.code_reviewer_primary": (str,),
    "model_routing.code_reviewer_mode": (str,),
    "model_routing.code_reviewer_fallback": (str,),
    "model_routing.ollama_local_model": (str,),
    "model_routing.ollama_local_backup": (str,),
    "model_routing.force_local_only": (bool,),
    "lintgate.format_modified_files": (bool,),
    "logging.level": (str,),
    "logging.log_dir": (str,),
    "logging.json_stderr": (bool,),
    "logging.langsmith": (bool,),
    "logging.max_bytes": (int,),
    "logging.backup_count": (int,),
    "test_generation.enabled": (bool,),
    "test_generation.max_iterations": (int,),
    "speculative.enabled": (bool,),
    "speculative.num_variants": (int,),
    "speculative.temperature": (int, float),
    "speculative.selection_strategy": (str,),
    "speculative.worktree_base_dir": (str,),
    "llm_dispatch.max_tokens_default": (int,),
    "llm_dispatch.max_tokens_per_role": (dict,),
    "metrics.burn_rate_window_minutes": (int,),
    "metrics.metrics_dir": (str,),
    "deployment.enabled": (bool,),
    "deployment.compose_file": (str,),
    "deployment.health_check_interval_seconds": (int,),
    "deployment.health_check_timeout_seconds": (int,),
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
    # product_spec_dir is mandatory: the harness mandates a folder of .txt
    # files describing the product, and that folder MUST live at the
    # workspace root. We enforce both presence and the bare-folder-name
    # rule here so the operator gets an exit-2 at config-load time
    # (before lock acquisition, gateway init, GitGuardian, etc.) instead
    # of a softer runtime failure later. The folder-exists + non-empty
    # .txt-file check is separate (cmd_run does it after workspace_path
    # is known).
    spec_dir = config.get("product_spec_dir")
    if spec_dir is None:
        errors.append(
            "'product_spec_dir' is required. Set a top-level string key in "
            "config.json with the NAME of a folder at the workspace root "
            "that holds the product-specification .txt files. The name must "
            "be a bare folder name — no path separators, no absolute paths, "
            "no `..`. Example: \"product_spec_dir\": \"product_spec\"."
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
            errors.append(
                f"'model_routing.{field}' references unknown model "
                f"'{val}'. Declare it under 'models' or pick one of: "
                f"{sorted(models.keys())}"
            )

    for field in _OPTIONAL_ROUTING_FIELDS:
        val = routing.get(field, "")
        if isinstance(val, str) and val.strip() and val not in models:
            errors.append(
                f"'model_routing.{field}' is set to '{val}' but no model "
                f"by that key exists in 'models'. Either declare it or "
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

    # LLM dispatch: clamp max_tokens to a range that makes sense across
    # every supported provider. Below 256 → useless truncated replies;
    # above 32768 → blows past per-request output caps on every supported
    # model. Also validate that per-role values are positive ints (the
    # role names themselves aren't enumerated here — unknown roles in
    # max_tokens_per_role get silently ignored at dispatch time so adding
    # a new NodeRole doesn't require a validator update).
    dispatch_cfg = config.get("llm_dispatch", {})
    if isinstance(dispatch_cfg, dict):
        _MIN_MAX_TOKENS = 256
        _MAX_MAX_TOKENS = 32768
        default_mt = dispatch_cfg.get("max_tokens_default")
        if isinstance(default_mt, int) and not isinstance(default_mt, bool):
            if default_mt < _MIN_MAX_TOKENS or default_mt > _MAX_MAX_TOKENS:
                errors.append(
                    f"'llm_dispatch.max_tokens_default' must be in "
                    f"[{_MIN_MAX_TOKENS}, {_MAX_MAX_TOKENS}], got {default_mt}."
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
                if not isinstance(role_mt, int) or isinstance(role_mt, bool):
                    errors.append(
                        f"'llm_dispatch.max_tokens_per_role.{role_name}' "
                        f"must be an int, got {type(role_mt).__name__}."
                    )
                    continue
                if role_mt < _MIN_MAX_TOKENS or role_mt > _MAX_MAX_TOKENS:
                    errors.append(
                        f"'llm_dispatch.max_tokens_per_role.{role_name}' "
                        f"must be in [{_MIN_MAX_TOKENS}, {_MAX_MAX_TOKENS}], "
                        f"got {role_mt}."
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


def _detect_default_build_command(workspace_path: str) -> Optional[str]:
    """Pick a sensible build command by sniffing workspace markers.

    Returns None when the workspace gives no hint — caller falls back to
    the historical default. Probed in priority order so a polyglot repo
    with a Makefile (that actually declares a ``build:`` target) still
    uses it. A Makefile present but missing the ``build:`` target is
    treated as if it weren't there — we fall through to manifest-based
    detection so the operator gets an actionable build command instead
    of an instant exit-127 in a make-less sandbox image.
    """
    if not workspace_path or not os.path.isdir(workspace_path):
        return None

    def has(name: str) -> bool:
        return os.path.exists(os.path.join(workspace_path, name))

    if _makefile_has_target(workspace_path, "build"):
        return "make build"
    if has("pyproject.toml"):
        return "python3 -m pip install -e . && python3 -m pytest -q"
    if has("requirements.txt"):
        return "python3 -m pip install -r requirements.txt && python3 -m pytest -q"
    if has("package.json"):
        return "npm install && npm test"
    if has("Cargo.toml"):
        return "cargo build && cargo test"
    if has("go.mod"):
        return "go build ./... && go test ./..."
    # Last-chance heuristic: any .py file (top level OR one level deep,
    # e.g. ``app/__init__.py`` after LLM scaffolds a package) → bootstrap
    # pytest. The branch previously returned a bare ``python3 -m pytest -q``
    # with no install step, so freshly-scaffolded workspaces hit exit 1
    # ("pytest not installed") and the repair LLM kept editing manifests
    # that the build_command never honoured. Install pytest explicitly so
    # the first build can actually succeed.
    try:
        for entry in os.listdir(workspace_path):
            if entry.endswith(".py"):
                return "python3 -m pip install pytest && python3 -m pytest -q"
            full = os.path.join(workspace_path, entry)
            if os.path.isdir(full) and not entry.startswith("."):
                try:
                    if any(child.endswith(".py") for child in os.listdir(full)):
                        return "python3 -m pip install pytest && python3 -m pytest -q"
                except OSError:
                    continue
    except OSError:
        pass
    return None


def resolve_build_command(
    cli_build_cmd: Optional[str],
    config: dict[str, Any],
    workspace_path: Optional[str] = None,
) -> str:
    """
    Resolve the build command using hierarchical discovery:
        1. CLI flag --build-cmd (if provided)
        2. .harness_config.json 'build_command' key
        3. Workspace sniff (Makefile / pyproject.toml / package.json / etc.)
        4. Default 'make build'
    """
    if cli_build_cmd:
        logger.info("[cli] Using build command from CLI flag: %s", cli_build_cmd)
        return cli_build_cmd
    config_cmd = config.get("build_command", "")
    if config_cmd:
        logger.info("[cli] Using build command from config: %s", config_cmd)
        return config_cmd
    if workspace_path:
        detected = _detect_default_build_command(workspace_path)
        if detected:
            logger.info("[cli] Detected build command from workspace markers: %s", detected)
            return detected
    fallback = "make build"
    logger.info("[cli] No build command configured. Using default: %s", fallback)
    return fallback


# ---------------------------------------------------------------------------
# 2. HITL Interactive Menu Loop
# ---------------------------------------------------------------------------

def _gatekeeper_auto_approves() -> bool:
    """
    True when the gatekeeper should skip interactive approval — set in CI
    or when the user opted in via HARNESS_AUTO_APPROVE, or when stdin is
    not a TTY (a piped invocation has no way to answer the prompt).

    Unlike the deploy preview gate (which fails closed on non-TTY because
    LLM-generated containers are about to launch), the spec/architecture
    gatekeeper has lower blast radius — a non-TTY here just means CI, so
    auto-approve is safe.
    """
    return (
        os.environ.get("CI", "").lower() == "true"
        or os.environ.get("HARNESS_AUTO_APPROVE", "").lower() == "true"
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
    else:
        logger.warning("[gatekeeper] Unknown gate: %s. Proceeding.", gate)
        return {"node_state": {"gatekeeper_action": "approve", "current_gate": gate}}

    # Non-interactive auto-approval. The spec lists CI / HARNESS_AUTO_APPROVE
    # as supported, but the gatekeeper was previously blocking on input()
    # even when those were set — making CI runs hang forever waiting on
    # stdin. Honor the env vars here as well as a non-TTY stdin.
    if _gatekeeper_auto_approves():
        logger.info(
            "[gatekeeper] %s auto-approved (non-interactive: CI / HARNESS_AUTO_APPROVE / no TTY).",
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

        if gate == "REQUIREMENTS":
            print(f"Requirements written to {file_label}. Please review the specification.")
            print("Options:")
            print(f"  [a] Approve & Proceed to {next_phase}")
            print("  [e] Refine via text feedback")
            print("  [m] Pause for manual local edits in IDE")
            print("  [s] Save & Quit (resume later)")
        elif gate == "ARCHITECTURE":
            print(f"Technical layout blueprints written to {file_label}. Please review module boundaries.")
            print("Options:")
            print("  [a] Approve & Begin Coding/Patching")
            print("  [e] Refine layout parameters")
            print("  [m] Pause for manual edits")
            print("  [s] Save & Quit (resume later)")
        elif gate == "DEPLOYMENT":
            print(f"Application fully compiled. Docker Composition written to {file_label}.")
            print("Please review container network bridges and volumes before firing.")
            print("Options:")
            print(f"  [a] Approve & Execute Infrastructure {next_phase}")
            print("  [e] Refine variables")
            print("  [m] Pause for manual edits")
            print("  [s] Save & Quit (resume later)")
        print()

        from harness.hitl import get_channel as _get_channel
        choice = _get_channel().prompt(
            f"[HITL:{gate_label}] Select action",
            ["a", "e", "m", "s"],
            default="a",
        )

        if choice == "a":
            logger.info("[gatekeeper] %s approved by developer.", gate_label)
            return {
                "messages": messages,
                "loop_counter": loop_counter,
                "node_state": {"gatekeeper_action": "approve", "current_gate": gate},
            }

        elif choice == "e":
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

            logger.info("[gatekeeper] %s refine requested: %d chars of feedback.", gate_label, len(notes))
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
            print(f"  harness resume --session-id {session_id}")
            if workspace and workspace != os.getcwd():
                print(f"  harness resume --session-id {session_id} -r {workspace}")
            print("=" * 60)
            print()
            logger.info("[gatekeeper] %s suspended by developer. Session: %s", gate_label, session_id)
            return {
                "messages": messages,
                "loop_counter": loop_counter,
                "node_state": {"gatekeeper_action": "suspend", "current_gate": gate},
            }

        else:
            print(f"[Gatekeeper] Unknown option: '{choice}'. Please choose a, e, m, or s.")


def discovery_interview_loop(state: dict[str, Any]) -> dict[str, Any]:
    """
    Sequential discovery interview for requirements/architecture/deployment phases.

    Walks the operator through one question at a time. Each question shows the
    LLM's recommended answer (from the discovery node's ``suggested_answer``
    field); pressing Enter accepts it, typing text overrides it. Commands:
        SUSPEND — save & quit (resumable via ``harness resume``).
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
    print(f"  harness resume --session-id {session_id}")
    if workspace and workspace != os.getcwd():
        print(f"  harness resume --session-id {session_id} -r {workspace}")
    print("=" * 60)
    print()
    logger.info(
        "[discovery] %s phase suspended by developer. Session: %s",
        phase_label, session_id,
    )
    node_state["hitl_suspend"] = True
    node_state["suspended_from"] = "discovery_interview"
    return {"messages": messages, "node_state": node_state}


def _reset_iteration_counters(
    loop_counter: Optional[dict[str, Any]], *, total_repairs: int = 0,
) -> dict[str, Any]:
    """Reset only the iteration counters in ``loop_counter`` while preserving
    diagnostic trackers (``replace_block_misses_per_file``,
    ``consecutive_zero_patch_rounds``, etc.) that the repair loop relies on
    for prompt directives across HITL resume.

    Wiping the whole dict here (the original behavior) was the root cause
    behind sessions like 2d0164f0 ping-ponging through HITL: the
    ``_format_replace_block_miss_directive`` only fires at ≥2 consecutive
    misses per file, so resetting that counter to zero on every resume meant
    the LLM never received the "use a different operation" directive and went
    straight back to the same broken REPLACE_BLOCK pattern.

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
    base["patching"] = 0
    base["repair"] = 0
    base["compiler"] = 0
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

    new_build_cmd = fresh_config.get("build_command")
    old_build_cmd = state.get("build_command")
    if isinstance(new_build_cmd, str) and new_build_cmd and new_build_cmd != old_build_cmd:
        state["build_command"] = new_build_cmd
        logger.info(
            "[HITL] build_command refreshed from disk: %r -> %r",
            old_build_cmd, new_build_cmd,
        )

    new_allow_network = fresh_config.get("allow_network")
    if isinstance(new_allow_network, bool) and new_allow_network != state.get("allow_network"):
        state["allow_network"] = new_allow_network
        logger.info(
            "[HITL] allow_network refreshed from disk: %s",
            new_allow_network,
        )


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

    while True:
        print()
        print("=" * 80)
        print(f"[HUMAN-IN-THE-LOOP INTERVENTION] Trigger: {trigger}")
        print(f"  Budget: ${budget_remaining:.4f} / $2.00 | Loop Counter: {loop_counter.get('total_repairs', 0)}")
        print(f"  Exit Code: {exit_code}")
        print(f"  Modified Files: {len(modified_files)}")
        print("=" * 80)
        print()
        print("CRITICAL INFORMATION:")
        print(error_text)
        print()
        print("Options:")
        print("  [v] View active file diffs")
        print("  [r] Resume graph execution (re-run compilation node)")
        print("  [e] Inject manual hint instruction string for the repair node")
        print("  [m] Pause for manual edits (notifies harness to wait while you fix files in your IDE)")
        print("  [b] Increase session budget limit (+ $2.00)")
        print("  [s] Save & Quit (resume later)")
        print("  [q] Abandon session and execute Git rollback")
        print()

        from harness.hitl import get_channel as _get_channel
        choice = _get_channel().prompt(
            "[HITL] Select action",
            ["v", "r", "e", "m", "b", "s", "q"],
            default="r",
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
            logger.info("[HITL] Developer chose to resume. Loop counter reset to 2. Routing to compiler_node.")
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
                logger.info("[HITL] Hint injected. Loop counter reset to 1. Resuming.")
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
            logger.info("[HITL] Manual edits confirmed. Compiler errors cleared. Resuming to compiler_node.")
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
            print(f"  harness resume --session-id {session_id}")
            if workspace_path and workspace_path != os.getcwd():
                print(f"  harness resume --session-id {session_id} -r {workspace_path}")
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

_REQUIREMENTS_SYNTHESIS_PROMPT = """You are a Principal Systems Architect and Technical Product Manager.
Transform the following raw notes into a comprehensive, professionally structured
SPEC_REQUIREMENTS.md document.

## Output Sections

### 1. Executive Summary
- One paragraph describing the system's purpose and business value.

### 2. Functional Requirements (FR)
- **FR-XXX**: Title
  - Description: What the system must do.
  - Priority: Must Have / Should Have / Could Have.
  - Acceptance Criteria: Given/When/Then format.

### 3. System Scope
- In-scope features and modules.
- Out-of-scope items explicitly excluded.

### 4. Technical Constraints
- Language, framework, database, and infrastructure requirements.
- Performance targets (latency, throughput).
- Security requirements.

### 5. Explicit Edge Cases
- Error states: what happens when things go wrong.
- Boundary conditions: maximum/minimum values, concurrency limits.
- Recovery scenarios: retry logic, fallback behavior.

### 6. Non-Functional Requirements
- Reliability, scalability, observability.

## Raw Notes
{raw_notes}

## Formatting
Output as clean, well-structured Markdown. Use proper headings, bullet points,
and code blocks where appropriate. Do not include any text outside the document.
Do NOT wrap the whole document in an outer ```markdown … ``` fence — emit the
Markdown body directly, starting with the first heading. Fences are reserved
for code blocks INSIDE the document."""


async def synthesize_requirements(
    manifest_path: str,
    output_dir: str,
    gateway: Any,
) -> str:
    """
    Read raw notes from a manifest file, route to LLM for synthesis,
    and write SPEC_REQUIREMENTS.md to the output directory.

    Args:
        manifest_path: Path to the raw notes/text file.
        output_dir: Directory to write SPEC_REQUIREMENTS.md.
        gateway: Initialized LLM Gateway instance.

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

    logger.info("[requirements] Synthesizing SPEC_REQUIREMENTS.md from %d chars of raw notes...", len(raw_notes))

    from harness.gateway import NodeRole
    prompt = _REQUIREMENTS_SYNTHESIS_PROMPT.format(raw_notes=raw_notes)
    messages = [
        {"role": "system", "content": "You are a technical documentation expert. Output clean, structured Markdown."},
        {"role": "user", "content": prompt},
    ]

    try:
        response, budget = await gateway.dispatch(
            messages=messages,
            role=NodeRole.PLANNING,
            budget_remaining_usd=2.00,
        )
    except Exception as exc:
        raise RuntimeError(f"LLM synthesis failed: {exc}") from exc

    from harness.trust import validate_synthesized_spec
    content, trust_errors = validate_synthesized_spec(response.content.strip())
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
                 spec_path, len(content), response.usage.cost_usd)
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
    prompt = _ARCHITECTURE_SYNTHESIS_PROMPT.format(requirements=requirements)
    messages = [
        {"role": "system", "content": "You are a technical architecture expert. Output clean, structured Markdown."},
        {"role": "user", "content": prompt},
    ]

    try:
        response, _ = await gateway.dispatch(
            messages=messages,
            role=NodeRole.PLANNING,
            budget_remaining_usd=2.00,
        )
    except Exception as exc:
        raise RuntimeError(f"LLM architecture synthesis failed: {exc}") from exc

    from harness.trust import validate_synthesized_spec
    content, trust_errors = validate_synthesized_spec(response.content.strip())
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
        spec_path, len(content), response.usage.cost_usd,
    )
    return spec_path


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

    response, budget = await gateway.dispatch(
        messages=messages,
        role=NodeRole.PLANNING,
        budget_remaining_usd=2.00,
    )

    content = response.content.strip()
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

        print()
        print("=" * 72)
        print("[REQUIREMENT REFINEMENT GATE]")
        print(f"  Specification: {spec_path}")
        print(f"  Size: {spec_size:,} characters")
        print("=" * 72)
        print()
        print("[A] Approve — Lock this specification and proceed to graph execution.")
        print("[B] Refine — Provide additional notes to improve the specification.")
        print("[C] Manual — Edit the file in your IDE, then press Enter to continue.")
        print()

        from harness.hitl import get_channel as _get_channel
        choice = _get_channel().prompt(
            "[Requirements] Select action",
            ["a", "b", "c"],
            default="a",
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
    contain one or more ``.txt`` files. Reading order is alphabetical;
    each file's body is prefixed with a ``## <filename>`` section header
    so the synthesis LLM can see file boundaries.

    Returns the consolidated content as a single string on success.
    Returns ``None`` and prints a clear, user-facing error to stderr on
    any of these failure modes:

    - configured folder missing.
    - configured folder exists but contains no ``.txt`` files.
    """
    product_spec_dir = resolved_spec_dir

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
                "more `.txt` files describing the product, OR update\n"
                "`product_spec_dir` to a directory that does exist. The\n"
                "config value can be an absolute path (anywhere on the\n"
                "filesystem) or a path relative to the workspace."
            ),
        )
        return None

    txt_files = sorted(
        f for f in os.listdir(product_spec_dir)
        if f.endswith(".txt") and os.path.isfile(os.path.join(product_spec_dir, f))
    )
    if not txt_files:
        _fail(
            "Configured product_spec_dir contains no .txt files",
            (
                f"`{product_spec_dir}` exists but holds no `.txt` files. Add\n"
                "at least one `.txt` file with the product specification and\n"
                "re-run."
            ),
        )
        return None

    sections: list[str] = [
        f"# Product Specification (consolidated from {len(txt_files)} file(s))",
        "",
        "Source files:",
        *(f"  - {f}" for f in txt_files),
        "",
    ]
    for fname in txt_files:
        fpath = os.path.join(product_spec_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as exc:
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
        len(txt_files), product_spec_dir, len(consolidated),
    )
    return consolidated


_DEFAULT_CHANGE_REQUESTS_DIR = "change_requests"
_CHANGE_REQUESTS_ARCHIVE_SUBDIR = "applied"
# Same pattern as harness.graph._CR_FILENAME_PREFIX. Duplicated here to
# avoid a cli → graph import dependency at module load (graph imports cli
# helpers in places). A single source of truth would require a small
# shared helpers module; the duplication is cheap and the regex is stable.
_CR_FILENAME_PREFIX = re.compile(r"^CR-(\d+)(?:[-_].*)?\.txt$", re.IGNORECASE)


def _resolve_change_requests_dir(workspace_path: str, config_value: Optional[str]) -> str:
    """Resolve the ``change_requests_dir`` config value to an absolute path
    under the workspace root. Falls back to ``change_requests`` when the
    config key is absent. Same shape as :func:`_resolve_product_spec_dir`."""
    name = (config_value or _DEFAULT_CHANGE_REQUESTS_DIR).strip() or _DEFAULT_CHANGE_REQUESTS_DIR
    return os.path.normpath(os.path.join(workspace_path, name))


def _list_pending_change_request_files(change_requests_dir: str) -> list[str]:
    """Return the sorted list of `.txt` filenames at the top of
    ``change_requests_dir`` (excluding the ``applied/`` archive). Returns
    an empty list when the directory is missing.

    Files are returned as basenames; callers join with ``change_requests_dir``
    to get absolute paths. Sorted alphabetically — the same order the
    ingest node uses to assign sequential CR-N IDs.
    """
    if not os.path.isdir(change_requests_dir):
        return []
    try:
        entries = os.listdir(change_requests_dir)
    except OSError:
        return []
    pending: list[str] = []
    for entry in sorted(entries):
        full = os.path.join(change_requests_dir, entry)
        if entry == _CHANGE_REQUESTS_ARCHIVE_SUBDIR:
            continue
        if entry.endswith(".txt") and os.path.isfile(full):
            pending.append(entry)
    return pending


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
        m = _CR_FILENAME_PREFIX.match(original_name) if original_name else None
        if m is not None:
            tail = original_name[m.end(1):].lstrip("-_") or ".txt"
            if not tail.endswith(".txt"):
                tail = tail + ".txt"
            base_name = tail
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
            os.replace(src, dst)
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
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
    except OSError as exc:
        logger.warning(
            "[change_requests] Could not write manifest %s: %s",
            manifest_path, exc,
        )


def _list_workspace_entries_to_delete(
    workspace_path: str, spec_dirname: str,
) -> list[str]:
    """Enumerate the workspace-root entries that ``--new_build=true``
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
            capture_output=True, text=True, timeout=15,
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


def _print_new_build_preview(
    workspace_path: str,
    spec_dirname: str,
    files_to_delete: list[str],
    orphan_branches: list[str],
    checkpoint_sessions: list[Any],
) -> None:
    """Print a human-friendly preview of every destructive action
    ``--new_build=true`` is about to take, so the operator can review
    before confirming."""
    print(file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print("--new_build=true — REVIEW BEFORE PROCEEDING", file=sys.stderr)
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
    print("=" * 72, file=sys.stderr)


def _perform_new_build_reset(
    workspace_path: str, spec_dirname: str,
) -> None:
    """When ``--new_build=true`` fires, hard-reset the workspace.

    Three steps:

    1. Checkout the base branch (``master`` if it exists, else ``main``).
    2. Delete every file / directory at the workspace root EXCEPT the
       preserved set (``.git/`` and the configured ``spec_dirname``) and
       commit the deletions on the base branch.
    3. Delete every orphaned ``agent/patch-*`` branch in the repo.

    Runs BEFORE GitGuardian creates the new session's patch branch, so
    the new branch is forked from a now-clean base. Best-effort: any step
    that fails is logged but does not abort the harness — GitGuardian
    will still create the patch branch from whatever the working tree
    looks like after this function returns.

    When ``--git=disable`` (``_git_enabled()`` is False), steps 1 and 3
    are skipped and step 2 runs without a commit — the file deletion
    still happens so the workspace is cleaned for a fresh run, but no
    git subprocess calls are made.
    """
    def _git(*args: str) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(
            ["git", "-C", workspace_path, *args],
            capture_output=True, text=True, timeout=60,
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
            "[new_build] --git=disable — clearing workspace files without "
            "git operations (no checkout, no commit, no branch cleanup)."
        )

    # Preserved at workspace root. .git/ can't be deleted without
    # destroying the repo; the configured product-spec folder is the
    # source of truth for the next run and must survive.
    preserved = frozenset({".git", spec_dirname})
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
        commit = _git("commit", "-m", "harness: --new_build reset")
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

    Used by ``--new_build=true`` cleanup so that "starting fresh" includes
    the persistence layer, not just the working tree. Session ↔ workspace
    association is indirect (the workspace path lives in the serialized
    LangGraph checkpoint blob under ``channel_values.workspace_path``),
    so we enumerate sessions via :func:`harness.storage.list_all_sessions`
    — the same canonical path ``harness status`` already uses — and match
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


def _attempt_git_rollback(workspace_path: str) -> None:
    """Attempt a git checkout to restore modified files to their original state.

    No-op when ``--git=disable`` — without a repo there's no rollback target,
    so the workspace stays in whatever state the failure produced. The log
    line makes that explicit so the operator knows their files weren't
    silently restored.
    """
    if not _git_enabled():
        logger.info(
            "[HITL] Git rollback skipped: --git=disable. Workspace files "
            "remain in the state the failure left them in."
        )
        return
    import subprocess
    try:
        result = subprocess.run(
            ["git", "-C", workspace_path, "checkout", "--", "."],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info("[HITL] Git rollback successful.")
        else:
            logger.warning("[HITL] Git rollback failed: %s", result.stderr.strip())
    except Exception as exc:
        logger.warning("[HITL] Git rollback error: %s", exc)


# ---------------------------------------------------------------------------
# 3. Subcommand Handlers
# ---------------------------------------------------------------------------

async def cmd_run(args: argparse.Namespace) -> int:
    """
    Execute the `harness run` subcommand.

    Steps:
        1. Resolve workspace path.
        2. Discover configuration.
        3. Resolve build command.
        4. Initialize checkpointer.
        5. Compile the graph.
        6. Execute the graph with the provided prompt.
        7. Handle HITL breakpoints if triggered.

    Examples:
        harness run -r /path/to/repo -p "Add JWT authentication"
        harness run -r ./myproject -p "Refactor the auth module" --new_build=false
    """
    # Bare invocation: `harness run` with no --workspace and no --prompt.
    # Drop into the interactive setup wizard, which fills in args.workspace,
    # args.prompt, args.git, args.new_build, and args.discover before we
    # continue — OR, when the operator picks "resume existing session",
    # sets args.session_id and tells us to hand off to cmd_resume instead.
    # Half-bare (one flag set, the other missing) is the same error as
    # today — argparse won't catch it now that we dropped required=True,
    # so we enforce both-or-neither here explicitly.
    workspace_given = getattr(args, "workspace", None) is not None
    prompt_given = getattr(args, "prompt", None) is not None
    if not workspace_given and not prompt_given:
        from harness.wizard import run_setup_wizard
        if run_setup_wizard(args) == "resume":
            return await cmd_resume(args)
    elif workspace_given ^ prompt_given:
        missing = "--prompt/-p" if not prompt_given else "--workspace/-w"
        print(
            f"\nerror: {missing} is required when the other is given. "
            f"To use the interactive setup, omit BOTH flags.\n",
            file=sys.stderr,
        )
        return 2

    workspace_path = os.path.abspath(args.workspace)
    if not os.path.isdir(workspace_path):
        logger.error("Workspace path does not exist: %s", workspace_path)
        return 1
    if _refuse_if_workspace_is_harness_root(workspace_path):
        return 1

    # Record git mode for every downstream code path that touches git
    # (GitGuardian init, _attempt_git_rollback, _perform_new_build_reset).
    # Default to enabled when the attribute is missing — keeps tests and
    # programmatic callers that construct args manually working unchanged.
    _set_git_enabled(getattr(args, "git", "enable") == "enable")

    # FIRST: deterministic config check. Reads + validates the canonical
    # config file with no side effects. Raises ConfigError (caught by
    # main()) on any problem — missing file, JSON syntax error, unknown
    # keys, missing required fields, wrong types, or missing API key env
    # vars for routed models. By running this before _acquire_workspace_lock
    # we avoid leaving a stale lock file when the operator's config is bad.
    config = discover_config(workspace_path)

    # P1.7: workspace-level advisory lock. Without this, two concurrent
    # `harness run -r <same workspace>` invocations both read and write
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

    build_command = resolve_build_command(args.build_cmd, config, workspace_path)

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
    )

    # Extract budget and sandbox settings
    token_budget = config.get("token_budget", {})
    budget_usd = token_budget.get("hard_cap_usd", 2.00)
    allow_network = args.allow_network or config.get("allow_network", False)

    # Apply CLI overrides for reviewer cycle caps before gateway init so the
    # gateway picks them up. Clamping happens inside create_gateway_from_config.
    spec_cycles = getattr(args, "spec_review_cycles", None)
    code_cycles = getattr(args, "code_review_cycles", None)
    if spec_cycles is not None or code_cycles is not None:
        node_throttle_cfg = config.setdefault("node_throttle", {})
        if spec_cycles is not None:
            node_throttle_cfg["max_doc_review_cycles"] = spec_cycles
        if code_cycles is not None:
            node_throttle_cfg["max_code_review_cycles"] = code_cycles

    # Initialize the LLM Gateway and inject it for graph nodes
    from harness.gateway import create_gateway_from_config
    from harness.graph import set_gateway, run_graph

    gateway = create_gateway_from_config(config)
    set_gateway(gateway)

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

    # --- Change-request mode detection (existing-project delta path) ---
    # The harness routes an existing project's bug-fix / feature-add work
    # through the gatekeeper pipeline (PR-2+) by reading `.txt` files from
    # `change_requests_dir`. The hard rule is: when --new_build=false the
    # folder MUST contain at least one .txt file. This replaces the old
    # implicit "use the existing product_spec" path with a file-driven
    # workflow that gives every run a checked-in audit trail.
    #
    # When --new_build=true (greenfield), the change_requests/ folder is
    # ignored — greenfield uses product_spec_dir as before.
    new_build_active = bool(getattr(args, "new_build", False))
    cr_dir_abs = _resolve_change_requests_dir(
        workspace_path, config.get("change_requests_dir"),
    )
    pending_change_requests = _list_pending_change_request_files(cr_dir_abs)
    change_request_mode = bool(pending_change_requests) and not new_build_active
    if not new_build_active and not pending_change_requests:
        print(file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        print(
            "Existing-project run requires at least one change request",
            file=sys.stderr,
        )
        print("=" * 72, file=sys.stderr)
        print(
            "The harness needs at least one `.txt` file under:\n\n"
            f"  {cr_dir_abs}\n\n"
            "describing the bug to fix or feature to add. Each file becomes\n"
            "a numbered Change Request (CR-N) that flows through the\n"
            "gatekeeper review and is archived after the session terminates.\n\n"
            "To proceed:\n"
            "  1. Create the folder if it does not exist.\n"
            "  2. Add one or more `.txt` files describing the changes.\n"
            "  3. Re-run `harness run`.\n\n"
            "If you are starting a fresh build, pass --new_build=true\n"
            "instead — that flow uses `product_spec_dir` and skips this\n"
            "check.\n",
            file=sys.stderr,
        )
        print("=" * 72, file=sys.stderr)
        logger.error(
            "[change_requests] --new_build=false but no .txt files at %s",
            cr_dir_abs,
        )
        return 1
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
            "that holds the product-specification .txt files.\n\n"
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

    # Validate the folder exists AND contains at least one .txt file.
    # product_spec_dir is the SOLE source for the product spec on greenfield
    # runs — the harness no longer accepts a --manifest override. We preload
    # the consolidated content here so the requirement-refinement step below
    # doesn't walk the folder a second time.
    #
    # In change-request mode (existing-project deltas) the product_spec
    # folder is not consulted — the change_requests/ folder drives the run
    # instead. The config value's NAME is still validated above so other
    # subsystems that reference spec_dirname (e.g. --new_build cleanup
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

    # --new_build cleanup runs BEFORE GitGuardian creates the session's
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
        total_destructive = (
            len(files_to_delete) + len(orphan_branches) + len(checkpoint_sessions)
        )
        if total_destructive == 0:
            # No destructive work — skip the prompt entirely. The
            # cleanup functions will be no-ops; we still call them so
            # the log line ("No prior checkpoints", etc.) appears.
            logger.info(
                "[new_build] --new_build=true but nothing to clean "
                "(no extra files at workspace root, no orphan patch "
                "branches, no prior checkpoints for this workspace). "
                "Skipping the confirmation prompt."
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
                from harness.hitl import get_channel as _get_channel
                confirmed = _get_channel().confirm(
                    "Proceed with the destructive --new_build reset above?",
                    default=False,
                )
                if not confirmed:
                    print(
                        "\n--new_build reset cancelled. Re-run without "
                        "--new_build=true (or fix the workspace state) "
                        "before retrying.",
                        file=sys.stderr,
                    )
                    logger.warning(
                        "[new_build] Operator declined the reset. Exiting."
                    )
                    return 1
        logger.warning(
            "[new_build] --new_build=true — resetting workspace before "
            "starting the session. Files outside `%s/` and `.git/` will be "
            "deleted from the base branch.", spec_dirname,
        )
        _perform_new_build_reset(workspace_path, spec_dirname)
        # And purge every prior checkpoint + JSONL transcript that targeted
        # this workspace, so "fresh start" includes the persistence layer.
        # Runs BEFORE GitGuardian creates this session's patch branch and
        # before any checkpoint write for this session, so list_all_sessions
        # can't accidentally match (and delete) the run we're about to start.
        await _purge_workspace_checkpoints(workspace_path, config)

    # Initialize GitGuardian for branch lifecycle management. When
    # --git=disable, _make_git_guardian returns a no-op stub so the
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
    if not change_request_mode:
        import tempfile as _tempfile
        fd, manifest_path = _tempfile.mkstemp(
            prefix=f"harness_spec_{session_id[:8]}_", suffix=".txt",
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(preloaded_consolidated_spec)
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
            # Resolve output_dir relative to the workspace, not the CWD where harness was invoked
            output_dir = args.output_dir
            if not os.path.isabs(output_dir):
                output_dir = os.path.join(workspace_path, output_dir)
            spec_path = await synthesize_requirements(
                manifest_path=manifest_path,
                output_dir=output_dir,
                gateway=gateway,
            )
            # Pre-flight spec review: fire whenever doc_reviewer_primary is
            # configured, regardless of whether --discover was passed.
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
            spec_override = await interactive_review_loop(spec_path, gateway)
            logger.info("[requirements] Specification locked. %d characters approved.", len(spec_override))

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
    # If we got here, manifest_path is always set — _load_consolidated_product_spec
    # either succeeded or already returned 1 above. The old "fall through with
    # no spec" branch is gone with the product_spec/ folder mandate.

    thread_id = args.thread_id if args.thread_id else session_id

    logger.info("=" * 60)
    logger.info("AI Agent Harness — Starting Graph Execution")
    logger.info("  Workspace:  %s", workspace_path)
    logger.info("  Build Cmd:  %s", build_command)
    logger.info("  Session ID: %s", session_id)
    logger.info("  Thread ID:  %s", thread_id)
    logger.info("  Budget:     $%.2f", budget_usd)
    logger.info("  Network:    %s", "enabled" if allow_network else "blocked")
    logger.info("  Prompt:     %s", args.prompt[:100] + ("..." if len(args.prompt) > 100 else ""))
    logger.info("  Discovery:  %s", "enabled (--discover)" if getattr(args, "discover", False) else "skipped (pass --discover to enable)")
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
            # Discovery runs only when --discover is explicitly passed.
            # --skip-discovery (old flag) is a no-op now but kept for compat.
            skip_discovery=not getattr(args, "discover", False),
            lintgate_config=config.get("lintgate", {}),
            deployment_config=config.get("deployment", {}),
            deployment_defaults=load_deployment_defaults(),
            sandbox_config=config.get("sandbox", {}),
            test_generation_config=config.get("test_generation", {}),
            speculative_config=config.get("speculative", {}),
            compiler_config=config.get("compiler", {}),
            change_request_mode=change_request_mode,
            change_requests_dir_abs=cr_dir_abs if change_request_mode else "",
            archive_target_dir=archive_target_dir,
            change_requests_config=config.get("change_requests", {}),
        )
    except Exception:
        logger.exception("Graph execution failed with unhandled exception.")
        git_guardian.rollback()
        git_guardian.pop_stash()
        await checkpointer.conn.close()
        return 1

    exit_code = final_state.get("exit_code", -1)
    modified_files = final_state.get("modified_files", [])
    token_tracker = final_state.get("token_tracker", {})
    total_cost = token_tracker.get("total_cost_usd", 0.0)

    # Distinguish HITL Save & Quit (intentional pause; operator will
    # `harness resume`) from a hard failure. Previously both took the same
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
        # subsequent `harness resume --session-id <id>` picks up against
        # the same files. The pre-session stash stays parked (the operator
        # can list it with `git stash list` and pop it manually if they
        # need the prior work); popping it here could merge-conflict with
        # the LLM's edits and surprise the operator.
        agent_branch = getattr(git_guardian, "_patch_branch", None) or "agent/patch-<unknown>"
        logger.info(
            "[cli] HITL suspend: leaving %d LLM-modified file(s) on branch "
            "'%s'. Resume with `harness resume --session-id %s` to continue "
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
    else:
        # Real build failure with no operator intervention — rollback as before.
        git_guardian.rollback(modified_files)
        git_guardian.pop_stash()

    logger.info("=" * 60)
    logger.info("Graph Execution Complete")
    logger.info("  Exit Code:      %d", exit_code)
    logger.info("  Modified Files: %d", len(modified_files))
    for f in modified_files:
        logger.info("    - %s", f)
    logger.info("  Token Cost:     $%.6f", total_cost)
    logger.info("  Session ID:     %s", session_id)
    logger.info("=" * 60)

    # Archive consumed change-request .txt files into
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

    return 0 if exit_code == 0 else 1


async def cmd_resume(args: argparse.Namespace) -> int:
    """
    Execute the `harness resume` subcommand.

    Restores a previously checkpointed session from SQLite and resumes
    graph execution from the exact checkpoint boundary.

    Example:
        harness resume --session-id my-session-abc123
        harness resume --session-id my-session -r /path/to/repo
    """
    from harness.storage import HarnessAsyncSqliteSaver

    workspace_path = os.path.abspath(args.workspace) if args.workspace else os.getcwd()
    if _refuse_if_workspace_is_harness_root(workspace_path):
        return 1

    # Record git mode for the resumed session — same contract as cmd_run.
    # See the comment in cmd_run for why this is module-level state.
    _set_git_enabled(getattr(args, "git", "enable") == "enable")

    config = discover_config(workspace_path)
    persistence_cfg = config.get("persistence", {})
    db_path = persistence_cfg.get("db_path", "~/.harness/checkpoints.db")
    ttl_days = persistence_cfg.get("ttl_days", 30)
    redact_messages = bool(persistence_cfg.get("redact_messages", True))

    checkpointer = await HarnessAsyncSqliteSaver.from_db_path(
        db_path=db_path, ttl_days=ttl_days, redact_messages=redact_messages,
    )

    # Verify that the thread exists
    config_for_get = {"configurable": {"thread_id": args.session_id}}
    existing = await checkpointer.aget(config_for_get)
    if existing is None:
        logger.error("No checkpoint found for session '%s'.", args.session_id)
        await checkpointer.conn.close()
        return 1

    # Bind the active session_id immediately so any pre-graph dispatches
    # (e.g. checkpoint health-check helpers, future hooks) and the in-graph
    # dispatches that follow all stamp the correct session into
    # ~/.harness/debug/<sid>_<seqno>_<role>_<model>.txt filenames. See the
    # matching call in cmd_run.
    from harness.observability import set_active_session_id
    set_active_session_id(args.session_id)

    build_command = resolve_build_command(args.build_cmd, config, workspace_path)
    token_budget = config.get("token_budget", {})
    budget_usd = token_budget.get("hard_cap_usd", 2.00)
    allow_network = args.allow_network or config.get("allow_network", False)

    # Initialize the LLM Gateway and inject it for graph nodes
    from harness.gateway import create_gateway_from_config
    from harness.graph import set_gateway, run_graph

    gateway = create_gateway_from_config(config)
    set_gateway(gateway)

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
                "Use `harness status --all` to list available sessions.",
                args.session_id,
            )
            return 1
        try:
            _deserialize_checkpoint_blob(row[0], strict=True)
        except CheckpointCorruptedError as exc:
            logger.error(
                "[resume] Checkpoint for session '%s' is corrupted: %s\n"
                "  Options:\n"
                "    - Start a fresh session with `harness run -r %s -p '<prompt>'`.\n"
                "    - Restore checkpoints.db from a known-good backup.\n"
                "    - Run `harness purge --session-id %s` to drop only this session.",
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
                "    - Start a fresh session with `harness run -r %s -p '<prompt>'`.\n"
                "    - Run `harness purge --session-id %s` to drop only this session.",
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
    # inspect_session() helper that powers `harness status`. Failures
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
    # to the agent one. When --git=disable, _make_git_guardian returns a
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
            deployment_config=config.get("deployment", {}),
            deployment_defaults=load_deployment_defaults(),
            sandbox_config=config.get("sandbox", {}),
            test_generation_config=config.get("test_generation", {}),
            speculative_config=config.get("speculative", {}),
            compiler_config=config.get("compiler", {}),
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
            "'%s'. Resume with `harness resume --session-id %s` to continue "
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
    return 0 if exit_code == 0 else 1


async def cmd_status(args: argparse.Namespace) -> int:
    """
    Execute the `harness status` subcommand.

    Reads the SQLite checkpoint database read-only and prints a clean
    text snapshot of the specified session's state without triggering
    any graph execution.

    Examples:
        harness status --session-id my-session
        harness status --all
    """
    from harness.storage import HarnessAsyncSqliteSaver, inspect_session, list_all_sessions

    workspace_path = os.path.abspath(args.workspace) if args.workspace else os.getcwd()
    config = discover_config(workspace_path)
    persistence_cfg = config.get("persistence", {})
    db_path = persistence_cfg.get("db_path", "~/.harness/checkpoints.db")
    ttl_days = persistence_cfg.get("ttl_days", 30)

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

    summary = await inspect_session(db_path, args.session_id)
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
# 3b. `harness doctor` — first-run healthcheck
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
    """Workspace is a git repo (rev-parse --git-dir) AND HEAD resolves."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "-C", workspace_path, "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
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
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        return "warn", "git repo detected, but HEAD verify timed out"
    if head_result.returncode != 0:
        return "warn", (
            f"git repo at {workspace_path} has no commits yet (unborn HEAD); "
            "make an initial commit before 'harness run' to enable speculative repair"
        )
    return "pass", f"git repo detected at {workspace_path}"


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
                capture_output=True, text=True, timeout=5,
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
                capture_output=True, text=True, timeout=5,
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
                f"`harness purge --session-id <id>` to drop them.",
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


def _doctor_check_global_config() -> tuple[str, str]:
    """The in-repo global config file at <myharness_root>/config/config.json exists.

    Without it, discover_config falls back to harness/cli.json's empty-routing
    defaults and the first LLM dispatch will fail with no model configured.
    """
    path = _get_global_config_path()
    if not os.path.isfile(path):
        return "fail", (
            f"missing {path} — run scripts/setup.py or copy config/config.json.example to config/config.json"
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
    ``.txt`` file.

    The value must be a bare folder name (no path separators, no
    absolute paths, no `..`). The harness mandates the spec folder lives
    inside the workspace so the operator's product description is
    versioned alongside the code that implements it. The operator should
    hear about a misconfiguration in `harness doctor` before they try
    `harness run`.
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
    if not os.path.isdir(resolved):
        return (
            "fail",
            f"`{spec_dirname.strip()}/` folder not found at workspace "
            f"root — expected at {resolved!r}. Create it and add one or "
            f"more .txt files",
        )
    txt_files = [
        f for f in os.listdir(resolved)
        if f.endswith(".txt") and os.path.isfile(os.path.join(resolved, f))
    ]
    if not txt_files:
        return (
            "fail",
            f"`{spec_dirname.strip()}/` exists but contains no .txt "
            "files — add at least one .txt file with the product "
            "specification",
        )
    return (
        "pass",
        f"`{spec_dirname.strip()}/` at workspace root contains "
        f"{len(txt_files)} .txt file(s)",
    )


def _doctor_check_tree_sitter() -> tuple[str, str]:
    """Tree-sitter and the language-pack catalogue are importable and at
    least one grammar loads + parses.

    The harness uses tree-sitter in two places — :class:`HybridPatcher`
    for AST-aware code modifications (``harness/patcher.py``) and
    :class:`DependencyGraph` for cross-file symbol extraction
    (``harness/impact.py``). When the pip package
    ``tree-sitter-language-pack`` is missing or its bundled grammars stop
    loading on a new Python ABI, both subsystems silently fall back to
    regex extraction — every harness run continues to work but loses
    structural awareness, and no operator-facing signal warns about it.
    This check surfaces that degradation BEFORE the next harness run.

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
            language = get_language(grammar_name)  # type: ignore[arg-type]
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


async def cmd_doctor(args: argparse.Namespace) -> int:
    """
    Execute the `harness doctor` subcommand.

    Runs healthchecks and prints a green/yellow/red summary. Under the
    single-source-config contract the very first check is "config" —
    if it fails, the harness can't load anything else, so every
    downstream check is marked "skipped" and the doctor returns
    non-zero. When the config is valid we proceed to git, sandbox,
    checkpoint DB, and the live API-key ping.

    Exits 0 if every executed check passes (warn is non-blocking),
    non-zero on any failure or when config validation prevents the
    rest of the checks from running.

    Examples:
        harness doctor
        harness doctor -r /path/to/repo
    """
    workspace_path = os.path.abspath(args.workspace) if args.workspace else os.getcwd()
    # Silence the chatty INFO logging from discover_config; we surface
    # the result via the explicit "config" check.
    logging.getLogger("harness.cli").setLevel(logging.ERROR)

    # --- Step 1: deterministic config check. EVERYTHING else depends on
    # this passing — if the canonical config doesn't load and validate,
    # there's nothing meaningful to check downstream. We still run the
    # workspace-independent git check (operators may be debugging a
    # config-broken setup from a non-git directory).
    config_status, config_detail = _doctor_check_config(workspace_path)

    checks: list[tuple[str, tuple[str, str]]] = [
        ("config", (config_status, config_detail)),
        ("git repo", _doctor_check_git(workspace_path)),
    ]

    if config_status == "pass":
        # Config is valid; load it for downstream checks and run them.
        # discover_config can't raise here because the check just passed.
        config = discover_config(workspace_path)
        api_keys_result = await _doctor_check_api_keys(config)
        checks.extend([
            ("product spec", _doctor_check_product_spec(config, workspace_path)),
            ("api keys (live)", api_keys_result),
            ("tree-sitter", _doctor_check_tree_sitter()),
            ("sandbox backend", _doctor_check_sandbox(config)),
            ("checkpoint db", _doctor_check_checkpoint_db(config)),
        ])
    else:
        # Config invalid → mark downstream checks as skipped so the
        # operator sees they exist but understands they can't run yet.
        skipped_detail = "skipped — fix the config check above first"
        checks.extend([
            ("product spec", ("skip", skipped_detail)),
            ("api keys (live)", ("skip", skipped_detail)),
            ("tree-sitter", ("skip", skipped_detail)),
            ("sandbox backend", ("skip", skipped_detail)),
            ("checkpoint db", ("skip", skipped_detail)),
        ])

    print()
    print("=" * 72)
    print(f"harness doctor — workspace: {workspace_path}")
    print(f"canonical config: {_get_global_config_path()}")
    print("=" * 72)
    for label, (status, detail) in checks:
        print(_format_doctor_line(status, label, detail))
    print("=" * 72)

    failures = [label for label, (status, _) in checks if status == "fail"]
    warnings = [label for label, (status, _) in checks if status == "warn"]
    skipped = [label for label, (status, _) in checks if status == "skip"]
    if failures:
        print(f"FAIL: {len(failures)} check(s) failed: {', '.join(failures)}")
        if "config" in failures:
            print(
                "Fix the config file at the path shown above and re-run "
                "`harness doctor` — the harness will not proceed with "
                "invalid configuration."
            )
        return 1
    if warnings:
        print(f"OK with warnings ({len(warnings)}): {', '.join(warnings)}")
    elif skipped:
        # Defensive — shouldn't be reached because skipped requires a fail.
        print(f"PARTIAL: {len(skipped)} check(s) skipped.")
        return 1
    else:
        print("OK: all checks passed.")
    return 0


async def cmd_purge(args: argparse.Namespace) -> int:
    """
    Execute the `harness purge` subcommand.

    Wipes all checkpoint data from the SQLite database.

    Examples:
        harness purge --session-id my-session
        harness purge --all
    """
    workspace_path = os.path.abspath(args.workspace) if args.workspace else os.getcwd()
    config = discover_config(workspace_path)
    persistence_cfg = config.get("persistence", {})
    db_path = persistence_cfg.get("db_path", "~/.harness/checkpoints.db")
    ttl_days = persistence_cfg.get("ttl_days", 30)

    from harness.storage import HarnessAsyncSqliteSaver, purge_checkpoints

    if args.all:
        print("WARNING: This will delete ALL checkpoint data permanently.")
        from harness.hitl import get_channel as _get_channel
        confirmed = _get_channel().confirm("Type 'yes' to confirm purge of all checkpoint data", default=False)
        if not confirmed:
            print("Purge cancelled.")
            return 0
        deleted = await purge_checkpoints(db_path)
        print(f"Purged {deleted} rows from the checkpoint database.")
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


async def cmd_cache_clear(args: argparse.Namespace) -> int:
    """Execute ``harness cache clear``.

    Enumerates harness-owned Docker volumes (those prefixed with
    ``harness-`` by default; configurable via ``sandbox.cache_volumes_prefix``)
    and removes them. With ``--session-id`` only the volumes scoped to that
    session are touched; otherwise every harness-owned volume is removed.

    Idempotent: a volume that has already been removed (or never existed)
    is treated as success, not an error.

    Examples:
        harness cache clear
        harness cache clear --session-id sess-abc123
        harness cache clear --yes  # skip the confirmation prompt
        harness cache clear --dry-run
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
            capture_output=True, text=True, timeout=15,
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
        print(f"No harness cache volumes found for {scope}.")
        return 0

    print(f"Found {len(candidates)} harness cache volume(s):")
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
            capture_output=True, text=True, timeout=30,
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
    """Execute the `harness metrics` subcommand (P2.7).

    Aggregates per-session cost/usage from the JSONL logs and renders it
    as a human report (stdout), JSON dump, Prometheus exposition text,
    or roll-up table across every session in the log directory.

    Examples:
        harness metrics --session-id abc123
        harness metrics --all
        harness metrics --session-id abc123 --prometheus
        harness metrics --all --json --output -
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
    # the configured metrics_dir unless --output overrides.
    if args.json:
        if args.session_id:
            body = json.dumps(metrics_list[0].to_jsonable(), indent=2) + "\n"
            default_name = f"{args.session_id}.json"
        else:
            body = json.dumps(
                {"sessions": [m.to_jsonable() for m in metrics_list]},
                indent=2,
            ) + "\n"
            default_name = "sessions.json"
        _emit_output(body, default_name, metrics_dir, args.output, write_atomic)
    elif args.prometheus:
        body = format_prometheus(metrics_list, hard_cap_usd=hard_cap_usd)
        default_name = (
            f"{args.session_id}.prom" if args.session_id else "all.prom"
        )
        _emit_output(body, default_name, metrics_dir, args.output, write_atomic)
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
        prog="harness",
        description=(
            "AI Agent Harness — Production-grade, model-agnostic LangGraph agent\n"
            "for autonomous code generation, sandboxed builds, and bulletproof persistence.\n\n"
            "Quick Start:\n"
            "  harness run -r /path/to/repo -p \"Your engineering task description\"\n"
            "  harness -h                     Show this help\n"
            "  harness --version              Print the installed harness version\n"
            "  harness run -h                 Show run subcommand help\n"
            "  harness status --all           List all checkpointed sessions\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  harness run -r ./myproject -p \"Add JWT authentication\"\n"
            "  harness run -r /path/to/repo -p \"Refactor logging\" --manifest notes.txt\n"
            "  harness resume --session-id abc123\n"
            "  harness status --session-id abc123\n"
            "  harness purge --all\n"
        ),
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"harness {_get_harness_version()}",
        help="Print the installed harness version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- `harness run` ---
    run_parser = subparsers.add_parser("run", help="Execute the agent graph on a workspace")
    run_parser.add_argument(
        "--output-dir", "-o",
        default="./docs",
        help="Directory to write SPEC_REQUIREMENTS.md (default: ./docs).",
    )
    # --workspace and --prompt are NOT required at the argparse level so that
    # `harness run` with no flags can drop the user into the interactive
    # setup wizard (harness.wizard.run_setup_wizard). The handler enforces
    # "both or neither" — passing only one still errors out the same way.
    run_parser.add_argument(
        "--workspace", "-w", "-r",
        default=None,
        help="Absolute or relative path to the target repository root.",
    )
    run_parser.add_argument(
        "--prompt", "-p",
        default=None,
        help="The engineering task description (e.g., 'Refactor the auth module to use JWT').",
    )
    run_parser.add_argument(
        "--build-cmd",
        default=None,
        help="Override the build command (e.g., 'make build'). Falls back to .harness_config.json or 'make build'.",
    )
    run_parser.add_argument(
        "--session-id",
        default=None,
        help="Human-readable session identifier. Auto-generated UUIDv4 if not provided.",
    )
    run_parser.add_argument(
        "--thread-id",
        default=None,
        help="LangGraph thread ID for checkpoint lookups. Defaults to session-id.",
    )
    run_parser.add_argument(
        "--allow-network",
        action="store_true",
        default=False,
        help="Permit outbound network traffic in the sandbox.",
    )
    run_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable debug-level logging.",
    )
    # Discovery is OFF by default — most tasks are incremental patches on an
    # existing codebase where the 20-minute exhaustive Q&A adds no value.
    # Pass --discover to enable the full requirements/architecture interview.
    run_parser.add_argument(
        "--discover",
        action="store_true",
        default=False,
        dest="discover",
        help=(
            "Run the full requirements/architecture/deployment discovery "
            "pipeline before code generation. Recommended for greenfield "
            "projects or when working from a blank workspace. Skipped by "
            "default for incremental patching sessions."
        ),
    )
    # Keep --skip-discovery as a no-op alias for backward compatibility
    # (it was the old default=False flag; now discovery is already off).
    run_parser.add_argument(
        "--skip-discovery", "-s",
        action="store_true",
        default=False,
        dest="skip_discovery_compat",
        help=argparse.SUPPRESS,  # hidden; no longer needed but kept for scripts
    )
    # P1.7: workspace-lock override. When another live `harness run` holds
    # the workspace's session lock, the new run normally refuses to start.
    # Pass this flag to take the lock anyway — meant for the recovery case
    # where a previous process crashed without releasing it.
    run_parser.add_argument(
        "--force-lock",
        action="store_true",
        default=False,
        dest="force_lock",
        help=(
            "Bypass the workspace session lock. Use ONLY when the previous "
            "harness run process crashed and the .harness_session.lock file "
            "is stale; running two live sessions concurrently against one "
            "workspace will corrupt each other's patches."
        ),
    )
    # Reviewer cycle caps. Activation is purely by which model slots are set
    # in .harness_config.json; these flags only control how many times the
    # cycle runs when active. Clamped to [0, 5]; 0 suspends the reviewer.
    run_parser.add_argument(
        "--spec-review-cycles",
        type=int,
        default=None,
        dest="spec_review_cycles",
        help=(
            "Override max_doc_review_cycles for this run (0-5). 0 suspends the "
            "doc reviewer without clearing doc_reviewer_primary in config."
        ),
    )
    run_parser.add_argument(
        "--code-review-cycles",
        type=int,
        default=None,
        dest="code_review_cycles",
        help=(
            "Override max_code_review_cycles for this run (0-5). 0 suspends "
            "the code reviewer without clearing code_reviewer_primary in config."
        ),
    )
    # --new-build / --new_build accepts true|false. Default false (treat the
    # workspace as steady-state). When true, the harness deletes every file
    # / dir at workspace root except product_spec/ and .git/, commits the
    # cleanup on the base branch (master or main), and deletes every
    # orphaned agent/patch-* branch — see _perform_new_build_reset. No
    # confirmation prompt; the operator opted in by passing the flag.
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
    run_parser.add_argument(
        "--new-build", "--new_build",
        dest="new_build",
        type=_bool_choice,
        default=False,
        metavar="true|false",
        help=(
            "When true, treat this as a brand-new app: delete every file "
            "and directory at the workspace root EXCEPT `product_spec/` and "
            "`.git/`, commit the deletions on the base branch (master / "
            "main), and remove every orphaned `agent/patch-*` branch in the "
            "repo. The harness prints a preview and asks for confirmation "
            "before deleting anything; pass --yes to skip the prompt for "
            "automation. Runs before the patch branch for this session is "
            "created, so the new branch forks from a fully clean baseline. "
            "Defaults to false (steady-state — workspace contents are preserved)."
        ),
    )
    run_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        dest="assume_yes",
        help=(
            "Skip the --new_build=true confirmation prompt. Intended for "
            "automation / non-interactive runs. Has no effect when "
            "--new_build is false."
        ),
    )
    # --git enable|disable. Default 'enable' preserves today's behavior
    # (workspace is a git repo, GitGuardian stashes/branches/rolls back).
    # 'disable' skips every git-aware path so users whose target repo
    # isn't under git can still run the harness. Security scanners like
    # gitleaks still run (they scan files, not history).
    run_parser.add_argument(
        "--git",
        choices=["enable", "disable"],
        default="enable",
        help=(
            "Whether the workspace is a git repo. 'enable' (default) uses "
            "GitGuardian for stash/patch-branch/rollback and requires the "
            "workspace to be a git repo. 'disable' skips every git-aware "
            "step — pick this when the target repo isn't under git. "
            "Security scanners (gitleaks, etc.) still run either way."
        ),
    )

    # --- `harness resume` ---
    resume_parser = subparsers.add_parser("resume", help="Resume a crashed or interrupted session from its checkpoint")
    resume_parser.add_argument(
        "--session-id",
        required=True,
        help="The session/thread ID to resume.",
    )
    resume_parser.add_argument(
        "--workspace", "-w", "-r",
        default=None,
        help="Workspace path (auto-detected from checkpoint if omitted).",
    )
    resume_parser.add_argument(
        "--prompt", "-p",
        default=None,
        help="Optional additional prompt to append to the resumed session.",
    )
    resume_parser.add_argument(
        "--build-cmd",
        default=None,
        help="Override the build command.",
    )
    resume_parser.add_argument(
        "--allow-network",
        action="store_true",
        default=False,
        help="Permit outbound network traffic in the sandbox.",
    )
    resume_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable debug-level logging.",
    )
    resume_parser.add_argument(
        "--git",
        choices=["enable", "disable"],
        default="enable",
        help=(
            "Whether the workspace is a git repo. Should match the value "
            "used when the session was originally started; passing a "
            "different value than the original run may corrupt state."
        ),
    )

    # --- `harness status` ---
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
        "--workspace", "-w", "-r",
        default=None,
        help="Workspace path (for config discovery). Defaults to current directory.",
    )

    # --- `harness doctor` ---
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run first-run healthchecks (git, api keys, sandbox, db, config)",
    )
    doctor_parser.add_argument(
        "--workspace", "-w", "-r",
        default=None,
        help="Workspace path to check (defaults to current directory).",
    )
    doctor_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable debug-level logging.",
    )

    # --- `harness purge` ---
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
        "--workspace", "-w", "-r",
        default=None,
        help="Workspace path (for config discovery). Defaults to current directory.",
    )

    # --- `harness metrics` ---
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
        "--json",
        action="store_true",
        default=False,
        help="Write machine-readable JSON to <metrics_dir>/ (or stdout with --output -).",
    )
    metrics_parser.add_argument(
        "--prometheus",
        action="store_true",
        default=False,
        help="Write Prometheus text-exposition output to <metrics_dir>/ (or stdout with --output -).",
    )
    metrics_parser.add_argument(
        "--output",
        default=None,
        help="Override the destination path. Use '-' to emit to stdout.",
    )
    metrics_parser.add_argument(
        "--window-minutes",
        type=int,
        default=None,
        help="Burn-rate trailing window in minutes (default 10; clamped to [1, 1440]).",
    )
    metrics_parser.add_argument(
        "--workspace", "-w", "-r",
        default=None,
        help="Workspace path (for config discovery). Defaults to current directory.",
    )

    # --- `harness cache <action>` ---
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
        "--workspace", "-w", "-r",
        default=None,
        help="Workspace path (for config discovery). Defaults to current directory.",
    )

    return parser


# ---------------------------------------------------------------------------
# 5. Main Entry Point
# ---------------------------------------------------------------------------

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
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Set logging level
    if getattr(args, "verbose", False):
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("harness").setLevel(logging.DEBUG)

    try:
        if args.command == "run":
            return asyncio.run(cmd_run(args))
        elif args.command == "resume":
            return asyncio.run(cmd_resume(args))
        elif args.command == "status":
            return asyncio.run(cmd_status(args))
        elif args.command == "doctor":
            return asyncio.run(cmd_doctor(args))
        elif args.command == "purge":
            return asyncio.run(cmd_purge(args))
        elif args.command == "metrics":
            return asyncio.run(cmd_metrics(args))
        elif args.command == "cache":
            if getattr(args, "cache_action", None) == "clear":
                return asyncio.run(cmd_cache_clear(args))
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