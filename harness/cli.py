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

    cfg = _strip_comments(raw)
    validate_config_strict(cfg, source=path)

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
    "manifest_file", "redaction", "security", "skills", "deployment",
    "speculative", "impact", "lintgate", "logging", "languages",
    "test_generation", "metrics",
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
    }),
    "token_budget": frozenset({
        "hard_cap_usd", "context_window_threshold_pct",
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
}


# Per-field type schema used by validate_config_strict. Keys are dotted paths
# matching the structure in config.json. A value's runtime type must be in
# the listed tuple; bool is excluded from int matches via an explicit check
# because Python's bool is a subclass of int.
_TYPE_SCHEMA: dict[str, tuple[type, ...]] = {
    "build_command": (str,),
    "allow_network": (bool,),
    "manifest_file": (str,),
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
    "token_budget.hard_cap_usd": (int, float),
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

# Providers that DON'T need an API key env var (run locally / on-host).
# Anything else is treated as remote and gated on {PROVIDER}_API_KEY.
_LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama"})


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

    # --- 5. Env var presence for every model referenced by routing ---
    referenced_models: set[str] = set()
    for field in (*_REQUIRED_ROUTING_FIELDS, *_OPTIONAL_ROUTING_FIELDS):
        val = routing.get(field, "")
        if isinstance(val, str) and val.strip() and val in models:
            referenced_models.add(val)

    missing_env: dict[str, list[str]] = {}  # env_var → [model_keys needing it]
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
            continue
        if provider.lower() in _LOCAL_PROVIDERS:
            continue
        env_var = f"{provider.upper()}_API_KEY"
        if not os.environ.get(env_var, "").strip():
            missing_env.setdefault(env_var, []).append(model_key)

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


def _detect_default_build_command(workspace_path: str) -> Optional[str]:
    """Pick a sensible build command by sniffing workspace markers.

    Returns None when the workspace gives no hint — caller falls back to
    the historical default. Probed in priority order so a polyglot repo
    with a Makefile still uses it.
    """
    if not workspace_path or not os.path.isdir(workspace_path):
        return None

    def has(name: str) -> bool:
        return os.path.exists(os.path.join(workspace_path, name))

    if has("Makefile") or has("makefile") or has("GNUmakefile"):
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
    Multi-question streaming interface for exhaustive discovery phases.
    
    Reads state["discovery_questions"] (JSON from requirements/architecture discovery node),
    displays questions grouped by engineering modules, collects answers,
    and routes back to the discovery node for evaluation.
    
    Type 'DONE' to attempt finalization. If critical unknowns remain, the loop
    refuses to exit and displays [CRITICAL UNKNOWN DETECTED].
    """

    gate = state.get("current_gate", "REQUIREMENTS")
    discovery_data = state.get("discovery_questions", {})
    modules = discovery_data.get("modules", [])
    messages = list(state.get("messages", []))
    node_state = state.get("node_state", {})
    critical_remaining = node_state.get("discovery_critical_remaining", 0)
    complete = node_state.get("discovery_complete", False)
    round_num = node_state.get("discovery_question_count", 0)

    phase_label = "REQUIREMENTS" if gate == "REQUIREMENTS" else "ARCHITECTURE"

    if complete:
        logger.info("[discovery] %s discovery complete. Proceeding.", phase_label)
        return {"messages": messages, "node_state": node_state}

    # Display the header
    print()
    print("=" * 80)
    print(f"[HARNESS ARCHITECT SYSTEM AUDIT: {phase_label} PHASE] — Round {round_num}")
    print("=" * 80)
    print("The Architect has compiled a list of critical structural questions to eliminate all unknowns:")
    print()

    for module in modules:
        mod_name = module.get("name", "Module")
        questions = module.get("questions", [])
        if not questions:
            continue
        print(f"[MODULE: {mod_name}]")
        for q in questions:
            qid = q.get("id", "?")
            text = q.get("text", "")
            critical_marker = " **CRITICAL**" if q.get("critical") else ""
            print(f"  - {qid}:{critical_marker} {text}")
        print()

    if critical_remaining > 0:
        print(f"[CRITICAL]: {critical_remaining} critical question(s) remain unanswered.")
    print("-" * 80)
    print("Type your answers (referencing question numbers if preferred), 'DONE' to finalize, or 'SUSPEND' to save & quit.")
    print("-" * 80)

    from harness.hitl import get_channel as _get_channel
    response = _get_channel().notes("User Response")

    if response.upper() == "SUSPEND":
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
        logger.info("[discovery] %s phase suspended by developer. Session: %s", phase_label, session_id)
        node_state["hitl_suspend"] = True
        return {"messages": messages, "node_state": node_state}

    if response.upper() == "DONE":
        if critical_remaining > 0:
            # Refuse to exit with critical unknowns
            print()
            print("=" * 60)
            print(f"[CRITICAL UNKNOWN DETECTED]: {critical_remaining} critical question(s) still require answers.")
            print("You must specify the remaining variables before this phase can be finalized.")
            print("=" * 60)
            print()
            # Set flag so the router knows user tried to skip
            node_state["user_done_with_critical"] = True
            return {
                "messages": messages,
                "node_state": node_state,
            }
        else:
            node_state["discovery_complete"] = True
            logger.info("[discovery] User finalized %s phase. All questions resolved.", phase_label)
            print("[Discovery] All questions resolved. Finalizing specification...")
            return {
                "messages": messages,
                "node_state": node_state,
            }

    if not response:
        # Empty input, loop again
        return {"messages": messages, "node_state": node_state}

    # Append user's answers to conversation
    messages.append({"role": "user", "content": f"[Discovery Response - {phase_label}]: {response}"})
    node_state["discovery_complete"] = False  # Will be re-evaluated by discovery node
    logger.info("[discovery] Received user response (%d chars). Routing back for evaluation.", len(response))

    return {
        "messages": messages,
        "node_state": node_state,
    }


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
            # Reset total_repairs to 2 so route_after_compiler allows one more repair_node pass
            state["loop_counter"] = {"patching": 0, "repair": 0, "compiler": 0, "total_repairs": 2}
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
                # Reset loop counter to give AI another fresh attempt
                state["loop_counter"] = {"patching": 0, "repair": 0, "compiler": 0, "total_repairs": 1}
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
            # Reset loop counter and clear compiler errors since manual fix was applied
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
            # Reset loop counter to give the repair loop a full fresh cycle
            state["loop_counter"] = {"patching": 0, "repair": 0, "compiler": 0, "total_repairs": 0}
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
and code blocks where appropriate. Do not include any text outside the document."""


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
reference them by FR-id when justifying a design decision."""


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


def _attempt_git_rollback(workspace_path: str) -> None:
    """Attempt a git checkout to restore modified files to their original state."""
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
        harness run -r ./myproject -p "Refactor the auth module" --manifest notes.txt
    """
    workspace_path = os.path.abspath(args.workspace)
    if not os.path.isdir(workspace_path):
        logger.error("Workspace path does not exist: %s", workspace_path)
        return 1

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

    # Initialize GitGuardian for branch lifecycle management
    from harness.security import GitGuardian
    git_guardian = GitGuardian(workspace_path)
    git_guardian.stash_if_dirty()
    git_guardian.create_patch_branch(session_id)

    # --- Requirement Refinement Layer (product_spec.txt auto-discovery or --manifest override) ---
    spec_override: Optional[str] = None

    # Resolve the manifest file path:
    #   1. --manifest flag (explicit override, highest priority)
    #   2. Auto-discovered product_spec.txt in workspace root (convention)
    #   3. None — proceed with prompt-only execution
    manifest_path: Optional[str] = None
    if args.manifest:
        manifest_path = os.path.abspath(args.manifest)
        if not os.path.isfile(manifest_path):
            logger.error("[requirements] Explicit manifest file not found: %s", manifest_path)
            return 1
        logger.info("[requirements] Using explicit manifest: %s", manifest_path)
    else:
        # Auto-discovery: look for product_spec.txt in workspace root
        manifest_file = config.get("manifest_file", "product_spec.txt")
        auto_manifest = os.path.join(workspace_path, manifest_file)
        if os.path.isfile(auto_manifest):
            manifest_path = auto_manifest
            logger.info("[requirements] Auto-discovered product spec: %s", manifest_path)

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
            if doc_reviewer_primary:
                logger.info(
                    "[requirements] doc_reviewer_primary=%s configured — running pre-flight spec review.",
                    doc_reviewer_primary,
                )
                from harness.graph import review_and_revise_spec
                review_result = await review_and_revise_spec(
                    spec_path,
                    "REQUIREMENTS",
                    gateway=gateway,
                    budget_remaining_usd=budget_usd,
                    user_goal=args.prompt or "",
                )
                if review_result["ok"] and review_result.get("review_path"):
                    logger.info(
                        "[requirements] Review written to %s; spec revised in place.",
                        review_result["review_path"],
                    )
                    budget_usd = review_result["new_budget_usd"]
            else:
                logger.info(
                    "[requirements] doc_reviewer_primary not configured — skipping pre-flight spec review."
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
                    if doc_reviewer_primary:
                        logger.info(
                            "[architecture] doc_reviewer_primary=%s configured — running architecture spec review.",
                            doc_reviewer_primary,
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
                                "[architecture] Review written to %s; spec revised in place.",
                                arch_review_result["review_path"],
                            )
                            budget_usd = arch_review_result["new_budget_usd"]
                    else:
                        logger.info(
                            "[architecture] doc_reviewer_primary not configured — skipping architecture spec review."
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
    else:
        logger.info("[requirements] No product spec file found. Place '%s' at the workspace root with your product requirements, or use --manifest to specify an alternate file.",
                     config.get("manifest_file", "product_spec.txt"))

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
            sandbox_config=config.get("sandbox", {}),
            test_generation_config=config.get("test_generation", {}),
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

    # Git lifecycle: commit on success, rollback on failure
    if exit_code == 0:
        git_guardian.commit_all_changes(session_id, modified_files, exit_code)
        git_guardian.restore_original_branch()
    else:
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
            lintgate_config=config.get("lintgate", {}),
            deployment_config=config.get("deployment", {}),
            sandbox_config=config.get("sandbox", {}),
            test_generation_config=config.get("test_generation", {}),
        )
    except Exception:
        logger.exception("Resume execution failed.")
        await checkpointer.conn.close()
        return 1

    exit_code = final_state.get("exit_code", -1)
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
            ("api keys (live)", api_keys_result),
            ("sandbox backend", _doctor_check_sandbox(config)),
            ("checkpoint db", _doctor_check_checkpoint_db(config)),
        ])
    else:
        # Config invalid → mark downstream checks as skipped so the
        # operator sees they exist but understands they can't run yet.
        skipped_detail = "skipped — fix the config check above first"
        checks.extend([
            ("api keys (live)", ("skip", skipped_detail)),
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
        "--manifest", "-m",
        default=None,
        help="Path to a raw notes/text file to synthesize into SPEC_REQUIREMENTS.md before execution.",
    )
    run_parser.add_argument(
        "--output-dir", "-o",
        default="./docs",
        help="Directory to write SPEC_REQUIREMENTS.md (default: ./docs).",
    )
    run_parser.add_argument(
        "--workspace", "-w", "-r",
        required=True,
        help="Absolute or relative path to the target repository root.",
    )
    run_parser.add_argument(
        "--prompt", "-p",
        required=True,
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