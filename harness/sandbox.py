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
import logging
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union


# Credential scrubbing is centralised in harness/trust.py so every
# subprocess-spawner uses the same allowlist.
from harness.trust import SCRUBBED_BUILD_ENV_VARS as _SCRUBBED_BUILD_ENV_VARS  # noqa: E402

# OS-dispatch primitives — see harness/_platform.py docstring.
from harness import _platform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Teane pytest diagnostics plugin — Layer 2 of the diagnostic enrichment
# stack (Layer 1 is parser-side ``--showlocals`` capture). This plugin
# lives inside the harness source tree at
# ``harness/pytest_plugins/teane_diagnostics.py`` and is exposed to every
# sandboxed build via env vars — no workspace-side file needed.
# ---------------------------------------------------------------------------

_TEANE_PLUGIN_HOST_DIR: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "pytest_plugins")
)
_TEANE_PLUGIN_MODULE_NAME: str = "teane_diagnostics"
# Container-side mount target for Docker. Chosen under /opt so it can't
# collide with a workspace directory the operator may already have.
_TEANE_PLUGIN_CONTAINER_DIR: str = "/opt/teane_pytest_plugins"


def _teane_diagnostics_available() -> bool:
    """True when the plugin file is on disk. False in stripped-down
    installs (single-file bundles, minimal test harnesses) — cleanly
    disables the injection without a hard failure."""
    return os.path.isfile(
        os.path.join(_TEANE_PLUGIN_HOST_DIR, f"{_TEANE_PLUGIN_MODULE_NAME}.py")
    )


def _teane_diagnostics_injection_enabled() -> bool:
    """Harness-side kill switch. Off when the operator sets
    ``TEANE_DIAGNOSTICS_INJECT`` on the host process to a falsy value
    (``0`` / ``off`` / ``false`` / ``no`` / ``disabled``). Default: on.

    This is distinct from the plugin's own escape hatches (which run
    INSIDE the sandbox, after the plugin has already been imported). If
    the plugin file itself somehow becomes broken (bad edit, stripped
    install, incompatible pytest version), the operator can flip this
    env var and every backend stops injecting the plugin path — the
    sandbox goes back to pre-Layer-2 behaviour without touching config
    files or reinstalling.
    """
    raw = os.environ.get("TEANE_DIAGNOSTICS_INJECT", "").strip().lower()
    return raw not in {"0", "off", "false", "no", "disabled"}


def _apply_teane_diagnostics_env(
    env: dict[str, str], container_path: str,
) -> None:
    """Extend ``env`` in place with the two vars pytest needs to find and
    load the teane_diagnostics plugin. Called AFTER the defaults →
    extra_env merge so the addition composes with any operator-supplied
    ``PYTHONPATH`` / ``PYTEST_ADDOPTS`` rather than being clobbered by
    it. See the DockerBackend + BareBackend wiring for the pattern.

    * ``PYTHONPATH`` — prepend so ``import teane_diagnostics`` resolves
      first; the operator's path suffix remains reachable.
    * ``PYTEST_ADDOPTS`` — append (with a leading space) so operator
      addopts (e.g. speculative's ``-o cache_dir=...``) come first and
      the plugin flag composes at the tail. Idempotent — a second call
      won't inject the ``-p`` flag twice.

    Callers must gate on ``_teane_diagnostics_available()`` and
    ``_teane_diagnostics_injection_enabled()`` BEFORE invoking this
    helper. Both are cheap and orthogonal, but bundling them here would
    hide the intent at the call site.
    """
    existing_pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{container_path}:{existing_pypath}" if existing_pypath else container_path
    )
    existing_addopts = env.get("PYTEST_ADDOPTS", "")
    if f"-p {_TEANE_PLUGIN_MODULE_NAME}" not in existing_addopts:
        env["PYTEST_ADDOPTS"] = existing_addopts + f" -p {_TEANE_PLUGIN_MODULE_NAME}"


# The harness's kitchen-sink builder image. One image with every toolchain
# the supported stacks need (Python 3.12 uv-managed + pip + venv + uv + pytest +
# pytest-cov + pytest-xdist, Java JDK 21 + Maven + Gradle, Node 20 LTS +
# npm + yarn + pnpm + tsc + jest + ts-jest, SQLite, Playwright +
# Chromium, plus make/gcc/git/curl). Built from
# ``harness/vendor/Dockerfile.builder``. Pinning here means the sandbox
# never needs to swap images or apt-get inside the container at runtime —
# the latter is impossible under ``--user $UID:$GID`` mode anyway.
#
# Pinned to the content-addressable manifest digest (set by buildx during
# the local build) so every operator runs against bit-for-bit the same
# image. Rebuild + re-pin recipe:
#   docker build --pull \
#     -f harness/vendor/Dockerfile.builder \
#     -t harness-builder:latest \
#     -t harness-builder:$(date +%Y-%m-%d) \
#     harness/vendor/
#   docker inspect harness-builder:latest --format '{{.RepoDigests}}'
# Then paste the new digest below. The image is local-only by default;
# a registry push (e.g. to ghcr.io/<owner>/harness-builder) is optional
# and only required for multi-host fleets.
BUILDER_IMAGE = (
    "harness-builder"
    "@sha256:92f163b2817a13cda603d93d9b34686e4b4f1fae6cfe907f2dccd208a1bcad19"
)


# ---------------------------------------------------------------------------
# 1. Types
# ---------------------------------------------------------------------------

@dataclass
class FixSuggestion:
    """
    A compiler-emitted machine-applicable fix for a diagnostic.

    Populated by parsers that read structured diagnostic output and lift
    a machine-applicable replacement span. Consumed by harness.autofix to
    apply the fix without spending an LLM call.

    Spans are 1-indexed (matches the compiler's own coordinate system).
    """
    replacement: str             # exact text to substitute
    span_start_line: int         # 1-indexed
    span_start_col: int          # 1-indexed
    span_end_line: int
    span_end_col: int
    applicability: str           # "machine-applicable" | "maybe-incorrect" | "unspecified"


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
    # Full pytest node id (``tests/foo.py::TestBar::test_baz``) when the
    # diagnostic was extracted from a pytest FAILED summary row. Populated
    # by ``PythonParser._parse_pytest_summary`` and consumed by
    # ``compiler_node``'s isolation re-run — a single anomalous test that
    # PASSES on its own but FAILS in the full suite is the fingerprint of
    # shared-state pollution (module-level singletons, ``@lru_cache``
    # across event loops, class-level fixtures). The re-run needs the
    # nodeid to invoke only that test. Empty for non-pytest diagnostics.
    pytest_nodeid: str = ""
    suggested_fix: Optional["FixSuggestion"] = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "severity": self.severity,
            "error_code": self.error_code,
            "message": self.message,
            "semantic_context": self.semantic_context,
        }
        if self.pytest_nodeid:
            out["pytest_nodeid"] = self.pytest_nodeid
        if self.suggested_fix is not None:
            out["suggested_fix"] = {
                "replacement": self.suggested_fix.replacement,
                "span_start_line": self.suggested_fix.span_start_line,
                "span_start_col": self.suggested_fix.span_start_col,
                "span_end_line": self.suggested_fix.span_end_line,
                "span_end_col": self.suggested_fix.span_end_col,
                "applicability": self.suggested_fix.applicability,
            }
        return out


@dataclass
class BuildResult:
    """Result of a sandboxed build execution."""
    exit_code: int
    raw_output: str
    diagnostics: list[DiagnosticObject] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    timed_out: bool = False
    # True when the disk log streamer hit its byte cap. raw_output is
    # incomplete in this case — diagnostic parsing may miss the real error
    # if it occurred after the cap was hit. Surfaced so downstream nodes
    # can warn the user instead of treating the truncated tail as ground truth.
    log_truncated: bool = False
    # The COMPLETE captured log (all stdout, then all stderr — streamer
    # concatenation order), untouched by filter_critical_errors. On failure
    # ``raw_output`` is the filtered view, and the filter's no-critical-match
    # fallback ("last 50 lines") can reduce the log to installer chatter when
    # the runner's report matches no critical pattern (finsearch b674f3ca:
    # pytest fixture errors — "ERROR at setup of", "E fixture 'x' not
    # found" — match nothing in _CRITICAL_ERROR_PATTERNS). Failure-surface
    # extraction in compiler_node must scan THIS field, never raw_output.
    full_output: str = ""


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
    ) -> tuple[int, str, bool, bool]:
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
        if not _platform.is_linux():
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
    ) -> tuple[int, str, bool, bool]:
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

        # Build inner shell commands: bind-mount caches → cd workspace → execute build.
        # Paths go through shlex.quote so operator-supplied paths
        # containing apostrophes or spaces don't escape the single-quoted
        # context. Audit §3.11.
        import shlex as _shlex
        inner_commands: list[str] = []

        for cache_path in cache_mounts:
            expanded = os.path.expanduser(cache_path)
            if os.path.isdir(expanded):
                qe = _shlex.quote(expanded)
                inner_commands.append(f"mkdir -p {qe} 2>/dev/null || true")
                inner_commands.append(
                    f"mount --bind -o ro {qe} {qe} 2>/dev/null || true"
                )

        inner_commands.append(f"cd {_shlex.quote(workspace_path)}")
        inner_commands.append(shell_cmd)

        inner_script = " && ".join(inner_commands)
        ns_args.extend(["--", "sh", "-c", inner_script])
        return ns_args


# ---------------------------------------------------------------------------
# 4. DockerBackend — Docker Container Isolation
# ---------------------------------------------------------------------------

# Substrings that strongly suggest a writable cache (named volume or host
# cache) has corrupted entries — pip wheel-hash mismatches, npm cacache
# integrity failures. When any of these turn up in the build output we append
# a one-line "try clearing the cache" hint to the BuildResult so the operator
# (and any LLM repair loop reading the transcript) doesn't burn a debugging
# cycle on a recoverable corruption.
# Match against lowercased output to be case-tolerant. Over-triggering is
# safe — a stray hint is one line; missing a real signature wastes minutes.
_CACHE_CORRUPTION_SIGNATURES: tuple[str, ...] = (
    # pip / pip-tools
    "these packages do not match the hashes from the requirements file",
    "could not match the hash",
    "is not a known hash",
    # npm / cacache
    "cacache: integrity check failed",
    "eintegrity",
    "sha512-",  # paired with "eintegrity" usually; weak signal alone but kept conservative below
)
# Tighter set used when the only hit is "sha512-": we require the npm-specific
# error wrapper to also be present, since sha512- appears in benign output too.
_NPM_INTEGRITY_PAIR = ("sha512-", "eintegrity")


def _cache_corruption_hint(raw_output: str) -> Optional[str]:
    """Return a one-line hint when the build output looks cache-corrupted,
    or None otherwise. Caller appends to BuildResult.raw_output."""
    if not raw_output:
        return None
    lower = raw_output.lower()
    hit = False
    for sig in _CACHE_CORRUPTION_SIGNATURES:
        if sig == "sha512-":
            continue  # handled below — requires the EINTEGRITY pair
        if sig in lower:
            hit = True
            break
    if not hit and all(s in lower for s in _NPM_INTEGRITY_PAIR):
        hit = True
    if not hit:
        return None
    return (
        "\n[sandbox-hint] Build output contains a cache-corruption signature "
        "(hash mismatch / integrity failure / corrupt registry index). If you "
        "have `sandbox.cache_volumes` enabled, try `teane cache clear` "
        "(optionally `--session-id <id>`) and rerun. If you don't, your host "
        "cache (~/.cache/pip, ~/.npm) may be damaged — clear the "
        "affected tool's cache directory."
    )


def _cache_volume_name(
    cache_path: str,
    session_id: Optional[str],
    prefix: str = "harness",
) -> str:
    """Derive a deterministic, host-stable, session-namespaced Docker volume
    name from a read-only cache mount path.

    Operators configure ``sandbox.readonly_cache_mounts`` with tool-specific
    paths (``~/.cache/pip``, ``~/.npm``). When
    ``sandbox.cache_volumes`` is on, we swap each read-only host bind for a
    writable named volume so the tool can persist downloaded wheels /
    tarballs / crates back across containers. Volume names are derived from
    the basename of the cache path so the volume's purpose is greppable from
    ``docker volume ls``. Session id is appended to scope reuse — variant 1's
    typo-installed package can't poison a different operator's session.

    ``~/.cache/pip`` collapses its basename to ``pip`` rather than ``cache``
    (the parent dir name is more informative for the tool-specific mounts the
    operator actually configures). Empty / missing session id falls back to
    ``global`` so callers that don't pass one still get a stable name.
    """
    expanded = os.path.expanduser(cache_path).rstrip(os.sep)
    base = os.path.basename(expanded) or "cache"
    if base == "cache":
        parent = os.path.basename(os.path.dirname(expanded)) or "cache"
        base = parent.lstrip(".") or "cache"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", base.lstrip(".")).strip("-") or "cache"
    sid = (session_id or "global").strip() or "global"
    sid_slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", sid).strip("-") or "global"
    return f"{prefix}-{slug}-{sid_slug}"


def _docker_mount_path(p: str) -> str:
    """Normalise a filesystem path for use as a Docker ``-v`` / ``-w`` argument.

    Docker on Linux and macOS accepts the native path verbatim — this helper
    is a pure pass-through there, returning ``p`` byte-identical to its
    input so the existing Linux/macOS test argv-assertions continue to
    match unchanged.

    On Windows, ``docker run -v C:\\Users\\foo:C:\\Users\\foo:rw`` is
    ambiguous (the ``:`` in ``C:`` is also the host/container separator),
    so Docker Desktop's CLI rejects it. The canonical fix is to express
    the path in POSIX form (``/c/Users/foo``) and let Docker Desktop's
    bind-mount layer translate back. This helper does that conversion:
    lower-case the drive letter, strip the colon, prepend ``/``, and
    flip backslashes to forward slashes. ``C:\\Users\\foo`` becomes
    ``/c/Users/foo``; ``D:\\src\\app`` becomes ``/d/src/app``.

    Uses ``ntpath.splitdrive`` (not ``os.path.splitdrive``) so the
    Windows-style drive-letter split works even when this code is
    cross-tested from a Linux host — ``os.path.splitdrive`` dispatches
    based on the running platform and would return ``("", path)`` on a
    Linux test harness, missing the drive letter.
    """
    if platform.system() != "Windows":
        return p
    import ntpath
    drive, rest = ntpath.splitdrive(p)
    if drive.endswith(":"):
        return f"/{drive[:-1].lower()}{rest}".replace("\\", "/")
    return p.replace("\\", "/")


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
        - Workspace ownership restoration on exit (chown root-owned bind-mount
          files back to the host user, so __pycache__/ and friends don't end
          up needing sudo to delete on the host).
    """

    # Fixed in-container cache paths the builder image (Dockerfile.builder)
    # pre-creates world-writable. The image's ENV block also points
    # PIP_CACHE_DIR / UV_CACHE_DIR / npm_config_cache at these paths, so when
    # ``cache_volumes_enabled`` is on we just need to mount a named Docker
    # volume on each path — pip / uv / npm pick it up automatically without
    # the workspace having to configure anything. Independent of any
    # operator-supplied ``readonly_cache_mounts``: those still get their
    # own bind/volume mounts on top.
    _DEFAULT_CACHE_VOLUME_PATHS: tuple[str, ...] = (
        "/cache/pip",
        "/cache/uv",
        "/cache/npm",
    )

    def __init__(
        self,
        image: str = BUILDER_IMAGE,
        memory_limit: str = "512m",
        cpu_limit: str = "1.0",
        pids_limit: int = 100,
        docker_path: str = "docker",
        read_only_root: bool = True,
        restore_workspace_ownership: bool = True,
        cache_volumes_enabled: bool = True,
        cache_volumes_session_id: Optional[str] = None,
        cache_volumes_prefix: str = "harness",
    ):
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.pids_limit = pids_limit
        self.docker_path = docker_path
        # When True (default), mount writable named Docker volumes on
        # ``_DEFAULT_CACHE_VOLUME_PATHS`` (/cache/pip, /cache/uv, /cache/npm)
        # so pip / uv / npm downloads persist across containers. Volume names
        # are derived from the basename of each mount path (see
        # ``_cache_volume_name``); ``cache_volumes_session_id`` optionally
        # namespaces them — when left ``None`` (the default for global
        # sharing) the slug collapses to ``global`` and every session reuses
        # the same wheels / tarballs. Any operator-supplied
        # ``readonly_cache_mounts`` are still mounted on top with the same
        # rule.
        self.cache_volumes_enabled = cache_volumes_enabled
        self.cache_volumes_session_id = cache_volumes_session_id
        self.cache_volumes_prefix = cache_volumes_prefix
        # Track which named volumes we've already ensured to exist this
        # process — `docker volume inspect` is faster than `docker volume
        # create`, but each call is still a fork/exec. Memoise so the second
        # variant in a session doesn't pay the cost.
        self._ensured_volumes: set[str] = set()
        # When True (default) the container's root FS is mounted read-only and
        # only /tmp is writable. Setting this to False is required for builds
        # that install packages into system locations (pip install -e .,
        # npm install -g) because pip's --user fallback writes
        # to /root/.local which is *also* on the read-only root FS. The
        # container is --rm so dropping read-only does not leak state.
        self.read_only_root = read_only_root
        # When True (default on Linux when the host user is non-root) we
        # append a `find -uid 0 -exec chown <uid>:<gid>` trailer to the shell
        # entrypoint so any files the in-container build wrote as root
        # (notably pytest's __pycache__/) land owned by the host user via the
        # bind-mount. Set to False to opt out — useful with rootless docker /
        # podman where the user-namespace remapping already handles ownership.
        self.restore_workspace_ownership = restore_workspace_ownership

    @property
    def name(self) -> str:
        return f"docker({self.image})"

    def is_available(self) -> bool:
        """
        Check if Docker is installed AND the daemon is reachable by *this*
        user. Distinguishes three failure shapes so users debugging an
        unexpected fallback to unshare/bare see the real reason:

          - binary missing            → silent False (expected)
          - daemon not running        → logged warning
          - permission denied         → logged error with suggested fix
        """
        if not shutil.which(self.docker_path):
            return False
        try:
            result = subprocess.run(
                [self.docker_path, "info"],
                capture_output=True,
                timeout=10,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning("[sandbox] Docker availability check failed: %s", e)
            return False

        if result.returncode == 0:
            return True

        # docker info failed — surface why. stderr typically contains:
        #   "permission denied while trying to connect to the Docker daemon socket"
        #   "Cannot connect to the Docker daemon at unix:///var/run/docker.sock"
        stderr = (result.stderr or "").lower()
        if "permission denied" in stderr:
            logger.error(
                "[sandbox] Docker is installed but the daemon socket is not accessible to this user. "
                "Add the user to the 'docker' group (`sudo usermod -aG docker $USER`) and re-login, "
                "or run the harness with sufficient privileges. Falling back to non-Docker backend."
            )
        elif "cannot connect" in stderr or "is the docker daemon running" in stderr:
            logger.warning(
                "[sandbox] Docker is installed but the daemon is not running. "
                "Start it (`sudo systemctl start docker`) or use Docker Desktop. "
                "Falling back to non-Docker backend."
            )
        else:
            logger.warning(
                "[sandbox] `docker info` failed (exit=%d). Falling back to non-Docker backend. "
                "stderr=%s",
                result.returncode,
                (result.stderr or "").strip()[:200],
            )
        return False

    async def run(
        self,
        command: str,
        workspace_path: str,
        timeout_seconds: int = 300,
        allow_network: bool = False,
        readonly_cache_mounts: Optional[list[str]] = None,
        extra_env: Optional[dict[str, str]] = None,
    ) -> tuple[int, str, bool, bool]:
        # Pre-create any named cache volumes for this run. Without this the
        # `--mount type=volume,source=...` in the docker run argv would
        # auto-create the volume implicitly, but auto-create silently
        # initialises an empty volume — we want explicit `docker volume
        # create` so failures (out of disk, daemon socket perms) surface
        # with their actual error message before the build starts.
        effective_mounts = list(readonly_cache_mounts or [])
        # Always layer the builder image's fixed /cache/* paths on top.
        # These need to reach _build_docker_command even when
        # ``cache_volumes_enabled`` is False, because the image bakes
        # ``ENV UV_CACHE_DIR=/cache/uv`` (etc.) and, without a writable
        # mount or a host bind, the env-fallback loop there is what
        # redirects them to /tmp/*-cache. Skipping this list entirely
        # (the previous behaviour when the flag was off) left the env
        # unchanged and every uv invocation crashed with a Read-only FS
        # error on ``/cache/uv/CACHEDIR.TAG`` — see the comment below at
        # ``_cache_paths_needing_env_fallback``.
        effective_mounts = [
            p for p in self._DEFAULT_CACHE_VOLUME_PATHS
            if p not in effective_mounts
        ] + effective_mounts
        if self.cache_volumes_enabled:
            self._ensure_cache_volumes(effective_mounts)
        docker_cmd = self._build_docker_command(
            command,
            workspace_path,
            allow_network,
            effective_mounts,
            extra_env or {},
            timeout_seconds,
        )
        logger.info("[sandbox:docker] Running in Docker container (image=%s, mem=%s).", self.image, self.memory_limit)
        logger.debug("[sandbox:docker] Command: %s", " ".join(docker_cmd))
        try:
            return await _execute_subprocess_with_timeout(docker_cmd, timeout_seconds)
        finally:
            # Belt-and-suspenders: if the in-container ownership-restore
            # trailer was skipped (container OOM-killed, SIGKILLed by the
            # outer timeout, or `restore_workspace_ownership=False`), sweep
            # the bind-mount on the host. Best-effort: silently exits if the
            # host user lacks CAP_CHOWN.
            #
            # Wrapped in run_in_executor so the synchronous ``find -exec
            # chown`` subprocess (timeout=30s) doesn't stall the asyncio
            # event loop on large workspaces. Audit §2.10.
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, self._host_side_ownership_sweep, workspace_path,
                )
            except Exception as sweep_exc:  # noqa: BLE001
                logger.debug("[sandbox] host-side sweep failed: %s", sweep_exc)

    def _ensure_cache_volumes(self, cache_mounts: list[str]) -> None:
        """Idempotently `docker volume create` each cache mount's named volume.

        Skips creation when the volume already exists (per the cached set or
        per `docker volume inspect`). First-call latency per volume is ~50ms
        on a warm daemon; subsequent calls hit the in-process memo set.
        """
        for cache_path in cache_mounts:
            expanded = os.path.expanduser(cache_path)
            volume = _cache_volume_name(
                expanded,
                self.cache_volumes_session_id,
                self.cache_volumes_prefix,
            )
            if volume in self._ensured_volumes:
                continue
            try:
                inspect = subprocess.run(
                    [self.docker_path, "volume", "inspect", volume],
                    capture_output=True,
                    timeout=10,
                )
                if inspect.returncode == 0:
                    self._ensured_volumes.add(volume)
                    continue
                create = subprocess.run(
                    [self.docker_path, "volume", "create", volume],
                    capture_output=True,
                    timeout=15,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                if create.returncode != 0:
                    logger.warning(
                        "[sandbox:docker] Failed to create cache volume %r: %s. "
                        "Falling back to no cache for this mount this run.",
                        volume, (create.stderr or "").strip()[:200],
                    )
                    continue
                logger.info(
                    "[sandbox:docker] Created cache volume %r for %s.",
                    volume, cache_path,
                )
                self._ensured_volumes.add(volume)
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
                logger.warning(
                    "[sandbox:docker] docker volume probe for %r failed: %s. "
                    "Falling back to no cache for this mount this run.",
                    volume, exc,
                )

    # In-container HOME used when --user passes a non-root UID. /tmp is the
    # writable tmpfs already provisioned at line 414, so the host user's HOME
    # lives there without a separate mount. Stable name so test assertions
    # and operators tailing logs can grep for it.
    _BUILDER_HOME = "/tmp/builder-home"

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
            "--tmpfs", "/tmp:exec",              # Writable /tmp for build artifacts
            f"--memory={self.memory_limit}",     # Memory limit
            f"--cpus={self.cpu_limit}",          # CPU limit
            f"--pids-limit={self.pids_limit}",   # Prevent fork bombs
            "--stop-timeout", str(max(5, timeout_seconds // 10)),  # Graceful stop
        ]

        # Run as the host user when possible. The container's processes
        # otherwise default to UID 0 (root), which (a) makes pip emit the
        # well-known "Running pip as the 'root' user can result in broken
        # permissions" warning and (b) is conceptually wrong — the harness
        # is meant to act on the user's behalf, not as root. We gate on
        # Linux + non-root host (macOS/Windows Docker Desktop already remap
        # ownership via the FUSE layer; root on root is a no-op).
        run_as_host_user = self._should_run_as_host_user()
        host_uid: Optional[int] = None
        host_gid: Optional[int] = None
        if run_as_host_user:
            host_uid = os.getuid()
            host_gid = os.getgid()
            cmd.extend(["--user", f"{host_uid}:{host_gid}"])

        if self.read_only_root:
            cmd.append("--read-only")
            if not run_as_host_user:
                # When the root FS is RO, pip / npm will try the
                # per-user fallback (~/.local, ~/.cache, ~/.npm). Without a
                # writable HOME those installs fail with "Read-only file
                # system: '/root/...'" *after* downloading every wheel. Give
                # them a tmpfs to land in. When running as the host user we
                # set HOME to the existing /tmp tmpfs (see env vars below),
                # so /root never gets written to.
                cmd.extend(["--tmpfs", "/root:exec"])

        # Network isolation
        if allow_network:
            cmd.extend(["--network", "bridge"])  # Docker default bridge network
        else:
            cmd.extend(["--network", "none"])     # Complete network isolation

        # Mount workspace read-write. _docker_mount_path is a pass-through
        # on Linux/macOS (string returned byte-identical) and converts the
        # path to POSIX form on Windows so Docker Desktop's CLI parser
        # doesn't choke on the ``:`` in ``C:\``.
        ws_mount = _docker_mount_path(workspace_path)
        cmd.extend(["-v", f"{ws_mount}:{ws_mount}:rw"])

        # Cache mounts. When cache_volumes is on, swap each :ro host bind for
        # a writable named Docker volume scoped to the session — the tool's
        # downloads persist across containers and the next compile in the
        # same session reuses them. Otherwise emit the historical read-only
        # host bind so behaviour is byte-for-byte unchanged when the flag is
        # off. We only emit volume mounts that we successfully ensured in
        # _ensure_cache_volumes; failed-to-create volumes fall back to the
        # read-only host bind so the build can still cold-fill from the host
        # cache (better than no cache at all).
        # Track cache paths that did NOT get a writable volume mount so
        # we can inject fallback ``*_CACHE_DIR`` env vars pointing at
        # /tmp/*-cache. Without this, an unavailable /cache/uv (volume-
        # create failed AND no host bind path) leaves the builder image's
        # ENV UV_CACHE_DIR=/cache/uv pointing at the read-only root FS,
        # and the LLM burns entire repair loops trying to redirect uv via
        # pyproject.toml edits (session cec4d124, 2026-07-07 — 45 rounds
        # against an empty pyproject.toml).
        _cache_paths_needing_env_fallback: list[str] = []
        for cache_path in cache_mounts:
            expanded = os.path.expanduser(cache_path)
            expanded_mount = _docker_mount_path(expanded)
            if self.cache_volumes_enabled:
                volume = _cache_volume_name(
                    expanded,
                    self.cache_volumes_session_id,
                    self.cache_volumes_prefix,
                )
                if volume in self._ensured_volumes:
                    cmd.extend([
                        "--mount",
                        f"type=volume,source={volume},target={expanded_mount}",
                    ])
                    continue
                # ensure failed → fall through to the read-only host bind.
            if os.path.isdir(expanded):
                cmd.extend(["-v", f"{expanded_mount}:{expanded_mount}:ro"])
            else:
                # Neither a writable named volume nor a readable host bind.
                # The path is on the container's read-only root FS, so any
                # tool that tries to write there fails opaquely. Queue an
                # env override.
                _cache_paths_needing_env_fallback.append(expanded)

        # Set working directory (same Windows path conversion as the mount).
        cmd.extend(["-w", _docker_mount_path(workspace_path)])

        # Environment variables. Default suppression of pyc emission into the
        # bind-mounted workspace — pytest otherwise leaves root-owned
        # __pycache__/ trees the host user can't rm without sudo. Defaults
        # are merged UNDER extra_env so the speculative path's per-variant
        # PYTHONPYCACHEPREFIX (see speculative._build_variant_cache_env) still
        # takes precedence. When running as the host user we also need to
        # redirect HOME (the container's /etc/passwd usually only has root
        # set, so a non-root UID has no resolvable HOME) and pre-route pip
        # into per-user mode so the existing build command `pip install X`
        # works without further command rewriting.
        defaults = self._default_pyc_env()
        if run_as_host_user:
            defaults.update(self._default_user_mode_env())
        # Cache-fallback env vars for any /cache/* path that didn't get a
        # writable mount. See loop above.
        defaults.update(
            self._cache_fallback_env(_cache_paths_needing_env_fallback)
        )
        merged_env = {**defaults, **extra_env}
        # Teane pytest diagnostics plugin — Layer 2. Injects POST-MERGE so
        # our contributions compose with (not clobber) any operator env
        # override — e.g. speculative sets ``PYTEST_ADDOPTS=-o cache_dir=...``
        # per variant, and we need our ``-p teane_diagnostics`` to
        # coexist. Bind-mounts the plugin dir read-only into the container.
        # Skipped entirely when the plugin file isn't on disk (stripped
        # installs) or the operator turned off injection via env
        # ``TEANE_DIAGNOSTICS_INJECT=off`` (host-side kill switch). The
        # plugin is inert against non-pytest builds (Java, npm) — pytest
        # never runs to load it — so cost of the injection is one -v arg
        # and two env vars per sandbox invocation.
        if _teane_diagnostics_available() and _teane_diagnostics_injection_enabled():
            cmd.extend([
                "-v",
                f"{_docker_mount_path(_TEANE_PLUGIN_HOST_DIR)}"
                f":{_TEANE_PLUGIN_CONTAINER_DIR}:ro",
            ])
            _apply_teane_diagnostics_env(merged_env, _TEANE_PLUGIN_CONTAINER_DIR)
        for key, value in merged_env.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Image and entrypoint. When running as a non-root UID we have to
        # ensure ``$HOME`` exists before the build command runs, because
        # pip's per-user install path is computed from HOME and pip will
        # crash with "Could not find an activated virtualenv (required)" or
        # similar if HOME doesn't exist on disk. ``mkdir -p`` is cheap and
        # idempotent.
        wrapped_cmd = shell_cmd
        if run_as_host_user:
            wrapped_cmd = f'mkdir -p "$HOME" && ( {wrapped_cmd} )'

        # When the container runs as root we additionally wrap with an
        # ownership-restoring trailer so any files the in-container root
        # process wrote into the bind-mount land owned by the host user.
        # When we already passed --user to docker every write is host-owned
        # from the start and the trailer becomes a redundant find walk;
        # skip it to save the (typically small but non-zero) scan time.
        if not run_as_host_user:
            wrapped_cmd = self._wrap_shell_cmd_with_ownership_restore(
                wrapped_cmd, workspace_path,
            )

        cmd.append(self.image)
        cmd.extend(["sh", "-c", wrapped_cmd])

        return cmd

    def _should_run_as_host_user(self) -> bool:
        """True when the container should be launched with
        ``--user $UID:$GID`` instead of the image's default UID 0.

        Gated on:
          - the operator hasn't opted out via the same
            ``restore_workspace_ownership=False`` switch (single config knob
            for "behave like the host user");
          - the host is Linux (Docker Desktop on macOS / Windows already
            remaps ownership via FUSE, no benefit to switching UIDs);
          - the host user is non-root (root → root in container is the
            current behaviour and avoids surprising permission failures
            for operators running the harness in CI containers).
        """
        if not self.restore_workspace_ownership:
            return False
        if platform.system() != "Linux":
            return False
        getuid = getattr(os, "getuid", None)
        getgid = getattr(os, "getgid", None)
        if getuid is None or getgid is None:
            return False
        return getuid() != 0

    def _default_user_mode_env(self) -> dict[str, str]:
        """Env vars needed when the container runs as a non-root UID with
        no matching entry in the image's /etc/passwd.

        - ``HOME`` points at the writable /tmp tmpfs so pip / npm
          have somewhere to write per-user caches and installed packages.
        - ``PIP_USER=1`` flips ``pip install`` to per-user install mode
          (writes to ``$HOME/.local/lib/...`` instead of
          ``/usr/local/lib/python*/site-packages`` which would EACCES).
        - ``PIP_ROOT_USER_ACTION=ignore`` silences pip's noisy
          "running as the 'root' user" warning when the build does
          occasionally land back on root (e.g. via sudo).
        - ``PATH`` is prefixed with ``$HOME/.local/bin`` so the entry-point
          scripts pip installs there (``pytest``, ``ruff``, ``mypy``) are
          found by subsequent steps in the same build command.
        """
        home = self._BUILDER_HOME
        return {
            "HOME": home,
            "PIP_USER": "1",
            "PIP_ROOT_USER_ACTION": "ignore",
            "PATH": f"{home}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }

    def _host_side_ownership_sweep(self, workspace_path: str) -> None:
        """Best-effort host-side fallback for the in-container chown trailer.

        The in-container trailer (see :meth:`_wrap_shell_cmd_with_ownership_restore`)
        catches the common path. This sweep covers the rest: container OOM
        kills, SIGKILL from the outer timeout, or operators who explicitly
        disabled ``restore_workspace_ownership``. Run as a subprocess so we
        don't block the asyncio loop; swallow every failure path because the
        host user often *can't* chown root-owned files (would need
        ``CAP_CHOWN``), and that's the expected case — the in-container
        trailer is the real defence.
        """
        if not _platform.is_linux():
            return
        if not workspace_path or not os.path.isdir(workspace_path):
            return
        getuid = getattr(os, "getuid", None)
        getgid = getattr(os, "getgid", None)
        if getuid is None or getgid is None:
            return
        uid = getuid()
        gid = getgid()
        if uid == 0:
            return
        try:
            subprocess.run(
                ["find", workspace_path, "-uid", "0", "-exec",
                 "chown", f"{uid}:{gid}", "{}", "+"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    @staticmethod
    def _default_pyc_env() -> dict[str, str]:
        """Default env vars that prevent pytest / Python from writing
        ``__pycache__/*.pyc`` next to source files inside the bind-mounted
        workspace. Both vars are belt-and-braces:
          - ``PYTHONDONTWRITEBYTECODE=1`` suppresses pyc emission entirely.
          - ``PYTHONPYCACHEPREFIX=/tmp/pycache`` redirects any pyc that does
            get emitted (e.g. by a sub-interpreter that ignores the first var)
            into the container's writable tmpfs.
        Callers can override either by passing the same key in ``extra_env``;
        the merge in :meth:`_build_docker_command` makes ``extra_env`` win.
        """
        return {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": "/tmp/pycache",
        }

    # Fallback cache-dir env-var mapping for /cache/* paths that couldn't
    # be mounted writable. The builder image's ENV sets each tool's cache
    # var to the corresponding /cache/* path — when that path lands on
    # the read-only root FS (no volume, no host bind), we must override
    # to a writable tmpfs path or the tool errors out opaquely.
    _CACHE_PATH_ENV_MAP: dict[str, tuple[str, str]] = {
        "/cache/pip": ("PIP_CACHE_DIR", "/tmp/pip-cache"),
        "/cache/uv": ("UV_CACHE_DIR", "/tmp/uv-cache"),
        "/cache/npm": ("npm_config_cache", "/tmp/npm-cache"),
    }

    @classmethod
    def _cache_fallback_env(cls, missing_cache_paths: list[str]) -> dict[str, str]:
        """Return env-var overrides for cache paths that didn't get a
        writable mount. See ``_CACHE_PATH_ENV_MAP`` for the mapping.

        Absent input → empty dict. Unknown paths are ignored (no mapping
        entry). Callers merge the result into ``defaults`` at build time
        so ``extra_env`` still wins.
        """
        out: dict[str, str] = {}
        for path in missing_cache_paths or ():
            entry = cls._CACHE_PATH_ENV_MAP.get(path)
            if entry is None:
                continue
            var, fallback = entry
            out[var] = fallback
        return out

    def _wrap_shell_cmd_with_ownership_restore(
        self, shell_cmd: str, workspace_path: str,
    ) -> str:
        """Wrap ``shell_cmd`` so that on exit (success OR failure), any files
        owned by uid 0 inside the bind-mounted workspace are chowned back to
        the host user's uid:gid.

        Only fires on Linux when the host user is non-root and the backend was
        constructed with ``restore_workspace_ownership=True``. On macOS /
        Windows the Docker bind-mount FUSE layer already remaps ownership; on
        a host running as root no remap is needed. In all those cases the
        original ``shell_cmd`` is returned unchanged.
        """
        if not self.restore_workspace_ownership:
            return shell_cmd
        if platform.system() != "Linux":
            return shell_cmd
        getuid = getattr(os, "getuid", None)
        getgid = getattr(os, "getgid", None)
        if getuid is None or getgid is None:
            return shell_cmd
        uid = getuid()
        gid = getgid()
        if uid == 0:
            # Host is root — chown to 0:0 is a no-op, skip the find walk.
            return shell_cmd

        quoted_ws = shlex.quote(workspace_path)
        # `find -uid 0 -exec chown +` only touches files the container
        # actually wrote as root, leaving legitimately differently-owned
        # files (e.g. a vendored .git/objects tree) alone. The trailing
        # `|| true` keeps the build's exit code even if chown trips on a
        # transient file (race with the build's own cleanup).
        trailer = (
            f"; __rc=$?; "
            f"find {quoted_ws} -uid 0 -exec chown {uid}:{gid} {{}} + "
            f"2>/dev/null || true; "
            f"exit $__rc"
        )
        return f"( {shell_cmd} ){trailer}"


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
    ) -> tuple[int, str, bool, bool]:
        # OS-specific shell dispatch is centralised in _platform.shell_argv:
        # POSIX → ``sh -c "cd <quoted> && <cmd>"`` (byte-identical to the
        # previous inline branch); Windows → ``sh -c`` if Git Bash / WSL
        # exposed one on PATH, else ``cmd /c "cd /d <path> && <cmd>"``.
        cmd = _platform.shell_argv(command, workdir=workspace_path)
        logger.info("[sandbox:bare] Running without isolation (bare subprocess).")
        # Layer 2 pytest diagnostics plugin — no mount step needed in the
        # bare backend because the child process shares the host FS and
        # can import from the plugin's host path directly. Same helper as
        # the docker path so the plugin file, injection logic, and parser
        # stay in sync across backends. Same host-side kill switch.
        merged_env = dict(extra_env or {})
        if _teane_diagnostics_available() and _teane_diagnostics_injection_enabled():
            _apply_teane_diagnostics_env(merged_env, _TEANE_PLUGIN_HOST_DIR)
        return await _execute_subprocess_with_timeout(cmd, timeout_seconds, extra_env=merged_env)


# ---------------------------------------------------------------------------
# 6. Shared Subprocess Execution with PGID Management
# ---------------------------------------------------------------------------

async def _execute_subprocess_with_timeout(
    cmd: list[str],
    timeout_seconds: int,
    extra_env: Optional[dict[str, str]] = None,
    log_buffer_mode: str = "disk",
    max_log_size_mb: int = 500,
    log_temp_dir: Optional[str] = None,
) -> tuple[int, str, bool, bool]:
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
        Tuple of (exit_code, combined_stdout_stderr, timed_out, log_truncated).
        log_truncated is True when the streamer's byte cap was hit — the
        returned output is missing data and downstream diagnostic parsing
        may not reflect the true error.
    """
    timed_out = False
    exit_code = -1

    # Build environment. Inheriting os.environ exposes LLM API keys
    # (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...) and other secrets to
    # build commands and to any LLM-generated process the build spawns.
    # Scrub the well-known set by default; the user can re-export
    # whatever they actually need via extra_env.
    env = {k: v for k, v in os.environ.items() if k not in _SCRUBBED_BUILD_ENV_VARS}
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

    proc: Optional[asyncio.subprocess.Process] = None
    pgid: Optional[int] = None
    reader_tasks: list[asyncio.Task[Any]] = []
    try:
        await streamer.open()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **_platform.new_process_group_kwargs(),  # POSIX: start_new_session; Windows: CREATE_NEW_PROCESS_GROUP
            env=env,
        )

        # Capture the process group ID for clean tree termination.
        # On Windows ``os.getpgid`` doesn't exist; leave ``pgid`` as ``None``
        # so the timeout-watchdog falls through to the cross-platform
        # ``proc.kill()`` path inside ``_kill_process_group``.
        if hasattr(os, "getpgid"):
            try:
                pgid = os.getpgid(proc.pid)
            except (ProcessLookupError, OSError):
                pgid = proc.pid

        async def _read_stdout() -> None:
            assert proc is not None and proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                await streamer.write_stdout(line)

        async def _read_stderr() -> None:
            assert proc is not None and proc.stderr is not None
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                await streamer.write_stderr(line)

        async def _wait_with_timeout() -> int:
            assert proc is not None
            try:
                return await asyncio.wait_for(proc.wait(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                return -1  # Sentinel for timeout

        reader_tasks = [
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
            # Timeout — kill the entire process group (non-blocking; uses
            # await asyncio.sleep so other coroutines keep making progress
            # during the 3 s SIGTERM→SIGKILL grace. Audit §1.12.)
            timed_out = True
            logger.warning(
                "[sandbox] Build timed out after %ds. Killing process group %s.",
                timeout_seconds,
                pgid,
            )
            await _kill_process_group_async(pgid, proc)
            # Do NOT call proc.communicate() here — the reader tasks below
            # are still draining stdout/stderr, and communicate() racing
            # them on the same pipes can deadlock or duplicate output. Just
            # wait briefly for the killed process to actually terminate,
            # then let the reader tasks naturally hit EOF and exit.
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("[sandbox] Process did not terminate within 5s of SIGKILL.")

            exit_code = -9  # SIGKILL equivalent

        # Cancel any reader tasks still running. After the process exits
        # they hit EOF and complete naturally; on timeout-kill they may
        # already have exited or need an explicit cancel.
        for task in reader_tasks:
            if not task.done():
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
        if pending:
            for task in pending:
                if not task.done():
                    task.cancel()
            # Await the cancelled tasks so their CancelledError is
            # actually consumed; without this, asyncio logs
            # "Task was destroyed but it is pending!" and the cancel
            # acknowledgement leaks.
            await asyncio.gather(*pending, return_exceptions=True)

    except FileNotFoundError:
        await streamer.close()
        return 127, f"Command not found: {cmd[0]}", False, False
    except PermissionError:
        await streamer.close()
        return 126, f"Permission denied: {cmd[0]}", False, False
    except asyncio.CancelledError:
        # The caller (or a higher-level cancellation, e.g. Ctrl-C, gateway
        # timeout, parent task cancel) is unwinding us. We MUST NOT leak
        # the subprocess (it would run as an orphan) or the streamer temp
        # files. Kill the process group, cancel reader tasks, close the
        # streamer, and re-raise. Audit §1.4.
        logger.info("[sandbox] Cancelled mid-build; tearing down subprocess + streamer.")
        if proc is not None and proc.returncode is None:
            try:
                await _kill_process_group_async(pgid, proc)
            except Exception:  # noqa: BLE001
                logger.exception("[sandbox] failed to kill process group on cancel.")
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        for task in reader_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
        try:
            await streamer.close()
        except Exception:  # noqa: BLE001
            pass
        raise
    except Exception as exc:
        logger.exception("[sandbox] Unexpected subprocess error.")
        await streamer.close()
        return 1, f"Subprocess error: {exc}", False, False

    full_output = await streamer.read_all()
    log_truncated = getattr(streamer, "is_truncated", lambda: False)()
    await streamer.close(keep_on_success=(exit_code == 0))
    return exit_code, full_output, timed_out, log_truncated


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
        temp_dir: Optional[str] = None,
    ) -> None:
        self.max_size_bytes = max_size_mb * 1024 * 1024
        # Default-resolve here (not at the param) so the platform check
        # happens at call time, not import time. On POSIX this gives
        # ``/tmp/.harness`` byte-identically to the historical default;
        # on Windows it gives ``%TEMP%\.harness`` (avoids FileNotFoundError
        # when os.makedirs would otherwise see a literal ``/tmp`` path).
        self.temp_dir = temp_dir if temp_dir is not None else _platform.harness_temp_dir()
        self._stdout_file: Optional[Any] = None  # tempfile.NamedTemporaryFile
        self._stderr_file: Optional[Any] = None
        self._stdout_path: str = ""
        self._stderr_path: str = ""
        self._total_bytes: int = 0
        self._overflow_count: int = 0

    # Aging janitor: keep retained logs (from successful builds whose
    # close(keep_on_success=True) intentionally left the temp files
    # behind) for this many days before sweeping at next open(). Audit
    # §2.9. Set to 0 to keep forever (matches the historical behaviour).
    _LOG_RETENTION_DAYS: int = 7

    async def open(self) -> None:
        """Create temp files for log output."""
        os.makedirs(self.temp_dir, exist_ok=True)
        # Boot-time janitor: remove harness_*.std{out,err}.log files older
        # than _LOG_RETENTION_DAYS so kept-on-success files and crash-
        # leaked tmps don't accumulate forever. Audit §2.9.
        self._janitor_sweep_old_logs()

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

    def _janitor_sweep_old_logs(self) -> None:
        """Remove harness_* log files older than _LOG_RETENTION_DAYS.

        Best-effort: any IO error is swallowed (the streamer's hot path
        must not stall on a janitor failure). Audit §2.9.
        """
        retention = self.__class__._LOG_RETENTION_DAYS
        if retention <= 0:
            return
        cutoff = time.time() - (retention * 86400.0)
        try:
            for name in os.listdir(self.temp_dir):
                if not name.startswith("harness_"):
                    continue
                if not (name.endswith(".stdout.log") or name.endswith(".stderr.log")):
                    continue
                full = os.path.join(self.temp_dir, name)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                if st.st_mtime < cutoff:
                    try:
                        os.unlink(full)
                    except OSError:
                        pass
        except OSError:
            return

    async def write_stdout(self, data: bytes) -> None:
        if self._stdout_file is None:
            return
        if self._total_bytes >= self.max_size_bytes:
            if self._overflow_count == 0:
                logger.error(
                    "[logstream:disk] Build log exceeded %.0fMB cap. Truncating from this point "
                    "onward — downstream diagnostic parsing may miss the real error if it occurs "
                    "after this position. Raise sandbox.log_buffer_max_mb to capture more.",
                    self.max_size_bytes / (1024 * 1024),
                )
            self._overflow_count += 1
            return
        self._stdout_file.write(data)
        self._total_bytes += len(data)

    async def write_stderr(self, data: bytes) -> None:
        if self._stderr_file is None:
            return
        if self._total_bytes >= self.max_size_bytes:
            if self._overflow_count == 0:
                logger.error(
                    "[logstream:disk] Build log exceeded %.0fMB cap. Truncating from this point "
                    "onward — downstream diagnostic parsing may miss the real error if it occurs "
                    "after this position. Raise sandbox.log_buffer_max_mb to capture more.",
                    self.max_size_bytes / (1024 * 1024),
                )
            self._overflow_count += 1
            return
        self._stderr_file.write(data)
        self._total_bytes += len(data)

    def is_truncated(self) -> bool:
        """True if any write was dropped because the cap was hit."""
        return self._overflow_count > 0

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
    """Synchronous SIGTERM-then-SIGKILL of an entire process group/tree.

    NOTE: This blocks the event loop with ``time.sleep(3.0)`` between the
    TERM and the KILL. Prefer :func:`_kill_process_group_async` from async
    callers — see audit §1.12 — but keep this sync entry-point for the
    handful of callers that already had to be synchronous (atexit
    shutdown hooks, etc.).

    POSIX: SIGTERM the group via ``os.killpg``, sleep, SIGKILL the group.
    Windows: ``taskkill /T /PID`` (graceful), wait, then ``taskkill /T /F``
    (force) — both walk the WMI parent-child tree so grandchildren get
    reaped along with the parent. Dispatched through
    :func:`harness._platform.kill_process_tree`.
    """
    if pgid is not None and hasattr(os, "killpg"):
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
        return
    # Windows OR pgid-unavailable fallback: kill the whole tree of the
    # spawned process by pid. taskkill /T walks the parent-child tree on
    # Windows; on POSIX-without-killpg we just SIGKILL the parent.
    pid = getattr(proc, "pid", None)
    if pid is not None:
        _platform.kill_process_tree(pid, force=False)
        time.sleep(3.0)
        _platform.kill_process_tree(pid, force=True)
        return
    try:
        proc.kill()
    except ProcessLookupError:
        pass


async def _kill_process_group_async(
    pgid: Optional[int], proc: asyncio.subprocess.Process,
) -> None:
    """Async SIGTERM→SIGKILL of a process group/tree without blocking the event loop.

    Replaces the time.sleep(3.0) in _kill_process_group with await
    asyncio.sleep(3.0), so callers inside the asyncio runtime no longer
    freeze every other coroutine (gateway dispatch, MCP polling, SSE
    callbacks, schedule daemon ticks) for the grace period. Audit §1.12.

    Windows path delegates to :func:`harness._platform.kill_process_tree`
    which shells out to ``taskkill /T`` so grandchildren are reaped too.
    """
    if pgid is not None and hasattr(os, "killpg"):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            return
        await asyncio.sleep(3.0)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        return
    pid = getattr(proc, "pid", None)
    if pid is not None:
        _platform.kill_process_tree(pid, force=False)
        await asyncio.sleep(3.0)
        _platform.kill_process_tree(pid, force=True)
        return
    try:
        proc.kill()
    except ProcessLookupError:
        pass


async def run_subprocess_kill_on_timeout(
    argv: list[str],
    *,
    timeout: float,
    cwd: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
    capture_stderr_separately: bool = True,
) -> tuple[int, bytes, bytes, bool]:
    """Run a subprocess that is reliably killed on ``asyncio.TimeoutError``.

    The bare ``asyncio.wait_for(proc.communicate(), timeout=…)`` pattern
    used across the codebase cancels the communicate task but DOES NOT
    kill the child. The child + its pipe FDs + (for docker/find/etc.)
    its background work all keep running. Audit §2.3 / §2.5 / §2.6 / §2.13.

    Returns ``(exit_code, stdout, stderr, timed_out)``. On timeout the
    process group is SIGTERMed then SIGKILLed, ``exit_code`` is -9, and
    ``timed_out`` is True.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=(asyncio.subprocess.PIPE if capture_stderr_separately
                else asyncio.subprocess.STDOUT),
        cwd=cwd,
        env=env,
        **_platform.new_process_group_kwargs(),
    )
    pgid: Optional[int] = None
    if hasattr(os, "getpgid"):
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            pgid = proc.pid
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        # ``proc.returncode`` can be None after communicate() returns if
        # the process closed its pipes while still alive (rare; happens
        # with double-fork daemons). Coercing None to 0 would silently
        # report success on that path; use -1 as the "no real code"
        # sentinel so callers can distinguish.
        rc = proc.returncode if proc.returncode is not None else -1
        return rc, stdout or b"", stderr or b"", False
    except asyncio.TimeoutError:
        await _kill_process_group_async(pgid, proc)
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass
        return -9, b"", b"timed out", True
    except asyncio.CancelledError:
        # Caller is unwinding — don't leak the child.
        await _kill_process_group_async(pgid, proc)
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        raise


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
    # Prod-import smoke markers. The smoke script in graph.py emits a
    # PROD_IMPORT_SMOKE_FAILURES header followed by one FAIL: line per
    # failing module, and _run_prod_import_smoke_check parses that
    # header out of raw_output. Without these patterns, exception types
    # not otherwise listed here (ValidationError, AttributeError,
    # NameError, RuntimeError, KeyError, ValueError, …) get stripped by
    # this filter — the smoke parser then sees no header and falls back
    # to a useless install-chatter diagnostic, and the repair loop burns
    # iterations on blind guesses.
    re.compile(r"^FAIL:\s"),
    re.compile(r"PROD_IMPORT_SMOKE_(?:FAILURES|OK)"),
    # Pytest ERROR-section shapes (finsearch b674f3ca). Setup/teardown/
    # collection errors print "ERROR at setup of X" block headers,
    # "ERROR path.py::node" summary rows (no colon after ERROR, so the
    # ``\bERROR:\s`` pattern above misses every one of them), and a
    # fixture-lookup cause line ("fixture 'client' not found") that names
    # no exception class. A report whose only failures are these shapes
    # matches NOTHING else in this list, and the no-match fallback in
    # filter_critical_errors then reduces the whole log to its last 50
    # lines — the stderr installer tail. Case-sensitive on ERROR
    # deliberately: pytest always prints it uppercase, and lowercase
    # "error at setup" prose from app logging must not drag five lines
    # of context each into the filtered view.
    re.compile(r"\bERROR at (setup|teardown|collection) of\b"),
    re.compile(r"\bERROR collecting\b"),
    re.compile(r"^=+ ERRORS =+$"),
    re.compile(r"^ERROR\s+\S+\.py(?:::|\s|$)"),
    re.compile(r"\bfixture '[^']+' not found\b"),
    # "No tests collected" markers across stacks. These look benign but
    # graph.py's `_is_no_tests_collected` uses them to fold the runner's
    # non-zero exit to success (the batch just has no tests yet).
    # Without preserving them here, `filter_critical_errors` drops the
    # marker whenever ANY earlier line in the log matches a critical
    # pattern (session 648309aa: uv's `warning: Failed to hardlink` fired
    # `\bFAILED\b` and its 5-line context pushed pytest's tail out of the
    # kept window). The repair loop then spun on a non-error.
    re.compile(r"\bno tests ran in\b", re.IGNORECASE),                     # pytest
    re.compile(r"\bno tests? (?:were )?found\b", re.IGNORECASE),           # Jest / Vitest / Mocha / Gradle
    re.compile(r"\bno test files? found\b", re.IGNORECASE),                # Vitest / Mocha
    re.compile(r"\bthere are no tests? to run\b", re.IGNORECASE),          # Maven Surefire
    re.compile(r"\bno tests were executed\b", re.IGNORECASE),              # Maven Surefire (failIfNoTests)
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

_STRUCTURED_COMPILER_FLAGS: dict[str, str] = {}


def _detect_compiler(build_command: str) -> Optional[str]:
    """Heuristically detect which compiler toolchain a build command uses."""
    cmd_lower = build_command.lower()
    for compiler in ["make", "cmake"]:
        if compiler in cmd_lower:
            return compiler.split()[0]
    return None


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
    """Extract structured diagnostics from compiler output using appropriate parser.

    Routes through the parser_registry plugin set first (covers Python/pytest,
    Java/Maven/Gradle, TypeScript/tsc) so toolchains beyond the legacy
    hard-coded ones produce structured diagnostics instead of an empty list.
    Falls back to the generic regex parser when the registry returns nothing.
    """
    try:
        from harness.parser_registry import detect_and_parse
        registry_diags = detect_and_parse(
            raw_output,
            build_command=build_command,
            workspace_path=workspace_path,
        )
        if registry_diags:
            return registry_diags
    except Exception as exc:  # noqa: BLE001
        logger.debug("[extract_diagnostics] parser_registry failed: %s", exc)

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
        **kwargs: Backend-specific configuration (e.g., image='python:3.12' for docker).

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

    try:
        from harness.observability import log_failure
        log_failure(
            "sandbox_start_failed",
            reason="auto_detect_no_backend",
            docker_available=False,
            unshare_available=False,
            unsafe_bare_opt_in=False,
        )
    except Exception:  # noqa: BLE001 — telemetry must never mask the real error
        pass
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
            "docker_image": "harness-builder:latest",
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
        extra_env: Optional[dict[str, str]] = None,
        sandbox_config: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ):
        self.workspace_path = os.path.abspath(workspace_path)
        self.allow_network = allow_network
        # Fall back to the process-wide default validator (set by cmd_run via
        # harness.security.set_command_validator) so every executor gets
        # defense-in-depth without each call site having to pass one. Tests
        # and one-off scripts that don't initialise a global remain unchanged.
        if command_validator is None:
            try:
                from harness.security import get_command_validator
                command_validator = get_command_validator()
            except Exception:  # noqa: BLE001 — fail open: validator is defense in depth
                command_validator = None
        self.command_validator = command_validator
        self.extra_env = extra_env or {}

        cfg = sandbox_config or {}
        # Per-call config overrides — sandbox_config keys win when both supplied.
        self.timeout_seconds = cfg.get("timeout_seconds", timeout_seconds)
        cfg_mounts = cfg.get("readonly_cache_mounts")
        self.readonly_cache_mounts = (
            list(cfg_mounts) if cfg_mounts is not None
            else (readonly_cache_mounts or [])
        )
        self.pgid_kill_on_timeout = cfg.get(
            "pgid_kill_on_timeout", pgid_kill_on_timeout,
        )

        if backend is not None:
            self.backend = backend
        else:
            docker_kwargs: dict[str, Any] = {}
            if "docker_image" in cfg:
                docker_kwargs["image"] = cfg["docker_image"]
            if "docker_memory_limit" in cfg:
                docker_kwargs["memory_limit"] = cfg["docker_memory_limit"]
            if "docker_cpu_limit" in cfg:
                docker_kwargs["cpu_limit"] = cfg["docker_cpu_limit"]
            if "docker_pids_limit" in cfg:
                docker_kwargs["pids_limit"] = cfg["docker_pids_limit"]
            if "read_only_root" in cfg:
                docker_kwargs["read_only_root"] = bool(cfg["read_only_root"])
            if "restore_workspace_ownership" in cfg:
                docker_kwargs["restore_workspace_ownership"] = bool(
                    cfg["restore_workspace_ownership"]
                )
            # sandbox.cache_volumes opts the docker backend into writable
            # named Docker volumes mounted on the builder image's fixed
            # /cache/{pip,uv,npm} paths (see DockerBackend._DEFAULT_CACHE_VOLUME_PATHS),
            # so pip / uv / npm downloads persist across containers.
            # Default ON — pip's content-addressed cache makes cross-session
            # sharing safe, and the alternative (every compile re-downloads
            # every wheel) is the dominant runtime cost on cold workspaces.
            #
            # sandbox.cache_volumes_scope controls whether the volume name is
            # session-namespaced ("session") or shared across sessions
            # ("global", the default). Operators who explicitly want the old
            # per-session isolation set it to "session" — useful when running
            # untrusted code from different tenants under the same daemon.
            cache_volumes_on = cfg.get("cache_volumes", True)
            if cache_volumes_on:
                docker_kwargs["cache_volumes_enabled"] = True
                scope = str(cfg.get("cache_volumes_scope", "global")).lower()
                if scope == "session":
                    docker_kwargs["cache_volumes_session_id"] = session_id
                # else: leave session_id at None → _cache_volume_name uses
                # "global" and every session reuses the same volume.
                cvp = cfg.get("cache_volumes_prefix")
                if cvp:
                    docker_kwargs["cache_volumes_prefix"] = str(cvp)
            else:
                docker_kwargs["cache_volumes_enabled"] = False
            requested_backend = (cfg.get("backend", "auto") or "auto").lower()
            if requested_backend in ("auto", ""):
                # auto-detect: forward docker kwargs (auto-detect itself
                # only applies them to DockerBackend; non-docker fallbacks
                # get no kwargs).
                self.backend = _auto_detect_backend(**docker_kwargs)
            elif requested_backend == "docker":
                self.backend = create_backend("docker", **docker_kwargs)
            else:
                # unshare / bare don't take image kwargs; pass none.
                self.backend = create_backend(requested_backend)

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
                    full_output=str(exc),
                )

        start_time = time.monotonic()

        logger.info(
            "[sandbox] Executing build with backend=%s: %s",
            self.backend.name,
            build_command.replace("\n", "\\n"),
        )

        try:
            from harness.observability import emit_event
            emit_event("build_start", backend=self.backend.name, command=build_command)
        except Exception:  # noqa: BLE001
            pass

        exit_code, raw_output, timed_out, log_truncated = await self.backend.run(
            command=build_command,
            workspace_path=self.workspace_path,
            timeout_seconds=self.timeout_seconds,
            allow_network=self.allow_network,
            readonly_cache_mounts=self.readonly_cache_mounts,
            extra_env=self.extra_env or None,
        )

        elapsed = time.monotonic() - start_time

        # Filter the output to extract critical errors
        filtered = filter_critical_errors(raw_output)

        # Parse structured diagnostics
        diagnostics = extract_diagnostics(raw_output, build_command, self.workspace_path)

        if log_truncated:
            logger.error(
                "[sandbox] Build log was truncated at the configured cap. The error "
                "shown below may not be the root cause — extracted diagnostics: %d.",
                len(diagnostics),
            )

        logger.info(
            "[sandbox] Build finished: backend=%s exit=%d elapsed=%.2fs timed_out=%s diagnostics=%d log_truncated=%s",
            self.backend.name, exit_code, elapsed, timed_out, len(diagnostics), log_truncated,
        )

        try:
            from harness.observability import emit_event
            emit_event(
                "build_end",
                backend=self.backend.name,
                exit_code=exit_code,
                elapsed_seconds=round(elapsed, 3),
                timed_out=timed_out,
                log_truncated=log_truncated,
                diagnostics=len(diagnostics),
            )
        except Exception:  # noqa: BLE001
            pass

        # Cache-corruption signature scan. Append a one-line hint to the
        # returned raw_output (the same field the LLM repair loop reads) so
        # the next iteration sees the recoverable-corruption signal rather
        # than chasing the real-looking hash mismatch as a code bug. Scan
        # the original raw_output (filter_critical_errors may have dropped
        # the signature line under aggressive filtering).
        result_output = filtered if exit_code != 0 else raw_output
        full_output = raw_output
        hint = _cache_corruption_hint(raw_output)
        if hint:
            result_output = (result_output or "") + hint
            full_output = (full_output or "") + hint

        return BuildResult(
            exit_code=exit_code,
            raw_output=result_output,
            diagnostics=diagnostics,
            elapsed_seconds=elapsed,
            timed_out=timed_out,
            log_truncated=log_truncated,
            full_output=full_output,
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