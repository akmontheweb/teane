"""Language Server Protocol client pool — brownfield semantic navigation.

Spawns real language servers as stdio subprocesses and exposes ground-truth
symbol navigation (find-references / go-to-definition) to (a) the planner as
``lsp__*`` skills invokable via ``<<<LSP_CALL ...>>>`` blocks, and (b) the
harness itself as prefetch helpers that upgrade three heuristic sites
(diagnostics-gate impact expansion, repair-prompt caller map, change-request
impact augment) from the tree-sitter :class:`harness.impact.DependencyGraph`
to authoritative results — with the DependencyGraph kept as the fallback tier.

Scope of Phase 1
================
- Servers: ``pyright-langserver --stdio`` (Python) and
  ``typescript-language-server --stdio`` (TS/TSX/JS/JSX). **jdtls (Java) is
  deferred to Phase 2**: it cannot be launched as a plain ``<binary> --stdio``
  (launcher jar, per-workspace ``-data`` dir, JVM flags), which breaks the
  uniform spawn/probe model of this slice — and its primary payoff for this
  harness is *diagnostics*, which the compiler pipeline already covers for
  Java, not navigation.
- Brownfield only: the cli starts the pool for ``flow != "build"``
  (``lsp.enabled_flows``, default patch/test). Greenfield behaviour is
  byte-identical — no pool, no prompt section, heuristics unchanged.
- Environment-health probe: a server is only spawned when the workspace can
  actually resolve imports (builds run inside Docker, so host-side deps may
  legitimately be absent). Python: a ``.venv``/``venv`` dir at the workspace
  root (override: ``lsp.python_require_venv=false``). TypeScript:
  ``tsconfig.json`` AND ``node_modules`` present.
- No auto-restart: a dead server fails its pending futures once; every
  consumer checks :attr:`StdioLspClient.alive` via the pool and falls back
  to heuristics. Nothing about the pool lives in AgentState — checkpoint /
  resume simply cold-starts (or skips) a fresh pool.

Architecture is a deliberate clone of :mod:`harness.mcp_client` — same
subprocess lifecycle, pending-future correlation, shutdown and atexit
semantics (private attr names ``_proc``/``_started``/``_pending`` are load-
bearing: cli.py's pool registry and its synchronous atexit backstop are
duck-typed against them, and tests pre-set ``_proc`` to inject a fake
transport). The divergences are protocol-level:

1. Framing is ``Content-Length: N\\r\\n\\r\\n<body>`` (LSP base protocol),
   not newline-delimited JSON.
2. The handshake is ``initialize`` → ``initialized`` with a ``rootUri``.
3. The read loop classifies THREE message shapes: responses (resolve
   pending future), notifications (drop), and **server→client requests,
   which MUST be answered** — pyright stalls waiting on
   ``workspace/configuration`` otherwise.
4. Positional requests are wrapped in ``didOpen``/``didClose`` so the
   server has deterministically seen the exact on-disk text the positions
   refer to (typescript-language-server may not seed a file's project
   until an open event).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from urllib.parse import unquote, urlparse

from harness import _platform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

# Server binaries allowed to spawn (passed as extra_allowlist to the shared
# trust validator; `npx` is already in the trust default allowlist).
_LSP_COMMAND_ALLOWLIST = ["pyright-langserver", "typescript-language-server"]

_DEFAULT_REQUEST_TIMEOUT = 15.0
_MAX_RPC_BODY_BYTES = 10 * 1024 * 1024
# Files larger than this are queried without didOpen (server reads disk).
_MAX_DIDOPEN_BYTES = 2 * 1024 * 1024
# One-shot grace: pyright indexes asynchronously after initialize; an empty
# first symbol query retries once after this delay.
_READINESS_RETRY_DELAY = 2.0

# LSP SymbolKind values worth querying references for when building a
# caller map: Class=5, Function=12, Variable=13, Constant=14.
_CALLER_MAP_SYMBOL_KINDS = frozenset({5, 12, 13, 14})

# Directories whose hits are never useful navigation results.
_EXCLUDED_RESULT_DIRS = ("node_modules/", ".venv/", "venv/")


# ---------------------------------------------------------------------------
# 2. Config
# ---------------------------------------------------------------------------

@dataclass
class LspServerConfig:
    """One language server: how to spawn it and which files it owns."""
    name: str                          # "python" | "typescript"
    command: list[str]
    language_ids: dict[str, str]       # extension → LSP languageId


_PYTHON_SERVER = LspServerConfig(
    name="python",
    command=["pyright-langserver", "--stdio"],
    language_ids={".py": "python", ".pyi": "python"},
)
_TYPESCRIPT_SERVER = LspServerConfig(
    name="typescript",
    command=["typescript-language-server", "--stdio"],
    language_ids={
        ".ts": "typescript",
        ".tsx": "typescriptreact",
        ".js": "javascript",
        ".jsx": "javascriptreact",
    },
)


@dataclass
class LspPoolConfig:
    enabled: bool = True
    enabled_flows: list[str] = field(default_factory=lambda: ["patch", "test"])
    request_timeout_seconds: float = _DEFAULT_REQUEST_TIMEOUT
    python_require_venv: bool = True
    prefetch_budget_seconds: float = 20.0

    @classmethod
    def from_config(cls, config: Optional[dict[str, Any]]) -> "LspPoolConfig":
        section = ((config or {}).get("lsp") or {})
        return cls(
            enabled=bool(section.get("enabled", True)),
            enabled_flows=[
                str(f) for f in (section.get("enabled_flows") or ["patch", "test"])
            ],
            request_timeout_seconds=max(
                1.0, float(section.get("request_timeout_seconds", _DEFAULT_REQUEST_TIMEOUT)),
            ),
            python_require_venv=bool(section.get("python_require_venv", True)),
            prefetch_budget_seconds=max(
                1.0, float(section.get("prefetch_budget_seconds", 20.0)),
            ),
        )


class LspError(RuntimeError):
    """JSON-RPC error response or transport/process failure."""

    def __init__(self, error: Any):
        if isinstance(error, dict):
            message = error.get("message") or str(error)
        else:
            message = str(error)
        super().__init__(message)
        self.error = error


# ---------------------------------------------------------------------------
# 3. Environment-health probe
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    healthy: bool
    reason: str = ""


def probe_workspace_health(
    server_name: str,
    workspace_path: str,
    *,
    python_require_venv: bool = True,
) -> ProbeResult:
    """Can this workspace actually resolve imports for ``server_name``?

    Builds run inside Docker, so a brownfield repo's dependencies may only
    exist in the container. A language server pointed at a dep-less host
    workspace returns unresolved-import garbage — worse than the heuristic
    fallback, so we refuse to spawn rather than limp along.
    """
    ws = os.path.abspath(workspace_path)
    if server_name == "python":
        if not python_require_venv:
            return ProbeResult(True)
        if os.path.isdir(os.path.join(ws, ".venv")) or os.path.isdir(
            os.path.join(ws, "venv")
        ):
            return ProbeResult(True)
        return ProbeResult(
            False,
            "no .venv/venv at workspace root (imports unresolvable; set "
            "lsp.python_require_venv=false to override)",
        )
    if server_name == "typescript":
        if not os.path.isfile(os.path.join(ws, "tsconfig.json")):
            return ProbeResult(False, "no tsconfig.json at workspace root")
        if not os.path.isdir(os.path.join(ws, "node_modules")):
            return ProbeResult(
                False, "no node_modules at workspace root (run npm install)",
            )
        return ProbeResult(True)
    return ProbeResult(False, f"unknown server {server_name!r}")


# ---------------------------------------------------------------------------
# 4. Stdio LSP client
# ---------------------------------------------------------------------------

class StdioLspClient:
    """Single-server LSP client over an asyncio subprocess + JSON-RPC with
    Content-Length framing. Lifecycle and correlation mirror
    :class:`harness.mcp_client.StdioMcpClient`; see module docstring for
    the protocol divergences.
    """

    def __init__(
        self,
        config: LspServerConfig,
        workspace_path: str,
        *,
        timeout_seconds: float = _DEFAULT_REQUEST_TIMEOUT,
    ):
        self.config = config
        self.workspace_path = os.path.abspath(workspace_path)
        self.timeout_seconds = timeout_seconds
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._stderr_task: Optional[asyncio.Task[None]] = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._next_id = 0
        self._write_lock = asyncio.Lock()
        self._started = False
        self._readiness_grace_used = False

    # -- lifecycle ------------------------------------------------------

    @property
    def alive(self) -> bool:
        return (
            self._started
            and self._proc is not None
            and self._proc.returncode is None
        )

    async def start(self) -> None:
        """Spawn the server and run the LSP initialize handshake.

        Raises :class:`LspError` on handshake failure; ``ValueError`` when
        the command is rejected by the trust allowlist.
        """
        if self._started:
            return
        from harness.trust import validate_mcp_server_command
        validate_mcp_server_command(
            self.config.command, extra_allowlist=_LSP_COMMAND_ALLOWLIST,
        )
        # Test hook (same contract as StdioMcpClient): a pre-set _proc
        # means a fake transport is wired — skip spawn and task creation.
        if self._proc is None:
            from harness.trust import safe_subprocess_env
            env = safe_subprocess_env({})
            logger.info(
                "[lsp:%s] spawning %s (root=%s)",
                self.config.name, " ".join(self.config.command),
                self.workspace_path,
            )
            self._proc = await asyncio.create_subprocess_exec(
                *self.config.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                **_platform.new_process_group_kwargs(),
            )
            self._reader_task = asyncio.create_task(
                self._read_loop(), name=f"lsp-reader-{self.config.name}",
            )
            self._stderr_task = asyncio.create_task(
                self._stderr_loop(), name=f"lsp-stderr-{self.config.name}",
            )
        root_uri = Path(self.workspace_path).as_uri()
        await self._call(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": root_uri,
                "workspaceFolders": [
                    {"uri": root_uri, "name": os.path.basename(self.workspace_path)},
                ],
                "capabilities": {
                    "textDocument": {
                        "references": {},
                        "definition": {},
                        "documentSymbol": {
                            "hierarchicalDocumentSymbolSupport": True,
                        },
                        "synchronization": {},
                    },
                    "workspace": {"symbol": {}, "workspaceFolders": True},
                },
                "clientInfo": {"name": "teane", "version": "1.0"},
            },
            timeout=self.timeout_seconds,
        )
        await self._notify("initialized", {})
        self._started = True
        logger.info("[lsp:%s] ready", self.config.name)

    async def shutdown(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        # Snapshot-and-clear so the reader loop can't race set_result
        # against our set_exception (same hazard as mcp_client audit §1.13).
        pending_snapshot = dict(self._pending)
        self._pending.clear()
        for fut in pending_snapshot.values():
            if not fut.done():
                try:
                    fut.set_exception(LspError({"message": "client shutting down"}))
                except asyncio.InvalidStateError:
                    pass
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
        if self._stderr_task and not self._stderr_task.done():
            self._stderr_task.cancel()
        try:
            if proc.returncode is None:
                if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError, OSError):
                        proc.terminate()
                else:
                    _platform.kill_process_tree(proc.pid, force=False)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                        kill_sig = getattr(signal, "SIGKILL", signal.SIGTERM)
                        try:
                            os.killpg(os.getpgid(proc.pid), kill_sig)
                        except (ProcessLookupError, PermissionError, OSError):
                            proc.kill()
                    else:
                        _platform.kill_process_tree(proc.pid, force=True)
                    await proc.wait()
        finally:
            self._proc = None
            self._started = False
            logger.info("[lsp:%s] shut down", self.config.name)

    # -- protocol surface -------------------------------------------------

    async def _call_with_readiness_grace(
        self, method: str, params: dict[str, Any],
    ) -> Any:
        """One-shot per-client grace for empty results: the server indexes
        the workspace asynchronously after initialize, so the FIRST empty
        answer (symbols or cross-file references alike) may just mean
        "not indexed yet". Sleep once, retry once, then trust emptiness
        for the rest of the session."""
        result = await self._call(method, params, timeout=self.timeout_seconds)
        if not result and not self._readiness_grace_used:
            self._readiness_grace_used = True
            await asyncio.sleep(_READINESS_RETRY_DELAY)
            result = await self._call(method, params, timeout=self.timeout_seconds)
        return result

    async def workspace_symbols(self, query: str) -> list[dict[str, Any]]:
        result = await self._call_with_readiness_grace(
            "workspace/symbol", {"query": query},
        )
        return list(result or [])

    async def document_symbols(self, rel_path: str) -> list[dict[str, Any]]:
        async with self._open_document(rel_path):
            result = await self._call_with_readiness_grace(
                "textDocument/documentSymbol",
                {"textDocument": {"uri": self._uri(rel_path)}},
            )
        return list(result or [])

    async def references(
        self,
        rel_path: str,
        line: int,
        character: int,
        *,
        include_declaration: bool = False,
    ) -> list[dict[str, Any]]:
        async with self._open_document(rel_path):
            result = await self._call_with_readiness_grace(
                "textDocument/references",
                {
                    "textDocument": {"uri": self._uri(rel_path)},
                    "position": {"line": line, "character": character},
                    "context": {"includeDeclaration": include_declaration},
                },
            )
        return list(result or [])

    async def definition(
        self, rel_path: str, line: int, character: int,
    ) -> list[dict[str, Any]]:
        async with self._open_document(rel_path):
            result = await self._call(
                "textDocument/definition",
                {
                    "textDocument": {"uri": self._uri(rel_path)},
                    "position": {"line": line, "character": character},
                },
                timeout=self.timeout_seconds,
            )
        if isinstance(result, dict):  # single Location
            return [result]
        return list(result or [])

    # -- internals --------------------------------------------------------

    def _uri(self, rel_path: str) -> str:
        p = rel_path if os.path.isabs(rel_path) else os.path.join(
            self.workspace_path, rel_path,
        )
        return Path(p).as_uri()

    @asynccontextmanager
    async def _open_document(self, rel_path: str) -> AsyncIterator[None]:
        """didOpen/didClose bracket so positions refer to text the server
        has deterministically seen. Oversize/unreadable files are queried
        without opening (the server reads from disk best-effort)."""
        abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(
            self.workspace_path, rel_path,
        )
        opened = False
        try:
            if os.path.isfile(abs_path) and os.path.getsize(abs_path) <= _MAX_DIDOPEN_BYTES:
                ext = os.path.splitext(abs_path)[1].lower()
                language_id = self.config.language_ids.get(ext, "plaintext")
                with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
                await self._notify("textDocument/didOpen", {
                    "textDocument": {
                        "uri": self._uri(rel_path),
                        "languageId": language_id,
                        "version": 1,
                        "text": text,
                    },
                })
                opened = True
        except OSError:
            pass
        try:
            yield
        finally:
            if opened:
                try:
                    await self._notify("textDocument/didClose", {
                        "textDocument": {"uri": self._uri(rel_path)},
                    })
                except Exception:  # noqa: BLE001 — close is best-effort
                    pass

    async def _call(
        self, method: str, params: dict[str, Any], *, timeout: float,
    ) -> Any:
        if self._proc is None or self._proc.stdin is None:
            raise LspError({"message": "transport not connected"})
        async with self._write_lock:
            self._next_id += 1
            req_id = self._next_id
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[req_id] = future
        msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        try:
            await self._send(msg)
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise LspError({
                "message": f"timeout waiting for {method} response after {timeout}s",
            }) from exc
        finally:
            self._pending.pop(req_id, None)
            if not future.done():
                future.cancel()

    async def _notify(self, method: str, params: Optional[dict[str, Any]] = None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        await self._send(msg)

    async def _send(self, msg: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise LspError({"message": "stdin closed"})
        body = json.dumps(msg).encode("utf-8")
        frame = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
        async with self._write_lock:
            self._proc.stdin.write(frame)
            await self._proc.stdin.drain()

    async def _read_loop(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        stdout = self._proc.stdout
        try:
            while True:
                # Headers are tiny; the default StreamReader limit is fine
                # for readuntil here. Body goes through readexactly, which
                # is not limit-bound.
                try:
                    header_blob = await stdout.readuntil(b"\r\n\r\n")
                except asyncio.IncompleteReadError as exc:
                    if not exc.partial:
                        break
                    logger.debug(
                        "[lsp:%s] EOF with partial header (%d bytes)",
                        self.config.name, len(exc.partial),
                    )
                    break
                except asyncio.LimitOverrunError:
                    logger.error(
                        "[lsp:%s] header block exceeded stream limit; "
                        "dropping transport.", self.config.name,
                    )
                    break
                content_length = -1
                for header_line in header_blob.split(b"\r\n"):
                    name, _, value = header_line.partition(b":")
                    if name.strip().lower() == b"content-length":
                        try:
                            content_length = int(value.strip())
                        except ValueError:
                            content_length = -1
                        break
                if content_length < 0:
                    logger.error(
                        "[lsp:%s] frame without Content-Length; dropping "
                        "transport.", self.config.name,
                    )
                    break
                if content_length > _MAX_RPC_BODY_BYTES:
                    logger.error(
                        "[lsp:%s] oversize frame (%d bytes); dropping "
                        "transport.", self.config.name, content_length,
                    )
                    break
                try:
                    body = await stdout.readexactly(content_length)
                except asyncio.IncompleteReadError:
                    break
                try:
                    msg = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    logger.debug(
                        "[lsp:%s] dropping malformed frame: %s",
                        self.config.name, exc,
                    )
                    continue
                await self._classify_and_handle(msg)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("[lsp:%s] reader crashed", self.config.name)
        finally:
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(LspError({"message": "transport closed"}))

    async def _classify_and_handle(self, msg: dict[str, Any]) -> None:
        """Three-way classification (the key divergence from MCP):
        response → resolve future; notification → drop; server→client
        REQUEST → must answer, or pyright stalls waiting on it."""
        has_id = "id" in msg and msg.get("id") is not None
        has_method = bool(msg.get("method"))
        if has_id and has_method:
            await self._respond_to_server_request(msg)
            return
        if not has_id:
            logger.debug(
                "[lsp:%s] notification %s ignored",
                self.config.name, msg.get("method", "?"),
            )
            return
        future = self._pending.get(msg["id"])
        if future is None or future.done():
            return
        if "error" in msg:
            future.set_exception(LspError(msg["error"]))
        else:
            # LSP results may legitimately be null (= no locations found);
            # preserve that instead of coercing to {} like the MCP client.
            future.set_result(msg.get("result"))

    async def _respond_to_server_request(self, msg: dict[str, Any]) -> None:
        method = str(msg.get("method") or "")
        params = msg.get("params") or {}
        response: dict[str, Any] = {"jsonrpc": "2.0", "id": msg["id"]}
        if method == "workspace/configuration":
            items = params.get("items") or []
            response["result"] = [None] * max(1, len(items))
        elif method in (
            "window/workDoneProgress/create",
            "client/registerCapability",
            "client/unregisterCapability",
        ):
            response["result"] = None
        elif method == "workspace/workspaceFolders":
            root_uri = Path(self.workspace_path).as_uri()
            response["result"] = [
                {"uri": root_uri, "name": os.path.basename(self.workspace_path)},
            ]
        else:
            response["error"] = {"code": -32601, "message": f"method not found: {method}"}
        try:
            await self._send(response)
        except Exception:  # noqa: BLE001 — a failed reply surfaces as timeouts later
            logger.debug(
                "[lsp:%s] failed to answer server request %s",
                self.config.name, method,
            )

    async def _stderr_loop(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        try:
            while True:
                line = await self._proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.debug("[lsp:%s:stderr] %s", self.config.name, text)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("[lsp:%s] stderr reader crashed", self.config.name)


# ---------------------------------------------------------------------------
# 5. Pool
# ---------------------------------------------------------------------------

class LspClientPool:
    """Owns the language-server clients for one workspace/session.

    Single-process scope, not thread-safe (matches the harness's
    single-threaded async runtime). ``clients`` and the clients' ``_proc``
    attrs are duck-typed against by cli.py's shared pool registry — do not
    rename.
    """

    def __init__(self, config: LspPoolConfig, workspace_path: str):
        self.config = config
        self.workspace_path = os.path.abspath(workspace_path)
        self.clients: dict[str, StdioLspClient] = {}
        self.skipped: list[dict[str, str]] = []
        self._dead_logged: set[str] = set()
        try:
            import atexit
            atexit.register(self._atexit_kill)
        except Exception:  # noqa: BLE001
            pass

    def _atexit_kill(self) -> None:
        """Synchronous best-effort SIGTERM at interpreter shutdown (the
        event loop may already be gone) — verbatim MCP-pool semantics."""
        for client in list(self.clients.values()):
            proc = getattr(client, "_proc", None)
            if proc is None or getattr(proc, "returncode", 0) is not None:
                continue
            try:
                pid = proc.pid
            except Exception:  # noqa: BLE001
                continue
            try:
                if hasattr(os, "killpg"):
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGTERM)
                    except (ProcessLookupError, OSError):
                        try:
                            os.kill(pid, signal.SIGTERM)
                        except OSError:
                            pass
                else:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except OSError:
                        pass
            except Exception:  # noqa: BLE001
                pass

    @property
    def started(self) -> bool:
        return any(c._started for c in self.clients.values())

    def healthy(self) -> bool:
        return any(c.alive for c in self.clients.values())

    def client_for_file(self, path: str) -> Optional[StdioLspClient]:
        ext = os.path.splitext(path)[1].lower()
        for client in self.clients.values():
            if ext in client.config.language_ids:
                if client.alive:
                    return client
                if client.config.name not in self._dead_logged:
                    self._dead_logged.add(client.config.name)
                    logger.warning(
                        "[lsp:pool] server %r died — navigation for %s files "
                        "degrades to heuristics for the rest of the session.",
                        client.config.name, ext,
                    )
                return None
        return None

    async def start(self, workspace_tags: set[str]) -> None:
        """Select servers by workspace stack tags, probe environment
        health, spawn the survivors concurrently. Per-server failures are
        isolated — one bad server never blocks the pool."""
        if not self.config.enabled:
            return
        candidates: list[LspServerConfig] = []
        if "python" in workspace_tags:
            candidates.append(_PYTHON_SERVER)
        if {"typescript", "node"} & workspace_tags:
            candidates.append(_TYPESCRIPT_SERVER)

        to_spawn: list[LspServerConfig] = []
        for spec in candidates:
            if shutil.which(spec.command[0]) is None:
                self.skipped.append({
                    "server": spec.name,
                    "reason": f"{spec.command[0]} not on PATH",
                })
                continue
            probe = probe_workspace_health(
                spec.name, self.workspace_path,
                python_require_venv=self.config.python_require_venv,
            )
            if not probe.healthy:
                self.skipped.append({"server": spec.name, "reason": probe.reason})
                logger.info(
                    "[lsp:pool] skipping %s: %s", spec.name, probe.reason,
                )
                continue
            to_spawn.append(spec)

        async def _start_one(spec: LspServerConfig) -> None:
            client = StdioLspClient(
                spec, self.workspace_path,
                timeout_seconds=self.config.request_timeout_seconds,
            )
            try:
                await client.start()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[lsp:pool] server %r failed to start: %s. Skipping.",
                    spec.name, exc,
                )
                self.skipped.append({"server": spec.name, "reason": str(exc)})
                await client.shutdown()
                return
            self.clients[spec.name] = client

        if to_spawn:
            await asyncio.gather(*(_start_one(s) for s in to_spawn))

    async def shutdown(self) -> None:
        if self.clients:
            await asyncio.gather(
                *(c.shutdown() for c in self.clients.values()),
                return_exceptions=True,
            )
            self.clients.clear()
        global _active_pool
        if _active_pool is self:
            _active_pool = None


# ---------------------------------------------------------------------------
# 6. Module-level accessor (graph-side access; NOTHING in AgentState)
# ---------------------------------------------------------------------------

_active_pool: Optional[LspClientPool] = None


def set_active_pool(pool: Optional[LspClientPool]) -> None:
    global _active_pool
    _active_pool = pool


def get_active_pool() -> Optional[LspClientPool]:
    return _active_pool


def clear_active_pool() -> None:
    set_active_pool(None)


# ---------------------------------------------------------------------------
# 7. Result normalization
# ---------------------------------------------------------------------------

def _uri_to_rel(uri: str, workspace_path: str) -> Optional[str]:
    """file:// URI → forward-slash workspace-relative path, or None when
    the location is outside the workspace / in a toolchain dir."""
    if not uri.startswith("file:"):
        return None
    path = unquote(urlparse(uri).path)
    ws = os.path.abspath(workspace_path)
    try:
        rel = os.path.relpath(path, ws)
    except ValueError:
        return None
    rel = rel.replace(os.sep, "/")
    if rel.startswith(".."):
        return None
    if any(rel.startswith(d) or f"/{d}" in f"/{rel}" for d in _EXCLUDED_RESULT_DIRS):
        return None
    return rel


def _location_fields(entry: dict[str, Any]) -> tuple[str, int, int]:
    """Extract (uri, line, character) from Location / LocationLink /
    SymbolInformation / WorkspaceSymbol shapes, defensively."""
    loc = entry.get("location") if isinstance(entry.get("location"), dict) else entry
    uri = str(loc.get("uri") or loc.get("targetUri") or "")
    rng = loc.get("range") or loc.get("targetSelectionRange") or loc.get("targetRange") or {}
    start = rng.get("start") or {}
    try:
        line = int(start.get("line", 0))
        char = int(start.get("character", 0))
    except (TypeError, ValueError):
        line, char = 0, 0
    return uri, line, char


def _normalize_locations(
    raw: Any, workspace_path: str, *, exclude_rel: str = "",
) -> list[dict[str, Any]]:
    """Any LSP location-ish payload → deduped
    ``[{"file": rel, "line": int, "character": int}]`` inside the workspace."""
    if raw is None:
        return []
    entries = raw if isinstance(raw, list) else [raw]
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        uri, line, char = _location_fields(entry)
        rel = _uri_to_rel(uri, workspace_path)
        if rel is None or rel == exclude_rel:
            continue
        key = (rel, line)
        if key in seen:
            continue
        seen.add(key)
        out.append({"file": rel, "line": line, "character": char})
    return out


# ---------------------------------------------------------------------------
# 8. High-level query helpers (shared by skills AND harness prefetch sites)
# ---------------------------------------------------------------------------

def _symbol_name(entry: dict[str, Any]) -> str:
    return str(entry.get("name") or "")


async def find_symbol_locations(
    pool: LspClientPool, symbol: str, file: str = "",
) -> list[dict[str, Any]]:
    """Resolve a plain symbol name to definition locations via
    workspace/symbol. Ranking: exact name match first, then results in the
    ``file`` hint, preserving server order within each tier."""
    clients: list[StdioLspClient]
    if file:
        client = pool.client_for_file(file)
        clients = [client] if client else []
    else:
        clients = [c for c in pool.clients.values() if c.alive]
    candidates: list[tuple[int, dict[str, Any]]] = []
    for client in clients:
        try:
            raw = await client.workspace_symbols(symbol)
        except LspError as exc:
            logger.debug("[lsp] workspace/symbol failed on %s: %s",
                         client.config.name, exc)
            continue
        for entry in raw or []:
            if not isinstance(entry, dict):
                continue
            uri, line, char = _location_fields(entry)
            rel = _uri_to_rel(uri, pool.workspace_path)
            if rel is None:
                continue
            rank = 0 if _symbol_name(entry) == symbol else 2
            if file and rel == file.replace(os.sep, "/"):
                rank -= 1
            candidates.append((rank, {
                "file": rel, "line": line, "character": char,
                "name": _symbol_name(entry),
            }))
    candidates.sort(key=lambda pair: pair[0])
    seen: set[tuple[str, int]] = set()
    out: list[dict[str, Any]] = []
    for _rank, loc in candidates:
        key = (loc["file"], loc["line"])
        if key in seen:
            continue
        seen.add(key)
        out.append(loc)
    return out


async def find_references_by_symbol(
    pool: LspClientPool, symbol: str, file: str = "",
) -> list[dict[str, Any]]:
    """Symbol name → best definition via workspace/symbol → references."""
    locations = await find_symbol_locations(pool, symbol, file)
    if not locations:
        return []
    best = locations[0]
    client = pool.client_for_file(best["file"])
    if client is None:
        return []
    try:
        raw = await client.references(
            best["file"], best["line"], best["character"],
            include_declaration=False,
        )
    except LspError as exc:
        logger.debug("[lsp] references failed for %s: %s", symbol, exc)
        return []
    return _normalize_locations(raw, pool.workspace_path)


async def find_definition_by_symbol(
    pool: LspClientPool, symbol: str, file: str = "",
) -> list[dict[str, Any]]:
    """Symbol name → definition locations. workspace/symbol results ARE
    definition sites, so the resolved locations are returned directly
    (capped to the plausible few)."""
    locations = await find_symbol_locations(pool, symbol, file)
    return [
        {k: loc[k] for k in ("file", "line", "character")} for loc in locations[:10]
    ]


def _flatten_document_symbols(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Top-level entries from either hierarchical DocumentSymbol[] or flat
    SymbolInformation[]; returns dicts with name/kind and a start position."""
    out = []
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        if "selectionRange" in entry:  # DocumentSymbol
            start = (entry.get("selectionRange") or {}).get("start") or {}
        else:  # SymbolInformation
            start = (((entry.get("location") or {}).get("range") or {})
                     .get("start") or {})
        try:
            line = int(start.get("line", 0))
            char = int(start.get("character", 0))
        except (TypeError, ValueError):
            line, char = 0, 0
        out.append({
            "name": _symbol_name(entry),
            "kind": int(entry.get("kind") or 0),
            "line": line,
            "character": char,
        })
    return out


async def callers_of_file(
    pool: LspClientPool,
    rel_path: str,
    *,
    max_symbols: int = 20,
    budget_seconds: float = 10.0,
) -> dict[str, set[str]]:
    """Ground-truth reverse dependencies of one file: for each of its
    top-level public symbols, who references it? Returns
    ``{caller_rel_file: {symbol, ...}}`` excluding the file itself.
    Deadline-bounded; returns whatever accumulated when time runs out."""
    client = pool.client_for_file(rel_path)
    if client is None:
        return {}
    deadline = time.monotonic() + budget_seconds
    try:
        raw_symbols = await client.document_symbols(rel_path)
    except LspError as exc:
        logger.debug("[lsp] documentSymbol failed for %s: %s", rel_path, exc)
        return {}
    symbols = [
        s for s in _flatten_document_symbols(raw_symbols)
        if s["kind"] in _CALLER_MAP_SYMBOL_KINDS
        and s["name"] and not s["name"].startswith("_")
    ][:max_symbols]
    rel_norm = rel_path.replace(os.sep, "/")
    callers: dict[str, set[str]] = {}
    for sym in symbols:
        if time.monotonic() >= deadline:
            logger.debug(
                "[lsp] callers_of_file(%s) hit %.0fs budget after %d symbol(s)",
                rel_path, budget_seconds, len(callers),
            )
            break
        try:
            raw_refs = await client.references(
                rel_path, sym["line"], sym["character"],
                include_declaration=False,
            )
        except LspError:
            continue
        for loc in _normalize_locations(
            raw_refs, pool.workspace_path, exclude_rel=rel_norm,
        ):
            callers.setdefault(loc["file"], set()).add(sym["name"])
    return callers


# ---------------------------------------------------------------------------
# 9. Skills
# ---------------------------------------------------------------------------

def _elapsed_ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


async def _skill_find_references(symbol: str = "", file: str = "", **_kw: Any) -> dict[str, Any]:
    pool = get_active_pool()
    if not symbol:
        return {"error": "missing required parameter: symbol"}
    if pool is None or not pool.healthy():
        return {"error": "lsp unavailable — fall back to reading files"}
    t0 = time.monotonic()
    locations = await find_references_by_symbol(pool, symbol, file)
    try:
        from harness.observability import emit_event
        emit_event("lsp_query", tool="find_references", symbol=symbol,
                   elapsed_ms=_elapsed_ms(t0), results=len(locations))
    except Exception:  # noqa: BLE001
        pass
    return {"symbol": symbol, "locations": locations[:50], "count": len(locations)}


async def _skill_go_to_definition(symbol: str = "", file: str = "", **_kw: Any) -> dict[str, Any]:
    pool = get_active_pool()
    if not symbol:
        return {"error": "missing required parameter: symbol"}
    if pool is None or not pool.healthy():
        return {"error": "lsp unavailable — fall back to reading files"}
    t0 = time.monotonic()
    locations = await find_definition_by_symbol(pool, symbol, file)
    try:
        from harness.observability import emit_event
        emit_event("lsp_query", tool="go_to_definition", symbol=symbol,
                   elapsed_ms=_elapsed_ms(t0), results=len(locations))
    except Exception:  # noqa: BLE001
        pass
    return {"symbol": symbol, "locations": locations, "count": len(locations)}


def register_lsp_skills(pool: LspClientPool) -> int:
    """Register the two navigation skills. The skill fns re-resolve the
    active pool at call time (not the closure), so a drained/dead pool
    degrades to a polite error string rather than a stale handle."""
    from harness.skills import (
        SkillParameter, SkillSchema, SkillType, ToolSkill, register,
    )
    params = [
        SkillParameter(
            name="symbol", type="string", required=True,
            description="Plain symbol name (function/class/constant), no positions.",
        ),
        SkillParameter(
            name="file", type="string", required=False,
            description="Optional workspace-relative file hint to disambiguate.",
        ),
    ]
    register(ToolSkill(
        SkillSchema(
            name="lsp__find_references",
            description=(
                "Find every location in the workspace that references the "
                "given symbol (language-server ground truth, not text search)."
            ),
            skill_type=SkillType.TOOL,
            parameters=list(params),
            tags=["lsp", "navigation"],
        ),
        fn=_skill_find_references,
    ))
    register(ToolSkill(
        SkillSchema(
            name="lsp__go_to_definition",
            description=(
                "Resolve the given symbol name to its definition location(s) "
                "in the workspace via the language server."
            ),
            skill_type=SkillType.TOOL,
            parameters=list(params),
            tags=["lsp", "navigation"],
        ),
        fn=_skill_go_to_definition,
    ))
    return 2


# ---------------------------------------------------------------------------
# 10. LSP_CALL block parser — same DSL shape as MCP_CALL
# ---------------------------------------------------------------------------

_LSP_BLOCK_RE = re.compile(
    r"<<<\s*LSP_CALL\s+(.*?)>>>",
    re.DOTALL | re.IGNORECASE,
)
# Values MUST be quoted (clone of _MCP_STR_KWARG_RE) — the prompt section
# shows the quoted form.
_LSP_STR_KWARG_RE = re.compile(
    r"""(\w+)\s*=\s*(?P<q>['"])(.*?)(?<!\\)(?P=q)""",
    re.DOTALL,
)

_LSP_TOOLS = frozenset({"find_references", "go_to_definition"})


def parse_lsp_blocks(content: str) -> list[Any]:
    """Extract ``<<<LSP_CALL tool="find_references" symbol="X" file="y.py">>>``
    blocks. Returns :class:`harness.web_tools.ParsedToolBlock` objects so the
    graph tool loop iterates them uniformly with WEB/MCP blocks. Unknown
    tools yield an empty skill_name → the loop surfaces "not registered"."""
    from harness.web_tools import ParsedToolBlock
    blocks: list[ParsedToolBlock] = []
    if not isinstance(content, str) or "<<<" not in content:
        return blocks
    for match in _LSP_BLOCK_RE.finditer(content):
        body = match.group(1)
        kwargs: dict[str, Any] = {}
        for kw in _LSP_STR_KWARG_RE.finditer(body):
            kwargs[kw.group(1)] = kw.group(3).replace('\\"', '"').replace("\\'", "'")
        tool = kwargs.pop("tool", "")
        skill_name = f"lsp__{tool}" if tool in _LSP_TOOLS else ""
        blocks.append(ParsedToolBlock(
            skill_name=skill_name,
            kwargs={k: v for k, v in kwargs.items() if k in ("symbol", "file")},
            raw=match.group(0),
        ))
    return blocks


def strip_lsp_blocks(content: str) -> str:
    """Remove every ``<<<LSP_CALL ...>>>`` block from ``content``."""
    if not isinstance(content, str) or "<<<" not in content:
        return content
    return _LSP_BLOCK_RE.sub("", content)
