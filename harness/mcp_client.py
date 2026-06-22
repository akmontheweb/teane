"""Model Context Protocol (MCP) client — stdio transport.

Connects the harness to one or more **MCP servers** declared in
``config.json``'s ``mcp.servers`` block. Each server exposes a set of
typed tools which are registered into :class:`harness.skills.SkillRegistry`
as :class:`McpToolSkill` instances (name = ``mcp__<server>__<tool>``).
The planner / patcher / repair LLM invokes them via the existing text-DSL
interceptor in :func:`harness.graph._run_tool_loop`, using a block of the
form::

    <<<MCP_CALL server="github" tool="get_issue" args='{"owner":"x","repo":"y","number":42}'>>>

Scope of v1
===========
- **stdio transport only.** Almost every MCP server in the wild today
  ships as ``npx -y @some-org/server-name`` or ``python -m server`` —
  both stdio. HTTP/SSE transport, MCP ``prompts`` / ``resources`` /
  ``sampling`` capabilities, and per-tool ACL are deferred.
- **Hand-rolled JSON-RPC 2.0**, newline-delimited JSON framing per the
  current MCP spec. No dependency on the upstream ``mcp`` SDK so the
  core install stays clean. Upgrade to the SDK is a follow-up if/when
  the spec evolves enough to make hand-rolled drift expensive.

Safety
======
- Server commands run through :func:`harness.trust.validate_mcp_server_command`
  before any process is spawned: command-allowlist (``npx`` / ``node`` /
  ``python`` / ``uvx`` / ``docker``), hard-deny on shell wrappers, scan
  for shell metacharacters, refuse absolute paths under ``/etc``,
  ``/root``, ``/proc``, ``/sys``.
- Each call has a per-tool timeout (default 30 s, configurable). Hung
  servers don't stall the graph indefinitely.
- ``atexit`` handler in :meth:`McpClientPool.shutdown` SIGTERMs every
  subprocess then SIGKILLs after a short grace period — no leaked
  processes when the harness exits.
- Tool results are pushed through the same redactor + length cap the
  web tools use before re-entering the LLM conversation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
from dataclasses import dataclass, field
from typing import Any, Optional

from harness import _platform
from harness.skills import (
    SkillSchema,
    SkillType,
    ToolSkill,
)
from harness.trust import validate_mcp_server_command

logger = logging.getLogger(__name__)


# Wire protocol version we send during the initialize handshake. MCP is
# backwards-compatible across minor versions; servers that don't match
# negotiate down. Pinning a specific date keeps the behaviour predictable
# in the test pack.
_MCP_PROTOCOL_VERSION = "2024-11-05"

# Default per-call timeout. Clamped in config validation.
_DEFAULT_TOOL_CALL_TIMEOUT = 30.0

# Cap on tool result size pushed back into the LLM conversation. MCP servers
# can legitimately return large payloads (logs, file dumps); we truncate to
# keep the context window healthy.
_DEFAULT_RESULT_MAX_BYTES = 200_000


# ---------------------------------------------------------------------------
# 1. Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class McpServerConfig:
    """Per-server config loaded from a single entry of
    ``config.mcp.servers``."""

    name: str
    transport: str = "stdio"  # only "stdio" is supported in v1
    command: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # When set, the harness reads the named env var and adds it to the
    # spawned process's env as ``MCP_API_KEY`` (or under the server's
    # documented key — operators put the right key name in the env dict
    # directly when needed; api_key_env is just a convenience).
    api_key_env: str = ""
    # Workspace tags this server is relevant for (e.g. ["python", "web"]).
    # Empty list = applies to every workspace (status-quo behaviour).
    # ``register_mcp_skills`` uses the intersection against the runtime
    # workspace tags to decide whether to expose this server's tools.
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "McpServerConfig":
        return cls(
            name=str(raw.get("name", "")).strip(),
            transport=str(raw.get("transport", "stdio")),
            command=list(raw.get("command", [])),
            env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
            api_key_env=str(raw.get("api_key_env", "")),
            tags=[str(t) for t in (raw.get("tags") or [])],
        )


@dataclass
class McpPoolConfig:
    """Top-level ``mcp`` section."""

    enabled: bool = False
    tool_call_timeout_seconds: float = _DEFAULT_TOOL_CALL_TIMEOUT
    allow_local_filesystem_servers: bool = False
    command_allowlist: list[str] = field(default_factory=list)
    result_max_bytes: int = _DEFAULT_RESULT_MAX_BYTES
    servers: list[McpServerConfig] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: Optional[dict[str, Any]]) -> "McpPoolConfig":
        section = ((config or {}).get("mcp") or {})
        servers = [
            McpServerConfig.from_dict(s)
            for s in (section.get("servers") or [])
            if isinstance(s, dict)
        ]
        return cls(
            enabled=bool(section.get("enabled", False)),
            tool_call_timeout_seconds=float(
                section.get("tool_call_timeout_seconds", _DEFAULT_TOOL_CALL_TIMEOUT)
            ),
            allow_local_filesystem_servers=bool(
                section.get("allow_local_filesystem_servers", False)
            ),
            command_allowlist=list(section.get("command_allowlist", []) or []),
            result_max_bytes=int(section.get("result_max_bytes", _DEFAULT_RESULT_MAX_BYTES)),
            servers=servers,
        )


# ---------------------------------------------------------------------------
# 2. JSON-RPC errors
# ---------------------------------------------------------------------------

class McpError(RuntimeError):
    """Raised when an MCP call returns a JSON-RPC error response or when
    the transport / process layer fails before a response arrives.

    Wraps the raw error dict so the caller (typically the graph tool
    interceptor) can surface a structured failure back to the LLM
    instead of crashing the dispatch.
    """

    def __init__(self, error: Any):
        if isinstance(error, dict):
            message = error.get("message") or str(error)
        else:
            message = str(error)
        super().__init__(message)
        self.error = error


# ---------------------------------------------------------------------------
# 3. Stdio MCP client
# ---------------------------------------------------------------------------

class StdioMcpClient:
    """Single-server MCP client over an asyncio subprocess + JSON-RPC.

    Lifecycle:
      1. ``start()`` — validate the command, spawn the subprocess with a
         scrubbed env, do the ``initialize`` handshake, cache the tool
         list.
      2. ``call_tool(name, args)`` — JSON-RPC ``tools/call`` with a
         per-call timeout.
      3. ``shutdown()`` — SIGTERM, drain stderr for diagnostic output,
         SIGKILL after a 5 s grace period.
    """

    def __init__(
        self,
        config: McpServerConfig,
        *,
        timeout_seconds: float = _DEFAULT_TOOL_CALL_TIMEOUT,
        extra_allowlist: Optional[list[str]] = None,
    ):
        self.config = config
        self.timeout_seconds = timeout_seconds
        self._extra_allowlist = extra_allowlist or []
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._stderr_task: Optional[asyncio.Task[None]] = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._next_id = 0
        self._write_lock = asyncio.Lock()
        self._tools: list[dict[str, Any]] = []
        self._started = False
        self._server_info: dict[str, Any] = {}

    @property
    def tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    @property
    def server_info(self) -> dict[str, Any]:
        return dict(self._server_info)

    async def start(self) -> None:
        """Spawn the subprocess, do the initialize handshake, list tools.

        Raises ``McpError`` if the handshake fails. ``ValueError`` if the
        command is rejected by :func:`validate_mcp_server_command`.
        """
        if self._started:
            return
        validate_mcp_server_command(
            self.config.command, extra_allowlist=self._extra_allowlist,
        )
        # When ``_proc`` is already populated (test path: pre-wired fake
        # transport), skip the subprocess spawn and reader-task creation
        # entirely — the caller is in charge of those. The handshake
        # below runs identically either way.
        if self._proc is None:
            # Build a scrubbed env: start from the harness's safe env and
            # overlay the operator-supplied env + (optional) api key.
            from harness.trust import safe_subprocess_env
            env = safe_subprocess_env(self.config.env)
            if self.config.api_key_env:
                key_value = os.environ.get(self.config.api_key_env, "")
                if key_value:
                    env[self.config.api_key_env] = key_value
            logger.info(
                "[mcp:%s] spawning %s",
                self.config.name, " ".join(self.config.command),
            )
            self._proc = await asyncio.create_subprocess_exec(
                *self.config.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                # POSIX: start_new_session; Windows: CREATE_NEW_PROCESS_GROUP.
                # Either way the child is the root of its own group so the
                # shutdown path can reap the whole tree cleanly.
                **_platform.new_process_group_kwargs(),
            )
            self._reader_task = asyncio.create_task(
                self._read_loop(), name=f"mcp-reader-{self.config.name}",
            )
            self._stderr_task = asyncio.create_task(
                self._stderr_loop(), name=f"mcp-stderr-{self.config.name}",
            )
        # initialize handshake
        result = await self._call(
            "initialize",
            {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "teane", "version": "1.0"},
            },
            timeout=self.timeout_seconds,
        )
        self._server_info = result.get("serverInfo") or {}
        await self._notify("notifications/initialized")
        # tools/list
        tools_result = await self._call(
            "tools/list", {}, timeout=self.timeout_seconds,
        )
        self._tools = list(tools_result.get("tools") or [])
        self._started = True
        logger.info(
            "[mcp:%s] ready (%d tool(s) advertised: %s)",
            self.config.name, len(self._tools),
            ", ".join(t.get("name", "?") for t in self._tools[:8]),
        )

    async def list_tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self._started:
            raise McpError({"message": "client not started"})
        return await self._call(
            "tools/call",
            {"name": tool_name, "arguments": arguments or {}},
            timeout=self.timeout_seconds,
        )

    async def shutdown(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        # Snapshot-and-clear so the reader loop can't re-look up an entry
        # we're about to set_exception on (and race with set_result),
        # which would raise InvalidStateError. Audit §1.13.
        pending_snapshot = dict(self._pending)
        self._pending.clear()
        for fut in pending_snapshot.values():
            if not fut.done():
                try:
                    fut.set_exception(McpError({"message": "client shutting down"}))
                except asyncio.InvalidStateError:
                    pass
        # Stop the reader first so we don't fight stdout EOF on terminate.
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
        if self._stderr_task and not self._stderr_task.done():
            self._stderr_task.cancel()
        try:
            if proc.returncode is None:
                # Graceful: SIGTERM the whole process group on POSIX;
                # taskkill /T (no /F) on Windows for a parallel "send
                # Ctrl+Break to the tree" semantic. Both leave grandchildren
                # a moment to clean up before the force-kill below.
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
                    # Force: SIGKILL the group on POSIX; taskkill /T /F on
                    # Windows so grandchildren are reaped too.
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
            logger.info("[mcp:%s] shut down", self.config.name)

    # -- internals -----------------------------------------------------

    async def _call(
        self, method: str, params: dict[str, Any], *, timeout: float,
    ) -> dict[str, Any]:
        if self._proc is None or self._proc.stdin is None:
            raise McpError({"message": "transport not connected"})
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
            raise McpError({
                "message": f"timeout waiting for {method} response after {timeout}s",
            }) from exc
        finally:
            # Drop our registration and, if the caller was cancelled
            # while the request was still in flight, cancel the future
            # too so it can be garbage-collected immediately rather than
            # waiting for shutdown to drain it. Audit §1.13.
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
            raise McpError({"message": "stdin closed"})
        line = (json.dumps(msg) + "\n").encode("utf-8")
        self._proc.stdin.write(line)
        await self._proc.stdin.drain()

    # Hard cap on a single JSON-RPC line. A misbehaving server returning
    # multi-MB without a newline could OOM the harness via unbounded
    # buffer growth — audit §4.9. 10 MiB comfortably accommodates real
    # tool-result payloads while bounding the worst case.
    _MAX_RPC_LINE_BYTES: int = 10 * 1024 * 1024

    async def _read_loop(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        try:
            while True:
                try:
                    line = await self._proc.stdout.readuntil(b"\n")
                except asyncio.LimitOverrunError as exc:
                    # The pipe has more than the StreamReader's limit
                    # buffered without a newline — surface a clear error
                    # then break (audit §4.9).
                    logger.error(
                        "[mcp:%s] server emitted a line exceeding %d bytes "
                        "without a newline; dropping transport.",
                        self.config.name, getattr(exc, "consumed", 0),
                    )
                    break
                except asyncio.IncompleteReadError as exc:
                    # EOF / partial — emit whatever was buffered (if any)
                    # then exit the loop on next iteration.
                    line = exc.partial
                    if not line:
                        break
                if len(line) > self._MAX_RPC_LINE_BYTES:
                    logger.error(
                        "[mcp:%s] dropping oversize line (%d bytes)",
                        self.config.name, len(line),
                    )
                    continue
                if not line:
                    break
                try:
                    msg = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    logger.debug(
                        "[mcp:%s] dropping malformed line: %s", self.config.name, exc,
                    )
                    continue
                req_id = msg.get("id")
                if req_id is None:
                    # Notification from the server — log + ignore for v1.
                    method = msg.get("method", "?")
                    logger.debug(
                        "[mcp:%s] notification %s ignored", self.config.name, method,
                    )
                    continue
                future = self._pending.get(req_id)
                if future is None or future.done():
                    continue
                if "error" in msg:
                    future.set_exception(McpError(msg["error"]))
                else:
                    future.set_result(msg.get("result") or {})
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("[mcp:%s] reader crashed", self.config.name)
        finally:
            # Wake any remaining pending callers with a clear failure.
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(McpError({"message": "transport closed"}))

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
                    logger.debug("[mcp:%s:stderr] %s", self.config.name, text)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("[mcp:%s] stderr reader crashed", self.config.name)


# ---------------------------------------------------------------------------
# 4. McpClientPool
# ---------------------------------------------------------------------------

class McpClientPool:
    """Owns the set of :class:`StdioMcpClient` instances declared in
    ``config.mcp.servers`` and proxies tool calls to the right one.

    Single-process scope. Not thread-safe; the harness's async runtime
    is single-threaded so this matches the rest of the codebase.
    """

    def __init__(self, config: McpPoolConfig):
        self.config = config
        self.clients: dict[str, StdioMcpClient] = {}
        # Library-level safety net: ensure spawned MCP subprocesses are
        # SIGTERMed at interpreter exit even when the harness is embedded
        # (tests, dashboard subcommands constructing pools directly) and
        # cli.py's atexit hook never registered. Audit §2.8.
        try:
            import atexit
            atexit.register(self._atexit_kill)
        except Exception:  # noqa: BLE001
            pass

    def _atexit_kill(self) -> None:
        """Synchronous best-effort SIGTERM at interpreter shutdown.

        Cannot rely on asyncio at this point — the event loop may already
        be gone. Walk the live clients and signal their process groups
        directly. Audit §2.8.
        """
        for client in list(self.clients.values()):
            proc = getattr(client, "_proc", None)
            if proc is None:
                continue
            if getattr(proc, "returncode", 0) is not None:
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

    async def start(self) -> dict[str, list[dict[str, Any]]]:
        """Spawn every configured server concurrently. Returns the
        mapping ``server_name → tool list`` so the caller (skill
        registration) can wire tools without re-querying.

        A server whose ``start`` fails is logged and skipped — one bad
        server in the config never prevents the rest from coming up.
        """
        if not self.config.enabled:
            return {}
        # Optional safety: refuse to spawn filesystem servers unless the
        # operator explicitly turned that knob on. Filesystem MCP gives
        # the LLM unmediated file I/O on the host, bypassing the sandbox.
        for server in self.config.servers:
            if (not self.config.allow_local_filesystem_servers
                    and self._looks_like_filesystem_server(server)):
                raise ValueError(
                    f"mcp server {server.name!r} looks like a local filesystem "
                    f"server (command={server.command}). Set "
                    f"mcp.allow_local_filesystem_servers=true to opt in."
                )

        async def _start_one(srv: McpServerConfig) -> tuple[str, list[dict[str, Any]]]:
            client = StdioMcpClient(
                srv,
                timeout_seconds=self.config.tool_call_timeout_seconds,
                extra_allowlist=self.config.command_allowlist,
            )
            try:
                await client.start()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[mcp:pool] server %r failed to start: %s. Skipping.",
                    srv.name, exc,
                )
                await client.shutdown()
                return srv.name, []
            self.clients[srv.name] = client
            return srv.name, client.tools

        results = await asyncio.gather(
            *(_start_one(s) for s in self.config.servers if s.name and s.command),
            return_exceptions=False,
        )
        return {name: tools for name, tools in results}

    async def shutdown(self) -> None:
        if not self.clients:
            return
        await asyncio.gather(
            *(c.shutdown() for c in self.clients.values()),
            return_exceptions=True,
        )
        self.clients.clear()

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any],
    ) -> dict[str, Any]:
        client = self.clients.get(server_name)
        if client is None:
            raise McpError({
                "message": (
                    f"mcp server {server_name!r} not registered. "
                    f"Known servers: {sorted(self.clients.keys())}."
                ),
            })
        return await client.call_tool(tool_name, arguments)

    def list_all_tools(self) -> dict[str, list[dict[str, Any]]]:
        return {name: client.tools for name, client in self.clients.items()}

    # -- internals -----------------------------------------------------

    @staticmethod
    def _looks_like_filesystem_server(srv: McpServerConfig) -> bool:
        """Heuristic gate on ``mcp.allow_local_filesystem_servers``. Matches
        the official `@modelcontextprotocol/server-filesystem` and obvious
        rewordings. False negatives are OK — the operator-extended
        allowlist is the real authorisation point."""
        joined = " ".join(srv.command).lower()
        return any(
            needle in joined
            for needle in ("server-filesystem", "filesystem", "fs-server", "fileserver")
        )


# ---------------------------------------------------------------------------
# 5. McpToolSkill — wraps one MCP tool descriptor as a SkillRegistry entry
# ---------------------------------------------------------------------------

class McpToolSkill(ToolSkill):
    """Wraps a single MCP tool advertised by a server. The skill name
    follows the documented convention: ``mcp__<server_name>__<tool_name>``.

    ``execute(**kwargs)`` calls the pool's ``call_tool`` with the kwargs
    forwarded as JSON-RPC arguments. The raw MCP response is returned —
    typically ``{"content": [{"type":"text","text":"..."}], "isError": false}``
    — so the LLM tool-loop sees the structured result.
    """

    def __init__(
        self,
        *,
        server_name: str,
        tool_descriptor: dict[str, Any],
        pool: McpClientPool,
    ):
        self.server_name = server_name
        self.tool_name = str(tool_descriptor.get("name", ""))
        self.input_schema: dict[str, Any] = (
            tool_descriptor.get("inputSchema") or {}
        )
        self._pool = pool
        skill_name = f"mcp__{server_name}__{self.tool_name}"
        # We don't synthesize SkillParameter objects from the MCP
        # inputSchema in v1 — that's only needed when the harness flips
        # ``use_structured_tools=true`` and pipes the schema to the LLM
        # as a native function spec. For now the LLM discovers args via
        # the tool description in the system prompt and the inputSchema
        # surfaces through ``to_tool_schema`` once we wire that up.
        schema = SkillSchema(
            name=skill_name,
            description=str(tool_descriptor.get("description") or "")
                or f"MCP tool {self.tool_name} on server {server_name}.",
            skill_type=SkillType.TOOL,
            parameters=[],
            tags=["mcp", server_name, self.tool_name],
        )
        super().__init__(schema, fn=self._call)

    def to_tool_schema(self) -> dict[str, Any]:  # noqa: D401 — overrides
        """Return the OpenAI-style function-calling schema for this tool.

        Bridges directly to MCP's ``inputSchema`` (which is JSON Schema
        already), so when ``use_structured_tools`` lands the same object
        feeds the provider tool list with zero translation.
        """
        return {
            "type": "function",
            "function": {
                "name": self.schema.name,
                "description": self.schema.description,
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }

    async def _call(self, **kwargs: Any) -> dict[str, Any]:
        try:
            return await self._pool.call_tool(self.server_name, self.tool_name, kwargs)
        except McpError as exc:
            return {"error": str(exc), "mcp_error": exc.error}
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[mcp:%s:%s] unexpected error", self.server_name, self.tool_name,
            )
            return {"error": f"unexpected error: {exc}"}


def register_mcp_skills(
    pool: McpClientPool,
    *,
    workspace_tags: Optional[set[str]] = None,
) -> int:
    """Walk the pool's started clients and register one
    :class:`McpToolSkill` per advertised tool. Returns the total
    registered. Idempotent — re-registering a skill overwrites the
    previous entry, matching :meth:`SkillRegistry.register`.

    When ``workspace_tags`` is provided, a server is included only if
    its declared ``tags`` intersect ``workspace_tags``. Servers with no
    declared tags (empty list) are always included — that keeps the
    pre-existing default behaviour. When ``workspace_tags`` is None,
    no filtering is applied (also status-quo).
    """
    from harness.skills import register

    # Build a name → tag-set map from the pool's config so we can decide
    # per-server. Servers without an entry default to empty (= always
    # registered). Pool stubs in tests may not expose a ``config`` —
    # fall back to an empty map (no filtering possible) in that case.
    server_tag_map: dict[str, set[str]] = {}
    pool_config = getattr(pool, "config", None)
    if pool_config is not None and getattr(pool_config, "servers", None) is not None:
        server_tag_map = {s.name: set(s.tags) for s in pool_config.servers}

    count = 0
    skipped: list[str] = []
    for server_name, tools in pool.list_all_tools().items():
        server_tags = server_tag_map.get(server_name, set())
        if (
            workspace_tags is not None
            and server_tags
            and not (server_tags & workspace_tags)
        ):
            skipped.append(server_name)
            continue
        for tool in tools:
            if not isinstance(tool, dict) or not tool.get("name"):
                continue
            register(McpToolSkill(
                server_name=server_name, tool_descriptor=tool, pool=pool,
            ))
            count += 1
    if skipped:
        logger.info(
            "[mcp] skipped %d server(s) not matching workspace_tags=%s: %s",
            len(skipped), sorted(workspace_tags or set()), skipped,
        )
    return count


# ---------------------------------------------------------------------------
# 6. MCP_CALL block parser  ---  reuse the web_tools DSL shape
# ---------------------------------------------------------------------------

_MCP_BLOCK_RE = re.compile(
    r"<<<\s*MCP_CALL\s+(.*?)>>>",
    re.DOTALL | re.IGNORECASE,
)
_MCP_STR_KWARG_RE = re.compile(
    r"""(\w+)\s*=\s*(?P<q>['"])(.*?)(?<!\\)(?P=q)""",
    re.DOTALL,
)


def parse_mcp_blocks(content: str) -> list[Any]:
    """Extract every ``<<<MCP_CALL server="..." tool="..." args='...'>>>``
    block from an LLM response. Returns objects with the same shape as
    :class:`harness.web_tools.ParsedToolBlock` so the graph's tool loop
    can iterate uniformly.

    The ``args`` kwarg's value MUST be a JSON object (a string the parser
    decodes via ``json.loads``). When it can't be decoded, the block's
    kwargs land as ``{"_parse_error": ..., "_raw_args": ...}`` so the
    interceptor surfaces a clean error back to the LLM rather than
    swallowing the call.
    """
    from harness.web_tools import ParsedToolBlock
    blocks: list[ParsedToolBlock] = []
    if not isinstance(content, str) or "<<<" not in content:
        return blocks
    for match in _MCP_BLOCK_RE.finditer(content):
        body = match.group(1)
        kwargs: dict[str, Any] = {}
        for kw in _MCP_STR_KWARG_RE.finditer(body):
            key = kw.group(1)
            val = kw.group(3)
            # Cheap unescape for embedded \" / \' inside the JSON arg.
            val = val.replace('\\"', '"').replace("\\'", "'")
            kwargs[key] = val
        server = kwargs.pop("server", "")
        tool = kwargs.pop("tool", "")
        raw_args = kwargs.pop("args", "")
        if raw_args:
            try:
                parsed_args = json.loads(raw_args)
                if not isinstance(parsed_args, dict):
                    parsed_args = {"_value": parsed_args}
            except json.JSONDecodeError as exc:
                parsed_args = {"_parse_error": str(exc), "_raw_args": raw_args}
        else:
            parsed_args = {}
        skill_name = f"mcp__{server}__{tool}" if server and tool else ""
        blocks.append(ParsedToolBlock(
            skill_name=skill_name,
            kwargs=parsed_args,
            raw=match.group(0),
        ))
    return blocks


def strip_mcp_blocks(content: str) -> str:
    """Remove every ``<<<MCP_CALL ...>>>`` block from ``content``."""
    if not isinstance(content, str) or "<<<" not in content:
        return content
    return _MCP_BLOCK_RE.sub("", content)
