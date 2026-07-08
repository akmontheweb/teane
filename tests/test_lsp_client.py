"""Tests for harness/lsp_client.py — the brownfield LSP navigation pool.

Covers:
    - Content-Length framing round trip against a fake transport
      (initialize → workspace/symbol → references → shutdown)
    - Server→client requests (workspace/configuration) answered with a
      matching id while an in-flight client request still resolves —
      the #1 stall risk with pyright
    - Framing edges: split header/body, oversize frame, notifications
    - Environment-health probe matrix
    - Symbol resolution + normalization (SymbolInformation vs range-less
      WorkspaceSymbol, file-hint ranking, self/node_modules exclusion)
    - callers_of_file shape + budget behavior
    - Pool start gating (binary missing, probe unhealthy)
    - parse_lsp_blocks DSL + skill fail-open with no pool
    - LspPoolConfig parsing
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

import harness.lsp_client as lc
from harness.lsp_client import (
    LspClientPool,
    LspPoolConfig,
    LspServerConfig,
    StdioLspClient,
    _normalize_locations,
    _uri_to_rel,
    callers_of_file,
    clear_active_pool,
    find_references_by_symbol,
    find_symbol_locations,
    parse_lsp_blocks,
    probe_workspace_health,
    set_active_pool,
    strip_lsp_blocks,
)


# ---------------------------------------------------------------------------
# Fake transport with Content-Length framing
# ---------------------------------------------------------------------------

class _FakeStdin:
    def __init__(self) -> None:
        self.buffer = b""

    def write(self, data: bytes) -> None:
        self.buffer += data

    async def drain(self) -> None:
        return None

    def pop_frames(self) -> list[dict[str, Any]]:
        """Parse and consume every complete Content-Length frame written
        by the client so far."""
        frames: list[dict[str, Any]] = []
        while True:
            sep = self.buffer.find(b"\r\n\r\n")
            if sep < 0:
                break
            headers = self.buffer[:sep]
            length = -1
            for line in headers.split(b"\r\n"):
                name, _, value = line.partition(b":")
                if name.strip().lower() == b"content-length":
                    length = int(value.strip())
            body_start = sep + 4
            if length < 0 or len(self.buffer) < body_start + length:
                break
            frames.append(json.loads(self.buffer[body_start:body_start + length]))
            self.buffer = self.buffer[body_start + length:]
        return frames


class _FakeStdout:
    """Byte-stream fake supporting the readuntil/readexactly contract the
    LSP read loop uses."""

    def __init__(self) -> None:
        self._buf = b""
        self._cond = asyncio.Condition()
        self._eof = False

    async def readuntil(self, separator: bytes) -> bytes:
        async with self._cond:
            while separator not in self._buf and not self._eof:
                await self._cond.wait()
            idx = self._buf.find(separator)
            if idx < 0:
                partial, self._buf = self._buf, b""
                raise asyncio.IncompleteReadError(partial, None)
            end = idx + len(separator)
            chunk, self._buf = self._buf[:end], self._buf[end:]
            return chunk

    async def readexactly(self, n: int) -> bytes:
        async with self._cond:
            while len(self._buf) < n and not self._eof:
                await self._cond.wait()
            if len(self._buf) < n:
                partial, self._buf = self._buf, b""
                raise asyncio.IncompleteReadError(partial, n)
            chunk, self._buf = self._buf[:n], self._buf[n:]
            return chunk

    async def readline(self) -> bytes:  # stderr fake reuse
        async with self._cond:
            while b"\n" not in self._buf and not self._eof:
                await self._cond.wait()
            if b"\n" not in self._buf:
                return b""
            idx = self._buf.find(b"\n") + 1
            chunk, self._buf = self._buf[:idx], self._buf[idx:]
            return chunk

    async def push_raw(self, data: bytes) -> None:
        async with self._cond:
            self._buf += data
            self._cond.notify_all()

    async def push_frame(self, msg: dict[str, Any]) -> None:
        body = json.dumps(msg).encode("utf-8")
        await self.push_raw(
            f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body,
        )

    async def signal_eof(self) -> None:
        async with self._cond:
            self._eof = True
            self._cond.notify_all()


class _FakeProc:
    def __init__(self) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout()
        self.stderr = _FakeStdout()
        self.returncode: int | None = None
        self.pid = 99998

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def _make_client(tmp_path) -> tuple[StdioLspClient, _FakeProc]:
    fake = _FakeProc()
    client = StdioLspClient(
        LspServerConfig(
            name="python",
            command=["pyright-langserver", "--stdio"],
            language_ids={".py": "python"},
        ),
        str(tmp_path),
        timeout_seconds=2.0,
    )
    client._proc = fake  # type: ignore[assignment]  # test hook: skip spawn
    client._reader_task = asyncio.create_task(client._read_loop())
    client._stderr_task = asyncio.create_task(client._stderr_loop())
    return client, fake


async def _drain_requests(fake: _FakeProc) -> list[dict[str, Any]]:
    for _ in range(200):
        frames = fake.stdin.pop_frames()
        if frames:
            return frames
        await asyncio.sleep(0.005)
    return []


async def _respond_to(fake: _FakeProc, method: str, result: Any) -> dict[str, Any]:
    """Wait for the next client request with ``method`` and answer it."""
    for _ in range(200):
        for msg in fake.stdin.pop_frames():
            if msg.get("method") == method and "id" in msg:
                await fake.stdout.push_frame(
                    {"jsonrpc": "2.0", "id": msg["id"], "result": result},
                )
                return msg
        await asyncio.sleep(0.005)
    raise AssertionError(f"client never sent {method}")


def _uri(tmp_path, rel: str) -> str:
    return Path(os.path.join(str(tmp_path), rel)).as_uri()


# ---------------------------------------------------------------------------
# 1-3. Transport
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_roundtrip_initialize_symbol_references_shutdown(tmp_path):
    (tmp_path / "svc.py").write_text("def handler():\n    pass\n")
    client, fake = _make_client(tmp_path)

    async def server():
        await _respond_to(fake, "initialize", {"capabilities": {}})
        await _respond_to(fake, "workspace/symbol", [{
            "name": "handler", "kind": 12,
            "location": {"uri": _uri(tmp_path, "svc.py"),
                         "range": {"start": {"line": 0, "character": 4}}},
        }])
        await _respond_to(fake, "textDocument/references", [{
            "uri": _uri(tmp_path, "api.py"),
            "range": {"start": {"line": 3, "character": 0}},
        }])

    server_task = asyncio.create_task(server())
    await client.start()
    assert client.alive

    syms = await client.workspace_symbols("handler")
    assert syms[0]["name"] == "handler"
    refs = await client.references("svc.py", 0, 4)
    assert refs[0]["uri"].endswith("api.py")
    await server_task

    # didOpen/didClose notifications were sent around references.
    await asyncio.sleep(0.01)
    methods = [m.get("method") for m in fake.stdin.pop_frames()]
    assert "textDocument/didClose" in methods

    await client.shutdown()
    assert not client.alive


@pytest.mark.asyncio
async def test_server_to_client_request_answered_while_call_in_flight(
    tmp_path, monkeypatch,
):
    # Shrink the readiness-grace sleep so the scripted server's patience
    # window (~1s of polling) comfortably covers the retried request.
    monkeypatch.setattr(lc, "_READINESS_RETRY_DELAY", 0.05)
    client, fake = _make_client(tmp_path)

    async def server():
        init = await _respond_to(fake, "initialize", {"capabilities": {}})
        assert init["params"]["rootUri"].startswith("file://")
        # Client asks workspace/symbol; before answering, the SERVER sends
        # its own request (workspace/configuration) — pyright does exactly
        # this and stalls if unanswered.
        for _ in range(200):
            reqs = [m for m in fake.stdin.pop_frames()
                    if m.get("method") == "workspace/symbol"]
            if reqs:
                sym_req = reqs[0]
                break
            await asyncio.sleep(0.005)
        await fake.stdout.push_frame({
            "jsonrpc": "2.0", "id": 777,
            "method": "workspace/configuration",
            "params": {"items": [{"section": "python"}, {"section": "pyright"}]},
        })
        # Wait for the client's response to id 777.
        for _ in range(200):
            replies = [m for m in fake.stdin.pop_frames()
                       if m.get("id") == 777 and "method" not in m]
            if replies:
                assert replies[0]["result"] == [None, None]
                break
            await asyncio.sleep(0.005)
        else:
            raise AssertionError("client never answered workspace/configuration")
        # Now answer the original request — it must still resolve.
        await fake.stdout.push_frame(
            {"jsonrpc": "2.0", "id": sym_req["id"], "result": []},
        )
        # readiness-grace retry fires a second workspace/symbol on empty.
        await _respond_to(fake, "workspace/symbol", [])

    server_task = asyncio.create_task(server())
    await client.start()
    syms = await client.workspace_symbols("anything")
    assert syms == []
    await server_task
    await client.shutdown()


@pytest.mark.asyncio
async def test_framing_split_across_pushes(tmp_path):
    client, fake = _make_client(tmp_path)

    async def server():
        init = (await _drain_requests(fake))[0]
        body = json.dumps(
            {"jsonrpc": "2.0", "id": init["id"], "result": {}},
        ).encode()
        header = f"Content-Length: {len(body)}\r\n\r\n".encode()
        # Byte-dribble the frame: split inside the header AND inside the body.
        await fake.stdout.push_raw(header[:7])
        await asyncio.sleep(0.01)
        await fake.stdout.push_raw(header[7:] + body[:5])
        await asyncio.sleep(0.01)
        await fake.stdout.push_raw(body[5:])

    server_task = asyncio.create_task(server())
    await client.start()
    assert client.alive
    await server_task
    await client.shutdown()


@pytest.mark.asyncio
async def test_oversize_frame_drops_transport_and_fails_pending(tmp_path):
    client, fake = _make_client(tmp_path)

    async def server():
        init = (await _drain_requests(fake))[0]
        await fake.stdout.push_raw(
            f"Content-Length: {lc._MAX_RPC_BODY_BYTES + 1}\r\n\r\n".encode(),
        )
        _ = init

    server_task = asyncio.create_task(server())
    with pytest.raises(lc.LspError):
        await client.start()
    await server_task
    await client.shutdown()


@pytest.mark.asyncio
async def test_notification_ignored(tmp_path):
    client, fake = _make_client(tmp_path)

    async def server():
        init = (await _drain_requests(fake))[0]
        # Push a notification first — must not confuse correlation.
        await fake.stdout.push_frame({
            "jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
            "params": {"uri": "file:///x.py", "diagnostics": []},
        })
        await fake.stdout.push_frame(
            {"jsonrpc": "2.0", "id": init["id"], "result": {}},
        )

    server_task = asyncio.create_task(server())
    await client.start()
    assert client.alive
    await server_task
    await client.shutdown()


# ---------------------------------------------------------------------------
# 4. Probe matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("venv_dir,require,healthy", [
    (".venv", True, True),
    ("venv", True, True),
    (None, True, False),
    (None, False, True),
])
def test_probe_python(tmp_path, venv_dir, require, healthy):
    if venv_dir:
        (tmp_path / venv_dir).mkdir()
    result = probe_workspace_health(
        "python", str(tmp_path), python_require_venv=require,
    )
    assert result.healthy is healthy
    if not healthy:
        assert result.reason


@pytest.mark.parametrize("tsconfig,node_modules,healthy", [
    (True, True, True),
    (True, False, False),
    (False, True, False),
    (False, False, False),
])
def test_probe_typescript(tmp_path, tsconfig, node_modules, healthy):
    if tsconfig:
        (tmp_path / "tsconfig.json").write_text("{}")
    if node_modules:
        (tmp_path / "node_modules").mkdir()
    assert probe_workspace_health("typescript", str(tmp_path)).healthy is healthy


def test_probe_unknown_server(tmp_path):
    assert not probe_workspace_health("java", str(tmp_path)).healthy


# ---------------------------------------------------------------------------
# 5. Normalization + symbol resolution (stubbed client)
# ---------------------------------------------------------------------------

def test_uri_to_rel_filters(tmp_path):
    ws = str(tmp_path)
    assert _uri_to_rel(_uri(tmp_path, "src/a.py"), ws) == "src/a.py"
    assert _uri_to_rel(Path("/somewhere/else.py").as_uri(), ws) is None
    assert _uri_to_rel(_uri(tmp_path, "node_modules/x/i.ts"), ws) is None
    assert _uri_to_rel(_uri(tmp_path, ".venv/lib/x.py"), ws) is None
    assert _uri_to_rel("untitled:Untitled-1", ws) is None


def test_normalize_locations_shapes_and_dedupe(tmp_path):
    ws = str(tmp_path)
    raw = [
        # plain Location
        {"uri": _uri(tmp_path, "a.py"),
         "range": {"start": {"line": 3, "character": 2}}},
        # duplicate (same file+line)
        {"uri": _uri(tmp_path, "a.py"),
         "range": {"start": {"line": 3, "character": 9}}},
        # LocationLink
        {"targetUri": _uri(tmp_path, "b.py"),
         "targetSelectionRange": {"start": {"line": 7, "character": 0}}},
        # WorkspaceSymbol with range-less location
        {"name": "X", "location": {"uri": _uri(tmp_path, "c.py")}},
        # self-file exclusion
        {"uri": _uri(tmp_path, "self.py"),
         "range": {"start": {"line": 0, "character": 0}}},
    ]
    out = _normalize_locations(raw, ws, exclude_rel="self.py")
    assert [(o["file"], o["line"]) for o in out] == [
        ("a.py", 3), ("b.py", 7), ("c.py", 0),
    ]
    assert _normalize_locations(None, ws) == []
    # single Location dict (definition responses)
    assert _normalize_locations(raw[0], ws)[0]["file"] == "a.py"


class _StubClient:
    """Duck-typed StdioLspClient for pool-level helper tests."""

    def __init__(self, name, language_ids, *, symbols=None, refs=None,
                 doc_symbols=None):
        self.config = LspServerConfig(name, ["stub"], language_ids)
        self._started = True
        self._proc = None
        self._symbols = symbols or []
        self._refs = refs or {}
        self._doc_symbols = doc_symbols or []
        self.reference_calls: list[tuple[str, int, int]] = []

    @property
    def alive(self) -> bool:
        return self._started

    async def workspace_symbols(self, query):
        return [s for s in self._symbols if query in s.get("name", "")]

    async def document_symbols(self, rel_path):
        return list(self._doc_symbols)

    async def references(self, rel_path, line, character, *,
                         include_declaration=False):
        self.reference_calls.append((rel_path, line, character))
        return self._refs.get((rel_path, line), [])

    async def shutdown(self):
        self._started = False


def _stub_pool(tmp_path, client) -> LspClientPool:
    pool = LspClientPool(LspPoolConfig(), str(tmp_path))
    pool.clients[client.config.name] = client
    return pool


@pytest.mark.asyncio
async def test_find_symbol_locations_ranking(tmp_path):
    sym = lambda name, rel, line: {  # noqa: E731
        "name": name, "kind": 12,
        "location": {"uri": _uri(tmp_path, rel),
                     "range": {"start": {"line": line, "character": 0}}},
    }
    client = _StubClient("python", {".py": "python"}, symbols=[
        sym("get_user_name", "other.py", 1),   # partial match
        sym("get_user", "far.py", 2),          # exact, no file hint
        sym("get_user", "svc.py", 3),          # exact + file hint
    ])
    pool = _stub_pool(tmp_path, client)
    out = await find_symbol_locations(pool, "get_user", "svc.py")
    assert out[0]["file"] == "svc.py"           # exact + hint ranks first
    assert out[1]["file"] == "far.py"           # exact next
    assert out[-1]["file"] == "other.py"        # partial last


@pytest.mark.asyncio
async def test_find_references_by_symbol_end_to_end(tmp_path):
    definition = {
        "name": "handler", "kind": 12,
        "location": {"uri": _uri(tmp_path, "svc.py"),
                     "range": {"start": {"line": 5, "character": 4}}},
    }
    refs = [
        {"uri": _uri(tmp_path, "api.py"),
         "range": {"start": {"line": 2, "character": 0}}},
        {"uri": _uri(tmp_path, "node_modules/lib.py"),
         "range": {"start": {"line": 9, "character": 0}}},
    ]
    client = _StubClient("python", {".py": "python"},
                         symbols=[definition], refs={("svc.py", 5): refs})
    pool = _stub_pool(tmp_path, client)
    out = await find_references_by_symbol(pool, "handler")
    assert out == [{"file": "api.py", "line": 2, "character": 0}]


@pytest.mark.asyncio
async def test_callers_of_file_shape_and_private_skip(tmp_path):
    doc_symbols = [
        {"name": "PublicService", "kind": 5,
         "selectionRange": {"start": {"line": 1, "character": 6}},
         "children": []},
        {"name": "_private_helper", "kind": 12,
         "selectionRange": {"start": {"line": 9, "character": 4}}},
        {"name": "do_work", "kind": 12,
         "selectionRange": {"start": {"line": 20, "character": 4}}},
    ]
    refs = {
        ("svc.py", 1): [{"uri": _uri(tmp_path, "api.py"),
                         "range": {"start": {"line": 0, "character": 0}}}],
        ("svc.py", 20): [
            {"uri": _uri(tmp_path, "api.py"),
             "range": {"start": {"line": 4, "character": 0}}},
            {"uri": _uri(tmp_path, "svc.py"),   # self — excluded
             "range": {"start": {"line": 30, "character": 0}}},
        ],
    }
    client = _StubClient("python", {".py": "python"},
                         doc_symbols=doc_symbols, refs=refs)
    pool = _stub_pool(tmp_path, client)
    out = await callers_of_file(pool, "svc.py")
    assert out == {"api.py": {"PublicService", "do_work"}}
    assert all(call[0] == "svc.py" for call in client.reference_calls)
    assert len(client.reference_calls) == 2      # _private_helper skipped


@pytest.mark.asyncio
async def test_callers_of_file_budget_zero_returns_partial(tmp_path):
    client = _StubClient("python", {".py": "python"}, doc_symbols=[
        {"name": "a", "kind": 12,
         "selectionRange": {"start": {"line": 0, "character": 0}}},
    ])
    pool = _stub_pool(tmp_path, client)
    out = await callers_of_file(pool, "svc.py", budget_seconds=0.0)
    assert out == {}                              # deadline before first query
    assert client.reference_calls == []


# ---------------------------------------------------------------------------
# 6. Pool start gating
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pool_start_skips_missing_binary_and_unhealthy(tmp_path, monkeypatch):
    monkeypatch.setattr(lc.shutil, "which",
                        lambda t: "/bin/x" if t == "pyright-langserver" else None)
    # python candidate: binary present but no venv → probe-unhealthy.
    # typescript candidate: binary missing.
    pool = LspClientPool(LspPoolConfig(), str(tmp_path))
    await pool.start({"python", "typescript"})
    assert pool.clients == {}
    reasons = {s["server"]: s["reason"] for s in pool.skipped}
    assert "venv" in reasons["python"]
    assert "not on PATH" in reasons["typescript"]
    assert not pool.healthy()


@pytest.mark.asyncio
async def test_pool_disabled_or_no_tags_spawns_nothing(tmp_path):
    pool = LspClientPool(LspPoolConfig(enabled=False), str(tmp_path))
    await pool.start({"python"})
    assert pool.clients == {} and pool.skipped == []
    pool2 = LspClientPool(LspPoolConfig(), str(tmp_path))
    await pool2.start({"java"})
    assert pool2.clients == {}


def test_client_for_file_dead_server_logged_once(tmp_path):
    client = _StubClient("python", {".py": "python"})
    client._started = False
    pool = _stub_pool(tmp_path, client)
    assert pool.client_for_file("a.py") is None
    assert pool.client_for_file("b.py") is None
    assert pool._dead_logged == {"python"}
    assert pool.client_for_file("c.ts") is None   # no client for ext


# ---------------------------------------------------------------------------
# 7. DSL + skills
# ---------------------------------------------------------------------------

def test_parse_lsp_blocks():
    text = (
        'plan text <<<LSP_CALL tool="find_references" symbol="UserService" '
        'file="server/user.py">>> and '
        "<<<LSP_CALL tool='go_to_definition' symbol='save'>>> plus "
        '<<<LSP_CALL tool="rename" symbol="x">>>'
    )
    blocks = parse_lsp_blocks(text)
    assert len(blocks) == 3
    assert blocks[0].skill_name == "lsp__find_references"
    assert blocks[0].kwargs == {"symbol": "UserService", "file": "server/user.py"}
    assert blocks[1].skill_name == "lsp__go_to_definition"
    assert blocks[1].kwargs == {"symbol": "save"}
    assert blocks[2].skill_name == ""             # unknown tool → not registered
    stripped = strip_lsp_blocks(text)
    assert "LSP_CALL" not in stripped and "plan text" in stripped
    assert parse_lsp_blocks("no blocks here") == []


@pytest.mark.asyncio
async def test_skills_fail_open_without_pool(tmp_path):
    clear_active_pool()
    from harness.lsp_client import register_lsp_skills
    pool = LspClientPool(LspPoolConfig(), str(tmp_path))
    assert register_lsp_skills(pool) == 2
    from harness.skills import SkillRegistry
    skill = SkillRegistry().get("lsp__find_references")
    result = await skill.execute(symbol="X")
    assert "error" in result and "unavailable" in result["error"]
    # missing symbol also fails politely with an active (empty) pool
    set_active_pool(pool)
    try:
        result = await skill.execute(symbol="")
        assert "error" in result
    finally:
        clear_active_pool()


@pytest.mark.asyncio
async def test_skill_returns_locations_with_stub_pool(tmp_path):
    definition = {
        "name": "handler", "kind": 12,
        "location": {"uri": _uri(tmp_path, "svc.py"),
                     "range": {"start": {"line": 5, "character": 4}}},
    }
    client = _StubClient("python", {".py": "python"}, symbols=[definition],
                         refs={("svc.py", 5): [
                             {"uri": _uri(tmp_path, "api.py"),
                              "range": {"start": {"line": 1, "character": 0}}},
                         ]})
    pool = _stub_pool(tmp_path, client)
    set_active_pool(pool)
    try:
        from harness.lsp_client import _skill_find_references
        result = await _skill_find_references(symbol="handler")
        assert result["count"] == 1
        assert result["locations"][0]["file"] == "api.py"
    finally:
        clear_active_pool()


# ---------------------------------------------------------------------------
# 7b. Harness integration sites (three-tier fallback)
# ---------------------------------------------------------------------------

def _ws_ctx_kwargs(tmp_path):
    (tmp_path / "svc.py").write_text("def do_work():\n    return 1\n")
    (tmp_path / "api.py").write_text("from svc import do_work\n")
    return dict(
        files=[str(tmp_path / "svc.py")],
        workspace_path=str(tmp_path),
        loop_counter={},
    )


def test_formatter_uses_lsp_caller_map_when_provided(tmp_path):
    from harness.graph import _format_workspace_context_for_files
    kwargs = _ws_ctx_kwargs(tmp_path)
    out = _format_workspace_context_for_files(
        **kwargs,
        lsp_caller_maps={"svc.py": {"handlers/api.py": {"do_work"}}},
    )
    # LSP-provided caller (a path the dep graph would never produce for
    # this workspace) is rendered in the exact existing line format.
    assert "`handlers/api.py` imports: do_work" in out


def test_formatter_without_lsp_map_matches_default(tmp_path):
    from harness.graph import _format_workspace_context_for_files
    kwargs = _ws_ctx_kwargs(tmp_path)
    default = _format_workspace_context_for_files(**kwargs)
    explicit_none = _format_workspace_context_for_files(
        **kwargs, lsp_caller_maps=None,
    )
    assert default == explicit_none          # byte-identical fallback
    assert "api.py" in default               # dep-graph tier still works


def test_cr_impact_augment_lsp_tier_and_fallback(tmp_path, monkeypatch):
    from harness import graph as graph_mod
    (tmp_path / "util.py").write_text("def helper():\n    pass\n")
    (tmp_path / "caller.py").write_text("from util import helper\n")
    state = {
        "change_request_mode": True,
        "workspace_path": str(tmp_path),
        "modified_files": [],
    }
    # LSP tier: provided callers used verbatim, DependencyGraph not needed.
    out = graph_mod._cr_impact_augment(
        state, ["util.py"], lsp_callers=["caller.py", "other.py"],
    )
    assert "caller.py" in out
    # Fallback tier: None → DependencyGraph path (real graph, tiny repo).
    out2 = graph_mod._cr_impact_augment(state, ["util.py"], lsp_callers=None)
    assert any(p.endswith("caller.py") for p in out2)


@pytest.mark.asyncio
async def test_prefetch_helpers_with_stub_pool(tmp_path):
    """Covers the prefetch helpers end-to-end with a live-ish stub pool —
    a swallowed programming error in them (they're fail-open) would
    silently disable Sites B/C, so this test drives the success path."""
    from harness.graph import (
        _prefetch_lsp_caller_maps, _prefetch_lsp_immediate_callers,
    )
    client = _StubClient(
        "python", {".py": "python"},
        doc_symbols=[{"name": "do_work", "kind": 12,
                      "selectionRange": {"start": {"line": 0, "character": 4}}}],
        refs={("svc.py", 0): [
            {"uri": _uri(tmp_path, "api.py"),
             "range": {"start": {"line": 1, "character": 0}}},
        ]},
    )
    set_active_pool(_stub_pool(tmp_path, client))
    try:
        maps = await _prefetch_lsp_caller_maps(
            [str(tmp_path / "svc.py")], str(tmp_path))
        assert maps == {"svc.py": {"api.py": {"do_work"}}}
        callers = await _prefetch_lsp_immediate_callers(
            ["svc.py"], str(tmp_path))
        assert callers == ["api.py"]
    finally:
        clear_active_pool()
    # No pool → empty / None, never raises.
    assert await _prefetch_lsp_caller_maps(["svc.py"], str(tmp_path)) == {}
    assert await _prefetch_lsp_immediate_callers(["svc.py"], str(tmp_path)) is None


@pytest.mark.asyncio
async def test_gate_lsp_expansion_and_fallback(tmp_path):
    import harness.diagnostics_gate as dg
    (tmp_path / "svc.py").write_text("x = 1\n")
    (tmp_path / "api.py").write_text("import svc\n")
    files = [str(tmp_path / "svc.py")]
    # No pool → None (fall back).
    clear_active_pool()
    assert await dg._expand_with_impacted_lsp(files, str(tmp_path)) is None
    # Stub pool with callers → expanded list, same shape as heuristic tier.
    client = _StubClient(
        "python", {".py": "python"},
        doc_symbols=[{"name": "x", "kind": 13,
                      "selectionRange": {"start": {"line": 0, "character": 0}}}],
        refs={("svc.py", 0): [
            {"uri": _uri(tmp_path, "api.py"),
             "range": {"start": {"line": 0, "character": 7}}},
        ]},
    )
    set_active_pool(_stub_pool(tmp_path, client))
    try:
        out = await dg._expand_with_impacted_lsp(files, str(tmp_path))
        assert out is not None
        assert files[0] in out
        assert str(tmp_path / "api.py") in out
        # Empty caller result → None (never masks the heuristic tier).
        client._refs = {}
        assert await dg._expand_with_impacted_lsp(files, str(tmp_path)) is None
    finally:
        clear_active_pool()


@pytest.mark.asyncio
async def test_gate_batch_mode_scopes_expansion_to_batch_files(tmp_path, monkeypatch):
    """Agile (batch) brownfield: the gate's expansion input must be the
    CURRENT BATCH's files, not the cumulative session set."""
    import harness.diagnostics_gate as dg
    batch_file = tmp_path / "batch.py"
    old_file = tmp_path / "earlier.py"
    batch_file.write_text("b = 1\n")
    old_file.write_text("a = 1\n")
    captured: dict[str, Any] = {}

    async def fake_lsp_expand(files, ws):
        captured["files"] = list(files)
        return None                            # force heuristic tier
    monkeypatch.setattr(dg, "_expand_with_impacted_lsp", fake_lsp_expand)
    monkeypatch.setattr(dg, "_expand_with_impacted", lambda files, ws: files)
    monkeypatch.setattr(dg, "detect_checkers", lambda *a, **k: [])

    async def fake_git(args, cwd, timeout=30):
        return 0, "abc123"
    monkeypatch.setattr(dg, "_git", fake_git)
    import harness.lintgate as lg
    monkeypatch.setattr(lg, "_classify_files_by_git_status",
                        lambda files, ws: (set(files), set()))

    state = {
        "workspace_path": str(tmp_path),
        "modified_files": [str(old_file), str(batch_file)],
        "current_batch_id": 2,
        "batch_modified_files": [str(batch_file)],
        "node_state": {},
        "loop_counter": {},
        "session_id": "batchsess",
        "diagnostics_config": {"enabled": True, "scope": "impacted"},
    }
    await dg.diagnostics_node(state)
    assert captured["files"] == [str(batch_file)]


# ---------------------------------------------------------------------------
# 8. Config parsing
# ---------------------------------------------------------------------------

def test_lsp_section_registered_with_strict_validator():
    """Locks the sync between LspPoolConfig fields and cli.py's strict
    config validator — an lsp key the validator doesn't know about would
    make the shipped config.json fail validation (the exact regression
    the dashboard-wizard test caught for the diagnostics section)."""
    from dataclasses import fields
    from harness.cli import _KNOWN_NESTED_KEYS, _KNOWN_TOP_LEVEL_KEYS
    assert "lsp" in _KNOWN_TOP_LEVEL_KEYS
    assert _KNOWN_NESTED_KEYS["lsp"] == {f.name for f in fields(LspPoolConfig)}


def test_pool_config_from_config():
    cfg = LspPoolConfig.from_config({
        "lsp": {"enabled": False, "enabled_flows": ["patch"],
                "request_timeout_seconds": 0.1,     # clamped to 1.0
                "python_require_venv": False,
                "prefetch_budget_seconds": 5},
    })
    assert cfg.enabled is False
    assert cfg.enabled_flows == ["patch"]
    assert cfg.request_timeout_seconds == 1.0
    assert cfg.python_require_venv is False
    assert cfg.prefetch_budget_seconds == 5.0
    assert LspPoolConfig.from_config({}).enabled is True
    assert LspPoolConfig.from_config(None).enabled_flows == ["patch", "test"]
