"""Regression tests for the web tools slice.

Covers:
    - SSRF guard in ``harness.trust.validate_outbound_url``.
    - HTML → text helper produces readable output without script bodies.
    - DSL parser extracts ``<<<WEB_FETCH>>>`` / ``<<<WEB_SEARCH>>>`` blocks
      with mixed quoted and integer kwargs.
    - DSL parser is robust to malformed blocks (returns nothing rather
      than raising).
    - ``WebFetchSkill`` short-circuits when ``web_tools.enabled=false``.
    - ``WebFetchSkill`` enforces content-type allowlist + byte cap.
    - ``WebSearchSkill`` returns structured results via a stub backend.
    - ``register_builtin_skills(config=…)`` registers web tools only
      when enabled.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

import harness.web_tools as web_tools_module
from harness.skills import SkillRegistry, register_builtin_skills
from harness.trust import validate_outbound_url
from harness.web_tools import (
    SearchResult,
    WebFetchSkill,
    WebSearchSkill,
    WebToolsConfig,
    html_to_text,
    parse_tool_blocks,
    strip_tool_blocks,
)


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------

def test_validate_outbound_url_accepts_https():
    assert validate_outbound_url("https://docs.python.org/3/") == "https://docs.python.org/3/"


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "ftp://example.com/x",
    "javascript:alert(1)",
    "data:text/plain;base64,YWJj",
])
def test_validate_outbound_url_rejects_unsafe_scheme(url):
    with pytest.raises(ValueError):
        validate_outbound_url(url)


@pytest.mark.parametrize("url", [
    "http://localhost/admin",
    "http://127.0.0.1:8080/",
    "http://169.254.169.254/latest/meta-data",   # AWS metadata
    "http://10.0.0.5/",
    "http://192.168.1.10/",
    "http://172.16.0.1/",
])
def test_validate_outbound_url_blocks_private_ips_by_default(url):
    with pytest.raises(ValueError):
        validate_outbound_url(url)


def test_validate_outbound_url_allows_private_when_opted_in():
    assert validate_outbound_url(
        "http://10.0.0.5/", allow_private_ips=True,
    ) == "http://10.0.0.5/"


def test_validate_outbound_url_rejects_empty():
    with pytest.raises(ValueError):
        validate_outbound_url("")


# ---------------------------------------------------------------------------
# HTML → text
# ---------------------------------------------------------------------------

def test_html_to_text_strips_scripts_and_decodes_entities():
    html = """<html><head><script>alert(1)</script><style>body{}</style></head>
    <body><h1>Title&nbsp;&amp;&nbsp;Subtitle</h1>
    <p>Para&nbsp;one</p>

    <p>Para two&#x26;final</p></body></html>"""
    text = html_to_text(html)
    assert "alert" not in text
    assert "body{}" not in text
    assert "Title & Subtitle" in text
    assert "Para one" in text
    assert "Para two&final" in text


# ---------------------------------------------------------------------------
# DSL parser
# ---------------------------------------------------------------------------

def test_parse_tool_blocks_handles_mixed_kwargs():
    content = (
        'Some preamble.\n'
        '<<<WEB_FETCH url="https://docs.python.org/" max_bytes=50000>>>\n'
        'Then later:\n'
        '<<<WEB_SEARCH query="python asyncio cancellation" max_results=3>>>\n'
        'And done.'
    )
    blocks = parse_tool_blocks(content)
    assert len(blocks) == 2
    assert blocks[0].skill_name == "web_fetch"
    assert blocks[0].kwargs == {"url": "https://docs.python.org/", "max_bytes": 50000}
    assert blocks[1].skill_name == "web_search"
    assert blocks[1].kwargs == {"query": "python asyncio cancellation", "max_results": 3}


def test_parse_tool_blocks_ignores_content_without_blocks():
    assert parse_tool_blocks("nothing to see here") == []
    assert parse_tool_blocks("") == []
    assert parse_tool_blocks(None) == []  # type: ignore[arg-type]


def test_strip_tool_blocks_removes_blocks_keeps_surrounding_text():
    content = (
        'Heading\n'
        '<<<WEB_FETCH url="https://example.com">>>\n'
        'Trailing.'
    )
    stripped = strip_tool_blocks(content)
    assert "<<<" not in stripped
    assert "Heading" in stripped
    assert "Trailing." in stripped


# ---------------------------------------------------------------------------
# WebFetchSkill — disabled guard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_web_fetch_skill_returns_error_when_disabled():
    cfg = WebToolsConfig(enabled=False)
    skill = WebFetchSkill(cfg)
    result = await skill.execute(url="https://example.com/")
    assert isinstance(result, dict)
    assert "disabled" in result["error"].lower()


@pytest.mark.asyncio
async def test_web_fetch_skill_rejects_localhost_url_via_ssrf_guard():
    cfg = WebToolsConfig(enabled=True)
    skill = WebFetchSkill(cfg)
    result = await skill.execute(url="http://localhost/admin")
    assert "url rejected" in result["error"]


# ---------------------------------------------------------------------------
# WebFetchSkill — HTTP path stubbed via httpx mock transport
# ---------------------------------------------------------------------------

class _MockAsyncClient:
    """Stand-in for httpx.AsyncClient used by the fetch path."""

    def __init__(self, response: httpx.Response, captured: list[str]):
        self._response = response
        self._captured = captured

    async def __aenter__(self) -> "_MockAsyncClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def get(self, url: str) -> httpx.Response:
        self._captured.append(url)
        return self._response


@pytest.mark.asyncio
async def test_web_fetch_skill_truncates_at_max_bytes(monkeypatch):
    cfg = WebToolsConfig(enabled=True, max_bytes=64)
    skill = WebFetchSkill(cfg)
    big_body = b"a" * 10_000
    fake_response = httpx.Response(
        status_code=200,
        content=big_body,
        headers={"content-type": "text/plain"},
        request=httpx.Request("GET", "https://example.com/data"),
    )
    captured: list[str] = []
    monkeypatch.setattr(
        web_tools_module.httpx, "AsyncClient",
        lambda **_kw: _MockAsyncClient(fake_response, captured),
    )
    result = await skill.execute(url="https://example.com/data")
    assert result["truncated"] is True
    assert result["bytes_returned"] == 64
    assert len(result["content"]) <= 64
    assert captured == ["https://example.com/data"]


@pytest.mark.asyncio
async def test_web_fetch_skill_rejects_unwhitelisted_content_type(monkeypatch):
    cfg = WebToolsConfig(enabled=True)
    skill = WebFetchSkill(cfg)
    fake_response = httpx.Response(
        status_code=200,
        content=b"\x00\x01\x02",
        headers={"content-type": "application/octet-stream"},
        request=httpx.Request("GET", "https://example.com/blob"),
    )
    monkeypatch.setattr(
        web_tools_module.httpx, "AsyncClient",
        lambda **_kw: _MockAsyncClient(fake_response, []),
    )
    result = await skill.execute(url="https://example.com/blob")
    assert "content-type" in result["error"]


# ---------------------------------------------------------------------------
# WebSearchSkill — stub backend
# ---------------------------------------------------------------------------

class _StubBackend:
    name = "stub"

    def __init__(self, *, timeout_seconds: float = 0):  # noqa: ARG002
        return

    async def search(self, query: str, max_results: int):  # noqa: ARG002
        return [
            SearchResult(title="Result 1", url="https://r1.example/", snippet="s1"),
            SearchResult(title="Result 2", url="https://r2.example/", snippet="s2"),
            SearchResult(title="Result 3", url="https://r3.example/", snippet="s3"),
        ][:max_results]


@pytest.mark.asyncio
async def test_web_search_skill_returns_structured_results():
    cfg = WebToolsConfig(enabled=True, max_results=2, search_backend="stub")
    skill = WebSearchSkill(cfg)
    result = await skill._call(  # type: ignore[attr-defined]
        query="anything",
        _backend_factory=lambda name, timeout_seconds: _StubBackend(),
    )
    assert result["query"] == "anything"
    assert result["backend"] == "stub"
    assert len(result["results"]) == 2
    assert result["results"][0]["title"] == "Result 1"


@pytest.mark.asyncio
async def test_web_search_skill_rejects_empty_query():
    cfg = WebToolsConfig(enabled=True)
    skill = WebSearchSkill(cfg)
    result = await skill.execute(query="")
    assert "non-empty" in result["error"]


# ---------------------------------------------------------------------------
# Registry plumbing
# ---------------------------------------------------------------------------

def _drop_web_skills() -> None:
    """Wipe web skills from the (singleton) registry so each test starts
    from a known shape."""
    reg = SkillRegistry()
    for name in ("web_fetch", "web_search"):
        reg._skills.pop(name, None)  # type: ignore[attr-defined]


def test_register_builtin_skills_omits_web_tools_when_disabled():
    _drop_web_skills()
    register_builtin_skills(config={"web_tools": {"enabled": False}})
    reg = SkillRegistry()
    assert reg.get("web_fetch") is None
    assert reg.get("web_search") is None


def test_register_builtin_skills_registers_web_tools_when_enabled():
    _drop_web_skills()
    register_builtin_skills(config={"web_tools": {"enabled": True}})
    reg = SkillRegistry()
    assert reg.get("web_fetch") is not None
    assert reg.get("web_search") is not None
    _drop_web_skills()  # leave the registry clean for downstream tests


def test_register_builtin_skills_works_without_config_arg():
    """Historical call signature with no kwargs must still work — only
    pipeline + docgen skills register."""
    _drop_web_skills()
    register_builtin_skills()  # no config
    reg = SkillRegistry()
    assert reg.get("web_fetch") is None


# ---------------------------------------------------------------------------
# Configure-page overhaul: multi-instance web tools (backends list)
# ---------------------------------------------------------------------------

def test_web_tools_config_round_trips_backends_list():
    from harness.web_tools import WebToolsConfig
    cfg = WebToolsConfig.from_config({
        "web_tools": {
            "enabled": True,
            "search_backend": "duckduckgo_lite",
            "backends": [
                {"name": "brave", "enabled": True,
                 "search_backend": "brave", "api_key_env": "BRAVE_KEY"},
                {"name": "google", "enabled": False,
                 "search_backend": "google", "api_key_env": "GOOGLE_KEY"},
            ],
        }
    })
    assert len(cfg.backends) == 2
    names = [b["name"] for b in cfg.backends]
    assert "brave" in names and "google" in names


def test_web_tools_active_backends_filters_disabled_and_orders_primary_first():
    from harness.web_tools import WebToolsConfig
    cfg = WebToolsConfig.from_config({
        "web_tools": {
            "enabled": True,
            "search_backend": "duckduckgo_lite",
            "backends": [
                {"name": "brave", "enabled": True, "search_backend": "brave"},
                {"name": "google", "enabled": False, "search_backend": "google"},
                {"name": "no_backend", "enabled": True, "search_backend": ""},
            ],
        }
    })
    active = cfg.active_backends()
    backends = [b["search_backend"] for b in active]
    assert backends == ["duckduckgo_lite", "brave"]


def test_web_tools_active_backends_skips_primary_when_blank():
    from harness.web_tools import WebToolsConfig
    cfg = WebToolsConfig.from_config({
        "web_tools": {
            "search_backend": "",
            "backends": [
                {"name": "brave", "enabled": True, "search_backend": "brave"},
            ],
        }
    })
    active = cfg.active_backends()
    assert [b["search_backend"] for b in active] == ["brave"]


def test_web_tools_config_legacy_shape_still_loads():
    """Configs without a ``backends`` key keep working — the field
    defaults to an empty list."""
    from harness.web_tools import WebToolsConfig
    cfg = WebToolsConfig.from_config({
        "web_tools": {"enabled": True, "search_backend": "duckduckgo_lite"}
    })
    assert cfg.backends == []
    active = cfg.active_backends()
    assert active and active[0]["search_backend"] == "duckduckgo_lite"
