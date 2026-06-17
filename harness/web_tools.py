"""Web research tools — ``WebFetchSkill`` and ``WebSearchSkill``.

These skills give the planner / patcher / repair LLM the ability to read
external documentation and search the web before it commits to an
implementation plan. They follow the same ``SkillBase`` contract as the
docgen + pipeline skills already in :mod:`harness.skills`, so the
:class:`SkillRegistry` registers and dispatches them uniformly.

Why text-DSL and not native function-calling? The harness's existing
patcher pattern is text-DSL (``<<<CREATE_FILE>>>``, ``<<<READ_FILE>>>``,
SEARCH/REPLACE blocks). ``GatewayConfig.use_structured_tools`` is shipped
``false`` because the per-provider native tool-use wiring isn't complete
yet. Sticking with text-DSL keeps the LLM contract uniform: when
``use_structured_tools`` lands later, the same ``ToolSkill.to_tool_schema``
JSON is reusable as the native function schema.

The LLM emits a block like::

    <<<WEB_FETCH url="https://docs.python.org/3/library/asyncio.html" max_bytes=120000>>>
    <<<WEB_SEARCH query="python asyncio task cancellation best practices" max_results=5>>>

and the graph's tool-interceptor (see :func:`harness.graph.execute_tool_blocks`)
parses, dispatches via the registry, and feeds the result back into the
conversation as a ``user`` message so the next dispatch sees it. The
interceptor is capped to a small number of rounds per node call so the
LLM cannot loop.

Security model
==============
Every URL is run through :func:`harness.trust.validate_outbound_url`
before any HTTP call so the LLM cannot trick the harness into hitting
cloud-metadata endpoints (SSRF), localhost services, or RFC-1918 hosts
unless the operator opts in via ``web_tools.allow_private_ips=true``.
Result content is also pushed through :func:`harness.redactor.redact_messages`
before re-entering the LLM conversation so fetched HTML can't smuggle
API keys or other secrets back into the next dispatch.

Default search backend is ``duckduckgo_lite`` — no API key required —
because the goal is "works out of the box". For volume / enterprise
usage swap in ``tavily`` / ``brave`` / ``serpapi`` via
``web_tools.search_backend`` once the slice that adds them lands.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

from harness.skills import (
    SkillParameter,
    SkillSchema,
    SkillType,
    ToolSkill,
)
from harness.trust import validate_outbound_url

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Defaults + config dataclass
# ---------------------------------------------------------------------------

_DEFAULT_MAX_BYTES = 200_000
_DEFAULT_MAX_RESULTS = 5
_DEFAULT_TIMEOUT_SECONDS = 20.0
_ALLOWED_CONTENT_TYPES = (
    "text/html",
    "text/plain",
    "text/markdown",
    "application/json",
    "application/ld+json",
    "application/xml",
    "text/xml",
)


@dataclass
class WebToolsConfig:
    """Runtime config for the web tools, materialised from the
    ``web_tools`` section of ``config/config.json``. Every knob has a
    sane default so callers can construct a ``WebToolsConfig()`` for
    tests without a config file.
    """

    enabled: bool = False
    max_bytes: int = _DEFAULT_MAX_BYTES
    max_results: int = _DEFAULT_MAX_RESULTS
    search_backend: str = "duckduckgo_lite"
    api_key_env: str = ""
    allow_private_ips: bool = False
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    # Cap on web tool dispatches per single graph-node call. Without this
    # the LLM can loop indefinitely on web_fetch → web_fetch → ...
    tool_call_cap_per_dispatch: int = 3
    # Additional search-backend entries beyond the primary defined by
    # ``search_backend`` above. Each item: {name, enabled, search_backend,
    # api_key_env}. The configure page renders this as a + Add list so
    # operators can register multiple backends alongside the primary.
    backends: list[dict[str, Any]] = field(default_factory=list)

    def active_backends(self) -> list[dict[str, Any]]:
        """All configured backends in dispatch order: primary first
        (from the top-level scalars) then every enabled entry from
        ``backends``. Disabled entries are filtered out."""
        out: list[dict[str, Any]] = []
        if self.search_backend:
            out.append({
                "name": "primary",
                "enabled": self.enabled,
                "search_backend": self.search_backend,
                "api_key_env": self.api_key_env,
            })
        for entry in self.backends:
            if not isinstance(entry, dict):
                continue
            if not bool(entry.get("enabled", True)):
                continue
            if not str(entry.get("search_backend") or "").strip():
                continue
            out.append({
                "name": str(entry.get("name") or entry.get("search_backend") or ""),
                "enabled": True,
                "search_backend": str(entry["search_backend"]),
                "api_key_env": str(entry.get("api_key_env") or ""),
            })
        return out

    @classmethod
    def from_config(cls, config: Optional[dict[str, Any]]) -> "WebToolsConfig":
        section = ((config or {}).get("web_tools") or {})
        raw_backends = section.get("backends") or []
        backends: list[dict[str, Any]] = []
        if isinstance(raw_backends, list):
            for entry in raw_backends:
                if isinstance(entry, dict):
                    backends.append(dict(entry))
        return cls(
            enabled=bool(section.get("enabled", False)),
            max_bytes=int(section.get("max_bytes", _DEFAULT_MAX_BYTES)),
            max_results=int(section.get("max_results", _DEFAULT_MAX_RESULTS)),
            search_backend=str(section.get("search_backend", "duckduckgo_lite")),
            api_key_env=str(section.get("api_key_env", "")),
            allow_private_ips=bool(section.get("allow_private_ips", False)),
            timeout_seconds=float(
                section.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)
            ),
            tool_call_cap_per_dispatch=int(
                section.get("tool_call_cap_per_dispatch", 3)
            ),
            backends=backends,
        )


# ---------------------------------------------------------------------------
# 2. HTML → text helper
# ---------------------------------------------------------------------------

_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE
)
_TAG_RE = re.compile(r"<[^>]+>")
_ENTITY_RE = re.compile(r"&(?:nbsp|amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);")
_ENTITY_MAP = {
    "&nbsp;": " ",
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&apos;": "'",
}


def _decode_entities(text: str) -> str:
    """Decode the handful of HTML entities we expect to encounter without
    pulling in ``html.unescape`` (which has surprising behaviour on
    malformed input)."""
    def _sub(match: re.Match[str]) -> str:
        ent = match.group(0)
        if ent in _ENTITY_MAP:
            return _ENTITY_MAP[ent]
        if ent.startswith("&#x"):
            try:
                return chr(int(ent[3:-1], 16))
            except (ValueError, OverflowError):
                return ent
        if ent.startswith("&#"):
            try:
                return chr(int(ent[2:-1]))
            except (ValueError, OverflowError):
                return ent
        return ent
    return _ENTITY_RE.sub(_sub, text)


def html_to_text(html: str) -> str:
    """Strip tags from an HTML document and collapse whitespace.

    Deliberately dumb. The harness is reading docs, not rendering a SPA;
    if a page needs JS to be useful, the operator should swap in a
    headless-browser backend later. For 95% of doc / blog / spec pages
    this gets the readable bytes out cheaply.
    """
    if not html:
        return ""
    no_scripts = _SCRIPT_STYLE_RE.sub(" ", html)
    no_tags = _TAG_RE.sub(" ", no_scripts)
    decoded = _decode_entities(no_tags)
    # Collapse runs of whitespace, including newlines, into single spaces
    # while preserving paragraph boundaries (double-newline pattern).
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", decoded)]
    return "\n\n".join(p for p in paragraphs if p)


# ---------------------------------------------------------------------------
# 3. Search backends
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


class SearchBackend(ABC):
    name: str = "base"

    @abstractmethod
    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        ...


class DuckDuckGoLiteBackend(SearchBackend):
    """Scrapes the lightweight DuckDuckGo HTML endpoint. No API key.

    Rate-limit yourself; DDG will throttle aggressive scrapers. Suitable
    for occasional harness research calls (a few queries per session).
    For higher volume swap in Tavily / Brave / SerpAPI.
    """

    name = "duckduckgo_lite"

    _URL = "https://html.duckduckgo.com/html/"
    _RESULT_RE = re.compile(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
        r'(?:.*?<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>)?',
        re.DOTALL | re.IGNORECASE,
    )

    def __init__(self, timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS):
        self._timeout = timeout_seconds

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        params = {"q": query}
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout, connect=10.0),
            follow_redirects=True,
            headers={
                "User-Agent": "myharness-research/1.0 (+https://github.com/akmontheweb/myharness)",
            },
        ) as client:
            response = await client.post(self._URL, data=params)
            response.raise_for_status()
            html = response.text
        results: list[SearchResult] = []
        for match in self._RESULT_RE.finditer(html):
            url = match.group(1).strip()
            title_html = match.group(2) or ""
            snippet_html = match.group(3) or ""
            title = html_to_text(title_html).strip()
            snippet = html_to_text(snippet_html).strip()
            # DDG wraps real URLs in a redirect: /l/?uddg=<encoded-url>
            if url.startswith("/l/?") or url.startswith("//duckduckgo.com/l/?"):
                from urllib.parse import parse_qs
                qs = parse_qs(urlparse(url).query)
                if "uddg" in qs and qs["uddg"]:
                    url = qs["uddg"][0]
            if not url or not title:
                continue
            results.append(SearchResult(title=title, url=url, snippet=snippet))
            if len(results) >= max_results:
                break
        return results


def make_search_backend(
    name: str, *, timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
) -> SearchBackend:
    """Resolve a backend name to an instance. ``duckduckgo_lite`` is the
    only one shipped today; tavily / brave / serpapi land in a follow-up
    slice.
    """
    if name in ("duckduckgo_lite", "ddg", "duckduckgo"):
        return DuckDuckGoLiteBackend(timeout_seconds=timeout_seconds)
    raise ValueError(
        f"Unknown search backend {name!r}. "
        f"Supported: duckduckgo_lite. "
        f"Tavily / Brave / SerpAPI backends are planned for a follow-up slice."
    )


# ---------------------------------------------------------------------------
# 4. WebFetchSkill
# ---------------------------------------------------------------------------

async def _web_fetch_impl(
    cfg: WebToolsConfig,
    *,
    url: str,
    max_bytes: Optional[int] = None,
) -> dict[str, Any]:
    """Implementation body for the fetch skill. Returns a structured dict
    the LLM can read; never raises (errors come back as ``{"error": ...}``
    so the conversation continues)."""
    if not cfg.enabled:
        return {"error": "web_tools disabled in config (web_tools.enabled=false)"}
    try:
        validate_outbound_url(url, allow_private_ips=cfg.allow_private_ips)
    except ValueError as exc:
        return {"error": f"url rejected: {exc}"}
    # LLM-supplied max_bytes is clamped to the configured ceiling so the
    # LLM cannot ask for more than the operator allows. When the LLM
    # doesn't supply one, the configured cap is used verbatim. A
    # one-byte floor stops degenerate kwargs (max_bytes=0) from
    # returning nothing.
    if max_bytes is None:
        cap = cfg.max_bytes
    else:
        cap = max(1, min(int(max_bytes), cfg.max_bytes))
    headers = {
        "User-Agent": "myharness-research/1.0 (+https://github.com/akmontheweb/myharness)",
        "Accept": "text/html,application/json,text/plain,text/markdown;q=0.9,*/*;q=0.1",
    }
    try:
        # SSRF hardening (audit §3.1): we manually follow redirects so
        # each hop's Location header is re-validated through
        # validate_outbound_url. With follow_redirects=True, httpx would
        # silently follow a 302 to http://169.254.169.254/...; the
        # original-URL validator never sees it.
        current_url = url
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(cfg.timeout_seconds, connect=10.0),
            follow_redirects=False,
            headers=headers,
        ) as client:
            response = None
            for hop in range(5):  # bounded redirect chain
                response = await client.get(current_url)
                if response.status_code not in (301, 302, 303, 307, 308):
                    break
                next_loc = response.headers.get("location") or ""
                if not next_loc:
                    break
                # Resolve relative Location values against the current URL.
                try:
                    from urllib.parse import urljoin
                    candidate = urljoin(current_url, next_loc)
                except Exception:  # noqa: BLE001
                    return {"url": url, "error": f"unparseable Location header at hop {hop}: {next_loc!r}"}
                try:
                    validate_outbound_url(candidate, allow_private_ips=cfg.allow_private_ips)
                except ValueError as exc:
                    return {
                        "url": url,
                        "status_code": response.status_code,
                        "error": f"redirect target rejected at hop {hop}: {exc}",
                    }
                current_url = candidate
            else:
                return {"url": url, "error": "redirect chain exceeded 5 hops"}

        assert response is not None  # one of the loop iterations ran
        # Validate content-type
        ct = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
        if ct and not any(ct == allowed or ct.startswith(allowed) for allowed in _ALLOWED_CONTENT_TYPES):
            return {
                "url": url,
                "status_code": response.status_code,
                "content_type": ct,
                "error": (
                    f"content-type {ct!r} not in allowlist. "
                    f"Allowed prefixes: {_ALLOWED_CONTENT_TYPES}"
                ),
            }
        body_bytes = response.content[:cap]
        truncated = len(response.content) > cap
        # Decode using response's apparent encoding; fall back to utf-8
        text = body_bytes.decode(
            response.encoding or "utf-8", errors="replace"
        )
        if ct.startswith("text/html"):
            text = html_to_text(text)
        return {
            "url": url,
            "status_code": response.status_code,
            "content_type": ct,
            "content": text,
            "truncated": truncated,
            "bytes_returned": len(body_bytes),
        }
    except httpx.HTTPStatusError as exc:
        return {
            "url": url,
            "status_code": exc.response.status_code if exc.response else 0,
            "error": f"http error: {exc}",
        }
    except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
        return {"url": url, "error": f"network error: {exc}"}
    except Exception as exc:  # noqa: BLE001 — never propagate to caller
        logger.exception("[web_fetch] unexpected error for %s", url)
        return {"url": url, "error": f"unexpected error: {exc}"}


class WebFetchSkill(ToolSkill):
    """HTTP GET an LLM-supplied URL and return readable text + metadata.

    Honors ``web_tools.enabled``, ``web_tools.max_bytes``, ``web_tools.
    allow_private_ips``. Strips HTML, allowlists content-types, caps
    response size. Errors surface as ``{"error": ...}`` so the LLM can
    react rather than the graph crashing.
    """

    SKILL_NAME = "web_fetch"

    def __init__(self, cfg: WebToolsConfig):
        schema = SkillSchema(
            name=self.SKILL_NAME,
            description=(
                "Fetch the content of an HTTP/HTTPS URL and return readable text. "
                "Use to read documentation, RFCs, blog posts, or JSON responses "
                "before deciding on an implementation. Returns up to "
                f"{cfg.max_bytes} bytes; longer responses are truncated."
            ),
            skill_type=SkillType.TOOL,
            parameters=[
                SkillParameter("url", "string", "Absolute http/https URL.", required=True),
                SkillParameter(
                    "max_bytes",
                    "integer",
                    f"Optional cap on returned bytes (default {cfg.max_bytes}; "
                    f"hard upper bound from config).",
                    required=False,
                ),
            ],
            returns_description=(
                "Object with `url`, `status_code`, `content_type`, `content` "
                "(decoded text; HTML stripped to readable text), `truncated`, "
                "`bytes_returned`. On failure, `error` is set instead of `content`."
            ),
            tags=["web", "research", "fetch"],
        )
        self._cfg = cfg
        super().__init__(schema, fn=self._call)

    async def _call(self, **kwargs: Any) -> dict[str, Any]:
        return await _web_fetch_impl(self._cfg, **kwargs)


# ---------------------------------------------------------------------------
# 5. WebSearchSkill
# ---------------------------------------------------------------------------

async def _web_search_impl(
    cfg: WebToolsConfig,
    *,
    query: str,
    max_results: Optional[int] = None,
    _backend_factory: Optional[Any] = None,
) -> dict[str, Any]:
    if not cfg.enabled:
        return {"error": "web_tools disabled in config (web_tools.enabled=false)"}
    if not isinstance(query, str) or not query.strip():
        return {"error": "query must be a non-empty string"}
    cap = int(max_results) if max_results is not None else cfg.max_results
    cap = max(1, min(cap, cfg.max_results))
    try:
        factory = _backend_factory or make_search_backend
        backend = factory(cfg.search_backend, timeout_seconds=cfg.timeout_seconds)
        results = await backend.search(query, max_results=cap)
    except ValueError as exc:
        # Backend name unknown.
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("[web_search] unexpected error for query=%r", query)
        return {"error": f"unexpected error: {exc}"}
    return {
        "query": query,
        "backend": cfg.search_backend,
        "results": [r.to_dict() for r in results],
    }


class WebSearchSkill(ToolSkill):
    """Search the web via the configured backend (default
    ``duckduckgo_lite``). Returns a list of ``{title, url, snippet}``
    results so the LLM can pick a URL and ``web_fetch`` it.
    """

    SKILL_NAME = "web_search"

    def __init__(self, cfg: WebToolsConfig):
        schema = SkillSchema(
            name=self.SKILL_NAME,
            description=(
                "Search the web and return a list of result titles, URLs, and "
                "snippets. Use to discover authoritative sources before "
                "fetching one with web_fetch. Default backend: "
                f"{cfg.search_backend!r}."
            ),
            skill_type=SkillType.TOOL,
            parameters=[
                SkillParameter("query", "string", "Search query string.", required=True),
                SkillParameter(
                    "max_results",
                    "integer",
                    f"Optional cap (default {cfg.max_results}; hard upper "
                    f"bound from config).",
                    required=False,
                ),
            ],
            returns_description=(
                "Object with `query`, `backend`, and `results` "
                "(list of `{title, url, snippet}`). On failure, `error` is "
                "set instead of `results`."
            ),
            tags=["web", "research", "search"],
        )
        self._cfg = cfg
        super().__init__(schema, fn=self._call)

    async def _call(self, **kwargs: Any) -> dict[str, Any]:
        return await _web_search_impl(self._cfg, **kwargs)


# ---------------------------------------------------------------------------
# 6. Text-DSL parser  ----  matches the existing patcher pattern
# ---------------------------------------------------------------------------

_TOOL_BLOCK_RE = re.compile(
    r"<<<\s*(WEB_FETCH|WEB_SEARCH)\s+(.*?)>>>",
    re.DOTALL | re.IGNORECASE,
)
_KWARG_RE = re.compile(
    r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"'
)
_INT_KWARG_RE = re.compile(r"(\w+)\s*=\s*(\d+)")


@dataclass
class ParsedToolBlock:
    skill_name: str  # "web_fetch" | "web_search"
    kwargs: dict[str, Any] = field(default_factory=dict)
    raw: str = ""  # the full <<<...>>> source — needed for strip-from-content


def parse_tool_blocks(content: str) -> list[ParsedToolBlock]:
    """Extract every ``<<<WEB_FETCH ...>>>`` / ``<<<WEB_SEARCH ...>>>``
    block from an LLM response. Returns them in document order. Each
    block's kwargs are parsed from ``key="value"`` and ``key=integer``
    forms (matching the lightweight Claude-Code-style tool DSL).
    """
    blocks: list[ParsedToolBlock] = []
    if not isinstance(content, str) or "<<<" not in content:
        return blocks
    for match in _TOOL_BLOCK_RE.finditer(content):
        op = match.group(1).upper()
        body = match.group(2)
        skill_name = "web_fetch" if op == "WEB_FETCH" else "web_search"
        kwargs: dict[str, Any] = {}
        for kw in _KWARG_RE.finditer(body):
            kwargs[kw.group(1)] = kw.group(2).encode("utf-8").decode("unicode_escape")
        for kw in _INT_KWARG_RE.finditer(body):
            # Only attach if not already set by the string form.
            if kw.group(1) not in kwargs:
                kwargs[kw.group(1)] = int(kw.group(2))
        blocks.append(ParsedToolBlock(skill_name=skill_name, kwargs=kwargs, raw=match.group(0)))
    return blocks


def strip_tool_blocks(content: str) -> str:
    """Remove every ``<<<WEB_FETCH ...>>>`` / ``<<<WEB_SEARCH ...>>>``
    block from ``content``. Used by the graph interceptor to keep tool
    blocks out of the patcher input (so ``process_llm_patch_output``
    never sees them).
    """
    if not isinstance(content, str) or "<<<" not in content:
        return content
    return _TOOL_BLOCK_RE.sub("", content)


# ---------------------------------------------------------------------------
# 7. Registration helper used by SkillRegistry bootstrap
# ---------------------------------------------------------------------------

def register_web_tool_skills(cfg: WebToolsConfig) -> int:
    """Register WebFetchSkill + WebSearchSkill in the global registry.

    Idempotent: re-registering a skill silently overwrites the prior
    entry (matches the existing ``SkillRegistry.register`` contract).
    Returns the number of skills registered.
    """
    from harness.skills import register
    register(WebFetchSkill(cfg))
    register(WebSearchSkill(cfg))
    return 2
