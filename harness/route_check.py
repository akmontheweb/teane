"""
Client↔server HTTP route contract check.

Catches the failure mode where generated frontend code calls an API path
the backend never implements — e.g. a React client fetching
``/api/csrf-token`` when no such route exists, so every request 404s at
runtime and the app ships silently broken (the class of bug that made a
generated app read-only: all writes hit a missing CSRF endpoint).

``link_check`` proves every *import* resolves; this proves every *HTTP
call* resolves. The two are complementary seam checks. Where link_check is
pure-static, this needs the backend's authoritative route list — which for
a framework like FastAPI only exists at runtime via ``app.openapi()`` (it
composes router prefixes and mounts at import time). So the split is:

    - CLIENT extraction (this module): pure-static regex over JS/TS, cheap,
      in-process. Reuses link_check's file walker + comment stripper.
    - SERVER schema acquisition: an OpenAPI document. This module only
      *parses* a document it is handed (``extract_server_routes``) or loads
      a committed ``openapi.json`` (``_load_committed_openapi``). Obtaining
      the schema from a live app (sandbox ``app.openapi()``) lives in
      graph.py next to the prod-import smoke check, so this module stays
      import-light and unit-testable without a sandbox.

Design invariants (mirrors link_check):
    - Framework-agnostic on the server side: any OpenAPI producer works.
    - Never raises out of any public function — advisory, best-effort I/O.
      A malformed doc, an unreadable file, or an odd call site simply
      yields fewer diagnostics, never a crash and never a wrong one.
    - False-negative-biased: only HIGH-CONFIDENCE static client calls are
      emitted. A dynamic/computed URL is skipped, never guessed at — a
      false ``ROUTE_UNRESOLVED`` would poison the repair loop.

Diagnostics carry ``error_code="ROUTE_UNRESOLVED"`` and the same dict shape
``compiler_errors`` already uses, so they flow into the existing repair
loop exactly like ``LINK_BROKEN`` / ``PROD_IMPORT_SMOKE``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

# Reuse link_check's walker primitives so the two seam checks never drift
# on which dirs/extensions count as source or how comments are masked.
from harness.link_check import (
    _JS_SOURCE_EXTS,
    _NEVER_SOURCE_DIRS,
    _strip_js_comments,
)

logger = logging.getLogger(__name__)


# The HTTP methods a client can plausibly emit and that we map to OpenAPI
# operation keys. HEAD/OPTIONS/TRACE are intentionally excluded on the
# client side (rarely hand-written; matching them adds noise).
_CLIENT_METHODS: frozenset[str] = frozenset({"GET", "POST", "PUT", "DELETE", "PATCH"})
# OpenAPI operation keys we treat as routes when reading a schema's paths.
_OPENAPI_METHODS: frozenset[str] = frozenset({
    "get", "post", "put", "delete", "patch", "head", "options", "trace",
})


@dataclass(frozen=True)
class ClientCall:
    """One statically-extracted HTTP call from client source.

    Attributes:
        method: Upper-case HTTP method ("GET", "POST", ...).
        norm_path: Path normalized for matching (dynamic segments → ``{}``,
            query/hash stripped, trailing slash removed).
        raw_path: The literal path as written, for the diagnostic message.
        source_file: Workspace-relative path of the calling file.
        line: 1-based line of the call site.
    """
    method: str
    norm_path: str
    raw_path: str
    source_file: str
    line: int


@dataclass(frozen=True)
class BrokenRoute:
    """A client call whose (method, path) matches no server route."""
    source_file: str
    line: int
    method: str
    client_path: str
    available_routes: tuple[str, ...]


# ---------------------------------------------------------------------------
# Path normalization (applied to BOTH client and server paths)
# ---------------------------------------------------------------------------

_TEMPLATE_INTERP_RE = re.compile(r"\$\{[^}]*\}")   # JS `${id}`
_BRACE_PARAM_RE = re.compile(r"\{[^}]*\}")          # OpenAPI `{contact_id}` / already `{}`
_COLON_PARAM_RE = re.compile(r":[A-Za-z_]\w*")      # Express/rails `:id`


def _normalize_path(path: str) -> str:
    """Collapse a URL path to a comparable template.

    ``/api/contacts/${id}?x=1`` and ``/api/contacts/{contact_id}`` both
    become ``/api/contacts/{}``. Query string and fragment are dropped
    (they never affect route matching). Trailing slash removed except for
    the bare root ``/``.
    """
    if not path:
        return ""
    # Drop query + fragment.
    path = path.split("?", 1)[0].split("#", 1)[0]
    # Dynamic segments → a single wildcard token. Order matters: expand
    # `${...}` first (its braces would otherwise be eaten by the brace rule).
    path = _TEMPLATE_INTERP_RE.sub("{}", path)
    path = _BRACE_PARAM_RE.sub("{}", path)
    path = _COLON_PARAM_RE.sub("{}", path)
    # Normalize trailing slash (but keep root).
    if len(path) > 1:
        path = path.rstrip("/")
    return path


def _has_static_prefix(norm_path: str) -> bool:
    """True when the path is rooted and its first segment is static.

    This is the confidence firewall: a call like ``fetch(`${base}/x`)``
    normalizes to ``/{}/x`` (or ``{}/x``) — the structure itself is
    dynamic, so we can't know the real path and must NOT flag it. A call
    like ``/api/contacts/${id}`` → ``/api/contacts/{}`` has a concrete
    first segment and is safe to check.
    """
    if not norm_path.startswith("/"):
        return False
    segments = [s for s in norm_path.strip("/").split("/") if s != ""]
    if not segments:
        return False
    return segments[0] != "{}"


# ---------------------------------------------------------------------------
# Client-call extraction (pure-static)
# ---------------------------------------------------------------------------

# A quoted string literal: single/double/backtick. Uses a NAMED
# backreference so the pattern composes correctly regardless of how many
# capture groups precede it (a positional `\1` would bind to the wrong
# group once prefixed by e.g. the axios method group). Backtick-captured
# content keeps its `${...}` so normalization can collapse it.
_STR = r"""(?P<q>['"`])(?P<path>[^'"`]*?)(?P=q)"""

# fetch("/path", ...)  — method resolved from the bounded options object.
_FETCH_RE = re.compile(r"""\bfetch\s*\(\s*""" + _STR, re.DOTALL)
# axios.post("/path")  — method from the shorthand.
_AXIOS_SHORTHAND_RE = re.compile(
    r"""\baxios\s*\.\s*(?P<m>get|post|put|delete|patch)\s*\(\s*""" + _STR,
    re.IGNORECASE | re.DOTALL,
)
# axios( { ... } )  — config-object form; url + method read from the object.
_AXIOS_OBJ_OPEN_RE = re.compile(r"""\baxios\s*\(\s*\{""")
_URL_KEY_RE = re.compile(r"""\burl\s*:\s*""" + _STR, re.DOTALL)
# `method: "post"` — resolves fetch / axios-config verbs.
_METHOD_KEY_RE = re.compile(
    r"""\bmethod\s*:\s*['"`]?(?P<m>get|post|put|delete|patch)\b""",
    re.IGNORECASE,
)

# Hard cap on how far a delimiter matcher will scan for its closing pair —
# a runaway (unbalanced source) can never hang the walk.
_MAX_CALL_SPAN = 800


def _line_of(body: str, offset: int) -> int:
    """1-based line number of ``offset`` in the ORIGINAL body.

    Comment stripping preserves character offsets (equal-length blanking)
    but collapses newlines inside block comments, so line numbers must be
    counted against the original text, not the stripped copy.
    """
    return body.count("\n", 0, offset) + 1


def _delim_end(text: str, open_idx: int) -> int:
    """Index just past the delimiter matching ``text[open_idx]`` (``(`` or
    ``{``), or a capped fallback. Naive (no string-literal tracking), but
    bounded by ``_MAX_CALL_SPAN`` so it can never run away. Good enough to
    scope a single call's argument list / options object.
    """
    open_ch = text[open_idx]
    close_ch = ")" if open_ch == "(" else "}"
    depth = 0
    end_cap = min(len(text), open_idx + _MAX_CALL_SPAN)
    for i in range(open_idx, end_cap):
        c = text[i]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
    return end_cap


def _method_within(text: str, start: int, end: int) -> Optional[str]:
    """Return the ``method:`` verb between ``start`` and ``end``, or None."""
    m = _METHOD_KEY_RE.search(text, start, max(start, end))
    return m.group("m").upper() if m else None


def _scan_js_source(body: str, src_rel: str) -> list[ClientCall]:
    """Extract client HTTP calls from one JS/TS file's text."""
    stripped = _strip_js_comments(body)
    calls: list[ClientCall] = []
    seen: set[tuple[str, str, int]] = set()

    def _emit(method: str, raw_path: str, match_start: int) -> None:
        norm = _normalize_path(raw_path)
        if not _has_static_prefix(norm):
            return  # dynamic structure — skip (confidence firewall)
        line = _line_of(body, match_start)
        key = (method, norm, line)
        if key in seen:
            return
        seen.add(key)
        calls.append(ClientCall(
            method=method, norm_path=norm, raw_path=raw_path,
            source_file=src_rel, line=line,
        ))

    for m in _FETCH_RE.finditer(stripped):
        # Scope the method lookup to THIS fetch call's paren span so it
        # can't bleed into a following call's options object.
        paren = stripped.find("(", m.start())
        call_end = _delim_end(stripped, paren) if paren != -1 else m.end()
        method = _method_within(stripped, m.end(), call_end) or "GET"
        _emit(method, m.group("path"), m.start())

    for m in _AXIOS_SHORTHAND_RE.finditer(stripped):
        _emit(m.group("m").upper(), m.group("path"), m.start())

    for m in _AXIOS_OBJ_OPEN_RE.finditer(stripped):
        brace = stripped.find("{", m.start())
        if brace == -1:
            continue
        obj = stripped[brace:_delim_end(stripped, brace)]
        url_m = _URL_KEY_RE.search(obj)
        if not url_m:
            continue
        method_m = _METHOD_KEY_RE.search(obj)
        method = method_m.group("m").upper() if method_m else "GET"
        _emit(method, url_m.group("path"), m.start())

    return calls


def extract_client_calls(workspace_path: str) -> list[ClientCall]:
    """Walk ``workspace_path`` and return every high-confidence client HTTP
    call. Never raises — unreadable files are skipped at debug level.

    Prunes the same never-source directories as link_check (node_modules,
    dist, .venv, ...). Empty list when there is no client code.
    """
    if not workspace_path or not os.path.isdir(workspace_path):
        return []
    calls: list[ClientCall] = []
    for sub_root, sub_dirs, sub_files in os.walk(workspace_path):
        sub_dirs[:] = [
            d for d in sub_dirs
            if not d.startswith(".") and d not in _NEVER_SOURCE_DIRS
        ]
        for fname in sub_files:
            if os.path.splitext(fname)[1].lower() not in _JS_SOURCE_EXTS:
                continue
            src_abs = os.path.join(sub_root, fname)
            src_rel = os.path.relpath(src_abs, workspace_path)
            try:
                with open(src_abs, "r", encoding="utf-8", errors="replace") as f:
                    body = f.read()
            except OSError as exc:
                logger.debug("[route_check] Could not read %s: %s", src_abs, exc)
                continue
            calls.extend(_scan_js_source(body, src_rel))
    return calls


# ---------------------------------------------------------------------------
# Server schema (framework-agnostic — any OpenAPI producer)
# ---------------------------------------------------------------------------

# Where a project might commit/generate an OpenAPI document. Checking these
# first is free (no sandbox) and lets a team pin an authoritative contract.
_OPENAPI_FILENAMES: tuple[str, ...] = (
    "openapi.json", "openapi.yaml", "openapi.yml",
)
_OPENAPI_SEARCH_SUBDIRS: tuple[str, ...] = (
    "", "server", "client", "frontend", "docs", "api",
)


def _load_committed_openapi(workspace_path: str) -> Optional[dict]:
    """Best-effort load of a committed OpenAPI document, or None.

    JSON is always supported; YAML only if PyYAML happens to be importable
    (never a hard dependency). Any read/parse error → None (degrade).
    """
    if not workspace_path or not os.path.isdir(workspace_path):
        return None
    for sub in _OPENAPI_SEARCH_SUBDIRS:
        base = os.path.join(workspace_path, sub) if sub else workspace_path
        for name in _OPENAPI_FILENAMES:
            cand = os.path.join(base, name)
            if not os.path.isfile(cand):
                continue
            try:
                with open(cand, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
                if name.endswith(".json"):
                    doc = json.loads(text)
                else:
                    try:
                        import yaml  # type: ignore
                    except ImportError:
                        continue
                    doc = yaml.safe_load(text)
                if isinstance(doc, dict) and isinstance(doc.get("paths"), dict):
                    logger.info("[route_check] Using committed OpenAPI schema at %s.", cand)
                    return doc
            except Exception as exc:  # noqa: BLE001 — advisory
                logger.debug("[route_check] Could not parse %s: %s", cand, exc)
    return None


def extract_server_routes(openapi_doc: object) -> set[tuple[str, str]]:
    """Return ``{(METHOD, normalized_path)}`` from an OpenAPI document.

    Tolerates a malformed/empty doc by returning an empty set (the caller
    treats "no routes" as an unreliable schema and skips, so an empty set
    never mass-flags client calls in production).
    """
    routes: set[tuple[str, str]] = set()
    if not isinstance(openapi_doc, dict):
        return routes
    paths = openapi_doc.get("paths")
    if not isinstance(paths, dict):
        return routes
    for raw_path, ops in paths.items():
        if not isinstance(raw_path, str) or not isinstance(ops, dict):
            continue
        norm = _normalize_path(raw_path)
        for key in ops:
            if isinstance(key, str) and key.lower() in _OPENAPI_METHODS:
                routes.add((key.upper(), norm))
    return routes


# ---------------------------------------------------------------------------
# Matching + diagnostics
# ---------------------------------------------------------------------------

def match_routes(
    client_calls: list[ClientCall],
    server_routes: set[tuple[str, str]],
) -> list[BrokenRoute]:
    """Return one BrokenRoute per client call with no matching server route.

    ``available_routes`` (sorted "METHOD /path" strings) is attached as
    repair context so the LLM can see what DOES exist alongside what
    doesn't. Deduped per (file, method, raw_path) call site.
    """
    available = tuple(sorted(f"{m} {p}" for m, p in server_routes))
    broken: list[BrokenRoute] = []
    seen: set[tuple[str, str, str]] = set()
    for call in client_calls:
        if (call.method, call.norm_path) in server_routes:
            continue
        key = (call.source_file, call.method, call.raw_path)
        if key in seen:
            continue
        seen.add(key)
        broken.append(BrokenRoute(
            source_file=call.source_file,
            line=call.line,
            method=call.method,
            client_path=call.raw_path,
            available_routes=available,
        ))
    return broken


def broken_routes_to_diagnostics(
    broken: list[BrokenRoute],
) -> list[dict[str, object]]:
    """Convert BrokenRoute records to the diagnostic-dict shape the repair
    loop consumes (matches ``compiler_errors``). Mirrors
    ``link_check.broken_links_to_diagnostics``.

    Each diagnostic carries ``error_code="ROUTE_UNRESOLVED"`` so downstream
    routing recognizes it without re-parsing the message.
    """
    out: list[dict[str, object]] = []
    for br in broken:
        available = "\n".join(f"    - {r}" for r in br.available_routes) or "    (none)"
        out.append({
            "file": br.source_file,
            "line": br.line,
            "column": 0,
            "severity": "error",
            "error_code": "ROUTE_UNRESOLVED",
            "message": (
                f"Client calls {br.method} {br.client_path!r} in "
                f"{br.source_file}, but the backend exposes no matching "
                f"route (its OpenAPI schema has no {br.method} operation "
                f"for that path). Every request to this endpoint will 404 "
                f"at runtime. Either add the missing {br.method} route to "
                f"the backend, or fix the client path to match an existing "
                f"route."
            ),
            "semantic_context": (
                f"Available server routes:\n{available}"
            ),
            "missing_symbol": f"{br.method} {br.client_path}",
            "language": "js",
        })
    return out
