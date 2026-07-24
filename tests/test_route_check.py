"""Tests for the client↔server HTTP route contract check."""
from __future__ import annotations

import json
import os

from harness.route_check import (
    BrokenRoute,
    ClientCall,
    _load_committed_openapi,
    _normalize_path,
    broken_routes_to_diagnostics,
    extract_client_calls,
    extract_server_routes,
    match_routes,
)


def _w(tmp_path, rel: str, body: str) -> None:
    abs_path = os.path.join(str(tmp_path), rel)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(body)


def _check(tmp_path, openapi: dict) -> list[BrokenRoute]:
    """Run the full pipeline against an in-memory OpenAPI doc."""
    calls = extract_client_calls(str(tmp_path))
    routes = extract_server_routes(openapi)
    return match_routes(calls, routes)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def test_normalize_collapses_all_param_styles():
    assert _normalize_path("/api/contacts/${id}") == "/api/contacts/{}"
    assert _normalize_path("/api/contacts/{contact_id}") == "/api/contacts/{}"
    assert _normalize_path("/users/:id/posts") == "/users/{}/posts"


def test_normalize_strips_query_and_trailing_slash():
    assert _normalize_path("/api/x?page=1") == "/api/x"
    assert _normalize_path("/api/x/") == "/api/x"
    assert _normalize_path("/api/x#frag") == "/api/x"
    assert _normalize_path("/") == "/"  # root keeps its slash


# ---------------------------------------------------------------------------
# Matching — the core behaviour
# ---------------------------------------------------------------------------

def test_matched_route_passes(tmp_path):
    _w(tmp_path, "client/src/api.ts", 'fetch("/api/contacts");\n')
    broken = _check(tmp_path, {"paths": {"/api/contacts": {"get": {}}}})
    assert broken == []


def test_missing_route_flagged_csrf_repro(tmp_path):
    # The exact lumina blocker: client fetches /api/csrf-token before every
    # write; backend never implements it -> all writes 404.
    _w(tmp_path, "client/src/api/contacts.ts",
       'const t = await fetch("/api/csrf-token");\n'
       'fetch("/api/contacts", { method: "POST" });\n')
    openapi = {"paths": {
        "/api/contacts": {"get": {}, "post": {}},
        "/api/contacts/{contact_id}": {"put": {}, "delete": {}},
    }}
    broken = _check(tmp_path, openapi)
    assert len(broken) == 1
    assert broken[0].method == "GET"
    assert broken[0].client_path == "/api/csrf-token"
    # The POST /api/contacts call matched, so it is NOT flagged.
    diags = broken_routes_to_diagnostics(broken)
    assert diags[0]["error_code"] == "ROUTE_UNRESOLVED"


def test_method_mismatch_flagged(tmp_path):
    _w(tmp_path, "client/src/api.ts", 'axios.post("/api/contacts");\n')
    # Only GET exists — a POST must be flagged.
    broken = _check(tmp_path, {"paths": {"/api/contacts": {"get": {}}}})
    assert len(broken) == 1
    assert broken[0].method == "POST"


def test_path_param_templating_matches(tmp_path):
    _w(tmp_path, "client/src/api.ts",
       "const id = 1;\n"
       "fetch(`/api/contacts/${id}`, { method: 'PUT' });\n")
    broken = _check(tmp_path, {"paths": {"/api/contacts/{contact_id}": {"put": {}}}})
    assert broken == []


# ---------------------------------------------------------------------------
# Client extraction variants
# ---------------------------------------------------------------------------

def test_axios_shorthand_methods(tmp_path):
    _w(tmp_path, "client/src/api.ts",
       'axios.get("/a");\naxios.post("/b");\naxios.put("/c");\n'
       'axios.delete("/d");\naxios.patch("/e");\n')
    calls = {(c.method, c.norm_path) for c in extract_client_calls(str(tmp_path))}
    assert calls == {("GET", "/a"), ("POST", "/b"), ("PUT", "/c"),
                     ("DELETE", "/d"), ("PATCH", "/e")}


def test_axios_config_object(tmp_path):
    _w(tmp_path, "client/src/api.ts",
       'axios({ url: "/api/x", method: "put" });\n')
    calls = extract_client_calls(str(tmp_path))
    assert (calls[0].method, calls[0].norm_path) == ("PUT", "/api/x")


def test_fetch_default_method_is_get(tmp_path):
    _w(tmp_path, "client/src/api.ts", 'fetch("/api/x");\n')
    calls = extract_client_calls(str(tmp_path))
    assert calls[0].method == "GET"


def test_fetch_explicit_method(tmp_path):
    _w(tmp_path, "client/src/api.ts",
       'fetch("/api/x", { headers: {}, method: "DELETE" });\n')
    calls = extract_client_calls(str(tmp_path))
    assert calls[0].method == "DELETE"


# ---------------------------------------------------------------------------
# Confidence firewall — dynamic/unrooted calls are skipped, never flagged
# ---------------------------------------------------------------------------

def test_dynamic_base_url_skipped(tmp_path):
    _w(tmp_path, "client/src/api.ts",
       "const API = 'x';\nfetch(`${API}/contacts`);\n")
    # First path segment is dynamic -> not a ClientCall, so nothing to flag
    # even against an empty schema.
    assert extract_client_calls(str(tmp_path)) == []
    assert _check(tmp_path, {"paths": {}}) == []


def test_non_rooted_path_skipped(tmp_path):
    _w(tmp_path, "client/src/api.ts",
       'fetch("relative/x");\nfetch(someVar);\n')
    assert extract_client_calls(str(tmp_path)) == []


def test_query_string_stripped_and_matches(tmp_path):
    _w(tmp_path, "client/src/api.ts", 'fetch("/api/x?page=1&q=2");\n')
    assert _check(tmp_path, {"paths": {"/api/x": {"get": {}}}}) == []


def test_commented_out_call_ignored(tmp_path):
    _w(tmp_path, "client/src/api.ts",
       '// fetch("/api/ghost");\n'
       '/* fetch("/api/also-ghost"); */\n'
       'fetch("/api/real");\n')
    calls = extract_client_calls(str(tmp_path))
    assert {c.norm_path for c in calls} == {"/api/real"}


def test_trailing_slash_normalized(tmp_path):
    _w(tmp_path, "client/src/api.ts", 'fetch("/api/x/");\n')
    assert _check(tmp_path, {"paths": {"/api/x": {"get": {}}}}) == []


def test_node_modules_skipped(tmp_path):
    _w(tmp_path, "client/node_modules/pkg/index.js", 'fetch("/api/ghost");\n')
    _w(tmp_path, "client/src/api.ts", 'fetch("/api/real");\n')
    calls = extract_client_calls(str(tmp_path))
    assert {c.norm_path for c in calls} == {"/api/real"}


# ---------------------------------------------------------------------------
# Server extraction / schema loading / degradation
# ---------------------------------------------------------------------------

def test_extract_server_routes_all_methods():
    doc = {"paths": {
        "/a": {"get": {}, "post": {}, "parameters": []},
        "/b/{id}": {"delete": {}},
    }}
    assert extract_server_routes(doc) == {
        ("GET", "/a"), ("POST", "/a"), ("DELETE", "/b/{}"),
    }


def test_malformed_openapi_degrades_to_empty():
    assert extract_server_routes({}) == set()
    assert extract_server_routes({"paths": None}) == set()
    assert extract_server_routes("garbage") == set()
    assert extract_server_routes(None) == set()


def test_committed_openapi_loaded(tmp_path):
    doc = {"openapi": "3.1.0", "paths": {"/api/x": {"get": {}}}}
    _w(tmp_path, "openapi.json", json.dumps(doc))
    loaded = _load_committed_openapi(str(tmp_path))
    assert loaded is not None
    assert extract_server_routes(loaded) == {("GET", "/api/x")}


def test_committed_openapi_absent_returns_none(tmp_path):
    _w(tmp_path, "client/src/api.ts", 'fetch("/api/x");\n')
    assert _load_committed_openapi(str(tmp_path)) is None


def test_no_client_calls_returns_empty(tmp_path):
    _w(tmp_path, "server/app/main.py", "app = 1\n")
    assert extract_client_calls(str(tmp_path)) == []


def test_empty_openapi_flags_all_calls(tmp_path):
    _w(tmp_path, "client/src/api.ts", 'fetch("/api/x");\n')
    broken = _check(tmp_path, {"paths": {}})
    assert len(broken) == 1


# ---------------------------------------------------------------------------
# Diagnostic shape
# ---------------------------------------------------------------------------

def test_diagnostics_shape():
    br = BrokenRoute(
        source_file="client/src/api.ts", line=7, method="GET",
        client_path="/api/csrf-token",
        available_routes=("GET /api/contacts", "POST /api/contacts"),
    )
    diags = broken_routes_to_diagnostics([br])
    assert len(diags) == 1
    d = diags[0]
    assert d["error_code"] == "ROUTE_UNRESOLVED"
    assert d["file"] == "client/src/api.ts"
    assert d["line"] == 7
    assert d["severity"] == "error"
    assert "GET" in str(d["message"]) and "/api/csrf-token" in str(d["message"])
    assert "GET /api/contacts" in str(d["semantic_context"])


def test_match_dedupes_per_call_site():
    # Same missing call captured twice -> one BrokenRoute.
    calls = [
        ClientCall("GET", "/api/x", "/api/x", "a.ts", 1),
        ClientCall("GET", "/api/x", "/api/x", "a.ts", 1),
    ]
    assert len(match_routes(calls, set())) == 1
