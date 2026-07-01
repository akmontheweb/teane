"""Phase 0 regression: vendored htmx + Alpine + tokens.css are packaged
with the wheel and served through the existing `_serve_static` +
`_layout` plumbing.

Two levels of coverage:
1. `_serve_static` returns each vendor asset with the right content
   type (proves the packaged files exist and match the extension
   whitelist).
2. `_layout` embeds the three vendor <script defer> tags and the
   tokens.css <link> in the right order — Carbon first, tokens.css
   after so equal-specificity selectors in tokens.css win at cascade
   time. Ordering assertions are the load-order contract the rest of
   the phases build on.
"""

from __future__ import annotations

from harness.dashboard import DashboardConfig, _layout, _serve_static


def _cfg(tmp_path):
    return DashboardConfig.from_config(
        {
            "dashboard": {
                "log_dir": str(tmp_path / "logs"),
                "metrics_dir": str(tmp_path / "metrics"),
                "memory_dir": str(tmp_path / "memory"),
                "repo_index_dir": str(tmp_path / "idx"),
                "schedule_db": str(tmp_path / "schedule.db"),
                "static_dir": str(tmp_path / "static"),
                "enabled": True,
            }
        }
    )


def test_serve_vendored_htmx(tmp_path):
    status, ctype, data = _serve_static(_cfg(tmp_path), "vendor/htmx-1.9.12.min.js")
    assert status == 200
    assert "javascript" in ctype
    assert len(data) > 10_000  # minified htmx is ~48 KB
    assert b"htmx" in data.lower()


def test_serve_vendored_htmx_sse_extension(tmp_path):
    status, ctype, data = _serve_static(
        _cfg(tmp_path), "vendor/htmx-sse-1.9.12.min.js"
    )
    assert status == 200
    assert "javascript" in ctype
    assert len(data) > 1_000
    # The SSE extension registers itself via htmx.defineExtension.
    assert b"sse" in data.lower()


def test_serve_vendored_alpine(tmp_path):
    status, ctype, data = _serve_static(
        _cfg(tmp_path), "vendor/alpine-3.14.1.min.js"
    )
    assert status == 200
    assert "javascript" in ctype
    assert len(data) > 10_000  # minified Alpine is ~44 KB


def test_serve_tokens_css(tmp_path):
    status, ctype, data = _serve_static(_cfg(tmp_path), "css/tokens.css")
    assert status == 200
    assert ctype.startswith("text/css")
    # Sanity: our design-token custom properties are in the file.
    assert b"--t-accent" in data
    assert b"--t-sp-4" in data


def test_layout_emits_vendor_scripts_and_tokens_css(tmp_path):
    """The _layout template must:
    - Emit tokens.css AFTER Carbon (so equal-specificity tokens win).
    - Emit htmx BEFORE the htmx-sse extension (extensions register
      against the htmx global).
    - Emit both htmx assets and Alpine as `defer` so document order
      execution is preserved without blocking parse.
    """
    cfg = _cfg(tmp_path)
    html = _layout("Test", "<p>body</p>", cfg, active="status")

    # Vendor script tags present, all cache-busted.
    assert 'src="/static/vendor/htmx-1.9.12.min.js?v=' in html
    assert 'src="/static/vendor/htmx-sse-1.9.12.min.js?v=' in html
    assert 'src="/static/vendor/alpine-3.14.1.min.js?v=' in html
    for asset in (
        "vendor/htmx-1.9.12.min.js",
        "vendor/htmx-sse-1.9.12.min.js",
        "vendor/alpine-3.14.1.min.js",
    ):
        idx = html.find(asset)
        # Everything vendor must load deferred — check the closest
        # preceding `<script` tag opened with `defer`.
        prefix = html[:idx]
        last_script_open = prefix.rfind("<script")
        assert last_script_open != -1
        assert "defer" in prefix[last_script_open:], (
            f"vendor asset {asset!r} not loaded with defer"
        )

    # tokens.css <link> present and cache-busted.
    assert 'href="/static/css/tokens.css?v=' in html

    # Ordering: tokens.css MUST come after the Carbon CDN <link> so its
    # rules win at equal specificity.
    carbon_idx = html.find("carbon-components")
    tokens_idx = html.find("/static/css/tokens.css")
    assert carbon_idx != -1, "Carbon CDN link missing from _layout"
    assert tokens_idx != -1
    assert carbon_idx < tokens_idx, (
        "tokens.css must be loaded AFTER Carbon so equal-specificity "
        "rules in tokens.css win"
    )

    # Ordering: htmx core loads before its SSE extension.
    htmx_idx = html.find("vendor/htmx-1.9.12.min.js")
    sse_idx = html.find("vendor/htmx-sse-1.9.12.min.js")
    assert htmx_idx < sse_idx, "htmx must load before htmx-sse extension"


def test_layout_still_loads_carbon_and_app_css(tmp_path):
    """Phase 0 is dual-load: Carbon + app.css + tokens.css all present.
    The migration off Carbon happens after Phase 8 — until then this
    triple-stylesheet contract must hold.
    """
    cfg = _cfg(tmp_path)
    html = _layout("Test", "", cfg, active="status")
    assert "carbon-components" in html
    assert "/static/css/app.css?v=" in html
    assert "/static/css/tokens.css?v=" in html


def test_vendor_files_reject_dot_dot_traversal(tmp_path):
    """Defense-in-depth: no `..` traversal even into the new vendor dir."""
    status, _, _ = _serve_static(_cfg(tmp_path), "vendor/../vendor/htmx-1.9.12.min.js")
    # Regex + the `..` guard both refuse this; either way the response is 404.
    assert status == 404
