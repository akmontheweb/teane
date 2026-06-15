"""Tests for the static web-asset reference scanner.

Covers `harness.web_asset_scan.scan_web_asset_references` (Layer 2 of the
web-app validation gate) and the CLI entry point used by the Layer 3
Makefile build target.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from contextlib import redirect_stderr

import pytest

from harness.web_asset_scan import (
    AssetRefDiagnostic,
    main as cli_main,
    scan_web_asset_references,
)


def _write(workspace: str, relpath: str, content: str) -> str:
    """Helper: create a file inside the tmp workspace, mkdirs as needed."""
    abs_path = os.path.join(workspace, relpath)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return abs_path


class TestScannerHappyPath:
    def test_all_refs_resolve_returns_no_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "index.html",
                   '<link rel="stylesheet" href="style.css">\n'
                   '<script src="src/main.js"></script>\n')
            _write(tmp, "style.css", "body { color: red; }\n")
            _write(tmp, "src/main.js", "console.log('hi');\n")
            diagnostics = scan_web_asset_references(tmp)
            assert diagnostics == []

    def test_empty_workspace_returns_no_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert scan_web_asset_references(tmp) == []


class TestScannerFindsMissing:
    def test_missing_css_file_one_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "index.html",
                   '<link rel="stylesheet" href="missing.css">\n')
            diagnostics = scan_web_asset_references(tmp)
            assert len(diagnostics) == 1
            d = diagnostics[0]
            assert d.referring_file == "index.html"
            assert d.raw_reference == "missing.css"
            assert d.line == 1

    def test_path_mismatch_suggests_correct_path(self):
        """The exact ticktaktoe bug: HTML references src/styles.css but only
        style.css exists at root."""
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "index.html",
                   '<link rel="stylesheet" href="src/styles.css">\n')
            _write(tmp, "style.css", "body { }\n")
            diagnostics = scan_web_asset_references(tmp)
            assert len(diagnostics) == 1
            d = diagnostics[0]
            assert d.raw_reference == "src/styles.css"
            assert d.suggested_path == "style.css"

    def test_missing_js_import_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "src/main.js",
                   'import { foo } from "./missing.js";\n')
            diagnostics = scan_web_asset_references(tmp)
            assert len(diagnostics) == 1
            assert diagnostics[0].raw_reference == "./missing.js"

    def test_missing_img_src_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "index.html",
                   '<img src="logo.png" alt="Logo">\n')
            diagnostics = scan_web_asset_references(tmp)
            assert len(diagnostics) == 1
            assert diagnostics[0].raw_reference == "logo.png"

    def test_css_url_unresolved_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "style.css",
                   'body { background: url("bg.png"); }\n')
            diagnostics = scan_web_asset_references(tmp)
            assert len(diagnostics) == 1
            assert diagnostics[0].raw_reference == "bg.png"

    def test_css_import_unresolved_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "style.css",
                   '@import "missing.css";\n')
            diagnostics = scan_web_asset_references(tmp)
            assert len(diagnostics) == 1
            assert diagnostics[0].raw_reference == "missing.css"


class TestScannerSkips:
    def test_http_references_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "index.html",
                   '<link rel="stylesheet" href="https://cdn.example.com/x.css">\n'
                   '<script src="http://cdn.example.com/x.js"></script>\n'
                   '<a href="https://example.com">link</a>\n')
            assert scan_web_asset_references(tmp) == []

    def test_protocol_relative_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "index.html",
                   '<script src="//cdn.example.com/x.js"></script>\n')
            assert scan_web_asset_references(tmp) == []

    def test_mailto_tel_data_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "index.html",
                   '<a href="mailto:x@y.com">mail</a>\n'
                   '<a href="tel:+15551234">call</a>\n'
                   '<img src="data:image/png;base64,iVBOR">\n')
            assert scan_web_asset_references(tmp) == []

    def test_anchor_only_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "index.html",
                   '<a href="#section">jump</a>\n')
            assert scan_web_asset_references(tmp) == []

    def test_bare_module_specifier_skipped(self):
        """`import "react"` is a bare specifier resolved by the bundler —
        not a path the scanner should validate."""
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "src/main.js",
                   'import React from "react";\n'
                   'import { foo } from "lodash/fp";\n')
            assert scan_web_asset_references(tmp) == []

    def test_relative_import_with_resolved_path_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "src/main.js", 'import { foo } from "./util.js";\n')
            _write(tmp, "src/util.js", "export const foo = 1;\n")
            assert scan_web_asset_references(tmp) == []


class TestScannerScopedToChangedFiles:
    def test_changed_files_filter_skips_other_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            # File NOT in the changed set has a broken ref.
            _write(tmp, "other.html",
                   '<link rel="stylesheet" href="ghost.css">\n')
            # File IN the changed set has a broken ref.
            _write(tmp, "index.html",
                   '<link rel="stylesheet" href="missing.css">\n')
            diagnostics = scan_web_asset_references(
                tmp, changed_files=["index.html"]
            )
            assert len(diagnostics) == 1
            assert diagnostics[0].referring_file == "index.html"

    def test_changed_files_with_no_web_files_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "main.py", "print('hi')\n")
            diagnostics = scan_web_asset_references(
                tmp, changed_files=["main.py"]
            )
            assert diagnostics == []


class TestDiagnosticFormatting:
    def test_format_compiler_style_includes_suggestion(self):
        d = AssetRefDiagnostic(
            referring_file="index.html",
            line=8,
            column=14,
            raw_reference="src/styles.css",
            resolved_path="src/styles.css",
            suggested_path="style.css",
        )
        out = d.format_compiler_style()
        assert out.startswith("index.html:8:14: error:")
        assert "unresolved asset reference 'src/styles.css'" in out
        assert "did you mean 'style.css'" in out

    def test_format_compiler_style_without_suggestion(self):
        d = AssetRefDiagnostic(
            referring_file="index.html", line=1, column=1,
            raw_reference="ghost.png", resolved_path="ghost.png",
        )
        out = d.format_compiler_style()
        assert "did you mean" not in out


class TestCli:
    def test_cli_zero_exit_when_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "index.html", '<link rel="stylesheet" href="x.css">\n')
            _write(tmp, "x.css", "body { }\n")
            assert cli_main([tmp]) == 0

    def test_cli_nonzero_exit_when_unresolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "index.html", '<link href="missing.css">\n')
            buf = io.StringIO()
            with redirect_stderr(buf):
                rc = cli_main([tmp])
            assert rc == 1
            out = buf.getvalue()
            assert "index.html:1:" in out
            assert "error:" in out
            assert "missing.css" in out


class TestLintgateIntegration:
    """End-to-end: lintgate_node picks up web asset diagnostics and surfaces
    them in node_state for the repair loop."""

    @pytest.mark.asyncio
    async def test_lintgate_node_surfaces_web_asset_errors(self):
        from harness.lintgate import lintgate_node

        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "index.html",
                   '<link rel="stylesheet" href="src/styles.css">\n')
            _write(tmp, "style.css", "body { }\n")

            state = {
                "modified_files": ["index.html"],
                "workspace_path": tmp,
            }
            result = await lintgate_node(state)
            lg = result["node_state"]["lintgate"]
            assert lg["web_asset_errors"], "expected at least one asset error"
            err = lg["web_asset_errors"][0]
            assert err["referring_file"] == "index.html"
            assert err["raw_reference"] == "src/styles.css"
            assert err["suggested_path"] == "style.css"
            # Also flows through lint_errors so the existing repair loop
            # (graph.py:3481) consumes it.
            assert any("src/styles.css" in e for e in lg["lint_errors"])

    @pytest.mark.asyncio
    async def test_lintgate_node_skips_scan_for_backend_only(self):
        """Backend-only workspaces (no HTML, no frontend framework) should
        not pay the cost of the scan."""
        from harness.lintgate import lintgate_node

        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "main.py", "print('hi')\n")
            state = {
                "modified_files": ["main.py"],
                "workspace_path": tmp,
            }
            result = await lintgate_node(state)
            lg = result["node_state"]["lintgate"]
            # When there are no formatters AND no html, an early-return path
            # may fire; either way, no web asset errors should be reported.
            assert lg.get("web_asset_errors", []) == []


class TestInventoryPostPatchCheck:
    """End-to-end: lintgate reads SPEC_ARCHITECTURE.md, parses the JSON
    inventory, and surfaces a diagnostic when a manifest file is missing
    from disk."""

    @pytest.mark.asyncio
    async def test_missing_file_from_inventory_surfaced(self):
        from harness.lintgate import lintgate_node

        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "index.html", '<html></html>\n')
            os.makedirs(os.path.join(tmp, "docs"))
            _write(tmp, "docs/SPEC_ARCHITECTURE.md",
                   '# Architecture\n\n```json\n'
                   '{"files": ['
                   '{"path": "index.html", "kind": "html"},'
                   '{"path": "style.css", "kind": "css"}'
                   ']}\n```\n')

            state = {
                "modified_files": ["index.html"],
                "workspace_path": tmp,
                "spec_architecture_path": os.path.join(
                    tmp, "docs", "SPEC_ARCHITECTURE.md"
                ),
            }
            result = await lintgate_node(state)
            lg = result["node_state"]["lintgate"]
            # The style.css declared in the inventory wasn't written.
            assert any(
                "style.css" in err and "no file was written" in err
                for err in lg["lint_errors"]
            ), f"expected MISSING_FROM_DISK; got {lg['lint_errors']!r}"

    @pytest.mark.asyncio
    async def test_full_inventory_match_no_diagnostics(self):
        from harness.lintgate import lintgate_node

        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "index.html", '<link href="style.css">\n')
            _write(tmp, "style.css", "body { }\n")
            os.makedirs(os.path.join(tmp, "docs"))
            _write(tmp, "docs/SPEC_ARCHITECTURE.md",
                   '# Architecture\n\n```json\n'
                   '{"files": ['
                   '{"path": "index.html", "kind": "html"},'
                   '{"path": "style.css", "kind": "css"}'
                   ']}\n```\n')

            state = {
                "modified_files": ["index.html", "style.css"],
                "workspace_path": tmp,
                "spec_architecture_path": os.path.join(
                    tmp, "docs", "SPEC_ARCHITECTURE.md"
                ),
            }
            result = await lintgate_node(state)
            lg = result["node_state"]["lintgate"]
            # No web-asset and no inventory misses.
            assert lg.get("web_asset_errors", []) == []
            assert not any(
                "no file was written" in e for e in lg["lint_errors"]
            )


class TestAutofixR6:
    """Tests for the R6 web-asset reference dispatcher in harness/autofix.py."""

    @pytest.mark.asyncio
    async def test_r6_rewrites_unique_reference(self):
        from harness.autofix import (
            apply_autofixes,
            web_asset_diagnostics_to_standard,
        )

        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "index.html",
                   '<link rel="stylesheet" href="src/styles.css">\n')
            _write(tmp, "style.css", "body { }\n")

            web_errs = [{
                "referring_file": "index.html",
                "line": 1, "column": 14,
                "raw_reference": "src/styles.css",
                "resolved_path": "src/styles.css",
                "suggested_path": "style.css",
            }]
            diagnostics = web_asset_diagnostics_to_standard(web_errs)
            unhandled, applied = await apply_autofixes(diagnostics, tmp)
            assert len(applied) == 1
            assert applied[0].fix_kind == "web_asset"
            assert unhandled == []

            with open(os.path.join(tmp, "index.html")) as f:
                rewritten = f.read()
            assert 'href="style.css"' in rewritten
            assert "src/styles.css" not in rewritten

    @pytest.mark.asyncio
    async def test_r6_skips_when_no_suggestion(self):
        from harness.autofix import (
            apply_autofixes,
            web_asset_diagnostics_to_standard,
        )

        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "index.html", '<link href="missing.css">\n')
            web_errs = [{
                "referring_file": "index.html",
                "line": 1, "column": 13,
                "raw_reference": "missing.css",
                "resolved_path": "missing.css",
                "suggested_path": None,
            }]
            diagnostics = web_asset_diagnostics_to_standard(web_errs)
            unhandled, applied = await apply_autofixes(diagnostics, tmp)
            # No suggestion → R6 returns None → escalates to LLM.
            assert applied == []
            assert len(unhandled) == 1

    @pytest.mark.asyncio
    async def test_r6_skips_when_reference_appears_multiple_times(self):
        """If raw_reference is ambiguous within the file, punt to LLM."""
        from harness.autofix import (
            apply_autofixes,
            web_asset_diagnostics_to_standard,
        )

        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "index.html",
                   '<link rel="stylesheet" href="src/styles.css">\n'
                   '<link rel="preload" href="src/styles.css">\n')
            _write(tmp, "style.css", "body { }\n")

            web_errs = [{
                "referring_file": "index.html",
                "line": 1, "column": 14,
                "raw_reference": "src/styles.css",
                "resolved_path": "src/styles.css",
                "suggested_path": "style.css",
            }]
            diagnostics = web_asset_diagnostics_to_standard(web_errs)
            unhandled, applied = await apply_autofixes(diagnostics, tmp)
            # Ambiguous anchor → R6 returns None.
            assert applied == []

    @pytest.mark.asyncio
    async def test_r6_ignores_non_web_asset_diagnostics(self):
        """A regular compiler diagnostic should not trigger R6."""
        from harness.autofix import _try_asset_reference_fix

        with tempfile.TemporaryDirectory() as tmp:
            diag = {
                "file": "main.py",
                "error_code": "F821",
                "message": "undefined name 'x'",
            }
            assert _try_asset_reference_fix(diag, tmp) is None


class TestRegressionTicktaktoe:
    """The exact bug that motivated this PR."""

    def test_ticktaktoe_shape_reproduces_and_suggests_fix(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Mirror the ticktaktoe project shape: an index.html at root
            # referencing src/styles.css (which doesn't exist) and various
            # src/*.js modules (which do exist).
            _write(tmp, "index.html",
                   '<!DOCTYPE html>\n'
                   '<html>\n'
                   '<head>\n'
                   '  <link rel="stylesheet" href="src/styles.css">\n'
                   '</head>\n'
                   '<body>\n'
                   '  <script type="module" src="src/main.js"></script>\n'
                   '</body>\n'
                   '</html>\n')
            _write(tmp, "src/main.js", "console.log('hi');\n")
            # An almost-matching CSS file at root — the LLM meant this one.
            _write(tmp, "style.css", "body { color: black; }\n")

            diagnostics = scan_web_asset_references(tmp)
            assert len(diagnostics) == 1
            d = diagnostics[0]
            assert d.referring_file == "index.html"
            assert d.raw_reference == "src/styles.css"
            assert d.suggested_path == "style.css"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
