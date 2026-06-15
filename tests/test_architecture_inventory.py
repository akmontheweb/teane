"""Tests for the structured file inventory (Layer 1 of the web-app gate).

Covers parse_inventory, cross_check_inventories, and check_files_on_disk
in harness/architecture_inventory.py.
"""

from __future__ import annotations

import os
import tempfile

from harness.architecture_inventory import (
    FileEntry,
    InventoryDiagnostic,
    check_files_on_disk,
    cross_check_inventories,
    parse_inventory,
)


# --- parse_inventory ---------------------------------------------------------


class TestParseInventory:
    def test_parses_well_formed_block(self):
        spec = """
# Architecture

Some prose.

```json
{
  "files": [
    {"path": "index.html", "purpose": "entry", "kind": "html"},
    {"path": "style.css", "purpose": "styling", "kind": "css"}
  ]
}
```

More prose.
"""
        result = parse_inventory(spec)
        assert result.ok
        assert len(result.files) == 2
        assert result.files[0].path == "index.html"
        assert result.files[0].kind == "html"
        assert result.files[1].path == "style.css"

    def test_missing_block_yields_error(self):
        spec = "# Architecture\n\nNo inventory anywhere here.\n"
        result = parse_inventory(spec)
        assert not result.ok
        assert "no fenced json" in result.error.lower()

    def test_empty_document_yields_error(self):
        result = parse_inventory("")
        assert not result.ok
        assert "empty" in result.error.lower()

    def test_malformed_json_skipped_then_falls_through(self):
        """A broken JSON block should not crash — we keep scanning for a
        valid one. If none exists, return the no-block error."""
        spec = """
```json
{not valid json}
```
"""
        result = parse_inventory(spec)
        assert not result.ok

    def test_skips_json_block_without_files_array(self):
        """JSON blocks for other purposes (e.g. example config) must not
        be mistaken for the inventory."""
        spec = """
```json
{"some_other_data": [1, 2, 3]}
```

```json
{"files": [{"path": "index.html"}]}
```
"""
        result = parse_inventory(spec)
        assert result.ok
        assert len(result.files) == 1
        assert result.files[0].path == "index.html"

    def test_files_array_with_missing_path_errors(self):
        spec = """
```json
{"files": [{"purpose": "entry"}]}
```
"""
        result = parse_inventory(spec)
        assert not result.ok
        assert "path" in result.error.lower()

    def test_path_normalization_strips_leading_dotslash(self):
        spec = """
```json
{"files": [{"path": "./src/main.js"}]}
```
"""
        result = parse_inventory(spec)
        assert result.ok
        assert result.files[0].path == "src/main.js"

    def test_untagged_fence_fallback(self):
        """Some LLMs forget the ```json language hint. Untagged fences
        with valid JSON should still work."""
        spec = """
```
{"files": [{"path": "index.html"}]}
```
"""
        result = parse_inventory(spec)
        assert result.ok
        assert result.files[0].path == "index.html"


# --- cross_check_inventories -------------------------------------------------


class TestCrossCheck:
    def test_perfect_match_yields_no_diagnostics(self):
        arch = [FileEntry("index.html"), FileEntry("style.css")]
        plan = [FileEntry("index.html"), FileEntry("style.css")]
        assert cross_check_inventories(arch, plan) == []

    def test_missing_from_plan_diagnostic(self):
        arch = [FileEntry("index.html"), FileEntry("style.css")]
        plan = [FileEntry("index.html")]
        diags = cross_check_inventories(arch, plan)
        assert len(diags) == 1
        assert diags[0].kind == "MISSING_FROM_PLAN"
        assert diags[0].file == "style.css"
        assert diags[0].is_error is True

    def test_path_mismatch_emits_suggestion(self):
        """Architecture says style.css at root; planning says src/styles.css.
        This is the exact ticktaktoe failure mode."""
        arch = [FileEntry("style.css")]
        plan = [FileEntry("src/styles.css")]
        diags = cross_check_inventories(arch, plan)
        # One PATH_MISMATCH for arch -> plan rename. No EXTRA_IN_PLAN
        # because the plan entry was already accounted for by the mismatch.
        assert len(diags) == 1
        d = diags[0]
        assert d.kind == "PATH_MISMATCH"
        assert d.file == "style.css"
        assert d.suggested_path == "src/styles.css"
        assert d.is_error is True

    def test_extra_in_plan_is_advisory(self):
        arch = [FileEntry("index.html")]
        plan = [FileEntry("index.html"), FileEntry("extra.js")]
        diags = cross_check_inventories(arch, plan)
        assert len(diags) == 1
        assert diags[0].kind == "EXTRA_IN_PLAN"
        assert diags[0].file == "extra.js"
        assert diags[0].is_error is False  # advisory only

    def test_path_mismatch_handles_pluralization_drift(self):
        """style.css vs styles.css — stem-cousin match."""
        arch = [FileEntry("style.css")]
        plan = [FileEntry("styles.css")]
        diags = cross_check_inventories(arch, plan)
        assert len(diags) == 1
        assert diags[0].kind == "PATH_MISMATCH"
        assert diags[0].suggested_path == "styles.css"

    def test_ambiguous_basename_stays_missing(self):
        """If multiple plan files share the basename, don't emit a
        confident PATH_MISMATCH — surface MISSING_FROM_PLAN instead."""
        arch = [FileEntry("config.json")]
        plan = [FileEntry("src/config.json"), FileEntry("tests/config.json")]
        diags = cross_check_inventories(arch, plan)
        kinds = {d.kind for d in diags}
        # Architecture's config.json is MISSING_FROM_PLAN; plan entries are
        # both EXTRA_IN_PLAN (advisory).
        assert "MISSING_FROM_PLAN" in kinds


# --- check_files_on_disk -----------------------------------------------------


class TestPostPatchCheck:
    def test_all_files_present_no_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("index.html", "style.css"):
                with open(os.path.join(tmp, name), "w") as fh:
                    fh.write("")
            manifest = [FileEntry("index.html"), FileEntry("style.css")]
            assert check_files_on_disk(manifest, tmp) == []

    def test_missing_file_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "index.html"), "w") as fh:
                fh.write("")
            manifest = [FileEntry("index.html"), FileEntry("style.css")]
            diags = check_files_on_disk(manifest, tmp)
            assert len(diags) == 1
            assert diags[0].kind == "MISSING_FROM_DISK"
            assert diags[0].file == "style.css"

    def test_nested_path_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "src"))
            with open(os.path.join(tmp, "src", "main.js"), "w") as fh:
                fh.write("")
            manifest = [FileEntry("src/main.js")]
            assert check_files_on_disk(manifest, tmp) == []


# --- Regression: full ticktaktoe scenario -----------------------------------


class TestRegressionTicktaktoe:
    def test_ticktaktoe_architecture_planning_mismatch_caught(self):
        """The exact failure: SPEC_ARCHITECTURE.md lists style.css at root;
        the planning manifest (had we required one) would list src/styles.css.
        Layer 1 should catch this before any file is written."""
        arch_md = """
# Architecture

## Module Inventory

```
├── index.html
├── style.css
└── src/
    └── main.js
```

```json
{
  "files": [
    {"path": "index.html", "purpose": "entry", "kind": "html"},
    {"path": "style.css", "purpose": "styling", "kind": "css"},
    {"path": "src/main.js", "purpose": "bootstrap", "kind": "js"}
  ]
}
```
"""
        plan_md = """
# Plan

```json
{
  "files": [
    {"path": "index.html", "kind": "html"},
    {"path": "src/styles.css", "kind": "css"},
    {"path": "src/main.js", "kind": "js"}
  ]
}
```
"""
        arch_result = parse_inventory(arch_md)
        plan_result = parse_inventory(plan_md)
        assert arch_result.ok and plan_result.ok
        diags = cross_check_inventories(arch_result.files, plan_result.files)
        # Expect exactly one PATH_MISMATCH for style.css → src/styles.css.
        mismatches = [d for d in diags if d.kind == "PATH_MISMATCH"]
        assert len(mismatches) == 1
        assert mismatches[0].file == "style.css"
        assert mismatches[0].suggested_path == "src/styles.css"


# --- Compiler-style formatting ----------------------------------------------


class TestDiagnosticFormatting:
    def test_format_compiler_style_includes_suggestion(self):
        d = InventoryDiagnostic(
            kind="PATH_MISMATCH",
            file="style.css",
            message="architecture lists 'style.css' but planning references 'src/styles.css' instead",
            suggested_path="src/styles.css",
        )
        out = d.format_compiler_style()
        assert out.startswith("style.css:1:1: error:")
        assert "did you mean 'src/styles.css'" in out
