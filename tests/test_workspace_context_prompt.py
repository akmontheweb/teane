"""Tests for the workspace-context prompt enrichment helpers added to
``harness.graph`` (fixes 1+2+3 from the ciod post-mortem).

The trio surfaces three pieces of workspace-level knowledge into the
repair prompt whenever the persistent-blocker directive fires:

  1. ``_record_file_modification_history`` — per-file session log of
     which operations already ran against a file, so the LLM stops
     emitting the same REWRITE_FILE twice in a row.
  2. ``_format_workspace_context_for_files`` — renders per-file
     modification history + currently-defined symbols + callers +
     missing-symbol candidates into a compact markdown block.
  3. ``_infer_missing_exports_for_module`` — grep-based cross-check
     for names other files import from a module but which aren't in
     that module's own export set (the ciod b9369w5uu
     ``BlacklistedToken`` bug pattern).

Support matrix mirrors the harness's declared stack: Python + Java for
backend, TS/TSX/JSX for frontend. Non-Python files exercise the
symbol+callers path but not the ``_infer_missing_exports`` grep
(deliberately Python-only for now)."""

from __future__ import annotations

import os
import tempfile
import textwrap

from harness.graph import (
    _format_workspace_context_for_files,
    _infer_missing_exports_for_module,
    _record_file_modification_history,
)


class _FakeResult:
    """Minimal duck-typed stand-in for ``PatchBlockResult`` — the real
    class carries a lot of extra fields we don't need here."""

    def __init__(
        self,
        *,
        file: str,
        operation: str,
        success: bool,
        no_op: bool = False,
        error: str | None = None,
    ) -> None:
        self.file = file
        # Match the shape callers unwrap in graph.py: ``r.operation.value``
        # when the op is an enum, or the raw string when it's a string.
        # We hand a plain string; the graph code path uses ``str(op)``.
        self.operation = operation
        self.success = success
        self.no_op = no_op
        self.error = error


class TestRecordFileModificationHistory:
    def test_appends_entry_per_result(self):
        lc: dict = {}
        _record_file_modification_history(
            lc,
            3,
            [
                _FakeResult(file="a.py", operation="replace_block", success=True),
                _FakeResult(file="b.py", operation="rewrite_file", success=False, no_op=True,
                            error="REWRITE_FILE no-op: content byte-identical"),
            ],
        )
        hist = lc["file_modification_history"]
        assert set(hist.keys()) == {"a.py", "b.py"}
        assert hist["a.py"][0][:4] == [3, "replace_block", True, False]
        # note is the truncated first line of error
        assert hist["b.py"][0][:4] == [3, "rewrite_file", False, True]
        assert "byte-identical" in hist["b.py"][0][4]

    def test_ignores_results_with_no_file(self):
        lc: dict = {}
        _record_file_modification_history(
            lc, 1,
            [_FakeResult(file="", operation="create_file", success=True)],
        )
        assert lc["file_modification_history"] == {}

    def test_accumulates_across_rounds(self):
        lc: dict = {}
        _record_file_modification_history(
            lc, 1, [_FakeResult(file="a.py", operation="replace_block", success=False)],
        )
        _record_file_modification_history(
            lc, 2, [_FakeResult(file="a.py", operation="rewrite_file", success=True)],
        )
        entries = lc["file_modification_history"]["a.py"]
        assert [e[0] for e in entries] == [1, 2]

    def test_bounded_per_file(self):
        # Feed more than the cap; oldest entries are evicted, newest kept.
        lc: dict = {}
        for i in range(25):
            _record_file_modification_history(
                lc, i,
                [_FakeResult(file="x.py", operation="replace_block", success=True)],
            )
        entries = lc["file_modification_history"]["x.py"]
        assert len(entries) <= 12
        # Last entry should be the most recent round we recorded.
        assert entries[-1][0] == 24

    def test_survives_bad_prior_state(self):
        # Corrupted checkpoint — history is a string, not a dict.
        lc: dict = {"file_modification_history": "garbage"}
        _record_file_modification_history(
            lc, 1,
            [_FakeResult(file="a.py", operation="create_file", success=True)],
        )
        # Was replaced with a fresh dict.
        assert isinstance(lc["file_modification_history"], dict)
        assert "a.py" in lc["file_modification_history"]


class TestInferMissingExportsPythonOnly:
    def test_finds_names_imported_but_not_exported(self):
        with tempfile.TemporaryDirectory() as td:
            # models/__init__.py exports only User.
            os.makedirs(os.path.join(td, "server", "models"))
            with open(os.path.join(td, "server", "models", "__init__.py"), "w") as f:
                f.write("class User:\n    pass\n")
            # A caller expects BlacklistedToken + CsrfToken too — the
            # classic ciod b9369w5uu bug pattern.
            os.makedirs(os.path.join(td, "server", "auth"))
            with open(os.path.join(td, "server", "auth", "services.py"), "w") as f:
                f.write(
                    "from server.models import User, BlacklistedToken, CsrfToken\n"
                )
            from harness.impact import DependencyGraph
            g = DependencyGraph(td)
            g.build()
            module = os.path.join(td, "server", "models", "__init__.py")
            missing = _infer_missing_exports_for_module(g, td, module)
            assert "BlacklistedToken" in missing
            assert "CsrfToken" in missing
            assert "User" not in missing  # already exported

    def test_returns_empty_for_non_python_module(self):
        with tempfile.TemporaryDirectory() as td:
            from harness.impact import DependencyGraph
            g = DependencyGraph(td)
            g.build()
            assert _infer_missing_exports_for_module(
                g, td, os.path.join(td, "src", "index.ts"),
            ) == []

    def test_ignores_star_imports(self):
        # `from X import *` shouldn't fabricate a missing "star" name.
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "pkg"))
            with open(os.path.join(td, "pkg", "core.py"), "w") as f:
                f.write("A = 1\n")
            with open(os.path.join(td, "pkg", "user.py"), "w") as f:
                f.write("from pkg.core import *\n")
            from harness.impact import DependencyGraph
            g = DependencyGraph(td)
            g.build()
            module = os.path.join(td, "pkg", "core.py")
            assert _infer_missing_exports_for_module(g, td, module) == []


class TestFormatWorkspaceContext:
    def test_empty_when_no_files(self):
        assert _format_workspace_context_for_files([], "/tmp", {}) == ""
        assert _format_workspace_context_for_files({""}, "/tmp", {}) == ""

    def test_python_backend_symbol_and_callers(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "server", "models"))
            with open(os.path.join(td, "server", "models", "__init__.py"), "w") as f:
                f.write("class User:\n    pass\n\nclass Product:\n    pass\n")
            os.makedirs(os.path.join(td, "server", "auth"))
            with open(os.path.join(td, "server", "auth", "services.py"), "w") as f:
                f.write(
                    "from server.models import User, BlacklistedToken\n"
                    "def login(): pass\n"
                )
            rel = "server/models/__init__.py"
            block = _format_workspace_context_for_files(
                {rel}, td, {"file_modification_history": {}},
            )
            assert "`server/models/__init__.py`" in block
            assert "modification history" in block.lower()
            # Symbol registry surfaces both defined classes.
            assert "User" in block
            assert "Product" in block
            # Caller listing surfaces the auth module + its imports.
            assert "server/auth/services.py" in block
            # Missing-symbol candidate: BlacklistedToken.
            assert "BlacklistedToken" in block
            assert "Missing-symbol candidates" in block

    def test_history_note_surfaces_no_op_failure(self):
        # When the LLM already emitted a REWRITE_FILE no-op against a
        # file, the banner must call it out explicitly.
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "m.py"), "w") as f:
                f.write("x = 1\n")
            loop_counter = {"file_modification_history": {}}
            _record_file_modification_history(
                loop_counter, 5,
                [_FakeResult(
                    file="m.py",
                    operation="rewrite_file",
                    success=False,
                    no_op=True,
                    error="REWRITE_FILE no-op: content byte-identical",
                )],
            )
            block = _format_workspace_context_for_files(
                {"m.py"}, td, loop_counter,
            )
            assert "Round 5" in block
            assert "NO-OP FAILURE" in block

    def test_java_backend_symbols_surface(self):
        # Java support: the DependencyGraph parses .java exports via
        # the tree-sitter-java grammar. Verify symbols surface but the
        # Python-only missing-export cross-check does NOT run.
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "src", "main", "java", "com", "app"))
            java_path = os.path.join(
                td, "src", "main", "java", "com", "app", "UserService.java",
            )
            with open(java_path, "w") as f:
                f.write(textwrap.dedent("""
                    package com.app;
                    public class UserService {
                        public void login() {}
                    }
                """).lstrip())
            rel = "src/main/java/com/app/UserService.java"
            block = _format_workspace_context_for_files(
                {rel}, td, {"file_modification_history": {}},
            )
            assert "UserService" in block
            # Python-only missing-export section MUST NOT appear on .java.
            assert "Missing-symbol candidates" not in block

    def test_typescript_frontend_symbols_surface(self):
        # React+TS+Tailwind: .tsx components export symbols the graph
        # can see. Verify a TSX file's default/named exports render.
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "src", "components"))
            tsx_path = os.path.join(td, "src", "components", "Button.tsx")
            with open(tsx_path, "w") as f:
                f.write(textwrap.dedent("""
                    export const Button = () => <button>Click</button>;
                    export const PrimaryButton = () => <Button />;
                """).lstrip())
            rel = "src/components/Button.tsx"
            block = _format_workspace_context_for_files(
                {rel}, td, {"file_modification_history": {}},
            )
            # At least ONE of the exported symbols surfaces (grammar
            # coverage varies by tree-sitter build; the harness's
            # baseline covers `export const`).
            assert "Button" in block
            assert "Missing-symbol candidates" not in block

    def test_missing_dep_graph_module_does_not_crash(self, monkeypatch):
        # If DependencyGraph raises during build, the formatter should
        # still return the modification-history section without dying.
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "m.py"), "w") as f:
                f.write("x = 1\n")
            loop_counter = {
                "file_modification_history": {
                    "m.py": [[1, "replace_block", True, False, ""]],
                }
            }

            # Force graph build failure by monkeypatching the module
            # attribute the helper imports.
            import harness.impact as _impact_mod

            class _Bad:
                def __init__(self, *a, **kw): raise RuntimeError("boom")

            monkeypatch.setattr(_impact_mod, "DependencyGraph", _Bad)
            block = _format_workspace_context_for_files(
                {"m.py"}, td, loop_counter,
            )
            # History still rendered; symbol sections silently skipped.
            assert "Round 1" in block
            assert "modification history" in block.lower()
