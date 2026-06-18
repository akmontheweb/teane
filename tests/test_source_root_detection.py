"""Tests for harness/impact.py:_detect_source_root.

The helper picks the dominant top-level directory containing source files,
so the LLM and the patcher can constrain new code to the workspace's
existing layout. Returns None when the layout is flat or ambiguous.
"""

from __future__ import annotations

from pathlib import Path

from harness.impact import _detect_source_root, _detect_source_roots


def _touch(path: Path) -> None:
    """Create an empty file at ``path``, including parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


class TestPreferredNameBias:

    def test_app_dominant_workspace_returns_app(self, tmp_path):
        for name in ("calculator.py", "auth.py", "models.py"):
            _touch(tmp_path / "app" / name)
        _touch(tmp_path / "pyproject.toml")
        assert _detect_source_root(str(tmp_path)) == "app"

    def test_src_dominant_workspace_returns_src(self, tmp_path):
        for name in ("foo.ts", "bar.ts", "baz.ts"):
            _touch(tmp_path / "src" / name)
        _touch(tmp_path / "package.json")
        assert _detect_source_root(str(tmp_path)) == "src"

    def test_lib_dominant_workspace_returns_lib(self, tmp_path):
        for name in ("foo.rs", "bar.rs"):
            _touch(tmp_path / "lib" / name)
        _touch(tmp_path / "Cargo.toml")
        assert _detect_source_root(str(tmp_path)) == "lib"

    def test_preferred_name_beats_larger_non_preferred(self, tmp_path):
        # vendored 'random_thing/' has more files, but 'app/' is the
        # idiomatic location — we MUST prefer it. Otherwise the LLM
        # would be told to write code into a vendored tree.
        for i in range(10):
            _touch(tmp_path / "random_thing" / f"f{i}.py")
        for name in ("a.py", "b.py"):
            _touch(tmp_path / "app" / name)
        assert _detect_source_root(str(tmp_path)) == "app"


class TestNonPreferredFallback:

    def test_dominant_non_preferred_directory_wins(self, tmp_path):
        # Go monorepo with `internal/` would already be preferred, so use
        # an unusual name to exercise the fallback path: only `core/`
        # has any source. The dominance test (≥80% or >3-vs-0) fires.
        for i in range(5):
            _touch(tmp_path / "core" / f"f{i}.py")
        _touch(tmp_path / "Makefile")
        assert _detect_source_root(str(tmp_path)) == "core"


class TestAmbiguousLayouts:

    def test_flat_workspace_returns_none(self, tmp_path):
        # All source at root → no source-root to constrain to.
        for name in ("a.py", "b.py", "c.py"):
            _touch(tmp_path / name)
        _touch(tmp_path / "pyproject.toml")
        assert _detect_source_root(str(tmp_path)) is None

    def test_evenly_split_returns_none(self, tmp_path):
        # 2 in foo/, 2 in bar/, neither preferred → ambiguous.
        for name in ("a.py", "b.py"):
            _touch(tmp_path / "foo" / name)
            _touch(tmp_path / "bar" / name)
        assert _detect_source_root(str(tmp_path)) is None

    def test_empty_workspace_returns_none(self, tmp_path):
        # No source files anywhere.
        _touch(tmp_path / "README.md")
        assert _detect_source_root(str(tmp_path)) is None

    def test_only_tests_returns_none(self, tmp_path):
        # tests/ is in _NEVER_SOURCE_DIRS, so it doesn't count.
        for name in ("test_a.py", "test_b.py"):
            _touch(tmp_path / "tests" / name)
        assert _detect_source_root(str(tmp_path)) is None

    def test_only_docs_returns_none(self, tmp_path):
        # docs/ ignored even if it contains .py files.
        _touch(tmp_path / "docs" / "conf.py")
        assert _detect_source_root(str(tmp_path)) is None


class TestRobustness:

    def test_handles_hidden_dirs(self, tmp_path):
        # .git, .venv, .cache must be skipped without breaking detection.
        for name in ("a.py", "b.py"):
            _touch(tmp_path / "app" / name)
        _touch(tmp_path / ".venv" / "site-packages" / "ignored.py")
        _touch(tmp_path / ".git" / "hooks" / "stuff.py")
        assert _detect_source_root(str(tmp_path)) == "app"

    def test_returns_none_for_nonexistent_path(self, tmp_path):
        assert _detect_source_root(str(tmp_path / "does-not-exist")) is None
        assert _detect_source_root("") is None

    def test_multilanguage_workspace(self, tmp_path):
        # Polyglot Python+TS under src/ — counts both, preferred-name
        # bias still picks src.
        for name in ("a.py", "b.ts", "c.tsx"):
            _touch(tmp_path / "src" / name)
        assert _detect_source_root(str(tmp_path)) == "src"

    def test_excludes_node_modules(self, tmp_path):
        # node_modules is huge but must be ignored — otherwise every
        # JS workspace would land there.
        for i in range(50):
            _touch(tmp_path / "node_modules" / "vendor" / f"f{i}.js")
        for name in ("app.ts", "utils.ts"):
            _touch(tmp_path / "src" / name)
        assert _detect_source_root(str(tmp_path)) == "src"


class TestDetectSourceRootsMultiRoot:
    """``_detect_source_roots`` (plural) returns every substantial
    top-level source dir. This is what the patcher allowlist actually
    consumes — using the singular ``_detect_source_root`` rejects every
    patch targeting the smaller side of a ``client/`` + ``server/``
    monorepo."""

    def test_client_server_monorepo_returns_both(self, tmp_path):
        # The reported failure case: React frontend in client/, Express
        # backend in server/. Both should be in the allowlist; before
        # this change, _detect_source_root picked whichever had more
        # files and the other side's patches were all rejected.
        for name in ("App.jsx", "index.js", "Login.jsx", "api.js"):
            _touch(tmp_path / "client" / "src" / name)
        for name in ("server.js", "routes.js", "db.js"):
            _touch(tmp_path / "server" / "src" / name)
        roots = _detect_source_roots(str(tmp_path))
        assert set(roots) == {"client", "server"}
        # Order is descending by file count.
        assert roots[0] == "client"  # 4 files > 3

    def test_frontend_backend_naming_returns_both(self, tmp_path):
        # Variant naming: same shape, different folder names.
        for name in ("a.tsx", "b.tsx", "c.tsx"):
            _touch(tmp_path / "frontend" / name)
        for name in ("a.py", "b.py", "c.py"):
            _touch(tmp_path / "backend" / name)
        assert set(_detect_source_roots(str(tmp_path))) == {"frontend", "backend"}

    def test_preferred_name_wins_solo_even_with_other_dirs(self, tmp_path):
        # `src/` is preferred — when it has source, it's authoritative
        # even if a non-preferred dir incidentally contains code.
        for name in ("a.py", "b.py"):
            _touch(tmp_path / "src" / name)
        for name in ("c.py", "d.py"):
            _touch(tmp_path / "scratch" / name)
        assert _detect_source_roots(str(tmp_path)) == ["src"]

    def test_filters_out_tiny_scratch_dirs(self, tmp_path):
        # client/ has 20 files, examples/ would have 1. examples is
        # already in _NEVER_SOURCE_DIRS, but the substantial-threshold
        # rule should also drop any 1-file sibling like "scratch/".
        for i in range(20):
            _touch(tmp_path / "client" / f"f{i}.js")
        _touch(tmp_path / "scratch" / "tiny.js")
        roots = _detect_source_roots(str(tmp_path))
        assert roots == ["client"]

    def test_substantial_dirs_below_dominance_threshold(self, tmp_path):
        # Singular returns None here (60/40 split doesn't pass the 80%
        # dominance rule). Plural keeps both because both are
        # substantial.
        for i in range(6):
            _touch(tmp_path / "api" / f"f{i}.py")
        for i in range(4):
            _touch(tmp_path / "worker" / f"f{i}.py")
        assert _detect_source_root(str(tmp_path)) is None
        assert set(_detect_source_roots(str(tmp_path))) == {"api", "worker"}

    def test_evenly_split_returns_both(self, tmp_path):
        # Same shape as the singular's test_evenly_split_returns_none,
        # but the plural returns both because each dir meets the
        # substantial threshold (≥2 files, ≥15% of largest).
        for name in ("a.py", "b.py"):
            _touch(tmp_path / "foo" / name)
            _touch(tmp_path / "bar" / name)
        assert set(_detect_source_roots(str(tmp_path))) == {"foo", "bar"}

    def test_flat_workspace_returns_empty(self, tmp_path):
        for name in ("a.py", "b.py"):
            _touch(tmp_path / name)
        assert _detect_source_roots(str(tmp_path)) == []

    def test_empty_workspace_returns_empty(self, tmp_path):
        _touch(tmp_path / "README.md")
        assert _detect_source_roots(str(tmp_path)) == []

    def test_excludes_never_source_dirs(self, tmp_path):
        # node_modules, build, dist, docs etc. must be excluded even
        # in the multi-root path.
        for i in range(50):
            _touch(tmp_path / "node_modules" / "pkg" / f"f{i}.js")
        for i in range(10):
            _touch(tmp_path / "build" / f"f{i}.js")
        for name in ("a.ts", "b.ts", "c.ts"):
            _touch(tmp_path / "client" / name)
            _touch(tmp_path / "server" / name)
        assert set(_detect_source_roots(str(tmp_path))) == {"client", "server"}

    def test_returns_empty_for_nonexistent_path(self, tmp_path):
        assert _detect_source_roots(str(tmp_path / "nope")) == []
        assert _detect_source_roots("") == []


class TestAllowlistEndToEndMonorepo:
    """End-to-end check: ``_build_patcher_allowlist`` + ``is_path_allowed``
    must accept co-located tests under any detected source root. This is
    the exact regression that produced the "Skill allowlist rejected
    patch to client/src/components/*.test.jsx" warnings."""

    def test_co_located_tests_under_multi_root_pass(self, tmp_path):
        from harness.graph import _build_patcher_allowlist
        from harness.trust import is_path_allowed

        for name in ("App.jsx", "Login.jsx", "Dashboard.jsx"):
            _touch(tmp_path / "client" / "src" / "components" / name)
        for name in ("server.js", "routes.js"):
            _touch(tmp_path / "server" / "src" / name)

        allowlist = _build_patcher_allowlist(str(tmp_path))
        assert allowlist is not None

        # The exact paths from the reported failure:
        assert is_path_allowed(
            "client/src/components/SearchBar.test.jsx",
            str(tmp_path), allowlist,
        )
        assert is_path_allowed(
            "client/src/hooks/useAuth.test.js",
            str(tmp_path), allowlist,
        )
        # Server-side patches still pass.
        assert is_path_allowed(
            "server/src/routes/auth.test.js",
            str(tmp_path), allowlist,
        )
        # Out-of-root paths are still blocked.
        assert not is_path_allowed(
            "evil/exfil.js", str(tmp_path), allowlist,
        )

    def test_single_root_allowlist_unchanged(self, tmp_path):
        # Sanity: preferred-name single root still produces the same
        # shape it did before (no behaviour change for the common case).
        from harness.graph import _build_patcher_allowlist
        from harness.trust import is_path_allowed

        for name in ("main.py", "models.py", "views.py"):
            _touch(tmp_path / "app" / name)
        _touch(tmp_path / "pyproject.toml")

        allowlist = _build_patcher_allowlist(str(tmp_path))
        assert allowlist is not None
        assert is_path_allowed("app/handlers.py", str(tmp_path), allowlist)
        assert is_path_allowed("tests/test_handlers.py", str(tmp_path), allowlist)
        assert not is_path_allowed("scratch/notes.py", str(tmp_path), allowlist)
