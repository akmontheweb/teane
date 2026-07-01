"""Regression tests for the repair-loop robustness pass triggered by
session 7c30bce2:

  - Tier 1A: ``_compose_node_build_command`` installs subdir deps when
    root delegates via ``cd X && npm run build``.
  - Tier 1B: ``_diagnostics_look_like_install_failure`` flags missing-
    module dominated diagnostics so the reflection prompt can bias the
    judge toward an install/build fix.
  - Tier 2A: REPLACE_BLOCK matcher tolerates single ↔ double quote drift.
  - Tier 2B: REPLACE_BLOCK matcher tolerates leading-whitespace drift
    (tabs vs spaces).
  - Tier 2C: idempotency no-op detection survives whitespace drift in
    the LLM's re-emitted ``replace`` text.
  - Tier 3A: judge-ignored gate treats an attempted-but-failed patch on
    a judge-named file as compliance, not distraction.
  - Tier 3B: no_progress accounting does not tick on the first round
    that produces real diagnostics after an empty prior reading.
  - Tier 3C: patcher rollup log carries a per-file reason classifier.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from harness.cli import (
    _compose_node_build_command,
    _extract_delegate_subdirs,
)
from harness.graph import (
    _diagnostics_look_like_install_failure,
    _patches_touched_judge_files,
)
from harness.patcher import (
    TextPatcher,
    _classify_patch_failure,
    _quote_normalize,
    _whitespace_tolerant_match,
)


# ---------------------------------------------------------------------------
# Tier 1A — _compose_node_build_command handles root-delegating-to-subdir
# ---------------------------------------------------------------------------


def _write_pkg(path: Path, data: dict) -> str:
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_compose_node_build_command_installs_delegated_subdirs(tmp_path):
    """When root scripts.build is ``cd client && npm run build`` and
    ``client/package.json`` exists, the composed command MUST run
    ``cd client && npm install`` before the root install. Without this,
    the subdir's deps never materialise and the build fails with 100+
    missing-module errors (session 7c30bce2 root cause)."""
    (tmp_path / "client").mkdir()
    _write_pkg(tmp_path / "client" / "package.json", {"name": "client"})
    root_pkg = _write_pkg(tmp_path / "package.json", {
        "name": "root",
        "scripts": {
            "build": "cd client && npm run build",
            "dev": "concurrently \"cd client && npm run dev\"",
        },
        "devDependencies": {"concurrently": "^8.0.0"},
    })
    cmd = _compose_node_build_command(root_pkg)
    assert "(cd client && npm install)" in cmd
    # Subdir install runs FIRST so the root build (which cds in) finds
    # node_modules populated.
    assert cmd.index("(cd client && npm install)") < cmd.index("npm install &&")


def test_compose_node_build_command_skips_workspaces(tmp_path):
    """When the root package.json declares ``workspaces``, npm itself
    installs subdir deps — we must NOT emit redundant ``cd X && npm
    install`` steps."""
    (tmp_path / "packages" / "app").mkdir(parents=True)
    _write_pkg(tmp_path / "packages" / "app" / "package.json", {"name": "app"})
    root_pkg = _write_pkg(tmp_path / "package.json", {
        "name": "root",
        "workspaces": ["packages/*"],
        "scripts": {"build": "cd packages/app && npm run build"},
    })
    cmd = _compose_node_build_command(root_pkg)
    assert "(cd packages/app && npm install)" not in cmd


def test_compose_node_build_command_skips_missing_subdir(tmp_path):
    """A stale ``cd <name>`` reference in a script with no matching
    package.json on disk must NOT inject a doomed ``cd``. Without this
    guard the build dies with ``No such file or directory`` and traps
    the repair loop on a non-bug."""
    root_pkg = _write_pkg(tmp_path / "package.json", {
        "name": "root",
        "scripts": {"build": "cd ghost && npm run build"},
    })
    cmd = _compose_node_build_command(root_pkg)
    assert "cd ghost" not in cmd


def test_extract_delegate_subdirs_rejects_unsafe_names():
    """Subdir names containing bash metacharacters must be dropped so
    we never inject a shell-injection primitive into the build cmd."""
    scripts = {
        "build": "cd ; rm -rf / && npm run build",  # semicolon
        "test": "cd $EVIL && npm test",             # variable
    }
    assert _extract_delegate_subdirs(scripts) == []


def test_extract_delegate_subdirs_dedups_and_orders():
    scripts = {
        "build": "cd client && npm run build",
        "dev": "cd client && npm run dev",
        "test": "cd server && npm test",
    }
    out = _extract_delegate_subdirs(scripts)
    assert out == ["client", "server"]


# ---------------------------------------------------------------------------
# Tier 1B — install-failure heuristic flags the right pattern
# ---------------------------------------------------------------------------


def test_install_failure_heuristic_flags_ts2307_dominated_set():
    """The session 7c30bce2 pattern: a wall of TS2307 missing-module
    errors → heuristic returns True so the judge prompt gets the hint."""
    diags = [
        {"error_code": "TS2307", "message": "Cannot find module 'react'"},
        {"error_code": "TS2307", "message": "Cannot find module 'axios'"},
        {"error_code": "TS2307", "message": "Cannot find module 'vitest'"},
        {"error_code": "TS17004", "message": "Cannot use JSX without a flag"},
    ]
    assert _diagnostics_look_like_install_failure(diags) is True


def test_install_failure_heuristic_ignores_pure_code_errors():
    """A failing-set of real type errors → False; we don't want the
    install-failure hint contaminating a legitimate code repair."""
    diags = [
        {"error_code": "TS2322", "message": "Type 'string' is not assignable"},
        {"error_code": "TS2345", "message": "Argument of type 'number'"},
        {"error_code": "TS2554", "message": "Expected 2 arguments, got 1"},
    ]
    assert _diagnostics_look_like_install_failure(diags) is False


def test_install_failure_heuristic_handles_python_imports():
    """Python ModuleNotFoundError / ImportError patterns also flag."""
    diags = [
        {"error_code": "ModuleNotFoundError", "message": "No module named 'fastapi'"},
        {"error_code": "ImportError", "message": "cannot import name 'X'"},
    ]
    assert _diagnostics_look_like_install_failure(diags) is True


def test_install_failure_heuristic_empty_input():
    assert _diagnostics_look_like_install_failure([]) is False


# ---------------------------------------------------------------------------
# Tier 2A — quote-style tolerance
# ---------------------------------------------------------------------------


def test_quote_normalize_collapses_both_quote_chars():
    """The normalizer reduces both kinds of quote to the same char so a
    rstrip-tolerant comparison can match across quote-style drift."""
    a = _quote_normalize("import X from 'lib';")
    b = _quote_normalize('import X from "lib";')
    assert a == b


def test_whitespace_tolerant_match_with_quote_normalize_finds_single_match():
    """Search uses single quotes; file uses double quotes — the
    quote-aware normalizer yields a single byte-offset match."""
    original = 'import X from "lib";\n'
    search = "import X from 'lib';"
    matches = _whitespace_tolerant_match(
        original, search, normalize=_quote_normalize,
    )
    assert len(matches) == 1
    assert matches[0] == 0


# ---------------------------------------------------------------------------
# Tier 2B — leading-whitespace tolerance
# ---------------------------------------------------------------------------


def test_whitespace_tolerant_match_default_rejects_indent_drift():
    """The default rstrip-only normaliser must NOT match across leading
    tab-vs-space drift — that would be unsafe (e.g. matching code at the
    wrong block nesting level)."""
    original = "build:\n\trm -rf out\n"  # tab-indented
    search = "build:\n    rm -rf out\n"  # space-indented
    assert _whitespace_tolerant_match(original, search) == []


def test_whitespace_tolerant_match_full_strip_handles_indent_drift():
    """With ``normalize=str.strip`` the matcher tolerates leading
    whitespace drift — Tier 2B's indent-tolerant fallback in
    replace_block uses this for Makefile / Python / YAML."""
    original = "build:\n\trm -rf out\n"
    search = "build:\n    rm -rf out\n"
    matches = _whitespace_tolerant_match(original, search, normalize=str.strip)
    assert len(matches) == 1


# ---------------------------------------------------------------------------
# Tier 2C — idempotency no-op detection survives whitespace drift
# ---------------------------------------------------------------------------


def test_replace_block_no_op_detected_under_whitespace_drift(tmp_path):
    """LLM re-emits a patch whose ``replace`` text matches the file's
    current content modulo whitespace — must be classified as no-op so
    the repair loop doesn't trip on a non-bug."""
    target = tmp_path / "Makefile"
    target.write_text("build:\n\trm -rf out\n", encoding="utf-8")  # tabs
    engine = TextPatcher(str(tmp_path))
    # LLM emits a "replace" that the file already matches, with spaces
    # where the file has tabs. Search picks the (no-longer-present)
    # pre-patch shape so exact-byte search misses.
    result = asyncio.run(engine.replace_block(
        "Makefile",
        search="build:\n\techo hi\n",           # pre-patch shape, gone
        replace="build:\n    rm -rf out\n",     # already at target (spaces)
    ))
    assert result.success is True
    assert getattr(result, "no_op", False) is True


# ---------------------------------------------------------------------------
# Tier 3A — judge-ignored gate considers attempted patches
# ---------------------------------------------------------------------------


class _StubResult:
    """Minimal PatchResult shape for the gate's getattr inspection."""
    def __init__(self, file, success, no_op=False):
        self.file = file
        self.success = success
        self.no_op = no_op


def test_patches_touched_judge_files_success_only_baseline():
    """Default (success-only) mode: a SUCCESS on the judge-named file
    returns True; a FAILED attempt does not."""
    results = [_StubResult("client/src/App.tsx", success=False)]
    assert _patches_touched_judge_files(results, ["src/App.tsx"]) is False


def test_patches_touched_judge_files_include_attempts_counts_failures():
    """include_attempts mode (Tier 3A): a FAILED attempt on the judge-
    named file ALSO counts as touched — the LLM tried to comply but
    the REPLACE_BLOCK search missed for mechanical reasons."""
    results = [_StubResult("client/src/App.tsx", success=False)]
    assert _patches_touched_judge_files(
        results, ["src/App.tsx"], include_attempts=True,
    ) is True


def test_patches_touched_judge_files_no_ops_excluded_in_attempt_mode():
    """no-op results are neither successes nor failed attempts; they
    must not count as touched even in include_attempts mode."""
    results = [_StubResult("client/src/App.tsx", success=True, no_op=True)]
    assert _patches_touched_judge_files(
        results, ["src/App.tsx"], include_attempts=True,
    ) is False


# ---------------------------------------------------------------------------
# Tier 3C — patcher failure-reason classifier
# ---------------------------------------------------------------------------


def test_classify_patch_failure_recognises_search_miss():
    err = "Search block not found in foo.py. Your search did not match..."
    assert _classify_patch_failure(err) == "search miss"


def test_classify_patch_failure_recognises_allowlist():
    err = "Patch to .env rejected: not in skill allowlist."
    assert _classify_patch_failure(err) == "allowlist denied"


def test_classify_patch_failure_recognises_ambiguous():
    err = "Search block matched 3 regions in Makefile under whitespace-tolerant comparison."
    assert _classify_patch_failure(err) == "ambiguous match"


def test_classify_patch_failure_recognises_missing_file():
    err = "File not found: foo.py. Use CREATE_FILE for new files."
    assert _classify_patch_failure(err) == "file missing"


def test_classify_patch_failure_falls_back_for_unknown():
    assert _classify_patch_failure("disk full") == "error"
    assert _classify_patch_failure("") == "unknown"
