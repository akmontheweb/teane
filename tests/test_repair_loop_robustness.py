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

import pytest

import asyncio
import json
from pathlib import Path

from harness.cli import (
    _compose_node_build_command,
    _extract_delegate_subdirs,
)
from harness.graph import (
    _RE_WS_SOURCE_PATH,
    _detect_state_reverts,
    _workspace_imports_of,
    _build_repair_reflection_prompt,
    _diagnostics_look_like_install_failure,
    _missing_module_matches_workspace_source,
    _parse_repair_reflection_verdict,
    _patches_touched_judge_files,
    _reflection_verdict_is_low_signal,
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


def test_install_failure_heuristic_flags_pytest_asyncio_missing():
    """Session 116667f5 pattern: pytest collector raises
    ``AssertionError: Failed: async def functions are not natively
    supported.`` for every async test when pytest-asyncio is missing
    from ``requirements.txt`` or ``asyncio_mode`` isn't set. The error
    code is a bare ``AssertionError`` — none of the code prefixes
    match — so only the message-fragment path can flip the heuristic
    and point the judge at the manifest instead of the test source."""
    diags = [
        {
            "error_code": "AssertionError",
            "message": (
                "Failed: async def functions are not natively supported."
            ),
        },
        {
            "error_code": "AssertionError",
            "message": (
                "Failed: async def functions are not natively supported."
            ),
        },
        {"error_code": "OperationalError", "message": "OperationalError"},
        {
            "error_code": "RuntimeError",
            "message": "RuntimeError: Database connection failed",
        },
    ]
    # Half the set (2/4) matches — the majority threshold `matched * 2 >=
    # len(diagnostics)` is satisfied (2*2 == 4), so the hint fires.
    assert _diagnostics_look_like_install_failure(diags) is True


# ---------------------------------------------------------------------------
# Tier 1D — reflection-prompt escape-hatch gates on file:line availability
# and the parser strips any leftover ``<file>`` placeholder. Session
# 116667f5: repair #2 resolved the only prior error (SyntaxError) and
# introduced 4 new AssertionError/OperationalError/RuntimeError entries
# with no file:line. ``top_persisted_diagnostics`` was empty; the judge
# reasoned into the "insufficient data" branch but had no <file> to
# substitute, so it shipped the literal placeholder back. The verdict
# then contaminated the next repair round's prompt and logs.
# ---------------------------------------------------------------------------


def test_reflection_prompt_omits_file_placeholder_when_no_locations():
    """Empty ``top_persisted_diagnostics`` → the fallback rule MUST tell
    the LLM to emit the plain ``no diagnostic locations available``
    string and MUST NOT tell it to emit the ``<file>`` template. Without
    this branch the LLM parrots the template placeholder verbatim (the
    session-116667f5 failure)."""
    prompt = _build_repair_reflection_prompt(
        prior_diagnostics_count=1,
        current_diagnostics_count=4,
        resolved_fingerprints=["SyntaxError::invalid syntax"],
        persisted_fingerprints=[],
        new_fingerprints=[
            "AssertionError::async def functions are not natively supported",
            "OperationalError::OperationalError",
        ],
        top_persisted_diagnostics=[],  # this is the trigger condition
        install_failure_likely=False,
        path_wiring_module=None,
    )
    # Plain fallback is prescribed …
    assert "no diagnostic locations available" in prompt
    # … and the LLM is explicitly told NOT to emit the placeholder.
    assert "do not emit the placeholder" in prompt.lower()
    # The old placeholder template must NOT appear as an instruction — a
    # readback of the placeholder in the fallback rule would defeat the
    # gate. The only allowed mention of ``<file>`` in this branch of the
    # prompt is the negative one above ("do NOT emit").
    fallback_section = prompt.split("No diagnostic in the failing set", 1)[1]
    assert "investigate <file>'s data flow" not in fallback_section


def test_reflection_prompt_keeps_file_placeholder_when_locations_exist():
    """Symmetric to the above: when at least one persistent diagnostic
    carries a real file:line, the ``<file>``-substitution branch stays
    active — but the LLM is now told explicitly to substitute an actual
    path (belt-and-suspenders for the parser guard)."""
    prompt = _build_repair_reflection_prompt(
        prior_diagnostics_count=2,
        current_diagnostics_count=2,
        resolved_fingerprints=[],
        persisted_fingerprints=["AssertionError::assert x == y"],
        new_fingerprints=[],
        top_persisted_diagnostics=[{
            "error_code": "AssertionError",
            "message": "assert 3 == 2",
            "file": "tests/test_math.py", "line": 42,
        }],
        install_failure_likely=False,
        path_wiring_module=None,
    )
    # The <file>-substitution branch is the one we render.
    assert "investigate <file>'s data flow" in prompt
    # And the substitute-with-a-real-path safeguard is present.
    assert "REPLACED by an actual path" in prompt
    # The no-locations branch's plain fallback string must NOT appear.
    assert "no diagnostic locations available" not in prompt


def test_reflection_prompt_mentions_async_def_in_install_exception():
    """The grounding-rules EXCEPTION list must name the pytest-asyncio
    signature so the LLM can classify it as manifest-class even when
    the outer install-failure hint isn't fired (e.g. a mixed set with
    only one such error)."""
    prompt = _build_repair_reflection_prompt(
        prior_diagnostics_count=1,
        current_diagnostics_count=1,
        resolved_fingerprints=[],
        persisted_fingerprints=["AssertionError::async def"],
        new_fingerprints=[],
        top_persisted_diagnostics=[{
            "error_code": "AssertionError",
            "message": "async def functions are not natively supported",
            "file": "tests/test_a.py", "line": 10,
        }],
        install_failure_likely=False,
        path_wiring_module=None,
    )
    assert "async def functions are not natively supported" in prompt
    assert "pytest-asyncio" in prompt


def test_parse_reflection_verdict_strips_leftover_file_placeholder():
    """Defence-in-depth for the prompt gate. Even if a future LLM ships
    the literal ``<file>`` template back — because it inherited an old
    prompt cache prefix, or because it ignored the rule — the parser
    MUST rewrite ``real_blocker`` to the plain fallback so no downstream
    consumer (system-message injection, logs, events) sees a template
    placeholder."""
    raw = json.dumps({
        "verdict": "PROGRESS",
        "real_blocker": (
            "insufficient data — investigate <file>'s data flow into "
            "the assertion"
        ),
        # Actionable filler: the placeholder guard is what this test is
        # about, and an edit-shaped recommendation keeps it isolated from
        # the separate actionability guard (see
        # test_reflection_judge_grounding.py), which would blank an
        # "Investigate …" recommendation before we could assert on it.
        "recommendation": "Edit src/math_utils.py:42 to return a float.",
    })
    v = _parse_repair_reflection_verdict(raw)
    assert v is not None
    assert v["verdict"] == "PROGRESS"
    assert "<file>" not in v["real_blocker"]
    assert v["real_blocker"] == (
        "insufficient data — no diagnostic locations available"
    )
    # An actionable recommendation is left alone — the placeholder guard
    # only touches real_blocker.
    assert v["recommendation"] == "Edit src/math_utils.py:42 to return a float."


def test_parse_reflection_verdict_strips_no_location_marker():
    """The harness renders diagnostics without a file field as the
    literal ``<no location>`` marker. If the LLM copies that verbatim
    into ``real_blocker`` (session cf3fcd27-style behaviour), the same
    guard rewrites it — a marker is just as unhelpful as ``<file>``."""
    raw = json.dumps({
        "verdict": "REGRESSION",
        "real_blocker": "The assertion at <no location> is broken",
        "recommendation": "Read the traceback.",
    })
    v = _parse_repair_reflection_verdict(raw)
    assert v is not None
    assert v["verdict"] == "REGRESSION"
    assert v["real_blocker"] == (
        "insufficient data — no diagnostic locations available"
    )


def test_parse_reflection_verdict_leaves_substituted_blocker_alone():
    """A well-formed verdict where the LLM correctly substituted a real
    path MUST pass through untouched — the guard fires only on the
    unresolved placeholder, not on any string mentioning ``file``."""
    raw = json.dumps({
        "verdict": "PROGRESS",
        "real_blocker": (
            "insufficient data — investigate tests/test_math.py's data "
            "flow into the assertion"
        ),
        "recommendation": "Read tests/test_math.py:42.",
    })
    v = _parse_repair_reflection_verdict(raw)
    assert v is not None
    assert "tests/test_math.py" in v["real_blocker"]
    assert "<file>" not in v["real_blocker"]
    # No rewrite happened — the original sentence is preserved verbatim.
    assert v["real_blocker"].startswith("insufficient data — investigate tests/")


# ---------------------------------------------------------------------------
# Tier 1C — path-wiring vs install-failure disambiguation. Session
# 3193a24f: pytest ran with CWD=`server/` (subdir detector's fault) and
# `server/tests/conftest.py` imported `from server.app...`. Pytest
# reported `ModuleNotFoundError: No module named 'server'` and returned
# exit 4 (usage error, because conftest failed at rootdir discovery).
# The install-failure heuristic matched on shape and the judge steered
# repair toward adding `server` to requirements.txt — nonsensical, since
# `server/` was a scaffolded source directory. Three rounds burned.
# ---------------------------------------------------------------------------


def test_missing_module_returns_workspace_source_dir(tmp_path):
    """When the missing name IS a workspace source dir, return it — the
    caller uses this to override the install-failure hint."""
    (tmp_path / "server").mkdir()
    diags = [{
        "error_code": "ModuleNotFoundError",
        "message": "ModuleNotFoundError: No module named 'server'",
    }]
    assert _missing_module_matches_workspace_source(diags, str(tmp_path)) == "server"


def test_missing_module_handles_dotted_paths(tmp_path):
    """`No module named 'server.app'` also resolves to the top-level
    package — sys.path resolution keys on the leading segment."""
    (tmp_path / "server").mkdir()
    diags = [{
        "error_code": "ModuleNotFoundError",
        "message": "No module named 'server.app.config'",
    }]
    assert _missing_module_matches_workspace_source(diags, str(tmp_path)) == "server"


def test_missing_module_returns_none_when_not_workspace_source(tmp_path):
    """`No module named 'fastapi'` — fastapi is not a workspace dir, so
    the classic install-failure path stays in charge."""
    diags = [{
        "error_code": "ModuleNotFoundError",
        "message": "No module named 'fastapi'",
    }]
    assert _missing_module_matches_workspace_source(diags, str(tmp_path)) is None


def test_missing_module_returns_none_for_pure_code_errors(tmp_path):
    """Non-missing-module diagnostics never trip this heuristic."""
    (tmp_path / "server").mkdir()
    diags = [
        {"error_code": "AssertionError", "message": "assert 3 == 2"},
        {"error_code": "TypeError", "message": "unexpected argument"},
    ]
    assert _missing_module_matches_workspace_source(diags, str(tmp_path)) is None


def test_missing_module_handles_ts2307(tmp_path):
    """TS2307's ``Cannot find module 'client'`` form also matches when
    the name is a workspace source dir (monorepo React case)."""
    (tmp_path / "client").mkdir()
    diags = [{
        "error_code": "TS2307",
        "message": "Cannot find module 'client'",
    }]
    assert _missing_module_matches_workspace_source(diags, str(tmp_path)) == "client"


def test_reflection_prompt_emits_wiring_hint_and_suppresses_install_hint():
    """When path_wiring_module is set, the prompt MUST carry the
    wiring-hint block and MUST NOT carry the install-failure block —
    even if the caller also passed install_failure_likely=True. Without
    this override the judge sees both hints and still steers toward the
    manifest (session 3193a24f actual behaviour)."""
    prompt = _build_repair_reflection_prompt(
        prior_diagnostics_count=1,
        current_diagnostics_count=1,
        resolved_fingerprints=[],
        persisted_fingerprints=["ModuleNotFoundError::server"],
        new_fingerprints=[],
        top_persisted_diagnostics=[{
            "error_code": "ModuleNotFoundError",
            "message": "No module named 'server'",
            "file": "tests/conftest.py", "line": 16,
        }],
        install_failure_likely=True,
        path_wiring_module="server",
    )
    assert "PATH/WIRING FAILURE HINT" in prompt
    assert "``server``" in prompt
    # Install-failure hint MUST be absent — it would send the LLM to
    # requirements.txt for a source-dir name.
    assert "ENVIRONMENT-FAILURE HINT" not in prompt
    # And the prompt must tell the judge NOT to name the manifest.
    assert "DO NOT recommend editing requirements.txt" in prompt


def test_reflection_prompt_falls_back_to_install_hint_when_no_wiring():
    """path_wiring_module=None → the existing install-failure hint is
    the only environment hint — proves the override is scoped correctly
    and doesn't accidentally suppress the install-failure path."""
    prompt = _build_repair_reflection_prompt(
        prior_diagnostics_count=1,
        current_diagnostics_count=1,
        resolved_fingerprints=[],
        persisted_fingerprints=["ModuleNotFoundError::fastapi"],
        new_fingerprints=[],
        top_persisted_diagnostics=[{
            "error_code": "ModuleNotFoundError",
            "message": "No module named 'fastapi'",
            "file": "server/app/main.py", "line": 1,
        }],
        install_failure_likely=True,
        path_wiring_module=None,
    )
    assert "ENVIRONMENT-FAILURE HINT" in prompt
    assert "PATH/WIRING FAILURE HINT" not in prompt


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


def test_patches_touched_judge_files_counts_no_ops_in_attempt_mode():
    """A no-op on a judge-named file IS engagement with the target.

    Reversed 2026-09-14. The previous contract ("neither successes nor
    failed attempts") conflated "no change resulted" with "no attempt
    made", and the difference matters: an identity REPLACE_BLOCK returns
    ``success=False, no_op=True`` and means the model looked at the
    judge's file and found nothing to change.

    lumina-fresh-20260911-1107: forced onto ``birthday_service.py`` by the
    Fix G MUST-MODIFY promotion, the model emitted search == replace four
    rounds running because ``list_upcoming()`` was in fact correct — the
    real defect was a test posting future-dated births that the app's own
    validator rejects. Each of those rounds incremented
    ``judge_ignored_streak`` and escalated the banner to push harder at
    the same correct file. The model was right and was penalised for it.
    """
    for success in (True, False):
        results = [
            _StubResult("client/src/App.tsx", success=success, no_op=True),
        ]
        assert _patches_touched_judge_files(
            results, ["src/App.tsx"], include_attempts=True,
        ) is True


def test_patches_touched_judge_files_still_excludes_no_ops_by_default():
    """Default mode is progress accounting, not engagement accounting —
    a no-op changed nothing and must not read as a landed patch."""
    results = [_StubResult("client/src/App.tsx", success=True, no_op=True)]
    assert _patches_touched_judge_files(
        results, ["src/App.tsx"],
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


# ---------------------------------------------------------------------------
# Fix D — low-signal escalation. Session 116667f5 spun for six repair
# rounds because pytest's bare AssertionError diagnostics gave the judge
# nothing to localize; it kept returning the ``insufficient data`` sentinel
# and the loop treated that as "hold the counter steady" — silently
# spinning until the total_repairs ceiling. The escalation gates on a
# NEW counter (``consecutive_low_signal_rounds``) so a streak of low-
# signal verdicts widens the repair prompt with the raw build-output
# tail without polluting the distraction circuit-breaker.
# ---------------------------------------------------------------------------


def test_reflection_verdict_low_signal_matches_sentinel():
    """The insufficient-data sentence in ``real_blocker`` must be
    recognised regardless of which fallback form the judge picked
    (per-file substitution OR the no-locations plain string)."""
    assert _reflection_verdict_is_low_signal({
        "verdict": "DISTRACTION",
        "real_blocker": (
            "insufficient data — investigate tests/test_db_base.py's "
            "data flow into the assertion"
        ),
        "recommendation": "Focus on the AssertionError.",
    })
    assert _reflection_verdict_is_low_signal({
        "verdict": "REGRESSION",
        "real_blocker": "insufficient data — no diagnostic locations available",
        "recommendation": "Look at the raw build output.",
    })
    # Leading whitespace and mixed case must still trip the check —
    # LLM outputs sometimes drift and the sentinel check is our only
    # signal that the verdict is content-free.
    assert _reflection_verdict_is_low_signal({
        "verdict": "DISTRACTION",
        "real_blocker": "  Insufficient Data — nothing to localise",
        "recommendation": "",
    })


def test_reflection_verdict_low_signal_rejects_grounded_blocker():
    """A grounded blocker that names a real file:line must NOT be
    treated as low-signal — that would silence a legitimately actionable
    verdict and let the distraction counter drift forever."""
    assert not _reflection_verdict_is_low_signal({
        "verdict": "DISTRACTION",
        "real_blocker": (
            "tests/test_math.py:42 asserts 3 == 2 because reduce() "
            "adds instead of multiplying"
        ),
        "recommendation": "Change '+' to '*' in reduce().",
    })
    assert not _reflection_verdict_is_low_signal({
        "verdict": "PROGRESS",
        "real_blocker": "",
        "recommendation": "",
    })


def test_low_signal_counter_included_in_resume_gate_keys():
    """The resume path zeros the counters that gate the repair loop so
    a resumed session doesn't immediately re-trip HITL. The new
    ``consecutive_low_signal_rounds`` counter must join that reset list
    or resume-then-fail-again would come back with the escalation
    already primed — attaching the raw tail on iteration 1 before the
    judge has weighed in on the fresh diagnostics."""
    import inspect

    from harness.graph import _reset_stale_gate_counters_on_resume
    src = inspect.getsource(_reset_stale_gate_counters_on_resume)
    assert "consecutive_low_signal_rounds" in src, (
        "resume-reset must zero consecutive_low_signal_rounds so a "
        "resumed session gets a fresh escalation budget"
    )


# ---------------------------------------------------------------------------
# Tier 2D — indent-tolerant tier re-anchors the REPLACEMENT indentation
# ---------------------------------------------------------------------------


def test_indent_tolerant_replace_reanchors_python_indentation(tmp_path):
    """When the LLM's whole block is uniformly outdented (method emitted
    at column 0 while the file has it at column 4), the strip tier
    matches — but the replacement must land at the FILE's indentation,
    not the LLM's, or the patch corrupts Python."""
    target = tmp_path / "svc.py"
    target.write_text(
        "class Svc:\n"
        "    def score(self):\n"
        "        return 1\n",
        encoding="utf-8",
    )
    engine = TextPatcher(str(tmp_path))
    result = asyncio.run(engine.replace_block(
        "svc.py",
        search="def score(self):\n    return 1\n",      # outdented by 4
        replace="def score(self):\n    return 2\n",     # same drift
    ))
    assert result.success is True
    body = target.read_text(encoding="utf-8")
    assert "    def score(self):\n        return 2\n" in body, (
        f"replacement landed at the wrong indentation:\n{body}"
    )


def test_indent_reanchor_helper_handles_strip_direction(tmp_path):
    """Delta in the other direction: the LLM over-indented relative to
    the file — the common prefix is removed from the replacement."""
    from harness.patcher import _reindent_replace_for_match
    original = "def f():\n    return 1\n"
    search = "    def f():\n        return 1\n"      # over-indented by 4
    replace = "    def f():\n        return 2\n"
    fixed = _reindent_replace_for_match(original, search, replace, 0)
    assert fixed == "def f():\n    return 2\n"


def test_indent_reanchor_falls_back_verbatim_on_ragged_delta():
    from harness.patcher import _reindent_replace_for_match
    original = "  a\n      b\n"
    search = "a\n  b\n"  # deltas differ per line (2 vs 4) — inconsistent
    replace = "c\n"
    assert _reindent_replace_for_match(original, search, replace, 0) == "c\n"


# ---------------------------------------------------------------------------
# Tier 2E — elided-middle ("...") matching
# ---------------------------------------------------------------------------


def test_elided_middle_replaces_anchors_and_preserves_middle(tmp_path):
    target = tmp_path / "mod.py"
    target.write_text(
        "def top():\n"
        "    return 'old-top'\n"
        "\n"
        "def middle():\n"
        "    return 'untouched'\n"
        "\n"
        "def bottom():\n"
        "    return 'old-bottom'\n",
        encoding="utf-8",
    )
    engine = TextPatcher(str(tmp_path))
    result = asyncio.run(engine.replace_block(
        "mod.py",
        search=(
            "def top():\n"
            "    return 'old-top'\n"
            "...\n"
            "def bottom():\n"
            "    return 'old-bottom'\n"
        ),
        replace=(
            "def top():\n"
            "    return 'new-top'\n"
            "...\n"
            "def bottom():\n"
            "    return 'new-bottom'\n"
        ),
    ))
    assert result.success is True
    body = target.read_text(encoding="utf-8")
    assert "new-top" in body and "new-bottom" in body
    assert "return 'untouched'" in body, "elided middle must be preserved"
    assert "old-top" not in body and "old-bottom" not in body


def test_elided_middle_refuses_ambiguous_anchor():
    from harness.patcher import _elided_match_replace
    original = "x = 1\ny = 2\nx = 1\nz = 3\n"
    # First anchor "x = 1" appears twice — must refuse, not guess.
    assert _elided_match_replace(
        original, "x = 1\n...\nz = 3\n", "x = 9\n...\nz = 9\n",
    ) is None


def test_elided_middle_requires_matching_marker_counts():
    from harness.patcher import _elided_match_replace
    original = "a\nb\nc\n"
    # Search has one marker, replace has none — shape mismatch.
    assert _elided_match_replace(original, "a\n...\nc\n", "a2\nc2\n") is None


def test_elided_middle_ignored_when_no_marker():
    from harness.patcher import _elided_match_replace
    assert _elided_match_replace("a\nb\n", "a\nb\n", "a\nc\n") is None


# ---------------------------------------------------------------------------
# Revert / oscillation detection (lumina-run3-20260914-1330).
#
# Over six repair rounds the judge alternated between `default=1` and
# `default=lambda: 1` on one SQLAlchemy column, and the model applied every
# reversal:
#
#     0028  1 -> lambda: 1      0035  lambda: 1 -> 1
#     0031  lambda: 1 -> 1      0037  1 -> lambda: 1
#     0033  1 -> lambda: 1      0039  lambda: 1 -> 1
#
# Seven edits, net effect zero, twelve consecutive DISTRACTION verdicts, and
# the run died on the auto-resume cap. Both spellings are equivalent —
# SQLAlchemy applies `default=` at flush time either way — so neither could
# satisfy an assertion on a freshly constructed instance.
#
# Every existing counter read this as progress, because each patch applied
# cleanly. This is the only signal that sees the cycle.
# ---------------------------------------------------------------------------


def _write(ws, rel, text):
    import os
    p = os.path.join(ws, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)


class TestRevertDetection:

    def test_a_b_a_cycle_is_flagged(self, tmp_path):
        ws, rel, lc = str(tmp_path), "app/models/birthday.py", {}
        _write(ws, rel, "default=1")
        assert _detect_state_reverts(ws, [rel], lc) == []      # A, first sight
        _write(ws, rel, "default=lambda: 1")
        assert _detect_state_reverts(ws, [rel], lc) == []      # B, new state
        _write(ws, rel, "default=1")
        assert _detect_state_reverts(ws, [rel], lc) == [rel]   # A again

    def test_forward_progress_is_never_flagged(self, tmp_path):
        """Three genuinely different states in a row are convergence, not a
        cycle — the detector must stay silent."""
        ws, rel, lc = str(tmp_path), "app/m.py", {}
        for text in ("v1", "v2", "v3", "v4"):
            _write(ws, rel, text)
            assert _detect_state_reverts(ws, [rel], lc) == []

    def test_repeated_observation_of_a_stable_file_is_never_a_revert(
        self, tmp_path,
    ):
        """``modified_files`` is CUMULATIVE across the session, so every
        repair round re-hashes every file ever touched. Recording each
        sighting made a stable file's history grow [A, A, A...] and it then
        matched its own earlier copy.

        lumina-run4-20260914-1608 flagged 17 files per round on this —
        .gitignore and Makefile among them — which would have drowned the
        real signal and told the judge it was cycling when it was not.
        Recording transitions rather than observations is the fix.
        """
        ws, rel, lc = str(tmp_path), "app/m.py", {}
        _write(ws, rel, "stable")
        for _ in range(8):
            assert _detect_state_reverts(ws, [rel], lc) == []
        # One state recorded, not eight.
        assert lc["file_state_history"][rel] == lc["file_state_history"][rel][:1]

    def test_a_revert_still_fires_after_many_stable_observations(
        self, tmp_path,
    ):
        """Suppressing repeats must not suppress the real signal."""
        ws, rel, lc = str(tmp_path), "app/m.py", {}
        _write(ws, rel, "A")
        for _ in range(5):
            _detect_state_reverts(ws, [rel], lc)
        _write(ws, rel, "B")
        for _ in range(5):
            _detect_state_reverts(ws, [rel], lc)
        _write(ws, rel, "A")
        assert _detect_state_reverts(ws, [rel], lc) == [rel]

    def test_unchanged_content_is_not_a_revert(self, tmp_path):
        """Writing the same bytes again is a no-op, which the patcher
        already reports. Double-counting it here would fire on every
        idempotent re-emission."""
        ws, rel, lc = str(tmp_path), "app/m.py", {}
        _write(ws, rel, "same")
        _detect_state_reverts(ws, [rel], lc)
        assert _detect_state_reverts(ws, [rel], lc) == []

    def test_the_lumina_six_round_oscillation(self, tmp_path):
        """Replay the real sequence and count how early it is caught."""
        ws, rel, lc = str(tmp_path), "server/app/models/birthday.py", {}
        seq = ["default=1", "default=lambda: 1", "default=1",
               "default=lambda: 1", "default=1", "default=lambda: 1",
               "default=1"]
        first_catch = None
        for i, text in enumerate(seq):
            _write(ws, rel, text)
            if _detect_state_reverts(ws, [rel], lc) and first_catch is None:
                first_catch = i
        # Caught on the third write — the first return to a prior state —
        # not after seven.
        assert first_catch == 2

    def test_per_file_isolation(self, tmp_path):
        ws, lc = str(tmp_path), {}
        _write(ws, "a.py", "x")
        _write(ws, "b.py", "y")
        _detect_state_reverts(ws, ["a.py", "b.py"], lc)
        _write(ws, "a.py", "z")
        _write(ws, "b.py", "y2")
        _detect_state_reverts(ws, ["a.py", "b.py"], lc)
        _write(ws, "a.py", "x")          # a reverts, b does not
        _write(ws, "b.py", "y3")
        assert _detect_state_reverts(ws, ["a.py", "b.py"], lc) == ["a.py"]

    def test_history_is_bounded(self, tmp_path):
        ws, rel, lc = str(tmp_path), "app/m.py", {}
        for i in range(40):
            _write(ws, rel, f"v{i}")
            _detect_state_reverts(ws, [rel], lc, keep=5)
        assert len(lc["file_state_history"][rel]) <= 5

    def test_missing_file_is_skipped(self, tmp_path):
        assert _detect_state_reverts(str(tmp_path), ["gone.py"], {}) == []

    def test_empty_input_is_safe(self, tmp_path):
        assert _detect_state_reverts(str(tmp_path), [], {}) == []
        assert _detect_state_reverts(str(tmp_path), ["", None], {}) == []


class TestRevertBlockRendering:

    def _kwargs(self):
        return {
            "prior_diagnostics_count": 3,
            "current_diagnostics_count": 3,
            "resolved_fingerprints": [],
            "persisted_fingerprints": ["a", "b", "c"],
            "new_fingerprints": [],
            "top_persisted_diagnostics": [
                {"error_code": "AssertionError",
                 "file": "server/tests/test_birthday_model.py",
                 "line": 37, "message": "assert None == 1"},
            ],
        }

    def test_absent_by_default(self):
        from harness.graph import _build_repair_reflection_prompt
        assert "YOUR GUIDANCE IS CYCLING" not in \
            _build_repair_reflection_prompt(**self._kwargs())

    def test_single_revert_is_noise(self):
        from harness.graph import _build_repair_reflection_prompt
        assert "YOUR GUIDANCE IS CYCLING" not in _build_repair_reflection_prompt(
            **self._kwargs(), file_revert_streak=1,
            file_revert_files=["server/app/models/birthday.py"],
        )

    def test_renders_at_two_and_forbids_the_alternatives(self):
        from harness.graph import _build_repair_reflection_prompt
        prompt = _build_repair_reflection_prompt(
            **self._kwargs(), file_revert_streak=2,
            file_revert_files=["server/app/models/birthday.py"],
        )
        assert "YOUR GUIDANCE IS CYCLING" in prompt
        assert "birthday.py" in prompt
        # It must tell the judge BOTH alternatives are wrong — telling the
        # model to try harder is what produced the loop.
        assert "are therefore wrong" in prompt
        assert "Do NOT recommend either of them again" in prompt
        # And offer the way out.
        assert "cannot localise it" in prompt

    def test_no_files_means_no_block(self):
        from harness.graph import _build_repair_reflection_prompt
        assert "YOUR GUIDANCE IS CYCLING" not in _build_repair_reflection_prompt(
            **self._kwargs(), file_revert_streak=5, file_revert_files=[],
        )


class TestWorkspaceImportResolution:
    """`_workspace_imports_of` is the downstream half of the widened
    evidence: the modules a failing file imports, which is where the wiring
    that shapes its output lives."""

    def _mk(self, tmp_path):
        import os
        for rel, body in [
            ("server/app/main.py", "x = 1\n"),
            ("server/app/db.py", "y = 2\n"),
            ("server/tests/test_api.py",
             "import pytest\n"
             "from fastapi.testclient import TestClient\n"
             "from server.app.main import create_app\n"
             "from server.app.db import get_db\n"),
        ]:
            p = os.path.join(str(tmp_path), rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write(body)
        return str(tmp_path)

    def test_resolves_workspace_modules(self, tmp_path):
        ws = self._mk(tmp_path)
        out = _workspace_imports_of(ws, ["server/tests/test_api.py"])
        assert "server/app/main.py" in out
        assert "server/app/db.py" in out

    def test_third_party_imports_are_skipped(self, tmp_path):
        """Naming pytest or fastapi as the blocker would be worse than
        naming nothing."""
        ws = self._mk(tmp_path)
        out = _workspace_imports_of(ws, ["server/tests/test_api.py"])
        assert not any("pytest" in f or "fastapi" in f for f in out)

    def test_capped(self, tmp_path):
        ws = self._mk(tmp_path)
        assert len(_workspace_imports_of(
            ws, ["server/tests/test_api.py"], cap=1)) == 1

    def test_missing_file_is_safe(self, tmp_path):
        assert _workspace_imports_of(str(tmp_path), ["nope.py"]) == []

    def test_relative_imports_are_skipped(self, tmp_path):
        import os
        p = os.path.join(str(tmp_path), "m.py")
        open(p, "w").write("from . import sibling\nfrom .pkg import thing\n")
        assert _workspace_imports_of(str(tmp_path), ["m.py"]) == []


# ---------------------------------------------------------------------------
# The judge's named blocker is prefetched (lumina-run5-20260914-2120).
#
# The import prefetch is one hop from the failing test and capped at 6,
# filled in iteration order. The judge named server/app/api/birthdays.py
# every round and the model asked to read it twice, but it was never
# inlined: the file is TWO hops out (the test imports server.app.main;
# main.py registers the router from api/birthdays.py), so the one-hop walk
# never reached it while earlier imports consumed the cap. The prompt
# mentioned that path 13 times and carried its source zero times.
# ---------------------------------------------------------------------------

class TestJudgeNamedPathExtraction:
    """Ordering the prefetch depends on pulling paths out of judge prose.
    Over-matching would inline noise and crowd out the real blocker."""

    def test_extracts_paths_from_a_real_blocker_sentence(self):
        text = (
            "The endpoint at server/app/api/birthdays.py:66 raises the right "
            "error but server/app/main.py overwrites the message"
        )
        assert _RE_WS_SOURCE_PATH.findall(text) == [
            "server/app/api/birthdays.py", "server/app/main.py",
        ]

    @pytest.mark.parametrize("text", [
        "the service layer sorts by first_name rather than last_name",
        "the birthdays module is fine",
        "check the config",
        "version is None instead of 1",
    ])
    def test_prose_without_paths_yields_nothing(self, text):
        assert _RE_WS_SOURCE_PATH.findall(text) == []

    def test_a_bare_filename_is_not_a_path(self):
        """Needs a directory separator — 'main.py' alone is ambiguous and
        would resolve to the wrong file in a multi-package workspace."""
        assert _RE_WS_SOURCE_PATH.findall("the bug is in main.py") == []

    @pytest.mark.parametrize("ext", ["py", "ts", "tsx", "js", "go", "rs", "java"])
    def test_common_source_extensions(self, ext):
        assert _RE_WS_SOURCE_PATH.findall(f"see src/thing.{ext} for it") == [
            f"src/thing.{ext}",
        ]

    def test_non_source_extensions_are_ignored(self):
        """A .md or .lock path in a verdict is not code to inline."""
        assert _RE_WS_SOURCE_PATH.findall("see docs/notes.md and yarn.lock") == []

    def test_line_suffix_does_not_leak_into_the_path(self):
        assert _RE_WS_SOURCE_PATH.findall("at a/b/c.py:66 the call fails") == [
            "a/b/c.py",
        ]


class TestRevertStreakSurvivesNoopRounds:
    """A no-op round must HOLD the oscillation signal, not clear it.

    lumina-run6-20260914-1230: config.py reverted on three consecutive
    rounds (streak 1, 2, 3), then a no-op round on the judge's own target
    reset it to zero — so the reflection prompt built next never carried
    the cycling block, despite the loop still oscillating. "No revert this
    round" is not the same as "the oscillation stopped": a no-op has
    nothing to revert, and is itself evidence of being stuck.
    """

    class _R:
        def __init__(self, success=True, no_op=False, file="a.py"):
            self.success, self.no_op, self.file = success, no_op, file

    def _reset_branch(self, patch_results, streak_before):
        """Mirror of the reset rule in repair_node's revert block."""
        lc = {"file_revert_streak": streak_before, "file_revert_files": ["a.py"]}
        forward = any(r.success and not r.no_op for r in patch_results)
        if forward:
            lc["file_revert_streak"] = 0
            lc["file_revert_files"] = []
        return lc

    def test_a_noop_round_holds_the_streak(self):
        lc = self._reset_branch([self._R(success=False, no_op=True)], 3)
        assert lc["file_revert_streak"] == 3

    def test_a_round_with_no_patches_at_all_holds_the_streak(self):
        assert self._reset_branch([], 3)["file_revert_streak"] == 3

    def test_genuine_forward_progress_clears_it(self):
        lc = self._reset_branch([self._R(success=True, no_op=False)], 3)
        assert lc["file_revert_streak"] == 0
        assert lc["file_revert_files"] == []

    def test_an_idempotency_noop_alongside_nothing_else_holds(self):
        lc = self._reset_branch([self._R(success=True, no_op=True)], 2)
        assert lc["file_revert_streak"] == 2
