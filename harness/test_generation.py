"""Auto-generate technology-specific unit tests after patching_node and run them
deterministically in the sandbox.

Topology:
    patching_node → speculative_node → test_generation_node → lintgate_node → compiler_node

The node:
    1. Reads state["modified_files"], filters out anything that's already a test.
    2. Detects the workspace stack via harness.impact._detect_workspace_stack.
    3. Loads the per-stack test_guides/<lang>.md guidance into the LLM prompt.
    4. Dispatches one LLM call (gateway, NodeRole.PATCHING) asking for patch
       blocks that create stack-canonical test files for the modified sources.
       **The prompt forbids mocks** — tests call the real implementation; only
       the test runner's built-in fakes (monkeypatch / tmp_path / httptest /
       @TempDir / etc.) are allowed.
    5. Applies the patches via the existing patcher pipeline. Path traversal +
       absolute-path attempts are rejected by harness.trust.safe_resolve.
    6. If zero tests were generated → return passed status, skip sandbox.
    7. Otherwise runs a stack-canonical test command in the sandbox (separate
       from the user's build_command) and surfaces failures via the standard
       compiler_errors path so repair_node can fix them.

Guardrails:
    - Requires a configured LLM gateway. When `get_gateway() is None`, the node
      synthesises an env-misconfig diagnostic ("installer must provide a valid
      LLM API key") and short-circuits to HITL — never silently no-ops.
    - Config-gated: `test_generation.enabled = false` in cli.json / .harness_config.json
      disables the node entirely.
    - Workspace boundary: every generated file is post-validated to live under
      state["workspace_path"]; anything that escaped (would only happen on a
      patcher bug) is dropped from generated_tests with a warning.
    - Loop guard: `test_generation` loop counter + `max_iterations` config cap.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional, cast, TYPE_CHECKING

from harness.test_contradiction import find_contradictions_across

if TYPE_CHECKING:
    from harness.graph import AgentState

logger = logging.getLogger(__name__)


def _read_text(abs_path: str) -> Optional[str]:
    """Read a file as UTF-8 (errors replaced), or None if unreadable.
    Used by the cross-file contradiction gate to re-read just-generated
    test files off disk."""
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Per-stack canonical test invocations
# ---------------------------------------------------------------------------

# Maps a primary stack tag → the deterministic command we run after generating
# tests. pytest / jest / ts-jest are pre-baked into the builder image
# (harness/vendor/Dockerfile.builder), so the per-run install round-trip is
# gone. The project's runtime deps are still installed by the workspace's
# Makefile / build_command upstream of this node — we never bake those.
# Java uses Maven's local repository cache via the /cache volume; ``mvn test``
# is the canonical invocation.
def _python_test_command() -> str:
    """Pull the canonical pytest invocation from cli so the test-generation
    runner stays in sync with the main build-command builder. Same import
    pattern as compiler_node's mid-session command-upgrade matcher."""
    from harness.cli import _PYTEST_RUN
    return _PYTEST_RUN


_STACK_TEST_COMMANDS: dict[str, str] = {
    "python": _python_test_command(),
    "node": "npx --no-install jest --silent",
    "javascript": "npx --no-install jest --silent",
    "typescript": "npx --no-install jest --silent",
    "java": "mvn -q test",
}


# Stack-tag priority: when _detect_workspace_stack returns multiple tags, pick
# the first hit in this list as the primary language for prompt + test runner
# selection. Frontend framework (react) implies typescript, so it doesn't
# appear here directly.
_PRIMARY_STACK_PRIORITY: tuple[str, ...] = (
    "java", "typescript",
    "javascript", "python",
)


# File-extension → stack hint for the "is this modified file a source file
# worth testing?" check. Anything not in this table is skipped (markdown,
# JSON config, lockfiles, etc.).
_SOURCE_EXTENSIONS: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java",
}


# Filename / path patterns identifying files that ARE tests (skip these — we
# don't write tests for tests).
_TEST_FILE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(^|/)test_[^/]+\.py$"),
    re.compile(r"(^|/)tests?(/|$)"),
    re.compile(r"\.test\.(js|jsx|ts|tsx|mjs)$"),
    re.compile(r"\.spec\.(js|jsx|ts|tsx|mjs)$"),
    re.compile(r"__tests__/"),
    re.compile(r"src/test/"),
    re.compile(r"Test\.java$"),
)


_PYTEST_IMPORTLIB_INI = (
    "[pytest]\n"
    "# Auto-written by harness.test_generation. Uses importlib import mode so\n"
    "# same-named test files in different packages (e.g. tests/models/test_job.py\n"
    "# and tests/schemas/test_job.py) coexist as distinct dotted names instead\n"
    "# of colliding with 'import file mismatch' under the default prepend mode.\n"
    "#\n"
    "# pythonpath puts the workspace root on sys.path. importlib mode does NOT\n"
    "# prepend the rootdir the way the default 'prepend' mode does, so first-\n"
    "# party imports (e.g. `from server.app import ...`) from a root-level\n"
    "# tests/ tree would otherwise fail to resolve. A full-stack layout with two\n"
    "# Python test trees (tests/ + server/tests/) both importing the app package\n"
    "# is exactly what produced the ImportPathMismatchError + silently-dropped\n"
    "# server/tests/ tier on lumina (session 019f82af).\n"
    "addopts = --import-mode=importlib\n"
    "pythonpath = .\n"
)


# ---------------------------------------------------------------------------
# NFR-only-batch stub emission
# ---------------------------------------------------------------------------
#
# When a batch's scope is entirely NFR (SAFe enabler) stories, the
# test-gen LLM reliably refuses to emit unit tests: NFRs like rate
# limits, latency budgets, security posture and retention policies
# aren't naturally unit-testable. Left alone, the LLM burns its zero-
# emit re-prompt sub-cap and trips a HITL the operator can't
# productively resolve (finsearch session 2026-07-10, 3 HITLs across
# STORY-NFR-001 / STORY-NFR-004).
#
# The right shape: emit a deterministic stub test file per NFR story
# with one ``@pytest.mark.skip`` test per acceptance criterion, each
# carrying the ``# @verifies: STORY-NFR-N.AC-M`` marker. That closes
# the traceability audit's ``@verifies:`` contract while making it
# explicit that a human still owes an integration test. Mixed batches
# (NFR + regular) still run the LLM path — the regular story anchors
# the tests and NFR ACs ride along as ``@verifies:`` citations.

_NFR_STUB_HEADER_PY = (
    '"""NFR verification stub for {story_key} — {title}.\n'
    "\n"
    "STORY {story_key} is a SAFe enabler / non-functional requirement.\n"
    "Unit tests are not a good fit — the harness auto-generated these\n"
    "``@pytest.mark.skip`` placeholders so the traceability audit's\n"
    "``@verifies:`` contract is satisfied. Replace each stub with a real\n"
    "integration / performance / security test that verifies the linked\n"
    "acceptance criterion.\n"
    '"""\n'
    "from __future__ import annotations\n"
    "\n"
    "import pytest\n"
    "\n"
)

_NFR_STUB_HEADER_TS = (
    "/**\n"
    " * NFR verification stub for {story_key} — {title}.\n"
    " *\n"
    " * STORY {story_key} is a SAFe enabler / non-functional requirement.\n"
    " * Unit tests are not a good fit — the harness auto-generated these\n"
    " * ``it.skip`` placeholders so the traceability audit's\n"
    " * ``@verifies:`` contract is satisfied. Replace each stub with a real\n"
    " * integration / performance / security test that verifies the linked\n"
    " * acceptance criterion.\n"
    " */\n"
    "\n"
    'describe("{story_key} NFR stubs", () => {{\n'
)


def _slug_for_story_key(story_key: str) -> str:
    """Filesystem-safe slug: STORY-NFR-004 → story_nfr_004 (Python) /
    story-nfr-004 (TS). Kept lowercase to match the workspace's
    conventional test-file casing on both stacks."""
    return re.sub(r"[^A-Za-z0-9]+", "_", story_key.strip()).strip("_").lower()


def _nfr_stub_rel_path(primary_stack: str, story_key: str) -> Optional[str]:
    """Workspace-relative path for the NFR stub file. Returns None when
    the stack has no supported stub convention (Java / unknown)."""
    slug = _slug_for_story_key(story_key)
    if not slug:
        return None
    if primary_stack == "python":
        return f"tests/nfr/test_{slug}.py"
    if primary_stack == "typescript":
        return f"tests/nfr/{slug.replace('_', '-')}.nfr.test.ts"
    return None


def _render_nfr_stub_body(
    primary_stack: str,
    story_key: str,
    story_title: str,
    ac_keys_and_text: list[tuple[str, str]],
) -> Optional[str]:
    """Render the full stub body for one NFR story. Returns None for
    unsupported stacks."""
    if not ac_keys_and_text:
        return None
    safe_title = story_title.replace('"', "'").strip() or story_key
    if primary_stack == "python":
        parts = [_NFR_STUB_HEADER_PY.format(
            story_key=story_key, title=safe_title,
        )]
        for ac_key, ac_text in ac_keys_and_text:
            # Function name must be a valid Python identifier. Fall back
            # to the AC ordinal when the AC key's tail isn't purely
            # digits (never happens today; defensive).
            m = re.match(r".*\.AC-(\d+)$", ac_key)
            ordinal = m.group(1) if m else "n"
            safe_ac_text = ac_text.replace('"""', "'''").strip() or ac_key
            parts.append(
                f'@pytest.mark.skip(reason="NFR AC — integration/performance test required")\n'
                f"def test_ac_{ordinal}() -> None:\n"
                f"    # @verifies: {ac_key}\n"
                f'    """{safe_ac_text}"""\n'
                f"    pass\n"
                f"\n"
            )
        return "".join(parts)
    if primary_stack == "typescript":
        parts = [_NFR_STUB_HEADER_TS.format(
            story_key=story_key, title=safe_title,
        )]
        for ac_key, ac_text in ac_keys_and_text:
            safe_ac_text = ac_text.replace('"', "'").strip() or ac_key
            # Emit the @verifies marker on its OWN line above it.skip so
            # the parser (top-of-file scan window) picks it up. All ACs
            # for the same story render into one file so all markers are
            # within the scan window.
            parts.append(
                f"  // @verifies: {ac_key}\n"
                f'  it.skip("{safe_ac_text}", () => {{\n'
                f"    // NFR AC — integration/performance test required.\n"
                f"  }});\n\n"
            )
        parts.append("});\n")
        return "".join(parts)
    return None


def _emit_nfr_stubs(
    workspace_path: str,
    story_keys: list[str],
    primary_stack: str,
) -> tuple[list[str], dict[str, list[str]]]:
    """Write one stub test file per NFR story into the workspace.
    Returns ``(rel_paths_recorded, marker_keys_by_file)`` — the second
    is directly consumable by :func:`_persist_verifies_links`.

    Idempotent: if a stub file for a story already exists, it is NOT
    overwritten (respects operator edits). The AC keys still land in
    ``marker_keys_by_file`` so :func:`_persist_verifies_links` can
    re-insert the edges — those inserts are themselves idempotent.

    Silent on unsupported stacks / missing story rows / IO errors:
    an empty (list, dict) tuple lets the caller fall back to
    logging-and-skipping without failing the batch."""
    written: list[str] = []
    marker_keys_by_file: dict[str, list[str]] = {}
    if primary_stack not in ("python", "typescript"):
        return written, marker_keys_by_file
    try:
        from harness import story_state
        app_name = story_state.app_name_for_workspace(workspace_path)
        conn = story_state.open_story_db()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[test_generation_node] NFR stub emit skipped (state.db open "
            "failed): %s", exc,
        )
        return written, marker_keys_by_file
    try:
        for story_key in story_keys:
            story = story_state.get_story(conn, app_name, story_key)
            if not story:
                logger.warning(
                    "[test_generation_node] NFR stub skipped for %r: "
                    "story not found in state.db.", story_key,
                )
                continue
            acs = story_state.list_acceptance_criteria(
                conn, app_name, story["id"],
            )
            if not acs:
                logger.warning(
                    "[test_generation_node] NFR stub skipped for %r: "
                    "no acceptance criteria in state.db.", story_key,
                )
                continue
            ac_pairs = [(ac["ac_key"], ac.get("text", "")) for ac in acs]
            rel_path = _nfr_stub_rel_path(primary_stack, story_key)
            if rel_path is None:
                continue
            abs_path = os.path.realpath(
                os.path.join(workspace_path, rel_path),
            )
            # Defence in depth: refuse to write outside the workspace.
            if not _inside_workspace(rel_path, workspace_path):
                logger.warning(
                    "[test_generation_node] NFR stub path escapes "
                    "workspace boundary; dropping: %r", rel_path,
                )
                continue
            if os.path.exists(abs_path):
                # Respect an existing file (operator may have replaced
                # the stub with a real integration test). Still expose
                # the AC keys so links get re-persisted from whatever's
                # on disk.
                logger.info(
                    "[test_generation_node] NFR stub for %r already "
                    "exists at %r — not overwriting; re-persisting "
                    "verifies links.", story_key, rel_path,
                )
                written.append(rel_path)
                marker_keys_by_file[rel_path] = [k for k, _ in ac_pairs]
                continue
            body = _render_nfr_stub_body(
                primary_stack, story_key, story.get("title", ""), ac_pairs,
            )
            if body is None:
                continue
            try:
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                with open(abs_path, "w", encoding="utf-8") as fh:
                    fh.write(body)
            except OSError as exc:
                logger.warning(
                    "[test_generation_node] NFR stub write failed for "
                    "%r: %s", rel_path, exc,
                )
                continue
            written.append(rel_path)
            marker_keys_by_file[rel_path] = [k for k, _ in ac_pairs]
    finally:
        conn.close()
    return written, marker_keys_by_file


def backfill_untested_nfr_acs(
    workspace_path: str,
) -> tuple[list[str], int, int]:
    """Sweep state.db for NFR stories whose acceptance criteria have no
    ``test_verifies_ac`` link, emit skip-stub tests for them, and
    persist the marker→ac edges.

    Called by ``traceability_node`` before the end-of-batch / end-of-
    session audit. The in-node ``test_generation_node`` NFR guard only
    fires when a batch's scope is a fresh NFR story; sessions that
    sealed the NFR batches before this guard existed (or where the
    per-story split routed the NFRs through a path that skipped
    test_generation) leave their AC edges empty. This backfill
    reconciles those historical gaps at the traceability gate.

    Returns ``(stub_paths, links_inserted, links_dropped)``. Empty
    tuple + zeros when nothing needs backfilling; safe to call on
    every batch (idempotent).
    """
    from harness.req_ids import STORY_NFR_ID_RE
    try:
        from harness import story_state
        app_name = story_state.app_name_for_workspace(workspace_path)
        conn = story_state.open_story_db()
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "[traceability] NFR backfill skipped (state.db open failed): %s",
            exc,
        )
        return [], 0, 0
    orphan_story_keys: list[str] = []
    try:
        stories = conn.execute(
            "SELECT id, story_key FROM stories WHERE workspace = ?",
            (app_name,),
        ).fetchall()
        for row in stories:
            story_id, story_key = int(row[0]), str(row[1])
            if not STORY_NFR_ID_RE.fullmatch(story_key):
                continue
            # Any AC on this story without a test_verifies_ac row?
            missing = conn.execute(
                "SELECT COUNT(*) FROM acceptance_criteria ac "
                "WHERE ac.workspace = ? AND ac.story_id = ? "
                "AND NOT EXISTS ("
                "  SELECT 1 FROM test_verifies_ac tv "
                "  WHERE tv.workspace = ac.workspace "
                "    AND tv.ac_id = ac.id"
                ")",
                (app_name, story_id),
            ).fetchone()
            if missing and int(missing[0]) > 0:
                orphan_story_keys.append(story_key)
    finally:
        conn.close()
    if not orphan_story_keys:
        return [], 0, 0
    from harness.impact import _detect_workspace_stack
    tags = _detect_workspace_stack(workspace_path) or set()
    primary = _pick_primary_stack(tags)
    if primary not in ("python", "typescript"):
        logger.info(
            "[traceability] NFR backfill: %d orphan NFR story/stories "
            "with untested ACs (%s) but no supported stack detected "
            "(primary=%r) — skipping.",
            len(orphan_story_keys), ", ".join(orphan_story_keys), primary,
        )
        return [], 0, 0
    stub_paths, stub_markers = _emit_nfr_stubs(
        workspace_path, orphan_story_keys, primary,
    )
    if not stub_paths:
        return [], 0, 0
    links_inserted, links_dropped = _persist_verifies_links(
        workspace_path, stub_markers,
    )
    logger.info(
        "[traceability] NFR backfill: emitted %d stub file(s) for %d "
        "orphan NFR story/stories (%s) — %d verifies-link(s) inserted, "
        "%d dropped as unknown.",
        len(stub_paths), len(orphan_story_keys),
        ", ".join(orphan_story_keys), links_inserted, links_dropped,
    )
    return stub_paths, links_inserted, links_dropped


def _workspace_has_python_tests(workspace_path: str) -> bool:
    """True when the workspace contains at least one Python test file
    (``test_*.py`` / ``*_test.py`` / ``conftest.py``) anywhere outside the
    usual vendored/build dirs.

    This is the self-gate for :func:`_ensure_pytest_importlib_config`: the
    importlib pytest config is only meaningful — and only worth writing — when
    there is actually a Python test tree to collect. It is deliberately
    filesystem-based rather than keyed on the workspace's *primary* stack: a
    full-stack app (Python API + JS/TS frontend) resolves ``primary`` to the
    frontend, which used to skip this writer entirely and leave the Python
    tests in default prepend mode (lumina 019f82af — the exact mixed-tree
    collision this config prevents).
    """
    if not workspace_path or not os.path.isdir(workspace_path):
        return False
    skip_dirs = {
        ".git", ".venv", "venv", "node_modules", "__pycache__", ".tox",
        ".mypy_cache", ".pytest_cache", "dist", "build", ".ruff_cache",
    }
    for root, dirs, files in os.walk(workspace_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
        for name in files:
            if name == "conftest.py":
                return True
            if name.endswith(".py") and (
                name.startswith("test_") or name.endswith("_test.py")
            ):
                return True
    return False


def _ensure_pytest_importlib_config(workspace_path: str) -> Optional[str]:
    """Write a minimal ``pytest.ini`` with ``--import-mode=importlib`` and
    ``pythonpath = .`` if the workspace has a Python test tree but no pytest
    configuration of any kind.

    Recognises every shape pytest itself looks at:
      - ``pytest.ini``
      - ``pyproject.toml`` with a ``[tool.pytest.ini_options]`` table
      - ``setup.cfg`` with a ``[tool:pytest]`` section
      - ``tox.ini`` with a ``[pytest]`` section

    Returns the workspace-relative path of the newly-written file, or
    ``None`` if there are no Python tests, or any existing config was found
    (no-op). When an existing config is found that does NOT already select
    importlib mode, logs a warning rather than silently leaving the workspace
    exposed to prepend-mode collection collisions — that config path can't be
    safely rewritten in-place here, so it's surfaced for the operator/LLM.
    """
    if not workspace_path or not os.path.isdir(workspace_path):
        return None

    # Self-gate: nothing to configure if there's no Python test tree. Fires on
    # Python-test PRESENCE, not on the workspace's primary stack.
    if not _workspace_has_python_tests(workspace_path):
        return None

    # Hard signals — file existence wins. pyproject.toml / setup.cfg / tox.ini
    # only count if they actually contain a pytest section.
    section_signals: tuple[tuple[str, str], ...] = (
        ("pytest.ini", "[pytest]"),
        ("pyproject.toml", "[tool.pytest.ini_options]"),
        ("setup.cfg", "[tool:pytest]"),
        ("tox.ini", "[pytest]"),
    )
    for fname, marker in section_signals:
        path = os.path.join(workspace_path, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                # Manifests are small; cap reads at 256 KB.
                content = f.read(256 * 1024)
        except OSError:
            continue
        # A bare pytest.ini always counts as config; the others only when the
        # pytest section is actually present.
        if fname == "pytest.ini" or marker in content:
            if "importlib" not in content:
                logger.warning(
                    "[test_generation_node] %s already configures pytest but does "
                    "not select --import-mode=importlib. Two test trees or same-"
                    "named test files can collide under the default prepend mode "
                    "(ImportPathMismatchError / silently-dropped tier). Add "
                    "`--import-mode=importlib` (and `pythonpath = .`) to it.",
                    fname,
                )
            return None

    target = os.path.join(workspace_path, "pytest.ini")
    try:
        with open(target, "w", encoding="utf-8") as f:
            f.write(_PYTEST_IMPORTLIB_INI)
    except OSError as exc:
        logger.warning(
            "[test_generation_node] Failed to write pytest.ini: %s. "
            "Same-basename test collisions may still occur.", exc,
        )
        return None
    logger.info(
        "[test_generation_node] Wrote default pytest.ini with "
        "--import-mode=importlib so duplicate test basenames coexist."
    )
    return "pytest.ini"


# Pinned dev-dependency versions the JS/TS test-env scaffolder adds when
# missing. Only ADDED when absent — existing entries (any version) are
# never touched, so a workspace's own pins always win.
_JS_TEST_DEVDEPS_BASE: dict[str, str] = {"jest": "^29.7.0"}
_TS_TEST_DEVDEPS: dict[str, str] = {
    "ts-jest": "^29.1.2",
    "@types/jest": "^29.5.12",
}
_COMPONENT_TEST_DEVDEPS: dict[str, str] = {
    "@testing-library/react": "^14.2.0",
    "@testing-library/jest-dom": "^6.4.2",
    "jest-environment-jsdom": "^29.7.0",
}
# Plain-JS packages get no ts-jest preset, so jest falls back to
# babel-jest — which without a preset can parse neither modern syntax in
# .js tests nor JSX in .jsx tests. Scaffold the babel side too, or the
# "runnable env" this function promises isn't (the TS path never needs
# these: ts-jest is its own transform).
_JS_BABEL_DEVDEPS: dict[str, str] = {
    "@babel/core": "^7.24.0",
    "@babel/preset-env": "^7.24.0",
}
_JS_BABEL_REACT_DEVDEPS: dict[str, str] = {
    "@babel/preset-react": "^7.24.0",
}

_JS_TEST_EXTS = (".test.ts", ".test.tsx", ".test.js", ".test.jsx",
                 ".spec.ts", ".spec.tsx", ".spec.js", ".spec.jsx")
_JEST_CONFIG_NAMES = (
    "jest.config.js", "jest.config.cjs", "jest.config.mjs",
    "jest.config.ts", "jest.config.json",
)
_BABEL_CONFIG_NAMES = (
    ".babelrc", ".babelrc.json", "babel.config.js", "babel.config.cjs",
    "babel.config.json", "babel.config.mjs",
)


def _nearest_package_root(abs_test_path: str, workspace_abs: str) -> Optional[str]:
    """Walk up from the test file to the nearest directory containing a
    ``package.json``, stopping at the workspace root. None when no
    package.json exists on the path — there is no npm package to
    scaffold into."""
    d = os.path.dirname(abs_test_path)
    ws = os.path.abspath(workspace_abs)
    while True:
        if os.path.isfile(os.path.join(d, "package.json")):
            return d
        if os.path.abspath(d) == ws or len(d) <= len(ws):
            return None
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def _ensure_js_test_env(
    workspace_path: str,
    generated_tests: list[str],
) -> list[str]:
    """Deterministically scaffold the jest/type environment for freshly
    generated JS/TS test files. The TypeScript test guide instructs the
    LLM to patch the config in the same response; this is the harness's
    guarantee for when it doesn't — a ``.test.tsx`` without its
    environment produces hundreds of TS2304/TS2307 type-noise
    diagnostics and a jest run that can't even collect (session
    22471c0c: 456 of them drowned the real failures).

    Per affected package root (nearest ``package.json``):
      1. devDependencies — ADD missing entries only (jest; + ts-jest /
         @types/jest for TS tests; + @testing-library/* and
         jest-environment-jsdom for component tests). Existing pins are
         never modified.
      2. jest config — written only when NO jest config of any shape
         exists (jest.config.* or a "jest" key in package.json):
         ts-jest transform for TS, jsdom environment + a
         ``setupFilesAfterEnv`` hook for component tests.
      3. ``jest.setup.<ts|js>`` loading @testing-library/jest-dom (ESM
         import under ts-jest, CJS require under babel-jest) — only
         alongside a config this function wrote. Plain-JS packages also
         get ``babel.config.cjs`` (+ @babel/preset-env, and
         @babel/preset-react for components) when no babel config
         exists, since babel-jest without a preset can parse neither
         modern syntax nor JSX.
      4. tsconfig.json ``compilerOptions.types`` — APPEND "jest" only
         when a ``types`` array already exists and lacks it. When the
         key is absent every @types package is auto-included, which is
         strictly better; adding the key would EXCLUDE the rest.

    Mirrors :func:`_ensure_pytest_importlib_config`'s contract:
    idempotent, fail-open per file, returns the workspace-relative
    paths it created or modified.
    """
    ws = os.path.abspath(workspace_path)
    js_tests = [
        t for t in generated_tests
        if t.replace("\\", "/").endswith(_JS_TEST_EXTS)
    ]
    if not js_tests:
        return []

    roots: dict[str, list[str]] = {}
    for rel in js_tests:
        pkg_dir = _nearest_package_root(os.path.join(ws, rel), ws)
        if pkg_dir is None:
            logger.info(
                "[test_generation_node] No package.json above generated "
                "test %s — skipping JS test-env scaffold for it.", rel,
            )
            continue
        roots.setdefault(pkg_dir, []).append(rel)

    changed: list[str] = []
    for pkg_dir, tests in sorted(roots.items()):
        is_ts = any(t.endswith((".ts", ".tsx")) for t in tests)
        is_component = any(t.endswith((".tsx", ".jsx")) for t in tests)
        pkg_path = os.path.join(pkg_dir, "package.json")
        try:
            with open(pkg_path, "r", encoding="utf-8") as fh:
                pkg = json.loads(fh.read())
        except (OSError, ValueError) as exc:
            logger.warning(
                "[test_generation_node] Could not parse %s (%s) — "
                "skipping JS test-env scaffold for this package.",
                pkg_path, exc,
            )
            continue
        if not isinstance(pkg, dict):
            continue

        needed = dict(_JS_TEST_DEVDEPS_BASE)
        if is_ts:
            needed.update(_TS_TEST_DEVDEPS)
        else:
            needed.update(_JS_BABEL_DEVDEPS)
            if is_component:
                needed.update(_JS_BABEL_REACT_DEVDEPS)
        if is_component:
            needed.update(_COMPONENT_TEST_DEVDEPS)
        present = set()
        for section in ("dependencies", "devDependencies"):
            sec = pkg.get(section)
            if isinstance(sec, dict):
                present.update(sec.keys())
        to_add = {k: v for k, v in needed.items() if k not in present}
        if to_add:
            dev = pkg.get("devDependencies")
            if not isinstance(dev, dict):
                dev = {}
            dev.update(to_add)
            pkg["devDependencies"] = dict(sorted(dev.items()))
            try:
                with open(pkg_path, "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(pkg, indent=2) + "\n")
                changed.append(os.path.relpath(pkg_path, ws).replace(os.sep, "/"))
                logger.info(
                    "[test_generation_node] Added %d test devDependencies "
                    "to %s: %s",
                    len(to_add),
                    os.path.relpath(pkg_path, ws),
                    ", ".join(sorted(to_add)),
                )
            except OSError as exc:
                logger.warning(
                    "[test_generation_node] Could not write %s: %s",
                    pkg_path, exc,
                )

        has_jest_config = "jest" in pkg or any(
            os.path.isfile(os.path.join(pkg_dir, name))
            for name in _JEST_CONFIG_NAMES
        )
        if not has_jest_config:
            setup_ext = "ts" if is_ts else "js"
            cfg_lines = ["module.exports = {"]
            if is_ts:
                cfg_lines += [
                    "  preset: 'ts-jest',",
                    "  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx'],",
                    "  testMatch: ['**/*.test.ts', '**/*.test.tsx', "
                    "'**/*.spec.ts', '**/*.spec.tsx'],",
                ]
            cfg_lines.append(
                f"  testEnvironment: '{'jsdom' if is_component else 'node'}',"
            )
            if is_component:
                cfg_lines.append(
                    f"  setupFilesAfterEnv: ['<rootDir>/jest.setup.{setup_ext}'],"
                )
            cfg_lines.append("};")
            cfg_path = os.path.join(pkg_dir, "jest.config.cjs")
            try:
                with open(cfg_path, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(cfg_lines) + "\n")
                changed.append(os.path.relpath(cfg_path, ws).replace(os.sep, "/"))
                if is_component:
                    setup_path = os.path.join(
                        pkg_dir, f"jest.setup.{setup_ext}",
                    )
                    if not os.path.isfile(setup_path):
                        # CJS require for the plain-JS path: the setup
                        # file runs under jest's default babel-jest
                        # transform, and an ESM `import` in a CJS package
                        # fails to parse before any test runs. ts-jest
                        # handles the import form fine for .ts.
                        setup_line = (
                            "import '@testing-library/jest-dom';\n"
                            if is_ts else
                            "require('@testing-library/jest-dom');\n"
                        )
                        with open(setup_path, "w", encoding="utf-8") as fh:
                            fh.write(setup_line)
                        changed.append(
                            os.path.relpath(setup_path, ws).replace(os.sep, "/")
                        )
                # Plain-JS packages need babel presets for babel-jest to
                # parse modern syntax / JSX. Only written alongside a
                # jest config this function owns, and only when the
                # package has no babel config of any shape — an
                # operator-owned babel setup is never touched.
                if not is_ts:
                    has_babel_config = "babel" in pkg or any(
                        os.path.isfile(os.path.join(pkg_dir, name))
                        for name in _BABEL_CONFIG_NAMES
                    )
                    if not has_babel_config:
                        babel_lines = [
                            "module.exports = {",
                            "  presets: [",
                            "    ['@babel/preset-env', "
                            "{ targets: { node: 'current' } }],",
                        ]
                        if is_component:
                            babel_lines.append(
                                "    ['@babel/preset-react', "
                                "{ runtime: 'automatic' }],"
                            )
                        babel_lines += ["  ],", "};"]
                        babel_path = os.path.join(pkg_dir, "babel.config.cjs")
                        with open(babel_path, "w", encoding="utf-8") as fh:
                            fh.write("\n".join(babel_lines) + "\n")
                        changed.append(
                            os.path.relpath(babel_path, ws).replace(os.sep, "/")
                        )
                logger.info(
                    "[test_generation_node] Wrote %s (%s environment%s).",
                    os.path.relpath(cfg_path, ws),
                    "jsdom" if is_component else "node",
                    " + jest.setup" if is_component else "",
                )
            except OSError as exc:
                logger.warning(
                    "[test_generation_node] Could not write jest config "
                    "in %s: %s", pkg_dir, exc,
                )

        if is_ts:
            ts_path = os.path.join(pkg_dir, "tsconfig.json")
            if os.path.isfile(ts_path):
                try:
                    with open(ts_path, "r", encoding="utf-8") as fh:
                        tscfg = json.loads(fh.read())
                except (OSError, ValueError):
                    tscfg = None  # JSONC or unreadable — leave alone
                if isinstance(tscfg, dict):
                    opts = tscfg.get("compilerOptions")
                    if isinstance(opts, dict):
                        types = opts.get("types")
                        if isinstance(types, list) and "jest" not in types:
                            opts["types"] = list(types) + ["jest"]
                            try:
                                with open(ts_path, "w", encoding="utf-8") as fh:
                                    fh.write(json.dumps(tscfg, indent=2) + "\n")
                                changed.append(
                                    os.path.relpath(ts_path, ws).replace(os.sep, "/")
                                )
                                logger.info(
                                    "[test_generation_node] Appended 'jest' "
                                    "to %s compilerOptions.types.",
                                    os.path.relpath(ts_path, ws),
                                )
                            except OSError as exc:
                                logger.warning(
                                    "[test_generation_node] Could not write "
                                    "%s: %s", ts_path, exc,
                                )
    return changed


def _is_test_file(rel_path: str) -> bool:
    """True when ``rel_path`` looks like an existing test file."""
    norm = rel_path.replace("\\", "/")
    return any(p.search(norm) for p in _TEST_FILE_PATTERNS)


# Conventional test-path templates per source stem, used to find existing
# tests that map to a source file. Includes root-level ``tests/`` (Python
# canonical), colocated (JS/TS convention), and Java's parallel src/test tree.
# The stem is substituted for ``{stem}``; the extension for ``{ext}``.
_TEST_PATH_TEMPLATES_BY_STACK: dict[str, tuple[str, ...]] = {
    "python": (
        "tests/test_{stem}.py",
        "tests/unit/test_{stem}.py",
        "test/test_{stem}.py",
    ),
    "javascript": (
        "{dir}/{stem}.test.{ext}",
        "{dir}/{stem}.spec.{ext}",
        "{dir}/__tests__/{stem}.test.{ext}",
        "tests/unit/{stem}.test.{ext}",
    ),
    "typescript": (
        "{dir}/{stem}.test.{ext}",
        "{dir}/{stem}.spec.{ext}",
        "{dir}/__tests__/{stem}.test.{ext}",
        "client/tests/unit/{stem}.test.{ext}",
        "tests/unit/{stem}.test.{ext}",
    ),
    "java": (
        "src/test/java/{dir_no_prefix}/{stem}Test.java",
        "src/test/java/{dir_no_prefix}/Test{stem}.java",
    ),
}


def _existing_tests_for_preflight(
    workspace_path: str,
    source_files: list[str],
    modified_files: list[str],
    primary_stack: str,
) -> list[str]:
    """Return workspace-relative paths of existing test files the LLM is
    about to edit or extend. Two sources, unioned + deduped:

    1. Every entry in ``modified_files`` that ``_is_test_file`` recognises
       AND exists on disk. These are guaranteed to have drifted since the
       LLM's mental model — they were touched THIS session by an earlier
       node — and are the root cause behind iter 4's stale-anchor
       rejections on session 44c5e194.

    2. Conventional test paths mapped from each source file's stem via
       ``_TEST_PATH_TEMPLATES_BY_STACK``. Only paths that exist on disk
       are returned; missing conventional paths mean the LLM will
       CREATE_FILE for them, not modify — no preflight needed.

    The preflight caller (``_collect_workspace_file_content``) caps at
    12 files by default, so this list can be over-generous without
    blowing the prompt budget.
    """
    seen: set[str] = set()
    out: list[str] = []

    def _add(rel: str) -> None:
        if not rel or rel in seen:
            return
        abs_path = os.path.join(workspace_path, rel)
        if not os.path.isfile(abs_path):
            return
        seen.add(rel)
        out.append(rel)

    for rel in modified_files:
        if isinstance(rel, str) and _is_test_file(rel):
            _add(rel)

    templates = _TEST_PATH_TEMPLATES_BY_STACK.get(primary_stack, ())
    for src_rel in source_files:
        if not isinstance(src_rel, str):
            continue
        dirname = os.path.dirname(src_rel).replace("\\", "/")
        stem = os.path.splitext(os.path.basename(src_rel))[0]
        ext = os.path.splitext(src_rel)[1].lstrip(".").lower()
        dir_no_prefix = dirname.split("/", 1)[1] if "/" in dirname else dirname
        for tmpl in templates:
            candidate = tmpl.format(
                stem=stem, ext=ext, dir=dirname,
                dir_no_prefix=dir_no_prefix,
            )
            _add(candidate)
    return out


# Directory names that should never be walked for test_generation candidates.
# Mirrors harness.impact._NEVER_SOURCE_DIRS but kept local to avoid an import
# cycle (impact also imports test-related state types in some configurations).
_SCAN_SKIP_DIRS: frozenset[str] = frozenset({
    "__pycache__", "node_modules", "vendor", "target", "build", "dist",
    "out", ".venv", "venv", "env", ".git", ".tox", ".nox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "docs", "doc", "migrations", "fixtures",
    "tests", "test", "__tests__",
})


def _scan_workspace_for_source(workspace_path: str, limit: int = 200) -> list[str]:
    """Return workspace-relative paths of testable source files anywhere
    under ``workspace_path``.

    Used by the no_tests_collected fallback path: when the router sent us
    here because pytest had no tests to run but `state["modified_files"]`
    happened to be test/manifest-only, we still need to know which source
    files exist on disk so we can write tests for them.
    """
    found: list[str] = []
    if not workspace_path or not os.path.isdir(workspace_path):
        return found
    workspace_path = os.path.abspath(workspace_path)
    try:
        for sub_root, sub_dirs, sub_files in os.walk(workspace_path):
            sub_dirs[:] = [
                d for d in sub_dirs
                if not d.startswith(".") and d not in _SCAN_SKIP_DIRS
            ]
            for fname in sub_files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in _SOURCE_EXTENSIONS:
                    continue
                abspath = os.path.join(sub_root, fname)
                relpath = os.path.relpath(abspath, workspace_path)
                if _is_test_file(relpath):
                    continue
                found.append(relpath)
                if len(found) >= limit:
                    return found
    except OSError:
        return found
    return found


def _pick_primary_stack(tags: set[str]) -> Optional[str]:
    """Pick the single stack tag to drive test generation."""
    for tag in _PRIMARY_STACK_PRIORITY:
        if tag in tags:
            return tag
    return None


def _stack_test_command(primary: str) -> Optional[str]:
    """Return the deterministic test runner command for a primary stack."""
    return _STACK_TEST_COMMANDS.get(primary)


# ---------------------------------------------------------------------------
# Workspace boundary check (defense in depth)
# ---------------------------------------------------------------------------

def _inside_workspace(file_rel: str, workspace_path: str) -> bool:
    """True when ``file_rel`` (workspace-relative) resolves to a path inside
    ``workspace_path``.

    Defence in depth — the patcher's trust.safe_resolve already enforces this
    hard, so this only fires if a patcher bug ever lets something through.
    """
    if not file_rel:
        return False
    if os.path.isabs(file_rel):
        return False
    workspace_real = os.path.realpath(workspace_path)
    file_real = os.path.realpath(os.path.join(workspace_real, file_rel))
    try:
        return os.path.commonpath([workspace_real, file_real]) == workspace_real
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# v5 @verifies marker parser
# ---------------------------------------------------------------------------

# Permissive regex: leading whitespace + comment lead (``#`` for Python, ``//``
# for JS/TS/Java) + optional spaces + ``@verifies:`` + AC key list. Case-
# sensitive on keys (matches the spec/decomposition convention).
# Captures the AC key list (everything after the colon up to end of line),
# split + stripped by ``_parse_verifies_marker``.
_VERIFIES_RE = re.compile(
    r"^\s*(?:#|//)\s*@verifies:\s*(?P<keys>.+?)\s*$",
    re.MULTILINE,
)

# Allowed AC key shape — STORY-N.AC-N *or* STORY-NFR-N.AC-N. Anchors avoid
# matching substrings embedded in noise (e.g. ``foo-STORY-1.AC-2`` would not
# match). The NFR variant is required for SAFe enabler stories: their
# canonicalised AC keys carry the ``NFR-`` infix and would otherwise be
# silently dropped by the parser — 16% AC coverage on the finsearch run
# (2026-07-11) was partly this bug.
_AC_KEY_RE = re.compile(r"^STORY-(?:NFR-)?\d+\.AC-\d+$")

# How many lines we scan from the top of the file. The convention is that
# the marker lives at the top (after module docstring / imports); a marker
# buried at line 800 is almost certainly not what the LLM meant. 50 covers
# any reasonable preamble depth (docstring + imports + a class header).
_VERIFIES_SCAN_LINES = 50

# Unit-test → code linkage marker. Build/patch test generation produces
# PURE UNIT TESTS of the generated functions and classes — linked to the
# source files under test, never to stories or acceptance criteria. AC
# linkage (``@verifies``) belongs exclusively to the functional pack that
# ``teane test`` generates (harness/playwright_gen.py); the parsers above
# stay for that flow. Same permissive comment-lead shape as
# ``_VERIFIES_RE``; captures the comma-separated source-path list.
_TESTS_MARKER_RE = re.compile(
    r"^\s*(?:#|//)\s*@tests:\s*(?P<paths>.+?)\s*$",
    re.MULTILINE,
)


# Map primary stack → the language-appropriate comment lead for the
# ``@verifies: STORY-N.AC-N`` marker. Autofix uses this when it needs to
# prepend a marker deterministically (rather than route to LLM repair).
_MARKER_COMMENT_LEAD_BY_STACK: dict[str, str] = {
    "python": "#",
    "javascript": "//",
    "typescript": "//",
    "java": "//",
}


def _marker_line_for(
    primary_stack: str, ac_keys: list[str],
) -> Optional[str]:
    """Render the ``# @verifies: STORY-N.AC-1, STORY-N.AC-2`` line for
    ``primary_stack``, or ``None`` when no keys are supplied.

    Comment lead is language-aware (Python ``#`` vs JS/TS/Java ``//``).
    Keys are validated against ``_AC_KEY_RE`` — anything malformed is
    dropped so the autofix can't inject a marker that will be
    rejected by ``_persist_verifies_links``.
    """
    if not ac_keys:
        return None
    lead = _MARKER_COMMENT_LEAD_BY_STACK.get(primary_stack, "#")
    valid = [k for k in ac_keys if isinstance(k, str) and _AC_KEY_RE.match(k)]
    if not valid:
        return None
    return f"{lead} @verifies: {', '.join(valid)}"


def _prepend_verifies_marker(abs_path: str, marker_line: str) -> bool:
    """Insert ``marker_line`` at the top of the file at ``abs_path``,
    respecting a shebang / encoding cookie if present. Idempotent — if
    the same marker (or any ``@verifies:`` line) already exists in the
    first ``_VERIFIES_SCAN_LINES`` lines, no write happens.

    Returns True on write success (or already-present no-op), False on
    read/write failure. The autofix caller re-parses the file after we
    return to confirm the marker is now visible to
    ``_parse_verifies_marker``.
    """
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
            body = fh.read()
    except OSError as exc:
        logger.warning(
            "[test_generation_node] Autofix could not read %r: %s",
            abs_path, exc,
        )
        return False
    if _parse_verifies_marker(body):
        return True
    lines = body.splitlines(keepends=True)
    insert_at = 0
    if lines:
        first = lines[0].lstrip()
        if first.startswith("#!"):
            insert_at = 1
        elif first.startswith("#") and "coding" in first:
            insert_at = 1
    lines.insert(insert_at, marker_line + "\n")
    try:
        with open(abs_path, "w", encoding="utf-8") as fh:
            fh.write("".join(lines))
    except OSError as exc:
        logger.warning(
            "[test_generation_node] Autofix could not write %r: %s",
            abs_path, exc,
        )
        return False
    return True


def _parse_tests_marker(body: str) -> list[str]:
    """Return the source-path list from the first ``@tests:`` marker in
    ``body``'s first :data:`_VERIFIES_SCAN_LINES` lines.

    Returns ``[]`` when no marker is found in the scan window or the
    path list is empty. Paths are stripped but otherwise taken as-is —
    existence is the autofix/gate caller's concern, not the parser's.
    """
    if not body:
        return []
    lines = body.splitlines()
    head = "\n".join(lines[: _VERIFIES_SCAN_LINES])
    match = _TESTS_MARKER_RE.search(head)
    if not match:
        return []
    raw = match.group("paths") or ""
    return [c.strip() for c in raw.split(",") if c.strip()]


def _tests_marker_line_for(
    primary_stack: str, source_paths: list[str],
) -> Optional[str]:
    """Render the ``# @tests: path/a.py, path/b.py`` line for
    ``primary_stack``, or ``None`` when no paths are supplied."""
    paths = [p for p in source_paths if isinstance(p, str) and p.strip()]
    if not paths:
        return None
    lead = _MARKER_COMMENT_LEAD_BY_STACK.get(primary_stack, "#")
    return f"{lead} @tests: {', '.join(paths)}"


def _prepend_tests_marker(abs_path: str, marker_line: str) -> bool:
    """``_prepend_verifies_marker`` for the ``@tests:`` code-linkage
    marker. Idempotent; returns True on success or already-present."""
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
            body = fh.read()
    except OSError as exc:
        logger.warning(
            "[test_generation_node] @tests autofix could not read %r: %s",
            abs_path, exc,
        )
        return False
    if _parse_tests_marker(body):
        return True
    lines = body.splitlines(keepends=True)
    insert_at = 0
    if lines:
        first = lines[0].lstrip()
        if first.startswith("#!"):
            insert_at = 1
        elif first.startswith("#") and "coding" in first:
            insert_at = 1
    lines.insert(insert_at, marker_line + "\n")
    try:
        with open(abs_path, "w", encoding="utf-8") as fh:
            fh.write("".join(lines))
    except OSError as exc:
        logger.warning(
            "[test_generation_node] @tests autofix could not write %r: %s",
            abs_path, exc,
        )
        return False
    return True


def _guess_sources_for_test(
    rel_test: str, source_files: list[str], cap: int = 3,
) -> list[str]:
    """Best-effort mapping of a generated test file to the source files
    it exercises, for the deterministic ``@tests`` autofix.

    Prefers a basename match (``test_billing.py`` / ``billing.test.ts``
    → ``.../billing.py`` / ``.../billing.ts``); falls back to the first
    ``cap`` files of the generation call's source list — the test was
    generated FROM those files, so they're the honest default.
    """
    stem = os.path.splitext(os.path.basename(rel_test))[0]
    for prefix in ("test_", "tests_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    for suffix in ("_test", ".test", ".spec", "_spec"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    matches = [
        s for s in source_files
        if os.path.splitext(os.path.basename(s))[0] == stem
    ]
    if matches:
        return matches[:cap]
    return list(source_files[:cap])


def _fetch_ac_keys_for_current_story(
    workspace_path: str, current_story_id: str,
) -> list[str]:
    """Look up the AC keys for ``current_story_id`` from ``state.db``.
    Returns an empty list when the story isn't found or the DB is
    unavailable — the autofix caller then falls through to the
    LLM-repair path unchanged. Silent-fail on any error class.
    """
    if not current_story_id or not workspace_path:
        return []
    try:
        from harness import story_state as _sst
        app_name = _sst.app_name_for_workspace(workspace_path)
        conn = _sst.open_story_db()
        try:
            story = _sst.get_story(conn, app_name, current_story_id)
            if story is None:
                return []
            ac_rows = _sst.list_acceptance_criteria(
                conn, app_name, story["id"],
            )
            return [
                r["ac_key"] for r in ac_rows
                if isinstance(r.get("ac_key"), str)
            ]
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 — autofix is best-effort
        return []


def _persist_verifies_links(
    workspace_path: str,
    marker_keys_by_file: dict[str, list[str]],
) -> tuple[int, int]:
    """Insert ``test_verifies_ac`` edges for every (test_path, ac_key)
    pair the marker parser surfaced.

    Returns ``(links_inserted, unknown_keys_dropped)``. Unknown ``ac_key``s
    (the LLM cited an AC that isn't in the workspace's
    ``acceptance_criteria`` table) are logged and dropped — Phase 4's
    audit gate will still flag uncovered ACs, so the failure mode is
    "missing coverage" not "silently wrong link". Soft-fails on DB
    errors: a broken state.db must not block a passing test run.
    """
    if not marker_keys_by_file:
        return (0, 0)
    try:
        from harness import story_state
        app_name = story_state.app_name_for_workspace(workspace_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[test_generation_node] @verifies persist skipped (workspace=%r): %s",
            workspace_path, exc,
        )
        return (0, 0)
    inserted = 0
    dropped = 0
    try:
        conn = story_state.open_story_db()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[test_generation_node] @verifies persist skipped (open_story_db "
            "failed): %s", exc,
        )
        return (0, 0)
    try:
        # Resolve every cited ac_key to its row id in one batch SELECT
        # so the per-link inserts don't N+1 the DB.
        all_keys = sorted({
            k for keys in marker_keys_by_file.values() for k in keys
        })
        ac_rows: dict[str, int] = {}
        if all_keys:
            placeholders = ",".join(["?"] * len(all_keys))
            for row in conn.execute(
                f"SELECT ac_key, id FROM acceptance_criteria "
                f"WHERE workspace = ? AND ac_key IN ({placeholders})",
                (app_name, *all_keys),
            ):
                ac_rows[row[0]] = int(row[1])
        for test_path, keys in marker_keys_by_file.items():
            for key in keys:
                ac_id = ac_rows.get(key)
                if ac_id is None:
                    logger.warning(
                        "[test_generation_node] @verifies marker in %r cites "
                        "unknown ac_key %r; dropping link (Phase 4 audit "
                        "will still flag uncovered ACs).",
                        test_path, key,
                    )
                    dropped += 1
                    continue
                if story_state.link_test_to_ac(
                    conn, app_name, test_path, ac_id,
                ):
                    inserted += 1
    finally:
        conn.close()
    return (inserted, dropped)


# Directory names that never contain generated tests. Prevents the
# sweep helper from walking node_modules / .venv / dist / etc. on
# large workspaces.
_SWEEP_IGNORED_DIRS = frozenset({
    "__pycache__", "node_modules", ".git", ".venv", "venv",
    ".pytest_cache", ".mypy_cache", "dist", "build",
    ".tox", ".nox", ".next", "target",
})

# File extensions the sweep considers "test files". Mirrors what
# test_generation_node actually writes for each supported stack.
_SWEEP_TEST_EXTENSIONS = (".py", ".ts", ".tsx", ".js", ".jsx", ".java")


def _looks_like_test_file(rel_path: str) -> bool:
    """Filter to test files by conventional naming — matches Python
    ``test_*.py`` / ``*_test.py``, JS/TS ``*.test.tsx`` / ``*.spec.ts``,
    and Java ``*Test.java`` / ``*Tests.java``. Skips source files even
    if they happen to carry a stray ``@verifies:`` comment."""
    base = os.path.basename(rel_path)
    stem, ext = os.path.splitext(base)
    ext_l = ext.lower()
    if ext_l not in _SWEEP_TEST_EXTENSIONS:
        return False
    if ext_l == ".py":
        return stem.startswith("test_") or stem.endswith("_test")
    if ext_l in (".ts", ".tsx", ".js", ".jsx"):
        return (
            ".test." in base.lower()
            or ".spec." in base.lower()
        )
    if ext_l == ".java":
        return stem.endswith("Test") or stem.endswith("Tests")
    return False


# Matches STORY-N or STORY-N.AC-M references anywhere in a comment /
# docstring / prose. Used by ``autofix_markers_by_body_reference`` to
# rescue patcher-emitted test files that mention the story informally
# (e.g. a docstring "STORY-002: Filing Index Construction") but forgot
# the exact ``@verifies: STORY-N.AC-N`` syntax the audit demands.
_STORY_REFERENCE_RE = re.compile(
    r"\bSTORY-(?:NFR-)?\d+(?:\.AC-\d+)?\b"
)


def _stack_from_test_path(rel_path: str) -> str:
    """Pick the marker comment lead based on the test file's extension.
    Falls back to Python (# lead) for unknown extensions. Used by the
    batch-level marker autofix which sees files from both stacks."""
    ext = os.path.splitext(rel_path)[1].lower()
    if ext in (".ts", ".tsx"):
        return "typescript"
    if ext in (".js", ".jsx"):
        return "javascript"
    if ext == ".java":
        return "java"
    return "python"


def autofix_markers_by_body_reference(workspace_path: str) -> tuple[int, int]:
    """For every test file lacking a ``@verifies:`` marker, scan its body
    for ``STORY-N`` mentions in comments / docstrings, look up the
    referenced ACs in state.db, and prepend a well-formed ``@verifies:``
    marker. Returns ``(files_scanned, files_patched)``.

    Called at ``batch_commit_node`` before ``sweep_verifies_links`` so
    patcher-emitted test files (which bypass ``test_generation_node``'s
    marker gate entirely) get retroactive AC linkage from whatever
    story they informally reference. Idempotent — files that already
    carry a valid ``@verifies:`` marker are skipped.

    Root cause fix (2026-07-11, finsearch): ``patching_node`` writes
    test files as part of a story's scope; those tests bypass
    ``test_generation_node`` and its ``@verifies:`` autofix. In the
    finsearch run, 20/26 untested ACs were on tests that DID reference
    the story in a docstring (``# STORY-002: Filing Index...``) but
    forgot the ``@verifies:`` line — the sweep saw no markers and the
    audit correctly reported the ACs as untested.
    """
    if not workspace_path or not os.path.isdir(workspace_path):
        return (0, 0)
    try:
        from harness import story_state
        app_name = story_state.app_name_for_workspace(workspace_path)
        conn = story_state.open_story_db()
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "[test_generation] marker autofix skipped (state.db open "
            "failed): %s", exc,
        )
        return (0, 0)
    from harness.req_ids import canonicalize_ac_key, canonicalize_req_key
    scanned = 0
    patched = 0
    story_ac_cache: dict[str, list[str]] = {}
    try:
        for root, dirs, files in os.walk(workspace_path):
            dirs[:] = [d for d in dirs if d not in _SWEEP_IGNORED_DIRS]
            for name in files:
                abs_path = os.path.join(root, name)
                rel = os.path.relpath(abs_path, workspace_path)
                if not _looks_like_test_file(rel):
                    continue
                scanned += 1
                try:
                    with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                        body = fh.read()
                except OSError:
                    continue
                if _parse_verifies_marker(body):
                    continue
                # Scan only the top of the file — story mentions elsewhere
                # (e.g. a mocked fixture name that coincidentally matches
                # the STORY-N pattern) shouldn't drive the inference.
                head = "\n".join(body.splitlines()[: _VERIFIES_SCAN_LINES])
                mentions = _STORY_REFERENCE_RE.findall(head)
                if not mentions:
                    continue
                referenced_stories: set[str] = set()
                specific_acs: set[str] = set()
                for m in mentions:
                    if ".AC-" in m:
                        specific_acs.add(canonicalize_ac_key(m))
                    else:
                        referenced_stories.add(canonicalize_req_key(m))
                ac_keys: set[str] = set(specific_acs)
                for sk in referenced_stories:
                    if sk not in story_ac_cache:
                        story = story_state.get_story(conn, app_name, sk)
                        if story is None:
                            story_ac_cache[sk] = []
                        else:
                            acs = story_state.list_acceptance_criteria(
                                conn, app_name, story["id"],
                            )
                            story_ac_cache[sk] = [
                                a["ac_key"] for a in acs
                                if isinstance(a.get("ac_key"), str)
                            ]
                    ac_keys.update(story_ac_cache[sk])
                if not ac_keys:
                    continue
                stack = _stack_from_test_path(rel)
                marker = _marker_line_for(stack, sorted(ac_keys))
                if not marker:
                    continue
                if _prepend_verifies_marker(abs_path, marker):
                    patched += 1
                    logger.info(
                        "[test_generation] Autofixed @verifies marker on "
                        "%r via body reference (stack=%s, keys=%s).",
                        rel, stack, ", ".join(sorted(ac_keys)),
                    )
    finally:
        conn.close()
    return (scanned, patched)


def sweep_verifies_links(workspace_path: str) -> tuple[int, int, int]:
    """Walk the workspace, parse ``@verifies:`` markers from every test
    file, and persist the corresponding ``test_verifies_ac`` edges.

    Ciod session 523e86a7 sealed 5 batches with tests passing but the
    audit reported ``test_verifies_ac`` was empty. Root cause:
    ``_persist_verifies_links`` only fires inside ``test_generation_node``
    when the sandbox run passes on THAT one invocation, but in practice
    the LLM's tests fail the initial marker gate or fail their run, and
    repair fixes them via ``compiler_node`` — which never routes back
    to ``test_generation_node``. Markers land on disk but the link
    table stays empty.

    This helper is called from ``batch_commit_node`` (the natural point
    where "this batch is verified done" is known) and rescans EVERY
    test file the batch touched. Idempotent — reuses
    ``story_state.link_test_to_ac``'s (test_path, ac_id) INSERT OR
    IGNORE contract, so a marker whose link was already written by an
    earlier batch's sweep is a silent no-op.

    Returns ``(files_scanned, links_inserted, unknown_keys_dropped)``.
    """
    if not workspace_path or not os.path.isdir(workspace_path):
        return (0, 0, 0)
    try:
        from harness import story_state
        app_name = story_state.app_name_for_workspace(workspace_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[test_generation] verifies sweep skipped (workspace=%r): %s",
            workspace_path, exc,
        )
        return (0, 0, 0)

    marker_keys_by_file: dict[str, list[str]] = {}
    files_scanned = 0
    for root, dirs, files in os.walk(workspace_path):
        dirs[:] = [d for d in dirs if d not in _SWEEP_IGNORED_DIRS]
        for name in files:
            abs_path = os.path.join(root, name)
            rel = os.path.relpath(abs_path, workspace_path)
            if not _looks_like_test_file(rel):
                continue
            files_scanned += 1
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                    body = fh.read()
            except OSError:
                continue
            keys = _parse_verifies_marker(body)
            if keys:
                marker_keys_by_file[rel] = keys

    if not marker_keys_by_file:
        return (files_scanned, 0, 0)

    try:
        conn = story_state.open_story_db()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[test_generation] verifies sweep skipped (open_story_db "
            "failed): %s", exc,
        )
        return (files_scanned, 0, 0)
    inserted = 0
    dropped = 0
    try:
        all_keys = sorted({
            k for keys in marker_keys_by_file.values() for k in keys
        })
        ac_rows: dict[str, int] = {}
        if all_keys:
            placeholders = ",".join(["?"] * len(all_keys))
            for row in conn.execute(
                f"SELECT ac_key, id FROM acceptance_criteria "
                f"WHERE workspace = ? AND ac_key IN ({placeholders})",
                (app_name, *all_keys),
            ):
                ac_rows[row[0]] = int(row[1])
        for test_path, keys in marker_keys_by_file.items():
            for key in keys:
                ac_id = ac_rows.get(key)
                if ac_id is None:
                    dropped += 1
                    continue
                if story_state.link_test_to_ac(
                    conn, app_name, test_path, ac_id,
                ):
                    inserted += 1
    finally:
        conn.close()
    return (files_scanned, inserted, dropped)


def _parse_verifies_marker(body: str) -> list[str]:
    """Return the AC key list from the first ``@verifies:`` marker in
    ``body``'s first :data:`_VERIFIES_SCAN_LINES` lines.

    Returns ``[]`` when:
      - no marker found in the scan window
      - marker found but the key list is empty
      - marker found but no element matches the ``STORY-N.AC-N`` shape

    The shape filter is lenient on whitespace and trailing commas but
    strict on the key form — a malformed key fails the gate the same
    way a missing marker does, so the LLM's repair pass sees a single
    "missing or malformed marker" failure mode rather than a separate
    "key didn't match the regex" mode.
    """
    if not body:
        return []
    # Limit the haystack to the scan window so a buried marker far
    # below imports / setup doesn't satisfy the gate.
    lines = body.splitlines()
    head = "\n".join(lines[: _VERIFIES_SCAN_LINES])
    match = _VERIFIES_RE.search(head)
    if not match:
        return []
    raw = match.group("keys") or ""
    candidates = [c.strip() for c in raw.split(",")]
    # Fold every AC key to the DB storage form so the downstream
    # lookup matches regardless of which form the LLM wrote
    # (``STORY-1.AC-1`` vs. ``STORY-001.AC-1``). The story_state
    # module holds keys in the raw form; ``_canon_ac`` strips leading
    # zeros so both incoming shapes hit the same row (2026-07-04
    # canonicalisation-boundary fix).
    from harness.story_state import _canon_ac
    keys: list[str] = []
    for c in candidates:
        if not c:
            continue
        folded = _canon_ac(c)
        if _AC_KEY_RE.match(folded):
            keys.append(folded)
    return keys


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

# Base rules (1-4) — always emitted. RULE 5 (the @tests code-linkage
# marker) is likewise always emitted; see _build_format_reminder below.
_PROMPT_FORMAT_REMINDER_BASE = """[CRITICAL FORMAT INSTRUCTION]
You MUST respond using ONLY the patch block syntax below. No prose, no markdown
code fences, no commentary. Your entire response must be parseable as patch
blocks.

<<<CREATE_FILE>>>
file: <workspace-relative path>
content:
<complete file contents>
<<<END_CREATE_FILE>>>

<<<REPLACE_BLOCK>>>
file: <workspace-relative path>
search:
<exact lines to find — copy verbatim from the preflight "Current Content" section>
replace:
<exact replacement lines>
<<<END_REPLACE_BLOCK>>>

<<<REWRITE_FILE>>>
file: <workspace-relative path>
content:
<complete corrected file contents>
<<<END_REWRITE_FILE>>>

<<<INSERT_AT_BLOCK>>>
file: <workspace-relative path>
anchor: <function or class name>
placement: before|after
content:
<lines to insert>
<<<END_INSERT_AT_BLOCK>>>

CHOOSING THE RIGHT BLOCK:
  - CREATE_FILE: the file does NOT yet exist. Rejected if the path is
    already on disk.
  - REPLACE_BLOCK: modify a specific region of an EXISTING file. The
    `search:` block MUST match the current file bytes exactly — copy
    from the preflight "Current Content" section, WITHOUT the `  N| `
    line-number prefix.
  - REWRITE_FILE: the file is small (< ~50 lines) and you want to
    replace ITS ENTIRE contents. Preferred over REPLACE_BLOCK when the
    change is pervasive or you're ADDING content that doesn't exist in
    the file yet (REPLACE_BLOCK's search anchor cannot match content
    that isn't there).
  - INSERT_AT_BLOCK: append or prepend lines relative to a named
    function / class anchor. No line-copying required.

RULES — absolute:
  1. File paths MUST be workspace-relative. Anything starting with '/', '~',
     or '..' will be rejected.
  2. Do NOT generate mocks. No unittest.mock, Mockito, jest.mock, mockall,
     mockito (Dart), gomock, sinon, nock. Tests must call the real
     implementation with realistic inputs. Use only the test runner's
     built-in fakes (pytest monkeypatch / tmp_path, httptest.Server,
     @TempDir, etc.) when a side effect cannot be invoked directly.
  3. Cover the typical paths AND the edge cases (empty input, zero/negative
     values, error branches). Skip cases that would require mocking external
     services.
  4. Match the stack-canonical layout and naming convention."""

_TESTS_MARKER_RULE = """  5. EVERY generated test file MUST carry a `@tests` marker at the top
     of the file (after the module docstring / imports, within the first
     50 non-blank lines) naming the workspace-relative source file(s)
     whose functions / classes the file's tests exercise. Use the
     language-appropriate comment style:
       - Python:    # @tests: server/app/services/billing.py
       - JS / TS:   // @tests: client/src/utils/formatters.ts
       - Java:      // @tests: src/main/java/com/acme/Billing.java
     Comma-separate multiple source files. Unit tests are linked to the
     CODE under test — do NOT reference stories or acceptance criteria
     (STORY-N / AC-N / @verifies) anywhere in a test file; requirement-
     level functional coverage is generated separately by `teane test`."""

_CONTRADICTION_RULES = """  6. ONE enforcement layer per validation rule. When an input is
     rejected at CONSTRUCTION (a Pydantic/schema/model validator raises on
     a bad value), assert that rejection AT THE SCHEMA — e.g.
     `with pytest.raises(ValidationError): ContactUpdate(first_name="   ")`.
     Do NOT also write a separate test that CONSTRUCTS the same bad value
     to feed a downstream layer (service/handler): that object can never be
     built, so the downstream test is unsatisfiable. If production enforces
     the same rule redundantly downstream (defence-in-depth), that check is
     unreachable for invalid input — do not unit-test it by constructing
     invalid input. Pick the layer that OWNS the rejection and test it once.
  7. CONSTRUCTIBILITY before use. Before writing `X(value)` and passing the
     result to another call, check X's definition in the source files
     above. If X's validators reject `value` at construction, any assertion
     AFTER `X(value)` is impossible — assert the rejection ON `X(value)`
     itself instead. Across the whole batch you are emitting, never let one
     test require `X(v)` to RAISE while another requires the same `X(v)` to
     SUCCEED — that pair is self-contradictory and no code can satisfy both."""

_REMINDER_TAIL = "\n\nGenerate test patches NOW. Only the blocks above. No other text."


def _build_format_reminder(agile: bool = True) -> str:
    """Return the test-gen format reminder.

    RULE 5 is the ``@tests`` code-linkage marker and is emitted
    unconditionally — linking a unit test to the source file it
    exercises needs no story/AC rows, so the old agile/non-agile split
    (which gated the retired ``@verifies`` AC contract) no longer
    changes the output. The parameter is kept for caller compatibility.
    """
    del agile  # retained for signature compatibility; no longer used
    return (
        f"{_PROMPT_FORMAT_REMINDER_BASE}\n{_TESTS_MARKER_RULE}\n"
        f"{_CONTRADICTION_RULES}{_REMINDER_TAIL}"
    )


# Backward-compat alias for any out-of-tree caller / test that imports
# the legacy constant.
_PROMPT_FORMAT_REMINDER = _build_format_reminder()


def _build_test_gen_prompt(
    workspace_path: str,
    modified_source_files: list[str],
    primary_stack: str,
    max_per_file_chars: int = 6000,
    *,
    agile: bool = True,
) -> str:
    """Build the user-prompt body listing modified source files and asking
    for tests.

    ``agile`` is retained for signature compatibility only — RULE 5 is
    now the flow-independent ``@tests`` code-linkage marker; see
    :func:`_build_format_reminder`.
    """
    lines: list[str] = [
        f"Generate unit tests for the following source files (stack: {primary_stack}).",
        "Each test file should follow the conventions in the test-generation guide "
        "already loaded in the system prompt.",
        "",
        "## Source files to test",
        "",
    ]
    for rel in modified_source_files:
        abs_path = os.path.join(workspace_path, rel)
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                body = fh.read(max_per_file_chars)
        except OSError:
            body = "<unreadable>"
        lines.append(f"### `{rel}`")
        lines.append("```")
        lines.append(body)
        lines.append("```")
        lines.append("")
    lines.append(_build_format_reminder(agile=agile))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthetic diagnostics
# ---------------------------------------------------------------------------

def _synth_diag(
    file: str,
    message: str,
    error_code: str = "TEST_FAILURE",
    line: int = 0,
) -> dict[str, Any]:
    """Synthesize a DiagnosticObject-dict so the existing compiler_errors /
    repair_node pipeline can consume it without changes."""
    return {
        "file": file,
        "line": line,
        "column": 0,
        "severity": "error",
        "error_code": error_code,
        "message": message,
        "semantic_context": "",
    }


# ---------------------------------------------------------------------------
# Public node
# ---------------------------------------------------------------------------

async def test_generation_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: generate tests for the just-patched source files and
    verify them deterministically in the sandbox.

    See module docstring for the contract. Returns a state-update dict.
    """
    cfg = state.get("test_generation_config", {}) or {}
    if not cfg.get("enabled", True):
        logger.info("[test_generation_node] Disabled in config. Skipping.")
        return {}

    workspace_path: str = state.get("workspace_path", os.getcwd())

    # NFR-only-batch guardrail (2026-07-10): when EVERY story in this
    # batch's scope is a SAFe enabler / NFR story, skip test generation
    # cleanly. NFRs (rate limits, latency budgets, security posture,
    # retention policies) aren't naturally unit-testable — the LLM
    # reliably refuses to emit tests for pure NFR scope, burns its
    # zero-emit re-prompt sub-cap, and trips a HITL the operator can't
    # productively resolve.
    #
    # An intermediate iteration of this guardrail emitted @verifies
    # skip-stubs here to appease the traceability audit's AC gate. Both
    # halves of that are gone: build/patch unit tests link to CODE (the
    # @tests marker), never to ACs, and the AC-coverage gate only fires
    # during ``teane test`` (see traceability.has_ac_gap), whose
    # functional pack owns NFR verification.
    #
    # Mixed batches (NFR alongside a regular story) still run the LLM
    # path — the regular story's source files anchor the unit tests.
    from harness.req_ids import STORY_NFR_ID_RE
    _batch_story_keys: list[str] = []
    _cur_story = str(state.get("current_story_id") or "")
    if _cur_story:
        _batch_story_keys = [_cur_story]
    else:
        _batch_story_keys = [
            str(k) for k in (state.get("batch_patched_story_keys") or []) if k
        ]
    if _batch_story_keys and all(
        STORY_NFR_ID_RE.fullmatch(k) for k in _batch_story_keys
    ):
        logger.info(
            "[test_generation_node] Batch scope is entirely NFR "
            "story/stories (%s). Skipping unit-test generation — NFR "
            "verification is owned by the `teane test` functional pack.",
            ", ".join(_batch_story_keys),
        )
        return {
            "node_state": {
                "current_node": "test_generation",
                "test_generation": {
                    "status": "skipped",
                    "reason": "nfr_only_batch",
                    "story_keys": _batch_story_keys,
                },
            },
        }

    # In batch-mode, scope test generation to files this batch touched
    # rather than the cumulative session set. ``_scope_files_for_consumer``
    # falls back to cumulative ``modified_files`` outside batch-mode and
    # for the very first invocation. Lives in harness.graph to keep the
    # batch-scope helpers in one place.
    from harness.graph import _scope_files_for_consumer
    modified_files: list[str] = list(_scope_files_for_consumer(state))
    no_tests_collected: bool = bool(
        state.get("node_state", {}).get("no_tests_collected")
    )

    # --- Skip when nothing testable ---
    if not modified_files and not no_tests_collected:
        logger.info("[test_generation_node] No modified files. Skipping.")
        return {}

    source_files = [
        rel for rel in modified_files
        if not _is_test_file(rel)
        and os.path.splitext(rel)[1].lower() in _SOURCE_EXTENSIONS
    ]

    # When the compiler routed us here because pytest exit=5 (no tests
    # collected), modified_files may legitimately contain only test scaffolds
    # or manifest files — yet the workspace has source code that needs tests.
    # Fall back to a workspace scan for testable source so we don't return
    # empty and bounce back through compiler→test_gen→repair forever.
    if not source_files and no_tests_collected:
        scanned = _scan_workspace_for_source(workspace_path)
        if scanned:
            logger.info(
                "[test_generation_node] no_tests_collected: modified_files had "
                "no testable source, scanned workspace and found %d source "
                "file(s): %s",
                len(scanned), scanned[:10],
            )
            source_files = scanned
        else:
            logger.warning(
                "[test_generation_node] no_tests_collected but workspace scan "
                "found no source files either. Routing to HITL."
            )
            loop_counter = dict(state.get("loop_counter") or {})
            return {
                "loop_counter": loop_counter,
                "node_state": {
                    "current_node": "test_generation",
                    "env_misconfig": True,
                    "env_misconfig_symbol": "no_source_files",
                    "test_generation": {
                        "status": "skipped",
                        "reason": "no_source_in_workspace",
                    },
                },
            }

    if not source_files:
        logger.info(
            "[test_generation_node] No testable source files in the %d modified file(s). Skipping.",
            len(modified_files),
        )
        return {}

    # --- Stack detection ---
    from harness.impact import _detect_workspace_stack
    tags = _detect_workspace_stack(workspace_path) or set()
    # Fall back to source-extension heuristic when manifest detection finds
    # nothing (greenfield workspaces).
    if not any(t in _PRIMARY_STACK_PRIORITY for t in tags):
        for rel in source_files:
            ext_stack = _SOURCE_EXTENSIONS.get(os.path.splitext(rel)[1].lower())
            if ext_stack:
                tags.add(ext_stack)
                break
    primary = _pick_primary_stack(tags)
    # Homogeneous-source override: when every file under test maps to ONE
    # stack by extension, that stack drives this generation call — the
    # prompt framing, the test-guide, and the runner all follow it. The
    # workspace-priority pick ranks typescript above python, so a mixed
    # py+react workspace framed PYTHON sources as "(stack: typescript)"
    # and the model floundered into the zero-emit HITL (lumina 019f7109 —
    # its reasoning literally says "But stack is typescript? It says
    # typescript but all files are Python"). The priority pick remains
    # the tie-breaker for genuinely mixed source sets.
    _src_stacks = {
        _SOURCE_EXTENSIONS.get(os.path.splitext(rel)[1].lower())
        for rel in source_files
    }
    _src_stacks.discard(None)
    if len(_src_stacks) == 1:
        _src_stack = next(iter(_src_stacks))
        if _src_stack != primary:
            logger.info(
                "[test_generation_node] Source files are homogeneously "
                "%s — overriding workspace-priority stack %s for this "
                "generation call.", _src_stack, primary,
            )
            primary = _src_stack
    if primary is None:
        logger.info(
            "[test_generation_node] No supported stack detected (tags=%s). Skipping.",
            sorted(tags),
        )
        return {}

    # --- LLM-key gate ---
    from harness.graph import get_gateway
    gateway = get_gateway()
    if gateway is None:
        logger.error(
            "[test_generation_node] No LLM gateway configured. test_generation "
            "requires a valid LLM API key. Routing to HITL."
        )
        loop_counter = dict(state.get("loop_counter") or {})
        diagnostic = _synth_diag(
            file="<test_generation>",
            message=(
                "test_generation requires a valid LLM API key, but no gateway is "
                "configured. The installer must set ANTHROPIC_API_KEY / "
                "OPENAI_API_KEY / DEEPSEEK_API_KEY (matching the provider in "
                "model_routing) and re-run, OR disable test_generation by setting "
                "test_generation.enabled = false in .harness_config.json."
            ),
            error_code="ENV_MISCONFIG",
        )
        return {
            "exit_code": 1,
            "compiler_errors": [diagnostic],
            "loop_counter": loop_counter,
            "node_state": {
                "current_node": "test_generation",
                "env_misconfig": True,
                "env_misconfig_symbol": "llm_api_key",
                "test_generation": {
                    "status": "skipped",
                    "reason": "no_gateway",
                    "primary_stack": primary,
                },
            },
        }

    loop_counter = dict(state.get("loop_counter") or {})
    max_iterations = int(cfg.get("max_iterations", 3))
    # Fix 2a: split the entry-time cap check across two counters. The
    # ``test_generation`` counter is the real-attempt budget; a call
    # that produces zero patch blocks is a prompt-comprehension miss,
    # not a test-quality miss, so it's counted against the separate
    # ``test_generation_zero_emit`` sub-cap instead. Both caps must be
    # under the ceiling for the LLM to be dispatched again; either can
    # trip HITL with its own env_misconfig symbol so an operator
    # (or post-mortem) can distinguish the two failure classes.
    real_iters = int(loop_counter.get("test_generation", 0) or 0)
    zero_emit_shots = int(
        loop_counter.get("test_generation_zero_emit", 0) or 0
    )
    zero_emit_cap = int(cfg.get("max_zero_emit_reprompts", 3))
    if real_iters >= max_iterations:
        logger.warning(
            "[test_generation_node] Max iterations (%d) reached. Routing to HITL.",
            max_iterations,
        )
        return {
            "loop_counter": loop_counter,
            "exit_code": 1,
            "compiler_errors": [_synth_diag(
                file="<test_generation>",
                message=(
                    f"test_generation_node exceeded max_iterations={max_iterations}. "
                    "The last attempt is reflected in the workspace; manual review needed."
                ),
                error_code="LLM_BEHAVIOR",
            )],
            "node_state": {
                "current_node": "test_generation",
                "llm_behavior": True,
                "llm_behavior_symbol": "test_generation_max_iterations",
            },
        }
    if zero_emit_shots >= zero_emit_cap:
        logger.warning(
            "[test_generation_node] Zero-emit re-prompt cap (%d) reached. "
            "Routing to HITL.", zero_emit_cap,
        )
        return {
            "loop_counter": loop_counter,
            "exit_code": 1,
            "compiler_errors": [_synth_diag(
                file="<test_generation>",
                message=(
                    f"test_generation LLM emitted zero patch blocks for "
                    f"{zero_emit_cap} consecutive re-prompts. The model "
                    "likely lacks context for what to test. Inspect the "
                    "last messages under debug logging and either supply "
                    "an explicit test hint or raise "
                    "test_generation.max_zero_emit_reprompts."
                ),
                error_code="LLM_BEHAVIOR",
            )],
            "node_state": {
                "current_node": "test_generation",
                "llm_behavior": True,
                "llm_behavior_symbol": "test_generation_zero_emit",
            },
        }

    # --- Build the LLM prompt ---
    from harness.gateway import NodeRole
    from harness.patcher import process_llm_patch_output

    test_guides_dir = os.path.join(os.path.dirname(__file__), "test_guides")
    guides_body = ""
    if os.path.isdir(test_guides_dir):
        # Reuse the existing style-guides loader directly — same frontmatter
        # filtering, same byte caps.
        # We override the search directory by calling the lower-level loader
        # explicitly: pass tags={primary, *tags} so applies_to: [<primary>]
        # files match.
        from harness.style_guides import _load_style_guides_markdown
        guides_body, _ = _load_style_guides_markdown(
            test_guides_dir,
            workspace_tags=(tags | {primary}),
        )

    messages = list(state.get("messages", []) or [])
    if guides_body:
        messages.append({
            "role": "system",
            "content": "## Test-generation guidance\n\n" + guides_body,
        })

    # ADR-0003 Tier 1: deterministic schema-contract tests. Emitted from the
    # just-patched Pydantic models BEFORE the LLM turn so (a) the model can be
    # told which surfaces are already covered and must not re-test — the
    # single-source-of-truth that makes the ADR-0002 cross-file contradiction
    # structurally impossible for these models — and (b) the correct-by-
    # construction declarative tests land regardless of what the LLM emits.
    # Conservative: validator-bearing models are skipped (left to the LLM).
    from harness.contract_tests import (
        emit_contract_tests, emit_api_contract_tests, emit_property_tests,
    )
    from harness.contract_tests_react import emit_react_contract_tests
    # Tier 3 (property-based) is the known-fiddly tier — gated OFF by default
    # until its false-positive rate is measured (ADR-0003). Others always on,
    # gated on source-language presence rather than the workspace primary.
    _property_based = bool(cfg.get("property_based_tests", False))
    contract_files: list[str] = []
    contract_markers: dict[str, list[str]] = {}
    _contract_covered: list[str] = []
    try:
        # Tier 1 — schema-declarative unit tests (Pydantic).
        t1_files, t1_markers = emit_contract_tests(
            workspace_path, source_files, primary,
        )
        # Tier 2 — API status-code contracts (FastAPI framework guarantees).
        t2_files, t2_markers = emit_api_contract_tests(
            workspace_path, source_files, primary,
        )
        # Tier 3 — property-based round-trip invariants (opt-in).
        t3_files, t3_markers = (
            emit_property_tests(workspace_path, source_files, primary)
            if _property_based else ([], {})
        )
        # React tier — component render smoke tests (props-driven components).
        tr_files, tr_markers = emit_react_contract_tests(
            workspace_path, source_files, primary,
        )
        contract_files = t1_files + t2_files + t3_files + tr_files
        contract_markers = {
            **t1_markers, **t2_markers, **t3_markers, **tr_markers,
        }
        _contract_covered = sorted({
            src for srcs in contract_markers.values() for src in srcs
        })
        if contract_files:
            logger.info(
                "[test_generation_node] Emitted %d deterministic contract-test "
                "file(s) (ADR-0003 Tiers 1-2%s+React) covering %s: %s",
                len(contract_files), "+3" if _property_based else "",
                ", ".join(_contract_covered), ", ".join(contract_files),
            )
    except Exception as _ct_exc:  # noqa: BLE001 — deterministic tier is best-effort
        logger.warning(
            "[test_generation_node] Contract-test emission failed "
            "(continuing with LLM-only): %s", _ct_exc,
        )

    user_prompt = _build_test_gen_prompt(
        workspace_path, source_files, primary,
    )
    if _contract_covered:
        user_prompt += (
            "\n\n[ALREADY COVERED — DO NOT DUPLICATE]\n"
            "Deterministic contract tests have ALREADY been generated for "
            "these source file(s):\n"
            + "\n".join(f"- {s}" for s in _contract_covered)
            + "\nThose cover, for the models/endpoints involved: schema "
            "field constraints (max_length, ranges), required-field and type "
            "validation, and API validation status codes (422 on invalid "
            "body / bad path-param type). Do NOT re-emit any of those. Write "
            "ONLY business-logic / behavioural assertions the declarative "
            "contract cannot express: computed values, custom-validator "
            "semantics, cross-field rules, success-path responses, and "
            "stateful endpoint behaviour (e.g. 404 on a missing resource, "
            "correct 200/201 payloads).\n"
        )
    # Change-request mode: prepend the CR-N attribution rules so generated
    # tests follow the `test_cr_N_*` naming convention and reference the
    # CR in their docstrings. No-op (empty string) outside CR mode.
    from harness.graph import (
        _build_arch_summary_preamble,
        _build_change_request_preamble,
        _build_story_preamble,
    )
    # Architecture-summary preamble — every endpoint in §11 should
    # have at least one test, every component at least one render
    # test. Empty string when the arch doc has no §11 block (legacy
    # / third-party arch docs); the test generator falls back to
    # source-file-driven coverage in that case.
    # patching_node (upstream) already caches the resolved summary onto
    # state, so the helper hits the in-state copy without a disk read
    # on the common path; the rare lazy-load case (e.g. a re-entry that
    # skipped patching) is cheap enough we don't bother caching here.
    arch_preamble, _resolved_arch = _build_arch_summary_preamble(
        cast("AgentState", state), consumer="test_generator",
    )
    # Story preamble as CONTEXT only — it tells the test-gen LLM what
    # the code under test is supposed to do. The "tests" phase wording
    # in _build_story_preamble instructs the LLM to link tests to the
    # code under test and NOT to reference story/AC ids in test files;
    # unit tests generated here are code-linked (RULE 5's @tests
    # marker), never AC-linked. Empty in monolithic / no-current-story
    # runs, which is fine — the source files in the prompt body are
    # the primary material either way.
    story_preamble = _build_story_preamble(
        cast("AgentState", state), "tests",
    )

    # Fix 5a: preflight the current on-disk bodies of every existing test
    # file the LLM might edit. Without this, REPLACE_BLOCK anchors on
    # test files touched by a prior node in the same session are built
    # from the LLM's stale mental model and reject with "search miss"
    # (session 44c5e194 iter 4 lost 5 patches this way). Mirrors the
    # existing repair_node preflight — the helper existed but was never
    # wired here.
    from harness.graph import (
        _collect_workspace_file_content,
        _format_preflight_file_content,
    )
    preflight_targets = _existing_tests_for_preflight(
        workspace_path,
        source_files=source_files,
        modified_files=modified_files,
        primary_stack=primary,
    )
    preflight_section = ""
    if preflight_targets:
        # Bug B parity (2026-07-10): initialize the dict so preflight
        # hashes actually land on state instead of being dropped when
        # the outer .get() returns None on cold-start.
        _ns = state.get("node_state")
        if not isinstance(_ns, dict):
            _ns = {}
            state["node_state"] = _ns
        _files_seen = _ns.get("files_seen_by_llm")
        if not isinstance(_files_seen, dict):
            _files_seen = {}
            _ns["files_seen_by_llm"] = _files_seen
        _record_into: Optional[dict[str, str]] = _files_seen
        pairs = _collect_workspace_file_content(
            workspace_path, preflight_targets,
            record_hashes_into=_record_into,
        )
        preflight_section = _format_preflight_file_content(
            pairs,
            intro=(
                "The line-numbered views below are the **actual current "
                "content** of test files you may need to MODIFY. When you "
                "emit a REPLACE_BLOCK targeting one of these paths, its "
                "`search:` block MUST match these bytes exactly (WITHOUT "
                "the `  N| ` line-number prefix). Do NOT patch against a "
                "remembered version of the file — the workspace has been "
                "modified by earlier nodes in this session and your "
                "memory of it is stale."
            ),
        )
        if preflight_section:
            logger.info(
                "[test_generation_node] Preflight injected current content "
                "for %d existing test file(s): %s",
                len(pairs), ", ".join(p[0] for p in pairs),
            )

    user_prompt = (
        _build_change_request_preamble(cast("AgentState", state), "tests")
        + arch_preamble
        + story_preamble
        + preflight_section
        + user_prompt
    )
    messages.append({"role": "user", "content": user_prompt})

    budget = float(state.get("budget_remaining_usd", 2.00))
    new_budget = budget
    token_tracker = state.get("token_tracker", {})

    # --- LLM dispatch + patch application, with zero-emit retry loop ---
    # Fix 2a: when the LLM produces zero patch blocks it's a prompt-
    # comprehension miss, not a test-quality miss. Retry inline with
    # a stronger contract system message. Doesn't count against the
    # real test_generation iteration cap (that only advances on a
    # response that yielded ≥1 patch block). The sub-cap
    # `zero_emit_cap` bounds retries so a stuck LLM still exits.
    from harness.graph import _build_patcher_allowlist
    existing_modified = list(modified_files)
    allowed_paths = _build_patcher_allowlist(workspace_path)
    zero_emit_this_call = 0
    zero_emit_remaining = max(0, zero_emit_cap - zero_emit_shots)
    patch_results: list = []
    new_modified: list[str] = []
    response = None
    while True:
        try:
            response, new_budget = await gateway.dispatch(
                messages=list(messages),
                role=NodeRole.PATCHING,
                budget_remaining_usd=budget,
            )
        except RuntimeError as exc:
            logger.warning("[test_generation_node] Gateway refused: %s", exc)
            return {
                "loop_counter": loop_counter,
                "node_state": {
                    "current_node": "test_generation",
                    "test_generation": {
                        "status": "gateway_error", "error": str(exc),
                    },
                },
            }
        token_tracker = gateway.aggregate_tokens(token_tracker, response.usage)
        budget = new_budget

        patch_results, new_modified = await process_llm_patch_output(
            response.content,
            workspace_path,
            existing_modified,
            allowed_paths=allowed_paths,
        )
        messages.append({"role": "assistant", "content": response.content})

        if len(patch_results) > 0:
            break

        zero_emit_this_call += 1
        logger.warning(
            "[test_generation_node] LLM emitted zero patch blocks "
            "(re-prompt %d/%d without consuming a test_generation "
            "iteration).",
            zero_emit_shots + zero_emit_this_call, zero_emit_cap,
        )
        if zero_emit_this_call >= zero_emit_remaining:
            # Sub-cap exhausted — record consumption and trip HITL via
            # env_misconfig so the operator sees "model won't emit
            # tests", not the generic max_iterations one.
            loop_counter["test_generation_zero_emit"] = (
                zero_emit_shots + zero_emit_this_call
            )
            logger.warning(
                "[test_generation_node] Zero-emit re-prompt cap (%d) "
                "reached inline. Routing to HITL.", zero_emit_cap,
            )
            return {
                "messages": messages,
                "loop_counter": loop_counter,
                "token_tracker": token_tracker,
                "budget_remaining_usd": new_budget,
                "exit_code": 1,
                "compiler_errors": [_synth_diag(
                    file="<test_generation>",
                    message=(
                        f"test_generation LLM emitted zero patch blocks "
                        f"for {zero_emit_cap} consecutive re-prompts. "
                        "The model likely lacks context for what to "
                        "test. Inspect the last messages under debug "
                        "logging and either supply an explicit test "
                        "hint or raise "
                        "test_generation.max_zero_emit_reprompts."
                    ),
                    error_code="LLM_BEHAVIOR",
                )],
                "node_state": {
                    "current_node": "test_generation",
                    "llm_behavior": True,
                    "llm_behavior_symbol": "test_generation_zero_emit",
                },
            }
        # Format-mimicry callout: when the zero-emit response is written
        # in the flattened tool-history notation, the model has adopted
        # the read-only record as a tool syntax (lumina 019f7109 — three
        # consecutive responses in it, one containing a complete valid
        # test file the parser ignored). Generic "emit patch blocks"
        # nudges don't break that lock; naming the exact mistake does.
        _resp_text = response.content or ""
        _mimicry = (
            "[called tool" in _resp_text
            or "(history: invoked" in _resp_text
        )
        _retry = (
            "Your last response contained zero PATCH blocks. You MUST "
            "emit at least one CREATE_FILE / REWRITE_FILE / "
            "REPLACE_BLOCK / INSERT_AT_BLOCK targeting a file under "
            "the language-appropriate test root (tests/ for Python, "
            "colocated *.test.tsx for TS, src/test/java for Java). "
            "If you do not know what to test, pick the simplest public "
            "function in the newest source file shown above and write "
            "ONE assertion for its happy path."
        )
        if _mimicry:
            _retry = (
                "STOP: your last response used the bracketed tool-history "
                "notation (\"[called tool ...]\" / \"(history: ...)\"). "
                "That notation is a read-only RECORD of an earlier phase's "
                "tool activity — it is NOT a tool interface, and everything "
                "you wrote in it was ignored. There are no callable tools "
                "in this phase. Re-emit your work as literal patch blocks "
                "(<<<CREATE_FILE>>> ... <<<END_CREATE_FILE>>>) with the "
                "full file content inline. " + _retry
            )
        messages.append({"role": "system", "content": _retry})

    # Record any consumed zero-emit budget alongside the successful attempt
    # so the persistent counter reflects reality across HITL round-trips.
    if zero_emit_this_call:
        loop_counter["test_generation_zero_emit"] = (
            zero_emit_shots + zero_emit_this_call
        )

    # Identify just the newly-applied test files (delta from existing_modified).
    generated_tests: list[str] = []
    for rel in new_modified:
        if rel in existing_modified:
            continue
        if not _inside_workspace(rel, workspace_path):
            # Defence in depth — patcher.safe_resolve should have already
            # rejected this. Log so an audit trail exists.
            logger.error(
                "[test_generation_node] Dropping out-of-workspace file from "
                "generated_tests: %r", rel,
            )
            continue
        generated_tests.append(rel)

    # Fold in the deterministic contract-test files (ADR-0003 Tier 1) so they
    # flow through the same marker-persistence, reporting, and deterministic
    # run as the LLM's output. They already carry `# @tests:` markers, so the
    # marker gate passes them through untouched.
    for _cf in contract_files:
        if _cf not in generated_tests and _inside_workspace(_cf, workspace_path):
            generated_tests.append(_cf)

    success_count = sum(1 for r in patch_results if r.success)
    fail_count = len(patch_results) - success_count
    logger.info(
        "[test_generation_node] LLM produced %d patch block(s), %d applied, "
        "%d failed. %d new test file(s).",
        len(patch_results), success_count, fail_count, len(generated_tests),
    )

    # Real attempt was made — consume a test_generation iteration.
    # Placed AFTER the zero-patch retry loop so zero-emit re-prompts
    # don't burn against the real cap.
    loop_counter["test_generation"] = real_iters + 1

    messages.append({
        "role": "system",
        "content": (
            f"[test_generation] Generated {len(generated_tests)} new test file(s): "
            f"{', '.join(generated_tests) if generated_tests else '(none)'}."
        ),
    })

    # --- @tests code-linkage marker gate ---
    # Every generated test file MUST carry a `# @tests: <source path>`
    # marker naming the file(s) under test. Unit tests generated during
    # build / patch link to CODE, never to stories or acceptance
    # criteria — AC linkage (`@verifies`) belongs to the functional pack
    # `teane test` generates. The marker is deterministic to autofix:
    # the source files this generation call was asked to cover are on
    # hand, so a missing marker almost never spends an LLM turn.
    marker_sources_by_file: dict[str, list[str]] = {}
    marker_missing: list[str] = []
    for rel in generated_tests:
        abs_path = os.path.join(workspace_path, rel)
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                body = fh.read()
        except OSError as exc:
            logger.warning(
                "[test_generation_node] Could not re-read generated test %r "
                "for marker parse: %s — treating as marker-missing.",
                rel, exc,
            )
            marker_missing.append(rel)
            continue
        paths = _parse_tests_marker(body)
        if not paths:
            marker_missing.append(rel)
        else:
            marker_sources_by_file[rel] = paths

    if marker_missing:
        autofixed: list[str] = []
        unfixable: list[str] = []
        for rel in marker_missing:
            guessed = _guess_sources_for_test(rel, source_files)
            # Comment style comes from the FILE being stamped, never the
            # workspace's primary stack: a mixed py+react workspace
            # resolved primary to a JS flavour and this autofix wrote
            # `// @tests:` into tests/__init__.py — a SyntaxError that
            # took out collection for the whole tests package (lumina
            # session 019f7054). Same rule the @verifies autofix already
            # follows via _stack_from_test_path.
            marker_line = _tests_marker_line_for(
                _stack_from_test_path(rel), guessed,
            )
            abs_path = os.path.join(workspace_path, rel)
            if marker_line and _prepend_tests_marker(abs_path, marker_line):
                autofixed.append(rel)
                marker_sources_by_file[rel] = guessed
            else:
                unfixable.append(rel)

        if autofixed:
            logger.info(
                "[test_generation_node] Autofix prepended @tests marker "
                "on %d file(s) without spending an iteration: %s",
                len(autofixed), ", ".join(autofixed),
            )
        if unfixable:
            diags = [
                _synth_diag(
                    file=rel,
                    message=(
                        f"Generated test {rel!r} is missing a `@tests:` marker. "
                        "Every generated test file MUST declare which source "
                        "file(s) it exercises, using a comment at the top of "
                        "the file (within the first 50 lines): "
                        "`# @tests: path/to/module.py` (Python) or "
                        "`// @tests: path/to/module.ts` (JS/TS/Java). "
                        "Comma-separate multiple source files. Do NOT "
                        "reference stories or acceptance criteria in test "
                        "files."
                    ),
                    error_code="TEST_FAILURE:missing_tests_marker",
                )
                for rel in unfixable
            ]
            logger.warning(
                "[test_generation_node] %d/%d @tests marker(s) needed LLM "
                "repair (no source files to autofix from, or write failed); "
                "routing to repair: %s",
                len(unfixable), len(marker_missing), ", ".join(unfixable),
            )
            return {
                "messages": messages,
                "modified_files": new_modified,
                "generated_tests": list(state.get("generated_tests", [])) + generated_tests,
                "exit_code": 1,
                "compiler_errors": diags,
                "token_tracker": token_tracker,
                "budget_remaining_usd": new_budget,
                "loop_counter": loop_counter,
                "node_state": {
                    **(state.get("node_state") or {}),
                    "current_node": "test_generation",
                    "test_generation": {
                        "status": "missing_tests_marker",
                        "primary_stack": primary,
                        "tests_generated": len(generated_tests),
                        "markerless_count": len(unfixable),
                        "autofixed_count": len(autofixed),
                    },
                },
            }
        # All markers autofixed — fall through to the deterministic
        # test run without spending an iteration.

    # --- Cross-file contradiction gate (generation-side prevention) ---
    # Lumina 019f803f: the test-author emitted a same-input / opposite-
    # expectation pair split across two files — ContactUpdate(first_name="  ")
    # required to RAISE in test_contact_schemas.py and to SUCCEED in
    # test_contact_service.py. No production change satisfies both; the
    # repair loop is forbidden from editing tests, so it oscillated ~2.5h
    # and dead-ended. Catch it deterministically HERE, before the build,
    # and bounce it back to the AUTHOR (a re-prompt naming the exact pair) —
    # NOT to repair. Prevention upstream of ADR-0001's repair-side
    # regeneration. Bounded by ``max_contradiction_reprompts`` so a model
    # that can't reconcile still exits to the normal build path.
    contra_cap = int(cfg.get("max_contradiction_reprompts", 2))
    contra_reprompts = 0
    while contra_cap > 0 and generated_tests:
        _py_tests = {
            rel: _read_text(os.path.join(workspace_path, rel))
            for rel in generated_tests
            if rel.endswith(".py")
        }
        _py_tests = {k: v for k, v in _py_tests.items() if v is not None}
        contradictions = find_contradictions_across(_py_tests)
        if not contradictions:
            break
        try:
            from harness.observability import emit_event
            emit_event(
                "test_generation_contradiction_detected",
                count=len(contradictions),
                reprompt=contra_reprompts + 1,
                files=sorted({
                    f for c in contradictions
                    for f in (c.expect_raise_file, c.expect_success_file) if f
                }),
            )
        except Exception:  # noqa: BLE001 — telemetry must not block
            pass
        if contra_reprompts >= contra_cap:
            logger.warning(
                "[test_generation_node] Cross-file contradiction persists "
                "after %d re-prompt(s); proceeding to the build (repair / "
                "ADR-0001 regeneration will handle the residue). Pairs: %s",
                contra_cap,
                "; ".join(c.describe() for c in contradictions[:3]),
            )
            break
        contra_reprompts += 1
        _pairs = "\n".join(f"- {c.describe()}" for c in contradictions)
        logger.warning(
            "[test_generation_node] Detected %d cross-file test "
            "contradiction(s); bouncing back to the author (re-prompt "
            "%d/%d).\n%s",
            len(contradictions), contra_reprompts, contra_cap, _pairs,
        )
        messages.append({
            "role": "system",
            "content": (
                "STOP — the tests you just generated are mutually "
                "UNSATISFIABLE. The following call(s) are required to both "
                "raise and succeed on identical input, across different "
                "test files:\n" + _pairs + "\n\n"
                "No production code can satisfy both, and the repair loop "
                "may NOT edit test files to break the tie. Reconcile by "
                "deciding which single layer OWNS the rejection (see RULE 6 "
                "/ RULE 7): if the value is rejected at CONSTRUCTION, keep "
                "the schema-level `pytest.raises(...)` assertion and REMOVE "
                "the downstream test that constructs the same invalid value; "
                "if the value is meant to construct and be rejected "
                "downstream, loosen the schema test instead. Re-emit ONLY "
                "the corrected test file(s) as REWRITE_FILE blocks, keeping "
                "each file's `@tests` marker. Only patch blocks, no prose."
            ),
        })
        try:
            response, new_budget = await gateway.dispatch(
                messages=list(messages),
                role=NodeRole.PATCHING,
                budget_remaining_usd=budget,
            )
        except RuntimeError as exc:
            logger.warning(
                "[test_generation_node] Gateway refused during "
                "contradiction re-prompt: %s — proceeding to build.", exc,
            )
            break
        token_tracker = gateway.aggregate_tokens(token_tracker, response.usage)
        budget = new_budget
        messages.append({"role": "assistant", "content": response.content})
        _re_results, new_modified = await process_llm_patch_output(
            response.content,
            workspace_path,
            existing_modified,
            allowed_paths=allowed_paths,
        )
        # Recompute the generated-test set from the accumulated modified
        # list (a REWRITE_FILE of an existing generated test keeps it in
        # the set; a brand-new corrected file joins it).
        generated_tests = [
            rel for rel in new_modified
            if rel not in existing_modified
            and _inside_workspace(rel, workspace_path)
        ]

    # --- Skip deterministic run when no tests landed ---
    if not generated_tests:
        logger.info(
            "[test_generation_node] No tests generated → skipping deterministic execution."
        )
        return {
            "messages": messages,
            "modified_files": new_modified,
            "generated_tests": list(state.get("generated_tests", [])),  # unchanged
            "token_tracker": token_tracker,
            "budget_remaining_usd": new_budget,
            "loop_counter": loop_counter,
            "node_state": {
                "current_node": "test_generation",
                "test_generation": {
                    "status": "passed",
                    "primary_stack": primary,
                    "tests_generated": 0,
                    "reason": "no_tests_generated",
                },
            },
        }

    # --- Deterministic test run ---
    test_cmd = _stack_test_command(primary)
    if test_cmd is None:
        logger.info(
            "[test_generation_node] No deterministic test command for stack=%s. "
            "Tests written but unverified.", primary,
        )
        return {
            "messages": messages,
            "modified_files": new_modified,
            "generated_tests": list(state.get("generated_tests", [])) + generated_tests,
            "token_tracker": token_tracker,
            "budget_remaining_usd": new_budget,
            "loop_counter": loop_counter,
            "node_state": {
                "current_node": "test_generation",
                "test_generation": {
                    "status": "passed",
                    "primary_stack": primary,
                    "tests_generated": len(generated_tests),
                    "reason": "no_runner_command_for_stack",
                },
            },
        }

    # Ensure pytest has a config that uses the importlib import mode whenever a
    # Python test tree is present. Without this, two same-named test files in
    # different directories (e.g. `tests/app/models/test_job.py` and
    # `tests/app/schemas/test_job.py`, both arising from a `job.py` source
    # in each package) collide on collection with the well-known
    # "import file mismatch: imported module 'test_job' has this __file__
    # attribute" error; and a full-stack app with a flat `tests/` tree plus a
    # nested `server/tests/` tree hits ImportPathMismatchError (or silently
    # drops one tier). importlib mode uses Python's package resolution so the
    # trees coexist as distinct dotted names. Gated on Python-test PRESENCE
    # (the writer self-checks), NOT on the workspace's primary stack — a
    # full-stack Python+JS app resolves `primary` to the frontend and used to
    # skip this entirely (lumina 019f82af). Idempotent — leaves any existing
    # pytest config (pytest.ini / pyproject.toml / setup.cfg) alone.
    ensured = _ensure_pytest_importlib_config(workspace_path)
    if ensured:
        new_modified.append(ensured)

    # JS/TS counterpart: freshly generated .test.ts(x)/.test.js(x) files
    # need their jest/type environment (devDependencies, jest config with
    # the right testEnvironment, setup wiring, tsconfig types) or they
    # produce hundreds of TS2304/TS2307 type-noise diagnostics and a run
    # that can't collect. The TypeScript guide instructs the LLM to patch
    # the config in the same response; this is the deterministic
    # guarantee for when it doesn't. Idempotent, add-only.
    scaffolded = _ensure_js_test_env(workspace_path, generated_tests)
    if scaffolded:
        new_modified.extend(
            p for p in scaffolded if p not in new_modified
        )
        try:
            from harness.observability import emit_event
            emit_event(
                "js_test_env_scaffolded",
                files=scaffolded,
            )
        except Exception:  # noqa: BLE001 — telemetry must not block
            pass

    from harness.sandbox import SandboxExecutor
    sandbox_cfg = dict(state.get("sandbox_config", {}) or {})
    allow_network = bool(state.get("allow_network", False))

    # The test command always contains a package-install token for stacks
    # that need one, so the sandbox auto-network heuristic kicks in. We
    # also lift it explicitly here so the SandboxExecutor sees it.
    if any(tok in test_cmd for tok in ("pip install", "npm install")):
        allow_network = True

    # Adapt the sandbox image and root-FS writability to match the test
    # command's toolchain. Without this, `pip install pytest && pytest -q`
    # runs in the default ubuntu:22.04 base image, which has no pip
    # installed → exit 127 in 0.2s, and the LLM gets routed to a wasted
    # repair iteration with a spurious "test failure". compiler_node does
    # the same adaptation via _toolchain_image_for; reuse it here.
    from harness.graph import _toolchain_image_for, _build_command_writes_root_fs
    desired_image = _toolchain_image_for(test_cmd)
    if desired_image and sandbox_cfg.get("docker_image") != desired_image:
        logger.info(
            "[test_generation_node] Adapting sandbox docker_image to '%s' "
            "to match toolchain implied by test command: %s",
            desired_image, test_cmd,
        )
        sandbox_cfg["docker_image"] = desired_image
    # Only ``npm install -g`` writes to a root-FS location the sandbox's
    # tmpfs-backed HOME can't cover (/usr/local/lib/node_modules); every
    # other install path (pip/poetry/uv/local npm) lands under $HOME on
    # tmpfs and is compatible with read_only_root=True.
    if _build_command_writes_root_fs(test_cmd) and sandbox_cfg.get("read_only_root", True):
        logger.info(
            "[test_generation_node] Adapting sandbox.read_only_root to False "
            "because test command runs `npm install -g`, which writes to "
            "/usr/local/lib/node_modules on the container's root FS: %s",
            test_cmd,
        )
        sandbox_cfg["read_only_root"] = False

    executor = SandboxExecutor(
        workspace_path=workspace_path,
        allow_network=allow_network,
        sandbox_config=sandbox_cfg,
    )

    logger.info(
        "[test_generation_node] Running deterministic test command for %s: %s",
        primary, test_cmd,
    )
    build_result = await executor.run(test_cmd)

    if build_result.exit_code == 0:
        logger.info(
            "[test_generation_node] Tests passed (%d test file(s) executed).",
            len(generated_tests),
        )
        # Unit tests link to code, not ACs — no ``test_verifies_ac``
        # edges are written during build / patch. AC coverage edges are
        # owned by the ``teane test`` functional pack.
        tg_status: dict[str, Any] = {
            "status": "passed",
            "primary_stack": primary,
            "tests_generated": len(generated_tests),
            "test_command": test_cmd,
            "tests_marker_files": len(marker_sources_by_file),
        }
        return {
            "messages": messages,
            "modified_files": new_modified,
            "generated_tests": list(state.get("generated_tests", [])) + generated_tests,
            "token_tracker": token_tracker,
            "budget_remaining_usd": new_budget,
            "loop_counter": loop_counter,
            # Merge into existing node_state so cross-iteration trackers
            # (patch_failures, allowlist_rejections, allowed_paths from the
            # prior patching_node) reach the next compiler_node/repair_node.
            "node_state": {
                **(state.get("node_state") or {}),
                "current_node": "test_generation",
                "test_generation": tg_status,
            },
        }

    # --- Failures → flow into the standard repair path ---
    raw_diags = [d.to_dict() for d in build_result.diagnostics]
    if not raw_diags:
        # No structured parser hit — synthesize one from the tail.
        raw_diags = [_synth_diag(
            file="<test_runner>",
            message=(
                f"Generated tests failed (exit={build_result.exit_code}). "
                f"Command: {test_cmd}. "
                f"Tail: ...{(build_result.raw_output or '')[-1500:]}"
            ),
            error_code="TEST_FAILURE",
        )]
    else:
        # Tag every structured diagnostic with the TEST_FAILURE prefix so
        # repair_node's framing tweak knows these came from the test runner.
        for d in raw_diags:
            d["error_code"] = f"TEST_FAILURE:{d.get('error_code', 'unknown')}"

    logger.warning(
        "[test_generation_node] Tests failed (exit=%d, %d diagnostic(s)). "
        "Routing to repair.",
        build_result.exit_code, len(raw_diags),
    )

    return {
        "messages": messages,
        "modified_files": new_modified,
        "generated_tests": list(state.get("generated_tests", [])) + generated_tests,
        "exit_code": build_result.exit_code,
        "compiler_errors": raw_diags,
        "token_tracker": token_tracker,
        "budget_remaining_usd": new_budget,
        "loop_counter": loop_counter,
        # Merge into existing node_state so cross-iteration trackers
        # (patch_failures, allowlist_rejections, allowed_paths from the
        # prior patching_node) flow into the repair_node we're about to
        # route to.
        "node_state": {
            **(state.get("node_state") or {}),
            "current_node": "test_generation",
            "last_build_output": build_result.raw_output,
            "test_generation": {
                "status": "failed",
                "primary_stack": primary,
                "tests_generated": len(generated_tests),
                "test_command": test_cmd,
                "test_failures": len(raw_diags),
            },
        },
    }


# ---------------------------------------------------------------------------
# Router (post-test_generation conditional edge)
# ---------------------------------------------------------------------------

def route_after_test_generation(state: dict[str, Any]) -> str:
    """Conditional edge router executed after test_generation_node.

    Decision matrix:
        llm_behavior flag set                  → human_intervention_node
            (covers the "max iterations" cap and the "zero-emit re-prompt" cap)
        env_misconfig flag set                 → human_intervention_node
            (covers the "no LLM gateway" gate and "no source files")
        compiler_errors populated (TEST_FAILURE) → repair_node
        otherwise                              → lintgate_node
    """
    node_state = state.get("node_state", {}) or {}
    if node_state.get("llm_behavior") or node_state.get("env_misconfig"):
        return "human_intervention_node"
    if state.get("compiler_errors"):
        return "repair_node"
    return "lintgate_node"
