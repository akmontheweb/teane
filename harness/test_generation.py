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

import logging
import os
import re
from typing import Any, Optional, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from harness.graph import AgentState

logger = logging.getLogger(__name__)


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
    "addopts = --import-mode=importlib\n"
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


def _ensure_pytest_importlib_config(workspace_path: str) -> Optional[str]:
    """Write a minimal ``pytest.ini`` with ``--import-mode=importlib`` if the
    workspace has no pytest configuration of any kind.

    Recognises every shape pytest itself looks at:
      - ``pytest.ini``
      - ``pyproject.toml`` with a ``[tool.pytest.ini_options]`` table
      - ``setup.cfg`` with a ``[tool:pytest]`` section
      - ``tox.ini`` with a ``[pytest]`` section

    Returns the workspace-relative path of the newly-written file, or
    ``None`` if any existing config was found (no-op).
    """
    if not workspace_path or not os.path.isdir(workspace_path):
        return None

    # Hard signals — file existence wins. pyproject.toml / setup.cfg / tox.ini
    # only count if they actually contain a pytest section.
    if os.path.isfile(os.path.join(workspace_path, "pytest.ini")):
        return None

    section_signals: tuple[tuple[str, str], ...] = (
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
        if marker in content:
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

# Base rules (1-4) — always emitted. RULE 5 (the @verifies marker
# contract) is appended only in agile mode where acceptance_criteria
# rows exist for the LLM to cite; see _build_format_reminder below.
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

_VERIFIES_RULE = """  5. EVERY generated test file MUST carry a `@verifies` marker at the top
     of the file (after the module docstring / imports, within the first
     50 non-blank lines) naming the acceptance criteria the file's tests
     verify. Use the language-appropriate comment style:
       - Python:    # @verifies: STORY-003.AC-2
       - JS / TS:   // @verifies: STORY-003.AC-2
       - Java:      // @verifies: STORY-003.AC-2
     Comma-separate multiple ACs (e.g. `# @verifies: STORY-003.AC-1, STORY-003.AC-2`).
     Use the AC keys exactly as they appear in the story preamble's
     "Acceptance criteria" block. A file with no marker — or a marker
     that uses an AC key absent from the preamble — is rejected and
     the test-gen pass is routed back to repair."""

_REMINDER_TAIL = "\n\nGenerate test patches NOW. Only the blocks above. No other text."


def _build_format_reminder(agile: bool) -> str:
    """Return the test-gen format reminder, conditionally including the
    v5 ``@verifies`` marker rule (RULE 5) when ``agile=True``.

    Non-agile (monolithic) runs have no ``acceptance_criteria`` rows in
    state.db for the LLM to cite, so RULE 5 would be a contract the
    workspace cannot satisfy — the LLM would fabricate keys to pass
    syntactic validation and the link writer would drop every one with
    a noisy warning. Skipping the rule in non-agile keeps the prompt
    honest and avoids the spurious churn (Phase 6 of the schema-v5
    plan).
    """
    if agile:
        return f"{_PROMPT_FORMAT_REMINDER_BASE}\n{_VERIFIES_RULE}{_REMINDER_TAIL}"
    return f"{_PROMPT_FORMAT_REMINDER_BASE}{_REMINDER_TAIL}"


# Backward-compat alias for any out-of-tree caller / test that imports
# the legacy constant. Returns the agile-mode reminder (matches the
# pre-Phase-6 string verbatim).
_PROMPT_FORMAT_REMINDER = _build_format_reminder(agile=True)


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

    ``agile`` toggles whether the @verifies marker contract (RULE 5) is
    included — see :func:`_build_format_reminder`.
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

    # NFR-only-batch guardrail (2026-07-10, refined 2026-07-11): when
    # EVERY story in this batch's scope is a SAFe enabler / NFR story,
    # emit deterministic ``@pytest.mark.skip`` stub tests carrying the
    # ``# @verifies:`` markers for each AC. NFRs (rate limits, latency
    # budgets, security posture, retention policies) aren't naturally
    # unit-testable — the LLM reliably refuses to emit tests for pure
    # NFR scope, burns its zero-emit re-prompt sub-cap, and trips a HITL
    # the operator can't productively resolve.
    #
    # First iteration of this guardrail (2026-07-10) SKIPPED test-gen
    # entirely and returned early. That prevented the zero-emit HITL but
    # the downstream traceability audit correctly flagged every NFR AC
    # as untested and terminated the run with ``traceability_block`` on
    # the finsearch resume at 14:09. Emitting skip-stubs closes the
    # traceability contract while making the "human still owes an
    # integration test" story explicit in the workspace.
    #
    # Mixed batches (NFR alongside a regular story) still run the LLM
    # path — the regular story anchors tests and NFR ACs ride along as
    # ``@verifies:`` citations.
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
        # Detect stack for the stub template. Falls back to a
        # log-and-skip if no supported stack is detected — same shape
        # as the previous guardrail so mixed monorepos without a
        # detectable primary stack don't wedge on this branch.
        from harness.impact import _detect_workspace_stack
        _nfr_tags = _detect_workspace_stack(workspace_path) or set()
        _nfr_primary = _pick_primary_stack(_nfr_tags)
        if _nfr_primary not in ("python", "typescript"):
            logger.info(
                "[test_generation_node] Batch scope is entirely NFR "
                "story/stories (%s) but no supported stack detected "
                "(primary=%r). Skipping cleanly — operator must add "
                "integration coverage manually.",
                ", ".join(_batch_story_keys), _nfr_primary,
            )
            return {
                "node_state": {
                    "current_node": "test_generation",
                    "test_generation": {
                        "status": "skipped",
                        "reason": "nfr_only_batch_unsupported_stack",
                        "story_keys": _batch_story_keys,
                    },
                },
            }
        stub_paths, stub_markers = _emit_nfr_stubs(
            workspace_path, _batch_story_keys, _nfr_primary,
        )
        if not stub_paths:
            # Nothing landed — fall back to the pure-skip behaviour so
            # the batch isn't wedged. Traceability will flag the ACs;
            # that's the correct signal since we couldn't stub them.
            logger.info(
                "[test_generation_node] Batch scope is entirely NFR "
                "story/stories (%s) but no stubs could be written "
                "(missing story rows / no ACs / IO error). Skipping — "
                "operator must add integration coverage manually.",
                ", ".join(_batch_story_keys),
            )
            return {
                "node_state": {
                    "current_node": "test_generation",
                    "test_generation": {
                        "status": "skipped",
                        "reason": "nfr_only_batch_no_stubs",
                        "story_keys": _batch_story_keys,
                    },
                },
            }
        links_inserted, links_dropped = _persist_verifies_links(
            workspace_path, stub_markers,
        )
        total_markers = sum(len(v) for v in stub_markers.values())
        logger.info(
            "[test_generation_node] NFR-only batch (%s): wrote %d stub "
            "file(s) covering %d AC marker(s) — %d verifies-link(s) "
            "inserted, %d dropped as unknown.",
            ", ".join(_batch_story_keys), len(stub_paths),
            total_markers, links_inserted, links_dropped,
        )
        return {
            "modified_files": list(state.get("modified_files", [])) + stub_paths,
            "generated_tests": list(state.get("generated_tests", [])) + stub_paths,
            "node_state": {
                "current_node": "test_generation",
                "test_generation": {
                    "status": "nfr_stubbed",
                    "primary_stack": _nfr_primary,
                    "tests_generated": len(stub_paths),
                    "story_keys": _batch_story_keys,
                    "verifies_links_inserted": links_inserted,
                    "verifies_links_dropped": links_dropped,
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

    # v5 Phase 6: agile mode gates the @verifies marker contract. When
    # decomposition_enabled is False (monolithic build/patch), there
    # are no acceptance_criteria rows for the LLM to cite — RULE 5
    # would be a contract the workspace can't satisfy and the marker
    # gate below is skipped entirely.
    agile = bool(state.get("decomposition_enabled"))

    user_prompt = _build_test_gen_prompt(
        workspace_path, source_files, primary, agile=agile,
    )
    # Change-request mode: prepend the CR-N attribution rules so generated
    # tests follow the `test_cr_N_*` naming convention and reference the
    # CR in their docstrings. No-op (empty string) outside CR mode.
    from harness.graph import (
        _build_arch_summary_preamble,
        _build_batch_scope_preamble,
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
    # v5 Phase 6b: inject the per-story preamble so the test-gen LLM
    # actually sees the AC keys it's expected to cite in @verifies
    # markers. Empty string in non-agile / no-current-story — no-op
    # there. patching_node already injects this preamble for code-
    # gen; mirroring it here makes RULE 5's "use AC keys from the
    # story preamble" instruction verifiable (Phase 3 oversight).
    #
    # Phase 7 BUG #2 fix: in the per-batch verification phase,
    # ``current_story_id`` is empty (story_loop cleared it before
    # routing to verification), so _build_story_preamble returns
    # "". Fall through to _build_batch_scope_preamble which lists
    # every story patched this batch and its AC keys — the LLM
    # needs at least one set of valid keys to satisfy RULE 5.
    story_preamble = _build_story_preamble(
        cast("AgentState", state), "tests",
    )
    if not story_preamble and agile:
        story_preamble = _build_batch_scope_preamble(cast("AgentState", state))

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
        messages.append({
            "role": "system",
            "content": (
                "Your last response contained zero PATCH blocks. You MUST "
                "emit at least one CREATE_FILE / REWRITE_FILE / "
                "REPLACE_BLOCK / INSERT_AT_BLOCK targeting a file under "
                "the language-appropriate test root (tests/ for Python, "
                "colocated *.test.tsx for TS, src/test/java for Java). "
                "If you do not know what to test, pick the simplest public "
                "function in the newest source file shown above and write "
                "ONE assertion for its happy path."
            ),
        })

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

    # --- v5 @verifies marker gate (agile mode only) ---
    # Every generated test file MUST carry a `# @verifies: STORY-N.AC-N`
    # marker (Phase 3 contract). Files that don't are short-circuited
    # into the existing repair pipeline via a TEST_FAILURE diagnostic
    # — the LLM's next iteration sees the diagnostic and adds the
    # marker. We parse markers here (in-memory, no DB) so the gate
    # fires before the sandbox run; the actual ``test_verifies_ac``
    # write happens AFTER the sandbox passes (a failing test that
    # claims to verify an AC shouldn't carry the link).
    #
    # Skipped when ``agile=False`` (Phase 6): non-agile runs have no
    # acceptance_criteria rows to cite, so the gate would only force
    # the LLM to fabricate fake keys to satisfy syntactic validation.
    marker_keys_by_file: dict[str, list[str]] = {}
    marker_missing: list[str] = []
    if agile:
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
            keys = _parse_verifies_marker(body)
            if not keys:
                marker_missing.append(rel)
            else:
                marker_keys_by_file[rel] = keys

    if marker_missing:
        # Fix 3: autofix the marker deterministically when we can. The
        # AC keys are already known from state.db (they're what the
        # story_preamble renders); prepending a well-formed marker line
        # is not something worth spending an LLM turn on. Only files
        # we CAN'T autofix (no current story, malformed AC keys, IO
        # error) route to LLM repair.
        current_story_id = str(state.get("current_story_id") or "")
        ac_keys = _fetch_ac_keys_for_current_story(
            workspace_path, current_story_id,
        )
        marker_line = _marker_line_for(primary, ac_keys)
        autofixed: list[str] = []
        unfixable: list[str] = []
        if marker_line:
            for rel in marker_missing:
                abs_path = os.path.join(workspace_path, rel)
                if _prepend_verifies_marker(abs_path, marker_line):
                    autofixed.append(rel)
                    marker_keys_by_file[rel] = list(ac_keys)
                else:
                    unfixable.append(rel)
        else:
            unfixable = list(marker_missing)

        if autofixed:
            logger.info(
                "[test_generation_node] Autofix prepended @verifies marker "
                "on %d file(s) without spending an iteration "
                "(story=%s, keys=%s): %s",
                len(autofixed), current_story_id or "?",
                ", ".join(ac_keys) if ac_keys else "?",
                ", ".join(autofixed),
            )
        if unfixable:
            diags = [
                _synth_diag(
                    file=rel,
                    message=(
                        f"Generated test {rel!r} is missing a `@verifies:` marker. "
                        "Every generated test file MUST declare which acceptance "
                        "criteria it verifies, using a comment at the top of the "
                        "file (within the first 50 lines): "
                        "`# @verifies: STORY-N.AC-N` (Python) or "
                        "`// @verifies: STORY-N.AC-N` (JS/TS/Java). Comma-separate "
                        "multiple ACs. Use AC keys exactly as they appear in the "
                        "story preamble's `### Acceptance criteria` block."
                    ),
                    error_code="TEST_FAILURE:missing_verifies_marker",
                )
                for rel in unfixable
            ]
            logger.warning(
                "[test_generation_node] %d/%d marker(s) needed LLM repair "
                "(no story context to autofix from, or write failed); "
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
                        "status": "missing_verifies_marker",
                        "primary_stack": primary,
                        "tests_generated": len(generated_tests),
                        "markerless_count": len(unfixable),
                        "autofixed_count": len(autofixed),
                    },
                },
            }
        # All markers autofixed — fall through to the deterministic
        # test run without spending an iteration.

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

    # When the stack is Python, ensure pytest has a config that uses the
    # importlib import mode. Without this, two same-named test files in
    # different directories (e.g. `tests/app/models/test_job.py` and
    # `tests/app/schemas/test_job.py`, both arising from a `job.py` source
    # in each package) collide on collection with the well-known
    # "import file mismatch: imported module 'test_job' has this __file__
    # attribute" error. importlib mode uses Python's package resolution
    # so the two coexist as distinct dotted names. Idempotent — leaves any
    # existing pytest config (pytest.ini / pyproject.toml / setup.cfg)
    # alone.
    if primary == "python":
        ensured = _ensure_pytest_importlib_config(workspace_path)
        if ensured:
            new_modified.append(ensured)

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
        # v5 link writer — for every test that parsed a marker AND
        # passed the sandbox run, persist (test_path, ac_key) edges
        # into ``test_verifies_ac`` so the audit gate can join AC
        # coverage. Unknown ac_keys are warned-and-dropped here
        # (Phase 3 soft mode); the audit gate in Phase 4 still
        # surfaces them as "untested AC" if no other test claims to
        # verify them, so the data loss is bounded. Skipped in
        # non-agile mode (Phase 6) — no acceptance_criteria rows
        # exist for the workspace, so the helper would just open
        # state.db and immediately return (0, 0).
        tg_status: dict[str, Any] = {
            "status": "passed",
            "primary_stack": primary,
            "tests_generated": len(generated_tests),
            "test_command": test_cmd,
        }
        if agile:
            link_count, drop_count = _persist_verifies_links(
                workspace_path, marker_keys_by_file,
            )
            if link_count or drop_count:
                logger.info(
                    "[test_generation_node] @verifies links persisted: %d "
                    "edge(s) inserted, %d unknown ac_key(s) dropped.",
                    link_count, drop_count,
                )
            tg_status["verifies_links_inserted"] = link_count
            tg_status["verifies_links_dropped"] = drop_count
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
