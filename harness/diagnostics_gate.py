"""
Static Diagnostics Gate — read-only type-checkers between lintgate and compiler.

Runs fast CLI type-checkers over the files the current batch touched (plus
their reverse-dependency closure) and feeds precise diagnostics into the
existing repair loop BEFORE the expensive compiler/test run. A signature
change that breaks an unmodified caller surfaces here in seconds instead of
minutes — and gets repaired autonomously instead of burning toward HITL.

Coverage follows the supported stacks:
    Python     → pyright (``--outputjson``) preferred, mypy fallback
    TS / TSX   → ``tsc --noEmit`` against the nearest tsconfig.json
    Java       → deliberately NOT covered: javac needs the full Maven/Gradle
                 classpath, so a "fast pre-check" degenerates into the build
                 itself, which compiler_node already runs (and JavaParser
                 already parses).

Design invariants:
    - READ-ONLY. Never rewrites a file, so it is safe on the
      repair_node → compiler_node path where lintgate is bypassed to keep
      SEARCH/REPLACE anchors stable.
    - FAIL-OPEN. Missing tool, timeout, crash, malformed output, non-git
      workspace — every infrastructure failure degrades to "this checker
      contributes nothing" and the run proceeds to compiler_node. The gate
      may only ever add signal, never block on its own plumbing.
    - BASELINE-DIFFED. Pre-existing diagnostics in a brownfield repo are
      fingerprinted from a detached HEAD worktree and suppressed; only NEW
      diagnostics route to repair. Warnings are dropped entirely — the gate
      must stay high-signal or it burns repair rounds on style noise.
    - Emits through the EXISTING channels: new diagnostics land in
      ``compiler_errors`` (so repair_node, autofix, and the reflection judge
      consume them unchanged) together with the mandatory fingerprint
      rotation from ``_rotate_diag_fingerprints_delta``.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from harness.observability import emit_event
from harness.parser_registry import MypyParser, PyrightJSONParser, TypeScriptParser
from harness.sandbox import (
    BaseLanguageParser,
    DiagnosticObject,
    run_subprocess_kill_on_timeout,
)

logger = logging.getLogger(__name__)

_PY_EXTS = (".py", ".pyi")
_TS_EXTS = (".ts", ".tsx")
_GATE_EXTS = _PY_EXTS + _TS_EXTS

# Cap on reverse-dependency files pulled in beyond the batch's own set.
_MAX_IMPACTED_EXTRA = 50

_DIGITS_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")

# Untracked toolchain dirs the baseline worktree needs symlinked in so the
# checkers resolve third-party imports the same way they do in the live
# workspace (git worktree materialises tracked files only).
_TOOLCHAIN_LINK_DIRS = ("node_modules", ".venv", "venv")


@dataclass
class CheckerSpec:
    """A fully-resolved checker invocation produced by :func:`detect_checkers`."""
    tool: str                                  # "pyright" | "mypy" | "tsc"
    argv: list[str]
    parser: type[BaseLanguageParser]
    # Workspace-relative paths this invocation is responsible for. For
    # per-file tools the same files appear in argv; for whole-project tsc
    # this is the post-parse filter set.
    scoped_files: list[str] = field(default_factory=list)
    filter_output: bool = False                # True → tsc: filter parsed output


def _norm_rel(path: str, root: str) -> str:
    """Normalise ``path`` to a forward-slash path relative to ``root``."""
    p = path
    if os.path.isabs(p):
        try:
            p = os.path.relpath(p, root)
        except ValueError:
            # Different drive on Windows — keep absolute.
            pass
    return p.replace(os.sep, "/")


def diagnostic_fingerprint(diag: Any, root: str) -> str:
    """Line/column-insensitive identity of a diagnostic.

    ``relpath|error_code|normalized_message`` — no positions, and digits in
    the message are stripped, so a pre-existing error that merely shifted
    lines under a patch still matches its baseline fingerprint.
    """
    if isinstance(diag, DiagnosticObject):
        file, code, msg = diag.file, diag.error_code, diag.message
    else:
        file = str(diag.get("file", ""))
        code = str(diag.get("error_code", ""))
        msg = str(diag.get("message", ""))
    norm_msg = _WS_RE.sub(" ", _DIGITS_RE.sub("", msg)).strip().lower()
    return f"{_norm_rel(file, root)}|{code}|{norm_msg}"


def _find_nearest_tsconfig(ts_file_abs: str, workspace_path: str) -> Optional[str]:
    """Walk up from the file's directory to the workspace root looking for
    a tsconfig.json. Returns the absolute path or None."""
    ws = os.path.abspath(workspace_path)
    d = os.path.dirname(os.path.abspath(ts_file_abs))
    while True:
        candidate = os.path.join(d, "tsconfig.json")
        if os.path.isfile(candidate):
            return candidate
        if os.path.normpath(d) == os.path.normpath(ws):
            return None
        parent = os.path.dirname(d)
        if parent == d or len(parent) < len(ws):
            return None
        d = parent


def detect_checkers(
    files: list[str],
    cfg: dict[str, Any],
    workspace_path: str,
    scratch_dir: str,
) -> list[CheckerSpec]:
    """Resolve which checkers apply to ``files`` and build their argv.

    Python: pyright preferred over mypy (faster cold start, structured
    output, machine-applicable positions); mypy only when pyright is not
    on PATH. TS/TSX: one tsc invocation per distinct nearest tsconfig.
    ``.java`` never matches — see module docstring.
    """
    tools_cfg = (cfg.get("tools") or {})
    ws = os.path.abspath(workspace_path)
    py_files: list[str] = []
    ts_files: list[str] = []
    for f in files:
        abs_f = f if os.path.isabs(f) else os.path.join(ws, f)
        ext = os.path.splitext(abs_f)[1].lower()
        if ext in _PY_EXTS:
            py_files.append(abs_f)
        elif ext in _TS_EXTS:
            ts_files.append(abs_f)

    specs: list[CheckerSpec] = []

    if py_files:
        rels = [_norm_rel(f, ws) for f in py_files]
        if tools_cfg.get("pyright", True) and shutil.which("pyright"):
            specs.append(CheckerSpec(
                tool="pyright",
                argv=["pyright", "--outputjson", *py_files],
                parser=PyrightJSONParser,
                scoped_files=rels,
            ))
        elif tools_cfg.get("mypy", True) and shutil.which("mypy"):
            cache_dir = os.path.join(scratch_dir, "mypy_cache")
            specs.append(CheckerSpec(
                tool="mypy",
                argv=[
                    "mypy", "--no-error-summary", "--show-column-numbers",
                    "--show-error-codes", "--no-color-output",
                    "--follow-imports=silent", "--cache-dir", cache_dir,
                    *py_files,
                ],
                parser=MypyParser,
                scoped_files=rels,
            ))
        else:
            logger.info(
                "[diagnostics_gate] %d Python file(s) in scope but neither "
                "pyright nor mypy is available/enabled — skipping.",
                len(py_files),
            )

    if ts_files:
        if tools_cfg.get("tsc", True) and shutil.which("tsc"):
            by_tsconfig: dict[str, list[str]] = {}
            for f in ts_files:
                tsconfig = _find_nearest_tsconfig(f, ws)
                if tsconfig is None:
                    logger.info(
                        "[diagnostics_gate] No tsconfig.json found above %s "
                        "— skipping tsc for it.", _norm_rel(f, ws),
                    )
                    continue
                by_tsconfig.setdefault(tsconfig, []).append(f)
            for tsconfig, group in by_tsconfig.items():
                specs.append(CheckerSpec(
                    tool="tsc",
                    # --pretty false: newer tsc force-enables pretty output in
                    # some environments, which breaks TypeScriptParser's
                    # parens-coordinate format.
                    argv=["tsc", "--noEmit", "--pretty", "false", "-p", tsconfig],
                    parser=TypeScriptParser,
                    scoped_files=[_norm_rel(f, ws) for f in group],
                    filter_output=True,
                ))
        elif ts_files:
            logger.info(
                "[diagnostics_gate] %d TS file(s) in scope but tsc is not "
                "available/enabled — skipping.", len(ts_files),
            )

    return specs


async def run_checker(
    spec: CheckerSpec,
    workspace_path: str,
    timeout: float,
) -> tuple[list[DiagnosticObject], bool]:
    """Run one checker; return (error-severity diagnostics, timed_out).

    Every failure mode returns an empty list — the gate only ever adds
    signal (see module docstring).
    """
    try:
        exit_code, stdout, stderr, timed_out = await run_subprocess_kill_on_timeout(
            spec.argv, timeout=timeout, cwd=workspace_path,
        )
    except FileNotFoundError:
        logger.info("[diagnostics_gate] %s vanished from PATH mid-run.", spec.tool)
        return [], False
    except Exception as e:  # noqa: BLE001 — fail-open by contract
        logger.warning("[diagnostics_gate] %s crashed: %s", spec.tool, e)
        return [], False
    if timed_out:
        logger.warning(
            "[diagnostics_gate] %s timed out after %.0fs — contributing "
            "nothing this round (fail-open).", spec.tool, timeout,
        )
        return [], True
    output = (
        stdout.decode("utf-8", errors="replace")
        + "\n"
        + stderr.decode("utf-8", errors="replace")
    )
    try:
        diags = spec.parser.parse_diagnostics(output)
    except Exception as e:  # noqa: BLE001
        logger.warning("[diagnostics_gate] %s parse failed: %s", spec.tool, e)
        return [], False
    # Errors only: warnings don't justify a repair round.
    diags = [d for d in diags if str(d.severity).lower() == "error"]
    if spec.filter_output:
        scoped = set(spec.scoped_files)
        diags = [d for d in diags if _norm_rel(d.file, workspace_path) in scoped]
    return diags, False


async def _git(args: list[str], cwd: str, timeout: float = 30) -> tuple[int, str]:
    try:
        rc, out, _err, timed_out = await run_subprocess_kill_on_timeout(
            ["git", *args], timeout=timeout, cwd=cwd,
        )
    except Exception:  # noqa: BLE001
        return 1, ""
    if timed_out:
        return 124, ""
    return rc, out.decode("utf-8", errors="replace").strip()


async def capture_baseline(
    workspace_path: str,
    files: list[str],
    cfg: dict[str, Any],
    scratch_dir: str,
) -> tuple[Optional[set[str]], str]:
    """Fingerprint HEAD's diagnostics for ``files`` via a detached worktree.

    Returns ``(fingerprints, head_sha)`` — or ``(None, sha)`` when the
    worktree route is unavailable (caller degrades to created-only mode).
    The worktree is removed before returning, success or not.
    """
    rc, head_sha = await _git(["rev-parse", "HEAD"], cwd=workspace_path)
    if rc != 0 or not head_sha:
        return None, ""

    wt_path = os.path.join(
        scratch_dir, f"diag_baseline_{head_sha[:8]}_{os.getpid()}",
    )
    rc, _ = await _git(
        ["worktree", "add", "--detach", wt_path, head_sha],
        cwd=workspace_path, timeout=60,
    )
    if rc != 0:
        return None, head_sha

    try:
        # Checkers need untracked toolchain dirs (node_modules, venvs) to
        # resolve third-party imports; the worktree has tracked files only.
        for link_dir in _TOOLCHAIN_LINK_DIRS:
            src = os.path.join(workspace_path, link_dir)
            dst = os.path.join(wt_path, link_dir)
            if os.path.isdir(src) and not os.path.exists(dst):
                try:
                    os.symlink(src, dst, target_is_directory=True)
                except OSError:
                    pass

        ws = os.path.abspath(workspace_path)
        rels = [_norm_rel(f, ws) for f in files]
        existing = [
            os.path.join(wt_path, rel) for rel in rels
            if os.path.isfile(os.path.join(wt_path, rel))
        ]
        if not existing:
            return set(), head_sha

        timeout = float(cfg.get("timeout_seconds", 120))
        fingerprints: set[str] = set()
        for spec in detect_checkers(
            existing, cfg, wt_path, os.path.join(scratch_dir, "baseline"),
        ):
            diags, _timed_out = await run_checker(spec, wt_path, timeout)
            fingerprints.update(diagnostic_fingerprint(d, wt_path) for d in diags)
        return fingerprints, head_sha
    finally:
        await _git(["worktree", "remove", "--force", wt_path], cwd=workspace_path)
        await _git(["worktree", "prune"], cwd=workspace_path)


def _expand_with_impacted(
    files: list[str], workspace_path: str,
) -> list[str]:
    """Add reverse-dependency files (capped) so a signature change surfaces
    the break in its unmodified callers, not just the modified file."""
    try:
        from harness.impact import DependencyGraph
        dg = DependencyGraph(workspace_path)
        impacted = dg.get_impacted_files(files)
    except Exception as e:  # noqa: BLE001 — expansion is best-effort
        logger.debug("[diagnostics_gate] impact expansion failed: %s", e)
        return files
    ws = os.path.abspath(workspace_path)
    seen = {_norm_rel(f, ws) for f in files}
    out = list(files)
    extra = 0
    for impacted_file, _symbols in impacted:
        if extra >= _MAX_IMPACTED_EXTRA:
            logger.info(
                "[diagnostics_gate] impact expansion capped at %d extra "
                "file(s); remainder unchecked this round.", _MAX_IMPACTED_EXTRA,
            )
            break
        rel = _norm_rel(impacted_file, ws)
        abs_f = os.path.join(ws, rel)
        if rel in seen or os.path.splitext(rel)[1].lower() not in _GATE_EXTS:
            continue
        if not os.path.isfile(abs_f):
            continue
        seen.add(rel)
        out.append(abs_f)
        extra += 1
    return out


async def diagnostics_node(state: dict[str, Any]) -> dict[str, Any]:
    """Read-only type-check gate between lintgate_node and compiler_node.

    Emits NEW (non-baseline) error diagnostics into ``compiler_errors`` with
    the mandatory fingerprint rotation, and bumps
    ``loop_counter["diagnostics_rounds_since_compile"]`` so
    ``route_after_diagnostics`` can bound the diagnostics ⇄ repair cycle.
    Clean or degraded runs touch neither channel.
    """
    node_state = dict(state.get("node_state") or {})
    cfg = state.get("diagnostics_config") or {}
    if not cfg.get("enabled", True):
        node_state["diagnostics"] = {"status": "disabled"}
        return {"node_state": node_state}

    from harness.graph import (  # deferred, mirrors lintgate's import pattern
        _rotate_diag_fingerprints_delta,
        _scope_files_for_consumer,
    )
    from harness.lintgate import _classify_files_by_git_status

    t0 = time.monotonic()
    workspace = state.get("workspace_path", os.getcwd())
    ws = os.path.abspath(workspace)

    scoped: list[str] = []
    for f in _scope_files_for_consumer(state):
        abs_f = f if os.path.isabs(f) else os.path.join(ws, f)
        if os.path.splitext(abs_f)[1].lower() in _GATE_EXTS and os.path.isfile(abs_f):
            scoped.append(abs_f)
    if not scoped:
        node_state["diagnostics"] = {"status": "clean", "new": 0, "total": 0}
        return {"node_state": node_state}

    if (cfg.get("scope") or "impacted") == "impacted":
        scoped = _expand_with_impacted(scoped, ws)

    session_id = str(state.get("session_id") or "nosession")[:8]
    scratch_dir = os.path.join(tempfile.gettempdir(), f"teane_diag_{session_id}")
    os.makedirs(scratch_dir, exist_ok=True)

    # --- Baseline -----------------------------------------------------------
    created_files, pre_existing = _classify_files_by_git_status(scoped, ws)
    created_rels = {_norm_rel(f, ws) for f in created_files}
    cached = state.get("diagnostics_baseline") or {}
    baseline_fps: Optional[set[str]] = None
    head_sha = ""
    mode = "created-only"
    if not pre_existing:
        # Greenfield fast path: everything in scope is session-created.
        baseline_fps = set()
        mode = "worktree"  # empty baseline is exact, not degraded
        _rc, head_sha = await _git(["rev-parse", "HEAD"], cwd=ws)
    else:
        _rc, head_sha = await _git(["rev-parse", "HEAD"], cwd=ws)
        if (
            cached.get("mode") == "worktree"
            and head_sha and cached.get("commit") == head_sha
        ):
            baseline_fps = set(cached.get("fingerprints") or [])
            mode = "worktree"
        else:
            fps, sha = await capture_baseline(ws, scoped, cfg, scratch_dir)
            if fps is not None:
                baseline_fps, head_sha, mode = fps, sha, "worktree"

    # --- Current run --------------------------------------------------------
    timeout = float(cfg.get("timeout_seconds", 120))
    specs = detect_checkers(scoped, cfg, ws, scratch_dir)
    all_diags: list[DiagnosticObject] = []
    tools_run: list[str] = []
    timed_out_tools: list[str] = []
    for spec in specs:
        diags, timed_out = await run_checker(spec, ws, timeout)
        tools_run.append(spec.tool)
        if timed_out:
            timed_out_tools.append(spec.tool)
        all_diags.extend(diags)

    if baseline_fps is not None:
        new_diags = [
            d for d in all_diags
            if diagnostic_fingerprint(d, ws) not in baseline_fps
        ]
    else:
        # created-only degraded mode: only session-created files can flag.
        new_diags = [
            d for d in all_diags if _norm_rel(d.file, ws) in created_rels
        ]

    elapsed = round(time.monotonic() - t0, 2)
    status = "partial" if timed_out_tools else ("ok" if specs else "skipped")
    diag_summary = {
        "status": status,
        "new": len(new_diags),
        "total": len(all_diags),
        "baseline": len(all_diags) - len(new_diags),
        "tools": tools_run,
        "timed_out": timed_out_tools,
        "baseline_mode": mode if baseline_fps is not None else "created-only",
        "files_checked": len(scoped),
        "elapsed_s": elapsed,
    }
    node_state["diagnostics"] = diag_summary
    emit_event("diagnostics_gate", **diag_summary)
    logger.info(
        "[diagnostics_node] %s: %d new / %d total diagnostic(s) across %d "
        "file(s) in %.1fs (baseline=%s, tools=%s).",
        status, len(new_diags), len(all_diags), len(scoped), elapsed,
        diag_summary["baseline_mode"], ",".join(tools_run) or "none",
    )

    out: dict[str, Any] = {"node_state": node_state}
    if baseline_fps is not None and head_sha:
        out["diagnostics_baseline"] = {
            "commit": head_sha,
            "mode": "worktree",
            "fingerprints": sorted(baseline_fps),
        }

    if new_diags:
        errors = [d.to_dict() for d in new_diags]
        out["compiler_errors"] = errors
        # Mandatory whenever a gate emits compiler_errors toward repair —
        # see _rotate_diag_fingerprints_delta's docstring.
        out.update(_rotate_diag_fingerprints_delta(state, errors))
        loop_counter = dict(state.get("loop_counter") or {})
        loop_counter["diagnostics_rounds_since_compile"] = (
            int(loop_counter.get("diagnostics_rounds_since_compile") or 0) + 1
        )
        out["loop_counter"] = loop_counter
    return out
