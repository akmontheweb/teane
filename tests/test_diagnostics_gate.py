"""Tests for harness/diagnostics_gate.py — the read-only type-checker gate.

Covers:
    - diagnostic_fingerprint line/column-insensitivity + digit normalisation
    - detect_checkers preference order, config gating, tsconfig discovery,
      and the deliberate .java exclusion
    - run_checker fail-open on timeout / missing tool / parser crash
    - capture_baseline fail-open outside a git repo
    - diagnostics_node: disabled / no-files / clean / new-diagnostics paths,
      including the compiler_errors handoff, fingerprint rotation parity
      with _rotate_diag_fingerprints_delta, and the per-cycle loop counter
    - route_after_diagnostics guard matrix (each guard independently forces
      compiler_node; all-pass routes to repair_node)
"""

from __future__ import annotations


import pytest

import harness.diagnostics_gate as dg
from harness.diagnostics_gate import (
    CheckerSpec,
    detect_checkers,
    diagnostic_fingerprint,
    diagnostics_node,
    run_checker,
)
from harness.parser_registry import PyrightJSONParser, TypeScriptParser
from harness.sandbox import DiagnosticObject


# ---------------------------------------------------------------------------
# diagnostic_fingerprint
# ---------------------------------------------------------------------------

def test_fingerprint_is_line_and_column_insensitive():
    a = DiagnosticObject(file="/w/a.py", line=10, column=4,
                         error_code="X1", message="expected 3 args, got 4")
    b = DiagnosticObject(file="/w/a.py", line=99, column=1,
                         error_code="X1", message="expected 3 args, got 4")
    assert diagnostic_fingerprint(a, "/w") == diagnostic_fingerprint(b, "/w")


def test_fingerprint_normalises_digits_and_paths():
    a = {"file": "/w/pkg/a.py", "error_code": "E", "message": "line 12 bad"}
    b = {"file": "pkg/a.py", "error_code": "E", "message": "line 99  bad"}
    assert diagnostic_fingerprint(a, "/w") == diagnostic_fingerprint(b, "/w")


def test_fingerprint_distinguishes_files_and_codes():
    base = {"file": "a.py", "error_code": "E1", "message": "m"}
    assert diagnostic_fingerprint(base, "/w") != diagnostic_fingerprint(
        {**base, "file": "b.py"}, "/w")
    assert diagnostic_fingerprint(base, "/w") != diagnostic_fingerprint(
        {**base, "error_code": "E2"}, "/w")


# ---------------------------------------------------------------------------
# detect_checkers
# ---------------------------------------------------------------------------

def _which_factory(available: set[str]):
    return lambda tool: f"/usr/bin/{tool}" if tool in available else None


def test_detect_prefers_pyright_over_mypy(tmp_path, monkeypatch):
    monkeypatch.setattr(dg.shutil, "which", _which_factory({"pyright", "mypy"}))
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    specs = detect_checkers([str(f)], {}, str(tmp_path), str(tmp_path / "s"))
    assert [s.tool for s in specs] == ["pyright"]
    assert "--outputjson" in specs[0].argv


def test_detect_falls_back_to_mypy(tmp_path, monkeypatch):
    monkeypatch.setattr(dg.shutil, "which", _which_factory({"mypy"}))
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    specs = detect_checkers([str(f)], {}, str(tmp_path), str(tmp_path / "s"))
    assert [s.tool for s in specs] == ["mypy"]
    assert "--no-error-summary" in specs[0].argv


def test_detect_respects_per_tool_config_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(dg.shutil, "which", _which_factory({"pyright", "mypy"}))
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    cfg = {"tools": {"pyright": False, "mypy": False}}
    assert detect_checkers([str(f)], cfg, str(tmp_path), str(tmp_path / "s")) == []


def test_detect_tsc_requires_tsconfig(tmp_path, monkeypatch):
    monkeypatch.setattr(dg.shutil, "which", _which_factory({"tsc"}))
    src = tmp_path / "src"
    src.mkdir()
    f = src / "App.tsx"
    f.write_text("export {}\n")
    # No tsconfig anywhere → no spec.
    assert detect_checkers([str(f)], {}, str(tmp_path), str(tmp_path / "s")) == []
    # Nearest tsconfig wins and output-filtering is on.
    (tmp_path / "tsconfig.json").write_text("{}")
    specs = detect_checkers([str(f)], {}, str(tmp_path), str(tmp_path / "s"))
    assert len(specs) == 1
    assert specs[0].tool == "tsc" and specs[0].filter_output
    assert specs[0].argv[-1] == str(tmp_path / "tsconfig.json")
    assert specs[0].scoped_files == ["src/App.tsx"]


def test_detect_never_matches_java(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dg.shutil, "which", _which_factory({"pyright", "mypy", "tsc"}))
    f = tmp_path / "Main.java"
    f.write_text("class Main {}\n")
    assert detect_checkers([str(f)], {}, str(tmp_path), str(tmp_path / "s")) == []


# ---------------------------------------------------------------------------
# run_checker — fail-open contract
# ---------------------------------------------------------------------------

def _spec(parser=PyrightJSONParser, filter_output=False, scoped=None):
    return CheckerSpec(tool="pyright", argv=["pyright", "--outputjson"],
                       parser=parser, scoped_files=scoped or [],
                       filter_output=filter_output)


@pytest.mark.asyncio
async def test_run_checker_timeout_contributes_nothing(monkeypatch):
    async def fake_run(argv, *, timeout, cwd=None, env=None,
                       capture_stderr_separately=True):
        return 1, b"", b"", True
    monkeypatch.setattr(dg, "run_subprocess_kill_on_timeout", fake_run)
    diags, timed_out = await run_checker(_spec(), "/w", timeout=1)
    assert diags == [] and timed_out


@pytest.mark.asyncio
async def test_run_checker_missing_tool_fails_open(monkeypatch):
    async def fake_run(argv, *, timeout, cwd=None, env=None,
                       capture_stderr_separately=True):
        raise FileNotFoundError(argv[0])
    monkeypatch.setattr(dg, "run_subprocess_kill_on_timeout", fake_run)
    diags, timed_out = await run_checker(_spec(), "/w", timeout=1)
    assert diags == [] and not timed_out


@pytest.mark.asyncio
async def test_run_checker_drops_warnings_and_filters_scoped(monkeypatch):
    tsc_out = (
        "src/a.ts(1,1): error TS2304: Cannot find name 'x'.\n"
        "src/b.ts(2,1): error TS2304: Cannot find name 'y'.\n"
        "src/a.ts(3,1): warning TS6133: 'z' is declared but never used.\n"
    )
    async def fake_run(argv, *, timeout, cwd=None, env=None,
                       capture_stderr_separately=True):
        return 2, tsc_out.encode(), b"", False
    monkeypatch.setattr(dg, "run_subprocess_kill_on_timeout", fake_run)
    spec = CheckerSpec(tool="tsc", argv=["tsc"], parser=TypeScriptParser,
                       scoped_files=["src/a.ts"], filter_output=True)
    diags, _ = await run_checker(spec, "/w", timeout=1)
    # Warning dropped; b.ts filtered out as unscoped.
    assert [d.file for d in diags] == ["src/a.ts"]
    assert diags[0].severity == "error"


# ---------------------------------------------------------------------------
# capture_baseline — fail-open outside git
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_capture_baseline_non_git_dir_degrades(tmp_path):
    fps, sha = await dg.capture_baseline(
        str(tmp_path), [str(tmp_path / "a.py")], {}, str(tmp_path / "s"))
    assert fps is None and sha == ""


# ---------------------------------------------------------------------------
# diagnostics_node
# ---------------------------------------------------------------------------

def _base_state(tmp_path, **overrides):
    state = {
        "workspace_path": str(tmp_path),
        "modified_files": [],
        "node_state": {},
        "loop_counter": {},
        "session_id": "testsess",
        "diagnostics_config": {"enabled": True, "scope": "modified"},
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_node_disabled_is_noop(tmp_path):
    state = _base_state(tmp_path, diagnostics_config={"enabled": False})
    out = await diagnostics_node(state)
    assert out["node_state"]["diagnostics"]["status"] == "disabled"
    assert "compiler_errors" not in out


@pytest.mark.asyncio
async def test_node_no_gate_files_is_clean(tmp_path):
    java = tmp_path / "Main.java"
    java.write_text("class Main {}\n")
    state = _base_state(tmp_path, modified_files=[str(java)])
    out = await diagnostics_node(state)
    assert out["node_state"]["diagnostics"]["status"] == "clean"
    assert "compiler_errors" not in out


@pytest.mark.asyncio
async def test_node_preserves_existing_node_state(tmp_path):
    state = _base_state(tmp_path, diagnostics_config={"enabled": False},
                        node_state={"lintgate": {"errors": 0}})
    out = await diagnostics_node(state)
    # copy-and-merge, not replace: lintgate results must survive.
    assert out["node_state"]["lintgate"] == {"errors": 0}


async def _run_node_with_fake_checker(tmp_path, monkeypatch, diags,
                                      state_overrides=None):
    """Drive diagnostics_node with a stubbed checker + created-file baseline."""
    py = tmp_path / "svc.py"
    py.write_text("x: int = 'nope'\n")
    monkeypatch.setattr(dg, "detect_checkers", lambda *a, **k: [_spec()])

    async def fake_run_checker(spec, workspace_path, timeout):
        return diags, False
    monkeypatch.setattr(dg, "run_checker", fake_run_checker)

    async def fake_git(args, cwd, timeout=30):
        return 0, "deadbeefcafe"
    monkeypatch.setattr(dg, "_git", fake_git)

    # Everything counts as created-this-session → greenfield fast path.
    import harness.lintgate as lg
    monkeypatch.setattr(
        lg, "_classify_files_by_git_status",
        lambda files, ws: (set(files), set()),
    )
    state = _base_state(tmp_path, modified_files=[str(py)],
                        **(state_overrides or {}))
    return await diagnostics_node(state)


@pytest.mark.asyncio
async def test_node_new_diagnostics_populate_compiler_errors(tmp_path, monkeypatch):
    diag = DiagnosticObject(file="svc.py", line=1, column=10,
                            error_code="reportAssignmentType",
                            message="int is not str", severity="error")
    out = await _run_node_with_fake_checker(tmp_path, monkeypatch, [diag])
    summary = out["node_state"]["diagnostics"]
    assert summary["status"] == "ok" and summary["new"] == 1
    assert out["compiler_errors"][0]["error_code"] == "reportAssignmentType"
    assert out["loop_counter"]["diagnostics_rounds_since_compile"] == 1
    assert out["diagnostics_baseline"]["commit"] == "deadbeefcafe"


@pytest.mark.asyncio
async def test_node_rotation_parity_with_shared_helper(tmp_path, monkeypatch):
    from harness.graph import _rotate_diag_fingerprints_delta
    diag = DiagnosticObject(file="svc.py", line=1, column=1,
                            error_code="E", message="m", severity="error")
    prior_state = {"last_diag_fingerprints": ["E::old"], "last_diag_count": 3}
    out = await _run_node_with_fake_checker(
        tmp_path, monkeypatch, [diag], state_overrides=prior_state)
    expected = _rotate_diag_fingerprints_delta(
        {**prior_state}, out["compiler_errors"])
    for key in ("prior_diag_fingerprints", "last_diag_fingerprints",
                "prior_diag_count", "last_diag_count"):
        assert out[key] == expected[key], key


@pytest.mark.asyncio
async def test_node_clean_run_touches_no_repair_channels(tmp_path, monkeypatch):
    out = await _run_node_with_fake_checker(tmp_path, monkeypatch, [])
    assert out["node_state"]["diagnostics"]["new"] == 0
    for key in ("compiler_errors", "last_diag_fingerprints",
                "prior_diag_fingerprints", "loop_counter"):
        assert key not in out


@pytest.mark.asyncio
async def test_node_baseline_suppresses_preexisting(tmp_path, monkeypatch):
    """A diagnostic whose fingerprint is in the cached worktree baseline
    must not route to repair, even at a different line."""
    py = tmp_path / "svc.py"
    py.write_text("x: int = 'nope'\n")
    old = DiagnosticObject(file="svc.py", line=7, column=1,
                           error_code="E1", message="pre-existing mess",
                           severity="error")
    monkeypatch.setattr(dg, "detect_checkers", lambda *a, **k: [_spec()])

    async def fake_run_checker(spec, workspace_path, timeout):
        # Same identity as baseline, shifted line.
        return [DiagnosticObject(file="svc.py", line=42, column=1,
                                 error_code="E1", message="pre-existing mess",
                                 severity="error")], False
    monkeypatch.setattr(dg, "run_checker", fake_run_checker)

    async def fake_git(args, cwd, timeout=30):
        return 0, "cafebabe"
    monkeypatch.setattr(dg, "_git", fake_git)

    import harness.lintgate as lg
    monkeypatch.setattr(
        lg, "_classify_files_by_git_status",
        lambda files, ws: (set(), set(files)),  # pre-existing file
    )
    baseline_fp = diagnostic_fingerprint(old, str(tmp_path))
    state = _base_state(
        tmp_path, modified_files=[str(py)],
        diagnostics_baseline={"commit": "cafebabe", "mode": "worktree",
                              "fingerprints": [baseline_fp]},
    )
    out = await diagnostics_node(state)
    assert out["node_state"]["diagnostics"]["new"] == 0
    assert out["node_state"]["diagnostics"]["baseline"] == 1
    assert "compiler_errors" not in out


# ---------------------------------------------------------------------------
# route_after_diagnostics guard matrix
# ---------------------------------------------------------------------------

def _routing_state(**overrides):
    state = {
        "node_state": {"diagnostics": {"status": "ok", "new": 2}},
        "loop_counter": {"total_repairs": 0,
                         "diagnostics_rounds_since_compile": 1},
        "budget_remaining_usd": 1.0,
        "diagnostics_config": {"max_rounds": 2},
    }
    state.update(overrides)
    return state


def test_route_all_guards_pass_goes_to_repair():
    from harness.graph import route_after_diagnostics
    assert route_after_diagnostics(_routing_state()) == "repair_node"


def test_route_partial_status_with_new_still_repairs():
    from harness.graph import route_after_diagnostics
    s = _routing_state(node_state={"diagnostics": {"status": "partial", "new": 1}})
    assert route_after_diagnostics(s) == "repair_node"


@pytest.mark.parametrize("mutation", [
    {"node_state": {"diagnostics": {"status": "ok", "new": 0}}},
    {"node_state": {"diagnostics": {"status": "skipped", "new": 3}}},
    {"node_state": {"diagnostics": {"status": "disabled"}}},
    {"node_state": {}},
    {"budget_remaining_usd": 0.0},
    {"loop_counter": {"total_repairs": 99,
                      "diagnostics_rounds_since_compile": 1}},
    {"loop_counter": {"total_repairs": 0,
                      "diagnostics_rounds_since_compile": 3}},
])
def test_route_each_guard_independently_forces_compiler(mutation):
    from harness.graph import route_after_diagnostics
    assert route_after_diagnostics(_routing_state(**mutation)) == "compiler_node"
