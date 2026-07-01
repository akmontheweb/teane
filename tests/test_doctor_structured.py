"""Phase 1 regression: doctor emits structured JSON via `--json` and
the human-readable printer still returns the same exit-code contract.

Two levels:
1. Data-layer contract on ``DoctorResult`` + the two renderers — the
   web wizard (Phase 3) will consume these directly, so their shape
   is a public contract now.
2. CLI contract: ``teane doctor --json`` returns a valid JSON payload
   with the required top-level keys and matches the historical
   exit-code semantics (0 = all pass, 1 = any fail/skip).

We deliberately avoid running the live check functions (``git``, MCP,
sandbox probes) — they need a real workspace with a valid config and
would make the test flaky. Instead, we assemble a synthetic
``list[DoctorResult]`` and exercise the two renderer functions.
"""

from __future__ import annotations

import json

from harness.cli import (
    DoctorResult,
    render_doctor_human,
    render_doctor_json,
)


def _results(*rows):
    return [DoctorResult(name=n, status=s, detail=d) for n, s, d in rows]


def test_render_json_shape_is_stable():
    results = _results(
        ("config", "pass", "loaded from /etc/teane/config.json"),
        ("git repo", "pass", "clean working tree"),
        ("sandbox backend", "warn", "docker not installed; falling back to unshare"),
    )
    text = render_doctor_json(results, "/workspace", "/etc/teane/config.json")
    payload = json.loads(text)

    assert payload["workspace"] == "/workspace"
    assert payload["config_path"] == "/etc/teane/config.json"
    assert isinstance(payload["results"], list)
    assert len(payload["results"]) == 3
    assert payload["results"][0] == {
        "name": "config",
        "status": "pass",
        "detail": "loaded from /etc/teane/config.json",
    }
    assert payload["summary"] == {
        "pass": 2,
        "warn": 1,
        "fail": 0,
        "skip": 0,
        "exit_code": 0,
    }


def test_render_json_summary_exit_code_reflects_failures():
    results = _results(
        ("config", "fail", "invalid JSON at line 12"),
        ("git repo", "skip", "skipped — fix the config check above first"),
    )
    payload = json.loads(render_doctor_json(results, "/w", "/c"))
    assert payload["summary"]["fail"] == 1
    assert payload["summary"]["skip"] == 1
    assert payload["summary"]["exit_code"] == 1


def test_render_human_returns_zero_when_all_pass():
    results = _results(
        ("config", "pass", "loaded ok"),
        ("git repo", "pass", "clean"),
    )
    text, code = render_doctor_human(results, "/w", "/c")
    assert code == 0
    assert "OK: all checks passed." in text
    assert "config" in text
    assert "git repo" in text


def test_render_human_returns_nonzero_when_any_fail():
    results = _results(
        ("config", "fail", "invalid config"),
        ("git repo", "pass", "clean"),
    )
    text, code = render_doctor_human(results, "/w", "/c")
    assert code == 1
    assert "FAIL" in text
    # When config fails specifically, the human renderer nudges the
    # operator toward the fix — a subtle but load-bearing message.
    assert "config" in text.lower()


def test_render_human_returns_nonzero_on_pure_skip():
    results = _results(
        ("config", "pass", "ok"),
        ("git repo", "skip", "skipped"),
    )
    text, code = render_doctor_human(results, "/w", "/c")
    assert code == 1
    assert "PARTIAL" in text


def test_render_human_warns_but_returns_zero_on_warn_only():
    results = _results(
        ("config", "pass", "ok"),
        ("sandbox backend", "warn", "docker not installed"),
    )
    text, code = render_doctor_human(results, "/w", "/c")
    assert code == 0
    assert "OK with warnings" in text


def test_doctor_result_is_frozen_dataclass():
    r = DoctorResult(name="x", status="pass", detail="ok")
    # Frozen — mutating a field raises.
    try:
        r.status = "fail"  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError("DoctorResult must be frozen so callers can safely share instances")


def test_render_json_pretty_prints_with_newline():
    """`teane doctor --json | jq` is the intended consumer — the output
    must be indented JSON, not a single line."""
    text = render_doctor_json(
        _results(("config", "pass", "ok")),
        "/w",
        "/c",
    )
    assert "\n" in text
    assert '  "workspace"' in text  # 2-space indent
