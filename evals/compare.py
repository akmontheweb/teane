#!/usr/bin/env python3
"""Print a delta table between a current eval ``results.json`` and a saved
``baseline.json`` (audit #29).

Both files have the shape emitted by ``run_eval.py``:
    {"tasks": [{"name": ..., "success": ..., "total_cost_usd": ..., ...}, ...],
     "summary": {"pass_count": ..., "fail_count": ..., ...}}

The script joins rows by ``name`` and prints, per task, the delta in:
  - success (▲ regression, ▼ improvement, · no change)
  - total_cost_usd
  - llm_call_count
  - tokens_in / tokens_out / cached_tokens
  - cache_hit_rate
  - wall_clock_s

Tasks present in one file but not the other are reported with a NEW or
GONE marker. Exit code is non-zero when any task regressed (was PASS in
baseline, is FAIL now) so the script can gate CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE = EVAL_DIR / "baseline.json"
DEFAULT_CURRENT = EVAL_DIR / "results.json"


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        print(f"error: {path} does not exist.", file=sys.stderr)
        sys.exit(2)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh) or {}


def _by_name(blob: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {t["name"]: t for t in blob.get("tasks", []) if "name" in t}


def _fmt_delta(curr: float, base: float, fmt: str = "{:+.4f}") -> str:
    if base == curr:
        return f"{curr:.4f}"
    return f"{curr:.4f} ({fmt.format(curr - base)})"


def _verdict_marker(curr_pass: bool, base_pass: bool) -> str:
    if curr_pass == base_pass:
        return "·"
    return "▼" if curr_pass else "▲"  # ▲ = regression


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two eval result snapshots.")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--current", default=str(DEFAULT_CURRENT))
    args = parser.parse_args()

    baseline = _load(Path(args.baseline))
    current = _load(Path(args.current))
    base_by_name = _by_name(baseline)
    curr_by_name = _by_name(current)

    all_names = sorted(set(base_by_name) | set(curr_by_name))
    if not all_names:
        print("(no tasks in either snapshot)")
        return 0

    regressions: list[str] = []
    print(f"{'task':<28} {'verdict':<8} {'cost_usd':<24} {'calls':<10} {'hit_rate':<14} {'wall_s':<14}")
    print("-" * 100)
    for name in all_names:
        b = base_by_name.get(name)
        c = curr_by_name.get(name)
        if b is None:
            print(f"{name:<28} {'NEW':<8} {c.get('total_cost_usd', 0.0):<24} "
                  f"{c.get('llm_call_count', 0):<10} {c.get('cache_hit_rate', 0.0):<14} "
                  f"{c.get('wall_clock_s', 0.0):<14}")
            continue
        if c is None:
            print(f"{name:<28} {'GONE':<8}")
            continue
        verdict = _verdict_marker(bool(c.get("success")), bool(b.get("success")))
        if verdict == "▲":
            regressions.append(name)
        cost = _fmt_delta(c.get("total_cost_usd", 0.0), b.get("total_cost_usd", 0.0))
        calls = (
            f"{c.get('llm_call_count', 0)} "
            f"({c.get('llm_call_count', 0) - b.get('llm_call_count', 0):+d})"
        )
        hit = _fmt_delta(c.get("cache_hit_rate", 0.0), b.get("cache_hit_rate", 0.0))
        wall = _fmt_delta(c.get("wall_clock_s", 0.0), b.get("wall_clock_s", 0.0), fmt="{:+.3f}")
        print(f"{name:<28} {verdict:<8} {cost:<24} {calls:<10} {hit:<14} {wall:<14}")

    print()
    b_sum = baseline.get("summary") or {}
    c_sum = current.get("summary") or {}
    print(f"pass:  baseline={b_sum.get('pass_count', 0)}  current={c_sum.get('pass_count', 0)}")
    print(f"fail:  baseline={b_sum.get('fail_count', 0)}  current={c_sum.get('fail_count', 0)}")
    print(f"cost:  baseline=${b_sum.get('total_cost_usd', 0.0):.4f}  "
          f"current=${c_sum.get('total_cost_usd', 0.0):.4f}")

    if regressions:
        print(f"\nREGRESSIONS: {', '.join(regressions)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
