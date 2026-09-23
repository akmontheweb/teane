#!/usr/bin/env python3
"""Summarise ``spec_region_usage`` events from a run log (ADR-0008 item 2).

Answers the question the ADR rests on: which roles actually use the spec
region they are handed on every call, and which requirements are ever used at
all?

    python3 scripts/spec_usage_report.py ~/.harness/logs/<session>.jsonl

Requires the run to have been made with ``debug.measure_spec_usage: true``.
Reading the output: a role with calls but zero used is carrying the region as
freight — that is the ADR's premise holding for that role. A role with a high
used rate needs its context kept, or replaced by a slice that provably
contains what it was using.
"""

from __future__ import annotations

import collections
import json
import sys


def main(path: str) -> int:
    per_role: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"calls": 0, "used": 0, "cited": 0, "echoed": 0, "empty": 0}
    )
    req_hits: collections.Counter = collections.Counter()
    spec_chars = 0
    req_blocks = 0
    total = 0

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("event") != "spec_region_usage":
                continue
            total += 1
            spec_chars = max(spec_chars, int(rec.get("spec_chars") or 0))
            req_blocks = max(req_blocks, int(rec.get("req_blocks") or 0))
            row = per_role[str(rec.get("role") or "?")]
            row["calls"] += 1
            if rec.get("empty_response"):
                row["empty"] += 1
            cited = list(rec.get("cited") or [])
            echoed = list(rec.get("echoed") or [])
            if cited:
                row["cited"] += 1
            if echoed:
                row["echoed"] += 1
            if rec.get("used"):
                row["used"] += 1
            for key in set(cited) | set(echoed):
                req_hits[key] += 1

    if not total:
        print(f"No spec_region_usage events in {path}.")
        print("Was the run made with debug.measure_spec_usage: true?")
        return 1

    print(f"Spec region: {spec_chars:,} chars, {req_blocks} requirement blocks")
    print(f"Measured dispatches: {total}\n")
    print(f"{'role':<28}{'calls':>7}{'used':>7}{'cited':>7}{'echoed':>8}{'use rate':>10}")
    print("-" * 67)
    for role, row in sorted(per_role.items(), key=lambda kv: -kv[1]["calls"]):
        rate = (row["used"] / row["calls"]) if row["calls"] else 0.0
        print(f"{role:<28}{row['calls']:>7}{row['used']:>7}{row['cited']:>7}"
              f"{row['echoed']:>8}{rate:>9.0%}")
    grand_used = sum(r["used"] for r in per_role.values())
    print("-" * 67)
    print(f"{'ALL':<28}{total:>7}{grand_used:>7}"
          f"{'':>7}{'':>8}{grand_used / total:>9.0%}\n")

    print("Requirements ever used (by call count):")
    if not req_hits:
        print("  (none — no call reproduced anything unique to the spec region)")
    else:
        for key, n in req_hits.most_common():
            print(f"  {key:<24}{n:>5}")
    unused = req_blocks - len(req_hits)
    if unused > 0:
        print(f"\n  {unused} of {req_blocks} requirement blocks were never used "
              f"by any call.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
