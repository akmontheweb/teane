"""Every LLM dispatch must scope its prefix-cache family.

``cache_family`` defaults to ``role.value``, so two callsites that share a
role but NOT a system prompt land in the same bucket and evict each other.
On deepseek-v4-pro a cached input token costs $0.003625/1M against
$0.435/1M uncached — a 120x discount — so a thrashed family is not a
cosmetic reporting problem, it is the dominant input cost.

Measured across 21 retained sessions (9.86M input tokens, 84% cached):
caching saved $3.58 of a $4.29 all-uncached input bill. Planning was the
visible casualty before ``17b4b5d`` scoped the graph callsites — 10% hit
in session 01a03317 against 91% in 01a164b2 afterwards.

This test exists because that fix covered graph.py and left five
dispatches elsewhere on the bare default, including a subagent skill
dispatching as PATCHING with its own system prompt — which shares a
bucket with the patching node, the largest family in a run (4.6M input
tokens at 89% hit).
"""

import glob
import os
import re

import harness.graph as _graph

_DISPATCH_RE = re.compile(
    r"(?:gateway|self\.gateway|gw|session\.gateway)\.dispatch\("
    r"(.{0,900}?)\n\s*\)",
    re.S,
)
_ROLE_RE = re.compile(r"role\s*=\s*NodeRole\.(\w+)")


def _harness_dir() -> str:
    return os.path.dirname(os.path.abspath(_graph.__file__))


def _dispatches():
    """(file, line, role, has_cache_family) for every dispatch callsite."""
    out = []
    for path in sorted(glob.glob(os.path.join(_harness_dir(), "*.py"))):
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        for m in _DISPATCH_RE.finditer(src):
            body = m.group(1)
            role = _ROLE_RE.search(body)
            if not role:
                continue
            out.append((
                os.path.basename(path),
                src[: m.start()].count("\n") + 1,
                role.group(1),
                "cache_family" in body,
            ))
    return out


def test_the_scanner_finds_the_known_callsites():
    """Guard against the regex silently matching nothing — a scan that
    finds zero dispatches would make every assertion below vacuous."""
    found = _dispatches()
    assert len(found) > 15, f"only found {len(found)} dispatch callsites"
    assert any(f == "graph.py" for f, _, _, _ in found)


def test_every_dispatch_scopes_its_cache_family():
    bare = [
        f"{f}:{ln} role={role}"
        for f, ln, role, has in _dispatches() if not has
    ]
    assert not bare, (
        "These dispatches fall back to cache_family=role.value and will "
        "share a prefix-cache bucket with every other callsite of that "
        "role, evicting each other's cached prefix at a 120x cost "
        "multiple:\n  " + "\n  ".join(bare)
    )


def test_families_are_namespaced_by_role():
    """A family should read ``<role>:<callsite>`` so a log line says which
    callsite drifted, not just which role."""
    for path in sorted(glob.glob(os.path.join(_harness_dir(), "*.py"))):
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        for m in re.finditer(r'cache_family\s*=\s*f?"([^"]+)"', src):
            fam = m.group(1)
            assert ":" in fam, (
                f"{os.path.basename(path)}: cache_family {fam!r} is not "
                f"namespaced — use '<role>:<callsite>'"
            )


def test_speculative_variants_do_not_share_one_family():
    """Each variant is seeded with a different style prefix, so a shared
    family would report drift on every variant and let them evict one
    another as well as the real patching node."""
    with open(os.path.join(_harness_dir(), "speculative.py"), encoding="utf-8") as fh:
        src = fh.read()
    fams = re.findall(r'cache_family\s*=\s*f"([^"]+)"', src)
    assert fams, "speculative.py scopes no cache family"
    for fam in fams:
        assert "{spec.index}" in fam, (
            f"variant family {fam!r} is not keyed per variant"
        )
