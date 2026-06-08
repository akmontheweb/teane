# Production-Readiness Audit

**Date:** 2026-06-08
**Scope:** `myharness` AI agent harness — full repo at HEAD `866ecd9`.
**Assessment:** ~65% production-ready. Solid core (545 passing tests, pre-commit
gate, security model, multi-stack support shipping); the gap is in *operability*
— first-run discoverability, CI enforcement, diagnostic affordances, and docs.

The harness is safe to use today by someone who built it. It is not yet safe to
hand to a second engineer without a walkthrough.

---

## Methodology

Ten dimensions evaluated, each scored Pass / Partial / Gap. Findings rolled into
four tiers ordered by user-visible blast radius, not implementation cost.

| Dimension                         | Status   | Notes |
|-----------------------------------|----------|-------|
| Test coverage (correctness)       | Pass     | 545 passing tests, pre-commit hook gates every commit, regression pack covers all modules. |
| Continuous integration            | Gap      | No GitHub Actions workflow. Local hook is bypassable with `--no-verify`. |
| Documentation                     | Partial  | Specs exist (`SPEC_ARCHITECTURE.md`, `SPEC_REQUIREMENTS.md`, `BUG_REPORT.md`), but `README.md` is a stub. No `CONTRIBUTING.md`. No command reference. |
| First-run experience              | Gap      | No `harness doctor`. Users hit silent failures on missing API keys, broken sandbox backend, or misconfigured routing with no first-class diagnostic. |
| Config validation                 | Partial  | `model_routing` keys get fuzzy-match suggestions (`cli.py:1498`). Nested sections (`token_budget`, `sandbox`, `persistence`, `deployment`, `lintgate`) silently accept typos. |
| Observability                     | Pass     | Structured logging across all modules, optional LangSmith tracing via env vars, token-budget telemetry, SQLite checkpoint timeline. |
| Security                          | Pass     | HITL gate fixed (Bug 1), path-traversal guard via `harness.trust.safe_resolve` (Bug 2), redactor scrubs secrets in logs, sandbox runs as non-root in Docker. |
| Error handling                    | Pass     | Exponential backoff with jitter on LLM calls, hard token-budget cap, checkpoint TTL GC, graceful fallback when tree-sitter grammar missing. |
| Multi-stack readiness             | Pass     | Python / Java / Node / Dart parsers + grammars wired. Flutter routing short-circuits the deploy chain. Stack-aware skill filtering live. |
| Packaging & install               | Partial  | `pyproject.toml` declares deps but no published wheel, no version pinning policy, no install verification command. |

---

## Tier 1 — Ship-blockers (do before declaring v1.1 done)

Estimated total effort: **~4.5 hours.**

### T1.1 — Add GitHub Actions CI workflow (~30 min)

**Problem:** Tests run only via the local pre-commit hook, which is bypassable
with `git commit --no-verify`. A second engineer pushing on a fresh clone with a
broken hook would silently ship regressions.

**Fix:** `.github/workflows/ci.yml` running `pytest tests/ -q --tb=short` on
every push to `main` and every PR. Matrix on Python 3.11 / 3.12 / 3.13.

**Acceptance:** Status check appears on PRs; failing tests block merge.

### T1.2 — `harness doctor` healthcheck command (~1 hour)

**Problem:** Users debugging first-run failures hit a wall: *did my API key
load? is Docker reachable? is my config valid? is the checkpoint DB writable?*
None of these have a first-class diagnostic — only error messages buried in
logs after an attempted run.

**Fix:** New subcommand `harness doctor` that runs five checks and reports
status with green/yellow/red markers:

1. Workspace is a git repo (`git rev-parse --git-dir`).
2. API keys present for the configured routing (env-var presence check, per
   provider in `model_routing`).
3. Sandbox backend reachable (`docker info` if `backend=docker`; `unshare
   --user echo` if `unshare`).
4. Checkpoint DB writable (attempt to open `~/.harness/checkpoints.db`).
5. Config parses cleanly (re-run `discover_config` + `_validate_config_keys`).

**Acceptance:** `harness doctor` exits 0 on a healthy install, non-zero with a
human-readable summary on failure.

### T1.3 — Deeper config typo detection (~1 hour)

**Problem:** `_validate_config_keys` only fuzzy-matches keys inside
`model_routing` (`cli.py:1498`). A typo like `token_budget.hrad_cap_usd` is
silently ignored — config loads, agent runs without a budget cap, user gets
billed for it.

**Fix:** Extend `_validate_config_keys` to recurse into known nested sections
with per-section `_KNOWN_KEYS` sets. Reuse the existing fuzzy-match machinery
from `gateway.py:_validate_routing_keys` (lines ~1498–1551).

Sections to cover: `sandbox`, `token_budget`, `persistence`, `model_routing`,
`deployment`, `lintgate`.

**Acceptance:** Typed key `token_budget.hrad_cap_usd` produces
`Did you mean 'hard_cap_usd'?` warning at config load.

### T1.4 — Expand README.md (~2 hours)

**Problem:** README is one paragraph. Anyone evaluating or onboarding has no
quick-start, no command reference, no troubleshooting, no config overview.

**Fix:** Add sections:

- *What myharness is* — one paragraph.
- *Quick start* — install + minimal `harness run` example.
- *Command reference* — `run`, `resume`, `status`, `purge`, `doctor` with flag
  tables.
- *Configuration* — overview + link to `SPEC_REQUIREMENTS.md`.
- *Troubleshooting* — common failure modes (missing API key, sandbox unreachable,
  config typo) and how `harness doctor` surfaces each.
- *Architecture* — pointer to `SPEC_ARCHITECTURE.md`.
- *Contributing* — pointer to (still-to-write) `CONTRIBUTING.md`.

**Acceptance:** A new engineer can clone, install, configure, and run their
first task using only the README.

---

## Tier 2 — Should ship before declaring v1.2 (~1 day)

### T2.1 — `CONTRIBUTING.md`

Document the pre-commit gate, the test layout, the commit-message convention,
and the "don't add features beyond the task" rule that's already in
`CLAUDE.md`. Without this, contributors will recreate patterns the project has
already rejected.

### T2.2 — Versioning and release process

`pyproject.toml` has no version-bump policy. No CHANGELOG. No `git tag`
discipline. Adopt SemVer + Keep-a-Changelog + a `make release` target that
verifies clean tree, runs tests, bumps version, tags, and pushes.

### T2.3 — Structured failure-mode catalog in logs

Logging is comprehensive but every module invents its own message format. Adopt
a shared `log_event(event_name, **fields)` helper so failures can be grepped by
event name (e.g., `sandbox_start_failed`, `token_budget_exhausted`,
`hitl_gate_blocked`) instead of by string fragment.

---

## Tier 3 — Quality-of-life (~2–3 days)

### T3.1 — End-to-end example workspaces

Ship `examples/fastapi-counter/`, `examples/flutter-counter/`,
`examples/spring-boot-todo/` so first-run users have a working target without
constructing their own scaffold.

### T3.2 — Token-budget dashboard

The data exists in the checkpoint DB and structured logs but there's no
ergonomic way to look at it. A `harness status --tokens` view that summarizes
spend per task and per provider would close the loop.

### T3.3 — Sandbox image preflight

`harness doctor` covers reachability, but a separate `harness sandbox prebuild`
command that pulls/builds the required images proactively would eliminate
first-run latency surprises.

---

## Tier 4 — Nice to have

### T4.1 — Web dashboard

A read-only web UI over the checkpoint DB for browsing past runs. Out of scope
for v1.x; relevant when there are enough users to demand it.

### T4.2 — Cross-platform support audit

Currently Linux-tested only. macOS likely works for most paths (Docker backend
is portable; `unshare` is Linux-only). Windows is unlikely without WSL2.
Document the matrix; don't promise what isn't tested.

---

## What's already done well (do not re-litigate)

- **Test discipline.** 545 tests, pre-commit gate, fast pytest run (~30s on a
  developer laptop). Test-first culture is visible in every recent commit.
- **Security posture.** Three Bug-Report findings (HITL inversion, lintgate
  path escape, async refine crash) closed in `1f91e25`. Path-traversal guard is
  consistent. Secrets are redacted before logging.
- **Multi-stack support.** Tree-sitter grammars, parsers, lintgate formatters,
  and skill files now cover Python / Java / Node / Dart / Flutter consistently.
  Stack detection drives routing and prompt assembly.
- **Recovery.** SQLite checkpoint store with WAL, TTL GC, and resume-from-step
  semantics. The user can `Ctrl-C` a run and pick it back up.
- **Observability hooks.** Structured logging, optional LangSmith, token
  telemetry. The bones for production debugging are in place.

---

## Bug-report status

All eight findings in `docs/BUG_REPORT.md` are closed as of `866ecd9`. See
commits `1f91e25` (Bugs 1–3), `22f7990` (Bugs 5–6), `b4c0694` (Bug 4 proper
fix), `866ecd9` (Bug 7). Bug 8 was closed earlier as a duplicate. No
outstanding bug-report items.

---

## Recommendation

**Close out Tier 1, then declare v1.1.** Tier 1 is the difference between "the
author can use it" and "a second engineer can use it without supervision." The
~4.5-hour estimate is realistic; the work is well-scoped and uncontroversial.

Tier 2 and beyond should wait for actual user feedback. Building T3.1 example
workspaces before anyone has tried the harness on a real task is speculative.
