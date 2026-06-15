# Contributing to myharness

Thanks for taking the time to contribute. This document covers the conventions
the project enforces so contributors don't have to rediscover them from a
rejected PR.

## Development setup

```bash
pip install -e ".[dev]"
make hooks-install
make test
```

The `make hooks-install` step wires the pre-commit hook into your local git
checkout. After that, every `git commit` runs the full pytest pack and blocks
the commit if anything fails.

## The pre-commit gate

The hook lives in `.pre-commit-config.yaml`. It runs
`python -m pytest tests/ -q --tb=short` on every commit, regardless of which
files changed.

**Do not bypass it casually.** The hook exists because:

- Test discipline is the project's main correctness guarantee — the
  pack covers every module. The current count is surfaced live by
  `make coverage` (terminal summary + HTML report under `htmlcov/`);
  see the Coverage section in `README.md` for how to run it.
- A green local hook is what makes the GitHub Actions matrix (3.11/3.12/3.13)
  green too.
- `git commit --no-verify` is reserved for genuine emergencies (e.g. a flaky
  external test you've already triaged in a separate issue). Use it
  deliberately, not as a default.

CI on `main` and PRs runs the same suite — bypassing the local hook just
delays the failure, it doesn't avoid it.

## Test layout

Tests live in `tests/` with one file per module under test:

| Test file | Module under test |
|-----------|-------------------|
| `tests/test_cli_basics.py` | `harness/cli.py` (config, helpers) |
| `tests/test_doctor.py` | `harness/cli.py` (`harness doctor`) |
| `tests/test_graph_basics.py` | `harness/graph.py` |
| `tests/test_harness.py` | Cross-module + storage regression |
| `tests/test_hitl.py` | `harness/hitl.py` |
| `tests/test_impact_basic.py` | `harness/impact.py` |
| `tests/test_lintgate_basic.py` | `harness/lintgate.py` |
| `tests/test_observability.py` | `harness/observability.py` |
| `tests/test_parser_basic.py` | `harness/parser_registry.py` |
| `tests/test_redactor.py` | `harness/redactor.py` |
| `tests/test_security_basic.py` | `harness/security.py` |
| `tests/test_skills_filter.py` | `harness/skills.py` |
| `tests/test_speculative_basic.py` | `harness/speculative.py` |
| `tests/test_storage_basic.py` | `harness/storage.py` |
| `tests/test_trust.py` | `harness/trust.py` |
| `tests/test_phase7_finals.py` | Integration / regression coverage |

When you add a module, add a matching `tests/test_<module>.py` file. Most
tests are synchronous and use `pytest`'s built-in fixtures (`tmp_path`,
`monkeypatch`, `caplog`). Async tests use `pytest-asyncio`'s `@pytest.mark.asyncio`.

## Commit-message convention

The project follows [Conventional Commits](https://www.conventionalcommits.org/)
with a short scope:

```
<type>(<scope>): <imperative summary>

<optional body — wrap at 72 cols, explain *why*, not *what*>

<optional Co-Authored-By trailers>
```

Types in active use, with examples from the log:

| Type | When to use | Example |
|------|-------------|---------|
| `feat` | New user-visible capability | `feat(skills): language-aware skill filtering via applies_to frontmatter` |
| `fix` | Bug fix referencing the broken behavior | `fix(cli): close TOCTOU race in _read_spec_file (Bug 7)` |
| `docs` | Docs-only change | `docs(skills): seed stack-aware skills for the full harness stack` |
| `test` | Tests-only change (no production code change) | `test: phase 7 — skills, patcher, speculative tests` |
| `chore` | Build / tooling / housekeeping | `chore: pre-commit regression gate for the framework` |
| `refactor` | Code reorganization with no behavior change | (use sparingly) |

Scopes match the module name where the change lives (`cli`, `graph`,
`sandbox`, `gateway`, `skills`, `parser`, etc.) or `p0` for cross-cutting
hot-fixes.

Keep the subject line under 72 characters. Use the body to explain the
*motivation* — what was broken, what user-visible behavior changes — rather
than restating the diff.

## Versioning and releases

myharness follows [SemVer](https://semver.org/):

- **MAJOR** — backwards-incompatible change to the CLI surface, config
  schema, or checkpoint format.
- **MINOR** — new capability, new subcommand, new config section. The
  Tier 1 closeout shipped as v1.1.0.
- **PATCH** — bug fix, doc update, CI fix.

`CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/).
Add an entry to the `[Unreleased]` section in the same commit as your
behavior change. The `make release` target handles bump + tag + push.

## What NOT to do

myharness is opinionated about scope. The following patterns get rejected,
so save yourself a round trip:

- **Don't add features beyond the task.** A bug fix doesn't need surrounding
  cleanup; a one-shot operation doesn't need a helper. Three similar lines
  is better than a premature abstraction.
- **Don't add backwards-compatibility shims for code you're rewriting.**
  If a flag becomes unused, delete it. Don't leave a `// removed` comment.
- **Don't add error handling, fallbacks, or validation for scenarios that
  can't happen.** Trust internal code and framework guarantees. Only
  validate at system boundaries (user input, external APIs).
- **Don't write WHAT-comments.** Well-named identifiers already describe
  *what*. Comments should explain *why* — a hidden constraint, a workaround
  for a specific bug, behavior that would surprise a reader.
- **Don't introduce new dependencies casually.** Every dep adds install
  surface, supply-chain risk, and Python-version friction. If you can
  do it in 20 lines of stdlib, do it in 20 lines of stdlib.

## Submitting a PR

1. Open against `main`. Feature branches go through review even for
   single-line fixes.
2. The CI matrix (Python 3.11/3.12/3.13) must be green.
3. Reference the relevant `BUG_REPORT.md` finding,
   `production-readiness-audit.md` tier item, or GitHub issue in the PR
   description.
4. Update `CHANGELOG.md` under `[Unreleased]`.
5. Don't squash unless the reviewer asks — clean per-step commits make
   `git bisect` useful.

## Where to read more

- [`README.md`](README.md) — quick-start, command reference, troubleshooting.
- [`docs/SPEC_REQUIREMENTS.md`](docs/SPEC_REQUIREMENTS.md) — functional and
  non-functional requirements the harness is built against.
- [`docs/SPEC_ARCHITECTURE.md`](docs/SPEC_ARCHITECTURE.md) — module map and
  graph topology.
- [`docs/BUG_REPORT.md`](docs/BUG_REPORT.md) — historical bug catalogue
  (all currently closed).
- [`docs/production-readiness-audit.md`](docs/production-readiness-audit.md)
  — the audit driving Tier 1 / Tier 2 / Tier 3 roadmap.
