# ADR-0006: In-Build Acceptance-Criterion Verification

**Status:** Accepted (design ratified 2026-08-21; **Phase 0 in progress** — dual-altitude generators; graph node + later phases pending, see Action Items). Supersedes the build↔test AC boundary set by [[ADR-0005]] and the `flow == "test"`-only AC gate.
**Date:** 2026-08-21
**Deciders:** Teane harness maintainers
**Related:** [[ADR-0001]] (test-author regeneration for unsatisfiable tests), [[ADR-0003]] (hybrid deterministic + LLM test generation), [[ADR-0004]] (constraint NFRs → acceptance criteria), [[ADR-0005]] (shift-left test quality to minimize repair)

## Context

Today teane verifies two different things in two different places, and there is a
hole between them.

- **The build/patch loop verifies *code*, via *unit* tests.** `test_generation_node`
  emits code-linked unit tests (`@tests:` markers), runs them in the sandbox, and
  routes failures to repair. The acceptance bar for a story is "its generated unit
  tests pass."
- **`teane test` verifies *acceptance criteria*, via *browser E2E*, *post-deploy*.**
  `test_node → test_runtime.run_test_pipeline` generates one Playwright scenario per
  AC (`@verifies:` markers), seeds data, and drives Chromium against a deployed app.

The AC-coverage gate (`traceability.has_ac_gap`, `harness/traceability.py:149-162`)
is therefore enforced **only when `flow == "test"`** — build and patch deliberately
do not block on it. That deferral was a scar, not a preference: blocking build on AC
coverage created an **unfixable auto-resume loop** (finsearch session 156032347 —
25/124 ACs untested at end-of-build, headless resume ping-ponging through
`traceability_node` with no marker it could add to make progress). The lesson stands.

Two problems compound from that boundary:

1. **Behavioral defects escape the build entirely.** Unit tests are self-generated
   and (per [[ADR-0005]]) verified *structurally* — contradiction/unsatisfiability
   proving, assertion *counts*, contract wiring. Nothing in the build loop checks
   that the story actually *does what the AC says* against real collaborators. A
   story can seal `done` with green unit tests and still violate its own acceptance
   criteria. That class of defect is not discovered until the separate `teane test`
   run — after deploy, after every other story is built, when the fix is most
   expensive and the repair context is cold.

2. **The one gate that would catch it is a rubber stamp.** The `teane test`
   acceptance layer never actually verifies behavior, because the LLM-backed
   generators are unwired. `playwright_gen.fallback_scenarios`
   (`harness/playwright_gen.py:148`) emits placeholder bodies —
   `await expect(page).toHaveTitle(/.+/)` (`_placeholder_body:221-226`) — and
   `test_data_gen.fallback_seed` (`harness/test_data_gen.py:237`) emits a single
   `_teane_test_meta` stub row. The `PipelineOverrides.scenario_generator` /
   `seed_generator` seams exist but `cli.py` never constructs them, and no
   LLM-backed generator exists anywhere in the tree. So a green "AC coverage: 100%"
   can mean "a `@pytest.mark.skip` stub or a title-check exists for each AC," not
   "each criterion is verified."

The net effect measured on lumina runs: the build converges on "compiles + passes
its own unit tests," ships, and acceptance is never behaviorally checked. This is
the single largest correctness gap in the pipeline.

**Forces at play:**

- **North-star: autonomous processing.** Acceptance signal is worthless if it
  re-creates the finsearch HITL loop. Any in-build AC check must have a defined
  can't-satisfy-yet escape, never an infinite block.
- **Repair is expensive and cold at the end.** [[ADR-0005]] established that repair
  is the dominant build cost. A defect surfaced at the point the story is built —
  small, in-context — is dramatically cheaper to fix than the same defect surfaced
  post-deploy across a finished app.
- **Browser E2E is the wrong tool for the inner loop.** It needs a fully deployed,
  running app; for the first N stories the app may not boot at all. Chromium per
  story is the most expensive and flakiest thing that could go in the repair loop.
- **Most ACs are verifiable without a browser.** For a web backend, the majority of
  ACs are expressible as in-process HTTP assertions (FastAPI `TestClient` /
  supertest) against real collaborators — exactly the level [[ADR-0003]]'s contract
  tests already run at, with sandbox plumbing that already exists.
- **The partial-app problem is real.** An AC for STORY-005 may depend on auth from
  STORY-020, unbuilt. Running it early will fail for reasons unrelated to STORY-005.
  Attribution is mandatory or repair thrashes.

## Decision

**Verify acceptance criteria *inside* the build loop, at integration altitude, per
batch, as a best-effort gate with a deferral escape — and keep browser E2E as the
post-deploy whole-app gate.** Three sub-decisions plus one prerequisite.

### 0. Prerequisite — real dual-altitude scenario + seed generators

Replace the placeholder `fallback_scenarios` / `fallback_seed` with LLM-backed
generators wired through the existing `PipelineOverrides` seams. Each generated
scenario carries, in addition to today's `@verifies: STORY-N.AC-M`:

- an **altitude**: `integration` (in-process app/`TestClient`, runnable in the build
  sandbox) or `e2e` (browser, needs a deployed app); and
- a **classification** the generator assigns per AC: `backend-verifiable` vs
  `ui-only`.

The generator emits an `integration` scenario for every `backend-verifiable` AC and
an `e2e` scenario for every AC (so the post-deploy pass remains complete). Seed data
becomes real, schema-typed, per-AC fixtures rather than one stub row. Nothing below
is worth building until this lands — without it, in-build acceptance would rubber-
stamp faster.

### 1. Altitude — integration in the build, browser post-deploy

The build loop runs only `integration`-altitude scenarios, in-process, in the
existing sandbox: import the app object, reset + seed the SQLite DB, assert against
real routes/collaborators. No browser, no port, no deploy. `ui-only` ACs are tagged
and deferred to the post-deploy `teane test` browser pass, which remains the
authority for the UI layer and whole-app regression.

Rationale: integration altitude is cheap (ms, no boot), deterministic (no Chromium
flake), and covers the majority of ACs, while the faithful-but-fragile browser layer
stays out of the inner loop and remains the final gate.

### 2. Granularity — per batch, on green compile + unit tests

A new `acceptance_node` runs once per batch, after the batch's code has compiled
green and its unit tests pass, before `batch_commit`. It executes every
`integration` scenario whose story is in this batch **and is newly runnable** (story
done, AC not already passed, dependencies satisfiable). Batch is the natural unit —
stories seal in batches (`seal_batch_atomically`) and the fixed setup (import app,
reset+seed DB) amortizes across the batch's ACs. A batch-of-one degenerates to
exactly per-story verification.

**Run each AC once.** A passed AC records a `test_verifies_ac` edge with build-time
provenance and is not re-run in later batches; only newly-runnable and
previously-deferred ACs execute each batch. This bounds cost — the whole acceptance
suite is not re-run every batch.

### 3. Failure handling — triage into three buckets; never hard-block

Every failing AC is classified (reusing the [[ADR-0005]] `test_triage` CODE_GAP vs
TEST_BUG machinery and the repair reflection judge) into:

- **Attributable failure** — the AC exercises code this batch touched and the
  behavior is genuinely wrong → feed the repair loop as a `TEST_FAILURE:`-style
  diagnostic, bounded by a config-driven per-batch **acceptance-repair cap**.
- **Dependency-blocked** — the AC needs a route/feature from a story not yet built
  (app won't boot, or a prerequisite endpoint 404s) → mark
  `deferred:blocked-by-dependency`, **do not fail the batch**, re-queue for the batch
  that satisfies the dependency.
- **Ui-only** — cannot run in-process → mark `deferred:needs-browser`, hand to the
  post-deploy `teane test` pass.

Only the attributable bucket blocks, and only up to the cap. After the cap, residual
attributable failures downgrade to a parked defect and the batch seals
`complete_with_blocks` (the escape `seal_batch_atomically` already has). The
end-of-run traceability gate stays the backstop. This is what preserves the finsearch
lesson: real early signal, but no state in which the build cannot make progress.

Like [[ADR-0005]]'s classifier, triage is conservative and defaults to
`dependency-blocked`/defer on doubt — it can add a repair round or defer, but can
never mask a real code defect by turning it into a hard failure that stalls the run.

### Node contract

```
... → test_generation_node → compiler_node (green)
        → acceptance_node        ← NEW
        → route_after_acceptance:
             attributable failure AND under cap → repair_node   (loops back)
             else                                → code_review_node → batch_commit
```

`acceptance_node`, per batch: (1) collect newly-runnable `integration` ACs; (2) reset
+ seed DB, import app; (3) run each scenario once; (4) record pass edges with
build-time provenance; (5) triage failures into the three buckets and populate
`node_state.acceptance` (passed / attributable / deferred sets) for
`route_after_acceptance`. Fail-open by construction — a harness error in
`acceptance_node` (boot failure, seed failure) degrades the whole batch to
`dependency-blocked`/defer, never a spurious code failure.

### State / traceability changes

- `test_verifies_ac` edges gain a **provenance** column (`build` vs `test`) so an AC
  verified in-build at integration altitude is distinguishable from one verified
  post-deploy in the browser, and so `teane test` can skip re-verifying build-passed
  backend ACs while still owning `ui-only` ones.
- A per-AC **deferral state** (`blocked-by-dependency` / `needs-browser`) persisted in
  `state.db`, so deferred ACs are re-queued deterministically and surface in
  `TRACEABILITY.md` as pending-not-failed.
- The build-flow traceability gate stays advisory on AC coverage (build never
  hard-blocks on it); `teane test` remains the enforcing gate, now fed real coverage.

### Config knobs (all config-driven per the no-hardcoded-caps rule)

`acceptance.enabled` (default off for first rollout), `acceptance.altitude`
(`integration` | `integration+e2e`), `acceptance.max_repair_rounds_per_batch`,
`acceptance.boot_timeout_seconds` (Phase 2), `acceptance.rerun_passed`
(default false). Each lands with a `config_help.md` entry, the cli validator
(`_KNOWN_NESTED_KEYS` + `_TYPE_SCHEMA`) update, and web-forms auto-render.

## Options Considered

### Option A: Status quo — all AC verification deferred to post-deploy `teane test`

| Dimension | Assessment |
|-----------|------------|
| Complexity | None |
| Cost | Poor — behavioral defects surface post-deploy, cold, across a finished app |
| Autonomy | Fine (no new loop) but assurance is illusory |
| Correctness | **Poor** — the gate is a placeholder rubber stamp; behavior is never checked in-build |

**Cons:** Leaves the largest correctness gap open. Even once the generators are
wired, discovery-at-the-end means the most expensive possible fix point.

### Option B: Integration-altitude, per-batch, in-build + browser post-deploy (this ADR)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium — one new node + generators + deferral state; reuses sandbox, triage, seal-with-blocks |
| Cost | Cheap per AC (in-process); defect caught small and in-context |
| Autonomy | High — deferral escape means no un-progressable state; conservative triage |
| Correctness | Strong — real behavioral check against real collaborators at build time |

**Pros:** Attacks the correctness gap where the fix is cheapest; reuses existing
machinery ([[ADR-0003]] `TestClient` plumbing, [[ADR-0005]] triage, `seal_batch_
atomically` blocks path); keeps the flaky browser layer out of the loop. **Cons:**
`ui-only` ACs still only verified post-deploy; the dual-altitude generator is real
new work; deferral/re-queue is a new state machine.

### Option C: Full browser E2E per story, in-build

| Dimension | Assessment |
|-----------|------------|
| Complexity | High — ephemeral full-stack boot per story, Chromium in the loop |
| Cost | Very high — boot + browser per story × repair rounds; multiplies wall-clock |
| Autonomy | Poor — early stories can't boot; flaky browser failures thrash repair |
| Correctness | Highest fidelity per AC, if it can run at all |

**Cons:** Puts the most expensive, flakiest verification in the inner loop, against
an app that for early stories does not boot. Re-creates the finsearch loop pressure
unless heavily gated — at which point it converges on Option B anyway. The faithful
browser check belongs at the post-deploy whole-app boundary, not per story.

### Option D: Static AC-contract check only (no scenario execution)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low — extend `route_check` to assert each AC's route/handler exists |
| Cost | Trivial |
| Autonomy | High |
| Correctness | **Shallow** — proves the endpoint exists, not that it behaves |

**Cons:** Same failure mode as today's contract tests — verifies wiring, not
behavior. Useful as a cheap *pre-filter* (skip acceptance for an AC whose route
doesn't exist yet → auto-defer), but not a substitute for executing the criterion.
Folded into Option B as the dependency-satisfiability pre-check, not adopted alone.

## Trade-off Analysis

The tension is **in-build verification cost vs. escaped-defect cost**. Option A
spends nothing and lets every behavioral defect escape to a post-deploy gate that
today verifies nothing — the worst correctness position. Option B spends a bounded,
cheap in-process run per batch to catch behavioral defects while the story is small
and in-context, with a deferral escape that structurally cannot stall the run.
Option C buys higher per-AC fidelity at a cost and flakiness that belong at the
whole-app boundary, not the inner loop. Option D is too shallow to be the check but
is the right cheap pre-filter.

Option B composes with the ADR chain the way [[ADR-0005]] composed with [[ADR-0003]]:
[[ADR-0003]] owns deterministic contract tests, [[ADR-0005]] owns validating
LLM-authored unit tests before repair, and ADR-0006 owns **executing acceptance
criteria against real collaborators at build time**, with the browser E2E pass
demoted to `ui-only` + whole-app regression. The conservative triage default bounds
the downside exactly as [[ADR-0005]]'s does: it can add a repair round or defer,
never manufacture a stall or mask a code defect.

The load-bearing risk is the dual-altitude generator quality — a weak generator makes
in-build acceptance either noisy (false attributable failures → repair thrash) or
hollow (weak assertions → false green). This is why Phase 0 ships and is measured
before the graph node is wired (mirroring [[ADR-0005]]'s "flip the gate, measure the
first live run" discipline).

## Consequences

**Easier:**
- Behavioral acceptance defects are caught at build time, small and in-context, at
  the cheapest possible fix point rather than post-deploy across a finished app.
- The `teane test` gate stops being a rubber stamp — backend ACs arrive pre-verified
  with `build` provenance; the browser pass focuses on `ui-only` + regression.
- Reuses the existing sandbox, `TestClient` plumbing, triage, reflection judge, and
  `complete_with_blocks` seal path — little net-new infrastructure below the
  generator.

**Harder:**
- The dual-altitude LLM scenario + seed generators are real new work and the quality
  linchpin; a weak generator degrades the whole gate.
- A new deferral/re-queue state machine (`blocked-by-dependency` / `needs-browser`)
  and a `test_verifies_ac` provenance column to maintain and render.
- Dependency-satisfiability detection (is an AC runnable yet?) is a new heuristic
  that must bias toward defer to avoid false attributable failures.
- Per-batch acceptance adds runtime to the build; bounded by run-once provenance and
  the repair cap, but non-zero.

**To revisit:**
- Whether `ui-only` ACs justify Phase 2 ephemeral in-sandbox boot (real server on a
  random port) for a subset, or stay entirely post-deploy — decide from the measured
  `needs-browser` deferral share.
- Whether batch planning should order/limit stories so an AC becomes runnable soon
  after its story (shrinking the `blocked-by-dependency` backlog).
- Whether the acceptance-repair cap should be spec-driven (story complexity) rather
  than fixed.
- Whether build-passed backend ACs can safely skip the post-deploy browser pass
  entirely, or should be spot-checked for regression.

## Action Items

1. [~] **Phase 0 — dual-altitude generators.** Real LLM-backed
   `scenario_generator` + `seed_generator` wired through `PipelineOverrides`,
   emitting `integration` + `e2e` altitudes and `backend-verifiable` | `ui-only`
   classification per AC, with real schema-typed seed fixtures. Ship and **measure
   generated-scenario quality on a real lumina story** before wiring the node
   (per [[ADR-0005]] discipline). Replaces `playwright_gen.fallback_scenarios` /
   `test_data_gen.fallback_seed` as the default when configured.
   **Landed (2026-08-21):** `harness/acceptance_gen.py` — the async LLM generator
   (`generate_acceptance_scenarios`, `planning` role), per-AC context gatherer
   (pulls AC *prose* + discovered routes), the anti-rubber-stamp validator
   (`validate_scenarios` drops tautologies / `toHaveTitle(/.+/)` / assert-less /
   unknown-`verifies`), an honest designed-to-fail offline fallback, and
   integration(pytest)/e2e(Playwright) renderers. Config: `acceptance.*` in all
   four registration points (`config.json`, `_KNOWN_TOP_LEVEL_KEYS`,
   `_KNOWN_NESTED_KEYS`, `_TYPE_SCHEMA`) + `config_help.md`; **default off**.
   The **LLM seed generator** (`generate_seed_data_llm`) grounds on the workspace's
   actual `CREATE TABLE` DDL (`discover_table_schemas` — lumina has no
   SPEC_DATA_MODEL.md), tags rows `_verifies`, and validates against the real
   schema (`validate_seed` drops unknown tables / assertionless rows / caps rows).
   `scripts/acceptance_preview.py` (with `--seed`) is the measurement vehicle. 35
   unit tests green; ruff clean; adjacent suites (data-gen/playwright/runtime) green.
   **Seed→conftest wiring COMPLETE:** the generated conftest gives each test an
   isolated DB (`discover_db_env_var` finds the pydantic-settings override, e.g.
   `LUMINA_DB_PATH`) and applies the seed via a self-contained stdlib-sqlite3
   `_apply_seed` inlined into the conftest (it CANNOT import teane's `harness` — the
   conftest runs in the workspace sandbox). Integration scenarios are also
   prompt-forced to be self-contained (arrange via API), so verification holds even
   without seed. Seeding only engages when the DB is isolatable (known path).
   **Remaining:** run the LLM path against a real lumina story and eyeball the
   integration/e2e/seed output (the ADR-0005 "measure before wiring" gate).
2. [~] **Phase 1 — `acceptance_node` (integration altitude).** LANDED (code,
   default-off): `harness/acceptance_node.py` (orchestration + `route_after_acceptance`)
   spliced between compiler-green and code review (`graph.py`: import + `add_node` +
   compiler-edge retarget to `acceptance_node` + conditional edges); the triage engine
   `harness/acceptance_run.py` (attributable / dependency-blocked / ui-only, conservative
   defer-on-doubt, pytest `-rA` output parser); `acceptance_config` threaded through
   `AgentState`/`run_graph`/cli (both sites); cap knob `acceptance.max_repair_rounds_per_batch`.
   Guards: `enabled=false` → pure pass-through (graph behaves exactly as before, one no-op
   hop); any error (no gateway / no app / sandbox crash / collection error) → defer, never
   fail the batch. Graph builds+compiles; full suite 5123 passed / 0 regressions.
   **Pending live validation** (same gate as Phase 0): the sandbox-run adapter
   (`_make_sandbox_runner`, cross-thread `_await`, pytest command) and the conftest's
   `client` fixture have not been exercised against a real sandbox run yet.
3. [x] **Deferral state + provenance.** `test_verifies_ac.provenance` column added
   ADDITIVELY (v5→v6 `_migrate_v5_to_v6` `ALTER TABLE`, preserves existing edges;
   `SCHEMA_VERSION=6`); `acceptance_deferrals` table + `record_/clear_/list_acceptance_deferral`;
   `acs_verified_with_provenance` (run-each-AC-once) + `story_keys_for_batch`. `link_test_to_ac`
   carries `provenance='build'`. Re-queue: the node re-attempts `deferred:blocked-by-dependency`
   ACs each batch. TRACEABILITY rendering of deferred-not-failed still TODO.
4. [~] **Dependency-satisfiability (Option D).** Folded into runtime triage rather than a
   pre-run route gate: a 404 / ImportError / missing-fixture failure classifies as
   `deferred:blocked-by-dependency` (never attributable), so unbuilt-prerequisite ACs defer
   without a separate pre-check. A cheap pre-run route-exists filter remains a possible optimisation.
5. [ ] **Phase 2 — ephemeral full-stack boot** for `integration`+ ACs that need the
   real server (uvicorn on a random port, health-probe, run, teardown), reusing the
   `test_runtime` reachability logic. Only if Phase-1 telemetry shows a material
   set of ACs that need it.
6. [ ] **Phase 3 — browser E2E stays post-deploy** as the `ui-only` + whole-app
   regression gate, now fed a real generator and skipping build-verified backend ACs.
7. [ ] **Config surface** — `acceptance.*` keys with `config_help.md`, cli validator
   (`_KNOWN_NESTED_KEYS` + `_TYPE_SCHEMA`), and web-forms parity.
8. [ ] **Measure and decide** — from Phase-1 telemetry (attributable-failure catch
   rate, false-divert rate, deferral shares), decide the default-on flip and whether
   Phase 2 is warranted.
