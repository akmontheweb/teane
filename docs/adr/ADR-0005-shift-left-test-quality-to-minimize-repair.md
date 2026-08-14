# ADR-0005: Shift-Left Test Quality to Minimize the Repair Loop

**Status:** Accepted (items 1–2 shipped, gate default **on** as of 2026-08-13 — pending first live validation; suite right-sizing 4–6 phased — see Action Items)
**Date:** 2026-08-13
**Deciders:** Teane harness maintainers
**Related:** [[ADR-0001]] (repair-side test regeneration), [[ADR-0002]] (generation-side contradiction prevention), [[ADR-0003]] (hybrid deterministic + LLM test generation)

## Context

The repair loop is the single dominant cost of an agile build. Instrumenting
the lumina greenfield run **019ff418** (a full FastAPI + React birthday-tracker,
3 functional stories) over its ~4h44m span:

| Phase | LLM calls | Share |
|-------|-----------|-------|
| **Repair loop** (repair + reflection judgment) | **133** | **77%** |
| Initial generation (planning + patching) | 36 | 21% |
| Reviews | 4 | 2% |

Repair is ~98s/call on the reasoning model, and it is **sequential by
construction** — each round depends on the previous test result — so those 133
calls are the tent-pole of wall-clock. The intuitive fix is "generate better
*code* so there's less to repair." The telemetry says otherwise: **repair is
overwhelmingly fighting bad *tests*, not bad code.**

Categorising what the 85 repair rounds actually churned on (run 019ff418 log +
the backed-up final workspace):

- **The top fixation targets are all tests:** `tests/unit/test_database.py`
  (26 touches), `tests/unit/test_contact_service.py` (18), far above any
  production file. The concrete blockers that stalled the run were a
  `row_factory` bug (a test reading `row["name"]` off a raw `aiosqlite`
  connection that never set `row_factory`) and a `pytest.raises(HTTPException)`
  that **did not raise** — even though the service raises 404 correctly. Both
  are **test** bugs; no production change can satisfy them.
- **16 repair edits were refused by the test-tamper guard** — the repair LLM
  repeatedly, and correctly, concluded the *test* was wrong, but is forbidden
  to edit tests. So it thrashes: bending correct code to a wrong oracle, the
  reflection judge flags DISTRACTION, the round escalates to the reasoning
  model, and eventually a `reflection_distraction_loop` trips HITL (which, in a
  headless run, terminates the build — 019ff418 never reached the NFR or
  security stages).
- **The suite is sprawling and fragmented.** For a 6-source-file app the
  harness produced **16 test files / 125 test functions across four parallel
  tiers** — `tests/unit/`, `tests/integration/`, `tests/contract/`, *and* a
  package-nested `server/tests/`. `contact_repository` is tested in **three**
  separate files; near-duplicates exist (`test_contact_model.py` **and**
  `test_contact_models.py`). Every one of those 125 functions is a fresh
  chance for a test bug, and every test bug is disproportionately expensive
  because repair cannot fix it.

**Why the existing ADRs don't already cover this.** ADR-0001 recovers *after* a
bad test reddens the build (repair-side regeneration of an
already-declared-unsatisfiable test). ADR-0002 prevents *contradictory* test
batches. ADR-0003 moves the *contract-derivable* test class to deterministic,
correct-by-construction generation. All three are live and working — the
deterministic Tier 1/2/3 tests were **not** among the blockers. The residual
77% concentrates in exactly the surface those ADRs leave to the LLM:

1. **ADR-0003 Tier-4 (semantic, LLM-authored) tests** are still buggy at birth
   (`row_factory`, mis-wired mocks, `did not raise`) and still flow **straight
   into the expensive repair loop** to be discovered.
2. **Volume and fragmentation** are unbounded — the deterministic tiers *add*
   files, and nothing collapses the LLM tiers, the multi-tier split, or the
   `server/tests` vs `tests/unit` divergence into one canonical suite.
3. **Discovery is via the loop, not a gate.** A bad Tier-4 test costs ~dozens
   of repair rounds to surface, when a single targeted regeneration at
   generation time would cost one.

**Forces at play:**

- **North-star: autonomous processing.** A build that stalls at HITL on a
  self-inflicted test bug is the worst outcome. Repair should only ever be
  fixing genuine *code-vs-spec* gaps — which it is good at — never thrashing on
  a wrong protected oracle.
- **The tamper guard is correct and must stay.** Letting repair edit tests to
  go green is the reward-hack ADR-0001 exists to prevent. The answer is not to
  relax the guard but to stop bad tests from reaching repair.
- **Comprehensive ≠ sprawling.** The completeness goal (exhaustive per-module
  coverage) is right, but four overlapping tiers + duplicates + a split tree
  multiply bug surface without multiplying assurance.
- **Cheap-to-verify at generation time.** A freshly generated test can be run
  once, in the sandbox that already exists, before repair is ever entered.

## Decision

**Shift test-quality enforcement left — out of the repair loop and into
generation time** — via a pre-repair triage gate plus suite right-sizing, so
that the first time the repair loop runs, every failure it sees is a real
code-vs-spec gap rather than a test bug.

### 1. Pre-repair test-triage gate (in `test_generation_node`)

After the LLM Tier-4 dispatch and the existing marker/contradiction gates, and
**before** the graph hands off to `compiler → repair`:

1. **Run the newly generated tests once** in the sandbox (the deterministic
   sandbox run already exists at the end of `test_generation_node`; this
   extends it to classify, not just report).
2. **Cluster failures and classify each as `test-bug` vs `code-gap`** using
   cheap, deterministic signals — not another expensive judge:
   - `test-bug` heuristics (regenerate the test): the failing line is in a
     test file and matches a known test-defect fingerprint — a `patch(...)`
     target that does not resolve to a real symbol; an asserted-raised
     exception the code provably never raises (AST: no `raise` of that type on
     the exercised path); raw-row subscript access with no `row_factory`;
     import/collection errors in the test module; a mock of an *internal*
     module (see gate 3).
   - `code-gap` (let it flow to repair): a plain value/behaviour assertion
     mismatch against real collaborators — the birthday math, sort order, the
     7-day window. This is the class repair is *good* at.
3. **Regenerate `test-bug` clusters in place**, reusing ADR-0001's regeneration
   machinery **proactively** (and the ADR-0005-adjacent `salvage_canonical_
   rewrite` recovery, commit `5cb41d7`), bounded to K attempts, before the
   build ever enters repair. Only genuine `code-gap` failures are passed
   downstream.

The classifier is deliberately conservative and mechanical — when it cannot
confidently label a failure `test-bug`, it defaults to `code-gap` (repair), so
the gate can never *hide* a real code defect; the worst case is the status quo.

### 2. Right-size to one canonical suite per module

- **1:1 canonical test file per source module** (the standing completeness
  model), replacing the current unit/integration/contract/`server/tests`
  fan-out for small greenfield builds. The deterministic ADR-0003 tiers emit
  *into* the canonical file rather than spawning parallel trees.
- **Collapse redundant tiers by app size.** For a small greenfield app, an
  in-process unit test and a `TestClient` integration test of the same route
  are near-duplicative; keep the higher-signal one. Gate the extra tiers behind
  size/complexity thresholds.
- **Eliminate the `server/tests` vs `tests/unit` split at authoring time** —
  canonicalise the test root once, so the same module is never tested by two
  divergent files (the amplifier behind the doubled blocker count).

### 3. Enforce the no-mock policy as a gate, not a suggestion

`test_generation_node` already *tells* the LLM to prefer real collaborators
(the `test_generation.py` module header states the prompt forbids mocks), but
nothing enforces it, and mock mis-wiring was a top churn signal. Add a static gate: **reject tests that mock internal
modules** (repositories, db, services) — as opposed to true external I/O — and
steer regeneration toward real collaborators (in-memory SQLite + FastAPI
`TestClient`). Over-mocked tests are both lower-signal and the most
bug-prone-at-birth.

## Options Considered

### Option A: Status quo — repair discovers and works around bad tests

| Dimension | Assessment |
|-----------|------------|
| Complexity | None |
| Cost | Poor — repair is 77% of calls, and the tail case stalls the build at HITL |
| Autonomy | Poor — self-inflicted test bugs are the top HITL trigger |
| Correctness | Fine (guard holds) but at extreme wall-clock cost |

**Cons:** Leaves the measured cost centre exactly where it is; a single
protected test bug can burn dozens of rounds and terminate a headless run.

### Option B: Shift-left triage gate + right-sizing + no-mock gate (this ADR)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium — one new gate in an existing node + suite-shaping rules |
| Cost | Regeneration is ~1 cheap round vs ~dozens of repair rounds |
| Autonomy | High — repair only sees real code gaps; fewer distraction-loop HITLs |
| Correctness | Preserved — conservative classifier defaults to repair; tamper guard untouched |

**Pros:** Attacks the 77% at its source; reuses existing machinery (sandbox
run, ADR-0001 regeneration, ADR-0003 emitters); the tamper guard stays intact;
right-sizing shrinks bug surface directly. **Cons:** Classifier heuristics need
tuning and telemetry; suite-shaping thresholds are a new knob.

### Option C: Contract-first co-generation of code + tests

| Dimension | Assessment |
|-----------|------------|
| Complexity | High — new interface-contract artifact + two-sided generation |
| Cost | Highest upfront; strongest at eliminating divergence |
| Autonomy | High |
| Correctness | Sound — code and tests share one pinned contract |

**Pros:** Kills the code↔test divergence (`did not raise`, mock-target
mismatch) at the root by generating both sides against one explicit
signature/exception/return contract. **Cons:** A large change to the generation
pipeline; overlaps ADR-0003's deterministic derivation; better sequenced
*after* Option B proves the cost model. Recorded as the durable follow-on, not
the first move.

### Option D: Parallelize the repair loop instead

| Dimension | Assessment |
|-----------|------------|
| Complexity | High (worktree fan-out, merge) |
| Cost | Compresses wall-clock but not total work |
| Autonomy | No change — still stalls on protected test bugs |

**Cons:** Treats the symptom. Parallelism is complementary (a separate design)
but does not reduce the number of repair rounds; a loop thrashing on an
unfixable test still thrashes, just in parallel. Reducing repair first is
higher-leverage.

## Trade-off Analysis

The tension is **generation-time cost vs. repair-time cost**. Option A spends
nothing upfront and pays ~77% of the build in repair, with a fat tail that
terminates headless runs. Option B spends a bounded amount at generation time —
one sandbox run plus targeted regenerations — to keep bad tests out of a loop
where each is ~dozens of times more expensive to resolve. The conservative
classifier bounds the downside: it can only *add* a regeneration or fall
through to today's behaviour; it can never mask a real code defect, because
ambiguous failures route to repair.

Option B also **subsumes part of the repair loop's job the way ADR-0003
subsumed ADR-0002's**: bad tests are prevented from entering repair rather than
worked around inside it. It composes cleanly with the existing ADR chain —
ADR-0003 owns *what* deterministic tests to emit; ADR-0005 owns *validating the
LLM-authored remainder before repair* and *bounding total suite size*. Option C
is the deeper structural cure but overlaps ADR-0003 and is best sequenced after
B's telemetry confirms the cost model. Option D is orthogonal and can layer on
later once the loop is doing genuine work.

## Consequences

**Easier:**
- Repair only ever sees real code-vs-spec gaps — its actual competency —
  cutting round count and the distraction-loop HITL class.
- Bad LLM Tier-4 tests are caught in ~1 cheap regeneration round instead of
  surfacing through dozens of expensive repair rounds.
- A single canonical suite removes the split-tree blocker amplification and the
  duplicate-file surface.

**Harder:**
- A new classifier to maintain, with per-defect-fingerprint heuristics that
  must stay conservative (default to repair on doubt).
- Suite-size thresholds are a new tuning surface; too aggressive and coverage
  drops, too loose and sprawl returns.
- The no-mock static gate needs a reliable "internal module vs external I/O"
  distinction per stack.

**To revisit:**
- Whether the triage gate ever justifies Option C (contract-first) — decide
  from B's measured residual repair share.
- Per-stack extension of the no-mock and canonical-suite rules (TS/React).
- Whether right-sizing thresholds should be spec-driven (app size/complexity)
  rather than fixed.

## Action Items

1. [x] **Repair-cost telemetry, split by oracle class.** `repair_node` now
       classifies each round's diagnostics (`harness/test_triage.py`) and
       accumulates `triage_test_bug_diags` / `triage_code_gap_diags` /
       `triage_rounds_with_test_bug` into `loop_counter` → `last_build.json`,
       plus a per-round `[repair_node:triage]` log line. Observational only;
       lands the measurement baseline for the 77%→target drop.
2. [x] **Pre-repair test-triage gate** in `test_generation_node`
       (`_run_pre_repair_triage_gate`): on a failed test run, classify
       failures, regenerate `TEST_BUG` clusters via the ADR-0001 machinery
       (`test_regeneration_node`), re-run once, and pass only the residual
       (code-gap) failures to repair. Config-gated
       `test_generation.pre_repair_triage` (default off) +
       `triage_gate_max_regens` (default 3). Conservative: only unambiguous
       test-authoring bugs are diverted, so it can never mask a code defect.
3. [~] **`test-bug` fingerprint library** — shipped with the two proven
       high-confidence fingerprints (`test-undefined-name`,
       `test-raw-row-subscript`, grounded in the real 019ff418 diagnostics).
       Still to add and grow from telemetry: unresolved `patch()` target (AST),
       asserted-but-never-raised exception (AST), test-module import/collection
       error.
4. [ ] **Canonical single-suite authoring** — one test file per source module;
       deterministic tiers emit into it; eliminate the `server/tests` vs
       `tests/unit` split at authoring time (fold the recurring split-tree fix).
5. [ ] **Tier collapse by app size** — thresholds that drop near-duplicative
       integration/contract tiers for small greenfield builds.
6. [ ] **No-mock static gate** — reject internal-module mocks; steer
       regeneration to in-memory SQLite + `TestClient`.
7. [ ] **Measure and decide on Option C** — from the post-gate residual repair
       share, decide whether contract-first co-generation is worth building.

**Status note (2026-08-13):** items 1–2 shipped; 3 partial. The gate was
enabled by default (`test_generation.pre_repair_triage: true`) to exercise it on
the next build; it had not yet run live at flip time, so the first run with the
telemetry is the real validation of the test-bug-vs-code-gap split and the
regeneration hit-rate. Revert to `false` if that run shows a poor hit-rate or
any false-divert.
