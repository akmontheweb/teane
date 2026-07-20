# ADR-0002: Generation-Side Prevention of Contradictory Test Batches

**Status:** Accepted (Python Tier A shipped; see Action Items for the residue)
**Date:** 2026-07-20
**Deciders:** Teane harness maintainers
**Related:** [[ADR-0001]] (test-author regeneration for unsatisfiable tests)

## Context

ADR-0001 built a **recovery** path: once the repair loop declares a
tamper-guarded test `UNSATISFIABLE_TEST`, an autonomy ladder either
regenerates it (Tier A: machine-provable) or escalates. That path is
reached only *after* a bad test has landed, the build has gone red, and the
repair LLM has volunteered the declaration.

Lumina session `019f803f` showed a failure that path never catches. The
test-generation phase emitted a **same-input / opposite-expectation pair
split across two files**:

- `server/tests/test_contact_schemas.py` requires
  `ContactUpdate(first_name="   ")` to **raise** `ValidationError`
  (schema-layer rejection).
- `server/tests/test_contact_service.py` **constructs** the identical
  `ContactUpdate(first_name="   ")` — requiring it to *succeed* — then
  expects the service to raise `HTTPException(422)`.

No production code satisfies both: the call must both raise and not-raise at
construction. The repair loop is forbidden from editing tests, so it
oscillated for ~2.5 hours and ~$1.75 — flipping the schema between "raise"
and "strip-to-None", turning one test green and the other red each round —
before a `ReadTimeout` and an operator kill. ADR-0001's machinery stayed
dormant the entire time (full analysis below).

**Why ADR-0001 did not fire (two structural gaps):**

1. **The trigger gate cannot open for a contradictory *pair*.** The
   `UNSATISFIABLE_TEST` escape is offered only when the repair judge finds
   *nothing* production-side to blame — only a tamper-guarded test file. But
   for whichever single test is red this round, there genuinely *is* a
   schema edit that greens it (the one that re-reddens the other). So the
   judge always names a production file, the escape is never offered, the
   LLM never declares, and `route_after_unsatisfiable` — where the
   deterministic detector lives — is never reached. The pair defeats the
   exact judgment meant to catch it. (10 repair rounds, 10 `PROGRESS`
   verdicts, 0 `STUCK`, 0 escapes offered.)
2. **The Tier A detector is single-file.** `find_contradictions(source)`
   scans one file; this contradiction is cross-file, so even if the gate
   opened, per-file scanning sees neither half as self-contradictory.

**Why the test-author wrote it in the first place (the true origin):**

The SRS scenario "Update with missing first name" specifies a *behavioral
outcome* ("error displayed, no update") with **no enforcement layer named**
— correct for a user story. The implementer resolved that ambiguity as
**defense-in-depth**, enforcing the rule at *both* the schema (a
`model_validator` that raises) and the service (a defensive
`HTTPException(422)`). The test-author then wrote **one unit test per
enforcement site**, and the two sites demand opposite construction
semantics of the same object. Defense-in-depth on a *constructor-level*
validation is not independently unit-testable per layer: the upstream
rejection shadows the downstream check, making the downstream test
construct an object the schema forbids. Even with both files in a single
generation call, the author reasoned per-module against the behavioral AC
rather than the *reachable* code contract — the "anchor UTs on code, not the
SRS" principle ([[teane-test-traceability-model]]) violated at *generation*
time, not just regeneration.

**Forces at play:**

- **North-star (overriding):** autonomous end-to-end processing. A run
  should not burn hours on a defect a cheap deterministic check catches in
  milliseconds, and it should never dead-end at a human for a
  machine-provable test bug.
- **Safety invariant (non-negotiable):** the make-it-pass loop must never
  edit tests. Prevention lives in the *test-author* phase, whose job is
  writing tests — it is not a laundering path for the repair loop.
- **Reality:** a contradictory generated test is a defect of the
  test-author. The cheapest place to fix it is before it ever reaches the
  build — the author still has the full batch in context and can reconcile
  in one re-prompt.

## Decision

Add a **generation-side prevention layer** that operates *before* the build,
complementing (not replacing) ADR-0001's post-build recovery. Two parts:

### 1. Prompt rules 6–7 (soft, cheap, up-front)

Extend the test-author's format reminder
(`harness/test_generation.py::_build_format_reminder`) with two rules phrased
as **local, checkable actions** against code already in the prompt:

- **RULE 6 — one enforcement layer per validation.** When an input is
  rejected at construction, assert that rejection at the schema; do not also
  write a downstream test that constructs the same invalid value. If
  production enforces the rule redundantly downstream, that check is
  unreachable for invalid input — do not unit-test it by constructing
  invalid input.
- **RULE 7 — constructibility before use.** Before `X(value)` is passed
  onward, check X's validators. If they reject `value` at construction,
  assert the rejection on `X(value)` itself. Across the batch, never require
  the same `X(v)` to both raise and succeed.

Prompt rules lower the *rate* but cannot drive it to zero (nondeterminism,
long context, genuine mis-modeling of layer composition). They are the cheap
first layer, not the guarantee.

### 2. Deterministic cross-file scan + author bounce (the guarantee)

- **`test_contradiction.find_contradictions_across(files)`** — the
  generalization of the single-file Tier A detector to the whole test batch
  a single generation call emits. Same conservative bias (identical
  normalized call, distinct locations only); reports a signature required to
  RAISE in one `(file, test)` and SUCCEED in a different `(file, test)`,
  naming both files.
- **A generation-time gate** in `test_generation_node`, after the `@tests`
  marker gate and **before the build**: scan the just-generated Python
  tests; on a hit, **bounce back to the author** with a re-prompt quoting
  the exact unsatisfiable pair and RULE 6/7 guidance. Bounded by
  `test_generation.max_contradiction_reprompts` (default 2). This is the
  critical routing choice — the bounce goes to the **test-author, never to
  repair**, because repair is forbidden from editing tests (routing there is
  the trap that caused the 2.5h oscillation).

### Routing

```
test_generation_node
   ├─ @tests marker gate  (existing)
   ├─ cross-file contradiction scan  (NEW)
   │     ├─ hit, reprompts < cap ─▶ re-prompt AUTHOR (name the pair) ─▶ re-scan
   │     └─ hit, cap reached ──────▶ log + telemetry ─▶ proceed to build
   └─ deterministic sandbox run  (existing)
```

## Options Considered

### Option A: Prompt rules only

| Dimension | Assessment |
|-----------|------------|
| Complexity | Very low |
| Effectiveness | Partial — probabilistic; the model already had both files in context and still contradicted |
| Cost | ~free |

**Cons:** No guarantee. The failure is a *global* property across files; a
general instruction asks the model to spot the same property it missed.

### Option B: Deterministic scan only (route to repair on hit)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low–medium |
| Effectiveness | Catches it, but **routes wrong** |
| Safety | Broken if it routes to repair |

**Cons:** Routing a contradiction to the repair loop reproduces the exact
019f803f deadlock — repair cannot edit tests. Rejected as specified;
salvaged by bouncing to the author instead.

### Option C: Prompt rules 6–7 + cross-file scan + author bounce (this ADR)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium (detector generalization, node gate, config) |
| Effectiveness | High — soft prevention catches the common case, the deterministic scan is the backstop |
| Safety | High — prevention lives in the author phase; no test-editing by repair |

**Pros:** The prompt rules make the scan rarely fire; the scan converts
"usually avoids it" into "never ships it"; the bounce keeps the fix in the
phase that owns tests. Reuses the existing AST scanner.
**Cons:** Python-only for now; residue after the cap still flows to the
build.

## Trade-off Analysis

The tension is **coverage vs. footprint**. Option A is the smallest change
but leaves a provable defect class uncaught. Option B catches it but, as
first specified, re-creates the deadlock by routing to the wrong phase.
Option C resolves both: deterministic detection (a checkable property, not a
model judgment) plus a bounce that respects the anti-reward-hack invariant
by keeping test edits in the author phase. Cost on the lumina case is one
extra cheap re-prompt versus ~2.5 hours of oscillation.

This layer is *upstream* of ADR-0001. ADR-0001 recovers after a bad test
lands and the build reddens; ADR-0002 prevents the bad batch from reaching
the build at all. The two are independent and both wanted: prevention is
cheaper, but recovery still covers tests that only reveal their conflict at
runtime, and non-Python stacks the AST scan does not yet parse.

## Consequences

**Easier:**
- A whole provable class of generation defects is caught in milliseconds
  before any build, at the author, with the offending pair named.
- The repair loop is no longer handed batches it structurally cannot fix.

**Harder:**
- Two more prompt rules and a node gate to reason about; a new config knob.
- The cross-file detector must stay as conservative as the single-file one —
  a false positive wrongly bounces a valid batch.

**To revisit:**
- Whether the cap-exhausted residue should route to `test_regeneration_node`
  rather than proceed to the build.
- Extending the scan to JS/TS (the AST detector is Python-only).

## Action Items

1. [x] `find_contradictions_across(files)` — cross-file same-input /
       opposite-expectation detector; `Contradiction` carries per-side
       filenames; `describe()` names both files.
2. [x] Prompt RULES 6–7 in `_build_format_reminder`
       (single-enforcement-layer + constructibility).
3. [x] Generation-time gate in `test_generation_node`: scan Python tests
       after the `@tests` gate, bounce to the **author** on a hit, bounded
       by `test_generation.max_contradiction_reprompts` (default 2).
4. [x] Config knob registered for type-validation and dashboard rendering
       (`_TYPE_SCHEMA` + `_KNOWN_NESTED_KEYS["test_generation"]`).
5. [ ] Make ADR-0001's **repair-side** detector cross-file too (consume
       `find_contradictions_across` at `route_after_unsatisfiable`), so
       recovery covers the same shape prevention now catches.
6. [ ] Route cap-exhausted residue to `test_regeneration_node` instead of
       proceeding to the build.
7. [ ] Extend the contradiction scan to JS/TS test batches.
8. [ ] Telemetry: track `test_generation_contradiction_detected` rate to
       measure how often prompt rules 6–7 fail and the scan has to fire.
