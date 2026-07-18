# ADR-0001: Test-Author Regeneration Path for Unsatisfiable Tests

**Status:** Proposed
**Date:** 2026-07-18
**Deciders:** Teane harness maintainers

## Context

When the repair loop cannot make a red build green because the failing
assertion lives in a **test file**, it is forbidden from editing the test
(reward-hacking guard — `harness/patcher.py` `[test-protected]`, rationale in
`harness/patch_feedback.py:108`: *"weaken tests to make a red build pass — that
is reward-hacking, not a fix"*). The repair LLM's only escape is to emit a
single `UNSATISFIABLE_TEST: <path> — <reason>` line
(`harness/graph.py:17242`), which sets `node_state["unsatisfiable_test"]`.
`route_after_compiler` (`harness/graph.py:19794`) then re-validates the test is
still failing and routes to `human_intervention_node` — a **blocking HITL
stop**.

Observed failure (lumina session `019f73a0`): the test-generation phase emitted
a **self-contradictory test pair** in `tests/backend/test_contact_models.py` —
`test_none_fields_allowed` expects `ContactUpdate(first_name=None)` to succeed,
while `test_all_none_raises` expects the *identical* call to raise. No
production change can satisfy both. Both DeepSeek and Kimi correctly diagnosed
this; DeepSeek burned 4 cheap shots re-attempting the (rejected) test edit
before escalation, and the run ultimately dead-ended at HITL.

**Forces at play:**

- **North-star (overriding):** teane's goal is autonomous end-to-end
  processing. HITL is a *failure of automation*, acceptable only when there is
  absolutely no other way out — in the code, the design, or a redesign. Every
  arrow that points at `human_intervention_node` must first be interrogated for
  an autonomous alternative.
- **Safety invariant (non-negotiable):** the make-it-pass loop must never be
  able to edit the spec (the tests). Whatever we build cannot become a
  laundering path for the same reward-hack the `[test-protected]` guard exists
  to prevent.
- **Reality:** a *contradictory* or *impossible* generated test is a defect of
  the **test-author** (`test_generation_node`), not the code-fixer. Fixing it is
  legitimate work for the test-author phase, which already writes tests from the
  spec. The current design has no wire from "test declared unsatisfiable" back
  to that phase — it only knows how to call a human.

## Decision

Introduce a **test-author regeneration path** for declared-unsatisfiable tests,
governed by an explicit **autonomy ladder**: every candidate HITL exit must
climb the ladder and be shown to have *no autonomous rung left* before it is
allowed to halt for a human. Regeneration is code-contract-anchored (unit
tests link to code, not the SRS), runs as the test-author phase (not the
code-fixer), and is gated by a mechanical
anti-weakening check.

### The autonomy ladder (applied at every `→ human_intervention_node` arrow)

A stop is only legitimate at the bottom rung. Climb from the top; halt only when
no higher rung applies.

1. **Code rung — deterministic resolution.** Can the defect be resolved with no
   model judgment at all? The machine-checkable classes qualify: two tests with
   identical input and opposite expected outcomes; a test that fails to parse; a
   structurally impossible assertion. Resolve deterministically.
2. **Contract-inference rung.** For a *unit* test the source of truth is the
   CODE it maps to (1:1), never the SRS — teane's traceability model links unit
   tests to code (`@tests`), not to stories/ACs. Consult, in order: the mapped
   code module's contract (signatures, docstrings, validators, type hints), the
   sibling passing tests in the file, and only *then* the SRS as a **tiebreaker**
   for genuinely ambiguous intent. (Lumina: the `contact.py` model already
   carries a validator that raises *"at least one field must be provided"*, so
   the code contract alone resolves it — `ContactUpdate(first_name=None)` must
   raise, making `test_none_fields_allowed` the defective half. The SRS agrees
   but isn't needed.)
3. **Spec-consistent-default rung (async provenance, not a blocking prompt).**
   If inference underdetermines intent, pick the most spec-consistent
   interpretation, apply it, and record it as a **reviewable decision**
   (file annotation + staged post-mortem rule + dashboard flag). The human veto
   becomes *asynchronous and optional* — processing continues; the operator
   reviews after the run.
4. **Escalate-up-the-pipeline rung.** If the ambiguity's true source is an
   under-specified requirement, loop back to the requirements/spec-discovery
   phase (the spec-author) to tighten the AC — resolving it at source without a
   human.
5. **HITL rung (irreducible residue only).** A genuinely irreducible conflict —
   e.g. two ACs in the SRS that contradict *each other* with no artifact
   breaking the tie. Even here the output is a precise, decision-ready question,
   never an open-ended stop.

`human_intervention_node` must additionally record **why autonomy was
exhausted** (which rungs were attempted), so the HITL surface is measurable and
can be driven toward zero over time.

### Component: `test_regeneration_node`

Distinct from `test_generation_node` (which writes *new* tests for
just-patched source). This node *regenerates one declared-defective UNIT test
as a comprehensive suite for its 1:1-mapped code module, anchored on the code
contract*.

- **Inputs (code-first):** the mapped code module — located via the defective
  test's `# @tests: <source>` marker — as the primary contract (signatures,
  docstrings, validators, public symbols); the defective test file + its sibling
  tests; the failing pytest output and the repair LLM's `UNSATISFIABLE_TEST`
  reason; the SRS **only as a tiebreaker**, never cited in the test.
- **Authority:** the CODE contract — **never** "make the build green." If the
  corrected assertion then fails because the *production* code is wrong, that
  failure correctly flows back to the repair loop (production fix), not to
  another test rewrite.
- **Output:** a full `REWRITE_FILE` of that one test file — a comprehensive unit
  suite over the module's public surface — or a give-up signal that advances the
  ladder.

### Anti-reward-hack gate (mechanical, not trust-based)

Before accepting a regenerated test:

1. **Coverage non-regression** — the regen must not weaken/reduce assertions for
   the same behavior. A regen that deletes the contradictory assertions and
   asserts nothing is rejected. *This is the gate that stops "regeneration" from
   becoming "gut the test."*
2. **Code linkage required** — the regen must keep its `# @tests: <source>`
   marker (unit tests link to code, never to stories/ACs — teane's traceability
   model); no marker → roll back and advance the ladder. A public-symbol
   coverage advisory drives toward an exhaustive per-module suite.
3. **Principled contradiction resolution** — for the machine-checkable pair,
   keep the test consistent with the code contract, fix/remove the other, log
   the driving symbol.

### Routing

At the existing `unsatisfiable_test` branch (`harness/graph.py:19811`), replace
the unconditional `→ human_intervention_node` with `route_after_unsatisfiable`:

```
unsatisfiable_test (still failing)
        │
        ▼
 route_after_unsatisfiable  (climbs the autonomy ladder)
   ├── rung 1–2 resolvable, attempts<cap ─────────────▶ test_regeneration_node ─▶ compiler_node (re-verify)
   ├── rung 3 (default+provenance, tier_b_auto) ───────▶ test_regeneration_node ─▶ compiler_node
   ├── rung 4 (spec under-specified) ──────────────────▶ requirements/spec-discovery phase
   └── rung 5 (irreducible) ───────────────────────────▶ human_intervention_node  (records rungs attempted)
```

Verification outcomes after regen re-runs the sandbox build:
- **Passes + coverage holds** → clear flag, back to `compiler_node`.
- **Fails because production code is wrong** → clear flag, route to
  `repair_node` (the virtuous case: a broken test became a real one).
- **Still test-side-failing / coverage-regressed / attempts exhausted** →
  advance the ladder (default+provenance, then HITL).

## Options Considered

### Option A: Keep blocking HITL (status quo)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low (no change) |
| Autonomy | Poor — every bad generated test halts the run |
| Safety | High (test-protection absolute) |
| Team familiarity | High |

**Pros:** Simplest; zero risk of automated test tampering.
**Cons:** Directly violates the north-star; a single buggy generated test
strands an otherwise-autonomous run; wastes cheap-shot + escalation budget
reaching a dead end.

### Option B: Let the repair loop edit tests when it declares them wrong

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Autonomy | High |
| Safety | **Unacceptable** — reintroduces reward-hacking |
| Team familiarity | High |

**Pros:** Maximally autonomous, trivial to implement.
**Cons:** Destroys the invariant that makes green builds mean something; the
make-it-pass loop will weaken tests. Rejected outright.

### Option C: Test-author regeneration + autonomy ladder (this ADR)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium (new node, routing helper, gates, config) |
| Autonomy | High — HITL narrows to irreducible spec conflicts |
| Safety | High — code-contract-anchored authority + mechanical anti-weakening gate; different phase from the code-fixer |
| Team familiarity | Medium (reuses test_generation machinery) |

**Pros:** Preserves the safety invariant (regen authority is the spec, not the
build state; runs as the test-author); climbs autonomous rungs before any human;
converts a class of dead-ends into either autonomous fixes or real production
repairs.
**Cons:** More moving parts; the coverage-non-regression heuristic needs tuning;
Tier B (model-declared-only) carries residual trust risk, mitigated by config
gating.

## Trade-off Analysis

The central tension is **autonomy vs. the anti-reward-hack invariant**. Option B
maximizes the former by sacrificing the latter — unacceptable. Option A does the
reverse — unacceptable under the north-star. Option C resolves the tension by
changing *who* is allowed to edit the test and *on whose authority*: the
test-author phase, anchored to the spec, never the code-fixer anchored to the
build result. The mechanical coverage-non-regression gate converts "trust the
model's claim the test is wrong" into "verify the regen didn't weaken coverage"
— a checkable property rather than a judgment call. The autonomy ladder ensures
we never settle at a HITL stop that a higher rung could have resolved, and makes
the residual HITL surface explicit and measurable.

Residual risk is concentrated in **Tier B** (the LLM declares unsatisfiability
but it isn't machine-provable). This is gated behind `tier_b_auto` (default
false) so the conservative rollout automates only the provable Tier A cases.

## Consequences

**Easier:**
- Autonomous runs survive buggy generated tests (the common case: spec is clear,
  only the generated test was wrong).
- HITL narrows from "any test repair can't fix" to "the spec itself is
  ambiguous," and even that is pushed to async review where possible.
- A broken test that masked a real bug converts into a legitimate production
  repair.

**Harder:**
- More graph surface to reason about and test; a new node lifecycle.
- The coverage-non-regression check is heuristic and will need telemetry-driven
  tuning to avoid both false accepts (weakened tests) and false rejects (valid
  simplifications).

**To revisit:**
- Whether Tier B should ever auto-run, based on Tier A telemetry.
- Whether rung 4 (escalate to spec-author) needs its own guard against
  spec-rewrite reward-hacking.
- Metric target: HITL stops per autonomous run trending toward zero.

## Action Items

1. [x] Machine-checkable defect **pre-filter** (Tier A) — `harness/test_contradiction.py`:
       contradictory-pair detector (same input, opposite expectation) +
       unparseable-test detector. (Impossible-assertion detector deferred.)
2. [x] `route_after_unsatisfiable` encoding the autonomy ladder; replaces the
       direct `→ human_intervention_node` at the unsatisfiable branch.
3. [x] `test_regeneration_node` — **code-first** inputs (mapped module via the
       `@tests` marker + public symbols + sibling tests; SRS tiebreaker only),
       `REWRITE_FILE` output, give-up signals; graph edges to `compiler_node`
       (re-verify) / `repair_node` (production fix).
4. [x] Gates: **coverage-non-regression** + **code-linkage** (`@tests` marker),
       only-declared-path, and a public-symbol coverage advisory.
5. [ ] Add rung-3 async-provenance path (annotation + staged rule + dashboard
       flag) so a spec-consistent default never blocks processing.
6. [ ] Wire rung-4 escalation back to the requirements/spec-discovery phase.
7. [ ] Make `human_intervention_node` record *which rungs were attempted* and
       *why autonomy was exhausted*; surface a HITL-per-run metric.
8. [x] Config block `test_regeneration` (`enabled`, `max_attempts_per_test`,
       `tier_b_auto=false`, `require_code_linkage=true`,
       `coverage_nonregression=true`) + strict-validation keys.
9. [x] Shipped **Tier A only** first (provably safe, covers the lumina case);
       Tier B gated behind `tier_b_auto` pending telemetry.
