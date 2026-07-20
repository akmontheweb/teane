# ADR-0003: Hybrid Deterministic + LLM Unit-Test Generation

**Status:** Accepted (Tier 1 / Python schema-declarative prototyped; Tiers 2–3 phased — see Action Items)
**Date:** 2026-07-20
**Deciders:** Teane harness maintainers
**Related:** [[ADR-0001]] (repair-side test regeneration), [[ADR-0002]] (generation-side contradiction prevention)

## Context

ADR-0001 and ADR-0002 both treat the *symptom* of a single failure mode:
the LLM writing bad unit tests. 0001 recovers after a bad test reddens the
build; 0002 prevents a contradictory batch from reaching the build. Neither
asks the prior question — **should the LLM be writing these tests at all?**

A measured look at where the pain goes (session 019f803f post-mortem, plus
the commit and guard-comment record) found:

- Test-related fixes **tripled** as a share of recent engineering effort
  (~8% → ~22% of commits) as infra/provider churn settled out.
- Weighted by *cost*, tests dominate far beyond their commit share: the
  019f803f contradiction alone burned ~2.5h and ~$1.75, versus one-line
  gateway fixes. The new incident telemetry (cause × cost × wall-clock)
  exists to keep measuring this.
- **Every expensive test incident in the record is the *contract-derivable*
  class** — schema validation, constructibility, marker/scope mechanics,
  test-env type errors — not the semantic core (birthday math, sort order,
  the 7-day window). We could not find a marquee deadlock caused by a
  *semantically wrong business-logic assertion*.

The reason is structural. A unit test asserts *intended* behavior; code
encodes *actual* behavior. But for the contract-derivable class the intent
is **already formally pinned in the code**: a Pydantic `Field(max_length=100)`
*is* a machine-readable spec of its own validation contract. Asking an LLM to
re-derive that in prose is asking it to guess at something already written
down — and it guesses inconsistently (019f803f: it disagreed with *itself*
about which layer rejects an empty `first_name`). The LLM is fumbling
precisely the part that needs no judgment, while the part that genuinely
needs judgment is quietly *not* where the deadlocks come from.

**The seam already exists.** `test_generation_node` already interleaves
deterministic test-code emission with the LLM pass — `_emit_nfr_stubs`
renders real `def test_ac_N()` bodies deterministically and records their
markers; `_ensure_js_test_env` scaffolds the jest/type environment (603d3e8);
`_ensure_pytest_importlib_config` writes pytest config. This ADR widens that
existing path rather than inventing a new one.

**Forces at play:**

- **North-star:** autonomous processing. A deadlock the code already
  determines the answer to should never reach a human, and should never be
  generated wrong in the first place.
- **Correctness of the oracle:** tests derived *purely* from code are
  tautological — they pin current behavior, they cannot verify *correct*
  behavior. This is why the LLM cannot be fully removed: genuine business
  logic needs an intent oracle the code doesn't contain.
- **Safety:** the harness runs untrusted generated code. A deterministic
  emitter must **not import** generated modules to introspect them (arbitrary
  code execution); it must work from the AST.
- **Single source of truth:** whichever layer owns a validation, only *one*
  test should assert it. Two layers testing the same construction with
  opposite expectations is the 019f803f contradiction — impossible once a
  deterministic pass owns the schema-construction surface.

## Decision

Generate unit tests with a **hybrid** strategy, split by whether the intended
behavior is *mechanically derivable from the code contract*:

### Tier 1 — Schema-declarative (deterministic, AST-derived)

From each generated model's **declarative** constraints, emit provably-correct
tests with no model judgment:

- `Field(max_length=N)` → boundary pair: length N constructs, N+1 raises.
- `Field(ge=…, le=…, gt=…, lt=…)` → in-range constructs, out-of-range raises.
- required (non-`Optional`, no default) → omitting it raises.
- type annotations → coercion / wrong-type rejection.
- serialization round-trip from the model shape.

Conservative by construction: for any field whose type the emitter cannot
synthesize a valid value for, it emits **no** test for that model (a missing
test is safe; a wrong test is not — same bias as the contradiction detector).

### Tier 2 — API contract (deterministic, route-derived)

From the web framework's route table (FastAPI decorators / an OpenAPI dump),
emit status-code and schema-validation tests at the HTTP boundary: declared
`response_model` shapes, 422 on invalid body, 404 on missing resource where
the route contract expresses it. Runs against the app in the sandbox, never
by importing it in-process.

### Tier 3 — Property-based (deterministic strategies, LLM-free invariants)

Derive Hypothesis strategies from type hints + declarative constraints and
assert **structural** invariants that hold regardless of business logic:
serialization round-trips, idempotence, type/range of outputs. Explicitly
*not* value-correctness (that's Tier 4). Strategy tuning is the known-fiddly
part (ADR-0001 flagged it); Tier 3 ships behind a config flag, off by default,
until its false-positive rate is measured.

### Tier 4 — Semantic (LLM, narrowed)

Everything the deterministic tiers cannot derive: business logic where correct
≠ derivable-from-types (age math, "within 7 days", sort-by-closest-date), and
custom-validator semantics whose exact triggering input is a judgment call.
The LLM is told **which surfaces the deterministic tiers already own** and
instructed to write *only* the semantic remainder. This narrowing is the
contradiction fix at the source: the LLM never writes schema-construction
tests, so it cannot contradict the deterministic ones.

### Ordering in `test_generation_node`

```
test_generation_node
   ├─ Tier 1/2/3 deterministic emit  (NEW; mirrors _emit_nfr_stubs)
   │     → record @tests markers, note covered surfaces
   ├─ LLM dispatch (Tier 4)          (narrowed: "schema/API tests exist;
   │     write ONLY business-logic assertions for X, Y, Z")
   ├─ @tests marker gate             (existing)
   ├─ cross-file contradiction gate  (ADR-0002 — now a backstop)
   └─ deterministic sandbox run      (existing)
```

The deterministic tiers become the **single source of truth** for their
surfaces, so ADR-0002's gate and RULE 6/7 stop being load-bearing for the
covered classes and become cheap insurance.

### The honest boundary

The clean deterministic line is **declarative vs imperative**. Declarative
`Field(...)` constraints are unambiguous and fully deterministic. Custom
`@field_validator` / `@model_validator` bodies are a gray zone: the AST can
see *that* a validator raises and roughly on which branch, but pinning the
exact triggering input is semantic. Notably, 019f803f's whitespace rejection
was a *custom* validator — Tier 1 would not have generated that specific
assertion. What Tier 1 **does** is claim ownership of the schema-construction
surface, so the LLM never writes the *contradicting* service-side version.
The contradiction dies; the specific whitespace assertion stays Tier 4.

## Options Considered

### Option A: Status quo — LLM writes all unit tests

| Dimension | Assessment |
|-----------|------------|
| Autonomy | Poor — the contract-derivable class is the top incident-cost source |
| Correctness | LLM guesses at contracts already written in code, inconsistently |
| Complexity | None (no change) |

**Cons:** Leaves the measured, expensive failure class in the hands of the
component worst-suited to it.

### Option B: Fully deterministic — no LLM in test generation

| Dimension | Assessment |
|-----------|------------|
| Autonomy | High |
| Correctness | **Broken** — tests derived from code are tautological about intent |
| Complexity | High for the semantic tier (no oracle exists) |

**Cons:** Cannot verify business-logic correctness; a bug present at
generation time is baked into the test. Rejected.

### Option C: Hybrid, split by derivability (this ADR)

| Dimension | Assessment |
|-----------|------------|
| Autonomy | High — the expensive class becomes deterministic |
| Correctness | Sound — deterministic where intent is in the code; LLM where it isn't |
| Complexity | Medium — new emitter with per-stack, per-tier backends |

**Pros:** Removes the LLM from exactly the class causing the pain; makes the
covered surface a single source of truth (kills contradictions structurally);
keeps the LLM where it's genuinely load-bearing and *not* currently failing.
**Cons:** Real build (per-stack backends); Tier 3 strategy tuning; the
custom-validator gray zone.

## Trade-off Analysis

The tension is **autonomy/correctness vs. build cost**. Option B maximizes
determinism but sacrifices the intent oracle — unacceptable. Option A is free
but leaves the measured cost centre unaddressed. Option C spends real
engineering to move *only* the derivable class to deterministic generation,
where determinism is both correct (intent is in the code) and eliminating (a
single source of truth cannot contradict itself). The residual risk
concentrates in Tier 3 (property strategies) and the custom-validator gray
zone, both gated/phased so the conservative, high-payoff Tier 1 ships first.

Crucially, this **subsumes** ADR-0002 for the covered surface: prevention by
persuasion (RULE 6/7) and detection (the contradiction gate) are replaced by
*structural impossibility* — the LLM doesn't write those tests at all.

## Consequences

**Easier:**
- The top incident-cost class (contract-derivable tests) becomes
  deterministic and correct-by-construction.
- Contradictions on the covered surface become impossible, not merely caught.
- The LLM's test surface shrinks to where it actually adds value, reducing
  its error rate on everything else.

**Harder:**
- A new module with per-stack (Python-first; TS/zod later) and per-tier
  backends to maintain.
- The AST value-synthesizer must stay conservative (skip what it can't prove).
- Tier 3 needs false-positive telemetry before it can default on.

**To revisit:**
- Whether Tier 3 ever defaults on, from its measured false-positive rate.
- Extending Tiers 1–2 to the TS stack (zod schemas / TS route contracts).
- Whether the custom-validator gray zone is worth a heuristic emitter or
  stays Tier 4.

## Action Items

1. [x] **Tier 1 (Python schema-declarative) prototype** — `harness/contract_tests.py`:
       AST-parse generated Pydantic models, synthesize minimal valid instances,
       emit boundary/required/type tests with `@tests` markers; conservative
       skip on unsynthesizable fields; idempotent. Wired into
       `test_generation_node` before the LLM dispatch.
2. [ ] **Narrow the LLM prompt** — tell the Tier-4 dispatch which model
       surfaces the deterministic pass already covered; instruct it to write
       only business-logic assertions.
3. [ ] **Tier 2 (API contract)** — FastAPI/OpenAPI route-derived status-code
       tests, executed in the sandbox.
4. [ ] **Tier 3 (property-based)** — Hypothesis strategies from type hints;
       structural invariants only; config-gated, default off; false-positive
       telemetry.
5. [ ] **Config block** `test_generation.contract_tests`
       (`enabled`, `tiers`, `property_based=false`) + validation + form schema.
6. [ ] **Measure** — use the ADR-0002/incident telemetry `test_share` before
       and after to confirm the contract-derivable incident class drops.
7. [ ] **TS stack** — zod schema + TS route contracts (Tiers 1–2 for TS).
