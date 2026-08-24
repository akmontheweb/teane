# ADR-0007: Decomposition-Quality Review

**Status:** Accepted (design ratified 2026-08-21; **Phase 1 in progress** — deterministic gate + LLM review node, advisory; Phase 2 auto-remediation pending, see Action Items). Default off.
**Date:** 2026-08-21
**Deciders:** Teane harness maintainers
**Related:** [[ADR-0004]] (constraint-NFR embedding at decomposition), [[ADR-0005]] (shift-left test quality), [[ADR-0006]] (in-build acceptance verification). Builds on the existing `harness/semantic_review.py` (coverage review) and `harness/spec_reconciler.py` (structural reconciliation).

## Context

The decomposition artifact — the features / stories / acceptance criteria that
`decomposition_node` produces from the SRS — is the contract the entire build
executes against. Everything downstream inherits its quality: each story is one
patching pass, each AC becomes a test ([[ADR-0005]]) and, now, an in-build
acceptance check ([[ADR-0006]]). A bad decomposition is not a cosmetic problem —
it propagates:

- **An over-large story** ("epic masquerading as a story") overloads a single
  patching pass, blows the emission/repair budget, and is the kind of thing that
  trips the no-progress / hard-iteration caps.
- **Overlapping or duplicate stories** produce duplicate or conflicting code
  across passes (the same amplifier [[ADR-0005]] fought in the test suite, one
  level up).
- **A vague or non-atomic AC** ("works correctly", "handles errors and validates
  input and logs") produces a weak or contradictory test at generation time
  ([[ADR-0005]]'s exact churn source) and is **unverifiable in-build** — [[ADR-0006]]'s
  acceptance run can only assert a criterion that is atomic and testable.
- **A wrong or missing `depends_on`** mis-orders the batch plan; [[ADR-0006]]'s
  dependency-blocked deferrals grow, and stories build against prerequisites that
  don't exist yet.

**What reviews decomposition today, and what it misses.** Three mechanisms touch
the artifact, and none judges craftsmanship:

1. `spec_reconciler_node` (deterministic, always on) rewrites the LLM's stories
   against the spec's authoritative IDs — catching renumbering, silent
   story-drop, phantom features. **Structural integrity, not quality.**
2. `semantic_review.py` (`semantic_coverage_review_node`, LLM, **off by default**)
   asks, per feature, whether the stories *together satisfy the feature's intent*
   vs. merely cite it — and flags one-sided client↔server contracts.
   **Coverage/intent, not craft.** It is also disabled (`traceability.semantic_review`
   absent), so the one cross-model reviewer that exists (`NodeRole.DECOMPOSITION_REVIEWER`,
   e.g. `moonshot:kimi-k3`, independent of the `deepseek` decomposer) never runs.
3. `gap_fill` drafts a covering story for a *feature with no stories* — coverage
   existence, not quality.

So there is **no review of decomposition craftsmanship**: story sizing/atomicity,
overlap/duplication, AC testability/atomicity/ambiguity, dependency correctness,
and over-/under-decomposition are left entirely to the decomposition model's own
output. That surface is exactly where a cheap upstream check prevents expensive
downstream churn — the [[ADR-0005]] thesis, applied to the artifact that seeds
the whole build.

**Forces at play:**

- **North-star autonomy / minimal HITL.** A quality review that stalls the build
  at HITL on a subjective judgement is worse than no review. Enforcement must be
  bounded and default-advisory, never an infinite loop (the finsearch/ciod
  lesson that shaped [[ADR-0006]] and the traceability gate).
- **Determinism where possible.** Duplicate stories, dangling/circular
  dependencies, zero-AC stories, and dependency references to non-existent
  stories are *provable* — they need no LLM and carry no false-positive risk.
  Only the subjective dimensions (sizing, atomicity, testability, overlap
  semantics) need a model. This is the [[ADR-0003]]/[[ADR-0005]] two-tier split.
- **Independent judgement.** Quality review must not run on the model that wrote
  the decomposition — the `DECOMPOSITION_REVIEWER` role already provides a
  cross-model reviewer; it should judge in a reasoning mode (the config default
  routes it `non_thinking`, wrong for a judgement task).
- **Comprehensive ≠ sprawling.** The review must push toward *right-sized*
  decomposition, not maximal story count.

## Decision

Add a **two-tier decomposition-quality review** that runs after reconciliation,
before the STORIES gate — a deterministic gate plus an adversarial LLM review —
emitting structured, actionable findings, **advisory by default**, with an
optional enforce and a phased bounded auto-remediation.

### 1. Deterministic quality gate (no LLM)

Provable defects, computed from `state.db` (stories: `title`, `description`,
`depends_on`, `scope_files`, `feature_id`; ACs: `text`, `ordinal`). Each is
high-confidence and false-positive-free:

- **Dangling dependency** — a `depends_on` `story_key` that no story defines.
- **Circular dependency** — a cycle in the `depends_on` graph.
- **Zero-AC story** — a story with no acceptance criteria (unverifiable,
  un-testable — nothing for [[ADR-0005]]/[[ADR-0006]] to hang on).
- **Duplicate story** — two stories with identical/near-identical normalized
  titles (and overlapping `scope_files`).
- **Empty/degenerate AC** — an AC whose text is empty or a placeholder.

These are reported as findings and, under enforce, block deterministically (like
the reconciler's structural errors).

### 2. Adversarial LLM quality review (`DECOMPOSITION_REVIEWER`)

For the subjective dimensions, hand the reviewer each feature's stories (title +
description + ACs + `depends_on`) and ask it to score against an **INVEST-derived
rubric**, returning strict JSON findings:

- **Right-sizing** — is a story too large (should be split; name the seams) or
  trivially small (should be merged)?
- **Overlap** — do two stories cover the same behaviour (should be merged /
  disambiguated)?
- **AC quality** — is each AC **atomic** (one behaviour, no "and/or" bundling),
  **testable** (an observable outcome, not an implementation detail), and
  **unambiguous** (no "works well" / "handles errors")?
- **Dependency correctness** — is a *needed* dependency missing (story B uses
  what only story A builds, but doesn't declare it)? (Complements the
  deterministic dangling/circular check.)
- **Decomposition balance** — is a feature over-decomposed (fragmented) or
  under-decomposed (one story hiding several behaviours)?

Each finding carries `story_key` (or pair), `dimension`, `severity`
(`high|medium|low`), `problem`, and a concrete `suggested_action`
(`split` / `merge` / `rewrite_ac` / `add_dependency` / `resize`). Mirrors
`semantic_review.review_semantic_coverage`: strict-JSON, fail-open (no reviewer
configured / dispatch error / non-JSON → no findings, never blocks on its own
error). The reviewer SHOULD run in a thinking mode
(`decomposition_reviewer_mode: thinking`) — this is a judgement task.

### 3. Remediation — phased

- **Phase 1 (advisory + enforce).** Log findings + emit a
  `decomposition_quality_findings` observability event; write a
  `docs/DECOMPOSITION_REVIEW.md`. Under `decomposition.quality_enforce`, print an
  actionable report and exit non-zero (operator revises the spec/decomposition),
  bounded by a cycle cap so a headless run fires once and exits — never
  ping-pongs (the `TRACEABILITY_BLOCK_CYCLE_CAP` pattern).
- **Phase 2 (bounded auto-remediation).** Feed the structured findings back to
  the decomposition model for ONE targeted revision pass (split/merge/rewrite the
  named stories/ACs only — not a full re-decomposition), re-reconcile, and
  proceed. Capped by `decomposition.max_remediation_cycles` (mirrors
  `gap_fill`'s `max_requirement_gap_fill_cycles`); deterministic-fixable findings
  (e.g. an obviously dangling dep) are applied without the LLM where safe. Never
  loops: after the cap, remaining findings downgrade to advisory and the build
  proceeds.

### Config (new `decomposition` section; all config-driven per no-hardcoded-caps)

`decomposition.quality_review` (bool, default **off**), `quality_enforce` (bool,
default off), `quality_auto_remediate` (bool, default off — Phase 2),
`max_remediation_cycles` (int), `max_stories_per_review` (int cap on prompt
size). Registered across `cli.py` (`_KNOWN_TOP_LEVEL_KEYS` if new section,
`_KNOWN_NESTED_KEYS`, `_TYPE_SCHEMA`) + `config_help.md`, and threaded into
`AgentState` like `test_generation_config`.

### Node contract & placement

```
decomposition_node → spec_reconciler_node → [semantic_coverage_review_node]
    → decomposition_quality_review_node   ← NEW (deterministic gate + LLM review)
    → route_after_decomposition_quality:
         enforce AND blocking findings AND under cap → (Phase 2) re-decompose / END
         else                                        → human_gatekeeper_node (STORIES gate)
```

`decomposition_quality_review_node` lives in a new `harness/decomposition_review.py`
(mirrors `semantic_review.py`), registered in `graph.py` and gated on
`decomposition.quality_review`; pass-through no-op when off, so the graph is
unchanged by default. It composes with `semantic_coverage_review_node` (coverage)
rather than replacing it — craft vs. intent are orthogonal.

## Options Considered

### Option A: Status quo — no decomposition-craft review

| Dimension | Assessment |
|-----------|------------|
| Complexity | None |
| Cost | Poor — bad stories/ACs surface as expensive downstream churn (patching overload, duplicate code, weak tests, unverifiable ACs) |
| Autonomy | Fine, but the defects it lets through are top HITL/loop-trip causes |
| Correctness | Decomposition quality is unmonitored |

### Option B: Two-tier (deterministic + LLM) advisory review, phased remediation (this ADR)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium — one node (mirrors semantic_review) + deterministic checks; Phase 2 remediation reuses gap_fill's bounded-cycle pattern |
| Cost | One extra reviewer call per run (deterministic tier is free); pays for itself by preventing downstream churn |
| Autonomy | High — advisory default, bounded enforce, no infinite loop |
| Correctness | Deterministic tier is provable; LLM tier is cross-model + fail-open |

**Pros:** attacks the seed of downstream churn at the cheapest point; reuses the
`DECOMPOSITION_REVIEWER` role and the `semantic_review`/`gap_fill` machinery;
deterministic tier gives zero-false-positive wins immediately; composes with
coverage review. **Cons:** the LLM tier is a judgement call that can
false-positive (hence advisory default + thinking-mode reviewer); Phase 2
remediation is a new revision path to bound carefully.

### Option C: LLM-only quality review

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Cost | One call |
| Correctness | Misses the cheap provable wins; pays LLM cost + false-positive risk for defects a deterministic check nails for free |

**Cons:** Wastes the [[ADR-0003]]/[[ADR-0005]] lesson — dangling deps, zero-AC
stories, and duplicates are provable and shouldn't be a model's judgement call.

### Option D: Deterministic-only quality gate

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Cost | Free |
| Correctness | Catches structural defects but NOT the subjective majority (sizing, atomicity, testability, overlap semantics) — which is most of decomposition quality |

**Cons:** The hardest, highest-value quality problems (is this AC testable? is
this story too big?) are inherently semantic. Necessary but insufficient — it is
tier 1 of Option B, not a standalone answer.

### Option E: Fold craft checks into the existing `semantic_review`

**Cons:** Conflates two orthogonal questions (does the decomposition *cover* the
intent vs. is it *well-formed*) into one prompt and one verdict, making both
weaker and the findings harder to act on. Keep them as sibling nodes sharing the
reviewer role.

## Trade-off Analysis

The tension is **upfront review cost vs. downstream churn**, one altitude above
[[ADR-0005]]. Option A spends nothing and pays in patching overload, duplicate
code, weak tests, and unverifiable ACs — the defects that most often trip the
no-progress/HITL rails. Option B spends the deterministic tier for free and one
reviewer call to catch the semantic majority, advisory by default so it can only
*inform* until the operator opts into enforce/remediation. The conservative
defaults bound the downside exactly as [[ADR-0005]]'s and [[ADR-0006]]'s do:
advisory + fail-open + bounded-cycle enforce means the review can add signal or a
bounded revision, never manufacture a stall.

It composes cleanly with the ADR chain: `spec_reconciler` owns structural
integrity, `semantic_review` owns coverage/intent, and ADR-0007 owns
craftsmanship — and it strengthens both [[ADR-0005]] (testable ACs → better tests)
and [[ADR-0006]] (atomic/testable ACs → verifiable in-build; correct deps → fewer
dependency-blocked deferrals) at their shared source.

## Consequences

**Easier:**
- Bad stories/ACs are caught before they seed the build, at the cheapest fix
  point — one revision vs. dozens of downstream repair/deferral rounds.
- The dormant cross-model `DECOMPOSITION_REVIEWER` finally earns its keep on a
  second, orthogonal axis.
- Testable/atomic ACs directly improve [[ADR-0005]] test quality and [[ADR-0006]]
  acceptance verifiability; correct dependencies shrink [[ADR-0006]] deferrals.

**Harder:**
- A new judgement surface with false-positive risk (mitigated: advisory default,
  thinking-mode reviewer, deterministic tier carries the provable load).
- Phase 2 auto-remediation is a new bounded revision path (split/merge/rewrite)
  that must never loop and must not fight the reconciler's authoritative IDs.
- An INVEST rubric encoded in a prompt is a tuning surface (severity thresholds,
  what counts as "too large").

**To revisit:**
- Whether `decomposition.quality_review` should default **on** (advisory) once
  false-positive rates are measured — like [[ADR-0004]]/[[ADR-0005]] flipped their
  flags after a live run.
- Whether right-sizing should be spec-driven (feature complexity) rather than a
  fixed heuristic.
- Whether Phase 2 remediation subsumes part of `gap_fill` (both are bounded
  decomposition-revision loops) into one revision engine.

## Action Items

1. [ ] **Deterministic quality gate** — `harness/decomposition_review.py`:
       dangling/circular deps, zero-AC stories, duplicate stories, empty ACs, from
       `state.db`. Pure, unit-testable, zero false positives. Findings + enforce
       block.
2. [ ] **LLM adversarial quality review** — INVEST-rubric prompt over
       features/stories/ACs on `NodeRole.DECOMPOSITION_REVIEWER`; strict-JSON
       findings (`story_key`, `dimension`, `severity`, `problem`,
       `suggested_action`); fail-open. Mirror
       `semantic_review.review_semantic_coverage`.
3. [ ] **`decomposition_quality_review_node`** + `route_after_decomposition_quality`
       wired after reconcile/semantic-review, before the STORIES gate; pass-through
       when `decomposition.quality_review` is off. Writes
       `docs/DECOMPOSITION_REVIEW.md`; emits `decomposition_quality_findings`.
4. [ ] **Config** `decomposition.*` (all registration points + `config_help.md`);
       thread into `AgentState`. Recommend `decomposition_reviewer_mode: thinking`.
5. [ ] **Phase 2 — bounded auto-remediation** — feed findings back for one targeted
       split/merge/rewrite pass, re-reconcile, cap via
       `decomposition.max_remediation_cycles`; deterministic-fixable findings
       applied without the LLM where safe. Gated `quality_auto_remediate` (default
       off).
6. [ ] **Measure & decide** — run advisory on a real build (e.g. lumina), measure
       false-positive rate and the downstream-churn delta, then decide the
       default-on flip and whether Phase 2 is warranted.
