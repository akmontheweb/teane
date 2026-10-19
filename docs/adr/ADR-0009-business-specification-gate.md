# ADR-0009: Business Specification Gate Before Requirements Synthesis

**Status:** Proposed — implementation not started. Action item 1 is a measurement that decides whether anything after it gets built: this ADR's own premise is unverified, and the mechanism it proposes is the kind that most easily becomes ceremony. Default off when it lands.
**Date:** 2026-10-19
**Deciders:** Teane harness maintainers
**Related:** [[ADR-0008]] (scoped spec context — the direct, measured cost of adding a third upstream document), [[ADR-0007]] (decomposition-quality review — the consumer for the anti-scope and epic-coverage checks), [[ADR-0006]] (in-build acceptance verification — business KPIs become verifiable thresholds), [[ADR-0004]] (NFR embedding). Builds on the existing docgen gate machinery: `write_spec_node` (`graph.py:26529`), `spec_review_node` (`graph.py:27104`), `_HITL_GATES` (`cli.py:2922`).

## Context

Intake today performs two different transformations in a single LLM call.

`product_spec_dir` is a mandatory config key. `_load_consolidated_product_spec`
(`cli.py:6181`) reads every `.txt` / `.md` / `.pdf` in that folder in
alphabetical order, prefixes each with a `## <filename>` header, and returns one
consolidated string. That string is written to a manifest and handed to
`synthesize_requirements` (`cli.py:5139`), which inlines it **verbatim** under
`## Raw Product Notes` in the `requirements_doc.md` skill. One dispatch later,
`docs/SPEC_REQUIREMENTS.md` exists — and under `--agile` it is already a full
SAFe artifact: Epic → Feature → Story → Gherkin acceptance criteria, INVEST
validation, eight quality gates, traceability matrix
(`skills/docgen/requirements_doc.md:77`). `synthesize_architecture`
(`cli.py:5454`) then reads that file alone.

So the same call must both (a) extract and normalise business intent out of
whatever prose, meeting notes or sales deck landed in the folder, and (b)
formalise it into a testable requirements hierarchy. There is no artifact
between them. Nothing records what is deliberately **out** of scope, which
business outcome an Epic serves, what the success metric is, or why a
requirement is Must-Have — the traceability matrix prints a MoSCoW column that
no upstream document justifies.

The agile metamodel the harness already uses is explicitly business-anchored at
its top — its hierarchy block labels the Epic a *"business outcome boundary"*
(`skills/docgen/requirements_doc.md:88`). But nothing produces or validates that
business outcome. The model
invents the outcome and slices it into stories in one breath, and the invention
is indistinguishable from the derivation by the time anyone reads the output.

**A front-end for part of this already exists and is off.**
`requirements_discovery_node` (`graph.py:25938`) runs a 13-sector
cross-examination (`_REQUIREMENTS_SECTORS`, `graph.py:2576`) that includes
`BUSINESS LOGIC & STATE MACHINES`, `COMPLIANCE & DATA CLASSIFICATION` and
`OBSERVABILITY & SUCCESS METRICS`, with follow-up rounds and per-round sector
re-focus. `--spec-discovery` defaults to **false** (`cli.py:12724`), and when it
does run it produces the *same* artifact rather than a distinct business one.

**Forces at play:**

- **Ceremony is the default outcome, and it is measured, not feared.**
  [[ADR-0008]]'s action-item-2 instrumentation (run
  `lumina-run12-20260923-0914`) found `repair` used the anchored spec region in
  **0 of 24** dispatches, `patching` in 2 of 13, **9 of 13 requirement blocks
  were never used**, and 25 of 63 dispatches never carried the region at all —
  against 99,670 chars anchored to every call. A third upstream document that
  is written, reviewed, HITL-approved and anchored, but consumed by nothing,
  makes that strictly worse. **An artifact nobody reads is a regression, not a
  neutral addition.**
- **Double transformation is lossy, and the loss lands exactly where it hurts.**
  Today `requirements_doc.md` sees the operator's own words. If a business pass
  is inserted *in the chain* — notes → business doc → RSD — then the RSD is
  synthesised from a summary and every detail the business pass dropped is
  unrecoverable. Detail loss at requirements altitude is the documented cause of
  vague, untestable acceptance criteria ([[ADR-0005]]).
- **What the [[ADR-0007]] findings do and do not justify.** The lumina
  decomposition review produced 18 real findings, and they are tempting ammunition
  here, but most are *story-craft* failures one layer below this ADR: "Add Contact
  modal UI" is an untestable AC written at RSD time, and no business document
  would have prevented it. Only two findings plausibly trace upstream —
  STORY-001 bundling API + UI + NFR (an epic boundary drawn around no stated
  outcome) and the STORY-004/005 overlap. This ADR must be argued on scope
  control and outcome traceability, not on that finding count.
- **North-star autonomy / minimal HITL.** A fourth human gate on a subjective
  artifact is the most expensive kind. It is justified only because business
  framing is the one place where a human correction is *cheap*: deleting an
  invented outcome is one line, whereas the same invention discovered at
  decomposition time is an Epic, three Features, a dozen stories with ACs, and a
  renumbering pass under `requirements_doc.md` Gate 5.
- **Determinism where possible.** Whether an Epic cites a business objective,
  and whether a stated objective reached any Epic, are *provable*. Whether the
  vision is any good is not. This is the [[ADR-0003]]/[[ADR-0005]]/[[ADR-0007]]
  two-tier split applied one altitude up.

## Decision

Add an optional **BUSINESS gate** before requirements synthesis that emits
`docs/SPEC_BUSINESS.md`, **and make it earn its place by being consumed
deterministically** rather than merely anchored. The artifact is a product
vision plus an epic-level backlog — not a BRD, which is both anti-agile and the
shape that grows without bound.

Phase 0 (action item 1) is a measurement that can kill the rest of this ADR.

### 1. A fourth docgen gate, reusing the existing mechanism

`current_gate` is a plain string (`graph.py:167`) and every gate-shaped
component already branches on it, so this is the fourth instance of a working
pattern rather than new machinery:

- `NodeRole.BUSINESS_ANALYST` in `gateway.py:39` — a distinct role, which buys
  its own cache-drift bucket (the `91d6555` / `29cfd86` lesson: judgment-shaped
  prompts in a shared family flip the recorded prefix hash and force misses on
  the real loop dispatch) and lets an operator route a different model, which is
  the right instinct for a framing task.
- `skills/docgen/business_spec.md` — same `docgen_prompts.load` loader, same
  per-workspace override path (`{workspace}/skills/docgen/business_spec.md`).
- `current_gate = "BUSINESS"` plus a filename branch in `write_spec_node` and
  `spec_review_node`, which inherits the independent-reviewer
  critique-and-revise cycle at no additional design cost.
- `_HITL_GATES += ("business", "hitl_business", "business")` — inherits
  `_resolve_hitl_flags`' three-tier precedence and the auto-approve fallbacks.
- Optionally a `business_discovery_node` reusing the sector-interview pattern
  when `--spec-discovery` is on, with a business-sector list of its own.

**The gate is additive, never a link in the chain.** `synthesize_requirements`
keeps receiving the verbatim manifest and *additionally* receives
`SPEC_BUSINESS.md`. The raw notes are never replaced by their summary.

**Naming.** `SPEC_BUSINESS.md`, not anything containing "product spec" —
`product_spec/` already means the raw input folder, and a "business
specification" sitting beside it will be mistaken for the input.

### 2. The artifact: vision plus epic-level backlog

Bounded on purpose, with a hard section list in the prompt so it cannot sprawl:
product vision; personas/actors; business objectives with measurable success
metrics; business rules and invariants; an explicit in-scope / **out-of-scope**
boundary; and a prioritised epic-level backlog with MoSCoW rationale. No
features, no stories, no acceptance criteria — those belong to the RSD, and
duplicating them here creates two sources of truth for the same statement.

### 3. The consumption contract

This is the half that decides whether the ADR is worth implementing, and it is
deliberately ordered by value:

1. **An ID family `BO-NNN`** (business objective) in `req_ids.py`, joining the
   `FR`/`NFR`/`US` and `EPIC`/`FEAT`/`STORY`/`STORY-NFR` families. Implement the
   two-format parser from day one — heading form *and* declaration form
   (bold/bullet/table). Heading-only parsing is the exact defect that produced an
   empty requirement registry → every story rejected → unsatisfiable auto-repair
   → opaque `Expecting value: line 1 col 1` → HITL hang.
2. **A deterministic bidirectional coverage gate.** Every `EPIC` cites ≥1 `BO`;
   every `BO` reaches ≥1 `EPIC`. Same class as
   `decomposition_review.deterministic_findings` — provable, no LLM, no
   false-positive risk. The reverse direction is the one nothing in the harness
   can currently detect: an orphan `BO` is silently dropped scope.
3. **An anti-scope check** in the [[ADR-0007]] reviewer, fed the machine-readable
   out-of-scope list: flag any story implementing something explicitly excluded.
   Nothing today can detect *invented* scope in either direction.
4. **KPI thresholds inherit downward.** Measurable metrics stated at `BO` level
   become the source for NFR enabler-story thresholds, closing [[ADR-0008]]
   action item 7 (capability-NFR ACs carrying no measurable threshold) at its
   source rather than adding another reviewer check to catch it afterwards.
5. **Traceability extends** (`traceability.py:168`) to BO → EPIC → FEAT → STORY
   → test, so the matrix can answer "which business outcome is unverified?"

### 4. Context budget: generated, not anchored

`SPEC_BUSINESS.md` is the most cross-cutting and least per-call-relevant
document in the set, which makes it the worst possible anchor candidate. Under
[[ADR-0008]]'s two-tier model it is Tier-1 material **at most**, and the
starting rule is stronger than that: available to planning, decomposition and
acceptance-scenario generation; **excluded from `patching` and `repair`
entirely**. Note that whole-spec readers would otherwise pick it up by default
(`decomposition.py:1182` concatenates the spec documents wholesale).

Prove it with the instrument that already exists — `harness/spec_usage.py` plus
`scripts/spec_usage_report.py`, re-enabling `debug.measure_spec_usage` — before
any default is flipped. That instrument has already earned its keep once by
refuting [[ADR-0008]]'s own drafted slice scope within a day.

### Config (new `business` section; all config-driven per no-hardcoded-caps)

- `business.enabled` (default `false`) — run the gate at all.
- `business.coverage_enforce` (default `false`) — BO↔EPIC coverage findings
  block rather than advise.
- `business.anti_scope_check` (default `false`) — feed the out-of-scope list to
  the [[ADR-0007]] reviewer.
- `business.max_review_cycles` — reuses the `spec_review_node` cycle cap shape.
- `hitl.business` (default `true`, per `_HITL_GATES` precedent).
- `model_routing.business_analyst_*` — empty means fall back to the planning
  model, mirroring `DECOMPOSITION`.

All four registration points plus `config_help.md`, per the standing rule.

### Node contract & placement

Runs once, before requirements synthesis, pass-through when
`business.enabled` is false. Fail-open: a failed or empty business pass logs and
proceeds to requirements synthesis with the raw notes exactly as today — the
gate can add signal, never manufacture a stall. The coverage gate runs after
requirements synthesis (it needs both documents) and reports alongside the
existing spec-review findings.

## Options Considered

### Option A: Status quo — one call from raw notes to the full RSD

**Pros:** No new artifact, no new gate, no new prompt budget; the requirements
model sees the operator's unmediated words.
**Cons:** Business framing is never inspectable before it is fossilised into
dozens of stories; invented scope is cheapest to kill at the moment it cannot be
seen; out-of-scope, success metrics and MoSCoW rationale have nowhere to live;
Epic boundaries are asserted against no stated outcome.

### Option B: Turn on `--spec-discovery`, add no artifact

**Pros:** Zero new machinery. The 13 existing sectors already interrogate
success metrics, business logic and compliance; this may close most of the gap
today.
**Cons:** Produces the same artifact, so there is still no separable business
layer, no BO IDs, no deterministic coverage gate and no out-of-scope list to
check against. It is an interview, not a contract — but it is the correct thing
to *try first*, which is why it is action item 1.

### Option C: BUSINESS gate with a deterministic consumption contract (this ADR)

**Pros:** Reuses four existing mechanisms (gate string, write/review nodes, HITL
gate, docgen loader); puts the provable load on a zero-false-positive
deterministic gate; gives invented and dropped scope their first detector in
either direction; grounds MoSCoW and NFR thresholds in a stated outcome.
**Cons:** A fourth document and a fourth human gate; a new ID family with the
registry fragility that implies; real prompt-budget pressure that must be
measured rather than assumed; the vision section itself stays unverifiable.

### Option D: A business section inside `SPEC_REQUIREMENTS.md`, not a separate document

**Pros:** No new gate, no new file, no new anchor; one source of truth; the
section is visible to everything that already reads the spec.
**Cons:** Defeats the stated purpose — the business framing is produced by the
same call, in the same breath, as the stories it is supposed to constrain, so it
cannot be inspected or corrected first. `gap_fill` and `spec_reconciler` both
rewrite that file, so a business section is exposed to rewriting by mechanisms
that have no business mandate.

### Option E: Business doc replaces the raw notes as the requirements input

**Pros:** Clean pipeline; the smallest possible prompt at the requirements step.
**Cons:** Rejected outright. The RSD would be synthesised from a summary, every
detail dropped by the business pass would be permanently unrecoverable, and
detail loss at requirements altitude is the documented cause of the untestable
ACs that [[ADR-0005]] and [[ADR-0007]] exist to fight.

## Trade-off Analysis

The core trade is **earlier scope control vs. a third document nobody reads**,
and [[ADR-0008]] has already measured which way this class of change usually
falls: 9 of 13 requirement blocks unused, `repair` reading nothing across 24
dispatches. That evidence is the strongest argument *against* this ADR as
naively built, and it is why the decision is weighted toward the consumption
contract rather than the artifact. Phrased plainly: **the document is the cheap
half and the deterministic BO↔EPIC gate is the valuable half**, and an
implementation that ships the first without the second should be read as having
failed, not as having completed Phase 1.

The honest case for proceeding rests on one asymmetry rather than on any finding
count. Business-level defects are rare but structurally expensive: an invented
outcome is one line to delete at business altitude and a renumbering pass at
decomposition altitude, while a dropped outcome is, today, undetectable at any
altitude. Option B may capture much of the first effect with no new machinery,
which is precisely why it is measured before anything is built.

The out-of-scope list is the piece with no existing substitute anywhere in the
chain. `spec_reconciler` owns structural integrity, `semantic_review` owns
coverage of what was *stated*, and [[ADR-0007]] owns craftsmanship — but no
mechanism asks whether the system is building something nobody asked for.
Coverage review cannot find it by construction: unrequested scope is perfectly
self-consistent.

The HITL cost is real and partly offsetting. A fourth gate on a subjective
artifact cuts against the autonomy north-star, and it is defensible only because
`spec_review_node` already provides an independent critique-and-revise cycle
that can absorb most correction without a human, and because the human's
alternative is correcting the same mistake after it has been multiplied by
decomposition.

## Consequences

**Easier:**
- Invented scope and dropped scope both become detectable, the latter for the
  first time.
- MoSCoW priority, Epic boundaries and NFR thresholds acquire a stated source
  instead of being asserted — which closes [[ADR-0008]] item 7 upstream.
- Business framing becomes correctable at the altitude where a correction is one
  line.
- Traceability can answer an outcome-level question ("is this business objective
  verified?") that it structurally cannot answer today.

**Harder:**
- A third upstream document under a context budget that [[ADR-0008]] has already
  shown to be badly spent; it must be excluded from `patching`/`repair` and
  proven with `spec_usage`, not assumed.
- A new ID family and registry parser — the single most defect-prone surface
  touched here, with a known failure mode (empty registry → HITL hang) to design
  against from the first commit.
- A fourth human gate, against the minimal-HITL north-star.
- The vision/persona prose is judgment-only; the deterministic tier cannot carry
  it, so part of the artifact is permanently unverifiable.

**To revisit:**
- Whether `business.enabled` should default on, after the Phase-4 measurement —
  the flip discipline [[ADR-0004]], [[ADR-0005]] and [[ADR-0007]] all followed.
- Whether a business pass should also run in change-request mode, where scope
  creep is arguably more likely than in greenfield.
- Whether `--spec-discovery` should default true independently of this ADR, if
  action item 1 shows the interview alone closes most of the gap.
- Whether the out-of-scope list belongs in `state.db` rather than only in
  Markdown, once a second consumer appears.

## Action Items

1. [ ] **Phase 0 — measure the premise; this gates everything below.** Take
       lumina's real `product_spec/` plus one or two other inputs. Run the
       existing pipeline, then run it again with `--spec-discovery true`, and
       hand-audit both `SPEC_REQUIREMENTS.md` outputs for *business-level*
       defects specifically: invented scope, dropped/unaddressed outcome,
       unjustified MoSCoW, absent success metric, Epic boundary around no stated
       outcome. **If those classes are rare, or if discovery alone closes them,
       stop and close this ADR as Rejected** — record the finding either way, as
       [[ADR-0008]] item 2 did.
2. [ ] **Phase 1 — the gate, advisory, no consumer.** `NodeRole.BUSINESS_ANALYST`,
       `skills/docgen/business_spec.md` with a hard section list, `BUSINESS` gate
       branches in `write_spec_node` / `spec_review_node`, `_HITL_GATES` entry,
       config block + all four registration points + `config_help.md`. Writes
       `docs/SPEC_BUSINESS.md`. Flag off. Additive: `synthesize_requirements`
       receives the business doc *alongside* the verbatim manifest.
3. [ ] **Phase 2 — the value half.** `BO-NNN` family in `req_ids.py` with
       two-format parsing and the empty-registry guards from the outset;
       deterministic bidirectional BO↔EPIC coverage gate; findings advisory,
       `business.coverage_enforce` off.
4. [ ] **Phase 3 — the remaining consumers.** Anti-scope check wired into the
       [[ADR-0007]] reviewer; BO KPI thresholds inherited by NFR enabler stories;
       `traceability.py` extended to BO → EPIC → FEAT → STORY → test.
5. [ ] **Phase 4 — measure, then decide the defaults.** Re-enable
       `debug.measure_spec_usage` and run `scripts/spec_usage_report.py` on a
       build with the gate on: confirm which roles actually read
       `SPEC_BUSINESS.md`, exclude the rest, and only then consider the
       default-on flip. Report the BO↔EPIC finding rate and its false-positive
       rate alongside it.
6. [ ] **Decide the discovery question separately.** Whatever action item 1 shows
       about `--spec-discovery`'s default is a standalone change and should not
       wait on this ADR.
