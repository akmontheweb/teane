# ADR-0004: Embed Constraint NFRs as Story Acceptance Criteria

**Status:** Proposed — implementation in progress behind `planning.embed_constraint_nfrs` (default off). Landed: classify+embed (`1d1275c`), deterministic→LLM classifier at refinement (`e6aa635`), shared-policy primitive (this change). Pending: augment-path parity, coverage/traceability extension, lumina A/B before flipping the default.
**Date:** 2026-07-29
**Deciders:** Teane harness maintainers
**Related:** [[ADR-0003]] (hybrid test generation — NFR ACs become Tier-1/4 tests), [[ADR-0002]] (generation-side contradiction prevention), decomposition NFR cross-domain-drop exemption (`df33a89`), drop-detection (`146f5cd`), paired producer+consumer decomposition (`13d0525`)

## Context

The decomposition planner emits non-functional requirements as **separate
execution stories** (`STORY-NFR-NNN`), parented under the feature they relate
to but scheduled and patched as their own work items. In practice these run
*after* the functional stories that built the code they constrain, which makes
every constraint-style NFR an afterthought: the functional story writes
`create_contact()`, and a later NFR story is supposed to retrofit sanitization,
error-shaping, or validation into code that is already "done."

This is not hypothetical. **lumina session 019fa046** shipped with
`STORY-NFR-004` ("input sanitization against XSS / SQL injection") as a
separate story whose `scope_files` were `schemas/contact.py`,
`contacts_service.py`, and `ContactCard.tsx`. The cross-domain-drop heuristic
stripped **all three** — an NFR's vocabulary (`xss`, `injection`, `sanitize`)
shares no domain token with the contact-named modules it must touch — and no
sanitization landed anywhere in the server. `df33a89` patched the *symptom*
(NFR stories are now exempt from the drop), but the underlying mismatch stands:

**Forces at play:**

- **Sequencing.** A separate NFR story runs after its functional story, so the
  constraint is retrofitted, not built in — and under budget pressure it is the
  first thing dropped.
- **Structural mismatch.** Decomposition is one-story-one-domain-module. A
  constraint NFR is *cross-cutting* — "sanitize all input" is a property of
  every write path — so it never maps cleanly to a single scope, and the
  domain-word guards (drop-detection, cross-domain-drop) fight it.
- **The design already half-admits it.** `decomposition.py` instructs that NFR
  stories be "scope[d] to the production module whose behaviour the NFR
  constrains." If an NFR is *defined by* the functional module it constrains, it
  is a property of that module's story, and a separate story just adds a seam to
  re-bridge.
- **But not every NFR is a constraint.** lumina also carried "frontend
  interactive within 2s," "runs fully offline," and "SQLite survives unexpected
  shutdown." These are *capabilities* — architectural work with no single
  functional home. Forcing them into one functional story distorts it as badly
  as leaving sanitization out of one.
- **Consistency.** One NFR ("sanitize input", "all errors RFC-7807") applies to
  many functional stories. Whatever the model, the rule needs a single owner or
  five stories will each re-derive it slightly differently.

The question this ADR answers: **should constraint NFRs be embedded into the
functional story's acceptance criteria — built and verified in the same
patching/repair pass — instead of generated as separate stories that run
later?**

## Decision

Split NFRs by kind at requirements-refinement time and route them differently.

1. **Classify** each NFR as `constraint` (a property of a specific behaviour —
   validation, sanitization, error-shaping, per-view a11y) or `capability` (an
   architectural work item with no single functional home — offline mode, load
   budget, durability, observability, a rate limiter). A small LLM
   classification with a deterministic keyword fallback; ambiguous → `capability`
   (fails safe toward an explicit, tracked story rather than a silently-embedded
   one).

2. **Constraint NFRs become acceptance criteria on the functional story /
   requirement they constrain**, sourced from a **shared NFR policy** so the rule
   has one authored owner. Concretely (as implemented):
   - The **policy is the constraint NFR's single definition** — its
     `#### … NFR-NNN` block in `SPEC_REQUIREMENTS.md`, tagged `**Class:**
     constraint` at refinement. The **policy id is the normalised NFR key**
     (`NFR-002`). `_nfr_policy_registry(spec)` returns these as the queryable
     single source.
   - Embedding **derives every copy from that one definition in a single
     deterministic pass**, so the rule cannot drift across stories by
     construction — the failure mode a naïve "copy the AC into each story"
     invites. Each folded-in AC is **attributed to its owner** with an
     `[NFR:<policy-id>]` tag (`[NFR:NFR-002] submitting <script> is stored
     literally and rendered escaped`), making the single-ownership explicit and
     machine-checkable: all `[NFR:NFR-002]` ACs across the plan trace to the one
     NFR-002 policy. `_nfr_policy_id_of(ac)` recovers the owner.
   - The AC text stays **self-contained** (not a bare "see POLICY-SEC-1"
     reference) so it remains individually testable by the AC-coverage gate and
     test-gen; the tag supplies provenance without hollowing out the AC.
   One definition, applied and verified inside every functional flow, with
   provenance preserved.

3. **Capability NFRs remain enabler stories**, but are **sequenced as
   dependencies** (built before/around the functional stories they govern) and
   their policy text is injected into those stories' build context, so they are
   foundational rather than trailing.

4. **Traceability becomes** `NFR-policy → per-story AC → @verifies test`. The
   embedded AC is a first-class citizen of the existing AC-coverage gate
   (`afff628`) and the `test_verifies_ac` model, so a build cannot go green
   having silently skipped it — the exact failure of 019fa046.

The payoff is in the loop: a constraint AC rides through **decomposition**
(scope comes from the functional domain — no cross-domain-drop to fight),
**patching** (the AC is in the build prompt from round one, so the endpoint is
written compliant), **test-gen** (a `@verifies`-linked test lands beside the
endpoint's other tests), and **compiler → repair** (non-compliance is a failing
test that drives repair, enforced by the same machinery as functional bugs and
now steered by the source-fed reflection judge). The retrofit window disappears.

## Options Considered

### Option A: Status quo — separate NFR execution stories

| Dimension | Assessment |
|-----------|------------|
| Sequencing | Poor — NFR runs after the code it constrains; retrofit by construction |
| Decomposition fit | Poor — cross-cutting NFR vs one-domain scope; fights the drop guards |
| Coverage risk | High — a dropped/deferred NFR story silently under-implements (019fa046) |
| Consistency | Neutral — one story per NFR, but disconnected from the flows it governs |
| Complexity | None (no change) |

**Pros:** No pipeline change; NFRs are explicitly visible as tracked items.
**Cons:** The afterthought problem is structural; the coverage failure is
measured, not theoretical; requires the drop-heuristic exemptions as ongoing
patches.

### Option B: Embed ALL NFRs as per-story ACs (eliminate NFR stories)

| Dimension | Assessment |
|-----------|------------|
| Sequencing | Excellent — everything built in-pass |
| Decomposition fit | Good for constraints; **wrong for capabilities** |
| Coverage risk | Low for constraints; capability NFRs get distorted or lost in a functional story |
| Consistency | Poor unless a shared-policy mechanism is added |
| Complexity | Medium |

**Pros:** Maximally removes the afterthought; simplest mental model.
**Cons:** Capability NFRs (offline mode, durability, load budget) have no single
functional home — cramming them into one story bloats it and hides the work;
throws away the legitimate enabler-story case.

### Option C: Hybrid — constraints embed from a shared policy, capabilities stay enablers (this ADR)

| Dimension | Assessment |
|-----------|------------|
| Sequencing | Excellent for constraints; capabilities sequenced as up-front dependencies |
| Decomposition fit | Good — constraints ride the functional domain scope; capabilities keep an explicit story |
| Coverage risk | Low — embedded ACs enter the AC-coverage gate; capabilities stay tracked |
| Consistency | Good — shared policy = one owner, applied per-story |
| Complexity | Medium-High — classify step + policy mechanism + traceability change |

**Pros:** Removes the afterthought for the common (constraint) case *and* keeps
the genuinely-standalone work explicit; makes the cross-domain-drop exemption
mostly moot because those NFRs stop being separate stories.
**Cons:** Real pipeline change across refinement, decomposition, gates, and
traceability; a classifier that mis-labels a capability as a constraint could
bury architectural work in an AC (mitigated by failing ambiguous → capability).

## Trade-off Analysis

The core trade is **afterthought-elimination vs. classifier risk**. Option B
maximises the first but has no answer for capability NFRs; Option A avoids
classifier risk but keeps the measured coverage failure and needs perpetual
drop-heuristic patches. Option C accepts one new decision point (the
constraint/capability classifier) to get the sequencing win where it matters
while preserving the enabler case where it matters.

The classifier is the only genuinely new risk. It is bounded: the failure mode
is a mislabel, the fail-safe is "ambiguous → capability" (an explicit tracked
story is a recoverable outcome; a silently-embedded-then-dropped capability is
not), and the boundary is the well-worn *constraint-on-a-behaviour* vs
*capability-of-its-own* line, which humans apply reliably and an LLM can with a
short rubric plus keyword priors (`sanitize/validate/escape/error` → constraint;
`offline/latency/durability/observability/throughput/rate-limit` → capability).

Consistency is handled the same way in B and C, and is not optional: without a
shared policy, embedding duplicates the rule across stories and invites drift.
The policy block is the single-owner primitive that makes embedding safe.

## Consequences

**Easier:**
- Constraint NFRs are built compliant in the first patching pass and enforced by
  the repair loop — no retrofit, no separate scheduling.
- The cross-domain-drop / drop-detection exemptions for NFR stories become
  largely unnecessary (those NFRs are no longer stories).
- NFR coverage is enforced by the existing AC-coverage gate and `@verifies`
  traceability, closing the 019fa046 silent-skip.

**Harder:**
- A new requirements-refinement classify step and a shared-policy authoring +
  injection mechanism must be built and maintained.
- Decomposition, the AC-coverage/test gates, and the traceability model all
  currently assume NFR stories exist; the transition must not drop NFR coverage
  while it is in flight (the very failure we are preventing).
- Cross-story NFRs need the policy-reference indirection to stay consistent —
  more moving parts than a copied AC.

**To revisit:**
- Whether capability NFRs should also carry auto-generated verification (a perf
  budget test, a durability test) rather than remaining prose enablers.
- Whether the classifier needs a human-review escape for the ambiguous middle,
  or the fail-safe default suffices in practice.
- Whether the shared-policy block should be spec-authored, harness-synthesised,
  or both.

## Action Items

1. [x] Land this ADR as **Proposed**; socialise the constraint/capability
   boundary and the "eliminate-all vs keep-capability-enablers" fork. (`4fc46eb`)
2. [x] Add the NFR classifier to requirements-refinement (deterministic-first,
   LLM only for the ambiguous band, ambiguous → capability), behind a config flag
   (`planning.embed_constraint_nfrs`, default off). (`e6aa635`)
3. [x] Define the shared NFR-policy structure and the per-story AC attribution
   syntax (`[NFR:<policy-id>]` tag; `_nfr_policy_registry` as the single source;
   drift-safe by single-pass derivation). **Remaining:** inject the
   consolidated policy definitions into the functional stories' build context
   (the registry exists; the prompt wiring is pending).
4. [x] Teach decomposition to emit constraint-NFR-derived ACs on functional
   stories and to stop emitting separate `STORY-NFR-NNN` for the constraint
   class; keep capability NFRs as sequenced enablers. (`1d1275c`) **Remaining:**
   augment/CR-path parity (`_validate_augment_payload`).
5. [ ] Extend the AC-coverage gate and `test_verifies_ac` traceability to the
   embedded `[NFR:<id>]` ACs; confirm no coverage regression vs a run that used
   separate NFR stories.
6. [ ] Prototype behind the flag and **A/B against a lumina rebuild** — verify
   `STORY-NFR-002`-class sanitization now lands in the functional pass and is
   test-enforced.
7. [ ] On green A/B, flip the flag default and update [[ADR-0003]] cross-links
   (embedded NFR ACs are new inputs to the Tier-1/Tier-4 test emitters).
