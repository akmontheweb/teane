# ADR-0008: Per-Story Spec Context Instead of the Whole-Spec Anchor

**Status:** Proposed — nothing implemented. Drafted 2026-09-23 after the lumina run 9→11 sequence.
**Date:** 2026-09-23
**Deciders:** Teane harness maintainers
**Related:** [[ADR-0004]] (NFR embedding — the per-story NFR policy block is the precedent mechanism this ADR generalises), [[ADR-0003]] (hybrid test generation — consumes ACs, not spec prose), [[ADR-0006]] (in-build acceptance verification — the acceptance generator is already excerpt-based)

## Context

Every node in the graph currently sees the **entire requirements and
architecture specification** on every call. It is not part of
`_build_system_prompt` (`graph.py:1716`), which contains no spec body at all —
it arrives as `spec_override`, prepended to `messages[0]` by
`create_initial_state` (`graph.py:503–535`) and assembled in
`cli.py:7905–7931`:

```
spec_override = _slim_spec_for_prompt(SPEC_REQUIREMENTS.md) + SPEC_ARCHITECTURE.md
```

`_slim_spec_for_prompt` (`cli.py:5260`) is **not** a truncation. It strips
planner-only `**Priority:** / **Estimate:**`-style fields and collapses blank
lines — roughly 30% — and applies **no character cap whatsoever**. A 42 KB
requirements document goes in whole, followed by a 33 KB architecture document.

Measured on `lumina-run11-20260923-0040`: `system_prompt_built` reports
**99,670 chars** — about 25k tokens anchored to every call the run makes.

**Forces at play:**

- **The cost argument is the weak one.** The anchor is deliberately stable so it
  prefix-caches (`graph.py:1721–1723`: *"This prompt is never mutated or
  truncated — it maximizes prefix caching"*). Cached tokens are cheap. This ADR
  is not primarily about spend.
- **The accuracy argument is the strong one.** A model asked to fix
  `_effective_month_day` is handed 13 requirements, 4 NFR enabler stories, a
  traceability matrix and a deployment section. Run 11 is the worked example of
  the failure this invites: the acceptance generator invented
  `body[0]['first_name']` from criterion *prose* while the one authoritative
  fact — `response_model=BirthdayPage` — was not in front of it (fixed in
  `ba2f949`). More document did not mean more accuracy; it meant more surface
  to confabulate from.
- **The prompts already apologise for it.** `test_regeneration`'s spec section
  is headed *"Specification (TIEBREAKER ONLY — do not cite in the test)"*. Its
  content is `messages[0][:4000]` (`test_regeneration.py:616–620`) — a blind
  head-cut of the anchored prompt, which lands on whatever section the spec
  happens to open with, related to the failing test only by accident.
- **The largest consumers do not need the anchor.** `decomposition_node`
  (`decomposition.py:2277, 2294`), `patch_reconcile_node` (`graph.py:25139`)
  and `story_reopen_node` (`graph.py:25011`) each read both documents **from
  disk themselves**. They are planning or reconciling the whole plan, so
  whole-spec is correct for them — and they already have it without the
  anchored copy. The nodes that inherit it by accident are the loop nodes
  (repair, patching, test-gen, regeneration) where confabulation is expensive.
- **The slicing already exists and is already persisted.** `_ingest_requirements`
  (`decomposition.py:1052`) writes one `requirements` row per requirement
  (`story_state.py:317`) carrying `req_key`, the **verbatim `body`**,
  `source_path` and `source_line`. `source_line` has two writers
  (`decomposition.py:1085`, `story_state.py:1026`) and **no reader** in any
  prompt builder. The document is already broken into addressable pieces;
  nothing consumes them at prompt-build time.
- **There is a working precedent.** `_build_nfr_policy_block`
  (`graph.py:24546`), via `_build_story_preamble` (`graph.py:24596`), injects
  **only the NFR policies a story's ACs actually cite**. That is per-requirement
  selective injection, in production, for one requirement class.

`docs/SPEC_REQUIREMENTS.md` divides cleanly along this seam. Of ~41 KB:

| Part | Size | Relevance |
|---|---|---|
| 8 leading `##` sections (product decisions, assumptions, data model, time/date semantics, API conventions, frontend routing, deployment, test conventions) | ~7.5 KB | cross-cutting; every node |
| EPIC / FEAT / STORY blocks | ~24 KB | one story at a time |
| `## Enabler Stories — Non-Functional Requirements` | ~6.6 KB | cited stories only |

The question this ADR answers: **should the loop nodes receive a per-story
slice of the specification assembled from the requirements DB, instead of the
whole document anchored in `messages[0]`?**

A second question arrived with the same mechanism and is answered here because
it shares it: **should capability NFR stories be abolished in favour of an NFR
section on each core story?**

## Decision

### 1. Two-tier spec context

**Tier 1 — anchored, stable, cached.** `spec_override` carries only the
cross-cutting preamble: the leading `##` sections of `SPEC_REQUIREMENTS.md` and
the architecture summary. These are genuinely needed everywhere, change rarely,
and are the ideal cache prefix. Sectioning uses the existing heading walk in
`req_ids.parse_spec_requirements` (`req_ids.py:342`) — no new parser.

**Tier 2 — per-story, injected, not anchored.** `_build_story_preamble`
assembles the working slice from the DB:

- the story's own `requirements.body`;
- its parent feature and epic bodies, reached through the existing
  `**Parent feature:** FEAT-001` markers (`story_state.py:1361`);
- its acceptance criteria rows;
- the NFR policies its ACs cite — already implemented.

The slice is injected as a **later message**. The cached prefix is never
mutated, which is the invariant `graph.py:1721` depends on.

**Whole-spec readers are untouched.** Decomposition, patch-reconcile and
story-reopen keep reading both documents from disk. This ADR narrows what the
*loop* sees, not what the *planner* sees.

### 2. `test_regeneration`'s tiebreaker becomes the story slice

`spec_tiebreaker = messages[0][:4000]` is replaced by the Tier-2 slice for the
story that owns the failing test. A targeted slice is what the section's own
heading claims to be.

### 3. No physical file split

The requirements document stays one file. It is **appended to** by
`gap_fill.py:384`, rebuilt from by `spec_reconciler.py:820`, and read whole by
decomposition, traceability and the reviewers. Twenty files plus an index would
add a consistency problem to every one of those paths to obtain slicing that
the `requirements` table already provides. Per-story spec cards on disk, if
wanted for human reading, are a renderer over the same rows and a separate
decision.

### 4. NFR delivery: [[ADR-0004]]'s split stands

Capability NFR stories are **not** abolished. Nothing in the run 9→11 evidence
contradicts ADR-0004, and its failure case is still live: in session 019fa046 a
separate sanitization story had all three scope files stripped and no
sanitization landed anywhere. Abolishing capability stories reintroduces that
shape for rate limiting, restart durability and latency budgets — work with no
single functional home. Five stories each told to honour NFR-004 produce five
partial rate limiters or none; ADR-0004 names this force directly.

Constraint NFRs are already embedded as ACs on the functional stories, which is
the substance of the "NFR section on core stories" proposal, in production since
`1d1275c`.

Two real defects visible in the current lumina plan are in scope instead:

- **Capability NFR ACs are not testable.** `STORY-NFR-001` carries exactly one
  criterion: *"Dashboard and directory latency budget."* No endpoint, no
  percentile, no number. Compare `STORY-001`'s 11 concrete ACs. An unverifiable
  AC is why these stories read as dead weight. The decomposition-quality gate
  ([[ADR-0007]]) is the natural home for the check.
- **Constraint-NFR ACs are replicated per story.** The four `[NFR:NFR-002]`
  criteria appear on all five functional stories — 20 rows. This inflates no
  single prompt (a story's prompt carries only its own ACs) but it does inflate
  **verification**: every copy becomes its own acceptance scenario. The
  `[NFR:<policy-id>]` tag already identifies the owner, so verifying a policy
  once per run rather than once per story is machine-checkable today. The AC
  rows stay self-contained — ADR-0004 requires that for the coverage gate — so
  this is a verification-side change only.

## Options Considered

### Option A: Status quo — whole spec anchored in `messages[0]`

**Pros:** Maximal prefix caching; zero risk of withholding something a node
needed; no work.
**Cons:** Every loop-node call carries ~41 KB of mostly-irrelevant requirements;
the confabulation surface is the whole product spec; the tiebreaker section is
a blind head-cut; `requirements.body` and `source_line` remain written-only.

### Option B: Physically split the spec into per-requirement files

**Pros:** Slices addressable by path; no parser needed at read time.
**Cons:** `gap_fill` appends to the spec, `spec_reconciler` rebuilds from it,
traceability and the reviewers read it whole — each becomes a multi-file
consistency problem; the operator loses one readable document; and it duplicates
slicing the DB already has.

### Option C: Two-tier anchored preamble + per-story DB slice (this ADR)

**Pros:** Reuses `requirements.body` (already populated), the heading walk
(already written and tested) and `_build_story_preamble` (already the injection
point for exactly this, for NFRs); the cached prefix stays stable; each node
sees the work in hand; the tiebreaker becomes meaningful.
**Cons:** A node that silently relied on distant spec text loses it; slice
assembly is new code on the hot path; a story needing a sibling's contract
depends on the parent-feature body carrying it.

### Option D: Retrieval over the spec (`repo_index`)

`repo_index.py` is a complete retrieval subsystem — chunker, TF-IDF and
embedding backends, SQLite store, `query_top_chunks`,
`render_results_for_injection` with a 4 KB injection cap.

**Pros:** No hand-authored slicing rule; handles cross-cutting references
naturally.
**Cons:** Off by default; indexes **code**, never `docs/SPEC_*.md`; introduces
retrieval-quality failure (a missed chunk is a silently missing requirement)
where the DB gives an exact, authored mapping. A spec requirement has a
**primary key**; retrieval is the wrong instrument for a keyed lookup.

### Option E: Cap `_slim_spec_for_prompt` at N chars

**Pros:** One-line change; bounded prompt immediately.
**Cons:** A head-cut keeps whatever the document opens with and drops whatever
it ends with, unrelated to the work. This is the tiebreaker bug generalised to
every node.

## Trade-off Analysis

The core trade is **relevance vs. the risk of withholding**. Option A can never
withhold and can never focus; Option C focuses and must prove it withholds
nothing that mattered. The mitigation is that the loop nodes' real inputs are
already structured — ACs, diagnostics, source code — and the spec functions as
background. ADR-0004's policy block is the existence proof: one requirement
class has been delivered by selective injection since `2ed24df`, and no node
has been shown to need the rest.

Caching deserves care rather than caution. The anchor exists to be cached, and
run-10 logs already show `cache_prefix_drift` events, so slices must be appended
as later messages and never spliced into the prefix. Done that way the prefix
gets *more* stable, not less: today it carries a spec that changes whenever
`gap_fill` appends to it.

Option D is the tempting one and the wrong one. Retrieval earns its place when
the mapping from work to context is unknown. Here it is known, authored and
stored: this story satisfies these requirements, and the join table
(`story_satisfies_req`, `story_state.py:344`) already records it.

The NFR half is not a new decision. It reaffirms [[ADR-0004]] against a
proposal to go further, and converts the dissatisfaction behind that proposal
into two checkable defects — an untestable AC and per-story duplicate
verification — rather than deleting the enabler stories that make
architectural work visible.

## Consequences

**Easier:**
- A loop-node prompt shows the story being worked, its parents, its ACs and its
  NFR policies — traceable to rows, not to a 41 KB document.
- `requirements.body` / `source_line` acquire their first reader, and
  `test_defects._write_source_spec`'s stub (`test_defects.py:311–330`, still
  emitting *"Phase 4 will inline the spec excerpt here"*) gets the lookup it
  was waiting for.
- The spec can grow with the product without growing every prompt.

**Harder:**
- Slice assembly must handle a story with no requirement row, an unparented
  story, and a spec edited mid-run — each is a silent under-injection if wrong.
- Two context paths exist (whole-spec planners, sliced loop) and must not drift.
- Debugging shifts: "what did the model see?" stops being one anchored blob and
  becomes a per-story assembly, so the slice should be logged.

**To revisit:**
- Whether the architecture document deserves the same treatment; `arch_summary`
  (`arch_summary.py:111`) already renders a compact preamble from its §11 block
  and may make the whole document unnecessary for loop nodes.
- Whether the Tier-1 preamble should be spec-authored (an explicit
  "cross-cutting" section) rather than inferred from heading position.
- Whether capability NFR stories, once given testable ACs, should carry
  auto-generated verification — the open question ADR-0004 already flagged.

## Action Items

1. [ ] Land this ADR as **Proposed** and settle the Tier-1 boundary: inferred
   from heading position, or spec-authored.
2. [ ] Instrument before changing anything: log, per call, which nodes read
   `messages[0]`'s spec region, over one lumina run. Option C's premise is that
   the loop nodes do not need it; that premise should be measured, not
   grepped.
3. [ ] Implement the Tier-2 slice in `_build_story_preamble` behind a config
   flag (default off), reusing `req_ids.parse_spec_requirements` and the
   `story_satisfies_req` join.
4. [ ] Switch `test_regeneration`'s `spec_tiebreaker` to the Tier-2 slice.
5. [ ] Trim `spec_override` to the Tier-1 preamble under the same flag; record
   `system_prompt_built` chars and per-call input tokens before and after.
6. [ ] A/B on a lumina rebuild: compare acceptance-scenario accuracy and
   repair-round count against the run-11 baseline.
7. [ ] Independently of the above: add a decomposition-quality check
   ([[ADR-0007]]) for capability-NFR ACs that carry no measurable threshold.
8. [ ] Verify each `[NFR:<policy-id>]` policy once per run rather than once per
   citing story, in acceptance verification only; AC rows stay self-contained.
