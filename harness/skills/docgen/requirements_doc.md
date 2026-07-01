# SKILL: Requirements Specification

{AGILE_MODE_DIRECTIVE}

## Role in the pipeline

This skill governs the **requirements** step of the teane build pipeline. The
harness produces a complete application across several phases; this skill fires
once after intake and before architecture / code generation:

```
Step 1 — Intake          (CLI args, --agile flag, product notes from --prompt or spec dir)
Step 2 — Requirements    ← THIS SKILL  (writes docs/SPEC_REQUIREMENTS.md)
Step 3 — Architecture    (synthesize_architecture reads SPEC_REQUIREMENTS.md)
Step 4 — Code generation (scaffold + patch source files against the spec)
Step 5 — Validation      (build, tests, optional spec_review reviewer pass)
Step 6 — Deployment      (optional, via the deploy flow)
```

The `--agile` flag changes the FORMAT of the document this skill emits; it does
not skip or short-circuit any other phase.

---

## Inputs available to this skill

The harness passes the following as part of the LLM dispatch — do not invent
fields the harness has not provided:

| Field                | Source                                  | How it arrives                                                                                   |
|----------------------|-----------------------------------------|--------------------------------------------------------------------------------------------------|
| Product notes        | `--prompt` arg / spec-dir manifest      | Verbatim in the user message under `## Raw Product Notes`.                                       |
| Agile mode           | `--agile` (resolved tri-state)          | Substituted at the top of this prompt as `{AGILE_MODE_DIRECTIVE}` (`AGILE` or `DEFAULT`).        |
| Prior spec context   | `docs/SPEC_REQUIREMENTS.md` (when present in change-request mode) | Prepended to the user message by `_build_change_request_preamble`. |
| Change-request files | `change_requests/CR-N.txt`              | Listed and inlined by the change-request preamble when `change_request_mode` is active.          |

There is **no** `--scope`, `--patch-rsd`, or `--rsd-format` flag in this
harness. Scope is implicit from the product notes; the output format is always
Markdown; "patching" an existing spec is driven by change-request files, not a
flag (see §Change-request mode below).

---

## Output produced for downstream phases

This skill writes a single Markdown artifact:

```
<workspace>/docs/SPEC_REQUIREMENTS.md
```

The artifact must contain, at minimum:

- A requirements block (format depends on Path A or Path B below).
- At least one NFR each covering **performance**, **security**, and
  **availability** (see Gate 6).
- A traceability matrix linking every requirement to a test hook stub ID
  (Phase 5 populates the actual test files; the IDs here are placeholders for
  the test-generation step to consume).

Downstream consumers — `synthesize_architecture`, the code-generation graph
nodes, and `review_and_revise_spec` — treat this file as the source of truth
for what the system must do.

---

## Path A — Agile RSD (`--agile` active)

Follow this path only when the EXECUTION MODE banner above says **AGILE**.
Format follows SAFe hierarchy aligned to Scrum execution, with Gherkin
acceptance criteria and INVEST validation. This is also the format
`harness/decomposition.py` expects when the agile story planner reads the spec
to slice it into stories.

### Hierarchy

```
Epic  (business outcome boundary)
└── Feature  (1 Epic → 2–10 Features; shippable capability)
    └── User Story  (1 Feature → 2–8 Stories; sprint-sized slice)
        └── Acceptance Criteria  (Gherkin scenarios)
```

### Epic template

```markdown
## Epic: <EPIC-NNN> — <Short title>

**Vision statement:** <One sentence: what business outcome does this enable?>
**Business driver:** <Pain point or opportunity — not a feature list>
**Scope:** <Boundary statement — what is inside this epic>
**Out of scope:** <Explicit exclusions — must be present, even if "None">
**Success metrics:** <2–4 quantified KPIs, e.g. "reduce p95 checkout latency by 30%">
**Priority:** [Must Have | Should Have | Could Have | Won't Have]
**Estimated size:** [XS | S | M | L | XL]
**Dependencies:** <Other epics or external systems, or "None">
```

### Feature template

```markdown
### Feature: <FEAT-NNN> — <Short title>
**Parent epic:** EPIC-NNN
**Description:** <2–3 sentences. What capability and to whom?>
**Benefit hypothesis:** As a result of <feature>, <persona> will be able to
  <outcome>, which will achieve <measurable benefit>.
**Feature-level AC:** <1–3 demo-ready conditions — NOT story Gherkin>
**Priority:** [Must Have | Should Have | Could Have | Won't Have]
**Estimate:** <Fibonacci: 1 2 3 5 8 13 21 — cap at 13; split if larger>
**Owner:** <Role, not name>
```

### User Story template — mandatory format, no deviations permitted

````markdown
#### Story: <STORY-NNN> — <Short imperative title>
**Parent feature:** FEAT-NNN

**As a** <specific named role — NOT "user">
**I want** <one concrete action or capability>
**So that** <business or user value delivered>

**Acceptance Criteria:**

```gherkin
Scenario: <Happy-path title>
  Given <system precondition>
  When  <actor performs specific action>
  Then  <observable outcome>
  And   <additional assertion if needed>

Scenario: <Negative or edge-case title>
  Given <precondition>
  When  <action that triggers the edge case>
  Then  <expected safe outcome>
```

**INVEST check** — verify all six before writing output:
- [ ] Independent: buildable and testable without another unfinished story
- [ ] Negotiable: scope adjustable without losing core value
- [ ] Valuable: delivers standalone value to user or business
- [ ] Estimable: sizeable within ±1 Fibonacci point
- [ ] Small: completable within one sprint (≤5 points recommended)
- [ ] Testable: every AC has a binary pass/fail outcome

**Estimate:** <Fibonacci points>
**Priority:** [Must Have | Should Have | Could Have | Won't Have]
**Test hook ID:** TEST-<NNN>  ← stub; Phase 5 populates the test file
**Definition of Ready:**
- [ ] "As a / I want / So that" format verified
- [ ] ≥1 happy-path and ≥1 negative Gherkin scenario present
- [ ] All external dependencies identified
- [ ] Estimate agreed
- [ ] No unresolved blocking questions
````

### Definition of Done — appended once per Feature

```markdown
**Definition of Done — FEAT-NNN:**
- [ ] All child stories accepted
- [ ] Unit test coverage ≥ 80% for new code paths
- [ ] Integration tests pass in CI
- [ ] No open P1 or P2 defects
- [ ] Security review completed (if PII or auth scope)
- [ ] API docs / README updated
- [ ] Accessibility: WCAG 2.1 AA verified (if UI scope)
- [ ] Performance: meets NFR thresholds in STORY-NFR-NNN
- [ ] Feature demo recorded
```

### NFRs in agile mode — modelled as Enabler Stories

````markdown
#### Enabler Story: <STORY-NFR-NNN> — <Short title>
**Type:** [Architecture | Infrastructure | Technical Debt | Research]
**Description:** <What technical capability or constraint this addresses>
**Acceptance Criteria:**
```gherkin
Scenario: NFR threshold met
  Given the system is under <load condition>
  When  <trigger>
  Then  <measurable outcome — latency, uptime, error rate, etc.>
```
**Linked features:** <FEAT-NNN list>
**Test hook ID:** TEST-NFR-<NNN>
````

### Traceability matrix — agile mode

```markdown
## Traceability matrix

| Story ID      | Feature ID | Epic ID  | Priority    | Est | Test hook    | Status |
|---------------|------------|----------|-------------|-----|--------------|--------|
| STORY-001     | FEAT-001   | EPIC-001 | Must Have   | 3   | TEST-001     | Draft  |
| STORY-NFR-001 | FEAT-001   | EPIC-001 | Must Have   | 2   | TEST-NFR-001 | Draft  |
```

---

## Path B — Default RSD (`--agile` inactive)

Follow this path only when the EXECUTION MODE banner above says **DEFAULT**.
Format follows ISO/IEC/IEEE 29148:2018. Generate only sections relevant to the
product notes; omit empty sections silently rather than emitting a stub.

### Document structure

```markdown
# Software Requirements Specification

## 1. Introduction
### 1.1 Purpose
### 1.2 Scope
### 1.3 Definitions, acronyms, abbreviations
### 1.4 References
### 1.5 Document overview

## 2. Overall description
### 2.1 Product perspective
### 2.2 Product functions  (summary only — detail in §3)
### 2.3 User classes and characteristics
### 2.4 Operating environment
### 2.5 Design and implementation constraints
### 2.6 Assumptions and dependencies

## 3. System features (Functional Requirements)
### 3.N <Feature name>
#### 3.N.1 Description and priority
#### 3.N.2 Stimulus / response sequences
#### 3.N.3 Functional requirements
  FR-<ID>: <Requirement using "shall">

## 4. External interface requirements
### 4.1 User interfaces
### 4.2 Hardware interfaces
### 4.3 Software interfaces
### 4.4 Communication interfaces

## 5. Non-functional requirements
  NFR-PERF-NNN:  <Performance — must include measurable threshold>
  NFR-SEC-NNN:   <Security>
  NFR-AVAIL-NNN: <Availability>
  NFR-SCALE-NNN: <Scalability>
  NFR-MAINT-NNN: <Maintainability>
  NFR-COMP-NNN:  <Compliance / regulatory>

## 6. Use cases
  UC-NNN: Actor, preconditions, main flow, alternate flows, postconditions

## 7. Constraints
  (Legal, regulatory, hardware, platform, third-party API limits)

## 8. Requirements traceability matrix (RTM)

| Req ID       | Description summary   | Source      | Priority | Test hook | Status |
|--------------|-----------------------|-------------|----------|-----------|--------|
| FR-001       | ...                   | Stakeholder | High     | TEST-001  | Draft  |
| NFR-PERF-001 | ...                   | Architecture| High     | TEST-P-001| Draft  |
```

### Requirement statement rules — default path

- Use **"shall"** for mandatory requirements, **"should"** for recommendations.
- Each requirement is atomic: one condition, one outcome, one ID.
- No ambiguous qualifiers: never use "fast", "user-friendly", "flexible",
  "robust", "appropriate", "as needed". Replace with quantified thresholds and
  measurement methods.
- Every NFR must specify: threshold value, measurement method, target
  environment.
  - WRONG: "The system shall be fast."
  - RIGHT: "NFR-PERF-001: The /search endpoint shall return HTTP 200 with
    results in < 200 ms at P95 under 500 concurrent users measured in the
    staging environment."

---

## Quality gates — both paths

Resolve each gate before emitting output. A gate that cannot be satisfied from
the product notes must surface as a NOTE in the document, not a silent pass.

### Gate 1 — Completeness
- Agile: every Story has ≥1 happy-path Gherkin scenario AND ≥1 negative /
  edge-case scenario.
- Default: every FR has a corresponding entry in the RTM and a test hook stub.

### Gate 2 — Measurability
No AC, FR, or NFR may use vague qualifiers. Every condition must be binary
pass/fail.

### Gate 3 — INVEST (agile path only)
All six INVEST criteria must pass for every Story.
- Fails "Small" → split the story before emitting. Each resulting piece MUST
  get its own fresh `STORY-NNN` from the next-available integer in the
  global sequence — do NOT extrude a suffix (``STORY-011A``, ``STORY-011B``)
  or decimal (``STORY-011.1``) off the original ID. See "ID numbering
  convention" below.
- Fails "Testable" → rewrite AC until each scenario has a binary outcome.
- Fails "Independent" → identify the blocking dependency and add it as a
  separate story with a `blocks:` annotation.

### Gate 4 — Persona specificity (agile path only)
"As a user" is not acceptable. Use a role that maps to a real actor in the
system (e.g. "As a billing administrator", "As an unauthenticated visitor",
"As a background job processor").

### Gate 5 — Duplicate ID check
Scan all IDs for collisions before emitting. On collision: renumber the later
occurrence.

### Gate 6 — NFR coverage (mandatory)
The document must include at least one requirement covering each of
**performance**, **security**, and **availability**, regardless of path. If
the product notes don't specify thresholds, supply conservative industry
defaults and explicitly mark them as assumptions.

### Gate 7 — Downstream readability
Every requirement (Story AC or FR) must be unambiguous when read in isolation
by Phase 3 (architecture) and Phase 4 (code generation). If interpreting a
requirement needs context from prose elsewhere in the document, inline that
context in the requirement itself.

---

## ID numbering convention

```
EPIC-001, EPIC-002, ...
FEAT-001, FEAT-002, ...       (global sequence, not scoped per Epic)
STORY-001, STORY-002, ...     (global sequence, not scoped per Feature)
STORY-NFR-001, ...            (Enabler / NFR stories — agile path)
FR-001, FR-002, ...           (default path functional requirements)
NFR-PERF-001, NFR-SEC-001, NFR-AVAIL-001, ...  (default path NFRs)
UC-001, UC-002, ...           (default path use cases)
TEST-001, TEST-NFR-001, ...   (test hook stubs — populated by Phase 5)
```

Every ID in every family MUST match exactly one of the shapes shown above —
a fixed prefix followed by a zero-padded integer (three digits or more).
**Never** append a letter suffix (``STORY-011A``, ``FR-014B``), a decimal
(``STORY-011.1``, ``FR-014.2``), a dotted child (``STORY-011.a``), or any
other extension when you need "another one like the previous". Every
requirement — including one produced by splitting a larger story under
Gate 3 or renumbering under Gate 5 — gets its own fresh integer from the
next-available position in that family's global sequence.

When the user message includes a prior `SPEC_REQUIREMENTS.md` (change-request
mode — see below), continue from the highest existing ID in each sequence.
Do not renumber existing IDs even if gaps exist.

---

## Change-request mode

When the user message carries a change-request preamble (one or more
`CR-N` blocks plus the prior spec), apply these rules instead of regenerating
from scratch:

1. Treat the prior spec as the baseline. Carry baseline requirements forward
   verbatim unless a CR explicitly changes them.
2. Emit only the deltas — new, modified, or removed requirements — and wrap
   each modified passage with the markers the change-request flow expects:
   ```
   <!-- BEGIN CR-N -->
   ...new or modified requirement(s)...
   <!-- END CR-N -->
   ```
3. Continue ID sequences from the highest existing ID in the prior spec.
4. Re-run all quality gates against the full document (baseline + deltas), not
   just the deltas.

The harness preserves the prior spec body verbatim below a revision header
when it writes the file — your job is to emit the delta block that gets
prepended, not to repeat the entire baseline.

---

## Output format rules

1. Output is **Markdown only**. There is no JSON output mode in this harness.
2. Begin the document directly with the first heading (`# Software Requirements
   Specification`). Do NOT wrap the document in an outer ```markdown ``` fence
   — fenced blocks are reserved for code / Gherkin / tables INSIDE the body.
3. Max section depth: 4 levels (`####`). Restructure rather than go deeper.
4. Write requirements in active voice, present tense.
   - RIGHT: "The system shall validate the token before processing the request."
   - WRONG: "Token validation will be performed by the system prior to processing."
5. Emit no preamble, no postscript, no "Here is your specification:" framing —
   the file is written verbatim to disk.

---

## What this skill does NOT do

The following are out of scope for the requirements step:

- Generating source code, file scaffolds, or directory structures (Phase 4).
- Designing component boundaries, choosing frameworks, or defining API shapes
  beyond requirement-level interface descriptions (Phase 3).
- Writing test cases or populating the test files referenced by the test hook
  IDs (Phase 5).
- Making UI/UX decisions beyond what is specified in the product notes.
- Inferring requirements the product notes don't support. If a critical piece
  of information is missing, emit it as a documented assumption with an
  `ASM-NNN` ID and a one-line rationale so the reviewer can confirm or
  override it.
