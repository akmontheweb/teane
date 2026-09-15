# Lumina run issues — 2026-09-11 → 2026-09-15

Companion to `finsearch_run_issues.md`. Record of five `teane build` runs
against `~/work/projects/lumina` (the Birthday Manager target workspace),
the harness defects they exposed, and what was done about each.

Written as a session summary, not a transcript.

---

## Run ledger

| Run | Session | Outcome | Terminated by |
|---|---|---|---|
| 1 | `lumina-testrun-20260911-1102` | exit 1, 0 files, $0.009, 2 min | `decomposition_validation_failed` |
| 2 | `lumina-fresh-20260911-1107` | exit 1, 53 files, $0.39, 33 min | `reflection_distraction_loop` 3/3 |
| 3 | `lumina-verify-20260914-1151` | killed manually, 1h47m | runaway loop (827 compiles) |
| 4 | `lumina-run4-20260914-1608` | exit 1, 49 files, $0.32, 16 min | `persistent_build_failure` 3/3 |
| 5 | `lumina-run5-20260914-2120` | exit 1, 45 files, $0.38, 22 min | `zero_patch_loop` 3/3 |

Trajectory of the repair loop across runs — the clearest signal that the
work landed:

| | Run 3 | Run 4 | Run 5 |
|---|---|---|---|
| Reflection verdicts | 12 × DISTRACTION | 6 / 2 / 1 | REGRESSION, PROGRESS, DISTRACTION, PROGRESS… |
| Failing diagnostics | pinned at 4 | down to 1 | **reached 0** |
| Compiler runs | 827 | 17 | 17 |

A judge that never says PROGRESS across twelve rounds is stuck, not
judging. By run 5 it discriminated between rounds and the diagnostic count
actually descended.

---

## Defects found and fixed

Twenty-one commits, `5de4459` → `d951b7a`. Grouped by theme.

### Structured output

* **`5de4459`** — the post-mortem prompt received only a trigger string and
  the build command (221 tokens). It read the word "decomposition" as
  Python module layout and wrote a learned rule about import paths that had
  nothing to do with the planner returning prose. That rule persisted into
  repo memory for injection into later runs. Now carries the node error,
  the reflection verdict, the judge's blocker and its recommendation.
* **`6ca0627`** — `decomposition_node` was single-dispatch, so one bad
  serialisation roll was a hard build failure at the first node. Run 1 died
  in two minutes having spent $0.009; a rerun on the same spec decomposed
  cleanly. Retries once with a JSON-only nudge, and the headless banner no
  longer tells the operator to edit a spec that parsed correctly.
* **`457ced3`** — B3: the plan is submitted as a strict tool call, with a
  second non-thinking pass that serialises prose if the first answers in
  prose. Five consecutive runs have landed it on attempt 1.

### Provider portability

Three provider-specific assumptions were baked into deliberately
model-agnostic paths. All three were invisible in logs and only surfaced by
auditing on-disk prompt dumps.

* **`f31b538` / `022ab60`** — `tool_choice` did not exist. Added, then
  immediately found that DeepSeek 400s on any compelled choice in thinking
  mode (`"Thinking mode does not support this tool_choice"`). A per-model
  capability flag cannot work on a platform where users pick the model, so
  the gateway now catches the refusal, downgrades to `auto`, retries, and
  remembers the pair for the session.
* **`2745ad5`** — `strict` tools, with the same recovery. Strict survives
  thinking mode where forcing does not, making it the more portable
  guarantee.
* **`e3a4140`** — the harness spoke one tool-call dialect. Run 5 lost four
  repair rounds because deepseek-v4-pro asked to read a file in Anthropic's
  XML syntax and the harness saw prose. Now recognises Anthropic XML,
  Hermes/Qwen, Mistral, Llama `<|python_tag|>`, DeepSeek special tokens and
  Nemotron, plus a permissive JSON tier for dialects nobody has catalogued.
* **`d951b7a`** — Anthropic requires `messages[0]` to be a user turn.
  Hoisting system messages out of the OpenAI-shaped array exposed a leading
  assistant turn on 5 of 16 measured repair dispatches, so routing `repair`
  at an Anthropic model would reject roughly a third of rounds.

### The repair loop

* **`7252c6f`** — the `UNSATISFIABLE_TEST` escape was model-declared, so a
  model that never writes the line declines it forever. Run 2 offered it 15
  times and was correct every time; the build died anyway. Now taken on the
  model's behalf after N declines with no real patch.
* **`6b8760e`** — a no-op on the judge's own target was scored as ignoring
  the judge. It is the opposite: the model went to the named file and
  reported nothing to change. Now evidence *against* the judge.
* **`8450051` / `314a024`** — revert detection. Run 3 spent six rounds
  alternating between `default=1` and `default=lambda: 1` on one SQLAlchemy
  column; both are equivalent (flush-time either way) and every progress
  counter read the oscillation as forward motion. The first implementation
  false-positived on 17 files per round because `modified_files` is
  cumulative — fixed by recording state transitions rather than
  observations.
* **`47a0997`** — the judge's grounding rule ("the file:line locations
  shown are the ONLY files you may name") made a defect one hop from the
  diagnostics structurally unnameable. Run 4 ended on exactly that: the
  endpoint raised the right error and a bare-except middleware in
  `main.py` overwrote its message. `main.py` is line 15 of the failing
  test's own import list and appeared in no diagnostic. Evidence widened
  rather than permission loosened.
* **`7d804f3`** — an unbounded `router → regeneration(no-op) → compiler`
  loop. Run 3 span 813 identical cycles in ~68 minutes. Nothing stopped it
  because the loop makes no LLM calls, so budget — the usual backstop —
  never moved off $0.24.
* **`7585667`** — a generated test using `@pytest.fixture` without
  importing pytest could not be fixed by anyone: the repair loop is
  forbidden to edit tests, and no production change makes `pytest` defined.
  Carve-outs now key on collection *phase* rather than an allowlist of
  error codes.

### Decomposition quality

* **`5e93724`** — the cross-domain scope guard compared tokens by exact set
  intersection, dropping `ConfirmDialog.tsx` from the delete-*confirmation*
  story and `NavBar.tsx` from a *navigation* story. Prefix matching, not a
  stemmer.
* **`d036611`** — ADR-0004 fans one NFR policy across every story it
  constrains, and the quality reviewer read the repeated criteria as
  duplication and recommended merging stories ADR-0004 requires to stay
  separate. Two harness features contradicting each other.
* **`41d4472`** — `quality_enforce` was binary: discard every finding or
  fail the build. With 13 high-severity findings on an ordinary spec,
  enforcing would have bricked the run. Bounded auto-remediation rewrites
  flagged criterion text in place, preserving row ids so `test_verifies_ac`
  edges survive.

### Observability

* **`29cfd86` / `91d6555`** — 31 dispatch sites passed no `cache_family`,
  so drift warnings for planning / doc_reviewer / patching were false by
  construction. Naming families after the enclosing function was then too
  coarse for `review_and_revise_spec`, which serves both spec gates.

---

## Open — context budget

Measured across runs 4 and 5:

| role | n | median in | median out | ratio |
|---|---|---|---|---|
| repair | 29 | 67,394 | 165 | **~400:1** |
| patching | 22 | 64,223 | 795 | ~80:1 |
| planning | 6 | 8,068 | 20,315 | inverted (generative) |
| judgment | 24 | 4,358 | 150 | ~29:1 |

80% of a repair prompt is one 175k-char system message carrying both specs,
the design system and the patch DSL rules. On a round fixing Python date
arithmetic it included 3,032 chars of `tailwind.config.js` guidance and
2,409 of an interaction-states matrix.

Simultaneously, the file the model asked for twice —
`server/app/api/birthdays.py` — was **mentioned 13 times but never
inlined**. Overfed on the irrelevant, starved of the specific.

### Why the needed file was missing

`graph.py:17489` already prefetches the failing test's first-party imports
(capped at 6). It missed `api/birthdays.py` because that file is **two hops
away**: `test_birthdays_api.py` imports `server.app.main`, and `main.py`
registers the router from `api/birthdays.py`. The `max_depth=2` parameter
in `_first_party_imports_for` governs *path resolution*
(`search.py` vs `search/__init__.py`), not transitive imports — the
prefetch is strictly one hop.

### Proposed change 2 — inline the file the judge names

Seed `files_for_preflight` with the judge's named files **before** the cap
of 6 is consumed, rather than letting them compete with alphabetical
iteration order. The judge names the blocker every round, so it is the
highest-value candidate and the signal is already computed
(`last_reflection_verdict`). A few lines; reuses existing machinery.

Rejected alternative: walking imports transitively. Broader, exhausts the
cap with breadth nobody asked for, and would still not prioritise the file
that matters.

### Change 2 — SHIPPED as `af47985`

Judge-named paths go to the front of `files_for_preflight`, capped at 3,
existence-checked and deduped against the diagnostics.

### Change 1 — HELD, deliberately (2026-09-15)

Two flaws surfaced while designing it, both of which argue against building
it on present evidence.

**Dropping the design system from repair is unsafe.** Repair touches
frontend files — run 4 generated and repaired `Dashboard.tsx`. Removing the
styling contract from repair prompts would let frontend rounds emit
off-system components. The existing tag-based filter
(`_detect_workspace_stack`) is already correct: it loads the web design
system because the workspace genuinely has a React frontend. The waste is
that it ships on a *backend round*, which is a per-round property.

**Making the filter per-round destroys what it optimises.** A
diagnostic-aware filter changes `messages[0]` whenever the diagnostic mix
flips backend↔frontend, and `messages[0]` IS the ~46k cached prefix. That
trades ~12k chars of context for repeated cache re-creation — the same
thrash two commits this session were spent eliminating.

The one cut that is both safe and role-stable is the test-authoring policy
(`## Test Strategy`, coverage targets, coverage gate — 10,821 chars
measured). Repair is architecturally forbidden to edit tests, so that
guidance is dead weight in a repair prompt regardless of the round.

| Section | Chars | Usable by repair? |
|---|---|---|
| Test Strategy + Coverage | 10,821 | No — repair cannot edit tests |
| Design system | 12,753 | Sometimes — frontend rounds need it |
| Patch syntax + rules | 15,251 | Essential |

**Decision: hold entirely until a run says more.** The justification for
trimming — that 67k tokens of largely irrelevant context degrades a
165-token answer — is the one claim in this audit that is measured in
volume but unproven in effect. Change 2 may have removed the actual
starvation, in which case the context budget was never the problem.

Revisit if a future run shows the model asking for files it was already
given, or repair quality degrading as the spec grows.

### Superseded sketch — scope the system prompt by round type

`messages[0]` is assembled once in `cli.py` (requirements doc plus
`SPEC_ARCHITECTURE.md`, appended at 8156) and shared by every role. It is
also the ~46k-token cached prefix at an 82% hit rate.

Making it role-specific **fragments the cache** — one prefix per role
instead of one overall. Each still caches within its role (repair has 29
calls, so it amortises), but cache *creation* is paid 4–5 times.

So the design is a filtered view rather than a rebuild: keep one canonical
assembled prompt, and apply a deterministic role-aware filter that drops
whole `##`/`###` sections by heading. Repair drops the design system,
coverage targets/gates and the `tailwind.config.js` module (~12k chars
measured); a backend-only diagnostic set additionally drops
`### Frontend modules`. Determinism matters — a filter that varies
round-to-round destroys the per-role prefix it is trying to shrink.

### Sequencing

Change 2 is clearly worth doing: small, targeted, fixes a demonstrated
starvation. Change 1 is worth **measuring** before committing to — the cost
is attention, not dollars, and that is the one claim here not backed by
evidence. Do 2, re-run, and see whether the model still asks for files it
should have been handed.

---

## Open — unverified live

Shipped with unit coverage, never exercised by a real build: the corrected
revert detector, the widened judge evidence, carve-out 3, all four dialect
layers, and the Anthropic message-shape fix (which additionally needs an
`ANTHROPIC_API_KEY` to confirm).

## Open — not defects

* Every routing role has `primary == fallback` (`deepseek-v4-pro`), with
  `best_of_n` and `speculative` off, so every escalation is a re-roll of
  the stuck model. A deliberate cost choice.
* Two NFR stories in run 2 were decomposed with zero acceptance criteria —
  schema-valid, semantically empty.
* The zero-AC deterministic finding uses `rewrite_ac` as its action though
  the remedy is *create*, so remediation matches it and silently no-ops.
* `judge_target_noop_streak` resets on any real patch, so in a partially
  stuck loop it rarely reaches the 2 it needs to render.

---

## Recurring lesson

Three separate fixes failed the same way before being corrected: a
capability catalogue that cannot be completed, a format enumeration that
cannot be exhaustive, a progress counter that mistook observation for
change. The pattern that worked each time was the same — **do not depend on
knowing everything in advance; make the failure detectable and
self-correcting.** Learn-and-downgrade for capabilities, a permissive tier
plus corrective feedback for formats, transitions rather than observations
for state.

The corollary for this codebase: a round that fails because the harness did
not *understand* it must never be counted as a model that *refused*. Every
progress counter reads "zero patches" as refusal, and that single
conflation turned a parse gap into a false `UNSATISFIABLE_TEST` verdict.
