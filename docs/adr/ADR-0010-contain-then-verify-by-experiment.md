# ADR-0010: Contain Failure, Then Verify by Experiment — Rethinking the In-Build Test Loop

**Status:** Accepted in direction. **Step 1 (park-and-continue) shipped, default on, not yet validated live.** Steps 2–4 are proposed and each is gated on the previous step's live result — see Action Items.
**Date:** 2026-11-28
**Deciders:** Teane harness maintainers
**Related:** [[ADR-0001]] (test-author regeneration), [[ADR-0003]] (deterministic test tiers), [[ADR-0005]] (shift-left test quality), [[ADR-0006]] (in-build acceptance — Step 1 extends its `_park_over_cap` containment to the level where runs actually die). Builds on `hitl_menu_loop` (`cli.py`), `acceptance_gen.quarantine_function`, `story_state.record_acceptance_deferral` / `record_defect`.

## Context

Nearly every failure of the last three months has been in test execution. That
is not a backlog of unrelated bugs. Every lumina run on record since September
terminated inside the generate-tests → run → attribute-blame → repair loop, and
each on a *different* mechanism:

| Run | Result | Terminated by |
|---|---|---|
| 5 | 12/16 | killed by hand (best result on record) |
| 6 | 4/16 | HITL class counter ratcheted 1→6 across five *resolved* blockers |
| 8 | 0/16 | test regeneration spent both attempts writing nothing |
| 9 | 0/16 | generated unit test with wrong arithmetic; coverage gate refused its removal |
| 11 | 4/16 | acceptance test asserted the spec's shorthand; regeneration reproduced it |
| 12 | 0/16 | regenerated test could not run (`patch('datetime.date.today')`) |
| 13 | 0/16 | repair produced 123k characters of deliberation and no patch |
| 14 | 0/16 | not examined for this ADR |
| 16 | 0/16 | harness declared a correct test unsatisfiable after four declines |
| 20 | 4/16 | harness generated a wrong contract test, guarded it, bent correct code to it |
| 01a21db1 | 0/18 | evidence flicker; three correct tests declared unsatisfiable |

Since run 5 the story count reads 4, 0, 0, 4, 0, 0, 0, 4, 0. In the same
period, of 152 commits, 115 are fixes and 92 touch this loop. Each fix was
validated live, and the next run died one step over. `repair_node` is now
5,274 lines and reads or writes 80 distinct `loop_counter` keys.

The fixes also swing. `7252c6f` forces the `UNSATISFIABLE_TEST` escape after
declined offers, to stop thrash; run 16 then blames a correct test; `e64be1c`
adds an evidence veto; run 20's set-wide veto blocks a genuinely unsatisfiable
test; `5adb3ac` makes the veto per-file; run 01a21db1 forces the escape inside
an evidence flicker; `8b5db6c` makes evidence sticky. A further veto was built
and reverted the same day because it re-blocked run 20's case. That is one
decision boundary being moved by hand, one incident at a time, with a costly
error on each side of it.

Four structural causes sit under all of it.

1. **There is no oracle.** Code, tests, the reflection judge and the reviewers
   are one model (`deepseek-v4-pro`, `non_thinking`). Generated tests are
   tamper-guarded as if they were true. When test and code disagree, blame is
   inferred from failure *text* — regex fingerprints, one AST shape, a judge
   that reads logs. The facts that settle the question usually need an
   experiment, and nothing in the loop can run one.
2. **The model has no execution tool.** `tool_schemas.py` offers edit and read
   tools only. A patch is verified by the next full outer cycle. Sandbox runs
   average about two seconds (201 builds in 6.6 minutes across runs 11–12), so
   this is not a cost constraint. Run 01a21db1 spent 176 repair dispatches on
   roughly 13 rounds: the model already loops, blind.
3. **Authority is split.** Repair may not edit tests. The test author may not
   edit code, gets two attempts, and regenerates from the same possibly-wrong
   source. A one-line test fix costs a distraction streak, an offer, three
   declines, a forced escape, a router hop, regeneration and its gates.
4. **Failure is not contained.** With `hitl.repair: false`, every HITL trip
   auto-resumes, and when a session, class or blocker cap is exhausted
   `hitl_menu_loop` sets `hitl_abandon` and the run ends. Run 12 had 13 of 17
   criteria verified and sealed zero stories. [[ADR-0006]] already parks an
   over-cap acceptance failure and seals the batch — but only when control
   returns to `acceptance_node`, and the inner repair loop that dies on a cap
   never returns there.

A fifth point is narrower. The unit tier is anchored on the code under test
(one test file per source file, `# @tests:`), yet a unit test that fails at
birth defaults to `CODE_GAP` and goes to repair, which may only change the
code. [[ADR-0005]] measured 77% of repair calls fighting such tests.

**Forces at play:**

- **The objective is a working application.** Story count is the measure; a
  run where every mechanism behaves and no story seals is a failure.
- **North-star: autonomous processing.** A run that stops for a human who is
  not there has failed. Parking with a record is autonomy; termination is not.
- **The tamper guard exists for a reason.** Letting the code author rewrite
  tests to go green is the reward hack [[ADR-0001]] was written to prevent.
  Any design that loosens it must replace the protection, not drop it.
- **A parked defect must stay visible.** Sealing a story over a quarantined
  test is honest only if the record says so and the quarantine clears itself.
- **This model is weak at tool calling.** Run 13 and the dialect fixes of
  September show it. Anything that leans on a longer tool loop is a risk.

## Decision

**Change the shape of the loop in four ordered steps, keeping its parts.** Step
1 is decided and built. Steps 2–4 are the intended direction; each waits on the
live result of the step before it.

### Step 1 — Park the blocker and continue (shipped)

When a headless auto-resume cap is exhausted, and the thing blocking the build
is a bounded set of failing tests, quarantine those tests and resume instead of
abandoning the run.

- **Where.** `hitl_menu_loop`, at the two places a headless run gives up on a
  test it cannot fix: the cap-exhausted branch, and the branch for a test file
  no node may author (`test_regen_unsupported` / `test_regen_stuck`).
  Budget exhaustion and decomposition failure still terminate.
- **What is parked.** The failing pytest node ids, taken from the diagnostics
  and from the raw build output (the diagnostic dedup keys on file and error
  code, so it under-reports). Each function gets the existing
  `teane-quarantine` non-strict `xfail` marker through
  `acceptance_gen.quarantine_function`.
- **When parking is refused,** and the run terminates as before: no failing
  test can be identified; a test module fails to collect; a failure is not a
  Python test; the failing set is larger than
  `hitl.max_parked_tests_per_blocker` (a large set means a broken build, not a
  blocker); `hitl.max_parked_blockers` is spent; or every failing test is
  already parked, which means the previous park did not turn the build green.
- **Verification is empirical.** Nothing tries to prove in advance that the
  park will work. The resumed compile says so. If the same blocker returns,
  there is nothing new to park and the run terminates one cycle later.
- **The record.** An acceptance test's criterion gets an `attributable`
  deferral, so the batch reports it unverified. Every parked test gets a
  `defects` row with severity `parked_at_hitl_cap`. The park is logged as a
  banner, emitted as a `blocker_parked` event, and kept in
  `loop_counter["parked_blockers"]`.
- **Staying parked.** `acceptance_node` regenerates unverified scenarios on
  every entry, which would drop the marker and start another inner repair loop.
  So a park that includes an acceptance test also spends the batch's
  `acceptance.max_repair_rounds_per_batch`; on re-entry the existing
  `_park_over_cap` path quarantines whatever is still attributable and seals.
- **Caps keep their meaning.** No counter is reset by a park. The class budget
  is refunded by the existing progress rule when the parked blocker is gone
  and a new one appears. Once the session cap is spent, each later blocker is
  parked on its first escalation rather than retried.

### Step 2 — Give the repair model a `run_tests` tool (proposed)

Add one execution tool to the existing in-round tool loop: run named pytest
selectors in the sandbox and return the outcome. The model can then check a
hypothesis before the outer cycle, and can show a test to be wrong by running
it. Bounded by a per-round tool-call cap.

### Step 3 — Make the unit tier characterization tests (proposed)

A unit test is anchored on the code. Generate the unit tier against code that
is already acceptance-green, and require each test to pass at birth. One that
does not is the suspect and goes back to its author, unless the failure shows
the application crashing. Turn the property tier off, as [[ADR-0003]] says it
should be until its false-positive rate is measured.

### Step 4 — One agent per story, with authority over test and code (proposed)

Let the repairing agent edit a test as well as the code. Replace the tamper
guard with a narrow independent review of each test edit: *does the changed
assertion still verify the criterion's text?* Keep `teane test` as the
held-out gate. Then delete the escape ladder this makes redundant.

### Config (Step 1; all config-driven per no-hardcoded-caps)

| Key | Default | Meaning |
|---|---|---|
| `hitl.park_blocker_on_cap` | `true` | Park instead of terminating. `false` restores the old behaviour exactly. |
| `hitl.max_parked_blockers` | `12` | Parks allowed per session, clamped [0, 200]. `0` also disables parking. |
| `hitl.max_parked_tests_per_blocker` | `12` | Largest failing set one park may quarantine, clamped [1, 200]. |

## Options Considered

### Option A: Keep fixing the ladder

Each incident gets a targeted fix. This is what the last 92 commits did. The
fixes are individually sound and validated; the run results show no trend, and
the boundary between "blame the test" and "blame the code" keeps moving.

### Option B: Raise the caps

Buys more retries on the same blocker. `5bda93e` already did this once; run 6
then died on the larger number. It delays the termination without changing
what happens at it.

### Option C: Contain first, then change how blame is settled (this ADR)

Stop a single unfixable test from ending the run, then give the model the
means to settle blame by running things, then remove the tier and the split
that generate most of the false failures.

### Option D: Replace the pipeline with a general agent that has a shell

One agent, a sandbox shell, tests as feedback. This is what most current
coding agents do and it removes causes 1–3 at once. It also discards the
decomposition, traceability, gating and cost machinery that works, and leans
hardest on the tool-calling this model is weakest at.

### Option E: Drop in-build tests and rely on `teane test`

Removes the loop. It also returns to the state [[ADR-0006]] was written to
leave: behavioural defects found only after everything is built.

## Trade-off Analysis

Option A has had a fair trial and the trend is flat. B has been tried. E gives
up the acceptance tier, which found real defects in runs 16 and 20. D is the
cleanest end state but the largest bet on the weakest capability, and it is
all-or-nothing.

Option C reaches most of D's end state in steps that each stand alone. Step 1
changes no decision the loop makes; it only replaces "end the run" with "park
and continue" at a point where the loop has already given up. It cannot make a
run seal fewer stories than it does today, and `false` restores today's
behaviour byte for byte.

What Step 1 costs is honesty pressure. A story can now seal `done` over a
quarantined test. [[ADR-0006]] already accepts that for acceptance failures
over the repair cap; this widens it to the unit tier and to the HITL level.
The mitigation is the record — deferral, defect row, event, and a marker that
clears itself when the test passes — and the bound on how much may be parked.

The cap on tests per park is the judgment call. Too low and a real blocker
spanning several criteria terminates the run as before; too high and an
application that barely works is sealed around a wall of quarantined tests.
Twelve is a starting point taken from run 01a21db1, whose blockers held seven
to nine failing tests. It should be revisited with data.

## Consequences

**Easier:**
- A run survives a test nobody can fix. The batch seals, later batches run.
- Every later step can be tried without a single bad test ending the trial.
- Post-mortems get a list of parked blockers instead of one termination line.

**Harder:**
- "Stories done" needs reading alongside "tests parked". The count can rise
  while quality does not.
- A quarantined unit test is not self-clearing. Only the acceptance path
  strips a marker on XPASS today.
- After the session cap is spent the harness parks on first escalation, which
  gives up on tests one more resume might have fixed.

**Not covered by Step 1:**
- JavaScript tests. Only pytest node ids are parked; a failing vitest suite
  still terminates the run.
- A build that does not compile or collect. There is nothing to quarantine.
- Blame attribution itself. Every mechanism that decides test-versus-code is
  untouched; Step 1 only bounds what a wrong decision can cost.

## Action Items

1. [x] **Step 1** — `harness/blocker_parking.py`, wired into `hitl_menu_loop`;
   three `hitl.*` keys registered and documented; tests in
   `tests/test_blocker_parking.py`.
2. [ ] **Validate Step 1 live** on a clean lumina build. Report stories sealed,
   blockers parked, and tests parked per blocker. The step has worked if the
   run ends on budget or completion rather than on a cap.
3. [ ] **Report parks at the end of a run** — a summary line and a field in
   the build record, so a sealed story's parked tests are one glance away.
4. [ ] **Second pass over parked tests** with whatever budget remains at the
   end of the run, with fresh context.
5. [ ] **Self-clear a quarantined unit test** on XPASS, as the acceptance path
   does.
6. [ ] **Step 2** — `run_tests` in the repair tool surface. Decide after
   item 2.
7. [ ] **Step 3** — characterization unit tier; property tier off.
8. [ ] **Step 4** — single-agent authority with a test-edit review; then
   remove the ladder machinery it replaces.
