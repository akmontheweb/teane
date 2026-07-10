# Teane run issues — finsearch build (session `44c5e194`)

Diagnostic dive triggered after the finsearch build claimed "done" while
producing ~1/5 of the specified features. Session ended with
`exit_code=1` from the traceability-block gate, but state.db was left
with every story marked `done` and every batch `complete`.

Source signals:
- Log: `~/.harness/logs/44c5e194-5715-451f-92c6-84362eeb7453.jsonl`
- State: `~/.harness/state.db` (workspace=`finsearch`)
- Traceability: `finsearch/docs/TRACEABILITY.md` — every row "0 code / 0 tests"

Each entry below has a status marker. Fixes land in this repo (`teane`);
finsearch-side ancillary bugs are called out but not fixed here.

Status legend: `[ ]` open · `[~]` in progress · `[x]` fixed & verified · `[!]` deliberate no-op / out of scope

---

## A. Root-cause bugs (teane) — highest leverage

### A1. `file_links` unpopulated in batch mode  `[x]`
`story_state.link_file` is the only writer, and it's called only from
`story_complete_node` (`harness/story_loop.py:779`). The graph bypasses
`story_complete_node` in batch mode (comment at `harness/graph.py:391-398`).
`batch_commit_node` → `seal_batch_atomically` (`harness/story_state.py:1951`)
seals stories `done` but does no linking. Result: `file_links: 0 rows` for
finsearch and every batch-mode run. TRACEABILITY.md reads "0 code / 0 tests"
for every story regardless of what shipped.

**Fix plan:** thread `batch_modified_files` into `seal_batch_atomically`; for
each modified path, call `link_file` against every story in the batch (kind
inferred from path). Attribute-per-file granularity isn't recoverable
post-hoc; batch-level is the honest lower bound.

### A2. pytest exit 5 ("no tests collected") is treated as build success  `[x]`
`compiler_node` log: *"Test runner reported no tests collected (exit=5) but
workspace has source files — treating as success and advancing the graph."*
Early greenfield rounds legitimately have no tests, but once
`test_generation_node` has emitted anything, "no tests collected" means
the runner isn't seeing them — not that everything passes. This is the
single biggest reason 8+ epics show shipped code with zero test attribution.

**Fix plan:** carve-out lifts only when the session has never entered
`test_generation_node`. After the first test-gen call, exit 5 must become
a repair-eligible failure.

### A3. `test_generation` HITL auto-resumes back to a passing compiler  `[x]`
Triggers `env_misconfig:test_generation_zero_emit` (×6) and
`test_generation_max_iterations` (×3) all `auto_resume=3/3` under
`HARNESS_AUTO_APPROVE`. Router sends control to `compiler_node`, which
passes on no-tests-collected (see A2), which seals the batch. HITL is
decorative under headless auto-resume.

**Fix plan:** mostly resolved by A2. Additionally, `zero_emit` HITL trigger
should escalate to `blocked` rather than looping when auto-resume budget
exhausted.

### A4. `seal_batch_atomically` seals stories regardless of whether patches landed  `[x]`
Batches 85–89 (feature 528, PLATFORM/NFR block) sealed 5-story groups
with `total_repairs=0` and `modified_files ∈ {2,3,7,10,11,12}`. No gate
on "did this batch produce any code touching this story's scope."

**Fix plan:** in `seal_batch_atomically`, require ≥1 file in
`batch_modified_files` OR downgrade the batch to `complete_with_blocks`
and park unlinked stories as `blocked` with a defect. (Overlaps with A1's
plumbing — do them together.)

### A5. `commit_on_story` default `false` + non-git workspace fails silently  `[x]`
`config/config.json:518` defaults `agile_defaults.commit_on_story: false`.
Non-git workspaces get `_commit_for_batch` returning `None` silently.
`commits: 0 rows` + `committed_sha NULL` on every batch → no rollback
point, no traceability commit column.

**Fix plan:** on session start, detect non-git workspace or
`commit_on_story=false` and emit a WARN with an actionable hint. Auto-init
git is too aggressive; opinionated warn is right.

### A6. `traceability_block` terminates with `exit_code=1` but doesn't roll back sealed batches  `[x]`
End of session: `reqs 76/84 (90%), ACs 18/135 (13%); untraced=8, untested=117`.
Router forces END to prevent an infinite HITL loop (learned from ciod
`523e86a7`'s 376-iteration incident), but state.db keeps every `done` /
`complete` row and TRACEABILITY.md is regenerated from that stale state.

**Fix plan:** when traceability gate forces END with exit≠0, downgrade
sealed-but-unlinked stories to `blocked` and re-render the doc before
returning. The exit is honest; the DB must be too.

---

## B. Planner / scope drift

### B1. `spec_reconciler` drift warning is a soft log  `[x]`
`[spec_reconciler] 2 LLM stories had no spec match (drift signal):
STORY-039='Report Customization', STORY-040='Source Traceability'` —
warned, then ignored.

**Fix plan:** on drift, force reconciliation against `SPEC_REQUIREMENTS.md`
and either remap or park the drifted stories as `blocked` before batching
starts. Silent warns get ignored.

### B2. Planner assigned bogus `scope_files` for STORY-032  `[x]`
`[story_loop] next story: STORY-032 — Source Traceability
(scope_files=['server/services/forecast.py',
'client/src/components/ForecastTab.tsx',
'server/tests/unit/test_forecast.py'])`
Those files don't exist and are unrelated to Source Traceability.

**Fixed with a five-part change** — three prompt edits (nudge the LLM
away from filling scope_files, add a domain-consistency rule, show a
`[]` example) plus two code changes (feed the workspace file tree
into the augment prompt so brownfield planning cites real paths, and
a deterministic drop-cross-domain guard in the validator). See fix
log below.

---

## C. Repair-loop fixation

### C1. Reflection flip-flopped `COMPLETED ↔ FAILED` for 6+ turns  `[x]`
Between 03:05 and 03:16 the reflection kept alternating between "should
be FAILED" and "should be COMPLETED" for `tests/test_ingestion.py:71`.
No memory of the last recommendation being the opposite.

**Fix plan:** carry the previous-round `recommendation` into the
distraction check; two rounds with opposing recommendations on the same
file/line = fixation, not distraction, and should escalate.

### C2. 12 consecutive turns on the same wrong hypothesis (PYTHONPATH/subdir)  `[x]`
From 14:16 → 14:26, twelve reflection rounds all said "pytest may be
running from a subdir." `consecutive_distraction_rounds` hit 3 three
times but reset each time — the reset condition lets the fixation loop
indefinitely.

**Fix plan:** don't reset `consecutive_distraction_rounds` on a PROGRESS
verdict if the underlying `real_blocker` string is unchanged from the
prior round; track a `same_hypothesis_streak` alongside.

### C3. `repair_fixation_breaker` fired only once despite many fixation windows  `[x]`
Threshold is too high given C2's evidence.

**Fix plan:** lower breaker threshold; add a hard cap based on
same-hypothesis streak from C2.

### C4. `repair_reflection_promoted_to_lead` fired 7/7 times with `verdict=DISTRACTION`  `[!]`
Every reflection promotion was on a distraction verdict — the promotion
criterion is inverted or misjudging.

**Fix plan:** audit `_maybe_promote_reflection_to_lead` (or equivalent);
promotion should require PROGRESS + high-confidence, not the current
signal.

**Update after deeper reading:** the promotion criterion is not inverted
— DISTRACTION verdicts *with real grounds* are exactly what should be
promoted (the design intent is "when the LLM is missing the point, hand
the reflection judge the wheel"). The finsearch symptom (7/7 = DISTRACTION)
is expected downstream of C1/C2: as long as the LLM keeps failing to
solve the real blocker, reflection keeps calling it a DISTRACTION and
those promotions are correct. Fixing C1/C2 breaks the underlying fixation
loop earlier via `consecutive_distraction_rounds` no longer resetting on
same-hypothesis PROGRESS verdicts — so the 7-round string of promotions
shouldn't accumulate in the first place. Deferred as its own fix.

---

## D. Patcher state drift

### D1. LLM re-emits `create_file` for existing files (15+ occurrences)  `[x]`
Files touched multiple times in a session: `server/models.py`,
`server/database.py`, `server/main.py`, `tests/test_ingestion.py`,
`tests/test_narrative_summary.py`, etc. Patcher refuses safely, but no
memory of "this file exists" survives to the next turn — same failure
recurs.

**Fix plan:** after a `create_file` refusal, inject a per-file anti-hint
into the next repair prompt: "you already tried create_file on X — the
file exists — use replace_block/insert_at_block."

### D2. `Search block not found — file may have drifted`  `[!]`
Patch base is stale versus disk. Patching preamble not shipping fresh
reads on every turn.

**Fix plan:** on a drift error, force a fresh read of the affected file
into the next turn's context.

**Already fixed by commit `d3462d9` (2026-07-09):** the "universal
stale-view guard" gates every REPLACE_BLOCK / DELETE_BLOCK / INSERT_AT
against files modified in the session — the LLM is required to emit a
fresh `READ_FILE` before its next edit. See
`_pre_patch_screen` guard 2 in `harness/graph.py:6909`. The finsearch
session that surfaced D2 pre-dated that fix. No further action.

---

## E. Infra / dependency

### E1. prod-smoke discovered missing deps one at a time (23 rounds)  `[!]`
`pydantic_settings`, `sqlalchemy`, `asyncpg`, `lxml` all discovered
serially over ~23 rounds. LLM added each to `requirements.txt`
individually — wasted ~5–10 repair turns.

**Fix plan:** first-pass dependency scanner should walk imports vs
`requirements.txt` upfront and seed missing entries in one pass, before
prod-smoke gets a chance to iterate.

**Investigation:** the per-round batching is already in place — a single
`DEPS_NOT_INSTALLED` diagnostic carries a `missing_packages` array and
`autofix._try_deps_not_installed` appends all packages in one edit
(`harness/graph.py:6179`, `harness/autofix.py:995`). The 23 fail-to-
import rounds are spread across 18 batches (1-2 packages per batch,
each batch adding new source with new imports) — not a single 23-round
cascade. Upfront static import scan is a larger design change beyond
this pass; deferred. Log the finding.

### E2. Rate-limit circuit opened 12+ times, diverted to local Ollama  `[!]`
Weaker model, worse patches, more retries. Compounded by DNS
"Temporary failure in name resolution" 11+ times. Infra/network — out of
scope for a teane code fix beyond what circuit-breaker already does.

### E3. Semgrep timed out at 15s  `[x]`
Security scan flagged INCOMPLETE, session terminated the phase clean.

**Fix plan:** raise per-scanner timeout floor to 60s (or make configurable);
warn distinctly when a scanner timed out vs when it crashed.

### E4. `requirements.txt` ended with `lxml==6.1.0.` (trailing dot)  `[x]`
A repair patch corrupted the version pin. Reflection caught it at
15:08:24 but the session was already out of HITL budget.

**Fix plan:** add a post-patch validator that runs `pip-compile --dry-run`
or a regex-lint on requirements.txt after any patch touches it. Reject
the patch on syntax failure so it doesn't burn the next repair budget.

### E5. `cache_prefix_drift` fired 34 times  `[!]`
Prompt-cache misses on every drift — meaningfully more $ per turn. Deeper
cache-management issue; not fixing in this pass.

---

## F. Finsearch-side (not teane)

### F1. `narrative_summary.py` discards the registered prompt  `[!]`
`server/services/narrative_summary.py:88-93` calls
`get_prompt("narrative_summary", text=text)`, receives a formatted prompt
string, and then discards it by calling `client.summarize(text)` with the
raw text. STORY-019 code that shipped doesn't actually route through the
prompt registry. Ancillary to this diagnosis — belongs on finsearch's
backlog.

---

## Fix log

Entries appended as each item lands.

### A1 + A4 — `seal_batch_atomically` now links files and blocks empty batches
- `harness/story_state.py:seal_batch_atomically` gained a
  `batch_files: list[(path, kind)]` kwarg. For every constituent story
  it inserts a `file_links(workspace, story_id, path, kind, batch_id)`
  row per file (upserting on the `(story_id, path, kind)` unique key so
  re-seals stay idempotent).
- Same function now parks stories `blocked` with a defect
  `severity='empty_batch_seal'` when `batch_files` is empty — instead
  of silently marking them `done`. Return type flipped to
  `(done_keys, blocked_count_after_seal)` so `batch_commit_node` sees
  the corrected count.
- `harness/story_loop.py:batch_commit_node` classifies each
  `batch_modified_files` entry via `_classify_file` and passes them
  through. It also unpacks the new tuple return so downstream
  `blocked_count` telemetry reflects any empty-seal parks.
- Tests: two new cases in `tests/test_batch_commit_node.py` —
  `test_empty_batch_parks_stories_blocked_not_done` (A4) and
  `test_batch_files_populate_file_links` (A1). Existing tests that
  relied on the buggy "empty batch → done" behavior updated to seed a
  synthetic `batch_modified_files=["src/a.py"]`.
- Verified: `pytest tests/test_batch_commit_node.py tests/test_batch_commit.py tests/test_story_state.py tests/test_story_loop.py tests/test_traceability.py tests/test_traceability_e2e.py tests/test_batch_gate_progress.py tests/test_batch_scope.py tests/test_batch_sizing.py tests/test_batch_topo_order.py tests/test_story_complete_traceability.py` → 249 passed.

### A2 + A3 — pytest exit 5 no longer trivially passes once test_gen has run
- `harness/graph.py` compiler_node: the "no tests collected → fold to
  success" branch is now conditional on `loop_counter["test_generation"]
  == 0` AND `state["generated_tests"]` being empty. Once
  `test_generation_node` has fired at least once, exit-5 synthesises a
  `TESTS_NOT_COLLECTED` compiler diagnostic pointing at the common
  causes (PYTHONPATH, conftest, testpaths, package layout, wrong CWD).
  Routes to repair instead of sealing the batch green.
- A3 resolves as a downstream consequence: the HITL cycle triggered by
  `test_generation_zero_emit` used to resume, run compiler, get a fake
  pass, seal the batch. Now the compiler surfaces
  `TESTS_NOT_COLLECTED` and repair actually has something to work on.
- Tests: new `TestNoTestsCollectedCarveOut` class in
  `tests/test_env_misconfig.py` with three cases —
  `test_greenfield_exit_5_still_folds_to_success` (preserves the
  legitimate early carve-out), `test_post_testgen_exit_5_surfaces_diagnostic`
  (files were emitted but not collected), and
  `test_testgen_iterated_but_zero_emit_still_surfaces` (test_gen ran
  but produced nothing).
- Verified: `pytest tests/test_env_misconfig.py` → 52 passed.

### A5 — warn once at session start when commits will be no-ops
- New helper `_warn_if_commits_will_be_no_ops` in `harness/graph.py`,
  called from `run_graph` right after session id resolution. Emits a
  single `WARNING` when either (a) the workspace has no `.git` dir or
  (b) the workspace is a git repo but `commit_on_story` is False. Both
  branches include an actionable fix hint. No-op when
  `decomposition_enabled` is False (single-shot runs don't hit the
  batch commit path).
- Tests: new `TestCommitNoOpWarning` class in
  `tests/test_env_misconfig.py` with four cases covering the matrix
  (non-git, git-but-flag-off, both-fine, decomposition-disabled).
- Verified: `pytest tests/test_env_misconfig.py::TestCommitNoOpWarning`
  → 4 passed.

### A6 — rollback unlinked done stories when traceability forces END
- New `rollback_unlinked_done_stories(conn, workspace, session_id)` in
  `harness/story_state.py`. Finds every story where `status='done'`
  AND `NOT EXISTS (file_links WHERE story_id = s.id)`, downgrades to
  `blocked`, records a `severity='traceability_rollback'` open defect
  per story.
- `route_after_installation_doc` in `harness/graph.py` now calls a new
  private helper `_rollback_unlinked_before_end(state)` on the
  cap-hit END branch. Rollback is best-effort — any exception logs
  a warning but doesn't prevent END. The router still exits cleanly
  (preserves the ciod `523e86a7` loop guard); the persisted state is
  now honest.
- Tests: three new cases in `tests/test_story_state.py` covering the
  rollback helper itself; one new case
  `test_end_at_cap_rolls_back_unlinked_done_stories` in
  `tests/test_traceability.py` covering the router → rollback wire-up.
- Verified: `pytest tests/test_traceability.py tests/test_story_state.py
  tests/test_batch_commit_node.py tests/test_env_misconfig.py` →
  170 passed.

### B1 — spec drift is now a WARNING with actionable guidance
- `_match_llm_to_spec` in `harness/spec_reconciler.py`: bumped drift
  log from `INFO` to `WARNING` with a clear "SPEC DRIFT" prefix and
  a two-option remediation hint (either add the stories to
  `SPEC_REQUIREMENTS.md` or fix the planner prompt). Full list of
  drifted stories now printed at DEBUG when >5 exist so operators
  investigating post-hoc can find them without re-running.
- Test: new `test_spec_drift_logged_as_warning` in
  `tests/test_spec_reconciler_links.py` asserting the WARNING fires
  with the story key and title.
- Verified: `pytest tests/test_spec_reconciler_links.py` → 3 passed.

### C1 + C2 + C3 — hypothesis-fingerprint fixation detector
- New `_hypothesis_fingerprint(verdict, workspace_path)` in
  `harness/graph.py` — returns a `frozenset[str]` of workspace-
  relative files named in the reflection's `real_blocker` +
  `recommendation`. Layered on the existing `_verdict_referenced_files`
  helper.
- New `_same_hypothesis_streak(current_fp, recent_fps)` — walks the
  rolling window backwards and counts consecutive intersections.
  Intersection (not equality) is deliberate: naming
  `{a.py, b.py}` then narrowing to `{a.py}` is the same hypothesis.
- Wired into the PROGRESS branch of the reflection verdict handler.
  A rolling window of the last 5 fingerprints is kept in
  `loop_counter["recent_hypothesis_fingerprints"]`. When the current
  fingerprint's streak against the window is ≥2, PROGRESS no longer
  resets `consecutive_distraction_rounds` — the LLM is oscillating on
  the same target, so the HITL circuit breaker keeps counting.
- C3: `fixation_breaker` now also fires when
  `same_hypothesis_streak >= 3` even if `no_progress_repairs` is low.
  Same-narrative fixation is a stronger signal than fingerprint-count
  stalls (the finsearch loop kept "making progress" on the count
  while never solving the actual bug).
- Tests: new `TestHypothesisFingerprint` class in
  `tests/test_low_signal_verdict_pipeline.py` with 7 cases covering
  workspace-relative extraction, empty-verdict handling, intersection
  vs equality semantics, streak-break on non-intersecting round, and
  streak-break on empty middle round.
- Verified: full-suite run `pytest tests/ --ignore tests/test_cli_basics.py`
  → 3863 passed, 1 skipped.

### B2 — five-part fix for cross-domain scope_files hallucination
- **Prompt #1 — empty is the default.** The good-story bullet in
  `_build_decomposition_prompt` no longer says "when you have a
  high-confidence guess"; it now says empty `[]` is the right default
  and only include a path when an AC literally names it or the file
  is already on disk. Same wording added to the augment prompt.
- **Prompt #2 — domain-consistency rule.** Both prompts now spell
  out: "Every scope_files entry MUST share a domain word with the
  story title, feature name, or an acceptance criterion" and cite
  the finsearch example ("Source Traceability" cannot point at
  `forecast.py`).
- **Prompt #3 — filled + empty example pair.** The STRICT JSON
  example in the initial prompt now has two stories: STORY-001 with
  a defensible `scope_files` entry, STORY-002 with `[]` and an
  explanatory note. Augment prompt example flipped to show `[]` as
  the shape.
- **Prompt #4 — workspace file tree.** New
  `_build_workspace_file_tree_hint(workspace_path, max_files=200)`
  walks the tree, prunes noise dirs (node_modules, __pycache__,
  .git, .venv, dist, build, coverage, .pytest_cache, etc.), and
  emits an alphabetical bullet list. Wired into
  `_build_decomposition_augment_prompt` so brownfield planning cites
  real paths. Greenfield path unchanged (nothing on disk yet).
- **Guard #5 — deterministic drop.** New
  `_drop_cross_domain_scope_files(story_key, scope, story_title,
  feature_name, feature_key, acceptance_criteria)` filters entries
  whose lowercased path tokens don't intersect the story's context
  tokens. Extracts tokens with camelCase + snake_case + hyphen
  splitting via new `_scope_path_tokens` / `_context_tokens`
  helpers; both share a `_SCOPE_GENERIC_TOKENS` filter for
  structural noise (src, tests, server, client, api, service,
  common file extensions, etc.). Falls through when context is
  empty (very short titles) rather than nuking everything. Every
  drop logs the story_key + path + both token sets for post-mortem
  tracing. Called from both `_validate_stories_payload` and
  `_validate_augment_payload` after `_enforce_stack_on_scope_files`
  (so JS→TSX rewrite still runs first).
- **Tests:** 20 new cases across `TestCrossDomainScopeGuard`,
  `TestScopePathTokens`, `TestContextTokens`,
  `TestWorkspaceFileTreeHint`, `TestB2PromptEdits` in
  `tests/test_decomposition.py`. Existing stack-normalize tests
  updated to give story titles/ACs that share tokens with the
  scope_files paths.
- Verified: `pytest tests/` → 4009 passed, 1 skipped.

### D1 — sticky per-session CREATE_FILE rejection accumulator
- New `_update_sticky_create_rejections(state, this_round_failures,
  this_round_modified)` in `harness/graph.py`. Reads existing
  `state["sticky_create_rejections"]`, adds any file where this
  round's failures include `operation="create_file"` AND `error`
  containing `"already exists"`, and DROPS any path that appears in
  `this_round_modified` (the LLM successfully used a non-CREATE
  operation → memory of the rejection is no longer needed).
- Wired into the return dict of both `patching_node` and
  `repair_node`, so the field survives every round of the session.
- Prompt: new `## STICKY: files with prior CREATE_FILE rejections`
  section injected into the repair prompt before the general workspace
  inventory, listing up to 30 sticky paths with a strong "do NOT
  emit CREATE_FILE" directive. Handles overflow with a `(+ N more)`
  footer.
- Tests: new `TestStickyCreateRejections` class in
  `tests/test_audit_graph_helpers.py` with 8 cases (fresh add,
  preserve prior, dedup, clear-on-modify, ignore non-create failures,
  ignore create failures without "already exists" wording, tolerate
  missing state key, tolerate malformed entries).
- Verified: `pytest tests/test_audit_graph_helpers.py` → 16 passed.

### E3 — semgrep timeout default 15s → 60s
- `harness/security.py:1940` and `1952`: `sast_timeout_seconds`
  default raised from 15 to 60. Operators can lower via
  `security.sast_timeout_seconds` when latency matters more than
  coverage.
- No new tests — this is a config default change with existing test
  coverage on the timeout handling machinery.
- Verified: `pytest tests/ -k security` → 158 passed.

### E4 — post-patch validator rejects invalid requirements.txt lines
- `harness/patcher.py:_validate_syntax` extended to cover
  `requirements.txt`, `requirements-dev.txt`, `requirements-test.txt`.
  Uses `packaging.requirements.Requirement` to parse each non-comment
  line; returns an error for the first invalid line naming the
  lineno + offending text. Comments, blank lines, and pip flags
  (`-r`, `-e`, `-c`, `--extra-index-url`) are skipped. Existing
  post-patch rollback machinery handles the rest.
- Falls back silently when `packaging` isn't importable — we don't
  block patches on our own missing dep.
- Tests: 5 new cases in `tests/test_patcher_validation.py` covering
  the happy path, the exact finsearch corruption (`lxml==6.1.0.`),
  requirements-dev.txt / requirements-test.txt symmetry, pip-flag
  passthrough, and basename-scoped filename check.
- Verified: `pytest tests/test_patcher_validation.py` → 22 passed.

### C4 — reflection promotion 7/7 = DISTRACTION (deferred, not-a-bug)
- Deeper reading of `_reflection_grounds_in_diagnostics` and the
  gated promotion branch showed the criterion is not inverted —
  DISTRACTION with real grounds is exactly what promotion targets
  ("when the LLM is missing the point, hand the reflection judge the
  wheel"). The 7-round string is a downstream symptom of C1/C2's
  fixation loop; the C1/C2 fix breaks that loop earlier by no longer
  resetting the distraction counter on same-hypothesis PROGRESS
  verdicts, so promotions won't accumulate the same way. No code
  change here.






