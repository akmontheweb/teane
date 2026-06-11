# Production Readiness Assessment

**Date:** 2026-06-10
**Target:** First paying customer / pilot
**Scope:** the harness itself (not the apps it generates)

---

## Verdict

**P0 + P1 + P2 cleared (2026-06-10).** All three P0 gaps, all ten P1 gaps, and all nine P2 polish items — including P2.7 (metrics endpoint), originally deferred but pulled forward — are closed. Regression suite is at **868 tests passing**, **ruff clean** end-to-end, and the CI quality job (lint + format-check + mypy) is wired with ruff as the blocking gate. CI matrix now covers Linux (blocking) plus macOS and Windows (advisory).

The harness is ready to charge a customer subject to the standard pilot caveats (Linux primary, single-tenant, hands-on hotfix availability).

---

## Severity scheme

- **P0 — Blocks pilot.** A real customer using the harness with no fix would experience data loss, secret leak, or persistent failure inside their first hour. Must fix before invoicing.
- **P1 — Must fix before scale.** Pilot can ship with this open if the user accepts the risk in writing, but a second customer or longer pilot exposes it.
- **P2 — Polish.** Quality, ergonomics, future-proofing. Not load-bearing for the pilot.

Every finding includes file:line so it's a punch-list item, not a discussion topic.

---

## Summary

| Dimension | P0 | P1 | P2 | Status |
|---|---|---|---|---|
| Security — secret handling | 0 | 0 | 0 | fixed |
| Security — sandbox & file writes | 0 | 0 | 0 | fixed |
| Security — input & supply chain | 0 | 0 | 0 | fixed |
| Reliability — LLM dispatch | 0 | 0 | 0 | fixed |
| Reliability — checkpoint & concurrency | 0 | 0 | 0 | fixed |
| Reliability — repair / HITL | 0 | 0 | 0 | fixed |
| Operations — CI & packaging | 0 | 0 | 0 | fixed |
| Operations — logging & lifecycle | 0 | 0 | 0 | fixed |
| Operations — license & versioning | 0 | 0 | 0 | fixed |
| Operations — defense-in-depth wiring | 0 | 0 | 0 | fixed |
| **Total** | **0** | **0** | **0** | |

**P0 closeout summary (2026-06-10):**
- P0.1 — checkpoint message redaction wired in `harness/storage.py` (`HarnessAsyncSqliteSaver.aput` / `aput_writes`); opt-out via `persistence.redact_messages: false`. Regression tests: `tests/test_storage_basic.py::TestCheckpointMessageRedaction`.
- P0.2 — process-wide `CommandValidator` set in `cmd_run` / `cmd_resume`; `SandboxExecutor.__init__` falls back to the global when no validator is passed. Regression tests: `tests/test_security_basic.py::TestSandboxExecutorPicksUpGlobalValidator`.
- P0.3 — `LICENSE` (MIT) added at repo root; `pyproject.toml` switched to `license = { file = "LICENSE" }`.

**P2 closeout summary (2026-06-10):**
- P2.1 — `requirements-prod.txt` with exact transitive pins; documented `pip install -e . --constraint requirements-prod.txt`.
- P2.2 — 1 MB pre-flight + depth-10 recursion guard in `harness/trust.py::validate_discovery_json`. Tests: `tests/test_trust.py::TestValidateDiscoveryJson::test_oversized_total_response_rejected`, `test_deeply_nested_response_rejected`.
- P2.3 — `RotatingFileHandler` default (10 MB × 5 backups) in `harness/observability.py`; `logging.max_bytes` / `logging.backup_count` exposed and whitelisted. Tests: `tests/test_observability.py::TestConfigureLogging::test_rotating_handler_used_by_default`, `test_rotation_actually_rotates_when_size_exceeded`.
- P2.4 — checkpoint metadata stamped with `_harness_schema_version` (current `CHECKPOINT_SCHEMA_VERSION = 1`); `validate_checkpoint_schema` + `CheckpointSchemaMismatchError` enforced by `cmd_resume`. Tests: `tests/test_storage_basic.py::TestCheckpointSchemaVersion`.
- P2.5 — real `--version` / `-V` action wired in `build_parser`; reads `importlib.metadata.version("ai-agent-harness")`. README command table updated.
- P2.6 — CI matrix extended with `macos-latest` + `windows-latest`, advisory via `continue-on-error`.
- P2.7 — `harness/metrics.py` aggregates `<id>.jsonl*` for per-session cost, tokens, error counts, burn rate, projected exhaustion; new `harness metrics` subcommand (`--session-id` / `--all` / `--json` / `--prometheus` / `--output` / `--window-minutes`). Atomic writes into `~/.harness/metrics/` (override globally via `metrics.metrics_dir`). Tests: `tests/test_metrics.py` (23 cases).
- P2.8 — `docs/RUNBOOK.md` covering checkpoint corruption / budget exhaustion / sandbox / lock / LLM silence. Linked from README.
- P2.9 — `cmd_purge --session-id` now also removes the live JSONL log file and rotated backups; failures logged but don't abort.

---

## P0 — Block pilot  *(all fixed 2026-06-10)*

### P0.1  LLM messages persist unencrypted in the checkpoint DB
**Dimension:** security · secret handling
**Where (was):** `harness/storage.py`, checkpoint file `~/.harness/checkpoints.db`
**What was missing:** The harness stored the full LangGraph state (including `state["messages"]`) in SQLite as msgpack blobs. A secret pasted into a prompt landed at rest with no opt-out.
**Fix landed:** `HarnessAsyncSqliteSaver` now overrides `aput` and `aput_writes` to scrub the `messages` channel through `harness.redactor.redact_messages` before delegating to LangGraph's serializer. Opt-out via `persistence.redact_messages: false` (default `true`). Fail-open on redactor errors so a redactor crash never blocks the checkpoint write.
**Regression tests:** `tests/test_storage_basic.py::TestCheckpointMessageRedaction` — three asyncio tests:
- `test_aput_redacts_messages_in_checkpoint` writes an `sk-…` key inside a message and asserts the byte sequence is NOT in the SQLite blob.
- `test_aput_writes_redacts_messages_in_pending_writes` does the same for the per-channel writes table.
- `test_redact_disabled_persists_raw` confirms the opt-out works when the operator wants verbatim transcripts.
**Status:** Fixed

### P0.2  CommandValidator exists but is never wired
**Dimension:** operations · defense-in-depth
**Where (was):** every `SandboxExecutor(...)` site (`harness/graph.py:1198`, `harness/test_generation.py:510`, `harness/speculative.py:254`, `harness/skills.py:229`, `harness/sandbox.py:1473`) defaulted `command_validator=None`. The guard at `sandbox.py:1328` short-circuited validation entirely.
**Fix landed:** Added `set_command_validator()` / `get_command_validator()` in `harness/security.py` (mirrors the redactor's global-scanner pattern). `cmd_run` and `cmd_resume` now call `set_command_validator(create_command_validator_from_config(config))` during session startup. `SandboxExecutor.__init__` falls back to the global default when `command_validator` is not explicitly passed — every call site now picks it up without modification.
**Regression tests:** `tests/test_security_basic.py::TestSandboxExecutorPicksUpGlobalValidator`:
- `test_executor_inherits_global_validator` — confirms the global is picked up.
- `test_executor_explicit_validator_wins_over_global` — confirms an explicit argument overrides.
- `test_executor_no_global_no_explicit_is_none` — confirms safe default behavior when nothing is set.
**Status:** Fixed

### P0.3  No `LICENSE` file at repo root
**Dimension:** operations · license & legal
**Where (was):** repo root had no `LICENSE`; `pyproject.toml:10` carried `license = { text = "MIT" }` only.
**Fix landed:** Added `LICENSE` at repo root with standard MIT text and `Copyright (c) 2026 AI Agent Harness Team`. Switched `pyproject.toml` to `license = { file = "LICENSE" }` so the file ships in the wheel and GitHub auto-detects the license.
**Status:** Fixed

---

## P1 — Must fix before scale  *(all fixed 2026-06-10)*
<!-- See P2 section below for the polish item closeout — all open items shipped in the same sweep; P2.7 is intentionally deferred. -->


### P1.1  Patcher allowlist falls back to permissive when source root is unclear  *(Status: Fixed)*
**Fix landed:** `harness/graph.py::_build_patcher_allowlist` now returns a **conservative** allowlist (`src/`, `lib/`, `app/`, `pkg/`, `cmd/`, `tests/`, `test/`, `__tests__/`, plus `_ROOT_ALLOWLIST_FILES` and any `requirements*.txt`) when the source-root heuristic can't decide. Logged at WARNING when the fallback fires so operators can fix detection.
**Regression tests:** `tests/test_source_root_enforcement.py::TestBuildPatcherAllowlist::test_returns_conservative_fallback_for_flat_workspace`, `TestPatchingNodeAllowlist::test_flat_workspace_conservative_fallback_{blocks_root_writes,allows_src_writes}`.

### P1.2  Patcher writes are vulnerable to symlink races  *(Status: Fixed)*
**Fix landed:** `harness/patcher.py::_awrite` now refuses to write through any path where `os.path.islink(filepath)` is true, and uses `O_NOFOLLOW` belt-and-braces on Linux/macOS to catch races. Documented Windows limitation (no portable O_NOFOLLOW; the islink check still applies).
**Regression tests:** `tests/test_patcher_symlink_guard.py` — three asyncio tests confirming the symlink target stays intact, normal writes still work, and new-file writes are unaffected.

### P1.3  Auto-enable `allow_network=True` on detected pip/npm install  *(Status: Fixed)*
**Fix landed:** Network auto-flip in `harness/graph.py::_apply_toolchain_adaptation` now gated on `sandbox.auto_enable_network_for_install` (default **false**). When the heuristic fires but the opt-in is off, the function declines and logs a WARNING pointing the operator at the config key. The whitelist (`harness/cli.py::_KNOWN_NESTED_KEYS["sandbox"]`) accepts the new key.
**Regression tests:** `tests/test_harness.py::TestToolchainAdaptation::test_auto_network_off_refuses_to_flip` plus updated `test_adapts_network_for_pip_install` exercising the opt-in path.

### P1.4  Budget can be exceeded by a single LLM call  *(Status: Fixed)*
**Fix landed:** New `BudgetTooLowError` raised pre-flight in `harness/gateway.py::Gateway.dispatch` when `(input_chars/4 × input_rate) + (4000 × output_rate) > budget_remaining_usd`. Provider is never contacted on refusal. Calls landing within 20% of the cap emit a WARNING so the operator notices the approach.
**Regression test:** `tests/test_gateway_guards.py::test_preflight_budget_refuses_oversized_call`.

### P1.5  Empty LLM response causes silent 3-cycle wait before HITL  *(Status: Fixed)*
**Fix landed:** Gateway now retries an empty content body up to two extra times after the existing transport-retry loop returns; if still empty, raises a new `EmptyLLMResponseError`. `repair_node` distinguishes this from generic budget exhaustion and sets `node_state["llm_silent"]=True`. `route_after_compiler` short-circuits to HITL immediately on `llm_silent` instead of waiting for the 3-cycle repair cap. Telemetry event `llm_empty_response` emitted on the failure path.
**Regression tests:** `tests/test_gateway_guards.py::test_empty_llm_response_raises_after_retries`, `test_empty_then_recovers_succeeds`.

### P1.6  Checkpoint corruption silently falls back to empty state  *(Status: Fixed)*
**Fix landed:** `harness/storage.py::_deserialize_checkpoint_blob` gained a `strict=True` parameter that raises a new `CheckpointCorruptedError` when every decoder fails. Promoted the unpack-failure log from DEBUG to WARNING, and added a separate WARNING for the JSON-fallback failure. `harness/cli.py::cmd_resume` now pre-flights the latest checkpoint with `strict=True` and surfaces a clear operator message (start fresh / restore backup / purge session). `_doctor_check_checkpoint_db` opens the 5 most recent checkpoints and reports any that fail to deserialize.

### P1.7  Concurrent sessions on the same workspace can clobber each other  *(Status: Fixed)*
**Fix landed:** New `_acquire_workspace_lock` helper in `harness/cli.py` takes an `fcntl.flock` exclusive non-blocking lock on `<workspace>/.harness_session.lock` at `cmd_run` start. Handle pinned in a module-level slot so the OS holds the lock for the process lifetime. New `--force-lock` CLI flag for stuck-lock recovery. Logged WARNING when forced, ERROR + exit when refused.
**Regression tests:** `tests/test_workspace_lock.py` — three tests covering refusal, force, and the unlocked happy path; uses a sibling subprocess so the cross-process flock semantics are actually exercised.

### P1.8  No linting / type-check in CI  *(Status: Fixed)*
**Fix landed:** Added a `quality` job to `.github/workflows/ci.yml` running `ruff check` (blocking), `ruff format --check` (continue-on-error pending one-shot reformat), and `mypy harness/` (continue-on-error pending typing backlog). Mirror added to `.pre-commit-config.yaml` so local commits get the same gate. Repository is currently **ruff-check clean** end-to-end.

### P1.9  No circuit breaker around persistent 429s  *(Status: Fixed)*
**Fix landed:** `Gateway` tracks rate-limit failures in a deque with a 5-minute rolling window and a threshold of 3 hits. `_circuit_is_open()` returns True when the threshold is crossed; `dispatch()` diverts the next call to `force_local=True` (local Ollama). Failures recorded automatically after the retry loop exits on HTTP 429 or 5xx. WARNING logged when the breaker opens, including the cooldown duration.
**Regression test:** `tests/test_gateway_guards.py::test_rate_limit_circuit_breaker_opens_after_threshold`.

### P1.10  Discovery loop has no hard iteration cap  *(Status: Fixed)*
**Fix landed:** Added `max_discovery_iterations: int = 10` to `GatewayConfig`, clamped to [1, 30] at config load. `route_after_discovery` short-circuits to `write_spec_node` with a WARNING when `discovery_question_count >= max_discovery_iterations`. The key is wired through every config layer — `config/config.json`, `config/config.json.example`, `.harness_config.json`, `.harness_config.json.template`, `harness/cli.json`, and the setup-script generator. Whitelisted in `harness/cli.py::_KNOWN_NESTED_KEYS["node_throttle"]`.

---

## P2 — Polish  *(all items fixed 2026-06-10)*

### P2.1  Dependencies are range-pinned, not exact-pinned
**Where (was):** `pyproject.toml:14-28` — every dep is `>=` only (`langgraph>=0.4.0`, `httpx>=0.28.0`, etc.).
**Fix landed:** Added `requirements-prod.txt` at repo root with exact pins of the full transitive closure (snapshot 2026-06-10). Documented `pip install -e . --constraint requirements-prod.txt` as the pilot install command in `README.md`. Constraints file is inert during a dev install (only deps requested by `pip install -e .` are touched) so the dev path stays flexible.
**Status:** Fixed

### P2.2  JSON parser has no total-size cap or recursion limit
**Where (was):** `harness/trust.py::validate_discovery_json` capped per-field text and module count but not total payload size or nesting depth.
**Fix landed:** Added `_MAX_DISCOVERY_BYTES = 1_000_000` pre-flight check (rejects before `json.loads`) and `_MAX_DISCOVERY_DEPTH = 10` post-parse check via a cycle-safe `_json_depth` walker. Both produce explicit errors in the standard `(data, errors)` return tuple — no exceptions thrown.
**Regression tests:** `tests/test_trust.py::TestValidateDiscoveryJson::test_oversized_total_response_rejected`, `test_deeply_nested_response_rejected`, `test_at_depth_limit_accepted`.
**Status:** Fixed

### P2.3  Log rotation is not wired
**Where (was):** `harness/observability.py:187` used `logging.FileHandler`, not `RotatingFileHandler`.
**Fix landed:** `configure_logging` now defaults to `RotatingFileHandler(maxBytes=10_000_000, backupCount=5)`. Two new params `max_bytes` and `backup_count` are wired through `harness/cli.py::cmd_run` reading `logging.max_bytes` / `logging.backup_count`, and whitelisted in `_KNOWN_NESTED_KEYS["logging"]`. Setting `max_bytes=0` falls back to the legacy `FileHandler` so any operator pinning a single non-rotating file (e.g. for an external log shipper) can opt out.
**Regression tests:** `tests/test_observability.py::TestConfigureLogging::test_rotating_handler_used_by_default`, `test_rotation_actually_rotates_when_size_exceeded`, `test_max_bytes_zero_uses_plain_file_handler`.
**Status:** Fixed

### P2.4  No checkpoint schema versioning
**Where (was):** `harness/storage.py` stored msgpack/JSON blobs with no schema-version field. `AgentState` is `TypedDict, total=False`.
**Fix landed:** Introduced `CHECKPOINT_SCHEMA_VERSION = 1` (initial version) plus `MIN_RESUMABLE_SCHEMA_VERSION` and `SCHEMA_VERSION_METADATA_KEY = "_harness_schema_version"`. `HarnessAsyncSqliteSaver.aput` now stamps the version into the metadata dict on every write (separate SQLite column from the channel blob, so it can't disturb LangGraph's own state-restore). New `validate_checkpoint_schema()` helper plus `CheckpointSchemaMismatchError` exception. `cmd_resume` calls the validator right after the corrupted-blob check; legacy (pre-versioning) checkpoints WARN-and-allow, future versions are refused with a clear operator message pointing at upgrade / fresh-start / purge.
**Regression tests:** `tests/test_storage_basic.py::TestCheckpointSchemaVersion::test_aput_stamps_schema_version_in_metadata`, `test_validate_accepts_current_version`, `test_validate_refuses_future_version`, `test_validate_legacy_checkpoint_warns_but_allows`, `test_validate_non_integer_version_rejected`.
**Status:** Fixed

### P2.5  `harness --version` not in README command table
**Where (was):** `README.md` command reference didn't list `--version`. Despite the previous note, argparse did NOT actually auto-provide it — `python -m harness.cli --version` errored.
**Fix landed:** Added a real `--version` / `-V` action to the root parser in `harness/cli.py::build_parser`, wired to a new `_get_harness_version()` helper that reads the installed distribution version via `importlib.metadata`. Listed in the README command table and in the parser's quick-start help block. Falls back to `"(unknown)"` when run from an uninstalled in-tree copy.
**Status:** Fixed

### P2.6  CI matrix is Linux-only despite documented multi-platform support
**Where (was):** `.github/workflows/ci.yml` ran `runs-on: ubuntu-latest` only.
**Fix landed:** The `test` job now uses a parameterised `os` matrix — Linux is the blocking target (Python 3.11/3.12/3.13), with `macos-latest` (py3.12) and `windows-latest` (py3.12) added via `include` and guarded by `continue-on-error: ${{ matrix.os != 'ubuntu-latest' }}`. Advisory results surface platform-specific regressions (fcntl on Windows, path handling on macOS) without gating merges until the platform backlog is triaged.
**Status:** Fixed

### P2.7  No metrics endpoint / budget burn-rate forecast
**Where (was):** `harness/observability.py` emitted structured events but offered no aggregation surface — operators couldn't see cumulative cost without a hand-rolled jq pipeline, and there was no "at current burn, budget exhausted in X minutes" projection.
**Fix landed:** New `harness/metrics.py` reads `<id>.jsonl` + `<id>.jsonl.*` rotated backups, sums `llm_call` cost/tokens, counts failure events (`llm_empty_response`, `llm_circuit_open`, `token_budget_exhausted`, `sandbox_start_failed`, `hitl_gate_blocked`), computes a sliding-window burn rate, and projects exhaustion against `token_budget.hard_cap_usd`. New `harness metrics` subcommand wires it through the CLI (`--session-id`, `--all`, `--json`, `--prometheus`, `--output`, `--window-minutes`). Machine-readable outputs land in `~/.harness/metrics/` by default (configurable globally via `metrics.metrics_dir` in `~/.harness/config.json`) and are written atomically (`<dest>.tmp` → `os.replace`) so node_exporter textfile collectors never see a half-written file. No HTTP daemon — a cron job emitting `--prometheus` is enough for pilot scale.
**Regression tests:** `tests/test_metrics.py` — 23 cases covering aggregation across rotated backups, malformed-line tolerance, burn-rate window math, projection edge cases (zero burn, already exhausted), human/table/Prometheus formatting, atomic write guarantees, config-override resolution, and CLI smoke (stdout, file write, `--output -`, no-logs exit code).
**Status:** Fixed

### P2.8  No runbook for common operator failures
**Where (was):** `docs/` had installation, architecture, and spec docs but no `docs/RUNBOOK.md`.
**Fix landed:** Added `docs/RUNBOOK.md` covering the top five failure modes (checkpoint corrupted, budget exhausted mid-session, sandbox can't start, workspace lock refused, persistent LLM silence) with symptom / diagnostic command / fix recipe per entry plus a one-liners appendix and escalation path. Linked from the README troubleshooting section. Content cross-references `harness doctor` so the runbook stays the layer-2 escalation after the layer-1 healthchecks.
**Status:** Fixed

### P2.9  No `harness purge --session-id` for selective deletion
**Where (was):** `harness purge` already accepted `--session-id` and called `adelete_thread` — but only on the checkpoint DB. The per-session JSONL transcript at `~/.harness/logs/<id>.jsonl` (plus any rotated `.jsonl.1`, `.jsonl.2` backups) was left behind, defeating the GDPR-deletion intent.
**Fix landed:** `cmd_purge` now follows the checkpoint delete with a best-effort cleanup of the live log file and every rotated backup matching `<id>.jsonl*`. Failures to remove individual files are logged at WARNING and counted out — they don't abort the purge. Removal count is printed so the operator can confirm coverage.
**Status:** Fixed

---

## Out of scope

This audit deliberately did NOT cover:

- **GA / multi-tenant SaaS** — the harness is single-user, single-host today. The whole "user-global config holds API keys" design is incompatible with multi-tenancy. A separate audit is needed if that direction is pursued.
- **Regulated industries** (HIPAA, PCI, FedRAMP) — checkpoint encryption, audit logging, key-management integration would all need separate work.
- **Generated-app quality** — whether the apps the harness produces are themselves production-ready was treated as the operator's concern.
- **Open-source release** — install UX, security defaults, and license clarity weren't sized against a public-repo bar.
- **Adversarial LLM behavior** — the audit assumed honest-but-fallible providers; it didn't enumerate prompt-injection or jailbreak surfaces specific to the workflow.

---

## Re-assessment trigger

Re-run this audit when any of these happens:

- Multi-tenant or multi-user support is added.
- The checkpoint format or `AgentState` schema changes materially.
- A new sandbox backend lands or `allow_network` semantics change.
- The harness is exposed as a service (HTTP, MCP, RPC) rather than a CLI.
- A new LLM provider or local-inference path is wired into the gateway.
- A security incident — actual or near-miss — uncovers a class of issue not on this list.

When re-running, mark each finding's `Status:` as `Fixed` (with the commit hash that closed it) rather than deleting, so the file is also a record of what got hardened.
