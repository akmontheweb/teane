# AI Agent Harness — Requirements Specification

*Refreshed from current codebase state. Companion to `SPEC_ARCHITECTURE.md`.*

---

## 1. Executive Summary

AI Agent Harness is a production-grade, model-agnostic autonomous coding agent built on LangGraph. It accepts natural language engineering tasks (greenfield) OR a folder of `change_requests/*.txt` files (brownfield), generates precise code patches via LLMs, verifies them through sandboxed builds, and OPTIONALLY brings the app up locally as a docker-compose dev environment (gated by `--dev-deployment`; off by default so operators can take the generated code to their own deployment pipeline). It runs under budget guardrails, security scanning, and git lifecycle management. The system supports exhaustive multi-phase discovery (requirements → architecture → deployment) with per-question Enter-to-accept defaults and an optional org-wide `deployment.json` policy, one-shot reverse-engineering of `SPEC_ARCHITECTURE.md` on first contact with brownfield repos, human-in-the-loop intervention points, checkpoint-based crash recovery, cross-model speculative repair escalation, and stack-aware multi-language workflows across Python / Java / Node / Go / Rust / Dart / Flutter — all built on a single kitchen-sink builder image so polyglot workspaces share one container.

---

## 2. Functional Requirements (FR)

### FR-001: CLI Subcommand Routing
- **Description:** The system MUST provide a `harness` CLI with subcommands `run`, `resume`, `status`, `doctor`, `purge`, and `metrics`, each with their own argument parsers and help text. The root parser MUST also accept a `--version` / `-V` flag that prints the installed package version (resolved via `importlib.metadata.version("ai-agent-harness")`) and exits.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given `harness -h`, the system displays help with all six subcommands listed.
  - Given `harness run -h`, the system displays run-specific help with all flags documented.
  - Given `harness --version`, the system prints `harness <X.Y.Z>` and exits 0; the version falls back to `(unknown)` for uninstalled in-tree runs.

### FR-002: Workspace-Bound Execution
- **Description:** The system MUST accept a `--workspace` / `-r` flag pointing to an existing directory. All generated code, specs, and deployment artifacts MUST land inside this workspace.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given a valid `--workspace` path, `SPEC_REQUIREMENTS.md` is written to `{workspace}/docs/`.
  - Given an invalid or missing workspace path, the system exits with error code 1.

### FR-003: Model-Agnostic LLM Gateway
- **Description:** The system MUST support multiple LLM providers (DeepSeek, Anthropic Claude, OpenAI, Ollama) through a unified `Gateway` interface. Provider selection is per-`NodeRole` (planning, patching, repair).
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given a DeepSeek API key in config, calls to `/chat/completions` succeed.
  - Given an Anthropic API key, calls to `/messages` succeed with system prompt extraction.
  - Given no API keys, Ollama is auto-detected as a zero-cost local fallback.

### FR-004: Hierarchical Configuration Discovery
- **Description:** Configuration MUST be loaded in priority order: workspace `.harness_config.json` → `~/.harness/config.json` → shipped `cli.json` fallback. Nested dicts MUST be deep-merged.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given a workspace with `.harness_config.json` overriding `token_budget.hard_cap_usd`, the override takes effect.
  - Given no workspace config, `~/.harness/config.json` values are used.
  - Given neither, `harness/cli.json` hardcoded defaults are used.

### FR-005: Code Patch Generation and Application
- **Description:** The system MUST generate code patches in a strict SEARCH/REPLACE block syntax and apply them to workspace files via a hybrid patcher (AST-aware + text fallback).
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given an LLM response containing `<<<REPLACE_BLOCK>>>` blocks, the patcher locates and replaces the target text.
  - Given an LLM response containing `<<<CREATE_FILE>>>` blocks, the patcher creates the specified file.
  - Given a REPLACE_BLOCK where the SEARCH text doesn't match, the patcher logs a failure.

### FR-006: Sandboxed Build Verification
- **Description:** The system MUST execute the project's build command inside an isolated sandbox. Auto-detect priority is Docker → unshare (Linux namespaces) → bare (opt-in via `HARNESS_ALLOW_UNSAFE_SANDBOX=true`). Build output MUST be parsed for structured diagnostics.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given `build_command: "make build"`, the command runs inside a sandbox and returns exit code + diagnostics.
  - Given a compilation error in Rust / GCC-Clang / Go / Python / Java / TypeScript / Dart / generic format, structured `DiagnosticObject` dicts are extracted.
  - Given a timeout of 300 seconds, builds exceeding the limit are killed with PGID-based process group termination.
  - Given no available backend and no opt-in, the harness raises `RuntimeError` and emits a `sandbox_start_failed` event.

### FR-007: Repair Loop with Budget Guardrail
- **Description:** On build failure, the system MUST route to a repair node that analyzes compiler diagnostics and generates fix patches. After 3 failed repair attempts, the system MUST escalate to human intervention. If the budget ($2.00 default) is exhausted, execution MUST stop.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given a build failure and < 3 prior repair attempts, repair_node is invoked.
  - Given 3 failed repair attempts, human_intervention_node is triggered.
  - Given `budget_remaining_usd <= 0`, all LLM calls are refused.

### FR-008: Cross-Model Repair Escalation
- **Description:** Repair attempts 1-2 MUST use the cheap primary model. Repair attempt 3 MUST escalate to the expensive fallback reasoning model.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given a configured `repair_fallback` model and 2 prior failures, the 3rd repair attempt uses the fallback model.
  - Given no fallback model configured, the primary model is reused for all attempts.

### FR-009: Human-in-the-Loop Intervention
- **Description:** When the repair limit is hit or budget is exhausted, the system MUST present an interactive HITL menu with options: view diffs, resume, inject hint, pause for manual edits, increase budget, save and quit (resumable), or abandon with git rollback. The transport is pluggable via `harness/hitl.py` (`StdinChannel`, `FileChannel`, `HttpChannel`).
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given HITL triggered, a menu with [v/r/e/m/b/s/q] options is displayed.
  - Given user selects [b] (increase budget), `budget_remaining_usd` increases by $2.00 and the menu re-displays.
  - Given user selects [s] (save & quit), the session is checkpointed and the developer is shown the exact `harness resume --session-id` command.
  - Given user selects [q] and confirms, `git checkout -- .` is executed, the session ends, and a `hitl_gate_blocked` event is emitted.

### FR-010: Secret Redaction Before API Calls
- **Description:** All outbound LLM messages MUST pass through a `SecretScanner` that detects and redacts API keys, tokens, JWT secrets, and high-entropy strings before transmission.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given a message containing `sk-...` (OpenAI key format), it is replaced with `[REDACTED:sha256:xxxx]`.
  - Given a message containing `ghp_...` (GitHub token), it is replaced.
  - Given a message with no secrets, it passes through unchanged.

### FR-011: Git Lifecycle Management
- **Description:** Every harness session MUST create an isolated `agent/patch-{session_id[:8]}` branch. On build success, changes are committed. On failure, the branch is deleted and the working tree is rolled back.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given a clean workspace, `git stash` is skipped and the patch branch is created from HEAD.
  - Given a dirty workspace, changes are stashed before branch creation and popped after.
  - Given build success, changes are committed to the patch branch and the original branch is restored.

### FR-012: Exhaustive Discovery Pipeline
- **Description:** Before code generation, the system MUST run a multi-phase discovery pipeline (requirements → architecture → deployment) where the LLM cross-examines the developer across structured sectors, with interactive question/answer loops and critical-unknown tracking.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `--discover` IS set (opt-in), requirements discovery runs with an 8-sector cross-examination prompt. Discovery is skipped by default.
  - Given all discovery questions are answered, `discovery_complete` is set to true and the spec is written.
  - Given critical questions remain unanswered and the user types DONE, the loop refuses to exit.

### FR-013: Spec File Generation from Discovery
- **Description:** The discovery pipeline MUST serialize interview Q&A into `SPEC_REQUIREMENTS.md`, `SPEC_ARCHITECTURE.md`, and `DEPLOYMENT_BLUEPRINT.md` in `{workspace}/docs/`.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given requirements discovery completes, `SPEC_REQUIREMENTS.md` is written with all Q&A compiled.
  - Given architecture discovery completes, `SPEC_ARCHITECTURE.md` is written.
  - Given deployment discovery completes, `DEPLOYMENT_BLUEPRINT.md` is written.

### FR-014: Pre-Flight Manifest Refinement
- **Description:** The system MUST support a `--manifest` flag pointing to a raw notes file. The LLM synthesizes these notes into a structured `SPEC_REQUIREMENTS.md`, presents an interactive review loop (approve/refine/manual), and injects the approved spec as the system prompt. When a pre-flight spec is approved, the graph's discovery pipeline MUST be skipped to prevent overwriting the approved spec.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `--manifest notes.txt`, the LLM synthesizes a structured spec.
  - Given the user approves the spec, `skip_discovery` is auto-set to True in the graph.
  - Given the user refines the spec, additional LLM calls update the document.

### FR-015: Workspace Requirements Manifest Auto-Discovery
- **Description:** If `--manifest` is not provided, the system MUST auto-detect `product_spec.txt` (configurable via `manifest_file` key) in the workspace root.
- **Priority:** Could Have
- **Acceptance Criteria:**
  - Given `product_spec.txt` exists in the workspace root, it is used as the manifest without explicit `--manifest` flag.

### FR-016: Checkpoint Persistence and Crash Recovery
- **Description:** The system MUST persist graph state to a SQLite database (WAL mode) at every node transition. `harness resume --session-id` MUST restore and continue from the last checkpoint. Each checkpoint's metadata MUST carry a `_harness_schema_version` stamp (current `CHECKPOINT_SCHEMA_VERSION = 1`); `cmd_resume` MUST pre-flight the most recent blob with strict deserialization and refuse to load on `CheckpointCorruptedError` or `CheckpointSchemaMismatchError`. The `messages` channel MUST be redacted through `harness.redactor` before serialization (opt-out via `persistence.redact_messages: false`, default `true`).
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given a running graph, checkpoints are written to `~/.harness/checkpoints.db` with the schema version stamped in metadata.
  - Given `harness resume --session-id <id>`, the graph resumes from the checkpointed state after the pre-flight check passes.
  - Given a corrupted checkpoint blob, `cmd_resume` exits with an operator-readable message offering fresh-start / restore-backup / purge-session options.
  - Given a checkpoint stamped with a future schema version, `cmd_resume` refuses with an upgrade-or-purge message.
  - Given a non-existent session ID, resume exits with error code 1.
  - Given a prompt containing an API-key-shaped secret, the byte sequence is absent from the on-disk SQLite checkpoint blob.

### FR-017: Read-Only Status Inspection
- **Description:** `harness status --all` MUST list all checkpointed sessions with session ID, created time, updated time, and workspace path. `harness status --session-id <id>` MUST display a full state snapshot.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given `harness status --all`, a table with SESSION ID, UPDATED, CREATED, and WORKSPACE columns is printed.
  - Given `harness status --session-id <id>`, a detailed state dump with all fields is printed.
  - Given a non-existent session ID, a "not found" message is printed.

### FR-018: Session Data Purging
- **Description:** `harness purge --all` MUST delete all checkpoint data after confirmation. `harness purge --session-id <id>` MUST delete that session's checkpoints AND its per-session JSONL log file (`<id>.jsonl`) plus any rotated backups (`<id>.jsonl.*`). Log-file removal is best-effort: a single OS error MUST log a WARNING and continue rather than abort the purge.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `harness purge --all` and user confirms "yes", all rows in the checkpoints DB are deleted.
  - Given `harness purge --session-id <id>`, only that thread's checkpoints are deleted and the count of removed log files is printed.
  - Given a session whose log file cannot be removed (permissions, race), the checkpoint deletion still completes and the failure is logged at WARNING.

### FR-019: Lint Gate (Deterministic Format Verification)
- **Description:** Before each build, modified files MUST be auto-formatted and linted using language-specific tools. Lintgate ships specs for `.py` / `.pyi` (ruff), `.go` (gofmt), `.rs` (rustfmt + clippy), `.ts` / `.tsx` / `.js` / `.jsx` / `.css` / `.html` / `.json` / `.yaml` / `.yml` / `.md` (prettier), `.c` / `.h` / `.cpp` / `.cc` / `.cxx` / `.hpp` (clang-format), `.java` (google-java-format), `.dart` (`dart format`), `.sh` / `.bash` (shfmt), and `.sql` (sqlfluff). Lint errors are surfaced in the build output. By default, formatting only runs on files actually patched this session (`lintgate.format_modified_files=false`); linters run on all modified files.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given modified `.py` files, ruff format + ruff check are executed.
  - Given modified `.dart` files, `dart format` runs.
  - Given no matching formatter for a file extension, it is skipped.

### FR-020: Multi-Variant Speculative Execution
- **Description:** After patching, the system MAY generate N parallel code variants, compile each in isolated git worktrees, and select the winner by first_success, fewest_changes, or all_pass strategy.
- **Priority:** Could Have
- **Acceptance Criteria:**
  - Given `speculative.enabled: true` in config, N variants are generated in parallel.
  - Given one variant compiles successfully and others fail, the successful variant is selected.
  - Given all variants fail, the system falls back to the original patching flow.

### FR-021: Container Deployment
- **Description:** After a successful build and clean security scan AND when the operator has opted in via `--dev-deployment` (see FR-044), the system MUST scan workspace telemetry, synthesize a deployment architecture blueprint, generate Dockerfiles + docker-compose.yml + Caddyfile, build containers, and run health checks. Without `--dev-deployment` the graph ends at the clean-scan boundary and the workspace is handed back with the generated code in place but no Docker artifacts.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `--dev-deployment` and a Python workspace with `requirements.txt`, a Python Dockerfile is generated.
  - Given the deployment blueprint, `docker compose up --build -d` is executed (Compose V2 syntax, no hyphen).
  - Given containers are running, health check polling confirms readiness within 30s.
  - Given `--dev-deployment` is NOT set, no Dockerfile / compose / `docker compose up` is produced and the graph routes to END after the security scan.

### FR-022: Security Scanning Gate
- **Description:** After successful build, the system MUST run gitleaks (secret detection) and bandit/semgrep (SAST) on the workspace. Findings route to patching for fix, with a limit of 2 security fix attempts.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given a workspace with hardcoded secrets, gitleaks detects them.
  - Given security findings and < 2 prior security fix attempts, the system routes to patching_node.
  - Given 2 failed security fix attempts, HITL is triggered.

### FR-023: Memory Cleanse on Success
- **Description:** When a build succeeds after repair loops, the system MUST compress verbose intermediate repair messages into a single structured summary to keep the conversation history compact for prefix caching.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given a successful build after 2 repair attempts, messages are cleansed to 4 entries (system prompt + planning message + final patch + compression summary).

### FR-024: Impact Analysis
- **Description:** Before applying patches, the system MAY analyze the dependency graph of the workspace to warn about files that may be impacted by the proposed changes.
- **Priority:** Could Have
- **Acceptance Criteria:**
  - Given a Python workspace with cross-file imports, the dependency graph is built.
  - Given a patch to a file with 3 downstream dependents, those dependents are listed in the impact result.

### FR-025: First-Run Healthcheck (`harness doctor`)
- **Description:** The CLI MUST expose `harness doctor`, which runs six healthchecks and reports each as PASS / WARN / FAIL with a colored marker (suppressed when stdout is not a TTY or `NO_COLOR` is set): git repo presence, global config presence, API keys per configured `model_routing` provider, sandbox backend reachability, checkpoint DB writability and corruption scan over the 5 most recent rows, and config parse cleanliness (re-running `discover_config` + `_validate_config_keys`). The api-keys check MUST consider a provider satisfied when EITHER the `{PROVIDER}_API_KEY` env var OR the `models["<provider>:<model>"].api_key` config field is set (matching the runtime resolution in `gateway.BaseProviderClient.__init__`); the PASS message MUST report the source (`(env)` vs `(config)`) per model so operators see which key the runtime would actually use. The api-keys check MUST also issue a one-token chat call against each provider in parallel to confirm the resolved key actually authenticates against the configured model; HTTP-status-specific FAIL messages distinguish key-rejected (401), no-model-access (403), model-not-found (404), rate-limited (429), provider error (5xx), and network failures. Set `HARNESS_DOCTOR_SKIP_LIVE=true` to skip the live ping (CI / headless / outbound-network-blocked environments) — the doctor then reports presence and source only.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given a healthy install, `harness doctor` exits 0.
  - Given an API key only in the `models["<key>"].api_key` config field (no env var), the `api keys` check reports PASS with `(config)` next to the model id (after a successful live ping).
  - Given an env var AND a config field both set, the PASS message reports `(env)` (env wins precedence, matching the runtime).
  - Given neither env var nor config field set for a routed non-Ollama provider, the `api keys` check reports FAIL with a message naming BOTH the env var to set AND the `models."<key>".api_key` path; the command exits non-zero.
  - Given a configured key that returns HTTP 401, the live ping reports FAIL with `HTTP 401 — API key rejected` and the command exits non-zero.
  - Given `HARNESS_DOCTOR_SKIP_LIVE=true`, no outbound HTTP request fires and the PASS detail notes `(live ping skipped via HARNESS_DOCTOR_SKIP_LIVE)`.
  - Given a corrupted checkpoint blob among the 5 most recent rows, the `checkpoint db` check reports FAIL with the row identifier.
  - Given a typoed nested config key, the `config parse` check reports WARN with the fuzzy-match suggestion.

### FR-026: Multi-Stack Tree-Sitter Coverage
- **Description:** The patcher, impact analyzer, and diagnostic parsers MUST cover Python, Java, JavaScript/TypeScript, Dart (Flutter), Rust, Go, and C/C++ uniformly. Grammars MUST come from a single bundled wheel (`tree-sitter-language-pack`) to avoid the dependency churn of upgrading six individual grammar packages.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given a `.dart` file, `DartParser` extracts diagnostics in Dart's compiler format.
  - Given a Java compilation error, `JavaParser` parses it; given TypeScript, `TypeScriptParser`.
  - Given an unknown extension, parsing falls back to `GenericParser` (regex on `file:line:col: severity: message`).

### FR-027: Stack-Aware Skill Filtering
- **Description:** Skill files in `harness/skills/` MAY declare an `applies_to: [tag1, tag2]` YAML frontmatter. At graph assembly, the workspace is fingerprinted into a tag set; skill files with a non-overlapping `applies_to` set MUST be excluded from the LLM prompt. Skill files with no frontmatter MUST always load (universal skills).
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given a Flutter workspace (tags include `flutter`, `dart`), `flutter.md` loads and `python_fastapi.md` does not.
  - Given a workspace tag set that doesn't intersect any `applies_to` declaration, only frontmatter-free skills load.

### FR-028: Flutter / Mobile Routing Short-Circuit
- **Description:** Flutter projects don't fit the docker-compose-up deploy model (the artifact is a mobile binary). On a clean security scan, if the workspace is detected as a Flutter project, the graph MUST route directly to END instead of through the deployment pipeline.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given a workspace with `pubspec.yaml` declaring a Flutter SDK dep and a clean security scan, the deploy pipeline is skipped and the graph terminates.

### FR-029: Structured Failure-Event Catalogue
- **Description:** Failure sites MUST emit structured events via `harness.observability.log_failure(name, **fields)` (ERROR-level mirror of the existing `emit_event` helper). Each event MUST carry a snake_case `event` field so failures are grep-able from the per-session JSONL log by name instead of by string fragment. Initial catalogue: `sandbox_start_failed`, `token_budget_exhausted`, `hitl_gate_blocked`.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given the gateway refuses dispatch because `budget_remaining_usd <= 0`, a `token_budget_exhausted` event is emitted with `hard_cap_usd` and `role`.
  - Given a JSONL session log, `jq 'select(.event == "sandbox_start_failed")'` returns all sandbox-bootstrap failures.

### FR-030: Recursive Config Typo Detection
- **Description:** `_validate_config_keys` MUST warn on unknown top-level config keys (e.g., `model_routin`) AND on unknown nested keys (e.g., `token_budget.hrad_cap_usd`) with fuzzy-match suggestions. The check covers `sandbox`, `token_budget`, `node_throttle`, `persistence`, `model_routing`, `deployment`, `lintgate`, and `logging`. Keys starting with `_` (comment keys) MUST be skipped.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `{"token_budget": {"hrad_cap_usd": 1.0}}`, the validator emits a WARNING containing `Unknown config key 'token_budget.hrad_cap_usd'` and `did you mean 'hard_cap_usd'?`.
  - Given a clean config, no `Unknown config key` warnings are emitted.

### FR-031: Continuous Integration Gate
- **Description:** Every push to `main` and every pull request MUST trigger a GitHub Actions workflow that runs the full pytest pack on Python 3.11 / 3.12 / 3.13 on `ubuntu-latest` (blocking) plus `macos-latest` and `windows-latest` on Python 3.12 (advisory via `continue-on-error: true`). A separate `quality` job MUST run `ruff check` (blocking gate) plus `ruff format --check` and `mypy harness/` (advisory until typing/format backlog clears). The workflow MUST set `CI=true` and `HARNESS_AUTO_APPROVE=true` so HITL gates auto-approve in headless mode. A failing blocking job MUST block merge; advisory failures MUST NOT.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given a PR that breaks any test, the `pytest (py3.11, ubuntu-latest)` / `(py3.12, ubuntu-latest)` / `(py3.13, ubuntu-latest)` job(s) fail and block merge.
  - Given a ruff-check violation in `harness/` or `tests/`, the `quality` job fails and blocks merge.
  - Given a Linux-only regression that breaks the macOS or Windows run, the `pytest` job for that OS reports failure but merge is NOT blocked (advisory).
  - Given a green pytest + ruff-check run on all blocking targets, the workflow reports `success`.

### FR-032: Cost-Metrics Aggregation (`harness metrics`)
- **Description:** The CLI MUST expose `harness metrics`, which reads `<id>.jsonl` plus rotated backups (`<id>.jsonl.*`) under `logging.log_dir`, aggregates `llm_call` cost / tokens, counts tracked failure events (`token_budget_exhausted`, `llm_empty_response`, `llm_circuit_open`, `sandbox_start_failed`, `hitl_gate_blocked`), computes a trailing-window burn-rate in USD/min, and projects exhaustion against `token_budget.hard_cap_usd`. Flags: `--session-id`, `--all`, `--json`, `--prometheus`, `--output` (path or `-` for stdout), `--window-minutes`. Human-readable output goes to stdout; machine-readable outputs (`--json` / `--prometheus`) write atomically (`<dest>.tmp` → `os.replace`) into `metrics.metrics_dir` (default `~/.harness/metrics/`).
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given a session log with three `llm_call` records (cost $0.10, $0.20, $0.05), `harness metrics --session-id <id>` prints `Total cost: $0.3500` and a non-zero burn rate.
  - Given `--prometheus` and no `--output`, the file `~/.harness/metrics/<id>.prom` is written atomically with `# HELP` / `# TYPE` headers for every documented metric.
  - Given `--output -`, the payload is streamed to stdout and no file is written.
  - Given an empty log directory, `--all` exits with code 1.

### FR-033: Checkpoint Message Redaction
- **Description:** `HarnessAsyncSqliteSaver.aput` and `aput_writes` MUST scrub the `messages` channel through `harness.redactor.redact_messages` before delegating to LangGraph's serializer. Opt-out via `persistence.redact_messages: false` (default `true`). Redactor crashes MUST fail open (log a WARNING and persist the original value) so the checkpoint write itself can never be blocked by a redactor bug.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given a prompt containing an `sk-…`-shaped API key and `redact_messages: true`, the raw key bytes are absent from the SQLite blob.
  - Given `redact_messages: false`, the raw bytes ARE present (opt-out is honoured).
  - Given a forced redactor exception, the WARNING is logged and the checkpoint write succeeds.

### FR-034: Process-Wide Command Validator
- **Description:** `cmd_run` and `cmd_resume` MUST instantiate a `CommandValidator` via `create_command_validator_from_config(config)` and register it process-wide using `set_command_validator()`. `SandboxExecutor.__init__` MUST fall back to the global default when `command_validator=None` is passed. Explicit constructor arguments override the global.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given a session whose sandbox executor is created without an explicit validator, the global instance is used.
  - Given an explicit validator argument, it overrides the global.
  - Given no global and no explicit value, the executor's `command_validator` is `None`.

### FR-035: Pre-Flight Budget Refusal
- **Description:** Before issuing any LLM call, the gateway MUST estimate cost as `(input_chars / 4) × input_rate + (4000 × output_rate)`. If the estimate exceeds `budget_remaining_usd`, the gateway MUST raise `BudgetTooLowError` without contacting the provider. A WARNING MUST be emitted when a call lands within 20% of the cap.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given a 100k-character prompt against a $0.01 remaining budget, `BudgetTooLowError` is raised and no HTTP request is made.
  - Given a call estimated at 85% of remaining budget, a WARNING is logged before dispatch.

### FR-036: Empty-LLM-Response Handling
- **Description:** When a provider returns an empty content body, the gateway MUST retry up to two additional times after the existing transport-retry loop. If still empty, the gateway MUST raise `EmptyLLMResponseError` and `emit_event("llm_empty_response", ...)`. `repair_node` MUST set `node_state["llm_silent"] = True` on this exception; `route_after_compiler` MUST short-circuit to HITL immediately on `llm_silent=True` rather than waiting for the 3-cycle repair cap.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given three consecutive empty responses, `EmptyLLMResponseError` is raised and the `llm_empty_response` event is recorded.
  - Given an empty-then-successful sequence, the call returns the second content without raising.

### FR-037: Rate-Limit Circuit Breaker
- **Description:** The gateway MUST track HTTP 429 / 5xx failures in a 5-minute sliding window. When the count reaches 3, `_circuit_is_open()` MUST return True and the next `dispatch()` MUST force `force_local=True` (Ollama). A WARNING with the cooldown duration MUST be logged when the breaker opens.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given 3 rate-limit failures within 5 minutes, the next call is routed to the local Ollama backend.
  - Given the window expiring, subsequent failures restart the count.

### FR-038: Workspace Single-Writer Lock
- **Description:** `cmd_run` MUST acquire an `fcntl.flock(LOCK_EX | LOCK_NB)` on `<workspace>/.harness_session.lock` at startup. On `BlockingIOError` (another session holds the lock), the CLI MUST exit 1 unless `--force-lock` is passed; with `--force-lock`, a WARNING MUST be logged and the lock acquired. The handle MUST be pinned in a module-level slot so the OS holds the lock for the process lifetime. Platforms without `fcntl` (native Windows) MUST log a DEBUG message and skip locking.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given two `harness run` invocations against the same workspace, the second exits with a clear "lock held by PID X" message.
  - Given `--force-lock`, the second invocation proceeds after logging a WARNING.
  - Given a Windows native run, lock acquisition is skipped without error.

### FR-039: Discovery JSON Trust Guards
- **Description:** `trust.validate_discovery_json` MUST reject responses larger than 1 MB (UTF-8 byte length) before invoking `json.loads`, and MUST reject parsed trees deeper than 10 levels (cycle-safe depth walk). The existing per-question text cap (10,000 chars) and module-count cap (50) MUST remain in effect.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given a 2 MB payload, the validator returns an error containing `"exceeds 1000000 bytes"` without attempting to parse.
  - Given a depth-15 nested object, the validator returns an error containing `"nesting depth"`.
  - Given a normal depth-4 discovery response, no depth or size error is reported.

### FR-040: Log Rotation
- **Description:** `configure_logging` MUST install a `RotatingFileHandler` for the per-session JSONL by default (max 10 MB, 5 backups), configurable via `logging.max_bytes` and `logging.backup_count`. Setting `max_bytes: 0` MUST opt out and fall back to a plain `FileHandler` for operators pinning a single non-rotating file (e.g. for an external log shipper).
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given default config, the live JSONL is a `RotatingFileHandler` with `maxBytes=10_000_000`, `backupCount=5`.
  - Given enough writes to exceed `maxBytes`, the live file plus `.1` backup co-exist.
  - Given `max_bytes: 0`, the handler is a plain `FileHandler` (no rotation).

### FR-041: Patcher Symlink Guard and Conservative Allowlist
- **Description:** The async writer in `harness/patcher.py` MUST refuse to write through any path where `os.path.islink(target)` is true and MUST use `O_NOFOLLOW` on Linux/macOS to catch races. When the source-root heuristic in `harness/graph.py::_build_patcher_allowlist` cannot decide on a project layout, the function MUST fall back to a conservative allowlist (`src/`, `lib/`, `app/`, `pkg/`, `cmd/`, `tests/`, `test/`, `__tests__/`, `_ROOT_ALLOWLIST_FILES`, and any `requirements*.txt`) and log a WARNING so the operator can fix detection. Windows native has no portable `O_NOFOLLOW`; the `islink` check still applies.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given a write target that is a symlink, the patcher logs a WARNING and refuses, leaving the symlink target intact.
  - Given a flat workspace with no recognised source root, only paths under the conservative allowlist are writable; writes to the workspace root are refused.

### FR-042: Sandbox Network Auto-Enable Opt-In
- **Description:** The harness MUST NOT automatically enable `allow_network=True` on detected pip/npm install commands unless `sandbox.auto_enable_network_for_install: true` is set. When detection fires with the opt-in off, the function MUST log a WARNING pointing the operator at the config key. The opt-in MUST be whitelisted in the `sandbox` section of `_KNOWN_NESTED_KEYS`.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `build_command: "pip install -e ."` and `auto_enable_network_for_install: false` (default), the sandbox does NOT auto-enable network.
  - Given the same build command and the opt-in `true`, network IS auto-enabled.

### FR-043: Hard Cap on Discovery Loop
- **Description:** `node_throttle.max_discovery_iterations` (default 10, clamped to `[1, 30]` at config load) MUST hard-cap the number of discovery loop iterations. `route_after_discovery` MUST short-circuit to `write_spec_node` with a WARNING when `discovery_question_count >= max_discovery_iterations`. The key MUST appear in every config layer (cli.json, config.json, .harness_config.json, templates) and in the `node_throttle` whitelist.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `max_discovery_iterations: 3` and a fourth discovery question, the graph routes to `write_spec_node` instead of issuing the LLM call.
  - Given a value outside `[1, 30]`, it is clamped at load and logged.

### FR-044: Opt-In Deployment Phase (`--dev-deployment`)
- **Description:** The deployment phase (deployment discovery → `DEPLOYMENT_BLUEPRINT.md` → gatekeeper approval → `docker compose up`) MUST be off by default. `harness run` MUST accept `--dev-deployment` / `--dev_deployment` (`action="store_true"`) on `run_parser` and thread it through `run_graph(dev_deployment=...)` into `AgentState["dev_deployment"]`. `route_after_security_scan` MUST consult the flag: with a clean scan and `dev_deployment=False`, the router MUST return `"__end__"`; with `dev_deployment=True` it MUST return `"deployment_discovery_node"`. The Flutter short-circuit (FR-028) MUST run before the flag check so mobile builds end regardless of the flag. The existing `deployment.enabled` config switch is a NARROWER gate that only short-circuits the docker step inside `deployment_node` once the phase is already running.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given `harness run` with no `--dev-deployment`, after a clean security scan the run ends with no Dockerfile / compose / containers produced and `[cli] Code generated at <path>. Deployment phase skipped.` is logged.
  - Given `harness run --dev-deployment` and a clean security scan, the router enters `deployment_discovery_node`.
  - Given a Flutter project with `--dev-deployment`, the run still ends at the Flutter short-circuit (mobile build, no docker-compose).
  - Given `--dev-deployment` AND `deployment.enabled: false` in config, the phase enters discovery and writes `DEPLOYMENT_BLUEPRINT.md`, but `deployment_node` skips the docker step with `{"skipped": True, "reason": "disabled"}`.

### FR-045: Change-Request Folder Mode
- **Description:** The harness MUST support a `change_requests/` folder at the workspace root containing one or more `.txt` files, each a self-contained ask. `cmd_run` MUST detect the folder (or be told via the wizard) and route through `ingest_change_requests_node` instead of the bare-prompt path. The ingest node MUST (1) walk only the top-level `.txt` files, skipping `applied/`; (2) assign monotonic `CR-N` IDs starting at `max(applied/**/CR-*.txt) + 1`; (3) respect operator-supplied IDs in filenames matching `CR-<N>-<rest>.txt`, aborting on collisions with archived IDs; (4) concatenate file contents under `# === CR-N: <relative-path> ===` separators and inject the result as the first user message. At session end, consumed files MUST be moved into `change_requests/applied/<session-id>/` with a `manifest.json` recording the status (`success` / `cancelled` / `failed-build`). When both `-p "..."` and a populated folder are supplied, the folder wins and the prompt is dropped with a WARNING.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given an empty `change_requests/` folder under `--new_build=false`, the CLI exits with a clear error directing the operator to add at least one `.txt` file.
  - Given files `feature-x.txt` + `CR-12-bugfix.txt` and prior archive `applied/abcd/CR-3-old.txt`, the new IDs are CR-4 (feature-x) and CR-12 (bugfix); a collision with CR-3 aborts.
  - Given a successful run, the consumed `.txt` files land under `change_requests/applied/<session-id>/` with `manifest.json` recording `status: "success"`.
  - Given `CR-7` is assigned, the LLM's first user message references it inside a `# === CR-7: feature-x.txt ===` block; downstream specs, source comments, tests, and the commit trailer carry the `CR-7` marker so `grep -rn "CR-7" .` returns all linked artifacts.

### FR-046: Reverse-Engineer Architecture on First Contact
- **Description:** When a change-request session opens against a repo with NO `docs/SPEC_ARCHITECTURE.md`, `reverse_engineer_architecture_node` MUST run once to synthesize a baseline architecture spec from a representative file sample (≤30 files / ≤100 KB cumulative), biased toward entry-point basenames (`main.py`, `app.py`, `pyproject.toml`, `package.json`, `index.ts`, `go.mod`) and skipping noise dirs (`.git`, `node_modules`, `__pycache__`, `dist`, `build`, `.venv`). The node MUST be gated by `change_requests.reverse_engineer_budget_usd` (default `$0.50`) and skip with an INFO log when the remaining session budget is below the cap (downstream delta-mode discovery still runs). On subsequent change-request sessions the file already exists and the node is a no-op.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given no prior `docs/SPEC_ARCHITECTURE.md` and `budget_remaining_usd > 0.50`, the node fires one planning-role LLM call and writes the file.
  - Given the file already exists, the node skips with a log line and no LLM call is made.
  - Given `budget_remaining_usd < change_requests.reverse_engineer_budget_usd`, the node skips with a budget-gate log line; the delta-mode discovery that follows still runs.

### FR-047: Setup Wizard for Bare `harness run`
- **Description:** When `harness run` is invoked with no `-r` / `-p` flags, the CLI MUST drop the operator into an interactive setup wizard (`harness/wizard.py:run_setup_wizard`). The wizard MUST first ask "new session or resume?". For a new session it MUST collect workspace path, prompt (or change-requests folder confirmation), `--new_build true|false` (default `false` for existing code so the harness does not clobber files), and `--git enable|disable`. Resume MUST jump straight to `harness resume` with the chosen session. The wizard's behaviour MUST be skippable via direct flag passing; passing any one of `-r`, `-p`, or `--manifest` MUST bypass the wizard entirely.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `harness run` with no flags, the wizard prompts: new vs resume → workspace → prompt-source → `--new_build` → `--git`.
  - Given resume is chosen, the wizard lists checkpointed sessions newest-first and hands off to `harness resume --session-id <chosen>`.
  - Given `harness run -r /tmp/x -p "fix bug"`, the wizard is skipped.

### FR-048: Per-Question Discovery Defaults + Optional Org-Wide `deployment.json`
- **Description:** Each discovery question MUST accept a bare Enter (empty input) as "use the default value baked into the prompt." The harness MUST also load an optional org-wide policy file from `config/deployment.json` (or `~/.harness/deployment.json`); when present, its already-resolved fields MUST be injected into the deployment-discovery LLM prompt as known answers so the planner does not re-ask. The file is OPTIONAL — absence preserves the full questionnaire. A `config/deployment.json.example` MUST ship with the repo as a template.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given a discovery question with a documented default and the operator hits Enter, the default is recorded as the answer and a CONFIRM line is logged.
  - Given `~/.harness/deployment.json` declares `target_environment: "compose-dev"`, the deployment-discovery LLM is told that field is resolved and asks no question about it.
  - Given no `deployment.json` file is present, the full questionnaire runs as before.

### FR-049: Workspace Git-Awareness Toggle (`--git enable|disable`)
- **Description:** `harness run` MUST accept `--git enable|disable` (default `enable`). When `enable`, `GitGuardian` performs stash → patch-branch → commit/rollback as today and requires the workspace to be a git repo. When `disable`, every git-aware step MUST be skipped (`_make_git_guardian` returns a no-op stub with the same interface) so operators whose target repo isn't under git can still run the harness. File-scanning security tools (gitleaks, bandit, semgrep) MUST still run in either mode — they scan files, not history.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `--git enable` and a non-git workspace, the CLI exits 1 with a "not a git repo" message.
  - Given `--git disable` and a non-git workspace, the run proceeds and security scanners still execute against the file tree.
  - Given `--git disable` and a HITL abandon, no rollback is attempted and the workspace is left as the LLM left it.

### FR-050: Kitchen-Sink Builder Sandbox Image
- **Description:** The harness MUST ship a single multi-stack Docker image (`harness/vendor/Dockerfile.builder`) that contains Python, Node.js, Go, Java, Rust, Dart, and Make toolchains plus a slim base. The graph MUST stop dispatching a per-command Docker image (the old "per-build-command" lookup is retired); compiler/lintgate/test-generation nodes all run inside the same builder image. Slim toolchain images (`python:3.12-slim`, `node:20-slim`, etc.) MUST still be honoured as swappable bases when the operator pins one in `sandbox.docker_image`, but `make`-based builds MUST always have `make` available (bootstrap-installed by the sandbox layer if missing).
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given the default config and a polyglot workspace (Python + Node), both stacks build inside the same container without per-command image dispatch.
  - Given `sandbox.docker_image: "python:3.12-slim"` and a `make build` command, the sandbox layer ensures `make` is available before invoking the build.
  - Given a `sh: 1: <cmd>: not found` error in build output, the parser surfaces the missing tool without the `/bin/` prefix mismatch.

---

## 3. System Scope

### In-Scope
- CLI interface with 6 subcommands (run, resume, status, doctor, purge, metrics) plus `--version`
- LangGraph-based agent graph with 20+ nodes
- Multi-provider LLM gateway (DeepSeek, Anthropic, OpenAI, Ollama)
- Hierarchical JSON configuration with deep merge + recursive typo detection
- SEARCH/REPLACE patch application with AST-aware fallback
- Sandboxed build execution (Docker → unshare → bare, in auto-detect priority)
- Structured diagnostic parsing for Rust, GCC/Clang, Go, Python, Java, TypeScript, Dart, and a generic fallback
- Cross-model speculative repair escalation (cheap → expensive)
- Human-in-the-loop interactive menu with 7 actions, pluggable transport (stdin / file / HTTP webhook)
- Zero-knowledge secret redaction before all API calls
- Git branch lifecycle management (stash, patch branch, commit, rollback)
- Exhaustive 3-phase discovery pipeline with structured Q&A loops (opt-in via `--discover`)
- Pre-flight manifest → spec synthesis with interactive review
- SQLite checkpoint persistence with WAL mode, 30-day TTL GC, schema-version stamping, strict-deserialize pre-flight on resume, and message redaction on every aput / aput_writes
- Read-only session status inspector with timestamp and workspace display
- First-run healthcheck (`harness doctor`) covering six environment preconditions, with the api-keys check matching the runtime resolution policy (env var OR `models["<key>"].api_key`)
- Cost-metrics aggregation (`harness metrics`) with human / JSON / Prometheus output, sliding-window burn rate, and projected exhaustion against `token_budget.hard_cap_usd`
- Per-session JSONL log file with `RotatingFileHandler` (10 MB × 5 backups by default), configurable via `logging.max_bytes` / `logging.backup_count`
- fcntl-based workspace lock (`.harness_session.lock`) preventing concurrent sessions on the same workspace; `--force-lock` for stale-lock recovery
- Pre-flight LLM-budget refusal (`BudgetTooLowError`), empty-response retry with `EmptyLLMResponseError` route-to-HITL short-circuit, and a rate-limit circuit breaker that diverts to local Ollama after 3 hits in 5 min
- Structured failure-event catalogue (`log_failure(name, **fields)`)
- Lint gate with auto-detected formatters per language (Python, Java, JS/TS, Dart, Go, Rust, C/C++, shell, SQL, markdown, YAML, JSON, HTML, CSS)
- Multi-variant speculative compilation in parallel git worktrees
- Container deployment pipeline (telemetry → blueprint → Dockerfile → docker compose v2 → health check); **opt-in via `--dev-deployment`** (off by default — clean security scan ends the run otherwise); short-circuits to END for Flutter / mobile projects regardless of the flag
- Change-request folder mode (`change_requests/*.txt` → monotonic CR-N IDs → marker propagation through specs / source / tests / commits → `applied/<session-id>/` archive with `manifest.json`) for incremental work against existing repos
- One-shot reverse-engineer of `SPEC_ARCHITECTURE.md` on first contact with a brownfield repo, gated by `change_requests.reverse_engineer_budget_usd` ($0.50 default)
- Interactive setup wizard on bare `harness run` (new-vs-resume → workspace → prompt-source → `--new_build` → `--git`)
- Per-question Enter-to-accept defaults during discovery + optional org-wide `deployment.json` policy file (template at `config/deployment.json.example`) that pre-resolves deployment-discovery answers
- Workspace git-awareness toggle (`--git enable|disable`); `disable` runs every git-aware step as a no-op so non-git workspaces still work
- Single kitchen-sink builder image (`harness/vendor/Dockerfile.builder`, Python + Node + Go + Java + Rust + Dart + Make) shared by compiler / lintgate / test-generation nodes; per-command image dispatch retired
- Per-stack Makefile skills (`harness/skills/makefile_python.md`, `makefile_node.md`, `makefile_go.md`, `makefile_java.md`, `makefile_rust.md`, `makefile_dart.md`) so the LLM emits a real `Makefile` for each stack
- Post-build security scanning (gitleaks + bandit/semgrep)
- Conversation memory cleanse for prefix-cache optimization
- Dependency graph impact analysis backed by tree-sitter grammars for Python, Java, JS/TS, Dart, Rust, Go, and C/C++
- Two-tier skills system (harness-level + project-level markdown conventions) with stack-aware filtering via `applies_to:` frontmatter
- Patcher symlink guard (`O_NOFOLLOW` + `os.path.islink`) and a conservative fallback allowlist when the source root cannot be auto-detected
- Discovery JSON trust guards: 1 MB byte-size cap + depth-10 recursion guard in `trust.validate_discovery_json`
- Sandbox network auto-enable gated by `sandbox.auto_enable_network_for_install` opt-in (default false)
- Hard cap on the discovery interview loop via `node_throttle.max_discovery_iterations` (default 10)
- GitHub Actions CI: pytest pack on Linux (Python 3.11 / 3.12 / 3.13, blocking) plus macOS + Windows (Python 3.12, advisory `continue-on-error`); separate `quality` job with `ruff check` blocking
- MIT `LICENSE` at repo root; `requirements-prod.txt` with exact transitive pins for reproducible pilot installs

### Out-of-Scope
- Interactive IDE plugin or VS Code extension
- Web-based dashboard or GUI (intentionally deferred per `docs/production-readiness-audit.md` T4.1 — out of scope for v1.x without user demand)
- Multi-user concurrent session management
- Cloud-hosted SaaS offering
- Non-Git version control systems (Mercurial, SVN)
- Guaranteed native Windows support (Windows + WSL2 is best-effort; the `unshare` backend and `fcntl` workspace lock are Linux-only). Windows is covered in CI as advisory (`continue-on-error`) to surface regressions but is not a blocking platform.
- Real-time streaming collaboration
- Built-in code review or PR management
- Training or fine-tuning of LLMs
- Example workspaces shipped in-tree (deferred per audit T3.1 — speculative without user feedback)

---

## 4. Technical Constraints

### Language and Runtime
- **Language:** Python 3.11+ (CI matrix: 3.11 / 3.12 / 3.13)
- **Async Model:** asyncio with `async/await` throughout
- **Type System:** TypedDict for LangGraph compatibility; no Pydantic dependency (removed — see `SPEC_ARCHITECTURE.md` §5.8).
- **Package Manager:** pip + pyproject.toml

### Key Dependencies (runtime)
| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| langgraph | 0.4.0 | Stateful graph execution with checkpointing |
| langgraph-checkpoint-sqlite | 2.0.0 | SQLite persistence backend |
| aiofiles | 24.0.0 | Async file I/O |
| tree-sitter | 0.23.0 | AST-aware code manipulation |
| tree-sitter-language-pack | 1.8.0 | Bundled grammars for 165+ languages (Python / Java / JS / TS / TSX / Dart / Rust / Go / Swift / …). Replaces six individual `tree-sitter-*` grammar packages. |
| httpx | 0.28.0 | Async HTTP client for LLM API calls |
| uuid7 | 0.1.0 | Time-sortable UUID generation |
| typing-extensions | 4.12.0 | TypedDict and type hint backports |

### Key Dependencies (dev / test)
| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| pytest | 8.0.0 | Test runner |
| pytest-asyncio | 0.24.0 | `@pytest.mark.asyncio` support |
| ruff | 0.8.0 | Lint + format |
| mypy | 1.13.0 | Strict type checking |
| pre-commit | 3.7.0 | Local commit gate |
| msgpack | 1.0.0 | Required by the storage GC regression test (not a runtime dep — runtime code falls back to JSON if absent). |

### Platform Requirements
- **OS:** Linux is the blocking CI target (Python 3.11 / 3.12 / 3.13). macOS and Windows (both Python 3.12) are covered as advisory `continue-on-error` jobs in the matrix — regressions surface but do not block merge. The `unshare` backend and fcntl-based workspace lock are Linux-only.
- **Sandbox backend:** Docker daemon, or Linux user-namespace support (`unshare --user`), or `HARNESS_ALLOW_UNSAFE_SANDBOX=true` opt-in for the bare backend.
- **Disk:** ~10MB for checkpoint database per 30-day window; up to ~50MB per session for rotated JSONL logs (10 MB live × 5 backups, default).
- **Network:** Outbound HTTPS required for LLM API calls (unless `force_local_only` + Ollama).

### Performance Targets
- CLI startup (config discovery): < 100ms
- Checkpoint read (single session): < 50ms
- Build sandbox startup (unshare): < 500ms
- Build sandbox startup (Docker): < 2s
- LLM dispatch with retry: < 30s per call (exponential backoff with jitter)

### Security Requirements
- No secrets in outbound API calls (pre-transmission redaction)
- No secrets in checkpoints database — the `messages` channel is scrubbed through `harness.redactor` on every `aput` / `aput_writes` (opt-out via `persistence.redact_messages: false`)
- Sandbox network isolation by default (no outbound network unless `--allow-network`); auto-enable on detected pip/npm install is gated behind `sandbox.auto_enable_network_for_install` opt-in (default false)
- Process-wide `CommandValidator` enforced by every `SandboxExecutor` (set in `cmd_run` / `cmd_resume`)
- Patcher refuses writes to symlinks (`O_NOFOLLOW` + `os.path.islink`); falls back to a conservative allowlist when the source root cannot be auto-detected
- Discovery JSON parser refuses payloads > 1 MB or depth > 10 to bound trust-boundary input
- PGID-based process group termination on timeout (no orphaned child processes)
- Single-writer workspace lock prevents concurrent sessions from clobbering each other
- Git rollback on session abandonment (no unrecoverable workspace corruption)
- `harness purge --session-id` removes both checkpoint rows AND the per-session JSONL transcripts for GDPR-style deletion requests

---

## 5. Explicit Edge Cases

### Error States
- **LLM API unreachable:** Exponential backoff with jitter (3 retries). Gateway logs failure and returns error to calling node. Graph routes to HITL if all retries fail.
- **Manifest file empty:** `synthesize_requirements()` raises `RuntimeError("Manifest file is empty.")`.
- **Workspace path does not exist:** CLI exits with code 1 before any graph execution.
- **No checkpointer configured:** Graph runs ephemerally with `MemorySaver`; no crash recovery.
- **Config JSON malformed:** Error logged with exact file path; harness refuses to proceed.
- **Config key typo (top-level or nested):** WARNING logged with fuzzy-match suggestion (FR-030); the entry is ignored and the harness continues.
- **Build command not found:** Sandbox returns exit code 127; parsed as a generic diagnostic.
- **Sandbox auto-detect fails entirely:** Raises `RuntimeError` and emits `sandbox_start_failed` event; falls back to bare only when `HARNESS_ALLOW_UNSAFE_SANDBOX=true` is set.
- **Token budget exhausted:** Gateway raises `RuntimeError` and emits `token_budget_exhausted` event.
- **Pre-flight budget too low for next call:** Gateway raises `BudgetTooLowError` without contacting the provider; emits a WARNING.
- **LLM returns empty content:** Gateway retries up to 2x; on exhaustion raises `EmptyLLMResponseError` and emits `llm_empty_response`. `repair_node` sets `llm_silent=True`; the graph short-circuits to HITL.
- **Rate-limit circuit opens:** After 3 HTTP 429/5xx failures in 5 minutes, gateway diverts the next call to `force_local=True` (Ollama) and logs a WARNING.
- **Workspace lock held by another session:** `cmd_run` exits 1 with a "lock held by PID X" message unless `--force-lock` is passed.
- **Checkpoint corrupted:** `cmd_resume` raises `CheckpointCorruptedError`; the operator is offered fresh-start / restore-backup / purge-session options.
- **Checkpoint schema mismatch:** `cmd_resume` raises `CheckpointSchemaMismatchError` and refuses to load if the stamped `_harness_schema_version` is higher than `CHECKPOINT_SCHEMA_VERSION`. Legacy (unstamped, pre-P2.4) checkpoints warn-and-allow.
- **Patcher write target is a symlink:** Refused with a WARNING; symlink target is left intact.
- **Discovery JSON > 1 MB or depth > 10:** `validate_discovery_json` returns an error without parsing or after a depth walk; the response is rejected.
- **Discovery loop hits `max_discovery_iterations`:** Route short-circuits to `write_spec_node` with a WARNING; no further questions are issued.
- **Auto-network detection fires but opt-in is off:** Sandbox does NOT enable network; logs a WARNING pointing at `sandbox.auto_enable_network_for_install`.
- **Log file grows past `logging.max_bytes`:** `RotatingFileHandler` rolls the live file to `.1`, shifting older backups; oldest is dropped at `logging.backup_count`.
- **HITL abandon chosen:** `_attempt_git_rollback()` runs and `hitl_gate_blocked` event is emitted.
- **gitleaks not installed:** Security scan falls back to Python regex-based secret scanner.
- **msgpack module missing:** `_deserialize_checkpoint_blob()` falls back to JSON text decoding for legacy rows.
- **`harness doctor` failure:** Non-zero exit with a one-line summary listing failed checks; warnings (e.g. only-Ollama routing) do not block exit 0.
- **`harness metrics` with no logs:** `--all` exits 1; `--session-id <id>` against a missing session exits 1 so cron detects regression.
- **`change_requests/` folder empty under `--new_build=false`:** CLI exits 1 with a clear error telling the operator to add at least one `.txt` file; there is no implicit "use the prior product_spec" fallback.
- **Change-request ID collision with archive:** A filename `CR-<N>-<rest>.txt` whose `N` clashes with an existing `change_requests/applied/**/CR-<N>-*.txt` aborts the session so the operator can rename and retry.
- **Both `-p "..."` and a populated `change_requests/` folder supplied:** The folder wins and the seed prompt is dropped with a WARNING log line; the folder is the single source of truth.
- **Bare `harness run` with no flags:** Drops the operator into the setup wizard; supplying any of `-r`, `-p`, or `--manifest` bypasses the wizard.
- **`--dev-deployment` not set + clean security scan:** Graph ends at the security-scan boundary; no Dockerfile / compose / `docker compose up` is produced. A `[cli] Code generated at <path>. Deployment phase skipped.` line is logged.
- **`--git disable` + HITL abandon:** No git rollback is attempted; the workspace is left as the LLM left it (matches the operator's stated intent of running outside git).

### Boundary Conditions
- **Max repair iterations:** 3 (hardcoded in `route_after_compiler`)
- **Max security fix attempts:** 2 (hardcoded in `route_after_security_scan`)
- **Max discovery iterations:** 10 (default; `node_throttle.max_discovery_iterations`, clamped to `[1, 30]`)
- **Default budget cap:** $2.00 USD
- **Sandbox timeout:** 300 seconds (configurable via `sandbox.timeout_seconds`)
- **Checkpoint TTL:** 30 days (configurable via `persistence.ttl_days`)
- **Checkpoint schema version:** `CHECKPOINT_SCHEMA_VERSION = 1` (current); minimum resumable: 1
- **Log file rotation:** 10 MB live size, 5 backups (default; `logging.max_bytes` / `logging.backup_count`; `0` disables rotation)
- **Discovery JSON byte cap:** 1 MB (`_MAX_DISCOVERY_BYTES`)
- **Discovery JSON depth cap:** 10 (`_MAX_DISCOVERY_DEPTH`)
- **Rate-limit circuit breaker threshold:** 3 failures in 5-minute sliding window
- **Empty-LLM-response retry budget:** 2 extra retries after the transport-retry loop
- **Pre-flight budget approach WARNING threshold:** within 20% of remaining cap
- **Burn-rate window for `harness metrics`:** 10 minutes (default; `metrics.burn_rate_window_minutes`, clamped to `[1, 1440]`)
- **Default metrics output dir:** `~/.harness/metrics/` (`metrics.metrics_dir`)
- **Max files per directory in tree snapshot:** 50
- **Max directory depth in tree snapshot:** 4
- **Max skills file chars:** 4000 (harness) / 3000 (project)
- **HITL raw build output display:** Last 2000 characters
- **Repair prompt raw output fallback:** Last 2000 characters
- **Token budget context window threshold:** 85% (truncation trigger)
- **Disk log buffer max size:** 500MB
- **Default `--dev-deployment`:** off (deployment phase opt-in)
- **Reverse-engineer architecture budget cap:** $0.50 USD (`change_requests.reverse_engineer_budget_usd`)
- **Change-request file scan:** `change_requests/` top-level `.txt` files only; `applied/` archive subdirectory is skipped

### Recovery Scenarios
- **Process killed mid-graph:** Next `harness run` loads from latest checkpoint; LangGraph replays from the boundary.
- **Network timeout during LLM call:** Gateway retries with exponential backoff + jitter (up to 3 attempts), then the rate-limit circuit breaker may divert to Ollama if the failure pattern persists.
- **Build timeout in sandbox:** PGID-based `kill(-pgid, SIGKILL)` → `SIGTERM` escalation after 5s.
- **Single corrupted session:** `harness purge --session-id <id>` removes only that thread's checkpoints AND its JSONL log + rotated backups; other sessions are unaffected.
- **Corrupted checkpoint DB across the board:** `harness purge --all` wipes and recreates; sessions are lost but the workspace is untouched.
- **Stale workspace lock from a crashed prior session:** `harness run -r <ws> -p '...' --force-lock` releases the stale lock and acquires a fresh one (operator confirms the prior PID is gone). See `docs/RUNBOOK.md` § 4.
- **Git stash conflict:** `git stash pop` may fail if stash conflicts; harness logs warning and continues (working tree is in the patch branch state).
- **Self-serve recovery playbooks:** `docs/RUNBOOK.md` covers the top-five operator failure modes (checkpoint corrupted, budget exhausted mid-session, sandbox can't start, workspace lock refused, persistent LLM silence) with symptom / diagnostic / fix recipes.

---

## 6. Non-Functional Requirements

### Reliability
- Checkpoint after every node transition (crash-safe within one graph step)
- WAL mode SQLite tolerates unexpected process termination
- All LLM calls wrapped in try/except with graceful degradation
- GitGuardian ensures workspace is never left in a corrupted state

### Scalability
- Single-user CLI tool (no multi-tenancy requirements)
- Checkpoint DB scales to thousands of sessions without performance degradation (SQLite with indexes)
- Disk-buffered log streaming keeps RAM constant regardless of build output size

### Observability
- Structured Python logging with timestamps, levels, and module names
- All LLM calls logged with token counts and cost
- All file writes logged with path and byte count
- Session introspection via `harness status` without graph execution
- Build output captured in full (stdout + stderr) via disk log streamer

### Maintainability
- TypedDict schemas for compile-time type safety across nodes (Pydantic was evaluated and removed — see `SPEC_ARCHITECTURE.md` §5.8)
- All nodes are isolated async functions with explicit state → state contracts
- Gateway providers implement a common interface for easy addition of new providers
- Diagnostic parsers are registered via a plugin registry (parser_registry.py)
- Skills are registered via a singleton SkillRegistry for extensibility
- Configuration is externalized to JSON files (no hardcoded model names or API keys in source)