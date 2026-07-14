# Teane — Requirements Specification

*Refreshed from current codebase state. Companion to `SPEC_ARCHITECTURE.md`.*

> **Terminology note (FR-072):** the legacy `teane run` verb has been split
> into four targets — `build` / `patch` / `deploy` / `test`. Older FR texts
> below that say `teane run` describe the shared engine now invoked through
> those targets: read `teane run --new-build true` as `teane build`,
> `--new-build false` as `teane patch`, and `--deploy-dev true` as
> `teane deploy`. The bare-invocation setup wizard (FR-047) now triggers on
> bare `teane build` / `teane patch`. Historical acceptance criteria are
> preserved as written for traceability.

---

## 1. Executive Summary

Teane is a production-grade, model-agnostic autonomous coding agent built on LangGraph. It accepts natural language engineering tasks (greenfield, `teane build`) OR a folder of change-request files (brownfield, `teane patch`, agile or waterfall), generates precise code patches via LLMs, verifies them through sandboxed builds — with a static diagnostics gate (FR-075) catching type errors before every compile — and OPTIONALLY brings the app up locally as a docker-compose dev environment (`teane deploy`) and exercises it end-to-end with Playwright (`teane test`, FR-073), whose failures feed back into `teane patch` as change requests. Failed runs distill learned rules into per-repo memory (FR-076); brownfield runs get language-server-backed semantic navigation (FR-077). It runs under budget guardrails, security scanning, and git lifecycle management. The system supports exhaustive multi-phase discovery (requirements → architecture → deployment) with per-question Enter-to-accept defaults and an optional org-wide `deployment_defaults` section in `config.json`, one-shot reverse-engineering of `SPEC_ARCHITECTURE.md` on first contact with brownfield repos, human-in-the-loop intervention points, checkpoint-based crash recovery, and cross-model speculative repair escalation. The supported stack is locked: backend is Python (FastAPI / Flask / Django) OR Java (Spring Boot); web is React + TypeScript + TailwindCSS, Vite-built. Any other stack is rejected at config-load time.

---

## 2. Functional Requirements (FR)

### FR-001: CLI Subcommand Routing
- **Description:** The system MUST provide a `harness` CLI with subcommands `run`, `resume`, `status`, `doctor`, `purge`, and `metrics`, each with their own argument parsers and help text. The root parser MUST also accept a `--version` / `-V` flag that prints the installed package version (resolved via `importlib.metadata.version("teane")`) and exits.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given `teane -h`, the system displays help with all six subcommands listed.
  - Given `teane run -h`, the system displays run-specific help with all flags documented.
  - Given `teane --version`, the system prints `teane <X.Y.Z>` and exits 0; the version falls back to `(unknown)` for uninstalled in-tree runs.

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

**Locked stack via `core_languages`.** The merged config MUST contain a `core_languages` block enforcing the supported stack:

```yaml
core_languages:
  backend_language: "Python"   # or "Java"; blank → "Python"
  web_language: ["React", "TypeScript", "TailwindCSS"]   # exact set; blank → defaults
```

Unsupported values (e.g. `backend_language: "Go"`, `web_language: ["Vue", ...]`) cause the harness to exit with code 2 at config-load time, before any logging, lock acquisition, or LLM-gateway initialisation. Blank values resolve to the documented defaults (`Python` for backend; the React+TypeScript+TailwindCSS triple for web). The `build_command` config key and `--build-cmd` CLI flag have been REMOVED — the harness auto-wires the build command from workspace markers (`pyproject.toml` → `pytest`, `pom.xml` → `mvn -B test`, `package.json` → `npm install && npm run build && npm test`).

### FR-005: Code Patch Generation and Application
- **Description:** The system MUST generate code patches in a strict SEARCH/REPLACE block syntax and apply them to workspace files via a hybrid patcher (AST-aware + text fallback).
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given an LLM response containing `<<<REPLACE_BLOCK>>>` blocks, the patcher locates and replaces the target text.
  - Given an LLM response containing `<<<CREATE_FILE>>>` blocks, the patcher creates the specified file.
  - Given a REPLACE_BLOCK where the SEARCH text doesn't match, the patcher logs a failure.

### FR-006: Sandboxed Build Verification
- **Description:** The system MUST execute the project's build command inside an isolated sandbox. Auto-detect priority is Docker → unshare (Linux namespaces) → bare (opt-in via `HARNESS_ALLOW_UNSAFE_SANDBOX=true`). The build command is auto-wired from workspace markers (`pyproject.toml` → `pytest`, `pom.xml` → `mvn -B test`, `package.json` → `npm install && npm run build && npm test`); there is no `build_command` config key or `--build-cmd` CLI flag. Build output MUST be parsed for structured diagnostics.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given a workspace with `pyproject.toml`, the harness auto-wires `pytest` and runs it inside a sandbox, returning exit code + diagnostics.
  - Given a compilation error in Python / Java / TypeScript / generic format, structured `DiagnosticObject` dicts are extracted.
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
  - Given user selects [s] (save & quit), the session is checkpointed and the developer is shown the exact `teane resume --session-id` command.
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
  - Given `--spec-discovery true` is passed (opt-in), requirements discovery runs with an 8-sector cross-examination prompt. Discovery is skipped by default.
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
- **Description:** The system MUST persist graph state to a SQLite database (WAL mode) at every node transition. `teane resume --session-id` MUST restore and continue from the last checkpoint. Each checkpoint's metadata MUST carry a `_harness_schema_version` stamp (current `CHECKPOINT_SCHEMA_VERSION = 1`); `cmd_resume` MUST pre-flight the most recent blob with strict deserialization and refuse to load on `CheckpointCorruptedError` or `CheckpointSchemaMismatchError`. The `messages` channel MUST be redacted through `harness.redactor` before serialization (opt-out via `persistence.redact_messages: false`, default `true`).
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given a running graph, checkpoints are written to `~/.harness/checkpoints.db` with the schema version stamped in metadata.
  - Given `teane resume --session-id <id>`, the graph resumes from the checkpointed state after the pre-flight check passes.
  - Given a corrupted checkpoint blob, `cmd_resume` exits with an operator-readable message offering fresh-start / restore-backup / purge-session options.
  - Given a checkpoint stamped with a future schema version, `cmd_resume` refuses with an upgrade-or-purge message.
  - Given a non-existent session ID, resume exits with error code 1.
  - Given a prompt containing an API-key-shaped secret, the byte sequence is absent from the on-disk SQLite checkpoint blob.

### FR-017: Read-Only Status Inspection
- **Description:** `teane status --all` MUST list all checkpointed sessions with session ID, created time, updated time, and workspace path. `teane status --session-id <id>` MUST display a full state snapshot.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given `teane status --all`, a table with SESSION ID, UPDATED, CREATED, and WORKSPACE columns is printed.
  - Given `teane status --session-id <id>`, a detailed state dump with all fields is printed.
  - Given a non-existent session ID, a "not found" message is printed.

### FR-018: Session Data Purging
- **Description:** `teane purge --all` MUST delete all checkpoint data after confirmation. `teane purge --session-id <id>` MUST delete that session's checkpoints AND its per-session JSONL log file (`<id>.jsonl`) plus any rotated backups (`<id>.jsonl.*`). Log-file removal is best-effort: a single OS error MUST log a WARNING and continue rather than abort the purge.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `teane purge --all` and user confirms "yes", all rows in the checkpoints DB are deleted.
  - Given `teane purge --session-id <id>`, only that thread's checkpoints are deleted and the count of removed log files is printed.
  - Given a session whose log file cannot be removed (permissions, race), the checkpoint deletion still completes and the failure is logged at WARNING.

### FR-019: Lint Gate (Deterministic Format Verification)
- **Description:** Before each build, modified files MUST be auto-formatted and linted using language-specific tools. Lintgate ships specs for `.py` / `.pyi` (ruff), `.ts` / `.tsx` / `.js` / `.jsx` / `.css` / `.html` / `.json` / `.yaml` / `.yml` / `.md` (prettier), and `.java` (google-java-format). Lint errors are surfaced in the build output. By default, formatting only runs on files actually patched this session (`lintgate.format_modified_files=false`); linters run on all modified files.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given modified `.py` files, ruff format + ruff check are executed.
  - Given modified `.tsx` files, prettier runs.
  - Given no matching formatter for a file extension, it is skipped.

### FR-020: Multi-Variant Speculative Execution
- **Description:** After patching, the system MAY generate N parallel code variants, compile each in isolated git worktrees, and select the winner by first_success, fewest_changes, or all_pass strategy.
- **Priority:** Could Have
- **Acceptance Criteria:**
  - Given `speculative.enabled: true` in config, N variants are generated in parallel.
  - Given one variant compiles successfully and others fail, the successful variant is selected.
  - Given all variants fail, the system falls back to the original patching flow.

### FR-021: Container Deployment
- **Description:** After a successful build and clean security scan AND when the operator has opted in via `--deploy-dev` (see FR-044), the system MUST scan workspace telemetry, synthesize a deployment architecture blueprint, generate Dockerfiles + docker-compose.yml + Caddyfile, build containers, and run health checks. Without `--deploy-dev` the graph ends at the clean-scan boundary and the workspace is handed back with the generated code in place but no Docker artifacts.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `--deploy-dev` and a Python workspace with `requirements.txt`, a Python Dockerfile is generated.
  - Given the deployment blueprint, `docker compose up --build -d` is executed (Compose V2 syntax, no hyphen).
  - Given containers are running, health check polling confirms readiness within 30s.
  - Given `--deploy-dev` is NOT set, no Dockerfile / compose / `docker compose up` is produced and the graph routes to END after the security scan.

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

### FR-025: First-Run Healthcheck (`teane doctor`)
- **Description:** The CLI MUST expose `teane doctor`, which runs six healthchecks and reports each as PASS / WARN / FAIL with a colored marker (suppressed when stdout is not a TTY or `NO_COLOR` is set): git repo presence, global config presence, API keys per configured `model_routing` provider, sandbox backend reachability, checkpoint DB writability and corruption scan over the 5 most recent rows, and config parse cleanliness (re-running `discover_config` + `_validate_config_keys`). The api-keys check MUST consider a provider satisfied when EITHER the `{PROVIDER}_API_KEY` env var OR the `models["<provider>:<model>"].api_key` config field is set (matching the runtime resolution in `gateway.BaseProviderClient.__init__`); the PASS message MUST report the source (`(env)` vs `(config)`) per model so operators see which key the runtime would actually use. The api-keys check MUST also issue a one-token chat call against each provider in parallel to confirm the resolved key actually authenticates against the configured model; HTTP-status-specific FAIL messages distinguish key-rejected (401), no-model-access (403), model-not-found (404), rate-limited (429), provider error (5xx), and network failures. Set `HARNESS_DOCTOR_SKIP_LIVE=true` to skip the live ping (CI / headless / outbound-network-blocked environments) — the doctor then reports presence and source only.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given a healthy install, `teane doctor` exits 0.
  - Given an API key only in the `models["<key>"].api_key` config field (no env var), the `api keys` check reports PASS with `(config)` next to the model id (after a successful live ping).
  - Given an env var AND a config field both set, the PASS message reports `(env)` (env wins precedence, matching the runtime).
  - Given neither env var nor config field set for a routed non-Ollama provider, the `api keys` check reports FAIL with a message naming BOTH the env var to set AND the `models."<key>".api_key` path; the command exits non-zero.
  - Given a configured key that returns HTTP 401, the live ping reports FAIL with `HTTP 401 — API key rejected` and the command exits non-zero.
  - Given `HARNESS_DOCTOR_SKIP_LIVE=true`, no outbound HTTP request fires and the PASS detail notes `(live ping skipped via HARNESS_DOCTOR_SKIP_LIVE)`.
  - Given a corrupted checkpoint blob among the 5 most recent rows, the `checkpoint db` check reports FAIL with the row identifier.
  - Given a typoed nested config key, the `config parse` check reports WARN with the fuzzy-match suggestion.

### FR-026: Multi-Stack Tree-Sitter Coverage
- **Description:** The patcher, impact analyzer, and diagnostic parsers MUST cover the locked stack: Python, Java, and JavaScript/TypeScript (React + Tailwind, Vite-built). Grammars MUST come from a single bundled wheel (`tree-sitter-language-pack`) to avoid per-language dependency churn.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given a Java compilation error, `JavaParser` parses it; given TypeScript, `TypeScriptParser`.
  - Given a Python traceback, `PythonParser` extracts file / line / message.
  - Given an unknown extension, parsing falls back to `GenericParser` (regex on `file:line:col: severity: message`).

### FR-027: Stack-Aware Skill Filtering
- **Description:** Skill files in `harness/skills/` MAY declare an `applies_to: [tag1, tag2]` YAML frontmatter. At graph assembly, the workspace is fingerprinted into a tag set drawn from the locked stack (`python`, `java`, `spring`, `fastapi`, `flask`, `django`, `react`, `typescript`, `tailwind`); skill files with a non-overlapping `applies_to` set MUST be excluded from the LLM prompt. Skill files with no frontmatter MUST always load (universal skills).
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given a Spring Boot workspace (tags include `java`, `spring`), `spring_boot.md` loads and `python_fastapi.md` does not.
  - Given a workspace tag set that doesn't intersect any `applies_to` declaration, only frontmatter-free skills load.

### FR-028: Stack-Lock Enforcement at Config Load
- **Description:** The harness MUST refuse any workspace whose detected stack falls outside the locked set: backend in {Python, Java}, web exactly {React, TypeScript, TailwindCSS}. The `core_languages` config block enforces this; unsupported values exit with code 2 at config-load time. There is no Flutter / mobile path: greenfield deployment requires the supported web stack, otherwise the run ends at the security-scan boundary like any non-deployed run.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given `core_languages.backend_language: "Go"`, the harness exits with code 2 at config-load time naming the offending field.
  - Given `core_languages.web_language: ["Vue", "TypeScript", "TailwindCSS"]`, the harness exits with code 2.
  - Given a blank `core_languages.backend_language`, the harness defaults to `Python` and proceeds.

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

### FR-032: Cost-Metrics Aggregation (`teane metrics`)
- **Description:** The CLI MUST expose `teane metrics`, which reads `<id>.jsonl` plus rotated backups (`<id>.jsonl.*`) under `logging.log_dir`, aggregates `llm_call` cost / tokens, counts tracked failure events (`token_budget_exhausted`, `llm_empty_response`, `llm_circuit_open`, `sandbox_start_failed`, `hitl_gate_blocked`), computes a trailing-window burn-rate in USD/min, and projects exhaustion against `token_budget.hard_cap_usd`. Flags: `--session-id`, `--all`, `--json`, `--prometheus`, `--output` (path or `-` for stdout), `--window-minutes`. Human-readable output goes to stdout; machine-readable outputs (`--json` / `--prometheus`) write atomically (`<dest>.tmp` → `os.replace`) into `metrics.metrics_dir` (default `~/.harness/metrics/`).
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given a session log with three `llm_call` records (cost $0.10, $0.20, $0.05), `teane metrics --session-id <id>` prints `Total cost: $0.3500` and a non-zero burn rate.
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
  - Given two `teane run` invocations against the same workspace, the second exits with a clear "lock held by PID X" message.
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
  - Given an auto-wired build command containing `pip install -e .` and `auto_enable_network_for_install: false` (default), the sandbox does NOT auto-enable network.
  - Given the same build command and the opt-in `true`, network IS auto-enabled.

### FR-043: Hard Cap on Discovery Loop
- **Description:** `node_throttle.max_discovery_iterations` (default 10, clamped to `[1, 30]` at config load) MUST hard-cap the number of discovery loop iterations. `route_after_discovery` MUST short-circuit to `write_spec_node` with a WARNING when `discovery_question_count >= max_discovery_iterations`. The key MUST appear in every config layer (cli.json, config.json, .harness_config.json, templates) and in the `node_throttle` whitelist.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `max_discovery_iterations: 3` and a fourth discovery question, the graph routes to `write_spec_node` instead of issuing the LLM call.
  - Given a value outside `[1, 30]`, it is clamped at load and logged.

### FR-044: Opt-In Deployment Phase (`--deploy-dev`)
- **Description:** The deployment phase (optional deployment discovery → `DEPLOYMENT_BLUEPRINT.md` → gatekeeper approval → `docker compose up`) MUST be off by default. `teane run` MUST accept `--deploy-dev true|false` (default `false`) on `run_parser` and thread it through `run_graph(dev_deployment=...)` into `AgentState["dev_deployment"]`. `route_after_security_scan` MUST consult the flag: with a clean scan and `dev_deployment=False`, the router MUST return `"__end__"`; with `dev_deployment=True` it MUST return `"deployment_discovery_node"` (when `--cd-discovery true`) or `"deployment_node"` (when `--cd-discovery false`, reading `deployment.json` directly). The existing `deployment.enabled` config switch is a NARROWER gate that only short-circuits the docker step inside `deployment_node` once the phase is already running.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given `teane run` with no `--deploy-dev`, after a clean security scan the run ends with no Dockerfile / compose / containers produced and `[cli] Code generated at <path>. Deployment phase skipped.` is logged.
  - Given `teane run --deploy-dev true --cd-discovery true` and a clean security scan, the router enters `deployment_discovery_node`.
  - Given `--deploy-dev` AND `deployment.enabled: false` in config, the phase enters discovery and writes `DEPLOYMENT_BLUEPRINT.md`, but `deployment_node` skips the docker step with `{"skipped": True, "reason": "disabled"}`.

### FR-045: Change-Request Folder Mode
- **Description:** The harness MUST support a `change_requests/` folder at the workspace root containing one or more spec files (`.txt`, `.md`, `.pdf`), each a self-contained ask. `cmd_run` MUST detect the folder (or be told via the wizard) and route through `ingest_change_requests_node` instead of the bare-prompt path. The ingest node MUST (1) walk only the top-level spec files, skipping `applied/`; (2) assign monotonic `CR-N` IDs starting at `max(applied/**/CR-*) + 1`; (3) respect operator-supplied IDs in filenames matching `CR-<N>-<rest>.{txt,md,pdf}`, aborting on collisions with archived IDs; (4) extract file contents — `.txt`/`.md` as UTF-8, `.pdf` via `pypdf` — concatenate them under `# === CR-N: <relative-path> ===` separators and inject the result as the first user message. At session end, consumed files MUST be moved into `change_requests/applied/<session-id>/` (extension preserved) with a `manifest.json` recording the status (`success` / `cancelled` / `failed-build`). When both `-p "..."` and a populated folder are supplied, the folder wins and the prompt is dropped with a WARNING.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given an empty `change_requests/` folder under `--new-build false`, the CLI exits with a clear error directing the operator to add at least one spec file (`.txt`, `.md`, or `.pdf`).
  - Given files `feature-x.txt` + `CR-12-bugfix.md` and prior archive `applied/abcd/CR-3-old.txt`, the new IDs are CR-4 (feature-x) and CR-12 (bugfix); a collision with CR-3 aborts.
  - Given a successful run, the consumed spec files land under `change_requests/applied/<session-id>/` (extension preserved) with `manifest.json` recording `status: "success"`.
  - Given `CR-7` is assigned, the LLM's first user message references it inside a `# === CR-7: feature-x.txt ===` block; downstream specs, source comments, tests, and the commit trailer carry the `CR-7` marker so `grep -rn "CR-7" .` returns all linked artifacts.

### FR-046: Reverse-Engineer Architecture on First Contact
- **Description:** When a change-request session opens against a repo with NO `docs/SPEC_ARCHITECTURE.md`, `reverse_engineer_architecture_node` MUST run once to synthesize a baseline architecture spec from a representative file sample (≤30 files / ≤100 KB cumulative), biased toward entry-point basenames (`main.py`, `app.py`, `pyproject.toml`, `package.json`, `pom.xml`, `index.ts`) and skipping noise dirs (`.git`, `node_modules`, `__pycache__`, `dist`, `build`, `.venv`). The node MUST be gated by `change_requests.reverse_engineer_budget_usd` (default `$0.50`) and skip with an INFO log when the remaining session budget is below the cap (downstream delta-mode discovery still runs). On subsequent change-request sessions the file already exists and the node is a no-op.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given no prior `docs/SPEC_ARCHITECTURE.md` and `budget_remaining_usd > 0.50`, the node fires one planning-role LLM call and writes the file.
  - Given the file already exists, the node skips with a log line and no LLM call is made.
  - Given `budget_remaining_usd < change_requests.reverse_engineer_budget_usd`, the node skips with a budget-gate log line; the delta-mode discovery that follows still runs.

### FR-047: Setup Wizard for Bare `teane run`
- **Description:** When `teane run` is invoked with no `-w` / `-p` flags, the CLI MUST drop the operator into an interactive setup wizard (`harness/wizard.py:run_setup_wizard`). The wizard MUST first ask "new session or resume?". For a new session it MUST collect workspace path, prompt, `--git true|false` (default `false`), `--new-build true|false` (default `false`), and `--spec-discovery true|false` (default `false`). Resume MUST jump straight to `teane resume` with the chosen session. The wizard's behaviour MUST be skippable via direct flag passing; passing either `-w` or `-p` MUST bypass the wizard entirely.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `teane run` with no flags, the wizard prompts: new vs resume → workspace → prompt → `--git` → `--new-build` → `--spec-discovery`.
  - Given resume is chosen, the wizard lists checkpointed sessions newest-first and hands off to `teane resume --session-id <chosen>`.
  - Given `teane run -w /tmp/x -p "fix bug"`, the wizard is skipped.

### FR-048: Per-Question Discovery Defaults + Optional Org-Wide `deployment_defaults` Section
- **Description:** Each discovery question MUST accept a bare Enter (empty input) as "use the default value baked into the prompt." The harness MUST also load an optional org-wide policy from the `deployment_defaults` section of `config/config.json`; when populated, its already-resolved fields MUST be injected into the deployment-discovery LLM prompt as known answers so the planner does not re-ask. The section is OPTIONAL — when absent or `{}`, the full questionnaire is preserved. `config/config.json` MUST document the section's schema and example values inline via its `_deployment_defaults_comment` field.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given a discovery question with a documented default and the operator hits Enter, the default is recorded as the answer and a CONFIRM line is logged.
  - Given `config.json` includes `deployment_defaults.network.reverse_proxy = "caddy"`, the deployment-discovery LLM is told that field is resolved and asks no question about it.
  - Given no `deployment_defaults` section is present (or it is `{}`), the full questionnaire runs as before.

### FR-049: Workspace Git-Awareness Toggle (`--git true|false`)
- **Description:** `teane run` MUST accept `--git true|false` (default `false`). When `true`, `GitGuardian` performs stash → patch-branch → commit/rollback as today and requires the workspace to be a git repo. When `false`, every git-aware step MUST be skipped (`_make_git_guardian` returns a no-op stub with the same interface) so operators whose target repo isn't under git can still run the harness. File-scanning security tools (gitleaks, bandit, semgrep) MUST still run in either mode — they scan files, not history.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `--git true` and a non-git workspace, the CLI exits 1 with a "not a git repo" message.
  - Given `--git false` and a non-git workspace, the run proceeds and security scanners still execute against the file tree.
  - Given `--git false` and a HITL abandon, no rollback is attempted and the workspace is left as the LLM left it.

### FR-050: Kitchen-Sink Builder Sandbox Image
- **Description:** The harness MUST ship a single multi-stack Docker image (`harness/vendor/Dockerfile.builder`) that contains Python, Java, and Node.js (for the React + TypeScript + TailwindCSS web build) toolchains plus a slim base. The graph MUST stop dispatching a per-command Docker image (the old "per-build-command" lookup is retired); compiler/lintgate/test-generation nodes all run inside the same builder image. The build command itself is auto-wired from workspace markers (`pyproject.toml` → `pytest`, `pom.xml` → `mvn -B test`, `package.json` → `npm install && npm run build && npm test`) — there is no `build_command` config key or `--build-cmd` CLI flag. Slim toolchain images (`python:3.12-slim`, `node:20-slim`, `eclipse-temurin:21-jdk`) MUST still be honoured as swappable bases when the operator pins one in `sandbox.docker_image`.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given the default config and a polyglot workspace (Python backend + React/TS frontend), both stacks build inside the same container without per-command image dispatch.
  - Given a workspace with `pyproject.toml`, the auto-wired build command runs `pytest` inside the sandbox.
  - Given a `sh: 1: <cmd>: not found` error in build output, the parser surfaces the missing tool without the `/bin/` prefix mismatch.

### FR-051: MCP (Model Context Protocol) Client
- **Description:** The harness MUST support connecting to one or more MCP servers declared in `config.mcp.servers` and exposing each server's advertised tools as `mcp__<server>__<tool>` skills in the `SkillRegistry`. The MCP client MUST implement JSON-RPC 2.0 over stdio (newline-delimited frames) without depending on the upstream `mcp` SDK so the core install stays dependency-clean. Server commands MUST be validated through `harness.trust.validate_mcp_server_command` (allowlist of `npx`/`node`/`python*`/`uvx`/`docker`; hard-deny on shells / `sudo` / `rm`; shell-metacharacter scan; `/etc /root /proc /sys` path rejection). Filesystem MCP servers MUST be gated behind `mcp.allow_local_filesystem_servers=true`.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `mcp.enabled=true` and a valid stdio server, `teane doctor` lists the server with the count of advertised tools.
  - Given the planner emits a `<<<MCP_CALL server="x" tool="y" args='{...}'>>>` block, the graph's `_run_tool_loop` intercepts it, dispatches via the MCP client, and feeds the result back as a user message.
  - Given a server command that fails the allowlist, the pool refuses to start and logs the rejection reason; one bad server never blocks the rest of the pool.

### FR-052: Provider Prompt Caching
- **Description:** The gateway MUST emit Anthropic `cache_control: {"type": "ephemeral"}` markers on the system block (and on the first user message when ≥ 4 KB) for cache-capable models when `llm_dispatch.prompt_cache_enabled=true` (default). For OpenAI / DeepSeek the gateway already deducts `cache_read_input_tokens` at the discounted rate; the gateway MUST also run a **prefix-stability drift detector** that hashes the first two messages per `(session, role)` and emits a `cache_prefix_drift` observability event when the hash changes between consecutive calls — surfacing silent cache misses on auto-cache providers.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given an Anthropic dispatch with `supports_cache=true`, the request payload's `system` field is a list-of-blocks with `cache_control: {"type": "ephemeral"}`.
  - Given two consecutive dispatches for the same session+role with a mutated immutable preamble, a `cache_prefix_drift` event is emitted.
  - Given `llm_dispatch.prompt_cache_enabled=false`, Anthropic requests fall back to the legacy string-form `system` payload.

### FR-053: Web Research Tools (`WebFetchSkill`, `WebSearchSkill`)
- **Description:** The harness MUST expose `web_fetch` and `web_search` to the planner via text-DSL blocks (`<<<WEB_FETCH url="...">>>`, `<<<WEB_SEARCH query="...">>>`) when `web_tools.enabled=true`. Default backend: `duckduckgo_lite` (no API key). Outbound URLs MUST be validated through `harness.trust.validate_outbound_url` which rejects `file://`/`javascript:` schemes, loopback / link-local / RFC-1918 hosts (SSRF guard) unless `web_tools.allow_private_ips=true`. Response size MUST be capped at `web_tools.max_bytes` and content-type MUST be on the allowlist (`text/html`, `text/plain`, `text/markdown`, `application/json`, `application/xml`).
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given the planner emits a `<<<WEB_FETCH url="https://docs.python.org/">>>` block, the graph intercepts, fetches the URL, and feeds the readable text back as a user message.
  - Given an LLM-supplied URL targeting `169.254.169.254`, `validate_outbound_url` rejects it before the HTTP call.
  - Given `web_tools.enabled=false`, the skills are not registered and any tool block is left in the response with a "tool not registered" notice.

### FR-054: GitHub Integration (`teane gh`)
- **Description:** The harness MUST ship a `teane gh` subcommand family wrapping the `gh` CLI (no new Python dep). `teane gh issue --repo X --number Y` MUST pull an issue body and write it to the workspace's `change_requests/CR-<N>-<slug>.txt` so the existing change-request flow processes it. `teane gh pr-create` MUST open a PR from the workspace's current branch; `teane gh pr-comment` MUST post a comment on an existing PR. Authentication MUST defer to whatever `gh auth status` reports.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `gh` is on PATH and authenticated, `teane gh issue --repo owner/repo --number 42` creates `change_requests/CR-N-<slug>.txt` with the issue body.
  - Given `gh` is NOT on PATH, the subcommand exits non-zero with a clear "install gh CLI from cli.github.com" message.
  - Given a PR-create call from a branch with no commits ahead of base, the `gh pr create` exit code surfaces verbatim.

### FR-055: Runtime-Extensible Skills Directory
- **Description:** `register_builtin_skills(config)` MUST walk `~/.harness/user_skills/` (or the path named by `skills.user_skills_dir`) at startup and import every non-`_`-prefixed `*.py` file. Each loaded module MAY call `harness.skills.register(MySkill(...))` to add a `ToolSkill`/`PipelineSkill`/`SubAgentSkill`, or `harness.web_tools.register_backend(name, factory)` to plug in an alternative web-search backend, without modifying core code. Failures (syntax error, missing dep, import-time exception) MUST log and continue so one bad file never blocks startup. The loader MUST fall back to the legacy default `~/.harness/skills/` when only the legacy directory exists, and MUST emit a one-time deprecation INFO naming both paths so operators know to migrate.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given a valid `~/.harness/user_skills/demo.py` that calls `register(...)` at module load, the skill appears in `SkillRegistry.list_all()` after `teane run` starts.
  - Given a `~/.harness/user_skills/broken.py` that raises `RuntimeError` at import time, the harness logs the failure and continues without crashing.
  - Given neither `~/.harness/user_skills/` nor `~/.harness/skills/` exists, the loader silently no-ops.
  - Given only the legacy `~/.harness/skills/` exists (operator has not yet migrated), the loader uses it AND logs one INFO line per process pointing at the new default.

### FR-056: Repository Semantic Retrieval (`teane index`)
- **Description:** The harness MUST ship a per-workspace semantic-retrieval index buildable via `teane index build`. Two backends MUST be supported: zero-dep `tfidf` (default, deterministic, pure Python with identifier-aware tokenisation) and opt-in `openai_embeddings` (using `OPENAI_API_KEY`, falling back to TF-IDF when the key is missing). Index storage MUST be SQLite at `~/.harness/repo_index/repo_index.db`. When `repo_index.enabled=true`, `planning_node` MUST query top-K chunks for the user prompt and inject them as a system context block capped at `repo_index.inject_max_bytes`. `teane index {build, status, clear}` MUST be exposed as a CLI subcommand family.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `teane index build -r /repo`, the SQLite store at `~/.harness/repo_index/repo_index.db` contains one row per chunk with `(workspace_id, file_path, chunk_index, vector_json)`.
  - Given `repo_index.enabled=true`, the planner's system message includes a `### Repository context (semantic retrieval)` block when the index has been built.
  - Given `OPENAI_API_KEY` is unset with `repo_index.backend=openai_embeddings`, the backend falls back to TF-IDF with a one-time warning.

### FR-057: Per-Repository Session Memory
- **Description:** The harness MUST persist a markdown session log per repository at `~/.harness/memory/<repo_id>.md`, where `repo_id` is the SHA-256 (first 16 hex chars) of `git remote get-url origin` if available, else the absolute workspace path. `planning_node` MUST read the file at start and prepend it as an extra system message; `cmd_run` and `cmd_resume` MUST append a session note (prompt summary, modified files, exit code) at end-of-run. File size MUST be FIFO-trimmed to `memory.max_bytes`. Default: `enabled=true`.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given a cloned repo with `git remote get-url origin` returning a stable URL, the same memory file is read across hosts.
  - Given `memory.enabled=false`, no read or write happens and the planner context excludes the memory block.
  - Given a memory file exceeds `memory.max_bytes`, the FIFO trim drops the oldest `## Session` sections; the most recent entry is always preserved.

### FR-058: Interactive Refinement REPL (`teane chat`)
- **Description:** The harness MUST ship a `teane chat` subcommand that opens an interactive REPL reusing the Gateway, redactor, web/MCP tool loop, repo-memory injection, and (when enabled) repo-index injection. The REPL MUST NEVER auto-apply patches — the LLM may emit SEARCH/REPLACE blocks but they only land when the operator types `/apply` and confirms. Slash commands: `/help`, `/exit`, `/clear`, `/files`, `/apply`, `/build`, `/save <path>`, `/budget`, `/memory`. The conversation MUST be in-memory only in v1 (no cross-session persistence).
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given a `teane chat -r /repo --budget 1.00` invocation, the REPL accepts a prompt, dispatches through the gateway, and surfaces the response in the terminal.
  - Given the assistant emits patch blocks, `/apply` invokes `process_llm_patch_output` against the workspace with a per-session HITL confirmation.
  - Given `/build`, the auto-wired build command runs in the sandbox and the first 80 lines of output surface in the REPL.

### FR-059: Coverage Reporting
- **Description:** The harness MUST ship a `make coverage` target driven by `pytest-cov`. The target MUST emit a terminal summary, an HTML report under `htmlcov/`, and an XML report at `coverage.xml`. No CI gate on the coverage number is required in v1 — the metric is for visibility.
- **Priority:** Could Have
- **Acceptance Criteria:**
  - Given `make coverage` succeeds, `htmlcov/index.html` exists and contains the per-file coverage view.
  - Given `pytest-cov` is installed via the `dev` extras, the target completes without operator manual install.

### FR-060: Multi-Agent Fan-Out Primitive
- **Description:** The harness MUST expose a parallel-agent runner (`harness.fanout.run_parallel_agents`) with bounded asyncio semaphore concurrency (default 8) and shared-budget reservation/refund accounting. An adversarial-skeptic helper `run_with_verification` MUST run a finder + N independent verifiers and decide by majority vote. The runner MUST be exposed to the planner as a `SubAgentFanoutSkill` registered in the `SkillRegistry` so the planner can emit `<<<FANOUT_QUERY prompts='[...]'>>>` blocks for N parallel queries.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given a list of `AgentSpec`s, `run_parallel_agents` returns results in input order, with `AgentResult.success=False` for any that failed (no exception escapes the runner).
  - Given the shared budget would be exceeded mid-fan-out, the runner rejects subsequent reservations and the corresponding `AgentResult.error` mentions "budget exhausted".
  - Given a `voted` verification with majority-refuted votes, `Verdict.is_real=False`.

### FR-061: Configuration-Driven Speculative Execution (Rebuild)
- **Description:** `harness.speculative.speculate_node` MUST expose six independent strategy axes via `speculative.*` config: `trigger` ∈ {`always`, `first_attempt_only`, `after_n_repair_failures` [default; threshold 2], `manual`}; `diversity_mode` ∈ {`temperature`, `prompt`, `model` [default], `mixed`}; `cost_strategy` ∈ {`equal_cost`, `cheap_first_sequential` [default], `cheap_parallel_then_expensive`, `all_cheap`}; `selection_strategy` ∈ {`first_pass` [default; `first_success` alias], `fewest_changes`, `voted`, `all_pass`}; `salvage_strategy` ∈ {`none` [default], `fewest_errors`, `voted_partial`, `merge`}; `voting` `{n_judges, judge_role}`. Legacy configs without the new keys MUST auto-upgrade with a one-time deprecation warning to byte-identical legacy behaviour.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `trigger=after_n_repair_failures` and `loop_counter.repair < threshold`, `speculate_node` is a no-op (falls through to the standard flow).
  - Given `cost_strategy=cheap_first_sequential`, variants dispatch one at a time using `cheap_model`; the last variant uses `expensive_model`.
  - Given a legacy config of `{enabled, num_variants, temperature, selection_strategy}`, `_upgrade_legacy_config` populates `diversity_mode=temperature`, `cost_strategy=equal_cost`, `salvage_strategy=merge`, `trigger=first_attempt_only` with a WARNING log.

### FR-062: Cron-Driven Scheduled-Job Daemon (`teane schedule`)
- **Description:** The harness MUST ship a `teane schedule {run, list, validate, once, history}` subcommand family backed by `harness/schedule.py`. The daemon MUST parse a hand-rolled cron syntax subset: `every Nm/h/d`, `hourly :MM`, `daily HH:MM`, `weekly DAY HH:MM` (DAY ∈ mon..sun); all times UTC. Each job MUST run as a `teane run` subprocess with a per-job log at `~/.harness/schedule_logs/<job>/<iso8601>.log`. History MUST persist to SQLite at `~/.harness/schedule.db`. `on_success` / `on_failure` MUST be generic shell hooks invoked via `/bin/sh -c` with `HARNESS_JOB_NAME` / `HARNESS_JOB_EXIT_CODE` / `HARNESS_JOB_DURATION_SEC` / `HARNESS_JOB_LOG_PATH` exported. In-flight tracking MUST prevent double-firing.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `schedule.enabled=true` and one due job, one `teane schedule run` tick spawns a subprocess and records the result in `schedule.db`.
  - Given an in-flight job from a prior tick, the next tick does NOT fire a second instance.
  - Given a malformed schedule string, `teane schedule validate` exits non-zero listing the offending job + the supported forms.

### FR-063: Read-Only Web Dashboard (`teane web`)
- **Description:** The harness MUST ship a `teane web` subcommand that runs a localhost-only HTTP server (default bind `127.0.0.1`, port 8729) over the harness's on-disk state. Views MUST include: sessions list, per-session detail, cost burn-down (Chart.js via CDN), scheduled-job history, repo-index status, per-repo memory list. The server MUST support optional bearer-token auth via `dashboard.token_env`; when set but the env var is empty the server MUST refuse to start (fail-closed). Zero new Python dependencies (stdlib `http.server.ThreadingHTTPServer`).
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `teane web --host 127.0.0.1 --port 8729`, an unauthenticated request without `Authorization` returns 401 when `token_env` is configured.
  - Given the workspace has session logs and a built repo index, all five views render without error.
  - Given `dashboard.token_env` names an empty env var, `start_server` raises `RuntimeError` and the subcommand exits 2.

### FR-064: Interactive Web App (Dashboard Tier B + C)
- **Description:** When `dashboard.writes_enabled` is true (the default), the dashboard MUST add form-based editing of config sections (form schema derived from the live `_KNOWN_NESTED_KEYS` + `_TYPE_SCHEMA` tables), memory-file editing, schedule-job CRUD, and a "New run" form with both "Run now" (spawns `teane run` subprocess) and "Schedule it" (enqueues `web.db:web_oneshot_jobs` row picked up by the schedule daemon). Live event streams MUST flow via Server-Sent Events at `/api/sessions/<id>/events`. HITL prompts MUST surface in the UI via the existing `harness/hitl.py:HttpChannel`: the dashboard registers as the webhook URL, blocks the harness's POST while the UI displays the prompt, and signals back when the operator answers. Chat notes MUST queue per session and ride into the next HITL gate's `extra_notes`. Write paths MUST require a CSRF double-submit cookie + `X-CSRF-Token` header. Config writes MUST be atomic (tempfile + `os.replace`) and re-validated through `validate_config_strict` before landing.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given the default `dashboard.writes_enabled: true` and a valid CSRF token, `POST /config/<section>` with a valid form body updates `config.json` atomically and re-renders the section with "Saved." flash.
  - Given a write request without `X-CSRF-Token`, the server returns 403.
  - Given `POST /run/now` with a workspace + prompt, the dashboard spawns the subprocess, sets `HARNESS_HITL_WEBHOOK_URL=http://<host>:<port>/hitl/webhook?session=<id>`, and registers the PID in the process registry.
  - Given the harness POSTs a HITL prompt to `/hitl/webhook`, the request blocks until the UI POSTs `/sessions/<id>/hitl/answer` with the operator's decision.

### FR-065: Per-Batch Verification Topology
- **Description:** When `--agile=true`, the harness MUST run the compile / code-review / test pipeline ONCE PER BATCH against the batch's combined patches, not once per story. Stories within a batch are patched sequentially via the `story_loop_node → patching_node → story_loop_node` cursor; when the batch is exhausted (every story has patched), control flows to `speculative_node → test_generation_node → lintgate_node → compiler_node → code_review_node`. Consumer nodes (code-review, test-generation, lintgate) MUST read their input from the new `batch_modified_files` state field via `_scope_files_for_consumer(state)` so each gate only sees the current batch's files, not the cumulative session set. The pre-batch behavior (monolithic mode, when no batch is active) MUST be preserved bit-for-bit.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given a 3-story batch, `compiler_node` runs exactly ONCE for the batch, not three times.
  - Given `current_batch_id > 0`, `code_review_node` reads `batch_modified_files`; given `current_batch_id == 0`, it falls back to `modified_files`.
  - Given `current_batch_id == 0` (monolithic), the patching → speculative edge fires as before — no behavioural regression for non-story-mode runs.

### FR-066: Per-Batch Sealing with Atomic Status + Optional Commit
- **Description:** When `current_batch_id > 0` and `code_review_node` passes cleanly, control MUST route to `batch_commit_node` (NOT `story_complete_node`). The commit node MUST mark every constituent story as `done` in a single transaction, resolve any open defects per story, and (when `agile_defaults.commit_on_story=true` in `~/.harness/config.json`) run `_commit_for_batch(workspace, batch_id, story_keys_with_titles)` which writes a single `BATCH-N: STORY-1: ...; STORY-2: ...` git commit and persists the SHA into `batches.committed_sha`. The node MUST reset `current_batch_id`, `current_story_id`, `batch_modified_files`, and the per-batch `loop_counter` keys (`patching`, `repair`, `compiler`, `total_repairs`, `review_code`, `consecutive_zero_patch_rounds`, `missing_dep_consecutive_same`) before returning so the next batch starts clean.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given a 3-story batch that passes review, `batch_commit_node` marks STORY-1, STORY-2, and STORY-3 as `done` in one transaction and the `batches.status` row updates to `complete`.
  - Given `agile_defaults.commit_on_story=true` and a workspace with `.git/`, the SHA recorded in `batches.committed_sha` matches `HEAD` after sealing and the commit subject begins with `BATCH-N:`.
  - Given a story was `blocked` mid-batch, `batches.status` is `complete_with_blocks` and the blocked story's status stays `blocked`.

### FR-067: End-of-Session Regression Gate
- **Description:** After every batch in a session has sealed and `security_scan_node` exits clean for the first time, the router MUST insert one final regression run through `end_of_session_regression_node` before any deployment path. The node delegates to `compiler_node` for the build/test execution but stamps `node_state.end_of_session_phase = True` and uses its own counter (`loop_counter.end_of_session_regression_repair`) with the operator-configurable cap `gateway.config.max_end_of_session_regression_cycles` (default 3). On clean exit it routes to the deployment destination (`deployment_discovery_node` / `deployment_node` / `installation_doc_node`) per the same precedence as `route_after_security_scan`'s clean path. On failure it routes through `repair_node → compiler_node` until the cap is reached, then HITL. The router MUST treat `end_of_session_regression_repair > 0` as a marker that the gate already ran — a re-entry to `security_scan_node` via the repair loop MUST NOT re-enter the EoS regression (prevents an infinite security ↔ EoS loop).
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given `security_scan_node` returns clean and `end_of_session_regression_repair == 0`, `route_after_security_scan` returns `"end_of_session_regression_node"`.
  - Given `end_of_session_regression_repair >= 1` and `security_scan_node` returns clean again (after a repair round), the router skips EoS and goes directly to the deployment destination.
  - Given `node_throttle.max_end_of_session_regression_cycles = 3` and the build fails 3 times in EoS, the router routes to `human_intervention_node`.

### FR-068: End-of-Session Repair Authority
- **Description:** When the repair loop fires while `node_state.end_of_session_phase` is True, `repair_node` MUST surface a wider file-context window to the LLM and skip the cheap-model cycle. Specifically: (1) the diagnostic-file cap raises from 12 to `node_throttle.end_of_session_repair_diagnostic_cap` (default 30); (2) the workspace-inventory snapshot cap raises from 50 to `node_throttle.end_of_session_repair_inventory_cap` (default 150); (3) when `node_throttle.end_of_session_force_reasoning_model` is True (default), `use_escalation` is forced True on the first attempt so the senior reasoning model handles the diagnosis directly. The repair prompt MUST prepend an EoS-aware framing block telling the LLM that the failing tests likely involve shared utilities the security-scan repair touched, with explicit instruction to look at imports / recent modifications / utility files.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given `node_state.end_of_session_phase = True`, `_repair_file_caps(state)` returns `(30, 150)` (or operator-configured values).
  - Given `node_state.end_of_session_phase = True` and `end_of_session_force_reasoning_model = True`, `use_escalation` is True on the first repair attempt (counter 0).
  - Given `end_of_session_phase` is unset or False, the cap helper returns the default `(12, 50)` and escalation follows the existing repair-count-driven rule (last attempt only).

### FR-069: Change-Request Impact-Aware Repair Context
- **Description:** When `state["change_request_mode"]` is True and the repair loop fires, the repair LLM MUST receive up to +6 additional file slices beyond the diagnostic-named files, drawn from: (a) the workspace's most-depended-on files (`DependencyGraph.high_fanout_files`) intersected with the session's `modified_files` (so only utilities THIS CR amended appear), and (b) immediate one-hop callers of every diagnostic file (`DependencyGraph.immediate_callers_of`). The augmentation MUST de-duplicate against the existing diagnostic list and the inventory list. The repair prompt MUST prepend a CR-aware framing block telling the LLM that the failing tests may involve features outside the CR's stated scope and that the augmented file list is the primary suspect set. Outside CR mode the augmentation MUST be a no-op.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `change_request_mode = True` and a workspace where `utils.py` is imported by 4+ files, `_cr_impact_augment(state, [failing_test_file])` returns `utils.py` in its result list when `utils.py` is also in `modified_files`.
  - Given `change_request_mode = False`, `_cr_impact_augment(state, diag_files)` returns an empty list.
  - Given 8 consumers of a touched utility, the augmenter caps additions at 6 files (`_CR_EXTRA_FILE_CAP`).

### FR-070: Mid-Batch Resume via Per-Gate Progress Markers
- **Description:** The harness MUST track per-batch verification-gate progress in `AgentState.batch_gate_progress: dict[str, dict[str, bool]]`. `compiler_node` MUST set the active batch's `compile_passed = True` on clean exit when `current_batch_id > 0`; `code_review_node` MUST set `review_passed = True` when it passes without re-patching. On resume after a crash, `route_after_story_loop` (when `batch_complete == True`) MUST consult the active batch's markers and skip ahead to the next un-passed gate — `compile_passed` False → `speculative_node`; True with `review_passed` False → `code_review_node`; both True → `batch_commit_node`. `batch_commit_node` MUST pop the sealed batch's entry from `batch_gate_progress` before returning.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given a crash mid-test-loop after compile passed but review didn't run, `route_after_story_loop` returns `"code_review_node"` instead of `"speculative_node"`.
  - Given `batch_gate_progress["3"] = {"compile_passed": True}` and `current_batch_id = 2`, the router does NOT short-circuit (progress is per-batch and key-matched).
  - Given `batch_commit_node` runs on batch 5, `batch_gate_progress` no longer contains a `"5"` entry after the seal.

### FR-071: Intra-Batch Story Dependency Ordering
- **Description:** When two stories with a dependency relationship (`STORY-B.depends_on = ["STORY-A"]`) are placed in the same batch, the patcher MUST process STORY-A before STORY-B. (1) `batch_sizing.deterministic_batches` MUST topologically order each emitted batch's `story_keys` via `_topo_sort_within_batch`. (2) The LLM batch-sizing prompt (`build_batch_sizing_prompt`) MUST require dependents to appear after their deps in `story_keys`. (3) `batch_sizing.validate_batches` MUST allow same-batch deps (relaxing the prior "deps must be in earlier batch" rule) but MUST reject any batch where a dep appears AFTER its dependent in `story_keys`. (4) `story_loop._next_story_in_batch` MUST defensively skip stories whose intra-batch deps aren't yet `done` even when the recorded sequence is correct, protecting against corrupted DB rows or resumed sessions that landed mid-rewind.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given STORY-B depends on STORY-A and both are in batch 1 with `story_keys = ["STORY-A", "STORY-B"]`, validation passes and STORY-A is picked first.
  - Given `story_keys = ["STORY-B", "STORY-A"]` with the same dep, `validate_batches` returns an error containing "must come BEFORE its dependent".
  - Given STORY-A is `planned` and STORY-B (deps `["STORY-A"]`) is also `planned`, `_next_story_in_batch` returns STORY-A; once STORY-A is `done`, the next call returns STORY-B.

### FR-072: Four-Target CLI Surface (`build` / `patch` / `deploy` / `test`)
- **Description:** The CLI MUST expose four primary targets replacing the legacy `run` verb, each pinning `args.flow` before delegating to the shared engine: `teane build` (greenfield; `flow="build"`, `new_build=True`, workspace reset), `teane patch` (brownfield reconcile; `flow="patch"`, `new_build=False`, consumes `change_requests/*` files; `--agile` engages story decomposition on the same flow), `teane deploy` (`flow="deploy"`; artifact synthesis + dev container + health-check sign-off), and `teane test` (`flow="test"`; see FR-073). `_resolve_cli_exit_code` MUST return deterministic exit codes for automation: `0` clean, `1` partial success, `2` config error, `3` budget exhausted, `4` infrastructure failure — so `teane build && teane deploy` chains behave correctly in CI.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given `teane patch --agile true`, `args.flow == "patch"` is pinned BEFORE agile resolution, and story decomposition runs on the patch flow.
  - Given a session terminated by budget exhaustion, the process exit code is `3` (not a generic `1`).
  - Given a config with an unknown top-level key, the process exits `2` before any LLM call.

### FR-073: End-to-End Test Target (`teane test`)
- **Description:** `teane test` MUST run Playwright end-to-end tests against the deployed compose stack (`harness/test_target.py`, generation in `harness/playwright_gen.py`). A prerequisite gate MUST verify a prior clean tracked flow via `flow_state.record_flow_completion` before tests run. Every e2e failure MUST be emitted as a `CR-DEFECT-*` change-request file (`harness/test_defects.py`) consumable by a subsequent `teane patch` run — closing the build → deploy → test → patch loop without operator authorship.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given no prior clean build/patch/deploy flow for the workspace, `teane test` refuses with a clear prerequisite message.
  - Given a failing e2e scenario, a `CR-DEFECT-*.txt` file appears in the change-requests folder with reproduction context.
  - Given `teane patch` runs after a failed `teane test`, the defect files are ingested as change requests.

### FR-074: Spec Reconciler (Authoritative Requirement IDs)
- **Description:** In story mode, `spec_reconciler_node` (`harness/spec_reconciler.py`) MUST treat `SPEC_REQUIREMENTS.md` as the sole authority for feature/story IDs: decomposition output is reconciled against the spec, LLM output is consumed ONLY as `scope_files` enrichment, dropped enabler stories are recovered, and structural parent traceability (`story_satisfies_req`) is repopulated. `traceability_block` escalation cycles MUST be capped.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given an LLM decomposition that renames or invents requirement IDs, the reconciled stories carry ONLY IDs present in `SPEC_REQUIREMENTS.md`.
  - Given an enabler story omitted by the LLM pass, reconciliation restores it with parent linkage.
  - Given repeated traceability blocks, the cycle cap routes to HITL instead of looping.

### FR-075: Static Diagnostics Gate (Type-Checkers in the Repair Loop)
- **Description:** A read-only `diagnostics_node` (`harness/diagnostics_gate.py`) MUST run between `lintgate_node` and `compiler_node` (and on the `repair_node` re-entry edge), executing fast CLI type-checkers over the batch's files plus their reverse-dependency closure: pyright (`--outputjson`, mypy fallback) for Python and `tsc --noEmit` for TS/TSX. Java is exempt (the build IS its type check). Pre-existing brownfield diagnostics MUST be suppressed via a detached HEAD-worktree fingerprint baseline keyed to the HEAD SHA (batch commits invalidate it); only NEW error-severity diagnostics route to `repair_node`, emitted through the existing `compiler_errors` channel with the mandatory `_rotate_diag_fingerprints_delta` rotation. `route_after_diagnostics` MUST never escalate to HITL (escalation stays single-sourced in `route_after_compiler`) and MUST be doubly bounded (shared `total_repairs` cap + per-compile-cycle `diagnostics.max_rounds`, default 2). Every infrastructure failure (missing tool, timeout, non-git workspace) MUST fail open to `compiler_node`.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given a patch introducing a type error in a session-created file, the gate emits it as a NEW diagnostic and routes to `repair_node` before any compile.
  - Given a pre-existing type error at HEAD that merely shifted lines under a patch, the baseline suppresses it (line-insensitive fingerprints).
  - Given `diagnostics_rounds_since_compile > max_rounds`, the router falls through to `compiler_node` unconditionally.

### FR-076: Automated Failure Post-Mortems (HITL Learning Loop)
- **Description:** When a HITL breakpoint fires, `human_intervention_node` MUST emit a `hitl_fired` observability event and stage a distilled `[learned-rule:<trigger>] fp=<hash>` note in `state["post_mortem_note"]` (`harness/post_mortem.py`); the cli finalize path MUST be the single writer, appending the note to per-repo memory via `append_session_note(extra_notes=...)` (auto-injected into the next run's planner prompt), generating a note for failed runs that never reached HITL, and fingerprint-deduping repeat failure classes. LLM distillation MUST run on a synthetic budget floor (`post_mortem.max_cost_usd`, default $0.10) so it works when the session budget is exhausted, with a deterministic per-trigger template fallback so the loop never no-ops. Any clean (exit-0) run MUST retire all active rules (`[learned-rule(retired):...]`) — a green run proves the failure class no longer bites, and stale rules poison future prompts.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given a headless run abandoned at the auto-resume cap, the staged rule still reaches the repo memory file.
  - Given the same failure class in two consecutive sessions, the second session's note is skipped as a duplicate (`post_mortem_skipped` event).
  - Given a subsequent clean run, previously active rules are tag-rewritten to retired and no longer injected as active guidance.

### FR-077: LSP Semantic Navigation (Brownfield, Phase 1)
- **Description:** For brownfield flows only (`flow != "build"`, `lsp.enabled_flows` default `["patch","test"]`), the harness MUST start an LSP client pool (`harness/lsp_client.py`; pyright-langserver for Python, typescript-language-server for TS/TSX; jdtls deferred to Phase 2) gated by an environment-health probe: Python requires a `.venv`/`venv` at the workspace root (override `lsp.python_require_venv=false`); TypeScript requires `tsconfig.json` AND `node_modules`. The pool MUST power (a) planner tools `lsp__find_references` / `lsp__go_to_definition` via `<<<LSP_CALL ...>>>` blocks, and (b) three harness prefetch sites — diagnostics-gate impact expansion, repair-prompt caller maps, change-request impact augment — each with three-tier fallback (LSP → `DependencyGraph` → nothing). Greenfield behaviour MUST stay byte-identical (no pool, no prompt section). No auto-restart: a dead server degrades silently to heuristics; nothing about the pool lives in `AgentState` (checkpoint/resume-safe).
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `teane build` on any workspace, no LSP process is spawned and the system prompt contains no LSP section.
  - Given `teane patch` on a Python workspace without a venv, the pool skips the server with a probe reason in the `lsp_pool_started` event and all three sites use `DependencyGraph`.
  - Given a healthy pool whose server dies mid-session, subsequent prefetches return empty/None without raising and the skills answer with a polite error string.

### FR-078: Unattended-Run Hardening (Auto-Resume Cap + Anti-Drift Screens)
- **Description:** To keep headless runs from burning budget in loops: (1) consecutive headless HITL auto-resumes MUST be capped, with direct-abandon at the cap instead of ping-ponging through the `[q]` confirm; (2) a pre-patch anti-drift screen MUST reject patch blocks before application when they trip `[screen:over-cap]` (>3 surgical edits to one file per turn), `[screen:stuck-reread]` (edits against a stale file view after consecutive REPLACE_BLOCK misses), or `[screen:repeat-search]` (a search hash already rejected); (3) the patcher MUST reject `REPLACE_BLOCK`/`DELETE_BLOCK` operations against essentially-empty files (Guard 4).
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given a headless session hitting the auto-resume cap, the run abandons directly with `hitl_gate_blocked` logged, without an interactive confirm round-trip.
  - Given a fourth surgical edit to the same file in one turn, the screen rejects it with `[screen:over-cap]` before the patcher runs.
  - Given a `REPLACE_BLOCK` against an empty file, the patch is rejected with a Guard-4 error instead of a silent no-op loop.

### FR-079: Per-File Patcher Rejection Feedback to LLM
- **Description:** `patching_node` and `repair_node` MUST route every post-patcher LLM status message through `harness.patch_feedback.compose_patch_feedback`, which appends per-file `(file, operation, classification-tag): directive` blocks derived from `harness.patcher._classify_patch_failure`. Directive text MUST cover the primary failure classes — `file missing` → "use CREATE_FILE"; `search miss` → "READ_FILE and copy exact bytes"; `ambiguous match` → "add more context"; `rejected: file already exists` → "use REPLACE_BLOCK"; `path denied` / `allowlist denied` → move under allowed prefix; `no blocks parsed` → parser-miss diagnostic. Both nodes MUST share the same helper so future feedback improvements ship in one place; the helper preserves the parse-miss path (`fail_count == 0`) previously introduced by cc9ab6a.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given a patcher round that rejects an INSERT_AT_BLOCK on a non-existent file, the next LLM system message contains the file path AND the "use CREATE_FILE" directive.
  - Given a mixed rejection round (search miss + file missing), the LLM sees a per-file classification tag on each entry, capped at 5 entries.
  - Given `patching_node` and `repair_node` in the same session, both nodes emit status messages via `compose_patch_feedback` — the helper is not duplicated.

### FR-080: Line-Coverage Gate for Generated Apps (Operator-Configurable)
- **Description:** Every stack-specific Makefile skill (`harness/skills/makefile_python.md`, `makefile_node.md`) MUST require the LLM to emit a `test:` target that (a) enables line coverage and (b) applies a threshold gate when the operator has opted into enforcement. Two config knobs under `coverage` in `config.json` govern the behaviour:
  - `coverage.min_pct` — integer 0-100, default 70. The line-coverage percentage the LLM writes into pytest's `--cov-fail-under` flag and Jest's `coverageThreshold.global.{lines,statements}`.
  - `coverage.enforce` — bool, default true. When true, under-threshold builds exit non-zero (pytest / Jest's own exit code IS the gate) and `compiler_node` routes to `repair_node` to write more tests. When false, coverage is still measured (report generated) but the fail-under flag / `coverageThreshold` block is omitted, so the build passes regardless of coverage%.

  Values are injected into the skill markdown at prompt-build time via `{{coverage.*}}` substitution markers (`{{coverage.min_pct}}`, `{{coverage.pytest_fail_flag}}`, `{{coverage.jest_threshold_snippet}}`). The LLM sees the resolved text — no conditional reasoning required. `harness/skills/unit_tests_python.md` and `unit_tests_react.md` MUST tell the LLM what a unit test IS (mocked I/O, sub-millisecond, one behaviour) and IS NOT (e2e journeys — those belong to `teane test`).

- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given `coverage.enforce=true` and `coverage.min_pct=70` (defaults), the LLM's generated Makefile `test:` target contains `--cov=<pkg> --cov-fail-under=70` (Python) or the `coverageThreshold.global.lines=70` block (React/TS).
  - Given `coverage.enforce=false`, the Makefile still runs coverage but omits the fail-under flag / `coverageThreshold` block; a build with 60% coverage exits zero.
  - Given `coverage.min_pct=85` and enforce=true, generated Makefiles use `--cov-fail-under=85` and `coverageThreshold.global.lines=85`.
  - Given a build where UTs pass but coverage is under threshold with enforce=true, `make test` exits non-zero and the compile→repair loop re-enters to add more tests.
  - Given a malformed `coverage.min_pct` (string, non-integer), the harness falls back to the default 70 without crashing.

### FR-081: Flow-Aware Traceability Gate (Build/Patch enforce Reqs; Test enforces ACs)
- **Description:** The end-of-session traceability audit (`installation_doc_node` in `harness/graph.py`; `TraceabilityReport.has_req_gap()` / `.has_ac_gap()` in `harness/traceability.py`) MUST split its enforcement decision by flow: every flow enforces `has_req_gap` (a requirement lacking a satisfying story is a planner failure, always blocking), but AC coverage (`has_ac_gap` — an acceptance criterion lacking a linked `test_verifies_ac` row) is enforced ONLY when `state["flow"] == "test"`. Rationale: unit tests generated during `teane build` / `teane patch` link to code modules; ACs are closed by Playwright tests emitted by `teane test`. Blocking `build`/`patch` on AC coverage produced an unfixable headless auto-resume loop (finsearch session 156032347 — 25/124 ACs untested at end-of-build, `traceability_block` cycled to no effect). The per-batch soft warning in `story_loop.py::traceability_node` MUST reflect the same split ("AC coverage is closed by `teane test` and does not block build/patch"). The `traceability.enforce=false` operator switch continues to short-circuit both.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given `flow="build"` and a report with `untested_acs` populated but `untraced` empty, the audit exits clean (no `traceability_block`).
  - Given `flow="test"` and the same report, the audit blocks via `traceability_block`.
  - Given any flow with `untraced` populated, the audit blocks regardless of AC coverage.
  - Given `traceability.enforce=false`, both gates degrade to advisory printout only.

### FR-082: System-Prompt Diet — RSD Prose Stripping + Repair Message Pruning
- **Description:** Two prompt-diet transforms MUST run to bound the tokens shipped to the LLM without breaking prefix-cache economics: (1) `harness.cli._slim_spec_for_prompt` runs once at spec-load time to strip planner-only fields (`Business driver`, `Success metrics`, `Priority`, `Estimated size`, `Wave`, `Iteration`) from the RSD before it prepends the system prompt, while preserving every code-grounding field (assumptions, story titles, scope, out-of-scope, dependencies, ACs). The transform is applied at both spec-load sites (reuse-docs path AND interactive-review path); the resulting system prompt stays byte-identical across the session so prefix cache survives. (2) `harness.repair_context.prune_repair_messages` runs on every `repair_node` invocation and, when `loop_counter["total_repairs"] > 3`, keeps only `messages[0..2]` (immutable system prompt + initial user task) and the last 6 messages. Everything between is dropped so the LLM stops arguing with its own past assistant turns. Both transforms are surgical and require no config knobs, no state DB additions, and no cache invalidation.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given an agile RSD containing `**Business driver:**` and `**Success metrics:**` blocks, the slimmer removes them and preserves adjacent `**Vision statement:**` and `**Dependencies:**` blocks.
  - Given a `repair_node` invocation with `total_repairs=1`, the message array is passed through verbatim (full-history mode).
  - Given a `repair_node` invocation with `total_repairs=5` and a 15-message input, the pruned array is 8 messages long (2 head + 6 tail) and `messages[0:2]` is byte-identical to the pre-prune head (cache anchor preserved).

---

## 3. System Scope

### In-Scope
- CLI interface with four primary targets (`build`, `patch`, `deploy`, `test` — FR-072) plus supporting subcommand families (`resume`, `chat`, `status`, `doctor`, `pre-flight`, `audit`, `purge`, `metrics`, `gh {issue, pr-create, pr-comment}`, `index {build, status, clear}`, `schedule {run, list, validate, once, history}`, `dashboard`/`web`, `cache clear`) and `--version`
- LangGraph-based agent graph with 20+ nodes
- Multi-provider LLM gateway (DeepSeek, Anthropic, OpenAI, Ollama)
- Hierarchical JSON configuration with deep merge + recursive typo detection
- SEARCH/REPLACE patch application with AST-aware fallback
- Sandboxed build execution (Docker → unshare → bare, in auto-detect priority)
- Structured diagnostic parsing for Python, Java, TypeScript (React + Tailwind, Vite-built), and a generic fallback
- Cross-model speculative repair escalation (cheap → expensive)
- Human-in-the-loop interactive menu with 7 actions, pluggable transport (stdin / file / HTTP webhook)
- Zero-knowledge secret redaction before all API calls
- Git branch lifecycle management (stash, patch branch, commit, rollback)
- Exhaustive requirements + architecture discovery pipeline with structured Q&A loops (opt-in via `--spec-discovery true`)
- Pre-flight manifest → spec synthesis with interactive review
- SQLite checkpoint persistence with WAL mode, 30-day TTL GC, schema-version stamping, strict-deserialize pre-flight on resume, and message redaction on every aput / aput_writes
- Read-only session status inspector with timestamp and workspace display
- First-run healthcheck (`teane doctor`) covering six environment preconditions, with the api-keys check matching the runtime resolution policy (env var OR `models["<key>"].api_key`)
- Cost-metrics aggregation (`teane metrics`) with human / JSON / Prometheus output, sliding-window burn rate, and projected exhaustion against `token_budget.hard_cap_usd`
- Per-session JSONL log file with `RotatingFileHandler` (10 MB × 5 backups by default), configurable via `logging.max_bytes` / `logging.backup_count`
- fcntl-based workspace lock (`.harness_session.lock`) preventing concurrent sessions on the same workspace; `--force-lock` for stale-lock recovery
- Pre-flight LLM-budget refusal (`BudgetTooLowError`), empty-response retry with `EmptyLLMResponseError` route-to-HITL short-circuit, and a rate-limit circuit breaker that diverts to local Ollama after 3 hits in 5 min
- Structured failure-event catalogue (`log_failure(name, **fields)`)
- Lint gate with auto-detected formatters per language (Python, Java, JS/TS, markdown, YAML, JSON, HTML, CSS)
- Multi-variant speculative compilation in parallel git worktrees
- Container deployment pipeline (telemetry → blueprint → Dockerfile → docker compose v2 → health check); **opt-in via `--deploy-dev`** (off by default — clean security scan ends the run otherwise)
- Change-request folder mode (`change_requests/*.{txt,md,pdf}` → monotonic CR-N IDs → marker propagation through specs / source / tests / commits → `applied/<session-id>/` archive with `manifest.json`) for incremental work against existing repos
- One-shot reverse-engineer of `SPEC_ARCHITECTURE.md` on first contact with a brownfield repo, gated by `change_requests.reverse_engineer_budget_usd` ($0.50 default)
- Interactive setup wizard on bare `teane run` (new-vs-resume → workspace → prompt → `--git` → `--new-build` → `--spec-discovery`)
- Per-question Enter-to-accept defaults during discovery + optional org-wide `deployment_defaults` section in `config.json` (schema documented inline in `config/config.json`) that pre-resolves deployment-discovery answers
- Workspace git-awareness toggle (`--git true|false`, default `false`); when `false`, every git-aware step is a no-op so non-git workspaces still work
- Single kitchen-sink builder image (`harness/vendor/Dockerfile.builder`, Python + Java + Node — for the React/TS/Tailwind web build) shared by compiler / lintgate / test-generation nodes; per-command image dispatch retired; build command auto-wired from workspace markers (no `build_command` config key, no `--build-cmd` CLI flag)
- Per-stack Makefile skills (`harness/skills/makefile_python.md`, `makefile_java.md`, `makefile_node.md`) so the LLM emits a real `Makefile` for each supported stack
- Post-build security scanning (gitleaks + bandit/semgrep)
- Conversation memory cleanse for prefix-cache optimization
- Dependency graph impact analysis backed by tree-sitter grammars for the locked stack (Python, Java, JavaScript/TypeScript)
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
| tree-sitter-language-pack | 1.8.0 | Bundled grammars for the locked stack (Python / Java / JS / TS / TSX). |
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
- `teane purge --session-id` removes both checkpoint rows AND the per-session JSONL transcripts for GDPR-style deletion requests

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
- **`teane doctor` failure:** Non-zero exit with a one-line summary listing failed checks; warnings (e.g. only-Ollama routing) do not block exit 0.
- **`teane metrics` with no logs:** `--all` exits 1; `--session-id <id>` against a missing session exits 1 so cron detects regression.
- **`change_requests/` folder empty under `--new-build false`:** CLI exits 1 with a clear error telling the operator to add at least one spec file (`.txt`, `.md`, or `.pdf`); there is no implicit "use the prior product_spec" fallback.
- **Change-request ID collision with archive:** A filename `CR-<N>-<rest>.{txt,md,pdf}` whose `N` clashes with an existing `change_requests/applied/**/CR-<N>-*` aborts the session so the operator can rename and retry.
- **Both `-p "..."` and a populated `change_requests/` folder supplied:** The folder wins and the seed prompt is dropped with a WARNING log line; the folder is the single source of truth.
- **Bare `teane run` with no flags:** Drops the operator into the setup wizard; supplying any of `-r`, `-p`, or `--manifest` bypasses the wizard.
- **`--deploy-dev` not set + clean security scan:** Graph ends at the security-scan boundary; no Dockerfile / compose / `docker compose up` is produced. A `[cli] Code generated at <path>. Deployment phase skipped.` line is logged.
- **`--git false` + HITL abandon:** No git rollback is attempted; the workspace is left as the LLM left it (matches the operator's stated intent of running outside git).
- **MCP server command rejected by allowlist:** Pool start logs the rejection and skips the server; the rest of the pool continues. The `teane doctor` check for that server reports `fail` with the rejection reason.
- **MCP server start times out:** Server is skipped from the pool; its tools are absent from `SkillRegistry`. The LLM emitting `<<<MCP_CALL server=<name> ...>>>` sees a "server not registered" tool result.
- **Filesystem MCP server attempted with `allow_local_filesystem_servers=false`:** Pool start raises `ValueError`; the dashboard/doctor surface the gating reason.
- **Prompt cache prefix drift detected:** Warning logged + `cache_prefix_drift` event emitted; dispatch continues normally with the cache miss.
- **Anthropic API rejects the cache_control payload shape:** Operator can flip `llm_dispatch.prompt_cache_enabled=false` to revert to the legacy string-form system payload as a single-flag rollback.
- **Web tool URL fails SSRF guard:** Tool returns `{"error": "url rejected: ..."}` instead of fetching; the LLM sees the error message in its tool-result message.
- **Web tool content-type not in allowlist:** Tool returns `{"error": "content-type ... not in allowlist"}` without the body.
- **`gh` CLI not on PATH for `teane gh` subcommands:** Subcommand exits 1 with the install hint pointing at `https://cli.github.com/`.
- **User skill file raises at import time:** Loader logs the file path + exception and continues with the next file; the registry shows the skills from successful imports only.
- **Repo index built with one backend, queried with another:** Query loads the backend named in `repo_meta.backend` regardless of the live config; mismatched configs silently use the persisted backend.
- **Repo index never built but `repo_index.enabled=true`:** Planner injection no-ops cleanly; no warning beyond a debug log.
- **Per-repo memory file unreadable (permissions):** Read returns empty string; write silently fails with a warning log. Session continues without the memory block.
- **`teane chat` budget exhausted mid-session:** REPL prints "budget exhausted (use /budget to confirm)" and refuses further dispatches. Operator types `/exit` to leave.
- **`teane chat` `/apply` against an assistant message with no patch blocks:** Reports "no patch blocks detected in the last reply"; no files touched.
- **Speculative trigger not met:** `speculate_node` logs the reason (`patching_count=X > 1` or `repair_count=X < threshold`) and falls through to the standard flow.
- **Speculative cost_strategy=cheap_first_sequential with one cheap variant succeeding:** Subsequent variants are NOT dispatched (true cost savings); the registry reports `variant_results` for only the dispatched ones.
- **Speculative legacy config (no new strategy keys):** `_upgrade_legacy_config` injects the legacy-compatible defaults with a one-time `WARNING: legacy config detected` log line.
- **Fan-out shared budget exhausted mid-batch:** Subsequent agents return `AgentResult.success=False` with `error="shared budget exhausted ..."`; in-flight agents complete normally.
- **Schedule daemon: due job in-flight from previous tick:** Job is skipped this tick (no double-fire); next tick after exit re-evaluates.
- **Schedule daemon: one-shot web job consumed:** Row's `consumed_at` is set; the row remains in `web_oneshot_jobs` for audit but is never picked up again.
- **Dashboard write request without `X-CSRF-Token`:** Server returns 403 with `csrf token mismatch`. The form action retries after a fresh GET re-issues the cookie.
- **Dashboard `token_env` names an empty env var:** `resolve_expected_token` raises `RuntimeError`; subcommand exits 2.
- **Dashboard HITL webhook held longer than 600s:** Server returns 504 to the harness; the harness's HttpChannel raises and the gate falls through to the local StdinChannel fallback (or the next configured channel).
- **Dashboard config save fails strict validation:** Disk file untouched; form re-renders with per-field error messages from `validate_config_strict`.

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
- **Burn-rate window for `teane metrics`:** 10 minutes (default; `metrics.burn_rate_window_minutes`, clamped to `[1, 1440]`)
- **Default metrics output dir:** `~/.harness/metrics/` (`metrics.metrics_dir`)
- **Max files per directory in tree snapshot:** 50
- **Max directory depth in tree snapshot:** 4
- **Max skills file chars:** 8000 (harness) / 3000 (project) — harness cap raised from 4 KB so cross-cutting skills (`makefile_python.md` with the coverage gate, `unit_tests_*.md`) fit without truncation; matches the style-guide cap
- **Max style_guides file chars:** 8192 (focused) / 24576 (composite) — focused cap raised from 4 KB to give cross-cutting rules (datetime, path handling, package init, concurrency) room to sit alongside the base language guide
- **Coverage gate (generated apps):** operator-configurable via `coverage.min_pct` (int 0-100, default 70) and `coverage.enforce` (bool, default true) in `config.json`; injected into shipped skill markdown at prompt-build time via `{{coverage.*}}` substitution markers
- **Repair-history prune threshold:** 4 rounds (`DEFAULT_PRUNE_AFTER_ROUND=3` in `harness/repair_context.py`) — from round 4 onward, `repair_node` keeps only `messages[0:2]` + last 6 turns
- **Repair-history tail size:** 6 messages (`DEFAULT_KEEP_TAIL=6` in `harness/repair_context.py`)
- **HITL raw build output display:** Last 2000 characters
- **Repair prompt raw output fallback:** Last 2000 characters
- **Token budget context window threshold:** 85% (truncation trigger)
- **Disk log buffer max size:** 500MB
- **Default `--deploy-dev`:** false (deployment phase opt-in)
- **Default `--cd-discovery`:** false (container-deployment discovery opt-in)
- **Default `--spec-discovery`:** false (requirements + architecture interviews opt-in)
- **Default `--git`:** false (GitGuardian opt-in)
- **Default `--new-build`:** false (steady-state)
- **Default `--hitl-requirement` / `--hitl-architecture` / `--hitl-repair` / `--hitl-deployment` / `--hitl-layout-divergence`:** resolved as CLI flag (when explicitly passed) > `config.json`'s `hitl.*` block > `true` (gates prompt unless the operator opts out at either tier). Auto-approve fallbacks (CI=true, HARNESS_AUTO_APPROVE=true, non-TTY stdin) still override on top — those force auto-approve regardless of the resolved value.
- **Default `--allow-network`:** true (sandbox has network unless `--allow-network false`)
- **Reverse-engineer architecture budget cap:** $0.50 USD (`change_requests.reverse_engineer_budget_usd`)
- **Change-request file scan:** `change_requests/` top-level spec files (`.txt`, `.md`, `.pdf`); `applied/` archive subdirectory is skipped
- **MCP tool-call timeout:** 30s (default; `mcp.tool_call_timeout_seconds`)
- **MCP result payload cap:** 200 KB (default; `mcp.result_max_bytes`)
- **Web tools per-fetch byte cap:** 200 KB (default; `web_tools.max_bytes`)
- **Web tools per-dispatch tool-loop cap:** 3 rounds (default; `web_tools.tool_call_cap_per_dispatch`)
- **Repo memory file cap:** 100 KB total (default; `memory.max_bytes`); 8 KB injected to planner (default; `memory.inject_max_bytes`)
- **Repo index chunk window:** 200 lines with 20-line overlap (default; `repo_index.chunk_lines`, `repo_index.chunk_overlap`)
- **Repo index top-K:** 5 chunks (default; `repo_index.top_k`); 4 KB injection cap (default; `repo_index.inject_max_bytes`)
- **Fan-out concurrency cap:** 8 agents (default; `max_concurrency` arg to `run_parallel_agents`)
- **Schedule daemon tick interval:** 60s (default; `schedule.tick_seconds`)
- **Schedule daemon command allowlist:** Built-in `harness` binary path; operators can override via `schedule.harness_binary`
- **Dashboard default bind:** `127.0.0.1:8729` (default; `dashboard.host`, `dashboard.port`)
- **Dashboard sessions enumeration cap:** 200 sessions (default; `dashboard.sessions_max`)
- **Dashboard HITL webhook hold timeout:** 600 s (operator UI must answer within 10 minutes or the harness's HttpChannel sees a 504)
- **Speculative trigger threshold:** 2 repair failures (default; `speculative.n_repair_failures_threshold`)
- **Speculative voting judges:** 3 (default; `speculative.voting.n_judges`); judge role: `code_reviewer` (default)

### Recovery Scenarios
- **Process killed mid-graph:** Next `teane run` loads from latest checkpoint; LangGraph replays from the boundary.
- **Network timeout during LLM call:** Gateway retries with exponential backoff + jitter (up to 3 attempts), then the rate-limit circuit breaker may divert to Ollama if the failure pattern persists.
- **Build timeout in sandbox:** PGID-based `kill(-pgid, SIGKILL)` → `SIGTERM` escalation after 5s.
- **Single corrupted session:** `teane purge --session-id <id>` removes only that thread's checkpoints AND its JSONL log + rotated backups; other sessions are unaffected.
- **Corrupted checkpoint DB across the board:** `teane purge --all` wipes and recreates; sessions are lost but the workspace is untouched.
- **Stale workspace lock from a crashed prior session:** `teane run -r <ws> -p '...' --force-lock` releases the stale lock and acquires a fresh one (operator confirms the prior PID is gone). See `docs/RUNBOOK.md` § 4.
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
- Session introspection via `teane status` without graph execution
- Build output captured in full (stdout + stderr) via disk log streamer

### Maintainability
- TypedDict schemas for compile-time type safety across nodes (Pydantic was evaluated and removed — see `SPEC_ARCHITECTURE.md` §5.8)
- All nodes are isolated async functions with explicit state → state contracts
- Gateway providers implement a common interface for easy addition of new providers
- Diagnostic parsers are registered via a plugin registry (parser_registry.py)
- Skills are registered via a singleton SkillRegistry for extensibility
- Configuration is externalized to JSON files (no hardcoded model names or API keys in source)