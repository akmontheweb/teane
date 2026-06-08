# AI Agent Harness — Requirements Specification

*Auto-generated from exhaustive codebase analysis of the harness itself.*

---

## 1. Executive Summary

AI Agent Harness is a production-grade, model-agnostic autonomous coding agent built on LangGraph. It accepts natural language engineering tasks, generates precise code patches via LLMs, verifies them through sandboxed builds, and deploys containerized applications — all under budget guardrails, security scanning, and git lifecycle management. The system supports exhaustive multi-phase discovery (requirements → architecture → deployment), human-in-the-loop intervention points, checkpoint-based crash recovery, and cross-model speculative repair escalation.

---

## 2. Functional Requirements (FR)

### FR-001: CLI Subcommand Routing
- **Description:** The system MUST provide a `harness` CLI with subcommands `run`, `resume`, `status`, and `purge`, each with their own argument parsers and help text.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given `harness -h`, the system displays help with all subcommands listed.
  - Given `harness run -h`, the system displays run-specific help with all flags documented.

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
- **Description:** The system MUST execute the project's build command inside an isolated sandbox (Linux namespaces via `unshare`, Docker container, or bare subprocess). Build output MUST be parsed for structured diagnostics.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given `build_command: "make build"`, the command runs inside a sandbox and returns exit code + diagnostics.
  - Given a compilation error in Go/Rust/C/Python format, structured `DiagnosticObject` dicts are extracted.
  - Given a timeout of 300 seconds, builds exceeding the limit are killed with PGID-based process group termination.

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
- **Description:** When the repair limit is hit or budget is exhausted, the system MUST present an interactive stdin menu with options: view diffs, resume, inject hint, pause for manual edits, increase budget, or abandon with git rollback.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given HITL triggered, a menu with [v/r/e/m/b/q] options is displayed.
  - Given user selects [b] (increase budget), `budget_remaining_usd` increases by $2.00 and the menu re-displays.
  - Given user selects [q] and confirms, `git checkout -- .` is executed and the session ends.

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
- **Description:** The system MUST persist graph state to a SQLite database (WAL mode) at every node transition. `harness resume --session-id` MUST restore and continue from the last checkpoint.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given a running graph, checkpoints are written to `~/.harness/checkpoints.db`.
  - Given `harness resume --session-id <id>`, the graph resumes from the checkpointed state.
  - Given a non-existent session ID, resume exits with error code 1.

### FR-017: Read-Only Status Inspection
- **Description:** `harness status --all` MUST list all checkpointed sessions with session ID, created time, updated time, and workspace path. `harness status --session-id <id>` MUST display a full state snapshot.
- **Priority:** Must Have
- **Acceptance Criteria:**
  - Given `harness status --all`, a table with SESSION ID, UPDATED, CREATED, and WORKSPACE columns is printed.
  - Given `harness status --session-id <id>`, a detailed state dump with all fields is printed.
  - Given a non-existent session ID, a "not found" message is printed.

### FR-018: Session Data Purging
- **Description:** `harness purge --all` MUST delete all checkpoint data after confirmation. `harness purge --session-id <id>` MUST delete a specific session's checkpoints.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given `harness purge --all` and user confirms "yes", all rows in the checkpoints DB are deleted.
  - Given `harness purge --session-id <id>`, only that thread's checkpoints are deleted.

### FR-019: Lint Gate (Deterministic Format Verification)
- **Description:** Before each build, modified files MUST be auto-formatted and linted using language-specific tools (ruff, gofmt, prettier, rustfmt, clang-format). Lint errors are surfaced in the build output.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given modified `.py` files, ruff format + ruff check are executed.
  - Given modified `.go` files, gofmt is executed.
  - Given no matching formatter for a file extension, it is skipped.

### FR-020: Multi-Variant Speculative Execution
- **Description:** After patching, the system MAY generate N parallel code variants, compile each in isolated git worktrees, and select the winner by first_success, fewest_changes, or all_pass strategy.
- **Priority:** Could Have
- **Acceptance Criteria:**
  - Given `speculative.enabled: true` in config, N variants are generated in parallel.
  - Given one variant compiles successfully and others fail, the successful variant is selected.
  - Given all variants fail, the system falls back to the original patching flow.

### FR-021: Container Deployment
- **Description:** After successful build and security scan, the system MUST scan workspace telemetry, synthesize a deployment architecture blueprint, generate Dockerfiles + docker-compose.yml + Caddyfile, build containers, and run health checks.
- **Priority:** Should Have
- **Acceptance Criteria:**
  - Given a Python workspace with `requirements.txt`, a Python Dockerfile is generated.
  - Given the deployment blueprint, `docker-compose up --build -d` is executed.
  - Given containers are running, health check polling confirms readiness within 30s.

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

---

## 3. System Scope

### In-Scope
- CLI interface with 4 subcommands (run, resume, status, purge)
- LangGraph-based agent graph with 20+ nodes
- Multi-provider LLM gateway (DeepSeek, Anthropic, OpenAI, Ollama)
- Hierarchical JSON configuration with deep merge
- SEARCH/REPLACE patch application with AST-aware fallback
- Sandboxed build execution (unshare, Docker, bare)
- Structured diagnostic parsing for Go, Rust, GCC/Clang, Python
- Cross-model speculative repair escalation (cheap → expensive)
- Human-in-the-loop interactive menu with 6 actions
- Zero-knowledge secret redaction before all API calls
- Git branch lifecycle management (stash, patch branch, commit, rollback)
- Exhaustive 3-phase discovery pipeline with structured Q&A loops
- Pre-flight manifest → spec synthesis with interactive review
- SQLite checkpoint persistence with WAL mode and 30-day TTL GC
- Read-only session status inspector with timestamp and workspace display
- Lint gate with auto-detected formatters per language
- Multi-variant speculative compilation in parallel git worktrees
- Container deployment pipeline (telemetry → blueprint → Dockerfile → compose → health check)
- Post-build security scanning (gitleaks + bandit/semgrep)
- Conversation memory cleanse for prefix-cache optimization
- Dependency graph impact analysis
- Two-tier skills system (harness-level + project-level markdown conventions)

### Out-of-Scope
- Interactive IDE plugin or VS Code extension
- Web-based dashboard or GUI
- Multi-user concurrent session management
- Cloud-hosted SaaS offering
- Non-Git version control systems (Mercurial, SVN)
- Windows sandbox backends (only Linux supported)
- Real-time streaming collaboration
- Built-in code review or PR management
- Training or fine-tuning of LLMs

---

## 4. Technical Constraints

### Language and Runtime
- **Language:** Python 3.11+
- **Async Model:** asyncio with `async/await` throughout
- **Type System:** TypedDict for LangGraph compatibility, Pydantic for runtime validation
- **Package Manager:** pip + pyproject.toml

### Key Dependencies
| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| langgraph | 0.4.0 | Stateful graph execution with checkpointing |
| langgraph-checkpoint-sqlite | 2.0.0 | SQLite persistence backend |
| aiofiles | 24.0.0 | Async file I/O |
| tree-sitter | 0.23.0 | AST-aware code manipulation |
| httpx | 0.28.0 | Async HTTP client for LLM API calls |
| pydantic | 2.10.0 | Runtime validation and serialization |
| msgpack | 1.1.0 | Deserialization of checkpoint BLOBs |
| uuid7 | 0.1.0 | Time-sortable UUID generation |
| typing-extensions | 4.12.0 | TypedDict and type hint backports |

### Platform Requirements
- **OS:** Linux (namespace isolation requires Linux kernel 4.0+)
- **Optional:** Docker daemon for container-based sandbox and deployment
- **Disk:** ~10MB for checkpoint database per 30-day window
- **Network:** Outbound HTTPS required for LLM API calls (unless using Ollama local-only)

### Performance Targets
- CLI startup (config discovery): < 100ms
- Checkpoint read (single session): < 50ms
- Build sandbox startup (unshare): < 500ms
- Build sandbox startup (Docker): < 2s
- LLM dispatch with retry: < 30s per call (exponential backoff with jitter)

### Security Requirements
- No secrets in outbound API calls (pre-transmission redaction)
- No secrets in checkpoints database (values are in LangGraph state, not separately encrypted)
- Sandbox network isolation by default (no outbound network unless `--allow-network`)
- PGID-based process group termination on timeout (no orphaned child processes)
- Git rollback on session abandonment (no unrecoverable workspace corruption)

---

## 5. Explicit Edge Cases

### Error States
- **LLM API unreachable:** Exponential backoff with jitter (3 retries). Gateway logs failure and returns error to calling node. Graph routes to HITL if all retries fail.
- **Manifest file empty:** `synthesize_requirements()` raises `RuntimeError("Manifest file is empty.")`.
- **Workspace path does not exist:** CLI exits with code 1 before any graph execution.
- **No checkpointer configured:** Graph runs ephemerally with `MemorySaver`; no crash recovery.
- **Config JSON malformed:** Error logged with exact file path; harness refuses to proceed.
- **Build command not found:** Sandbox returns exit code 127; parsed as a generic diagnostic.
- **Docker daemon unreachable:** `create_backend("auto")` falls back from Docker → unshare → bare.
- **gitleaks not installed:** Security scan falls back to Python regex-based secret scanner.
- **msgpack module missing:** `_deserialize_checkpoint_blob()` falls back to JSON text decoding for legacy rows.

### Boundary Conditions
- **Max repair iterations:** 3 (hardcoded in `route_after_compiler`)
- **Max security fix attempts:** 2 (hardcoded in `route_after_security_scan`)
- **Default budget cap:** $2.00 USD
- **Sandbox timeout:** 300 seconds (configurable via `sandbox.timeout_seconds`)
- **Checkpoint TTL:** 30 days (configurable via `persistence.ttl_days`)
- **Max files per directory in tree snapshot:** 50
- **Max directory depth in tree snapshot:** 4
- **Max skills file chars:** 4000 (harness) / 3000 (project)
- **HITL raw build output display:** Last 2000 characters
- **Repair prompt raw output fallback:** Last 2000 characters
- **Token budget context window threshold:** 85% (truncation trigger)
- **Disk log buffer max size:** 500MB

### Recovery Scenarios
- **Process killed mid-graph:** Next `harness run` loads from latest checkpoint; LangGraph replays from the boundary.
- **Network timeout during LLM call:** Gateway retries with exponential backoff + jitter (up to 3 attempts).
- **Build timeout in sandbox:** PGID-based `kill(-pgid, SIGKILL)` → `SIGTERM` escalation after 5s.
- **Corrupted checkpoint DB:** `harness purge --all` wipes and recreates; sessions are lost but workspace is untouched.
- **Git stash conflict:** `git stash pop` may fail if stash conflicts; harness logs warning and continues (working tree is in the patch branch state).

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
- TypedDict + Pydantic dual schema for compile-time and runtime type safety
- All nodes are isolated async functions with explicit state → state contracts
- Gateway providers implement a common interface for easy addition of new providers
- Diagnostic parsers are registered via a plugin registry (parser_registry.py)
- Skills are registered via a singleton SkillRegistry for extensibility
- Configuration is externalized to JSON files (no hardcoded model names or API keys in source)