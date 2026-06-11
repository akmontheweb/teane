# AI Agent Harness — Architecture Specification

*Refreshed from current codebase state. Companion to `SPEC_REQUIREMENTS.md`.*

---

## 1. System Context (C4 Level 1)

AI Agent Harness sits between the developer and their codebase, acting as an autonomous engineering agent. It accepts natural language prompts, generates code patches, verifies them via sandboxed builds, and applies them to the workspace — all under budget and security guardrails.

```
┌──────────────┐     ┌─────────────────────────────────────┐     ┌──────────────┐
│              │     │                                     │     │              │
│  Developer   │────▶│       AI Agent Harness              │────▶│   Git Repo   │
│  (CLI/IDE)   │     │  (LangGraph Agent + Sandbox)        │     │  (Workspace) │
│              │◀────│                                     │◀────│              │
└──────────────┘     └──────────────┬──────────────────────┘     └──────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
             ┌──────────┐   ┌──────────┐   ┌──────────────┐
             │ DeepSeek │   │ Anthropic│   │ Ollama (Local)│
             │   API    │   │ (Claude) │   │              │
             └──────────┘   └──────────┘   └──────────────┘
                    │               │               │
                    ▼               ▼               ▼
             ┌─────────────────────────────────────────┐
             │          LLM Gateway (harness/gateway)   │
             │  - Model routing by NodeRole             │
             │  - Budget enforcement                    │
             │  - Secret redaction before transit        │
             │  - Context window guardrail              │
             │  - Exponential backoff + jitter          │
             └─────────────────────────────────────────┘
```

**External Systems:**
- **DeepSeek API** — Primary cheap model for patching (OpenAI-compatible `/v1/chat/completions`)
- **Anthropic API** — Reasoning/fallback model for repair escalation (`/v1/messages`)
- **OpenAI API** — Optional provider (`/v1/chat/completions`)
- **Ollama** — Local inference server, zero-cost fallback, used when budget is low or `force_local_only` is set

---

## 2. Container Diagram (C4 Level 2)

The harness is a single-process Python application with these deployable/service boundaries:

```
┌────────────────────────────────────────────────────────────────────┐
│                       HARNESS CLI PROCESS                          │
│                                                                    │
│  ┌───────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │   CLI Layer        │  │  Persistence      │  │  Git Lifecycle │  │
│  │  (harness/cli.py)  │  │  (harness/storage)│  │  (harness/     │  │
│  │                    │  │                   │  │   security.py) │  │
│  │  - Argparse        │  │  - AsyncSqliteSaver│  │                │  │
│  │  - Config discovery│  │  - Checkpoint CRUD │  │  - Patch branch│  │
│  │  - HITL menus      │  │  - 30-day TTL GC  │  │  - Stash/dirty │  │
│  │  - Subcommand routing│ │  - Status inspector│  │  - Commit/     │  │
│  └─────────┬─────────┘  └────────┬─────────┘  │    rollback    │  │
│            │                     │             └────────────────┘  │
│            ▼                     ▼                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    LangGraph Runtime                         │   │
│  │                   (harness/graph.py)                         │   │
│  │                                                              │   │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │   │
│  │  │ Planning │──▶│ Patching │──▶│  Lint    │──▶│ Compiler │ │   │
│  │  │   Node   │   │   Node   │   │  Gate    │   │   Node   │ │   │
│  │  └──────────┘   └──────────┘   └──────────┘   └────┬─────┘ │   │
│  │                                                     │       │   │
│  │                              ┌──────────────────────┼───┐   │   │
│  │                              │    exit 0?           │   │   │   │
│  │                              │  yes         no      │   │   │   │
│  │                              ▼              ▼       │   │   │   │
│  │                       ┌──────────┐   ┌──────────┐  │   │   │
│  │                       │ Security │   │  Repair  │  │   │   │
│  │                       │   Scan   │   │   Node   │  │   │   │
│  │                       └────┬─────┘   └────┬─────┘  │   │   │
│  │                            │              │         │   │   │
│  │                            ▼              ▼         │   │   │
│  │                       ┌──────────┐   ┌──────────┐  │   │   │
│  │                       │ Deploy   │   │   HITL   │  │   │   │
│  │                       │   Node   │   │   Node   │  │   │   │
│  │                       └────┬─────┘   └──────────┘  │   │   │
│  │                            │                        │   │   │
│  │                            ▼                        │   │   │
│  │                         [END]                       │   │   │
│  └─────────────────────────────────────────────────────┘   │   │
│                                                              │   │
│  ┌──────────────────────────────────────────────────────────┐│   │
│  │              Discovery Pipeline (Three-Phase)             ││   │
│  │  requirements → interview → write_spec → gatekeeper →     ││   │
│  │  architecture → interview → write_spec → gatekeeper →     ││   │
│  │  deployment → interview → write_spec → gatekeeper → END   ││   │
│  └──────────────────────────────────────────────────────────┘│   │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Diagram (C4 Level 3)

### 3.1 Module Decomposition

```
harness/
├── __init__.py           # Package init, version, __all__
├── cli.json              # Fallback defaults (shipped config)
├── cli.py                # CLI entry, subcommand routing (run / resume / status / doctor /
│                         # purge / metrics), HITL menus, config discovery, doctor healthchecks
│   ├── _get_harness_version()   # importlib.metadata lookup feeding --version / -V
│   ├── _acquire_workspace_lock()# fcntl single-writer lock (FR-038); --force-lock override
│   ├── discover_config()        # Hierarchical merge: workspace → home → cli.json
│   ├── _validate_config_keys()  # Recursive top-level + nested typo detection (FR-030)
│   ├── cmd_run / cmd_resume / cmd_status / cmd_purge / cmd_metrics
│   ├── cmd_doctor()             # 6-check healthcheck (FR-025)
│   ├── _doctor_check_git / global_config / api_keys / sandbox / checkpoint_db / config
│   ├── _ping_provider_live()    # 1-token chat call per provider (PASS only after auth confirmed)
│   ├── human_gatekeeper_node()  # Three-phase HITL gatekeeper
│   ├── hitl_menu_loop()         # 7-action HITL menu: [v/r/e/m/b/s/q]
│   ├── _emit_output()           # Routes machine-readable metrics to file / stdout
│   └── interactive_review_loop()# Pre-flight manifest review
├── gateway.py            # Model-agnostic LLM Gateway
│   ├── GatewayConfig     # Runtime config dataclass (incl. max_discovery_iterations)
│   ├── Gateway           # Orchestrator: dispatch + budget + retry + circuit breaker
│   │   ├── _preflight_budget_check() # Raises BudgetTooLowError before any HTTP call (FR-035)
│   │   ├── _circuit_is_open()        # 3-in-5min rate-limit breaker → divert to Ollama (FR-037)
│   │   └── _retry_empty_response()   # Up to 2 retries on empty content (FR-036)
│   ├── BudgetTooLowError / EmptyLLMResponseError  # Pre-flight + empty-content exceptions
│   ├── DeepSeekProvider  # OpenAI-compatible /v1/chat/completions
│   ├── AnthropicProvider # Claude /v1/messages with system prompt extraction
│   ├── OpenAIProvider    # Standard /v1/chat/completions
│   ├── OllamaProvider    # Local inference, free
│   ├── BaseProviderClient# api_key resolution: explicit arg → env var → spec.api_key
│   ├── retry_with_backoff()  # Exponential backoff + jitter
│   └── check_context_window() # 85% threshold truncation
├── graph.py              # LangGraph StateGraph topology
│   ├── AgentState        # TypedDict state schema
│   ├── planning_node()   # LLM: generate implementation blueprint
│   ├── patching_node()   # LLM: generate SEARCH/REPLACE patches
│   ├── compiler_node()   # Deterministic: run build in sandbox
│   ├── repair_node()     # LLM: analyze errors, fix, escalate to fallback model
│   ├── human_intervention_node() # Set HITL flags
│   ├── requirements_discovery_node() # LLM: 8-sector requirements discovery
│   ├── architecture_discovery_node() # LLM: 8-sector architecture discovery
│   ├── deployment_discovery_node()   # LLM: 4-sector deployment discovery
│   ├── write_spec_node() # Serialize discovery to .md files
│   ├── generate_deployment_spec_node() # Produce DEPLOYMENT_BLUEPRINT.md
│   ├── route_after_compiler()    # Conditional: repair / HITL (short-circuits on llm_silent) / security_scan
│   ├── route_after_discovery()   # Conditional: write_spec / discovery loop (capped at max_discovery_iterations, FR-043)
│   ├── route_after_gatekeeper()  # Conditional: next phase / refinement loop
│   ├── route_after_security_scan() # Conditional: patch / HITL / deployment
│   ├── route_after_hitl()        # Conditional: compiler / END
│   ├── _build_patcher_allowlist()# Conservative fallback when source root unclear (FR-041)
│   ├── _apply_toolchain_adaptation() # pip/npm network auto-enable gated by config opt-in (FR-042)
│   ├── apply_memory_cleanse()    # Compress verbose repair messages
│   ├── build_graph()             # Assemble full StateGraph
│   └── run_graph()               # Async entry point
├── sandbox.py            # Sandbox execution engine
│   ├── SandboxBackend    # ABC for isolation backends
│   ├── UnshareBackend    # Linux namespace isolation
│   ├── DockerBackend     # Docker container isolation
│   ├── BareBackend       # No isolation (fallback)
│   ├── SandboxExecutor   # Orchestrator
│   ├── DiskLogStreamer   # Temp-file buffered log I/O
│   ├── MemoryLogStreamer # In-memory log accumulator
│   ├── filter_critical_errors()  # Regex log interceptor
│   ├── _execute_subprocess_with_timeout() # PGID-managed subprocess
│   └── extract_diagnostics()     # Multi-language diagnostic parser
├── patcher.py            # Hybrid file modification engine
│   ├── PatchBlock        # Parsed patch instruction
│   ├── PatchResult       # Operation result
│   ├── TextPatcher       # Exact-match SEARCH/REPLACE
│   ├── TreeSitterPatcher # AST-aware rewriting
│   ├── HybridPatcher     # Auto-selects best strategy
│   ├── _awrite()         # Symlink guard: refuses os.path.islink targets + O_NOFOLLOW (FR-041)
│   ├── parse_patch_blocks() # Extract blocks from LLM text
│   └── process_llm_patch_output() # Primary integration point
├── security.py           # Lifecycle & security
│   ├── GitGuardian       # Branch creation, commit, rollback
│   ├── CommandValidator  # Whitelist/blocklist command scanner
│   ├── set_command_validator / get_command_validator # Process-wide singleton (FR-034)
│   ├── create_command_validator_from_config()  # Built once in cmd_run / cmd_resume
│   ├── HITLGate          # Pre-execution sensitive operation confirmation
│   └── security_scan_node() # SAST + secret scanning gatekeeper
├── storage.py            # Checkpoint persistence
│   ├── HarnessAsyncSqliteSaver # LangGraph checkpointer with message redaction +
│   │                            # schema-version stamping on every aput / aput_writes
│   ├── CHECKPOINT_SCHEMA_VERSION / SCHEMA_VERSION_METADATA_KEY  # Version stamp (FR-016, FR-033)
│   ├── CheckpointCorruptedError / CheckpointSchemaMismatchError # Surfaced by cmd_resume pre-flight
│   ├── validate_checkpoint_schema() # Refuses future versions; warns on legacy unstamped blobs
│   ├── _deserialize_checkpoint_blob(strict=) # Strict mode raises instead of returning {}
│   ├── CheckpointSummary # Read-only state snapshot
│   ├── generate_session_id()
│   ├── inspect_session() # Read-only status inspector
│   └── list_all_sessions()
├── deploy.py             # Containerization & deployment
│   ├── scan_workspace_telemetry() # Deterministic workspace scanner
│   ├── synthesize_architecture()  # LLM: JSON blueprint → compose
│   ├── generate_assets_from_blueprint() # Dockerfile, compose, Caddyfile
│   ├── health_check_loop() # docker inspect polling
│   └── deployment_node() # Phase orchestrator
├── lintgate.py           # Deterministic format verification
│   ├── FormatterSpec     # Tool command spec
│   ├── lintgate_node()   # Pre-build format + lint runner
│   └── _resolve_path()   # Workspace-relative path resolution
├── redactor.py           # Zero-knowledge secret scanner
│   ├── SecretScanner     # Regex + entropy-based detection
│   ├── RedactionResult   # Replacement stats
│   ├── redact_text()     # String redaction
│   └── redact_messages() # Message list redaction
├── speculative.py        # Multi-variant compilation
│   ├── VariantResult     # Per-variant compilation result
│   ├── SpeculativeResult # Aggregate speculation result
│   ├── speculate_node()  # N-variant parallel compilation
│   └── _select_winner()  # first_success / fewest_changes / all_pass
├── impact.py             # Semantic dependency graph
│   ├── DependencyGraph   # Cross-file dependency scanner
│   ├── ImpactAnalyzer    # Pre-patch impact checker
│   └── ImpactResult      # Warning + impacted files
├── skills.py             # Unified skill registry
│   ├── SkillBase         # ABC for all skill types
│   ├── ToolSkill         # LLM-invokable function
│   ├── PipelineSkill     # LangGraph node wrapper
│   ├── SubAgentSkill     # Autonomous mini-agent
│   ├── DocGenSkill       # Documentation sub-agent
│   └── SkillRegistry     # Global singleton
│   # NOTE: stack-aware skill filtering (FR-027) lives in
│   # graph.py:_parse_skill_frontmatter() — it reads the
│   # `applies_to:` YAML frontmatter on harness/skills/*.md
│   # files and intersects against the workspace tag set
│   # before loading skills into the prompt.
├── hitl.py               # Pluggable HITL transport (FR-009)
│   ├── HitlChannel       # ABC: prompt / notes / confirm / wait_for_manual_edit
│   ├── StdinChannel      # Default — interactive terminal
│   ├── FileChannel       # Read prompts/answers from JSONL files
│   ├── HttpChannel       # POST prompts to a webhook; receive answers as JSON
│   ├── get_channel / set_channel / reset_channel  # Process-wide singleton
│   └── _auto_approve()   # CI / HARNESS_AUTO_APPROVE / non-TTY auto-approve
├── observability.py      # Structured logging + JSONL session events
│   ├── JSONFormatter     # One JSON object per log line, with `extra=` merge
│   ├── configure_logging()  # Stderr + per-session JSONL (RotatingFileHandler,
│   │                          10 MB × 5 backups default; max_bytes=0 disables) + LangSmith (FR-040)
│   ├── emit_event()      # INFO-level structured event (successful / observational)
│   └── log_failure()     # ERROR-level structured failure event (FR-029)
│                         # Catalogue: sandbox_start_failed, token_budget_exhausted,
│                         #            hitl_gate_blocked, llm_empty_response, llm_circuit_open.
├── trust.py              # Workspace boundary enforcement + structured-output trust
│   ├── safe_resolve()           # Block path traversal outside workspace_root
│   ├── is_path_allowed()
│   ├── is_valid_docker_image / service_name / env_var_name / port_mapping
│   ├── validate_blueprint()     # Deploy blueprint schema check
│   ├── validate_discovery_json()# Discovery-LLM trust gate: 1 MB byte cap + depth-10 walk (FR-039)
│   ├── validate_blueprint_json()# Deploy-LLM output trust gate
│   ├── validate_synthesized_spec()  # Manifest-synthesis trust gate (Bug 7 closure)
│   └── safe_subprocess_env()    # Scrub envrionment passed to sandbox subprocess
├── metrics.py            # Cost-metrics aggregation for `harness metrics` (FR-032)
│   ├── SessionMetrics    # @dataclass: per-session cost / tokens / errors / burn rate
│   ├── parse_jsonl_file()        # Tolerant line-by-line JSONL reader
│   ├── _sorted_session_log_files()  # Live <id>.jsonl + rotated .N backups in chronological order
│   ├── aggregate_session()       # Sum cost/tokens, count failures, compute window burn rate
│   ├── list_sessions()           # Distinct session IDs from filenames (dedups rotation suffix)
│   ├── project_exhaustion()      # Minutes until hard_cap_usd at current burn rate
│   ├── format_human / format_table / format_prometheus
│   └── write_atomic()    # <dest>.tmp → fsync → os.replace, atomic from a scraper's POV
└── parser_registry.py    # Diagnostic parser plugins (FR-026)
    ├── RustParser        # --error-format=json
    ├── GccClangParser    # -fdiagnostics-format=json
    ├── GoParser          # file:line:col: message
    ├── PythonParser      # Traceback extraction
    ├── JavaParser        # javac / maven / gradle diagnostic shapes
    ├── TypeScriptParser  # tsc / eslint
    ├── DartParser        # dart analyze / flutter build
    ├── GenericParser     # file:line:col: severity: message
    ├── register_parser / register_extension_parser
    ├── get_parser / get_parser_for_extension / list_registered_parsers
    └── detect_and_parse() # Auto-detect + parse
```

### 3.2 Data Flow

```
1. User prompt + workspace → CLI
                              │
                              ▼
2. Config discovery (+ models, routing, budget, sandbox)
                              │
                              ▼
3. GitGuardian: stash dirty, create patch branch
                              │
                              ▼
4. SecretScanner: register global redactor
                              │
                              ▼
5. Gateway: register models from config, create Gateway instance
                              │
                              ▼
6. run_graph() → create_initial_state()
                              │
                              ▼
        ┌─────────────────────────────────────────────────┐
        │         EXHAUSTIVE DISCOVERY PIPELINE           │
        │                                                 │
        │  requirements_discovery_node                    │
        │         │                                       │
        │         ▼                                       │
        │  discovery_interview_loop (CLI stdin)           │
        │         │                                       │
        │         ▼                                       │
        │  route_after_discovery → write_spec_node        │
        │         │                                       │
        │         ▼                                       │
        │  human_gatekeeper_node (approve/refine/manual)  │
        │         │                                       │
        │         ▼ (approve)                             │
        │  architecture_discovery_node                    │
        │         │                                       │
        │    [same loop as above]                         │
        │         │                                       │
        │         ▼ (approve gatekeeper)                  │
        └─────────┼───────────────────────────────────────┘
                  │
                  ▼
7. planning_node → LLM (planning_primary, thinking mode)
                  │
                  ▼
8. patching_node → LLM (patching_primary, non-thinking)
                  │
                  ▼
9. speculate_node → N LLM calls (temp>0) → parallel worktrees → select winner
                  │
                  ▼
10. lintgate_node → ruff/gofmt/prettier/rustfmt on modified files
                  │
                  ▼
11. compiler_node → SandboxExecutor → backend.run(build_command)
                  │
           ┌──────┴──────┐
           │ exit 0       │ exit ≠ 0
           ▼              ▼
12. security_scan_node   13. repair_node → LLM (repair_primary, thinking)
    │                          │
    │ clean? ──yes──▶          ├── lintgate_node
    │                          │
    │ findings?                ├── compiler_node (re-verify)
    │   │                      │
    │   ▼                      │  repairs < 3 → loop to repair_node
    │ patching_node            │  repairs >= 3 → human_intervention_node
    │   │                      │                 │
    │   ▼                      │                 ├── [hint] → repair_node
    │ lintgate → compiler      │                 ├── [manual] → compiler_node
    │                          │                 ├── [resume] → compiler_node
    │                          │                 └── [abandon] → END
    │                          │
    ▼                          │
14. Flutter detected? ─yes─▶ [END]  (FR-028 — mobile builds bypass docker compose)
    │ no                       │
    ▼                          │
15. deployment_discovery_node │
    │                          │
    ▼                          │
16. generate_deployment_spec  │
    │                          │
    ▼                          │
17. human_gatekeeper (DEPLOY) │
    │ (approve)                │
    ▼                          │
18. deployment_node            │
    ├── scan_workspace_telemetry()
    ├── synthesize_architecture()
    ├── generate_assets_from_blueprint()
    ├── docker compose up --build -d   # V2 syntax, no hyphen
    └── health_check_loop()
        │
        ▼
      [END]

Independent of the graph, `harness doctor` reuses the same config-
discovery + checkpoint-DB code paths to run five healthchecks (git
repo, API keys per routed provider, sandbox backend reachable,
checkpoint DB writable, config parses cleanly) and reports
PASS/WARN/FAIL with colored markers.
```

### 3.3 State Mutation per Node

```
AgentState fields and which nodes write to them:

┌──────────────────────────┬──────────────────────────────────────────────┐
│ Field                    │ Written By                                   │
├──────────────────────────┼──────────────────────────────────────────────┤
│ messages                 │ planning_node, patching_node, repair_node,    │
│                          │   lintgate_node, security_scan_node,          │
│                          │   deployment_node, discovery_nodes            │
│ modified_files           │ patching_node, repair_node,                   │
│                          │   process_llm_patch_output()                  │
│ compiler_errors          │ compiler_node, security_scan_node,            │
│                          │   deployment_node                             │
│ token_tracker            │ planning_node, patching_node, repair_node     │
│ loop_counter             │ ALL nodes (increment their counter)           │
│ budget_remaining_usd     │ planning_node, patching_node, repair_node     │
│ exit_code                │ compiler_node                                 │
│ node_state               │ ALL nodes (metadata + routing signals)        │
│ current_gate             │ requirements_discovery, architecture_discovery│
│                          │   deployment_discovery, generate_deployment   │
│ spec_requirements_path   │ write_spec_node                               │
│ spec_architecture_path   │ write_spec_node                               │
│ deployment_blueprint_path│ generate_deployment_spec_node                 │
└──────────────────────────┴──────────────────────────────────────────────┘
```

---

## 4. Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **Orchestration** | LangGraph ≥ 0.4.0 | Stateful graph execution with checkpointing; typed state schema |
| **Language** | Python 3.11+ (CI: 3.11 / 3.12 / 3.13) | TypedDict, asyncio improvements, `None`-aware operators |
| **Persistence** | aiosqlite + WAL mode | Crash-safe, zero-config, survives reboots; WAL for concurrent reads |
| **File I/O** | aiofiles ≥ 24.0 | Non-blocking disk ops with sync fallback for missing dep |
| **AST Parsing** | tree-sitter + tree-sitter-language-pack ≥ 1.8 | Single wheel covering 165+ grammars (Python / Java / JS / TS / TSX / Dart / Rust / Go / Swift / …); replaces six individual grammar packages and gives us Dart coverage that has no standalone PyPI distribution |
| **HTTP Client** | httpx ≥ 0.28 | Async HTTP/2 with connection pooling and timeout management |
| **Config** | JSON (discovered hierarchically) | Workspace `.harness_config.json` → `~/.harness/config.json` → `cli.json` |
| **Testing** | pytest + pytest-asyncio | Async test support, fixture injection, coverage |
| **CI** | GitHub Actions matrix | `pytest tests/ -q --tb=short` across Python 3.11 / 3.12 / 3.13 on `ubuntu-latest` (blocking) + Python 3.12 on `macos-latest` and `windows-latest` (advisory `continue-on-error`); separate `quality` job runs `ruff check` (blocking) plus `ruff format` and `mypy` (advisory). |
| **Pre-commit** | pre-commit + local pytest hook + ruff | Same suite + lint gate runs locally as in CI; bypassable with `--no-verify` for emergencies only |
| **Linting** | ruff ≥ 0.8 | Fast Python linter + formatter; `ruff check harness/ tests/` is the blocking CI gate |
| **Type Checking** | mypy ≥ 1.13 (strict mode) | TypedDict validation; advisory in CI pending typing backlog |
| **Pinned install (pilot)** | `requirements-prod.txt` | Exact transitive pins for reproducible installs: `pip install -e . --constraint requirements-prod.txt` |
| **License** | MIT (`LICENSE` at repo root) | `pyproject.toml` references the file so the wheel ships it and GitHub auto-detects |
| **Sandbox (primary)** | Docker CLI | Strongest isolation, built-in resource limits; preferred by `backend: "auto"` |
| **Sandbox (fallback)** | Linux unshare(2) | Kernel namespace isolation without Docker dependency |
| **Sandbox (opt-in)** | bare subprocess | Zero isolation; opt-in via `HARNESS_ALLOW_UNSAFE_SANDBOX=true` for environments where neither Docker nor user-namespaces are available |
| **Secrets** | SHA-256 hashing | Stable hash for traceability without exposing values |
| **Release** | `make release` + `scripts/release.py` | SemVer bump → CHANGELOG roll → tag → push; refuses dirty trees and off-`main` runs |

**Dependency Versions (pyproject.toml):**
```
# runtime
langgraph>=0.4.0
langgraph-checkpoint-sqlite>=2.0.0
aiofiles>=24.0.0
tree-sitter>=0.23.0
tree-sitter-language-pack>=1.8.0
httpx>=0.28.0
uuid7>=0.1.0
typing-extensions>=4.12.0

# dev (extras = "dev")
pytest>=8.0.0
pytest-asyncio>=0.24.0
ruff>=0.8.0
mypy>=1.13.0
pre-commit>=3.7.0
msgpack>=1.0.0          # storage GC regression test; runtime falls back to JSON if missing
```

---

## 5. Key Design Decisions

### 5.1 Hybrid Patcher: AST-Aware with Text Fallback

**Decision**: Use tree-sitter for AST-level structural patching on supported languages, with exact-match text SEARCH/REPLACE as universal fallback.

**Rationale**: Pure text-based SEARCH/REPLACE fails on whitespace/indentation drift between LLM-generated patches and actual files. AST-aware patching locates nodes by structural signature and replaces only the node's bytes, preserving all surrounding code. The text fallback ensures the system works on any file type without tree-sitter grammars installed.

**Trade-off**: Tree-sitter adds a native dependency. The fallback to TextPatcher is automatic and transparent, so the system degrades gracefully.

### 5.2 Disk-Buffered Log Streaming

**Decision**: Stream build output to NamedTemporaryFiles on disk rather than accumulating in memory.

**Rationale**: Large builds (C++, Rust) can produce gigabytes of output. Disk-buffered mode keeps RAM usage constant. The `DiskLogStreamer` enforces a 500MB max size limit, writes stdout/stderr to separate temp files, and reads back via line-by-line iteration. Temp files are auto-cleaned after execution.

**Trade-off**: Slightly higher latency for small builds due to disk I/O. In-memory `MemoryLogStreamer` is available as an alternative via `log_buffer_mode: "memory"`.

### 5.3 Cross-Model Speculative Repair Escalation

**Decision**: Repair attempts 1-2 use the cheap primary model; repair attempt 3 escalates to the expensive fallback model with thinking mode.

**Rationale**: Most compilation errors are simple (missing import, wrong type) and the cheap model can fix them. Only the hardest problems warrant the reasoning model's higher cost. This saves 60-80% of repair costs vs always using the expensive model.

**Trade-off**: Adds complexity to `repair_node` with temporary config mutation + restore in a `finally` block.

### 5.4 Exhaustive Zero-Unknowns Discovery

**Decision**: Before any code is generated, the planning LLM cross-examines the developer across 8 structured sectors (requirements) + 8 technical sectors (architecture) + 4 deployment sectors, each with follow-up loops and critical/unknown tracking.

**Rationale**: LLMs produce better code when given exhaustive context. The multi-phase discovery eliminates ambiguous requirements before patches are generated, reducing downstream repair loops.

**Trade-off**: Adds significant pre-generation latency and LLM token cost. Discovery is **off by default**; opt in with `--discover` on greenfield projects or when working from a blank workspace. The legacy `--skip-discovery` flag remains as a hidden no-op alias for scripts.

### 5.5 Secret Redaction Before Every API Call

**Decision**: All outbound LLM messages pass through `SecretScanner.redact_messages()` before transmission. The redactor uses 15+ high-confidence regex patterns plus entropy analysis for unknown token formats.

**Rationale**: Developers may accidentally include API keys, tokens, or credentials in their prompts or code context. The redactor acts as a safety net, preventing secrets from ever leaving the local machine.

**Trade-off**: Regex-based detection has false negatives (custom secret formats) and false positives (long random strings). The entropy-based fallback mitigates unknowns; the hash mode (`[REDACTED:sha256:xxxxxxxx]`) allows tracing without exposure.

### 5.6 Hierarchical Config with Deep Merge

**Decision**: Configuration is loaded in priority order: workspace `.harness_config.json` → `~/.harness/config.json` → shipped `cli.json` fallback. Nested dicts are deep-merged rather than replaced.

**Rationale**: Different projects need different models, budgets, and sandbox configs. Deep merge allows overriding a single nested key (e.g., `token_budget.hard_cap_usd`) without re-declaring the entire section.

### 5.7 GitGuardian: Isolated Patch Branches

**Decision**: Every harness session creates an `agent/patch-{session_id[:8]}` branch off the current HEAD. On success, changes are committed and the original branch restored. On failure, the patch branch is deleted with checkout rollback.

**Rationale**: The harness must never corrupt the developer's working state. Stashing pre-existing changes + isolated branches + automatic rollback provides defense-in-depth against accidental destruction.

### 5.8 TypedDict-Only State Schema

**Decision**: `AgentState` is a `TypedDict` (for LangGraph compatibility). An earlier version of this codebase also defined a parallel `AgentStatePydantic(BaseModel)` and companion `TokenTrackerPydantic` / `DiagnosticObjectPydantic` / `MessagePydantic` classes. These were removed because they were never imported anywhere outside their own definition block — they added no runtime validation, imposed an optional `pydantic` dependency, and their claimed "dual schema" was fictional.

**Rationale**: LangGraph's `StateGraph` requires a TypedDict schema. State factories (`create_initial_state`) already provide safe defaults; Pydantic's per-field validation would add per-call overhead without catching bugs that TypedDict's structural contract (plus existing regression tests) doesn't already catch. The Pydantic option remains available as a future addition if a clear use case emerges.

### 5.9 Docker-First Sandbox Selection

**Decision**: Auto-detection now prioritizes Docker over unshare: `docker → unshare → bare`. Previously it was `unshare → docker → bare`.

**Rationale**: Docker provides stronger isolation boundaries (containers vs namespaces) with built-in resource limits (memory, CPU, PID caps). The `unshare` backend is still available as a faster fallback when Docker is unreachable. User-specified backends (`"unshare"`, `"docker"`, `"bare"`) bypass auto-detection entirely.

**Trade-off**: Slightly higher startup latency on container cold-start (~1-2s). The unshare backend remains the faster option on systems where Docker is unavailable or unnecessary.

### 5.10 Gateway Typo Resilience

**Decision**: When `_validate_routing_keys()` detects unregistered model names in config (likely typos), it no longer raises a blocking `ValueError`. Instead, it logs the error and auto-falls back to the configured `ollama_local_model` if available.

**Rationale**: Stopping execution for a typo in `.harness_config.json` is unnecessarily disruptive. Ollama is configured as a zero-cost fallback — using it keeps the graph alive while alerting the developer to fix their config.

### 5.11 Repair Prompt Fallback Triad

**Decision**: The repair node now composes its prompt from three sources in priority order: (1) structured compiler diagnostics, (2) lintgate errors, (3) raw build output tail (last 2000 chars). If no structured diagnostics exist, the raw build output is appended.

**Rationale**: Many build tools produce output that doesn't match any structured parser (Makefiles, shell scripts, custom build systems). The raw output fallback ensures the LLM always has context to generate a fix, even when diagnostic parsing produces zero results.

### 5.12 Strict Format Reminders for Code Generation

**Decision**: Both `patching_node` and `repair_node` inject a `[CRITICAL FORMAT INSTRUCTION]` message immediately before the LLM dispatch call. This message shows exact patch block templates and forbids markdown, explanations, or text outside blocks.

**Rationale**: Smaller/faster models (used as `patching_primary`) often ignore the system prompt's format instructions when they're buried in a long initial prompt. A short, forceful reminder immediately before the call dramatically increases patch block compliance.

### 5.13 Code Quality Standards in Prompts

**Decision**: The system prompt (`messages[0]`) now includes a **Code Quality Standards** section (modularity, error handling, type hints, edge cases, production-ready code). Both format reminders include a one-line quality directive.

**Rationale**: Autonomous code generation without quality guardrails produces fragile, throwaway code. Embedding quality expectations in every LLM call ensures generated code is modular, well-typed, and production-ready.

### 5.14 HITL Raw Build Output Display

**Decision**: When the HITL menu shows "No compiler errors captured" but `node_state.last_build_output` exists, it now displays the raw build output (last 2000 chars) instead of leaving the developer blind.

**Rationale**: The developer needs to see what actually failed before choosing an action ([e] hint, [m] manual fix, [r] retry). Previously, with zero structured diagnostics, the HITL screen gave no actionable information.

### 5.15 First-Run Healthcheck (`harness doctor`)

**Decision**: A dedicated `harness doctor` subcommand surfaces the five environment preconditions that previously turned into silent first-run failures: git repo presence, API keys per routed provider, sandbox backend reachability, checkpoint DB writability, and config parse cleanliness. Each check returns one of PASS / WARN / FAIL with a colored marker; the command exits non-zero on any FAIL.

**Rationale**: Before doctor, users debugging a broken install had to read error messages buried in `harness run` logs. Surfacing the preconditions explicitly turns "why didn't anything happen?" into "your `OPENAI_API_KEY` is missing." Each check is also a smoke test for the underlying config path, so doctor doubles as a sanity check after editing `.harness_config.json`.

**Trade-off**: Adds five subprocess + filesystem probes (~50ms each, parallel where possible) per invocation. Acceptable for a deliberate operator command.

### 5.16 Pluggable HITL Transport

**Decision**: The HITL menu is rendered through an `HitlChannel` interface with three built-in implementations: `StdinChannel` (default — interactive terminal), `FileChannel` (read prompts/answers from JSONL files; useful for replay and tests), `HttpChannel` (POST prompt → receive JSON answer; useful for remote operators / web dashboards).

**Rationale**: The original implementation hard-coded `input()` calls inside the gatekeeper nodes, which made every HITL site uniquely difficult to test. Routing through an ABC let us write deterministic tests against `FileChannel` and unblocked the still-deferred web dashboard (T4.1) without committing to it.

**Trade-off**: One extra indirection per prompt. The channel is a process-wide singleton, so non-CI tests reset it via `reset_channel()`.

### 5.17 Multi-Stack Coverage via `tree-sitter-language-pack`

**Decision**: Replace six individual `tree-sitter-*` grammar packages with the single `tree-sitter-language-pack` wheel, which bundles 165+ grammars including Python, Java, JS/TS/TSX, Dart, Rust, Go, and Swift. Patcher, impact analyzer, and the new `JavaParser` / `TypeScriptParser` / `DartParser` all read from the same registry.

**Rationale**: Six grammar packages meant six upgrade cadences, six release-note streams, and one of them (Dart) had no standalone PyPI distribution at all. Consolidating to one wheel buys us Dart coverage and amortizes the grammar churn into a single dependency line. Adding a new language is now "register a parser" instead of "add a new dependency."

**Trade-off**: Slightly larger install footprint (~15 MB of bundled grammars). The footprint is paid once at install time, not per-run.

### 5.18 Stack-Aware Skill Filtering

**Decision**: Skill files in `harness/skills/` may declare an `applies_to: [tag1, tag2]` YAML frontmatter (parsed by `graph.py:_parse_skill_frontmatter`). At graph assembly, the workspace is fingerprinted to a tag set (`python`, `flutter`, `spring`, `react`, …); skill files whose `applies_to` doesn't intersect the workspace tags are excluded from the LLM prompt. Files without frontmatter always load (universal skills).

**Rationale**: A user working on a Flutter app should not see a 4000-character Django Channels skill in their prompt. Filtering at the frontmatter level keeps the prompt budget small without forcing the harness to "guess" relevance from filename pattern matching.

**Trade-off**: Skill authors have to remember to add the frontmatter — but the failure mode is permissive (no frontmatter → always load), so the worst case is a too-large prompt, not a missing skill.

### 5.19 Flutter / Mobile Routing Short-Circuit

**Decision**: On a clean security scan, if the workspace looks like a Flutter project (`pubspec.yaml` with `flutter:` SDK dep, detected by `impact._is_flutter_project`), the graph routes directly to END instead of through the docker compose deploy pipeline.

**Rationale**: Flutter's artifact is a mobile binary (APK / AAB / IPA / web bundle), not a docker compose service stack. Running the deploy pipeline on a Flutter project would produce an unrunnable Dockerfile and waste budget on a synthesize-architecture LLM call. Short-circuiting matches the user's mental model — "build and stop."

**Trade-off**: Flutter projects don't get the deploy-blueprint HITL gate. That's correct for v1.x; if users ask for cloud-build wiring we can add a `flutter:` deploy backend.

### 5.20 Structured Failure-Event Catalogue

**Decision**: Failure sites emit structured events via `harness.observability.log_failure(name, **fields)` — an ERROR-level mirror of the existing `emit_event` helper. Each event carries a snake_case `event` field, so failures are grep-able from the per-session JSONL log by event name instead of by string fragment. The catalogue: `sandbox_start_failed`, `token_budget_exhausted`, `hitl_gate_blocked`, `llm_empty_response`, `llm_circuit_open`.

**Rationale**: Logging was already comprehensive but inconsistent — every module invented its own `logger.error("...")` format, so an operator scanning a failure across modules had to grep multiple substrings. A named event catalogue makes the failure modes a first-class queryable shape: `jq 'select(.event == "token_budget_exhausted")'`. The catalogue also feeds `harness metrics`, which counts each event per session.

**Trade-off**: New failure sites need a name. The `log_failure` docstring lists naming conventions (`_failed`, `_exhausted`, `_blocked`) and the canonical catalogue, so authors can extend it without inventing new patterns.

### 5.21 Checkpoint Message Redaction (P0.1)

**Decision**: `HarnessAsyncSqliteSaver.aput` / `aput_writes` scrub the `messages` channel through `harness.redactor.redact_messages` before letting LangGraph's serializer touch it. Opt-out via `persistence.redact_messages: false`. Redactor crashes fail open (log a WARNING, persist the original) so a redactor bug can never block the checkpoint write itself.

**Rationale**: Pre-transmission redaction (5.5) already kept secrets out of outbound API calls, but the *checkpoint* was unguarded. A pasted API key landed at rest in `~/.harness/checkpoints.db` (msgpack blob) — a privacy/exposure hole if backups left the host. Scrubbing on the write path closes the hole without touching the in-memory state the running graph holds.

**Trade-off**: Operators who want verbatim transcripts (audit, replay) need to flip the opt-out explicitly. That's the safer default — silent loss-of-fidelity is preferred to silent loss-of-secrecy.

### 5.22 Patcher Symlink Guard + Conservative Allowlist (P1.1, P1.2)

**Decision**: The async writer in `harness/patcher.py` refuses to write through any path where `os.path.islink(target)` is true and uses `O_NOFOLLOW` on Linux/macOS to catch races. When the source-root heuristic can't decide on a project layout, `_build_patcher_allowlist` returns a conservative set (`src/`, `lib/`, `app/`, `pkg/`, `cmd/`, `tests/`, `test/`, `__tests__/`, plus `_ROOT_ALLOWLIST_FILES` and any `requirements*.txt`) and logs a WARNING so the operator can fix detection.

**Rationale**: A malicious LLM output (or a confused one) shouldn't be able to walk an attacker-controlled symlink out of the workspace, and a flat / unfamiliar workspace shouldn't open the harness up to whole-tree rewrites. Both are defence-in-depth — neither is the primary boundary, but together they collapse the blast radius of an LLM hallucination from "anywhere on disk" to "the part of the workspace that looks like source."

**Trade-off**: Windows native has no portable `O_NOFOLLOW`; the `islink` check is the only guard there (TOCTOU window, but small). The conservative allowlist may refuse a legitimate write in an exotic layout — that surfaces as a clear WARNING the operator can act on.

### 5.23 Network Auto-Enable Opt-In (P1.3)

**Decision**: `_apply_toolchain_adaptation` no longer auto-flips `allow_network=True` on detected pip/npm install commands unless `sandbox.auto_enable_network_for_install: true` is set. When the heuristic fires with the opt-in off, the function logs a WARNING pointing the operator at the config key.

**Rationale**: Auto-enabling network on heuristic match is a least-surprise *footgun* — it silently relaxes the sandbox boundary because the build command looks plausible-installish. The opt-in keeps the heuristic available for operators who know they want it while making the default match the principle of least privilege.

**Trade-off**: Operators who *do* rely on the heuristic must remember one config flag. The WARNING in the log makes the missing flag discoverable on the first session that hits it.

### 5.24 Pre-Flight Budget Refusal + Empty-LLM Handling + Circuit Breaker (P1.4 / P1.5 / P1.9)

**Decision**: Three reliability primitives on the gateway path, working together:
- **Pre-flight budget** (`BudgetTooLowError`): estimate `(input_chars/4) × input_rate + 4000 × output_rate`; refuse before any HTTP call if the estimate exceeds `budget_remaining_usd`. WARNING when a call lands within 20% of the cap.
- **Empty-response retry** (`EmptyLLMResponseError`): retry up to two extra times on empty content; on exhaustion, `repair_node` sets `llm_silent=True` and `route_after_compiler` short-circuits to HITL instead of waiting for the 3-cycle repair cap.
- **Rate-limit circuit breaker**: 3 HTTP 429/5xx failures in a 5-minute sliding window opens the breaker; the next `dispatch` diverts to `force_local=True` (Ollama).

**Rationale**: Each closes a specific failure mode the pilot can hit in the first hour — runaway prompt overflowing the cap, a stuck provider returning empty bodies, persistent throttling. Doing all three on the same surface (the gateway) keeps the policy in one place and the routing simple.

**Trade-off**: The budget pre-flight is conservative (assumes worst-case 4000-token output); a few legitimate calls will be refused that *would* have fit. Operators can raise the cap. Empty-response detection treats `""` as the signal — a real LLM that legitimately returns an empty string (rare) gets retried twice unnecessarily.

### 5.25 Strict Checkpoint Deserialize + Schema Versioning (P1.6 / P2.4)

**Decision**: `_deserialize_checkpoint_blob(strict=True)` raises `CheckpointCorruptedError` instead of returning `{}`. `cmd_resume` pre-flights the most recent blob with `strict=True` and surfaces a clear operator message on corruption. `doctor` scans the 5 most recent rows. Every checkpoint metadata blob now carries `_harness_schema_version` (current `CHECKPOINT_SCHEMA_VERSION = 1`); `validate_checkpoint_schema` refuses futures (`CheckpointSchemaMismatchError`) and warns-then-allows legacy unstamped blobs.

**Rationale**: The previous "silently restore empty state" path was the worst possible failure shape — the graph appeared to resume but actually restarted from scratch, often clobbering the workspace with a fresh first patch. Strict mode + a clear operator message ("fresh start / restore backup / purge session") makes the recovery path explicit. Schema versioning future-proofs the on-disk format against the first MAJOR bump that adds a required AgentState field.

**Trade-off**: Operators with corrupted checkpoints now see a hard refusal instead of an opaque fresh-start. That's the desired outcome — see `docs/RUNBOOK.md` § 1 for the recovery recipe.

### 5.26 Single-Writer Workspace Lock (P1.7)

**Decision**: `cmd_run` acquires an `fcntl.flock(LOCK_EX | LOCK_NB)` on `<workspace>/.harness_session.lock` at startup. The handle is pinned in a module-level slot so the OS holds it for the process lifetime. `--force-lock` releases a stale lock and acquires fresh, logging a WARNING. Platforms without `fcntl` (Windows native) skip locking with a DEBUG log.

**Rationale**: Two `harness run` sessions on the same workspace were a real footgun — race on patch branches, fight over the build command, write to the same files. An advisory file lock is the cheap correct fix, and `--force-lock` gives the operator an escape hatch for the crashed-prior-session case (see `docs/RUNBOOK.md` § 4).

**Trade-off**: Windows native gets no enforcement — single-writer is the operator's responsibility there. We accept this rather than pull in `msvcrt.locking` (different semantics, more surprises).

### 5.27 Discovery JSON Trust Guards (P2.2)

**Decision**: `trust.validate_discovery_json` rejects payloads larger than 1 MB (UTF-8 byte length) before invoking `json.loads`, and rejects parsed trees deeper than 10 levels via a cycle-safe `_json_depth` walker. The existing per-question (10,000 chars) and module-count (50) caps remain.

**Rationale**: A malicious or runaway LLM could synthesize a 50 MB JSON or a billion-laughs-style nested object that hits Python's default 1000-frame recursion limit and crashes the process. The byte cap kills the obvious pathological input before parsing; the depth walk catches the rest.

**Trade-off**: A pathological-but-legitimate response above either cap is rejected. A normal discovery is depth ~4 and a few KB, so the margin is two orders of magnitude.

### 5.28 Log Rotation (P2.3)

**Decision**: `configure_logging` installs a `RotatingFileHandler` for the per-session JSONL by default (`maxBytes=10_000_000`, `backupCount=5`). Configurable via `logging.max_bytes` and `logging.backup_count`. Setting `max_bytes=0` falls back to plain `FileHandler` for operators pinning a single non-rotating file.

**Rationale**: The plain `FileHandler` had no upper bound — a long pilot session could silently fill the customer's disk over weeks. 10 MB × 5 backups gives ~50 MB per session for post-mortem coverage without unbounded growth.

**Trade-off**: A rotation in the middle of a session means the most-recent file is the *live* one, with older content in `.1`, `.2`, …; tools that read the JSONL (notably `harness metrics`) sort the rotated suffixes chronologically before iterating.

### 5.29 Process-Wide CommandValidator (P0.2)

**Decision**: `cmd_run` and `cmd_resume` build a `CommandValidator` via `create_command_validator_from_config(config)` and register it process-wide with `set_command_validator()`. `SandboxExecutor.__init__` falls back to the global default when no explicit validator is passed. Mirrors the redactor's global-scanner pattern.

**Rationale**: The validator existed but every `SandboxExecutor(...)` call site defaulted `command_validator=None`, and the inner guard short-circuited validation entirely on None. The runtime quietly ran without the defence layer that the codebase already shipped. A process-wide registration ensures every executor — including ones added in the future — picks it up without modification.

**Trade-off**: Tests that previously instantiated `SandboxExecutor` directly with `command_validator=None` and expected no validation now inherit the global if one is set; tests explicitly assert behaviour with and without the global. Explicit constructor argument still wins, so test isolation is unaffected.

### 5.30 Cost-Metrics Aggregation (P2.7)

**Decision**: `harness metrics` (CLI surface) plus `harness/metrics.py` (pure aggregation) reconstruct per-session cost, token, and error metrics by reading `<id>.jsonl` + `<id>.jsonl.*` rotated backups. Outputs: human (stdout), `--json`, `--prometheus`. Machine-readable outputs default to `~/.harness/metrics/` (configurable via `metrics.metrics_dir`) and are written atomically (`<dest>.tmp` → `os.replace`). Burn rate is computed over a trailing window (default 10 minutes); exhaustion is projected against `token_budget.hard_cap_usd`.

**Rationale**: Operators running multiple sessions had no way to see aggregate cost or "when will I hit the cap at this rate" without a hand-rolled jq pipeline. Doing it as a read-only CLI (instead of an HTTP daemon) matches the single-tenant pilot scope — a cron job emitting `--prometheus` is enough for node_exporter textfile-collector scrape, with no auth/network surface to harden.

**Trade-off**: Metrics live entirely on disk (the JSONL logs are the source of truth). Purging logs deletes the metrics record too — by design, since `harness purge --session-id` is the GDPR-deletion path. For longer-term retention, operators redirect `logging.log_dir` and `metrics.metrics_dir` to a managed location.

---

## 6. Data Model Overview

### 6.1 AgentState (Primary State Object)

```
AgentState
├── workspace_path: str              # Absolute path to target repo
├── messages: list[MessageDict]      # Conversation history
│   ├── role: "system"|"user"|"assistant"|"tool"
│   ├── content: str
│   ├── name: Optional[str]
│   ├── tool_calls: Optional[list]
│   └── tool_call_id: Optional[str]
├── modified_files: list[str]        # Paths edited this session
├── compiler_errors: list[DiagnosticObjectDict]
│   ├── file: str
│   ├── line: int
│   ├── column: int
│   ├── severity: "error"|"warning"
│   ├── error_code: str
│   ├── message: str
│   └── semantic_context: str
├── token_tracker: TokenTrackerDict
│   ├── total_input_tokens: int
│   ├── total_output_tokens: int
│   ├── total_cached_tokens: int
│   ├── total_cost_usd: float
│   └── per_model: dict[str, dict]   # Per-model breakdown
├── loop_counter: dict[str, int]     # {patching, repair, compiler, total_repairs, security, deployment}
├── allow_network: bool
├── build_command: str               # e.g., "make build"
├── budget_remaining_usd: float
├── session_id: str                  # UUIDv4 or user-provided
├── exit_code: int                   # Last compiler exit code
├── node_state: dict[str, Any]       # Node-specific metadata
├── current_gate: str                # "REQUIREMENTS"|"ARCHITECTURE"|"DEPLOYMENT"|""
├── spec_requirements_path: str
├── spec_architecture_path: str
└── deployment_blueprint_path: str
```

### 6.2 Checkpoint Schema (SQLite)

```
Table: checkpoints
├── thread_id: TEXT (PK composite)
├── checkpoint_ns: TEXT (PK composite)
├── checkpoint_id: TEXT (PK composite)
├── parent_checkpoint_id: TEXT
├── type: TEXT
├── checkpoint: BLOB (msgpack — channel_values with `messages` redacted via harness.redactor)
├── metadata: BLOB (msgpack — includes `_harness_schema_version` stamp, current = 1)
├── created_at: TEXT
└── updated_at: TEXT

Table: writes
├── thread_id, checkpoint_ns, checkpoint_id,
│   task_id, idx (PK composite)
├── channel: TEXT
├── type: TEXT
├── value: BLOB
└── created_at: TEXT

Table: blobs
├── thread_id, checkpoint_ns, channel,
│   version (PK composite)
├── type: TEXT
├── blob: BLOB
└── created_at: TEXT
```

### 6.3 Model Registry

```
_MODEL_REGISTRY: dict[str, ModelSpec]
├── key: "provider:model_id" (e.g., "openai:gpt-4o")
└── ModelSpec
    ├── provider: "deepseek"|"anthropic"|"openai"|"ollama"
    ├── model_id: str
    ├── context_window: int
    ├── input_cost_per_1m: float
    ├── output_cost_per_1m: float
    ├── cached_input_cost_per_1m: float
    ├── api_base_url: str
    ├── supports_thinking: bool
    └── supports_cache: bool
```

---

## 7. Integration Points

### 7.1 Gateway ↔ Providers
- **Protocol**: HTTPS REST (httpx AsyncClient)
- **DeepSeek**: POST `{base_url}/chat/completions` (OpenAI-compatible JSON)
- **Anthropic**: POST `{base_url}/messages` with `x-api-key` header, system prompt extracted to top-level field
- **OpenAI**: POST `{base_url}/chat/completions`
- **Ollama**: POST `{base_url}/chat/completions` (no auth, localhost)

### 7.2 Sandbox ↔ Build Tools
- **Protocol**: asyncio subprocess with PGID management
- **Unshare**: `unshare --mount --pid --fork --mount-proc [--net] -- sh -c "<command>"`
- **Docker**: `docker run --rm --read-only --tmpfs /tmp:exec --memory=... --network=none|bridge -v ...`
- **Bare**: `sh -c "cd <workspace> && <command>"`

### 7.3 Persistence ↔ LangGraph
- **Protocol**: `AsyncSqliteSaver` implementing LangGraph's `BaseCheckpointSaver` interface
- **Methods**: `put(config, checkpoint, metadata, new_versions)` → `get(config)` → `list(config, limit, before)`
- **Journal**: WAL mode for concurrent read/write safety

### 7.4 File I/O
- **Primary**: aiofiles (async) for all patcher operations
- **Fallback**: sync `open()` when aiofiles is not installed
- **Temp Files**: `tempfile.NamedTemporaryFile` for sandbox log buffering

### 7.5 External Tools (Optional, Runtime-Detected)
- **gitleaks**: Secret scanning (`detect --no-git --report-format json`)
- **bandit**: Python SAST (`-r -f json -ll -q`)
- **semgrep**: Universal SAST (`scan --config=auto --json --quiet`)
- **ruff**: Python formatting (`format --quiet`) and linting (`check --fix --quiet`)
- **gofmt**: Go formatting (`-w`)
- **prettier**: JS / TS / TSX / JSX / CSS / HTML / JSON / YAML / Markdown formatting (`--write`)
- **rustfmt** + **clippy**: Rust formatting + lint
- **clang-format**: C / C++ formatting (`-i`)
- **google-java-format**: Java formatting
- **dart format**: Dart formatting (Flutter / Dart projects)
- **shfmt**: shell-script formatting
- **sqlfluff**: SQL linting + formatting
- **docker compose** (V2 — no hyphen): Container orchestration (`up --build -d`, `down`). The legacy `docker-compose` V1 binary is no longer probed.

---

## 8. Deployment & Environment

### 8.1 Runtime Requirements
- Python 3.11+ (3.11 / 3.12 / 3.13 covered by CI; macOS + Windows on 3.12 as advisory `continue-on-error` matrix entries)
- Linux is the blocking CI target. macOS and Windows + WSL2 are best-effort via the Docker backend; the `unshare` backend and fcntl workspace lock are Linux-only.
- Git 2.x+ (for branch lifecycle management)
- Sandbox: Docker daemon (preferred), OR Linux user-namespace support
  (`unshare --user`), OR opt-in bare via `HARNESS_ALLOW_UNSAFE_SANDBOX=true`.
- tree-sitter grammars ship in-tree via `tree-sitter-language-pack`; no
  per-language install needed.

### 8.2 Configuration Files
| File | Location | Purpose |
|------|----------|---------|
| `cli.json` | Shipped with package | Absolute fallback defaults |
| `~/.harness/config.json` | User home | Global default models and settings |
| `.harness_config.json` | Workspace root | Per-project override (highest priority) |
| `requirements-prod.txt` | Repo root | Exact transitive pins for reproducible pilot installs (`pip install -e . --constraint requirements-prod.txt`) |
| `LICENSE` | Repo root | MIT license; referenced from `pyproject.toml` so wheels ship it |

**Top-level config sections**: `build_command`, `allow_network`, `sandbox`, `token_budget`, `node_throttle`, `models`, `model_routing`, `persistence`, `logging`, `lintgate`, `deployment`, `test_generation`, `metrics`.

**Key recent additions**:
- `persistence.redact_messages` (default `true`) — opt out of checkpoint message redaction.
- `sandbox.auto_enable_network_for_install` (default `false`) — opt in to auto-enabling network on detected pip/npm install.
- `node_throttle.max_discovery_iterations` (default `10`, clamped `[1, 30]`) — hard cap on discovery loop.
- `logging.max_bytes`, `logging.backup_count` — rotation knobs for the per-session JSONL file (default 10 MB × 5).
- `metrics.metrics_dir` (default `~/.harness/metrics`) and `metrics.burn_rate_window_minutes` (default `10`) — `harness metrics` output and window.

### 8.3 Environment Variables
| Variable | Purpose |
|----------|---------|
| `DEEPSEEK_API_KEY` | DeepSeek API authentication (env wins precedence over `models["<key>"].api_key` config field) |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) API authentication |
| `OPENAI_API_KEY` | OpenAI API authentication |
| `CI` | Detect CI environment (auto-approve HITL gate behavior) |
| `HARNESS_AUTO_APPROVE` | Force auto-approve for non-interactive runs |
| `HARNESS_ALLOW_UNSAFE_SANDBOX` | Opt in to the `bare` (zero-isolation) sandbox backend when neither Docker nor `unshare` is available. Never set this outside a disposable VM. |
| `NO_COLOR` | Suppress ANSI colour markers in `harness doctor` output. |
| `HARNESS_DOCTOR_SKIP_LIVE` | Truthy value (`1` / `true` / `yes`) skips the live 1-token chat ping the `api keys` check makes against each configured provider. Falls back to key-presence-only validation. Useful for CI runs where outbound HTTPS is blocked. |
| `LANGCHAIN_API_KEY` | Required when `logging.langsmith=true` to forward traces to LangSmith. |
| `LANGCHAIN_TRACING_V2`, `LANGSMITH_PROJECT` | Additional LangSmith trace routing knobs honoured by `configure_logging`. |

API keys can also live in `models["<provider>:<model>"].api_key` inside any config layer — the gateway's resolution order is explicit arg → env var → config field, and `harness doctor` reflects the same policy.

### 8.4 Generated Files (during execution)
- `docs/SPEC_REQUIREMENTS.md` — Requirements specification
- `docs/SPEC_ARCHITECTURE.md` — Architecture specification
- `docs/DEPLOYMENT_BLUEPRINT.md` — Container deployment blueprint
- `Dockerfile` / `Dockerfile.<service>` — Per-service container images
- `docker-compose.yml` — Multi-service orchestration
- `Caddyfile` — Reverse proxy routing rules
- `~/.harness/checkpoints.db` — Session checkpoint database (WAL mode; metadata includes `_harness_schema_version`)
- `~/.harness/logs/<session-id>.jsonl[.N]` — Per-session structured JSONL log (rotated)
- `~/.harness/metrics/<session-id>.{json,prom}` — `harness metrics` outputs (configurable via `metrics.metrics_dir`)
- `<workspace>/.harness_session.lock` — fcntl single-writer lock; auto-released when the process exits
- `/tmp/.harness/` — Temporary sandbox build logs (auto-cleaned)