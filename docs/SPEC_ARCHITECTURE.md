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
│   ├── load_deployment_defaults()  # Optional org-wide deployment.json policy (FR-048)
│   ├── cmd_run / cmd_resume / cmd_status / cmd_purge / cmd_metrics
│   ├── cmd_doctor()             # 6-check healthcheck (FR-025)
│   ├── _doctor_check_git / global_config / api_keys / sandbox / checkpoint_db / config
│   ├── _ping_provider_live()    # 1-token chat call per provider (PASS only after auth confirmed)
│   ├── human_gatekeeper_node()  # Three-phase HITL gatekeeper
│   ├── hitl_menu_loop()         # 7-action HITL menu: [v/r/e/m/b/s/q]
│   ├── _emit_output()           # Routes machine-readable metrics to file / stdout
│   ├── _archive_consumed_change_requests()  # Move consumed CR-*.txt into applied/<sid>/ + manifest.json (FR-045)
│   ├── _make_git_guardian()     # Returns no-op stub when --git=disable (FR-049)
│   └── interactive_review_loop()# Pre-flight manifest review
├── wizard.py             # Interactive setup wizard for bare `harness run` (FR-047)
│   ├── run_setup_wizard()       # Top-level: new vs resume → workspace → prompt-source → --new_build → --git
│   ├── _prompt_new_or_resume()  # First fork: greenfield/brownfield or `harness resume <id>`
│   ├── _choose_session()        # Lists checkpointed sessions newest-first for resume
│   └── _confirm_change_requests_folder()  # Detects change_requests/ and offers brownfield mode
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
│   ├── patching_node()   # LLM: generate SEARCH/REPLACE patches (CR-N markers in change-request mode)
│   ├── compiler_node()   # Deterministic: run build in sandbox
│   ├── repair_node()     # LLM: analyze errors, fix, escalate to fallback model
│   ├── human_intervention_node() # Set HITL flags
│   ├── requirements_discovery_node() # LLM: 8-sector requirements discovery (delta-mode in CR sessions)
│   ├── architecture_discovery_node() # LLM: 8-sector architecture discovery (delta-mode in CR sessions)
│   ├── deployment_discovery_node()   # LLM: 4-sector deployment discovery (deployment.json defaults injected)
│   ├── ingest_change_requests_node() # Parse change_requests/*.txt, assign CR-N IDs (FR-045)
│   ├── reverse_engineer_architecture_node() # One-shot SPEC_ARCHITECTURE.md synthesis on first contact (FR-046)
│   ├── write_spec_node() # Serialize discovery to .md files (delta-merge in CR sessions)
│   ├── generate_deployment_spec_node() # Produce DEPLOYMENT_BLUEPRINT.md + cr_attribution
│   ├── route_after_start()       # Conditional: ingest_change_requests / patching / requirements_discovery
│   ├── route_after_compiler()    # Conditional: repair / HITL (short-circuits on llm_silent) / security_scan
│   ├── route_after_discovery()   # Conditional: write_spec / discovery loop (capped at max_discovery_iterations, FR-043)
│   ├── route_after_gatekeeper()  # Conditional: next phase / refinement loop
│   ├── route_after_security_scan() # Conditional: repair / HITL / END (Flutter, FR-028) / END (no --dev-deployment, FR-044) / deployment_discovery
│   ├── route_after_hitl()        # Conditional: compiler / END
│   ├── _build_patcher_allowlist()# Conservative fallback when source root unclear (FR-041); includes Node-JS allowlist
│   ├── _apply_toolchain_adaptation() # pip/npm network auto-enable gated by config opt-in (FR-042)
│   ├── apply_memory_cleanse()    # Compress verbose repair messages
│   ├── build_graph()             # Assemble full StateGraph
│   └── run_graph()               # Async entry point (now threads dev_deployment, change_request_mode, …)
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
├── parser_registry.py    # Diagnostic parser plugins (FR-026)
│   ├── RustParser        # --error-format=json
│   ├── GccClangParser    # -fdiagnostics-format=json
│   ├── GoParser          # file:line:col: message
│   ├── PythonParser      # Traceback extraction
│   ├── JavaParser        # javac / maven / gradle diagnostic shapes
│   ├── TypeScriptParser  # tsc / eslint
│   ├── DartParser        # dart analyze / flutter build
│   ├── GenericParser     # file:line:col: severity: message
│   ├── register_parser / register_extension_parser
│   ├── get_parser / get_parser_for_extension / list_registered_parsers
│   └── detect_and_parse() # Auto-detect + parse
├── mcp_client.py         # Model Context Protocol client (FR-051)
│   ├── McpServerConfig / McpPoolConfig
│   ├── StdioMcpClient    # JSON-RPC 2.0 over newline-delimited stdio
│   ├── McpClientPool     # Concurrent server startup; per-server lifecycle
│   ├── McpToolSkill      # ToolSkill subclass; mcp__<server>__<tool> names
│   ├── parse_mcp_blocks / strip_mcp_blocks  # <<<MCP_CALL>>> text-DSL
│   └── register_mcp_skills(pool)
├── web_tools.py          # WebFetchSkill / WebSearchSkill (FR-053)
│   ├── WebToolsConfig    # max_bytes / max_results / allow_private_ips / backends
│   ├── html_to_text      # Tag stripper + entity decode
│   ├── DuckDuckGoLiteBackend  # No-key search backend
│   ├── WebFetchSkill / WebSearchSkill   # ToolSkill subclasses
│   ├── parse_tool_blocks / strip_tool_blocks  # <<<WEB_FETCH>>>/<<<WEB_SEARCH>>>
│   └── register_web_tool_skills(cfg)
├── github_integration.py # `harness gh` family (FR-054)
│   ├── gh_path / gh_available / gh_auth_status
│   ├── fetch_issue       # gh issue view --json
│   ├── ingest_issue_to_change_request  # Bridge → change_requests/CR-N-<slug>.txt
│   ├── create_pr         # gh pr create
│   └── post_pr_comment   # gh pr comment
├── repo_index.py         # Semantic retrieval index (FR-056)
│   ├── RepoIndexConfig   # backend / chunk_lines / top_k / inject_max_bytes
│   ├── Chunker           # File walker + line windowing
│   ├── _tokenize         # Identifier-aware (CamelCase + snake_case split)
│   ├── TfidfBackend      # Pure-Python TF-IDF (default)
│   ├── OpenAIEmbeddingsBackend  # Opt-in dense backend
│   ├── build_index / get_stats / clear_index / query_top_chunks
│   └── async_query_top_chunks  # asyncio.to_thread wrapper for planner
├── repo_memory.py        # Per-repo session log (FR-057)
│   ├── RepoMemoryConfig  # enabled / dir / max_bytes / inject_max_bytes
│   ├── repo_identity     # SHA-256 of git origin URL or workspace path
│   ├── read_repo_memory  # Returns trimmed file content for planner
│   └── append_session_note  # FIFO-trimmed atomic append
├── chat.py               # harness chat REPL (FR-058)
│   ├── ChatSession       # Dataclass with reader/writer injection points
│   ├── run_chat          # Top-level loop; handles slash commands
│   ├── _handle_user_turn # Per-turn LLM dispatch + tool loop
│   ├── _run_build        # /build executes configured build in sandbox
│   ├── _apply_patches_from_last  # /apply with HITL confirmation
│   └── _inject_repo_memory / _maybe_inject_repo_index
├── fanout.py             # Multi-agent fan-out (FR-060)
│   ├── AgentSpec / AgentResult / Verdict
│   ├── run_parallel_agents  # Bounded semaphore + budget reservation/refund
│   ├── run_with_verification  # Adversarial skeptic pattern
│   ├── _parse_first_json     # JSON-from-noise extractor for judge verdicts
│   ├── make_fanout_skill     # SubAgentFanoutSkill — exposes fan-out to planner
│   └── register_fanout_skill # Registered in SkillRegistry at startup
├── schedule.py           # Cron-driven daemon (FR-062)
│   ├── Schedule / Job / ScheduleConfig
│   ├── parse_schedule    # Hand-rolled cron subset (every Nm/h/d / daily / weekly)
│   ├── next_run          # Compute next firing time (UTC; tz-aware required)
│   ├── execute_job_once  # Subprocess + history + hooks
│   ├── ScheduleDaemon    # Main loop with in-flight tracking
│   │   ├── tick_once     # Fires due config jobs + due web one-shots
│   │   ├── _due_oneshots / _fire_oneshot  # Web one-shot integration
│   │   └── run_forever
│   ├── record_run_started / record_run_finished / history_for_job
│   └── _run_hook         # /bin/sh -c with HARNESS_JOB_* env vars
├── dashboard.py          # Read-only viewer + interactive web app (FR-063 / FR-064)
│   ├── DashboardConfig   # Host/port/token/csrf/writes_enabled/web_db_path
│   ├── list_sessions / cost_burn_series / list_memory_files
│   ├── repo_index_status / list_schedule_runs / read_memory_file
│   ├── resolve_expected_token / resolve_csrf_token / check_auth / check_csrf
│   ├── read_config_file / write_config_section_atomic   # Tier B
│   ├── write_memory_file / add_schedule_job_to_config
│   ├── spawn_harness_run / cancel_session               # Tier C
│   ├── tail_session_events  # Generator backing SSE stream
│   ├── dispatch(cfg, path)  # Pure routing function — tested without server
│   ├── make_request_handler # Factory closes over cfg + tokens
│   │   ├── do_GET / do_HEAD / _stream_sse
│   │   ├── do_POST          # Routes /config/* /memory/* /run/* /sessions/* /hitl/*
│   │   ├── _handle_config_save / _handle_memory_save / _handle_run_now
│   │   ├── _handle_run_schedule / _handle_cancel / _handle_note
│   │   ├── _handle_hitl_answer / _handle_hitl_webhook  # Blocking webhook
│   │   └── _handle_schedule_add
│   └── start_server      # Returns ServerHandle (carries csrf_token for tests)
├── web_state.py          # Web app runtime state (FR-064)
│   ├── WebProcess / ProcessRegistry    # Spawned-subprocess tracking
│   ├── HitlQueue / PendingHitl         # Webhook ↔ UI bridge
│   ├── open_web_db / web.db schema     # audit_log / run_presets / web_oneshot_jobs / chat_notes
│   ├── queue_chat_note / consume_chat_notes / pending_chat_notes
│   ├── add_oneshot_job / list_pending_oneshot_jobs / mark_oneshot_consumed
│   ├── save_run_preset / list_run_presets / delete_run_preset
│   └── append_audit / list_audit
└── web_forms.py          # Form schema derivation (FR-064)
    ├── FormField / FormSection / FormParseError
    ├── kind_for_type_tuple    # bool→checkbox, int/float→number, list/dict→JSON textarea
    ├── build_section          # Pulls from _KNOWN_NESTED_KEYS + _TYPE_SCHEMA
    ├── all_sections           # All top-level config sections
    ├── parse_value / parse_section_post  # Form POST → typed Python values
    └── renderable_dotted_keys # Coverage gate for the drift detector test
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
        route_after_start (FR-045):
          - change_request_mode=True → ingest_change_requests_node →
              reverse_engineer_architecture_node (if first contact, FR-046) →
              discovery pipeline runs in DELTA MODE
          - skip_discovery=True                → patching_node
          - else                                → requirements_discovery_node
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
14b. --dev-deployment set? ─no─▶ [END]  (FR-044 — deployment phase is opt-in)
    │ yes                      │
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
│ spec_architecture_path   │ write_spec_node, reverse_engineer_architecture│
│ deployment_blueprint_path│ generate_deployment_spec_node                 │
│ dev_deployment           │ run_graph (from --dev-deployment CLI flag)    │
│ change_request_mode      │ run_graph (from CLI / wizard)                 │
│ change_requests_dir_abs  │ run_graph (resolved folder path)              │
│ change_request_files     │ ingest_change_requests_node                   │
│ archive_target_dir       │ run_graph (applied/<sid>/ destination)        │
│ change_requests_config   │ run_graph (from config["change_requests"])    │
│ deployment_defaults      │ run_graph (from config/deployment.json)       │
│ sandbox_config           │ run_graph (from config["sandbox"])            │
│ lintgate_config          │ run_graph (from config["lintgate"])           │
│ deployment_config        │ run_graph (from config["deployment"])         │
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

### 5.31 Opt-In Deployment Phase (`--dev-deployment`)

**Decision**: The deployment phase is gated by a new CLI flag `--dev-deployment` (default `False`). The flag is threaded into `AgentState["dev_deployment"]` via `run_graph(dev_deployment=...)` and `create_initial_state(dev_deployment=...)`. `route_after_security_scan` reads `state.get("dev_deployment", False)` after a clean scan; without it the router returns `"__end__"`. The narrower `deployment.enabled` config switch still exists and gates only the docker step inside `deployment_node` (so a user can opt into the phase but still skip `docker compose up`).

**Rationale**: Users split cleanly into two camps — "generate code and ship via my own pipeline" and "bring it up locally." The old auto-deploy default surprised the first camp by mutating the workspace with Dockerfiles and starting containers. Making the phase opt-in respects both intents, and keeping `deployment.enabled` as an independent narrower gate avoids forcing the operator to choose between "no phase at all" and "phase including docker run."

**Trade-off**: Default behaviour changed — previously a clean security scan auto-rolled into deployment; now the run ends and the operator must re-run with `--dev-deployment`. CI scripts and demo recordings that relied on auto-deploy need to add the flag. The `[cli] Code generated at <path>. Deployment phase skipped.` log line makes the new default visible on the first post-upgrade run.

### 5.32 Change-Request Folder Mode (FR-045 / FR-046)

**Decision**: Existing-project runs are file-driven via a `change_requests/` folder at the workspace root. `ingest_change_requests_node` walks the top-level `.txt` files (skipping `applied/`), assigns monotonic `CR-N` IDs that respect operator-supplied filename prefixes, concatenates the file contents under `# === CR-N: <path> ===` headers, and injects the result as the LLM's first user message. The CR IDs propagate through specs, source comments, test names / docstrings, deployment-blueprint `cr_attribution`, and the commit trailer. At session end `_archive_consumed_change_requests` moves the consumed files into `change_requests/applied/<session-id>/` with a `manifest.json` recording status (`success` / `cancelled` / `failed-build`). When the workspace has no prior `docs/SPEC_ARCHITECTURE.md`, `reverse_engineer_architecture_node` runs once (budget-gated by `change_requests.reverse_engineer_budget_usd`, default $0.50) to synthesize a baseline before discovery enters delta mode.

**Rationale**: Brownfield work is the dominant use case — the harness was created for greenfield scaffolding but operators kept reaching for it on existing repos. A file-driven, CR-N-tagged workflow gives the operator (and reviewers) a single artifact ID that links every spec line, source comment, test, and infra change back to one ask. `grep -rn "CR-7" .` after the session shows the entire footprint of a single change request, which is what code-review and audit need.

**Trade-off**: The change-request flow runs through the same HITL gates as greenfield, so users do not get to skip discovery just because the repo "already exists." Delta-mode discovery and gatekeeper short-circuits keep the friction proportional to the size of the change. The `applied/<session-id>/` archive accumulates over time; operators rotate or prune it manually.

### 5.33 Optional Org-Wide `deployment.json` Policy + Enter-to-Accept Discovery (FR-048)

**Decision**: Discovery prompts now bake a default value into each question, and a bare Enter records the default as the answer. An optional `~/.harness/deployment.json` (or `config/deployment.json`; template at `config/deployment.json.example`) declares pre-resolved deployment-discovery fields (`target_environment`, `container_runtime`, `reverse_proxy`, `secret_store`, …). `load_deployment_defaults()` loads it at startup and `deployment_discovery_node` injects the resolved fields into the planner prompt so no question fires for them. Absent file = full questionnaire as before.

**Rationale**: Most teams have a fixed deployment story (same target env, same reverse proxy, same secret store across all projects). Re-answering the same questions on every project was friction with no signal. An org-wide policy file with per-question defaults captures the team's standard once and lets the discovery loop focus on the project-specific unknowns.

**Trade-off**: Defaults baked into discovery prompts can drift from policy if both are edited independently; the org-wide file wins because it is loaded at startup and treated as already-resolved by the planner. The `_KNOWN_NESTED_KEYS` whitelist is the authoritative schema for `deployment.json` — typos there get the same WARN-and-ignore treatment as the main config.

### 5.34 Setup Wizard for Bare `harness run` (FR-047)

**Decision**: When `harness run` is invoked without `-r`, `-p`, or `--manifest`, the CLI hands off to `harness/wizard.py:run_setup_wizard`. The wizard first asks new-vs-resume; for new it walks workspace → prompt-source → `--new_build` (defaults `false` so the harness does not clobber files in an existing repo) → `--git`. Resume lists checkpointed sessions newest-first and re-enters `cmd_resume`. Any direct flag bypasses the wizard.

**Rationale**: The harness used to fail with an argparse error when invoked bare. Operators ran `harness run --help`, hand-built a command, and often missed `--git` or `--new_build`. A wizard turns first-run discovery into a guided dialog without changing the contract for power users — flags still work.

**Trade-off**: Two paths to invoke the same `cmd_run` (wizard vs flags). Tests exercise both; the wizard is kept thin (it just resolves into the same `args` namespace `argparse` would have produced).

### 5.35 Single Kitchen-Sink Builder Image (FR-050)

**Decision**: `harness/vendor/Dockerfile.builder` ships a multi-stack base image (Python + Node + Go + Java + Rust + Dart + Make). The old per-build-command image dispatch in `harness/graph.py` is retired — compiler, lintgate, and test-generation nodes all run inside the same container. Slim toolchain images (`python:3.12-slim`, `node:20-slim`, …) are still honoured as swappable bases when pinned in `sandbox.docker_image`; the sandbox layer bootstrap-installs `make` if the chosen image doesn't ship it.

**Rationale**: Per-command image dispatch was a constant source of cache thrash — a polyglot workspace (Python service + React frontend) needed two images, each with its own pip / npm cache, and the dispatch logic in graph.py grew an exception for every new stack. A single kitchen-sink image collapses the matrix, lets the prefix cache stay hot across compile / lint / test, and removes the dispatch code path entirely.

**Trade-off**: The builder image is ~3-4 GB on first pull. Once pulled, every subsequent build reuses the layer cache and per-build sandbox startup time drops by ~1.5s vs the dispatch path. Operators who want a smaller surface can still pin a slim image via `sandbox.docker_image` and accept the missing toolchains.

### 5.36 MCP Client over Hand-Rolled JSON-RPC (FR-051)

**Decision**: `harness/mcp_client.py` implements MCP client transport (stdio only in v1) via hand-rolled JSON-RPC 2.0 over newline-delimited JSON frames. No dependency on the upstream `mcp` Python SDK. Server commands flow through `harness/trust.py:validate_mcp_server_command` before any subprocess spawn. `McpClientPool` owns one `StdioMcpClient` per declared server; each tool advertised by `tools/list` registers as an `McpToolSkill` in the global `SkillRegistry` under the `mcp__<server>__<tool>` naming convention.

**Rationale**: MCP is the de-facto interop standard (Claude Code, Cursor, Continue.dev, Goose, Cline, OpenAI agents all speak it). Adding it unlocks the entire MCP server ecosystem (databases, browsers, GitHub, search) without writing one-off tool integrations. The hand-rolled JSON-RPC keeps the core install dependency-clean — the upstream SDK pulls in ~10 transitives. The protocol surface we need (initialize / tools/list / tools/call) is ~150 LoC.

**Trade-off**: We track the spec ourselves; spec evolution (e.g. WebSocket transport, prompts/resources/sampling capabilities) will land as additive code rather than via a library bump. v1 explicitly defers HTTP/SSE transport and the non-tools capabilities — most servers in the wild are stdio + tools.

### 5.37 Provider Prompt Caching + Prefix-Stability Hasher (FR-052)

**Decision**: `harness/gateway.py:AnthropicProvider.chat_completion` rewrites `payload["system"]` to list-of-blocks form with `cache_control: {"type": "ephemeral"}` when the model carries `supports_cache=True` and the gateway flag `prompt_cache_enabled=True`. The first user message gains a second cache breakpoint when ≥ 4 KB. Independently of provider, `Gateway.dispatch` runs a prefix-stability hasher (`hash_stable_prefix`) over the first two messages keyed by `(session, role)` and emits a `cache_prefix_drift` observability event when consecutive calls disagree. OpenAI / DeepSeek auto-cache cost accounting was already wired in `extract_usage` + `compute_cost`; the hasher surfaces silent cache misses against those providers too.

**Rationale**: Caching is the dominant cost optimisation available — Anthropic prefix caching alone cuts long-session cost 60-90% on repeat dispatches. The hasher exists because OpenAI / DeepSeek auto-caches fire only on byte-identical prefixes; one whitespace drift in the planning blueprint silently kills the hit. The drift event makes the leak observable.

**Trade-off**: Anthropic's list-of-blocks payload shape is a request-side change; `prompt_cache_enabled=false` is the single-flag rollback if a future API change rejects the shape. The drift hasher's `(session, role)` key is heuristic — it correctly catches the common cases (impact analysis context, READ_FILE results, planning blueprint) but misses cross-role caching opportunities.

### 5.38 Web Research Tools via Text-DSL (FR-053)

**Decision**: `harness/web_tools.py` ships `WebFetchSkill` + `WebSearchSkill` as `ToolSkill` instances. The planner emits `<<<WEB_FETCH url="...">>>` / `<<<WEB_SEARCH query="...">>>` blocks; `harness/graph.py:_run_tool_loop` intercepts them, dispatches via `SkillRegistry`, appends the result back as a `user` message, and re-dispatches up to `tool_call_cap_per_dispatch` rounds. Default search backend `duckduckgo_lite` requires no API key. Outbound URLs gate through `harness/trust.py:validate_outbound_url` (SSRF guard). Content-type allowlist + `max_bytes` cap on every fetch.

**Rationale**: Peer agents routinely fetch docs and search the web; the harness had no primitive. Text-DSL keeps consistency with the existing patcher DSL (`<<<READ_FILE>>>`, `<<<CREATE_FILE>>>`) and avoids waiting on the unfinished native-function-calling wiring. The SSRF guard is hard-required — the LLM could otherwise trick the harness into hitting AWS metadata endpoints.

**Trade-off**: Text-DSL means the LLM can emit malformed blocks; the parser is regex-bounded so malformed blocks fall through as literal text. When native function-calling lands the same `ToolSkill.to_tool_schema` JSON is reusable; the migration is additive.

### 5.39 GitHub Integration via `gh` CLI (FR-054)

**Decision**: `harness/github_integration.py` shells out to the `gh` CLI for issue read, PR create, PR comment. No new Python dep. The `harness gh issue --repo X --number Y` subcommand writes the issue body into the workspace's `change_requests/CR-<N>-<slug>.txt` so the existing change-request flow (PR-1 → PR-3) handles the planning + patching. Authentication flows through `gh auth status` — we don't store tokens.

**Rationale**: The `gh` CLI is the canonical GitHub interface; it handles auth, rate limits, and API surface evolution. Using it directly means we don't ship our own GitHub client and avoid the maintenance tax of an OAuth flow + token storage. Wiring into the existing CR flow means the harness's change-request machinery (delta-aware planning, CR-N markers in patches, archive on success) just works for issue-driven sessions.

**Trade-off**: Operators must have `gh` on PATH; corporate environments without it need to install it separately. The harness can't ship a vendored binary because `gh`'s platform variants are non-trivial.

### 5.40 Runtime-Extensible Skills Directory (FR-055)

**Decision**: `register_builtin_skills(config)` walks `~/.harness/skills/` (or the path named by `skills.user_skills_dir`) and imports every non-`_`-prefixed `*.py` file via `importlib.util.spec_from_file_location`. Each file's module-level body runs at import time; calls to `harness.skills.register(MySkill(...))` populate the global registry. Per-file try/except wraps each import so one bad file logs and is skipped.

**Rationale**: Pre-shipped style guides (`harness/skills/*.md`) are not user-extensible. Letting operators drop `*.py` files into a directory matches the Claude Code skills loader pattern and keeps the harness's plugin model boringly simple — no manifest files, no entry point declarations, just import + register.

**Trade-off**: Running arbitrary Python at startup is a foot-gun if `~/.harness/skills/` is world-writable; the doctor doesn't probe this. Operators are expected to treat the directory like their `~/.bashrc` — trusted source.

### 5.41 Repository Semantic Retrieval (FR-056)

**Decision**: `harness/repo_index.py` ships two backends behind an `IndexBackend` ABC: `TfidfBackend` (default, zero-dep, deterministic, identifier-aware tokenisation that splits CamelCase / snake_case into sub-tokens) and `OpenAIEmbeddingsBackend` (opt-in via `OPENAI_API_KEY`; falls back to TF-IDF on missing key). Index storage: SQLite at `~/.harness/repo_index/repo_index.db` with `(workspace_id, file_path, chunk_index, file_sha, content, vector_json)`. `planning_node` calls `async_query_top_chunks` once per dispatch and injects the top-K chunks as a system message capped at `inject_max_bytes`. CLI: `harness index {build, status, clear}`.

**Rationale**: `harness/impact.py` is AST-only ("which symbols reference this function"); semantic retrieval complements it with "which other code chunks are semantically similar to the prompt". TF-IDF is the right default — deterministic, no API cost, no model download, identifier-aware tokenisation works surprisingly well on code. OpenAI embeddings is the upgrade path; both backends share the same storage schema so swapping is config-only.

**Trade-off**: TF-IDF's relevance ceiling is lower than dense embeddings; the gap shows on prompts that paraphrase concepts ("auth flow" vs `def authenticate(...)`). Operators who care upgrade by flipping `repo_index.backend=openai_embeddings`.

### 5.42 Per-Repository Session Memory (FR-057)

**Decision**: `harness/repo_memory.py` writes a flat markdown log per repository at `~/.harness/memory/<repo_id>.md`, where `repo_id` is the first 16 hex chars of `sha256(git remote get-url origin)` (or the workspace path when no remote). Each session ends with an append (prompt summary, modified files, exit code) atomically via `tempfile + os.replace`. Read happens at `planning_node` start; recent entries inject as an extra system message. FIFO trim by `## Session` boundary keeps the file under `memory.max_bytes`.

**Rationale**: LangGraph checkpoints resume one thread only — there's no per-repo "what we learned last time". A small markdown log gives the planner continuity across sessions for free. The `git remote get-url origin` identity makes the log portable across clones of the same repo so cross-machine continuity works without operator config.

**Trade-off**: The memory file is markdown, not structured data — the planner reads it as opaque context. We could persist structured fields (sectors discussed, files touched, design decisions) but the cost of schematising outweighs the value when the planner can pattern-match on prose.

### 5.43 Interactive Refinement REPL (`harness chat`) (FR-058)

**Decision**: `harness/chat.py` opens an interactive stdin loop that reuses the Gateway, redactor, web/MCP `_run_tool_loop`, repo-memory injection, and (when enabled) repo-index injection. Patches are NEVER applied automatically; the LLM emits SEARCH/REPLACE blocks but they only land when the operator types `/apply` and confirms. Slash commands: `/help`, `/exit`, `/clear`, `/files`, `/apply`, `/build`, `/save`, `/budget`, `/memory`. The REPL accepts dependency-injectable reader/writer hooks so unit tests drive it with scripted input.

**Rationale**: `harness run` is autonomous; `harness chat` is the inverse — conversational back-and-forth where the operator stays in control. Reusing the gateway / redactor / tools means budgeting, secret stripping, and web/MCP all keep working. The dependency-injection on reader/writer is the test-affordance pattern from `harness/hitl.py:HitlChannel`.

**Trade-off**: In-memory only in v1 — closing the REPL loses the conversation. `--resume` is a clean follow-up; persistence would land in the existing checkpoint store.

### 5.44 Multi-Agent Fan-Out Primitive (FR-060)

**Decision**: `harness/fanout.py` provides `AgentSpec` / `AgentResult` / `Verdict` dataclasses, `run_parallel_agents` with bounded asyncio semaphore concurrency + shared-budget reservation/refund, and `run_with_verification` for the adversarial-skeptic pattern (one finder + N skeptics; majority vote). The runner is exposed to the planner as a `SubAgentFanoutSkill` registered in `SkillRegistry`; the planner can emit `<<<FANOUT_QUERY prompts='[...]'>>>` for N parallel queries.

**Rationale**: Speculative execution (`harness/speculative.py`) and `SubAgentSkill` already had fan-out for narrow purposes; lifting it into a reusable primitive lets future graph integrations (parallel discovery per sector, parallel test generation per module, parallel security-fix attempts per finding) build on the same machinery. The reservation/refund accounting prevents N concurrent dispatches from overspending the cap.

**Trade-off**: Budget accounting is per-call, not per-stage — a fan-out can spend the whole remaining budget if every agent's `budget_hint` is high enough. Operators control this via the per-call `max_concurrency` and `budget_hint` knobs.

### 5.45 Configuration-Driven Speculative Execution (FR-061 — rebuild)

**Decision**: `harness/speculative.py:speculate_node` exposes six independent strategy axes via config: `trigger` (when to engage), `diversity_mode` (how variants differ), `cost_strategy` (cost shape), `selection_strategy` (who wins), `salvage_strategy` (what to do when all fail), `voting` (judges for `selection_strategy=voted`). The default config (`trigger=after_n_repair_failures`, `diversity_mode=model`, `cost_strategy=cheap_first_sequential`, `selection_strategy=first_pass`, `salvage_strategy=none`) targets positive ROI by holding speculative back until sequential repair has stalled and by using cheaper models. `_upgrade_legacy_config` maps old-shape configs to byte-identical legacy behaviour with a one-time WARNING.

**Rationale**: The original speculative shipped disabled because the fixed pipeline (3 same-model variants at temp 0.3 → first-pass → merge-on-fail) had negative ROI: variants converged on the same wrong solution, merge salvage produced incoherent workspaces, repair couldn't recover. The rebuild factors out the orchestration decisions so operators can pick a strategy that fits their workload.

**Trade-off**: Six axes is a lot of config surface. The shipped defaults are chosen carefully; documentation and the `harness/web_forms.py` schema derivation keep the surface manageable.

### 5.46 Cron-Driven Scheduled Job Daemon (FR-062)

**Decision**: `harness/schedule.py` ships a foreground daemon (`harness schedule run`) that ticks every `tick_seconds` (default 60s), polls config-declared jobs in `schedule.jobs` AND web one-shot jobs in `~/.harness/web.db:web_oneshot_jobs`, and fires due jobs as `harness run` subprocesses with their own per-job log file. History persists to `~/.harness/schedule.db`. The cron syntax subset (`every Nm/h/d`, `hourly :MM`, `daily HH:MM`, `weekly DAY HH:MM`) is hand-rolled, no `croniter` dep. Notifications are generic shell hooks (`on_success` / `on_failure`) with `HARNESS_JOB_*` env vars exported.

**Rationale**: Recurring runs ("regenerate failing tests every night", "open the security review every Monday") are high-value workloads. A foreground daemon under systemd / docker is simpler than reinventing process management. The hand-rolled cron subset covers >90% of real use cases; full POSIX cron is a clean follow-up. Generic shell hooks let operators wire Slack/Discord/PagerDuty/email in one curl line without us shipping per-vendor notifier code.

**Trade-off**: Hand-rolled cron means we don't support `30 2 * * mon` — operators who need the full mini-language must either restructure or wait for the croniter follow-up. In-flight tracking is per-daemon-process; two daemons reading the same `schedule.db` would double-fire. Operators run one daemon per scheduler.

### 5.47 Web Dashboard — Read-Only Tier (FR-063)

**Decision**: `harness/dashboard.py` runs a stdlib `http.server.ThreadingHTTPServer` (default bind `127.0.0.1:8729`). Routes render the harness's on-disk state: sessions list + per-session detail (from `~/.harness/logs/*.jsonl`), cost burn-down chart (Chart.js via CDN), scheduled-job history (from `~/.harness/schedule.db`), repo-index status (from `~/.harness/repo_index/repo_index.db`), per-repo memory files. Routing is a pure `dispatch(cfg, path)` function so tests exercise it without standing up a server. Optional bearer-token auth via `dashboard.token_env`; the server refuses to start when the named env var is empty (fail-closed).

**Rationale**: Operators wanted to see session history, cost, and scheduled-job state without tailing JSONL files. The dashboard is one artefact (no editor extension to install). Server-rendered HTML + Chart.js CDN keeps the build story trivial (no npm). Localhost-only default + opt-in token auth covers both single-user laptop and remote-access scenarios safely.

**Trade-off**: Stdlib `http.server` is sync + threaded; high-traffic scenarios would prefer aiohttp/starlette. For single-operator dashboard load it's plenty.

### 5.48 Interactive Web App — Tier B + C (FR-064)

**Decision**: `harness web` (with the default `dashboard.writes_enabled: true`) extends the read-only dashboard with: form-based config editing (forms derived from `harness/web_forms.py:build_section` which walks `harness/cli.py:_KNOWN_NESTED_KEYS` + `_TYPE_SCHEMA`), memory-file editing, schedule-job CRUD, a "New run" form supporting both "Run now" (`harness/dashboard.py:spawn_harness_run` spawns subprocess + registers PID in `harness/web_state.py:ProcessRegistry`) and "Schedule it" (appends to `~/.harness/web.db:web_oneshot_jobs` which the schedule daemon picks up), SSE event stream at `/api/sessions/<id>/events`, HITL bridge via the existing `harness/hitl.py:HttpChannel` (the dashboard registers as the webhook URL and blocks the harness's POST while the UI displays the prompt, signalling back when the operator answers), and per-session chat notes queued for the next HITL gate. Write paths require a CSRF double-submit cookie + `X-CSRF-Token` header.

**Rationale**: Form-based config editing eliminates a class of operator errors (typos, wrong types) without us hand-curating a per-section UI — the forms are *derived* from the strict validator, so new config keys appear in the UI the day they land in `_TYPE_SCHEMA`. The HITL bridge reuses `HttpChannel` rather than reinventing — `HttpChannel` already exists exactly for this scenario; the dashboard is its first consumer. CSRF double-submit cookie is the simplest defence against cross-origin form submission while remaining usable from vanilla JS.

**Trade-off**: One Python file owns ~2,500 LoC of mixed HTTP + HTML + Python. A future split into a `harness/dashboard/` package is straightforward. The HITL "block the webhook handler until the operator answers" pattern relies on threaded request handling (works because `ThreadingHTTPServer`); under aiohttp it would be a long-poll. Both work.

### 5.49 Web App State Layer (`harness/web_state.py`)

**Decision**: A separate module owns the dashboard's runtime state so `harness/dashboard.py` stays focused on HTTP routing + rendering. `web_state.py` holds: `ProcessRegistry` (`{session_id → WebProcess}` of spawned subprocesses with TTL after termination), `HitlQueue` (`{request_id → PendingHitl}` with `threading.Event` to release blocked webhook handlers), and a small SQLite schema at `~/.harness/web.db` with tables `audit_log`, `run_presets`, `web_oneshot_jobs`, `chat_notes`. The schedule daemon imports `list_pending_oneshot_jobs` / `mark_oneshot_consumed` to drain the one-shot table during its tick.

**Rationale**: Splitting state from HTTP makes the unit tests trivial — `tests/test_web_state.py` exercises the contracts without standing up a server. The web.db is operator-local and never shared across hosts; the four tables are intentionally narrow.

**Trade-off**: An in-memory registry means the dashboard restart drops the live process tracking; the watcher threads that mark `exit_code` survive only as long as the dashboard. Operators who need persistence run the schedule daemon for everything (cron-style) and use the dashboard purely for observability.

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
├── deployment_blueprint_path: str
├── dev_deployment: bool             # Opt-in deployment phase gate (FR-044)
├── change_request_mode: bool        # True when change_requests/ folder was found / chosen (FR-045)
├── change_requests_dir_abs: str     # Resolved abs path of change_requests/
├── change_request_files: list[dict] # [{cr_id: int, original_name: str, abs_path: str}]
├── archive_target_dir: str          # change_requests/applied/<session-id>/ (set in run_graph)
├── change_requests_config: dict[str, Any]  # config["change_requests"] (e.g. reverse_engineer_budget_usd)
├── deployment_defaults: dict[str, Any]     # Org-wide deployment.json policy (FR-048); empty when absent
├── sandbox_config: dict[str, Any]          # config["sandbox"] threaded into state (P0)
├── lintgate_config: dict[str, Any]         # config["lintgate"] threaded into state
└── deployment_config: dict[str, Any]       # config["deployment"] threaded into state
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

### 6.4 Web App SQLite Store (`~/.harness/web.db`)

The dashboard's runtime state and operator-driven mutations persist
to a small SQLite at `~/.harness/web.db` (configurable via
`dashboard.web_db_path`). Four tables, deliberately narrow:

```
web_db
├── audit_log
│   ├── id INTEGER PK AUTOINCREMENT
│   ├── ts TEXT          # ISO8601 UTC
│   ├── action TEXT      # config_save | run_now | run_schedule | cancel | hitl_answer | …
│   ├── target TEXT      # section name / session id / job id
│   └── detail TEXT      # JSON-ish, opaque
├── run_presets
│   ├── name TEXT PK
│   ├── workspace TEXT
│   ├── prompt TEXT
│   ├── harness_args TEXT   # JSON list
│   └── created_at TEXT
├── web_oneshot_jobs
│   ├── id INTEGER PK AUTOINCREMENT
│   ├── name TEXT
│   ├── fire_at_utc TEXT     # ISO8601 UTC
│   ├── workspace TEXT
│   ├── prompt TEXT
│   ├── harness_args TEXT    # JSON list
│   ├── created_at TEXT
│   └── consumed_at TEXT     # NULL until schedule daemon fires it
└── chat_notes
    ├── id INTEGER PK AUTOINCREMENT
    ├── session_id TEXT
    ├── ts TEXT
    ├── note TEXT
    └── consumed_at TEXT     # NULL until prepended to next HITL extra_notes
```

The schedule daemon's `tick_once` reads `web_oneshot_jobs` alongside its
config-driven jobs; entries fire once and are marked `consumed_at`.
The dashboard's `_handle_hitl_answer` drains pending `chat_notes` for
the session and prepends them to the response's `extra_notes`.

### 6.5 Repo Index SQLite Store (`~/.harness/repo_index/repo_index.db`)

Per-workspace semantic retrieval index. Two tables:

```
repo_index.db
├── repo_meta
│   ├── workspace_id TEXT PK  # SHA-256 hex prefix of absolute workspace path
│   ├── backend TEXT          # "tfidf" | "openai_embeddings"
│   ├── idf_json TEXT         # TF-IDF IDF vector serialised for query-time use
│   ├── built_at TEXT         # ISO8601 UTC
│   └── chunk_count INTEGER
└── repo_chunks
    ├── workspace_id TEXT
    ├── file_path TEXT        # POSIX-relative
    ├── chunk_index INTEGER
    ├── file_sha TEXT
    ├── content TEXT
    ├── vector_json TEXT      # Sparse TF-IDF dict OR dense embedding list
    └── PRIMARY KEY (workspace_id, file_path, chunk_index)
```

### 6.6 Schedule History SQLite Store (`~/.harness/schedule.db`)

Per-job execution history persisted by the schedule daemon so
`harness schedule list` / `history` survives restarts:

```
schedule.db
└── schedule_runs
    ├── job_name TEXT
    ├── started_at TEXT       # ISO8601 UTC; PK component
    ├── ended_at TEXT
    ├── exit_code INTEGER
    ├── duration_sec REAL
    ├── log_path TEXT
    └── PRIMARY KEY (job_name, started_at)
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
| `~/.harness/deployment.json` (optional) | User home | Org-wide deployment-discovery defaults (FR-048); absent = full questionnaire |
| `config/deployment.json.example` | Repo root | Template for the optional org-wide deployment policy |
| `requirements-prod.txt` | Repo root | Exact transitive pins for reproducible pilot installs (`pip install -e . --constraint requirements-prod.txt`) |
| `LICENSE` | Repo root | MIT license; referenced from `pyproject.toml` so wheels ship it |

**Top-level config sections**: `build_command`, `allow_network`, `sandbox`, `token_budget`, `node_throttle`, `models`, `model_routing`, `persistence`, `logging`, `lintgate`, `deployment`, `test_generation`, `metrics`, `change_requests`, `speculative`, `compiler`.

**Key recent additions** (since the prior spec snapshot):
- `change_requests.reverse_engineer_budget_usd` (default `$0.50`) — budget cap for the first-contact architecture reverse-engineer LLM call (FR-046).
- `speculative.temperature` / `num_variants` / `strategy` — externalised speculative tuning knobs (previously hard-coded).
- `compiler.run_prod_import_smoke_check` (default `true`) — toggle the post-build production-import smoke check.
- Per-role `max_tokens` overrides in `model_routing` (e.g., `planning_max_tokens`) — externalised gateway dispatch caps.

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
- `<workspace>/change_requests/` — Operator-authored `CR-N-<name>.txt` files; consumed change-request inputs (FR-045)
- `<workspace>/change_requests/applied/<session-id>/` — Archive of consumed `.txt` files + `manifest.json` recording `status` (`success` / `cancelled` / `failed-build`) and the linked modified files
- `<workspace>/Dockerfile` / `docker-compose.yml` / `Caddyfile` — Only produced when `--dev-deployment` is passed (FR-044)
- `/tmp/.harness/` — Temporary sandbox build logs (auto-cleaned)