# Changelog

All notable changes to myharness are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

- **MAJOR** — backwards-incompatible change to the CLI surface, config
  schema, or checkpoint format.
- **MINOR** — new capability, new subcommand, new config section.
- **PATCH** — bug fix, doc update, CI fix.

## [Unreleased]

### Added
- **Tier 1 capability** — Cron-driven scheduled-job daemon (#13).
  New `harness/schedule.py` + `harness schedule {run,list,validate,
  once,history}` subcommands. Hand-rolled cron syntax subset
  (`every Nm/h/d`, `hourly :MM`, `daily HH:MM`, `weekly DAY HH:MM`)
  covers >90% of real use cases without depending on `croniter`. Each
  job runs as `harness run` in its own subprocess; SQLite history at
  `~/.harness/schedule.db` carries started/ended/exit_code/duration/
  log_path per fire. Notifications via generic shell hooks
  (`on_success` / `on_failure`) with `HARNESS_JOB_NAME` /
  `HARNESS_JOB_EXIT_CODE` / `HARNESS_JOB_DURATION_SEC` /
  `HARNESS_JOB_LOG_PATH` exported — operators wire Slack/Discord/
  PagerDuty/email in one curl line. Default off.
- **Tier 1 capability** — Read-only web dashboard (#14). New
  `harness/dashboard.py` + `harness dashboard` subcommand. Surfaces
  the harness's existing on-disk state: sessions list + per-session
  detail (from `~/.harness/logs/*.jsonl`), cost burn-down chart
  (Chart.js via CDN; air-gapped operators drop a local file into
  `dashboard.static_dir`), scheduled-job runs (from the #13 SQLite
  store), repo-index status (from #6 SQLite store), per-repo memory
  files (from #7). Default bind `127.0.0.1` so accidental public
  exposure isn't possible without an explicit config change.
  Optional bearer-token auth via `dashboard.token_env` —
  `hmac.compare_digest` constant-time match; tokens never logged.
  When `token_env` is set but the env var is empty the server
  refuses to start (fail-closed). Zero new Python dependencies —
  uses stdlib `http.server.ThreadingHTTPServer`.
- **Tier 4** — `make coverage` target backed by `pytest-cov`. Emits
  terminal summary + HTML report under `htmlcov/` + XML at
  `coverage.xml`. No CI gate on the coverage number — the metric is
  for visibility; we'll set a defensible floor only after the rapid
  feature-shipping phase settles. README's "545-test regression
  pack" claim retired in favour of the live `make coverage`
  output (#9).

### Changed
- **Tier 1 capability (rebuild)** — `harness/speculative.py` is now
  configuration-driven across six independent strategy axes (#12). The
  legacy 1056-LoC fixed pipeline (3 parallel temperature variants on
  one model → first-pass → optional merge salvage) generated negative
  ROI on observed sessions; the rebuild exposes:
  - `trigger`: `always` / `first_attempt_only` (legacy default) /
    `after_n_repair_failures` (NEW DEFAULT, threshold 2) / `manual`.
    The default holds speculative back until sequential repair has
    stalled — the high-value moment.
  - `diversity_mode`: `temperature` (legacy) / `prompt` (per-variant
    system-prompt style from a built-in library: minimal-diff,
    balanced, thorough, conservative, bold) / `model` (NEW DEFAULT —
    different models per variant; real architectural independence,
    not temperature noise) / `mixed`.
  - `cost_strategy`: `equal_cost` (legacy) / `cheap_first_sequential`
    (NEW DEFAULT — dispatch cheap_model variants one at a time, only
    use expensive_model if all cheap fail; expected cost ~1.1× rather
    than 3×) / `cheap_parallel_then_expensive` / `all_cheap`.
  - `selection_strategy`: `first_pass` (NEW DEFAULT) /
    `fewest_changes` / `voted` (NEW — dispatches `voting.n_judges`
    adversarial reviewers via the `harness/fanout.py` infrastructure
    from #11 and picks the variant with the highest accept-rate) /
    `all_pass`. The legacy `first_success` value is an accepted alias.
  - `salvage_strategy`: `none` (NEW DEFAULT — fall back to sequential
    repair against the untouched workspace, strictly safer than the
    legacy merge path which often produced incoherent workspaces) /
    `fewest_errors` / `voted_partial` / `merge` (legacy behaviour
    preserved as opt-in).
  - `voting`: `{n_judges, judge_role}` for the `voted` selection
    strategy.
  Legacy configs auto-upgrade to a backwards-compatible mapping
  (`diversity_mode=temperature`, `cost_strategy=equal_cost`,
  `salvage_strategy=merge`, `trigger=first_attempt_only`) with a
  one-time deprecation warning so existing flows are byte-identical.
  Worktree machinery, `_build_variant_cache_env`, salvage helpers,
  and `VariantResult` are unchanged — the rewrite is the
  orchestration layer only.

### Added
- **Tier 1 capability** — Interactive refinement REPL (`harness chat`).
  New `harness/chat.py` ships a stdin-driven REPL that reuses the
  Gateway, redactor, web/MCP tool-loop, repo-memory injection, and
  repo-index injection (when enabled). Slash commands: `/help`,
  `/exit`, `/clear`, `/files`, `/apply` (per-session HITL confirmation
  before applying SEARCH/REPLACE blocks from the last assistant
  reply), `/build` (runs the workspace's build command in the
  sandbox), `/save`, `/budget`, `/memory`. In-memory only — `--resume`
  is a clean follow-up. Tests inject scripted reader/writer functions
  so the REPL is unit-testable without real stdin.
- **Tier 1 capability** — Graph-level multi-agent fan-out
  infrastructure (`harness/fanout.py`). `AgentSpec` / `AgentResult`
  dataclasses, `run_parallel_agents()` with bounded asyncio semaphore
  concurrency (default 8) and shared-budget reservation/refund
  accounting, `run_with_verification()` adversarial pattern (one
  finder + N skeptics; majority vote decides). New
  `SubAgentFanoutSkill` registered in `register_builtin_skills()`
  exposes the runner to the planner as a text-DSL tool —
  `<<<FANOUT_QUERY prompts='[...]'>>>` lets the planner dispatch N
  parallel sub-queries with one block. Lifts fan-out from
  speculative.py / SubAgentSkill (single-purpose) into a reusable
  primitive that future graph integrations (parallel discovery per
  sector, parallel test gen per module, parallel security-fix
  attempts per finding) can build on.
- **Tier 1 capability** — Runtime-extensible skills directory.
  ``register_builtin_skills()`` now walks ``~/.harness/skills`` (or the
  path named by ``skills.user_skills_dir``) and imports every ``*.py``
  file at startup. Each loaded module can call
  ``harness.skills.register(MySkill(...))`` to add a ``ToolSkill``,
  ``PipelineSkill``, or ``SubAgentSkill`` without forking the repo —
  same import-side-effect contract Claude Code uses. Bad files log +
  skip without taking down startup.
- **Tier 1 capability** — Persistent per-repo session memory
  (``harness/repo_memory.py``). Planner reads
  ``~/.harness/memory/<repo_id>.md`` at the start of every ``harness run``
  and injects the recent entries as an extra system message;
  ``cmd_run`` / ``cmd_resume`` append a fresh session entry (prompt
  summary, modified files, exit status) at the end. Repo identity =
  SHA-256 of ``git remote get-url origin`` (or the workspace path when
  there's no remote), so cross-machine continuity works for cloned
  repos. FIFO trim caps the file at ``memory.max_bytes``. Default
  enabled.
- **Tier 1 capability** — GitHub integration
  (``harness/github_integration.py``). New ``harness gh issue --repo X
  --number Y`` pulls an issue body and writes it into the workspace's
  ``change_requests/CR-N-<slug>.txt`` so the existing change-request
  flow (PR-1 → PR-3) handles the rest. ``harness gh pr-create`` opens
  a PR from the current branch; ``harness gh pr-comment`` posts a
  comment. Shells out to the ``gh`` CLI (no new Python dep). Optional
  ``github.gh_path`` for non-PATH installs.
- **Tier 1 capability** — Repository semantic retrieval index
  (``harness/repo_index.py``). New ``harness index build`` /
  ``status`` / ``clear`` CLI subcommands. Two backends: a zero-dep
  ``TfidfBackend`` (default, deterministic, pure-Python TF-IDF with
  identifier-aware tokenisation) and an opt-in
  ``OpenAIEmbeddingsBackend`` (``text-embedding-3-small`` via existing
  httpx + ``OPENAI_API_KEY``). SQLite store under
  ``~/.harness/repo_index/``. When ``repo_index.enabled=true``, the
  planner queries top-K chunks for the user prompt and injects them as
  a system context block — complements the AST-based ``impact.py`` with
  semantic retrieval. Default off.
- **Tier 1 parity** — Anthropic prompt caching: `AnthropicProvider` now
  emits `cache_control: {"type": "ephemeral"}` on the system block and
  on the first user message when it exceeds 4 KB. Gated by
  `llm_dispatch.prompt_cache_enabled` (default `true`); flip to `false`
  to fall back to the legacy string-form system payload. Cost
  accounting + `cache_read_input_tokens` / `cache_creation_input_tokens`
  extraction were already wired; this fills the missing request-side
  marker emission so the discount actually fires.
- **Tier 1 parity** — Prefix-stability drift detector in
  `Gateway.dispatch`. Hashes the first 2 messages of every request,
  scoped by `(session_id, role)`; logs a warning + emits a
  `cache_prefix_drift` observability event when the prefix changes
  between calls. Surfaces silent cache-misses on OpenAI and DeepSeek
  auto-caching where the cost is already correct but the discount
  vanishes when a graph node mutates the immutable preamble.
- **Tier 1 parity** — `WebFetchSkill` + `WebSearchSkill` (new
  `harness/web_tools.py`) exposed via text-DSL blocks (`<<<WEB_FETCH
  url="...">>>` / `<<<WEB_SEARCH query="...">>>`). Default backend:
  `duckduckgo_lite` (no API key). New SSRF guard
  `harness.trust.validate_outbound_url` rejects `file://`/`javascript:`,
  loopback / link-local (incl. AWS metadata 169.254.169.254) /
  RFC-1918 hosts unless `web_tools.allow_private_ips=true`. HTML→text
  stripper, content-type allowlist, byte cap, per-dispatch tool-loop
  cap. Off by default (`web_tools.enabled=false`).
- **Tier 1 parity** — MCP (Model Context Protocol) client (new
  `harness/mcp_client.py`). Hand-rolled JSON-RPC 2.0 over stdio
  (HTTP/SSE deferred); supports the `initialize` / `tools/list` /
  `tools/call` flow per the 2024-11-05 protocol version. Each
  configured MCP server's advertised tools register as
  `mcp__<server>__<tool>` skills in `SkillRegistry`. New
  `harness.trust.validate_mcp_server_command` enforces a command
  allowlist (`npx` / `npm` / `node` / `python*` / `uvx` / `pipx` /
  `docker`), hard-deny on shells/`sudo`/`rm`, shell-metacharacter
  scan, and `/etc` `/root` `/proc` `/sys` path rejection. Filesystem
  servers (which bypass the build sandbox) gated behind
  `mcp.allow_local_filesystem_servers=true`. Off by default
  (`mcp.enabled=false`). `harness doctor` adds one row per server when
  enabled, showing the tool count or the start error.
- **Tier 1 parity** — Tool-block interceptor in `harness/graph.py`:
  new `_run_tool_loop` parses both `<<<WEB_FETCH/SEARCH>>>` and
  `<<<MCP_CALL server="..." tool="..." args='{...}'>>>` blocks emitted
  by the planner, dispatches each via `SkillRegistry`, appends the
  result as a tool-result message, re-dispatches up to a configurable
  cap, and strips the blocks before downstream consumers (patcher,
  blueprint store, repair grep) see them. Wired into `planning_node`
  only in this slice; extension to `patching_node` / `repair_node` is
  a follow-up.
- **Tier 1 parity** — `register_builtin_skills(config=…)` now called
  from `cmd_run` / `cmd_resume` (previously only invoked from tests),
  bringing the existing pipeline + docgen skill registrations live at
  runtime as well as the new opt-in web/MCP tools.
- **Tier 4** — Platform support matrix in `README.md`. Documents which
  platforms (Linux / macOS / WSL2 / Windows) are CI-tested vs
  best-effort vs unsupported, per sandbox backend (`docker` / `unshare` /
  `bare`). T4.1 (web dashboard) is intentionally deferred — the audit
  itself flagged it as out of scope for v1.x.
- **Tier 2** — `CONTRIBUTING.md` covering pre-commit gate behavior, test
  layout, commit-message convention, SemVer policy, and the scope rules
  the project enforces on PRs.
- **Tier 2** — `CHANGELOG.md` (this file) and `make release` target that
  verifies a clean tree, runs tests, bumps the version, tags, and pushes.
- **Tier 2** — `harness.observability.log_failure(name, **fields)` helper
  plus a catalogue of named failure events: `sandbox_start_failed`,
  `token_budget_exhausted`, `hitl_gate_blocked`. Failures can now be
  grepped by event name from JSONL session logs instead of by string
  fragment.
- **Tier 1** — `harness doctor` subcommand: five healthchecks (git repo,
  API keys per routed provider, sandbox backend reachable, checkpoint DB
  writable, config parses cleanly) with green/yellow/red markers; non-zero
  exit on any failure.
- **Tier 1** — GitHub Actions CI workflow running `pytest` on push to
  `main` and PRs, matrix on Python 3.11 / 3.12 / 3.13.
- **Tier 1** — Recursive config typo detection: `_validate_config_keys`
  now walks known nested sections (`sandbox`, `token_budget`,
  `persistence`, `model_routing`, `deployment`, `lintgate`, `logging`,
  `node_throttle`) with fuzzy-match suggestions, so typos like
  `token_budget.hrad_cap_usd` surface as `did you mean 'hard_cap_usd'?`
  instead of silently no-op-ing.
- **Tier 1** — README rewrite: quick-start, command reference with flag
  tables, configuration overview, troubleshooting matrix keyed to
  `harness doctor` output.

### Fixed
- `tests/test_hitl.py` used `Optional[list]` in a function signature
  without importing `Optional`. Python 3.14 evaluates defaults lazily
  (PEP 649), so the `NameError` never fired locally; 3.11–3.13 raised at
  import. Module-level import added.
- `msgpack` pinned as a dev dep. `langgraph-checkpoint-sqlite` switched
  to `ormsgpack`, so the storage GC regression test (which builds a
  msgpack blob directly) no longer pulled it in transitively. Runtime
  path already handles `msgpack` missing.

## [1.0.0] - initial

Initial commit: LangGraph-orchestrated agent harness with sandboxed
builds (Docker / unshare / bare), SQLite checkpoint store with WAL + TTL
GC, model-agnostic gateway (OpenAI / Anthropic / DeepSeek / Ollama),
three-phase HITL gate (Requirements / Architecture / Deployment),
tree-sitter-backed multi-stack parsing (Python / Java / Node / Dart /
Flutter), structured logging with optional LangSmith tracing, and a
verified regression pack (current count surfaced via `make coverage`;
the file's text originally claimed "545 tests" — the actual number
has been higher since the slice batch landed and the claim has been
moved out of the docs and into the live make target).

Note: v1.0.0 is the version declared in `pyproject.toml` from project
inception; it was not git-tagged. The first formal tagged release will
be the Tier 1 + Tier 2 closeout above.

[Unreleased]: https://github.com/akmontheweb/myharness/commits/main
