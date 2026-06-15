# myharness

Production-grade, model-agnostic AI agent harness. LangGraph-orchestrated code
generation with sandboxed builds, structured persistence, multi-stack support
(Python / Java / Node / Dart / Flutter), and a hardened human-in-the-loop gate.

## What myharness is

An autonomous engineering loop you point at a repo and a task description. It
synthesizes a requirements spec, plans an architecture, generates and patches
code, runs the build in an isolated sandbox, and repairs compiler errors —
checkpointing every step to SQLite so a `Ctrl-C` is recoverable. Model routing
is per-role (planning / patching / repair) and provider-agnostic: OpenAI,
Anthropic, DeepSeek, and local Ollama models are all first-class.

## Quick start

The fastest way on a fresh machine — works on Linux, macOS, Windows
(WSL2 or native) — is the interactive bootstrap script:

```bash
git clone <repo-url> myharness && cd myharness
python3 scripts/setup.py        # or `make setup`
```

The script probes the platform, locates Python 3.11+, creates a venv,
runs `pip install -e .`, prompts for an LLM provider + API key, writes
`~/.harness/config.json`, runs `harness doctor`, and offers to print the
install commands for optional scanners and formatters. Re-running is
idempotent — existing venvs and config files are detected and reused.

Manual path (each step in detail):

```bash
# 1. Install
pip install -e ".[dev]"
make hooks-install

# 2. Configure API keys (only the providers you actually use)
export ANTHROPIC_API_KEY=sk-...
export OPENAI_API_KEY=sk-...
export DEEPSEEK_API_KEY=sk-...

# 3. Verify the install
harness doctor

# 4. Run on a target repo
harness run -r /path/to/repo -p "Add JWT authentication to the login endpoint"
```

For a pilot install where you need bit-exact reproducibility (recommended
when shipping to a paying customer), use the pinned constraints file:

```bash
pip install -e . --constraint requirements-prod.txt
```

`requirements-prod.txt` records the exact versions of every direct and
transitive dependency. Regenerate it after a deliberate dependency bump
with `pip freeze | grep -v '^-e ' | sort > requirements-prod.txt`.

On first run, `myharness` auto-generates `.harness_config.json` in the
workspace from your global config plus shipped defaults. Edit it to customize
per-project settings (model routing, build command, sandbox limits).

For a full deployment walkthrough — prerequisites, sandbox setup, API keys,
and platform-specific notes for Linux / macOS / Windows (WSL2 + native) —
see [docs/installation.md](docs/installation.md).

A `harness run` stops after a clean security scan by default — the
workspace holds the generated code and you can take it to whatever
deployment pipeline you already use. Pass `--dev-deployment` to opt into
the local docker-compose dev environment: the harness then walks
deployment discovery, generates `Dockerfile` / `docker-compose.yml` /
`Caddyfile` (and a `DEPLOYMENT_BLUEPRINT.md`), and brings the stack up
with health checks. The separate `deployment.enabled` config flag is a
narrower switch: it only gates the final `docker compose up` step once
the deployment phase is already running. See
[docs/app-deployment.md](docs/app-deployment.md) for the artefact
contract, the preview gate, and how to bring the same setup up on a
different host.

## Command reference

| Command | Purpose |
|---------|---------|
| `harness run` | Execute the full agent graph on a workspace. |
| `harness resume` | Resume a crashed or interrupted session from its checkpoint. |
| `harness chat` | Interactive refinement REPL — reuses the gateway, tools, and memory; no auto-apply. |
| `harness status` | Read-only inspection of a checkpointed session. |
| `harness doctor` | Run first-run healthchecks (git, API keys, sandbox, DB, MCP servers, config). |
| `harness purge` | Wipe checkpoint data. |
| `harness metrics` | Per-session cost / burn-rate / Prometheus aggregation from logs. |
| `harness index build/status/clear` | Manage the per-workspace semantic retrieval index (TF-IDF default; OpenAI embeddings opt-in). |
| `harness gh issue/pr-create/pr-comment` | GitHub integration via the `gh` CLI — issue → change-request, PR open, PR comment. |
| `harness cache clear` | Remove harness-owned Docker cache volumes (idempotent). |
| `harness --version` | Print the installed harness version and exit. |

### `harness run`

| Flag | Purpose |
|------|---------|
| `-r`, `--workspace` | Path to the target repository root (required). |
| `-p`, `--prompt` | The engineering task description (required). |
| `-m`, `--manifest` | Path to raw notes to synthesize into `SPEC_REQUIREMENTS.md`. |
| `-o`, `--output-dir` | Directory for the synthesized spec (default: `./docs`). |
| `--build-cmd` | Override the build command (otherwise from config or `make build`). |
| `--session-id` | Human-readable session ID (auto-generated UUIDv7 otherwise). |
| `--thread-id` | LangGraph thread ID for checkpoint lookups (defaults to session). |
| `--allow-network` | Permit outbound network traffic in the sandbox. |
| `--discover` | Run the full requirements / architecture / deployment interview. |
| `--dev-deployment` | After a clean security scan, continue into deployment discovery, `DEPLOYMENT_BLUEPRINT.md`, gatekeeper approval, and `docker compose up`. Off by default — without this flag the run ends after code generation. |
| `-v`, `--verbose` | Debug-level logging. |

### `harness resume`

| Flag | Purpose |
|------|---------|
| `--session-id` | Session/thread ID to resume (required). |
| `-r`, `--workspace` | Workspace path (auto-detected from checkpoint if omitted). |
| `-p`, `--prompt` | Optional additional prompt appended to the resumed session. |
| `--build-cmd` | Override the build command. |
| `--allow-network` | Permit outbound network in the sandbox. |
| `-v`, `--verbose` | Debug-level logging. |

### `harness chat`

Interactive refinement REPL. Reuses the LLM gateway, web/MCP tool
loop, per-repo memory, and (when enabled) the semantic-retrieval
index. Patches the LLM emits are NEVER applied automatically — type
`/apply` to commit them after reviewing.

| Flag | Purpose |
|------|---------|
| `-r`, `--workspace` | Workspace path. Defaults to current directory. |
| `--budget` | Optional per-session USD budget cap. Falls back to `token_budget.hard_cap_usd`. |

In-REPL commands: `/help`, `/exit`, `/clear`, `/files`, `/apply`,
`/build`, `/save <path>`, `/budget`, `/memory`. Anything not starting
with `/` is sent to the LLM as a user message.

### `harness index`

Per-workspace semantic retrieval index. When `repo_index.enabled=true`
the planner queries it for the top-K chunks relevant to the user prompt
and injects them as a system context block — complements the AST-based
impact analysis with semantic retrieval. Two backends:
zero-dependency `tfidf` (default) and opt-in `openai_embeddings`
(requires `OPENAI_API_KEY`).

| Subcommand | Purpose |
|------------|---------|
| `harness index build` | (Re)build the index for the workspace. |
| `harness index status` | Print backend, chunk count, file count, build timestamp. |
| `harness index clear` | Wipe the index for the workspace. |

Each accepts `-r/--workspace`.

### `harness gh`

GitHub integration wrapping the `gh` CLI — no new Python deps.
Authentication is whatever `gh auth status` reports.

| Subcommand | Purpose |
|------------|---------|
| `harness gh issue --repo OWNER/NAME --number N` | Pull an issue into `change_requests/CR-N-<slug>.txt` so the existing CR flow processes it. |
| `harness gh pr-create --title T --body B [--draft]` | Open a PR from the workspace's current branch. |
| `harness gh pr-comment --repo OWNER/NAME --number N --body B` | Post a comment on an existing PR. |

### `harness status`

| Flag | Purpose |
|------|---------|
| `--session-id` | Session to inspect. |
| `--all` | List all checkpointed sessions. |
| `-r`, `--workspace` | Workspace path (for config discovery; defaults to CWD). |

### `harness doctor`

| Flag | Purpose |
|------|---------|
| `-r`, `--workspace` | Workspace path to check (defaults to CWD). |
| `-v`, `--verbose` | Debug-level logging. |

Runs five checks (git repo / API keys / sandbox backend / checkpoint DB /
config parse), prints a colored summary, and exits 0 if all pass.

### `harness purge`

| Flag | Purpose |
|------|---------|
| `--all` | Delete all checkpoint data permanently (prompts to confirm). |
| `--session-id` | Purge a specific session only. |
| `-r`, `--workspace` | Workspace path (for config discovery). |

### `harness metrics`

| Flag | Purpose |
|------|---------|
| `--session-id` | Aggregate one session into a human report (stdout). |
| `--all` | Roll-up table across every session in the log dir (stdout). |
| `--json` | Emit JSON to `<metrics_dir>/<id>.json` (or `sessions.json` with `--all`). |
| `--prometheus` | Emit Prometheus exposition to `<metrics_dir>/<id>.prom` (or `all.prom`). |
| `--output <path>` | Override destination. Use `-` to write to stdout. |
| `--window-minutes <n>` | Burn-rate trailing window (default 10; `metrics.burn_rate_window_minutes`). |
| `-r`, `--workspace` | Workspace path (for config discovery). |

Default `metrics_dir` is `~/.harness/metrics/`; override globally via
`metrics.metrics_dir` in `~/.harness/config.json` (point it at e.g. a
node_exporter textfile collector directory). Writes are atomic
(`<dest>.tmp` → `os.replace`) so scrapers never see a half-written
file.

## Configuration

Configuration is layered, with later layers overriding earlier ones:

1. `harness/cli.json` — shipped defaults.
2. `~/.harness/config.json` — user-global config (API keys live here too).
3. `<workspace>/.harness_config.json` — per-project overrides; auto-generated
   on first run if missing.

Top-level sections (full list, in roughly the order they appear in
`config/config.json.example`):

| Section | What it governs |
|---------|-----------------|
| `build_command`, `allow_network`, `product_spec_dir`, `change_requests_dir` | Workspace + CLI defaults. |
| `sandbox` | Build-sandbox isolation (`docker` / `unshare` / `bare`), cache mounts, timeouts. |
| `token_budget`, `node_throttle`, `llm_dispatch` | Spend caps, per-stage iteration limits, per-call max_tokens. `llm_dispatch.prompt_cache_enabled` toggles Anthropic `cache_control` markers + the prefix-stability drift telemetry. |
| `models`, `model_routing` | Provider registry + per-role routing (planning / patching / repair / doc reviewer / code reviewer / Ollama fallback). |
| `persistence`, `logging`, `debug`, `metrics` | Checkpoint DB, log dir, LLM-call dumps, Prometheus textfile output. |
| `lintgate`, `compiler`, `patcher`, `speculative`, `test_generation` | Per-node behaviour knobs. The rebuilt `speculative` section exposes `trigger`, `diversity_mode`, `cost_strategy`, `selection_strategy`, `salvage_strategy`, and `voting` axes. |
| `web_tools` | `WebFetchSkill` + `WebSearchSkill` exposure to the planner. Default off. |
| `mcp` | Model Context Protocol client pool (stdio servers, command allowlist). Default off. |
| `skills` | Runtime-extensible user skills directory (default `~/.harness/skills/`). |
| `memory` | Per-repo session memory (planner reads prior-session notes; cmd_run appends on exit). Default on. |
| `repo_index` | Semantic retrieval index. Default off; built via `harness index build`. |
| `github` | Optional `gh_path` override for the `harness gh` subcommands. |
| `deployment` | Dockerfile / docker-compose / Caddy artefact generation. |

Unknown top-level *and* nested keys are warned about at load time with
fuzzy-match suggestions — see [`docs/SPEC_REQUIREMENTS.md`](docs/SPEC_REQUIREMENTS.md)
and `config/config.json.example` for the full per-key schema.

## Troubleshooting

Run `harness doctor` first — it tells you which subsystem is unhappy.

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `[FAIL] git repo` | Workspace isn't a git repo | `git init` in the workspace. |
| `[FAIL] api keys` (e.g. `OPENAI_API_KEY missing`) | Provider key not in env | `export OPENAI_API_KEY=...` (or set the matching `{PROVIDER}_API_KEY`). |
| `[FAIL] sandbox backend` (`docker info failed`) | Docker daemon not running | Start Docker, or switch `sandbox.backend` to `"unshare"` (Linux). |
| `[FAIL] sandbox backend` (`unshare ... failed`) | User namespaces disabled | Run on a host with user-namespace support, or fall back to Docker. |
| `[WARN] config parse` (`Unknown config key ...`) | Typo in `.harness_config.json` | Apply the suggested correction shown in the warning. |
| `[FAIL] checkpoint db` (`sqlite3 open failed`) | DB path not writable | Change `persistence.db_path` to a writable location. |

For failures during a session — checkpoint corruption, budget exhaustion,
sandbox dead mid-run, workspace lock stuck, persistent LLM silence — see
[`docs/RUNBOOK.md`](docs/RUNBOOK.md) for diagnostic commands and recovery
recipes per failure mode.

Logs are written to `~/.harness/logs/<session-id>.jsonl` (rotated at
10 MB by default; configurable via `logging.max_bytes` and
`logging.backup_count`) and stderr.

## Platform support

| Platform | `docker` backend | `unshare` backend | `bare` backend | CI coverage |
|----------|------------------|-------------------|----------------|-------------|
| Linux (Ubuntu 22.04+) | ✓ supported | ✓ supported | ✓ opt-in via `HARNESS_ALLOW_UNSAFE_SANDBOX=true` | ✓ Python 3.11 / 3.12 / 3.13 |
| macOS (Intel + Apple Silicon) | ✓ likely (Docker Desktop), untested | ✗ not available (`unshare` is Linux-only) | ✓ opt-in, untested | ✗ not in CI |
| Windows + WSL2 | ✓ likely (Docker Desktop), untested | ? depends on WSL2 kernel config | ✗ path handling untested | ✗ not in CI |
| Windows (native) | ✓ best-effort (Docker Desktop, Linux containers) — see [docs/installation.md](docs/installation.md) | ✗ not available | ✗ not recommended | ✗ not in CI |

Linux is the only platform covered by the CI matrix and the only one the
project actively tests. macOS and WSL2 are best-effort — the Docker
backend is portable and likely works, but nothing is guaranteed until
the matrix grows. File an issue if you hit a regression on either; we'll
take patches that don't compromise the Linux path.

The `bare` backend (zero isolation) is opt-in everywhere and runs
LLM-generated build commands directly on the host. Never enable it
outside a disposable VM.

## Architecture

See [`docs/SPEC_ARCHITECTURE.md`](docs/SPEC_ARCHITECTURE.md) for the module
map and graph topology. See [`docs/SPEC_REQUIREMENTS.md`](docs/SPEC_REQUIREMENTS.md)
for the functional/non-functional requirements the harness is built against.
For the flow when running `harness run` against a repository that already has
code (bug fix / feature add, skipping discovery), see
[`docs/existing-project-flow.md`](docs/existing-project-flow.md).

## Development

```bash
pip install -e ".[dev]"
make hooks-install
make test
```

### Coverage

`make coverage` runs the pack with `pytest-cov`, prints a terminal
summary, and writes an HTML report under `htmlcov/` plus an XML
report at `coverage.xml` (handy for CI integrations).

```bash
make coverage           # terminal + htmlcov/ + coverage.xml
make coverage SHOW=1    # also open the HTML report in a browser
```

The report covers the live test count and the percentage line-coverage
of the `harness/` package. There is **no CI gate** on the coverage
number today — the metric is for visibility; we'll set a defensible
floor only after the rapid feature-shipping phase settles.

The pre-commit hook runs the full pytest pack and blocks any commit that
breaks the framework. GitHub Actions runs the same suite on every push and
PR across Python 3.11 / 3.12 / 3.13. To bypass the local hook intentionally
(emergencies only), use `git commit --no-verify` — CI still enforces it.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the test layout,
commit-message convention, pre-commit gate behavior, SemVer policy, and
the scope rules the project enforces on PRs.
