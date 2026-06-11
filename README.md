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

Once a `harness run` finishes a green build it has already produced your
`Dockerfile`, `docker-compose.yml`, and (when needed) `Caddyfile`, then
brought the dev env up with health checks. See
[docs/app-deployment.md](docs/app-deployment.md) for the artefact contract,
the preview gate, and how to bring the same setup up on a different host.

## Command reference

| Command | Purpose |
|---------|---------|
| `harness run` | Execute the full agent graph on a workspace. |
| `harness resume` | Resume a crashed or interrupted session from its checkpoint. |
| `harness status` | Read-only inspection of a checkpointed session. |
| `harness doctor` | Run first-run healthchecks (git, API keys, sandbox, DB, config). |
| `harness purge` | Wipe checkpoint data. |
| `harness metrics` | Per-session cost / burn-rate / Prometheus aggregation from logs. |
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

Top-level sections: `build_command`, `allow_network`, `sandbox`,
`token_budget`, `node_throttle`, `models`, `model_routing`, `persistence`,
`logging`, `lintgate`, `deployment`, `metrics`. Unknown top-level *and*
nested keys are warned about at load time with fuzzy-match suggestions —
see [`docs/SPEC_REQUIREMENTS.md`](docs/SPEC_REQUIREMENTS.md) for the full
schema.

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

## Development

```bash
pip install -e ".[dev]"
make hooks-install
make test
```

The pre-commit hook runs the full pytest pack and blocks any commit that
breaks the framework. GitHub Actions runs the same suite on every push and
PR across Python 3.11 / 3.12 / 3.13. To bypass the local hook intentionally
(emergencies only), use `git commit --no-verify` — CI still enforces it.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the test layout,
commit-message convention, pre-commit gate behavior, SemVer policy, and
the scope rules the project enforces on PRs.
