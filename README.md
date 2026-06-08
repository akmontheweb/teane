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

On first run, `myharness` auto-generates `.harness_config.json` in the
workspace from your global config plus shipped defaults. Edit it to customize
per-project settings (model routing, build command, sandbox limits).

## Command reference

| Command | Purpose |
|---------|---------|
| `harness run` | Execute the full agent graph on a workspace. |
| `harness resume` | Resume a crashed or interrupted session from its checkpoint. |
| `harness status` | Read-only inspection of a checkpointed session. |
| `harness doctor` | Run first-run healthchecks (git, API keys, sandbox, DB, config). |
| `harness purge` | Wipe checkpoint data. |

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

## Configuration

Configuration is layered, with later layers overriding earlier ones:

1. `harness/cli.json` — shipped defaults.
2. `~/.harness/config.json` — user-global config (API keys live here too).
3. `<workspace>/.harness_config.json` — per-project overrides; auto-generated
   on first run if missing.

Top-level sections: `build_command`, `allow_network`, `sandbox`,
`token_budget`, `node_throttle`, `models`, `model_routing`, `persistence`,
`logging`, `lintgate`, `deployment`. Unknown top-level *and* nested keys are
warned about at load time with fuzzy-match suggestions — see
[`docs/SPEC_REQUIREMENTS.md`](docs/SPEC_REQUIREMENTS.md) for the full schema.

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

Logs are written to `~/.harness/logs/<session-id>.log` and stderr.

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

A `CONTRIBUTING.md` covering the test layout, commit-message convention, and
the "don't add features beyond the task" rule is coming. For now, read the
`docs/SPEC_*` files before opening a PR.
