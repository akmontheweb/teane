# Installation Guide

Step-by-step deployment of **myharness** on a fresh machine.

Supported platforms: **Linux**, **macOS**, and **Windows** (WSL2 recommended; native works with Docker Desktop).

## 0. Quick install (scripted)

For most operators, the fastest path is the bootstrap script:

```bash
git clone <repo-url> myharness && cd myharness
python3 scripts/setup.py          # or `make setup`
```

It walks 11 phases interactively: platform / Python 3.11+ / git / sandbox-backend probes → venv creation → `pip install -e .` → LLM-provider wizard (writes `~/.harness/config.json` and persists the API key to your shell rc file with your consent) → `harness doctor` verification → optional install commands for security scanners and formatters. Re-runs are idempotent.

Flags worth knowing: `--venv <path>` overrides the default `~/.venvs/harness`, `--dev` adds the `[dev]` extras, `--provider <anthropic|openai|deepseek|ollama>` skips the wizard prompt, `--non-interactive` is for CI, `--no-doctor` skips the final verification. Run `python3 scripts/setup.py --help` for the full list.

Sections §1–§14 below remain the canonical manual reference and the source of truth for what the script does behind the scenes. Read them when you need to debug a failing phase or tune something the wizard skips (multi-provider routing, sandbox image overrides, headless deployment).

## 1. Overview & Scope

This guide is for an operator standing the harness up on a new workstation or server. It walks through prerequisites, sandbox choice, installation, API keys, optional tools, configuration, verification, and a first smoke test, in that order.

Out of scope:

- **Contributor onboarding** (pre-commit hooks, the `make test` loop, PR workflow) — see [CONTRIBUTING.md](../CONTRIBUTING.md).
- **Configuration field reference** (every key in `.harness_config.json`) — see [docs/SPEC_REQUIREMENTS.md](SPEC_REQUIREMENTS.md).
- **Architecture deep-dive** (graph topology, module map, sandbox internals) — see [docs/SPEC_ARCHITECTURE.md](SPEC_ARCHITECTURE.md).
- **Day-to-day usage** (command flags, workflows) — see the [README](../README.md).

## 2. Pick Your Platform Track

Each step below has a snippet for each track. Read the one that matches your machine; ignore the rest.

| Track | When to use | Sandbox backend you'll end up on |
|-------|-------------|----------------------------------|
| **Linux** (Ubuntu 22.04+, Debian, Fedora) | Servers, primary supported platform | Docker or `unshare` |
| **macOS** (Intel or Apple Silicon) | Developer workstation | Docker Desktop |
| **Windows + WSL2** (recommended Windows path) | Developer workstation on Windows | Docker Desktop (with WSL integration) or `unshare` inside the distro |
| **Windows native** (best-effort) | When WSL2 isn't an option | Docker Desktop (Linux containers) |

On Windows, **WSL2 is recommended** because the harness was developed Linux-first and every code path is exercised on Linux in CI. **Windows native is best-effort**: it works for the common flows, but `unshare` does not exist on native Windows, so Docker Desktop is required for sandbox isolation.

### What's new since v1.0

Three additions have shipped that change operator-facing setup or behaviour:

- **Deterministic autofix** — compiler-suggested fixes (rustc / gcc / clang fixits), missing-import insertion, and a small set of known-safe security autofixes (e.g. Bandit `B201` `debug=True → False`, Trivy version bumps with `FixedVersion`) now land **without** an LLM call. Nothing to install or configure; surfaces in logs as `[autofix]` lines.
- **Env-misconfig short-circuit** — when the sandbox build fails because a runtime is missing (`pytest` not installed in `python:3.12-slim`, `npm: command not found`), the router now exits to HITL on the **first** compile with a focused message instead of burning 3 LLM repair iterations. See §13 → Troubleshooting → HITL triggers.
- **Auto test generation** — after every patching round, a new node writes stack-canonical unit tests for the modified source files and runs them deterministically in the sandbox before lintgate. Requires a configured LLM API key. See **§8.5 Test generation** below for the config, the per-stack runner commands, and the sandbox-image caveat.

If you've deployed the harness before, the deltas you need to re-read are §6 (LLM keys are now mandatory for a green run), the new §8.5, and the three new rows in §13.

## 3. Prerequisites

### Linux (Ubuntu 22.04+)

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev \
                    git sqlite3 build-essential
```

Other distros: install the equivalent packages. Fedora/RHEL: `python3.11`, `python3.11-devel`, `git`, `sqlite`, `gcc`, `make`.

### macOS

```bash
brew install python@3.11 git sqlite
xcode-select --install   # one-time, for compiler fallbacks
```

The stock `/usr/bin/python3` is 3.9 on most macOS versions — always invoke `python3.11` explicitly.

### Windows + WSL2 (recommended Windows path)

```powershell
wsl --install -d Ubuntu-22.04
```

Reboot if prompted, set up the Ubuntu user, then **inside the WSL distro** follow the Linux Ubuntu snippet above.

### Windows native (best-effort)

1. **Python 3.11** — install from [python.org](https://www.python.org/downloads/windows/). On the first installer screen, tick **Add python.exe to PATH** and **Install launcher for all users**. Verify in a new shell: `py -3.11 --version`.
2. **Git for Windows** — install from [git-scm.com](https://git-scm.com/download/win). Bundles `git`, `git bash`, and an OpenSSH client.
3. **`sqlite3`** — ships inside Python's stdlib; no separate install.
4. **Build toolchain** — install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and pick the **Desktop development with C++** workload. Needed only as a fallback if a `tree-sitter` wheel doesn't exist for your Python version.
5. **Enable long paths** — Windows defaults to a 260-character `MAX_PATH`. Apply **both**:

   ```powershell
   git config --global core.longpaths true
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
       -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
   ```

   The registry edit needs an Administrator PowerShell and a reboot to take effect.

### Common

- ~500 MB free disk for the venv, tree-sitter grammars, and the checkpoint DB.
- Outbound HTTPS to your LLM provider (`api.anthropic.com`, `api.openai.com`, `api.deepseek.com`).

## 4. Choose a Sandbox Backend

| Backend | Linux | macOS | Win + WSL2 | Win native | Isolation |
|---------|-------|-------|------------|------------|-----------|
| `docker` (recommended) | ✓ | ✓ Docker Desktop | ✓ Docker Desktop + WSL integration | ✓ Docker Desktop, Linux containers | Strongest |
| `unshare` | ✓ | ✗ | ✓ inside the distro | ✗ not on Windows | Linux namespaces |
| `bare` (unsafe) | opt-in | opt-in | opt-in | opt-in | None — runs LLM-generated commands directly on the host |

### Docker

- **Linux**:

  ```bash
  sudo apt install -y docker.io
  sudo usermod -aG docker $USER
  newgrp docker   # or log out and back in
  docker info     # should succeed without sudo
  ```

- **macOS**: install [Docker Desktop](https://www.docker.com/products/docker-desktop/). Start it once and let it finish initializing.
- **Windows + WSL2**: install Docker Desktop on the Windows host, then in **Settings → Resources → WSL Integration** enable integration for your Ubuntu-22.04 distro.
- **Windows native**: install Docker Desktop. Confirm it's running in **Linux containers** mode (right-click the tray icon → "Switch to Linux containers" if needed). The harness's default sandbox image is `ubuntu:22.04`, which requires Linux containers.

### unshare

Pre-installed on every Ubuntu/Debian via `util-linux`. Smoke-test it:

```bash
unshare --user echo ok
```

If you see `ok`, you're good. If it fails, user namespaces are disabled at the kernel level (RHEL/Fedora may need `sysctl -w user.max_user_namespaces=15000` and a SELinux/AppArmor exception).

### bare

Opt-in only, by setting `HARNESS_ALLOW_UNSAFE_SANDBOX=true`. Never enable outside a disposable VM — it runs LLM-generated build commands directly on the host with zero isolation.

## 5. Clone & Install the Package

The package is currently distributed from source. A future `pip install ai-agent-harness` will work once published to PyPI; for now use source.

### Linux / macOS / WSL2

```bash
git clone <repo-url> myharness
cd myharness
python3.11 -m venv ~/.venvs/harness
source ~/.venvs/harness/bin/activate
pip install .
harness --version
```

For a **pilot install where you need bit-exact reproducibility**, use the
shipped constraints file:

```bash
pip install -e . --constraint requirements-prod.txt
```

This pins every direct and transitive dependency to a known-good version.
Regenerate after a deliberate dependency bump with
`pip freeze | grep -v '^-e ' | sort > requirements-prod.txt`.

### Windows native (PowerShell)

```powershell
git clone <repo-url> myharness
cd myharness
py -3.11 -m venv $HOME\.venvs\harness
& $HOME\.venvs\harness\Scripts\Activate.ps1
pip install .
harness --version
```

If `Activate.ps1` is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (one-time).

### Windows native (cmd)

```cmd
git clone <repo-url> myharness
cd myharness
py -3.11 -m venv %USERPROFILE%\.venvs\harness
%USERPROFILE%\.venvs\harness\Scripts\activate.bat
pip install .
harness --version
```

On Windows native the console script lands at `…\.venvs\harness\Scripts\harness.exe`. If `harness --version` fails after activation, your venv didn't activate — re-check the activation command.

For an editable install (recompile-free local edits), substitute `pip install -e .`.

## 6. Provision API Keys (Required)

> As of the auto test-generation feature (see [§8.5](#85-test-generation-new)), a configured LLM API key is **required** to reach a green build. The test-generation node refuses to run without one and routes the session to HITL with a focused `env_misconfig:llm_api_key` message.

Set at least one of these, depending on which providers appear in your `model_routing` (§8):

| Provider | Env var |
|----------|---------|
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Ollama (local) | none — no key needed |

Optional: `LANGCHAIN_API_KEY` + `LANGCHAIN_TRACING_V2=true` enable LangSmith tracing.

### Linux / macOS / WSL2

Append to `~/.bashrc` (or `~/.zshrc`):

```bash
export ANTHROPIC_API_KEY="sk-..."
```

Then `source ~/.bashrc` or open a new shell.

### Windows native (PowerShell)

```powershell
[System.Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY','sk-...','User')
```

Then close and reopen the shell — `SetEnvironmentVariable` does not affect the current process.

### Windows native (cmd)

```cmd
setx ANTHROPIC_API_KEY "sk-..."
```

`setx` makes the variable permanent but does **not** update the current shell. Open a new `cmd` window.

> **Do not** embed API keys in `.harness_config.json`. Use env vars so the config file stays committable.

## 7. (Optional) Install External Tools

Skip this entire section if you only want to run the harness against your own toolchain. Missing tools degrade gracefully — security scanners are skipped, formatters fall back to text-only patching.

### Security scanners

| Tool | Linux / WSL2 | macOS | Windows native |
|------|--------------|-------|----------------|
| `gitleaks` | `sudo apt install gitleaks` or [release binary](https://github.com/gitleaks/gitleaks/releases) | `brew install gitleaks` | `winget install gitleaks` or `scoop install gitleaks` |
| `bandit` | `pip install bandit` | `pip install bandit` | `pip install bandit` |
| `semgrep` | `pip install semgrep` | `brew install semgrep` | `pip install semgrep` (less battle-tested on native Windows — WSL2 is smoother) |
| `trivy` | [official install script](https://aquasecurity.github.io/trivy/latest/getting-started/installation/) | `brew install trivy` | `winget install AquaSecurity.Trivy` or `scoop install trivy` |

If `gitleaks` is missing, the harness falls back to a regex-based Python secret scanner.

### Formatters / linters (install only for languages you target)

| Tool | Install |
|------|---------|
| `ruff` (Python) | `pip install ruff` — works on every platform |
| `prettier` (JS / TS / JSON / MD) | `npm install -g prettier` |
| `gofmt` (Go) | bundled with Go (`brew install go`, apt, or [go.dev](https://go.dev/dl/)) |
| `rustfmt` (Rust) | `rustup component add rustfmt` |
| `clang-format` (C / C++) | `apt install clang-format`, `brew install clang-format`, or [LLVM Windows installer](https://releases.llvm.org/) |

### GitHub CLI (optional — required for `harness gh` subcommands)

The `harness gh issue` / `pr-create` / `pr-comment` subcommands shell out to the [`gh` CLI](https://cli.github.com/). Install once and authenticate; the harness never stores tokens.

| Platform | Install |
|----------|---------|
| Linux (Debian/Ubuntu) | `sudo apt install gh` (or the [official installer](https://github.com/cli/cli/blob/trunk/docs/install_linux.md)) |
| macOS | `brew install gh` |
| Windows native | `winget install GitHub.cli` |

Then authenticate once:
```bash
gh auth login
```

Skip this section if you only run the harness against your own change-request files; `harness gh` is the only place that needs `gh` on PATH.

### MCP servers (optional — required only if you enable `mcp.enabled=true`)

Most MCP servers ship as npm packages launchable via `npx`; install Node.js once (`brew install node`, `apt install nodejs npm`, or [nodejs.org](https://nodejs.org/)) and the rest are declarative. Example `mcp.servers` entries that work out of the box:

```jsonc
{
  "mcp": {
    "enabled": true,
    "servers": [
      {
        "name": "time",
        "transport": "stdio",
        "command": ["npx", "-y", "@modelcontextprotocol/server-time"]
      },
      {
        "name": "fetch",
        "transport": "stdio",
        "command": ["npx", "-y", "@modelcontextprotocol/server-fetch"]
      }
    ]
  }
}
```

Filesystem MCP servers (`@modelcontextprotocol/server-filesystem`) bypass the harness's build sandbox and require an explicit opt-in:
```jsonc
{
  "mcp": {
    "allow_local_filesystem_servers": true,
    ...
  }
}
```

`harness doctor` adds one row per server when `mcp.enabled=true` showing the tool count or the rejection reason.

## 8. Configure the Harness

Configuration layers, lowest to highest priority:

1. `harness/cli.json` — shipped defaults, do not edit.
2. `~/.harness/config.json` (Linux/macOS/WSL2) or `%USERPROFILE%\.harness\config.json` (Windows native) — your user-global config. Put model definitions and routing here.
3. `<workspace>/.harness_config.json` — per-project overrides, auto-generated on first run.

A minimal user-global config:

```json
{
  "models": {
    "claude-sonnet": {
      "provider": "anthropic",
      "model_id": "claude-sonnet-4-6",
      "context_window": 200000
    }
  },
  "model_routing": {
    "planning_primary": "anthropic:claude-sonnet",
    "patching_primary": "anthropic:claude-sonnet",
    "repair_primary": "anthropic:claude-sonnet"
  }
}
```

Every key in `model_routing` references a model registered under `models` (or one of the catalogue entries shipped in `harness/model_prices.json`). The full schema — every field of `sandbox`, `token_budget`, `node_throttle`, `persistence`, `logging`, `lintgate`, `deployment`, `test_generation`, `metrics` — is documented in [docs/SPEC_REQUIREMENTS.md](SPEC_REQUIREMENTS.md).

**API key resolution.** Each provider key is resolved at runtime in this order: (1) explicit constructor argument, (2) `{PROVIDER}_API_KEY` env var, (3) `models["<provider>:<model>"].api_key` field in any config layer. `harness doctor` checks the same two locations (env first, then config field) and reports the source per model in its `api keys` line, so an `[ OK ]` row reading `... (config)` confirms a config-only key would be picked up. Set the key in `~/.harness/config.json` for global use or per-workspace in `.harness_config.json`.

**Recent config additions worth knowing**:
- `persistence.redact_messages` (default `true`) — checkpoint message redaction. Flip to `false` to keep verbatim transcripts at rest.
- `sandbox.auto_enable_network_for_install` (default `false`) — opt in to auto-enabling network on detected pip / npm install commands. Without this, the heuristic only logs a WARNING.
- `node_throttle.max_discovery_iterations` (default `10`, clamped `[1, 30]`) — hard cap on the discovery question loop.
- `logging.max_bytes` / `logging.backup_count` (default `10_000_000` / `5`) — rotation knobs for the per-session JSONL file. Set `max_bytes: 0` to opt out of rotation.
- `metrics.metrics_dir` (default `~/.harness/metrics`) and `metrics.burn_rate_window_minutes` (default `10`) — output destination and trailing window for `harness metrics`.

**WSL2 only**: put your workspaces and the config under the WSL filesystem (`~/...`), **not** under `/mnt/c/...`. Windows-mount paths have order-of-magnitude slower I/O.

### Workspace single-writer lock

Every `harness run` acquires an `fcntl` lock on `<workspace>/.harness_session.lock` (Linux/macOS/WSL2). A second concurrent run on the same workspace exits with `lock held by PID X`. To recover after a hard kill that left the lock stale:

```bash
harness run -r <workspace> -p "<prompt>" --force-lock
```

Windows native skips locking entirely (no portable `fcntl`); single-writer is the operator's responsibility there. See `docs/RUNBOOK.md` § 4 for the full recovery recipe.

## 8.5 Test generation (new)

After every patching round, the harness writes stack-canonical unit tests for the modified source files and runs them deterministically in the sandbox before lintgate. The node — `harness/test_generation.py` — sits in the graph between `speculative_node` and `lintgate_node`.

**What it does.** Detects the workspace stack via `_detect_workspace_stack`, loads the matching `harness/test_guides/<lang>.md` into the LLM prompt, asks for `CREATE_FILE` / `INSERT_AT_BLOCK` patch blocks for the corresponding test files, applies them, then invokes a stack-canonical test command in the sandbox. **The guides instruct the LLM to write tests that exercise the real code — no mocks.** When a side effect can't be invoked directly, the tests use the test runner's built-in fakes (pytest `monkeypatch` / `tmp_path`, Go `httptest.NewServer`, JUnit `@TempDir`, etc.).

**Config defaults.** Shipped in `harness/cli.json` under `test_generation`. Defaults are `enabled: true` and `max_iterations: 2`. To disable, add to `~/.harness/config.json` or `<workspace>/.harness_config.json`:

```json
"test_generation": { "enabled": false }
```

**LLM API key is required.** Without a configured gateway the node synthesises an `env_misconfig:llm_api_key` diagnostic and routes the session to HITL. See §6 — provisioning at least one provider key is no longer optional when test generation is on.

**Per-stack test runner.** The deterministic command the sandbox runs per detected stack:

| Stack | Deterministic command |
|-------|------------------------|
| Python | `pip install -q pytest && python3 -m pytest -q` |
| Node / JavaScript | `npm install --no-save --silent jest && npx jest --silent` |
| TypeScript | `npm install --no-save --silent jest ts-jest typescript && npx jest --silent` |
| Go | `go test ./...` |
| Java | `mvn -q test` |
| Rust | `cargo test --quiet` |
| Dart | `dart test` |
| Flutter | `flutter test` |

`pip install` / `npm install` tokens trigger the harness's existing `_build_command_needs_network` heuristic, so the sandbox auto-enables outbound network for the test run — no manual `allow_network: true` required.

**Sandbox image caveat (read this).** The deterministic test runner re-uses whichever `sandbox.docker_image` the build_command auto-adapter picked. That adapter keys off `build_command` only — if your `build_command` is, say, `make build` but the workspace is Python, the image stays `ubuntu:22.04` and `pip install pytest` will fail with `pip: command not found`. Two workarounds:

1. Include a stack-implying token in `build_command` so the existing adapter picks the right image, e.g. `make build && python3 --version`.
2. Or set `sandbox.docker_image` explicitly in your `.harness_config.json`:
   ```json
   "sandbox": { "docker_image": "python:3.12-slim" }
   ```
   Per-stack images that ship the toolchain: `python:3.12-slim`, `node:20-slim`, `golang:1.22`, `rust:1.79-slim`, `eclipse-temurin:21-jdk` (Java), `dart:stable` (Dart), `ghcr.io/cirruslabs/flutter` (Flutter).

**Project-level test conventions.** Drop your own files under `<workspace>/test_guides/<lang>.md` with frontmatter `applies_to: [<stack-tag>]`. The loader prefers project files over the shipped defaults, so a workspace can tighten the conventions the LLM is given without forking the harness.

## 9. Verify the Install (`harness doctor`)

`cd` into any git repo and run:

```bash
harness doctor
```

Expected output: six check lines, all green.

| Check | What it verifies | Failure means |
|-------|------------------|---------------|
| `git repo` | Workspace is a git repo with at least one commit | `git init` and make a commit |
| `global config` | `~/.harness/config.json` exists and is valid JSON | Run the setup script or create it manually |
| `api keys` | Every provider referenced in `model_routing` has a key in EITHER its `*_API_KEY` env var OR its `models["<key>"].api_key` config field, AND each provider passes a one-token "hello" request that confirms the key actually authenticates | Set the missing key, fix an `HTTP 401 — API key rejected`, or set `HARNESS_DOCTOR_SKIP_LIVE=true` to disable the live ping (CI / headless) |
| `sandbox backend` | Docker or `unshare` is reachable | Re-do §4 for your platform |
| `checkpoint db` | `~/.harness/checkpoints.db` is writable AND the 5 most recent checkpoints deserialize cleanly | Adjust `persistence.db_path`, or `harness purge --session-id <id>` for a corrupted session |
| `config parse` | The merged config is valid JSON with known keys | Fix the typo it suggests |

The `api keys` PASS message includes the source per model — `... (env)` or `... (config)` — so you can confirm at a glance whether the runtime will resolve from your env vars or your config file. The check also makes a 1-token chat call per provider (in parallel) to confirm the key actually authenticates against the configured model. Cost is well under a tenth of a cent across all providers combined. To skip the live ping (e.g. in CI where outbound network is blocked), set `HARNESS_DOCTOR_SKIP_LIVE=true`; the doctor then reports the source per model and notes `(live ping skipped via HARNESS_DOCTOR_SKIP_LIVE)`.

Common failure modes the live ping surfaces:

| Detail line | Likely cause |
|---|---|
| `HTTP 401 — API key rejected` | Key is invalid, revoked, or has a typo |
| `HTTP 403 — key is valid but has no access to model '<id>'` | Key works for other models but not this one; pick another model or upgrade the account tier |
| `HTTP 404 — model '<id>' not found at provider` | Model id misspelt in `model_routing` |
| `HTTP 429 — rate limited` | Key works but quota is exhausted right now; try again later or use a different account |
| `timeout — provider unreachable or network blocked` | Local firewall / proxy issue, or provider is down |
| `connection failed (...)` | DNS or connectivity issue at this host |

Exit code `0` means you are ready to run.

**Windows native**: with `sandbox.backend = "auto"` (the default), the doctor probes Docker first; if Docker is up, the check passes and the missing `unshare` is ignored. If you hard-pinned `sandbox.backend = "unshare"`, the check will fail — change it to `"docker"` or `"auto"`.

## 10. First Run (Smoke Test)

Pick a tiny throwaway git repo:

```bash
git clone https://github.com/octocat/Hello-World.git sample
harness run -r ./sample -p "list the top-level files"
```

The harness will:

1. Auto-generate `./sample/.harness_config.json`.
2. Acquire an `fcntl` lock on `./sample/.harness_session.lock` (Linux/macOS/WSL2).
3. Create `~/.harness/checkpoints.db` (or `%USERPROFILE%\.harness\checkpoints.db` on Windows native) and `~/.harness/logs/<session-id>.jsonl` (rotated at 10 MB × 5 backups by default).
4. Run planning → patching → compile loop, checkpointing each step.

A successful smoke test exits 0. Inspect the run with `harness status --session-id <id>` or by tailing the JSONL log file. After the run, `harness metrics --session-id <id>` shows cost, burn rate, and projected exhaustion against your `token_budget.hard_cap_usd`.

See the [README command reference](../README.md#command-reference) for the full flag list. For diagnostics on a stuck or failed run, start with [`docs/RUNBOOK.md`](RUNBOOK.md).

## 11. Headless / Server Deployment

For unattended runs (CI, scheduled jobs, services):

- Set `CI=true` and `HARNESS_AUTO_APPROVE=true` to bypass interactive HITL prompts.
- Set `HARNESS_HITL_WEBHOOK_URL` (and `HARNESS_HITL_WEBHOOK_SECRET`) if you want sensitive operations to require approval via a webhook instead of stdin.
- Ensure NTP is running — clock skew breaks API auth on fresh VMs.
- With test_generation on (the default — see [§8.5](#85-test-generation-new)), each session spends additional LLM tokens for test authorship and may hit the default `token_budget.hard_cap_usd = 2.00` on larger changes. Raise the cap in your config, or set `test_generation.enabled: false` under CI when budget pressure matters.
- Track aggregate cost with a cron job: `harness metrics --all --prometheus --output /var/lib/node_exporter/textfile/harness.prom`. The atomic-write contract means a scraper never sees a half-written file.
- For long-running services, leave `logging.max_bytes` / `logging.backup_count` at defaults (10 MB × 5) — that's ~50 MB max per session, dropped oldest-first.
- If a scheduled job dies hard and leaves the workspace lock stale, the next run will refuse to start. Wrap with a retry that adds `--force-lock` on second attempt only.

### Linux (systemd)

One-shot run (single session, equivalent to the operator typing `harness run`):
```ini
[Service]
Environment=HOME=/srv/harness
Environment=CI=true
Environment=HARNESS_AUTO_APPROVE=true
Environment=ANTHROPIC_API_KEY=sk-...
WorkingDirectory=/srv/harness/workspace
ExecStart=/srv/harness/.venvs/harness/bin/harness run -r . -p "..."
```

Scheduled-job daemon (`harness schedule run` — runs `schedule.jobs` from
`config.json` on a cron-style timer; restart-on-failure recommended):
```ini
# /etc/systemd/system/harness-schedule.service
[Unit]
Description=myharness scheduled-job daemon
After=network-online.target

[Service]
Type=simple
User=harness
Environment=HOME=/srv/harness
Environment=ANTHROPIC_API_KEY=sk-...
WorkingDirectory=/srv/harness/workspace
ExecStart=/srv/harness/.venvs/harness/bin/harness schedule run
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Web dashboard (`harness web` — all features on by default; set
`dashboard.writes_enabled: false` in `config.json` for a read-only
deployment):
```ini
# /etc/systemd/system/harness-dashboard.service
[Unit]
Description=myharness web dashboard
After=network-online.target

[Service]
Type=simple
User=harness
Environment=HOME=/srv/harness
# Bearer token gate (required when binding off-localhost):
Environment=DASH_TOKEN=replace-with-a-long-random-string
# Optional: persistent CSRF token across restarts:
Environment=DASH_CSRF=replace-with-another-long-random-string
WorkingDirectory=/srv/harness/workspace
ExecStart=/srv/harness/.venvs/harness/bin/harness web
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then in `config.json`:
```jsonc
{
  "dashboard": {
    "enabled": true,
    "host": "127.0.0.1",         // localhost only by default
    "port": 8729,
    "token_env": "DASH_TOKEN",
    "csrf_token_env": "DASH_CSRF"
    // writes_enabled defaults to true; set to false for a read-only deployment
  }
}
```

`HOME` must be set explicitly on every unit so `~/.harness/` path expansion works.

Enable + start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now harness-schedule.service
sudo systemctl enable --now harness-dashboard.service
sudo journalctl -fu harness-schedule.service   # follow logs
```

Run separate units rather than bundling — the dashboard config changes mean only the dashboard restarts, and a stuck dashboard request never affects schedule timing.

### Windows native (Task Scheduler)

- Create a task running `…\.venvs\harness\Scripts\harness.exe` with arguments `run -r C:\path\to\workspace -p "..."`.
- Set **Start in** to the workspace directory.
- Add `CI`, `HARNESS_AUTO_APPROVE`, and `ANTHROPIC_API_KEY` as system or user env vars (Task Scheduler inherits the account's env).
- For a long-lived service, wrap the command with [NSSM](https://nssm.cc/).

## 12. Upgrading & Uninstalling

### Upgrade

```bash
cd myharness
git pull
pip install -U .
harness doctor
```

(Or `pip install -U ai-agent-harness` once published to PyPI.)

### Uninstall

```bash
pip uninstall ai-agent-harness
rm -rf ~/.harness     # Linux/macOS/WSL2
```

`~/.harness/` holds the checkpoint DB, per-session JSONL logs, metrics, repo memory files, the repo index (`repo_index/`), schedule history (`schedule.db`), web app state (`web.db`), and the user-skills directory (`skills/`). Remove only what you don't want to keep.

On Windows native: `pip uninstall ai-agent-harness` then `Remove-Item -Recurse $HOME\.harness` in PowerShell.

## 13. Troubleshooting

The canonical runtime-failure table lives in the [README → Troubleshooting](../README.md#troubleshooting) section. For the top-five mid-session failure modes (checkpoint corrupted, budget exhausted, sandbox dead, workspace lock refused, persistent LLM silence), see [`docs/RUNBOOK.md`](RUNBOOK.md). This list adds install-specific gotchas neither of those covers.

### All platforms

| Symptom | Fix |
|---------|-----|
| `python3.11: command not found` | Re-do §3 for your platform; on Windows, use `py -3.11` |
| `error: Microsoft Visual C++ 14.0 or greater is required` (Windows native, pip install) | Install MSVC Build Tools (§3) |
| `error: failed building wheel for tree-sitter-…` | Install the build toolchain (`build-essential` / Xcode CLT / MSVC), then `pip install --no-binary :all: tree-sitter` to force a source build |
| pip stalls behind a corporate proxy | `export HTTPS_PROXY=http://proxy:port` and `export NO_PROXY=localhost,127.0.0.1` before pip / harness commands |
| API auth fails with "invalid timestamp" | NTP isn't running; sync the clock |

### Runtime HITL triggers (new)

These show up in the HITL banner as `Trigger: <name>` and mean the harness short-circuited the LLM repair loop because no amount of LLM patching can fix the root cause.

| Trigger | Meaning | Fix |
|---------|---------|-----|
| `env_misconfig:<symbol>` (e.g. `env_misconfig:pytest`) | Sandbox build exited with "No module named X" / "command not found" / Docker `exec: "X": executable file not found`. The runtime is missing inside the container. | Either prepend an install step to `build_command` (e.g. `pip install <symbol> && <original-cmd>`) or set `sandbox.docker_image` to one that ships `<symbol>`. |
| `env_misconfig:llm_api_key` | test_generation cannot run because no LLM gateway is configured. | Set the matching `*_API_KEY` env var (see §6), or set `test_generation.enabled: false` in `.harness_config.json`. |
| `llm_silent` (HITL fires immediately) | Three consecutive empty responses from the LLM provider (`EmptyLLMResponseError`). | Check the provider's status page; retry, or route the affected node to a different model via `model_routing`. |
| `Auto-test run fails with "pip: command not found" / "npx: not found"` | The deterministic test runner from §8.5 is executing in a docker image that doesn't carry the stack toolchain. | See the [§8.5 sandbox-image caveat](#85-test-generation-new) — fix by adjusting `build_command` or pinning `sandbox.docker_image` explicitly. |
| `[FAIL] api keys` despite a key being set | Doctor used to only check env vars. The current build checks env AND `models["<key>"].api_key` config field — if doctor still reports FAIL, neither location has the key. The FAIL message names both. | Set the key in either `{PROVIDER}_API_KEY` env var or `models."<key>".api_key` in `~/.harness/config.json`. |
| `lock held by PID <n>` at startup | A prior `harness run` exited hard and left `<workspace>/.harness_session.lock` stale, or another live session is using the workspace. | Verify the PID is gone (`ps -p <n>`); then `harness run ... --force-lock` to release and reacquire. See `docs/RUNBOOK.md` § 4. |

### Linux

| Symptom | Fix |
|---------|-----|
| `docker: permission denied while trying to connect to the Docker daemon socket` | `sudo usermod -aG docker $USER` then log out and back in |
| `sqlite3.OperationalError: unable to open database file` under systemd / cron | `HOME` isn't set in the service environment; add `Environment=HOME=/srv/harness` |
| `unshare: write failed: /proc/self/uid_map: Operation not permitted` on RHEL/Fedora | User namespaces disabled; `sysctl -w user.max_user_namespaces=15000` and check SELinux/AppArmor policy |

### Windows native

| Symptom | Fix |
|---------|-----|
| `Activate.ps1 cannot be loaded because running scripts is disabled on this system` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `[Errno 2] No such file or directory: 'C:\\Users\\…\\some\\deeply\\nested\\…'` | Long paths not enabled; re-do the `LongPathsEnabled` step in §3 and reboot |
| `docker: Error response from daemon: …Linux containers are not enabled` | Switch Docker Desktop to Linux containers mode (tray icon → "Switch to Linux containers") |
| Outbound to `api.anthropic.com` blocked | Add a Windows Firewall outbound rule for the venv's `python.exe`, or whitelist the provider domains at the corporate firewall |
| `.harness_config.json` saved with CRLF line endings | Harmless for JSON parsing; if a downstream tool chokes, run `git config --global core.autocrlf input` and re-save with LF |

### Windows + WSL2

| Symptom | Fix |
|---------|-----|
| Builds run extremely slowly | Workspace is under `/mnt/c/...`; move it to the WSL filesystem (`~/...`) |
| API auth fails after the laptop sleeps | WSL2 clock drift; `sudo hwclock -s` (or install `chrony` and let it run) |
| `harness doctor`'s sandbox check fails despite Docker Desktop running | "WSL Integration" toggle is off for your distro — enable it in Docker Desktop → Settings → Resources → WSL Integration |

## 14. Next Steps

- [README → Command reference](../README.md#command-reference) — every flag of `harness run`, `resume`, `status`, `doctor`, `purge`.
- [docs/SPEC_REQUIREMENTS.md](SPEC_REQUIREMENTS.md) — full `.harness_config.json` schema.
- [docs/SPEC_ARCHITECTURE.md](SPEC_ARCHITECTURE.md) — graph topology and module map.
- [docs/app-deployment.md](app-deployment.md) — what the harness produces for deployment (Dockerfile, docker-compose.yml, Caddyfile), the preview gate, and how to bring the same dev env up on a different host.
- [CONTRIBUTING.md](../CONTRIBUTING.md) — dev environment, pre-commit, test loop.
- `harness/style_guides/*.md` and `harness/test_guides/*.md` — shipped per-language guidance the LLM sees during code and test generation. Drop your own overrides under `<workspace>/style_guides/` or `<workspace>/test_guides/` (same filenames win on collision).
