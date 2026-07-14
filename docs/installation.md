# Installation Guide

Step-by-step deployment of **teane** on a fresh machine.

Supported platforms: **Linux**, **macOS**, and **Windows** (WSL2 recommended; native works with Docker Desktop).

## 0. Quick install (scripted)

For most operators, the fastest path is the bootstrap script:

```bash
git clone <repo-url> teane && cd teane
python3 scripts/setup.py          # or `make setup`
```

It walks 11 phases interactively: platform / Python 3.11+ / git / sandbox-backend probes → venv creation → `pip install -e .` → LLM-provider wizard (writes `<repo>/config/config.json` and persists the API key to your shell rc file with your consent) → `teane doctor` verification → optional install commands for security scanners and formatters. Re-runs are idempotent.

Flags worth knowing: `--venv <path>` overrides the default `~/.venvs/teane`, `--dev` adds the `[dev]` extras, `--provider <anthropic|openai|deepseek|ollama>` skips the wizard prompt, `--non-interactive` is for CI, `--no-doctor` skips the final verification. Run `python3 scripts/setup.py --help` for the full list.

Sections §1–§14 below remain the canonical manual reference and the source of truth for what the script does behind the scenes. Read them when you need to debug a failing phase or tune something the wizard skips (multi-provider routing, sandbox image overrides, headless deployment).

## 1. Overview & Scope

This guide is for an operator standing the harness up on a new workstation or server. It walks through prerequisites, sandbox choice, installation, API keys, optional tools, configuration, verification, and a first smoke test, in that order.

Out of scope:

- **Configuration field reference** (every key in `<repo>/config/config.json`) — see [docs/SPEC_REQUIREMENTS.md](SPEC_REQUIREMENTS.md).
- **Architecture deep-dive** (graph topology, module map, sandbox internals) — see [docs/SPEC_ARCHITECTURE.md](SPEC_ARCHITECTURE.md).
- **Mid-session recovery** (checkpoint corruption, budget exhaustion, lock contention, stuck dashboards) — see [docs/RUNBOOK.md](RUNBOOK.md).

## 2. Pick Your Platform Track

Each step below has a snippet for each track. Read the one that matches your machine; ignore the rest.

| Track | When to use | Sandbox backend you'll end up on |
|-------|-------------|----------------------------------|
| **Linux** (Ubuntu 22.04+, Debian, Fedora) | Servers, primary supported platform | Docker or `unshare` |
| **macOS** (Intel or Apple Silicon) | Developer workstation | Docker Desktop |
| **Windows + WSL2** (recommended Windows path) | Developer workstation on Windows | Docker Desktop (with WSL integration) or `unshare` inside the distro |
| **Windows native** (best-effort) | When WSL2 isn't an option | Docker Desktop (Linux containers) |

On Windows, **WSL2 is recommended** because the harness was developed Linux-first and every code path is exercised on Linux in CI. **Windows native is best-effort**: it works for the common flows, but `unshare` does not exist on native Windows, so Docker Desktop is required for sandbox isolation.

### What's new (current release)

- **Four-target CLI** — the legacy `run` verb is split into `teane build` (greenfield: wipes the workspace except `product_spec/` and `.git/`, resets to a clean base branch; pair with `--yes` for unattended automation), `teane patch` (brownfield: reads change-request files from `change_requests/`; add `--agile true` for story-decomposition mode), `teane deploy` (compose synthesis + dev container + health checks), and `teane test` (Playwright e2e against the deployed stack; failures land as `CR-DEFECT-*` files that the next `teane patch` consumes). The old `--new-build` flag is gone — the target choice IS the mode. Exit codes are deterministic for CI chaining: 0 clean, 1 partial, 2 config error, 3 budget exhausted, 4 infrastructure failure.
- **Static diagnostics gate** — after every patch round, pyright/mypy (Python) and `tsc` (TS/TSX) run read-only BEFORE the compile; new type errors are repaired in-loop. Pre-existing brownfield errors are baseline-suppressed. Install the checkers from §7 to enable; missing tools degrade silently.
- **HITL learning loop** — failed runs and HITL escalations write a one-line `[learned-rule:*]` hypothesis into per-repo memory (`~/.harness/memory/`), injected into the next run's planner. A clean run retires all active rules automatically.
- **Brownfield LSP navigation** — `teane patch`/`test` can spawn pyright-langserver / typescript-language-server for ground-truth find-references (see §7 → LSP servers). Greenfield never starts them.
- **Unattended-run hardening** — headless HITL auto-resume cap with direct-abandon, pre-patch anti-drift screens, and empty-file patch guards keep hands-off runs from burning budget in loops.
- **Per-file patcher rejection feedback (FR-079)** — when the patcher rejects a block, the LLM's next-round system message now names each failed file, its operation, a classification tag (`file missing`, `search miss`, `ambiguous match`, `rejected: file already exists`, `path denied`, `allowlist denied`, `no blocks parsed`), and a directive telling it exactly what to do — e.g. "use CREATE_FILE", "READ_FILE and copy exact bytes", "add more context". Applies to both `patching_node` and `repair_node`. Previously the LLM saw only "Failed to apply N patch(es)." and re-emitted the same broken block.
- **70% line-coverage gate for generated apps (FR-080)** — `harness/skills/makefile_python.md` and `makefile_node.md` now require the LLM's generated `Makefile` (Python) and `package.json` (Jest) to enforce ≥70% line coverage. Enforcement rides on pytest's `--cov-fail-under=70` / Jest's `coverageThreshold.global.lines=70` exit codes: a build with UTs passing but coverage < 70% loops back through `repair_node` to add more tests. Threshold is intentionally hard-coded in the skill text — no config knob in v1. New `unit_tests_python.md` / `unit_tests_react.md` skills teach the LLM what a unit test IS vs ISN'T (`teane test`'s Playwright pack owns e2e / AC verification).
- **Flow-aware traceability gate (FR-081)** — build / patch runs now succeed with untested acceptance criteria; AC coverage is only enforced when `flow == "test"`. Rationale: unit tests generated during build/patch link to code modules, not ACs; the Playwright pack from `teane test` closes AC coverage. Previously build/patch could ping-pong through `traceability_block` when the LLM missed `@verifies:` markers on some tests, with no way to recover headless. Untraced *requirements* (`untraced` list) still block every flow — that's a planner failure no downstream target can fix.
- **System-prompt diet (FR-082)** — two lossless transforms reduce prompt tokens without breaking the prefix cache: (a) planner-only fields (`Business driver`, `Success metrics`, `Priority`, `Estimated size`, `Wave`, `Iteration`) stripped from the RSD once at spec-load time — ~30% smaller system prompt while every code-grounding field (assumptions, scope, ACs) survives; (b) `repair_node` prunes the mid-array history from round 4 onward, keeping only `messages[0:2]` + the last 6 turns so the LLM stops arguing with its own past attempts (finsearch STORY-042 spent 10+ rounds re-emitting variations of a stale round-4 hallucination before this fix).
- **Cross-cutting style rules for the locked stack** — new style-guide sections shipped in `harness/style_guides/python.md` and `typescript.md` covering datetime conventions (`from datetime import datetime, timezone`; avoid the `datetime.UTC` sandbox trap), cross-platform path handling (`pathlib.Path` end-to-end; no string arithmetic; `pytest tmp_path` fixture over hard-coded `/tmp`), package init (`__init__.py` must land before submodules), and concurrency (`threading.Lock` uses `with`, `asyncio.Lock` uses `async with` — never mix). Applies whenever the workspace stack tags include `python` or `typescript`.

### What's new since v1.0 (older milestones)

The biggest operator-facing changes since the layered-config era (bullets below predate the four-target CLI split — read `teane run --new-build true` as `teane build` and `--new-build false` as `teane patch`):

- **Single canonical config** — the harness now reads exactly one file: `<repo>/config/config.json`. There is no `~/.harness/config.json`, no per-workspace `.harness_config.json`, no shipped `harness/cli.json` defaults. Strict validation rejects unknown keys, wrong types, missing required fields, and missing API key env vars at startup — before any LLM call. A legacy `.harness_config.json` left in a workspace logs one INFO line and is otherwise ignored. See §8.
- **Mandatory `product_spec_dir`** — every run must point at a workspace-root folder of spec files (`.txt`, `.md`, `.pdf`) describing the product to build. `.pdf` bodies are extracted via `pypdf`. `teane doctor` and `teane run` both refuse to start when the key is missing, malformed, or the folder is empty.
- **Greenfield-vs-change-request modes** — `teane run --new-build true` wipes the workspace (except `product_spec/` and `.git/`) and resets to a clean base branch; `--new-build false` (the default) reads spec files (`.txt`, `.md`, `.pdf`) from `change_requests/` for steady-state work. Pair with `--yes` for unattended automation.
- **Interactive wizard on bare `teane run`** — invoking `teane run` with no flags drops the operator into a wizard that resolves API keys, workspace, prompt, and mode. The wizard never writes to disk; each bare run re-asks.
- **Deterministic autofix** — compiler-suggested fixes (rustc / gcc / clang fixits), missing-import insertion, and a small set of known-safe security autofixes (e.g. Bandit `B201` `debug=True → False`, Trivy version bumps with `FixedVersion`) now land **without** an LLM call. Surfaces in logs as `[autofix]` lines.
- **Env-misconfig short-circuit** — when the sandbox build fails because a runtime is missing (`pytest` not installed in `python:3.12-slim`, `npm: command not found`), the router now exits to HITL on the **first** compile with a focused message instead of burning 3 LLM repair iterations. See §13 → Troubleshooting → HITL triggers.
- **Auto test generation** — after every patching round, a node writes stack-canonical unit tests for the modified source files and runs them deterministically in the sandbox before lintgate. Requires a configured LLM API key. See **§8.5 Test generation** below.
- **End-of-run `INSTALLATION.md`** — on successful greenfield builds, the harness writes `<workspace>/docs/INSTALLATION.md` describing how to install, configure, run locally, and (when `--deploy-dev true` produced a docker-compose blueprint) deploy the generated app. The doc is grounded in the actual artifacts: workspace telemetry, root manifests (`requirements.txt`, `package.json`, `Makefile`, `.env.example`), the Build & Run section of `SPEC_ARCHITECTURE.md`, and the deployment blueprint when present. Controlled by `--install-doc true|false`; defaults to the value of `--new-build` (on for greenfield, off for change-request runs). Best-effort — a synthesis failure logs `[installation_doc] Synthesis failed; INSTALLATION.md not written: …` and does not roll back the build.
- **Carbon web UI** — `teane web start` / `teane web stop` runs a single-instance dashboard (default bind `127.0.0.1:9000`) for browsing sessions, editing config, viewing memory, scheduling runs, and answering HITL prompts in-browser. See §11.
- **Scheduled-job daemon** — `teane schedule run` fires `teane run` jobs from `config.json` on a cron-style timer. See §11.
- **Repo-index, per-repo memory, MCP client pool** — opt-in subsystems controlled from `config.json`. See the per-section comments in the shipped file for the contract.

## 2.5 Pre-flight Check (first step on any machine)

Before reading §3 in full, after the harness is installed, run:

```bash
teane pre-flight
```

or before install, from inside the cloned repo:

```bash
python -m harness.cli pre-flight
```

This auto-detects your OS (Windows / macOS / Linux) and prints a coloured checklist of every tool the harness needs, with an OS-appropriate install command for each missing item. The boundary against `teane doctor`:

- **`pre-flight`**: is this **machine** ready to run the harness at all? No workspace, no config — tool / runtime / system level checks.
- **`doctor`**: is this **workspace** configured correctly? Needs a `config/config.json` and a workspace path.

`pre-flight` exits `0` when no REQUIRED item failed (warnings on optional tools are fine to defer); it exits `1` only when a REQUIRED tool is missing. Useful flags:

| Flag | Purpose |
|---|---|
| `--quick` | Skip the live outbound-HTTPS probe (useful in CI / air-gapped runs) |
| `--json` | Machine-readable output for CI |
| `--no-color` | Strip ANSI codes (log capture) |
| `--platform {windows,linux,macos}` | Force a different OS's check set — useful for verifying Windows install instructions from a Linux dev box |

Sections of the report, in order:

- **REQUIRED** — Python 3.11+, git, home / temp writable, disk space, outbound HTTPS, plus per-OS items (Windows long-paths registry; macOS Xcode CLI tools).
- **SANDBOX** — Docker Desktop (or `unshare` on Linux; `taskkill` on Windows for the cross-platform tree-kill).
- **RECOMMENDED** — POSIX `sh` on Windows (Git Bash for schedule hooks), security scanners (gitleaks / bandit / semgrep / trivy), formatters (ruff / prettier / google-java-format / shellcheck).
- **OPTIONAL** — `gh` CLI, language toolchains (Python / Java / Node — the locked stack) for the stacks the LLM may target.
- **ENV** — informational; which provider API key env vars are set on this machine.

Once `pre-flight` reports green REQUIRED, follow §5 to install the harness package, then `teane doctor -r <workspace>` for the workspace-bound checks.

## 3. Prerequisites

> The same checklist below is what `teane pre-flight` (§2.5) probes for you automatically. If you'd rather follow the install commands the tool prints than read the matrix, skip to §5 (install the package), then run `teane pre-flight` and iterate.

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

The `sandbox.backend` key in `config.json` accepts `auto` (default), `docker`, `unshare`, or `bare`. `auto` probes Docker first and falls back to `unshare` on Linux/WSL2.

| Backend | Linux | macOS | Win + WSL2 | Win native | Isolation |
|---------|-------|-------|------------|------------|-----------|
| `docker` (recommended) | ✓ | ✓ Docker Desktop | ✓ Docker Desktop + WSL integration | ✓ Docker Desktop, Linux containers | Strongest |
| `unshare` | ✓ | ✗ | ✓ inside the distro | ✗ not on Windows | Linux namespaces |
| `bare` (unsafe) | opt-in | opt-in | opt-in | opt-in | None — runs LLM-generated commands directly on the host |
| `auto` | tries docker → falls back to unshare; both unavailable = startup failure |

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
- **Windows native**: install Docker Desktop. Confirm it's running in **Linux containers** mode (right-click the tray icon → "Switch to Linux containers" if needed). The default sandbox image is `harness-builder:latest`; the compiler node auto-swaps to a stack-specific image (`python:3.12-slim`, `node:20-slim`, `eclipse-temurin:21-jdk`) based on tokens it recognises in the auto-wired build command.

### unshare

Pre-installed on every Ubuntu/Debian via `util-linux`. Smoke-test it:

```bash
unshare --user echo ok
```

If you see `ok`, you're good. If it fails, user namespaces are disabled at the kernel level (RHEL/Fedora may need `sysctl -w user.max_user_namespaces=15000` and a SELinux/AppArmor exception).

### bare

Opt-in only — the harness refuses to run with `sandbox.backend = "bare"` unless the operator sets `HARNESS_ALLOW_UNSAFE_SANDBOX=true`. Never enable outside a disposable VM — it runs LLM-generated build commands directly on the host with zero isolation.

## 5. Clone & Install the Package

The package is currently distributed from source. A future `pip install teane` will work once published to PyPI; for now use source.

### Linux / macOS / WSL2

```bash
git clone <repo-url> teane
cd teane
python3.11 -m venv ~/.venvs/teane
source ~/.venvs/teane/bin/activate
pip install .
teane --version
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
git clone <repo-url> teane
cd teane
py -3.11 -m venv $HOME\.venvs\teane
& $HOME\.venvs\teane\Scripts\Activate.ps1
pip install .
teane --version
```

If `Activate.ps1` is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (one-time).

### Windows native (cmd)

```cmd
git clone <repo-url> teane
cd teane
py -3.11 -m venv %USERPROFILE%\.venvs\teane
%USERPROFILE%\.venvs\teane\Scripts\activate.bat
pip install .
teane --version
```

On Windows native the console script lands at `…\.venvs\teane\Scripts\teane.exe`. If `teane --version` fails after activation, your venv didn't activate — re-check the activation command.

For an editable install (recompile-free local edits), substitute `pip install -e .`.

#### Make-free workflows (Windows native)

The repo ships a `Makefile` but `make` is not installed by default on Windows. Each target maps to a one-liner you can run from PowerShell or `cmd` — drop the `make` wrapper and invoke the Python directly:

| Makefile target | Windows-native equivalent |
|---|---|
| (pre-install check) | `python -m harness.cli pre-flight` |
| `make setup` | `python scripts\setup.py` |
| `make build` | `python -m compileall .` |
| `make test` | `python -m pytest tests\ -q --tb=short` |
| `make coverage` | `python -m pytest tests\ --cov=harness --cov-report=term-missing:skip-covered --cov-report=html:htmlcov --cov-report=xml:coverage.xml -q --tb=short` |
| `make hooks-install` | `python -m pre_commit install` |
| `make release BUMP=patch` | `python scripts\release.py --bump=patch` |

Operators who prefer the `make` ergonomics can install GNU Make via `winget install GnuWin32.Make`, `scoop install make`, or Git Bash (which ships an old but workable `make`). None of those are required — the table above covers every shipped target.

## 6. Provision API Keys (Required)

> A configured LLM API key is **required** to reach a green build. Strict config validation in `discover_config()` exits with code 2 at startup when any provider referenced by `model_routing` has no key. The auto-test-generation node additionally refuses to run without one and routes to HITL with `env_misconfig:llm_api_key`.

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

> **Do not** embed live API keys in `config/config.json`. The schema slot is kept for documentation only; the gateway reads env vars at dispatch time and ignores any non-empty `api_key` field.

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
| `prettier` (JS / TS / TSX / JSON / MD — covers React + Tailwind) | `npm install -g prettier` |
| `google-java-format` (Java) | download the jar from [google/google-java-format releases](https://github.com/google/google-java-format/releases) |

### Type checkers (recommended — power the static diagnostics gate)

The diagnostics gate (`diagnostics_node`, config section `diagnostics`) runs
fast read-only type checks after every patch round, BEFORE the expensive
compile — new type errors get repaired in-loop instead of burning a build.
With none of these installed the gate silently contributes nothing
(fail-open); the run still works, just with later/coarser error signal.

| Tool | Install | Notes |
|------|---------|-------|
| `pyright` (Python — preferred) | `pip install pyright` (or `npm install -g pyright`) | Structured `--outputjson` diagnostics |
| `mypy` (Python — fallback) | `pip install mypy` | Used only when pyright isn't on PATH |
| `tsc` (TypeScript / TSX) | `npm install -g typescript` | Run as `tsc --noEmit` against the nearest `tsconfig.json` |

Java needs nothing here: the Maven/Gradle build itself is Java's type check.

### LSP servers (optional — brownfield semantic navigation)

For brownfield flows (`teane patch` / `teane test`), the harness can spawn
language servers for ground-truth find-references / go-to-definition
(config section `lsp`; see FR-077). Servers only start when the workspace
can resolve imports — Python needs a `.venv`/`venv` at the workspace root
(or set `lsp.python_require_venv=false`), TypeScript needs `tsconfig.json`
AND an installed `node_modules`. Missing servers degrade silently to the
tree-sitter dependency-graph heuristics; `teane build` never starts them.

| Tool | Install |
|------|---------|
| `pyright-langserver` (Python) | ships with `pip install pyright` |
| `typescript-language-server` (TS / TSX) | `npm install -g typescript-language-server typescript` |

### GitHub CLI (optional — required for `teane gh` subcommands)

The `teane gh issue` / `pr-create` / `pr-comment` subcommands shell out to the [`gh` CLI](https://cli.github.com/). Install once and authenticate; the harness never stores tokens.

| Platform | Install |
|----------|---------|
| Linux (Debian/Ubuntu) | `sudo apt install gh` (or the [official installer](https://github.com/cli/cli/blob/trunk/docs/install_linux.md)) |
| macOS | `brew install gh` |
| Windows native | `winget install GitHub.cli` |

Then authenticate once:
```bash
gh auth login
```

Skip this section if you only run the harness against your own change-request files; `teane gh` is the only place that needs `gh` on PATH.

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

`teane doctor` adds one row per server when `mcp.enabled=true` showing the tool count or the rejection reason.

## 8. Configure the Harness

The harness reads exactly one config file: `<teane_root>/config/config.json`. There are no fallbacks, no per-workspace overrides, no auto-generated files. The setup wizard writes this file with sane defaults; thereafter every behaviour change goes here.

Strict validation runs at every CLI entry point before any logging, lock, or LLM-gateway initialisation:
- Unknown top-level or nested keys → fail (catches typos like `token_budget.hrad_cap_usd` that used to silently no-op).
- Missing required fields (`product_spec_dir`, the three `*_primary` routing keys) → fail.
- Wrong types, malformed model keys, dangling routing references → fail.
- A provider referenced in `model_routing` with no `{PROVIDER}_API_KEY` env var → fail.

On any failure the harness prints the multi-line ConfigError to stderr and exits with code 2. The dashboard editor (Configure Harness page) and `teane doctor` run the same validator so the same mistakes surface in the same words.

A minimal `config/config.json` body:

```json
{
  "product_spec_dir": "product_spec",
  "core_languages": {
    "backend_language": "Python",
    "web_language": ["React", "TypeScript", "TailwindCSS"]
  },
  "allow_network": true,
  "models": {
    "anthropic:claude-sonnet-4": {
      "provider": "anthropic",
      "model_id": "claude-sonnet-4-20250514",
      "context_window": 200000,
      "input_cost_per_1m": 3.0,
      "output_cost_per_1m": 15.0,
      "api_base_url": "https://api.anthropic.com/v1",
      "supports_thinking": false,
      "supports_cache": true,
      "api_key": ""
    }
  },
  "model_routing": {
    "planning_primary": "anthropic:claude-sonnet-4",
    "planning_mode": "thinking_max",
    "patching_primary": "anthropic:claude-sonnet-4",
    "patching_mode": "no_thinking",
    "repair_primary": "anthropic:claude-sonnet-4",
    "repair_mode": "no_thinking"
  },
  "token_budget": { "hard_cap_usd": 3.0, "context_window_threshold_pct": 0.85 },
  "persistence": { "db_path": "~/.harness/checkpoints.db", "ttl_days": 30 }
}
```

The shipped `config/config.json` in the repo is annotated: every section has a sibling `_<section>_comment` string that documents every leaf field. `_*` keys are stripped at load time, so comment fields ship in the file but never reach the validator.

**Mandatory `core_languages` (locked stack).** Backend is Python (FastAPI / Flask / Django) OR Java (Spring Boot). Web is exactly React + TypeScript + TailwindCSS, Vite-built. Blank `backend_language` resolves to `Python`; blank `web_language` resolves to the documented triple. Any other value (e.g. `backend_language: "Go"`, `web_language: ["Vue", "TypeScript", "TailwindCSS"]`) causes the harness to exit with code 2 at config-load time before any logging or LLM-gateway init.

**Build command is auto-wired from workspace markers.** The legacy `build_command` config key and the `--build-cmd` CLI flag have been REMOVED. The harness now picks the build command at runtime from what it finds in the workspace: `pyproject.toml` → `pytest`, `pom.xml` → `mvn -B test`, `package.json` → `npm install && npm run build && npm test`.

**Mandatory `product_spec_dir`.** The value is a bare folder name (no path separators, no `..`, no absolute paths) that lives at the workspace root. The harness consolidates every spec file in alphabetical order — `.txt` and `.md` read as UTF-8, `.pdf` bodies extracted via `pypdf` — and feeds the result to the planning LLM. `teane doctor` checks the value is well-formed AND the folder exists with ≥1 spec file (`.txt`, `.md`, or `.pdf`).

**Model registry vs routing.**
- `models` is the registry — every LLM the gateway can dispatch to must be declared here with `provider`, `model_id`, `context_window`, `input_cost_per_1m`, `output_cost_per_1m`, `api_base_url`, `supports_thinking`, `supports_cache`, and an empty `api_key` slot. The key (e.g. `anthropic:claude-sonnet-4`) is the routing handle.
- `model_routing` binds the harness's internal roles (`planning`, `patching`, `repair`, optional `doc_reviewer` / `code_reviewer`) to registry keys. Only routed models need an env-var key; declared-but-unrouted entries are inert.
- `*_mode` controls extended thinking: `non_thinking`, `thinking`, or `thinking_max`. Only models with `supports_thinking: true` accept thinking modes.

The full schema — every field of `sandbox`, `token_budget`, `node_throttle`, `persistence`, `logging`, `lintgate`, `deployment`, `test_generation`, `metrics`, `web_tools`, `mcp`, `memory`, `repo_index`, `schedule`, `dashboard`, `deployment_defaults` — is documented in [docs/SPEC_REQUIREMENTS.md](SPEC_REQUIREMENTS.md) and in the inline comments of `config/config.json`.

**API key resolution.** Each provider key is resolved at runtime by `harness/gateway.py`: (1) explicit constructor argument, (2) `{PROVIDER}_API_KEY` env var (recommended), (3) the `api_key` field on the model entry as a last-resort dev knob (the shipped schema keeps this empty and discourages live keys). `teane doctor`'s `api keys (live)` check makes a 1-token chat call per routed provider and reports the source per model — `... (env)` or `... (config)` — so you can confirm at a glance where the runtime resolved from. Set `HARNESS_DOCTOR_SKIP_LIVE=true` to skip the live ping (CI / outbound-blocked hosts).

**Legacy per-workspace config files.** A `.harness_config.json` left over from the layered-config era is **ignored** with one INFO log line per run; the harness no longer reads it. Delete it at your leisure — nothing reads or writes it.

**Recent config additions worth knowing**:
- `patcher.enforce_read_before_edit` (default `false`) — when true, rejects REPLACE_BLOCK / DELETE_BLOCK / INSERT_AT_BLOCK against any file the LLM has not yet been shown this turn. Mirrors Claude Code's Read-before-Edit invariant.
- `llm_dispatch.continue_on_length` (per-role bool map) — when finish_reason=length, re-prompt the model with the partial reply up to 3 cycles. Default on for `patching` only; raising it on JSON roles (doc_reviewer / code_reviewer) frequently breaks schema parsing — see the inline comment in `config.json`.
- `llm_dispatch.prompt_cache_enabled` (default `true`) — emits Anthropic `cache_control` markers on the system block and runs prefix-stability drift detection. Flip to false only if a provider API change rejects the payload shape.
- `compiler.run_prod_import_smoke_check` (default `true`) — compiler_node imports every production module inside the sandbox before running the build, so module-level errors surface as `[prod-import]` diagnostics ahead of any cascade-amplified test failures.
- `debug.dump_llm_calls` (default `true`) — every LLM dispatch is written to `~/.harness/debug/*.txt`; `debug.dump_max_files` caps the directory (oldest by mtime pruned). Useful for post-mortem; turn off in production-style runs.
- `sandbox.cache_volumes` (default `false`) — swap each `readonly_cache_mounts` entry for a writable Docker named volume scoped to the session id. Pip / npm / maven downloads persist across containers in the session. Clean up with `teane cache clear`.
- `node_throttle.max_patch_repair_iterations` (default `3`, can be raised) — repair loop ceiling. After this many failing rebuilds the run routes to HITL.

**WSL2 only**: put your workspaces under the WSL filesystem (`~/...`), **not** `/mnt/c/...`. Windows-mount paths have order-of-magnitude slower I/O.

### Workspace single-writer lock

Every run (`teane build` / `patch` / `deploy` / `test`) and `teane resume` acquires an exclusive lock on `<workspace>/.harness_session.lock`. A second concurrent run on the same workspace exits with `lock held by PID X`. To recover after a hard kill that left the lock stale:

```bash
teane build -w <workspace> -p "<prompt>" --force-lock
```

`--force-lock` releases the stale lock and acquires a fresh one, logging a WARNING so the override is visible in the session record. The lock is implemented via `fcntl.flock` on Linux/macOS/WSL2 (advisory) and `msvcrt.locking` on Windows native (mandatory) — both dispatched through `harness/_filelock.py`. See `docs/RUNBOOK.md` § 4 for the full recovery recipe.

## 8.5 Test generation (new)

After every patching round, the harness writes stack-canonical unit tests for the modified source files and runs them deterministically in the sandbox before lintgate. The node — `harness/test_generation.py` — sits in the graph between `speculative_node` and `lintgate_node`.

**What it does.** Detects the workspace stack via `_detect_workspace_stack`, loads the matching `harness/test_guides/<lang>.md` into the LLM prompt, asks for `CREATE_FILE` / `INSERT_AT_BLOCK` patch blocks for the corresponding test files, applies them, then invokes a stack-canonical test command in the sandbox. **The guides instruct the LLM to write tests that exercise the real code — no mocks.** When a side effect can't be invoked directly, the tests use the test runner's built-in fakes (pytest `monkeypatch` / `tmp_path`, JUnit `@TempDir`, Jest fake-timers, etc.).

**Config defaults.** Shipped in `config/config.json` under `test_generation`. Defaults are `enabled: true` and `max_iterations: 3`. To disable, edit the same file:

```json
"test_generation": { "enabled": false }
```

**LLM API key is required.** Without a configured gateway the node synthesises an `env_misconfig:llm_api_key` diagnostic and routes the session to HITL. See §6 — provisioning at least one provider key is no longer optional when test generation is on.

**Per-stack test runner.** Stack-canonical guides ship at `harness/test_guides/*.md` for `python`, `java`, and `typescript` (covering the React + Tailwind web build). The deterministic command the sandbox runs per detected stack:

| Stack | Deterministic command |
|-------|------------------------|
| Python | `pip install -q pytest && python3 -m pytest -q` |
| Java | `mvn -q test` |
| TypeScript (React + Tailwind) | `npm install --no-save --silent jest ts-jest typescript && npx jest --silent` |

`pip install` / `npm install` / `mvn` tokens trigger the harness's existing install-network heuristic, so the sandbox auto-enables outbound network for the test run — no manual `allow_network: true` required.

**Sandbox image caveat (read this).** The deterministic test runner re-uses whichever `sandbox.docker_image` the auto-adapter picked for the build command. The adapter keys off the auto-wired build command — if your workspace markers don't unambiguously imply a stack, set `sandbox.docker_image` explicitly in `config/config.json`:

```json
"sandbox": { "docker_image": "python:3.12-slim" }
```

Per-stack images that ship the toolchain: `python:3.12-slim`, `eclipse-temurin:21-jdk` (Java), `node:20-slim` (React + TypeScript + Tailwind web).

**Project-level test conventions.** Drop your own files under `<workspace>/test_guides/<lang>.md` with frontmatter `applies_to: [<stack-tag>]`. The loader prefers project files over the shipped defaults, so a workspace can tighten the conventions the LLM is given without forking the harness.

## 9. Verify the Install (`teane doctor`)

`cd` into any workspace whose `product_spec/` folder is populated and run:

```bash
teane doctor
```

Expected output: a banner with the resolved canonical config path, then a row per check. `config` is the gate — if it fails, every downstream check is marked `skip` and doctor exits non-zero.

| Check | What it verifies | Failure means |
|-------|------------------|---------------|
| `config` | `<repo>/config/config.json` exists, parses as JSON, and passes strict validation | Fix the multi-line error printed under this row; doctor refuses to run downstream checks until config is clean |
| `git repo` | Workspace is a git repo with at least one commit | `git init` + first commit, or pass `-r <other-workspace>` |
| `product spec` | `product_spec_dir` is a valid workspace-root folder name AND the folder has ≥1 spec file (`.txt`, `.md`, or `.pdf`) | Create `<workspace>/<product_spec_dir>/` and drop the spec text in |
| `api keys (live)` | Every provider referenced in `model_routing` has a key in `{PROVIDER}_API_KEY` (or the `models[].api_key` field), AND each passes a one-token live ping that confirms the key authenticates against the configured model | Set the missing key, fix an `HTTP 401 — API key rejected`, or set `HARNESS_DOCTOR_SKIP_LIVE=true` (CI / outbound-blocked hosts) |
| `tree-sitter` | The `tree-sitter` import works and at least one bundled grammar parses a sample buffer | `pip install -U tree-sitter tree-sitter-language-pack`; without it, `patcher` and `impact` silently degrade to regex extraction |
| `sandbox backend` | Resolves `sandbox.backend`: `docker` probes `docker info`; `unshare` probes `unshare --user echo ok`; `auto` tries docker → unshare in order; `bare` warns (no isolation) | Re-do §4 for your platform |
| `checkpoint db` | `persistence.db_path` is writable AND the 5 most recent checkpoints deserialize cleanly | Adjust `persistence.db_path`, or `teane purge --session-id <id>` for a corrupted session |
| `patcher mode` | Reports the two patcher behaviour flags: `read-before-edit` (B5) and `native tool-use` (B6) | Informational; never fails |
| `external tools` | One row per shelled-out tool (formatters, security scanners, `gh`) the operator's config / workspace touches | `warn` for a missing optional tool; `pass` confirms PATH resolution |
| `mcp:<server>` | Only when `mcp.enabled: true` — starts each declared server subprocess and lists the advertised tools | Fix the `command` / runtime, or remove the server entry |

The `api keys (live)` PASS message includes the source per model — `... (env)` or `... (config)` — and a per-provider tag. Cost is well under a tenth of a cent across all providers combined. To skip the live ping (e.g. CI where outbound network is blocked), set `HARNESS_DOCTOR_SKIP_LIVE=true`; the doctor then reports the source per model and notes `(live ping skipped via HARNESS_DOCTOR_SKIP_LIVE)`.

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

Pick a throwaway git repo and drop a one-line spec under `product_spec/` (the harness refuses to start without it):

```bash
git clone https://github.com/octocat/Hello-World.git sample
mkdir -p sample/product_spec
echo "List the top-level files in this repository." > sample/product_spec/SPEC.txt
teane patch -w ./sample -p "list the top-level files"
```

The harness will:

1. Acquire an exclusive lock on `./sample/.harness_session.lock` (`fcntl.flock` on POSIX, `msvcrt.locking` on Windows native).
2. Create `~/.harness/checkpoints.db` (or `%USERPROFILE%\.harness\checkpoints.db` on Windows native) and `~/.harness/logs/<session-id>.jsonl` (rotated at 10 MB × 5 backups by default).
3. Consolidate every spec file (`.txt`, `.md`, `.pdf`) in `product_spec/`, then run the planning → patching → compile → lintgate loop, checkpointing each step.
4. Append a session note under `~/.harness/memory/<repo_id>.md` if `memory.enabled: true` (the default).

A successful smoke test exits 0. Inspect the run with `teane status --session-id <id>` or by tailing the JSONL log file. After the run, `teane metrics --session-id <id>` shows cost, burn rate, and projected exhaustion against your `token_budget.hard_cap_usd`.

**Bare `teane build` / `teane patch`** with no `-w`/`-p` flags drops into the interactive wizard — it walks API keys, workspace, prompt, `--git`, and `--spec-discovery` (the greenfield-vs-brownfield choice is already made by the target you invoked), then hands off to the engine or `cmd_resume`. The wizard never persists anything.

For diagnostics on a stuck or failed run, start with [`docs/RUNBOOK.md`](RUNBOOK.md).

## 11. Headless / Server Deployment

For unattended runs (CI, scheduled jobs, services):

- Set `CI=true` and `HARNESS_AUTO_APPROVE=true` to bypass interactive HITL prompts. The four `--hitl-*` flags (`req`, `arch`, `repair`, `deployment`) default to false anyway; the env vars are belt-and-braces.
- Set `HARNESS_HITL_WEBHOOK_URL` (and optional `HARNESS_HITL_WEBHOOK_SECRET` for HMAC-SHA256 signing) if you want sensitive operations to require approval via a webhook instead of stdin. The dashboard exports both automatically when it spawns runs, so HITL gates surface in the UI.
- Ensure NTP is running — clock skew breaks API auth on fresh VMs.
- With test_generation on (the default — see [§8.5](#85-test-generation-new)), each session spends additional LLM tokens for test authorship and may hit the default `token_budget.hard_cap_usd = 3.00` on larger changes. Raise the cap in your config, or set `test_generation.enabled: false` under CI when budget pressure matters.
- Track aggregate cost with a cron job: `teane metrics --all --prometheus --output /var/lib/node_exporter/textfile/harness.prom`. The atomic-write contract means a scraper never sees a half-written file.
- For long-running services, leave `logging.max_bytes` / `logging.backup_count` at defaults (10 MB × 5) — that's ~50 MB max per session, dropped oldest-first.
- If a scheduled job dies hard and leaves the workspace lock stale, the next run will refuse to start. Wrap with a retry that adds `--force-lock` on second attempt only.

### Linux (systemd)

`harness` ships three persistent processes you might want to systemd-ify: a one-shot build/patch run, the schedule daemon, and the web dashboard. Each runs the venv's `harness` console script. `HOME` must be set on every unit so `~/.harness/` path expansion works.

One-shot run (equivalent to the operator typing `teane patch`):
```ini
[Service]
Environment=HOME=/srv/harness
Environment=CI=true
Environment=HARNESS_AUTO_APPROVE=true
Environment=ANTHROPIC_API_KEY=sk-...
WorkingDirectory=/srv/harness/workspace
ExecStart=/srv/harness/.venvs/teane/bin/teane patch -w . -p "..."
```

Scheduled-job daemon (`teane schedule run` — fires `schedule.jobs` from
`config/config.json` on a cron-style timer; restart-on-failure recommended):
```ini
# /etc/systemd/system/harness-schedule.service
[Unit]
Description=teane scheduled-job daemon
After=network-online.target

[Service]
Type=simple
User=harness
Environment=HOME=/srv/harness
Environment=ANTHROPIC_API_KEY=sk-...
WorkingDirectory=/srv/harness/teane
ExecStart=/srv/harness/.venvs/teane/bin/teane schedule run
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Web dashboard (`teane web start` foreground; writes enabled by default — set
`dashboard.writes_enabled: false` in `config.json` for a read-only deployment).
The dashboard is single-instance per user, gated by `~/.harness/web.lock`; a
second `start` while the marker points at a live pid refuses to launch.

```ini
# /etc/systemd/system/harness-dashboard.service
[Unit]
Description=teane web dashboard
After=network-online.target

[Service]
Type=simple
User=harness
Environment=HOME=/srv/harness
# Bearer token gate (required when binding off-localhost):
Environment=DASH_TOKEN=replace-with-a-long-random-string
# Optional: persistent CSRF token across restarts:
Environment=DASH_CSRF=replace-with-another-long-random-string
WorkingDirectory=/srv/harness/teane
ExecStart=/srv/harness/.venvs/teane/bin/teane web start --host 127.0.0.1 --port 9000
ExecStop=/srv/harness/.venvs/teane/bin/teane web stop
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then in `config/config.json`:
```jsonc
{
  "dashboard": {
    "enabled": true,
    "host": "127.0.0.1",          // localhost only by default
    "port": 9000,
    "token_env": "DASH_TOKEN",
    "csrf_token_env": "DASH_CSRF",
    "writes_enabled": true,       // set false for a read-only deployment
    "hitl_webhook_secret": "",    // shared secret the harness POSTs with
    "web_db_path": "~/.harness/web.db"
  }
}
```

Enable + start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now harness-schedule.service
sudo systemctl enable --now harness-dashboard.service
sudo journalctl -fu harness-schedule.service   # follow logs
```

Run separate units rather than bundling — the dashboard config changes mean only the dashboard restarts, and a stuck dashboard request never affects schedule timing.

For ad-hoc background mode without systemd, `teane web start --background yes` re-spawns the server detached and logs to `~/.harness/web.log`; `teane web stop` reads the marker, SIGTERMs the pid, and escalates to SIGKILL after 5 s.

### Windows native (Task Scheduler)

- Create a task running `…\.venvs\teane\Scripts\teane.exe` with arguments `run -w C:\path\to\workspace -p "..." --new-build false`.
- Set **Start in** to the workspace directory.
- Add `CI`, `HARNESS_AUTO_APPROVE`, and `ANTHROPIC_API_KEY` as system or user env vars (Task Scheduler inherits the account's env).
- For a long-lived service, wrap the command with [NSSM](https://nssm.cc/).

## 12. Upgrading & Uninstalling

### Upgrade

```bash
cd teane
git pull
pip install -U .
teane doctor
```

(Or `pip install -U teane` once published to PyPI.)

### Uninstall

```bash
pip uninstall teane
rm -rf ~/.harness     # Linux/macOS/WSL2
```

`~/.harness/` holds the checkpoint DB, per-session JSONL logs, metrics, repo memory files, the repo index (`repo_index/`), schedule history (`schedule.db`), web app state (`web.db`), and the user-skills directory (`user_skills/` — legacy installs may still use `skills/`). Remove only what you don't want to keep.

On Windows native: `pip uninstall teane` then `Remove-Item -Recurse $HOME\.harness` in PowerShell.

## 13. Troubleshooting

The canonical mid-session failure recipes (checkpoint corrupted, budget exhausted, sandbox dead, workspace lock refused, persistent LLM silence, dashboard 401/403, schedule daemon stuck, repo index empty, …) live in [`docs/RUNBOOK.md`](RUNBOOK.md). This list adds install-specific gotchas the runbook doesn't cover.

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
| `env_misconfig:<symbol>` (e.g. `env_misconfig:pytest`) | Sandbox build exited with "No module named X" / "command not found" / Docker `exec: "X": executable file not found`. The runtime is missing inside the container. | Set `sandbox.docker_image` to one that ships `<symbol>` (`python:3.12-slim`, `eclipse-temurin:21-jdk`, `node:20-slim`), or add the dependency to the workspace's `pyproject.toml` / `pom.xml` / `package.json` so the auto-wired build command installs it. |
| `env_misconfig:llm_api_key` | test_generation cannot run because no LLM gateway is configured. | Set the matching `*_API_KEY` env var (see §6), or set `test_generation.enabled: false` in `config/config.json`. |
| `llm_silent` (HITL fires immediately) | Three consecutive empty responses from the LLM provider (`EmptyLLMResponseError`). | Check the provider's status page; retry, or route the affected node to a different model via `model_routing`. |
| `Auto-test run fails with "pip: command not found" / "npx: not found"` | The deterministic test runner from §8.5 is executing in a docker image that doesn't carry the stack toolchain. | See the [§8.5 sandbox-image caveat](#85-test-generation-new) — pin `sandbox.docker_image` explicitly to a per-stack image. |
| `[FAIL] api keys (live)` despite a key being set | Doctor checks env vars AND the `models["<key>"].api_key` config field. The FAIL detail line names which location is empty and what the live ping returned (`HTTP 401`, `HTTP 404`, …). | Set the key in `{PROVIDER}_API_KEY` env var (preferred) or in `config/config.json` under `models."<key>".api_key`. |
| `lock held by PID <n>` at startup | A prior run exited hard and left `<workspace>/.harness_session.lock` stale, or another live session is using the workspace. | Verify the PID is gone (`ps -p <n>`); then re-run with `--force-lock` to release and reacquire. See `docs/RUNBOOK.md` § 4. |

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
| `config/config.json` saved with CRLF line endings | Harmless for JSON parsing; if a downstream tool chokes, run `git config --global core.autocrlf input` and re-save with LF |

### Windows + WSL2

| Symptom | Fix |
|---------|-----|
| Builds run extremely slowly | Workspace is under `/mnt/c/...`; move it to the WSL filesystem (`~/...`) |
| API auth fails after the laptop sleeps | WSL2 clock drift; `sudo hwclock -s` (or install `chrony` and let it run) |
| `teane doctor`'s sandbox check fails despite Docker Desktop running | "WSL Integration" toggle is off for your distro — enable it in Docker Desktop → Settings → Resources → WSL Integration |

## 14. Next Steps

- `harness <subcommand> --help` — every flag of `teane build`, `patch`, `deploy`, `test`, `resume`, `status`, `doctor`, `purge`, `metrics`, `web start/stop`, `schedule run/list/validate/once/history`, `chat`, `index build/status/clear`, `gh issue/pr-create/pr-comment`, `cache clear`.
- [docs/SPEC_REQUIREMENTS.md](SPEC_REQUIREMENTS.md) — full `config/config.json` schema.
- [docs/SPEC_ARCHITECTURE.md](SPEC_ARCHITECTURE.md) — graph topology and module map.
- [docs/RUNBOOK.md](RUNBOOK.md) — mid-session failure recipes for operators.
- [docs/EDGE_CASE_AUDIT.md](EDGE_CASE_AUDIT.md) — known edge cases and their handling.
- `harness/style_guides/*.md` and `harness/test_guides/*.md` — shipped per-language guidance the LLM sees during code and test generation. Drop your own overrides under `<workspace>/style_guides/` or `<workspace>/test_guides/` (same filenames win on collision).
- `harness/skills/*.md` — stack scaffolds the planner uses for greenfield projects in the locked stack (Django, FastAPI, Flask, Spring Boot, React + TypeScript + TailwindCSS).
