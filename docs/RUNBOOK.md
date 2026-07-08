# Harness Operations Runbook

Self-serve recovery for the failure modes operators hit most often. Each
entry has a one-line symptom, a diagnostic command that confirms the
cause, and a fix recipe with explicit commands.

When in doubt, **run `teane doctor` first** — it executes a strict-validation
pass on the canonical config plus per-subsystem healthchecks (git repo,
product spec, live API-key ping, tree-sitter, sandbox backend, checkpoint
DB, patcher mode, external tools, optional MCP servers) and prints a
colored summary pointing at the broken subsystem. The harness reads ONE
config file: `<teane_root>/config/config.json`. There is no
`~/.harness/config.json` and no per-workspace overrides.

```bash
teane doctor -r /path/to/workspace
```

If `doctor` is green and you're still stuck, the entries below cover the
failure modes that have actually caused operator pain. They are
ordered by frequency, not severity.

---

## 1. Checkpoint corrupted — `teane resume` refuses to load

**Symptom**

```
[resume] Checkpoint for session '<id>' is corrupted: ...
  Options:
    - Start a fresh session with `teane build -w <ws> -p '<prompt>'` (or `teane patch` for brownfield).
    - Restore checkpoints.db from a known-good backup.
    - Run `teane purge --session-id <id>` to drop only this session.
```

**Diagnose**

```bash
# Confirm which session(s) have unreadable blobs without altering them.
teane doctor
# Look for the "checkpoint db" check — it scans the 5 most recent rows.

# Inspect the offending session non-destructively:
teane status --session-id <id>
```

**Fix**

Choose one of three paths, in order of preference:

1. **Restore from backup** (preferred when a recent backup exists):
   ```bash
   cp ~/.harness/checkpoints.db.bak ~/.harness/checkpoints.db
   teane resume --session-id <id>
   ```
2. **Drop only the broken session, keep all others** (`teane purge` also removes the rotated JSONL log files for the session):
   ```bash
   teane purge --session-id <id>
   teane build -w <ws> -p "<original prompt>"   # start fresh (or `teane patch`)
   ```
3. **Last resort, nuke everything**:
   ```bash
   teane purge --all
   ```

**Why it happens.** The msgpack blob in the SQLite store didn't decode.
Usually a partial write from a hard crash mid-checkpoint, or a disk-full
event during a flush. WAL-mode SQLite recovers most cases automatically;
the warning means recovery already ran and the row is still bad.

---

## 2. Budget exhausted mid-session

**Symptom**

```
ERROR  Budget exceeded: cumulative spend $2.0123 > cap $2.00
```

or, when a single planned call would push you over:

```
ERROR  BudgetTooLowError: pre-flight estimate ($0.0123) exceeds remaining ($0.0080)
```

**Diagnose**

```bash
# Headline: total cost, burn rate, projected exhaustion at current rate.
teane metrics --session-id <id>

# Roll-up across every session in the log dir:
teane metrics --all

# Legacy fallback (when the harness binary itself is the problem):
grep '"event": "llm_call"' ~/.harness/logs/<session-id>.jsonl | \
  jq -s 'map({model, cost_usd, tokens_in, tokens_out}) |
         group_by(.model) |
         map({model: .[0].model, total_cost: (map(.cost_usd) | add), calls: length})'
```

The `teane metrics` output shows total cost, per-window burn rate,
and an estimated minutes-until-exhaustion at the current rate. The
legacy jq recipe still works and is useful when the CLI itself is the
thing that's broken.

**Fix**

- **Raise the cap and resume** (most common):
  ```bash
  # Edit <teane_root>/config/config.json:
  #   "token_budget": { "hard_cap_usd": 5.00 }
  teane resume --session-id <id>
  ```
- **Re-route an expensive node to a cheaper model**: edit `model_routing.*`
  in the same file to point a hot node (e.g. `code_reviewer_primary`)
  at a smaller model.
- **Force local Ollama for the rest of the session**: set
  `model_routing.force_local_only: true` (and ensure
  `model_routing.ollama_local_model` references an installed Ollama model)
  then resume.

**Why it happens.** Discovery loops, doc reviews, code reviews, and the
auto-test-generation node can each take 3–5 LLM round-trips. A complex
workspace with all of them active will land in the $1–$3 range. The
shipped `hard_cap_usd: 3.00` is a guardrail, not a target.

---

## 3. Sandbox can't start

**Symptom**

```
ERROR  Sandbox backend init failed: docker: command not found
```

or, when Docker is present but unhealthy:

```
ERROR  Sandbox init failed: cannot connect to Docker daemon
```

**Diagnose**

```bash
teane doctor
# Look at the "sandbox backend" check — it prints the backend in use
# and whether the binary/daemon is reachable.

# Manual probes:
docker info               # for docker backend
unshare --user echo ok    # for unshare backend (Linux/WSL2 only)
```

**Fix**

- **Daemon not running** (Linux):
  ```bash
  sudo systemctl start docker
  sudo usermod -aG docker $USER && newgrp docker
  ```
- **Wrong backend selected** — edit `sandbox.backend` in
  `<teane_root>/config/config.json`. Valid values: `auto` (try docker
  → unshare), `docker`, `unshare` (Linux user-namespaces), `bare`
  (host execution — requires `HARNESS_ALLOW_UNSAFE_SANDBOX=true` and
  should only ever run in a disposable VM).
- **Image missing** — let the compiler_node auto-swap to a stack-specific
  image on first run, or pre-pull:
  ```bash
  docker pull python:3.12-slim
  ```

**Why it happens.** First-run installs frequently forget the docker group
add (requires a re-login), and laptop suspend can leave the daemon socket
stale.

---

## 4. Workspace lock refused

**Symptom**

```
ERROR  Another harness session holds the lock on this workspace
       (PID <n>, started <ts>). Pass --force-lock to override.
```

**Diagnose**

```bash
# Identify the holder:
cat <workspace>/.harness_session.lock
ps -p <PID>          # is the PID still alive?

# If the PID is gone, the lock file is stale (process crashed).
```

**Fix**

- **Holder still alive** — wait for it, or kill it cleanly:
  ```bash
  kill <PID>          # SIGTERM first; SIGKILL only if it hangs
  ```
- **Holder dead, lock stale**:
  ```bash
  teane build -w <ws> -p "<prompt>" --force-lock   # same flag on `teane patch`
  ```
  `--force-lock` releases the stale lock and acquires a fresh one. It
  logs a WARNING so the override is visible in the session record.
  `teane resume` also accepts `--force-lock`.

**Why it happens.** The lock is an `fcntl.flock` exclusive lock — the OS
releases it when the process dies. A SIGKILL'd process should release
cleanly; a crashed kernel or hung NFS mount can leave it. Windows native
runs have no fcntl and skip locking entirely (single-writer is the
operator's responsibility there).

---

## 5. LLM keeps returning empty responses

**Symptom**

```
ERROR  EmptyLLMResponseError: provider returned empty content after 3 retries
```

The session may pause at HITL with `llm_silent=True`.

**Diagnose**

```bash
# Recent LLM errors and retry counts:
grep -E '"event": "(llm_empty_response|llm_circuit_open|llm_call)"' \
  ~/.harness/logs/<session-id>.jsonl | tail -20

# Check whether the circuit breaker fired:
grep llm_circuit_open ~/.harness/logs/<session-id>.jsonl
```

**Fix**

- **Provider outage** — wait it out, or switch routes:
  ```bash
  # In <teane_root>/config/config.json, point the affected node at a different model:
  #   "model_routing": { "planning_primary": "anthropic:claude-sonnet-4" }
  teane resume --session-id <id>
  ```
- **API key revoked, out of credit, or wrong model id** — `teane doctor`
  now makes a 1-token chat call per configured provider and reports the
  specific HTTP code:
  - `HTTP 401 — API key rejected` → key is bad, update it.
  - `HTTP 403 — key valid but no access to model '<id>'` → either pick
    a different model in `model_routing` or upgrade the account tier.
  - `HTTP 404 — model not found` → fix the model id spelling.
  - `HTTP 429 — rate limited` → quota exhausted right now; retry later
    or rotate to a different account.
  - `timeout / connection failed` → outbound network blocked at this
    host; check firewall / proxy.
  In CI or any environment where outbound HTTPS is blocked, set
  `HARNESS_DOCTOR_SKIP_LIVE=true` to fall back to a key-presence-only
  check.
- **Rate-limited** — the gateway's circuit breaker (3 failures / 5 min
  window) auto-diverts to local Ollama. Confirm Ollama is running:
  ```bash
  ollama list                       # Does the listed local model exist?
  curl -fsS http://localhost:11434  # Is the daemon reachable?
  ```

**Why it happens.** Three classes: transient provider hiccups (retried
automatically), expired credentials (operator fix), and rate limits
(circuit breaker handles automatically; persistent 429s mean the
provider's quota is too low for the workload).

---

## 6. MCP server fails to start (`teane doctor` shows `fail`)

### Symptom
- `teane doctor` reports `mcp:<server>: command rejected: ...` or
  `mcp:<server>: start failed: ...`.
- Planner emits `<<<MCP_CALL>>>` blocks; the tool result body is
  `{"error": "mcp server 'X' not registered..."}`.

### Diagnose
```bash
# `teane doctor` lists every configured server and the start outcome:
teane doctor 2>&1 | grep '^mcp:'

# Print the resolved server commands the harness will run:
python -c "
from harness.mcp_client import McpPoolConfig
from harness.cli import discover_config
cfg = McpPoolConfig.from_config(discover_config('.'))
for s in cfg.servers:
    print(s.name, '->', s.command)
"

# Manually start the server to see its stderr:
npx -y @modelcontextprotocol/server-fetch   # adjust to your config
```

### Fix
- **Command not in allowlist:** Add the binary basename to
  `mcp.command_allowlist` in `config/config.json` (the built-in allowlist
  covers `npx`, `npm`, `node`, `python`, `python3`, `uvx`, `pipx`,
  `docker`).
- **Filesystem server rejected:** Set
  `mcp.allow_local_filesystem_servers: true` if you've reviewed the
  blast radius — filesystem MCP gives the LLM raw host I/O.
- **`npx` hangs on first launch:** Pre-install once outside the
  harness so the package is cached: `npx -y @scope/server-name --help`.
- **Server crashes with `MODULE_NOT_FOUND`:** Check Node.js version
  meets the server's `engines` requirement.
- **Tools/call payload too large:** Raise `mcp.result_max_bytes` (default
  `200000`) if downstream LLM truncation is the wrong outcome for your
  server.

---

## 7. Prompt cache misses — `cache_prefix_drift` events flooding the log

### Symptom
- Anthropic / OpenAI / DeepSeek cost is much higher than expected on
  long sessions.
- `~/.harness/logs/<id>.jsonl` contains repeated
  `{"event": "cache_prefix_drift", ...}` lines.

### Diagnose
```bash
# How many drift events fired this session:
grep cache_prefix_drift ~/.harness/logs/<id>.jsonl | wc -l

# Which roles are drifting most:
grep cache_prefix_drift ~/.harness/logs/<id>.jsonl | jq -r .role | sort | uniq -c

# Hash transitions per role (helps spot WHICH preamble mutated):
grep cache_prefix_drift ~/.harness/logs/<id>.jsonl |
  jq -r '"\(.role): \(.prev_hash) → \(.now_hash)"'
```

### Fix
- A drift on the `planning` role usually means the blueprint generator
  is including a timestamp or random ID. Check the planning system
  prompt for mutating content.
- A drift on `patching` is most often a `READ_FILE` result where the
  file's mtime or contents changed — expected after a real edit.
- To disable the warning telemetry entirely while keeping caches:
  this is not a knob — the WARN is the point. Cache misses ARE
  happening; the event surfaces them.
- To roll back Anthropic cache markers entirely (e.g. a provider API
  change rejects the payload shape), set
  `llm_dispatch.prompt_cache_enabled: false` in `config/config.json`.

---

## 8. Schedule daemon stuck — jobs not firing

### Symptom
- `teane schedule list` shows a job that should have fired hours ago.
- `~/.harness/schedule.db` has no `schedule_runs` row for the expected
  fire time.

### Diagnose
```bash
# Is the daemon actually running?
ps aux | grep "teane schedule run" | grep -v grep
systemctl status harness-schedule.service   # under systemd

# What does the daemon think the next fire times are?
teane schedule list

# Validate the cron syntax:
teane schedule validate
```

### Fix
- **Daemon not running:** Start it (`teane schedule run`). The daemon
  does not auto-launch on `teane build` / `teane patch`; it's a separate process.
- **`schedule.enabled` is `false`:** Flip to `true` in
  `config/config.json` and restart the daemon.
- **Job marked `enabled: false`:** Flip to `true` on the per-job entry
  in `schedule.jobs[]` and restart the daemon.
- **Cron syntax silently fell through:** `teane schedule validate`
  surfaces the rejection. Supported subset: `every 15m` / `every 6h` /
  `every 3d` / `hourly :MM` / `daily HH:MM` / `weekly DAY HH:MM` (all
  times UTC; DAY ∈ `mon`–`sun`). Full POSIX cron like `30 2 * * mon` is
  NOT supported — use the subset above.
- **Run a job out of band:** `teane schedule once <name>` fires it
  immediately, regardless of schedule.
- **In-flight job stuck:** Check the per-job log under
  `~/.harness/schedule_logs/<job>/` — the daemon won't fire a second
  instance while the previous one is alive. Kill the stale process
  if the workspace lock is held.

---

## 9. Web dashboard returns 401 / 403

### Symptom
- Browser shows "401 unauthorized: missing Authorization header".
- POSTs from the editing UI return "403: csrf token mismatch" or
  "403: writes disabled".

### Diagnose
```bash
# Is the dashboard actually running on the expected port?
ss -tlnp | grep 9000       # default port (override with --port)

# Read the marker file to see what start parameters the live instance has:
cat ~/.harness/web.lock

# What token does the dashboard expect?
echo $DASH_TOKEN | head -c 8   # first 8 chars only — don't paste full

# Test bearer auth manually:
curl -fsS -H "Authorization: Bearer $DASH_TOKEN" \
  http://127.0.0.1:9000/sessions
```

### Fix
- **`dashboard.token_env` set but env var empty:** Server refuses to
  start (fail-closed). Export the env var in the systemd unit's
  `Environment=` directive.
- **CSRF token mismatch:** The token rotates per server restart unless
  `dashboard.csrf_token_env` pins it. After a restart, the browser's
  cookie is stale — reload the page to get a fresh cookie.
- **403 "writes disabled":** Writes are on by default. If you see this
  after starting `teane web start`, something in `config/config.json`
  set `dashboard.writes_enabled: false` — flip it back to `true`
  (or remove the override entirely).
- **Browser refuses to set cookies on HTTP:** Default
  `SameSite=Strict` cookies work on `http://localhost` but some
  browsers tighten this. Use a real domain + HTTPS via a reverse
  proxy for remote access.

---

## 10. HITL webhook timeout (504 from the dashboard)

### Symptom
- Harness logs `httpx.ReadTimeout` or 504 from the HITL POST.
- Dashboard's pending-HITL panel shows the prompt but the operator
  never clicked.

### Diagnose
```bash
# Check whether the dashboard is reachable from the harness:
curl -fsS http://127.0.0.1:9000/      # adjust host:port
```

### Fix
- Default block is 600 s (10 minutes), set by
  `dashboard.hitl_webhook_timeout_seconds`. After that the harness falls
  back to the next configured channel — `StdinChannel` by default.
  Raise the value in `config/config.json` if your operators routinely
  need longer to respond.
- If the dashboard process restarts while the harness's POST is in
  flight, the connection drops and the harness sees a connection
  reset; the gate falls through to stdin.

---

## 11. `~/.harness/web.db` corrupt — dashboard crashes on startup

### Symptom
- `teane web start` logs `sqlite3.DatabaseError: database disk image
  is malformed`.
- The dashboard UI shows 500 errors on `/run/schedule` or
  `/sessions/<id>/note`.

### Diagnose
```bash
# Check the file:
sqlite3 ~/.harness/web.db "PRAGMA integrity_check;"
```

### Fix
- **Outright corruption:** The file is operator-local and small.
  Delete it; on next dashboard start the schema migration recreates
  empty tables. Audit log + saved presets + queued chat notes are
  lost; runs and schedule state are unaffected.
  ```bash
  teane web stop                                  # or systemctl stop
  mv ~/.harness/web.db ~/.harness/web.db.broken
  teane web start                                 # or systemctl start
  ```
- **Schema mismatch after harness upgrade:** Same fix — wipe and
  recreate. The schema is `CREATE TABLE IF NOT EXISTS` at module
  load, so dropping the file is the supported migration path
  while the schema is stable.

---

## 12. Repo index returns nothing — `teane index status` shows zero chunks

### Symptom
- `teane index status` reports `Chunks: 0` / `No index built yet`.
- Planner doesn't include the "Repository context (semantic
  retrieval)" block even with `repo_index.enabled: true`.

### Diagnose
```bash
# Was the index ever built for this workspace?
sqlite3 ~/.harness/repo_index/repo_index.db \
  "SELECT workspace_id, backend, chunk_count, built_at FROM repo_meta;"

# What does the chunker see?
python -c "
from harness.repo_index import Chunker, RepoIndexConfig
walker = Chunker(RepoIndexConfig())
for p in walker.walk('.'):
    print(p)
" | head -20
```

### Fix
- **Never built:** Run `teane index build -w /path/to/workspace`.
- **Built but the workspace path changed:** The index is keyed by
  workspace path SHA. Re-build it after moving the workspace.
- **`repo_index.enabled: false`:** Flip to `true` in
  `config/config.json` so the planner injects retrieval results.
- **Chunker excludes the files you expect:** Edit `repo_index.*` in
  `config/config.json`. The defaults skip `node_modules`,
  `__pycache__`, `.venv`, `dist`, `build`, `target`, lock files, and
  `.min.js` / `.min.css`.
- **`openai_embeddings` backend, missing API key:** Loader logs a
  WARN and falls back to TF-IDF. Set `OPENAI_API_KEY` or accept the
  TF-IDF backend.

---

## 13. Per-repo memory not loaded — planner doesn't see prior session notes

### Symptom
- `~/.harness/memory/<repo_id>.md` exists but the planner's system
  context doesn't include the "Prior session memory for this
  repository" block.

### Diagnose
```bash
# Confirm the repo identity used:
python -c "
from harness.repo_memory import repo_identity, memory_file_path, RepoMemoryConfig
import os
ws = os.path.abspath('.')
cfg = RepoMemoryConfig()
print('repo_id:', repo_identity(ws))
print('memory file:', memory_file_path(ws, cfg))
print('exists:', os.path.isfile(memory_file_path(ws, cfg)))
"
```

### Fix
- **`memory.enabled: false`:** Flip to `true` in `config/config.json`.
- **Memory file written for a different repo identity:** The identity
  is `SHA256(git remote get-url origin)` when a remote is configured,
  else `SHA256(absolute workspace path)`. Cloning the same repo to a
  different path produces the same identity (good); changing the
  origin URL or working without a remote in a new path produces a
  new identity. Move the old file to the new path-identity hash if
  you want continuity.
- **File grew past `inject_max_bytes`:** Older `## Session ...`
  entries drop from the injection (FIFO trim by section boundary).
  Increase `memory.inject_max_bytes` (default 8000) in
  `config/config.json` if needed. `memory.max_bytes` (default 100000)
  caps the file itself.

---

## 14. Dashboard shows a session as "running" forever (stale badge)

### Symptom

The web dashboard's status / live page keeps showing a session in the
"running" state long after the run subprocess actually exited.
Most common cause: the child was killed with `SIGKILL` (which the
parent's watcher thread can't trap via `proc.wait`), or the dashboard
process itself was hard-restarted.

### Diagnose

```bash
# Look at the registry's view (this is what the UI renders)
curl -fsS -H "Authorization: Bearer $DASH_TOKEN" \
  http://127.0.0.1:9000/live | grep running

# Then check whether the PID actually exists:
ps -p <PID-FROM-UI>
```

### Fix

The dashboard now sweeps the registry for orphan PIDs on every
`/live`, `/status`, and "running for workspace?" lookup (see
`ProcessRegistry._prune_dead_locked`). One refresh of the page after
the child died is enough to flip the badge to terminated.

If a badge persists across multiple refreshes:

1. The dashboard process itself is wedged — `teane web stop` then
   `teane web start`.
2. Or the PID was *recycled* by the kernel between the kill and the
   check (rare but possible on busy hosts). Restarting the dashboard
   clears the registry.

---

## 15. Dashboard marker file corrupt — `teane web start` refuses to launch

### Symptom

`teane web start` exits with "a teane web instance is already
running (pid X, ...)" even though no process is listening on the
port. The marker file at `~/.harness/web.lock` may have a malformed
JSON body, a PID that doesn't exist, or stale ownership from a
previous boot.

The marker has the schema
`{pid, host, port, mode, log_path, started_at}` and is written
atomically (tempfile + os.replace), so partial writes are rare —
but SIGKILL on the dashboard or a host crash mid-shutdown leaves a
stale marker pointing at a now-dead pid.

### Diagnose

```bash
cat ~/.harness/web.lock     # what does the marker think is running?
ps -p $(jq -r .pid ~/.harness/web.lock 2>/dev/null) || echo "stale"
ss -tlnp | grep 9000        # is anything listening on the dashboard port?
```

A stale marker has a PID that doesn't exist; a corrupt marker fails to
parse as JSON.

### Fix

`teane web start` is supposed to auto-clean a stale marker (the
dashboard checks `os.kill(pid, 0)` and treats `ProcessLookupError` as
"prior process gone, marker is junk"). If the auto-clean isn't
happening, remove the marker by hand:

```bash
rm ~/.harness/web.lock
teane web start
```

The marker is a freshness hint, not a critical lock — the dashboard
itself rebinds the listening socket on start, so a stray marker can't
actually prevent a clean boot once removed.

---

## 16. FD-limit pressure when running many dashboard-spawned builds

### Symptom

Long-lived dashboards that spawn many build/patch subprocesses
("Run now" / "Run resume" / scheduled one-shot jobs) eventually start
hitting `OSError: [Errno 24] Too many open files` from inside the
spawn path.

### Diagnose

```bash
# How many FDs is the dashboard process holding?
ls /proc/$(jq -r .pid ~/.harness/web.lock)/fd | wc -l

# And what does the OS allow?
ulimit -n
```

A few hundred is normal. Several thousand suggests an FD leak.

### Fix

The known leak in the spawn path (parent retaining the per-session
stdout sink) was fixed; see `spawn_harness_run` / `spawn_harness_resume`
in `harness/dashboard.py`. If you're still climbing, raise the soft
limit before invoking the dashboard:

```bash
ulimit -n 8192
teane web start --background yes
```

For systemd, set `LimitNOFILE=` in the unit file. The Docker base
image picks the host value at start — bump it on the host or set
`--ulimit nofile=...` on `docker run`.

---

## 17. Startup `ConfigError` — harness exits 2 before any work happens

### Symptom

`teane build`, `patch`, `resume`, `doctor`, `metrics`, `purge`, or any other
subcommand exits immediately with a multi-line error to stderr that
ends with `exit code 2`. No log file, no checkpoint, no LLM call —
the harness refused to start because strict config validation found a
problem.

Common error openings:
- `Canonical config not found at <path>` — the file is missing entirely.
- `Invalid JSON in <path>: ...` — `config/config.json` doesn't parse.
- `Unknown key 'X' at <path>` — a typo (e.g. `token_budget.hrad_cap_usd`).
- `model_routing.planning_primary references unknown model 'Y'` — the
  routing key points at a `models` entry that doesn't exist.
- `Provider 'anthropic' requires env var 'ANTHROPIC_API_KEY' (not set)` —
  a routed provider has no key in env or `models[].api_key`.
- `product_spec_dir not set in config.json` — the mandatory key is
  missing.

### Diagnose

```bash
# Run doctor — its `config` row repeats the same error in its first slot
# and skips every downstream check until config is clean:
teane doctor

# Verify the canonical path the harness resolves to:
python -c "from harness.cli import _get_global_config_path; print(_get_global_config_path())"

# Validate the JSON in isolation:
python -m json.tool <teane_root>/config/config.json > /dev/null
```

### Fix

- **File missing:** re-run `python3 scripts/setup.py` (the bootstrap
  script writes a minimal canonical config) or restore the file from
  git.
- **Typoed key:** the error names the offending key. Fix the
  spelling — the harness knows the canonical key names and won't
  silently no-op.
- **Bad routing reference:** add the model under `models` or change
  the routing key to a model that exists. The shipped config has
  pre-populated entries you can copy-paste.
- **Missing env var:** set the matching `{PROVIDER}_API_KEY` env var.
  As a last resort, populate `models["<key>"].api_key` directly — but
  don't commit live keys to git.
- **Missing `product_spec_dir`:** add a top-level
  `"product_spec_dir": "product_spec"` (or whatever folder name you
  use), create the folder at your workspace root, and drop at least
  one spec file (`.txt`, `.md`, or `.pdf`) in.

The dashboard's Configure Harness page runs the same validator and
shows the same error inline, so non-CLI operators can fix and save
without leaving the browser.

---

## 18. Diagnostics gate silent or flagging the wrong things

### Symptom

Type errors that should have been caught before the compile only
surface at `compiler_node` (gate silent) — or the opposite: the gate
routes to repair for errors that pre-date the session (baseline not
suppressing), burning repair rounds on brownfield code nobody touched.

### Diagnose

```bash
# Every gate pass logs one structured event — status, tool list,
# new-vs-baseline counts, baseline mode, elapsed:
grep '"event": "diagnostics_gate"' ~/.harness/logs/<session>.jsonl | tail -5

# Gate contributes nothing when no checker is on PATH:
which pyright mypy tsc

# Baseline mode matters: "worktree" = exact HEAD diff;
# "created-only" = degraded (non-git workspace or worktree failure) —
# only session-created files can flag, pre-existing files never do.
```

Key fields: `status: "skipped"` → no applicable checker (install one);
`timed_out: ["tsc"]` recurring → large TS project blowing
`diagnostics.timeout_seconds` (raise it, or accept the gate skipping
TS — fail-open by design); `baseline: 0` with `new` unexpectedly high
on brownfield → baseline capture failed, check the log for
`diag_baseline` worktree errors.

### Fix

- **Gate silent:** install `pyright` (`pip install pyright`) and/or
  `typescript` (`npm i -g typescript`). Per-tool kill switches live in
  `config.json` `diagnostics.tools`.
- **Pre-existing errors flagged:** confirm the workspace is a git repo
  with a clean HEAD to baseline against; `git worktree list` should be
  empty of stale `diag_baseline_*` entries (`git worktree prune`).
- **Gate → repair ping-pong:** bounded by `diagnostics.max_rounds`
  (default 2) on top of the shared repair cap — if you see more than
  `max_rounds` consecutive `diagnostics_node → repair_node` transitions
  in the `node_transition` events, that's a bug, file it.
- **Disable entirely:** `"diagnostics": {"enabled": false}`.

---

## 19. Learned rule is wrong — post-mortem note poisoning the planner

### Symptom

After a failed session, every subsequent run on the same repo plans
around a `[learned-rule:<trigger>]` hypothesis that is wrong or stale
(visible in the planner context under "Prior session memory for this
repository").

### Diagnose

```bash
# Active + retired rules for this repo (16-char id from the workspace):
grep -n "learned-rule" ~/.harness/memory/<repo_id>.md

# Which sessions wrote / skipped / retired rules:
grep -E '"event": "(post_mortem_written|post_mortem_skipped|post_mortem_rules_retired)"' \
  ~/.harness/logs/*.jsonl
```

### Fix

- **Normal case — no action:** any clean (exit-0) run on the repo
  retires ALL active rules automatically
  (`post_mortem.retire_on_clean_run`, default true). One green run and
  the poison is gone.
- **Immediate manual removal:** edit
  `~/.harness/memory/<repo_id>.md` and delete the `- Notes:` line
  carrying the bad rule (or rewrite its tag to
  `[learned-rule(retired):...]` — retired rules are kept for forensics
  but treated as inactive).
- **Stop generating rules:** `"post_mortem": {"enabled": false}`.
- Rules are deliberately framed as "Hypothesis from failed run <sid>"
  and capped at one line — if you see multi-line or heading-bearing
  notes, the sanitizer has a bug; file it.

---

## 20. LSP pool didn't start on a brownfield run

### Symptom

`teane patch` runs fine but navigation stays on heuristics: no
`lsp__find_references` section in the planner prompt, `lsp_fallback`
events in the log, or the startup log says
`[cli:lsp] no servers started`.

### Diagnose

```bash
# The start event lists what launched and WHY anything was skipped:
grep '"event": "lsp_pool_started"' ~/.harness/logs/<session>.jsonl
# skipped reasons you'll see:
#   "pyright-langserver not on PATH"      → install it
#   "no .venv/venv at workspace root..."  → env-health probe refused
#   "no node_modules at workspace root"   → npm install first

# Binaries present?
which pyright-langserver typescript-language-server

# Remember the gate conditions: pool ONLY starts when
#   flow != "build"  AND  flow ∈ lsp.enabled_flows (default patch/test)
#   AND the workspace stack tags include python/typescript/node.
```

### Fix

- **Probe refused Python:** create the venv the repo's deps live in
  (`python -m venv .venv && .venv/bin/pip install -r requirements.txt`)
  — the probe exists because a server without resolvable imports
  returns unresolved-import garbage. If your deps are system-wide, set
  `"lsp": {"python_require_venv": false}`.
- **Probe refused TS:** run `npm install` in the workspace so
  `node_modules` exists; ensure a root `tsconfig.json`.
- **Server died mid-session:** expected degradation — one warning log,
  then every consumer falls back to the DependencyGraph for the rest
  of the session (no auto-restart in Phase 1). Re-run the session to
  get a fresh pool.
- **Wrong expectations on greenfield:** by design `teane build` never
  starts the pool; greenfield half-built projects make LSP results
  worse than the heuristics.
- **Disable entirely:** `"lsp": {"enabled": false}`.

---

## Appendix: Useful one-liners

```bash
# List all checkpointed sessions:
teane status --all

# Inspect a session without resuming:
teane status --session-id <id>

# Tail a live session's structured log:
tail -f ~/.harness/logs/<session-id>.jsonl | jq -c

# Total spend by session (last 30 days):
for f in ~/.harness/logs/*.jsonl; do
  echo "$(basename "$f" .jsonl): $(grep llm_call "$f" |
    jq -s '[.[].cost_usd] | add // 0')"
done | sort -t: -k2 -n -r | head

# Force-purge all log files older than 30 days:
find ~/.harness/logs -name '*.jsonl*' -mtime +30 -delete

# Tail the dashboard's audit log (who saved what config when):
sqlite3 -separator ' | ' ~/.harness/web.db \
  "SELECT ts, action, target FROM audit_log ORDER BY id DESC LIMIT 30;"

# Pending one-shot jobs the dashboard enqueued:
sqlite3 -header -column ~/.harness/web.db \
  "SELECT id, name, fire_at_utc, workspace FROM web_oneshot_jobs
   WHERE consumed_at IS NULL ORDER BY fire_at_utc;"

# Schedule-daemon history for one job:
sqlite3 -header -column ~/.harness/schedule.db \
  "SELECT started_at, ended_at, exit_code, duration_sec
   FROM schedule_runs WHERE job_name = 'JOB-NAME-HERE'
   ORDER BY started_at DESC LIMIT 20;"

# Repo index summary for every workspace ever indexed:
sqlite3 -header -column ~/.harness/repo_index/repo_index.db \
  "SELECT workspace_id, backend, chunk_count, built_at FROM repo_meta;"

# List MCP servers' advertised tools (post-doctor):
teane doctor 2>&1 | grep '^mcp:'

# Drift events (prompt cache misses) in the most recent session:
last_log=$(ls -t ~/.harness/logs/*.jsonl | head -1)
grep cache_prefix_drift "$last_log" | jq -c '{role, prev_hash, now_hash}'

# Currently-pending HITL prompts (dashboard process must be alive):
curl -fsS -H "Authorization: Bearer $DASH_TOKEN" \
  http://127.0.0.1:9000/sessions/SESSION-ID-HERE/hitl/pending | jq .

# Live SSE event stream for one session (Ctrl-C to stop):
curl -fsS -N -H "Authorization: Bearer $DASH_TOKEN" \
  http://127.0.0.1:9000/api/sessions/SESSION-ID-HERE/events

# Inspect the canonical config without loading it:
jq 'del(.. | .api_key?)' <teane_root>/config/config.json | less

# Clear harness-owned Docker cache volumes (sandbox.cache_volumes=true):
teane cache clear --dry-run        # preview
teane cache clear --session-id <id>
```

## Escalation

If none of the above resolves the issue:

1. Capture the session log (`~/.harness/logs/<session-id>.jsonl`) and the
   `teane doctor` output.
2. Capture the harness version: `teane --version`.
3. Open an issue at the project tracker with all three.
