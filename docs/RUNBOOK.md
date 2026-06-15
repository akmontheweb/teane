# Harness Operations Runbook

Self-serve recovery for the failure modes operators hit most often. Each
entry has a one-line symptom, a diagnostic command that confirms the
cause, and a fix recipe with explicit commands.

When in doubt, **run `harness doctor` first** — it executes six healthchecks
(git repo, global config, API keys, sandbox backend, checkpoint DB, config
parse) and prints a colored summary pointing at the broken subsystem.

```bash
harness doctor -r /path/to/workspace
```

If `doctor` is green and you're still stuck, the entries below cover the
five failure modes that have actually caused operator pain. They are
ordered by frequency, not severity.

---

## 1. Checkpoint corrupted — `harness resume` refuses to load

**Symptom**

```
[resume] Checkpoint for session '<id>' is corrupted: ...
  Options:
    - Start a fresh session with `harness run -r <ws> -p '<prompt>'`.
    - Restore checkpoints.db from a known-good backup.
    - Run `harness purge --session-id <id>` to drop only this session.
```

**Diagnose**

```bash
# Confirm which session(s) have unreadable blobs without altering them.
harness doctor
# Look for the "checkpoint db" check — it scans the 5 most recent rows.

# Inspect the offending session non-destructively:
harness status --session-id <id>
```

**Fix**

Choose one of three paths, in order of preference:

1. **Restore from backup** (preferred when a recent backup exists):
   ```bash
   cp ~/.harness/checkpoints.db.bak ~/.harness/checkpoints.db
   harness resume --session-id <id>
   ```
2. **Drop only the broken session, keep all others**:
   ```bash
   harness purge --session-id <id>
   harness run -r <ws> -p "<original prompt>"   # start fresh
   ```
3. **Last resort, nuke everything**:
   ```bash
   harness purge --all
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
harness metrics --session-id <id>

# Roll-up across every session in the log dir:
harness metrics --all

# Legacy fallback (when the harness binary itself is the problem):
grep '"event": "llm_call"' ~/.harness/logs/<session-id>.jsonl | \
  jq -s 'map({model, cost_usd, tokens_in, tokens_out}) |
         group_by(.model) |
         map({model: .[0].model, total_cost: (map(.cost_usd) | add), calls: length})'
```

The `harness metrics` output shows total cost, per-window burn rate,
and an estimated minutes-until-exhaustion at the current rate. The
legacy jq recipe still works and is useful when the CLI itself is the
thing that's broken.

**Fix**

- **Raise the cap and resume** (most common):
  ```bash
  # Edit ~/.harness_config.json (workspace) or ~/.harness/config.json (user-global):
  #   "token_budget": { "hard_cap_usd": 5.00 }
  harness resume --session-id <id>
  ```
- **Re-route an expensive node to a cheaper model**: edit `model_routing.*`
  in the same config to point a hot node (e.g. `code_reviewer_primary`)
  at a smaller model.
- **Force local Ollama for the rest of the session**: set
  `model_routing.force_local_only: true` and resume.

**Why it happens.** Discovery loops, doc reviews, and code reviews can each
take 3–5 LLM round-trips. A complex workspace with all three active will
land in the $1–$3 range. The default `hard_cap_usd: 2.00` is a guardrail,
not a target.

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
harness doctor
# Look at the "sandbox backend" check — it prints the backend in use
# and whether the binary/daemon is reachable.

# Manual probes:
docker info       # for docker backend
podman info       # for podman backend
which firejail    # for firejail backend
```

**Fix**

- **Daemon not running** (Linux):
  ```bash
  sudo systemctl start docker
  sudo usermod -aG docker $USER && newgrp docker
  ```
- **Wrong backend selected** — edit `sandbox.backend` in
  `~/.harness_config.json`. Valid values: `docker`, `podman`, `firejail`,
  `none` (no sandbox, host execution — use only when isolation isn't
  needed, e.g. CI).
- **Image missing** — let the harness pull on first run, or pre-pull:
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
  harness run -r <ws> -p "<prompt>" --force-lock
  ```
  `--force-lock` releases the stale lock and acquires a fresh one. It
  logs a WARNING so the override is visible in the session record.

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
  # In ~/.harness/config.json, point the affected node at a different model:
  #   "model_routing": { "planning_primary": "anthropic:claude-sonnet-4-6" }
  harness resume --session-id <id>
  ```
- **API key revoked, out of credit, or wrong model id** — `harness doctor`
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

## 6. MCP server fails to start (`harness doctor` shows `fail`)

### Symptom
- `harness doctor` reports `mcp:<server>: command rejected: ...` or
  `mcp:<server>: start failed: ...`.
- Planner emits `<<<MCP_CALL>>>` blocks; the tool result body is
  `{"error": "mcp server 'X' not registered..."}`.

### Diagnose
```bash
# Print the resolved server commands the harness will run:
python -c "
from harness.mcp_client import McpPoolConfig
from harness.cli import discover_config
cfg = McpPoolConfig.from_config(discover_config('.'))
for s in cfg.servers:
    print(s.name, '->', s.command)
"

# Manually start the server to see its stderr:
npx -y @modelcontextprotocol/server-time   # adjust to your config
```

### Fix
- **Command not in allowlist:** Add the binary basename to
  `mcp.command_allowlist` (the built-in allowlist covers `npx`, `npm`,
  `node`, `python`, `python3`, `uvx`, `pipx`, `docker`).
- **Filesystem server rejected:** Set
  `mcp.allow_local_filesystem_servers: true` if you've reviewed the
  blast radius — filesystem MCP gives the LLM raw host I/O.
- **`npx` hangs on first launch:** Pre-install once outside the
  harness so the package is cached: `npx -y @scope/server-name --help`.
- **Server crashes with `MODULE_NOT_FOUND`:** Check Node.js version
  meets the server's `engines` requirement.

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
  `llm_dispatch.prompt_cache_enabled: false` in `config.json`.

---

## 8. Schedule daemon stuck — jobs not firing

### Symptom
- `harness schedule list` shows a job that should have fired hours ago.
- `~/.harness/schedule.db` has no `schedule_runs` row for the expected
  fire time.

### Diagnose
```bash
# Is the daemon actually running?
ps aux | grep "harness schedule run" | grep -v grep
systemctl status harness-schedule.service   # under systemd

# What does the daemon think the next fire times are?
harness schedule list

# Validate the cron syntax:
harness schedule validate
```

### Fix
- **Daemon not running:** Start it. The daemon does not auto-launch on
  `harness run`; it's a separate process.
- **Job marked `enabled: false`:** Flip to `true` in `config.json` and
  restart the daemon.
- **Cron syntax silently fell through:** `harness schedule validate`
  surfaces the rejection. Common mistakes: `daily 2:30` (must be
  `02:30`); `weekly monday 03:00` (must be `mon`); full POSIX cron
  like `30 2 * * mon` (use the supported subset).
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
ss -tlnp | grep 8729       # default port

# What token does the dashboard expect?
echo $DASH_TOKEN | head -c 8   # first 8 chars only — don't paste full

# Test bearer auth manually:
curl -fsS -H "Authorization: Bearer $DASH_TOKEN" \
  http://127.0.0.1:8729/sessions
```

### Fix
- **`dashboard.token_env` set but env var empty:** Server refuses to
  start (fail-closed). Export the env var in the systemd unit's
  `Environment=` directive.
- **CSRF token mismatch:** The token rotates per server restart unless
  `dashboard.csrf_token_env` pins it. After a restart, the browser's
  cookie is stale — reload the page to get a fresh cookie.
- **403 "writes disabled":** Launch with `--writes-enabled` OR set
  `dashboard.writes_enabled: true` in `config.json`. The flag also
  matters for "Run from web" and HITL gates.
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
curl -fsS http://127.0.0.1:8729/      # adjust host:port
```

### Fix
- Default block is 600 s (10 minutes). After that the harness falls
  back to the next configured channel — `StdinChannel` by default.
- Tell the operator to answer faster, OR raise the dashboard's
  internal block timeout (currently hardcoded in
  `harness/dashboard.py:_handle_hitl_webhook`).
- If the dashboard process restarts while the harness's POST is in
  flight, the connection drops and the harness sees a connection
  reset; the gate falls through to stdin.

---

## 11. `~/.harness/web.db` corrupt — dashboard crashes on startup

### Symptom
- `harness dashboard` logs `sqlite3.DatabaseError: database disk image
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
  systemctl stop harness-dashboard.service   # or kill the process
  mv ~/.harness/web.db ~/.harness/web.db.broken
  systemctl start harness-dashboard.service
  ```
- **Schema mismatch after harness upgrade:** Same fix — wipe and
  recreate. The schema is `CREATE TABLE IF NOT EXISTS` at module
  load, so dropping the file is the supported migration path
  while the schema is stable.

---

## 12. Repo index returns nothing — `harness index status` shows zero chunks

### Symptom
- `harness index status` reports `Chunks: 0` / `No index built yet`.
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
- **Never built:** Run `harness index build -r /path/to/workspace`.
- **Built but the workspace path changed:** The index is keyed by
  workspace path SHA. Re-build it after moving the workspace.
- **Chunker excludes the files you expect:** Edit
  `repo_index.exclude_globs` / `repo_index.text_extensions` in
  `config.json`. The default skips `node_modules`, `__pycache__`,
  `.venv`, `dist`, `build`, `target`, lock files, and `.min.js` /
  `.min.css`.
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
- **`memory.enabled: false`:** Flip to `true` in `config.json`.
- **Memory file written for a different repo identity:** The identity
  is `SHA256(git remote get-url origin)` when a remote is configured,
  else `SHA256(absolute workspace path)`. Cloning the same repo to a
  different path produces the same identity (good); changing the
  origin URL or working without a remote in a new path produces a
  new identity. Move the old file to the new path-identity hash if
  you want continuity.
- **File grew past `inject_max_bytes`:** Older `## Session ...`
  entries drop from the injection (FIFO trim by section boundary).
  Increase `memory.inject_max_bytes` if needed.

---

## Appendix: Useful one-liners

```bash
# List all checkpointed sessions:
harness status --all

# Inspect a session without resuming:
harness status --session-id <id>

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
harness doctor 2>&1 | grep '^mcp:'

# Drift events (prompt cache misses) in the most recent session:
last_log=$(ls -t ~/.harness/logs/*.jsonl | head -1)
grep cache_prefix_drift "$last_log" | jq -c '{role, prev_hash, now_hash}'

# Currently-pending HITL prompts (dashboard process must be alive):
curl -fsS -H "Authorization: Bearer $DASH_TOKEN" \
  http://127.0.0.1:8729/sessions/SESSION-ID-HERE/hitl/pending | jq .

# Live SSE event stream for one session (Ctrl-C to stop):
curl -fsS -N -H "Authorization: Bearer $DASH_TOKEN" \
  http://127.0.0.1:8729/api/sessions/SESSION-ID-HERE/events
```

## Escalation

If none of the above resolves the issue:

1. Capture the session log (`~/.harness/logs/<session-id>.jsonl`) and the
   `harness doctor` output.
2. Capture the harness version: `harness --version`.
3. Open an issue at the project tracker with all three.
