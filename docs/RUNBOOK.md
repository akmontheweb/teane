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
```

## Escalation

If none of the above resolves the issue:

1. Capture the session log (`~/.harness/logs/<session-id>.jsonl`) and the
   `harness doctor` output.
2. Capture the harness version: `harness --version`.
3. Open an issue at the project tracker with all three.
