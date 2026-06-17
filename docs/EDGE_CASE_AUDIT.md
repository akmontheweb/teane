# myharness — Edge-Case Audit

Read-only audit of the harness for failure-prone paths. Findings are grouped by concern area and ranked by severity (**Critical / High / Medium / Low**). Each entry cites a concrete file:line, the failure scenario, and a recommended guard. **No code has been changed** — these are recommendations only, pending your review.

Severity legend:
- **Critical** — known data-loss, security breach, or unrecoverable corruption path
- **High** — silent wrong behaviour, resource exhaustion, or hard-to-diagnose hang
- **Medium** — degraded behaviour, leaks under sustained load, or rare-but-real edge case
- **Low** — defensive hardening, code-clarity, or future-proofing

---

## Table of Contents

1. [Concurrency & Race Conditions](#1-concurrency--race-conditions)
2. [Subprocess / Process Lifecycle / Resource Leaks](#2-subprocess--process-lifecycle--resource-leaks)
3. [Security & Input Validation](#3-security--input-validation)
4. [LLM Gateway & External Network](#4-llm-gateway--external-network)
5. [CLI / Config / Persistence](#5-cli--config--persistence)
6. [Patcher / Graph State Machine](#6-patcher--graph-state-machine)
7. [Cross-Cutting Themes](#7-cross-cutting-themes)
8. [Suggested Prioritisation](#8-suggested-prioritisation)

---

## 1. Concurrency & Race Conditions

### 1.1 [Critical] Web one-shot jobs can fire twice — read/mark-consumed TOCTOU
**Files:** `harness/schedule.py:619-678`; `harness/web_state.py:490-535`

`tick_once()` → `_due_oneshots()` runs a plain `SELECT … WHERE consumed_at IS NULL`. The matching `mark_oneshot_consumed` only runs **after** `execute_job_once` completes (minutes/hours later). A second daemon poll, a `harness schedule once` invocation, or a crash-restart in that window re-reads the same row and re-fires the job.

**Recommendation:** Convert the read+claim into a single transactional `UPDATE web_oneshot_jobs SET consumed_at=? WHERE id=? AND consumed_at IS NULL` with `RETURNING`; if `rowcount==0`, another worker won the claim — skip.

### 1.2 [Critical] `cancel_session` PID-reuse race may SIGTERM the wrong process group
**File:** `harness/dashboard.py:4299-4311` (and `harness/schedule.py:475` for the daemon path)

```python
os.killpg(os.getpgid(entry.pid), _signal.SIGTERM)
```
`entry.pid` is a stored integer. Between `is_running` and `getpgid`, the kernel can recycle the PID to an unrelated process — `killpg` then signals an editor, another harness, or any unrelated process group.

**Recommendation:** Hold the registry lock during the kill; double-check `proc.returncode is None` on the live `asyncio.Process` handle; capture `pgid` immediately after spawn (`start_new_session=True` guarantees `pid == pgid`) and store it alongside the `WebProcess` entry.

### 1.3 [Critical] Storage redaction wrapper persists **unredacted** messages on any exception
**File:** `harness/storage.py:267-306`

`aput` / `aput_writes` try-blocks (lines 271-281, 291-306) catch errors with `logger.warning(...)` then **fall through** to `super().aput(...)` with the **original** checkpoint. Any exception inside the redactor (unusual checkpoint shape, frozen dict, custom AgentState) silently writes plaintext that may contain operator-pasted API keys / PII.

**Recommendation:** On exception, substitute a synthetic safe-list (or raise). Losing one checkpoint is better than leaking secrets to disk.

### 1.4 [Critical] `_execute_subprocess_with_timeout` leaks process + temp files on asyncio cancellation
**File:** `harness/sandbox.py:990-1083`

The outer `try:` does not catch `asyncio.CancelledError` (a `BaseException` since 3.8). When Ctrl-C / parent-cancellation lands while a sandbox build runs, the subprocess, its log temp files in `/tmp/.harness/`, and the open pipes all leak. Repeated cancelled runs accumulate fd/disk debt; with docker/unshare backends, namespaces and containers leak too.

**Recommendation:** Wrap the body in `try:` … `finally:` cleanup; catch `CancelledError` explicitly, kill the process group, close streamers, then re-raise.

### 1.5 [High] Schedule daemon `_in_flight` desyncs from reality on crash
**File:** `harness/schedule.py:551-612`

`_in_flight` is an in-memory `set[str]` reset to empty on every `__init__`. Subprocesses are launched with `start_new_session=True` so they survive a daemon crash. On restart, `_in_flight` is empty and `schedule_runs` has no `still-running` marker — the next tick can fire the same job again while the original is still running.

**Recommendation:** Persist a `pid` column in `schedule_runs`, NULLed at finish. On boot, probe still-running rows with `_pid_alive`; either reattach or treat as orphan and skip the next firing.

### 1.6 [High] `consume_chat_notes` allows duplicate consumption under concurrent readers
**File:** `harness/web_state.py:408-431`

Default SQLite isolation gives the SELECT a SHARED lock. Two readers (dashboard handler + HITL webhook handler) can both SELECT the same rows, both UPDATE, both return them.

**Recommendation:** `conn.execute("BEGIN IMMEDIATE")` before SELECT, or use `UPDATE … RETURNING` (sqlite ≥ 3.35).

### 1.7 [High] SSE generator threads leak when registry TTL drops the entry
**Files:** `harness/dashboard.py:4318-4415`; `harness/web_state.py:258-268`

`tail_session_events` exits only when it finds a terminated registry entry. `_prune_expired_locked` deletes entries 5 min after `terminated_at`. An operator who leaves the console tab open >5 min after the run ends has the SSE thread spin forever — one thread per stale tab, with `ThreadingMixIn`.

**Recommendation:** Retain a `(session_id, log_path)` mapping past TTL, or give up after N idle polls. Switch to inotify (`asyncio.add_reader`). Periodically write `:keepalive\n\n` to detect dead clients.

### 1.8 [High] `fanout._run_one` leaks budget reservation on cancellation / odd response shapes
**File:** `harness/fanout.py:153-216`

`actual_cost = float(getattr(response.usage, "cost_usd", 0.0) or 0.0)` (line 201) is outside any except block. A missing `.usage`, a non-numeric cost, or `asyncio.CancelledError` from `wait_for` leaks the reservation — future fanouts wrongly reject agents with "shared budget exhausted".

**Recommendation:** Refund inside `finally:`. Catch `CancelledError` explicitly, refund, re-raise.

### 1.9 [Medium] `_acquire_workspace_lock` truncates the lock file before `flock`
**File:** `harness/cli.py:182-228`

`open(lock_path, "w")` truncates *before* `fcntl.flock`. A second process's truncation wipes the holder's diagnostic PID line. Also: the module-level `_WORKSPACE_LOCK_HANDLE = fh` is overwritten when called twice in-process — the first lock's handle becomes GC-eligible and the kernel quietly releases the lock.

**Recommendation:** Open with `O_RDWR|O_CREAT` (no truncate), flock first, then write PID. Store handles in a dict keyed by workspace path, never overwrite.

### 1.10 [Medium] Dashboard `has_running_for_workspace` → `spawn_harness_run` TOCTOU
**File:** `harness/dashboard.py:3281-3309`

Two concurrent `POST /run/now` for the same workspace both pass the check and both spawn. The second child crashes at its own workspace-lock with `exit 1`, but only after spinning up — wasteful and confusing in the UI.

**Recommendation:** Hold a `threading.Lock` around (check, spawn) in the handler.

### 1.11 [Medium] `web.db` / `schedule.db` not in WAL mode — SQLITE_BUSY under contention
**Files:** `harness/web_state.py:99-112`; `harness/schedule.py:336-341`

`storage.py:126-155` sets WAL + `busy_timeout` for the checkpoint DB but `open_web_db` and `_open_history` do neither. Concurrent writers (dashboard `append_audit` while the schedule daemon does `record_run_started`) get `OperationalError: database is locked`. No retry path — request becomes a 500.

**Recommendation:** Execute `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000;` on every connection open.

### 1.12 [Medium] `_kill_process_group`'s `time.sleep(3.0)` blocks the asyncio loop
**File:** `harness/sandbox.py:1368-1398`

Called from the async timeout branch; synchronous sleep freezes every other coroutine for 3 s. With the subsequent 5 s `proc.wait()`, the loop can be pinned for 8 s+ per cancel.

**Recommendation:** Convert `_kill_process_group` to `async def` and use `await asyncio.sleep(3.0)`.

### 1.13 [Medium] MCP client `_pending` future cleanup race on cancellation
**File:** `harness/mcp_client.py:320-393`

Caller cancelled while `_pending[req_id]` is set → future leaks until shutdown. During shutdown, `set_exception` on a future the reader concurrently `set_result`'d raises `InvalidStateError`.

**Recommendation:** In `_call`'s `finally:`, cancel the future if not done. In `shutdown`, snapshot-and-clear before iterating.

### 1.14 [Medium] `repo_memory.append_session_note` is not atomic across concurrent appenders
**File:** `harness/repo_memory.py:193-275`

Read-modify-write: two appenders can both read the same `prior`, both rewrite to the same `<path>.tmp`, both `os.replace` — last writer wins, the other's section is lost. `os.replace` is atomic for readers, not writers.

**Recommendation:** Use a unique tmp name (`<path>.<pid>.<uuid>.tmp`); hold `fcntl.flock(path, LOCK_EX)` for the whole RMW; or switch the new section to `O_APPEND` write and run FIFO-trimming as a separate maintenance pass.

### 1.15 [Medium] `HitlQueue.answer` can lose responses on timeout race
**Files:** `harness/web_state.py:309-378`; `harness/dashboard.py:3729-3754`

If operator answers in the exact ms the handler's wait times out, `clear_pending` removes the entry before `answer` lands → handler uses default response, operator UI thinks it was accepted.

**Recommendation:** `clear_pending` must check `entry.event.is_set()` (or a "claimed" flag) and refuse to clear if the answer landed.

### 1.16 [Low] Schedule history `INSERT OR REPLACE` clobbers same-second runs
**File:** `harness/schedule.py:344-356`

PK is `(job_name, started_at)`. Two runs in the same second collapse — the first becomes invisible.

**Recommendation:** Add `INTEGER PRIMARY KEY AUTOINCREMENT`; use the pair as a UNIQUE index only.

### 1.17 [Low] Lazy-init globals double-instantiate under `ThreadingMixIn`
**File:** `harness/dashboard.py:3861-3884`

`get_process_registry` / `get_hitl_queue` lazy init has no lock — two simultaneous threads on first call each create a registry. The losing thread's `register()` calls vanish.

**Recommendation:** Eagerly initialise in `start_server`, or guard with a module-level lock.

---

## 2. Subprocess / Process Lifecycle / Resource Leaks

### 2.1 [Critical] Schedule daemon cancellation orphans in-flight `harness run` subprocess
**File:** `harness/schedule.py:681-699`

On SIGTERM, `run_forever`'s `CancelledError` branch returns 0 without killing the child. The harness subprocess (own session) keeps running indefinitely as an orphan. Repeated stop/start cycles accumulate orphans.

**Recommendation:** Add a `try: … finally:` shutdown that drains `_in_flight`: SIGTERM each child's pgid, wait 5 s, SIGKILL on timeout, then return.

### 2.2 [Critical] `cancel_session` sends SIGTERM only — no SIGKILL escalation
**File:** `harness/dashboard.py:4299-4311`

A harness child blocked in a synchronous LLM HTTP call or stuck in HITL wait can ignore SIGTERM. `_handle_session_purge` then proceeds to delete data while the live process recreates it — half-purged sessions appear. (`cmd_web_stop` at `cli.py:5872-5905` already has the correct pattern.)

**Recommendation:** Escalate to SIGKILL after a 5 s grace; refuse to proceed with purge if the child is still alive.

### 2.3 [High] All `docker compose` calls leak the subprocess on timeout
**File:** `harness/deploy.py:1108-1130, 1138-1149, 1209-1217, 1369-1394, 1441-1448`

Every site does `await asyncio.wait_for(proc.communicate(), timeout=...)`. On `TimeoutError`, `wait_for` cancels the inner task — **but does not kill the child**. The compose `up --build -d` keeps running in the background; the operator never knows.

**Recommendation:** Standard pattern: `try: await wait_for(communicate(), …); except TimeoutError: proc.kill(); await proc.wait(); raise`.

### 2.4 [High] `docker compose` invoked with no `-p` — same-basename workspaces collide
**File:** `harness/deploy.py` (`_compose_argv()`)

No `-p` and no `COMPOSE_PROJECT_NAME`. Two workspaces both named `app` share a Docker project namespace; `down --remove-orphans` from one removes the other's containers; concurrent `up`s race.

**Recommendation:** Derive a stable project name from `hashlib.sha256(os.path.realpath(workspace_path).encode()).hexdigest()[:12]`; pass via `-p`.

### 2.5 [High] `_run_hook` leaks the `/bin/sh -c <hook>` tree on timeout
**File:** `harness/schedule.py:509-544`

A wedged hook (curl to an unreachable host) leaks one subprocess + two pipe FDs **per job fire**. On a 1-minute tick, fd exhaustion within an hour. No `start_new_session=True` either, so killing the parent wouldn't catch children.

**Recommendation:** Add `start_new_session=True`; on TimeoutError, `os.killpg(os.getpgid(proc.pid), SIGTERM)` then SIGKILL after grace.

### 2.6 [High] Lintgate / formatter subprocesses leak on timeout
**File:** `harness/lintgate.py:556-561, 615-620, 648-651`

Same pattern. One file × one lint pass × one repair iteration leaks one interpreter on timeout. Repair can loop dozens of times.

**Recommendation:** Same kill-on-timeout pattern.

### 2.7 [High] Manifest tempfile never deleted
**File:** `harness/cli.py:3955-3964`

`tempfile.mkstemp` for the consolidated spec; no `os.unlink` anywhere. Every greenfield `harness run` leaks one `/tmp/harness_spec_*.txt` containing the full product spec (potentially proprietary / secret).

**Recommendation:** Wrap in `try: … finally: os.unlink(manifest_path)`; or use `NamedTemporaryFile(delete=True)` if synthesise_requirements accepts a file object.

### 2.8 [High] MCP atexit cleanup is in `cli.py`, not the library
**File:** `harness/mcp_client.py:33-35` (docstring) vs `harness/cli.py:3467`

Module docstring claims an `atexit` hook in `McpClientPool.shutdown`. The actual hook lives in `cli.py:3443-3467` and only fires when `_register_mcp_pool` ran. Embedded / test consumers that instantiate `McpClientPool` directly get no cleanup — an uncaught exception leaks every MCP subprocess.

**Recommendation:** Move `atexit.register` into `McpClientPool.__init__`, or fix the docstring.

### 2.9 [Medium] `DiskLogStreamer` temp files lingered indefinitely on `keep_on_success`
**File:** `harness/sandbox.py:1146-1192, 1342-1348`

`NamedTemporaryFile(delete=False)` log files; on SIGKILL or exception outside the caught set, they never get cleaned. Successful builds (default) intentionally retain them with **no aging mechanism** — `/tmp/.harness/harness_*.std{out,err}.log` grows forever.

**Recommendation:** Boot-time janitor pass: delete files older than N days. Or use `delete=True` + duplicate-fd trick when retention is not requested.

### 2.10 [Medium] `_host_side_ownership_sweep` blocks the event loop for up to 30 s
**File:** `harness/sandbox.py:799-833`

Synchronous `subprocess.run(find …, timeout=30)` runs in the async `finally` block. Large workspaces with many root-owned files stall the loop on every docker build (success or failure).

**Recommendation:** Wrap in `loop.run_in_executor`.

### 2.11 [Medium] Dashboard launches `proc = Popen(...)` with `stdout_fh` open; on Popen failure, empty log file lingers
**File:** `harness/dashboard.py:4159-4169, 4254-4264`

`open(log_path + ".stdout", "ab")` runs before `Popen`. If Popen raises, the file stays. Slow accumulation in the log dir.

**Recommendation:** `os.unlink` on the Popen failure path.

### 2.12 [Medium] Web stop deletes marker **before** killing — runaway server becomes unrecoverable
**File:** `harness/cli.py:5866 vs 5872`

`_delete_web_marker()` runs first. On `OSError` (EPERM — pid alive but not owned by current uid), the marker is gone but the server still runs. Future `harness web stop` reports "no server running".

**Recommendation:** Delete the marker **after** confirming exit, or only on `ProcessLookupError`.

### 2.13 [Medium] `_run_docker_inspect` swallows `Exception`; on timeout the inspect subprocess leaks silently
**File:** `harness/deploy.py:1108-1130`

`health_check_loop` calls this every ~2 s per service. A docker daemon lock-up turns into one zombie inspect per service per tick — explosive leak with no log line.

**Recommendation:** Explicit `TimeoutError` branch that kills the proc; structured warning so operator sees it.

### 2.14 [Medium] SQLite connection leaks in `_open_history` / `open_web_db` if `executescript` raises
**Files:** `harness/schedule.py:336-376`; `harness/web_state.py:99-112`

`connect()` succeeds, then `executescript(SCHEMA_SQL)` raises (disk full, corrupt DB, migration bug) → connection leaked.

**Recommendation:** Wrap in `try: … except: conn.close(); raise`.

### 2.15 [Medium] Watcher thread races with `_prune_dead_locked` and overwrites the real exit code with -1
**File:** `harness/dashboard.py:4178-4192`

Between subprocess death and `_watch` thread scheduling, `_prune_dead_locked` sees `not _pid_alive(p.pid)`, sets `exit_code = -1`. Watcher then overwrites back to the real code, but the UI shows -1 in the meantime.

**Recommendation:** Skip prune for entries with a watcher pending; always defer to `mark_terminated`.

### 2.16 [Low] `gh pr create` passes large bodies via argv → ARG_MAX (≈128 KB on Linux)
**File:** `harness/github_integration.py:280-311`

Autogenerated security-review summaries can exceed the limit and produce a cryptic E2BIG.

**Recommendation:** Use `--body-file -` and pass body via stdin pipe for large content.

---

## 3. Security & Input Validation

### 3.1 [Critical] SSRF via redirect — `httpx` `follow_redirects=True` only validates the original URL
**File:** `harness/web_tools.py:265-272, 344-350`

`validate_outbound_url(url, …)` runs once, then `client.get(url)` follows 3xx to wherever. An attacker-controlled public host can redirect to `http://169.254.169.254/latest/meta-data/iam/...` and the harness fetches it.

**Recommendation:** Either `follow_redirects=False` (handle manually with re-validation per hop), or use an `httpx.Client` event-hook that re-validates each redirect target.

### 3.2 [Critical] SSRF via DNS rebinding — hostnames never resolved before validation
**File:** `harness/trust.py:444-460, 500-510`

`_ip_in_private_range(host)` returns False for any hostname. `evil.com` with an A record pointing at `10.0.0.1` or `169.254.169.254` sails through. Combined with redirect-follow above, `web_fetch` becomes a cloud-metadata read primitive.

**Recommendation:** Pre-resolve via `socket.getaddrinfo` (all families); reject if any resolved IP is private/link-local/loopback; pin the socket to that IP for the actual request.

### 3.3 [High] `CommandValidator` whitelist allows `sh`/`bash`/`env` and short-circuits the loop → full validator bypass
**File:** `harness/security.py:460, 591`

```python
if base_cmd in ("", "sh", "bash", ".", "source", "export", "env", "exec"):
    continue
```
`bash -c 'cat /etc/shadow'` is one segment; base_cmd is `bash`; the inner command is never inspected.

**Recommendation:** Remove `sh`/`bash`/`dash` from `DEFAULT_ALLOWED_COMMANDS`. If shell-wrap support is genuinely needed, parse the `-c` argument with `shlex.split` and recursively validate.

### 3.4 [High] Shell injection in `_run_hook` via dashboard-controllable `HARNESS_JOB_NAME`
**File:** `harness/schedule.py:509-544`

`asyncio.create_subprocess_shell(hook, env=env)`. `env["HARNESS_JOB_NAME"]` comes from a web-oneshot row's operator-supplied `name` via `dashboard.py:3369`. Hooks templated as `"echo $HARNESS_JOB_NAME"` then execute arbitrary shell on injected characters. Also: dashboard's `/config-tree/<section>` POST can rewrite `schedule.jobs[].on_success` to arbitrary shell on a CSRF-less / loopback-bound dashboard.

**Recommendation:** Validate `name` against `[A-Za-z0-9._-]{1,64}`. Switch hooks to `create_subprocess_exec(["sh", "-c", hook])` and refuse hooks containing `$HARNESS_JOB_*` interpolation unless explicitly opted-in.

### 3.5 [High] `safe_resolve` TOCTOU + symlink confusion on non-existent paths
**File:** `harness/trust.py:32-63`

`os.path.realpath` of a path with non-existent tail returns the resolved-prefix + literal tail. Symlinks injected after validation but before open succeed. `_awrite`'s `O_NOFOLLOW` checks only the target file, not parent components.

**Recommendation:** Walk parents with `O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC` step-by-step, then `openat` for the final write. Or restrict writes to a dedicated `O_DIRECTORY` fd captured at workspace open.

### 3.6 [High] `_handle_session_purge` runs `["harness", "purge", ...]` via PATH lookup
**File:** `harness/dashboard.py:3582-3585`

If the dashboard process's PATH includes `.` or a writable directory, an attacker on the same host can shadow `harness` → RCE in the dashboard's privilege.

**Recommendation:** Resolve via `shutil.which("harness")` at startup; refuse to start if the path isn't absolute and inside a system prefix.

### 3.7 [High] Redactor false-negatives — common secrets not covered
**File:** `harness/redactor.py:39-68, 255-263`

- GCP service-account JSON keys not matched.
- Azure storage account keys (`AccountKey=…`) not matched.
- Slack webhook URLs (`hooks.slack.com/services/…`) not matched.
- Discord bot tokens not matched.
- npm tokens (`npm_<36>`) not matched.
- AWS secret access keys: rely on entropy detection, which is **off by default** (`entropy_detection: bool = False`).
- Private-key regex requires `\n` after BEGIN — fails on `\r\n` and JSON-inlined `-----BEGIN…\\n…`.
- **List-form `content` (Anthropic typed blocks) is skipped entirely** at `redactor.py:255-263` — `tool_use.input` / `tool_result.content` carrying secrets ship to the provider AND get written to `~/.harness/debug/*.txt` by `_dump_llm_call_to_disk`.

**Recommendation:** Add missing patterns; turn on entropy detection by default; recurse into list-form content with per-block redaction.

### 3.8 [High] XSS via operator-supplied `chart_js_url` / `carbon_css_url`
**File:** `harness/dashboard.py:1124-1128`

`html.escape` does not block `javascript:` schemes. If an attacker can flip these config values (e.g. CSRF on a default loopback-only dashboard with auth off), the next operator visit loads arbitrary JS in the dashboard origin → pivot to token reads / HITL approvals.

**Recommendation:** Restrict scheme + host: allowlist `https://` + a small set of known CDNs, or pin to bundled local assets.

### 3.9 [High] `audit_log.detail` persists raw form JSON including secrets
**File:** `harness/dashboard.py:3139-3140, 3216-3219`

`append_audit(..., detail=json.dumps(parsed, default=str))` — `parsed` includes `models[].api_key`, `dashboard.token_env` content, etc. The `FormField.secret` flag is ignored by the audit serialiser.

**Recommendation:** Walk `parsed` and replace any field whose schema marks `secret=True` with `"<redacted>"` before serialising.

### 3.10 [High] Default dashboard config binds 127.0.0.1 with auth **off** → drive-by DNS-rebinding risk
**File:** `harness/dashboard.py:3150-3231` and config defaults

Tree-shaped config endpoint allows arbitrary nested writes through `_handle_config_tree_save`. With auth disabled on loopback and a browser hitting the local port via DNS-rebinding, the same-origin policy can be defeated for state-changing POSTs.

**Recommendation:** Require auth even on loopback (default-secure); validate `Host:` header against an explicit allowlist; refuse to bind on non-loopback without `token_env` set.

### 3.11 [Medium] `SandboxBackend` interpolates `workspace_path` into `sh -c` without `shlex.quote`
**File:** `harness/sandbox.py:266-277, 927`

Workspaces containing `'` break out of the single-quoted segment.

**Recommendation:** `shlex.quote(workspace_path)` at every interpolation site.

### 3.12 [Medium] `gh` argv validation gaps — `repo` / `base` could start with `-`
**File:** `harness/github_integration.py:174-178, 287-291`

`subprocess.run(["gh", "pr", "create", "--base", base, ...])` — argv mode blocks shell injection, but `base="--draft"` produces flag confusion. Same for `repo` in `fetch_issue`.

**Recommendation:** Reject any value starting with `-`; for `repo`, regex `^[A-Za-z0-9][\w.-]*/[\w.-]+$`.

### 3.13 [Medium] `_browse_response` exposes the entire host filesystem
**File:** `harness/dashboard.py:274-318`

No allowlist; symlinks followed; reachable via DNS rebinding on the default config.

**Recommendation:** Restrict to a set of allowlisted roots (workspace, home, /tmp); require auth.

### 3.14 [Medium] `_handle_hitl_webhook` accepts client-supplied `request_id` and overwrites existing entries
**Files:** `harness/dashboard.py:3700-3754`; `harness/web_state.py:309-317`

`register_pending` unconditionally overwrites. A repeat / malicious `request_id` orphans the original blocked handler — the prior caller may receive a response intended for a new prompt.

**Recommendation:** Refuse to overwrite; or always mint server-side `uuid.uuid4().hex` and return it.

### 3.15 [Medium] Tree-config save lets dotted writes reach `schedule.jobs[].on_success` (shell hook content)
**File:** `harness/dashboard.py:3150-3231`

`parse_tree` accepts arbitrary nested dotted keys. If the strict validator only validates *known* nested keys, dotted-path POSTs land unchanged. Combined with finding 3.4, lets a CSRF-capable attacker plant a shell hook.

**Recommendation:** Validator must reject unknown nested keys with a hard error, not a warning.

### 3.16 [Low] `safe_subprocess_env` allowlist misses `*_PROXY`, `SSH_AUTH_SOCK`, `KUBECONFIG`
**File:** `harness/trust.py:407-422`

Build can leak via proxy, use forwarded ssh-agent, or read the kubeconfig file.

**Recommendation:** Move to an explicit allowlist (default-deny) for the sandbox subprocess.

### 3.17 [Low] `audit_log` retention uses lexicographic compare on ISO-8601 strings
**File:** `harness/web_state.py:671`

Works as long as every writer uses the same fixed-width ISO format. A future writer with a different format silently dropped rows.

**Recommendation:** Switch to `INTEGER` epoch column.

---

## 4. LLM Gateway & External Network

### 4.1 [High] Read/connect/write timeouts not retried — single slow Anthropic call blows up the dispatch
**File:** `harness/gateway.py:1313`

```python
except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.PoolTimeout) as exc:
```
`httpx.ReadTimeout`, `httpx.ConnectTimeout`, `httpx.WriteTimeout` derive from `TimeoutException` — not from the caught siblings. Only `PoolTimeout` happens to inherit both. Net effect: the most common transient failure mode (ReadTimeout on a busy provider) propagates unhandled.

**Recommendation:** Add `httpx.TimeoutException` (parent of read/connect/write timeouts) to the except tuple.

### 4.2 [High] Circuit breaker undercounts — only counts *fully exhausted* dispatches
**File:** `harness/gateway.py:2138-2151, 1650-1653`

`_record_rate_limit_failure` fires only after `retry_with_backoff` burns all 5 retries. With `_circuit_failure_threshold=3` and `max_retries=5`, the breaker only opens after ~18 actual 429s — far weaker than the comment claims.

**Recommendation:** Increment the failure counter on every 429, not just on full exhaustion. Reset the counter on first success after a streak.

### 4.3 [High] `_parse_json_response` wraps `JSONDecodeError` as `HTTPStatusError(status=200)` → never retried
**File:** `harness/gateway.py:422-435`

A transient 200 + HTML proxy banner is wrapped with `status=200`; `retry_with_backoff:1301-1311` only retries 429/5xx → falls into the non-retryable `raise`.

**Recommendation:** Synthesise as a retryable error (e.g. raise a custom `RetryableProviderError`); `retry_with_backoff` should retry it.

### 4.4 [High] Empty-content retries silently double-bill — partial provider costs never deducted
**File:** `harness/gateway.py:2172-2208`

The empty-retry loop reassigns `response`; only the *last* response's `usage.cost_usd` is deducted. Up to 2× billing leak per dispatch on partial-failure tails.

**Recommendation:** Accumulate cost across all attempts; deduct the sum.

### 4.5 [High] Mid-dispatch cancellation leaves provider-billed work unaccounted
**File:** `harness/gateway.py:2127-2257`

Cancellation between `retry_with_backoff` return and `budget_remaining_usd -= cost` skips the deduction. The provider has charged.

**Recommendation:** `try: … finally:` that deducts the cost when `response` was bound.

### 4.6 [Medium] Token estimator is `chars // 4 + 50` — wrong by 2-3× on CJK / code / JSON
**File:** `harness/gateway.py:1078-1096, 2073-2089`

Both `check_context_window` and pre-flight budget projection use this estimate. Misclassifies in both directions: false `BudgetTooLowError` on English-heavy prompts, false-pass on CJK / minified JSON → provider 413 / OOM.

**Recommendation:** Use a real tokenizer per provider (`tiktoken` for OpenAI/DeepSeek; Anthropic's `count_tokens` endpoint for Claude); fall back to char/3 for unknown.

### 4.7 [Medium] Providers treat 200-OK with `{"error": …}` body as empty success
**File:** `harness/gateway.py:628, 915, 997`

`choices[0]` missing → content = `""` → empty-retry loop fires 3 more billed retries → final `EmptyLLMResponseError` masks the real provider error message in `data["error"]["message"]`.

**Recommendation:** Inspect `data.get("error")` first; if present, raise a structured error carrying the provider's message.

### 4.8 [Medium] Low-budget Ollama fallback crashes when `ollama_local_model` is empty
**File:** `harness/gateway.py:1903-1910`

When budget < 0.05, rewrites `model_key = "ollama:" + ""` → `create_provider` raises ValueError. Crashes exactly when the operator most needs graceful degradation.

**Recommendation:** Guard the rewrite with `if not self.config.ollama_local_model: log + return BudgetTooLowError`.

### 4.9 [Medium] `StdioMcpClient._read_loop` uses unbounded `readline()` — single huge line can OOM
**File:** `harness/mcp_client.py:360`

`limit=` not passed; a malformed server returning megabytes without `\n` buffers until OOM or `LimitOverrunError` (then handshake fails with no diagnostic).

**Recommendation:** Pass `limit=10*1024*1024`; on `LimitOverrunError`, log the server name and the first 512 bytes.

### 4.10 [Medium] `OpenAIEmbeddingsBackend` has no retry, ignores `ssl_verify`, discards partial work
**File:** `harness/repo_index.py:397-416, 423-434`

A 429 on chunk 90/200 raises mid-loop → 89 already-computed vectors discarded. Empty chunk content (empty source file) → batch rejected → entire index aborts. Corporate proxies that work for the gateway silently fail here.

**Recommendation:** Reuse `retry_with_backoff`; filter empty chunks; honour `ssl_verify`; checkpoint partial results.

### 4.11 [Medium] `WebFetchSkill` reads full body into memory before applying byte cap
**File:** `harness/web_tools.py:272-274, 350-363`

`response = client.get(url)` reads everything; cap is applied to `response.content` only after. A multi-GB target only stopped by the 20 s timeout.

**Recommendation:** Use `client.stream("GET", url)` + accumulate up to cap.

### 4.12 [Medium] `retry_with_backoff` jitter collapses when `delay > max_delay`
**File:** `harness/gateway.py:1308-1321`

`delay = base * 2**attempt`; `jittered = delay * (0.5 + random() * 0.5)`; `min(jittered, max_delay)`. For attempt ≥ 6, `delay > 60` → jittered is always ≥ 30 → `min(.., 60)` floors at 60 — synchronized retry storms across concurrent dispatches.

**Recommendation:** Apply `min(delay, max_delay)` **before** jitter.

### 4.13 [Medium] No total elapsed-time bound on `retry_with_backoff`
**File:** `harness/gateway.py:1280-1325`

`Retry-After: 86400` is taken literally — dispatch sleeps for 24 h. Cumulative sleep can hit 300+ s.

**Recommendation:** Honour `Retry-After` but cap to `max_delay`; enforce a `max_total_seconds` ceiling on the whole retry sequence.

### 4.14 [Medium] Anthropic typed-content tool_results not redacted
**File:** `harness/redactor.py:255-263` (also covered in 3.7 above)

Listed here because it's an LLM-gateway-side data-leak risk.

### 4.15 [Low] `RateLimit-Reset` epoch heuristic mis-classifies stale-epoch values
**File:** `harness/gateway.py:1372-1382`

A value of `1700000000` (epoch in 2023) is compared with `now` (epoch in 2026); `target > now` is False → returned as-is (1.7 B seconds).

**Recommendation:** Require explicit unit metadata, or sanity-cap both interpretations to `max_delay`.

### 4.16 [Low] Anthropic thinking mode silently rewrites caller-supplied `max_tokens`
**File:** `harness/gateway.py:794-800`

`max_tokens=512` + thinking → silently bumped to 2048. Defeats per-role caps with no log line.

**Recommendation:** Log a WARN; expose the original request alongside the rewritten one.

### 4.17 [Low] `health_check_loop` keeps polling on terminally-unhealthy services
**File:** `harness/deploy.py:1188-1190`

`status=running, health=unhealthy` keeps looping for the full timeout, then reports a misleading "timed out".

**Recommendation:** Treat `unhealthy` (or repeated `unhealthy`) as terminal failure; surface "container unhealthy".

---

## 5. CLI / Config / Persistence

### 5.1 [High] `cmd_resume` never acquires the workspace lock
**File:** `harness/cli.py:4351` (vs `cli.py:3649` for `cmd_run`)

Two concurrent `harness resume --session-id X` on the same workspace clobber each other's patches.

**Recommendation:** Call `_acquire_workspace_lock` at the top of `cmd_resume`.

### 5.2 [High] `--session-id` unsanitized — path injection, file-name corruption
**File:** `harness/storage.py:440` (`generate_session_id`)

Flows into log filenames, CR archive directory names, git branch names. `--session-id "../../etc/passwd"` or RTL-override chars break things.

**Recommendation:** Reject anything not matching `^[A-Za-z0-9._-]{1,64}$`.

### 5.3 [High] `inspect_session().is_active` boolean has wrong logic
**File:** `harness/storage.py:746`

```python
is_active=exit_code not in (0, -1) and exit_code != 0,
```
A never-built session (`exit_code = -1` default until `compiler_node` runs) is marked inactive — the dashboard / `harness status` under-reports running sessions.

**Recommendation:** Replace with `is_active = (exit_code is None) or (exit_code == -1 and not terminated_at)` — depends on the real intent.

### 5.4 [High] Checkpoint TTL GC silently skips corrupted rows
**File:** `harness/storage.py:344-356`

Corrupted blob → `_deserialize_checkpoint_blob` returns `{}` → `ts == ""` → `continue` skips delete. Corrupted threads accumulate forever.

**Recommendation:** Delete rows whose blob fails strict decode after a grace period (e.g. 60 days), or quarantine into a `corrupted_checkpoints` table.

### 5.5 [High] CI auto-approval checks `CI=true` literal; most CI providers set `CI=1`
**File:** `harness/hitl.py:113`; `harness/cli.py:1403`

GitHub Actions sets `CI=true` (works). GitLab / Jenkins / Circle set `CI=1` or `CI=yes` → not detected → interactive `input()` hangs.

**Recommendation:** Mirror `_bool_choice` semantics: `{"true","1","yes","on"}` (case-insensitive).

### 5.6 [High] `cmd_run` exception path closes the checkpointer without commit
**File:** `harness/cli.py:4234-4239`

`await checkpointer.conn.close()` without prior `await checkpointer.conn.commit()` — last unflushed checkpoint is lost. Resume can't recover the state the user just saw on screen.

**Recommendation:** `try: await checkpointer.conn.commit(); finally: await checkpointer.conn.close()`.

### 5.7 [High] `HttpChannel._post` returns the gate's `default` on **every** failure mode
**File:** `harness/hitl.py:425-428`

Webhook down → log + return default. For REQUIREMENTS / ARCHITECTURE gates (default = approve), this is **silent auto-approve** the operator never sees.

**Recommendation:** Fail-closed for approval gates: raise `HitlChannelUnavailable`; let the caller decide.

### 5.8 [High] `HttpChannel` retries with synchronous `time.sleep` inside an async caller
**File:** `harness/hitl.py:386-422`

Up to ~14 s of blocking sleep freezes the event loop.

**Recommendation:** Switch the channel to `httpx.AsyncClient` and `await asyncio.sleep`.

### 5.9 [Medium] `_archive_consumed_change_requests` uses `os.replace` cross-FS → silent fail → CR re-processed
**File:** `harness/cli.py:2994`

NFS / bind-mount / btrfs subvol boundaries break `os.replace` with `EXDEV`. Current except logs a warning; the `.txt` stays in place → re-applied next run.

**Recommendation:** Fall back to `shutil.move` on EXDEV; or hash-rename to `<cr>.applied.<hash>` in-place to make idempotency robust.

### 5.10 [Medium] `_archive_consumed_change_requests` manifest write is not atomic
**File:** `harness/cli.py:3019-3026`

SIGKILL mid-write leaves truncated JSON; reads fail thereafter.

**Recommendation:** Use `metrics.write_atomic` (tmp file + fsync + replace).

### 5.11 [Medium] `metrics.write_atomic` fixed tmp filename + missing dir fsync
**File:** `harness/metrics.py:496-518`

Two concurrent writers collide on `<dest>.tmp`. After `os.replace`, no `fsync` on the parent dir — rename may not survive a power loss.

**Recommendation:** `tempfile.NamedTemporaryFile(dir=…, delete=False)` for unique tmp; `os.fsync` on the parent dir fd after replace.

### 5.12 [Medium] `RotatingFileHandler` is not multi-process safe
**File:** `harness/observability.py:265-270`

Two processes sharing one `<session>.jsonl` (e.g. `--force-lock` resume next to a still-running run) race on `doRollover` → truncated JSONL records.

**Recommendation:** Either embed PID in the filename, or switch to a multi-process-safe handler (e.g. `concurrent_log_handler`).

### 5.13 [Medium] Typos in `llm_dispatch.continue_on_length` keys silently no-op
**File:** `harness/cli.py:1166-1180`

Role-name typos (`"planing"` for `"planning"`) pass validation; per-role default kicks in; operator thinks they've overridden it.

**Recommendation:** Validate role names against `NodeRole`.

### 5.14 [Medium] API-key validation accepts any non-empty string
**File:** `harness/cli.py:886-924`

`find_missing_api_keys` only checks emptiness. Placeholder values (`your-api-key-here`), wrong-provider keys, keys with whitespace all pass. Crashes mid-graph after the patch branch is created.

**Recommendation:** Apply provider-specific regex sanity checks at startup (or 1-token ping per `_doctor_check_api_keys`).

### 5.15 [Low] `validate_checkpoint_schema` allows resume when **metadata** blob is corrupted
**File:** `harness/storage.py:589-617`

Corrupted metadata → `{}` → version is None → "legacy" warning → resume proceeds.

**Recommendation:** Treat metadata decode failure as schema-mismatch.

### 5.16 [Low] `_deserialize_checkpoint_blob` JSON path uses `errors="replace"`
**File:** `harness/storage.py:530`

Truncated UTF-8 silently substitutes U+FFFD.

**Recommendation:** Use `errors="strict"`; msgpack covers the binary case.

### 5.17 [Low] `purge_checkpoints` lacks explicit transaction → partial wipe on mid-loop OSError
**File:** `harness/storage.py:883-889`

`DELETE FROM writes` succeeds, then `DELETE FROM checkpoints` raises → `writes` is empty but `checkpoints` retains rows that reference nothing.

**Recommendation:** Wrap in `BEGIN IMMEDIATE` + explicit rollback.

### 5.18 [Low] `_walk_dotted` only descends dicts — type schema can never grow into list entries
**File:** `harness/cli.py:1227-1239`

Documented limitation; future `mcp.servers.0.name` style entries would silently fail to validate.

**Recommendation:** Document explicitly or extend `_walk_dotted` to handle integer indices.

### 5.19 [Low] `_get_global_config_path` doesn't resolve symlinks
**File:** `harness/cli.py:266-275`

`pip install -e` from a symlinked checkout produces the wrong global config path.

**Recommendation:** `os.path.realpath(__file__)`; or use `importlib.resources` to colocate config.

---

## 6. Patcher / Graph State Machine

### 6.1 [High] `route_after_compiler` lets `MISSING_DEP` autofix bypass *both* iteration and zero-patch tripwires
**File:** `harness/graph.py:5319-5366`

`consecutive_zero >= 2 and not has_autofixable` → only escalates when no autofix is available. An alternating cycle `pip missing → autofixed → wheel missing → autofixed → pip missing again` resets `missing_dep_last_symbol` and never escalates.

**Recommendation:** Track a generic "no real patch progress in N iterations" counter that is **not gated** on `has_autofixable`.

### 6.2 [High] `speculate_node` winner path drops earlier-pass `modified_files`
**File:** `harness/speculative.py:1042 vs 1086-1091`

Winner path: `"modified_files": winner.modified_files` (REPLACE). Salvage path: merges with `prior_modified`. In `after_n_repair_failures` trigger mode, patching_node already wrote files — winner-replace drops them from state even though they remain on disk → commit / test_generation visibility lost.

**Recommendation:** Merge in both paths (always preserve prior `modified_files`).

### 6.3 [High] Patching tool-loop hits read-file cap with no feedback to the model
**File:** `harness/graph.py:1438-1494`

`_PATCHING_READ_FILE_CAP = 6` — once exceeded, the loop returns the **last response with its read_file tool_calls intact**; downstream code treats them as patches.

**Recommendation:** After the cap, inject a synthetic tool-result message ("read cap reached; emit patches now") and let the model regenerate.

### 6.4 [High] `ImpactAnalyzer` / `DependencyGraph` are entirely orphaned
**File:** `harness/impact.py` (no callers in the live graph)

The docstring claims "Called inside patching_node and repair_node after process_llm_patch_output() succeeds." No node actually invokes them. Patches land without any cross-file impact warning.

**Recommendation:** Either wire `ImpactAnalyzer` in, or delete the module to avoid the "looks like a safety net" mirage.

### 6.5 [High] `repo_index` has no incremental update — planner sees pre-patch files mid-session
**File:** `harness/repo_index.py:522-585`

`build_index` is full-rewrite; no SHA-based skip / incremental update. Planning that re-runs after a refine cycle queries stale chunks.

**Recommendation:** Hash + mtime check per chunk; or invalidate per-file on patcher write events.

### 6.6 [High] Autofix file reads use `errors="strict"` → `UnicodeDecodeError` crashes the autofix node
**File:** `harness/autofix.py:935-939, 1007-1011, 1072-1076, 290-294, 838-842`

`except OSError:` does not catch `UnicodeDecodeError(ValueError)`. The autofix dies; the diagnostic is not marked unhandled; LLM fallback never runs.

**Recommendation:** Open with `errors="replace"`; or catch `UnicodeError` alongside `OSError`.

### 6.7 [High] `_python_module_path` produces invalid Python imports for dash-named dirs
**File:** `harness/autofix.py:623-633`

`rel.replace(os.sep, ".")[:-3]` blindly. `my-pkg/foo.py` → `from my-pkg.foo import bar` → SyntaxError. LLM sees a worse error than the original NameError.

**Recommendation:** Validate the constructed module path with `keyword.iskeyword` per part and `str.isidentifier`; skip emission if invalid.

### 6.8 [High] `compiler_node` has no per-tool exit-code interpretation
**File:** `harness/graph.py:3990+`

Any non-zero exit triggers repair. Tools that emit advisory non-zero (e.g. `terraform validate` on benign drift) loop the LLM endlessly. `_BUILD_OUTPUT_NOISE_PATTERNS` only filters the LLM's view, not the routing decision.

**Recommendation:** Allow operators to declare "advisory" non-zero exit codes per `build_command`; surface them as warnings, not failures.

### 6.9 [Medium] `_strip_build_output_noise` only strips single-line deprecation warnings
**File:** `harness/graph.py:2606-2623`

Multi-line `DeprecationWarning` blocks (header + code context) pass through; LLM still sees the stacktrace.

**Recommendation:** Match the warning header + N following lines until a blank line or new diagnostic.

### 6.10 [Medium] Pip-resolution-conflict regex risk on long single-line logs
**File:** `harness/graph.py:2552-2560`

`.+` over `^ERROR: …` with no anchored upper bound; mitigated by `raw_output[-4000:]` slice but worth bounding.

**Recommendation:** Replace `.+` with `[^\n]{1,500}`.

### 6.11 [Medium] `_strip_line_number_prefixes` can mangle real content
**File:** `harness/patcher.py:1601-1621`

`_LINE_NUMBER_PREFIX_RE` requires every non-blank line to match — a markdown file documenting the patcher (containing `"  3| …"` in a code fence) gets its real content stripped.

**Recommendation:** Require a contiguous numeric run from line 1 (or skip stripping if any line lacks a prefix).

### 6.12 [Medium] Lintgate auto-fix silently modifies files **after** patcher commit
**File:** `harness/lintgate.py`

Ruff/prettier auto-fix paths rewrite files; the patcher's `files_seen_by_llm` hash is stale → subsequent SEARCH/REPLACE on those files mis-matches.

**Recommendation:** Re-hash modified files after lintgate; or run lintgate as a pre-patch gate, not post-patch.

### 6.13 [Medium] `_resolve_path` (lintgate) → TOCTOU between validate and subprocess
**File:** `harness/lintgate.py:827-859`

Validated path can be swapped before the formatter sees it. Minor since lintgate runs on trusted files.

**Recommendation:** Pass paths via a fd (`/proc/self/fd/N`) where supported.

---

## 7. Cross-Cutting Themes

- **Inconsistent `node_state` merge discipline.** `compiler_node` and `test_generation_node` merge; `patching_node`, `lintgate_node`, `speculative_node` replace. A checkpoint reaching a replacer loses prior signals (see 6.2).
- **Default-fail-open posture.** Auto-approve on non-TTY, webhook-default-on-failure, validators that swallow errors and continue — many failure paths route to "proceed" instead of "halt". For an agent that mutates the workspace, default-deny is usually the right policy.
- **Atomic-write claims that aren't atomic.** Several callers reimplement tmp+rename; only some `fsync`; only `metrics.write_atomic` does it correctly. Centralise this primitive (5.10 / 5.11 / 1.14).
- **Subprocess timeouts that don't kill.** A consistent pattern across `deploy.py`, `lintgate.py`, `schedule.py`: `wait_for(communicate(), timeout)` without follow-up `proc.kill()` (2.3 / 2.5 / 2.6). One shared helper would fix all sites.
- **Singletons and shared mutable globals** (`_global_command_validator`, `_WORKSPACE_LOCK_HANDLE`, `_process_registry`, `_hitl_queue`) are convenient but fragile under tests, the dashboard's threading model, and any future in-process re-entry. Prefer dependency injection.
- **Token / cost accounting has multiple silent-leak paths** (4.4 / 4.5 / 1.8). Hard caps are advisory in practice.
- **SSRF posture is loose** (3.1 / 3.2 + the `web_fetch` LLM-callable surface). Cloud-metadata access is one easy DNS record away. Worth treating as a single coordinated hardening pass.

---

## 8. Suggested Prioritisation

If addressing one batch at a time:

**Batch A — security-sensitive correctness (recommend first):**
- 1.3 (unredacted checkpoints), 3.1 (SSRF redirect), 3.2 (DNS rebinding), 3.3 (`sh`/`bash` bypass), 3.4 (hook injection), 3.6 (`harness` PATH lookup), 3.7 (redactor false-negatives — esp. typed-content blocks), 3.10 (default-secure dashboard).

**Batch B — silent data loss / corruption:**
- 1.1 (double-firing one-shots), 1.2 (PID-reuse kill), 1.4 (cancel leaks), 5.6 (no commit before close), 5.4 (stale corrupted rows), 6.2 (winner-replace).

**Batch C — operator pain (hangs, leaks, false approvals):**
- 2.1 (orphan harness), 2.2 (no SIGKILL escalation), 2.3 / 2.5 / 2.6 (timeout-but-no-kill family), 4.1 (timeout not retried), 4.2 (weak circuit breaker), 5.5 (CI auto-approve regex), 5.7 (webhook fail-open), 5.8 (sync sleep in async).

**Batch D — defensive hardening:**
- Everything Medium / Low above.

---

*Audit conducted via 6 parallel read-only sweeps over the harness modules. No code modifications were made. Quoted line numbers are from HEAD = `b83703d`. Recommendations are advisory — please confirm before any implementation.*
