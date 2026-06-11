# App Deployment

How the harness deploys the app it just built — what artefacts you get, what commands run, what the operator does next.

## TL;DR

- **You don't write the `docker-compose.yml`.** The harness's `deployment_node` does it for you after the security scan returns clean, deriving services / volumes / networks from a deterministic workspace telemetry scan + one LLM call.
- **You don't run `docker-compose up`** either — the harness runs `docker-compose up --build -d` itself and polls the declared health endpoints. If health checks fail, it routes back to the LLM repair loop.
- **The dev-env deployment is gated.** A Phase 3.5 preview gate shows you the proposed Dockerfile / compose / Caddyfile before they execute. Approve interactively, or bypass under CI with `HARNESS_AUTO_APPROVE=true` (or `CI=true`).
- **Checkpoint message redaction is on by default.** Any `messages`-channel content that gets checkpointed during the deployment phase is scrubbed through `harness.redactor` before SQLite. Set `persistence.redact_messages: false` only if you need verbatim transcripts at rest (e.g. audit replay).
- Config switch: `deployment.enabled` (default `true`). Set `false` to skip the whole phase and own deployment yourself.
- **Limitation**: the bring-up runs on the local Docker daemon the harness has access to. There is no SSH-driven remote-deploy subcommand yet — for remote dev, copy the generated artefacts and run the same `docker-compose up` there.

## The four phases of `deployment_node`

Implemented in `harness/deploy.py:deployment_node` (line 1109). Runs after the architecture-spec gate is approved (or auto-approved) and the build passes the security scan.

| Phase | What runs | Mode | Output |
|------|-----------|------|--------|
| 1 | `scan_workspace_telemetry` (`harness/deploy.py:278`) | Deterministic, token-free | Telemetry dict — detected languages, frameworks, databases, port hints, volume needs |
| 2 | `synthesize_architecture` (`harness/deploy.py:407`) | LLM-driven | JSON blueprint enumerating services, volumes, networks, optional reverse proxy |
| 3 | `generate_assets_from_blueprint` (`harness/deploy.py:849`) | Deterministic, token-free | Writes `Dockerfile`, `docker-compose.yml`, and (when needed) `Caddyfile` directly into the workspace |
| 3.5 | Preview gate | Interactive HITL or env-var bypass | Operator approves the rendered Dockerfile + compose before execution |
| 4 | `health_check_loop` (`harness/deploy.py:1014`) | `docker-compose up --build -d` + polling | Containers running and reporting healthy, OR failures routed to `repair_node` |

On phase failure the node populates `compiler_errors` with a `DEPLOYMENT_SYNTHESIS_FAILED` / `DEPLOYMENT_GENERATION_FAILED` / `DEPLOYMENT_NO_COMPOSE_FILE` diagnostic and lets the standard repair loop attempt a fix.

## When does it run?

The deployment phase is reached only after several other gates pass:

1. `patching_node` lands code.
2. `lintgate_node` is clean.
3. `compiler_node` exits 0.
4. `security_scan_node` is clean (see `route_after_security_scan`, `harness/graph.py:2226`).
5. `deployment_discovery_node` collects any unknowns (or skips when none).
6. The human gatekeeper approves the `DEPLOYMENT` gate (or `HARNESS_AUTO_APPROVE=true` / `CI=true` bypasses).
7. Then `deployment_node` runs.

So a green build that doesn't have a deployment-discovery loop still reaches the deployment phase quickly; greenfield projects spend more time in steps 5–6 collecting infra unknowns.

## Configuration

Defaults are baked into `harness/deploy.py` (lines 1122–1130) and can be overridden in `~/.harness/config.json` or `<workspace>/.harness_config.json`:

```json
"deployment": {
  "enabled": true,
  "compose_file": "docker-compose.yml",
  "health_check_interval_seconds": 2.0,
  "health_check_timeout_seconds": 30.0
}
```

To skip the deployment phase entirely (when you want to own deployment yourself), set `"deployment": { "enabled": false }`. The graph ends after the security scan, with the source + tests + Dockerfile/compose **not** generated.

## What lands in the workspace

| File | Generating node | Always or conditional |
|------|-----------------|----------------------|
| `Dockerfile` (one per service in the blueprint) | `deployment_node` Phase 3 | Always (when phase 3 runs) |
| `docker-compose.yml` | `deployment_node` Phase 3 | Always (when phase 3 runs) |
| `Caddyfile` | `deployment_node` Phase 3 | Only when a reverse proxy is in the blueprint |
| `DEPLOYMENT_BLUEPRINT.md` | `generate_deployment_spec_node` | When deployment discovery ran |
| `SPEC_ARCHITECTURE.md` | `write_spec_node` | Always |
| `SPEC_REQUIREMENTS.md` | `write_spec_node` | Always |
| Source code | `patching_node` | Always |
| Tests (per stack) | `test_generation_node` | Always (when `test_generation.enabled = true`) |

## The deploy commands, made explicit

The harness already runs the first command for you when phase 4 fires. These are what you run **after** a successful run if you want to stop, tail logs, or restart the dev env:

```bash
# Bring the dev env up (the harness already did this once)
docker-compose up --build -d

# Tail logs across all services
docker-compose logs -f

# Stop and remove containers (keeps volumes)
docker-compose down

# Stop, remove containers AND volumes (wipes data)
docker-compose down -v
```

If a service won't come up after a code change, re-run `harness resume --session-id <id>` — the repair loop will iterate on the compose/Dockerfile or the code until it lands. Resume pre-flights the latest checkpoint with strict deserialization and a schema-version check; if either fails you'll get an operator-readable message naming the recovery options (`docs/RUNBOOK.md` § 1). Track cost as you iterate with `harness metrics --session-id <id>` — it reports total spend, the trailing 10-minute burn rate, and projected minutes-to-exhaustion at the current rate against `token_budget.hard_cap_usd`.

## The Phase 3.5 preview gate

The Dockerfile + docker-compose + Caddyfile are synthesized partly from LLM JSON. A prompt-injected dependency manifest or a confused model could put `RUN curl … | sh` in a Dockerfile, so the preview gate shows you the rendered artefacts before `docker-compose up --build` runs.

- **Interactive**: read the preview the harness prints, type `y` to approve or `n` to abandon.
- **Non-interactive / CI**: set one of:
  ```bash
  export HARNESS_AUTO_APPROVE=true
  # or
  export CI=true
  ```
  Either bypasses the gate. Use this when you trust the loop (e.g. you've reviewed the architecture blueprint up-front) or when running headless.

## Health-check failures

Phase 4 polls the health endpoints declared in the blueprint at `deployment.health_check_interval_seconds` until `deployment.health_check_timeout_seconds` elapses. On failure:

- The per-service status is captured in `compiler_errors` with a `DEPLOYMENT_*` error code.
- The router sends the state to `repair_node`, which has visibility into both the application code and the generated Dockerfile/compose.
- The LLM gets a chance to fix either layer — bad health probe path in the compose, missing migration step in the Dockerfile, application bug that makes the service exit on startup, etc.
- If the repair loop hits its cap, the run routes to HITL with the per-service status visible.

## Bringing the same setup up on a remote dev box

There is no `harness deploy --remote ssh://host` today. To deploy to a different machine:

1. Copy the workspace (or at minimum the generated `Dockerfile`, `docker-compose.yml`, and `Caddyfile` if present) to the target host.
2. Ensure the target has Docker and Docker Compose installed.
3. Run `docker-compose up --build -d` on the target.

The generated compose file does not assume hostnames or filesystem paths the local machine has uniquely, so this works as-is for typical dev/staging targets. Production targets usually want different volume mounts and secrets, which the operator overrides via a `docker-compose.override.yml`.

## What the harness does NOT generate today

Operator-facing gaps to be aware of, so you don't go looking for files that aren't there:

- **No `install.sh` / `setup.sh`** for non-Docker dev installs. Operators that want a pip / venv / npm dev setup derive it from `build_command` in `.harness_config.json` and the source tree.
- **No `Makefile install` target.**
- **No systemd unit files**, Helm charts, or Kubernetes manifests. The current compose file is the only orchestration artefact.
- **No SSH-driven remote-deploy subcommand** (`harness deploy --target ssh://host`).
- **No auto-invocation of the README docgen** — when the workspace doesn't have a `README.md`, you can run `DocGenSkill("readme")` manually (it has an "Installation" section template), but the harness does not call it automatically at end of run.

Future work directions: a stack-aware `install_generation_node` mirroring `harness/test_generation.py` (with `install_guides/<lang>.md` per stack) is the closest precedent; deterministic verification would run `bash install.sh && <build_command>` in a fresh sandbox container.

## Reference

- `harness/deploy.py:deployment_node` (line 1109) — canonical implementation.
- `harness/deploy.py:scan_workspace_telemetry` (line 278), `synthesize_architecture` (line 407), `generate_assets_from_blueprint` (line 849), `health_check_loop` (line 1014) — phase entry points.
- `harness/graph.py:route_after_security_scan` (line 2226) — where the deployment phase joins the graph.
- [docs/installation.md](installation.md) — how to install the harness itself.
- [docs/SPEC_ARCHITECTURE.md](SPEC_ARCHITECTURE.md) — graph topology and module map (§5.21–§5.30 cover the recent reliability + security primitives that wrap deployment).
- [docs/RUNBOOK.md](RUNBOOK.md) — top-five operator failure modes with diagnostic + fix recipes.
