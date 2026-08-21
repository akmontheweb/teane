## Deployment — Dockerfiles & docker-compose

### When this skill applies

You are generating container deployment artifacts (a `dockerfile` string per
build service in the architecture blueprint) for a locally-deployed app. The
build context for EVERY service is the **workspace root** (`context: .` in
compose) — so every `COPY` path is written relative to the root, e.g.
`COPY ./server/requirements.txt ./`, `COPY ./client/ .`.

### Universal rules (all services)

- **Base images:** slim/alpine, pinned major version (e.g. `python:3.12-slim`,
  `node:20-alpine`, `caddy:2-alpine`).
- **Non-root at runtime.** Compose runs build services as the host UID/GID
  (`user: "${UID:-1000}:${GID:-1000}"`), so files copied into the image are
  root-owned and unreadable/unwritable by that user. Copy application files
  with `COPY --chown=1000:1000 ...` (or create a user and `chown -R`), and put
  any writable dirs (npm cache, tmp) somewhere the runtime user owns.
- **Healthcheck must hit a real, always-present endpoint.** FastAPI always
  serves `/docs`; a bare app may not have `/health`. Prefer an endpoint you
  know exists.
- **Never assume a lockfile exists.** Greenfield apps ship `package.json` /
  `requirements.txt` but NOT `package-lock.json`. Commands that require a
  lockfile (`npm ci`) must fall back (`npm install`).
- **Surface build failures — do not swallow them.** Never end a build step with
  `2>/dev/null || true`; a masked failure produces an empty/broken image that
  crashes at runtime with a confusing error. Let the build fail loudly so the
  repair loop sees the real error.

### Python / FastAPI service

- Install deps, then copy source:
  ```dockerfile
  FROM python:3.12-slim
  WORKDIR /app
  COPY ./server/requirements.txt ./
  RUN pip install --no-cache-dir -r requirements.txt
  COPY --chown=1000:1000 ./server/ .
  ENV PYTHONUNBUFFERED=1
  ```
- **ASGI entrypoint:** the build context root is COPYed to `/app`, so a nested
  `server/app/main.py` is imported as `app.main:app`, NOT `main:app`. Inspect
  the actual file that constructs `app = FastAPI(...)` and use its module path
  relative to the COPYed root. CMD:
  `CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]`
- Do NOT hand-roll a multi-stage `COPY --from=builder /usr/local/lib/python*/…`
  copy — the `python*` glob is literal in the COPY destination and lands
  packages off `sys.path`. A single stage with `pip install` is correct and
  simpler.

### React / Vite (or other SPA) service

- A Vite/CRA app builds to static files and is SERVED statically — it is NOT
  `node dist/index.js`.
  ```dockerfile
  FROM node:20-alpine AS builder
  WORKDIR /app
  COPY ./client/package.json ./client/package-lock.json* ./
  RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi
  COPY ./client/ .
  RUN npm run build

  FROM node:20-alpine
  WORKDIR /app
  RUN npm install -g serve
  COPY --from=builder --chown=1000:1000 /app/dist ./dist
  ENV NODE_ENV=production
  CMD ["serve", "-s", "dist", "-l", "3000"]
  ```
- Install ALL deps (build tools like `vite`/`typescript` are devDependencies)
  so `npm run build` can run. Do not pass `--omit=dev` in the build stage.
- Serve the build output (`dist/` for Vite, `build/` for CRA) with `serve -s`
  or an nginx image; do not `npm start` a SPA.

### Reverse proxy (caddy / nginx)

- A proxy is NOT a language app — build from the proxy base image and let the
  config be bind-mounted by compose (compose already mounts
  `./Caddyfile:/etc/caddy/Caddyfile`):
  ```dockerfile
  FROM caddy:2-alpine
  ```
- Never render a proxy with a python/node template (no `COPY requirements.txt`).

### Self-check before returning each `dockerfile`

1. Does every `COPY` path resolve from the workspace ROOT?
2. Is the runtime user able to read (and, where needed, write) the files?
3. For Python: is the uvicorn module path correct for the real file layout?
4. For a SPA: does it build to static assets and serve them (not `node …`)?
5. No `|| true` masking on build steps; healthcheck hits a real endpoint.
