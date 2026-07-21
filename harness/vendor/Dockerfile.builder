# Kitchen-sink builder image for myharness.
#
# Bakes every toolchain the harness's supported stacks need into a single
# image so the sandbox never has to swap images or apt-get inside the
# container at runtime (the latter is impossible under --user $UID:$GID
# mode, which is the default).
#
# Covers: Python 3.12 (uv-managed CPython) + pip + venv + uv, Java JDK 21
# + Maven + Gradle, Node 20 LTS + npm + yarn + pnpm + tsc, SQLite,
# Playwright + Chromium. Plus make, gcc, git, curl as the universal glue.
#
# Python comes from `uv python install`, NOT Ubuntu's apt: jammy's
# python3.11 package is frozen at 3.11.0rc1 — an unreleased RELEASE
# CANDIDATE — so every generated app was being built and tested against
# a 2022 pre-final interpreter while the harness host runs 3.14. The
# uv-managed build lands at ${UV_PYTHON_INSTALL_DIR} (world-readable for
# --user mode) and /usr/local/bin/python3 points at it, which outranks
# any /usr/bin/python3 an apt dependency may drag in. Bumping Python is
# now a one-line version change below.
#
# Test toolchain is pre-baked, NOT generated-app runtime deps:
#   - Python: pytest, pytest-cov, pytest-xdist installed system-wide so
#     test_generation_node skips the `pip install pytest` round trip on
#     every run. Project runtime deps (fastapi, django, …) are still
#     resolved from the workspace's manifest, not pre-baked — pre-baking
#     them would mask missing entries in requirements.txt and produce
#     "works in sandbox, broken everywhere else" projects.
#   - Node: jest, ts-jest, @types/jest, typescript installed globally so
#     the JS / TS test commands skip the `npm install jest` round trip.
#   - uv: pre-installed and on PATH. Generated Makefiles use `uv pip
#     install` instead of `pip install` — same `requirements.txt` /
#     `pyproject.toml` semantics, 10-30× faster cold installs.
#
# /cache/{pip,uv,npm} are pre-created world-writable so the harness can
# bind a writable named Docker volume there without a runtime chown
# round trip. Cache env vars (PIP_CACHE_DIR, UV_CACHE_DIR,
# npm_config_cache) point at these paths so the next compile in any
# session reuses downloaded wheels / tarballs.
#
# Build (local, single-host — the default workflow):
#   docker build --pull \
#     -f harness/vendor/Dockerfile.builder \
#     -t harness-builder:latest \
#     -t harness-builder:$(date +%Y-%m-%d) \
#     harness/vendor/
#   docker inspect harness-builder:latest --format '{{.RepoDigests}}'
#
# Then paste the digest into harness/sandbox.py:BUILDER_IMAGE so the
# sandbox is content-addressed even on a local-only image (buildx stamps
# RepoDigests for local builds — no registry push required).
#
# Build + push (multi-host fleets only — replace <owner> with your handle):
#   docker buildx build \
#     --platform linux/amd64,linux/arm64 \
#     -t ghcr.io/<owner>/harness-builder:$(date +%Y-%m-%d) \
#     -t ghcr.io/<owner>/harness-builder:stable \
#     --push -f harness/vendor/Dockerfile.builder harness/vendor/

FROM eclipse-temurin:21-jdk-jammy

# uv as a static binary (no Python needed to bootstrap it).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Managed-CPython home. Set as ENV (not a build ARG) so runtime uv
# invocations — including under --user $UID:$GID — discover the
# interpreter without a writable home directory.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ARG PYTHON_VERSION=3.12

# NodeSource for Node 20 LTS — Ubuntu Jammy's apt ships Node 12, too old
# for current web frameworks.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg \
 && mkdir -p /etc/apt/keyrings \
 && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
      | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
 && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
      > /etc/apt/sources.list.d/nodesource.list \
 && apt-get update && apt-get install -y --no-install-recommends \
      maven gradle \
      nodejs \
      sqlite3 libsqlite3-dev \
      make gcc git \
 && npm install -g yarn pnpm typescript playwright jest ts-jest @types/jest \
 && uv python install "${PYTHON_VERSION}" \
 && PY="$(uv python find "${PYTHON_VERSION}")" \
 && ln -sf "$PY" /usr/local/bin/python3 \
 && ln -sf "$PY" /usr/local/bin/python \
 # pip directly into the managed interpreter (python-build-standalone
 # ships pip bundled — no ensurepip, which PEP668 also blocks). The
 # interpreter is PEP668-marked ("managed by uv"), and `uv pip install
 # --python` refuses non-venv targets while `--system` overrides
 # `--python`'s interpreter choice — pip with --break-system-packages
 # is the sanctioned override for an immutable baked image.
 && "$PY" -m pip install --no-cache-dir --break-system-packages --upgrade \
      pip setuptools wheel \
 && "$PY" -m pip install --no-cache-dir --break-system-packages \
      pytest pytest-cov pytest-xdist pytest-timeout hypothesis \
 && chmod -R a+rX /opt/uv-python \
 && PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers \
      npx --yes playwright install --with-deps chromium \
 && chmod -R a+rX /opt/playwright-browsers \
 && mkdir -p /cache/pip /cache/uv /cache/npm \
 && chmod -R a+rwX /cache \
 && rm -rf /var/lib/apt/lists/* \
 && python3 --version && python3 -m pip --version && python3 -m pytest --version

ENV PIP_ROOT_USER_ACTION=ignore \
    JAVA_HOME=/opt/java/openjdk \
    PATH=/opt/java/openjdk/bin:$PATH \
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers \
    PIP_CACHE_DIR=/cache/pip \
    UV_CACHE_DIR=/cache/uv \
    npm_config_cache=/cache/npm

WORKDIR /workspace
