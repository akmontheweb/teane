"""
Inference-Driven Provisioner — Dynamic Containerization from Codebase Telemetry.

This module implements:
    Phase 1 (scan_workspace_telemetry): Deterministic, token-free directory scanning
        using pathlib to detect package manifests, framework signatures, database/service
        anchors, and port bindings across the workspace.
    Phase 2 (synthesize_architecture): Routes telemetry + SPEC_ARCHITECTURE.md to
        the planning LLM with strict JSON output enforcement. Returns a typed
        architecture blueprint dict.
    Phase 3 (generate_assets_from_blueprint): Programmatically constructs multi-stage
        Dockerfiles, docker-compose.yml, and proxy routing configs (Caddyfile) from
        the LLM's JSON blueprint — zero LLM tokens used.
    Phase 4 (deployment_node): Orchestrates Phases 1-3, then builds and monitors
        containers with health-check polling. On failure, captures logs and populates
        compiler_errors for automated repair.

Integration:
    - Wired after security_scan_node in the graph. On success → END, on failure → repair_node.
    - Uses existing gateway injection, patcher engine, and sandbox executor.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import shutil
import subprocess
import sys
import time as time_module
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM-output validators — delegated to harness/trust.py
# ---------------------------------------------------------------------------
# Identifier validators and blueprint validation live in the central trust
# module. Local aliases are kept so call sites in this file don't change.
from harness.trust import (  # noqa: E402
    validate_blueprint as _validate_blueprint,
)


# Files we always show in the deploy preview (in this order).
_PREVIEW_FILES = ("docker-compose.yml", "Dockerfile", "Caddyfile")
# Max chars per file to display so the preview stays scannable.
_PREVIEW_MAX_CHARS = 4000


def _auto_approve_deploy() -> bool:
    """
    Return True if the user has explicitly authorized non-interactive
    deploys via env var. We do NOT auto-approve on non-TTY alone: a
    piped-in deploy must opt in, otherwise we fail closed.
    """
    return (
        os.environ.get("CI", "").lower() == "true"
        or os.environ.get("HARNESS_AUTO_APPROVE", "").lower() == "true"
    )


def _read_preview(workspace_path: str, generated_files: list[str]) -> str:
    """
    Build a human-readable preview of the LLM-generated deploy artifacts.
    Reads docker-compose.yml, all Dockerfile(s), and Caddyfile if present.
    """
    seen: set[str] = set()
    sections: list[str] = []

    def _emit(rel: str) -> None:
        if rel in seen:
            return
        seen.add(rel)
        abs_path = os.path.join(workspace_path, rel)
        if not os.path.isfile(abs_path):
            return
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                body = f.read()
        except OSError as e:
            sections.append(f"--- {rel} (read failed: {e}) ---")
            return
        if len(body) > _PREVIEW_MAX_CHARS:
            body = body[:_PREVIEW_MAX_CHARS] + f"\n... [truncated; full file is {len(body)} chars]"
        sections.append(f"--- {rel} ---\n{body}")

    # Predictable order first, then anything else generated.
    for name in _PREVIEW_FILES:
        _emit(name)
    for path in generated_files:
        rel = os.path.relpath(path, workspace_path) if os.path.isabs(path) else path
        # Show any Dockerfile.* variants too
        if rel not in seen and (rel.endswith(".yml") or rel.startswith("Dockerfile") or rel.endswith("Caddyfile")):
            _emit(rel)

    if not sections:
        return "(no deploy artifacts found to preview)"
    return "\n\n".join(sections)


async def _prompt_deploy_approval(preview: str) -> bool:
    """
    Show the deploy preview and require explicit confirmation before any
    `docker-compose up` runs.

    Routing:
    - HARNESS_AUTO_APPROVE / CI → auto-approve (opt-in, logged as warning).
    - Non-TTY without opt-in → fail closed.
    - Interactive TTY → prompt via HitlChannel.
    """
    from harness.hitl import get_channel as _get_channel
    channel = _get_channel()

    print("\n" + "=" * 72, file=sys.stderr)
    print(" DEPLOY PREVIEW — LLM-generated containers about to be launched", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print(preview, file=sys.stderr)
    print("=" * 72, file=sys.stderr)

    if not channel.is_interactive():
        if _auto_approve_deploy():
            logger.warning(
                "[deployment_node] Auto-approving deploy preview (CI/HARNESS_AUTO_APPROVE set)."
            )
            return True
        logger.error(
            "[deployment_node] Refusing deploy: no TTY for interactive approval and "
            "HARNESS_AUTO_APPROVE is not set. Re-run interactively or set the env var."
        )
        return False

    return channel.confirm(
        "Proceed with `docker-compose up --build -d`?", default=False
    )


# ---------------------------------------------------------------------------
# Phase 1: Workspace Telemetry Scanner (Deterministic, Token-Free)
# ---------------------------------------------------------------------------

# Package manifest files to detect by language
_PACKAGE_MANIFESTS: dict[str, list[str]] = {
    "python": ["requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", "Pipfile", "poetry.lock"],
    "node": ["package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"],
    "go": ["go.mod", "go.sum"],
    "rust": ["Cargo.toml", "Cargo.lock"],
    "java": ["pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle"],
    "ruby": ["Gemfile", "Gemfile.lock"],
    "php": ["composer.json", "composer.lock"],
    "dotnet": ["*.csproj", "*.fsproj", "*.sln"],
}

# Service/technology anchor keywords to search for in configs and source
_SERVICE_ANCHORS: dict[str, list[str]] = {
    "postgres": ["postgres", "postgresql", "psql", "PG_"],
    "mysql": ["mysql", "mariadb", "MYSQL_"],
    "redis": ["redis", "REDIS_"],
    "mongodb": ["mongodb", "mongo", "MONGO_"],
    "keycloak": ["keycloak", "KEYCLOAK_"],
    "caddy": ["caddy", "Caddyfile"],
    "nginx": ["nginx", "NGINX_"],
    "kafka": ["kafka", "KAFKA_"],
    "rabbitmq": ["rabbitmq", "RABBITMQ_", "amqp"],
    "elasticsearch": ["elasticsearch", "ELASTICSEARCH_"],
    "celery": ["celery", "CELERY_"],
}

# Framework signature files/directories
_FRAMEWORK_SIGNATURES: dict[str, list[str]] = {
    "django": ["manage.py"],
    "flask": ["app.py", "wsgi.py"],
    "fastapi": ["main.py"],  # heuristic — may also be flask
    "nextjs": ["next.config.js", "next.config.mjs", "next.config.ts"],
    "react": ["src/App.tsx", "src/App.jsx", "src/App.js"],
    "express": ["app.js", "server.js"],
    "rails": ["config/routes.rb", "app/controllers"],
    "spring": ["src/main/java", "src/main/resources/application.properties"],
    "laravel": ["artisan", "app/Http/Controllers"],
}


def _find_files(workspace: Path, patterns: list[str]) -> list[str]:
    """Find files in workspace matching any of the given patterns (shell-style globs)."""
    found: list[str] = []
    for pattern in patterns:
        try:
            matches = list(workspace.rglob(pattern))
            for m in matches[:5]:  # Limit per pattern
                if m.is_file() and ".git" not in str(m) and "node_modules" not in str(m):
                    rel = str(m.relative_to(workspace))
                    if rel not in found:
                        found.append(rel)
        except Exception:
            pass
    return found


def _find_dirs(workspace: Path, patterns: list[str]) -> list[str]:
    """Find directories matching any of the patterns."""
    found: list[str] = []
    for pattern in patterns:
        try:
            matches = list(workspace.rglob(pattern))
            for m in matches[:5]:
                if m.is_dir() and ".git" not in str(m) and "node_modules" not in str(m):
                    rel = str(m.relative_to(workspace))
                    if rel not in found:
                        found.append(rel)
        except Exception:
            pass
    return found


def _search_anchors_in_files(workspace: Path, keywords: list[str]) -> bool:
    """Search source/config files for service anchor keywords. Returns True if any found."""
    config_files = []
    for ext in ("*.env", "*.env.*", "*.yml", "*.yaml", "*.toml", "*.json", "*.py", "*.ts", "*.js", "*.go"):
        config_files.extend(list(workspace.rglob(ext)))

    scanned = 0
    for fpath in config_files:
        if scanned > 100:
            break
        if ".git" in str(fpath) or "node_modules" in str(fpath):
            continue
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace").lower()
            for kw in keywords:
                if kw.lower() in content:
                    return True
            scanned += 1
        except (OSError, UnicodeDecodeError):
            pass
    return False


def _extract_port_hints(workspace: Path) -> list[int]:
    """Extract port numbers from .env and common config files."""
    ports: set[int] = set()
    env_files = list(workspace.rglob("*.env*"))
    for fp in env_files[:10]:
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
            import re
            for match in re.finditer(r'(?:PORT|port)\s*[:=]\s*(\d{2,5})', content):
                port = int(match.group(1))
                if 1 < port < 65536:
                    ports.add(port)
        except (OSError, ValueError):
            pass

    # Also check docker-compose files
    for fp in list(workspace.rglob("docker-compose*.yml"))[:5]:
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
            import re
            for match in re.finditer(r'["\x27]?(\d{2,5}):\d{2,5}["\x27]?', content):
                port = int(match.group(1))
                if 1 < port < 65536:
                    ports.add(port)
        except (OSError, ValueError):
            pass

    return sorted(ports)[:10]


def scan_workspace_telemetry(workspace_path: str) -> dict[str, Any]:
    """
    Deterministic, token-free workspace scanner.

    Detects:
        - Package manifests by language
        - Database/service anchors (redis, postgres, mysql, etc.)
        - Web server frameworks (nginx, caddy)
        - Auth services (keycloak)
        - Framework signatures (django, nextjs, react, etc.)
        - Source directory structure
        - Port hints from .env and compose files

    Args:
        workspace_path: Absolute path to the project root.

    Returns:
        JSON telemetry dictionary with all structural findings.
    """
    workspace = Path(workspace_path)
    app_name = workspace.name

    # Scan manifests
    languages: list[str] = []
    manifests_found: dict[str, list[str]] = {}
    for lang, patterns in _PACKAGE_MANIFESTS.items():
        found = _find_files(workspace, patterns)
        if found:
            languages.append(lang)
            manifests_found[lang] = found

    # Scan service anchors
    databases_detected: list[str] = []
    web_servers_detected: list[str] = []
    auth_services_detected: list[str] = []
    queue_services_detected: list[str] = []

    for service, keywords in _SERVICE_ANCHORS.items():
        if _search_anchors_in_files(workspace, keywords):
            if service in ("postgres", "mysql", "mongodb", "elasticsearch"):
                databases_detected.append(service)
            elif service in ("caddy", "nginx"):
                web_servers_detected.append(service)
            elif service == "keycloak":
                auth_services_detected.append(service)
            elif service in ("kafka", "rabbitmq", "celery"):
                queue_services_detected.append(service)
            else:
                databases_detected.append(service)  # redis, etc.

    # Scan framework signatures
    frameworks_detected: list[str] = []
    for fw, signatures in _FRAMEWORK_SIGNATURES.items():
        found_files = _find_files(workspace, signatures)
        found_dirs = _find_dirs(workspace, signatures)
        if found_files or found_dirs:
            frameworks_detected.append(fw)

    # Scan source directories
    src_dirs: list[str] = []
    for candidate in ("src", "app", "api", "lib", "pkg", "cmd", "services", "packages", "frontend", "backend"):
        candidate_path = workspace / candidate
        if candidate_path.is_dir():
            src_dirs.append(candidate)

    # Port hints
    port_hints = _extract_port_hints(workspace)

    # Check for existing infrastructure files
    has_dockerfile = (workspace / "Dockerfile").exists()
    has_compose = (workspace / "docker-compose.yml").exists() or (workspace / "docker-compose.yaml").exists()
    has_caddyfile = (workspace / "Caddyfile").exists()

    telemetry = {
        "app_name": app_name,
        "workspace_path": workspace_path,
        "languages": languages,
        "package_manifests": manifests_found,
        "frameworks_detected": frameworks_detected,
        "databases_detected": databases_detected,
        "web_servers_detected": web_servers_detected,
        "auth_services_detected": auth_services_detected,
        "queue_services_detected": queue_services_detected,
        "src_directories": src_dirs,
        "port_hints": port_hints,
        "existing_infrastructure": {
            "dockerfile": has_dockerfile,
            "docker_compose": has_compose,
            "caddyfile": has_caddyfile,
        },
    }

    logger.info(
        "[deploy:telemetry] Scan complete: langs=%s, dbs=%s, fw=%s, ports=%s",
        languages, databases_detected, frameworks_detected, port_hints,
    )

    return telemetry


# ---------------------------------------------------------------------------
# Phase 2: Architectural Synthesis (LLM Composer)
# ---------------------------------------------------------------------------

# Strict JSON schema for the LLM response
_ARCHITECTURE_JSON_SCHEMA = """
{
  "services": {
    "<service_name>": {
      "base_image": "python:3.12-slim",
      "build_context": "./api",
      "ports": ["8000:8000"],
      "environment_keys_needed": ["DB_HOST", "REDIS_URL"],
      "depends_on_services": ["postgres"],
      "requires_healthcheck_cmd": "curl -f http://localhost:8000/health || exit 1",
      "volumes": ["./api:/app"]
    }
  },
  "volumes": {
    "<volume_name>": { "driver": "local" }
  },
  "networks": {
    "<network_name>": { "driver": "bridge" }
  },
  "proxy_service": "caddy" or null
}
"""


async def synthesize_architecture(
    telemetry: dict[str, Any],
    workspace_path: str,
    spec_arch_path: str = "SPEC_ARCHITECTURE.md",
) -> dict[str, Any]:
    """
    Route telemetry + SPEC_ARCHITECTURE.md content to the planning LLM.
    Enforces strict JSON output schema describing the container architecture.

    Args:
        telemetry: Output from scan_workspace_telemetry().
        workspace_path: Project root path.
        spec_arch_path: Path to the architecture specification file.

    Returns:
        Parsed architecture blueprint dict matching the JSON schema.
    """
    from harness.graph import get_gateway

    gateway = get_gateway()
    if gateway is None:
        logger.error("[deploy:compose] No gateway configured. Cannot synthesize architecture.")
        return _fallback_blueprint(telemetry)

    # Read SPEC_ARCHITECTURE.md
    spec_path = Path(workspace_path) / spec_arch_path
    spec_content = ""
    if spec_path.is_file():
        try:
            spec_content = spec_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            spec_content = "(could not read SPEC_ARCHITECTURE.md)"

    # Build the prompt
    prompt = f"""You are a Principal DevOps Architect. Analyze the workspace telemetry below
and the project's SPEC_ARCHITECTURE.md to design the complete container infrastructure.

## Workspace Telemetry
```json
{json.dumps(telemetry, indent=2)}
```

## SPEC_ARCHITECTURE.md
{spec_content if spec_content else "(no SPEC_ARCHITECTURE.md found)"}

## Your Task
Design the optimal container architecture. Return ONLY a valid JSON object matching this EXACT schema:

```json
{_ARCHITECTURE_JSON_SCHEMA}
```

### Rules
1. Create exactly ONE service per source directory or language sub-project.
2. If no databases/web servers/auth services are detected, only create app services.
3. Use slim/alpine base images always.
4. Configure proper healthchecks for every service.
5. Link services via depends_on_services where dependencies exist.
6. If the workspace has a web framework, add a web router/proxy service (Caddy or Nginx).
7. Use port_hints from telemetry to set correct port mappings.
8. Do NOT include any text outside the JSON object. Only return valid JSON."""

    logger.info("[deploy:compose] Synthesizing architecture with planning LLM...")

    from harness.gateway import NodeRole
    messages = [
        {"role": "system", "content": "You are a DevOps infrastructure architect. You output ONLY valid JSON. No markdown, no explanation, no code blocks around the JSON."},
        {"role": "user", "content": prompt},
    ]

    try:
        response, budget = await gateway.dispatch(
            messages=messages,
            role=NodeRole.PLANNING,
            budget_remaining_usd=2.00,
        )

        # Parse the JSON response
        content = response.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        blueprint = json.loads(content)
        logger.info("[deploy:compose] Architecture synthesized: %d service(s).", len(blueprint.get("services", {})))
        return blueprint

    except json.JSONDecodeError as exc:
        logger.warning("[deploy:compose] LLM returned invalid JSON: %s. Falling back.", exc)
        return _fallback_blueprint(telemetry)
    except Exception:
        logger.exception("[deploy:compose] Architecture synthesis failed.")
        return _fallback_blueprint(telemetry)


def _fallback_blueprint(telemetry: dict[str, Any]) -> dict[str, Any]:
    """Generate a basic fallback blueprint from telemetry when LLM is unavailable."""
    services: dict[str, dict[str, Any]] = {}
    app_name = telemetry.get("app_name", "app")

    # Create one service per source directory
    src_dirs = telemetry.get("src_directories", [])
    if not src_dirs:
        src_dirs = ["."]

    languages = telemetry.get("languages", ["python"])

    for i, src_dir in enumerate(src_dirs):
        svc_name = f"{app_name}" if len(src_dirs) == 1 else f"{app_name}-{src_dir.replace('/', '-')}"
        lang = languages[0] if languages else "python"

        base_images = {
            "python": "python:3.12-slim",
            "node": "node:20-alpine",
            "go": "golang:1.22-alpine",
            "rust": "rust:1.78-slim",
            "java": "eclipse-temurin:21-jre-alpine",
        }
        image = base_images.get(lang, "alpine:3.20")

        services[svc_name] = {
            "base_image": image,
            "build_context": f"./{src_dir}" if src_dir != "." else ".",
            "ports": [f"{8000 + i}:{8000 + i}"],
            "environment_keys_needed": [],
            "depends_on_services": [],
            "requires_healthcheck_cmd": "",
            "volumes": [],
        }

    blueprint: dict[str, Any] = {
        "services": services,
        "volumes": {},
        "networks": {"app-network": {"driver": "bridge"}},
        "proxy_service": None,
    }

    # Add database services
    for db in telemetry.get("databases_detected", []):
        db_configs = {
            "postgres": {"base_image": "postgres:16-alpine", "ports": ["5432:5432"]},
            "mysql": {"base_image": "mysql:8.4", "ports": ["3306:3306"]},
            "redis": {"base_image": "redis:7-alpine", "ports": ["6379:6379"]},
            "mongodb": {"base_image": "mongo:7", "ports": ["27017:27017"]},
        }
        if db in db_configs:
            services[db] = {
                "base_image": db_configs[db]["base_image"],
                "build_context": "",
                "ports": db_configs[db]["ports"],
                "environment_keys_needed": [],
                "depends_on_services": [],
                "requires_healthcheck_cmd": "",
                "volumes": [f"{db}-data:/var/lib/{db}"],
            }
            # Link app services to DB
            for svc_name in list(services.keys()):
                if svc_name != db:
                    services[svc_name].setdefault("depends_on_services", []).append(db)

    # Add Caddy if web servers detected
    if telemetry.get("web_servers_detected"):
        services["caddy"] = {
            "base_image": "caddy:2-alpine",
            "build_context": "",
            "ports": ["80:80", "443:443"],
            "environment_keys_needed": [],
            "depends_on_services": [s for s in services if s != "caddy"],
            "requires_healthcheck_cmd": "",
            "volumes": ["./Caddyfile:/etc/caddy/Caddyfile"],
        }
        blueprint["proxy_service"] = "caddy"

    return blueprint


# ---------------------------------------------------------------------------
# Phase 3: Deterministic Multi-Stage Code Generation
# ---------------------------------------------------------------------------

# Language-specific Dockerfile templates
_DOCKERFILE_TEMPLATES = {
    "python": """# Multi-stage Python Dockerfile
FROM python:{python_version}-slim AS builder
WORKDIR /app
COPY {build_context}/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

FROM python:{python_version}-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python*/site-packages /usr/local/lib/python*/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY {build_context}/ .
ENV PYTHONUNBUFFERED=1
{healthcheck}
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{port}"]
""",

    "node": """# Multi-stage Node.js Dockerfile
FROM node:20-alpine AS deps
WORKDIR /app
COPY {build_context}/package.json {build_context}/package-lock.json* ./
RUN npm ci --only=production

FROM node:20-alpine AS builder
WORKDIR /app
COPY {build_context}/ .
COPY --from=deps /app/node_modules ./node_modules
RUN npm run build 2>/dev/null || true

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist 2>/dev/null || COPY --from=builder /app ./
COPY --from=deps /app/node_modules ./node_modules
ENV NODE_ENV=production
{healthcheck}
CMD ["node", "dist/index.js"]
""",

    "go": """# Multi-stage Go Dockerfile
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY {build_context}/go.mod {build_context}/go.sum* ./
RUN go mod download
COPY {build_context}/ .
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /app/server .

FROM alpine:3.20
RUN apk add --no-cache ca-certificates
WORKDIR /app
COPY --from=builder /app/server .
{healthcheck}
CMD ["./server"]
""",

    "rust": """# Multi-stage Rust Dockerfile
FROM rust:1.78-slim AS builder
WORKDIR /app
COPY {build_context}/ .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/* /app/server
{healthcheck}
CMD ["./server"]
""",

    "java": """# Multi-stage Java Dockerfile
FROM eclipse-temurin:21-jdk AS builder
WORKDIR /app
COPY {build_context}/ .
RUN ./gradlew build -x test 2>/dev/null || mvn package -DskipTests

FROM eclipse-temurin:21-jre-alpine
WORKDIR /app
COPY --from=builder /app/build/libs/*.jar app.jar 2>/dev/null || \
COPY --from=builder /app/target/*.jar app.jar 2>/dev/null
{healthcheck}
CMD ["java", "-jar", "app.jar"]
""",
}


def _generate_dockerfile(
    service_name: str,
    service_spec: dict[str, Any],
    language: str,
    workspace_path: str,
    *,
    cr_attribution: Optional[dict[str, str]] = None,
) -> str:
    """Generate a multi-stage Dockerfile for a service.

    ``cr_attribution`` (change-request mode only) maps a service name to a
    one-line ``"CR-N: <reason>"`` summary; when ``service_name`` is in the
    dict the resulting Dockerfile is prefixed with ``# CR-N: <reason>`` so
    a reader can ``grep CR-N`` and find the build artifact for the
    originating request. ``None`` (the default) yields byte-identical
    output to pre-change-request behaviour.
    """
    tmpl = _DOCKERFILE_TEMPLATES.get(language, _DOCKERFILE_TEMPLATES["python"])

    build_context = service_spec.get("build_context", ".").strip("/")
    if build_context == ".":
        build_context = "."

    port = "8080"
    if service_spec.get("ports"):
        first_port = service_spec["ports"][0]
        port = first_port.split(":")[-1] if ":" in first_port else first_port

    python_version = "3.12"

    healthcheck_cmd = service_spec.get("requires_healthcheck_cmd", "")
    if healthcheck_cmd:
        healthcheck = f'HEALTHCHECK --interval=10s --timeout=5s --retries=3 CMD {healthcheck_cmd}'
    else:
        healthcheck = ""

    dockerfile = tmpl.format(
        python_version=python_version,
        build_context=build_context,
        port=port,
        healthcheck=healthcheck,
    )
    if cr_attribution:
        marker = cr_attribution.get(service_name)
        if marker:
            dockerfile = f"# {marker}\n" + dockerfile
    return dockerfile


def _dockerfile_name_for(svc_name: str, services: dict[str, Any]) -> str:
    """Return the on-disk Dockerfile filename for a service.

    Both ``_generate_compose_file`` (which writes the ``dockerfile:`` field
    in docker-compose.yml) and ``generate_assets_from_blueprint`` (which
    writes the file to disk) must agree by construction. Use the same
    helper to avoid the previous divergence where compose used
    ``build_context != "."`` and asset generation used "first service vs
    others" — they could disagree and produce missing-file errors.

    Convention:
      - The first service with a build_context keeps the plain ``Dockerfile``
        name so Docker's default lookup works for single-service projects.
      - Every additional build-context service uses ``Dockerfile.<svc_name>``.
      - Services without a build_context (e.g. ``postgres`` pulled as an
        image) return an empty string.
    """
    if not services.get(svc_name, {}).get("build_context"):
        return ""
    build_services = [n for n, spec in services.items() if spec.get("build_context")]
    if not build_services:
        return ""
    return "Dockerfile" if svc_name == build_services[0] else f"Dockerfile.{svc_name}"


def _generate_compose_file(
    blueprint: dict[str, Any],
    *,
    cr_attribution: Optional[dict[str, str]] = None,
) -> str:
    """Generate a docker-compose.yml from the architecture blueprint.

    Adds default resource limits to every service (mem_limit, cpus,
    pids_limit) so a runaway container — leak, fork-bomb, or simply a
    misconfigured workload — can't OOM the host. Defaults are conservative
    but can be overridden per-service via blueprint.services.<svc>.limits.

    ``cr_attribution`` (change-request mode only) maps a service name to a
    one-line ``"CR-N: <reason>"`` summary; each annotated service block is
    preceded by ``# CR-N: <reason>`` so the request that introduced or
    changed the service is grep-able in the rendered YAML. ``None``
    yields byte-identical output to pre-change-request behaviour.
    """
    services = blueprint.get("services", {})
    volumes_cfg = blueprint.get("volumes", {})
    networks_cfg = blueprint.get("networks", {})
    default_limits = blueprint.get(
        "default_limits",
        {"memory": "512m", "cpus": "1.0", "pids": 200},
    )
    lines = ['version: "3.9"', "", "services:"]

    for svc_name, svc_spec in services.items():
        if cr_attribution:
            marker = cr_attribution.get(svc_name)
            if marker:
                lines.append(f"  # {marker}")
        lines.append(f"  {svc_name}:")

        if svc_spec.get("build_context"):
            lines.append("    build:")
            lines.append(f"      context: {svc_spec.get('build_context', '.')}")
            lines.append(f"      dockerfile: {_dockerfile_name_for(svc_name, services)}")
        else:
            lines.append(f"    image: {svc_spec.get('base_image', 'alpine:3.20')}")

        if svc_spec.get("ports"):
            lines.append("    ports:")
            for port_mapping in svc_spec["ports"]:
                lines.append(f'      - "{port_mapping}"')

        if svc_spec.get("environment_keys_needed"):
            lines.append("    environment:")
            for key in svc_spec["environment_keys_needed"]:
                lines.append(f"      - {key}=${{{key}}}")

        if svc_spec.get("depends_on_services"):
            lines.append("    depends_on:")
            for dep in svc_spec["depends_on_services"]:
                lines.append(f"      - {dep}")

        if svc_spec.get("volumes"):
            lines.append("    volumes:")
            for vol in svc_spec["volumes"]:
                lines.append(f"      - {vol}")

        if svc_spec.get("requires_healthcheck_cmd"):
            health_cmd = svc_spec["requires_healthcheck_cmd"]
            lines.append("    healthcheck:")
            lines.append(f"      test: [\"CMD-SHELL\", \"{health_cmd}\"]")
            lines.append("      interval: 10s")
            lines.append("      timeout: 5s")
            lines.append("      retries: 3")

        # Resource limits — per-service override wins, else defaults from
        # blueprint, else hardcoded floor.
        svc_limits = svc_spec.get("limits", {}) if isinstance(svc_spec.get("limits"), dict) else {}
        mem = svc_limits.get("memory", default_limits["memory"])
        cpus = svc_limits.get("cpus", default_limits["cpus"])
        pids = svc_limits.get("pids", default_limits["pids"])
        lines.append(f"    mem_limit: {mem}")
        lines.append(f"    cpus: \"{cpus}\"")
        lines.append(f"    pids_limit: {pids}")

        lines.append("    networks:")
        for net_name in networks_cfg:
            lines.append(f"      - {net_name}")

        lines.append("")

    # Volumes
    if volumes_cfg:
        lines.append("volumes:")
        for vol_name in volumes_cfg:
            lines.append(f"  {vol_name}:")
        lines.append("")

    # Networks
    if networks_cfg:
        lines.append("networks:")
        for net_name, net_spec in networks_cfg.items():
            driver = net_spec.get("driver", "bridge") if isinstance(net_spec, dict) else "bridge"
            lines.append(f"  {net_name}:")
            lines.append(f"    driver: {driver}")
        lines.append("")

    return "\n".join(lines)


def _generate_caddyfile(
    blueprint: dict[str, Any],
    *,
    cr_attribution: Optional[dict[str, str]] = None,
) -> str:
    """Generate a Caddyfile from the architecture blueprint.

    ``cr_attribution`` (change-request mode only) maps a service name to
    a one-line ``"CR-N: <reason>"`` summary; each annotated reverse-proxy
    stanza is preceded by ``# CR-N: <reason>``. ``None`` yields
    byte-identical output to pre-change-request behaviour.
    """
    services = blueprint.get("services", {})
    lines = ["# Auto-generated Caddyfile", ""]

    for svc_name, svc_spec in services.items():
        if svc_name == "caddy":
            continue
        ports = svc_spec.get("ports", [])
        if ports:
            container_port = ports[0].split(":")[-1] if ":" in ports[0] else ports[0]
            # Derive a domain-like name
            domain = f"{svc_name}.localhost" if svc_name != services.get("app_name", "app") else "localhost"
            if cr_attribution:
                marker = cr_attribution.get(svc_name)
                if marker:
                    lines.append(f"# {marker}")
            lines.append(f"{domain} {{")
            lines.append(f"    reverse_proxy {svc_name}:{container_port}")
            lines.append("}")
            lines.append("")

    if not any("reverse_proxy" in line for line in lines):
        lines.append(":80 {")
        lines.append("    respond \"Caddy is running. No services configured.\" 200")
        lines.append("}")

    return "\n".join(lines)


def generate_assets_from_blueprint(
    blueprint: dict[str, Any],
    telemetry: dict[str, Any],
    workspace_path: str,
    *,
    cr_attribution: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """
    Programmatically construct Dockerfiles, docker-compose.yml, and Caddyfile
    from the synthesized blueprint. Zero LLM tokens used.

    Args:
        blueprint: The architecture blueprint from synthesize_architecture().
        telemetry: The telemetry dict from scan_workspace_telemetry().
        workspace_path: Path to write generated files.
        cr_attribution: Optional mapping of service name -> ``"CR-N: <reason>"``
            summary used in change-request mode. When supplied, the
            generated Dockerfile, compose, and Caddyfile blocks for each
            annotated service carry a ``# CR-N: <reason>`` comment so a
            reader can ``grep CR-N`` and trace deployment artifacts to
            the originating request. When ``None`` (or when not in
            change-request mode), the call site falls back to
            ``blueprint.get("cr_attribution")`` so the deployment
            synthesizer can carry attribution data inline with the
            blueprint instead of plumbing a separate channel. Output is
            byte-identical to pre-change-request behaviour when neither
            source is set.

    Returns:
        Dict with list of generated file paths.
    """
    if cr_attribution is None:
        cr_attribution = blueprint.get("cr_attribution")
    if cr_attribution is not None and not isinstance(cr_attribution, dict):
        logger.warning(
            "[deploy:generate] Ignoring non-dict cr_attribution=%r — "
            "expected {service_name: 'CR-N: <reason>'}.",
            type(cr_attribution).__name__,
        )
        cr_attribution = None
    # Validate before generating anything — refuse to write files for a
    # blueprint that would inject newlines/semicolons into Dockerfile or
    # YAML. The preview gate downstream is the user's last defense; this
    # is defense in depth.
    validation_errors = _validate_blueprint(blueprint)
    if validation_errors:
        return {
            "success": False,
            "generated": [],
            "message": "Blueprint rejected by validator:\n  - " + "\n  - ".join(validation_errors),
        }

    generated: list[str] = []
    services = blueprint.get("services", {})
    languages = telemetry.get("languages", ["python"])
    primary_lang = languages[0] if languages else "python"
    workspace = Path(workspace_path)

    # Generate Dockerfiles per service. Skip services with no build context
    # (pure image pulls like postgres/redis) — fixed from the previous
    # `svc_name != svc_name` condition which was always False.
    for svc_name, svc_spec in services.items():
        if not svc_spec.get("build_context"):
            continue

        # Determine language for this service
        svc_lang = primary_lang
        build_ctx = svc_spec.get("build_context", ".")
        if build_ctx == ".":
            svc_lang = primary_lang

        dockerfile_content = _generate_dockerfile(
            svc_name, svc_spec, svc_lang, workspace_path,
            cr_attribution=cr_attribution,
        )
        dockerfile_name = _dockerfile_name_for(svc_name, services)
        dockerfile_path = workspace / dockerfile_name
        dockerfile_path.write_text(dockerfile_content, encoding="utf-8")
        generated.append(str(dockerfile_path.relative_to(workspace)))
        logger.info("[deploy:generate] Generated %s", dockerfile_name)

    # Generate docker-compose.yml
    compose_content = _generate_compose_file(
        blueprint, cr_attribution=cr_attribution,
    )
    compose_path = workspace / "docker-compose.yml"
    compose_path.write_text(compose_content, encoding="utf-8")
    generated.append("docker-compose.yml")
    logger.info("[deploy:generate] Generated docker-compose.yml (%d services)", len(services))

    # Generate Caddyfile if proxy service specified
    if blueprint.get("proxy_service") == "caddy" or "caddy" in services:
        caddy_path = workspace / "Caddyfile"
        caddy_content = _generate_caddyfile(
            blueprint, cr_attribution=cr_attribution,
        )
        caddy_path.write_text(caddy_content, encoding="utf-8")
        generated.append("Caddyfile")
        logger.info("[deploy:generate] Generated Caddyfile")

    return {
        "success": True,
        "generated": generated,
        "message": f"Generated {len(generated)} infrastructure file(s): {', '.join(generated)}.",
    }


# ---------------------------------------------------------------------------
# Phase 4: Health Check & Deployment Orchestrator
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _compose_argv() -> tuple[str, ...]:
    """Return the argv prefix to invoke Docker Compose.

    Docker Compose V2 (``docker compose``, a Go plugin) has been the
    default since Docker Desktop 4.4+ / Engine 20.10+, and V1
    (``docker-compose``, Python) was end-of-lifed in July 2023 — many
    modern Linux distros and CI images no longer ship it. Prefer V2; only
    fall back to the legacy binary if the V2 plugin is not present.

    The detection runs once per process (lru_cache) so we don't probe
    Docker on every health-check / teardown call.
    """
    if shutil.which("docker"):
        try:
            result = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                timeout=5.0,
                check=False,
            )
            if result.returncode == 0:
                return ("docker", "compose")
        except (subprocess.TimeoutExpired, OSError):
            pass
    if shutil.which("docker-compose"):
        logger.warning(
            "[deploy] Docker Compose V2 plugin not detected; falling back to "
            "legacy docker-compose binary (EOL since July 2023)."
        )
        return ("docker-compose",)
    # Neither resolves — return the V2 form so the eventual subprocess
    # error message is the modern one. The caller's error handling will
    # surface "command not found".
    return ("docker", "compose")


async def _run_docker_inspect(container_name: str) -> dict[str, Any]:
    """Run docker inspect and return parsed status."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "inspect", container_name,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        if proc.returncode != 0:
            return {"name": container_name, "status": "error", "health": "none", "exit_code": -1, "running": False, "error": stderr.decode("utf-8", errors="replace").strip()}

        data = json.loads(stdout.decode("utf-8"))
        if isinstance(data, list) and len(data) > 0:
            c = data[0]
            state = c.get("State", {})
            return {
                "name": container_name,
                "status": state.get("Status", "unknown"),
                "health": state.get("Health", {}).get("Status", "none"),
                "exit_code": state.get("ExitCode", 0),
                "running": state.get("Running", False),
                "error": "",
            }
        return {"name": container_name, "status": "not_found", "health": "none", "exit_code": -1, "running": False, "error": "Not found"}
    except Exception as exc:
        return {"name": container_name, "status": "error", "health": "none", "exit_code": -1, "running": False, "error": str(exc)}


async def _get_compose_services(workspace_path: str, compose_file: str) -> list[str]:
    """Get service names from docker-compose config."""
    compose_path = os.path.join(workspace_path, compose_file)
    if not os.path.isfile(compose_path):
        return []
    try:
        proc = await asyncio.create_subprocess_exec(
            *_compose_argv(), "-f", compose_path, "config", "--services",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=workspace_path,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        if proc.returncode == 0:
            return [line.strip() for line in stdout.decode("utf-8").splitlines() if line.strip()]
        return []
    except Exception:
        return []


async def health_check_loop(
    workspace_path: str,
    compose_file: str = "docker-compose.yml",
    interval_seconds: float = 2.0,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Poll docker inspect for all compose services. Returns success/failure with diagnostics."""
    services = await _get_compose_services(workspace_path, compose_file)
    if not services:
        return {"success": True, "healthy": [], "failed": [], "message": "No services found."}

    logger.info("[deploy:health] Polling %d service(s) every %.1fs for up to %.0fs.", len(services), interval_seconds, timeout_seconds)

    start = time_module.monotonic()
    failed: list[dict[str, Any]] = []
    last_status: dict[str, str] = {}

    while time_module.monotonic() - start < timeout_seconds:
        all_healthy = True
        current: dict[str, str] = {}

        for svc in services:
            result = await _run_docker_inspect(svc)
            if result.get("error"):
                all_healthy = False
                current[svc] = f"error: {result['error']}"
                continue

            status = result["status"]
            current[svc] = f"{status} (health={result['health']})"

            if status in ("exited", "dead", "removing") or (not result["running"] and status != "created"):
                failed.append(result)
                all_healthy = False
                break

            if status not in ("running",) and result["health"] not in ("healthy",):
                all_healthy = False

        if current != last_status:
            logger.info("[deploy:health] %s", "; ".join(f"{s}={v}" for s, v in current.items()))
            last_status = current

        if all_healthy:
            elapsed = time_module.monotonic() - start
            return {"success": True, "healthy": services, "failed": [], "elapsed_seconds": elapsed}

        if failed:
            break

        await asyncio.sleep(interval_seconds)

    # Capture logs on failure
    logs_output = ""
    compose_path = os.path.join(workspace_path, compose_file)
    if os.path.isfile(compose_path):
        try:
            proc = await asyncio.create_subprocess_exec(
                *_compose_argv(), "-f", compose_path, "logs", "--tail=100",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                cwd=workspace_path,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            logs_output = stdout.decode("utf-8", errors="replace") or stderr.decode("utf-8", errors="replace")
        except Exception:
            logs_output = "Failed to capture logs."

    diagnostics: list[dict[str, Any]] = []
    if failed:
        for fc in failed:
            diagnostics.append({
                "file": compose_file, "line": 0, "column": 0, "severity": "error",
                "error_code": "DEPLOYMENT_CONTAINER_EXITED",
                "message": f"[DEPLOYMENT FAULT]: Container '{fc['name']}' exited with code {fc['exit_code']}.",
                "semantic_context": f"Exit: {fc['exit_code']} | Error: {fc.get('error', '')}",
            })
    else:
        pending = [s for s in services if "running" not in last_status.get(s, "")]
        diagnostics.append({
            "file": compose_file, "line": 0, "column": 0, "severity": "error",
            "error_code": "DEPLOYMENT_HEALTHCHECK_TIMEOUT",
            "message": f"[DEPLOYMENT FAULT]: Health check timed out. {len(pending)} service(s) not healthy: {', '.join(pending[:5])}.",
            "semantic_context": f"Timeout: {timeout_seconds}s | Services: {', '.join(services)}",
        })

    return {
        "success": False,
        "healthy": [],
        "failed": [fc["name"] for fc in failed],
        "elapsed_seconds": time_module.monotonic() - start,
        "logs": logs_output,
        "diagnostics": diagnostics,
    }


async def deployment_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: Inference-Driven Containerization & Deployment Orchestrator.

    Phases:
        1. Scan workspace telemetry (deterministic, token-free)
        2. Synthesize architecture blueprint via LLM (telemetry + SPEC_ARCHITECTURE.md)
        3. Generate Dockerfiles, docker-compose.yml, Caddyfile from blueprint
        4. Build, launch, and health-check containers

    On failure: populates compiler_errors, increments loop_counter["deployment"].
    """
    deploy_cfg = state.get("deployment_config", {}) or {}
    enabled = deploy_cfg.get("enabled", True)

    if not enabled:
        return {"node_state": {"deployment": {"skipped": True, "reason": "disabled"}}}

    workspace_path = state.get("workspace_path", os.getcwd())
    compose_file = deploy_cfg.get("compose_file", "docker-compose.yml")
    health_interval = deploy_cfg.get("health_check_interval_seconds", 2.0)
    health_timeout = deploy_cfg.get("health_check_timeout_seconds", 30.0)

    logger.info("[deployment_node] Starting inference-driven provisioning...")

    # --- Phase 1: Telemetry ---
    telemetry = scan_workspace_telemetry(workspace_path)
    logger.info("[deployment_node] Phase 1 complete: %d language(s), %d DB(s), %d framework(s).",
                 len(telemetry["languages"]), len(telemetry["databases_detected"]), len(telemetry["frameworks_detected"]))

    # --- Phase 2: Synthesize ---
    blueprint = await synthesize_architecture(telemetry, workspace_path)
    if not blueprint or not blueprint.get("services"):
        return {
            "compiler_errors": [{
                "file": "deployment", "line": 0, "column": 0, "severity": "error",
                "error_code": "DEPLOYMENT_SYNTHESIS_FAILED",
                "message": "[DEPLOYMENT FAULT]: Failed to synthesize architecture blueprint.",
                "semantic_context": str(blueprint),
            }],
            "loop_counter": {"deployment": 1},
            "node_state": {"deployment": {"phase": "synthesis_failed"}},
        }

    logger.info("[deployment_node] Phase 2 complete: %d service(s) in blueprint.", len(blueprint.get("services", {})))

    # --- Phase 3: Generate ---
    # In change-request mode, source per-service CR attribution from
    # either the blueprint (the deployment synthesizer can populate it
    # inline as part of its delta-aware output) or from a state-level
    # override the deployment_discovery_node may set. ``None`` (the
    # default, and the greenfield case) yields byte-identical infra
    # files to pre-change-request behaviour.
    cr_attribution: Optional[dict[str, str]] = None
    if state.get("change_request_mode", False):
        ns = state.get("node_state", {}) or {}
        cr_attribution = (
            ns.get("deployment_cr_attribution")
            or blueprint.get("cr_attribution")
        )
    gen_result = generate_assets_from_blueprint(
        blueprint, telemetry, workspace_path, cr_attribution=cr_attribution,
    )
    if not gen_result.get("success"):
        return {
            "compiler_errors": [{
                "file": "deployment", "line": 0, "column": 0, "severity": "error",
                "error_code": "DEPLOYMENT_GENERATION_FAILED",
                "message": f"[DEPLOYMENT FAULT]: Failed to generate assets. {gen_result.get('message', '')}",
                "semantic_context": str(gen_result),
            }],
            "loop_counter": {"deployment": 1},
            "node_state": {"deployment": {"phase": "generation_failed"}},
        }

    logger.info("[deployment_node] Phase 3 complete: %d file(s) generated.", len(gen_result.get("generated", [])))

    # --- Phase 4: Build & Health Check ---
    compose_path = os.path.join(workspace_path, compose_file)
    if not os.path.isfile(compose_path):
        return {
            "compiler_errors": [{
                "file": compose_file, "line": 0, "column": 0, "severity": "error",
                "error_code": "DEPLOYMENT_NO_COMPOSE_FILE",
                "message": f"[DEPLOYMENT FAULT]: {compose_file} not found after generation.",
                "semantic_context": f"Generated files: {gen_result.get('generated', [])}",
            }],
            "loop_counter": {"deployment": 1},
        }

    # --- Phase 3.5: Preview gate ---
    # The Dockerfile/compose/Caddyfile we're about to execute were synthesized
    # from LLM JSON. A prompt-injected manifest or a confused model can put
    # arbitrary `RUN curl … | sh` in them. Require explicit consent (with an
    # env-var bypass for opted-in CI) before `docker-compose up`.
    preview = _read_preview(workspace_path, gen_result.get("generated", []))
    approved = await _prompt_deploy_approval(preview)
    if not approved:
        logger.info("[deployment_node] User declined deploy preview; aborting before docker-compose up.")
        return {
            "node_state": {
                "deployment": {
                    "skipped": True,
                    "reason": "user_declined_preview",
                    "phase": "preview_gate",
                }
            },
        }

    # Build
    try:
        proc = await asyncio.create_subprocess_exec(
            *_compose_argv(), "-f", compose_path, "up", "--build", "-d",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=workspace_path,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180.0)
        if proc.returncode != 0:
            return {
                "compiler_errors": [{
                    "file": compose_file, "line": 0, "column": 0, "severity": "error",
                    "error_code": "DEPLOYMENT_BUILD_FAILED",
                    "message": f"[DEPLOYMENT FAULT]: docker-compose build failed (exit={proc.returncode}).",
                    "semantic_context": stderr.decode("utf-8", errors="replace")[:500],
                }],
                "loop_counter": {"deployment": 1},
            }
        logger.info("[deployment_node] Container build successful.")
    except asyncio.TimeoutError:
        return {
            "compiler_errors": [{
                "file": compose_file, "line": 0, "column": 0, "severity": "error",
                "error_code": "DEPLOYMENT_BUILD_TIMEOUT",
                "message": "[DEPLOYMENT FAULT]: Build timed out after 180s.",
            }],
            "loop_counter": {"deployment": 1},
        }
    except FileNotFoundError:
        return {
            "compiler_errors": [{
                "file": compose_file, "line": 0, "column": 0, "severity": "error",
                "error_code": "DEPLOYMENT_DOCKER_UNAVAILABLE",
                "message": "[DEPLOYMENT FAULT]: docker-compose not installed.",
            }],
            "loop_counter": {"deployment": 1},
        }

    # Health check
    health_result = await health_check_loop(workspace_path, compose_file, health_interval, health_timeout)

    if health_result["success"]:
        messages = list(state.get("messages", []))
        messages.append({"role": "system", "content": f"[Deployment] All {len(health_result['healthy'])} container(s) healthy."})
        return {
            "messages": messages,
            "node_state": {"deployment": {"success": True, "healthy": health_result["healthy"], "blueprint": blueprint}},
        }

    # Failure
    loop_counter = state.get("loop_counter", {})
    loop_counter = dict(loop_counter)
    loop_counter["deployment"] = loop_counter.get("deployment", 0) + 1

    messages = list(state.get("messages", []))
    status_parts = [f"[Deployment] {len(health_result.get('failed', []))} container(s) failed:"]
    for diag in health_result.get("diagnostics", [])[:3]:
        status_parts.append(f"  - {diag['message']}")
    messages.append({"role": "system", "content": "\n".join(status_parts)})

    return {
        "compiler_errors": health_result.get("diagnostics", []),
        "messages": messages,
        "loop_counter": loop_counter,
        "node_state": {"deployment": {"success": False, "failed": health_result.get("failed", []), "attempt": loop_counter["deployment"]}},
    }


async def teardown_containers(workspace_path: str, compose_file: str = "docker-compose.yml") -> bool:
    """Stop and remove all containers."""
    compose_path = os.path.join(workspace_path, compose_file)
    if not os.path.isfile(compose_path):
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            *_compose_argv(), "-f", compose_path, "down", "--remove-orphans",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=workspace_path,
        )
        await asyncio.wait_for(proc.communicate(), timeout=30.0)
        return True
    except Exception:
        return False