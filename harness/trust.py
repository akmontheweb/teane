"""
Central LLM-output trust boundary.

Every place the harness consumes a string produced by an LLM — patch blocks,
architecture blueprints, discovery JSON, synthesised specs, subprocess
environments — should be validated here before the output touches the
filesystem, containers, or configuration.

This module consolidates:
  - Path traversal guards (moved from patcher.py)
  - Blueprint / identifier validators (moved from deploy.py)
  - New validators for discovery JSON and synthesised spec content
  - Subprocess environment scrubbing (moved from sandbox.py)

Re-exports are provided so existing callers (patcher, deploy, sandbox) can
continue to import from their original locations without change — they just
delegate to this module internally.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional


# ---------------------------------------------------------------------------
# 1. Path Traversal Guard
# ---------------------------------------------------------------------------

def safe_resolve(workspace_root: str, filepath: str) -> str:
    """
    Resolve an LLM-supplied ``filepath`` against ``workspace_root`` and
    reject anything that would land outside the workspace.

    Guards against:
      - empty / None paths
      - absolute paths (``/etc/passwd``)
      - parent-traversal (``../../etc/passwd``)
      - symlinks pointing outside the workspace (via realpath)
      - Windows mixed-drive joins

    Returns the absolute, real (symlink-resolved) path on success.
    Raises ``ValueError`` on any rejection.
    """
    if not filepath:
        raise ValueError("filepath must be non-empty")
    if os.path.isabs(filepath):
        raise ValueError(f"absolute path rejected: {filepath!r}")

    workspace_real = os.path.realpath(workspace_root)
    candidate = os.path.realpath(os.path.join(workspace_real, filepath))

    try:
        common = os.path.commonpath([candidate, workspace_real])
    except ValueError as e:
        raise ValueError(f"unresolvable path: {filepath!r}") from e

    if common != workspace_real:
        raise ValueError(f"path escapes workspace: {filepath!r} -> {candidate}")

    return candidate


# ---------------------------------------------------------------------------
# 2. Docker / Compose / Caddyfile Identifier Validators
# ---------------------------------------------------------------------------

_VALID_DOCKER_IMAGE_RE = re.compile(
    r"^[a-z0-9]+(?:[._-][a-z0-9]+)*"
    r"(?::\d+)?"
    r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*"
    r"(?::[a-zA-Z0-9_][a-zA-Z0-9._-]{0,127})?"
    r"(?:@sha256:[a-f0-9]{64})?$"
)
_VALID_SERVICE_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,62}$")
_VALID_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_VALID_PORT_MAPPING_RE = re.compile(r"^(?:\d+:)?\d+(?:/(?:tcp|udp))?$")


def is_valid_docker_image(name: Any) -> bool:
    return isinstance(name, str) and bool(_VALID_DOCKER_IMAGE_RE.match(name))


def is_valid_service_name(name: Any) -> bool:
    return isinstance(name, str) and bool(_VALID_SERVICE_NAME_RE.match(name))


def is_valid_env_var_name(name: Any) -> bool:
    return isinstance(name, str) and bool(_VALID_ENV_VAR_NAME_RE.match(name))


def is_valid_port_mapping(port: Any) -> bool:
    if isinstance(port, int):
        return 0 < port < 65536
    return isinstance(port, str) and bool(_VALID_PORT_MAPPING_RE.match(port))


def validate_blueprint(blueprint: dict[str, Any]) -> list[str]:
    """
    Walk an architecture blueprint dict and return a list of validation errors.
    Empty list means the blueprint is safe to render into Dockerfile / compose /
    Caddyfile templates.
    """
    errors: list[str] = []
    services = blueprint.get("services", {})
    if not isinstance(services, dict):
        errors.append("services must be a dict")
        return errors

    for svc_name, svc_spec in services.items():
        if not is_valid_service_name(svc_name):
            errors.append(f"invalid service name: {svc_name!r}")
            continue
        if not isinstance(svc_spec, dict):
            errors.append(f"service {svc_name!r}: spec must be a dict")
            continue

        if svc_spec.get("base_image") and not is_valid_docker_image(svc_spec["base_image"]):
            errors.append(f"service {svc_name!r}: invalid base_image {svc_spec['base_image']!r}")

        for port in svc_spec.get("ports", []) or []:
            if not is_valid_port_mapping(port):
                errors.append(f"service {svc_name!r}: invalid port mapping {port!r}")

        for env_key in svc_spec.get("environment_keys_needed", []) or []:
            if not is_valid_env_var_name(env_key):
                errors.append(f"service {svc_name!r}: invalid env var name {env_key!r}")

        for dep in svc_spec.get("depends_on_services", []) or []:
            if not is_valid_service_name(dep):
                errors.append(f"service {svc_name!r}: invalid depends_on entry {dep!r}")

    return errors


# ---------------------------------------------------------------------------
# 3. Discovery JSON Validator
# ---------------------------------------------------------------------------

# Maximum allowed length for any single question text field.
_MAX_QUESTION_TEXT_LEN = 10_000
# Maximum number of modules in a discovery response.
_MAX_MODULES = 50


def validate_discovery_json(content: str) -> tuple[dict[str, Any], list[str]]:
    """
    Parse and validate a discovery JSON response from the LLM.

    Expected shape:
        {
          "modules": [
            {
              "name": "...",
              "questions": [{"id": "Q1.1", "text": "...", "critical": false}]
            }
          ],
          "complete": false,
          "summary": "..."   # optional
        }

    Returns:
        (parsed_dict, errors) — errors is empty on success.
    """
    errors: list[str] = []

    # Strip code fences (LLMs sometimes wrap JSON in ```json ... ```)
    stripped = _strip_code_fences(content)
    if not stripped.strip():
        errors.append("discovery response is empty")
        return {}, errors

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as e:
        errors.append(f"discovery response is not valid JSON: {e}")
        return {}, errors

    if not isinstance(data, dict):
        errors.append("discovery response must be a JSON object")
        return {}, errors

    # "complete" is the most critical field — must be a bool
    if "complete" in data and not isinstance(data["complete"], bool):
        errors.append(f"'complete' must be a boolean, got {type(data['complete']).__name__}")

    modules = data.get("modules", [])
    if not isinstance(modules, list):
        errors.append("'modules' must be a list")
        return data, errors

    if len(modules) > _MAX_MODULES:
        errors.append(f"too many modules ({len(modules)} > {_MAX_MODULES})")

    for i, module in enumerate(modules):
        if not isinstance(module, dict):
            errors.append(f"modules[{i}] must be an object")
            continue
        for j, q in enumerate(module.get("questions", []) or []):
            if not isinstance(q, dict):
                errors.append(f"modules[{i}].questions[{j}] must be an object")
                continue
            text = q.get("text", "")
            if isinstance(text, str) and len(text) > _MAX_QUESTION_TEXT_LEN:
                errors.append(
                    f"modules[{i}].questions[{j}].text exceeds "
                    f"{_MAX_QUESTION_TEXT_LEN} chars — possible injection"
                )
            # Reject NUL bytes in any string field
            for key in ("id", "text"):
                val = q.get(key, "")
                if isinstance(val, str) and "\x00" in val:
                    errors.append(
                        f"modules[{i}].questions[{j}].{key} contains NUL byte"
                    )

    return data, errors


# ---------------------------------------------------------------------------
# 4. Blueprint JSON Validator
# ---------------------------------------------------------------------------

def validate_blueprint_json(content: str) -> tuple[dict[str, Any], list[str]]:
    """
    Parse a blueprint JSON string, strip code fences, then run
    validate_blueprint over the parsed dict.

    Returns:
        (blueprint_dict, errors) — errors is empty on success.
    """
    errors: list[str] = []
    stripped = _strip_code_fences(content)
    if not stripped.strip():
        errors.append("blueprint response is empty")
        return {}, errors

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as e:
        errors.append(f"blueprint is not valid JSON: {e}")
        return {}, errors

    if not isinstance(data, dict):
        errors.append("blueprint must be a JSON object")
        return {}, errors

    validation_errors = validate_blueprint(data)
    errors.extend(validation_errors)
    return data, errors


# ---------------------------------------------------------------------------
# 5. Synthesised Spec Validator
# ---------------------------------------------------------------------------

_MAX_SPEC_BYTES = 256 * 1024  # 256 KB


def validate_synthesized_spec(content: str) -> tuple[str, list[str]]:
    """
    Validate a synthesised SPEC_REQUIREMENTS.md / SPEC_ARCHITECTURE.md string.

    Checks:
      - Not empty
      - UTF-8 encodable (should already be — belt-and-suspenders)
      - No NUL bytes or C0 control chars (except LF, CR, TAB)
      - Within the 256 KB length cap

    Returns:
        (validated_content, errors)
    """
    errors: list[str] = []

    if not content or not content.strip():
        errors.append("synthesised spec is empty")
        return content, errors

    try:
        encoded = content.encode("utf-8")
    except UnicodeEncodeError as e:
        errors.append(f"spec contains non-UTF-8 characters: {e}")
        return content, errors

    if len(encoded) > _MAX_SPEC_BYTES:
        errors.append(
            f"spec exceeds {_MAX_SPEC_BYTES // 1024} KB "
            f"({len(encoded) // 1024} KB) — possible injection or runaway output"
        )

    # Detect NUL bytes and other C0 control chars outside LF/CR/TAB
    for i, ch in enumerate(content):
        cp = ord(ch)
        if cp == 0 or (cp < 0x20 and ch not in "\n\r\t"):
            errors.append(
                f"spec contains control character U+{cp:04X} at position {i}"
            )
            break  # report once — don't flood

    return content, errors


# ---------------------------------------------------------------------------
# 6. Subprocess Environment Scrubbing
# ---------------------------------------------------------------------------

# Variables stripped from the build process's inherited environment before
# subprocess launch. Any subprocess-spawner (sandbox, speculative, skills)
# should use safe_subprocess_env() instead of os.environ.copy() directly.
SCRUBBED_BUILD_ENV_VARS: frozenset[str] = frozenset({
    # LLM providers
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
    "MISTRAL_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY",
    "COHERE_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
    "GROQ_API_KEY", "TOGETHER_API_KEY", "PERPLEXITY_API_KEY",
    # Cloud provider credentials
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "GCP_SERVICE_ACCOUNT_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
    "AZURE_CLIENT_SECRET", "AZURE_TENANT_ID",
    # VCS tokens
    "GITHUB_TOKEN", "GH_TOKEN", "GITLAB_TOKEN",
    # Package registry / SaaS
    "NPM_TOKEN", "PYPI_TOKEN", "STRIPE_SECRET_KEY", "SLACK_TOKEN",
    "DATABASE_URL",  # may carry credentials; re-export explicitly if needed
})


def safe_subprocess_env(extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    """
    Return a copy of os.environ with known credential variables stripped,
    then merge in ``extra`` (which may re-add specific secrets the build
    legitimately needs).

    Every subprocess runner in the harness should call this instead of
    ``os.environ.copy()`` directly.
    """
    env = {k: v for k, v in os.environ.items() if k not in SCRUBBED_BUILD_ENV_VARS}
    if extra:
        env.update(extra)
    return env


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences (```json ... ```)."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1
        end = len(lines)
        if lines[-1].strip() == "```":
            end = -1
        text = "\n".join(lines[start:end])
    return text
