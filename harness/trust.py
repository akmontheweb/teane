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
from typing import Any, Iterable, Optional

from harness import _platform


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
      - **NUL byte** in the path (would terminate C strings early)
      - symlinks pointing outside the workspace (via realpath)
      - **symlinks in any *parent* component** that resolve outside the
        workspace — audit §3.5. realpath alone on a non-existent path
        returns ``workspace_real + filepath`` literally; we explicitly
        walk each existing parent component so a symlink inside the
        workspace cannot redirect a write outside it.
      - Windows mixed-drive joins

    Returns the absolute, real (symlink-resolved) path on success.
    Raises ``ValueError`` on any rejection.
    """
    if not filepath:
        raise ValueError("filepath must be non-empty")
    if "\x00" in filepath:
        raise ValueError("NUL byte rejected in path")
    if os.path.isabs(filepath):
        raise ValueError(f"absolute path rejected: {filepath!r}")

    workspace_real = os.path.realpath(workspace_root)
    joined = os.path.join(workspace_real, filepath)
    candidate = os.path.realpath(joined)

    try:
        common = os.path.commonpath([candidate, workspace_real])
    except ValueError as e:
        raise ValueError(f"unresolvable path: {filepath!r}") from e

    if common != workspace_real:
        raise ValueError(f"path escapes workspace: {filepath!r} -> {candidate}")

    # Per-component symlink walk (audit §3.5): catch the case where any
    # existing parent component is itself a symlink pointing outside the
    # workspace. ``os.path.realpath`` on a non-existent leaf returns
    # ``<resolved-prefix>/<literal-tail>``, so a malicious symlink
    # ``workspace/proxy → /etc`` followed by a CREATE_FILE of
    # ``proxy/new.txt`` would *appear* to land inside the workspace
    # because ``new.txt`` doesn't exist yet. Resolving each existing
    # component individually catches it.
    parts = []
    head, tail = os.path.split(joined)
    while head and head != workspace_real and head not in parts:
        parts.append(head)
        head, _ = os.path.split(head)
    for component in parts:
        if not os.path.lexists(component):
            continue
        try:
            real_component = os.path.realpath(component)
        except OSError:
            continue
        try:
            shared = os.path.commonpath([real_component, workspace_real])
        except ValueError:
            raise ValueError(f"unresolvable path: {filepath!r}")
        if shared != workspace_real:
            raise ValueError(
                f"parent symlink escapes workspace: {component} -> "
                f"{real_component} (rejecting {filepath!r})"
            )

    return candidate


def is_path_allowed(
    filepath: str,
    workspace_root: str,
    allowed_paths: Optional["Iterable[str]"],
) -> bool:
    """
    Check whether ``filepath`` falls within the optional allowlist.

    Each entry in ``allowed_paths`` is treated as a workspace-relative
    file path or directory prefix. A file matches if its resolved
    workspace-relative form equals an allowlist entry exactly, or starts
    with an allowlist entry that ends in ``/`` (or matches a directory).

    Returns:
      - True when ``allowed_paths`` is None or empty (no restriction).
      - True when the file is inside at least one allowlist entry.
      - False otherwise.

    The function never raises — invalid inputs simply return False. Callers
    that need a hard error should call ``safe_resolve`` first.
    """
    if not allowed_paths:
        return True

    try:
        resolved = safe_resolve(workspace_root, filepath)
    except ValueError:
        return False  # path escapes workspace — not allowed regardless

    workspace_real = os.path.realpath(workspace_root)
    rel = os.path.relpath(resolved, workspace_real).replace(os.sep, "/")

    for entry in allowed_paths:
        if not isinstance(entry, str) or not entry:
            continue
        # removeprefix — not lstrip — because lstrip("./") strips any
        # leading "." or "/" character, mangling dotfile entries like
        # ".env.example" into "env.example" so they never match on disk.
        e = entry.strip().removeprefix("./").replace(os.sep, "/")
        if not e:
            continue
        # Exact file match
        if rel == e:
            return True
        # Directory prefix: "src/" or "src/auth"
        prefix = e if e.endswith("/") else e + "/"
        if rel.startswith(prefix):
            return True

    return False


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
# Total response-size cap. A well-formed discovery response is a few KB;
# 1 MB is two orders of magnitude above that, so the cap only triggers on
# pathological input (malicious LLM, runaway generation).
_MAX_DISCOVERY_BYTES = 1_000_000
# Recursion depth cap. Discovery JSON is shallow (object -> modules -> questions
# -> primitives, depth ~4). Python's default recursion limit is 1000, so a
# nested response could DoS the parser before our other guards see it.
_MAX_DISCOVERY_DEPTH = 10


def _json_depth(node: Any, _seen: Optional[set[int]] = None) -> int:
    """Return the maximum nesting depth of a parsed JSON tree.

    Cycle-safe via an id() set so a malformed object graph can't OOM us.
    A primitive is depth 0; ``{"a": 1}`` is depth 1; ``{"a": [1]}`` is depth 2.
    """
    if _seen is None:
        _seen = set()
    if id(node) in _seen:
        return 0
    if isinstance(node, dict):
        _seen.add(id(node))
        if not node:
            return 1
        return 1 + max(_json_depth(v, _seen) for v in node.values())
    if isinstance(node, list):
        _seen.add(id(node))
        if not node:
            return 1
        return 1 + max(_json_depth(v, _seen) for v in node)
    return 0


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

    # Pre-flight size guard. UTF-8 string length in bytes is the right unit —
    # JSON parsing allocates roughly that much memory plus parser overhead.
    if len(content.encode("utf-8", errors="replace")) > _MAX_DISCOVERY_BYTES:
        errors.append(
            f"discovery response exceeds {_MAX_DISCOVERY_BYTES} bytes — refusing to parse"
        )
        return {}, errors

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

    depth = _json_depth(data)
    if depth > _MAX_DISCOVERY_DEPTH:
        errors.append(
            f"discovery response nesting depth {depth} exceeds {_MAX_DISCOVERY_DEPTH}"
        )
        return data, errors

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

# Non-ASCII code-point ratio cap. An English SPEC_REQUIREMENTS.md with the
# usual smart-quote / em-dash / arrow flourishes stays well under 1%
# non-ASCII. When a planning model drifts and renders the whole document
# in Japanese / Chinese / Arabic / etc. (see 2026-07-10 finsearch incident:
# input spec was English, output was 100% Japanese) the ratio jumps past
# 30% instantly. 5% leaves headroom for legitimate Unicode punctuation
# without missing a full-document translation.
_MAX_SPEC_NON_ASCII_RATIO = 0.05
_NON_ASCII_SAMPLE_LEN = 80

# Requirement-ID families declared by the requirements_doc skill's "ID
# numbering convention" section. Every valid ID in each family is
# ``<PREFIX>-<zero-padded integer>`` — no letter suffix, no decimal, no
# dotted child. The regex below catches the malformed shapes at the
# trust boundary; the source-level fix lives in the skill prompt itself
# (see harness/skills/docgen/requirements_doc.md Gate 3 + ID convention).
# Failure mode: LLM emits ``STORY-011B`` or ``FR-014.2`` when splitting a
# larger requirement instead of allocating the next integer from the
# global sequence. Commit 018cf92 fixed the same shape at the
# decomposition prompt; this catches it one hop earlier at spec synthesis.
_MALFORMED_REQUIREMENT_ID_RE = re.compile(
    r"\b(?:"
    r"EPIC|FEAT|STORY(?:-NFR)?|FR|"
    r"NFR-(?:PERF|SEC|AVAIL|SCALE|MAINT|COMP)|"
    r"UC|TEST(?:-NFR)?"
    r")-\d+(?:[A-Za-z]|\.\d)\w*"
)


# Structural heading pattern used by the requirements_doc skill: the four
# SAFe labels (Epic / Feature / Story / Enabler Story) followed by an
# identifier token. Captures the token so we can whitelist its prefix
# against the canonical seven families the harness recognises. See the
# 2026-07-10 finsearch incident: the planning model rendered an enabler
# story as ``### Enabler Story: EN-001 — Fiscal Year Window Calculation``
# — a bespoke ``EN-`` prefix that neither the requirements ingest
# (``req_ids._HEADING_RE``) nor the reconciler (``spec_reconciler._STORY_RE``)
# knew how to parse, but which the planner echoed back verbatim as a
# ``story_key`` and blew up the decomposition validator at HITL.
_STRUCTURAL_HEADING_RE = re.compile(
    r"^\s*#{2,6}\s+"
    r"(?:Epic|Feature|Story|Enabler\s+Story)\s*:\s+"
    r"(?P<id>[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9])",
    re.MULTILINE,
)

# Canonical requirement-ID families the harness understands end-to-end
# (see ``harness/req_ids.py`` module docstring). Anything else must be
# rejected at the trust boundary so the bad spec never lands on disk.
_KNOWN_STRUCTURAL_ID_RE = re.compile(
    r"^(?:"
    r"EPIC-\d{1,4}"
    r"|FEAT-\d{1,4}"
    r"|STORY-NFR-\d{1,4}"
    r"|STORY-\d{1,4}"
    r"|FR-\d{1,4}"
    r"|NFR-[A-Z]+-\d{1,4}"
    r"|US-\d{1,3}-\d{1,3}"
    r")$"
)


def validate_synthesized_spec(content: str) -> tuple[str, list[str]]:
    """
    Validate a synthesised SPEC_REQUIREMENTS.md / SPEC_ARCHITECTURE.md string.

    Checks:
      - Not empty
      - UTF-8 encodable (should already be — belt-and-suspenders)
      - No NUL bytes or C0 control chars (except LF, CR, TAB)
      - Within the 256 KB length cap
      - No requirement IDs with letter suffixes or decimal extensions
      - Non-ASCII ratio under 5% (catches planning-model language drift —
        an English input yielding a Japanese/Chinese/etc. spec)
      - Every Epic/Feature/Story/Enabler-Story heading uses an ID from
        one of the seven canonical families (catches novel prefixes like
        ``EN-`` that the reconciler and requirements ingest can't parse
        but the planner still echoes back as ``story_key``)

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

    # Normalise Unicode hyphen/dash variants (U+2011 non-breaking
    # hyphen, U+2013 en dash, …) to ASCII so the malformed-ID regex
    # can't be bypassed by an LLM that stylises identifiers with
    # non-ASCII dashes. Same fix applied at the parser and validator
    # layers — see ``req_ids.normalize_dashes``.
    from harness.req_ids import normalize_dashes
    bad_ids = _MALFORMED_REQUIREMENT_ID_RE.findall(normalize_dashes(content))
    if bad_ids:
        unique = sorted(set(bad_ids))
        sample = ", ".join(unique[:5])
        overflow = f" (+ {len(unique) - 5} more)" if len(unique) > 5 else ""
        errors.append(
            "spec contains requirement IDs with letter suffixes or decimal "
            "extensions — every ID must be <PREFIX>-<zero-padded integer> "
            f"with no extension: {sample}{overflow}"
        )

    # Novel-prefix guard for structural headings. Every Epic/Feature/
    # Story/Enabler-Story heading in the spec must use an ID from the
    # seven canonical families the harness parses end-to-end. See the
    # 2026-07-10 finsearch incident (``EN-001`` for an enabler story
    # instead of the mandated ``STORY-NFR-001``): the ingest and
    # reconciler silently skipped it, but the planner echoed it back as
    # a story_key and blew up the decomposition validator at HITL.
    normalised = normalize_dashes(content)
    bad_prefix_ids = [
        m.group("id")
        for m in _STRUCTURAL_HEADING_RE.finditer(normalised)
        if not _KNOWN_STRUCTURAL_ID_RE.match(m.group("id"))
    ]
    if bad_prefix_ids:
        unique = sorted(set(bad_prefix_ids))
        sample = ", ".join(unique[:5])
        overflow = f" (+ {len(unique) - 5} more)" if len(unique) > 5 else ""
        errors.append(
            "spec contains Epic/Feature/Story headings with unrecognised "
            "ID prefixes — every ID must belong to one of the seven "
            "canonical families (EPIC, FEAT, STORY, STORY-NFR, FR, "
            "NFR-<CAT>, US). Enabler stories use STORY-NFR-NNN, not "
            f"bespoke prefixes: {sample}{overflow}"
        )

    # Language-drift guard. Fires when the planning model renders the
    # whole document in a non-Latin script even though the input notes
    # and skill prompt are English. See 2026-07-10 finsearch incident:
    # docs/SPEC_REQUIREMENTS.md came back 100% Japanese from an English
    # product spec. The skill prompt now pins the output language, and
    # this check is the enforcement backstop.
    non_ascii = sum(1 for ch in content if ord(ch) >= 0x80)
    if non_ascii and (non_ascii / len(content)) > _MAX_SPEC_NON_ASCII_RATIO:
        first_idx = next(i for i, ch in enumerate(content) if ord(ch) >= 0x80)
        sample = content[
            max(0, first_idx - 10) : first_idx + _NON_ASCII_SAMPLE_LEN
        ]
        ratio_pct = (non_ascii / len(content)) * 100
        errors.append(
            f"spec is {ratio_pct:.1f}% non-ASCII (cap "
            f"{_MAX_SPEC_NON_ASCII_RATIO * 100:.0f}%) — planning model likely "
            f"drifted to a non-English language. First non-ASCII segment: "
            f"{sample!r}"
        )

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
    # Audit §3.16: scrub these too — a build doesn't need them by default,
    # and leaving them through lets a hostile LLM-generated test exfiltrate
    # the operator's host kubeconfig, ssh-agent socket, or proxy creds.
    "SSH_AUTH_SOCK", "SSH_AGENT_PID",
    "KUBECONFIG",
})


# Prefix patterns (case-insensitive) for env vars to scrub. Useful when
# the family is open-ended (HTTP_PROXY, HTTPS_PROXY, NO_PROXY, ALL_PROXY,
# plus their lowercase forms; *_COOKIES; etc.). Audit §3.16.
_SCRUBBED_BUILD_ENV_PREFIXES: tuple[str, ...] = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
)


def safe_subprocess_env(extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    """
    Return a copy of os.environ with known credential variables stripped,
    then merge in ``extra`` (which may re-add specific secrets the build
    legitimately needs).

    Every subprocess runner in the harness should call this instead of
    ``os.environ.copy()`` directly.
    """
    env = {
        k: v for k, v in os.environ.items()
        if k not in SCRUBBED_BUILD_ENV_VARS
        and not any(k == p or k.startswith(p + "_") for p in _SCRUBBED_BUILD_ENV_PREFIXES)
    }
    if extra:
        env.update(extra)
    return env


# ---------------------------------------------------------------------------
# Outbound URL guard (web tools, MCP HTTP transports)
# ---------------------------------------------------------------------------

def _ip_in_private_range(host: str) -> bool:
    """Return True when ``host`` is an IP literal inside a non-routable or
    cloud-metadata range — RFC 1918, link-local (169.254/16), loopback
    (127/8), or 0.0.0.0/8. Returns False for hostnames (they're resolved
    by :func:`_resolved_addresses_are_safe`) and for malformed IPs (so the
    caller decides).
    """
    import ipaddress
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.is_loopback or ip.is_link_local or ip.is_private or ip.is_unspecified:
        return True
    # AWS / GCP / Azure instance metadata services all live on
    # 169.254.169.254 — already caught by is_link_local, but spelled out
    # here for the reader who's reviewing this for SSRF safety.
    return False


def _resolve_host_addresses(host: str) -> list[str]:
    """Resolve ``host`` to all of its address families (A + AAAA).

    Returns a list of textual addresses. Raises ``ValueError`` on
    resolution failure. The caller should pass each address through
    :func:`_ip_in_private_range` to decide whether the host is safe.

    Pre-resolving and inspecting every result closes the DNS-rebinding
    audit hole (§3.2): the earlier guard accepted any hostname and let
    httpx do the resolution opaquely, so ``evil.example.com → A 10.0.0.1``
    or ``→ A 169.254.169.254`` got through.
    """
    import socket
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"DNS resolution failed for {host!r}: {exc}") from exc
    out: list[str] = []
    for family, _socktype, _proto, _canonname, sockaddr in infos:
        if not sockaddr:
            continue
        addr = sockaddr[0]
        if family == socket.AF_INET6:
            # sockaddr is (host, port, flowinfo, scopeid) — first element is
            # the IPv6 textual address.
            addr = sockaddr[0]
        if addr and addr not in out:
            out.append(str(addr))
    return out


def validate_outbound_url(
    url: str,
    *,
    allow_private_ips: bool = False,
    allowed_schemes: tuple[str, ...] = ("http", "https"),
    resolve_dns: bool = True,
) -> str:
    """Validate an LLM-supplied outbound URL before the harness fetches it.

    Designed for use by ``WebFetchSkill`` / ``WebSearchSkill`` (and any
    future HTTP-transport MCP client) so the LLM cannot trick the harness
    into hitting cloud-metadata endpoints (SSRF), localhost services, or
    internal RFC-1918 hosts when ``allow_private_ips`` is False.

    Guards against:
        - Empty / non-string URLs
        - Schemes outside the whitelist (default ``http`` / ``https``)
          — explicitly rejects ``file://``, ``gopher://``, ``ftp://``,
          ``data:``, ``javascript:``
        - IP literals inside loopback / link-local / RFC-1918 / unspecified
          ranges unless ``allow_private_ips=True``
        - Hostnames ``localhost`` / ``localhost.localdomain``
        - **Hostnames whose DNS resolves to private/loopback/link-local
          addresses** when ``resolve_dns=True`` (audit §3.2). Set to False
          only for tests that don't want to hit the network.
        - Missing netloc

    Returns the URL unchanged on success.
    Raises ``ValueError`` on any rejection.
    """
    from urllib.parse import urlparse

    if not isinstance(url, str) or not url.strip():
        raise ValueError("url must be a non-empty string")
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in allowed_schemes:
        raise ValueError(
            f"scheme {parsed.scheme!r} not in allowlist {allowed_schemes}"
        )
    if not parsed.netloc:
        raise ValueError(f"url missing host: {url!r}")
    host = parsed.hostname or ""
    if not host:
        raise ValueError(f"url has no parseable host: {url!r}")
    host_lower = host.lower()
    if not allow_private_ips:
        if host_lower in ("localhost", "localhost.localdomain", "ip6-localhost"):
            raise ValueError(f"localhost rejected: {url!r}")
        if _ip_in_private_range(host_lower):
            raise ValueError(
                f"private/loopback/link-local IP rejected (SSRF guard): {url!r}"
            )
        # DNS rebinding guard (§3.2): for non-IP hosts, resolve and check
        # every A / AAAA. We accept the (small) cost of one synchronous
        # getaddrinfo call here in exchange for closing the cloud-metadata
        # primitive. Skipped only when the caller explicitly opts out.
        try:
            import ipaddress as _ipa
            _ipa.ip_address(host_lower)
            is_literal_ip = True
        except ValueError:
            is_literal_ip = False
        if (not is_literal_ip) and resolve_dns:
            try:
                addrs = _resolve_host_addresses(host_lower)
            except ValueError:
                # Resolution failure is itself a refusal — better to fail
                # closed than let httpx silently resolve to whatever its
                # own resolver picks.
                raise
            for addr in addrs:
                if _ip_in_private_range(addr):
                    raise ValueError(
                        f"hostname {host_lower!r} resolves to a "
                        f"private/loopback/link-local address ({addr}) — "
                        f"refusing to fetch (SSRF / DNS-rebinding guard)."
                    )
    return url


# ---------------------------------------------------------------------------
# MCP server command allowlist
# ---------------------------------------------------------------------------

# Commands the harness will agree to spawn as MCP servers. Deliberately
# narrow: anything that lets the LLM ship arbitrary shell payloads
# (``bash -c``, ``sh -c``, ``sudo``) is rejected. The allowlist covers
# the four ways MCP servers ship in practice today:
#   - ``npx`` / ``npm`` exec for Node packages (the dominant pattern)
#   - ``node`` for hand-written Node scripts
#   - ``python`` / ``python3`` / ``uvx`` / ``pipx`` for Python servers
#   - ``docker`` for containerized servers (``docker run --rm -i …``)
# Operators wanting a custom binary outside this list must add it to
# ``mcp.command_allowlist`` in config — explicit opt-in only.
_MCP_DEFAULT_COMMAND_ALLOWLIST = frozenset({
    "npx", "npm", "node",
    "python", "python3", "uvx", "pipx",
    "docker",
})

# Hard-deny: even when the operator extends the allowlist, these names
# remain forbidden. The LLM can synthesize a server config with any name
# in the prompt that produced it, so we keep a backstop list.
_MCP_HARD_DENY = frozenset({
    "sudo", "su", "doas",
    "sh", "bash", "zsh", "fish", "ksh", "dash",
    "eval", "exec",
    "rm", "mv", "dd",
    "/bin/sh", "/bin/bash", "/bin/zsh",
})


def validate_mcp_server_command(
    cmd: list[str],
    *,
    extra_allowlist: Optional["Iterable[str]"] = None,
) -> list[str]:
    """Validate a list-form MCP server launch command (e.g. ``["npx", "-y",
    "@modelcontextprotocol/server-filesystem", "/workspace"]``).

    Rejected:
      - non-list / empty inputs
      - any binary not in the allowlist (default + operator-provided)
      - any binary in the hard-deny list, regardless of allowlist
      - absolute paths under ``/etc``, ``/root``, ``/proc``, ``/sys``
        (operators can still launch a server by absolute path under
        ``/usr/local/bin`` etc., if they extend the allowlist with the
        basename of their binary)
      - shell metacharacters in any argv element — ``;``, ``|``, ``&``,
        ``$`` followed by ``(``, ``` ` ``` — anything that would let the
        argv smuggle a second command past a naive shell

    Returns the validated command unchanged on success.
    Raises ``ValueError`` on any rejection. Never executes anything.
    """
    if not isinstance(cmd, list) or not cmd:
        raise ValueError("mcp command must be a non-empty list of strings")
    if not all(isinstance(arg, str) and arg for arg in cmd):
        raise ValueError("mcp command args must all be non-empty strings")

    head = cmd[0]
    basename = os.path.basename(head)
    if basename in _MCP_HARD_DENY or head in _MCP_HARD_DENY:
        raise ValueError(f"mcp command rejected (hard deny): {head!r}")

    allow = set(_MCP_DEFAULT_COMMAND_ALLOWLIST)
    if extra_allowlist:
        for extra in extra_allowlist:
            if isinstance(extra, str) and extra:
                allow.add(extra)
    if basename not in allow and head not in allow:
        raise ValueError(
            f"mcp command {head!r} not in allowlist {sorted(allow)}. "
            f"Add to mcp.command_allowlist in config to opt in explicitly."
        )

    if os.path.isabs(head):
        for forbidden in ("/etc/", "/root/", "/proc/", "/sys/"):
            if head.startswith(forbidden):
                raise ValueError(
                    f"mcp command path under {forbidden!r} rejected: {head!r}"
                )
        if _platform.is_windows():
            # Reject the Windows equivalents of /etc, /proc, /sys, /root.
            # Compare case-insensitively because NTFS is case-insensitive
            # and an attacker could otherwise bypass via case variation.
            head_lc = head.lower().replace("/", "\\")
            for forbidden in (
                "c:\\windows\\",
                "c:\\program files\\",
                "c:\\program files (x86)\\",
                "c:\\programdata\\",
            ):
                if head_lc.startswith(forbidden):
                    raise ValueError(
                        f"mcp command path under {forbidden!r} rejected: {head!r}"
                    )

    # Shell-metacharacter scan over every argv element. We're not going
    # through a shell, but a sloppy server config could still smuggle
    # a payload if anything downstream `shell=True`'s. Defence in depth.
    bad_chars_re = re.compile(r"[;|&`\n\r]|\$\(")
    for i, arg in enumerate(cmd):
        if bad_chars_re.search(arg):
            raise ValueError(
                f"mcp command argv[{i}] contains shell metacharacters: {arg!r}"
            )

    return cmd


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences (```json ... ```).

    This is the canonical implementation. Other modules previously
    hand-rolled their own (``decomposition.strip_json_fence``, inline
    regexes in ``graph.py`` / ``deploy.py``); prefer this one for any
    new caller and migrate old sites incrementally.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1
        end = len(lines)
        if lines[-1].strip() == "```":
            end = -1
        text = "\n".join(lines[start:end])
    return text


# Backwards-compatible private alias. Existing call sites in this module
# (and the explicit import in graph.py) continue to work; new code should
# import the public ``strip_code_fences`` name.
_strip_code_fences = strip_code_fences
