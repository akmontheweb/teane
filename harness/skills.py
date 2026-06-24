"""
Unified Skill Registry — Tools, Pipelines, Sub-Agents, and Documentation Skills.

This module implements:
    - SkillBase: ABC for all skill types
    - ToolSkill: LLM-invokable function (function-calling)
    - PipelineSkill: LangGraph node wrapper
    - SubAgentSkill: Autonomous mini-agent with own prompt and execution loop
    - DocGenSkill: Specialized sub-agent for generating project documentation
    - SkillRegistry: Global singleton for registration, discovery, and dispatch

Documentation Skills (SubAgentSkills):
    - arch_doc_generator: Architecture Decision Record (ADR / C4)
    - functional_spec_generator: Functional Specification document
    - requirements_doc_generator: Requirements Specification with traceability
    - api_doc_generator: API Reference from route/schema analysis
    - readme_generator: Project README.md with install/usage/contributing

Integration:
    - Gateway: exposes ToolSkills as function-calling schemas
    - Graph: PipelineSkills registered as LangGraph nodes
    - CLI: Documentation skills invoked via --prompt "Generate arch doc"
"""

from __future__ import annotations

import logging
import os as _os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Types
# ---------------------------------------------------------------------------

class SkillType(str, Enum):
    TOOL = "tool"
    PIPELINE = "pipeline"
    SUBAGENT = "subagent"
    DOCGEN = "docgen"


@dataclass
class SkillParameter:
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    enum: Optional[list[str]] = None


@dataclass
class SkillSchema:
    name: str
    description: str
    skill_type: SkillType
    parameters: list[SkillParameter] = field(default_factory=list)
    returns_description: str = ""
    tags: list[str] = field(default_factory=list)
    module: str = ""
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# 2. SkillBase ABC
# ---------------------------------------------------------------------------

class SkillBase(ABC):
    def __init__(self, schema: SkillSchema):
        self.schema = schema

    @property
    def name(self) -> str:
        return self.schema.name

    @property
    def skill_type(self) -> SkillType:
        return self.schema.skill_type

    def to_tool_schema(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        ...


# ---------------------------------------------------------------------------
# 3. ToolSkill
# ---------------------------------------------------------------------------

class ToolSkill(SkillBase):
    def __init__(self, schema: SkillSchema, fn: Callable[..., Awaitable[Any]]):
        super().__init__(schema)
        self._fn = fn

    def to_tool_schema(self) -> dict[str, Any]:
        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []
        for param in self.schema.parameters:
            prop: dict[str, Any] = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)
        return {
            "type": "function",
            "function": {
                "name": self.schema.name,
                "description": self.schema.description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        }

    async def execute(self, **kwargs: Any) -> Any:
        try:
            return await self._fn(**kwargs)
        except Exception as exc:
            logger.exception("[skills] Tool '%s' failed.", self.schema.name)
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# 4. PipelineSkill
# ---------------------------------------------------------------------------

class PipelineSkill(SkillBase):
    def __init__(self, schema: SkillSchema, node_fn: Callable[..., Awaitable[dict[str, Any]]]):
        super().__init__(schema)
        self._node_fn = node_fn

    def to_node_fn(self) -> Callable[..., Awaitable[dict[str, Any]]]:
        return self._node_fn

    async def execute(self, **kwargs: Any) -> Any:
        state = kwargs.get("state", {})
        try:
            return await self._node_fn(state)
        except Exception as exc:
            logger.exception("[skills] Pipeline '%s' failed.", self.schema.name)
            return {"node_state": {"error": str(exc)}}


# ---------------------------------------------------------------------------
# 5. SubAgentSkill
# ---------------------------------------------------------------------------

class SubAgentSkill(SkillBase):
    def __init__(
        self,
        schema: SkillSchema,
        system_prompt: str,
        model_override: str = "",
        max_iterations: int = 3,
        allowed_paths: Optional[list[str]] = None,
    ):
        """
        Args:
            allowed_paths: Optional list of workspace-relative file paths
                or directory prefixes. When supplied, the sub-agent's
                generated patches are restricted to these paths. Patches
                targeting any other file are rejected. When None (default),
                no restriction is applied — useful for trusted, broad-scope
                skills like a general refactoring agent, but RECOMMENDED to
                set for narrowly-scoped skills so an LLM can't drift into
                unrelated files. Per-call override available via
                ``execute(allowed_paths=...)``.
        """
        super().__init__(schema)
        self.system_prompt = system_prompt
        self.model_override = model_override
        self.max_iterations = max_iterations
        self.allowed_paths = allowed_paths

    async def execute(self, **kwargs: Any) -> Any:
        from harness.graph import get_gateway
        from harness.patcher import process_llm_patch_output
        from harness.sandbox import SandboxExecutor

        task = kwargs.get("task", "")
        workspace_path = kwargs.get("workspace_path", "")
        build_command = kwargs.get("build_command", "")
        state = kwargs.get("state", {})
        # Per-call override beats constructor allowlist
        allowed_paths = kwargs.get("allowed_paths", self.allowed_paths)

        gateway = get_gateway()
        if gateway is None:
            return {"error": "No gateway configured.", "success": False}

        logger.info("[skills:subagent] '%s' starting (allowed_paths=%s).",
                     self.schema.name,
                     "<unrestricted>" if allowed_paths is None else list(allowed_paths))
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]

        try:
            for iteration in range(self.max_iterations):
                from harness.gateway import NodeRole
                budget = state.get("budget_remaining_usd", 2.00)

                response, new_budget = await gateway.dispatch(
                    messages=list(messages),
                    role=NodeRole.PATCHING,
                    budget_remaining_usd=budget,
                )

                if workspace_path and response.content:
                    patch_results, modified_files = await process_llm_patch_output(
                        response.content, workspace_path,
                        existing_modified_files=[],
                        allowed_paths=allowed_paths,
                    )
                else:
                    modified_files = []

                messages.append({"role": "assistant", "content": response.content})

                # Verify with build if configured
                if build_command and workspace_path:
                    executor = SandboxExecutor(workspace_path=workspace_path)
                    result = await executor.run(build_command)
                    if result.exit_code == 0:
                        return {"success": True, "modified_files": modified_files,
                                "iterations": iteration + 1}
                    else:
                        diag = "\n".join(d.message for d in result.diagnostics[:5])
                        messages.append({"role": "user", "content": f"Build failed:\n{diag}\nPlease fix."})

            return {"success": True, "modified_files": modified_files,
                    "iterations": self.max_iterations, "message": "Completed (max iterations)."}
        except Exception as exc:
            logger.exception("[skills:subagent] '%s' failed.", self.schema.name)
            return {"error": str(exc), "success": False}


# ---------------------------------------------------------------------------
# 6. DocGenSkill — Documentation Generation Sub-Agent
# ---------------------------------------------------------------------------

_DOCGEN_SYSTEM_PROMPTS = {
    # arch_doc and requirements are externalized to
    # harness/skills/docgen/{arch_doc,requirements_doc}.md and resolved via
    # ``_get_docgen_prompt`` below. The keys remain present (empty string)
    # so ``_DOCGEN_SYSTEM_PROMPTS["readme"]`` lookups don't surprise callers
    # that still iterate the dict, but the values here are unused for these
    # two types — the file on disk is the source of truth.
    "arch_doc": "",

    "functional_spec": """You are a Senior Technical Writer and Systems Analyst. Generate a Functional Specification document from the codebase.

## Functional Specification

### 1. System Overview
- Purpose and scope of the system
- Key capabilities

### 2. Module-by-Module Functional Breakdown
For each module/package found in the codebase:
- **Module Name**: Purpose
- **Public API / Exports**: Functions, classes, interfaces exposed
- **Inputs**: What data/parameters each function accepts
- **Outputs**: What each function returns
- **Side Effects**: Database writes, file I/O, network calls
- **Error Handling**: Exceptions raised, error codes

### 3. Data Flow Diagrams (textual)
- Request lifecycle
- Data transformation pipeline

### 4. Configuration & Environment
- Required environment variables
- Config files and their schemas

### 5. Integration Points
- External APIs called
- Message queues used
- File formats consumed/produced

Output as a structured Markdown document.""",

    "requirements": "",

    "api_doc": """You are an API Documentation Specialist. Generate comprehensive API reference documentation.

## API Reference

### 1. Base URL & Authentication
- Base URL for all endpoints
- Authentication method (Bearer token, API key, OAuth)
- Required headers

### 2. Endpoints
For each API endpoint found in the codebase:

#### `METHOD /path`
- **Description**: What this endpoint does
- **Request Parameters**:
  - Path parameters
  - Query parameters
  - Request body (JSON schema)
- **Response**:
  - Success response (status code, JSON schema)
  - Error responses (status codes, error format)
- **Example**:
  ```
  Request:  GET /api/users/42
  Response: 200 { "id": 42, "name": "..." }
  ```

### 3. Data Models
- Core schemas shared across endpoints
- Enum values and their meanings

### 4. Rate Limiting & Pagination
- Rate limit headers
- Pagination format (cursor-based, offset-based)

Output as a structured Markdown document.""",

    "readme": """You are a Developer Advocate and Technical Writer. Generate a comprehensive README.md for the project.

## README.md

### Project Title & Badge
- Project name
- Build status, coverage, license badges (placeholders)

### Overview
- 2-3 sentence description of what this project does
- Key features bullet list

### Installation
- Prerequisites
- Step-by-step installation instructions
- Configuration steps

### Quick Start
- Minimal example to get running
- Expected output

### Usage Guide
- Common use cases with code examples
- CLI commands reference
- API usage examples

### Project Structure
- Directory tree with descriptions
- Key files explained

### Development
- How to set up a dev environment
- How to run tests
- How to build

### Contributing
- Contribution guidelines
- Code style
- PR process

### License
- License type (from pyproject.toml or LICENSE file)

Output as a single README.md formatted document."""
}


# Doc types whose system prompt lives in harness/skills/docgen/*.md instead
# of the inline dict above. Mapping is (doc_type → filename stem).
_DOCGEN_EXTERNAL_PROMPTS = {
    "arch_doc": "arch_doc",
    "requirements": "requirements_doc",
}


def _get_docgen_prompt(doc_type: str) -> str:
    """Resolve the system prompt for a docgen doc_type.

    Externalized types (arch_doc, requirements) load from disk so the
    prompt can be edited without touching code. Other types fall back to
    the inline ``_DOCGEN_SYSTEM_PROMPTS`` dict; an unknown type falls all
    the way back to the readme prompt (matches prior behavior).
    """
    if doc_type in _DOCGEN_EXTERNAL_PROMPTS:
        from harness import docgen_prompts
        return docgen_prompts.load(_DOCGEN_EXTERNAL_PROMPTS[doc_type])
    return _DOCGEN_SYSTEM_PROMPTS.get(doc_type, _DOCGEN_SYSTEM_PROMPTS["readme"])


class DocGenSkill(SubAgentSkill):
    """
    Specialized sub-agent for generating project documentation.

    Differs from generic SubAgentSkill by:
    - Having pre-built system prompts for each doc type
    - Writing output to a specific file path
    - Including a post-generation build step (e.g., mdbook, mkdocs)
    """
    def __init__(
        self,
        doc_type: str,
        output_file: str,
        model_override: str = "",
        max_iterations: int = 2,
    ):
        prompt = _get_docgen_prompt(doc_type)
        schema = SkillSchema(
            name=f"{doc_type}_generator",
            description=f"Generate a {doc_type} document for the project.",
            skill_type=SkillType.DOCGEN,
            parameters=[
                SkillParameter("output_file", "string", f"Path to write the {doc_type} document."),
            ],
            tags=["documentation", doc_type, "generation"],
        )
        super().__init__(schema=schema, system_prompt=prompt, model_override=model_override, max_iterations=max_iterations)
        self.output_file = output_file
        self.doc_type = doc_type


async def _build_dir_snapshot(workspace_path: str) -> str:
    """Build a directory tree snapshot for documentation context."""
    lines: list[str] = []
    try:
        for root, dirs, files in _os.walk(workspace_path):
            depth = root[len(workspace_path):].count(_os.sep)
            if depth > 4:
                dirs.clear()
                continue
            dirs[:] = [d for d in sorted(dirs) if not d.startswith(".") and d not in
                       ("node_modules", "__pycache__", "target", "build", "dist", ".git", ".tox", "venv", ".venv")]
            indent = "  " * (depth + 1)
            rel = _os.path.relpath(root, workspace_path)
            if rel == ".":
                lines.append(f"{_os.path.basename(workspace_path)}/")
            else:
                lines.append(f"{indent[:-2]}{_os.path.basename(root)}/")
            for f in sorted(files)[:30]:
                lines.append(f"{indent}{f}")
    except Exception:
        pass
    return "\n".join(lines)


async def generate_documentation(
    doc_type: str,
    workspace_path: str,
    task_description: str = "",
    output_file: str = "",
    model_override: str = "",
) -> dict[str, Any]:
    """
    Generate a documentation document for the project.

    Args:
        doc_type: One of 'arch_doc', 'functional_spec', 'requirements', 'api_doc', 'readme'.
        workspace_path: Path to the project root.
        task_description: Additional context/requirements for the document.
        output_file: Where to write the generated document.
        model_override: Specific model to use.

    Returns:
        Result dict with success, file path, and metadata.
    """
    from harness.graph import get_gateway

    gateway = get_gateway()
    if gateway is None:
        return {"success": False, "error": "No gateway configured. Cannot generate documentation."}

    system_prompt = _get_docgen_prompt(doc_type)

    # Build directory snapshot for context
    tree = await _build_dir_snapshot(workspace_path)

    # Build the full task prompt
    task = f"""Generate the {doc_type} document for the following project.

## Project Directory Structure
```
{tree}
```

## Workspace Path
{workspace_path}

## Additional Context
{task_description or "Generate a comprehensive document from the codebase structure above."}

## Output Instructions
Write the complete document to: {output_file}
Use <<<CREATE_FILE>>> blocks to create the documentation file."""

    logger.info("[skills:docgen] Generating %s → %s", doc_type, output_file)

    # Dispatch to LLM
    from harness.gateway import NodeRole
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    try:
        response, budget = await gateway.dispatch(
            messages=messages,
            role=NodeRole.PLANNING,
            budget_remaining_usd=2.00,
        )

        # Validate and write the generated document
        if response.content:
            # Path validation: output_file must be workspace-relative and
            # must not escape via absolute path or parent traversal.
            from harness.trust import safe_resolve, validate_synthesized_spec
            try:
                abs_output = safe_resolve(workspace_path, output_file)
            except ValueError as ve:
                return {"success": False, "error": f"output_file path rejected: {ve}"}

            # Validate the content before writing
            content, trust_errors = validate_synthesized_spec(response.content)
            if trust_errors:
                return {"success": False, "error": f"docgen content failed trust validation: {trust_errors}"}

            _os.makedirs(_os.path.dirname(abs_output), exist_ok=True)
            with open(abs_output, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("[skills:docgen] Document written to %s (%d chars).", output_file, len(content))
            return {
                "success": True,
                "file": output_file,
                "size_chars": len(content),
                "cost_usd": response.usage.cost_usd,
                "message": f"Generated {doc_type} document at {output_file}.",
            }
        else:
            return {"success": False, "error": "LLM returned empty content."}
    except Exception as exc:
        logger.exception("[skills:docgen] Generation failed.")
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# 7. SkillRegistry — Global Singleton
# ---------------------------------------------------------------------------

class SkillRegistry:
    _instance: Optional["SkillRegistry"] = None
    _skills: dict[str, SkillBase]

    def __new__(cls) -> "SkillRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._skills = {}
        return cls._instance

    def register(self, skill: SkillBase) -> None:
        self._skills[skill.name] = skill
        logger.info("[skills] Registered %s '%s'.", skill.skill_type.value, skill.name)

    def get(self, name: str) -> Optional[SkillBase]:
        return self._skills.get(name)

    def list_by_type(self, skill_type: SkillType) -> list[SkillBase]:
        return [s for s in self._skills.values() if s.skill_type == skill_type]

    def list_all(self) -> list[SkillBase]:
        return list(self._skills.values())

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [s.to_tool_schema() for s in self._skills.values()
                if s.skill_type == SkillType.TOOL and s.to_tool_schema()]

    def get_pipeline_nodes(self) -> dict[str, Callable[..., Awaitable[dict[str, Any]]]]:
        nodes: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {}
        for s in self._skills.values():
            if s.skill_type == SkillType.PIPELINE and isinstance(s, PipelineSkill):
                nodes[s.name] = s.to_node_fn()
        return nodes

    async def dispatch(self, name: str, **kwargs: Any) -> Any:
        skill = self._skills.get(name)
        if skill is None:
            raise KeyError(f"Skill '{name}' not registered. Available: {list(self._skills.keys())}")
        return await skill.execute(**kwargs)


# ---------------------------------------------------------------------------
# 8. Convenience Functions
# ---------------------------------------------------------------------------

def register(skill: SkillBase) -> None:
    SkillRegistry().register(skill)


def get_skill(name: str) -> Optional[SkillBase]:
    return SkillRegistry().get(name)


def get_tool_schemas() -> list[dict[str, Any]]:
    return SkillRegistry().get_tool_schemas()


def get_pipeline_nodes() -> dict[str, Callable[..., Awaitable[dict[str, Any]]]]:
    return SkillRegistry().get_pipeline_nodes()


# ---------------------------------------------------------------------------
# 9. Register Built-in Documentation Skills
# ---------------------------------------------------------------------------

def register_docgen_skills() -> int:
    """Register all documentation generation skills."""
    count = 0

    register(DocGenSkill(
        doc_type="arch_doc", output_file="docs/architecture.md",
        model_override="planning_primary", max_iterations=2,
    ))
    count += 1

    register(DocGenSkill(
        doc_type="functional_spec", output_file="docs/functional-spec.md",
        model_override="planning_primary", max_iterations=2,
    ))
    count += 1

    register(DocGenSkill(
        doc_type="requirements", output_file="docs/requirements.md",
        model_override="planning_primary", max_iterations=2,
    ))
    count += 1

    register(DocGenSkill(
        doc_type="api_doc", output_file="docs/api-reference.md",
        model_override="patching_primary", max_iterations=2,
    ))
    count += 1

    register(DocGenSkill(
        doc_type="readme", output_file="README.md",
        model_override="patching_primary", max_iterations=2,
    ))
    count += 1

    logger.info("[skills] Registered %d documentation generation skill(s).", count)
    return count


# ---------------------------------------------------------------------------
# 10. Register All Built-in Skills
# ---------------------------------------------------------------------------

def register_builtin_skills(config: Optional[dict[str, Any]] = None) -> int:
    """Register all built-in pipeline, tool, and documentation skills.

    Args:
        config: Optional parsed config.json dict. When provided, opt-in
            skill groups (currently: web tools) are registered if their
            section turns them on. When ``None`` (the historical call
            signature kept for the test suite), no opt-in skills are
            registered — same behaviour as before this slice.

    Returns the number of skills registered. Each individual registration
    is wrapped in try/except so a failure in one optional skill does NOT
    take down startup — we log + continue.
    """
    count = 0

    # Pipeline: lintgate
    try:
        from harness.lintgate import lintgate_node
        register(PipelineSkill(SkillSchema(
            name="lintgate", description="Auto-format modified files using language-specific formatters.",
            skill_type=SkillType.PIPELINE, tags=["formatting", "pre-build"],
        ), node_fn=lintgate_node))
        count += 1
    except ImportError:
        pass

    # Pipeline: speculative
    try:
        from harness.speculative import speculate_node
        register(PipelineSkill(SkillSchema(
            name="speculative", description="Multi-variant parallel compilation.",
            skill_type=SkillType.PIPELINE, tags=["compilation", "parallel"],
        ), node_fn=speculate_node))
        count += 1
    except ImportError:
        pass

    # Pipeline: security_scan
    try:
        from harness.security import security_scan_node
        register(PipelineSkill(SkillSchema(
            name="security_scan", description="SAST + secret scanning gatekeeper.",
            skill_type=SkillType.PIPELINE, tags=["security", "audit"],
        ), node_fn=security_scan_node))
        count += 1
    except ImportError:
        pass

    # Documentation skills
    count += register_docgen_skills()

    # Opt-in: web tools (web_fetch, web_search). Only register when the
    # operator turned them on in config.json; default is off. Wrapped so
    # any import or registration failure (e.g. httpx unavailable in an
    # exotic test env) logs and skips rather than killing startup.
    try:
        from harness.web_tools import WebToolsConfig, register_web_tool_skills
        web_cfg = WebToolsConfig.from_config(config)
        if web_cfg.enabled:
            count += register_web_tool_skills(web_cfg)
        else:
            logger.debug("[skills] web_tools disabled in config; skipping registration.")
    except Exception as exc:  # noqa: BLE001 — additive skill registration must never block startup
        logger.warning("[skills] web tools registration skipped: %s", exc)

    # Always-on: the multi-agent fan-out tool (#11). Exposed via the
    # text-DSL pattern as ``<<<FANOUT_QUERY prompts='[...]'>>>`` —
    # consistent with web/MCP tool wiring. Failures registering this
    # are non-fatal; the planner just won't have the tool available.
    try:
        from harness.fanout import register_fanout_skill
        count += register_fanout_skill()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[skills] fanout skill registration skipped: %s", exc)

    # Opt-in: user skills directory. Operators drop `*.py` files under
    # ``~/.harness/user_skills`` (or the directory named by ``skills.user_skills_dir``);
    # each module is imported at startup and can call ``harness.skills.register``
    # to add its own ToolSkill / PipelineSkill / SubAgentSkill, OR call
    # ``harness.web_tools.register_backend(name, factory)`` to plug in an
    # alternative web-search backend (Tavily, Brave, SerpAPI, in-house) without
    # forking the harness — the operator then flips ``web_tools.search_backend``
    # to the registered name. The contract is "import side-effect registration"
    # — same pattern Claude Code uses, same pattern the built-ins above use.
    # Bad files (syntax errors, missing deps) log and are skipped so one bad
    # file doesn't take down the harness.
    try:
        user_count = load_user_skills_directory(config)
        count += user_count
    except Exception as exc:  # noqa: BLE001
        logger.warning("[skills] user skills directory load skipped: %s", exc)

    logger.info("[skills] Registered %d total built-in skill(s).", count)
    return count


# ---------------------------------------------------------------------------
# 11. User Skills Directory Loader  (#5 — runtime-extensible skills)
# ---------------------------------------------------------------------------

# New default. Chosen to disambiguate from the BUNDLED markdown directory
# at ``harness/skills/`` inside the installed package — that one ships
# stack scaffolds (react.md, python_django.md, …) the planner reads, and
# is unrelated to operator-supplied Python. Operators kept dropping
# ``*.py`` files in the wrong place because the names collided.
_DEFAULT_USER_SKILLS_DIR = "~/.harness/user_skills"

# Legacy default — operators who relied on the implicit default (no
# ``skills.user_skills_dir`` in their config) still have files at the
# old path. _resolve_user_skills_dir() falls back to this when the new
# default doesn't exist but the legacy one does, and logs a one-time
# deprecation INFO so the operator knows to migrate. Operators who set
# the config key explicitly are unaffected either way.
_LEGACY_USER_SKILLS_DIR = "~/.harness/skills"

_legacy_fallback_warned = False


def _resolve_user_skills_dir(config: Optional[dict[str, Any]]) -> str:
    """Resolve the user-skills directory from config, falling back to the
    new default — or, transitionally, the legacy default — when no
    explicit value is set.

    Resolution order:
        1. ``config["skills"]["user_skills_dir"]`` — explicit operator
           choice; honoured verbatim (no fallback applied).
        2. ``_DEFAULT_USER_SKILLS_DIR`` (~/.harness/user_skills) — new
           default; returned as-is even if the directory doesn't yet
           exist, so a fresh install creates files in the right place.
        3. ``_LEGACY_USER_SKILLS_DIR`` (~/.harness/skills) — fallback
           used ONLY when (a) the operator didn't set the key AND (b)
           the new directory doesn't exist AND (c) the legacy directory
           does. Logs a one-time deprecation notice naming both paths
           so operators know to ``mv`` and either accept the new
           default or pin the legacy path in config.
    """
    global _legacy_fallback_warned
    section = ((config or {}).get("skills") or {})
    explicit = section.get("user_skills_dir")
    if explicit:
        return _os.path.expanduser(str(explicit))

    new_default = _os.path.expanduser(_DEFAULT_USER_SKILLS_DIR)
    if _os.path.isdir(new_default):
        return new_default

    legacy = _os.path.expanduser(_LEGACY_USER_SKILLS_DIR)
    if _os.path.isdir(legacy):
        if not _legacy_fallback_warned:
            logger.info(
                "[skills] Using legacy user-skills directory %s. The default "
                "moved to %s to disambiguate from the bundled markdown "
                "scaffolds at harness/skills/. Move your *.py files to the "
                "new location, or pin the old path explicitly via "
                "skills.user_skills_dir in config.json to silence this notice.",
                legacy, new_default,
            )
            _legacy_fallback_warned = True
        return legacy

    # Neither directory exists. Return the NEW default so an operator who
    # creates the directory afterward gets the modern path without
    # having to touch config.
    return new_default


def load_user_skills_directory(config: Optional[dict[str, Any]] = None) -> int:
    """Import every ``*.py`` file under the user skills directory.

    Each file's module-level body runs at import time and may call
    :func:`register` to add new skills to the global registry. The loader:
      - silently no-ops if the directory does not exist (default install
        has no user skills);
      - imports files in deterministic sorted order;
      - skips files whose name starts with ``_`` (private helpers);
      - wraps each import in try/except so one bad file does not abort
        the rest of the load — failures log a clear warning naming the
        offending path and exception.

    Returns the number of files successfully imported (not the number of
    skills they registered — a single file can register many).
    """
    target_dir = _resolve_user_skills_dir(config)
    if not _os.path.isdir(target_dir):
        logger.debug("[skills] user skills dir %s does not exist; skipping.", target_dir)
        return 0

    import importlib.util
    import sys

    count = 0
    try:
        entries = sorted(_os.listdir(target_dir))
    except OSError as exc:
        logger.warning("[skills] cannot list %s: %s", target_dir, exc)
        return 0

    for filename in entries:
        if not filename.endswith(".py"):
            continue
        if filename.startswith("_"):
            continue
        path = _os.path.join(target_dir, filename)
        if not _os.path.isfile(path):
            continue
        module_name = f"harness_user_skill_{filename[:-3]}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                logger.warning("[skills] could not build import spec for %s", path)
                continue
            module = importlib.util.module_from_spec(spec)
            # Insert into sys.modules BEFORE exec so the module can refer
            # to itself by name (rare but allowed) and so any sub-imports
            # see a partially-initialised parent — matches importlib's
            # documented contract.
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            count += 1
            logger.info("[skills] loaded user skill module %s from %s",
                         module_name, path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[skills] user skill file %s failed to load: %s",
                path, exc,
            )
            sys.modules.pop(module_name, None)
    if count:
        logger.info("[skills] loaded %d user skill module(s) from %s.",
                     count, target_dir)
    return count