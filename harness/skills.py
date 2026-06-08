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
    ):
        super().__init__(schema)
        self.system_prompt = system_prompt
        self.model_override = model_override
        self.max_iterations = max_iterations

    async def execute(self, **kwargs: Any) -> Any:
        from harness.graph import get_gateway
        from harness.patcher import process_llm_patch_output
        from harness.sandbox import SandboxExecutor

        task = kwargs.get("task", "")
        workspace_path = kwargs.get("workspace_path", "")
        build_command = kwargs.get("build_command", "")
        state = kwargs.get("state", {})

        gateway = get_gateway()
        if gateway is None:
            return {"error": "No gateway configured.", "success": False}

        logger.info("[skills:subagent] '%s' starting.", self.schema.name)
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
                        response.content, workspace_path, existing_modified_files=[],
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
    "arch_doc": """You are a Principal Software Architect. Generate a comprehensive Architecture Decision Record (ADR) for the project.

Scan the directory structure and codebase below. Produce a document with these sections:

## Architecture Decision Record

### 1. System Context (C4 Level 1)
- What does this system do? Who are its users?
- External systems it interacts with.

### 2. Container Diagram (C4 Level 2)
- Major deployable units (web app, API server, database, cache, message queue)
- Communication protocols between them.

### 3. Component Diagram (C4 Level 3)
- Key modules/packages and their responsibilities
- Data flow between components
- Dependency relationships

### 4. Technology Stack
- Languages, frameworks, databases, infrastructure
- Version requirements

### 5. Key Design Decisions
- Why specific patterns were chosen (or should be chosen)
- Trade-offs considered

### 6. Data Model Overview
- Core entities and their relationships
- Storage strategy

Output as a well-formatted Markdown document. Use the file structure provided to infer the architecture.""",

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

    "requirements": """You are a Business Analyst and Requirements Engineer. Generate a Requirements Specification document.

## Requirements Specification

### 1. Executive Summary
- Project purpose and business value

### 2. Functional Requirements (FR)
For each feature found in the codebase or described in the task:
- **FR-XXX**: Title
  - Description
  - Priority (Must Have / Should Have / Could Have)
  - Acceptance Criteria (Given/When/Then format)

### 3. Non-Functional Requirements (NFR)
- **NFR-001**: Performance (response time, throughput)
- **NFR-002**: Security (authentication, authorization, data protection)
- **NFR-003**: Reliability (uptime, error rates)
- **NFR-004**: Scalability (horizontal/vertical scaling targets)

### 4. Traceability Matrix
| Requirement | Module/File | Status |
|---|---|---|
| FR-001 | src/auth/login.py | Implemented |
| FR-002 | src/api/users.py | Partial |

### 5. Constraints & Assumptions
- Technical constraints
- Business constraints
- Assumptions made during analysis

Output as a structured Markdown document.""",

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
        prompt = _DOCGEN_SYSTEM_PROMPTS.get(doc_type, _DOCGEN_SYSTEM_PROMPTS["readme"])
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

    prompt_key = doc_type
    system_prompt = _DOCGEN_SYSTEM_PROMPTS.get(prompt_key, _DOCGEN_SYSTEM_PROMPTS["readme"])

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

    def __new__(cls) -> "SkillRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._skills: dict[str, SkillBase] = {}
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

def register_builtin_skills() -> int:
    """Register all built-in pipeline, tool, and documentation skills."""
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

    logger.info("[skills] Registered %d total built-in skill(s).", count)
    return count