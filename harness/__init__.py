"""
Teane — production-grade, model-agnostic LangGraph agent for autonomous code generation,
sandboxed build execution, and bulletproof persistence.

Modules:
    graph           — LangGraph StateGraph topology, typed state schema, Pydantic models, and all node implementations.
    gateway         — Model-agnostic LLM gateway with prefix caching, token tracking, and budget enforcement.
    sandbox         — Pluggable isolation backends (unshare/Docker/bare), async subprocess, log streaming.
    patcher         — Hybrid tree-sitter AST-aware + text SEARCH/REPLACE file modification engine (aiofiles).
    skills          — Unified skill registry (tools, pipelines, sub-agents) + 5 documentation generators.
    deploy          — Container build + health-check orchestration; Dockerfile/compose generation, docker inspect polling.
    speculative     — Multi-variant speculative compilation; generates 3 parallel patches, compiles in isolated worktrees.
    lintgate        — Deterministic auto-format/lint verification node (gofmt, ruff, prettier, rustfmt, etc.).
    impact          — Semantic code graph dependency scanner + AST impact analysis for downstream breakage warnings.
    redactor        — Zero-knowledge secret scanner; strips/hashes API keys, tokens, and credentials before LLM transit.
    storage         — Pluggable checkpointer backend (AsyncSqliteSaver) with 30-day TTL garbage collection.
    security        — Git branch lifecycle management + deterministic command whitelist + HITL gate + SAST node.
    parser_registry — Language-specific diagnostic parser plugin registry.
    cli             — CLI entry point, subcommand routing, HITL interactive menu loop.
"""

__version__ = "1.0.0"
__all__ = [
    "graph",
    "gateway",
    "sandbox",
    "patcher",
    "skills",
    "deploy",
    "speculative",
    "lintgate",
    "impact",
    "redactor",
    "storage",
    "security",
    "parser_registry",
    "cli",
]
