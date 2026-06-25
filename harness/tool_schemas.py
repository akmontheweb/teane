"""Tool-use schemas for the patching/repair LLM dispatch (B6 foundation).

Defines the typed function/tool schemas that mirror the harness's
``<<<REPLACE_BLOCK>>>``-style text DSL. The schemas live here so:

- Provider request builders (``AnthropicProvider``, ``OpenAIProvider``,
  ``DeepSeekProvider``, ``OllamaProvider`` in ``harness/gateway.py``) can
  import a single canonical definition rather than duplicating each
  tool's input_schema per provider.
- Provider response parsers populate ``LLMResponse.tool_calls`` with
  uniform ``{"name", "input", "id"}`` dicts regardless of vendor wire
  format.
- ``harness/graph.py`` (``patching_node`` / ``repair_node``) can call
  :func:`tool_calls_to_patch_blocks` to translate the parsed structured
  responses back into ``PatchBlock`` objects that the existing
  ``HybridPatcher`` apply pipeline handles unchanged.

The schemas intentionally mirror the existing DSL's semantics (no new
operations) so the host pipeline doesn't need to know whether the LLM
used native tool-use or the text DSL — they converge at ``PatchBlock``.

Activation is gated by ``GatewayConfig.use_structured_tools`` (false by
default). When true, providers that support tool-use receive these
schemas in their chat_completion call; otherwise the legacy text DSL
keeps running. See ``config.patcher.use_structured_tools`` in
``config/config.json``.
"""

from __future__ import annotations

from typing import Any

from harness.patcher import OperationType, PatchBlock, Placement


# ---------------------------------------------------------------------------
# JSON-Schema fragments — one per canonical patch operation
# ---------------------------------------------------------------------------

# Note: the ``count`` field on replace_file / delete_block mirrors B2 in
# the text DSL — "unique" is the default, "all" replaces every match,
# "first" replaces only the first.

EDIT_FILE_SCHEMA: dict[str, Any] = {
    "name": "edit_file",
    "description": (
        "Replace an exact-match block of text within an existing file. "
        "Equivalent to <<<REPLACE_BLOCK>>> in the text DSL. The search "
        "string MUST be a verbatim substring of the on-disk file — copy "
        "bytes from a READ_FILE result or a closest-match window, never "
        "guess."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Workspace-relative path to the file.",
            },
            "old_string": {
                "type": "string",
                "description": "Exact substring to replace.",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text.",
            },
            "count": {
                "type": "string",
                "enum": ["unique", "all", "first"],
                "default": "unique",
                "description": (
                    "Match-count policy. 'unique' fails on >1 match (default), "
                    "'all' replaces every occurrence, 'first' replaces only "
                    "the first."
                ),
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    },
}

CREATE_FILE_SCHEMA: dict[str, Any] = {
    "name": "create_file",
    "description": (
        "Create a new file with the given content. Equivalent to "
        "<<<CREATE_FILE>>> in the text DSL. Rejected if the target "
        "already exists with different content; safe no-op when the "
        "target already exists with identical content."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Workspace-relative path for the new file.",
            },
            "content": {
                "type": "string",
                "description": "Complete file contents to write.",
            },
        },
        "required": ["file_path", "content"],
    },
}

DELETE_BLOCK_SCHEMA: dict[str, Any] = {
    "name": "delete_block",
    "description": (
        "Remove an exact-match block of text from an existing file. "
        "Equivalent to <<<DELETE_BLOCK>>> in the text DSL."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Workspace-relative path to the file.",
            },
            "search": {
                "type": "string",
                "description": "Exact text to remove.",
            },
            "count": {
                "type": "string",
                "enum": ["unique", "all", "first"],
                "default": "unique",
                "description": "Same semantics as edit_file's count.",
            },
        },
        "required": ["file_path", "search"],
    },
}

INSERT_AT_BLOCK_SCHEMA: dict[str, Any] = {
    "name": "insert_at_block",
    "description": (
        "Insert content immediately before or after a named function or "
        "class. Equivalent to <<<INSERT_AT_BLOCK>>> in the text DSL."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Workspace-relative path to the file.",
            },
            "anchor": {
                "type": "string",
                "description": (
                    "Name of the function or class to anchor on. For "
                    "languages with tree-sitter support the patcher uses "
                    "AST-aware lookup; otherwise it falls back to "
                    "substring search."
                ),
            },
            "placement": {
                "type": "string",
                "enum": ["before", "after"],
                "description": (
                    "Insert immediately before or after the anchor's "
                    "first matching node."
                ),
            },
            "content": {
                "type": "string",
                "description": "Text to insert.",
            },
        },
        "required": ["file_path", "anchor", "placement", "content"],
    },
}

READ_FILE_SCHEMA: dict[str, Any] = {
    "name": "read_file",
    "description": (
        "Ask the harness for the current bytes of a file. The host "
        "resolves this inline and re-dispatches you in the same "
        "iteration with the line-numbered content as a follow-up "
        "user message. Use this BEFORE writing any edit when you do "
        "not know — or are unsure of — a file's current bytes. "
        "Mirrors Claude Code's Read-before-Edit invariant."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Workspace-relative path to the file.",
            },
            "start_line": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Optional 1-indexed first line to return. Omit for "
                    "whole-file output (capped at the harness's default "
                    "size limits)."
                ),
            },
            "end_line": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Optional 1-indexed last line to return (inclusive). "
                    "Omit for whole-file output. Must be >= start_line."
                ),
            },
        },
        "required": ["file_path"],
    },
}


# Canonical ordering of tool schemas. Order matters because some
# providers (notably Anthropic) preserve the order in their UI surfaces;
# we want the most-used tools first.
PATCH_TOOLS: list[dict[str, Any]] = [
    READ_FILE_SCHEMA,
    EDIT_FILE_SCHEMA,
    CREATE_FILE_SCHEMA,
    DELETE_BLOCK_SCHEMA,
    INSERT_AT_BLOCK_SCHEMA,
]


# ---------------------------------------------------------------------------
# Provider-shape adapters
# ---------------------------------------------------------------------------

def to_anthropic_tools(tools: list[dict[str, Any]] = PATCH_TOOLS) -> list[dict[str, Any]]:
    """Return ``tools`` in Anthropic's Messages-API ``tools=[...]`` shape.

    Anthropic accepts the raw ``{name, description, input_schema}`` dicts
    directly, so this is mostly a copy. Kept as a function so future
    Anthropic-specific tweaks (e.g. caching control on tool blocks) have
    one place to land.
    """
    return [dict(t) for t in tools]


def to_openai_tools(tools: list[dict[str, Any]] = PATCH_TOOLS) -> list[dict[str, Any]]:
    """Return ``tools`` in the OpenAI function-calling shape used by
    OpenAI / DeepSeek / Ollama-OpenAI-compat:

        [{"type": "function", "function": {"name", "description", "parameters"}}]

    OpenAI calls the schema field ``parameters`` (vs Anthropic's
    ``input_schema``) but the JSON shape is identical.
    """
    out: list[dict[str, Any]] = []
    for t in tools:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        })
    return out


# ---------------------------------------------------------------------------
# Tool-call → PatchBlock translation
# ---------------------------------------------------------------------------

# READ_FILE tool calls do not become PatchBlocks — they are intercepted
# by the host and re-dispatched. The translator returns None for
# read_file and the caller resolves it via the existing READ_FILE
# resolver path. This keeps the downstream patcher unchanged.
def tool_call_to_patch_block(call: dict[str, Any]) -> "PatchBlock | None":
    """Translate one ``{"name", "input"}`` tool call into a ``PatchBlock``.

    Returns ``None`` when the call is ``read_file`` (handled separately
    by the host) or when the tool name is unknown.
    """
    name = call.get("name", "")
    args = call.get("input") or {}
    if not isinstance(args, dict):
        return None
    if name == "edit_file":
        return PatchBlock(
            operation=OperationType.REPLACE_BLOCK,
            file=str(args.get("file_path", "")).strip(),
            search=str(args.get("old_string", "")),
            replace=str(args.get("new_string", "")),
            count=str(args.get("count", "unique") or "unique").strip().lower(),
        )
    if name == "create_file":
        return PatchBlock(
            operation=OperationType.CREATE_FILE,
            file=str(args.get("file_path", "")).strip(),
            content=str(args.get("content", "")),
        )
    if name == "delete_block":
        return PatchBlock(
            operation=OperationType.DELETE_BLOCK,
            file=str(args.get("file_path", "")).strip(),
            search=str(args.get("search", "")),
            count=str(args.get("count", "unique") or "unique").strip().lower(),
        )
    if name == "insert_at_block":
        placement_str = str(args.get("placement", "after") or "after").strip().lower()
        placement = Placement.BEFORE if placement_str == "before" else Placement.AFTER
        return PatchBlock(
            operation=OperationType.INSERT_AT_BLOCK,
            file=str(args.get("file_path", "")).strip(),
            anchor=str(args.get("anchor", "")),
            placement=placement,
            content=str(args.get("content", "")),
        )
    # Line-coordinate ops. PATCH_TOOLS does not yet expose tool
    # definitions for these — Layer-2 rule-table autofixes and Layer-1
    # semgrep ``extra.fix`` patches are constructed directly from
    # PatchBlock and never round-trip through tool calls. The dispatch
    # branches are added so that if/when tool definitions are added,
    # the translator already round-trips correctly and isn't a silent
    # drop. The same class of dispatch-missing-ops bug burned us in
    # ``autofix._apply_block`` (2026-06-25 security HITL loop) — this
    # is the parallel preventive fix.
    if name == "insert_at_line":
        try:
            line_no = int(args.get("line", 0) or 0)
        except (TypeError, ValueError):
            line_no = 0
        return PatchBlock(
            operation=OperationType.INSERT_AT_LINE,
            file=str(args.get("file_path", "")).strip(),
            line=line_no,
            content=str(args.get("content", "")),
            expected_file_hash=str(args.get("expected_file_hash", "") or "").strip().lower(),
        )
    if name == "replace_line_range":
        try:
            start_line = int(args.get("line", 0) or 0)
            end_line = int(args.get("end_line", 0) or 0)
        except (TypeError, ValueError):
            start_line = end_line = 0
        return PatchBlock(
            operation=OperationType.REPLACE_LINE_RANGE,
            file=str(args.get("file_path", "")).strip(),
            line=start_line,
            end_line=end_line,
            content=str(args.get("content", "")),
            expected_file_hash=str(args.get("expected_file_hash", "") or "").strip().lower(),
        )
    # read_file is host-resolved, not a patch.
    return None


def tool_calls_to_patch_blocks(
    calls: list[dict[str, Any]],
) -> tuple[list[PatchBlock], list[dict[str, Any]]]:
    """Partition tool calls into (patch_blocks, read_file_calls).

    Patch blocks feed the existing apply pipeline; read_file calls are
    resolved inline by the host (same single-turn semantics as the
    READ_FILE text DSL block). Calls with unknown names are dropped on
    the floor — host can log if it cares.
    """
    blocks: list[PatchBlock] = []
    reads: list[dict[str, Any]] = []
    for call in calls or []:
        if call.get("name") == "read_file":
            reads.append(call)
            continue
        block = tool_call_to_patch_block(call)
        if block is not None:
            blocks.append(block)
    return blocks, reads
