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

import logging
from typing import Any, Optional

from harness.patcher import OperationType, PatchBlock, Placement

logger = logging.getLogger(__name__)


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

REWRITE_FILE_SCHEMA: dict[str, Any] = {
    "name": "rewrite_file",
    "description": (
        "Overwrite an EXISTING file wholesale with new content. "
        "Equivalent to <<<REWRITE_FILE>>> in the text DSL. This is the "
        "escape hatch for whole-file replacement — prefer edit_file for "
        "surgical changes and reach for this only when a file needs to be "
        "rebuilt end-to-end (or when a stuck-file REPLACE_BLOCK loop has "
        "unlocked it). You MUST supply the COMPLETE corrected contents "
        "(imports, every class, every def, every existing test) — a "
        "partial rewrite deletes whatever you leave out. Emitting content "
        "byte-identical to what is already on disk is rejected as a no-op, "
        "so change something or pick a different target. Post-patch parse "
        "validation still applies: unparseable output is rolled back."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Workspace-relative path to the existing file.",
            },
            "content": {
                "type": "string",
                "description": "Complete new file contents to write.",
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


INSERT_AT_LINE_SCHEMA: dict[str, Any] = {
    "name": "insert_at_line",
    "description": (
        "Insert content BEFORE a specific 1-indexed line in a file. "
        "Equivalent to <<<INSERT_AT_LINE>>> in the text DSL. Prefer "
        "edit_file / insert_at_block when you can describe the target "
        "by content or anchor name; use this when you only have line "
        "coordinates (e.g. a diagnostic from a scanner)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Workspace-relative path to the file.",
            },
            "line": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "1-indexed line number; the inserted block lands "
                    "BEFORE this line. Use ``last_line + 1`` to append."
                ),
            },
            "content": {
                "type": "string",
                "description": "Text to insert.",
            },
            "expected_file_hash": {
                "type": "string",
                "description": (
                    "Optional sha256 of the file at the time the caller "
                    "decided on the line number. When set the patcher "
                    "refuses the patch if the file has drifted. Leave "
                    "blank to trust the line number unconditionally."
                ),
            },
        },
        "required": ["file_path", "line", "content"],
    },
}

REPLACE_LINE_RANGE_SCHEMA: dict[str, Any] = {
    "name": "replace_line_range",
    "description": (
        "Replace a 1-indexed inclusive line range with new content. "
        "Equivalent to <<<REPLACE_LINE_RANGE>>> in the text DSL. Empty "
        "``content`` deletes the range. Use when you only have line "
        "coordinates (scanner diagnostic, compiler error span)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Workspace-relative path to the file.",
            },
            "line": {
                "type": "integer",
                "minimum": 1,
                "description": "1-indexed first line of the range (inclusive).",
            },
            "end_line": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "1-indexed last line of the range (inclusive). "
                    "Must be >= line."
                ),
            },
            "content": {
                "type": "string",
                "description": (
                    "Replacement text. Empty string deletes the range."
                ),
            },
            "expected_file_hash": {
                "type": "string",
                "description": (
                    "Optional sha256 of the file at the time the caller "
                    "decided on the line range. When set the patcher "
                    "refuses the patch if the file has drifted."
                ),
            },
        },
        "required": ["file_path", "line", "end_line", "content"],
    },
}


# Canonical ordering of tool schemas. Order matters because some
# providers (notably Anthropic) preserve the order in their UI surfaces;
# we want the most-used tools first. The line-coordinate ops sit at the
# end of the patch tools — they're for the rare cases where the LLM has
# line numbers but not content/anchor.
PATCH_TOOLS: list[dict[str, Any]] = [
    READ_FILE_SCHEMA,
    EDIT_FILE_SCHEMA,
    CREATE_FILE_SCHEMA,
    REWRITE_FILE_SCHEMA,
    DELETE_BLOCK_SCHEMA,
    INSERT_AT_BLOCK_SCHEMA,
    INSERT_AT_LINE_SCHEMA,
    REPLACE_LINE_RANGE_SCHEMA,
]


# ---------------------------------------------------------------------------
# Provider-shape adapters
# ---------------------------------------------------------------------------

def to_anthropic_tools(tools: list[dict[str, Any]] = PATCH_TOOLS) -> list[dict[str, Any]]:
    """Return ``tools`` in Anthropic's Messages-API ``tools=[...]`` shape.

    Anthropic accepts the raw ``{name, description, input_schema}`` dicts
    directly, so this is mostly a copy. ``strict`` (when a caller has set
    it via :func:`with_strict`) is already a top-level field on the tool,
    which is exactly where Anthropic wants it — alongside ``name`` /
    ``description`` / ``input_schema``, NOT on ``tool_choice``.
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
        fn: dict[str, Any] = {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        }
        # OpenAI nests ``strict`` INSIDE the function object, where
        # Anthropic keeps it top-level on the tool. Same flag, different
        # depth — the only reason this adapter has to know about it.
        if t.get("strict"):
            fn["strict"] = True
        out.append({"type": "function", "function": fn})
    return out


# ---------------------------------------------------------------------------
# strict mode — schema-valid tool input, independent of tool_choice
# ---------------------------------------------------------------------------
#
# ``strict`` constrains the ARGUMENTS a tool call carries; ``tool_choice``
# constrains WHETHER one happens. They are independent, which matters on a
# platform: a model that refuses to be compelled may still honour strict,
# so strict is the more portable of the two guarantees.
#
# Both providers require the schema to be closed — ``additionalProperties:
# false`` throughout, and ``required`` present on every object. A strict
# tool whose schema is not closed is rejected, so ``with_strict`` hardens
# the schema rather than trusting the caller to have done it.

def to_strict_schema(schema: Any) -> Any:
    """Return ``schema`` closed for strict mode, recursively.

    Sets ``additionalProperties: false`` on every object node and makes
    ``required`` list every declared property. Promoting optional fields
    to required is deliberate and is what strict mode demands: the model
    must emit the key, though an empty array / empty string / null-union
    still expresses "nothing here". Callers who need a genuinely absent
    key should model it as a nullable type rather than an optional one.

    Input is never mutated — nested dicts and lists are rebuilt.
    """
    if isinstance(schema, list):
        return [to_strict_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {k: to_strict_schema(v) for k, v in schema.items()}
    if out.get("type") == "object" or "properties" in out:
        props = out.get("properties")
        if isinstance(props, dict):
            out["additionalProperties"] = False
            out["required"] = list(props.keys())
    return out


def with_strict(
    tools: list[dict[str, Any]] = PATCH_TOOLS, *, enabled: bool = True,
) -> list[dict[str, Any]]:
    """Return ``tools`` marked strict, with every schema closed to match.

    ``enabled=False`` returns the tools unchanged (and strips any strict
    marker), so a caller can flip the guarantee from config without
    branching at the call site.
    """
    out: list[dict[str, Any]] = []
    for t in tools:
        copy = dict(t)
        if enabled:
            copy["strict"] = True
            if "input_schema" in copy:
                copy["input_schema"] = to_strict_schema(copy["input_schema"])
        else:
            copy.pop("strict", None)
        out.append(copy)
    return out


def strip_strict(tools: Optional[list[dict[str, Any]]]) -> Optional[list[dict[str, Any]]]:
    """Return ``tools`` with every ``strict`` marker removed.

    The recovery path for a provider that rejects strict mode. The closed
    schema is left in place — it is valid ordinary JSON Schema and costs
    nothing; only the flag that triggers the refusal is dropped.
    """
    if not tools:
        return tools
    return [{k: v for k, v in t.items() if k != "strict"} for t in tools]


# ---------------------------------------------------------------------------
# tool_choice — offering a tool vs. compelling one
# ---------------------------------------------------------------------------
#
# Attaching ``tools`` only OFFERS them: every backend here is free to answer
# in prose and ignore the array entirely. That is exactly the failure that
# kills single-dispatch structured nodes — the planner narrates its plan in
# ``content`` and never calls the tool (lumina-testrun-20260911-1102).
# ``tool_choice`` is the knob that turns an offer into a requirement.
#
# Callers use a provider-agnostic vocabulary and the adapters below translate:
#
#   None          omit the field — provider default (usually "auto")
#   "auto"        model decides whether to call a tool
#   "required"    model MUST call one of the supplied tools (any of them)
#   "<tool name>" model MUST call that specific tool
#
# NEVER send tool_choice without a non-empty ``tools`` array — OpenAI-compat
# backends 400 the request outright.

TOOL_CHOICE_AUTO = "auto"
TOOL_CHOICE_REQUIRED = "required"


def to_openai_tool_choice(choice: Optional[str]) -> Any:
    """Translate a canonical ``tool_choice`` into the OpenAI-compat shape.

    Used by OpenAI, DeepSeek, Google (OpenAI-compat endpoint), Moonshot and
    Ollama. ``"auto"``/``"required"`` are bare strings on the wire; a
    specific tool is the nested ``{"type": "function", ...}`` object.
    """
    if choice is None:
        return None
    if choice in (TOOL_CHOICE_AUTO, TOOL_CHOICE_REQUIRED):
        return choice
    return {"type": "function", "function": {"name": choice}}


def to_anthropic_tool_choice(choice: Optional[str]) -> Optional[dict[str, Any]]:
    """Translate a canonical ``tool_choice`` into Anthropic's Messages-API
    shape.

    Anthropic spells "any of them" as ``{"type": "any"}`` rather than
    ``"required"``, and always uses an object rather than a bare string.
    """
    if choice is None:
        return None
    if choice == TOOL_CHOICE_AUTO:
        return {"type": "auto"}
    if choice == TOOL_CHOICE_REQUIRED:
        return {"type": "any"}
    return {"type": "tool", "name": choice}


def resolve_tool_choice_for_thinking(
    choice: Optional[str], *, thinking: bool, rejects_forced: bool,
) -> Optional[str]:
    """Downgrade a COMPELLING ``tool_choice`` to ``"auto"`` when the model
    rejects forcing in thinking mode.

    DeepSeek 400s the whole request with ``"Thinking mode does not support
    this tool_choice"`` for BOTH a named tool and ``"required"`` — only
    ``"auto"`` (or omission) is accepted alongside ``thinking:
    {"type": "enabled"}``. Verified against deepseek-v4-pro 2026-09-13.

    Since every routing role in a default config points at DeepSeek, a
    caller that combines a thinking role with a forced tool would hard-fail
    every dispatch. Downgrading (rather than raising) keeps the gateway's
    fail-open posture: the tools array still goes out, the model still
    overwhelmingly calls the tool, and the caller's own fallback handles
    the residue. Losing the guarantee beats losing the request.
    """
    if choice is None or not thinking or not rejects_forced:
        return choice
    if choice == TOOL_CHOICE_AUTO:
        return choice
    logger.warning(
        "[tool_schemas] tool_choice=%r downgraded to 'auto': this model "
        "rejects a compelled tool_choice in thinking mode. Structure is "
        "no longer guaranteed — keep the caller's fallback path live.",
        choice,
    )
    return TOOL_CHOICE_AUTO


def validate_tool_choice(
    choice: Optional[str], tools: Optional[list[dict[str, Any]]],
) -> Optional[str]:
    """Return ``choice`` when it is coherent against ``tools``, else None.

    Guards the two ways a caller can produce a guaranteed-400 request:
    naming a tool that was never supplied, and asking for a choice with no
    tools at all. Returning None (rather than raising) keeps the gateway's
    fail-open contract — a malformed choice degrades to the provider
    default instead of aborting a paid dispatch.
    """
    if choice is None:
        return None
    if not tools:
        logger.warning(
            "[tool_schemas] tool_choice=%r ignored: no tools supplied.", choice,
        )
        return None
    if choice in (TOOL_CHOICE_AUTO, TOOL_CHOICE_REQUIRED):
        return choice
    names = {str(t.get("name")) for t in tools if isinstance(t, dict)}
    if choice not in names:
        logger.warning(
            "[tool_schemas] tool_choice=%r ignored: not among supplied "
            "tools %s.", choice, sorted(names),
        )
        return None
    return choice


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
    if name == "rewrite_file":
        return PatchBlock(
            operation=OperationType.REWRITE_FILE,
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
    # Line-coordinate ops. Schemas are registered in PATCH_TOOLS so the
    # LLM can emit these via structured tool-use. Layer-2 rule-table
    # autofixes and Layer-1 semgrep ``extra.fix`` patches also construct
    # the same PatchBlock shape directly without going through tool
    # calls. The dispatch branches handle both paths uniformly.
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
