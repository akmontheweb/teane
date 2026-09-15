"""Recognise tool invocations a model wrote as TEXT rather than as native
tool calls, across every major dialect.

Why this exists
---------------
When native tool-use is off — a role the harness runs tool-less, a provider
that dropped the ``tools`` array, a model that simply ignored it — a model
still asks for things. It just asks in whatever syntax its training baked
in. The harness's own text DSL (``<<<READ_FILE>>>``) is one dialect among
many, and a request in any other one parses as prose.

lumina-run5-20260914-2120 lost four repair rounds and terminated on
``zero_patch_loop`` because deepseek-v4-pro asked to read a file in
Anthropic's XML syntax. One of those rounds spent $0.010 to produce 29
output tokens. This is not one provider's quirk: it is a known, active
problem across the ecosystem (anthropics/claude-code#49747 is the same
failure in the other direction — a Claude model reverting to its legacy
XML format mid-response under length pressure).

Design
------
Enumeration alone is a losing strategy — the next model ships a new
dialect and you find out in production. So the module has three tiers:

1. ``_DIALECTS`` — literal delimiters for the families that exist today.
2. ``_generic_json_invocations`` — a permissive scan for the JSON core
   (``name`` + ``arguments``/``parameters``) that nearly every dialect
   wraps. This catches formats nobody has catalogued.
3. Whatever still fails to parse is REPORTED rather than silently
   dropped, so the caller can tell the model it used an unrecognised
   syntax instead of burning the round. See ``graph.py``'s unparsed-round
   handling — a round that lands nothing because it was not understood
   must never be mistaken for a model that refuses to act.

Argument-key note: Llama uses ``parameters`` where everyone else uses
``arguments``. Both are accepted everywhere; a parser keying on one
silently misses the other.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Canonical arg aliases for a file path, most-specific first.
_PATH_KEYS = ("file", "path", "file_path", "filename", "filepath")
_RANGE_KEYS = ("range", "lines", "line_range")


# ---------------------------------------------------------------------------
# Reasoning tags
# ---------------------------------------------------------------------------
# DeepSeek-R1, Qwen3 and GLM emit <think>...</think>, and an invocation can
# sit inside or after it. Strip reasoning BEFORE extracting, or a tool call
# the model only *considered* gets executed.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove ``<think>...</think>`` spans."""
    return _THINK_RE.sub("", text or "")


# ---------------------------------------------------------------------------
# Dialect 1 — Anthropic XML  <invoke name="x"><parameter name="k">v</parameter>
# ---------------------------------------------------------------------------
_ANTHROPIC_INVOKE = re.compile(
    r"<invoke\s+name=[\"'](?P<name>[^\"']+)[\"']\s*>(?P<body>.*?)</invoke>",
    re.DOTALL | re.IGNORECASE,
)
_ANTHROPIC_PARAM = re.compile(
    r"<parameter\s+name=[\"'](?P<key>[^\"']+)[\"']\s*>(?P<val>.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Dialect 2 — Hermes / Qwen  <tool_call>{json}</tool_call>
# ---------------------------------------------------------------------------
_HERMES = re.compile(
    r"<tool_call>\s*(?P<json>\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Dialect 3 — Mistral  [TOOL_CALLS] [ {...}, {...} ]
# ---------------------------------------------------------------------------
_MISTRAL = re.compile(r"\[TOOL_CALLS\]\s*(?P<json>\[.*?\])", re.DOTALL)

# ---------------------------------------------------------------------------
# Dialect 4 — Llama 3.x  <|python_tag|>{json}
# ---------------------------------------------------------------------------
# One level of brace nesting, because the payload is
# {"name": ..., "parameters": {...}} — a non-greedy \{.*?\} truncates at the
# inner closing brace and silently yields unparseable JSON.
_LLAMA_TAG = re.compile(
    r"<\|python_tag\|>\s*(?P<json>\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})",
    re.DOTALL,
)

# ---------------------------------------------------------------------------
# Dialect 5 — DeepSeek special tokens.
# NOTE the non-ASCII characters: U+FF5C fullwidth pipe and U+2581 lower-one-
# eighth block. A regex written with ASCII "|" and "_" will never match.
# ---------------------------------------------------------------------------
_DEEPSEEK = re.compile(
    "<｜tool▁call▁begin｜>"
    r"(?P<name>.*?)"
    "<｜tool▁sep｜>"
    r"(?P<json>.*?)"
    "<｜tool▁call▁end｜>",
    re.DOTALL,
)

# ---------------------------------------------------------------------------
# Dialect 6 — Nemotron  <function=name><parameter=key>value</parameter>
# ---------------------------------------------------------------------------
_NEMOTRON = re.compile(
    r"<function=(?P<name>[A-Za-z0-9_.-]+)\s*>(?P<body>.*?)</function>",
    re.DOTALL | re.IGNORECASE,
)
_NEMOTRON_PARAM = re.compile(
    r"<parameter=(?P<key>[A-Za-z0-9_.-]+)\s*>(?P<val>.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Tier 2 — the JSON core almost every dialect wraps.
# Guarded hard: a bare object only counts when it has BOTH a string ``name``
# and an ``arguments``/``parameters`` mapping, so ordinary JSON in a code
# block cannot be mistaken for an invocation.
# ---------------------------------------------------------------------------
_JSON_OBJ = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _args_of(payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The arguments mapping under either key name, or None."""
    for key in ("arguments", "parameters", "args", "input"):
        val = payload.get(key)
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _invocation(
    name: str, args: dict[str, Any], dialect: str, raw: str = "",
) -> dict[str, Any]:
    """``raw`` is the literal matched text, so a caller can splice a
    rewritten block back in place of exactly what the model wrote."""
    return {
        "name": str(name).strip(), "args": args,
        "dialect": dialect, "raw": raw,
    }


def _from_json_payload(
    raw: str, dialect: str, span: str = "",
) -> list[dict[str, Any]]:
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return []
    items = payload if isinstance(payload, list) else [payload]
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("function")
        if isinstance(name, dict):           # OpenAI-ish {"function": {"name": ...}}
            name = name.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        out.append(_invocation(name, _args_of(item) or {}, dialect, span or raw))
    return out


def _generic_json_invocations(text: str) -> list[dict[str, Any]]:
    """Tier 2: any JSON object carrying a name plus an arguments mapping."""
    out: list[dict[str, Any]] = []
    for m in _JSON_OBJ.finditer(text):
        blob = m.group(0)
        if '"name"' not in blob:
            continue
        if not any(f'"{k}"' in blob for k in ("arguments", "parameters", "args", "input")):
            continue
        out.extend(_from_json_payload(blob, "generic_json", blob))
    return out


def extract_tool_invocations(text: str) -> list[dict[str, Any]]:
    """Every tool invocation written as text, across all known dialects.

    Returns ``[{"name", "args", "dialect"}]``. Reasoning spans are stripped
    first. Tier-2 generic JSON only runs when no literal dialect matched,
    so a well-formed ``<tool_call>`` is never double-counted.
    """
    if not text:
        return []
    body = strip_reasoning(text)
    out: list[dict[str, Any]] = []

    for m in _ANTHROPIC_INVOKE.finditer(body):
        args = {
            pm.group("key").lower(): pm.group("val").strip()
            for pm in _ANTHROPIC_PARAM.finditer(m.group("body"))
        }
        out.append(_invocation(m.group("name"), args, "anthropic_xml", m.group(0)))

    for m in _NEMOTRON.finditer(body):
        args = {
            pm.group("key").lower(): pm.group("val").strip()
            for pm in _NEMOTRON_PARAM.finditer(m.group("body"))
        }
        out.append(_invocation(m.group("name"), args, "nemotron", m.group(0)))

    for m in _HERMES.finditer(body):
        out.extend(_from_json_payload(m.group("json"), "hermes_qwen", m.group(0)))
    for m in _MISTRAL.finditer(body):
        out.extend(_from_json_payload(m.group("json"), "mistral", m.group(0)))
    for m in _LLAMA_TAG.finditer(body):
        out.extend(_from_json_payload(m.group("json"), "llama_python_tag", m.group(0)))
    for m in _DEEPSEEK.finditer(body):
        name = m.group("name").strip()
        try:
            args = json.loads(m.group("json"))
        except (ValueError, TypeError):
            args = {}
        if name:
            out.append(
                _invocation(
                    name, args if isinstance(args, dict) else {},
                    "deepseek", m.group(0),
                )
            )

    if not out:
        out = _generic_json_invocations(body)
    return out


# ---------------------------------------------------------------------------
# Mapping to the harness DSL
# ---------------------------------------------------------------------------
# Only read_file is rewritten today: it is what a tool-less model actually
# asks for, and it is the one the harness can satisfy inline without
# consuming a repair slot. Anything else is REPORTED, not guessed at —
# silently converting an unrecognised call into a file operation would be
# far worse than telling the model its syntax was not understood.
_READ_ALIASES = frozenset({
    "read_file", "readfile", "read", "view_file", "view", "cat", "open_file",
    "str_replace_based_edit_tool",  # Anthropic text-editor tool, view command
})


def _first(args: dict[str, Any], keys: tuple[str, ...]) -> Optional[str]:
    for k in keys:
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, (int, float)):
            return str(v)
    return None


def to_read_file_dsl(inv: dict[str, Any]) -> Optional[str]:
    """Render a read-file invocation as the canonical DSL block, or None."""
    if inv.get("name", "").lower() not in _READ_ALIASES:
        return None
    args = inv.get("args") or {}
    path = _first(args, _PATH_KEYS)
    if not path:
        return None
    lines = ["<<<READ_FILE>>>", f"file: {path}"]
    rng = _first(args, _RANGE_KEYS)
    if rng and re.fullmatch(r"\d+\s*-\s*\d+|\d+", rng.strip()):
        lines.append(f"range: {rng.strip()}")
    lines.append("<<<END_READ_FILE>>>")
    return "\n".join(lines)


def unmapped_invocations(text: str) -> list[dict[str, Any]]:
    """Text-dialect invocations the harness cannot translate.

    The caller uses this to tell the model its syntax was not understood —
    the difference between "could not act" and "would not act", which every
    downstream progress counter depends on.
    """
    return [
        inv for inv in extract_tool_invocations(text)
        if to_read_file_dsl(inv) is None
    ]
