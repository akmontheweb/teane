"""``teane chat`` — interactive refinement REPL (#8).

Why this exists
===============
``teane run`` is autonomous. Once started it owns the loop until the
build is clean, the budget is gone, or HITL intervenes. That's the
right contract for "drop a prompt and walk away" workflows — but
operators routinely want the inverse: a quick conversational
back-and-forth against a workspace. "Read this file, suggest an
approach, let me iterate, and *maybe* apply a patch when I say so."

That's what ``teane chat`` is. It:
- Reuses the Gateway, redactor, and tool-loop infrastructure so token
  budgeting, secret stripping, web tools, and MCP all keep working.
- Reuses the per-repo memory and (when enabled) repo-index injection
  paths so the chat has the same priors the planner does.
- Per-patch HITL approval — the LLM can emit SEARCH/REPLACE blocks
  but they're never applied without the operator typing ``/apply``.
- ``/build`` runs the workspace's configured build command in the
  same sandbox the graph uses.

What this is NOT
================
- Not persistent across invocations (v1). Each ``teane chat`` starts
  a fresh in-memory conversation. ``--resume`` is a clean follow-up.
- Not the place to scaffold a new project — that's still ``teane run``.
- Not a wire-protocol IDE client — the ``HttpChannel`` HITL transport
  already covers that use case for editor integrations.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from harness.graph import MessageDict

logger = logging.getLogger(__name__)


_HELP_TEXT = """\
Interactive teane chat commands:

  /help              show this help
  /exit  /quit       end the session
  /clear             reset the conversation history
  /files             list files modified during this session
  /apply             apply SEARCH/REPLACE patches from the LAST assistant
                     reply (with per-patch confirmation)
  /build             run the workspace's configured build command in the
                     sandbox; report exit code + first 80 lines of output
  /save <path>       save the conversation transcript to <path>
  /budget            show remaining budget + total spend
  /memory            re-read the per-repo memory file and re-inject it

Anything that does not start with `/` is sent to the LLM as a user
message. Patches and tool blocks in the response are NOT applied
automatically — use /apply to commit them.
"""


_DEFAULT_SYSTEM_PROMPT = """\
You are a code assistant in an interactive REPL. The user is iterating
on a real workspace; you can see file paths and prior memory but you
do not have unrestricted shell access — anything you propose ships
only when the user types `/apply`.

When asked to inspect a file, emit `<<<READ_FILE path="..." >>>` and
wait for the read result on the next user turn. When ready to suggest a
change, emit standard SEARCH/REPLACE blocks against absolute or
workspace-relative paths. Never apply blindly — explain what each
patch does and ask for confirmation before producing the next set.

Web research and MCP tools are available via the existing text-DSL
blocks (`<<<WEB_FETCH>>>`, `<<<WEB_SEARCH>>>`, `<<<MCP_CALL>>>`).
"""


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class ChatSession:
    workspace_path: str
    gateway: Any
    config: dict[str, Any]
    budget_remaining_usd: float
    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    total_cost_usd: float = 0.0
    # Decoupled I/O so the unit tests can drive the REPL with stubs.
    reader: Callable[[str], str] = field(default_factory=lambda: input)
    writer: Callable[[str], None] = field(default_factory=lambda: print)


def _system_prompt(session: ChatSession) -> str:
    head = _DEFAULT_SYSTEM_PROMPT
    return (
        f"{head}\n"
        f"Workspace: {session.workspace_path}\n"
        f"Session id: {session.session_id}\n"
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

async def run_chat(
    *,
    workspace_path: str,
    gateway: Any,
    config: dict[str, Any],
    initial_budget_usd: float,
    session_id: str = "",
    reader: Optional[Callable[[str], str]] = None,
    writer: Optional[Callable[[str], None]] = None,
) -> int:
    """Drive the chat REPL until the user exits.

    Returns ``0`` on clean exit, non-zero on a hard error during the
    final cleanup (memory write etc.).
    """
    if not session_id:
        import uuid
        session_id = f"chat-{uuid.uuid4().hex[:8]}"
    if reader is None:
        reader = input
    if writer is None:
        writer = print
    session = ChatSession(
        workspace_path=os.path.abspath(workspace_path),
        gateway=gateway,
        config=config,
        budget_remaining_usd=float(initial_budget_usd),
        session_id=session_id,
        reader=reader,
        writer=writer,
    )

    # Pre-load: system prompt + memory + (optional) repo index injection.
    session.messages.append({
        "role": "system", "content": _system_prompt(session),
    })
    _inject_repo_memory(session)
    # The repo-index injection requires a query — defer until the
    # first user turn.

    writer("teane chat — type /help for commands, /exit to quit.")
    writer(f"workspace: {session.workspace_path}")
    writer(f"budget   : ${session.budget_remaining_usd:.2f}")
    writer("")

    while True:
        try:
            line = reader("you> ")
        except (EOFError, KeyboardInterrupt):
            writer("")
            break
        if not line:
            continue
        if line.startswith("/"):
            should_exit = await _handle_command(session, line)
            if should_exit:
                break
            continue
        # Plain user message → LLM dispatch.
        cont = await _handle_user_turn(session, line)
        if not cont:
            break

    # End-of-session: append a memory entry summarising the chat.
    _persist_session_summary(session)
    return 0


# ---------------------------------------------------------------------------
# Per-turn LLM dispatch
# ---------------------------------------------------------------------------

async def _handle_user_turn(session: ChatSession, user_text: str) -> bool:
    if session.budget_remaining_usd <= 0.0:
        session.writer(
            "[chat] budget exhausted (use /budget to confirm). "
            "Raise hard_cap_usd or end the session."
        )
        return False

    # Inject repo-index hits scoped to the current turn (every prompt
    # gets its own retrieval — different turns ask different things).
    _maybe_inject_repo_index(session, user_text)

    session.messages.append({"role": "user", "content": user_text})

    try:
        from harness.gateway import NodeRole
        response, new_budget = await session.gateway.dispatch(
            messages=list(session.messages),
            role=NodeRole.PLANNING,
            budget_remaining_usd=session.budget_remaining_usd,
        )
    except Exception as exc:  # noqa: BLE001 — never crash the REPL on a bad dispatch
        session.writer(f"[chat] gateway error: {exc}")
        # Drop the user turn we appended so the conversation stays clean.
        session.messages.pop()
        return True
    cost = max(0.0, session.budget_remaining_usd - new_budget)
    session.total_cost_usd += cost
    session.budget_remaining_usd = new_budget

    # Resolve tool blocks (web/MCP) in the response before showing it.
    # Mirrors the pattern planning_node uses.
    try:
        from harness.graph import _run_tool_loop, _web_tool_cap_from_state
        from harness.gateway import NodeRole as _NR
        cap = _web_tool_cap_from_state({})
        final_content, new_messages, new_budget, rounds = await _run_tool_loop(
            initial_response_content=response.content,
            messages=cast("list[MessageDict]", session.messages),
            gateway=session.gateway,
            role=_NR.PLANNING,
            budget=session.budget_remaining_usd,
            cap=cap,
        )
        session.messages = cast("list[dict[str, Any]]", new_messages)
        session.budget_remaining_usd = new_budget
        if rounds:
            session.writer(f"[chat] ran {rounds} tool round(s)")
    except Exception as exc:  # noqa: BLE001
        logger.debug("[chat] tool loop skipped: %s", exc)
        final_content = response.content

    session.messages.append({"role": "assistant", "content": final_content})

    session.writer("")
    session.writer("llm>")
    session.writer(final_content)
    session.writer("")
    session.writer(
        f"[cost ${cost:.4f} / spent ${session.total_cost_usd:.4f} / "
        f"left ${session.budget_remaining_usd:.4f}]"
    )
    return True


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

async def _handle_command(session: ChatSession, line: str) -> bool:
    """Dispatch a ``/`` command. Returns ``True`` when the REPL should exit."""
    parts = line.strip().split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("/exit", "/quit", "/q"):
        session.writer("[chat] bye.")
        return True
    if cmd == "/help":
        session.writer(_HELP_TEXT)
        return False
    if cmd == "/clear":
        # Keep the seeded system prompts; drop the conversation history.
        keep = [m for m in session.messages if m.get("role") == "system"]
        session.messages = keep
        session.writer("[chat] cleared conversation history (system prompts kept).")
        return False
    if cmd == "/files":
        if not session.modified_files:
            session.writer("[chat] no files modified yet this session.")
        else:
            for f in session.modified_files:
                session.writer(f"  {f}")
        return False
    if cmd == "/budget":
        session.writer(
            f"[chat] spent ${session.total_cost_usd:.4f} / "
            f"left  ${session.budget_remaining_usd:.4f}"
        )
        return False
    if cmd == "/memory":
        _inject_repo_memory(session, replace_existing=True)
        session.writer("[chat] re-read per-repo memory and re-injected.")
        return False
    if cmd == "/save":
        if not arg:
            session.writer("[chat] usage: /save <path>")
            return False
        await _save_transcript(session, arg)
        return False
    if cmd == "/apply":
        await _apply_patches_from_last(session)
        return False
    if cmd == "/build":
        await _run_build(session)
        return False
    session.writer(f"[chat] unknown command {cmd!r}. /help for the list.")
    return False


# ---------------------------------------------------------------------------
# /apply  — process the last assistant reply through the patcher
# ---------------------------------------------------------------------------

async def _apply_patches_from_last(session: ChatSession) -> None:
    last_assistant = next(
        (m for m in reversed(session.messages) if m.get("role") == "assistant"),
        None,
    )
    if last_assistant is None:
        session.writer("[chat] no assistant reply yet — nothing to apply.")
        return
    content = str(last_assistant.get("content") or "")
    try:
        from harness.patcher import process_llm_patch_output
    except ImportError as exc:
        session.writer(f"[chat] patcher import failed: {exc}")
        return

    # HITL prompt — patcher may rewrite multiple files. v1 keeps the
    # confirmation coarse-grained ("apply N blocks? [y/N]") so the LLM
    # cannot fragment intent across many micro-prompts.
    session.writer("[chat] scanning last reply for patch blocks ...")
    ack = session.reader("[chat] apply patches now? [y/N] ").strip().lower()
    if ack not in ("y", "yes"):
        session.writer("[chat] skipped.")
        return
    try:
        results, new_modified = await process_llm_patch_output(
            content, session.workspace_path,
            existing_modified_files=session.modified_files,
        )
    except Exception as exc:  # noqa: BLE001
        session.writer(f"[chat] patch application failed: {exc}")
        return
    if not results:
        session.writer("[chat] no patch blocks detected in the last reply.")
        return
    ok = sum(1 for r in results if getattr(r, "success", False))
    fail = len(results) - ok
    session.modified_files = list(dict.fromkeys(session.modified_files + new_modified))
    session.writer(
        f"[chat] applied {ok} block(s); {fail} failed. "
        f"Modified files now: {len(session.modified_files)}"
    )
    if fail:
        for r in results:
            if not getattr(r, "success", False):
                session.writer(f"  - failed: {getattr(r, 'file_path', '?')}: "
                                f"{getattr(r, 'error', '?')}")


# ---------------------------------------------------------------------------
# /build  — run the configured build command in the sandbox
# ---------------------------------------------------------------------------

async def _run_build(session: ChatSession) -> None:
    build_command = session.config.get("build_command") or "make build"
    session.writer(f"[chat] running build: {build_command}")
    try:
        from harness.sandbox import SandboxExecutor
        executor = SandboxExecutor(
            workspace_path=session.workspace_path,
            backend=(session.config.get("sandbox") or {}).get("backend", "auto"),
            allow_network=bool(session.config.get("allow_network", False)),
        )
        result = await executor.run(build_command)
    except Exception as exc:  # noqa: BLE001
        session.writer(f"[chat] build failed to start: {exc}")
        return
    output = (getattr(result, "stdout", "") or "") + (getattr(result, "stderr", "") or "")
    lines = output.splitlines()
    head = "\n".join(lines[:80])
    session.writer(head or "[chat] (no output)")
    if len(lines) > 80:
        session.writer(f"... ({len(lines) - 80} more lines)")
    session.writer(f"[chat] exit code: {getattr(result, 'exit_code', '?')}")


# ---------------------------------------------------------------------------
# Memory + index injection
# ---------------------------------------------------------------------------

def _inject_repo_memory(
    session: ChatSession, *, replace_existing: bool = False,
) -> None:
    try:
        from harness.repo_memory import RepoMemoryConfig, read_repo_memory
        mem_cfg = RepoMemoryConfig.from_config(session.config)
        if not mem_cfg.enabled:
            return
        text = read_repo_memory(session.workspace_path, mem_cfg)
        if not text:
            return
        marker = "### Prior session memory for this repository"
        block = (
            f"{marker}\n\n"
            f"Each entry summarises a past run on this repo.\n\n{text}"
        )
        if replace_existing:
            session.messages = [
                m for m in session.messages
                if not (
                    m.get("role") == "system"
                    and marker in str(m.get("content") or "")
                )
            ]
        # Insert after the very first system message so the canonical
        # system prompt stays at index 0 for cache-marker purposes.
        idx = 1 if session.messages else 0
        session.messages.insert(idx, {"role": "system", "content": block})
    except Exception as exc:  # noqa: BLE001
        logger.debug("[chat] memory injection skipped: %s", exc)


def _maybe_inject_repo_index(session: ChatSession, query: str) -> None:
    try:
        from harness.repo_index import (
            RepoIndexConfig,
            query_top_chunks,
            render_results_for_injection,
        )
        idx_cfg = RepoIndexConfig.from_config(session.config)
        if not idx_cfg.enabled:
            return
        results = query_top_chunks(session.workspace_path, query, cfg=idx_cfg)
        block = render_results_for_injection(
            results, max_bytes=idx_cfg.inject_max_bytes,
        )
        if not block:
            return
        marker = "### Repository context (semantic retrieval)"
        # Replace any prior retrieval injection — each turn gets its own.
        session.messages = [
            m for m in session.messages
            if not (
                m.get("role") == "system"
                and marker in str(m.get("content") or "")
            )
        ]
        idx = 1 if session.messages else 0
        session.messages.insert(idx, {
            "role": "system",
            "content": f"{marker}\n\n" + block,
        })
    except Exception as exc:  # noqa: BLE001
        logger.debug("[chat] index injection skipped: %s", exc)


# ---------------------------------------------------------------------------
# End-of-session memory write + transcript save
# ---------------------------------------------------------------------------

def _persist_session_summary(session: ChatSession) -> None:
    try:
        from harness.repo_memory import RepoMemoryConfig, append_session_note
        mem_cfg = RepoMemoryConfig.from_config(session.config)
        if not mem_cfg.enabled:
            return
        # Summarise the first user message as the "prompt" — a chat
        # session can have many turns but the original ask is the
        # signal worth preserving.
        first_user = next(
            (m for m in session.messages if m.get("role") == "user"),
            {"content": "(empty chat session)"},
        )
        summary = str(first_user.get("content") or "")
        append_session_note(
            session.workspace_path,
            session_id=session.session_id,
            prompt_summary=summary,
            modified_files=list(session.modified_files),
            exit_code=0,
            cfg=mem_cfg,
            extra_notes=f"interactive chat ({len(session.messages)} messages, "
                        f"${session.total_cost_usd:.4f})",
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("[chat] memory persist skipped: %s", exc)


async def _save_transcript(session: ChatSession, dest: str) -> None:
    target = os.path.expanduser(dest)
    try:
        with open(target, "w", encoding="utf-8") as f:
            for m in session.messages:
                role = m.get("role", "?")
                content = str(m.get("content") or "")
                f.write(f"---\n## {role}\n---\n{content}\n\n")
    except OSError as exc:
        session.writer(f"[chat] could not write {target}: {exc}")
        return
    session.writer(f"[chat] saved transcript to {target}")
