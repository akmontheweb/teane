"""
Interactive setup wizard for bare ``teane run`` invocations.

When the user types ``teane run`` with no flags, ``cmd_run`` calls
:func:`run_setup_wizard` to walk them through the minimum set of choices
needed to start a run. The wizard first asks whether this is a new
session or a resume of an existing checkpointed session:

- "new"    → API keys, workspace, prompt, --git, --new-build, --spec-discovery.
- "resume" → session id (picked from a recent-sessions list or typed
             free-text); ``cmd_run`` then delegates to ``cmd_resume``.

:func:`run_setup_wizard` returns a mode string (``"run"`` or ``"resume"``)
so the caller can dispatch correctly. ``args`` is mutated in place either
way.

The wizard does NOT persist anything. Each bare ``teane run`` re-asks
every question. Model routing, sandbox backend, lintgate, deployment,
and budget all stay in ``config/config.json``.

Reuses :func:`harness.hitl.get_channel` for every non-secret prompt so
the wizard inherits the existing HITL infrastructure (file-replay for
tests, HTTP webhook for IDE plugins). API key prompts bypass HitlChannel
and use :func:`getpass.getpass` so keys never echo to the terminal.
"""

from __future__ import annotations

import argparse
import getpass
import logging
import os
import sys
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from harness.hitl import HitlChannel

logger = logging.getLogger(__name__)


def run_setup_wizard(args: argparse.Namespace) -> str:
    """Drive the interactive setup flow and mutate ``args`` in place.

    Order:
        0. New vs resume. ``resume`` skips the rest of the wizard, asks for
           a session id, and returns ``"resume"`` so the caller delegates
           to :func:`harness.cli.cmd_resume`. ``new`` continues below.
        1. API key check — prompt for any missing ``{PROVIDER}_API_KEY``
           env vars for models referenced by ``model_routing``.
        2. Workspace path.
        3. Engineering prompt / task.
        4. ``--git true|false``.
        5. ``--new-build true|false``.
        6. ``--spec-discovery true|false``.
        7. Summary + confirm. ``n`` loops back into the wizard from step 2.

    Returns ``"run"`` for a fresh run or ``"resume"`` to delegate to
    ``cmd_resume``. Raises ``SystemExit(2)`` when invoked non-interactively
    or when the user declines to provide a required API key.
    """
    # Imported lazily so the wizard module doesn't drag cli.py's import
    # graph into every script that just wants to query argparse defaults.
    from harness.cli import ConfigError, load_raw_config
    from harness.hitl import get_channel

    channel = get_channel()
    if not channel.is_interactive():
        print(
            "\nInteractive setup required: `teane run` was invoked with no\n"
            "--workspace or --prompt, but stdin is not a terminal (or auto-approve\n"
            "is set). Either run from a TTY, or pass --workspace and --prompt\n"
            "explicitly.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    print()
    print("=" * 72)
    print("teane run — interactive setup")
    print("=" * 72)
    print(
        "You ran `teane run` without any flags. The wizard will walk you\n"
        "through the minimum settings needed to start a run. None of your\n"
        "answers will be persisted — every bare `teane run` re-asks.\n"
    )

    # Config is loaded once and reused for the API-key check and for the
    # checkpoint-db path that the session picker reads.
    try:
        config = load_raw_config()
    except ConfigError as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    # ------------------------------------------------------------------
    # Step 0: new session vs resume existing session
    # ------------------------------------------------------------------
    if _ask_session_mode(channel) == "resume":
        # Resumed sessions reuse the keys validated by the original run.
        # If a key has been unset since, cmd_resume's first LLM dispatch
        # will surface the error — no point re-prompting here.
        db_path = config.get("persistence", {}).get(
            "db_path", "~/.harness/checkpoints.db",
        )
        args.session_id = _ask_session_id(channel, db_path)
        return "resume"

    # ------------------------------------------------------------------
    # Step 1: API key prerequisite check (new sessions only)
    # ------------------------------------------------------------------
    _check_and_prompt_api_keys(config)

    # ------------------------------------------------------------------
    # Steps 2-6: runtime choices, with a summary-confirm loop
    # ------------------------------------------------------------------
    while True:
        workspace = _ask_workspace(channel)
        prompt = _ask_prompt(channel)
        git_mode = _ask_git(channel)
        new_build = _ask_new_build(channel)
        discover = _ask_discover(channel)

        _print_summary(workspace, prompt, git_mode, new_build, discover)
        if channel.confirm("Run with these settings?", default=True):
            break
        print("\nLet's go through the choices again.\n")

    args.workspace = workspace
    args.prompt = prompt
    args.git = git_mode
    args.new_build = new_build
    args.spec_discovery = discover
    # If the operator picks --new-build via the wizard, they've already
    # given explicit consent — skip the secondary --yes prompt that
    # cmd_run would otherwise show.
    if new_build:
        args.assume_yes = True
    return "run"


# ---------------------------------------------------------------------------
# Step 1 — API keys
# ---------------------------------------------------------------------------

def _check_and_prompt_api_keys(config: dict[str, Any]) -> None:
    """Scan ``config["models"]`` for any provider whose ``{PROVIDER}_API_KEY``
    env var is unset, and prompt the user to enter each missing key. Sets
    ``os.environ[env_var]`` so downstream code (gateway dispatch, sandbox
    subprocesses) sees the value. Keys are NOT written to disk."""
    from harness.cli import find_missing_api_keys

    missing = find_missing_api_keys(config)
    if not missing:
        return

    print(
        "\nThe harness needs API keys for the LLM models in your "
        "config.json,\nbut some are not set in the environment. Enter "
        "them now — your\ninput is hidden and not written to disk."
    )
    for env_var in sorted(missing):
        models_using = ", ".join(missing[env_var])
        for attempt in range(2):
            try:
                value = getpass.getpass(
                    f"  {env_var} (for {models_using}): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nSetup interrupted.", file=sys.stderr)
                raise SystemExit(2)
            if value:
                os.environ[env_var] = value
                print("    set (in this process only).")
                break
            if attempt == 0:
                print("    Empty — try again.")
        else:
            print(
                f"\nNo value provided for {env_var}. The harness can't run "
                f"without it. Aborting.",
                file=sys.stderr,
            )
            raise SystemExit(2)


# ---------------------------------------------------------------------------
# Step 0 — new vs resume
# ---------------------------------------------------------------------------

def _ask_session_mode(channel: "HitlChannel") -> str:
    """Ask whether to start a new session or resume an existing one.

    Returns ``"run"`` (new) or ``"resume"``. Default is ``"run"`` so a
    user who hits Enter through every prompt gets today's behavior.
    """
    print("\nStep 0: Start a new session or resume an existing one?")
    print("  n = new session (walk through full setup, fresh checkpoint)")
    print("  r = resume existing session (restore from checkpoint by session id)")
    choice = channel.prompt(
        "Choose [n/r]", options=["n", "r"], default="n",
    ).strip().lower()
    return "resume" if choice == "r" else "run"


def _list_recent_sessions_sync(
    db_path: str, limit: int = 10,
) -> list[tuple[str, str, str, str]]:
    """Return ``[(thread_id, workspace_path, created_ts, updated_ts)]`` for
    the most recently updated sessions in the checkpoint DB.

    ``created_ts`` comes from the *earliest* checkpoint row for the thread;
    ``updated_ts`` from the *latest*. Both are raw ISO-8601 strings — the
    caller is responsible for formatting (or accepting ``""`` when the blob
    is corrupted or missing a ``ts`` field).

    Synchronous mirror of :func:`harness.storage.list_all_sessions`'s SELECT
    so the wizard can render a picker without spinning up an async runtime
    from inside the already-running ``cmd_run`` event loop.

    Returns an empty list when the DB is missing, empty, or unreadable.
    """
    expanded = os.path.expanduser(db_path)
    if not os.path.isfile(expanded):
        return []

    try:
        from harness.storage import _deserialize_checkpoint_blob
    except ImportError:
        return []

    import sqlite3

    rows: list[tuple[str, str, str, str]] = []
    try:
        with sqlite3.connect(expanded) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT c.thread_id, c.checkpoint
                   FROM checkpoints c
                   INNER JOIN (
                       SELECT thread_id, MAX(checkpoint_id) AS max_cp_id
                       FROM checkpoints
                       GROUP BY thread_id
                   ) AS latest ON c.thread_id = latest.thread_id
                              AND c.checkpoint_id = latest.max_cp_id
                   ORDER BY c.checkpoint_id DESC
                   LIMIT ?""",
                (limit,),
            )
            latest_rows = cursor.fetchall()

            # Earliest-checkpoint ts per thread for the "created" column.
            # Up to 10 thread_ids → 10 indexed lookups, cheap to issue
            # one-at-a-time and keeps the query trivial to read.
            for row in latest_rows:
                thread_id = row["thread_id"]
                workspace_path = ""
                updated_ts = ""
                try:
                    cp = _deserialize_checkpoint_blob(row["checkpoint"])
                    updated_ts = str(cp.get("ts", "") or "")
                    channel_values = cp.get("channel_values", {})
                    if isinstance(channel_values, dict):
                        wp = channel_values.get("workspace_path", "")
                        if isinstance(wp, dict):
                            wp = wp.get("value", "")
                        workspace_path = str(wp) if wp else ""
                except Exception:
                    # Corrupt blob — still show the thread id so the
                    # operator can purge it or pick a different session.
                    pass

                created_ts = ""
                try:
                    first = conn.execute(
                        "SELECT checkpoint FROM checkpoints "
                        "WHERE thread_id = ? "
                        "ORDER BY checkpoint_id ASC LIMIT 1",
                        (thread_id,),
                    ).fetchone()
                    if first is not None:
                        first_cp = _deserialize_checkpoint_blob(first["checkpoint"])
                        created_ts = str(first_cp.get("ts", "") or "")
                except Exception:
                    pass

                rows.append((thread_id, workspace_path, created_ts, updated_ts))
    except sqlite3.Error:
        return []
    return rows


def _ask_session_id(channel: "HitlChannel", db_path: str) -> str:
    """Prompt for a session id. Show a numbered list of recent sessions
    when the checkpoint DB has any; the operator can pick a number or
    type any session id directly.

    Re-prompts on empty input. No existence check — ``cmd_resume``
    validates and prints a clean error for unknown ids.
    """
    try:
        from harness.storage import _format_checkpoint_ts
    except ImportError:
        def _format_checkpoint_ts(ts: str) -> str:  # type: ignore[misc]
            return ts or "(unknown)"

    recent = _list_recent_sessions_sync(db_path, limit=10)

    if recent:
        print("\nRecent checkpointed sessions (most recent first):")
        for i, (thread_id, workspace_path, created_ts, updated_ts) in enumerate(
            recent, start=1,
        ):
            created = _format_checkpoint_ts(created_ts)
            updated = _format_checkpoint_ts(updated_ts)
            where = workspace_path or "(workspace unknown)"
            print(f"  {i:>2}. {thread_id}")
            # When a session has only one checkpoint, created == updated;
            # collapse to a single line so the picker stays scannable.
            if created == updated:
                print(f"      {created}  {where}")
            else:
                print(f"      created {created}  updated {updated}")
                print(f"      {where}")
        print(
            "\nPick a number from the list, or paste a session id directly."
        )
    else:
        print(
            "\nNo checkpointed sessions found in the database "
            f"({db_path}). Paste a session id to attempt a resume anyway."
        )

    while True:
        raw = channel.notes("Session id (or list number)").strip()
        if not raw:
            print("  The session id can't be empty. Try again.")
            continue
        if recent and raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(recent):
                return recent[idx - 1][0]
            print(
                f"  {idx} is out of range (1-{len(recent)}). Try again."
            )
            continue
        return raw


# ---------------------------------------------------------------------------
# Steps 1-5 — runtime prompts
# ---------------------------------------------------------------------------

def _ask_workspace(channel: "HitlChannel") -> str:
    default = os.getcwd()
    while True:
        print("\nStep 1 of 5: Workspace path (the target repo to operate on).")
        print(f"  Default: {default}")
        raw = channel.notes(
            "Enter a path, or press Enter to accept the default"
        ).strip()
        candidate = raw or default
        resolved = os.path.abspath(os.path.expanduser(candidate))
        if not os.path.isdir(resolved):
            print(f"  Not a directory: {resolved}. Try again.")
            continue
        return resolved


def _ask_prompt(channel: "HitlChannel") -> str:
    while True:
        print("\nStep 2 of 5: Engineering task / prompt.")
        print("  Example: \"Refactor the auth module to use JWT.\"")
        raw = channel.notes("Enter the task description").strip()
        if raw:
            return raw
        print("  The prompt can't be empty. Try again.")


def _ask_git(channel: "HitlChannel") -> bool:
    print("\nStep 3 of 5: Enable GitGuardian for the workspace?")
    print("  y = true   (GitGuardian stashes / branches / rolls back; requires a git repo)")
    print("  n = false  (skip every git step — pick this if no git repo)")
    choice = channel.prompt(
        "Choose [y/n]", options=["y", "n"], default="n",
    ).strip().lower()
    return choice == "y"


def _ask_new_build(channel: "HitlChannel") -> bool:
    print("\nStep 4 of 5: Treat this as a brand-new build?")
    print(
        "  Deletes every file at the workspace root EXCEPT product_spec/\n"
        "  and .git/. Defaults to no (preserve existing files)."
    )
    return channel.confirm("New build?", default=False)


def _ask_discover(channel: "HitlChannel") -> bool:
    print("\nStep 5 of 5: Run the full discovery pipeline?")
    print(
        "  Discovery walks through requirements / architecture / deployment\n"
        "  Q&A before code generation. Recommended for greenfield projects;\n"
        "  skip for incremental patching. Defaults to no."
    )
    return channel.confirm("Run discovery?", default=False)


def _print_summary(
    workspace: str, prompt: str, git_mode: bool, new_build: bool, discover: bool,
) -> None:
    print()
    print("-" * 72)
    print("Summary")
    print("-" * 72)
    print(f"  Workspace        : {workspace}")
    short_prompt = prompt if len(prompt) <= 60 else prompt[:57] + "..."
    print(f"  Prompt           : {short_prompt}")
    print(f"  --git            : {'true' if git_mode else 'false'}")
    print(f"  --new-build      : {'true' if new_build else 'false'}")
    print(f"  --spec-discovery : {'true' if discover else 'false'}")
    print("-" * 72)
