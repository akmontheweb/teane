"""
Pluggable persistence layer with AsyncSqliteSaver as the canonical backend.

This module implements:
    - Re-exports the official LangGraph AsyncSqliteSaver from langgraph-checkpoint-sqlite
      so that `isinstance(saver, BaseCheckpointSaver)` passes LangGraph's internal validation.
    - 30-day TTL automatic garbage collection — fired on every teane run/status init.
    - Session ID management: accepts user-provided --session-id, falls back to UUIDv4.
    - `teane status` read-only inspector: queries the SQLite DB and prints a
      clean text snapshot of any checkpointed session without executing graph nodes.
    - `teane purge --all` command integration: wipes all checkpoint data.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


_CLEANSED_CONTENT_PLACEHOLDER = (
    "[ERROR: Redaction Failure — Content Cleansed before checkpoint persistence "
    "to prevent secret leakage. See harness logs for the underlying redactor "
    "error.]"
)


def _force_cleanse_checkpoint_messages(checkpoint: Any) -> Any:
    """Last-resort fail-SAFE: rebuild a checkpoint with cleansed messages.

    Used by aput's outer except: if the redaction wrapper itself raised
    (unusual checkpoint shape, frozen dict, custom AgentState), this
    walks the checkpoint defensively and replaces every messages-channel
    entry's content with the placeholder. If even that walk fails the
    messages channel is removed entirely — losing one checkpoint's
    messages is preferable to persisting raw content.
    """
    try:
        if not isinstance(checkpoint, dict):
            return checkpoint
        cv = checkpoint.get("channel_values")
        if not isinstance(cv, dict) or "messages" not in cv:
            return checkpoint
        new_channels = dict(cv)
        try:
            new_channels["messages"] = _cleansed_messages(cv["messages"])
        except Exception:  # noqa: BLE001
            new_channels.pop("messages", None)
        return {**checkpoint, "channel_values": new_channels}
    except Exception:  # noqa: BLE001 — final fallback drops messages entirely
        try:
            if isinstance(checkpoint, dict):
                cv = checkpoint.get("channel_values")
                if isinstance(cv, dict):
                    new_channels = {k: v for k, v in cv.items() if k != "messages"}
                    return {**checkpoint, "channel_values": new_channels}
        except Exception:  # noqa: BLE001
            pass
        return checkpoint


def _force_cleanse_writes(writes: Any) -> Any:
    """Last-resort fail-SAFE for aput_writes: cleanse `messages` channel writes.

    Used when the redaction loop itself raises. We rebuild the writes tuple
    list defensively and replace any `messages` value with the cleansed form.
    On any structural surprise the offending entry is dropped.
    """
    out: list[Any] = []
    try:
        for entry in writes:
            try:
                channel, value = entry
            except Exception:  # noqa: BLE001
                continue
            if channel == "messages":
                try:
                    out.append((channel, _cleansed_messages(value)))
                except Exception:  # noqa: BLE001
                    # Drop this write entirely rather than persist raw content.
                    continue
            else:
                out.append((channel, value))
    except Exception:  # noqa: BLE001
        # Catastrophic — return empty writes list rather than persist raw content.
        return []
    return out


def _cleansed_messages(messages: Any) -> list[Any]:
    """Return a copy of ``messages`` with every entry's ``content`` field
    replaced by a fixed placeholder.

    Used by the storage layer's fail-safe path when the redactor crashes:
    the checkpoint still gets written (LangGraph needs it to resume) but
    no raw message content reaches disk. Preserves message role and any
    structural metadata so resume logic can still walk the message list.
    """
    if not isinstance(messages, list):
        return messages
    cleansed: list[Any] = []
    for msg in messages:
        if isinstance(msg, dict):
            scrubbed = dict(msg)
            if "content" in scrubbed:
                scrubbed["content"] = _CLEANSED_CONTENT_PLACEHOLDER
            cleansed.append(scrubbed)
        else:
            cleansed.append(_CLEANSED_CONTENT_PLACEHOLDER)
    return cleansed


# ---------------------------------------------------------------------------
# 1. Types
# ---------------------------------------------------------------------------

@dataclass
class CheckpointSummary:
    """
    Read-only snapshot of a checkpointed session, as returned by `teane status`.
    """
    thread_id: str
    session_id: str = ""
    current_node: str = ""
    exit_code: int = -1
    budget_remaining_usd: float = 0.0
    total_cost_usd: float = 0.0
    modified_files: list[str] = field(default_factory=list)
    loop_counters: dict[str, int] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    is_active: bool = False
    workspace_path: str = ""


# ---------------------------------------------------------------------------
# 2. Re-export the official LangGraph AsyncSqliteSaver
# ---------------------------------------------------------------------------

# The official langgraph-checkpoint-sqlite AsyncSqliteSaver is a fully-compliant
# BaseCheckpointSaver subclass. We re-export it so that graph.compile(checkpointer=...)
# passes ensure_valid_checkpointer() with zero friction.
#
# Our "HarnessAsyncSqliteSaver" thin wrapper adds:
#   - The `from_db_path` classmethod (SQLite path-based constructor)
#   - TTL-based automatic garbage collection on initialisation
#   - The same interface the CLI expects

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver as _OfficialAsyncSqliteSaver  # noqa: E402


# ---------------------------------------------------------------------------
# Checkpoint schema versioning  (P2.4)
# ---------------------------------------------------------------------------
#
# Every checkpoint blob carries a schema version stamped into its metadata.
# Bump CHECKPOINT_SCHEMA_VERSION when AgentState gains a required field or
# the on-disk channel layout changes in a way that older checkpoints can't
# be safely resumed against. cmd_resume refuses to load a checkpoint whose
# version is higher than the running harness (older harness, newer file)
# or whose version is older than MIN_RESUMABLE_SCHEMA_VERSION (incompatible
# format). Missing version = legacy checkpoint (pre-versioning); we WARN
# and allow resume so the rollout doesn't strand existing users.
#
# The version is stored under ``_harness_schema_version`` inside the
# checkpoint *metadata* dict (a separate SQLite column from the channel
# blob) so injecting it can't disturb LangGraph's own state-restore logic.
CHECKPOINT_SCHEMA_VERSION = 1
MIN_RESUMABLE_SCHEMA_VERSION = 1
SCHEMA_VERSION_METADATA_KEY = "_harness_schema_version"


class CheckpointSchemaMismatchError(RuntimeError):
    """Raised when a checkpoint's stamped schema version is incompatible.

    Distinct from CheckpointCorruptedError (blob decode failure): the blob
    decoded fine, but its version field signals it can't be safely resumed
    by the running harness build.
    """


async def _configure_sqlite_pragmas(conn: Any, db_path: str) -> None:
    """
    Set crash-safety pragmas on a freshly-opened aiosqlite connection and
    verify they took effect.

    PRAGMA journal_mode=WAL fails silently on some configurations (network
    filesystems, read-only mounts) — we read the result back and downgrade
    synchronous mode if WAL didn't apply, to keep durability honest.
    """
    # WAL: enables concurrent readers + crash-recoverable writes. Returns
    # the *actual* journal mode in use, which may differ from requested.
    cur = await conn.execute("PRAGMA journal_mode=WAL;")
    row = await cur.fetchone()
    actual_mode = (row[0] if row else "").lower() if row else ""
    if actual_mode != "wal":
        logger.warning(
            "[storage] PRAGMA journal_mode=WAL did not take effect at %s "
            "(got %r). Falling back to synchronous=FULL for durability.",
            db_path, actual_mode,
        )
        # If WAL isn't available, we can't rely on NORMAL's looser fsync.
        await conn.execute("PRAGMA synchronous=FULL;")
    else:
        await conn.execute("PRAGMA synchronous=NORMAL;")

    # Allow short waits for writer locks before raising SQLITE_BUSY.
    await conn.execute("PRAGMA busy_timeout=5000;")
    # Foreign keys are off by default but cheap to enforce.
    await conn.execute("PRAGMA foreign_keys=ON;")
    await conn.commit()


class HarnessAsyncSqliteSaver(_OfficialAsyncSqliteSaver):
    """
    Thin wrapper around the official langgraph-checkpoint-sqlite AsyncSqliteSaver
    that adds TTL garbage collection, a path-based constructor, and
    secret-redaction on the checkpoint write path.

    Usage:
        async with HarnessAsyncSqliteSaver.from_db_path("~/.harness/checkpoints.db") as saver:
            compiled = graph.compile(checkpointer=saver)
    """

    _db_path: str
    _ttl_days: int
    # When True, every aput / aput_writes redacts the `messages` channel
    # through harness.redactor before letting LangGraph serialize it. This
    # keeps secrets the user pasted into a prompt out of the on-disk SQLite
    # blob. Defaults to True; opt out via persistence.redact_messages: false.
    _redact_messages_on_checkpoint: bool

    @classmethod
    async def from_db_path(
        cls,
        db_path: str = "~/.harness/checkpoints.db",
        ttl_days: int = 30,
        redact_messages: bool = True,
    ) -> "HarnessAsyncSqliteSaver":
        """
        Create a HarnessAsyncSqliteSaver from a filesystem path.

        Manages connection lifecycle internally. Runs schema initialization
        and TTL garbage collection before returning.
        """
        import aiosqlite

        expanded_path = os.path.expanduser(db_path)
        db_dir = os.path.dirname(expanded_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        conn = await aiosqlite.connect(expanded_path)
        await _configure_sqlite_pragmas(conn, expanded_path)

        instance = cls(conn)
        await instance.setup()

        # Attach metadata for GC & inspection
        instance._db_path = expanded_path
        instance._ttl_days = ttl_days
        instance._redact_messages_on_checkpoint = redact_messages

        # Run 30-day TTL garbage collection
        await instance._run_gc()

        logger.info(
            "[storage] HarnessAsyncSqliteSaver initialised at %s "
            "(TTL=%d days, redact_messages=%s).",
            expanded_path,
            ttl_days,
            redact_messages,
        )
        return instance

    # ------------------------------------------------------------------
    # Redaction hook on the write path.
    #
    # LangGraph routes every state transition through aput (full snapshot)
    # and aput_writes (per-channel pending writes). Both serialise via
    # self.serde — we intercept *before* serde sees the data so the redacted
    # form is what lands in SQLite.
    # ------------------------------------------------------------------

    def _redact_messages_list(self, messages: Any) -> Any:
        """Redact a messages-channel value in place. Returns the (possibly
        modified) value.

        Fail-SAFE policy: a redactor crash MUST NOT cause raw, unredacted
        message content to land on disk. Earlier versions of this method
        fell back to ``return messages`` on exception, which silently
        persisted plaintext secrets to the SQLite checkpoint. Instead, on
        any redactor failure we replace every message's ``content`` with a
        cleansed placeholder so the checkpoint still gets written (so
        LangGraph can resume) but secrets never reach disk.
        """
        if not getattr(self, "_redact_messages_on_checkpoint", True):
            return messages
        if not isinstance(messages, list):
            return messages
        try:
            from harness.redactor import redact_messages
        except Exception:  # noqa: BLE001
            # Redactor module itself failed to import — cleanse rather than
            # persisting raw content. Same fail-safe behavior as a runtime
            # crash inside redact_messages().
            logger.warning(
                "[storage] redactor module unavailable; cleansing message "
                "content before persistence to avoid secret leakage."
            )
            return _cleansed_messages(messages)
        try:
            # redact_messages tolerates non-dict items and is a no-op if the
            # global scanner hasn't been configured.
            return redact_messages(messages)
        except Exception as exc:  # noqa: BLE001 — fail SAFE on redactor crashes
            logger.warning(
                "[storage] message redaction failed (%s); cleansing content "
                "before persistence to avoid secret leakage.", exc,
            )
            return _cleansed_messages(messages)

    async def aput(self, config, checkpoint, metadata, new_versions):
        # Redact the `messages` channel inside the full state snapshot before
        # delegating to the LangGraph serializer. We mutate a shallow copy so
        # the in-memory state the running graph holds is untouched.
        #
        # Fail-SAFE: any exception in the redaction path MUST cleanse the
        # messages channel before persistence. The earlier "log + fall through"
        # approach silently persisted raw operator content (potentially
        # carrying API keys / PII) when an unusual checkpoint shape tripped
        # the redactor. See EDGE_CASE_AUDIT.md §1.3.
        try:
            channel_values = checkpoint.get("channel_values") if isinstance(checkpoint, dict) else None
            if isinstance(channel_values, dict) and "messages" in channel_values:
                redacted = self._redact_messages_list(channel_values["messages"])
                if redacted is not channel_values["messages"]:
                    # Avoid mutating the live state object.
                    new_channels = dict(channel_values)
                    new_channels["messages"] = redacted
                    checkpoint = {**checkpoint, "channel_values": new_channels}
        except Exception as exc:  # noqa: BLE001 — fail SAFE, never leak raw content
            logger.warning(
                "[storage] aput redaction wrapper failed (%s); cleansing all "
                "messages before persistence to avoid secret leakage.", exc,
            )
            checkpoint = _force_cleanse_checkpoint_messages(checkpoint)
        # Stamp the checkpoint schema version into the metadata blob (P2.4).
        # Use a shallow copy so we don't mutate LangGraph's internal dict.
        if isinstance(metadata, dict):
            metadata = {**metadata, SCHEMA_VERSION_METADATA_KEY: CHECKPOINT_SCHEMA_VERSION}
        return await super().aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self, config: Any, writes: Any, task_id: Any, task_path: str = "",
    ) -> Any:
        # aput_writes records the pending channel writes from a node return.
        # For the `messages` channel that value is the new messages list.
        #
        # Fail-SAFE: see aput() above. On any redactor exception we cleanse
        # every `messages` write rather than letting raw content through.
        try:
            redacted_writes = []
            mutated = False
            for channel, value in writes:
                if channel == "messages":
                    new_value = self._redact_messages_list(value)
                    if new_value is not value:
                        mutated = True
                    redacted_writes.append((channel, new_value))
                else:
                    redacted_writes.append((channel, value))
            if mutated:
                writes = redacted_writes
        except Exception as exc:  # noqa: BLE001 — fail SAFE, never leak raw content
            logger.warning(
                "[storage] aput_writes redaction wrapper failed (%s); "
                "cleansing all messages writes before persistence to avoid "
                "secret leakage.", exc,
            )
            writes = _force_cleanse_writes(writes)
        return await super().aput_writes(config, writes, task_id, task_path)

    async def _run_gc(self) -> int:
        """
        Delete checkpoint rows for threads whose latest checkpoint is older
        than ``self._ttl_days``.

        LangGraph stores an ISO 8601 ``ts`` field inside the msgpack-encoded
        checkpoint blob; we deserialize the latest row per thread, compare
        its timestamp to ``now - ttl_days``, and bulk-delete expired threads
        from both ``checkpoints`` and ``writes`` in a single transaction.

        Returns the total number of rows deleted across both tables.
        Setting ``ttl_days <= 0`` disables GC.
        """
        ttl_days = getattr(self, "_ttl_days", 0)
        if ttl_days is None or ttl_days <= 0:
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)

        try:
            cursor = await self.conn.execute(
                """SELECT c.thread_id, c.checkpoint
                   FROM checkpoints c
                   INNER JOIN (
                       SELECT thread_id, MAX(checkpoint_id) AS max_cp_id
                       FROM checkpoints
                       GROUP BY thread_id
                   ) AS latest ON c.thread_id = latest.thread_id
                              AND c.checkpoint_id = latest.max_cp_id"""
            )
            rows = await cursor.fetchall()
        except Exception as e:  # noqa: BLE001
            logger.warning("[storage] GC scan failed (%s); skipping.", e)
            return 0

        expired_threads: list[str] = []
        for thread_id, blob in rows:
            cp = _deserialize_checkpoint_blob(blob)
            ts_value = cp.get("ts", "") if isinstance(cp, dict) else ""
            if not ts_value or not isinstance(ts_value, str):
                # No usable ts: either truly corrupted (cannot be resumed)
                # or the blob decoded to a dict that lacks a `ts` field.
                # Probe with strict mode — if the decoder genuinely cannot
                # read the blob, the row is unrecoverable and we delete it
                # rather than let it accumulate forever (audit §5.4).
                # A blob that decodes cleanly but lacks `ts` is left alone
                # (might be a transient LangGraph format change).
                try:
                    _deserialize_checkpoint_blob(blob, strict=True)
                    continue
                except CheckpointCorruptedError:
                    logger.warning(
                        "[storage] GC: marking thread %s for deletion "
                        "(checkpoint blob is undecodable).",
                        thread_id,
                    )
                    expired_threads.append(thread_id)
                    continue
            try:
                dt = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            if dt < cutoff:
                expired_threads.append(thread_id)

        if not expired_threads:
            logger.debug("[storage] GC: no expired threads (TTL=%d days).", ttl_days)
            return 0

        placeholders = ",".join("?" * len(expired_threads))
        try:
            cursor = await self.conn.execute(
                f"DELETE FROM writes WHERE thread_id IN ({placeholders})",
                expired_threads,
            )
            deleted = cursor.rowcount or 0
            cursor = await self.conn.execute(
                f"DELETE FROM checkpoints WHERE thread_id IN ({placeholders})",
                expired_threads,
            )
            deleted += cursor.rowcount or 0
            await self.conn.commit()
        except Exception as e:  # noqa: BLE001
            logger.warning("[storage] GC delete failed (%s); rolling back.", e)
            try:
                await self.conn.rollback()
            except Exception:  # noqa: BLE001
                pass
            return 0

        logger.info(
            "[storage] TTL GC: removed %d rows across %d expired threads (TTL=%d days).",
            deleted,
            len(expired_threads),
            ttl_days,
        )
        return deleted

    @property
    def db_path(self) -> str:
        """Return the filesystem path of the backing SQLite database."""
        return getattr(self, "_db_path", "")

    @classmethod
    async def from_conn_string_with_gc(
        cls,
        conn_string: str,
        ttl_days: int = 30,
    ) -> "HarnessAsyncSqliteSaver":
        """
        Create from a SQLite connection string, then run GC.
        Use when you need the official constructor semantics + GC.
        """
        import aiosqlite

        expanded_path = os.path.expanduser(conn_string)
        db_dir = os.path.dirname(expanded_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        conn = await aiosqlite.connect(expanded_path)
        await _configure_sqlite_pragmas(conn, expanded_path)

        instance = cls(conn)
        await instance.setup()
        instance._db_path = expanded_path
        instance._ttl_days = ttl_days
        await instance._run_gc()
        return instance


# Backwards-compatible alias — code that imports AsyncSqliteSaver from storage
# will get the Harness wrapper which IS a valid BaseCheckpointSaver.
AsyncSqliteSaver = HarnessAsyncSqliteSaver

# ---------------------------------------------------------------------------
# 2b. Direct BaseCheckpointSaver alias (for isinstance checks)
# ---------------------------------------------------------------------------
from langgraph.checkpoint.base import BaseCheckpointSaver  # noqa: E402

BaseCheckpointer = BaseCheckpointSaver  # alias for backwards-compat


# ---------------------------------------------------------------------------
# 3. Session ID Management
# ---------------------------------------------------------------------------

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def generate_session_id(user_provided: Optional[str] = None) -> str:
    """
    Generate a session ID. Returns the user-provided value if given,
    otherwise falls back to a random UUIDv4.

    Args:
        user_provided: Optional user-supplied session ID string.

    Returns:
        A session ID string.

    Raises ValueError when ``user_provided`` is non-empty but contains
    characters that would corrupt downstream filenames (log JSONL,
    CR archive directory, git branch name) or break shell echo (NUL,
    control chars, RTL override, path separators). Audit §5.2.
    """
    if user_provided and user_provided.strip():
        session_id = user_provided.strip()
        if not _SESSION_ID_RE.fullmatch(session_id):
            raise ValueError(
                f"--session-id {session_id!r} must match [A-Za-z0-9._-]{{1,64}}. "
                f"The value flows into log filenames, archive directory names, "
                f"and git branch names; unsanitised values can break those paths."
            )
        logger.info("[storage] Using user-provided session ID: %s", session_id)
        return session_id

    session_id = str(uuid.uuid4())
    logger.info("[storage] Auto-generated session ID (UUIDv4): %s", session_id)
    return session_id


# ---------------------------------------------------------------------------
# 4. Status Inspector — Read-Only Session Snapshot
# ---------------------------------------------------------------------------

class CheckpointCorruptedError(RuntimeError):
    """Raised when a checkpoint blob cannot be deserialized by any decoder.

    Callers that are about to ACT on a checkpoint (notably ``cmd_resume``
    restoring graph state) should let this propagate so the operator sees
    a clear "checkpoint corrupted" message instead of a silent fresh-start.
    Callers that are merely SCANNING many rows (TTL GC, status listings)
    should swallow it so one bad row doesn't halt the batch.
    """


def _deserialize_checkpoint_blob(blob: Any, *, strict: bool = False) -> dict[str, Any]:
    """
    Deserialize a checkpoint column BLOB from the SQLite store.

    LangGraph's AsyncSqliteSaver stores checkpoints as msgpack-encoded
    binary blobs (via JsonPlusSerializer). Falls back to JSON for
    backwards compatibility with any legacy text-based rows.

    Args:
        blob: the raw column value from SQLite.
        strict: when True, raise CheckpointCorruptedError if every decoder
            fails (used by cmd_resume's pre-flight validation). When False
            (default), return ``{}`` so batch scanners (TTL GC, status
            listings) don't halt on a single bad row.

    Returns ``{}`` on failure when ``strict=False``.
    """
    if blob is None:
        return {}

    if isinstance(blob, (bytes, bytearray)):
        # LangGraph uses msgpack for the `checkpoint` column and plain JSON
        # for the `metadata` column. Both pass through this helper, so we
        # sniff the first non-whitespace byte and pick the right decoder
        # first. The previous "msgpack-then-JSON" order spammed a noisy
        # WARNING on every metadata blob (msgpack consumed the leading `{`
        # as a fixmap header then raised "received extra data" on the
        # remaining JSON bytes), confusing operators into thinking their
        # checkpoint was corrupted when in fact the resume was succeeding
        # via the silent JSON fallback.
        try:
            import msgpack  # type: ignore[import-untyped]
        except ImportError:
            msgpack = None

        # First non-whitespace byte → format hint. JSON objects/arrays/strings
        # always start with `{`, `[`, or `"` (0x7B / 0x5B / 0x22). Anything
        # else (msgpack tag bytes 0x80-0xDF, etc.) is treated as msgpack.
        sniff_idx = 0
        while sniff_idx < len(blob) and blob[sniff_idx] in (0x20, 0x09, 0x0A, 0x0D):
            sniff_idx += 1
        first_byte = blob[sniff_idx] if sniff_idx < len(blob) else 0
        looks_like_json = first_byte in (0x7B, 0x5B, 0x22)  # { [ "

        def _try_msgpack() -> tuple[bool, Any, Optional[str]]:
            if msgpack is None:
                return False, None, "msgpack module unavailable"
            try:
                return True, msgpack.unpackb(blob, raw=False), None
            except Exception as e:  # noqa: BLE001
                return False, None, str(e)

        def _try_json() -> tuple[bool, Any, Optional[str]]:
            # Audit §5.16: strict decode so a truncated UTF-8 sequence
            # in a message field surfaces as a corruption signal rather
            # than getting silently mangled with U+FFFD replacements.
            try:
                return True, json.loads(blob.decode("utf-8", errors="strict")), None
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                return False, None, str(e)

        attempts = (_try_json, _try_msgpack) if looks_like_json else (_try_msgpack, _try_json)
        first_err: Optional[str] = None
        for attempt in attempts:
            ok, value, err = attempt()
            if ok:
                return value
            first_err = first_err or err

        # Both decoders failed → real corruption. Now the WARNING is meaningful.
        logger.warning(
            "[storage] Checkpoint blob decode failed (msgpack and JSON both "
            "rejected it; first byte=0x%02x len=%d).",
            first_byte, len(blob),
        )
        if strict:
            raise CheckpointCorruptedError(
                f"Checkpoint blob could not be decoded as msgpack or JSON: {first_err}"
            )
        return {}

    # Plain text JSON path (legacy / backwards-compat)
    if isinstance(blob, str):
        try:
            return json.loads(blob)
        except json.JSONDecodeError as exc:
            logger.warning("[storage] Plain-text JSON decode failed (%s).", exc)
            if strict:
                raise CheckpointCorruptedError(
                    "Plain-text checkpoint blob is not valid JSON."
                ) from exc
            return {}

    if strict:
        raise CheckpointCorruptedError(
            f"Unsupported checkpoint blob type: {type(blob).__name__}"
        )
    return {}


def validate_checkpoint_schema(metadata_blob: Any) -> Optional[int]:
    """Validate a checkpoint's stamped schema version (P2.4).

    Returns the resolved version (None when missing). Raises
    CheckpointSchemaMismatchError when the stamped version is incompatible
    with the running harness.

    Policy:
      - Missing version: legacy checkpoint, predates P2.4. WARNs and returns
        None so cmd_resume still proceeds — failing here would strand every
        operator who upgrades.
      - Version > CHECKPOINT_SCHEMA_VERSION: written by a newer harness.
        Refuse: silently loading an unknown-future state risks data loss.
      - Version < MIN_RESUMABLE_SCHEMA_VERSION: incompatible older format.
        Refuse with the same explicit error.
    """
    # Audit §5.15: use strict decode so a corrupted metadata blob is
    # treated as a schema mismatch instead of silently allowing resume
    # against unknown state. A blob that decodes cleanly but lacks the
    # version stamp falls into the legacy path below.
    try:
        metadata = _deserialize_checkpoint_blob(metadata_blob, strict=True)
    except CheckpointCorruptedError as exc:
        raise CheckpointSchemaMismatchError(
            f"Checkpoint metadata blob is undecodable: {exc}. Refusing to "
            f"resume — the on-disk state cannot be trusted."
        ) from exc
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get(SCHEMA_VERSION_METADATA_KEY)
    if raw is None:
        logger.warning(
            "[storage] Checkpoint has no schema version stamp — treating as "
            "pre-P2.4 legacy and allowing resume. Re-running this session "
            "will write a versioned checkpoint going forward."
        )
        return None
    try:
        version = int(raw)
    except (TypeError, ValueError) as exc:
        raise CheckpointSchemaMismatchError(
            f"Checkpoint schema_version is not an integer: {raw!r}"
        ) from exc
    if version > CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointSchemaMismatchError(
            f"Checkpoint was written by a newer harness "
            f"(checkpoint schema v{version}, this harness supports v{CHECKPOINT_SCHEMA_VERSION}). "
            f"Upgrade the harness or start a fresh session."
        )
    if version < MIN_RESUMABLE_SCHEMA_VERSION:
        raise CheckpointSchemaMismatchError(
            f"Checkpoint schema v{version} is older than the minimum supported "
            f"version (v{MIN_RESUMABLE_SCHEMA_VERSION}). Start a fresh session."
        )
    return version


def _format_checkpoint_ts(ts_value: Any) -> str:
    """
    Convert a LangGraph checkpoint 'ts' value into a human-readable local
    datetime string.

    The ``ts`` field is an ISO 8601 UTC string (e.g. "2026-06-08T14:30:00.000000Z").
    Returns a string like "2026-06-08 10:30:00" in the local timezone.
    Falls back to "(unknown)" if the value cannot be parsed.
    """
    if not ts_value or not isinstance(ts_value, str):
        return "(unknown)"

    try:
        from datetime import datetime
        # Strip trailing 'Z' and parse ISO 8601 UTC
        cleaned = ts_value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        # If the parsed datetime is timezone-aware, convert to local
        if dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None:
            dt = dt.astimezone()
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return "(unknown)"


async def inspect_session(
    db_path: str,
    thread_id: str,
) -> Optional[CheckpointSummary]:
    """
    Read a checkpoint from the SQLite database and return a human-readable
    summary without triggering any graph execution.

    Used by `teane status --session-id <uuid>`.

    Args:
        db_path: Path to the checkpoints SQLite database.
        thread_id: The thread/session ID to inspect.

    Returns:
        CheckpointSummary if found, None otherwise.
    """
    expanded_path = os.path.expanduser(db_path)
    if not os.path.isfile(expanded_path):
        logger.warning("[storage] Database not found at %s.", expanded_path)
        return None

    import aiosqlite

    async with aiosqlite.connect(expanded_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT thread_id, checkpoint, metadata
               FROM checkpoints
               WHERE thread_id = ?
               ORDER BY checkpoint_id DESC
               LIMIT 1""",
            (thread_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            logger.warning("[storage] No checkpoint found for thread '%s'.", thread_id)
            return None

        checkpoint = _deserialize_checkpoint_blob(row["checkpoint"])

        # Extract state fields from the checkpoint blob
        channel_values = checkpoint.get("channel_values", {})
        state = channel_values if isinstance(channel_values, dict) else {}

        exit_code = state.get("exit_code", -1)
        if isinstance(exit_code, dict):
            exit_code = exit_code.get("value", -1)

        budget_remaining = state.get("budget_remaining_usd", 0.0)
        if isinstance(budget_remaining, dict):
            budget_remaining = budget_remaining.get("value", 0.0)

        token_tracker = state.get("token_tracker", {})
        if isinstance(token_tracker, dict) and "total_cost_usd" in token_tracker:
            total_cost = token_tracker["total_cost_usd"]
        elif isinstance(token_tracker, dict) and "value" in token_tracker:
            total_cost = token_tracker.get("value", {}).get("total_cost_usd", 0.0)
        else:
            total_cost = 0.0

        modified_files = state.get("modified_files", [])
        if isinstance(modified_files, dict):
            modified_files = modified_files.get("value", [])

        loop_counters = state.get("loop_counter", {})
        if isinstance(loop_counters, dict) and not any(isinstance(v, dict) for v in loop_counters.values()):
            pass
        elif isinstance(loop_counters, dict):
            loop_counters = loop_counters.get("value", {})

        node_state = state.get("node_state", {})
        current_node = ""
        if isinstance(node_state, dict):
            current_node = node_state.get("current_node", "")
        elif isinstance(node_state, str):
            current_node = node_state

        # Extract timestamps from the LangGraph checkpoint "ts" field (ISO 8601)
        ts_value = checkpoint.get("ts", "")
        created_fmt = _format_checkpoint_ts(ts_value)
        # The latest checkpoint's ts is both created and updated time
        updated_fmt = created_fmt

        # Extract workspace_path from channel_values
        workspace_path = state.get("workspace_path", "")
        if isinstance(workspace_path, dict):
            workspace_path = workspace_path.get("value", "")
        workspace_path = str(workspace_path) if workspace_path else ""

        return CheckpointSummary(
            thread_id=row["thread_id"],
            session_id=thread_id,
            current_node=current_node,
            exit_code=int(exit_code) if exit_code is not None else -1,
            budget_remaining_usd=float(budget_remaining) if budget_remaining is not None else 0.0,
            total_cost_usd=float(total_cost) if total_cost is not None else 0.0,
            modified_files=list(modified_files) if modified_files else [],
            loop_counters=dict(loop_counters) if loop_counters else {},
            created_at=created_fmt,
            updated_at=updated_fmt,
            # is_active = "session is still in flight (no terminal exit)".
            # exit_code is -1 BOTH for never-built (compiler_node hasn't run yet)
            # and for sessions that ended without recording a real code. The
            # earlier formula `not in (0, -1) and != 0` excluded both meanings
            # of -1, so a never-built session was reported INACTIVE — which
            # under-counts running sessions in `teane status` and in the
            # dashboard. Now: active when there's no recorded exit code OR
            # the recorded value is -1 AND we've never built (current_node
            # isn't a terminal node). Audit §5.3 (BREAKING).
            is_active=(exit_code is None) or (exit_code == -1 and current_node not in ("END", "human_intervention_node")),
            workspace_path=workspace_path,
        )


async def list_all_sessions(db_path: str, limit: int = 50) -> list[CheckpointSummary]:
    """
    List summaries of all checkpointed sessions, ordered by most recently updated.

    Reads the latest checkpoint JSON blob for each thread to extract
    created/updated timestamps and workspace path.

    Args:
        db_path: Path to the checkpoints SQLite database.
        limit: Maximum number of sessions to return.

    Returns:
        List of CheckpointSummary objects.
    """
    expanded_path = os.path.expanduser(db_path)
    if not os.path.isfile(expanded_path):
        return []

    import aiosqlite

    summaries: list[CheckpointSummary] = []
    async with aiosqlite.connect(expanded_path) as db:
        db.row_factory = aiosqlite.Row
        # Subquery: for each thread_id, get the row with the largest checkpoint_id
        cursor = await db.execute(
            """SELECT c.thread_id, c.checkpoint_id, c.checkpoint
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
        rows = await cursor.fetchall()
        for row in rows:
            created_at = "(unknown)"
            updated_at = "(unknown)"
            workspace_path = ""

            try:
                cp = _deserialize_checkpoint_blob(row["checkpoint"])
                # Extract timestamp from the LangGraph "ts" field
                ts_value = cp.get("ts", "")
                created_at = _format_checkpoint_ts(ts_value)
                updated_at = created_at  # same for latest checkpoint

                # Extract workspace_path from channel_values
                channel_values = cp.get("channel_values", {})
                if isinstance(channel_values, dict):
                    wp = channel_values.get("workspace_path", "")
                    if isinstance(wp, dict):
                        wp = wp.get("value", "")
                    workspace_path = str(wp) if wp else ""
            except Exception:
                pass  # use fallback values

            summaries.append(CheckpointSummary(
                thread_id=row["thread_id"],
                session_id=row["thread_id"],
                created_at=created_at,
                updated_at=updated_at,
                workspace_path=workspace_path,
            ))
    return summaries


# ---------------------------------------------------------------------------
# 5. Checkpointer Factory
# ---------------------------------------------------------------------------

async def create_checkpointer(
    backend: str = "sqlite",
    db_path: str = "~/.harness/checkpoints.db",
    ttl_days: int = 30,
) -> BaseCheckpointSaver[Any]:
    """
    Factory: create the appropriate checkpointer backend.

    Args:
        backend: One of 'sqlite', 'memory', 'redis', 'postgres'.
                 Currently only 'sqlite' and 'memory' are implemented.
        db_path: Path to the SQLite database (for 'sqlite' backend).
        ttl_days: TTL for automatic garbage collection.

    Returns:
        A BaseCheckpointSaver instance (LangGraph-compliant).

    Raises:
        ValueError: If the backend is not recognized.
    """
    if backend == "sqlite":
        return await HarnessAsyncSqliteSaver.from_db_path(db_path=db_path, ttl_days=ttl_days)
    elif backend == "memory":
        try:
            from langgraph.checkpoint.memory import MemorySaver
            logger.info("[storage] Using in-memory MemorySaver (ephemeral).")
            return MemorySaver()
        except ImportError:
            logger.warning("[storage] MemorySaver not available. Falling back to AsyncSqliteSaver (:memory:).")
            return await HarnessAsyncSqliteSaver.from_db_path(db_path=":memory:", ttl_days=ttl_days)
    elif backend in ("redis", "postgres"):
        raise NotImplementedError(
            f"Backend '{backend}' is not yet implemented. "
            f"Use 'sqlite' for local development or 'memory' for ephemeral runs."
        )
    else:
        raise ValueError(
            f"Unknown backend: '{backend}'. Supported: 'sqlite', 'memory'."
        )


async def purge_checkpoints(db_path: str) -> int:
    """
    Delete ALL checkpoint data from the database. Returns row count deleted.

    Args:
        db_path: Path to the checkpoints SQLite database.

    Returns:
        Total number of rows deleted.
    """
    expanded_path = os.path.expanduser(db_path)
    if not os.path.isfile(expanded_path):
        logger.warning("[storage] No database at %s — nothing to purge.", expanded_path)
        return 0

    import aiosqlite

    async with aiosqlite.connect(expanded_path) as db:
        # Audit §5.17: explicit BEGIN IMMEDIATE + rollback-on-error so
        # a mid-loop OSError between the two DELETEs can't leave the
        # DB with writes wiped but orphan checkpoints retained.
        deleted = 0
        try:
            await db.execute("BEGIN IMMEDIATE")
            cursor = await db.execute("DELETE FROM writes")
            deleted = cursor.rowcount or 0
            cursor = await db.execute("DELETE FROM checkpoints")
            deleted += cursor.rowcount or 0
            await db.commit()
        except Exception:
            try:
                await db.rollback()
            except Exception:  # noqa: BLE001
                pass
            raise

    logger.info("[storage] Purged all data: %d rows deleted.", deleted)
    return deleted