"""
Pluggable Human-in-the-Loop (HITL) transport.

All interactive prompts in the harness — gatekeeper approvals, repair hints,
deploy previews, purge confirmations, discovery Q&A — are routed through a
HitlChannel so the I/O surface can be swapped without touching call sites.

Built-in implementations:
  StdinChannel  — current stdin/print behaviour (default when neither
                  HARNESS_HITL_WEBHOOK_URL nor HARNESS_HITL_FILE is set).
  FileChannel   — pre-recorded answers loaded from a JSON file specified by
                  the HARNESS_HITL_FILE environment variable. Used for scripted
                  integration tests and CI runs without a TTY.
  HttpChannel   — POSTs each prompt to a webhook URL (HARNESS_HITL_WEBHOOK_URL)
                  and reads the JSON reply. Enables IDE plugins, agent-servers,
                  and any integration that wants to drive the harness over HTTP.

Environment variable priority in get_channel():
  HARNESS_HITL_WEBHOOK_URL → HttpChannel
  HARNESS_HITL_FILE        → FileChannel
  (default)                → StdinChannel

Usage in call sites::

    from harness.hitl import get_channel

    choice = get_channel().prompt("Select action", ["a", "e", "m", "s"])
    ok = get_channel().confirm("Proceed?")
    hint = get_channel().notes("Enter feedback")
    get_channel().wait_for_manual_edit("/path/to/file.md")
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. HitlChannel ABC
# ---------------------------------------------------------------------------

class HitlChannel(ABC):
    """Abstract base for all HITL I/O transports."""

    @abstractmethod
    def prompt(
        self,
        message: str,
        options: list[str],
        default: Optional[str] = None,
        option_labels: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Present a menu prompt and return the user's selection.

        Args:
            message: The prompt text (shown once before option list).
            options: Valid single-character (or short) answer strings.
            default: Answer returned automatically in non-interactive mode.
                     If None and the channel is non-interactive, raises.
            option_labels: Optional human-readable description for each
                     option key. Stdin/File channels ignore this; the
                     HttpChannel forwards it in the webhook body so a
                     UI on the other end can render a labeled dropdown
                     instead of a free-text input.

        Returns:
            The selected option string (lowercased).
        """

    @abstractmethod
    def confirm(self, message: str, default: bool = False) -> bool:
        """Present a y/N confirmation. Returns True if confirmed."""

    @abstractmethod
    def notes(self, message: str) -> str:
        """
        Prompt for multi-word free-text input (e.g., a repair hint).
        Returns the text entered by the user (may be empty string).
        """

    @abstractmethod
    def wait_for_manual_edit(self, filepath: str) -> None:
        """
        Block until the user signals they have finished editing ``filepath``.
        In non-interactive channels, returns immediately.
        """

    def is_interactive(self) -> bool:
        """Return True when the channel is connected to a live human."""
        return False


# ---------------------------------------------------------------------------
# 2. StdinChannel
# ---------------------------------------------------------------------------

def _auto_approve() -> bool:
    """True when the environment requests non-interactive execution."""
    return (
        os.environ.get("CI", "").lower() == "true"
        or os.environ.get("HARNESS_AUTO_APPROVE", "").lower() == "true"
        or not sys.stdin.isatty()
    )


class StdinChannel(HitlChannel):
    """
    Interactive stdin/stdout channel — the default.

    Respects HARNESS_AUTO_APPROVE=true, CI=true, and non-TTY stdin by
    returning the ``default`` value without blocking. If no default is
    provided in auto-approve mode, the first option is used.
    """

    def is_interactive(self) -> bool:
        return not _auto_approve()

    def prompt(
        self,
        message: str,
        options: list[str],
        default: Optional[str] = None,
        option_labels: Optional[dict[str, str]] = None,
    ) -> str:
        del option_labels  # stdin already prints its own menu
        opts_str = "/".join(options)
        if _auto_approve():
            chosen = default if default is not None else (options[0] if options else "")
            logger.info("[hitl] Auto-approved prompt %r → %r", message[:60], chosen)
            return chosen

        while True:
            try:
                answer = input(f"{message} [{opts_str}]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n[HITL] Input interrupted.", file=sys.stderr)
                return default if default is not None else (options[0] if options else "")
            if not options or answer in [o.lower() for o in options]:
                return answer

    def confirm(self, message: str, default: bool = False) -> bool:
        if _auto_approve():
            logger.info("[hitl] Auto-confirmed: %r → %s", message[:60], default)
            return default

        hint = "[Y/n]" if default else "[y/N]"
        try:
            answer = input(f"{message} {hint}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[HITL] Input interrupted.", file=sys.stderr)
            return default
        if not answer:
            return default
        return answer in ("y", "yes")

    def notes(self, message: str) -> str:
        if _auto_approve():
            return ""
        try:
            return input(f"{message}\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[HITL] Input interrupted.", file=sys.stderr)
            return ""

    def wait_for_manual_edit(self, filepath: str) -> None:
        if _auto_approve():
            logger.info("[hitl] Auto-skipping wait_for_manual_edit: %s", filepath)
            return
        try:
            input(f"[HITL] Edit {filepath} then press Enter to continue...")
        except (EOFError, KeyboardInterrupt):
            print("\n[HITL] Continuing.", file=sys.stderr)


# ---------------------------------------------------------------------------
# 3. FileChannel
# ---------------------------------------------------------------------------

class FileChannel(HitlChannel):
    """
    Pre-recorded answers loaded from a JSON file.

    The file path is taken from the HARNESS_HITL_FILE environment variable.
    File format::

        [
          {"prompt": "REQUIREMENTS", "answer": "a"},
          {"prompt": "deploy preview", "answer": "y"},
          {"prompt": "refine", "answer": "add more detail about auth"}
        ]

    Matching is by substring — the first entry whose ``prompt`` value is a
    substring of the actual prompt message (case-insensitive) is used.

    Unmatched prompts raise ``RuntimeError`` (fail-closed). This is
    intentional: a script that doesn't pre-record all prompts should fail
    loudly rather than silently proceeding or hanging.
    """

    def __init__(self, answers_path: str) -> None:
        with open(answers_path, "r", encoding="utf-8") as f:
            raw: list[dict[str, str]] = json.load(f)
        self._answers: list[tuple[str, str]] = [
            (entry["prompt"], entry["answer"]) for entry in raw
        ]
        self._used: set[int] = set()
        logger.info("[hitl:file] Loaded %d pre-recorded answers from %s", len(self._answers), answers_path)

    def _lookup(self, message: str) -> str:
        message_lower = message.lower()
        for i, (prompt, answer) in enumerate(self._answers):
            if prompt.lower() in message_lower:
                if i not in self._used:
                    self._used.add(i)
                    logger.info("[hitl:file] Matched prompt %r → %r", prompt, answer)
                    return answer
        raise RuntimeError(
            f"[hitl:file] No pre-recorded answer for prompt: {message[:120]!r}. "
            f"Add an entry to the HARNESS_HITL_FILE to cover this prompt."
        )

    def is_interactive(self) -> bool:
        return False

    def prompt(
        self,
        message: str,
        options: list[str],
        default: Optional[str] = None,
        option_labels: Optional[dict[str, str]] = None,
    ) -> str:
        del option_labels  # file channel matches by message substring
        answer = self._lookup(message)
        logger.info("[hitl:file] prompt → %r", answer)
        return answer

    def confirm(self, message: str, default: bool = False) -> bool:
        answer = self._lookup(message)
        return answer.lower() in ("y", "yes", "true", "1")

    def notes(self, message: str) -> str:
        return self._lookup(message)

    def wait_for_manual_edit(self, filepath: str) -> None:
        logger.info("[hitl:file] Skipping wait_for_manual_edit: %s", filepath)


# ---------------------------------------------------------------------------
# 4. HttpChannel — HTTP webhook transport
# ---------------------------------------------------------------------------

class HttpChannel(HitlChannel):
    """
    HTTP webhook HITL channel.

    Sends each prompt to a remote HTTP endpoint and reads the response.
    Designed for IDE plugins, agent-server orchestrators, and any integration
    that needs to drive the harness over a network interface.

    Configuration (via environment variables):
        HARNESS_HITL_WEBHOOK_URL      — Required. HTTP/HTTPS endpoint to POST to.
        HARNESS_HITL_WEBHOOK_SECRET   — Optional. HMAC-SHA256 signing key. When
                                        set, a ``X-Harness-Signature`` header
                                        (``sha256=<hex>``) is added to each
                                        request so the server can verify origin.
        HARNESS_HITL_WEBHOOK_TIMEOUT  — Optional. Request timeout in seconds
                                        (default 30). Increase for slow human
                                        review workflows.
        HARNESS_HITL_WEBHOOK_RETRIES  — Optional. Number of retries on transient
                                        errors (default 2).

    Request format (POST, Content-Type: application/json):
        {
          "type":    "prompt" | "confirm" | "notes" | "wait_for_edit",
          "message": "<prompt text>",
          "options": ["a", "b", "c"],          // only for "prompt" type
          "default": "a",                        // null if no default
          "option_labels": {"a": "Approve",     // optional; only for "prompt".
                            "b": "Edit hint",   // present when the caller wants
                            "c": "Manual"}      // the UI to render a labeled dropdown
        }

    Expected response (HTTP 200, Content-Type: application/json):
        { "answer": "<string>" }

    For "confirm" the answer is interpreted as truthy when it equals
    "y", "yes", "true", or "1" (case-insensitive).
    For "wait_for_edit" the answer is ignored — any 200 response unblocks.

    If the server returns a non-200 status or the request fails, an error
    is logged and the ``default`` value is used as a fallback.
    """

    def __init__(
        self,
        url: str,
        secret: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.url = url
        self._secret = secret
        self.timeout = timeout
        self.max_retries = max_retries
        logger.info("[hitl:http] Webhook channel configured → %s", url)

    def is_interactive(self) -> bool:
        return True  # a human is on the other end of the webhook

    def _build_payload(self, type_: str, message: str,
                       options: Optional[list[str]] = None,
                       default: Optional[str] = None,
                       option_labels: Optional[dict[str, str]] = None) -> bytes:
        body: dict[str, Any] = {
            "type": type_,
            "message": message,
            "options": options or [],
            "default": default,
        }
        if option_labels:
            body["option_labels"] = option_labels
        return json.dumps(body, ensure_ascii=False).encode("utf-8")

    def _sign(self, body: bytes) -> str:
        """Return ``sha256=<hex>`` HMAC signature for the body."""
        assert self._secret is not None
        sig = hmac.new(
            self._secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()
        return f"sha256={sig}"

    def _post(self, payload: bytes, default_answer: str) -> str:
        """
        POST ``payload`` to the webhook URL and return the ``answer`` string.
        Retries on transient network errors; returns ``default_answer`` on failure.

        Emits ``hitl_pending`` / ``hitl_resolved`` structured events into
        the per-session JSONL so the dashboard's SSE stream can surface
        the HITL banner the instant the prompt is sent — and clear it
        the instant the operator's answer comes back. The harness uses
        ``emit_event`` (deferred import) to avoid bootstrap-order tangles.
        """
        headers = {"Content-Type": "application/json"}
        if self._secret:
            headers["X-Harness-Signature"] = self._sign(payload)

        # Decode just enough metadata for the structured event. The
        # payload was JSON-encoded in ``_build_payload`` immediately
        # before this call, so ``json.loads`` here is cheap and reliable.
        meta_type = ""
        meta_message = ""
        try:
            meta = json.loads(payload.decode("utf-8"))
            meta_type = str(meta.get("type", ""))
            meta_message = str(meta.get("message", ""))[:200]
        except Exception:  # noqa: BLE001
            pass
        try:
            from harness.observability import emit_event
            # Avoid kwarg names that collide with reserved LogRecord
            # attributes (``message`` / ``asctime`` raise KeyError when
            # passed via ``extra=``); ``prompt_message`` is a safe alias.
            emit_event(
                "hitl_pending",
                hitl_type=meta_type,
                webhook_url=self.url,
                prompt_message=meta_message,
            )
        except Exception:  # noqa: BLE001
            pass

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    self.url,
                    data=payload,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)
                    answer = str(data.get("answer", default_answer))
                    try:
                        from harness.observability import emit_event
                        emit_event(
                            "hitl_resolved",
                            hitl_type=meta_type,
                            answer=answer[:200],
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    return answer
            except urllib.error.HTTPError as exc:
                logger.warning(
                    "[hitl:http] Webhook returned HTTP %d on attempt %d/%d.",
                    exc.code, attempt + 1, self.max_retries + 1,
                )
                last_err = exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "[hitl:http] Webhook error on attempt %d/%d: %s",
                    attempt + 1, self.max_retries + 1, exc,
                )
                last_err = exc

            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 8))  # brief backoff between retries

        logger.error(
            "[hitl:http] Webhook failed after %d attempt(s): %s. Using default: %r.",
            self.max_retries + 1, last_err, default_answer,
        )
        return default_answer

    def prompt(
        self,
        message: str,
        options: list[str],
        default: Optional[str] = None,
        option_labels: Optional[dict[str, str]] = None,
    ) -> str:
        effective_default = default if default is not None else (options[0] if options else "")
        payload = self._build_payload(
            "prompt", message, options, effective_default,
            option_labels=option_labels,
        )
        answer = self._post(payload, effective_default)
        logger.info("[hitl:http] prompt %r → %r", message[:60], answer)
        return answer

    def confirm(self, message: str, default: bool = False) -> bool:
        default_str = "y" if default else "n"
        payload = self._build_payload("confirm", message, None, default_str)
        answer = self._post(payload, default_str)
        result = answer.lower() in ("y", "yes", "true", "1")
        logger.info("[hitl:http] confirm %r → %s", message[:60], result)
        return result

    def notes(self, message: str) -> str:
        payload = self._build_payload("notes", message, None, "")
        answer = self._post(payload, "")
        logger.info("[hitl:http] notes → %d chars", len(answer))
        return answer

    def wait_for_manual_edit(self, filepath: str) -> None:
        payload = self._build_payload("wait_for_edit", filepath, None, "done")
        self._post(payload, "done")
        logger.info("[hitl:http] wait_for_manual_edit %s — unblocked.", filepath)


# ---------------------------------------------------------------------------
# 5. Module-level channel registry
# ---------------------------------------------------------------------------

_channel: Optional[HitlChannel] = None


def get_channel() -> HitlChannel:
    """
    Return the active HITL channel.

    Selection order:
      1. A channel explicitly installed via set_channel().
      2. HttpChannel  when HARNESS_HITL_WEBHOOK_URL is set.
      3. FileChannel  when HARNESS_HITL_FILE is set.
      4. StdinChannel (default).
    """
    global _channel
    if _channel is not None:
        return _channel

    webhook_url = os.environ.get("HARNESS_HITL_WEBHOOK_URL", "").strip()
    if webhook_url:
        secret = os.environ.get("HARNESS_HITL_WEBHOOK_SECRET", "").strip() or None
        try:
            timeout = float(os.environ.get("HARNESS_HITL_WEBHOOK_TIMEOUT", "30"))
        except ValueError:
            timeout = 30.0
        try:
            retries = int(os.environ.get("HARNESS_HITL_WEBHOOK_RETRIES", "2"))
        except ValueError:
            retries = 2
        _channel = HttpChannel(webhook_url, secret=secret, timeout=timeout, max_retries=retries)
        return _channel

    hitl_file = os.environ.get("HARNESS_HITL_FILE", "").strip()
    if hitl_file:
        _channel = FileChannel(hitl_file)
        return _channel

    _channel = StdinChannel()
    return _channel


def set_channel(channel: HitlChannel) -> None:
    """Install a specific channel — useful in tests and embeddings."""
    global _channel
    _channel = channel


def reset_channel() -> None:
    """Reset the channel to auto-detect on next call — use in tests."""
    global _channel
    _channel = None
