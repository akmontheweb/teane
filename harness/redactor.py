"""
Encrypted Variable Redaction — Zero-Knowledge Telemetry.

This module implements:
    - SecretScanner: Regex + entropy-based secret detector that scans text for
      API keys, tokens, passwords, private keys, connection strings, and other
      sensitive credentials before they leave the local machine.
    - Redaction pipeline: Automatically strips or hashes secrets from messages
      before transmission to remote LLM endpoints, from compiler output before
      filtering, and from the system prompt before embedding.
    - Configurable modes: "hash" (replace with SHA-256 fragment), "mask" (replace
      with [REDACTED]), or "strip" (remove entirely).

Integration points:
    - Gateway.dispatch(): redacts all messages before API call
    - SandboxExecutor.run(): redacts build output before log filtering
    - _build_system_prompt(): redacts directory tree snapshot
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Secret Detection Patterns
# ---------------------------------------------------------------------------

# High-confidence regex patterns for known secret formats. These are
# always on — they have low false-positive rates because they match
# specific provider prefixes / structures.
#
# Private-key / SSH-key regexes accept BOTH ``\n`` and ``\r\n`` line
# endings AND inlined ``\\n`` escapes (the common pattern when a key
# value is embedded inside a JSON string field). Audit §3.7.
_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # OpenAI API keys
    (re.compile(r'\b(sk-(?:proj-)?[A-Za-z0-9]{20,})\b'), "OpenAI API key"),
    # Anthropic API keys
    (re.compile(r'\b(sk-ant-api[0-9]{2}-[A-Za-z0-9_-]{40,})\b'), "Anthropic API key"),
    # GitHub tokens (personal access, OAuth, fine-grained PAT)
    (re.compile(r'\b(gh[pousr]_[A-Za-z0-9]{20,})\b'), "GitHub token"),
    (re.compile(r'\b(github_pat_[A-Za-z0-9_]{20,})\b'), "GitHub fine-grained PAT"),
    # Hugging Face tokens
    (re.compile(r'\b(hf_[A-Za-z0-9]{20,})\b'), "Hugging Face token"),
    # AWS Access Key ID (NOT secret key — that requires entropy detection)
    (re.compile(r'\b(AKIA[0-9A-Z]{16})\b'), "AWS Access Key"),
    # npm registry tokens (audit §3.7)
    (re.compile(r'\b(npm_[A-Za-z0-9]{30,})\b'), "npm token"),
    # Slack webhook URLs (audit §3.7)
    (re.compile(r'(https://hooks\.slack\.com/services/T[A-Za-z0-9]+/B[A-Za-z0-9]+/[A-Za-z0-9]+)'), "Slack webhook"),
    # Discord bot tokens (three base64url segments separated by ``.``)
    (re.compile(r'\b([MN][A-Za-z0-9_-]{23}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,})\b'), "Discord bot token"),
    # Azure storage account key (audit §3.7) — ``AccountKey=<base64>``
    (re.compile(r'(AccountKey=[A-Za-z0-9+/=]{20,})'), "Azure storage key"),
    # GCP service-account JSON keys — fingerprint by `"type": "service_account"`
    # plus the surrounding object braces. Catches inlined credentials in
    # JSON strings or pasted secrets. Audit §3.7.
    (re.compile(r'(\{[^{}]*"type"\s*:\s*"service_account"[^{}]*"private_key"[^{}]*})', re.DOTALL), "GCP service-account key"),
    # Generic API key patterns (long alphanumeric strings following 'key=', 'token=', etc.)
    (re.compile(r'(?i)(?:api[_-]?key|secret|token|password|auth)\s*[:=]\s*[\'"]?([A-Za-z0-9+/._\-=]{20,})[\'"]?'), "Generic credential"),
    # JWT tokens
    (re.compile(r'\b(eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)\b'), "JWT token"),
    # Private key PEM blocks — tolerate \r\n and inlined \\n escapes.
    (re.compile(r'-----BEGIN (?:RSA|EC|DSA|OPENSSH|ENCRYPTED) PRIVATE KEY-----(?:\\n|\n|\r\n)(?:[A-Za-z0-9+/=\s\\]{40,}?)-----END'), "Private key"),
    # Connection strings with credentials
    (re.compile(r'(?:postgres|mysql|mongodb|redis|sqlite)://[^:]+:[^@\s]+@'), "Database connection string"),
    # Stripe keys
    (re.compile(r'\b(sk_live_[A-Za-z0-9]{24,})\b'), "Stripe live key"),
    (re.compile(r'\b(pk_live_[A-Za-z0-9]{24,})\b'), "Stripe publishable key"),
    # Google OAuth
    (re.compile(r'\b([0-9]+-[A-Za-z0-9_]{32,}\.apps\.googleusercontent\.com)\b'), "Google OAuth client"),
    # Slack tokens
    (re.compile(r'\b(xox[bpras]-[A-Za-z0-9-]{10,})\b'), "Slack token"),
    # Private SSH keys — tolerate \r\n and inlined \\n.
    (re.compile(r'-----BEGIN OPENSSH PRIVATE KEY-----(?:\\n|\n|\r\n)(?:[A-Za-z0-9+/=\s\\]{40,}?)-----END'), "SSH private key"),
]

# Note: bare "high-entropy hex/base64" matchers (formerly part of
# _SECRET_PATTERNS) were removed because they eat git commit SHAs,
# UUIDs-without-dashes, file content hashes, base64-encoded protobufs,
# and many other non-secret strings — a ~30–50% false-positive rate on
# realistic code messages. They are now part of the opt-in entropy
# pass (see SecretScanner.entropy_detection, default off).


def _is_high_entropy(value: str) -> bool:
    """
    Check if a string has high Shannon entropy (likely a secret, not a word).

    Only applies to strings longer than 16 characters to reduce false positives
    on regular dictionary words or short identifiers.
    """
    if len(value) < 16:
        return False

    # Shannon entropy in bits/char: H = -sum(p_i * log2(p_i))
    freq: dict[str, int] = {}
    for char in value:
        freq[char] = freq.get(char, 0) + 1
    total = float(len(value))

    entropy_bits_per_char = 0.0
    for count in freq.values():
        p = count / total
        entropy_bits_per_char -= p * math.log2(p)

    # Pure hex has theoretical max ~4 bits/char; base64 ~6 bits/char.
    # Require >4.5 bits/char so we trip on mixed-alphabet base64-ish secrets
    # but not on pure-hex git SHAs (those are also filtered by the
    # _ENTROPY_SKIP_PATTERNS shortcut).
    min_entropy = 4.5 if len(value) < 40 else 4.0
    return entropy_bits_per_char > min_entropy


# ---------------------------------------------------------------------------
# 2. Redaction Engine
# ---------------------------------------------------------------------------

class RedactionResult:
    """Result of a redaction pass."""
    def __init__(self) -> None:
        self.replacements: int = 0
        self.redacted_types: set[str] = set()


class SecretScanner:
    """
    Scans text for secrets using regex patterns and entropy analysis.

    Can operate in three modes:
        - "hash": Replace secrets with [REDACTED:sha256:abcdef12] (traceable, safe)
        - "mask": Replace secrets with [REDACTED] (simple, irreversible)
        - "strip": Remove the secret entirely (leaves [])

    Also supports custom patterns from .harness_config.json.
    """

    def __init__(
        self,
        mode: str = "hash",
        custom_patterns: Optional[list[str]] = None,
        scan_files: Optional[list[str]] = None,
        entropy_detection: bool = False,
    ):
        self.mode = mode
        self.custom_patterns = custom_patterns or []
        self.scan_files = scan_files or [".env", ".env.local", "*.pem", "*.key"]
        # Entropy detection is opt-in: it has a high false-positive rate on
        # ordinary code (SHAs, UUIDs, base64 protobufs, content hashes).
        # Enable only when the threat model demands it and the workspace
        # is mostly natural-language / config rather than code.
        self.entropy_detection = entropy_detection
        self._custom_regex: list[tuple[re.Pattern[str], str]] = [
            (re.compile(p), "custom") for p in self.custom_patterns
        ]

    def _redact_value(self, value: str, pattern_label: str) -> str:
        """Redact a single secret value based on the configured mode."""
        if self.mode == "strip":
            return ""

        if self.mode == "hash":
            # Stable hash allows developer to trace which secret was exposed
            hash_snippet = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
            return f"[REDACTED:{pattern_label}:sha256:{hash_snippet}]"

        # Default: "mask"
        return "[REDACTED]"

    def redact_text(self, text: str) -> tuple[str, RedactionResult]:
        """
        Scan and redact secrets from a text string.

        Args:
            text: The text content to scan.

        Returns:
            Tuple of (redacted_text, RedactionResult with stats).
        """
        result = RedactionResult()
        redacted = text

        # Apply all secret patterns
        all_patterns = list(_SECRET_PATTERNS) + list(self._custom_regex)
        for pattern, label in all_patterns:
            def _replace(match: re.Match[str], lbl: str = label) -> str:
                secret_val = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group(0)
                result.replacements += 1
                result.redacted_types.add(lbl)
                return self._redact_value(secret_val, lbl)

            redacted = pattern.sub(_replace, redacted)

        # Entropy-based detection is opt-in. The high false-positive rate
        # on realistic code (git SHAs, UUIDs, file hashes, base64 protobufs)
        # made it harmful when always on.
        if self.entropy_detection and self.mode != "strip":
            redacted = self._redact_high_entropy_strings(redacted, result)

        if result.replacements > 0:
            logger.info(
                "[redactor] Redacted %d secret(s) of types: %s",
                result.replacements,
                ", ".join(sorted(result.redacted_types)),
            )

        return redacted, result

    # Common non-secret patterns to skip during entropy pass.
    _ENTROPY_SKIP_PATTERNS: tuple[re.Pattern[str], ...] = (
        # UUID with dashes
        re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE),
        # Pure hex (git SHAs, file hashes, blob IDs)
        re.compile(r'^[0-9a-fA-F]+$'),
        # Base64 of well-known content lengths (very rough — defers to entropy)
    )

    def _redact_high_entropy_strings(self, text: str, result: RedactionResult) -> str:
        """
        Opt-in pass: find standalone alphanumeric strings with high entropy
        that may be API keys or tokens not matching any known provider format.

        Skips common false-positive shapes (UUIDs, pure hex like git SHAs).
        The entropy threshold in ``_is_high_entropy`` is tuned to require
        mixed-case + digits + symbols — pure hex won't trip it.
        """
        # Find standalone alphanumeric+special tokens > 24 chars (raised from
        # 20 to further reduce FPs on short hashes, JWT ids, etc.)
        potential_secrets = re.finditer(
            r'(?<!\w)([A-Za-z0-9+/=_-]{24,})(?!\w)',
            text,
        )

        replacements: list[tuple[int, int, str]] = []

        for match in potential_secrets:
            value = match.group(1)
            if any(p.match(value) for p in self._ENTROPY_SKIP_PATTERNS):
                continue
            if _is_high_entropy(value):
                replacements.append((match.start(1), match.end(1), self._redact_value(value, "entropy")))

        chars = list(text)
        for start, end, replacement in reversed(replacements):
            chars[start:end] = replacement
            result.replacements += 1
            result.redacted_types.add("high-entropy")

        return "".join(chars)

    def redact_messages(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], RedactionResult]:
        """
        Redact secrets from all message content fields.

        Args:
            messages: List of message dicts with 'content' keys.

        Returns:
            Tuple of (redacted_messages, combined RedactionResult).

        Handles BOTH the string-content shape used by the OpenAI /
        DeepSeek chat completions API and the **typed-content list**
        shape used by Anthropic (and by the harness for tool-use turns)
        where ``content`` is a list of ``{"type": "text"|"tool_use"|
        "tool_result", ...}`` blocks. Audit §3.7 — the earlier
        implementation silently skipped list-form content, so secrets
        inside ``tool_use.input`` or ``tool_result.content`` shipped
        to the provider AND landed in ``~/.harness/debug/*.txt`` dumps.
        """
        combined = RedactionResult()
        redacted = []
        for msg in messages:
            new_msg = dict(msg)
            content = new_msg.get("content", "")
            if isinstance(content, str):
                new_content, result = self.redact_text(content)
                new_msg["content"] = new_content
                combined.replacements += result.replacements
                combined.redacted_types.update(result.redacted_types)
            elif isinstance(content, list):
                new_msg["content"] = self._redact_typed_content_list(content, combined)
            redacted.append(new_msg)
        return redacted, combined

    def _redact_typed_content_list(
        self, blocks: list[Any], combined: "RedactionResult",
    ) -> list[Any]:
        """Walk an Anthropic-style typed-content list and redact every
        string-valued field. Audit §3.7."""
        out: list[Any] = []
        for block in blocks:
            if isinstance(block, dict):
                out.append(self._redact_dict_strings(block, combined))
            elif isinstance(block, str):
                new_text, result = self.redact_text(block)
                combined.replacements += result.replacements
                combined.redacted_types.update(result.redacted_types)
                out.append(new_text)
            else:
                out.append(block)
        return out

    def _redact_dict_strings(
        self, d: dict[str, Any], combined: "RedactionResult",
    ) -> dict[str, Any]:
        """Recurse into a typed-content block, redacting any string leaves.

        Handles common nested shapes:
          {"type": "text", "text": "..."}                  ← Anthropic text block
          {"type": "tool_use", "input": {...}}             ← tool call args
          {"type": "tool_result", "content": "..." | [..]}  ← tool output
        """
        out: dict[str, Any] = {}
        for key, value in d.items():
            if isinstance(value, str):
                new_str, result = self.redact_text(value)
                combined.replacements += result.replacements
                combined.redacted_types.update(result.redacted_types)
                out[key] = new_str
            elif isinstance(value, dict):
                out[key] = self._redact_dict_strings(value, combined)
            elif isinstance(value, list):
                out[key] = self._redact_typed_content_list(value, combined)
            else:
                out[key] = value
        return out

    def redact_file_if_sensitive(self, filepath: str) -> Optional[str]:
        """
        Check if a file matches the scan_files patterns and should be fully redacted.

        Returns:
            "[REDACTED FILE: filename]" if the file should be redacted entirely,
            None if the file should be shown normally.
        """
        basename = os.path.basename(filepath)
        for pattern in self.scan_files:
            # Simple glob: *.pem, .env*, etc.
            if pattern.startswith("*."):
                if basename.endswith(pattern[1:]):
                    return f"[REDACTED FILE: {basename}]"
            elif basename == pattern or basename.startswith(pattern):
                return f"[REDACTED FILE: {basename}]"
        return None


# ---------------------------------------------------------------------------
# 3. Convenience Redaction Functions
# ---------------------------------------------------------------------------

_global_scanner: Optional[SecretScanner] = None


def set_redactor(scanner: SecretScanner) -> None:
    """Set the global SecretScanner instance for use by other modules."""
    global _global_scanner
    _global_scanner = scanner


def get_redactor() -> Optional[SecretScanner]:
    """Get the global SecretScanner instance."""
    return _global_scanner


def redact_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convenience: redact messages using the global scanner.
    Returns unmodified messages if no scanner is configured.
    """
    if _global_scanner is None:
        return messages
    redacted_messages, result = _global_scanner.redact_messages(messages)
    return redacted_messages


def redact_text(text: str) -> str:
    """
    Convenience: redact a text string using the global scanner.
    Returns unmodified text if no scanner is configured.
    """
    if _global_scanner is None:
        return text
    redacted, _result = _global_scanner.redact_text(text)
    return redacted


# ---------------------------------------------------------------------------
# 4. Factory from Config
# ---------------------------------------------------------------------------

def create_redactor_from_config(config_dict: dict[str, Any]) -> SecretScanner:
    """
    Build a SecretScanner from the 'redaction' section of .harness_config.json.

    Args:
        config_dict: Merged configuration dictionary.

    Returns:
        Configured SecretScanner instance.
    """
    rc = config_dict.get("redaction", {})

    scanner = SecretScanner(
        mode=rc.get("mode", "hash"),
        custom_patterns=rc.get("custom_patterns", None),
        scan_files=rc.get("scan_files", None),
        entropy_detection=bool(rc.get("entropy_detection", False)),
    )

    # Register as global for convenience functions
    set_redactor(scanner)
    return scanner