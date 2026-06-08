"""Tests for harness/redactor.py — Secret scanning and redaction."""

import pytest
from harness.redactor import (
    SecretScanner,
    RedactionResult,
    _is_high_entropy,
)


class TestIsHighEntropy:
    """Test entropy detection for potential secrets."""

    def test_empty_string_is_not_high_entropy(self):
        """Empty string should return False."""
        assert _is_high_entropy("") is False

    def test_short_string_is_not_high_entropy(self):
        """Strings < 16 chars are not checked for entropy."""
        assert _is_high_entropy("short") is False
        assert _is_high_entropy("12345678901234") is False  # 14 chars
        assert _is_high_entropy("123456789012345") is False  # 15 chars

    def test_low_entropy_string_is_not_high_entropy(self):
        """Repeated characters have low entropy."""
        assert _is_high_entropy("aaaaaaaaaaaaaaaa") is False  # 16 chars, all same
        assert _is_high_entropy("abcdefghijklmnop") is False  # repeating alpha

    def test_pure_hex_is_not_high_entropy(self):
        """Pure hex (like git SHAs) should be low entropy."""
        # 40 char hex SHA → ~4 bits/char, below threshold
        assert _is_high_entropy("1234567890abcdef1234567890abcdef12345678") is False

    def test_high_entropy_mixed_string(self):
        """Mixed case + numbers + symbols should be high entropy."""
        # A random-looking base64ish string with high entropy
        assert _is_high_entropy("aB3kL9mP2qX7wR4tY6uV8sJ1cN5fG0dH") is True
        assert _is_high_entropy("sk-proj-aAbBcCdDeEfFgGhHiIjJkKlMnNo") is True

    def test_entropy_threshold_at_boundary(self):
        """Test strings near the entropy threshold."""
        # Build a string that's designed to be just above threshold
        # A real secret-like string
        high_entropy_str = "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"
        assert _is_high_entropy(high_entropy_str) is True


class TestSecretScanner:
    """Test SecretScanner pattern detection and redaction."""

    def test_github_token_pat_detected(self):
        """GitHub fine-grained PAT should be detected."""
        scanner = SecretScanner(mode="mask")
        text = "My token is github_pat_abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrst"
        redacted, result = scanner.redact_text(text)
        assert "[REDACTED]" in redacted
        assert result.replacements == 1
        assert "GitHub fine-grained PAT" in result.redacted_types

    def test_openai_key_detected(self):
        """OpenAI API key should be detected."""
        scanner = SecretScanner(mode="mask")
        text = "OpenAI key: sk-proj-1234567890abcdefghijklmnop"
        redacted, result = scanner.redact_text(text)
        assert "[REDACTED]" in redacted
        assert result.replacements == 1

    def test_anthropic_key_detected(self):
        """Anthropic API key should be detected."""
        scanner = SecretScanner(mode="mask")
        text = "sk-ant-api01-abcdefghijklmnopqrstuvwxyz1234567890abcdef"
        redacted, result = scanner.redact_text(text)
        assert "[REDACTED]" in redacted
        assert result.replacements == 1

    def test_jwt_token_detected(self):
        """JWT tokens should be detected."""
        scanner = SecretScanner(mode="mask")
        text = "JWT: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        redacted, result = scanner.redact_text(text)
        assert result.replacements > 0
        assert "[REDACTED]" in redacted

    def test_redaction_mode_hash(self):
        """Hash mode should include sha256 digest."""
        scanner = SecretScanner(mode="hash")
        text = "sk-proj-1234567890abcdefghijklmnop"
        redacted, result = scanner.redact_text(text)
        assert "[REDACTED:" in redacted
        assert "sha256:" in redacted
        assert result.replacements == 1

    def test_redaction_mode_strip(self):
        """Strip mode should remove the secret entirely."""
        scanner = SecretScanner(mode="strip")
        text = "prefix sk-proj-1234567890abcdefghijklmnop suffix"
        redacted, result = scanner.redact_text(text)
        assert "1234567890abcdefghijklmnop" not in redacted
        assert "prefix  suffix" in redacted or "prefix suffix" in redacted
        assert result.replacements == 1

    def test_multiple_secrets_in_text(self):
        """Multiple secrets should all be redacted."""
        scanner = SecretScanner(mode="mask")
        text = "token1: sk-proj-111111111111111111111111 token2: sk-proj-222222222222222222222222"
        redacted, result = scanner.redact_text(text)
        assert redacted.count("[REDACTED]") == 2
        assert result.replacements == 2

    def test_no_secrets_in_text(self):
        """Text without secrets should be unchanged."""
        scanner = SecretScanner(mode="mask")
        text = "This is normal text with no secrets"
        redacted, result = scanner.redact_text(text)
        assert redacted == text
        assert result.replacements == 0
        assert len(result.redacted_types) == 0

    def test_custom_patterns(self):
        """Custom regex patterns should be applied."""
        scanner = SecretScanner(
            mode="mask",
            custom_patterns=[r"SECRET_\w+"],
        )
        text = "my SECRET_PASSWORD is here"
        redacted, result = scanner.redact_text(text)
        assert "[REDACTED]" in redacted
        assert result.replacements == 1
        assert "custom" in result.redacted_types

    def test_entropy_detection_disabled_by_default(self):
        """Entropy detection should be opt-in."""
        scanner = SecretScanner(entropy_detection=False)
        # A high-entropy string that doesn't match any pattern
        text = "random string with aAbBcCdDeEfFgGhHiIjJkKlMnOpQrStUvWx in it"
        redacted, result = scanner.redact_text(text)
        # Should not redact entropy-based unless enabled
        assert result.replacements == 0

    def test_entropy_detection_enabled(self):
        """When enabled, high-entropy strings should be redacted."""
        scanner = SecretScanner(entropy_detection=True, mode="mask")
        # A high-entropy string that doesn't match patterns (no 'key=', 'token=', etc)
        text = "here is aAbBcCdDeEfFgGhHiIjJkKlMnOpQrStUvWxYz0123456789 in text"
        redacted, result = scanner.redact_text(text)
        assert "[REDACTED]" in redacted
        assert "high-entropy" in result.redacted_types

    def test_entropy_detection_skips_uuids(self):
        """UUIDs should be skipped during entropy pass."""
        scanner = SecretScanner(entropy_detection=True, mode="mask")
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        text = f"UUID: {uuid}"
        redacted, result = scanner.redact_text(text)
        assert uuid in redacted  # UUID should not be redacted
        assert result.replacements == 0

    def test_entropy_detection_skips_pure_hex(self):
        """Pure hex (git SHAs) should be skipped."""
        scanner = SecretScanner(entropy_detection=True, mode="mask")
        sha = "1234567890abcdef1234567890abcdef12345678"
        text = f"commit: {sha}"
        redacted, result = scanner.redact_text(text)
        assert sha in redacted  # SHA should not be redacted
        assert result.replacements == 0

    def test_redact_messages(self):
        """redact_messages should redact content in message dicts."""
        scanner = SecretScanner(mode="mask")
        messages = [
            {"role": "user", "content": "My token is sk-proj-1234567890abcdefghijklmnop"},
            {"role": "assistant", "content": "Here's the response"},
        ]
        redacted_msgs, result = scanner.redact_messages(messages)
        assert "[REDACTED]" in redacted_msgs[0]["content"]
        assert redacted_msgs[1]["content"] == "Here's the response"
        assert result.replacements == 1

    def test_redact_messages_with_non_string_content(self):
        """redact_messages should handle non-string content gracefully."""
        scanner = SecretScanner(mode="mask")
        messages = [
            {"role": "user", "content": "normal text"},
            {"role": "tool", "content": 12345},  # numeric content
        ]
        redacted_msgs, result = scanner.redact_messages(messages)
        assert len(redacted_msgs) == 2
        assert redacted_msgs[1]["content"] == 12345  # unchanged

    def test_redact_file_if_sensitive(self):
        """redact_file_if_sensitive should match scan_files patterns."""
        scanner = SecretScanner(
            scan_files=[".env", ".env.local", "*.pem", "*.key"]
        )
        assert "[REDACTED FILE:" in scanner.redact_file_if_sensitive(".env")
        assert "[REDACTED FILE:" in scanner.redact_file_if_sensitive(".env.local")
        assert "[REDACTED FILE:" in scanner.redact_file_if_sensitive("id_rsa.pem")
        assert "[REDACTED FILE:" in scanner.redact_file_if_sensitive("cert.key")
        assert scanner.redact_file_if_sensitive("normal.txt") is None

    def test_multiline_secret_redaction(self):
        """Secrets spanning multiple lines should be redacted."""
        scanner = SecretScanner(mode="mask")
        text = """
        This is a multiline secret:
        sk-proj-1234567890abcdefghijklmnop
        Another secret: sk-ant-api01-abcdefghijklmnopqrstuvwxyz1234567890abcdef
        """
        redacted, result = scanner.redact_text(text)
        assert result.replacements == 2
        assert redacted.count("[REDACTED]") == 2

    def test_redaction_result_aggregation(self):
        """RedactionResult should aggregate types correctly."""
        scanner = SecretScanner(mode="mask")
        text = "OpenAI: sk-proj-abcdefghijklmnopqrst GitHub: github_pat_abcdefghijklmnopqrst12345"
        redacted, result = scanner.redact_text(text)
        assert result.replacements == 2
        assert "OpenAI API key" in result.redacted_types
        assert "GitHub fine-grained PAT" in result.redacted_types
