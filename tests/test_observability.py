"""Tests for harness/observability.py — JSONFormatter, session log, LangSmith."""
import json
import logging
import os
import tempfile

import pytest


class TestJSONFormatter:

    def test_emits_required_fields(self):
        from harness.observability import JSONFormatter
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="harness.test", level=logging.INFO,
            pathname="", lineno=0, msg="hello", args=(), exc_info=None,
        )
        line = formatter.format(record)
        obj = json.loads(line)
        assert "ts" in obj
        assert obj["level"] == "INFO"
        assert obj["logger"] == "harness.test"
        assert obj["msg"] == "hello"

    def test_merges_extra_fields(self):
        from harness.observability import JSONFormatter
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="harness.test", level=logging.DEBUG,
            pathname="", lineno=0, msg="event", args=(), exc_info=None,
        )
        record.cost_usd = 0.0123
        record.model = "anthropic:claude-test"
        line = formatter.format(record)
        obj = json.loads(line)
        assert obj["cost_usd"] == 0.0123
        assert obj["model"] == "anthropic:claude-test"

    def test_excludes_stdlib_internals(self):
        from harness.observability import JSONFormatter
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="x", level=logging.INFO,
            pathname="/x.py", lineno=5, msg="m", args=(), exc_info=None,
        )
        line = formatter.format(record)
        obj = json.loads(line)
        # Internal stdlib attrs must NOT appear at the top level
        assert "pathname" not in obj
        assert "lineno" not in obj
        assert "args" not in obj

    def test_output_is_valid_json(self):
        from harness.observability import JSONFormatter
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="h", level=logging.WARNING,
            pathname="", lineno=0, msg="warn %s", args=("value",), exc_info=None,
        )
        json.loads(formatter.format(record))  # raises on invalid JSON


class TestConfigureLogging:

    def test_creates_session_log_file(self):
        from harness.observability import configure_logging
        with tempfile.TemporaryDirectory() as log_dir:
            path = configure_logging(
                session_id="test-session-abc123",
                log_dir=log_dir,
                level="INFO",
            )
            assert path is not None
            assert os.path.isfile(path)
            assert path.endswith("test-session-abc123.jsonl")

    def test_log_file_contains_valid_jsonl(self):
        from harness.observability import configure_logging
        with tempfile.TemporaryDirectory() as log_dir:
            path = configure_logging(
                session_id="test-jsonl",
                log_dir=log_dir,
                level="DEBUG",
            )
            # Emit a record
            logging.getLogger("harness.test").info("hello from test")
            # Flush and read
            for handler in logging.getLogger().handlers:
                handler.flush()
            with open(path) as f:
                lines = [l.strip() for l in f if l.strip()]
            assert len(lines) >= 1
            for line in lines:
                json.loads(line)  # every line must be valid JSON

    def test_returns_none_on_unwritable_log_dir(self, monkeypatch):
        from harness.observability import configure_logging
        import harness.observability as obs_mod

        def bad_fh(*a, **kw):
            raise OSError("simulated disk full")

        monkeypatch.setattr(obs_mod.logging, "FileHandler", bad_fh)
        with tempfile.TemporaryDirectory() as log_dir:
            path = configure_logging(
                session_id="fail-session",
                log_dir=log_dir,
                level="INFO",
            )
        # Must not raise — returns None on failure
        assert path is None

    def test_no_langsmith_without_api_key(self, monkeypatch):
        # When LANGCHAIN_API_KEY is absent, LangSmith init must skip silently.
        from harness.observability import configure_logging
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        with tempfile.TemporaryDirectory() as log_dir:
            # Should not raise regardless of whether langsmith is installed
            configure_logging(
                session_id="ls-test",
                log_dir=log_dir,
                langsmith_enabled=True,
            )
        # Verify the env vars are NOT set (no partial initialisation leaked)
        assert os.environ.get("LANGSMITH_TRACING_V2") != "true" or \
               os.environ.get("LANGSMITH_PROJECT", "").startswith("harness-") is False


class TestEmitEvent:

    def test_emit_produces_json_with_event_field(self):
        from harness.observability import configure_logging, emit_event
        with tempfile.TemporaryDirectory() as log_dir:
            path = configure_logging(session_id="emit-test", log_dir=log_dir, level="DEBUG")
            emit_event("llm_call", model="x:y", cost_usd=0.01, tokens_in=100)
            for handler in logging.getLogger().handlers:
                handler.flush()
            with open(path) as f:
                lines = [json.loads(l) for l in f if l.strip()]
            events = [l for l in lines if l.get("event") == "llm_call"]
            assert len(events) >= 1
            assert events[0]["model"] == "x:y"
            assert events[0]["cost_usd"] == 0.01
