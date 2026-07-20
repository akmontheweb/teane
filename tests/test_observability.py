"""Tests for harness/observability.py — JSONFormatter, session log, LangSmith."""
import json
import logging
import os
import tempfile



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
                lines = [ln.strip() for ln in f if ln.strip()]
            assert len(lines) >= 1
            for line in lines:
                json.loads(line)  # every line must be valid JSON

    def test_returns_none_on_unwritable_log_dir(self, monkeypatch):
        from harness.observability import configure_logging
        import logging.handlers as logging_handlers

        def bad_fh(*a, **kw):
            raise OSError("simulated disk full")

        # P2.3: file handler is now RotatingFileHandler by default — patch
        # that to simulate disk failure. Also patch the plain FileHandler in
        # case max_bytes=0 is passed (legacy path).
        monkeypatch.setattr(logging_handlers, "RotatingFileHandler", bad_fh)
        monkeypatch.setattr(logging, "FileHandler", bad_fh)
        with tempfile.TemporaryDirectory() as log_dir:
            path = configure_logging(
                session_id="fail-session",
                log_dir=log_dir,
                level="INFO",
            )
        # Must not raise — returns None on failure
        assert path is None

    def test_rotating_handler_used_by_default(self):
        from harness.observability import configure_logging
        from logging.handlers import RotatingFileHandler
        with tempfile.TemporaryDirectory() as log_dir:
            configure_logging(
                session_id="rotate-default",
                log_dir=log_dir,
                level="INFO",
            )
            handlers = logging.getLogger().handlers
            file_handlers = [h for h in handlers if isinstance(h, RotatingFileHandler)]
            assert file_handlers, "default config should install RotatingFileHandler"
            rfh = file_handlers[0]
            assert rfh.maxBytes == 10_000_000
            assert rfh.backupCount == 5

    def test_rotation_actually_rotates_when_size_exceeded(self):
        from harness.observability import configure_logging
        with tempfile.TemporaryDirectory() as log_dir:
            path = configure_logging(
                session_id="rotate-small",
                log_dir=log_dir,
                level="DEBUG",
                max_bytes=2_000,    # tiny cap so a few records trip rotation
                backup_count=2,
            )
            # Each JSON record is well under 2 KB; emit enough to force at
            # least one rotation.
            log = logging.getLogger("harness.rotate.test")
            for i in range(200):
                log.info("rotation-test record %d with some filler payload xxxxxxxxxxxxxxxxxxxxxxxxxxx", i)
            for h in logging.getLogger().handlers:
                h.flush()
            # After rotation the live file and at least one .1 backup must exist.
            assert os.path.isfile(path)
            assert os.path.isfile(path + ".1"), "expected at least one rotated backup"

    def test_max_bytes_zero_uses_plain_file_handler(self):
        from harness.observability import configure_logging
        from logging.handlers import RotatingFileHandler
        with tempfile.TemporaryDirectory() as log_dir:
            configure_logging(
                session_id="rotate-off",
                log_dir=log_dir,
                level="INFO",
                max_bytes=0,
            )
            handlers = logging.getLogger().handlers
            assert not any(isinstance(h, RotatingFileHandler) for h in handlers), \
                "max_bytes=0 must opt out of rotation"
            assert any(isinstance(h, logging.FileHandler) for h in handlers)

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
                lines = [json.loads(ln) for ln in f if ln.strip()]
            events = [ev for ev in lines if ev.get("event") == "llm_call"]
            assert len(events) >= 1
            assert events[0]["model"] == "x:y"
            assert events[0]["cost_usd"] == 0.01

    def test_log_failure_emits_error_level_with_event_field(self):
        from harness.observability import configure_logging, log_failure
        with tempfile.TemporaryDirectory() as log_dir:
            path = configure_logging(session_id="fail-test", log_dir=log_dir, level="DEBUG")
            log_failure(
                "sandbox_start_failed",
                reason="auto_detect_no_backend",
                docker_available=False,
            )
            for handler in logging.getLogger().handlers:
                handler.flush()
            with open(path) as f:
                lines = [json.loads(ln) for ln in f if ln.strip()]
            events = [ev for ev in lines if ev.get("event") == "sandbox_start_failed"]
            assert len(events) >= 1
            assert events[0]["level"] == "ERROR"
            assert events[0]["reason"] == "auto_detect_no_backend"
            assert events[0]["docker_available"] is False

    def test_log_failure_token_budget_exhausted(self):
        from harness.observability import configure_logging, log_failure
        with tempfile.TemporaryDirectory() as log_dir:
            path = configure_logging(session_id="budget-test", log_dir=log_dir, level="DEBUG")
            log_failure(
                "token_budget_exhausted",
                hard_cap_usd=2.0,
                budget_remaining_usd=-0.01,
            )
            for handler in logging.getLogger().handlers:
                handler.flush()
            with open(path) as f:
                events = [json.loads(ln) for ln in f if ln.strip() and "token_budget_exhausted" in ln]
            assert len(events) >= 1
            assert events[0]["hard_cap_usd"] == 2.0

    def test_emit_with_langsmith_enabled(self, monkeypatch):
        """emit_event should handle LangSmith integration if available."""
        from harness.observability import configure_logging, emit_event
        # Mock langsmith availability
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key")
        monkeypatch.setenv("LANGSMITH_PROJECT", "harness-test")

        with tempfile.TemporaryDirectory() as log_dir:
            path = configure_logging(session_id="ls-emit-test", log_dir=log_dir, level="DEBUG")
            emit_event("test_event", detail="test data")
            for handler in logging.getLogger().handlers:
                handler.flush()
            # Event should be logged regardless of LangSmith availability
            assert path is not None


class TestClassifyIncidentCause:
    """The normalized cause vocabulary the incident analysis groups by.
    Pins the trigger→cause mapping so a category rename is a deliberate,
    test-visible change."""

    def test_test_triggers_bucket_as_test(self):
        from harness.observability import classify_incident_cause as clf
        assert clf("unsatisfiable_test:server/tests/test_x.py") == "test_unsatisfiable"
        assert clf("llm_behavior:test_generation_zero_emit") == "test_generation"
        assert clf("traceability_block") == "test_traceability"

    def test_patching_and_repair_triggers(self):
        from harness.observability import classify_incident_cause as clf
        assert clf("zero_patch_loop:2") == "patching_zero_patch"
        assert clf("replace_block_stuck:app/main.py+1") == "patching_stuck_target"
        assert clf("all_allowlist_rejected:3") == "patching_allowlist"
        assert clf("consecutive_distraction:3") == "repair_distraction"
        assert clf("repair_loop_limit") == "repair_limit"

    def test_infra_and_spec_and_budget(self):
        from harness.observability import classify_incident_cause as clf
        assert clf("budget_exhausted") == "budget_exhausted"
        assert clf("no_progress_failsafe") == "no_progress"
        assert clf("decomposition_validation_failed") == "spec_decomposition"
        assert clf("env_misconfig:docker") == "env_infra"
        assert clf("build_command_blocked:rm") == "env_infra"
        assert clf("security_fix_limit:2/2") == "security"

    def test_unknown_and_empty_fall_through(self):
        from harness.observability import classify_incident_cause as clf
        assert clf("persistent_build_failure") == "build_failure"
        assert clf("unknown") == "other"
        assert clf("") == "other"


class TestEmitIncident:
    def _read_incidents(self, path):
        for handler in logging.getLogger().handlers:
            handler.flush()
        with open(path) as f:
            return [
                json.loads(ln) for ln in f
                if ln.strip() and '"incident"' in ln
            ]

    def test_incident_carries_cause_cost_and_test_flag(self):
        from harness.observability import configure_logging, emit_incident
        with tempfile.TemporaryDirectory() as log_dir:
            path = configure_logging(
                session_id="inc-1", log_dir=log_dir, level="DEBUG",
            )
            emit_incident(
                trigger="zero_patch_loop:2",
                session_id="inc-1",
                usd_spent=1.7558,
                on_test_file=True,
                rounds=2,
                story_id="STORY-NFR-005",
                modified_files=7,
            )
            events = self._read_incidents(path)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "incident"
        assert e["cause"] == "patching_zero_patch"
        assert e["trigger"] == "zero_patch_loop:2"
        assert e["usd_spent"] == 1.7558
        assert e["on_test_file"] is True
        assert e["rounds"] == 2
        assert e["story_id"] == "STORY-NFR-005"
        # wall_clock_s defaults to process uptime (a non-negative float).
        assert isinstance(e["wall_clock_s"], (int, float))
        assert e["wall_clock_s"] >= 0

    def test_none_fields_are_omitted(self):
        from harness.observability import configure_logging, emit_incident
        with tempfile.TemporaryDirectory() as log_dir:
            path = configure_logging(
                session_id="inc-2", log_dir=log_dir, level="DEBUG",
            )
            # No usd/test-flag/rounds supplied.
            emit_incident(trigger="budget_exhausted", session_id="inc-2")
            events = self._read_incidents(path)
        assert len(events) == 1
        e = events[0]
        assert e["cause"] == "budget_exhausted"
        assert "usd_spent" not in e       # None → omitted
        assert "on_test_file" not in e
        assert "rounds" not in e
        # wall_clock_s is always present (falls back to process uptime).
        assert "wall_clock_s" in e


class TestSummarizeIncidents:
    def _write_log(self, log_dir, sid, incidents):
        from harness.observability import configure_logging, emit_incident
        path = configure_logging(session_id=sid, log_dir=log_dir, level="DEBUG")
        for kw in incidents:
            emit_incident(**kw)
        for handler in logging.getLogger().handlers:
            handler.flush()
        return path

    def test_buckets_by_cause_with_cost_and_test_share(self):
        from harness.observability import summarize_incidents
        with tempfile.TemporaryDirectory() as d:
            p1 = self._write_log(d, "s1", [
                dict(trigger="unsatisfiable_test:t.py", usd_spent=1.5,
                     wall_clock_s=9000, on_test_file=True),
                dict(trigger="zero_patch_loop:2", usd_spent=0.5,
                     wall_clock_s=120, on_test_file=True),   # test-caused repair
                dict(trigger="budget_exhausted", usd_spent=2.0,
                     wall_clock_s=300, on_test_file=False),
            ])
            summary = summarize_incidents([p1])
        assert summary["total_incidents"] == 3
        # 2 of 3 are test-related: the test_unsatisfiable one + the
        # zero_patch loop flagged on_test_file.
        assert summary["test_related"] == 2
        assert summary["test_share"] == round(2 / 3, 3)
        assert summary["by_cause"]["test_unsatisfiable"]["count"] == 1
        assert summary["by_cause"]["test_unsatisfiable"]["usd"] == 1.5
        assert summary["by_cause"]["patching_zero_patch"]["on_test"] == 1
        assert summary["total_usd"] == 4.0

    def test_skips_malformed_and_missing_files(self):
        from harness.observability import summarize_incidents
        with tempfile.TemporaryDirectory() as d:
            good = self._write_log(d, "ok", [
                dict(trigger="budget_exhausted", usd_spent=1.0),
            ])
            bad = os.path.join(d, "broken.jsonl")
            with open(bad, "w") as f:
                f.write('{not json\n"incident" but broken}\n')
            summary = summarize_incidents(
                [good, bad, os.path.join(d, "nope.jsonl")],
            )
        assert summary["total_incidents"] == 1

    def test_empty_inputs_return_zero_share(self):
        from harness.observability import summarize_incidents
        assert summarize_incidents([])["test_share"] == 0.0
