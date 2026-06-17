"""Tests for harness/hitl.py — pluggable HITL transport."""
import hashlib
import hmac
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import pytest

from harness.hitl import (
    FileChannel,
    HttpChannel,
    StdinChannel,
    get_channel,
    reset_channel,
    set_channel,
)


# ---------------------------------------------------------------------------
# Minimal in-process HTTP server for webhook tests
# ---------------------------------------------------------------------------

def _make_server(response_body: bytes = b'{"answer": "ok"}',
                 response_status: int = 200,
                 captured: Optional[list] = None) -> HTTPServer:
    """Spin up a localhost HTTP server that records requests and returns a fixed reply."""

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            if captured is not None:
                captured.append({
                    "path": self.path,
                    "headers": dict(self.headers),
                    "body": json.loads(body) if body else {},
                })
            self.send_response(response_status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, *args, **kwargs):
            pass  # silence test output

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    return server


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    """Reset channel singleton and env vars before each test."""
    reset_channel()
    monkeypatch.delenv("HARNESS_HITL_FILE", raising=False)
    monkeypatch.delenv("HARNESS_AUTO_APPROVE", raising=False)
    monkeypatch.delenv("CI", raising=False)
    yield
    reset_channel()


# ---------------------------------------------------------------------------
# StdinChannel — auto-approve path (no actual stdin needed)
# ---------------------------------------------------------------------------

class TestStdinChannelAutoApprove:

    def test_prompt_returns_default_in_ci(self, monkeypatch):
        monkeypatch.setenv("CI", "true")
        ch = StdinChannel()
        assert ch.prompt("Choose action", ["a", "b", "c"], default="b") == "b"

    def test_prompt_returns_first_option_when_no_default(self, monkeypatch):
        monkeypatch.setenv("HARNESS_AUTO_APPROVE", "true")
        ch = StdinChannel()
        assert ch.prompt("Choose action", ["x", "y"]) == "x"

    def test_confirm_returns_default_in_ci(self, monkeypatch):
        monkeypatch.setenv("CI", "true")
        ch = StdinChannel()
        assert ch.confirm("Proceed?", default=True) is True
        assert ch.confirm("Proceed?", default=False) is False

    def test_notes_returns_empty_in_ci(self, monkeypatch):
        monkeypatch.setenv("CI", "true")
        ch = StdinChannel()
        assert ch.notes("Enter hint") == ""

    def test_wait_for_manual_edit_returns_immediately_in_ci(self, monkeypatch):
        monkeypatch.setenv("CI", "true")
        ch = StdinChannel()
        ch.wait_for_manual_edit("/any/path")  # must not block

    def test_is_interactive_false_in_ci(self, monkeypatch):
        monkeypatch.setenv("CI", "true")
        ch = StdinChannel()
        assert ch.is_interactive() is False


# ---------------------------------------------------------------------------
# FileChannel
# ---------------------------------------------------------------------------

class TestFileChannel:

    def _write_answers(self, tmp_dir: str, entries: list[dict]) -> str:
        path = os.path.join(tmp_dir, "answers.json")
        with open(path, "w") as f:
            json.dump(entries, f)
        return path

    def test_prompt_matches_by_substring(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_answers(td, [
                {"prompt": "REQUIREMENTS", "answer": "a"},
                {"prompt": "deploy preview", "answer": "y"},
            ])
            ch = FileChannel(path)
            assert ch.prompt("[HITL:REQUIREMENTS] Select action", ["a", "e"]) == "a"

    def test_confirm_parses_yes_variants(self):
        with tempfile.TemporaryDirectory() as td:
            for answer in ("y", "yes", "YES", "true", "1"):
                path = self._write_answers(td, [{"prompt": "Proceed", "answer": answer}])
                ch = FileChannel(path)
                assert ch.confirm("Proceed?") is True

    def test_confirm_parses_no_variants(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_answers(td, [{"prompt": "Proceed", "answer": "n"}])
            ch = FileChannel(path)
            assert ch.confirm("Proceed with changes?") is False

    def test_notes_returns_recorded_text(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_answers(td, [
                {"prompt": "Enter hint", "answer": "add retry logic"}
            ])
            ch = FileChannel(path)
            assert ch.notes("[HITL] Enter hint/instruction") == "add retry logic"

    def test_wait_for_manual_edit_returns_immediately(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_answers(td, [])
            ch = FileChannel(path)
            ch.wait_for_manual_edit("/some/file.md")  # must not block

    def test_is_interactive_false(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_answers(td, [])
            ch = FileChannel(path)
            assert ch.is_interactive() is False

    def test_unmatched_prompt_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_answers(td, [
                {"prompt": "REQUIREMENTS", "answer": "a"}
            ])
            ch = FileChannel(path)
            with pytest.raises(RuntimeError, match="No pre-recorded answer"):
                ch.prompt("[HITL:DEPLOYMENT] Select action", ["a", "b"])

    def test_all_entries_consumed_independently(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_answers(td, [
                {"prompt": "gate1", "answer": "a"},
                {"prompt": "gate2", "answer": "b"},
            ])
            ch = FileChannel(path)
            assert ch.prompt("gate1 prompt", ["a", "b"]) == "a"
            assert ch.prompt("gate2 prompt", ["a", "b"]) == "b"


# ---------------------------------------------------------------------------
# get_channel factory
# ---------------------------------------------------------------------------

class TestGetChannel:

    def test_returns_file_channel_when_env_set(self, monkeypatch):
        with tempfile.TemporaryDirectory() as td:
            answers = os.path.join(td, "a.json")
            with open(answers, "w") as f:
                json.dump([], f)
            monkeypatch.setenv("HARNESS_HITL_FILE", answers)
            ch = get_channel()
            assert isinstance(ch, FileChannel)

    def test_returns_stdin_channel_by_default(self):
        ch = get_channel()
        assert isinstance(ch, StdinChannel)

    def test_set_channel_overrides_factory(self, monkeypatch):
        with tempfile.TemporaryDirectory() as td:
            answers = os.path.join(td, "b.json")
            with open(answers, "w") as f:
                json.dump([], f)
            custom = FileChannel(answers)
            set_channel(custom)
            assert get_channel() is custom

    def test_reset_channel_clears_singleton(self):
        ch1 = get_channel()
        reset_channel()
        ch2 = get_channel()
        # After reset, a new instance is created
        assert ch1 is not ch2

    def test_file_channel_fails_closed_on_missing_answers(self, monkeypatch):
        """FileChannel with no matching entry raises — not silently skip."""
        with tempfile.TemporaryDirectory() as td:
            answers = os.path.join(td, "c.json")
            with open(answers, "w") as f:
                json.dump([], f)
            monkeypatch.setenv("HARNESS_HITL_FILE", answers)
            ch = get_channel()
            with pytest.raises(RuntimeError, match="No pre-recorded answer"):
                ch.confirm("Are you sure?")

    def test_webhook_env_selects_http_channel(self, monkeypatch):
        monkeypatch.setenv("HARNESS_HITL_WEBHOOK_URL", "http://127.0.0.1:19999/hitl")
        ch = get_channel()
        assert isinstance(ch, HttpChannel)

    def test_webhook_takes_priority_over_file(self, monkeypatch):
        monkeypatch.setenv("HARNESS_HITL_WEBHOOK_URL", "http://127.0.0.1:19999/hitl")
        monkeypatch.setenv("HARNESS_HITL_FILE", "/some/file.json")
        ch = get_channel()
        assert isinstance(ch, HttpChannel)


# ---------------------------------------------------------------------------
# HttpChannel
# ---------------------------------------------------------------------------

class TestHttpChannel:

    def _serve(self, response_body=b'{"answer": "a"}', status=200):
        captured: list = []
        server = _make_server(response_body, status, captured)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        port = server.server_address[1]
        url = f"http://127.0.0.1:{port}"
        return server, url, captured

    def test_prompt_sends_correct_payload_and_returns_answer(self):
        server, url, captured = self._serve(b'{"answer": "b"}')
        try:
            ch = HttpChannel(url)
            result = ch.prompt("Choose option", ["a", "b", "c"], default="a")
            assert result == "b"
            assert len(captured) == 1
            assert captured[0]["body"]["type"] == "prompt"
            assert captured[0]["body"]["message"] == "Choose option"
            assert captured[0]["body"]["options"] == ["a", "b", "c"]
            # option_labels is omitted from the payload when the caller
            # didn't pass any — older webhook receivers must not see an
            # unexpected field. New receivers will see it when supplied
            # (covered by the next test).
            assert "option_labels" not in captured[0]["body"]
        finally:
            server.shutdown()

    def test_prompt_forwards_option_labels_when_supplied(self):
        """When the caller passes option_labels, HttpChannel forwards
        them in the webhook body so a UI on the other end can render a
        labeled dropdown instead of a single-letter input. cli.py's
        hitl_menu_loop passes the [v]/[r]/[e]/... menu this way."""
        server, url, captured = self._serve(b'{"answer": "r"}')
        try:
            ch = HttpChannel(url)
            labels = {
                "v": "View diffs",
                "r": "Resume graph execution",
                "q": "Abandon session",
            }
            result = ch.prompt(
                "Select action",
                ["v", "r", "q"],
                default="r",
                option_labels=labels,
            )
            assert result == "r"
            body = captured[0]["body"]
            assert body["option_labels"] == labels
            assert body["default"] == "r"
        finally:
            server.shutdown()

    def test_confirm_true_on_yes_answer(self):
        server, url, _ = self._serve(b'{"answer": "yes"}')
        try:
            ch = HttpChannel(url)
            assert ch.confirm("Are you sure?") is True
        finally:
            server.shutdown()

    def test_confirm_false_on_no_answer(self):
        server, url, _ = self._serve(b'{"answer": "no"}')
        try:
            ch = HttpChannel(url)
            assert ch.confirm("Are you sure?") is False
        finally:
            server.shutdown()

    def test_notes_returns_text(self):
        server, url, _ = self._serve(b'{"answer": "retry with smaller batches"}')
        try:
            ch = HttpChannel(url)
            result = ch.notes("Enter hint")
            assert result == "retry with smaller batches"
        finally:
            server.shutdown()

    def test_wait_for_edit_unblocks_on_200(self):
        server, url, _ = self._serve(b'{"answer": "done"}')
        try:
            ch = HttpChannel(url)
            ch.wait_for_manual_edit("/workspace/foo.py")  # must return
        finally:
            server.shutdown()

    def test_request_body_type_for_wait_is_wait_for_edit(self):
        server, url, captured = self._serve(b'{"answer": "ok"}')
        try:
            ch = HttpChannel(url)
            ch.wait_for_manual_edit("/workspace/main.py")
            assert captured[0]["body"]["type"] == "wait_for_edit"
        finally:
            server.shutdown()

    def test_hmac_signature_header_sent_when_secret_set(self):
        server, url, captured = self._serve(b'{"answer": "a"}')
        secret = "my-secret-key"
        try:
            ch = HttpChannel(url, secret=secret)
            ch.prompt("Choose", ["a"], default="a")
            assert "X-Harness-Signature" in captured[0]["headers"]
            sig = captured[0]["headers"]["X-Harness-Signature"]
            assert sig.startswith("sha256=")
            # Verify the HMAC is correct
            body_bytes = json.dumps({
                "type": "prompt",
                "message": "Choose",
                "options": ["a"],
                "default": "a",
            }, ensure_ascii=False).encode("utf-8")
            expected = "sha256=" + hmac.new(
                secret.encode(), body_bytes, hashlib.sha256
            ).hexdigest()
            assert sig == expected
        finally:
            server.shutdown()

    def test_no_signature_header_without_secret(self):
        server, url, captured = self._serve(b'{"answer": "a"}')
        try:
            ch = HttpChannel(url, secret=None)
            ch.prompt("Choose", ["a"], default="a")
            assert "X-Harness-Signature" not in captured[0]["headers"]
        finally:
            server.shutdown()

    def test_falls_back_to_default_on_server_error(self):
        # 500 response → fallback to default, no exception raised
        server, url, _ = self._serve(b'{"error": "boom"}', status=500)
        try:
            ch = HttpChannel(url, max_retries=0)
            result = ch.prompt("Choose", ["a", "b"], default="a")
            assert result == "a"
        finally:
            server.shutdown()

    def test_falls_back_to_default_on_bad_json(self):
        server, url, _ = self._serve(b"not json at all", status=200)
        try:
            ch = HttpChannel(url, max_retries=0)
            result = ch.confirm("Proceed?", default=False)
            assert result is False  # default, not an exception
        finally:
            server.shutdown()

    def test_is_interactive_true(self):
        ch = HttpChannel("http://127.0.0.1:9999")
        assert ch.is_interactive() is True

    def test_connection_timeout_fallback(self):
        """If the webhook server is unreachable, should fallback to default."""
        ch = HttpChannel("http://127.0.0.1:1", max_retries=0)  # invalid port
        # Should not raise, should return default
        result = ch.prompt("Choose", ["a", "b"], default="a")
        assert result == "a"

    def test_non_200_status_with_valid_json(self):
        """Non-200 status should fallback even if JSON is valid."""
        server, url, _ = self._serve(b'{"answer": "should_ignore"}', status=400)
        try:
            ch = HttpChannel(url, max_retries=0)
            result = ch.prompt("Choose", ["a", "b"], default="b")
            assert result == "b"  # falls back to default, ignores answer
        finally:
            server.shutdown()


class TestFileChannelEdgeCases:
    """Additional coverage for FileChannel edge cases."""

    def test_file_with_multiple_answers_consumed_sequentially(self):
        """FileChannel should consume answers sequentially from file."""
        with tempfile.TemporaryDirectory() as td:
            response_file = os.path.join(td, "answers.json")
            with open(response_file, "w") as f:
                json.dump([
                    {"prompt": "first", "answer": "a"},
                    {"prompt": "second", "answer": "b"},
                    {"prompt": "third", "answer": "c"},
                ], f)
            ch = FileChannel(response_file)
            assert ch.notes("first prompt") == "a"
            assert ch.notes("second prompt") == "b"
            assert ch.notes("third prompt") == "c"

    def test_file_channel_raises_on_unmatched_prompt_second_time(self):
        """FileChannel should raise when same prompt is re-used (not pre-recorded twice)."""
        with tempfile.TemporaryDirectory() as td:
            response_file = os.path.join(td, "answers.json")
            with open(response_file, "w") as f:
                json.dump([
                    {"prompt": "choose", "answer": "a"},
                ], f)
            ch = FileChannel(response_file)
            assert ch.prompt("choose action", ["a", "b"]) == "a"
            # Second call to same prompt should fail (not pre-recorded twice)
            with pytest.raises(RuntimeError, match="No pre-recorded answer"):
                ch.prompt("choose again", ["a", "b"])
