"""Tests for harness/cli.py — `teane doctor` first-run healthcheck."""

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path

import pytest


from harness.cli import (
    _doctor_check_api_keys,
    _doctor_check_checkpoint_db,
    _doctor_check_config,
    _doctor_check_git,
    _doctor_check_patcher_mode,
    _doctor_check_sandbox,
    _format_doctor_line,
)


class TestDoctorPatcherMode:
    def test_default_mode_reports_b5_on_b6_on(self):
        # Both flags default ON since native tool-use became the default
        # dispatch path (patcher.use_structured_tools=false opts back
        # into the legacy text DSL).
        status, detail = _doctor_check_patcher_mode({})
        assert status == "pass"
        assert "read-before-edit ON" in detail
        assert "native tool-use ON" in detail

    def test_b6_opt_out_reports_text_dsl(self):
        status, detail = _doctor_check_patcher_mode({
            "patcher": {"use_structured_tools": False},
        })
        assert status == "pass"
        assert "native tool-use OFF" in detail
        assert "text DSL active" in detail

    def test_b5_off_b6_on_surfaces_both(self):
        status, detail = _doctor_check_patcher_mode({
            "patcher": {
                "enforce_read_before_edit": False,
                "use_structured_tools": True,
            },
        })
        assert status == "pass"
        assert "read-before-edit OFF" in detail
        assert "native tool-use ON" in detail


@pytest.fixture(autouse=True)
def _skip_live_ping(monkeypatch):
    """Default: every doctor api-keys test skips the live HTTP ping.

    The live ping was added after the original tests landed; without
    this fixture every key-resolution assertion would also need to mock
    httpx. The handful of tests that exercise the live-ping path opt
    back in by clearing the env var explicitly and patching httpx.
    """
    monkeypatch.setenv("HARNESS_DOCTOR_SKIP_LIVE", "true")


def _run_api_keys_check(config):
    """Sync wrapper — the check is async, but most tests assert on the
    returned (status, detail) tuple, not concurrency."""
    return asyncio.run(_doctor_check_api_keys(config))


class TestDoctorGitCheck:
    @staticmethod
    def _init_repo_with_commit(tmpdir: str) -> None:
        """git init + one empty commit so HEAD resolves. Uses inline identity
        so the test doesn't depend on the runner's global git config."""
        subprocess.run(["git", "init", "-q", tmpdir], check=True)
        subprocess.run(
            ["git", "-C", tmpdir,
             "-c", "user.email=test@example.com", "-c", "user.name=Test",
             "commit", "--allow-empty", "-q", "-m", "init"],
            check=True,
        )

    def test_passes_in_a_git_repo_with_a_commit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._init_repo_with_commit(tmpdir)
            status, _detail = _doctor_check_git(tmpdir)
            assert status == "pass"

    def test_warns_on_unborn_head(self):
        # Regression: a freshly `git init`'d repo with zero commits has an
        # unborn HEAD, which silently breaks speculative repair. Doctor must
        # warn instead of pretending all is well.
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(["git", "init", "-q", tmpdir], check=True)
            status, detail = _doctor_check_git(tmpdir)
            assert status == "warn"
            assert "no commits" in detail or "unborn" in detail.lower()

    def test_fails_outside_a_git_repo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            status, detail = _doctor_check_git(tmpdir)
            assert status == "fail"
            assert "not a git repo" in detail


class TestDoctorApiKeysCheck:
    def test_warns_when_no_routing_models_configured(self):
        config = {"model_routing": {}}
        status, detail = _run_api_keys_check(config)
        assert status == "warn"
        assert "no non-ollama models" in detail

    def test_warns_when_only_ollama_configured(self):
        config = {"model_routing": {"planning_primary": "ollama:llama3"}}
        status, detail = _run_api_keys_check(config)
        assert status == "warn"

    def test_fails_when_provider_key_missing(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = {"model_routing": {"planning_primary": "openai:gpt-4o"}}
        status, detail = _run_api_keys_check(config)
        assert status == "fail"
        assert "OPENAI_API_KEY" in detail

    def test_passes_when_all_keys_present(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config = {
            "model_routing": {
                "planning_primary": "anthropic:claude-opus-4-5",
                "patching_primary": "deepseek:deepseek-coder",
            },
        }
        status, detail = _run_api_keys_check(config)
        assert status == "pass"
        # New format reports model_key and source — env vs config — so
        # the operator can spot which keys came from where at a glance.
        assert "anthropic:claude-opus-4-5 (env)" in detail
        assert "deepseek:deepseek-coder (env)" in detail

    def test_passes_when_key_only_in_config_field(self, monkeypatch):
        """Regression for the doctor-vs-runtime mismatch: gateway.py:331
        accepts a key from ``models[<key>].api_key`` when the env var is
        empty, but the doctor used to only probe env. Now it matches."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = {
            "model_routing": {"planning_primary": "openai:gpt-4o-mini"},
            "models": {"openai:gpt-4o-mini": {"api_key": "sk-from-config"}},
        }
        status, detail = _run_api_keys_check(config)
        assert status == "pass"
        assert "openai:gpt-4o-mini (config)" in detail

    def test_env_takes_precedence_over_config(self, monkeypatch):
        """Mirrors gateway.py:331 — env wins. Doctor must report the same
        source the runtime would actually use so it doesn't lie about
        which key is in play."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        config = {
            "model_routing": {"planning_primary": "openai:gpt-4o-mini"},
            "models": {"openai:gpt-4o-mini": {"api_key": "sk-from-config"}},
        }
        status, detail = _run_api_keys_check(config)
        assert status == "pass"
        assert "openai:gpt-4o-mini (env)" in detail
        assert "(config)" not in detail

    def test_fails_when_neither_env_nor_config_have_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = {
            "model_routing": {"planning_primary": "openai:gpt-4o-mini"},
            "models": {"openai:gpt-4o-mini": {}},  # explicit empty entry
        }
        status, detail = _run_api_keys_check(config)
        assert status == "fail"
        # The recovery path must show both options so the operator
        # doesn't think env var is the only place to set it.
        assert "OPENAI_API_KEY" in detail
        assert "models.\"openai:gpt-4o-mini\".api_key" in detail

    def test_config_field_with_whitespace_only_treated_as_missing(self, monkeypatch):
        """An accidental ``"api_key": "   "`` shouldn't count as set —
        the runtime would treat it the same way (empty after strip)."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = {
            "model_routing": {"planning_primary": "openai:gpt-4o-mini"},
            "models": {"openai:gpt-4o-mini": {"api_key": "   "}},
        }
        status, _ = _run_api_keys_check(config)
        assert status == "fail"


class _FakeResponse:
    """Minimal stand-in for httpx.Response used by the live-ping tests."""
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text


class _FakeAsyncClient:
    """In-place stand-in for httpx.AsyncClient. ``script`` is a callable
    that maps (url, headers, json_body) → (status_code, text). Lets a
    single test stub multiple provider endpoints with different responses.
    """
    def __init__(self, *_args, **_kwargs):
        pass
    async def __aenter__(self):
        return self
    async def __aexit__(self, *_exc):
        return False

    # Replaced by individual tests via monkeypatch on the class.
    async def post(self, url, *, headers=None, json=None):
        raise NotImplementedError


def _install_fake_async_client(monkeypatch, post_handler):
    """Patch httpx.AsyncClient so doctor's live ping calls ``post_handler``.
    ``post_handler(url, headers, body)`` returns ``_FakeResponse``."""
    import httpx

    class _Client(_FakeAsyncClient):
        async def post(self, url, *, headers=None, json=None):
            return post_handler(url, headers or {}, json or {})

    monkeypatch.setattr(httpx, "AsyncClient", _Client)


class TestDoctorApiKeysLivePing:
    """Live ping: with HARNESS_DOCTOR_SKIP_LIVE unset, the doctor makes
    a 1-token chat call per provider to confirm the key actually
    authenticates against the model. These tests opt out of the
    autouse skip fixture and patch httpx.AsyncClient.
    """

    def test_live_ping_pass_returns_pass(self, monkeypatch):
        monkeypatch.delenv("HARNESS_DOCTOR_SKIP_LIVE", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real")

        def handler(url, headers, body):
            assert "api.openai.com" in url
            assert headers["Authorization"] == "Bearer sk-real"
            assert body["max_tokens"] == 1
            return _FakeResponse(200, '{"id":"chatcmpl-x"}')

        _install_fake_async_client(monkeypatch, handler)
        config = {"model_routing": {"planning_primary": "openai:gpt-4o-mini"}}
        status, detail = _run_api_keys_check(config)
        assert status == "pass"
        assert "live" in detail
        assert "openai:gpt-4o-mini" in detail

    def test_live_ping_401_returns_fail_with_clear_message(self, monkeypatch):
        monkeypatch.delenv("HARNESS_DOCTOR_SKIP_LIVE", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-bad")

        def handler(_url, _headers, _body):
            return _FakeResponse(401, '{"error":"invalid_api_key"}')

        _install_fake_async_client(monkeypatch, handler)
        config = {"model_routing": {"planning_primary": "openai:gpt-4o-mini"}}
        status, detail = _run_api_keys_check(config)
        assert status == "fail"
        assert "401" in detail
        assert "API key rejected" in detail or "rejected" in detail

    def test_live_ping_429_distinguished_from_auth_failure(self, monkeypatch):
        monkeypatch.delenv("HARNESS_DOCTOR_SKIP_LIVE", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-real")

        def handler(_url, _headers, _body):
            return _FakeResponse(429, '{"error":"rate_limit"}')

        _install_fake_async_client(monkeypatch, handler)
        config = {"model_routing": {"planning_primary": "deepseek:deepseek-v4-pro"}}
        status, detail = _run_api_keys_check(config)
        assert status == "fail"
        assert "429" in detail
        assert "rate" in detail.lower() or "quota" in detail.lower()

    def test_live_ping_connect_error_fails_with_network_hint(self, monkeypatch):
        import httpx
        monkeypatch.delenv("HARNESS_DOCTOR_SKIP_LIVE", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real")

        class _ExplodingClient(_FakeAsyncClient):
            async def post(self, *_a, **_kw):
                raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(httpx, "AsyncClient", _ExplodingClient)
        config = {"model_routing": {"planning_primary": "anthropic:claude-sonnet-4"}}
        status, detail = _run_api_keys_check(config)
        assert status == "fail"
        assert "connection failed" in detail.lower() or "connect" in detail.lower()

    def test_live_ping_timeout_fails_with_timeout_hint(self, monkeypatch):
        import httpx
        monkeypatch.delenv("HARNESS_DOCTOR_SKIP_LIVE", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real")

        class _TimeoutClient(_FakeAsyncClient):
            async def post(self, *_a, **_kw):
                raise httpx.ConnectTimeout("timed out")

        monkeypatch.setattr(httpx, "AsyncClient", _TimeoutClient)
        config = {"model_routing": {"planning_primary": "openai:gpt-4o-mini"}}
        status, detail = _run_api_keys_check(config)
        assert status == "fail"
        assert "timeout" in detail.lower() or "unreachable" in detail.lower()

    def test_live_ping_uses_correct_anthropic_headers(self, monkeypatch):
        """Anthropic uses x-api-key + anthropic-version, NOT Bearer.
        If the doctor sends the wrong header the live ping reports 401
        against a real provider — same diagnostic failure as a bad key.
        """
        monkeypatch.delenv("HARNESS_DOCTOR_SKIP_LIVE", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real")
        captured: dict = {}

        def handler(url, headers, body):
            captured["url"] = url
            captured["headers"] = headers
            captured["body"] = body
            return _FakeResponse(200, '{"id":"msg_x"}')

        _install_fake_async_client(monkeypatch, handler)
        config = {"model_routing": {"planning_primary": "anthropic:claude-sonnet-4"}}
        status, _detail = _run_api_keys_check(config)
        assert status == "pass"
        assert "api.anthropic.com" in captured["url"]
        assert captured["headers"]["x-api-key"] == "sk-real"
        assert "anthropic-version" in captured["headers"]
        assert "Authorization" not in captured["headers"]
        assert captured["body"]["model"] == "claude-sonnet-4"
        assert captured["body"]["max_tokens"] == 1

    def test_skip_live_env_var_short_circuits(self, monkeypatch):
        """With HARNESS_DOCTOR_SKIP_LIVE=true (the autouse default), no
        network request fires even with a key set. Confirms the opt-out
        works for CI / headless and proves the rest of the suite's
        autouse fixture is doing its job."""
        # The autouse fixture already sets the env var; assert the live
        # ping is NEVER invoked by patching httpx.AsyncClient to explode.
        import httpx

        class _ExplodingClient(_FakeAsyncClient):
            async def post(self, *_a, **_kw):
                raise AssertionError("live ping must not fire when SKIP_LIVE=true")

        monkeypatch.setattr(httpx, "AsyncClient", _ExplodingClient)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
        config = {"model_routing": {"planning_primary": "openai:gpt-4o-mini"}}
        status, detail = _run_api_keys_check(config)
        assert status == "pass"
        assert "live ping skipped" in detail

    def test_live_ping_parallel_across_providers(self, monkeypatch):
        """Two providers configured → two parallel pings. Assertion is
        weak (both must be hit) since asyncio gather scheduling order
        isn't deterministic — just confirm both were dispatched."""
        monkeypatch.delenv("HARNESS_DOCTOR_SKIP_LIVE", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-2")
        urls_hit: list[str] = []

        def handler(url, _headers, _body):
            urls_hit.append(url)
            return _FakeResponse(200, "{}")

        _install_fake_async_client(monkeypatch, handler)
        config = {
            "model_routing": {
                "planning_primary": "openai:gpt-4o-mini",
                "patching_primary": "anthropic:claude-sonnet-4",
            },
        }
        status, _detail = _run_api_keys_check(config)
        assert status == "pass"
        assert any("openai.com" in u for u in urls_hit)
        assert any("anthropic.com" in u for u in urls_hit)


class TestDoctorLivePingOpenAICompat:
    """Live-ping probe coverage for the OpenAI-compatible provider family.

    Regression for the sibling-drift that shipped Moonshot support: the
    gateway's provider registry gained `moonshot` (and already had
    `google`) while `_ping_provider_live` still hardcoded only
    openai/deepseek, so `teane doctor` reported `unknown provider
    'moonshot'` for a model that dispatched fine. The probe now builds
    the URL from the model's configured `api_base_url` so every
    OpenAI-shape provider is reachable and region hosts are honoured.
    """

    def test_moonshot_probes_configured_base_with_bearer_auth(
        self, monkeypatch,
    ):
        monkeypatch.delenv("HARNESS_DOCTOR_SKIP_LIVE", raising=False)
        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-kimi")
        captured: dict = {}

        def handler(url, headers, body):
            captured["url"] = url
            captured["headers"] = headers
            captured["body"] = body
            return _FakeResponse(200, "{}")

        _install_fake_async_client(monkeypatch, handler)
        config = {
            "model_routing": {"planning_primary": "moonshot:kimi-3"},
            "models": {
                "moonshot:kimi-3": {
                    "provider": "moonshot",
                    "model_id": "kimi-3",
                    "api_base_url": "https://api.moonshot.ai/v1",
                },
            },
        }
        status, detail = _run_api_keys_check(config)
        assert status == "pass"
        assert "moonshot:kimi-3" in detail and "live" in detail
        assert captured["url"] == "https://api.moonshot.ai/v1/chat/completions"
        assert captured["headers"]["Authorization"] == "Bearer sk-kimi"
        assert captured["body"]["model"] == "kimi-3"
        assert captured["body"]["max_tokens"] == 1

    def test_moonshot_cn_region_base_is_honoured(self, monkeypatch):
        monkeypatch.delenv("HARNESS_DOCTOR_SKIP_LIVE", raising=False)
        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-kimi")
        captured: dict = {}

        def handler(url, _headers, _body):
            captured["url"] = url
            return _FakeResponse(200, "{}")

        _install_fake_async_client(monkeypatch, handler)
        config = {
            "model_routing": {"planning_primary": "moonshot:kimi-3"},
            "models": {
                "moonshot:kimi-3": {
                    "api_base_url": "https://api.moonshot.cn/v1",
                },
            },
        }
        status, _detail = _run_api_keys_check(config)
        assert status == "pass"
        assert captured["url"] == "https://api.moonshot.cn/v1/chat/completions"

    def test_google_openai_compat_base_is_probed(self, monkeypatch):
        # The pre-existing latent gap the same fix closes: google was also
        # in the gateway registry but not the probe list.
        monkeypatch.delenv("HARNESS_DOCTOR_SKIP_LIVE", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "sk-gg")
        captured: dict = {}

        def handler(url, headers, _body):
            captured["url"] = url
            captured["headers"] = headers
            return _FakeResponse(200, "{}")

        _install_fake_async_client(monkeypatch, handler)
        config = {
            "model_routing": {"planning_primary": "google:gemini-3.5-flash"},
            "models": {
                "google:gemini-3.5-flash": {
                    "api_base_url": (
                        "https://generativelanguage.googleapis.com/"
                        "v1beta/openai/"
                    ),
                },
            },
        }
        status, _detail = _run_api_keys_check(config)
        assert status == "pass"
        assert captured["url"] == (
            "https://generativelanguage.googleapis.com/v1beta/openai/"
            "chat/completions"
        )
        assert captured["headers"]["Authorization"] == "Bearer sk-gg"

    def test_moonshot_without_base_url_fails_with_actionable_message(
        self, monkeypatch,
    ):
        # No hardcoded default host for moonshot (region-specific), so an
        # entry that omits api_base_url must fail loudly rather than
        # probe the wrong host.
        monkeypatch.delenv("HARNESS_DOCTOR_SKIP_LIVE", raising=False)
        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-kimi")

        def handler(_url, _headers, _body):  # pragma: no cover - must not fire
            raise AssertionError("no probe should fire without a base url")

        _install_fake_async_client(monkeypatch, handler)
        config = {
            "model_routing": {"planning_primary": "moonshot:kimi-3"},
            "models": {"moonshot:kimi-3": {"provider": "moonshot"}},
        }
        status, detail = _run_api_keys_check(config)
        assert status == "fail"
        assert "api_base_url" in detail

    def test_openai_still_pings_default_host_without_models_entry(
        self, monkeypatch,
    ):
        # Back-compat: openai/deepseek keep their well-known default host
        # when the config omits an explicit api_base_url.
        monkeypatch.delenv("HARNESS_DOCTOR_SKIP_LIVE", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-o")
        captured: dict = {}

        def handler(url, _headers, _body):
            captured["url"] = url
            return _FakeResponse(200, "{}")

        _install_fake_async_client(monkeypatch, handler)
        config = {"model_routing": {"planning_primary": "openai:gpt-4o-mini"}}
        status, _detail = _run_api_keys_check(config)
        assert status == "pass"
        assert captured["url"] == "https://api.openai.com/v1/chat/completions"


class TestDoctorSandboxCheck:
    def test_bare_backend_warns(self):
        status, detail = _doctor_check_sandbox({"sandbox": {"backend": "bare"}})
        assert status == "warn"
        assert "bare" in detail

    def test_docker_missing_fails(self, monkeypatch):
        # Force shutil.which("docker") to return None inside the check.
        import shutil
        real_which = shutil.which
        monkeypatch.setattr(shutil, "which", lambda name: None if name == "docker" else real_which(name))
        status, detail = _doctor_check_sandbox({"sandbox": {"backend": "docker"}})
        assert status == "fail"
        assert "docker" in detail.lower()


class TestDoctorCheckpointDbCheck:
    def test_writable_db_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "subdir", "ckpt.db")
            status, detail = _doctor_check_checkpoint_db(
                {"persistence": {"db_path": db}}
            )
            assert status == "pass"
            assert db in detail


class TestDoctorConfigCheck:
    def test_clean_canonical_config_passes(self, monkeypatch):
        # Under the single-source contract the doctor's "config" check
        # delegates entirely to discover_config + validate_config_strict.
        # When the canonical config is valid AND every routed model has
        # its env var set, the check passes.
        monkeypatch.setenv("OPENAI_API_KEY", "stub")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "stub")
        with tempfile.TemporaryDirectory() as tmpdir:
            status, detail = _doctor_check_config(tmpdir)
            assert status == "pass", detail

    def test_legacy_workspace_config_is_ignored(self, monkeypatch):
        # Legacy .harness_config.json files in the workspace are NOT
        # parsed; they're logged-and-ignored. Doctor returns pass when
        # the CANONICAL config is valid regardless of what the legacy
        # workspace file contains.
        monkeypatch.setenv("OPENAI_API_KEY", "stub")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "stub")
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy = Path(tmpdir) / ".harness_config.json"
            legacy.write_text('{"token_budget": {"hrad_cap_usd": 1.0}}')
            status, _ = _doctor_check_config(tmpdir)
            assert status == "pass"

    def test_missing_env_var_fails(self, monkeypatch):
        # Strict validation now refuses to load when a model referenced
        # by routing has no API key env var set. Doctor reports a clean
        # fail with the env var name in the message.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            status, detail = _doctor_check_config(tmpdir)
            assert status == "fail"
            assert "API key environment variable" in detail


class TestDoctorLineFormatting:
    def test_line_includes_label_and_detail(self):
        line = _format_doctor_line("pass", "api keys", "all present")
        assert "api keys" in line
        assert "all present" in line

    def test_status_marker_present(self):
        assert "OK" in _format_doctor_line("pass", "x", "y")
        assert "WARN" in _format_doctor_line("warn", "x", "y")
        assert "FAIL" in _format_doctor_line("fail", "x", "y")
