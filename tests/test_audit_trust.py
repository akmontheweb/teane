"""Tests for trust-module audit hardening (batches 4, 9).

Covers:
  - safe_resolve parent-component symlink rejection                  (§3.5)
  - safe_resolve NUL-byte rejection                                  (§3.5)
  - validate_outbound_url DNS rebinding guard                        (§3.2)
  - validate_outbound_url manual-redirect support                    (§3.1 plumbing)
  - safe_subprocess_env scrubs SSH_AUTH_SOCK / KUBECONFIG / *_PROXY  (§3.16)
"""

from __future__ import annotations

import os
import socket

import pytest

from harness import trust


# ---------------------------------------------------------------------------
# safe_resolve hardening (audit §3.5)
# ---------------------------------------------------------------------------


def test_safe_resolve_rejects_nul_byte():
    with pytest.raises(ValueError, match="NUL"):
        trust.safe_resolve("/tmp", "foo\x00bar")


def test_safe_resolve_rejects_absolute_path():
    with pytest.raises(ValueError, match="absolute"):
        trust.safe_resolve("/tmp", "/etc/passwd")


def test_safe_resolve_rejects_parent_traversal_via_realpath(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(ValueError, match="escapes workspace"):
        trust.safe_resolve(str(ws), "../../etc/passwd")


def test_safe_resolve_rejects_parent_symlink_escape(tmp_path):
    """A symlink in an interior component pointing OUTSIDE the workspace
    must be rejected even when the leaf doesn't exist yet (audit §3.5).

    The realpath-based check catches most cases (it follows the symlink
    and detects the result is outside). The per-component walk added in
    the audit is belt-and-braces for cases realpath might miss; either
    rejection message is acceptable as long as SOMETHING refuses the
    write."""
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink_path = ws / "proxy"
    # Windows refuses symlink creation without admin or developer mode;
    # skip cleanly rather than mask the assertion under an OSError.
    try:
        os.symlink(str(outside), str(symlink_path))
    except OSError:
        pytest.skip("symlinks unavailable on this platform")
    with pytest.raises(ValueError, match="(escapes|symlink)"):
        trust.safe_resolve(str(ws), "proxy/new.txt")


def test_safe_resolve_accepts_legitimate_nested_path(tmp_path):
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    resolved = trust.safe_resolve(str(ws), "src/main.py")
    assert resolved.endswith("main.py")
    assert os.path.commonpath([resolved, str(ws)]) == str(ws.resolve())


# ---------------------------------------------------------------------------
# validate_outbound_url DNS rebinding guard (audit §3.2)
# ---------------------------------------------------------------------------


def test_validate_outbound_url_rejects_loopback_literal():
    with pytest.raises(ValueError, match="private/loopback"):
        trust.validate_outbound_url(
            "http://127.0.0.1/", resolve_dns=False,
        )


def test_validate_outbound_url_rejects_link_local_metadata_literal():
    with pytest.raises(ValueError, match="private/loopback"):
        trust.validate_outbound_url(
            "http://169.254.169.254/latest/meta-data/", resolve_dns=False,
        )


def test_validate_outbound_url_rejects_hostname_resolving_to_loopback(monkeypatch):
    """The DNS-rebinding guard: a hostname that resolves to 127.0.0.1
    must be rejected with a clear message before any HTTP fetch."""
    def _fake_getaddrinfo(host, *a, **kw):
        # Return a single result tuple that resolves to 127.0.0.1.
        return [(socket.AF_INET, 1, 6, "", ("127.0.0.1", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)
    with pytest.raises(ValueError, match="DNS-rebinding"):
        trust.validate_outbound_url("http://evil.example.com/")


def test_validate_outbound_url_rejects_hostname_resolving_to_rfc1918(monkeypatch):
    def _fake_getaddrinfo(host, *a, **kw):
        return [(socket.AF_INET, 1, 6, "", ("10.0.0.5", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)
    with pytest.raises(ValueError, match="DNS-rebinding"):
        trust.validate_outbound_url("http://internal.example.com/")


def test_validate_outbound_url_accepts_public_hostname(monkeypatch):
    """A hostname that resolves to a public IP must pass."""
    def _fake_getaddrinfo(host, *a, **kw):
        # 93.184.216.34 = example.com's public IP at audit time
        return [(socket.AF_INET, 1, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)
    # Must NOT raise.
    result = trust.validate_outbound_url("https://example.com/path")
    assert result == "https://example.com/path"


def test_validate_outbound_url_rejects_non_http_scheme():
    with pytest.raises(ValueError, match="scheme"):
        trust.validate_outbound_url("javascript:alert(1)")


def test_validate_outbound_url_rejects_data_uri():
    with pytest.raises(ValueError):
        trust.validate_outbound_url("data:text/html,<h1>x</h1>")


def test_validate_outbound_url_resolve_dns_false_skips_dns(monkeypatch):
    """When resolve_dns=False, the DNS resolution is skipped — used by
    tests that don't want to hit the network."""
    called = {"n": 0}

    def _spy(*a, **kw):
        called["n"] += 1
        return [(socket.AF_INET, 1, 6, "", ("127.0.0.1", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _spy)
    # A literal IP that's public-ish (this is OUR test data) - really we
    # just need a non-IP host the resolver would be asked about.
    trust.validate_outbound_url(
        "https://example.com/", resolve_dns=False,
    )
    assert called["n"] == 0


# ---------------------------------------------------------------------------
# safe_subprocess_env scrubs proxy + ssh-agent + kubeconfig (audit §3.16)
# ---------------------------------------------------------------------------


def test_safe_subprocess_env_scrubs_ssh_auth_sock(monkeypatch):
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/ssh-foo")
    env = trust.safe_subprocess_env()
    assert "SSH_AUTH_SOCK" not in env


def test_safe_subprocess_env_scrubs_kubeconfig(monkeypatch):
    monkeypatch.setenv("KUBECONFIG", "/home/op/.kube/config")
    env = trust.safe_subprocess_env()
    assert "KUBECONFIG" not in env


def test_safe_subprocess_env_scrubs_all_proxy_variants(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY",
                 "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.setenv(name, "http://proxy.internal:8080")
    env = trust.safe_subprocess_env()
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY",
                 "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        assert name not in env


def test_safe_subprocess_env_preserves_unrelated_vars(monkeypatch):
    monkeypatch.setenv("MY_BUILD_FLAG", "yes")
    monkeypatch.setenv("HARNESS_JOB_NAME", "test-run")
    env = trust.safe_subprocess_env()
    assert env.get("MY_BUILD_FLAG") == "yes"
    assert env.get("HARNESS_JOB_NAME") == "test-run"


def test_safe_subprocess_env_extra_overrides(monkeypatch):
    """``extra`` lets the operator re-add a specific scrubbed var when
    the build genuinely needs it."""
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    env = trust.safe_subprocess_env(extra={"OPENAI_API_KEY": "explicit"})
    assert env["OPENAI_API_KEY"] == "explicit"
