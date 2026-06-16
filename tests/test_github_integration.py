"""Regression tests for the GitHub integration (#4).

These tests stub ``shutil.which`` and ``subprocess.run`` so we don't
require the ``gh`` CLI to be installed. The shape of the gh JSON
contract is fixed enough that snapshot-style mocking is appropriate.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

import pytest

import harness.github_integration as gh_module
from harness.github_integration import (
    GhAuthStatus,
    GithubIssue,
    create_pr,
    fetch_issue,
    gh_auth_status,
    gh_available,
    ingest_issue_to_change_request,
    post_pr_comment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_completed(
    returncode: int = 0, stdout: str = "", stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr,
    )


def _stub_which_present(monkeypatch, path: str = "/usr/local/bin/gh") -> None:
    monkeypatch.setattr(gh_module.shutil, "which", lambda _name: path)


def _stub_which_absent(monkeypatch) -> None:
    monkeypatch.setattr(gh_module.shutil, "which", lambda _name: None)


# ---------------------------------------------------------------------------
# 1. Availability
# ---------------------------------------------------------------------------

def test_gh_available_true_when_on_path(monkeypatch):
    _stub_which_present(monkeypatch)
    assert gh_available() is True


def test_gh_available_false_when_missing(monkeypatch):
    _stub_which_absent(monkeypatch)
    assert gh_available() is False


def test_gh_auth_status_passes_through_returncode(monkeypatch):
    _stub_which_present(monkeypatch)
    captured: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        captured["cmd"] = cmd
        return _fake_completed(
            returncode=0,
            stdout="github.com\n  ✓ Logged in to github.com as akmontheweb (oauth_token)",
            stderr="",
        )

    monkeypatch.setattr(gh_module.subprocess, "run", fake_run)
    status = gh_auth_status()
    assert isinstance(status, GhAuthStatus)
    assert status.ok is True
    assert "Logged in" in status.detail
    assert captured["cmd"][1:3] == ["auth", "status"]


def test_gh_auth_status_when_missing(monkeypatch):
    _stub_which_absent(monkeypatch)
    status = gh_auth_status()
    assert status.ok is False
    assert "gh not installed" in status.detail


# ---------------------------------------------------------------------------
# 2. fetch_issue
# ---------------------------------------------------------------------------

def test_fetch_issue_parses_json(monkeypatch):
    _stub_which_present(monkeypatch)
    payload = {
        "number": 42,
        "title": "[bug] login fails when password contains emoji",
        "body": "When the password contains an emoji, the form rejects it.",
        "labels": [{"name": "bug"}, {"name": "auth"}],
        "state": "OPEN",
        "author": {"login": "alice"},
        "url": "https://github.com/octo/octo-app/issues/42",
    }

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        # sanity-check the gh args we send
        assert cmd[1:5] == ["issue", "view", "42", "--repo"]
        assert "--json" in cmd
        return _fake_completed(stdout=json.dumps(payload))

    monkeypatch.setattr(gh_module.subprocess, "run", fake_run)
    issue = fetch_issue("octo/octo-app", 42)
    assert isinstance(issue, GithubIssue)
    assert issue.number == 42
    assert issue.title.startswith("[bug]")
    assert "emoji" in issue.body
    assert "bug" in issue.labels and "auth" in issue.labels
    assert issue.state == "OPEN"
    assert issue.author == "alice"
    assert issue.url.endswith("/42")


def test_fetch_issue_raises_when_gh_missing(monkeypatch):
    _stub_which_absent(monkeypatch)
    with pytest.raises(RuntimeError, match="gh CLI"):
        fetch_issue("x/y", 1)


def test_fetch_issue_raises_on_nonzero(monkeypatch):
    _stub_which_present(monkeypatch)
    monkeypatch.setattr(
        gh_module.subprocess, "run",
        lambda *a, **kw: _fake_completed(returncode=1, stderr="not found"),
    )
    with pytest.raises(RuntimeError, match="gh issue view exit=1"):
        fetch_issue("x/y", 99999)


# ---------------------------------------------------------------------------
# 3. to_change_request_text
# ---------------------------------------------------------------------------

def test_issue_renders_change_request_text():
    issue = GithubIssue(
        number=7, title="Add OIDC login", body="We need an OIDC provider...",
        labels=["enhancement"], state="OPEN", author="alice",
        url="https://github.com/x/y/issues/7",
    )
    rendered = issue.to_change_request_text("x/y")
    assert "# Add OIDC login" in rendered
    assert "OIDC provider" in rendered
    assert "https://github.com/x/y/issues/7" in rendered
    assert "alice" in rendered
    assert "enhancement" in rendered


def test_issue_handles_empty_body():
    issue = GithubIssue(
        number=1, title="t", body="", labels=[], state="CLOSED", author="bob", url="u",
    )
    rendered = issue.to_change_request_text("x/y")
    assert "(no body provided)" in rendered
    assert "Labels: (none)" in rendered


# ---------------------------------------------------------------------------
# 4. ingest_issue_to_change_request
# ---------------------------------------------------------------------------

def test_ingest_writes_cr_file_with_next_number(monkeypatch, tmp_path):
    _stub_which_present(monkeypatch)
    payload = {
        "number": 11, "title": "Fix login bug", "body": "details",
        "labels": [{"name": "bug"}], "state": "OPEN",
        "author": {"login": "alice"},
        "url": "https://github.com/x/y/issues/11",
    }
    monkeypatch.setattr(
        gh_module.subprocess, "run",
        lambda *a, **kw: _fake_completed(stdout=json.dumps(payload)),
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # Pre-existing CR — next ingest should land at CR-2
    cr_dir = workspace / "change_requests"
    cr_dir.mkdir()
    (cr_dir / "CR-1-prior.txt").write_text("old change request")
    path = ingest_issue_to_change_request(str(workspace), "x/y", 11)
    assert os.path.basename(path).startswith("CR-2-")
    assert "fix-login-bug" in path
    content = open(path, encoding="utf-8").read()
    assert "# Fix login bug" in content
    assert "issues/11" in content


def test_ingest_creates_change_requests_dir_when_absent(monkeypatch, tmp_path):
    _stub_which_present(monkeypatch)
    payload = {
        "number": 1, "title": "First", "body": "", "labels": [],
        "state": "OPEN", "author": {"login": "x"},
        "url": "https://github.com/x/y/issues/1",
    }
    monkeypatch.setattr(
        gh_module.subprocess, "run",
        lambda *a, **kw: _fake_completed(stdout=json.dumps(payload)),
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    path = ingest_issue_to_change_request(str(workspace), "x/y", 1)
    assert os.path.basename(path).startswith("CR-1-")
    assert os.path.isdir(workspace / "change_requests")


# ---------------------------------------------------------------------------
# 5. create_pr
# ---------------------------------------------------------------------------

def test_create_pr_parses_url(monkeypatch, tmp_path):
    _stub_which_present(monkeypatch)
    captured: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        return _fake_completed(stdout="https://github.com/x/y/pull/57\n")

    monkeypatch.setattr(gh_module.subprocess, "run", fake_run)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    pr = create_pr(str(workspace), title="Fix bug", body="details", base="main")
    assert pr.url == "https://github.com/x/y/pull/57"
    assert pr.number == 57
    assert captured["cwd"] == str(workspace)
    assert "--title" in captured["cmd"]
    assert "Fix bug" in captured["cmd"]


def test_create_pr_rejects_empty_title(monkeypatch):
    _stub_which_present(monkeypatch)
    with pytest.raises(ValueError):
        create_pr("/tmp", title="", body="x")


def test_create_pr_raises_on_gh_failure(monkeypatch, tmp_path):
    _stub_which_present(monkeypatch)
    monkeypatch.setattr(
        gh_module.subprocess, "run",
        lambda *a, **kw: _fake_completed(returncode=1, stderr="no diff"),
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    with pytest.raises(RuntimeError, match="gh pr create exit=1"):
        create_pr(str(workspace), title="t", body="b")


# ---------------------------------------------------------------------------
# 6. post_pr_comment
# ---------------------------------------------------------------------------

def test_post_pr_comment_sends_body(monkeypatch):
    _stub_which_present(monkeypatch)
    captured: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        captured["cmd"] = cmd
        return _fake_completed(returncode=0)

    monkeypatch.setattr(gh_module.subprocess, "run", fake_run)
    post_pr_comment("x/y", 42, "Looks good!")
    assert "pr" in captured["cmd"] and "comment" in captured["cmd"]
    assert "42" in captured["cmd"]
    assert "Looks good!" in captured["cmd"]


def test_post_pr_comment_rejects_empty_body(monkeypatch):
    _stub_which_present(monkeypatch)
    with pytest.raises(ValueError):
        post_pr_comment("x/y", 42, "")


# ---------------------------------------------------------------------------
# default_repo() — configure-page GitHub fields (default_owner/default_repo)
# ---------------------------------------------------------------------------

def test_default_repo_returns_owner_slash_name_when_both_set():
    from harness.github_integration import default_repo
    cfg = {"github": {"default_owner": "acme", "default_repo": "harness"}}
    assert default_repo(cfg) == "acme/harness"


def test_default_repo_returns_none_when_either_missing():
    from harness.github_integration import default_repo
    assert default_repo({"github": {"default_owner": "acme"}}) is None
    assert default_repo({"github": {"default_repo": "harness"}}) is None
    assert default_repo({"github": {}}) is None
    assert default_repo(None) is None


def test_default_repo_strips_whitespace():
    from harness.github_integration import default_repo
    cfg = {"github": {"default_owner": "  acme  ", "default_repo": " harness "}}
    assert default_repo(cfg) == "acme/harness"
