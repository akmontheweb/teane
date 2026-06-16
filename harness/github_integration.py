"""GitHub integration (#4) — shells out to the ``gh`` CLI.

Why ``gh`` and not PyGithub?
============================
``gh`` is already broadly installed on developer + CI hosts, handles
auth (``gh auth login`` + ``GH_TOKEN`` env var) without us touching
credentials, tracks the GitHub API surface for us, and adds zero Python
dependencies. The trade-off is that operators must have ``gh`` on
``PATH``; the doctor and the CLI subcommands surface a clear error when
it's missing.

What v1 covers
==============
- **Issue ingest** — ``fetch_issue(repo, number)`` reads an issue via
  ``gh issue view --json``; ``ingest_issue_to_change_request(workspace,
  repo, number)`` writes the resolved issue into the workspace's
  ``change_requests/`` directory as ``CR-N-<slug>.txt`` so the
  existing change-request flow (PR-1 → PR-3) handles the rest.
- **PR creation** — ``create_pr(workspace, title, body, base)`` shells
  to ``gh pr create`` from the workspace. Returns the PR URL.
- **PR comment** — ``post_pr_comment(repo, pr_number, body)``.

What's deferred
===============
- Issue search / list, label manipulation, reviewer requests.
- Multi-repo cross-references (the ``gh`` CLI handles these directly).
- Posting structured review comments at file / line granularity
  (``gh api`` works; we just don't ship a thin wrapper yet).
- LLM-facing ``GitHubFetchIssueSkill`` — straightforward follow-up
  using the existing ``ToolSkill`` shape from web tools.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


_DEFAULT_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# 1. Availability + auth checks
# ---------------------------------------------------------------------------

def gh_path(config: Optional[dict[str, Any]] = None) -> Optional[str]:
    """Resolve the ``gh`` binary path.

    Honors ``github.gh_path`` from config, then falls back to a
    ``shutil.which("gh")`` lookup. Returns ``None`` when ``gh`` is not
    installed.
    """
    section = ((config or {}).get("github") or {})
    configured = section.get("gh_path")
    if configured and os.path.isfile(configured):
        return str(configured)
    found = shutil.which("gh")
    return found


def gh_available(config: Optional[dict[str, Any]] = None) -> bool:
    return gh_path(config) is not None


def default_repo(config: Optional[dict[str, Any]] = None) -> Optional[str]:
    """Resolve the default ``owner/name`` pair from ``github.default_owner``
    and ``github.default_repo``.

    Returns ``None`` when either side is missing — callers should
    keep requiring an explicit ``repo`` arg in that case. The configure
    page surfaces both fields as plain text inputs; callers that build
    a PR or ingest an issue without a repo override now have a config
    fallback to lean on.
    """
    section = ((config or {}).get("github") or {})
    owner = str(section.get("default_owner") or "").strip()
    name = str(section.get("default_repo") or "").strip()
    if not owner or not name:
        return None
    return f"{owner}/{name}"


@dataclass
class GhAuthStatus:
    ok: bool
    detail: str  # human-readable summary


def gh_auth_status(config: Optional[dict[str, Any]] = None) -> GhAuthStatus:
    """Run ``gh auth status`` and parse the outcome.

    ``gh auth status`` prints to stderr on both success and failure; we
    just need the return code + a short summary line for diagnostics.
    """
    binary = gh_path(config)
    if binary is None:
        return GhAuthStatus(ok=False, detail="gh not installed")
    try:
        result = subprocess.run(
            [binary, "auth", "status"],
            capture_output=True,
            text=True,
            timeout=_DEFAULT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return GhAuthStatus(ok=False, detail=f"gh auth status failed: {exc}")
    combined = (result.stdout + result.stderr).strip().splitlines()
    summary = next(
        (line.strip() for line in combined if "Logged in" in line or "not logged" in line.lower()),
        combined[0] if combined else "(no output)",
    )
    return GhAuthStatus(ok=result.returncode == 0, detail=summary)


# ---------------------------------------------------------------------------
# 2. Issue read
# ---------------------------------------------------------------------------

@dataclass
class GithubIssue:
    number: int
    title: str
    body: str
    labels: list[str]
    state: str
    author: str
    url: str

    def to_change_request_text(self, repo: str) -> str:
        """Render this issue as the .txt body the existing change-request
        flow expects — a free-form markdown blob with explicit sections
        the planner / patcher can read."""
        labels = ", ".join(self.labels) if self.labels else "(none)"
        return (
            f"# {self.title}\n\n"
            f"## Description\n{self.body or '(no body provided)'}\n\n"
            f"## Source\n"
            f"- GitHub: {self.url}\n"
            f"- Repo: {repo}\n"
            f"- Issue: #{self.number}\n"
            f"- Author: {self.author}\n"
            f"- State: {self.state}\n"
            f"- Labels: {labels}\n"
        )


def fetch_issue(
    repo: str,
    number: int,
    *,
    config: Optional[dict[str, Any]] = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> GithubIssue:
    """Fetch a single issue via ``gh issue view`` and return a
    ``GithubIssue``. Raises ``RuntimeError`` if gh is unavailable, the
    issue does not exist, or the gh call returns non-zero.
    """
    binary = gh_path(config)
    if binary is None:
        raise RuntimeError(
            "gh CLI not found on PATH. Install it from "
            "https://cli.github.com/ and run `gh auth login`."
        )
    fields = "number,title,body,labels,state,author,url"
    cmd = [
        binary, "issue", "view", str(number),
        "--repo", repo,
        "--json", fields,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_seconds, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"gh call failed: {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"gh issue view exit={result.returncode}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"gh returned malformed JSON: {exc}") from exc
    return GithubIssue(
        number=int(data.get("number", number)),
        title=str(data.get("title") or "").strip(),
        body=str(data.get("body") or "").strip(),
        labels=[lbl.get("name", "") for lbl in (data.get("labels") or []) if isinstance(lbl, dict)],
        state=str(data.get("state") or "").upper(),
        author=str((data.get("author") or {}).get("login", "")),
        url=str(data.get("url") or f"https://github.com/{repo}/issues/{number}"),
    )


# ---------------------------------------------------------------------------
# 3. Issue → change_requests/ ingest
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, max_length: int = 40) -> str:
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    return slug[:max_length] if slug else "issue"


def _next_cr_number(change_requests_dir: str) -> int:
    """Find the next free ``CR-N`` number in the change-requests folder."""
    if not os.path.isdir(change_requests_dir):
        return 1
    highest = 0
    for name in os.listdir(change_requests_dir):
        m = re.match(r"^CR-(\d+)", name)
        if m:
            try:
                highest = max(highest, int(m.group(1)))
            except ValueError:
                continue
    return highest + 1


def ingest_issue_to_change_request(
    workspace_path: str,
    repo: str,
    number: int,
    *,
    change_requests_dir: str = "change_requests",
    config: Optional[dict[str, Any]] = None,
) -> str:
    """Fetch a GitHub issue and persist it as the next ``CR-N-<slug>.txt``
    file in the workspace's change-requests folder. Returns the absolute
    path of the written file so the caller can print it.
    """
    issue = fetch_issue(repo, number, config=config)
    target_dir = os.path.join(workspace_path, change_requests_dir)
    os.makedirs(target_dir, exist_ok=True)
    cr_num = _next_cr_number(target_dir)
    slug = _slugify(issue.title)
    filename = f"CR-{cr_num}-{slug}.txt"
    path = os.path.join(target_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(issue.to_change_request_text(repo))
    logger.info("[gh] wrote %s from %s#%d", path, repo, number)
    return path


# ---------------------------------------------------------------------------
# 4. PR creation
# ---------------------------------------------------------------------------

@dataclass
class CreatedPullRequest:
    url: str
    number: Optional[int]


def create_pr(
    workspace_path: str,
    *,
    title: str,
    body: str,
    base: str = "main",
    draft: bool = False,
    config: Optional[dict[str, Any]] = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> CreatedPullRequest:
    """Open a pull request from the workspace's current branch via
    ``gh pr create``. Returns the parsed URL + number. Raises
    ``RuntimeError`` on gh failure.
    """
    binary = gh_path(config)
    if binary is None:
        raise RuntimeError("gh CLI not found on PATH")
    if not title or not title.strip():
        raise ValueError("PR title must be non-empty")
    cmd = [
        binary, "pr", "create",
        "--title", title,
        "--body", body or "",
        "--base", base,
    ]
    if draft:
        cmd.append("--draft")
    try:
        result = subprocess.run(
            cmd, cwd=workspace_path,
            capture_output=True, text=True,
            timeout=timeout_seconds, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"gh pr create failed: {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"gh pr create exit={result.returncode}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    url = result.stdout.strip().splitlines()[-1] if result.stdout else ""
    number_match = re.search(r"/pull/(\d+)", url)
    number = int(number_match.group(1)) if number_match else None
    return CreatedPullRequest(url=url, number=number)


# ---------------------------------------------------------------------------
# 5. PR comment
# ---------------------------------------------------------------------------

def post_pr_comment(
    repo: str,
    pr_number: int,
    body: str,
    *,
    config: Optional[dict[str, Any]] = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> None:
    binary = gh_path(config)
    if binary is None:
        raise RuntimeError("gh CLI not found on PATH")
    if not body or not body.strip():
        raise ValueError("PR comment body must be non-empty")
    cmd = [
        binary, "pr", "comment", str(pr_number),
        "--repo", repo,
        "--body", body,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_seconds, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"gh pr comment failed: {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"gh pr comment exit={result.returncode}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
