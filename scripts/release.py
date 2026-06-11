"""
Release driver invoked by `make release`.

Steps:
    1. Refuse if the working tree is dirty.
    2. Refuse if we're not on `main`.
    3. Run the pytest pack — refuse on any failure.
    4. Parse current version from pyproject.toml.
    5. Bump per --bump (patch / minor / major).
    6. Verify CHANGELOG.md has an [Unreleased] section with content.
    7. Rewrite CHANGELOG.md: [Unreleased] -> [<new_version>] - <today>.
    8. Rewrite pyproject.toml version.
    9. Confirm with the operator (`yes` to proceed).
    10. Commit, tag, and push (with the tag).

Exits non-zero on any failure. Idempotent in the sense that step (1)
catches a half-finished release attempt.
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command from the repo root, raising on non-zero exit."""
    kwargs.setdefault("cwd", REPO_ROOT)
    kwargs.setdefault("text", True)
    return subprocess.run(cmd, check=True, **kwargs)


def die(msg: str) -> None:
    print(f"[release] FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def ensure_clean_tree() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    if result.stdout.strip():
        die(
            "working tree is dirty. Commit or stash before releasing:\n"
            f"{result.stdout}"
        )


def ensure_on_main() -> None:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    branch = result.stdout.strip()
    if branch != "main":
        die(f"releases must be cut from 'main', currently on '{branch}'.")


def run_tests() -> None:
    print("[release] Running pytest pack...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        die("pytest failed; refusing to release with a red suite.")


def current_version() -> tuple[int, int, int]:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"(\d+)\.(\d+)\.(\d+)"', text, re.MULTILINE)
    if not match:
        die("could not find version = \"X.Y.Z\" in pyproject.toml.")
    return tuple(int(g) for g in match.groups())  # type: ignore[return-value]


def bump_version(version: tuple[int, int, int], kind: str) -> tuple[int, int, int]:
    major, minor, patch = version
    if kind == "major":
        return major + 1, 0, 0
    if kind == "minor":
        return major, minor + 1, 0
    if kind == "patch":
        return major, minor, patch + 1
    die(f"unknown bump kind '{kind}'. Use patch / minor / major.")
    return version  # unreachable


def rewrite_pyproject(new_version: str) -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    new_text, count = re.subn(
        r'^(version\s*=\s*")\d+\.\d+\.\d+(")',
        rf'\g<1>{new_version}\g<2>',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        die("could not rewrite version in pyproject.toml.")
    PYPROJECT.write_text(new_text, encoding="utf-8")


def rewrite_changelog(new_version: str, today: str) -> None:
    text = CHANGELOG.read_text(encoding="utf-8")
    if "## [Unreleased]" not in text:
        die("CHANGELOG.md has no [Unreleased] section.")

    # Extract the [Unreleased] block to ensure it actually has content.
    blocks = text.split("## [Unreleased]", 1)
    after = blocks[1]
    next_heading = re.search(r"\n## \[", after)
    unreleased_body = after[: next_heading.start()] if next_heading else after
    if not unreleased_body.strip():
        die("CHANGELOG.md [Unreleased] section is empty; nothing to release.")

    new_text = text.replace(
        "## [Unreleased]",
        f"## [Unreleased]\n\n## [{new_version}] - {today}",
        1,
    )
    CHANGELOG.write_text(new_text, encoding="utf-8")


def confirm(prompt: str) -> bool:
    answer = input(f"[release] {prompt} [type 'yes' to proceed] ").strip().lower()
    return answer == "yes"


def main() -> int:
    parser = argparse.ArgumentParser(description="Cut a SemVer release.")
    parser.add_argument(
        "--bump",
        choices=("patch", "minor", "major"),
        default="patch",
        help="Which SemVer component to bump (default: patch).",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip the final `git push` step (still commits + tags locally).",
    )
    args = parser.parse_args()

    ensure_clean_tree()
    ensure_on_main()
    run_tests()

    current = current_version()
    new = bump_version(current, args.bump)
    current_str = ".".join(str(c) for c in current)
    new_str = ".".join(str(c) for c in new)
    today = datetime.date.today().isoformat()

    print()
    print(f"[release] Current version: {current_str}")
    print(f"[release] New version:     {new_str}  ({args.bump} bump)")
    print(f"[release] Date:            {today}")
    print()

    if not confirm(f"Cut v{new_str}? This will rewrite pyproject.toml + CHANGELOG.md, commit, tag v{new_str}, and push."):
        print("[release] Aborted by operator.")
        return 1

    rewrite_pyproject(new_str)
    rewrite_changelog(new_str, today)

    run(["git", "add", "pyproject.toml", "CHANGELOG.md"])
    run([
        "git", "commit", "-m",
        f"chore(release): v{new_str}\n\nBump version and roll CHANGELOG.md.",
    ])
    run(["git", "tag", "-a", f"v{new_str}", "-m", f"Release v{new_str}"])

    if args.no_push:
        print("[release] Local commit + tag created. Skipping push (--no-push).")
        print("[release] To finish: git push && git push --tags")
        return 0

    run(["git", "push"])
    run(["git", "push", "origin", f"v{new_str}"])
    print(f"[release] v{new_str} pushed to origin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
