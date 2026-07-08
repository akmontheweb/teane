"""``teane pre-flight`` — standalone machine-readiness probe.

The boundary against :mod:`harness.cli` ``doctor``:

* ``pre-flight``: is this *machine* ready to run the harness at all?
  No workspace, no config — tool/runtime/system level checks. Run
  it on a fresh machine before anything else.
* ``doctor``: is this *workspace* configured correctly? Needs a
  ``config/config.json`` and a workspace path.

The same subcommand works on Windows, macOS, and Linux. The OS is
auto-detected via :mod:`harness._platform`; the install commands shown
for any missing tool adapt to the host OS (``winget``/``scoop`` on
Windows, ``brew`` on macOS, ``apt`` on Debian/Ubuntu, generic hint
elsewhere).

Outputs:

* TTY: a sectioned coloured checklist (REQUIRED / SANDBOX /
  RECOMMENDED / OPTIONAL / ENV) plus a one-line summary footer.
* JSON: a flat list of :class:`CheckResult` dicts plus a ``summary``
  block. Used by CI to gate automation.

Exit codes (set by the caller):

* ``0`` — no FAIL rows. The machine is ready; next step is
  ``teane doctor -r <workspace>``.
* ``1`` — at least one FAIL row. The operator must install the
  flagged required tools before the harness can run.
"""

from __future__ import annotations

import json as _json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from typing import Optional

from harness import _platform


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

# Status values mirror the existing ``cmd_doctor`` vocabulary so renderers
# can share colour mappings without translation tables.
STATUS_PASS = "pass"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"
STATUS_SKIP = "skip"


@dataclass
class CheckResult:
    """One row of the pre-flight checklist."""

    name: str
    status: str
    detail: str
    section: str = "REQUIRED"  # REQUIRED / SANDBOX / RECOMMENDED / OPTIONAL / ENV
    install_cmd: str = ""
    feature: str = ""


# ---------------------------------------------------------------------------
# Install-recipe map — single source of truth for "how do I get tool X?"
# ---------------------------------------------------------------------------
# Keyed by tool name. Value is a dict from OS short name to the copy-pasteable
# command. Each command is shown verbatim in the WARN/FAIL detail row.
# Falls back to the "generic" entry when the host OS isn't a key.

INSTALL_RECIPES: dict[str, dict[str, str]] = {
    "git": {
        "linux": "sudo apt install git   # Debian/Ubuntu; use your distro's PM otherwise",
        "macos": "brew install git",
        "windows": "winget install Git.Git   # or download from https://git-scm.com/download/win",
    },
    "docker": {
        "linux": "sudo apt install docker.io && sudo usermod -aG docker $USER",
        "macos": "brew install --cask docker   # then launch Docker Desktop",
        "windows": "winget install Docker.DockerDesktop   # Linux containers mode",
    },
    "node": {
        "linux": "sudo apt install nodejs npm   # or use nvm for a newer version",
        "macos": "brew install node",
        "windows": "winget install OpenJS.NodeJS",
    },
    "java": {
        "linux": "sudo apt install default-jdk",
        "macos": "brew install openjdk@21",
        "windows": "winget install EclipseAdoptium.Temurin.21.JDK",
    },
    "gh": {
        "linux": "sudo apt install gh   # or https://cli.github.com/",
        "macos": "brew install gh",
        "windows": "winget install GitHub.cli",
    },
    "gitleaks": {
        "linux": "sudo apt install gitleaks   # or https://github.com/gitleaks/gitleaks/releases",
        "macos": "brew install gitleaks",
        "windows": "winget install gitleaks.gitleaks",
    },
    "bandit": {"linux": "pip install bandit", "macos": "pip install bandit", "windows": "pip install bandit"},
    "semgrep": {
        "linux": "pip install semgrep",
        "macos": "brew install semgrep",
        "windows": "pip install semgrep",
    },
    "trivy": {
        "linux": "see https://aquasecurity.github.io/trivy/latest/getting-started/installation/",
        "macos": "brew install trivy",
        "windows": "winget install AquaSecurity.Trivy",
    },
    "ruff": {"linux": "pip install ruff", "macos": "pip install ruff", "windows": "pip install ruff"},
    "prettier": {
        "linux": "npm install -g prettier",
        "macos": "npm install -g prettier",
        "windows": "npm install -g prettier",
    },
    "shfmt": {
        "linux": "sudo apt install shfmt",
        "macos": "brew install shfmt",
        "windows": "scoop install shfmt",
    },
    "shellcheck": {
        "linux": "sudo apt install shellcheck",
        "macos": "brew install shellcheck",
        "windows": "scoop install shellcheck",
    },
    "pyright": {
        "linux": "pip install pyright   # or: npm install -g pyright",
        "macos": "pip install pyright   # or: npm install -g pyright",
        "windows": "pip install pyright",
    },
    "mypy": {"linux": "pip install mypy", "macos": "pip install mypy", "windows": "pip install mypy"},
    "tsc": {
        "linux": "npm install -g typescript",
        "macos": "npm install -g typescript",
        "windows": "npm install -g typescript",
    },
}


def _install_recipe(tool: str, platform_name: str) -> str:
    return INSTALL_RECIPES.get(tool, {}).get(platform_name, "")


def _detected_platform_name() -> str:
    if _platform.is_windows():
        return "windows"
    if _platform.is_macos():
        return "macos"
    return "linux"


# ---------------------------------------------------------------------------
# Core probes (any OS)
# ---------------------------------------------------------------------------

def probe_python() -> CheckResult:
    """Python 3.11+ — the harness package requires it."""
    major, minor, micro = sys.version_info[:3]
    version = f"{major}.{minor}.{micro}"
    if (major, minor) >= (3, 11):
        return CheckResult(
            name="Python 3.11+",
            status=STATUS_PASS,
            detail=f"python {version}",
            section="REQUIRED",
        )
    return CheckResult(
        name="Python 3.11+",
        status=STATUS_FAIL,
        detail=f"python {version} — need 3.11 or newer",
        install_cmd="see https://www.python.org/downloads/",
        section="REQUIRED",
    )


def probe_git() -> CheckResult:
    """git on PATH. The harness shells out for branch / stash / status."""
    git_path = shutil.which("git")
    if git_path is None:
        return CheckResult(
            name="git",
            status=STATUS_FAIL,
            detail="not on PATH",
            install_cmd=_install_recipe("git", _detected_platform_name()),
            section="REQUIRED",
        )
    try:
        result = subprocess.run(
            [git_path, "--version"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=5, check=False,
        )
        version = (result.stdout or "").strip() or "version unknown"
    except (OSError, subprocess.TimeoutExpired):
        version = "version probe failed"
    return CheckResult(name="git", status=STATUS_PASS, detail=version, section="REQUIRED")


def probe_home_writable() -> CheckResult:
    """``~/`` must be writable — the harness uses ~/.harness/* for state."""
    home = os.path.expanduser("~")
    if not home or home == "~":
        return CheckResult(
            name="home dir writable",
            status=STATUS_FAIL,
            detail="cannot resolve ~ (HOME/USERPROFILE unset?)",
            section="REQUIRED",
        )
    if os.access(home, os.W_OK):
        return CheckResult(
            name="home dir writable",
            status=STATUS_PASS,
            detail=home,
            section="REQUIRED",
        )
    return CheckResult(
        name="home dir writable",
        status=STATUS_FAIL,
        detail=f"{home} not writable",
        section="REQUIRED",
    )


def probe_temp_writable() -> CheckResult:
    """System temp dir must be writable — used for sandbox logs, worktrees."""
    tmp = _platform.harness_temp_dir("")
    try:
        # We don't need a permanent file, just a writability probe.
        with tempfile.NamedTemporaryFile(dir=tmp, delete=True):
            pass
        return CheckResult(
            name="system temp writable",
            status=STATUS_PASS,
            detail=tmp,
            section="REQUIRED",
        )
    except OSError as exc:
        return CheckResult(
            name="system temp writable",
            status=STATUS_FAIL,
            detail=f"{tmp} not writable: {exc}",
            section="REQUIRED",
        )


def probe_disk_space(min_mb: int = 500) -> CheckResult:
    """At least ``min_mb`` MB free in home — for venv + tree-sitter + DB."""
    home = os.path.expanduser("~")
    try:
        usage = shutil.disk_usage(home)
        free_mb = usage.free // (1024 * 1024)
    except OSError as exc:
        return CheckResult(
            name="disk space",
            status=STATUS_WARN,
            detail=f"probe failed: {exc}",
            section="REQUIRED",
        )
    if free_mb >= min_mb:
        return CheckResult(
            name="disk space",
            status=STATUS_PASS,
            detail=f"{free_mb // 1024 if free_mb >= 1024 else free_mb}"
                   f"{' GB' if free_mb >= 1024 else ' MB'} free in {home}",
            section="REQUIRED",
        )
    return CheckResult(
        name="disk space",
        status=STATUS_FAIL,
        detail=f"only {free_mb} MB free in {home} (need ≥{min_mb} MB)",
        section="REQUIRED",
    )


def probe_outbound_https(host: str = "api.anthropic.com",
                        port: int = 443,
                        timeout: float = 5.0) -> CheckResult:
    """TCP connection to an LLM provider host. Catches firewalls/proxies
    before the operator wastes time on API-key debugging."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
        return CheckResult(
            name="outbound HTTPS",
            status=STATUS_PASS,
            detail=f"{host}:{port} reachable",
            section="REQUIRED",
        )
    except (socket.gaierror, socket.timeout, OSError) as exc:
        return CheckResult(
            name="outbound HTTPS",
            status=STATUS_WARN,
            detail=f"{host}:{port} unreachable: {exc} "
                   "(corporate firewall? check api.anthropic.com / api.openai.com)",
            section="REQUIRED",
        )


# ---------------------------------------------------------------------------
# Sandbox probes
# ---------------------------------------------------------------------------

def probe_docker() -> CheckResult:
    """``docker info`` succeeds, and (on Windows/macOS) Linux containers
    are the active mode."""
    docker_path = shutil.which("docker")
    if docker_path is None:
        return CheckResult(
            name="Docker daemon",
            status=STATUS_WARN,
            detail="docker not on PATH (sandbox falls back to bare unless "
                   "unshare is available)",
            install_cmd=_install_recipe("docker", _detected_platform_name()),
            section="SANDBOX",
        )
    try:
        result = subprocess.run(
            [docker_path, "info", "--format", "{{.OperatingSystem}};{{.OSType}}"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult(
            name="Docker daemon",
            status=STATUS_WARN,
            detail=f"docker info failed: {exc}",
            section="SANDBOX",
        )
    if result.returncode != 0:
        return CheckResult(
            name="Docker daemon",
            status=STATUS_WARN,
            detail="docker daemon not reachable (is Docker Desktop running?)",
            section="SANDBOX",
        )
    parts = (result.stdout or "").strip().split(";")
    os_type = parts[1].strip() if len(parts) > 1 else ""
    if os_type and os_type != "linux":
        return CheckResult(
            name="Docker daemon",
            status=STATUS_WARN,
            detail=f"Linux containers required, but daemon reports OSType={os_type!r} "
                   "— switch Docker Desktop to Linux containers mode",
            section="SANDBOX",
        )
    detail = (parts[0].strip() or "running") if parts else "running"
    return CheckResult(
        name="Docker daemon",
        status=STATUS_PASS,
        detail=f"{detail} ({os_type or 'linux'} containers)",
        section="SANDBOX",
    )


def probe_unshare() -> CheckResult:
    """Linux user-namespace support. Skipped on macOS/Windows."""
    if not _platform.is_linux():
        return CheckResult(
            name="unshare",
            status=STATUS_SKIP,
            detail="Linux-only sandbox backend",
            section="SANDBOX",
        )
    if shutil.which("unshare") is None:
        return CheckResult(
            name="unshare",
            status=STATUS_WARN,
            detail="not on PATH (sandbox.backend=unshare unavailable)",
            section="SANDBOX",
        )
    try:
        result = subprocess.run(
            ["unshare", "--user", "echo", "ok"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return CheckResult(
            name="unshare",
            status=STATUS_WARN,
            detail="probe failed",
            section="SANDBOX",
        )
    if result.returncode == 0 and (result.stdout or "").strip() == "ok":
        return CheckResult(
            name="unshare",
            status=STATUS_PASS,
            detail="user namespaces working",
            section="SANDBOX",
        )
    return CheckResult(
        name="unshare",
        status=STATUS_WARN,
        detail="user namespaces disabled (RHEL/Fedora may need "
               "sysctl -w user.max_user_namespaces=15000)",
        section="SANDBOX",
    )


# ---------------------------------------------------------------------------
# Windows-specific probes
# ---------------------------------------------------------------------------

def probe_taskkill() -> CheckResult:
    """``taskkill /T /F /PID`` powers the cross-platform tree-kill on Windows."""
    if not _platform.is_windows():
        return CheckResult(name="taskkill", status=STATUS_SKIP, detail="Windows-only", section="SANDBOX")
    if shutil.which("taskkill") is None:
        return CheckResult(
            name="taskkill",
            status=STATUS_FAIL,
            detail="not on PATH — process tree kill will degrade to parent-only",
            install_cmd="taskkill ships with every supported Windows; check %WINDIR%\\System32 is on PATH",
            section="SANDBOX",
        )
    return CheckResult(name="taskkill", status=STATUS_PASS, detail="on PATH", section="SANDBOX")


def probe_posix_sh() -> CheckResult:
    """POSIX ``sh`` on PATH — needed on Windows for schedule hooks using
    POSIX syntax (pipes, ``&&``, redirects). Without it, hooks run under
    cmd.exe and POSIX syntax silently fails."""
    if not _platform.is_windows():
        return CheckResult(
            name="POSIX sh",
            status=STATUS_SKIP,
            detail="POSIX shell is the default on this platform",
            section="RECOMMENDED",
        )
    sh = _platform.posix_shell_path()
    if sh:
        return CheckResult(
            name="POSIX sh",
            status=STATUS_PASS,
            detail=f"{sh} — schedule hooks can use POSIX syntax (pipes, &&)",
            section="RECOMMENDED",
        )
    return CheckResult(
        name="POSIX sh",
        status=STATUS_WARN,
        detail="not on PATH — schedule hooks will run under cmd.exe; "
               "POSIX syntax (pipes, &&, redirects) won't work",
        install_cmd="winget install Git.Git   # Git Bash bundles sh",
        section="RECOMMENDED",
    )


def probe_long_paths() -> CheckResult:
    """Windows LongPathsEnabled registry value. Workspace paths >260
    chars fail silently when this is 0 (the OS default)."""
    if not _platform.is_windows():
        return CheckResult(
            name="long paths",
            status=STATUS_SKIP,
            detail="POSIX path limit ~4096 — no registry knob needed",
            section="REQUIRED",
        )
    try:
        import winreg  # type: ignore[import-not-found, unused-ignore]
    except ImportError:
        return CheckResult(
            name="long paths",
            status=STATUS_WARN,
            detail="winreg module unavailable",
            section="REQUIRED",
        )
    install_cmd = (
        'New-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet'
        '\\Control\\FileSystem" -Name "LongPathsEnabled" -Value 1 '
        '-PropertyType DWORD -Force   # then reboot. Run as Administrator.'
    )
    try:
        key = winreg.OpenKey(  # type: ignore[attr-defined, unused-ignore]
            winreg.HKEY_LOCAL_MACHINE,  # type: ignore[attr-defined, unused-ignore]
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
        )
        try:
            val, _kind = winreg.QueryValueEx(key, "LongPathsEnabled")  # type: ignore[attr-defined, unused-ignore]
        finally:
            winreg.CloseKey(key)  # type: ignore[attr-defined, unused-ignore]
    except (OSError, FileNotFoundError):
        return CheckResult(
            name="long paths",
            status=STATUS_WARN,
            detail="LongPathsEnabled key not found",
            install_cmd=install_cmd,
            section="REQUIRED",
        )
    if int(val) == 1:
        return CheckResult(
            name="long paths",
            status=STATUS_PASS,
            detail="LongPathsEnabled=1",
            section="REQUIRED",
        )
    return CheckResult(
        name="long paths",
        status=STATUS_FAIL,
        detail="LongPathsEnabled=0 — workspace paths >260 chars will fail",
        install_cmd=install_cmd,
        section="REQUIRED",
    )


def probe_git_longpaths_config() -> CheckResult:
    """``git config --global core.longpaths`` should be ``true`` on
    Windows so git can read/write paths >260 chars."""
    if not _platform.is_windows():
        return CheckResult(
            name="git longpaths config",
            status=STATUS_SKIP,
            detail="Windows-only",
            section="REQUIRED",
        )
    git_path = shutil.which("git")
    if git_path is None:
        return CheckResult(
            name="git longpaths config",
            status=STATUS_SKIP,
            detail="git not on PATH",
            section="REQUIRED",
        )
    try:
        result = subprocess.run(
            [git_path, "config", "--global", "core.longpaths"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return CheckResult(
            name="git longpaths config",
            status=STATUS_WARN,
            detail="probe failed",
            section="REQUIRED",
        )
    val = (result.stdout or "").strip().lower()
    if val == "true":
        return CheckResult(
            name="git longpaths config",
            status=STATUS_PASS,
            detail="core.longpaths=true",
            section="REQUIRED",
        )
    return CheckResult(
        name="git longpaths config",
        status=STATUS_WARN,
        detail=f"core.longpaths={val!r} — paths >260 chars may fail in git",
        install_cmd="git config --global core.longpaths true",
        section="REQUIRED",
    )


# ---------------------------------------------------------------------------
# macOS-specific probes
# ---------------------------------------------------------------------------

def probe_xcode_cli() -> CheckResult:
    """Xcode Command Line Tools provide C compiler + make on macOS."""
    if not _platform.is_macos():
        return CheckResult(
            name="Xcode CLI tools",
            status=STATUS_SKIP,
            detail="macOS-only",
            section="REQUIRED",
        )
    xcselect = shutil.which("xcode-select")
    if xcselect is None:
        return CheckResult(
            name="Xcode CLI tools",
            status=STATUS_FAIL,
            detail="xcode-select not on PATH",
            install_cmd="xcode-select --install",
            section="REQUIRED",
        )
    try:
        result = subprocess.run(
            [xcselect, "-p"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return CheckResult(
            name="Xcode CLI tools",
            status=STATUS_WARN,
            detail="probe failed",
            section="REQUIRED",
        )
    if result.returncode == 0 and (result.stdout or "").strip():
        return CheckResult(
            name="Xcode CLI tools",
            status=STATUS_PASS,
            detail=(result.stdout or "").strip(),
            section="REQUIRED",
        )
    return CheckResult(
        name="Xcode CLI tools",
        status=STATUS_FAIL,
        detail="not installed",
        install_cmd="xcode-select --install",
        section="REQUIRED",
    )


# ---------------------------------------------------------------------------
# Optional tool probes
# ---------------------------------------------------------------------------

_LANGUAGE_TOOLS = (
    ("node", "Node.js (MCP servers, prettier, npx)"),
    ("java", "Java runtime (Spring Boot / Maven / Gradle)"),
)

_SECURITY_TOOLS = (
    ("gitleaks", "secret scanning (regex fallback if missing)"),
    ("bandit", "Python SAST"),
    ("semgrep", "universal SAST"),
    ("trivy", "dependency vulnerability scan"),
)

_FORMATTER_TOOLS = (
    ("ruff", "Python format + lint"),
    ("prettier", "JS/TS/CSS/MD format"),
    ("shellcheck", "shell lint"),
)

_TYPECHECK_TOOLS = (
    ("pyright", "Python type check (diagnostics gate)"),
    ("mypy", "Python type check fallback (diagnostics gate)"),
    ("tsc", "TypeScript type check (diagnostics gate)"),
)


def _probe_optional_binary(tool: str, feature: str, section: str) -> CheckResult:
    """Generic 'is this binary on PATH?' probe used by the language /
    security / formatter groupings. PASS reports the version string when
    one is cheaply available."""
    path = shutil.which(tool)
    if path is None:
        return CheckResult(
            name=tool,
            status=STATUS_WARN,
            detail=f"not on PATH ({feature})",
            install_cmd=_install_recipe(tool, _detected_platform_name()),
            section=section,
            feature=feature,
        )
    return CheckResult(
        name=tool,
        status=STATUS_PASS,
        detail=f"on PATH ({feature})",
        section=section,
        feature=feature,
    )


def probe_language_toolchains() -> list[CheckResult]:
    return [_probe_optional_binary(t, f, "OPTIONAL") for t, f in _LANGUAGE_TOOLS]


def probe_security_scanners() -> list[CheckResult]:
    return [_probe_optional_binary(t, f, "RECOMMENDED") for t, f in _SECURITY_TOOLS]


def probe_formatters() -> list[CheckResult]:
    return [_probe_optional_binary(t, f, "RECOMMENDED") for t, f in _FORMATTER_TOOLS]


def probe_typecheckers() -> list[CheckResult]:
    return [_probe_optional_binary(t, f, "RECOMMENDED") for t, f in _TYPECHECK_TOOLS]


def probe_gh_cli() -> CheckResult:
    return _probe_optional_binary("gh", "`teane gh` subcommands", "OPTIONAL")


def probe_llm_api_keys() -> list[CheckResult]:
    """Reports which provider env vars are set. INFO-level — pre-flight
    can't know which providers are routed (that's config-dependent and
    a `teane doctor` concern). Useful as a 'what does this machine
    have wired up?' overview."""
    keys = [
        ("ANTHROPIC_API_KEY", "Anthropic Claude"),
        ("OPENAI_API_KEY", "OpenAI GPT"),
        ("DEEPSEEK_API_KEY", "DeepSeek"),
    ]
    out: list[CheckResult] = []
    for env_name, provider in keys:
        if os.environ.get(env_name):
            out.append(CheckResult(
                name=env_name,
                status=STATUS_PASS,
                detail=f"set ({provider})",
                section="ENV",
            ))
        else:
            out.append(CheckResult(
                name=env_name,
                status=STATUS_SKIP,
                detail=f"not set ({provider}; only needed if routed)",
                section="ENV",
            ))
    return out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def run_all(*, platform_override: Optional[str] = None,
            quick: bool = False) -> list[CheckResult]:
    """Run the right probe set for the detected (or overridden) platform.

    ``platform_override`` in {"windows", "linux", "macos"} forces the
    OS-specific check set; useful for documentation verification on a
    different host. By default ("auto"/None) the OS is detected via
    :mod:`harness._platform`.

    ``quick=True`` skips the live network probe.
    """
    # Apply override by monkeypatching the _platform predicates *for this
    # call only*. We do this inline rather than threading a flag through
    # every probe — the probes stay simple.
    restore: list[tuple[str, object]] = []
    if platform_override and platform_override != "auto":
        for attr, expected in [
            ("is_windows", platform_override == "windows"),
            ("is_linux", platform_override == "linux"),
            ("is_macos", platform_override == "macos"),
        ]:
            restore.append((attr, getattr(_platform, attr)))
            setattr(_platform, attr, (lambda v=expected: v))

    try:
        results: list[CheckResult] = []

        # Required.
        results.append(probe_python())
        results.append(probe_git())
        results.append(probe_home_writable())
        results.append(probe_temp_writable())
        results.append(probe_disk_space())
        if not quick:
            results.append(probe_outbound_https())
        else:
            results.append(CheckResult(
                name="outbound HTTPS",
                status=STATUS_SKIP,
                detail="skipped via --quick",
                section="REQUIRED",
            ))

        # Per-OS required.
        if _platform.is_windows():
            results.append(probe_long_paths())
            results.append(probe_git_longpaths_config())
        if _platform.is_macos():
            results.append(probe_xcode_cli())

        # Sandbox.
        results.append(probe_docker())
        if _platform.is_linux():
            results.append(probe_unshare())
        if _platform.is_windows():
            results.append(probe_taskkill())

        # Recommended.
        results.append(probe_posix_sh())
        results.extend(probe_security_scanners())
        results.extend(probe_formatters())
        results.extend(probe_typecheckers())

        # Optional.
        results.append(probe_gh_cli())
        results.extend(probe_language_toolchains())

        # Env (informational).
        results.extend(probe_llm_api_keys())

        return results
    finally:
        for attr, original in restore:
            setattr(_platform, attr, original)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

_SECTION_ORDER = ("REQUIRED", "SANDBOX", "RECOMMENDED", "OPTIONAL", "ENV")


def _colors(no_color: bool) -> tuple[str, str, str, str]:
    if no_color or not sys.stdout.isatty() or os.environ.get("NO_COLOR", "") != "":
        return ("", "", "", "")
    return ("\033[32m", "\033[33m", "\033[31m", "\033[0m")


def _marker(status: str, no_color: bool) -> str:
    green, yellow, red, reset = _colors(no_color)
    if status == STATUS_PASS:
        return f"{green}✓{reset}"
    if status == STATUS_WARN:
        return f"{yellow}⚠{reset}"
    if status == STATUS_FAIL:
        return f"{red}✗{reset}"
    return "·"


def render_tty(results: list[CheckResult], *, no_color: bool = False,
               platform_name: Optional[str] = None) -> str:
    """Sectioned coloured checklist."""
    if platform_name is None:
        platform_name = _detected_platform_name()
    lines: list[str] = []
    lines.append("")
    lines.append(f"teane pre-flight — {platform_name}")
    lines.append("")

    by_section: dict[str, list[CheckResult]] = {s: [] for s in _SECTION_ORDER}
    for r in results:
        by_section.setdefault(r.section, []).append(r)

    for section in _SECTION_ORDER:
        rows = by_section.get(section, [])
        if not rows:
            continue
        lines.append(f"  {section}")
        for r in rows:
            marker = _marker(r.status, no_color)
            lines.append(f"  {marker} {r.name:<28} {r.detail}")
            if r.install_cmd and r.status in (STATUS_WARN, STATUS_FAIL):
                lines.append(f"      → {r.install_cmd}")
        lines.append("")

    # Summary.
    counts = {STATUS_PASS: 0, STATUS_WARN: 0, STATUS_FAIL: 0, STATUS_SKIP: 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    green, yellow, red, reset = _colors(no_color)
    summary = (
        f"  {green}{counts[STATUS_PASS]} ✓{reset} · "
        f"{yellow}{counts[STATUS_WARN]} ⚠{reset} · "
        f"{red}{counts[STATUS_FAIL]} ✗{reset} · "
        f"{counts[STATUS_SKIP]} skipped"
    )
    lines.append(summary)
    if counts[STATUS_FAIL] == 0:
        lines.append("  Ready for `teane doctor -r <workspace>`.")
    else:
        lines.append(
            "  Install the failed REQUIRED items above, then re-run "
            "`teane pre-flight`."
        )
    lines.append("")
    return "\n".join(lines)


def render_json(results: list[CheckResult], *,
                platform_name: Optional[str] = None) -> str:
    """Machine-readable. Stable shape: ``{platform, results: [...], summary: {...}}``."""
    if platform_name is None:
        platform_name = _detected_platform_name()
    counts = {STATUS_PASS: 0, STATUS_WARN: 0, STATUS_FAIL: 0, STATUS_SKIP: 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    payload = {
        "platform": platform_name,
        "results": [asdict(r) for r in results],
        "summary": {
            "pass": counts[STATUS_PASS],
            "warn": counts[STATUS_WARN],
            "fail": counts[STATUS_FAIL],
            "skip": counts[STATUS_SKIP],
            "exit_code": 1 if counts[STATUS_FAIL] > 0 else 0,
        },
    }
    return _json.dumps(payload, indent=2)
