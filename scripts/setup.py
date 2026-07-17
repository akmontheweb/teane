#!/usr/bin/env python3
"""Interactive cross-platform bootstrap for teane.

Runs through eleven phases that mirror docs/installation.md, but with
prompts and probes wired up so the operator types a few answers and
ends with a green `teane doctor`. Stdlib only — runs anywhere a
Python 3.9+ interpreter is on PATH (we then locate Python 3.14+
ourselves before creating the venv).

Examples
--------
    python3 scripts/setup.py
    python3 scripts/setup.py --venv /opt/harness/venv --dev
    python3 scripts/setup.py --non-interactive --provider anthropic

The script never sudo's, never installs system packages on the user's
behalf; it prints the platform-specific install command and asks the
operator to run it. The only thing the script does install is the
harness's Python deps, and that goes into a venv it owns.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
MODEL_CATALOGUE = REPO_ROOT / "harness" / "model_prices.json"
DEFAULT_VENV = "~/.venvs/teane"
CONFIG_DIR = REPO_ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Per-provider default model keys. These reference catalogue entries in
# harness/model_prices.json — the script verifies the chosen key actually
# exists before writing the config.
DEFAULT_MODELS_BY_PROVIDER: dict[str, str] = {
    "anthropic": "anthropic:claude-sonnet-4-6",
    "openai": "openai:gpt-4o-mini",
    "deepseek": "deepseek:deepseek-v4-flash",
    "ollama": "ollama:llama3.2",
}

PROVIDER_ENV_VAR: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "ollama": "",   # local — no key needed
}

TOTAL_PHASES = 11


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR", "") == ""


def _color(code: str, text: str) -> str:
    if not _supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def _green(t: str) -> str: return _color("32", t)
def _red(t: str) -> str: return _color("31", t)
def _yellow(t: str) -> str: return _color("33", t)
def _bold(t: str) -> str: return _color("1", t)


def _banner(phase: int, title: str) -> None:
    print()
    print(_bold(f"[{phase}/{TOTAL_PHASES}] {title}"))


def _ok(msg: str) -> None:
    print(f"  {_green('✓')} {msg}")


def _warn(msg: str) -> None:
    print(f"  {_yellow('!')} {msg}")


def _fail(msg: str) -> None:
    print(f"  {_red('✗')} {msg}")


def _info(msg: str) -> None:
    print(f"    {msg}")


# ---------------------------------------------------------------------------
# Phase 1 — Platform detection
# ---------------------------------------------------------------------------

def _detect_platform() -> str:
    """Return one of: linux, darwin, windows, wsl2."""
    system = platform.system().lower()
    if system == "darwin":
        return "darwin"
    if system == "windows":
        return "windows"
    # Linux — distinguish WSL2 from native by sniffing /proc/version.
    try:
        proc_version = Path("/proc/version").read_text(errors="replace").lower()
        if "microsoft" in proc_version or "wsl" in proc_version:
            return "wsl2"
    except OSError:
        pass
    return "linux"


# ---------------------------------------------------------------------------
# Phase 2 — Python 3.14+ probe
# ---------------------------------------------------------------------------

def _verify_python(path: str) -> Optional[tuple[int, int]]:
    """Return ``(major, minor)`` when ``path`` is an executable Python ≥ 3.14.

    Validates with ``--version`` rather than trusting ``shutil.which``,
    because pyenv shims can land on PATH for versions pyenv hasn't
    actually installed — the shim exists, but invoking it exits 127.
    Returns None on any failure so the caller falls through to the
    next candidate.
    """
    try:
        result = subprocess.run(
            path.split() + ["--version"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"Python (\d+)\.(\d+)", result.stdout + result.stderr)
    if not match:
        return None
    major, minor = int(match.group(1)), int(match.group(2))
    if (major, minor) < (3, 14):
        return None
    return (major, minor)


def _find_python314() -> Optional[str]:
    """Return a working Python 3.14+ interpreter, or None.

    Each candidate is validated with a ``--version`` exec — a pyenv shim
    that lives on PATH but exec-fails is treated as not-found, so the
    search moves on to the next interpreter.
    """
    # Try explicit version names first
    for candidate in ("python3.14", "python3.15", "python3.16"):
        path = shutil.which(candidate)
        if path and _verify_python(path) is not None:
            return path
    # Windows: `py -3.14`
    py = shutil.which("py")
    if py:
        full = f"{py} -3.14"
        if _verify_python(full) is not None:
            return full
    # Parse `python3 --version` as a last resort (must be ≥ 3.14)
    for candidate in ("python3", "python"):
        path = shutil.which(candidate)
        if path and _verify_python(path) is not None:
            return path
    return None


# ---------------------------------------------------------------------------
# Phase 3-5 — System tool probes
# ---------------------------------------------------------------------------

def _probe_git() -> Optional[str]:
    return shutil.which("git")


def _probe_sqlite3() -> Optional[str]:
    return shutil.which("sqlite3")


def _probe_docker() -> Optional[str]:
    """Return the docker binary path if `docker info` succeeds, else None."""
    docker = shutil.which("docker")
    if not docker:
        return None
    try:
        result = subprocess.run(
            [docker, "info"],
            capture_output=True, text=True, timeout=8,
        )
        if result.returncode == 0:
            return docker
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def _probe_unshare() -> bool:
    """True if `unshare --user echo ok` works (Linux/WSL2 only)."""
    unshare = shutil.which("unshare")
    if not unshare:
        return False
    try:
        result = subprocess.run(
            [unshare, "--user", "echo", "ok"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0 and result.stdout.strip() == "ok"
    except (subprocess.TimeoutExpired, OSError):
        return False


def _probe_build_toolchain(platform_id: str) -> bool:
    """Best-effort check for a compiler toolchain (for tree-sitter wheels)."""
    if platform_id in ("linux", "wsl2"):
        return shutil.which("gcc") is not None or shutil.which("cc") is not None
    if platform_id == "darwin":
        try:
            result = subprocess.run(
                ["xcode-select", "-p"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except OSError:
            return False
    # Windows: we trust the build wheels and don't probe MSVC eagerly.
    return True


# ---------------------------------------------------------------------------
# Install-command lookup table
# ---------------------------------------------------------------------------

_INSTALL_COMMANDS: dict[tuple[str, str], str] = {
    # (tool, platform) → install command
    ("python3.14", "linux"):   "sudo apt install -y python3.14 python3.14-venv python3.14-dev",
    ("python3.14", "wsl2"):    "sudo apt install -y python3.14 python3.14-venv python3.14-dev",
    ("python3.14", "darwin"):  "brew install python@3.14",
    ("python3.14", "windows"): "Install Python 3.14 from https://www.python.org/downloads/windows/ (tick 'Add to PATH')",
    ("git", "linux"):    "sudo apt install -y git",
    ("git", "wsl2"):     "sudo apt install -y git",
    ("git", "darwin"):   "brew install git",
    ("git", "windows"):  "Install Git for Windows from https://git-scm.com/download/win",
    ("sqlite3", "linux"):    "sudo apt install -y sqlite3",
    ("sqlite3", "wsl2"):     "sudo apt install -y sqlite3",
    ("sqlite3", "darwin"):   "brew install sqlite",
    ("sqlite3", "windows"):  "Bundled with Python's stdlib — no separate install needed",
    ("docker", "linux"):     "sudo apt install -y docker.io && sudo usermod -aG docker $USER && newgrp docker",
    ("docker", "wsl2"):      "Install Docker Desktop on the Windows host, then enable WSL Integration for this distro",
    ("docker", "darwin"):    "Install Docker Desktop: https://www.docker.com/products/docker-desktop/",
    ("docker", "windows"):   "Install Docker Desktop (Linux containers mode): https://www.docker.com/products/docker-desktop/",
    ("build-toolchain", "linux"):   "sudo apt install -y build-essential",
    ("build-toolchain", "wsl2"):    "sudo apt install -y build-essential",
    ("build-toolchain", "darwin"):  "xcode-select --install",
    ("build-toolchain", "windows"): "Install Microsoft C++ Build Tools (Desktop development with C++): https://visualstudio.microsoft.com/visual-cpp-build-tools/",
}


def _install_command_for(tool: str, platform_id: str) -> str:
    """Look up the platform-specific install command for a tool."""
    return _INSTALL_COMMANDS.get((tool, platform_id), f"Install {tool} for your platform.")


# ---------------------------------------------------------------------------
# Phase 6 — Venv
# ---------------------------------------------------------------------------

def _venv_python(venv_path: Path) -> Path:
    """Return the venv's python executable path."""
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _venv_pip(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "pip.exe"
    return venv_path / "bin" / "pip"


def _venv_harness(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "teane.exe"
    return venv_path / "bin" / "harness"


def _activation_command(venv_path: Path, platform_id: str) -> str:
    if platform_id == "windows":
        return f"& {venv_path}\\Scripts\\Activate.ps1"
    return f"source {venv_path}/bin/activate"


# ---------------------------------------------------------------------------
# Phase 7 — pip install
# ---------------------------------------------------------------------------

def _run_pip_install(venv_path: Path, dev: bool) -> bool:
    """Run pip install -e .[dev?] inside the venv. Returns True on success."""
    pip = _venv_pip(venv_path)
    if not pip.is_file():
        _fail(f"pip not found at {pip}. Is the venv valid?")
        return False
    target = f"{REPO_ROOT}{'[dev]' if dev else ''}"
    cmd = [str(pip), "install", "-e", target]
    print(f"    Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT)
    except OSError as exc:
        _fail(f"pip install failed: {exc}")
        return False
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Phase 8 — Config wizard
# ---------------------------------------------------------------------------

def _load_model_catalogue() -> set[str]:
    """Return the set of model keys in harness/model_prices.json."""
    try:
        data = json.loads(MODEL_CATALOGUE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return {k for k in data.keys() if not k.startswith("_")}


def _build_default_config(provider: str, model_key: str) -> dict:
    """Build the config body to write to ``<root>/config/config.json``.

    Reads the shipped ``config/config.json`` as the base template — that
    file IS the canonical, strictly-validated config with every required
    section (product_spec_dir, models registry, sandbox, token_budget,
    persistence, deployment_defaults, mcp, …). We then patch ONLY the
    provider-specific ``model_routing.*_primary`` fields so the operator's
    provider choice is honoured. Every other section is preserved
    verbatim.

    Prior versions of this function returned a 28-line stub containing
    only ``model_routing`` + ``node_throttle`` — writing that stub over
    the shipped config left ``discover_config`` raising ConfigError on
    missing product_spec_dir / sandbox / etc., and the operator would
    have to restore the shipped config from git before doing anything.
    Session finsearch/2026-07-14 tripped exactly this on macOS.
    """
    # Base: the shipped canonical config, byte-loaded and JSON-parsed.
    # This is guaranteed to pass strict validation because CI runs
    # ``teane doctor`` against it. If we can't read it for any reason,
    # fail loudly — the caller should NOT proceed and clobber the file.
    try:
        base = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read shipped config template at {CONFIG_FILE}: {exc}. "
            "This file MUST exist in the repo — it's the base every "
            "operator config is derived from. Restore it from git: "
            "`git checkout config/config.json`."
        ) from exc
    if not isinstance(base, dict):
        raise RuntimeError(
            f"Shipped config template at {CONFIG_FILE} is not a JSON "
            f"object (got {type(base).__name__}). Restore from git."
        )
    # Provider-specific routing patch — only touch these keys, everything
    # else in the shipped template flows through.
    routing = dict(base.get("model_routing") or {})
    routing.update({
        "planning_primary": model_key,
        "patching_primary": model_key,
        "repair_primary": model_key,
    })
    base["model_routing"] = routing
    base["_generated_by_setup"] = (
        f"scripts/setup.py patched model_routing.*_primary → {model_key} "
        f"(provider={provider}); all other sections preserved from the "
        f"shipped config template."
    )
    return base


def _config_is_richer_than_default(existing_path: Path) -> bool:
    """Guard against downgrading a hand-tuned config.

    The shipped config template ships every section the harness needs;
    if the operator has added their own custom sections (e.g. custom
    ``skills``, extra ``models`` entries, tuned ``mcp.servers``), a
    naive overwrite would silently discard them. This returns True when
    the existing file has ANY top-level section not present in the
    shipped template we're about to write, or when it has substantially
    more nested content — signal to the caller to warn / require force
    before overwriting.
    """
    try:
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Existing is unreadable — safe to overwrite (we can't lose
        # info we can't read).
        return False
    if not isinstance(existing, dict):
        return False
    try:
        template = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Can't read template either — this is an error path handled by
        # _build_default_config's raise; treat as "assume rich" to be
        # conservative.
        return True
    if not isinstance(template, dict):
        return True
    def _real_keys(d: dict) -> set[str]:
        return {k for k in d if not k.startswith("_")}
    existing_keys = _real_keys(existing)
    template_keys = _real_keys(template)
    # Extra top-level sections the operator added → richer.
    if existing_keys - template_keys:
        return True
    # Non-empty models registry beyond what template ships is a common
    # customisation worth preserving.
    ex_models_raw = existing.get("models")
    tpl_models_raw = template.get("models")
    ex_models: dict = ex_models_raw if isinstance(ex_models_raw, dict) else {}
    tpl_models: dict = tpl_models_raw if isinstance(tpl_models_raw, dict) else {}
    if len(ex_models) > len(tpl_models):
        return True
    return False


def _idempotent_append(rc_path: Path, line: str) -> bool:
    """Append ``line`` to ``rc_path`` only when not already present.

    Returns True if the file was modified, False when the line was
    already there (or the file couldn't be read).
    """
    line_stripped = line.strip()
    try:
        existing = rc_path.read_text(encoding="utf-8") if rc_path.exists() else ""
    except OSError:
        return False
    for existing_line in existing.splitlines():
        if existing_line.strip() == line_stripped:
            return False
    try:
        with rc_path.open("a", encoding="utf-8") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write(line.rstrip("\n") + "\n")
    except OSError:
        return False
    return True


def _detect_shell_rc() -> Optional[Path]:
    """Return the user's shell rc file (`~/.bashrc` / `~/.zshrc` / …)."""
    shell = os.environ.get("SHELL", "")
    home = Path.home()
    if "zsh" in shell:
        return home / ".zshrc"
    if "bash" in shell:
        return home / ".bashrc"
    if "fish" in shell:
        return home / ".config" / "fish" / "config.fish"
    # Default: use .profile (works for sh + most shells)
    return home / ".profile"


# ---------------------------------------------------------------------------
# Phase 9 — teane doctor
# ---------------------------------------------------------------------------

def _run_harness_doctor(venv_path: Path, workspace: Path) -> tuple[int, str]:
    """Run `teane doctor` from inside the venv. Returns (exit_code, output)."""
    harness = _venv_harness(venv_path)
    if not harness.is_file():
        return 127, f"harness console script not found at {harness}"
    try:
        result = subprocess.run(
            [str(harness), "doctor", "-w", str(workspace)],
            capture_output=True, text=True, timeout=60,
        )
        return result.returncode, (result.stdout or "") + (result.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, "teane doctor timed out after 60s"
    except OSError as exc:
        return -1, f"teane doctor invocation failed: {exc}"


# ---------------------------------------------------------------------------
# Phase 10 — Optional tools
# ---------------------------------------------------------------------------

_SCANNER_COMMANDS: dict[str, dict[str, str]] = {
    "gitleaks": {
        "linux":   "sudo apt install gitleaks  # or download from https://github.com/gitleaks/gitleaks/releases",
        "wsl2":    "sudo apt install gitleaks  # or download from https://github.com/gitleaks/gitleaks/releases",
        "darwin":  "brew install gitleaks",
        "windows": "winget install gitleaks  # or `scoop install gitleaks`",
    },
    "bandit":  {p: "pip install bandit"  for p in ("linux", "wsl2", "darwin", "windows")},
    "semgrep": {p: "pip install semgrep" for p in ("linux", "wsl2", "darwin", "windows")},
    "trivy": {
        "linux":   "Install: https://aquasecurity.github.io/trivy/latest/getting-started/installation/",
        "wsl2":    "Install: https://aquasecurity.github.io/trivy/latest/getting-started/installation/",
        "darwin":  "brew install trivy",
        "windows": "winget install AquaSecurity.Trivy",
    },
}

_FORMATTER_COMMANDS: dict[str, dict[str, str]] = {
    "ruff (Python)":          {p: "pip install ruff" for p in ("linux", "wsl2", "darwin", "windows")},
    "prettier (JS/TS)":       {p: "npm install -g prettier" for p in ("linux", "wsl2", "darwin", "windows")},
    "gofmt (Go)":             {p: "bundled with Go — install Go from https://go.dev/dl/" for p in ("linux", "wsl2", "darwin", "windows")},
    "rustfmt (Rust)":         {p: "rustup component add rustfmt" for p in ("linux", "wsl2", "darwin", "windows")},
    "clang-format (C/C++)": {
        "linux":   "sudo apt install clang-format",
        "wsl2":    "sudo apt install clang-format",
        "darwin":  "brew install clang-format",
        "windows": "Install LLVM Windows: https://releases.llvm.org/",
    },
}


def _print_tool_commands(tools: dict[str, dict[str, str]], platform_id: str) -> None:
    for tool, by_platform in tools.items():
        cmd = by_platform.get(platform_id, by_platform.get("linux", "(no install command)"))
        print(f"      {_bold(tool):<24}  {cmd}")


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------

def _prompt(message: str, default: str, *, interactive: bool) -> str:
    if not interactive:
        return default
    suffix = f" [{default}]" if default else ""
    try:
        reply = input(f"    {message}{suffix}: ").strip()
    except EOFError:
        return default
    return reply or default


def _confirm(message: str, default: bool, *, interactive: bool) -> bool:
    if not interactive:
        return default
    yes_no = "Y/n" if default else "y/N"
    try:
        reply = input(f"    {message} [{yes_no}]: ").strip().lower()
    except EOFError:
        return default
    if not reply:
        return default
    return reply in ("y", "yes")


def _prompt_secret(message: str, *, interactive: bool) -> str:
    if not interactive:
        return ""
    try:
        return getpass.getpass(f"    {message}: ").strip()
    except EOFError:
        return ""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Interactive bootstrap for teane.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--venv", default=DEFAULT_VENV,
                        help=f"Path to the venv directory (default: {DEFAULT_VENV})")
    parser.add_argument("--dev", action="store_true",
                        help="Install the [dev] extras (for contributors)")
    parser.add_argument("--provider",
                        choices=sorted(DEFAULT_MODELS_BY_PROVIDER.keys()),
                        default=None,
                        help="Preset LLM provider (skips the wizard prompt)")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Skip all prompts and use defaults; fail fast on missing required tools")
    parser.add_argument("--no-doctor", action="store_true",
                        help="Skip the final `teane doctor` invocation")
    parser.add_argument(
        "--overwrite-config", action="store_true",
        help=(
            "Force overwrite of an existing config/config.json even "
            "when it carries custom sections (extra models, custom "
            "skills, tuned mcp.servers). Without this flag, setup "
            "refuses to overwrite richer configs to avoid clobbering "
            "operator customisations."
        ),
    )
    args = parser.parse_args(argv)

    interactive = not args.non_interactive
    venv_path = Path(os.path.expanduser(args.venv)).resolve()

    print(_bold(_green("=" * 64)))
    print(_bold(_green("  teane setup")))
    print(_bold(_green("=" * 64)))
    print()

    # ---- Phase 1: Platform ------------------------------------------------
    _banner(1, "Platform detection")
    platform_id = _detect_platform()
    _ok(f"Platform: {platform_id}")

    # ---- Phase 2: Python --------------------------------------------------
    _banner(2, "Probing Python 3.14+")
    python314 = _find_python314()
    if not python314:
        _fail("Python 3.14+ not found on PATH.")
        _info(f"Install: {_install_command_for('python3.14', platform_id)}")
        return 2
    _ok(f"Python 3.14+ found: {python314}")

    # ---- Phase 3: git + sqlite -------------------------------------------
    _banner(3, "Probing git and sqlite3")
    git_path = _probe_git()
    if git_path:
        _ok(f"git: {git_path}")
    else:
        _fail("git not found.")
        _info(f"Install: {_install_command_for('git', platform_id)}")
        return 3
    sqlite_path = _probe_sqlite3()
    if sqlite_path:
        _ok(f"sqlite3: {sqlite_path}")
    else:
        _warn(f"sqlite3 CLI not on PATH (Python's sqlite3 module still works). "
              f"To install the CLI: {_install_command_for('sqlite3', platform_id)}")

    # ---- Phase 4: Sandbox backend ----------------------------------------
    _banner(4, "Probing sandbox backend")
    docker_path = _probe_docker()
    has_unshare = _probe_unshare() if platform_id in ("linux", "wsl2") else False
    if docker_path:
        _ok(f"Docker daemon reachable: {docker_path}")
    if has_unshare:
        _ok("unshare --user works (Linux namespaces available)")
    if not docker_path and not has_unshare:
        _warn("Neither Docker nor unshare is available — the harness will need "
              "HARNESS_ALLOW_UNSAFE_SANDBOX=true to run with no isolation.")
        _info(f"Install Docker: {_install_command_for('docker', platform_id)}")

    # ---- Phase 5: Build toolchain -----------------------------------------
    _banner(5, "Probing build toolchain")
    if _probe_build_toolchain(platform_id):
        _ok("Build toolchain found (tree-sitter source-build fallback ready)")
    else:
        _warn(f"No build toolchain detected. tree-sitter usually has prebuilt "
              f"wheels for Python 3.14+ — if `pip install` fails on a wheel "
              f"build, run: {_install_command_for('build-toolchain', platform_id)}")

    # ---- Phase 6: Venv ----------------------------------------------------
    _banner(6, "Creating venv")
    if venv_path.exists():
        reuse = _confirm(f"Venv already exists at {venv_path}. Reuse it?",
                         default=True, interactive=interactive)
        if not reuse:
            _fail("Aborting — please remove the venv or pass --venv with a different path.")
            return 6
        _ok(f"Reusing existing venv: {venv_path}")
    else:
        venv_path.parent.mkdir(parents=True, exist_ok=True)
        cmd: list[str] = python314.split() + ["-m", "venv", str(venv_path)]
        print(f"    Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except (subprocess.TimeoutExpired, OSError) as exc:
            _fail(f"venv creation failed: {exc}")
            return 6
        if result.returncode != 0:
            _fail(f"venv creation failed (exit {result.returncode}):")
            _info(result.stderr.strip()[:500])
            return 6
        _ok(f"Venv created at {venv_path}")

    # ---- Phase 7: pip install --------------------------------------------
    _banner(7, "Installing the harness Python package")
    if not _run_pip_install(venv_path, dev=args.dev):
        _fail("pip install failed.")
        return 7
    _ok("Package installed.")

    # ---- Phase 8: Config wizard ------------------------------------------
    _banner(8, "Configuring LLM provider")
    catalogue_keys = _load_model_catalogue()
    if args.provider:
        provider = args.provider
        _info(f"Using preset provider: {provider}")
    else:
        provider = _prompt(
            "Provider [anthropic|openai|deepseek|ollama]",
            default="anthropic",
            interactive=interactive,
        ).lower()
    if provider not in DEFAULT_MODELS_BY_PROVIDER:
        _fail(f"Unknown provider {provider!r}. Choose one of: "
              f"{sorted(DEFAULT_MODELS_BY_PROVIDER)}")
        return 8
    model_key = DEFAULT_MODELS_BY_PROVIDER[provider]
    if catalogue_keys and model_key not in catalogue_keys:
        _warn(f"Default model {model_key} not in catalogue. "
              f"Falling back to first {provider}: entry.")
        catalogue_match = next(
            (k for k in sorted(catalogue_keys) if k.startswith(f"{provider}:")),
            None,
        )
        if catalogue_match:
            model_key = catalogue_match
    _ok(f"Model: {model_key}")

    # API key (skip for ollama)
    env_var = PROVIDER_ENV_VAR[provider]
    if env_var:
        existing = os.environ.get(env_var, "")
        if existing:
            _ok(f"{env_var} already set in current environment.")
            api_key = existing
        else:
            api_key = _prompt_secret(
                f"Enter {env_var} (hidden; press Enter to skip)",
                interactive=interactive,
            )
            if not api_key:
                _warn(f"No {env_var} provided. The harness will fail at the first LLM call "
                      f"until you set it in your environment.")
            else:
                # Offer to persist
                if platform_id in ("linux", "wsl2", "darwin"):
                    persist = _confirm(
                        f"Persist {env_var} to your shell rc file?",
                        default=True, interactive=interactive,
                    )
                    if persist:
                        rc_path = _detect_shell_rc()
                        if rc_path:
                            line = f'export {env_var}="{api_key}"'
                            if _idempotent_append(rc_path, line):
                                _ok(f"Appended {env_var} export to {rc_path}")
                                _info("Open a new shell or run `source` to pick it up.")
                            else:
                                _ok(f"{env_var} export already present in {rc_path}")
                else:
                    _info(f"On Windows native: run `setx {env_var} \"...\"` in a new "
                          f"terminal to persist this across sessions.")
    else:
        api_key = ""

    # Write <root>/config/config.json — always the FULL shipped template
    # with model_routing patched to the operator's chosen provider. Never
    # a bare stub; see _build_default_config docstring for why.
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_FILE.exists():
        # Safety: when the existing file has extra top-level sections or
        # a richer models registry than the shipped template, refuse to
        # overwrite by default — the operator has customisations we
        # would silently lose.
        richer = _config_is_richer_than_default(CONFIG_FILE)
        if richer and not getattr(args, "overwrite_config", False):
            _warn(
                f"{CONFIG_FILE} has custom sections not present in the "
                f"shipped template — overwriting would DROP those. "
                f"Keeping existing config; re-run with --overwrite-config "
                f"to force."
            )
        else:
            overwrite = _confirm(
                f"{CONFIG_FILE} already exists. Overwrite (only "
                f"model_routing.*_primary changes; every other section "
                f"is preserved from the shipped template)?",
                default=False, interactive=interactive,
            )
            if not overwrite:
                _ok(f"Keeping existing {CONFIG_FILE}.")
            else:
                CONFIG_FILE.write_text(
                    json.dumps(_build_default_config(provider, model_key), indent=2),
                    encoding="utf-8",
                )
                _ok(f"Wrote {CONFIG_FILE}")
    else:
        CONFIG_FILE.write_text(
            json.dumps(_build_default_config(provider, model_key), indent=2),
            encoding="utf-8",
        )
        _ok(f"Wrote {CONFIG_FILE}")

    # ---- Phase 8.5: harness-builder sandbox image ----------------------
    # The default sandbox.docker_image is `harness-builder:latest`,
    # a custom local build (harness/vendor/Dockerfile.builder) bundling
    # JDK 21 + Node 20 + Python + uv + Playwright. It's not on any
    # registry, so first sandbox run fails with "image not found" if
    # the operator hasn't built it. Doctor now catches this, but it's
    # cheaper (in operator time) to build now than to hit the error on
    # first teane run.
    _banner(85, "Harness-builder sandbox image")
    dockerfile = REPO_ROOT / "harness" / "vendor" / "Dockerfile.builder"
    if shutil.which("docker") is None:
        _info("Docker not on PATH — skipping harness-builder image build. "
              "Install Docker Engine to enable the default sandbox.")
    elif not dockerfile.is_file():
        _info(f"Dockerfile.builder not shipped at {dockerfile} — skipping.")
    else:
        try:
            inspect = subprocess.run(
                ["docker", "image", "inspect", "harness-builder:latest"],
                capture_output=True, text=True, timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            inspect = None
            _warn(f"Could not probe existing image: {exc}")
        if inspect is not None and inspect.returncode == 0:
            _ok("harness-builder:latest already built locally.")
        else:
            build_now = _confirm(
                "harness-builder:latest is not built locally. Build "
                "it now (takes ~5-10 minutes; downloads ~2GB)?",
                default=False, interactive=interactive,
            )
            if build_now:
                _info("Running `docker build` — output streams below…")
                try:
                    proc = subprocess.run(
                        [
                            "docker", "build", "--pull",
                            "-f", str(dockerfile),
                            "-t", "harness-builder:latest",
                            str(dockerfile.parent),
                        ],
                        timeout=1800,  # 30 minutes hard cap
                    )
                    if proc.returncode == 0:
                        _ok("harness-builder:latest built successfully.")
                    else:
                        _warn(
                            f"docker build exited {proc.returncode}. "
                            f"Re-run manually: `docker build --pull -f "
                            f"{dockerfile} -t harness-builder:latest "
                            f"{dockerfile.parent}`"
                        )
                except (subprocess.TimeoutExpired, OSError) as exc:
                    _warn(f"docker build failed: {exc}")
            else:
                _info(
                    "Skipped. Build it later with: `docker build --pull "
                    f"-f {dockerfile} -t harness-builder:latest "
                    f"{dockerfile.parent}` — or override "
                    "sandbox.docker_image in config to a public image "
                    "(e.g. python:3.12-slim)."
                )

    # ---- Phase 9: teane doctor -----------------------------------------
    _banner(9, "Running teane doctor")
    if args.no_doctor:
        _ok("Skipped per --no-doctor.")
    else:
        # Use REPO_ROOT as the workspace so the git-repo check passes.
        workspace = REPO_ROOT
        # Pass the API key forward so doctor sees it even before the operator
        # opens a new shell.
        doctor_env = dict(os.environ)
        if env_var and api_key:
            doctor_env[env_var] = api_key
        harness = _venv_harness(venv_path)
        if not harness.is_file():
            _warn("harness console script not on disk — skipping doctor.")
        else:
            try:
                proc = subprocess.run(
                    [str(harness), "doctor", "-w", str(workspace)],
                    capture_output=True, text=True, timeout=60,
                    env=doctor_env,
                )
                print(proc.stdout)
                if proc.stderr.strip():
                    print(proc.stderr)
                if proc.returncode == 0:
                    _ok("teane doctor passed.")
                else:
                    _warn(f"teane doctor exited {proc.returncode}. "
                          "Re-check the failing rows above and consult "
                          "docs/installation.md §13.")
            except (subprocess.TimeoutExpired, OSError) as exc:
                _warn(f"Could not invoke teane doctor: {exc}")

    # ---- Phase 10: Optional tools -----------------------------------------
    _banner(10, "Optional tools")
    want_scanners = _confirm(
        "Print install commands for security scanners (gitleaks / bandit / semgrep / trivy)?",
        default=False, interactive=interactive,
    )
    if want_scanners:
        print()
        print("    Run any of these you want; missing scanners degrade gracefully:")
        _print_tool_commands(_SCANNER_COMMANDS, platform_id)
    want_formatters = _confirm(
        "Print install commands for language formatters (ruff / prettier / gofmt / rustfmt / clang-format)?",
        default=False, interactive=interactive,
    )
    if want_formatters:
        print()
        print("    Install only the ones for languages you target:")
        _print_tool_commands(_FORMATTER_COMMANDS, platform_id)

    # ---- Phase 11: Summary ------------------------------------------------
    _banner(11, "Setup complete")
    activation = _activation_command(venv_path, platform_id)
    print()
    print(_bold("Next steps:"))
    print(f"  1. Activate the venv:   {_green(activation)}")
    smoke_cmd = _green('teane run -w <workspace> -p "<task>"')
    print(f"  2. Smoke run:           {smoke_cmd}")
    print(f"  3. Re-verify:           {_green('teane doctor')}")
    print()
    print(_bold("Reference docs:"))
    print("  - docs/installation.md       — full install guide (every step in detail)")
    print("  - docs/SPEC_REQUIREMENTS.md  — full config schema")
    print("  - README.md                  — command reference + troubleshooting")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
