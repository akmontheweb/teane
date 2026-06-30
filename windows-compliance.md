# Windows Compatibility Audit — teane

The codebase already has solid Windows abstractions in place (`harness/_platform.py`, `harness/_filelock.py`, Windows-aware `harness/preflight.py`, `_docker_mount_path` for `C:\` translation, `taskkill /T /F`, `CREATE_NEW_PROCESS_GROUP`, `Scripts/python.exe` in `scripts/setup.py`). Findings below are remaining gaps.

## BLOCKERS

None — `BareBackend` is the non-Docker fallback on Windows; `UnshareBackend` is correctly gated on `is_linux()`.

## LIKELY BREAKS

- ~~`harness/schedule.py:755` — bare `os.getpgid(proc.pid)`~~ → Re-checked: site is already inside `if hasattr(os, "getpgid"):` (line 753). Audit was wrong.
- ~~`harness/sandbox.py:1609` — bare `os.getpgid(proc.pid)`~~ → Re-checked: site is already inside `if hasattr(os, "getpgid"):` (line 1607). Audit was wrong.
- ~~`harness/cli.py:4835` — `_sync_kill_mcp_subprocesses` calls `os.killpg(os.getpgid(...))` unguarded~~ → FIXED: early-return on `not hasattr(_os, "killpg")`.
- ~~`harness/playwright_gen.py:346` — `_chromium_cache_present` checks only `~/.cache/ms-playwright`~~ → FIXED: now probes both `~/.cache/ms-playwright` and `%LOCALAPPDATA%\ms-playwright` via new `_chromium_cache_dirs()`.
- ~~`harness/playwright_gen.py:372` — bare `["npx", ...]` doesn't resolve `.cmd` on Windows~~ → FIXED: `cmd` now uses `shutil.which("npx")` so the `.cmd`/`.exe` shim is picked up.
- ~~`Makefile:12,15` — hardcoded `python3`/`pytest`; operators must use `scripts/setup.py` directly~~ → ALREADY DOCUMENTED: `docs/installation.md` §5 "Make-free workflows (Windows native)" maps every Makefile target to a direct `python scripts\...` invocation.

## MINOR

- `harness/parser_registry.py:190-191` — detects site-packages via `os.sep + "lib" + os.sep + "python"`; Windows uses `\Lib\site-packages\` (no `python`), so detection misses. Cosmetic, affects internal filtering.
- Many tests hardcode `"/tmp/..."` as opaque workspace strings — not touching the FS, so they pass.
- `harness/sandbox.py:264-294` UnshareBackend, `harness/sandbox.py:702` `os.getuid/getgid`, `harness/sandbox.py:877-885` `find -uid 0` — all correctly gated on `is_linux()`.

## OK (portable)

- All `os.killpg`/`os.getpgid` sites in `harness/mcp_client.py`, `harness/_platform.py`, `harness/web_state.py`, `harness/sandbox.py:_kill_process_group` are guarded by `hasattr(os, "killpg")` and route through `_platform.kill_process_tree` → `taskkill /T /F`.
- `harness/_filelock.py` dispatches `fcntl` ↔ `msvcrt`.
- `harness/preflight.py` has Windows probes (taskkill, POSIX sh, LongPathsEnabled).
- `scripts/setup.py:288-308` handles `Scripts/python.exe` and `Activate.ps1`.
- `harness/observability.py` uses `os.path.expanduser("~/.harness/...")` — `~` resolves to `USERPROFILE` on Windows.
- Container-internal POSIX paths (`/tmp/builder-home`, `/tmp/teane-venv`, Dockerfile templates) are correct — the container is Linux.
- Trust/security denylists referencing `/bin/sh`, `/etc/`, `chmod 777` etc. are LLM-output pattern matching, not host FS access.

## Bottom line

Teane should run on Windows for the common paths (CLI, Docker-backed builds, scheduling). All four fixes from the audit are now in place — two by edits in this round (`cli.py` `killpg` guard + `playwright_gen.py` cache-dir/`npx` resolution), two were already in tree on re-check (`schedule.py` / `sandbox.py` `getpgid` calls are guarded; `docs/installation.md` already documents the Make-free Windows entry points).
