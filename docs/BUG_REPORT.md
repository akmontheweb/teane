# AI Agent Harness — Code Review Bug Report

**Date:** 2026-06-08  
**Reviewer:** Automated Code Review  
**Scope:** All 18 source modules (~13,000 lines) + main test file (2,859 lines)  
**Commit:** `55addce3f16e717507fe66c5dbe69ebc7a8ff40f`

---

## Executive Summary

A full review of the codebase identified **8 bugs** across multiple modules. Three are classified as **Critical** (security-impacting logic errors or workspace-boundary violations), two as **Medium** (functional breakage in specific conditions), and three as **Minor** (code quality / robustness issues).

Each finding below includes the file, line ranges, affected code, root cause analysis, impact assessment, and a recommended fix.

---

## Summary Table

| # | Severity | File | Lines | Title |
|---|----------|------|-------|-------|
| 1 | 🔴 Critical | `harness/security.py` | 669–679 | HITLGate CI auto-approve logic is inverted |
| 2 | 🔴 Critical | `harness/lintgate.py` | 523–528 | `_resolve_path` accepts arbitrary absolute paths |
| 3 | 🔴 Critical | `harness/deploy.py` | 728 vs 870 | Dockerfile naming mismatch between generation and compose |
| 4 | 🟡 Medium | `harness/impact.py` | 218–219 | Tree-sitter extraction only supports Python |
| 5 | 🟡 Medium | `harness/deploy.py` | 937 | Uses deprecated `docker-compose` binary |
| 6 | 🟢 Minor | `harness/cli.py` | 1038–1040 | Unsafe `asyncio.get_event_loop().run_until_complete()` |
| 7 | 🟢 Minor | `harness/cli.py` | 915–923 | TOCTOU race condition in `_read_spec_file` |
| 8 | 🟢 Minor | `harness/graph.py` | 278–304 | Silent failure when workspace root is unreachable |

---

## Bug 1 — HITLGate CI Auto-Approve Logic Is Inverted

- **Severity:** 🔴 Critical (Security)
- **File:** `harness/security.py`
- **Lines:** 669–679
- **Function:** `HITLGate.prompt_approval()`

### Current Code

```python
def prompt_approval(self, matches, llm_content="", context=""):
    ...
    if self._is_ci_environment():
        if self.auto_approve_in_ci:
            logger.warning(
                "[hitl_gate] CI environment detected. %d sensitive pattern(s) blocked: %s",
                ...
            )
            return False  # Block in CI
        else:
            logger.info("[hitl_gate] CI detected but auto_approve_in_ci=False. Proceeding without prompt.")
            return True   # <--- AUTO-APPROVES — THIS IS WRONG
```

### Root Cause

The boolean is inverted. When `auto_approve_in_ci=True`, the gate **blocks** (returns `False`). When `auto_approve_in_ci=False`, the gate **silently approves** (returns `True`). The docstring says the opposite: "*In CI environments with auto_approve_in_ci=True, always returns False (blocks by default in non-interactive mode) to prevent unattended sensitive operations.*"

### Impact

In any CI/CD pipeline that sets `CI=true` but does NOT also set `auto_approve_in_ci=True`, ALL sensitive operations (git push, terraform apply, database migrations, destructive rm -rf, etc.) will be **silently auto-approved** with no human review. This is the default `GatewayConfig` state (`auto_approve_in_ci=True`), so the bug is dormant with default config but activates when a user sets `auto_approve_in_ci: false` in their config — expecting it to always block, but getting the opposite.

### Recommended Fix

Swap the `return` values:

```python
if self.auto_approve_in_ci:
    logger.info("[hitl_gate] CI detected with auto_approve_in_ci=True. Proceeding without prompt.")
    return True   # User explicitly opted into CI auto-approval
else:
    logger.warning(
        "[hitl_gate] CI environment detected with auto_approve_in_ci=False. "
        "%d sensitive pattern(s) blocked.", len(matches), ...
    )
    return False  # Block — no interactive prompt available in CI
```

Also update the docstring to match.

---

## Bug 2 — `_resolve_path` Accepts Arbitrary Absolute Paths

- **Severity:** 🔴 Critical (Security)
- **File:** `harness/lintgate.py`
- **Lines:** 523–528
- **Function:** `_resolve_path()`

### Current Code

```python
def _resolve_path(filepath: str, workspace_path: str) -> Optional[str]:
    """Resolve a filepath against the workspace."""
    if os.path.isabs(filepath):
        return filepath if os.path.exists(filepath) else None
    full = os.path.join(workspace_path, filepath)
    return full if os.path.exists(full) else None
```

### Root Cause

This function accepts ANY absolute path that exists on the host filesystem, with no workspace-boundary check. Compare with `patcher.py` which uses `safe_resolve()` from `harness/trust.py` that rejects absolute paths, parent-traversal, and symlink escapes.

### Impact

An LLM that generates a patch targeting `/etc/passwd` or `/home/user/.ssh/id_rsa` could cause lintgate to run formatters and linters against system files outside the workspace. While the formatters would only read these files (not write), a malicious build command combined with this path leak could expose sensitive file contents.

### Recommended Fix

Use the central `safe_resolve` from `harness/trust.py`:

```python
from harness.trust import safe_resolve as _safe_resolve

def _resolve_path(filepath: str, workspace_path: str) -> Optional[str]:
    """Resolve a filepath against the workspace with boundary protection."""
    try:
        resolved = _safe_resolve(workspace_path, filepath)
        return resolved if os.path.exists(resolved) else None
    except ValueError:
        return None
```

---

## Bug 3 — Dockerfile Naming Mismatch Between Generation and Compose

- **Severity:** 🔴 Critical (Functional)
- **File:** `harness/deploy.py`
- **Lines:** 728 (compose) vs 870 (generation)
- **Functions:** `_generate_compose_file()` and `generate_assets_from_blueprint()`

### Current Code

**Generation** (`generate_assets_from_blueprint`, line ~870):
```python
# First service gets plain "Dockerfile", others get "Dockerfile.<name>"
dockerfile_name = f"Dockerfile.{svc_name}" if svc_name != list(services.keys())[0] else "Dockerfile"
```

**Compose** (`_generate_compose_file`, line 728):
```python
# Uses build_context != "." as the distinguisher — unrelated to the naming logic above
lines.append(f"      dockerfile: Dockerfile.{svc_name}" if svc_spec.get("build_context", ".") != "." else "      dockerfile: Dockerfile")
```

### Root Cause

Two different heuristics decide the Dockerfile name. The generation pass uses **service ordering** (first vs subsequent), while the compose pass uses **build_context value** (`"."` vs something else). These can diverge. For example, a first service with `build_context: "./api"` gets `Dockerfile` on disk but `docker-compose.yml` references `Dockerfile.api`.

### Impact

Docker Compose will fail with "file not found" when the naming heuristics disagree, blocking deployment entirely.

### Recommended Fix

Unify on one heuristic. Either:
1. Name all Dockerfiles as `Dockerfile.<service_name>` (simplest, always consistent)
2. Or pass the generated filename mapping from `generate_assets_from_blueprint()` into `_generate_compose_file()`.

Option 1 is simplest — change line 870 to always use `Dockerfile.{svc_name}`:
```python
dockerfile_name = f"Dockerfile.{svc_name}"
```
And update compose to always use the same pattern.

---

## Bug 4 — Tree-Sitter Extraction Only Supports Python

- **Severity:** 🟡 Medium (Functional)
- **File:** `harness/impact.py`
- **Lines:** 218–219
- **Function:** `_try_tree_sitter_extract()`

### Current Code

```python
def _try_tree_sitter_extract(self, filepath, source, lang, symbols):
    try:
        import tree_sitter_python
        grammar_map: dict[str, Any] = {
            "python": tree_sitter_python,
        }
        ...
        grammar_module = grammar_map.get(lang)
        if grammar_module is None:
            return False  # silently falls back to regex
```

### Root Cause

`grammar_map` hardcodes only `{"python": tree_sitter_python}`. There is no attempt to dynamically import `tree_sitter_rust`, `tree_sitter_typescript`, `tree_sitter_go`, `tree_sitter_c`, etc. for the other languages listed in `_EXTENSION_TO_TREE_SITTER` (Rust, TypeScript, JavaScript, Go, C/C++, Java). Every non-Python file silently falls back to regex-based symbol extraction.

### Impact

For non-Python codebases, the dependency graph builder never uses AST-level accuracy. Import/symbol detection is purely regex-based, which misses aliased imports (e.g., `import numpy as np`), re-exports, and complex module structures. Impact analysis results are less reliable.

### Recommended Fix

Extend `grammar_map` with dynamic imports:

```python
grammar_map: dict[str, dict] = {"python": tree_sitter_python}
# Try to import grammars for other registered languages
_lang_packages = {
    "rust": "tree_sitter_rust",
    "typescript": "tree_sitter_typescript",
    "tsx": "tree_sitter_typescript",
    "javascript": "tree_sitter_javascript",
    "go": "tree_sitter_go",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
    "java": "tree_sitter_java",
}
if lang in _lang_packages:
    try:
        grammar_module = __import__(_lang_packages[lang], fromlist=["language"])
        grammar_map[lang] = grammar_module
    except ImportError:
        return False  # fall through to regex
```

---

## Bug 5 — Uses Deprecated `docker-compose` Binary

- **Severity:** 🟡 Medium (Functional)
- **File:** `harness/deploy.py`
- **Line:** 937
- **Function:** `_get_compose_services()`

### Current Code

```python
proc = await asyncio.create_subprocess_exec(
    "docker-compose", "-f", compose_path, "config", "--services",
    ...
)
```

### Root Cause

Calls the legacy `docker-compose` (v1, Python-based) binary. Docker Compose V2 (`docker compose`, a Go plugin) has been the default since Docker Desktop 4.4+ / Docker Engine 20.10+. Many modern installations no longer ship the `docker-compose` alias.

### Impact

On hosts without the legacy binary, `_get_compose_services()` silently fails (returns `[]`), and `health_check_loop()` returns `{"success": True, "healthy": [], ...}` because no services were detected — a false success that skips all health checks.

### Recommended Fix

Use `docker compose` (plugin form) with fallback:

```python
# Try "docker compose" first (V2), fall back to "docker-compose" (V1)
for compose_cmd in (["docker", "compose"], ["docker-compose"]):
    if shutil.which(compose_cmd[0]) if len(compose_cmd) == 1 else shutil.which(compose_cmd[1]):
        break
proc = await asyncio.create_subprocess_exec(
    *compose_cmd, "-f", compose_path, "config", "--services",
    ...
)
```

---

## Bug 6 — Unsafe Event Loop Pattern in `interactive_review_loop`

- **Severity:** 🟢 Minor (Reliability)
- **File:** `harness/cli.py`
- **Lines:** 1038–1040
- **Function:** `interactive_review_loop()`

### Current Code

```python
try:
    updated = asyncio.get_event_loop().run_until_complete(
        _refine_requirements(spec_path, notes, gateway)
    )
```

### Root Cause

`asyncio.get_event_loop().run_until_complete()` raises `RuntimeError: This event loop is already running` when called from within an async context. `interactive_review_loop` is a synchronous function called from async `cmd_run()`. If the event loop is already running (which it is after any `await` call), this pattern fails.

### Impact

The requirement refinement "Refine" option crashes at runtime if invoked after the event loop has started. The user sees a cryptic error instead of getting their spec refined.

### Recommended Fix

Use `asyncio.run()` which always creates a fresh event loop:

```python
try:
    updated = asyncio.run(
        _refine_requirements(spec_path, notes, gateway)
    )
```

Or better, refactor `interactive_review_loop` to be an async function and use `await`.

---

## Bug 7 — TOCTOU Race Condition in `_read_spec_file`

- **Severity:** 🟢 Minor (Reliability)
- **File:** `harness/cli.py`
- **Lines:** 915–923
- **Function:** `_read_spec_file()`

### Current Code

```python
def _read_spec_file(spec_path: str) -> str:
    """Read a specification file from disk."""
    if not os.path.isfile(spec_path):
        return ""
    try:
        with open(spec_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""
```

### Root Cause

A time-of-check-to-time-of-use (TOCTOU) race: the file is checked with `os.path.isfile()` and then opened. Between the check and the open, the file could be deleted or replaced (e.g., by an external process during manual edit mode).

### Impact

In the unlikely race window, an uncaught `FileNotFoundError` or `PermissionError` would propagate and crash the interactive review loop instead of being handled gracefully.

### Recommended Fix

Rely on the try/except alone:

```python
def _read_spec_file(spec_path: str) -> str:
    try:
        with open(spec_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""
```

---

## Bug 8 — Silent Failure When Workspace Root Is Unreachable

- **Severity:** 🟢 Minor (Observability)
- **File:** `harness/graph.py`
- **Lines:** 278–304
- **Function:** `_snapshot_directory_tree()`

### Current Code

```python
def _snapshot_directory_tree(path, max_depth=4, max_files_per_dir=50):
    lines: list[str] = []
    try:
        for root, dirs, files in os.walk(path):
            ...
    except (OSError, PermissionError) as exc:
        lines.append(f"[Error reading directory: {exc}]")
    return "\n".join(lines)
```

### Root Cause

If `os.walk()` fails on the very first iteration (workspace root unreadable, broken symlink, permission denied after a `chmod`), the function returns a single error line: `"[Error reading directory: ...]"`. This string is injected directly into the system prompt at `messages[0]` as the "directory structure snapshot" with no downstream warning.

### Impact

The LLM receives a system prompt claiming the repository root is an unreadable error, but has no context that the tree is incomplete. It may still attempt to generate patches for a codebase it cannot see, leading to hallucinated file paths and wasted token budget.

### Recommended Fix

Log a warning and return a more descriptive fallback:

```python
except (OSError, PermissionError) as exc:
    lines.append(f"[Error reading directory: {exc}]")
    logger.warning("[graph] Could not snapshot directory tree at %s: %s", path, exc)
return "\n".join(lines) if lines else f"[Unable to read directory structure at {path}]"
```

---

## Appendix: Review Methodology

1. Read all 18 source files in `harness/` end-to-end (~13,000 lines)
2. Read `tests/test_harness.py` (2,859 lines) for regression coverage context
3. Focused on: path traversal guards, authentication/authorization logic, input validation boundaries, async safety, error handling completeness, and configuration/CLI interface contracts
4. Cross-referenced between modules (e.g., comparing `lintgate._resolve_path` against `patcher.safe_resolve`, comparing `deploy.generate_assets_from_blueprint` against `deploy._generate_compose_file`)