---
applies_to: [python]
---

## Python Style Guide

### Source
- PEP 8 — Style Guide for Python Code (https://peps.python.org/pep-0008/)
- Google Python Style Guide (https://google.github.io/styleguide/pyguide.html)

### Philosophy
Code is read far more often than it is written. Optimize for the reader. Prefer the obvious solution. Conform to PEP 8 and PEP 257 (docstrings) by default; deviate only when the surrounding file consistently does, and follow the file's existing convention rather than introducing a second one.

### Layout & formatting
- 4-space indentation. No tabs. No trailing whitespace.
- Maximum line length 79 chars (PEP 8) or 80 (Google). Long string literals may use implicit concatenation across lines.
- Two blank lines between top-level functions/classes; one blank line between methods.
- Imports: stdlib first, then third-party, then local — each group separated by a blank line, each group alphabetically sorted. Use absolute imports; reserve relative imports for intra-package use.
- Never use wildcard `from x import *` except in `__init__.py` re-exports.
- Surround binary operators with single spaces; no space around `=` for keyword args or default values.

### Naming
- `lower_snake_case` for functions, methods, variables, modules.
- `CapWords` for classes and type variables.
- `UPPER_SNAKE_CASE` for module-level constants.
- Single trailing underscore (`class_`) to avoid clashing with built-ins; never use a single leading underscore as a public-API marker.
- A leading underscore signals "module-private"; two leading underscores trigger name mangling — use sparingly.

### Strings & types
- Prefer f-strings for interpolation; `%`-formatting only for logging where the lazy evaluation matters (`logger.info("got %s", x)`).
- Use `str.startswith()` / `str.endswith()` over slicing.
- Add type hints to all public function signatures (Google guide requires it; PEP 8 strongly recommends it). Use `from __future__ import annotations` to defer evaluation when forward references would otherwise need quoting.

### Errors & control flow
- Catch the narrowest exception that makes sense; bare `except:` is forbidden — use `except Exception:` at minimum.
- Don't suppress an exception by `except Foo: pass` unless you also leave a comment explaining why. Re-raise with `raise` (not `raise e`) to preserve the traceback.
- Prefer EAFP (try/except) over LBYL (if/check/then) when the check race is a real concern.
- Use context managers (`with open(...) as f:`) for any resource that needs deterministic cleanup.

### Functions & docstrings
- Triple-double-quoted docstrings (`"""..."""`) for every public module, class, and function. One-line summary in the imperative mood; if it's multi-line, blank line after the summary, then details (Google "Args / Returns / Raises" sections are recommended).
- Default argument values must not be mutable (`def f(x=[])` is a bug). Use `None` and create the mutable inside.
- Avoid more than ~5 positional args; force keyword args with `*` after that.

### Datetime & timezones
One convention end-to-end. Mixed naive/aware causes silent bugs.
- Import: `from datetime import datetime, timezone`; use `timezone.utc`. NEVER `datetime.UTC` (3.11+ only, often absent in sandboxes).
- Now: `datetime.now(timezone.utc)`. NEVER `datetime.utcnow()` (naive, deprecated 3.12) or `datetime.now()` (local, naive).
- Wire/storage: ISO 8601 via `.isoformat()`, parse `datetime.fromisoformat()`.
- DB: SQLAlchemy `DateTime(timezone=True)`, Postgres `TIMESTAMPTZ`, SQLite `TEXT` (ISO 8601). Never store naive.
- Aware-vs-naive `TypeError`: fix at the source (parser / ORM / boundary), not each callsite.
- Mocks: patch at the CALLING module, return aware values.

### Filesystem paths (Linux / macOS / Windows)
Cross-platform means `pathlib.Path` end-to-end. String path arithmetic breaks silently on Windows (backslash vs forward slash).
- Use `pathlib.Path` for every new path. Do NOT mix `os.path.join` and `Path`/`/` in the same module.
- Never build paths by string concatenation (`base + "/" + name`) — always `Path(base) / name`.
- Cast at boundaries: `Path(os.environ["FOO"])`, `Path(cfg["dir"])`. Downstream code sees `Path`, never `str`.
- `Path.resolve()` returns absolute AND resolves symlinks. `Path.absolute()` does NOT resolve symlinks (subtle — pick deliberately).
- Compare as `Path`, not `str`. `Path("/a/b") == "/a/b"` is False; on Windows `"C:\\a" != "c:/a"` even for the same file.
- Tests: use pytest's `tmp_path` fixture (returns `Path`). Do NOT hardcode `/tmp/...` — Windows CI has no `/tmp`.
- Reading files: `path.read_text(encoding="utf-8")` beats `open(path, "r").read()`. Always name the encoding explicitly.

### Concurrency & async
Sync and async locking primitives are NOT interchangeable — pick the right one for the caller.
- `threading.Lock` and `multiprocessing.Lock` support `with x:` (sync context manager) ONLY. Using `async with threading.Lock():` is a bug — no `__aenter__`, silent misbehaviour or runtime `AttributeError` depending on Python version. If a coroutine needs a lock, use `asyncio.Lock`.
- `asyncio.Lock`, `asyncio.Semaphore`, `asyncio.Event` support `async with` only. Do NOT use `with asyncio.Lock():`.
- Never call blocking primitives (`time.sleep`, `requests.get`, `queue.Queue.get`) inside a coroutine — they stall the event loop. Use `await asyncio.sleep`, `httpx.AsyncClient`, `asyncio.Queue`.
- Module-level `asyncio.Lock()` is safe on Python 3.10+ (uses the running loop lazily), but test frameworks that spin fresh loops per test (pytest-asyncio default) will orphan any lock still held from a prior test. Provide a `reset_*()` helper that clears cached locks and call it from an `autouse` fixture.
- CPU-bound work in async code: offload to `asyncio.to_thread(...)`; never `time.sleep` or a busy loop.
- Mixing threads and asyncio in one process: schedule cross-boundary work via `loop.call_soon_threadsafe` or `asyncio.run_coroutine_threadsafe` — never mutate coroutine state from a thread directly.

### Packages & `__init__.py`
Every new submodule needs its package init to exist BEFORE the module lands, otherwise imports break the whole tree.
- When creating `pkg/mod.py`, first ensure `pkg/__init__.py` exists — CREATE_FILE it (empty is fine) if not.
- Do not `from .sibling import X` if `sibling.py` was created this same round without its own path being verified.
- Namespace packages (no `__init__.py`) are only correct when the WHOLE project chooses them; do not mix with regular packages in one repo.
- Do not stuff import side-effects into `__init__.py` — keep it a re-export surface.

### Misc
- Use `is` / `is not` only for singletons (`None`, `True`, `False`).
- Use `isinstance(x, T)` not `type(x) is T`.
- Prefer comprehensions and generators over `map`/`filter` with lambdas.
