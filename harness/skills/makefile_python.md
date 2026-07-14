---
applies_to: [python]
---

## Build — Python Makefile

### When this skill applies
The workspace is a Python project — detected via `requirements.txt`, `pyproject.toml`, `setup.py`, or any top-level `.py` file. Applies to FastAPI, Django, Flask, library, and CLI projects alike. The harness runs `make build` by default; without a Makefile it falls back to noisy command-adaptation logic that's harder to reproduce locally.

### Installer: ALWAYS `uv pip install` — NEVER plain `pip install`
The harness sandbox pre-installs [`uv`](https://github.com/astral-sh/uv) on the system PATH. `uv pip install` is a drop-in replacement for `pip install` that reads the same `requirements.txt` / `pyproject.toml` and writes to the same site-packages, but resolves and installs **10–30× faster** on cold caches. The sandbox also persists `uv`'s download cache between containers, so the second build in a session installs from local wheels.

**Rules — absolute:**
- Every install line in your Makefile MUST start with `uv pip install` (with `--system` so it targets the container's system Python, not a venv).
- Do NOT write `pip install`, `pip3 install`, `python3 -m pip install`, `poetry install`, or `pdm install`. The harness recognises `uv pip install` / `uv sync` / `uv add` as install steps and configures the sandbox accordingly; other forms still work but are slower and miss the harness-managed cache.
- Do NOT create a virtualenv (`python -m venv`, `uv venv`). The container is already isolated and pytest / uv themselves are on PATH; an inner venv just adds latency.

### Always emit a `Makefile` in your first patch
Pick the variant matching the dependency manifest you're also creating (or that already exists). Each variant has separate `build:` and `test:` targets plus a `.PHONY:` line, so operators can run `make test` independently.

### Coverage gate (STRICTLY ENFORCED)
Every `test:` target MUST include `--cov=<pkg>` (one flag per top-level source package — never `--cov=.`) and `--cov-fail-under=70`. `pytest-cov` is pre-installed in the sandbox; do NOT add it to `requirements.txt`. Pytest's own exit code IS the gate — no custom scripts, no stdout grep.

**With `requirements.txt`:**
```make
.PHONY: build test all clean

build:
	uv pip install --system -r requirements.txt

test:
	python3 -m pytest -q --cov=server --cov-fail-under=70

all: build test

clean:
	rm -rf __pycache__ .pytest_cache build dist *.egg-info .coverage
```

**With `pyproject.toml`** (editable install — covers Poetry, setuptools, hatch, PDM):
```make
.PHONY: build test all clean

build:
	uv pip install --system -e .

test:
	python3 -m pytest -q --cov=src --cov-fail-under=70

all: build test

clean:
	rm -rf __pycache__ .pytest_cache build dist *.egg-info .coverage
```

**Bare workspace** (no manifest yet — only when you also can't create one):
The sandbox already has pytest pre-installed, so `build:` is a no-op. Still emit the target so `make all` works.
```make
.PHONY: build test all

build:
	@true

test:
	python3 -m pytest -q --cov=. --cov-fail-under=70

all: build test
```
Substitute `--cov=<pkg>` with your actual source root(s). NEVER omit `--cov` and `--cov-fail-under`; a build that runs zero tests would otherwise report success.

### Conventions to follow
- Use TAB indentation inside recipes — Make rejects spaces with `*** missing separator. Stop.`
- The `build:` target installs dependencies and nothing else. Don't run tests from `build:` — that's what `test:` is for.
- Declare every target in `.PHONY:` so file-name collisions (`build/` dir, `test/` dir) don't suppress execution.
- Don't shell-pipe `&&` across recipe lines — each recipe line runs in its own subshell. Either keep both commands on one line with `&&`, or split into separate targets.

### Common patches the LLM gets wrong
- Using spaces instead of tabs for recipe indentation (silent fail).
- Calling `pip install` instead of `uv pip install --system` — slower and bypasses the harness's persistent install cache.
- Forgetting the `--system` flag on `uv pip install` — uv refuses to install into the system Python without it (safety guardrail) and the build fails with "Use `--system` to install into the system Python".
- Mixing `pytest` and `python -m pytest` across targets — pick one (prefer `python3 -m pytest` so the import path matches `build:`).
- Forgetting `.PHONY:` and then debugging why `make test` skipped when a `test/` directory exists.
- Hard-coding a virtualenv path (`venv/bin/pip`) — the harness runs inside a clean Docker container; venvs aren't needed.
- Adding `pytest` / `pytest-cov` / `pytest-xdist` to `requirements.txt` — they're pre-installed in the sandbox. Only add them as dev dependencies if the project will be installed outside the sandbox too.
