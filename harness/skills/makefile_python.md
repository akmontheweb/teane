---
applies_to: [python]
---

## Build — Python Makefile

### When this skill applies
The workspace is a Python project — detected via `requirements.txt`, `pyproject.toml`, `setup.py`, or any top-level `.py` file. Applies to FastAPI, Django, Flask, library, and CLI projects alike. The harness runs `make build` by default; without a Makefile it falls back to noisy command-adaptation logic that's harder to reproduce locally.

### Installer: ALWAYS `uv pip install` into the interpreter the tests use
The harness sandbox pre-installs [`uv`](https://github.com/astral-sh/uv) on the system PATH. `uv pip install` is a drop-in replacement for `pip install` that reads the same `requirements.txt` / `pyproject.toml`, but resolves and installs **10–30× faster** on cold caches, and the sandbox persists `uv`'s download cache between containers.

The critical part: install into **the same interpreter your tests run on**. In the sandbox, `python3` (and the pre-baked `pytest`) is a **uv-managed CPython** — NOT the base OS `python3`. Install with `--python /usr/local/bin/python3` (the managed interpreter) plus `--break-system-packages` (the managed interpreter is PEP668-marked, so uv requires the override to write into it). Do NOT use `uv pip install --system`: `--system` targets the base OS Python instead, so your deps land in an interpreter the tests never import from and every test fails with `ModuleNotFoundError`.

**Rules — absolute:**
- Every install line MUST be `uv pip install --python /usr/local/bin/python3 --break-system-packages …`. Define `PYTHON := /usr/local/bin/python3` once and reuse it in both `build:` (install) and `test:` (`$(PYTHON) -m pytest`) so install and test share one interpreter.
- Do NOT use `--system`, `pip install`, `pip3 install`, `poetry install`, or `pdm install`.
- Do NOT create a virtualenv (`python -m venv`, `uv venv`). Installing into the managed interpreter as above is both correct and faster; an inner venv just adds latency.

### Always emit a `Makefile` in your first patch
Pick the variant matching the dependency manifest you're also creating (or that already exists). Each variant has separate `build:` and `test:` targets plus a `.PHONY:` line, so operators can run `make test` independently.

### Coverage gate (STRICTLY ENFORCED)
Every `test:` target MUST include `--cov=<pkg>` (one flag per top-level source package — never `--cov=.`) and `{{coverage.pytest_fail_flag}}` (operator-configurable via `coverage.min_pct` in `config.json`; the shipped default is 70). `pytest-cov` is pre-installed in the sandbox; do NOT add it to `requirements.txt`. Pytest's own exit code IS the gate — no custom scripts, no stdout grep.

**With `requirements.txt`:**
```make
.PHONY: build test all clean

# The sandbox's managed CPython — what `python3` resolves to and what the
# pre-baked pytest runs on. Install into THIS interpreter so deps are
# importable by the tests.
PYTHON := /usr/local/bin/python3

build:
	uv pip install --python $(PYTHON) --break-system-packages -r requirements.txt

test:
	$(PYTHON) -m pytest -q --cov=server{{coverage.pytest_fail_flag}}

all: build test

clean:
	rm -rf __pycache__ .pytest_cache build dist *.egg-info .coverage
```

**With `pyproject.toml`** (editable install — covers Poetry, setuptools, hatch, PDM):
```make
.PHONY: build test all clean

PYTHON := /usr/local/bin/python3

build:
	uv pip install --python $(PYTHON) --break-system-packages -e .

test:
	$(PYTHON) -m pytest -q --cov=src{{coverage.pytest_fail_flag}}

all: build test

clean:
	rm -rf __pycache__ .pytest_cache build dist *.egg-info .coverage
```

**Bare workspace** (no manifest yet — only when you also can't create one):
The sandbox already has pytest pre-installed, so `build:` is a no-op. Still emit the target so `make all` works.
```make
.PHONY: build test all

PYTHON := /usr/local/bin/python3

build:
	@true

test:
	$(PYTHON) -m pytest -q --cov=.{{coverage.pytest_fail_flag}}

all: build test
```
Substitute `--cov=<pkg>` with your actual source root(s). NEVER omit `--cov` — a build that runs zero tests would otherwise report success. Emit the pytest invocation exactly as shown (including whatever the operator's `coverage.enforce` setting resolved to for the fail-under flag).

### Conventions to follow
- Use TAB indentation inside recipes — Make rejects spaces with `*** missing separator. Stop.`
- The `build:` target installs dependencies and nothing else. Don't run tests from `build:` — that's what `test:` is for.
- Declare every target in `.PHONY:` so file-name collisions (`build/` dir, `test/` dir) don't suppress execution.
- Don't shell-pipe `&&` across recipe lines — each recipe line runs in its own subshell. Either keep both commands on one line with `&&`, or split into separate targets.

### Common patches the LLM gets wrong
- Using spaces instead of tabs for recipe indentation (silent fail).
- Using `uv pip install --system` — `--system` targets the base OS Python, NOT the managed `python3` the tests run on, so every test fails with `ModuleNotFoundError`. Always install with `--python /usr/local/bin/python3 --break-system-packages`.
- Calling plain `pip install` — slower and bypasses the harness's persistent uv cache.
- Mixing `pytest` and `$(PYTHON) -m pytest` across targets — use `$(PYTHON) -m pytest` so the test import path matches the interpreter `build:` installed into.
- Forgetting `.PHONY:` and then debugging why `make test` skipped when a `test/` directory exists.
- Hard-coding a virtualenv path (`venv/bin/pip`) — the harness runs inside a clean Docker container; venvs aren't needed.
- Adding `pytest` / `pytest-cov` / `pytest-xdist` to `requirements.txt` — they're pre-installed in the sandbox. Only add them as dev dependencies if the project will be installed outside the sandbox too.
