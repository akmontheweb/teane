---
applies_to: [html]
---

## Build — Static Web App Makefile

### When this skill applies
The workspace ships static HTML/CSS/JS that the operator opens in a browser — no Node bundler, no build step. Detected when `index.html` (or any `.html`) sits at the workspace root AND no frontend framework is present (React/Vue/Angular bring their own Makefile / package.json conventions).

If the project ALSO has a `package.json` with a JS test runner (e.g. `jest`), keep the `test:` target wired up — see "Variant with tests" below.

### Why a real Makefile matters here
The harness sandbox runs `make build` to verify the project. For static sites a bare `@echo "Build successful"` placeholder is useless — the sandbox sees exit 0 and reports "all green" while the app is broken (missing CSS, dangling `<script src>`, etc.).

**The `build:` target must actually do something:** run the harness's static asset reference scanner against the workspace. The scanner exits non-zero with `file:line:col: error: unresolved asset reference '...'` messages on any broken local reference — which the sandbox's diagnostic extractor picks up and feeds into the repair loop. With this target in place, the existing sandbox+patch loop catches the bug class without any browser tooling.

### Always emit a `Makefile` in your first patch

**Plain static site (no tests):**
```make
.PHONY: build serve clean

build:
	python3 -m harness.web_asset_scan .

serve:
	@echo "Open http://localhost:8000 in your browser. Ctrl-C to stop."
	python3 -m http.server 8000

clean:
	rm -rf __pycache__
```

**Variant with tests (jest etc):**
```make
.PHONY: build serve test all clean

build:
	python3 -m harness.web_asset_scan .

serve:
	@echo "Open http://localhost:8000 in your browser. Ctrl-C to stop."
	python3 -m http.server 8000

test:
	npx jest --verbose

all: build test

clean:
	rm -rf __pycache__ node_modules coverage
```

### Conventions to follow
- Use TAB indentation inside recipes — Make rejects spaces.
- `build:` runs the asset scanner. Do NOT replace this with `echo` — the sandbox needs a real check here.
- `serve:` exists so the operator can actually run the app. Opening `index.html` via `file://` doesn't work for ES modules (CORS blocks them) — `make serve` is the documented escape hatch.
- Default port 8000 matches the harness convention. Operators expect it.
- Don't add `install:` — there's nothing to install for a pure static site.

### Common patches the LLM gets wrong
- **Echo placeholder for `build:`** — meaningless to the sandbox. Always invoke the scanner.
- **Missing `serve:` target** — leaves operators stranded at the `file://` problem with no documented fix. Always include it.
- **Spaces instead of tabs** in recipe indentation. Silent fail; Make ignores the rule.
- **Adding `npm install` to `build:`** for a project with no `package.json`. If there's no JS dependency manifest, the asset scanner is the entire build.
