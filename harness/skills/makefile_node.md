---
applies_to: [node, typescript]
---

## Build — Node.js / TypeScript Makefile

### When this skill applies
The workspace has `package.json` (any Node project) or `tsconfig.json` (TypeScript). Covers Express, Nest, Fastify, Next.js, Vite, plain Node CLIs, and library packages. The harness runs `make build` by default; without a Makefile it falls back to noisy command-adaptation logic.

### Always emit a `Makefile` in your first patch
Pick the package manager from lockfile presence: `pnpm-lock.yaml` → pnpm, `yarn.lock` → yarn, else npm (the default). For TypeScript, include a typecheck step in `build:`.

### Coverage gate (STRICTLY ENFORCED)
Every `test:` target MUST run Jest with coverage. When the operator sets `coverage.enforce=true` (default) the build ALSO fails if line coverage < {{coverage.min_pct}}%; when `coverage.enforce=false` coverage is still reported but the build passes regardless. Both branches use `coverage.min_pct` from `config.json` (shipped default 70). Two pieces of config:

1. In `package.json`, emit the block exactly as shown below. The `{{coverage.jest_threshold_snippet}}` marker resolves to the `"coverageThreshold": {...}` fragment when `coverage.enforce=true`, or to an empty string when `coverage.enforce=false` — either way ship the `"jest"` block as rendered:
   ```json
   {
     "scripts": { "test": "jest --coverage" },
     "jest": {
       {{coverage.jest_threshold_snippet}}"collectCoverageFrom": [
         "src/**/*.{ts,tsx,js,jsx}",
         "!src/**/*.d.ts",
         "!src/**/index.{ts,tsx}",
         "!src/main.tsx"
       ]
     }
   }
   ```
2. Keep the Makefile target as just `npm test` (or `pnpm test`) — Jest itself enforces the threshold via non-zero exit when the coverageThreshold block is present. Do NOT invent alternative gates (grep, custom scripts).

`collectCoverageFrom` is what pins the denominator — without it Jest measures only files touched by tests, which silently hides uncovered modules. Adjust the globs to your `src/` layout; do NOT include test files themselves in the denominator (`!**/*.test.*`).

Vitest projects: the same `--coverage` flag applies (via `@vitest/coverage-v8`); Vitest reads `coverage.thresholds.lines` from `vite.config.ts` — mirror the `{{coverage.min_pct}}` value there.

**Plain Node (no TypeScript):**
```make
.PHONY: build test all clean

build:
	npm install

test:
	npm test

all: build test

clean:
	rm -rf node_modules coverage
```

**TypeScript:**
```make
.PHONY: build test all clean

build:
	npm install
	npx tsc --noEmit

test:
	npm test

all: build test

clean:
	rm -rf node_modules dist coverage
```

**pnpm variant** (when `pnpm-lock.yaml` exists):
```make
.PHONY: build test all clean

build:
	pnpm install --frozen-lockfile

test:
	pnpm test

all: build test

clean:
	rm -rf node_modules coverage
```

### Conventions to follow
- Use TAB indentation inside recipes — Make rejects spaces.
- `build:` installs deps (and typechecks for TS); `test:` runs the test runner. Don't conflate them.
- Use `npm install` (not `npm ci`) for greenfield runs — `npm ci` requires an existing `package-lock.json` that the LLM hasn't generated yet. Once a lockfile lands, the operator can swap to `ci` for reproducibility.
- For TypeScript, run `tsc --noEmit` in `build:` to catch type errors before tests run. Type errors at test-time are noisier than at build-time.

### Common patches the LLM gets wrong
- Spaces instead of tabs for recipe indentation (silent fail).
- Using `npm ci` on greenfield — fails because no lockfile exists yet.
- Forgetting `.PHONY:` and watching `make build` skip when a `build/` directory exists.
- Calling `tsc` directly (path issues); use `npx tsc` so it resolves from `node_modules/.bin`.
- Adding `npm run lint` to `build:` without ensuring a `lint` script exists in `package.json` — pick targets that exist or add the script in the same patch.
