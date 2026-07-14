---
applies_to: [typescript]
---

## TypeScript Style Guide

### Source
- TypeScript Deep Dive — Basarat Ali Syed (https://basarat.gitbook.io/typescript/styleguide)

### Naming
- `PascalCase` for classes, interfaces, type aliases, enums.
- `camelCase` for variables, parameters, properties, functions, methods.
- Do **not** prefix interfaces with `I`. `User` and `UserService`, never `IUser`.
- Type-parameter names: single capital letter (`T`, `K`, `V`) or descriptive `PascalCase` ending in `T` (`RequestT`) when readability calls for it.
- File names: `camelCase` for general utilities, `PascalCase` for files that export a single matching component/class.

### Types & inference
- Let TypeScript infer when the inference is obvious (`const n = 1`, not `const n: number = 1`).
- Annotate at boundaries: every exported function's parameters and return type; every public field of an exported class.
- Prefer `interface` for object shapes you may extend or merge across declarations; prefer `type` for unions, intersections, mapped, and conditional types.
- Use `readonly` on properties that should not mutate; `Readonly<T>` and `ReadonlyArray<T>` on inputs you do not own.
- Use `unknown` instead of `any` whenever you don't know the type — `unknown` forces narrowing before use; `any` silently disables checking.
- Avoid `null`; prefer `undefined` plus optional properties (`x?: T`). Be consistent: don't mix both in the same API.

### Strictness
- Enable `strict: true` in `tsconfig.json`. The individual strict flags (`strictNullChecks`, `noImplicitAny`, `strictFunctionTypes`, `strictBindCallApply`) catch real bugs.
- Enable `noUncheckedIndexedAccess` — array/object index access returns `T | undefined`, which forces guarding before use.
- Don't suppress errors with `@ts-ignore` or `@ts-expect-error` without an adjacent comment explaining why and what blocks fixing it properly.

### Imports & modules
- ES module syntax only — no `require()`.
- Group imports: third-party first, then internal absolute, then relative — blank line between groups.
- Prefer named imports; reserve default exports for single-purpose modules.

### Generics
- Use generics whenever a function/class is genuinely polymorphic. Do not use generics to reproduce dynamic typing (`<T>(x: T) => any`).
- Constrain type params (`<T extends Foo>`) so the body can use methods of the bound, not `any`.

### Enums vs. unions
- Prefer string-literal union types over `enum` for simple value sets — they tree-shake cleanly and don't introduce a runtime object.
- Use `const enum` only inside the same compilation unit; never expose across package boundaries.

### Datetime & timezones
`Date` is genuinely broken (locale-sensitive parsing, silent NaN on invalid input, no timezone in the value itself). Pick ONE convention.
- Wire / storage format: ISO 8601 UTC strings via `new Date().toISOString()` (always ends in `Z`). Never epoch numbers on the wire — they lose the "this is a time" typing.
- Parsing: `new Date(isoString)` accepts ISO 8601 reliably; check `isNaN(d.getTime())` before use. Never parse locale strings — reject at the boundary or use a library.
- Server + client agree on UTC. Convert to local ONLY at display time via `Intl.DateTimeFormat(undefined, {...})` — never with hand-rolled `getHours()` math.
- Prefer `number` (epoch ms via `Date.now()`) for durations, deadlines, and TTLs — treat `Date` as a boundary/display type only.
- If the app does calendar arithmetic (add days, quarters, business hours), pull in `date-fns` (tree-shakeable) rather than reinventing.
- Test mocks: `jest.useFakeTimers({ now: new Date('2026-01-01T00:00:00Z') })` or Vitest `vi.setSystemTime(new Date(...))`. Always restore in `afterEach`.

### Filesystem paths (server / Node) — Linux / macOS / Windows
Client React code has no filesystem — this applies to server code only.
- Import `path` as `import path from 'node:path'`. Use `path.join()`, `path.resolve()`, `path.dirname()`; never string concatenation with `+`.
- `path.sep` differs (`/` vs `\`). Assume nothing — always go through `path` helpers when interoperating with OS paths.
- URL segments are always `/`: use `path.posix.join()` for URL construction, plain `path.join()` for filesystem. Do not swap.
- Absolute vs relative: `path.isAbsolute()` before joining user input, and `path.resolve()` when you need canonical form.
- `import.meta.url` + `fileURLToPath()` for locating source files in ESM (Windows-safe); do NOT use `__dirname` in `.mts`/`.ts` ESM code.
- Encoding: always specify `'utf8'` on `fs.readFile`/`fs.writeFile`. Never rely on the default.

### Formatting
- 2-space indent; semicolons; single quotes; trailing commas on multi-line lists.
- Place the type annotation on the same line as the identifier (`const x: T = ...`), not the next line.
- No `namespace` for new code — use ES modules.
