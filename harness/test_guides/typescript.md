---
applies_to: [typescript]
---

## TypeScript Test Generation Guide (Jest + ts-jest)

Same conventions as the JavaScript guide — Jest, real implementations, no mocks. Adapt for TypeScript:

### File placement
- Co-locate next to source as `<name>.test.ts`, or under `__tests__/<name>.test.ts`.

### Imports & types
- `import { divide } from '../src/calculator';` (ES modules form; `ts-jest` accepts it).
- Annotate the test data types when they aren't obvious from inference (e.g. `const cases: ReadonlyArray<[number, number, number]> = [...]`).
- Avoid `any`. If the production code returns a discriminated union, narrow it with `if ('error' in result)` before asserting fields.

### Async
- Use `async`/`await` test functions; do not return promises from sync test functions.
- `await expect(promise).resolves.toEqual(...)` / `.rejects.toThrow(...)`.

### What NOT to do
- No `jest.mock`, no `jest.fn`, no `ts-mockito`, no `sinon`.
- Do not invent `Partial<T>` shapes to stand in for real values; construct full objects (use a factory function defined in the test file if the type has many required fields).

### Test-environment scaffolding — create the config WITH the tests
A `.test.ts(x)` file whose type environment is missing produces hundreds
of `TS2304 Cannot find name 'expect'` / `TS2307 Cannot find module
'@testing-library/react'` diagnostics that drown every real error. When
you generate tests, verify the environment exists and patch it in the
SAME response if it doesn't:
- `package.json` devDependencies must include `jest`, `ts-jest`,
  `@types/jest` — plus `@testing-library/react`,
  `@testing-library/jest-dom`, and `jest-environment-jsdom` when testing
  React components.
- `tsconfig.json` `compilerOptions.types` must list `"jest"` (and
  `"@testing-library/jest-dom"` if used), and `include` must cover the
  test files.
- Component tests need `testEnvironment: "jsdom"` in the jest config and
  a `jest.setup.ts` importing `@testing-library/jest-dom`, wired via
  `setupFilesAfterEach`.
- NEVER paper over missing types with `@ts-ignore`, `declare const
  expect`, or hand-rolled ambient declarations — fix the config.

### React components (@testing-library/react)
- `render(<Panel {...props} />)` then query via `screen` by role or
  accessible text: `screen.getByRole('button', { name: /save/i })`.
  Do not assert on class names, DOM structure, or component internals.
- Interactions through `@testing-library/user-event`
  (`await userEvent.click(...)`) — not `fireEvent` — and `await
  screen.findByText(...)` for anything that appears asynchronously.
- One render per test; no shared component instances across tests.

### Minimal example
```typescript
import { divide } from '../src/calculator';

describe('divide', () => {
  it('returns quotient for integers', () => {
    expect(divide(10, 2)).toBe(5);
  });

  it('throws on zero divisor', () => {
    expect(() => divide(1, 0)).toThrow(/cannot divide by zero/);
  });

  it.each<[number, number, number]>([[0, 1, 0], [-4, 2, -2]])(
    'divide(%i, %i) === %i',
    (a, b, expected) => {
      expect(divide(a, b)).toBe(expected);
    },
  );
});
```
