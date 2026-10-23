---
applies_to: [node, javascript]
---

## JavaScript Test Generation Guide (Jest)

Write Jest-style unit tests for the JavaScript source files just modified. Tests call the **real implementation** with realistic inputs — **do not write mocks**. No `jest.mock`, no `jest.fn`, no manual stubs. When a side effect cannot be invoked directly, use a real local resource (an in-memory file, a sqlite memory DB, a local `http.createServer` listener) and tear it down in `afterEach`.

### File placement
- Co-locate next to source as `<name>.test.js`, or place under `__tests__/<name>.test.js`. Match whatever convention the existing tree uses.

### Structure
- `describe('<Symbol>', () => { … })` per exported function or class.
- `test('<behaviour>', () => { … })` (or `it`) for each case.
- `beforeEach` / `afterEach` for setup/teardown of real resources (temp dirs via `fs.mkdtemp(os.tmpdir())`, sqlite `:memory:`, ephemeral ports).

### Assertions
- `expect(value).toBe(literal)` for primitives.
- `expect(value).toEqual(struct)` for deep equality.
- `expect(() => fn()).toThrow(/pattern/)` for exceptions.
- `await expect(promise).resolves.toEqual(value)` / `.rejects.toThrow(...)` for async.

### Dates and times — a test must not depend on the day it runs
A test whose expected value changes with the calendar passes today and fails
months later with no code change. Jest's fake timers are the clock, not a mock,
so the no-mocking rule does not forbid them.

```js
beforeEach(() => jest.useFakeTimers().setSystemTime(new Date('2026-03-01')));
afterEach(() => jest.useRealTimers());
```

Or assert the invariant rather than a literal arrangement:
```js
expect(dates).toEqual([...dates].sort());        // "sorted by date"
expect(new Set(names)).toEqual(new Set(['Early', 'Late']));  // membership
```

Never assert a hardcoded order or value that depends on today — recurring
dates, ages, "upcoming"/"within N days" windows, expiry, relative buckets
("this week"), and timezone-dependent day boundaries all qualify. If the
expected value would differ were the test run in another month or timezone,
freeze the clock or assert the rule.

### What NOT to do
- No `jest.mock('...')`, no `jest.spyOn()`, no `jest.fn()`.
- No assertion whose expected value depends on today's date — see "Dates and times" above.
- No `nock`, no `sinon`. If the code under test makes an HTTP call, the test starts a local `http.createServer` on `port: 0`, points the code at it, asserts, then closes it in `afterEach`.
- No "test doubles" implemented inline (e.g., `const fakeDb = { query: () => [] }`). Use an in-process real implementation instead.

### Test-environment scaffolding — create the config WITH the tests
If the workspace has no jest wiring yet, patch it in the SAME response
as the tests: `jest` in `package.json` devDependencies (plus
`jest-environment-jsdom` and a `testEnvironment: "jsdom"` jest config for
DOM/component tests). A test file without its runner config just shifts
the failure to the environment and drowns real errors in noise.

### Minimal example
```javascript
const { divide } = require('../src/calculator');

describe('divide', () => {
  test('returns quotient for integers', () => {
    expect(divide(10, 2)).toBe(5);
  });

  test('throws on zero divisor', () => {
    expect(() => divide(1, 0)).toThrow(/cannot divide by zero/);
  });

  test.each([[0, 1, 0], [-4, 2, -2], [7, 7, 1]])('divide(%i, %i) === %i', (a, b, expected) => {
    expect(divide(a, b)).toBe(expected);
  });
});
```
