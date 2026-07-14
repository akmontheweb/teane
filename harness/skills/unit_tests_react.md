---
applies_to: [react, typescript, node]
---

## React / TypeScript — Unit Test Skill (Jest + React Testing Library)

### When this skill applies
Any React + TypeScript workspace with `package.json` declaring `react` and (typically) `@testing-library/react`. Applies to Vite / Create React App / Next.js projects alike. The sandbox pre-installs `jest`, `@testing-library/react`, `@testing-library/user-event`, `@testing-library/jest-dom`, and `jest-environment-jsdom`.

### Coverage gate
The operator's `coverage.enforce` setting decides whether under-threshold builds fail:
- `coverage.enforce=true` (default) — build/patch succeeds only when Jest exits zero. The `package.json` you emit carries a `coverageThreshold` block; under-threshold trips repair_node to write more tests.
- `coverage.enforce=false` — coverage is still measured (report generated) but the `coverageThreshold` block is omitted, so the build passes regardless of coverage%.

The `package.json` you emit already resolves the threshold block correctly for the current operator setting (see the makefile_node skill). Aim for coverage BEYOND {{coverage.min_pct}}% where reasonable. Prioritize business logic (services, hooks, reducers, utilities) over presentation glue (index.tsx bootstraps, static wrappers).

### What IS a unit test (belongs here)
- One component / hook / service / util exercised in isolation.
- No network — all `fetch` / `axios` / API modules mocked (`jest.mock('../services/api')`).
- No real timers unless the code under test IS timer logic — otherwise `jest.useFakeTimers()`.
- Runs in milliseconds under jsdom.

### What ISN'T a unit test (do NOT write here)
- End-to-end user journeys through the deployed app — `teane test` owns those (Playwright against the compose stack).
- Cross-component integration tests that spin the entire app tree — a smell that the components are too coupled.
- Screenshot / visual-regression tests.

### File layout
- Co-locate: `components/Button.tsx` → `components/Button.test.tsx`. RTL convention; Jest picks up `*.test.tsx` automatically.
- ONE tests convention per repo: either co-located `*.test.tsx` files OR a parallel `src/__tests__/` tree. Never both — mixed tests trees confuse the coverage collector.
- Test files import from the module under test via relative path (`./Button`), not the compiled `dist/` output.

### Patterns (RTL idioms — these, not custom scaffolding)
- Components: `render(<Foo prop="x" />)`, then `screen.getByRole(...)` / `screen.getByLabelText(...)`. Never `getByTestId` unless nothing else fits — it's a leak of implementation into the test.
- User interaction: `userEvent.setup()` from `@testing-library/user-event` — NOT the legacy `fireEvent` (misses async updates).
- Async assertions: `await screen.findBy...` for elements that appear after an update; `await waitFor(() => ...)` for arbitrary state.
- Hooks: `renderHook(() => useMyHook(arg))` from `@testing-library/react`. Assert on `result.current`.
- Timers: `jest.useFakeTimers({ now: new Date('2026-01-01T00:00:00Z') })`; advance with `jest.advanceTimersByTime(ms)`; restore in `afterEach` with `jest.useRealTimers()`.
- API mocks: `jest.mock('./api')` at file top; then `(api.getFoo as jest.Mock).mockResolvedValue(...)`. Prefer mocking the service module the component imports, not global `fetch`.
- Cleanup: `afterEach(() => cleanup())` is already automatic in Jest 27+ with RTL — do not re-add.

### Assertion style
- `expect(...).toBeInTheDocument()` / `.toHaveTextContent(...)` / `.toHaveValue(...)` from `@testing-library/jest-dom`.
- For thrown errors: `await expect(fn()).rejects.toThrow(SpecificError)`.
- `expect.assertions(N)` for async paths where you must confirm all N assertions ran.
- Never assert on `console.log` output — split the code so the value you care about is returned.

### Anti-patterns that inflate coverage without value
- `render(<Button />); expect(screen.getByRole('button')).toBeInTheDocument()` — tests that RTL works, not your code.
- Snapshot tests without behaviour assertions — locks in accidental markup, drifts.
- Mocking the component under test.
- Testing internal state via `container.firstChild` traversal — the DOM is not your API.
