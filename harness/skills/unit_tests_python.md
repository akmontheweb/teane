---
applies_to: [python]
---

## Python — Unit Test Skill (pytest)

### When this skill applies
Any Python workspace (FastAPI, Django, Flask, plain library, CLI). The sandbox pre-installs `pytest`, `pytest-asyncio`, `pytest-cov`, `pytest-mock`, `freezegun`, and `responses` — do NOT add them to `requirements.txt` unless the project ships outside the sandbox too.

### Coverage gate
The operator's `coverage.enforce` setting decides whether under-threshold builds fail:
- `coverage.enforce=true` (default) — build/patch succeeds only when `pytest --cov-fail-under={{coverage.min_pct}}` exits zero. Under-threshold trips repair_node to write more tests.
- `coverage.enforce=false` — coverage is still measured (report generated) but under-threshold does NOT fail the build.

The Makefile you emit already resolves the fail-under flag correctly for the current operator setting (see the makefile_python skill). Aim for coverage BEYOND {{coverage.min_pct}}% where reasonable — {{coverage.min_pct}}% is the floor, not the target. Prioritize business logic (services, extractors, validators) over glue code (config loaders, main.py bootstraps).

### What IS a unit test (and belongs here)
- A single function / class / method exercised in isolation with real inputs and mocked I/O (DB, HTTP, filesystem, clock, RNG).
- Runs in milliseconds. No network, no real DB, no `time.sleep(1)`.
- One behaviour per test — a failing test tells you exactly what broke.

### What ISN'T a unit test (do NOT write these during build/patch)
- End-to-end user journeys (`teane test` owns those — Playwright against the deployed compose stack).
- Live-database or live-network integration checks — flaky in the sandbox and out of scope for the compile+repair gate.
- Manual smoke scripts, benchmark scripts, one-off REPL debugging.

### File layout
- `server/app/tests/test_<module>.py` mirrors `server/app/<module>.py`. One test file per production module.
- Never create both `server/tests/` AND `server/app/tests/` — pick ONE tests root per package. The harness's `DUPLICATE_TEST_ROOT` guard rejects the second one.
- Test files import from the production package via its absolute dotted path (`from server.app.services.foo import bar`) — never relative-import back up out of `tests/`.

### Patterns (pytest idioms — use these, not custom scaffolding)
- Isolation from disk: `tmp_path` fixture returns a `pathlib.Path` unique per test.
- Isolation from time: `freezegun.freeze_time("2026-01-01T00:00:00Z")` decorator OR fixture. Mock `datetime.now` at the CALLING module (`patch("mymodule.datetime")`) — never at the `datetime` stdlib module.
- Isolation from HTTP: `responses` for `requests`, `httpx.MockTransport` for `httpx`, `AsyncMock` for async clients.
- Isolation from DB: `Session(engine)` on SQLite in-memory (`sqlite:///:memory:`) works for SQLAlchemy; for raw sqlite, use `tmp_path / "test.db"`.
- Parameterisation: `@pytest.mark.parametrize` for value tables — one test function, many rows, better failure messages than a loop.
- Async: mark async tests with `@pytest.mark.asyncio` (pytest-asyncio in `mode=auto` picks them up automatically).
- Autouse fixtures for module-level state reset: if your production module has module-level caches / dicts / locks, add an `autouse=True` fixture that clears them before every test.

### Assertion style
- `assert x == y` — pytest's introspection prints the diff. Never `self.assertEqual` (unittest style).
- `with pytest.raises(SpecificError) as exc: ...` for expected failures. Assert on `exc.value` details, not just the class.
- For floats: `assert x == pytest.approx(y, rel=1e-6)`.
- Never assert on log output or print — split the tested function so the return value carries the information you want to check.

### Anti-patterns that inflate coverage without value
- Testing that a function calls `logger.info` (implementation-detail; brittle).
- One giant test that exercises 20 branches — split it. One test = one behaviour.
- Mocking the function under test (mocks its own return value, asserts the mock was called — provably tautological).
- Tests that only assert `assert result is not None` — the type checker already knows.
