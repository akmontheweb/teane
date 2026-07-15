---
applies_to: [python]
---

## Python Test Generation Guide

Write pytest-style unit tests for the Python source files just modified. Tests exercise the **real implementation** with realistic inputs — **do not write mock objects**. When a side effect is genuinely impractical to invoke directly (filesystem outside `tmp_path`, environment variables, system clock, network), use pytest's built-in fixtures instead of inventing a mock framework.

### File placement
- Project root tests directory is `tests/`. If a package layout uses `src/<pkg>/`, mirror it as `tests/<pkg>/`.
- One test file per source file, named `test_<module>.py`.

### Structure
- Group cases into classes named `Test<Symbol>` so failures surface in a readable hierarchy.
- Function names: `test_<behavior_under_test>` — describe the behaviour, not the input.
- Use `pytest.mark.parametrize` when a single behaviour is exercised across multiple inputs; one test function with N IDs beats N near-identical functions.

### Fixtures the test runner already provides — use these instead of mocks
- `tmp_path` — `pathlib.Path` scoped to the test; use for any filesystem read/write.
- `tmp_path_factory` — session-scoped equivalent.
- `monkeypatch` — set/unset environment variables, `setattr` on attributes, `chdir`.
- `capsys` / `capfd` — capture stdout/stderr to assert on output.
- `caplog` — capture log records; `caplog.records` for structured assertions.

### Style
- `assert` statements only — no `unittest.TestCase` boilerplate.
- One assertion per outcome; many assertions in one test is fine as long as they describe one behaviour.
- For exceptions: `with pytest.raises(ValueError, match="..."):` — match the message so a renamed exception doesn't pass silently.
- For floats: `pytest.approx(expected, rel=1e-6)`.
- For collections: `assert result == expected` — never check `len` and a sample element separately when an equality check works.

### What NOT to do
- Do not use `unittest.mock.patch`, `Mock`, `MagicMock`, `mocker.patch`, or `pytest-mock`. The tests must call the real function.
- Do not stub HTTP — if the code under test makes network calls, the test runner uses a local fake server (e.g., `http.server` in a thread) or marks the test `pytest.mark.network` and the harness skips it deterministically.
- Do not import the production code under a fake name; use the actual import path.

### FastAPI / SQLAlchemy API tests — the ONLY safe database pattern
API tests that override `get_db` keep re-introducing two whole-suite
failure classes. Both rules are non-negotiable:

1. **Engine**: use a plain private in-memory database with `StaticPool` —
   `create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)`.
   NEVER use `sqlite:///:memory:` without StaticPool (each pooled
   connection gets its own empty database, and `TestClient` runs the app
   on a different thread — "no such table" forever) and NEVER use
   `file::memory:?cache=shared` (every test module using that URL shares
   ONE database; their fixtures create/drop each other's tables and the
   suite fails in ways single-file runs don't reproduce).
2. **Override lifetime**: NEVER assign `app.dependency_overrides[...]` at
   module import time. Pytest imports every test module at collection, so
   the last module's override silently wins for the ENTIRE suite. Install
   and remove the override inside a fixture:

```python
from sqlalchemy.pool import StaticPool

engine = create_engine(
    "sqlite://", connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

@pytest.fixture(autouse=True)
def db_override():
    Base.metadata.create_all(bind=engine)
    def _get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()
    app.dependency_overrides[get_db] = _get_db
    yield
    app.dependency_overrides.pop(get_db, None)
    Base.metadata.drop_all(bind=engine)
```

### Minimal example
```python
import pytest
from mypkg.calculator import divide

class TestDivide:
    def test_returns_quotient_for_integers(self):
        assert divide(10, 2) == 5

    def test_raises_on_zero_divisor(self):
        with pytest.raises(ZeroDivisionError, match="cannot divide by zero"):
            divide(1, 0)

    @pytest.mark.parametrize("a,b,expected", [(0, 1, 0), (-4, 2, -2), (7, 7, 1)])
    def test_table(self, a, b, expected):
        assert divide(a, b) == expected
```
