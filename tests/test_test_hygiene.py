"""Generated tests that make correct production code look wrong.

Both checks were predicted from the shape of earlier failures rather than
from a run that had already lost to them, then confirmed present in a real
workspace. They are the expensive class: the repair loop spends its rounds
editing production code that is fine, because the test is the thing that is
wrong.
"""

from __future__ import annotations

from harness.test_hygiene import run_test_hygiene_checks


ENV_CONFIGURED_APP = (
    "import os\n"
    "def settings():\n"
    "    return os.environ['DATABASE_PATH']\n"
)

UTC_CLOCK = (
    "from datetime import date, datetime, timezone\n"
    "class Clock:\n"
    "    @staticmethod\n"
    "    def today_utc() -> date:\n"
    "        return datetime.now(timezone.utc).date()\n"
)


def _ws(tmp_path, *, app: str = "", clock: str = "", tests: dict | None = None):
    (tmp_path / "app").mkdir(parents=True, exist_ok=True)
    (tmp_path / "app" / "config.py").write_text(app or "x = 1\n")
    if clock:
        (tmp_path / "app" / "clock.py").write_text(clock)
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    for name, body in (tests or {}).items():
        (tmp_path / "tests" / name).write_text(body)
    return str(tmp_path)


def _codes(ws):
    return [d["error_code"] for d in run_test_hygiene_checks(ws)]


class TestClientBoundAtImport:
    """lumina-run17-20260924-2351: contract tests returned 500 instead of
    422 because the app's connection was opened against a path fixed at
    import, and the judge spent its rounds guessing at row factories and
    config defaults."""

    MODULE_LEVEL = (
        "from fastapi.testclient import TestClient\n"
        "from app.main import app\n"
        "client = TestClient(app)\n"
        "def test_x():\n"
        "    assert client.get('/').status_code == 200\n"
    )

    def test_module_level_client_is_flagged(self, tmp_path):
        ws = _ws(tmp_path, app=ENV_CONFIGURED_APP,
                 tests={"test_api.py": self.MODULE_LEVEL})
        assert _codes(ws) == ["TEST_CLIENT_BOUND_AT_IMPORT"]

    def test_a_client_built_inside_a_test_is_fine(self, tmp_path):
        # The real workspace does this in two of three files; flagging them
        # would be the false positive that poisons the repair loop.
        ws = _ws(tmp_path, app=ENV_CONFIGURED_APP, tests={"test_api.py": (
            "from fastapi.testclient import TestClient\n"
            "from app.main import app\n"
            "def test_x():\n"
            "    with TestClient(app) as c:\n"
            "        assert c.get('/').status_code == 200\n"
        )})
        assert _codes(ws) == []

    def test_a_client_built_in_a_fixture_is_fine(self, tmp_path):
        ws = _ws(tmp_path, app=ENV_CONFIGURED_APP, tests={"conftest.py": (
            "import pytest\n"
            "from fastapi.testclient import TestClient\n"
            "from app.main import app\n"
            "@pytest.fixture\n"
            "def client():\n"
            "    return TestClient(app)\n"
        )})
        assert _codes(ws) == []

    def test_an_app_without_env_config_is_not_flagged(self, tmp_path):
        # Without a runtime-chosen database there is nothing for an
        # import-time binding to freeze.
        ws = _ws(tmp_path, app="x = 1\n",
                 tests={"test_api.py": self.MODULE_LEVEL})
        assert _codes(ws) == []

    def test_production_code_is_not_a_test_file(self, tmp_path):
        (tmp_path / "app").mkdir(parents=True)
        (tmp_path / "app" / "config.py").write_text(ENV_CONFIGURED_APP)
        (tmp_path / "app" / "probe.py").write_text(self.MODULE_LEVEL)
        assert _codes(str(tmp_path)) == []


class TestNaiveLocalDates:
    """The application runs on UTC; a test computing `date.today()` uses the
    machine's local date. At UTC+5:30 the two differ for five and a half
    hours of every day, so the expectation is off by one for those hours and
    the test passes or fails by the time of day it runs."""

    NAIVE = (
        "import datetime\n"
        "def test_upcoming(client):\n"
        "    today = datetime.date.today()\n"
        "    assert today is not None\n"
    )

    def test_a_naive_today_is_flagged(self, tmp_path):
        ws = _ws(tmp_path, clock=UTC_CLOCK, tests={"test_dates.py": self.NAIVE})
        assert _codes(ws) == ["NAIVE_LOCAL_DATE_IN_TEST"]

    def test_an_explicit_utc_call_is_fine(self, tmp_path):
        ws = _ws(tmp_path, clock=UTC_CLOCK, tests={"test_dates.py": (
            "from datetime import datetime, timezone\n"
            "def test_x():\n"
            "    today = datetime.now(timezone.utc).date()\n"
            "    assert today is not None\n"
        )})
        assert _codes(ws) == []

    def test_utcnow_is_flagged_though_it_names_utc(self, tmp_path):
        # datetime.utcnow() returns a NAIVE datetime; comparing it with an
        # aware one raises, and using it as "now" repeats the mismatch.
        ws = _ws(tmp_path, clock=UTC_CLOCK, tests={"test_dates.py": (
            "import datetime\n"
            "def test_x():\n"
            "    n = datetime.datetime.utcnow()\n"
            "    assert n is not None\n"
        )})
        assert _codes(ws) == ["NAIVE_LOCAL_DATE_IN_TEST"]

    def test_an_app_without_a_utc_clock_is_not_flagged(self, tmp_path):
        ws = _ws(tmp_path, tests={"test_dates.py": self.NAIVE})
        assert _codes(ws) == []

    def test_one_finding_per_file(self, tmp_path):
        ws = _ws(tmp_path, clock=UTC_CLOCK, tests={"test_dates.py": (
            "import datetime\n"
            "def test_a():\n    x = datetime.date.today()\n"
            "def test_b():\n    y = datetime.date.today()\n"
            "def test_c():\n    z = datetime.date.today()\n"
        )})
        assert len(_codes(ws)) == 1


class TestNeverFatal:
    def test_an_unparseable_test_file_is_skipped(self, tmp_path):
        ws = _ws(tmp_path, app=ENV_CONFIGURED_APP,
                 tests={"test_broken.py": "def (:\n"})
        assert _codes(ws) == []

    def test_an_empty_workspace_is_silent(self, tmp_path):
        assert run_test_hygiene_checks(str(tmp_path)) == []

    def test_the_preflight_carries_the_findings(self, tmp_path):
        from harness.static_preflight import run_static_preflight
        ws = _ws(tmp_path, app=ENV_CONFIGURED_APP, tests={
            "test_api.py": TestClientBoundAtImport.MODULE_LEVEL})
        codes = {d["error_code"] for d in run_static_preflight(ws)}
        assert "TEST_CLIENT_BOUND_AT_IMPORT" in codes

    def test_a_broken_checker_never_blocks_a_build(self, tmp_path, monkeypatch):
        from harness import test_hygiene
        from harness.static_preflight import run_static_preflight

        def _boom(*a, **k):
            raise RuntimeError("checker exploded")

        monkeypatch.setattr(test_hygiene, "run_test_hygiene_checks", _boom)
        run_static_preflight(_ws(tmp_path))  # must not raise
