"""A criterion's Given is seeded, not POSTed.

lumina-run24-20260929-2327: STORY-003.AC-2 says "Given the database contains
250 employee records". The generator's only sanctioned arrangement was the
API, so the test POSTed 250 times; NFR-004 throttles mutations to 30 a
minute, so the arrangement itself drew 429s. The repair loop then flipped the
rate limiter between 30 and 250 until the run died — no production value
satisfies both, because the conflict was in the test's arrangement.

Under per-test isolation the conftest now offers ``client.seed(table, rows)``
and the prompt sends a Given there, keeping the API for the When.
"""

from __future__ import annotations

import ast
import sqlite3

import pytest

from harness import acceptance_gen as ag

_DISCOVERY = {"module": "server.app.main", "symbol": "app", "kind": "singleton"}


class TestPrompt:
    def test_isolation_sends_a_given_to_seed(self):
        prompt = ag.build_system_prompt(db_isolated=True)
        assert "client.seed(" in prompt
        assert "arrange every prerequisite via the" not in prompt
        assert "WHEN" in prompt and "GIVEN" in prompt

    def test_a_shared_database_keeps_the_api_rule(self):
        # No per-test database to write into, so no seed.
        prompt = ag.build_system_prompt(db_isolated=False)
        assert "client.seed(" not in prompt
        assert "arrange every prerequisite via the" in prompt

    def test_no_slot_is_left_unfilled(self):
        for isolated in (True, False):
            for clock in (True, False):
                p = ag.build_system_prompt(db_isolated=isolated, clock_available=clock)
                assert "{arrange_rule}" not in p
                assert "{db_isolation_rule}" not in p and "{clock_rule}" not in p

    def test_the_real_tables_reach_the_user_prompt(self):
        ctx = ag.StoryAcceptanceContext(
            story_key="STORY-003", title="Directory", description="",
            db_tables={"employees": ["id INTEGER PRIMARY KEY",
                                     "first_name TEXT NOT NULL"]},
        )
        prompt = ag.build_user_prompt(ctx, max_scenarios=4)
        assert "## Database tables" in prompt
        assert "employees(id INTEGER PRIMARY KEY, first_name TEXT NOT NULL)" in prompt


class TestConftest:
    def test_isolated_conftest_compiles_and_points_seed_at_the_test_db(self):
        src = ag.render_acceptance_conftest(_DISCOVERY, db_env_var="DATABASE_PATH")
        ast.parse(src)
        assert "def seed(self, table, rows):" in src
        assert "c._acceptance_db = _db" in src

    def test_seed_is_never_withdrawn_without_isolation(self):
        # The conftest is re-rendered each pass while test files persist; a
        # method that disappears with discovery is lumina-run18's
        # freeze_today withdrawal again. It stays, and says why it can't run.
        src = ag.render_acceptance_conftest(_DISCOVERY, db_env_var=None)
        ast.parse(src)
        assert "def seed(self, table, rows):" in src
        assert "_acceptance_db = " not in src


# The method exactly as the conftest renders it, on a bare class.
_ns: dict = {}
exec("class _Seeder:\n" + ag._SEED_METHOD_SRC, _ns)  # noqa: S102 — the rendered method itself
_Seeder = _ns["_Seeder"]


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "acceptance.db")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "first_name TEXT NOT NULL, born TEXT NOT NULL, tags TEXT)")
    conn.commit()
    conn.close()
    return path


class TestSeedMethod:
    def test_rows_land_and_their_ids_come_back(self, db):
        import datetime
        s = _Seeder()
        s._acceptance_db = db
        ids = s.seed("employees", [
            {"first_name": "Alice", "born": datetime.date(1990, 1, 16),
             "tags": ["a", "b"]},
            {"first_name": "Bob", "born": "1985-02-10"},
        ])
        assert ids == [1, 2]
        rows = sqlite3.connect(db).execute(
            "SELECT first_name, born, tags FROM employees ORDER BY id").fetchall()
        assert rows == [("Alice", "1990-01-16", '["a", "b"]'),
                        ("Bob", "1985-02-10", None)]

    def test_a_wrong_table_names_the_real_schema(self, db):
        s = _Seeder()
        s._acceptance_db = db
        with pytest.raises(RuntimeError) as exc:
            s.seed("employee", [{"first_name": "A", "born": "2000-01-01"}])
        msg = str(exc.value)
        assert "no such table" in msg
        assert "employees(id, first_name, born, tags)" in msg

    def test_without_a_test_database_it_says_so(self):
        with pytest.raises(RuntimeError, match="per-test database"):
            _Seeder().seed("employees", [{"first_name": "A"}])


class TestDiscoverDbTables:
    def test_raw_sql_keeps_the_constraints_that_decide_an_insert(self, tmp_path):
        (tmp_path / "db.py").write_text(
            '_SCHEMA = """\nCREATE TABLE IF NOT EXISTS employees (\n'
            "    id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
            "    first_name TEXT NOT NULL,\n"
            "    created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),\n"
            "    UNIQUE(first_name, created_at)\n);\n\"\"\"\n")
        assert ag.discover_db_tables(str(tmp_path)) == {"employees": [
            "id INTEGER PRIMARY KEY AUTOINCREMENT",
            "first_name TEXT NOT NULL",
            "created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc'))",
            "UNIQUE(first_name, created_at)",
        ]}

    def test_a_sqlalchemy_model_is_read_too(self, tmp_path):
        (tmp_path / "models.py").write_text(
            "from sqlalchemy import Column, Integer, String\n"
            "from sqlalchemy.orm import Mapped, mapped_column\n\n"
            "class Employee(Base):\n"
            "    __tablename__ = 'employees'\n"
            "    id = Column(Integer, primary_key=True)\n"
            "    first_name: Mapped[str] = mapped_column(String(100), nullable=False)\n"
            "    note = Column(String, server_default='')\n")
        assert ag.discover_db_tables(str(tmp_path)) == {"employees": [
            "id INTEGER PRIMARY KEY", "first_name STRING NOT NULL",
            "note STRING DEFAULT"]}

    def test_tests_and_vendored_trees_are_not_the_schema(self, tmp_path):
        for d in ("tests", ".venv", "node_modules"):
            (tmp_path / d).mkdir()
            (tmp_path / d / "x.sql").write_text("CREATE TABLE fake (a INT);")
        assert ag.discover_db_tables(str(tmp_path)) == {}
