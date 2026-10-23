"""Tests for the acceptance seed helper's binding to the real schema, and
for refusing to auto-resume a blocker no node may author.

Both reproduce lumina session 01a164b2, which reached acceptance with a green
unit suite and then died at 0/16 stories:

* ``client.seed`` wrote rows with raw SQL, so none of the ORM's Python-side
  behaviour applied. ``id`` is ``Mapped[str]`` with ``default=uuid4`` — a
  default that never reaches the DDL — and the helper returned
  ``cur.lastrowid``, sqlite's integer rowid. Every ``GET /thing/{ids[0]}``
  after a seed therefore addressed a row that did not exist (404).
* ``created_at`` / ``updated_at`` were NOT NULL with Python-side defaults too,
  so the insert failed. The repair LLM edited the conftest to ``setdefault``
  them on every row, table-blind, which then broke seeding a table that has
  neither column. repair_node may not edit tests and the regeneration node
  does not author acceptance suites, so the run had no way out and spent its
  auto-resume allowance cycling through the same build.
"""

import sqlite3
import tempfile

import pytest

from harness.acceptance_gen import _SEED_METHOD_SRC, discover_db_tables


def _guide_text(name):
    """Read a test guide the way test_generation.py locates them."""
    import os

    import harness.test_generation as tg

    path = os.path.join(
        os.path.dirname(tg.__file__), "test_guides", name,
    )
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _client(db_path):
    """A stand-in for the generated conftest's client, carrying the real
    rendered ``seed`` method at the indent the conftest renders it at."""
    src = (
        "class _C:\n"
        "    def __init__(self, db):\n"
        "        self._acceptance_db = db\n"
        + _SEED_METHOD_SRC
    )
    ns: dict = {}
    exec(compile(src, "<conftest>", "exec"), ns)
    return ns["_C"](db_path)


@pytest.fixture()
def db(tmp_path):
    """lumina's real shapes: a text primary key with a Python-side default,
    NOT NULL bookkeeping columns, and a second table that has neither."""
    path = str(tmp_path / "acceptance.db")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE birthdays ("
        " id VARCHAR(36) NOT NULL PRIMARY KEY,"
        " owner_id VARCHAR(64),"
        " first_name VARCHAR(80) NOT NULL,"
        " birth_month INTEGER NOT NULL,"
        " category VARCHAR(32) NOT NULL,"
        " created_at DATETIME NOT NULL,"
        " updated_at DATETIME NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE device_profiles ("
        " id VARCHAR(36) NOT NULL PRIMARY KEY,"
        " owner_id VARCHAR(64) NOT NULL,"
        " timezone VARCHAR(64) NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE counters (id INTEGER PRIMARY KEY, label TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()
    return path


class TestSeedReturnsTheRowsRealIdentity:
    def test_text_primary_key_comes_back_not_the_rowid(self, db):
        ids = _client(db).seed(
            "birthdays",
            [{"owner_id": "o1", "first_name": "Sam", "birth_month": 1}],
        )
        # An integer here is the 404 bug: the API routes by the uuid.
        assert isinstance(ids[0], str)
        stored = sqlite3.connect(db).execute(
            "SELECT id FROM birthdays"
        ).fetchone()[0]
        assert ids[0] == stored

    def test_supplied_primary_key_is_preserved(self, db):
        ids = _client(db).seed(
            "birthdays",
            [{"id": "fixed-id", "owner_id": "o1", "first_name": "Sam",
              "birth_month": 1}],
        )
        assert ids == ["fixed-id"]

    def test_integer_primary_key_still_autoincrements(self, db):
        ids = _client(db).seed("counters", [{"label": "a"}, {"label": "b"}])
        # INTEGER PRIMARY KEY is sqlite's rowid alias — don't hand it a uuid.
        assert ids == [1, 2]

    def test_each_row_gets_a_distinct_key(self, db):
        ids = _client(db).seed(
            "birthdays",
            [{"first_name": "A", "birth_month": 1},
             {"first_name": "B", "birth_month": 2}],
        )
        assert len(set(ids)) == 2


class TestSeedSuppliesWhatTheOrmWouldHave:
    def test_not_null_timestamps_are_filled(self, db):
        _client(db).seed("birthdays", [{"first_name": "Sam", "birth_month": 1}])
        row = sqlite3.connect(db).execute(
            "SELECT created_at, updated_at FROM birthdays"
        ).fetchone()
        assert row[0] and row[1]

    def test_other_not_null_columns_are_filled(self, db):
        # ``category`` is NOT NULL with a Python-side default; without a fill
        # the insert fails outright and the LLM goes editing the conftest.
        _client(db).seed("birthdays", [{"first_name": "Sam", "birth_month": 1}])
        assert sqlite3.connect(db).execute(
            "SELECT count(*) FROM birthdays WHERE category IS NOT NULL"
        ).fetchone()[0] == 1

    def test_supplied_values_always_win(self, db):
        _client(db).seed(
            "birthdays",
            [{"first_name": "Sam", "birth_month": 1, "category": "family",
              "created_at": "2026-01-01T00:00:00+00:00"}],
        )
        row = sqlite3.connect(db).execute(
            "SELECT category, created_at FROM birthdays"
        ).fetchone()
        assert row == ("family", "2026-01-01T00:00:00+00:00")

    def test_nullable_columns_are_left_alone(self, db):
        _client(db).seed("birthdays", [{"first_name": "Sam", "birth_month": 1}])
        assert sqlite3.connect(db).execute(
            "SELECT owner_id FROM birthdays"
        ).fetchone()[0] is None

    def test_a_table_without_those_columns_still_seeds(self, db):
        # The regression that killed the run: a table-blind created_at
        # setdefault made this raise.
        ids = _client(db).seed(
            "device_profiles",
            [{"owner_id": "owner-1", "timezone": "America/New_York"}],
        )
        assert len(ids) == 1
        assert sqlite3.connect(db).execute(
            "SELECT timezone FROM device_profiles"
        ).fetchone()[0] == "America/New_York"


class TestSeedRejectsWhatTheSchemaCannotHold:
    def test_unknown_column_names_the_real_ones(self, db):
        with pytest.raises(RuntimeError) as exc:
            _client(db).seed(
                "device_profiles",
                [{"owner_id": "x", "timezone": "UTC", "created_at": "now"}],
            )
        msg = str(exc.value)
        assert "created_at" in msg
        # Actionable: the generator needs the real column list to fix itself.
        assert "timezone" in msg and "owner_id" in msg

    def test_unknown_table_names_the_real_ones(self, db):
        with pytest.raises(RuntimeError) as exc:
            _client(db).seed("nope", [{"a": 1}])
        assert "birthdays" in str(exc.value)

    def test_missing_database_says_so(self):
        c = _client(tempfile.mktemp())
        c._acceptance_db = None
        with pytest.raises(RuntimeError, match="per-test database"):
            c.seed("birthdays", [{"first_name": "Sam"}])


class TestPythonSideDefaultsAreReportedAsSuch:
    """A raw INSERT does not get an ORM ``default=``; the prompt must not
    claim the database will fill it."""

    def test_orm_default_is_not_called_a_database_default(self, tmp_path):
        (tmp_path / "m.py").write_text(
            "from sqlalchemy.orm import Mapped, mapped_column\n"
            "class M(Base):\n"
            "    __tablename__ = 'things'\n"
            "    id: Mapped[str] = mapped_column(String(36), primary_key=True,"
            " default=lambda: 'x')\n"
            "    label: Mapped[str] = mapped_column(String(8), nullable=False,"
            " default='hi')\n"
            "    made: Mapped[str] = mapped_column(String(8), nullable=False,"
            " server_default='now')\n"
        )
        cols = discover_db_tables(str(tmp_path))["things"]
        by_name = {c.split()[0]: c for c in cols}
        assert "PYTHON-DEFAULT" in by_name["label"]
        assert "PYTHON-DEFAULT" in by_name["id"]
        # A real DDL default still reads as one.
        assert "DEFAULT" in by_name["made"]
        assert "PYTHON-DEFAULT" not in by_name["made"]


class TestUnauthorableTestBlockerIsTerminal:
    """A blocker only a human can edit must not spend auto-resumes."""

    def test_trigger_is_specific_and_names_the_file(self):
        from harness.graph import _infer_hitl_trigger

        state = {
            "node_state": {
                "test_regen_unsupported": "tests/acceptance/conftest.py",
            },
            "exit_code": 1,
            "budget_remaining_usd": 9.0,
        }
        assert _infer_hitl_trigger(state, max_repair=3) == (
            "unauthorable_test_blocker:tests/acceptance/conftest.py"
        )

    def test_it_outranks_persistent_build_failure(self):
        # Both hold at once — exit_code != 0 is what got us here. The
        # generic label sent lumina 01a164b2 round the loop four times.
        from harness.graph import _infer_hitl_trigger

        state = {
            "node_state": {"test_regen_unsupported": "tests/acceptance/x.py"},
            "exit_code": 1,
            "budget_remaining_usd": 9.0,
            "loop_counter": {"total_repairs": 0},
        }
        assert _infer_hitl_trigger(state, max_repair=3).startswith(
            "unauthorable_test_blocker"
        )

    def test_without_the_flag_nothing_changes(self):
        from harness.graph import _infer_hitl_trigger

        state = {"node_state": {}, "exit_code": 1, "budget_remaining_usd": 9.0}
        assert _infer_hitl_trigger(state, max_repair=3) == (
            "persistent_build_failure"
        )

    def test_the_stuck_sibling_is_covered_too(self):
        """``test_regen_stuck`` (node routed in with no test to write) is
        the same dead end: resuming re-runs the identical no-op."""
        from harness.graph import _infer_hitl_trigger

        state = {
            "node_state": {"test_regen_stuck": True},
            "exit_code": 1,
            "budget_remaining_usd": 9.0,
        }
        assert _infer_hitl_trigger(state, max_repair=3) == (
            "unauthorable_test_blocker"
        )

    def test_stuck_false_does_not_trigger_it(self):
        """The node writes the key even when it is NOT stuck, so a falsy
        value must read as 'fine', not as the dead end."""
        from harness.graph import _infer_hitl_trigger

        state = {
            "node_state": {"test_regen_stuck": False},
            "exit_code": 1,
            "budget_remaining_usd": 9.0,
        }
        assert _infer_hitl_trigger(state, max_repair=3) == (
            "persistent_build_failure"
        )

    def test_exit_code_is_infrastructure_failure(self):
        from harness.cli import (
            EXIT_INFRASTRUCTURE_FAILURE,
            _resolve_cli_exit_code,
        )

        assert _resolve_cli_exit_code(
            graph_exit_code=1,
            final_state={
                "node_state": {
                    "unauthorable_test_blocker": "tests/acceptance/conftest.py",
                },
            },
        ) == EXIT_INFRASTRUCTURE_FAILURE

    def test_post_mortem_advice_exists_and_keeps_the_path(self):
        from harness.post_mortem import _TRIGGER_ADVICE

        advice = _TRIGGER_ADVICE["unauthorable_test_blocker"]
        assert "no node may author" in advice
        assert "Auto-resume cannot help" in advice

    def test_resume_clears_the_flags(self):
        """An operator who fixed the file and resumed must not be
        terminated again on the stale refusal."""
        import inspect

        import harness.cli as cli

        src = inspect.getsource(cli)
        # Both keys appear in the resume-time cleanup tuple.
        assert '"test_regen_unsupported",' in src
        assert '"unauthorable_test_blocker",' in src


class TestDateDeterminismGuidance:
    """The generator produced a seasonal flake because nothing told it not
    to: seeded May 10 + Dec 25, asserted a literal sort order without
    freezing time. It passes Dec 26-May 10 and fails the rest of the year."""

    @pytest.mark.parametrize(
        "guide,tool",
        [
            ("python.md", "freeze_time"),
            ("javascript.md", "setSystemTime"),
            ("typescript.md", "setSystemTime"),
            ("java.md", "Clock.fixed"),
        ],
    )
    def test_every_guide_names_its_clock_control(self, guide, tool):
        text = _guide_text(guide)
        assert tool in text, f"{guide} must tell the generator to use {tool}"
        assert "depend on the day it runs" in text

    @pytest.mark.parametrize(
        "guide", ["python.md", "javascript.md", "typescript.md", "java.md"],
    )
    def test_every_guide_offers_the_invariant_alternative(self, guide):
        text = _guide_text(guide).lower()
        # Freezing is not always right; asserting the rule often is.
        assert "sorted" in text
