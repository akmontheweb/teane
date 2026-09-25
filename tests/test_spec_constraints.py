"""A constraint the spec states must reach the generated model.

lumina-run20-20260925-1342 ended on three distraction-loop trips with the
judge flipping between "relax the last_name validator" and "add
field_validators that reject...". Underneath: the spec's data model says
`first_name | TEXT | NOT NULL, length 1-100 after trimming`, the generated
model declared `first_name: str = Field(..., description=...)`, and the
property-test generator read that bare annotation as "any string up to 200
characters must construct". Repair could satisfy the property test or the
validation tests, never both.
"""

from __future__ import annotations

from harness.spec_constraints import (
    constraint_diagnostics,
    parse_documented_lengths,
    unconstrained_fields,
)


SPEC = """# Software Requirements Specification

## Data model

### Employee table

| Column | Type | Constraints |
| --- | --- | --- |
| `id` | INTEGER | PRIMARY KEY |
| `first_name` | TEXT | NOT NULL, length 1-100 after trimming |
| `last_name` | TEXT | NOT NULL, length 1-100 after trimming |
| `date_of_birth` | TEXT | NOT NULL, ISO YYYY-MM-DD |
"""


def _ws(tmp_path, schema_src, spec=SPEC):
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "SPEC_REQUIREMENTS.md").write_text(spec)
    (tmp_path / "app").mkdir(parents=True, exist_ok=True)
    (tmp_path / "app" / "schemas.py").write_text(schema_src)
    return str(tmp_path)


UNCONSTRAINED = (
    "from pydantic import BaseModel, Field\n"
    "class BirthdayCreate(BaseModel):\n"
    "    first_name: str = Field(..., description='first')\n"
    "    last_name: str = Field(..., description='last')\n"
    "    date_of_birth: str = Field(...)\n"
)


class TestParsingTheTable:
    def test_reads_a_numeric_range(self):
        got = parse_documented_lengths(SPEC)
        assert got["first_name"] == {"min_length": 1, "max_length": 100}
        assert got["last_name"] == {"min_length": 1, "max_length": 100}

    def test_an_en_dash_range_is_read(self):
        # Spec tables use U+2013, which a plain `-` pattern misses.
        got = parse_documented_lengths(
            "| `nickname` | TEXT | NOT NULL, length 2–64 after trimming |")
        assert got["nickname"] == {"min_length": 2, "max_length": 64}

    def test_a_column_called_name_is_not_skipped(self):
        # Found while writing these: skipping header rows BY NAME also
        # excluded a real column called `name`, silently dropping the very
        # rule this module exists to carry.
        got = parse_documented_lengths("| `name` | TEXT | NOT NULL, length 1-50 |")
        assert got["name"] == {"min_length": 1, "max_length": 50}

    def test_other_phrasings(self):
        assert parse_documented_lengths(
            "| `a` | TEXT | between 3 and 9 |")["a"] == {
                "min_length": 3, "max_length": 9}
        assert parse_documented_lengths(
            "| `b` | TEXT | maximum length 40 |")["b"] == {"max_length": 40}

    def test_prose_without_numbers_is_not_guessed(self):
        assert parse_documented_lengths(
            "| `c` | TEXT | NOT NULL, reasonable length |") == {}

    def test_header_rows_are_skipped(self):
        assert "Column" not in parse_documented_lengths(SPEC)

    def test_junk(self):
        assert parse_documented_lengths("") == {}
        assert parse_documented_lengths("no table here at all") == {}


class TestFindingTheOmission:
    def test_the_run20_fields_are_found(self, tmp_path):
        found = unconstrained_fields(_ws(tmp_path, UNCONSTRAINED))
        assert {f["field"] for f in found} == {"first_name", "last_name"}
        assert all(f["model"] == "BirthdayCreate" for f in found)
        assert found[0]["min_length"] == 1 and found[0]["max_length"] == 100

    def test_a_declared_constraint_is_not_a_finding(self, tmp_path):
        src = (
            "from pydantic import BaseModel, Field\n"
            "class BirthdayCreate(BaseModel):\n"
            "    first_name: str = Field(..., min_length=1, max_length=100)\n"
        )
        assert unconstrained_fields(_ws(tmp_path, src)) == []

    def test_a_different_constraint_is_left_alone(self, tmp_path):
        # Declaring something other than the spec is a judgement call, not an
        # omission — and a diagnostic that argues with a deliberate choice is
        # the false positive this module is built to avoid.
        src = (
            "from pydantic import BaseModel, Field\n"
            "class BirthdayCreate(BaseModel):\n"
            "    first_name: str = Field(..., max_length=50)\n"
        )
        assert unconstrained_fields(_ws(tmp_path, src)) == []

    def test_response_models_are_exempt(self, tmp_path):
        # A response echoes what is already stored; demanding validators on
        # it is noise.
        src = (
            "from pydantic import BaseModel\n"
            "class BirthdayResponse(BaseModel):\n"
            "    first_name: str\n"
            "class BirthdayPage(BaseModel):\n"
            "    first_name: str\n"
        )
        assert unconstrained_fields(_ws(tmp_path, src)) == []

    def test_non_string_fields_are_exempt(self, tmp_path):
        src = (
            "from pydantic import BaseModel\n"
            "class Thing(BaseModel):\n"
            "    first_name: int\n"
        )
        assert unconstrained_fields(_ws(tmp_path, src)) == []

    def test_undocumented_fields_are_exempt(self, tmp_path):
        src = (
            "from pydantic import BaseModel\n"
            "class Thing(BaseModel):\n"
            "    nickname: str\n"
        )
        assert unconstrained_fields(_ws(tmp_path, src)) == []

    def test_no_spec_means_no_findings(self, tmp_path):
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "schemas.py").write_text(UNCONSTRAINED)
        assert unconstrained_fields(str(tmp_path)) == []

    def test_tests_dir_is_skipped(self, tmp_path):
        ws = _ws(tmp_path, "x = 1\n")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "schemas.py").write_text(UNCONSTRAINED)
        assert unconstrained_fields(ws) == []


class TestTheDiagnostic:
    def test_it_names_the_fix_and_the_reason(self, tmp_path):
        diags = constraint_diagnostics(_ws(tmp_path, UNCONSTRAINED))
        assert len(diags) == 2
        d = diags[0]
        assert d["error_code"] == "SPEC_CONSTRAINT_NOT_DECLARED"
        assert d["severity"] == "error"
        assert "min_length=1, max_length=100" in d["message"]
        assert "Field(..." in d["message"]
        assert "oscillates" in d["message"], (
            "the message should say what goes wrong, not just what is missing"
        )

    def test_it_is_bounded(self, tmp_path):
        assert constraint_diagnostics(
            _ws(tmp_path, UNCONSTRAINED), limit=1) == constraint_diagnostics(
                _ws(tmp_path, UNCONSTRAINED), limit=1)
        assert len(constraint_diagnostics(
            _ws(tmp_path, UNCONSTRAINED), limit=1)) == 1

    def test_clean_workspace_is_silent(self, tmp_path):
        src = (
            "from pydantic import BaseModel, Field\n"
            "class BirthdayCreate(BaseModel):\n"
            "    first_name: str = Field(..., min_length=1, max_length=100)\n"
            "    last_name: str = Field(..., min_length=1, max_length=100)\n"
        )
        assert constraint_diagnostics(_ws(tmp_path, src)) == []


class TestPreflightWiring:
    def test_the_check_runs_in_the_preflight(self, tmp_path):
        from harness.static_preflight import run_static_preflight
        diags = run_static_preflight(_ws(tmp_path, UNCONSTRAINED))
        codes = {d["error_code"] for d in diags}
        assert "SPEC_CONSTRAINT_NOT_DECLARED" in codes

    def test_a_broken_check_never_blocks_a_build(self, tmp_path, monkeypatch):
        from harness import spec_constraints
        from harness.static_preflight import run_static_preflight

        def _boom(*a, **k):
            raise RuntimeError("checker exploded")

        monkeypatch.setattr(spec_constraints, "constraint_diagnostics", _boom)
        run_static_preflight(_ws(tmp_path, UNCONSTRAINED))  # must not raise
