"""lumina 01a21929 (run 20): the harness wrote a test it could not withdraw.

The run ended 4/16 with its last 71 minutes and 30% of its $5 budget spent on
``server/app/schemas/birthday.py`` — a file that was not broken. Five defects
compounded, and each class below pins one of them.

1. :class:`TestPydanticV2Optionality` — the Tier-1 generator classified
   ``X | None`` with no default as OPTIONAL. In Pydantic v2 only a default
   makes a field optional, so the happy-path test it emitted omitted five
   required fields and failed on first contact with "Field required". The
   judge's REQUIRED ACTION then became "make last_name, birth_year and
   phone_number optional on BirthdayResponse": the generator's bug was
   propagated into the application's own API contract, because the test file
   is tamper-guarded and its claim was read as ground truth.

2. :class:`TestTheStalenessSweep` — ``_valid_kwargs`` skips a model whose
   required field carries a ``@field_validator``, precisely because its happy
   path is no longer derivable. That check ran once, at emission (00:16:18).
   Repair added such a validator at 04:02:29, inside the batch's repair loop,
   and the file written before it stayed on disk asserting
   ``BirthdayResponse(created_at='aaa')`` constructs. Nothing re-derived it,
   and no production change could satisfy it.

3. :class:`TestTheVetoIsPerFile` — the UNSATISFIABLE_TEST escape was vetoed
   for the whole round by the FIRST failing test carrying production-fault
   evidence. ``test_story_006_acceptance.py`` legitimately carried it (200
   where the criterion requires 404). That veto fired 20+ times and withheld
   the escape from the contract test in (2), which genuinely was
   unsatisfiable. The one correct diagnosis available was unreachable.

4. :class:`TestRelatedFilesReachesFixtureWiredTests` /
   :class:`TestTheRouteIndex` — RELATED FILES came from a static import walk.
   An ADR-0006 acceptance test imports nothing but ``pytest``: its app
   arrives through a fixture and the generated conftest names it in a
   string. So the walk returned one production file for the whole failing
   set, the grounding rule made every other file unnameable, and the judge
   named that one file for eleven consecutive verdicts. The real defects were
   in ``api/birthdays.py`` (no owner filter on GET-by-id) and ``main.py`` (no
   ``RequestValidationError`` handler).

The negative tests matter as much as the positive ones. (3) must not reopen
the run-16 hole it was built to close — evidence still vetoes blame on the
file that HAS evidence — and (2) must never touch a file the harness did not
write.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from harness import graph as g
from harness import route_index as ri
from harness.contract_tests import (
    _valid_kwargs,
    parse_pydantic_models,
    refresh_generated_contract_tests,
    render_contract_test,
)

# --------------------------------------------------------------------------
# Verbatim fixtures from the run.
# --------------------------------------------------------------------------

#: ``server/app/schemas/birthday.py`` as the patcher first created it at
#: 00:14:32 — the version the contract test was generated from at 00:16:18.
#: Every nullable field is ``X | None`` with NO default.
SCHEMA_AT_EMISSION = textwrap.dedent('''
    """Birthday request and response schemas."""
    from pydantic import BaseModel, ConfigDict


    class BirthdayResponse(BaseModel):
        """Full persisted birthday record."""

        model_config = ConfigDict(from_attributes=True)

        id: str
        first_name: str
        last_name: str | None
        birth_month: int
        birth_day: int
        birth_year: int | None
        category: str
        phone_number: str | None
        email: str | None
        notes: str | None
        created_at: str
        updated_at: str
''')

#: The same model after repair added ``_coerce_dt`` at 04:02:29. The
#: validator is what made the already-emitted happy-path test unsatisfiable.
SCHEMA_AFTER_REPAIR = textwrap.dedent('''
    """Birthday request and response schemas."""
    from datetime import datetime

    from pydantic import BaseModel, ConfigDict, field_validator


    class BirthdayResponse(BaseModel):
        model_config = ConfigDict(from_attributes=True)

        id: str
        first_name: str
        created_at: str
        updated_at: str

        @field_validator("created_at", "updated_at", mode="before")
        @classmethod
        def _coerce_dt(cls, value: object) -> str:
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, str):
                datetime.fromisoformat(value)
                return value
            raise ValueError("must be a datetime or ISO 8601 string")
''')

T006 = "tests/acceptance/test_story_006_acceptance.py"
CONTRACT = "server/tests/contract/test_birthday_contract.py"

#: The 006 failure, verbatim shape: the app answered 200 where AC-3 requires
#: 404 for another owner's record. Real production-fault evidence.
DIAG_006 = {
    "file": T006, "line": 74, "severity": "error",
    "error_code": "AssertionError",
    "message": (
        'AssertionError: {"id":"930f2379","first_name":"Grace",'
        '"created_at":"2025-01-01T00:00:00"}\n'
        "assert 200 == 404"
    ),
    "semantic_context": (
        'failing source: resp = client.get(f"/api/v1/birthdays/{record_id}")\n'
        "assertion-rewrite:\n  assert 200 == 404\n"
        "runtime object detail (harness enrichment):\n"
        "  resp = <Response status_code=200>\n"
    ),
}

#: The contract-test failure. A bare ValidationError on a construction call:
#: no crash body, no double-wrapped envelope, nothing the application did.
DIAG_CONTRACT = {
    "file": CONTRACT, "line": 114, "severity": "error",
    "error_code": "ValidationError",
    "message": (
        "ValidationError: pydantic_core._pydantic_core.ValidationError: "
        "2 validation errors for BirthdayResponse"
    ),
    "semantic_context": (
        "failing source: obj = BirthdayResponse(id='aaa', first_name='aaa', "
        "created_at='aaa', updated_at='aaa')\n"
        "assertion-rewrite:\n  created_at\n"
        "  Value error, Invalid isoformat string: 'aaa'\n"
    ),
}


T007 = "tests/acceptance/test_story_007_acceptance.py"

#: A SECOND evidenced failure, so the map has to report more than one. Run 20
#: had exactly this: 006 and 007 both carried evidence while the contract test
#: carried none.
DIAG_007 = {
    "file": T007, "line": 86, "severity": "error",
    "error_code": "AssertionError",
    "message": (
        'AssertionError: {"error":{"code":"APP_ERROR",'
        '"message":"Internal server error"}}\n'
        "assert 500 == 400"
    ),
    "semantic_context": (
        'failing source: resp2 = client.get("/api/v1/birthdays?page_size=0")\n'
        "runtime object detail (harness enrichment):\n"
        "  resp2 = <Response status_code=500>\n"
    ),
}


def _state(*diags, ws="/ws"):
    return {"workspace_path": ws, "compiler_errors": list(diags)}


# ==========================================================================
# 1. Pydantic v2 optionality
# ==========================================================================

class TestPydanticV2Optionality:
    def _fields(self, src):
        [m] = parse_pydantic_models(src, rel_path="app/m.py")
        return {f.name: f for f in m.fields}

    def test_optional_without_a_default_is_required(self):
        """``last_name: str | None`` — the exact shape that broke run 20."""
        fields = self._fields(SCHEMA_AT_EMISSION)
        for name in (
            "last_name", "birth_year", "phone_number", "email", "notes",
        ):
            assert fields[name].required is True, name

    def test_optional_with_a_default_is_not_required(self):
        fields = self._fields(textwrap.dedent('''
            from pydantic import BaseModel, Field
            class M(BaseModel):
                a: str | None = None
                b: str | None = Field(default=None, max_length=5)
                c: int | None = Field(default=None, ge=1)
        '''))
        assert [f.required for f in fields.values()] == [False, False, False]

    def test_optional_bracket_form_without_a_default_is_required(self):
        fields = self._fields(textwrap.dedent('''
            from typing import Optional
            from pydantic import BaseModel
            class M(BaseModel):
                a: Optional[str]
        '''))
        assert fields["a"].required is True

    def test_the_happy_path_names_every_required_field(self):
        """The regression itself: the emitted construction must not omit a
        field pydantic will demand."""
        models = parse_pydantic_models(
            SCHEMA_AT_EMISSION, rel_path="server/app/schemas/birthday.py",
        )
        body = render_contract_test(
            models, source_rel="server/app/schemas/birthday.py",
        )
        assert body
        line = next(
            ln for ln in body.splitlines()
            if "obj = BirthdayResponse(" in ln
        )
        for name in (
            "last_name", "birth_year", "phone_number", "email", "notes",
        ):
            assert f"{name}=" in line, f"{name} omitted from the happy path"

    def test_the_emitted_file_actually_passes_under_real_pydantic(
        self, tmp_path,
    ):
        """The only assertion that cannot be satisfied by agreeing with the
        generator: run its output against pydantic itself.

        Before the fix this failed with "5 validation errors for
        BirthdayResponse / last_name Field required".
        """
        pytest.importorskip("pydantic")
        (tmp_path / "mod.py").write_text(SCHEMA_AT_EMISSION)
        models = parse_pydantic_models(SCHEMA_AT_EMISSION, rel_path="mod.py")
        body = render_contract_test(models, source_rel="mod.py")
        assert body
        (tmp_path / "test_emitted.py").write_text(body)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "test_emitted.py"],
            cwd=tmp_path, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(tmp_path)},
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_a_validator_on_a_nullable_field_now_skips_the_model(self):
        """Knock-on effect, and the one that matters: a ``@field_validator``
        on an ``X | None`` field used to be invisible to the skip rule
        because the field was not considered required."""
        [m] = parse_pydantic_models(textwrap.dedent('''
            from pydantic import BaseModel, field_validator
            class M(BaseModel):
                a: str | None

                @field_validator("a")
                @classmethod
                def _check(cls, v):
                    raise ValueError("never valid")
        '''))
        assert _valid_kwargs(m) is None


# ==========================================================================
# 2. The staleness sweep
# ==========================================================================

def _workspace_with_generated_contract(tmp_path, schema_src):
    """A workspace holding the schema plus the Tier-1 file generated from
    the AT-EMISSION version of it."""
    src_rel = "server/app/schemas/birthday.py"
    (tmp_path / "server/app/schemas").mkdir(parents=True)
    (tmp_path / "server/tests/contract").mkdir(parents=True)
    (tmp_path / src_rel).write_text(schema_src)
    emitted = render_contract_test(
        parse_pydantic_models(SCHEMA_AT_EMISSION, rel_path=src_rel),
        source_rel=src_rel,
    )
    assert emitted
    (tmp_path / CONTRACT).write_text(emitted)
    return src_rel


class TestTheStalenessSweep:
    def test_a_model_that_lost_its_happy_path_retires_its_test(
        self, tmp_path,
    ):
        """Run 20's killer. ``created_at`` gains a validator, so nothing is
        derivable for that model any more; the file asserting the old shape
        must not survive the change."""
        _workspace_with_generated_contract(tmp_path, SCHEMA_AT_EMISSION)
        (tmp_path / "server/app/schemas/birthday.py").write_text(
            SCHEMA_AFTER_REPAIR,
        )
        refreshed, retired = refresh_generated_contract_tests(str(tmp_path))
        assert retired == [CONTRACT], (refreshed, retired)
        assert not (tmp_path / CONTRACT).exists()

    def test_a_model_that_merely_changed_shape_is_rewritten(self, tmp_path):
        src_rel = _workspace_with_generated_contract(
            tmp_path, SCHEMA_AT_EMISSION,
        )
        # Keep BirthdayResponse derivable but add a constraint.
        (tmp_path / src_rel).write_text(
            SCHEMA_AT_EMISSION.replace(
                "first_name: str",
                "first_name: str = Field(max_length=7)",
            ).replace(
                "from pydantic import BaseModel, ConfigDict",
                "from pydantic import BaseModel, ConfigDict, Field",
            )
        )
        refreshed, retired = refresh_generated_contract_tests(str(tmp_path))
        assert refreshed == [CONTRACT] and retired == []
        text = (tmp_path / CONTRACT).read_text()
        assert "first_name_max_length" in text

    def test_an_unchanged_model_is_left_alone(self, tmp_path):
        _workspace_with_generated_contract(tmp_path, SCHEMA_AT_EMISSION)
        before = (tmp_path / CONTRACT).read_text()
        refreshed, retired = refresh_generated_contract_tests(str(tmp_path))
        assert (refreshed, retired) == ([], [])
        assert (tmp_path / CONTRACT).read_text() == before

    def test_a_vanished_source_retires_its_test(self, tmp_path):
        src_rel = _workspace_with_generated_contract(
            tmp_path, SCHEMA_AT_EMISSION,
        )
        (tmp_path / src_rel).unlink()
        _, retired = refresh_generated_contract_tests(str(tmp_path))
        assert retired == [CONTRACT]

    def test_a_file_the_harness_did_not_write_is_never_touched(
        self, tmp_path,
    ):
        """The sweep deletes and overwrites, so its ownership check is the
        thing standing between it and an operator's own test file."""
        _workspace_with_generated_contract(tmp_path, SCHEMA_AFTER_REPAIR)
        hand_written = "# @tests: server/app/schemas/birthday.py\ndef test_x():\n    pass\n"
        (tmp_path / CONTRACT).write_text(hand_written)
        refreshed, retired = refresh_generated_contract_tests(str(tmp_path))
        assert (refreshed, retired) == ([], [])
        assert (tmp_path / CONTRACT).read_text() == hand_written

    def test_a_traversing_header_is_ignored(self, tmp_path):
        """The ``# @tests:`` banner is data read off disk and the sweep uses
        it to open a file, so it must stay inside the workspace."""
        _workspace_with_generated_contract(tmp_path, SCHEMA_AT_EMISSION)
        text = (tmp_path / CONTRACT).read_text()
        (tmp_path / CONTRACT).write_text(
            text.replace(
                "# @tests: server/app/schemas/birthday.py",
                "# @tests: ../../../../etc/passwd",
                1,
            )
        )
        refreshed, retired = refresh_generated_contract_tests(str(tmp_path))
        assert (refreshed, retired) == ([], [])
        assert (tmp_path / CONTRACT).exists()

    def test_the_swept_file_passes_under_real_pydantic(self, tmp_path):
        """End to end: an unsatisfiable file becomes a satisfiable one."""
        pytest.importorskip("pydantic")
        _workspace_with_generated_contract(tmp_path, SCHEMA_AT_EMISSION)
        for pkg in ("server", "server/app", "server/app/schemas"):
            (tmp_path / pkg / "__init__.py").write_text("")
        # Before the sweep the emitted file is already fine; break it the way
        # run 20 did, then sweep, then run what is left.
        (tmp_path / "server/app/schemas/birthday.py").write_text(
            SCHEMA_AFTER_REPAIR,
        )
        refresh_generated_contract_tests(str(tmp_path))
        if not (tmp_path / CONTRACT).exists():
            return  # retired outright — nothing left to contradict the model
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", CONTRACT],
            cwd=tmp_path, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(tmp_path)},
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr


# ==========================================================================
# 3. The production-fault veto, per file
# ==========================================================================

class TestTheVetoIsPerFile:
    def test_the_map_reports_every_evidenced_file_not_just_the_first(self):
        """Two evidenced files and one clean one. ``_production_fault_evidence``
        collapses this to a single pair by design; the map must not."""
        state = _state(DIAG_006, DIAG_007, DIAG_CONTRACT)
        got = g._production_fault_evidence_map(
            state, {T006, T007, CONTRACT},
        )
        assert set(got) == {T006, T007}, got
        assert all(got.values())
        assert CONTRACT not in got, (
            "a bare ValidationError on a constructor call is not something "
            "the application did"
        )
        # The single-file helper still answers the single-file question, and
        # that is exactly why a set-wide caller must not use it.
        first, _why = g._production_fault_evidence(
            state, {T006, T007, CONTRACT},
        )
        assert first == T006 and len(got) == 2

    def test_the_clean_file_survives_the_mixed_set(self):
        """The run-20 shape: evidenced files and one genuinely unsatisfiable
        file, same round. The escape must stay available for the last one."""
        state = _state(DIAG_006, DIAG_007, DIAG_CONTRACT)
        guarded = {T006, T007, CONTRACT}
        pf_map = g._production_fault_evidence_map(state, guarded)
        clean = {f for f in guarded if f not in pf_map}
        assert clean == {CONTRACT}
        # What the call site passes to the gate: "" while anything is clean.
        assert g._should_offer_unsatisfiable_escape(
            has_guarded_blocker=True,
            has_mandatable_target=True,
            distraction_streak=3,
            production_fault_evidence="" if clean else "x",
        ) is True

    def test_an_all_evidenced_set_still_vetoes_the_whole_round(self):
        """Run 16's hole must stay closed: with nothing clean, the veto is
        still the round's answer."""
        state = _state(DIAG_006)
        guarded = {T006}
        pf_map = g._production_fault_evidence_map(state, guarded)
        clean = {f for f in guarded if f not in pf_map}
        assert clean == set()
        assert g._should_offer_unsatisfiable_escape(
            has_guarded_blocker=True,
            has_mandatable_target=True,
            distraction_streak=5,
            production_fault_evidence=(
                "" if clean else next(iter(pf_map.values()))
            ),
        ) is False

    def test_the_banner_names_every_evidenced_file(self):
        pf_map = {T006: "the application crashed", "tests/b.py": "a 500"}
        text = g._production_fault_banner(
            T006, pf_map[T006], pf_map,
        )
        assert T006 in text and "tests/b.py" in text and "a 500" in text

    def test_the_banner_does_not_repeat_its_own_subject(self):
        text = g._production_fault_banner(T006, "crashed", {T006: "crashed"})
        assert text.count(T006) == 1

    def test_the_floor_forces_the_clean_file_and_spares_the_evidenced_one(
        self,
    ):
        """The declined-offer floor is what killed run 16 by forcing against
        evidence. Per file, it must still reach the floor on a clean sibling
        while never counting the evidenced one."""
        declined: dict[str, int] = {}
        by_file = {
            T006: "the application crashed handling the request",
            T007: "the application returned a generic 500",
        }
        kw = dict(
            any_real_patch=False, already_declared=False, floor=3,
            production_fault_by_file=by_file,
        )
        offer = {T006, T007, CONTRACT}
        assert g._apply_unsat_offer_floor(declined, offer, **kw) is None
        assert g._apply_unsat_offer_floor(declined, offer, **kw) is None
        forced = g._apply_unsat_offer_floor(declined, offer, **kw)
        assert forced is not None and forced[0] == CONTRACT
        for evidenced in (T006, T007):
            assert evidenced not in declined, (
                f"{evidenced} carries evidence and must never be counted"
            )

    def test_the_floor_never_fires_when_every_file_is_evidenced(self):
        declined: dict[str, int] = {}
        by_file = {T006: "the application crashed"}
        for _ in range(5):
            assert g._apply_unsat_offer_floor(
                declined, {T006},
                any_real_patch=False, already_declared=False, floor=3,
                production_fault_by_file=by_file,
            ) is None
        assert declined == {}


# ==========================================================================
# 4a. The route index
# ==========================================================================

def _routed_workspace(tmp_path):
    """A workspace shaped like lumina: the route module declares a bare
    ``/{record_id}``, and the ``/api/v1/birthdays`` prefix is contributed by
    an ``include_router`` call in a DIFFERENT file."""
    (tmp_path / "server/app/api").mkdir(parents=True)
    (tmp_path / "tests/acceptance").mkdir(parents=True)
    for pkg in ("server", "server/app", "server/app/api"):
        (tmp_path / pkg / "__init__.py").write_text("")
    (tmp_path / "server/app/api/birthdays.py").write_text(textwrap.dedent('''
        from fastapi import APIRouter

        router = APIRouter()

        @router.get("/{record_id}")
        def get_birthday(record_id: str):
            """No owner filter — run 20's defect 1."""
            return {}

        @router.get("")
        def list_birthdays():
            return []
    '''))
    (tmp_path / "server/app/api/router.py").write_text(textwrap.dedent('''
        from fastapi import APIRouter

        from server.app.api import birthdays

        api_router = APIRouter()
        api_router.include_router(
            birthdays.router, prefix="/birthdays", tags=["birthdays"],
        )
    '''))
    (tmp_path / "server/app/main.py").write_text(textwrap.dedent('''
        from fastapi import FastAPI

        from server.app.api.router import api_router


        def create_app() -> FastAPI:
            """No RequestValidationError handler — run 20's defect 2."""
            app = FastAPI()
            app.include_router(api_router, prefix="/api/v1")
            return app
    '''))
    # The generated acceptance conftest: declares its wiring as STRINGS.
    (tmp_path / "tests/acceptance/conftest.py").write_text(textwrap.dedent('''
        """AUTO-GENERATED acceptance conftest (ADR-0006)."""
        import pytest

        _APP_MODULE = 'server.app.main'
        _APP_SYMBOL = 'create_app'
        _CLOCK_BINDINGS = ['server.app.api.birthdays', 'server.app.api.deps']


        @pytest.fixture
        def client():
            raise NotImplementedError
    '''))
    # The acceptance test itself: imports nothing but pytest.
    (tmp_path / T006).write_text(textwrap.dedent('''
        import pytest

        # @verifies: STORY-006.AC-3
        def test_unauthorized_record_access(client):
            record_id = "x"
            resp = client.get(f"/api/v1/birthdays/{record_id}")
            assert resp.status_code == 404, resp.text
    '''))
    return tmp_path


class TestTheRouteIndex:
    def test_an_include_router_prefix_from_another_file_is_resolved(
        self, tmp_path,
    ):
        ws = str(_routed_workspace(tmp_path))
        index = ri.build_route_index(ws)
        paths = {t for t, _ in index}
        assert "/api/v1/birthdays/{record_id}" in paths, sorted(paths)
        assert "/api/v1/birthdays" in paths, sorted(paths)

    def test_an_observed_request_resolves_to_its_route_module(
        self, tmp_path,
    ):
        ws = str(_routed_workspace(tmp_path))
        hits = ri.route_modules_for_paths(
            ws, ["/api/v1/birthdays/930f2379-980e"],
        )
        assert hits == ["server/app/api/birthdays.py"]

    def test_an_f_string_placeholder_matches_like_a_path_param(self):
        found = ri.request_paths_in_text(
            'failing source: resp = client.get(f"/api/v1/birthdays/'
            '{record_id}")'
        )
        assert found == ["/api/v1/birthdays/{record_id}"]

    def test_a_file_path_is_not_mistaken_for_a_route(self):
        assert ri.request_paths_in_text('open("/tmp/teane-venv/bin/x.py")') == []

    def test_a_shorter_route_does_not_match_a_longer_request(self, tmp_path):
        ws = str(_routed_workspace(tmp_path))
        assert ri.route_modules_for_paths(ws, ["/api/v1/other/thing/deep"]) == []

    def test_the_conftest_chain_is_found_for_a_nested_test(self, tmp_path):
        ws = str(_routed_workspace(tmp_path))
        assert ri.conftest_chain_for(ws, [T006]) == [
            "tests/acceptance/conftest.py",
        ]

    def test_dotted_module_strings_resolve_to_workspace_files(
        self, tmp_path,
    ):
        ws = str(_routed_workspace(tmp_path))
        refs = ri.module_refs_in(ws, ["tests/acceptance/conftest.py"])
        assert "server/app/main.py" in refs
        assert "server/app/api/birthdays.py" in refs

    def test_a_third_party_dotted_string_is_not_claimed(self, tmp_path):
        ws = str(_routed_workspace(tmp_path))
        (tmp_path / "tests/acceptance/conftest.py").write_text(
            "_X = 'sqlalchemy.orm.session'\n_Y = 'server.app.main'\n"
        )
        refs = ri.module_refs_in(ws, ["tests/acceptance/conftest.py"])
        assert refs == ["server/app/main.py"]


# ==========================================================================
# 4b. RELATED FILES reaches a fixture-wired test
# ==========================================================================

class TestRelatedFilesReachesFixtureWiredTests:
    def test_the_import_walk_alone_sees_nothing(self, tmp_path):
        """Vacuity guard. If this ever returns a production file the widening
        below is no longer load-bearing and this suite should say so."""
        ws = str(_routed_workspace(tmp_path))
        assert g._workspace_imports_of(ws, [T006]) == []

    def test_both_real_defect_files_become_nameable(self, tmp_path):
        """The two files the judge could not name for eleven verdicts."""
        ws = str(_routed_workspace(tmp_path))
        imports, _ = g._related_files_evidence(
            ws, [T006], diagnostics=[DIAG_006],
        )
        assert "server/app/api/birthdays.py" in imports, imports
        assert "server/app/main.py" in imports, imports

    def test_the_route_module_is_ranked_ahead_of_fixture_wiring(
        self, tmp_path,
    ):
        """The slots are capped, so order decides what survives. The route
        the failure actually exercised is the most specific evidence."""
        ws = str(_routed_workspace(tmp_path))
        imports, _ = g._related_files_evidence(
            ws, [T006], diagnostics=[DIAG_006],
        )
        assert imports[0] == "server/app/api/birthdays.py", imports

    def test_the_cap_is_honoured(self, tmp_path):
        ws = str(_routed_workspace(tmp_path))
        imports, _ = g._related_files_evidence(
            ws, [T006], cap=1, diagnostics=[DIAG_006],
        )
        assert imports == ["server/app/api/birthdays.py"]

    def test_the_diagnostic_file_itself_is_never_listed_as_related(
        self, tmp_path,
    ):
        ws = str(_routed_workspace(tmp_path))
        imports, importers = g._related_files_evidence(
            ws, [T006], diagnostics=[DIAG_006],
        )
        assert T006 not in imports and T006 not in importers

    def test_no_diagnostics_still_returns_the_static_walk(self, tmp_path):
        """Backwards compatibility: the original behaviour for a test that
        does import its target."""
        ws = _routed_workspace(tmp_path)
        (ws / "server/tests").mkdir(parents=True, exist_ok=True)
        (ws / "server/tests/test_direct.py").write_text(
            "from server.app.main import create_app\n"
        )
        imports, _ = g._related_files_evidence(
            str(ws), ["server/tests/test_direct.py"],
        )
        assert imports == ["server/app/main.py"]

    def test_a_missing_workspace_degrades_quietly(self):
        imports, importers = g._related_files_evidence(
            "/nonexistent-workspace-xyz", [T006], diagnostics=[DIAG_006],
        )
        assert imports == [] and importers == []


class TestTheConfigKeyIsWired:
    def test_the_key_is_known_and_typed(self):
        from harness import cli
        assert (
            "max_related_evidence_files"
            in cli._KNOWN_NESTED_KEYS["node_throttle"]
        )
        assert (
            cli._TYPE_SCHEMA["node_throttle.max_related_evidence_files"]
            == (int,)
        )

    def test_the_default_matches_the_shipped_config(self):
        import json
        import pathlib
        cfg = json.loads(
            (
                pathlib.Path(__file__).resolve().parents[1]
                / "config" / "config.json"
            ).read_text()
        )
        assert cfg["node_throttle"]["max_related_evidence_files"] == 6
        assert g._related_files_cap() >= 1

    def test_it_is_documented(self):
        import pathlib
        help_md = (
            pathlib.Path(__file__).resolve().parents[1]
            / "config" / "config_help.md"
        ).read_text()
        assert "max_related_evidence_files" in help_md


class TestTheSweepNeverDeletesOnATransientError:
    """Regression tests for a defect in the sweep itself, found reviewing
    5adb3ac rather than in a run.

    The sweep runs at the top of EVERY build — including the builds where
    repair has just left a production file half-edited. ``parse_pydantic_models``
    swallows ``SyntaxError`` and returns ``[]``, which is indistinguishable
    from "no models here", so an unparseable source read as "nothing is
    derivable" and the generated test was deleted. Permanently: the emitters
    only run once per batch, so nothing would ever bring it back.
    """

    def test_a_syntax_error_in_the_source_leaves_the_test_alone(
        self, tmp_path,
    ):
        src_rel = _workspace_with_generated_contract(
            tmp_path, SCHEMA_AT_EMISSION,
        )
        (tmp_path / src_rel).write_text(
            SCHEMA_AT_EMISSION + "\n    def oops(  # unclosed\n"
        )
        refreshed, retired = refresh_generated_contract_tests(str(tmp_path))
        assert (refreshed, retired) == ([], [])
        assert (tmp_path / CONTRACT).exists()

    def test_an_empty_source_leaves_the_test_alone(self, tmp_path):
        """A half-written file reads as empty."""
        src_rel = _workspace_with_generated_contract(
            tmp_path, SCHEMA_AT_EMISSION,
        )
        (tmp_path / src_rel).write_text("")
        refreshed, retired = refresh_generated_contract_tests(str(tmp_path))
        assert (refreshed, retired) == ([], [])
        assert (tmp_path / CONTRACT).exists()

    def test_a_genuinely_deleted_source_still_retires(self, tmp_path):
        """The guard must not cost the case it was built for."""
        src_rel = _workspace_with_generated_contract(
            tmp_path, SCHEMA_AT_EMISSION,
        )
        (tmp_path / src_rel).unlink()
        _, retired = refresh_generated_contract_tests(str(tmp_path))
        assert retired == [CONTRACT]

    def test_a_source_that_parses_to_nothing_still_retires(self, tmp_path):
        """And nor the case where the models genuinely went away."""
        src_rel = _workspace_with_generated_contract(
            tmp_path, SCHEMA_AT_EMISSION,
        )
        (tmp_path / src_rel).write_text("X = 1\n")
        _, retired = refresh_generated_contract_tests(str(tmp_path))
        assert retired == [CONTRACT]
