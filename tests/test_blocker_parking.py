"""A test nobody can fix must cost one test, not the run (ADR-0010 Step 1).

Until this existed, a headless build that exhausted an auto-resume cap had one
outcome: ``hitl_abandon`` and the run ended.

* lumina run 12 ended that way with 13 of 17 criteria verified and ZERO
  stories sealed.
* lumina run 01a21db1 ended that way on three acceptance tests that were
  correct, with $9.93 of $12 unspent.

ADR-0006 already parks an acceptance failure that survives the acceptance
repair cap, but only when control returns to ``acceptance_node`` — and the
inner repair loop that dies on a HITL cap never returns there. These tests pin
the same containment at the level where runs actually died: the failing tests
are carried as recorded expected failures and the build goes on, or — whenever
that is not safe or not possible — the run terminates exactly as before.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import textwrap

import pytest

from harness import blocker_parking as bp
from harness import story_state

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

T001 = "tests/acceptance/test_story_001_acceptance.py"
T014 = "tests/acceptance/test_story_014_acceptance.py"
UNIT = "server/tests/test_birthday_service.py"

ACCEPTANCE_SRC = textwrap.dedent('''\
    """Generated acceptance suite."""
    import pytest


    # @verifies: STORY-001.AC-1
    def test_create_returns_201(client):
        assert client.post("/x").status_code == 201


    # @verifies: STORY-001.AC-2
    def test_rate_limit_101_requests_returns_429(client):
        assert client.get("/x").status_code == 429


    # @verifies: STORY-001.AC-3
    @pytest.mark.parametrize("n", [1, 2])
    def test_paged(client, n):
        assert n == 0
''')

UNIT_SRC = textwrap.dedent('''\
    # @tests: server/app/services/birthday_service.py


    def test_upcoming_window():
        assert 1 == 2


    def test_fine():
        assert True
''')


def _diag(nodeid: str, **kw) -> dict:
    d = {
        "file": nodeid.split("::", 1)[0], "line": 10, "severity": "error",
        "error_code": "AssertionError", "message": "AssertionError",
        "pytest_nodeid": nodeid,
    }
    d.update(kw)
    return d


@pytest.fixture
def ws(tmp_path) -> pathlib.Path:
    root = tmp_path / "lumina-park"
    (root / "tests" / "acceptance").mkdir(parents=True)
    (root / "server" / "tests").mkdir(parents=True)
    (root / T001).write_text(ACCEPTANCE_SRC)
    (root / T014).write_text(ACCEPTANCE_SRC.replace("STORY-001", "STORY-014"))
    (root / UNIT).write_text(UNIT_SRC)
    return root


def _state(ws: pathlib.Path, *diags: dict, raw: str = "", **extra) -> dict:
    st = {
        "workspace_path": str(ws),
        "session_id": "sess-park",
        "compiler_errors": list(diags),
        "exit_code": 1,
        "loop_counter": {},
        "node_state": {"last_build_output": raw},
        "modified_files": [],
        "current_batch_id": 7,
        "acceptance_config": {"max_repair_rounds_per_batch": 2},
    }
    st.update(extra)
    return st


RATE = f"{T001}::test_rate_limit_101_requests_returns_429"
CREATE = f"{T001}::test_create_returns_201"


# ---------------------------------------------------------------------------
# Finding the failing set
# ---------------------------------------------------------------------------


class TestFindingWhatFailed:
    def test_a_node_id_names_the_file_and_the_function(self, ws):
        tests, obstacles = bp.failing_tests([_diag(RATE)], "", str(ws))
        assert obstacles == []
        assert [(t.file, t.function) for t in tests] == [
            (T001, "test_rate_limit_101_requests_returns_429"),
        ]

    def test_the_raw_output_recovers_what_the_dedup_hid(self, ws):
        # PythonParser._dedup keys on (file, error_code), so three
        # AssertionErrors in one file reach compiler_errors as ONE diagnostic
        # with ONE node id. Parking from that alone would quarantine one
        # test, resume, and meet the second.
        raw = "\n".join([
            f"FAILED {RATE} - AssertionError",
            f"FAILED {CREATE} - assert 404 == 201",
            f"FAILED {T001}::test_paged[1] - assert 1 == 0",
            "===== 3 failed, 9 passed in 1.2s =====",
        ])
        tests, obstacles = bp.failing_tests([_diag(RATE)], raw, str(ws))
        assert obstacles == []
        assert {t.function for t in tests} == {
            "test_rate_limit_101_requests_returns_429",
            "test_create_returns_201",
            "test_paged",
        }

    def test_parametrisation_and_classes_reduce_to_the_function(self, ws):
        tests, _ = bp.failing_tests([
            _diag(f"{UNIT}::TestSvc::test_upcoming_window[utc-30]"),
            _diag(f"{UNIT}::TestSvc::test_upcoming_window[est-30]"),
        ], "", str(ws))
        assert [(t.file, t.function) for t in tests] == [
            (UNIT, "test_upcoming_window"),
        ]

    def test_an_absolute_path_is_made_workspace_relative(self, ws):
        tests, _ = bp.failing_tests(
            [_diag(f"{ws}/{RATE}")], "", str(ws),
        )
        assert tests[0].file == T001

    def test_a_module_that_cannot_collect_is_an_obstacle(self, ws):
        # "ERROR path.py - ImportError" has no ::function tail. An xfail on a
        # function is never reached when the module does not import.
        raw = f"ERROR {T001} - ImportError: cannot import name 'app'"
        tests, obstacles = bp.failing_tests([], raw, str(ws))
        assert tests == []
        assert obstacles and "fails to collect" in obstacles[0]

    def test_a_non_python_failure_is_an_obstacle(self, ws):
        # A vitest or tsc failure has no marker this module can write.
        _, obstacles = bp.failing_tests([
            _diag(RATE),
            {"file": "client/src/App.test.tsx", "line": 4,
             "message": "expected 1 to be 2"},
        ], "", str(ws))
        assert obstacles == [
            "client/src/App.test.tsx is not a Python test failure",
        ]

    def test_a_loose_python_frame_is_tolerated(self, ws):
        # A failing test whose innermost frame is production code yields a
        # second diagnostic with no node id. The resumed compile, not this
        # function, tells that from an independent build error.
        tests, obstacles = bp.failing_tests([
            _diag(RATE),
            {"file": "server/app/main.py", "line": 30, "message": "ValueError"},
        ], "", str(ws))
        assert obstacles == [] and len(tests) == 1


# ---------------------------------------------------------------------------
# Deciding
# ---------------------------------------------------------------------------


class TestWhenParkingIsRefused:
    def _decide(self, state, **kw):
        base = dict(max_tests=12, max_blockers=12)
        base.update(kw)
        return bp.decide(state, **base)

    def test_a_bounded_failing_set_is_parkable(self, ws):
        d = self._decide(_state(ws, _diag(RATE)))
        assert d.parkable and len(d.tests) == 1

    def test_no_failing_test_means_nothing_to_park(self, ws):
        d = self._decide(_state(ws))
        assert not d.parkable and "no failing pytest function" in d.refusal

    def test_a_large_failing_set_is_a_broken_build_not_a_blocker(self, ws):
        # Sealing an application behind a wall of quarantined tests is worse
        # than stopping.
        diags = [_diag(f"{UNIT}::test_{i}") for i in range(13)]
        d = self._decide(_state(ws, *diags))
        assert not d.parkable
        assert "exceed max_parked_tests_per_blocker (12)" in d.refusal

    def test_the_session_bound_on_parks_holds(self, ws):
        st = _state(ws, _diag(RATE))
        st["loop_counter"][bp.LC_PARKED_BLOCKERS] = [{}] * 3
        d = self._decide(st, max_blockers=3)
        assert not d.parkable and "3/3 blockers already parked" in d.refusal

    def test_zero_disables_parking(self, ws):
        d = self._decide(_state(ws, _diag(RATE)), max_blockers=0)
        assert not d.parkable and "disabled" in d.refusal

    def test_a_park_that_did_not_work_is_not_repeated(self, ws):
        # THE verification rule. Nothing proves in advance that a park will
        # turn the build green; the resumed compile says so. If the same
        # tests are failing again they already carry a marker, marking them
        # again cannot help, and the run must end rather than loop.
        st = _state(ws, _diag(RATE))
        st["loop_counter"][bp.LC_PARKED_NODEIDS] = [
            f"{T001}::test_rate_limit_101_requests_returns_429",
        ]
        d = self._decide(st)
        assert not d.parkable
        assert "already parked and the build is still red" in d.refusal

    def test_only_the_new_failures_are_parked(self, ws):
        st = _state(ws, _diag(RATE), _diag(CREATE))
        st["loop_counter"][bp.LC_PARKED_NODEIDS] = [
            f"{T001}::test_rate_limit_101_requests_returns_429",
        ]
        d = self._decide(st)
        assert d.parkable
        assert [t.function for t in d.tests] == ["test_create_returns_201"]

    def test_an_obstacle_refuses_the_whole_park(self, ws):
        raw = f"ERROR {T014} - ImportError: boom"
        d = self._decide(_state(ws, _diag(RATE), raw=raw))
        assert not d.parkable and "fails to collect" in d.refusal


# ---------------------------------------------------------------------------
# Writing the markers
# ---------------------------------------------------------------------------


class TestTheQuarantineItWrites:
    def test_the_marker_is_the_existing_self_describing_one(self, ws):
        tests, _ = bp.failing_tests([_diag(RATE)], "", str(ws))
        out = bp.apply(str(ws), tests, "parked: no node could fix it")
        src = (ws / T001).read_text()
        assert out.files == [T001] and len(out.parked) == 1
        assert src.count("teane-quarantine") == 1
        assert "strict=False" in src
        # Only the failing function is marked.
        marker_line = next(
            i for i, ln in enumerate(src.splitlines())
            if "teane-quarantine" in ln
        )
        assert "def test_rate_limit_101" in src.splitlines()[marker_line + 1]

    def test_the_criterion_the_test_verifies_is_reported(self, ws):
        tests, _ = bp.failing_tests([_diag(RATE)], "", str(ws))
        out = bp.apply(str(ws), tests, "r")
        assert out.ac_keys == ["STORY-001.AC-2"]

    def test_a_unit_test_has_no_criterion(self, ws):
        tests, _ = bp.failing_tests(
            [_diag(f"{UNIT}::test_upcoming_window")], "", str(ws),
        )
        out = bp.apply(str(ws), tests, "r")
        assert out.ac_keys == [] and out.files == [UNIT]

    def test_the_verifies_marker_still_resolves_afterwards(self, ws):
        # acceptance_node finds a criterion's test through the comment that
        # sits DIRECTLY above the function. A marker inserted in the wrong
        # place would orphan the criterion from its own test.
        from harness import acceptance_gen as ag

        tests, _ = bp.failing_tests(
            [_diag(RATE), _diag(f"{T001}::test_paged[1]")], "", str(ws),
        )
        bp.apply(str(ws), tests, "r")
        mapping = ag.verifies_by_function((ws / T001).read_text())
        assert mapping["test_rate_limit_101_requests_returns_429"][0] == (
            "STORY-001.AC-2"
        )
        assert mapping["test_paged"][0] == "STORY-001.AC-3"
        assert mapping["test_create_returns_201"][0] == "STORY-001.AC-1"

    def test_a_missing_file_is_skipped_not_fatal(self, ws):
        out = bp.apply(str(ws), [
            bp.FailingTest("gone.py::test_x", "gone.py", "test_x"),
            *bp.failing_tests([_diag(RATE)], "", str(ws))[0],
        ], "r")
        assert out.files == [T001]

    def test_a_parked_test_no_longer_fails_the_build(self, ws):
        # The property everything rests on, checked against real pytest:
        # before the park the suite exits non-zero, after it exits zero, and
        # the test that still passes is still collected and still passes.
        def _run() -> subprocess.CompletedProcess:
            return subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                 "-p", "no:randomly", UNIT],
                cwd=ws, capture_output=True, text=True,
            )

        before = _run()
        assert before.returncode != 0, before.stdout
        tests, _ = bp.failing_tests(
            [_diag(f"{UNIT}::test_upcoming_window")], before.stdout, str(ws),
        )
        bp.apply(str(ws), tests, "r")
        after = _run()
        assert after.returncode == 0, after.stdout + after.stderr
        assert "1 passed" in after.stdout and "1 xfailed" in after.stdout


# ---------------------------------------------------------------------------
# The whole park
# ---------------------------------------------------------------------------


class TestParkingABlocker:
    def _park(self, state, **cfg):
        return bp.park_blocker(
            state, trigger="unsatisfiable_test:" + T001,
            cap_reason="per-trigger cap 6/6 for 'unsatisfiable_test'",
            hitl_config=cfg,
        )

    def test_the_run_01a21db1_blocker_is_parked(self, ws):
        # The shape that ended the run: failing acceptance tests across two
        # files, a class cap spent.
        st = _state(ws, _diag(RATE), _diag(f"{T014}::test_create_returns_201"))
        out = self._park(st)
        assert out is not None and len(out.parked) == 2
        assert "teane-quarantine" in (ws / T001).read_text()
        assert "teane-quarantine" in (ws / T014).read_text()

    def test_the_park_is_recorded_in_loop_counter(self, ws):
        st = _state(ws, _diag(RATE))
        self._park(st)
        lc = st["loop_counter"]
        assert lc[bp.LC_PARKED_NODEIDS] == [
            f"{T001}::test_rate_limit_101_requests_returns_429",
        ]
        (entry,) = lc[bp.LC_PARKED_BLOCKERS]
        assert entry["ac_keys"] == ["STORY-001.AC-2"]
        assert entry["trigger"].startswith("unsatisfiable_test")
        # Checkpoints are JSON; the record must survive one.
        assert json.loads(json.dumps(lc)) == lc

    def test_the_touched_files_join_modified_files(self, ws):
        st = _state(ws, _diag(RATE))
        self._park(st)
        assert st["modified_files"] == [T001]

    def test_no_auto_resume_counter_is_touched(self, ws):
        # The class budget is refunded by the existing progress rule when
        # the parked blocker is gone. If the park did NOT work these caps
        # must still be spent, so the next trip ends the run instead of
        # buying another six resumes.
        st = _state(ws, _diag(RATE))
        st["loop_counter"].update({
            "hitl_auto_resumes_taken": 18,
            "hitl_auto_resumes_per_trigger": {"unsatisfiable_test": 6},
            "hitl_auto_resumes_per_blocker": {"sig": 9},
        })
        self._park(st)
        lc = st["loop_counter"]
        assert lc["hitl_auto_resumes_taken"] == 18
        assert lc["hitl_auto_resumes_per_trigger"] == {"unsatisfiable_test": 6}
        assert lc["hitl_auto_resumes_per_blocker"] == {"sig": 9}

    def test_a_second_trip_on_the_same_blocker_terminates(self, ws):
        st = _state(ws, _diag(RATE))
        assert self._park(st) is not None
        # The compile came back red on the same test.
        assert self._park(st) is None

    def test_an_acceptance_park_spends_the_batch_repair_rounds(self, ws):
        # acceptance_node regenerates unverified scenarios on every entry —
        # dropping the marker — and routes an attributable failure back to
        # repair while rounds remain: a second inner loop on the test just
        # parked. With the rounds spent, re-entry takes the existing
        # _park_over_cap path and seals.
        st = _state(ws, _diag(RATE))
        self._park(st)
        assert st["loop_counter"]["acceptance_repair"] == 2
        assert st["loop_counter"]["acceptance_repair_batch"] == 7

    def test_a_unit_park_leaves_the_acceptance_rounds_alone(self, ws):
        st = _state(ws, _diag(f"{UNIT}::test_upcoming_window"))
        self._park(st)
        assert "acceptance_repair" not in st["loop_counter"]

    def test_the_unauthorable_flags_are_cleared(self, ws):
        # Left set, they would condemn the NEXT blocker on its first trip.
        st = _state(ws, _diag(RATE))
        st["node_state"].update({
            "test_regen_unsupported": T001, "test_regen_stuck": True,
        })
        self._park(st)
        assert "test_regen_unsupported" not in st["node_state"]
        assert "test_regen_stuck" not in st["node_state"]

    def test_a_refused_park_changes_nothing(self, ws):
        st = _state(ws)  # nothing failing
        before = (ws / T001).read_text()
        assert self._park(st) is None
        assert st["loop_counter"] == {} and st["modified_files"] == []
        assert (ws / T001).read_text() == before

    @pytest.mark.parametrize("trigger", [
        "budget_exhausted", "build_env_timeout", "security_fix_limit:8/2",
        "traceability_block", "decomposition_missing", "env_misconfig:uv",
    ])
    def test_a_blocker_that_is_not_a_failing_test_is_not_parked(
        self, ws, trigger,
    ):
        # compiler_errors can still hold stale test failures when the trip
        # is about a sandbox timeout or a security finding. Quarantining
        # them would not unblock anything.
        st = _state(ws, _diag(RATE))
        assert bp.park_blocker(
            st, trigger=trigger, cap_reason="session cap 18/18",
        ) is None
        assert "teane-quarantine" not in (ws / T001).read_text()

    def test_the_flag_restores_termination(self, ws):
        st = _state(ws, _diag(RATE))
        assert self._park(st, park_blocker_on_cap=False) is None

    def test_a_state_override_outranks_config(self, ws):
        st = _state(ws, _diag(RATE), hitl_park_blocker_on_cap=False)
        assert self._park(st, park_blocker_on_cap=True) is None

    def test_the_per_park_bound_is_config_driven(self, ws):
        st = _state(ws, _diag(RATE), _diag(CREATE))
        assert self._park(st, max_parked_tests_per_blocker=1) is None
        assert self._park(st, max_parked_tests_per_blocker=2) is not None


class TestThePermanentRecord:
    def _seed(self, ws) -> tuple[str, int]:
        app = story_state.app_name_for_workspace(str(ws))
        conn = story_state.open_story_db(workspace_path=str(ws))
        try:
            story_state.ensure_feature(conn, app, "test", name="Test feature")
            (key,) = story_state.create_stories(conn, app, [{
                "title": "Create a celebration", "feature": "test",
                "acceptance_criteria": ["creates", "rate limits", "pages"],
            }])
            ac = story_state.get_ac_by_key(conn, app, f"{key}.AC-2")
            return key, int(ac["id"])
        finally:
            conn.close()

    def test_the_criterion_is_recorded_unverified(self, ws):
        key, ac_id = self._seed(ws)
        assert key == "STORY-001"
        st = _state(ws, _diag(RATE))
        bp.park_blocker(st, trigger="zero_patch_loop:3", cap_reason="cap 6/6")
        app = story_state.app_name_for_workspace(str(ws))
        conn = story_state.open_story_db(workspace_path=str(ws))
        try:
            (row,) = story_state.list_acceptance_deferrals(conn, app)
            assert row["ac_id"] == ac_id
            assert row["status"] == "attributable"
            assert "parked at the HITL cap" in row["detail"]
        finally:
            conn.close()

    def test_every_parked_test_gets_a_defect_row(self, ws):
        self._seed(ws)
        st = _state(
            ws, _diag(RATE), _diag(f"{UNIT}::test_upcoming_window"),
        )
        bp.park_blocker(st, trigger="zero_patch_loop:3", cap_reason="cap 6/6")
        app = story_state.app_name_for_workspace(str(ws))
        conn = story_state.open_story_db(workspace_path=str(ws))
        try:
            rows = conn.execute(
                "SELECT d.severity, d.summary, d.status, s.story_key "
                "FROM defects d LEFT JOIN stories s ON s.id = d.story_id "
                "WHERE d.workspace = ? ORDER BY d.summary", (app,),
            ).fetchall()
        finally:
            conn.close()
        assert [r[0] for r in rows] == [bp.DEFECT_SEVERITY] * 2
        assert all(r[2] == "open" for r in rows)
        # The unit test belongs to no story; the acceptance test to its own.
        assert sorted(str(r[3]) for r in rows) == ["None", "STORY-001"]

    def test_a_missing_story_db_row_does_not_undo_the_park(self, ws):
        # No story seeded at all: the marker is what stops the run dying,
        # and it must be written even when there is nothing to attach the
        # record to.
        st = _state(ws, _diag(RATE))
        assert bp.park_blocker(
            st, trigger="zero_patch_loop:3", cap_reason="cap",
        ) is not None
        assert "teane-quarantine" in (ws / T001).read_text()


# ---------------------------------------------------------------------------
# Config and wiring
# ---------------------------------------------------------------------------


class TestTheBoundsAreConfigDriven:
    KEYS = (
        "park_blocker_on_cap", "max_parked_blockers",
        "max_parked_tests_per_blocker",
    )

    @pytest.mark.parametrize(("raw", "want"), [
        (-3, 0), (0, 0), (12, 12), (999, 200), ("x", 12), (None, 12),
    ])
    def test_max_parked_blockers_is_clamped(self, raw, want):
        assert bp.clamp_max_parked_blockers(raw) == want

    @pytest.mark.parametrize(("raw", "want"), [
        (-3, 1), (0, 1), (12, 12), (999, 200), ("x", 12), (None, 12),
    ])
    def test_max_tests_per_park_is_clamped(self, raw, want):
        assert bp.clamp_max_tests_per_park(raw) == want

    @pytest.mark.parametrize("key", KEYS)
    def test_config_json_carries_the_key(self, key):
        cfg = json.loads((REPO_ROOT / "config" / "config.json").read_text())
        assert key in cfg["hitl"]

    @pytest.mark.parametrize("key", KEYS)
    def test_cli_accepts_and_type_checks_the_key(self, key):
        from harness import cli

        assert key in cli._KNOWN_NESTED_KEYS["hitl"]
        assert f"hitl.{key}" in cli._TYPE_SCHEMA

    @pytest.mark.parametrize("key", KEYS)
    def test_config_help_documents_the_key(self, key):
        assert key in (REPO_ROOT / "config" / "config_help.md").read_text()


class TestTheHitlLoopParksBeforeItTerminates:
    SRC = (REPO_ROOT / "harness" / "cli.py").read_text()

    def test_both_terminations_try_to_park_first(self):
        # The cap-exhausted branch and the un-authorable-test branch are the
        # two places a headless run gives up on a test it cannot fix.
        assert self.SRC.count("_park_blocker_at_cap(") == 3  # def + 2 calls

    def test_the_park_precedes_the_cap_termination(self):
        park = self.SRC.index("_parked = _park_blocker_at_cap(")
        banner = self.SRC.index("TERMINATED — HITL auto-resume cap exhausted")
        assert park < banner
        # ...and the termination is the ELSE of a successful park, not a
        # sibling that also runs.
        assert "elif _cap_hit:" in self.SRC[park:banner]

    def test_a_fault_in_parking_degrades_to_termination(self, monkeypatch):
        # Parking improves on termination; it is never a precondition for
        # it. A fault must not raise into the HITL loop.
        from harness import cli

        def _boom(*a, **k):
            raise RuntimeError("disk full")

        monkeypatch.setattr(bp, "park_blocker", _boom)
        assert cli._park_blocker_at_cap(
            {}, trigger="zero_patch_loop:3", cap_reason="cap",
        ) is None


# ---------------------------------------------------------------------------
# Through the real HITL loop
# ---------------------------------------------------------------------------


def _force_headless(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness import cli

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("HARNESS_AUTO_APPROVE", raising=False)
    monkeypatch.setattr(cli, "_HITL_FLAGS", {"repair": False})


def _capped_state(ws: pathlib.Path, *diags: dict, **node_flags) -> dict:
    """A state whose SESSION cap is already spent — the run-ending shape."""
    st = _state(ws, *diags)
    st.update({
        "budget_remaining_usd": 9.93,
        "budget_initial_usd": 12.0,
        "hitl_auto_resume_cap": 3,
    })
    st["loop_counter"].update({
        "hitl_auto_resumes_taken": 3, "total_repairs": 5,
    })
    st["node_state"].update({
        "hitl_trigger": "unsatisfiable_test:" + T001,
        "hitl_active": True, "hitl_awaiting_input": True,
    })
    st["node_state"].update(node_flags)
    return st


class TestASpentCapNoLongerEndsTheRun:
    def test_a_failing_test_is_parked_and_the_graph_resumes(
        self, ws, monkeypatch, capsys,
    ):
        from harness import cli

        _force_headless(monkeypatch)
        result = cli.hitl_menu_loop(_capped_state(ws, _diag(RATE)))
        ns = result["node_state"]
        # Resumed, not abandoned: route_after_hitl sends this to the
        # compiler instead of END.
        assert not ns.get("hitl_abandon")
        assert not ns.get("hitl_auto_resume_cap_hit")
        assert ns.get("hitl_resolved") is True
        assert ns.get("hitl_active") is False
        assert "teane-quarantine" in (ws / T001).read_text()
        assert "TERMINATED" not in capsys.readouterr().err
        # The park is not an auto-resume and is not charged as one.
        assert result["loop_counter"]["hitl_auto_resumes_taken"] == 3
        assert len(result["loop_counter"][bp.LC_PARKED_BLOCKERS]) == 1

    def test_an_unparkable_blocker_terminates_exactly_as_before(
        self, ws, monkeypatch, capsys,
    ):
        from harness import cli

        _force_headless(monkeypatch)
        raw = f"ERROR {T001} - ImportError: cannot import name 'app'"
        st = _capped_state(ws)
        st["node_state"]["last_build_output"] = raw
        before = (ws / T001).read_text()
        result = cli.hitl_menu_loop(st)
        ns = result["node_state"]
        assert ns.get("hitl_abandon") is True
        assert ns.get("hitl_auto_resume_cap_hit") is True
        assert "TERMINATED — HITL auto-resume cap exhausted" in (
            capsys.readouterr().err
        )
        assert (ws / T001).read_text() == before

    def test_the_same_blocker_twice_terminates_on_the_second_trip(
        self, ws, monkeypatch,
    ):
        # The whole safety argument in one test: an ineffective park costs
        # exactly one cycle.
        from harness import cli

        _force_headless(monkeypatch)
        first = cli.hitl_menu_loop(_capped_state(ws, _diag(RATE)))
        assert not first["node_state"].get("hitl_abandon")
        # The compile came back red on the same test; HITL trips again.
        first["compiler_errors"] = [_diag(RATE)]
        first["node_state"].update({
            "hitl_trigger": "unsatisfiable_test:" + T001,
            "hitl_active": True, "hitl_awaiting_input": True,
            "hitl_resolved": False,
        })
        second = cli.hitl_menu_loop(first)
        assert second["node_state"].get("hitl_abandon") is True

    def test_parking_off_terminates(self, ws, monkeypatch):
        from harness import cli

        _force_headless(monkeypatch)
        st = _capped_state(ws, _diag(RATE))
        st["harness_config"] = {"hitl": {"park_blocker_on_cap": False}}
        result = cli.hitl_menu_loop(st)
        assert result["node_state"].get("hitl_abandon") is True
        assert "teane-quarantine" not in (ws / T001).read_text()

    def test_a_test_no_node_may_author_is_parked_not_fatal(
        self, ws, monkeypatch,
    ):
        # lumina 01a164b2 ended on this branch: repair cannot edit the test,
        # regeneration declined it, so the run terminated. Nothing in the
        # graph can fix it — which is exactly what parking is for.
        from harness import cli

        _force_headless(monkeypatch)
        st = _capped_state(ws, _diag(RATE), test_regen_unsupported=T001)
        st["loop_counter"]["hitl_auto_resumes_taken"] = 0  # no cap spent
        result = cli.hitl_menu_loop(st)
        ns = result["node_state"]
        assert not ns.get("hitl_abandon")
        assert "test_regen_unsupported" not in ns
        assert not ns.get("unauthorable_test_blocker")
        assert "teane-quarantine" in (ws / T001).read_text()
        # Resumed BY the park on this trip — not by falling through to an
        # ordinary auto-resume on a later pass of the menu loop, which
        # would be charged as one.
        assert ns.get("hitl_resolved") is True
        assert result["loop_counter"]["hitl_auto_resumes_taken"] == 0
