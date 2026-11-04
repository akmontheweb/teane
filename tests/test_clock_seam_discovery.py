"""Discovery must find the clock seam the app actually has.

lumina run 14 (01a1a622, 2026-11-04) deferred **8 criteria across STORY-002
and STORY-004** — every ``not-runtime`` classification in the run — on this
rationale:

    'Requires pinning today to February 29 of a leap year, a specific
     calendar date that cannot be expressed relative to today; the app has
     no injectable clock, so it needs a clock injection point or freezegun'

The model was RIGHT about the HTTP surface it was asked to test against:
``server/app/api/birthdays.py`` computes ``today = today_in_timezone(tz)``
internally and reads ``feb29_policy`` from settings, so neither is a request
parameter. And it was obeying the prompt — ``_CLOCK_ABSENT_RULE`` says in so
many words that ``not-runtime`` "is the correct answer and it is always
available to you".

The defect was one line in ``discover_clock_override``::

    if "clock" not in text.lower():
        continue

Discovery looked only for a FastAPI dependency provider whose source mentions
"clock". lumina's seam is ``today_in_timezone`` — a plain module-level
function in ``services/timezone_service.py``, imported into
``api/birthdays.py`` — and it is exactly the ``setattr`` target
``_CLOCK_ABSENT_RULE`` itself blesses as "legitimate". A false negative in
discovery became a permanent loss of coverage, and a story whose criteria are
deferred cannot seal.

So this is not a prompt problem and not a model problem: the harness told the
generator the truth it had, and the truth it had was wrong.
"""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from harness import acceptance_gen as ag


def _app(tmp_path: pathlib.Path, files: dict[str, str]) -> str:
    for rel, src in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src))
    return str(tmp_path)


#: Run 14's shape, reduced: the helper in one module, the caller in another,
#: and the word "clock" nowhere in either file.
LUMINA = {
    "server/app/services/timezone_service.py": '''
        from datetime import datetime

        def today_in_timezone(timezone_name: str) -> datetime:
            """Today's date in the caller's timezone."""
            return datetime.now()
    ''',
    "server/app/api/birthdays.py": '''
        from datetime import timedelta

        from server.app.services.timezone_service import (
            today_in_timezone,
        )

        def upcoming(timezone_name: str, days: int):
            today = today_in_timezone(timezone_name)
            return today + timedelta(days=days)
    ''',
}


class TestItFindsRun14sSeam:
    def test_the_module_symbol_seam_is_discovered(self, tmp_path):
        out = ag.discover_clock_override(_app(tmp_path, LUMINA))
        assert out is not None, (
            "this is the run-14 regression: a perfectly patchable helper "
            "reported as 'no injectable clock'"
        )
        assert out["kind"] == "module_symbol"
        assert out["symbol"] == "today_in_timezone"

    def test_the_word_clock_appears_nowhere(self, tmp_path):
        """Pinning the actual cause: the old scan short-circuited on
        ``"clock" not in text.lower()``, so this fixture was invisible."""
        for src in LUMINA.values():
            assert "clock" not in textwrap.dedent(src).lower()

    def test_every_binding_module_is_recorded(self, tmp_path):
        """``from x import today_in_timezone`` binds a NEW name in the
        importer. Patching only the definer leaves the call site pointing at
        the real function — a freeze that silently does nothing, which is the
        lumina-17/21/22 failure."""
        out = ag.discover_clock_override(_app(tmp_path, LUMINA))
        assert out["bindings"] == [
            "server.app.api.birthdays",
            "server.app.services.timezone_service",
        ]

    def test_the_annotated_return_type_is_carried(self, tmp_path):
        """``today_in_timezone`` is annotated ``-> datetime`` despite its
        name. A stub handing back a ``date`` makes the app's own arithmetic
        raise, which reads as a production bug and is not one."""
        out = ag.discover_clock_override(_app(tmp_path, LUMINA))
        assert out["returns"] == "datetime"


class TestTheDependencySeamStillWins:
    DEP = {
        "app/deps.py": '''
            class Clock:
                def today(self): ...
                def now(self): ...

            def get_clock() -> Clock:
                return Clock()
        ''',
        "app/util.py": '''
            from datetime import date

            def today() -> date:
                return date.today()
        ''',
    }

    def test_a_declared_clock_dependency_is_preferred(self, tmp_path):
        """An app that declares one MEANS that seam; the module scan is the
        fallback, not a replacement."""
        out = ag.discover_clock_override(_app(tmp_path, self.DEP))
        assert out["kind"] == "dependency"
        assert out["symbol"] == "get_clock"

    def test_the_dependency_shape_is_unchanged(self, tmp_path):
        out = ag.discover_clock_override(_app(tmp_path, self.DEP))
        assert "date_methods" in out and "datetime_methods" in out


class TestWhatIsNotASeam:
    def test_an_app_with_no_clock_helper_still_returns_none(self, tmp_path):
        """``_CLOCK_ABSENT_RULE`` must stay reachable: when there genuinely
        is no seam, deferring IS the right answer."""
        assert ag.discover_clock_override(_app(tmp_path, {
            "app/main.py": '''
                def add(a, b):
                    return a + b
            ''',
        })) is None

    @pytest.mark.parametrize("name", [
        "today_in_timezone", "now", "utcnow", "utc_now", "current_date",
        "get_today", "local_now", "date_now", "_today",
    ])
    def test_the_vocabulary_covers_the_usual_spellings(self, name):
        assert ag._DATE_PROVIDER_NAME_RE.match(name)

    @pytest.mark.parametrize("name", [
        "today_or_tomorrow", "nowhere", "knowledge", "update_now_playing",
        "snowflake",
    ])
    def test_it_does_not_match_a_coincidental_name(self, name):
        assert not ag._DATE_PROVIDER_NAME_RE.match(name)

    def test_a_many_argument_helper_is_not_a_clock(self, tmp_path):
        """The stub ignores its arguments, so it can stand in for a zero- or
        one-arg helper. Three arguments means it computes something else."""
        assert ag.discover_clock_override(_app(tmp_path, {
            "app/util.py": '''
                def now(a, b, c):
                    return a
            ''',
        })) is None


class TestTheRenderedFreeze:
    def _render(self, tmp_path):
        clock = ag.discover_clock_override(_app(tmp_path, LUMINA))
        return ag._render_frozen_clock(clock)

    def test_it_renders_a_freeze_today(self, tmp_path):
        _mod, method = self._render(tmp_path)
        assert "def freeze_today(self, d):" in method

    def test_it_patches_every_binding(self, tmp_path):
        mod, _m = self._render(tmp_path)
        assert "server.app.api.birthdays" in mod
        assert "server.app.services.timezone_service" in mod

    def test_modules_are_resolved_at_call_time_not_import_time(self, tmp_path):
        """The client fixture purges app modules and re-imports; a module
        captured at conftest import is a different object from the one the
        rebuilt app uses, and setattr on it reaches nothing."""
        _mod, method = self._render(tmp_path)
        assert "_importlib.import_module(_name)" in method

    def test_a_freeze_that_patches_nothing_raises(self, tmp_path):
        """Silence here is the whole bug class: a test that reads as pinned
        while the app keeps real time."""
        _mod, method = self._render(tmp_path)
        assert "assert _patched" in method

    def test_the_datetime_return_type_is_honoured(self, tmp_path):
        mod, _m = self._render(tmp_path)
        assert "_CLOCK_RETURNS_DATETIME = True" in mod

    def test_a_date_helper_returns_a_bare_date(self, tmp_path):
        clock = ag.discover_clock_override(_app(tmp_path, {
            "app/util.py": '''
                from datetime import date

                def today() -> date:
                    return date.today()
            ''',
        }))
        mod, _m = ag._render_frozen_clock(clock)
        assert "_CLOCK_RETURNS_DATETIME = False" in mod

    def test_the_rendered_source_is_valid_python(self, tmp_path):
        """It is written into a conftest; a syntax error there takes the
        whole acceptance suite with it."""
        import ast

        mod, method = self._render(tmp_path)
        ast.parse(mod)
        ast.parse("class _C:\n" + method)

    def test_the_freeze_actually_pins_a_module_level_helper(self, tmp_path):
        """Executes the rendered code against real modules — the behaviour,
        not its text."""
        import datetime
        import sys

        root = _app(tmp_path, LUMINA)
        sys.path.insert(0, root)
        try:
            for m in [k for k in sys.modules if k.startswith("server")]:
                del sys.modules[m]
            mod_src, method_src = self._render(tmp_path)
            ns: dict = {}
            exec(mod_src + "class _Client:\n    pass\n", ns)  # noqa: S102
            exec(  # noqa: S102
                "class _Client2(_Client):" + method_src, ns,
            )
            import importlib

            api = importlib.import_module("server.app.api.birthdays")
            ns["_Client2"]().freeze_today(datetime.date(2024, 2, 29))
            assert api.today_in_timezone("UTC").date() == datetime.date(
                2024, 2, 29,
            ), "the call site still sees the real clock"
        finally:
            sys.path.remove(root)
            for m in [k for k in sys.modules if k.startswith("server")]:
                del sys.modules[m]


class TestTheGeneratorIsToldTheTruth:
    def test_a_discovered_seam_selects_the_available_rule(self):
        prompt = ag.build_system_prompt(db_isolated=True, clock_available=True)
        assert "freeze_today" in prompt
        assert "exposes no injectable clock" not in prompt

    def test_no_seam_keeps_the_absent_rule_reachable(self):
        prompt = ag.build_system_prompt(db_isolated=True, clock_available=False)
        assert "exposes no injectable clock" in prompt


class TestADisprovedClockClaim:
    """When discovery PROVED a seam and the model still blames the clock, the
    rationale is false rather than merely unsupported — a different defect
    with a different fix, so it is recorded differently."""

    CTX = None

    def _ctx(self):
        return ag.StoryAcceptanceContext(
            story_key="STORY-002", title="Leap day", description="",
            acceptance_criteria=[{
                "ac_key": "STORY-002.AC-1",
                "text": ("For a contact with a February 29 birthday, the "
                         "resolved date in a leap year is February 29."),
            }],
        )

    DATA = {"scenarios": [{
        "verifies": "STORY-002.AC-1",
        "classification": ag.CLASS_NOT_RUNTIME,
        "rationale": ("Requires pinning today to February 29 of a leap year; "
                      "the app has no injectable clock, so it needs a clock "
                      "injection point or freezegun"),
    }]}

    def test_it_is_marked_false_when_a_clock_exists(self):
        out = ag._not_runtime_acs(self.DATA, self._ctx(), clock_available=True)
        assert out["STORY-002.AC-1"].startswith("FALSE CLASSIFICATION")

    def test_it_is_only_suspect_when_no_clock_exists(self):
        """Without a seam the model was following instructions, so calling
        its answer false would be the harness blaming it for the harness's
        own blind spot."""
        out = ag._not_runtime_acs(self.DATA, self._ctx(), clock_available=False)
        assert out["STORY-002.AC-1"].startswith("SUSPECT CLASSIFICATION")

    def test_the_criterion_is_still_recorded_either_way(self):
        """A criterion with no scenario AND no record is strictly worse."""
        for avail in (True, False):
            out = ag._not_runtime_acs(
                self.DATA, self._ctx(), clock_available=avail,
            )
            assert "STORY-002.AC-1" in out

    @pytest.mark.parametrize("rationale", [
        "needs an injectable clock",
        "requires pinning today to a fixed date",
        "freezegun would be needed here",
        "no clock is available",
        "requires pinning the calendar date to Wednesday",
    ])
    def test_the_claim_is_recognised_in_its_usual_phrasings(self, rationale):
        assert ag._CLOCK_CLAIM_RE.search(rationale)

    def test_a_genuine_repo_rationale_is_not_treated_as_a_clock_claim(self):
        data = {"scenarios": [{
            "verifies": "STORY-002.AC-1",
            "classification": ag.CLASS_NOT_RUNTIME,
            "rationale": "This is about the CI pipeline, not the running app",
        }]}
        out = ag._not_runtime_acs(data, self._ctx(), clock_available=True)
        assert not out["STORY-002.AC-1"].startswith("FALSE CLASSIFICATION")
