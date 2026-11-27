"""A generated test must be able to RUN against the code it names.

Every other test-quality gate compares a generated test against the SPEC.
``find_contradictions_across`` proves one test contradicts another; the
spec-consistency reviewer asks which spec rule fixes a computed expectation.
Neither asks whether the call matches the signature of the model it calls, and
lumina 01a21929 (run 20) shipped a test that could not execute with both of
them green:

* the emitted happy path was ``BirthdayResponse(id='aaa', first_name='aaa',
  birth_month=1, birth_day=1, category='aaa', created_at='aaa',
  updated_at='aaa')`` against a model that also declared ``last_name``,
  ``birth_year``, ``phone_number``, ``email`` and ``notes`` as required;
* pydantic answered "5 validation errors ... Field required" on every build;
* the prover saw no raise/success pair, so it was out of scope;
* the spec reviewer had the test AND the model in front of it — it reads the
  ``# @tests:`` marker to fetch the module — and logged "no false computed
  expectations in 10 generated test file(s)" at 00:17:03, because the failure
  is not a computed value at odds with a spec rule, and because the reviewer
  is forbidden from flagging a test "merely because today's production code
  does not satisfy it yet", which is exactly the shape "this fails because
  last_name is required" presents.

:class:`TestItNeverFlagsATestThatOutrunsTheCode` and
:class:`TestEveryUnprovableShapeIsSkipped` are the important half. This gate
re-prompts the author to rewrite a test, so a false positive here edits a
correct test — the silent failure this project treats as worse than the bug.
Anything it cannot prove from the model's own declaration is skipped.
"""

from __future__ import annotations

import textwrap

import pytest

from harness.contract_tests import find_unexecutable_model_calls

#: ``server/app/schemas/birthday.py`` as the patcher created it at 00:14:32 —
#: every nullable field ``X | None`` with NO default, so all twelve are
#: required under Pydantic v2.
SCHEMA = textwrap.dedent('''
    from pydantic import BaseModel, ConfigDict


    class BirthdayResponse(BaseModel):
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

SCHEMA_REL = "server/app/schemas/birthday.py"
TEST_REL = "server/tests/contract/test_birthday_contract.py"

#: Verbatim from the file emitted at 00:16:18.
RUN20_TEST = textwrap.dedent('''
    # @tests: server/app/schemas/birthday.py
    import pytest
    from pydantic import ValidationError
    from server.app.schemas.birthday import BirthdayResponse


    def test_birthday_response_valid_construction():
        obj = BirthdayResponse(id='aaa', first_name='aaa', birth_month=1, birth_day=1, category='aaa', created_at='aaa', updated_at='aaa')
        assert obj is not None

    def test_birthday_response_requires_id():
        with pytest.raises(ValidationError):
            BirthdayResponse(first_name='aaa', birth_month=1, birth_day=1, category='aaa', created_at='aaa', updated_at='aaa')
''')


@pytest.fixture
def ws(tmp_path):
    """A workspace holding the schema, ready for a test file to be added."""
    (tmp_path / "server/app/schemas").mkdir(parents=True)
    (tmp_path / "server/tests/contract").mkdir(parents=True)
    (tmp_path / SCHEMA_REL).write_text(SCHEMA)

    def _run(test_src: str, rel: str = TEST_REL):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(textwrap.dedent(test_src))
        return find_unexecutable_model_calls(
            str(tmp_path), {rel: textwrap.dedent(test_src)},
        )

    _run.path = tmp_path  # type: ignore[attr-defined]
    return _run


class TestTheRun20Test:
    def test_the_omitted_required_fields_are_all_named(self, ws):
        [f] = ws(RUN20_TEST)
        assert f.model == "BirthdayResponse"
        assert f.test == "test_birthday_response_valid_construction"
        assert set(f.missing) == {
            "last_name", "birth_year", "phone_number", "email", "notes",
        }

    def test_the_finding_says_what_will_happen_and_why(self, ws):
        [f] = ws(RUN20_TEST)
        text = f.describe()
        assert "Field required" in text
        assert "no default" in text
        assert "No production change" in text
        assert TEST_REL in text and "last_name" in text

    def test_the_raises_sibling_is_not_flagged(self, ws):
        """``test_birthday_response_requires_id`` omits ``id`` ON PURPOSE —
        that is a required-omission test and the emitter writes one per
        required field. Flagging those would condemn the whole tier."""
        found = ws(RUN20_TEST)
        assert [f.test for f in found] == [
            "test_birthday_response_valid_construction",
        ]

    def test_a_complete_construction_is_clean(self, ws):
        """What the fixed generator now emits."""
        assert ws('''
            # @tests: server/app/schemas/birthday.py
            from server.app.schemas.birthday import BirthdayResponse

            def test_valid():
                obj = BirthdayResponse(
                    id='aaa', first_name='aaa', last_name='aaa',
                    birth_month=1, birth_day=1, birth_year=1,
                    category='aaa', phone_number='aaa', email='aaa',
                    notes='aaa', created_at='aaa', updated_at='aaa',
                )
                assert obj is not None
        ''') == []


class TestItNeverFlagsATestThatOutrunsTheCode:
    def test_a_model_that_does_not_exist_yet_is_not_flagged(self, ws):
        """The module is there and the class is not. The test is RIGHT and the
        build is supposed to be red until production adds it."""
        assert ws('''
            from server.app.schemas.birthday import NotYetWritten

            def test_future():
                obj = NotYetWritten(id='a')
                assert obj
        ''') == []

    def test_a_module_that_does_not_exist_yet_is_not_flagged(self, ws):
        assert ws('''
            from server.app.services.not_built import Thing

            def test_future():
                obj = Thing()
                assert obj
        ''') == []

    def test_a_missing_method_is_not_flagged(self, ws):
        """Only constructions are in scope. A method the model does not have
        yet is production lagging the test, not an unexecutable call."""
        assert ws('''
            from server.app.schemas.birthday import BirthdayResponse

            def test_future(obj):
                assert BirthdayResponse.not_a_method() is None
        ''') == []

    def test_a_same_named_class_from_elsewhere_is_not_claimed(self, ws):
        """Resolution is by import. A third-party ``BirthdayResponse`` is not
        the workspace's, and guessing would flag a call we know nothing
        about."""
        assert ws('''
            from some_vendor_sdk import BirthdayResponse

            def test_vendor():
                obj = BirthdayResponse(id='aaa')
                assert obj
        ''') == []


class TestEveryUnprovableShapeIsSkipped:
    def _schema(self, ws, body: str):
        (ws.path / SCHEMA_REL).write_text(textwrap.dedent(body))

    def test_a_model_validator_can_inject_fields(self, ws):
        """``mode="before"`` can fill a field in, so absence proves nothing."""
        self._schema(ws, '''
            from pydantic import BaseModel, model_validator


            class M(BaseModel):
                a: str
                b: str

                @model_validator(mode="before")
                @classmethod
                def _fill(cls, data):
                    data.setdefault("b", "x")
                    return data
        ''')
        assert ws('''
            from server.app.schemas.birthday import M

            def test_x():
                obj = M(a='1')
                assert obj
        ''') == []

    def test_an_aliased_field_changes_the_accepted_keyword(self, ws):
        self._schema(ws, '''
            from pydantic import BaseModel, Field


            class M(BaseModel):
                a: str
                b: str = Field(alias="bAlias")
        ''')
        assert ws('''
            from server.app.schemas.birthday import M

            def test_x():
                obj = M(a='1', bAlias='2')
                assert obj
        ''') == []

    def test_double_star_kwargs_are_not_statically_known(self, ws):
        assert ws('''
            from server.app.schemas.birthday import BirthdayResponse

            def test_x(payload):
                obj = BirthdayResponse(**payload)
                assert obj
        ''') == []

    def test_a_positional_argument_is_skipped(self, ws):
        """A model could define its own ``__init__``; we only know the
        declarative one."""
        assert ws('''
            from server.app.schemas.birthday import BirthdayResponse

            def test_x():
                obj = BirthdayResponse({'id': 'a'})
                assert obj
        ''') == []

    def test_a_construction_inside_try_is_skipped(self, ws):
        assert ws('''
            from server.app.schemas.birthday import BirthdayResponse

            def test_x():
                try:
                    BirthdayResponse(id='a')
                except Exception:
                    pass
        ''') == []

    def test_a_construction_inside_a_functional_raises_is_skipped(self, ws):
        """Anything lexically inside a ``raises(...)`` call is left alone.

        This particular shape is itself broken — the construction evaluates
        before ``raises`` is even called — but a rejection-in-progress is the
        author's intent and second-guessing it is how a gate starts editing
        tests it does not understand. Conservative on purpose.
        """
        assert ws('''
            import pytest
            from server.app.schemas.birthday import BirthdayResponse

            def test_x():
                pytest.raises(ValueError, BirthdayResponse(id='a'))
        ''') == []

    def test_a_relative_import_is_not_resolved(self, ws):
        assert ws('''
            from .birthday import BirthdayResponse

            def test_x():
                obj = BirthdayResponse(id='a')
                assert obj
        ''') == []

    def test_a_model_with_no_required_field_is_never_flagged(self, ws):
        self._schema(ws, '''
            from pydantic import BaseModel


            class M(BaseModel):
                a: str = "x"
                b: int | None = None
        ''')
        assert ws('''
            from server.app.schemas.birthday import M

            def test_x():
                obj = M()
                assert obj
        ''') == []

    def test_an_unparseable_test_is_left_to_the_other_prover(self, ws):
        assert ws("def test_x(:\n    pass\n") == []

    def test_a_call_outside_any_test_function_is_ignored(self, ws):
        """Module-level setup is not an assertion about the model."""
        assert ws('''
            from server.app.schemas.birthday import BirthdayResponse

            FIXTURE = BirthdayResponse(id='aaa')

            def helper():
                return BirthdayResponse(id='aaa')
        ''') == []


class TestTheFindingsAreStable:
    def test_order_is_deterministic(self, ws):
        """A re-prompt is built from these, so the message must not reshuffle
        between rounds."""
        src = '''
            from server.app.schemas.birthday import BirthdayResponse

            def test_b():
                obj = BirthdayResponse(id='aaa')

            def test_a():
                obj = BirthdayResponse(first_name='aaa')
        '''
        first = [(f.file, f.line) for f in ws(src)]
        second = [(f.file, f.line) for f in ws(src)]
        assert first == second == sorted(first)
        assert len(first) == 2

    def test_a_pydantic_v1_style_optional_default_is_respected(self, ws):
        """``X | None = None`` HAS a default, so omitting it is fine. The
        distinction this gate rests on is default-vs-no-default, not
        nullable-vs-not."""
        (ws.path / SCHEMA_REL).write_text(textwrap.dedent('''
            from pydantic import BaseModel


            class M(BaseModel):
                a: str
                b: str | None = None
        '''))
        assert ws('''
            from server.app.schemas.birthday import M

            def test_x():
                obj = M(a='1')
                assert obj
        ''') == []


class TestTheConfigKeyIsWired:
    def test_the_key_is_known_and_typed(self):
        from harness import cli
        assert (
            "unexecutable_call_review"
            in cli._KNOWN_NESTED_KEYS["test_generation"]
        )
        assert (
            cli._TYPE_SCHEMA["test_generation.unexecutable_call_review"]
            == (bool,)
        )

    def test_it_ships_enabled_and_documented(self):
        import json
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[1]
        cfg = json.loads((root / "config" / "config.json").read_text())
        assert cfg["test_generation"]["unexecutable_call_review"] is True
        help_md = (root / "config" / "config_help.md").read_text()
        assert "unexecutable_call_review" in help_md

    def test_the_raises_predicate_is_shared_with_the_other_gate(self):
        """Both deterministic gates must agree on what opens a "must raise"
        context, or one will flag what the other deliberately writes."""
        from harness import test_contradiction as tc
        assert "is_raises_call" in tc.__all__


class _RecordingGateway:
    """Records the messages the author was re-prompted with.

    Answers nothing, so the gate's "landed no patch" branch ends the loop
    after one round — which is all this needs: the assertion is about what
    the author was TOLD, not about it complying.
    """

    class config:
        repair_fallback = ""
        planning_fallback = ""

    def __init__(self) -> None:
        self.seen: list[list[dict]] = []

    def select_model(self, _role):
        return "stub:model"

    async def dispatch(self, *, messages, role, budget_remaining_usd, **kw):
        self.seen.append([dict(m) for m in messages])

        class _R:
            content = ""

            class usage:
                cost_usd = 0.001
                input_tokens = 10
                output_tokens = 10
                cached_tokens = 0
                model = "stub"

        return _R(), budget_remaining_usd - 0.001

    def aggregate_tokens(self, tracker, usage, role=None):
        return dict(tracker or {})


class TestTheGateActuallyRunsIt:
    """The detector working is not proof the gate calls it.

    ``_run_contradiction_gate`` carried a defect for as long as it was inline
    precisely because nothing executed it, so the wiring gets its own test.
    """

    async def _drive(self, tmp_path, *, cfg=None):
        import harness.test_generation as tg

        (tmp_path / "server/app/schemas").mkdir(parents=True)
        (tmp_path / "server/tests/contract").mkdir(parents=True)
        (tmp_path / SCHEMA_REL).write_text(SCHEMA)
        (tmp_path / TEST_REL).write_text(RUN20_TEST)
        gw = _RecordingGateway()
        out = await tg._run_contradiction_gate(
            state={
                "harness_config": {
                    "patcher": {"use_structured_tools": False},
                },
            },
            gateway=gw,
            workspace_path=str(tmp_path),
            generated_tests=[TEST_REL],
            messages=[{"role": "user", "content": "generate tests"}],
            existing_modified=[],
            allowed_paths=None,
            budget=10.0,
            token_tracker={},
            cfg=cfg if cfg is not None else {},
        )
        return gw, out

    @pytest.mark.asyncio
    async def test_the_author_is_re_prompted_with_the_missing_fields(
        self, tmp_path,
    ):
        gw, out = await self._drive(tmp_path)
        assert out["reprompts"] == 1, "the gate never re-prompted"
        banner = "\n".join(
            m["content"] for conv in gw.seen for m in conv
            if m.get("role") == "system"
        )
        assert "cannot execute against the model they name" in banner
        for name in ("last_name", "birth_year", "phone_number"):
            assert name in banner, name

    @pytest.mark.asyncio
    async def test_the_author_is_told_not_to_delete_or_wrap_the_test(
        self, tmp_path,
    ):
        """The remediation must be unambiguous. "Delete it" and "wrap it in
        pytest.raises" both make the suite green by discarding the
        requirement — the silent failure this gate must never cause."""
        gw, _ = await self._drive(tmp_path)
        banner = "\n".join(
            m["content"] for conv in gw.seen for m in conv
            if m.get("role") == "system"
        )
        assert "Do NOT delete the test" in banner
        assert "pytest.raises" in banner
        assert "PASSING each missing field" in banner
        assert "not production lagging the test" in banner

    @pytest.mark.asyncio
    async def test_the_flag_turns_it_off(self, tmp_path):
        gw, out = await self._drive(
            tmp_path, cfg={"unexecutable_call_review": False},
        )
        assert out["reprompts"] == 0
        assert gw.seen == []
