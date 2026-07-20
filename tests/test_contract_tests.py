"""ADR-0003 Tier 1 — deterministic schema-contract test emitter.

The emitter's whole value rests on one property: every test it emits is
CORRECT against the real model, or it emits nothing. These tests pin the
conservative boundary (validator-bearing models skipped) as hard as the
positive cases, because a wrong emitted test reintroduces the 019f803f
failure it exists to prevent.
"""

from harness.contract_tests import (
    emit_contract_tests,
    parse_pydantic_models,
    render_contract_test,
)


_PLAIN_MODEL = '''
from pydantic import BaseModel, Field
from typing import Optional


class Widget(BaseModel):
    name: str = Field(max_length=10)
    qty: int = Field(ge=0, le=100)
    note: Optional[str] = None
'''

_VALIDATOR_MODEL = '''
from pydantic import BaseModel, Field, field_validator, model_validator


class Account(BaseModel):
    email: str = Field(max_length=50)

    @field_validator("email")
    @classmethod
    def _check(cls, v):
        if "@" not in v:
            raise ValueError("bad email")
        return v


class Balance(BaseModel):
    amount: int

    @model_validator(mode="after")
    def _nonneg(self):
        if self.amount < 0:
            raise ValueError("negative")
        return self
'''


class TestParsing:
    def test_extracts_constraints_and_requiredness(self):
        [m] = parse_pydantic_models(_PLAIN_MODEL, rel_path="app/w.py")
        assert m.name == "Widget"
        by = {f.name: f for f in m.fields}
        assert by["name"].required is True
        assert by["name"].constraints["max_length"] == 10
        assert by["qty"].constraints["ge"] == 0 and by["qty"].constraints["le"] == 100
        assert by["note"].required is False   # Optional
        assert m.module_import == "app.w"

    def test_detects_field_and_model_validators(self):
        models = {m.name: m for m in parse_pydantic_models(_VALIDATOR_MODEL)}
        assert models["Account"].fields[0].has_custom_validator is True
        assert models["Balance"].has_model_validator is True

    def test_non_pydantic_class_ignored(self):
        src = "class Plain:\n    x = 1\n"
        assert parse_pydantic_models(src) == []

    def test_syntax_error_returns_empty(self):
        assert parse_pydantic_models("class X(:\n") == []


class TestRendering:
    def test_plain_model_emits_correct_tests(self):
        models = parse_pydantic_models(_PLAIN_MODEL, rel_path="app/w.py")
        body = render_contract_test(models, source_rel="app/w.py")
        assert body is not None
        assert "# @tests: app/w.py" in body
        assert "from app.w import Widget" in body
        # happy path + required(name) + max_length(name) + range(qty x2)
        assert "def test_widget_valid_construction" in body
        assert "def test_widget_requires_name" in body
        assert "def test_widget_name_max_length" in body
        assert "def test_widget_qty_below_min" in body
        assert "def test_widget_qty_above_max" in body
        # 'note' is Optional → no required test for it
        assert "requires_note" not in body

    def test_validator_bearing_models_skipped(self):
        # Account has a field_validator on its required field; Balance has a
        # model_validator. Neither can be given a provably-valid instance.
        models = parse_pydantic_models(_VALIDATOR_MODEL, rel_path="app/a.py")
        body = render_contract_test(models, source_rel="app/a.py")
        assert body is None

    def test_unsynthesisable_type_skips_model(self):
        src = (
            "from pydantic import BaseModel\n"
            "class HasNested(BaseModel):\n"
            "    inner: SomeOtherModel\n"   # required, non-scalar → can't synthesise
        )
        models = parse_pydantic_models(src, rel_path="app/n.py")
        assert render_contract_test(models, source_rel="app/n.py") is None


class TestEmitToDisk:
    def _write(self, tmp_path, rel, body):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)

    def test_emits_file_with_marker_and_is_idempotent(self, tmp_path):
        self._write(tmp_path, "app/w.py", _PLAIN_MODEL)
        written, markers = emit_contract_tests(
            str(tmp_path), ["app/w.py"], "python",
        )
        assert written == ["tests/contract/test_w_contract.py"]
        assert markers["tests/contract/test_w_contract.py"] == ["app/w.py"]
        out = (tmp_path / "tests/contract/test_w_contract.py").read_text()
        assert "# @tests: app/w.py" in out

        # Second run: file exists → not overwritten, but marker edge recorded.
        again_written, again_markers = emit_contract_tests(
            str(tmp_path), ["app/w.py"], "python",
        )
        assert again_written == []
        assert "tests/contract/test_w_contract.py" in again_markers

    def test_non_python_stack_is_noop(self, tmp_path):
        self._write(tmp_path, "app/w.py", _PLAIN_MODEL)
        assert emit_contract_tests(str(tmp_path), ["app/w.py"], "typescript") == ([], {})

    def test_skips_test_files_and_validator_models(self, tmp_path):
        self._write(tmp_path, "app/a.py", _VALIDATOR_MODEL)
        self._write(tmp_path, "tests/test_thing.py", "def test_x(): pass\n")
        written, _ = emit_contract_tests(
            str(tmp_path), ["app/a.py", "tests/test_thing.py"], "python",
        )
        assert written == []  # validator model skipped; test file ignored
