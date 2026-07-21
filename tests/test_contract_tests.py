"""ADR-0003 Tier 1 — deterministic schema-contract test emitter.

The emitter's whole value rests on one property: every test it emits is
CORRECT against the real model, or it emits nothing. These tests pin the
conservative boundary (validator-bearing models skipped) as hard as the
positive cases, because a wrong emitted test reintroduces the 019f803f
failure it exists to prevent.
"""

from harness.contract_tests import (
    emit_api_contract_tests,
    emit_contract_tests,
    find_fastapi_app,
    parse_fastapi_routes,
    parse_pydantic_models,
    render_api_contract_test,
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


# ---------------------------------------------------------------------------
# Tier 2 — API status-code contracts
# ---------------------------------------------------------------------------

_MAIN = '''
from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI()
    return app


app = create_app()
'''

_ROUTES = '''
from fastapi import APIRouter, Depends
from app.schemas import ContactCreate, ContactUpdate

router = APIRouter(prefix="/api/contacts")


@router.get("")
def list_contacts(db=Depends(get_db)):
    ...


@router.post("", status_code=201)
def create(payload: ContactCreate, db=Depends(get_db)):
    ...


@router.put("/{contact_id}")
def update(contact_id: int, payload: ContactUpdate, db=Depends(get_db)):
    ...


@router.delete("/{contact_id}", status_code=200)
def delete(contact_id: int, db=Depends(get_db)):
    ...
'''

_SCHEMAS_FOR_ROUTES = '''
from pydantic import BaseModel
from typing import Optional


class ContactCreate(BaseModel):
    name: str          # required


class ContactUpdate(BaseModel):
    name: Optional[str] = None   # all-optional
'''


class TestFindApp:
    def test_finds_factory_and_direct(self):
        assert find_fastapi_app(_MAIN, rel_path="app/main.py") == ("app.main", "app")
        direct = "from fastapi import FastAPI\napp = FastAPI()\n"
        assert find_fastapi_app(direct, rel_path="m.py") == ("m", "app")

    def test_none_when_no_app(self):
        assert find_fastapi_app("x = 1\n", rel_path="m.py") is None


class TestParseRoutes:
    def _routes(self):
        model_required = {"ContactCreate": True, "ContactUpdate": False}
        return {
            r.func_name: r
            for r in parse_fastapi_routes(
                _ROUTES, rel_path="app/api.py", model_required=model_required,
            )
        }

    def test_router_prefix_applied_and_body_detected(self):
        r = self._routes()
        assert r["create"].path == "/api/contacts"
        assert r["create"].method == "post"
        assert r["create"].body_model == "ContactCreate"
        assert r["create"].body_required is True

    def test_optional_body_is_not_required(self):
        r = self._routes()
        assert r["update"].body_model == "ContactUpdate"
        assert r["update"].body_required is False   # all-optional body

    def test_int_path_params_detected(self):
        r = self._routes()
        assert r["update"].int_path_params == ("contact_id",)
        assert r["delete"].int_path_params == ("contact_id",)
        assert r["list_contacts"].int_path_params == ()

    def test_depends_param_not_treated_as_body(self):
        r = self._routes()
        # `db=Depends(get_db)` must never be picked as the body model.
        assert r["list_contacts"].body_model is None


class TestRenderApi:
    def test_emits_only_deterministic_422s(self):
        model_required = {"ContactCreate": True, "ContactUpdate": False}
        routes = parse_fastapi_routes(
            _ROUTES, rel_path="app/api.py", model_required=model_required,
        )
        body = render_api_contract_test(
            routes, app_module="app.main", app_var="app", source_rel="app/api.py",
        )
        assert body is not None
        assert "from app.main import app" in body
        assert "TestClient(app)" in body
        # POST (required body) → empty-body 422
        assert "test_post_api_contacts_empty_body_422" in body
        # PUT/DELETE int path → bad-type 422
        assert "test_put_api_contacts_contact_id_bad_type_422" in body
        assert "test_delete_api_contacts_contact_id_bad_type_422" in body
        # PUT body is all-optional → NO empty-body test for it
        assert "test_put_api_contacts_empty_body_422" not in body
        # GET has nothing deterministic → no test
        assert "test_get_" not in body

    def test_none_when_no_testable_routes(self):
        src = (
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            "@router.get('/health')\n"
            "def health():\n    ...\n"
        )
        routes = parse_fastapi_routes(src, rel_path="h.py", model_required={})
        assert render_api_contract_test(
            routes, app_module="m", app_var="app", source_rel="h.py",
        ) is None


class TestEmitApiToDisk:
    def _w(self, tmp_path, rel, body):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)

    def test_emits_when_app_and_routes_present(self, tmp_path):
        self._w(tmp_path, "app/main.py", _MAIN)
        self._w(tmp_path, "app/api.py", _ROUTES)
        self._w(tmp_path, "app/schemas.py", _SCHEMAS_FOR_ROUTES)
        written, markers = emit_api_contract_tests(
            str(tmp_path),
            ["app/main.py", "app/api.py", "app/schemas.py"],
            "python",
        )
        assert written == ["tests/contract/test_api_api_contract.py"]
        out = (tmp_path / written[0]).read_text()
        assert "from app.main import app" in out
        assert "empty_body_422" in out

    def test_noop_without_app_instance(self, tmp_path):
        # Routes but no FastAPI app anywhere → can't build a TestClient.
        self._w(tmp_path, "app/api.py", _ROUTES)
        self._w(tmp_path, "app/schemas.py", _SCHEMAS_FOR_ROUTES)
        assert emit_api_contract_tests(
            str(tmp_path), ["app/api.py", "app/schemas.py"], "python",
        ) == ([], {})

    def test_non_python_noop(self, tmp_path):
        self._w(tmp_path, "app/main.py", _MAIN)
        assert emit_api_contract_tests(
            str(tmp_path), ["app/main.py"], "typescript",
        ) == ([], {})
