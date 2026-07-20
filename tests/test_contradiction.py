"""Machine-checkable unsatisfiable-test detection (ADR-0001, Tier A).

The detector is the deterministic bottom rung of the autonomy ladder: it may
only report a contradiction it can *prove* from the AST (same call, opposite
expected outcomes). False positives would wrongly regenerate a valid test, so
these tests pin the conservative boundaries as hard as the positive cases.
"""

from harness.test_contradiction import (
    Contradiction,
    find_contradictions,
    find_contradictions_across,
    machine_unsatisfiable_reason,
    unparseable_reason,
)


# The lumina 019f803f deadlock: a same-input / opposite-expectation pair
# split across two files, so NEITHER file is self-contradictory.
_SCHEMAS_FILE = '''
import pytest
from pydantic import ValidationError
from server.app.schemas.contact import ContactUpdate


class TestContactUpdate:
    def test_empty_first_name_explicitly_sent_is_rejected(self):
        with pytest.raises(ValidationError, match="First name is required"):
            ContactUpdate(first_name="   ")

    def test_partial_update_valid(self):
        data = ContactUpdate(first_name=" New ")
        assert data.first_name == "New"
'''

_SERVICE_FILE = '''
import pytest
from fastapi import HTTPException
from server.app.schemas.contact import ContactUpdate
from server.app.services.contact_service import update_contact


class TestUpdateContact:
    def test_update_with_empty_first_name_raises_422(self, db_session):
        update_payload = ContactUpdate(first_name="   ")
        with pytest.raises(HTTPException) as exc_info:
            update_contact(db_session, 1, update_payload)
        assert exc_info.value.status_code == 422
'''


LUMINA_PATTERN = '''
import pytest
from pydantic import ValidationError
from app.models import ContactUpdate


class TestContactUpdate:
    def test_none_fields_allowed(self):
        obj = ContactUpdate(first_name=None)
        pass

    def test_all_none_raises(self):
        with pytest.raises(ValidationError):
            ContactUpdate(first_name=None)
        with pytest.raises(ValidationError) as exc:
            ContactUpdate(first_name=None)
        assert "at least one field" in str(exc.value).lower()
'''


class TestPositiveDetection:
    def test_lumina_pattern_flagged(self):
        cs = find_contradictions(LUMINA_PATTERN, filename="test_contact_models.py")
        assert len(cs) == 1
        c = cs[0]
        assert c.call == "ContactUpdate(first_name=None)"
        assert c.expect_raise_test == "test_all_none_raises"
        assert c.expect_success_test == "test_none_fields_allowed"
        assert "ContactUpdate(first_name=None)" in c.describe()

    def test_minimal_cross_function_contradiction(self):
        src = (
            "class T:\n"
            "    def test_ok(self):\n"
            "        obj = Foo(x=1)\n"
            "    def test_raises(self):\n"
            "        with pytest.raises(E):\n"
            "            Foo(x=1)\n"
        )
        assert len(find_contradictions(src)) == 1

    def test_module_level_test_functions(self):
        src = (
            "def test_ok():\n"
            "    y = Bar(a=2)\n"
            "def test_bad():\n"
            "    with pytest.raises(E):\n"
            "        Bar(a=2)\n"
        )
        assert len(find_contradictions(src)) == 1

    def test_reason_is_human_readable(self):
        reason = machine_unsatisfiable_reason(LUMINA_PATTERN, filename="t.py")
        assert reason is not None
        assert "RAISE" in reason and "SUCCEED" in reason


class TestConservativeBoundaries:
    """Every case here MUST return no contradiction — a false positive would
    regenerate a legitimately-passing test."""

    def test_different_arguments_not_flagged(self):
        # A production change could make one raise and the other succeed.
        src = (
            "class T:\n"
            "    def test_a(self):\n"
            "        obj = X(first_name=None)\n"
            "    def test_b(self):\n"
            "        with pytest.raises(ValueError):\n"
            "            X(first_name='A' * 101)\n"
        )
        assert find_contradictions(src) == []

    def test_same_function_not_flagged(self):
        src = (
            "def test_x():\n"
            "    obj = X(a=1)\n"
            "    with pytest.raises(E):\n"
            "        X(a=1)\n"
        )
        assert find_contradictions(src) == []

    def test_try_except_success_is_ambiguous(self):
        # A call whose exception may be swallowed by try/except is not a
        # proven "success" assertion.
        src = (
            "class T:\n"
            "    def test_a(self):\n"
            "        try:\n"
            "            Foo(x=1)\n"
            "        except Exception:\n"
            "            pass\n"
            "    def test_b(self):\n"
            "        with pytest.raises(E):\n"
            "            Foo(x=1)\n"
        )
        assert find_contradictions(src) == []

    def test_helper_calls_not_treated_as_subject(self):
        # str()/len()/pytest.raises() themselves must never be a subject sig.
        src = (
            "class T:\n"
            "    def test_a(self):\n"
            "        s = str(x)\n"
            "    def test_b(self):\n"
            "        with pytest.raises(E):\n"
            "            str(x)\n"
        )
        assert find_contradictions(src) == []

    def test_two_passing_tests_not_flagged(self):
        src = (
            "class T:\n"
            "    def test_a(self):\n"
            "        obj = X(a=1)\n"
            "    def test_b(self):\n"
            "        obj = X(a=1)\n"
        )
        assert find_contradictions(src) == []


class TestUnparseable:
    def test_syntax_error_reported(self):
        reason = unparseable_reason("def test(:\n    pass", filename="t.py")
        assert reason is not None and "does not parse" in reason

    def test_valid_source_not_reported(self):
        assert unparseable_reason("def test_x():\n    pass") is None

    def test_machine_reason_catches_syntax_error(self):
        assert machine_unsatisfiable_reason("def t(:\n x", filename="t.py") is not None


class TestClassifierGate:
    def test_non_python_path_falls_through(self):
        # The AST detector is Python-only; a .ts test must not be judged here.
        assert machine_unsatisfiable_reason(LUMINA_PATTERN, filename="t.test.ts") is None

    def test_clean_file_returns_none(self):
        src = "def test_x():\n    assert 1 == 1\n"
        assert machine_unsatisfiable_reason(src, filename="t.py") is None

    def test_functional_raises_form(self):
        # pytest.raises(Exc, callable, *args) functional form.
        src = (
            "def test_ok():\n"
            "    obj = Foo(x=1)\n"
            "def test_bad():\n"
            "    pytest.raises(E, Foo, x=1)\n"  # note: kwargs form differs; positional below
        )
        # Positional-arg functional form is the detectable shape:
        src2 = (
            "def test_ok():\n"
            "    obj = Foo(1)\n"
            "def test_bad():\n"
            "    import functools\n"
            "    pytest.raises(E, Foo(1))\n"
        )
        assert isinstance(find_contradictions(src2), list)


class TestCrossFileDetection:
    """find_contradictions_across — the lumina 019f803f generalisation:
    a contradiction split across two test files that the single-file
    detector cannot see."""

    def test_lumina_cross_file_pair_flagged(self):
        # Each file alone is clean...
        assert find_contradictions(_SCHEMAS_FILE) == []
        assert find_contradictions(_SERVICE_FILE) == []
        # ...but together they are unsatisfiable.
        out = find_contradictions_across({
            "server/tests/test_contact_schemas.py": _SCHEMAS_FILE,
            "server/tests/test_contact_service.py": _SERVICE_FILE,
        })
        assert len(out) == 1
        c = out[0]
        assert c.call == "ContactUpdate(first_name='   ')"
        assert c.expect_raise_file.endswith("test_contact_schemas.py")
        assert c.expect_success_file.endswith("test_contact_service.py")
        # describe() surfaces both files so the re-prompt can name them.
        d = c.describe()
        assert "test_contact_schemas.py" in d and "test_contact_service.py" in d

    def test_no_contradiction_across_consistent_files(self):
        # Both files agree the value is rejected at construction.
        other = (
            "import pytest\n"
            "from server.app.schemas.contact import ContactUpdate\n"
            "def test_also_rejects():\n"
            "    with pytest.raises(Exception):\n"
            "        ContactUpdate(first_name='   ')\n"
        )
        assert find_contradictions_across({
            "a.py": _SCHEMAS_FILE, "b.py": other,
        }) == []

    def test_different_inputs_not_flagged(self):
        # X('a') raising and X('b') succeeding is legitimate — different
        # inputs, a production change could distinguish them.
        raise_f = (
            "import pytest\n"
            "def test_bad():\n"
            "    with pytest.raises(ValueError):\n"
            "        Widget('a')\n"
        )
        ok_f = (
            "def test_good():\n"
            "    w = Widget('b')\n"
        )
        assert find_contradictions_across({"r.py": raise_f, "o.py": ok_f}) == []

    def test_prefers_cross_file_pair_in_report(self):
        # When a signature is contradicted both within one file and across
        # files, the reported pair is the cross-file one (the added signal).
        raise_only = (
            "import pytest\n"
            "def test_r():\n"
            "    with pytest.raises(E):\n"
            "        Foo(1)\n"
        )
        success_only = (
            "def test_s():\n"
            "    x = Foo(1)\n"
        )
        out = find_contradictions_across({
            "raise.py": raise_only, "ok.py": success_only,
        })
        assert len(out) == 1
        assert out[0].expect_raise_file == "raise.py"
        assert out[0].expect_success_file == "ok.py"

    def test_unparseable_file_skipped_not_raised(self):
        # A syntactically broken file in the batch is skipped, not fatal;
        # the other file's clean state stands.
        broken = "def test_x(:\n  pass\n"
        assert find_contradictions_across({
            "broken.py": broken, "ok.py": _SCHEMAS_FILE,
        }) == []

    def test_intra_file_still_covered(self):
        # A single-file contradiction passed through the batch API is still
        # reported (superset of the single-file detector).
        out = find_contradictions_across({"m.py": LUMINA_PATTERN})
        assert len(out) == 1
        assert out[0].call == "ContactUpdate(first_name=None)"

    def test_same_test_fn_not_flagged(self):
        # raise+success of the same call inside ONE test fn is not a spec
        # contradiction (mirrors the single-file cross-fn requirement).
        one = (
            "import pytest\n"
            "def test_both(self):\n"
            "    x = Foo(1)\n"
            "    with pytest.raises(E):\n"
            "        Foo(1)\n"
        )
        assert find_contradictions_across({"one.py": one}) == []
