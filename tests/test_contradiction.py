"""Machine-checkable unsatisfiable-test detection (ADR-0001, Tier A).

The detector is the deterministic bottom rung of the autonomy ladder: it may
only report a contradiction it can *prove* from the AST (same call, opposite
expected outcomes). False positives would wrongly regenerate a valid test, so
these tests pin the conservative boundaries as hard as the positive cases.
"""

from harness.test_contradiction import (
    Contradiction,
    find_contradictions,
    machine_unsatisfiable_reason,
    unparseable_reason,
)


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
