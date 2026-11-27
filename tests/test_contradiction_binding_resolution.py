"""A call signature is not an input unless its arguments are resolved.

``_normalize_call`` compared ``ast.unparse`` text and called a match "identical
input". That is false the moment an argument arrives through a variable, and
lumina 01a21929 (run 20) hit it on real generated tests::

    def test_feb29_leap_year(self):
        contact = _contact(2, 29)
        result = resolve_next_occurrence(contact, date(2024, 1, 1), "UTC")
        assert result == date(2024, 2, 29)

    def test_invalid_date_raises(self):
        contact = _contact(2, 31)
        with pytest.raises(BirthdateValidationError):
            resolve_next_occurrence(contact, date(2024, 1, 1), "UTC")

Both unparse to ``resolve_next_occurrence(contact, date(2024, 1, 1), 'UTC')``,
so the prover announced "identical input, opposite expected outcomes, so no
production change can satisfy both" and bounced the batch back to the test
author twice (00:16:35, 00:16:51) before shipping it to the build anyway.

Feb 29 is a real leap day. Feb 31 is not a date. The inputs differ, the
expectations are consistent, and both tests were correct — they are still on
disk and were never part of a failing set. Had the author complied with the
proof it would have deleted a correct requirement, which the project treats as
worse than the bug the pass exists for, because it is silent.

The module's own docstring already promised this could not happen: "Differing
arguments (``X(a)`` vs ``X(b)``) are never flagged". The fix makes that true by
resolving bindings into the signature, so the two sides differ on their own.

:class:`TestItStillProvesWhatItShould` is the half that matters most. Going
quiet is not a fix — a pair that genuinely binds the same value must still be
proven, or this change has traded a false positive for a blind spot.
"""

from __future__ import annotations

import textwrap

from harness.test_contradiction import (
    find_contradictions,
    find_contradictions_across,
)


def _find(src: str, filename: str = "server/tests/test_x.py"):
    return find_contradictions(textwrap.dedent(src), filename=filename)


#: Verbatim from lumina 01a21929's ``server/tests/test_birthday_service.py``,
#: trimmed to the pair the prover accused.
RUN20_PAIR = '''
    import pytest
    from datetime import date


    class TestResolveNextOccurrence:
        def test_feb29_leap_year(self):
            contact = _contact(2, 29)
            result = resolve_next_occurrence(contact, date(2024, 1, 1), "UTC")
            assert result == date(2024, 2, 29)

        def test_invalid_date_raises(self):
            contact = _contact(2, 31)
            with pytest.raises(BirthdateValidationError):
                resolve_next_occurrence(contact, date(2024, 1, 1), "UTC")
'''


class TestTheRun20FalsePositive:
    def test_the_accused_pair_is_not_a_contradiction(self):
        assert _find(RUN20_PAIR) == []

    def test_differing_bound_values_are_never_flagged(self):
        """The module docstring's promise, now actually kept."""
        assert _find('''
            import pytest
            def test_a():
                c = make(2, 29)
                r = resolve(c, "UTC")
            def test_b():
                c = make(2, 31)
                with pytest.raises(E):
                    resolve(c, "UTC")
        ''') == []

    def test_a_differing_receiver_is_not_a_contradiction(self):
        """The bound name can be the method's receiver rather than an
        argument, and that is just as much an input."""
        assert _find('''
            import pytest
            def test_a():
                obj = Thing(1)
                obj.save()
            def test_b():
                obj = Thing(2)
                with pytest.raises(E):
                    obj.save()
        ''') == []

    def test_a_differing_keyword_value_is_not_a_contradiction(self):
        assert _find('''
            import pytest
            def test_a():
                p = payload(ok=True)
                r = Thing(body=p)
            def test_b():
                p = payload(ok=False)
                with pytest.raises(E):
                    Thing(body=p)
        ''') == []

    def test_it_holds_across_files_too(self):
        """``find_contradictions_across`` shares the scanner, so the
        cross-file detector inherits the fix rather than needing its own."""
        assert find_contradictions_across({
            "server/tests/test_a.py": textwrap.dedent('''
                def test_ok():
                    c = make(2, 29)
                    r = resolve(c, "UTC")
            '''),
            "server/tests/test_b.py": textwrap.dedent('''
                import pytest
                def test_raises():
                    c = make(2, 31)
                    with pytest.raises(E):
                        resolve(c, "UTC")
            '''),
        }) == []


class TestItStillProvesWhatItShould:
    def test_the_original_literal_arg_contradiction(self):
        """lumina ``test_contact_models.py`` — the case the prover was built
        for. No variables involved, so nothing about it changes."""
        [c] = _find('''
            import pytest
            def test_all_none_raises():
                with pytest.raises(ValueError):
                    ContactUpdate(first_name=None)
            def test_none_fields_allowed():
                obj = ContactUpdate(first_name=None)
        ''')
        assert c.call == "ContactUpdate(first_name=None)"
        assert c.expect_raise_test == "test_all_none_raises"
        assert c.expect_success_test == "test_none_fields_allowed"

    def test_the_same_bound_value_on_both_sides_is_still_proven(self):
        """The decisive one. Resolving bindings must not mean ignoring any
        signature that has a variable in it — an identical binding is an
        identical input, and the contradiction is real."""
        [c] = _find('''
            import pytest
            def test_a():
                c = make(2, 29)
                r = resolve(c, "UTC")
            def test_b():
                c = make(2, 29)
                with pytest.raises(E):
                    resolve(c, "UTC")
        ''')
        assert c.call == "resolve(make(2, 29), 'UTC')", c.call

    def test_the_resolved_value_appears_in_the_message(self):
        """The signature is what the author is shown. Substituting makes the
        accusation legible instead of naming a variable the author then has to
        chase."""
        [c] = _find('''
            import pytest
            def test_a():
                c = make(2, 29)
                r = resolve(c, "UTC")
            def test_b():
                c = make(2, 29)
                with pytest.raises(E):
                    resolve(c, "UTC")
        ''')
        assert "make(2, 29)" in c.describe()
        assert "contact" not in c.describe()

    def test_a_module_constant_resolves_the_same_in_every_test(self):
        [c] = _find('''
            import pytest
            PAYLOAD = dict(a=1)
            def test_a():
                r = Thing(PAYLOAD)
            def test_b():
                with pytest.raises(E):
                    Thing(PAYLOAD)
        ''')
        assert c.call == "Thing(dict(a=1))", c.call

    def test_a_function_local_binding_shadows_the_module_one(self):
        """Scope has to resolve the way Python does, or the substitution is
        just a different wrong answer."""
        assert _find('''
            import pytest
            VALUE = make(1)
            def test_a():
                r = Thing(VALUE)
            def test_b():
                VALUE = make(2)
                with pytest.raises(E):
                    Thing(VALUE)
        ''') == []

    def test_the_same_receiver_binding_is_still_proven(self):
        [c] = _find('''
            import pytest
            def test_a():
                obj = Thing(1)
                obj.save()
            def test_b():
                obj = Thing(1)
                with pytest.raises(E):
                    obj.save()
        ''')
        assert c.call == "Thing(1).save()", c.call

    def test_a_fixture_parameter_needs_no_resolution(self):
        """Two tests taking the same fixture receive the same object, so the
        name is already an input and must not be treated as unresolvable."""
        [c] = _find('''
            import pytest
            def test_a(client):
                r = fetch(client)
            def test_b(client):
                with pytest.raises(E):
                    fetch(client)
        ''')
        assert c.call == "fetch(client)"


class TestItDeclinesWhatItCannotResolve:
    def test_a_for_loop_target_is_not_an_input(self):
        assert _find('''
            import pytest
            def test_a():
                for v in [1, 2]:
                    r = Thing(v)
            def test_b():
                for v in [3, 4]:
                    with pytest.raises(E):
                        Thing(v)
        ''') == []

    def test_a_with_as_target_is_not_an_input(self):
        assert _find('''
            import pytest
            def test_a():
                with open("a") as fh:
                    r = Thing(fh)
            def test_b():
                with open("b") as fh:
                    with pytest.raises(E):
                        Thing(fh)
        ''') == []

    def test_tuple_unpacking_is_not_an_input(self):
        assert _find('''
            import pytest
            def test_a():
                x, y = pair(1)
                r = Thing(x)
            def test_b():
                x, y = pair(2)
                with pytest.raises(E):
                    Thing(x)
        ''') == []

    def test_an_imported_name_is_not_resolved_across_files(self):
        """Conservative by choice: a name bound in another module cannot be
        compared, so the pair falls through to the next rung of the ladder
        instead of being guessed at."""
        assert find_contradictions_across({
            "server/tests/test_a.py": textwrap.dedent('''
                from server.tests.fixtures import PAYLOAD
                def test_ok():
                    r = Thing(PAYLOAD)
            '''),
            "server/tests/test_b.py": textwrap.dedent('''
                import pytest
                from server.tests.fixtures import PAYLOAD
                def test_raises():
                    with pytest.raises(E):
                        Thing(PAYLOAD)
            '''),
        }) == []


class TestTheExistingGuardsStillHold:
    def test_a_state_dependent_sequence_through_a_bound_name(self):
        """The uniqueness-constraint shape, with the payload in a variable.
        Substitution must keep both calls identical so the dual
        raise/success classification still marks the signature
        state-dependent."""
        assert _find('''
            import pytest
            def test_rejects_duplicate(conn):
                emp = make_employee()
                create(conn, emp)
                with pytest.raises(Dup):
                    create(conn, emp)
            def test_assigns_generated_id(conn):
                emp = make_employee()
                r = create(conn, emp)
        ''') == []

    def test_an_ambient_test_is_still_excluded(self):
        assert _find('''
            import pytest
            def test_a(monkeypatch):
                monkeypatch.setenv("PORT", "8080")
                r = get_settings()
            def test_b(monkeypatch):
                monkeypatch.setenv("PORT", "bad")
                with pytest.raises(E):
                    get_settings()
        ''') == []

    def test_a_self_referential_binding_does_not_recurse(self):
        """``x = x + 1`` must not be substituted back into itself.

        Terminating is half the assertion — a recursive substitution would
        hang here. The other half is the answer: the binding is skipped, ``x``
        falls back to being the test's own parameter, both sides derive it the
        same way, so the contradiction stands and is reported against the
        unsubstituted name.
        """
        [c] = _find('''
            import pytest
            def test_a(x):
                x = x + 1
                r = Thing(x)
            def test_b(x):
                x = x + 1
                with pytest.raises(E):
                    Thing(x)
        ''')
        assert c.call == "Thing(x)"

    def test_a_single_function_contradicting_itself_is_not_flagged(self):
        assert _find('''
            import pytest
            def test_only():
                obj = ContactUpdate(first_name=None)
                with pytest.raises(ValueError):
                    ContactUpdate(first_name=None)
        ''') == []
