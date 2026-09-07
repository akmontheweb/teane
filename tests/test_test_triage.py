"""ADR-0005 pre-repair triage classifier.

Fixtures are the REAL diagnostics from lumina run 019ff418 (the run whose
repair loop was 77% of LLM calls). The classifier must divert only
unambiguous test-authoring bugs to regeneration and leave every code-gap /
ambiguous failure to repair.
"""

from __future__ import annotations

from harness.sandbox import DiagnosticObject
from harness.test_triage import (
    FailureClass,
    classify_diagnostic,
    summarize_failures,
)


def _diag(file="", error_code="", message="", **kw):
    return DiagnosticObject(
        file=file, error_code=error_code, message=message, **kw
    )


class TestTestBugFingerprints:
    def test_undefined_name_in_test_is_test_bug(self):
        # tests/unit/test_contact_service.py:83:0 - NameError: name 'patch' is not defined
        d = _diag(
            "tests/unit/test_contact_service.py", "NameError",
            "name 'patch' is not defined",
        )
        r = classify_diagnostic(d)
        assert r.fclass is FailureClass.TEST_BUG
        assert r.fingerprint == "test-undefined-name"
        assert r.confidence == "high"
        assert "patch" in r.reason

    def test_raw_row_subscript_in_test_is_test_bug(self):
        # tests/unit/test_database.py:41:0 - TypeError: tuple indices must be integers or slices, not str
        d = _diag(
            "tests/unit/test_database.py", "TypeError",
            "tuple indices must be integers or slices, not str",
        )
        r = classify_diagnostic(d)
        assert r.fclass is FailureClass.TEST_BUG
        assert r.fingerprint == "test-raw-row-subscript"

    def test_unresolved_patch_target_is_test_bug(self):
        # AttributeError: <module 'server.app.services'> does not have the attribute 'foo'
        d = _diag(
            "tests/unit/test_contact_service.py", "AttributeError",
            "<module 'server.app.services.contact_service'> does not have the "
            "attribute 'nonexistent'",
        )
        r = classify_diagnostic(d)
        assert r.fclass is FailureClass.TEST_BUG
        assert r.fingerprint == "test-unresolved-patch-target"

    def test_bad_symbol_import_is_test_bug(self):
        # ImportError: cannot import name 'ContactModelz' from 'server.app.models.contact'
        d = _diag(
            "tests/unit/test_contact_model.py", "ImportError",
            "cannot import name 'ContactModelz' from "
            "'server.app.models.contact'",
        )
        r = classify_diagnostic(d)
        assert r.fclass is FailureClass.TEST_BUG
        assert r.fingerprint == "test-bad-symbol-import"
        assert "ContactModelz" in r.reason

    def test_same_typeerror_in_source_is_code_gap(self):
        # The row-subscript message in a SOURCE file is not a test-authoring
        # bug — repair owns it.
        d = _diag(
            "server/app/database.py", "TypeError",
            "tuple indices must be integers or slices, not str",
        )
        assert classify_diagnostic(d).fclass is FailureClass.CODE_GAP


class TestConservativeDefaults:
    def test_did_not_raise_defaults_to_code_gap(self):
        # tests/unit/test_contact_service.py:0:0 - Failed: DID NOT RAISE HTTPException
        # Ambiguous — often a DOWNSTREAM symptom of a real code bug. Repair.
        d = _diag(
            "tests/unit/test_contact_service.py", "Failed",
            "DID NOT RAISE HTTPException",
        )
        assert classify_diagnostic(d).fclass is FailureClass.CODE_GAP

    def test_behaviour_assertion_in_test_is_code_gap(self):
        # server/tests/test_contact_service.py:185 - AssertionError: assert 500 == 404
        d = _diag(
            "server/tests/test_contact_service.py", "AssertionError",
            "assert 500 == 404",
        )
        assert classify_diagnostic(d).fclass is FailureClass.CODE_GAP

    def test_real_code_bug_in_source_is_code_gap(self):
        # server/app/database.py:36 - TypeError: 'coroutine' object does not support ...
        d = _diag(
            "server/app/database.py", "TypeError",
            "'coroutine' object does not support the asynchronous context "
            "manager protocol",
        )
        assert classify_diagnostic(d).fclass is FailureClass.CODE_GAP

    def test_missing_module_import_stays_code_gap(self):
        # 'No module named X' is a missing module/dependency (repair/env owns
        # it), NOT a test-authoring bug — must not be diverted to regeneration.
        d = _diag("tests/unit/test_x.py", "ModuleNotFoundError",
                  "No module named 'aiosqlite'")
        assert classify_diagnostic(d).fclass is FailureClass.CODE_GAP

    def test_asserted_but_never_raised_stays_code_gap(self):
        # DID NOT RAISE is fundamentally ambiguous — "test invented a bogus
        # contract" vs "code should raise but doesn't (a real gap)" are
        # indistinguishable from the failure alone. Conservative default holds.
        d = _diag("tests/unit/test_contact_service.py", "Failed",
                  "DID NOT RAISE ValueError")
        assert classify_diagnostic(d).fclass is FailureClass.CODE_GAP

    def test_unknown_test_exception_defaults_to_code_gap(self):
        d = _diag("tests/unit/test_x.py", "KeyError", "'days_until_next'")
        assert classify_diagnostic(d).fclass is FailureClass.CODE_GAP

    def test_empty_file_is_code_gap(self):
        assert classify_diagnostic(_diag("", "NameError",
                                         "name 'x' is not defined")).fclass \
            is FailureClass.CODE_GAP


class TestOverrideAndDictInput:
    def test_is_test_file_override_forces_test_frame(self):
        # A co-located/oddly-named test the predicate might miss: caller can
        # assert it IS a test frame.
        d = _diag("weird/path.py", "NameError", "name 'patch' is not defined")
        r = classify_diagnostic(d, is_test_file=True)
        assert r.fclass is FailureClass.TEST_BUG

    def test_strips_test_failure_prefix(self):
        # test_generation_node tags routed diagnostics as TEST_FAILURE:<type>;
        # the classifier must still match the underlying fingerprint.
        d = _diag("tests/unit/test_database.py", "TEST_FAILURE:TypeError",
                  "tuple indices must be integers or slices, not str")
        r = classify_diagnostic(d)
        assert r.fclass is FailureClass.TEST_BUG
        assert r.fingerprint == "test-raw-row-subscript"

    def test_accepts_dict_diagnostic(self):
        d = {
            "file": "tests/unit/test_database.py",
            "error_code": "TypeError",
            "message": "tuple indices must be integers or slices, not str",
        }
        assert classify_diagnostic(d).fclass is FailureClass.TEST_BUG


class TestSummary:
    def test_summary_counts_and_fingerprints(self):
        diags = [
            _diag("tests/unit/test_database.py", "TypeError",
                  "tuple indices must be integers or slices, not str"),
            _diag("tests/unit/test_contact_service.py", "NameError",
                  "name 'patch' is not defined"),
            _diag("tests/unit/test_contact_service.py", "Failed",
                  "DID NOT RAISE HTTPException"),
            _diag("server/app/database.py", "TypeError",
                  "'coroutine' object does not support ..."),
        ]
        s = summarize_failures(diags)
        assert s["test_bug"] == 2
        assert s["code_gap"] == 2
        assert s["fingerprints"] == {
            "test-raw-row-subscript": 1,
            "test-undefined-name": 1,
        }
        assert len(s["results"]) == 4


class TestContextFingerprints:
    """lumina session 01a079dc. Two test-authoring defects whose ``message``
    carries no signal at all — the evidence lives entirely in the parser's
    ``semantic_context`` enrichment. Both defaulted to CODE_GAP, and the
    repair loop spent 8 rounds patching production code that was correct.
    """

    CAPLOG_CTX = (
        "failing source: assert len(caplog.records) == 1\n"
        "assertion-rewrite:\n"
        "  assert 2 == 1\n"
        "  +  where 2 = len([<LogRecord: audit, 20, "
        "server/app/middleware/audit.py, 34, \"{...}\">, "
        "<LogRecord: httpx, 20, "
        "/tmp/teane-venv/lib/python3.12/site-packages/httpx/_client.py, "
        "1025, \"HTTP Request: %s %s\">])\n"
    )

    TESTCLIENT_CTX = (
        "failing source: raise RuntimeError(\"boom\")\n"
        "workspace call chain (outermost → innermost):\n"
        "  server/tests/test_main.py:25 in boom\n"
        "  server/app/middleware/audit.py:21 in dispatch\n"
        "locals at failure frame:\n"
        "  __class__ = <class 'starlette.testclient.TestClient'>\n"
        "  client = <starlette.testclient.TestClient object at 0x7d80366415e0>\n"
        "  url = '/boom'\n"
    )

    def test_caplog_third_party_record_is_test_bug(self):
        d = _diag(
            "server/tests/test_middleware.py", "AssertionError",
            "AssertionError: assert 2 == 1",
            semantic_context=self.CAPLOG_CTX,
        )
        r = classify_diagnostic(d)
        assert r.fclass is FailureClass.TEST_BUG
        assert r.fingerprint == "test-caplog-third-party-records"
        assert r.confidence == "high"
        assert "caplog" in r.reason

    def test_caplog_over_own_records_stays_code_gap(self):
        """An unfiltered caplog assertion that counted only the app's OWN
        records is a genuine behaviour assertion — the app may really be
        logging the wrong number of times. Both halves of the fingerprint
        are required."""
        ctx = (
            "failing source: assert len(caplog.records) == 2\n"
            "assertion-rewrite:\n"
            "  assert 1 == 2\n"
            "  +  where 1 = len([<LogRecord: audit, 20, "
            "server/app/middleware/audit.py, 34, \"{...}\">])\n"
        )
        d = _diag(
            "server/tests/test_middleware.py", "AssertionError",
            "AssertionError: assert 1 == 2", semantic_context=ctx,
        )
        assert classify_diagnostic(d).fclass is FailureClass.CODE_GAP

    def test_third_party_record_without_caplog_assertion_stays_code_gap(self):
        ctx = (
            "failing source: assert response.status_code == 200\n"
            "locals at failure frame:\n"
            "  rec = <LogRecord: httpx, 20, "
            "/usr/lib/python3/site-packages/httpx/_client.py, 1025, \"x\">\n"
        )
        d = _diag(
            "server/tests/test_api.py", "AssertionError",
            "AssertionError: assert 500 == 200", semantic_context=ctx,
        )
        assert classify_diagnostic(d).fclass is FailureClass.CODE_GAP

    def test_testclient_reraise_is_test_bug(self):
        d = _diag(
            "server/tests/test_main.py", "RuntimeError", "boom",
            semantic_context=self.TESTCLIENT_CTX,
        )
        r = classify_diagnostic(d)
        assert r.fclass is FailureClass.TEST_BUG
        assert r.fingerprint == "test-client-reraises-server-exception"
        assert "raise_server_exceptions" in r.reason

    def test_testclient_assertion_failure_stays_code_gap(self):
        """An AssertionError inside a TestClient-driven test is an ordinary
        behaviour assertion — the app really may be returning the wrong
        status. Only a non-assertion exception propagating out qualifies."""
        d = _diag(
            "server/tests/test_main.py", "AssertionError",
            "AssertionError: assert 404 == 200",
            semantic_context=self.TESTCLIENT_CTX,
        )
        assert classify_diagnostic(d).fclass is FailureClass.CODE_GAP

    def test_production_frame_never_reaches_context_fingerprints(self):
        """Both fingerprints are gated behind the test-frame precondition.
        A production innermost frame — i.e. the exception originated in app
        code — stays a code-gap no matter what the context says."""
        d = _diag(
            "server/app/middleware/audit.py", "RuntimeError", "boom",
            semantic_context=self.TESTCLIENT_CTX,
        )
        assert classify_diagnostic(d).fclass is FailureClass.CODE_GAP

    def test_missing_context_is_harmless(self):
        d = _diag(
            "server/tests/test_middleware.py", "AssertionError",
            "AssertionError: assert 2 == 1",
        )
        assert classify_diagnostic(d).fclass is FailureClass.CODE_GAP
