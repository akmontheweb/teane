"""ADR-0005 item #4 — the DUPLICATE_TEST_ROOT guard must catch the split tree.

The recurring bug (lumina 019ff418): the same module gets tests under
``tests/unit/``, ``server/tests/`` AND ``tests/integration/`` at once. The
original guard compared byte-identical test-scoped suffixes, so the differing
intermediate tier directory (``unit/`` vs nothing vs ``integration/``) slipped
past it. The guard now compares TIER-NORMALISED suffixes so the split collapses,
while distinctly-named tier tests (ADR-0003 contract tests) stay untouched.
"""

from __future__ import annotations

import os

from harness.patcher import (
    _canonical_test_suffix,
    _extract_test_scoped_suffix,
    _detect_duplicate_test_root,
)


class TestCanonicalTestSuffix:
    def test_strips_leading_tier_subdir(self):
        assert _canonical_test_suffix("tests/unit/test_x.py") == "tests/test_x.py"
        assert _canonical_test_suffix("tests/integration/test_x.py") == \
            "tests/test_x.py"

    def test_no_tier_is_unchanged(self):
        assert _canonical_test_suffix("tests/test_x.py") == "tests/test_x.py"

    def test_only_one_tier_segment_stripped_deeper_pkg_kept(self):
        assert _canonical_test_suffix("tests/unit/pkg/test_x.py") == \
            "tests/pkg/test_x.py"

    def test_non_tier_subdir_kept(self):
        # A real package dir named 'contacts' after tests/ is NOT a tier.
        assert _canonical_test_suffix("tests/contacts/test_x.py") == \
            "tests/contacts/test_x.py"


def _touch(root: str, rel: str) -> None:
    abs_path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w") as f:
        f.write("# @tests: src\ndef test_x():\n    assert True\n")


class TestDetectDuplicateTestRoot:
    def test_exact_suffix_mirror_still_caught(self, tmp_path):
        # The original case: repo-root vs package-nested, identical suffix.
        _touch(str(tmp_path), "server/tests/test_database.py")
        msg = _detect_duplicate_test_root("tests/test_database.py", str(tmp_path))
        assert msg is not None and "DUPLICATE_TEST_ROOT" in msg

    def test_tier_split_now_caught(self, tmp_path):
        # NEW: same module under a package-nested root vs a repo-root TIER dir.
        _touch(str(tmp_path), "server/tests/test_database.py")
        msg = _detect_duplicate_test_root(
            "tests/unit/test_database.py", str(tmp_path),
        )
        assert msg is not None and "server/tests/test_database.py" in msg

    def test_two_tiers_same_root_caught(self, tmp_path):
        _touch(str(tmp_path), "tests/integration/test_database.py")
        msg = _detect_duplicate_test_root(
            "tests/unit/test_database.py", str(tmp_path),
        )
        assert msg is not None

    def test_distinct_contract_basename_not_flagged(self, tmp_path):
        # ADR-0003 contract tests have their OWN basename → not the same module
        # file → must NOT be rejected.
        _touch(str(tmp_path), "tests/contract/test_contact_contract.py")
        msg = _detect_duplicate_test_root(
            "tests/unit/test_contact.py", str(tmp_path),
        )
        assert msg is None

    def test_different_modules_not_flagged(self, tmp_path):
        _touch(str(tmp_path), "server/tests/test_alpha.py")
        msg = _detect_duplicate_test_root(
            "tests/unit/test_beta.py", str(tmp_path),
        )
        assert msg is None

    def test_same_exact_path_not_flagged(self, tmp_path):
        # Editing the file that already exists at that path is not a duplicate.
        _touch(str(tmp_path), "tests/unit/test_database.py")
        msg = _detect_duplicate_test_root(
            "tests/unit/test_database.py", str(tmp_path),
        )
        assert msg is None

    def test_non_test_basename_skipped(self, tmp_path):
        _touch(str(tmp_path), "server/tests/conftest.py")
        # conftest is not a test module basename → guard doesn't apply.
        assert _detect_duplicate_test_root(
            "tests/conftest.py", str(tmp_path),
        ) is None

    def test_colocated_tests_under_different_parents_not_flagged(self, tmp_path):
        # TS colocated fix: src/x.ts and src/hooks/x.ts each have a colocated
        # __tests__/x.test.ts. They share the __tests__/x.test.ts suffix but
        # test DIFFERENT modules — must NOT be flagged as a duplicate tree.
        _touch(str(tmp_path), "client/src/__tests__/x.test.ts")
        msg = _detect_duplicate_test_root(
            "client/src/hooks/__tests__/x.test.ts", str(tmp_path),
        )
        assert msg is None, (
            "colocated __tests__ under different parents test different modules "
            f"and must not be rejected as duplicates; got: {msg}"
        )

    def test_colocated_exact_same_location_not_flagged(self, tmp_path):
        # Editing the file that already exists at the same colocated path is the
        # 'already exists' case, not a cross-root duplicate.
        _touch(str(tmp_path), "client/src/__tests__/x.test.ts")
        assert _detect_duplicate_test_root(
            "client/src/__tests__/x.test.ts", str(tmp_path),
        ) is None
