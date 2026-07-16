"""Co-located test-file protection (harness/graph.py::_is_test_artifact).

The patching test-filters and the repair-loop tamper guard used the dir-based
_is_test_path, which missed co-located tests (Button.test.tsx, foo_test.py next
to the source — the norm in React/TS). They now use the union _is_test_artifact
(_is_test_path OR _is_test_file), so both patching and repair protect
co-located tests. These tests lock that in and guard against production files
being misclassified.
"""

from __future__ import annotations

import pytest

from harness.graph import _is_test_artifact, _is_test_path, _reject_test_patch_blocks
from harness.patcher import OperationType


class _Block:
    def __init__(self, file, operation):
        self.file = file
        self.operation = operation


class TestPredicate:
    @pytest.mark.parametrize("path", [
        "src/components/Button.test.tsx",
        "app/foo_test.py",
        "pkg/service.spec.ts",
        "a/b/thing_test.go",
        "web/src/Widget.spec.jsx",
    ])
    def test_colocated_tests_now_caught(self, path):
        # These were missed by the old dir-based predicate.
        assert _is_test_artifact(path) is True
        assert _is_test_path(path) is False

    @pytest.mark.parametrize("path", [
        "tests/test_x.py", "test/x.py", "__tests__/a.js", "conftest.py", "pytest.ini",
    ])
    def test_dir_based_still_caught(self, path):
        assert _is_test_artifact(path) is True

    @pytest.mark.parametrize("path", [
        "src/app/service.py", "lib/util.ts", "components/Button.tsx", "main.py",
    ])
    def test_production_not_flagged(self, path):
        assert _is_test_artifact(path) is False


class TestGuardCatchesColocated:
    def test_repair_guard_rejects_colocated_test_edit(self):
        kept, rejections = _reject_test_patch_blocks([
            _Block("src/components/Button.test.tsx", OperationType.DELETE_BLOCK),
            _Block("src/app/service.py", OperationType.REPLACE_BLOCK),
        ])
        assert [b.file for b in kept] == ["src/app/service.py"]
        assert len(rejections) == 1
        assert rejections[0].file == "src/components/Button.test.tsx"
        assert "[test-protected]" in (rejections[0].error or "")
