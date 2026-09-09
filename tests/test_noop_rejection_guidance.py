"""The no-op rejection messages must not lead with "look somewhere else".

Both no-op rejections used to open with "This file is already correct — the
bug is somewhere ELSE", then offer the mechanical cause second. That ordering
presents a GUESS as the first hypothesis, and it is wrong precisely when the
file the model is targeting is itself the regression.

lumina 01a079dc: code_review changed ``models/birthday.py``'s column type and
broke a suite that had just gone 20/20 green. The repair loop targeted that
exact file twice — a search miss, then an identity REWRITE_FILE — and both
times the rejection told it the file was probably fine and the bug was
elsewhere. It was not: the file WAS the regression. Two rounds, then a
zero-patch HITL.

The messages now order causes by how checkable they are: the mechanical
mistake first, "this file is the regression, change it BACK" second, and the
"bug is elsewhere" guess last and explicitly conditional.
"""

from __future__ import annotations

import asyncio

from harness.patcher import TextPatcher


def _rewrite_noop_error(tmp_path) -> str:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    engine = TextPatcher(str(tmp_path))
    # rewrite_file appends the trailing newline itself.
    result = asyncio.run(engine.rewrite_file("a.py", content="x = 1"))
    assert result.success is False and result.no_op is True
    return result.error or ""


def _replace_noop_error(tmp_path) -> str:
    (tmp_path / "b.py").write_text("x = 1\n", encoding="utf-8")
    engine = TextPatcher(str(tmp_path))
    result = asyncio.run(
        engine.replace_block("b.py", search="x = 1\n", replace="x = 1\n")
    )
    assert result.success is False and result.no_op is True
    return result.error or ""


def _order(msg: str) -> tuple[int, int]:
    """(position of the 'this file is the regression' cause,
        position of the 'bug is elsewhere' guess)."""
    return msg.index("is itself the regression"), msg.index("somewhere ELSE")


class TestRewriteFileNoOp:
    def test_regression_cause_precedes_the_look_elsewhere_guess(self, tmp_path):
        regression_at, elsewhere_at = _order(_rewrite_noop_error(tmp_path))
        assert regression_at < elsewhere_at

    def test_names_the_change_it_back_remedy(self, tmp_path):
        msg = _rewrite_noop_error(tmp_path)
        assert "change it BACK" in msg
        assert "modification history" in msg

    def test_look_elsewhere_is_explicitly_conditional(self, tmp_path):
        # It must read as a last resort, not as the leading diagnosis.
        msg = _rewrite_noop_error(tmp_path)
        assert "Only if neither holds" in msg

    def test_still_forbids_re_emitting_identical_content(self, tmp_path):
        msg = _rewrite_noop_error(tmp_path)
        assert "Do NOT emit REWRITE_FILE again" in msg


class TestReplaceBlockNoOp:
    def test_regression_cause_precedes_the_look_elsewhere_guess(self, tmp_path):
        regression_at, elsewhere_at = _order(_replace_noop_error(tmp_path))
        assert regression_at < elsewhere_at

    def test_mechanical_cause_comes_first(self, tmp_path):
        # Copying the original into `replace` is directly checkable, so it
        # leads.
        msg = _replace_noop_error(tmp_path)
        assert msg.index("copied the original text") < msg.index(
            "is itself the regression"
        )

    def test_names_the_change_it_back_remedy(self, tmp_path):
        msg = _replace_noop_error(tmp_path)
        assert "change it BACK" in msg
        assert "modification history" in msg

    def test_still_forbids_re_emitting_the_same_block(self, tmp_path):
        msg = _replace_noop_error(tmp_path)
        assert "Do NOT re-emit this block unchanged" in msg
