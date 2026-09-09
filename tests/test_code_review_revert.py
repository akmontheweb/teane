"""Reviewer-regression guard: a code_review re-patch must not be allowed to
break a build that was green moments before.

``code_review_node`` runs ONLY on a green build, so a failure on the compile
immediately after its re-patch is caused by that re-patch — the workspace was
verified seconds earlier. Before this guard the failure went to the repair
loop as if it were ordinary breakage, and the loop spent its rounds chasing a
regression the reviewer had introduced.

lumina 01a079dc: a re-patch acting on 20 findings changed
``models/birthday.py``'s column from ``String`` to ``Date`` while every caller
kept passing ISO strings. 20/20 acceptance criteria had just passed; the next
build went red across the repository and service tiers, and repair burned its
rounds re-emitting byte-identical content for that file until it hit a
zero-patch HITL.
"""

from __future__ import annotations

from harness.graph import _revert_code_review_repatch, _snapshot_patch_targets


def _blocks(*files: str) -> str:
    return "\n".join(
        "<<<REPLACE_BLOCK>>>\n"
        f"file: {f}\n"
        "search:\nold\nreplace:\nnew\n"
        "<<<END_REPLACE_BLOCK>>>"
        for f in files
    )


class TestSnapshot:
    def test_captures_current_bytes_of_targeted_files(self, tmp_path):
        (tmp_path / "a.py").write_text("ORIGINAL A\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("ORIGINAL B\n", encoding="utf-8")
        snap = _snapshot_patch_targets(_blocks("a.py", "b.py"), str(tmp_path))
        assert snap == {"a.py": "ORIGINAL A\n", "b.py": "ORIGINAL B\n"}

    def test_missing_file_records_none_so_revert_deletes_it(self, tmp_path):
        snap = _snapshot_patch_targets(_blocks("new.py"), str(tmp_path))
        assert snap == {"new.py": None}

    def test_unparseable_payload_yields_empty_snapshot(self, tmp_path):
        # An empty snapshot means "cannot revert" — the caller proceeds
        # normally, so a parse problem can never block progress.
        assert _snapshot_patch_targets("not a patch payload", str(tmp_path)) == {}

    def test_each_file_snapshotted_once(self, tmp_path):
        (tmp_path / "a.py").write_text("ORIGINAL\n", encoding="utf-8")
        snap = _snapshot_patch_targets(_blocks("a.py", "a.py"), str(tmp_path))
        assert snap == {"a.py": "ORIGINAL\n"}


class TestRevert:
    def test_restores_modified_content(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("ORIGINAL\n", encoding="utf-8")
        snap = _snapshot_patch_targets(_blocks("a.py"), str(tmp_path))
        f.write_text("REVIEWER BROKE IT\n", encoding="utf-8")
        restored = _revert_code_review_repatch(snap, str(tmp_path))
        assert restored == ["a.py"]
        assert f.read_text(encoding="utf-8") == "ORIGINAL\n"

    def test_deletes_a_file_the_repatch_created(self, tmp_path):
        snap = {"created.py": None}
        (tmp_path / "created.py").write_text("new\n", encoding="utf-8")
        restored = _revert_code_review_repatch(snap, str(tmp_path))
        assert restored == ["created.py"]
        assert not (tmp_path / "created.py").exists()

    def test_untouched_file_is_not_reported_as_restored(self, tmp_path):
        # The reviewer may target a file and land nothing on it (a rejected
        # or no-op block). Rewriting identical bytes would be noise.
        f = tmp_path / "a.py"
        f.write_text("ORIGINAL\n", encoding="utf-8")
        snap = _snapshot_patch_targets(_blocks("a.py"), str(tmp_path))
        assert _revert_code_review_repatch(snap, str(tmp_path)) == []
        assert f.read_text(encoding="utf-8") == "ORIGINAL\n"

    def test_unwritable_file_does_not_raise(self, tmp_path):
        # Best-effort per file: one failure must not abort the rest.
        d = tmp_path / "sub"
        d.mkdir()
        (d / "a.py").write_text("ORIGINAL\n", encoding="utf-8")
        snap = {"sub/a.py": "ORIGINAL\n", "sub": "not a file"}
        _revert_code_review_repatch(snap, str(tmp_path))  # must not raise

    def test_round_trip_restores_the_verified_state(self, tmp_path):
        # The shape of the incident: a column type flipped under callers that
        # were green with the old one.
        model = tmp_path / "models.py"
        good = "date_of_birth: Mapped[str] = mapped_column(String(10))\n"
        model.write_text(good, encoding="utf-8")
        snap = _snapshot_patch_targets(_blocks("models.py"), str(tmp_path))
        model.write_text(
            "date_of_birth: Mapped[date] = mapped_column(Date)\n",
            encoding="utf-8",
        )
        _revert_code_review_repatch(snap, str(tmp_path))
        assert model.read_text(encoding="utf-8") == good


class TestWiring:
    def test_snapshot_is_taken_before_the_patch_is_applied(self):
        src = open("harness/graph.py", encoding="utf-8").read()
        snap_at = src.index("pre_patch_snapshot = _snapshot_patch_targets")
        apply_at = src.index("patch_results, new_modified_files = await process_llm_patch_output")
        assert snap_at < apply_at, "snapshot must precede the patch application"

    def test_snapshot_is_cleared_after_one_reverify(self):
        # Valid for exactly the build that follows the re-patch; replaying it
        # later would silently undo real repair work.
        src = open("harness/graph.py", encoding="utf-8").read()
        assert 'node_state.pop("code_review_revert_snapshot", None)' in src

    def test_snapshot_only_stored_when_the_repatch_really_landed(self):
        src = open("harness/graph.py", encoding="utf-8").read()
        assert "pre_patch_snapshot if repatched else {}" in src
