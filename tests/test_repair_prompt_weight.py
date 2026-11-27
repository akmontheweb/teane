"""The repair prompt paid full price for things repair could not use.

Measured across all 314 repair dispatches of lumina 01a21929 (run 20):

    median prompt   68,252 tokens
    median cached   54,656  (80%)
    median BILLED   12,840
    median output       84
    repair spend    $3.52 of the run's $4.95

Because 80% of the prompt is a cache hit, the big static prefix is nearly
free and the PER-ROUND tail is where the money goes. Two things in that tail
were pure waste:

1. **The guarded-test bodies.** ``test_birthday_contract.py`` went in at
   11,445 chars — 236 line-numbered lines, every passing test included — to
   a model that may not edit it. Two more guarded test files followed. The
   cause was a collision between two rules: the persistent-blocker rule
   promotes any judge-named file to a full read (3,000 lines / 100 KB) so the
   LLM can compose a ``REPLACE_BLOCK`` spanning the failing line, and the
   judge names the guarded test on precisely the rounds where it IS the
   blocker. Right for a production file; pure cost for one nobody may patch.

2. **The structured diagnostic payload.** 6,054 chars of JSON restating the
   markdown diagnostics, its own preamble conceding "Both views show the same
   underlying data".

Repair's revealed preference argues the same way: of 197 ``READ_FILE`` pulls
it asked for production code (``api/birthdays.py`` 68, ``birthday_service.py``
54, ``birthday_repository.py`` 38) and for a guarded test only 16 times.

:class:`TestTheFailingLineSurvives` is the test that matters. Shrinking a view
until the model can no longer see what failed would trade money for rounds,
and a repair round costs ~18k effective tokens — about five rounds' worth of
the savings here.
"""

from __future__ import annotations

import pathlib

import pytest

import harness.graph as g


def _write(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p


def _late_failing_test(tmp_path, fillers=240):
    """A guarded test whose failing assertion sits near the END — the shape
    the full-read promotion was invented for (session badv74q2n stalled four
    rounds because truncation hid ``edgar.py:172``)."""
    body = ["import pytest", "from app.m import M", ""]
    for i in range(fillers):
        body += [f"def test_filler_{i}():", f"    assert M(x={i}) is not None", ""]
    body += ["def test_the_failing_one():", "    assert M(x='bad') == 1", ""]
    p = _write(tmp_path, "test_late.py", body)
    # 1-indexed line of the assert: it is the second-to-last element, and
    # line k is body[k - 1].
    return p, len(body) - 1


class TestTheFailingLineSurvives:
    def test_a_late_failing_line_stays_visible(self, tmp_path):
        """Plain truncation hides it; the window must not."""
        p, fail = _late_failing_test(tmp_path)
        default, _ = g._render_file_with_line_numbers_and_hash(str(p))
        windowed, _ = g._render_file_windows_with_line_numbers_and_hash(
            str(p), {fail}, head_lines=40, radius=25,
        )
        assert f"{fail}|" not in default, "baseline assumption broke"
        assert f"{fail}|" in windowed

    def test_it_is_smaller_than_both_alternatives(self, tmp_path):
        p, fail = _late_failing_test(tmp_path)
        promoted, _ = g._render_file_with_line_numbers_and_hash(
            str(p),
            max_lines=g._PERSISTENT_BLOCKER_FILE_LINE_CAP,
            max_chars=g._PERSISTENT_BLOCKER_FILE_CHAR_CAP,
        )
        default, _ = g._render_file_with_line_numbers_and_hash(str(p))
        windowed, _ = g._render_file_windows_with_line_numbers_and_hash(
            str(p), {fail}, head_lines=40, radius=25,
        )
        assert len(windowed) < len(default) < len(promoted)

    def test_line_numbers_are_the_files_own(self, tmp_path):
        """A diagnostic says ``:114``; the view has to agree with it.
        Renumbering a windowed view is how you get a REPLACE_BLOCK anchored
        at the wrong place."""
        p, fail = _late_failing_test(tmp_path)
        windowed, _ = g._render_file_windows_with_line_numbers_and_hash(
            str(p), {fail}, head_lines=40, radius=25,
        )
        line = next(
            ln for ln in windowed.splitlines()
            if ln.strip().startswith(f"{fail}|")
        )
        assert "assert M(x='bad') == 1" in line

    def test_the_head_is_kept_so_imports_are_readable(self, tmp_path):
        p, fail = _late_failing_test(tmp_path)
        windowed, _ = g._render_file_windows_with_line_numbers_and_hash(
            str(p), {fail}, head_lines=40, radius=25,
        )
        assert "import pytest" in windowed
        assert "from app.m import M" in windowed

    def test_elisions_are_announced_and_name_the_escape(self, tmp_path):
        """A silently shortened file is worse than a long one: the model
        cannot know to ask. Every gap says READ_FILE."""
        p, fail = _late_failing_test(tmp_path)
        windowed, _ = g._render_file_windows_with_line_numbers_and_hash(
            str(p), {fail}, head_lines=40, radius=25,
        )
        gaps = [ln for ln in windowed.splitlines() if "elided" in ln]
        assert gaps
        assert all("READ_FILE" in ln for ln in gaps)

    def test_every_failing_line_gets_its_own_window(self, tmp_path):
        p, fail = _late_failing_test(tmp_path)
        windowed, _ = g._render_file_windows_with_line_numbers_and_hash(
            str(p), {100, fail}, head_lines=10, radius=3,
        )
        assert "100|" in windowed and f"{fail}|" in windowed


class TestItDegradesToTodaysBehaviour:
    def test_no_lines_of_interest_falls_back(self, tmp_path):
        p, _ = _late_failing_test(tmp_path)
        windowed, _ = g._render_file_windows_with_line_numbers_and_hash(
            str(p), set(), head_lines=40, radius=25,
        )
        default, _ = g._render_file_with_line_numbers_and_hash(str(p))
        assert windowed == default

    def test_zero_bounds_fall_back(self, tmp_path):
        p, fail = _late_failing_test(tmp_path)
        windowed, _ = g._render_file_windows_with_line_numbers_and_hash(
            str(p), {fail}, head_lines=0, radius=0,
        )
        default, _ = g._render_file_with_line_numbers_and_hash(str(p))
        assert windowed == default

    def test_a_small_file_is_rendered_whole_and_identically(self, tmp_path):
        """When the window would elide nothing, the output must be
        byte-identical to the unwindowed path — no stray markers."""
        p = _write(tmp_path, "test_small.py", [
            "import pytest", "", "def test_x():", "    assert 1 == 2",
        ])
        windowed, _ = g._render_file_windows_with_line_numbers_and_hash(
            str(p), {4}, head_lines=40, radius=25,
        )
        default, _ = g._render_file_with_line_numbers_and_hash(str(p))
        assert windowed == default
        assert "elided" not in windowed

    def test_a_missing_file_returns_nothing(self, tmp_path):
        out, h = g._render_file_windows_with_line_numbers_and_hash(
            str(tmp_path / "nope.py"), {3}, head_lines=5, radius=2,
        )
        assert out is None and h is None

    def test_the_hash_is_of_the_whole_file_not_the_view(self, tmp_path):
        """The B5 drift detector compares this hash against disk, so a
        windowed view must not change it."""
        p, fail = _late_failing_test(tmp_path)
        _, full_hash = g._render_file_with_line_numbers_and_hash(
            str(p),
            max_lines=g._PERSISTENT_BLOCKER_FILE_LINE_CAP,
            max_chars=g._PERSISTENT_BLOCKER_FILE_CHAR_CAP,
        )
        _, win_hash = g._render_file_windows_with_line_numbers_and_hash(
            str(p), {fail}, head_lines=40, radius=25,
        )
        assert win_hash == full_hash


class TestTheCollectorHonoursWindows:
    def test_a_window_beats_a_full_read_for_the_same_path(self, tmp_path):
        """The two rules ask for opposite things about a file nobody may
        patch, and the cheap one has to win."""
        p, fail = _late_failing_test(tmp_path)
        rel = p.name
        both = g._collect_workspace_file_content(
            str(tmp_path), [rel],
            full_read_paths={rel},
            window_paths={rel: {fail}},
            window_head_lines=40, window_radius=25,
        )
        full_only = g._collect_workspace_file_content(
            str(tmp_path), [rel], full_read_paths={rel},
        )
        assert len(both[0][1]) < len(full_only[0][1])
        assert f"{fail}|" in both[0][1]

    def test_an_unwindowed_path_is_untouched(self, tmp_path):
        p, fail = _late_failing_test(tmp_path)
        other = _write(tmp_path, "prod.py", ["x = 1", "y = 2"])
        pairs = dict(g._collect_workspace_file_content(
            str(tmp_path), [p.name, other.name],
            window_paths={p.name: {fail}},
            window_head_lines=40, window_radius=25,
        ))
        assert "elided" in pairs[p.name]
        assert "elided" not in pairs[other.name]

    def test_hashes_are_still_recorded_for_windowed_files(self, tmp_path):
        p, fail = _late_failing_test(tmp_path)
        seen: dict[str, str] = {}
        g._collect_workspace_file_content(
            str(tmp_path), [p.name],
            record_hashes_into=seen,
            window_paths={p.name: {fail}},
            window_head_lines=40, window_radius=25,
        )
        assert p.name in seen and len(seen[p.name]) == 64


class TestTheRepairNodeWiring:
    """Source-level, following the precedent in
    ``test_correction_round_bookkeeping.py``: the repair preflight is not
    separately drivable, and a silently unwired window would leave every
    unit above green while the prompt went back to full reads."""

    def _src(self):
        return (
            pathlib.Path(g.__file__).read_text(encoding="utf-8")
        )

    def test_the_preflight_call_passes_windows(self):
        src = self._src()
        call = src.index("preflight_pairs = _collect_workspace_file_content(")
        window = src.index("window_paths=_guarded_windows", call)
        close = src.index("full_read_paths=_persistent_files,", call)
        assert window < close + 200

    def test_a_windowed_file_is_removed_from_the_full_read_set(self):
        """Both rules firing on one path is the bug; the subtraction is what
        stops the expensive one winning."""
        assert "_persistent_files -= set(_guarded_windows)" in self._src()

    def test_only_test_artifacts_are_windowed(self):
        """A production file must never be windowed — repair has to compose
        a REPLACE_BLOCK against it."""
        src = self._src()
        block = src[src.index("_guarded_windows: dict[str, set[int]] = {}"):]
        block = block[:block.index("_persistent_files -= set(")]
        assert "_is_test_artifact(_norm_w)" in block
        assert "_window_carveout" in block


class TestTheConfigIsWired:
    def test_the_keys_are_known_and_typed(self):
        from harness import cli
        for key in (
            "structured_diagnostic_payload",
            "guarded_test_window_head_lines",
            "guarded_test_window_radius",
        ):
            assert key in cli._KNOWN_NESTED_KEYS["repair"], key
        assert cli._TYPE_SCHEMA["repair.structured_diagnostic_payload"] == (bool,)
        assert cli._TYPE_SCHEMA["repair.guarded_test_window_head_lines"] == (int,)
        assert cli._TYPE_SCHEMA["repair.guarded_test_window_radius"] == (int,)

    def test_the_duplicate_payload_ships_off(self):
        import json
        root = pathlib.Path(__file__).resolve().parents[1]
        cfg = json.loads((root / "config" / "config.json").read_text())
        assert cfg["repair"]["structured_diagnostic_payload"] is False

    def test_the_window_ships_enabled(self):
        import json
        root = pathlib.Path(__file__).resolve().parents[1]
        cfg = json.loads((root / "config" / "config.json").read_text())
        assert cfg["repair"]["guarded_test_window_head_lines"] == 40
        assert cfg["repair"]["guarded_test_window_radius"] == 25

    def test_the_section_is_documented(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        help_md = (root / "config" / "config_help.md").read_text()
        for key in (
            "structured_diagnostic_payload",
            "guarded_test_window_head_lines",
            "guarded_test_window_radius",
        ):
            assert key in help_md, key

    @pytest.mark.parametrize("raw,expect", [
        (40, 40), (0, 0), (-5, 0), (99999, 500), ("nonsense", 40),
    ])
    def test_the_head_bound_is_clamped(self, raw, expect):
        from harness.gateway import create_gateway_from_config
        cfg = {
            "model_routing": {},
            "repair": {"guarded_test_window_head_lines": raw},
        }
        gw = create_gateway_from_config(cfg)
        assert gw.config.repair_guarded_test_window_head_lines == expect
