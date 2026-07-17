"""Reflection-judge calibration.

repair_node emits one ``judge_calibration`` event per reflected round,
pairing the judge's verdict with the deterministic fingerprint-survival
outcome for the SAME round; ``teane metrics`` aggregates them into a
confusion matrix + precision/recall. Ground truth costs nothing — every
long run doubles as a calibration dataset.
"""

from __future__ import annotations

import json
import os

from harness.graph import _judge_calibration_cell
from harness.metrics import SessionMetrics, aggregate_session, format_human


def _verdict(v: str, blocker: str = "the real blocker is X"):
    return {"verdict": v, "real_blocker": blocker, "recommendation": "do Y"}


class TestCellClassification:
    def test_true_positive(self):
        assert _judge_calibration_cell(_verdict("PROGRESS"), True) == "tp"

    def test_false_positive_is_the_expensive_error(self):
        assert _judge_calibration_cell(_verdict("PROGRESS"), False) == "fp"

    def test_true_negative(self):
        assert _judge_calibration_cell(_verdict("DISTRACTION"), False) == "tn"

    def test_false_negative(self):
        assert _judge_calibration_cell(_verdict("REGRESSION"), True) == "fn"

    def test_low_signal_abstains_regardless_of_outcome(self):
        v = _verdict("PROGRESS", blocker="insufficient data — no diagnostics")
        assert _judge_calibration_cell(v, True) == "low_signal"
        assert _judge_calibration_cell(v, False) == "low_signal"

    def test_verdict_case_insensitive(self):
        assert _judge_calibration_cell(_verdict("progress"), True) == "tp"

    def test_working_hypothesis_abstains_regardless_of_outcome(self):
        # Regression: WH was bucketed with DISTRACTION/REGRESSION, so a
        # judge that correctly deferred on a factually-advancing round
        # recorded an fn — corrupting the recall the metric isolates.
        v = _verdict("WORKING_HYPOTHESIS")
        assert _judge_calibration_cell(v, True) == "working_hypothesis"
        assert _judge_calibration_cell(v, False) == "working_hypothesis"


class TestMetricsMath:
    def _m(self, **cells):
        m = SessionMetrics(session_id="s")
        m.judge_confusion = dict(cells)
        return m

    def test_precision_recall_accuracy(self):
        m = self._m(tp=6, fp=2, tn=3, fn=1, low_signal=2)
        assert m.judge_precision() == 6 / 8
        assert m.judge_recall() == 6 / 7
        assert m.judge_accuracy() == 9 / 12
        assert m.judge_low_signal_rate() == 2 / 14

    def test_empty_matrix_returns_none(self):
        m = self._m()
        assert m.judge_precision() is None
        assert m.judge_recall() is None
        assert m.judge_accuracy() is None
        assert m.judge_low_signal_rate() is None

    def test_only_abstentions(self):
        m = self._m(low_signal=5)
        assert m.judge_precision() is None
        assert m.judge_low_signal_rate() == 1.0

    def test_working_hypothesis_excluded_from_matrix_math(self):
        m = self._m(tp=6, fp=2, tn=3, fn=1, working_hypothesis=4)
        # Precision/recall/accuracy unchanged by deferrals...
        assert m.judge_precision() == 6 / 8
        assert m.judge_recall() == 6 / 7
        assert m.judge_accuracy() == 9 / 12
        # ...but the deferral rate is tracked over all labeled verdicts.
        assert m.judge_working_hypothesis_rate() == 4 / 16

    def test_jsonable_includes_calibration(self):
        d = self._m(tp=1, fp=1).to_jsonable()
        assert d["judge_confusion"] == {"tp": 1, "fp": 1}
        assert d["judge_precision"] == 0.5


class TestAggregation:
    def _write_log(self, tmp_path, session_id, records):
        log_dir = str(tmp_path)
        path = os.path.join(log_dir, f"{session_id}.jsonl")
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return log_dir

    def test_events_accumulate_into_confusion_matrix(self, tmp_path):
        sid = "cal-test"
        recs = (
            [{"event": "judge_calibration", "cell": "tp",
              "ts": "2026-07-16T01:00:00+00:00"}] * 3
            + [{"event": "judge_calibration", "cell": "fp",
                "ts": "2026-07-16T01:01:00+00:00"}] * 2
            + [{"event": "judge_calibration", "cell": "low_signal",
                "ts": "2026-07-16T01:02:00+00:00"}]
            + [{"event": "judge_calibration", "cell": "bogus"}]  # ignored
        )
        log_dir = self._write_log(tmp_path, sid, recs)
        m = aggregate_session(sid, log_dir)
        assert m.judge_confusion == {"tp": 3, "fp": 2, "low_signal": 1}
        assert m.judge_precision() == 0.6

    def test_format_human_renders_calibration_line(self, tmp_path):
        sid = "cal-fmt"
        log_dir = self._write_log(tmp_path, sid, [
            {"event": "judge_calibration", "cell": "tp"},
            {"event": "judge_calibration", "cell": "fn"},
        ])
        out = format_human(aggregate_session(sid, log_dir), hard_cap_usd=2.0)
        assert "Judge calibration:" in out
        assert "recall=50%" in out
        assert "tp=1" in out and "fn=1" in out

    def test_no_events_no_line(self, tmp_path):
        sid = "cal-none"
        log_dir = self._write_log(tmp_path, sid, [])
        out = format_human(aggregate_session(sid, log_dir), hard_cap_usd=2.0)
        assert "Judge calibration:" not in out
