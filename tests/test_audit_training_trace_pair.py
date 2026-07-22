import json

import pytest

from scripts.audit_training_trace_pair import build_training_trace_pair_audit


def _row(iteration, *, loss=0.5, camera="view"):
    return {
        "iteration": iteration,
        "rgb_l1": loss,
        "rgb_loss": loss,
        "normal_loss": 0.0,
        "distortion_loss": 0.0,
        "total_loss": loss,
        "ema_total_loss": loss,
        "gaussians": 10,
        "camera": camera,
        "elapsed_sec": float(iteration),
    }


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_trace_pair_ignores_wall_clock_and_reports_no_numeric_drift(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write(baseline, [_row(1), _row(100, loss=0.2)])
    rows = [_row(1), _row(100, loss=0.2)]
    rows[0]["elapsed_sec"] = 99.0
    _write(candidate, rows)

    report = build_training_trace_pair_audit(baseline, candidate)

    assert report["passed"]
    assert report["identity_mismatch_count"] == 0
    assert report["numeric_max_abs_delta"]["rgb_l1"] == 0.0


def test_trace_pair_reports_optimization_drift_without_false_identity_failure(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write(baseline, [_row(1)])
    _write(candidate, [_row(1, loss=0.7)])

    report = build_training_trace_pair_audit(baseline, candidate)

    assert report["passed"]
    assert report["identity_mismatch_count"] == 0
    assert report["numeric_max_abs_delta"]["rgb_l1"] == pytest.approx(0.2)


def test_trace_pair_fails_if_the_logged_camera_stream_changes(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write(baseline, [_row(1, camera="first")])
    _write(candidate, [_row(1, camera="second")])

    report = build_training_trace_pair_audit(baseline, candidate)

    assert not report["passed"]
    assert report["identity_mismatches"] == [
        {
            "iteration": 1,
            "field": "camera",
            "baseline": "first",
            "candidate": "second",
        }
    ]
