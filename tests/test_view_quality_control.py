from __future__ import annotations

import numpy as np

from view_quality_control import classify_view_records, normalize_qvec, pose_integrity_metrics


def _record(index: int) -> dict:
    return {
        "image_name": f"frame{index:03d}.png",
        "sequence": "seq1" if index < 10 else "seq2",
        "pose_cluster": index % 2,
        "laplacian_variance": 1.0,
        "track_count": 1000,
        "shared_track_max": 500,
        "track_projection_valid_ratio": 0.98,
        "exposure_warning": 0.1,
        "semantic_warning": 0.0,
        "intrinsics_robust_z": 0.0,
        "gray_mean": 0.5,
        "dark_ratio": 0.0,
        "clipped_ratio": 0.0,
        "contrast_p90": 0.5,
        "thing_invalid_ratio": 0.0,
        "tree_invalid_ratio": 0.0,
        "pose_recoverable": True,
        "pose_normalization_required": False,
        "quaternion_norm_error": 0.0,
    }


def test_pose_integrity_detects_recoverable_nonunit_quaternion():
    qvec = np.array([0.003230, -0.005520, -1.004446, 0.280388])

    metrics = pose_integrity_metrics(qvec)
    normalized = normalize_qvec(qvec)

    assert metrics["pose_recoverable"] is True
    assert metrics["pose_normalization_required"] is True
    assert metrics["quaternion_norm_error"] > 0.04
    assert metrics["rotation_orthogonality_error"] > 0.3
    assert np.isclose(np.linalg.norm(normalized), 1.0)


def test_recoverable_pose_is_kept_for_dense_normalization_but_not_clean_chart_pool():
    records = [_record(index) for index in range(20)]
    records[0]["pose_normalization_required"] = True
    records[0]["quaternion_norm_error"] = 0.04

    classified, summary = classify_view_records(records, max_reject_fraction=0.05)
    by_name = {record["image_name"]: record for record in classified}

    assert by_name["frame000.png"]["status"] == "soft_keep"
    assert "pose_requires_normalization" in by_name["frame000.png"]["severe_reasons"]
    assert summary["pose_normalization_required"] == 1


def test_unrecoverable_pose_bypasses_reject_quota_and_coverage_guard():
    records = [_record(index) for index in range(20)]
    records[0]["pose_recoverable"] = False
    records[0]["quaternion_norm_error"] = float("inf")

    classified, summary = classify_view_records(records, max_reject_fraction=0.0)
    by_name = {record["image_name"]: record for record in classified}

    assert by_name["frame000.png"]["status"] == "hard_reject"
    assert summary["hard_invariant_reject"] == 1
