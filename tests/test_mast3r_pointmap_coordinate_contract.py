import numpy as np

from scripts.audit_mast3r_pointmaps import assess_coordinate_contract


def test_coordinate_contract_accepts_camera_consistent_pointmaps():
    result = assess_coordinate_contract(
        np.asarray([0.10, 0.39]),
        np.asarray([0.17, 1.55]),
        all_finite=True,
        max_chart_projection_p50_px=2.0,
        max_chart_projection_p90_px=4.0,
    )

    assert result["passed"]
    assert result["failures"] == []


def test_coordinate_contract_rejects_similarity_mismatch():
    result = assess_coordinate_contract(
        np.asarray([11.0, 363.0]),
        np.asarray([90.0, 696.0]),
        all_finite=True,
        max_chart_projection_p50_px=2.0,
        max_chart_projection_p90_px=4.0,
    )

    assert not result["passed"]
    assert result["failures"] == [
        "chart_projection_p50_exceeds_limit",
        "chart_projection_p90_exceeds_limit",
    ]


def test_sparse_export_contract_uses_export_tolerance_not_raw_pointmap_tolerance():
    result = assess_coordinate_contract(
        np.asarray([0.40, 1.50]),
        np.asarray([1.14, 5.84]),
        all_finite=True,
        max_chart_projection_p50_px=4.0,
        max_chart_projection_p90_px=8.0,
    )

    assert result["passed"]


def test_strict_pose_pipeline_runs_coordinate_audit_before_alignment():
    source = open("train.py", encoding="utf-8").read()
    assert "pointmap_coordinate_audit_command" in source
    assert "--require-coordinate-contract" in source
    assert "--require-sparse-export-coordinate-contract" in source
    assert "--require-sparse-export-track-contract" in source
    assert source.index("run_command_safe(pointmap_coordinate_audit_command)") < source.index(
        "run_command_safe(align_charts_command)"
    )
