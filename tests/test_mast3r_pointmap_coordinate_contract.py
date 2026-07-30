import json

import numpy as np

from scripts.audit_mast3r_pointmaps import assess_coordinate_contract
from scripts.merge_mast3r_fixed_world_pointmaps import (
    _load_base_pointmap_records,
    _pointmap_projection_error,
)


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


def test_merge_validator_checks_points_not_only_camera_matrix(tmp_path):
    path = tmp_path / "frame.json"
    points = np.asarray(
        [
            [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]],
            [[0.0, 1.0, 2.0], [1.0, 1.0, 2.0]],
        ],
        dtype=np.float32,
    )
    path.write_text(
        json.dumps(
            {
                "rgb": None,
                "points": points.tolist(),
                "confs": np.full((2, 2), 2.0).tolist(),
            }
        ),
        encoding="utf-8",
    )
    p50, p90 = _pointmap_projection_error(
        pointmap_path=path,
        camera_to_world=np.eye(4),
        intrinsics=np.asarray([2.0, 2.0, 0.0, 0.0]),
        image_size=(2, 2),
    )
    assert p90 < 1e-6

    shifted = np.eye(4)
    shifted[0, 3] = 5.0
    p50, _ = _pointmap_projection_error(
        pointmap_path=path,
        camera_to_world=shifted,
        intrinsics=np.asarray([2.0, 2.0, 0.0, 0.0]),
        image_size=(2, 2),
    )
    assert p50 > 4.0


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


def test_complementary_merge_preserves_content_addressed_base_pointmaps(
    tmp_path,
):
    pointmap = tmp_path / "frame.json"
    pointmap.write_text("{}", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(pointmap.read_bytes()).hexdigest()
    index = tmp_path / "pointmap_index.json"
    index.write_text(
        json.dumps(
            {
                "coordinate_frame": "cambridge_fixed_world",
                "camera_order": ["frame"],
                "records": {
                    "frame": {
                        "path": str(pointmap),
                        "bytes": pointmap.stat().st_size,
                        "sha256": digest,
                    }
                },
                "source_by_camera": {"frame": "old-producer"},
                "producer_audit": [],
            }
        ),
        encoding="utf-8",
    )
    records, sources, audit = _load_base_pointmap_records(
        {
            "artifacts": [
                {
                    "name": "mast3r_pointmap_index",
                    "path": str(index),
                }
            ]
        }
    )
    assert list(records) == ["frame"]
    assert records["frame"]["sha256"] == digest
    assert sources["frame"] == "old-producer"
    assert audit["accepted_unique_pointmaps"] == 1
