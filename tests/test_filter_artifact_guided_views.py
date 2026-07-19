import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.filter_artifact_guided_views import filter_artifact_guided_views


def _build_stage(root: Path, classification: str) -> Path:
    stage = root / "stage1"
    select = stage / "select-gs"
    merged = stage / "select-gs-inpainted-merged"
    planes = stage / "select-gs-planes"
    for path in (select, merged, planes):
        path.mkdir(parents=True)

    height, width = 60, 80
    yy, xx = np.mgrid[:height, :width]
    raw = np.stack([(xx * 7) % 255, (yy * 11) % 255, ((xx + yy) * 5) % 255], axis=2).astype(np.uint8)
    known = np.ones((height, width), dtype=bool)
    known[25:35, 35:45] = False
    generated = raw.copy()
    generated[~known] = np.asarray([240, 20, 180], dtype=np.uint8)
    Image.fromarray(raw).save(select / "ori_warp_frame000000.png")
    Image.fromarray(np.uint8(known) * 255).save(select / "mask_frame000000.png")
    Image.fromarray(generated).save(merged / "predict_warp_frame000000.png")

    baseline_depth = np.full((height, width), 4.0, dtype=np.float32)
    generated_depth = np.full((height, width), 3.0, dtype=np.float32)
    Image.fromarray(baseline_depth, mode="F").save(select / "depth_frame000000.tiff")
    Image.fromarray(generated_depth, mode="F").save(planes / "depth_frame000000.tiff")
    points = np.asarray([[0.0, 0.0, 4.0], [0.2, 0.1, 4.0]], dtype=np.float32)
    points_path = stage / "repair_points_frame000000.npy"
    np.save(points_path, points)

    camera = {
        "name": "pseudo",
        "width": width,
        "height": height,
        "w2c": np.eye(4).tolist(),
        "fx": 60.0,
        "fy": 60.0,
        "cx": 40.0,
        "cy": 30.0,
    }
    (stage / "artifact_guided_manifest.json").write_text(
        json.dumps(
            {
                "pseudo_views": [
                    {
                        "pseudo_view_index": 0,
                        "classification": classification,
                        "pseudo_camera": camera,
                        "repair_points_path": str(points_path),
                    }
                ]
            }
        )
    )
    (planes / "depth_alignment_diagnostics.json").write_text(
        json.dumps(
            [
                {
                    "frame": 0,
                    "accepted": True,
                    "relative_rmse": 0.05,
                }
            ]
        )
    )
    return stage


def test_static_patch_uses_supported_baseline_depth(tmp_path):
    stage = _build_stage(tmp_path, "supported_static_patch")
    accepted, policies = filter_artifact_guided_views(
        stage,
        min_visible_sharpness=0.0,
        min_generated_hole_sharpness=0.0,
        apply_depth_policy=True,
    )

    assert accepted == [0]
    assert policies[0].training_mode == "rgb_only_baseline_depth"
    assert policies[0].baseline_depth_median_relative_error == 0.0
    depth = np.asarray(Image.open(stage / "select-gs-planes/depth_frame000000.tiff"))
    assert np.allclose(depth, 4.0)


def test_wrong_geometry_rejects_depth_that_is_worse_than_baseline(tmp_path):
    stage = _build_stage(tmp_path, "wrong_geometry")
    accepted, policies = filter_artifact_guided_views(
        stage,
        min_visible_sharpness=0.0,
        min_generated_hole_sharpness=0.0,
    )

    assert accepted == []
    assert "generated_depth_not_supported" in policies[0].reasons
    assert "generated_depth_does_not_improve_baseline" in policies[0].reasons
