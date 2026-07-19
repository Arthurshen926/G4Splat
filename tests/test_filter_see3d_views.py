import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts.filter_see3d_views import filter_see3d_views


def _write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(path)


def _fixture(tmp_path: Path) -> Path:
    stage = tmp_path / "stage1"
    (stage / "select-gs").mkdir(parents=True)
    (stage / "select-gs-inpainted-merged").mkdir()
    (stage / "select-gs-planes").mkdir()
    np.savez(stage / "stage1_see3d_cameras.npz", n_views=2)
    checker = np.indices((64, 64)).sum(axis=0) % 2
    sharp = np.repeat((checker * 255)[..., None], 3, axis=2).astype(np.uint8)
    blurry = cv2.GaussianBlur(sharp, (31, 31), 0)
    for index, image in enumerate((sharp, blurry)):
        _write_rgb(stage / "select-gs" / f"ori_warp_frame{index:06d}.png", image)
        _write_rgb(
            stage
            / "select-gs-inpainted-merged"
            / f"predict_warp_frame{index:06d}.png",
            image,
        )
        Image.fromarray(np.full((64, 64), 255, dtype=np.uint8), mode="L").save(
            stage / "select-gs" / f"mask_frame{index:06d}.png"
        )
    diagnostics = {
        "frames": [
            {
                "frame": index,
                "accepted": True,
                "relative_rmse": 0.05,
                "support_pixels": 30_000,
            }
            for index in range(2)
        ]
    }
    (stage / "select-gs-planes" / "depth_alignment_diagnostics.json").write_text(
        json.dumps(diagnostics)
    )
    return stage


def test_quality_filter_rejects_a_blurry_baseline_pseudo_view(tmp_path):
    stage = _fixture(tmp_path)

    accepted, records = filter_see3d_views(
        stage, min_visible_sharpness=100.0, min_alignment_support_pixels=1
    )

    assert accepted == [0]
    assert records[0].accepted is True
    assert records[1].reasons == ["blurry_or_smeared_baseline_render"]
    assert json.loads((stage / "accepted_view_indices.json").read_text()) == [0]
