import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.build_warmstart_refine_depths import build_warmstart_refine_depths


def _save_float_tiff(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.float32), mode="F").save(path)


def test_real_depth_requires_depth_support_and_semantic_keep(tmp_path):
    depth = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
    _save_float_tiff(tmp_path / "depth_frame000000.tiff", depth)
    np.save(tmp_path / "visibility_frame000000.npy", [[1.0, 0.1], [2.0, 2.0]])
    np.save(tmp_path / "semantic_keep_frame000000.npy", [[1, 1], [1, 0]])

    diagnostics = build_warmstart_refine_depths(tmp_path)

    confidence = np.asarray(Image.open(tmp_path / "confident_map_frame000000.png"))
    refined = np.asarray(Image.open(tmp_path / "refine_depth_frame000000.tiff"))
    assert confidence.tolist() == [[255, 0], [0, 0]]
    assert refined.tolist() == [[1.0, 2.0], [0.0, 4.0]]
    assert diagnostics[0].frame_type == "real"
    assert diagnostics[0].final_confident_fraction == 0.25


def test_pseudo_depth_keeps_valid_aligned_candidates_without_real_masks(tmp_path):
    _save_float_tiff(
        tmp_path / "depth_frame000000.tiff",
        np.array([[1.0, 0.0], [3.0, 4.0]], dtype=np.float32),
    )
    (tmp_path / "pseudo.json").write_text(json.dumps([0]))

    diagnostics = build_warmstart_refine_depths(
        tmp_path, pseudo_ids_path=tmp_path / "pseudo.json"
    )

    confidence = np.asarray(Image.open(tmp_path / "confident_map_frame000000.png"))
    assert confidence.tolist() == [[255, 0], [255, 255]]
    assert diagnostics[0].frame_type == "pseudo"
    assert diagnostics[0].final_confident_fraction == 0.75
