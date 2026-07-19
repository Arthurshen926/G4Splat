from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from scripts.constrain_warmstart_ply import constrain_warmstart_ply


DTYPE = [
    ("x", "f4"), ("y", "f4"), ("z", "f4"),
    ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ("f_rest_0", "f4"), ("opacity", "f4"),
    ("scale_0", "f4"), ("scale_1", "f4"), ("mip_filter", "f4"),
]


def _write(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(values, "vertex")], text=False).write(path)


def test_constrain_ply_restores_prefix_and_caps_only_suffix(tmp_path):
    baseline = np.zeros(1, dtype=DTYPE)
    baseline["x"] = 7.0
    baseline["mip_filter"] = 0.25
    candidate = np.zeros(2, dtype=DTYPE)
    candidate[0]["x"] = -2.0
    candidate[0]["mip_filter"] = 0.5
    candidate[1]["opacity"] = 10.0
    candidate[1]["scale_0"] = np.log(2.0)
    candidate[1]["scale_1"] = np.log(3.0)
    candidate[1]["f_dc_0"] = 20.0
    candidate[1]["f_rest_0"] = 4.0
    baseline_path = tmp_path / "baseline.ply"
    candidate_path = tmp_path / "candidate.ply"
    output_path = tmp_path / "output.ply"
    _write(baseline_path, baseline)
    _write(candidate_path, candidate)

    report = constrain_warmstart_ply(
        baseline_path, candidate_path, output_path, max_opacity=0.2, max_scale=0.3
    )

    result = PlyData.read(output_path)["vertex"].data
    assert result[0]["x"] == 7.0
    assert result[0]["mip_filter"] == 0.25
    assert np.exp(result[1]["scale_0"]) <= 0.300001
    assert 1.0 / (1.0 + np.exp(-result[1]["opacity"])) <= 0.200001
    assert result[1]["f_rest_0"] == 0.0
    assert report["opacity_clamped_count"] == 1
