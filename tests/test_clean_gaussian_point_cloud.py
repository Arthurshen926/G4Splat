import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "clean_gaussian_point_cloud.py"
)
SPEC = importlib.util.spec_from_file_location("clean_gaussian_point_cloud", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
clean_vertices = MODULE.clean_vertices
visualization_vertices = MODULE.visualization_vertices


def _vertices():
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("primitive_class", "f4"),
    ]
    value = np.zeros(4, dtype=dtype)
    value["x"] = [0.0, 0.01, 1.0, 10.0]
    value["opacity"] = [
        np.log(0.5 / 0.5),
        np.log(0.001 / 0.999),
        np.log(0.5 / 0.5),
        np.log(0.5 / 0.5),
    ]
    value["scale_0"] = np.log(0.01)
    value["scale_1"] = np.log(0.01)
    value["primitive_class"][2] = 1
    return value


def test_clean_vertices_tracks_exclusive_reasons():
    retained, indices, audit = clean_vertices(
        _vertices(),
        opacity_min=0.01,
        drop_nonstructural=True,
        isolation_distance=0.5,
        isolation_scale_ratio=4.0,
    )
    assert indices.tolist() == [0]
    assert len(retained) == 1
    assert audit["exclusive_removal_reason_counts"] == {
        "nonfinite": 0,
        "nonstructural": 1,
        "opacity": 1,
        "isolated": 1,
    }


def test_visualization_vertices_has_rgb_and_provenance():
    vertices = _vertices()[[0]]
    vertices["f_dc_0"] = 0.0
    vertices["f_dc_1"] = 0.0
    vertices["f_dc_2"] = 0.0
    output = visualization_vertices(vertices, np.asarray([17]))
    assert (output[["red", "green", "blue"]].tolist()[0]) == (128, 128, 128)
    assert output["source_index"].tolist() == [17]
