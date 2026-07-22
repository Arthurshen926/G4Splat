from pathlib import Path

import numpy as np

from scripts.audit_mast3r_pointmaps import sparse_export_track_contract
from colmap.read_write_model import (
    Camera,
    Image,
    Point3D,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
)


def _write_sparse_export(root: Path, point2d_index: int) -> None:
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True)
    write_cameras_binary(
        {1: Camera(1, "PINHOLE", 16, 16, np.array([8.0, 8.0, 8.0, 8.0]))},
        str(sparse / "cameras.bin"),
    )
    write_images_binary(
        {
            1: Image(
                1,
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.zeros(3),
                1,
                "chart.png",
                np.array([[8.0, 8.0]]),
                np.array([9]),
            )
        },
        str(sparse / "images.bin"),
    )
    write_points3D_binary(
        {
            9: Point3D(
                9,
                np.array([0.0, 0.0, 1.0]),
                np.array([0, 0, 0]),
                0.0,
                np.array([1]),
                np.array([point2d_index]),
            )
        },
        str(sparse / "points3D.bin"),
    )


def test_sparse_export_track_contract_accepts_compact_image_index(tmp_path):
    _write_sparse_export(tmp_path, point2d_index=0)
    assert sparse_export_track_contract(tmp_path)["passed"]


def test_sparse_export_track_contract_rejects_dense_pixel_index(tmp_path):
    _write_sparse_export(tmp_path, point2d_index=5)
    report = sparse_export_track_contract(tmp_path)
    assert not report["passed"]
    assert report["invalid_track_count"] == 1
