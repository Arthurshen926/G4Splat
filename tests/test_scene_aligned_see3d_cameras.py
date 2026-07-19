from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "2d-gaussian-splatting"))

from guidance.cam_utils import (  # noqa: E402
    build_pose_graph_interpolated_c2ws,
    estimate_camera_vertical_axis,
)


def _look_at_pose(center: np.ndarray, vertical: np.ndarray) -> np.ndarray:
    forward = -center / np.linalg.norm(center)
    right = np.cross(vertical, forward)
    right /= np.linalg.norm(right)
    image_vertical = np.cross(forward, right)
    image_vertical /= np.linalg.norm(image_vertical)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.stack([right, image_vertical, forward], axis=1)
    pose[:3, 3] = center
    return pose


def test_pose_graph_interpolation_respects_non_z_vertical_axis():
    vertical = np.array([0.0, 1.0, 0.0])
    centers = [
        np.array([4.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 4.0]),
        np.array([-4.0, 0.0, 0.0]),
        np.array([0.0, 0.0, -4.0]),
    ]
    poses = np.stack([_look_at_pose(center, vertical) for center in centers])

    estimated_vertical = estimate_camera_vertical_axis(poses)
    generated = build_pose_graph_interpolated_c2ws(
        poses,
        n_frames=8,
        neighbor_count=2,
    )

    assert np.dot(estimated_vertical, vertical) > 0.999
    assert generated.shape == (8, 4, 4)
    assert np.max(np.abs(generated[:, 1, 3])) < 1e-6
    assert np.min(generated[:, :3, 1] @ vertical) > 0.999


def test_pose_graph_interpolation_rejects_invalid_shapes():
    invalid = np.zeros((3, 3, 4), dtype=np.float32)

    try:
        build_pose_graph_interpolated_c2ws(invalid)
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("invalid camera poses must be rejected")
