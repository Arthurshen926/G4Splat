import numpy as np

from outdoor.evidence_store import ROLE_CANOPY, ROLE_RIGID
from outdoor.role_aware_initialization import (
    _selection_mask,
    _surface_frames,
    sparse_rigid_depth_maps,
)


def test_surface_selection_excludes_canopy_even_with_many_observations():
    tracks = {
        "role_probabilities": np.asarray(
            [
                [0.90, 0.02, 0.02, 0.01, 0.05],
                [0.10, 0.85, 0.01, 0.01, 0.03],
                [0.72, 0.20, 0.01, 0.01, 0.06],
            ],
            dtype=np.float32,
        ),
        "valid_observation_count": np.asarray([5, 20, 8]),
        "reprojection_error": np.asarray([0.5, 0.3, 0.8]),
    }
    keep = _selection_mask(
        tracks,
        minimum_rigid_probability=0.7,
        maximum_canopy_probability=0.15,
        minimum_observations=2,
        maximum_reprojection_error=2.0,
    )
    assert keep.tolist() == [True, False, False]


def test_surface_frames_produce_two_tangent_scales_and_unit_quaternion():
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)
    )
    xyz = np.column_stack(
        [xx.reshape(-1), yy.reshape(-1), np.zeros(xx.size)]
    ).astype(np.float32)
    scales, quaternion, normal = _surface_frames(
        xyz, np.full(len(xyz), 0.01, dtype=np.float32)
    )
    assert scales.shape == (len(xyz), 2)
    assert quaternion.shape == (len(xyz), 4)
    np.testing.assert_allclose(
        np.linalg.norm(quaternion, axis=1), 1.0, atol=1e-5
    )
    assert np.median(np.abs(normal[:, 2])) > 0.99


def test_sparse_rigid_zbuffer_keeps_nearest_depth():
    view = {
        "image_id": 7,
        "width": 100,
        "height": 100,
        "rotation": np.eye(3),
        "translation": np.zeros(3),
        "fx": 100.0,
        "fy": 100.0,
        "cx": 50.0,
        "cy": 50.0,
    }
    points = np.asarray([[0, 0, 4], [0, 0, 2]], dtype=np.float32)
    depth = sparse_rigid_depth_maps(
        [view],
        points,
        resolution_scale=1.0,
        dilation_pixels=1,
    )[7]
    assert depth[50, 50] == 2.0

