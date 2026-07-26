import numpy as np
import pytest

from outdoor.mast3r_track_graph import _triangulate, validate_track_gate


def _archive(path, *, count=20, observations=3, sequences=2, error=0.4, angle=2.0):
    np.savez_compressed(
        path,
        track_id=np.arange(count, dtype=np.int64),
        valid_observation_count=np.full(count, observations, dtype=np.int16),
        sequence_count=np.full(count, sequences, dtype=np.int16),
        dominant_role=np.zeros(count, dtype=np.int8),
        reprojection_error=np.full(count, error, dtype=np.float32),
        triangulation_angle_median=np.full(count, angle, dtype=np.float32),
    )


def test_true_multiview_tracks_pass_gate(tmp_path):
    path = tmp_path / "tracks.npz"
    _archive(path)
    result = validate_track_gate(
        path,
        minimum_tracks=10,
        minimum_global_rigid_tracks=10,
    )
    assert result["passed"]
    assert result["values"]["median_observations"] == 3


def test_single_observation_sparse_export_is_rejected(tmp_path):
    path = tmp_path / "tracks.npz"
    _archive(path, observations=1, sequences=1)
    with pytest.raises(RuntimeError, match="median_observations"):
        validate_track_gate(
            path,
            minimum_tracks=10,
            minimum_global_rigid_tracks=1,
        )


def test_triangulation_covariance_converts_pixels_to_ray_angle():
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    c2w[1, 0, 3] = 1.0
    focal = np.full(2, 500.0)
    # World point (0,0,5): the second camera observes x=-1 in camera space.
    pixels = np.asarray([[256.0, 144.0], [156.0, 144.0]])
    xyz, _, _, angle_and_covariance = _triangulate(
        pixels,
        np.asarray([0, 1], dtype=np.int32),
        c2w,
        focal,
        512,
        288,
        np.ones(2),
    )
    assert np.allclose(xyz, [0, 0, 5], atol=1e-4)
    assert np.sqrt(angle_and_covariance[3:]).max() < 0.02
