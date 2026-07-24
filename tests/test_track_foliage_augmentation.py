import numpy as np

from scripts.augment_foliage_seed_with_sfm_tracks import _track_frames


def test_track_frames_create_compact_oriented_three_axis_gaussians():
    xyz = np.asarray(
        [[value * 0.03, 0.002 * (value % 2), 0.0] for value in range(20)],
        dtype=np.float32,
    )
    scales, quaternion, linearity = _track_frames(xyz, 0.01, 0.04, 0.12)
    assert scales.shape == (20, 3)
    assert quaternion.shape == (20, 4)
    assert np.all(scales >= 0.01)
    assert np.all(scales[:, 0] <= 0.12)
    assert float(np.median(linearity)) > 2.0
    assert np.allclose(np.linalg.norm(quaternion, axis=1), 1.0, atol=1e-5)
