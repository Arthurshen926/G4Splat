import numpy as np

from outdoor.foliage_geometry import cluster_tree_instances
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


def test_tree_instances_do_not_merge_spatially_separate_tracks():
    first = np.stack(
        [np.linspace(0, 0.9, 10), np.zeros(10), np.zeros(10)], axis=1
    )
    second = first + np.asarray([5.0, 0.0, 0.0])
    labels = cluster_tree_instances(
        np.concatenate([first, second], axis=0),
        connection_radius=0.25,
        minimum_tracks=4,
    )
    assert len(np.unique(labels[:10])) == 1
    assert len(np.unique(labels[10:])) == 1
    assert labels[0] != labels[10]
