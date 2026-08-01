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


def test_tree_instances_do_not_chain_across_an_unphysical_extent():
    # Every adjacent pair is within the radius, but the full chain is not one
    # physical crown and must not collapse to one latent/replacement group.
    xyz = np.stack(
        [np.arange(30, dtype=np.float64), np.zeros(30), np.zeros(30)],
        axis=1,
    )
    labels = cluster_tree_instances(
        xyz,
        connection_radius=1.1,
        minimum_tracks=4,
        maximum_component_extent=6.0,
    )
    assert len(np.unique(labels)) >= 4
    for label in np.unique(labels):
        component = xyz[labels == label]
        assert np.linalg.norm(component.ptp(axis=0)) <= 6.0 + 1e-8


def test_dense_tree_instance_uses_bounded_neighbour_graph():
    grid = np.stack(
        np.meshgrid(
            np.arange(18),
            np.arange(18),
            np.arange(3),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    xyz = grid.astype(np.float64) * 0.12

    labels = cluster_tree_instances(
        xyz,
        connection_radius=1.5,
        maximum_component_extent=12.0,
        maximum_neighbours=12,
    )

    assert len(np.unique(labels)) == 1


def test_sparse_bridge_does_not_merge_two_dense_tree_crowns():
    rng = np.random.default_rng(4)
    first = rng.normal(scale=0.12, size=(40, 3))
    second = rng.normal(scale=0.12, size=(40, 3))
    second[:, 0] += 5.0
    bridge = np.column_stack(
        [
            np.arange(1.0, 5.0, 1.0),
            np.zeros(4),
            np.zeros(4),
        ]
    )

    labels = cluster_tree_instances(
        np.concatenate([first, bridge, second], axis=0),
        connection_radius=1.1,
        minimum_tracks=8,
        maximum_component_extent=12.0,
        minimum_core_neighbours=4,
    )

    assert len(np.unique(labels[:40])) == 1
    assert len(np.unique(labels[-40:])) == 1
    assert labels[0] != labels[-1]
