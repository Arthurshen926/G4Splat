import json

import numpy as np
import pytest

from outdoor.mast3r_track_graph import (
    CameraUniqueUnionFind,
    _PointmapCache,
    TRACK_GRAPH_VERSION,
    descriptor_cycle_consistency,
    descriptor_view_direction_cost,
    _project,
    _target_raster_observations,
    _triangulate,
    _unproject,
    validate_track_gate,
)


def test_union_find_rejects_transitive_duplicate_camera_observation():
    union = CameraUniqueUnionFind()
    camera0_first = union.add(0)
    camera1 = union.add(1)
    camera2 = union.add(2)
    camera0_second = union.add(0)

    assert union.union(camera0_first, camera1)
    assert union.union(camera1, camera2)
    assert not union.union(camera2, camera0_second)
    assert union.find(camera0_first) == union.find(camera2)
    assert union.find(camera0_second) != union.find(camera2)
from outdoor.training_evidence import _rigid_global_track_mask


def _archive(path, *, count=20, observations=3, sequences=2, error=0.4, angle=2.0):
    offsets = np.arange(count + 1, dtype=np.int64) * observations
    total = int(offsets[-1])
    np.savez_compressed(
        path,
        schema_version=np.asarray(TRACK_GRAPH_VERSION),
        track_id=np.arange(count, dtype=np.int64),
        valid_observation_count=np.full(count, observations, dtype=np.int16),
        sequence_count=np.full(count, sequences, dtype=np.int16),
        dominant_role=np.zeros(count, dtype=np.int8),
        reprojection_error=np.full(count, error, dtype=np.float32),
        triangulation_angle_median=np.full(count, angle, dtype=np.float32),
        observation_offsets=offsets,
        observation_camera_indices=(
            np.arange(total, dtype=np.int32) % 2
        ),
        observation_pixels=np.zeros((total, 2), dtype=np.float32),
        observation_camera_depth=np.ones(total, dtype=np.float32),
        camera_names=np.asarray(["a", "b"]),
        camera_image_sizes=np.asarray([[512, 288], [512, 288]], dtype=np.int32),
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


def test_stale_track_schema_is_rejected_before_quality_statistics(tmp_path):
    path = tmp_path / "tracks.npz"
    _archive(path)
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values["schema_version"] = np.asarray("stale-v2")
    np.savez_compressed(path, **values)
    with pytest.raises(RuntimeError, match="schema is stale"):
        validate_track_gate(
            path,
            minimum_tracks=10,
            minimum_global_rigid_tracks=10,
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


def test_projection_uses_anisotropic_offcenter_exact_k():
    xyz = np.asarray([[1.0, 1.0, 4.0]], dtype=np.float64)
    u, v, depth, valid = _project(
        xyz,
        np.eye(4, dtype=np.float64),
        np.asarray([800.0, 600.0, 301.0, 199.0]),
        640,
        480,
    )
    assert valid[0]
    assert depth[0] == pytest.approx(4.0)
    assert u[0] == pytest.approx(501.0)
    assert v[0] == pytest.approx(349.0)


def test_descriptor_pairing_rejects_opposite_optical_axes():
    source = np.asarray([0.0, 0.0, 1.0])
    target = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    )
    cost = descriptor_view_direction_cost(source, target)
    np.testing.assert_allclose(cost, [0.0, 2.0])


def test_descriptor_cycle_audit_is_not_an_unconditional_one():
    no_loop_rank, pair_consistency = descriptor_cycle_consistency(
        0.5, 1.5, descriptor_edge_count=1, observation_node_count=2
    )
    loop_rank, loop_consistency = descriptor_cycle_consistency(
        0.5, 1.5, descriptor_edge_count=3, observation_node_count=3
    )
    assert no_loop_rank == 0
    assert loop_rank == 1
    assert 0.5 < pair_consistency < loop_consistency < 1.0


def test_unprojection_round_trips_an_independent_exact_k_observation():
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 3] = [1.0, -2.0, 0.5]
    intrinsics = np.asarray([800.0, 600.0, 301.0, 199.0])
    pixels = np.asarray([[501.0, 349.0], [301.0, 199.0]])
    depth = np.asarray([4.0, 2.0])
    xyz = _unproject(pixels, depth, c2w, intrinsics)
    u, v, recovered_depth, valid = _project(
        xyz, c2w, intrinsics, 800, 600
    )
    assert valid.all()
    np.testing.assert_allclose(np.column_stack([u, v]), pixels, atol=1e-5)
    np.testing.assert_allclose(recovered_depth, depth, atol=1e-6)


def test_target_observation_uses_sampled_pointmap_pixel_not_search_projection():
    projected_u = np.asarray([10.49, 20.51, -2.0, 99.0])
    projected_v = np.asarray([4.49, 5.51, 8.0, 99.0])
    pixels, rows, columns = _target_raster_observations(
        projected_u, projected_v, width=32, height=16
    )
    np.testing.assert_array_equal(columns, [10, 21, 0, 31])
    np.testing.assert_array_equal(rows, [4, 6, 8, 15])
    np.testing.assert_array_equal(
        pixels,
        np.asarray([[10, 4], [21, 6], [0, 8], [31, 15]], dtype=np.float32),
    )
    # The old bug stored the proposal itself, so triangulation merely proved
    # that a source point lies on rays created by projecting that same point.
    assert not np.allclose(
        pixels[:2],
        np.column_stack([projected_u[:2], projected_v[:2]]),
    )


def test_pointmap_cache_retargets_internal_focal_rays_to_fixed_k(tmp_path):
    height, width = 3, 4
    rows, columns = np.mgrid[:height, :width]
    depth = np.full((height, width), 2.0, dtype=np.float32)
    internal_focal = 2.0
    internal_cx, internal_cy = 1.5, 1.0
    points = np.stack(
        [
            (columns - internal_cx) * depth / internal_focal,
            (rows - internal_cy) * depth / internal_focal,
            depth,
        ],
        axis=-1,
    ).astype(np.float32)
    pointmap_root = tmp_path / "pointmaps"
    pointmap_root.mkdir()
    (pointmap_root / "frame.json").write_text(
        json.dumps(
            {
                "points": points.reshape(-1, 3).tolist(),
                "confs": np.full((height, width), 2.0).tolist(),
            }
        )
    )
    c2w = np.eye(4, dtype=np.float64)
    exact_k = np.asarray([4.0, 3.0, 1.25, 0.75], dtype=np.float64)
    cache = _PointmapCache(
        pointmap_root,
        camera_contract={"frame": (c2w, exact_k)},
    )
    fixed = cache.get("frame")["points"].reshape(-1, 3)
    u, v, recovered_depth, valid = _project(
        fixed, c2w, exact_k, width, height
    )
    assert valid.all()
    np.testing.assert_allclose(
        np.column_stack([u, v]),
        np.column_stack([columns.reshape(-1), rows.reshape(-1)]),
        atol=1e-5,
    )
    np.testing.assert_allclose(recovered_depth, depth.reshape(-1), atol=1e-6)


def test_surface_observation_factor_excludes_canopy_and_sequence_local_tracks():
    probability = np.asarray(
        [
            [0.90, 0.02, 0.02, 0.01, 0.05],
            [0.10, 0.85, 0.01, 0.01, 0.03],
            [0.90, 0.02, 0.02, 0.01, 0.05],
            [0.80, 0.18, 0.00, 0.00, 0.02],
        ],
        dtype=np.float32,
    )
    archive = {
        "role_probabilities": probability,
        "sequence_count": np.asarray([2, 3, 1, 2]),
        "cycle_consistency": np.asarray([0.9, 0.9, 0.9, 0.9]),
        "pointmap_spread": np.asarray([0.02, 0.02, 0.02, 0.02]),
    }
    assert _rigid_global_track_mask(archive).tolist() == [
        True,
        False,
        False,
        False,
    ]
