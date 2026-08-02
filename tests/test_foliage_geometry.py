import numpy as np
import torch

from outdoor.foliage_geometry import (
    _accumulate_supported_depth_nll,
    _allocate_dense_ray_budgets,
    _bounded_candidate_ray_rows,
    _confirmed_free_space_mask,
    _dense_component_ray_posterior,
    _dynamic_birth_limit_for_view,
    _incidence_mask,
    _invert_track_image_incidence,
    _priority_candidate_voxels,
    _spatially_stratified_coverage_rows,
    _supported_view_geometry_statistics,
    semantic_tree_tracks,
)
from outdoor.blue_noise_sampling import (
    deterministic_blue_noise_rows,
    stable_sampling_seed,
)
from outdoor.foliage_view_graph import (
    greedy_diverse_views,
    sequence_balanced_diverse_views,
    sequence_id,
    support_geometry,
)
from outdoor.training_evidence import EvidenceEpochSampler


def _budget_view(image_id):
    return {"image_id": image_id}


def test_track_image_incidence_is_inverted_once_for_selected_cameras():
    rows = _invert_track_image_incidence(
        [{3, 5}, {5}, set(), {7, 3}, {11}],
        [3, 5, 7],
    )

    np.testing.assert_array_equal(rows[3], np.asarray([0, 3]))
    np.testing.assert_array_equal(rows[5], np.asarray([0, 1]))
    np.testing.assert_array_equal(rows[7], np.asarray([3]))
    assert 11 not in rows
    np.testing.assert_array_equal(
        _incidence_mask(rows[3], 5),
        np.asarray([True, False, False, True, False]),
    )


def test_weak_continuous_view_gets_more_births_not_larger_splats():
    tracks = [
        {"dav2_alignment_status": "accepted"},
        {"dav2_alignment_status": "weak_continuous"},
        {},
    ]
    normal, normal_weak, normal_area = _dynamic_birth_limit_for_view(
        tracks,
        np.asarray([True, False, True]),
        ordinary_limit=384,
        weak_continuous_limit=2048,
    )
    expanded, expanded_weak, expanded_area = _dynamic_birth_limit_for_view(
        tracks,
        np.asarray([False, True, False]),
        ordinary_limit=384,
        weak_continuous_limit=2048,
    )

    assert normal == 384
    assert normal_area == 384
    assert not normal_weak
    assert expanded == 2048
    assert expanded_area == 384
    assert expanded_weak


def test_large_accepted_view_gets_continuous_pixel_bandwidth():
    tracks = [{"dav2_alignment_status": "accepted"}]

    limit, weak, area_limit = _dynamic_birth_limit_for_view(
        tracks,
        np.asarray([True]),
        ordinary_limit=384,
        weak_continuous_limit=2048,
        eligible_pixel_count=199_229,
        target_pixels_per_birth=192.0,
    )

    assert not weak
    assert area_limit == 1_038
    assert limit == 1_038


def test_pixel_bandwidth_is_bounded_by_adaptive_cap():
    limit, weak, area_limit = _dynamic_birth_limit_for_view(
        [{}],
        np.asarray([True]),
        ordinary_limit=384,
        weak_continuous_limit=2048,
        eligible_pixel_count=10_000_000,
        target_pixels_per_birth=192.0,
    )

    assert not weak
    assert area_limit == 2_048
    assert limit == 2_048


def test_dynamic_coverage_scaffold_is_two_dimensional_and_order_invariant():
    width = height = 6
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    sample_u = u.reshape(-1).astype(np.float64)
    sample_v = v.reshape(-1).astype(np.float64)
    rows = np.arange(width * height, dtype=np.int64)

    selected = _spatially_stratified_coverage_rows(
        rows, sample_u, sample_v, 4
    )
    shuffled = _spatially_stratified_coverage_rows(
        rows[::-1], sample_u, sample_v, 4
    )

    np.testing.assert_array_equal(selected, shuffled)
    assert np.ptp(sample_u[selected]) >= width / 2
    assert np.ptp(sample_v[selected]) >= height / 2
    pairwise = np.sqrt(
        (sample_u[selected, None] - sample_u[selected][None]) ** 2
        + (sample_v[selected, None] - sample_v[selected][None]) ** 2
    )
    assert pairwise[np.triu_indices(len(selected), 1)].min() >= 1.0


def test_blue_noise_sampling_is_camera_dephased_without_12_or_24_column_peak():
    height, width = 360, 640
    rows, columns = np.indices((height, width))
    candidates = np.arange(height * width, dtype=np.int64)
    first = deterministic_blue_noise_rows(
        candidates,
        columns.reshape(-1),
        rows.reshape(-1),
        128,
        seed=stable_sampling_seed("camera", 408),
    )
    second = deterministic_blue_noise_rows(
        candidates[::-1],
        columns.reshape(-1),
        rows.reshape(-1),
        128,
        seed=stable_sampling_seed("camera", 409),
    )
    assert len(first) == 128
    assert not np.array_equal(first, second)
    density = np.zeros((height, width), dtype=np.float64)
    density.reshape(-1)[first] = 1.0
    spectrum = np.abs(np.fft.rfft(density.sum(axis=0)))
    # A 24/12-column camera-plane lattice produces exact harmonics at these
    # bins with amplitude close to the DC sample count.
    for grid_columns in (12, 24):
        assert spectrum[grid_columns] < 0.35 * len(first)


def test_global_dense_ray_budget_preserves_coverage_and_exact_total():
    views = [_budget_view(index) for index in range(4)]
    rasters = {
        0: np.zeros((10, 10), dtype=np.int32),
        1: np.pad(
            np.zeros((8, 8), dtype=np.int32),
            ((0, 2), (0, 2)),
            constant_values=-1,
        ),
        2: np.pad(
            np.zeros((4, 4), dtype=np.int32),
            ((0, 6), (0, 6)),
            constant_values=-1,
        ),
        3: np.pad(
            np.zeros((2, 2), dtype=np.int32),
            ((0, 8), (0, 8)),
            constant_values=-1,
        ),
    }

    budgets, audit = _allocate_dense_ray_budgets(
        views,
        rasters,
        maximum_per_view=50,
        total_budget=120,
        minimum_per_view=10,
    )

    assert sum(budgets.values()) == 120
    assert all(4 <= value <= 50 for value in budgets.values())
    assert budgets[0] >= budgets[1] >= budgets[2] >= budgets[3]
    assert audit["allocated_total"] == 120
    assert audit["selected_view_count"] == 4
    assert audit["policy"].startswith("global_budget")


def test_dense_ray_budget_default_retains_independent_per_view_cap():
    views = [_budget_view(10), _budget_view(11)]
    rasters = {
        10: np.zeros((5, 5), dtype=np.int32),
        11: np.pad(
            np.zeros((2, 2), dtype=np.int32),
            ((0, 3), (0, 3)),
            constant_values=-1,
        ),
    }
    budgets, audit = _allocate_dense_ray_budgets(
        views,
        rasters,
        maximum_per_view=8,
    )
    assert budgets == {10: 8, 11: 4}
    assert audit["allocated_total"] == 12
    assert audit["policy"] == "independent_per_view_cap"


def test_global_dense_ray_budget_rejects_impossible_floor():
    views = [_budget_view(0), _budget_view(1)]
    rasters = {
        0: np.zeros((10, 10), dtype=np.int32),
        1: np.zeros((10, 10), dtype=np.int32),
    }
    with np.testing.assert_raises_regex(ValueError, "per-view floor"):
        _allocate_dense_ray_budgets(
            views,
            rasters,
            maximum_per_view=20,
            total_budget=10,
            minimum_per_view=8,
        )


def test_candidate_bound_ray_basis_is_bounded_spatial_and_type_complete():
    rows = np.arange(36, dtype=np.int64)
    pixel_u = np.tile(np.arange(6, dtype=np.float64), 6)
    pixel_v = np.repeat(np.arange(6, dtype=np.float64), 6)
    observation_type = np.zeros(36, dtype=np.int8)
    observation_type[:20] = 1
    observation_type[20:30] = -1

    first = _bounded_candidate_ray_rows(
        rows,
        pixel_u,
        pixel_v,
        observation_type,
        maximum_rows=12,
    )
    second = _bounded_candidate_ray_rows(
        rows,
        pixel_u,
        pixel_v,
        observation_type,
        maximum_rows=12,
    )

    assert len(first) == 12
    np.testing.assert_array_equal(first, second)
    assert set(observation_type[first].tolist()) == {-1, 0, 1}
    assert np.ptp(pixel_u[first]) >= 4
    assert np.ptp(pixel_v[first]) >= 4
    assert not len(
        _bounded_candidate_ray_rows(
            rows,
            pixel_u,
            pixel_v,
            observation_type,
            maximum_rows=0,
        )
    )


def test_sequence_id_is_not_frame_identity():
    assert sequence_id("seq7__frame00042.png") == "seq7"
    assert sequence_id("seq9/frame00001.png") == "seq9"


def test_support_geometry_requires_real_sequence_and_baseline_diversity():
    centers = {
        1: np.array([0.0, 0.0, 0.0]),
        2: np.array([1.0, 0.0, 0.0]),
        3: np.array([0.0, 2.0, 0.0]),
    }
    sequences = {1: "seq1", 2: "seq1", 3: "seq2"}

    result = support_geometry(
        np.array([0.0, 0.0, 5.0]), [1, 2, 3], centers, sequences
    )

    assert result["camera_count"] == 3
    assert result["sequence_count"] == 2
    assert result["max_baseline"] > 2
    assert result["max_triangulation_angle_degrees"] > 20


def test_diverse_view_selection_penalizes_repeated_sequences():
    records = [
        {
            "image_name": "seq1__a",
            "sequence_id": "seq1",
            "canopy_fraction": 0.9,
            "camera_center": np.array([0.0, 0.0, 0.0]),
        },
        {
            "image_name": "seq1__b",
            "sequence_id": "seq1",
            "canopy_fraction": 0.89,
            "camera_center": np.array([0.01, 0.0, 0.0]),
        },
        {
            "image_name": "seq2__a",
            "sequence_id": "seq2",
            "canopy_fraction": 0.8,
            "camera_center": np.array([2.0, 0.0, 0.0]),
        },
    ]

    selected = greedy_diverse_views(records, limit=2, minimum_center_distance=0.5)

    assert {row["sequence_id"] for row in selected} == {"seq1", "seq2"}


def test_sequence_balanced_selection_reserves_each_traversal():
    records = [
        {
            "image_id": index,
            "image_name": f"seq1__{index}",
            "sequence_id": "seq1",
            "canopy_fraction": 1.0 - index * 0.01,
            "camera_center": np.array([float(index), 0.0, 0.0]),
        }
        for index in range(8)
    ]
    records.append(
        {
            "image_id": 99,
            "image_name": "seq2__a",
            "sequence_id": "seq2",
            "canopy_fraction": 0.05,
            "camera_center": np.array([20.0, 0.0, 0.0]),
        }
    )
    selected = sequence_balanced_diverse_views(records, limit=4)
    assert {row["sequence_id"] for row in selected} == {"seq1", "seq2"}


def test_evidence_epoch_sampler_visits_every_factor_without_fixed_stride():
    sampler = EvidenceEpochSampler(11, seed=3)
    batches = [sampler.next(4) for _ in range(3)]
    assert len(np.unique(np.concatenate(batches)[:11])) == 11
    audit = sampler.audit()
    assert audit["never_visited"] == 0
    assert audit["completed_epochs"] == 1


def test_evidence_epoch_sampler_tail_does_not_wrap_and_repeat():
    sampler = EvidenceEpochSampler(11, seed=3)
    first = sampler.next(8)
    tail = sampler.next(8)
    assert len(first) == 8
    assert len(tail) == 3
    assert len(np.unique(np.concatenate([first, tail]))) == 11
    assert sampler.audit()["completed_epochs"] == 1
    assert sampler.audit()["maximum_visits"] == 1


def test_empty_cambridge_point2d_rows_use_tracked_calibrated_reprojection():
    mask = torch.ones((10, 10), dtype=torch.bool)
    mask[5, 5] = False

    class MaskLookup:
        masks = {"seq1/a.png": (mask, mask, mask, mask), "seq2/a.png": (mask, mask, mask, mask)}

        @staticmethod
        def source_name_for(name):
            return name

    images = {
        image_id: {
            "id": image_id,
            "name": name,
            "qvec": np.array([1.0, 0.0, 0.0, 0.0]),
            "tvec": np.zeros(3),
            "camera_id": 1,
            "xys": np.empty((0, 2)),
        }
        for image_id, name in [(1, "seq1/a.png"), (2, "seq1/a.png"), (3, "seq2/a.png")]
    }
    cameras = {
        1: {
            "model": "PINHOLE",
            "width": 10,
            "height": 10,
            "params": np.array([10.0, 10.0, 5.0, 5.0]),
        }
    }
    points = [
        {
            "id": 7,
            "xyz": np.array([0.0, 0.0, 5.0]),
            "rgb": np.array([20, 80, 30], dtype=np.uint8),
            "error": 0.2,
            "image_ids": np.array([1, 2, 3]),
            "point2d_indices": np.array([0, 0, 0]),
        }
    ]

    accepted = semantic_tree_tracks(
        points, images, cameras, MaskLookup(), minimum_tree_fraction=1.0
    )

    assert len(accepted) == 1
    assert accepted[0]["tree_sequence_count"] == 2
    assert accepted[0]["tree_image_ids"].tolist() == [1, 2, 3]


def test_supplemental_depth_candidates_never_evict_primary_cells():
    primary = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
    supplemental = np.asarray(
        [[10.0 + index, 0.0, 0.0] for index in range(20)],
        dtype=np.float64,
    )
    cells, _ = _priority_candidate_voxels(
        primary,
        supplemental,
        voxel_size=1.0,
        primary_dilation_steps=1,
        supplemental_dilation_steps=1,
        maximum_voxels=40,
        seed=5,
    )
    primary_cells = {
        tuple(row)
        for row in np.stack(
            np.meshgrid(
                np.arange(-1, 2),
                np.arange(-1, 2),
                np.arange(-1, 2),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)
    }
    assert primary_cells.issubset({tuple(row) for row in cells})
    assert len(cells) == 40


def test_only_visible_unoccluded_depth_violations_are_confirmed_free_space():
    confirmed = _confirmed_free_space_mask(
        valid=np.asarray([True, True, False, True]),
        occluded=np.asarray([False, True, False, False]),
        in_front_of_depth_posterior=np.asarray(
            [True, True, True, False]
        ),
    )

    assert confirmed.tolist() == [True, False, False, False]


def test_depth_nll_excludes_unknown_and_occluded_candidates():
    total = np.zeros(4, dtype=np.float64)
    residual = np.asarray([1.0, 50.0, -25.0, 2.0])
    sigma = np.asarray([0.5, 0.5, 0.5, 1.0])
    # Rows 1 and 2 represent large residuals classified as unknown/occluded.
    # They must not inflate a candidate's geometry covariance.
    supported = np.asarray([True, False, False, True])

    indices = _accumulate_supported_depth_nll(
        total, residual, sigma, supported
    )

    assert indices.tolist() == [0, 3]
    assert total.tolist() == [4.0, 0.0, 0.0, 4.0]


def test_sparse_supported_view_geometry_matches_dense_pair_reference():
    centers = np.asarray(
        [[0.2, -0.1, 4.0], [1.0, 0.5, 5.0], [0.0, 0.0, 3.0]]
    )
    cameras = np.asarray(
        [
            [-2.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [2.0, 0.5, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    support = np.asarray(
        [
            [True, True, False, True],
            [False, True, True, False],
            [True, False, False, False],
        ]
    )
    baseline, angle = _supported_view_geometry_statistics(
        centers, support, cameras
    )

    expected_baseline = np.zeros(3)
    expected_angle = np.zeros(3)
    for first in range(len(cameras)):
        for second in range(first + 1, len(cameras)):
            rows = support[:, first] & support[:, second]
            expected_baseline[rows] = np.maximum(
                expected_baseline[rows],
                np.linalg.norm(cameras[first] - cameras[second]),
            )
            first_ray = centers[rows] - cameras[first]
            second_ray = centers[rows] - cameras[second]
            cosine = np.sum(first_ray * second_ray, axis=1) / (
                np.linalg.norm(first_ray, axis=1)
                * np.linalg.norm(second_ray, axis=1)
            )
            expected_angle[rows] = np.maximum(
                expected_angle[rows],
                np.degrees(np.arccos(np.clip(cosine, -1, 1))),
            )

    np.testing.assert_allclose(baseline, expected_baseline, rtol=1e-6)
    np.testing.assert_allclose(angle, expected_angle, rtol=1e-6)


def test_dense_component_rays_create_ownerless_hole_proposals():
    view = {
        "image_id": 7,
        "width": 4,
        "height": 4,
        "fx": 4.0,
        "fy": 4.0,
        "cx": 2.0,
        "cy": 2.0,
        "rotation": np.eye(3),
        "translation": np.zeros(3),
        "tree_keep_mask": np.zeros((4, 4), dtype=bool),
    }
    record, proposals, dynamic_births = _dense_component_ray_posterior(
        view,
        np.asarray([[0.0, 0.0, 5.0]]),
        np.asarray([3], dtype=np.int32),
        np.asarray([0.1]),
        np.asarray([[20, 80, 30]], dtype=np.uint8),
        np.asarray([True]),
        depth_neighbor_pixels=10.0,
        maximum_rays=16,
        maximum_proposals=4,
    )

    assert record is not None
    assert len(record["camera_id"]) == 16
    assert np.all(record["candidate"] == -1)
    assert np.all(record["type"] == 1)
    assert len(proposals) == 4
    assert all(row["_dense_ray_proposal"] for row in proposals)
    assert {row["_tree_instance_id"] for row in proposals} == {3}
    np.testing.assert_allclose(
        [row["xyz"][2] for row in proposals], 5.0
    )
    # The default basis is larger than this tiny raster, so every row is
    # represented here. Production views explicitly decouple the much denser
    # posterior table from a bounded finite-footprint renderer basis.
    assert len(dynamic_births) == 16
    assert all(
        row["_dense_ray_dynamic_birth"] for row in dynamic_births
    )
    assert {row["_tree_instance_id"] for row in dynamic_births} == {3}
    assert {
        tuple(row["observation_camera_ids"].tolist())
        for row in dynamic_births
    } == {(7,)}
    assert all(row["ray_footprint_scale"] > 0 for row in dynamic_births)
    np.testing.assert_allclose(
        [row["observation_depth"][0] for row in dynamic_births],
        5.0,
    )


def test_dense_component_propagates_measured_depth_uncertainty():
    view = {
        "image_id": 7,
        "width": 4,
        "height": 4,
        "fx": 4.0,
        "fy": 4.0,
        "cx": 2.0,
        "cy": 2.0,
        "rotation": np.eye(3),
        "translation": np.zeros(3),
        "tree_keep_mask": np.zeros((4, 4), dtype=bool),
    }
    record, _, dynamic_births = _dense_component_ray_posterior(
        view,
        np.asarray([[0.0, 0.0, 5.0]]),
        np.asarray([3], dtype=np.int32),
        np.asarray([0.1]),
        np.asarray([[20, 80, 30]], dtype=np.uint8),
        np.asarray([True]),
        observation_depth_sigmas=np.asarray([1.2]),
        depth_neighbor_pixels=10.0,
        maximum_rays=16,
        maximum_proposals=0,
    )

    assert record is not None
    assert np.median(record["hit_end"] - record["hit_start"]) > 2.0
    assert min(row["position_sigma"] for row in dynamic_births) >= 1.2
    # Weak geometry remains in both the immutable ray table and the renderer
    # basis, but it no longer receives the same existence/topology authority
    # as a precise metric hit.
    assert len(record["camera_id"]) == 16
    assert len(dynamic_births) == 16
    assert max(row["tree_fraction"] for row in dynamic_births) < 0.2
    assert min(row["ray_depth_nll"] for row in dynamic_births) > 20.0


def test_dense_dynamic_births_use_exact_contracted_target_rgb():
    target_rgb = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    view = {
        "image_id": 8,
        "width": 4,
        "height": 4,
        "fx": 4.0,
        "fy": 4.0,
        "cx": 2.0,
        "cy": 2.0,
        "rotation": np.eye(3),
        "translation": np.zeros(3),
        "tree_keep_mask": np.zeros((4, 4), dtype=bool),
        "target_rgb": target_rgb,
    }
    _, proposals, dynamic_births = _dense_component_ray_posterior(
        view,
        np.asarray([[0.0, 0.0, 5.0]]),
        np.asarray([3], dtype=np.int32),
        np.asarray([0.1]),
        # Deliberately unrelated: this is canonical interpolation evidence,
        # not the exact-view dynamic colour target.
        np.asarray([[240, 240, 240]], dtype=np.uint8),
        np.asarray([True]),
        depth_neighbor_pixels=10.0,
        maximum_rays=16,
        maximum_proposals=4,
    )

    np.testing.assert_array_equal(
        np.stack([row["rgb"] for row in dynamic_births]),
        target_rgb.reshape(-1, 3),
    )
    assert {
        row["_dense_ray_rgb_source"] for row in dynamic_births
    } == {"exact_training_target_raster_pixel_center_bilinear"}
    assert all(
        np.array_equal(row["rgb"], [240, 240, 240])
        for row in proposals
    )


def test_dense_component_supports_multiple_instances_in_one_tree_mask():
    view = {
        "image_id": 9,
        "width": 4,
        "height": 4,
        "fx": 4.0,
        "fy": 4.0,
        "cx": 2.0,
        "cy": 2.0,
        "rotation": np.eye(3),
        "translation": np.zeros(3),
        # One connected foreground component, but two physical crowns.
        "tree_keep_mask": np.zeros((4, 4), dtype=bool),
    }
    record, _, dynamic_births = _dense_component_ray_posterior(
        view,
        np.asarray([[-1.25, 0.0, 5.0], [1.25, 0.0, 5.0]]),
        np.asarray([3, 4], dtype=np.int32),
        np.asarray([0.1, 0.1]),
        np.asarray([[20, 80, 30], [40, 100, 50]], dtype=np.uint8),
        np.asarray([True, True]),
        depth_neighbor_pixels=10.0,
        maximum_rays=16,
        maximum_proposals=4,
    )

    assert record is not None
    assert len(record["camera_id"]) == 16
    assert len(dynamic_births) == 16
    assert {row["_tree_instance_id"] for row in dynamic_births} == {3, 4}


def test_dense_ray_table_is_decoupled_from_instance_balanced_birth_basis():
    width, height = 8, 4
    instance_raster = np.full((height, width), 3, dtype=np.int32)
    instance_raster[:, width // 2 :] = 4
    view = {
        "image_id": 11,
        "width": width,
        "height": height,
        "fx": 100.0,
        "fy": 100.0,
        "cx": width / 2,
        "cy": height / 2,
        "rotation": np.eye(3),
        "translation": np.zeros(3),
        "tree_keep_mask": np.ones((height, width), dtype=bool),
    }
    depth = 5.0
    track_u = np.asarray([1.5, 5.5])
    track_v = np.asarray([1.5, 1.5])
    track_xyz = np.column_stack(
        [
            (track_u - view["cx"]) * depth / view["fx"],
            (track_v - view["cy"]) * depth / view["fy"],
            np.full(2, depth),
        ]
    )
    record, _, compact = _dense_component_ray_posterior(
        view,
        track_xyz,
        np.asarray([3, 4], dtype=np.int32),
        np.asarray([0.1, 0.1]),
        np.asarray([[20, 80, 30], [40, 100, 50]], dtype=np.uint8),
        np.asarray([True, True]),
        depth_neighbor_pixels=100.0,
        maximum_rays=32,
        maximum_proposals=0,
        maximum_dynamic_births=4,
        instance_raster=instance_raster,
    )
    _, _, dense = _dense_component_ray_posterior(
        view,
        track_xyz,
        np.asarray([3, 4], dtype=np.int32),
        np.asarray([0.1, 0.1]),
        np.asarray([[20, 80, 30], [40, 100, 50]], dtype=np.uint8),
        np.asarray([True, True]),
        depth_neighbor_pixels=100.0,
        maximum_rays=32,
        maximum_proposals=0,
        maximum_dynamic_births=32,
        instance_raster=instance_raster,
    )

    assert record is not None
    assert len(record["camera_id"]) == 32
    assert len(compact) == 4
    assert {row["_tree_instance_id"] for row in compact} == {3, 4}
    assert {
        row["_dense_ray_basis_role"] for row in compact
    } == {"coverage", "frequency_residual"}
    coverage_scale = [
        row["ray_footprint_scale"]
        for row in compact
        if row["_dense_ray_basis_role"] == "coverage"
    ]
    residual_scale = [
        row["ray_footprint_scale"]
        for row in compact
        if row["_dense_ray_basis_role"] == "frequency_residual"
    ]
    assert min(coverage_scale) > max(residual_scale)
    confidence_by_uv = {
        tuple(
            (pixel + 0.5) / np.asarray([width, height])
        ): confidence
        for pixel, confidence in zip(
            record["pixel"], record["confidence"]
        )
    }
    for birth in compact:
        uv = tuple(birth["observation_uv"][0])
        np.testing.assert_allclose(
            birth["tree_fraction"],
            confidence_by_uv[uv],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            1.0 / np.sqrt(1.0 + birth["ray_depth_nll"]),
            confidence_by_uv[uv],
            rtol=1e-6,
        )
    assert min(row["ray_footprint_scale"] for row in compact) > max(
        row["ray_footprint_scale"] for row in dense
    )
    assert {
        row["_dense_ray_basis_role"] for row in dense
    } == {"coverage", "frequency_residual"}


def test_dense_evidence_budget_covers_image_plane_not_raster_diagonal():
    width = height = 16
    view = {
        "image_id": 12,
        "width": width,
        "height": height,
        "fx": 100.0,
        "fy": 100.0,
        "cx": width / 2,
        "cy": height / 2,
        "rotation": np.eye(3),
        "translation": np.zeros(3),
        "tree_keep_mask": np.ones((height, width), dtype=bool),
    }
    depth = 5.0
    track_u = track_v = np.asarray([width / 2])
    track_xyz = np.column_stack(
        [
            (track_u - view["cx"]) * depth / view["fx"],
            (track_v - view["cy"]) * depth / view["fy"],
            np.full(1, depth),
        ]
    )
    record, _, births = _dense_component_ray_posterior(
        view,
        track_xyz,
        np.asarray([3], dtype=np.int32),
        np.asarray([0.1]),
        np.asarray([[20, 80, 30]], dtype=np.uint8),
        np.asarray([True]),
        depth_neighbor_pixels=100.0,
        maximum_rays=16,
        maximum_proposals=0,
        maximum_dynamic_births=16,
        instance_raster=np.full((height, width), 3, dtype=np.int32),
    )

    assert record is not None
    assert len(record["pixel"]) == len(births) == 16
    pixel = np.asarray(record["pixel"])
    quadrants = (
        (pixel[:, 0] >= width / 2).astype(np.int32)
        + 2 * (pixel[:, 1] >= height / 2).astype(np.int32)
    )
    assert set(quadrants.tolist()) == {0, 1, 2, 3}
    assert len(np.unique(pixel[:, 0])) >= 4
    assert len(np.unique(pixel[:, 1])) >= 4
    assert {
        row["_dense_ray_basis_role"] for row in births
    } == {"coverage", "frequency_residual"}


def test_dense_component_ray_budget_is_filled_after_posterior_filtering():
    # Only the centre of this silhouette is close enough to a measured depth
    # anchor.  Sampling the budget before validity would choose the two
    # endpoints and return no owners at all.
    width = 8
    view = {
        "image_id": 10,
        "width": width,
        "height": 1,
        "fx": 8.0,
        "fy": 8.0,
        "cx": 4.0,
        "cy": 0.5,
        "rotation": np.eye(3),
        "translation": np.zeros(3),
        "tree_keep_mask": np.ones((1, width), dtype=bool),
    }
    anchor_u = 3.5
    depth = 5.0
    anchor_x = (anchor_u - view["cx"]) * depth / view["fx"]
    record, _, dynamic_births = _dense_component_ray_posterior(
        view,
        np.asarray([[anchor_x, 0.0, depth]]),
        np.asarray([3], dtype=np.int32),
        np.asarray([0.1]),
        np.asarray([[20, 80, 30]], dtype=np.uint8),
        np.asarray([True]),
        depth_neighbor_pixels=1.1,
        maximum_rays=2,
        maximum_proposals=0,
        instance_raster=np.full((1, width), 3, dtype=np.int32),
    )

    assert record is not None
    assert len(record["camera_id"]) == 2
    assert len(dynamic_births) == 2
    assert all(row["ray_footprint_scale"] > 0 for row in dynamic_births)
