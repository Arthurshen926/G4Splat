import numpy as np
import torch

from outdoor.foliage_geometry import (
    _accumulate_supported_depth_nll,
    _allocate_dense_ray_budgets,
    _confirmed_free_space_mask,
    _dense_component_ray_posterior,
    _priority_candidate_voxels,
    _supported_view_geometry_statistics,
    semantic_tree_tracks,
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
    # Canonical hole proposals remain bounded, but every calibrated hit gets
    # a sequence-local renderer primitive. Otherwise a dense posterior row
    # can supervise no parameter when the canonical hull rejects its cell.
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
    } == {"exact_training_target_raster"}
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
