import numpy as np
import torch

from outdoor.foliage_geometry import (
    evidence_conditioned_leaf_optical_mass,
    measured_static_skeleton_geometry,
)
from scripts.repair_foliage_seed_contract import (
    _merge_sequence_groups,
    sequence_local_correspondence_groups,
)
from scripts.repair_foliage_optical_prior import _repair_opacity_payload


def test_measured_skeleton_uses_continuous_multiview_line_evidence():
    centers = np.column_stack(
        [
            np.linspace(0.0, 0.36, 16),
            np.zeros(16),
            np.zeros(16),
        ]
    ).astype(np.float32)
    result = measured_static_skeleton_geometry(
        centers,
        np.tile(np.asarray([[0.35, 0.27, 0.18]]), (16, 1)),
        np.zeros(16, dtype=np.int32),
        np.full(16, 4, dtype=np.int16),
        np.full(16, 2, dtype=np.int16),
        np.full(16, 0.85, dtype=np.float32),
        np.full(16, 1.0, dtype=np.float32),
        np.zeros(16, dtype=np.int16),
        voxel_size=0.06,
        minimum_confidence=0.08,
    )
    assert result["selected"].any()
    assert result["confidence"].max() > result["confidence"].min()
    selected_scales = result["scales"][result["selected"]]
    assert np.all(selected_scales[:, 0] > selected_scales[:, 1])
    np.testing.assert_allclose(
        np.linalg.norm(
            result["quaternions"][result["selected"]], axis=1
        ),
        1.0,
        atol=1e-5,
    )


def test_leaf_optical_mass_is_spatial_bounded_and_contradiction_aware():
    initial, floor = evidence_conditioned_leaf_optical_mass(
        torch.tensor([0.70, 0.90, 0.90]),
        torch.tensor([1, 1, 3]),
        torch.tensor([0, 0, 0]),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([0, 0, 0]),
    )
    contradicted, contradicted_floor = (
        evidence_conditioned_leaf_optical_mass(
            torch.tensor([0.90]),
            torch.tensor([1]),
            torch.tensor([2]),
            torch.tensor([4.0]),
            torch.tensor([2]),
        )
    )

    assert torch.all((initial >= 0.08) & (initial <= 0.40))
    assert torch.all((floor >= 0.020) & (floor <= 0.060))
    # Equal owner/free-space evidence gives equal optical initialization even
    # when geometry occupancy differs; the geometry-aware floor still tracks
    # that distinction.
    torch.testing.assert_close(initial[1], initial[0])
    assert floor[1] > floor[0]
    assert initial[2] > initial[1]
    assert floor[2] > floor[1]
    assert contradicted[0] < initial[1]
    assert contradicted_floor[0] < floor[1]


def test_optical_prior_repair_changes_only_observed_dynamic_alpha():
    payload = {
        "opacities": torch.full((4, 1), 0.02),
        "layer_role": torch.tensor([2, 2, 2, 1]),
        "initialization_source": torch.tensor([4, 3, 1, 4]),
        "occupancy_probability": torch.tensor([0.1, 0.2, 0.9, 0.9]),
        "support_view_count": torch.tensor([1, 1, 3, 3]),
        "unknown_view_count": torch.zeros(4),
        "ray_depth_nll": torch.tensor([99.0, 24.0, 0.0, 0.0]),
        "free_space_violation_count": torch.zeros(4),
        "centers": torch.arange(12).reshape(4, 3),
        "audit": {},
    }
    result = _repair_opacity_payload(payload)

    assert result["opacities"][0, 0] > 0.30
    # DAV2-only geometry has no exact calibrated owner-pixel optical
    # authority and therefore keeps its original alpha.
    assert result["opacities"][1, 0] == payload["opacities"][1, 0]
    torch.testing.assert_close(
        result["opacities"][2:], payload["opacities"][2:]
    )
    torch.testing.assert_close(result["centers"], payload["centers"])
    audit = result["audit"]["optical_existence_prior_repair"]
    assert audit["candidate_count"] == 1
    assert audit["lifted_count"] == 1
    assert not audit["historical_rgb_fit_used"]


def test_sequence_graph_never_merges_same_camera_samples():
    centers = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.01, 0.00, 0.00],
            [0.002, 0.00, 0.00],
            [1.00, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    groups = sequence_local_correspondence_groups(
        centers,
        np.zeros(4, dtype=np.int32),
        np.asarray([10, 10, 11, 11], dtype=np.int32),
        {10: ("seq1", 1), 11: ("seq1", 2)},
        np.ones(4, dtype=bool),
        radius=0.05,
        maximum_frame_gap=4,
        maximum_group_size=3,
    )
    assert len(groups) == 1
    cameras = np.asarray([10, 10, 11, 11], dtype=np.int32)
    assert len(np.unique(cameras[groups[0]])) == len(groups[0])
    assert len(groups[0]) == 2


def test_sequence_group_merge_preserves_ray_bound_prefix_and_observations():
    count = 4
    payload = {
        "centers": torch.tensor(
            [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [2.02, 0, 0]]
        ),
        "initialization_center": torch.tensor(
            [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [2.02, 0, 0]]
        ),
        "colors": torch.full((count, 3), 0.5),
        "scales": torch.full((count, 3), 0.02),
        "opacities": torch.full((count, 1), 0.02),
        "quaternions": torch.tensor(
            [[1.0, 0, 0, 0]] * count
        ),
        "layer_role": torch.tensor([0, 0, 2, 2], dtype=torch.int8),
        "track_id": torch.tensor([-1, -1, -20, -21]),
        "tree_instance_id": torch.zeros(count, dtype=torch.int32),
        "initialization_source": torch.tensor(
            [0, 0, 4, 4], dtype=torch.int8
        ),
        "occupancy_probability": torch.ones(count),
        "position_covariance": torch.eye(3)[None].repeat(count, 1, 1)
        * 0.01,
        "reprojection_error": torch.ones(count),
        "ray_depth_nll": torch.ones(count),
        "free_space_violation_count": torch.zeros(
            count, dtype=torch.int16
        ),
        "unknown_view_count": torch.zeros(count, dtype=torch.int16),
        "maximum_baseline": torch.zeros(count),
        "maximum_triangulation_angle_degrees": torch.zeros(count),
        "support_camera_ids": torch.tensor(
            [[-1, -1], [-1, -1], [10, -1], [11, -1]],
            dtype=torch.int32,
        ),
        "support_view_count": torch.tensor(
            [2, 2, 1, 1], dtype=torch.int16
        ),
        "support_sequence_count": torch.tensor(
            [2, 2, 1, 1], dtype=torch.int16
        ),
        "observation_camera_ids": torch.tensor(
            [[-1, -1], [-1, -1], [10, -1], [11, -1]],
            dtype=torch.int32,
        ),
        "observation_uv": torch.tensor(
            [
                [[float("nan"), float("nan")]] * 2,
                [[float("nan"), float("nan")]] * 2,
                [[0.2, 0.3], [float("nan"), float("nan")]],
                [[0.21, 0.3], [float("nan"), float("nan")]],
            ]
        ),
        "observation_depth": torch.tensor(
            [
                [float("nan"), float("nan")],
                [float("nan"), float("nan")],
                [3.0, float("nan")],
                [3.1, float("nan")],
            ]
        ),
        "ray_evidence": {"offsets": torch.zeros(3, dtype=torch.int64)},
    }
    merged, audit = _merge_sequence_groups(
        payload,
        [np.asarray([2, 3], dtype=np.int64)],
        {10: ("seq1", 1), 11: ("seq1", 2)},
    )
    assert audit["removed_duplicate_rows"] == 1
    assert len(merged["centers"]) == 3
    assert set(merged["observation_camera_ids"][-1].tolist()) == {10, 11}
    assert len(merged["ray_evidence"]["offsets"]) == 3
