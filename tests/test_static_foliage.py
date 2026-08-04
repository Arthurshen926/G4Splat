import torch

from outdoor.hybrid_gaussian_renderer import (
    LAYER_CANONICAL_CROWN,
    LAYER_DYNAMIC_LEAF,
    LAYER_STATIC_SKELETON,
    VolumetricFoliageModel,
    view_depth_local_optical_replacement,
)
from outdoor.static_foliage import fuse_sequence_evidence_into_static_leaves
from outdoor.static_ray_birth import StaticRayBirthAccumulator


def _payload():
    # Two envelope cells, one skeleton, four associated observations and two
    # ownerless cross-sequence hit proposals in the same 15 cm voxel.
    count = 9
    centers = torch.tensor(
        [
            [0.0, 0.0, 3.0],
            [1.0, 0.0, 3.0],
            [0.0, -0.4, 3.0],
            [0.02, 0.01, 3.00],
            [-0.01, 0.00, 3.02],
            [1.01, 0.00, 3.00],
            [1.02, 0.01, 3.01],
            [2.01, 0.01, 3.00],
            [2.03, 0.02, 3.02],
        ],
        dtype=torch.float32,
    )
    layer = torch.tensor(
        [
            LAYER_CANONICAL_CROWN,
            LAYER_CANONICAL_CROWN,
            LAYER_STATIC_SKELETON,
            LAYER_DYNAMIC_LEAF,
            LAYER_DYNAMIC_LEAF,
            LAYER_DYNAMIC_LEAF,
            LAYER_DYNAMIC_LEAF,
            LAYER_DYNAMIC_LEAF,
            LAYER_DYNAMIC_LEAF,
        ],
        dtype=torch.int8,
    )
    replacement = torch.tensor([0, 1, -1, 0, 0, 1, 1, -1, -1])
    camera = torch.tensor([-1, -1, -1, 0, 1, 0, 0, 0, 1])
    support = torch.full((count, 2), -1, dtype=torch.int32)
    support[:, 0] = camera.to(torch.int32)
    colors = torch.tensor(
        [
            [0.20, 0.40, 0.20],
            [0.25, 0.45, 0.25],
            [0.30, 0.20, 0.10],
            [0.18, 0.50, 0.18],
            [0.22, 0.46, 0.20],
            [0.90, 0.90, 0.90],
            [0.10, 0.10, 0.10],
            [0.15, 0.55, 0.20],
            [0.17, 0.50, 0.18],
        ]
    )
    scales = torch.full((count, 3), 0.05)
    return {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": centers,
        "colors": colors,
        "scales": scales,
        "opacities": torch.full((count, 1), 0.05),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(
            count, 1
        ),
        "primitive_role": torch.zeros(count, dtype=torch.int8),
        "layer_role": layer,
        "track_id": torch.full((count,), -1, dtype=torch.int64),
        "tree_instance_id": torch.tensor([0, 1, 0, 0, 0, 1, 1, 2, 2]),
        "initialization_source": torch.zeros(count, dtype=torch.int8),
        "occupancy_probability": torch.ones(count),
        "position_covariance": torch.diag_embed(scales.square()),
        "reprojection_error": torch.zeros(count),
        "track_linearity": torch.zeros(count),
        "support_camera_ids": support,
        "observation_camera_ids": support.clone(),
        "observation_uv": torch.zeros(count, 2, 2),
        "observation_depth": torch.full((count, 2), 3.0),
        "support_view_count": torch.ones(count, dtype=torch.int16),
        "support_sequence_count": torch.ones(count, dtype=torch.int16),
        "ray_depth_nll": torch.zeros(count),
        "free_space_violation_count": torch.zeros(count, dtype=torch.int16),
        "unknown_view_count": torch.zeros(count, dtype=torch.int16),
        "static_skeleton_confidence": torch.zeros(count),
        "scale_ceiling": torch.full((count, 3), 0.2),
        "replacement_group": replacement,
        "ray_evidence": {},
        "audit": {
            "selected_views": [
                {"image_id": 0, "sequence_id": "seq0"},
                {"image_id": 1, "sequence_id": "seq1"},
            ]
        },
    }


def test_sequence_observations_fuse_to_one_static_model():
    payload, audit = fuse_sequence_evidence_into_static_leaves(
        _payload(), canonical_sequence_policy="scene"
    )
    assert not bool((payload["layer_role"] == LAYER_DYNAMIC_LEAF).any())
    # Group 0 has two cameras; group 1 has only camera 0. The ownerless voxel
    # has cameras from two independent sequences and becomes a ray birth.
    assert audit["fused_static_leaf_clusters"] == 1
    assert audit["ownerless_static_births"] == 1
    assert int(payload["static_detail"].sum()) == 2
    assert payload["replacement_group"][payload["static_detail"]].tolist() == [
        0,
        -1,
    ]
    # Tree 0 has a tie between seq0 and seq1. Stable canonical selection uses
    # seq0 rather than averaging the two incompatible leaf positions.
    detail = payload["static_detail"] & (
        payload["replacement_group"] == 0
    )
    assert torch.allclose(
        payload["centers"][detail][0], torch.tensor([0.02, 0.01, 3.00])
    )
    assert int(payload["support_sequence_count"][detail][0]) == 2
    ownerless = payload["static_detail"] & (
        payload["replacement_group"] < 0
    )
    # The promoted ownerless voxel must retain the complete calibrated
    # camera union.  support_view_count without these ids cannot route exact
    # hit/RGB supervision back to both observations.
    assert payload["support_camera_ids"][ownerless][0].tolist() == [0, 1]
    assert payload["observation_camera_ids"][ownerless][0].tolist() == [0, 1]
    assert audit["canonical_sequence_tree_count"] == 2
    assert audit["canonical_sequence_policy"] == "scene"
    assert audit["canonical_scene_sequence"] == "seq0"
    assert audit["canonical_sequence_histogram"] == {"seq0": 2}
    assert audit["maximum_modes_per_group"] == 2


def test_per_tree_snapshot_is_static_production_default():
    _, scene = fuse_sequence_evidence_into_static_leaves(
        _payload(), canonical_sequence_policy="scene"
    )
    _, production = fuse_sequence_evidence_into_static_leaves(_payload())
    assert scene["canonical_scene_sequence"] == "seq0"
    assert scene["canonical_sequence_histogram"] == {"seq0": 2}
    assert production["canonical_scene_sequence"] is None
    assert production["canonical_sequence_policy"] == "per_tree"
    assert "per_tree_canonical_sequence" in production["contract"]


def test_integrated_mass_survives_scale_refinement():
    payload, _ = fuse_sequence_evidence_into_static_leaves(_payload())
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)
    mass = foliage.integrated_optical_mass().detach().clone()
    with torch.no_grad():
        foliage.log_scales -= torch.log(torch.tensor(2.0))
        foliage.restore_integrated_optical_mass(mass)
    assert torch.allclose(
        foliage.integrated_optical_mass(), mass, rtol=2e-5, atol=2e-7
    )


def test_static_detail_replaces_only_projected_local_envelope():
    roles = torch.tensor(
        [LAYER_CANONICAL_CROWN, LAYER_CANONICAL_CROWN], dtype=torch.int8
    )
    replacement = view_depth_local_optical_replacement(
        roles,
        torch.tensor([0, 0]),
        torch.tensor([0.2, 0.2]),
        torch.full((2, 3), 0.05),
        torch.tensor([[10.0, 10.0], [10.0, 10.0]]),
        torch.tensor([3.0, 3.0]),
        focal_x=100.0,
        focal_y=100.0,
        cross_section=torch.ones(2),
        canonical_mask=torch.tensor([True, False]),
        detail_mask=torch.tensor([False, True]),
    )
    assert replacement[0] == 0.5
    assert replacement[1] == 0


def test_static_replacement_spends_only_overlapping_detail_optical_mass():
    roles = torch.tensor(
        [LAYER_CANONICAL_CROWN, LAYER_CANONICAL_CROWN], dtype=torch.int8
    )
    replacement = view_depth_local_optical_replacement(
        roles,
        torch.tensor([0, 0]),
        torch.tensor([0.2, 0.9]),
        torch.full((2, 3), 0.05),
        torch.tensor([[10.0, 10.0], [10.0, 10.0]]),
        torch.tensor([3.0, 3.0]),
        focal_x=100.0,
        focal_y=100.0,
        cross_section=torch.tensor([1.0, 25.0]),
        canonical_mask=torch.tensor([True, False]),
        detail_mask=torch.tensor([False, True]),
    )
    envelope_tau = -torch.log1p(torch.tensor(-0.2))
    remaining_alpha = 0.2 * (1.0 - replacement[0])
    removed_mass = envelope_tau + torch.log1p(-remaining_alpha)
    detail_tau = -torch.log1p(torch.tensor(-0.9))
    # Only 1/25 of the broad detail footprint overlaps the canonical row.
    # Its high optical depth still contains enough mass to replace that row,
    # but it cannot spend more than the overlapping optical-mass budget.
    assert replacement[0] > 0
    assert replacement[0] <= 0.5
    assert removed_mass <= detail_tau * 25.0 * (1.0 / 25.0) + 1.0e-6


def test_static_replacement_is_invariant_to_canonical_split():
    roles = torch.full((3,), LAYER_CANONICAL_CROWN, dtype=torch.int8)
    replacement = view_depth_local_optical_replacement(
        roles,
        torch.zeros(3, dtype=torch.int64),
        torch.full((3,), 0.2),
        torch.full((3, 3), 0.05),
        torch.tensor([[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]]),
        torch.full((3,), 3.0),
        focal_x=100.0,
        focal_y=100.0,
        cross_section=torch.tensor([0.5, 0.5, 1.0]),
        canonical_mask=torch.tensor([True, True, False]),
        detail_mask=torch.tensor([False, False, True]),
    )
    parent = view_depth_local_optical_replacement(
        torch.full((2,), LAYER_CANONICAL_CROWN, dtype=torch.int8),
        torch.zeros(2, dtype=torch.int64),
        torch.full((2,), 0.2),
        torch.full((2, 3), 0.05),
        torch.tensor([[10.0, 10.0], [10.0, 10.0]]),
        torch.full((2,), 3.0),
        focal_x=100.0,
        focal_y=100.0,
        cross_section=torch.ones(2),
        canonical_mask=torch.tensor([True, False]),
        detail_mask=torch.tensor([False, True]),
    )
    # The two half-area descendants contain exactly the parent's optical
    # mass. Their boundary-safe retirement equals the unsplit parent's rather
    # than losing authority to a per-child radius-ratio penalty.
    assert torch.allclose(replacement[:2], parent[:1].expand(2), atol=1e-6)
    assert torch.all(replacement[:2] == 0.5)


def test_uncovered_hits_require_cross_sequence_consensus_before_birth():
    accumulator = StaticRayBirthAccumulator(voxel_size=0.15)
    accumulator.add(
        {
            "centers": torch.tensor([[2.00, 0.00, 3.00]]),
            "colors": torch.tensor([[0.10, 0.20, 0.30]]),
            "confidence": torch.tensor([0.8]),
            "camera_id": 0,
        },
        sequence_id="seq0",
    )
    centers, colors, cameras, sequences, audit = accumulator.drain(
        maximum_births=8
    )
    assert len(centers) == 0
    accumulator.add(
        {
            "centers": torch.tensor([[2.03, 0.02, 3.01]]),
            "colors": torch.tensor([[0.30, 0.20, 0.10]]),
            "confidence": torch.tensor([0.9]),
            "camera_id": 1,
        },
        sequence_id="seq1",
    )
    centers, colors, cameras, sequences, audit = accumulator.drain(
        maximum_births=8
    )
    assert len(centers) == 1
    assert colors.shape == (1, 3)
    assert torch.isfinite(colors).all()
    assert cameras.shape == (1, 2)
    assert int(sequences[0]) == 2
    assert audit["born"] == 1


def test_cross_sequence_hit_intervals_form_visual_hull_birth():
    accumulator = StaticRayBirthAccumulator(
        voxel_size=0.15,
        visual_hull_voxel_size=0.30,
        maximum_segment_samples=16,
    )

    def add_segment(
        *, center, origin, direction, hit_start, hit_end, camera, sequence
    ):
        accumulator.add(
            {
                "centers": torch.tensor([center], dtype=torch.float32),
                "colors": torch.tensor([[0.2, 0.4, 0.2]]),
                "confidence": torch.tensor([0.9]),
                "camera_id": camera,
                "origins": torch.tensor([origin], dtype=torch.float32),
                "directions": torch.tensor(
                    [direction], dtype=torch.float32
                ),
                "hit_start": torch.tensor([hit_start]),
                "hit_end": torch.tensor([hit_end]),
            },
            sequence_id=sequence,
        )

    # The posterior midpoints are 30 cm apart and therefore fail the legacy
    # 15 cm midpoint test. Their calibrated hit segments intersect around
    # (0, 0, 3), which is the persistent visual-hull evidence we need.
    add_segment(
        center=[0.0, 0.0, 3.0],
        origin=[-1.0, 0.0, 3.0],
        direction=[1.0, 0.0, 0.0],
        hit_start=0.5,
        hit_end=1.5,
        camera=0,
        sequence="seq0",
    )
    assert len(accumulator.drain(maximum_births=8)[0]) == 0
    add_segment(
        center=[0.0, 0.3, 3.0],
        origin=[0.0, -1.0, 3.0],
        direction=[0.0, 1.0, 0.0],
        hit_start=0.9,
        hit_end=1.7,
        camera=1,
        sequence="seq1",
    )
    centers, _, cameras, sequences, audit = accumulator.drain(
        maximum_births=8
    )
    assert len(centers) >= 1
    assert audit["visual_hull_born"] >= 1
    assert audit["segment_proposals"] == 2
    assert int((cameras[0] >= 0).sum()) == 2
    assert int(sequences[0]) == 2


def test_v3_midpoint_birth_state_restores_into_visual_hull_accumulator():
    producer = StaticRayBirthAccumulator(voxel_size=0.15)
    producer.add(
        {
            "centers": torch.tensor([[2.0, 0.0, 3.0]]),
            "confidence": torch.tensor([0.8]),
            "camera_id": 0,
        },
        sequence_id="seq0",
    )
    legacy = producer.capture()
    legacy["version"] = "static-ray-birth-accumulator-v3"
    for key in (
        "visual_hull_voxel_size",
        "maximum_segment_samples",
        "segment_proposals",
        "visual_hull_cells",
    ):
        legacy.pop(key, None)

    restored = StaticRayBirthAccumulator(voxel_size=0.15)
    restored.restore(legacy)
    assert len(restored.cells) == 1
    assert len(restored.visual_hull_cells) == 0
    assert restored.total_proposals == 1


def test_runtime_ray_birth_is_static_and_gets_local_static_ownership():
    payload, _ = fuse_sequence_evidence_into_static_leaves(_payload())
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)
    before = len(foliage)
    event = foliage.append_static_ray_births(
        torch.tensor([[0.1, 0.0, 3.0]])
    )
    assert event["appended"] == 1
    assert len(foliage) == before + 1
    assert bool(foliage.static_leaf_mask[-1])
    assert not bool(foliage.dynamic_leaf_mask[-1])
    assert foliage.replacement_group[-1] >= 0
    assert event["replacement_group_association"]["assigned"] == 1
    assert foliage.support_sequence_count[-1] == 2


def test_runtime_ray_birth_without_same_tree_envelope_remains_ownerless():
    payload, _ = fuse_sequence_evidence_into_static_leaves(_payload())
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)
    event = foliage.append_static_ray_births(
        torch.tensor([[3.0, 0.0, 3.0]])
    )
    assert foliage.replacement_group[-1] == -1
    assert event["replacement_group_association"]["assigned"] == 0
