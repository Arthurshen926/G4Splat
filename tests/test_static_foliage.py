import pytest
import torch

from outdoor.hybrid_gaussian_renderer import (
    LAYER_CANONICAL_CROWN,
    LAYER_DYNAMIC_LEAF,
    LAYER_STATIC_SKELETON,
    PROPOSAL_SPLIT,
    VERIFICATION_MEASURED_SINGLE,
    VERIFICATION_UNVERIFIED,
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
    # Group 0 has two cameras; group 1 has only camera 0. Both calibrated
    # occupancy cells survive.  The ownerless voxel has only one observation
    # in the selected scene-canonical sequence, so the other traversal is
    # unknown and cannot manufacture a static-map seed.
    assert audit["fused_static_leaf_clusters"] == 2
    assert audit["ownerless_static_births"] == 0
    assert int(payload["static_detail"].sum()) == 2
    assert payload["replacement_group"][payload["static_detail"]].tolist() == [
        0,
        1,
    ]
    # Tree 0 has a tie between seq0 and seq1. Stable canonical selection uses
    # seq0 rather than averaging the two incompatible leaf positions.
    detail = payload["static_detail"] & (
        payload["replacement_group"] == 0
    )
    assert torch.allclose(
        payload["centers"][detail][0], torch.tensor([0.02, 0.01, 3.00])
    )
    # The local leaf mode belongs to the selected static snapshot only. The
    # broad parent may span traversals, but that must not fabricate appearance
    # or verification support for this particular leaf position.
    assert int(payload["support_sequence_count"][detail][0]) == 1
    assert int(payload["verified_sequence_count"][detail][0]) == 1
    assert int(payload["verified_camera_count"][detail][0]) == 1
    assert int(payload["verification_state"][detail][0]) == (
        VERIFICATION_MEASURED_SINGLE
    )
    # A single-view mode remains exact-owner/DC-only, but receives a small
    # mass-conserving share from its envelope instead of being born invisible.
    assert float(payload["opacities"][detail][0]) > 0.0
    assert audit["canonical_sequence_tree_count"] == 2
    assert audit["canonical_sequence_policy"] == "scene"
    assert audit["canonical_scene_sequence"] == "seq0"
    assert audit["canonical_sequence_histogram"] == {"seq0": 2}
    assert audit["maximum_modes_per_group"] == 2


def test_scene_canonical_ownerless_seed_requires_two_canonical_cameras():
    payload = _payload()
    # Camera 2 is a second calibrated view in seq0.  Move the second
    # ownerless observation from non-canonical seq1 to this camera.
    payload["support_camera_ids"][8, 0] = 2
    payload["observation_camera_ids"][8, 0] = 2
    payload["audit"]["selected_views"].append(
        {"image_id": 2, "sequence_id": "seq0"}
    )
    fused, audit = fuse_sequence_evidence_into_static_leaves(
        payload,
        canonical_sequence_policy="scene",
        fixed_camera_sequences=[
            {"image_id": 0, "sequence_id": "seq0"},
            {"image_id": 1, "sequence_id": "seq1"},
            {"image_id": 2, "sequence_id": "seq0"},
        ],
    )
    ownerless = fused["static_detail"] & (
        fused["replacement_group"] < 0
    )
    assert audit["ownerless_static_births"] == 1
    assert audit["ownerless_camera_scope"] == "explicit_canonical_sequence"
    assert fused["support_camera_ids"][ownerless][0].tolist() == [0, 2]
    assert fused["verified_camera_ids"][ownerless][0].tolist() == [0, 2]
    assert int(fused["verified_camera_count"][ownerless][0]) == 2
    assert int(fused["verified_sequence_count"][ownerless][0]) == 1
    assert int(fused["verification_state"][ownerless][0]) == 1
    assert float(fused["opacities"][ownerless][0]) == pytest.approx(0.04)


def test_static_fusion_uses_complete_fixed_camera_sequence_contract():
    payload = _payload()
    payload["audit"]["selected_views"] = [
        {"image_id": 0, "sequence_id": "seq0"}
    ]
    fused, audit = fuse_sequence_evidence_into_static_leaves(
        payload,
        canonical_sequence_policy="per_tree",
        fixed_camera_sequences=[
            {"image_id": 0, "sequence_id": "seq0"},
            {"image_id": 1, "sequence_id": "seq1"},
        ],
    )
    assert audit["camera_sequence_metadata_source"] == (
        "runtime_fixed_camera_contract"
    )
    assert audit["mapped_associated_rows"] == 4
    assert audit["unmapped_associated_rows"] == 0
    detail_groups = fused["replacement_group"][fused["static_detail"]]
    assert 0 in detail_groups.tolist()


def test_selected_mode_fuses_all_consistent_group_observations():
    payload = _payload()
    # These observations belong to one persistent group and one coherent
    # acquisition, but fall in different 8 cm seed voxels.  Selecting one
    # bounded mode must not turn the other calibrated observation into
    # discarded evidence.
    payload["centers"][3] = torch.tensor([0.01, 0.0, 3.0])
    payload["centers"][4] = torch.tensor([0.09, 0.0, 3.0])
    fused, audit = fuse_sequence_evidence_into_static_leaves(
        payload,
        maximum_modes_per_group=1,
        fixed_camera_sequences=[
            {"image_id": 0, "sequence_id": "seq0"},
            {"image_id": 1, "sequence_id": "seq0"},
        ],
    )
    detail = fused["static_detail"] & (
        fused["replacement_group"] == 0
    )
    assert int(detail.sum()) == 1
    assert torch.allclose(
        fused["centers"][detail][0], torch.tensor([0.05, 0.0, 3.0])
    )
    assert fused["support_camera_ids"][detail][0].tolist() == [0, 1]
    assert int(fused["support_view_count"][detail][0]) == 2
    # The single-view second group is admitted as low-authority occupancy and
    # therefore contributes its two same-camera rows as well.
    assert audit["canonical_source_rows"] == 4
    assert audit["contributing_multiview_rows"] == 2
    assert audit["multiview_canonical_modes"] == 1


def test_cross_sequence_verified_cell_gets_local_canonical_fallback():
    payload = _payload()
    # Make both persistent cells part of one tree. Group zero selects seq0 as
    # the tree snapshot, while group one is observed only in seq1/seq2. It is
    # independently cross-sequence verified and must not disappear merely
    # because the tree-wide snapshot did not see it.
    payload["tree_instance_id"][1] = 0
    payload["tree_instance_id"][5:7] = 0
    payload["support_camera_ids"][5, 0] = 2
    payload["support_camera_ids"][6, 0] = 3
    payload["observation_camera_ids"][5, 0] = 2
    payload["observation_camera_ids"][6, 0] = 3
    fused, audit = fuse_sequence_evidence_into_static_leaves(
        payload,
        canonical_sequence_policy="per_tree",
        fixed_camera_sequences=[
            {"image_id": 0, "sequence_id": "seq0"},
            {"image_id": 1, "sequence_id": "seq0"},
            {"image_id": 2, "sequence_id": "seq1"},
            {"image_id": 3, "sequence_id": "seq2"},
        ],
    )
    assert audit["canonical_cross_sequence_fallback_groups"] == 1
    detail_groups = fused["replacement_group"][fused["static_detail"]]
    assert 0 in detail_groups.tolist()
    assert 1 in detail_groups.tolist()
    group_one = fused["static_detail"] & (
        fused["replacement_group"] == 1
    )
    # Equal camera support and posterior mass are resolved by the stable
    # lower sequence id, exactly matching the production lexicographic
    # contract after vectorizing the dense fallback selection.
    assert torch.allclose(
        fused["centers"][group_one][0],
        torch.tensor([1.01, 0.00, 3.00]),
    )


def test_single_view_cell_missing_from_tree_snapshot_remains_static_occupancy():
    payload = _payload()
    # Group zero makes seq0 the coherent tree snapshot.  Group one belongs to
    # the same tree but is visible only from camera 2 in seq1.  A moving leaf
    # cannot be expected to match a second metric voxel, so the calibrated
    # single-view cell must survive with low authority rather than disappear.
    payload["tree_instance_id"][1] = 0
    payload["tree_instance_id"][5:7] = 0
    payload["support_camera_ids"][5:7, 0] = 2
    payload["observation_camera_ids"][5:7, 0] = 2
    fused, audit = fuse_sequence_evidence_into_static_leaves(
        payload,
        canonical_sequence_policy="per_tree",
        fixed_camera_sequences=[
            {"image_id": 0, "sequence_id": "seq0"},
            {"image_id": 1, "sequence_id": "seq0"},
            {"image_id": 2, "sequence_id": "seq1"},
        ],
    )
    group_one = fused["static_detail"] & (
        fused["replacement_group"] == 1
    )
    assert int(group_one.sum()) == 1
    assert fused["support_camera_ids"][group_one][0, 0] == 2
    assert int(fused["verified_camera_count"][group_one][0]) == 1
    assert float(fused["opacities"][group_one][0]) > 0.0
    assert audit["single_view_candidate_groups"] >= 1
    assert audit["canonical_local_fallback_groups"] >= 1
    assert audit["canonical_cross_sequence_fallback_groups"] == 0


def test_canonical_camera_quality_breaks_equal_support_tie():
    payload = _payload()
    payload["audit"]["fixed_camera_sequences"] = [
        {
            "image_id": 0,
            "sequence_id": "seq0",
            "canonical_quality": 0.25,
        },
        {
            "image_id": 1,
            "sequence_id": "seq1",
            "canonical_quality": 2.0,
        },
    ]
    fused, audit = fuse_sequence_evidence_into_static_leaves(
        payload, canonical_sequence_policy="per_tree"
    )
    detail = fused["static_detail"] & (
        fused["replacement_group"] == 0
    )
    assert torch.allclose(
        fused["centers"][detail][0], torch.tensor([-0.01, 0.00, 3.02])
    )
    assert audit["canonical_sequence_histogram"] == {
        "seq0": 1,
        "seq1": 1,
    }
    assert audit["canonical_camera_quality_source"] == (
        "initialization_fixed_camera_contract"
    )


def test_scene_snapshot_is_static_production_default():
    _, production = fuse_sequence_evidence_into_static_leaves(_payload())
    _, mosaic = fuse_sequence_evidence_into_static_leaves(
        _payload(), canonical_sequence_policy="per_tree"
    )
    assert production["canonical_scene_sequence"] == "seq0"
    assert production["canonical_sequence_histogram"] == {"seq0": 2}
    assert production["canonical_sequence_policy"] == "scene"
    assert mosaic["canonical_scene_sequence"] is None
    assert mosaic["canonical_sequence_policy"] == "per_tree"
    assert "per_tree_canonical_sequence" in mosaic["contract"]


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


def test_split_children_drop_parent_proof_and_keep_mass_and_lineage():
    payload, _ = fuse_sequence_evidence_into_static_leaves(_payload())
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)
    parent = torch.nonzero(
        foliage.persistent_envelope_mask, as_tuple=False
    ).flatten()[:1]
    foliage.features.data[parent, 1:] = 2.0
    mass_before = foliage.integrated_optical_mass().sum()
    parent_lineage = int(foliage.lineage_id[parent])
    foliage.split(parent, birth_iteration=17)
    children = foliage.parent_lineage_id == parent_lineage
    assert int(children.sum()) == 2
    assert torch.all(
        foliage.verification_state[children] == VERIFICATION_UNVERIFIED
    )
    assert torch.all(foliage.birth_iteration[children] == 17)
    assert torch.all(foliage.evidence_primitive_id[children] == -1)
    assert torch.all(foliage.features[children, 1:] == 0)
    assert torch.allclose(
        foliage.integrated_optical_mass().sum(),
        mass_before,
        rtol=2e-5,
        atol=2e-7,
    )


def test_unresolved_split_sparse_parent_snapshot_survives_checkpoint_restore():
    payload, _ = fuse_sequence_evidence_into_static_leaves(_payload())
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)
    parent = torch.nonzero(
        foliage.persistent_envelope_mask, as_tuple=False
    ).flatten()[:1]
    parent_support = foliage.support_camera_ids[parent].clone()
    foliage.split(parent, birth_iteration=17)

    state = foliage.capture()
    restored = VolumetricFoliageModel(1, device="cpu")
    restored.restore(state)

    assert int((restored.proposal_kind == PROPOSAL_SPLIT).sum()) == 2
    assert len(restored.split_parent_snapshot_family_id) == 1
    assert restored.proposal_parent_support_camera_ids.shape[0] == 1
    torch.testing.assert_close(
        restored.proposal_parent_support_camera_ids,
        parent_support,
    )
    assert (
        int(restored.next_split_proposal_family_id)
        > int(restored.split_parent_snapshot_family_id.max())
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


def test_scene_snapshot_birth_accepts_two_cameras_in_canonical_sequence():
    accumulator = StaticRayBirthAccumulator(voxel_size=0.15)
    for camera_id, center in (
        (0, [2.00, 0.00, 3.00]),
        (1, [2.03, 0.02, 3.01]),
    ):
        accumulator.add(
            {
                "centers": torch.tensor([center]),
                "colors": torch.tensor([[0.10, 0.20, 0.30]]),
                "confidence": torch.tensor([0.8]),
                "camera_id": camera_id,
            },
            sequence_id="seq0",
        )
    centers, _, cameras, sequences, audit = accumulator.drain(
        maximum_births=8,
        minimum_cameras=2,
        minimum_sequences=1,
    )
    assert len(centers) == 1
    assert int((cameras[0] >= 0).sum()) == 2
    assert int(sequences[0]) == 1
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
    consumed_before = set(accumulator.consumed_visual_hull_cells)

    # A later pair of cameras traversing the same occupied cells must refine
    # the extant model, not recreate another alpha owner in every drained
    # visual-hull voxel.
    add_segment(
        center=[0.0, 0.0, 3.0],
        origin=[-1.0, 0.0, 3.0],
        direction=[1.0, 0.0, 0.0],
        hit_start=0.5,
        hit_end=1.5,
        camera=2,
        sequence="seq2",
    )
    add_segment(
        center=[0.0, 0.3, 3.0],
        origin=[0.0, -1.0, 3.0],
        direction=[0.0, 1.0, 0.0],
        hit_start=0.9,
        hit_end=1.7,
        camera=3,
        sequence="seq3",
    )
    repeated_centers = accumulator.drain(maximum_births=8)[0]
    repeated_keys = {
        tuple(int(value) for value in row.tolist())
        for row in torch.floor(
            repeated_centers / accumulator.visual_hull_voxel_size
        ).to(torch.int64)
    }
    assert repeated_keys.isdisjoint(consumed_before)
    assert accumulator.suppressed_consumed_cells > 0


def test_consumed_ray_birth_cells_survive_capture_and_legacy_recovery():
    producer = StaticRayBirthAccumulator(voxel_size=0.15)

    def add(camera, sequence):
        producer.add(
            {
                "centers": torch.tensor([[0.01, 0.01, 3.01]]),
                "confidence": torch.tensor([0.9]),
                "camera_id": camera,
            },
            sequence_id=sequence,
        )

    add(0, "seq0")
    add(1, "seq1")
    assert len(producer.drain(maximum_births=1)[0]) == 1
    state = producer.capture()
    assert state["version"] == "static-ray-birth-accumulator-v5"

    restored = StaticRayBirthAccumulator(voxel_size=0.15)
    restored.restore(state)
    restored.add(
        {
            "centers": torch.tensor([[0.01, 0.01, 3.01]]),
            "confidence": torch.tensor([0.9]),
            "camera_id": 2,
        },
        sequence_id="seq2",
    )
    restored.add(
        {
            "centers": torch.tensor([[0.01, 0.01, 3.01]]),
            "confidence": torch.tensor([0.9]),
            "camera_id": 3,
        },
        sequence_id="seq3",
    )
    assert len(restored.drain(maximum_births=1)[0]) == 0

    # Pre-v5 checkpoints have no tombstone list. Runtime model provenance
    # reconstructs both grids before any new proposal is accumulated.
    legacy = dict(state)
    legacy["version"] = "static-ray-birth-accumulator-v4"
    legacy.pop("consumed_midpoint_cells")
    legacy.pop("consumed_visual_hull_cells")
    recovered = StaticRayBirthAccumulator(voxel_size=0.15)
    recovered.restore(legacy)
    assert recovered.mark_consumed_centers(
        torch.tensor([[0.01, 0.01, 3.01]])
    ) == 2
    recovered.add(
        {
            "centers": torch.tensor([[0.01, 0.01, 3.01]]),
            "confidence": torch.tensor([0.9]),
            "camera_id": 4,
        },
        sequence_id="seq4",
    )
    assert len(recovered.cells) == 0


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
    # Give the nearest structural skeleton its own tree identity without an
    # envelope. Runtime births inherit tree identity from the nearest static
    # scaffold; absence of an envelope in that same tree must leave the birth
    # unassociated.
    skeleton = torch.nonzero(
        foliage.layer_role == LAYER_STATIC_SKELETON,
        as_tuple=False,
    ).flatten()
    assert len(skeleton) == 1
    foliage.tree_instance_id[skeleton] = 99
    event = foliage.append_static_ray_births(
        foliage.xyz[skeleton].detach().clone()
    )
    assert foliage.replacement_group[-1] == -1
    assert event["replacement_group_association"]["assigned"] == 0


def test_missing_static_detail_receiver_is_exact_mass_factorization():
    payload, _ = fuse_sequence_evidence_into_static_leaves(_payload())
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)
    # Remove the existing group-1 detail to reproduce an envelope cell whose
    # low-frequency owner has no local high-frequency receiver.
    remove = foliage.static_leaf_mask & (foliage.replacement_group == 1)
    foliage.prune(remove)
    parent = torch.nonzero(
        foliage.persistent_envelope_mask
        & (foliage.replacement_group == 1),
        as_tuple=False,
    ).flatten()
    assert len(parent) == 1
    before_count = len(foliage)
    before_mass = foliage.integrated_optical_mass().clone()
    before_tau = -torch.log1p(-foliage.opacities[parent])
    before_xyz = foliage.xyz[parent].clone()
    before_scale = foliage.scales[parent].clone()
    before_color = foliage.features[parent].clone()

    event = foliage.materialize_static_detail_receivers(
        parent,
        optical_mass_fraction=0.05,
        birth_iteration=123,
    )
    child = torch.tensor([before_count])
    assert event["materialized"] == 1
    assert len(foliage) == before_count + 1
    assert bool(foliage.persistent_envelope_mask[parent])
    assert bool(foliage.static_leaf_mask[child])
    assert foliage.replacement_group[child] == foliage.replacement_group[parent]
    torch.testing.assert_close(foliage.xyz[child], before_xyz)
    torch.testing.assert_close(foliage.scales[child], before_scale)
    torch.testing.assert_close(foliage.features[child], before_color)
    after_tau = -torch.log1p(
        -foliage.opacities[torch.cat([parent, child])]
    )
    torch.testing.assert_close(after_tau.sum(), before_tau.sum())
    torch.testing.assert_close(
        foliage.integrated_optical_mass().sum(), before_mass.sum()
    )
    assert foliage.birth_iteration[child] == 123
    assert torch.equal(
        foliage.support_camera_ids[child], foliage.support_camera_ids[parent]
    )


def test_late_detail_receiver_preserves_envelope_handoff_history():
    payload, _ = fuse_sequence_evidence_into_static_leaves(_payload())
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)
    remove = foliage.static_leaf_mask & (foliage.replacement_group == 1)
    foliage.prune(remove)
    parent = torch.nonzero(
        foliage.persistent_envelope_mask
        & (foliage.replacement_group == 1),
        as_tuple=False,
    ).flatten()
    assert len(parent) == 1
    current_mass = foliage.integrated_optical_mass()[parent].clone()
    foliage.handoff_retired_fraction[parent] = 0.4
    reference = current_mass / 0.6
    foliage.handoff_reference_mass[parent] = reference
    foliage.replacement_overlap_ema[parent] = 0.7
    foliage.replacement_observation_count[parent] = 5
    foliage.replacement_camera_signature[parent] = 11

    event = foliage.materialize_static_detail_receivers(
        parent, optical_mass_fraction=0.05, birth_iteration=123
    )
    child = torch.tensor([int(event["_new_start"])])

    assert event["late_handoff_history_preserved"]
    torch.testing.assert_close(
        foliage.handoff_retired_fraction[parent], torch.tensor([0.4])
    )
    torch.testing.assert_close(
        foliage.handoff_reference_mass[parent], 0.95 * reference
    )
    torch.testing.assert_close(
        foliage.integrated_optical_mass()[parent], 0.95 * current_mass
    )
    torch.testing.assert_close(
        foliage.integrated_optical_mass()[child], 0.05 * current_mass
    )
    torch.testing.assert_close(
        foliage.replacement_overlap_ema[parent], torch.tensor([0.7])
    )
    assert foliage.replacement_observation_count[parent].item() == 5
    assert foliage.replacement_camera_signature[parent].item() == 11
    assert foliage.handoff_retired_fraction[child].item() == 0.0
    assert foliage.replacement_observation_count[child].item() == 0


def test_static_detail_receiver_cannot_duplicate_existing_group():
    payload, _ = fuse_sequence_evidence_into_static_leaves(_payload())
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)
    parent = torch.nonzero(
        foliage.persistent_envelope_mask
        & (foliage.replacement_group == 0),
        as_tuple=False,
    ).flatten()
    with pytest.raises(ValueError, match="already has a receiver"):
        foliage.materialize_static_detail_receivers(parent)


def test_bounded_nearest_reference_rows_matches_full_cdist(monkeypatch):
    query = torch.tensor(
        [[0.1, 0.0, 0.0], [2.2, 0.0, 0.0], [5.0, 0.0, 0.0]]
    )
    reference = torch.tensor(
        [
            [9.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [5.5, 0.0, 0.0],
            [5.5, 0.0, 0.0],
        ]
    )
    rows = torch.tensor([4, 2, 1, 3])
    expected_distance, expected_local = torch.cdist(
        query, reference[rows]
    ).min(dim=1)
    expected_rows = rows[expected_local]

    original_cdist = torch.cdist
    pair_counts = []

    def audited_cdist(first, second, *args, **kwargs):
        pair_counts.append(len(first) * len(second))
        return original_cdist(first, second, *args, **kwargs)

    monkeypatch.setattr(torch, "cdist", audited_cdist)
    distance, nearest_rows = (
        VolumetricFoliageModel._nearest_reference_rows_bounded(
            query,
            reference,
            rows,
            maximum_pairwise_entries=6,
        )
    )
    torch.testing.assert_close(distance, expected_distance)
    assert torch.equal(nearest_rows, expected_rows)
    assert max(pair_counts) <= 6
    # Rows 4 and 3 are tied for the last query. The original first-row rule
    # must survive reference tiling.
    assert nearest_rows[-1].item() == 4
