import torch

from outdoor.hybrid_gaussian_renderer import projected_gaussian_cross_section
from outdoor.static_foliage import (
    fuse_sequence_evidence_into_static_leaves,
    _initialize_static_detail_group_mass_handoff,
)


def _mass(result):
    scales = torch.as_tensor(result["scales"]).float()
    opacity = torch.as_tensor(result["opacities"]).float().reshape(-1)
    tau = -torch.log1p(-opacity.clamp(0.0, 1.0 - 1.0e-6))
    return tau * projected_gaussian_cross_section(scales)


def _cross_sequence_fusion_payload(second_center=(0.03, 0.02, 3.01)):
    centers = torch.tensor(
        [[0.0, 0.0, 3.0], [0.02, 0.01, 3.0], second_center],
        dtype=torch.float32,
    )
    support = torch.tensor([[-1, -1], [0, -1], [1, -1]], dtype=torch.int32)
    return {
        "centers": centers,
        "colors": torch.tensor(
            [[0.2, 0.4, 0.2], [0.18, 0.5, 0.18], [0.2, 0.48, 0.2]]
        ),
        "scales": torch.tensor(
            [[0.08, 0.05, 0.025], [0.03, 0.02, 0.01], [0.03, 0.02, 0.01]]
        ),
        "opacities": torch.full((3, 1), 0.05),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        "layer_role": torch.tensor([0, 2, 2], dtype=torch.int8),
        "replacement_group": torch.tensor([0, 0, 0], dtype=torch.int64),
        "tree_instance_id": torch.zeros(3, dtype=torch.int32),
        "support_camera_ids": support,
        "observation_camera_ids": support.clone(),
        "support_view_count": torch.tensor([0, 1, 1], dtype=torch.int16),
        "support_sequence_count": torch.tensor([0, 1, 1], dtype=torch.int16),
        "occupancy_probability": torch.ones(3),
        "audit": {
            "selected_views": [
                {"image_id": 0, "sequence_id": "seq0"},
                {"image_id": 1, "sequence_id": "seq1"},
            ]
        },
    }


def test_verified_anisotropic_multimode_detail_conserves_group_mass():
    # Rows 0/1 are the two persistent envelopes. Rows 2/3 are differently
    # shaped verified modes in group zero; row 4 is an unverified hypothesis in
    # group one and therefore has no retirement authority.
    result = {
        "scales": torch.tensor(
            [
                [0.40, 0.20, 0.05],
                [0.30, 0.12, 0.04],
                [0.18, 0.06, 0.025],
                [0.09, 0.10, 0.020],
                [0.12, 0.05, 0.015],
            ]
        ),
        "opacities": torch.tensor([[0.08], [0.07], [0.10], [0.04], [0.10]]),
    }
    before = _mass(result)
    requested_ratio = before[2] / before[3]

    audit = _initialize_static_detail_group_mass_handoff(
        result,
        envelope_rows=torch.tensor([0, 1]),
        detail_rows=torch.tensor([2, 3, 4]),
        detail_groups=torch.tensor([0, 0, 1]),
        detail_verified=torch.tensor([True, True, False]),
        detail_retirement_authorized=torch.tensor([True, True, False]),
        detail_support_camera_ids=torch.tensor(
            [[7, 8, 9, -1], [8, 9, 10, -1], [11, -1, -1, -1]]
        ),
        detail_verified_sequence_count=torch.tensor([2, 2, 1]),
        minimum_envelope_retained_fraction=0.20,
    )
    after = _mass(result)

    torch.testing.assert_close(after[0] + after[2] + after[3], before[0])
    torch.testing.assert_close(after[1] + after[4], before[1])
    assert after[0] >= 0.20 * before[0]
    torch.testing.assert_close(after[2] / after[3], requested_ratio)
    assert after[4] == 0
    assert audit["maximum_group_mass_increase"] == 0


def test_initial_handoff_metadata_is_reversible_and_evidence_owned():
    result = {
        "scales": torch.tensor(
            [
                [0.30, 0.16, 0.04],
                [0.11, 0.07, 0.025],
                [0.08, 0.06, 0.020],
            ]
        ),
        "opacities": torch.tensor([[0.06], [0.08], [0.08]]),
    }
    before = _mass(result)
    _initialize_static_detail_group_mass_handoff(
        result,
        envelope_rows=torch.tensor([0]),
        detail_rows=torch.tensor([1, 2]),
        detail_groups=torch.tensor([0, 0]),
        detail_verified=torch.tensor([True, True]),
        detail_retirement_authorized=torch.tensor([True, True]),
        detail_support_camera_ids=torch.tensor([[3, 4], [5, 6]]),
        detail_verified_sequence_count=torch.tensor([2, 1]),
    )
    after = _mass(result)

    reference = result["handoff_reference_mass"][0]
    retired = result["handoff_retired_fraction"][0]
    torch.testing.assert_close(reference, before[0])
    torch.testing.assert_close(reference * (1.0 - retired), after[0])
    torch.testing.assert_close(reference, after[0] + after[1] + after[2])

    # All distinct support cameras are real readiness observations. The
    # stored prior reproduces the initialized retirement exactly under the
    # downstream desired = overlap * min(count/3,1) * .95 equation.
    assert result["replacement_observation_count"][0] == 3
    readiness = result["replacement_observation_count"][0].float() / 3.0
    desired = result["replacement_overlap_ema"][0] * readiness * 0.95
    torch.testing.assert_close(desired, retired)
    assert result["replacement_camera_signature"][0] != 0

    # Same-sequence multiview evidence receives a bounded 40% group tier. It
    # gains optical leverage without adding mass; cross-sequence evidence in
    # the same group permits the larger tier shared proportionally by modes.
    assert after[2] > 0
    assert result["handoff_retired_fraction"][2] == 0
    assert result["replacement_observation_count"][2] == 0


def test_fusion_separates_canonical_appearance_from_cross_sequence_witness():
    original = _cross_sequence_fusion_payload()
    original_envelope_mass = _mass(original)[0]
    fused, audit = fuse_sequence_evidence_into_static_leaves(original)
    detail = fused["static_detail"] & (fused["replacement_group"] == 0)
    envelope = ~fused["static_detail"] & (fused["replacement_group"] == 0)

    assert int(detail.sum()) == 1
    # Appearance is owned only by the selected canonical snapshot.
    assert fused["support_camera_ids"][detail][0].tolist() == [0, -1]
    assert int(fused["support_sequence_count"][detail][0]) == 1
    # Geometry verification comes from the same metric cell in seq0 and seq1.
    assert fused["observation_camera_ids"][detail][0].tolist() == [0, 1]
    # Positive RGB/refinement ownership remains the canonical appearance
    # snapshot; cross-sequence cameras are a separate handoff witness table.
    assert int(fused["verified_camera_count"][detail][0]) == 1
    assert int(fused["verified_sequence_count"][detail][0]) == 2
    assert int(fused["verification_state"][detail][0]) == 1
    assert audit["cross_sequence_geometry_verified_modes"] == 1

    fused_mass = _mass(fused)
    torch.testing.assert_close(
        fused_mass[detail].sum() + fused_mass[envelope].sum(),
        original_envelope_mass,
    )
    assert fused["handoff_retired_fraction"][envelope][0] > 0


def test_parent_cross_sequence_count_does_not_verify_a_different_detail_cell():
    original = _cross_sequence_fusion_payload(
        second_center=(0.18, 0.02, 3.01)
    )
    original_envelope_mass = _mass(original)[0]
    fused, audit = fuse_sequence_evidence_into_static_leaves(original)
    detail = fused["static_detail"] & (fused["replacement_group"] == 0)
    envelope = ~fused["static_detail"] & (fused["replacement_group"] == 0)

    assert int(fused["support_sequence_count"][detail][0]) == 1
    assert int(fused["verified_sequence_count"][detail][0]) == 1
    # The calibrated canonical seed remains trainable even though the parent
    # traversal does not verify this exact leaf cell. It may be born as a
    # bounded one-camera occupancy hypothesis but cannot retire envelope mass
    # or use multiview refinement.
    assert int(fused["verification_state"][detail][0]) == 1
    assert int(fused["verified_camera_count"][detail][0]) == 1
    assert float(fused["opacities"][detail][0]) > 0.0
    fused_mass = _mass(fused)
    torch.testing.assert_close(
        fused_mass[envelope].sum(), original_envelope_mass
    )
    assert 0.0 < float(fused_mass[detail].sum()) <= float(
        0.05 * original_envelope_mass + 1.0e-7
    )
    assert float(fused["handoff_retired_fraction"][envelope][0]) == 0.0
    assert audit["initial_mass_additive_only_modes"] == 1
    assert audit["cross_sequence_geometry_verified_modes"] == 0
