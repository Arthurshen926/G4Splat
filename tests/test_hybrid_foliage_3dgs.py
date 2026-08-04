from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from outdoor.hybrid_gaussian_renderer import (
    LAYER_CANONICAL_CROWN,
    LAYER_DYNAMIC_LEAF,
    VolumetricFoliageModel,
    _identity_with_gradient_gate,
    _restore_sparse_volume_rows,
    dynamic_visibility_gate,
    view_depth_local_optical_replacement,
)


def test_view_depth_local_replacement_retires_same_ray_occluded_envelope():
    roles = torch.tensor(
        [
            LAYER_CANONICAL_CROWN,
            LAYER_CANONICAL_CROWN,
            LAYER_CANONICAL_CROWN,
            LAYER_DYNAMIC_LEAF,
        ],
        dtype=torch.int8,
    )
    suppression = view_depth_local_optical_replacement(
        roles,
        torch.zeros(4, dtype=torch.int64),
        torch.full((4,), 0.2),
        torch.full((4, 3), 0.05),
        torch.tensor(
            [[10.0, 10.0], [110.0, 10.0], [10.0, 10.0], [10.0, 10.0]]
        ),
        torch.tensor([5.0, 5.0, 8.0, 5.0]),
        focal_x=100.0,
        focal_y=100.0,
        cross_section=torch.full((4,), 0.01),
    )
    assert suppression[0] > 0.1
    assert suppression[1] < 1.0e-3
    # The exact detail first hit at z=5 owns transmittance behind it; keeping
    # the same-ray visual-hull sample at z=8 would double that optical mass.
    assert suppression[2] > 0.1
    assert suppression[3] == 0


def test_view_depth_local_replacement_uses_projected_depth_covariance():
    roles = torch.tensor(
        [LAYER_CANONICAL_CROWN, LAYER_DYNAMIC_LEAF], dtype=torch.int8
    )
    suppression = view_depth_local_optical_replacement(
        roles,
        torch.zeros(2, dtype=torch.int64),
        torch.full((2,), 0.2),
        # Broad image-plane support must not be mistaken for depth support.
        torch.tensor([[2.0, 2.0, 0.01], [2.0, 2.0, 0.01]]),
        torch.tensor([[10.0, 10.0], [10.0, 10.0]]),
        torch.tensor([5.0, 6.0]),
        focal_x=100.0,
        focal_y=100.0,
        cross_section=torch.full((2,), 0.01),
        depth_radius=torch.full((2,), 0.01),
    )
    assert suppression[0] < 1.0e-6


def test_dynamic_leaf_inherits_shared_canonical_group_displacement():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor(
            [[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]]
        ),
        "scales": torch.full((2, 3), 0.05),
        "colors": torch.full((2, 3), 0.4),
        "opacities": torch.full((2, 1), 0.05),
        "quaternions": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 2
        ),
        "layer_role": torch.tensor(
            [LAYER_CANONICAL_CROWN, LAYER_DYNAMIC_LEAF],
            dtype=torch.int8,
        ),
        "tree_instance_id": torch.tensor([3, 3], dtype=torch.int32),
        "replacement_group": torch.tensor([0, 0], dtype=torch.int64),
    }
    model = VolumetricFoliageModel(1, device="cpu")
    model.initialize_from_volume_state(payload)
    with torch.no_grad():
        model.xyz[0, 0] += 0.3
    conditioned, _, _ = model.conditioned_state(
        torch.zeros(model.dynamic_rank), include_dynamic=True
    )
    assert torch.allclose(
        conditioned[1],
        torch.tensor([0.5, 0.0, 2.0]),
        atol=1e-6,
    )


def test_visible_dynamic_temporal_state_deforms_shared_canonical_group():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor(
            [[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]]
        ),
        "scales": torch.full((2, 3), 0.05),
        "colors": torch.full((2, 3), 0.4),
        "opacities": torch.full((2, 1), 0.05),
        "quaternions": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 2
        ),
        "layer_role": torch.tensor(
            [LAYER_CANONICAL_CROWN, LAYER_DYNAMIC_LEAF],
            dtype=torch.int8,
        ),
        "tree_instance_id": torch.tensor([3, 3], dtype=torch.int32),
        "replacement_group": torch.tensor([0, 0], dtype=torch.int64),
    }
    model = VolumetricFoliageModel(1, device="cpu")
    model.initialize_from_volume_state(payload)
    with torch.no_grad():
        model.deformation_basis[1, 0, 0] = 0.25
        model.dynamic_feature_basis[1, 0, 1] = 0.5
        model.dynamic_opacity_basis[1, 0, 0] = 0.75
    code = torch.zeros(model.dynamic_rank)
    code[0] = 1.0

    xyz, features, opacity = model.conditioned_state(
        code,
        include_dynamic=True,
        conditioned_visibility_gate=torch.tensor([1.0, 1.0]),
    )

    assert xyz[0, 0].item() == pytest.approx(0.25)
    assert xyz[1, 0].item() == pytest.approx(0.45)
    assert features[0, 0, 1].item() == pytest.approx(
        model.features[0, 0, 1].item() + 0.5
    )
    assert opacity[0] > model.opacities[0]


def test_gradient_ownership_gate_is_forward_identity_and_backward_selective():
    value = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )
    gate = torch.tensor([1.0, 0.0, 0.25])

    routed = _identity_with_gradient_gate(value, gate)
    assert torch.equal(routed.detach(), value.detach())
    routed.sum().backward()

    assert torch.equal(
        value.grad,
        torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.25, 0.25]]),
    )


def test_dynamic_base_and_temporal_residual_have_independent_owners():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.1, 0.2, 2.0]]),
        "scales": torch.tensor([[0.05, 0.06, 0.07]]),
        "colors": torch.tensor([[0.2, 0.4, 0.6]]),
        "opacities": torch.tensor([[0.05]]),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "layer_role": torch.tensor([LAYER_DYNAMIC_LEAF], dtype=torch.int8),
    }
    model = VolumetricFoliageModel(1, device="cpu")
    model.initialize_from_volume_state(payload)
    code = torch.ones(model.dynamic_rank)
    zero = torch.zeros(1)
    one = torch.ones(1)

    xyz, features, opacity = model.conditioned_state(
        code,
        include_dynamic=True,
        base_geometry_gradient_gate=zero,
        dynamic_geometry_gradient_gate=one,
        base_appearance_gradient_gate=zero,
        dynamic_appearance_gradient_gate=one,
        base_opacity_gradient_gate=zero,
        dynamic_opacity_gradient_gate=one,
    )
    (xyz.sum() + features.sum() + opacity.sum()).backward()

    assert not bool(model.xyz.grad.any())
    assert not bool(model.features.grad.any())
    assert not bool(model.opacity_logits.grad.any())
    assert bool(model.deformation_basis.grad.any())
    assert bool(model.dynamic_feature_basis.grad.any())
    assert bool(model.dynamic_opacity_basis.grad.any())

    model.zero_grad(set_to_none=True)
    xyz, features, opacity = model.conditioned_state(
        code,
        include_dynamic=True,
        base_geometry_gradient_gate=one,
        dynamic_geometry_gradient_gate=zero,
        base_appearance_gradient_gate=one,
        dynamic_appearance_gradient_gate=zero,
        base_opacity_gradient_gate=one,
        dynamic_opacity_gradient_gate=zero,
    )
    (xyz.sum() + features.sum() + opacity.sum()).backward()

    assert bool(model.xyz.grad.any())
    assert bool(model.features.grad.any())
    assert bool(model.opacity_logits.grad.any())
    assert not bool(model.deformation_basis.grad.any())
    assert not bool(model.dynamic_feature_basis.grad.any())
    assert not bool(model.dynamic_opacity_basis.grad.any())


def test_independent_volume_state_does_not_use_legacy_surfel_geometry():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[1.0, 2.0, 3.0]]),
        "scales": torch.tensor([[0.1, 0.2, 0.3]]),
        "colors": torch.tensor([[0.2, 0.4, 0.6]]),
        "opacities": torch.tensor([[0.02]]),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    }
    model = VolumetricFoliageModel(1, device="cpu")

    assert model.initialize_from_volume_state(payload) == 1
    assert torch.equal(model.xyz.detach(), payload["centers"])
    assert torch.allclose(model.scales.detach(), payload["scales"])
    assert torch.allclose(model.opacities.detach(), payload["opacities"])
    assert not hasattr(model, "initialize_from_surfel_residual")
    assert not hasattr(model, "append_from_structural")


def test_dynamic_observation_metadata_survives_split_and_legacy_gate_is_safe():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.1, 0.0, 2.0]]),
        "scales": torch.tensor([[0.1, 0.1, 0.1]]),
        "colors": torch.tensor([[0.2, 0.4, 0.6]]),
        "opacities": torch.tensor([[1e-5]]),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "layer_role": torch.tensor([LAYER_DYNAMIC_LEAF], dtype=torch.int8),
        "support_camera_ids": torch.tensor([[7]], dtype=torch.int32),
        "observation_camera_ids": torch.tensor([[7]], dtype=torch.int32),
        "observation_uv": torch.tensor([[[0.5, 0.5]]]),
        "observation_depth": torch.tensor([[2.0]]),
    }
    model = VolumetricFoliageModel(1, device="cpu")
    model.initialize_from_volume_state(payload)

    gate = dynamic_visibility_gate(model, 7, None, None)
    assert gate.tolist() == [0.0]
    model.split(torch.tensor([0]))
    assert model.observation_camera_ids.tolist() == [[7], [7]]
    assert torch.equal(
        model.observation_uv,
        payload["observation_uv"].repeat_interleave(2, dim=0),
    )
    assert torch.equal(
        model.observation_depth,
        payload["observation_depth"].repeat_interleave(2, dim=0),
    )


def test_exact_tree_support_continuously_suppresses_temporal_fallback():
    exact_count = 512
    class FoliageStub(SimpleNamespace):
        def __len__(self):
            return len(self.xyz)

    foliage = FoliageStub(
        xyz=torch.zeros(exact_count + 1, 3),
        dynamic_leaf_mask=torch.ones(
            exact_count + 1, dtype=torch.bool
        ),
        support_camera_ids=torch.cat(
            [
                torch.full((exact_count, 1), 7, dtype=torch.int32),
                torch.full((1, 1), 8, dtype=torch.int32),
            ]
        ),
        tree_instance_id=torch.zeros(
            exact_count + 1, dtype=torch.int32
        ),
        split_generation=torch.zeros(
            exact_count + 1, dtype=torch.int16
        ),
    )
    sequence = torch.full((10,), -1, dtype=torch.int16)
    sequence[7:9] = 0
    frame = torch.zeros(10, dtype=torch.int32)
    frame[7] = 10
    frame[8] = 11

    gate = dynamic_visibility_gate(
        foliage, 7, sequence, frame
    )

    assert torch.equal(gate[:exact_count], torch.ones(exact_count))
    unsuppressed = 0.35 * torch.exp(torch.tensor(-1.0 / 12.0))
    assert 0 < gate[-1] < 0.5 * unsuppressed


def test_ungrouped_dynamic_leaf_is_visible_only_in_exact_owner_camera():
    class FoliageStub(SimpleNamespace):
        def __len__(self):
            return len(self.xyz)

    foliage = FoliageStub(
        xyz=torch.zeros(2, 3),
        dynamic_leaf_mask=torch.ones(2, dtype=torch.bool),
        support_camera_ids=torch.tensor([[7], [8]], dtype=torch.int32),
        tree_instance_id=torch.zeros(2, dtype=torch.int32),
        split_generation=torch.zeros(2, dtype=torch.int16),
        replacement_group=torch.tensor([-1, -1], dtype=torch.int64),
    )
    sequence = torch.full((10,), -1, dtype=torch.int16)
    sequence[7:9] = 0
    frame = torch.zeros(10, dtype=torch.int32)
    frame[7] = 10
    frame[8] = 11

    gate = dynamic_visibility_gate(foliage, 7, sequence, frame)

    assert gate.tolist() == [1.0, 0.0]


def test_dynamic_leaf_with_retired_canonical_group_becomes_exact_only():
    class FoliageStub(SimpleNamespace):
        def __len__(self):
            return len(self.xyz)

    foliage = FoliageStub(
        xyz=torch.zeros(2, 3),
        dynamic_leaf_mask=torch.tensor([False, True]),
        canonical_crown_mask=torch.tensor([True, False]),
        support_camera_ids=torch.tensor([[-1], [8]], dtype=torch.int32),
        tree_instance_id=torch.zeros(2, dtype=torch.int32),
        split_generation=torch.zeros(2, dtype=torch.int16),
        # Group 0 still has a canonical row; group 1 was retired.
        replacement_group=torch.tensor([0, 1], dtype=torch.int64),
    )
    sequence = torch.full((10,), -1, dtype=torch.int16)
    sequence[7:9] = 0
    frame = torch.zeros(10, dtype=torch.int32)
    frame[7] = 10
    frame[8] = 11

    gate = dynamic_visibility_gate(foliage, 7, sequence, frame)

    assert gate.tolist() == [1.0, 0.0]


def test_dynamic_visibility_support_reduction_is_chunk_invariant():
    class FoliageStub(SimpleNamespace):
        def __len__(self):
            return len(self.xyz)

    foliage = FoliageStub(
        xyz=torch.zeros(7, 3),
        dynamic_leaf_mask=torch.tensor(
            [True, True, True, True, False, True, True]
        ),
        support_camera_ids=torch.tensor(
            [
                [7, -1, -1],
                [8, 9, -1],
                [6, 8, 20],
                [-1, -1, -1],
                [7, 8, 9],
                [0, 1, 2],
                [9, 7, -1],
            ],
            dtype=torch.int32,
        ),
        tree_instance_id=torch.tensor(
            [0, 0, 0, 1, 1, 2, 0], dtype=torch.int32
        ),
        split_generation=torch.tensor(
            [0, 1, 2, 0, 0, 3, 1], dtype=torch.int16
        ),
    )
    sequence = torch.tensor(
        [2, 2, 2, -1, -1, -1, 0, 0, 0, 0],
        dtype=torch.int16,
    )
    frame = torch.tensor(
        [1, 8, 20, 0, 0, 0, 2, 10, 12, 30],
        dtype=torch.int32,
    )

    unchunked = dynamic_visibility_gate(
        foliage, 7, sequence, frame, support_chunk_size=100
    )
    one_row_chunks = dynamic_visibility_gate(
        foliage, 7, sequence, frame, support_chunk_size=1
    )
    uneven_chunks = dynamic_visibility_gate(
        foliage, 7, sequence, frame, support_chunk_size=3
    )

    torch.testing.assert_close(one_row_chunks, unchunked)
    torch.testing.assert_close(uneven_chunks, unchunked)
    with pytest.raises(ValueError, match="support_chunk_size"):
        dynamic_visibility_gate(
            foliage, 7, sequence, frame, support_chunk_size=0
        )


def test_sparse_volume_rows_restore_full_topology_order():
    compact = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [31.0, 32.0], [41.0, 42.0]]
    )

    restored = _restore_sparse_volume_rows(
        compact,
        structural_count=2,
        active_volume_indices=torch.tensor([1, 3]),
        volume_count=4,
    )

    assert restored.tolist() == [
        [10.0, 11.0],
        [20.0, 21.0],
        [0.0, 0.0],
        [31.0, 32.0],
        [0.0, 0.0],
        [41.0, 42.0],
    ]


def test_volume_split_preserves_integrated_projected_optical_depth():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.0, 0.0, 2.0]]),
        "scales": torch.tensor([[0.12, 0.20, 0.30]]),
        "colors": torch.tensor([[0.2, 0.4, 0.6]]),
        "opacities": torch.tensor([[0.08]]),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    }
    model = VolumetricFoliageModel(1, device="cpu")
    model.initialize_from_volume_state(payload)
    before_tau = -torch.log1p(-model.opacities)
    before_mass = before_tau * model.scales[:, 0] * model.scales[:, 1]
    before_opacity = model.opacities.detach().clone()

    model.split(torch.tensor([0]))

    after_tau = -torch.log1p(-model.opacities)
    after_mass = (
        after_tau * model.scales[:, 0] * model.scales[:, 1]
    ).sum()
    torch.testing.assert_close(after_mass, before_mass.sum())
    # The default two-way split must not create opacity above the parent:
    # role ceilings applied by the trainer would otherwise destroy the mass
    # that this split just conserved.
    torch.testing.assert_close(
        model.opacities,
        before_opacity.repeat_interleave(2, dim=0),
    )


def test_volume_split_preserves_consistent_evidence_mass_and_generation():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.0, 0.0, 2.0]]),
        "scales": torch.tensor([[0.12, 0.20, 0.30]]),
        "colors": torch.tensor([[0.2, 0.4, 0.6]]),
        "opacities": torch.tensor([[0.08]]),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "support_view_count": torch.tensor([3], dtype=torch.int16),
        "support_sequence_count": torch.tensor([2], dtype=torch.int16),
    }
    model = VolumetricFoliageModel(1, device="cpu")
    model.initialize_from_volume_state(payload)

    model.split(torch.tensor([0]))

    assert model.support_view_count.tolist() == [1, 1]
    assert model.support_sequence_count.tolist() == [1, 1]
    assert model.split_generation.tolist() == [1, 1]
    assert bool(
        (
            model.support_sequence_count
            <= model.support_view_count
        ).all()
    )


def test_dense_ray_bandwidth_ceiling_is_inherited_by_split():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.0, 0.0, 2.0]]),
        "scales": torch.tensor([[0.06, 0.08, 0.10]]),
        "scale_ceiling": torch.tensor([[0.066, 0.088, 0.11]]),
        "colors": torch.tensor([[0.2, 0.4, 0.6]]),
        "opacities": torch.tensor([[0.02]]),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "initialization_source": torch.tensor([4], dtype=torch.int8),
    }
    model = VolumetricFoliageModel(1, device="cpu")
    model.initialize_from_volume_state(payload)
    before = model.scale_ceiling.clone()

    model.split(torch.tensor([0]))

    expected = (before / (2.0**0.5)).repeat_interleave(2, dim=0)
    torch.testing.assert_close(model.scale_ceiling, expected)
    assert torch.isfinite(model.scale_ceiling).all()


def test_ray_evidence_identity_survives_split_but_not_dynamic_clone():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor(
            [[0.0, 0.0, 2.0], [0.4, 0.0, 2.0]]
        ),
        "scales": torch.full((2, 3), 0.1),
        "colors": torch.tensor(
            [[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]]
        ),
        "opacities": torch.full((2, 1), 0.02),
        "quaternions": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 2
        ),
        # Prefix sums retain the fact that primitive zero has two measured
        # rays and primitive one has one.  The model needs only the source
        # identities; the sparse measurements stay in FoliageRayEvidence.
        "ray_evidence": {
            "offsets": torch.tensor([0, 2, 3], dtype=torch.int64)
        },
    }
    model = VolumetricFoliageModel(1, device="cpu")
    model.initialize_from_volume_state(payload)

    assert model.evidence_primitive_id.tolist() == [0, 1]
    model.split(torch.tensor([0]))
    # The unsplit row keeps verified evidence. Displaced children keep source
    # zero only as a ray re-verification candidate, never as inherited proof.
    assert model.evidence_primitive_id.tolist() == [1, -1, -1]
    assert model.candidate_evidence_primitive_id.tolist() == [1, 0, 0]

    model.append_dynamic_leaves(torch.tensor([1]))
    assert model.evidence_primitive_id.tolist() == [1, -1, -1, -1]
    assert model.candidate_evidence_primitive_id.tolist() == [1, 0, 0, -1]
    assert model.dynamic_leaf_mask.tolist() == [False, False, False, True]


def test_hybrid_contract_uses_one_rasterizer_and_no_fixed_branch_composite():
    root = Path(__file__).resolve().parents[1]
    renderer = (root / "outdoor/hybrid_gaussian_renderer.py").read_text()
    trainer = (root / "scripts/train_hybrid_foliage_3dgs.py").read_text()
    binding = (
        root
        / "2d-gaussian-splatting/submodules/diff-surfel-rasterization"
        / "diff_surfel_rasterization/__init__.py"
    ).read_text()
    forward = (
        root
        / "2d-gaussian-splatting/submodules/diff-surfel-rasterization"
        / "cuda_rasterizer/forward.cu"
    ).read_text()
    assert renderer.count("MixedGaussianRasterizer(settings)") == 1
    assert "volume_geometry_gradient_gate" in renderer
    assert "volume_appearance_gradient_gate" in renderer
    assert "volume_opacity_gradient_gate" in renderer
    assert "volume_dynamic_geometry_gradient_gate" in renderer
    assert "volume_dynamic_appearance_gradient_gate" in renderer
    assert "volume_dynamic_opacity_gradient_gate" in renderer
    assert (
        "self.xyz, base_geometry_gradient_gate" in renderer
    )
    assert (
        "self.features, base_appearance_gradient_gate" in renderer
    )
    assert (
        "self.opacity_logits.squeeze(-1)" in renderer
    )
    assert "rasterize_mixed_gaussians" in binding
    assert "mixedRenderCUDA" in forward
    assert "id < surface_count" in forward
    assert "if (!(opacities[local_idx] > 0.0f))" in forward
    assert "if (!isfinite(distance) || !(distance < 0.0f))" in forward
    assert "native_2d_surfel_and_3d_ewa_shared_tile_depth_sort" in trainer
    assert '"fixed_order_branch_compositing": False' in trainer
    assert '"true_volumetric_foliage_3dgs": True' in trainer
    assert "historical_full_real_rgb_initialization" in trainer
    assert "rejected_zero_step_identity.json" in trainer
    assert "Replacement audit contains no candidates" in trainer
    assert "retirement_eligible_indices" in trainer
    assert '"three_stage_replace_and_retire": True' in trainer


def test_legacy_canopy_repair_only_overrides_opacity():
    root = Path(__file__).resolve().parents[1]
    renderer = (
        root / "2d-gaussian-splatting/gaussian_renderer/__init__.py"
    ).read_text()
    trainer = (
        root / "scripts/train_legacy_canopy_attenuation_2dgs.py"
    ).read_text()
    assert "opacity_override=None" in renderer
    assert '"structural_geometry_changed": False' in trainer
    assert '"new_gaussians_changed": False' in trainer
    assert "minimum_canopy_views" in trainer
