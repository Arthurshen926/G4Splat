from pathlib import Path

import torch

from outdoor.hybrid_gaussian_renderer import (
    LAYER_DYNAMIC_LEAF,
    VolumetricFoliageModel,
    dynamic_visibility_gate,
)


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
    assert "rasterize_mixed_gaussians" in binding
    assert "mixedRenderCUDA" in forward
    assert "id < surface_count" in forward
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
