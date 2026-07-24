from pathlib import Path
from types import SimpleNamespace

import torch

from outdoor.hybrid_gaussian_renderer import VolumetricFoliageModel


def test_surfel_residual_is_lifted_to_three_axis_volume():
    surfels = SimpleNamespace(
        get_scaling=torch.tensor([[2.0, 4.0], [3.0, 5.0]]),
        get_xyz=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        get_rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
        _opacity=torch.tensor([[-3.0], [-2.0]]),
        get_features=torch.arange(24.0).reshape(2, 4, 3),
    )
    model = VolumetricFoliageModel(1, device="cpu")
    count = model.initialize_from_surfel_residual(
        surfels, torch.tensor([False, True]), normal_scale_ratio=0.5
    )
    assert count == 1
    assert model.scales.shape == (1, 3)
    assert torch.allclose(model.scales[0], torch.tensor([3.0, 5.0, 2.0]))
    assert torch.equal(model.xyz[0], surfels.get_xyz[1])


def test_hybrid_contract_uses_one_rasterizer_and_no_fixed_branch_composite():
    root = Path(__file__).resolve().parents[1]
    renderer = (root / "outdoor/hybrid_gaussian_renderer.py").read_text()
    trainer = (root / "scripts/train_hybrid_foliage_3dgs.py").read_text()
    assert renderer.count("rasterization(") == 1
    assert "single_global_covariance_projection_and_depth_sort" in trainer
    assert '"fixed_order_branch_compositing": False' in trainer
    assert '"true_volumetric_foliage_3dgs": True' in trainer
    assert "historical_full_real_rgb_initialization" in trainer


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
