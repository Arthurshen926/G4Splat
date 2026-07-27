from types import SimpleNamespace

import torch

from scripts.train_unified_outdoor_teacher import (
    TRAINING_PROFILES,
    _dynamic_enabled,
    _dynamic_observation_factor,
    _geometry_losses,
    _masked_ssim_loss,
    _phase,
    _surface_capture_to_device,
)
from outdoor.hybrid_gaussian_renderer import (
    LAYER_DYNAMIC_LEAF,
    VolumetricFoliageModel,
)


def test_fast_profile_overlaps_dynamic_foliage_with_topology():
    assert TRAINING_PROFILES["fast"]["iterations"] == 30_000
    assert _phase(2_399, 30_000, "fast") == "canonical_bootstrap"
    assert _phase(2_400, 30_000, "fast") == "topology"
    assert not _dynamic_enabled(2_399, 30_000, "fast")
    assert _dynamic_enabled(2_400, 30_000, "fast")
    assert _phase(11_999, 30_000, "fast") == "topology"
    assert _phase(12_000, 30_000, "fast") == "dynamic_appearance"
    assert _phase(21_899, 30_000, "fast") == "dynamic_appearance"
    assert _phase(21_900, 30_000, "fast") == "ownership_cleanup"
    assert _phase(26_999, 30_000, "fast") == "ownership_cleanup"
    assert _phase(27_000, 30_000, "fast") == "canonical_polish"


def test_quality_profile_keeps_final_benchmark_schedule():
    assert TRAINING_PROFILES["quality"]["iterations"] == 80_000
    assert _phase(4_799, 80_000) == "canonical_bootstrap"
    assert _phase(4_800, 80_000) == "topology"
    assert _dynamic_enabled(4_800, 80_000, "quality")
    assert _phase(46_399, 80_000) == "topology"
    assert _phase(46_400, 80_000) == "dynamic_appearance"


def test_masked_ssim_does_not_create_zero_boundary_error():
    image = torch.rand(3, 32, 32)
    weight = torch.zeros(32, 32)
    weight[8:24, 8:24] = 1
    assert _masked_ssim_loss(image, image, weight).item() < 1e-6


def test_dynamic_observation_factor_has_geometry_gradient_at_tiny_opacity():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.2, 0.0, 2.0]]),
        "scales": torch.tensor([[0.1, 0.1, 0.1]]),
        "colors": torch.tensor([[0.2, 0.4, 0.6]]),
        "opacities": torch.tensor([[1e-6]]),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "layer_role": torch.tensor([LAYER_DYNAMIC_LEAF], dtype=torch.int8),
        "support_camera_ids": torch.tensor([[7]], dtype=torch.int32),
        "observation_camera_ids": torch.tensor([[7]], dtype=torch.int32),
        "observation_uv": torch.tensor([[[0.5, 0.5]]]),
        "observation_depth": torch.tensor([[2.0]]),
    }
    model = VolumetricFoliageModel(1, device="cpu")
    model.initialize_from_volume_state(payload)
    camera = SimpleNamespace(
        colmap_id=7,
        world_view_transform=torch.eye(4),
        focal_x=100.0,
        focal_y=100.0,
        cx=50.0,
        cy=50.0,
        image_width=100,
        image_height=100,
    )
    loss, observed, audit = _dynamic_observation_factor(
        model, camera, torch.zeros(model.dynamic_rank)
    )
    assert audit["matched"] == 1
    assert observed.tolist() == [True]
    assert loss > 0
    loss.backward()
    assert model.xyz.grad is not None
    assert model.xyz.grad.norm() > 0
    assert model.opacity_logits.grad is None


def test_surface_resume_capture_moves_parameters_buffers_and_metadata():
    capture = (
        0,
        torch.nn.Parameter(torch.ones(2, 3)),
        torch.nn.Parameter(torch.ones(2, 1, 3)),
        torch.nn.Parameter(torch.ones(2, 15, 3)),
        torch.nn.Parameter(torch.ones(2, 2)),
        torch.nn.Parameter(torch.ones(2, 4)),
        torch.nn.Parameter(torch.ones(2, 1)),
        torch.ones(2),
        torch.ones(2, 1),
        torch.ones(2, 1),
        {"state": {}, "param_groups": []},
        1.0,
        {
            "source_type": torch.ones(2, dtype=torch.int16),
            "label": "preserved",
        },
    )
    moved = _surface_capture_to_device(
        capture, torch.device("cpu")
    )

    assert all(
        moved[index].device.type == "cpu" for index in range(1, 10)
    )
    assert moved[12]["source_type"].device.type == "cpu"
    assert moved[12]["label"] == "preserved"
    assert all(
        isinstance(moved[index], torch.nn.Parameter)
        and moved[index].is_leaf
        for index in range(1, 7)
    )


def test_geometry_losses_do_not_propagate_invalid_evidence():
    depth = torch.tensor(
        [[[float("nan"), 0.0], [2.0, 4.0]]],
        requires_grad=True,
    )
    package = SimpleNamespace(
        depth=depth,
        normal_world=torch.tensor(
            [
                [[float("nan"), 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        ),
    )
    evidence = {
        "chart_depth": torch.tensor(
            [[[float("nan"), 2.0], [2.5, 4.5]]]
        ),
        "chart_weight": torch.ones(1, 2, 2),
        "plane_depth": torch.tensor(
            [[[1.0, float("inf")], [2.5, 4.5]]]
        ),
        "plane_weight": torch.ones(1, 2, 2),
        "plane_normal_world": torch.tensor(
            [
                [[float("nan"), 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        ),
        "rho_mean": torch.tensor(
            [[[float("nan"), 0.5], [0.5, 0.25]]]
        ),
        "rho_variance": torch.tensor(
            [[[float("nan"), 0.1], [0.1, 0.1]]]
        ),
        "support_view_count": torch.full((1, 2, 2), 3.0),
        "mono_depth": torch.tensor(
            [[[float("nan"), 1.0], [2.0, 3.0]]]
        ),
    }
    args = SimpleNamespace(
        geometry_weight=0.08,
        plane_weight=0.06,
        normal_weight=0.025,
        ordinal_weight=0.015,
    )
    loss, values = _geometry_losses(
        package, evidence, torch.ones(2, 2), args
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in values.values())
    loss.backward()
    assert torch.isfinite(depth.grad).all()


def test_chart_owner_uses_single_support_metric_inverse_depth():
    depth = torch.full((1, 2, 2), 4.0, requires_grad=True)
    package = SimpleNamespace(
        depth=depth,
        normal_world=torch.zeros(3, 2, 2),
    )
    evidence = {
        "chart_depth": torch.full((1, 2, 2), 2.0),
        "chart_weight": torch.ones(1, 2, 2),
        "rho_mean": torch.full((1, 2, 2), 0.5),
        "rho_variance": torch.full((1, 2, 2), 0.01),
        "source_bitmask": torch.full(
            (1, 2, 2), 2, dtype=torch.uint8
        ),
        # A calibrated Chart is a valid metric observation even when it has
        # no second independent camera in the cache.
        "support_view_count": torch.ones(1, 2, 2),
    }
    args = SimpleNamespace(
        geometry_weight=0.12,
        plane_weight=0.10,
        normal_weight=0.04,
        ordinal_weight=0.015,
    )
    loss, values = _geometry_losses(
        package, evidence, torch.ones(2, 2), args
    )
    assert values["chart"] == 0.0
    assert values["inverse"] > 0.0
    assert values["chart_pixels"] == 4
    assert values["inverse_pixels"] == 4
    assert values["raw_chart_fallback_pixels"] == 0
    loss.backward()
    assert depth.grad.abs().sum() > 0


def test_fused_cache_drops_invalid_boundary_instead_of_raw_fallback():
    depth = torch.full((1, 2, 2), 4.0, requires_grad=True)
    package = SimpleNamespace(
        depth=depth,
        normal_world=torch.zeros(3, 2, 2),
    )
    evidence = {
        # If this atlas edge leaked into the loss it would dominate.
        "chart_depth": torch.full((1, 2, 2), 1e-5),
        "chart_weight": torch.ones(1, 2, 2),
        "rho_mean": torch.full((1, 2, 2), 0.5),
        "rho_variance": torch.full((1, 2, 2), 0.01),
        "source_bitmask": torch.full(
            (1, 2, 2), 2, dtype=torch.uint8
        ),
        "support_view_count": torch.ones(1, 2, 2),
    }
    evidence["rho_mean"][0, 0, 0] = 0
    args = SimpleNamespace(
        geometry_weight=0.12,
        plane_weight=0.10,
        normal_weight=0.04,
        ordinal_weight=0.015,
    )
    loss, values = _geometry_losses(
        package, evidence, torch.ones(2, 2), args
    )
    assert torch.isfinite(loss)
    assert values["inverse_pixels"] == 3
    assert values["chart"] == 0.0
    assert values["raw_chart_fallback_pixels"] == 0
