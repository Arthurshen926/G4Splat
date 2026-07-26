from types import SimpleNamespace

import torch

from scripts.train_unified_outdoor_teacher import (
    TRAINING_PROFILES,
    _geometry_losses,
    _phase,
    _surface_capture_to_device,
)


def test_fast_profile_reaches_dynamic_foliage_at_12k():
    assert TRAINING_PROFILES["fast"]["iterations"] == 30_000
    assert _phase(5_999, 30_000, "fast") == "canonical_bootstrap"
    assert _phase(6_000, 30_000, "fast") == "topology"
    assert _phase(11_999, 30_000, "fast") == "topology"
    assert _phase(12_000, 30_000, "fast") == "dynamic_appearance"
    assert _phase(21_899, 30_000, "fast") == "dynamic_appearance"
    assert _phase(21_900, 30_000, "fast") == "ownership_cleanup"
    assert _phase(26_999, 30_000, "fast") == "ownership_cleanup"
    assert _phase(27_000, 30_000, "fast") == "canonical_polish"


def test_quality_profile_keeps_final_benchmark_schedule():
    assert TRAINING_PROFILES["quality"]["iterations"] == 80_000
    assert _phase(15_999, 80_000) == "canonical_bootstrap"
    assert _phase(16_000, 80_000) == "topology"
    assert _phase(47_999, 80_000) == "topology"
    assert _phase(48_000, 80_000) == "dynamic_appearance"


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
