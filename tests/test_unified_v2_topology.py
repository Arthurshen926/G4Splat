from types import SimpleNamespace

import torch

from outdoor.hybrid_gaussian_renderer import VolumetricFoliageModel
from outdoor.standard_3dgs import (
    STUDENT_ROLE_CROWN,
    STUDENT_ROLE_RIGID,
    STUDENT_ROLE_SKY,
)
from scripts.distill_standard_3dgs import _adapt_student_topology
from scripts.train_unified_outdoor_teacher import (
    _adapt_volume,
    _migrate_volume_optimizer,
    _volume_optimizer,
)


def _model(count=4):
    model = VolumetricFoliageModel(0, dynamic_rank=1, device="cpu")
    model.initialize_from_volume_state(
        {
            "version": "independent_sfm_semantic_canopy_volume_v1",
            "centers": torch.arange(count * 3).reshape(count, 3).float(),
            "scales": torch.full((count, 3), 0.1),
            "quaternions": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]]
            ).repeat(count, 1),
            "opacities": torch.full((count, 1), 0.02),
            "colors": torch.full((count, 3), 0.5),
            "primitive_role": torch.zeros(count, dtype=torch.int8),
            "layer_role": torch.zeros(count, dtype=torch.int8),
            "support_view_count": torch.full(
                (count,), 4, dtype=torch.int16
            ),
            "support_sequence_count": torch.full(
                (count,), 2, dtype=torch.int16
            ),
            "occupancy_probability": torch.full((count,), 0.8),
            "free_space_violation_count": torch.zeros(
                count, dtype=torch.int16
            ),
        }
    )
    return model


def _stats(count):
    return {
        "gradient": torch.ones(count),
        "gradient_count": torch.ones(count),
        "radius": torch.full((count,), 10.0),
        "contribution": torch.ones(count),
        "residual": torch.ones(count),
        "rigid": torch.zeros(count),
    }


def test_teacher_prunes_then_splits_and_migrates_adam_state():
    model = _model()
    model.occupancy_probability[0] = 0.01
    appearance = torch.nn.Linear(1, 1)
    sky = torch.nn.Linear(1, 1)
    args = SimpleNamespace(
        volume_position_lr=1e-3,
        volume_feature_lr=1e-3,
        volume_opacity_lr=1e-3,
        volume_scale_lr=1e-3,
        volume_rotation_lr=1e-3,
        dynamic_lr=1e-3,
        appearance_lr=1e-3,
        sky_lr=1e-3,
        maximum_volume_splits=2,
        maximum_volume_gaussians=10,
        volume_split_radius=8.0,
    )
    optimizer = _volume_optimizer(args, model, appearance, sky)
    initial_loss = sum(
        parameter.square().sum()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    initial_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    event = _adapt_volume(args, model, _stats(4))
    assert event["pruned"] == 1
    assert event["split_parents"] == 2
    assert 0 not in event["_new_to_old"].tolist()
    optimizer = _migrate_volume_optimizer(
        args,
        model,
        appearance,
        sky,
        optimizer,
        event["_new_to_old"],
    )
    assert optimizer.param_groups[0]["params"][0] is model.xyz
    before = model.xyz.detach().clone()
    model.xyz.square().sum().backward()
    optimizer.step()
    assert not torch.equal(before, model.xyz)


def test_teacher_can_split_static_skeleton_and_canonical_crown():
    model = _model(4)
    model.layer_role.copy_(
        torch.tensor([1, 0, 0, 2], dtype=torch.int8)
    )
    appearance = torch.nn.Linear(1, 1)
    sky = torch.nn.Linear(1, 1)
    args = SimpleNamespace(
        volume_position_lr=1e-3,
        volume_feature_lr=1e-3,
        volume_opacity_lr=1e-3,
        volume_scale_lr=1e-3,
        volume_rotation_lr=1e-3,
        dynamic_lr=1e-3,
        appearance_lr=1e-3,
        sky_lr=1e-3,
        maximum_volume_splits=4,
        maximum_volume_gaussians=16,
        volume_split_radius=3.0,
    )
    event = _adapt_volume(args, model, _stats(4))
    assert event["role_split_parents"]["static_skeleton"] == 1
    assert event["role_split_parents"]["canonical_crown"] > 0
    assert int(model.static_skeleton_mask.sum()) == 2


def test_teacher_volume_budget_caps_net_growth():
    model = _model(4)
    args = SimpleNamespace(
        maximum_volume_splits=10,
        maximum_volume_gaussians=100,
        volume_split_radius=3.0,
    )
    event = _adapt_volume(
        args, model, _stats(4), volume_budget=5
    )
    assert event["split_parents"] == 1
    assert event["new_count"] == 5
    assert event["budget"] == 5
    assert event["remaining"] == 0


def test_student_density_control_protects_sky_and_splits_crown():
    model = _model(3)
    model.primitive_role.copy_(
        torch.tensor(
            [STUDENT_ROLE_SKY, STUDENT_ROLE_RIGID, STUDENT_ROLE_CROWN],
            dtype=torch.int8,
        )
    )
    model.opacity_logits.data[0] = -20
    model.opacity_logits.data[1] = -20
    stats = {
        "gradient": torch.ones(3),
        "gradient_count": torch.ones(3),
        "radius": torch.full((3,), 10.0),
        "contribution": torch.ones(3),
        "residual": torch.ones(3),
    }
    args = SimpleNamespace(
        prune_opacity=0.001,
        split_radius=7.0,
        split_gradient_threshold=1e-5,
        maximum_splits_per_event=2,
        maximum_gaussians=10,
    )
    event = _adapt_student_topology(args, model, stats)
    assert event["pruned"] == 1
    assert bool((model.primitive_role == STUDENT_ROLE_SKY).any())
    assert event["split_parents"] == 1
