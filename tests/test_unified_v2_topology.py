from types import SimpleNamespace

import torch

from outdoor.hybrid_gaussian_renderer import (
    VERIFICATION_VERIFIED,
    VolumetricFoliageModel,
)
from outdoor.standard_3dgs import (
    STUDENT_ROLE_CROWN,
    STUDENT_ROLE_RIGID,
    STUDENT_ROLE_SKY,
)
from scripts.distill_standard_3dgs import _adapt_student_topology
from scripts.train_unified_outdoor_teacher import (
    _adapt_volume,
    _balanced_evidence_fraction,
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
            "track_id": torch.arange(count, dtype=torch.int64),
            "support_view_count": torch.full(
                (count,), 4, dtype=torch.int16
            ),
            "support_sequence_count": torch.full(
                (count,), 2, dtype=torch.int16
            ),
            "support_camera_ids": torch.arange(
                count, dtype=torch.int32
            )[:, None],
            "observation_camera_ids": torch.arange(
                count, dtype=torch.int32
            )[:, None],
            "observation_uv": torch.zeros(count, 1, 2),
            "observation_depth": torch.ones(count, 1),
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


def _exact_conditioned_stats(count):
    stats = _stats(count)
    stats.update(
        {
            "conditioned_gradient": torch.ones(count),
            "conditioned_gradient_count": torch.ones(count),
            "conditioned_radius": torch.full((count,), 10.0),
            # The auxiliary row factor was deliberately not sampled. Exact
            # conditioned screen evidence must still drive topology.
            "observation_gradient": torch.zeros(count),
            "observation_count": torch.zeros(count),
        }
    )
    return stats


def test_conditioned_sampling_balances_per_view_exposure():
    fraction = _balanced_evidence_fraction(1487, 192)
    per_evidence_view = fraction / 192 + (1.0 - fraction) / 1487
    per_other_view = (1.0 - fraction) / 1487

    assert abs(per_evidence_view / per_other_view - 12.0) < 1e-6
    assert 0.55 < fraction < 0.62


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
    old_xyz_parameter = model.xyz
    old_xyz_moment = optimizer.state[old_xyz_parameter][
        "exp_avg"
    ].detach().clone()

    def _legacy_two_stage_mutation(*_args, **_kwargs):
        raise AssertionError(
            "topology adaptation must use one atomic replace-and-split"
        )

    # A regression to prune() followed by split_adaptive() is numerically
    # correct on this tiny fixture but holds three full topologies resident at
    # the real 2M budget. Ensure the trainer uses the fused mutation instead.
    model.prune = _legacy_two_stage_mutation
    model.split_adaptive = _legacy_two_stage_mutation
    event = _adapt_volume(args, model, _stats(4))
    assert event["pruned"] == 1
    # The requested capacity is expressed in net-growth slots. After the
    # contradiction row is pruned, two broad binary parents can each consume
    # one slot in the same atomic mutation.
    assert event["split_parents"] == 2
    assert event["net_growth"] == 2
    assert 0 not in event["_new_to_old"].tolist()
    previous_optimizer = optimizer
    optimizer = _migrate_volume_optimizer(
        args,
        model,
        appearance,
        sky,
        optimizer,
        event["_new_to_old"],
    )
    torch.testing.assert_close(
        optimizer.state[model.xyz]["exp_avg"],
        old_xyz_moment[event["_new_to_old"]],
    )
    assert old_xyz_parameter not in previous_optimizer.state
    assert not previous_optimizer.state
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


def test_teacher_canonical_descendants_can_split_recursively():
    model = _model(2)
    args = SimpleNamespace(
        maximum_volume_splits=2,
        maximum_volume_gaussians=16,
        volume_split_radius=3.0,
    )
    first = _adapt_volume(args, model, _stats(2))
    assert first["split_parents"] == 2
    assert bool((model.split_generation == 1).all())
    # Split metadata intentionally halves the candidate evidence count, but a
    # displaced child cannot recurse until real owner rays verify its centre.
    assert bool((model.support_sequence_count == 1).all())
    blocked = _adapt_volume(args, model, _stats(len(model)))
    assert blocked["split_parents"] == 0
    model.verification_state.fill_(VERIFICATION_VERIFIED)
    second = _adapt_volume(args, model, _stats(len(model)))
    assert second["split_parents"] == 2
    assert int((model.split_generation == 2).sum()) == 4


def test_teacher_dynamic_split_requires_screen_bandwidth_deficit():
    model = _model(2)
    model.layer_role.fill_(2)
    stats = _exact_conditioned_stats(2)
    stats["conditioned_radius"] = torch.tensor([2.9, 3.1])
    args = SimpleNamespace(
        maximum_volume_splits=2,
        maximum_volume_gaussians=8,
        volume_split_radius=3.0,
    )

    event = _adapt_volume(args, model, stats)

    assert event["split_parents"] == 1
    assert event["eligible_count_by_role"]["dynamic_leaf"] == 1
    assert event["selection_audit"]["radius_pixels_minimum"] >= 3.0
    assert event["dynamic_topology_evidence"]["contract"] == (
        "persistent_owner_plus_exact_conditioned_screen_gradient"
    )


def test_teacher_dynamic_phase_keeps_canonical_and_dynamic_topology_adaptive():
    model = _model(3)
    model.layer_role.copy_(torch.tensor([0, 0, 2], dtype=torch.int8))
    stats = _exact_conditioned_stats(3)
    args = SimpleNamespace(
        maximum_volume_splits=3,
        maximum_volume_gaussians=12,
        volume_split_radius=3.0,
    )

    event = _adapt_volume(
        args, model, stats, phase="dynamic_appearance"
    )

    assert event["role_split_parents"] == {
        "static_skeleton": 0,
        "canonical_crown": 2,
        "dynamic_leaf": 1,
    }
    assert event["eligible_count_by_role"]["canonical_crown"] == 2


def test_teacher_settles_full_volume_budget_without_losing_lineage():
    model = _model(2)
    args = SimpleNamespace(
        maximum_volume_splits=2,
        maximum_volume_gaussians=4,
        volume_split_radius=3.0,
    )
    first = _adapt_volume(args, model, _stats(2))
    assert first["new_count"] == 4
    assert first["split_parents"] == 2

    stats = _stats(4)
    stats["contribution"] = torch.tensor([4.0, 1.0, 3.0, 0.5])
    second = _adapt_volume(args, model, stats)

    assert second["old_count"] == 4
    assert second["new_count"] == 4
    assert second["reallocated_pruned"] == 0
    assert second["split_parents"] == 0
    assert second["capacity_reallocation"]["shortfall"] == 0
    assert second["saturated_budget_settle"] is True
    assert len(torch.unique(model.track_id)) == 2


def test_full_budget_settle_preserves_physical_role_capacity():
    model = _model(4)
    model.layer_role.copy_(
        torch.tensor([0, 0, 2, 2], dtype=torch.int8)
    )
    args = SimpleNamespace(
        maximum_volume_splits=4,
        maximum_volume_gaussians=8,
        volume_split_radius=3.0,
    )
    first = _adapt_volume(args, model, _exact_conditioned_stats(4))
    assert first["new_count"] == 8
    before = {
        "canonical": int(model.canonical_crown_mask.sum()),
        "dynamic": int(model.dynamic_leaf_mask.sum()),
    }

    stats = _exact_conditioned_stats(8)
    # A saturated model must not delete either role merely to fund another
    # generic residual-driven split. Strong contradiction pruning remains a
    # separate path and can create real free capacity.
    stats["contribution"][model.canonical_crown_mask] = 10.0
    stats["contribution"][model.dynamic_leaf_mask] = 0.01
    second = _adapt_volume(args, model, stats)

    assert second["new_count"] == 8
    assert second["capacity_reallocation"]["selected_by_role"] == {
        "static_skeleton": 0,
        "canonical_crown": 0,
        "dynamic_leaf": 0,
    }
    assert int(model.canonical_crown_mask.sum()) == before["canonical"]
    assert int(model.dynamic_leaf_mask.sum()) == before["dynamic"]
    assert second["saturated_budget_settle"] is True


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
