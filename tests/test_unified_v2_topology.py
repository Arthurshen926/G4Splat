from types import SimpleNamespace

import pytest
import torch

from outdoor.hybrid_gaussian_renderer import (
    PROPOSAL_RAY_BIRTH,
    VERIFICATION_UNVERIFIED,
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
    _has_two_distinct_supported_camera_ids,
    _has_two_distinct_camera_ids_across_tables,
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


def test_wide_camera_support_scan_is_duplicate_safe_and_dtype_preserving():
    # The production initialization has 105 support slots. Exercise that
    # width directly: duplicate canonical IDs are one witness, invalid and
    # noncanonical IDs do not count, and no int64 table/sort is required.
    support = torch.full((5, 105), -1, dtype=torch.int32)
    support[0, :3] = torch.tensor([2, 2, 2])
    support[1, :4] = torch.tensor([2, 2, 7, 7])
    support[2, :4] = torch.tensor([2, 5, 7, 9])
    support[3, :3] = torch.tensor([7, 9, 11])
    support[4, :4] = torch.tensor([-5, 2, 99, 7])
    canonical = torch.zeros(12, dtype=torch.bool)
    canonical[2] = True
    canonical[7] = True

    result = _has_two_distinct_supported_camera_ids(
        support, canonical
    )

    assert result.tolist() == [False, True, True, False, True]
    assert support.dtype == torch.int32


def test_support_and_verified_camera_tables_form_one_exact_evidence_set():
    support = torch.tensor([[2, -1], [2, -1], [2, 7]], dtype=torch.int32)
    verified = torch.tensor([[7, -1], [2, 2], [7, 7]], dtype=torch.int32)
    canonical = torch.zeros(8, dtype=torch.bool)
    canonical[[2, 7]] = True

    result = _has_two_distinct_camera_ids_across_tables(
        (support, verified), canonical
    )

    assert result.tolist() == [True, False, True]


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


def test_scene_canonical_split_filters_every_role_before_mutation():
    model = _model(3)
    model.layer_role.copy_(
        torch.tensor([1, 0, 2], dtype=torch.int8)
    )
    # Only row 1 has two distinct cameras in the canonical sequence.  Rows 0
    # and 2 each have a second candidate camera, but it belongs to seq1 and is
    # therefore unreachable by the scene-canonical verification lifecycle.
    model.support_camera_ids = torch.tensor(
        [[0, 1], [0, 2], [0, 1]], dtype=torch.int32
    )
    model.proposal_parent_support_camera_ids = torch.empty(
        0, 2, dtype=torch.int32
    )
    model.observation_camera_ids.fill_(-1)
    views = {
        0: SimpleNamespace(image_name="seq0/frame000.png"),
        1: SimpleNamespace(image_name="seq1/frame000.png"),
        2: SimpleNamespace(image_name="seq0/frame001.png"),
    }
    args = SimpleNamespace(
        maximum_volume_splits=3,
        maximum_volume_gaussians=16,
        volume_split_radius=3.0,
        reconstruction_target="static",
        static_canonical_sequence_policy="scene",
        canonical_child_support_color_refresh=False,
    )

    event = _adapt_volume(
        args,
        model,
        _exact_conditioned_stats(3),
        view_by_camera_id=views,
        canonical_rgb_sequence="seq0",
        current_iteration=100,
    )

    assert event["role_split_parents"] == {
        "static_skeleton": 0,
        "canonical_crown": 1,
        "dynamic_leaf": 0,
    }
    child = model.proposal_kind != 0
    assert int(child.sum()) == 4
    assert bool((model.support_view_count[child] == 2).all())
    assert set(model.support_camera_ids[child].flatten().tolist()) == {0, 2}


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


def test_teacher_smoothly_backpressures_unverified_topology_debt():
    model = _model(10)
    model.verification_state[0] = VERIFICATION_UNVERIFIED
    args = SimpleNamespace(
        maximum_volume_splits=10,
        maximum_volume_gaussians=40,
        volume_split_radius=3.0,
        volume_verification_debt_soft_fraction=0.05,
        volume_verification_debt_hard_fraction=0.15,
    )

    event = _adapt_volume(args, model, _stats(10))

    assert event["verification_debt"]["fraction"] == pytest.approx(0.1)
    assert event["verification_debt"][
        "ordinary_split_capacity_scale"
    ] == pytest.approx(0.5)
    assert event["effective_split_limit"] == 5
    assert event["split_parents"] <= 5


def test_teacher_retires_only_expired_zero_witness_low_utility_child():
    model = _model(3)
    model.verification_state[0] = VERIFICATION_UNVERIFIED
    # A real split child starts with no independent camera witness.  The
    # generic fixture represents initialized/verified evidence rows and thus
    # carries one witness until the lifecycle fields are reset explicitly.
    model.verified_camera_count[0] = 0
    model.birth_iteration[0] = 1
    model.opacity_logits.data[0] = torch.logit(torch.tensor(0.001))
    stats = _stats(3)
    stats["contribution"] = torch.tensor([0.0, 1.0, 1.0])
    event = _adapt_volume(
        SimpleNamespace(
            maximum_volume_splits=0,
            maximum_volume_gaussians=8,
            volume_split_radius=3.0,
            child_verification_grace_iterations=500,
            child_verification_timeout_iterations=3000,
        ),
        model,
        stats,
        current_iteration=3001,
    )

    assert event["pruned"] == 1
    lifecycle = event["child_verification_lifecycle"]
    assert lifecycle["expired_candidates"] == 1
    assert lifecycle["expired_zero_witness_low_utility_pruned"] == 1


def test_post_growth_maintenance_rolls_back_expired_partial_witness_family():
    model = _model(1)
    model.evidence_primitive_id[0] = 17
    parent_mass = model.integrated_optical_mass().sum().clone()
    model.split_adaptive(
        torch.tensor([0]), torch.tensor([2]), birth_iteration=100
    )
    # One visit is useful evidence, but it is not the required independent
    # two-camera verification and must not make a failed split permanent.
    model.verified_camera_count.fill_(1)
    stats = _stats(len(model))
    event = _adapt_volume(
        SimpleNamespace(
            maximum_volume_splits=20,
            maximum_volume_gaussians=8,
            volume_split_radius=3.0,
            child_verification_grace_iterations=500,
            child_verification_timeout_iterations=3000,
            maximum_volume_family_rollbacks_per_event=8,
            skeleton_opacity_ceiling=0.40,
            dynamic_leaf_opacity_ceiling=0.40,
            canonical_crown_opacity_ceiling=0.20,
        ),
        model,
        stats,
        current_iteration=3100,
        split_capacity_scale=0.0,
        prune_only_maintenance=True,
    )

    assert event["prune_only_maintenance"]
    assert event["split_parents"] == 0
    assert event["children"] == 0
    assert event["new_count"] == 1
    assert event["child_verification_lifecycle"][
        "failed_family_rollback"
    ]["rolled_back_families"] == 1
    torch.testing.assert_close(
        model.integrated_optical_mass().sum(),
        parent_mass,
        rtol=1e-5,
        atol=1e-7,
    )


def test_expired_skeleton_split_rollback_removes_transaction_siblings():
    model = _model(1)
    model.layer_role.fill_(1)
    parent_mass = model.integrated_optical_mass().sum().clone()
    model.split_adaptive(
        torch.tensor([0]),
        torch.tensor([4]),
        birth_iteration=100,
        allow_static_skeleton=True,
    )
    assert int(model.static_skeleton_mask.sum()) == 4

    event = _adapt_volume(
        SimpleNamespace(
            maximum_volume_splits=0,
            maximum_volume_gaussians=8,
            volume_split_radius=3.0,
            child_verification_grace_iterations=500,
            child_verification_timeout_iterations=3000,
            maximum_volume_family_rollbacks_per_event=8,
            skeleton_opacity_ceiling=0.40,
            dynamic_leaf_opacity_ceiling=0.40,
            canonical_crown_opacity_ceiling=0.20,
        ),
        model,
        _stats(len(model)),
        current_iteration=3100,
        split_capacity_scale=0.0,
        prune_only_maintenance=True,
    )

    assert event["new_count"] == 1
    assert int(model.static_skeleton_mask.sum()) == 1
    assert int((model.proposal_kind != 0).sum()) == 0
    assert len(model.split_parent_snapshot_family_id) == 0
    assert event["child_verification_lifecycle"][
        "failed_family_rollback"
    ]["removed_children"] == 3
    torch.testing.assert_close(
        model.integrated_optical_mass().sum(),
        parent_mass,
        rtol=1e-5,
        atol=1e-7,
    )


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


def test_expired_ray_birth_reaches_terminal_rejection_even_if_it_paints():
    model = _model(1)
    model.static_detail.fill_(True)
    model.proposal_kind.fill_(PROPOSAL_RAY_BIRTH)
    model.verification_state.fill_(VERIFICATION_UNVERIFIED)
    model.verified_camera_count.zero_()
    model.birth_iteration.fill_(0)
    args = SimpleNamespace(
        maximum_volume_splits=0,
        maximum_volume_gaussians=1,
        volume_split_radius=3.0,
        child_verification_timeout_iterations=10,
    )
    stats = _stats(1)
    # High contribution/opactity used to keep a one-witness proposal alive
    # forever. Lifecycle timeout is now terminal for a ray-birth proposal.
    stats["contribution"].fill_(100.0)
    model.opacity_logits.data.fill_(0.0)

    event = _adapt_volume(
        args,
        model,
        stats,
        current_iteration=10,
    )

    assert (
        event["child_verification_lifecycle"][
            "expired_ray_birth_terminally_rejected"
        ]
        == 1
    )
    assert event["new_count"] == 0


def test_lineage_allocator_never_reuses_pruned_highest_id_across_restore():
    model = _model(2)
    model.append_dynamic_leaves(torch.tensor([1]))
    first_new_id = int(model.lineage_id.max())
    remove = torch.zeros(len(model), dtype=torch.bool)
    remove[-1] = True
    model.prune(remove)
    assert first_new_id not in model.lineage_id.tolist()

    state = model.capture()
    restored = VolumetricFoliageModel(0, dynamic_rank=1, device="cpu")
    restored.restore(state)
    restored.append_dynamic_leaves(torch.tensor([1]))

    assert int(restored.lineage_id.max()) > first_new_id
    assert len(torch.unique(restored.lineage_id)) == len(restored)


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
