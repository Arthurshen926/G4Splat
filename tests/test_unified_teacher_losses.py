from types import SimpleNamespace
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest
import torch

from scripts.train_unified_outdoor_teacher import (
    DYNAMIC_LIFECYCLE_REPAIR_TARGET,
    SURFACE_OWNERSHIP_REPAIR_TARGET,
    STATIC_DETAIL_GLOBAL_CLEANUP_CONTRACT,
    STATIC_DETAIL_ISOLATED_CONTRACT,
    PERSISTENT_ENVELOPE_GLOBAL_CLEANUP_CONTRACT,
    STATIC_RAY_BIRTH_VISUAL_HULL_CONTRACT,
    TRAINING_PROFILES,
    VOLUME_OPACITY_SETTLE_CONTRACT,
    _adapt_volume,
    _activation_iteration,
    _accumulate_volume_stats,
    _adaptive_geometry_scale,
    _assert_pointmap_posterior_contract,
    _backward_conditioned_foliage,
    _dynamic_enabled,
    _dynamic_opacity_ceiling,
    _foliage_enabled,
    _full_epoch_schedule,
    _dynamic_opacity_floor,
    _dynamic_observation_factor,
    _dav2_observation_patch_loss,
    _enforce_retirement_only_fallback_opacity_gradient,
    _evidence_biased_schedule,
    _balanced_instance_topk,
    _boundary_evidence_weight,
    _branch_isolated_rgb_losses,
    _bounded_surface_retirement_fractions,
    _canopy_topology_signal,
    _confirmed_contradiction_fraction,
    _complete_evidence_epoch_batch_size,
    _continued_surface_iteration,
    _conditioned_branch_active,
    _conditioned_base_gradient_gate,
    _conditioned_photo_losses,
    _counterfactual_volume_transparency_loss,
    _evidence_adaptive_role_quotas,
    _geometry_losses,
    _inherit_surface_optimizer_contract,
    _load_dav2_observation_patches,
    _masked_clamp_max_,
    _masked_ssim_loss,
    _parameter_loss_gradient_audit,
    _periodic_schedule_index,
    _apply_volume_role_gradients,
    _apply_volume_opacity_settle_policy,
    _apply_static_optical_policy,
    _apply_static_ray_local_mass_handoff,
    _accumulate_static_replacement_evidence,
    _update_static_child_verification_from_render,
    _apply_mature_surface_gradient_policy,
    _phase,
    _public_phase_name,
    _prefix_stable_resume_ray_batch,
    _resolved_phase_schedule,
    _resume_conditioned_visit_counts,
    _refresh_split_child_owner_colors,
    _retain_checkpoint_snapshot,
    _rigid_completion_seed_indices,
    _rigid_residual_patch_loss,
    _resume_training_contract_differences,
    _soft_surface_canopy_conflict,
    _static_detail_exclusive_topology_active,
    _static_detail_ray_trainable,
    _static_detail_isolated_gradient_gates,
    _static_ray_candidate_masks,
    _static_detail_canonical_ownership_gate,
    _static_detail_global_cleanup_loss,
    _persistent_envelope_global_cleanup_loss,
    _surface_capture_to_device,
    _surface_topology_active,
    _trainer_repair_hash_change_is_allowed,
    _chart_topology_active,
    _volume_topology_ramp_scale,
    _validate_fixed_cameras,
    _validate_initialization_rgb_source,
    _validate_initialization_protocol,
    _validate_surface_warmstart,
    _volume_stats,
    _volume_split_authority,
    _volume_topology_active,
    _volume_topology_authority,
    _write_rigid_stage_surface_handoff,
)
from outdoor.hybrid_gaussian_renderer import (
    LAYER_DYNAMIC_LEAF,
    VERIFICATION_UNVERIFIED,
    VERIFICATION_VERIFIED,
    VolumetricFoliageModel,
)
from outdoor.training_evidence import (
    FoliageRayEvidence,
    OutdoorGeometryEvidence,
)

RIGID_SURFACE_OPTIMIZER = {
    "position_lr_init": 1.6e-5,
    "position_lr_final": 1.6e-6,
    "position_lr_delay_mult": 0.01,
    "position_lr_max_steps": 20_000,
    "feature_lr": 0.0025,
    "opacity_lr": 0.05,
    "scaling_lr": 0.005,
    "rotation_lr": 0.001,
}


def _static_optical_policy_fixture(authority: float):
    opacity = torch.nn.Parameter(torch.zeros(2, 1))
    opacity.grad = torch.tensor([[-2.0], [-3.0]])
    foliage = SimpleNamespace(
        xyz=torch.nn.Parameter(torch.zeros(2, 3)),
        log_scales=torch.nn.Parameter(torch.zeros(2, 3)),
        quaternions=torch.nn.Parameter(torch.zeros(2, 4)),
        opacity_logits=opacity,
        features=torch.nn.Parameter(torch.zeros(2, 1, 3)),
        deformation_basis=torch.nn.Parameter(torch.zeros(2, 1)),
        dynamic_feature_basis=torch.nn.Parameter(torch.zeros(2, 1)),
        dynamic_opacity_basis=torch.nn.Parameter(torch.zeros(2, 1)),
        persistent_envelope_mask=torch.tensor([True, True]),
        static_leaf_mask=torch.tensor([False, False]),
        replacement_observation_count=torch.tensor([3, 3]),
        replacement_overlap_ema=torch.full((2,), float(authority)),
    )
    moment = torch.tensor([[-4.0], [-5.0]])
    optimizer = SimpleNamespace(state={opacity: {"exp_avg": moment}})
    return foliage, optimizer, moment


def test_disabled_static_replacement_does_not_report_or_apply_attenuation():
    foliage, optimizer, moment = _static_optical_policy_fixture(0.0)
    gradient_before = foliage.opacity_logits.grad.clone()
    moment_before = moment.clone()
    audit = _apply_static_optical_policy(
        foliage, optimizer, "ownership_cleanup"
    )
    assert audit["envelope_growth_rows_attenuated"] == 0
    assert torch.equal(foliage.opacity_logits.grad, gradient_before)
    assert torch.equal(moment, moment_before)


def test_local_static_replacement_attenuates_only_authorized_growth():
    foliage, optimizer, moment = _static_optical_policy_fixture(0.5)
    audit = _apply_static_optical_policy(
        foliage, optimizer, "ownership_cleanup"
    )
    readiness = 1.0 - torch.exp(torch.tensor(-1.0))
    expected_multiplier = 1.0 - 0.5 * readiness
    assert audit["envelope_growth_rows_attenuated"] == 2
    assert torch.allclose(
        foliage.opacity_logits.grad,
        torch.tensor([[-2.0], [-3.0]]) * expected_multiplier,
    )
    assert torch.allclose(
        moment, torch.tensor([[-4.0], [-5.0]]) * expected_multiplier
    )


def test_unverified_envelope_child_cannot_be_retired_or_growth_attenuated():
    foliage, optimizer, moment = _static_optical_policy_fixture(1.0)
    foliage.verification_state = torch.tensor(
        [VERIFICATION_UNVERIFIED, VERIFICATION_VERIFIED],
        dtype=torch.int8,
    )
    audit = _apply_static_optical_policy(
        foliage, optimizer, "ownership_cleanup"
    )
    readiness = 1.0 - torch.exp(torch.tensor(-1.0))
    assert foliage.opacity_logits.grad[0].item() == -2.0
    torch.testing.assert_close(
        foliage.opacity_logits.grad[1],
        torch.tensor([-3.0 * (1.0 - readiness)]),
    )
    assert audit["opacity_growth_removed_by_policy"][
        "unverified_child"
    ] == 0.0
    assert moment[0].item() == -4.0


def test_real_ray_handoff_is_bounded_and_reversible():
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(
        {
            "version": "independent_sfm_semantic_canopy_volume_v1",
            "centers": torch.tensor([[0.0, 0.0, 2.0]]),
            "scales": torch.full((1, 3), 0.1),
            "colors": torch.full((1, 3), 0.4),
            "opacities": torch.full((1, 1), 0.2),
            "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "layer_role": torch.tensor([0], dtype=torch.int8),
        }
    )
    initial = foliage.integrated_optical_mass().clone()
    foliage.replacement_overlap_ema.fill_(1.0)
    foliage.replacement_observation_count.fill_(3)
    optimizer = SimpleNamespace(state={})
    retired = _apply_static_ray_local_mass_handoff(
        foliage, optimizer, maximum_fraction_per_event=0.02
    )
    assert retired["changed_rows"] == 1
    assert torch.allclose(
        foliage.integrated_optical_mass(), initial * 0.98, rtol=1e-5
    )
    foliage.replacement_overlap_ema.zero_()
    restored = _apply_static_ray_local_mass_handoff(
        foliage, optimizer, maximum_fraction_per_event=0.02
    )
    assert restored["restored_mass"] > 0
    assert torch.allclose(
        foliage.integrated_optical_mass(), initial, rtol=1e-5
    )


def test_unverified_envelope_is_ineligible_for_mass_handoff():
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(
        {
            "version": "independent_sfm_semantic_canopy_volume_v1",
            "centers": torch.tensor([[0.0, 0.0, 2.0]]),
            "scales": torch.full((1, 3), 0.1),
            "colors": torch.full((1, 3), 0.4),
            "opacities": torch.full((1, 1), 0.2),
            "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "layer_role": torch.tensor([0], dtype=torch.int8),
        }
    )
    initial = foliage.integrated_optical_mass().clone()
    foliage.verification_state.fill_(VERIFICATION_UNVERIFIED)
    foliage.replacement_overlap_ema.fill_(1.0)
    foliage.replacement_observation_count.fill_(3)
    audit = _apply_static_ray_local_mass_handoff(
        foliage,
        SimpleNamespace(state={}),
        maximum_fraction_per_event=0.02,
    )
    assert audit["changed_rows"] == 0
    torch.testing.assert_close(
        foliage.integrated_optical_mass(), initial
    )
    # A legacy/broken checkpoint may already contain retirement state on a
    # row that is subsequently recognized as unverified.  The next handoff
    # event must restore that mass instead of leaving a hidden hole.
    foliage.verification_state.fill_(VERIFICATION_VERIFIED)
    _apply_static_ray_local_mass_handoff(
        foliage,
        SimpleNamespace(state={}),
        maximum_fraction_per_event=0.02,
    )
    assert foliage.integrated_optical_mass().item() < initial.item()
    foliage.verification_state.fill_(VERIFICATION_UNVERIFIED)
    repaired = _apply_static_ray_local_mass_handoff(
        foliage,
        SimpleNamespace(state={}),
        maximum_fraction_per_event=0.02,
    )
    assert repaired["repaired_ineligible_rows"] == 1
    torch.testing.assert_close(
        foliage.integrated_optical_mass(), initial, rtol=1e-5, atol=1e-7
    )


def test_replacement_readiness_counts_distinct_cameras_only():
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(
        {
            "version": "independent_sfm_semantic_canopy_volume_v1",
            "centers": torch.tensor([[0.0, 0.0, 2.0]]),
            "scales": torch.full((1, 3), 0.1),
            "colors": torch.full((1, 3), 0.4),
            "opacities": torch.full((1, 1), 0.2),
            "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "layer_role": torch.tensor([0], dtype=torch.int8),
        }
    )
    value = torch.ones(1)
    visible = torch.ones(1, dtype=torch.bool)
    _accumulate_static_replacement_evidence(
        foliage, value, visible, camera_id=7, decay=0.0
    )
    _accumulate_static_replacement_evidence(
        foliage, value, visible, camera_id=7, decay=0.0
    )
    assert foliage.replacement_observation_count.item() == 1
    _accumulate_static_replacement_evidence(
        foliage, value, visible, camera_id=8, decay=0.0
    )
    assert foliage.replacement_observation_count.item() == 2


def test_static_child_reverification_refreshes_dc_from_distinct_real_views():
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(
        {
            "version": "independent_sfm_semantic_canopy_volume_v1",
            "centers": torch.tensor([[0.0, 0.0, 2.0]]),
            "scales": torch.full((1, 3), 0.1),
            "colors": torch.full((1, 3), 0.9),
            "opacities": torch.full((1, 1), 0.2),
            "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "layer_role": torch.tensor([0], dtype=torch.int8),
            "support_camera_ids": torch.tensor(
                [[7, 8]], dtype=torch.int32
            ),
            "support_sequence_count": torch.tensor(
                [2], dtype=torch.int16
            ),
            "evidence_primitive_id": torch.tensor(
                [11], dtype=torch.int64
            ),
        }
    )
    foliage.split(torch.tensor([0]), birth_iteration=20)
    foliage.support_sequence_count.fill_(2)
    optimizer = SimpleNamespace(state={})
    sequence_lookup = torch.full((9,), -1, dtype=torch.int16)
    sequence_lookup[7] = 0
    sequence_lookup[8] = 1

    def package(rgb):
        responsibility = torch.zeros(len(foliage), 7)
        responsibility[0, 0] = 1.0
        responsibility[0, 1] = 1.0
        responsibility[0, 4:7] = torch.tensor(rgb)
        return SimpleNamespace(
            responsibility=responsibility, structural_count=0
        )

    first = _update_static_child_verification_from_render(
        foliage,
        package([0.2, 0.4, 0.6]),
        camera_id=7,
        camera_sequence_lookup=sequence_lookup,
        volume_optimizer=optimizer,
    )
    assert first["new_color_witnesses"] == 1
    assert foliage.verification_state[0].item() == 0
    torch.testing.assert_close(
        foliage.features[0, 0] * 0.28209479177387814 + 0.5,
        torch.tensor([0.2, 0.4, 0.6]),
    )

    second = _update_static_child_verification_from_render(
        foliage,
        package([0.4, 0.4, 0.4]),
        camera_id=8,
        camera_sequence_lookup=sequence_lookup,
        volume_optimizer=optimizer,
    )
    assert second["newly_verified"] == 1
    assert foliage.verified_camera_count[0].item() == 2
    assert foliage.verified_sequence_count[0].item() == 2
    assert foliage.evidence_primitive_id[0].item() == 11
    assert foliage.candidate_evidence_primitive_id[0].item() == -1
    torch.testing.assert_close(
        foliage.features[0, 0] * 0.28209479177387814 + 0.5,
        torch.tensor([0.3, 0.4, 0.5]),
    )


def test_ray_epoch_batch_capacity_uses_final_horizon_not_prefix():
    batch, minimum, calls = _complete_evidence_epoch_batch_size(
        512,
        1_773_981,
        12_000,
        4,
    )
    assert calls == 3000
    assert minimum == 621
    assert batch == 621
    assert batch * calls >= 1_773_981
    # A 1k diagnostic prefix of the same 12k method must not silently choose
    # a different batch; callers pass the final schedule horizon.
    assert _complete_evidence_epoch_batch_size(
        512, 1_773_981, 12_000, 4
    ) == (batch, minimum, calls)


def _ray_epoch_capacity_audit(*, prior, remaining):
    scheduled = 3000
    available = scheduled - prior
    required = available - 2
    return {
        "batch_boundary_contract": (
            "short_tail_never_wraps_into_next_camera_epoch"
        ),
        "capacity_contract": (
            "sum_per_camera_ceil_rows_over_batch_lte_scheduled_calls"
        ),
        "capacity_scope": (
            "clean_epoch_start"
            if prior == 0
            else "resume_remaining_unvisited_rows"
        ),
        "scheduled_factor_calls": scheduled,
        "prior_factor_calls": prior,
        "available_factor_calls": available,
        "camera_table_count": 750,
        "full_effective_row_count": 3_079_774,
        "remaining_unvisited_row_count": remaining,
        "clean_epoch_aggregate_minimum_batch_with_five_percent_margin": 1078,
        "clean_epoch_minimum_batch_for_complete_per_camera_epoch": 1173,
        "clean_epoch_effective_batch": 1173,
        "runtime_aggregate_minimum_batch_with_five_percent_margin": 1078,
        "minimum_batch_for_runtime_completion": 1173,
        "required_factor_calls": required,
        "unused_factor_calls": 2,
        "complete_epoch_capacity": True,
    }


def test_resume_ray_capacity_compares_fixed_epoch_not_runtime_cursor():
    saved = {
        "ray_posterior_maximum_rays": 1173,
        "ray_evidence_epoch_capacity": _ray_epoch_capacity_audit(
            prior=0, remaining=3_079_774
        ),
    }
    current = {
        "ray_posterior_maximum_rays": 1173,
        "ray_evidence_epoch_capacity": _ray_epoch_capacity_audit(
            prior=125, remaining=2_933_149
        ),
    }
    assert not _resume_training_contract_differences(saved, current)

    changed_evidence = {
        **current,
        "ray_evidence_epoch_capacity": {
            **current["ray_evidence_epoch_capacity"],
            "full_effective_row_count": 3_079_775,
        },
    }
    assert _resume_training_contract_differences(
        saved, changed_evidence
    ) == {"ray_evidence_epoch_capacity"}


def test_resume_allows_only_exact_6k_volume_topology_settle():
    saved = {
        "schedule_horizon": 12_000,
        "volume_densify_until_iteration": 12_000,
    }
    settled = {
        "schedule_horizon": 12_000,
        "volume_densify_until_iteration": 6_000,
    }
    assert not _resume_training_contract_differences(
        saved,
        settled,
        allow_volume_topology_settle_migration=True,
    )
    with pytest.raises(RuntimeError, match="12k->6k"):
        _resume_training_contract_differences(
            saved,
            {**settled, "volume_densify_until_iteration": 7_000},
            allow_volume_topology_settle_migration=True,
        )


def test_static_detail_isolated_resume_migrates_only_explicit_contract():
    saved = {"reconstruction_target": "static"}
    detail = {
        "contract": STATIC_DETAIL_ISOLATED_CONTRACT,
        "every": 2,
        "weight": 1.0,
    }
    current = {
        **saved,
        "static_detail_isolated_supervision": detail,
        "static_training_stages": {
            "detail_training_signals": [
                "surface_plus_detail_isolated_rgb_high_frequency"
            ]
        },
        "parameter_loss_permission_matrix": {
            "static_leaf_optical_mass": [
                "detail_isolated_rgb_high_frequency"
            ]
        },
        "static_detail_topology": {
            "exclusive_after_envelope_stage": True
        },
    }
    assert _resume_training_contract_differences(saved, current) == {
        "static_detail_isolated_supervision",
        "static_training_stages",
        "parameter_loss_permission_matrix",
        "static_detail_topology",
    }
    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_static_detail_isolated_repair_migration=True,
    )
    with pytest.raises(RuntimeError, match="valid v48"):
        _resume_training_contract_differences(
            saved,
            {
                **current,
                "static_detail_isolated_supervision": {
                    **detail,
                    "every": -1,
                },
            },
            allow_static_detail_isolated_repair_migration=True,
        )


def test_volume_stats_ownership_gate_routes_means2d_only_to_detail_rows():
    class Foliage:
        xyz = torch.zeros(3, 3)

        def __len__(self):
            return 3

    stats = _volume_stats(Foliage())
    means = torch.zeros(3, 3, requires_grad=True)
    means.grad = torch.tensor(
        [[3.0, 4.0, 0.0], [5.0, 12.0, 0.0], [8.0, 15.0, 0.0]]
    )
    package = SimpleNamespace(
        structural_count=0,
        radii=torch.tensor([2.0, 3.0, 4.0]),
        volume_means2d=means,
        responsibility=torch.ones(3, 4),
    )
    detail = torch.tensor([False, True, False])
    _accumulate_volume_stats(stats, package, detail)
    assert torch.equal(stats["gradient"], torch.tensor([0.0, 13.0, 0.0]))
    assert torch.equal(stats["gradient_count"], detail.float())
    assert torch.equal(stats["radius"], torch.tensor([0.0, 3.0, 0.0]))
    assert torch.equal(stats["contribution"], detail.float())


def test_static_detail_gradients_are_owned_by_support_sequences():
    class Foliage:
        xyz = torch.zeros(4, 3)
        static_leaf_mask = torch.tensor([False, True, True, True])
        support_camera_ids = torch.tensor(
            [[-1, -1], [0, -1], [1, -1], [0, 1]], dtype=torch.int32
        )

        def __len__(self):
            return 4

    lookup = torch.tensor([0, 1], dtype=torch.int16)
    gate = _static_detail_canonical_ownership_gate(Foliage(), 0, lookup)
    assert torch.equal(gate, torch.tensor([1.0, 1.0, 0.0, 1.0]))
    missing = _static_detail_canonical_ownership_gate(
        Foliage(), 4, lookup
    )
    assert torch.equal(missing, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_static_canonical_ownership_resume_migrates_only_new_contract():
    saved = {"reconstruction_target": "static"}
    ownership = {
        "enabled": True,
        "forward_visibility": "unconditional_static",
        "gradient_owner": "persisted_support_camera_sequences",
    }
    current = {
        **saved,
        "static_detail_canonical_ownership": ownership,
        "static_detail_isolated_supervision": {
            "view_schedule": {
                "contract": (
                    "uniform_complete_epochs_over_visible_canonical_"
                    "support_cameras"
                ),
                "camera_count": 4,
            }
        },
    }
    assert _resume_training_contract_differences(saved, current) == {
        "static_detail_canonical_ownership",
        "static_detail_isolated_supervision",
    }
    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_static_canonical_ownership_repair_migration=True,
    )
    with pytest.raises(RuntimeError, match="enabled v51"):
        _resume_training_contract_differences(
            saved,
            {
                **current,
                "static_detail_canonical_ownership": {
                    **ownership,
                    "enabled": False,
                },
            },
            allow_static_canonical_ownership_repair_migration=True,
        )


def test_trainer_repair_migrates_static_global_negative_permission_contract():
    saved = {
        "reconstruction_target": "static",
        "static_training_stages": {"detail_training_signals": ["old"]},
        "static_detail_canonical_ownership": {
            "noncanonical_views": "render_only_no_detail_parameter_update"
        },
        "parameter_loss_permission_matrix": {
            "static_leaf_xyz_scale": ["canonical_only"]
        },
    }
    cleanup = {
        "contract": STATIC_DETAIL_GLOBAL_CLEANUP_CONTRACT,
        "every": 2,
        "weight": 0.25,
    }
    current = {
        **saved,
        "static_detail_global_cleanup": cleanup,
        "static_training_stages": {
            "detail_training_signals": [
                "all_view_rigid_free_counterfactual_cleanup"
            ]
        },
        "static_detail_canonical_ownership": {
            "noncanonical_views": (
                "render_plus_global_negative_geometry_optical_cleanup"
            )
        },
        "parameter_loss_permission_matrix": {
            "static_leaf_xyz_scale": [
                "all_view_rigid_free_counterfactual_cleanup"
            ]
        },
    }
    assert _resume_training_contract_differences(saved, current) == {
        "static_detail_global_cleanup",
        "static_training_stages",
        "static_detail_canonical_ownership",
        "parameter_loss_permission_matrix",
    }
    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_trainer_repair_migration=True,
    )


def test_trainer_repair_migrates_persistent_envelope_cleanup_contract():
    saved = {
        "reconstruction_target": "static",
        "static_training_stages": {
            "envelope_training_signals": ["canonical_low_frequency_rgb"]
        },
        "parameter_loss_permission_matrix": {
            "envelope_optical_mass": ["ray_free_hit"]
        },
    }
    cleanup = {
        "contract": PERSISTENT_ENVELOPE_GLOBAL_CLEANUP_CONTRACT,
        "every": 2,
        "weight": 0.125,
    }
    current = {
        **saved,
        "persistent_envelope_global_cleanup": cleanup,
        "static_training_stages": {
            "envelope_training_signals": [
                "envelope_only_all_view_rigid_free_counterfactual_cleanup"
            ]
        },
        "parameter_loss_permission_matrix": {
            "envelope_optical_mass": [
                "all_view_rigid_free_counterfactual_cleanup"
            ]
        },
    }
    assert _resume_training_contract_differences(saved, current) == {
        "persistent_envelope_global_cleanup",
        "static_training_stages",
        "parameter_loss_permission_matrix",
    }
    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_trainer_repair_migration=True,
    )


def test_trainer_repair_migrates_static_ray_visual_hull_contract():
    saved = {"reconstruction_target": "static"}
    current = {
        **saved,
        "static_ray_birth": {
            "contract": STATIC_RAY_BIRTH_VISUAL_HULL_CONTRACT,
            "midpoint_voxel_size": 0.15,
            "visual_hull_voxel_size": 0.30,
        },
    }
    assert _resume_training_contract_differences(saved, current) == {
        "static_ray_birth"
    }
    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_trainer_repair_migration=True,
    )


def test_static_staged_topology_switches_to_detail_after_envelope_phase():
    args = SimpleNamespace(
        reconstruction_target="static",
        static_detail_exclusive_topology=True,
    )
    assert not _static_detail_exclusive_topology_active(args, "topology")
    assert _static_detail_exclusive_topology_active(
        args, "static_foliage"
    )
    args.static_detail_exclusive_topology = False
    assert not _static_detail_exclusive_topology_active(
        args, "static_foliage"
    )
    args.static_detail_exclusive_topology = True
    args.reconstruction_target = "sequence_conditioned_legacy"
    assert not _static_detail_exclusive_topology_active(
        args, "static_foliage"
    )


def test_resume_allows_only_exact_6k_volume_opacity_settle():
    saved = {
        "schedule_horizon": 12_000,
        "volume_densify_until_iteration": 12_000,
    }
    settle = {
        "contract": VOLUME_OPACITY_SETTLE_CONTRACT,
        "policy": "retirement_only",
        "start_iteration": 6_000,
        "trainable_after_settle": (
            "geometry_scale_rotation_sh_and_dynamic_deformation_feature"
        ),
    }
    current = {**saved, "volume_opacity_settle": settle}
    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_volume_opacity_settle_migration=True,
    )
    freeze = {
        **current,
        "volume_opacity_settle": {**settle, "policy": "freeze"},
    }
    assert not _resume_training_contract_differences(
        current,
        freeze,
        allow_volume_opacity_settle_migration=True,
    )
    with pytest.raises(RuntimeError, match="audited 6k settle point"):
        _resume_training_contract_differences(
            saved,
            {
                **current,
                "volume_opacity_settle": {
                    **settle,
                    "start_iteration": 7_000,
                },
            },
            allow_volume_opacity_settle_migration=True,
        )


def test_resume_ray_capacity_rejects_inconsistent_runtime_audit():
    saved = {
        "ray_evidence_epoch_capacity": _ray_epoch_capacity_audit(
            prior=0, remaining=3_079_774
        )
    }
    corrupt = {
        "ray_evidence_epoch_capacity": {
            **_ray_epoch_capacity_audit(
                prior=125, remaining=2_933_149
            ),
            "available_factor_calls": 1,
        }
    }
    with pytest.raises(RuntimeError, match="internally inconsistent"):
        _resume_training_contract_differences(saved, corrupt)


def test_resume_ray_batch_keeps_checkpoint_value_when_still_feasible():
    resume = {
        "training_contract": {"ray_posterior_maximum_rays": 1173}
    }
    rows = (2_345, 1_172, 1)
    assert _prefix_stable_resume_ray_batch(
        1172, resume, rows, available_factor_calls=4
    ) == 1173
    with pytest.raises(RuntimeError, match="cannot complete"):
        _prefix_stable_resume_ray_batch(
            1172, resume, rows, available_factor_calls=3
        )


def test_ray_epoch_batch_capacity_accounts_for_per_camera_tail_calls():
    batch, exact_minimum, calls = _complete_evidence_epoch_batch_size(
        512,
        3_003,
        4,
        1,
        per_camera_rows=[1_001, 1_001, 1_001],
    )
    # Aggregate capacity would choose 789, but that needs six independent
    # camera-table calls. Four calls can complete all three tables only once
    # each table fits in a single call.
    assert calls == 4
    assert exact_minimum == 1_001
    assert batch == 1_001
    assert sum((count + batch - 1) // batch for count in [1_001] * 3) <= calls


def test_ray_epoch_batch_capacity_uses_resume_remaining_call_budget():
    batch, exact_minimum, calls = _complete_evidence_epoch_batch_size(
        512,
        2_400,
        4,
        1,
        margin=1.0,
        per_camera_rows=[1_200, 1_200],
        available_factor_calls=2,
    )
    assert calls == 2
    assert exact_minimum == 1_200
    assert batch == 1_200


def test_trainer_repair_resume_accepts_chart_only_python_change():
    assert _trainer_repair_hash_change_is_allowed(
        {"chart_surface_model"}, enabled=True
    )
    assert _trainer_repair_hash_change_is_allowed(
        {"trainer", "training_evidence"}, enabled=True
    )
    assert not _trainer_repair_hash_change_is_allowed(
        {"hybrid_renderer"}, enabled=True
    )
    assert not _trainer_repair_hash_change_is_allowed(
        {"chart_surface_model"}, enabled=False
    )


def test_dav2_observation_patch_contract_is_topology_independent(tmp_path):
    seed = tmp_path / "surface_seed.npz"
    np.savez_compressed(
        seed,
        source_type=np.asarray([4, 4, 1], dtype=np.int8),
        dav2_observation_view_name=np.asarray(
            ["view_a", "view_a", ""]
        ),
        dav2_observation_uv=np.asarray(
            [[0.25, 0.25], [0.75, 0.75], [np.nan, np.nan]],
            dtype=np.float32,
        ),
    )
    lookup = _load_dav2_observation_patches(
        seed, device=torch.device("cpu")
    )
    assert set(lookup) == {"view_a"}
    assert lookup["view_a"].shape == (2, 2)

    target = torch.zeros(3, 8, 8)
    prediction = torch.ones(3, 8, 8, requires_grad=True)
    loss, sampled = _dav2_observation_patch_loss(
        prediction,
        target,
        lookup["view_a"],
        radius=1,
    )
    assert sampled == 18
    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    assert prediction.grad is not None
    assert float(prediction.grad.abs().sum()) > 0


def test_mature_handoff_appearance_policy_preserves_geometry_gradients():
    surface = SimpleNamespace(
        _xyz=torch.nn.Parameter(torch.ones(2, 3)),
        _scaling=torch.nn.Parameter(torch.ones(2, 2)),
        _rotation=torch.nn.Parameter(torch.ones(2, 4)),
        _opacity=torch.nn.Parameter(torch.ones(2, 1)),
        _features_dc=torch.nn.Parameter(torch.ones(2, 1, 3)),
        _features_rest=torch.nn.Parameter(torch.ones(2, 3, 3)),
    )
    for parameter in vars(surface).values():
        parameter.grad = torch.ones_like(parameter)

    chart_parameter = torch.nn.Parameter(torch.ones(2, 3))
    chart_parameter.grad = torch.ones_like(chart_parameter)
    chart_surface = torch.nn.ParameterList([chart_parameter])
    audit = _apply_mature_surface_gradient_policy(
        surface,
        "appearance_only",
        chart_surface=chart_surface,
    )

    for parameter in (
        surface._xyz,
        surface._scaling,
        surface._rotation,
        surface._opacity,
    ):
        assert not bool(parameter.grad.any())
    assert bool(surface._features_dc.grad.any())
    assert bool(surface._features_rest.grad.any())
    assert chart_parameter.grad is None
    assert audit["geometry_trainable"] is False
    assert audit["chart_geometry_trainable"] is False
    assert audit["appearance_trainable"] is True


def test_mature_handoff_disables_chart_uv_topology():
    args = SimpleNamespace(
        mature_handoff_surface_policy="appearance_only",
        densify_from_iter=1,
        densify_until_iter=100,
        chart_quadtree_growth_fraction=0.3,
    )
    assert not _chart_topology_active(10, args)
    args.mature_handoff_surface_policy = "joint"
    assert _chart_topology_active(10, args)


def test_surface_retirement_is_continuous_and_optical_mass_bounded():
    previous = torch.zeros(3)
    desired = torch.tensor([0.8, 0.4, 0.0])
    optical_depth = torch.ones(3)
    updated, audit = _bounded_surface_retirement_fractions(
        previous,
        desired,
        optical_depth,
        maximum_mass_fraction=0.05,
    )
    torch.testing.assert_close(
        updated, torch.tensor([0.1, 0.05, 0.0])
    )
    assert np.isclose(audit["requested_optical_mass_fraction"], 0.4)
    assert np.isclose(audit["realized_optical_mass_fraction"], 0.05)
    assert updated[0] > updated[1] > updated[2]


def test_volume_topology_ramp_is_continuous_and_horizon_bound():
    args = SimpleNamespace(
        volume_topology_ramp_iterations=100,
        training_profile="hybrid_handoff_quality",
        phase_schedule_horizon=1000,
    )
    activation = _activation_iteration(
        TRAINING_PROFILES["hybrid_handoff_quality"]["dynamic_start"],
        1000,
    )
    before = _volume_topology_ramp_scale(activation - 2, args)
    middle = _volume_topology_ramp_scale(activation + 48, args)
    after = _volume_topology_ramp_scale(activation + 198, args)

    assert before == 0.0
    assert 0.45 < middle < 0.55
    assert after == 1.0


def test_parameter_loss_audit_reports_role_specific_mass_direction():
    foliage = SimpleNamespace(
        xyz=torch.zeros(2, 3, requires_grad=True),
        log_scales=torch.zeros(2, 3, requires_grad=True),
        opacity_logits=torch.zeros(2, 1, requires_grad=True),
        features=torch.zeros(2, 1, 3, requires_grad=True),
        static_skeleton_mask=torch.tensor([False, False]),
        persistent_envelope_mask=torch.tensor([True, False]),
        static_leaf_mask=torch.tensor([False, True]),
    )
    loss = -foliage.opacity_logits[0, 0] + foliage.opacity_logits[1, 0]

    audit = _parameter_loss_gradient_audit(
        foliage, {"ray_free_hit": loss}
    )

    mass = audit["losses"]["ray_free_hit"]["mass"]
    assert mass["envelope"]["increase_entries"] == 1
    assert mass["detail"]["decrease_entries"] == 1


def test_static_volume_topology_ramp_uses_foliage_not_disabled_dynamic_start():
    args = SimpleNamespace(
        volume_topology_ramp_iterations=100,
        training_profile="static_handoff_fast",
        reconstruction_target="static",
        phase_schedule_horizon=1000,
    )
    assert _volume_topology_ramp_scale(0, args) > 0.0
    assert 0.45 < _volume_topology_ramp_scale(48, args) < 0.55
    assert _volume_topology_ramp_scale(198, args) == 1.0


def test_masked_clamp_updates_original_parameter_rows():
    values = torch.nn.Parameter(
        torch.tensor([[2.0], [-1.0], [3.0], [0.0]])
    )
    mask = torch.tensor([True, False, True, False])
    _masked_clamp_max_(values, mask, 0.5)
    torch.testing.assert_close(
        values.detach(),
        torch.tensor([[0.5], [-1.0], [0.5], [0.0]]),
    )


def test_retained_checkpoint_snapshot_survives_rolling_replace(tmp_path):
    rolling = tmp_path / "hybrid_teacher_checkpoint.pth"
    rolling.write_bytes(b"iteration-12000")
    snapshot = _retain_checkpoint_snapshot(rolling, 12_000)
    assert snapshot.read_bytes() == b"iteration-12000"
    assert snapshot.stat().st_ino == rolling.stat().st_ino
    replacement = tmp_path / "replacement.pth"
    replacement.write_bytes(b"iteration-13000")
    replacement.replace(rolling)
    assert rolling.read_bytes() == b"iteration-13000"
    assert snapshot.read_bytes() == b"iteration-12000"


def test_adaptive_geometry_scale_targets_bounded_gradient_ratio():
    assert _adaptive_geometry_scale(
        2.0, 4.0, target_ratio=0.25
    ) == 0.125
    assert _adaptive_geometry_scale(
        100.0, 0.01, target_ratio=0.25
    ) == 10.0
    assert _adaptive_geometry_scale(
        0.01, 100.0, target_ratio=0.25
    ) == 0.001
    assert _adaptive_geometry_scale(
        2.0, 4.0, target_ratio=0.0
    ) == 1.0


def test_mature_surface_handoff_continues_absolute_lr_iteration():
    assert _continued_surface_iteration(1, 24_000) == 24_001
    assert _continued_surface_iteration(1_000, 24_000) == 25_000
    with np.testing.assert_raises_regex(ValueError, "Invalid"):
        _continued_surface_iteration(0, 24_000)


def test_branch_isolated_rgb_never_routes_rigid_loss_through_mixed_image():
    structural = torch.full(
        (3, 16, 16), 0.4, requires_grad=True
    )
    mixed = torch.full((3, 16, 16), 0.2, requires_grad=True)
    target = torch.full((3, 16, 16), 0.7)
    rigid = torch.ones(16, 16)
    canopy = torch.zeros(16, 16)
    losses = _branch_isolated_rgb_losses(
        structural,
        mixed,
        target,
        background_weight=rigid,
        canopy_weight=canopy,
        rigid_weight=rigid,
        rigid_high_frequency_weight=rigid,
        lambda_dssim=0.2,
    )
    (
        losses["background"]
        + losses["rigid"]
        + losses["rigid_high_frequency"]
    ).backward()
    assert structural.grad is not None
    assert float(structural.grad.abs().sum()) > 0
    assert mixed.grad is None


def test_conditioned_context_photo_is_excluded_from_foliage_objective():
    prediction = torch.full(
        (3, 16, 16), 0.2, requires_grad=True
    )
    target = torch.full((3, 16, 16), 0.8)
    task = {
        "p_canopy": torch.zeros(16, 16),
        "p_sky": torch.ones(16, 16),
        "p_rigid": torch.zeros(16, 16),
    }
    losses = _conditioned_photo_losses(prediction, target, task)
    losses["canopy"].backward(retain_graph=True)
    assert prediction.grad is not None
    torch.testing.assert_close(
        prediction.grad, torch.zeros_like(prediction.grad)
    )
    prediction.grad.zero_()
    losses["combined"].backward()
    assert float(prediction.grad.abs().sum()) > 0
    assert float(losses["context"]) > 0


def test_boundary_evidence_is_continuous_and_never_hard_zero():
    boundary = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    weight = _boundary_evidence_weight(boundary)

    torch.testing.assert_close(
        weight,
        torch.tensor([1.0, 0.8125, 0.625, 0.4375, 0.25]),
    )
    assert bool((weight > 0).all())


def test_capacity_repair_resume_allows_only_exact_audited_contract():
    saved = {
        "maximum_volume_gaussians": 1_500_000,
        "maximum_volume_splits_per_event": 12_000,
        "geometry_weight": 0.12,
    }
    current = {
        **saved,
        "maximum_volume_gaussians": 2_000_000,
        "maximum_volume_splits_per_event": 20_000,
        "volume_densify_until_iteration": 10_000,
        "boundary_supervision_contract": (
            "continuous_semantic_boundary_evidence_floor025_no_binary_gate"
        ),
        "volume_reallocation_contract": (
            "role_matched_lineage_safe_retire_then_evidence_adaptive_split"
        ),
    }

    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_volume_capacity_repair_migration=True,
    )
    bad = {**current, "maximum_volume_gaussians": 2_100_000}
    with np.testing.assert_raises_regex(
        RuntimeError, "target contract mismatch"
    ):
        _resume_training_contract_differences(
            saved,
            bad,
            allow_volume_capacity_repair_migration=True,
        )


def test_trainer_repair_migrates_observation_immunity_contract():
    saved = {
        "dynamic_lifecycle_contract": (
            "persistent_observation_identity_survives_sampling_windows"
        ),
        "geometry_weight": 0.12,
    }
    current = {
        **saved,
        "dynamic_lifecycle_contract": DYNAMIC_LIFECYCLE_REPAIR_TARGET,
    }

    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_trainer_repair_migration=True,
    )
    assert _resume_training_contract_differences(
        saved,
        {**current, "geometry_weight": 0.13},
        allow_trainer_repair_migration=True,
    ) == {"geometry_weight"}


def test_trainer_repair_migrates_only_rigid_patch_nms_selection():
    old_selection = (
        "detached_rigid_residual_times_target_edge_local_pool"
    )
    new_selection = (
        "detached_rigid_residual_times_target_edge_spatial_nms_pool"
    )
    saved = {
        "rigid_residual_patch": {
            "weight": 0.2,
            "maximum_patches_per_view": 64,
            "patch_size": 11,
            "selection": old_selection,
        }
    }
    current = {
        "rigid_residual_patch": {
            **saved["rigid_residual_patch"],
            "selection": new_selection,
        }
    }

    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_trainer_repair_migration=True,
    )
    assert _resume_training_contract_differences(
        saved,
        current,
        allow_trainer_repair_migration=False,
    ) == {"rigid_residual_patch"}
    changed_weight = {
        "rigid_residual_patch": {
            **current["rigid_residual_patch"],
            "weight": 0.3,
        }
    }
    assert _resume_training_contract_differences(
        saved,
        changed_weight,
        allow_trainer_repair_migration=True,
    ) == {"rigid_residual_patch"}


def test_trainer_repair_extends_chart_topology_with_volume_horizon():
    saved = {
        "densify_until_iter": 3000,
        "volume_densify_until_iteration": 10000,
        "other": "fixed",
    }
    current = {
        "densify_until_iter": 10000,
        "volume_densify_until_iteration": 10000,
        "other": "fixed",
    }

    assert _resume_training_contract_differences(saved, current) == {
        "densify_until_iter"
    }
    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_trainer_repair_migration=True,
    )
    assert _resume_training_contract_differences(
        saved,
        {**current, "other": "changed"},
        allow_trainer_repair_migration=True,
    ) == {"other"}


def test_conditioned_schedule_resume_migrates_only_sampling_identity():
    saved = {
        "sampling_schedule_sha256": "legacy",
        "conditioned_sampling": {
            "coverage_preserving": False,
            "evidence_steps": 7020,
        },
        "geometry_weight": 0.12,
    }
    current = {
        "sampling_schedule_sha256": "repaired",
        "conditioned_sampling": {
            "coverage_preserving": True,
            "evidence_steps": 6339,
            "minimum_full_scene_visits": 3,
        },
        "geometry_weight": 0.12,
    }

    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_conditioned_schedule_repair_migration=True,
    )
    assert _resume_training_contract_differences(
        saved,
        current,
        allow_conditioned_schedule_repair_migration=False,
    ) == {
        "sampling_schedule_sha256",
        "conditioned_sampling",
    }
    assert _resume_training_contract_differences(
        saved,
        {**current, "geometry_weight": 0.13},
        allow_conditioned_schedule_repair_migration=True,
    ) == {"geometry_weight"}

    already_repaired = {
        **saved,
        "conditioned_sampling": {"coverage_preserving": True},
    }
    with pytest.raises(RuntimeError, match="legacy replacing schedule"):
        _resume_training_contract_differences(
            already_repaired,
            current,
            allow_conditioned_schedule_repair_migration=True,
        )


def test_surface_ownership_resume_changes_only_the_audited_contract():
    saved = {
        "mature_handoff_surface_policy": "appearance_only",
        "surface_ownership_contract": (
            "surface_opacity_follows_mature_handoff_policy"
        ),
        "geometry_weight": 0.12,
    }
    current = {
        **saved,
        **SURFACE_OWNERSHIP_REPAIR_TARGET,
    }

    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_surface_ownership_repair_migration=True,
    )
    bad = {**current, "geometry_weight": 0.13}
    assert _resume_training_contract_differences(
        saved,
        bad,
        allow_surface_ownership_repair_migration=True,
    ) == {"geometry_weight"}


def test_v38_resume_migrates_runtime_ray_capacity_and_topology_settle():
    saved = {
        "schedule_horizon": 12_000,
        "volume_densify_until_iteration": 10_000,
        "ray_posterior_maximum_rays": 1_000,
        "ray_posterior_configured_maximum_rays": 512,
        "ray_evidence_epoch_capacity": {
            "scheduled_factor_calls": 3_000,
            "effective_row_count": 2_854_438,
            "minimum_batch_with_five_percent_margin": 1_000,
            "complete_epoch_capacity": True,
        },
    }
    current = {
        "schedule_horizon": 12_000,
        "volume_densify_until_iteration": 12_000,
        "volume_topology_phase_contract": (
            "screen_adaptive_until_configured_end__disabled_during_"
            "canonical_polish_settle"
        ),
        "conditioned_settle_contract": (
            "canonical_polish_disables_topology_not_conditioned_optimization"
        ),
        "counterfactual_transparency": {
            "contract": (
                "detached_surface_only_relative_rgb_advantage_routes_only_"
                "volume_optical_depth"
            ),
            "weight": 0.06,
            "margin": 0.02,
            "temperature": 0.015,
            "conditioned_every": 4,
            "gradient_owner": "volume_opacity_only",
            "responsibility_gradient": "detached",
        },
        "ray_posterior_maximum_rays": 1_532,
        "ray_posterior_configured_maximum_rays": 512,
        "ray_evidence_epoch_capacity": {
            "scheduled_factor_calls": 3_000,
            "batch_boundary_contract": (
                "short_tail_never_wraps_into_next_camera_epoch"
            ),
            "capacity_contract": (
                "sum_per_camera_ceil_rows_over_batch_lte_scheduled_calls"
            ),
            "capacity_scope": "resume_remaining_unvisited_rows",
            "prior_factor_calls": 2_250,
            "available_factor_calls": 750,
            "camera_table_count": 751,
            "full_effective_row_count": 2_854_438,
            "remaining_unvisited_row_count": 633_251,
            "clean_epoch_aggregate_minimum_batch_with_five_percent_margin": 1_000,
            "clean_epoch_minimum_batch_for_complete_per_camera_epoch": 1_090,
            "clean_epoch_effective_batch": 1_090,
            "runtime_aggregate_minimum_batch_with_five_percent_margin": 887,
            "minimum_batch_for_runtime_completion": 1_532,
            "required_factor_calls": 750,
            "unused_factor_calls": 0,
            "complete_epoch_capacity": True,
        },
    }
    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_v38_causal_repair_migration=True,
    )
    assert _resume_training_contract_differences(saved, current) == {
        "counterfactual_transparency",
        "ray_evidence_epoch_capacity",
        "ray_posterior_maximum_rays",
        "volume_densify_until_iteration",
        "volume_topology_phase_contract",
        "conditioned_settle_contract",
    }


def test_rigid_completion_births_only_independently_supported_deficits():
    seed = {
        "xyz": np.asarray(
            [[0.01, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "scales": np.full((3, 2), 0.03, dtype=np.float32),
        "geometry_confidence": np.asarray(
            [1.0, 0.8, 1.0], dtype=np.float32
        ),
        "position_sigma": np.full(3, 0.04, dtype=np.float32),
        "pointmap_cross_sequence_supported": np.asarray(
            [True, True, False]
        ),
        "persistent_geometry_evidence": np.asarray(
            [False, False, False]
        ),
    }
    selected, audit = _rigid_completion_seed_indices(
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        seed,
        maximum_seeds=4,
    )
    assert selected.tolist() == [1]
    assert audit["candidate_count"] == 1
    assert audit["cross_sequence_pointmap_selected"] == 1
    disabled, disabled_audit = _rigid_completion_seed_indices(
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        seed,
        maximum_seeds=0,
    )
    assert disabled.tolist() == []
    assert not disabled_audit["enabled"]


def test_conditioned_base_gradient_gate_excludes_canonical_forward_owners():
    foliage = SimpleNamespace(
        dynamic_leaf_mask=torch.tensor(
            [False, False, True, True, True]
        ),
        xyz=torch.zeros(5, 3),
    )
    gate = _conditioned_base_gradient_gate(
        foliage,
        torch.tensor([1.0, 1.0, 1.0, 0.35, 0.0]),
    )
    torch.testing.assert_close(
        gate,
        torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
    )


def test_conditioned_base_gradient_gate_rejects_misaligned_topology():
    foliage = SimpleNamespace(
        dynamic_leaf_mask=torch.tensor([False, True]),
        xyz=torch.zeros(2, 3),
    )
    with np.testing.assert_raises_regex(ValueError, "one value"):
        _conditioned_base_gradient_gate(
            foliage, torch.ones(3)
        )


def test_evidence_adaptive_role_quotas_normalize_producer_density():
    quotas = _evidence_adaptive_role_quotas(
        {
            "static_skeleton": 59,
            "canonical_crown": 32_781,
            "dynamic_leaf": 136_302,
        },
        {
            "static_skeleton": 403,
            "canonical_crown": 122_391,
            "dynamic_leaf": 938_283,
        },
        12_000,
    )
    assert quotas["static_skeleton"] == 59
    assert quotas["canonical_crown"] > quotas["dynamic_leaf"]
    assert sum(quotas.values()) == 12_000


def test_evidence_adaptive_role_quotas_ignore_duplicate_row_multiplier():
    eligible = {
        "static_skeleton": 100,
        "canonical_crown": 4_000,
        "dynamic_leaf": 8_000,
    }
    population = {
        "static_skeleton": 1_000,
        "canonical_crown": 40_000,
        "dynamic_leaf": 80_000,
    }
    baseline = _evidence_adaptive_role_quotas(
        eligible, population, 6_000
    )
    duplicated = _evidence_adaptive_role_quotas(
        {**eligible, "dynamic_leaf": 80_000},
        {**population, "dynamic_leaf": 800_000},
        6_000,
    )
    assert baseline == duplicated


def test_evidence_adaptive_role_quotas_use_observed_dynamic_population():
    quotas = _evidence_adaptive_role_quotas(
        {
            "static_skeleton": 0,
            "canonical_crown": 4_000,
            "dynamic_leaf": 8_000,
        },
        {
            "static_skeleton": 1,
            "canonical_crown": 40_000,
            "dynamic_leaf": 800_000,
        },
        6_000,
        observable_counts={
            "static_skeleton": 0,
            "canonical_crown": 20_000,
            "dynamic_leaf": 10_000,
        },
    )
    # Unvisited owner-camera rows must not dilute the deficit measured on
    # exact dynamic contexts in this topology epoch.
    assert quotas["dynamic_leaf"] > quotas["canonical_crown"]
    assert sum(quotas.values()) == 6_000


def test_evidence_adaptive_role_quotas_use_continuous_geometry_authority():
    quotas = _evidence_adaptive_role_quotas(
        {
            "static_skeleton": 0,
            "canonical_crown": 4_000,
            "dynamic_leaf": 8_000,
        },
        {
            "static_skeleton": 1,
            "canonical_crown": 40_000,
            "dynamic_leaf": 800_000,
        },
        6_000,
        observable_counts={
            "static_skeleton": 0,
            "canonical_crown": 20_000,
            "dynamic_leaf": 10_000,
        },
        effective_eligible_mass={
            "static_skeleton": 0.0,
            "canonical_crown": 4_000.0,
            # Every visible dynamic row has a gradient, but its weak depth
            # posterior supplies only five percent effective authority.
            "dynamic_leaf": 400.0,
        },
    )
    assert quotas["canonical_crown"] > quotas["dynamic_leaf"]
    assert sum(quotas.values()) == 6_000


def test_volume_topology_authority_is_continuous_and_owner_aware():
    foliage = SimpleNamespace(
        ray_depth_nll=torch.tensor([0.0, 15.0, 15.0, 399.0]),
        occupancy_probability=torch.tensor([0.0, 0.0, 0.10, 0.0]),
        dynamic_leaf_mask=torch.tensor([False, True, True, True]),
        replacement_group=torch.tensor([-1, 7, -1, -1]),
    )
    authority = _volume_topology_authority(foliage)
    torch.testing.assert_close(
        authority,
        torch.tensor([1.0, 0.25, 0.03625, 0.0025]),
    )
    assert bool((authority > 0).all())


def test_exact_ray_optical_split_authority_does_not_claim_better_depth():
    foliage = SimpleNamespace(
        ray_depth_nll=torch.tensor([399.0, 399.0]),
        occupancy_probability=torch.tensor([0.01, 0.01]),
        dynamic_leaf_mask=torch.tensor([True, True]),
        replacement_group=torch.tensor([-1, -1]),
        initialization_source=torch.tensor([4, 3], dtype=torch.int8),
        observation_camera_ids=torch.tensor(
            [[7], [7]], dtype=torch.int32
        ),
    )

    geometry = _volume_topology_authority(foliage)
    split = _volume_split_authority(foliage)

    assert geometry.tolist() == pytest.approx([0.002975, 0.002975])
    assert split[0].item() == 1.0
    assert split[1].item() == pytest.approx(0.002975)


def test_evidence_adaptive_role_quotas_never_invent_or_drop_capacity():
    assert _evidence_adaptive_role_quotas(
        {
            "static_skeleton": 3,
            "canonical_crown": 0,
            "dynamic_leaf": 9,
        },
        {
            "static_skeleton": 30,
            "canonical_crown": 1,
            "dynamic_leaf": 90,
        },
        6,
    ) == {
        "static_skeleton": 3,
        "canonical_crown": 0,
        "dynamic_leaf": 3,
    }
    assert _evidence_adaptive_role_quotas(
        {
            "static_skeleton": 3,
            "canonical_crown": 0,
            "dynamic_leaf": 9,
        },
        {
            "static_skeleton": 30,
            "canonical_crown": 1,
            "dynamic_leaf": 90,
        },
        20,
    ) == {
        "static_skeleton": 3,
        "canonical_crown": 0,
        "dynamic_leaf": 9,
    }
    with np.testing.assert_raises_regex(ValueError, "non-negative"):
        _evidence_adaptive_role_quotas(
            {
                "static_skeleton": -1,
                "canonical_crown": 0,
                "dynamic_leaf": 1,
            },
            {
                "static_skeleton": 1,
                "canonical_crown": 1,
                "dynamic_leaf": 1,
            },
            1,
        )
    with np.testing.assert_raises_regex(ValueError, "exceeds"):
        _evidence_adaptive_role_quotas(
            {
                "static_skeleton": 2,
                "canonical_crown": 0,
                "dynamic_leaf": 0,
            },
            {
                "static_skeleton": 1,
                "canonical_crown": 1,
                "dynamic_leaf": 1,
            },
            1,
        )


def test_resume_contract_allows_only_camera_container_cache_digest_change():
    saved = {
        "camera_geometry_sha256": "same-geometry",
        "fixed_camera_validation": {"exact_image_identity": True},
        "camera_intrinsics_contract_sha256": "cache-four",
        "geometry_weight": 0.12,
    }
    current = {
        **saved,
        "camera_intrinsics_contract_sha256": "cache-all",
    }
    assert not _resume_training_contract_differences(saved, current)
    changed_geometry = {
        **current,
        "camera_geometry_sha256": "different-geometry",
    }
    assert _resume_training_contract_differences(
        saved, changed_geometry
    ) == {
        "camera_geometry_sha256",
        "camera_intrinsics_contract_sha256",
    }
    changed_loss = {**current, "geometry_weight": 0.20}
    assert _resume_training_contract_differences(
        saved, changed_loss
    ) == {
        "camera_intrinsics_contract_sha256",
        "geometry_weight",
    }


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


def test_handoff_profile_starts_foliage_without_rebuilding_rigid_surface():
    profile = TRAINING_PROFILES["hybrid_handoff_quality"]

    assert profile["iterations"] == 30_000
    assert profile["foliage_start"] == 0.0
    assert _phase(599, 30_000, "hybrid_handoff_quality") == (
        "canonical_bootstrap"
    )
    assert _phase(600, 30_000, "hybrid_handoff_quality") == "topology"
    assert _phase(7_499, 30_000, "hybrid_handoff_quality") == "topology"
    assert _phase(7_500, 30_000, "hybrid_handoff_quality") == (
        "static_foliage"
    )
    assert not _dynamic_enabled(
        599, 30_000, "hybrid_handoff_quality"
    )
    assert _dynamic_enabled(
        600, 30_000, "hybrid_handoff_quality"
    )


def test_absolute_schedule_makes_short_run_an_exact_method_prefix():
    profile = "hybrid_handoff_quality"
    horizon = 12_000
    assert _resolved_phase_schedule(profile, horizon) == (
        ("canonical_bootstrap", 240),
        ("topology", 3_000),
        ("static_foliage", 4_200),
        ("dynamic_appearance", 8_640),
        ("ownership_cleanup", 11_040),
        ("canonical_polish", 12_000),
    )
    assert _activation_iteration(0.02, horizon) == 241
    short = _full_epoch_schedule(1487, 4_000, 1702)
    complete = _full_epoch_schedule(1487, horizon, 1702)
    np.testing.assert_array_equal(short, complete[:4_000])
    for step in (0, 239, 240, 2_999, 3_000, 3_999):
        # The requested stop iteration is deliberately absent: phase and
        # branch identity depend only on the persisted method horizon.
        assert _phase(step, horizon, profile) == _phase(
            step, horizon, profile
        )
        assert _dynamic_enabled(step, horizon, profile) == (
            step >= 240
        )
        assert _foliage_enabled(step, horizon, profile)


def test_surface_topology_schedule_can_extend_past_named_topology_phase():
    args = SimpleNamespace(
        densify_from_iter=500,
        densify_until_iter=10_500,
    )
    assert _phase(8_999, 30_000, "hybrid_handoff_quality") == (
        "static_foliage"
    )
    assert _surface_topology_active(8_999, args)
    assert _surface_topology_active(10_499, args)
    assert not _surface_topology_active(10_500, args)


def test_volume_topology_stops_before_canonical_polish_settle():
    args = SimpleNamespace(volume_densify_until_iteration=12_000)
    assert _volume_topology_active(10_999, args, "ownership_cleanup")
    assert not _volume_topology_active(11_040, args, "canonical_polish")
    assert not _volume_topology_active(12_000, args, "ownership_cleanup")


def test_conditioned_branch_keeps_training_during_topology_settle():
    horizon = 12_000
    profile = "hybrid_handoff_quality"
    assert _phase(11_040, horizon, profile) == "canonical_polish"
    assert _conditioned_branch_active(11_040, horizon, profile)
    assert _conditioned_branch_active(11_999, horizon, profile)


def test_conditioned_bases_keep_gradients_during_topology_settle():
    foliage = SimpleNamespace(
        static_skeleton_mask=torch.tensor([True, False, False]),
        dynamic_leaf_mask=torch.tensor([False, True, False]),
        static_skeleton_confidence=torch.tensor([1.0, 1.0, 1.0]),
        xyz=torch.zeros(3, 3, requires_grad=True),
        log_scales=torch.zeros(3, 3, requires_grad=True),
        opacity_logits=torch.zeros(3, 1, requires_grad=True),
        deformation_basis=torch.zeros(3, 2, requires_grad=True),
        dynamic_feature_basis=torch.zeros(3, 2, requires_grad=True),
        dynamic_opacity_basis=torch.zeros(3, 2, requires_grad=True),
    )
    for parameter in (
        foliage.xyz,
        foliage.log_scales,
        foliage.opacity_logits,
        foliage.deformation_basis,
        foliage.dynamic_feature_basis,
        foliage.dynamic_opacity_basis,
    ):
        parameter.grad = torch.ones_like(parameter)

    _apply_volume_role_gradients(
        foliage, "canonical_polish", dynamic_active=True
    )

    for parameter in (
        foliage.deformation_basis,
        foliage.dynamic_feature_basis,
        foliage.dynamic_opacity_basis,
    ):
        assert torch.all(parameter.grad[foliage.dynamic_leaf_mask] == 1)
        assert torch.all(parameter.grad[~foliage.dynamic_leaf_mask] == 0)


def test_volume_opacity_freeze_keeps_geometry_and_colour_trainable():
    foliage = SimpleNamespace(
        opacity_logits=torch.nn.Parameter(torch.zeros(3, 1)),
        dynamic_opacity_basis=torch.nn.Parameter(torch.zeros(3, 2)),
        xyz=torch.nn.Parameter(torch.zeros(3, 3)),
        features=torch.nn.Parameter(torch.zeros(3, 3)),
    )
    for parameter in (
        foliage.opacity_logits,
        foliage.dynamic_opacity_basis,
        foliage.xyz,
        foliage.features,
    ):
        parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.Adam(
        [
            foliage.opacity_logits,
            foliage.dynamic_opacity_basis,
            foliage.xyz,
            foliage.features,
        ],
        lr=1e-3,
    )
    args = SimpleNamespace(
        volume_opacity_settle_policy="freeze",
        volume_opacity_settle_start_iteration=6_000,
        volume_opacity_retirement_until_iteration=6_000,
    )
    before = _apply_volume_opacity_settle_policy(
        foliage, optimizer, 5_999, args
    )
    assert not before["active"]
    assert foliage.opacity_logits.grad is not None

    audit = _apply_volume_opacity_settle_policy(
        foliage, optimizer, 6_000, args
    )
    assert audit["active"]
    assert audit["base_rows_frozen"] == 3
    assert audit["temporal_rows_frozen"] == 3
    assert foliage.opacity_logits.grad is None
    assert foliage.dynamic_opacity_basis.grad is None
    assert foliage.xyz.grad is not None
    assert foliage.features.grad is not None


def test_volume_opacity_retirement_only_removes_growth_gradient_and_momentum():
    opacity = torch.nn.Parameter(torch.zeros(3, 1))
    temporal = torch.nn.Parameter(torch.zeros(3, 2))
    foliage = SimpleNamespace(
        opacity_logits=opacity,
        dynamic_opacity_basis=temporal,
    )
    opacity.grad = torch.tensor([[-2.0], [1.0], [-0.5]])
    temporal.grad = torch.ones_like(temporal)
    optimizer = torch.optim.Adam([opacity, temporal], lr=1e-3)
    optimizer.state[opacity]["exp_avg"] = torch.tensor(
        [[-0.25], [0.50], [-0.10]]
    )
    args = SimpleNamespace(
        volume_opacity_settle_policy="retirement_only",
        volume_opacity_settle_start_iteration=6_000,
        volume_opacity_retirement_until_iteration=9_000,
    )
    audit = _apply_volume_opacity_settle_policy(
        foliage, optimizer, 6_000, args
    )
    torch.testing.assert_close(
        opacity.grad, torch.tensor([[0.0], [1.0], [0.0]])
    )
    torch.testing.assert_close(
        optimizer.state[opacity]["exp_avg"],
        torch.tensor([[0.0], [0.50], [0.0]]),
    )
    assert audit["base_growth_rows_suppressed"] == 2
    assert audit["base_growth_momentum_entries_suppressed"] == 2
    assert temporal.grad is None

    opacity.grad = torch.ones_like(opacity)
    temporal.grad = torch.ones_like(temporal)
    after_window = _apply_volume_opacity_settle_policy(
        foliage, optimizer, 9_000, args
    )
    assert after_window["effective_policy"] == "freeze"
    assert opacity.grad is None
    assert temporal.grad is None


def test_known_invalid_free_space_initialization_cannot_train():
    with np.testing.assert_raises_regex(
        RuntimeError, "invalid free-space contradiction metadata"
    ):
        _validate_initialization_protocol(
            {
                "version": (
                    "outdoor-role-aware-initialization-v14-priority-"
                    "dav2-hole-fill"
                )
            }
        )
    _validate_initialization_protocol(
        {
            "version": (
                "outdoor-role-aware-initialization-v15-"
                "confirmed-free-space"
            )
        }
    )


def test_exact_dynamic_rgb_initialization_requires_matching_training_rgb():
    contract = {
        "image_count": 3,
        "name_set_sha256": "names",
        "content_mapping_sha256": "content",
        "producer_manifest_sha256": "producer",
        "target_storage": "uint8",
        "canonical_image_size_wh": [640, 360],
    }
    initialization = {
        "rgb_source": dict(contract),
        "foliage": {
            "dense_dynamic_rgb_source": "exact_training_target_raster"
        },
    }
    _validate_initialization_rgb_source(initialization, dict(contract))
    mismatched = dict(contract)
    mismatched["content_mapping_sha256"] = "different"
    with np.testing.assert_raises_regex(
        RuntimeError, "RGB source contract mismatch"
    ):
        _validate_initialization_rgb_source(initialization, mismatched)
    with np.testing.assert_raises_regex(
        RuntimeError, "no persisted RGB source contract"
    ):
        _validate_initialization_rgb_source(
            {"foliage": initialization["foliage"]}, contract
        )


def test_native_rigid_surface_handoff_requires_exact_digest_and_no_colmap(
    tmp_path,
):
    surface = tmp_path / "surface.ply"
    surface.write_bytes(b"rigid surface")
    digest = hashlib.sha256(surface.read_bytes()).hexdigest()
    manifest = tmp_path / "handoff.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "native-rigid-surface-handoff-v1",
                "eligible_for_hybrid_surface_handoff": True,
                "rejection_reasons": [],
                "surface_ply": str(surface.resolve()),
                "surface_ply_sha256": digest,
                "camera_geometry_sha256": "camera",
                "historical_gaussian_input_used": False,
                "colmap_points_or_tracks_used": False,
                "surface_ownership_contract": (
                    "rigid_pixels_tree_sky_transient_excluded"
                ),
                "surface_optimizer": RIGID_SURFACE_OPTIMIZER,
            }
        ),
        encoding="utf-8",
    )

    accepted = _validate_surface_warmstart(surface, manifest)

    assert accepted["surface_ply_sha256"] == digest
    assert accepted["handoff_manifest_sha256"]
    assert accepted["surface_optimizer"] == RIGID_SURFACE_OPTIMIZER
    opt = SimpleNamespace(position_lr_init=1.6e-4)
    inherited = _inherit_surface_optimizer_contract(opt, accepted)
    assert inherited == RIGID_SURFACE_OPTIMIZER
    assert opt.position_lr_init == 1.6e-5
    assert opt.position_lr_max_steps == 20_000
    surface.write_bytes(b"mutated")
    with np.testing.assert_raises_regex(RuntimeError, "digest mismatch"):
        _validate_surface_warmstart(surface, manifest)


def test_native_rigid_handoff_accepts_explicit_sfm_coverage_only(tmp_path):
    surface = tmp_path / "surface.ply"
    surface.write_bytes(b"mast3r-primary plus sparse coverage")
    manifest = tmp_path / "handoff.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "native-rigid-surface-handoff-v1",
                "eligible_for_hybrid_surface_handoff": True,
                "rejection_reasons": [],
                "surface_ply": str(surface.resolve()),
                "surface_ply_sha256": hashlib.sha256(
                    surface.read_bytes()
                ).hexdigest(),
                "historical_gaussian_input_used": False,
                "colmap_points_or_tracks_used": True,
                "sfm_track_usage_mode": "coverage_only",
                "geometry_authority": "mast3r_matcha_primary",
                "surface_ownership_contract": (
                    "rigid_pixels_tree_sky_transient_excluded"
                ),
                "surface_optimizer": RIGID_SURFACE_OPTIMIZER,
            }
        ),
        encoding="utf-8",
    )
    accepted = _validate_surface_warmstart(surface, manifest)
    assert accepted["colmap_points_or_tracks_used"]
    assert accepted["sfm_track_usage_mode"] == "coverage_only"


def test_legacy_rigid_handoff_recovers_validated_producer_schedule(tmp_path):
    surface = tmp_path / "surface.ply"
    surface.write_bytes(b"legacy rigid surface")
    digest = hashlib.sha256(surface.read_bytes()).hexdigest()
    manifest = tmp_path / "rigid_surface_handoff.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "native-rigid-surface-handoff-v1",
                "eligible_for_hybrid_surface_handoff": True,
                "surface_ply": str(surface.resolve()),
                "surface_ply_sha256": digest,
                "surface_iteration": 24_000,
                "historical_gaussian_input_used": False,
                "colmap_points_or_tracks_used": False,
                "surface_ownership_contract": (
                    "rigid_pixels_tree_sky_transient_excluded"
                ),
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "iterations": 24_000,
                "surface_ply": str(surface.resolve()),
                "training_contract": {
                    "surface_optimizer": RIGID_SURFACE_OPTIMIZER
                },
            }
        ),
        encoding="utf-8",
    )

    accepted = _validate_surface_warmstart(surface, manifest)

    assert accepted["surface_optimizer"] == RIGID_SURFACE_OPTIMIZER
    assert accepted["surface_optimizer_source"] == (
        "validated_legacy_producer_result"
    )


def test_geometry_trained_rigid_stage_writes_eligible_handoff(tmp_path):
    output = tmp_path / "rigid"
    point_cloud = output / "point_cloud" / "iteration_24000"
    point_cloud.mkdir(parents=True)
    surface = point_cloud / "point_cloud.ply"
    surface.write_bytes(b"geometry-trained rigid surface")
    initialization_directory = tmp_path / "initialization"
    initialization_directory.mkdir()
    initialization_manifest = (
        initialization_directory / "initialization_manifest.json"
    )
    initialization = {
        "historical_model_initialization": False,
        "surface": {
            "historical_trained_ply_used": False,
            "colmap_points_or_tracks_used": False,
            "canopy_surface_seed_count": 0,
        },
    }
    initialization_manifest.write_text(
        json.dumps(initialization), encoding="utf-8"
    )
    camera_contract = {
        "camera_geometry_sha256": "exact-camera-geometry"
    }
    (output / "camera_intrinsics_contract.json").write_text(
        json.dumps(camera_contract), encoding="utf-8"
    )

    destination = _write_rigid_stage_surface_handoff(
        output=output,
        iteration=24_000,
        surface_iteration=48_000,
        surface_point_count=321,
        initialization=initialization,
        initialization_directory=initialization_directory,
        evidence_store={
            "geometry_source": "mast3r_only",
            "colmap_points_or_tracks_used": False,
        },
        camera_runtime_contract=camera_contract,
        fixed_camera_validation={"camera_count": 1487},
        geometry_audit={
            "source_consumption_count": {
                "chart_native_factor": 12_000,
                "track_observation_factor": 12_000,
            }
        },
        surface_optimizer=RIGID_SURFACE_OPTIMIZER,
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["eligible_for_hybrid_surface_handoff"]
    assert payload["producer"] == "hybrid_rigid_stage1"
    assert payload["surface_iteration"] == 48_000
    assert payload["surface_ownership_contract"] == (
        "rigid_pixels_tree_sky_transient_excluded"
    )
    accepted = _validate_surface_warmstart(surface, destination)
    assert accepted["camera_geometry_sha256"] == "exact-camera-geometry"
    assert accepted["surface_optimizer_source"] == "handoff_manifest"


def test_legacy_mast3r_only_store_without_colmap_boolean_is_eligible(
    tmp_path,
):
    """Legacy stores expressed the same guarantee without a boolean field."""
    output = tmp_path / "rigid"
    point_cloud = output / "point_cloud" / "iteration_24000"
    point_cloud.mkdir(parents=True)
    surface = point_cloud / "point_cloud.ply"
    surface.write_bytes(b"geometry-trained rigid surface")
    initialization_directory = tmp_path / "initialization"
    initialization_directory.mkdir()
    initialization = {
        "historical_model_initialization": False,
        "surface": {
            "historical_trained_ply_used": False,
            "colmap_points_or_tracks_used": False,
            "canopy_surface_seed_count": 0,
        },
    }
    (initialization_directory / "initialization_manifest.json").write_text(
        json.dumps(initialization), encoding="utf-8"
    )
    camera_contract = {
        "camera_geometry_sha256": "exact-camera-geometry"
    }
    (output / "camera_intrinsics_contract.json").write_text(
        json.dumps(camera_contract), encoding="utf-8"
    )

    destination = _write_rigid_stage_surface_handoff(
        output=output,
        iteration=24_000,
        surface_point_count=321,
        initialization=initialization,
        initialization_directory=initialization_directory,
        evidence_store={
            "geometry_source": "mast3r_only",
            "artifacts": [
                {
                    "name": "mast3r_track_graph",
                    "source_type": "mast3r_fixed_camera_track_graph",
                    "coordinate_frame": "COLMAP_world",
                }
            ],
        },
        camera_runtime_contract=camera_contract,
        fixed_camera_validation={"camera_count": 1487},
        geometry_audit={
            "source_consumption_count": {
                "chart_native_factor": 12_000,
                "track_observation_factor": 12_000,
            }
        },
        surface_optimizer=RIGID_SURFACE_OPTIMIZER,
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["eligible_for_hybrid_surface_handoff"]
    assert not payload["colmap_points_or_tracks_used"]


def test_legacy_store_with_colmap_track_artifact_fails_closed(tmp_path):
    output = tmp_path / "rigid"
    point_cloud = output / "point_cloud" / "iteration_24000"
    point_cloud.mkdir(parents=True)
    (point_cloud / "point_cloud.ply").write_bytes(b"surface")
    initialization_directory = tmp_path / "initialization"
    initialization_directory.mkdir()
    initialization = {
        "historical_model_initialization": False,
        "surface": {
            "historical_trained_ply_used": False,
            "colmap_points_or_tracks_used": False,
            "canopy_surface_seed_count": 0,
        },
    }
    (initialization_directory / "initialization_manifest.json").write_text(
        json.dumps(initialization), encoding="utf-8"
    )
    camera_contract = {
        "camera_geometry_sha256": "exact-camera-geometry"
    }
    (output / "camera_intrinsics_contract.json").write_text(
        json.dumps(camera_contract), encoding="utf-8"
    )

    destination = _write_rigid_stage_surface_handoff(
        output=output,
        iteration=24_000,
        surface_point_count=1,
        initialization=initialization,
        initialization_directory=initialization_directory,
        evidence_store={
            "geometry_source": "mast3r_only",
            "artifacts": [
                {
                    "name": "colmap_tracks",
                    "source_type": "colmap_tracks",
                }
            ],
        },
        camera_runtime_contract=camera_contract,
        fixed_camera_validation={"camera_count": 1487},
        geometry_audit={
            "source_consumption_count": {
                "chart_native_factor": 1,
                "track_observation_factor": 1,
            }
        },
        surface_optimizer=RIGID_SURFACE_OPTIMIZER,
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert not payload["eligible_for_hybrid_surface_handoff"]
    assert payload["colmap_points_or_tracks_used"]


def test_fixed_camera_contract_checks_exact_scaled_k_and_pose():
    view = SimpleNamespace(
        image_name="frame.png",
        image_width=640,
        image_height=360,
        focal_x=400.0,
        focal_y=420.0,
        cx=321.0,
        cy=179.0,
        R=np.eye(3),
        T=np.asarray([1.0, 2.0, 3.0]),
    )
    contract = {
        "records": [
            {
                "image_name": "frame.png",
                "camera": {
                    "width": 1280,
                    "height": 720,
                    "fx": 800.0,
                    "fy": 840.0,
                    "cx": 642.0,
                    "cy": 358.0,
                },
                "T_world_to_camera": [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        ]
    }
    audit = _validate_fixed_cameras([view], contract)
    assert audit["camera_count"] == 1
    assert max(audit["maximum_absolute_error"].values()) == 0.0

    bad_view = SimpleNamespace(**vars(view))
    bad_view.focal_x += 0.1
    with np.testing.assert_raises_regex(
        RuntimeError, "camera calibration differs"
    ):
        _validate_fixed_cameras([bad_view], contract)


def test_contradiction_fraction_uses_all_confirmed_binary_evidence():
    positive = torch.tensor([4, 1, 0], dtype=torch.int16)
    confirmed_free = torch.tensor([3, 1, 2], dtype=torch.int16)

    ratio = _confirmed_contradiction_fraction(
        positive, confirmed_free
    )

    assert torch.allclose(
        ratio, torch.tensor([3.0 / 7.0, 0.5, 1.0])
    )
    assert ratio[0] < 0.55


def test_strong_contradiction_retires_observed_dynamic_identity():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor(
            [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]
        ),
        "scales": torch.full((2, 3), 0.05),
        "colors": torch.full((2, 3), 0.4),
        "opacities": torch.full((2, 1), 0.001),
        "quaternions": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 2
        ),
        "layer_role": torch.tensor(
            [LAYER_DYNAMIC_LEAF, 0], dtype=torch.int8
        ),
        "support_camera_ids": torch.tensor(
            [[7], [8]], dtype=torch.int32
        ),
        "observation_camera_ids": torch.tensor(
            [[7], [-1]], dtype=torch.int32
        ),
        "observation_uv": torch.tensor(
            [[[0.5, 0.5]], [[0.0, 0.0]]]
        ),
        "observation_depth": torch.tensor([[2.0], [0.0]]),
        "support_view_count": torch.tensor([1, 2], dtype=torch.int16),
        "support_sequence_count": torch.tensor(
            [1, 2], dtype=torch.int16
        ),
        "free_space_violation_count": torch.tensor(
            [10, 0], dtype=torch.int16
        ),
        "occupancy_probability": torch.tensor([0.01, 0.9]),
    }
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)
    stats = _volume_stats(foliage)

    event = _adapt_volume(
        SimpleNamespace(
            volume_split_radius=3.0,
            maximum_volume_gaussians=2,
            maximum_volume_splits=0,
        ),
        foliage,
        stats,
        volume_budget=2,
    )

    assert len(foliage) == 1
    assert event["pruned"] == 1
    prune = event["contradiction_prune_evidence"]
    assert prune["strong_contradiction_candidates"] == 1
    assert prune["observed_unverified_rows_prunable"] == 1


def test_full_volume_budget_settles_without_reallocation_churn():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]),
        "scales": torch.full((2, 3), 0.05),
        "colors": torch.full((2, 3), 0.4),
        "opacities": torch.full((2, 1), 0.2),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
        "layer_role": torch.tensor([0, 0], dtype=torch.int8),
        "support_view_count": torch.tensor([4, 4], dtype=torch.int16),
        "support_sequence_count": torch.tensor([2, 2], dtype=torch.int16),
        "occupancy_probability": torch.tensor([0.9, 0.9]),
    }
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)
    stats = _volume_stats(foliage)
    stats["radius"].fill_(10.0)
    stats["contribution"].fill_(1.0)
    stats["residual"].fill_(1.0)
    stats["gradient"].fill_(1.0)
    stats["gradient_count"].fill_(1.0)

    event = _adapt_volume(
        SimpleNamespace(
            volume_split_radius=3.0,
            maximum_volume_gaussians=2,
            maximum_volume_splits=2,
        ),
        foliage,
        stats,
        volume_budget=2,
    )

    assert len(foliage) == 2
    assert event["requested_splits"] == 2
    assert event["split_parents"] == 0
    assert event["capacity_reallocation"]["selected"] == 0
    assert event["saturated_budget_settle"] is True


def test_weak_conflict_keeps_independent_cross_sequence_positive_support():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor(
            [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]
        ),
        "scales": torch.full((2, 3), 0.05),
        "colors": torch.full((2, 3), 0.4),
        "opacities": torch.full((2, 1), 0.001),
        "quaternions": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 2
        ),
        "layer_role": torch.tensor(
            [LAYER_DYNAMIC_LEAF, 0], dtype=torch.int8
        ),
        "support_camera_ids": torch.tensor(
            [[7, 9, 10, 11], [8, 12, -1, -1]], dtype=torch.int32
        ),
        "observation_camera_ids": torch.tensor(
            [[7, 9], [-1, -1]], dtype=torch.int32
        ),
        "observation_uv": torch.tensor(
            [
                [[0.5, 0.5], [0.6, 0.5]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
        "observation_depth": torch.tensor(
            [[2.0, 2.1], [0.0, 0.0]]
        ),
        "support_view_count": torch.tensor([4, 2], dtype=torch.int16),
        "support_sequence_count": torch.tensor(
            [2, 2], dtype=torch.int16
        ),
        "free_space_violation_count": torch.tensor(
            [2, 0], dtype=torch.int16
        ),
        "occupancy_probability": torch.tensor([0.10, 0.9]),
    }
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)

    event = _adapt_volume(
        SimpleNamespace(
            volume_split_radius=3.0,
            maximum_volume_gaussians=2,
            maximum_volume_splits=0,
        ),
        foliage,
        _volume_stats(foliage),
        volume_budget=2,
    )

    assert len(foliage) == 2
    assert event["pruned"] == 0
    prune = event["contradiction_prune_evidence"]
    assert prune["weak_positive_rows_protected"] == 1


def test_read_only_temporal_fallback_cannot_drive_volume_topology():
    class FoliageStub:
        xyz = torch.zeros(2, 3)

        def __len__(self):
            return 2

    stats = _volume_stats(FoliageStub())
    means2d = torch.zeros(2, 3, requires_grad=True)
    means2d.grad = torch.tensor(
        [[2.0, 0.0, 0.0], [9.0, 0.0, 0.0]]
    )
    package = SimpleNamespace(
        structural_count=0,
        radii=torch.tensor([5.0, 7.0]),
        volume_means2d=means2d,
        responsibility=torch.tensor(
            [[3.0, 0.0, 2.0, 4.0], [8.0, 0.0, 6.0, 7.0]]
        ),
    )

    _accumulate_volume_stats(
        stats,
        package,
        ownership_gate=torch.tensor([1.0, 0.0]),
        exact_conditioned_dynamic=True,
    )

    assert stats["radius"].tolist() == [5.0, 0.0]
    assert stats["gradient"].tolist() == [2.0, 0.0]
    assert stats["gradient_count"].tolist() == [1.0, 0.0]
    assert stats["conditioned_radius"].tolist() == [5.0, 0.0]
    assert stats["conditioned_gradient"].tolist() == [2.0, 0.0]
    assert stats["conditioned_gradient_count"].tolist() == [1.0, 0.0]
    assert stats["contribution"].tolist() == [3.0, 0.0]
    assert stats["rigid"].tolist() == [2.0, 0.0]
    assert stats["residual"].tolist() == [4.0, 0.0]


def test_conditioned_backward_preserves_independent_screen_gradient_leaf():
    foliage = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        foliage.weight.fill_(2.0)
    volume_means2d = torch.zeros(1, 3, requires_grad=True)
    loss = (
        foliage(torch.ones(1, 1)).sum()
        + 3.0 * volume_means2d[:, :2].sum()
    )

    _backward_conditioned_foliage(loss, foliage, volume_means2d)

    assert foliage.weight.grad is not None
    assert volume_means2d.grad is not None
    assert volume_means2d.grad[:, :2].tolist() == [[3.0, 3.0]]


def test_fallback_opacity_can_retire_but_cannot_establish_occupancy():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor(
            [
                [0.0, 0.0, 2.0],
                [0.1, 0.0, 2.0],
                [0.2, 0.0, 2.0],
                [0.3, 0.0, 2.0],
            ]
        ),
        "scales": torch.full((4, 3), 0.05),
        "colors": torch.full((4, 3), 0.4),
        "opacities": torch.full((4, 1), 0.015),
        "quaternions": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 4
        ),
        "layer_role": torch.tensor([0, 2, 2, 2], dtype=torch.int8),
    }
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)
    foliage.opacity_logits.grad = torch.tensor(
        [[-1.0], [-2.0], [3.0], [-4.0]]
    )
    foliage.dynamic_opacity_basis.grad = torch.ones_like(
        foliage.dynamic_opacity_basis
    )

    audit = _enforce_retirement_only_fallback_opacity_gradient(
        foliage,
        visibility_gate=torch.tensor([1.0, 0.5, 0.4, 1.0]),
        exact_gradient_gate=torch.tensor([1.0, 0.0, 0.0, 1.0]),
    )

    # Canonical and exact-owner rows remain unconstrained. The negative
    # fallback gradient that would grow opacity is removed, while the
    # positive retirement gradient remains.
    assert foliage.opacity_logits.grad[:, 0].tolist() == [
        -1.0,
        0.0,
        3.0,
        -4.0,
    ]
    assert not bool(
        foliage.dynamic_opacity_basis.grad[1:3].any()
    )
    assert bool(foliage.dynamic_opacity_basis.grad[[0, 3]].all())
    assert audit["fallback_opacity_rows"] == 2
    assert audit["fallback_opacity_increase_rows_suppressed"] == 1
    assert (
        audit["fallback_opacity_increase_gradient_suppressed"] == 2.0
    )
    assert audit["fallback_temporal_opacity_rows_suppressed"] == 2


def test_dynamic_opacity_floor_uses_local_dense_ray_evidence():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor(
            [[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]
        ),
        "scales": torch.full((2, 3), 0.05),
        "colors": torch.full((2, 3), 0.4),
        "opacities": torch.full((2, 1), 0.015),
        "quaternions": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 2
        ),
        "layer_role": torch.full(
            (2,), LAYER_DYNAMIC_LEAF, dtype=torch.int8
        ),
        "initialization_source": torch.tensor(
            [4, 1], dtype=torch.int8
        ),
        "occupancy_probability": torch.tensor([0.9, 0.9]),
        "support_view_count": torch.tensor(
            [2, 2], dtype=torch.int16
        ),
        "unknown_view_count": torch.zeros(2, dtype=torch.int16),
        "ray_depth_nll": torch.zeros(2),
        "free_space_violation_count": torch.zeros(
            2, dtype=torch.int16
        ),
    }
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)

    floor = _dynamic_opacity_floor(foliage)

    assert 0.04 < floor[0] <= 0.06
    assert floor[1] == 0.015


def test_uncertain_ownerless_dense_birth_has_spatial_opacity_ceiling():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor(
            [
                [0.0, 0.0, 2.0],
                [0.1, 0.0, 2.0],
                [0.2, 0.0, 2.0],
                [0.1, 0.0, 2.0],
            ]
        ),
        "scales": torch.full((4, 3), 0.05),
        "colors": torch.full((4, 3), 0.4),
        "opacities": torch.full((4, 1), 0.03),
        "quaternions": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 4
        ),
        "layer_role": torch.tensor(
            [
                LAYER_DYNAMIC_LEAF,
                LAYER_DYNAMIC_LEAF,
                LAYER_DYNAMIC_LEAF,
                0,
            ],
            dtype=torch.int8,
        ),
        "initialization_source": torch.tensor(
            [4, 4, 1, 0], dtype=torch.int8
        ),
        "occupancy_probability": torch.tensor([0.1, 0.1, 0.1, 0.9]),
        "support_view_count": torch.ones(4, dtype=torch.int16),
        "unknown_view_count": torch.zeros(4, dtype=torch.int16),
        "ray_depth_nll": torch.tensor([99.0, 99.0, 99.0, 0.0]),
        "free_space_violation_count": torch.zeros(
            4, dtype=torch.int16
        ),
        "replacement_group": torch.tensor([-1, 0, -1, 0]),
    }
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)

    ceiling = _dynamic_opacity_ceiling(foliage, 0.40)

    floor = _dynamic_opacity_floor(foliage)
    # Same depth posterior, but a physically local canonical owner grants
    # more replacement authority than an unconserved ownerless birth.
    assert floor[0] <= ceiling[0] < ceiling[1] < 0.10
    # Non-DAV2/non-dense legacy dynamics retain the normal global ceiling.
    assert ceiling[2].item() == pytest.approx(0.40)


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
    assert audit["ray_hit"] > 0
    assert audit["hit_alpha_mean"] < 1e-4
    assert audit["ray_contract"] == (
        "exact_owner_complete_lineage_analytic_hit_and_"
        "prehit_transmittance"
    )
    assert model.opacity_logits.grad is not None
    # Gradient descent must increase exact-owner optical mass even from an
    # almost transparent initialization.
    assert model.opacity_logits.grad[0, 0] < 0


def test_dynamic_observation_ray_likelihood_rewards_owner_hit_mass():
    def model(opacity: float, z: float = 2.0):
        payload = {
            "version": "independent_sfm_semantic_canopy_volume_v1",
            "centers": torch.tensor([[0.0, 0.0, z]]),
            "scales": torch.tensor([[0.1, 0.1, 0.1]]),
            "colors": torch.tensor([[0.2, 0.4, 0.6]]),
            "opacities": torch.tensor([[opacity]]),
            "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "layer_role": torch.tensor(
                [LAYER_DYNAMIC_LEAF], dtype=torch.int8
            ),
            "support_camera_ids": torch.tensor(
                [[7]], dtype=torch.int32
            ),
            "observation_camera_ids": torch.tensor(
                [[7]], dtype=torch.int32
            ),
            "observation_uv": torch.tensor([[[0.5, 0.5]]]),
            "observation_depth": torch.tensor([[2.0]]),
        }
        result = VolumetricFoliageModel(1, device="cpu")
        result.initialize_from_volume_state(payload)
        return result

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
    low = model(0.02)
    high = model(0.80)
    before = model(0.80, z=1.0)

    low_loss, _, low_audit = _dynamic_observation_factor(
        low, camera, torch.zeros(low.dynamic_rank)
    )
    high_loss, _, high_audit = _dynamic_observation_factor(
        high, camera, torch.zeros(high.dynamic_rank)
    )
    _, _, before_audit = _dynamic_observation_factor(
        before, camera, torch.zeros(before.dynamic_rank)
    )

    assert high_audit["hit_alpha_mean"] > low_audit["hit_alpha_mean"]
    assert high_loss < low_loss
    assert before_audit["prehit_free"] > high_audit["prehit_free"]


def test_dynamic_observation_factor_aggregates_every_split_descendant():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.0, 0.0, 2.0]]),
        "scales": torch.tensor([[0.1, 0.1, 0.1]]),
        "colors": torch.tensor([[0.2, 0.4, 0.6]]),
        "opacities": torch.tensor([[0.10]]),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "layer_role": torch.tensor(
            [LAYER_DYNAMIC_LEAF], dtype=torch.int8
        ),
        "track_id": torch.tensor([41], dtype=torch.int64),
        "support_camera_ids": torch.tensor([[7]], dtype=torch.int32),
        "observation_camera_ids": torch.tensor(
            [[7]], dtype=torch.int32
        ),
        "observation_uv": torch.tensor([[[0.5, 0.5]]]),
        "observation_depth": torch.tensor([[2.0]]),
    }
    model = VolumetricFoliageModel(1, device="cpu")
    model.initialize_from_volume_state(payload)
    event = model.split(torch.tensor([0]))
    assert event["children"] == 2
    assert model.track_id.tolist() == [41, 41]
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

    _, _, audit = _dynamic_observation_factor(
        model,
        camera,
        torch.zeros(model.dynamic_rank),
        maximum_observations=1,
    )

    assert audit["available"] == 1
    assert audit["matched"] == 1
    assert audit["matched_descendants"] == 2


def test_dynamic_observation_factor_rotates_through_dense_view_births():
    count = 5
    centers = torch.tensor(
        [[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [0.2, 0.0, 2.0],
         [0.3, 0.0, 2.0], [0.4, 0.0, 2.0]]
    )
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": centers,
        "scales": torch.full((count, 3), 0.1),
        "colors": torch.full((count, 3), 0.4),
        "opacities": torch.full((count, 1), 0.015),
        "quaternions": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * count
        ),
        "layer_role": torch.full(
            (count,), LAYER_DYNAMIC_LEAF, dtype=torch.int8
        ),
        "support_camera_ids": torch.full(
            (count, 1), 7, dtype=torch.int32
        ),
        "observation_camera_ids": torch.full(
            (count, 1), 7, dtype=torch.int32
        ),
        "observation_uv": torch.stack(
            [
                centers[:, 0] * 0.5 + 0.5,
                torch.full((count,), 0.5),
            ],
            dim=1,
        )[:, None],
        "observation_depth": torch.full((count, 1), 2.0),
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
    covered = torch.zeros(count, dtype=torch.bool)
    for update in range(3):
        _, observed, audit = _dynamic_observation_factor(
            model,
            camera,
            torch.zeros(model.dynamic_rank),
            maximum_observations=2,
            sample_update=update,
        )
        assert audit["available"] == count
        covered |= observed
    assert covered.all()


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


def test_chart_native_factor_downsamples_prediction_not_measurement():
    evidence = OutdoorGeometryEvidence.__new__(OutdoorGeometryEvidence)
    evidence.frame_by_stem = {"frame00001": 0}
    evidence.chart = {
        "active": np.asarray([True]),
        "depth": np.full((1, 2, 2), 4.0, dtype=np.float32),
        "confidence": np.ones((1, 2, 2), dtype=np.float32),
        "reference_mask": np.ones((1, 2, 2), dtype=bool),
    }
    evidence.consumed = {"chart_native_factor": 0}
    depth = torch.full((4, 4), 4.0, requires_grad=True)
    package = SimpleNamespace(
        surface_depth=depth[None],
        surface_alpha=torch.ones(1, 4, 4),
    )
    matched, audit = evidence.chart_native_factor(
        "frame00001.png", package, torch.ones(4, 4)
    )
    assert audit["pixels"] == 4
    assert matched < 1e-6
    mismatched_package = SimpleNamespace(
        surface_depth=(depth * 2.0)[None],
        surface_alpha=torch.ones(1, 4, 4),
    )
    mismatched, _ = evidence.chart_native_factor(
        "frame00001.png", mismatched_package, torch.ones(4, 4)
    )
    assert mismatched > 0
    evidence.chart["precision"] = np.full(
        (1, 2, 2), 0.1, dtype=np.float32
    )
    low_trust, low_trust_audit = evidence.chart_native_factor(
        "frame00001.png", mismatched_package, torch.ones(4, 4)
    )
    assert low_trust < 0.2 * mismatched
    assert low_trust_audit["absolute_precision"] < 0.11
    mismatched.backward()
    assert depth.grad.abs().sum() > 0


def test_dense_mast3r_pointmap_factor_uses_fixed_camera_world_depth():
    evidence = OutdoorGeometryEvidence.__new__(OutdoorGeometryEvidence)
    evidence.pointmap_records = {"frame00001": {"path": "unused"}}
    evidence._pointmap_cache = {
        "frame00001": (
            np.asarray(
                [
                    [-0.5, -0.5, 4.0],
                    [0.5, -0.5, 4.0],
                    [-0.5, 0.5, 4.0],
                    [0.5, 0.5, 4.0],
                ],
                dtype=np.float32,
            ),
            np.full(4, 2.0, dtype=np.float32),
        )
    }
    evidence.consumed = {"mast3r_pointmap_native_factor": 0}
    evidence._pointmap_posterior_stats = {
        "factor_calls": 0,
        "posterior_factor_calls": 0,
        "single_sequence_fallback_factor_calls": 0,
        "pixels": 0,
        "cross_sequence_supported_pixels": 0,
        "single_sequence_low_precision_pixels": 0,
        "missing_posterior_fallback_pixels": 0,
        "precision_sum": 0.0,
    }
    view = SimpleNamespace(
        image_name="frame00001.png",
        image_width=4,
        image_height=4,
        R=np.eye(3, dtype=np.float32),
        T=np.zeros(3, dtype=np.float32),
    )
    depth = torch.full((4, 4), 4.0, requires_grad=True)
    package = SimpleNamespace(
        surface_depth=depth[None],
        surface_alpha=torch.ones(1, 4, 4),
    )
    matched, audit = evidence.mast3r_pointmap_native_factor(
        view, package, torch.ones(4, 4)
    )
    assert audit["pixels"] == 4
    assert audit["absolute_precision"] == pytest.approx(0.03)
    assert audit["cross_sequence_supported_pixels"] == 0
    assert audit["single_sequence_low_precision_pixels"] == 4
    assert audit["missing_posterior_fallback_pixels"] == 4
    assert not audit["posterior_available"]
    assert matched < 1e-6
    mismatched, _ = evidence.mast3r_pointmap_native_factor(
        view,
        SimpleNamespace(
            surface_depth=(2.0 * depth)[None],
            surface_alpha=torch.ones(1, 4, 4),
        ),
        torch.ones(4, 4),
    )
    assert mismatched > 0
    mismatched.backward()
    assert depth.grad.abs().sum() > 0
    assert evidence.consumed["mast3r_pointmap_native_factor"] == 2
    assert (
        evidence._pointmap_posterior_stats[
            "single_sequence_fallback_factor_calls"
        ]
        == 2
    )
    assert (
        evidence._pointmap_posterior_stats[
            "cross_sequence_supported_pixels"
        ]
        == 0
    )


def test_production_pointmap_posterior_preflight_fails_closed():
    incomplete = SimpleNamespace(
        audit=lambda: {
            "pointmap_cross_sequence_posterior": {
                "pointmap_camera_count": 56,
                "available_camera_count": 0,
                "missing_camera_count": 56,
            }
        }
    )
    with pytest.raises(RuntimeError, match="56/56 cameras are missing"):
        _assert_pointmap_posterior_contract(
            incomplete, allow_missing=False
        )
    ablation = _assert_pointmap_posterior_contract(
        incomplete, allow_missing=True
    )
    assert not ablation["production_complete"]
    assert ablation["missing_posterior_ablation"]

    complete = SimpleNamespace(
        audit=lambda: {
            "pointmap_cross_sequence_posterior": {
                "pointmap_camera_count": 56,
                "available_camera_count": 56,
                "missing_camera_count": 0,
            }
        }
    )
    production = _assert_pointmap_posterior_contract(
        complete, allow_missing=False
    )
    assert production["production_complete"]
    assert not production["missing_posterior_ablation"]


def test_dense_pointmap_factor_respects_cross_sequence_precision():
    evidence = OutdoorGeometryEvidence.__new__(OutdoorGeometryEvidence)
    evidence.pointmap_records = {
        "frame00001": {
            "path": "unused",
            "cross_sequence_posterior": {"path": "unused"},
        }
    }
    evidence._pointmap_cache = {
        "frame00001": (
            np.asarray(
                [
                    [-0.5, -0.5, 4.0],
                    [0.5, -0.5, 4.0],
                    [-0.5, 0.5, 4.0],
                    [0.5, 0.5, 4.0],
                ],
                dtype=np.float32,
            ),
            np.full(4, 2.0, dtype=np.float32),
        )
    }
    low_precision = np.full((2, 2), 0.05, dtype=np.float32)
    cross_sequence_supported = np.zeros((2, 2), dtype=bool)
    evidence._pointmap_posterior_cache = {
        "frame00001": (low_precision, cross_sequence_supported)
    }
    evidence._pointmap_posterior_stats = {
        "factor_calls": 0,
        "posterior_factor_calls": 0,
        "single_sequence_fallback_factor_calls": 0,
        "pixels": 0,
        "cross_sequence_supported_pixels": 0,
        "single_sequence_low_precision_pixels": 0,
        "missing_posterior_fallback_pixels": 0,
        "precision_sum": 0.0,
    }
    evidence.consumed = {"mast3r_pointmap_native_factor": 0}
    view = SimpleNamespace(
        image_name="frame00001.png",
        image_width=4,
        image_height=4,
        R=np.eye(3, dtype=np.float32),
        T=np.zeros(3, dtype=np.float32),
    )
    package = SimpleNamespace(
        surface_depth=torch.full((1, 4, 4), 8.0),
        surface_alpha=torch.ones(1, 4, 4),
    )
    low_trust, audit = evidence.mast3r_pointmap_native_factor(
        view, package, torch.ones(4, 4)
    )

    low_precision.fill(1.0)
    cross_sequence_supported.fill(True)
    full_trust, _ = evidence.mast3r_pointmap_native_factor(
        view, package, torch.ones(4, 4)
    )

    assert audit["absolute_precision"] < 0.06
    assert audit["cross_sequence_supported_pixels"] == 0
    assert low_trust < 0.1 * full_trust
    assert evidence._pointmap_posterior_stats["factor_calls"] == 2
    assert evidence._pointmap_posterior_stats["posterior_factor_calls"] == 2
    assert (
        evidence._pointmap_posterior_stats[
            "cross_sequence_supported_pixels"
        ]
        == 4
    )


def test_foliage_unknown_ray_is_not_misclassified_as_free_space():
    def evidence(kind):
        return FoliageRayEvidence(
            {
                "camera_ids": torch.tensor([7], dtype=torch.int32),
                "pixels": torch.tensor([[0.5, 0.5]]),
                "source_image_sizes": torch.tensor(
                    [[1, 1]], dtype=torch.int32
                ),
                "free_end_depth": torch.tensor([3.0]),
                "hit_start_depth": torch.tensor([3.0]),
                "hit_end_depth": torch.tensor([4.0]),
                "observation_type": torch.tensor(
                    [kind], dtype=torch.int8
                ),
                "confidence": torch.ones(1),
            }
        )

    view = SimpleNamespace(
        colmap_id=7,
        image_width=1,
        image_height=1,
    )
    package = SimpleNamespace(
        # This mass is in front of the confirmed-free boundary and therefore
        # would be penalized if an unknown observation leaked into free-space.
        volume_depth=torch.tensor([[[1.0]]]),
        volume_alpha=torch.tensor([[[0.5]]]),
    )
    unknown_loss, unknown_audit = evidence(0).factor(view, package)
    free_loss, free_audit = evidence(-1).factor(view, package)

    evidence_audit = evidence(0).audit()
    assert {
        key: evidence_audit[key]
        for key in (
            "ray_count",
            "hit_count",
            "confirmed_free_count",
            "unknown_count",
            "camera_count",
            "explicit_source_resolution",
            "unknown_is_excluded_from_free_loss",
        )
    } == {
        "ray_count": 1,
        "hit_count": 0,
        "confirmed_free_count": 0,
        "unknown_count": 1,
        "camera_count": 1,
        "explicit_source_resolution": True,
        "unknown_is_excluded_from_free_loss": True,
    }
    assert unknown_audit["rays"] == 1
    assert unknown_loss == 0
    assert free_audit["free"] > 0
    assert free_loss > 0


def test_foliage_ray_pixels_are_scaled_from_source_to_render_resolution():
    evidence = FoliageRayEvidence(
        {
            "camera_ids": torch.tensor([7], dtype=torch.int32),
            # Centre of source pixel (2, 2) in a 4x4 source raster maps to
            # render pixel (1, 1) in the 2x2 training raster.
            "pixels": torch.tensor([[2.0, 2.0]]),
            "source_image_sizes": torch.tensor(
                [[4, 4]], dtype=torch.int32
            ),
            "free_end_depth": torch.tensor([3.0]),
            "hit_start_depth": torch.tensor([3.0]),
            "hit_end_depth": torch.tensor([4.0]),
            "observation_type": torch.tensor([-1], dtype=torch.int8),
            "confidence": torch.ones(1),
        }
    )
    view = SimpleNamespace(colmap_id=7, image_width=2, image_height=2)
    package = SimpleNamespace(
        volume_depth=torch.tensor([[[0.0, 0.0], [0.0, 1.0]]]),
        volume_alpha=torch.tensor([[[0.0, 0.0], [0.0, 0.5]]]),
    )

    loss, audit = evidence.factor(view, package)

    assert audit["rays"] == 1
    assert audit["free"] > 0
    assert loss > 0


def test_foliage_ray_table_without_source_resolution_fails_closed():
    with np.testing.assert_raises_regex(
        RuntimeError, "source-raster resolution"
    ):
        FoliageRayEvidence(
            {
                "camera_ids": torch.tensor([7], dtype=torch.int32),
                "pixels": torch.tensor([[10.0, 20.0]]),
                "free_end_depth": torch.tensor([3.0]),
                "hit_start_depth": torch.tensor([3.0]),
                "hit_end_depth": torch.tensor([4.0]),
                "observation_type": torch.tensor([1], dtype=torch.int8),
                "confidence": torch.ones(1),
            }
        )


def _interval_evidence(kind, *, confidence=1.0):
    return FoliageRayEvidence(
        {
            "offsets": torch.tensor([0, 1], dtype=torch.int64),
            "camera_ids": torch.tensor([7], dtype=torch.int32),
            "pixels": torch.tensor([[50.0, 50.0]]),
            "source_image_sizes": torch.tensor(
                [[100, 100]], dtype=torch.int32
            ),
            "free_end_depth": torch.tensor([2.5]),
            "hit_start_depth": torch.tensor([2.8]),
            "hit_end_depth": torch.tensor([3.2]),
            "observation_type": torch.tensor(
                [kind], dtype=torch.int8
            ),
            "confidence": torch.full((1,), float(confidence)),
        }
    )


def _ownerless_interval_evidence(*, confidence=1.0):
    return FoliageRayEvidence(
        {
            "offsets": torch.tensor([0, 0], dtype=torch.int64),
            "camera_ids": torch.tensor([7], dtype=torch.int32),
            "pixels": torch.tensor([[50.0, 50.0]]),
            "source_image_sizes": torch.tensor(
                [[100, 100]], dtype=torch.int32
            ),
            "free_end_depth": torch.tensor([2.5]),
            "hit_start_depth": torch.tensor([2.8]),
            "hit_end_depth": torch.tensor([3.2]),
            "observation_type": torch.ones(1, dtype=torch.int8),
            "confidence": torch.full((1,), float(confidence)),
        }
    )


def _interval_foliage(z):
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.0, 0.0, float(z)]]),
        "scales": torch.tensor([[0.2, 0.2, 0.2]]),
        "colors": torch.tensor([[0.2, 0.4, 0.6]]),
        "opacities": torch.tensor([[0.5]]),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "ray_evidence": {
            "offsets": torch.tensor([0, 1], dtype=torch.int64)
        },
    }
    model = VolumetricFoliageModel(1, device="cpu")
    model.initialize_from_volume_state(payload)
    return model


def _interval_dynamic_foliage(z, *, support_camera_id=7):
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.0, 0.0, float(z)]]),
        "scales": torch.tensor([[0.2, 0.2, 0.2]]),
        "colors": torch.tensor([[0.2, 0.4, 0.6]]),
        "opacities": torch.tensor([[0.5]]),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "layer_role": torch.tensor([2], dtype=torch.int8),
        "support_camera_ids": torch.tensor(
            [[support_camera_id]], dtype=torch.int32
        ),
        # Observation-space births deliberately have no canonical ray id.
        "ray_evidence": {"offsets": torch.tensor([0], dtype=torch.int64)},
    }
    model = VolumetricFoliageModel(1, device="cpu")
    model.initialize_from_volume_state(payload)
    return model


def _interval_static_detail_foliage(z, *, support_camera_id=7):
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.0, 0.0, float(z)]]),
        "scales": torch.tensor([[0.2, 0.2, 0.2]]),
        "colors": torch.tensor([[0.2, 0.4, 0.6]]),
        "opacities": torch.tensor([[0.1]]),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "layer_role": torch.tensor([0], dtype=torch.int8),
        "static_detail": torch.tensor([True]),
        "support_camera_ids": torch.tensor(
            [[support_camera_id]], dtype=torch.int32
        ),
        "support_sequence_count": torch.tensor([2], dtype=torch.int16),
        # The observation-space ray is ownerless; exact ownership comes from
        # the fused static-detail support table.
        "ray_evidence": {"offsets": torch.tensor([0], dtype=torch.int64)},
    }
    model = VolumetricFoliageModel(1, device="cpu")
    model.initialize_from_volume_state(payload)
    return model


def _solid_refresh_camera(camera_id, color):
    return SimpleNamespace(
        colmap_id=camera_id,
        world_view_transform=torch.eye(4),
        focal_x=2.0,
        focal_y=2.0,
        cx=1.0,
        cy=1.0,
        image_width=4,
        image_height=4,
        original_image=torch.as_tensor(color, dtype=torch.float32)[
            :, None, None
        ].expand(3, 4, 4),
    )


def test_split_child_color_refresh_uses_owner_and_static_multiview_rgb():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.0, 0.0, 2.0]] * 2),
        "scales": torch.full((2, 3), 0.05),
        "colors": torch.full((2, 3), 0.25),
        "opacities": torch.full((2, 1), 0.2),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
        "layer_role": torch.tensor([LAYER_DYNAMIC_LEAF, 0], dtype=torch.int8),
        "observation_camera_ids": torch.tensor(
            [[7, -1], [7, 8]], dtype=torch.int32
        ),
        "observation_uv": torch.full((2, 2, 2), 0.5),
        "observation_depth": torch.tensor([[2.0, 0.0], [2.0, 2.0]]),
    }
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)
    cameras = {
        7: _solid_refresh_camera(7, [1.0, 0.0, 0.0]),
        8: _solid_refresh_camera(8, [0.0, 0.0, 1.0]),
    }

    audit = _refresh_split_child_owner_colors(
        foliage,
        child_start=0,
        owner_camera_ids=torch.tensor([7, -1]),
        view_by_camera_id=cameras,
    )

    dc_rgb = foliage.features[:, 0] * 0.28209479177387814 + 0.5
    torch.testing.assert_close(dc_rgb[0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(dc_rgb[1], torch.tensor([0.5, 0.0, 0.5]))
    assert audit["dynamic_refreshed_children"] == 1
    assert audit["canonical_refreshed_children"] == 1


def test_static_child_color_refresh_rejects_depth_inconsistent_view():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.0, 0.0, 2.0]]),
        "scales": torch.full((1, 3), 0.05),
        "colors": torch.full((1, 3), 0.25),
        "opacities": torch.full((1, 1), 0.2),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "layer_role": torch.tensor([0], dtype=torch.int8),
        "observation_camera_ids": torch.tensor([[7, 8]], dtype=torch.int32),
        "observation_uv": torch.full((1, 2, 2), 0.5),
        "observation_depth": torch.tensor([[2.0, 4.0]]),
    }
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)

    audit = _refresh_split_child_owner_colors(
        foliage,
        child_start=0,
        owner_camera_ids=torch.tensor([-1]),
        view_by_camera_id={
            7: _solid_refresh_camera(7, [0.0, 1.0, 0.0]),
            8: _solid_refresh_camera(8, [1.0, 0.0, 0.0]),
        },
    )

    dc_rgb = foliage.features[0, 0] * 0.28209479177387814 + 0.5
    # One surviving view is not enough to create a saturated novel-view
    # colour. It is bounded around the inherited 0.25 DC and higher-order SH
    # is removed.
    torch.testing.assert_close(dc_rgb, torch.tensor([0.175, 0.325, 0.175]))
    assert torch.count_nonzero(foliage.features[0, 1:]) == 0
    assert audit["canonical_refreshed_children"] == 1
    assert audit["canonical_low_support_dc_only_children"] == 1


def test_dynamic_child_color_refresh_rejects_owner_depth_mismatch():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.0, 0.0, 4.0]]),
        "scales": torch.full((1, 3), 0.05),
        "colors": torch.full((1, 3), 0.25),
        "opacities": torch.full((1, 1), 0.2),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "layer_role": torch.tensor(
            [LAYER_DYNAMIC_LEAF], dtype=torch.int8
        ),
        "observation_camera_ids": torch.tensor(
            [[7]], dtype=torch.int32
        ),
        "observation_uv": torch.full((1, 1, 2), 0.5),
        "observation_depth": torch.tensor([[2.0]]),
    }
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)

    audit = _refresh_split_child_owner_colors(
        foliage,
        child_start=0,
        owner_camera_ids=torch.tensor([7]),
        view_by_camera_id={
            7: _solid_refresh_camera(7, [1.0, 1.0, 1.0])
        },
    )

    dc_rgb = foliage.features[0, 0] * 0.28209479177387814 + 0.5
    torch.testing.assert_close(dc_rgb, torch.full((3,), 0.25))
    assert audit["dynamic_depth_candidate_children"] == 1
    assert audit["dynamic_refreshed_children"] == 0


def test_visual_hull_static_child_refresh_uses_cross_sequence_support():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.0, 0.0, 2.0]]),
        "scales": torch.full((1, 3), 0.05),
        "colors": torch.full((1, 3), 0.25),
        "opacities": torch.full((1, 1), 0.2),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "layer_role": torch.tensor([0], dtype=torch.int8),
        "support_camera_ids": torch.tensor([[7, 8]], dtype=torch.int32),
        # Split-time evidence accounting may conservatively reduce this
        # scalar even though the immutable support-camera identity set is
        # retained. The fallback must use the latter.
        "support_sequence_count": torch.tensor([1], dtype=torch.int16),
    }
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)

    audit = _refresh_split_child_owner_colors(
        foliage,
        child_start=0,
        owner_camera_ids=torch.tensor([-1]),
        view_by_camera_id={
            7: _solid_refresh_camera(7, [0.2, 0.4, 0.6]),
            8: _solid_refresh_camera(8, [0.6, 0.4, 0.2]),
        },
    )

    dc_rgb = foliage.features[0, 0] * 0.28209479177387814 + 0.5
    torch.testing.assert_close(dc_rgb, torch.tensor([0.4, 0.4, 0.4]))
    assert audit["canonical_support_fallback_children"] == 1
    assert audit["canonical_refreshed_children"] == 1


def test_static_child_support_fallback_runs_after_all_depth_samples_fail():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.0, 0.0, 2.0]]),
        "scales": torch.full((1, 3), 0.05),
        "colors": torch.full((1, 3), 0.25),
        "opacities": torch.full((1, 1), 0.2),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "layer_role": torch.tensor([0], dtype=torch.int8),
        "observation_camera_ids": torch.tensor([[7, 8]], dtype=torch.int32),
        "observation_uv": torch.full((1, 2, 2), 0.5),
        # Both explicit posteriors reject the child at z=2.
        "observation_depth": torch.tensor([[4.0, 4.0]]),
        "support_camera_ids": torch.tensor([[7, 8]], dtype=torch.int32),
        "support_sequence_count": torch.tensor([2], dtype=torch.int16),
    }
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)

    audit = _refresh_split_child_owner_colors(
        foliage,
        child_start=0,
        owner_camera_ids=torch.tensor([-1]),
        view_by_camera_id={
            7: _solid_refresh_camera(7, [0.2, 0.4, 0.6]),
            8: _solid_refresh_camera(8, [0.6, 0.4, 0.2]),
        },
    )

    dc_rgb = foliage.features[0, 0] * 0.28209479177387814 + 0.5
    torch.testing.assert_close(dc_rgb, torch.tensor([0.4, 0.4, 0.4]))
    assert audit["canonical_depth_candidate_children"] == 1
    assert audit["canonical_depth_refreshed_children"] == 0
    assert audit["canonical_support_fallback_children"] == 1


def test_static_child_color_refresh_uses_current_metric_depth_posterior(
    tmp_path,
):
    np.save(tmp_path / "rho_mean_frame000000.npy", np.full((4, 4), 0.5))
    np.save(
        tmp_path / "rho_variance_frame000000.npy",
        np.full((4, 4), 1.0e-4),
    )
    np.save(
        tmp_path / "support_view_count_frame000000.npy",
        np.full((4, 4), 2, dtype=np.uint8),
    )
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.0, 0.0, 2.0]]),
        "scales": torch.full((1, 3), 0.05),
        "colors": torch.full((1, 3), 0.25),
        "opacities": torch.full((1, 1), 0.2),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "layer_role": torch.tensor([0], dtype=torch.int8),
        "observation_camera_ids": torch.tensor([[7]], dtype=torch.int32),
        "observation_uv": torch.full((1, 1, 2), 0.5),
        # The inherited parent-lineage observation no longer matches this
        # spatial child, while dense metric evidence at its current pixel does.
        "observation_depth": torch.tensor([[4.0]]),
    }
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)
    camera = _solid_refresh_camera(7, [0.0, 1.0, 0.0])
    camera.image_name = "metric"

    audit = _refresh_split_child_owner_colors(
        foliage,
        child_start=0,
        owner_camera_ids=torch.tensor([-1]),
        view_by_camera_id={7: camera},
        geometry_evidence=SimpleNamespace(
            inverse_root=tmp_path,
            frame_by_stem={"metric": 0},
        ),
    )

    dc_rgb = foliage.features[0, 0] * 0.28209479177387814 + 0.5
    torch.testing.assert_close(dc_rgb, torch.tensor([0.175, 0.325, 0.175]))
    assert audit["canonical_depth_refreshed_children"] == 1
    assert audit["metric_depth_candidate_children"] == 1
    assert audit["metric_depth_refreshed_children"] == 1


def test_static_child_color_refresh_uses_current_foliage_ray_posterior():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.0, 0.0, 2.0]]),
        "scales": torch.full((1, 3), 0.05),
        "colors": torch.full((1, 3), 0.25),
        "opacities": torch.full((1, 1), 0.2),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "layer_role": torch.tensor([0], dtype=torch.int8),
        "observation_camera_ids": torch.tensor([[7]], dtype=torch.int32),
        # Both inherited fields are stale after subdivision.  The immutable
        # current-pixel ray posterior below is the valid attachment.
        "observation_uv": torch.tensor([[[0.9, 0.9]]]),
        "observation_depth": torch.tensor([[4.0]]),
    }
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)
    rays = FoliageRayEvidence(
        {
            "camera_ids": torch.tensor([7], dtype=torch.int32),
            "pixels": torch.tensor([[1.0, 1.0]]),
            "source_image_sizes": torch.tensor(
                [[4, 4]], dtype=torch.int32
            ),
            "free_end_depth": torch.tensor([1.9]),
            "hit_start_depth": torch.tensor([1.9]),
            "hit_end_depth": torch.tensor([2.1]),
            "observation_type": torch.ones(1, dtype=torch.int8),
            "confidence": torch.ones(1),
            "depth_coordinate": "camera_z",
        }
    )

    audit = _refresh_split_child_owner_colors(
        foliage,
        child_start=0,
        owner_camera_ids=torch.tensor([-1]),
        view_by_camera_id={
            7: _solid_refresh_camera(7, [0.0, 1.0, 0.0])
        },
        foliage_ray_evidence=rays,
    )

    dc_rgb = foliage.features[0, 0] * 0.28209479177387814 + 0.5
    torch.testing.assert_close(dc_rgb, torch.tensor([0.025, 0.475, 0.025]))
    assert audit["ray_depth_candidate_children"] == 1
    assert audit["ray_depth_refreshed_children"] == 1
    assert audit["canonical_refreshed_children"] == 1
    assert audit["canonical_single_ray_dc_children"] == 1


def _interval_camera():
    return SimpleNamespace(
        colmap_id=7,
        R=np.eye(3, dtype=np.float32),
        camera_center=torch.zeros(3),
        focal_x=100.0,
        focal_y=100.0,
        cx=50.0,
        cy=50.0,
        image_width=100,
        image_height=100,
    )


def test_candidate_interval_factor_rewards_mass_inside_measured_hit():
    evidence = _interval_evidence(1)
    correct = _interval_foliage(3.05)
    wrong = _interval_foliage(1.0)

    correct_loss, audit = evidence.interval_factor(
        _interval_camera(), correct
    )
    wrong_loss, _ = evidence.interval_factor(
        _interval_camera(), wrong
    )

    assert audit["candidate_evaluations"] == 1
    assert audit["hit_rays"] == 1
    assert correct_loss < wrong_loss
    correct_loss.backward()
    assert correct.opacity_logits.grad is not None
    assert correct.opacity_logits.grad.abs().sum() > 0


def test_candidate_interval_factor_respects_static_stage_mask():
    evidence = _interval_evidence(1)
    foliage = _interval_foliage(3.05)

    loss, audit = evidence.interval_factor(
        _interval_camera(),
        foliage,
        canonical_candidate_mask=torch.zeros(1, dtype=torch.bool),
    )

    assert audit["candidate_evaluations"] == 0
    assert audit["rays_with_candidates"] == 0
    assert not loss.requires_grad


def test_candidate_interval_separates_global_free_from_owned_positive_hit():
    evidence = _interval_evidence(1)
    globally_visible = _interval_foliage(3.05)
    global_only_loss, global_only_audit = evidence.interval_factor(
        _interval_camera(),
        globally_visible,
        canonical_candidate_mask=torch.ones(1, dtype=torch.bool),
        canonical_hit_candidate_mask=torch.zeros(1, dtype=torch.bool),
    )
    global_only_loss.backward()

    evidence = _interval_evidence(1)
    positive_owner = _interval_foliage(3.05)
    owned_loss, owned_audit = evidence.interval_factor(
        _interval_camera(),
        positive_owner,
        canonical_candidate_mask=torch.ones(1, dtype=torch.bool),
        canonical_hit_candidate_mask=torch.ones(1, dtype=torch.bool),
    )
    owned_loss.backward()

    assert global_only_audit["canonical_candidate_evaluations"] == 1
    assert global_only_audit["canonical_hit_candidate_evaluations"] == 0
    assert owned_audit["canonical_hit_candidate_evaluations"] == 1
    # A non-owner receives only the pre-hit free-space contradiction and is
    # therefore asked to reduce optical mass. The canonical owner receives
    # the measured hit and is asked to add mass at the same location.
    assert float(globally_visible.opacity_logits.grad) > 0
    assert float(positive_owner.opacity_logits.grad) < 0


def test_candidate_interval_confidence_separates_geometry_and_existence():
    strong_evidence = _interval_evidence(1, confidence=1.0)
    weak_evidence = _interval_evidence(1, confidence=0.1)
    strong_foliage = _interval_foliage(3.05)
    weak_foliage = _interval_foliage(3.05)

    strong_loss, strong_audit = strong_evidence.interval_factor(
        _interval_camera(), strong_foliage
    )
    weak_loss, weak_audit = weak_evidence.interval_factor(
        _interval_camera(), weak_foliage
    )

    strong_loss.backward()
    weak_loss.backward()
    # The hit pixel remains one unit of optical-existence evidence.  Only
    # metric placement/free-space gradients pay the weak depth confidence.
    assert weak_loss > 0.9 * strong_loss
    torch.testing.assert_close(
        weak_foliage.opacity_logits.grad,
        strong_foliage.opacity_logits.grad,
        rtol=0.03,
        atol=1e-5,
    )
    torch.testing.assert_close(
        weak_foliage.xyz.grad,
        0.1 * strong_foliage.xyz.grad,
        rtol=0.03,
        atol=1e-5,
    )
    torch.testing.assert_close(
        weak_foliage.log_scales.grad,
        0.1 * strong_foliage.log_scales.grad,
        rtol=0.03,
        atol=1e-5,
    )
    assert weak_audit["hit_geometry"] < strong_audit["hit_geometry"]
    assert weak_audit["hit_optical_existence"] > 0
    assert strong_audit["hit_optical_existence"] == 0
    assert strong_audit["mean_confidence"] == pytest.approx(1.0)
    assert weak_audit["mean_confidence"] == pytest.approx(0.1)
    assert weak_audit["confidence_normalization"] == (
        "depth_geometry_weighted_sum_over_ray_count__missing_confidence_"
        "mass_routes_to_opacity_only_existence"
    )


def test_candidate_interval_factor_audits_allowed_behind_mass():
    evidence = _interval_evidence(1)
    foliage = _interval_foliage(4.0)

    _, audit = evidence.interval_factor(_interval_camera(), foliage)

    assert audit["behind_mass"] > 0.1
    assert audit["hit"] > 1.0


def test_candidate_interval_factor_converts_camera_z_on_off_axis_ray():
    evidence = FoliageRayEvidence(
        {
            "offsets": torch.tensor([0, 1], dtype=torch.int64),
            "camera_ids": torch.tensor([7], dtype=torch.int32),
            # fx=50 and cx=50 below: x=80 is direction (0.6, 0, 1).
            "pixels": torch.tensor([[80.0, 50.0]]),
            "source_image_sizes": torch.tensor(
                [[100, 100]], dtype=torch.int32
            ),
            "free_end_depth": torch.tensor([2.5]),
            "hit_start_depth": torch.tensor([2.8]),
            "hit_end_depth": torch.tensor([3.2]),
            "observation_type": torch.ones(1, dtype=torch.int8),
            "confidence": torch.ones(1),
        }
    )
    foliage = _interval_foliage(3.0)
    with torch.no_grad():
        foliage.xyz[0] = torch.tensor([1.8, 0.0, 3.0])
    camera = _interval_camera()
    camera.focal_x = 50.0
    camera.focal_y = 50.0

    loss, audit = evidence.interval_factor(camera, foliage)

    # The Gaussian has Euclidean range ~=3.50 but camera z=3.0.  It is a
    # measured hit only after the evidence interval is converted to the unit
    # ray coordinate used by the analytic integral.
    assert audit["depth_coordinate"] == (
        "camera_z_converted_to_unit_ray_distance"
    )
    assert audit["candidate_evaluations"] == 1
    assert audit["hit"] < 2.0
    loss.backward()
    assert foliage.xyz.grad is not None


def test_candidate_interval_tiny_hit_mass_retains_opacity_gradient():
    evidence = _interval_evidence(1)
    foliage = _interval_foliage(3.0)
    with torch.no_grad():
        foliage.opacity_logits.fill_(-20.0)

    loss, audit = evidence.interval_factor(
        _interval_camera(), foliage
    )
    loss.backward()

    assert audit["hit"] > 0
    assert foliage.opacity_logits.grad is not None
    assert foliage.opacity_logits.grad.abs().sum() > 0


def test_ownerless_ray_uses_preverified_canonical_or_exact_dynamic():
    evidence = FoliageRayEvidence(
        {
            # Primitive zero owns the first row.  The second row is a dense
            # observation-space posterior with no seed identity.
            "offsets": torch.tensor([0, 1], dtype=torch.int64),
            "camera_ids": torch.tensor([7, 7], dtype=torch.int32),
            "pixels": torch.full((2, 2), 50.0),
            "source_image_sizes": torch.full(
                (2, 2), 100, dtype=torch.int32
            ),
            "free_end_depth": torch.full((2,), 2.5),
            "hit_start_depth": torch.full((2,), 2.8),
            "hit_end_depth": torch.full((2,), 3.2),
            "observation_type": torch.ones(2, dtype=torch.int8),
            "confidence": torch.ones(2),
        }
    )
    audit = evidence.audit()
    assert audit["seed_bound_ray_count"] == 1
    assert audit["observation_space_ray_count"] == 1
    assert audit["factor_camera_count"] == 1

    canonical = _interval_foliage(3.0)
    # This lineage was established independently before the dense
    # observation-space ray. The ray may refine it but cannot create/promote
    # such support itself.
    canonical.support_sequence_count.fill_(2)
    dynamic = _interval_dynamic_foliage(3.0)
    # Check the two physical consumers independently: the seed-bound row uses
    # canonical mass; the ownerless row can use an already verified canonical
    # lineage or its exact-camera dynamic residual.
    canonical_loss, canonical_interval = evidence.interval_factor(
        _interval_camera(), canonical, maximum_rays=2
    )
    evidence = FoliageRayEvidence(
        {
            "offsets": torch.tensor([0, 1], dtype=torch.int64),
            "camera_ids": torch.tensor([7, 7], dtype=torch.int32),
            "pixels": torch.full((2, 2), 50.0),
            "source_image_sizes": torch.full(
                (2, 2), 100, dtype=torch.int32
            ),
            "free_end_depth": torch.full((2,), 2.5),
            "hit_start_depth": torch.full((2,), 2.8),
            "hit_end_depth": torch.full((2,), 3.2),
            "observation_type": torch.ones(2, dtype=torch.int8),
            "confidence": torch.ones(2),
        }
    )
    dynamic_loss, dynamic_interval = evidence.interval_factor(
        _interval_camera(), dynamic, maximum_rays=2
    )
    assert canonical_interval["rays"] == 2
    assert canonical_interval["hit_rays"] == 2
    assert canonical_interval["ownerless_hit_rays"] == 1
    assert canonical_interval["ownerless_canonical_hit_rays"] == 1
    assert canonical_interval["verified_ownerless_hit_rays"] == 1
    assert canonical_interval["ownerless_supported_hit_rays"] == 1
    assert canonical_interval["ownerless_missing_supported_hit_rays"] == 0
    assert canonical_interval["excluded_ownerless_hit_rays"] == 0
    assert canonical_interval["exact_dynamic_candidate_evaluations"] == 0
    assert dynamic_interval["exact_dynamic_candidate_evaluations"] == 2
    assert dynamic_interval["ownerless_dynamic_hit_rays"] == 1
    assert dynamic_interval["ownerless_missing_dynamic_hit_rays"] == 0
    assert dynamic_interval["ownerless_supported_hit_rays"] == 1
    canonical_loss.backward()
    dynamic_loss.backward()
    assert canonical.opacity_logits.grad.abs().sum() > 0
    assert dynamic.opacity_logits.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_ownerless_verified_row_audit_crosses_cuda_to_cpu_safely():
    evidence = FoliageRayEvidence(
        {
            "offsets": torch.tensor([0, 0], dtype=torch.int64),
            "camera_ids": torch.tensor([7], dtype=torch.int32),
            "pixels": torch.full((1, 2), 50.0),
            "source_image_sizes": torch.full(
                (1, 2), 100, dtype=torch.int32
            ),
            "free_end_depth": torch.tensor([2.5]),
            "hit_start_depth": torch.tensor([2.8]),
            "hit_end_depth": torch.tensor([3.2]),
            "observation_type": torch.ones(1, dtype=torch.int8),
            "confidence": torch.ones(1),
        }
    )
    foliage = _interval_foliage(3.0).cuda()
    foliage.support_sequence_count.fill_(2)

    loss, audit = evidence.interval_factor(_interval_camera(), foliage)

    assert loss.is_cuda
    assert audit["ownerless_canonical_hit_rays"] == 1
    assert audit["ownerless_missing_supported_hit_rays"] == 0
    assert evidence.ownerless_canonical_verified.tolist() == [True]


def test_ownerless_hit_without_canonical_interval_support_is_dynamic_only():
    evidence = FoliageRayEvidence(
        {
            # No seed owns this dense traversal-local observation.
            "offsets": torch.tensor([0, 0], dtype=torch.int64),
            "camera_ids": torch.tensor([7], dtype=torch.int32),
            "pixels": torch.tensor([[80.0, 50.0]]),
            "source_image_sizes": torch.tensor(
                [[100, 100]], dtype=torch.int32
            ),
            "free_end_depth": torch.tensor([2.5]),
            "hit_start_depth": torch.tensor([2.8]),
            "hit_end_depth": torch.tensor([3.2]),
            "observation_type": torch.ones(1, dtype=torch.int8),
            "confidence": torch.ones(1),
        }
    )
    # A canonical Gaussian on a different ray must not be pulled toward this
    # sequence-local leaf merely because it is the global top-k fallback.
    foliage = _interval_foliage(3.0)
    camera = _interval_camera()
    camera.focal_x = 50.0
    camera.focal_y = 50.0

    loss, audit = evidence.interval_factor(camera, foliage)

    assert audit["ownerless_hit_rays"] == 1
    assert audit["verified_ownerless_hit_rays"] == 0
    assert audit["excluded_ownerless_hit_rays"] == 1
    assert audit["ownerless_dynamic_hit_rays"] == 0
    assert audit["ownerless_missing_dynamic_hit_rays"] == 1
    assert audit["ownerless_missing_supported_hit_rays"] == 1
    # Missing exact-owner mass remains a measurable posterior deficit, but
    # the unrelated canonical Gaussian receives no gradient from it.
    assert float(loss) > 5.0
    assert not loss.requires_grad
    assert evidence.audit()["unique_verified_ownerless_rows"] == 0


def test_ownerless_hit_rejects_dynamic_birth_owned_by_another_camera():
    evidence = FoliageRayEvidence(
        {
            "offsets": torch.tensor([0, 0], dtype=torch.int64),
            "camera_ids": torch.tensor([7], dtype=torch.int32),
            "pixels": torch.tensor([[50.0, 50.0]]),
            "source_image_sizes": torch.tensor(
                [[100, 100]], dtype=torch.int32
            ),
            "free_end_depth": torch.tensor([2.5]),
            "hit_start_depth": torch.tensor([2.8]),
            "hit_end_depth": torch.tensor([3.2]),
            "observation_type": torch.ones(1, dtype=torch.int8),
            "confidence": torch.ones(1),
        }
    )
    foliage = _interval_dynamic_foliage(3.0, support_camera_id=8)

    loss, audit = evidence.interval_factor(_interval_camera(), foliage)

    assert audit["exact_dynamic_candidate_evaluations"] == 0
    assert audit["ownerless_dynamic_hit_rays"] == 0
    assert audit["ownerless_missing_dynamic_hit_rays"] == 1
    assert audit["ownerless_missing_supported_hit_rays"] == 1
    assert float(loss) > 5.0
    assert not loss.requires_grad


def test_ownerless_ray_survives_dynamic_to_static_detail_fusion():
    evidence = _ownerless_interval_evidence(confidence=0.1)
    foliage = _interval_static_detail_foliage(3.0, support_camera_id=7)

    loss, audit = evidence.interval_factor(
        _interval_camera(),
        foliage,
        canonical_candidate_mask=torch.ones(1, dtype=torch.bool),
        # This is the topology-stage contract: detail is not yet a canonical
        # RGB-hit owner, but its exact fused ray ownership remains valid.
        canonical_hit_candidate_mask=torch.zeros(1, dtype=torch.bool),
    )
    loss.backward()

    assert audit["exact_static_candidate_evaluations"] == 1
    assert audit["ownerless_static_hit_rays"] == 1
    assert audit["ownerless_supported_hit_rays"] == 1
    assert audit["ownerless_missing_supported_hit_rays"] == 0
    assert audit["hit_optical_existence"] > 0
    assert foliage.opacity_logits.grad is not None
    assert float(foliage.opacity_logits.grad) < 0


def test_ownerless_ray_rejects_static_detail_owned_by_another_camera():
    evidence = _ownerless_interval_evidence()
    foliage = _interval_static_detail_foliage(3.0, support_camera_id=8)

    loss, audit = evidence.interval_factor(
        _interval_camera(),
        foliage,
        canonical_candidate_mask=torch.ones(1, dtype=torch.bool),
        canonical_hit_candidate_mask=torch.zeros(1, dtype=torch.bool),
    )

    assert audit["exact_static_candidate_evaluations"] == 0
    assert audit["ownerless_static_hit_rays"] == 0
    assert audit["ownerless_missing_supported_hit_rays"] == 1
    assert float(loss) > 5.0
    # Static detail is globally visible, so the non-owner still supplies
    # valid pre-hit free-space evidence; it must not supply a positive hit.
    assert loss.requires_grad
    loss.backward()
    assert float(foliage.opacity_logits.grad) >= 0


def test_static_detail_ray_prefit_starts_in_topology_only():
    assert not _static_detail_ray_trainable("canonical_bootstrap")
    assert _static_detail_ray_trainable("topology")
    assert _static_detail_ray_trainable("static_foliage")
    assert not _static_detail_ray_trainable("canonical_polish")


def test_static_detail_isolated_sh_is_all_view_but_geometry_mass_are_owned():
    ownership = torch.tensor([1.0, 0.0, 1.0])
    geometry, appearance, opacity = _static_detail_isolated_gradient_gates(
        ownership
    )
    assert geometry is ownership
    assert appearance is None
    assert opacity is ownership


def test_periodic_camera_schedule_consumes_a_contiguous_evidence_epoch():
    schedule = np.asarray([3, 1, 0, 2], dtype=np.int64)
    executed_steps = [1, 3, 5, 7]
    consumed = [
        int(schedule[_periodic_schedule_index(step, 2)])
        for step in executed_steps
    ]
    assert consumed == schedule.tolist()
    with pytest.raises(ValueError, match="not a scheduled"):
        _periodic_schedule_index(0, 2)
    with pytest.raises(ValueError, match="positive"):
        _periodic_schedule_index(0, 0)


def test_static_ray_prefit_has_owned_positive_hits_and_global_free_space():
    class Foliage:
        xyz = torch.zeros(3, 3)
        static_leaf_mask = torch.tensor([False, True, True])
        support_camera_ids = torch.tensor(
            [[-1, -1], [0, -1], [1, -1]], dtype=torch.int32
        )

        def __len__(self):
            return 3

    args = SimpleNamespace(
        reconstruction_target="static",
        static_detail_canonical_ownership=True,
    )
    active = torch.tensor([True, False, False])
    lookup = torch.tensor([0, 1], dtype=torch.int16)
    free, hit = _static_ray_candidate_masks(
        args,
        Foliage(),
        active,
        phase="topology",
        camera_id=0,
        camera_sequence_lookup=lookup,
    )
    assert torch.equal(free, torch.tensor([True, True, True]))
    assert torch.equal(hit, torch.tensor([True, True, False]))

    _, bootstrap_hit = _static_ray_candidate_masks(
        args,
        Foliage(),
        active,
        phase="canonical_bootstrap",
        camera_id=0,
        camera_sequence_lookup=lookup,
    )
    assert torch.equal(bootstrap_hit, active)


def test_candidate_interval_descendant_index_rebuilds_after_split():
    evidence = _interval_evidence(1)
    foliage = _interval_foliage(3.0)

    _, before = evidence.interval_factor(_interval_camera(), foliage)
    foliage.split(torch.tensor([0]))
    _, after = evidence.interval_factor(_interval_camera(), foliage)

    assert before["candidate_evaluations"] == 1
    assert after["candidate_evaluations"] == 2
    # The parent evidence row is a re-verification candidate, not proof at
    # either displaced child centre.
    assert foliage.evidence_primitive_id.tolist() == [-1, -1]
    assert foliage.candidate_evidence_primitive_id.tolist() == [0, 0]


def test_interval_factor_uses_all_gaussians_on_same_ray_not_only_lineage():
    evidence = _interval_evidence(1)
    wrong = _interval_foliage(1.0)
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        # Primitive zero produced the retained evidence row but lies in
        # front. Primitive one has different lineage and correctly explains
        # the measured hit interval.
        "centers": torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 3.05]]
        ),
        "scales": torch.full((2, 3), 0.2),
        "colors": torch.full((2, 3), 0.4),
        "opacities": torch.full((2, 1), 0.5),
        "quaternions": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 2
        ),
        "ray_evidence": {
            "offsets": torch.tensor([0, 1, 1], dtype=torch.int64)
        },
    }
    complete = VolumetricFoliageModel(1, device="cpu")
    complete.initialize_from_volume_state(payload)

    wrong_loss, _ = evidence.interval_factor(
        _interval_camera(), wrong
    )
    complete_loss, audit = evidence.interval_factor(
        _interval_camera(), complete
    )

    assert audit["candidate_evaluations"] == 2
    assert complete_loss < wrong_loss
    complete_loss.backward()
    assert complete.opacity_logits.grad[1].abs().sum() > 0


def test_candidate_interval_sampling_eventually_covers_every_row():
    count = 6
    evidence = FoliageRayEvidence(
        {
            "offsets": torch.arange(count + 1, dtype=torch.int64),
            "camera_ids": torch.full(
                (count,), 7, dtype=torch.int32
            ),
            "pixels": torch.full((count, 2), 50.0),
            "source_image_sizes": torch.full(
                (count, 2), 100, dtype=torch.int32
            ),
            "free_end_depth": torch.full((count,), 2.5),
            "hit_start_depth": torch.full((count,), 2.8),
            "hit_end_depth": torch.full((count,), 3.2),
            "observation_type": torch.ones(
                count, dtype=torch.int8
            ),
            "confidence": torch.ones(count),
        }
    )
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor(
            [[0.0, 0.0, 3.0]] * count
        ),
        "scales": torch.full((count, 3), 0.2),
        "colors": torch.full((count, 3), 0.4),
        "opacities": torch.full((count, 1), 0.5),
        "quaternions": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * count
        ),
        "ray_evidence": {
            "offsets": torch.arange(count + 1, dtype=torch.int64)
        },
    }
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)

    for update in range(3):
        evidence.interval_factor(
            _interval_camera(),
            foliage,
            maximum_rays=2,
            sample_update=update,
        )

    assert evidence.audit()["interval_unique_rows"] == count
    assert evidence.audit()["interval_row_coverage"] == 1.0


def test_ray_camera_schedule_balances_row_coverage_not_camera_calls():
    camera_ids = torch.tensor([7, 7] + [8] * 6, dtype=torch.int32)
    count = len(camera_ids)
    evidence = FoliageRayEvidence(
        {
            "offsets": torch.arange(count + 1, dtype=torch.int64),
            "camera_ids": camera_ids,
            "pixels": torch.full((count, 2), 50.0),
            "source_image_sizes": torch.full(
                (count, 2), 100, dtype=torch.int32
            ),
            "free_end_depth": torch.full((count,), 2.5),
            "hit_start_depth": torch.full((count,), 2.8),
            "hit_end_depth": torch.full((count,), 3.2),
            "observation_type": torch.ones(count, dtype=torch.int8),
            "confidence": torch.ones(count),
        }
    )
    foliage = _interval_foliage(3.0)
    views = {
        camera_id: SimpleNamespace(
            **{**vars(_interval_camera()), "colmap_id": camera_id}
        )
        for camera_id in (7, 8)
    }
    scheduled = []
    for update in range(4):
        camera_id = evidence.scheduled_camera_id(update)
        scheduled.append(camera_id)
        evidence.interval_factor(
            views[camera_id], foliage, maximum_rays=2
        )

    # Camera 8 has three times as many rows, so equal row coverage requires
    # one call for camera 7 and three calls for camera 8.
    assert scheduled == [7, 8, 8, 8]
    audits = evidence.audit()["camera_epoch_audits"]
    assert audits["7"]["coverage"] == 1.0
    assert audits["8"]["coverage"] == 1.0


def test_candidate_interval_factor_penalizes_only_mass_before_free_boundary():
    evidence = _interval_evidence(-1)
    before = _interval_foliage(1.0)
    behind = _interval_foliage(3.0)

    before_loss, audit = evidence.interval_factor(
        _interval_camera(), before
    )
    behind_loss, _ = evidence.interval_factor(
        _interval_camera(), behind
    )

    assert audit["confirmed_free_rays"] == 1
    assert before_loss > 100 * behind_loss


def test_foliage_ray_runtime_state_survives_exact_resume():
    evidence = _interval_evidence(-1)
    foliage = _interval_foliage(3.0)
    evidence.interval_factor(_interval_camera(), foliage)
    state = evidence.capture_runtime_state()

    restored = _interval_evidence(-1)
    restored.restore_runtime_state(state)

    assert restored.audit() == evidence.audit()
    assert torch.equal(
        restored.ownerless_canonical_verified,
        evidence.ownerless_canonical_verified,
    )


def test_foliage_ray_resume_continues_exact_per_camera_epoch():
    count = 7
    evidence = FoliageRayEvidence(
        {
            "offsets": torch.arange(count + 1, dtype=torch.int64),
            "camera_ids": torch.full(
                (count,), 7, dtype=torch.int32
            ),
            "pixels": torch.full((count, 2), 50.0),
            "source_image_sizes": torch.full(
                (count, 2), 100, dtype=torch.int32
            ),
            "free_end_depth": torch.full((count,), 2.5),
            "hit_start_depth": torch.full((count,), 2.8),
            "hit_end_depth": torch.full((count,), 3.2),
            # The last row is explicitly unknown and is not part of a
            # free/hit evidence epoch.
            "observation_type": torch.tensor(
                [1, 1, -1, 1, -1, 1, 0], dtype=torch.int8
            ),
            "confidence": torch.ones(count),
        }
    )
    foliage = _interval_foliage(3.0)
    evidence.interval_factor(
        _interval_camera(), foliage, maximum_rays=2
    )
    restored = FoliageRayEvidence(
        {
            "offsets": torch.arange(count + 1, dtype=torch.int64),
            "camera_ids": torch.full(
                (count,), 7, dtype=torch.int32
            ),
            "pixels": torch.full((count, 2), 50.0),
            "source_image_sizes": torch.full(
                (count, 2), 100, dtype=torch.int32
            ),
            "free_end_depth": torch.full((count,), 2.5),
            "hit_start_depth": torch.full((count,), 2.8),
            "hit_end_depth": torch.full((count,), 3.2),
            "observation_type": torch.tensor(
                [1, 1, -1, 1, -1, 1, 0], dtype=torch.int8
            ),
            "confidence": torch.ones(count),
        }
    )
    restored.restore_runtime_state(evidence.capture_runtime_state())

    evidence.interval_factor(
        _interval_camera(), foliage, maximum_rays=2
    )
    restored.interval_factor(
        _interval_camera(), foliage, maximum_rays=2
    )

    assert torch.equal(
        restored.interval_row_visits, evidence.interval_row_visits
    )
    assert restored.capture_runtime_state()[
        "camera_sampler_states"
    ]["7"]["cursor"] == evidence.capture_runtime_state()[
        "camera_sampler_states"
    ]["7"]["cursor"]
    audit = restored.audit()
    assert audit["interval_effective_rows"] == 6
    assert audit["interval_unique_rows"] == 4
    assert audit["camera_epoch_audits"]["7"]["never_visited"] == 2


def test_hybrid_volume_owner_activates_immediately_after_bootstrap():
    assert TRAINING_PROFILES["hybrid_fast"]["foliage_start"] == 0.08
    assert TRAINING_PROFILES["hybrid_quality"]["foliage_start"] == 0.12
    assert (
        dict(TRAINING_PROFILES["hybrid_quality"]["phases"])[
            "static_foliage"
        ]
        == 0.50
    )
    assert (
        TRAINING_PROFILES["hybrid_quality"]["dynamic_start"]
        == 0.50
    )
    assert (
        TRAINING_PROFILES["hybrid_handoff_quality"]["dynamic_start"]
        == 0.02
    )
    assert not _dynamic_enabled(
        599, 30_000, "hybrid_handoff_quality"
    )
    assert _dynamic_enabled(
        600, 30_000, "hybrid_handoff_quality"
    )


def test_conditioned_schedule_balances_exact_evidence_in_every_prefix():
    rgb = np.tile(np.arange(4, dtype=np.int64), 4)
    schedule, audit = _evidence_biased_schedule(
        rgb,
        [10, 11],
        fraction=0.75,
        seed=7,
        active_mask=np.ones(len(rgb), dtype=bool),
        minimum_full_scene_visits=3,
    )

    evidence = np.isin(schedule, [10, 11])
    # Three visits per full-scene camera are protected. Only four slots are
    # replaceable, but they are interleaved across the entire interval rather
    # than delayed until all three protected epochs have run.
    assert int(evidence.sum()) == 4
    assert np.array_equal(np.flatnonzero(evidence), [0, 4, 8, 12])
    for view_id in range(4):
        assert int((schedule == view_id).sum()) == 3
    assert abs(int((schedule == 10).sum()) - int((schedule == 11).sum())) <= 1
    assert audit["evidence_steps"] == 4
    assert audit["requested_evidence_steps"] == 12
    assert audit["protected_full_scene_steps"] == 12
    assert (
        audit["protected_placement"]
        == "uniform_interleaved_complete_epochs"
    )
    assert audit["minimum_full_scene_visits"] == 3
    assert audit["coverage_preserving"]
    assert audit["realized_fraction"] == 0.25


def test_conditioned_schedule_protects_only_active_rgb_visits():
    rgb = np.tile(np.arange(3, dtype=np.int64), 4)
    active = np.zeros(len(rgb), dtype=bool)
    active[3:9] = True
    schedule, audit = _evidence_biased_schedule(
        rgb,
        [10],
        fraction=0.75,
        seed=5,
        active_mask=active,
        minimum_full_scene_visits=3,
    )

    # There are only two complete active epochs, so both are protected. The
    # inactive prefix/suffix remain bit-identical to RGB while the active
    # interval is a fresh balanced conditioned epoch.
    assert np.array_equal(schedule[:3], rgb[:3])
    assert np.array_equal(schedule[9:], rgb[9:])
    assert sorted(schedule[3:9].tolist()) == [0, 0, 1, 1, 2, 2]
    assert audit["minimum_full_scene_visits"] == 2
    assert audit["protected_full_scene_steps"] == 6
    assert audit["evidence_steps"] == 0


def test_resume_conditioned_visits_migrate_the_saved_schedule_prefix():
    counts, audit = _resume_conditioned_visit_counts(
        {
            "iteration": 10,
            "schedule_horizon": 10,
            "training_profile": "hybrid_handoff_quality",
            "schedules": {
                "conditioned": np.asarray(
                    [0, 1, 2, 0, 1, 2, 0, 1, 2, 2],
                    dtype=np.int64,
                )
            },
        },
        view_count=3,
    )
    # Iteration 10 is canonical polish for this compressed profile, so its
    # final camera row was stored in the schedule but never optimized.
    assert counts.tolist() == [3, 3, 3]
    assert audit["source"] == "legacy_saved_schedule_exact_migration"
    assert audit["carried_conditioned_steps"] == 9


def test_resume_conditioned_visits_prefers_explicit_runtime_ledger():
    counts, audit = _resume_conditioned_visit_counts(
        {
            "iteration": 6,
            "conditioned_visit_counts": torch.tensor([0, 4, 1]),
            # A repaired future schedule must not rewrite the old prefix.
            "schedules": {"conditioned": np.asarray([2] * 10)},
        },
        view_count=3,
    )
    assert counts.tolist() == [0, 4, 1]
    assert audit["source"] == "explicit_runtime_ledger"
    assert audit["carried_conditioned_steps"] == 5


def test_volume_split_selection_reserves_each_tree_before_refill():
    indices = torch.arange(8)
    score = torch.tensor([9.0, 8.0, 7.0, 6.0, 2.0, 1.0, 0.5, 0.1])
    # A global top-k would allocate everything to tree 0.
    instances = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
    selected = _balanced_instance_topk(
        indices, score, instances, quota=4
    )

    selected_instances = instances[selected]
    assert set(selected_instances.tolist()) == {0, 1, 2}
    assert selected.tolist()[0] == 0


def test_volume_split_selection_balances_spatial_cells_inside_one_tree():
    indices = torch.arange(8)
    score = torch.tensor([9.0, 8.0, 7.0, 6.0, 2.0, 1.0, 0.5, 0.1])
    instances = torch.zeros(8, dtype=torch.int64)
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.1, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.1, 0.0, 0.0],
        ]
    )
    selected = _balanced_instance_topk(
        indices,
        score,
        instances,
        quota=4,
        xyz=xyz,
        spatial_cell_size=1.0,
    )

    selected_cells = torch.floor(xyz[selected, 0]).to(torch.int64)
    assert set(selected_cells.tolist()) == {0, 2, 4}


def test_volume_split_selection_reserves_exact_owner_contexts():
    indices = torch.arange(9)
    score = torch.tensor(
        [100.0, 90.0, 80.0, 70.0, 3.0, 2.0, 1.0, 0.5, 0.1]
    )
    instances = torch.zeros(9, dtype=torch.int64)
    contexts = torch.tensor([10, 10, 10, 10, 20, 20, 30, 30, 30])
    selected = _balanced_instance_topk(
        indices,
        score,
        instances,
        quota=4,
        context_id=contexts,
    )

    assert len(selected) == 4
    assert set(contexts[selected].tolist()) == {10, 20, 30}


def test_volume_split_selection_reserves_lineage_families_inside_owner():
    indices = torch.arange(8)
    score = torch.tensor([100.0, 90.0, 80.0, 70.0, 1.0, 0.9, 0.8, 0.7])
    instances = torch.zeros(8, dtype=torch.int64)
    contexts = torch.full((8,), 10, dtype=torch.int64)
    families = torch.tensor([3, 3, 3, 3, 4, 4, 4, 4])

    selected = _balanced_instance_topk(
        indices,
        score,
        instances,
        quota=4,
        context_id=contexts,
        lineage_family_id=families,
    )

    assert len(selected) == 4
    assert (families[selected] == 3).sum().item() == 2
    assert (families[selected] == 4).sum().item() == 2


def test_extra_surface_topology_uses_nonzero_rigid_weight():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_unified_outdoor_teacher.py"
    ).read_text()
    assert 'topology_task["w_plane"]' in source
    assert (
        'topology_task["p_rigid"]\n'
        '                * topology_task["w_topology"]'
    ) not in source


def test_canopy_topology_signal_tracks_target_high_frequency():
    task = {"w_topology": torch.ones(8, 8)}
    flat = torch.full((3, 8, 8), 0.5)
    checker = (
        torch.arange(8)[:, None] + torch.arange(8)[None, :]
    ).remainder(2).float()
    checker = checker[None].repeat(3, 1, 1)

    assert (
        _canopy_topology_signal(checker, task).mean()
        > _canopy_topology_signal(flat, task).mean()
    )


def test_rigid_residual_patch_pool_routes_gradient_to_small_edge():
    target = torch.zeros(3, 48, 48)
    for y, x in ((8, 8), (8, 34), (34, 8), (34, 34)):
        target[:, y : y + 3, x : x + 3] = 1
    prediction = torch.zeros_like(target, requires_grad=True)
    rigid = torch.ones(48, 48)
    loss, audit = _rigid_residual_patch_loss(
        prediction, target, rigid, maximum_patches=4, radius=2
    )
    loss.backward()
    assert audit["patches"] == 4
    # Four radius-two patches can cover at most 100 pixels. Spatial NMS keeps
    # them on independent residual components instead of spending all four
    # slots on adjacent centres around one component.
    assert audit["pixels"] >= 80
    assert audit["independent_peak_candidates"] >= 4
    for y, x in ((8, 8), (8, 34), (34, 8), (34, 34)):
        assert prediction.grad[:, y : y + 3, x : x + 3].abs().sum() > 0


def test_chart_uv_topology_obeys_mature_world_geometry_freeze():
    args = SimpleNamespace(
        mature_handoff_surface_policy="appearance_only",
        densify_from_iter=10,
        densify_until_iter=20,
        chart_quadtree_growth_fraction=0.3,
    )
    assert not _surface_topology_active(9, args)
    assert not _chart_topology_active(9, args)
    args.mature_handoff_surface_policy = "joint"
    assert _chart_topology_active(9, args)


def test_surface_canopy_ownership_only_penalizes_front_layer():
    task = {
        "p_canopy_core": torch.ones(2, 2),
        "p_boundary_uncertain": torch.zeros(2, 2),
    }

    def package(surface_depth, volume_depth, volume_alpha=0.8):
        return SimpleNamespace(
            surface_alpha=torch.ones(1, 2, 2),
            volume_alpha=torch.full((1, 2, 2), volume_alpha),
            surface_depth=torch.full((1, 2, 2), surface_depth),
            volume_depth=torch.full((1, 2, 2), volume_depth),
        )

    in_front = _soft_surface_canopy_conflict(
        package(1.0, 2.0), task
    )
    behind = _soft_surface_canopy_conflict(
        package(3.0, 2.0), task
    )
    gap_ray = _soft_surface_canopy_conflict(
        package(3.0, 0.0, volume_alpha=0.0), task
    )

    assert in_front > 20 * behind
    assert behind < gap_ray
    assert gap_ray < 0.03


def test_counterfactual_transparency_routes_only_volume_alpha_gradient():
    target = torch.zeros(3, 2, 2)
    mixed = torch.full((3, 2, 2), 0.8, requires_grad=True)
    surface = torch.zeros(3, 2, 2, requires_grad=True)
    volume_alpha = torch.full((1, 2, 2), 0.5, requires_grad=True)
    surface_alpha = torch.ones(1, 2, 2, requires_grad=True)
    task = {
        "p_rigid": torch.ones(2, 2),
        "p_canopy": torch.zeros(2, 2),
        "p_canopy_core": torch.zeros(2, 2),
        "p_transient": torch.zeros(2, 2),
        "p_boundary_uncertain": torch.zeros(2, 2),
        "w_rgb": torch.ones(2, 2),
    }
    loss, audit = _counterfactual_volume_transparency_loss(
        mixed,
        surface,
        target,
        volume_alpha,
        surface_alpha,
        task,
    )
    loss.backward()
    assert audit["supported_pixels"] == 4
    assert audit["mean_responsibility"] > 0.99
    assert volume_alpha.grad is not None
    assert float(volume_alpha.grad.sum()) > 0
    # RGB comparison and surface support are detached evidence, never a
    # second trainable image owner.
    assert mixed.grad is None
    assert surface.grad is None
    assert surface_alpha.grad is None


def test_static_global_cleanup_routes_only_negative_geometry_optical_signal():
    target = torch.zeros(3, 2, 2)
    mixed = torch.full((3, 2, 2), 0.8, requires_grad=True)
    surface = torch.zeros(3, 2, 2, requires_grad=True)
    volume_alpha = torch.full((1, 2, 2), 0.5, requires_grad=True)
    surface_alpha = torch.ones(1, 2, 2, requires_grad=True)
    task = {
        "p_rigid": torch.ones(2, 2),
        "p_canopy": torch.zeros(2, 2),
        "p_canopy_core": torch.zeros(2, 2),
        "p_sky": torch.zeros(2, 2),
        "p_transient": torch.zeros(2, 2),
        "p_boundary_uncertain": torch.zeros(2, 2),
        "w_rgb": torch.ones(2, 2),
    }

    loss, audit = _static_detail_global_cleanup_loss(
        mixed,
        surface,
        target,
        volume_alpha,
        surface_alpha,
        task,
        counterfactual_weight=0.5,
    )
    loss.backward()

    assert audit["contract"] == STATIC_DETAIL_GLOBAL_CLEANUP_CONTRACT
    assert audit["rigid_free_supported_pixels"] == 4
    assert audit["gradient_permissions"] == {
        "static_detail_xyz_scale_rotation": True,
        "static_detail_optical_mass": True,
        "static_detail_sh": False,
        "surface_sky_uncertainty": False,
        "topology_statistics": False,
    }
    assert volume_alpha.grad is not None
    assert float(volume_alpha.grad.sum()) > 0
    assert mixed.grad is None
    assert surface.grad is None
    assert surface_alpha.grad is None


def test_persistent_envelope_cleanup_routes_only_global_negative_signal():
    target = torch.zeros(3, 2, 2)
    mixed = torch.full((3, 2, 2), 0.8, requires_grad=True)
    surface = torch.zeros(3, 2, 2, requires_grad=True)
    volume_alpha = torch.full((1, 2, 2), 0.5, requires_grad=True)
    surface_alpha = torch.ones(1, 2, 2, requires_grad=True)
    task = {
        "p_rigid": torch.ones(2, 2),
        "p_canopy": torch.zeros(2, 2),
        "p_canopy_core": torch.zeros(2, 2),
        "p_sky": torch.zeros(2, 2),
        "p_transient": torch.zeros(2, 2),
        "p_boundary_uncertain": torch.zeros(2, 2),
        "w_rgb": torch.ones(2, 2),
    }

    loss, audit = _persistent_envelope_global_cleanup_loss(
        mixed,
        surface,
        target,
        volume_alpha,
        surface_alpha,
        task,
        counterfactual_weight=0.5,
    )
    loss.backward()

    assert audit["contract"] == PERSISTENT_ENVELOPE_GLOBAL_CLEANUP_CONTRACT
    assert audit["rigid_free_supported_pixels"] == 4
    assert audit["gradient_permissions"] == {
        "persistent_envelope_xyz_scale_rotation": True,
        "persistent_envelope_optical_mass": True,
        "persistent_envelope_sh": False,
        "surface_sky_uncertainty": False,
        "topology_statistics": False,
    }
    assert volume_alpha.grad is not None
    assert float(volume_alpha.grad.sum()) > 0
    assert mixed.grad is None
    assert surface.grad is None
    assert surface_alpha.grad is None


def test_static_public_phase_names_never_advertise_dynamic_output():
    assert (
        _public_phase_name("dynamic_appearance", "static")
        == "static_detail_refinement"
    )
    assert (
        _public_phase_name("ownership_cleanup", "static")
        == "static_joint_optical_cleanup"
    )
    assert (
        _public_phase_name(
            "dynamic_appearance", "sequence_conditioned_legacy"
        )
        == "dynamic_appearance"
    )


def test_counterfactual_transparency_softly_retains_real_leaf_owner():
    target = torch.zeros(3, 1, 2)
    mixed = torch.tensor([[[0.0, 0.4]], [[0.0, 0.4]], [[0.0, 0.4]]])
    surface = torch.tensor([[[0.8, 0.0]], [[0.8, 0.0]], [[0.8, 0.0]]])
    alpha = torch.full((1, 1, 2), 0.5, requires_grad=True)
    task = {
        "p_rigid": torch.zeros(1, 2),
        "p_canopy": torch.ones(1, 2),
        "p_canopy_core": torch.ones(1, 2),
        "p_transient": torch.zeros(1, 2),
        "p_boundary_uncertain": torch.zeros(1, 2),
        "w_rgb": torch.ones(1, 2),
    }
    _, audit = _counterfactual_volume_transparency_loss(
        mixed, surface, target, alpha, torch.ones(1, 1, 2), task
    )
    # The first ray is correctly explained by the mixed leaf, while the
    # second is better explained by the surface gap. Both remain continuous.
    assert 0 < audit["mean_responsibility"] < 1
