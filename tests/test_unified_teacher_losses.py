from types import SimpleNamespace
from pathlib import Path
import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from scripts.train_unified_outdoor_teacher import (
    CANONICAL_OCCLUSION_ACTIVATION_CONTRACT,
    CANONICAL_OCCLUSION_DIRECTIONAL_COMPLETION_CONTRACT,
    CANONICAL_OCCLUSION_COMPLETION_CONTRACT,
    CANONICAL_OCCLUSION_ORDER_CONTRACT,
    MOGE3_CANOPY_OPTICAL_CONTRACT,
    DYNAMIC_LIFECYCLE_REPAIR_TARGET,
    SURFACE_SCREEN_TOPOLOGY_CONTRACT,
    SURFACE_OWNERSHIP_REPAIR_TARGET,
    STATIC_DETAIL_GLOBAL_CLEANUP_CONTRACT,
    STATIC_DETAIL_GLOBAL_CLEANUP_V113_PREDECESSOR_CONTRACT,
    STATIC_DETAIL_GLOBAL_CLEANUP_V102_PREDECESSOR_CONTRACT,
    STATIC_DETAIL_GLOBAL_CLEANUP_V106_PREDECESSOR_CONTRACT,
    STATIC_DETAIL_GLOBAL_CLEANUP_V99_PREDECESSOR_CONTRACT,
    STATIC_DETAIL_GLOBAL_CLEANUP_V83_PREDECESSOR_CONTRACT,
    STATIC_DETAIL_ISOLATED_CONTRACT,
    STATIC_DETAIL_ISOLATED_V84_PREDECESSOR_CONTRACT,
    STATIC_DETAIL_ISOLATED_V85_PREDECESSOR_CONTRACT,
    STATIC_STAGE_RGB_ROLE_CONTRACT,
    STATIC_STAGE_RGB_ROLE_V85_PREDECESSOR_CONTRACT,
    STATIC_VOLUME_ISOLATED_CONTRACT,
    STATIC_VOLUME_ISOLATED_V84_PREDECESSOR_CONTRACT,
    STATIC_VOLUME_ISOLATED_V85_PREDECESSOR_CONTRACT,
    STATIC_VOLUME_INTRINSIC_ALPHA_FLOOR,
    SHARED_ENVELOPE_OWNERSHIP_LOCALIZATION_CONTRACT,
    STATIC_DETAIL_SAME_SEQUENCE_GEOMETRY_WEIGHT,
    PERSISTENT_ENVELOPE_GLOBAL_CLEANUP_CONTRACT,
    PERSISTENT_ENVELOPE_GLOBAL_CLEANUP_V113_PREDECESSOR_CONTRACT,
    PERSISTENT_ENVELOPE_GLOBAL_CLEANUP_V99_PREDECESSOR_CONTRACT,
    STATIC_RAY_BIRTH_VISUAL_HULL_CONTRACT,
    STATIC_OPTICAL_POLICY_CONTRACT,
    STATIC_OPTICAL_POLICY_PREDECESSOR_CONTRACT,
    STATIC_OPTICAL_POLICY_STAGE3_FREEZE_PREDECESSOR_CONTRACT,
    STATIC_OPTICAL_POLICY_V86_PREDECESSOR_CONTRACT,
    TRAINING_PROFILES,
    VOLUME_OPACITY_SETTLE_CONTRACT,
    VOLUME_OPACITY_SETTLE_V113_PREDECESSOR_CONTRACT,
    MOGE3_CANOPY_OPTICAL_V113_PREDECESSOR_CONTRACT,
    _apply_training_profile_optimizer_defaults,
    _adapt_volume,
    _activation_iteration,
    _accumulate_volume_stats,
    _adaptive_geometry_scale,
    _assert_pointmap_posterior_contract,
    _assert_clean_training_output,
    _backward_conditioned_foliage,
    _backward_canonical_with_foliage_supplement,
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
    _checkpoint_due,
    _save_checkpoint,
    _write_json_atomic,
    _confirmed_contradiction_fraction,
    _complete_evidence_epoch_batch_size,
    _continued_surface_iteration,
    _conditioned_branch_active,
    _conditioned_base_gradient_gate,
    _conditioned_photo_losses,
    _counterfactual_volume_transparency_loss,
    _canonical_occlusion_completion_loss,
    _canonical_occlusion_completion_owner_mask,
    _canonical_occlusion_order_loss,
    _moge3_canopy_optical_loss,
    _moge3_depth_query_bounds,
    _moge3_uncovered_hit_proposals,
    _spatially_stratified_proposal_indices,
    _configure_moge3_runtime,
    _cap_canonical_occlusion_order_gradient_responsibility,
    _canonical_occlusion_schedule_active,
    _canonical_static_unknown_protection_mask,
    _enforce_canonical_occlusion_order_trust_region,
    _evidence_adaptive_role_quotas,
    _extend_volume_stats_for_births,
    _reset_volume_stats_after_topology,
    _geometry_losses,
    _projected_occluded_rigid_geometry_loss,
    _inherit_surface_optimizer_contract,
    _initialize_static_ray_birth_mass_handoff,
    _load_dav2_observation_patches,
    _masked_clamp_max_,
    _masked_ssim_loss,
    _parameter_loss_gradient_audit,
    _persistent_static_evidence_mask,
    _moge3_canopy_hit_owner_mask,
    _open_shared_envelope_localization_rows,
    _periodic_schedule_index,
    _apply_volume_role_gradients,
    _apply_volume_opacity_settle_policy,
    _apply_static_detail_opacity_settle_policy,
    _apply_strict_rigid_front_opacity_policy,
    _apply_persistent_rigid_front_conflict_policy,
    _persistent_rigid_front_conflict_rows,
    _persistent_optical_debt_active,
    _persistent_optical_ownership_states,
    _surface_replacement_retirement_allowed,
    _material_optical_source_gradient,
    _update_persistent_rigid_front_conflict_debt,
    _update_persistent_positive_optical_demand,
    _select_shared_envelope_localization_actions,
    _shared_envelope_localization_active,
    _post_growth_verification_maintenance_due,
    _shared_envelope_detail_split_plane_normals,
    _apply_shared_envelope_localization_model_transaction,
    _enforce_strict_rigid_front_post_step,
    _enforce_shared_optical_owner_post_step,
    _static_detail_optical_trust_groups,
    _enforce_static_detail_optical_mass_trust_region,
    _apply_static_optical_policy,
    _apply_static_ray_local_mass_handoff,
    _rebase_v86_static_optical_ownership_state,
    _finalize_static_ray_local_mass_handoff,
    _apply_surface_scale_limits_preserve_optical_mass,
    _surface_screen_limit_rows,
    _accumulate_static_replacement_evidence,
    _update_static_child_verification_from_render,
    _update_chart_atlas_learning_rate,
    _apply_mature_surface_gradient_policy,
    _phase,
    _public_phase_name,
    _prefix_stable_resume_ray_batch,
    _resolved_phase_schedule,
    _resolve_cpu_intraop_threads,
    _resume_conditioned_visit_counts,
    _restore_resume_camera_schedules,
    _restore_expired_factorized_receiver_mass,
    _restore_cuda_rng_states,
    _refresh_split_child_owner_colors,
    _retain_checkpoint_snapshot,
    _rollback_failed_split_families,
    _rigid_completion_seed_indices,
    _rigid_residual_patch_loss,
    _oriented_multiscale_high_frequency_loss,
    _resume_training_contract_differences,
    _restore_volume_optimizer_state,
    _soft_surface_canopy_conflict,
    _static_detail_exclusive_topology_active,
    _static_detail_ray_trainable,
    _static_detail_positive_opacity_growth_scale,
    _static_ray_hit_opacity_gradient_scale,
    _static_detail_isolated_gradient_gates,
    _static_detail_evidence_sequence_view_indices,
    _static_replacement_evidence_view_indices,
    _static_volume_isolated_gradient_gates,
    _static_volume_intrinsic_color_inputs,
    _static_stage_rgb_gradient_gates,
    _static_ray_candidate_masks,
    _static_detail_canonical_ownership_gate,
    _static_detail_positive_evidence_sequence_gate,
    _static_detail_same_sequence_appearance_gate,
    _static_detail_same_sequence_optical_gate,
    _static_detail_same_sequence_geometry_gate,
    _static_detail_optical_regularizer_weight,
    _static_detail_cleanup_gradient_gates,
    _static_detail_consensus_hit_gate,
    _static_detail_global_cleanup_loss,
    _surface_front_depth_query_bounds,
    _static_detail_global_negative_owner_mask,
    _static_scene_rgb_ownership,
    _static_spatial_uncertainty_active,
    _select_missing_static_detail_receiver_parents,
    _persistent_envelope_global_cleanup_loss,
    _surface_capture_to_device,
    _surface_topology_active,
    _trainer_repair_hash_change_is_allowed,
    _chart_topology_active,
    _volume_topology_ramp_scale,
    _validate_fixed_cameras,
    _validate_initialization_rgb_source,
    _validate_initialization_protocol,
    _validate_foliage_rigid_calibration,
    _validate_surface_warmstart,
    _volume_stats,
    _volume_split_authority,
    _volume_topology_active,
    _volume_topology_authority,
    _verification_debt_capacity_scale,
    _verification_debt_mask,
    _update_volume_learning_rates,
    _zero_volume_opacity_optimizer_rows,
    _enforce_static_detail_sh_trust_region,
    _write_rigid_stage_surface_handoff,
)
from outdoor.hybrid_gaussian_renderer import (
    LAYER_DYNAMIC_LEAF,
    PROPOSAL_NONE,
    PROPOSAL_RAY_BIRTH,
    PROPOSAL_SPLIT,
    VERIFICATION_UNVERIFIED,
    VERIFICATION_VERIFIED,
    VERIFICATION_MEASURED_SINGLE,
    VERIFICATION_FACTORIZED_RECEIVER,
    VERIFICATION_REJECTED,
    VolumetricFoliageModel,
    projected_gaussian_cross_section,
    static_detail_forward_visibility_gate,
)
from outdoor.training_evidence import (
    FoliageRayEvidence,
    OutdoorGeometryEvidence,
)
from scene import GaussianModel

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
RIGID_SURFACE_OPTIMIZER_WITH_POLISH = {
    **RIGID_SURFACE_OPTIMIZER,
    "non_position_lr_decay_from": 18_000,
    "non_position_lr_decay_until": 24_000,
    "non_position_lr_final_mult": 0.1,
}


def test_surface_screen_scale_projection_preserves_local_optical_mass():
    opacity = torch.logit(torch.tensor([[0.2], [0.4]]))
    surface = SimpleNamespace(
        _scaling=torch.zeros(2, 2),
        _opacity=opacity.clone(),
        get_opacity=torch.sigmoid(opacity),
    )
    before_tau = -torch.log1p(-surface.get_opacity[:, 0])
    before_mass = before_tau * surface._scaling.exp().prod(dim=1)
    audit = _apply_surface_scale_limits_preserve_optical_mass(
        surface,
        torch.tensor([0, 1]),
        torch.tensor([0.5, 1.0]),
        maximum_scale=2.0,
    )
    after_alpha = torch.sigmoid(surface._opacity[:, 0])
    after_mass = (
        -torch.log1p(-after_alpha)
        * surface._scaling.exp().prod(dim=1)
    )

    assert torch.allclose(after_mass, before_mass, rtol=1e-5, atol=1e-7)
    assert torch.allclose(surface._scaling[0].exp(), torch.full((2,), 0.5))
    assert audit["changed_rows"] == 1
    assert audit["unrealized_mass"] < 1e-6


def test_atlas_residual_screen_limit_includes_live_chart_prefix():
    surface = SimpleNamespace(get_xyz=torch.zeros(8, 3))

    class ChartProbe:
        @staticmethod
        def live_surface_rows(_surface):
            assert _surface is surface
            # Row 6 overlaps the residual suffix; row 99 must be rejected.
            return torch.tensor([1, 4, 6, 99])

    rows = _surface_screen_limit_rows(
        surface,
        ChartProbe(),
        "atlas_residual",
        residual_start=6,
        structural_count=8,
        device=torch.device("cpu"),
    )

    assert rows.tolist() == [1, 4, 6, 7]


def test_frozen_screen_limit_does_not_mutate_handoff_rows():
    rows = _surface_screen_limit_rows(
        SimpleNamespace(get_xyz=torch.zeros(3, 3)),
        None,
        "appearance_only",
        residual_start=3,
        structural_count=3,
        device=torch.device("cpu"),
    )
    assert rows.numel() == 0


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
        dynamic_leaf_mask=torch.tensor([False, False]),
        static_skeleton_mask=torch.tensor([False, False]),
        verification_state=torch.full(
            (2,), VERIFICATION_VERIFIED, dtype=torch.int8
        ),
        verified_camera_count=torch.full((2,), 2, dtype=torch.int16),
        verified_sequence_count=torch.ones(2, dtype=torch.int16),
        proposal_kind=torch.full((2,), PROPOSAL_NONE, dtype=torch.int8),
        replacement_observation_count=torch.tensor([3, 3]),
        replacement_overlap_ema=torch.full((2,), float(authority)),
        handoff_retired_fraction=torch.full((2,), float(authority)),
    )
    moment = torch.tensor([[-4.0], [-5.0]])
    optimizer = SimpleNamespace(state={opacity: {"exp_avg": moment}})
    return foliage, optimizer, moment


def test_disabled_static_replacement_does_not_report_or_apply_attenuation():
    foliage, optimizer, moment = _static_optical_policy_fixture(0.0)
    gradient_before = foliage.opacity_logits.grad.clone()
    moment_before = moment.clone()
    audit = _apply_static_optical_policy(
        foliage, optimizer, "topology"
    )
    assert audit["envelope_growth_rows_attenuated"] == 0
    assert torch.equal(foliage.opacity_logits.grad, gradient_before)
    assert torch.equal(moment, moment_before)


def test_local_static_replacement_attenuates_only_authorized_growth():
    foliage, optimizer, moment = _static_optical_policy_fixture(0.5)
    audit = _apply_static_optical_policy(
        foliage, optimizer, "topology"
    )
    expected_multiplier = 0.5
    assert audit["envelope_growth_rows_attenuated"] == 2
    assert torch.allclose(
        foliage.opacity_logits.grad,
        torch.tensor([[-2.0], [-3.0]]) * expected_multiplier,
    )
    assert torch.allclose(
        moment, torch.tensor([[-4.0], [-5.0]]) * expected_multiplier
    )


def test_unverified_child_growth_requires_exact_owner_ray_factor():
    foliage, optimizer, moment = _static_optical_policy_fixture(1.0)
    foliage.verification_state = torch.tensor(
        [VERIFICATION_UNVERIFIED, VERIFICATION_VERIFIED],
        dtype=torch.int8,
    )
    audit = _apply_static_optical_policy(
        foliage, optimizer, "topology"
    )
    assert foliage.opacity_logits.grad[0].item() == 0.0
    torch.testing.assert_close(
        foliage.opacity_logits.grad[1],
        torch.tensor([-0.15]),
    )
    assert audit["unverified_growth_rows_blocked"] == 1
    assert audit["unverified_growth_magnitude_blocked"] == pytest.approx(2.0)
    assert moment[0].item() == 0.0

    foliage.opacity_logits.grad = torch.tensor([[-2.0], [-3.0]])
    audit = _apply_static_optical_policy(
        foliage,
        optimizer,
        "topology",
        exact_owner_ray_opacity_gradient=torch.tensor([[-0.75], [0.0]]),
    )
    assert foliage.opacity_logits.grad[0].item() == pytest.approx(-0.75)
    assert audit["unverified_growth_magnitude_blocked"] == pytest.approx(1.25)


def test_detail_visibility_keeps_only_canonical_evidence_envelope_growth():
    foliage, optimizer, moment = _static_optical_policy_fixture(0.0)
    # Row zero has -0.75 canonical ray growth and -1.25 auxiliary growth;
    # only the former remains. Row one receives legitimate free-space cleanup.
    foliage.opacity_logits.grad[1] = 3.0
    moment[1] = 5.0
    canonical_evidence = torch.tensor([[-0.75], [0.5]])

    audit = _apply_static_optical_policy(
        foliage,
        optimizer,
        "static_foliage",
        canonical_evidence_opacity_gradient=canonical_evidence,
    )

    assert audit["envelope_positive_growth_attenuation"] == (
        "stage3_canonical_geometry_ray_plus_bounded_occlusion_deficit__"
        "ray_hit_opacity_is_phase_annealed_at_source__stage2_actual_"
        "local_handoff_retired_fraction"
    )
    assert audit["envelope_growth_rows_attenuated"] == 1
    assert foliage.opacity_logits.grad[0].item() == -0.75
    assert foliage.opacity_logits.grad[1].item() == 3.0
    assert moment[0].item() == -4.0
    assert moment[1].item() == 5.0


def test_detail_stage_preserves_source_separated_occlusion_growth_only():
    foliage, optimizer, moment = _static_optical_policy_fixture(0.0)
    foliage.opacity_logits.grad = torch.tensor([[-2.0], [3.0]])
    canonical_evidence = torch.tensor([[-0.75], [0.5]])
    canonical_occlusion = torch.tensor([[-0.50], [0.0]])

    audit = _apply_static_optical_policy(
        foliage,
        optimizer,
        "static_foliage",
        canonical_evidence_opacity_gradient=canonical_evidence,
        canonical_occlusion_opacity_gradient=canonical_occlusion,
    )

    # The remaining -0.75 on row zero came from unrestricted canopy RGB and
    # is removed. Ray evidence plus bounded coverage evidence survive.
    torch.testing.assert_close(
        foliage.opacity_logits.grad,
        torch.tensor([[-1.25], [3.0]]),
    )
    assert audit["canonical_occlusion_growth"][
        "persistent_envelope"
    ]["rows"] == 1
    assert audit["canonical_occlusion_growth"][
        "persistent_envelope"
    ]["magnitude"] == pytest.approx(0.5)
    # A currently authorized positive source keeps the retained Adam growth
    # moment alive; noncanonical iterations will clear it again.
    assert moment[0].item() == -4.0


def test_static_late_optical_creation_anneals_without_a_hard_gate():
    assert _static_ray_hit_opacity_gradient_scale("static", "topology") == 1.0
    assert _static_ray_hit_opacity_gradient_scale(
        "static", "dynamic_appearance"
    ) == 0.35
    assert _static_ray_hit_opacity_gradient_scale(
        "static", "ownership_cleanup"
    ) == 0.10
    assert _static_ray_hit_opacity_gradient_scale(
        "conditioned", "ownership_cleanup"
    ) == 1.0
    assert _static_detail_positive_opacity_growth_scale(
        "dynamic_appearance"
    ) == 1.0
    assert _static_detail_positive_opacity_growth_scale(
        "ownership_cleanup"
    ) == 1.0
    assert _static_ray_hit_opacity_gradient_scale(
        "static", "canonical_polish"
    ) == 0.10


def test_ownership_cleanup_keeps_single_source_detail_growth_and_cleanup():
    foliage, optimizer, _ = _static_optical_policy_fixture(0.0)
    foliage.persistent_envelope_mask.zero_()
    foliage.static_leaf_mask.fill_(True)
    foliage.opacity_logits.grad = torch.tensor([[-4.0], [3.0]])
    optimizer.state[foliage.opacity_logits]["exp_avg"] = torch.tensor(
        [[-8.0], [5.0]]
    )

    audit = _apply_static_optical_policy(
        foliage, optimizer, "ownership_cleanup"
    )

    torch.testing.assert_close(
        foliage.opacity_logits.grad, torch.tensor([[-4.0], [3.0]])
    )
    torch.testing.assert_close(
        optimizer.state[foliage.opacity_logits]["exp_avg"],
        torch.tensor([[-8.0], [5.0]]),
    )
    assert audit["static_detail_growth_rows_attenuated"] == 0


def test_canonical_polish_freezes_geometry_but_keeps_detail_dc_mass_calibration():
    foliage, optimizer, moment = _static_optical_policy_fixture(0.0)
    foliage.persistent_envelope_mask = torch.tensor([True, False])
    foliage.static_leaf_mask = torch.tensor([False, True])
    foliage.static_skeleton_mask = torch.zeros(2, dtype=torch.bool)
    foliage.dynamic_leaf_mask = torch.zeros(2, dtype=torch.bool)
    foliage.xyz.grad = torch.full_like(foliage.xyz, 2.0)
    foliage.log_scales.grad = torch.full_like(foliage.log_scales, 3.0)
    foliage.quaternions.grad = torch.full_like(foliage.quaternions, 4.0)

    audit = _apply_static_optical_policy(
        foliage, optimizer, "canonical_polish"
    )

    torch.testing.assert_close(foliage.xyz.grad[0], torch.zeros(3))
    torch.testing.assert_close(foliage.xyz.grad[1], torch.zeros(3))
    torch.testing.assert_close(
        foliage.opacity_logits.grad,
        torch.tensor([[0.0], [-3.0]]),
    )
    torch.testing.assert_close(moment, torch.tensor([[0.0], [-5.0]]))
    assert not audit["joint_polish_freeze"]
    assert audit["static_detail_mass_trainable"]
    assert audit["static_detail_ray_trainable"]


def test_canonical_polish_freezes_persistent_detail_high_order_sh_and_moments():
    foliage, optimizer, _ = _static_optical_policy_fixture(0.0)
    foliage.persistent_envelope_mask = torch.tensor([False, False])
    foliage.static_leaf_mask = torch.tensor([True, True])
    foliage.features = torch.nn.Parameter(torch.zeros(2, 3, 3))
    foliage.features.grad = torch.ones_like(foliage.features)
    first = torch.ones_like(foliage.features)
    second = torch.ones_like(foliage.features)
    optimizer.state[foliage.features] = {
        "exp_avg": first,
        "exp_avg_sq": second,
    }

    audit = _apply_static_optical_policy(
        foliage, optimizer, "canonical_polish"
    )

    torch.testing.assert_close(
        foliage.features.grad[:, 0], torch.ones(2, 3)
    )
    torch.testing.assert_close(
        foliage.features.grad[:, 1:], torch.zeros(2, 2, 3)
    )
    torch.testing.assert_close(first[:, 1:], torch.zeros(2, 2, 3))
    torch.testing.assert_close(second[:, 1:], torch.zeros(2, 2, 3))
    assert audit["persistent_detail_high_order_sh_rows_frozen"] == 2


def test_canonical_polish_injects_strict_envelope_cleanup_once_after_policy():
    foliage, optimizer, moment = _static_optical_policy_fixture(0.0)
    cleanup = torch.tensor([[0.75], [0.0]])

    _apply_static_optical_policy(foliage, optimizer, "canonical_polish")
    audit = _apply_strict_rigid_front_opacity_policy(
        foliage, optimizer, cleanup
    )

    torch.testing.assert_close(
        foliage.opacity_logits.grad, torch.tensor([[0.75], [0.0]])
    )
    torch.testing.assert_close(moment, torch.zeros_like(moment))
    assert audit["owner_rows"] == 1
    assert audit["source_magnitude"] == pytest.approx(0.75)


def test_noncanonical_tree_union_is_unknown_but_rigid_outside_survives():
    background = torch.ones(3, 3)
    canopy = torch.full((3, 3), 0.5)
    observed = torch.zeros(3, 3)
    observed[0, 0] = 1.0
    alpha = torch.zeros(1, 3, 3, requires_grad=True)
    with torch.no_grad():
        alpha[0, 2, 2] = 1.0e-5

    routed_background, routed_canopy, unknown, is_canonical = (
        _static_scene_rgb_ownership(
            background,
            canopy,
            observed,
            alpha,
            reconstruction_target="static",
            canonical_sequence_policy="scene",
            canonical_sequence="seq0",
            view_sequence="seq1",
            dilation_pixels=0,
        )
    )

    assert not is_canonical
    assert unknown[0, 0].item() == 1.0
    assert unknown[2, 2].item() == 1.0
    assert routed_background[1, 1].item() == 1.0
    assert routed_background[0, 0].item() == 0.0
    assert routed_background[2, 2].item() == 0.0
    torch.testing.assert_close(routed_canopy, torch.zeros_like(canopy))
    assert not unknown.requires_grad


def test_canonical_tree_rgb_weights_are_not_uncertainty_downweighted():
    background = torch.rand(2, 2)
    canopy = torch.rand(2, 2)
    routed_background, routed_canopy, unknown, is_canonical = (
        _static_scene_rgb_ownership(
            background,
            canopy,
            torch.ones(2, 2),
            torch.ones(1, 2, 2),
            reconstruction_target="static",
            canonical_sequence_policy="scene",
            canonical_sequence="seq0",
            view_sequence="seq0",
        )
    )
    assert is_canonical
    assert routed_background is background
    assert routed_canopy is canopy
    torch.testing.assert_close(unknown, torch.zeros_like(unknown))


def test_same_sequence_outside_anchor_frames_keeps_tree_rgb_authority():
    background = torch.ones(2, 2)
    canopy = torch.ones(2, 2)
    routed_background, routed_canopy, unknown, is_canonical = (
        _static_scene_rgb_ownership(
            background,
            canopy,
            torch.ones(2, 2),
            torch.zeros(1, 2, 2),
            reconstruction_target="static",
            canonical_sequence_policy="scene",
            canonical_sequence="seq0",
            view_sequence="seq0",
            view_is_canonical_snapshot=False,
            dilation_pixels=0,
        )
    )

    assert is_canonical
    torch.testing.assert_close(unknown, torch.zeros_like(unknown))
    assert routed_background is background
    assert routed_canopy is canopy


def test_detail_visibility_without_source_gradient_uses_safe_freeze():
    foliage, optimizer, moment = _static_optical_policy_fixture(0.0)

    _apply_static_optical_policy(
        foliage, optimizer, "static_foliage"
    )

    torch.testing.assert_close(
        foliage.opacity_logits.grad, torch.zeros(2, 1)
    )
    torch.testing.assert_close(moment, torch.zeros(2, 1))


def test_static_detail_stage_freezes_envelope_high_order_sh_only():
    foliage, optimizer, _ = _static_optical_policy_fixture(0.0)
    foliage.features = torch.nn.Parameter(torch.zeros(2, 3, 3))
    foliage.features.grad = torch.ones_like(foliage.features)
    feature_moment = torch.full_like(foliage.features, 2.0)
    optimizer.state[foliage.features] = {"exp_avg": feature_moment}

    audit = _apply_static_optical_policy(
        foliage, optimizer, "static_foliage"
    )

    assert audit["envelope_high_order_sh_rows_frozen"] == 2
    torch.testing.assert_close(
        foliage.features.grad[:, 0], torch.ones(2, 3)
    )
    torch.testing.assert_close(
        foliage.features.grad[:, 1:], torch.zeros(2, 2, 3)
    )
    torch.testing.assert_close(feature_moment[:, 0], torch.full((2, 3), 2.0))
    torch.testing.assert_close(feature_moment[:, 1:], torch.zeros(2, 2, 3))


def test_static_detail_visibility_is_evidence_tiered_end_to_end():
    class Foliage(SimpleNamespace):
        def __len__(self):
            return len(self.xyz)

    foliage = Foliage(
        xyz=torch.zeros(6, 3),
        dynamic_leaf_mask=torch.zeros(6, dtype=torch.bool),
        static_leaf_mask=torch.tensor([False, True, True, True, True, True]),
        verification_state=torch.tensor(
            [
                VERIFICATION_VERIFIED,
                VERIFICATION_VERIFIED,
                VERIFICATION_MEASURED_SINGLE,
                VERIFICATION_FACTORIZED_RECEIVER,
                VERIFICATION_UNVERIFIED,
                VERIFICATION_REJECTED,
            ],
            dtype=torch.int8,
        ),
        verified_camera_count=torch.tensor([2, 2, 1, 0, 0, 0]),
        verified_sequence_count=torch.tensor([1, 1, 1, 0, 0, 0]),
        proposal_kind=torch.tensor(
            [
                PROPOSAL_NONE,
                PROPOSAL_NONE,
                PROPOSAL_NONE,
                PROPOSAL_NONE,
                PROPOSAL_RAY_BIRTH,
                PROPOSAL_NONE,
            ],
            dtype=torch.int8,
        ),
        support_camera_ids=torch.tensor(
            [[-1], [1], [7], [7], [7], [7]], dtype=torch.int32
        ),
    )

    deployment = static_detail_forward_visibility_gate(
        foliage, 7, include_pending_exact=False
    )
    training = static_detail_forward_visibility_gate(
        foliage, 7, include_pending_exact=True
    )
    other_camera = static_detail_forward_visibility_gate(
        foliage, 8, include_pending_exact=False
    )

    torch.testing.assert_close(
        deployment, torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    )
    torch.testing.assert_close(
        training, torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    )
    torch.testing.assert_close(
        other_camera, torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    )


def test_static_detail_opacity_settle_closes_gradient_and_adam_growth():
    opacity = torch.nn.Parameter(torch.zeros(3, 1))
    opacity.grad = torch.tensor([[-1.0], [-2.0], [3.0]])
    first_moment = torch.tensor([[-1.0], [-4.0], [5.0]])
    foliage = SimpleNamespace(
        opacity_logits=opacity,
        static_leaf_mask=torch.tensor([False, True, True]),
    )
    optimizer = SimpleNamespace(
        state={opacity: {"exp_avg": first_moment}}
    )
    args = SimpleNamespace(
        volume_opacity_settle_policy="retirement_only",
        volume_opacity_settle_start_iteration=10,
        volume_opacity_retirement_until_iteration=20,
    )

    audit = _apply_static_detail_opacity_settle_policy(
        foliage, optimizer, step=10, args=args
    )
    torch.testing.assert_close(
        opacity.grad, torch.tensor([[-1.0], [0.0], [3.0]])
    )
    torch.testing.assert_close(
        first_moment, torch.tensor([[-1.0], [0.0], [5.0]])
    )
    assert audit["growth_rows_suppressed"] == 1

    opacity.grad = torch.tensor([[-1.0], [2.0], [-3.0]])
    first_moment.copy_(torch.tensor([[-1.0], [2.0], [-3.0]]))
    audit = _apply_static_detail_opacity_settle_policy(
        foliage, optimizer, step=20, args=args
    )
    torch.testing.assert_close(
        opacity.grad, torch.tensor([[-1.0], [0.0], [0.0]])
    )
    torch.testing.assert_close(
        first_moment, torch.tensor([[-1.0], [0.0], [0.0]])
    )
    assert audit["effective_policy"] == "freeze"


def _optical_trust_foliage():
    rows = 5
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(
        {
            "version": "independent_sfm_semantic_canopy_volume_v1",
            "centers": torch.stack(
                [torch.tensor([float(i), 0.0, 2.0]) for i in range(rows)]
            ),
            "scales": torch.full((rows, 3), 0.1),
            "colors": torch.full((rows, 3), 0.4),
            "opacities": torch.full((rows, 1), 0.2),
            "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(
                rows, 1
            ),
            "layer_role": torch.zeros(rows, dtype=torch.int8),
            "static_detail": torch.ones(rows, dtype=torch.bool),
            "verification_state": torch.tensor(
                [VERIFICATION_VERIFIED] * 4 + [VERIFICATION_MEASURED_SINGLE],
                dtype=torch.int8,
            ),
            "verified_camera_count": torch.tensor(
                [2, 2, 2, 2, 1], dtype=torch.int16
            ),
            "verified_sequence_count": torch.ones(rows, dtype=torch.int16),
            "proposal_kind": torch.full(
                (rows,), PROPOSAL_NONE, dtype=torch.int8
            ),
        }
    )
    return foliage


def test_canonical_completion_uses_exact_detail_or_envelope_not_both():
    rows = 4
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(
        {
            "version": "independent_sfm_semantic_canopy_volume_v1",
            "centers": torch.tensor(
                [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0],
                 [0.0, 0.0, 2.0], [2.0, 0.0, 2.0]]
            ),
            "scales": torch.full((rows, 3), 0.1),
            "colors": torch.full((rows, 3), 0.4),
            "opacities": torch.full((rows, 1), 0.2),
            "quaternions": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]]
            ).repeat(rows, 1),
            "layer_role": torch.zeros(rows, dtype=torch.int8),
            "static_detail": torch.tensor([False, False, True, True]),
            "replacement_group": torch.tensor([0, 1, 0, 1]),
            "support_camera_ids": torch.tensor(
                [[-1], [-1], [7], [8]], dtype=torch.int32
            ),
            "verification_state": torch.full(
                (rows,), VERIFICATION_VERIFIED, dtype=torch.int8
            ),
            "verified_camera_count": torch.full(
                (rows,), 2, dtype=torch.int16
            ),
            "verified_sequence_count": torch.ones(
                rows, dtype=torch.int16
            ),
            "proposal_kind": torch.full(
                (rows,), PROPOSAL_NONE, dtype=torch.int8
            ),
        }
    )
    owner = _canonical_occlusion_completion_owner_mask(
        foliage,
        camera_id=7,
        camera_sequence_lookup=torch.zeros(9, dtype=torch.int32),
    )

    # Exact detail row 2 replaces envelope row 0 in group 0. Envelope row 1
    # remains the fallback for group 1; non-owner detail row 3 stays frozen.
    torch.testing.assert_close(
        owner,
        torch.tensor([False, True, True, False]),
    )


def test_static_detail_optical_trust_routes_only_exact_sources_and_caps_each_event():
    foliage = _optical_trust_foliage()
    reference = foliage.integrated_optical_mass().detach().clone()
    foliage.opacity_settle_reference_mass.copy_(reference)
    foliage.opacity_logits.grad = torch.tensor(
        [[-3.0], [-4.0], [5.0], [-6.0], [-7.0]]
    )
    first_moment = torch.tensor([[-2.0], [-2.0], [2.0], [-2.0], [-2.0]])
    optimizer = SimpleNamespace(
        state={foliage.opacity_logits: {"exp_avg": first_moment}}
    )
    args = SimpleNamespace(
        volume_opacity_settle_policy="evidence_trust_region",
        volume_opacity_settle_start_iteration=10,
        volume_opacity_retirement_until_iteration=20,
        static_detail_optical_trust_lower=0.85,
        static_detail_optical_trust_upper=1.15,
        static_detail_optical_trust_voxel_size=0.10,
    )
    exact_growth = torch.zeros_like(foliage.opacity_logits)
    exact_retirement = torch.zeros_like(foliage.opacity_logits)
    exact_growth[1] = -0.5
    exact_retirement[2] = 0.7
    exact_growth[3] = -0.8
    exact_retirement[3] = 0.9
    exact_measured = torch.zeros_like(foliage.opacity_logits)
    exact_measured[4] = -0.4

    audit = _apply_static_detail_opacity_settle_policy(
        foliage,
        optimizer,
        step=10,
        args=args,
        exact_measured_optical_gradient=exact_measured,
        exact_optical_growth_gradient=exact_growth,
        exact_optical_retirement_gradient=exact_retirement,
    )
    torch.testing.assert_close(
        foliage.opacity_logits.grad,
        torch.tensor([[0.0], [-0.5], [0.7], [0.9], [-0.4]]),
    )
    torch.testing.assert_close(
        first_moment,
        torch.tensor([[0.0], [0.0], [2.0], [0.0], [0.0]]),
    )
    assert audit["exact_growth_rows"] == 2
    assert audit["exact_retirement_rows"] == 2
    assert audit["exact_conflict_rows_growth_vetoed"] == 1
    assert audit["ordinary_growth_rows_blocked"] >= 1

    pre_mass = foliage._opacity_settle_pre_mass.clone()
    area = projected_gaussian_cross_section(foliage.scales.detach())
    requested = pre_mass.clone()
    requested[0] *= 1.10  # no exact hit: no growth permission
    requested[1] *= 1.30  # exact hit: bounded to one 1.15x event
    alpha = -torch.expm1(-requested / area)
    with torch.no_grad():
        foliage.opacity_logits.copy_(torch.logit(alpha)[:, None])
    projection = _enforce_static_detail_optical_mass_trust_region(
        foliage, optimizer, args
    )
    realized = foliage.integrated_optical_mass().detach()
    torch.testing.assert_close(realized[0], pre_mass[0])
    torch.testing.assert_close(realized[1], pre_mass[1] * 1.15)
    assert projection["projected_up_rows"] == 0
    assert projection["projected_down_rows"] == 2
    assert projection["exact_growth_authorized_rows"] == 1
    assert foliage.opacity_settle_reference_mass[1] > reference[1]


def test_static_detail_optical_trust_allows_only_exact_measured_cleanup():
    foliage = _optical_trust_foliage()
    foliage.opacity_settle_reference_mass.copy_(
        foliage.integrated_optical_mass().detach()
    )
    foliage.opacity_logits.grad = torch.tensor(
        [[0.0], [0.0], [0.0], [0.0], [-3.0]]
    )
    first_moment = foliage.opacity_logits.grad.detach().clone()
    optimizer = SimpleNamespace(
        state={foliage.opacity_logits: {"exp_avg": first_moment}}
    )
    args = SimpleNamespace(
        volume_opacity_settle_policy="evidence_trust_region",
        volume_opacity_settle_start_iteration=10,
        volume_opacity_retirement_until_iteration=20,
        static_detail_optical_trust_lower=0.85,
        static_detail_optical_trust_upper=1.15,
        static_detail_optical_trust_voxel_size=0.10,
    )
    exact_cleanup = torch.zeros_like(foliage.opacity_logits)
    exact_cleanup[-1] = 2.0

    audit = _apply_static_detail_opacity_settle_policy(
        foliage,
        optimizer,
        step=10,
        args=args,
        exact_measured_cleanup_gradient=exact_cleanup,
    )

    assert float(foliage.opacity_logits.grad[-1]) == 2.0
    assert float(first_moment[-1]) == 0.0
    assert audit["measured_single_cleanup_retirement_rows"] == 1
    assert audit["frozen_rows"] == 0


def test_static_detail_optical_trust_allows_local_mass_redistribution():
    foliage = _optical_trust_foliage()
    # Rows 0/1 share one immutable 10cm cell. Row 0 is contradicted and may
    # fall below its own old 85% floor because row 1 preserves local coverage.
    foliage.initialization_center[1] = foliage.initialization_center[0]
    reference = foliage.integrated_optical_mass().detach().clone()
    foliage.opacity_settle_reference_mass.copy_(reference)
    area = projected_gaussian_cross_section(foliage.scales.detach())
    target = reference.clone()
    target[0] *= 0.70
    target[1] *= 1.10
    alpha = -torch.expm1(-target / area)
    with torch.no_grad():
        foliage.opacity_logits.copy_(torch.logit(alpha)[:, None])
    optimizer = SimpleNamespace(state={foliage.opacity_logits: {}})
    args = SimpleNamespace(
        static_detail_optical_trust_lower=0.85,
        static_detail_optical_trust_upper=1.15,
        static_detail_optical_trust_voxel_size=0.10,
    )

    foliage._opacity_settle_pre_mass = target.clone()
    foliage._opacity_settle_exact_growth_rows = torch.zeros(
        len(foliage), dtype=torch.bool
    )
    audit = _enforce_static_detail_optical_mass_trust_region(
        foliage, optimizer, args
    )
    realized = foliage.integrated_optical_mass().detach()
    torch.testing.assert_close(realized[:2], target[:2])
    assert realized[0] < 0.85 * reference[0]
    assert audit["rows_below_individual_lower_after"] == 1
    assert audit["group_count"] == 3


def test_static_detail_optical_trust_never_restores_underfilled_group():
    foliage = _optical_trust_foliage()
    foliage.initialization_center[1] = foliage.initialization_center[0]
    reference = foliage.integrated_optical_mass().detach().clone()
    foliage.opacity_settle_reference_mass.copy_(reference)
    area = projected_gaussian_cross_section(foliage.scales.detach())
    target = reference.clone()
    target[:2] *= 0.40
    alpha = -torch.expm1(-target / area)
    with torch.no_grad():
        foliage.opacity_logits.copy_(torch.logit(alpha)[:, None])
    optimizer = SimpleNamespace(state={foliage.opacity_logits: {}})
    args = SimpleNamespace(
        static_detail_optical_trust_lower=0.85,
        static_detail_optical_trust_upper=1.15,
        static_detail_optical_trust_voxel_size=0.10,
    )

    foliage._opacity_settle_pre_mass = target.clone()
    foliage._opacity_settle_exact_growth_rows = torch.zeros(
        len(foliage), dtype=torch.bool
    )
    audit = _enforce_static_detail_optical_mass_trust_region(
        foliage, optimizer, args
    )

    torch.testing.assert_close(
        foliage.integrated_optical_mass().detach()[:2], target[:2]
    )
    assert audit["groups_below_lower_before"] == 1
    assert audit["projected_up_rows"] == 0
    assert audit["mass_after"] <= audit["mass_before"]


def test_static_detail_optical_trust_shards_dense_cells_and_never_builds_giant_groups():
    foliage = VolumetricFoliageModel(1, device="cpu")
    rows = 70
    foliage.initialize_from_volume_state(
        {
            "version": "independent_sfm_semantic_canopy_volume_v1",
            "centers": torch.zeros(rows, 3),
            "scales": torch.full((rows, 3), 0.1),
            "colors": torch.full((rows, 3), 0.4),
            "opacities": torch.full((rows, 1), 0.2),
            "quaternions": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]]
            ).repeat(rows, 1),
            "layer_role": torch.zeros(rows, dtype=torch.int8),
            "static_detail": torch.ones(rows, dtype=torch.bool),
            "verification_state": torch.full(
                (rows,), VERIFICATION_VERIFIED, dtype=torch.int8
            ),
            "verified_camera_count": torch.full(
                (rows,), 2, dtype=torch.int16
            ),
            "verified_sequence_count": torch.ones(
                rows, dtype=torch.int16
            ),
            "proposal_kind": torch.full(
                (rows,), PROPOSAL_NONE, dtype=torch.int8
            ),
        }
    )
    owned = torch.ones(rows, dtype=torch.bool)
    _, groups, group_count = _static_detail_optical_trust_groups(
        foliage, owned, 0.10, maximum_group_size=32
    )
    counts = torch.bincount(groups, minlength=group_count)
    assert group_count == 3
    assert int(counts.max()) <= 32
    assert int(counts.min()) >= 2


def test_static_detail_sh_trust_is_persistent_only_and_adam_safe():
    features = torch.nn.Parameter(torch.zeros(3, 3, 3))
    with torch.no_grad():
        features[1, 1:] = 4.0
        features[2, 1:] = 3.0
    first_moment = torch.ones_like(features)
    second_moment = torch.ones_like(features)
    foliage = SimpleNamespace(
        features=features,
        static_leaf_mask=torch.tensor([False, True, True]),
        dynamic_leaf_mask=torch.zeros(3, dtype=torch.bool),
        verification_state=torch.tensor(
            [
                VERIFICATION_VERIFIED,
                VERIFICATION_VERIFIED,
                VERIFICATION_MEASURED_SINGLE,
            ],
            dtype=torch.int8,
        ),
        verified_camera_count=torch.tensor([2, 2, 1]),
        verified_sequence_count=torch.ones(3, dtype=torch.int16),
        proposal_kind=torch.full((3,), PROPOSAL_NONE, dtype=torch.int8),
    )
    optimizer = SimpleNamespace(
        state={
            features: {
                "exp_avg": first_moment,
                "exp_avg_sq": second_moment,
            }
        }
    )

    audit = _enforce_static_detail_sh_trust_region(
        foliage, optimizer, maximum_high_order_norm=1.0
    )

    assert float(features[1, 1:].flatten().norm()) == pytest.approx(1.0)
    torch.testing.assert_close(features[2, 1:], torch.zeros(2, 3))
    torch.testing.assert_close(first_moment[1:, 1:], torch.zeros(2, 2, 3))
    torch.testing.assert_close(second_moment[1:, 1:], torch.zeros(2, 2, 3))
    assert audit["projected_rows"] == 1
    assert audit["nonpersistent_rows_zeroed"] == 1


def test_oriented_multiscale_high_frequency_penalizes_blur_and_wrong_direction():
    target = torch.zeros(3, 16, 16)
    target[:, :, 8:] = 1.0
    perpendicular = torch.zeros_like(target)
    perpendicular[:, 8:, :] = 1.0
    blurred = torch.nn.functional.avg_pool2d(
        target[None], 5, stride=1, padding=2
    )[0]
    weight = torch.ones(16, 16)

    identical = _oriented_multiscale_high_frequency_loss(
        target, target, weight
    )
    blur_loss = _oriented_multiscale_high_frequency_loss(
        blurred, target, weight
    )
    direction_loss = _oriented_multiscale_high_frequency_loss(
        perpendicular, target, weight
    )
    masked = _oriented_multiscale_high_frequency_loss(
        perpendicular, target, torch.zeros_like(weight)
    )

    assert float(identical) == pytest.approx(0.0, abs=1.0e-7)
    assert float(blur_loss) > 0
    assert float(direction_loss) > float(blur_loss)
    assert float(masked) == pytest.approx(0.0, abs=1.0e-7)


def test_oriented_multiscale_high_frequency_has_finite_flat_pixel_backward():
    # The forward loss can be finite even when d sqrt(gx^2 + gy^2) is NaN at
    # gx=gy=0.  Exercise both a fully flat image and the flat regions around a
    # real edge, because the latter is the production foliage failure mode.
    weight = torch.ones(16, 16)
    flat = torch.full((3, 16, 16), 0.4, requires_grad=True)
    flat_target = torch.full_like(flat, 0.6)
    flat_loss = _oriented_multiscale_high_frequency_loss(
        flat, flat_target, weight
    )
    flat_loss.backward()
    assert torch.isfinite(flat_loss)
    assert flat.grad is not None
    assert torch.isfinite(flat.grad).all()

    edge = torch.zeros(3, 16, 16, requires_grad=True)
    edge_target = torch.zeros_like(edge)
    edge_target[:, :, 8:] = 1.0
    edge_loss = _oriented_multiscale_high_frequency_loss(
        edge, edge_target, weight
    )
    edge_loss.backward()
    assert torch.isfinite(edge_loss)
    assert edge.grad is not None
    assert torch.isfinite(edge.grad).all()


def test_trainer_repair_migrates_static_optical_lifecycle_contract():
    saved = {
        "reconstruction_target": "static",
        "static_optical_policy_contract": (
            STATIC_OPTICAL_POLICY_PREDECESSOR_CONTRACT
        ),
    }
    current = {
        "reconstruction_target": "static",
        "static_optical_policy_contract": STATIC_OPTICAL_POLICY_CONTRACT,
    }

    assert _resume_training_contract_differences(saved, current) == {
        "static_optical_policy_contract"
    }
    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_trainer_repair_migration=True,
    )

    stage3_freeze = {
        "reconstruction_target": "static",
        "static_optical_policy_contract": (
            STATIC_OPTICAL_POLICY_STAGE3_FREEZE_PREDECESSOR_CONTRACT
        ),
    }
    assert not _resume_training_contract_differences(
        stage3_freeze,
        current,
        allow_trainer_repair_migration=True,
    )


def test_resume_weight_ablation_changes_only_canonical_occlusion_weights():
    completion = {
        "contract": "bounded_scene_canonical_completion",
        "weight": 0.25,
        "maximum_optical_scale": 4.0,
        "gradient_owner": "volume_opacity_only",
    }
    order = {
        "contract": "bounded_scene_canonical_order",
        "weight": 0.0,
        "clearance": 0.03,
        "gradient_owner": "volume_xyz_only",
    }
    saved = {
        "reconstruction_target": "static",
        "canonical_occlusion_completion": completion,
        "canonical_occlusion_order": order,
    }
    current = {
        **saved,
        "canonical_occlusion_completion": {
            **completion,
            "weight": 1.0,
        },
        "canonical_occlusion_order": {
            **order,
            "weight": 0.1,
        },
    }

    assert _resume_training_contract_differences(saved, current) == {
        "canonical_occlusion_completion",
        "canonical_occlusion_order",
    }
    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_canonical_occlusion_weight_ablation=True,
    )

    changed_owner = {
        **current,
        "canonical_occlusion_completion": {
            **current["canonical_occlusion_completion"],
            "gradient_owner": "volume_and_surface_opacity",
        },
    }
    with pytest.raises(
        RuntimeError,
        match="may change only.*completion/order weights",
    ):
        _resume_training_contract_differences(
            saved,
            changed_owner,
            allow_canonical_occlusion_weight_ablation=True,
        )

    with pytest.raises(
        RuntimeError,
        match="finite non-negative completion/order weights",
    ):
        _resume_training_contract_differences(
            saved,
            {
                **current,
                "canonical_occlusion_completion": {
                    **current["canonical_occlusion_completion"],
                    "weight": float("nan"),
                },
            },
            allow_canonical_occlusion_weight_ablation=True,
        )

    changed_order_owner = {
        **current,
        "canonical_occlusion_order": {
            **current["canonical_occlusion_order"],
            "gradient_owner": "volume_and_surface_xyz",
        },
    }
    with pytest.raises(
        RuntimeError,
        match="may change only.*completion/order weights",
    ):
        _resume_training_contract_differences(
            saved,
            changed_order_owner,
            allow_canonical_occlusion_weight_ablation=True,
        )


def test_trainer_repair_refuses_legacy_visibility_floor_to_intrinsic_v95():
    saved_completion = {
        "contract": CANONICAL_OCCLUSION_DIRECTIONAL_COMPLETION_CONTRACT,
        "weight": 1.0,
        "maximum_optical_scale": 4.0,
        "advantage_margin": 0.005,
        "advantage_temperature": 0.02,
        "positive_pixel_authority": "scene_canonical_only",
        "gradient_owner": "volume_opacity_only",
        "stage3_envelope_permission": (
            "source_separated_bounded_coverage_gradient_only"
        ),
        "surface_in_front_policy": (
            "audit_geometry_order_conflict_without_opacity_gradient"
        ),
    }
    current_completion = {
        **saved_completion,
        "contract": CANONICAL_OCCLUSION_COMPLETION_CONTRACT,
        "surface_leakage_tolerance": 0.03,
        "start_iteration": 18_000,
        "activation_contract": CANONICAL_OCCLUSION_ACTIVATION_CONTRACT,
        "surface_in_front_policy": (
            "strictly_zero_completion_gradient__separate_order_loss_"
            "owns_local_geometry__topology_owns_missing_optical_existence"
        ),
        "volume_source": "surface_zero_intrinsic_envelope_only_layer",
        "forward_population_authority": (
            "resolved_verified_envelope_equals_live_opacity_owner"
        ),
    }
    saved = {"canonical_occlusion_completion": saved_completion}
    current = {"canonical_occlusion_completion": current_completion}

    assert _resume_training_contract_differences(saved, current) == {
        "canonical_occlusion_completion"
    }
    assert _resume_training_contract_differences(
        saved,
        current,
        allow_trainer_repair_migration=True,
    ) == {"canonical_occlusion_completion"}


def test_trainer_repair_adds_only_exact_post_topology_activation():
    current = {
        "canonical_occlusion_completion": {
            "contract": CANONICAL_OCCLUSION_COMPLETION_CONTRACT,
            "weight": 1.0,
            "surface_leakage_tolerance": 0.03,
            "start_iteration": 18_000,
            "activation_contract": CANONICAL_OCCLUSION_ACTIVATION_CONTRACT,
        },
        "canonical_occlusion_order": {
            "contract": CANONICAL_OCCLUSION_ORDER_CONTRACT,
            "weight": 0.1,
            "start_iteration": 18_000,
            "activation_contract": CANONICAL_OCCLUSION_ACTIVATION_CONTRACT,
        },
    }
    saved = {
        field: {
            key: value
            for key, value in loss.items()
            if key not in {"start_iteration", "activation_contract"}
        }
        for field, loss in current.items()
    }
    assert _resume_training_contract_differences(saved, current) == {
        "canonical_occlusion_completion",
        "canonical_occlusion_order",
    }
    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_trainer_repair_migration=True,
    )

    changed = {
        **current,
        "canonical_occlusion_order": {
            **current["canonical_occlusion_order"],
            "weight": 0.2,
        },
    }
    assert _resume_training_contract_differences(
        saved,
        changed,
        allow_trainer_repair_migration=True,
    ) == {"canonical_occlusion_order"}


def test_trainer_repair_migrates_only_exact_v86_optical_ownership_contract():
    saved = {
        "reconstruction_target": "static",
        "static_optical_policy_contract": (
            STATIC_OPTICAL_POLICY_V86_PREDECESSOR_CONTRACT
        ),
        "static_ray_local_mass_handoff": {
            "authority_contract": (
                "native_t_before_alpha_pixel_coverage_depth_rigid_safe_"
                "distinct_view_persistent"
            )
        },
        "static_training_stages": {"version": "v86"},
    }
    current = {
        "reconstruction_target": "static",
        "static_optical_policy_contract": STATIC_OPTICAL_POLICY_CONTRACT,
        "static_ray_local_mass_handoff": {
            "authority_contract": (
                "native_t_before_alpha_pixel_coverage_depth_rigid_safe_"
                "distinct_view_static_persistence_expected_coverage_ema"
            )
        },
        "static_ray_optical_ownership": {
            "contract": (
                "full_hit_geometry_derivative__phase_annealed_hit_"
                "opacity_derivative__full_free_space_derivative"
            )
        },
        "static_training_stages": {"version": "v87"},
    }
    assert _resume_training_contract_differences(saved, current)
    assert not _resume_training_contract_differences(
        saved, current, allow_trainer_repair_migration=True
    )
    unsafe = json.loads(json.dumps(saved))
    unsafe["static_ray_local_mass_handoff"]["authority_contract"] = "other"
    assert _resume_training_contract_differences(
        unsafe, current, allow_trainer_repair_migration=True
    )


def test_v103_migrates_only_local_optical_redistribution_contracts():
    old_settle_contract = (
        "static_persistent_detail_uses_immutable_evidence_mass_anchor_and_"
        "symmetric_bidirectional_adam_updates_inside_a_per_row_trust_region__"
        "outward_gradient_and_momentum_are_closed_and_post_adam_mass_is_hard_"
        "projected__nonpersistent_detail_is_frozen__legacy_conditioned_mode_"
        "may_retain_bounded_retirement_only"
    )
    saved = {
        "volume_opacity_settle": {
            "contract": old_settle_contract,
            "policy": "evidence_trust_region",
            "start_iteration": 18_000,
            "retirement_until_iteration": 25_500,
            "static_detail_optical_trust_lower": 0.85,
            "static_detail_optical_trust_upper": 1.15,
            "trainable_after_settle": "old",
        },
        "static_detail_global_cleanup": {
            "contract": STATIC_DETAIL_GLOBAL_CLEANUP_V102_PREDECESSOR_CONTRACT,
            "every": 2,
            "weight": 0.25,
            "counterfactual_weight": 1.0,
            "camera_schedule": "old",
        },
        "static_training_stages": {"joint_polish": "old"},
        "parameter_loss_permission_matrix": {
            "static_leaf_optical_mass": ["old"]
        },
        "static_optical_policy_contract": "old",
        "unrelated": "exact",
    }
    current = {
        **saved,
        "volume_opacity_settle": {
            **saved["volume_opacity_settle"],
            "contract": VOLUME_OPACITY_SETTLE_CONTRACT,
            "static_detail_optical_trust_voxel_size": 0.10,
            "trainable_after_settle": "new",
        },
        "static_detail_global_cleanup": {
            **saved["static_detail_global_cleanup"],
            "contract": STATIC_DETAIL_GLOBAL_CLEANUP_CONTRACT,
            "camera_schedule": "dedicated",
        },
        "static_training_stages": {"joint_polish": "new"},
        "parameter_loss_permission_matrix": {
            "static_leaf_optical_mass": ["new"]
        },
        "static_optical_policy_contract": STATIC_OPTICAL_POLICY_CONTRACT,
    }
    assert _resume_training_contract_differences(saved, current)
    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_v103_local_optical_redistribution_migration=True,
    )
    unsafe = json.loads(json.dumps(current))
    unsafe["unrelated"] = "changed"
    assert _resume_training_contract_differences(
        saved,
        unsafe,
        allow_v103_local_optical_redistribution_migration=True,
    ) == {"unrelated"}


def test_trainer_repair_migrates_only_exact_v87_handoff_schedule_contract():
    old_candidate = (
        "verified_same_group_positive_evidence_sequence__seed_camera_table_"
        "is_not_complete_visibility__primitive_center_cannot_veto_pixel_ray_"
        "evidence"
    )
    new_candidate = (
        "verified_same_group_real_support_or_verified_camera__only_groups_"
        "with_live_verified_envelope__primitive_center_cannot_veto_pixel_"
        "ray_evidence"
    )
    schedule_contract = (
        "uniform_complete_epochs_over_real_support_or_verified_cameras_for_"
        "verified_detail_with_live_verified_envelope_group"
    )
    saved = {
        "reconstruction_target": "static",
        "sampling_schedule_sha256": "old",
        "static_ray_local_mass_handoff": {
            "evidence_every": 50,
            "candidate_contract": old_candidate,
            "audit_camera_schedule": "broad_sequence_cycle",
        },
    }
    current = {
        "reconstruction_target": "static",
        "sampling_schedule_sha256": "new",
        "static_ray_local_mass_handoff": {
            "evidence_every": 10,
            "candidate_contract": new_candidate,
            "audit_camera_schedule": {"contract": schedule_contract},
        },
    }
    assert _resume_training_contract_differences(saved, current)
    assert not _resume_training_contract_differences(
        saved, current, allow_trainer_repair_migration=True
    )
    unsafe = json.loads(json.dumps(saved))
    unsafe["static_ray_local_mass_handoff"]["candidate_contract"] = "other"
    assert _resume_training_contract_differences(
        unsafe, current, allow_trainer_repair_migration=True
    )
    wrong_cadence = json.loads(json.dumps(current))
    wrong_cadence["static_ray_local_mass_handoff"]["evidence_every"] = 11
    assert _resume_training_contract_differences(
        saved, wrong_cadence, allow_trainer_repair_migration=True
    )


def test_trainer_repair_registers_mass_safe_detail_receiver_contract():
    current = {
        "static_detail_receiver_materialization": {
            "birth_forward_contract": (
                "co_located_same_covariance_same_color_tau_partition"
            ),
            "integrated_optical_mass_conserved": True,
            "existing_detail_group_repeat_forbidden": True,
        }
    }
    assert _resume_training_contract_differences({}, current) == {
        "static_detail_receiver_materialization"
    }
    assert not _resume_training_contract_differences(
        {}, current, allow_trainer_repair_migration=True
    )


def test_trainer_repair_ignores_only_static_fusion_reduction_roundoff():
    identity = {
        "contract": "static-fusion-v1",
        "input_dynamic_rows": 12,
        "associated_dynamic_rows": 10,
        "camera_sequence_metadata_source": "fixed_camera_contract",
        "camera_sequence_metadata_count": 4,
        "canonical_sequence_policy": "per_tree",
        "canonical_mode_voxel_size": 0.08,
        "maximum_modes_per_group": 2,
        "minimum_supporting_views": 2,
        "minimum_supporting_sequences": 2,
        "voxel_size": 0.15,
        "output_static_rows": 8,
    }
    saved = {
        "initialization_manifest_sha256": "manifest",
        "foliage_seed_sha256": "seed",
        "static_fusion": {
            **identity,
            "initial_mass_transferred_detail_mass": 0.1000000001,
        },
    }
    current = {
        "initialization_manifest_sha256": "manifest",
        "foliage_seed_sha256": "seed",
        "static_fusion": {
            **identity,
            "initial_mass_transferred_detail_mass": 0.1000000002,
        },
    }
    assert _resume_training_contract_differences(saved, current) == {
        "static_fusion"
    }
    assert not _resume_training_contract_differences(
        saved, current, allow_trainer_repair_migration=True
    )
    changed_identity = json.loads(json.dumps(current))
    changed_identity["static_fusion"]["output_static_rows"] = 9
    assert _resume_training_contract_differences(
        saved,
        changed_identity,
        allow_trainer_repair_migration=True,
    ) == {"static_fusion"}


def test_missing_detail_receiver_selection_is_one_per_unrepresented_group():
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(
        {
            "version": "independent_sfm_semantic_canopy_volume_v1",
            "centers": torch.tensor(
                [
                    [0.0, 0.0, 2.0],
                    [0.1, 0.0, 2.0],
                    [1.0, 0.0, 2.0],
                    [2.0, 0.0, 2.0],
                ]
            ),
            "scales": torch.full((4, 3), 0.1),
            "colors": torch.full((4, 3), 0.4),
            "opacities": torch.full((4, 1), 0.2),
            "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(
                4, 1
            ),
            "layer_role": torch.zeros(4, dtype=torch.int8),
            "static_detail": torch.tensor([False, False, False, True]),
            "replacement_group": torch.tensor([0, 1, 2, 2]),
            "tree_instance_id": torch.tensor([0, 0, 1, 2]),
            "support_camera_ids": torch.tensor(
                [[0, 1], [0, 1], [2, 3], [4, 5]], dtype=torch.int32
            ),
            "support_view_count": torch.full((4,), 2, dtype=torch.int16),
            "support_sequence_count": torch.full(
                (4,), 2, dtype=torch.int16
            ),
        }
    )
    # Runtime splits preserve replacement identity, so two live envelope rows
    # may address the same physical group even though the seed contract starts
    # with one contiguous owner per group.
    foliage.replacement_group[1] = 0
    stats = {
        "contribution": torch.ones(4),
        "gradient": torch.tensor([1.0, 4.0, 2.0, 100.0]),
        "gradient_count": torch.ones(4),
        "radius": torch.tensor([2.0, 3.0, 4.0, 20.0]),
        "rigid": torch.zeros(4),
    }
    selected, audit = _select_missing_static_detail_receiver_parents(
        foliage, stats, maximum_receivers=8
    )
    assert selected.tolist() == [1]
    assert audit["unique_candidate_groups"] == 1
    assert audit["selected"] == 1

    selected, audit = _select_missing_static_detail_receiver_parents(
        foliage,
        stats,
        maximum_receivers=8,
        eligible_camera_ids=[0],
    )
    assert selected.numel() == 0
    assert audit["unpromotable_support_rows_excluded"] == 2


def test_expired_factorized_receiver_restores_parent_mass_before_removal():
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
            "replacement_group": torch.tensor([0]),
            "support_camera_ids": torch.tensor([[7, 8]], dtype=torch.int32),
        }
    )
    mass_before = foliage.integrated_optical_mass().sum().clone()
    event = foliage.materialize_static_detail_receivers(
        torch.tensor([0]), birth_iteration=100
    )
    child = int(event["_new_start"])
    expired = torch.zeros(len(foliage), dtype=torch.bool)
    expired[child] = True
    remove = expired.clone()

    parent_rows, audit = _restore_expired_factorized_receiver_mass(
        foliage, expired, remove
    )
    foliage.prune(remove)

    assert parent_rows.tolist() == [0]
    assert audit["restored_parent_rows"] == 1
    assert audit["orphaned_rows"] == 0
    torch.testing.assert_close(
        foliage.integrated_optical_mass().sum(),
        mass_before,
        rtol=2e-5,
        atol=1e-8,
    )


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


def test_v86_optical_rebase_is_render_exact_and_clears_only_crown_moments():
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(
        {
            "version": "independent_sfm_semantic_canopy_volume_v1",
            "centers": torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]),
            "scales": torch.full((2, 3), 0.1),
            "colors": torch.full((2, 3), 0.4),
            "opacities": torch.tensor([[0.2], [0.1]]),
            "quaternions": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
            ),
            "layer_role": torch.zeros(2, dtype=torch.int8),
            "static_detail": torch.tensor([False, True]),
        }
    )
    foliage.handoff_retired_fraction[0] = 0.3
    foliage.replacement_overlap_ema[0] = 0.8
    foliage.replacement_observation_count[0] = 5
    foliage.replacement_camera_signature[0] = 7
    parameters_before = {
        name: value.detach().clone()
        for name, value in (
            ("xyz", foliage.xyz),
            ("scale", foliage.log_scales),
            ("rotation", foliage.quaternions),
            ("opacity", foliage.opacity_logits),
            ("features", foliage.features),
        )
    }
    mass_before = foliage.integrated_optical_mass().clone()
    opacity_moment = torch.ones_like(foliage.opacity_logits)
    optimizer = SimpleNamespace(
        state={foliage.opacity_logits: {"exp_avg": opacity_moment}},
        param_groups=[
            {"name": "opacity", "params": [foliage.opacity_logits]}
        ],
    )

    audit = _rebase_v86_static_optical_ownership_state(foliage, optimizer)

    torch.testing.assert_close(foliage.integrated_optical_mass(), mass_before)
    for name, value in parameters_before.items():
        current = {
            "xyz": foliage.xyz,
            "scale": foliage.log_scales,
            "rotation": foliage.quaternions,
            "opacity": foliage.opacity_logits,
            "features": foliage.features,
        }[name]
        assert torch.equal(current, value)
    assert foliage.handoff_retired_fraction[0] == 0
    assert foliage.replacement_overlap_ema[0] == 0
    assert foliage.replacement_observation_count[0] == 0
    assert foliage.replacement_camera_signature[0] == 0
    torch.testing.assert_close(
        foliage.handoff_reference_mass[0], mass_before[0]
    )
    torch.testing.assert_close(opacity_moment, torch.zeros_like(opacity_moment))
    assert audit["realized_mass_change"] == 0.0


def test_runtime_ray_birth_initial_mass_is_ray_local_conserved_and_reversible():
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
            "tree_instance_id": torch.tensor([3], dtype=torch.int32),
            "support_camera_ids": torch.tensor(
                [[0, 1]], dtype=torch.int32
            ),
            "support_view_count": torch.tensor([2], dtype=torch.int16),
            "support_sequence_count": torch.tensor([2], dtype=torch.int16),
        }
    )
    pre_birth_mass = foliage.integrated_optical_mass().sum().clone()
    appended = foliage.append_static_ray_births(
        torch.tensor([[0.0, 0.0, 2.0]]),
        support_camera_ids=torch.tensor([[7, 8]], dtype=torch.int32),
        support_sequence_count=torch.tensor([2], dtype=torch.int16),
        birth_iteration=100,
    )
    birth_start = int(appended["_new_start"])
    replacement_rows = appended["_replacement_rows"]
    owner_rows = appended["_replacement_owner_rows"]
    assert replacement_rows.tolist() == [birth_start]
    assert owner_rows.tolist() == [0]

    identity_view = SimpleNamespace(
        world_view_transform=torch.eye(4),
        focal_x=100.0,
        focal_y=100.0,
        cx=50.0,
        cy=50.0,
    )
    optimizer = SimpleNamespace(state={})
    audit = _initialize_static_ray_birth_mass_handoff(
        foliage,
        optimizer,
        birth_rows=torch.tensor([birth_start]),
        owner_rows=owner_rows,
        view_by_camera_id={7: identity_view, 8: identity_view},
        maximum_fraction_per_event=0.02,
        maximum_additive_fraction_per_event=0.0,
    )

    realized = foliage.integrated_optical_mass()
    assert audit["ray_local_candidates"] == 1
    assert audit["changed_owner_rows"] == 1
    assert realized[birth_start] > 0
    assert realized[0] < pre_birth_mass
    # The newborn is funded entirely by its owner: no new integrated
    # extinction is added at the topology event.
    torch.testing.assert_close(
        realized.sum(), pre_birth_mass, rtol=1e-5, atol=1e-8
    )

    # Simulate failed-child retirement. Once the child disappears, the normal
    # reversible handoff restores exactly the donor mass rather than leaving a
    # background hole.
    failed = realized.clone()
    failed[birth_start] = 0
    foliage.restore_integrated_optical_mass(failed, minimum_opacity=1.0e-12)
    foliage.replacement_overlap_ema.zero_()
    foliage.replacement_observation_count.zero_()
    restored = _apply_static_ray_local_mass_handoff(
        foliage, optimizer, maximum_fraction_per_event=0.02
    )
    assert restored["restored_mass"] > 0
    torch.testing.assert_close(
        foliage.integrated_optical_mass()[0],
        pre_birth_mass,
        rtol=1e-5,
        atol=1e-8,
    )


def test_ownerless_strict_visual_hull_birth_gets_bounded_additive_mass():
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
            "support_camera_ids": torch.tensor(
                [[-1, -1]], dtype=torch.int32
            ),
            "tree_instance_id": torch.tensor([3], dtype=torch.int32),
        }
    )
    before = foliage.integrated_optical_mass().sum().clone()
    appended = foliage.append_static_ray_births(
        torch.tensor([[3.0, 0.0, 2.0]]),
        support_camera_ids=torch.tensor([[7, 8]], dtype=torch.int32),
        support_sequence_count=torch.tensor([2], dtype=torch.int16),
    )
    birth = int(appended["_new_start"])
    audit = _initialize_static_ray_birth_mass_handoff(
        foliage,
        SimpleNamespace(state={}),
        birth_rows=torch.tensor([birth]),
        owner_rows=torch.tensor([-1]),
        view_by_camera_id={},
        maximum_fraction_per_event=0.02,
    )
    assert audit["ray_local_candidates"] == 0
    after = foliage.integrated_optical_mass().sum()
    assert audit["additive_funded_children"] == 1
    assert 0.0 < audit["additive_child_mass"] <= float(before * 0.005)
    assert before < after <= before * 1.005 + 1.0e-8


@pytest.mark.parametrize("canonical,camera_names,support,expected", [
    (None, ["seq2__frame00001", "seq2__frame00002"], [7, 8], 0),
    ("seq2", ["seq2__frame00001", "seq2__frame00002"], [7, 8], 1),
    ("seq2", ["seq2__frame00001", "seq1__frame00002"], [7, 8], 0),
    ("seq2", ["seq2__frame00001", "seq2__frame00002"], [7, 7], 0),
    ("seq2", ["seq2__frame00001", "seq2__frame00002"], [7, 9], 0),
])
def test_ownerless_single_sequence_birth_requires_exact_canonical_consensus(
    canonical, camera_names, support, expected,
):
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
            "support_camera_ids": torch.tensor(
                [[-1, -1]], dtype=torch.int32
            ),
        }
    )
    before = foliage.integrated_optical_mass().sum().clone()
    appended = foliage.append_static_ray_births(
        torch.tensor([[3.0, 0.0, 2.0]]),
        support_camera_ids=torch.tensor([support], dtype=torch.int32),
        support_sequence_count=torch.tensor([1], dtype=torch.int16),
    )
    birth = int(appended["_new_start"])
    audit = _initialize_static_ray_birth_mass_handoff(
        foliage,
        SimpleNamespace(state={}),
        birth_rows=torch.tensor([birth]),
        owner_rows=torch.tensor([-1]),
        view_by_camera_id={
            7: SimpleNamespace(image_name=camera_names[0]),
            8: SimpleNamespace(image_name=camera_names[1]),
        },
        maximum_fraction_per_event=0.02,
        canonical_sequence=canonical,
    )
    assert audit["additive_funded_children"] == expected
    after = foliage.integrated_optical_mass().sum()
    if expected:
        assert 0 < audit["additive_child_mass"] <= float(before * .005)
        assert before < after <= before * 1.005 + 1e-8
    else:
        torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-8)


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


def test_unresolved_split_keeps_parent_handoff_state_atomically_frozen():
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
    optimizer = SimpleNamespace(state={})
    foliage.replacement_overlap_ema.fill_(1.0)
    foliage.replacement_observation_count.fill_(3)
    _apply_static_ray_local_mass_handoff(
        foliage, optimizer, maximum_fraction_per_event=0.10
    )
    mass = foliage.integrated_optical_mass().clone()
    retired = foliage.handoff_retired_fraction.clone()
    foliage.proposal_kind.fill_(PROPOSAL_SPLIT)
    foliage.split_proposal_family_id.fill_(7)
    foliage.replacement_overlap_ema.zero_()

    audit = _apply_static_ray_local_mass_handoff(
        foliage, optimizer, maximum_fraction_per_event=0.10
    )

    assert audit["changed_rows"] == 0
    assert audit["repaired_ineligible_rows"] == 0
    torch.testing.assert_close(foliage.integrated_optical_mass(), mass)
    torch.testing.assert_close(foliage.handoff_retired_fraction, retired)


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


def test_replacement_evidence_ignores_unrelated_visible_views():
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
    visible = torch.ones(1, dtype=torch.bool)
    candidate = torch.ones(1, dtype=torch.bool)
    _accumulate_static_replacement_evidence(
        foliage,
        torch.tensor([0.8]),
        visible,
        candidate_visible=candidate,
        camera_id=7,
        decay=0.95,
    )
    torch.testing.assert_close(
        foliage.replacement_overlap_ema, torch.tensor([0.8])
    )
    assert foliage.replacement_observation_count.item() == 1

    # Merely seeing the envelope in an unrelated camera is neither a
    # positive witness nor a contradiction of its local detail replacement.
    audit = _accumulate_static_replacement_evidence(
        foliage,
        torch.zeros(1),
        visible,
        candidate_visible=torch.zeros(1, dtype=torch.bool),
        camera_id=8,
        decay=0.95,
    )
    torch.testing.assert_close(
        foliage.replacement_overlap_ema, torch.tensor([0.8])
    )
    assert foliage.replacement_observation_count.item() == 1
    assert audit["contradicted_candidate_rows"] == 0

    # The same local group/depth candidate with zero real pixel authority is
    # explicit counter-evidence, with the same EMA semantics as a positive
    # observation rather than a one-way seasonal minimum.
    audit = _accumulate_static_replacement_evidence(
        foliage,
        torch.zeros(1),
        visible,
        candidate_visible=candidate,
        camera_id=8,
        decay=0.95,
    )
    torch.testing.assert_close(
        foliage.replacement_overlap_ema, torch.tensor([0.76])
    )
    assert foliage.replacement_observation_count.item() == 2
    assert audit["contradicted_candidate_rows"] == 1


def test_replacement_evidence_tracks_expected_persistent_support():
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
    visible = torch.ones(1, dtype=torch.bool)
    for camera_id, authority in ((1, 0.9), (2, 0.1), (3, 0.9)):
        _accumulate_static_replacement_evidence(
            foliage,
            torch.tensor([authority]),
            visible,
            camera_id=camera_id,
            decay=0.95,
        )
    # Positive and negative support views have symmetric temporal semantics.
    # This estimates persistence of a static representation instead of the
    # least leafy seasonal snapshot.
    torch.testing.assert_close(
        foliage.replacement_overlap_ema, torch.tensor([0.862])
    )
    assert foliage.replacement_observation_count.item() == 3


def test_post_step_handoff_preserves_new_unretired_mass():
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
    optimizer = SimpleNamespace(state={})
    initial = foliage.integrated_optical_mass().clone()
    foliage.replacement_overlap_ema.fill_(1.0)
    foliage.replacement_observation_count.fill_(3)
    _apply_static_ray_local_mass_handoff(
        foliage, optimizer, maximum_fraction_per_event=0.10
    )
    previous_fraction = foliage.handoff_retired_fraction.clone()

    # Simulate a legitimate optimizer/ray-hit mass increase before the next
    # local replacement event.  The post-step finalizer must fold this into
    # the reference instead of snapping back to the stale initial mass.
    grown = foliage.integrated_optical_mass() * 1.25
    foliage.restore_integrated_optical_mass(grown)
    audit = _finalize_static_ray_local_mass_handoff(
        foliage,
        optimizer,
        scheduled=True,
        maximum_fraction_per_event=0.02,
    )
    expected = grown * (
        (1.0 - foliage.handoff_retired_fraction)
        / (1.0 - previous_fraction)
    )
    torch.testing.assert_close(
        foliage.integrated_optical_mass(), expected, rtol=1e-5, atol=1e-8
    )
    assert audit["scheduled_after_optimizer_step"] is True
    assert foliage.integrated_optical_mass().item() > initial.item() * 0.9


def test_post_step_handoff_is_frozen_during_canonical_polish():
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
    foliage.replacement_overlap_ema.fill_(1.0)
    foliage.replacement_observation_count.fill_(3)
    optimizer = SimpleNamespace(state={})
    mass_before = foliage.integrated_optical_mass().clone()

    audit = _finalize_static_ray_local_mass_handoff(
        foliage,
        optimizer,
        scheduled=True,
        maximum_fraction_per_event=0.10,
        allow_mass_transfer=False,
    )

    torch.testing.assert_close(
        foliage.integrated_optical_mass(), mass_before
    )
    assert audit["mass_transfer_frozen"] is True
    assert audit["changed_rows"] == 0


def test_occlusion_order_gradient_responsibility_is_bounded_per_pixel():
    gradient = torch.arange(1, 31, dtype=torch.float32).reshape(10, 3)
    capped, audit = _cap_canonical_occlusion_order_gradient_responsibility(
        gradient,
        supported_pixels=1,
        maximum_rows_per_pixel=3,
    )

    assert int((capped.norm(dim=1) > 0).sum()) == 3
    assert audit["gradient_rows_before_responsibility_cap"] == 10
    assert audit["maximum_gradient_rows"] == 3
    assert audit["responsibility_cap_rows_blocked"] == 7
    # The helper may only remove complete row gradients, never alter a kept
    # vector or synthesize a new one.
    kept = capped.norm(dim=1) > 0
    torch.testing.assert_close(capped[kept], gradient[kept])


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


def _v114_optical_ownership_contract_pair():
    current = {
        "static_detail_global_cleanup": {
            "contract": STATIC_DETAIL_GLOBAL_CLEANUP_CONTRACT,
            "canonical_polish_weight_scale": 1.0,
            "forward_roles": [
                "surface_zero_exact_prehit_live_static_detail_owners"
            ],
            "optimized_quantity": "exact_prehit_optical_depth",
            "rigid_depth_permission": (
                "native_volume_prehit_before_detached_surface_median_z_"
                "minus_3cm"
            ),
            "behind_surface_policy": (
                "exact_zero_query_membership_and_gradient"
            ),
            "camera_schedule": {
                "contract": (
                    "uniform_complete_epochs_over_all_scene_canonical_"
                    "cameras__strict_negative_rigid_and_sky_evidence_only__"
                    "positive_detail_growth_remains_exact_support_camera_"
                    "owned"
                ),
                "canonical_sequence": "seq2",
                "camera_count": 352,
                "positive_exact_support_camera_count": 395,
                "schedule_sha256": "fixed-canonical-cleanup-schedule",
            },
        },
        "persistent_envelope_global_cleanup": {
            "contract": PERSISTENT_ENVELOPE_GLOBAL_CLEANUP_CONTRACT,
            "forward_roles": [
                "surface_zero_exact_prehit_resolved_persistent_envelope"
            ],
            "rigid_depth_permission": (
                "native_volume_prehit_before_detached_surface_median_z_"
                "minus_3cm"
            ),
            "behind_surface_policy": (
                "exact_zero_query_membership_and_gradient"
            ),
            "optimized_quantity": "exact_prehit_optical_depth",
        },
        "volume_opacity_settle": {
            "contract": VOLUME_OPACITY_SETTLE_CONTRACT,
            "trainable_after_settle": (
                "persistent_static_detail_exact_ray_and_moge3_hit_growth_"
                "only__exact_ray_moge3_and_rigid_prehit_retirement__ordinary_"
                "rgb_opacity_frozen__per_exact_event_growth_cap_without_post_"
                "adam_restoration__geometry_scale_rotation_sh_and_dynamic_"
                "deformation_feature"
            ),
        },
        "moge3_canopy_optical": {
            "contract": MOGE3_CANOPY_OPTICAL_CONTRACT,
        },
        "mature_handoff_surface_policy": "atlas_residual",
        "mature_handoff_surface_partition": {
            "rigid_prefix_rows": 1_135_055,
            "evidence_completion_suffix_rows": 20_019,
            "chart_atlas_learning_rate": 2e-4,
            "contract": (
                "mature_rigid_prefix_fixed__chart_inverse_depth_low_lr__"
                "evidence_completion_suffix_geometry_opacity_trainable"
            ),
        },
        "ray_evidence_epoch_capacity": _ray_epoch_capacity_audit(
            prior=125, remaining=2_933_149
        ),
        "cpu_parallelism": {"resolved_intraop_threads": 3},
    }
    saved = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in current.items()
    }
    saved["static_detail_global_cleanup"].update(
        {
            "contract": (
                STATIC_DETAIL_GLOBAL_CLEANUP_V113_PREDECESSOR_CONTRACT
            ),
            "canonical_polish_weight_scale": 0.10,
            "forward_roles": [
                "native_mixed_surface_plus_live_static_detail_owners"
            ],
            "optimized_quantity": (
                "signed_live_mixed_rgb_opacity_derivative"
            ),
            "rigid_depth_permission": (
                "exact_native_per_pixel_depth_sorted_volume_contribution"
            ),
            "behind_surface_policy": (
                "zero_or_surface_attenuated_native_volume_contribution"
            ),
        }
    )
    saved["static_detail_global_cleanup"]["camera_schedule"] = {
        **saved["static_detail_global_cleanup"]["camera_schedule"],
        "positive_exact_support_camera_count": 125,
    }
    saved["persistent_envelope_global_cleanup"].update(
        {
            "contract": (
                PERSISTENT_ENVELOPE_GLOBAL_CLEANUP_V113_PREDECESSOR_CONTRACT
            ),
            "forward_roles": [
                "native_mixed_surface_plus_resolved_persistent_envelope"
            ],
            "rigid_depth_permission": (
                "exact_native_per_pixel_sorted_envelope_contribution"
            ),
            "behind_surface_policy": (
                "zero_or_surface_attenuated_native_volume_contribution"
            ),
            "optimized_quantity": (
                "signed_live_mixed_rgb_opacity_derivative"
            ),
        }
    )
    saved["volume_opacity_settle"].update(
        {
            "contract": VOLUME_OPACITY_SETTLE_V113_PREDECESSOR_CONTRACT,
            "trainable_after_settle": (
                "persistent_static_detail_bidirectional_optical_mass_inside_"
                "immutable_per_row_lower_and_upper_source_barriers_without_"
                "post_adam_restoration__geometry_scale_rotation_sh_and_"
                "dynamic_deformation_feature"
            ),
        }
    )
    saved["moge3_canopy_optical"]["contract"] = (
        MOGE3_CANOPY_OPTICAL_V113_PREDECESSOR_CONTRACT
    )
    saved["mature_handoff_surface_partition"] = {
        **saved["mature_handoff_surface_partition"],
        "evidence_completion_suffix_rows": 20_000,
    }
    saved["ray_evidence_epoch_capacity"] = _ray_epoch_capacity_audit(
        prior=0, remaining=3_079_774
    )
    saved["cpu_parallelism"] = {"resolved_intraop_threads": 8}
    return saved, current


def test_v114_migration_normalizes_only_legal_runtime_resume_state():
    saved, current = _v114_optical_ownership_contract_pair()

    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_v114_optical_ownership_migration=True,
    )


def test_v114_migration_rejects_immutable_partition_or_ray_changes():
    saved, current = _v114_optical_ownership_contract_pair()
    changed_partition = {
        **current,
        "mature_handoff_surface_partition": {
            **current["mature_handoff_surface_partition"],
            "rigid_prefix_rows": 1_135_054,
        },
    }
    with pytest.raises(RuntimeError, match="exactly the four"):
        _resume_training_contract_differences(
            saved,
            changed_partition,
            allow_v114_optical_ownership_migration=True,
        )

    changed_ray = {
        **current,
        "ray_evidence_epoch_capacity": {
            **current["ray_evidence_epoch_capacity"],
            "full_effective_row_count": 3_079_775,
        },
    }
    with pytest.raises(RuntimeError, match="exactly the four"):
        _resume_training_contract_differences(
            saved,
            changed_ray,
            allow_v114_optical_ownership_migration=True,
        )

    changed_cleanup_schedule = {
        **current,
        "static_detail_global_cleanup": {
            **current["static_detail_global_cleanup"],
            "camera_schedule": {
                **current["static_detail_global_cleanup"]["camera_schedule"],
                "schedule_sha256": "different-cleanup-schedule",
            },
        },
    }
    with pytest.raises(
        RuntimeError, match="static_detail_global_cleanup fields"
    ):
        _resume_training_contract_differences(
            saved,
            changed_cleanup_schedule,
            allow_v114_optical_ownership_migration=True,
        )


def _v115_persistent_debt_contract_pair():
    _, saved = _v114_optical_ownership_contract_pair()
    saved = copy.deepcopy(saved)
    current = copy.deepcopy(saved)
    current["cpu_parallelism"] = {"resolved_intraop_threads": 6}
    current["ray_evidence_epoch_capacity"] = _ray_epoch_capacity_audit(
        prior=250, remaining=2_786_524
    )
    current["mature_handoff_surface_partition"][
        "evidence_completion_suffix_rows"
    ] = 20_037
    current["static_detail_global_cleanup"]["camera_schedule"][
        "positive_exact_support_camera_count"
    ] = 410
    current["exact_ray_optical_source_routing"] = {
        "contract": (
            "weighted_confirmed_free_is_separate_from_hit_likelihood__"
            "hit_prehit_tail_and_hit_existence_share_one_likelihood__"
            "independent_same_row_free_contradiction_vetoes_hit_growth__"
            "retirement_priority__envelope_only_source_extraction_active"
        ),
        "weight": 0.06,
        "growth_owner": "conflict_resolved_hit_likelihood_only",
        "retirement_owner": (
            "confirmed_free_plus_positive_hit_likelihood_derivative"
        ),
    }
    current["persistent_rigid_front_conflict_debt"] = {
        "contract": (
            "exact_rigid_front_contradiction_is_checkpointed_in_volume_"
            "stats__debt_is_monotone_until_representation_localization__"
            "completion_rgb_exact_hit_and_adam_growth_are_vetoed__all_"
            "retirement_and_post_step_handoff_monotonicity_remain_live"
        ),
        "state": "int16_per_primitive_observation_count",
        "checkpoint_field": (
            "volume_stats.rigid_front_conflict_observations"
        ),
        "topology_scope": "starts_after_volume_topology_settlement",
    }
    return saved, current


def test_v115_migration_adds_only_exact_sources_and_persistent_debt():
    saved, current = _v115_persistent_debt_contract_pair()
    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_v115_persistent_ownership_debt_migration=True,
    )

    changed = copy.deepcopy(current)
    changed["persistent_envelope_global_cleanup"]["optimized_quantity"] = (
        "different"
    )
    with pytest.raises(RuntimeError, match="exactly the two"):
        _resume_training_contract_differences(
            saved,
            changed,
            allow_v115_persistent_ownership_debt_migration=True,
        )

    changed = copy.deepcopy(current)
    changed["exact_ray_optical_source_routing"]["weight"] = 0.12
    with pytest.raises(RuntimeError, match="exact-ray routing target"):
        _resume_training_contract_differences(
            saved,
            changed,
            allow_v115_persistent_ownership_debt_migration=True,
        )


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


def test_resume_ray_capacity_accepts_completed_nonempty_epoch():
    saved_audit = _ray_epoch_capacity_audit(
        prior=0, remaining=3_079_774
    )
    completed_audit = {
        **_ray_epoch_capacity_audit(prior=750, remaining=0),
        "runtime_aggregate_minimum_batch_with_five_percent_margin": 0,
        "minimum_batch_for_runtime_completion": 0,
        "required_factor_calls": 0,
        "unused_factor_calls": 2250,
    }
    saved = {
        "ray_posterior_maximum_rays": 1173,
        "ray_evidence_epoch_capacity": saved_audit,
    }
    current = {
        "ray_posterior_maximum_rays": 1173,
        "ray_evidence_epoch_capacity": completed_audit,
    }

    assert not _resume_training_contract_differences(saved, current)


def test_joint_surface_partition_row_count_is_runtime_resume_state():
    saved = {
        "mature_handoff_surface_policy": "joint",
        "mature_handoff_surface_partition": {
            "rigid_prefix_rows": 670_645,
            "completion_suffix_rows": 0,
            "chart_atlas_learning_rate": 0.002,
            "contract": "policy_specific_legacy_surface_ownership",
        },
    }
    current = {
        **saved,
        "mature_handoff_surface_partition": {
            **saved["mature_handoff_surface_partition"],
            "rigid_prefix_rows": 665_467,
        },
    }
    assert not _resume_training_contract_differences(saved, current)


def test_nonjoint_surface_partition_remains_immutable_on_resume():
    saved = {
        "mature_handoff_surface_policy": "atlas_residual",
        "mature_handoff_surface_partition": {
            "rigid_prefix_rows": 670_645,
            "completion_suffix_rows": 20_000,
        },
    }
    current = {
        **saved,
        "mature_handoff_surface_partition": {
            **saved["mature_handoff_surface_partition"],
            "rigid_prefix_rows": 665_467,
        },
    }
    assert _resume_training_contract_differences(saved, current) == {
        "mature_handoff_surface_partition"
    }


def test_atlas_surface_suffix_cardinality_is_runtime_resume_state():
    saved = {
        "mature_handoff_surface_policy": "atlas_residual",
        "mature_handoff_surface_partition": {
            "rigid_prefix_rows": 1_135_055,
            "evidence_completion_suffix_rows": 20_000,
            "chart_atlas_learning_rate": 2e-4,
            "contract": (
                "mature_rigid_prefix_fixed__chart_inverse_depth_low_lr__"
                "evidence_completion_suffix_geometry_opacity_trainable"
            ),
        },
    }
    current = {
        **saved,
        "mature_handoff_surface_partition": {
            **saved["mature_handoff_surface_partition"],
            # Nineteen legal topology children were appended before the
            # retained checkpoint. The ownership boundary did not move.
            "evidence_completion_suffix_rows": 20_019,
        },
    }
    assert not _resume_training_contract_differences(saved, current)

    changed_boundary = {
        **current,
        "mature_handoff_surface_partition": {
            **current["mature_handoff_surface_partition"],
            "rigid_prefix_rows": 1_135_054,
        },
    }
    assert _resume_training_contract_differences(
        saved, changed_boundary
    ) == {"mature_handoff_surface_partition"}


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


def test_static_detail_gradients_are_owned_by_exact_support_cameras():
    class Foliage:
        xyz = torch.zeros(4, 3)
        dynamic_leaf_mask = torch.zeros(4, dtype=torch.bool)
        static_leaf_mask = torch.tensor([False, True, True, True])
        support_camera_ids = torch.tensor(
            [[-1, -1], [0, -1], [1, -1], [0, 1]], dtype=torch.int32
        )

        def __len__(self):
            return 4

    # Both cameras belong to one sequence; only exact camera zero may own the
    # update, proving sequence membership is not an appearance permission.
    lookup = torch.tensor([0, 0], dtype=torch.int16)
    gate = _static_detail_canonical_ownership_gate(Foliage(), 0, lookup)
    assert torch.equal(gate, torch.tensor([1.0, 1.0, 0.0, 1.0]))
    missing = _static_detail_canonical_ownership_gate(
        Foliage(), 4, lookup
    )
    assert torch.equal(missing, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_static_detail_appearance_softly_uses_same_sequence_cameras():
    class Foliage:
        xyz = torch.zeros(5, 3)
        static_leaf_mask = torch.tensor([False, True, True, True, True])
        dynamic_leaf_mask = torch.zeros(5, dtype=torch.bool)
        verification_state = torch.tensor(
            [
                VERIFICATION_VERIFIED,
                VERIFICATION_MEASURED_SINGLE,
                VERIFICATION_VERIFIED,
                VERIFICATION_VERIFIED,
                VERIFICATION_MEASURED_SINGLE,
            ],
            dtype=torch.int8,
        )
        verified_camera_count = torch.tensor([2, 1, 2, 2, 1])
        verified_sequence_count = torch.ones(5, dtype=torch.int16)
        proposal_kind = torch.full((5,), PROPOSAL_NONE, dtype=torch.int8)
        support_camera_ids = torch.tensor(
            [
                [-1, -1],
                [0, -1],
                [1, -1],
                [2, -1],
                [-1, -1],
            ],
            dtype=torch.int32,
        )

        def __len__(self):
            return 5

    foliage = Foliage()
    lookup = torch.tensor([0, 0, 1], dtype=torch.int16)
    exact = _static_detail_canonical_ownership_gate(
        foliage, 0, lookup
    )
    appearance = _static_detail_same_sequence_appearance_gate(
        foliage,
        0,
        lookup,
        exact,
        fallback_weight=0.35,
    )
    # Non-detail rows retain global appearance ownership. Exact camera zero
    # owns row one, camera one is the same acquisition and receives a soft
    # SH update, while the other sequence and unsupported row remain zero.
    torch.testing.assert_close(
        appearance, torch.tensor([1.0, 1.0, 0.35, 0.0, 0.0])
    )


def test_static_detail_optical_owner_uses_positive_evidence_sequences():
    class Foliage:
        xyz = torch.zeros(5, 3)
        static_leaf_mask = torch.tensor([False, True, True, True, True])
        dynamic_leaf_mask = torch.zeros(5, dtype=torch.bool)
        verification_state = torch.tensor(
            [
                VERIFICATION_VERIFIED,
                VERIFICATION_MEASURED_SINGLE,
                VERIFICATION_VERIFIED,
                VERIFICATION_VERIFIED,
                VERIFICATION_VERIFIED,
            ],
            dtype=torch.int8,
        )
        verified_camera_count = torch.tensor([2, 1, 2, 2, 2])
        verified_sequence_count = torch.ones(5, dtype=torch.int16)
        proposal_kind = torch.full((5,), PROPOSAL_NONE, dtype=torch.int8)
        support_camera_ids = torch.tensor(
            [
                [-1, -1],
                [0, -1],
                [1, -1],
                [-1, -1],
                [2, -1],
            ],
            dtype=torch.int32,
        )
        verified_camera_ids = torch.tensor(
            [
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [3, -1],
                [-1, -1],
            ],
            dtype=torch.int32,
        )

        def __len__(self):
            return 5

    foliage = Foliage()
    lookup = torch.tensor([0, 0, 1, 2], dtype=torch.int16)
    exact = _static_detail_canonical_ownership_gate(foliage, 0, lookup)
    sequence = _static_detail_positive_evidence_sequence_gate(
        foliage, 0, lookup
    )
    optical = _static_detail_same_sequence_optical_gate(
        foliage, 0, lookup, exact, fallback_weight=0.35
    )
    assert torch.equal(
        sequence, torch.tensor([False, True, True, False, False])
    )
    torch.testing.assert_close(
        optical, torch.tensor([1.0, 1.0, 0.35, 0.0, 0.0])
    )
    # A later verified camera is positive evidence even when it is not an
    # appearance seed/support camera.
    assert torch.equal(
        _static_detail_positive_evidence_sequence_gate(
            foliage, 3, lookup
        ),
        torch.tensor([False, False, False, True, False]),
    )
    # A persisted verification witness is also an exact camera owner. This
    # must not be broadened to the rest of its sequence.
    assert torch.equal(
        _static_detail_canonical_ownership_gate(foliage, 3, lookup),
        torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0]),
    )


def test_static_detail_direct_optical_priors_use_same_camera_permission():
    class Foliage:
        xyz = torch.zeros(5, 3)
        static_leaf_mask = torch.tensor([False, True, True, True, False])
        opacities = torch.full((5,), 0.5)

        def __len__(self):
            return len(self.xyz)

    foliage = Foliage()
    base = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
    optical = torch.tensor([1.0, 1.0, 0.35, 0.0, 1.0])

    routed = _static_detail_optical_regularizer_weight(
        foliage, base, optical
    )

    torch.testing.assert_close(
        routed, torch.tensor([0.9, 0.8, 0.245, 0.0, 0.5])
    )
    # The helper must not mutate the base posterior tensor reused by audits.
    torch.testing.assert_close(base, torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5]))


def test_static_detail_cleanup_keeps_geometry_read_only_and_uses_optical_gate():
    class Foliage:
        xyz = torch.zeros(5, 3)
        static_leaf_mask = torch.tensor([False, True, True, True, False])
        opacities = torch.full((5,), 0.5)

        def __len__(self):
            return len(self.xyz)

    foliage = Foliage()
    geometry = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0])
    optical = torch.tensor([1.0, 1.0, 0.35, 0.0, 1.0])

    geometry_gate, appearance_gate, opacity_gate = (
        _static_detail_cleanup_gradient_gates(
            foliage, geometry, optical
        )
    )

    torch.testing.assert_close(
        geometry_gate, torch.zeros(5)
    )
    torch.testing.assert_close(appearance_gate, torch.zeros(5))
    torch.testing.assert_close(
        opacity_gate, torch.tensor([0.0, 1.0, 0.35, 0.0, 0.0])
    )


def test_v83_repair_clears_only_static_detail_opacity_adam_rows():
    opacity = torch.nn.Parameter(torch.zeros(4, 1))
    optimizer = torch.optim.Adam(
        [{"params": [opacity], "name": "opacity"}], lr=1.0e-3
    )
    optimizer.state[opacity] = {
        "step": torch.tensor(17.0),
        "exp_avg": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        "exp_avg_sq": torch.tensor([[5.0], [6.0], [7.0], [8.0]]),
    }

    _zero_volume_opacity_optimizer_rows(
        optimizer, torch.tensor([1, 3])
    )

    state = optimizer.state[opacity]
    torch.testing.assert_close(
        state["exp_avg"], torch.tensor([[1.0], [0.0], [3.0], [0.0]])
    )
    torch.testing.assert_close(
        state["exp_avg_sq"], torch.tensor([[5.0], [0.0], [7.0], [0.0]])
    )
    assert float(state["step"]) == 17.0


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


def test_v100_migrates_only_exact_front_clipped_cleanup_contracts():
    detail = {
        "contract": STATIC_DETAIL_GLOBAL_CLEANUP_CONTRACT,
        "every": 2,
        "weight": 0.25,
        "forward_roles": ["surface_zero_static_detail_intrinsic"],
        "rigid_depth_permission": "strict_front",
        "behind_surface_policy": "zero_cleanup_gradient",
    }
    envelope = {
        "contract": PERSISTENT_ENVELOPE_GLOBAL_CLEANUP_CONTRACT,
        "every": 2,
        "weight": 0.125,
        "forward_roles": [
            "surface_zero_resolved_persistent_envelope_intrinsic"
        ],
        "rigid_depth_permission": "strict_front",
        "behind_surface_policy": "zero_cleanup_gradient",
    }
    current = {
        "invariant": "unchanged",
        "static_detail_global_cleanup": detail,
        "persistent_envelope_global_cleanup": envelope,
    }
    saved_detail = dict(detail)
    saved_detail["contract"] = (
        STATIC_DETAIL_GLOBAL_CLEANUP_V99_PREDECESSOR_CONTRACT
    )
    saved_detail["forward_roles"] = ["surface", "static_detail"]
    saved_detail.pop("rigid_depth_permission")
    saved_detail.pop("behind_surface_policy")
    saved_envelope = dict(envelope)
    saved_envelope["contract"] = (
        PERSISTENT_ENVELOPE_GLOBAL_CLEANUP_V99_PREDECESSOR_CONTRACT
    )
    saved_envelope["forward_roles"] = [
        "surface",
        "persistent_envelope",
    ]
    saved_envelope.pop("rigid_depth_permission")
    saved_envelope.pop("behind_surface_policy")
    saved = {
        "invariant": "unchanged",
        "static_detail_global_cleanup": saved_detail,
        "persistent_envelope_global_cleanup": saved_envelope,
    }
    assert _resume_training_contract_differences(saved, current) == {
        "static_detail_global_cleanup",
        "persistent_envelope_global_cleanup",
    }
    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_v100_front_clipped_cleanup_migration=True,
    )
    assert _resume_training_contract_differences(
        {**saved, "invariant": "changed"},
        current,
        allow_v100_front_clipped_cleanup_migration=True,
    ) == {"invariant"}
    with pytest.raises(RuntimeError, match="predecessor contract mismatch"):
        _resume_training_contract_differences(
            {
                **saved,
                "static_detail_global_cleanup": {
                    **saved_detail,
                    "weight": 0.5,
                },
            },
            current,
            allow_v100_front_clipped_cleanup_migration=True,
        )


def test_trainer_repair_migrates_v83_symmetric_sequence_optical_contract():
    saved = {
        "reconstruction_target": "static",
        "static_training_stages": {"detail_training_signals": ["old"]},
        "static_detail_global_cleanup": {
            "contract": STATIC_DETAIL_GLOBAL_CLEANUP_V83_PREDECESSOR_CONTRACT,
            "weight": 0.25,
        },
        "static_detail_canonical_ownership": {
            "geometry_mass_topology_owner": (
                "persisted_exact_support_cameras_and_verified_multiview"
            )
        },
        "parameter_loss_permission_matrix": {
            "static_leaf_optical_mass": ["exact_camera_only"]
        },
    }
    current = {
        **saved,
        "static_training_stages": {
            "detail_training_signals": ["symmetric_sequence_optical"]
        },
        "static_detail_global_cleanup": {
            "contract": STATIC_DETAIL_GLOBAL_CLEANUP_CONTRACT,
            "weight": 0.25,
        },
        "static_detail_canonical_ownership": {
            "geometry_topology_owner": (
                "persisted_exact_support_cameras_and_verified_multiview"
            ),
            "optical_mass_owner": (
                "exact_support_camera_weight1_plus_positive_evidence_"
                "sequence_soft_weight"
            ),
            "same_sequence_optical_weight": 0.35,
        },
        "parameter_loss_permission_matrix": {
            "static_leaf_optical_mass": ["symmetric_sequence_optical"]
        },
    }
    assert _resume_training_contract_differences(saved, current) == {
        "static_training_stages",
        "static_detail_global_cleanup",
        "static_detail_canonical_ownership",
        "parameter_loss_permission_matrix",
    }
    assert not _resume_training_contract_differences(
        saved, current, allow_trainer_repair_migration=True
    )
    changed = {
        **current,
        "static_detail_canonical_ownership": {
            **current["static_detail_canonical_ownership"],
            "same_sequence_optical_weight": 0.5,
        },
    }
    assert _resume_training_contract_differences(
        saved, changed, allow_trainer_repair_migration=True
    ) == {
        "static_training_stages",
        "static_detail_global_cleanup",
        "static_detail_canonical_ownership",
        "parameter_loss_permission_matrix",
    }


def test_trainer_repair_migrates_v84_static_intrinsic_color_coverage():
    saved = {
        "reconstruction_target": "static",
        "sampling_schedule_sha256": "old",
        "static_training_stages": {"detail_training_signals": ["old"]},
        "static_detail_isolated_supervision": {
            "contract": STATIC_DETAIL_ISOLATED_V84_PREDECESSOR_CONTRACT,
            "every": 2,
            "weight": 1.0,
        },
        "static_volume_isolated_supervision": {
            "contract": STATIC_VOLUME_ISOLATED_V84_PREDECESSOR_CONTRACT,
            "every": 8,
            "weight": 0.35,
        },
        "parameter_loss_permission_matrix": {
            "static_leaf_sh": ["sparse_seed_cameras"]
        },
    }
    current = {
        **saved,
        "sampling_schedule_sha256": "new",
        "static_training_stages": {
            "detail_training_signals": ["positive_sequence_intrinsic_color"]
        },
        "static_detail_isolated_supervision": {
            "contract": STATIC_DETAIL_ISOLATED_CONTRACT,
            "every": 2,
            "weight": 1.0,
        },
        "static_volume_isolated_supervision": {
            "contract": STATIC_VOLUME_ISOLATED_CONTRACT,
            "every": 4,
            "weight": 0.75,
            "intrinsic_alpha_floor": (
                STATIC_VOLUME_INTRINSIC_ALPHA_FLOOR
            ),
        },
        "parameter_loss_permission_matrix": {
            "static_leaf_sh": ["positive_sequence_intrinsic_color"]
        },
    }
    expected = {
        "sampling_schedule_sha256",
        "static_training_stages",
        "static_detail_isolated_supervision",
        "static_volume_isolated_supervision",
        "parameter_loss_permission_matrix",
    }
    assert _resume_training_contract_differences(saved, current) == expected
    assert not _resume_training_contract_differences(
        saved, current, allow_trainer_repair_migration=True
    )

    unsafe = json.loads(json.dumps(current))
    unsafe["static_volume_isolated_supervision"]["weight"] = 1.5
    assert _resume_training_contract_differences(
        saved, unsafe, allow_trainer_repair_migration=True
    ) == expected


def test_trainer_repair_migrates_v85_geometry_and_handoff_ownership():
    old_candidate = (
        "verified_same_group_exact_support_camera__primitive_center_"
        "cannot_veto_pixel_ray_evidence"
    )
    new_candidate = (
        "verified_same_group_positive_evidence_sequence__seed_camera_table_"
        "is_not_complete_visibility__primitive_center_cannot_veto_pixel_"
        "ray_evidence"
    )
    saved = {
        "reconstruction_target": "static",
        "static_training_stages": {
            "rgb_role_contract": STATIC_STAGE_RGB_ROLE_V85_PREDECESSOR_CONTRACT
        },
        "static_detail_isolated_supervision": {
            "contract": STATIC_DETAIL_ISOLATED_V85_PREDECESSOR_CONTRACT,
            "every": 2,
            "weight": 1.0,
        },
        "static_volume_isolated_supervision": {
            "contract": STATIC_VOLUME_ISOLATED_V85_PREDECESSOR_CONTRACT,
            "every": 4,
            "weight": 0.75,
        },
        "static_detail_canonical_ownership": {
            "geometry_topology_owner": (
                "persisted_exact_support_cameras_and_verified_multiview"
            )
        },
        "static_ray_local_mass_handoff": {
            "candidate_contract": old_candidate
        },
        "static_detail_global_cleanup": {"contract": "v85"},
        "parameter_loss_permission_matrix": {"static_leaf_sh": ["v85"]},
    }
    current = {
        **saved,
        "static_training_stages": {
            "rgb_role_contract": STATIC_STAGE_RGB_ROLE_CONTRACT
        },
        "static_detail_isolated_supervision": {
            "contract": STATIC_DETAIL_ISOLATED_CONTRACT,
            "every": 2,
            "weight": 1.0,
        },
        "static_volume_isolated_supervision": {
            "contract": STATIC_VOLUME_ISOLATED_CONTRACT,
            "every": 4,
            "weight": 0.75,
        },
        "static_detail_canonical_ownership": {
            "geometry_topology_owner": (
                "verified_exact_support_plus_positive_evidence_sequence_"
                "soft_geometry__topology_exact_support_only"
            ),
            "same_sequence_geometry_weight": (
                STATIC_DETAIL_SAME_SEQUENCE_GEOMETRY_WEIGHT
            ),
        },
        "static_ray_local_mass_handoff": {
            "candidate_contract": new_candidate
        },
        "static_detail_global_cleanup": {"contract": "v86"},
        "parameter_loss_permission_matrix": {"static_leaf_sh": ["v86"]},
    }
    assert _resume_training_contract_differences(saved, current)
    assert not _resume_training_contract_differences(
        saved, current, allow_trainer_repair_migration=True
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


def test_trainer_repair_migrates_consensus_hit_and_birth_backpressure():
    legacy_birth_contract = (
        "uncovered_hit_uses_complete_calibrated_ray_depth_interval__"
        "cross_sequence_segments_vote_in_multiscale_visual_hull_cells__"
        "minimum_two_cameras_and_two_sequences_remain_mandatory__"
        "newborn_is_low_mass_static_detail_and_free_space_prunable"
    )
    saved = {
        "static_training_stages": {"detail_training_signals": ["legacy"]},
        "static_detail_isolated_supervision": {"contract": "legacy"},
        "static_detail_global_cleanup": {"contract": "legacy"},
        "static_ray_birth": {"contract": legacy_birth_contract},
        "static_detail_canonical_ownership": {
            "gradient_owner": "persisted_support_camera_sequences"
        },
        "parameter_loss_permission_matrix": {
            "static_leaf_optical_mass": ["legacy"]
        },
    }
    current = {
        "static_training_stages": {"detail_training_signals": ["consensus"]},
        "static_detail_isolated_supervision": {
            "contract": STATIC_DETAIL_ISOLATED_CONTRACT
        },
        "static_detail_global_cleanup": {
            "contract": STATIC_DETAIL_GLOBAL_CLEANUP_CONTRACT
        },
        "static_ray_birth": {
            "contract": STATIC_RAY_BIRTH_VISUAL_HULL_CONTRACT,
            "verification_debt_backpressure": (
                "same_continuous_smoothstep_capacity_as_ordinary_split"
            ),
        },
        "static_detail_canonical_ownership": {
            "rgb_geometry_topology_owner": (
                "persisted_support_camera_sequences"
            ),
            "positive_ray_owner": (
                "persisted_support_or_verified_two_sequence_consensus"
            ),
        },
        "parameter_loss_permission_matrix": {
            "static_leaf_optical_mass": ["verified_consensus"]
        },
    }
    assert _resume_training_contract_differences(saved, current)
    assert not _resume_training_contract_differences(
        saved, current, allow_trainer_repair_migration=True
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
    assert _trainer_repair_hash_change_is_allowed(
        {"trainer", "hybrid_renderer"},
        enabled=True,
        allow_hybrid_renderer_topology_extension=True,
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


def test_mature_handoff_atlas_residual_trains_only_completion_geometry():
    surface = SimpleNamespace(
        get_xyz=torch.ones(4, 3),
        _xyz=torch.nn.Parameter(torch.ones(4, 3)),
        _scaling=torch.nn.Parameter(torch.ones(4, 2)),
        _rotation=torch.nn.Parameter(torch.ones(4, 4)),
        _opacity=torch.nn.Parameter(torch.ones(4, 1)),
        _features_dc=torch.nn.Parameter(torch.ones(4, 1, 3)),
        _features_rest=torch.nn.Parameter(torch.ones(4, 3, 3)),
    )
    for name, parameter in vars(surface).items():
        if isinstance(parameter, torch.nn.Parameter):
            parameter.grad = torch.ones_like(parameter)
    chart_parameter = torch.nn.Parameter(torch.ones(2, 3))
    chart_parameter.grad = torch.ones_like(chart_parameter)

    audit = _apply_mature_surface_gradient_policy(
        surface,
        "atlas_residual",
        chart_surface=torch.nn.ParameterList([chart_parameter]),
        residual_start=3,
    )

    for parameter in (
        surface._xyz,
        surface._scaling,
        surface._rotation,
        surface._opacity,
    ):
        assert not bool(parameter.grad[:3].any())
        assert bool(parameter.grad[3:].any())
    assert bool(surface._features_dc.grad.all())
    assert bool(chart_parameter.grad.all())
    assert audit["residual_rows"] == 1
    assert audit["topology_trainable"] is True
    assert (
        audit["topology_trainable_scope"]
        == "evidence_completion_suffix_only"
    )
    assert audit["chart_geometry_trainable"] is True


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


def test_parameter_loss_audit_skips_before_unbounded_dense_gradients():
    foliage = SimpleNamespace(
        xyz=torch.zeros(2, 3, requires_grad=True),
        log_scales=torch.zeros(2, 3, requires_grad=True),
        opacity_logits=torch.zeros(2, 1, requires_grad=True),
        features=torch.zeros(2, 1, 3, requires_grad=True),
        static_skeleton_mask=torch.tensor([False, False]),
        persistent_envelope_mask=torch.tensor([True, False]),
        static_leaf_mask=torch.tensor([False, True]),
    )
    loss = foliage.opacity_logits.sum()

    audit = _parameter_loss_gradient_audit(
        foliage, {"ray_free_hit": loss}, maximum_rows=1
    )

    assert audit["skipped"]
    assert audit["foliage_rows"] == 2
    assert audit["maximum_rows"] == 1
    assert audit["losses"] == {}


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


def test_atomic_checkpoint_keeps_one_valid_previous_generation(tmp_path):
    rolling = tmp_path / "hybrid_teacher_checkpoint.pth"
    _save_checkpoint(rolling, {"iteration": 10, "value": torch.arange(4)})
    _save_checkpoint(rolling, {"iteration": 20, "value": torch.arange(5)})

    previous = rolling.with_suffix(rolling.suffix + ".previous")
    assert torch.load(rolling)["iteration"] == 20
    assert torch.load(previous)["iteration"] == 10
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_atomic_json_never_leaves_a_fixed_shared_tmp_name(tmp_path):
    destination = tmp_path / "state.json"
    _write_json_atomic(destination, {"generation": 1})
    _write_json_atomic(destination, {"generation": 2})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "generation": 2
    }
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_clean_output_allows_only_live_supervisor_control_files(
    tmp_path, monkeypatch
):
    run = (tmp_path / "run").resolve()
    run.mkdir()
    command = ("/python", "/repo/train.py", "-m", str(run))
    command_hash = hashlib.sha256(
        json.dumps(
            list(command),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    paths = {
        "lock_file": run / ".training_supervisor.lock",
        "heartbeat_file": run / "training_supervisor_heartbeat.json",
        "contract_file": run / "training_supervisor_contract.json",
        "supervisor_log": run / "training_supervisor.log",
        "training_log": run / "training.log",
    }
    paths["supervisor_log"].write_text("")
    paths["training_log"].write_text("")
    paths["lock_file"].write_text(
        json.dumps(
            {"supervisor_pid": 456, "command_sha256": command_hash}
        )
    )
    paths["heartbeat_file"].write_text(
        json.dumps(
            {
                "state": "running",
                "training_pid": 123,
                "supervisor_pid": 456,
                "command_sha256": command_hash,
            }
        )
    )
    paths["contract_file"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_dir": str(run),
                "base_command": list(command),
                "command_sha256": command_hash,
                **{key: str(value) for key, value in paths.items()},
            }
        )
    )
    monkeypatch.setenv(
        "G4SPLAT_DETACHED_SUPERVISOR_CONTRACT",
        str(paths["contract_file"]),
    )

    _assert_clean_training_output(
        run,
        resume_requested=False,
        current_command=command,
        process_id=123,
        parent_process_id=456,
    )
    (run / "old-result.json").write_text("{}")
    with pytest.raises(FileExistsError, match="Refusing non-empty output"):
        _assert_clean_training_output(
            run,
            resume_requested=False,
            current_command=command,
            process_id=123,
            parent_process_id=456,
        )


def test_early_checkpoint_cadence_bounds_pre_milestone_loss():
    common = {
        "final_iteration": 30_000,
        "checkpoint_every": 3_000,
        "early_checkpoint_every": 1_000,
        "early_checkpoint_until": 2_000,
        "retained_iterations": {3_000, 6_000},
    }
    assert _checkpoint_due(1_000, **common)
    assert _checkpoint_due(2_000, **common)
    assert _checkpoint_due(3_000, **common)
    assert not _checkpoint_due(2_450, **common)
    assert _checkpoint_due(
        2_450, **common, graceful_stop_requested=True
    )
    assert _checkpoint_due(30_000, **common)


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


def test_rigid_completion_admits_only_calibrated_dav2_hole_as_weak_birth():
    seed = {
        "xyz": np.asarray(
            [[0.01, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "scales": np.full((3, 2), 0.03, dtype=np.float32),
        "geometry_confidence": np.asarray(
            [1.0, 0.2, 1.0], dtype=np.float32
        ),
        "position_sigma": np.asarray(
            [0.04, 0.25, 0.04], dtype=np.float32
        ),
        "pointmap_cross_sequence_supported": np.zeros(3, dtype=bool),
        "persistent_geometry_evidence": np.zeros(3, dtype=bool),
        "source_type": np.asarray(
            [
                GaussianModel.SOURCE_FREE_RESIDUAL,
                GaussianModel.SOURCE_DAV2_RIGID_HOLE,
                GaussianModel.SOURCE_FREE_RESIDUAL,
            ],
            dtype=np.int8,
        ),
    }
    selected, audit = _rigid_completion_seed_indices(
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        seed,
        maximum_seeds=4,
    )
    assert selected.tolist() == [1]
    assert audit["weak_dav2_hole_selected"] == 1
    assert "low_opacity_dav2" in audit["birth_authority"]


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


def test_evidence_adaptive_role_quotas_use_absolute_bandwidth_for_growth():
    quotas = _evidence_adaptive_role_quotas(
        {
            "static_skeleton": 7_000,
            "canonical_crown": 250_000,
            "dynamic_leaf": 0,
        },
        {
            "static_skeleton": 7_000,
            "canonical_crown": 250_000,
            "dynamic_leaf": 0,
        },
        1_481,
        effective_eligible_mass={
            "static_skeleton": 2_398.8,
            "canonical_crown": 79_034.5,
            "dynamic_leaf": 0.0,
        },
        normalize_by_observable_population=False,
    )
    # This reproduces the v110 trace that previously inverted the physical
    # demand (923 skeleton / 558 crown) after population normalization.
    assert quotas["canonical_crown"] > 1_400
    assert quotas["static_skeleton"] < 50
    assert sum(quotas.values()) == 1_481


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


def test_screen_evidence_resume_migrates_only_audited_topology_contract():
    saved = {
        "geometry_weight": 0.12,
        "chart_atlas_lifecycle": {"ordinary_opacity_retirement": False},
    }
    current = {
        **saved,
        "surface_screen_topology": dict(SURFACE_SCREEN_TOPOLOGY_CONTRACT),
    }

    assert _resume_training_contract_differences(saved, current) == {
        "surface_screen_topology"
    }
    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_surface_screen_evidence_repair_migration=True,
    )
    assert _resume_training_contract_differences(
        saved,
        {**current, "geometry_weight": 0.13},
        allow_surface_screen_evidence_repair_migration=True,
    ) == {"geometry_weight"}


def test_screen_evidence_resume_rejects_unattested_target_or_prior_contract():
    saved = {"geometry_weight": 0.12}
    corrupt_target = {
        **saved,
        "surface_screen_topology": {
            **SURFACE_SCREEN_TOPOLOGY_CONTRACT,
            "independent_radius_only_split_evidence": True,
        },
    }
    with pytest.raises(RuntimeError, match="audited residual-conditioned"):
        _resume_training_contract_differences(
            saved,
            corrupt_target,
            allow_surface_screen_evidence_repair_migration=True,
        )

    already_declared = {
        **saved,
        "surface_screen_topology": dict(SURFACE_SCREEN_TOPOLOGY_CONTRACT),
    }
    with pytest.raises(RuntimeError, match="exact v67 prefix"):
        _resume_training_contract_differences(
            already_declared,
            already_declared,
            allow_surface_screen_evidence_repair_migration=True,
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


def test_rigid_profile_resolves_geometry_balanced_optimizer_for_direct_runs():
    args = SimpleNamespace(
        training_profile="hybrid_rigid_stage1",
        geometry_gradient_ratio=0.0,
        position_lr_init=1.6e-4,
        position_lr_final=1.6e-6,
        position_lr_delay_mult=0.01,
    )
    audit = _apply_training_profile_optimizer_defaults(args, [])
    assert args.geometry_gradient_ratio == pytest.approx(0.15)
    assert args.position_lr_init == pytest.approx(1.6e-5)
    assert audit["resolved"]["geometry_gradient_ratio"]["source"] == (
        "named_profile"
    )
    assert audit["resolved"]["position_lr_init"]["source"] == (
        "named_profile"
    )


def test_rigid_profile_preserves_explicit_optimizer_ablation():
    args = SimpleNamespace(
        training_profile="hybrid_rigid_stage1",
        geometry_gradient_ratio=0.0,
        position_lr_init=1.6e-4,
        position_lr_final=1.6e-6,
        position_lr_delay_mult=0.01,
    )
    audit = _apply_training_profile_optimizer_defaults(
        args,
        ["--geometry-gradient-ratio=0.0", "--position_lr_init", "0.00016"],
    )
    assert args.geometry_gradient_ratio == 0.0
    assert args.position_lr_init == pytest.approx(1.6e-4)
    assert audit["resolved"]["geometry_gradient_ratio"]["source"] == (
        "explicit_cli"
    )
    assert audit["resolved"]["position_lr_init"]["source"] == (
        "explicit_cli"
    )


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


def test_static_handoff_starts_detail_at_measured_envelope_optimum():
    profile = TRAINING_PROFILES["static_handoff_quality"]

    assert profile["foliage_start"] == 0.0
    assert _phase(599, 30_000, "static_handoff_quality") == (
        "canonical_bootstrap"
    )
    assert _phase(600, 30_000, "static_handoff_quality") == "topology"
    assert _phase(1_499, 30_000, "static_handoff_quality") == "topology"
    assert _phase(1_500, 30_000, "static_handoff_quality") == (
        "static_foliage"
    )
    # This is a lifecycle transition after one full RGB camera epoch, not a
    # pass/fail gate.  The ray scheduler continues after handoff and consumes
    # the remaining calibrated evidence normally.
    assert 1_500 >= 1_487


def test_ray_birth_stat_extension_preserves_measured_prefix():
    stats = {
        "gradient": torch.tensor([1.0, 2.0]),
        "conditioned_context_id": torch.tensor(
            [7, 9], dtype=torch.int32
        ),
        "rigid_front_conflict_camera_id": torch.tensor(
            [11, 12], dtype=torch.int32
        ),
        "positive_optical_demand_observations": torch.tensor(
            [3, 4], dtype=torch.int16
        ),
    }
    extended = _extend_volume_stats_for_births(stats, 4)
    torch.testing.assert_close(
        extended["gradient"], torch.tensor([1.0, 2.0, 0.0, 0.0])
    )
    assert extended["conditioned_context_id"].tolist() == [7, 9, -1, -1]
    assert extended["rigid_front_conflict_camera_id"].tolist() == [
        11,
        12,
        -1,
        -1,
    ]
    assert extended["positive_optical_demand_observations"].tolist() == [
        3,
        4,
        0,
        0,
    ]


def test_cpu_parallelism_bounds_native_prefetch_callers():
    assert _resolve_cpu_intraop_threads(32, 8) == 3
    assert _resolve_cpu_intraop_threads(16, 8) == 1
    assert _resolve_cpu_intraop_threads(32, 0) == 4
    assert _resolve_cpu_intraop_threads(4, 8) == 1


def test_cpu_parallelism_honours_bounded_explicit_override():
    assert _resolve_cpu_intraop_threads(16, 8, 2) == 2
    assert _resolve_cpu_intraop_threads(4, 8, 16) == 4
    with pytest.raises(ValueError, match="non-negative"):
        _resolve_cpu_intraop_threads(16, -1)


def test_cpu_parallelism_is_runtime_only_for_resume_identity():
    saved = {
        "reconstruction_target": "static",
        "cpu_parallelism": {
            "available_cpus": 32,
            "resolved_intraop_threads": 3,
        },
    }
    current = {"reconstruction_target": "static"}
    assert not _resume_training_contract_differences(saved, current)


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


def test_exact_resume_restores_model_derived_camera_schedules():
    names = (
        "rgb",
        "conditioned",
        "geometry",
        "topology",
        "static_detail",
        "static_skeleton",
        "static_volume",
    )
    saved = {
        name: np.arange(12, dtype=np.int64) + 100 * index
        for index, name in enumerate(names)
    }
    computed = {name: value + 7 for name, value in saved.items()}

    restored, preserved = _restore_resume_camera_schedules(
        computed,
        {"schedules": saved},
        horizon=12,
    )

    assert preserved == frozenset(names)
    for name in names:
        np.testing.assert_array_equal(restored[name], saved[name])
        assert restored[name] is not saved[name]


def test_explicit_conditioned_schedule_repair_preserves_other_streams():
    names = (
        "rgb",
        "conditioned",
        "geometry",
        "topology",
        "static_detail",
        "static_skeleton",
        "static_volume",
    )
    saved = {
        name: np.arange(8, dtype=np.int64) + 10 * index
        for index, name in enumerate(names)
    }
    computed = {name: value + 1 for name, value in saved.items()}

    restored, preserved = _restore_resume_camera_schedules(
        computed,
        {"schedules": saved},
        horizon=8,
        allow_conditioned_repair=True,
    )

    assert "conditioned" not in preserved
    np.testing.assert_array_equal(
        restored["conditioned"], computed["conditioned"]
    )
    for name in set(names) - {"conditioned"}:
        np.testing.assert_array_equal(restored[name], saved[name])


def test_static_color_schedule_repair_replaces_only_color_streams():
    names = (
        "rgb",
        "conditioned",
        "geometry",
        "topology",
        "static_detail",
        "static_skeleton",
        "static_volume",
    )
    saved = {
        name: np.arange(8, dtype=np.int64) + 10 * index
        for index, name in enumerate(names)
    }
    computed = {name: value + 1 for name, value in saved.items()}

    restored, preserved = _restore_resume_camera_schedules(
        computed,
        {"schedules": saved},
        horizon=8,
        allow_static_color_repair=True,
    )

    assert "static_detail" not in preserved
    assert "static_volume" not in preserved
    np.testing.assert_array_equal(
        restored["static_detail"], computed["static_detail"]
    )
    np.testing.assert_array_equal(
        restored["static_volume"], computed["static_volume"]
    )
    for name in set(names) - {"static_detail", "static_volume"}:
        np.testing.assert_array_equal(restored[name], saved[name])


def test_static_replacement_schedule_repair_preserves_other_streams():
    saved = {
        "rgb": np.arange(8, dtype=np.int64),
        "static_detail": np.arange(8, dtype=np.int64) + 10,
    }
    computed = {
        **{name: value + 1 for name, value in saved.items()},
        "static_replacement": np.arange(8, dtype=np.int64) + 20,
    }
    restored, preserved = _restore_resume_camera_schedules(
        computed,
        {"schedules": saved},
        horizon=8,
        allow_static_replacement_repair=True,
    )
    assert preserved == frozenset(saved)
    np.testing.assert_array_equal(restored["rgb"], saved["rgb"])
    np.testing.assert_array_equal(
        restored["static_detail"], saved["static_detail"]
    )
    np.testing.assert_array_equal(
        restored["static_replacement"], computed["static_replacement"]
    )


def test_resume_camera_schedule_rejects_wrong_horizon():
    with pytest.raises(RuntimeError, match="fixed horizon"):
        _restore_resume_camera_schedules(
            {"rgb": np.arange(4, dtype=np.int64)},
            {"schedules": {"rgb": np.arange(3, dtype=np.int64)}},
            horizon=4,
        )


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


def test_canonical_occlusion_starts_strictly_after_volume_topology():
    args = SimpleNamespace(canonical_occlusion_start_iteration=12_000)
    assert not _canonical_occlusion_schedule_active(11_998, args)
    assert not _canonical_occlusion_schedule_active(11_999, args)
    assert _canonical_occlusion_schedule_active(12_000, args)


def test_canonical_occlusion_has_dedicated_cadence_and_stops_before_polish():
    args = SimpleNamespace(
        canonical_occlusion_start_iteration=18_000,
        canonical_occlusion_until_iteration=25_500,
        canonical_occlusion_every=4,
    )
    assert not _canonical_occlusion_schedule_active(17_999, args)
    assert _canonical_occlusion_schedule_active(18_003, args)
    assert not _canonical_occlusion_schedule_active(18_004, args)
    assert _canonical_occlusion_schedule_active(25_499, args)
    assert not _canonical_occlusion_schedule_active(25_503, args)


def test_conditioned_branch_keeps_training_during_topology_settle():
    horizon = 12_000
    profile = "hybrid_handoff_quality"
    assert _phase(11_040, horizon, profile) == "canonical_polish"
    assert _conditioned_branch_active(11_040, horizon, profile)
    assert _conditioned_branch_active(11_999, horizon, profile)


def test_static_spatial_uncertainty_does_not_enable_temporal_branch():
    assert not _static_spatial_uncertainty_active(
        "static", foliage_active=True, static_detail_active=True
    )
    assert not _static_spatial_uncertainty_active(
        "static", foliage_active=True, static_detail_active=False
    )
    assert not _static_spatial_uncertainty_active(
        "sequence_conditioned_legacy",
        foliage_active=True,
        static_detail_active=True,
    )


def test_static_uncertainty_optimizer_migration_appends_state_free_parameter():
    xyz = torch.nn.Parameter(torch.tensor([1.0]))
    old_appearance = torch.nn.Parameter(torch.tensor([2.0]))
    old = torch.optim.Adam(
        [
            {"params": [xyz], "lr": 1e-3, "name": "xyz"},
            {
                "params": [old_appearance],
                "lr": 2e-3,
                "name": "appearance",
            },
        ]
    )
    (xyz.square() + old_appearance.square()).backward()
    old.step()
    state = old.state_dict()

    new_xyz = torch.nn.Parameter(torch.tensor([1.0]))
    new_appearance = torch.nn.Parameter(torch.tensor([2.0]))
    decoder = torch.nn.Parameter(torch.tensor([0.0]))
    current = torch.optim.Adam(
        [
            {"params": [new_xyz], "lr": 1e-3, "name": "xyz"},
            {
                "params": [new_appearance, decoder],
                "lr": 2e-3,
                "name": "appearance",
            },
        ]
    )
    migrated = _restore_volume_optimizer_state(
        current, state, allow_static_uncertainty_extension=True
    )
    assert migrated
    assert "exp_avg" in current.state[new_xyz]
    assert "exp_avg" in current.state[new_appearance]
    assert current.state[decoder] == {}


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


def test_foliage_rigid_depth_calibration_requires_same_surface_digest():
    initialization = {
        "foliage": {
            "rigid_depth_calibration": {
                "enabled": True,
                "structural_ply_sha256": "rigid-a",
                "source": "trained_native_rigid_2dgs_surface_render",
                "depth_view_count": 527,
            }
        }
    }
    warmstart = {"surface_ply_sha256": "rigid-a"}

    audit = _validate_foliage_rigid_calibration(
        initialization, warmstart
    )

    assert audit["validated"]
    assert audit["depth_view_count"] == 527


def test_foliage_rigid_depth_calibration_rejects_cross_wired_handoff():
    initialization = {
        "foliage": {
            "rigid_depth_calibration": {
                "enabled": True,
                "structural_ply_sha256": "rigid-a",
            }
        }
    }
    with pytest.raises(RuntimeError, match="differs from the mixed surface"):
        _validate_foliage_rigid_calibration(
            initialization, {"surface_ply_sha256": "rigid-b"}
        )


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
                "training_profile": "hybrid_rigid_stage1",
                "surface_ply": str(surface.resolve()),
                "training_contract": {
                    "schedule_horizon": 32_000,
                    "surface_optimizer": RIGID_SURFACE_OPTIMIZER,
                },
            }
        ),
        encoding="utf-8",
    )

    accepted = _validate_surface_warmstart(surface, manifest)

    assert {
        key: accepted["surface_optimizer"][key]
        for key in RIGID_SURFACE_OPTIMIZER
    } == RIGID_SURFACE_OPTIMIZER
    assert accepted["surface_optimizer"]["non_position_lr_decay_from"] == 24_000
    assert accepted["surface_optimizer"]["non_position_lr_decay_until"] == 32_000
    assert accepted["surface_optimizer"]["non_position_lr_final_mult"] == 0.1
    assert accepted["surface_optimizer_source"] == (
        "validated_legacy_producer_result_plus_validated_rigid_"
        "lifecycle_inference"
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
        surface_optimizer=RIGID_SURFACE_OPTIMIZER_WITH_POLISH,
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
        surface_optimizer=RIGID_SURFACE_OPTIMIZER_WITH_POLISH,
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
        surface_optimizer=RIGID_SURFACE_OPTIMIZER_WITH_POLISH,
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


def test_canonical_backward_frees_shared_graph_after_foliage_supplement():
    class Foliage(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.opacity_logits = torch.nn.Parameter(torch.tensor([[2.0]]))

    foliage = Foliage()
    surface = torch.nn.Parameter(torch.tensor([[3.0]]))
    shared = surface * foliage.opacity_logits
    canonical_loss = shared.sum()
    supplement = foliage.opacity_logits.square().sum()

    canonical_opacity = _backward_canonical_with_foliage_supplement(
        canonical_loss,
        supplement,
        foliage,
        foliage_active=True,
    )

    torch.testing.assert_close(surface.grad, torch.tensor([[2.0]]))
    torch.testing.assert_close(
        foliage.opacity_logits.grad, torch.tensor([[7.0]])
    )
    torch.testing.assert_close(canonical_opacity, torch.tensor([[3.0]]))


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


def test_moge3_metric_depth_fills_only_unowned_rigid_pixels():
    depth = torch.full((1, 2, 2), 4.0, requires_grad=True)
    package = SimpleNamespace(
        depth=depth,
        normal_world=torch.zeros(3, 2, 2),
    )
    evidence = {
        "moge3_depth_m": torch.full((1, 2, 2), 2.0),
        "moge3_valid_mask": torch.ones(1, 2, 2),
        "moge3_refinement_log_depth_std": torch.zeros(1, 2, 2),
        "moge3_refinement_final_delta_log_depth": torch.zeros(1, 2, 2),
    }
    args = SimpleNamespace(
        geometry_weight=0.12,
        plane_weight=0.10,
        normal_weight=0.04,
        ordinal_weight=0.015,
        moge3_rigid_depth_weight=0.04,
    )
    loss, values = _geometry_losses(
        package, evidence, torch.ones(2, 2), args
    )
    assert values["moge3_rigid_depth"] > 0
    assert values["moge3_rigid_depth_pixels"] == 4
    loss.backward()
    assert depth.grad.abs().sum() > 0

    owned_depth = torch.full((1, 2, 2), 4.0, requires_grad=True)
    owned_package = SimpleNamespace(
        depth=owned_depth,
        normal_world=torch.zeros(3, 2, 2),
    )
    evidence.update(
        {
            "source_bitmask": torch.full(
                (1, 2, 2), 2, dtype=torch.uint8
            ),
            "rho_mean": torch.full((1, 2, 2), 0.5),
            "rho_variance": torch.full((1, 2, 2), 0.01),
            "support_view_count": torch.ones(1, 2, 2),
        }
    )
    _, owned_values = _geometry_losses(
        owned_package, evidence, torch.ones(2, 2), args
    )
    assert owned_values["moge3_rigid_depth_pixels"] == 0
    assert owned_values["moge3_rigid_depth"] == 0.0


def test_moge3_normal_is_camera_exact_confidence_weighted_and_owner_isolated():
    depth = torch.full((1, 2, 2), 2.0, requires_grad=True)
    predicted = torch.tensor(
        [
            [[0.7071068, 0.7071068], [0.7071068, 0.7071068]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.7071068, 0.7071068], [0.7071068, 0.7071068]],
        ],
        requires_grad=True,
    )
    package = SimpleNamespace(depth=depth, normal_world=predicted)
    target = torch.zeros(3, 2, 2)
    target[2] = 1.0
    evidence = {
        "moge3_depth_m": torch.full((1, 2, 2), 2.0),
        "moge3_valid_mask": torch.ones(1, 2, 2),
        "moge3_refinement_log_depth_std": torch.zeros(1, 2, 2),
        "moge3_refinement_final_delta_log_depth": torch.zeros(1, 2, 2),
        "moge3_normal_direct_camera": target.clone(),
        "moge3_normal_depth_exact_k_camera": target.clone(),
        "moge3_depth_normal_valid_mask": torch.ones(1, 2, 2),
        "source_bitmask": torch.tensor(
            [[[2, 0], [0, 0]]], dtype=torch.uint8
        ),
        "rho_mean": torch.tensor([[[0.5, 0.0], [0.0, 0.0]]]),
        "rho_variance": torch.tensor([[[0.01, 0.0], [0.0, 0.0]]]),
        "support_view_count": torch.tensor(
            [[[1.0, 0.0], [0.0, 0.0]]]
        ),
    }
    args = SimpleNamespace(
        geometry_weight=0.12,
        plane_weight=0.10,
        normal_weight=0.04,
        ordinal_weight=0.015,
        moge3_rigid_depth_weight=0.0,
        moge3_rigid_normal_weight=0.02,
    )
    loss, values = _geometry_losses(
        package,
        evidence,
        torch.ones(2, 2),
        args,
        world_from_camera_rotation=torch.eye(3),
    )
    assert values["moge3_rigid_normal"] > 0.0
    assert values["moge3_rigid_normal_pixels"] == 3
    assert values["moge3_rigid_normal_mean_precision"] == pytest.approx(1.0)
    loss.backward()
    assert predicted.grad[:, 0, 0].abs().sum() == 0
    assert predicted.grad[:, 0, 1].abs().sum() > 0
    assert depth.grad.abs().sum() == 0


def test_projected_occluded_rigid_factor_updates_depth_and_coverage_not_rgb():
    depth = torch.tensor([[[2.0, 0.0], [4.0, 0.0]]], requires_grad=True)
    alpha = torch.tensor(
        [[[0.25, 0.0], [0.50, 0.0]]], requires_grad=True
    )
    package = SimpleNamespace(surface_depth=depth, surface_alpha=alpha)
    task = {
        "projected_rigid_depth": torch.tensor([[3.0, 0.0], [4.0, 0.0]]),
        "p_projected_rigid_geometry": torch.tensor(
            [[0.8, 0.0], [0.6, 0.0]]
        ),
        "p_distortion_valid": torch.ones(2, 2),
    }
    loss, audit = _projected_occluded_rigid_geometry_loss(package, task)

    assert audit["supported_pixels"] == 2
    assert audit["depth_matched_pixels"] == 2
    assert audit["depth"] > 0
    assert audit["coverage"] > 0
    loss.backward()
    assert depth.grad[0, 0, 0] != 0
    assert depth.grad[0, 1, 0] == 0
    assert alpha.grad[0, 0, 0] != 0
    assert alpha.grad[0, 0, 1] == 0


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


def test_static_child_color_refresh_blocks_noncanonical_sequence_rgb():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.0, 0.0, 2.0]]),
        "scales": torch.full((1, 3), 0.05),
        "colors": torch.full((1, 3), 0.25),
        "opacities": torch.full((1, 1), 0.2),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "layer_role": torch.tensor([0], dtype=torch.int8),
        "observation_camera_ids": torch.tensor(
            [[7, 8, 9]], dtype=torch.int32
        ),
        "observation_uv": torch.full((1, 3, 2), 0.5),
        "observation_depth": torch.full((1, 3), 2.0),
    }
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)
    cameras = {
        7: _solid_refresh_camera(7, [1.0, 0.0, 0.0]),
        8: _solid_refresh_camera(8, [0.0, 0.0, 1.0]),
        9: _solid_refresh_camera(9, [0.0, 1.0, 0.0]),
    }
    cameras[7].image_name = "seq1__frame00001"
    cameras[8].image_name = "seq2__frame00001"
    cameras[9].image_name = "seq1__frame00002"

    audit = _refresh_split_child_owner_colors(
        foliage,
        child_start=0,
        owner_camera_ids=torch.tensor([-1]),
        view_by_camera_id=cameras,
        canonical_rgb_sequence="seq1",
    )

    dc_rgb = foliage.features[0, 0] * 0.28209479177387814 + 0.5
    torch.testing.assert_close(dc_rgb, torch.tensor([0.5, 0.5, 0.0]))
    assert audit["canonical_refreshed_children"] == 1
    assert audit["owner_camera_count"] == 2
    assert audit["canonical_rgb_sequence"] == "seq1"
    assert audit["noncanonical_rgb_camera_count_blocked"] == 1
    assert audit["noncanonical_rgb_children_blocked"] == 1


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


def test_static_child_color_refresh_deduplicates_explicit_and_support_camera():
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
        "observation_depth": torch.tensor([[2.0]]),
        "support_camera_ids": torch.tensor([[7, 8]], dtype=torch.int32),
        "support_sequence_count": torch.tensor([2], dtype=torch.int16),
    }
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)
    _refresh_split_child_owner_colors(
        foliage, child_start=0, owner_camera_ids=torch.tensor([-1]),
        view_by_camera_id={
            7: _solid_refresh_camera(7, [0.2, 0.4, 0.6]),
            8: _solid_refresh_camera(8, [0.6, 0.4, 0.2]),
        },
    )
    dc_rgb = foliage.features[0, 0] * 0.28209479177387814 + 0.5
    torch.testing.assert_close(dc_rgb, torch.tensor([0.4, 0.4, 0.4]))


@pytest.mark.parametrize("hit_depth,canopy,weight,transient,accepted", [
    (2.0, 1.0, 1.0, 0.0, True),
    (4.0, 1.0, 1.0, 0.0, False),
    (2.0, 0.0, 1.0, 0.0, False),
    (2.0, 1.0, 0.0, 0.0, False),
    (2.0, 1.0, 1.0, 1.0, False),
])
def test_measured_child_color_uses_calibrated_hit_and_observed_canopy(
    hit_depth, canopy, weight, transient, accepted,
):
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
        "observation_depth": torch.tensor([[2.0]]),
        "support_camera_ids": torch.tensor([[7, 8]], dtype=torch.int32),
        "support_sequence_count": torch.tensor([2], dtype=torch.int16),
    }
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)
    cameras = {7: _solid_refresh_camera(7, [0.2, 0.4, 0.6]),
               8: _solid_refresh_camera(8, [0.6, 0.4, 0.2])}
    for key, view in cameras.items():
        view.image_name = f"seq2/{key}"
    field = lambda value: torch.full((1, 4, 4), value)
    geometry = SimpleNamespace(inverse_root=None, fields=lambda *a, **k: {
        "moge3_depth_m": field(1.0),
        "moge3_canopy_depth_m": field(hit_depth),
        "moge3_valid_mask": field(1.0),
        "moge3_refinement_log_depth_std": field(0.0),
        "moge3_refinement_final_delta_log_depth": field(0.0),
    })
    task_canopy = field(canopy)[0]
    task_canopy[0] = 0  # Native task fields are HxW; first row is not the image.
    task = SimpleNamespace(fields=lambda *a, **k: {
        "p_canopy": task_canopy, "w_rgb": field(weight)[0],
        "p_transient": field(transient)[0],
    })
    audit = _refresh_split_child_owner_colors(
        foliage, child_start=0, owner_camera_ids=torch.tensor([-1]),
        view_by_camera_id=cameras, geometry_evidence=geometry,
        measured_canopy_task_fields=task,
    )
    dc_rgb = foliage.features[0, 0] * 0.28209479177387814 + 0.5
    torch.testing.assert_close(dc_rgb, torch.full((3,), 0.4 if accepted else 0.25))
    assert audit["measured_canopy_depth_refreshed_children"] == int(accepted)
    assert audit["canonical_low_support_dc_only_children"] == int(not accepted)


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


def test_candidate_interval_factor_returns_exact_source_components():
    evidence = _interval_evidence(1)
    foliage = _interval_foliage(3.05)
    total, audit, components = evidence.interval_factor(
        _interval_camera(), foliage, return_loss_components=True
    )

    torch.testing.assert_close(
        components["free"],
        components["confirmed_free"] + components["hit_prehit"],
    )
    torch.testing.assert_close(
        total, components["free"] + components["hit"]
    )
    assert audit["hit_rays"] == 1
    assert float(components["confirmed_free"]) == 0.0

    weight = 0.37
    total_gradient = torch.autograd.grad(
        weight * total,
        foliage.opacity_logits,
        retain_graph=True,
    )[0]
    confirmed_free_gradient = None
    if components["confirmed_free"].requires_grad:
        confirmed_free_gradient = torch.autograd.grad(
            weight * components["confirmed_free"],
            foliage.opacity_logits,
            retain_graph=True,
            allow_unused=True,
        )[0]
    if confirmed_free_gradient is None:
        confirmed_free_gradient = torch.zeros_like(foliage.opacity_logits)
    hit_likelihood_gradient = torch.autograd.grad(
        weight * (components["hit_prehit"] + components["hit"]),
        foliage.opacity_logits,
    )[0]
    torch.testing.assert_close(
        total_gradient,
        confirmed_free_gradient + hit_likelihood_gradient,
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    # A valid hit has a small Gaussian tail before its interval. Keeping that
    # tail in the same likelihood prevents a floating-point nonzero from being
    # misclassified as an independent free-space contradiction.
    assert float(components["hit_prehit"]) > 0.0
    assert float(hit_likelihood_gradient) < 0.0


def test_candidate_interval_anneals_only_hit_opacity_not_geometry_or_free():
    models = [_interval_foliage(3.05) for _ in range(3)]
    scales = (0.0, 0.5, 1.0)
    losses = []
    audits = []
    for model, scale in zip(models, scales):
        evidence = _interval_evidence(1)
        loss, audit = evidence.interval_factor(
            _interval_camera(),
            model,
            hit_opacity_gradient_scale=scale,
        )
        loss.backward()
        losses.append(loss.detach())
        audits.append(audit)

    # The forward likelihood and all geometric derivatives are identical.
    torch.testing.assert_close(losses[0], losses[1])
    torch.testing.assert_close(losses[0], losses[2])
    torch.testing.assert_close(models[0].xyz.grad, models[2].xyz.grad)
    torch.testing.assert_close(
        models[0].log_scales.grad, models[2].log_scales.grad
    )
    # Only the hit opacity derivative interpolates. The full-strength
    # pre-hit free-space derivative is the common offset at scale zero.
    zero = models[0].opacity_logits.grad
    half = models[1].opacity_logits.grad
    full = models[2].opacity_logits.grad
    torch.testing.assert_close(half, zero + 0.5 * (full - zero))
    assert audits[0]["hit_opacity_gradient_scale"] == 0.0
    assert audits[2]["hit_opacity_gradient_scale"] == 1.0


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


def test_ownerless_verified_canonical_retains_optical_existence_gradient():
    strong_evidence = _ownerless_interval_evidence(confidence=1.0)
    weak_evidence = _ownerless_interval_evidence(confidence=0.1)
    strong = _interval_foliage(3.0)
    weak = _interval_foliage(3.0)
    # This support is immutable evidence established before either dense
    # ownerless observation; consuming the ray must not promote the lineage.
    strong.support_sequence_count.fill_(2)
    weak.support_sequence_count.fill_(2)

    strong_loss, strong_audit = strong_evidence.interval_factor(
        _interval_camera(), strong
    )
    weak_loss, weak_audit = weak_evidence.interval_factor(
        _interval_camera(), weak
    )
    strong_loss.backward()
    weak_loss.backward()

    assert weak_audit["ownerless_canonical_hit_rays"] == 1
    assert weak_audit["verified_ownerless_hit_rays"] == 1
    assert weak_audit["hit_optical_existence"] > 0
    assert float(weak.opacity_logits.grad) < 0
    torch.testing.assert_close(
        weak.opacity_logits.grad,
        strong.opacity_logits.grad,
        rtol=0.03,
        atol=1.0e-5,
    )


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


def test_static_detail_ray_prefit_starts_in_bootstrap():
    assert _static_detail_ray_trainable("canonical_bootstrap")
    assert _static_detail_ray_trainable("topology")
    assert _static_detail_ray_trainable("static_foliage")
    assert _static_detail_ray_trainable("canonical_polish")


def test_static_detail_isolated_routes_geometry_and_owned_opacity_not_sh():
    ownership = torch.tensor([1.0, 0.0, 1.0])
    appearance_permission = torch.tensor([1.0, 0.35, 1.0])
    optical_permission = torch.tensor([1.0, 0.35, 0.0])
    geometry, appearance, opacity = _static_detail_isolated_gradient_gates(
        ownership, appearance_permission, optical_permission
    )
    assert geometry is ownership
    torch.testing.assert_close(appearance, torch.zeros_like(ownership))
    torch.testing.assert_close(opacity, optical_permission)


def test_static_detail_isolated_requires_explicit_optical_owner():
    ownership = torch.tensor([1.0])
    with np.testing.assert_raises_regex(ValueError, "explicit optical gate"):
        _static_detail_isolated_gradient_gates(ownership, ownership)


def test_static_detail_positive_sequence_gets_soft_geometry_not_topology():
    class Foliage:
        xyz = torch.zeros(3, 3)
        static_leaf_mask = torch.ones(3, dtype=torch.bool)
        support_camera_ids = torch.tensor([[0, -1], [1, -1], [2, -1]])
        verified_camera_ids = torch.tensor([[3, -1], [4, -1], [5, -1]])
        verification_state = torch.tensor(
            [VERIFICATION_VERIFIED] * 3, dtype=torch.int8
        )
        verified_camera_count = torch.tensor([2, 2, 1], dtype=torch.int16)
        verified_sequence_count = torch.ones(3, dtype=torch.int16)
        proposal_kind = torch.full((3,), PROPOSAL_NONE, dtype=torch.int8)
        dynamic_leaf_mask = torch.zeros(3, dtype=torch.bool)

        def __len__(self):
            return 3

    # Cameras 0/1/3/4 are sequence 0; cameras 2/5 are sequence 1.
    sequence_lookup = torch.tensor([0, 0, 1, 0, 0, 1], dtype=torch.int16)
    exact = torch.tensor([1.0, 0.0, 0.0])
    geometry = _static_detail_same_sequence_geometry_gate(
        Foliage(), 1, sequence_lookup, exact
    )
    torch.testing.assert_close(
        geometry,
        torch.tensor(
            [1.0, STATIC_DETAIL_SAME_SEQUENCE_GEOMETRY_WEIGHT, 0.0]
        ),
    )
    widened = _static_detail_same_sequence_geometry_gate(
        Foliage(), 1, sequence_lookup, exact, fallback_weight=1.0
    )
    torch.testing.assert_close(widened, torch.tensor([1.,1.,0.]))
    # This permission change never modifies the independent topology owner.
    torch.testing.assert_close(exact, torch.tensor([1.,0.,0.]))
    from scripts.train_unified_outdoor_teacher import _static_detail_refinement_gate
    torch.testing.assert_close(_static_detail_refinement_gate(Foliage(),exact),exact)


def test_static_volume_isolated_only_refines_verified_exact_detail():
    class Foliage:
        xyz = torch.zeros(5, 3)
        opacities = torch.ones(5)
        static_leaf_mask = torch.tensor([False, False, True, True, True])
        verification_state = torch.tensor(
            [
                VERIFICATION_VERIFIED,
                VERIFICATION_VERIFIED,
                VERIFICATION_VERIFIED,
                VERIFICATION_UNVERIFIED,
                VERIFICATION_VERIFIED,
            ],
            dtype=torch.int8,
        )
        proposal_kind = torch.full((5,), PROPOSAL_NONE, dtype=torch.int8)
        verified_camera_count = torch.tensor([2, 2, 2, 2, 1])
        verified_sequence_count = torch.ones(5, dtype=torch.int16)

        def __len__(self):
            return 5

    exact_owner = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    appearance_owner = torch.tensor([1.0, 1.0, 0.35, 0.35, 0.35])
    geometry, appearance, opacity = _static_volume_isolated_gradient_gates(
        Foliage(), exact_owner, appearance_owner
    )
    expected = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
    expected_appearance = torch.tensor([0.0, 0.0, 0.35, 0.0, 0.0])
    torch.testing.assert_close(geometry, expected)
    torch.testing.assert_close(appearance, expected_appearance)
    torch.testing.assert_close(opacity, torch.zeros_like(expected))


def test_static_detail_color_schedule_expands_positive_evidence_sequences():
    class View:
        def __init__(self, camera_id):
            self.colmap_id = camera_id

    class Foliage:
        static_leaf_mask = torch.tensor([True, True, True, False])
        dynamic_leaf_mask = torch.zeros(4, dtype=torch.bool)
        verification_state = torch.tensor(
            [
                VERIFICATION_VERIFIED,
                VERIFICATION_UNVERIFIED,
                VERIFICATION_VERIFIED,
                VERIFICATION_VERIFIED,
            ],
            dtype=torch.int8,
        )
        proposal_kind = torch.full((4,), PROPOSAL_NONE, dtype=torch.int8)
        verified_camera_count = torch.tensor([2, 2, 1, 3])
        verified_sequence_count = torch.tensor([1, 1, 1, 2])
        support_camera_ids = torch.tensor(
            [[1, -1], [5, -1], [5, -1], [5, -1]]
        )
        verified_camera_ids = torch.tensor(
            [[3, -1], [5, -1], [5, -1], [5, -1]]
        )

    views = [View(camera_id) for camera_id in range(6)]
    # Cameras 1/2 are sequence 0, 3/4 sequence 1, and 5 sequence 2.
    lookup = torch.tensor([-1, 0, 0, 1, 1, 2], dtype=torch.int64)

    indices, sequences = _static_detail_evidence_sequence_view_indices(
        views,
        Foliage(),
        lookup,
        [1, 2, 3, 4, 5],
    )

    # Only row zero is verified multiview static detail. Its sparse camera
    # table names 1 and 3, but every canopy camera in those sequences must be
    # scheduled. Unverified, single-camera and non-detail rows add nothing.
    assert indices == [1, 2, 3, 4]
    assert sequences == {0, 1}


def test_static_replacement_schedule_uses_only_paired_real_evidence_cameras():
    class View:
        def __init__(self, camera_id):
            self.colmap_id = camera_id

    class Foliage:
        replacement_group = torch.tensor([7, 8, 7, 8, 9, -1])
        persistent_envelope_mask = torch.tensor(
            [True, True, False, False, False, False]
        )
        static_leaf_mask = torch.tensor(
            [False, False, True, True, True, True]
        )
        dynamic_leaf_mask = torch.zeros(6, dtype=torch.bool)
        verification_state = torch.tensor(
            [1, 0, 1, 1, 1, 1], dtype=torch.int8
        )
        verified_camera_count = torch.tensor([2, 0, 2, 2, 2, 2])
        verified_sequence_count = torch.tensor([1, 0, 2, 2, 2, 2])
        proposal_kind = torch.full((6,), PROPOSAL_NONE, dtype=torch.int8)
        support_camera_ids = torch.tensor(
            [[-1], [-1], [1], [2], [3], [4]]
        )
        verified_camera_ids = torch.tensor(
            [[-1], [-1], [5], [5], [5], [5]]
        )

    indices, camera_ids, paired_rows = (
        _static_replacement_evidence_view_indices(
            [View(value) for value in range(6)],
            Foliage(),
            [0, 1, 2, 3, 4],
        )
    )
    # Only group 7 has a verified envelope. Group 8's envelope is
    # unverified, group 9 has no envelope, and the final detail is ownerless.
    assert paired_rows == 1
    assert camera_ids == {1, 5}
    # Exact camera five remains scheduled even though the coarse global
    # canopy filter omitted it; the row-level witness is the stronger fact.
    assert indices == [1, 5]


def test_static_volume_intrinsic_color_normalizes_low_alpha_once():
    background = torch.ones(3)
    alpha = torch.tensor([[[0.10, 0.005, 0.0025]]])
    color = torch.tensor(
        [[[0.2, 0.4, 0.6]], [[0.3, 0.5, 0.7]], [[0.4, 0.6, 0.8]]],
        requires_grad=True,
    )
    render = color * alpha + (1.0 - alpha) * background[:, None, None]
    intrinsic, weight = _static_volume_intrinsic_color_inputs(
        render,
        alpha,
        background,
        torch.ones(1, 3),
    )

    torch.testing.assert_close(intrinsic[:, :, :2], color[:, :, :2])
    torch.testing.assert_close(weight, torch.tensor([[1.0, 1.0, 0.5]]))
    intrinsic.sum().backward()
    # At and above the representation cull scale, normalized colour receives
    # an alpha-independent unit gradient instead of the old alpha^2 decay.
    torch.testing.assert_close(
        color.grad[:, :, :2], torch.ones_like(color.grad[:, :, :2])
    )
    assert STATIC_VOLUME_INTRINSIC_ALPHA_FLOOR == 0.005


def test_static_stage3_rgb_routes_envelope_only_to_appearance():
    class Foliage:
        xyz = torch.zeros(4, 3)
        opacities = torch.ones(4)
        persistent_envelope_mask = torch.tensor(
            [False, True, False, True]
        )

        def __len__(self):
            return 4

    ownership = torch.tensor([1.0, 1.0, 0.0, 1.0])
    appearance_permission = torch.tensor([1.0, 1.0, 0.35, 1.0])
    geometry, appearance, opacity = _static_stage_rgb_gradient_gates(
        Foliage(),
        ownership,
        detail_stage_active=True,
        appearance_gate=appearance_permission,
    )
    torch.testing.assert_close(
        geometry, torch.tensor([1.0, 0.0, 0.0, 0.0])
    )
    torch.testing.assert_close(opacity, geometry)
    torch.testing.assert_close(appearance, appearance_permission)
    # The caller's calibrated ownership tensor is immutable.
    torch.testing.assert_close(
        ownership, torch.tensor([1.0, 1.0, 0.0, 1.0])
    )


def test_static_stage2_rgb_keeps_envelope_geometry_and_mass_trainable():
    class Foliage:
        xyz = torch.zeros(2, 3)
        opacities = torch.ones(2)
        persistent_envelope_mask = torch.tensor([True, False])

        def __len__(self):
            return 2

    ownership = torch.tensor([1.0, 0.0])
    geometry, appearance, opacity = _static_stage_rgb_gradient_gates(
        Foliage(), ownership, detail_stage_active=False
    )
    assert geometry is ownership
    assert appearance is ownership
    assert opacity is ownership


def test_periodic_camera_schedule_consumes_a_contiguous_evidence_epoch():
    schedule = np.asarray([3, 1, 0, 2], dtype=np.int64)
    executed_steps = [1, 3, 5, 7]
    consumed = [
        int(schedule[_periodic_schedule_index(step, 2)])
        for step in executed_steps
    ]
    assert consumed == schedule.tolist()
    staggered_steps = [0, 2, 4, 6]
    staggered = [
        int(
            schedule[
                _periodic_schedule_index(step, 2, phase_offset=1)
            ]
        )
        for step in staggered_steps
    ]
    assert staggered == schedule.tolist()
    with pytest.raises(ValueError, match="not a scheduled"):
        _periodic_schedule_index(0, 2)
    with pytest.raises(ValueError, match="positive"):
        _periodic_schedule_index(0, 0)


def test_static_ray_prefit_uses_symmetric_positive_and_free_evidence():
    class Foliage:
        xyz = torch.zeros(4, 3)
        dynamic_leaf_mask = torch.zeros(4, dtype=torch.bool)
        static_leaf_mask = torch.tensor([False, True, True, True])
        support_camera_ids = torch.tensor(
            [[-1, -1], [0, -1], [1, 2], [1, 2]], dtype=torch.int32
        )
        verified_camera_ids = torch.full((4, 2), -1, dtype=torch.int32)
        support_sequence_count = torch.tensor([2, 1, 1, 1])
        verified_camera_count = torch.tensor([2, 1, 2, 2])
        verified_sequence_count = torch.tensor([2, 1, 2, 2])
        verification_state = torch.tensor(
            [
                VERIFICATION_VERIFIED,
                VERIFICATION_VERIFIED,
                VERIFICATION_VERIFIED,
                VERIFICATION_UNVERIFIED,
            ],
            dtype=torch.int8,
        )
        proposal_kind = torch.full((4,), PROPOSAL_NONE, dtype=torch.int8)

        def __len__(self):
            return 4

    args = SimpleNamespace(
        reconstruction_target="static",
        static_detail_canonical_ownership=True,
    )
    active = torch.tensor([True, False, False, False])
    lookup = torch.tensor([0, 1, 1], dtype=torch.int16)
    free, hit = _static_ray_candidate_masks(
        args,
        Foliage(),
        active,
        phase="topology",
        camera_id=0,
        camera_sequence_lookup=lookup,
    )
    # A global verification count is not a visibility table. Camera zero may
    # alter only its exact row-one support; rows two/three belong to cameras
    # one/two and remain read-only in this view.
    assert torch.equal(free, torch.tensor([True, True, False, False]))
    assert torch.equal(hit, torch.tensor([True, True, False, False]))
    assert torch.equal(
        _static_detail_consensus_hit_gate(Foliage()),
        torch.tensor([False, False, True, False]),
    )

    _, bootstrap_hit = _static_ray_candidate_masks(
        args,
        Foliage(),
        active,
        phase="canonical_bootstrap",
        camera_id=0,
        camera_sequence_lookup=lookup,
    )
    # Bootstrap now pre-fits the same exact-owner/verified static detail as
    # topology.  This prevents the first evidence epoch from being consumed
    # while hidden detail remains frozen.
    assert torch.equal(bootstrap_hit, hit)

    noncanonical_free, noncanonical_hit = _static_ray_candidate_masks(
        args,
        Foliage(),
        active,
        phase="canonical_polish",
        camera_id=1,
        camera_sequence_lookup=lookup,
        canonical_sequence_index=0,
    )
    # The persistent envelope remains cross-sequence calibrated, but a
    # different traversal can neither create nor delete canonical leaf
    # detail—even for a row with multi-sequence consensus metadata.
    assert torch.equal(
        noncanonical_free, torch.tensor([True, False, False, False])
    )
    assert torch.equal(
        noncanonical_hit, torch.tensor([True, False, False, False])
    )

    nonsnapshot_free, nonsnapshot_hit = _static_ray_candidate_masks(
        args,
        Foliage(),
        active,
        phase="canonical_polish",
        camera_id=0,
        camera_sequence_lookup=lookup,
        canonical_sequence_index=0,
        canonical_snapshot_camera_ids={2},
    )
    assert torch.equal(
        nonsnapshot_free, torch.tensor([True, False, False, False])
    )
    assert torch.equal(
        nonsnapshot_hit, torch.tensor([True, False, False, False])
    )


def test_verification_debt_capacity_is_continuous_and_shared():
    assert _verification_debt_capacity_scale(
        0.04, soft_fraction=0.05, hard_fraction=0.12
    ) == pytest.approx(1.0)
    midpoint = _verification_debt_capacity_scale(
        0.085, soft_fraction=0.05, hard_fraction=0.12
    )
    assert midpoint == pytest.approx(0.5)
    assert _verification_debt_capacity_scale(
        0.13, soft_fraction=0.05, hard_fraction=0.12
    ) == pytest.approx(0.0)
    assert _verification_debt_capacity_scale(
        0.13,
        soft_fraction=0.05,
        hard_fraction=0.12,
        minimum_scale=0.02,
    ) == pytest.approx(0.02)
    with pytest.raises(ValueError, match="0 <= soft < hard <= 1"):
        _verification_debt_capacity_scale(
            0.1, soft_fraction=0.2, hard_fraction=0.1
        )


def test_verification_debt_counts_receivers_and_unresolved_transactions():
    class DebtFixture(SimpleNamespace):
        def __len__(self):
            return len(self.xyz)

    foliage = DebtFixture(
        xyz=torch.zeros(6, 3),
        verification_state=torch.tensor(
            [
                VERIFICATION_VERIFIED,
                VERIFICATION_UNVERIFIED,
                VERIFICATION_FACTORIZED_RECEIVER,
                VERIFICATION_MEASURED_SINGLE,
                VERIFICATION_VERIFIED,
                VERIFICATION_REJECTED,
            ],
            dtype=torch.int8,
        ),
        proposal_kind=torch.tensor(
            [
                PROPOSAL_NONE,
                PROPOSAL_RAY_BIRTH,
                PROPOSAL_NONE,
                PROPOSAL_NONE,
                PROPOSAL_SPLIT,
                PROPOSAL_NONE,
            ],
            dtype=torch.int8,
        ),
    )
    debt = _verification_debt_mask(foliage)

    assert torch.equal(
        debt,
        torch.tensor([False, True, True, False, True, False]),
    )
    fraction = float(debt.float().mean())
    assert fraction == pytest.approx(0.5)
    assert _verification_debt_capacity_scale(
        fraction,
        soft_fraction=0.05,
        hard_fraction=0.12,
        minimum_scale=0.0,
    ) == pytest.approx(0.0)


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
    args.mature_handoff_surface_policy = "atlas_residual"
    assert _surface_topology_active(9, args)
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
        "p_unknown_ownership": torch.zeros(2, 2),
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


def _canonical_occlusion_fixture(*, surface_in_front: bool = False):
    target = torch.full((3, 2, 2), 0.2)
    surface = torch.full((3, 2, 2), 0.8, requires_grad=True)
    # Intrinsic foliage colour is 0.2 at alpha 0.30 on the renderer's white
    # background: V_white = .30*.20 + .70*1 = .76.  The hypothetical
    # volume-front composite is therefore .76 + .70*(.80-1) = .62.
    # This fixture is physically self-consistent and does not hand-inject a
    # mixed contribution behind an opaque surface.
    volume_white = torch.full((3, 2, 2), 0.76, requires_grad=True)
    alpha = torch.full((1, 2, 2), 0.30, requires_grad=True)
    surface_alpha = torch.ones(1, 2, 2, requires_grad=True)
    surface_depth = torch.full(
        (1, 2, 2), 2.0 if surface_in_front else 4.0,
        requires_grad=True,
    )
    volume_depth = torch.full(
        (1, 2, 2), 4.0 if surface_in_front else 2.0,
        requires_grad=True,
    )
    task = {
        "p_canopy": torch.ones(2, 2),
        "p_canopy_core": torch.ones(2, 2),
        "p_transient": torch.zeros(2, 2),
        "p_boundary_uncertain": torch.zeros(2, 2),
        "p_unknown_ownership": torch.zeros(2, 2),
        "w_rgb": torch.ones(2, 2),
    }
    return (
        volume_white,
        surface,
        target,
        alpha,
        surface_alpha,
        surface_depth,
        volume_depth,
        task,
    )


def test_moge3_depth_query_bounds_are_continuous_and_fail_closed():
    depth = torch.tensor([[[4.0, 8.0, float("nan")]]])
    evidence = {
        "moge3_depth_m": depth,
        "moge3_valid_mask": torch.tensor([[[1.0, 1.0, 1.0]]]),
        "moge3_refinement_log_depth_std": torch.tensor(
            [[[0.01, 0.08, 0.01]]]
        ),
        "moge3_refinement_final_delta_log_depth": torch.tensor(
            [[[0.0, 0.02, 0.0]]]
        ),
    }
    prehit_bounds, hit_bounds, valid, reliability, audit = (
        _moge3_depth_query_bounds(
        evidence, global_log_scale_sigma=0.05
        )
    )

    assert audit["contract"] == MOGE3_CANOPY_OPTICAL_CONTRACT
    assert audit["valid_pixels"] == 2
    assert valid.tolist() == [[True, True, False]]
    assert torch.isnan(prehit_bounds[:, 0, 2]).all()
    assert torch.isnan(hit_bounds[:, 0, 2]).all()
    assert torch.all(prehit_bounds[0, 0, :2] < depth[0, 0, :2])
    assert torch.allclose(prehit_bounds[1, 0, :2], depth[0, 0, :2])
    assert torch.all(hit_bounds[0, 0, :2] < depth[0, 0, :2])
    assert torch.all(hit_bounds[1, 0, :2] > depth[0, 0, :2])
    assert (hit_bounds[1, 0, 1] - hit_bounds[0, 0, 1]) > (
        hit_bounds[1, 0, 0] - hit_bounds[0, 0, 0]
    )
    assert audit["mean_hit_metric_half_width"] <= 0.400001
    assert reliability[0, 0] > reliability[0, 1] > 0
    assert reliability[0, 2].item() == 0.0


def test_moge3_global_scale_uncertainty_never_thickens_positive_hit():
    depth = torch.full((1, 1, 1), 20.0)
    evidence = {
        "moge3_depth_m": depth,
        "moge3_valid_mask": torch.ones_like(depth),
        "moge3_refinement_log_depth_std": torch.full_like(depth, 0.03),
        "moge3_refinement_final_delta_log_depth": torch.zeros_like(depth),
    }
    low = _moge3_depth_query_bounds(
        evidence, global_log_scale_sigma=0.0
    )
    high = _moge3_depth_query_bounds(
        evidence, global_log_scale_sigma=0.20
    )
    low_prehit, low_hit, _, low_reliability, _ = low
    high_prehit, high_hit, _, high_reliability, audit = high
    # More global gauge uncertainty makes only the negative free-space bound
    # conservative and reduces evidence weight. Positive support stays local.
    assert high_prehit[0].item() < low_prehit[0].item()
    assert torch.equal(high_hit, low_hit)
    assert high_reliability.item() < low_reliability.item()
    assert audit["mean_hit_metric_half_width"] <= 0.400001
    assert audit["mean_prehit_metric_offset"] > 1.0


def test_moge3_canopy_optical_loss_grows_hit_retires_prehit_and_ignores_invalid():
    prehit = torch.tensor(
        [[[0.2, 0.1, 0.8]]], requires_grad=True
    )
    hit = torch.tensor(
        [[[0.1, 0.7, 0.2]]], requires_grad=True
    )
    valid = torch.tensor([[True, True, False]])
    reliability = torch.ones(1, 3)
    task = {
        "p_canopy": torch.tensor([[1.0, 1.0, 1.0]]),
        "p_canopy_core": torch.tensor([[1.0, 1.0, 1.0]]),
        "p_rigid": torch.zeros(1, 3),
        "p_distortion_valid": torch.ones(1, 3),
        "p_transient": torch.zeros(1, 3),
        "w_rgb": torch.ones(1, 3),
    }
    loss, audit = _moge3_canopy_optical_loss(
        prehit, hit, task, valid, reliability
    )
    loss.backward()

    assert audit["contract"] == MOGE3_CANOPY_OPTICAL_CONTRACT
    assert audit["behind_interval_gradient"] is False
    # Optimizer descent reduces pre-hit alpha and increases deficient hit
    # alpha. The second hit remains below the 0.75 core target, so it also
    # receives a smaller but non-zero growth request.
    assert prehit.grad[0, 0, 0] > 0
    assert prehit.grad[0, 0, 1] > 0
    assert hit.grad[0, 0, 0] < 0
    assert hit.grad[0, 0, 1] < 0
    assert prehit.grad[0, 0, 2].item() == 0.0
    assert hit.grad[0, 0, 2].item() == 0.0


def test_birth_budget_cannot_be_monopolized_by_one_high_score_patch():
    rows = torch.tensor([0, 0, 1, 1, 0, 6, 6])
    columns = torch.tensor([0, 1, 0, 1, 6, 0, 6])
    score = torch.tensor([1., .99, .98, .97, .8, .8, .8])
    chosen = _spatially_stratified_proposal_indices(
        rows, columns, score, height=8, width=8, budget=4,
    )
    assert chosen.tolist() == [0, 4, 5, 6]
    # A sparse tile layout still fills the budget, without duplicate rays.
    chosen = _spatially_stratified_proposal_indices(
        rows[:4], columns[:4], score[:4], height=8, width=8, budget=3,
    )
    assert chosen.tolist() == [0, 1, 2]


def test_seeded_birth_sampling_explores_tiles_without_mutating_evidence_or_rng():
    rows = torch.arange(8).repeat_interleave(8)
    columns = torch.arange(8).repeat(8)
    score = torch.linspace(.5,1.,64)
    original = score.clone()
    rng = torch.get_rng_state().clone()
    def sample(seed):
        return _spatially_stratified_proposal_indices(rows,columns,score,
            height=8,width=8,budget=4,sampling_seed=seed)
    torch.testing.assert_close(sample(17),sample(17))
    union = set()
    for seed in range(32):
        chosen = sample(seed)
        assert len(chosen.unique()) == 4
        assert len(((rows[chosen]//4)*2+columns[chosen]//4).unique()) == 4
        union.update(chosen.tolist())
    assert len(union) > 32
    torch.testing.assert_close(score,original)
    torch.testing.assert_close(torch.get_rng_state(),rng)


def test_moge3_auxiliary_camera_is_bound_to_loss_birth_render_and_ledger():
    import ast
    import inspect
    import scripts.train_unified_outdoor_teacher as trainer
    body = ast.parse(inspect.getsource(trainer.main))
    calls = [node for node in ast.walk(body) if isinstance(node,ast.Call)]
    def named(name):
        return [node for node in calls if isinstance(node.func,ast.Name) and node.func.id == name]
    proposals = named("_moge3_uncovered_hit_proposals")
    assert len(proposals) == 1
    assert ast.unparse(proposals[0].args[0]) == "moge3_view"
    assert ast.unparse(proposals[0].args[3]) == "moge3_task"
    losses = named("_moge3_canopy_optical_loss")
    assert len(losses) == 1 and ast.unparse(losses[0].args[2]) == "moge3_task"
    ledger = [node for node in named("_update_persistent_positive_optical_demand")
              if ast.unparse(node.args[2]) == "moge3_canopy_optical_growth_gradient"]
    assert len(ledger) == 1
    camera = next(k.value for k in ledger[0].keywords if k.arg == "camera_id")
    assert "moge3_view.colmap_id" in ast.unparse(camera)
    render_count = 0
    for node in ast.walk(body):
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call):
            targets = {target.id for target in node.targets if isinstance(target,ast.Name)}
            if targets & {"moge3_canopy_optical_package_prehit","moge3_canopy_optical_package_hit",
                           "moge3_canopy_optical_package_coverage","verification_package"}:
                assert ast.unparse(node.value.args[0]) == "moge3_view"
                render_count += 1
    assert render_count == 4


def test_moge3_hit_coverage_audit_excludes_invalid_and_nontree_pixels():
    hit = torch.tensor([[0., 0.005, 0.2, 0.8, 0., 0.]])
    task = {
        "p_canopy": torch.tensor([[1., 1., 1., 1., 1., 0.]]),
        "p_canopy_core": torch.ones_like(hit),
        "p_rigid": torch.zeros_like(hit),
    }
    _, audit = _moge3_canopy_optical_loss(
        torch.zeros_like(hit), hit, task,
        torch.tensor([[True, True, True, True, False, True]]),
        torch.ones_like(hit),
    )
    assert audit["hit_supported_pixels"] == 4
    assert audit["hit_zero_coverage_pixels"] == 1
    assert audit["hit_positive_below_001_pixels"] == 1
    assert audit["hit_below_target_pixels"] == 3
    assert audit["hit_target_met_pixels"] == 1


def test_moge3_hit_owner_uses_current_metric_ray_for_persistent_detail():
    foliage = SimpleNamespace(
        static_leaf_mask=torch.tensor(
            [True, True, True, False, True, True]
        ),
        dynamic_leaf_mask=torch.zeros(6, dtype=torch.bool),
        verification_state=torch.tensor(
            [
                VERIFICATION_VERIFIED,
                VERIFICATION_MEASURED_SINGLE,
                VERIFICATION_MEASURED_SINGLE,
                VERIFICATION_VERIFIED,
                VERIFICATION_VERIFIED,
                VERIFICATION_VERIFIED,
            ],
            dtype=torch.int8,
        ),
        verified_camera_count=torch.tensor([2, 1, 1, 2, 1, 2]),
        verified_sequence_count=torch.ones(6, dtype=torch.int16),
        proposal_kind=torch.tensor(
            [
                PROPOSAL_NONE,
                PROPOSAL_NONE,
                PROPOSAL_NONE,
                PROPOSAL_NONE,
                PROPOSAL_NONE,
                PROPOSAL_SPLIT,
            ],
            dtype=torch.int8,
        ),
    )
    exact_camera = torch.tensor([False, True, False, True, True, True])

    owner = _moge3_canopy_hit_owner_mask(foliage, exact_camera)

    assert owner.tolist() == [True, True, False, False, False, False]


def test_moge3_canopy_optical_loss_keeps_low_logit_growth_finite_and_stops_at_target():
    logits = torch.tensor(
        [[[-4.0, -8.0, -10.0, torch.logit(torch.tensor(0.75)).item()]]],
        requires_grad=True,
    )
    hit = torch.sigmoid(logits)
    prehit = torch.zeros_like(hit)
    valid = torch.ones(1, 4, dtype=torch.bool)
    reliability = torch.ones(1, 4)
    task = {
        "p_canopy": torch.ones(1, 4),
        "p_canopy_core": torch.ones(1, 4),
        "p_rigid": torch.zeros(1, 4),
        "p_distortion_valid": torch.ones(1, 4),
        "p_transient": torch.zeros(1, 4),
        "w_rgb": torch.ones(1, 4),
    }

    loss, audit = _moge3_canopy_optical_loss(
        prehit, hit, task, valid, reliability
    )
    loss.backward()

    growth = -logits.grad[0, 0, :3]
    assert audit["hit_loss_domain"] == "one_sided_log_optical_depth_huber"
    assert torch.all(growth > 0)
    # The old alpha-space squared deficit lost orders of magnitude at -8/-10.
    # Log optical depth keeps all three source gradients in one useful band.
    assert float(growth.max() / growth.min()) < 1.10
    assert abs(float(logits.grad[0, 0, 3])) < 1.0e-7


def test_moge3_optical_mean_and_shared_parameter_gradient_are_resolution_invariant():
    outputs = []
    for height in (1, 7, 36):
        logits = torch.tensor([-2., -8.], requires_grad=True)
        prehit = logits[0].sigmoid().expand(height, 5)
        hit = logits[1].sigmoid().expand(height, 5)
        valid = torch.ones(height, 5, dtype=torch.bool)
        one = torch.ones(height, 5)
        task = {"p_canopy":one, "p_canopy_core":one, "p_rigid":one*0}
        loss, _, negative, positive = _moge3_canopy_optical_loss(
            prehit, hit, task, valid, one, return_components=True)
        gradient = torch.autograd.grad(loss,logits)[0]
        outputs.append((loss.detach(),negative.detach(),positive.detach(),gradient))
    for actual in outputs[1:]:
        for value,expected in zip(actual,outputs[0]):
            torch.testing.assert_close(value,expected)
    assert float(outputs[0][2]) > 7.0


def test_moge3_prehit_retirement_and_hit_growth_are_source_separated():
    logit = torch.tensor([[[0.0]]], requires_grad=True)
    alpha = torch.sigmoid(logit)
    valid = torch.ones(1, 1, dtype=torch.bool)
    task = {
        "p_canopy": torch.ones(1, 1),
        "p_canopy_core": torch.ones(1, 1),
        "p_rigid": torch.zeros(1, 1),
        "p_distortion_valid": torch.ones(1, 1),
        "p_transient": torch.zeros(1, 1),
        "w_rgb": torch.ones(1, 1),
    }
    total, _, prehit_loss, hit_loss = _moge3_canopy_optical_loss(
        alpha,
        alpha,
        task,
        valid,
        torch.ones(1, 1),
        return_components=True,
    )

    prehit_gradient = torch.autograd.grad(
        prehit_loss, logit, retain_graph=True
    )[0]
    hit_gradient = torch.autograd.grad(
        hit_loss, logit, retain_graph=True
    )[0]
    total_gradient = torch.autograd.grad(total, logit)[0]

    assert prehit_gradient.item() > 0
    assert hit_gradient.item() < 0
    torch.testing.assert_close(
        total_gradient, prehit_gradient + hit_gradient
    )
    # A row-level policy can now see the conflict and let retirement win;
    # inspecting only the net scalar gradient would have hidden one source.
    assert abs(float(total_gradient)) < (
        abs(float(prehit_gradient)) + abs(float(hit_gradient))
    )


def test_moge3_uncovered_hit_proposal_preserves_exact_ray_interval():
    class View:
        focal_x = 1.0
        focal_y = 1.0
        cx = 1.0
        cy = 1.0
        colmap_id = 17
        world_view_transform = torch.eye(4)
        camera_center = torch.zeros(3)
        original_image = torch.stack(
            [
                torch.full((3, 3), 0.2),
                torch.full((3, 3), 0.4),
                torch.full((3, 3), 0.6),
            ]
        )

    lower = torch.full((3, 3), float("nan"))
    upper = torch.full((3, 3), float("nan"))
    lower[1, 1], upper[1, 1] = 1.8, 2.2
    bounds = torch.stack([lower, upper])
    valid = torch.zeros((3, 3), dtype=torch.bool)
    valid[1, 1] = True
    reliability = torch.zeros((3, 3))
    reliability[1, 1] = 0.9
    canopy = torch.zeros((3, 3))
    canopy[1, 1] = 1.0
    task = {
        "p_canopy": canopy,
        "p_canopy_core": canopy,
        "p_distortion_valid": torch.ones((3, 3)),
        "p_transient": torch.zeros((3, 3)),
        "w_rgb": torch.ones((3, 3)),
    }
    proposals, audit = _moge3_uncovered_hit_proposals(
        View(),
        bounds,
        torch.zeros((1, 3, 3)),
        task,
        valid,
        reliability,
    )

    assert proposals is not None
    assert audit["selected_proposals"] == 1
    assert audit["direct_append"] is False
    assert proposals["camera_id"] == 17
    assert proposals["centers"][0].tolist() == pytest.approx(
        [0.0, 0.0, np.sqrt(1.8 * 2.2)]
    )
    assert proposals["directions"][0].tolist() == pytest.approx(
        [0.0, 0.0, 1.0]
    )
    assert proposals["hit_start"][0].item() == pytest.approx(1.8)
    assert proposals["hit_end"][0].item() == pytest.approx(2.2)
    assert proposals["colors"][0].tolist() == pytest.approx([0.2, 0.4, 0.6])

    no_proposal, no_audit = _moge3_uncovered_hit_proposals(
        View(),
        bounds,
        torch.full((1, 3, 3), 0.8),
        task,
        valid,
        reliability,
    )
    assert no_proposal is None
    assert no_audit["candidate_pixels"] == 0


def test_moge3_runtime_waits_for_causal_rigid_depth_scale():
    class Geometry:
        moge3_records = {"view": {}}

        def __init__(self):
            self.scale = None

        def configure_moge3_metric_scale(self, value):
            self.scale = float(value)

    class Args:
        moge3_rigid_depth_weight = 0.0
        moge3_canopy_optical_weight = 0.0

    geometry = Geometry()
    enabled, sigma = _configure_moge3_runtime(geometry, {}, Args())
    assert not enabled
    assert sigma == 0.0
    assert geometry.scale is None

    active = Args()
    active.moge3_rigid_depth_weight = 0.04
    with pytest.raises(RuntimeError, match="rigid-depth-calibrated"):
        _configure_moge3_runtime(geometry, {}, active)

    payload = {
        "audit": {
            "moge3_exact_k_front_hit": {
                "metric_to_cambridge_scale": 1.37,
                "global_log_scale_sigma": 0.064,
            }
        }
    }
    enabled, sigma = _configure_moge3_runtime(geometry, payload, active)
    assert enabled
    assert sigma == pytest.approx(0.064)
    assert geometry.scale == pytest.approx(1.37)

    chart_geometry = Geometry()
    chart_geometry.moge3_chart_scale_audit = {
        "metric_to_cambridge_scale": 0.83,
        "global_log_scale_sigma": 0.05,
    }
    enabled, sigma = _configure_moge3_runtime(
        chart_geometry, {}, active
    )
    assert enabled
    assert sigma == pytest.approx(0.05)
    assert chart_geometry.scale == pytest.approx(0.83)

    inconsistent = {
        "audit": {
            "moge3_exact_k_front_hit": {
                "metric_to_cambridge_scale": 1.37,
                "global_log_scale_sigma": 0.064,
            }
        }
    }
    with pytest.raises(RuntimeError, match="scene-scale authorities disagree"):
        _configure_moge3_runtime(chart_geometry, inconsistent, active)


def test_canonical_occlusion_completion_grows_only_volume_alpha():
    values = _canonical_occlusion_fixture()
    loss, audit = _canonical_occlusion_completion_loss(
        *values, canonical_view=True, surface_leakage_tolerance=0.03
    )
    loss.backward()

    volume_white, surface, _, alpha, surface_alpha, surface_depth, volume_depth, _ = (
        values
    )
    assert audit["contract"] == CANONICAL_OCCLUSION_COMPLETION_CONTRACT
    assert audit["supported_pixels"] == 4
    assert audit["ordering_conflict_pixels"] == 0
    assert audit["mean_optical_scale"] == pytest.approx(10.0 / 3.0, rel=1e-3)
    assert audit["mean_target_alpha"] > audit["mean_current_alpha"]
    # Uniform per-pixel support means the weighted reduction must equal the
    # scalar smooth-L1 map rather than being divided by image height.
    # This catches the former accidental division by image height.
    current_tau = -np.log1p(-0.30)
    target_tau = current_tau * (10.0 / 3.0)
    expected = abs(target_tau - current_tau) - 0.05
    assert float(loss) == pytest.approx(expected, rel=2e-4)
    assert audit["loss_normalization"] == (
        "one_scalar_channel_over_weighted_pixels"
    )
    # A negative dL/dalpha asks the optimizer to increase extinction.
    assert alpha.grad is not None
    assert float(alpha.grad.sum()) < 0
    assert volume_white.grad is None
    assert surface.grad is None
    assert surface_alpha.grad is None
    assert surface_depth.grad is None
    assert volume_depth.grad is None


def test_canonical_occlusion_completion_separates_wrong_depth_order():
    front_values = _canonical_occlusion_fixture(surface_in_front=False)
    front_loss, front_audit = _canonical_occlusion_completion_loss(
        *front_values, canonical_view=True, surface_leakage_tolerance=0.03
    )
    wrong_values = _canonical_occlusion_fixture(surface_in_front=True)
    wrong_loss, wrong_audit = _canonical_occlusion_completion_loss(
        *wrong_values, canonical_view=True, surface_leakage_tolerance=0.03
    )

    assert wrong_loss < 1.0e-4 * front_loss
    assert wrong_audit["ordering_conflict_pixels"] == 4
    assert wrong_audit["supported_pixels"] == 0
    assert front_audit["supported_pixels"] == 4


def test_canonical_completion_has_no_semantic_floor_without_volume_support():
    values = list(_canonical_occlusion_fixture(surface_in_front=True))
    # No intrinsic volume contribution means no measured colour direction or
    # optical owner. Topology/ray birth owns this hole; semantics alone may
    # not grow an arbitrary Gaussian, especially behind a facade.
    values[0] = torch.ones_like(values[0], requires_grad=True)
    values[3] = torch.zeros_like(values[3], requires_grad=True)
    loss, audit = _canonical_occlusion_completion_loss(
        *values,
        canonical_view=True,
        surface_leakage_tolerance=0.03,
    )
    loss.backward()

    volume_white, surface, _, alpha, surface_alpha, surface_depth, volume_depth, _ = (
        values
    )
    assert audit["directional_supported_pixels"] == 0
    assert audit["visibility_floor_supported_pixels"] == 0
    assert audit["supported_pixels"] == 0
    assert audit["ordering_conflict_pixels"] == 0
    assert audit["mean_visibility_floor_alpha"] == 0.0
    assert float(loss) == 0.0
    assert alpha.grad is not None
    torch.testing.assert_close(alpha.grad, torch.zeros_like(alpha))
    assert volume_white.grad is None
    assert surface.grad is None
    assert surface_alpha.grad is None
    assert surface_depth.grad is None
    assert volume_depth.grad is None


def test_noncanonical_occlusion_completion_has_no_positive_gradient():
    values = _canonical_occlusion_fixture()
    loss, audit = _canonical_occlusion_completion_loss(
        *values, canonical_view=False
    )
    loss.backward()

    alpha = values[3]
    assert audit["noncanonical_positive_updates_blocked"]
    assert audit["supported_pixels"] == 0
    torch.testing.assert_close(alpha.grad, torch.zeros_like(alpha))


def test_cuda_rng_restore_ignores_saved_surplus_devices(monkeypatch):
    restored = []
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state, device: restored.append((state, device)),
    )
    states = [
        torch.tensor([1], dtype=torch.uint8),
        torch.tensor([2], dtype=torch.uint8),
    ]
    audit = _restore_cuda_rng_states(states)

    assert len(restored) == 1
    assert restored[0][1] == 0
    torch.testing.assert_close(restored[0][0], states[0])
    assert audit["saved_device_count"] == 2
    assert audit["visible_device_count"] == 1
    assert audit["restored_device_count"] == 1


def test_canonical_occlusion_order_moves_only_live_volume_depth_forward():
    values = _canonical_occlusion_fixture(surface_in_front=True)
    loss, audit = _canonical_occlusion_order_loss(
        *values, canonical_view=True, maximum_forward_shift=3.0
    )
    loss.backward()

    mixed, surface, _, alpha, surface_alpha, surface_depth, volume_depth, _ = (
        values
    )
    assert audit["contract"] == CANONICAL_OCCLUSION_ORDER_CONTRACT
    assert audit["supported_pixels"] == 4
    assert audit["mean_requested_forward_shift"] == pytest.approx(2.03)
    assert loss > 0
    # Positive dL/dz makes gradient descent move the foliage toward the
    # camera. Every alternate way of hiding the conflict remains read-only.
    assert volume_depth.grad is not None
    assert float(volume_depth.grad.sum()) > 0
    assert mixed.grad is None
    assert surface.grad is None
    assert alpha.grad is None
    assert surface_alpha.grad is None
    assert surface_depth.grad is None
    assert not audit["surface_retirement_authorized"]
    assert not audit["opacity_gradient_authorized"]


def test_canonical_occlusion_order_is_live_when_front_composite_matches_target():
    values = list(_canonical_occlusion_fixture(surface_in_front=True))
    # A local 0.16 m ordering error with an already correct front-composite
    # colour needs geometry repair, not additional optical depth. The old
    # requests_growth gate made this exact success case a zero-gradient trap.
    values[5] = torch.full((1, 2, 2), 2.0, requires_grad=True)
    values[6] = torch.full((1, 2, 2), 2.13, requires_grad=True)
    alpha = values[3].detach()
    values[2] = (
        values[0].detach()
        + (1.0 - alpha) * (values[1].detach() - 1.0)
    )
    loss, audit = _canonical_occlusion_order_loss(
        *values,
        canonical_view=True,
        maximum_forward_shift=0.25,
        surface_leakage_tolerance=0.03,
    )
    loss.backward()

    assert audit["supported_pixels"] == 4
    assert audit["mean_optical_scale"] == pytest.approx(1.0, abs=1e-4)
    assert loss > 0
    assert values[6].grad is not None
    assert float(values[6].grad.sum()) > 0


def test_canonical_occlusion_order_trust_region_projects_adam_step_and_momentum():
    class Foliage:
        def __init__(self):
            self.xyz = torch.nn.Parameter(
                torch.tensor([[0.251, 0.0, 0.0], [0.40, 0.0, 0.0]])
            )
            self.initialization_center = torch.zeros(2, 3)

        def __len__(self):
            return len(self.xyz)

    foliage = Foliage()
    optimizer = torch.optim.Adam([foliage.xyz], lr=0.01)
    optimizer.state[foliage.xyz]["exp_avg"] = torch.tensor(
        [[-2.0, 3.0, 0.0], [-4.0, 0.0, 0.0]]
    )
    owner = torch.tensor([True, False])

    audit = _enforce_canonical_occlusion_order_trust_region(
        foliage,
        optimizer,
        owner,
        maximum_displacement=0.25,
    )

    assert audit["projected_rows"] == 1
    assert audit["outward_momentum_rows_cleared"] == 1
    assert foliage.xyz[0].norm() <= 0.250002
    # Only the outward radial Adam component is removed; tangent motion and
    # non-owner state remain intact.
    torch.testing.assert_close(
        optimizer.state[foliage.xyz]["exp_avg"][0],
        torch.tensor([0.0, 3.0, 0.0]),
    )
    torch.testing.assert_close(
        foliage.xyz[1], torch.tensor([0.40, 0.0, 0.0])
    )
    torch.testing.assert_close(
        optimizer.state[foliage.xyz]["exp_avg"][1],
        torch.tensor([-4.0, 0.0, 0.0]),
    )


def test_canonical_unknown_protection_accepts_measured_single_view_not_proposals():
    foliage = SimpleNamespace(
        dynamic_leaf_mask=torch.zeros(5, dtype=torch.bool),
        static_leaf_mask=torch.tensor([True, True, True, False, True]),
        verification_state=torch.tensor(
            [
                VERIFICATION_VERIFIED,
                VERIFICATION_UNVERIFIED,
                VERIFICATION_VERIFIED,
                VERIFICATION_VERIFIED,
                VERIFICATION_VERIFIED,
            ],
            dtype=torch.int8,
        ),
        verified_camera_count=torch.tensor([1, 1, 1, 1, 0]),
        proposal_kind=torch.tensor(
            [
                PROPOSAL_NONE,
                PROPOSAL_SPLIT,
                PROPOSAL_SPLIT,
                PROPOSAL_NONE,
                PROPOSAL_NONE,
            ],
            dtype=torch.int8,
        ),
    )

    protected = _canonical_static_unknown_protection_mask(foliage)

    assert protected.tolist() == [True, False, False, True, False]


def test_whole_image_static_authority_requires_resolved_two_camera_evidence():
    foliage = SimpleNamespace(
        dynamic_leaf_mask=torch.zeros(4, dtype=torch.bool),
        static_leaf_mask=torch.tensor([False, False, True, True]),
        verification_state=torch.full(
            (4,), VERIFICATION_VERIFIED, dtype=torch.int8
        ),
        verified_camera_count=torch.tensor([1, 2, 1, 2]),
        verified_sequence_count=torch.ones(4, dtype=torch.int16),
        proposal_kind=torch.tensor(
            [PROPOSAL_NONE, PROPOSAL_NONE, PROPOSAL_NONE, PROPOSAL_SPLIT],
            dtype=torch.int8,
        ),
    )

    authority = _persistent_static_evidence_mask(foliage)

    # Neither broad-envelope role nor detail role may turn one camera into a
    # whole-image completion/order permission. An unresolved split remains
    # excluded even after its per-row state flips to VERIFIED.
    assert authority.tolist() == [False, True, False, False]


def test_canonical_occlusion_order_rejects_nonlocal_depth_layer():
    values = _canonical_occlusion_fixture(surface_in_front=True)
    loss, audit = _canonical_occlusion_order_loss(
        *values, canonical_view=True, maximum_forward_shift=0.25
    )
    loss.backward()

    assert float(loss) == 0.0
    assert audit["supported_pixels"] == 0
    assert audit["out_of_trust_region_pixels"] == 4
    assert audit["mean_rejected_forward_shift"] == pytest.approx(2.03)
    torch.testing.assert_close(values[6].grad, torch.zeros_like(values[6]))


def test_canonical_occlusion_order_is_one_sided_and_noncanonical_zero():
    correct = _canonical_occlusion_fixture(surface_in_front=False)
    correct_loss, correct_audit = _canonical_occlusion_order_loss(
        *correct, canonical_view=True
    )
    assert float(correct_loss) == 0.0
    assert correct_audit["supported_pixels"] == 0

    unknown = _canonical_occlusion_fixture(surface_in_front=True)
    unknown_loss, unknown_audit = _canonical_occlusion_order_loss(
        *unknown, canonical_view=False
    )
    unknown_loss.backward()
    assert unknown_audit["noncanonical_geometry_updates_blocked"]
    torch.testing.assert_close(
        unknown[6].grad, torch.zeros_like(unknown[6])
    )


def test_static_global_cleanup_routes_only_negative_geometry_optical_signal():
    exact_prehit = torch.full(
        (1, 2, 2), 0.5, requires_grad=True
    )
    surface_alpha = torch.ones(1, 2, 2, requires_grad=True)
    task = {
        "p_rigid": torch.ones(2, 2),
        "p_canopy": torch.zeros(2, 2),
        "p_canopy_core": torch.zeros(2, 2),
        "p_sky": torch.zeros(2, 2),
        "p_transient": torch.zeros(2, 2),
        "p_boundary_uncertain": torch.zeros(2, 2),
        "p_unknown_ownership": torch.zeros(2, 2),
        "w_rgb": torch.ones(2, 2),
    }

    loss, audit = _static_detail_global_cleanup_loss(
        exact_prehit,
        surface_alpha,
        task,
        counterfactual_weight=0.5,
    )
    loss.backward()

    assert audit["contract"] == STATIC_DETAIL_GLOBAL_CLEANUP_CONTRACT
    assert audit["rigid_free_supported_pixels"] == 4
    assert audit["gradient_permissions"] == {
        "static_detail_xyz_scale_rotation": False,
        "persistent_static_detail_global_negative_optical_mass": True,
        "static_detail_sh": False,
        "surface_sky_uncertainty": False,
        "topology_statistics": False,
    }
    assert exact_prehit.grad is not None
    assert float(exact_prehit.grad.min()) > 0
    assert audit["rigid_volume_front_pixels"] == 4
    assert audit["rigid_behind_or_invalid_pixels_rejected"] == 0
    assert audit["volume_source"] == (
        "surface_zero_native_exact_volume_prehit_alpha"
    )
    assert audit["optimized_quantity"] == "exact_prehit_optical_depth"
    assert audit["rgb_benefit_veto_disabled"]
    assert surface_alpha.grad is None


def test_static_detail_global_negative_owner_is_camera_independent_and_persistent():
    foliage = SimpleNamespace(
        static_leaf_mask=torch.tensor([True, True, True, False]),
        dynamic_leaf_mask=torch.tensor([False, False, False, False]),
        verification_state=torch.tensor(
            [
                VERIFICATION_VERIFIED,
                VERIFICATION_VERIFIED,
                VERIFICATION_MEASURED_SINGLE,
                VERIFICATION_VERIFIED,
            ],
            dtype=torch.int8,
        ),
        verified_camera_count=torch.tensor([2, 1, 1, 3]),
        verified_sequence_count=torch.tensor([1, 1, 1, 1]),
        proposal_kind=torch.tensor(
            [PROPOSAL_NONE, PROPOSAL_NONE, PROPOSAL_NONE, PROPOSAL_NONE],
            dtype=torch.int8,
        ),
    )

    # Only the persistent two-camera detail is a global deployment owner.
    # No current-camera argument exists, so a non-support canonical camera
    # cannot accidentally lose its strict negative veto permission.
    torch.testing.assert_close(
        _static_detail_global_negative_owner_mask(foliage),
        torch.tensor([True, False, False, False]),
    )


def test_v107_cleanup_resume_migration_changes_only_negative_authority():
    fusion = {
        "contract": "scene canonical",
        "input_dynamic_rows": 1,
        "associated_dynamic_rows": 1,
        "camera_sequence_metadata_source": "fixed",
        "camera_sequence_metadata_count": 2,
        "canonical_sequence_policy": "scene",
        "canonical_mode_voxel_size": 0.1,
        "maximum_modes_per_group": 4,
        "minimum_supporting_views": 2,
        "minimum_supporting_sequences": 1,
        "voxel_size": 0.1,
        "output_static_rows": 1,
    }
    current_cleanup = {
        "contract": STATIC_DETAIL_GLOBAL_CLEANUP_CONTRACT,
        "every": 2,
        "weight": 0.25,
        "camera_schedule": {
            "contract": "all canonical",
            "camera_count": 352,
            "positive_exact_support_camera_count": 235,
            "schedule_sha256": "fixed",
        },
        "gradient_owners": [
            "persistent_static_detail_global_negative_optical_mass"
        ],
    }
    saved_cleanup = {
        **current_cleanup,
        "contract": STATIC_DETAIL_GLOBAL_CLEANUP_V106_PREDECESSOR_CONTRACT,
        "camera_schedule": (
            "dedicated_uniform_scene_canonical_exact_support_epoch"
        ),
        "gradient_owners": [
            "positive_evidence_sequence_static_detail_optical_mass"
        ],
    }
    assert not _resume_training_contract_differences(
        {
            "static_detail_global_cleanup": saved_cleanup,
            "static_fusion": {**fusion, "audit_sum": 1.0},
            "initialization_manifest_sha256": "init",
            "foliage_seed_sha256": "seed",
        },
        {
            "static_detail_global_cleanup": current_cleanup,
            "static_fusion": {**fusion, "audit_sum": 1.0 + 1e-7},
            "initialization_manifest_sha256": "init",
            "foliage_seed_sha256": "seed",
        },
        allow_v107_global_negative_cleanup_migration=True,
    )

    changed_weight = {
        **saved_cleanup,
        "weight": 0.5,
    }
    with pytest.raises(RuntimeError, match="changed more than"):
        _resume_training_contract_differences(
            {
                "static_detail_global_cleanup": changed_weight,
                "static_fusion": fusion,
                "initialization_manifest_sha256": "init",
                "foliage_seed_sha256": "seed",
            },
            {
                "static_detail_global_cleanup": current_cleanup,
                "static_fusion": fusion,
                "initialization_manifest_sha256": "init",
                "foliage_seed_sha256": "seed",
            },
            allow_v107_global_negative_cleanup_migration=True,
        )


def test_surface_front_depth_query_bounds_are_detached_and_fail_closed():
    surface_alpha = torch.tensor(
        [[[0.95, 0.949], [1.0, float("nan")]]], requires_grad=True
    )
    surface_depth = torch.tensor(
        [[[2.0, 3.0], [0.02, float("inf")]]], requires_grad=True
    )

    bounds, valid = _surface_front_depth_query_bounds(
        surface_alpha, surface_depth, clearance=0.03
    )

    assert bounds.shape == (2, 2, 2)
    torch.testing.assert_close(bounds[:, 0, 0], torch.tensor([1.97, 1.97]))
    assert valid.tolist() == [[True, False], [False, False]]
    assert torch.isnan(bounds[:, ~valid]).all()
    assert not bounds.requires_grad
    assert not valid.requires_grad


def test_static_global_cleanup_has_no_rgb_veto_and_fails_closed_without_surface():
    prehit = torch.full((1, 2, 2), 0.5, requires_grad=True)
    surface_alpha = torch.ones(1, 2, 2)
    task = {
        "p_rigid": torch.ones(2, 2),
        "p_canopy": torch.zeros(2, 2),
        "p_canopy_core": torch.zeros(2, 2),
        "p_sky": torch.zeros(2, 2),
        "p_transient": torch.zeros(2, 2),
        "p_boundary_uncertain": torch.zeros(2, 2),
        "p_unknown_ownership": torch.zeros(2, 2),
        "w_rgb": torch.ones(2, 2),
    }
    loss, audit = _static_detail_global_cleanup_loss(
        prehit,
        surface_alpha,
        task,
        counterfactual_weight=1.0,
    )
    loss.backward()
    assert float(loss) > 0.0
    assert audit["rgb_benefit_veto_disabled"]
    assert float(prehit.grad.min()) > 0.0

    unsupported = torch.full((1, 2, 2), 0.5, requires_grad=True)
    unsupported_loss, unsupported_audit = _static_detail_global_cleanup_loss(
        unsupported,
        torch.full_like(surface_alpha, 0.90),
        task,
        counterfactual_weight=1.0,
    )
    unsupported_loss.backward()
    assert float(unsupported_loss) == 0.0
    assert unsupported_audit["rigid_behind_or_invalid_pixels_rejected"] == 4
    torch.testing.assert_close(unsupported.grad, torch.zeros_like(unsupported))


def test_strict_rigid_front_priority_vetoes_growth_and_adam_momentum():
    opacity = torch.nn.Parameter(torch.zeros(3, 1))
    opacity.grad = torch.tensor([[-5.0], [1.0], [-2.0]])
    first_moment = torch.tensor([[-3.0], [2.0], [-4.0]])
    foliage = SimpleNamespace(opacity_logits=opacity)
    optimizer = SimpleNamespace(state={opacity: {"exp_avg": first_moment}})
    strict = torch.tensor([[2.0], [0.0], [1.5]])

    audit = _apply_strict_rigid_front_opacity_policy(
        foliage, optimizer, strict
    )

    torch.testing.assert_close(
        opacity.grad, torch.tensor([[2.0], [1.0], [1.5]])
    )
    torch.testing.assert_close(
        first_moment, torch.tensor([[0.0], [2.0], [0.0]])
    )
    assert audit["owner_rows"] == 2
    assert audit["growth_rows_vetoed"] == 2


def test_material_optical_source_rejects_numerical_ewa_tail():
    opacity = torch.nn.Parameter(torch.zeros(4, 1))
    foliage = SimpleNamespace(
        xyz=torch.zeros(4, 3),
        opacity_logits=opacity,
        static_leaf_mask=torch.tensor([True, True, True, False]),
        persistent_envelope_mask=torch.tensor([False, False, False, True]),
    )
    raw = torch.tensor([[1.0], [5.0e-4], [0.0], [2.0e-4]])

    filtered, audit = _material_optical_source_gradient(
        foliage,
        raw,
        direction="retirement",
        relative_magnitude_threshold=1.0e-3,
    )

    torch.testing.assert_close(
        filtered, torch.tensor([[1.0], [0.0], [0.0], [0.0]])
    )
    assert audit["raw_rows"] == 3
    assert audit["material_rows"] == 1
    assert audit["discarded_tail_rows"] == 2

    stats = {}
    owners, _ = _update_persistent_rigid_front_conflict_debt(
        foliage,
        stats,
        raw,
        relative_magnitude_threshold=1.0e-3,
    )
    assert owners.tolist() == [True, False, False, False]


@pytest.mark.parametrize("policy,allowed", [
    ("joint", True), ("appearance_only", False),
    ("frozen", False), ("atlas_residual", False),
])
def test_surface_retirement_respects_mature_opacity_authority(policy, allowed):
    assert _surface_replacement_retirement_allowed(policy) is allowed


@pytest.mark.parametrize("demands,expected_shared", [
    ([0, 0, 0, 0], [False, False, False, False]),
    ([0, 2, 0, 0], [False, False, False, False]),
    ([2, 0, 0, 0], [True, False, False, False]),
])
def test_shared_optical_owner_requires_same_row_demand(demands, expected_shared):
    class FoliageStub(SimpleNamespace):
        def __len__(self):
            return len(self.xyz)

    opacity = torch.nn.Parameter(torch.zeros(4, 1))
    foliage = FoliageStub(
        xyz=torch.zeros(4, 3),
        opacity_logits=opacity,
        static_leaf_mask=torch.tensor([False, True, False, True]),
        dynamic_leaf_mask=torch.zeros(4, dtype=torch.bool),
        persistent_envelope_mask=torch.tensor([True, False, True, False]),
        replacement_group=torch.tensor([10, 10, 20, 30]),
        proposal_kind=torch.zeros(4, dtype=torch.int8),
        verification_state=torch.full(
            (4,), VERIFICATION_VERIFIED, dtype=torch.int8
        ),
        verified_camera_count=torch.tensor([0, 2, 0, 2]),
        verified_sequence_count=torch.tensor([0, 1, 0, 1]),
        handoff_retired_fraction=torch.zeros(4),
        handoff_reference_mass=torch.zeros(4),
        integrated_optical_mass=lambda: torch.ones(4),
    )
    stats = {
        "rigid_front_conflict_observations": torch.tensor(
            [3, 0, 2, 0], dtype=torch.int16
        ),
        "positive_optical_demand_observations": torch.tensor(
            demands, dtype=torch.int16
        ),
    }
    conflict, pure, shared = _persistent_optical_ownership_states(
        foliage, stats
    )
    assert conflict.tolist() == [True, False, True, False]
    assert shared.tolist() == expected_shared
    assert torch.equal(pure, conflict & ~shared)

    opacity.grad = torch.tensor([[2.0], [-1.0], [-3.0], [-1.0]])
    first_moment = torch.tensor([[1.0], [-1.0], [-2.0], [-1.0]])
    optimizer = SimpleNamespace(state={opacity: {"exp_avg": first_moment}})
    audit = _apply_persistent_rigid_front_conflict_policy(
        foliage, optimizer, conflict, shared
    )
    # Shared row zero cannot be globally retired; pure-negative row two may
    # retire but may never regrow.
    torch.testing.assert_close(
        opacity.grad, torch.tensor([
            [0.0 if expected_shared[0] else 2.0], [-1.0], [0.0], [-1.0]
        ])
    )
    torch.testing.assert_close(
        first_moment, torch.tensor([
            [0.0 if expected_shared[0] else 1.0], [-1.0], [0.0], [-1.0]
        ])
    )
    assert audit["shared_retirement_rows_deferred"] == int(expected_shared[0])
    assert audit["growth_rows_vetoed"] == 1


def test_persistent_optical_debt_starts_after_volume_topology_settlement():
    args = SimpleNamespace(volume_densify_until_iteration=18_000)
    assert not _persistent_optical_debt_active(17_999, args)
    assert not _persistent_optical_debt_active(18_000, args)
    assert _persistent_optical_debt_active(18_001, args)


def test_shared_optical_owner_post_step_restores_exact_pre_step_opacity():
    class FoliageStub(SimpleNamespace):
        def __len__(self):
            return len(self.opacity_logits)

    opacity = torch.nn.Parameter(torch.tensor([[0.4], [-0.2]]))
    foliage = FoliageStub(
        opacity_logits=opacity,
        persistent_envelope_mask=torch.tensor([False, False]),
    )
    optimizer = SimpleNamespace(
        state={opacity: {"exp_avg": torch.tensor([[2.0], [-3.0]])}}
    )
    shared = torch.tensor([True, False])
    reference = torch.tensor([[0.1]])

    audit = _enforce_shared_optical_owner_post_step(
        foliage, optimizer, shared, reference
    )

    torch.testing.assert_close(opacity, torch.tensor([[0.1], [-0.2]]))
    torch.testing.assert_close(
        optimizer.state[opacity]["exp_avg"], torch.tensor([[0.0], [-3.0]])
    )
    assert audit["changed_rows_restored"] == 1


def test_shared_optical_owner_preserves_only_local_handoff_retirement():
    class FoliageStub(SimpleNamespace):
        def __len__(self):
            return len(self.opacity_logits)

    opacity = torch.nn.Parameter(torch.tensor([[-0.3], [0.4], [-0.2]]))
    foliage = FoliageStub(
        opacity_logits=opacity,
        persistent_envelope_mask=torch.tensor([True, True, False]),
        handoff_retired_fraction=torch.tensor([0.2, 0.2, 0.0]),
        handoff_reference_mass=torch.zeros(3),
        integrated_optical_mass=lambda: torch.ones(3),
    )
    optimizer = SimpleNamespace(
        state={opacity: {"exp_avg": torch.tensor([[1.0], [-2.0], [3.0]])}}
    )
    shared = torch.tensor([True, True, False])
    pre_handoff = torch.tensor([[0.1], [0.1]])

    audit = _enforce_shared_optical_owner_post_step(
        foliage,
        optimizer,
        shared,
        pre_handoff,
        allow_handoff_retirement=True,
    )

    # A verified local receiver may take mass from row zero.  Row one's
    # attempted restoration is forbidden while its rigid-front debt remains.
    torch.testing.assert_close(
        opacity, torch.tensor([[-0.3], [0.1], [-0.2]])
    )
    assert audit["handoff_retirement_rows_preserved"] == 1
    assert audit["changed_rows_restored"] == 1


def test_rigid_front_conflict_debt_persists_and_blocks_later_growth():
    opacity = torch.nn.Parameter(torch.zeros(3, 1))
    foliage = SimpleNamespace(
        xyz=torch.zeros(3, 3),
        opacity_logits=opacity,
        static_leaf_mask=torch.tensor([True, False, False]),
        persistent_envelope_mask=torch.tensor([False, True, False]),
    )
    stats = {
        "rigid_front_conflict_observations": torch.zeros(
            3, dtype=torch.int16
        )
    }
    owners, debt_audit = _update_persistent_rigid_front_conflict_debt(
        foliage, stats, torch.tensor([[1.0], [2.0], [3.0]])
    )

    # The third row is not an optical foliage owner and must not enter debt.
    assert owners.tolist() == [True, True, False]
    assert debt_audit["new_debt_rows"] == 2
    assert stats["rigid_front_conflict_observations"].tolist() == [1, 1, 0]

    # Simulate a later checkpoint/resume step with no cleanup render. The
    # persisted ledger, rather than the current cadence, still owns the veto.
    restored_stats = {
        name: value.clone() for name, value in stats.items()
    }
    restored = _persistent_rigid_front_conflict_rows(
        foliage, restored_stats
    )
    opacity.grad = torch.tensor([[-4.0], [-5.0], [-6.0]])
    first_moment = torch.tensor([[-2.0], [-3.0], [-4.0]])
    optimizer = SimpleNamespace(state={opacity: {"exp_avg": first_moment}})
    policy_audit = _apply_persistent_rigid_front_conflict_policy(
        foliage, optimizer, restored
    )

    torch.testing.assert_close(
        opacity.grad, torch.tensor([[0.0], [0.0], [-6.0]])
    )
    torch.testing.assert_close(
        first_moment, torch.tensor([[0.0], [0.0], [-4.0]])
    )
    assert policy_audit["growth_rows_vetoed"] == 2


def test_persistent_optical_ledgers_keep_strongest_camera_provenance():
    opacity = torch.nn.Parameter(torch.zeros(3, 1))
    foliage = SimpleNamespace(
        xyz=torch.zeros(3, 3),
        opacity_logits=opacity,
        static_leaf_mask=torch.tensor([True, False, False]),
        persistent_envelope_mask=torch.tensor([False, True, False]),
    )
    stats = {}
    owners, audit = _update_persistent_rigid_front_conflict_debt(
        foliage,
        stats,
        torch.tensor([[0.5], [0.2], [9.0]]),
        strict_source_events=(
            (torch.tensor([[0.5], [0.2], [9.0]]), 7),
            (torch.tensor([[0.1], [0.8], [0.0]]), 8),
        ),
    )

    assert owners.tolist() == [True, True, False]
    assert audit["provenance_camera_rows"] == 2
    # Row zero keeps camera seven because its later event was weaker; row one
    # updates to camera eight because that source was stronger.
    assert stats["rigid_front_conflict_camera_id"].tolist() == [7, 8, -1]
    torch.testing.assert_close(
        stats["rigid_front_conflict_max_gradient"],
        torch.tensor([0.5, 0.8, 0.0]),
    )
    assert stats["rigid_front_conflict_observations"].tolist() == [2, 2, 0]

    demand_audit = _update_persistent_positive_optical_demand(
        foliage,
        stats,
        torch.tensor([[-0.4], [-1.2], [-5.0]]),
        camera_id=11,
    )
    assert demand_audit["source_rows"] == 2
    assert stats["positive_optical_demand_camera_id"].tolist() == [11, 11, -1]
    torch.testing.assert_close(
        stats["positive_optical_demand_max_gradient"],
        torch.tensor([0.4, 1.2, 0.0]),
    )


def test_shared_envelope_localization_is_group_atomic_and_bounded():
    class FoliageStub(SimpleNamespace):
        def __len__(self):
            return len(self.xyz)

    count = 6
    foliage = FoliageStub(
        xyz=torch.zeros(count, 3),
        opacity_logits=torch.nn.Parameter(torch.zeros(count, 1)),
        static_leaf_mask=torch.tensor([False, True, False, False, False, True]),
        persistent_envelope_mask=torch.tensor([True, False, True, False, True, False]),
        dynamic_leaf_mask=torch.zeros(count, dtype=torch.bool),
        verification_state=torch.full(
            (count,), VERIFICATION_VERIFIED, dtype=torch.int8
        ),
        proposal_kind=torch.tensor(
            [PROPOSAL_NONE] * 5 + [PROPOSAL_SPLIT], dtype=torch.int8
        ),
        replacement_group=torch.tensor([10, 10, 20, 20, 30, 30]),
        support_camera_ids=torch.tensor([[7, 8]] * count),
        verified_camera_count=torch.full((count,), 2, dtype=torch.int16),
        verified_sequence_count=torch.ones(count, dtype=torch.int16),
        scales=torch.tensor(
            [
                [0.4, 0.4, 0.4],
                [0.3, 0.2, 0.1],
                [0.5, 0.5, 0.5],
                [0.2, 0.2, 0.2],
                [0.6, 0.6, 0.6],
                [0.4, 0.3, 0.2],
            ]
        ),
    )
    stats = {
        "rigid_front_conflict_observations": torch.tensor([3, 0, 2, 0, 4, 0], dtype=torch.int16),
        "rigid_front_conflict_camera_id": torch.tensor([7, -1, 8, -1, 7, -1], dtype=torch.int32),
        "rigid_front_conflict_max_gradient": torch.tensor([0.7, 0.0, 0.9, 0.0, 1.0, 0.0]),
        "positive_optical_demand_observations": torch.tensor([0, 2, 2, 0, 0, 3], dtype=torch.int16),
        "positive_optical_demand_camera_id": torch.tensor([-1, 8, 7, -1, -1, 8], dtype=torch.int32),
        "positive_optical_demand_max_gradient": torch.tensor([0.0, 0.5, 0.9, 0.0, 0.0, 0.9]),
    }

    factorize, split_detail, audit = (
        _select_shared_envelope_localization_actions(
            foliage,
            stats,
            torch.tensor([7, 8, 9]),
            maximum_groups=8,
        )
    )

    # Group ten already has durable detail, so only that detail is split.
    # Group twenty has no detail and receives one co-located receiver.
    # Group thirty is rejected atomically because one row is unresolved.
    assert factorize.tolist() == [2]
    assert split_detail.tolist() == [1]
    assert audit["eligible_groups"] == 2
    assert audit["selected_groups"] == 2
    assert audit["unresolved_groups_rejected"] == 1

    factorize_one, split_one, audit_one = (
        _select_shared_envelope_localization_actions(
            foliage,
            stats,
            torch.tensor([7, 8, 9]),
            maximum_groups=1,
        )
    )
    assert factorize_one.tolist() == []
    assert split_one.tolist() == [1]
    assert audit_one["eligible_groups"] == 2
    assert audit_one["selected_groups"] == 1
    assert audit_one["actionable_factorize_groups"] == 1
    assert audit_one["actionable_split_detail_groups"] == 1
    assert audit_one["reserved_split_detail_quota"] == 1

    camera_forward = torch.full((10, 3), float("nan"))
    camera_forward[7] = torch.tensor([0.0, 0.0, 2.0])
    camera_forward[8] = torch.tensor([0.0, 1.0, 0.0])
    normals, camera_ids = _shared_envelope_detail_split_plane_normals(
        foliage, stats, split_detail, camera_forward
    )
    assert camera_ids.tolist() == [7]
    torch.testing.assert_close(normals, torch.tensor([[0.0, 0.0, 1.0]]))


def test_detail_only_dual_owner_has_localization_exit():
    class FoliageStub(SimpleNamespace):
        def __len__(self):
            return len(self.xyz)

    foliage = FoliageStub(
        xyz=torch.zeros(1, 3),
        static_leaf_mask=torch.tensor([True]),
        persistent_envelope_mask=torch.tensor([False]),
        dynamic_leaf_mask=torch.tensor([False]),
        verification_state=torch.tensor([VERIFICATION_VERIFIED]),
        proposal_kind=torch.tensor([PROPOSAL_NONE]),
        replacement_group=torch.tensor([0]),
        support_camera_ids=torch.tensor([[0, 1]]),
        verified_camera_count=torch.tensor([2]),
        verified_sequence_count=torch.tensor([1]),
        scales=torch.ones(1, 3),
    )
    stats = {
        "rigid_front_conflict_observations": torch.tensor([1]),
        "rigid_front_conflict_camera_id": torch.tensor([0]),
        "rigid_front_conflict_max_gradient": torch.tensor([0.5]),
        "positive_optical_demand_observations": torch.tensor([1]),
        "positive_optical_demand_camera_id": torch.tensor([1]),
        "positive_optical_demand_max_gradient": torch.tensor([0.5]),
        "radius": torch.tensor([3.0]),
    }
    factorize, split, audit = _select_shared_envelope_localization_actions(
        foliage, stats, torch.tensor([0, 1]), maximum_groups=1,
        minimum_detail_radius_pixels=2.0,
    )
    assert factorize.numel() == 0
    assert split.tolist() == [0]
    assert audit["selected_groups"] == 1
    normals, cameras = _shared_envelope_detail_split_plane_normals(
        foliage, stats, split, torch.tensor([[0., 0., 1.], [0., 1., 0.]])
    )
    assert cameras.tolist() == [0]
    torch.testing.assert_close(normals, torch.tensor([[0., 0., 1.]]))
    # With no optical demand, verification cannot self-exempt this detail.
    stats["positive_optical_demand_observations"].zero_()
    conflict, pure, shared = _persistent_optical_ownership_states(foliage, stats)
    assert conflict.tolist() == pure.tolist() == [True]
    assert shared.tolist() == [False]
    # An ownerless birth can carry measured positive demand too. Missing
    # replacement-group metadata must not erase that optical responsibility.
    stats["positive_optical_demand_observations"].fill_(1)
    foliage.replacement_group.fill_(-1)
    _, pure, shared = _persistent_optical_ownership_states(foliage, stats)
    assert pure.tolist() == [False]
    assert shared.tolist() == [True]


def test_shared_envelope_localization_defers_to_lifecycle_maintenance():
    args = SimpleNamespace(
        volume_densify_until_iteration=18_000,
        volume_densify_every=200,
    )

    assert not _post_growth_verification_maintenance_due(18_000, args)
    assert not _post_growth_verification_maintenance_due(22_100, args)
    assert _post_growth_verification_maintenance_due(22_200, args)
    assert _post_growth_verification_maintenance_due(22_400, args)

    args.volume_densify_every = 0
    with pytest.raises(ValueError, match="must be positive"):
        _post_growth_verification_maintenance_due(22_200, args)


def test_shared_envelope_localization_transaction_conserves_owner_plane_mass_and_roles():
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(
        {
            "version": "independent_sfm_semantic_canopy_volume_v1",
            "centers": torch.tensor(
                [[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [1.0, 0.0, 2.0]]
            ),
            "scales": torch.full((3, 3), 0.1),
            "colors": torch.full((3, 3), 0.4),
            "opacities": torch.full((3, 1), 0.2),
            "quaternions": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]]
            ).repeat(3, 1),
            "layer_role": torch.zeros(3, dtype=torch.int8),
            "static_detail": torch.tensor([False, True, False]),
            "replacement_group": torch.tensor([0, 0, 1]),
            "support_camera_ids": torch.tensor([[7, 8]] * 3),
            "support_view_count": torch.full((3,), 2, dtype=torch.int16),
            "support_sequence_count": torch.ones(3, dtype=torch.int16),
            "verified_camera_ids": torch.tensor([[7, 8]] * 3),
            "verified_camera_count": torch.full((3,), 2, dtype=torch.int16),
            "verified_sequence_count": torch.ones(3, dtype=torch.int16),
        }
    )
    stats = {
        "rigid_front_conflict_observations": torch.tensor(
            [3, 0, 2], dtype=torch.int16
        ),
        "rigid_front_conflict_camera_id": torch.tensor(
            [7, -1, 8], dtype=torch.int32
        ),
        "rigid_front_conflict_max_gradient": torch.tensor([0.7, 0.0, 0.6]),
        "positive_optical_demand_observations": torch.tensor(
            [0, 2, 2], dtype=torch.int16
        ),
        "positive_optical_demand_camera_id": torch.tensor(
            [-1, 8, 7], dtype=torch.int32
        ),
        "positive_optical_demand_max_gradient": torch.tensor([0.0, 0.5, 0.4]),
    }
    factorize, split_detail, _ = (
        _select_shared_envelope_localization_actions(
            foliage,
            stats,
            torch.tensor([7, 8]),
            maximum_groups=4,
        )
    )
    mass_before = foliage.integrated_optical_mass().sum().clone()
    camera_forward = torch.full((9, 3), float("nan"))
    camera_forward[7] = torch.tensor([0.0, 0.0, 1.0])
    camera_forward[8] = torch.tensor([0.0, 1.0, 0.0])

    final_stats, event = (
        _apply_shared_envelope_localization_model_transaction(
            foliage,
            stats,
            factorize,
            split_detail,
            camera_forward,
            current_iteration=22_000,
            maximum_final_count=16,
            receiver_optical_mass_fraction=0.05,
        )
    )

    assert event["old_count"] == 3
    assert event["new_count"] == 5
    assert event["net_growth"] == 2
    assert event["factorization"]["materialized"] == 1
    assert event["detail_split"]["split_parents"] == 1
    assert event["detail_split"]["camera_plane_parents"] == 1
    assert event["split_conflict_camera_ids"] == [7]
    assert int((foliage.proposal_kind == PROPOSAL_SPLIT).sum()) == 2
    assert int(
        (
            foliage.verification_state
            == VERIFICATION_FACTORIZED_RECEIVER
        ).sum()
    ) == 1
    assert event["integrated_optical_mass_after_factorization"] == pytest.approx(
        float(mass_before), rel=5e-6, abs=1e-8
    )
    assert event["detail_split"][
        "camera_plane_optical_mass_after"
    ] == pytest.approx(
        event["detail_split"]["camera_plane_optical_mass_before"],
        rel=5e-6,
        abs=1e-8,
    )
    # Keeping the uncertain camera-depth extent can change the view-independent
    # maximum-cross-section proxy.  The conserved physical quantity for this
    # localization transaction is extinction in the calibrated owner plane.
    assert event["integrated_optical_mass_after"] > float(mass_before)
    assert all(len(value) == 5 for value in final_stats.values())
    # The appended receiver starts with neutral persistent evidence; split
    # children inherit the selected detail's positive demand until real
    # cameras independently resolve their proposal family.
    new_receiver = event["_new_receiver_rows"]
    assert final_stats[
        "positive_optical_demand_observations"
    ][new_receiver].tolist() == [0]
    split_children = foliage.proposal_kind == PROPOSAL_SPLIT
    assert final_stats[
        "positive_optical_demand_observations"
    ][split_children].tolist() == [2, 2]


def test_shared_envelope_selector_never_duplicates_an_existing_receiver():
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(
        {
            "version": "independent_sfm_semantic_canopy_volume_v1",
            "centers": torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]),
            "scales": torch.full((2, 3), 0.1),
            "colors": torch.full((2, 3), 0.4),
            "opacities": torch.full((2, 1), 0.2),
            "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
            "layer_role": torch.zeros(2, dtype=torch.int8),
            "static_detail": torch.tensor([False, True]),
            "replacement_group": torch.tensor([0, 0]),
            "support_camera_ids": torch.tensor([[7, 8], [7, 8]]),
            "support_view_count": torch.full((2,), 2, dtype=torch.int16),
            "support_sequence_count": torch.ones(2, dtype=torch.int16),
            "verified_camera_ids": torch.tensor([[7, 8], [7, 8]]),
            "verified_camera_count": torch.tensor([2, 0], dtype=torch.int16),
            "verified_sequence_count": torch.ones(2, dtype=torch.int16),
            "verification_state": torch.tensor(
                [VERIFICATION_VERIFIED, VERIFICATION_FACTORIZED_RECEIVER],
                dtype=torch.int8,
            ),
        }
    )
    stats = {
        "rigid_front_conflict_observations": torch.tensor(
            [3, 2], dtype=torch.int16
        ),
        "rigid_front_conflict_camera_id": torch.tensor(
            [7, 7], dtype=torch.int32
        ),
        "rigid_front_conflict_max_gradient": torch.tensor([0.7, 0.3]),
        "positive_optical_demand_observations": torch.tensor(
            [2, 2], dtype=torch.int16
        ),
        "positive_optical_demand_camera_id": torch.tensor(
            [8, 8], dtype=torch.int32
        ),
        "positive_optical_demand_max_gradient": torch.tensor([0.5, 0.5]),
    }
    factorize, split_detail, audit = (
        _select_shared_envelope_localization_actions(
            foliage,
            stats,
            torch.tensor([7, 8]),
            maximum_groups=4,
        )
    )
    assert factorize.numel() == 0
    assert split_detail.numel() == 0
    assert audit["preselected_groups"] == 1
    assert audit["selected_groups"] == 0
    assert audit["existing_unlocalizable_detail_groups_rejected"] == 1


def test_shared_envelope_selector_localizes_group_with_demandless_detail_row():
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(
        {
            "version": "independent_sfm_semantic_canopy_volume_v1",
            "centers": torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]),
            "scales": torch.tensor([[0.3, 0.2, 0.1], [0.2, 0.1, 0.1]]),
            "colors": torch.full((2, 3), 0.4),
            "opacities": torch.full((2, 1), 0.2),
            "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
            "layer_role": torch.zeros(2, dtype=torch.int8),
            "static_detail": torch.tensor([False, True]),
            "replacement_group": torch.tensor([0, 0]),
            "support_camera_ids": torch.tensor([[7, 8], [7, 8]]),
            "support_view_count": torch.full((2,), 2, dtype=torch.int16),
            "support_sequence_count": torch.ones(2, dtype=torch.int16),
            "verified_camera_ids": torch.tensor([[7, 8], [7, 8]]),
            "verified_camera_count": torch.full((2,), 2, dtype=torch.int16),
            "verified_sequence_count": torch.ones(2, dtype=torch.int16),
        }
    )
    stats = {
        "rigid_front_conflict_observations": torch.tensor([3, 0], dtype=torch.int16),
        "rigid_front_conflict_camera_id": torch.tensor([7, -1], dtype=torch.int32),
        "rigid_front_conflict_max_gradient": torch.tensor([0.7, 0.0]),
        # The group has positive tree demand through its shared envelope. The
        # already verified local detail need not independently duplicate that
        # bookkeeping merely to receive spatial bandwidth.
        "positive_optical_demand_observations": torch.tensor([2, 0], dtype=torch.int16),
        "positive_optical_demand_camera_id": torch.tensor([8, -1], dtype=torch.int32),
        "positive_optical_demand_max_gradient": torch.tensor([0.5, 0.0]),
    }

    factorize, split_detail, audit = _select_shared_envelope_localization_actions(
        foliage, stats, torch.tensor([7, 8]), maximum_groups=4
    )

    assert factorize.numel() == 0
    assert split_detail.tolist() == [1]
    assert audit["group_demand_fallback_detail_groups"] == 1
    assert audit["existing_unlocalizable_detail_groups_rejected"] == 0


def test_shared_envelope_selector_splits_verified_debt_bearing_detail():
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(
        {
            "version": "independent_sfm_semantic_canopy_volume_v1",
            "centers": torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]),
            "scales": torch.tensor([[0.3, 0.2, 0.1], [0.2, 0.1, 0.1]]),
            "colors": torch.full((2, 3), 0.4),
            "opacities": torch.full((2, 1), 0.2),
            "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
            "layer_role": torch.zeros(2, dtype=torch.int8),
            "static_detail": torch.tensor([False, True]),
            "replacement_group": torch.tensor([0, 0]),
            "support_camera_ids": torch.tensor([[7, 8], [7, 8]]),
            "support_view_count": torch.full((2,), 2, dtype=torch.int16),
            "support_sequence_count": torch.ones(2, dtype=torch.int16),
            "verified_camera_ids": torch.tensor([[7, 8], [7, 8]]),
            "verified_camera_count": torch.full((2,), 2, dtype=torch.int16),
            "verified_sequence_count": torch.ones(2, dtype=torch.int16),
        }
    )
    stats = {
        "rigid_front_conflict_observations": torch.tensor([3, 2], dtype=torch.int16),
        "rigid_front_conflict_camera_id": torch.tensor([7, 7], dtype=torch.int32),
        "rigid_front_conflict_max_gradient": torch.tensor([0.7, 0.4]),
        "positive_optical_demand_observations": torch.tensor([2, 3], dtype=torch.int16),
        "positive_optical_demand_camera_id": torch.tensor([8, 8], dtype=torch.int32),
        "positive_optical_demand_max_gradient": torch.tensor([0.5, 0.6]),
    }

    factorize, split_detail, audit = _select_shared_envelope_localization_actions(
        foliage, stats, torch.tensor([7, 8]), maximum_groups=4
    )

    assert factorize.numel() == 0
    assert split_detail.tolist() == [1]
    assert audit["existing_unlocalizable_detail_groups_rejected"] == 0


def test_shared_envelope_selector_does_not_oversplit_subpixel_detail():
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(
        {
            "version": "independent_sfm_semantic_canopy_volume_v1",
            "centers": torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]),
            "scales": torch.tensor([[0.3, 0.2, 0.1], [0.02, 0.01, 0.01]]),
            "colors": torch.full((2, 3), 0.4),
            "opacities": torch.full((2, 1), 0.2),
            "quaternions": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]]
            ).repeat(2, 1),
            "static_detail": torch.tensor([False, True]),
            "replacement_group": torch.tensor([0, 0]),
            "support_camera_ids": torch.tensor([[7, 8], [7, 8]]),
            "support_view_count": torch.full((2,), 2, dtype=torch.int16),
            "support_sequence_count": torch.ones(2, dtype=torch.int16),
            "verified_camera_ids": torch.tensor([[7, 8], [7, 8]]),
            "verified_camera_count": torch.full((2,), 2, dtype=torch.int16),
            "verified_sequence_count": torch.ones(2, dtype=torch.int16),
        }
    )
    stats = {
        "rigid_front_conflict_observations": torch.tensor(
            [3, 0], dtype=torch.int16
        ),
        "rigid_front_conflict_camera_id": torch.tensor(
            [7, -1], dtype=torch.int32
        ),
        "rigid_front_conflict_max_gradient": torch.tensor([0.7, 0.0]),
        "positive_optical_demand_observations": torch.tensor(
            [0, 3], dtype=torch.int16
        ),
        "positive_optical_demand_camera_id": torch.tensor(
            [-1, 8], dtype=torch.int32
        ),
        "positive_optical_demand_max_gradient": torch.tensor([0.0, 0.6]),
        "radius": torch.tensor([8.0, 0.8]),
    }

    factorize, split_detail, audit = (
        _select_shared_envelope_localization_actions(
            foliage,
            stats,
            torch.tensor([7, 8]),
            maximum_groups=4,
            minimum_detail_radius_pixels=2.0,
        )
    )

    assert factorize.numel() == 0
    assert split_detail.numel() == 0
    assert audit["insufficient_detail_bandwidth_rows"] == 1
    assert audit["existing_unlocalizable_detail_groups_rejected"] == 1


def test_shared_envelope_open_transaction_rows_include_proposals_and_receivers():
    foliage = SimpleNamespace(
        xyz=torch.zeros(4, 3),
        proposal_kind=torch.tensor(
            [PROPOSAL_NONE, PROPOSAL_SPLIT, PROPOSAL_NONE, PROPOSAL_NONE],
            dtype=torch.int8,
        ),
        verification_state=torch.tensor(
            [
                VERIFICATION_VERIFIED,
                VERIFICATION_MEASURED_SINGLE,
                VERIFICATION_FACTORIZED_RECEIVER,
                VERIFICATION_VERIFIED,
            ],
            dtype=torch.int8,
        ),
    )
    assert _open_shared_envelope_localization_rows(foliage).tolist() == [
        False,
        True,
        True,
        False,
    ]


def test_shared_envelope_transaction_rejects_duplicate_receiver_group():
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(
        {
            "version": "independent_sfm_semantic_canopy_volume_v1",
            "centers": torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]),
            "scales": torch.full((2, 3), 0.1),
            "colors": torch.full((2, 3), 0.4),
            "opacities": torch.full((2, 1), 0.2),
            "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
            "layer_role": torch.zeros(2, dtype=torch.int8),
            "static_detail": torch.tensor([False, True]),
            "replacement_group": torch.tensor([0, 0]),
            "support_camera_ids": torch.tensor([[7, 8], [7, 8]]),
            "support_view_count": torch.full((2,), 2, dtype=torch.int16),
            "support_sequence_count": torch.ones(2, dtype=torch.int16),
            "verified_camera_ids": torch.tensor([[7, 8], [7, 8]]),
            "verified_camera_count": torch.full((2,), 2, dtype=torch.int16),
            "verified_sequence_count": torch.ones(2, dtype=torch.int16),
        }
    )
    stats = {
        name: torch.zeros(2, dtype=dtype)
        for name, dtype in (
            ("rigid_front_conflict_observations", torch.int16),
            ("rigid_front_conflict_camera_id", torch.int32),
            ("rigid_front_conflict_max_gradient", torch.float32),
            ("positive_optical_demand_observations", torch.int16),
            ("positive_optical_demand_camera_id", torch.int32),
            ("positive_optical_demand_max_gradient", torch.float32),
        )
    }
    with pytest.raises(RuntimeError, match="second static-detail receiver"):
        _apply_shared_envelope_localization_model_transaction(
            foliage,
            stats,
            torch.tensor([0]),
            torch.empty(0, dtype=torch.long),
            torch.tensor([[0.0, 0.0, 1.0]] * 9),
            current_iteration=22_100,
            maximum_final_count=8,
            receiver_optical_mass_fraction=0.05,
        )


def test_shared_envelope_localization_window_is_explicit_and_post_topology():
    args = SimpleNamespace(
        reconstruction_target="static",
        maximum_shared_envelope_localizations_per_event=256,
        shared_envelope_localization_start_iteration=22_000,
        shared_envelope_localization_until_iteration=25_500,
        shared_envelope_localization_every=100,
    )
    assert not _shared_envelope_localization_active(
        22_000, args, "ownership_cleanup"
    )
    assert _shared_envelope_localization_active(
        22_100, args, "ownership_cleanup"
    )
    assert not _shared_envelope_localization_active(
        22_101, args, "ownership_cleanup"
    )
    assert not _shared_envelope_localization_active(
        25_600, args, "ownership_cleanup"
    )
    assert not _shared_envelope_localization_active(
        22_100, args, "canonical_polish"
    )
    args.maximum_shared_envelope_localizations_per_event = 0
    assert not _shared_envelope_localization_active(
        22_100, args, "ownership_cleanup"
    )


def test_v116_migration_adds_only_exact_shared_envelope_contract():
    localization = {
        "contract": SHARED_ENVELOPE_OWNERSHIP_LOCALIZATION_CONTRACT,
        "maximum_groups_per_event": 256,
        "every": 100,
        "start_iteration_exclusive": 22_000,
        "until_iteration_inclusive": 25_500,
        "receiver_optical_mass_fraction": 0.05,
        "generic_topology_reopened": False,
    }
    saved = {"stable_identity": "same"}
    current = {
        **saved,
        "shared_envelope_ownership_localization": localization,
    }
    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_v116_shared_envelope_localization_migration=True,
    )
    changed = copy.deepcopy(current)
    changed["shared_envelope_ownership_localization"][
        "maximum_groups_per_event"
    ] = 257
    with pytest.raises(RuntimeError, match="target changed"):
        _resume_training_contract_differences(
            saved,
            changed,
            allow_v116_shared_envelope_localization_migration=True,
        )


def test_topology_stat_reset_maps_persistent_rigid_front_debt():
    class FoliageStub(SimpleNamespace):
        def __len__(self):
            return len(self.xyz)

    prior = {
        "gradient": torch.tensor([4.0, 5.0, 6.0]),
        "rigid_front_conflict_observations": torch.tensor(
            [2, 0, 7], dtype=torch.int16
        ),
        "rigid_front_conflict_camera_id": torch.tensor(
            [9, -1, 12], dtype=torch.int32
        ),
        "rigid_front_conflict_max_gradient": torch.tensor(
            [0.2, 0.0, 0.7]
        ),
        "positive_optical_demand_observations": torch.tensor(
            [1, 3, 0], dtype=torch.int16
        ),
        "positive_optical_demand_camera_id": torch.tensor(
            [4, 5, -1], dtype=torch.int32
        ),
        "positive_optical_demand_max_gradient": torch.tensor(
            [0.4, 0.5, 0.0]
        ),
    }
    foliage = FoliageStub(xyz=torch.zeros(4, 3))
    mapped = _reset_volume_stats_after_topology(
        foliage, prior, torch.tensor([2, 0, 2, 1])
    )

    # Epoch-local screen measurements reset, while exact ownership debt
    # follows the same old-row map used for Adam migration.
    torch.testing.assert_close(mapped["gradient"], torch.zeros(4))
    assert mapped["rigid_front_conflict_observations"].tolist() == [
        7,
        2,
        7,
        0,
    ]
    assert mapped["rigid_front_conflict_camera_id"].tolist() == [
        12,
        9,
        12,
        -1,
    ]
    assert mapped["positive_optical_demand_camera_id"].tolist() == [
        -1,
        4,
        -1,
        5,
    ]

    unchanged = _reset_volume_stats_after_topology(
        FoliageStub(xyz=torch.zeros(3, 3)), prior, None
    )
    assert unchanged["rigid_front_conflict_observations"].tolist() == [
        2,
        0,
        7,
    ]


def test_static_global_cleanup_rejects_soft_rigid_tree_and_unknown_boundaries():
    prehit = torch.full((1, 1, 4), 0.5, requires_grad=True)
    task = {
        "p_rigid": torch.tensor([[0.79, 0.90, 0.90, 0.90]]),
        "p_canopy": torch.tensor([[0.0, 0.21, 0.0, 0.0]]),
        "p_canopy_core": torch.zeros(1, 4),
        "p_sky": torch.zeros(1, 4),
        "p_transient": torch.zeros(1, 4),
        "p_boundary_uncertain": torch.tensor([[0.0, 0.0, 0.11, 0.26]]),
        "p_unknown_ownership": torch.tensor([[0.0, 0.0, 0.11, 0.0]]),
        "w_rgb": torch.ones(1, 4),
    }

    loss, audit = _static_detail_global_cleanup_loss(
        prehit,
        torch.ones_like(prehit),
        task,
        counterfactual_weight=1.0,
    )
    loss.backward()

    assert float(loss) == 0.0
    assert audit["semantic_rigid_candidates_rejected"] == 4
    assert audit["rigid_free_supported_pixels"] == 0
    torch.testing.assert_close(prehit.grad, torch.zeros_like(prehit))


def test_strict_rigid_front_post_step_preserves_realized_retirement():
    foliage = _optical_trust_foliage()
    owners = torch.tensor([True, False, False, False, False])
    iteration_start = foliage.opacity_logits.detach()[owners].clone()
    with torch.no_grad():
        # Adam/settle has already realized 0.4 logit of strict retirement.
        foliage.opacity_logits[0] -= 0.4
    pre_handoff = foliage.opacity_logits.detach()[owners].clone()
    with torch.no_grad():
        # Reversible handoff restores 0.3.  This is still below the iteration
        # start, so a guard against iteration-start regrowth would miss it.
        foliage.opacity_logits[0] += 0.3
        foliage.opacity_logits[1] -= 0.5
    first_moment = torch.full_like(foliage.opacity_logits, -2.0)
    optimizer = SimpleNamespace(
        state={foliage.opacity_logits: {"exp_avg": first_moment}}
    )

    audit = _enforce_strict_rigid_front_post_step(
        foliage, optimizer, owners, pre_handoff
    )

    torch.testing.assert_close(foliage.opacity_logits[owners], pre_handoff)
    assert float(foliage.opacity_logits[owners] - iteration_start) < -0.39
    assert audit["restored_rows_clamped"] == 1
    assert audit["maximum_forbidden_logit_growth"] > 0.29
    assert first_moment[0].item() == 0.0
    assert first_moment[1].item() == -2.0


def test_strict_rigid_front_two_stage_guard_preserves_step_retirement():
    foliage = _optical_trust_foliage()
    opacity = foliage.opacity_logits
    optimizer = SimpleNamespace(
        state={opacity: {"exp_avg": torch.zeros_like(opacity)}}
    )
    owners = torch.tensor([True, False, False, False, False])
    step_start = torch.tensor([[0.5]])
    with torch.no_grad():
        opacity[0] = 0.6

    # A later projection has undone the optimizer's strict retirement and
    # even exceeded the step-start value. The first guard restores the global
    # step ceiling. A following handoff then tries a smaller partial restore;
    # the second guard must preserve the already-realized 0.5 ceiling rather
    # than merely checking against the original 0.5 again.
    first = _enforce_strict_rigid_front_post_step(
        foliage, optimizer, owners, step_start
    )
    assert float(opacity[0]) == pytest.approx(0.5)
    assert first["restored_rows_clamped"] == 1
    pre_handoff = opacity.detach()[owners].clone()
    with torch.no_grad():
        opacity[0] = 0.55
    second = _enforce_strict_rigid_front_post_step(
        foliage, optimizer, owners, pre_handoff
    )
    assert float(opacity[0]) == pytest.approx(0.5)
    assert second["restored_rows_clamped"] == 1


def test_persistent_envelope_cleanup_routes_only_global_negative_signal():
    volume_alpha = torch.full((1, 2, 2), 0.5, requires_grad=True)
    surface_alpha = torch.ones(1, 2, 2, requires_grad=True)
    task = {
        "p_rigid": torch.ones(2, 2),
        "p_canopy": torch.zeros(2, 2),
        "p_canopy_core": torch.zeros(2, 2),
        "p_sky": torch.zeros(2, 2),
        "p_transient": torch.zeros(2, 2),
        "p_boundary_uncertain": torch.zeros(2, 2),
        "p_unknown_ownership": torch.zeros(2, 2),
        "w_rgb": torch.ones(2, 2),
    }

    loss, audit = _persistent_envelope_global_cleanup_loss(
        volume_alpha,
        surface_alpha,
        task,
        counterfactual_weight=0.5,
    )
    loss.backward()

    assert audit["contract"] == PERSISTENT_ENVELOPE_GLOBAL_CLEANUP_CONTRACT
    assert audit["rigid_free_supported_pixels"] == 4
    assert audit["gradient_permissions"] == {
        "persistent_envelope_xyz_scale_rotation": False,
        "persistent_envelope_optical_mass": True,
        "persistent_envelope_sh": False,
        "surface_sky_uncertainty": False,
        "topology_statistics": False,
    }
    assert volume_alpha.grad is not None
    assert float(volume_alpha.grad.min()) > 0
    assert audit["rigid_volume_front_pixels"] == 4
    assert audit["rigid_behind_or_invalid_pixels_rejected"] == 0
    assert surface_alpha.grad is None


@pytest.mark.parametrize(
    "loss_function",
    (_persistent_envelope_global_cleanup_loss,),
)
def test_global_cleanup_strictly_rejects_volume_behind_rigid_surface(loss_function):
    # The native exact prehit query is identically zero for a behind volume.
    volume_alpha = torch.zeros(1, 2, 2, requires_grad=True)
    surface_alpha = torch.ones(1, 2, 2)
    task = {
        "p_rigid": torch.ones(2, 2),
        "p_canopy": torch.zeros(2, 2),
        "p_canopy_core": torch.zeros(2, 2),
        "p_sky": torch.zeros(2, 2),
        "p_transient": torch.zeros(2, 2),
        "p_boundary_uncertain": torch.zeros(2, 2),
        "p_unknown_ownership": torch.zeros(2, 2),
        "w_rgb": torch.ones(2, 2),
    }

    loss, audit = loss_function(
        volume_alpha,
        surface_alpha,
        task,
        counterfactual_weight=1.0,
    )
    loss.backward()

    assert float(loss) == 0.0
    assert audit["rigid_free_supported_pixels"] == 0
    assert audit["rigid_volume_front_pixels"] == 0
    assert audit["rigid_behind_or_invalid_pixels_rejected"] == 4
    # The query channel itself has a positive derivative at zero; CUDA maps it
    # to no primitive because every actual volume event is behind the bound.
    assert volume_alpha.grad is not None


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


def test_counterfactual_transparency_cannot_retire_canopy_without_rigid_permission():
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
    loss, audit = _counterfactual_volume_transparency_loss(
        mixed, surface, target, alpha, torch.ones(1, 1, 2), task
    )
    loss.backward()
    # Surface-better RGB inside a canopy label is diagnostically visible but
    # cannot authorize deletion without an independent rigid posterior.
    assert audit["supported_pixels"] == 0
    assert audit["canopy_rgb_retirement_blocked_pixels"] == 1
    assert audit["canopy_rgb_retirement_blocked_mass"] > 0
    assert torch.equal(alpha.grad, torch.zeros_like(alpha))


def test_failed_split_family_rolls_back_without_optical_hole():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[1.0, 2.0, 3.0]]),
        "scales": torch.tensor([[0.20, 0.16, 0.12]]),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "opacities": torch.tensor([[0.08]]),
        "colors": torch.tensor([[0.2, 0.6, 0.1]]),
        "primitive_role": torch.tensor([0], dtype=torch.int8),
        "track_linearity": torch.tensor([0.0]),
        "track_id": torch.tensor([-1]),
        "tree_instance_id": torch.tensor([7], dtype=torch.int32),
        "support_camera_ids": torch.tensor(
            [[11, 12, 13, 14]], dtype=torch.int32
        ),
        "support_view_count": torch.tensor([4], dtype=torch.int16),
        "support_sequence_count": torch.tensor([2], dtype=torch.int16),
        "occupancy_probability": torch.tensor([0.8]),
        "position_covariance": torch.eye(3)[None] * 0.02,
        "reprojection_error": torch.tensor([0.5]),
        "evidence_primitive_id": torch.tensor([17]),
    }
    foliage = VolumetricFoliageModel(1, dynamic_rank=0, device="cpu")
    foliage.initialize_from_volume_state(payload)
    parent_mass = foliage.integrated_optical_mass().sum().clone()
    parent_physical = {
        name: getattr(foliage, name)[0].detach().clone()
        for name in (
            "xyz",
            "log_scales",
            "quaternions",
            "opacity_logits",
            "features",
        )
    }
    split = foliage.split_adaptive(
        torch.tensor([0]), torch.tensor([2]), birth_iteration=100
    )
    assert split["children"] == 2
    assert len(foliage.split_parent_snapshot_family_id) == 1
    assert foliage.proposal_parent_support_camera_ids.shape == (1, 4)
    assert foliage.proposal_parent_position_covariance.shape == (1, 3, 3)
    torch.testing.assert_close(
        foliage.integrated_optical_mass().sum(), parent_mass
    )
    # A split is an atomic replacement. One verified child cannot authorize
    # throwing away the optical-mass share of its expired sibling.
    # Simulate severe optimizer drift: rollback must restore the transaction
    # snapshot, never merge this separation into one giant Gaussian.
    with torch.no_grad():
        foliage.xyz[0] += torch.tensor([4.0, 0.0, 0.0])
        foliage.xyz[1] -= torch.tensor([4.0, 0.0, 0.0])
        foliage.log_scales.add_(0.7)
        foliage.opacity_logits.add_(0.4)
        foliage.features.add_(0.3)
    foliage.verification_state[0] = VERIFICATION_VERIFIED
    foliage.candidate_evidence_primitive_id[0] = -1

    persistent_stats = {
        "rigid_front_conflict_observations": torch.tensor(
            [3, 5], dtype=torch.int16
        ),
        "rigid_front_conflict_camera_id": torch.tensor(
            [11, 12], dtype=torch.int32
        ),
        "rigid_front_conflict_max_gradient": torch.tensor([0.8, 0.6]),
        "positive_optical_demand_observations": torch.tensor(
            [2, 4], dtype=torch.int16
        ),
        "positive_optical_demand_camera_id": torch.tensor(
            [13, 14], dtype=torch.int32
        ),
        "positive_optical_demand_max_gradient": torch.tensor([0.3, 0.9]),
    }
    remove, representatives, audit = _rollback_failed_split_families(
        SimpleNamespace(
            maximum_volume_family_rollbacks_per_event=8,
            skeleton_opacity_ceiling=0.40,
            dynamic_leaf_opacity_ceiling=0.40,
            canonical_crown_opacity_ceiling=0.20,
        ),
        foliage,
        torch.tensor([False, True]),
        persistent_stats=persistent_stats,
    )

    assert audit["rolled_back_families"] == 1
    assert int(remove.sum()) == 1
    assert len(representatives) == 1
    representative = representatives[0]
    assert audit["persistent_ownership_ledgers_merged"] == 1
    assert persistent_stats[
        "rigid_front_conflict_observations"
    ][representative].item() == 5
    assert persistent_stats[
        "rigid_front_conflict_camera_id"
    ][representative].item() == 11
    assert persistent_stats[
        "rigid_front_conflict_max_gradient"
    ][representative].item() == pytest.approx(0.8)
    assert persistent_stats[
        "positive_optical_demand_observations"
    ][representative].item() == 4
    assert persistent_stats[
        "positive_optical_demand_camera_id"
    ][representative].item() == 14
    assert persistent_stats[
        "positive_optical_demand_max_gradient"
    ][representative].item() == pytest.approx(0.9)
    for name, expected in parent_physical.items():
        torch.testing.assert_close(
            getattr(foliage, name)[representative], expected
        )
    torch.testing.assert_close(
        foliage.integrated_optical_mass()[representative],
        parent_mass,
        rtol=1e-5,
        atol=1e-7,
    )
    assert foliage.evidence_primitive_id[representative].item() == 17
    assert foliage.parent_lineage_id[representative].item() == -1
    assert foliage.verification_state[representative].item() == VERIFICATION_VERIFIED
    assert foliage.proposal_kind[representative].item() == PROPOSAL_NONE
    assert foliage.split_proposal_family_id[representative].item() == -1
    assert len(foliage.split_parent_snapshot_family_id) == 0
    assert foliage.support_view_count[representative].item() == 4
    assert foliage.support_sequence_count[representative].item() == 2
    torch.testing.assert_close(
        foliage.support_camera_ids[representative],
        torch.tensor([11, 12, 13, 14], dtype=torch.int32),
    )
    torch.testing.assert_close(
        foliage.position_covariance[representative],
        torch.eye(3) * 0.02,
    )
    assert foliage.occupancy_probability[representative].item() == pytest.approx(
        0.8
    )
    torch.testing.assert_close(
        foliage.initialization_center[representative],
        torch.tensor([1.0, 2.0, 3.0]),
    )
    assert len(torch.unique(foliage.lineage_id)) == len(foliage)
    foliage.replace_and_split_adaptive(
        remove,
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
    )
    assert len(foliage) == 1
    torch.testing.assert_close(
        foliage.integrated_optical_mass().sum(),
        parent_mass,
        rtol=1e-5,
        atol=1e-7,
    )


def test_verified_localization_family_rearms_child_specific_ownership():
    foliage = VolumetricFoliageModel(1, dynamic_rank=0, device="cpu")
    foliage.initialize_from_volume_state(
        {
            "version": "independent_sfm_semantic_canopy_volume_v1",
            "centers": torch.tensor(
                [[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]
            ),
            "scales": torch.tensor(
                [[0.3, 0.2, 0.1], [0.2, 0.1, 0.1]]
            ),
            "quaternions": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]]
            ).repeat(2, 1),
            "opacities": torch.full((2, 1), 0.2),
            "colors": torch.full((2, 3), 0.4),
            "static_detail": torch.tensor([False, True]),
            "replacement_group": torch.tensor([0, 0]),
            "support_camera_ids": torch.tensor([[7, 8], [7, 8]]),
            "verified_camera_ids": torch.tensor([[7, 8], [7, 8]]),
            "verified_camera_count": torch.tensor([2, 2], dtype=torch.int16),
            "verified_sequence_count": torch.tensor([1, 1], dtype=torch.int16),
        }
    )
    foliage.split_adaptive(
        torch.tensor([1]), torch.tensor([2]), birth_iteration=100
    )
    foliage.verification_state.fill_(VERIFICATION_VERIFIED)
    family_rows = foliage.proposal_kind == PROPOSAL_SPLIT
    assert family_rows.tolist() == [False, True, True]
    stats = {
        "rigid_front_conflict_observations": torch.tensor(
            [9, 4, 4], dtype=torch.int16
        ),
        "rigid_front_conflict_camera_id": torch.tensor(
            [3, 7, 7], dtype=torch.int32
        ),
        "rigid_front_conflict_max_gradient": torch.tensor([0.9, 0.8, 0.8]),
        "positive_optical_demand_observations": torch.tensor(
            [8, 3, 3], dtype=torch.int16
        ),
        "positive_optical_demand_camera_id": torch.tensor(
            [4, 8, 8], dtype=torch.int32
        ),
        "positive_optical_demand_max_gradient": torch.tensor([0.7, 0.6, 0.6]),
    }

    remove, representatives, audit = _rollback_failed_split_families(
        SimpleNamespace(maximum_volume_family_rollbacks_per_event=0),
        foliage,
        torch.zeros(3, dtype=torch.bool),
        persistent_stats=stats,
    )

    assert not bool(remove.any())
    assert representatives.numel() == 0
    assert audit["resolved_verified_families"] == 1
    assert audit["resolved_child_ownership_ledgers_rearmed"] == 2
    assert bool((foliage.proposal_kind == PROPOSAL_NONE).all())
    assert bool((foliage.split_proposal_family_id == -1).all())
    for key, value in stats.items():
        expected = -1 if key.endswith("_camera_id") else 0
        torch.testing.assert_close(
            value[family_rows], torch.full_like(value[family_rows], expected)
        )
    assert stats["rigid_front_conflict_observations"][0].item() == 9
    assert stats["positive_optical_demand_observations"][0].item() == 8


def test_family_rollback_never_groups_ray_births_by_parent_lineage():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor(
            [[0.0, 0.0, 3.0], [0.1, 0.0, 3.0], [0.2, 0.0, 3.0]]
        ),
        "scales": torch.full((3, 3), 0.1),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        "opacities": torch.full((3, 1), 0.04),
        "colors": torch.full((3, 3), 0.5),
    }
    foliage = VolumetricFoliageModel(0, dynamic_rank=0, device="cpu")
    foliage.initialize_from_volume_state(payload)
    foliage.parent_lineage_id.fill_(77)
    foliage.proposal_kind.fill_(PROPOSAL_RAY_BIRTH)
    foliage.split_proposal_family_id.fill_(-1)

    remove, representatives, audit = _rollback_failed_split_families(
        SimpleNamespace(maximum_volume_family_rollbacks_per_event=8),
        foliage,
        torch.ones(3, dtype=torch.bool),
    )

    assert not bool(remove.any())
    assert len(representatives) == 0
    assert audit["rolled_back_families"] == 0


def test_volume_appearance_lr_decays_only_in_topology_free_polish():
    feature = torch.nn.Parameter(torch.ones(1))
    position = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam(
        [
            {"params": [feature], "lr": 0.1, "name": "features"},
            {"params": [position], "lr": 0.02, "name": "xyz"},
        ]
    )
    for group in optimizer.param_groups:
        group["base_lr"] = group["lr"]
    args = SimpleNamespace(
        training_profile="static_handoff_fast",
        phase_schedule_horizon=100,
        volume_polish_final_lr_multiplier=0.10,
    )

    before = _update_volume_learning_rates(
        optimizer, 81, args, "ownership_cleanup"
    )
    assert before["multiplier"] == pytest.approx(1.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    final = _update_volume_learning_rates(
        optimizer, 99, args, "canonical_polish"
    )
    assert final["progress"] == pytest.approx(1.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.02)


def test_chart_atlas_inherits_absolute_rigid_polish_lifecycle():
    atlas = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([{"params": [atlas], "lr": 2.0e-4}])
    optimizer.param_groups[0]["base_lr"] = 2.0e-4
    opt = SimpleNamespace(
        non_position_lr_decay_from=24_000,
        iterations=32_000,
        non_position_lr_final_mult=0.10,
    )

    prefix = _update_chart_atlas_learning_rate(optimizer, 16_000, opt)
    assert prefix["multiplier"] == pytest.approx(1.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-4)
    suffix = _update_chart_atlas_learning_rate(optimizer, 32_000, opt)
    assert suffix["multiplier"] == pytest.approx(0.10)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-5)
