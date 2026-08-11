#!/usr/bin/env python
"""Train one from-scratch mixed outdoor teacher from unified evidence."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import signal
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SURFEL_ROOT = REPO_ROOT / "2d-gaussian-splatting"
sys.path[:0] = [str(REPO_ROOT), str(SURFEL_ROOT)]

from arguments import (  # noqa: E402
    ModelParams,
    OptimizationParams,
    PipelineParams,
)
from outdoor.appearance_uncertainty import (  # noqa: E402
    OutdoorAppearanceUncertainty,
)
from outdoor.chart_surface_model import ChartSurfaceModel  # noqa: E402
from outdoor.directional_sky import (  # noqa: E402
    CanonicalDirectionalSky,
    composite_white_background,
)
from outdoor.evidence_store import (  # noqa: E402
    artifact_path,
    load_evidence_store,
    mast3r_is_geometry_authority,
    sfm_coverage_tracks_enabled,
)
from outdoor.foliage_geometry import (  # noqa: E402
    evidence_conditioned_dynamic_opacity_ceiling,
    evidence_conditioned_leaf_optical_mass,
)
from outdoor.foliage_view_graph import sequence_id  # noqa: E402
from outdoor.hybrid_gaussian_renderer import (  # noqa: E402
    LAYER_CANONICAL_CROWN,
    LAYER_DYNAMIC_LEAF,
    VERIFICATION_UNVERIFIED,
    VERIFICATION_VERIFIED,
    VolumetricFoliageModel,
    _rotation_matrices_from_quaternions,
    dynamic_visibility_gate,
    projected_gaussian_cross_section,
    render_hybrid,
)
from outdoor.hybrid_teacher_api import (  # noqa: E402
    STATIC_RAY_NORMALIZED_OPTICAL_POLICY,
    STATIC_RAY_SURFACE_EVIDENCE_POLICY,
    _static_ray_normalized_optical_mixture,
    _static_surface_evidence_mixture,
)
from outdoor.lazy_scene import LazyScene, rgb_source_contract  # noqa: E402
from outdoor.runtime_provenance import collect_runtime_provenance  # noqa: E402
from outdoor.static_foliage import (  # noqa: E402
    fuse_sequence_evidence_into_static_leaves,
)
from outdoor.static_ray_birth import StaticRayBirthAccumulator  # noqa: E402
from outdoor.task_fields import OutdoorTaskFieldLookup  # noqa: E402
from outdoor.training_evidence import (  # noqa: E402
    FoliageRayEvidence,
    OutdoorGeometryEvidence,
)
from scene import GaussianModel  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402


PREDECESSOR_PROTOCOL = (
    "cambridge_native_hybrid_teacher_v44_static_staged_coverage_topology_"
    "optical_audit"
)
PROTOCOL = (
    "cambridge_native_hybrid_teacher_v84_symmetric_sequence_optical_ownership"
)
STATIC_CANONICAL_OWNERSHIP_REPAIR_PREDECESSOR = {
    "protocol": (
        "cambridge_native_hybrid_teacher_v50_static_scene_snapshot_"
        "canonical_detail_schedule"
    ),
    "iteration": 3_000,
    "trainer": (
        "123a6c446900a079a979c737500208427a698aa2caa06487ee2b25fff03ffb6a"
    ),
    "hybrid_teacher_api": (
        "00d177835e9ddbb55f85cc301d3c4cb8a335c58af17c5fbe89212c64e40410ef"
    ),
}
STATIC_DETAIL_ISOLATED_REPAIR_PREDECESSOR = {
    "protocol": (
        "cambridge_native_hybrid_teacher_v45_static_canonical_viewset_"
        "multimode_optical_audit"
    ),
    "iteration": 3_000,
    "trainer": (
        "70630c734ec6956cd85eff96a37107098b23e8fa53849968d94503771a6b68e9"
    ),
    "hybrid_teacher_api": (
        "b4f15285bc7a380bf2a479ff5b14e9d0240acb6af1c684b0d76be97633148f4c"
    ),
}
STATIC_DETAIL_ISOLATED_APPEARANCE_PREDECESSOR_CONTRACT = (
    "surface_plus_static_detail_counterfactual_routes_rgb_high_frequency_"
    "and_screen_gradient_only_to_static_detail__geometry_and_topology_"
    "remain_support_sequence_owned__verified_cross_sequence_consensus_"
    "may_receive_optical_mass__sh_appearance_uses_all_real_canopy_views__"
    "persistent_envelope_cannot_occlude_its_training_signal"
)
STATIC_DETAIL_ISOLATED_CONTRACT = (
    "surface_plus_static_detail_counterfactual_routes_rgb_high_frequency_"
    "and_screen_gradient_only_to_verified_multiview_static_detail__"
    "unverified_rows_retain_exact_owner_dc_mass_and_ray_training__geometry_"
    "and_topology_remain_exact_support_camera_owned__sh_appearance_uses_"
    "soft_same_sequence_support__opacity_is_read_only__"
    "persistent_envelope_cannot_occlude_its_training_signal"
)
STATIC_STAGE_RGB_ROLE_PREDECESSOR_CONTRACT = (
    "stage2_envelope_rgb_geometry_mass__stage3_envelope_sh_only__"
    "ray_interval_and_global_counterfactual_retain_envelope_geometry_mass__"
    "support_owned_static_detail_rgb_geometry__verified_cross_sequence_"
    "consensus_static_detail_optical_mass"
)
STATIC_STAGE_RGB_ROLE_V83_PREDECESSOR_CONTRACT = (
    "stage2_envelope_rgb_geometry_mass__stage3_envelope_dc_only_no_positive_"
    "mass_growth__"
    "ray_interval_and_global_counterfactual_retain_envelope_geometry_mass__"
    "exact_support_camera_owned_static_detail_geometry_mass__soft_same_"
    "sequence_static_detail_sh__verified_"
    "multiview_only_high_bandwidth_refinement__verified_cross_"
    "sequence_consensus_static_detail_ray_optical_mass"
)
STATIC_STAGE_RGB_ROLE_CONTRACT = (
    "stage2_envelope_rgb_geometry_mass__stage3_envelope_dc_only_no_positive_"
    "mass_growth__ray_interval_and_global_counterfactual_retain_envelope_"
    "geometry_mass__exact_support_camera_owned_static_detail_geometry_and_"
    "topology__positive_optical_mass_and_free_space_use_symmetric_real_"
    "evidence_sequences__same_sequence_static_detail_sh_and_optical_mass_"
    "use_continuous_weights__verified_multiview_only_high_bandwidth_"
    "refinement__verified_cross_sequence_consensus_static_detail_ray_"
    "optical_mass"
)
STATIC_DETAIL_GLOBAL_CLEANUP_APPEARANCE_PREDECESSOR_CONTRACT = (
    "static_detail_is_unconditionally_visible__all_calibrated_views_route_"
    "rigid_free_space_and_surface_better_counterfactuals_only_to_static_"
    "detail_geometry_and_optical_mass__support_sequences_route_positive_"
    "rgb_geometry_and_topology__verified_cross_sequence_consensus_routes_"
    "positive_ray_geometry_and_optical_mass__all_real_canopy_views_route_"
    "sh_appearance_only__cleanup_never_drives_topology"
)
STATIC_DETAIL_GLOBAL_CLEANUP_V83_PREDECESSOR_CONTRACT = (
    "static_detail_is_unconditionally_visible__all_calibrated_views_route_"
    "rigid_free_space_and_surface_better_counterfactuals_only_to_static_"
    "detail_geometry_and_optical_mass__support_sequences_route_positive_"
    "rgb_geometry_optical_mass_sh_and_topology__verified_cross_sequence_"
    "consensus_routes_positive_ray_geometry_and_optical_mass__cleanup_"
    "never_drives_topology"
)
STATIC_DETAIL_GLOBAL_CLEANUP_CONTRACT = (
    "static_detail_is_unconditionally_visible__rigid_free_space_and_surface_"
    "better_counterfactuals_route_only_from_persisted_positive_evidence_"
    "sequences_to_static_detail_geometry_and_optical_mass__the_same_"
    "evidence_sequences_route_positive_rgb_and_ray_optical_mass__exact_"
    "support_cameras_retain_geometry_and_topology__verified_cross_sequence_"
    "consensus_remains_a_positive_ray_owner__cleanup_never_drives_topology"
)
PERSISTENT_ENVELOPE_GLOBAL_CLEANUP_CONTRACT = (
    "persistent_envelope_is_unconditionally_visible__all_calibrated_views_"
    "route_rigid_free_space_and_surface_better_counterfactuals_only_to_"
    "envelope_geometry_and_optical_mass__canopy_rgb_ray_hits_and_local_"
    "replacement_remain_separate_positive_owners__cleanup_never_updates_"
    "envelope_sh_or_drives_topology"
)
STATIC_RAY_BIRTH_VISUAL_HULL_CONTRACT = (
    "uncovered_hit_uses_complete_calibrated_ray_depth_interval__"
    "cross_sequence_segments_vote_in_multiscale_visual_hull_cells__"
    "minimum_two_cameras_and_two_sequences_remain_mandatory__"
    "newborn_is_low_mass_static_detail_and_free_space_prunable__"
    "continuous_verification_debt_backpressure"
)
COUNTERFACTUAL_TRANSPARENCY_CONTRACT = (
    "detached_surface_only_relative_rgb_advantage_requires_independent_"
    "rigid_semantic_permission_and_routes_only_volume_optical_depth__"
    "canopy_semantics_never_authorize_canopy_retirement"
)
RAY_EPOCH_BOUNDARY_CONTRACT = (
    "short_tail_never_wraps_into_next_camera_epoch"
)
RAY_EPOCH_CAPACITY_CONTRACT = (
    "sum_per_camera_ceil_rows_over_batch_lte_scheduled_calls"
)
VOLUME_TOPOLOGY_SETTLE_CONTRACT = (
    "screen_adaptive_until_configured_end__disabled_during_"
    "canonical_polish_settle"
)
VOLUME_OPACITY_SETTLE_CONTRACT = (
    "post_calibrated_optical_mass_freeze_or_retirement_only__base_adam_"
    "growth_momentum_constrained__temporal_opacity_frozen__geometry_scale_"
    "rotation_and_sh_continue__retirement_window_then_automatic_freeze"
)
STATIC_OPTICAL_POLICY_PREDECESSOR_CONTRACT = (
    "single_static_map__envelope_growth_continuously_attenuated_only_by_"
    "persistent_multiview_native_t_before_alpha_ray_local_overlap__same_"
    "group_primitive_proxy_is_candidate_only__occluded_envelope_retires_"
    "reversibly_in_integrated_optical_mass_space__static_detail_and_"
    "skeleton_mass_remain_"
    "trainable__hidden_detail_consumes_exact_ray_posterior_during_topology__"
    "newborn_dc_uses_distinct_view_native_responsibility_rgb__"
    "xyz_scale_rotation_and_mass_freeze_together_in_canonical_"
    "polish__scale_updates_preserve_integrated_optical_mass"
)
STATIC_OPTICAL_POLICY_STAGE3_FREEZE_PREDECESSOR_CONTRACT = (
    "single_static_map__stage3_envelope_positive_growth_and_high_order_sh_"
    "frozen_while_negative_cleanup_and_dc_remain_trainable__verified_support_"
    "camera_detail_groups_define_"
    "handoff_candidates__native_t_before_alpha_pixel_coverage_tracks_"
    "conservative_cross_view_lower_envelope__retirement_is_reversible_in_"
    "integrated_optical_mass_space__failed_zero_owner_witness_split_"
    "families_rollback_mass_conservingly__strict_cross_sequence_ray_birth_"
    "retains_independent_capacity__unverified_detail_keeps_dc_mass_and_ray_"
    "training_but_not_high_bandwidth_refinement__canonical_polish_freezes_"
    "geometry_covariance_and_mass_together__scale_updates_preserve_"
    "integrated_optical_mass"
)
STATIC_OPTICAL_POLICY_CONTRACT = (
    "single_static_map__stage3_envelope_positive_growth_only_from_canonical_"
    "geometry_ray_evidence_while_non_evidence_growth_and_high_order_sh_are_"
    "frozen_and_negative_cleanup_and_dc_remain_trainable__verified_support_"
    "camera_detail_groups_define_"
    "handoff_candidates__native_t_before_alpha_pixel_coverage_tracks_"
    "conservative_cross_view_lower_envelope__retirement_is_reversible_in_"
    "integrated_optical_mass_space__failed_zero_owner_witness_split_"
    "families_rollback_mass_conservingly__strict_cross_sequence_ray_birth_"
    "retains_independent_capacity__unverified_detail_keeps_dc_mass_and_ray_"
    "training_but_not_high_bandwidth_refinement__canonical_polish_freezes_"
    "geometry_covariance_and_mass_together__scale_updates_preserve_"
    "integrated_optical_mass"
)
VOLUME_OPACITY_SETTLE_PREDECESSOR_CONTRACT = (
    "post_calibrated_optical_mass_freeze_or_retirement_only__base_adam_"
    "growth_momentum_constrained__temporal_opacity_frozen__geometry_scale_"
    "rotation_and_sh_continue"
)
CONDITIONED_SETTLE_CONTRACT = (
    "canonical_polish_disables_topology_not_conditioned_optimization"
)
V38_CAUSAL_REPAIR_PREDECESSOR = {
    "protocol": PREDECESSOR_PROTOCOL,
    "trainer": "774a5d9beb6a1a6dad71a067108922c93d86a9ecdf9bdd4fb46d6fb836afdc3c",
    "training_evidence": (
        "aed7a5d0b63755c43e3b9046423fd00afe4dbb70f85c5509d52359f719ce35e3"
    ),
    "iteration": 9_000,
    "schedule_horizon": 12_000,
    "volume_densify_until_iteration": 10_000,
}
REJECTED_INITIALIZATION_PROTOCOLS = {
    # This protocol persisted every in-front-of-posterior projection as a
    # free-space violation, including invalid and rigidly occluded rays. Its
    # first volume topology event can therefore delete valid canonical crown.
    "outdoor-role-aware-initialization-v14-priority-dav2-hole-fill",
}
# Exact hashes of the checkpoint implementation that preceded the
# I/O/allocator-only fast path.  This narrow allow-list lets the active formal
# run resume without weakening the normal implementation/CUDA provenance
# guard for arbitrary code changes.
PERFORMANCE_RESUME_PREDECESSOR = {
    "trainer": "b662682037b4f67a1439a6867a1bf1925520871ebf84efff1602dd1a3bd75357",
    "lazy_scene": "eb1f67b0a0813b55d89933fa710d486e3515548510c2c8cd9811a4103c0f925b",
    "task_fields": "e654646d46ee172d32f96a8540173000327a470dd0bc9097d644ddffe8e155ea",
    # Older checkpoints did not hash the shared mask lookup separately.
    "mask_lookup": None,
}
# Exact implementation pair immediately before the volume split/cap repair.
# Resuming is safe because the checkpoint tensor schema, evidence, camera
# schedules, optimizers and rasterizer kernels are unchanged; only future
# two-child volume split geometry changes from shrink=1.6 to sqrt(2).
VOLUME_SPLIT_REPAIR_PREDECESSOR = {
    "trainer": "61f826d76bceeedf3868d60096f576127489f03f4ee56628e746e18bd38236c2",
    "hybrid_renderer": "ab3676b2fc968cefb60162f4e1265353a49a1444441be568639b6e42a9f6aab9",
}
# Exact implementation pair immediately before the tangent-area preserving
# surface split repair.  The checkpoint schema and optimiser state are
# unchanged; only future replacing splits use sqrt(N) scale shrink and retain
# the parent's opacity instead of losing projected optical mass to a ceiling.
SURFACE_SPLIT_REPAIR_PREDECESSOR = {
    "trainer": "e953bce6d7825c319c9a098f85ba7bb87cd08ad808f83cde47602fbf52743c3c",
    "gaussian_model": "5619eec7dab57f932e17c97cda6e3dda0aecd64910d1b62b49bb2b1aedf61ef4",
}
SURFACE_SCREEN_EVIDENCE_REPAIR_PREDECESSOR = {
    "protocol": (
        "cambridge_native_hybrid_teacher_v67_all_camera_role_posterior_"
        "chart_atlas_replace_only_retirement"
    ),
    "iteration": 2_000,
    "trainer": (
        "b44815ab2c7670acea170e234ec65e3e096127ebcd4f809d5b851f68f345aeb6"
    ),
    "gaussian_model": (
        "77e9902a6788e8ee09dfb9f3c1f56926d23a077cb817cc331b9ed5705ec4e9f3"
    ),
    "hybrid_teacher_api": (
        "516b90acefd8f05957709cae329aa4995f48f8f01fa79aeb6fa42e9344c14dee"
    ),
}
SURFACE_SCREEN_TOPOLOGY_CONTRACT = {
    "footprint_role": "continuous_multiplier_of_accumulated_loss_gradient",
    "responsibility": "current_topology_window_observation_mass",
    "independent_radius_only_split_evidence": False,
    "oversized_support_policy": "replace_split_not_prune",
}
PROJECTED_OCCLUDED_RIGID_GEOMETRY_CONTRACT = (
    "all_camera_stable_track_projection_supervises_surface_only_depth_and_"
    "sparse_coverage__tree_occluded_rows_never_supervise_current_view_rgb"
)
# Exact v82 implementation and optimization identity before the audited
# boundary/capacity repair.  This migration is deliberately narrower than the
# general trainer-repair escape hatch: only the retained 8k checkpoint may
# cross it, and the old/new capacities and topology horizon are fixed below.
VOLUME_CAPACITY_REPAIR_PREDECESSOR = {
    "trainer": "198417986719acf07de202d70efda38aa8a4bf716cb88ceb16c3e4e56cbdeeec",
    "iteration": 8_000,
    "maximum_volume_gaussians": 1_500_000,
    "maximum_volume_splits_per_event": 12_000,
}
VOLUME_CAPACITY_REPAIR_TARGET = {
    "maximum_volume_gaussians": 2_000_000,
    "maximum_volume_splits_per_event": 20_000,
    "volume_densify_until_iteration": 10_000,
}
BOUNDARY_SUPERVISION_CONTRACT = (
    "continuous_semantic_boundary_evidence_floor025_no_binary_gate"
)
VOLUME_REALLOCATION_CONTRACT = (
    "role_matched_lineage_safe_retire_then_evidence_adaptive_split"
)
VOLUME_TOPOLOGY_MUTATION_CONTRACT = (
    "single_materialization_retire_and_adaptive_split_then_single_adam_migration"
)
DYNAMIC_LIFECYCLE_REPAIR_PREDECESSOR = (
    "persistent_observation_identity_survives_sampling_windows"
)
DYNAMIC_LIFECYCLE_REPAIR_TARGET = (
    "observation_identity_is_provenance__strong_confirmed_free_"
    "overrides__weak_conflict_protected_only_by_independent_"
    "cross_sequence_positive_support"
)
RIGID_PATCH_NMS_REPAIR_PREDECESSOR = (
    "detached_rigid_residual_times_target_edge_local_pool"
)
RIGID_PATCH_NMS_REPAIR_TARGET = (
    "detached_rigid_residual_times_target_edge_spatial_nms_pool"
)
CHART_TOPOLOGY_HORIZON_REPAIR_PREDECESSOR = 3_000
CHART_TOPOLOGY_HORIZON_REPAIR_TARGET = 10_000
SURFACE_OWNERSHIP_REPAIR_PREDECESSOR = {
    "trainer": "bf58ca503999a4269c7dfd7bbd6c0331f6d19beb085c86ed69148ade320c9304",
    "iteration": 10_000,
    "mature_handoff_surface_policy": "appearance_only",
}
SURFACE_OWNERSHIP_REPAIR_TARGET = {
    "mature_handoff_surface_policy": "appearance_only",
    "surface_ownership_contract": (
        "frozen_gradient_event_bounded_local_semantic_optical_mass_retirement"
    ),
    "dynamic_replacement_contract": (
        "exact_dynamic_plus_canonical_local_depth_rgb_posterior_"
        "cross_sequence_semantic_responsibility_event_bounded_optical_mass"
    ),
    "rgb_gradient_ownership_contract": (
        "surface_and_sky_from_surface_only_rigid_render__"
        "canopy_rgb_from_jointly_sorted_mixed_render_to_foliage_only__"
        "surface_opacity_frozen_in_adam_and_retired_only_by_local_evidence"
    ),
    "semantic_ownership_contract": (
        "tree_front_depth_claim_plus_sky_transient_cross_sequence_responsibility"
    ),
    "surface_retirement_optical_mass_fraction_per_event": 0.0025,
}
TRAINING_PROFILES = {
    # Final quality schedule.  Keep this profile stable for the eventual
    # benchmark run.
    "quality": {
        "iterations": 80_000,
        "phases": (
            ("canonical_bootstrap", 0.06),
            ("topology", 0.58),
            ("dynamic_appearance", 0.85),
            ("ownership_cleanup", 0.95),
            ("canonical_polish", 1.00),
        ),
        "dynamic_start": 0.06,
    },
    # Reconstruction-quality validation schedule: the dynamic foliage branch
    # starts at 12k instead of 48k, while every phase still sees several full
    # passes over the 1,487 Cambridge training cameras.
    "fast": {
        "iterations": 30_000,
        "phases": (
            ("canonical_bootstrap", 0.08),
            ("topology", 0.40),
            ("dynamic_appearance", 0.73),
            ("ownership_cleanup", 0.90),
            ("canonical_polish", 1.00),
        ),
        "dynamic_start": 0.08,
    },
    # Teacher-final schedules used by the no-student mainline.  Legacy names
    # above remain stable so earlier checkpoints/tests keep their semantics.
    "hybrid_quality": {
        "iterations": 50_000,
        "phases": (
            ("canonical_bootstrap", 0.12),
            ("topology", 0.45),
            ("static_foliage", 0.50),
            ("dynamic_appearance", 0.82),
            ("ownership_cleanup", 0.93),
            ("canonical_polish", 1.00),
        ),
        # Let the volume owner explain canopy pixels as soon as the short
        # rigid bootstrap ends.  Delaying it until surface topology finished
        # forced 2D surfels/sky to fit trees for almost half the run and then
        # asked ownership cleanup to undo that contradiction.
        "foliage_start": 0.12,
        # Start after structural topology, while enough of the global volume
        # budget remains for sequence-conditioned leaves to densify.
        "dynamic_start": 0.50,
    },
    "hybrid_fast": {
        "iterations": 30_000,
        "phases": (
            ("canonical_bootstrap", 0.08),
            ("topology", 0.55),
            ("static_foliage", 0.68),
            ("dynamic_appearance", 0.88),
            ("ownership_cleanup", 0.96),
            ("canonical_polish", 1.00),
        ),
        "foliage_start": 0.08,
        "dynamic_start": 0.68,
    },
    # A mature no-COLMAP native 2DGS scaffold is handed off through the
    # native-rigid-surface-handoff-v1 contract.  Foliage starts immediately;
    # a short, bounded surface-topology window is retained only for newly
    # exposed rigid gaps.  This avoids asking a good facade model to repeat
    # the destructive empty-foliage bootstrap used by from-seed training.
    "hybrid_handoff_quality": {
        "iterations": 30_000,
        "phases": (
            ("canonical_bootstrap", 0.02),
            ("topology", 0.25),
            ("static_foliage", 0.35),
            ("dynamic_appearance", 0.72),
            ("ownership_cleanup", 0.92),
            ("canonical_polish", 1.00),
        ),
        "foliage_start": 0.0,
        # The handoff already contains a mature rigid scaffold and the
        # sequence-local seeds now have exact calibrated ray/depth owners.
        # Keeping them frozen until 10.5k left ownerless tree silhouettes
        # empty for the whole topology window; split cannot create support
        # where no live primitive exists. Retain only a short canonical
        # bootstrap, then let exact-view leaves learn while topology is still
        # able to refine their footprint.
        "dynamic_start": 0.02,
    },
    # Geometry-first diagnostic/foundation stage.  This is intentionally a
    # normal training profile rather than a pass/fail gate: its checkpoint,
    # render and metrics are always inspectable, while foliage/appearance
    # cannot hide an under-reconstructed rigid scaffold.
    "hybrid_rigid_stage1": {
        "iterations": 24_000,
        "phases": (
            ("canonical_bootstrap", 0.08),
            # The historical Cambridge control acquired topology early and
            # then improved for another 16k updates without changing support.
            # Recreating children through 85% of the run made every nominal
            # "polish" checkpoint a fresh, unsettled topology.
            ("topology", 0.50),
            ("canonical_polish", 1.00),
        ),
        "foliage_start": 2.0,
        "dynamic_start": 2.0,
    },
    # Production localization map: sequence/time observations are fused into
    # one persistent static leaf layer before optimization.  There is no
    # conditioned output branch and the final checkpoint is the trained map,
    # not a canonical subset of a larger database-view model.
    "static_handoff_quality": {
        "iterations": 20_000,
        "phases": (
            ("canonical_bootstrap", 0.02),
            # The 408--412 counterfactual series identifies the causal handoff
            # directly: tree PSNR improves through 2k (12.286 dB), then falls
            # to 11.086 dB at 3k while surface-only stays stable and
            # envelope-only collapses.  Waiting for all 639 ray cameras lets
            # broad envelope kernels overfit colour/opacity before local
            # detail replacement can act.  One complete 1,487-camera RGB
            # epoch is sufficient to establish the low-frequency envelope;
            # hand off at 1.5k on the 30k production horizon.  Ray factors
            # and volume topology continue independently, so later evidence
            # is not dropped and this is a smooth lifecycle transition, not
            # a metric gate.
            ("topology", 0.05),
            ("static_foliage", 0.45),
            ("dynamic_appearance", 0.65),
            ("ownership_cleanup", 0.85),
            ("canonical_polish", 1.00),
        ),
        "foliage_start": 0.0,
        "dynamic_start": 2.0,
    },
    "static_handoff_fast": {
        "iterations": 12_000,
        "phases": (
            ("canonical_bootstrap", 0.02),
            ("topology", 0.25),
            ("static_foliage", 0.50),
            ("dynamic_appearance", 0.65),
            ("ownership_cleanup", 0.82),
            ("canonical_polish", 1.00),
        ),
        "foliage_start": 0.0,
        "dynamic_start": 2.0,
    },
}

RIGID_PROFILE_OPTIMIZER_DEFAULTS = {
    "geometry_gradient_ratio": ("--geometry-gradient-ratio", 0.15),
    "position_lr_init": ("--position_lr_init", 1.6e-5),
    "position_lr_final": ("--position_lr_final", 1.6e-6),
    "position_lr_delay_mult": ("--position_lr_delay_mult", 0.01),
}

STATIC_HANDOFF_PROFILE_DEFAULTS = {
    "geometry_gradient_ratio": ("--geometry-gradient-ratio", 0.15),
    "mature_handoff_surface_policy": (
        "--mature-handoff-surface-policy",
        "atlas_residual",
    ),
    "volume_split_radius": ("--volume-split-radius", 2.0),
    "maximum_volume_radius_pixels": (
        "--maximum-volume-radius-pixels",
        24.0,
    ),
    "maximum_skeleton_radius_pixels": (
        "--maximum-skeleton-radius-pixels",
        12.0,
    ),
    "maximum_envelope_radius_pixels": (
        "--maximum-envelope-radius-pixels",
        24.0,
    ),
    "maximum_static_detail_radius_pixels": (
        "--maximum-static-detail-radius-pixels",
        12.0,
    ),
    "volume_polish_final_lr_multiplier": (
        "--volume-polish-final-lr-multiplier",
        0.10,
    ),
}


def _apply_training_profile_optimizer_defaults(
    args,
    cli_tokens: tuple[str, ...] | list[str],
) -> dict:
    """Make a direct trainer invocation obey its named profile.

    The production runner has always supplied the geometry-balanced rigid
    optimizer explicitly, but direct causal runs inherited the upstream 2DGS
    position LR and a disabled adaptive geometry gradient.  Those invocations
    still called themselves ``hybrid_rigid_stage1`` while optimizing a
    materially different method.  Resolve only options that the caller did
    not spell out, so intentional ablations remain possible and auditable.
    """
    tokens = tuple(str(value) for value in cli_tokens)

    def explicitly_set(option: str) -> bool:
        return option in tokens or any(
            token.startswith(option + "=") for token in tokens
        )

    audit: dict[str, object] = {
        "profile": str(args.training_profile),
        "policy": "explicit_cli_else_named_profile_default",
        "resolved": {},
    }
    defaults: dict[str, tuple[str, object]] = {}
    if args.training_profile == "hybrid_rigid_stage1":
        defaults.update(RIGID_PROFILE_OPTIMIZER_DEFAULTS)
        horizon = int(
            getattr(args, "phase_schedule_horizon", 0)
            or getattr(args, "iterations", 0)
            or TRAINING_PROFILES["hybrid_rigid_stage1"]["iterations"]
        )
        defaults.update(
            {
                "position_lr_max_steps": (
                    "--position_lr_max_steps",
                    min(max(horizon, 20_000), 30_000),
                ),
                "non_position_lr_decay_from": (
                    "--non_position_lr_decay_from",
                    int(round(0.75 * horizon)),
                ),
                "non_position_lr_final_mult": (
                    "--non_position_lr_final_mult",
                    0.10,
                ),
            }
        )
    elif args.training_profile in {
        "static_handoff_quality",
        "static_handoff_fast",
    }:
        defaults.update(STATIC_HANDOFF_PROFILE_DEFAULTS)
    for attribute, (option, profile_value) in defaults.items():
        explicit = explicitly_set(option)
        previous = getattr(args, attribute, None)
        if not explicit:
            setattr(args, attribute, profile_value)
        audit["resolved"][attribute] = {
            "value": getattr(args, attribute),
            "source": "explicit_cli" if explicit else "named_profile",
            "parser_default_before_resolution": previous,
        }
    return audit


def _resolve_cpu_intraop_threads(
    available_cpus: int,
    image_prefetch_workers: int,
    requested_threads: int = 0,
) -> int:
    """Bound the process-global CPU pool without starving the GPU feeder.

    Torch exposes one configured intra-op width, but the image-prefetch and
    native codec paths on this stack can still create helper pools from
    multiple Python callers.  Auto mode therefore reserves a core share for
    every caller and caps the configured pool at four.  A production A/B on
    the 32-core Cambridge host measured 3 threads at 0.75 s/step versus 8 at
    0.98 s/step while the latter created 115 process threads.
    """
    available_cpus = int(available_cpus)
    image_prefetch_workers = int(image_prefetch_workers)
    requested_threads = int(requested_threads)
    if available_cpus <= 0:
        raise ValueError("available_cpus must be positive")
    if image_prefetch_workers < 0:
        raise ValueError("image_prefetch_workers must be non-negative")
    if requested_threads < 0:
        raise ValueError("requested_threads must be non-negative")
    if requested_threads > 0:
        return min(requested_threads, available_cpus)
    consumers = max(image_prefetch_workers + 1, 1)
    return max(1, min(4, available_cpus // consumers))


def _configure_cpu_parallelism(args) -> dict[str, object]:
    try:
        available_cpus = len(os.sched_getaffinity(0))
        affinity_source = "sched_getaffinity"
    except (AttributeError, OSError):
        available_cpus = int(os.cpu_count() or 1)
        affinity_source = "os_cpu_count"
    intraop_threads = _resolve_cpu_intraop_threads(
        available_cpus,
        int(args.image_prefetch_workers),
        int(args.cpu_intraop_threads),
    )
    interop_threads = min(
        int(args.cpu_interop_threads), available_cpus
    )
    torch.set_num_threads(intraop_threads)
    torch.set_num_interop_threads(interop_threads)
    opencv_threads = None
    try:
        import cv2

        # Zero asks OpenCV to execute sequentially rather than creating a
        # second nested worker pool underneath Python/Torch prefetch threads.
        cv2.setNumThreads(0)
        opencv_threads = int(cv2.getNumThreads())
    except ImportError:
        pass
    return {
        "contract": (
            "bounded_process_global_torch_openmp_no_nested_opencv_pool"
        ),
        "available_cpus": int(available_cpus),
        "affinity_source": affinity_source,
        "image_prefetch_workers": int(args.image_prefetch_workers),
        "requested_intraop_threads": int(args.cpu_intraop_threads),
        "resolved_intraop_threads": int(torch.get_num_threads()),
        "resolved_interop_threads": int(torch.get_num_interop_threads()),
        "opencv_threads": opencv_threads,
    }


def _validate_initialization_protocol(initialization: dict) -> None:
    version = str(initialization.get("version", ""))
    if version in REJECTED_INITIALIZATION_PROTOCOLS:
        raise RuntimeError(
            f"Initialization protocol {version!r} contains invalid "
            "free-space contradiction metadata; rebuild the initialization "
            "with v15 or newer before training"
        )


def _trainer_repair_hash_change_is_allowed(
    changed: set[str],
    *,
    enabled: bool,
    allow_hybrid_renderer_topology_extension: bool = False,
) -> bool:
    """Authorize only model-state-compatible Python repair resumes.

    A causal repair can live entirely in the Chart/evidence module while the
    trainer entrypoint itself stays byte-identical.  Requiring ``trainer`` to
    change made the explicit repair flag unusable for exactly that case.
    CUDA, renderer and Gaussian-model changes remain excluded.
    """
    allowed = {
        "trainer",
        "training_evidence",
        "static_ray_birth",
        "chart_surface_model",
        # The public deployment compositor is hashed into the checkpoint for
        # reproducible evaluation but is not called by the training loop.
        "hybrid_teacher_api",
        # The training-only uncertainty field has an explicit zero-init
        # optimizer migration.
        "appearance_uncertainty",
    }
    if allow_hybrid_renderer_topology_extension:
        # v83 adds a topology-only, co-located optical-depth factorization
        # method. It does not change the renderer forward/backward or any
        # pre-existing tensor schema. Keep this outside the generic allowlist
        # so arbitrary renderer changes remain resume-incompatible.
        allowed.add("hybrid_renderer")
    return bool(
        enabled
        and changed
        and changed.issubset(allowed)
    )


def _validate_initialization_rgb_source(
    initialization: dict, training_rgb_source: dict
) -> None:
    """Require exact identity/content agreement for RGB-conditioned seeds."""
    initialization_rgb = initialization.get("rgb_source")
    exact_dynamic_rgb = (
        initialization.get("foliage", {}).get(
            "dense_dynamic_rgb_source"
        )
        == "exact_training_target_raster"
    )
    if initialization_rgb is None:
        if exact_dynamic_rgb:
            raise RuntimeError(
                "Exact dynamic-ray RGB initialization has no persisted RGB "
                "source contract"
            )
        return
    compared_fields = (
        "image_count",
        "name_set_sha256",
        "content_mapping_sha256",
        "producer_manifest_sha256",
        "target_storage",
        "canonical_image_size_wh",
    )
    differences = {
        field: {
            "initialization": initialization_rgb.get(field),
            "training": training_rgb_source.get(field),
        }
        for field in compared_fields
        if initialization_rgb.get(field) != training_rgb_source.get(field)
    }
    if differences:
        raise RuntimeError(
            "Initialization/training RGB source contract mismatch: "
            + json.dumps(differences, sort_keys=True)
        )


def _validate_fixed_cameras(views, scene_contract: dict) -> dict:
    """Prove that training uses the exact named Cambridge K/pose records."""
    raw_records = list(scene_contract["records"])
    records = {
        Path(record["image_name"]).stem: record
        for record in raw_records
    }
    if len(records) != len(raw_records):
        raise RuntimeError(
            "Evidence camera contract contains duplicate image identities"
        )
    loaded_names = {Path(str(view.image_name)).stem for view in views}
    expected_names = set(records)
    if loaded_names != expected_names or len(views) != len(records):
        raise RuntimeError(
            "Teacher cameras do not match the evidence contract: "
            f"loaded={len(loaded_names)}, expected={len(expected_names)}"
        )
    maximum = {
        "fx": 0.0,
        "fy": 0.0,
        "cx": 0.0,
        "cy": 0.0,
        "rotation": 0.0,
        "translation": 0.0,
    }
    for view in views:
        record = records[Path(str(view.image_name)).stem]
        camera = record["camera"]
        scale_x = float(view.image_width) / float(camera["width"])
        scale_y = float(view.image_height) / float(camera["height"])
        for name, actual, expected in (
            ("fx", view.focal_x, float(camera["fx"]) * scale_x),
            ("fy", view.focal_y, float(camera["fy"]) * scale_y),
            ("cx", view.cx, float(camera["cx"]) * scale_x),
            ("cy", view.cy, float(camera["cy"]) * scale_y),
        ):
            maximum[name] = max(
                maximum[name], abs(float(actual) - float(expected))
            )
        expected_world_to_camera = np.asarray(
            record["T_world_to_camera"], dtype=np.float64
        )
        maximum["rotation"] = max(
            maximum["rotation"],
            float(
                np.max(
                    np.abs(
                        np.asarray(view.R, dtype=np.float64).T
                        - expected_world_to_camera[:3, :3]
                    )
                )
            ),
        )
        maximum["translation"] = max(
            maximum["translation"],
            float(
                np.max(
                    np.abs(
                        np.asarray(view.T, dtype=np.float64)
                        - expected_world_to_camera[:3, 3]
                    )
                )
            ),
        )
    tolerance = {"intrinsics": 1e-5, "pose": 1e-8}
    if (
        max(maximum[name] for name in ("fx", "fy", "cx", "cy"))
        > tolerance["intrinsics"]
        or max(maximum[name] for name in ("rotation", "translation"))
        > tolerance["pose"]
    ):
        raise RuntimeError(
            "Loaded camera calibration differs from the immutable evidence "
            f"contract: maximum_absolute_error={maximum}"
        )
    return {
        "camera_count": len(views),
        "exact_image_identity": True,
        "maximum_absolute_error": maximum,
        "tolerance": tolerance,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.set_defaults(
        iterations=None,
        data_device="cpu",
        white_background=True,
        densify_until_iter=None,
        densification_interval=100,
        # This mainline starts from calibrated geometry seeds.  Repeated
        # global alpha resets were erasing the entire scaffold (including
        # freshly mass-conserving split children) every few thousand steps.
        opacity_reset_interval=0,
        # Match the proven native 2DGS outdoor control.  A 5e-4 threshold
        # retained tens of thousands of optically dead Chart witnesses and
        # caused the bounded model to stop reallocating capacity at 400k.
        opacity_cull=0.005,
        lambda_dssim=0.20,
    )
    parser.add_argument("--evidence-store", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument(
        "--surface-warmstart-ply",
        type=Path,
        help=(
            "Optional provenance-closed native rigid 2DGS surface. This "
            "replaces only the renderer initialization; foliage and all "
            "external factors still come from --initialization."
        ),
    )
    parser.add_argument(
        "--surface-warmstart-manifest",
        type=Path,
        help="Required native-rigid-surface-handoff-v1 provenance manifest.",
    )
    parser.add_argument(
        "--maximum-rigid-completion-seeds",
        type=int,
        default=20_000,
        help=(
            "Maximum independently supported MASt3R/Chart surface seeds to "
            "append where a mature rigid handoff has a measured spatial "
            "coverage deficit. Set to zero to disable this evidence-only "
            "background-completion stream."
        ),
    )
    parser.add_argument(
        "--training-profile",
        choices=tuple(TRAINING_PROFILES),
        default="quality",
    )
    parser.add_argument(
        "--reconstruction-target",
        choices=("static", "sequence_conditioned_legacy"),
        default="static",
        help=(
            "Production target. 'static' fuses sequence observations into "
            "one persistent localization map and never requires or trains a "
            "sequence-conditioned renderer branch. The legacy option exists "
            "only for reproducing older checkpoints."
        ),
    )
    parser.add_argument(
        "--static-leaf-min-supporting-views",
        type=int,
        default=2,
        help=(
            "Independent calibrated observations required before a former "
            "camera-owned leaf hypothesis becomes a persistent static leaf "
            "cluster."
        ),
    )
    parser.add_argument(
        "--static-canonical-sequence-policy",
        choices=("scene", "per_tree"),
        default="per_tree",
        help=(
            "Static production selects one internally consistent acquisition "
            "per independent tree instance. This preserves trees absent from "
            "one global sequence while still emitting one unconditional "
            "static map. 'scene' is retained as a strict counterfactual."
        ),
    )
    parser.add_argument(
        "--static-detail-canonical-ownership",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep static detail visible from every camera, but route its "
            "RGB, high-frequency, ray and screen-topology gradients only "
            "from calibrated sequences recorded in its support table. "
            "Disable only for the cross-time averaging ablation."
        ),
    )
    parser.add_argument(
        "--static-detail-same-sequence-appearance-weight",
        type=float,
        default=0.35,
        help=(
            "Continuous SH/RGB permission for calibrated cameras in the "
            "same acquisition sequence as a static-detail support camera. "
            "Geometry, opacity, ray ownership and topology remain exact-"
            "camera/verified-evidence owned. Set to zero for the former "
            "strict exact-camera appearance ablation."
        ),
    )
    parser.add_argument(
        "--static-detail-same-sequence-optical-weight",
        type=float,
        default=0.35,
        help=(
            "Continuous opacity/RGB permission for cameras in a persisted "
            "positive-evidence sequence of a static-detail primitive. Exact "
            "support cameras retain weight one; geometry and topology remain "
            "exact-camera/verified owned. This closes the former asymmetry "
            "where any camera could delete a leaf but only one seed camera "
            "could restore its optical coverage."
        ),
    )
    parser.add_argument(
        "--integrated-optical-mass-compensation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Preserve tau times projected cross-section across every Adam "
            "scale update and physical scale clamp."
        ),
    )
    parser.add_argument(
        "--static-replacement-ema-decay",
        type=float,
        default=0.90,
        help=(
            "EMA decay for active-view, depth-overlap leaf-to-envelope "
            "replacement evidence."
        ),
    )
    parser.add_argument(
        "--static-replacement-evidence-every",
        type=int,
        default=50,
        help=(
            "Cadence for envelope-only/detail-only real-pixel replacement "
            "responsibility audits. Zero disables hand-off evidence."
        ),
    )
    parser.add_argument(
        "--static-replacement-mass-fraction-per-event",
        type=float,
        default=0.02,
        help=(
            "Maximum reversible envelope optical-mass fraction transferred "
            "per real-pixel audit event."
        ),
    )
    parser.add_argument(
        "--child-verification-grace-iterations",
        type=int,
        default=500,
        help=(
            "Iterations during which an unverified split/ray child may grow "
            "but cannot be ordinarily pruned or split again."
        ),
    )
    parser.add_argument(
        "--child-verification-timeout-iterations",
        type=int,
        default=3000,
        help=(
            "Age after which an unverified, zero-witness, low-contribution "
            "and low-opacity child may be retired. This is long enough for "
            "roughly two complete Cambridge camera epochs and does not "
            "affect contributing or partially verified children."
        ),
    )
    parser.add_argument(
        "--volume-verification-debt-soft-fraction",
        type=float,
        default=0.05,
        help=(
            "Fraction of unverified volume rows at which ordinary screen-"
            "space splitting starts to be attenuated. Evidence-driven ray "
            "births and contradiction pruning remain active."
        ),
    )
    parser.add_argument(
        "--volume-verification-debt-hard-fraction",
        type=float,
        default=0.12,
        help=(
            "Fraction of unverified volume rows at which further ordinary "
            "splitting reaches zero. The transition from the soft fraction "
            "is smooth rather than a binary topology gate."
        ),
    )
    parser.add_argument(
        "--volume-verification-debt-minimum-split-scale",
        type=float,
        default=0.02,
        help=(
            "Minimum ordinary split capacity retained at the hard debt "
            "fraction. Verification debt is backpressure, not a topology "
            "deadlock: a small role-balanced exploration budget remains "
            "available while stale unsupported children are retired."
        ),
    )
    parser.add_argument(
        "--volume-verification-debt-minimum-birth-scale",
        type=float,
        default=0.25,
        help=(
            "Minimum capacity reserved for strict cross-camera and cross-"
            "sequence ray/depth posterior births. These candidates already "
            "carry stronger evidence than ordinary footprint splits and "
            "must not be starved by debt created by existing lineages."
        ),
    )
    parser.add_argument(
        "--phase-schedule-horizon",
        type=int,
        default=None,
        help=(
            "Absolute iteration horizon used to resolve all named phases, "
            "branch activations, evidence-anchor release, default topology "
            "end, volume cadence and camera schedules. It is independent of "
            "--iterations, which is only the requested stop iteration. "
            "Short runs with the same horizon are therefore exact prefixes "
            "of longer runs."
        ),
    )
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--dynamic-rank", type=int, default=4)
    parser.add_argument("--dynamic-seed-count", type=int, default=16_000)
    parser.add_argument(
        "--dynamic-evidence-view-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of conditioned-render steps assigned to a balanced "
            "cycle over cameras with exact foliage ray/depth observations. "
            "By default it is solved from a 12x per-view evidence-camera "
            "oversampling ratio instead of assigning a fixed 75%% of every "
            "scene to a possibly tiny keyframe set. The canonical RGB "
            "schedule remains a full epoch over all views."
        ),
    )
    parser.add_argument("--volume-position-lr", type=float, default=8e-5)
    parser.add_argument("--volume-feature-lr", type=float, default=8e-4)
    parser.add_argument(
        "--volume-opacity-lr",
        type=float,
        default=None,
        help=(
            "Adam learning rate for volumetric optical depth. It defaults "
            "to 4e-3 for every profile. Sparse exact-owner updates are "
            "handled by the camera schedule; accelerating opacity alone "
            "forms an opaque low-frequency layer before colour and topology "
            "have acquired the corresponding bandwidth."
        ),
    )
    parser.add_argument(
        "--volume-opacity-settle-policy",
        choices=("none", "freeze", "retirement_only"),
        default=None,
        help=(
            "Policy after the calibrated optical-mass stage. 'freeze' "
            "holds base and temporal opacity fixed; 'retirement_only' still "
            "allows base opacity to decrease while preventing further "
            "optical-mass growth. Geometry, covariance and SH colour remain "
            "trainable. The handoff profile defaults to retirement_only."
        ),
    )
    parser.add_argument(
        "--volume-opacity-settle-start-iteration",
        type=int,
        default=None,
        help=(
            "Last iteration with unconstrained volume-opacity Adam updates. "
            "For the mature handoff profile this defaults to half the fixed "
            "schedule horizon; other profiles default to the configured "
            "volume-topology end."
        ),
    )
    parser.add_argument(
        "--volume-opacity-retirement-until-iteration",
        type=int,
        default=None,
        help=(
            "Last iteration of retirement-only base-opacity refinement. "
            "After this point opacity is frozen while geometry/covariance/"
            "SH continue. The handoff default is a bounded 25%%-horizon "
            "window after settle begins, preventing indefinite thinning."
        ),
    )
    parser.add_argument("--volume-scale-lr", type=float, default=4e-4)
    parser.add_argument("--volume-rotation-lr", type=float, default=2e-4)
    parser.add_argument("--dynamic-lr", type=float, default=3e-4)
    parser.add_argument("--appearance-lr", type=float, default=8e-4)
    parser.add_argument("--sky-lr", type=float, default=2e-3)
    parser.add_argument(
        "--volume-polish-final-lr-multiplier",
        type=float,
        default=1.0,
        help=(
            "Final exponential LR multiplier for volume SH, spatial "
            "appearance uncertainty and sky during canonical_polish. Static "
            "named profiles default to 0.1; geometry/covariance/opacity are "
            "already frozen by the static ownership policy."
        ),
    )
    parser.add_argument("--geometry-every", type=int, default=2)
    parser.add_argument("--topology-every", type=int, default=8)
    parser.add_argument("--replacement-every", type=int, default=100)
    parser.add_argument("--geometry-weight", type=float, default=0.12)
    parser.add_argument(
        "--projected-rigid-depth-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the all-camera stable-track surface-only depth and "
            "sparse coverage factor. Tree-occluded rows never supervise "
            "current-view RGB. The RGB-reconstruction mainline keeps this "
            "at zero because the v86/v90 causal runs did not improve the "
            "aggregate rigid image metrics; pass an explicit positive "
            "weight only for the independently evaluated localization-"
            "geometry experiment."
        ),
    )
    parser.add_argument(
        "--dav2-observation-patch-weight",
        type=float,
        default=0.15,
        help=(
            "Weight of the source-view rigid RGB patch factor attached to "
            "posterior DAV2 hole observations. The factor samples the "
            "stored real camera pixels and remains external to Gaussian "
            "topology, so split children cannot erase its supervision."
        ),
    )
    parser.add_argument(
        "--pointmap-weight",
        type=float,
        default=0.12,
        help=(
            "Weight of dense fixed-world MASt3R pointmap ray/depth factors. "
            "These remain external observations after renderer seeds split."
        ),
    )
    parser.add_argument(
        "--allow-missing-pointmap-cross-sequence-posterior",
        action="store_true",
        help=(
            "Explicit ablation only: allow pointmap cameras without the "
            "immutable independent-traversal posterior. Missing rows remain "
            "at precision 0.03 and are never labelled cross-sequence "
            "supported. Production training fails closed by default."
        ),
    )
    parser.add_argument("--plane-weight", type=float, default=0.10)
    parser.add_argument("--normal-weight", type=float, default=0.04)
    parser.add_argument("--ordinal-weight", type=float, default=0.015)
    parser.add_argument("--track-weight", type=float, default=0.05)
    parser.add_argument(
        "--geometry-gradient-ratio",
        type=float,
        default=0.0,
        help=(
            "When positive, rescale each scheduled geometry update so its "
            "surface-xyz gradient norm targets this fraction of the current "
            "RGB/ownership gradient. The scale is detached and bounded; "
            "relative Chart/plane/track weights remain unchanged."
        ),
    )
    parser.add_argument(
        "--structure-weight",
        type=float,
        default=0.0,
        help=(
            "Legacy point-to-random-structure-center factor. Keep disabled; "
            "Chart native ray/UV factors carry MAtCha/G4 geometry."
        ),
    )
    parser.add_argument("--chart-anchor-weight", type=float, default=0.05)
    parser.add_argument(
        "--chart-atlas-lr",
        type=float,
        default=2e-3,
        help=(
            "Learning rate of the bounded coarse/fine Chart inverse-depth "
            "residual. The base calibrated depth and camera rays are fixed."
        ),
    )
    parser.add_argument(
        "--chart-atlas-regularization-weight",
        type=float,
        default=2e-3,
    )
    parser.add_argument(
        "--chart-quadtree-growth-fraction",
        type=float,
        default=0.30,
        help=(
            "Fraction of each surface net-growth budget reserved for "
            "error/radius-driven UV four-child replace-and-retire."
        ),
    )
    parser.add_argument(
        "--chart-quadtree-maximum-level",
        type=int,
        default=3,
        help=(
            "Maximum UV subdivision depth. The 128x72 fine atlas needs at "
            "most three binary levels to reach a 640x360 RGB target."
        ),
    )
    parser.add_argument("--ownership-weight", type=float, default=0.08)
    parser.add_argument(
        "--conditioned-ownership-weight", type=float, default=0.20
    )
    parser.add_argument(
        "--counterfactual-transparency-weight",
        type=float,
        default=0.06,
        help=(
            "Weight of the detached surface-only RGB counterfactual. It "
            "reduces volume optical depth only where the existing rigid "
            "surface explains the target better than the jointly rendered "
            "volume, including facade pixels inside a coarse canopy mask."
        ),
    )
    parser.add_argument(
        "--counterfactual-transparency-margin",
        type=float,
        default=0.02,
        help=(
            "RGB-L1 advantage required before the smooth counterfactual "
            "responsibility strongly favors surface transparency."
        ),
    )
    parser.add_argument(
        "--counterfactual-transparency-temperature",
        type=float,
        default=0.015,
        help=(
            "Temperature of the continuous surface-versus-mixed "
            "responsibility; this is never used as a hard ownership gate."
        ),
    )
    parser.add_argument(
        "--conditioned-counterfactual-every",
        type=int,
        default=4,
        help=(
            "Evaluate the extra conditioned surface-only counterfactual at "
            "this cadence. The canonical pass reuses its existing rigid "
            "render every iteration."
        ),
    )
    parser.add_argument("--occupancy-weight", type=float, default=0.06)
    parser.add_argument("--dynamic-weight", type=float, default=0.65)
    parser.add_argument("--appearance-weight", type=float, default=0.06)
    parser.add_argument("--high-frequency-weight", type=float, default=0.08)
    parser.add_argument(
        "--rigid-residual-patch-weight",
        type=float,
        default=0.08,
        help=(
            "Weight of the error/edge-ranked local rigid patch pool used for "
            "facade, window, railing and tree/building-boundary repair."
        ),
    )
    parser.add_argument(
        "--maximum-surface-gaussians",
        type=int,
        default=200_000,
    )
    parser.add_argument(
        "--maximum-surface-growth-per-event",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--maximum-surface-growth-multiplier",
        type=float,
        default=10.0,
        help=(
            "Safety cap relative to verified initialization. Cross-view "
            "consensus intentionally produces a sparse scaffold, so the "
            "default must still permit the known-good ~351k rigid capacity."
        ),
    )
    parser.add_argument(
        "--maximum-surface-scale",
        type=float,
        default=0.5,
        help=(
            "Scene-unit safety ceiling for a surface tangent scale. "
            "Projected footprint is controlled independently by "
            "--maximum-surface-radius-pixels; using the foliage voxel size "
            "as this ceiling under-covers distant rigid facades."
        ),
    )
    parser.add_argument(
        "--maximum-surface-radius-pixels",
        type=float,
        default=24.0,
        help=(
            "Per-view screen-space surface footprint ceiling. Oversized "
            "surfels are shrunk immediately and prioritized for local split."
        ),
    )
    parser.add_argument(
        "--bootstrap-evidence-release-fraction",
        type=float,
        help=(
            "Release temporary SfM/Chart renderer seeds after bootstrap. "
            "Defaults to the end of canonical_bootstrap."
        ),
    )
    parser.add_argument(
        "--bootstrap-evidence-scale-growth", type=float, default=1.5
    )
    parser.add_argument(
        "--bootstrap-evidence-opacity-ceiling", type=float, default=0.50
    )
    parser.add_argument("--maximum-volume-scale", type=float, default=0.25)
    parser.add_argument(
        "--maximum-dynamic-volume-scale",
        type=float,
        default=0.12,
        help=(
            "World-space scale ceiling for sequence-conditioned leaves. "
            "Dynamic high-frequency detail must not start as a coarser "
            "volume than the canonical crown."
        ),
    )
    parser.add_argument("--maximum-volume-gaussians", type=int, default=300_000)
    parser.add_argument("--maximum-volume-splits", type=int, default=1500)
    parser.add_argument(
        "--maximum-volume-growth-multiplier", type=float, default=12.0
    )
    parser.add_argument("--volume-split-radius", type=float, default=3.0)
    parser.add_argument("--volume-densify-every", type=int)
    parser.add_argument(
        "--volume-densify-until-iteration",
        type=int,
        default=None,
        help=(
            "Absolute last iteration at which volume topology may mutate. "
            "By default this is the end of dynamic_appearance, preserving "
            "the historical profile. An explicit value may extend measured "
            "screen-bandwidth refinement into ownership cleanup without "
            "changing the loss or camera schedules."
        ),
    )
    parser.add_argument(
        "--volume-topology-ramp-iterations",
        type=int,
        default=None,
        help=(
            "Smoothstep ramp from zero to the configured per-event volume "
            "growth budget. This lets ray-hit optical mass form before "
            "screen-bandwidth refinement starts. The handoff profile "
            "defaults to 8%% of the fixed phase horizon; other profiles "
            "default to zero."
        ),
    )
    parser.add_argument(
        "--mature-handoff-surface-policy",
        choices=(
            "joint",
            "atlas_residual",
            "appearance_only",
            "frozen",
        ),
        default="joint",
        help=(
            "Optimization ownership for a validated rigid handoff. "
            "atlas_residual keeps the mature surface geometry fixed, "
            "continues the calibrated Chart inverse-depth atlas at low "
            "learning rate, and trains only the appended evidence-driven "
            "coverage suffix in geometry/opacity. "
            "appearance_only preserves its xyz/scale/rotation/opacity and "
            "disables further surface topology while retaining SH colour "
            "polish; local surface retirement is applied separately from "
            "Adam using accumulated cross-sequence ownership evidence and an "
            "event optical-mass budget; frozen also disables SH updates. The "
            "default remains joint unless the pipeline explicitly requests "
            "preservation."
        ),
    )
    parser.add_argument(
        "--surface-retirement-optical-mass-fraction-per-event",
        type=float,
        default=0.0025,
        help=(
            "Maximum fraction of current structural optical depth that a "
            "single local replacement audit may remove. Retirement is "
            "allocated continuously by accumulated semantic/depth/RGB "
            "responsibility, never by an Adam opacity gradient."
        ),
    )
    parser.add_argument(
        "--mature-chart-atlas-lr-scale",
        type=float,
        default=0.10,
        help=(
            "Learning-rate multiplier for Chart inverse-depth continuation "
            "under --mature-handoff-surface-policy atlas_residual."
        ),
    )
    parser.add_argument(
        "--skeleton-opacity-ceiling", type=float, default=0.25
    )
    parser.add_argument(
        "--canonical-crown-opacity-ceiling", type=float, default=0.35
    )
    parser.add_argument(
        "--optical-replacement-policy",
        choices=("view_depth_local", "group_projected", "disabled"),
        default="disabled",
        help=(
            "Legacy/diagnostic renderer replacement policy. Production static "
            "training and deployment always use one disabled-policy native "
            "mixed pass; persistent envelope-to-detail handoff is measured by "
            "separate role-isolated evidence renders and applied to parameters."
        ),
    )
    parser.add_argument(
        "--deployment-optical-replacement-policy",
        choices=(
            "view_depth_local",
            "group_projected",
            "disabled",
            STATIC_RAY_NORMALIZED_OPTICAL_POLICY,
            STATIC_RAY_SURFACE_EVIDENCE_POLICY,
        ),
        default="disabled",
        help=(
            "Legacy deployment compositor selector. The production static "
            "teacher contract requires disabled so train and deployment share "
            "the same one-pass native image formation."
        ),
    )
    parser.add_argument(
        "--deployment-optical-responsibility-prior",
        type=float,
        default=0.0,
        help=(
            "Legacy static ray-normalized deployment pseudo-count. Production "
            "static export requires zero."
        ),
    )
    parser.add_argument(
        "--static-detail-isolated-every",
        type=int,
        default=2,
        help=(
            "For a static reconstruction, render surface+detail without the "
            "persistent envelope at this cadence once the detail stage is "
            "active. Its RGB/high-frequency and screen-space gradients are "
            "routed only to static detail. Set to zero only for an explicit "
            "ablation."
        ),
    )
    parser.add_argument(
        "--static-detail-isolated-weight",
        type=float,
        default=1.0,
        help=(
            "Weight of the branch-isolated static-detail RGB objective. The "
            "high-frequency term keeps the normal global high-frequency "
            "weight and is scaled by the same value."
        ),
    )
    parser.add_argument(
        "--static-skeleton-isolated-every",
        type=int,
        default=8,
        help=(
            "Render surface plus the static trunk/branch role on a camera "
            "with cross-sequence track evidence at this cadence. This "
            "supplies role-specific RGB, edge and screen-bandwidth evidence "
            "without asking the crown envelope to represent branches."
        ),
    )
    parser.add_argument(
        "--static-skeleton-isolated-weight",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--static-volume-isolated-every",
        type=int,
        default=8,
        help=(
            "Render a true volume-only intrinsic-colour counterfactual at "
            "this cadence. It routes canopy RGB/edge/screen-bandwidth only "
            "to static volume rows and never grants opacity growth."
        ),
    )
    parser.add_argument(
        "--static-volume-isolated-weight",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--static-detail-global-cleanup-every",
        type=int,
        default=2,
        help=(
            "For a static reconstruction, route all-camera rigid spill, "
            "confirmed free-space and surface-better counterfactuals to "
            "static-detail geometry/scale/rotation/optical mass at this "
            "cadence. Appearance stays read-only and this pass never feeds "
            "topology. Set to zero only for the ownership-conflict ablation."
        ),
    )
    parser.add_argument(
        "--static-detail-global-cleanup-weight",
        type=float,
        default=0.25,
        help=(
            "Weight of the all-view static-detail negative-evidence pass. "
            "Positive ray hits and RGB remain canonical-support-owned."
        ),
    )
    parser.add_argument(
        "--static-detail-global-counterfactual-weight",
        type=float,
        default=1.0,
        help=(
            "Relative weight of surface-better counterfactual evidence "
            "inside the all-view static-detail cleanup objective. This is "
            "intentionally independent of --counterfactual-transparency-"
            "weight: the latter belongs to the canonical RGB pass and "
            "must not silently attenuate the cleanup pass a second time."
        ),
    )
    parser.add_argument(
        "--persistent-envelope-global-cleanup-every",
        type=int,
        default=2,
        help=(
            "For a static reconstruction, render the persistent crown "
            "envelope without leaf detail and route all-camera rigid spill, "
            "free-space and surface-better counterfactuals only to envelope "
            "geometry/scale/rotation/optical mass. Appearance stays read-only "
            "and this pass never feeds topology."
        ),
    )
    parser.add_argument(
        "--persistent-envelope-global-cleanup-weight",
        type=float,
        default=0.125,
        help=(
            "Weight of the all-view persistent-envelope negative-evidence "
            "pass. This is intentionally weaker than leaf-detail cleanup "
            "because the envelope remains responsible for verified low-"
            "frequency crown coverage."
        ),
    )
    parser.add_argument(
        "--persistent-envelope-global-counterfactual-weight",
        type=float,
        default=1.0,
        help=(
            "Relative surface-better counterfactual weight inside the "
            "envelope-only cleanup pass, independent of the canonical full-"
            "render counterfactual weight."
        ),
    )
    parser.add_argument(
        "--static-detail-exclusive-topology",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After the persistent-envelope stage, keep its mass trainable "
            "and contradiction-prunable but spend canonical split capacity "
            "only on static detail. Disable only for a staged-topology "
            "ablation."
        ),
    )
    parser.add_argument(
        "--canonical-child-support-color-refresh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Refresh split canonical visual-hull children from their bounded "
            "cross-sequence support-camera set. The negative form is an "
            "explicit causal diagnostic; exact-depth canonical observations "
            "and exact-owner dynamic refresh remain enabled."
        ),
    )
    parser.add_argument(
        "--dynamic-leaf-opacity-ceiling", type=float, default=0.40
    )
    parser.add_argument(
        "--maximum-volume-radius-pixels", type=float, default=32.0
    )
    parser.add_argument(
        "--maximum-skeleton-radius-pixels",
        type=float,
        default=None,
        help=(
            "Projected-radius ceiling for the static trunk/branch role. "
            "Defaults to --maximum-volume-radius-pixels outside a named "
            "static handoff profile."
        ),
    )
    parser.add_argument(
        "--maximum-envelope-radius-pixels",
        type=float,
        default=None,
        help=(
            "Projected-radius ceiling for the persistent low-frequency "
            "crown envelope."
        ),
    )
    parser.add_argument(
        "--maximum-static-detail-radius-pixels",
        type=float,
        default=None,
        help=(
            "Projected-radius ceiling for static leaf/detail Gaussians."
        ),
    )
    parser.add_argument(
        "--maximum-volume-family-rollbacks-per-event",
        type=int,
        default=256,
        help=(
            "Maximum failed, entirely unverified split lineage families "
            "merged back to one evidence-supported representative per event."
        ),
    )
    parser.add_argument("--ray-posterior-every", type=int, default=4)
    parser.add_argument(
        "--ray-posterior-maximum-rays", type=int, default=512
    )
    parser.add_argument(
        "--ray-posterior-maximum-candidates",
        type=int,
        default=96,
    )
    parser.add_argument(
        "--static-ray-birth-voxel-size",
        type=float,
        default=0.15,
        help=(
            "Exact posterior-midpoint consensus voxel retained for legacy "
            "pending evidence and interval-free unit inputs."
        ),
    )
    parser.add_argument(
        "--static-ray-birth-visual-hull-voxel-size",
        type=float,
        default=0.30,
        help=(
            "Voxel size used to intersect complete cross-sequence foliage "
            "hit intervals. It must be no smaller than the midpoint voxel; "
            "promotion still requires two cameras and two sequences."
        ),
    )
    parser.add_argument(
        "--static-ray-birth-maximum-segment-samples",
        type=int,
        default=16,
        help=(
            "Maximum bounded samples used to voxelize one calibrated hit "
            "interval for static visual-hull consensus."
        ),
    )
    parser.add_argument(
        "--maximum-static-ray-births-per-event",
        type=int,
        default=2_048,
        help=(
            "Maximum strict cross-camera/cross-sequence visual-hull births "
            "materialized at one volume topology event. This is only an "
            "allocator bound; evidence requirements and the global volume "
            "budget remain unchanged."
        ),
    )
    parser.add_argument(
        "--maximum-static-detail-materializations-per-event",
        type=int,
        default=2_048,
        help=(
            "Maximum visible, evidence-supported envelope cells without any "
            "local detail owner that are factorized into co-located static-"
            "detail receivers at one topology event. This spends ordinary "
            "global volume capacity but creates no optical mass."
        ),
    )
    parser.add_argument(
        "--static-detail-materialization-mass-fraction",
        type=float,
        default=0.05,
        help=(
            "Fraction of a missing-detail envelope cell's optical depth "
            "transferred to its co-located detail receiver. The operation "
            "preserves total optical mass and the birth-time forward image."
        ),
    )
    parser.add_argument(
        "--static-ray-birth-additive-mass-fraction-per-event",
        type=float,
        default=0.005,
        help=(
            "Maximum explicit optical-mass increase allocated to strict "
            "cross-camera/cross-sequence visual-hull births which have no "
            "ray-local envelope donor. The fraction is relative to current "
            "volume mass at one topology event; donor-associated transfers "
            "remain exactly mass conserving."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=3000,
        help=(
            "Rolling checkpoint cadence. The full mixed state is about "
            "1.5 GB at the production budget, so the default aligns with "
            "the retained 3k quality milestones."
        ),
    )
    parser.add_argument(
        "--early-checkpoint-every",
        type=int,
        default=0,
        help=(
            "Optional denser rolling-checkpoint cadence before the first "
            "normal milestone. This changes only failure recovery, never "
            "the optimization schedule."
        ),
    )
    parser.add_argument(
        "--early-checkpoint-until",
        type=int,
        default=0,
        help=(
            "Inclusive last iteration for --early-checkpoint-every. Both "
            "arguments must be zero or both must be positive."
        ),
    )
    parser.add_argument(
        "--retain-checkpoint-iterations",
        type=int,
        nargs="*",
        default=(),
        help=(
            "Iterations whose checkpoint must also be retained under an "
            "immutable iteration-qualified name. This prevents the rolling "
            "checkpoint from erasing a better pre-cleanup state."
        ),
    )
    parser.add_argument(
        "--appearance-maximum-rgb-residual", type=float, default=0.04
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-indices", default="408,409,410")
    parser.add_argument("--view-cache-size", type=int, default=4)
    parser.add_argument("--image-prefetch-workers", type=int, default=2)
    parser.add_argument("--image-prefetch-depth", type=int, default=4)
    parser.add_argument(
        "--cpu-intraop-threads",
        type=int,
        default=0,
        help=(
            "Torch/OpenMP threads per calling Python thread. Zero divides "
            "the available CPU affinity across the trainer and image "
            "prefetch workers, capped at four."
        ),
    )
    parser.add_argument(
        "--cpu-interop-threads",
        type=int,
        default=1,
        help="Global Torch inter-op worker count (must be positive).",
    )
    parser.add_argument("--maintenance-every", type=int, default=1000)
    parser.add_argument("--allow-performance-resume", action="store_true")
    parser.add_argument(
        "--allow-trainer-repair-resume",
        action="store_true",
        help=(
            "Allow a checkpoint to cross a trainer-only causal repair while "
            "still requiring identical evidence, schedules, budgets, model "
            "implementations, and CUDA kernels."
        ),
    )
    parser.add_argument(
        "--allow-static-detail-isolated-repair-resume",
        action="store_true",
        help=(
            "Resume only the exact v45 3k checkpoint into v46. The new "
            "detail-isolated path is inactive throughout that prefix, while "
            "camera schedules, evidence, optimizer/model state and CUDA "
            "kernels remain identical."
        ),
    )
    parser.add_argument(
        "--allow-static-canonical-ownership-repair-resume",
        action="store_true",
        help=(
            "Resume only an exact v50 3k envelope prefix into the v51 "
            "visible canonical-detail schedule repair."
        ),
    )
    parser.add_argument(
        "--allow-conditioned-schedule-repair-resume",
        action="store_true",
        help=(
            "Allow only the conditioned-camera schedule to migrate from the "
            "legacy in-place evidence replacement to the coverage-preserving "
            "weighted epoch. Evidence, model/CUDA implementations, all other "
            "schedules, optimizer state and training geometry remain bound "
            "to the resumed checkpoint. Use together with "
            "--allow-trainer-repair-resume."
        ),
    )
    parser.add_argument(
        "--allow-volume-split-repair-resume",
        action="store_true",
        help=(
            "Resume only the exact audited predecessor across the "
            "mass-safe sqrt(2) volumetric split repair. Evidence, schedules, "
            "state schemas, optimizers and CUDA kernels must remain identical."
        ),
    )
    parser.add_argument(
        "--allow-surface-split-repair-resume",
        action="store_true",
        help=(
            "Resume only the exact audited predecessor across the "
            "tangent-area/optical-mass preserving 2D surfel split repair. "
            "Evidence, schedules, state schemas, optimizers and CUDA kernels "
            "must remain identical."
        ),
    )
    parser.add_argument(
        "--allow-surface-screen-evidence-repair-resume",
        action="store_true",
        help=(
            "Resume only the exact v67 2k pre-topology checkpoint into the "
            "residual-conditioned screen-footprint repair. Model/optimizer "
            "tensors, cameras, evidence and schedules remain bit-identical; "
            "only future surface split priority changes."
        ),
    )
    parser.add_argument(
        "--allow-volume-capacity-repair-resume",
        action="store_true",
        help=(
            "Resume only the exact retained v82 8k checkpoint into the "
            "audited 2M/20k/10k role-conserving capacity repair. This does "
            "not permit arbitrary trainer or optimization-contract changes."
        ),
    )
    parser.add_argument(
        "--allow-volume-topology-settle-resume",
        action="store_true",
        help=(
            "Resume the exact 6k/12k-horizon diagnostic checkpoint while "
            "ending volume topology at 6k. Camera/loss/ray schedules, model "
            "state, evidence, budgets and CUDA kernels remain unchanged; "
            "only post-6k split/retire churn is disabled. Use together with "
            "--allow-trainer-repair-resume."
        ),
    )
    parser.add_argument(
        "--allow-volume-opacity-settle-resume",
        action="store_true",
        help=(
            "Resume the exact audited 6k/12k-horizon checkpoint into either "
            "post-6k opacity freeze or retirement-only optical refinement. "
            "Camera/loss schedules, tensors, evidence and CUDA kernels stay "
            "unchanged. Use together with --allow-trainer-repair-resume."
        ),
    )
    parser.add_argument(
        "--allow-surface-ownership-repair-resume",
        action="store_true",
        help=(
            "Resume only the exact retained v83 10k checkpoint into the "
            "cross-sequence local surface-opacity retirement repair. Adam "
            "keeps opacity and rigid xyz/scale/rotation immutable; only the "
            "event-bounded evidence allocator may reduce optical mass."
        ),
    )
    parser.add_argument(
        "--allow-v38-causal-repair-resume",
        action="store_true",
        help=(
            "Resume only the exact v35 9k/12k-horizon checkpoint into the "
            "resume-aware exact per-camera ray epoch, counterfactual-"
            "transparency and settled volume-topology repair. Cameras, "
            "evidence, model/CUDA state and all sampling schedules remain "
            "identical. Use together with --allow-trainer-repair-resume."
        ),
    )
    parser.add_argument("--resume", type=Path)
    cli_tokens = tuple(sys.argv[1:])
    args = parser.parse_args()
    profile = TRAINING_PROFILES[args.training_profile]
    if args.iterations is None:
        args.iterations = int(profile["iterations"])
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    static_ray_policies = {
        STATIC_RAY_NORMALIZED_OPTICAL_POLICY,
        STATIC_RAY_SURFACE_EVIDENCE_POLICY,
    }
    if args.deployment_optical_responsibility_prior < 0:
        parser.error(
            "--deployment-optical-responsibility-prior must be non-negative"
        )
    if (
        args.reconstruction_target == "static"
        and (
            args.deployment_optical_replacement_policy != "disabled"
            or args.deployment_optical_responsibility_prior != 0
        )
    ):
        parser.error(
            "Production static deployment requires "
            "--deployment-optical-replacement-policy disabled and "
            "--deployment-optical-responsibility-prior 0"
        )
    if (
        args.reconstruction_target != "static"
        and args.deployment_optical_replacement_policy
        not in static_ray_policies
        and args.deployment_optical_responsibility_prior != 0
    ):
        parser.error(
            "--deployment-optical-responsibility-prior must be zero for a "
            "non-ray-normalized deployment policy"
        )
    if args.static_leaf_min_supporting_views < 2:
        parser.error("--static-leaf-min-supporting-views must be >= 2")
    if not 0.0 <= args.static_detail_same_sequence_appearance_weight <= 1.0:
        parser.error(
            "--static-detail-same-sequence-appearance-weight must be in "
            "[0, 1]"
        )
    if not 0.0 <= args.static_detail_same_sequence_optical_weight <= 1.0:
        raise ValueError(
            "--static-detail-same-sequence-optical-weight must be in "
            "[0, 1]"
        )
    if args.static_detail_isolated_every < 0:
        parser.error("--static-detail-isolated-every must be >= 0")
    if args.static_detail_isolated_weight < 0:
        parser.error("--static-detail-isolated-weight must be >= 0")
    if args.static_skeleton_isolated_every < 0:
        parser.error("--static-skeleton-isolated-every must be >= 0")
    if args.static_skeleton_isolated_weight < 0:
        parser.error("--static-skeleton-isolated-weight must be >= 0")
    if args.static_volume_isolated_every < 0:
        parser.error("--static-volume-isolated-every must be >= 0")
    if args.static_volume_isolated_weight < 0:
        parser.error("--static-volume-isolated-weight must be >= 0")
    if args.static_detail_global_cleanup_every < 0:
        parser.error("--static-detail-global-cleanup-every must be >= 0")
    if args.static_detail_global_cleanup_weight < 0:
        parser.error("--static-detail-global-cleanup-weight must be >= 0")
    if args.static_detail_global_counterfactual_weight < 0:
        parser.error(
            "--static-detail-global-counterfactual-weight must be >= 0"
        )
    if args.persistent_envelope_global_cleanup_every < 0:
        parser.error(
            "--persistent-envelope-global-cleanup-every must be >= 0"
        )
    if args.persistent_envelope_global_cleanup_weight < 0:
        parser.error(
            "--persistent-envelope-global-cleanup-weight must be >= 0"
        )
    if args.persistent_envelope_global_counterfactual_weight < 0:
        parser.error(
            "--persistent-envelope-global-counterfactual-weight must be >= 0"
        )
    if args.static_ray_birth_voxel_size <= 0:
        parser.error("--static-ray-birth-voxel-size must be positive")
    if (
        args.static_ray_birth_visual_hull_voxel_size
        < args.static_ray_birth_voxel_size
    ):
        parser.error(
            "--static-ray-birth-visual-hull-voxel-size must be >= "
            "--static-ray-birth-voxel-size"
        )
    if args.static_ray_birth_maximum_segment_samples < 2:
        parser.error(
            "--static-ray-birth-maximum-segment-samples must be >= 2"
        )
    if args.maximum_static_ray_births_per_event <= 0:
        parser.error(
            "--maximum-static-ray-births-per-event must be positive"
        )
    if args.maximum_static_detail_materializations_per_event < 0:
        parser.error(
            "--maximum-static-detail-materializations-per-event must be "
            "non-negative"
        )
    if not 0.0 < args.static_detail_materialization_mass_fraction <= 0.10:
        parser.error(
            "--static-detail-materialization-mass-fraction must lie in "
            "(0,0.1]"
        )
    if not (
        0.0
        <= args.static_ray_birth_additive_mass_fraction_per_event
        <= 0.02
    ):
        parser.error(
            "--static-ray-birth-additive-mass-fraction-per-event must lie "
            "in [0,0.02]"
        )
    if args.allow_static_detail_isolated_repair_resume and args.resume is None:
        parser.error(
            "--allow-static-detail-isolated-repair-resume requires --resume"
        )
    if (
        args.allow_surface_screen_evidence_repair_resume
        and args.resume is None
    ):
        parser.error(
            "--allow-surface-screen-evidence-repair-resume requires --resume"
        )
    if (
        args.allow_static_canonical_ownership_repair_resume
        and args.resume is None
    ):
        parser.error(
            "--allow-static-canonical-ownership-repair-resume requires "
            "--resume"
        )
    if not 0.0 <= args.static_replacement_ema_decay < 1.0:
        parser.error("--static-replacement-ema-decay must lie in [0,1)")
    if args.static_replacement_evidence_every < 0:
        parser.error("--static-replacement-evidence-every must be >= 0")
    if not 0.0 <= args.static_replacement_mass_fraction_per_event <= 0.10:
        parser.error(
            "--static-replacement-mass-fraction-per-event must lie in [0,0.1]"
        )
    if args.child_verification_grace_iterations < 0:
        parser.error("--child-verification-grace-iterations must be >= 0")
    if (
        args.child_verification_timeout_iterations
        <= args.child_verification_grace_iterations
    ):
        parser.error(
            "--child-verification-timeout-iterations must exceed the grace "
            "interval"
        )
    if not (
        0.0
        <= args.volume_verification_debt_soft_fraction
        < args.volume_verification_debt_hard_fraction
        <= 1.0
    ):
        parser.error(
            "volume verification debt fractions must satisfy "
            "0 <= soft < hard <= 1"
        )
    for name in (
        "volume_verification_debt_minimum_split_scale",
        "volume_verification_debt_minimum_birth_scale",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must lie in [0,1]")
    if args.training_profile.startswith("static_") and (
        args.reconstruction_target != "static"
    ):
        parser.error("static_* profiles require --reconstruction-target static")
    if args.phase_schedule_horizon is None:
        args.phase_schedule_horizon = int(profile["iterations"])
    if args.phase_schedule_horizon <= 0:
        parser.error("--phase-schedule-horizon must be positive")
    # Horizon-dependent optimizer defaults must be resolved only after both
    # the requested stop and immutable method horizon are known.  Resolving
    # them immediately after argparse silently used parser defaults for a
    # direct 32k rigid invocation.
    args.training_profile_optimizer_resolution = (
        _apply_training_profile_optimizer_defaults(args, cli_tokens)
    )
    if (
        args.allow_conditioned_schedule_repair_resume
        and (
            args.resume is None
            or not args.allow_trainer_repair_resume
        )
    ):
        parser.error(
            "--allow-conditioned-schedule-repair-resume requires both "
            "--resume and --allow-trainer-repair-resume"
        )
    if args.allow_v38_causal_repair_resume and (
        args.resume is None or not args.allow_trainer_repair_resume
    ):
        parser.error(
            "--allow-v38-causal-repair-resume requires both --resume and "
            "--allow-trainer-repair-resume"
        )
    if args.allow_volume_topology_settle_resume and (
        args.resume is None or not args.allow_trainer_repair_resume
    ):
        parser.error(
            "--allow-volume-topology-settle-resume requires both --resume "
            "and --allow-trainer-repair-resume"
        )
    if args.allow_volume_opacity_settle_resume and (
        args.resume is None or not args.allow_trainer_repair_resume
    ):
        parser.error(
            "--allow-volume-opacity-settle-resume requires both --resume "
            "and --allow-trainer-repair-resume"
        )
    if args.counterfactual_transparency_weight < 0:
        parser.error("--counterfactual-transparency-weight cannot be negative")
    if args.counterfactual_transparency_margin < 0:
        parser.error("--counterfactual-transparency-margin cannot be negative")
    if args.counterfactual_transparency_temperature <= 0:
        parser.error(
            "--counterfactual-transparency-temperature must be positive"
        )
    if args.conditioned_counterfactual_every <= 0:
        parser.error("--conditioned-counterfactual-every must be positive")
    if args.maximum_rigid_completion_seeds < 0:
        parser.error("--maximum-rigid-completion-seeds cannot be negative")
    if args.iterations > args.phase_schedule_horizon:
        parser.error(
            "--iterations cannot exceed --phase-schedule-horizon; choose "
            "the final method horizon up front so shorter runs remain exact "
            "prefixes"
        )
    if (args.surface_warmstart_ply is None) != (
        args.surface_warmstart_manifest is None
    ):
        parser.error(
            "--surface-warmstart-ply and --surface-warmstart-manifest "
            "must be provided together"
        )
    if args.maximum_surface_gaussians <= 0:
        parser.error("--maximum-surface-gaussians must be positive")
    if args.maximum_surface_growth_per_event <= 0:
        parser.error(
            "--maximum-surface-growth-per-event must be positive"
        )
    if args.maximum_surface_growth_multiplier < 1.0:
        parser.error("--maximum-surface-growth-multiplier must be >= 1")
    if args.chart_atlas_lr <= 0:
        parser.error("--chart-atlas-lr must be positive")
    if args.chart_atlas_regularization_weight < 0:
        parser.error(
            "--chart-atlas-regularization-weight cannot be negative"
        )
    if not 0 <= args.chart_quadtree_growth_fraction <= 1:
        parser.error(
            "--chart-quadtree-growth-fraction must lie in [0, 1]"
        )
    if args.chart_quadtree_maximum_level < 0:
        parser.error("--chart-quadtree-maximum-level cannot be negative")
    if args.rigid_residual_patch_weight < 0:
        parser.error("--rigid-residual-patch-weight cannot be negative")
    if args.maximum_volume_growth_multiplier < 1.0:
        parser.error("--maximum-volume-growth-multiplier must be >= 1")
    if args.maximum_surface_scale <= 0:
        parser.error("--maximum-surface-scale must be positive")
    if args.maximum_surface_radius_pixels <= 0:
        parser.error("--maximum-surface-radius-pixels must be positive")
    if args.bootstrap_evidence_scale_growth < 1:
        parser.error("--bootstrap-evidence-scale-growth must be >= 1")
    if not 0 < args.bootstrap_evidence_opacity_ceiling < 1:
        parser.error(
            "--bootstrap-evidence-opacity-ceiling must lie in (0, 1)"
        )
    if args.maximum_volume_scale <= 0:
        parser.error("--maximum-volume-scale must be positive")
    if args.maximum_dynamic_volume_scale <= 0:
        parser.error("--maximum-dynamic-volume-scale must be positive")
    if args.maximum_volume_radius_pixels <= 0:
        parser.error("--maximum-volume-radius-pixels must be positive")
    for attribute in (
        "maximum_skeleton_radius_pixels",
        "maximum_envelope_radius_pixels",
        "maximum_static_detail_radius_pixels",
    ):
        value = getattr(args, attribute)
        if value is None:
            value = float(args.maximum_volume_radius_pixels)
            setattr(args, attribute, value)
        if float(value) <= 0:
            parser.error(
                f"--{attribute.replace('_', '-')} must be positive"
            )
    if args.maximum_volume_family_rollbacks_per_event < 0:
        parser.error(
            "--maximum-volume-family-rollbacks-per-event must be non-negative"
        )
    if not 0.0 < args.volume_polish_final_lr_multiplier <= 1.0:
        parser.error(
            "--volume-polish-final-lr-multiplier must lie in (0, 1]"
        )
    if not (
        0.0
        <= args.surface_retirement_optical_mass_fraction_per_event
        <= 0.05
    ):
        parser.error(
            "--surface-retirement-optical-mass-fraction-per-event must "
            "lie in [0, 0.05]"
        )
    if not 0.0 < args.mature_chart_atlas_lr_scale <= 1.0:
        parser.error("--mature-chart-atlas-lr-scale must lie in (0, 1]")
    if args.volume_opacity_lr is None:
        args.volume_opacity_lr = 4e-3
    if args.volume_opacity_lr <= 0:
        parser.error("--volume-opacity-lr must be positive")
    if args.volume_topology_ramp_iterations is None:
        args.volume_topology_ramp_iterations = (
            int(round(0.08 * args.phase_schedule_horizon))
            if args.training_profile
            in {
                "hybrid_handoff_quality",
                "static_handoff_quality",
                "static_handoff_fast",
            }
            else 0
        )
    if args.volume_topology_ramp_iterations < 0:
        parser.error(
            "--volume-topology-ramp-iterations cannot be negative"
        )
    for name in (
        "skeleton_opacity_ceiling",
        "canonical_crown_opacity_ceiling",
        "dynamic_leaf_opacity_ceiling",
    ):
        value = float(getattr(args, name))
        if not 0.0 < value < 1.0:
            parser.error(
                f"--{name.replace('_', '-')} must lie in (0, 1)"
            )
    if (
        args.dynamic_evidence_view_fraction is not None
        and not 0.0 <= args.dynamic_evidence_view_fraction <= 1.0
    ):
        parser.error(
            "--dynamic-evidence-view-fraction must lie in [0, 1]"
        )
    if args.ray_posterior_every <= 0:
        parser.error("--ray-posterior-every must be positive")
    if args.ray_posterior_maximum_rays <= 0:
        parser.error("--ray-posterior-maximum-rays must be positive")
    if args.ray_posterior_maximum_candidates <= 0:
        parser.error(
            "--ray-posterior-maximum-candidates must be positive"
        )
    if args.geometry_gradient_ratio < 0:
        parser.error("--geometry-gradient-ratio must be non-negative")
    if args.projected_rigid_depth_weight < 0:
        parser.error("--projected-rigid-depth-weight must be non-negative")
    if args.image_prefetch_workers < 0:
        parser.error("--image-prefetch-workers must be non-negative")
    if args.image_prefetch_depth < 0:
        parser.error("--image-prefetch-depth must be non-negative")
    if args.cpu_intraop_threads < 0:
        parser.error("--cpu-intraop-threads must be non-negative")
    if args.cpu_interop_threads <= 0:
        parser.error("--cpu-interop-threads must be positive")
    if args.maintenance_every < 0:
        parser.error("--maintenance-every must be non-negative")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be non-negative")
    if args.early_checkpoint_every < 0:
        parser.error("--early-checkpoint-every must be non-negative")
    if args.early_checkpoint_until < 0:
        parser.error("--early-checkpoint-until must be non-negative")
    if bool(args.early_checkpoint_every) != bool(
        args.early_checkpoint_until
    ):
        parser.error(
            "--early-checkpoint-every and --early-checkpoint-until must "
            "both be zero or both be positive"
        )
    if args.early_checkpoint_until > args.iterations:
        parser.error(
            "--early-checkpoint-until cannot exceed --iterations"
        )
    retained = tuple(
        sorted(set(int(value) for value in args.retain_checkpoint_iterations))
    )
    if any(value <= 0 or value > args.iterations for value in retained):
        parser.error(
            "--retain-checkpoint-iterations must lie in [1, --iterations]"
        )
    args.retain_checkpoint_iterations = retained
    if args.volume_densify_every is None:
        # Six hundred steps yielded only sixteen volume events by 16k while
        # opacity saturated on 17--180 px foliage footprints. Hybrid runs
        # need subdivision on the same order of cadence as surface topology.
        target_cadence = (
            200
            if args.training_profile
            in {
                "hybrid_quality",
                "hybrid_fast",
                "static_handoff_quality",
                "static_handoff_fast",
            }
            else 600
        )
        args.volume_densify_every = min(
            target_cadence,
            max(100, int(args.phase_schedule_horizon) // 150),
        )
    if args.volume_densify_every <= 0:
        parser.error("--volume-densify-every must be positive")
    historical_volume_phases = {
        "topology",
        "static_foliage",
        "dynamic_appearance",
    }
    historical_volume_end = max(
        end
        for name, end in profile["phases"]
        if name in historical_volume_phases
    )
    if args.volume_densify_until_iteration is None:
        args.volume_densify_until_iteration = int(
            args.phase_schedule_horizon * historical_volume_end
        )
    if (
        args.volume_densify_until_iteration <= 0
        or args.volume_densify_until_iteration
        > args.phase_schedule_horizon
    ):
        parser.error(
            "--volume-densify-until-iteration must lie in [1, "
            "--phase-schedule-horizon]"
        )
    if args.volume_opacity_settle_policy is None:
        args.volume_opacity_settle_policy = (
            "retirement_only"
            if (
                args.training_profile == "hybrid_handoff_quality"
                and args.reconstruction_target
                == "sequence_conditioned_legacy"
            )
            else "none"
        )
    if args.volume_opacity_settle_start_iteration is None:
        if (
            args.training_profile == "hybrid_handoff_quality"
            and args.reconstruction_target
            == "sequence_conditioned_legacy"
        ):
            args.volume_opacity_settle_start_iteration = min(
                int(args.volume_densify_until_iteration),
                int(round(0.50 * args.phase_schedule_horizon)),
            )
        else:
            args.volume_opacity_settle_start_iteration = int(
                args.volume_densify_until_iteration
            )
    if not (
        0
        <= args.volume_opacity_settle_start_iteration
        <= args.phase_schedule_horizon
    ):
        parser.error(
            "--volume-opacity-settle-start-iteration must lie in [0, "
            "--phase-schedule-horizon]"
        )
    if args.volume_opacity_retirement_until_iteration is None:
        if args.volume_opacity_settle_policy == "retirement_only":
            args.volume_opacity_retirement_until_iteration = min(
                int(args.phase_schedule_horizon),
                int(args.volume_opacity_settle_start_iteration)
                + int(round(0.25 * args.phase_schedule_horizon)),
            )
        elif args.volume_opacity_settle_policy == "freeze":
            args.volume_opacity_retirement_until_iteration = int(
                args.volume_opacity_settle_start_iteration
            )
        else:
            args.volume_opacity_retirement_until_iteration = int(
                args.phase_schedule_horizon
            )
    if not (
        args.volume_opacity_settle_start_iteration
        <= args.volume_opacity_retirement_until_iteration
        <= args.phase_schedule_horizon
    ):
        parser.error(
            "--volume-opacity-retirement-until-iteration must lie between "
            "the settle start and --phase-schedule-horizon"
        )
    topology_end = next(
        end for name, end in profile["phases"] if name == "topology"
    )
    bootstrap_end = next(
        end
        for name, end in profile["phases"]
        if name == "canonical_bootstrap"
    )
    if args.bootstrap_evidence_release_fraction is None:
        args.bootstrap_evidence_release_fraction = float(bootstrap_end)
    if not (
        0.0
        < args.bootstrap_evidence_release_fraction
        <= float(topology_end)
    ):
        parser.error(
            "--bootstrap-evidence-release-fraction must be after step zero "
            "and no later than the end of topology"
        )
    if args.densify_until_iter is None:
        args.densify_until_iter = int(
            args.phase_schedule_horizon * topology_end
        )
    elif (
        args.densify_until_iter <= 0
        or args.densify_until_iter > args.phase_schedule_horizon
    ):
        parser.error(
            "--densify-until-iter must lie in [1, "
            "--phase-schedule-horizon]. The topology stream is independent "
            "of both the named loss phase and an early diagnostic stop."
        )
    optimization_args = optimization.extract(args)
    # GaussianModel historically used the requested stop as the implicit end
    # of non-position LR decay.  A short exact-prefix run must instead retain
    # the final method horizon, just like every other schedule in this trainer.
    optimization_args.iterations = int(args.phase_schedule_horizon)
    return args, model.extract(args), optimization_args, pipeline.extract(args)


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _surface_capture_to_device(capture, device: torch.device):
    """Move a CPU-mapped 2DGS capture to its training device.

    Upstream ``GaussianModel.restore`` assumes ``torch.load`` retained CUDA
    tensors.  Unified checkpoints are intentionally loaded on CPU to bound
    restart memory, so make that implicit contract explicit before restore.
    Optimizer state is left in the state dict; ``Optimizer.load_state_dict``
    casts it to the parameter device.
    """
    values = list(capture)
    for index in range(1, 10):
        if torch.is_tensor(values[index]):
            source = values[index]
            moved = source.detach().to(device=device)
            values[index] = (
                torch.nn.Parameter(
                    moved, requires_grad=source.requires_grad
                )
                if isinstance(source, torch.nn.Parameter)
                else moved
            )
    if len(values) >= 13 and values[12] is not None:
        values[12] = {
            name: value.to(device=device)
            if torch.is_tensor(value)
            else value
            for name, value in values[12].items()
        }
    return tuple(values)


def _save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checkpoint_due(
    iteration: int,
    *,
    final_iteration: int,
    checkpoint_every: int,
    early_checkpoint_every: int = 0,
    early_checkpoint_until: int = 0,
    retained_iterations: set[int] | frozenset[int] = frozenset(),
    graceful_stop_requested: bool = False,
) -> bool:
    """Return whether the exact live state must be atomically persisted.

    The early cadence is deliberately absent from the optimization contract:
    serializing state cannot change camera order, gradients, or topology. It
    only bounds recomputation before the first multi-GiB quality milestone.
    """

    iteration = int(iteration)
    return bool(
        graceful_stop_requested
        or iteration == int(final_iteration)
        or iteration in retained_iterations
        or (
            int(checkpoint_every) > 0
            and iteration % int(checkpoint_every) == 0
        )
        or (
            int(early_checkpoint_every) > 0
            and iteration <= int(early_checkpoint_until)
            and iteration % int(early_checkpoint_every) == 0
        )
    )


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@torch.no_grad()
def _deployment_surface_geometry(surface, chart_surface) -> dict:
    """Materialize live Chart geometry without changing resume parameters.

    ``surface.capture()`` must retain raw optimizer parameters for exact
    resume, while a checkpoint evaluator must see the live inverse-depth
    atlas geometry rendered during training.  Older intermediate checkpoints
    exposed only the former, so their read-only render silently used stale
    xyz/scale/rotation.  This compact snapshot duplicates geometry only (not
    SH features or Adam state) and therefore closes the evaluation contract
    without changing resume semantics.
    """
    xyz = surface.get_xyz.detach().clone()
    scaling = surface._scaling.detach().clone()
    rotation = surface._rotation.detach().clone()
    baked_rows = 0
    atlas = getattr(chart_surface, "atlas", None)
    if atlas is not None:
        primitive, _, geometry = atlas._bound_geometry(surface)
        if geometry is not None:
            centre, tangent, quaternion = geometry
            xyz[primitive] = centre
            scaling[primitive] = tangent.clamp_min(1e-6).log()
            rotation[primitive] = quaternion
            baked_rows = int(len(primitive))
    return {
        "version": "native-2dgs-deployment-geometry-v1",
        "xyz": xyz.cpu(),
        "scaling": scaling.cpu(),
        "rotation": rotation.cpu(),
        "point_count": int(len(xyz)),
        "chart_baked_rows": baked_rows,
        "resume_surface_unchanged": True,
    }


def _retain_checkpoint_snapshot(path: Path, iteration: int) -> Path:
    """Keep the current atomic checkpoint inode under a stable milestone name."""
    snapshot = path.with_name(
        f"{path.stem}_iteration_{int(iteration):06d}{path.suffix}"
    )
    if snapshot.exists():
        raise FileExistsError(
            "Refusing to overwrite retained checkpoint snapshot: "
            f"{snapshot}"
        )
    # The rolling checkpoint is atomically replaced at the next save. A hard
    # link therefore preserves this exact inode without serializing another
    # ~GB payload, while remaining a normal standalone checkpoint.
    os.link(path, snapshot)
    return snapshot


def _activation_iteration(fraction: float, schedule_horizon: int) -> int:
    """Resolve a strict ``iteration / horizon > fraction`` activation."""
    horizon = int(schedule_horizon)
    if horizon <= 0:
        raise ValueError("schedule_horizon must be positive")
    return int(np.floor(horizon * float(fraction))) + 1


def _resolved_phase_schedule(
    training_profile: str, schedule_horizon: int
) -> tuple[tuple[str, int], ...]:
    """Return the immutable absolute phase endpoints for one method."""
    horizon = int(schedule_horizon)
    if horizon <= 0:
        raise ValueError("schedule_horizon must be positive")
    resolved = []
    previous = 0
    for name, fraction in TRAINING_PROFILES[training_profile]["phases"]:
        endpoint = min(
            horizon,
            max(previous + 1, int(np.floor(horizon * float(fraction)))),
        )
        resolved.append((str(name), endpoint))
        previous = endpoint
    resolved[-1] = (resolved[-1][0], horizon)
    return tuple(resolved)


def _phase(
    step: int,
    schedule_horizon: int,
    training_profile: str = "quality",
) -> str:
    iteration = int(step) + 1
    phases = _resolved_phase_schedule(
        training_profile, schedule_horizon
    )
    for name, endpoint in phases:
        if iteration <= endpoint:
            return name
    return phases[-1][0]


def _public_phase_name(
    phase: str,
    reconstruction_target: str,
    training_profile: str | None = None,
) -> str:
    """Expose static method stages without legacy dynamic terminology.

    Checkpoint schedules retain their historical internal identifiers so an
    exact optimizer resume stays well defined.  For the production static
    target those identifiers are only schedule positions: no conditioned
    primitive, temporal code or dynamic render exists.  Public logs therefore
    report the actual static responsibility being optimized.
    """
    if str(reconstruction_target) != "static":
        return str(phase)
    if str(training_profile) == "hybrid_rigid_stage1":
        return {
            "canonical_bootstrap": "rigid_surface_bootstrap",
            "topology": "rigid_surface_topology",
            "canonical_polish": "rigid_surface_polish",
        }.get(str(phase), str(phase))
    return {
        "canonical_bootstrap": "rigid_static_bootstrap",
        "topology": "persistent_envelope_topology",
        "static_foliage": "static_detail_birth",
        "dynamic_appearance": "static_detail_refinement",
        "ownership_cleanup": "static_joint_optical_cleanup",
        "canonical_polish": "static_appearance_polish",
    }.get(str(phase), str(phase))


def _foliage_enabled(
    step: int, schedule_horizon: int, training_profile: str
) -> bool:
    start = float(
        TRAINING_PROFILES[training_profile].get("foliage_start", 0.0)
    )
    return (int(step) + 1) >= _activation_iteration(
        start, schedule_horizon
    )


def _dynamic_enabled(
    step: int, schedule_horizon: int, training_profile: str
) -> bool:
    start = float(TRAINING_PROFILES[training_profile]["dynamic_start"])
    return (int(step) + 1) >= _activation_iteration(
        start, schedule_horizon
    )


def _conditioned_branch_active(
    step: int,
    schedule_horizon: int,
    training_profile: str,
    reconstruction_target: str = "sequence_conditioned_legacy",
) -> bool:
    """Keep conditioned leaves trainable while final topology settles.

    ``canonical_polish`` is a topology-stable convergence phase, not a
    request to freeze the conditioned image-formation branch.  Freezing both
    at once left dynamic children created by the final replace/split event
    with only a few dozen owner-view updates while canonical volume continued
    to compensate for them during the remaining polish window.
    """
    return (
        reconstruction_target == "sequence_conditioned_legacy"
        and _dynamic_enabled(step, schedule_horizon, training_profile)
    )


def _surface_topology_active(step: int, args) -> bool:
    """Return the explicit surface-topology schedule, independent of phase."""
    if getattr(
        args, "mature_handoff_surface_policy", "joint"
    ) not in {"joint", "atlas_residual"}:
        return False
    iteration = int(step) + 1
    return (
        iteration >= max(int(args.densify_from_iter), 1)
        and iteration <= int(args.densify_until_iter)
    )


def _chart_topology_active(step: int, args) -> bool:
    """Apply the mature handoff policy to every owner of surface geometry.

    Chart rows are rendered from the external inverse-depth atlas, so freezing
    only ``surface._xyz`` while continuing to update/split the atlas does not
    preserve the rigid handoff.  In particular, the mixed-stage ownership
    loss can otherwise move facade geometry even though the runtime audit
    reports ``geometry_trainable=False``.  Atlas optimization and UV topology
    therefore remain active only for the explicit ``joint`` policy.
    """
    iteration = int(step) + 1
    return (
        getattr(
            args, "mature_handoff_surface_policy", "joint"
        )
        == "joint"
        and
        iteration >= max(int(args.densify_from_iter), 1)
        and iteration <= int(args.densify_until_iter)
        and float(args.chart_quadtree_growth_fraction) > 0
    )


def _volume_topology_active(step: int, args, phase: str) -> bool:
    """Keep the final polish phase topology-stable.

    A split changes both support and optimizer state.  Creating children in
    the last few updates leaves no time for their opacity, covariance and SH
    coefficients to settle, which turns high-frequency canopy evidence into
    displaced speckle.  ``volume_densify_until_iteration`` remains the outer
    absolute horizon, while the named polish phase is an explicit
    convergence stage rather than another topology stage.
    """
    return (
        (int(step) + 1) <= int(args.volume_densify_until_iteration)
        and str(phase) != "canonical_polish"
    )


def _volume_topology_ramp_scale(step: int, args) -> float:
    """Return a continuous optical-mass-before-bandwidth schedule.

    The dynamic branch already has a physically defined activation time.
    Starting the full split budget on that same update duplicates thousands
    of low-opacity rows before their owner-ray likelihood has taken even one
    optimizer step. A smoothstep ramp avoids a binary gate and keeps every
    short diagnostic run an exact prefix of the final fixed-horizon method.
    """
    ramp = int(args.volume_topology_ramp_iterations)
    if ramp <= 0:
        return 1.0
    # A static production profile deliberately sets dynamic_start > 1 to
    # declare that no sequence-conditioned branch exists.  Reusing that
    # sentinel as the topology activation kept the ramp at exactly zero for
    # the whole run, silently disabling every foliage split.  Static topology
    # starts with its persistent foliage representation; only the legacy 4D
    # path is tied to conditioned-branch activation.
    profile = TRAINING_PROFILES[args.training_profile]
    if getattr(args, "reconstruction_target", None) == "static":
        activation_fraction = float(profile["foliage_start"])
    else:
        activation_fraction = float(profile["dynamic_start"])
    activation = _activation_iteration(
        activation_fraction, args.phase_schedule_horizon
    )
    age = max((int(step) + 1) - activation + 1, 0)
    linear = float(np.clip(age / ramp, 0.0, 1.0))
    return linear * linear * (3.0 - 2.0 * linear)


def _dynamic_visibility_gate(
    foliage,
    view,
    camera_sequence_lookup: torch.Tensor,
    camera_frame_lookup: torch.Tensor,
) -> torch.Tensor:
    """Compatibility wrapper around the shared train/inference contract."""
    return dynamic_visibility_gate(
        foliage,
        int(view.colmap_id),
        camera_sequence_lookup,
        camera_frame_lookup,
    )


def _static_detail_canonical_ownership_gate(
    foliage,
    camera_id: int,
    camera_sequence_lookup: torch.Tensor | None,
    *,
    support_chunk_size: int = 65_536,
) -> torch.Tensor:
    """Route static-detail learning without changing forward visibility.

    A single static snapshot must remain renderable from every camera, but a
    leaf cluster cannot be optimized toward incompatible leaf positions from
    later traversals. Its persisted calibrated support cameras own RGB,
    positive ray-hit and topology gradients. Globally valid negative
    evidence (confirmed free space, rigid spill and surface-better
    counterfactuals) deliberately bypasses this gate in separate loss paths.
    Envelope and skeleton rows remain globally trainable; cross-sequence
    consensus births naturally retain all of their supporting sequences.
    """
    gate = torch.ones(
        len(foliage), device=foliage.xyz.device, dtype=foliage.xyz.dtype
    )
    detail = foliage.static_leaf_mask
    if not bool(detail.any()):
        return gate
    if foliage.support_camera_ids.numel() == 0 or camera_id < 0:
        gate[detail] = 0
        return gate
    if camera_sequence_lookup is not None and (
        camera_id >= len(camera_sequence_lookup)
        or int(camera_sequence_lookup[camera_id]) < 0
    ):
        gate[detail] = 0
        return gate
    if int(support_chunk_size) <= 0:
        raise ValueError("support_chunk_size must be positive")
    support = foliage.support_camera_ids
    owned = torch.zeros(len(foliage), dtype=torch.bool, device=gate.device)
    for start in range(0, len(foliage), int(support_chunk_size)):
        stop = min(start + int(support_chunk_size), len(foliage))
        support_chunk = support[start:stop]
        owned[start:stop] = (support_chunk == int(camera_id)).any(dim=1)
    gate[detail] = owned[detail].to(gate.dtype)
    return gate


def _static_detail_support_sequence_audit_gate(
    foliage,
    camera_id: int,
    camera_sequence_lookup: torch.Tensor | None,
    *,
    support_chunk_size: int = 65_536,
) -> torch.Tensor:
    """Reconstruct the former broad sequence permission for trace audit only."""
    gate = torch.zeros(
        len(foliage), device=foliage.xyz.device, dtype=torch.bool
    )
    if (
        camera_sequence_lookup is None
        or foliage.support_camera_ids.numel() == 0
        or camera_id < 0
        or camera_id >= len(camera_sequence_lookup)
        or int(camera_sequence_lookup[camera_id]) < 0
    ):
        return gate
    if int(support_chunk_size) <= 0:
        raise ValueError("support_chunk_size must be positive")
    support = foliage.support_camera_ids
    current_sequence = camera_sequence_lookup[camera_id]
    for start in range(0, len(foliage), int(support_chunk_size)):
        stop = min(start + int(support_chunk_size), len(foliage))
        chunk = support[start:stop]
        valid = (chunk >= 0) & (chunk < len(camera_sequence_lookup))
        safe = chunk.clamp(0, len(camera_sequence_lookup) - 1).long()
        gate[start:stop] = (
            valid & (camera_sequence_lookup[safe] == current_sequence)
        ).any(dim=1)
    return gate & foliage.static_leaf_mask


def _static_detail_positive_evidence_sequence_gate(
    foliage,
    camera_id: int,
    camera_sequence_lookup: torch.Tensor | None,
    *,
    support_chunk_size: int = 65_536,
) -> torch.Tensor:
    """Return sequences with persisted positive evidence for each detail.

    ``support_camera_ids`` identifies the appearance seed/owner snapshot,
    while ``verified_camera_ids`` records later real-ray positive witnesses.
    Both are positive evidence.  A camera in neither set's acquisition
    sequence cannot authoritatively create *or delete* that leaf: seasonal
    foliage absence is not calibrated free space for a different traversal.
    Forward visibility remains unconditional; this gate only routes future
    training gradients.
    """
    gate = torch.zeros(
        len(foliage), device=foliage.xyz.device, dtype=torch.bool
    )
    if (
        camera_sequence_lookup is None
        or camera_id < 0
        or camera_id >= len(camera_sequence_lookup)
        or int(camera_sequence_lookup[camera_id]) < 0
    ):
        return gate
    if int(support_chunk_size) <= 0:
        raise ValueError("support_chunk_size must be positive")
    tables = []
    for name in ("support_camera_ids", "verified_camera_ids"):
        table = getattr(foliage, name, None)
        if torch.is_tensor(table) and table.ndim == 2 and table.numel():
            tables.append(table)
    if not tables:
        return gate
    current_sequence = camera_sequence_lookup[camera_id]
    for start in range(0, len(foliage), int(support_chunk_size)):
        stop = min(start + int(support_chunk_size), len(foliage))
        matched = gate[start:stop]
        for table in tables:
            chunk = table[start:stop]
            valid = (chunk >= 0) & (chunk < len(camera_sequence_lookup))
            safe = chunk.clamp(0, len(camera_sequence_lookup) - 1).long()
            matched |= (
                valid & (camera_sequence_lookup[safe] == current_sequence)
            ).any(dim=1)
        gate[start:stop] = matched
    return gate & foliage.static_leaf_mask


def _static_detail_same_sequence_appearance_gate(
    foliage,
    camera_id: int,
    camera_sequence_lookup: torch.Tensor | None,
    exact_ownership_gate: torch.Tensor,
    *,
    fallback_weight: float,
    positive_evidence_sequence_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Give stable neighbouring cameras soft colour ownership only.

    The support table stores a small set of calibrated seed/owner cameras,
    not the complete set of views in which a leaf cluster can provide valid
    colour evidence.  Requiring an exact camera-id match therefore left most
    detail rows with no SH update at all.  Cameras from the same acquisition
    sequence observe the same physical foliage configuration and may refine
    its appearance, but they must not move it, create optical mass or request
    topology.  Cross-sequence residuals remain in the authoritative image and
    are robustified by the spatial uncertainty field; they receive no direct
    detail appearance permission here.
    """
    weight = float(fallback_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("same-sequence appearance weight must be in [0, 1]")
    exact = torch.as_tensor(
        exact_ownership_gate,
        device=foliage.xyz.device,
        dtype=foliage.xyz.dtype,
    ).reshape(-1)
    if len(exact) != len(foliage):
        raise ValueError("exact ownership gate must align with foliage")
    appearance = exact.clone()
    if weight <= 0.0 or not bool(foliage.static_leaf_mask.any()):
        return appearance
    same_sequence = (
        _static_detail_positive_evidence_sequence_gate(
            foliage,
            int(camera_id),
            camera_sequence_lookup,
        )
        if positive_evidence_sequence_gate is None
        else torch.as_tensor(
            positive_evidence_sequence_gate,
            device=foliage.xyz.device,
            dtype=torch.bool,
        ).reshape(-1)
    )
    if len(same_sequence) != len(foliage):
        raise ValueError("positive evidence sequence gate must align with foliage")
    fallback = (
        same_sequence
        & foliage.static_leaf_mask
        & (exact <= 0)
    )
    appearance[fallback] = weight
    return appearance


def _static_detail_same_sequence_optical_gate(
    foliage,
    camera_id: int,
    camera_sequence_lookup: torch.Tensor | None,
    exact_ownership_gate: torch.Tensor,
    *,
    fallback_weight: float,
    positive_evidence_sequence_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Route positive optical coverage symmetrically with leaf evidence.

    Exact support cameras retain unit authority. Other cameras from a
    persisted positive-evidence sequence receive a continuous opacity weight,
    enough to restore multi-view crown coverage without granting them
    geometry or topology ownership. Cameras from unrelated traversals receive
    neither positive optical gradients nor negative cleanup authority.
    """
    weight = float(fallback_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("same-sequence optical weight must be in [0, 1]")
    exact = torch.as_tensor(
        exact_ownership_gate,
        device=foliage.xyz.device,
        dtype=foliage.xyz.dtype,
    ).reshape(-1)
    if len(exact) != len(foliage):
        raise ValueError("exact ownership gate must align with foliage")
    optical = exact.clone()
    if weight <= 0.0 or not bool(foliage.static_leaf_mask.any()):
        return optical
    same_sequence = (
        _static_detail_positive_evidence_sequence_gate(
            foliage,
            int(camera_id),
            camera_sequence_lookup,
        )
        if positive_evidence_sequence_gate is None
        else torch.as_tensor(
            positive_evidence_sequence_gate,
            device=foliage.xyz.device,
            dtype=torch.bool,
        ).reshape(-1)
    )
    if len(same_sequence) != len(foliage):
        raise ValueError("positive evidence sequence gate must align with foliage")
    fallback = same_sequence & foliage.static_leaf_mask & (exact <= 0)
    optical[fallback] = weight
    return optical


def _static_detail_consensus_hit_gate(foliage) -> torch.Tensor:
    """Permit real hit rays to reuse already verified static consensus.

    Exact support-camera ownership is still the correct permission for RGB
    geometry and screen-space topology: a moving leaf must not be dragged to
    every traversal.  It is too narrow for the analytic ray factor, however.
    A detail primitive that has already been verified by at least two
    independent sequences is an established part of the one static map, so a
    later calibrated hit ray may refine its depth and optical mass without
    creating a new primitive or altering its support metadata.

    Keeping this permission separate also prevents an ownership miss from
    being misclassified as a geometric hole by ``StaticRayBirthAccumulator``.
    New and split children remain support-owned until real render witnesses
    promote them to ``VERIFICATION_VERIFIED``.
    """

    detail = foliage.static_leaf_mask
    verification = getattr(foliage, "verification_state", None)
    verified_cameras = getattr(foliage, "verified_camera_count", None)
    verified_sequences = getattr(foliage, "verified_sequence_count", None)
    if (
        verification is None
        or verified_cameras is None
        or verified_sequences is None
    ):
        return torch.zeros_like(detail)
    return (
        detail
        & (verification == VERIFICATION_VERIFIED)
        & (verified_cameras >= 2)
        & (verified_sequences >= 2)
    )


def _static_detail_refinement_gate(
    foliage,
    ownership_gate: torch.Tensor | None,
) -> torch.Tensor:
    """Reserve high-bandwidth refinement for persistent static detail.

    Unverified or single-sequence rows still receive DC colour, optical-mass
    and ray/depth-posterior gradients in the authoritative mixed render.  They
    must not, however, explain arbitrary image edges with geometry motion or
    higher-order appearance before independent real-camera evidence has made
    them part of the persistent static map.  Envelope and skeleton rows are
    left unchanged because this gate is also used by the volume-only stream.
    """

    gate = (
        torch.ones_like(foliage.opacities)
        if ownership_gate is None
        else torch.as_tensor(
            ownership_gate,
            device=foliage.xyz.device,
            dtype=foliage.xyz.dtype,
        ).reshape(-1).clone()
    )
    if len(gate) != len(foliage):
        raise ValueError("static detail refinement gate must align with foliage")
    detail = foliage.static_leaf_mask
    if not bool(detail.any()):
        return gate
    verification = getattr(foliage, "verification_state", None)
    verified_cameras = getattr(foliage, "verified_camera_count", None)
    verified_sequences = getattr(foliage, "verified_sequence_count", None)
    if verification is None or verified_cameras is None:
        gate[detail] = 0
        return gate
    # The static target deliberately chooses one coherent acquisition per
    # tree.  Independent cameras in that snapshot are valid multiview proof;
    # a second traversal is useful evidence but not a hard requirement that
    # would erase vegetation which changed between traversals.
    persistent = (
        (verification == VERIFICATION_VERIFIED)
        & (verified_cameras >= 2)
    )
    if verified_sequences is not None:
        persistent &= verified_sequences >= 1
    gate[detail] *= persistent[detail].to(gate.dtype)
    return gate


def _conditioned_base_gradient_gate(
    foliage, visibility_gate: torch.Tensor
) -> torch.Tensor:
    """Route conditioned base-state gradients to exact dynamic owners only.

    Canonical crown and static skeleton remain in the conditioned forward
    render for correct occlusion and depth sorting.  They already receive an
    unbiased update from the canonical full-epoch RGB stream, however, and
    must not receive a second, evidence-camera-biased update from the
    sequence-conditioned objective.  Allowing that update wrote transient
    leaf residuals into the localization map and also contaminated the
    canonical topology statistics accumulated from the conditioned render.
    """
    visibility_gate = torch.as_tensor(
        visibility_gate,
        device=foliage.dynamic_leaf_mask.device,
        dtype=foliage.xyz.dtype,
    ).reshape(-1)
    if visibility_gate.shape != foliage.dynamic_leaf_mask.shape:
        raise ValueError(
            "conditioned visibility gate must have one value per volume "
            "Gaussian"
        )
    return (
        foliage.dynamic_leaf_mask
        & (visibility_gate >= 0.999)
    ).to(visibility_gate.dtype)


def _dynamic_opacity_floor(foliage) -> torch.Tensor:
    """Return a spatial evidence posterior, not one global leaf threshold."""
    _, evidence_floor = evidence_conditioned_leaf_optical_mass(
        foliage.occupancy_probability,
        foliage.support_view_count,
        foliage.unknown_view_count,
        foliage.ray_depth_nll,
        foliage.free_space_violation_count,
    )
    dense_dynamic = (
        foliage.dynamic_leaf_mask
        & (foliage.initialization_source == 4)
    )
    legacy_floor = torch.full_like(evidence_floor, 0.015)
    return torch.where(dense_dynamic, evidence_floor, legacy_floor)


def _dynamic_opacity_ceiling(
    foliage, maximum_opacity: float
) -> torch.Tensor:
    """Return continuous spatial alpha authority for uncertain depth births."""
    evidence_ceiling = evidence_conditioned_dynamic_opacity_ceiling(
        foliage.occupancy_probability,
        foliage.support_view_count,
        foliage.unknown_view_count,
        foliage.ray_depth_nll,
        foliage.free_space_violation_count,
        foliage.replacement_group,
        maximum_opacity=maximum_opacity,
    )
    uncertain_depth_birth = foliage.dynamic_leaf_mask & (
        (foliage.initialization_source == 3)
        | (foliage.initialization_source == 4)
    )
    global_ceiling = torch.full_like(
        evidence_ceiling, float(maximum_opacity)
    )
    return torch.where(
        uncertain_depth_birth, evidence_ceiling, global_ceiling
    )


def _volume_topology_authority(foliage) -> torch.Tensor:
    """Return continuous geometry authority for adaptive volume bandwidth.

    Visibility and a non-zero image gradient establish that a primitive needs
    more screen-space bandwidth; they do not establish that its uncertain 3D
    location deserves the same number of children as a multi-view lineage.
    Keep every hypothesis eligible, but weight its topology demand by the
    stored depth posterior. An ungrouped dynamic birth has no local
    replace-and-retire mass denominator, so it additionally pays a continuous
    occupancy factor. Positive floors deliberately avoid turning this
    posterior into another accept/reject gate.
    """
    depth_authority = torch.rsqrt(
        1.0 + foliage.ray_depth_nll.clamp_min(0)
    ).clamp(0.05, 1.0)
    occupancy_authority = (
        0.05
        + 0.95 * foliage.occupancy_probability.clamp(0.0, 1.0)
    )
    local_authority = torch.where(
        foliage.dynamic_leaf_mask & (foliage.replacement_group < 0),
        occupancy_authority,
        torch.ones_like(depth_authority),
    )
    return (depth_authority * local_authority).clamp(0.0025, 1.0)


def _dense_exact_ray_bandwidth_mask(foliage) -> torch.Tensor:
    """Identify sequence-local renderer bases with one calibrated owner.

    These rows are not independent metric geometry.  They are samples of an
    already-established optical ray posterior and may therefore gain image
    bandwidth without claiming stronger depth evidence.
    """
    observation_count = (foliage.observation_camera_ids >= 0).sum(dim=1)
    return (
        foliage.dynamic_leaf_mask
        & (foliage.initialization_source == 4)
        & (observation_count == 1)
    )


def _volume_split_authority(foliage) -> torch.Tensor:
    """Separate 3D geometry authority from exact-ray optical subdivision.

    A weak depth posterior must limit creation of new 3D occupancy.  It must
    not suppress subdivision of a broad exact-owner EWA footprint in the
    calibrated image plane: that mutation preserves depth and covariance and
    only increases deployable RGB bandwidth.  Visibility, non-zero owner
    gradient and projected radius remain mandatory eligibility evidence in
    ``_adapt_volume``.
    """
    geometry_authority = _volume_topology_authority(foliage)
    exact_ray_bandwidth = _dense_exact_ray_bandwidth_mask(foliage)
    return torch.where(
        exact_ray_bandwidth,
        torch.ones_like(geometry_authority),
        geometry_authority,
    )


@torch.no_grad()
def _enforce_retirement_only_fallback_opacity_gradient(
    foliage,
    visibility_gate: torch.Tensor,
    exact_gradient_gate: torch.Tensor,
) -> dict[str, float | int]:
    """Make non-owner opacity evidence one-sided.

    A neighbouring frame is negative evidence for a sequence-local leaf only
    when the leaf obstructs that frame.  It is not positive evidence that may
    grow the leaf: only the calibrated owner observation can establish
    occupancy.  Adam performs ``parameter -= lr * gradient``, so a negative
    opacity-logit gradient would increase fallback opacity and is suppressed;
    a positive gradient remains able to retire the occluder.

    Temporal opacity bases are observation-conditioned appearance parameters,
    not free-space state, and therefore remain exact-owner-only.
    """
    visibility_gate = torch.as_tensor(
        visibility_gate,
        device=foliage.xyz.device,
        dtype=foliage.xyz.dtype,
    ).reshape(-1)
    exact_gradient_gate = torch.as_tensor(
        exact_gradient_gate,
        device=foliage.xyz.device,
        dtype=foliage.xyz.dtype,
    ).reshape(-1)
    if (
        len(visibility_gate) != len(foliage)
        or len(exact_gradient_gate) != len(foliage)
    ):
        raise ValueError("fallback gradient gates must match foliage rows")
    fallback = (
        foliage.dynamic_leaf_mask
        & (visibility_gate > 0)
        & (exact_gradient_gate < 0.999)
    )
    audit: dict[str, float | int] = {
        "fallback_opacity_rows": int(fallback.sum()),
        "fallback_opacity_increase_rows_suppressed": 0,
        "fallback_opacity_increase_gradient_suppressed": 0.0,
        "fallback_temporal_opacity_rows_suppressed": 0,
    }
    opacity_gradient = foliage.opacity_logits.grad
    if opacity_gradient is not None and bool(fallback.any()):
        row_gradient = opacity_gradient[:, 0]
        increasing = fallback & (row_gradient < 0)
        audit["fallback_opacity_increase_rows_suppressed"] = int(
            increasing.sum()
        )
        audit["fallback_opacity_increase_gradient_suppressed"] = float(
            (-row_gradient[increasing]).sum()
        )
        row_gradient[increasing] = 0
    temporal_gradient = foliage.dynamic_opacity_basis.grad
    if temporal_gradient is not None and bool(fallback.any()):
        active = fallback & (
            temporal_gradient.flatten(1).abs().sum(dim=1) > 0
        )
        audit["fallback_temporal_opacity_rows_suppressed"] = int(
            active.sum()
        )
        temporal_gradient[fallback] = 0
    return audit


def _dynamic_observation_factor(
    foliage,
    view,
    temporal_code: torch.Tensor,
    *,
    maximum_observations: int = 2048,
    sample_update: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Apply the exact dynamic observation as geometry *and* ray likelihood.

    Dense observation-space rays are deliberately excluded from the
    cross-sequence canonical posterior: a single traversal must not promote
    transient foliage into the localization map.  The former implementation
    stopped there, however.  It constrained a dynamic centre to the measured
    pixel/depth but never required the owner-camera descendants to contribute
    optical mass on that ray.  RGB could consequently fit a semi-transparent
    colour blend and leave the rigid facade visible through the whole crown.

    The centre term below remains opacity-independent, so transparency cannot
    evade geometry.  In addition, descendants sharing one immutable
    observation lineage now form an analytic anisotropic hit likelihood:
    their projected radial density and longitudinal Gaussian mass are
    accumulated inside the measured depth posterior, while mass before the
    interval is penalized.  This is a per-calibrated-ray constraint, not a
    binary tree-mask alpha target, and it remains valid after adaptive split.
    """
    zero = foliage.xyz.new_zeros(())
    empty = torch.zeros(
        len(foliage), dtype=torch.bool, device=foliage.xyz.device
    )
    if foliage.observation_camera_ids.numel() == 0:
        return zero, empty, {"matched": 0}
    matches = (
        foliage.observation_camera_ids == int(view.colmap_id)
    ) & foliage.dynamic_leaf_mask[:, None]
    primitive_indices, slots = torch.nonzero(matches, as_tuple=True)
    if not len(primitive_indices):
        return zero, empty, {"matched": 0}
    # Sampling must happen in immutable observation-lineage space, not in
    # renderer-row space. Adaptive split duplicates observation metadata on
    # every descendant. The v30 implementation truncated descendants to
    # 2,048 rows *before* grouping, so a split lineage was frequently scored
    # using only one child even though the loss claimed to aggregate its
    # complete optical mass. That made the ray factor depend on topology row
    # order and encouraged repeated split without ever forming a valid hit.
    lineage_identity = foliage.track_id[primitive_indices].to(torch.int64)
    missing_lineage = lineage_identity == -1
    lineage_identity = torch.where(
        missing_lineage,
        primitive_indices.to(torch.int64),
        lineage_identity,
    )
    observation_key = torch.stack(
        [
            (~missing_lineage).to(torch.int64),
            lineage_identity,
            slots.to(torch.int64),
        ],
        dim=1,
    )
    unique_observation_key, key_inverse = torch.unique(
        observation_key,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    available_observations = int(len(unique_observation_key))
    if available_observations > int(maximum_observations):
        # Rotate through observation lineages, then retain every descendant
        # of each chosen observation. This preserves both evidence-epoch
        # coverage and post-split optical-mass equivalence.
        start = (
            int(sample_update) * int(maximum_observations)
        ) % available_observations
        selected_keys = (
            torch.arange(
                int(maximum_observations),
                device=primitive_indices.device,
            )
            + start
        ) % available_observations
        selected_keys = selected_keys.to(
            device=primitive_indices.device,
            dtype=torch.long,
        )
        selected_rows = torch.isin(key_inverse, selected_keys)
        primitive_indices = primitive_indices[selected_rows]
        slots = slots[selected_rows]
        observation_key = observation_key[selected_rows]
    target_uv = foliage.observation_uv[primitive_indices, slots]
    target_depth = foliage.observation_depth[primitive_indices, slots]
    xyz, _, conditioned_opacity = foliage.conditioned_state(
        temporal_code, include_dynamic=True
    )
    homogeneous = torch.cat(
        [xyz[primitive_indices], torch.ones_like(xyz[primitive_indices, :1])],
        dim=1,
    )
    camera_xyz = homogeneous @ view.world_view_transform
    depth = camera_xyz[:, 2]
    predicted_uv = torch.stack(
        [
            (
                float(view.focal_x) * camera_xyz[:, 0]
                / depth.clamp_min(1e-5)
                + float(view.cx)
            )
            / float(view.image_width),
            (
                float(view.focal_y) * camera_xyz[:, 1]
                / depth.clamp_min(1e-5)
                + float(view.cy)
            )
            / float(view.image_height),
        ],
        dim=1,
    )
    valid = (
        torch.isfinite(target_uv).all(dim=1)
        & torch.isfinite(target_depth)
        & (target_depth > 0.05)
        & torch.isfinite(depth)
        & (depth > 0.05)
    )
    if not bool(valid.any()):
        return zero, empty, {"matched": 0}
    primitive_indices = primitive_indices[valid]
    observation_key = observation_key[valid]
    pixel_scale = target_uv.new_tensor(
        [float(view.image_width), float(view.image_height)]
    )
    pixel_error = (
        (predicted_uv[valid] - target_uv[valid]) * pixel_scale
    ).norm(dim=1)
    observation_depth_sigma = foliage.position_covariance[
        primitive_indices
    ].diagonal(dim1=-2, dim2=-1).sum(dim=1).sqrt().clamp(0.03, 3.00)
    depth_error = (
        depth[valid] - target_depth[valid]
    ) / observation_depth_sigma
    individual_ray = torch.log1p((pixel_error / 2.0).square())
    individual_depth = torch.log1p(depth_error.square())
    individual = individual_ray + 0.25 * individual_depth
    # A split lineage must explain one physical observation at least once;
    # forcing every descendant back to the same ray collapses adaptive volume
    # refinement. Normalized soft-min keeps a smooth gradient without making
    # the objective cheaper merely because a lineage has more children.
    _, inverse = torch.unique(
        observation_key,
        dim=0,
        sorted=False,
        return_inverse=True,
    )
    group_count = int(inverse.max()) + 1
    temperature = individual.new_tensor(0.10)
    score = -individual / temperature
    maxima = torch.full(
        (group_count,),
        -torch.inf,
        device=score.device,
        dtype=score.dtype,
    )
    maxima.scatter_reduce_(
        0, inverse, score, reduce="amax", include_self=True
    )
    exponential = torch.zeros_like(maxima)
    exponential.scatter_add_(
        0, inverse, torch.exp(score - maxima[inverse])
    )
    counts = torch.zeros_like(maxima)
    counts.scatter_add_(0, inverse, torch.ones_like(score))
    grouped = -temperature * (
        maxima + torch.log(exponential / counts.clamp_min(1))
    )
    geometry_loss = grouped.mean()

    # Evaluate the same observation in ray space.  The maximum local scale is
    # a conservative footprint for an arbitrarily rotated 3D EWA Gaussian;
    # using it for candidate support avoids a false zero gradient while the
    # exact mixed rasterizer still supplies the photometric gradient.
    primitive_scale = foliage.scales[primitive_indices].amax(
        dim=1
    ).clamp_min(1e-4)
    focal = individual.new_tensor(
        float(np.sqrt(float(view.focal_x) * float(view.focal_y)))
    )
    projected_sigma = (
        primitive_scale
        * focal
        / depth[valid].clamp_min(0.05)
    ).clamp(0.35, 32.0)
    radial_density = torch.exp(
        -0.5 * (pixel_error / projected_sigma).square().clamp_max(80)
    )
    local_alpha = (
        conditioned_opacity[primitive_indices]
        * radial_density
    ).clamp(0.0, 1.0 - 1e-6)
    local_tau = -torch.log1p(-local_alpha)

    sqrt_two = float(np.sqrt(2.0))

    def normal_cdf(value):
        return 0.5 * (1.0 + torch.erf(value / sqrt_two))

    lower = target_depth[valid] - 2.5 * observation_depth_sigma
    upper = target_depth[valid] + 2.5 * observation_depth_sigma
    longitudinal_sigma = primitive_scale
    inside_fraction = (
        normal_cdf((upper - depth[valid]) / longitudinal_sigma)
        - normal_cdf((lower - depth[valid]) / longitudinal_sigma)
    ).clamp(0.0, 1.0)
    before_fraction = normal_cdf(
        (lower - depth[valid]) / longitudinal_sigma
    ).clamp(0.0, 1.0)
    inside_tau = torch.zeros_like(maxima)
    before_tau = torch.zeros_like(maxima)
    inside_tau.scatter_add_(
        0, inverse, local_tau * inside_fraction
    )
    before_tau.scatter_add_(
        0, inverse, local_tau * before_fraction
    )
    hit_alpha = -torch.expm1(-inside_tau)
    # Spatial posterior width supplies a continuous observation reliability;
    # there is no global hard confidence gate.
    observation_reliability = (
        0.15 / (observation_depth_sigma + 0.15)
    ).clamp(0.02, 1.0)
    lineage_reliability = torch.zeros_like(maxima)
    lineage_reliability.scatter_reduce_(
        0,
        inverse,
        observation_reliability,
        reduce="amax",
        include_self=False,
    )
    lineage_reliability = lineage_reliability.clamp(0.02, 1.0)
    hit_loss = (
        -torch.log(hit_alpha + 1e-4) * lineage_reliability
    ).sum() / lineage_reliability.sum().clamp_min(1)
    prehit_free_loss = (
        before_tau * lineage_reliability
    ).sum() / lineage_reliability.sum().clamp_min(1)
    loss = geometry_loss + hit_loss + 0.10 * prehit_free_loss
    group_minimum = torch.full_like(maxima, torch.inf)
    group_minimum.scatter_reduce_(
        0, inverse, individual.detach(), reduce="amin", include_self=True
    )
    winner = individual.detach() <= group_minimum[inverse] + 1e-6
    observed = empty.clone()
    observed[primitive_indices[winner]] = True
    return loss, observed, {
        "matched": group_count,
        "matched_descendants": int(valid.sum()),
        "available": available_observations,
        "lineages": group_count,
        "ray_pixels": float(pixel_error.mean().detach()),
        "depth_sigma": float(depth_error.abs().mean().detach()),
        "geometry": float(geometry_loss.detach()),
        "ray_hit": float(hit_loss.detach()),
        "prehit_free": float(prehit_free_loss.detach()),
        "hit_alpha_mean": float(hit_alpha.mean().detach()),
        "hit_alpha_minimum": float(hit_alpha.min().detach()),
        "ray_contract": (
            "exact_owner_complete_lineage_analytic_hit_and_"
            "prehit_transmittance"
        ),
    }


def _full_epoch_schedule(count: int, iterations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = []
    while len(result) < iterations:
        result.extend(rng.permutation(count).tolist())
    return np.asarray(result[:iterations], dtype=np.int64)


def _cycle_schedule(
    indices: list[int], iterations: int, seed: int
) -> np.ndarray:
    if not indices:
        return np.full(iterations, -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    result = []
    values = np.asarray(indices, dtype=np.int64)
    while len(result) < iterations:
        result.extend(rng.permutation(values).tolist())
    return np.asarray(result[:iterations], dtype=np.int64)


def _periodic_schedule_index(step: int, every: int) -> int:
    """Map a scheduled training step to its contiguous event index.

    Advancing a full-step camera permutation and then executing only every
    ``N``-th entry silently discards the other ``N-1`` entries.  In a short
    phase that can leave a real camera with no appearance update at all.  A
    periodic evidence stream must consume one consecutive schedule entry per
    event so every completed permutation is a genuine camera epoch.
    """
    if every <= 0:
        raise ValueError("period must be positive")
    iteration = int(step) + 1
    if iteration <= 0 or iteration % int(every) != 0:
        raise ValueError("step is not a scheduled periodic event")
    return iteration // int(every) - 1


def _balanced_evidence_fraction(
    total_view_count: int,
    evidence_view_count: int,
    *,
    per_view_oversampling: float = 12.0,
) -> float:
    """Solve a scene-size-aware conditioned sampling fraction.

    The non-forced portion remains a uniform full-scene RGB schedule, so an
    evidence camera also receives its ordinary share there. If ``f`` is the
    forced evidence-step fraction, an effective per-view ratio ``r`` means

        (f / E + (1 - f) / N) / ((1 - f) / N) = r.

    The old fixed f=0.75 visited each of 96 evidence cameras roughly 43x as
    often as a non-evidence camera in StMarys. That produced excellent exact
    keyframes while the complete 1,487-view tree metric regressed.  The later
    4x policy swung too far in the other direction after evidence coverage
    expanded to 259 cameras: by iteration 4k each exact owner had received
    only about four conditioned updates and the dynamic branch collapsed
    while topology was still growing.  A scene-size-aware 12x target gives
    every exact owner repeated updates without returning to the old 43x
    keyframe memorization regime; the remaining schedule is still a uniform
    full-scene stream.
    """
    total = max(int(total_view_count), 0)
    evidence = min(max(int(evidence_view_count), 0), total)
    if total == 0 or evidence == 0:
        return 0.0
    if evidence == total:
        return 1.0
    ratio = max(float(per_view_oversampling), 1.0)
    fraction = (
        (ratio - 1.0) * evidence
        / (total + (ratio - 1.0) * evidence)
    )
    return float(np.clip(fraction, 0.15, 0.65))


def _complete_evidence_epoch_batch_size(
    configured_batch: int,
    effective_rows: int,
    schedule_horizon: int,
    factor_every: int,
    *,
    margin: float = 1.05,
    per_camera_rows: tuple[int, ...] | list[int] | None = None,
    available_factor_calls: int | None = None,
) -> tuple[int, int, int]:
    """Return a prefix-stable ray batch capable of one complete epoch.

    Ray evidence is consumed by independent per-camera samplers.  Therefore
    aggregate capacity ``batch * calls >= rows`` is necessary but not
    sufficient: every camera tail consumes a separate factor call.  When the
    immutable per-camera table sizes are available, solve the exact discrete
    contract ``sum(ceil(camera_rows / batch)) <= calls`` by binary search.
    """
    if (
        int(configured_batch) <= 0
        or int(effective_rows) < 0
        or int(schedule_horizon) <= 0
        or int(factor_every) <= 0
        or float(margin) < 1.0
    ):
        raise ValueError("Invalid foliage ray epoch capacity contract")
    factor_calls = (
        int(available_factor_calls)
        if available_factor_calls is not None
        else int(schedule_horizon) // int(factor_every)
    )
    if factor_calls <= 0:
        raise ValueError("Foliage ray epoch has no available factor calls")
    minimum = int(
        np.ceil(float(margin) * int(effective_rows) / factor_calls)
    )
    if per_camera_rows is None:
        return max(int(configured_batch), minimum), minimum, factor_calls

    camera_rows = tuple(int(count) for count in per_camera_rows)
    if any(count < 0 for count in camera_rows):
        raise ValueError("Per-camera foliage ray counts cannot be negative")
    if sum(camera_rows) != int(effective_rows):
        raise ValueError(
            "Per-camera foliage ray counts do not sum to effective_rows"
        )
    nonempty = tuple(count for count in camera_rows if count > 0)
    if not nonempty:
        return max(int(configured_batch), minimum), 0, factor_calls
    if len(nonempty) > factor_calls:
        raise ValueError(
            "Scheduled foliage ray factor calls cannot visit every camera"
        )

    def required_calls(batch: int) -> int:
        return sum(
            (count + int(batch) - 1) // int(batch)
            for count in nonempty
        )

    lower = max(
        1,
        (int(effective_rows) + factor_calls - 1) // factor_calls,
    )
    upper = max(nonempty)
    while lower < upper:
        candidate = (lower + upper) // 2
        if required_calls(candidate) <= factor_calls:
            upper = candidate
        else:
            lower = candidate + 1
    exact_minimum = lower
    return (
        max(int(configured_batch), minimum, exact_minimum),
        exact_minimum,
        factor_calls,
    )


def _prefix_stable_resume_ray_batch(
    computed_batch: int,
    resume: dict | None,
    per_camera_remaining_rows: tuple[int, ...] | list[int],
    available_factor_calls: int,
) -> int:
    """Keep the full-horizon ray batch fixed across exact-prefix resumes."""

    computed_batch = int(computed_batch)
    if resume is None:
        return computed_batch
    saved = int(
        resume.get("training_contract", {}).get(
            "ray_posterior_maximum_rays", -1
        )
    )
    if saved <= 0:
        raise RuntimeError(
            "Resume checkpoint has no fixed foliage ray batch contract"
        )
    required_calls = sum(
        (int(count) + saved - 1) // saved
        for count in per_camera_remaining_rows
        if int(count) > 0
    )
    if required_calls > int(available_factor_calls):
        raise RuntimeError(
            "Resume checkpoint foliage ray batch cannot complete the "
            "remaining per-camera evidence epoch"
        )
    # Re-solving the same epoch after rows have been consumed may change the
    # mathematical minimum by one or more rays.  That is a runtime optimum,
    # not permission to change the objective's batch/normalization mid-run.
    return saved


def _evidence_biased_schedule(
    rgb_schedule: np.ndarray,
    evidence_indices: list[int],
    *,
    fraction: float,
    seed: int,
    active_mask: np.ndarray | None = None,
    minimum_full_scene_visits: int = 3,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Mix full-scene RGB views with a coverage-preserving evidence cycle.

    Volume statistics are reset at every adaptation event. Under a 1,487-view
    uniform schedule, most exact foliage cameras were never visited inside a
    200-step window, so their observation gradients could not make the split
    candidate set. The conditioned branch gets its own schedule while the
    canonical branch retains the unchanged full-scene epoch.

    Evidence oversampling must not erase the only conditioned RGB visits of a
    non-evidence camera.  That happened when 65% of a full-scene epoch was
    replaced in-place: after 9k steps 63 StMarys database views had never
    trained conditioned appearance, including view 00408.  Reserve up to
    ``minimum_full_scene_visits`` occurrences of every camera inside the
    interval where the conditioned branch is actually active, then distribute
    the requested evidence steps across the remaining positions.  The
    evidence fraction is therefore an upper bound when the schedule is too
    short to satisfy both contracts.
    """
    rgb_schedule = np.asarray(rgb_schedule, dtype=np.int64)
    result = rgb_schedule.copy()
    fraction = float(np.clip(fraction, 0.0, 1.0))
    evidence_indices = sorted(set(map(int, evidence_indices)))
    minimum_full_scene_visits = max(int(minimum_full_scene_visits), 0)
    if active_mask is None:
        active = np.ones(len(result), dtype=bool)
    else:
        active = np.asarray(active_mask, dtype=bool).reshape(-1)
        if len(active) != len(result):
            raise ValueError(
                "Conditioned active mask must match the RGB schedule"
            )
    active_positions = np.flatnonzero(active)
    active_view_ids = np.unique(rgb_schedule)
    if len(active_positions) and len(active_view_ids):
        # A resumed active suffix can cut through the beginning and end of
        # two different RGB epochs. Reusing that clipped subsequence leaves a
        # few cameras with only two visits even when three complete visits fit
        # mathematically. Build a fresh full-scene conditioned epoch only
        # inside the active positions; the canonical RGB schedule itself stays
        # byte-identical and is independently checked on resume.
        result[active_positions] = _cycle_schedule(
            active_view_ids.tolist(),
            len(active_positions),
            seed + 104_729,
        )
    protected = np.zeros(len(result), dtype=bool)
    protected_per_view = min(
        minimum_full_scene_visits,
        (
            len(active_positions) // max(len(active_view_ids), 1)
            if len(active_positions)
            else 0
        ),
    )
    if protected_per_view > 0:
        # Do not protect the first K occurrences of every view.  That packs
        # all guaranteed full-scene visits into the front of the conditioned
        # interval and delays the evidence stream by K complete camera
        # epochs (4.2k steps for StMarys).  Canonical RGB already keeps its
        # independent full-scene epoch; conditioned sampling needs exact
        # owner evidence from the beginning while retaining a hard final
        # coverage guarantee.
        #
        # Spread exactly K*V protected slots over the complete active
        # interval, then fill them with K independently shuffled full-scene
        # cycles.  This guarantees K visits for every camera and makes both
        # protected RGB and evidence steps prefix-stable instead of running
        # them as two consecutive phases.
        protected_steps = protected_per_view * len(active_view_ids)
        cumulative = np.floor(
            np.arange(1, len(active_positions) + 1, dtype=np.float64)
            * protected_steps
            / max(len(active_positions), 1)
        )
        previous = np.concatenate(
            [np.zeros(1, dtype=np.float64), cumulative[:-1]]
        )
        protected_local = cumulative > previous
        protected_positions = active_positions[protected_local]
        if len(protected_positions) != protected_steps:
            raise RuntimeError(
                "Coverage scheduler failed to allocate exact protected slots"
            )
        protected[protected_positions] = True
        result[protected_positions] = _cycle_schedule(
            active_view_ids.tolist(),
            protected_steps,
            seed + 209_759,
        )
    replaceable = active & ~protected
    replaceable_positions = np.flatnonzero(replaceable)
    requested_evidence_steps = int(
        np.floor(len(active_positions) * fraction)
    )
    evidence_steps = min(
        requested_evidence_steps, len(replaceable_positions)
    )
    base_audit = {
        "active_steps": int(len(active_positions)),
        "active_view_count": int(len(active_view_ids)),
        "minimum_full_scene_visits": int(protected_per_view),
        "protected_full_scene_steps": int(protected.sum()),
        "protected_placement": "uniform_interleaved_complete_epochs",
        "replaceable_steps": int(len(replaceable_positions)),
        "requested_evidence_steps": int(requested_evidence_steps),
        "coverage_preserving": True,
    }
    if (
        not evidence_indices
        or fraction <= 0.0
        or not len(result)
        or evidence_steps <= 0
    ):
        return result, {
            "evidence_view_count": len(evidence_indices),
            "evidence_steps": 0,
            "rgb_steps": int(len(result)),
            "requested_fraction": fraction,
            "realized_fraction": 0.0,
            **base_audit,
        }
    # Error-diffusion placement over the replaceable positions keeps evidence
    # renders spread across the complete active interval. Protected visits
    # stay in their original full-scene epoch order.
    cumulative = np.floor(
        np.arange(
            1, len(replaceable_positions) + 1, dtype=np.float64
        )
        * evidence_steps
        / max(len(replaceable_positions), 1)
    )
    previous = np.concatenate(
        [np.zeros(1, dtype=np.float64), cumulative[:-1]]
    )
    choose_replaceable = cumulative > previous
    choose = np.zeros(len(result), dtype=bool)
    choose[
        replaceable_positions[choose_replaceable]
    ] = True
    evidence_steps = int(choose.sum())
    evidence_schedule = _cycle_schedule(
        evidence_indices, evidence_steps, seed
    )
    result[choose] = evidence_schedule
    return result, {
        "evidence_view_count": len(evidence_indices),
        "evidence_steps": evidence_steps,
        "rgb_steps": int(len(result) - evidence_steps),
        "requested_fraction": fraction,
        "realized_fraction": (
            evidence_steps / max(len(active_positions), 1)
        ),
        **base_audit,
    }


def _resume_conditioned_visit_counts(
    state: dict,
    view_count: int,
) -> tuple[np.ndarray, dict]:
    """Restore the visits that really reached the conditioned optimizer.

    A schedule-repair resume deliberately replaces only the *future*
    conditioned camera schedule.  Recounting the repaired schedule from step
    zero would therefore invent visits that never occurred before the resume
    boundary.  New checkpoints persist an explicit runtime ledger; legacy
    checkpoints are migrated exactly once from their own saved schedule and
    phase contract.
    """
    view_count = max(int(view_count), 0)
    explicit = state.get("conditioned_visit_counts")
    if explicit is not None:
        if torch.is_tensor(explicit):
            explicit = explicit.detach().cpu().numpy()
        counts = np.asarray(explicit)
        if (
            counts.ndim != 1
            or len(counts) != view_count
            or not np.issubdtype(counts.dtype, np.integer)
            or bool((counts < 0).any())
        ):
            raise RuntimeError(
                "Resume conditioned visit ledger is invalid"
            )
        return counts.astype(np.int64, copy=True), {
            "source": "explicit_runtime_ledger",
            "carried_conditioned_steps": int(counts.sum()),
        }

    schedule = state.get("schedules", {}).get("conditioned")
    profile = state.get(
        "training_profile",
        state.get("training_contract", {}).get("training_profile"),
    )
    horizon = int(
        state.get(
            "schedule_horizon",
            state.get("training_contract", {}).get(
                "schedule_horizon", 0
            ),
        )
    )
    stop = int(state.get("iteration", 0))
    if schedule is None or profile not in TRAINING_PROFILES or horizon <= 0:
        raise RuntimeError(
            "Resume checkpoint cannot reconstruct conditioned visits"
        )
    schedule = np.asarray(schedule, dtype=np.int64).reshape(-1)
    stop = min(max(stop, 0), len(schedule), horizon)
    active_steps = np.asarray(
        [
            step
            for step in range(stop)
            if _dynamic_enabled(step, horizon, str(profile))
            and _phase(step, horizon, str(profile))
            != "canonical_polish"
        ],
        dtype=np.int64,
    )
    active = schedule[active_steps] if len(active_steps) else schedule[:0]
    if active.size and (
        int(active.min()) < 0 or int(active.max()) >= view_count
    ):
        raise RuntimeError(
            "Resume conditioned schedule contains an invalid camera"
        )
    counts = np.bincount(active, minlength=view_count)[:view_count]
    return counts.astype(np.int64, copy=False), {
        "source": "legacy_saved_schedule_exact_migration",
        "carried_conditioned_steps": int(active.size),
        "resume_iteration": stop,
    }


def _schedule_digest(*schedules: np.ndarray) -> str:
    digest = hashlib.sha256()
    for schedule in schedules:
        digest.update(np.ascontiguousarray(schedule).view(np.uint8))
    return digest.hexdigest()


def _restore_resume_camera_schedules(
    computed: dict[str, np.ndarray],
    resume: dict | None,
    *,
    horizon: int,
    allow_conditioned_repair: bool = False,
) -> tuple[dict[str, np.ndarray], frozenset[str]]:
    """Restore immutable checkpoint schedules before hashing/consumption.

    Some evidence schedules are derived from the current primitive metadata.
    That metadata legitimately changes during training, so recomputing those
    schedules after loading a checkpoint does *not* reproduce the original
    optimization trajectory.  The checkpoint arrays are the authority for an
    exact resume.  Explicit schedule-repair modes may replace only the stream
    named by their migration contract; every other stream remains byte exact.
    """

    schedules = {
        name: np.asarray(value, dtype=np.int64)
        for name, value in computed.items()
    }
    if resume is None:
        return schedules, frozenset()
    saved = resume.get("schedules")
    if not isinstance(saved, dict):
        raise RuntimeError("Resume checkpoint has no camera schedules")
    preserved = set(schedules)
    if allow_conditioned_repair:
        preserved.discard("conditioned")
    expected_shape = (int(horizon),)
    for name in sorted(preserved):
        if name not in saved:
            raise RuntimeError(
                f"Resume checkpoint has no {name} camera schedule"
            )
        value = np.asarray(saved[name], dtype=np.int64)
        if value.shape != expected_shape:
            raise RuntimeError(
                "Resume camera schedule length does not match the fixed "
                f"horizon: {name} has {value.shape}, expected "
                f"{expected_shape}"
            )
        schedules[name] = value.copy()
    return schedules, frozenset(preserved)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _immutable_ray_epoch_capacity(value: dict) -> dict:
    """Validate a runtime ray-capacity audit and return its fixed identity.

    A checkpoint restores per-camera cursors, so ``prior_factor_calls`` and
    ``remaining_unvisited_row_count`` must change at every resume boundary.
    They are runtime provenance, not a different optimization contract.  The
    clean full-horizon batch, camera table and evidence row count are the
    immutable identity that makes a truncated run an exact prefix.
    """

    required = {
        "batch_boundary_contract",
        "capacity_contract",
        "capacity_scope",
        "scheduled_factor_calls",
        "prior_factor_calls",
        "available_factor_calls",
        "camera_table_count",
        "full_effective_row_count",
        "remaining_unvisited_row_count",
        "clean_epoch_aggregate_minimum_batch_with_five_percent_margin",
        "clean_epoch_minimum_batch_for_complete_per_camera_epoch",
        "clean_epoch_effective_batch",
        "runtime_aggregate_minimum_batch_with_five_percent_margin",
        "minimum_batch_for_runtime_completion",
        "required_factor_calls",
        "unused_factor_calls",
        "complete_epoch_capacity",
    }
    if set(value) != required:
        raise RuntimeError(
            "Ray evidence epoch capacity audit has an invalid schema"
        )
    scheduled = int(value["scheduled_factor_calls"])
    prior = int(value["prior_factor_calls"])
    available = int(value["available_factor_calls"])
    full_rows = int(value["full_effective_row_count"])
    remaining = int(value["remaining_unvisited_row_count"])
    required_calls = int(value["required_factor_calls"])
    unused_calls = int(value["unused_factor_calls"])
    expected_scope = (
        "clean_epoch_start" if prior == 0 else "resume_remaining_unvisited_rows"
    )
    clean_batch = int(value["clean_epoch_effective_batch"])
    clean_minimum = int(
        value["clean_epoch_minimum_batch_for_complete_per_camera_epoch"]
    )
    clean_aggregate = int(
        value[
            "clean_epoch_aggregate_minimum_batch_with_five_percent_margin"
        ]
    )
    camera_count = int(value["camera_table_count"])
    empty_evidence = (
        camera_count == 0
        and full_rows == 0
        and remaining == 0
        and clean_minimum == 0
        and clean_aggregate == 0
        and int(value["runtime_aggregate_minimum_batch_with_five_percent_margin"])
        == 0
        and int(value["minimum_batch_for_runtime_completion"]) == 0
        and required_calls == 0
    )
    nonempty_evidence = (
        camera_count > 0
        and full_rows > 0
        # A checkpoint can land exactly after every per-camera sampler has
        # completed its first no-wrap epoch.  The immutable evidence identity
        # is still non-empty even though no rows remain in that epoch; the
        # next interval starts a new coverage epoch from the persisted cursor.
        and 0 <= remaining <= full_rows
        and clean_batch >= max(clean_minimum, clean_aggregate)
    )
    if (
        value["batch_boundary_contract"] != RAY_EPOCH_BOUNDARY_CONTRACT
        or value["capacity_contract"] != RAY_EPOCH_CAPACITY_CONTRACT
        or value["capacity_scope"] != expected_scope
        or scheduled <= 0
        or prior < 0
        or prior >= scheduled
        or available != scheduled - prior
        or not (empty_evidence or nonempty_evidence)
        or required_calls < 0
        or required_calls > available
        or unused_calls != available - required_calls
        or not bool(value["complete_epoch_capacity"])
    ):
        raise RuntimeError(
            "Ray evidence epoch capacity audit is internally inconsistent"
        )
    return {
        "batch_boundary_contract": value["batch_boundary_contract"],
        "capacity_contract": value["capacity_contract"],
        "scheduled_factor_calls": scheduled,
        "camera_table_count": camera_count,
        "full_effective_row_count": full_rows,
        "clean_epoch_aggregate_minimum_batch_with_five_percent_margin": (
            clean_aggregate
        ),
        "clean_epoch_minimum_batch_for_complete_per_camera_epoch": (
            clean_minimum
        ),
        "clean_epoch_effective_batch": clean_batch,
    }


def _resume_training_contract_differences(
    saved: dict,
    current: dict,
    *,
    allow_trainer_repair_migration: bool = False,
    allow_conditioned_schedule_repair_migration: bool = False,
    allow_volume_capacity_repair_migration: bool = False,
    allow_volume_topology_settle_migration: bool = False,
    allow_volume_opacity_settle_migration: bool = False,
    allow_surface_ownership_repair_migration: bool = False,
    allow_v38_causal_repair_migration: bool = False,
    allow_static_detail_isolated_repair_migration: bool = False,
    allow_static_canonical_ownership_repair_migration: bool = False,
    allow_surface_screen_evidence_repair_migration: bool = False,
) -> set[str]:
    """Compare optimization identity without treating an RGB cache as K.

    ``camera_intrinsics_contract.json`` also records the size of the CPU
    image cache.  Its whole-file digest remains useful provenance, but that
    performance-only field cannot make an otherwise exact K/pose checkpoint
    ineligible for resume.  Camera identity is independently bound by the
    geometry digest and the exact per-camera validation payload.
    """
    # CPU pools change throughput only. They do not alter a camera/evidence
    # schedule, parameter owner, optimizer, CUDA image formation or model
    # tensor, and therefore must not make an interrupted run ineligible for
    # resume on a host with a different CPU allocation. Older checkpoints
    # persisted this runtime audit in the immutable optimization contract;
    # normalize that one legacy field away during comparison.
    saved = dict(saved)
    current = dict(current)
    saved.pop("cpu_parallelism", None)
    current.pop("cpu_parallelism", None)
    # Older checkpoints accidentally omitted three volume-topology controls
    # from the immutable optimization contract.  An explicit trainer-repair
    # resume may migrate those missing fields once; the next checkpoint
    # persists them and all later resumes compare them normally.
    if allow_trainer_repair_migration:
        saved = dict(saved)
        for key in (
            "maximum_volume_scale",
            "maximum_dynamic_volume_scale",
            "volume_split_radius",
        ):
            if key not in saved and key in current:
                saved[key] = current[key]
        current_receiver_materialization = current.get(
            "static_detail_receiver_materialization"
        )
        if (
            "static_detail_receiver_materialization" not in saved
            and isinstance(current_receiver_materialization, dict)
            and current_receiver_materialization.get("birth_forward_contract")
            == "co_located_same_covariance_same_color_tau_partition"
            and bool(
                current_receiver_materialization.get(
                    "integrated_optical_mass_conserved"
                )
            )
            and bool(
                current_receiver_materialization.get(
                    "existing_detail_group_repeat_forbidden"
                )
            )
        ):
            # The checkpoint tensors remain byte-for-byte valid.  This only
            # authorizes future topology events to factor envelope mass into
            # local detail receivers under the existing global budget.
            saved["static_detail_receiver_materialization"] = (
                current_receiver_materialization
            )
        saved_static_fusion = saved.get("static_fusion")
        current_static_fusion = current.get("static_fusion")
        static_fusion_identity_fields = (
            "contract",
            "input_dynamic_rows",
            "associated_dynamic_rows",
            "camera_sequence_metadata_source",
            "camera_sequence_metadata_count",
            "canonical_sequence_policy",
            "canonical_mode_voxel_size",
            "maximum_modes_per_group",
            "minimum_supporting_views",
            "minimum_supporting_sequences",
            "voxel_size",
            "output_static_rows",
        )
        if (
            isinstance(saved_static_fusion, dict)
            and isinstance(current_static_fusion, dict)
            and saved.get("initialization_manifest_sha256")
            == current.get("initialization_manifest_sha256")
            and saved.get("foliage_seed_sha256")
            == current.get("foliage_seed_sha256")
            and all(
                saved_static_fusion.get(key)
                == current_static_fusion.get(key)
                for key in static_fusion_identity_fields
            )
        ):
            # On resume the fused seed payload is audit-only: the exact
            # foliage tensors are restored from the checkpoint. Parallel CPU
            # reductions can change sub-ulp aggregate mass diagnostics even
            # when the immutable seed, camera set and fusion configuration
            # are identical. Compare those identity fields above, then retain
            # the saved audit rather than rejecting a valid model resume.
            current["static_fusion"] = saved_static_fusion
        if (
            saved.get("static_optical_policy_contract")
            in {
                STATIC_OPTICAL_POLICY_PREDECESSOR_CONTRACT,
                STATIC_OPTICAL_POLICY_STAGE3_FREEZE_PREDECESSOR_CONTRACT,
            }
            and current.get("static_optical_policy_contract")
            == STATIC_OPTICAL_POLICY_CONTRACT
            and saved.get("reconstruction_target") == "static"
            and current.get("reconstruction_target") == "static"
        ):
            # This repair changes only future gradient routing. The resumed
            # parameter/optimizer tensors, schedules, evidence and renderer
            # state are identical at the boundary.
            saved["static_optical_policy_contract"] = (
                STATIC_OPTICAL_POLICY_CONTRACT
            )
        # v69 correctly support-owned static-detail geometry, mass and
        # topology, but accidentally left SH appearance globally writable.
        # The same bug also scheduled isolated detail renders from traversals
        # with no positive detail owner. This migration preserves every
        # tensor, Adam moment and non-detail camera schedule while changing
        # only future positive RGB routing and its dedicated camera cycle.
        saved_stages = saved.get("static_training_stages")
        current_stages = current.get("static_training_stages")
        saved_detail = saved.get("static_detail_isolated_supervision")
        current_detail = current.get("static_detail_isolated_supervision")
        saved_cleanup = saved.get("static_detail_global_cleanup")
        current_cleanup = current.get("static_detail_global_cleanup")
        if (
            saved.get("reconstruction_target") == "static"
            and current.get("reconstruction_target") == "static"
            and isinstance(saved_stages, dict)
            and isinstance(current_stages, dict)
            and saved_stages.get("rgb_role_contract")
            == STATIC_STAGE_RGB_ROLE_PREDECESSOR_CONTRACT
            and current_stages.get("rgb_role_contract")
            == STATIC_STAGE_RGB_ROLE_CONTRACT
            and isinstance(saved_detail, dict)
            and isinstance(current_detail, dict)
            and saved_detail.get("contract")
            == STATIC_DETAIL_ISOLATED_APPEARANCE_PREDECESSOR_CONTRACT
            and current_detail.get("contract")
            == STATIC_DETAIL_ISOLATED_CONTRACT
            and saved_detail.get("every") == current_detail.get("every")
            and saved_detail.get("weight") == current_detail.get("weight")
            and isinstance(saved_cleanup, dict)
            and isinstance(current_cleanup, dict)
            and saved_cleanup.get("contract")
            == STATIC_DETAIL_GLOBAL_CLEANUP_APPEARANCE_PREDECESSOR_CONTRACT
            and current_cleanup.get("contract")
            == STATIC_DETAIL_GLOBAL_CLEANUP_CONTRACT
        ):
            saved["static_training_stages"] = current_stages
            saved["static_detail_isolated_supervision"] = current_detail
            saved["static_detail_global_cleanup"] = current_cleanup
            saved["sampling_schedule_sha256"] = current[
                "sampling_schedule_sha256"
            ]
        # v81 made geometry/mass ownership correctly camera exact, but used
        # that sparse seed-owner table as the SH permission as well.  v82
        # leaves every geometric/optical tensor and schedule unchanged and
        # only gives same-acquisition cameras a continuous appearance weight.
        # This is a future-gradient routing repair, so an explicit trainer
        # repair resume may preserve the checkpoint and optimizer tensors.
        saved_ownership = saved.get("static_detail_canonical_ownership")
        current_ownership = current.get(
            "static_detail_canonical_ownership"
        )
        if (
            saved.get("reconstruction_target") == "static"
            and current.get("reconstruction_target") == "static"
            and isinstance(saved_ownership, dict)
            and isinstance(current_ownership, dict)
            and saved_ownership.get("rgb_geometry_topology_owner")
            == "persisted_exact_support_cameras_and_verified_multiview"
            and current_ownership.get("geometry_mass_topology_owner")
            == "persisted_exact_support_cameras_and_verified_multiview"
            and current_ownership.get("appearance_owner")
            == (
                "exact_support_camera_weight1_plus_same_acquisition_"
                "sequence_soft_weight"
            )
            and 0.0
            <= float(
                current_ownership.get(
                    "same_sequence_appearance_weight", -1.0
                )
            )
            <= 1.0
        ):
            for key in (
                "static_training_stages",
                "static_detail_isolated_supervision",
                "static_volume_isolated_supervision",
                "static_detail_canonical_ownership",
                "parameter_loss_permission_matrix",
            ):
                if key in current:
                    saved[key] = current[key]
        # v83 materialized detail receivers, but positive opacity/ray
        # permission remained exact-camera sparse while negative free-space
        # cleanup remained all-camera global.  That asymmetric evidence
        # contract necessarily drives seasonal foliage toward transparency.
        # v84 changes only future gradient routing: checkpoint tensors,
        # optimizer moments, schedules, evidence tables and CUDA image
        # formation remain identical at the resume boundary.
        saved_ownership = saved.get("static_detail_canonical_ownership")
        current_ownership = current.get(
            "static_detail_canonical_ownership"
        )
        saved_cleanup = saved.get("static_detail_global_cleanup")
        current_cleanup = current.get("static_detail_global_cleanup")
        if (
            saved.get("reconstruction_target") == "static"
            and current.get("reconstruction_target") == "static"
            and isinstance(saved_ownership, dict)
            and isinstance(current_ownership, dict)
            and saved_ownership.get("geometry_mass_topology_owner")
            == "persisted_exact_support_cameras_and_verified_multiview"
            and current_ownership.get("geometry_topology_owner")
            == "persisted_exact_support_cameras_and_verified_multiview"
            and current_ownership.get("optical_mass_owner")
            == (
                "exact_support_camera_weight1_plus_positive_evidence_"
                "sequence_soft_weight"
            )
            and float(
                current_ownership.get(
                    "same_sequence_optical_weight", -1.0
                )
            )
            == 0.35
            and isinstance(saved_cleanup, dict)
            and isinstance(current_cleanup, dict)
            and saved_cleanup.get("contract")
            == STATIC_DETAIL_GLOBAL_CLEANUP_V83_PREDECESSOR_CONTRACT
            and current_cleanup.get("contract")
            == STATIC_DETAIL_GLOBAL_CLEANUP_CONTRACT
        ):
            for key in (
                "static_training_stages",
                "static_detail_global_cleanup",
                "static_detail_canonical_ownership",
                "parameter_loss_permission_matrix",
            ):
                if key in current:
                    saved[key] = current[key]
        current_static_uncertainty = current.get(
            "static_spatial_uncertainty"
        )
        if (
            "static_spatial_uncertainty" not in saved
            and isinstance(current_static_uncertainty, dict)
            and current_static_uncertainty.get("contract")
            == (
                "training_only_per_image_degree2_legendre_"
                "heteroscedastic_canopy_sky_field"
            )
            and not bool(current_static_uncertainty.get("rgb_residual"))
            and not bool(
                current_static_uncertainty.get(
                    "geometry_or_opacity_owner"
                )
            )
            and not bool(
                current_static_uncertainty.get("deployment_parameter")
            )
        ):
            # The appended decoder starts at exact zero, so the first resumed
            # forward is identical to v4. Existing Adam tensors migrate
            # losslessly; only future static-detail visits learn the robust
            # scale field.
            saved["static_spatial_uncertainty"] = (
                current_static_uncertainty
            )
        # Static v52 checkpoints coupled canonical appearance ownership to
        # every geometry/opacity contradiction.  The repaired contract adds
        # an all-camera negative-evidence path while preserving the exact
        # model tensors, optimizer state, schedules and CUDA image formation.
        # Permit this declaration-only migration only when the executable
        # target advertises the complete loss/parameter permission contract.
        current_global_cleanup = current.get(
            "static_detail_global_cleanup"
        )
        if (
            "static_detail_global_cleanup" not in saved
            and isinstance(current_global_cleanup, dict)
            and current_global_cleanup.get("contract")
            == STATIC_DETAIL_GLOBAL_CLEANUP_CONTRACT
        ):
            saved["static_detail_global_cleanup"] = current_global_cleanup
            for key in (
                "static_training_stages",
                "static_detail_canonical_ownership",
                "parameter_loss_permission_matrix",
            ):
                if key in current:
                    saved[key] = current[key]
        current_envelope_cleanup = current.get(
            "persistent_envelope_global_cleanup"
        )
        if (
            "persistent_envelope_global_cleanup" not in saved
            and isinstance(current_envelope_cleanup, dict)
            and current_envelope_cleanup.get("contract")
            == PERSISTENT_ENVELOPE_GLOBAL_CLEANUP_CONTRACT
        ):
            # The repair adds a role-isolated negative-evidence render only.
            # It does not change any tensor, schedule, evidence table or CUDA
            # image-formation implementation at the resume boundary.
            saved["persistent_envelope_global_cleanup"] = (
                current_envelope_cleanup
            )
            for key in (
                "static_training_stages",
                "parameter_loss_permission_matrix",
            ):
                if key in current:
                    saved[key] = current[key]
        current_static_ray_birth = current.get("static_ray_birth")
        if (
            "static_ray_birth" not in saved
            and isinstance(current_static_ray_birth, dict)
            and current_static_ray_birth.get("contract")
            == STATIC_RAY_BIRTH_VISUAL_HULL_CONTRACT
        ):
            # v3 checkpoints retain midpoint cells. The v4 accumulator
            # restores them exactly, then starts interval visual-hull voting
            # only from future ray batches; model and Adam tensors are
            # unchanged at the repair boundary.
            saved["static_ray_birth"] = current_static_ray_birth
        if (
            isinstance(saved.get("static_ray_birth"), dict)
            and isinstance(current_static_ray_birth, dict)
            and saved["static_ray_birth"].get("contract")
            == STATIC_RAY_BIRTH_VISUAL_HULL_CONTRACT
            and current_static_ray_birth.get("contract")
            == STATIC_RAY_BIRTH_VISUAL_HULL_CONTRACT
            and current_static_ray_birth.get("consumed_cell_policy")
            == "persistent_midpoint_and_visual_hull_tombstones"
            and current_static_ray_birth.get("initial_mass_policy")
            == "support_ray_local_transfer_from_verified_envelope"
        ):
            # The checkpoint model is unchanged at this boundary. Future
            # drained keys acquire persistent tombstones; pre-v5 keys are
            # reconstructed from extant initialization_source=6 rows. Future
            # births fund their initial mass from an envelope instead of
            # adding a second extinction layer.
            saved["static_ray_birth"] = current_static_ray_birth
        # v69 initially applied verification-debt backpressure only to
        # ordinary splits. At the same time, a support-sequence permission
        # miss could be reported as an uncovered hit and append another full
        # batch of unverified static births. The repair changes no tensor at
        # the resume boundary: it lets already verified two-sequence detail
        # consume future analytic hit rays and shares the existing continuous
        # debt capacity with future ray births.
        saved_ownership = saved.get("static_detail_canonical_ownership")
        current_ownership = current.get(
            "static_detail_canonical_ownership"
        )
        saved_ray_birth = saved.get("static_ray_birth")
        current_ray_birth = current.get("static_ray_birth")
        legacy_ray_birth_contract = (
            "uncovered_hit_uses_complete_calibrated_ray_depth_interval__"
            "cross_sequence_segments_vote_in_multiscale_visual_hull_cells__"
            "minimum_two_cameras_and_two_sequences_remain_mandatory__"
            "newborn_is_low_mass_static_detail_and_free_space_prunable"
        )
        if (
            isinstance(saved_ownership, dict)
            and isinstance(current_ownership, dict)
            and saved_ownership.get("gradient_owner")
            == "persisted_support_camera_sequences"
            and current_ownership.get("rgb_geometry_topology_owner")
            == "persisted_support_camera_sequences"
            and current_ownership.get("positive_ray_owner")
            == "persisted_support_or_verified_two_sequence_consensus"
            and isinstance(saved_ray_birth, dict)
            and isinstance(current_ray_birth, dict)
            and saved_ray_birth.get("contract")
            == legacy_ray_birth_contract
            and current_ray_birth.get("contract")
            == STATIC_RAY_BIRTH_VISUAL_HULL_CONTRACT
            and current_ray_birth.get("verification_debt_backpressure")
            == "same_continuous_smoothstep_capacity_as_ordinary_split"
        ):
            for key in (
                "static_training_stages",
                "static_detail_isolated_supervision",
                "static_detail_global_cleanup",
                "static_ray_birth",
                "static_detail_canonical_ownership",
                "parameter_loss_permission_matrix",
            ):
                if key in current:
                    saved[key] = current[key]
        if (
            saved.get("dynamic_lifecycle_contract")
            == DYNAMIC_LIFECYCLE_REPAIR_PREDECESSOR
            and current.get("dynamic_lifecycle_contract")
            == DYNAMIC_LIFECYCLE_REPAIR_TARGET
        ):
            saved["dynamic_lifecycle_contract"] = (
                DYNAMIC_LIFECYCLE_REPAIR_TARGET
            )
        saved_patch = saved.get("rigid_residual_patch")
        current_patch = current.get("rigid_residual_patch")
        if isinstance(saved_patch, dict) and isinstance(
            current_patch, dict
        ):
            saved_patch_without_selection = {
                key: value
                for key, value in saved_patch.items()
                if key != "selection"
            }
            current_patch_without_selection = {
                key: value
                for key, value in current_patch.items()
                if key != "selection"
            }
            if (
                saved_patch.get("selection")
                == RIGID_PATCH_NMS_REPAIR_PREDECESSOR
                and current_patch.get("selection")
                == RIGID_PATCH_NMS_REPAIR_TARGET
                and saved_patch_without_selection
                == current_patch_without_selection
            ):
                saved["rigid_residual_patch"] = current_patch
        # v21 explicitly extended evidence-driven volume topology to 10k but
        # accidentally left the continuous Chart UV quadtree on the generic
        # 3k default. Permit only this exact horizon repair; camera/evidence,
        # optimizer and phase schedules remain identical.
        if (
            saved.get("densify_until_iter")
            == CHART_TOPOLOGY_HORIZON_REPAIR_PREDECESSOR
            and current.get("densify_until_iter")
            == CHART_TOPOLOGY_HORIZON_REPAIR_TARGET
            and saved.get("volume_densify_until_iteration")
            == CHART_TOPOLOGY_HORIZON_REPAIR_TARGET
            and current.get("volume_densify_until_iteration")
            == CHART_TOPOLOGY_HORIZON_REPAIR_TARGET
        ):
            saved["densify_until_iter"] = (
                CHART_TOPOLOGY_HORIZON_REPAIR_TARGET
            )
    if allow_static_detail_isolated_repair_migration:
        saved = dict(saved)
        current_detail = current.get("static_detail_isolated_supervision")
        if (
            "static_detail_isolated_supervision" in saved
            or not isinstance(current_detail, dict)
            or current_detail.get("contract")
            != STATIC_DETAIL_ISOLATED_CONTRACT
            or int(current_detail.get("every", 0)) < 0
            or float(current_detail.get("weight", 0.0)) < 0.0
        ):
            raise RuntimeError(
                "Static-detail repair requires an exact predecessor without "
                "isolated supervision and a valid v48 target objective"
            )
        saved["static_detail_isolated_supervision"] = current_detail
        current_stages = current.get("static_training_stages")
        current_permissions = current.get(
            "parameter_loss_permission_matrix"
        )
        expected_signal = "surface_plus_detail_isolated_rgb_high_frequency"
        if (
            not isinstance(current_stages, dict)
            or expected_signal
            not in current_stages.get("detail_training_signals", [])
            or not isinstance(current_permissions, dict)
            or "detail_isolated_rgb_high_frequency"
            not in current_permissions.get("static_leaf_optical_mass", [])
        ):
            raise RuntimeError(
                "Static-detail repair target is missing its staged signal or "
                "parameter permission declaration"
            )
        saved["static_training_stages"] = current_stages
        saved["parameter_loss_permission_matrix"] = current_permissions
        current_topology = current.get("static_detail_topology")
        if (
            not isinstance(current_topology, dict)
            or not bool(current_topology.get("exclusive_after_envelope_stage"))
        ):
            raise RuntimeError(
                "Static-detail repair target must enable staged exclusive "
                "detail topology"
            )
        saved["static_detail_topology"] = current_topology
    if allow_static_canonical_ownership_repair_migration:
        saved = dict(saved)
        current_ownership = current.get(
            "static_detail_canonical_ownership"
        )
        if (
            not isinstance(current_ownership, dict)
            or not bool(current_ownership.get("enabled"))
            or current_ownership.get("forward_visibility")
            != "unconditional_static"
            or (
                current_ownership.get(
                    "rgb_geometry_topology_owner",
                    current_ownership.get("gradient_owner"),
                )
                != "persisted_support_camera_sequences"
            )
        ):
            raise RuntimeError(
                "Static canonical-ownership repair requires an exact v50 "
                "predecessor and the enabled v51 ownership contract"
            )
        saved_ownership = saved.get("static_detail_canonical_ownership")
        if saved_ownership is None:
            saved["static_detail_canonical_ownership"] = current_ownership
        elif saved_ownership != current_ownership:
            raise RuntimeError(
                "Static canonical detail schedule changed parameter "
                "ownership while migrating the view schedule"
            )
        current_detail = current.get("static_detail_isolated_supervision")
        view_schedule = (
            current_detail.get("view_schedule", {})
            if isinstance(current_detail, dict)
            else {}
        )
        if (
            not isinstance(current_detail, dict)
            or view_schedule.get("contract")
            != (
                "uniform_complete_epochs_over_visible_canonical_"
                "support_cameras"
            )
            or int(view_schedule.get("camera_count", 0)) <= 0
        ):
            raise RuntimeError(
                "v51 requires a complete visible canonical-camera detail "
                "schedule"
            )
        # The 3k predecessor is still in the envelope-only stage.  No static
        # detail forward or gradient has run, so replacing the same-view
        # dormant objective by its canonical-view schedule is exact.
        saved["static_detail_isolated_supervision"] = current_detail
    if allow_conditioned_schedule_repair_migration:
        saved = dict(saved)
        saved_sampling = saved.get("conditioned_sampling", {})
        current_sampling = current.get("conditioned_sampling", {})
        if not (
            isinstance(saved_sampling, dict)
            and isinstance(current_sampling, dict)
            and not bool(saved_sampling.get("coverage_preserving", False))
            and bool(current_sampling.get("coverage_preserving", False))
        ):
            raise RuntimeError(
                "Conditioned schedule repair requires a legacy replacing "
                "schedule and a coverage-preserving target"
            )
        # The digest changes only because the conditioned camera sequence is
        # repaired. ``schedule_hash`` below remains the stronger full-array
        # check; RGB, geometry and topology schedules are compared there
        # before this optimization-contract migration is accepted.
        saved["sampling_schedule_sha256"] = current[
            "sampling_schedule_sha256"
        ]
        saved["conditioned_sampling"] = current[
            "conditioned_sampling"
        ]
    if allow_volume_capacity_repair_migration:
        saved = dict(saved)
        expected_old = {
            "maximum_volume_gaussians": (
                VOLUME_CAPACITY_REPAIR_PREDECESSOR[
                    "maximum_volume_gaussians"
                ]
            ),
            "maximum_volume_splits_per_event": (
                VOLUME_CAPACITY_REPAIR_PREDECESSOR[
                    "maximum_volume_splits_per_event"
                ]
            ),
        }
        expected_new = {
            **VOLUME_CAPACITY_REPAIR_TARGET,
            "boundary_supervision_contract": (
                BOUNDARY_SUPERVISION_CONTRACT
            ),
            "volume_reallocation_contract": (
                VOLUME_REALLOCATION_CONTRACT
            ),
        }
        for key, value in expected_old.items():
            if saved.get(key) != value:
                raise RuntimeError(
                    "Capacity repair predecessor contract mismatch for "
                    f"{key}: {saved.get(key)} != {value}"
                )
        for key, value in expected_new.items():
            if current.get(key) != value:
                raise RuntimeError(
                    "Capacity repair target contract mismatch for "
                    f"{key}: {current.get(key)} != {value}"
                )
        # v82 predates these explicit fields. Its phase-based volume topology
        # ended at dynamic_appearance (8,640/12,000), which is preserved by
        # the default path. Only this exact migration writes the new target
        # values into the comparison copy.
        for key, value in expected_new.items():
            saved[key] = value
        saved["maximum_volume_gaussians"] = current[
            "maximum_volume_gaussians"
        ]
        saved["maximum_volume_splits_per_event"] = current[
            "maximum_volume_splits_per_event"
        ]
    if allow_volume_topology_settle_migration:
        saved = dict(saved)
        if (
            saved.get("schedule_horizon") != 12_000
            or current.get("schedule_horizon") != 12_000
            or saved.get("volume_densify_until_iteration") != 12_000
            or current.get("volume_densify_until_iteration") != 6_000
        ):
            raise RuntimeError(
                "Topology-settle migration requires an unchanged 12k "
                "schedule and exactly the audited 12k->6k volume topology "
                "end"
            )
        saved["volume_densify_until_iteration"] = 6_000
    if allow_volume_opacity_settle_migration:
        saved = dict(saved)
        saved_settle = saved.get("volume_opacity_settle")
        current_settle = current.get("volume_opacity_settle")
        predecessor_without_window = bool(
            isinstance(saved_settle, dict)
            and isinstance(current_settle, dict)
            and saved_settle.get("contract")
            == VOLUME_OPACITY_SETTLE_PREDECESSOR_CONTRACT
            and current_settle.get("contract")
            == VOLUME_OPACITY_SETTLE_CONTRACT
            and saved_settle.get("policy")
            == current_settle.get("policy")
            and saved_settle.get("start_iteration")
            == current_settle.get("start_iteration")
            and saved_settle.get("trainable_after_settle")
            == current_settle.get("trainable_after_settle")
            and current_settle.get("retirement_until_iteration")
            in {9_000, 12_000}
        )
        retirement_to_freeze = bool(
            isinstance(saved_settle, dict)
            and isinstance(current_settle, dict)
            and saved_settle.get("contract")
            in {
                VOLUME_OPACITY_SETTLE_PREDECESSOR_CONTRACT,
                VOLUME_OPACITY_SETTLE_CONTRACT,
            }
            and current_settle.get("contract")
            == VOLUME_OPACITY_SETTLE_CONTRACT
            and saved_settle.get("policy") == "retirement_only"
            and current_settle.get("policy") == "freeze"
            and saved_settle.get("start_iteration")
            == current_settle.get("start_iteration")
            and saved_settle.get("trainable_after_settle")
            == current_settle.get("trainable_after_settle")
        )
        if (
            saved.get("schedule_horizon") != 12_000
            or current.get("schedule_horizon") != 12_000
            or not isinstance(current_settle, dict)
            or current_settle.get("contract")
            != VOLUME_OPACITY_SETTLE_CONTRACT
            or current_settle.get("policy")
            not in {"freeze", "retirement_only"}
            or current_settle.get("start_iteration") != 6_000
            or (
                "volume_opacity_settle" in saved
                and not predecessor_without_window
                and not retirement_to_freeze
            )
        ):
            raise RuntimeError(
                "Opacity-settle migration requires an unchanged 12k "
                "schedule, the audited 6k settle point, a freeze or "
                "retirement-only target, and either a predecessor without "
                "this contract or the exact retirement-only -> freeze "
                "polish transition"
            )
        saved["volume_opacity_settle"] = current_settle
    if allow_surface_ownership_repair_migration:
        saved = dict(saved)
        old_policy = SURFACE_OWNERSHIP_REPAIR_PREDECESSOR[
            "mature_handoff_surface_policy"
        ]
        if saved.get("mature_handoff_surface_policy") != old_policy:
            raise RuntimeError(
                "Surface ownership repair predecessor policy mismatch: "
                f"{saved.get('mature_handoff_surface_policy')} != "
                f"{old_policy}"
            )
        for key, value in SURFACE_OWNERSHIP_REPAIR_TARGET.items():
            if current.get(key) != value:
                raise RuntimeError(
                    "Surface ownership repair target contract mismatch for "
                    f"{key}: {current.get(key)} != {value}"
                )
            saved[key] = value
    if allow_v38_causal_repair_migration:
        saved = dict(saved)
        if (
            saved.get("schedule_horizon") != 12_000
            or current.get("schedule_horizon") != 12_000
            or saved.get("volume_densify_until_iteration") != 10_000
            or current.get("volume_densify_until_iteration") != 12_000
        ):
            raise RuntimeError(
                "v38 migration requires an unchanged 12k schedule horizon "
                "and only the audited 10k->12k volume topology extension"
            )
        expected_counterfactual = {
            # This migration identifies the historical v38 target exactly;
            # the current v69 local-permission contract is intentionally not
            # resume-compatible with that causal checkpoint.
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
        }
        if current.get("counterfactual_transparency") != (
            expected_counterfactual
        ):
            raise RuntimeError(
                "v38 counterfactual transparency parameters differ from "
                "the audited causal-repair target"
            )
        if "counterfactual_transparency" in saved:
            raise RuntimeError(
                "v38 migration predecessor unexpectedly already contains "
                "the counterfactual transparency contract"
            )
        saved["counterfactual_transparency"] = expected_counterfactual
        saved["volume_densify_until_iteration"] = 12_000
        saved_ray = dict(saved.get("ray_evidence_epoch_capacity", {}))
        current_ray = current.get("ray_evidence_epoch_capacity", {})
        expected_saved_ray_keys = {
            "scheduled_factor_calls",
            "effective_row_count",
            "minimum_batch_with_five_percent_margin",
            "complete_epoch_capacity",
        }
        if (
            set(saved_ray) != expected_saved_ray_keys
            or not bool(saved_ray.get("complete_epoch_capacity", False))
            or saved_ray.get("scheduled_factor_calls")
            != current_ray.get("scheduled_factor_calls")
            or saved_ray.get("effective_row_count")
            != current_ray.get("full_effective_row_count")
            or saved_ray.get("minimum_batch_with_five_percent_margin")
            != current_ray.get(
                "clean_epoch_aggregate_minimum_batch_with_five_percent_margin"
            )
            or saved.get("ray_posterior_maximum_rays")
            != saved_ray.get("minimum_batch_with_five_percent_margin")
        ):
            raise RuntimeError(
                "v38 predecessor ray epoch is not the exact audited "
                "aggregate-capacity contract"
            )
        expected_current_ray_keys = {
            "batch_boundary_contract",
            "capacity_contract",
            "capacity_scope",
            "scheduled_factor_calls",
            "prior_factor_calls",
            "available_factor_calls",
            "camera_table_count",
            "full_effective_row_count",
            "remaining_unvisited_row_count",
            "clean_epoch_aggregate_minimum_batch_with_five_percent_margin",
            "clean_epoch_minimum_batch_for_complete_per_camera_epoch",
            "clean_epoch_effective_batch",
            "runtime_aggregate_minimum_batch_with_five_percent_margin",
            "minimum_batch_for_runtime_completion",
            "required_factor_calls",
            "unused_factor_calls",
            "complete_epoch_capacity",
        }
        current_batch = int(current.get("ray_posterior_maximum_rays", -1))
        scheduled_calls = int(
            current_ray.get("scheduled_factor_calls", -1)
        )
        prior_calls = int(current_ray.get("prior_factor_calls", -1))
        available_calls = int(
            current_ray.get("available_factor_calls", -1)
        )
        required_calls = int(current_ray.get("required_factor_calls", -1))
        exact_minimum = int(
            current_ray.get(
                "minimum_batch_for_runtime_completion", -1
            )
        )
        aggregate_minimum = int(
            current_ray.get(
                "runtime_aggregate_minimum_batch_with_five_percent_margin",
                -1,
            )
        )
        clean_exact_minimum = int(
            current_ray.get(
                "clean_epoch_minimum_batch_for_complete_per_camera_epoch",
                -1,
            )
        )
        clean_aggregate_minimum = int(
            current_ray.get(
                "clean_epoch_aggregate_minimum_batch_with_five_percent_margin",
                -1,
            )
        )
        configured_batch = int(
            current.get("ray_posterior_configured_maximum_rays", -1)
        )
        if (
            set(current_ray) != expected_current_ray_keys
            or current_ray.get("batch_boundary_contract")
            != RAY_EPOCH_BOUNDARY_CONTRACT
            or current_ray.get("capacity_contract")
            != RAY_EPOCH_CAPACITY_CONTRACT
            or current_ray.get("capacity_scope")
            != "resume_remaining_unvisited_rows"
            or int(current_ray.get("camera_table_count", 0)) <= 0
            or prior_calls != 2_250
            or available_calls != scheduled_calls - prior_calls
            or int(current_ray.get("remaining_unvisited_row_count", 0))
            <= 0
            or int(current_ray.get("remaining_unvisited_row_count", -1))
            >= int(current_ray.get("full_effective_row_count", -1))
            or int(current_ray.get("clean_epoch_effective_batch", -1))
            != max(
                configured_batch,
                clean_aggregate_minimum,
                clean_exact_minimum,
            )
            or current_batch
            != max(configured_batch, aggregate_minimum, exact_minimum)
            or required_calls < 0
            or required_calls > available_calls
            or int(current_ray.get("unused_factor_calls", -1))
            != available_calls - required_calls
            or not bool(current_ray.get("complete_epoch_capacity", False))
        ):
            raise RuntimeError(
                "v38 target does not satisfy the resume-aware exact per-"
                "camera ray epoch capacity contract"
            )
        if current.get("volume_topology_phase_contract") != (
            VOLUME_TOPOLOGY_SETTLE_CONTRACT
        ):
            raise RuntimeError(
                "v38 volume topology does not reserve canonical_polish for "
                "settling"
            )
        if current.get("conditioned_settle_contract") != (
            CONDITIONED_SETTLE_CONTRACT
        ):
            raise RuntimeError(
                "v39 conditioned settle must keep the conditioned branch "
                "active after volume topology stops"
            )
        saved["ray_posterior_maximum_rays"] = current_batch
        saved["ray_evidence_epoch_capacity"] = current_ray
        saved["volume_topology_phase_contract"] = (
            VOLUME_TOPOLOGY_SETTLE_CONTRACT
        )
        saved["conditioned_settle_contract"] = (
            CONDITIONED_SETTLE_CONTRACT
        )
    # Runtime cursor/capacity fields necessarily advance between an exact
    # prefix and its resume.  Validate both complete audits, then compare only
    # the fixed full-horizon evidence identity.  This does not waive evidence
    # changes: camera count, row count, schedule and clean batch remain bound.
    saved_ray = saved.get("ray_evidence_epoch_capacity")
    current_ray = current.get("ray_evidence_epoch_capacity")
    modern_ray_keys = {
        "batch_boundary_contract",
        "capacity_contract",
        "capacity_scope",
    }
    if (
        isinstance(saved_ray, dict)
        and isinstance(current_ray, dict)
        and modern_ray_keys.issubset(saved_ray)
        and modern_ray_keys.issubset(current_ray)
    ):
        saved = dict(saved)
        current = dict(current)
        saved["ray_evidence_epoch_capacity"] = (
            _immutable_ray_epoch_capacity(saved_ray)
        )
        current["ray_evidence_epoch_capacity"] = (
            _immutable_ray_epoch_capacity(current_ray)
        )

    # Surface topology changes the number of live rows after every retained
    # checkpoint.  That cardinality is runtime state, not optimization
    # identity.  ``joint`` owns every row symmetrically, so its whole
    # diagnostic partition is runtime state.  The mature handoff policies do
    # use ``rigid_prefix_rows`` as an immutable ownership boundary, but the
    # suffix cardinality still grows and shrinks through legal topology
    # events.  Comparing that derived count made every atlas-residual
    # checkpoint with even one new child impossible to resume.
    if (
        saved.get("mature_handoff_surface_policy") == "joint"
        and current.get("mature_handoff_surface_policy") == "joint"
    ):
        saved = dict(saved)
        current = dict(current)
        saved.pop("mature_handoff_surface_partition", None)
        current.pop("mature_handoff_surface_partition", None)
    elif (
        saved.get("mature_handoff_surface_policy")
        == current.get("mature_handoff_surface_policy")
        and saved.get("mature_handoff_surface_policy")
        in {"atlas_residual", "appearance_only"}
    ):
        saved_partition = dict(
            saved.get("mature_handoff_surface_partition", {})
        )
        current_partition = dict(
            current.get("mature_handoff_surface_partition", {})
        )
        for runtime_key in (
            "evidence_completion_suffix_rows",
            # Compatibility with early handoff checkpoints.
            "completion_suffix_rows",
        ):
            saved_partition.pop(runtime_key, None)
            current_partition.pop(runtime_key, None)
        saved["mature_handoff_surface_partition"] = saved_partition
        current["mature_handoff_surface_partition"] = current_partition

    if allow_surface_screen_evidence_repair_migration:
        if current.get("surface_screen_topology") != (
            SURFACE_SCREEN_TOPOLOGY_CONTRACT
        ):
            raise RuntimeError(
                "Surface screen-evidence repair target is not the audited "
                "residual-conditioned topology contract"
            )
        if saved.get("surface_screen_topology") is not None:
            raise RuntimeError(
                "Surface screen-evidence repair requires the exact v67 "
                "prefix without a screen-topology contract"
            )
        saved = dict(saved)
        saved["surface_screen_topology"] = dict(
            SURFACE_SCREEN_TOPOLOGY_CONTRACT
        )

    differences = {
        key
        for key in set(saved) | set(current)
        if saved.get(key) != current.get(key)
    }
    if (
        differences == {"camera_intrinsics_contract_sha256"}
        and saved.get("camera_geometry_sha256")
        == current.get("camera_geometry_sha256")
        and saved.get("fixed_camera_validation")
        == current.get("fixed_camera_validation")
    ):
        differences.clear()
    return differences


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim == 2:
        weight = weight[None]
    return (value * weight).sum() / (
        weight.sum() * value.shape[0]
    ).clamp_min(1.0)


def _gradient_norm(parameter: torch.Tensor) -> float:
    gradient = parameter.grad
    if gradient is None:
        return 0.0
    return float(torch.nan_to_num(gradient).norm().detach())


def _adaptive_geometry_scale(
    rgb_gradient_norm: torch.Tensor | float,
    geometry_gradient_norm: torch.Tensor | float,
    *,
    target_ratio: float,
    minimum: float = 0.001,
    maximum: float = 10.0,
) -> float:
    """Return a bounded, detached geometry/RGB gradient calibration.

    Geometry factors are evaluated on a deliberately smaller camera stream
    than RGB. Their raw loss values therefore cannot be compared to the image
    loss, and a fixed scalar silently changes meaning when the number of
    Chart pixels or track observations changes.  Calibrating the surface-xyz
    gradient at the update where both are present makes the intended force
    ratio explicit without allowing the scale itself to become learnable.
    """
    if float(target_ratio) <= 0:
        return 1.0
    rgb = float(torch.as_tensor(rgb_gradient_norm).detach())
    geometry = float(torch.as_tensor(geometry_gradient_norm).detach())
    if not np.isfinite(rgb) or not np.isfinite(geometry):
        return 1.0
    if rgb <= 1e-12 or geometry <= 1e-12:
        return 1.0
    return float(
        np.clip(
            float(target_ratio) * rgb / geometry,
            float(minimum),
            float(maximum),
        )
    )


def _continued_surface_iteration(
    local_iteration: int,
    handoff_iteration: int,
) -> int:
    """Map a mixed-stage step onto the mature surface's LR timeline."""
    local_iteration = int(local_iteration)
    handoff_iteration = int(handoff_iteration)
    if local_iteration <= 0 or handoff_iteration < 0:
        raise ValueError("Invalid surface learning-rate iteration")
    return handoff_iteration + local_iteration


def _teacher_gradient_audit(
    surface,
    foliage,
    package,
    *,
    surface_means2d_gradient_norm: float | None = None,
) -> dict[str, float]:
    if surface_means2d_gradient_norm is None:
        surface_means2d_gradient_norm = (
            0.0
            if package.surface_means2d is None
            or package.surface_means2d.grad is None
            else float(
                torch.nan_to_num(package.surface_means2d.grad).norm()
            )
        )
    return {
        "surface_xyz": _gradient_norm(surface._xyz),
        "surface_feature_dc": _gradient_norm(surface._features_dc),
        "surface_feature_rest": _gradient_norm(surface._features_rest),
        "surface_opacity": _gradient_norm(surface._opacity),
        "surface_scale": _gradient_norm(surface._scaling),
        "surface_rotation": _gradient_norm(surface._rotation),
        "surface_means2d": float(surface_means2d_gradient_norm),
        "foliage_xyz": _gradient_norm(foliage.xyz),
        "foliage_features": _gradient_norm(foliage.features),
        "foliage_opacity": _gradient_norm(foliage.opacity_logits),
    }


def _parameter_loss_gradient_audit(
    foliage,
    losses: dict[str, torch.Tensor],
) -> dict[str, object]:
    """Audit loss ownership by static role without changing optimization.

    A signed parameter gradient is more useful than one global norm here:
    for log-scale and opacity logits, a negative value requests growth while
    a positive value requests retirement.  The audit is sampled only on log
    steps and deliberately reports conflicts instead of imposing a brittle
    pass/fail gate.
    """
    parameters = {
        "xyz": foliage.xyz,
        "scale": foliage.log_scales,
        "mass": foliage.opacity_logits,
        "sh": foliage.features,
    }
    roles = {
        "skeleton": foliage.static_skeleton_mask,
        "envelope": foliage.persistent_envelope_mask,
        "detail": foliage.static_leaf_mask,
    }
    report: dict[str, object] = {
        "contract": "sampled_loss_by_static_role_parameter_signed_gradient",
        "losses": {},
        "warnings": [],
    }
    for loss_name, loss in losses.items():
        loss_report: dict[str, object] = {}
        if not torch.is_tensor(loss) or not loss.requires_grad:
            report["losses"][loss_name] = loss_report
            continue
        gradients = torch.autograd.grad(
            loss,
            tuple(parameters.values()),
            retain_graph=True,
            allow_unused=True,
        )
        for parameter_name, gradient in zip(parameters, gradients):
            if gradient is None:
                continue
            role_report = {}
            clean = torch.nan_to_num(gradient.detach())
            for role_name, mask in roles.items():
                values = clean[mask].reshape(-1)
                if not len(values):
                    continue
                role_report[role_name] = {
                    "l2": float(values.norm()),
                    "signed_sum": float(values.sum()),
                    "increase_entries": int((values < 0).sum()),
                    "decrease_entries": int((values > 0).sum()),
                }
            loss_report[parameter_name] = role_report
        report["losses"][loss_name] = loss_report

    ray_mass = (
        report["losses"].get("ray_free_hit", {})
        .get("mass", {})
    )
    for role_name in ("envelope", "detail"):
        stats = ray_mass.get(role_name, {})
        if (
            stats.get("increase_entries", 0) > 0
            and stats.get("decrease_entries", 0) > 0
        ):
            report["warnings"].append(
                f"ray_factor_has_mixed_mass_directions:{role_name}"
            )
    return report


def _photo_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    lambda_dssim: float,
) -> torch.Tensor:
    l1 = _weighted_mean((prediction - target).abs(), weight)
    structural = _masked_ssim_loss(prediction, target, weight)
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * structural


def _load_dav2_observation_patches(
    surface_seed: Path,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Load topology-independent DAV2 source-view pixel observations."""
    with np.load(surface_seed, allow_pickle=False) as seed:
        required = {
            "source_type",
            "dav2_observation_view_name",
            "dav2_observation_uv",
        }
        if not required.issubset(seed.files):
            return {}
        source = np.asarray(seed["source_type"])
        names = np.asarray(seed["dav2_observation_view_name"]).astype(str)
        uv = np.asarray(
            seed["dav2_observation_uv"], dtype=np.float32
        )
    valid = (
        (source == GaussianModel.SOURCE_DAV2_RIGID_HOLE)
        & (names != "")
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] <= 1)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] <= 1)
    )
    result = {}
    for name in np.unique(names[valid]):
        values = np.ascontiguousarray(uv[valid & (names == name)])
        result[str(Path(name).stem)] = torch.from_numpy(values).to(
            device=device, dtype=torch.float32
        )
    return result


def _dav2_observation_patch_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    uv: torch.Tensor | None,
    *,
    radius: int = 2,
) -> tuple[torch.Tensor, int]:
    """Sample a small real-RGB patch around each posterior observation."""
    if uv is None or not len(uv):
        return prediction.new_zeros(()), 0
    height, width = prediction.shape[-2:]
    offset = torch.arange(
        -int(radius),
        int(radius) + 1,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    dy, dx = torch.meshgrid(offset, offset, indexing="ij")
    base = uv.to(
        device=prediction.device, dtype=prediction.dtype
    )[:, None, None, :]
    patch = torch.stack(
        [
            2.0 * dx / max(float(width), 1.0),
            2.0 * dy / max(float(height), 1.0),
        ],
        dim=-1,
    )
    grid = (2.0 * base - 1.0 + patch[None]).reshape(
        1, -1, 1, 2
    )
    predicted = torch.nn.functional.grid_sample(
        prediction[None],
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    observed = torch.nn.functional.grid_sample(
        target.detach()[None],
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return (predicted - observed).abs().mean(), int(grid.shape[1])


def _branch_isolated_rgb_losses(
    structural_prediction: torch.Tensor,
    mixed_prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    background_weight: torch.Tensor,
    canopy_weight: torch.Tensor,
    rigid_weight: torch.Tensor,
    rigid_high_frequency_weight: torch.Tensor,
    lambda_dssim: float,
) -> dict[str, torch.Tensor]:
    """Route RGB evidence to the branch that can physically explain it.

    The values are intentionally computed in one helper so a later training
    edit cannot silently put rigid pixels back on the mixed prediction.  The
    caller differentiates ``canopy`` only with respect to foliage parameters.
    """
    return {
        "background": _photo_loss(
            structural_prediction,
            target,
            background_weight,
            lambda_dssim,
        ),
        "canopy": _photo_loss(
            mixed_prediction,
            target,
            canopy_weight,
            lambda_dssim,
        ),
        "rigid": _photo_loss(
            structural_prediction,
            target,
            rigid_weight,
            0.20,
        ),
        "rigid_high_frequency": _high_frequency_loss(
            structural_prediction,
            target,
            rigid_high_frequency_weight,
        ),
    }


def _conditioned_photo_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    task: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Separate dynamic-owner pixels from appearance-only context.

    A per-primitive exact-camera gate cannot by itself establish pixel
    ownership: a large dynamic Gaussian can be evidence-owned by the current
    camera while still projecting across a rigid facade or the sky. The
    combined loss remains the appearance objective, but only the canopy loss
    may update 3D foliage base state. This prevents colour fitting from
    hiding an alpha/geometry ownership violation.
    """
    canopy = task["p_canopy"].clamp(0, 1)
    context = (
        task["p_sky"] + 0.15 * task["p_rigid"]
    ).clamp(0, 1)
    return {
        "combined": _photo_loss(
            prediction,
            target,
            (canopy + context).clamp(0, 1),
            0.15,
        ),
        "canopy": _photo_loss(
            prediction,
            target,
            canopy,
            0.15,
        ),
        "context": _photo_loss(
            prediction,
            target,
            context,
            0.15,
        ),
    }


def _masked_ssim_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    window_size: int = 11,
) -> torch.Tensor:
    """SSIM over valid windows, without artificial zero-valued mask edges."""
    if weight.ndim == 2:
        weight = weight[None, None]
    elif weight.ndim == 3:
        weight = weight[None]
    prediction = prediction[None] if prediction.ndim == 3 else prediction
    target = target[None] if target.ndim == 3 else target
    padding = window_size // 2
    mu_prediction = F.avg_pool2d(
        prediction, window_size, stride=1, padding=padding
    )
    mu_target = F.avg_pool2d(
        target, window_size, stride=1, padding=padding
    )
    prediction_variance = F.avg_pool2d(
        prediction.square(), window_size, stride=1, padding=padding
    ) - mu_prediction.square()
    target_variance = F.avg_pool2d(
        target.square(), window_size, stride=1, padding=padding
    ) - mu_target.square()
    covariance = F.avg_pool2d(
        prediction * target, window_size, stride=1, padding=padding
    ) - mu_prediction * mu_target
    c1, c2 = 0.01**2, 0.03**2
    similarity = (
        (2 * mu_prediction * mu_target + c1)
        * (2 * covariance + c2)
        / (
            (mu_prediction.square() + mu_target.square() + c1)
            * (prediction_variance + target_variance + c2)
        ).clamp_min(1e-8)
    ).mean(1, keepdim=True)
    # A minimum pool erodes hard masks and preserves the confidence floor for
    # soft masks. Padding is explicitly invalid so image borders do not gain
    # synthetic support.
    eroded = -F.max_pool2d(
        -weight, window_size, stride=1, padding=padding
    )
    border = torch.ones_like(eroded)
    border[..., :padding, :] = 0
    border[..., -padding:, :] = 0
    border[..., :, :padding] = 0
    border[..., :, -padding:] = 0
    valid = eroded.clamp(0, 1) * border
    if not bool((valid > 0).any()):
        return prediction.new_zeros(())
    return ((1.0 - similarity) * valid).sum() / valid.sum().clamp_min(1)


def _high_frequency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Masked gradient loss for the sequence-conditioned leaf branch."""
    horizontal_weight = torch.minimum(weight[:, 1:], weight[:, :-1])
    vertical_weight = torch.minimum(weight[1:, :], weight[:-1, :])
    horizontal = (
        (prediction[:, :, 1:] - prediction[:, :, :-1])
        - (target[:, :, 1:] - target[:, :, :-1])
    ).abs().mean(0)
    vertical = (
        (prediction[:, 1:, :] - prediction[:, :-1, :])
        - (target[:, 1:, :] - target[:, :-1, :])
    ).abs().mean(0)
    return (
        (horizontal * horizontal_weight).sum()
        / horizontal_weight.sum().clamp_min(1)
        + (vertical * vertical_weight).sum()
        / vertical_weight.sum().clamp_min(1)
    )


def _rigid_residual_patch_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    rigid_weight: torch.Tensor,
    *,
    maximum_patches: int = 64,
    radius: int = 5,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Focus local bandwidth on high-error rigid texture and boundaries.

    Patch selection is detached, while every selected pixel keeps the normal
    RGB/SSIM/gradient derivative.  Ranking combines current residual and
    target image edges, so flat sky/tree regions cannot consume the pool and
    a small window frame is no longer diluted by the full 640x360 image.
    """
    zero = prediction.new_zeros(())
    if maximum_patches <= 0 or radius <= 0:
        return zero, {"patches": 0, "pixels": 0}
    with torch.no_grad():
        residual = (prediction - target).abs().mean(0)
        edge = torch.zeros_like(residual)
        edge[:, 1:] += (
            target[:, :, 1:] - target[:, :, :-1]
        ).abs().mean(0)
        edge[1:, :] += (
            target[:, 1:, :] - target[:, :-1, :]
        ).abs().mean(0)
        support = rigid_weight.clamp(0, 1)
        score = residual * (0.05 + edge) * support
        kernel = 2 * int(radius) + 1
        pooled = F.avg_pool2d(
            score[None, None],
            kernel,
            stride=1,
            padding=int(radius),
        )[0, 0]
        pooled[:radius] = 0
        pooled[-radius:] = 0
        pooled[:, :radius] = 0
        pooled[:, -radius:] = 0
        # A raw top-k over the dense pooled map selects many adjacent centres
        # around one strong edge.  In a real 640x360 Cambridge view the
        # nominal 64 x 11x11 pool consequently covered only ~1.1k pixels,
        # leaving most windows, railings and re-exposed facade holes without
        # any local bandwidth. Select spatially independent local maxima
        # instead. A suppression radius of 2r makes the resulting (2r+1)
        # patches non-redundant without imposing any scene- or loss-valued
        # quality gate. A dense stride-one 21x21 max-pool is unnecessarily
        # expensive for this detached scheduling decision. Partition the
        # raster into non-overlapping suppression cells and retain the exact
        # maximum (and its source index) from each cell. This bounds duplicate
        # centres per local feature while reducing the comparison count by
        # roughly the suppression-cell area.
        suppression_cell = max(2 * kernel - 1, 1)
        cell_score, cell_index = F.max_pool2d(
            pooled[None, None],
            suppression_cell,
            stride=suppression_cell,
            ceil_mode=True,
            return_indices=True,
        )
        cell_score = cell_score.reshape(-1)
        cell_index = cell_index.reshape(-1)
        valid_cell = cell_score > 0
        peak_count = int(valid_cell.sum())
        preliminary_count = min(
            max(int(maximum_patches) * 8, int(maximum_patches)),
            peak_count,
        )
        if preliminary_count <= 0:
            return zero, {"patches": 0, "pixels": 0}
        ranked_cells = torch.where(
            valid_cell, cell_score, torch.zeros_like(cell_score)
        )
        preliminary_cells = torch.topk(
            ranked_cells, k=preliminary_count
        ).indices
        preliminary_indices = (
            cell_index[preliminary_cells].detach().cpu().tolist()
        )
        selected_indices = []
        selected_coordinates = []
        minimum_separation = 2 * int(radius)
        width = int(pooled.shape[1])
        for flat_index in preliminary_indices:
            y, x = divmod(int(flat_index), width)
            if any(
                max(abs(y - other_y), abs(x - other_x))
                <= minimum_separation
                for other_y, other_x in selected_coordinates
            ):
                continue
            selected_indices.append(int(flat_index))
            selected_coordinates.append((y, x))
            if len(selected_indices) >= int(maximum_patches):
                break
        count = len(selected_indices)
        if count <= 0:
            return zero, {"patches": 0, "pixels": 0}
        indices = torch.as_tensor(
            selected_indices, device=pooled.device, dtype=torch.long
        )
        centres = torch.zeros_like(pooled)
        centres.reshape(-1)[indices] = 1
        patch_mask = F.max_pool2d(
            centres[None, None],
            kernel,
            stride=1,
            padding=int(radius),
        )[0, 0]
        patch_weight = patch_mask * support
    pixels = int((patch_weight > 0).sum())
    loss = _photo_loss(
        prediction, target, patch_weight, 0.10
    ) + 0.25 * _high_frequency_loss(
        prediction, target, patch_weight
    )
    return loss, {
        "patches": int(count),
        "pixels": pixels,
        "independent_peak_candidates": peak_count,
        "mean_rank_score": float(pooled.reshape(-1)[indices].mean()),
        "selection": (
            "detached_rigid_residual_times_target_edge_spatial_nms_pool"
        ),
    }


def _boundary_evidence_weight(
    task_or_boundary: dict[str, torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    """Retain continuous evidence at uncertain semantic boundaries.

    Boundary uncertainty should reduce confidence, not erase every ownership,
    free-space, topology and high-frequency gradient at exactly the pixels
    where foliage meets a facade.  A 0.25 floor is deliberately continuous;
    semantic role probabilities still decide which physical branch receives
    the retained evidence.
    """
    boundary = (
        task_or_boundary["p_boundary_uncertain"]
        if isinstance(task_or_boundary, dict)
        else task_or_boundary
    )
    return 1.0 - 0.75 * boundary.clamp(0.0, 1.0)


def _canopy_topology_signal(
    target: torch.Tensor, task: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Return target bandwidth demand for canonical volume densification."""
    low_frequency = F.avg_pool2d(
        target[None], kernel_size=3, stride=1, padding=1
    )[0]
    # ``w_topology`` keeps the conservative canopy-core stream. Add a
    # quarter-strength, distortion-aware boundary stream so a tree/facade
    # edge can request smaller volume footprints instead of remaining a broad
    # splat forever. This is soft evidence, never a birth/prune gate.
    boundary_probability = task.get(
        "p_canopy_boundary", torch.zeros_like(task["w_topology"])
    )
    boundary_topology = (
        0.25
        * boundary_probability
        * task.get("w_rgb", torch.ones_like(boundary_probability))
        * (
            1.0
            - task.get(
                "p_transient", torch.zeros_like(boundary_probability)
            )
        )
        * (
            1.0
            - task.get("p_sky", torch.zeros_like(boundary_probability))
        )
    )
    return (
        (target - low_frequency).abs().mean(0)
        * (task["w_topology"] + boundary_topology).clamp(0.0, 1.0)
    )


def _soft_surface_canopy_conflict(
    package, task: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Penalize only a structural layer visibly in front of claimed foliage.

    A semantic tree silhouette does not mean that every rigid surface on the
    ray is invalid: facade pixels visible through leaf gaps, and buildings
    behind the crown, remain legitimate geometry.  Use the jointly sorted
    branch depths to retire only front-running surfels.  The ownership gate
    is detached so the volume cannot reduce this term merely by changing its
    own alpha or depth.
    """
    surface_alpha = package.surface_alpha[0]
    volume_alpha = package.volume_alpha[0].detach().clamp(0, 1)
    surface_depth = package.surface_depth[0].detach()
    volume_depth = package.volume_depth[0].detach()
    both_valid = (
        (surface_depth > 0)
        & torch.isfinite(surface_depth)
        & (volume_depth > 0)
        & torch.isfinite(volume_depth)
        & (volume_alpha > 1e-4)
    )
    depth_softness = (
        0.04 + 0.015 * volume_depth.clamp_min(0)
    ).clamp(0.04, 0.25)
    surface_in_front = torch.sigmoid(
        (volume_depth - surface_depth - 0.03) / depth_softness
    )
    claimed_front = torch.where(
        both_valid,
        surface_in_front * volume_alpha,
        torch.zeros_like(volume_alpha),
    )
    # A very small unclaimed prior prevents an isolated surface primitive
    # from becoming the permanent tree owner without erasing real gap rays.
    conflict_gate = claimed_front + 0.02 * (1.0 - volume_alpha)
    canopy_probability = task.get("p_canopy", task["p_canopy_core"])
    weight = canopy_probability * _boundary_evidence_weight(task)
    return _weighted_mean(surface_alpha * conflict_gate, weight)


def _counterfactual_volume_transparency_loss(
    mixed_prediction: torch.Tensor,
    surface_prediction: torch.Tensor,
    target: torch.Tensor,
    volume_alpha: torch.Tensor,
    surface_alpha: torch.Tensor,
    task: dict[str, torch.Tensor],
    *,
    margin: float = 0.02,
    temperature: float = 0.015,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Route detached local RGB evidence only to volume transmittance.

    A coarse semantic canopy region cannot say whether an individual pixel
    is an opaque leaf or a view of the rigid surface through a leaf gap.  The
    renderer already produces a surface-only counterfactual for rigid RGB
    ownership.  Compare that image with the authoritative mixed image at the
    same calibrated ray, detach the comparison, and softly assign
    transparency responsibility wherever the existing surface is the better
    explanation *and an independent rigid posterior owns that ray*.  Canopy
    semantics are positive existence evidence and can never authorize their
    own deletion.  Only ``volume_alpha`` remains differentiable, so neither
    surface colour nor geometry can game this evidence.

    The sigmoid is deliberately continuous: ray hit likelihood can retain a
    real leaf when it explains the target, while facade pixels mislabeled by
    a coarse tree mask can still ask the volume to open a gap.
    """
    if mixed_prediction.shape != surface_prediction.shape:
        raise ValueError("Mixed and surface counterfactual RGB shapes differ")
    if mixed_prediction.shape != target.shape:
        raise ValueError("Counterfactual target RGB shape differs")
    if mixed_prediction.ndim != 3 or mixed_prediction.shape[0] != 3:
        raise ValueError("Counterfactual RGB tensors must have shape [3,H,W]")
    mixed_error = (mixed_prediction - target).abs().mean(0).detach()
    surface_error = (surface_prediction - target).abs().mean(0).detach()
    relative_advantage = mixed_error - surface_error
    responsibility = torch.sigmoid(
        (relative_advantage - float(margin)) / float(temperature)
    )
    rigid_permission = task["p_rigid"].clamp(0.0, 1.0)
    canopy_semantic = task.get("p_canopy", task["p_canopy_core"]).clamp(
        0.0, 1.0
    )
    base_weight = (
        responsibility
        * surface_alpha.detach().reshape_as(responsibility).clamp(0.0, 1.0)
        * task.get("w_rgb", torch.ones_like(rigid_permission))
        * (
            1.0
            - task.get(
                "p_transient", torch.zeros_like(rigid_permission)
            )
        )
        * _boundary_evidence_weight(task)
    )
    evidence_weight = (base_weight * rigid_permission).detach()
    # Persist the amount of deletion pressure that the predecessor would have
    # admitted solely because a pixel was labelled canopy.  This makes the
    # causal repair measurable in every normal training log.
    predecessor_permission = (
        rigid_permission + canopy_semantic
    ).clamp(0.0, 1.0)
    blocked_canopy_weight = (
        base_weight
        * (predecessor_permission - rigid_permission).clamp_min(0.0)
    ).detach()
    alpha = volume_alpha.clamp(0.0, 1.0 - 1e-5)
    optical_depth = -torch.log1p(-alpha)
    loss = _weighted_mean(optical_depth, evidence_weight)
    supported = evidence_weight > 1e-4
    return loss, {
        "contract": COUNTERFACTUAL_TRANSPARENCY_CONTRACT,
        "supported_pixels": int(supported.sum()),
        "mean_responsibility": float(
            responsibility[evidence_weight > 0].mean()
        )
        if bool((evidence_weight > 0).any())
        else 0.0,
        "mean_relative_rgb_advantage": float(
            relative_advantage[evidence_weight > 0].mean()
        )
        if bool((evidence_weight > 0).any())
        else 0.0,
        "mean_volume_alpha_on_support": float(
            alpha.detach().reshape_as(responsibility)[supported].mean()
        )
        if bool(supported.any())
        else 0.0,
        "canopy_rgb_retirement_blocked_pixels": int(
            (blocked_canopy_weight > 1e-4).sum()
        ),
        "canopy_rgb_retirement_blocked_mass": float(
            blocked_canopy_weight.sum()
        ),
    }


def _static_detail_global_cleanup_loss(
    mixed_prediction: torch.Tensor,
    surface_prediction: torch.Tensor,
    target: torch.Tensor,
    volume_alpha: torch.Tensor,
    surface_alpha: torch.Tensor,
    task: dict[str, torch.Tensor],
    *,
    counterfactual_weight: float,
    margin: float = 0.02,
    temperature: float = 0.015,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Apply only negative, globally valid evidence to static leaf detail.

    Static detail is visible in every deployment camera.  Its RGB and
    positive hit evidence must remain owned by the internally consistent
    canonical support set, but a primitive that occludes a rigid surface or
    occupies calibrated free space in *any* camera is globally invalid.  The
    former shared ownership gate blocked those contradictions together with
    appearance, leaving cross-sequence tree splats painted over buildings.

    This objective depends on the detail-only rendered alpha.  RGB values are
    detached inside the counterfactual and are used only to decide whether
    the immutable surface is the better explanation.  Consequently it can
    update projected geometry and optical mass, never SH, surface, sky,
    uncertainty or topology.
    """
    if float(counterfactual_weight) < 0:
        raise ValueError("counterfactual_weight cannot be negative")
    counterfactual, counterfactual_audit = (
        _counterfactual_volume_transparency_loss(
            mixed_prediction,
            surface_prediction,
            target,
            volume_alpha,
            surface_alpha,
            task,
            margin=margin,
            temperature=temperature,
        )
    )
    alpha = volume_alpha.reshape_as(task["p_rigid"]).clamp(
        0.0, 1.0 - 1e-5
    )
    rigid_free_weight = (
        (task["p_rigid"] + 0.35 * task["p_sky"]).clamp(0.0, 1.0)
        * task.get("w_rgb", torch.ones_like(task["p_rigid"]))
        * (1.0 - task.get("p_transient", torch.zeros_like(task["p_rigid"])))
        * _boundary_evidence_weight(task)
    ).detach()
    rigid_free = _weighted_mean(
        -torch.log1p(-alpha), rigid_free_weight
    )
    objective = rigid_free + float(counterfactual_weight) * counterfactual
    return objective, {
        "contract": STATIC_DETAIL_GLOBAL_CLEANUP_CONTRACT,
        "rigid_free": float(rigid_free.detach()),
        "rigid_free_supported_pixels": int(
            (rigid_free_weight > 1e-4).sum()
        ),
        "counterfactual_weight": float(counterfactual_weight),
        "counterfactual": counterfactual_audit,
        "gradient_permissions": {
            "static_detail_xyz_scale_rotation": True,
            "static_detail_optical_mass": True,
            "static_detail_sh": False,
            "surface_sky_uncertainty": False,
            "topology_statistics": False,
        },
    }


def _persistent_envelope_global_cleanup_loss(
    mixed_prediction: torch.Tensor,
    surface_prediction: torch.Tensor,
    target: torch.Tensor,
    volume_alpha: torch.Tensor,
    surface_alpha: torch.Tensor,
    task: dict[str, torch.Tensor],
    *,
    counterfactual_weight: float,
    margin: float = 0.02,
    temperature: float = 0.015,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Route globally valid contradictions to the persistent envelope only.

    The envelope is a deployable, unconditional low-frequency volume.  A
    canonical tree mask is therefore insufficient authority to keep it in
    front of a rigid surface that explains the same calibrated ray better.
    Rendering this role in isolation also prevents leaf/detail alpha from
    hiding the envelope's own counterfactual responsibility.

    As with static-detail cleanup, RGB and the immutable surface participate
    only as detached evidence.  This path can shrink, move or reduce envelope
    optical mass, but cannot recolour it or request topology capacity.
    """
    if float(counterfactual_weight) < 0:
        raise ValueError("counterfactual_weight cannot be negative")
    counterfactual, counterfactual_audit = (
        _counterfactual_volume_transparency_loss(
            mixed_prediction,
            surface_prediction,
            target,
            volume_alpha,
            surface_alpha,
            task,
            margin=margin,
            temperature=temperature,
        )
    )
    alpha = volume_alpha.reshape_as(task["p_rigid"]).clamp(
        0.0, 1.0 - 1e-5
    )
    rigid_free_weight = (
        (task["p_rigid"] + 0.35 * task["p_sky"]).clamp(0.0, 1.0)
        * task.get("w_rgb", torch.ones_like(task["p_rigid"]))
        * (1.0 - task.get("p_transient", torch.zeros_like(task["p_rigid"])))
        * _boundary_evidence_weight(task)
    ).detach()
    rigid_free = _weighted_mean(
        -torch.log1p(-alpha), rigid_free_weight
    )
    objective = rigid_free + float(counterfactual_weight) * counterfactual
    return objective, {
        "contract": PERSISTENT_ENVELOPE_GLOBAL_CLEANUP_CONTRACT,
        "rigid_free": float(rigid_free.detach()),
        "rigid_free_supported_pixels": int(
            (rigid_free_weight > 1e-4).sum()
        ),
        "counterfactual_weight": float(counterfactual_weight),
        "counterfactual": counterfactual_audit,
        "gradient_permissions": {
            "persistent_envelope_xyz_scale_rotation": True,
            "persistent_envelope_optical_mass": True,
            "persistent_envelope_sh": False,
            "surface_sky_uncertainty": False,
            "topology_statistics": False,
        },
    }


def _initialize_surface(
    surface: GaussianModel,
    surface_seed: Path,
    scene_extent: float,
    opt,
) -> dict:
    with np.load(surface_seed, allow_pickle=False) as seed:
        xyz = torch.from_numpy(seed["xyz"]).float().cuda()
        scales = torch.from_numpy(seed["scales"]).float().cuda()
        quaternions = torch.from_numpy(
            seed["quaternions"]
        ).float().cuda()
        rgb = torch.from_numpy(seed["rgb"]).float().cuda()
        opacity = torch.from_numpy(
            seed["initial_opacity"]
        ).float().cuda()
        source_type = torch.from_numpy(
            seed["source_type"]
        ).to(device="cuda", dtype=torch.int16)
        track_id = torch.from_numpy(
            seed["track_id"]
        ).to(device="cuda", dtype=torch.int64)
        confidence = torch.from_numpy(
            seed["geometry_confidence"]
        ).float().cuda()
        persistent = torch.from_numpy(
            seed.get(
                "persistent_geometry_evidence",
                (
                    (seed["track_id"] >= 0)
                    | (seed["source_type"] == 2)
                ),
            ).astype(bool)
        ).to(device="cuda", dtype=torch.bool)
    surface.create_from_parameters(
        xyz, scales, quaternions, rgb, scene_extent
    )
    surface._opacity = torch.nn.Parameter(
        torch.logit(opacity[:, None].clamp(1e-6, 1 - 1e-6))
    )
    surface._source_type = source_type
    surface._track_id = track_id
    surface._geometry_confidence = confidence
    surface._primitive_class.fill_(GaussianModel.PRIMITIVE_STRUCTURAL)
    # These are temporary renderer witnesses for the bootstrap only.  The
    # permanent MASt3R/Chart evidence lives in OutdoorGeometryEvidence and is
    # applied to rendered rays/depth.  Keeping this flag forever conflates an
    # observation node with a Gaussian and prevents local replace-and-retire.
    # Protect only independently persistent renderer witnesses.  The former
    # source-type shortcut protected every Chart sample, including explicitly
    # non-persistent coverage hypotheses, for the whole bootstrap window.
    # That made the renderer admission contract ineffective and delayed local
    # prune/replace precisely where evidence was weakest.
    protected = persistent
    surface._protected_flag.copy_(protected)
    surface._block_id.fill_(-1)
    surface.training_setup(opt)
    return {
        "surface_count": len(surface.get_xyz),
        "protected_anchor_count": int(protected.sum()),
        "protection_contract": "temporary_bootstrap_renderer_witness",
        "protection_selection": (
            "persistent_geometry_evidence_only"
        ),
        "track_anchor_count": int((track_id >= 0).sum()),
        "chart_anchor_count": int(
            (source_type == GaussianModel.SOURCE_CHART_RESIDUAL).sum()
        ),
        "source_counts": {
            str(int(value)): int((source_type == value).sum())
            for value in source_type.unique().cpu()
        },
    }


def _validate_surface_warmstart(
    surface_ply: Path,
    handoff_manifest: Path,
) -> dict:
    """Validate a no-history, explicitly scoped rigid-surface provenance."""
    surface_ply = surface_ply.expanduser().resolve()
    handoff_manifest = handoff_manifest.expanduser().resolve()
    if not surface_ply.is_file():
        raise FileNotFoundError(surface_ply)
    if not handoff_manifest.is_file():
        raise FileNotFoundError(handoff_manifest)
    payload = json.loads(handoff_manifest.read_text(encoding="utf-8"))
    if payload.get("protocol") != "native-rigid-surface-handoff-v1":
        raise RuntimeError("Unsupported rigid surface handoff protocol")
    if not bool(payload.get("eligible_for_hybrid_surface_handoff")):
        raise RuntimeError(
            "Rigid surface handoff is not eligible: "
            f"{payload.get('rejection_reasons', [])}"
        )
    expected_path = Path(payload["surface_ply"]).expanduser().resolve()
    if expected_path != surface_ply:
        raise RuntimeError(
            "Rigid surface handoff PLY identity mismatch: "
            f"manifest={expected_path}, requested={surface_ply}"
        )
    actual_sha256 = _file_sha256(surface_ply)
    if payload.get("surface_ply_sha256") != actual_sha256:
        raise RuntimeError("Rigid surface handoff PLY digest mismatch")
    chart_surface_state = payload.get("chart_surface_state")
    if chart_surface_state is not None:
        chart_surface_state = (
            Path(chart_surface_state).expanduser().resolve()
        )
        if not chart_surface_state.is_file():
            raise FileNotFoundError(
                f"Rigid Chart-atlas handoff is missing: "
                f"{chart_surface_state}"
            )
        actual_chart_sha256 = _file_sha256(chart_surface_state)
        if (
            payload.get("chart_surface_state_sha256")
            != actual_chart_sha256
        ):
            raise RuntimeError(
                "Rigid Chart-atlas handoff digest mismatch"
            )
        payload["chart_surface_state"] = str(chart_surface_state)
        payload["chart_surface_state_sha256"] = actual_chart_sha256
    if bool(payload.get("historical_gaussian_input_used", True)):
        raise RuntimeError("Historical Gaussian warm-starts are forbidden")
    colmap_used = bool(payload.get("colmap_points_or_tracks_used", True))
    coverage_only = (
        payload.get("sfm_track_usage_mode") == "coverage_only"
        and payload.get("geometry_authority") == "mast3r_matcha_primary"
    )
    if colmap_used and not coverage_only:
        raise RuntimeError(
            "Rigid surface handoff used COLMAP geometry without the explicit "
            "MASt3R-primary coverage-only contract"
        )
    if (
        payload.get("surface_ownership_contract")
        != "rigid_pixels_tree_sky_transient_excluded"
    ):
        raise RuntimeError(
            "Rigid surface handoff did not exclude tree/sky/transient pixels "
            "from the native surface owner"
        )
    # A mature surface is defined by both its parameters and the position-LR
    # schedule that produced them.  Continuing its absolute iteration on a
    # different schedule family is not continuation: the legacy 24k rigid
    # model used 1.6e-5 -> 1.6e-6 over 20k, while the generic mixed-stage
    # defaults are 1.6e-4 -> 1.6e-6 over 30k.  At the same absolute iteration
    # that silently raises the Cambridge xyz LR by 2.5x and measurably erodes
    # the rigid scaffold.
    surface_optimizer = payload.get("surface_optimizer")
    optimizer_source = "handoff_manifest"
    producer_result: dict | None = None

    def validated_producer_result() -> dict:
        nonlocal producer_result
        if producer_result is not None:
            return producer_result
        producer_result_path = handoff_manifest.parent / "result.json"
        if not producer_result_path.is_file():
            raise RuntimeError(
                "Rigid surface handoff is missing lifecycle fields and has "
                "no validated producer result.json"
            )
        candidate = json.loads(
            producer_result_path.read_text(encoding="utf-8")
        )
        result_surface = Path(
            candidate.get("surface_ply", "")
        ).expanduser().resolve()
        if result_surface != surface_ply:
            raise RuntimeError(
                "Rigid producer result surface identity mismatch"
            )
        if int(candidate.get("iterations", -1)) != int(
            payload.get("surface_iteration", -2)
        ):
            raise RuntimeError(
                "Rigid producer result iteration mismatch"
            )
        producer_result = candidate
        return candidate

    if surface_optimizer is None:
        # Compatibility path for already-produced v1 handoffs.  Their
        # content-addressed PLY and sibling result.json were written by the
        # same rigid run, but the early handoff schema omitted the schedule.
        # Validate the sibling's model identity before accepting its contract.
        surface_optimizer = validated_producer_result().get(
            "training_contract", {}
        ).get("surface_optimizer")
        optimizer_source = "validated_legacy_producer_result"
    required_optimizer_fields = {
        "position_lr_init",
        "position_lr_final",
        "position_lr_delay_mult",
        "position_lr_max_steps",
        "feature_lr",
        "opacity_lr",
        "scaling_lr",
        "rotation_lr",
    }
    if not isinstance(surface_optimizer, dict):
        raise RuntimeError(
            "Rigid surface optimizer contract is missing or malformed"
        )
    missing_optimizer_fields = sorted(
        required_optimizer_fields - set(surface_optimizer)
    )
    if missing_optimizer_fields:
        raise RuntimeError(
            "Rigid surface optimizer contract is incomplete: "
            + ", ".join(missing_optimizer_fields)
        )
    normalized_optimizer = {
        key: (
            int(surface_optimizer[key])
            if key == "position_lr_max_steps"
            else float(surface_optimizer[key])
        )
        for key in sorted(required_optimizer_fields)
    }
    lifecycle_fields = {
        "non_position_lr_decay_from",
        "non_position_lr_decay_until",
        "non_position_lr_final_mult",
    }
    if lifecycle_fields.issubset(surface_optimizer):
        normalized_optimizer.update(
            {
                "non_position_lr_decay_from": int(
                    surface_optimizer["non_position_lr_decay_from"]
                ),
                "non_position_lr_decay_until": int(
                    surface_optimizer["non_position_lr_decay_until"]
                ),
                "non_position_lr_final_mult": float(
                    surface_optimizer["non_position_lr_final_mult"]
                ),
            }
        )
    elif int(payload.get("surface_iteration", 0)) > 0:
        # v68 and earlier handoffs persisted the base LRs but accidentally
        # omitted their final low-LR lifecycle.  Infer it only from a sibling
        # result whose content-addressed PLY and producer iteration match this
        # exact handoff.  This closes the existing v95 16k -> absolute 32k
        # continuation without accepting an arbitrary schedule.
        producer = validated_producer_result()
        producer_contract = producer.get("training_contract", {})
        if (
            producer.get("training_profile") != "hybrid_rigid_stage1"
            or int(producer_contract.get("schedule_horizon", 0)) <= 0
        ):
            raise RuntimeError(
                "Legacy rigid handoff cannot infer a validated polish horizon"
            )
        horizon = int(producer_contract["schedule_horizon"])
        normalized_optimizer.update(
            {
                "non_position_lr_decay_from": int(round(0.75 * horizon)),
                "non_position_lr_decay_until": horizon,
                "non_position_lr_final_mult": 0.10,
            }
        )
        optimizer_source += "_plus_validated_rigid_lifecycle_inference"
    if (
        normalized_optimizer["position_lr_init"] <= 0
        or normalized_optimizer["position_lr_final"] <= 0
        or normalized_optimizer["position_lr_max_steps"] <= 0
    ):
        raise RuntimeError(
            "Rigid surface position-LR schedule must be positive"
        )
    return {
        **payload,
        "surface_ply": str(surface_ply),
        "surface_ply_sha256": actual_sha256,
        "handoff_manifest": str(handoff_manifest),
        "handoff_manifest_sha256": _file_sha256(handoff_manifest),
        "surface_optimizer": normalized_optimizer,
        "surface_optimizer_source": optimizer_source,
    }


def _validate_foliage_rigid_calibration(
    initialization: dict,
    surface_warmstart: dict | None,
) -> dict:
    """Require foliage occlusion calibration and rendered surface to agree.

    Rigid depth removes vegetation hypotheses that lie behind a facade.  If
    foliage is calibrated against one PLY and then trained with another, its
    free/hit posterior and the mixed renderer disagree before iteration one.
    The runner already constructed matching artifacts, but direct trainer
    invocation could silently cross-wire them.
    """
    calibration = initialization.get("foliage", {}).get(
        "rigid_depth_calibration", {}
    )
    if not bool(calibration.get("enabled", False)):
        return {
            "validated": False,
            "reason": "initialization_has_no_trained_rigid_depth_calibration",
        }
    if surface_warmstart is None:
        raise RuntimeError(
            "Rigid-depth-calibrated foliage requires the exact native rigid "
            "surface handoff used for calibration"
        )
    expected = str(calibration.get("structural_ply_sha256", ""))
    if not expected:
        expected = str(
            initialization.get("causal_reuse", {}).get(
                "rigid_calibration_ply_sha256", ""
            )
        )
    actual = str(surface_warmstart["surface_ply_sha256"])
    if not expected:
        raise RuntimeError(
            "Rigid-depth foliage calibration did not persist its structural "
            "PLY digest"
        )
    if expected != actual:
        raise RuntimeError(
            "Foliage rigid-depth calibration PLY differs from the mixed "
            f"surface handoff: calibration={expected}, handoff={actual}"
        )
    return {
        "validated": True,
        "structural_ply_sha256": actual,
        "calibration_source": calibration.get("source"),
        "depth_view_count": int(calibration.get("depth_view_count", 0)),
    }


def _inherit_surface_optimizer_contract(opt, surface_warmstart: dict) -> dict:
    """Make the handoff's complete surface optimizer schedule authoritative."""
    contract = dict(surface_warmstart["surface_optimizer"])
    for key, value in contract.items():
        setattr(opt, key, value)
    if "non_position_lr_decay_until" in contract:
        # GaussianModel uses ``iterations`` as this end point internally.
        opt.iterations = int(contract["non_position_lr_decay_until"])
    return contract


def _evidence_store_uses_colmap_geometry(evidence_store: dict) -> bool:
    """Resolve COLMAP *geometry* provenance without confusing its container.

    The first Evidence Store schema stated the no-COLMAP guarantee through
    ``geometry_source=mast3r_only`` and the camera-container contract, but did
    not persist the later boolean field.  Treating a missing boolean as
    ``True`` made valid legacy MASt3R-only stores fail the rigid-to-mixed
    handoff even though their immutable artifact list contained no COLMAP
    points or tracks.

    Explicit provenance remains authoritative.  For legacy manifests we also
    inspect artifact source types/names so a contradictory manifest fails
    closed instead of being accepted solely because of ``geometry_source``.
    ``COLMAP_world`` coordinates and serialized cameras are deliberately not
    considered COLMAP geometry.
    """
    explicit = evidence_store.get("colmap_points_or_tracks_used")
    if explicit is not None:
        return bool(explicit)
    if evidence_store.get("geometry_source") != "mast3r_only":
        return True
    forbidden_tokens = (
        "colmap_track",
        "colmap_point",
        "points3d",
    )
    for artifact in evidence_store.get("artifacts", []):
        identity = " ".join(
            str(artifact.get(key, "")).lower()
            for key in ("name", "source_type", "measurement")
        )
        if any(token in identity for token in forbidden_tokens):
            return True
    return False


def _write_rigid_stage_surface_handoff(
    *,
    output: Path,
    iteration: int,
    surface_iteration: int | None = None,
    surface_point_count: int,
    initialization: dict,
    initialization_directory: Path,
    evidence_store: dict,
    camera_runtime_contract: dict,
    fixed_camera_validation: dict,
    geometry_audit: dict,
    surface_optimizer: dict,
    chart_surface_state: Path | None = None,
) -> Path:
    """Close the geometry-trained rigid-stage -> mixed-stage contract.

    ``hybrid_rigid_stage1`` is rendered by the same native 2D surfel branch
    as the full teacher, but disables every foliage owner.  Unlike the
    standard RGB-only control, it continues to consume observation-level
    MASt3R tracks and source-raster MAtCha/G4 factors.  Persisting its surface
    through the same handoff protocol lets the mixed stage inherit that
    geometry without inheriting an optimizer, foliage state, or any hidden
    historical/COLMAP Gaussian initialization.
    """
    output = output.resolve()
    surface_ply = (
        output
        / "point_cloud"
        / f"iteration_{int(iteration)}"
        / "point_cloud.ply"
    ).resolve()
    if not surface_ply.is_file():
        raise FileNotFoundError(
            f"Rigid-stage surface PLY was not persisted: {surface_ply}"
        )
    if chart_surface_state is not None:
        chart_surface_state = chart_surface_state.expanduser().resolve()
        if not chart_surface_state.is_file():
            raise FileNotFoundError(
                f"Rigid-stage Chart atlas was not persisted: "
                f"{chart_surface_state}"
            )
    initialization_manifest = (
        initialization_directory / "initialization_manifest.json"
    ).resolve()
    camera_contract_path = (
        output / "camera_intrinsics_contract.json"
    ).resolve()
    surface_audit = initialization.get("surface", {})
    evidence_colmap_geometry_used = _evidence_store_uses_colmap_geometry(
        evidence_store
    )
    rejection_reasons = []
    sfm_coverage = sfm_coverage_tracks_enabled(evidence_store)
    if not mast3r_is_geometry_authority(evidence_store):
        rejection_reasons.append("geometry source is not MASt3R/MAtCha-primary")
    if evidence_colmap_geometry_used and not sfm_coverage:
        rejection_reasons.append(
            "Evidence Store used COLMAP geometry outside coverage-only mode"
        )
    if bool(initialization.get("historical_model_initialization", True)):
        rejection_reasons.append("historical Gaussian initialization was used")
    if bool(surface_audit.get("historical_trained_ply_used", True)):
        rejection_reasons.append("surface seed used a historical trained PLY")
    if (
        bool(surface_audit.get("colmap_points_or_tracks_used", True))
        and not sfm_coverage
    ):
        rejection_reasons.append(
            "surface seed used COLMAP geometry outside coverage-only mode"
        )
    if int(surface_audit.get("canopy_surface_seed_count", -1)) != 0:
        rejection_reasons.append("surface seed contains canopy-owned primitives")
    if int(surface_point_count) <= 0:
        rejection_reasons.append("rigid-stage surface is empty")
    if int(fixed_camera_validation.get("camera_count", 0)) <= 0:
        rejection_reasons.append("fixed calibrated cameras were not validated")
    observation_factors = geometry_audit.get(
        "source_consumption_count",
        # Keep the helper tolerant of the compact synthetic audit used by
        # older tests/tools, while the persisted runtime schema remains the
        # explicit source_consumption_count field.
        geometry_audit.get("consumed", {}),
    )
    if int(observation_factors.get("chart_native_factor", 0)) <= 0:
        rejection_reasons.append("source-raster Chart factor was never consumed")
    if int(observation_factors.get("track_observation_factor", 0)) <= 0:
        rejection_reasons.append("observation-level track factor was never consumed")
    dense_pointmap_expected = int(
        surface_audit.get("source_counts", {}).get(
            "mast3r_dense_rigid", 0
        )
    ) > 0
    if (
        dense_pointmap_expected
        and int(
            observation_factors.get(
                "mast3r_pointmap_native_factor", 0
            )
        )
        <= 0
    ):
        rejection_reasons.append(
            "dense MASt3R pointmap factor was never consumed"
        )

    handoff = {
        "protocol": "native-rigid-surface-handoff-v1",
        "producer": "hybrid_rigid_stage1",
        "eligible_for_hybrid_surface_handoff": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "surface_ply": str(surface_ply),
        "surface_ply_sha256": _file_sha256(surface_ply),
        # ``iteration`` identifies the local point-cloud artifact directory.
        # ``surface_iteration`` is the cumulative optimizer/LR iteration and
        # may be larger when a rigid stage itself continues a prior handoff.
        "surface_iteration": int(
            iteration
            if surface_iteration is None
            else surface_iteration
        ),
        "surface_point_count": int(surface_point_count),
        "input_manifest": str(initialization_manifest),
        "input_manifest_sha256": _file_sha256(initialization_manifest),
        "camera_intrinsics_contract": str(camera_contract_path),
        "camera_intrinsics_contract_sha256": _file_sha256(
            camera_contract_path
        ),
        "camera_geometry_sha256": camera_runtime_contract[
            "camera_geometry_sha256"
        ],
        "historical_gaussian_input_used": bool(
            initialization.get("historical_model_initialization", True)
            or surface_audit.get("historical_trained_ply_used", True)
        ),
        "colmap_points_or_tracks_used": bool(
            evidence_colmap_geometry_used
            or surface_audit.get("colmap_points_or_tracks_used", True)
        ),
        "sfm_track_usage_mode": (
            "coverage_only" if sfm_coverage else "disabled"
        ),
        "geometry_authority": (
            "mast3r_matcha_primary"
            if mast3r_is_geometry_authority(evidence_store)
            else "unsupported"
        ),
        "all_real_rgb_from_iteration_one": True,
        "surface_ownership_contract": (
            "rigid_pixels_tree_sky_transient_excluded"
        ),
        # The mixed stage must inherit this complete schedule before creating
        # its fresh Adam state.  Absolute-iteration offset alone is
        # insufficient when producer and consumer defaults differ.
        "surface_optimizer": {
            key: (
                int(value)
                if key == "position_lr_max_steps"
                else float(value)
            )
            for key, value in surface_optimizer.items()
        },
        "geometry_factor_contract": {
            "chart": "source_resolution_rendered_inverse_depth_observation",
            "track": "observation_level_rendered_surface_factor",
            "mast3r_pointmap": (
                "source_resolution_fixed_camera_rendered_inverse_depth"
            ),
            "chart_native_factor_calls": int(
                observation_factors.get("chart_native_factor", 0)
            ),
            "track_observation_factor_calls": int(
                observation_factors.get("track_observation_factor", 0)
            ),
            "mast3r_pointmap_native_factor_calls": int(
                observation_factors.get(
                    "mast3r_pointmap_native_factor", 0
                )
            ),
        },
        "fixed_camera_validation": fixed_camera_validation,
    }
    if chart_surface_state is not None:
        # The PLY contains a portable, baked 2DGS snapshot. Exact staged
        # continuation additionally needs the continuous inverse-depth atlas
        # and UV-quadtree topology; otherwise child cells become ordinary
        # world-space Gaussians on the next stage and silently lose their
        # calibrated ray ownership.
        handoff.update(
            {
                "chart_surface_state": str(chart_surface_state),
                "chart_surface_state_sha256": _file_sha256(
                    chart_surface_state
                ),
                "chart_surface_handoff_contract": (
                    "continuous_inverse_depth_atlas_and_uv_quadtree"
                ),
            }
        )
    destination = output / "rigid_surface_handoff.json"
    destination.write_text(
        json.dumps(handoff, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def _rigid_completion_seed_indices(
    existing_xyz: np.ndarray,
    seed: dict[str, np.ndarray],
    maximum_seeds: int,
) -> tuple[np.ndarray, dict]:
    """Select calibrated coverage hypotheses missing from a rigid handoff.

    The current Evidence Store may contain later complementary fixed-camera
    MASt3R pointmaps than the rigid producer. Merely constructing
    ``ChartSurfaceModel`` from those rows does not create a renderer primitive
    for a genuinely empty region. This selector turns only cross-sequence
    pointmaps or persistent track/Chart observations into low-opacity births.
    Admission is ranked continuously by measured footprint deficit and
    spatial precision rather than by a PSNR or confidence pass/fail gate.
    """
    maximum_seeds = int(maximum_seeds)
    if maximum_seeds <= 0:
        return np.empty(0, dtype=np.int64), {
            "enabled": False,
            "reason": "maximum_seed_budget_is_zero",
            "selected": 0,
        }
    required = {
        "xyz",
        "scales",
        "geometry_confidence",
        "position_sigma",
        "pointmap_cross_sequence_supported",
        "persistent_geometry_evidence",
    }
    missing = sorted(required - set(seed))
    if missing:
        raise RuntimeError(
            f"Surface completion seed is missing fields: {missing}"
        )
    xyz = np.asarray(seed["xyz"], dtype=np.float32)
    existing_xyz = np.asarray(existing_xyz, dtype=np.float32)
    if existing_xyz.ndim != 2 or existing_xyz.shape[1] != 3:
        raise ValueError("Existing rigid surface must have shape [N,3]")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Surface completion seed xyz must have shape [N,3]")
    if not len(xyz) or not len(existing_xyz):
        return np.empty(0, dtype=np.int64), {
            "enabled": True,
            "reason": "empty_seed_or_existing_surface",
            "selected": 0,
        }
    from scipy.spatial import cKDTree

    nearest_distance, _ = cKDTree(existing_xyz).query(
        xyz, k=1, workers=-1
    )
    tangent_radius = np.maximum(
        np.asarray(seed["scales"], dtype=np.float32).max(axis=1),
        0.03,
    )
    deficit = np.maximum(nearest_distance - tangent_radius, 0.0)
    confidence = np.nan_to_num(
        np.asarray(seed["geometry_confidence"], dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clip(min=0.0)
    sigma = np.nan_to_num(
        np.asarray(seed["position_sigma"], dtype=np.float32),
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    ).clip(min=1e-4)
    supported = (
        np.asarray(
            seed["pointmap_cross_sequence_supported"], dtype=bool
        )
        | np.asarray(seed["persistent_geometry_evidence"], dtype=bool)
    )
    source_type = np.asarray(
        seed.get(
            "source_type",
            np.full(len(xyz), -1, dtype=np.int8),
        )
    ).reshape(-1)
    if len(source_type) != len(xyz):
        raise RuntimeError("Surface completion source_type is misaligned")
    # DAV2 is still not metric authority.  These rows were already aligned
    # to the fixed-camera MASt3R/MAtCha scaffold and proposed only in a rigid
    # coverage hole.  They are therefore valid low-opacity *hypotheses* for
    # the mutable residual suffix, where RGB/multiview responsibility can
    # validate or retire them; they can never alter the frozen prefix.
    weak_dav2_hole = (
        source_type == GaussianModel.SOURCE_DAV2_RIGID_HOLE
    )
    precision = (
        confidence / (confidence + 0.25)
    ) / (1.0 + np.square(sigma / 0.12))
    normalized_deficit = np.clip(
        deficit / (tangent_radius + sigma + 0.02),
        0.0,
        1.0,
    )
    score = precision * normalized_deficit
    eligible = (
        (supported | weak_dav2_hole)
        & (deficit > 0)
        & np.isfinite(score)
        & np.isfinite(xyz).all(axis=1)
    )
    indices = np.flatnonzero(eligible)
    if len(indices) > maximum_seeds:
        order = np.argsort(-score[indices], kind="stable")
        indices = indices[order[:maximum_seeds]]
    indices = indices.astype(np.int64, copy=False)
    selected_distance = nearest_distance[indices]
    selected_score = score[indices]
    return indices, {
        "enabled": True,
        "selection_contract": (
            "metric_cross_sequence_or_persistent_evidence_or_calibrated_"
            "dav2_rigid_hole_hypothesis__continuous_existing_footprint_"
            "deficit_times_precision"
        ),
        "candidate_count": int(eligible.sum()),
        "selected": int(len(indices)),
        "maximum_seeds": maximum_seeds,
        "nearest_distance_median": (
            float(np.median(selected_distance)) if len(indices) else None
        ),
        "nearest_distance_p90": (
            float(np.quantile(selected_distance, 0.90))
            if len(indices)
            else None
        ),
        "selection_score_median": (
            float(np.median(selected_score)) if len(indices) else None
        ),
        "cross_sequence_pointmap_selected": int(
            np.asarray(
                seed["pointmap_cross_sequence_supported"], dtype=bool
            )[indices].sum()
        ),
        "persistent_geometry_selected": int(
            np.asarray(
                seed["persistent_geometry_evidence"], dtype=bool
            )[indices].sum()
        ),
        "weak_dav2_hole_selected": int(
            weak_dav2_hole[indices].sum()
        ),
        "birth_authority": (
            "metric_cross_sequence_or_persistent_evidence_plus_"
            "mast3r_calibrated_low_opacity_dav2_rigid_hole_hypothesis"
        ),
    }


def _initialize_surface_from_ply(
    surface: GaussianModel,
    surface_ply: Path,
    surface_seed: Path,
    scene_extent: float,
    opt,
    *,
    maximum_completion_seeds: int,
) -> dict:
    """Continue a mature native rigid 2DGS as the mixed surface branch."""
    surface.load_ply(str(surface_ply))
    if not len(surface.get_xyz):
        raise RuntimeError("Rigid surface handoff is empty")
    if surface.get_scaling.ndim != 2 or surface.get_scaling.shape[1] != 2:
        raise RuntimeError(
            "Hybrid structural branch requires native 2D surfel tangent scales"
        )
    surface.spatial_lr_scale = float(scene_extent)
    surface.max_radii2D = torch.zeros(
        len(surface.get_xyz),
        dtype=surface.get_xyz.dtype,
        device=surface.get_xyz.device,
    )
    surface._primitive_class.fill_(GaussianModel.PRIMITIVE_STRUCTURAL)
    # Permanent MASt3R/MAtCha observations remain in the Evidence Store; the
    # handoff must not turn their old renderer witnesses into immortal rows.
    surface._protected_flag.zero_()
    existing_count = int(len(surface.get_xyz))
    with np.load(surface_seed, allow_pickle=False) as archive:
        seed = {name: archive[name] for name in archive.files}
    completion_indices, completion_audit = (
        _rigid_completion_seed_indices(
            surface.get_xyz.detach().cpu().numpy(),
            seed,
            maximum_completion_seeds,
        )
    )
    if len(completion_indices):
        index = completion_indices
        appended = surface.append_from_parameters(
            torch.from_numpy(np.ascontiguousarray(seed["xyz"][index])),
            torch.from_numpy(
                np.ascontiguousarray(seed["scales"][index])
            ),
            torch.from_numpy(
                np.ascontiguousarray(seed["quaternions"][index])
            ),
            torch.from_numpy(np.ascontiguousarray(seed["rgb"][index])),
            initial_opacity=torch.from_numpy(
                np.ascontiguousarray(seed["initial_opacity"][index])
            ),
            source_type=torch.from_numpy(
                np.ascontiguousarray(seed["source_type"][index])
            ),
            track_id=torch.from_numpy(
                np.ascontiguousarray(seed["track_id"][index])
            ),
            geometry_confidence=torch.from_numpy(
                np.ascontiguousarray(
                    seed["geometry_confidence"][index]
                )
            ),
            # Completion births remain stable only through the short
            # bootstrap. Permanent observations live outside renderer
            # topology and survive after these flags are released.
            protected_flag=True,
        )
        if int(appended) != len(completion_indices):
            raise RuntimeError("Rigid completion append count mismatch")
    completion_audit["existing_surface_count"] = existing_count
    completion_audit["completed_surface_count"] = int(len(surface.get_xyz))
    surface.training_setup(opt)
    source_type = surface._source_type
    track_id = surface._track_id
    return {
        "surface_count": len(surface.get_xyz),
        "protected_anchor_count": int(surface._protected_flag.sum()),
        "protection_contract": "external_evidence_no_immutable_renderer_seed",
        "track_anchor_count": int((track_id >= 0).sum()),
        "chart_anchor_count": int(
            (source_type == GaussianModel.SOURCE_CHART_RESIDUAL).sum()
        ),
        "source_counts": {
            str(int(value)): int((source_type == value).sum())
            for value in source_type.unique().cpu()
        },
        "initialization": "native_rigid_surface_handoff",
        "higher_order_sh_preserved": True,
        "opacity_and_surface_parameters_preserved": True,
        "optimizer_state_preserved": False,
        "rigid_background_completion": completion_audit,
    }


def _trainable_parameters(module):
    """Return only parameters that may legally be explicit backward inputs.

    Compatibility tensors can remain registered in a module so historical
    checkpoints load exactly while the current image-formation contract keeps
    them frozen.  Passing those tensors through ``autograd.backward(inputs=)``
    asks PyTorch to retain gradients on ``requires_grad=False`` leaves and
    fails exactly when the conditioned branch first becomes active.
    """

    return tuple(
        parameter
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _volume_optimizer(args, foliage, appearance, sky):
    groups = [
            {
                "params": [foliage.xyz],
                "lr": args.volume_position_lr,
                "name": "xyz",
            },
            {
                "params": [foliage.features],
                "lr": args.volume_feature_lr,
                "name": "features",
            },
            {
                "params": [foliage.opacity_logits],
                "lr": args.volume_opacity_lr,
                "name": "opacity",
            },
            {
                "params": [foliage.log_scales],
                "lr": args.volume_scale_lr,
                "name": "scale",
            },
            {
                "params": [foliage.quaternions],
                "lr": args.volume_rotation_lr,
                "name": "rotation",
            },
            {
                "params": list(_trainable_parameters(appearance)),
                "lr": args.appearance_lr,
                "name": "appearance",
            },
            {
                "params": list(sky.parameters()),
                "lr": args.sky_lr,
                "name": "sky",
            },
        ]
    # A static production map has no temporal deformation/appearance/opacity
    # owner.  Keeping three N x rank compatibility tensors in Adam used about
    # a quarter gigabyte at 2M rows before optimizer moments, despite every
    # corresponding gradient being discarded.  Legacy conditioned models
    # retain the exact historical group and checkpoint layout.
    if foliage.dynamic_rank > 0:
        groups.insert(
            5,
            {
                "params": [
                    foliage.deformation_basis,
                    foliage.dynamic_feature_basis,
                    foliage.dynamic_opacity_basis,
                ],
                "lr": args.dynamic_lr,
                "name": "dynamic",
            },
        )
    optimizer = torch.optim.Adam(groups, eps=1e-15)
    for group in optimizer.param_groups:
        group["base_lr"] = float(group["lr"])
    return optimizer


def _restore_volume_optimizer_state(
    optimizer,
    state_dict,
    *,
    allow_static_uncertainty_extension: bool = False,
) -> bool:
    """Restore Adam, appending only the new v5 uncertainty decoder.

    The v5 spatial robust-scale decoder is registered after every v4
    appearance parameter.  Therefore its optimizer migration is exact: all
    existing ids and moments keep their ordering and one state-free parameter
    is appended to the appearance group.  No foliage, sky or geometry state
    is reinitialized.
    """

    try:
        optimizer.load_state_dict(state_dict)
        return False
    except ValueError:
        if not allow_static_uncertainty_extension:
            raise
    migrated = copy.deepcopy(state_dict)
    saved_groups = migrated.get("param_groups", [])
    current_groups = optimizer.param_groups
    saved_by_name = {group.get("name"): group for group in saved_groups}
    current_by_name = {group.get("name"): group for group in current_groups}
    saved_appearance = saved_by_name.get("appearance")
    current_appearance = current_by_name.get("appearance")
    if (
        saved_appearance is None
        or current_appearance is None
        or len(current_appearance["params"])
        != len(saved_appearance["params"]) + 1
        or any(
            len(current_by_name[name]["params"])
            != len(group["params"])
            for name, group in saved_by_name.items()
            if name != "appearance" and name in current_by_name
        )
    ):
        raise RuntimeError(
            "Static uncertainty repair encountered a non-appearance Adam "
            "layout change"
        )
    existing_ids = [
        int(parameter_id)
        for group in saved_groups
        for parameter_id in group["params"]
    ]
    appended_id = max(existing_ids, default=-1) + 1
    saved_appearance["params"].append(appended_id)
    optimizer.load_state_dict(migrated)
    return True


def _update_volume_learning_rates(
    optimizer,
    step: int,
    args,
    phase: str,
) -> dict[str, object]:
    """Apply a topology-free appearance convergence suffix.

    Static canonical polish already freezes xyz/covariance/opacity.  Keeping
    SH, spatial appearance and sky at their topology-stage LR made the final
    suffix oscillate instead of reproducing the stable low-LR finish of the
    historical Cambridge control.  Decay only those remaining image-formation
    groups and keep every earlier iteration at its exact configured base LR.
    """
    target_groups = {"features", "appearance", "sky", "dynamic"}
    multiplier = 1.0
    progress = 0.0
    polish_start = int(args.phase_schedule_horizon)
    schedule = _resolved_phase_schedule(
        args.training_profile, args.phase_schedule_horizon
    )
    previous_end = 0
    for name, endpoint in schedule:
        if name == "canonical_polish":
            polish_start = previous_end + 1
            break
        previous_end = endpoint
    if str(phase) == "canonical_polish":
        span = max(int(args.phase_schedule_horizon) - polish_start + 1, 1)
        progress = float(
            np.clip(((int(step) + 1) - polish_start + 1) / span, 0.0, 1.0)
        )
        final = max(float(args.volume_polish_final_lr_multiplier), 1.0e-8)
        multiplier = final**progress
    resolved: dict[str, float] = {}
    for group in optimizer.param_groups:
        base = float(group.setdefault("base_lr", group["lr"]))
        name = str(group.get("name", ""))
        group["lr"] = base * multiplier if name in target_groups else base
        resolved[name] = float(group["lr"])
    return {
        "contract": (
            "static_geometry_mass_frozen__remaining_sh_appearance_sky_"
            "exponential_low_lr_polish"
        ),
        "phase": str(phase),
        "polish_start_iteration": polish_start,
        "progress": progress,
        "multiplier": multiplier,
        "learning_rates": resolved,
    }


def _surface_non_position_lr_multiplier(iteration: int, opt) -> float:
    """Mirror the native 2DGS non-position suffix for auxiliary Chart state."""
    decay_from = int(opt.non_position_lr_decay_from)
    if decay_from < 0:
        return 1.0
    decay_until = max(int(opt.iterations), 1)
    span = max(decay_until - decay_from, 1)
    progress = float(
        np.clip((int(iteration) - decay_from) / span, 0.0, 1.0)
    )
    final = max(float(opt.non_position_lr_final_mult), 1.0e-8)
    return float(final**progress)


def _update_chart_atlas_learning_rate(
    optimizer, iteration: int, opt
) -> dict[str, float | int | str]:
    """Keep continuous Chart geometry in the same rigid polish lifecycle.

    The Chart inverse-depth atlas owns real surface geometry, so leaving it at
    a fixed LR while every baked surfel parameter enters a low-LR suffix is a
    hidden second geometry schedule.  Apply the producer's absolute-iteration
    non-position multiplier to its configured base LR as well.
    """
    multiplier = _surface_non_position_lr_multiplier(iteration, opt)
    learning_rates = []
    for group in optimizer.param_groups:
        base = float(group.setdefault("base_lr", group["lr"]))
        group["lr"] = base * multiplier
        learning_rates.append(float(group["lr"]))
    return {
        "contract": "chart_inverse_depth_follows_rigid_non_position_polish",
        "surface_iteration": int(iteration),
        "multiplier": float(multiplier),
        "minimum_learning_rate": min(learning_rates, default=0.0),
        "maximum_learning_rate": max(learning_rates, default=0.0),
    }


def _migrate_volume_optimizer(
    args,
    foliage,
    appearance,
    sky,
    previous,
    new_to_old,
):
    """Move Adam state to a replaced topology with bounded transient memory.

    ``previous`` is dead after this call.  Pop one parameter state at a time
    and release its old parameter reference as soon as the mapped state is
    installed.  Keeping the complete old optimizer alive while accumulating
    a complete new one doubled Adam residency at the 2M topology boundary.
    """
    current = _volume_optimizer(args, foliage, appearance, sky)
    previous_groups = {
        group["name"]: group for group in previous.param_groups
    }
    primitive = {
        "xyz",
        "features",
        "opacity",
        "scale",
        "rotation",
        "dynamic",
    }
    for group in current.param_groups:
        old = previous_groups.get(group["name"])
        if old is None or len(old["params"]) != len(group["params"]):
            continue
        for parameter_index, (old_parameter, new_parameter) in enumerate(
            zip(old["params"], group["params"])
        ):
            state = previous.state.pop(old_parameter, None)
            if not state:
                old["params"][parameter_index] = new_parameter
                continue
            if old_parameter is new_parameter:
                # Appearance and sky parameters do not change identity; the
                # state object can be transferred without any tensor copy.
                current.state[new_parameter] = state
                old["params"][parameter_index] = new_parameter
                continue
            migrated = {}
            for key in list(state):
                value = state.pop(key)
                if (
                    group["name"] in primitive
                    and torch.is_tensor(value)
                    and value.ndim > 0
                    and value.shape[0] == old_parameter.shape[0]
                ):
                    # Advanced indexing already creates independent storage;
                    # cloning it once more only doubles the largest transient.
                    migrated[key] = value[new_to_old]
                elif torch.is_tensor(value):
                    migrated[key] = value
                else:
                    migrated[key] = value
                del value
            current.state[new_parameter] = migrated
            old["params"][parameter_index] = new_parameter
            del state
    return current


@torch.no_grad()
def _zero_new_volume_optimizer_rows(optimizer, start: int) -> None:
    """Give newborn static rays a clean Adam state, not a parent's momentum."""
    start = int(start)
    for group in optimizer.param_groups:
        if group.get("name") not in {
            "xyz",
            "features",
            "opacity",
            "scale",
            "rotation",
            "dynamic",
        }:
            continue
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            for value in state.values():
                if (
                    torch.is_tensor(value)
                    and value.ndim > 0
                    and value.shape[0] == parameter.shape[0]
                ):
                    value[start:] = 0


@torch.no_grad()
def _zero_volume_optimizer_rows(
    optimizer, rows: torch.Tensor
) -> None:
    """Clear inherited Adam momentum for locally reconstructed rows."""
    rows = torch.as_tensor(rows, dtype=torch.long).reshape(-1)
    if not len(rows):
        return
    for group in optimizer.param_groups:
        if group.get("name") not in {
            "xyz",
            "features",
            "opacity",
            "scale",
            "rotation",
            "dynamic",
        }:
            continue
        for parameter in group["params"]:
            local_rows = rows.to(parameter.device)
            state = optimizer.state.get(parameter, {})
            for value in state.values():
                if (
                    torch.is_tensor(value)
                    and value.ndim > 0
                    and value.shape[0] == parameter.shape[0]
                ):
                    value[local_rows] = 0


@torch.no_grad()
def _zero_volume_opacity_optimizer_rows(
    optimizer, rows: torch.Tensor
) -> None:
    """Clear only Adam mass momentum after an exact optical repartition."""
    rows = torch.as_tensor(rows, dtype=torch.long).reshape(-1)
    if not len(rows):
        return
    for group in optimizer.param_groups:
        if group.get("name") != "opacity":
            continue
        for parameter in group["params"]:
            local_rows = rows.to(parameter.device)
            state = optimizer.state.get(parameter, {})
            for value in state.values():
                if (
                    torch.is_tensor(value)
                    and value.ndim > 0
                    and value.shape[0] == parameter.shape[0]
                ):
                    value[local_rows] = 0


def _volume_stats(foliage) -> dict[str, torch.Tensor]:
    count = len(foliage)
    device = foliage.xyz.device
    return {
        "gradient": torch.zeros(count, device=device),
        "gradient_count": torch.zeros(count, device=device),
        "radius": torch.zeros(count, device=device),
        # Dynamic topology must be driven by the exact-owner conditioned
        # render, not by whichever 2,048 observation rows happened to be
        # sampled by the auxiliary factor in the current topology window.
        # Keep this channel separate from the canonical render: dynamic
        # opacities are disabled there, but their projected radii are still
        # defined and would otherwise look like valid bandwidth evidence.
        "conditioned_gradient": torch.zeros(count, device=device),
        "conditioned_gradient_count": torch.zeros(count, device=device),
        "conditioned_radius": torch.zeros(count, device=device),
        # Camera id that supplied the strongest exact-owner screen-space
        # gradient in the current topology epoch.  This is transient
        # selection provenance, not learned model state.
        "conditioned_context_id": torch.full(
            (count,), -1, dtype=torch.int32, device=device
        ),
        "conditioned_context_score": torch.zeros(count, device=device),
        "contribution": torch.zeros(count, device=device),
        "residual": torch.zeros(count, device=device),
        "rigid": torch.zeros(count, device=device),
        "observation_gradient": torch.zeros(count, device=device),
        "observation_count": torch.zeros(count, device=device),
    }


def _extend_volume_stats_for_births(
    stats: dict[str, torch.Tensor], new_count: int
) -> dict[str, torch.Tensor]:
    """Append neutral statistics without erasing the measured old rows.

    Ray-driven births are allocated before ordinary splitting.  Resetting the
    whole statistics table after that append would discard the screen-space
    evidence that is supposed to rank existing split candidates.  Preserve
    the prefix exactly and give newborn rows a neutral first topology epoch;
    their explicit verification grace protects them until real cameras have
    observed them.
    """
    if not stats:
        raise ValueError("volume statistics cannot be empty")
    old_count = len(next(iter(stats.values())))
    new_count = int(new_count)
    if new_count < old_count:
        raise ValueError("ray birth cannot shrink volume statistics")
    if new_count == old_count:
        return stats
    extension = new_count - old_count
    result = {}
    for name, value in stats.items():
        if len(value) != old_count:
            raise ValueError("volume statistic tables have unequal lengths")
        fill_value = -1 if name == "conditioned_context_id" else 0
        suffix = torch.full(
            (extension,),
            fill_value,
            device=value.device,
            dtype=value.dtype,
        )
        result[name] = torch.cat([value, suffix], dim=0)
    return result


@torch.no_grad()
def _accumulate_volume_stats(
    stats,
    package,
    ownership_gate: torch.Tensor | None = None,
    *,
    exact_conditioned_dynamic: bool = False,
    owner_context_id: int | None = None,
) -> None:
    offset = package.structural_count
    radii = package.radii[offset:].float()
    if ownership_gate is None:
        ownership = torch.ones_like(radii)
    else:
        ownership = torch.as_tensor(
            ownership_gate, device=radii.device, dtype=radii.dtype
        ).reshape(-1)
        if ownership.shape != radii.shape:
            raise ValueError(
                "volume-stat ownership gate must match the volume count"
            )
        ownership = ownership.clamp(0.0, 1.0)
    stats["radius"] = torch.maximum(
        stats["radius"], radii * ownership
    )
    if exact_conditioned_dynamic:
        stats["conditioned_radius"] = torch.maximum(
            stats["conditioned_radius"], radii * ownership
        )
    if (
        package.volume_means2d is not None
        and package.volume_means2d.grad is not None
    ):
        gradient = package.volume_means2d.grad[:, :2].norm(dim=-1)
        visible = (radii > 0) & (ownership > 0)
        stats["gradient"][visible] += (
            gradient[visible] * ownership[visible]
        )
        stats["gradient_count"][visible] += ownership[visible]
        if exact_conditioned_dynamic:
            stats["conditioned_gradient"][visible] += (
                gradient[visible] * ownership[visible]
            )
            stats["conditioned_gradient_count"][visible] += ownership[
                visible
            ]
            if owner_context_id is not None:
                context_score = gradient * ownership
                stronger = visible & (
                    context_score > stats["conditioned_context_score"]
                )
                stats["conditioned_context_score"][stronger] = context_score[
                    stronger
                ]
                stats["conditioned_context_id"][stronger] = int(
                    owner_context_id
                )
    if package.responsibility is not None:
        volume = package.responsibility[offset:]
        stats["contribution"] += volume[:, 0] * ownership
        if volume.shape[1] >= 4:
            stats["rigid"] += volume[:, 2] * ownership
            stats["residual"] += volume[:, 3] * ownership


@torch.no_grad()
def _accumulate_dynamic_observation_stats(
    stats, foliage, observed: torch.Tensor
) -> None:
    if foliage.xyz.grad is None or not bool(observed.any()):
        return
    gradient = torch.nan_to_num(foliage.xyz.grad).norm(dim=1)
    stats["observation_gradient"][observed] += gradient[observed]
    stats["observation_count"][observed] += 1


def _backward_conditioned_foliage(
    loss: torch.Tensor,
    foliage,
    volume_means2d: torch.Tensor | None,
) -> None:
    """Backpropagate leaf ownership and preserve its topology signal.

    ``volume_means2d`` is an independent leaf created by the rasterizer.  A
    restricted ``backward(inputs=foliage.parameters())`` therefore updates
    the physical foliage parameters but deliberately skips this screen-space
    leaf.  That made conditioned RGB visibly train the dynamic branch while
    every adaptive-topology window reported zero conditioned gradients.
    Include the rasterizer leaf explicitly so the subsequent statistics pass
    sees the exact-owner image-bandwidth residual.
    """
    inputs = tuple(foliage.parameters())
    if volume_means2d is not None and volume_means2d.requires_grad:
        inputs = (*inputs, volume_means2d)
    torch.autograd.backward(loss, inputs=inputs, retain_graph=True)


def _confirmed_contradiction_fraction(
    positive_support: torch.Tensor,
    confirmed_free: torch.Tensor,
) -> torch.Tensor:
    """Return the negative fraction among confirmed binary observations."""
    positive_support = positive_support.float()
    confirmed_free = confirmed_free.float()
    return confirmed_free / (
        positive_support + confirmed_free
    ).clamp_min(1)


def _waterfill_group_allocations(
    counts: torch.Tensor,
    group_score: torch.Tensor,
    quota: int,
) -> torch.Tensor:
    """Deterministically water-fill a finite quota over non-empty groups."""
    allocations = torch.zeros_like(counts)
    remaining = min(max(int(quota), 0), int(counts.sum()))
    while remaining:
        active = torch.nonzero(
            allocations < counts, as_tuple=False
        ).flatten()
        if not len(active):
            break
        if remaining < len(active):
            chosen = active[
                torch.topk(group_score[active], remaining, sorted=True).indices
            ]
            allocations[chosen] += 1
            break
        level = max(remaining // len(active), 1)
        addition = torch.minimum(
            counts[active] - allocations[active],
            torch.full_like(allocations[active], level),
        )
        assigned = int(addition.sum())
        allocations[active] += addition
        remaining -= assigned
        if assigned == 0:
            break
    return allocations


def _within_group_rank(
    group: torch.Tensor,
    score: torch.Tensor,
) -> torch.Tensor:
    """Return zero-based descending-score rank without per-group loops."""
    score_order = torch.argsort(score, descending=True, stable=True)
    grouped_order = score_order[
        torch.argsort(group[score_order], stable=True)
    ]
    sorted_group = group[grouped_order]
    _, counts = torch.unique_consecutive(
        sorted_group, return_counts=True
    )
    starts = torch.cumsum(counts, dim=0) - counts
    sorted_rank = torch.arange(
        len(group), device=group.device, dtype=torch.long
    ) - torch.repeat_interleave(starts, counts)
    rank = torch.empty_like(sorted_rank)
    rank[grouped_order] = sorted_rank
    return rank


def _variable_group_topk(
    priority: torch.Tensor,
    group: torch.Tensor,
    allocations: torch.Tensor,
) -> torch.Tensor:
    """Select each group's requested top-k rows in one stable sort."""
    priority_order = torch.argsort(priority, descending=True, stable=True)
    grouped_order = priority_order[
        torch.argsort(group[priority_order], stable=True)
    ]
    sorted_group = group[grouped_order]
    _, counts = torch.unique_consecutive(
        sorted_group, return_counts=True
    )
    starts = torch.cumsum(counts, dim=0) - counts
    within = torch.arange(
        len(group), device=group.device, dtype=torch.long
    ) - torch.repeat_interleave(starts, counts)
    selected = within < allocations[sorted_group]
    return grouped_order[selected]


def _balanced_instance_topk(
    indices: torch.Tensor,
    score: torch.Tensor,
    tree_instance_id: torch.Tensor,
    quota: int,
    *,
    xyz: torch.Tensor | None = None,
    spatial_cell_size: float = 1.0,
    context_id: torch.Tensor | None = None,
    lineage_family_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select high-score candidates with vectorized hierarchical fairness.

    Context capacity is water-filled first.  Inside each context, a stable
    rank over (lineage, tree instance, spatial cell) gives every physical
    group one candidate before taking its second, and so on.  The predecessor
    implemented the same policy with one full candidate scan per group,
    becoming effectively O(GN) at outdoor scale.  This implementation uses a
    constant number of GPU sorts/scatters and is O(N log N).
    """
    quota = min(max(int(quota), 0), int(len(indices)))
    if quota == 0:
        return indices[:0]
    if quota == len(indices):
        return indices
    count = len(tree_instance_id)
    candidate_score = torch.nan_to_num(
        score[indices], nan=-1.0e30, neginf=-1.0e30, posinf=1.0e30
    )
    instances = tree_instance_id[indices].to(torch.int64)
    if context_id is None:
        candidate_context = torch.zeros_like(instances)
    else:
        context_id = torch.as_tensor(
            context_id, device=indices.device, dtype=torch.int64
        ).reshape(-1)
        if len(context_id) != count:
            raise ValueError(
                "context_id must have one value per topology row"
            )
        candidate_context = context_id[indices]
    composite_parts = [candidate_context[:, None]]
    if lineage_family_id is not None:
        lineage_family_id = torch.as_tensor(
            lineage_family_id,
            device=indices.device,
            dtype=torch.int64,
        ).reshape(-1)
        if len(lineage_family_id) != count:
            raise ValueError(
                "lineage_family_id must have one value per topology row"
            )
        composite_parts.append(lineage_family_id[indices, None])
    composite_parts.append(instances[:, None])
    if xyz is not None:
        cells = torch.floor(
            xyz[indices] / max(float(spatial_cell_size), 1.0e-4)
        ).to(torch.int64)
        composite_parts.append(cells)
    _, composite_group = torch.unique(
        torch.cat(composite_parts, dim=1),
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    composite_rank = _within_group_rank(
        composite_group, candidate_score
    )
    minimum = candidate_score.min()
    maximum = candidate_score.max()
    normalized_score = (candidate_score - minimum) / (
        maximum - minimum
    ).clamp_min(1.0e-12)
    # Integer group rank is the primary key; score resolves which group wins
    # when a context has fewer remaining slots than active physical groups.
    hierarchical_priority = (
        -composite_rank.to(candidate_score.dtype)
        + 0.25 * normalized_score
    )
    contexts, context_group, context_counts = torch.unique(
        candidate_context,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    context_score = torch.full(
        (len(contexts),),
        -1.0e30,
        device=score.device,
        dtype=score.dtype,
    )
    context_score.scatter_reduce_(
        0,
        context_group,
        candidate_score,
        reduce="amax",
        include_self=True,
    )
    allocations = _waterfill_group_allocations(
        context_counts, context_score, quota
    )
    selected_local = _variable_group_topk(
        hierarchical_priority, context_group, allocations
    )
    if len(selected_local) != quota:
        raise RuntimeError(
            "Vectorized balanced topology selection did not fill its quota"
        )
    return indices[selected_local]


def _evidence_adaptive_role_quotas(
    eligible_counts: dict[str, int],
    population_counts: dict[str, int],
    capacity: int,
    *,
    observable_counts: dict[str, int] | None = None,
    effective_eligible_mass: dict[str, float] | None = None,
    normalize_by_observable_population: bool = True,
) -> dict[str, int]:
    """Allocate capacity from the unresolved fraction of each physical role.

    A fixed 45/40 canonical/dynamic quota was not neutral: after conditioned
    gradients were isolated from canonical rows, it still moved roughly
    ninety thousand split slots from exact dynamic leaves into the canonical
    crown. The resulting canonical children had too few unbiased updates,
    while the dynamic branch lost the bandwidth it needed. Conversely, a
    phase-level zero for canonical demand leaves genuinely broad localization
    splats permanently frozen.

    Raw eligible row counts are not comparable either.  The current evidence
    store contains about 895k observation-space dynamic births but only 114k
    canonical visual-hull cells.  Allocating in proportion to row count made
    that producer sampling density self-reinforcing: a role with more
    duplicate ray samples received more children even when the *fraction* of
    its representation with a screen-bandwidth deficit was smaller.

    Eligible rows already encode screen-bandwidth deficit, evidence validity,
    rendered contribution and role-specific observation support. For generic
    row allocation the optional effective mass is normalized by the
    *observable population in this topology epoch*, preventing producer row
    density from becoming self-reinforcing. Physical split growth is
    different: one slot resolves one unit of projected bandwidth irrespective
    of how many low-frequency rows a role happens to contain. Its caller
    therefore supplies total authority-weighted screen-bandwidth demand and
    disables population normalization. Falling back to raw eligible counts
    preserves legacy callers. Every role remains eligible in every phase;
    there is no fixed role percentage or pass/fail gate.
    """
    role_order = (
        "static_skeleton",
        "canonical_crown",
        "dynamic_leaf",
    )
    unexpected = (
        set(eligible_counts) | set(population_counts)
    ) - set(role_order)
    if unexpected:
        raise ValueError(
            "Unknown volume roles in eligible counts: "
            + ", ".join(sorted(unexpected))
        )
    counts = {
        role: int(eligible_counts.get(role, 0)) for role in role_order
    }
    populations = {
        role: int(population_counts.get(role, 0)) for role in role_order
    }
    observable = (
        populations
        if observable_counts is None
        else {
            role: int(observable_counts.get(role, 0))
            for role in role_order
        }
    )
    effective_mass = (
        {role: float(counts[role]) for role in role_order}
        if effective_eligible_mass is None
        else {
            role: float(effective_eligible_mass.get(role, 0.0))
            for role in role_order
        }
    )
    if any(
        (not np.isfinite(value)) or value < 0
        for value in (
            *counts.values(),
            *populations.values(),
            *observable.values(),
            *effective_mass.values(),
        )
    ):
        raise ValueError(
            "Eligible, population and effective role demand must be "
            "non-negative"
        )
    for role in role_order:
        if counts[role] > populations[role]:
            raise ValueError(
                f"Eligible {role} count exceeds its live population"
            )
        if observable[role] > populations[role]:
            raise ValueError(
                f"Observable {role} count exceeds its live population"
            )
        if counts[role] > observable[role]:
            raise ValueError(
                f"Eligible {role} count exceeds its observable population"
            )
        if (
            normalize_by_observable_population
            and effective_mass[role] > counts[role] + 1e-5
        ):
            raise ValueError(
                f"Effective eligible {role} mass exceeds its eligible count"
            )
    requested = max(int(capacity), 0)
    total = sum(counts.values())
    allocatable = min(requested, total)
    quotas = {role: 0 for role in role_order}
    if allocatable == 0:
        return quotas

    order_index = {role: index for index, role in enumerate(role_order)}
    demand = {
        role: (
            (
                effective_mass[role] / float(observable[role])
                if observable[role] > 0
                else 0.0
            )
            if normalize_by_observable_population
            else effective_mass[role]
        )
        for role in role_order
    }
    remainder = allocatable
    while remainder:
        active = [
            role
            for role in role_order
            if quotas[role] < counts[role]
        ]
        if not active:
            break
        weight_sum = sum(demand[role] for role in active)
        if weight_sum <= 0:
            # This is reachable only through zero-sized numerical demand.
            # Fall back to remaining row capacity without inventing a role
            # preference.
            weights = {
                role: counts[role] - quotas[role] for role in active
            }
            weight_sum = float(sum(weights.values()))
        else:
            weights = {role: demand[role] for role in active}
        raw = {
            role: remainder * weights[role] / weight_sum
            for role in active
        }
        assigned = 0
        for role in active:
            addition = min(
                counts[role] - quotas[role],
                int(np.floor(raw[role])),
            )
            quotas[role] += addition
            assigned += addition
        remainder -= assigned
        if not remainder:
            break
        candidates = sorted(
            (
                role
                for role in active
                if quotas[role] < counts[role]
            ),
            key=lambda role: (
                -(raw[role] - np.floor(raw[role])),
                -weights[role],
                order_index[role],
            ),
        )
        if not candidates:
            break
        for role in candidates[:remainder]:
            quotas[role] += 1
        remainder = allocatable - sum(quotas.values())
    if sum(quotas.values()) != allocatable:
        raise RuntimeError("Evidence-adaptive role allocation lost capacity")
    return quotas


@torch.no_grad()
def _select_volume_reallocation_retirements(
    foliage,
    stats: dict[str, torch.Tensor],
    requested: int,
    *,
    excluded: torch.Tensor | None = None,
    requested_by_role: dict[str, int] | None = None,
) -> tuple[torch.Tensor, dict]:
    """Retire low-utility siblings without silently transferring role capacity."""
    count = len(foliage)
    device = foliage.xyz.device
    remove = torch.zeros(count, dtype=torch.bool, device=device)
    requested = min(max(int(requested), 0), count)
    role_order = (
        "static_skeleton",
        "canonical_crown",
        "dynamic_leaf",
    )
    if requested_by_role is not None:
        unexpected = set(requested_by_role) - set(role_order)
        if unexpected:
            raise ValueError(
                "Unknown reallocation roles: "
                + ", ".join(sorted(unexpected))
            )
        requested_by_role = {
            role: max(int(requested_by_role.get(role, 0)), 0)
            for role in role_order
        }
        if sum(requested_by_role.values()) > requested:
            raise ValueError(
                "Role-matched retirement requests cannot exceed requested"
            )
    empty_audit = {
        "requested": requested,
        "requested_by_role": requested_by_role,
        "eligible_redundant_descendants": 0,
        "eligible_redundant_descendants_by_role": {
            role: 0 for role in role_order
        },
        "selected": 0,
        "shortfall": requested,
        "selected_by_role": {role: 0 for role in role_order},
    }
    if requested == 0 or count == 0:
        return remove, empty_audit
    excluded = (
        torch.zeros(count, dtype=torch.bool, device=device)
        if excluded is None
        else torch.as_tensor(
            excluded, device=device, dtype=torch.bool
        ).reshape(count)
    )
    evidence_id = foliage.evidence_primitive_id.to(torch.int64)
    track_id = foliage.track_id.to(torch.int64)
    uses_evidence_id = evidence_id >= 0
    legacy_lineage_id = torch.where(uses_evidence_id, evidence_id, track_id)
    parent_lineage_id = getattr(
        foliage,
        "parent_lineage_id",
        torch.full_like(legacy_lineage_id, -1),
    ).to(torch.int64)
    lineage_id = torch.where(
        parent_lineage_id >= 0, parent_lineage_id, legacy_lineage_id
    )
    candidates = (
        (foliage.split_generation > 0)
        & (lineage_id != -1)
        & ~foliage.static_skeleton_mask
        & (
            getattr(
                foliage,
                "verification_state",
                torch.full_like(
                    foliage.layer_role, VERIFICATION_VERIFIED
                ),
            )
            == VERIFICATION_VERIFIED
        )
        & ~excluded
    )
    rows = torch.nonzero(candidates, as_tuple=False).flatten()
    if not len(rows):
        return remove, empty_audit

    contribution = stats["contribution"].float().clamp_min(0)
    gradient = (
        stats["gradient"].float()
        / stats["gradient_count"].float().clamp_min(1)
    ).clamp_min(0)
    rigid_fraction = (
        stats["rigid"].float()
        / contribution.clamp_min(1e-8)
    ).clamp(0, 1)
    observation_count = stats.get(
        "observation_count", torch.zeros_like(contribution)
    ).float()
    utility = (
        torch.log1p(contribution) * (1.0 - rigid_fraction)
        + 0.20 * torch.log1p(gradient)
        + 0.25 * torch.log1p(observation_count)
        + 0.10
        * foliage.opacities.float()
        * foliage.occupancy_probability.float().clamp(0, 1)
    )
    keys = torch.stack(
        [
            foliage.layer_role[rows].to(torch.int64),
            uses_evidence_id[rows].to(torch.int64),
            lineage_id[rows],
        ],
        dim=1,
    )
    _, inverse, group_counts = torch.unique(
        keys,
        dim=0,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    group_maximum = torch.full(
        (len(group_counts),),
        -torch.inf,
        device=device,
        dtype=utility.dtype,
    )
    group_maximum.scatter_reduce_(
        0, inverse, utility[rows], reduce="amax", include_self=True
    )
    maximum_rows = utility[rows] >= group_maximum[inverse] - 1e-12
    sentinel = torch.iinfo(torch.int64).max
    first_maximum = torch.full(
        (len(group_counts),),
        sentinel,
        device=device,
        dtype=torch.int64,
    )
    first_maximum.scatter_reduce_(
        0,
        inverse,
        torch.where(
            maximum_rows,
            rows,
            torch.full_like(rows, sentinel),
        ),
        reduce="amin",
        include_self=True,
    )
    redundant = (group_counts[inverse] > 1) & (
        rows != first_maximum[inverse]
    )
    eligible = rows[redundant]
    role_masks = {
        "static_skeleton": foliage.static_skeleton_mask,
        "canonical_crown": foliage.canonical_crown_mask,
        "dynamic_leaf": foliage.dynamic_leaf_mask,
    }
    eligible_by_role = {
        role: eligible[role_masks[role][eligible]]
        for role in role_order
    }
    if requested_by_role is None:
        take = min(requested, int(len(eligible)))
        chosen = eligible[:0]
        if take:
            chosen = eligible[
                torch.topk(
                    utility[eligible], take, largest=False, sorted=False
                ).indices
            ]
    else:
        chosen_parts = []
        for role in role_order:
            role_eligible = eligible_by_role[role]
            role_take = min(
                requested_by_role[role], int(len(role_eligible))
            )
            if role_take:
                chosen_parts.append(
                    role_eligible[
                        torch.topk(
                            utility[role_eligible],
                            role_take,
                            largest=False,
                            sorted=False,
                        ).indices
                    ]
                )
        chosen = (
            torch.cat(chosen_parts)
            if chosen_parts
            else eligible[:0]
        )
        take = int(len(chosen))
    if take:
        remove[chosen] = True
    selected_by_role = {
        "static_skeleton": int(
            foliage.static_skeleton_mask[chosen].sum()
        ),
        "canonical_crown": int(
            foliage.canonical_crown_mask[chosen].sum()
        ),
        "dynamic_leaf": int(
            foliage.dynamic_leaf_mask[chosen].sum()
        ),
    }
    return remove, {
        "requested": requested,
        "requested_by_role": requested_by_role,
        "eligible_redundant_descendants": int(len(eligible)),
        "eligible_redundant_descendants_by_role": {
            role: int(len(rows))
            for role, rows in eligible_by_role.items()
        },
        "selected": int(take),
        "shortfall": int(requested - take),
        "selected_by_role": selected_by_role,
    }


@torch.no_grad()
def _refresh_split_child_owner_colors(
    foliage,
    *,
    child_start: int,
    owner_camera_ids: torch.Tensor,
    view_by_camera_id: dict[int, object],
    support_fallback_enabled: bool = True,
    geometry_evidence=None,
    foliage_ray_evidence: FoliageRayEvidence | None = None,
) -> dict[str, object]:
    """Re-anchor optical children to full-resolution immutable RGB evidence.

    Copying a broad parent's optimized DC colour into every child preserves
    the very low-pass appearance that subdivision is meant to remove.  For
    exact-owner leaf rows, project each new child into its calibrated source
    camera.  For a canonical child, robustly pool the calibrated observations
    whose measured depth is consistent with the child's current projection.
    Both paths sample the immutable native target with pixel-centre bilinear
    coordinates; no camera-plane appearance grid participates.
    """
    owner_camera_ids = torch.as_tensor(
        owner_camera_ids,
        device=foliage.xyz.device,
        dtype=torch.long,
    ).reshape(-1)
    child_start = int(child_start)
    if not len(owner_camera_ids):
        return {
            "contract": (
                "full_resolution_depth_consistent_exact_dynamic_owner_"
                "plus_depth_consistent_multiview_canonical_pixel_centres"
            ),
            "candidate_children": 0,
            "refreshed_children": 0,
            "dynamic_candidate_children": 0,
            "dynamic_refreshed_children": 0,
            "dynamic_depth_candidate_children": 0,
            "canonical_candidate_children": 0,
            "canonical_refreshed_children": 0,
            "canonical_depth_candidate_children": 0,
            "canonical_depth_refreshed_children": 0,
            "canonical_support_fallback_children": 0,
            "canonical_low_support_dc_only_children": 0,
            "canonical_single_ray_dc_children": 0,
            "metric_depth_candidate_children": 0,
            "metric_depth_refreshed_children": 0,
            "ray_depth_candidate_children": 0,
            "ray_depth_refreshed_children": 0,
            "ray_depth_rejected_children": 0,
            "ray_source_pixel_distance_mean": 0.0,
            "ray_source_pixel_distance_p90": 0.0,
            "ray_source_pixel_distance_maximum": 0.0,
            "ray_depth_confidence_mean": 0.0,
            "owner_camera_count": 0,
            "skipped_without_scene_lookup": 0,
        }
    child_indices = torch.arange(
        child_start,
        child_start + len(owner_camera_ids),
        device=foliage.xyz.device,
        dtype=torch.long,
    )
    if int(child_indices[-1]) >= len(foliage):
        raise RuntimeError("Split child colour metadata exceeds new topology")
    child_roles = foliage.layer_role[child_indices]
    dynamic_child = child_roles == LAYER_DYNAMIC_LEAF
    canonical_child = child_roles == LAYER_CANONICAL_CROWN
    refreshed_mask = torch.zeros(
        len(child_indices), dtype=torch.bool, device=foliage.xyz.device
    )
    refreshed_cameras: set[int] = set()
    missing_scene_camera_children = 0
    dynamic_depth_candidate = torch.zeros_like(dynamic_child)
    metric_depth_candidate = torch.zeros_like(dynamic_child)
    metric_depth_refreshed = torch.zeros_like(dynamic_child)
    ray_depth_candidate = torch.zeros_like(dynamic_child)
    ray_depth_refreshed = torch.zeros_like(dynamic_child)
    ray_source_pixel_distance = foliage.xyz.new_full(
        (len(child_indices),), float("nan")
    )
    ray_depth_confidence = foliage.xyz.new_full(
        (len(child_indices),), float("nan")
    )
    metric_depth_cache: dict[int, tuple[torch.Tensor, ...] | None] = {}

    def metric_inverse_depth_fields(view) -> tuple[torch.Tensor, ...] | None:
        """Load immutable world-metric ray posterior without audit mutation."""
        camera_id = int(view.colmap_id)
        if camera_id in metric_depth_cache:
            return metric_depth_cache[camera_id]
        if (
            geometry_evidence is None
            or geometry_evidence.inverse_root is None
        ):
            metric_depth_cache[camera_id] = None
            return None
        stem = Path(str(view.image_name)).stem
        frame = geometry_evidence.frame_by_stem.get(stem)
        if frame is None:
            metric_depth_cache[camera_id] = None
            return None
        root = geometry_evidence.inverse_root
        paths = {
            "rho": root / f"rho_mean_frame{frame:06d}.npy",
            "variance": root / f"rho_variance_frame{frame:06d}.npy",
            "support": root / f"support_view_count_frame{frame:06d}.npy",
        }
        if not all(path.is_file() for path in paths.values()):
            metric_depth_cache[camera_id] = None
            return None
        loaded_fields = []
        for name in ("rho", "variance", "support"):
            array = np.array(
                np.load(paths[name]), copy=True, order="C"
            )
            loaded_fields.append(
                torch.from_numpy(array)
                .to(device=foliage.xyz.device, dtype=foliage.xyz.dtype)
                .unsqueeze(0)
                .unsqueeze(0)
            )
        fields = tuple(loaded_fields)
        metric_depth_cache[camera_id] = fields
        return fields

    def sample_projected_rgb(
        camera_id: int,
        local_child_indices: torch.Tensor,
        *,
        measured_depth: torch.Tensor | None = None,
        measured_uv: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return valid local rows and their exact-raster RGB samples."""
        nonlocal missing_scene_camera_children
        view = view_by_camera_id.get(camera_id)
        if view is None:
            if not view_by_camera_id:
                missing_scene_camera_children += int(
                    len(local_child_indices)
                )
                return local_child_indices[:0], foliage.xyz.new_empty((0, 3))
            raise RuntimeError(
                f"Split child evidence camera {camera_id} is absent from "
                "scene"
            )
        indices = child_indices[local_child_indices]
        xyz = foliage.xyz.detach()[indices]
        homogeneous = torch.cat(
            [xyz, torch.ones_like(xyz[:, :1])], dim=1
        )
        camera_xyz = homogeneous @ view.world_view_transform
        depth = camera_xyz[:, 2]
        pixel_x = (
            float(view.focal_x) * camera_xyz[:, 0]
            / depth.clamp_min(1e-5)
            + float(view.cx)
        )
        pixel_y = (
            float(view.focal_y) * camera_xyz[:, 1]
            / depth.clamp_min(1e-5)
            + float(view.cy)
        )
        normalized_uv = torch.stack(
            [
                (pixel_x + 0.5) / float(view.image_width),
                (pixel_y + 0.5) / float(view.image_height),
            ],
            dim=1,
        )
        valid = (
            torch.isfinite(normalized_uv).all(dim=1)
            & torch.isfinite(depth)
            & (depth > 0.05)
            & (normalized_uv[:, 0] >= 0)
            & (normalized_uv[:, 0] <= 1)
            & (normalized_uv[:, 1] >= 0)
            & (normalized_uv[:, 1] <= 1)
        )
        model_indices = child_indices[local_child_indices]
        depth_tolerance = torch.maximum(
            2.5 * foliage.scales.detach()[model_indices].amax(dim=1),
            depth.new_full(depth.shape, 0.03),
        )
        ray_spatial = torch.zeros_like(valid)
        ray_consistent = torch.zeros_like(valid)
        if foliage_ray_evidence is not None:
            ray_posterior = foliage_ray_evidence.sample_hit_depth_posterior(
                camera_id,
                torch.stack([pixel_x, pixel_y], dim=1),
                render_width=int(view.image_width),
                render_height=int(view.image_height),
                camera_z_depth=depth,
                depth_tolerance=depth_tolerance,
            )
            ray_spatial = ray_posterior["spatial_candidate"]
            ray_consistent = ray_posterior["depth_consistent"]
            ray_depth_candidate[
                local_child_indices[ray_spatial]
            ] = True
            accepted_ray_local = local_child_indices[ray_consistent]
            ray_source_pixel_distance[accepted_ray_local] = ray_posterior[
                "source_pixel_distance"
            ][ray_consistent]
            ray_depth_confidence[accepted_ray_local] = ray_posterior[
                "confidence"
            ][ray_consistent]
        metric_fields = metric_inverse_depth_fields(view)
        metric_valid = torch.zeros_like(valid)
        metric_consistent = torch.zeros_like(valid)
        if metric_fields is not None:
            sample_grid = (2.0 * normalized_uv - 1.0).reshape(
                1, -1, 1, 2
            )
            rho = F.grid_sample(
                metric_fields[0],
                sample_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).reshape(-1)
            rho_variance = F.grid_sample(
                metric_fields[1],
                sample_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).reshape(-1)
            support = F.grid_sample(
                metric_fields[2],
                sample_grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=False,
            ).reshape(-1)
            metric_valid = (
                torch.isfinite(rho)
                & (rho > 1.0e-6)
                & torch.isfinite(rho_variance)
                & (rho_variance >= 0)
                & (support >= 1)
            )
            metric_depth = rho.clamp_min(1.0e-6).reciprocal()
            metric_depth_sigma = (
                rho_variance.clamp_min(0).sqrt()
                / rho.square().clamp_min(1.0e-8)
            ).clamp(0.02, 1.0)
            metric_tolerance = torch.maximum(
                2.5 * metric_depth_sigma,
                torch.maximum(
                    2.5
                    * foliage.scales.detach()[model_indices].amax(dim=1),
                    depth.new_full(depth.shape, 0.05),
                ),
            )
            metric_consistent = metric_valid & (
                (depth - metric_depth).abs() <= metric_tolerance
            )
            metric_depth_candidate[local_child_indices[metric_valid]] = True
        if measured_depth is not None:
            measured_depth = measured_depth.to(
                device=depth.device, dtype=depth.dtype
            )
            # A child is offset inside its parent's physical footprint.  The
            # accepted interval therefore follows that child's current scale
            # but never grows into a broad, view-independent colour lookup.
            inherited_consistent = (
                torch.isfinite(measured_depth)
                & (measured_depth > 0.05)
                & ((depth - measured_depth).abs() <= depth_tolerance)
            )
            if measured_uv is not None:
                measured_uv = measured_uv.to(
                    device=depth.device, dtype=depth.dtype
                )
                projected_radius = (
                    float(np.sqrt(float(view.focal_x) * float(view.focal_y)))
                    * foliage.scales.detach()[model_indices].amax(dim=1)
                    / depth.clamp_min(0.05)
                ).clamp(2.0, 12.0)
                pixel_scale = normalized_uv.new_tensor(
                    [float(view.image_width), float(view.image_height)]
                )
                inherited_consistent &= (
                    ((normalized_uv - measured_uv) * pixel_scale)
                    .norm(dim=1)
                    <= projected_radius
                )
            # Current image-space ray intervals are the strongest witness.
            # Dense metric depth and the original lineage observation are
            # independent fallbacks, not gates that can veto a valid current
            # posterior attachment.
            valid &= (
                ray_consistent
                | metric_consistent
                | inherited_consistent
            )
        elif geometry_evidence is not None or foliage_ray_evidence is not None:
            # A support-camera identity alone is not a depth witness.  When
            # dense metric evidence is available, reject RGB lookup unless
            # that current child projection lies in its posterior interval.
            valid &= ray_consistent | metric_consistent
        if not bool(valid.any()):
            return local_child_indices[:0], foliage.xyz.new_empty((0, 3))
        target = view.original_image.to(
            device=foliage.xyz.device, non_blocking=True
        )[None]
        grid = (2.0 * normalized_uv[valid] - 1.0).reshape(1, -1, 1, 2)
        colors = F.grid_sample(
            target,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[0, :, :, 0].T.clamp(0, 1)
        refreshed_cameras.add(camera_id)
        metric_depth_refreshed[
            local_child_indices[valid & metric_consistent]
        ] = True
        ray_depth_refreshed[
            local_child_indices[valid & ray_consistent]
        ] = True
        return local_child_indices[valid], colors

    # An exact observation owns the dynamic child's base colour.
    for camera_id_tensor in torch.unique(owner_camera_ids[dynamic_child]):
        camera_id = int(camera_id_tensor)
        if camera_id < 0:
            continue
        local = torch.nonzero(
            dynamic_child & (owner_camera_ids == camera_id),
            as_tuple=False,
        ).flatten()
        measured_depth = None
        measured_uv = None
        # Exact dynamic ownership includes a calibrated hit depth.  A split
        # child displaced outside that posterior must retain the parent's
        # colour, not sample whichever sky/facade pixel happens to lie under
        # its new projection.  That missing check created the bright leaf
        # flecks visible after otherwise correct full-resolution refresh.
        if foliage.observation_camera_ids.shape[1]:
            child_rows = child_indices[local]
            observation_cameras = foliage.observation_camera_ids[
                child_rows
            ].long()
            matches = observation_cameras == camera_id
            has_depth_owner = matches.any(dim=1)
            local = local[has_depth_owner]
            if not len(local):
                continue
            child_rows = child_indices[local]
            matches = foliage.observation_camera_ids[
                child_rows
            ].long() == camera_id
            depth_slot = matches.to(torch.int64).argmax(dim=1)
            measured_depth = foliage.observation_depth[
                child_rows
            ].gather(1, depth_slot[:, None])[:, 0]
            measured_uv = foliage.observation_uv[
                child_rows
            ].gather(
                1,
                depth_slot[:, None, None].expand(-1, 1, 2),
            )[:, 0]
            dynamic_depth_candidate[local] = True
        valid_local, colors = sample_projected_rgb(
            camera_id,
            local,
            measured_depth=measured_depth,
            measured_uv=measured_uv,
        )
        valid_indices = child_indices[valid_local]
        foliage.features[valid_indices, 0] = (
            colors - 0.5
        ) / 0.28209479177387814
        refreshed_mask[valid_local] = True

    # Canonical children have no single appearance owner.  Pool only their
    # persisted, depth-consistent multi-view observations so transient leaves
    # and background pixels cannot recolour a shared static child.
    canonical_local = torch.nonzero(canonical_child, as_tuple=False).flatten()
    canonical_candidate = torch.zeros_like(canonical_child)
    canonical_depth_candidate = torch.zeros_like(canonical_child)
    canonical_depth_refreshed = torch.zeros_like(canonical_child)
    canonical_support_fallback = torch.zeros_like(canonical_child)
    color_sum = foliage.xyz.new_zeros((len(child_indices), 3))
    color_count = foliage.xyz.new_zeros(len(child_indices))
    color_observation_indices: list[torch.Tensor] = []
    color_observations: list[torch.Tensor] = []

    def record_color_observations(
        local_indices: torch.Tensor, colors: torch.Tensor
    ) -> None:
        if not len(local_indices):
            return
        color_sum.index_add_(0, local_indices, colors)
        color_count.index_add_(
            0,
            local_indices,
            torch.ones_like(local_indices, dtype=color_count.dtype),
        )
        color_observation_indices.append(local_indices)
        color_observations.append(colors)
    if len(canonical_local) and foliage.observation_camera_ids.shape[1]:
        observation_cameras = foliage.observation_camera_ids[
            child_indices[canonical_local]
        ].long()
        observation_depths = foliage.observation_depth[
            child_indices[canonical_local]
        ]
        observation_uvs = foliage.observation_uv[
            child_indices[canonical_local]
        ]
        for slot in range(observation_cameras.shape[1]):
            slot_camera = observation_cameras[:, slot]
            slot_depth = observation_depths[:, slot]
            slot_uv = observation_uvs[:, slot]
            valid_slot = slot_camera >= 0
            if not bool(valid_slot.any()):
                continue
            slot_local = canonical_local[valid_slot]
            slot_camera = slot_camera[valid_slot]
            slot_depth = slot_depth[valid_slot]
            slot_uv = slot_uv[valid_slot]
            canonical_candidate[slot_local] = True
            canonical_depth_candidate[slot_local] = True
            for camera_id_tensor in torch.unique(slot_camera):
                camera_id = int(camera_id_tensor)
                camera_choice = slot_camera == camera_id
                selected_local = slot_local[camera_choice]
                selected_depth = slot_depth[camera_choice]
                selected_uv = slot_uv[camera_choice]
                valid_local, colors = sample_projected_rgb(
                    camera_id,
                    selected_local,
                    measured_depth=selected_depth,
                    measured_uv=selected_uv,
                )
                if not len(valid_local):
                    continue
                canonical_depth_refreshed[valid_local] = True
                record_color_observations(valid_local, colors)
    # Visual-hull canonical rows carry the calibrated cameras that jointly
    # supported their cross-sequence intersection, but older evidence stores
    # do not duplicate a depth scalar per support view.  Those are still much
    # stronger colour witnesses than copying a broad parent.  Use a bounded
    # number only when no explicit observation table exists and the row has
    # independent sequence support; pooling across views suppresses transient
    # leaves and exposure outliers without creating camera-owned primitives.
    if (
        support_fallback_enabled
        and len(canonical_local)
        and foliage.support_camera_ids.shape[1]
    ):
        child_rows = child_indices[canonical_local]
        retained_support_count = (
            foliage.support_camera_ids[child_rows] >= 0
        ).sum(dim=1)
        # A persisted depth entry is only a *candidate*.  The child may have
        # moved outside every recorded posterior after a split.  Previously
        # merely having such an entry disabled support-camera fallback, even
        # when all projections failed; in a representative 7k event that
        # refreshed only 1,018 of 39,540 children.  Fall back whenever fewer
        # than two actual colour observations survived instead.
        fallback_local_mask = (
            (color_count[canonical_local] < 2)
            & (retained_support_count >= 2)
        )
        fallback_local = canonical_local[fallback_local_mask]
        if len(fallback_local):
            support_cameras = foliage.support_camera_ids[
                child_indices[fallback_local]
            ].long()
            maximum_support_views = min(support_cameras.shape[1], 8)
            for slot in range(maximum_support_views):
                slot_camera = support_cameras[:, slot]
                valid_slot = slot_camera >= 0
                if not bool(valid_slot.any()):
                    continue
                slot_local = fallback_local[valid_slot]
                slot_camera = slot_camera[valid_slot]
                canonical_candidate[slot_local] = True
                canonical_support_fallback[slot_local] = True
                for camera_id_tensor in torch.unique(slot_camera):
                    camera_id = int(camera_id_tensor)
                    selected_local = slot_local[
                        slot_camera == camera_id
                    ]
                    valid_local, colors = sample_projected_rgb(
                        camera_id, selected_local
                    )
                    if not len(valid_local):
                        continue
                    record_color_observations(valid_local, colors)
    valid_canonical = canonical_child & (color_count > 0)
    low_support_canonical = canonical_child & (color_count < 2)
    if (
        bool(low_support_canonical.any())
        and foliage.features.shape[1] > 1
    ):
        # A split cannot manufacture directional appearance evidence.  Even
        # a child with zero accepted colour observations must drop the broad
        # parent's higher-order SH; retaining it was the source of sparse
        # black/purple/white novel-view flecks after otherwise valid splits.
        foliage.features[
            child_indices[low_support_canonical], 1:
        ] = 0
    if bool(valid_canonical.any()):
        initial_mean = color_sum / color_count.clamp_min(1.0)[:, None]
        robust_sum = torch.zeros_like(color_sum)
        robust_weight = torch.zeros_like(color_count)
        flat_indices = torch.cat(color_observation_indices)
        flat_colors = torch.cat(color_observations)
        residual = torch.linalg.vector_norm(
            flat_colors - initial_mean[flat_indices], dim=1
        )
        huber_delta = 0.15
        weights = torch.minimum(
            torch.ones_like(residual),
            residual.new_full(residual.shape, huber_delta)
            / residual.clamp_min(1.0e-6),
        )
        robust_sum.index_add_(
            0, flat_indices, flat_colors * weights[:, None]
        )
        robust_weight.index_add_(0, flat_indices, weights)
        colors = robust_sum[valid_canonical] / robust_weight[
            valid_canonical, None
        ].clamp_min(1.0e-6)
        valid_indices = child_indices[valid_canonical]
        valid_counts = color_count[valid_canonical]
        low_support = valid_counts < 2
        if bool(low_support.any()):
            # One observation is not enough to invent a view-dependent leaf
            # colour.  Keep it close to the inherited parent DC and remove
            # inherited higher-order SH, which otherwise creates saturated
            # black/purple/white flecks in novel views.
            parent_dc = (
                foliage.features[valid_indices[low_support], 0]
                * 0.28209479177387814
                + 0.5
            ).clamp(0, 1)
            observed = colors[low_support]
            low_support_rows = torch.nonzero(
                valid_canonical, as_tuple=False
            ).flatten()[low_support]
            ray_verified = ray_depth_refreshed[low_support_rows]
            # One arbitrary RGB sample is too weak to define a static child,
            # but a sample whose *current projection* is attached to the
            # selected canonical snapshot's calibrated hit interval is the
            # explicit single-view exception in the static representation
            # contract.  Keep it DC-only and bounded, yet do not crush its
            # full-resolution contrast back to the low-pass parent.
            trust = torch.where(
                ray_verified,
                0.50
                + 0.40
                * torch.nan_to_num(
                    ray_depth_confidence[low_support_rows], nan=0.0
                ).clamp(0.0, 1.0),
                torch.full_like(parent_dc[:, 0], 0.50),
            )
            delta_limit = torch.where(
                ray_verified,
                torch.full_like(trust, 0.25),
                torch.full_like(trust, 0.15),
            )
            colors[low_support] = parent_dc + trust[:, None] * (
                observed - parent_dc
            ).clamp(-delta_limit[:, None], delta_limit[:, None])
        foliage.features[valid_indices, 0] = (
            colors - 0.5
        ) / 0.28209479177387814
        refreshed_mask[valid_canonical] = True
    accepted_ray_distance = ray_source_pixel_distance[
        ray_depth_refreshed
    ]
    accepted_ray_confidence = ray_depth_confidence[
        ray_depth_refreshed
    ]
    return {
        "contract": (
            "full_resolution_depth_consistent_exact_dynamic_owner_plus_"
            "depth_consistent_multiview_canonical_pixel_centres"
        ),
        "candidate_children": int(
            (
                (dynamic_child & (owner_camera_ids >= 0))
                | canonical_candidate
            ).sum()
        ),
        "refreshed_children": int(refreshed_mask.sum()),
        "dynamic_candidate_children": int(
            (dynamic_child & (owner_camera_ids >= 0)).sum()
        ),
        "dynamic_refreshed_children": int(
            (refreshed_mask & dynamic_child).sum()
        ),
        "dynamic_depth_candidate_children": int(
            dynamic_depth_candidate.sum()
        ),
        "canonical_candidate_children": int(canonical_candidate.sum()),
        "canonical_refreshed_children": int(
            (refreshed_mask & canonical_child).sum()
        ),
        "canonical_depth_candidate_children": int(
            canonical_depth_candidate.sum()
        ),
        "canonical_depth_refreshed_children": int(
            canonical_depth_refreshed.sum()
        ),
        "canonical_support_fallback_children": int(
            canonical_support_fallback.sum()
        ),
        "canonical_low_support_dc_only_children": int(
            low_support_canonical.sum()
        ),
        "canonical_single_ray_dc_children": int(
            (
                low_support_canonical
                & ray_depth_refreshed
            ).sum()
        ),
        "metric_depth_candidate_children": int(
            metric_depth_candidate.sum()
        ),
        "metric_depth_refreshed_children": int(
            metric_depth_refreshed.sum()
        ),
        "ray_depth_candidate_children": int(
            ray_depth_candidate.sum()
        ),
        "ray_depth_refreshed_children": int(
            ray_depth_refreshed.sum()
        ),
        "ray_depth_rejected_children": int(
            (ray_depth_candidate & ~ray_depth_refreshed).sum()
        ),
        "ray_source_pixel_distance_mean": (
            float(accepted_ray_distance.mean())
            if len(accepted_ray_distance)
            else 0.0
        ),
        "ray_source_pixel_distance_p90": (
            float(torch.quantile(accepted_ray_distance, 0.90))
            if len(accepted_ray_distance)
            else 0.0
        ),
        "ray_source_pixel_distance_maximum": (
            float(accepted_ray_distance.max())
            if len(accepted_ray_distance)
            else 0.0
        ),
        "ray_depth_confidence_mean": (
            float(accepted_ray_confidence.mean())
            if len(accepted_ray_confidence)
            else 0.0
        ),
        "owner_camera_count": len(refreshed_cameras),
        "skipped_without_scene_lookup": missing_scene_camera_children,
    }


@torch.no_grad()
def _rollback_failed_split_families(
    args,
    foliage,
    failed_candidate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Merge wholly failed child families back to one physical hypothesis.

    A split is a local proposal, not irreversible truth.  When every extant
    child of one parent has exhausted its real-camera verification window and
    remains optically/visually unused, deleting all children creates a hole.
    Restore one representative carrying the parent's candidate evidence and
    merge the family moments in world space.  This is deliberately local and
    never consults a whole-image metric gate.
    """
    count = len(foliage)
    device = foliage.xyz.device
    remove = torch.zeros(count, dtype=torch.bool, device=device)
    empty = torch.empty(0, dtype=torch.long, device=device)
    limit = int(
        getattr(args, "maximum_volume_family_rollbacks_per_event", 0)
    )
    base_audit: dict[str, object] = {
        "contract": (
            "all_extant_siblings_unverified_expired_zero_owner_witness_"
            "without_free_space_contradiction__local_lineage_merge_preserves_"
            "integrated_optical_mass"
        ),
        "eligible_families": 0,
        "rolled_back_families": 0,
        "removed_children": 0,
        "representatives": 0,
        "integrated_mass_before": 0.0,
        "integrated_mass_after": 0.0,
    }
    if not count or limit <= 0:
        return remove, empty, base_audit
    parent = foliage.parent_lineage_id.to(torch.int64)
    family_rows = parent >= 0
    if not bool(family_rows.any()):
        return remove, empty, base_audit
    rows = torch.nonzero(family_rows, as_tuple=False).flatten()
    parents, inverse, family_counts = torch.unique(
        parent[rows], sorted=True, return_inverse=True, return_counts=True
    )
    candidate_counts = torch.zeros_like(family_counts)
    candidate_counts.scatter_add_(
        0, inverse, failed_candidate[rows].to(candidate_counts.dtype)
    )
    eligible = (family_counts >= 2) & (candidate_counts == family_counts)
    eligible_offsets = torch.nonzero(eligible, as_tuple=False).flatten()
    base_audit["eligible_families"] = int(len(eligible_offsets))
    if not len(eligible_offsets):
        return remove, empty, base_audit
    chosen_offsets = eligible_offsets[:limit]
    chosen_parents = parents[chosen_offsets]
    representatives: list[torch.Tensor] = []
    before_mass = 0.0
    after_mass = 0.0
    integrated_mass = foliage.integrated_optical_mass().detach()
    for parent_id in chosen_parents:
        siblings = torch.nonzero(
            parent == parent_id, as_tuple=False
        ).flatten()
        sibling_mass = integrated_mass[siblings].clamp_min(1.0e-12)
        total_mass = sibling_mass.sum()
        weights = sibling_mass / total_mass
        representative = siblings[torch.argmax(sibling_mass)]
        center = (weights[:, None] * foliage.xyz[siblings]).sum(dim=0)
        rotations = _rotation_matrices_from_quaternions(
            foliage.normalized_quaternions[siblings]
        )
        scales = foliage.scales[siblings]
        covariance = torch.bmm(
            rotations * scales.square()[:, None, :],
            rotations.transpose(1, 2),
        )
        offset = foliage.xyz[siblings] - center
        second_moment = (
            weights[:, None, None]
            * (covariance + offset[:, :, None] * offset[:, None, :])
        ).sum(dim=0)
        representative_rotation = _rotation_matrices_from_quaternions(
            foliage.normalized_quaternions[representative][None]
        )[0]
        local_covariance = (
            representative_rotation.transpose(0, 1)
            @ second_moment
            @ representative_rotation
        )
        merged_scales = torch.diagonal(local_covariance).clamp_min(
            1.0e-10
        ).sqrt()
        if bool(foliage.static_skeleton_mask[representative]):
            alpha_ceiling = float(args.skeleton_opacity_ceiling)
        elif bool(foliage.dynamic_leaf_mask[representative]):
            alpha_ceiling = float(args.dynamic_leaf_opacity_ceiling)
        else:
            alpha_ceiling = float(args.canonical_crown_opacity_ceiling)
        maximum_tau = -np.log1p(-alpha_ceiling)
        merged_area = projected_gaussian_cross_section(
            merged_scales[None]
        )[0].clamp_min(1.0e-12)
        required_area = total_mass / max(maximum_tau, 1.0e-8)
        if bool(merged_area < required_area):
            merged_scales *= torch.sqrt(required_area / merged_area)
            merged_area = projected_gaussian_cross_section(
                merged_scales[None]
            )[0].clamp_min(1.0e-12)
        merged_alpha = -torch.expm1(-total_mass / merged_area)
        merged_alpha = merged_alpha.clamp(1.0e-6, alpha_ceiling)
        foliage.xyz[representative] = center
        foliage.log_scales[representative] = merged_scales.log()
        foliage.opacity_logits[representative, 0] = torch.logit(
            merged_alpha
        )
        for name in (
            "features",
            "deformation_basis",
            "dynamic_feature_basis",
            "dynamic_opacity_basis",
        ):
            value = getattr(foliage, name)
            reshape = (len(weights),) + (1,) * (value.ndim - 1)
            value[representative] = (
                weights.reshape(reshape) * value[siblings]
            ).sum(dim=0)
        foliage.initialization_center[representative] = center
        foliage.position_covariance[representative] = second_moment
        foliage.scale_ceiling[representative] = torch.maximum(
            foliage.scale_ceiling[siblings].amax(dim=0), merged_scales
        )
        foliage.occupancy_probability[representative] = (
            weights * foliage.occupancy_probability[siblings]
        ).sum()
        for name in (
            "support_view_count",
            "support_sequence_count",
            "free_space_violation_count",
            "unknown_view_count",
        ):
            value = getattr(foliage, name)
            maximum = torch.iinfo(value.dtype).max
            value[representative] = value[siblings].to(torch.int64).sum().clamp(
                max=maximum
            ).to(value.dtype)
        support_ids = torch.unique(
            foliage.support_camera_ids[siblings].reshape(-1)
        )
        support_ids = support_ids[support_ids >= 0]
        foliage.support_camera_ids[representative].fill_(-1)
        retained_ids = support_ids[
            : foliage.support_camera_ids.shape[1]
        ]
        foliage.support_camera_ids[
            representative, : len(retained_ids)
        ] = retained_ids
        generation = int(foliage.split_generation[siblings].min())
        foliage.split_generation[representative] = max(generation - 1, 0)
        foliage.lineage_id[representative] = int(parent_id)
        foliage.parent_lineage_id[representative] = -1
        foliage.birth_iteration[representative] = -1
        foliage.verification_state[representative] = VERIFICATION_VERIFIED
        candidate_evidence = foliage.candidate_evidence_primitive_id[siblings]
        candidate_evidence = candidate_evidence[candidate_evidence >= 0]
        if len(candidate_evidence):
            foliage.evidence_primitive_id[representative] = candidate_evidence[0]
        foliage.candidate_evidence_primitive_id[representative] = -1
        foliage.verified_camera_ids[representative].fill_(-1)
        verified_ids = retained_ids[: foliage.verified_camera_ids.shape[1]]
        foliage.verified_camera_ids[
            representative, : len(verified_ids)
        ] = verified_ids
        foliage.verified_camera_count[representative] = len(verified_ids)
        foliage.verified_sequence_count[representative] = max(
            min(int(foliage.support_sequence_count[representative]), 2), 1
        )
        foliage.replacement_overlap_ema[representative] = 0
        foliage.replacement_observation_count[representative] = 0
        foliage.replacement_camera_signature[representative] = 0
        foliage.handoff_retired_fraction[representative] = 0
        foliage.handoff_reference_mass[representative] = total_mass
        remove[siblings] = True
        remove[representative] = False
        representatives.append(representative)
        before_mass += float(total_mass)
        after_mass += float(foliage.integrated_optical_mass()[representative])
    representative_rows = torch.stack(representatives).to(torch.long)
    base_audit.update(
        {
            "rolled_back_families": int(len(representative_rows)),
            "removed_children": int(remove.sum()),
            "representatives": int(len(representative_rows)),
            "integrated_mass_before": before_mass,
            "integrated_mass_after": after_mass,
        }
    )
    return remove, representative_rows, base_audit


def _verification_debt_capacity_scale(
    fraction: float,
    *,
    soft_fraction: float,
    hard_fraction: float,
    minimum_scale: float = 0.0,
) -> float:
    """Continuous topology capacity while real-camera verification catches up."""

    fraction = float(fraction)
    soft_fraction = float(soft_fraction)
    hard_fraction = float(hard_fraction)
    if not 0.0 <= soft_fraction < hard_fraction <= 1.0:
        raise ValueError(
            "verification debt fractions must satisfy 0 <= soft < hard <= 1"
        )
    minimum_scale = float(minimum_scale)
    if not 0.0 <= minimum_scale <= 1.0:
        raise ValueError("minimum verification capacity must lie in [0,1]")
    position = float(
        np.clip(
            (fraction - soft_fraction)
            / (hard_fraction - soft_fraction),
            0.0,
            1.0,
        )
    )
    # Smoothstep is deliberately continuous: this is lifecycle backpressure,
    # not a metric gate.  A non-zero floor prevents old unverified lineages
    # from permanently starving new, independently supported evidence.
    smooth = 1.0 - position * position * (3.0 - 2.0 * position)
    return minimum_scale + (1.0 - minimum_scale) * smooth


@torch.no_grad()
def _select_missing_static_detail_receiver_parents(
    foliage,
    stats: dict[str, torch.Tensor],
    maximum_receivers: int,
) -> tuple[torch.Tensor, dict[str, int | float | str]]:
    """Select one visible envelope row per spatial group lacking detail.

    This is coverage repair, not unconstrained densification.  Existing
    detail groups are excluded before ranking, and a stable per-group rank
    prevents a broad/repeated envelope lineage from receiving more than one
    receiver in an event.  Residual, footprint, visibility, rigidity and
    occupancy remain continuous priorities instead of pass/fail thresholds.
    """
    count = len(foliage)
    device = foliage.xyz.device
    maximum_receivers = max(int(maximum_receivers), 0)
    empty = torch.empty(0, dtype=torch.long, device=device)
    base_audit: dict[str, int | float | str] = {
        "contract": (
            "one_per_missing_replacement_group__continuous_real_render_"
            "residual_priority"
        ),
        "persistent_envelope_rows": int(
            foliage.persistent_envelope_mask.sum()
        ),
        "groups_with_any_static_detail": 0,
        "missing_detail_envelope_rows": 0,
        "visible_evidence_supported_candidates": 0,
        "unique_candidate_groups": 0,
        "selected": 0,
        "score_minimum": 0.0,
        "score_median": 0.0,
        "score_maximum": 0.0,
    }
    if count == 0 or maximum_receivers == 0:
        return empty, base_audit
    required = {
        "contribution",
        "gradient",
        "gradient_count",
        "radius",
        "rigid",
    }
    missing_stats = required - set(stats)
    if missing_stats:
        raise ValueError(
            "Static-detail receiver selection lacks volume statistics: "
            + ", ".join(sorted(missing_stats))
        )
    for name in required:
        if len(stats[name]) != count:
            raise ValueError(
                f"Static-detail receiver statistic {name} is misaligned"
            )

    groups = foliage.replacement_group.long()
    detail_groups = torch.unique(
        groups[foliage.static_leaf_mask & (groups >= 0)]
    )
    envelope = foliage.persistent_envelope_mask & (groups >= 0)
    missing_detail = envelope
    if len(detail_groups):
        missing_detail &= ~torch.isin(groups, detail_groups)
    verification = getattr(
        foliage,
        "verification_state",
        torch.full_like(foliage.layer_role, VERIFICATION_VERIFIED),
    )
    contribution = stats["contribution"].float().clamp_min(0.0)
    visible_supported = (
        missing_detail
        & (verification == VERIFICATION_VERIFIED)
        & (foliage.support_view_count > 0)
        & (foliage.support_camera_ids >= 0).any(dim=1)
        & (contribution > 0)
    )
    rows = torch.nonzero(visible_supported, as_tuple=False).flatten()
    base_audit.update(
        {
            "groups_with_any_static_detail": int(len(detail_groups)),
            "missing_detail_envelope_rows": int(missing_detail.sum()),
            "visible_evidence_supported_candidates": int(len(rows)),
        }
    )
    if not len(rows):
        return empty, base_audit

    gradient = (
        stats["gradient"].float()
        / stats["gradient_count"].float().clamp_min(1.0)
    ).clamp_min(0.0)
    radius = stats["radius"].float().clamp_min(0.0)
    rigid_fraction = (
        stats["rigid"].float() / contribution.clamp_min(1.0e-8)
    ).clamp(0.0, 1.0)
    occupancy = foliage.occupancy_probability.float().clamp(0.0, 1.0)
    score = (
        torch.log1p(1_000.0 * gradient)
        + 0.35 * torch.log1p(radius)
        + 0.10 * torch.log1p(contribution)
        + 0.25 * occupancy
        + 0.25 * (1.0 - rigid_fraction)
    )
    # A replacement group is the physical coverage cell.  Retain its best
    # currently observed envelope row, then apply tree/spatial fairness across
    # groups.  This closes the historical failure mode where 300k detail rows
    # accumulated inside only ~14k groups.
    local_rank = _within_group_rank(groups[rows], score[rows])
    representatives = rows[local_rank == 0]
    selected = _balanced_instance_topk(
        representatives,
        score,
        foliage.tree_instance_id,
        min(maximum_receivers, len(representatives)),
        xyz=foliage.xyz,
        spatial_cell_size=0.30,
        lineage_family_id=groups,
    )
    selected_score = score[selected]
    base_audit.update(
        {
            "unique_candidate_groups": int(len(representatives)),
            "selected": int(len(selected)),
            "score_minimum": float(selected_score.min()),
            "score_median": float(selected_score.median()),
            "score_maximum": float(selected_score.max()),
        }
    )
    return selected, base_audit


def _adapt_volume(
    args,
    foliage,
    stats,
    *,
    volume_budget: int | None = None,
    phase: str | None = None,
    split_capacity_scale: float = 1.0,
    camera_forward_lookup: torch.Tensor | None = None,
    view_by_camera_id: dict[int, object] | None = None,
    geometry_evidence=None,
    foliage_ray_evidence: FoliageRayEvidence | None = None,
    current_iteration: int = -1,
):
    """Prune contradictions first, then split residuals in one mutation.

    The returned mapping always addresses the topology that entered this
    function, so Adam state can be migrated exactly once after both changes.
    """
    split_capacity_scale = float(
        np.clip(split_capacity_scale, 0.0, 1.0)
    )
    effective_split_limit = int(
        np.floor(
            int(args.maximum_volume_splits) * split_capacity_scale
        )
    )
    static_detail_exclusive_topology = (
        _static_detail_exclusive_topology_active(args, phase)
    )
    old_count = len(foliage)
    support = foliage.support_view_count.float()
    contradiction_ratio = _confirmed_contradiction_fraction(
        support,
        foliage.free_space_violation_count,
    )
    contribution_floor = torch.quantile(stats["contribution"], 0.10)
    low_contribution = stats["contribution"] <= contribution_floor
    # A fixed 0.002 cutoff sat below the effective opacity distribution in
    # real v95 checkpoints (minimum ~=0.00209), making the weak contradiction
    # branch unreachable.  Use a role-agnostic lower-tail statistic instead:
    # it expresses relative utility and adapts to the current opacity schedule.
    opacity_floor = torch.quantile(foliage.opacities, 0.10)
    low_opacity = foliage.opacities <= opacity_floor
    confirmed_free = foliage.free_space_violation_count.float()
    strong_contradiction = (
        (
            (contradiction_ratio >= 0.50)
            & (confirmed_free >= 2)
        )
        | (foliage.occupancy_probability < 0.06)
    )
    weak_contradiction = (
        (contradiction_ratio >= 0.30)
        | (foliage.occupancy_probability < 0.12)
    )
    # Observation identity is provenance, not permanent proof of existence.
    # Protect weak/conflicting candidates only when *independent* positive
    # evidence still outweighs the contradiction.  Strong free-space
    # contradiction always wins, including for rows that happened to be
    # sampled in the current topology window.  The former absolute
    # ``observed once => never prune`` rule accumulated exactly the persistent
    # low-opacity fog that the posterior was supposed to retire.
    independently_verified_positive = (
        (foliage.support_view_count >= 2)
        & (foliage.support_sequence_count >= 2)
        & (contradiction_ratio < 0.50)
    )
    weak_free_remove = (
        (contradiction_ratio >= 0.30)
        & (confirmed_free >= 1)
        & low_contribution
        & ~independently_verified_positive
    )
    weak_occupancy_remove = (
        (foliage.occupancy_probability < 0.12)
        & low_contribution
        & low_opacity
        & ~independently_verified_positive
    )
    weak_remove = weak_free_remove | weak_occupancy_remove
    remove = strong_contradiction | weak_remove
    verification_state = getattr(
        foliage,
        "verification_state",
        torch.full_like(foliage.layer_role, VERIFICATION_VERIFIED),
    )
    unverified_before = verification_state == VERIFICATION_UNVERIFIED
    verification_debt_fraction = (
        float(unverified_before.float().mean()) if old_count else 0.0
    )
    debt_soft = float(
        getattr(args, "volume_verification_debt_soft_fraction", 0.05)
    )
    debt_hard = float(
        getattr(args, "volume_verification_debt_hard_fraction", 0.12)
    )
    verification_capacity_scale = _verification_debt_capacity_scale(
        verification_debt_fraction,
        soft_fraction=debt_soft,
        hard_fraction=debt_hard,
        minimum_scale=float(
            getattr(
                args,
                "volume_verification_debt_minimum_split_scale",
                0.0,
            )
        ),
    )
    effective_split_limit = int(
        np.floor(
            effective_split_limit * verification_capacity_scale + 1.0e-6
        )
    )
    birth_iteration = getattr(
        foliage,
        "birth_iteration",
        torch.full_like(foliage.layer_role, -1, dtype=torch.int32),
    )
    verification_grace = (
        (verification_state == VERIFICATION_UNVERIFIED)
        & (birth_iteration >= 0)
        & (
            int(current_iteration) - birth_iteration
            < int(
                getattr(
                    args, "child_verification_grace_iterations", 500
                )
            )
        )
    )
    verification_age = int(current_iteration) - birth_iteration
    verification_expired = (
        (verification_state == VERIFICATION_UNVERIFIED)
        & (birth_iteration >= 0)
        & (
            verification_age
            >= int(
                getattr(
                    args,
                    "child_verification_timeout_iterations",
                    3000,
                )
            )
        )
    )
    verified_camera_count = getattr(
        foliage,
        "verified_camera_count",
        torch.zeros_like(foliage.support_view_count),
    )
    expired_unobserved_low_utility = (
        verification_expired
        & (verified_camera_count <= 0)
        & low_contribution
        & low_opacity
    )
    # During the grace interval positive hit/RGB evidence may grow a child,
    # while ordinary opacity/utility cleanup cannot erase it before its
    # candidate owner cameras are revisited.
    remove &= ~verification_grace
    # A split proposal that has received no positive witness after two full
    # camera epochs, contributes in the bottom utility decile and remains in
    # the opacity bottom decile has accumulated negative verification
    # evidence. Retire only that conjunction; age alone is never a prune
    # permission and partially verified/high-utility rows remain available.
    remove |= expired_unobserved_low_utility
    # Family rollback is a proposal-state transition, not an ordinary prune.
    # Generic render contribution can be high precisely because a wrong
    # child paints an image edge; it is not an independent real-camera owner
    # witness.  If every sibling timed out without such a witness, merge the
    # whole family back to one mass-conserving hypothesis.  Individual rows
    # outside a complete failed family retain the stricter low-utility prune.
    failed_family_candidate = (
        verification_expired
        & (verified_camera_count <= 0)
        & ~strong_contradiction
        & ~weak_contradiction
        & ~foliage.static_skeleton_mask
        & (foliage.candidate_evidence_primitive_id >= 0)
    )
    (
        rollback_remove,
        rollback_representatives,
        rollback_audit,
    ) = _rollback_failed_split_families(
        args, foliage, failed_family_candidate
    )
    if len(rollback_representatives):
        remove |= rollback_remove
        remove[rollback_representatives] = False
        expired_unobserved_low_utility[rollback_representatives] = False
        verification_expired[rollback_representatives] = False
    rollback_representative_mask = torch.zeros_like(remove)
    rollback_representative_mask[rollback_representatives] = True
    observation_count = stats.get(
        "observation_count", torch.zeros_like(stats["contribution"])
    )
    observation_gradient_sum = stats.get(
        "observation_gradient", torch.zeros_like(stats["contribution"])
    )
    has_exact_conditioned_stats = all(
        name in stats
        for name in (
            "conditioned_gradient",
            "conditioned_gradient_count",
            "conditioned_radius",
        )
    )
    conditioned_gradient_sum = stats.get(
        "conditioned_gradient",
        torch.zeros_like(stats["contribution"]),
    )
    conditioned_gradient_count = stats.get(
        "conditioned_gradient_count",
        torch.zeros_like(stats["contribution"]),
    )
    conditioned_radius = stats.get(
        "conditioned_radius", stats["radius"]
    )
    observed_before_prune = observation_count > 0
    persistent_observation = (
        (foliage.observation_camera_ids >= 0).any(dim=1)
        & foliage.dynamic_leaf_mask
    )
    observed_identity = observed_before_prune | persistent_observation
    weak_positive_protected = (
        weak_contradiction
        & low_contribution
        & independently_verified_positive
    )
    static_skeleton_contradiction_protected = (
        remove & foliage.static_skeleton_mask
    )
    # Trunk/branch skeleton is a separate physical owner.  It can be refined
    # by its line/rigidity factors but must not be deleted by a canopy
    # free-space posterior.  Apply that contract before recording the actual
    # contradiction-prune mask so the audit cannot claim a protected trunk was
    # removed.
    remove &= ~foliage.static_skeleton_mask
    observed_unverified_prunable = (
        observed_identity
        & remove
        & ~independently_verified_positive
    )
    contradiction_remove = remove & ~expired_unobserved_low_utility
    effective_budget = (
        args.maximum_volume_gaussians
        if volume_budget is None
        else min(args.maximum_volume_gaussians, volume_budget)
    )
    free_capacity = max(int(effective_budget) - old_count, 0)
    pre_gradient = (
        stats["gradient"] / stats["gradient_count"].clamp_min(1)
    )
    pre_rigid = (
        stats["rigid"] / stats["contribution"].clamp_min(1e-8)
    )
    topology_radius_before = torch.where(
        foliage.dynamic_leaf_mask,
        conditioned_radius,
        stats["radius"],
    )
    radius_ratio_before = (
        topology_radius_before
        / max(float(args.volume_split_radius), 1e-6)
    )
    # Screen footprint is an area demand.  Capping every broad primitive at
    # one made a 20 px paint splat indistinguishable from a 2.1 px residual
    # and forced both through the same binary split cadence.  One adaptive
    # event can spend at most three net growth slots on a parent (four
    # children), so retain severity continuously up to that realizable cost.
    coverage_demand = (
        radius_ratio_before.square() - 1.0
    ).clamp(0.0, 3.0)
    canonical_demand = (
        ~foliage.dynamic_leaf_mask
        & (stats["contribution"] > 0)
        & (pre_rigid < 0.15)
    )
    if static_detail_exclusive_topology:
        canonical_demand &= foliage.static_leaf_mask
    if has_exact_conditioned_stats:
        # Persistent UV/depth identity says who owns the primitive.  A
        # non-zero exact-owner conditioned screen gradient says that this
        # lineage currently lacks image bandwidth.  Neither condition depends
        # on the random auxiliary observation-row sample.
        dynamic_demand = (
            foliage.dynamic_leaf_mask
            & persistent_observation
            & (conditioned_gradient_count > 0)
            & (conditioned_gradient_sum > 0)
        )
    else:
        # Backward-compatible resume path for checkpoints produced before the
        # dedicated conditioned topology channel existed.
        dynamic_demand = foliage.dynamic_leaf_mask & (
            (observation_count > 0)
            | (
                ("observation_count" not in stats)
                & (stats["contribution"] > 0)
                & (pre_gradient > 0)
            )
        )
    topology_authority_before = _volume_split_authority(foliage)
    integrated_demand = (
        coverage_demand
        * topology_authority_before
        * (canonical_demand | dynamic_demand).to(coverage_demand.dtype)
    ).sum()
    requested_splits = min(
        effective_split_limit,
        int(torch.ceil(integrated_demand).item()),
    )
    retirement_requested = max(
        requested_splits
        - free_capacity
        - int(remove.sum()),
        0,
    )
    saturated_budget_settle = old_count >= int(effective_budget)
    if saturated_budget_settle:
        # A full budget is a topology-settle boundary, not permission to
        # replace a fixed fraction of the whole model forever.  Generic
        # residual-driven splits may use genuinely free capacity, and strong
        # contradiction pruning may create such capacity, but they cannot
        # manufacture it by retiring ~20k healthy descendants every event.
        retirement_requested = 0
    # Capacity allocation and retirement must obey the same physical-role
    # demand.  The former implementation selected retirements globally by
    # utility: at v82/8k it retired 11,900 dynamic rows while the subsequent
    # split created only 6,704, silently moving about 5.2k rows/event into the
    # canonical crown even though dynamic had the larger normalized deficit.
    # Compute the same continuous eligibility fractions before mutation and
    # ask each role to fund its own replacement. Free/contradiction capacity
    # remains unowned and can still follow the post-prune adaptive quotas.
    pre_evidence_eligible = (
        (stats["contribution"] > 0)
        & (pre_rigid < 0.15)
    )
    pre_canonical_lineage_supported = (
        foliage.support_sequence_count >= 2
    ) | (
        foliage.static_leaf_mask & (foliage.support_view_count >= 2)
    ) | (foliage.split_generation > 0)
    pre_skeleton_eligible = (
        pre_evidence_eligible
        & foliage.static_skeleton_mask
        & (
            stats["radius"]
            >= float(args.volume_split_radius)
        )
        & (foliage.support_sequence_count >= 2)
    )
    pre_canonical_eligible = (
        pre_evidence_eligible
        & foliage.canonical_crown_mask
        & (
            stats["radius"]
            >= float(args.volume_split_radius)
        )
        & pre_canonical_lineage_supported
        & (verification_state == VERIFICATION_VERIFIED)
        & ~rollback_representative_mask
    )
    if static_detail_exclusive_topology:
        pre_canonical_eligible &= foliage.static_leaf_mask
    if has_exact_conditioned_stats:
        pre_dynamic_eligible = (
            foliage.dynamic_leaf_mask
            & (
                conditioned_radius
                >= float(args.volume_split_radius)
            )
            & persistent_observation
            & (conditioned_gradient_count > 0)
            & (conditioned_gradient_sum > 0)
        )
        pre_dynamic_observable = conditioned_gradient_count > 0
    else:
        pre_dynamic_eligible = (
            foliage.dynamic_leaf_mask
            & (
                topology_radius_before
                >= float(args.volume_split_radius)
            )
            & (
                observed_before_prune
                | (
                    ("observation_count" not in stats)
                    & pre_evidence_eligible
                    & (pre_gradient > 0)
                )
            )
        )
        pre_dynamic_observable = (
            observed_before_prune
            | (stats["contribution"] > 0)
        )
    pre_eligible_counts = {
        "static_skeleton": int(pre_skeleton_eligible.sum()),
        "canonical_crown": int(pre_canonical_eligible.sum()),
        "dynamic_leaf": int(pre_dynamic_eligible.sum()),
    }
    pre_population_counts = {
        "static_skeleton": int(
            foliage.static_skeleton_mask.sum()
        ),
        "canonical_crown": int(
            foliage.canonical_crown_mask.sum()
        ),
        "dynamic_leaf": int(foliage.dynamic_leaf_mask.sum()),
    }
    pre_observable_counts = {
        "static_skeleton": int(
            (
                foliage.static_skeleton_mask
                & (stats["radius"] > 0)
            ).sum()
        ),
        "canonical_crown": int(
            (
                foliage.canonical_crown_mask
                & (stats["radius"] > 0)
                & (stats["contribution"] > 0)
            ).sum()
        ),
        "dynamic_leaf": int(
            (
                foliage.dynamic_leaf_mask
                & pre_dynamic_observable
            ).sum()
        ),
    }
    pre_effective_eligible_mass = {
        "static_skeleton": float(
            topology_authority_before[pre_skeleton_eligible].sum()
        ),
        "canonical_crown": float(
            topology_authority_before[pre_canonical_eligible].sum()
        ),
        "dynamic_leaf": float(
            topology_authority_before[pre_dynamic_eligible].sum()
        ),
    }
    retirement_quota_by_role = _evidence_adaptive_role_quotas(
        pre_eligible_counts,
        pre_population_counts,
        retirement_requested,
        observable_counts=pre_observable_counts,
        effective_eligible_mass=pre_effective_eligible_mass,
    )
    reallocation_remove, reallocation_audit = (
        _select_volume_reallocation_retirements(
            foliage,
            stats,
            retirement_requested,
            excluded=remove,
            requested_by_role=retirement_quota_by_role,
        )
    )
    remove |= reallocation_remove
    # Keep selection in the original index space.  The old implementation
    # materialized a complete pruned model here and a second complete model in
    # ``split_adaptive`` below while Adam still owned the original tensors.
    # At the 2M budget that transient three-topology residency caused the
    # observed 10k OOM.  Retired rows are now masked from all selection
    # statistics and the final replace+split is materialized exactly once.
    keep = ~remove
    pruned = int(remove.sum())
    working_stats = stats

    count = int(keep.sum())
    gradient = (
        working_stats["gradient"]
        / working_stats["gradient_count"].clamp_min(1)
    )
    residual = (
        working_stats["residual"]
        / working_stats["contribution"].clamp_min(1e-8)
    )
    rigid = (
        working_stats["rigid"]
        / working_stats["contribution"].clamp_min(1e-8)
    )
    observation_gradient = (
        working_stats.get(
            "observation_gradient",
            observation_gradient_sum,
        )
        / working_stats.get(
            "observation_count", observation_count
        ).clamp_min(1)
    )
    observation_evidence = (
        working_stats.get(
            "observation_count", observation_count
        )
        > 0
    ) & keep
    persistent_observation = (
        (foliage.observation_camera_ids >= 0).any(dim=1)
        & foliage.dynamic_leaf_mask
        & keep
    )
    conditioned_gradient = (
        working_stats.get(
            "conditioned_gradient",
            conditioned_gradient_sum,
        )
        / working_stats.get(
            "conditioned_gradient_count",
            conditioned_gradient_count,
        ).clamp_min(1)
    )
    conditioned_evidence = (
        working_stats.get(
            "conditioned_gradient_count",
            conditioned_gradient_count,
        )
        > 0
    ) & keep
    # Current checkpoints carry a dedicated exact-owner conditioned channel.
    # Legacy checkpoints do not, but their rendered contribution and
    # observation factor are still valid evidence that a dynamic row was
    # observable in this topology epoch.  Using the absent conditioned
    # counter unconditionally made those rows eligible below while assigning
    # them a zero observable population, which then failed quota allocation.
    dynamic_observable_evidence = (
        conditioned_evidence
        if has_exact_conditioned_stats
        else (
            observation_evidence
            | (working_stats["contribution"] > 0)
        )
    ) & keep
    topology_radius = torch.where(
        foliage.dynamic_leaf_mask,
        working_stats.get(
            "conditioned_radius", conditioned_radius
        ),
        working_stats["radius"],
    )
    evidence_eligible = (
        (working_stats["contribution"] > 0)
        & (rigid < 0.15)
        & keep
    )
    coverage_deficit = (
        topology_radius / max(args.volume_split_radius, 1e-6)
        - 1.0
    ).clamp_min(0)
    split_score = (
        # Before the conditioned render is active, the responsibility channel
        # carries target high-frequency demand.  Screen-space geometry
        # gradient remains a differentiable fallback so an empty residual
        # accumulator cannot turn volume splits into arbitrary index order.
        (residual + 0.10 * torch.log1p(gradient))
        * (1.0 + coverage_deficit)
        * (0.10 + gradient)
        * foliage.occupancy_probability.clamp(0.05, 1.0)
    )
    # A high residual against the initialization rays means the formal
    # information matrix was overconfident or the leaf moved between
    # traversals.  It should not receive the same split priority as a
    # geometrically coherent cell merely because RGB residual is high.
    topology_authority = _volume_split_authority(foliage)
    exact_ray_bandwidth = (
        _dense_exact_ray_bandwidth_mask(foliage) & keep
    )
    dynamic_optical_authority = torch.where(
        exact_ray_bandwidth,
        torch.ones_like(foliage.occupancy_probability),
        foliage.occupancy_probability.clamp(0.05, 1.0),
    )
    split_score = split_score * topology_authority
    dynamic_split_score = (
        # Screen-space gradient is the primary deployable topology signal.
        # The ray/depth observation factor remains a continuous refinement
        # priority, but no longer decides which rows are even eligible.
        (
            torch.log1p(conditioned_gradient)
            + 0.25 * torch.log1p(observation_gradient)
        )
        * (1.0 + coverage_deficit)
        * dynamic_optical_authority
        * topology_authority
    )
    skeleton_eligible = (
        evidence_eligible
        & foliage.static_skeleton_mask
        & (working_stats["radius"] >= args.volume_split_radius)
        & (foliage.support_sequence_count >= 2)
    )
    # Multi-view support proves the initial visual-hull cell.  Once that cell
    # is replaced, its children deliberately carry reduced discrete support
    # counts so they cannot masquerade as independent observations.  Reusing
    # the reduced count as a split gate made every canonical branch stop at
    # generation one (the common two-view parent became two one-view
    # children), which is exactly the coarse, blobby topology visible in the
    # canopy renders.  A rendered-residual descendant remains eligible for
    # recursive refinement through its lineage generation; ray likelihood is
    # evaluated globally and does not depend on duplicated evidence ids.
    canonical_lineage_supported = (
        foliage.support_sequence_count >= 2
    ) | (
        foliage.static_leaf_mask & (foliage.support_view_count >= 2)
    ) | (foliage.split_generation > 0)
    canonical_eligible = (
        evidence_eligible
        & foliage.canonical_crown_mask
        & (working_stats["radius"] >= args.volume_split_radius)
        & canonical_lineage_supported
        & (verification_state == VERIFICATION_VERIFIED)
        & ~rollback_representative_mask
    )
    if static_detail_exclusive_topology:
        canonical_eligible &= foliage.static_leaf_mask
    # Observation gradient says *where* the exact-view residual lives; it
    # does not by itself prove that another primitive is needed. Previously
    # every sampled dynamic leaf with any non-zero observation gradient was
    # eligible, even when its footprint was already sub-pixel. Since the
    # allocator then filled the entire per-event capacity, all 57 v42 events
    # saturated and individual lineages were split up to generation 23.
    # Require an actual screen-bandwidth deficit for every physical role.
    screen_bandwidth_deficit = (
        topology_radius >= float(args.volume_split_radius)
    )
    if has_exact_conditioned_stats:
        dynamic_eligible = (
            foliage.dynamic_leaf_mask
            & screen_bandwidth_deficit
            & persistent_observation
            & conditioned_evidence
            & (conditioned_gradient > 0)
        )
    else:
        dynamic_eligible = (
            foliage.dynamic_leaf_mask
            & screen_bandwidth_deficit
            & (
                (observation_evidence & (observation_gradient > 0))
                | (
                    ("observation_count" not in stats)
                    & evidence_eligible
                    & (
                        working_stats["radius"]
                        >= min(
                            1.0,
                            float(args.volume_split_radius),
                        )
                    )
                )
            )
        )
    split_score = torch.where(
        foliage.dynamic_leaf_mask, dynamic_split_score, split_score
    )
    # Residual is a continuous priority in ``split_score``. Do not turn its
    # within-window quantile into a hard eligibility gate: that gate changes
    # discontinuously with camera sampling and can retire two siblings while
    # admitting only one replacement at a full capacity boundary.
    capacity = min(
        requested_splits,
        max(effective_budget - count, 0),
    )
    split_parents = 0
    children = 0
    final_to_old = torch.nonzero(keep, as_tuple=False).flatten()
    role_split_counts = {
        "static_skeleton": 0,
        "canonical_crown": 0,
        "dynamic_leaf": 0,
    }
    all_eligible_mask = (
        skeleton_eligible | canonical_eligible | dynamic_eligible
    )
    eligible_counts = {
        "static_skeleton": int(skeleton_eligible.sum()),
        "canonical_crown": int(canonical_eligible.sum()),
        "dynamic_leaf": int(dynamic_eligible.sum()),
    }
    population_counts = {
        "static_skeleton": int(
            (foliage.static_skeleton_mask & keep).sum()
        ),
        "canonical_crown": int(
            (foliage.canonical_crown_mask & keep).sum()
        ),
        "dynamic_leaf": int(
            (foliage.dynamic_leaf_mask & keep).sum()
        ),
    }
    conditioned_context_id = working_stats.get(
        "conditioned_context_id",
        torch.full_like(
            foliage.layer_role, -1, dtype=torch.int32
        ),
    ).clone()
    # Checkpoints written before context provenance still retain immutable
    # observation-camera ids.  Use the first calibrated owner only for those
    # legacy rows so a resume neither loses capacity nor invents a context.
    missing_context = (
        foliage.dynamic_leaf_mask
        & keep
        & dynamic_observable_evidence
        & (conditioned_context_id < 0)
    )
    if bool(missing_context.any()) and foliage.observation_camera_ids.shape[1]:
        valid_observation = foliage.observation_camera_ids >= 0
        has_observation = valid_observation.any(dim=1)
        first_slot = valid_observation.to(torch.int8).argmax(dim=1)
        first_camera = foliage.observation_camera_ids.gather(
            1, first_slot[:, None]
        )[:, 0]
        fallback_context = missing_context & has_observation
        conditioned_context_id[fallback_context] = first_camera[
            fallback_context
        ]
    observable_counts = {
        "static_skeleton": int(
            (
                foliage.static_skeleton_mask
                & keep
                & (working_stats["radius"] > 0)
            ).sum()
        ),
        "canonical_crown": int(
            (
                foliage.canonical_crown_mask
                & keep
                & (working_stats["radius"] > 0)
                & (working_stats["contribution"] > 0)
            ).sum()
        ),
        "dynamic_leaf": int(
            (
                foliage.dynamic_leaf_mask
                & keep
                & dynamic_observable_evidence
                & (conditioned_context_id >= 0)
            ).sum()
        ),
    }
    # Split capacity is measured in net child rows, not eligible parents.
    # A parent at >=2x the target radius consumes three net rows for the
    # four-child mutation; an ordinary parent consumes one.  Using parent
    # count as the cap silently dropped two thirds of the available growth
    # whenever a role contained mostly broad Gaussians.
    role_masks = {
        "static_skeleton": skeleton_eligible,
        "canonical_crown": canonical_eligible,
        "dynamic_leaf": dynamic_eligible,
    }
    role_growth_capacity = {}
    effective_eligible_mass = {}
    target_radius = max(float(args.volume_split_radius), 1.0e-6)
    screen_bandwidth_deficit = (
        (topology_radius / target_radius).square() - 1.0
    ).clamp(0.0, 3.0)
    for name, mask in role_masks.items():
        broad = mask & (topology_radius >= 2.0 * target_radius)
        role_growth_capacity[name] = int(mask.sum()) + 2 * int(broad.sum())
        effective_eligible_mass[name] = float(
            (
                topology_authority[mask]
                * screen_bandwidth_deficit[mask]
            ).sum()
        )
    role_quotas = _evidence_adaptive_role_quotas(
        role_growth_capacity,
        role_growth_capacity,
        capacity,
        effective_eligible_mass=effective_eligible_mass,
        normalize_by_observable_population=False,
    )
    # Capacity fairness is defined over physical proposals, not descendant
    # rows.  Without a lineage key, one repeatedly split envelope family can
    # consume every canonical slot before another tree receives its first.
    candidate_family = foliage.candidate_evidence_primitive_id.to(torch.int64)
    parent_family = foliage.parent_lineage_id.to(torch.int64)
    lineage_family = foliage.lineage_id.to(torch.int64)
    static_lineage_family = torch.where(
        candidate_family >= 0,
        candidate_family * 4,
        torch.where(
            parent_family >= 0,
            parent_family * 4 + 1,
            lineage_family * 4 + 2,
        ),
    )
    if bool(all_eligible_mask.any()) and capacity > 0:
        selected_parts = []
        selected_child_count_parts = []
        role_growth_counts = {
            "static_skeleton": 0,
            "canonical_crown": 0,
            "dynamic_leaf": 0,
        }
        for name, mask in (
            ("static_skeleton", skeleton_eligible),
            ("canonical_crown", canonical_eligible),
            ("dynamic_leaf", dynamic_eligible),
        ):
            indices = torch.nonzero(mask, as_tuple=False).flatten()
            growth_quota = role_quotas[name]
            if growth_quota:
                role_context = (
                    conditioned_context_id
                    if name == "dynamic_leaf"
                    else None
                )
                role_lineage_family = (
                    foliage.initialization_source
                    if name == "dynamic_leaf"
                    else static_lineage_family
                )
                # A four-child mutation costs three net rows and halves
                # projected scale in one step.  Reserve it for parents at
                # least twice the target radius; smaller deficits retain the
                # ordinary two-child, one-row-growth split.
                quaternary_candidates = indices[
                    topology_radius[indices]
                    >= 2.0 * float(args.volume_split_radius)
                ]
                quaternary_count = min(
                    int(len(quaternary_candidates)),
                    int(growth_quota) // 3,
                )
                quaternary = _balanced_instance_topk(
                    quaternary_candidates,
                    split_score,
                    foliage.tree_instance_id,
                    quaternary_count,
                    xyz=foliage.xyz,
                    context_id=role_context,
                    lineage_family_id=role_lineage_family,
                )
                remaining_growth = int(growth_quota) - 3 * int(
                    len(quaternary)
                )
                binary_candidates = indices[
                    ~torch.isin(indices, quaternary)
                ]
                binary = _balanced_instance_topk(
                    binary_candidates,
                    split_score,
                    foliage.tree_instance_id,
                    remaining_growth,
                    xyz=foliage.xyz,
                    context_id=role_context,
                    lineage_family_id=role_lineage_family,
                )
                chosen = torch.cat([quaternary, binary])
                child_counts = torch.cat(
                    [
                        torch.full_like(quaternary, 4),
                        torch.full_like(binary, 2),
                    ]
                )
                selected_parts.append(chosen)
                selected_child_count_parts.append(child_counts)
                role_split_counts[name] = int(len(chosen))
                role_growth_counts[name] = int(
                    (child_counts - 1).sum()
                )
        selected = (
            torch.cat(selected_parts)
            if selected_parts
            else torch.empty(0, dtype=torch.long, device=foliage.xyz.device)
        )
        selected_child_counts = (
            torch.cat(selected_child_count_parts)
            if selected_child_count_parts
            else torch.empty(
                0, dtype=torch.long, device=foliage.xyz.device
            )
        )
        selected_growth = int((selected_child_counts - 1).sum())
        if selected_growth > capacity:
            raise RuntimeError(
                "Adaptive volume split exceeded its net-growth capacity"
            )
        instance_split_counts = {
            str(int(instance)): int(
                (
                    foliage.tree_instance_id[selected]
                    == instance
                ).sum()
            )
            for instance in torch.unique(
                foliage.tree_instance_id[selected]
            )
        }
        selected_spatial_group_count = int(
            torch.unique(
                torch.cat(
                    [
                        foliage.tree_instance_id[selected]
                        .to(torch.int64)[:, None],
                        torch.floor(foliage.xyz[selected]).to(
                            torch.int64
                        ),
                    ],
                    dim=1,
                ),
                dim=0,
            ).shape[0]
        )
        selected_radius = topology_radius[selected].float()
        selected_generation = foliage.split_generation[
            selected
        ].float()
        selected_demand = coverage_deficit[selected].float()
        selected_dynamic = selected[foliage.dynamic_leaf_mask[selected]]
        selected_exact_ray_bandwidth = exact_ray_bandwidth[selected]
        eligible_dynamic_contexts = torch.unique(
            conditioned_context_id[dynamic_eligible]
        )
        eligible_dynamic_contexts = eligible_dynamic_contexts[
            eligible_dynamic_contexts >= 0
        ]
        selected_dynamic_contexts = torch.unique(
            conditioned_context_id[selected_dynamic]
        )
        selected_dynamic_contexts = selected_dynamic_contexts[
            selected_dynamic_contexts >= 0
        ]
        if len(selected_dynamic_contexts):
            context_split_counts = torch.stack(
                [
                    (
                        conditioned_context_id[selected_dynamic]
                        == context
                    ).sum()
                    for context in selected_dynamic_contexts
                ]
            ).float()
            dynamic_context_audit = {
                "eligible_owner_camera_count": int(
                    len(eligible_dynamic_contexts)
                ),
                "selected_owner_camera_count": int(
                    len(selected_dynamic_contexts)
                ),
                "splits_per_selected_camera_minimum": int(
                    context_split_counts.min()
                ),
                "splits_per_selected_camera_median": float(
                    context_split_counts.median()
                ),
                "splits_per_selected_camera_maximum": int(
                    context_split_counts.max()
                ),
                "eligible_owner_camera_ids": [
                    int(value)
                    for value in eligible_dynamic_contexts.tolist()
                ],
                "selected_owner_camera_ids": [
                    int(value)
                    for value in selected_dynamic_contexts.tolist()
                ],
            }
        else:
            dynamic_context_audit = {
                "eligible_owner_camera_count": int(
                    len(eligible_dynamic_contexts)
                ),
                "selected_owner_camera_count": 0,
                "splits_per_selected_camera_minimum": 0,
                "splits_per_selected_camera_median": 0.0,
                "splits_per_selected_camera_maximum": 0,
                "eligible_owner_camera_ids": [
                    int(value)
                    for value in eligible_dynamic_contexts.tolist()
                ],
                "selected_owner_camera_ids": [],
            }
        selected_source_counts = {
            str(int(source)): int(
                (foliage.initialization_source[selected] == source).sum()
            )
            for source in torch.unique(
                foliage.initialization_source[selected]
            )
        }
        selection_audit = {
            "radius_pixels_minimum": float(selected_radius.min()),
            "radius_pixels_median": float(selected_radius.median()),
            "radius_pixels_p90": float(
                torch.quantile(selected_radius, 0.90)
            ),
            "radius_pixels_maximum": float(selected_radius.max()),
            "coverage_deficit_median": float(selected_demand.median()),
            "generation_median": float(selected_generation.median()),
            "generation_maximum": int(selected_generation.max()),
            "binary_split_parent_count": int(
                (selected_child_counts == 2).sum()
            ),
            "quaternary_split_parent_count": int(
                (selected_child_counts == 4).sum()
            ),
            "net_growth_slots": selected_growth,
            "dynamic_owner_context": dynamic_context_audit,
            "initialization_source_split_parents": selected_source_counts,
            "exact_ray_camera_plane_split_parents": int(
                selected_exact_ray_bandwidth.sum()
            ),
        }
        split_plane_normals = torch.full(
            (len(selected), 3),
            float("nan"),
            device=foliage.xyz.device,
            dtype=foliage.xyz.dtype,
        )
        if bool(selected_exact_ray_bandwidth.any()):
            if camera_forward_lookup is None:
                raise RuntimeError(
                    "Exact-ray optical subdivision requires calibrated "
                    "camera forward vectors"
                )
            camera_forward_lookup = torch.as_tensor(
                camera_forward_lookup,
                device=foliage.xyz.device,
                dtype=foliage.xyz.dtype,
            ).reshape(-1, 3)
            exact_selected_rows = torch.nonzero(
                selected_exact_ray_bandwidth, as_tuple=False
            ).flatten()
            owner_camera_ids = conditioned_context_id[
                selected[exact_selected_rows]
            ].to(torch.long)
            valid_owner = (
                (owner_camera_ids >= 0)
                & (owner_camera_ids < len(camera_forward_lookup))
            )
            if not bool(valid_owner.all()):
                raise RuntimeError(
                    "Exact-ray split selected a row without a calibrated "
                    "owner camera"
                )
            owner_forward = camera_forward_lookup[owner_camera_ids]
            if not bool(torch.isfinite(owner_forward).all()):
                raise RuntimeError(
                    "Exact-ray split owner camera has no finite forward "
                    "vector"
                )
            split_plane_normals[exact_selected_rows] = owner_forward
        split_event = foliage.replace_and_split_adaptive(
            remove,
            selected,
            selected_child_counts,
            allow_static_skeleton=True,
            split_plane_normals=split_plane_normals,
            birth_iteration=int(current_iteration),
        )
        split_child_start = int(split_event.pop("_child_start"))
        split_child_owner_camera_ids = conditioned_context_id[
            selected
        ].repeat_interleave(selected_child_counts)
        camera_plane_parent_count = int(
            split_event.get("camera_plane_parents", 0)
        )
        expected_camera_plane_parent_count = int(
            selected_exact_ray_bandwidth.sum()
        )
        if camera_plane_parent_count != expected_camera_plane_parent_count:
            raise RuntimeError(
                "Exact-ray selection and camera-plane mutation counts "
                "diverged"
            )
        selection_audit["camera_plane_mutated_parents"] = (
            camera_plane_parent_count
        )
        split_parents = int(split_event["split_parents"])
        children = int(split_event["children"])
        net_growth = int(split_event["net_growth"])
        final_to_old = split_event["_new_to_old"]
    else:
        instance_split_counts = {}
        selected_spatial_group_count = 0
        role_growth_counts = {
            "static_skeleton": 0,
            "canonical_crown": 0,
            "dynamic_leaf": 0,
        }
        net_growth = 0
        selection_audit = {
            "radius_pixels_minimum": 0.0,
            "radius_pixels_median": 0.0,
            "radius_pixels_p90": 0.0,
            "radius_pixels_maximum": 0.0,
            "coverage_deficit_median": 0.0,
            "generation_median": 0.0,
            "generation_maximum": 0,
            "binary_split_parent_count": 0,
            "quaternary_split_parent_count": 0,
            "net_growth_slots": 0,
            "dynamic_owner_context": {
                "eligible_owner_camera_count": 0,
                "selected_owner_camera_count": 0,
                "splits_per_selected_camera_minimum": 0,
                "splits_per_selected_camera_median": 0.0,
                "splits_per_selected_camera_maximum": 0,
                "eligible_owner_camera_ids": [],
                "selected_owner_camera_ids": [],
            },
            "initialization_source_split_parents": {},
            "exact_ray_camera_plane_split_parents": 0,
            "camera_plane_mutated_parents": 0,
        }
        prune_event = foliage.replace_and_split_adaptive(
            remove,
            torch.empty(
                0, dtype=torch.long, device=foliage.xyz.device
            ),
            torch.empty(
                0, dtype=torch.long, device=foliage.xyz.device
            ),
            allow_static_skeleton=True,
            birth_iteration=int(current_iteration),
        )
        split_child_start = int(prune_event.pop("_child_start"))
        split_child_owner_camera_ids = torch.empty(
            0, dtype=torch.long, device=foliage.xyz.device
        )
        final_to_old = prune_event["_new_to_old"]
    owner_color_refresh = _refresh_split_child_owner_colors(
        foliage,
        child_start=split_child_start,
        owner_camera_ids=split_child_owner_camera_ids,
        view_by_camera_id=(view_by_camera_id or {}),
        support_fallback_enabled=bool(
            getattr(args, "canonical_child_support_color_refresh", True)
        ),
        geometry_evidence=geometry_evidence,
        foliage_ray_evidence=foliage_ray_evidence,
    )
    mutated = pruned > 0 or split_parents > 0
    return {
        "old_count": old_count,
        "new_count": len(foliage),
        "split_parents": split_parents,
        "children": children,
        "net_growth": net_growth,
        "pruned": pruned,
        "contradiction_pruned": int(contradiction_remove.sum()),
        "contradiction_prune_evidence": {
            "contract": (
                "strong_confirmed_free_overrides_observation_identity__"
                "weak_low_utility_conflict_protects_only_independent_"
                "cross_sequence_positive_support"
            ),
            "strong_contradiction_candidates": int(
                strong_contradiction.sum()
            ),
            "weak_contradiction_candidates": int(
                weak_contradiction.sum()
            ),
            "independently_verified_positive_rows": int(
                independently_verified_positive.sum()
            ),
            "weak_positive_rows_protected": int(
                weak_positive_protected.sum()
            ),
            "static_skeleton_rows_protected": int(
                static_skeleton_contradiction_protected.sum()
            ),
            "observed_identity_rows": int(observed_identity.sum()),
            "observed_unverified_rows_prunable": int(
                observed_unverified_prunable.sum()
            ),
        },
        "reallocated_pruned": int(reallocation_remove.sum()),
        "capacity_reallocation": reallocation_audit,
        "saturated_budget_settle": saturated_budget_settle,
        "capacity_reallocation_contract": VOLUME_REALLOCATION_CONTRACT,
        "topology_mutation_contract": VOLUME_TOPOLOGY_MUTATION_CONTRACT,
        "split_child_owner_color_refresh": owner_color_refresh,
        "integrated_split_demand": float(integrated_demand),
        "requested_splits": int(requested_splits),
        "configured_split_limit": int(args.maximum_volume_splits),
        "effective_split_limit": effective_split_limit,
        "topology_ramp_scale": split_capacity_scale,
        "verification_debt": {
            "contract": (
                "continuous_unverified_population_backpressure_with_nonzero_"
                "ordinary_split_floor__failed_families_rollback_after_timeout"
            ),
            "unverified_rows": int(unverified_before.sum()),
            "fraction": verification_debt_fraction,
            "soft_fraction": debt_soft,
            "hard_fraction": debt_hard,
            "ordinary_split_capacity_scale": (
                verification_capacity_scale
            ),
        },
        "budget": int(effective_budget),
        "remaining": int(max(effective_budget - len(foliage), 0)),
        "role_split_parents": role_split_counts,
        "static_canonical_split_scope": (
            "static_detail_only_after_envelope_stage"
            if static_detail_exclusive_topology
            else "envelope_and_detail"
        ),
        "role_net_growth": role_growth_counts,
        "tree_instance_split_parents": instance_split_counts,
        "selected_spatial_group_count": selected_spatial_group_count,
        "eligible_count_by_role": eligible_counts,
        "dynamic_topology_evidence": {
            "contract": (
                "persistent_owner_plus_exact_conditioned_screen_gradient"
                if has_exact_conditioned_stats
                else "legacy_auxiliary_observation_sample"
            ),
            "persistent_owner_rows": int(persistent_observation.sum()),
            "conditioned_visible_rows": int(conditioned_evidence.sum()),
            "conditioned_nonzero_gradient_rows": int(
                (conditioned_gradient > 0).sum()
            ),
            "exact_ray_bandwidth_rows": int(exact_ray_bandwidth.sum()),
            "exact_ray_eligible_rows": int(
                (dynamic_eligible & exact_ray_bandwidth).sum()
            ),
            "geometry_authority_contract": (
                "metric_depth_authority_for_3d_growth__unit_authority_for_"
                "depth_preserving_exact_owner_camera_plane_subdivision"
            ),
        },
        "population_count_by_role": population_counts,
        "observable_count_by_role": observable_counts,
        "eligible_fraction_by_role": {
            role: (
                eligible_counts[role] / float(population_counts[role])
                if population_counts[role]
                else 0.0
            )
            for role in eligible_counts
        },
        "effective_eligible_demand_by_role": (
            effective_eligible_mass
        ),
        "net_growth_capacity_by_role": role_growth_capacity,
        "role_capacity_contract": (
            "absolute_authority_weighted_unresolved_projected_bandwidth__"
            "quota_and_global_budget_are_net_growth_rows"
        ),
        "mean_topology_authority_by_eligible_role": {
            role: (
                float(topology_authority[role_masks[role]].sum())
                / max(eligible_counts[role], 1)
            )
            for role in eligible_counts
        },
        "mean_unresolved_bandwidth_demand_by_eligible_role": {
            role: (
                effective_eligible_mass[role]
                / max(eligible_counts[role], 1)
            )
            for role in eligible_counts
        },
        "requested_split_quota_by_role": role_quotas,
        "split_capacity": int(capacity),
        "split_capacity_saturated": bool(
            capacity > 0 and net_growth == capacity
        ),
        "selection_audit": selection_audit,
        "child_verification_lifecycle": {
            "grace_protected_from_prune": int(verification_grace.sum()),
            "expired_candidates": int(verification_expired.sum()),
            "expired_zero_witness_low_utility_pruned": int(
                expired_unobserved_low_utility.sum()
            ),
            "unverified_before_mutation": int(
                (verification_state == VERIFICATION_UNVERIFIED).sum()
            ),
            "unverified_after_mutation": int(
                (
                    foliage.verification_state
                    == VERIFICATION_UNVERIFIED
                ).sum()
            ),
            "new_children_are_unverified": True,
            "unverified_split_eligible": 0,
            "failed_family_rollback": rollback_audit,
        },
        "selection_contract": (
            "continuous_screen_demand_then_lineage_safe_replace_retire_then_"
            "posterior_authority_weighted_observable_population_normalized_"
            "role_capacity_then_"
            "owner_context_instance_spatial_utility_then_"
            "lineage_family_then_screen_severity_adaptive_2_or_4_child_"
            "split_with_exact_ray_camera_plane_refinement"
        ),
        "phase": phase,
        "_rollback_old_rows": rollback_representatives,
        "_new_to_old": final_to_old if mutated else None,
    }


def _apply_volume_role_gradients(
    foliage, phase: str, *, dynamic_active: bool
) -> None:
    skeleton = foliage.static_skeleton_mask
    dynamic = foliage.dynamic_leaf_mask
    skeleton_confidence = getattr(
        foliage,
        "static_skeleton_confidence",
        torch.ones_like(foliage.xyz[:, 0]),
    ).clamp(0.0, 1.0)
    rigidity = (
        skeleton_confidence / (skeleton_confidence + 0.20)
    ).clamp(0.0, 1.0)
    if foliage.xyz.grad is not None:
        foliage.xyz.grad[skeleton] *= (
            0.35 - 0.27 * rigidity[skeleton]
        )[:, None]
        if phase != "canonical_bootstrap":
            foliage.xyz.grad[dynamic] *= 1.4
    if foliage.log_scales.grad is not None:
        foliage.log_scales.grad[skeleton] *= (
            0.22 - 0.17 * rigidity[skeleton]
        )[:, None]
    if foliage.opacity_logits.grad is not None:
        foliage.opacity_logits.grad[skeleton] *= (
            0.45 - 0.20 * rigidity[skeleton]
        )[:, None]
    for parameter in (
        foliage.deformation_basis,
        foliage.dynamic_feature_basis,
        foliage.dynamic_opacity_basis,
    ):
        if parameter.grad is not None:
            parameter.grad[~dynamic] = 0
            # ``canonical_polish`` is topology-stable, not a canonical-only
            # optimizer phase.  The last topology event creates/replaces
            # dynamic descendants immediately before this interval; zeroing
            # their conditioned bases here left those rows with only a few
            # appearance updates and made the final checkpoint look like an
            # unsettled topology snapshot.  Freeze these bases only before
            # the conditioned branch is actually active.
            if not dynamic_active:
                parameter.grad[dynamic] = 0


def _apply_volume_opacity_settle_policy(
    foliage,
    volume_optimizer,
    step: int,
    args,
) -> dict[str, object]:
    """Stop late RGB residuals from rebuilding an opaque low-pass canopy.

    By the settle point the exact owner/depth stage has already calibrated
    optical existence.  Geometry, covariance and SH colour still need many
    refinement steps, but unconstrained Adam opacity updates can cheaply lower
    RGB loss by increasing projected optical thickness everywhere.  That
    destroys parallax and edge bandwidth even when topology is held fixed.

    ``freeze`` skips both base and temporal opacity parameters.  The less
    restrictive ``retirement_only`` policy keeps positive base-logit
    gradients (Adam decreases opacity) but removes gradients and Adam momentum
    that would increase it, only through the configured bounded retirement
    window; it automatically becomes ``freeze`` afterwards.  Temporal opacity
    is frozen in both policies because its parameter group also owns
    deformation/colour bases and cannot be safely controlled by changing the
    shared group learning rate.
    """

    policy = str(args.volume_opacity_settle_policy)
    iteration = int(step) + 1
    start = int(args.volume_opacity_settle_start_iteration)
    retirement_end = int(
        args.volume_opacity_retirement_until_iteration
    )
    effective_policy = (
        "freeze"
        if policy == "retirement_only" and iteration > retirement_end
        else policy
    )
    audit: dict[str, object] = {
        "contract": VOLUME_OPACITY_SETTLE_CONTRACT,
        "policy": policy,
        "effective_policy": effective_policy,
        "start_iteration": start,
        "retirement_until_iteration": retirement_end,
        "active": bool(policy != "none" and iteration > start),
        "base_rows_frozen": 0,
        "base_growth_rows_suppressed": 0,
        "base_growth_gradient_suppressed": 0.0,
        "base_growth_momentum_entries_suppressed": 0,
        "temporal_rows_frozen": 0,
    }
    if not audit["active"]:
        return audit

    base_gradient = foliage.opacity_logits.grad
    temporal_gradient = foliage.dynamic_opacity_basis.grad
    if effective_policy == "freeze":
        if base_gradient is not None:
            audit["base_rows_frozen"] = int(
                (base_gradient.flatten(1).abs().sum(dim=1) > 0).sum()
            )
            foliage.opacity_logits.grad = None
    elif effective_policy == "retirement_only":
        if base_gradient is not None:
            increasing = base_gradient < 0
            audit["base_growth_rows_suppressed"] = int(
                increasing.flatten(1).any(dim=1).sum()
            )
            audit["base_growth_gradient_suppressed"] = float(
                (-base_gradient[increasing]).sum()
            )
            base_gradient.clamp_(min=0)
        # A non-negative current gradient is insufficient with Adam: a
        # negative first moment still increases the logit.  Constrain that
        # stored growth direction as part of the same optical contract.
        state = volume_optimizer.state.get(foliage.opacity_logits, {})
        first_moment = state.get("exp_avg")
        if torch.is_tensor(first_moment):
            increasing_momentum = first_moment < 0
            audit["base_growth_momentum_entries_suppressed"] = int(
                increasing_momentum.sum()
            )
            first_moment.clamp_(min=0)
    else:
        raise ValueError(
            f"Unknown volume opacity settle policy {effective_policy!r}"
        )

    if temporal_gradient is not None:
        audit["temporal_rows_frozen"] = int(
            (temporal_gradient.flatten(1).abs().sum(dim=1) > 0).sum()
        )
        # ``grad=None`` makes Adam skip this parameter entirely, including
        # its retained moments, while deformation and feature bases in the
        # same named group continue to update.
        foliage.dynamic_opacity_basis.grad = None
    return audit


@torch.no_grad()
def _measure_static_ray_local_replacement(
    view,
    surface,
    foliage,
    background: torch.Tensor,
    task: dict[str, torch.Tensor],
    _candidate_package,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, float | int | str],
]:
    """Measure detail takeover from real per-pixel mixed-kernel weights.

    Two role-isolated native mixed renders establish the actual alpha/depth
    explanations.  A third envelope render asks the existing CUDA audit path
    to integrate ``T_before * alpha * q(pixel)`` per primitive.  The old
    Persistent replacement groups and exact support-camera ids establish the
    local candidate. Projected depth, transmittance and coverage authority
    come only from real pixels in the native mixed renderer; a primitive-
    centre proxy is neither necessary nor allowed to veto those pixels.
    """
    envelope_mask = foliage.persistent_envelope_mask
    detail_mask = foliage.static_leaf_mask & (
        foliage.verification_state == VERIFICATION_VERIFIED
    ) & (foliage.verified_camera_count >= 2)
    verified_envelope_mask = envelope_mask & (
        foliage.verification_state == VERIFICATION_VERIFIED
    )
    empty = foliage.xyz.new_zeros(len(foliage))
    if not bool(envelope_mask.any()) or not bool(detail_mask.any()):
        return empty, envelope_mask & False, envelope_mask & False, {
            "contract": "native_t_before_alpha_ray_local_replacement",
            "scheduled": True,
            "supported_pixels": 0,
            "observed_envelope_rows": 0,
            "mean_pixel_authority": 0.0,
        }

    def role_render(mask, audit_fields=None):
        return render_hybrid(
            view,
            surface,
            foliage,
            background=background,
            include_dynamic=False,
            volume_role_mask=mask,
            optical_replacement_policy="disabled",
            structural_trainable_start=None,
            audit_fields=audit_fields,
        )

    envelope = role_render(envelope_mask)
    detail = role_render(detail_mask)
    epsilon = 1.0e-6
    envelope_alpha = envelope.volume_alpha.clamp(0.0, 1.0 - epsilon)
    detail_alpha = detail.volume_alpha.clamp(0.0, 1.0 - epsilon)
    envelope_tau = -torch.log1p(-envelope_alpha)
    detail_tau = -torch.log1p(-detail_alpha)
    coverage = (detail_tau / envelope_tau.clamp_min(epsilon)).clamp(0.0, 1.0)
    both = (envelope_tau > epsilon) & (detail_tau > epsilon)
    depth_scale = torch.maximum(
        0.04
        * torch.minimum(envelope.volume_depth, detail.volume_depth).abs(),
        envelope_tau.new_full(envelope_tau.shape, 0.04),
    )
    depth_residual = (
        envelope.volume_depth - detail.volume_depth
    ).abs() / depth_scale.clamp_min(epsilon)
    depth_compatibility = torch.exp(-0.5 * depth_residual.square())
    depth_compatibility = torch.where(
        both
        & torch.isfinite(envelope.volume_depth)
        & torch.isfinite(detail.volume_depth),
        depth_compatibility,
        torch.zeros_like(depth_compatibility),
    )
    canopy_core = task["p_canopy_core"].clamp(0.0, 1.0)
    rigid_probability = task["p_rigid"].clamp(0.0, 1.0)
    if canopy_core.ndim == 2:
        canopy_core = canopy_core.unsqueeze(0)
    if rigid_probability.ndim == 2:
        rigid_probability = rigid_probability.unsqueeze(0)
    rigid_safe = canopy_core * (1.0 - rigid_probability)
    pixel_authority = (
        coverage * depth_compatibility * rigid_safe
    ).clamp(0.0, 1.0)
    audit_fields = torch.cat(
        [pixel_authority, coverage, depth_compatibility, rigid_safe], dim=0
    ).contiguous()
    audited = role_render(envelope_mask, audit_fields=audit_fields)
    responsibility = audited.responsibility[
        audited.structural_count :
    ]
    total = responsibility[:, 0].clamp_min(epsilon)
    authority = (responsibility[:, 1] / total).clamp(0.0, 1.0)
    visible = verified_envelope_mask & (
        responsibility[:, 0] > epsilon
    )
    groups = foliage.replacement_group.long()
    valid_group = groups >= 0
    supported_detail = detail_mask & valid_group & (
        foliage.support_camera_ids == int(view.colmap_id)
    ).any(dim=1)
    group_has_supported_detail = torch.zeros(
        int(groups[valid_group].max()) + 1 if bool(valid_group.any()) else 0,
        dtype=torch.int32,
        device=groups.device,
    )
    if bool(supported_detail.any()):
        group_has_supported_detail.index_add_(
            0,
            groups[supported_detail],
            torch.ones(
                int(supported_detail.sum()),
                dtype=torch.int32,
                device=groups.device,
            ),
        )
    candidate_mask = torch.zeros_like(envelope_mask)
    if len(group_has_supported_detail):
        candidate_mask[valid_group] = (
            group_has_supported_detail[groups[valid_group]] > 0
        )
    candidate_visible = visible & candidate_mask
    # Real coverage, depth, transmittance and rigid safety all come from pixel
    # contributions. A supported row with zero authority is explicit negative
    # evidence for retirement, not an unrelated-view observation.
    authority = torch.where(
        candidate_visible, authority, torch.zeros_like(authority)
    )
    supported = pixel_authority > 0.05
    audit = {
        "contract": "native_t_before_alpha_ray_local_replacement",
        "scheduled": True,
        "supported_pixels": int(supported.sum()),
        "observed_envelope_rows": int(visible.sum()),
        "support_owned_group_candidate_rows": int(candidate_visible.sum()),
        "positive_authority_rows": int((authority > 0).sum()),
        "mean_pixel_authority": float(pixel_authority[supported].mean())
        if bool(supported.any())
        else 0.0,
        "mean_row_authority": float(authority[visible].mean())
        if bool(visible.any())
        else 0.0,
        "tree_alpha_coverage": float(
            (detail_alpha > 0.01).float()[canopy_core > 0.5].mean()
        )
        if bool((canopy_core > 0.5).any())
        else 0.0,
    }
    del envelope, detail, audited, responsibility
    return authority, visible, candidate_visible, audit


@torch.no_grad()
def _accumulate_static_replacement_evidence(
    foliage,
    authority_observation: torch.Tensor,
    visible: torch.Tensor,
    *,
    candidate_visible: torch.Tensor | None = None,
    camera_id: int,
    decay: float,
) -> dict[str, float | int | str]:
    """Persist candidate-local positive and negative ray evidence.

    Seeing an envelope is not evidence that its paired detail failed to take
    over: most calibrated views never project the detail assigned to that
    local cell.  The former update nevertheless decayed every visible row and
    counted every such camera as a replacement witness.  With roughly 1,487
    cameras this drove the overlap EMA toward zero even for repeatedly
    confirmed detail, so the envelope kept all of its optical mass.

    The stored value is a conservative streaming lower coverage envelope:
    worse support views reduce it immediately, while better observations
    recover it only through an EMA.  This is the correct direction for
    preventing holes: a globally retired fraction must be safe for every
    observed support view, not merely the best one. Zero authority from a
    support-owned group is explicit counter-evidence; unrelated views do not
    touch the state.
    """
    value = torch.as_tensor(
        authority_observation,
        device=foliage.xyz.device,
        dtype=foliage.xyz.dtype,
    ).reshape(-1).clamp(0.0, 1.0)
    visible = torch.as_tensor(
        visible, device=foliage.xyz.device, dtype=torch.bool
    ).reshape(-1) & foliage.persistent_envelope_mask & (
        foliage.verification_state == VERIFICATION_VERIFIED
    )
    if candidate_visible is None:
        candidate = visible
    else:
        candidate = torch.as_tensor(
            candidate_visible,
            device=foliage.xyz.device,
            dtype=torch.bool,
        ).reshape(-1) & visible
    if len(value) != len(foliage) or len(visible) != len(foliage):
        raise ValueError("replacement evidence must align with foliage rows")
    if len(candidate) != len(foliage):
        raise ValueError("replacement candidates must align with foliage rows")
    positive = candidate & (value > 1.0e-6)
    contradicted = candidate & ~positive
    if bool(candidate.any()):
        previous = foliage.replacement_overlap_ema[candidate]
        observed = value[candidate]
        first = foliage.replacement_observation_count[candidate] <= 0
        recovered = previous + (1.0 - float(decay)) * (
            observed - previous
        )
        updated = torch.where(
            first,
            observed,
            torch.where(observed < previous, observed, recovered),
        )
        foliage.replacement_overlap_ema[candidate] = updated
        bit = int(1) << (int(camera_id) % 62)
        old_signature = foliage.replacement_camera_signature[candidate]
        new_camera = (old_signature & bit) == 0
        foliage.replacement_camera_signature[candidate] = old_signature | bit
        count = foliage.replacement_observation_count[candidate].to(torch.int32)
        foliage.replacement_observation_count[candidate] = (
            count.add(new_camera.to(torch.int32))
            .clamp_max(torch.iinfo(torch.int16).max)
            .to(torch.int16)
        )
    readiness = (
        foliage.replacement_observation_count.float() / 3.0
    ).clamp(0.0, 1.0)
    authority = foliage.replacement_overlap_ema * readiness
    envelope = foliage.persistent_envelope_mask & (
        foliage.verification_state == VERIFICATION_VERIFIED
    )
    return {
        "contract": "persistent_distinct_view_real_ray_replacement_ema",
        "observed_envelope_rows": int(visible.sum()),
        "candidate_envelope_rows": int(candidate.sum()),
        "overlapping_envelope_rows": int(positive.sum()),
        "contradicted_candidate_rows": int(contradicted.sum()),
        "mean_distinct_views": float(
            foliage.replacement_observation_count[envelope].float().mean()
        )
        if bool(envelope.any())
        else 0.0,
        "mean_authority": float(authority[envelope].mean())
        if bool(envelope.any())
        else 0.0,
        "maximum_authority": float(authority[envelope].max())
        if bool(envelope.any())
        else 0.0,
    }


@torch.no_grad()
def _apply_static_ray_local_mass_handoff(
    foliage,
    volume_optimizer,
    *,
    maximum_fraction_per_event: float,
) -> dict[str, float | int | str]:
    """Move envelope mass only after persistent real-ray takeover evidence.

    The transition is reversible: if later calibrated views lose detail
    coverage, the reference envelope mass is restored gradually.  This is
    what prevents both duplicate fog and one-way transparent holes.
    """
    all_envelope = foliage.persistent_envelope_mask
    envelope = all_envelope & (
        foliage.verification_state == VERIFICATION_VERIFIED
    )
    previous = foliage.handoff_retired_fraction.clamp(0.0, 0.95)
    illegally_retired = all_envelope & ~envelope & (previous > 0)
    if (
        not bool(envelope.any())
        and not bool(illegally_retired.any())
    ) or maximum_fraction_per_event < 0:
        return {
            "contract": "reversible_integrated_optical_mass_handoff",
            "changed_rows": 0,
            "repaired_ineligible_rows": 0,
            "removed_mass": 0.0,
            "restored_mass": 0.0,
        }
    readiness = (
        foliage.replacement_observation_count.float() / 3.0
    ).clamp(0.0, 1.0)
    desired = (
        foliage.replacement_overlap_ema * readiness * 0.95
    ).clamp(0.0, 0.95)
    delta = (desired - previous).clamp(
        -float(maximum_fraction_per_event),
        float(maximum_fraction_per_event),
    )
    updated = torch.where(envelope, previous + delta, previous).clone()
    # No unverified child is allowed to carry retirement state.  This branch
    # is normally a no-op; it also repairs a checkpoint produced before the
    # verification owner was added to the handoff contract.
    updated[illegally_retired] = 0
    changed = (envelope & (delta.abs() > 1.0e-7)) | illegally_retired
    current_mass = foliage.integrated_optical_mass()
    reference = foliage.handoff_reference_mass.clamp_min(0.0)
    invalid_reference = all_envelope & (
        ~torch.isfinite(reference) | (reference <= 0)
    )
    reference[invalid_reference] = (
        current_mass[invalid_reference]
        / (1.0 - previous[invalid_reference]).clamp_min(1.0e-4)
    )
    target = current_mass.clone()
    target[changed] = reference[changed] * (1.0 - updated[changed])
    removed = (current_mass - target).clamp_min(0.0)
    restored = (target - current_mass).clamp_min(0.0)
    foliage.restore_integrated_optical_mass(target)
    foliage.handoff_retired_fraction.copy_(updated)
    state = volume_optimizer.state.get(foliage.opacity_logits, {})
    for value in state.values():
        if torch.is_tensor(value) and value.shape == foliage.opacity_logits.shape:
            value[changed] = 0
    return {
        "contract": "reversible_integrated_optical_mass_handoff",
        "changed_rows": int(changed.sum()),
        "repaired_ineligible_rows": int(illegally_retired.sum()),
        "removed_mass": float(removed.sum()),
        "restored_mass": float(restored.sum()),
        "mean_retired_fraction": float(updated[envelope].mean())
        if bool(envelope.any())
        else 0.0,
        "maximum_retired_fraction": float(updated[envelope].max())
        if bool(envelope.any())
        else 0.0,
    }


@torch.no_grad()
def _finalize_static_ray_local_mass_handoff(
    foliage,
    volume_optimizer,
    *,
    scheduled: bool,
    maximum_fraction_per_event: float,
) -> dict[str, float | int | str | bool]:
    """Fold the optimizer update into reference mass, then transfer mass.

    This helper is deliberately called only after ``volume_optimizer.step``
    and scale/mass compensation.  The first synchronization preserves valid
    new hit evidence in the unretired reference; the second records the new
    reversible retirement state.
    """
    realized_before = foliage.integrated_optical_mass()
    retained_before = (1.0 - foliage.handoff_retired_fraction).clamp_min(
        1.0e-4
    )
    foliage.handoff_reference_mass.copy_(
        realized_before / retained_before
    )
    audit: dict[str, float | int | str | bool]
    if scheduled:
        audit = _apply_static_ray_local_mass_handoff(
            foliage,
            volume_optimizer,
            maximum_fraction_per_event=maximum_fraction_per_event,
        )
    else:
        audit = {
            "contract": "reversible_integrated_optical_mass_handoff",
            "changed_rows": 0,
            "removed_mass": 0.0,
            "restored_mass": 0.0,
            "scheduled_after_optimizer_step": False,
        }
    realized_after = foliage.integrated_optical_mass()
    retained_after = (1.0 - foliage.handoff_retired_fraction).clamp_min(
        1.0e-4
    )
    foliage.handoff_reference_mass.copy_(
        realized_after / retained_after
    )
    audit["scheduled_after_optimizer_step"] = bool(scheduled)
    return audit


@torch.no_grad()
def _initialize_static_ray_birth_mass_handoff(
    foliage,
    volume_optimizer,
    *,
    birth_rows: torch.Tensor,
    owner_rows: torch.Tensor,
    view_by_camera_id: dict[int, object],
    maximum_fraction_per_event: float,
    maximum_additive_fraction_per_event: float = 0.005,
) -> dict[str, float | int | str]:
    """Fund newborn detail mass from its ray-local envelope owner.

    A cross-sequence ray proposal proves that a detail hypothesis is worth
    materializing; it does not authorize adding a second extinction layer on
    top of the persistent envelope.  For every assigned birth, evaluate its
    nearest same-tree envelope owner in all retained support cameras.  Only the
    conservative minimum image/depth overlap may move optical mass.  The
    newborn is then initialized with exactly the mass removed from the owner.
    A strict cross-camera/cross-sequence visual-hull proposal without a usable
    donor cannot learn if it is appended and immediately made transparent. It
    therefore receives a small explicit additive budget, bounded as a fraction
    of the pre-birth model mass for the whole event. A projected donor funds
    the overlapping component conservatively; any remaining visual-hull
    existence mass shares the same explicit event budget. Ordinary one-view
    hypotheses are never eligible for this fallback.

    The owner's pre-transfer reference mass and retired fraction make this
    reversible through :func:`_apply_static_ray_local_mass_handoff`: if the
    child later fails verification or is pruned, zero replacement authority
    restores the transferred envelope mass without a background hole.
    """

    birth_rows = torch.as_tensor(
        birth_rows, device=foliage.xyz.device, dtype=torch.long
    ).reshape(-1)
    owner_rows = torch.as_tensor(
        owner_rows, device=foliage.xyz.device, dtype=torch.long
    ).reshape(-1)
    if len(birth_rows) != len(owner_rows):
        raise ValueError("Ray-birth rows and envelope owners must align")
    if not 0.0 <= float(maximum_fraction_per_event) <= 0.10:
        raise ValueError("Initial ray-birth transfer fraction must lie in [0,0.1]")
    if not 0.0 <= float(maximum_additive_fraction_per_event) <= 0.02:
        raise ValueError(
            "Initial ray-birth additive fraction must lie in [0,0.02]"
        )
    empty_audit = {
        "contract": "support_ray_local_conservative_initial_mass_transfer",
        "candidates": int(len(birth_rows)),
        "ray_local_candidates": 0,
        "changed_owner_rows": 0,
        "requested_child_mass": 0.0,
        "retained_child_mass": 0.0,
        "retired_envelope_mass": 0.0,
        "additive_child_mass": 0.0,
        "additive_funded_children": 0,
        "additive_mass_budget": 0.0,
        "donor_funded_children": 0,
        "fully_donor_funded_children": 0,
        "unfunded_child_mass": 0.0,
        "mass_before_birth": float(foliage.integrated_optical_mass().sum()),
        "mass_after_transfer": float(foliage.integrated_optical_mass().sum()),
    }
    if not len(birth_rows):
        return empty_audit
    if bool(
        (birth_rows < 0).any()
        or (birth_rows >= len(foliage)).any()
        or (owner_rows >= len(foliage)).any()
    ):
        raise ValueError("Ray-birth transfer rows are outside the foliage model")
    if not bool(foliage.static_leaf_mask[birth_rows].all()):
        raise ValueError("Initial ray-birth transfer requires static detail rows")
    safe_owner_rows = owner_rows.clamp(0, max(len(foliage) - 1, 0))
    valid_owner = (
        (owner_rows >= 0)
        & foliage.persistent_envelope_mask[safe_owner_rows]
        & (
            foliage.verification_state[safe_owner_rows]
            == VERIFICATION_VERIFIED
        )
        & (foliage.replacement_group[birth_rows] >= 0)
        & (
            foliage.replacement_group[birth_rows]
            == foliage.replacement_group[safe_owner_rows]
        )
    )

    mass_before = foliage.integrated_optical_mass()
    pre_birth_total = mass_before.sum() - mass_before[birth_rows].sum()
    authority = mass_before.new_ones(len(birth_rows))
    valid_camera_count = torch.zeros(
        len(birth_rows), dtype=torch.int16, device=foliage.xyz.device
    )
    support = foliage.support_camera_ids[birth_rows]
    support_count = (support >= 0).sum(dim=1)
    unique_camera_ids = torch.unique(support[support >= 0]).tolist()
    rotations = _rotation_matrices_from_quaternions(
        foliage.normalized_quaternions
    )
    area = projected_gaussian_cross_section(foliage.scales)

    for camera_id_value in unique_camera_ids:
        camera_id = int(camera_id_value)
        view = view_by_camera_id.get(camera_id)
        local = torch.nonzero(
            valid_owner & (support == camera_id).any(dim=1),
            as_tuple=False,
        ).flatten()
        if view is None or not len(local):
            continue
        detail_rows = birth_rows[local]
        envelope_rows = owner_rows[local]
        pair_rows = torch.cat([envelope_rows, detail_rows])
        homogeneous = torch.cat(
            [
                foliage.xyz[pair_rows],
                torch.ones_like(foliage.xyz[pair_rows, :1]),
            ],
            dim=1,
        )
        camera_points = homogeneous @ view.world_view_transform.to(
            device=foliage.xyz.device, dtype=foliage.xyz.dtype
        )
        depth = camera_points[:, 2]
        inverse_depth = depth.clamp_min(1.0e-8).reciprocal()
        projected_xy = torch.stack(
            [
                float(view.focal_x) * camera_points[:, 0] * inverse_depth
                + float(view.cx),
                float(view.focal_y) * camera_points[:, 1] * inverse_depth
                + float(view.cy),
            ],
            dim=1,
        )
        pair_count = len(local)
        owner_xy, detail_xy = projected_xy.split(pair_count)
        owner_depth, detail_depth = depth.split(pair_count)
        focal = max(
            (float(view.focal_x) * float(view.focal_y)) ** 0.5,
            1.0e-6,
        )
        owner_radius = (
            area[envelope_rows].clamp_min(1.0e-12).sqrt()
            * focal
            / owner_depth.abs().clamp_min(1.0e-6)
        )
        detail_radius = (
            area[detail_rows].clamp_min(1.0e-12).sqrt()
            * focal
            / detail_depth.abs().clamp_min(1.0e-6)
        )
        pixel_sigma = (owner_radius + detail_radius).clamp_min(1.0)
        pixel_distance2 = (
            (owner_xy - detail_xy).square().sum(dim=1)
            / pixel_sigma.square()
        )
        camera_depth_axis = view.world_view_transform[:3, 2].to(
            device=foliage.xyz.device, dtype=foliage.xyz.dtype
        )
        camera_depth_axis = camera_depth_axis / camera_depth_axis.norm().clamp_min(
            1.0e-8
        )
        pair_rotation = rotations[pair_rows]
        local_depth_axis = torch.bmm(
            pair_rotation.transpose(1, 2),
            camera_depth_axis.expand(len(pair_rows), -1)[:, :, None],
        ).squeeze(-1)
        depth_radius = torch.sqrt(
            (
                local_depth_axis.square()
                * foliage.scales[pair_rows].square()
            ).sum(dim=1).clamp_min(1.0e-12)
        )
        owner_depth_radius, detail_depth_radius = depth_radius.split(pair_count)
        depth_sigma = (owner_depth_radius + detail_depth_radius).clamp_min(0.01)
        canonical_in_front = (
            detail_depth - owner_depth - 2.5 * depth_sigma
        ).clamp_min(0.0)
        depth_distance2 = canonical_in_front.square() / (
            2.5 * depth_sigma
        ).square()
        coverage_ceiling = (
            owner_radius.square()
            / detail_radius.square().clamp_min(1.0e-12)
        ).clamp(max=1.0)
        overlap = torch.exp(-0.5 * (pixel_distance2 + depth_distance2))
        overlap *= coverage_ceiling
        valid_projection = (
            torch.isfinite(owner_xy).all(dim=1)
            & torch.isfinite(detail_xy).all(dim=1)
            & torch.isfinite(owner_depth)
            & torch.isfinite(detail_depth)
            & (owner_depth > 0.05)
            & (detail_depth > 0.05)
        )
        overlap = torch.where(
            valid_projection, overlap, torch.zeros_like(overlap)
        )
        owner_area = owner_radius.square()
        detail_area = detail_radius.square()
        intersection = torch.minimum(
            detail_area * overlap,
            torch.minimum(owner_area, detail_area),
        )
        projected_iou = intersection / (
            owner_area + detail_area - intersection
        ).clamp_min(1.0e-8)
        boundary_safe = (2.5 * projected_iou).clamp(0.0, 1.0)
        pair_authority = torch.minimum(overlap, boundary_safe).clamp(0.0, 0.5)
        authority[local] = torch.minimum(authority[local], pair_authority)
        valid_camera_count[local] += valid_projection.to(torch.int16)

    required_camera_count = torch.minimum(
        support_count.clamp_min(1),
        torch.full_like(support_count, 2),
    )
    ray_local = valid_owner & (valid_camera_count >= required_camera_count)
    authority = torch.where(ray_local, authority, torch.zeros_like(authority))
    requested = mass_before[birth_rows] * authority

    assigned_local = torch.nonzero(valid_owner, as_tuple=False).flatten()
    unique_owner, owner_inverse = torch.unique(
        owner_rows[assigned_local], sorted=True, return_inverse=True
    )
    requested_by_owner = mass_before.new_zeros(len(unique_owner))
    requested_by_owner.scatter_add_(
        0, owner_inverse, requested[assigned_local]
    )
    previous_fraction = foliage.handoff_retired_fraction[unique_owner].clamp(
        0.0, 0.95
    )
    reference = foliage.handoff_reference_mass[unique_owner].clone()
    invalid_reference = ~torch.isfinite(reference) | (reference <= 0)
    reference[invalid_reference] = (
        mass_before[unique_owner][invalid_reference]
        / (1.0 - previous_fraction[invalid_reference]).clamp_min(1.0e-4)
    )
    # One birth event can only spend the same bounded fraction used by the
    # subsequent reversible hand-off.  At least five percent of every envelope
    # reference remains, so compact detail can never open an immediate hole.
    available = (
        mass_before[unique_owner] - 0.05 * reference
    ).clamp_min(0.0)
    event_capacity = reference * float(maximum_fraction_per_event)
    owner_capacity = torch.minimum(available, event_capacity)
    owner_scale = (
        owner_capacity / requested_by_owner.clamp_min(1.0e-12)
    ).clamp(0.0, 1.0)
    transfer = mass_before.new_zeros(len(birth_rows))
    transfer[assigned_local] = (
        requested[assigned_local] * owner_scale[owner_inverse]
    )
    transferred_by_owner = mass_before.new_zeros(len(unique_owner))
    transferred_by_owner.scatter_add_(
        0, owner_inverse, transfer[assigned_local]
    )

    # Visual-hull cells were promoted only after independent cameras and
    # sequences intersected their calibrated hit intervals. When no projected
    # donor exists, keeping their requested alpha at ~1e-12 creates an
    # irreversible dead point: it cannot contribute, verify or receive a useful
    # RGB gradient, while the consumed-cell tombstone prevents a retry. Give
    # only those strong ownerless/non-overlap proposals a bounded event-level
    # existence budget. A common scale preserves relative posterior strength.
    strong_visual_hull = (
        (support_count >= 2)
        & (foliage.support_sequence_count[birth_rows] >= 2)
    )
    # ``ray_local`` only means that both centres project validly; it does not
    # mean the donor actually funded the birth. In real Cambridge events most
    # assigned rows had sub-percent overlap and were still reduced to nearly
    # transparent points. Allocate the *unfunded remainder* of every strict
    # visual-hull proposal under one shared cap. The donor-transferred component
    # remains exactly conserved and is never counted twice.
    additive_eligible = strong_visual_hull & (
        transfer < mass_before[birth_rows] * (1.0 - 1.0e-6)
    )
    additive_requested = (
        mass_before[birth_rows] - transfer
    ).clamp_min(0.0) * additive_eligible.to(mass_before.dtype)
    additive_budget = pre_birth_total.clamp_min(0.0) * float(
        maximum_additive_fraction_per_event
    )
    additive_scale = torch.minimum(
        additive_budget / additive_requested.sum().clamp_min(1.0e-12),
        additive_budget.new_tensor(1.0),
    )
    additive = additive_requested * additive_scale

    target_mass = mass_before.clone()
    target_mass[birth_rows] = transfer + additive
    target_mass[unique_owner] = (
        mass_before[unique_owner] - transferred_by_owner
    ).clamp_min(0.0)
    foliage.restore_integrated_optical_mass(
        target_mass, minimum_opacity=1.0e-12
    )
    realized = foliage.integrated_optical_mass()
    foliage.handoff_reference_mass[unique_owner] = reference
    foliage.handoff_retired_fraction[unique_owner] = (
        1.0 - realized[unique_owner] / reference.clamp_min(1.0e-12)
    ).clamp(0.0, 0.95)
    foliage.handoff_reference_mass[birth_rows] = realized[birth_rows]
    foliage.handoff_retired_fraction[birth_rows] = 0

    changed_rows = torch.unique(torch.cat([birth_rows, unique_owner]))
    state = volume_optimizer.state.get(foliage.opacity_logits, {})
    for value in state.values():
        if torch.is_tensor(value) and value.shape == foliage.opacity_logits.shape:
            value[changed_rows] = 0
    after_total = realized.sum()
    tolerance = max(float(pre_birth_total.abs()), 1.0) * 1.0e-6
    realized_increase = after_total - pre_birth_total
    if float(realized_increase - additive.sum()) > tolerance:
        raise RuntimeError(
            "Initial ray-birth hand-off exceeded its explicit additive budget"
        )
    return {
        "contract": "support_ray_local_conservative_initial_mass_transfer",
        "candidates": int(len(birth_rows)),
        "ray_local_candidates": int(ray_local.sum()),
        "changed_owner_rows": int((transferred_by_owner > 0).sum()),
        "requested_child_mass": float(requested.sum()),
        "retained_child_mass": float(realized[birth_rows].sum()),
        "retired_envelope_mass": float(
            (mass_before[unique_owner] - realized[unique_owner]).sum()
        ),
        "additive_child_mass": float(additive.sum()),
        "additive_funded_children": int((additive > 0).sum()),
        "additive_mass_budget": float(additive_budget),
        "donor_funded_children": int((transfer > 0).sum()),
        "fully_donor_funded_children": int(
            (transfer >= mass_before[birth_rows] * (1.0 - 1.0e-6)).sum()
        ),
        "unfunded_child_mass": float(
            (mass_before[birth_rows] - realized[birth_rows]).clamp_min(0).sum()
        ),
        "mass_before_birth": float(pre_birth_total),
        "mass_after_transfer": float(after_total),
    }


@torch.no_grad()
def _update_static_child_verification_from_render(
    foliage,
    package,
    *,
    camera_id: int,
    camera_sequence_lookup: torch.Tensor | None,
    volume_optimizer=None,
) -> dict[str, int | str]:
    """Re-verify topology children from native real-ray contribution.

    Parent camera tables remain candidate rays only.  A candidate camera is
    recorded once when the displaced child really contributes to a canopy
    pixel and does not predominantly explain rigid content.  Verification
    then requires two cameras and every independent sequence available to the
    parent (up to two); no repeated visit can manufacture multiview support.
    """
    unverified = (
        foliage.verification_state == VERIFICATION_UNVERIFIED
    ) & ~foliage.dynamic_leaf_mask
    if not bool(unverified.any()) or package.responsibility is None:
        return {
            "contract": "real_ray_child_reverification",
            "candidate_rows": int(unverified.sum()),
            "new_camera_witnesses": 0,
            "new_color_witnesses": 0,
            "newly_verified": 0,
        }
    volume = package.responsibility[package.structural_count :]
    if len(volume) != len(foliage) or volume.shape[1] < 3:
        raise RuntimeError("Child verification responsibility is incomplete")
    total = volume[:, 0].clamp_min(1.0e-8)
    canopy_fraction = volume[:, 1] / total
    rigid_fraction = volume[:, 2] / total
    candidate_camera = (foliage.support_camera_ids == int(camera_id)).any(
        dim=1
    ) if foliage.support_camera_ids.shape[1] else torch.zeros_like(unverified)
    witnessed = (
        unverified
        & candidate_camera
        & (volume[:, 0] > 1.0e-5)
        & (canopy_fraction > rigid_fraction)
    )
    rows = torch.nonzero(witnessed, as_tuple=False).flatten()
    added = 0
    color_witnesses = 0
    if len(rows) and foliage.verified_camera_ids.shape[1]:
        table = foliage.verified_camera_ids[rows]
        already = (table == int(camera_id)).any(dim=1)
        free = table < 0
        has_free = free.any(dim=1)
        insert_local = (~already) & has_free
        if bool(insert_local.any()):
            insert_rows = rows[insert_local]
            insert_slots = free[insert_local].to(torch.int8).argmax(dim=1)
            previous_count = foliage.verified_camera_count[
                insert_rows
            ].float()
            foliage.verified_camera_ids[
                insert_rows, insert_slots
            ] = int(camera_id)
            added = int(len(insert_rows))
            # The main native render appends the observed full-resolution RGB
            # as audit fields 3:6.  Its per-primitive responsibility is
            # exactly sum(T_before * alpha * RGB), so division by the ordinary
            # responsibility yields a geometrically visible colour witness.
            # Use a clipped online mean: the first witness replaces the stale
            # parent DC; later independent views refine it without allowing a
            # single highlight or exposure outlier to create a bright speck.
            if volume.shape[1] >= 7:
                observed_rgb = (
                    volume[insert_rows, 4:7]
                    / total[insert_rows, None]
                ).clamp(0.0, 1.0)
                dc_constant = 0.28209479177387814
                current_rgb = (
                    foliage.features[insert_rows, 0] * dc_constant + 0.5
                ).clamp(0.0, 1.0)
                first = previous_count <= 0
                robust_delta = (observed_rgb - current_rgb).clamp(
                    -0.20, 0.20
                )
                updated_rgb = current_rgb + robust_delta / (
                    previous_count[:, None] + 1.0
                )
                updated_rgb[first] = observed_rgb[first]
                foliage.features[insert_rows, 0] = (
                    updated_rgb - 0.5
                ) / dc_constant
                # This step's gradient was computed at the copied parent
                # colour.  Do not let it or inherited Adam state immediately
                # undo the verified full-resolution observation.
                if foliage.features.grad is not None:
                    foliage.features.grad[insert_rows] = 0
                if volume_optimizer is not None:
                    feature_state = volume_optimizer.state.get(
                        foliage.features, {}
                    )
                    for state_value in feature_state.values():
                        if (
                            torch.is_tensor(state_value)
                            and state_value.shape
                            == foliage.features.shape
                        ):
                            state_value[insert_rows] = 0
                color_witnesses = int(len(insert_rows))
    valid_verified = foliage.verified_camera_ids >= 0
    foliage.verified_camera_count.copy_(
        valid_verified.sum(dim=1).clamp_max(
            torch.iinfo(torch.int16).max
        ).to(torch.int16)
    )
    if (
        camera_sequence_lookup is not None
        and foliage.verified_camera_ids.shape[1]
        and len(camera_sequence_lookup)
    ):
        safe = foliage.verified_camera_ids.clamp(
            0, len(camera_sequence_lookup) - 1
        ).long()
        sequence = camera_sequence_lookup[safe]
        valid = valid_verified & (sequence >= 0)
        # Support tables are narrow (normally <=8); an explicit per-slot
        # comparison is cheaper and exact compared with a global unique.
        sequence_count = torch.zeros(
            len(foliage), dtype=torch.int16, device=foliage.xyz.device
        )
        for slot in range(sequence.shape[1]):
            # ``valid[:, slot]`` is a view.  Mutating it in-place corrupts the
            # persisted validity table and makes the count depend on slot
            # order, so every uniqueness accumulator must work on a clone.
            first = valid[:, slot].clone()
            for previous_slot in range(slot):
                first &= ~(
                    valid[:, previous_slot]
                    & (sequence[:, previous_slot] == sequence[:, slot])
                )
            sequence_count += first.to(torch.int16)
        foliage.verified_sequence_count.copy_(sequence_count)
    required_sequences = torch.minimum(
        foliage.support_sequence_count.clamp_min(1),
        torch.full_like(foliage.support_sequence_count, 2),
    )
    verified = (
        unverified
        & (foliage.verified_camera_count >= 2)
        & (foliage.verified_sequence_count >= required_sequences)
    )
    foliage.verification_state[verified] = VERIFICATION_VERIFIED
    # The parent evidence row was only a candidate after displacement.  Once
    # two real owner cameras (and the required independent sequences) have
    # witnessed the child at its new position, promote that candidate into
    # the child's updated ray-posterior identity.
    promotable = verified & (foliage.candidate_evidence_primitive_id >= 0)
    foliage.evidence_primitive_id[promotable] = (
        foliage.candidate_evidence_primitive_id[promotable]
    )
    foliage.candidate_evidence_primitive_id[promotable] = -1
    return {
        "contract": "real_ray_child_reverification",
        "candidate_rows": int(unverified.sum()),
        "real_contributing_candidates": int(witnessed.sum()),
        "new_camera_witnesses": int(added),
        "new_color_witnesses": int(color_witnesses),
        "newly_verified": int(verified.sum()),
        "remaining_unverified": int(
            (foliage.verification_state == VERIFICATION_UNVERIFIED).sum()
        ),
    }


def _static_detail_stage_visible(phase: str) -> bool:
    return str(phase) not in {
        "canonical_bootstrap",
        "topology",
    }


def _static_detail_stage_trainable(phase: str) -> bool:
    return str(phase) not in {
        "canonical_bootstrap",
        "topology",
        "canonical_polish",
    }


def _static_spatial_uncertainty_active(
    reconstruction_target: str,
    *,
    foliage_active: bool,
    static_detail_active: bool,
) -> bool:
    """Enable robust spatial routing without enabling a temporal model.

    The final representation remains one canonical static map.  This module
    estimates only where cross-traversal RGB is unreliable after the detail
    branch exists; it owns no geometry, opacity, RGB residual or deployment
    parameter.
    """

    return bool(
        reconstruction_target == "static"
        and foliage_active
        and static_detail_active
    )


def _static_detail_ray_trainable(phase: str) -> bool:
    """Pre-fit hidden static detail from calibrated rays from iteration one.

    Static detail is absent from bootstrap/topology RGB renders and cannot
    steal a colour explanation from the persistent envelope or rigid scene.
    Its calibrated exact-owner ray/depth posterior is nevertheless valid
    before RGB activation.  Deferring this factor consumed the first and, in
    a short causal prefix, only evidence visit of low-id cameras while the
    intended detail owner was frozen.  Keep the final polish freeze, but let
    real hit/free/depth evidence establish geometry and optical existence
    from the first scheduled ray epoch.
    """
    return str(phase) != "canonical_polish"


def _static_detail_isolated_gradient_gates(
    geometry_gate: torch.Tensor | None,
    appearance_gate: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Keep all positive static-detail RGB parameters support-owned.

    A static detail row may be visible from every camera in the exported map,
    but an unrelated traversal is not positive evidence for that row's SH.
    Routing SH from every canopy view made the numerous one-sequence leaves
    average incompatible colours even while their geometry and mass stayed
    correctly support-owned.  Negative free-space/counterfactual evidence is
    still global in its separate loss path; this gate controls only positive
    RGB supervision.
    """
    if geometry_gate is None:
        raise ValueError(
            "static detail isolated supervision requires an explicit owner gate"
        )
    if appearance_gate is None:
        appearance_gate = geometry_gate
    opacity_gate = torch.zeros_like(geometry_gate)
    return geometry_gate, appearance_gate, opacity_gate


def _static_volume_isolated_gradient_gates(
    foliage,
    ownership_gate: torch.Tensor | None,
    appearance_gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Refine detail geometry exactly and appearance softly in-sequence."""
    refinement = _static_detail_refinement_gate(foliage, ownership_gate)
    detail = refinement * foliage.static_leaf_mask.to(refinement.dtype)
    if appearance_gate is None:
        appearance = detail
    else:
        appearance = _static_detail_refinement_gate(
            foliage, appearance_gate
        ) * foliage.static_leaf_mask.to(detail.dtype)
    return detail, appearance, torch.zeros_like(detail)


def _static_stage_rgb_gradient_gates(
    foliage,
    ownership_gate: torch.Tensor | None,
    *,
    detail_stage_active: bool,
    appearance_gate: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Keep the crown envelope from absorbing Stage-3 RGB residuals.

    The persistent envelope is the broad, ray-calibrated support for crown
    depth, porosity and average transmittance.  It legitimately learns
    geometry and optical mass from RGB while that support is being formed in
    Stage 2.  Once static detail is visible, however, continuing to route the
    same RGB residual to a much larger population of wider envelope kernels
    makes increasing envelope opacity the cheapest optimization path.  The
    result is an opaque low-pass tree even when ray evidence and leaf-detail
    candidates are correct.

    Stage 3 therefore keeps envelope SH trainable but detaches its geometry
    and optical mass *only in RGB renders*.  Ray hit/free/behind factors,
    role-isolated global counterfactual cleanup and reversible local mass
    handoff operate directly on the volume parameters and retain their
    existing authority.  Skeleton rows remain RGB-trainable, while static
    detail keeps its calibrated support-sequence ownership.  Forward
    visibility is unchanged.
    """
    if appearance_gate is None:
        appearance_gate = ownership_gate
    if not detail_stage_active:
        return ownership_gate, appearance_gate, ownership_gate
    base = (
        torch.ones_like(foliage.opacities)
        if ownership_gate is None
        else torch.as_tensor(
            ownership_gate,
            device=foliage.xyz.device,
            dtype=foliage.xyz.dtype,
        ).reshape(-1).clone()
    )
    if len(base) != len(foliage):
        raise ValueError("static RGB ownership gate must align with foliage")
    envelope = foliage.persistent_envelope_mask
    geometry = base.clone()
    opacity = base.clone()
    geometry[envelope] = 0
    opacity[envelope] = 0
    # Envelope and skeleton entries in ``base`` are one and therefore retain
    # all-view SH learning. Detail entries keep their persisted support-
    # sequence owner, preventing cross-traversal colour averaging without
    # changing forward visibility.
    appearance = (
        base
        if appearance_gate is None
        else torch.as_tensor(
            appearance_gate,
            device=foliage.xyz.device,
            dtype=foliage.xyz.dtype,
        ).reshape(-1)
    )
    if len(appearance) != len(foliage):
        raise ValueError("static RGB appearance gate must align with foliage")
    return geometry, appearance, opacity


def _static_ray_candidate_masks(
    args,
    foliage,
    active_canonical_volume: torch.Tensor,
    *,
    phase: str,
    camera_id: int,
    camera_sequence_lookup: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return independent free-space and positive-hit permissions.

    Hidden static detail must be allowed to pre-fit from its calibrated hit
    rays during topology.  The former inline construction added detail only
    to the free-space mask, so the branch could be weakened before RGB
    visibility but could never receive the positive optical-existence signal
    that justified it.  Positive hits remain support-sequence-owned; global
    free-space contradictions deliberately do not.
    """
    free_mask = active_canonical_volume.clone()
    hit_mask = active_canonical_volume.clone()
    if getattr(args, "reconstruction_target", None) != "static":
        return free_mask, hit_mask
    detail = foliage.static_leaf_mask
    if _static_detail_ray_trainable(phase):
        hit_mask = hit_mask | detail
    else:
        hit_mask = hit_mask & ~detail
    if getattr(args, "static_detail_canonical_ownership", True):
        evidence_sequence_owned = (
            _static_detail_positive_evidence_sequence_gate(
                foliage,
                int(camera_id),
                camera_sequence_lookup,
            )
            > 0
        )
        # Do not apply the detail-only ownership mask to envelope/skeleton
        # rows. For detail, exact support is sufficient immediately; verified
        # two-sequence consensus is additionally a global positive-hit owner
        # of the single static map. This closes the former loop
        # ``permission miss -> uncovered hit -> 2048 more low-mass births``.
        detail_permission = evidence_sequence_owned | (
            _static_detail_consensus_hit_gate(foliage)
        )
        # The same positive-evidence sequences own both hit and free-space
        # evidence.  The former all-camera free mask let an unrelated season
        # delete a leaf that only its seed camera was allowed to restore.
        free_mask = (free_mask & ~detail) | (detail & detail_permission)
        hit_mask = hit_mask & (
            ~foliage.static_leaf_mask | detail_permission
        )
    else:
        free_mask = free_mask | detail
    return free_mask, hit_mask


def _static_detail_exclusive_topology_active(args, phase: str | None) -> bool:
    return bool(
        getattr(args, "reconstruction_target", None) == "static"
        and getattr(args, "static_detail_exclusive_topology", True)
        and phase not in {None, "canonical_bootstrap", "topology"}
    )


def _apply_surface_scale_limits_preserve_optical_mass(
    surface,
    row_indices: torch.Tensor,
    isotropic_shrink: torch.Tensor,
    *,
    maximum_scale: float,
) -> dict[str, object]:
    """Apply local footprint limits without silently deleting optical mass.

    The former per-view radius correction reduced both tangent axes while
    leaving opacity unchanged.  Its optical mass therefore fell with the
    square of the shrink factor, including after densification had stopped.
    This projection conserves ``tau * tangent_area`` per primitive; any mass
    that cannot fit below the ordinary opacity logit ceiling is measured
    explicitly instead of being hidden by a global opacity boost.
    """
    rows = torch.as_tensor(
        row_indices, dtype=torch.long, device=surface._scaling.device
    ).reshape(-1)
    shrink = torch.as_tensor(
        isotropic_shrink,
        dtype=surface._scaling.dtype,
        device=surface._scaling.device,
    ).reshape(-1)
    if len(rows) != len(shrink):
        raise ValueError("surface scale rows and shrink values must align")
    if not len(rows):
        return {
            "contract": "local_tau_times_tangent_area_conservation",
            "rows": 0,
            "changed_rows": 0,
            "target_mass": 0.0,
            "realized_mass": 0.0,
            "unrealized_mass": 0.0,
        }
    shrink = torch.nan_to_num(
        shrink, nan=1.0, posinf=1.0, neginf=1.0
    ).clamp(1.0e-3, 1.0)
    old_log_scale = surface._scaling[rows].detach().clone()
    old_area = old_log_scale.exp().prod(dim=1)
    old_alpha = surface.get_opacity[rows, 0].detach().clamp(
        0.0, 1.0 - 1.0e-6
    )
    target_mass = -torch.log1p(-old_alpha) * old_area
    new_log_scale = old_log_scale + shrink.log()[:, None]
    new_log_scale = torch.minimum(
        new_log_scale,
        new_log_scale.new_tensor(float(maximum_scale)).log(),
    )
    changed = torch.any(new_log_scale < old_log_scale - 1.0e-9, dim=1)
    surface._scaling[rows] = new_log_scale
    new_area = new_log_scale.exp().prod(dim=1).clamp_min(1.0e-12)
    requested_tau = target_mass / new_area
    requested_alpha = -torch.expm1(-requested_tau)
    maximum_alpha = surface._opacity.new_tensor(4.0).sigmoid()
    realized_alpha = requested_alpha.clamp(1.0e-6, maximum_alpha)
    surface._opacity[rows, 0] = torch.logit(realized_alpha)
    realized_mass = -torch.log1p(-realized_alpha) * new_area
    unrealized = (target_mass - realized_mass).clamp_min(0.0)
    return {
        "contract": "local_tau_times_tangent_area_conservation",
        "rows": int(len(rows)),
        "changed_rows": int(changed.sum()),
        "target_mass": float(target_mass.sum()),
        "realized_mass": float(realized_mass.sum()),
        "unrealized_mass": float(unrealized.sum()),
        "maximum_alpha": float(maximum_alpha),
    }


def _surface_screen_limit_rows(
    surface,
    chart_surface,
    policy: str,
    *,
    residual_start: int,
    structural_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Resolve rows allowed to consume rasterizer footprint feedback.

    ``atlas_residual`` freezes ordinary handoff geometry, but its live Chart
    rows are not frozen: their rendered geometry is replaced by the inverse
    depth atlas every forward.  They therefore need the same persistent
    screen-space scale ceiling as trainable completion rows.  Returning one
    deduplicated row vector keeps package radii and surface state aligned.
    """
    count = int(structural_count)
    if policy == "joint":
        return torch.arange(count, device=device, dtype=torch.long)
    if policy != "atlas_residual":
        return torch.empty(0, device=device, dtype=torch.long)
    suffix = torch.arange(
        int(residual_start), count, device=device, dtype=torch.long
    )
    if chart_surface is None or not hasattr(
        chart_surface, "live_surface_rows"
    ):
        return suffix
    chart_rows = chart_surface.live_surface_rows(surface).to(
        device=device, dtype=torch.long
    )
    chart_rows = chart_rows[
        (chart_rows >= 0) & (chart_rows < count)
    ]
    if not len(chart_rows):
        return suffix
    return torch.unique(torch.cat([suffix, chart_rows]), sorted=True)


def _apply_static_optical_policy(
    foliage,
    volume_optimizer,
    phase: str,
    *,
    canonical_evidence_opacity_gradient: torch.Tensor | None = None,
) -> dict[str, object]:
    """Enforce the static parameter/loss ownership matrix continuously.

    Envelope mass may grow from ray-hit evidence and RGB while the Stage-2
    support is formed.  Once Stage-3 detail is visible, RGB is detached from
    envelope opacity, but canonical geometry/ray evidence must still be able
    to repair a real occlusion hole.  All other positive-growth components
    are removed while every negative/free-space component is retained.
    Static detail and skeleton mass never inherit the envelope-retirement
    rule.
    During final polish, geometry, covariance and mass freeze together so
    appearance cannot create holes by moving a fixed-alpha Gaussian.
    """
    audit: dict[str, object] = {
        "contract": STATIC_OPTICAL_POLICY_CONTRACT,
        "phase": phase,
        "joint_polish_freeze": phase == "canonical_polish",
        "envelope_growth_rows_attenuated": 0,
        "envelope_positive_growth_attenuation": (
            "stage3_canonical_geometry_ray_only__stage2_actual_local_"
            "handoff_retired_fraction"
        ),
        "envelope_high_order_sh_rows_frozen": 0,
        "mean_envelope_replacement_authority": 0.0,
        "static_detail_mass_trainable": _static_detail_stage_trainable(phase),
        "static_detail_ray_trainable": _static_detail_ray_trainable(phase),
        "skeleton_mass_trainable": phase != "canonical_polish",
        "opacity_growth_before_policy": {},
        "opacity_growth_removed_by_policy": {},
    }
    if phase == "canonical_polish":
        for parameter in (
            foliage.xyz,
            foliage.log_scales,
            foliage.quaternions,
            foliage.opacity_logits,
            foliage.deformation_basis,
            foliage.dynamic_feature_basis,
            foliage.dynamic_opacity_basis,
        ):
            parameter.grad = None
        envelope = foliage.persistent_envelope_mask
        if foliage.features.shape[1] > 1:
            if foliage.features.grad is not None:
                nonzero = (
                    foliage.features.grad[envelope, 1:]
                    .abs().flatten(1).sum(1)
                )
                audit["envelope_high_order_sh_rows_frozen"] = int(
                    (nonzero > 0).sum()
                )
                foliage.features.grad[envelope, 1:] = 0
            feature_state = volume_optimizer.state.get(
                foliage.features, {}
            )
            feature_moment = feature_state.get("exp_avg")
            if (
                torch.is_tensor(feature_moment)
                and feature_moment.ndim == 3
                and feature_moment.shape[1] > 1
            ):
                feature_moment[envelope, 1:] = 0
        return audit

    if not _static_detail_ray_trainable(phase):
        detail = foliage.static_leaf_mask
        for parameter in (
            foliage.xyz,
            foliage.log_scales,
            foliage.quaternions,
            foliage.opacity_logits,
            foliage.features,
        ):
            if parameter.grad is not None:
                parameter.grad[detail] = 0
            state = volume_optimizer.state.get(parameter, {})
            first_moment = state.get("exp_avg")
            if torch.is_tensor(first_moment):
                first_moment[detail] = 0

    verification_state = getattr(foliage, "verification_state", None)
    if verification_state is not None:
        unverified = verification_state == VERIFICATION_UNVERIFIED
        # DC and optical mass are allowed to settle on true owner rays, but a
        # child cannot invent view-dependent colour before multiview proof.
        if foliage.features.grad is not None and foliage.features.shape[1] > 1:
            foliage.features.grad[unverified, 1:] = 0
        feature_state = volume_optimizer.state.get(foliage.features, {})
        feature_moment = feature_state.get("exp_avg")
        if (
            torch.is_tensor(feature_moment)
            and feature_moment.ndim == 3
            and feature_moment.shape[1] > 1
        ):
            feature_moment[unverified, 1:] = 0

    envelope = foliage.persistent_envelope_mask
    detail_visible = _static_detail_stage_visible(phase)
    if detail_visible and foliage.features.shape[1] > 1:
        if foliage.features.grad is not None:
            nonzero = foliage.features.grad[envelope, 1:].abs().flatten(1).sum(1)
            audit["envelope_high_order_sh_rows_frozen"] = int(
                (nonzero > 0).sum()
            )
            foliage.features.grad[envelope, 1:] = 0
        feature_state = volume_optimizer.state.get(foliage.features, {})
        feature_moment = feature_state.get("exp_avg")
        if (
            torch.is_tensor(feature_moment)
            and feature_moment.ndim == 3
            and feature_moment.shape[1] > 1
        ):
            feature_moment[envelope, 1:] = 0
    retirement_envelope = envelope
    if verification_state is not None:
        retirement_envelope = envelope & (
            verification_state == VERIFICATION_VERIFIED
        )
    readiness = (
        foliage.replacement_observation_count.float() / 3.0
    ).clamp(0.0, 1.0)
    authority = (
        foliage.replacement_overlap_ema
        * readiness
        * retirement_envelope.float()
    ).detach().clamp(0.0, 1.0)
    if bool(retirement_envelope.any()):
        audit["mean_envelope_replacement_authority"] = float(
            authority[retirement_envelope].mean()
        )
    retired_fraction = getattr(
        foliage,
        "handoff_retired_fraction",
        torch.zeros_like(authority),
    )
    attenuation = retired_fraction.detach().clamp(0.0, 0.95)
    gradient = foliage.opacity_logits.grad
    if gradient is not None:
        skeleton = getattr(
            foliage,
            "static_skeleton_mask",
            torch.zeros_like(envelope),
        )
        detail = foliage.static_leaf_mask
        role_masks = {
            "skeleton": skeleton,
            "persistent_envelope": envelope,
            "static_detail": detail,
        }
        if verification_state is not None:
            role_masks["unverified_child"] = unverified
        for name, mask in role_masks.items():
            growth_values = (-gradient[mask]).clamp_min(0.0)
            audit["opacity_growth_before_policy"][name] = {
                "rows": int((growth_values > 0).flatten(1).any(dim=1).sum())
                if len(growth_values)
                else 0,
                "magnitude": float(growth_values.sum()),
            }
        gradient_before = gradient.detach().clone()
        # Newborn children must be able to establish optical existence before
        # they can participate in envelope retirement. Once detail is visible
        # the broad envelope may grow only from the canonical geometry/ray
        # objective. Stage-3 RGB already has a zero envelope-opacity gate;
        # keeping this source separately prevents any auxiliary loss from
        # recreating the old global low-pass opacity shortcut.
        growth_owner = envelope if detail_visible else retirement_envelope
        growth = growth_owner[:, None] & (gradient < 0)
        attenuated_growth = (
            growth
            if detail_visible
            else growth & (attenuation[:, None] > 0)
        )
        audit["envelope_growth_rows_attenuated"] = int(
            attenuated_growth.flatten(1).any(dim=1).sum()
        )
        if detail_visible:
            if canonical_evidence_opacity_gradient is None:
                # Backward-compatible conservative fallback for callers that
                # do not expose source-separated gradients.
                gradient[growth] = 0
            else:
                evidence = torch.as_tensor(
                    canonical_evidence_opacity_gradient,
                    device=gradient.device,
                    dtype=gradient.dtype,
                )
                if evidence.shape != gradient.shape:
                    raise ValueError(
                        "canonical evidence opacity gradient must align "
                        "with foliage opacity"
                    )
                non_evidence = gradient_before - evidence
                routed = evidence + non_evidence.clamp_min(0.0)
                gradient[envelope] = routed[envelope]
        else:
            gradient[growth] *= (
                1.0 - attenuation[:, None].expand_as(gradient)[growth]
            )
        removed = (gradient - gradient_before).clamp_min(0.0)
        for name, mask in role_masks.items():
            audit["opacity_growth_removed_by_policy"][name] = float(
                removed[mask].sum()
            )
    # Apply the same continuous authority to Adam's retained growth moment;
    # otherwise a zero current gradient can still increase envelope mass.
    state = volume_optimizer.state.get(foliage.opacity_logits, {})
    first_moment = state.get("exp_avg")
    if torch.is_tensor(first_moment):
        growth_owner = envelope if detail_visible else retirement_envelope
        growth_momentum = growth_owner[:, None] & (
            first_moment < 0
        )
        if detail_visible:
            if canonical_evidence_opacity_gradient is None:
                first_moment[growth_momentum] = 0
            else:
                evidence_growth = (
                    canonical_evidence_opacity_gradient.to(
                        device=first_moment.device,
                        dtype=first_moment.dtype,
                    )
                    < 0
                )
                first_moment[growth_momentum & ~evidence_growth] = 0
        else:
            first_moment[growth_momentum] *= (
                1.0
                - attenuation[:, None]
                .expand_as(first_moment)[growth_momentum]
            )
    for parameter in (
        foliage.deformation_basis,
        foliage.dynamic_feature_basis,
        foliage.dynamic_opacity_basis,
    ):
        parameter.grad = None
    return audit


def _apply_mature_surface_gradient_policy(
    surface,
    policy: str,
    *,
    chart_surface=None,
    residual_start: int | None = None,
) -> dict[str, object]:
    """Keep a validated rigid handoff from becoming a second pretrain.

    A handoff is useful only if the mixed stage preserves the geometry that
    established its rigid-query result.  Merely continuing the old learning
    rate was not preservation: in the first 2k mixed updates the surface
    population grew by more than 60k and its official-query non-tree PSNR
    fell by roughly 0.4 dB.  The appearance-only policy leaves native SH
    coefficients trainable while assigning xyz, tangent scale, rotation and
    opacity to the completed rigid stage. Semantic retirement is intentionally
    absent here: Adam normalizes even tiny positive opacity gradients and
    therefore turned a one-sided ownership gradient into almost global
    extinction. Local retirement is instead performed at replacement events
    from accumulated cross-sequence responsibility with an explicit optical
    mass budget.
    """
    if policy not in {
        "joint",
        "atlas_residual",
        "appearance_only",
        "frozen",
    }:
        raise ValueError(f"Unknown mature surface policy: {policy}")
    if policy == "joint":
        return {
            "policy": policy,
            "topology_trainable": True,
            "geometry_trainable": True,
            "chart_geometry_trainable": True,
            "appearance_trainable": True,
            "opacity_trainable": True,
            "opacity_gradient_contract": "all_losses",
        }
    if policy == "atlas_residual":
        if residual_start is None:
            raise RuntimeError(
                "atlas_residual requires the immutable handoff/residual row "
                "boundary"
            )
        residual_start = int(residual_start)
        if residual_start < 0 or residual_start > len(surface.get_xyz):
            raise RuntimeError("Surface residual row boundary is invalid")
        prefix = slice(0, residual_start)
        for parameter in (
            surface._xyz,
            surface._scaling,
            surface._rotation,
            surface._opacity,
        ):
            if parameter.grad is not None:
                parameter.grad[prefix] = 0
        return {
            "policy": policy,
            "topology_trainable": True,
            "topology_trainable_scope": "evidence_completion_suffix_only",
            "geometry_trainable": True,
            "geometry_trainable_scope": "evidence_completion_suffix_only",
            "residual_start": residual_start,
            "residual_rows": int(len(surface.get_xyz) - residual_start),
            "chart_geometry_trainable": True,
            "appearance_trainable": True,
            "opacity_trainable": True,
            "opacity_gradient_contract": "evidence_completion_suffix_only",
        }
    geometry = (
        surface._xyz,
        surface._scaling,
        surface._rotation,
    )
    for parameter in geometry:
        if parameter.grad is not None:
            parameter.grad.zero_()
    # Chart-owned rows do not use ``surface._xyz`` as their rendered centre:
    # their position, normal and tangent footprint come from the inverse-depth
    # atlas.  Clear (rather than merely zero) these gradients so Adam cannot
    # advance a resumed momentum buffer under an appearance-only/frozen
    # handoff.
    if chart_surface is not None:
        for parameter in chart_surface.parameters():
            parameter.grad = None
    if surface._opacity.grad is not None:
        surface._opacity.grad.zero_()
    if policy == "frozen":
        for parameter in (
            surface._features_dc,
            surface._features_rest,
        ):
            if parameter.grad is not None:
                parameter.grad.zero_()
    return {
        "policy": policy,
        "topology_trainable": False,
        "geometry_trainable": False,
        "chart_geometry_trainable": False,
        "appearance_trainable": policy == "appearance_only",
        "opacity_trainable": False,
        "opacity_gradient_contract": "frozen_event_bounded_retirement",
    }


def _apply_surface_spatial_confidence_gradients(surface) -> dict[str, float]:
    """Use per-primitive geometry uncertainty on geometric parameters.

    RGB/color and opacity remain free to validate a coverage witness.  XYZ,
    scale and rotation updates are continuously attenuated for spatially
    uncertain Chart/pointmap rows so they cannot move or broaden as quickly
    as independently supported geometry.
    """
    confidence = torch.nan_to_num(
        surface._geometry_confidence.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)
    reliability = 0.40 + 0.60 * confidence / (confidence + 0.25)
    for parameter in (
        surface._xyz,
        surface._scaling,
        surface._rotation,
    ):
        if parameter.grad is not None:
            parameter.grad *= reliability.reshape(
                (-1,) + (1,) * (parameter.grad.ndim - 1)
            )
    return {
        "minimum": float(reliability.min()) if len(reliability) else 0.0,
        "mean": float(reliability.mean()) if len(reliability) else 0.0,
        "maximum": float(reliability.max()) if len(reliability) else 0.0,
        "parameterization": "0.40+0.60*c/(c+0.25)",
        "spatial_per_primitive": True,
    }


@torch.no_grad()
def _masked_clamp_max_(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    maximum: torch.Tensor | float,
) -> None:
    """Clamp selected parameter rows without advanced-index copy semantics."""
    mask = torch.as_tensor(mask, device=tensor.device, dtype=torch.bool)
    if mask.ndim != 1 or mask.shape[0] != tensor.shape[0]:
        raise ValueError("masked clamp must select the tensor's first axis")
    if bool(mask.any()):
        # ``tensor[mask].clamp_`` is a silent no-op on the source tensor:
        # boolean advanced indexing returns a temporary. Assign the clamped
        # rows back explicitly so role opacity contracts are real.
        tensor[mask] = tensor[mask].clamp(max=maximum)


def _geometry_losses(
    package,
    evidence: dict[str, torch.Tensor],
    rigid: torch.Tensor,
    args,
    *,
    geometry_weight: torch.Tensor | None = None,
    plane_task_weight: torch.Tensor | None = None,
    native_chart_factor_active: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = package.depth.new_zeros(())
    losses = {
        "chart": zero,
        "plane": zero,
        "normal": zero,
        "inverse": zero,
        "ordinal": zero,
    }
    predicted_depth = package.depth[0]
    valid_prediction_mask = (
        torch.isfinite(predicted_depth) & (predicted_depth > 0)
    )
    safe_predicted_depth = torch.where(
        valid_prediction_mask,
        predicted_depth,
        torch.ones_like(predicted_depth),
    )
    valid_prediction = valid_prediction_mask.float()
    geometry_weight = (
        rigid
        if geometry_weight is None
        else geometry_weight * rigid
    )
    plane_task_weight = (
        geometry_weight
        if plane_task_weight is None
        else plane_task_weight * rigid
    )
    source_bits = evidence.get("source_bitmask")
    if source_bits is not None:
        source_bits = source_bits[0].to(torch.uint8)
        has_plane = (source_bits & 1) != 0
        has_chart = (source_bits & 2) != 0
        has_mono = (source_bits & 4) != 0
    else:
        has_plane = torch.zeros_like(rigid, dtype=torch.bool)
        has_chart = torch.zeros_like(rigid, dtype=torch.bool)
        has_mono = torch.zeros_like(rigid, dtype=torch.bool)
    # Exactly one dense metric source owns each pixel: plane beats Chart,
    # Chart beats calibrated mono.  When the versioned fused cache exists it
    # is the only Chart-derived metric target.  A per-pixel fallback to the
    # separately resampled atlas creates a boundary-only second target whose
    # validity interpolation does not match rho validity.
    plane_owner = has_plane
    chart_owner = has_chart & ~plane_owner
    mono_owner = has_mono & ~plane_owner & ~chart_owner
    fused_available = source_bits is not None and "rho_mean" in evidence
    inverse_valid = torch.zeros_like(rigid, dtype=torch.bool)
    inverse_support = torch.zeros_like(rigid)
    if fused_available:
        raw_rho = evidence["rho_mean"][0]
        raw_variance = evidence["rho_variance"][0]
        inverse_support = (
            evidence["support_view_count"][0] >= 1
        ).to(rigid.dtype)
        inverse_valid = (
            torch.isfinite(raw_rho)
            & (raw_rho > 0)
            & torch.isfinite(raw_variance)
            & (raw_variance > 0)
            & (inverse_support > 0)
        )
    if "chart_depth" in evidence:
        raw_target = evidence["chart_depth"][0]
        valid_target = torch.isfinite(raw_target) & (raw_target > 0)
        target = torch.where(
            valid_target, raw_target, torch.ones_like(raw_target)
        )
        weight = (
            evidence["chart_weight"][0]
            * geometry_weight
            * valid_prediction
            * valid_target.float()
        )
        if source_bits is not None:
            chart_fallback = (
                chart_owner
                if not fused_available
                else torch.zeros_like(chart_owner)
            )
            weight = weight * chart_fallback.float()
        relative = (
            safe_predicted_depth - target
        ).abs() / target.clamp_min(0.1)
        losses["chart"] = (
            torch.log1p(relative) * weight
        ).sum() / weight.sum().clamp_min(1)
    if "plane_depth" in evidence:
        raw_target = evidence["plane_depth"][0]
        valid_target = torch.isfinite(raw_target) & (raw_target > 0)
        target = torch.where(
            valid_target, raw_target, torch.ones_like(raw_target)
        )
        weight = (
            evidence["plane_weight"][0]
            * plane_task_weight
            * valid_prediction
            * valid_target.float()
        )
        if source_bits is not None:
            weight = weight * plane_owner.float()
        relative = (
            safe_predicted_depth - target
        ).abs() / target.clamp_min(0.1)
        losses["plane"] = (
            torch.log1p(relative) * weight
        ).sum() / weight.sum().clamp_min(1)
        if "plane_normal_world" in evidence:
            raw_predicted_normal = package.normal_world
            valid_predicted_normal = torch.isfinite(
                raw_predicted_normal
            ).all(0)
            safe_predicted_normal = torch.where(
                valid_predicted_normal[None],
                raw_predicted_normal,
                torch.zeros_like(raw_predicted_normal),
            )
            predicted_normal = F.normalize(
                safe_predicted_normal, dim=0, eps=1e-6
            )
            raw_target_normal = evidence["plane_normal_world"]
            valid_normal = torch.isfinite(raw_target_normal).all(0)
            safe_target_normal = torch.where(
                valid_normal[None],
                raw_target_normal,
                torch.zeros_like(raw_target_normal),
            )
            target_normal = F.normalize(
                safe_target_normal, dim=0, eps=1e-6
            )
            cosine = (
                predicted_normal * target_normal
            ).sum(0).abs()
            normal_weight = (
                weight
                * valid_normal.float()
                * valid_predicted_normal.float()
            )
            losses["normal"] = (
                (1.0 - cosine) * normal_weight
            ).sum() / normal_weight.sum().clamp_min(1)
    if "rho_mean" in evidence:
        rho = 1.0 / safe_predicted_depth.clamp_min(1e-3)
        raw_target = evidence["rho_mean"][0]
        raw_variance = evidence["rho_variance"][0]
        valid_target = (
            torch.isfinite(raw_target)
            & (raw_target > 0)
            & torch.isfinite(raw_variance)
            & (raw_variance > 0)
        )
        target = torch.where(
            valid_target, raw_target, torch.zeros_like(raw_target)
        )
        variance = torch.where(
            valid_target,
            raw_variance.clamp_min(1e-6),
            torch.ones_like(raw_variance),
        )
        weight = (
            geometry_weight
            * inverse_support
            * valid_prediction
            * valid_target.float()
        )
        if source_bits is not None:
            weight = weight * chart_owner.float()
        if native_chart_factor_active:
            # Chart is a 512x288 observation.  Its persistent factor is
            # evaluated after area-reducing the render to that native grid.
            # Reusing an interpolated rho cache here invents high-resolution
            # edge supervision and applies the same source twice.
            weight = torch.zeros_like(weight)
        nll = torch.log1p((rho - target).square() / variance)
        losses["inverse"] = (
            nll * weight
        ).sum() / weight.sum().clamp_min(1)
    if "mono_depth" in evidence:
        mono = evidence["mono_depth"][0]
        # DAV2 contributes only ordering.  Adjacent samples with negligible
        # monocular separation are ignored instead of treated as metric depth.
        pred_a = predicted_depth[::8, ::8]
        pred_b = predicted_depth[4::8, 4::8]
        mono_a = mono[::8, ::8]
        mono_b = mono[4::8, 4::8]
        height = min(pred_a.shape[0], pred_b.shape[0])
        width = min(pred_a.shape[1], pred_b.shape[1])
        delta = mono_a[:height, :width] - mono_b[:height, :width]
        sign = torch.sign(delta)
        finite_mono = mono[torch.isfinite(mono)]
        mono_scale = (
            finite_mono.abs().median().clamp_min(1e-3)
            if finite_mono.numel()
            else mono.new_tensor(1.0)
        )
        valid = (
            torch.isfinite(delta)
            & torch.isfinite(pred_a[:height, :width])
            & torch.isfinite(pred_b[:height, :width])
            & (delta.abs() > 0.01 * mono_scale)
        )
        rigid_pair = torch.minimum(
            geometry_weight[::8, ::8][:height, :width],
            geometry_weight[4::8, 4::8][:height, :width],
        )
        if source_bits is not None:
            mono_pair = (
                mono_owner[::8, ::8][:height, :width]
                & mono_owner[4::8, 4::8][:height, :width]
            )
            valid = valid & mono_pair
        order = sign * (
            pred_a[:height, :width] - pred_b[:height, :width]
        )
        losses["ordinal"] = (
            (
                F.softplus(-order) * rigid_pair
            )[valid].sum() / rigid_pair[valid].sum().clamp_min(1)
            if bool(valid.any())
            else zero
        )
    total = (
        args.geometry_weight
        * (losses["chart"] + losses["inverse"])
        + args.plane_weight * losses["plane"]
        + args.normal_weight * losses["normal"]
        + args.ordinal_weight * losses["ordinal"]
    )
    values = {
        name: float(value.detach()) for name, value in losses.items()
    }
    values.update(
        {
            "plane_pixels": int(plane_owner.sum()),
            "chart_pixels": int(chart_owner.sum()),
            "inverse_pixels": int((chart_owner & inverse_valid).sum()),
            "raw_chart_fallback_pixels": int(
                chart_owner.sum() if not fused_available else 0
            ),
        }
    )
    return total, values


def _projected_occluded_rigid_geometry_loss(
    package,
    task: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Train persistent surface geometry through a current-view occluder."""
    predicted_depth = package.surface_depth[0]
    surface_alpha = package.surface_alpha[0].clamp(0.0, 1.0)
    target_depth = task["projected_rigid_depth"]
    support = (
        task["p_projected_rigid_geometry"]
        * task["p_distortion_valid"]
    ).clamp(0.0, 1.0)
    valid_target = (
        (support > 0.0)
        & torch.isfinite(target_depth)
        & (target_depth > 0.0)
    )
    zero = predicted_depth.new_zeros(())
    if not bool(valid_target.any()):
        return zero, {
            "contract": PROJECTED_OCCLUDED_RIGID_GEOMETRY_CONTRACT,
            "supported_pixels": 0,
            "depth_matched_pixels": 0,
            "depth": 0.0,
            "coverage": 0.0,
        }
    weight = support * valid_target.to(support.dtype)
    valid_prediction = (
        torch.isfinite(predicted_depth) & (predicted_depth > 0.0)
    )
    depth_weight = weight * valid_prediction.to(weight.dtype)
    safe_prediction = torch.where(
        valid_prediction, predicted_depth, target_depth.detach()
    ).clamp_min(1.0e-6)
    log_residual = F.smooth_l1_loss(
        safe_prediction.log(),
        target_depth.clamp_min(1.0e-6).log(),
        reduction="none",
        beta=0.05,
    )
    depth_loss = (
        (log_residual * depth_weight).sum()
        / depth_weight.sum().clamp_min(1.0)
    )
    # A stable projected track is a sparse existence observation for the
    # surface-only branch. It may restore alpha/bandwidth behind foliage, but
    # cannot prescribe the RGB of the current occluded pixel.
    coverage_loss = (
        (-torch.log(surface_alpha + 1.0e-4) * weight).sum()
        / weight.sum().clamp_min(1.0)
    )
    loss = depth_loss + 0.05 * coverage_loss
    return loss, {
        "contract": PROJECTED_OCCLUDED_RIGID_GEOMETRY_CONTRACT,
        "supported_pixels": int(valid_target.sum()),
        "depth_matched_pixels": int((valid_target & valid_prediction).sum()),
        "depth": float(depth_loss.detach()),
        "coverage": float(coverage_loss.detach()),
    }


def _replacement_stats(count: int, device) -> dict[str, torch.Tensor]:
    return {
        "evidence": torch.zeros(count, 4, device=device),
        "semantic_evidence": torch.zeros(count, 3, device=device),
        "weight": torch.zeros(count, device=device),
        "observations": torch.zeros(
            count, dtype=torch.int16, device=device
        ),
        "sequence_bits": torch.zeros(
            count, dtype=torch.int64, device=device
        ),
        "retired": torch.zeros(
            count, dtype=torch.bool, device=device
        ),
        "retirement_fraction": torch.zeros(count, device=device),
    }


@torch.no_grad()
def _bounded_surface_retirement_fractions(
    previous_fraction: torch.Tensor,
    desired_fraction: torch.Tensor,
    current_optical_depth: torch.Tensor,
    *,
    maximum_mass_fraction: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Allocate local retirement without Adam's scale-normalization failure.

    ``desired_fraction`` is a continuous evidence posterior.  A global scale
    is applied only to its positive increment so the requested structural
    optical-depth removal cannot exceed a fixed fraction of the current
    scene. Unlike an opacity-gradient threshold, this preserves the relative
    strength and spatial sparsity of every candidate's accumulated evidence.
    """
    previous_fraction = torch.nan_to_num(
        previous_fraction.float(), nan=0.0
    ).clamp(0.0, 1.0)
    desired_fraction = torch.nan_to_num(
        desired_fraction.float(), nan=0.0
    ).clamp(0.0, 1.0)
    current_optical_depth = torch.nan_to_num(
        current_optical_depth.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    if not (
        previous_fraction.shape
        == desired_fraction.shape
        == current_optical_depth.shape
    ):
        raise ValueError("Surface retirement tensors must have equal shape")
    maximum_mass_fraction = float(maximum_mass_fraction)
    if not 0.0 <= maximum_mass_fraction <= 0.05:
        raise ValueError("maximum_mass_fraction must lie in [0, 0.05]")
    increment = (desired_fraction - previous_fraction).clamp_min(0.0)
    requested_mass = (
        current_optical_depth
        * increment
        / (1.0 - previous_fraction).clamp_min(1e-6)
    )
    requested_total = requested_mass.sum()
    current_total = current_optical_depth.sum()
    budget = current_total * maximum_mass_fraction
    scale = torch.where(
        requested_total > 0,
        (budget / requested_total.clamp_min(1e-12)).clamp_max(1.0),
        requested_total.new_zeros(()),
    )
    next_fraction = (
        previous_fraction + increment * scale
    ).clamp(0.0, 1.0)
    realized_mass = (
        current_optical_depth
        * (next_fraction - previous_fraction)
        / (1.0 - previous_fraction).clamp_min(1e-6)
    ).sum()
    return next_fraction, {
        "requested_optical_mass_fraction": float(
            requested_total / current_total.clamp_min(1e-12)
        ),
        "realized_optical_mass_fraction": float(
            realized_mass / current_total.clamp_min(1e-12)
        ),
        "allocation_scale": float(scale),
    }


@torch.no_grad()
def _replacement_audit(
    view,
    structural,
    foliage,
    background,
    task,
    target,
    sky_color,
    stats,
    sequence_bits,
    *,
    include_dynamic: bool = False,
    temporal_code: torch.Tensor | None = None,
    volume_gate: torch.Tensor | None = None,
    maximum_retirement_mass_fraction: float = 0.0025,
) -> dict[str, int | float]:
    render_kwargs = {
        "background": background,
        "structural_trainable_start": None,
        "temporal_code": temporal_code,
        "include_dynamic": bool(include_dynamic),
        "volume_gate": volume_gate,
    }
    posterior = render_hybrid(
        view,
        structural,
        foliage,
        volume_means_override=foliage.initialization_center,
        volume_opacity_scale=1.0,
        surface_gate=torch.zeros_like(
            structural.get_opacity.reshape(-1)
        ),
        **render_kwargs,
    )
    volume_only = render_hybrid(
        view,
        structural,
        foliage,
        surface_gate=torch.zeros_like(
            structural.get_opacity.reshape(-1)
        ),
        **render_kwargs,
    )
    package = render_hybrid(
        view,
        structural,
        foliage,
        **render_kwargs,
    )
    structural_count = int(package.structural_count)
    prediction = composite_white_background(
        package.render, package.alpha, sky_color
    )
    posterior_valid = (
        (posterior.volume_alpha[0] > 0.03)
        & (posterior.volume_depth[0] > 0)
    )
    new_match = torch.exp(
        -(
            package.volume_depth[0] - posterior.volume_depth[0]
        ).abs()
        / 0.45
    )
    old_match = torch.exp(
        -(
            package.surface_depth[0] - posterior.volume_depth[0]
        ).abs()
        / 0.45
    )
    # A same-depth structural/volume duplicate is still an ownership error:
    # it double-counts optical mass and produces the characteristic painted
    # crown/facade boundary. The previous (1-old_match) term made that common
    # case mathematically impossible to retire. Retain a conservative 0.35
    # duplicate proof and let multi-sequence RGB improvement decide how much
    # opacity is retired continuously.
    duplicate_or_wrong_depth = 0.35 + 0.65 * (1.0 - old_match)
    depth_proof = (
        new_match
        * duplicate_or_wrong_depth
        * posterior_valid
    ).clamp(0, 1)
    full_error = (prediction - target).abs().mean(0)
    volume_only_prediction = composite_white_background(
        volume_only.render, volume_only.alpha, sky_color
    )
    volume_error = (
        volume_only_prediction - target
    ).abs().mean(0)
    improvement = torch.sigmoid(
        (full_error - volume_error) / 0.02
    )
    del (
        posterior,
        volume_only,
        package,
        prediction,
        volume_only_prediction,
    )
    audit = render_hybrid(
        view,
        structural,
        foliage,
        audit_fields=torch.stack(
            [
                task["p_canopy_core"],
                task["p_rigid"],
                depth_proof,
                improvement,
                task["p_canopy"] * depth_proof,
                task["p_sky"],
                (
                    task["p_transient"]
                    * task["w_rgb"]
                    * _boundary_evidence_weight(task)
                ),
            ]
        ),
        **render_kwargs,
    ).responsibility
    if audit is None:
        return {"observed": 0, "newly_retired": 0}
    surface = audit[:structural_count]
    observed = surface[:, 0] > 0.02
    observed = observed & ~stats["retired"]
    stats["evidence"] += surface[:, 1:5]
    semantic_evidence = stats.setdefault(
        "semantic_evidence",
        torch.zeros(
            len(structural.get_xyz),
            3,
            device=structural.get_xyz.device,
        ),
    )
    semantic_evidence += surface[:, 5:8]
    stats["weight"] += surface[:, 0]
    stats["observations"][observed] += 1
    bit = sequence_bits[sequence_id(view.image_name)]
    stats["sequence_bits"][observed] |= bit
    average = stats["evidence"] / stats["weight"].clamp_min(1e-8)[:, None]
    sequence_count = torch.zeros_like(stats["observations"])
    for value in sequence_bits.values():
        sequence_count += (
            (stats["sequence_bits"] & value) != 0
        ).to(torch.int16)
    confidence = (
        average[:, 0]
        * (1.0 - average[:, 1])
        * average[:, 2]
        * average[:, 3]
    )
    evidence_id = structural._track_id
    has_external_owner = evidence_id != -1
    replaceable_owner = ~structural._protected_flag
    if bool(has_external_owner.any()):
        _, inverse = torch.unique(
            evidence_id[has_external_owner],
            sorted=False,
            return_inverse=True,
        )
        active_count = torch.zeros(
            int(inverse.max()) + 1,
            dtype=torch.int32,
            device=evidence_id.device,
        )
        active_count.scatter_add_(
            0,
            inverse,
            (~stats["retired"][has_external_owner]).to(torch.int32),
        )
        replaceable_owner[has_external_owner] |= (
            active_count[inverse] >= 2
        )
    # Replacement is a continuous evidence posterior, not a pass/fail gate.
    # One sequence cannot prove that a structural point is a transient canopy
    # occluder, so cross-sequence readiness starts at zero and then grows with
    # independent observations. The smooth proof transform avoids the old
    # 0.45 cliff where two almost-identical candidates took opposite paths.
    observation_readiness = (
        1.0
        - torch.exp(
            -stats["observations"].to(confidence.dtype) / 3.0
        )
    )
    sequence_readiness = (
        sequence_count.to(confidence.dtype) - 1.0
    ).clamp(0.0, 1.0)
    proof = torch.sigmoid((confidence - 0.25) / 0.08)
    proposed_fraction = (
        proof
        * observation_readiness
        * sequence_readiness
        * replaceable_owner.to(confidence.dtype)
    ).clamp(0.0, 1.0)
    semantic_average = (
        semantic_evidence
        / stats["weight"].clamp_min(1e-8)[:, None]
    )
    semantic_conflict = torch.maximum(
        semantic_average[:, 0],
        torch.maximum(
            0.50 * semantic_average[:, 1],
            0.35 * semantic_average[:, 2],
        ),
    ) * (1.0 - average[:, 1]).clamp(0.0, 1.0)
    # Squaring retains a continuous response while strongly concentrating
    # the finite event budget on primitives that repeatedly own tree-front,
    # sky, or transient pixels across independent sequences.
    semantic_fraction = (
        0.80
        * semantic_conflict.square()
        * observation_readiness
        * sequence_readiness
        * replaceable_owner.to(confidence.dtype)
    ).clamp(0.0, 0.80)
    desired_fraction = torch.maximum(
        proposed_fraction, semantic_fraction
    )
    previous_fraction = stats.setdefault(
        "retirement_fraction",
        torch.zeros_like(confidence),
    )
    opacity = structural.get_opacity.reshape(-1).clamp(
        1e-8, 1.0 - 1e-6
    )
    optical_depth = -torch.log1p(-opacity)
    next_fraction, mass_audit = _bounded_surface_retirement_fractions(
        previous_fraction,
        desired_fraction,
        optical_depth,
        maximum_mass_fraction=maximum_retirement_mass_fraction,
    )
    changed = next_fraction > previous_fraction + 1e-6
    if bool(changed.any()):
        # Scale optical depth by the incremental remaining-mass ratio. This
        # is stable under repeated audits and cannot repeatedly retire a weak
        # candidate merely because it happened to be rendered more often.
        remaining_ratio = (
            (1.0 - next_fraction[changed])
            / (1.0 - previous_fraction[changed]).clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        target_opacity = (
            1.0
            - torch.exp(
                -optical_depth[changed] * remaining_ratio
            )
        ).clamp(1e-8, 1.0 - 1e-6)
        structural._opacity[changed] = torch.logit(
            target_opacity
        )[:, None]
    stats["retirement_fraction"] = next_fraction
    newly_retired = (~stats["retired"]) & (next_fraction >= 0.95)
    stats["retired"] |= newly_retired
    return {
        "observed": int(observed.sum()),
        "updated": int(changed.sum()),
        "newly_retired": int(newly_retired.sum()),
        "retired_total": int(stats["retired"].sum()),
        "mean_retirement_fraction": float(next_fraction.mean()),
        "maximum_retirement_fraction": float(next_fraction.max()),
        "mean_semantic_conflict": float(semantic_conflict.mean()),
        "maximum_semantic_conflict": float(semantic_conflict.max()),
        **mass_audit,
        "replacement_mode": (
            "exact_dynamic_plus_canonical"
            if include_dynamic
            else "canonical"
        ),
        "exact_dynamic_evidence_rows": (
            int(
                (
                    foliage.dynamic_leaf_mask
                    & (volume_gate >= 0.999)
                ).sum()
            )
            if include_dynamic and volume_gate is not None
            else 0
        ),
    }


def _render_static_deployment_package(
    view,
    surface,
    foliage,
    background,
    *,
    canonical_canopy_enabled: bool,
    optical_replacement_policy: str,
    optical_responsibility_prior: float,
):
    """Render the exact static deployment contract during trainer eval."""
    volume_opacity_scale = 1.0 if canonical_canopy_enabled else 0.0
    ray_normalized = optical_replacement_policy in {
        STATIC_RAY_NORMALIZED_OPTICAL_POLICY,
        STATIC_RAY_SURFACE_EVIDENCE_POLICY,
    }
    if not canonical_canopy_enabled:
        # A rigid-stage render has no volume owner to arbitrate.  Deployment
        # policies such as static_ray_* are multi-pass compositor contracts,
        # not legal single-pass ``render_hybrid`` policies.  Passing one
        # through here crashed only after the expensive optimization had
        # completed.
        return render_hybrid(
            view,
            surface,
            foliage,
            background=background,
            include_dynamic=False,
            volume_opacity_scale=0.0,
            optical_replacement_policy="disabled",
        )
    if not ray_normalized:
        return render_hybrid(
            view,
            surface,
            foliage,
            background=background,
            include_dynamic=False,
            volume_opacity_scale=volume_opacity_scale,
            optical_replacement_policy=optical_replacement_policy,
        )

    skeleton = foliage.static_skeleton_mask

    def layer_pass(role_mask):
        return render_hybrid(
            view,
            surface,
            foliage,
            background=background,
            include_dynamic=False,
            volume_opacity_scale=1.0,
            volume_role_mask=role_mask,
            optical_replacement_policy="disabled",
        )

    envelope = layer_pass(
        skeleton | foliage.persistent_envelope_mask
    )
    detail = layer_pass(skeleton | foliage.static_leaf_mask)
    package, _ = _static_ray_normalized_optical_mixture(
        envelope,
        detail,
        symmetric_optical_prior=optical_responsibility_prior,
    )
    if optical_replacement_policy == STATIC_RAY_SURFACE_EVIDENCE_POLICY:
        surface_only = layer_pass(
            torch.zeros_like(foliage.layer_role, dtype=torch.bool)
        )
        package, _ = _static_surface_evidence_mixture(
            package, surface_only
        )
    return package


def _assert_pointmap_posterior_contract(
    geometry: OutdoorGeometryEvidence,
    *,
    allow_missing: bool,
) -> dict:
    audit = geometry.audit()["pointmap_cross_sequence_posterior"]
    missing = int(audit["missing_camera_count"])
    if missing > 0 and not allow_missing:
        raise RuntimeError(
            "MASt3R pointmap cross-sequence posterior is incomplete: "
            f"{missing}/{audit['pointmap_camera_count']} cameras are missing. "
            "Run scripts/build_mast3r_pointmap_cross_sequence_posterior.py "
            "or the build_evidence pipeline stage before production "
            "training. Use --allow-missing-pointmap-cross-sequence-posterior "
            "only for an explicitly labelled low-precision ablation."
        )
    return {
        **audit,
        "production_complete": missing == 0,
        "missing_posterior_ablation": bool(missing and allow_missing),
    }


def _restore_geometry_audit_counters_from_checkpoint(
    geometry: OutdoorGeometryEvidence, checkpoint: dict | None
) -> None:
    """Continue cumulative evidence accounting across a trainer resume.

    Geometry arrays are immutable external evidence, but their consumption
    counters live in the runtime lookup object.  Recreating that object while
    restoring the model previously reset every count to zero and could turn a
    valid rigid handoff into an ineligible one during export-only repair.
    """
    if checkpoint is None:
        return
    prior = checkpoint.get("geometry_evidence_audit")
    if not isinstance(prior, dict):
        return
    consumed = prior.get("source_consumption_count", {})
    if not isinstance(consumed, dict):
        raise RuntimeError(
            "Checkpoint geometry evidence consumption audit is invalid"
        )
    for name in geometry.consumed:
        value = int(consumed.get(name, 0))
        if value < 0:
            raise RuntimeError(
                "Checkpoint geometry evidence counter cannot be negative"
            )
        geometry.consumed[name] = value

    posterior = prior.get("pointmap_cross_sequence_posterior", {})
    if not isinstance(posterior, dict):
        raise RuntimeError(
            "Checkpoint pointmap posterior audit is invalid"
        )
    pixels = int(posterior.get("pixels", 0))
    mean_precision = float(posterior.get("mean_precision", 0.0))
    geometry._pointmap_posterior_stats.update(
        {
            "factor_calls": int(posterior.get("factor_calls", 0)),
            "posterior_factor_calls": int(
                posterior.get("posterior_factor_calls", 0)
            ),
            "single_sequence_fallback_factor_calls": int(
                posterior.get("single_sequence_fallback_factor_calls", 0)
            ),
            "pixels": pixels,
            "cross_sequence_supported_pixels": int(
                posterior.get("supported_pixels", 0)
            ),
            "single_sequence_low_precision_pixels": int(
                posterior.get("single_sequence_low_precision_pixels", 0)
            ),
            "missing_posterior_fallback_pixels": int(
                posterior.get("missing_posterior_fallback_pixels", 0)
            ),
            "precision_sum": mean_precision * pixels,
        }
    )


@torch.no_grad()
def _evaluate(
    views,
    indices,
    surface,
    foliage,
    sky,
    appearance,
    background,
    fields,
    camera_sequence_lookup,
    camera_frame_lookup,
    output: Path,
    *,
    canonical_canopy_enabled: bool,
    conditioned_enabled: bool,
    training_optical_replacement_policy: str = "disabled",
    deployment_optical_replacement_policy: str = "disabled",
    deployment_optical_responsibility_prior: float = 0.0,
) -> list[dict]:
    from PIL import Image as PILImage

    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in indices:
        if index < 0 or index >= len(views):
            continue
        view = views[index]
        task = fields.fields(
            view.image_name,
            (view.image_height, view.image_width),
            background.device,
        )
        package = _render_static_deployment_package(
            view,
            surface,
            foliage,
            background,
            canonical_canopy_enabled=canonical_canopy_enabled,
            optical_replacement_policy=(
                deployment_optical_replacement_policy
            ),
            optical_responsibility_prior=(
                deployment_optical_responsibility_prior
            ),
        )
        canonical = composite_white_background(
            package.render, package.alpha, sky(view)
        ).clamp(0, 1)
        conditioned = None
        if conditioned_enabled:
            conditioned_package = render_hybrid(
                view,
                surface,
                foliage,
                background=background,
                temporal_code=appearance.temporal_code(view.image_name),
                include_dynamic=True,
                volume_gate=_dynamic_visibility_gate(
                    foliage,
                    view,
                    camera_sequence_lookup,
                    camera_frame_lookup,
                ),
                optical_replacement_policy=(
                    training_optical_replacement_policy
                ),
            )
            conditioned = appearance(
                composite_white_background(
                    conditioned_package.render,
                    conditioned_package.alpha,
                    sky(view),
                ),
                view,
                task,
            ).clamp(0, 1)
        target = view.original_image.cuda()
        def metrics(value):
            mse = (value - target).square().mean().clamp_min(1e-12)
            return {
                "psnr": float(-10 * torch.log10(mse)),
                "ssim": float(ssim(value, target)),
                "mae": float((value - target).abs().mean()),
            }
        row = {
            "index": index,
            "image_name": str(view.image_name),
            "canonical": metrics(canonical),
        }
        if conditioned is not None:
            row["conditioned"] = metrics(conditioned)
        rows.append(row)
        images = {
            "gt": target,
            "canonical": canonical,
            "error_x4": (
                (conditioned if conditioned is not None else canonical)
                - target
            ).abs()
            * 4,
        }
        if conditioned is not None:
            images["conditioned"] = conditioned
        for name, value in images.items():
            image = (
                value.detach()
                .clamp(0, 1)
                .mul(255)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            PILImage.fromarray(image).save(
                output / f"{index:05d}_{name}.png"
            )
        release = getattr(view, "release_image", None)
        if release is not None:
            release()
    return rows


def main():
    args, dataset, opt, _pipe = _parse_args()
    cpu_parallelism_audit = _configure_cpu_parallelism(args)
    retained_checkpoint_iterations = set(
        args.retain_checkpoint_iterations
    )
    output = Path(dataset.model_path).resolve()
    evidence_store = load_evidence_store(args.evidence_store)
    if not mast3r_is_geometry_authority(evidence_store):
        raise RuntimeError(
            "Native hybrid Teacher requires MASt3R/MAtCha-primary geometry"
        )
    if evidence_store.get("final_model") != "hybrid_teacher":
        raise RuntimeError(
            "Native hybrid Teacher v2 must be the authoritative final model"
        )
    initialization = json.loads(
        (args.initialization / "initialization_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    _validate_initialization_protocol(initialization)
    rigid_placeholder = bool(
        initialization.get("initialization_contract", {}).get(
            "rigid_stage_placeholder_foliage", False
        )
    )
    if rigid_placeholder and args.training_profile != "hybrid_rigid_stage1":
        raise RuntimeError(
            "An inert rigid-stage foliage placeholder cannot initialize "
            "mixed or deployment training; rebuild the full foliage "
            "visual-hull/ray posterior initialization"
        )
    surface_warmstart = (
        _validate_surface_warmstart(
            args.surface_warmstart_ply,
            args.surface_warmstart_manifest,
        )
        if args.surface_warmstart_ply is not None
        else None
    )
    foliage_rigid_calibration_audit = (
        _validate_foliage_rigid_calibration(
            initialization, surface_warmstart
        )
    )
    if (
        args.training_profile
        in {
            "hybrid_handoff_quality",
            "static_handoff_quality",
            "static_handoff_fast",
        }
        and surface_warmstart is None
        and args.resume is None
    ):
        raise RuntimeError(
            "hybrid_handoff_quality requires a validated native rigid "
            "surface handoff"
        )
    if (
        args.mature_handoff_surface_policy != "joint"
        and surface_warmstart is None
        and args.resume is None
    ):
        raise RuntimeError(
            "A non-joint mature surface policy requires a validated rigid "
            "surface handoff"
        )
    if (
        args.mature_handoff_surface_policy
        in {"appearance_only", "frozen"}
        and args.maximum_rigid_completion_seeds > 0
    ):
        print(
            "[WARN] Rigid completion births inherit the selected mature "
            "surface policy. Use --maximum-rigid-completion-seeds 0 when "
            "the goal is exact handoff preservation."
        )
    surface_learning_rate_offset = (
        int(surface_warmstart.get("surface_iteration", 0))
        if surface_warmstart is not None
        else 0
    )
    if surface_learning_rate_offset < 0:
        raise RuntimeError(
            "Rigid surface handoff has a negative training iteration"
        )
    inherited_surface_optimizer = (
        _inherit_surface_optimizer_contract(opt, surface_warmstart)
        if surface_warmstart is not None
        else None
    )
    if initialization["evidence_hash"] != evidence_store["evidence_hash"]:
        raise RuntimeError("Initialization/evidence hash mismatch")
    training_rgb_source = rgb_source_contract(dataset)
    _validate_initialization_rgb_source(
        initialization, training_rgb_source
    )
    representation_audit_warnings = []
    foliage_initialization_audit = initialization.get("foliage", {})
    selected_sequences = {
        str(row["sequence_id"])
        for row in foliage_initialization_audit.get("selected_views", [])
    }
    if mast3r_is_geometry_authority(evidence_store):
        gate_failures = []
        if not foliage_initialization_audit.get(
            "cross_sequence_visual_hull", False
        ):
            gate_failures.append("cross-sequence visual hull is false")
        if len(selected_sequences) < 2:
            gate_failures.append(
                f"only {len(selected_sequences)} selected sequence(s)"
            )
        if int(foliage_initialization_audit.get("canonical_crown", 0)) <= 0:
            gate_failures.append("canonical crown is empty")
        if int(foliage_initialization_audit.get("static_skeleton", 0)) <= 0:
            gate_failures.append("static trunk/branch skeleton is empty")
        if (
            args.reconstruction_target == "sequence_conditioned_legacy"
            and int(
                foliage_initialization_audit.get(
                    "sequence_local_dynamic_tracks", 0
                )
            )
            <= 0
        ):
            gate_failures.append(
                "legacy sequence-conditioned leaf branch is empty"
            )
        if gate_failures:
            representation_audit_warnings.extend(gate_failures)
            print(
                "[WARN] Soft representation audit: "
                + "; ".join(gate_failures)
            )
    resume = _load(args.resume.resolve()) if args.resume else None
    v38_causal_repair_resume = False
    if args.allow_v38_causal_repair_resume:
        resume_hashes = (resume or {}).get("implementation_hashes", {})
        saved_contract = (resume or {}).get("training_contract", {})
        v38_causal_repair_resume = bool(
            resume is not None
            and resume.get("protocol")
            == V38_CAUSAL_REPAIR_PREDECESSOR["protocol"]
            and int(resume.get("iteration", -1))
            == V38_CAUSAL_REPAIR_PREDECESSOR["iteration"]
            and int(resume.get("schedule_horizon", -1))
            == V38_CAUSAL_REPAIR_PREDECESSOR["schedule_horizon"]
            and saved_contract.get("volume_densify_until_iteration")
            == V38_CAUSAL_REPAIR_PREDECESSOR[
                "volume_densify_until_iteration"
            ]
            and int(args.phase_schedule_horizon)
            == V38_CAUSAL_REPAIR_PREDECESSOR["schedule_horizon"]
            and int(args.volume_densify_until_iteration) == 12_000
            and all(
                resume_hashes.get(name)
                == V38_CAUSAL_REPAIR_PREDECESSOR[name]
                for name in ("trainer", "training_evidence")
            )
        )
        if not v38_causal_repair_resume:
            raise RuntimeError(
                "v38 causal repair resume does not match the exact v35 "
                "9k checkpoint, implementation hashes, 12k schedule horizon "
                "and 10k->12k volume-topology extension"
            )
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(f"Refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    surface = GaussianModel(dataset.sh_degree)
    scene = LazyScene(
        dataset,
        surface,
        image_cache_size=args.view_cache_size,
        image_prefetch_workers=args.image_prefetch_workers,
    )
    views = scene.getTrainCameras()
    dav2_observation_patches = _load_dav2_observation_patches(
        Path(initialization["surface_seed"]),
        device=torch.device("cuda"),
    )
    training_view_names = {
        str(Path(str(view.image_name)).stem) for view in views
    }
    missing_dav2_patch_views = sorted(
        set(dav2_observation_patches) - training_view_names
    )
    if missing_dav2_patch_views:
        raise RuntimeError(
            "DAV2 source-view patch observations reference cameras absent "
            f"from the fixed training scene: {missing_dav2_patch_views[:8]}"
        )
    scene_contract = json.loads(
        Path(evidence_store["scene_contract"]).read_text(encoding="utf-8")
    )
    fixed_camera_validation = _validate_fixed_cameras(
        views, scene_contract
    )
    camera_runtime_contract = json.loads(
        (output / "camera_intrinsics_contract.json").read_text(
            encoding="utf-8"
        )
    )
    sequence_names = sorted(
        {sequence_id(view.image_name) for view in views}
    )
    sequence_index = {
        name: index for index, name in enumerate(sequence_names)
    }
    maximum_camera_id = max(int(view.colmap_id) for view in views)
    camera_sequence_lookup = torch.full(
        (maximum_camera_id + 1,),
        -1,
        dtype=torch.int16,
        device="cuda",
    )
    camera_frame_lookup = torch.zeros(
        maximum_camera_id + 1,
        dtype=torch.int32,
        device="cuda",
    )
    camera_forward_lookup = torch.full(
        (maximum_camera_id + 1, 3),
        float("nan"),
        dtype=torch.float32,
        device="cuda",
    )
    for view in views:
        camera_sequence_lookup[int(view.colmap_id)] = sequence_index[
            sequence_id(view.image_name)
        ]
        stem = Path(str(view.image_name)).stem
        camera_frame_lookup[int(view.colmap_id)] = int(
            stem.rsplit("frame", 1)[-1]
        )
        camera_forward = view.world_view_transform[:3, 2].to(
            device="cuda", dtype=torch.float32
        )
        camera_forward_lookup[int(view.colmap_id)] = (
            camera_forward / camera_forward.norm().clamp_min(1e-8)
        )
    if resume is None:
        if surface_warmstart is None:
            surface_audit = _initialize_surface(
                surface,
                Path(initialization["surface_seed"]),
                scene.cameras_extent,
                opt,
            )
        else:
            if (
                surface_warmstart["camera_geometry_sha256"]
                != camera_runtime_contract["camera_geometry_sha256"]
            ):
                raise RuntimeError(
                    "Rigid surface handoff camera geometry differs from the "
                    "mixed Teacher camera contract"
                )
            surface_audit = _initialize_surface_from_ply(
                surface,
                Path(surface_warmstart["surface_ply"]),
                Path(initialization["surface_seed"]),
                scene.cameras_extent,
                opt,
                maximum_completion_seeds=(
                    args.maximum_rigid_completion_seeds
                ),
            )
    else:
        if (
            resume.get("protocol") != PROTOCOL
            and not v38_causal_repair_resume
            and not args.allow_trainer_repair_resume
            and not args.allow_static_detail_isolated_repair_resume
            and not args.allow_static_canonical_ownership_repair_resume
            and not args.allow_surface_screen_evidence_repair_resume
        ):
            raise RuntimeError("Unified teacher resume protocol mismatch")
        if resume["evidence_hash"] != evidence_store["evidence_hash"]:
            raise RuntimeError("Resume/evidence hash mismatch")
        surface.restore(
            _surface_capture_to_device(
                resume["surface"], torch.device("cuda")
            ),
            opt,
        )
        surface_audit = resume["surface_audit"]
    if len(surface.get_xyz) > args.maximum_surface_gaussians:
        raise RuntimeError(
            "Surface checkpoint/initialization exceeds the configured budget: "
            f"{len(surface.get_xyz)} > {args.maximum_surface_gaussians}. "
            "Use a clean pre-topology checkpoint or compact it before resume."
        )
    rigid_surface_residual_start = int(
        surface_audit.get("rigid_background_completion", {}).get(
            "existing_surface_count", len(surface.get_xyz)
        )
    )
    if not 0 <= rigid_surface_residual_start <= len(surface.get_xyz):
        raise RuntimeError("Rigid handoff/residual surface boundary is invalid")
    surface_budget = min(
        int(args.maximum_surface_gaussians),
        max(
            len(surface.get_xyz),
            int(
                np.ceil(
                    float(surface_audit["surface_count"])
                    * float(args.maximum_surface_growth_multiplier)
                )
            ),
        ),
    )

    # Static reconstruction is a distinct representation contract, not the
    # legacy conditioned model with its temporal branch merely hidden at
    # render time.  New static runs therefore allocate rank-zero compatibility
    # tensors.  A resume instantiates the rank recorded by its checkpoint so
    # historical static checkpoints remain loadable without silent surgery.
    foliage_dynamic_rank = (
        int(resume["foliage"].get("dynamic_rank", args.dynamic_rank))
        if resume is not None
        else (
            0
            if args.reconstruction_target == "static"
            else int(args.dynamic_rank)
        )
    )
    foliage = VolumetricFoliageModel(
        dataset.sh_degree, dynamic_rank=foliage_dynamic_rank
    ).cuda()
    seed_payload = _load(Path(initialization["foliage_seed"]))
    static_fusion_audit = None
    if args.reconstruction_target == "static":
        fixed_camera_sequences = [
            {
                "image_id": int(view.colmap_id),
                "image_name": str(view.image_name),
                "sequence_id": sequence_id(view.image_name),
            }
            for view in views
        ]
        seed_payload, static_fusion_audit = (
            fuse_sequence_evidence_into_static_leaves(
                seed_payload,
                minimum_supporting_views=(
                    args.static_leaf_min_supporting_views
                ),
                canonical_sequence_policy=(
                    args.static_canonical_sequence_policy
                ),
                fixed_camera_sequences=fixed_camera_sequences,
            )
        )
    if resume is None:
        foliage.initialize_from_volume_state(seed_payload)
        # Dynamic leaves already come from real sequence-local pointmaps.
        # Cloning canonical crown primitives created two co-located alpha
        # owners and the opaque green paint layer seen in conditioned views.
        dynamic_seed_count = int(foliage.dynamic_leaf_mask.sum())
        if (
            args.reconstruction_target == "static"
            and dynamic_seed_count != 0
        ):
            raise RuntimeError(
                "Static production initialization retained sequence-owned "
                "leaf primitives"
            )
        static_replacement_group_association = (
            foliage.associate_static_detail_replacement_groups()
            if args.reconstruction_target == "static"
            else None
        )
    else:
        foliage.restore(resume["foliage"])
        dynamic_seed_count = int(resume["dynamic_seed_count"])
        static_replacement_group_association = resume.get(
            "training_contract", {}
        ).get("static_replacement_group_association")
    initial_volume_count = int(seed_payload["centers"].shape[0])
    volume_budget = min(
        int(args.maximum_volume_gaussians),
        max(
            len(foliage),
            int(
                np.ceil(
                    initial_volume_count
                    * float(args.maximum_volume_growth_multiplier)
                )
            ),
        ),
    )

    semantic_contract = Path(evidence_store["semantic_contract"])
    semantic_payload = json.loads(
        semantic_contract.read_text(encoding="utf-8")
    )
    fields = OutdoorTaskFieldLookup(
        Path(dataset.source_path),
        Path(semantic_payload["tree_mask_pickle"]),
        semantic_contract,
        multiview_track_archive=artifact_path(
            evidence_store, "mast3r_multiview_tracks"
        ),
        projected_rigid_posterior_archive=artifact_path(
            evidence_store,
            "projected_rigid_conflict_posterior",
            required=False,
        ),
        max_cached_views=0,
    )
    geometry = OutdoorGeometryEvidence(args.evidence_store)
    pointmap_posterior_preflight = _assert_pointmap_posterior_contract(
        geometry,
        allow_missing=bool(
            args.allow_missing_pointmap_cross_sequence_posterior
        ),
    )
    foliage_rays = FoliageRayEvidence(seed_payload.get("ray_evidence"))
    static_ray_births = StaticRayBirthAccumulator(
        voxel_size=args.static_ray_birth_voxel_size,
        visual_hull_voxel_size=(
            args.static_ray_birth_visual_hull_voxel_size
        ),
        maximum_segment_samples=(
            args.static_ray_birth_maximum_segment_samples
        ),
    )
    if resume is not None:
        foliage_rays.restore_runtime_state(
            resume.get("foliage_ray_runtime_state")
        )
        static_ray_birth_state = resume.get("static_ray_birth_state")
        static_ray_births.restore(static_ray_birth_state)
        # Accumulator v3/v4 checkpoints deleted drained keys without a
        # tombstone. Reconstruct those keys from extant runtime-birth
        # provenance before another ray proposal can recreate the same static
        # occupancy cell after resume.
        if (static_ray_birth_state or {}).get("version") != (
            "static-ray-birth-accumulator-v5"
        ):
            runtime_birth_rows = foliage.initialization_source == 6
            static_ray_births.mark_consumed_centers(
                foliage.xyz.detach()[runtime_birth_rows]
            )
    foliage_ray_evidence_audit = foliage_rays.audit()
    per_camera_ray_rows = tuple(
        int(audit["count"])
        for audit in foliage_ray_evidence_audit[
            "camera_epoch_audits"
        ].values()
    )
    per_camera_unvisited_ray_rows = tuple(
        int(audit["never_visited"])
        for audit in foliage_ray_evidence_audit[
            "camera_epoch_audits"
        ].values()
    )
    # The per-camera epoch scheduler cannot finish one evidence pass when the
    # configured batch capacity is smaller than the immutable free/hit table.
    # Derive a deterministic lower bound from the final schedule horizon, not
    # the length of a diagnostic prefix, so prefix/resume runs keep exactly
    # the same factor contract.
    (
        clean_epoch_batch,
        clean_epoch_minimum_batch,
        scheduled_ray_factor_calls,
    ) = _complete_evidence_epoch_batch_size(
        args.ray_posterior_maximum_rays,
        foliage_ray_evidence_audit["interval_effective_rows"],
        args.phase_schedule_horizon,
        args.ray_posterior_every,
        per_camera_rows=per_camera_ray_rows,
    )
    prior_ray_factor_calls = int(
        foliage_ray_evidence_audit["interval_factor_calls"]
    )
    available_ray_factor_calls = (
        scheduled_ray_factor_calls - prior_ray_factor_calls
    )
    if available_ray_factor_calls <= 0:
        raise RuntimeError(
            "No scheduled foliage ray calls remain after checkpoint resume"
        )
    remaining_unvisited_ray_rows = sum(
        per_camera_unvisited_ray_rows
    )
    (
        effective_ray_posterior_maximum_rays,
        ray_epoch_minimum_batch,
        capacity_ray_factor_calls,
    ) = _complete_evidence_epoch_batch_size(
        args.ray_posterior_maximum_rays,
        remaining_unvisited_ray_rows,
        args.phase_schedule_horizon,
        args.ray_posterior_every,
        per_camera_rows=per_camera_unvisited_ray_rows,
        available_factor_calls=available_ray_factor_calls,
    )
    effective_ray_posterior_maximum_rays = (
        _prefix_stable_resume_ray_batch(
            effective_ray_posterior_maximum_rays,
            resume,
            per_camera_unvisited_ray_rows,
            capacity_ray_factor_calls,
        )
    )
    clean_epoch_aggregate_minimum_batch = int(
        np.ceil(
            1.05
            * foliage_ray_evidence_audit["interval_effective_rows"]
            / scheduled_ray_factor_calls
        )
    )
    runtime_epoch_aggregate_minimum_batch = int(
        np.ceil(
            1.05
            * remaining_unvisited_ray_rows
            / capacity_ray_factor_calls
        )
    )
    required_ray_factor_calls = sum(
        (
            count + effective_ray_posterior_maximum_rays - 1
        )
        // effective_ray_posterior_maximum_rays
        for count in per_camera_unvisited_ray_rows
        if count > 0
    )
    view_by_camera_id = {
        int(view.colmap_id): view for view in views
    }
    missing_ray_cameras = sorted(
        set(int(value) for value in foliage_rays.camera_id_values.tolist())
        - set(view_by_camera_id)
    )
    if missing_ray_cameras:
        raise RuntimeError(
            "Foliage ray posterior references calibrated cameras absent from "
            f"the training scene: {missing_ray_cameras[:8]}"
        )
    if foliage_ray_evidence_audit["hit_count"] <= 0:
        representation_audit_warnings.append(
            "foliage ray table has no confirmed hit interval"
        )
    if foliage_ray_evidence_audit["confirmed_free_count"] <= 0:
        representation_audit_warnings.append(
            "foliage ray table has no confirmed free-space interval"
        )
    chart_surface = ChartSurfaceModel(
        Path(initialization["surface_seed"]),
        evidence_store=args.evidence_store,
        seed=args.seed + 19,
        maximum_tangent_scale=args.maximum_surface_scale,
    ).cuda()
    chart_runtime_audit = chart_surface.audit()
    required_chart_contract = {
        "continuous_learnable_inverse_depth_atlas": True,
        "uv_domain_quadtree_densification": True,
        "renderer_binding": "native_2d_surfel_geometry_override",
    }
    chart_contract_mismatch = {
        key: (chart_runtime_audit.get(key), expected)
        for key, expected in required_chart_contract.items()
        if chart_runtime_audit.get(key) != expected
    }
    if chart_contract_mismatch:
        raise RuntimeError(
            "Runtime Chart atlas contradicts the declared mainline: "
            f"{chart_contract_mismatch}"
        )
    # The provider changes only the native 2D surfel geometry tensors passed
    # to the mixed rasterizer. It does not alter the renderer API or create a
    # second compositing pass.
    surface._chart_geometry_provider = chart_surface
    if resume is not None and resume.get("chart_surface") is not None:
        chart_surface.restore(resume["chart_surface"])
    elif (
        surface_warmstart is not None
        and surface_warmstart.get("chart_surface_state") is not None
    ):
        chart_surface.restore(
            _load(Path(surface_warmstart["chart_surface_state"]))
        )
    chart_atlas_learning_rate = float(args.chart_atlas_lr) * (
        float(args.mature_chart_atlas_lr_scale)
        if args.mature_handoff_surface_policy == "atlas_residual"
        else 1.0
    )
    chart_atlas_optimizer = torch.optim.Adam(
        list(chart_surface.parameters()), lr=chart_atlas_learning_rate
    )
    if (
        resume is not None
        and resume.get("chart_atlas_optimizer") is not None
    ):
        chart_atlas_optimizer.load_state_dict(
            resume["chart_atlas_optimizer"]
        )
    for group in chart_atlas_optimizer.param_groups:
        group["base_lr"] = float(chart_atlas_learning_rate)
    appearance = OutdoorAppearanceUncertainty(
        [view.image_name for view in views],
        rank=args.dynamic_rank,
        spatial_grid_size=24,
        maximum_rgb_residual=args.appearance_maximum_rgb_residual,
        device="cuda",
    )
    sky = CanonicalDirectionalSky(degree=2).cuda()
    if resume is not None:
        appearance.restore(resume["appearance"])
        sky.load_state_dict(resume["sky"])
    volume_optimizer = _volume_optimizer(
        args, foliage, appearance, sky
    )
    if resume is not None:
        uncertainty_optimizer_migrated = _restore_volume_optimizer_state(
            volume_optimizer,
            resume["volume_optimizer"],
            allow_static_uncertainty_extension=bool(
                args.allow_trainer_repair_resume
                and args.reconstruction_target == "static"
            ),
        )
        if uncertainty_optimizer_migrated:
            print(
                "Migrated the v4 appearance Adam group by appending the "
                "state-free stripe-free spatial uncertainty decoder; all "
                "existing foliage/appearance/sky moments were preserved."
            )
        if args.allow_trainer_repair_resume:
            dynamic = foliage.dynamic_leaf_mask
            dynamic_opacity = foliage.opacities[dynamic]
            if (
                bool(dynamic.any())
                and float(dynamic_opacity.median()) < 0.005
            ):
                with torch.no_grad():
                    repaired = dynamic & (
                        foliage.opacities < 0.01
                    )
                    foliage.opacity_logits[repaired, 0] = (
                        foliage.opacity_logits.new_tensor(0.01).logit()
                    )
                    state = volume_optimizer.state.get(
                        foliage.opacity_logits, {}
                    )
                    for value in state.values():
                        if (
                            torch.is_tensor(value)
                            and value.shape
                            == foliage.opacity_logits.shape
                        ):
                            value[repaired] = 0
                print(
                    "Trainer repair restored dynamic opacity prior and "
                    "cleared stale opacity Adam moments for "
                    f"{int(repaired.sum())} leaves."
                )

    rgb_schedule = _full_epoch_schedule(
        len(views), args.phase_schedule_horizon, args.seed + 1
    )
    view_index_by_camera_id = {
        int(view.colmap_id): index for index, view in enumerate(views)
    }
    static_detail_canopy_fraction_minimum = 0.005
    all_canopy_view_indices = [
        index
        for index, view in enumerate(views)
        if fields.canopy_fraction(
            view.image_name, (128, 128)
        )
        > static_detail_canopy_fraction_minimum
    ]
    verified_static_detail = (
        foliage.static_leaf_mask
        & (foliage.verification_state == VERIFICATION_VERIFIED)
        & (foliage.verified_camera_count >= 2)
    )
    detail_support_camera_ids = {
        int(camera_id)
        for camera_id in foliage.support_camera_ids[
            verified_static_detail
        ].detach().cpu().numpy().reshape(-1)
        if int(camera_id) >= 0
    }
    canopy_view_index_set = set(all_canopy_view_indices)
    static_detail_view_indices = sorted(
        {
            view_index_by_camera_id[camera_id]
            for camera_id in detail_support_camera_ids
            if camera_id in view_index_by_camera_id
            and view_index_by_camera_id[camera_id] in canopy_view_index_set
        }
    )
    if not static_detail_view_indices:
        static_detail_view_indices = (
            all_canopy_view_indices
            if all_canopy_view_indices
            else list(range(len(views)))
        )
    static_detail_schedule = _cycle_schedule(
        static_detail_view_indices,
        args.phase_schedule_horizon,
        args.seed + 5,
    )
    static_detail_schedule_audit = {
        "contract": (
            "uniform_complete_epochs_over_verified_static_detail_exact_"
            "support_cameras__geometry_and_sh_exact_camera_owned__opacity_"
            "read_only_in_isolated_stream"
        ),
        "exact_support_camera_count": int(len(detail_support_camera_ids)),
        "camera_count": int(len(static_detail_view_indices)),
        "canopy_fraction_minimum": (
            static_detail_canopy_fraction_minimum
        ),
        "schedule_sha256": _schedule_digest(static_detail_schedule),
    }
    multiview_role_stems = (
        {
            Path(name).stem
            for name in fields.multiview_roles.view_names
        }
        if fields.multiview_roles is not None
        else set()
    )
    posterior_skeleton_view_indices = {
        index
        for index, view in enumerate(views)
        if Path(str(view.image_name)).stem in multiview_role_stems
    }
    skeleton_observation_camera_ids = (
        foliage.observation_camera_ids[
            foliage.static_skeleton_mask
        ]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    model_skeleton_view_indices = {
        view_index_by_camera_id[int(camera_id)]
        for camera_id in np.unique(skeleton_observation_camera_ids)
        if int(camera_id) >= 0
        and int(camera_id) in view_index_by_camera_id
    }
    static_skeleton_view_indices = sorted(
        posterior_skeleton_view_indices | model_skeleton_view_indices
    )
    static_skeleton_schedule = (
        _cycle_schedule(
            static_skeleton_view_indices,
            args.phase_schedule_horizon,
            args.seed + 7,
        )
        if static_skeleton_view_indices
        else np.full(args.phase_schedule_horizon, -1, dtype=np.int64)
    )
    static_skeleton_schedule_audit = {
        "contract": (
            "uniform_complete_epochs_over_multiview_role_posterior_or_"
            "immutable_skeleton_observation_cameras__surface_plus_static_"
            "skeleton_role_isolated"
        ),
        "camera_count": int(len(static_skeleton_view_indices)),
        "multiview_posterior_camera_count": int(
            len(posterior_skeleton_view_indices)
        ),
        "model_observation_camera_count": int(
            len(model_skeleton_view_indices)
        ),
        "schedule_sha256": _schedule_digest(static_skeleton_schedule),
        "task_field_posterior": fields.audit()[
            "multiview_tree_role_posterior"
        ],
    }
    # The volume-isolated stream owns only verified detail bandwidth. Cycle
    # exact support cameras; envelope and skeleton retain independent low-pass
    # and role-isolated lifecycles and never receive this objective.
    static_volume_camera_ids = set(detail_support_camera_ids)
    static_volume_view_indices = sorted(
        {
            view_index_by_camera_id[camera_id]
            for camera_id in static_volume_camera_ids
            if camera_id in view_index_by_camera_id
        }
    )
    static_volume_schedule = (
        _cycle_schedule(
            static_volume_view_indices,
            args.phase_schedule_horizon,
            args.seed + 8,
        )
        if static_volume_view_indices
        else np.full(args.phase_schedule_horizon, -1, dtype=np.int64)
    )
    static_volume_schedule_audit = {
        "contract": (
            "uniform_complete_epochs_over_verified_static_detail_exact_"
            "support_cameras__detail_only_intrinsic_rgb_high_bandwidth"
        ),
        "camera_count": int(len(static_volume_view_indices)),
        "verified_detail_support_camera_count": int(
            len(detail_support_camera_ids)
        ),
        "schedule_sha256": _schedule_digest(static_volume_schedule),
    }
    observed_camera_ids = (
        foliage.observation_camera_ids[
            foliage.dynamic_leaf_mask
        ]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    dynamic_evidence_indices = [
        view_index_by_camera_id[int(camera_id)]
        for camera_id in np.unique(observed_camera_ids)
        if int(camera_id) >= 0
        and int(camera_id) in view_index_by_camera_id
    ]
    dynamic_evidence_index_set = set(dynamic_evidence_indices)
    automatic_evidence_fraction = (
        args.dynamic_evidence_view_fraction is None
    )
    if automatic_evidence_fraction:
        args.dynamic_evidence_view_fraction = (
            _balanced_evidence_fraction(
                len(views), len(dynamic_evidence_indices)
            )
        )
    conditioned_active_mask = np.asarray(
        [
            _conditioned_branch_active(
                step,
                args.phase_schedule_horizon,
                args.training_profile,
                args.reconstruction_target,
            )
            and _phase(
                step,
                args.phase_schedule_horizon,
                args.training_profile,
            )
            != "canonical_polish"
            for step in range(args.phase_schedule_horizon)
        ],
        dtype=bool,
    )
    if (
        resume is not None
        and args.allow_conditioned_schedule_repair_resume
    ):
        # A repair continuation cannot retroactively consume protected visits.
        # Rebuild the coverage contract over the remaining active suffix so
        # every database view receives as many complete visits as the
        # remaining schedule can physically hold.
        conditioned_active_mask[
            : int(resume["iteration"])
        ] = False
    conditioned_schedule, conditioned_schedule_audit = (
        _evidence_biased_schedule(
            rgb_schedule,
            dynamic_evidence_indices,
            fraction=args.dynamic_evidence_view_fraction,
            seed=args.seed + 4,
            active_mask=conditioned_active_mask,
            minimum_full_scene_visits=3,
        )
    )
    conditioned_schedule_audit.update(
        {
            "fraction_policy": (
                "automatic_12x_per_view"
                if automatic_evidence_fraction
                else "explicit"
            ),
            "evidence_to_other_per_view_ratio": None,
        }
    )
    conditioned_counts = np.bincount(
        conditioned_schedule, minlength=len(views)
    ).astype(np.float64)
    evidence_mask = np.zeros(len(views), dtype=bool)
    evidence_mask[dynamic_evidence_indices] = True
    if (
        bool(evidence_mask.any())
        and bool((~evidence_mask).any())
        and conditioned_counts[~evidence_mask].mean() > 0
    ):
        conditioned_schedule_audit[
            "evidence_to_other_per_view_ratio"
        ] = float(
            conditioned_counts[evidence_mask].mean()
            / conditioned_counts[~evidence_mask].mean()
        )
    by_stem = {
        Path(str(view.image_name)).stem: index
        for index, view in enumerate(views)
    }
    metric_geometry_indices = [
        by_stem[stem]
        for stem in geometry.metric_view_stems
        if stem in by_stem
    ]
    ordinal_geometry_indices = [
        by_stem[stem]
        for stem in geometry.ordinal_view_stems
        if stem in by_stem
    ]
    if metric_geometry_indices and ordinal_geometry_indices:
        metric_repeat = int(
            np.ceil(
                len(ordinal_geometry_indices)
                / len(metric_geometry_indices)
            )
        )
        geometry_indices = (
            metric_geometry_indices * metric_repeat
            + ordinal_geometry_indices
        )
    else:
        geometry_indices = (
            metric_geometry_indices or ordinal_geometry_indices
        )
    geometry_schedule = _cycle_schedule(
        geometry_indices, args.phase_schedule_horizon, args.seed + 2
    )
    # Extra surface-topology views should expose tree/building boundaries, not
    # be the views with the most canopy.  The old schedule then froze the
    # surface and accidentally trained only the sky model on those tree
    # pixels.  A mixed-coverage score focuses this stream on rigid content
    # adjacent to occluders while the normal RGB epoch still covers every
    # camera uniformly.
    canopy_fraction = [
        fields.canopy_fraction(
            view.image_name,
            # This is a view-priority statistic, not a supervision field.
            (128, 128),
        )
        for view in views
    ]
    boundary_indices = sorted(
        range(len(views)),
        key=lambda index: (
            canopy_fraction[index] * (1.0 - canopy_fraction[index])
        ),
        reverse=True,
    )[: max(64, min(256, len(views)))]
    topology_schedule = _cycle_schedule(
        boundary_indices, args.phase_schedule_horizon, args.seed + 3
    )
    camera_schedules, restored_schedule_names = (
        _restore_resume_camera_schedules(
            {
                "rgb": rgb_schedule,
                "conditioned": conditioned_schedule,
                "geometry": geometry_schedule,
                "topology": topology_schedule,
                "static_detail": static_detail_schedule,
                "static_skeleton": static_skeleton_schedule,
                "static_volume": static_volume_schedule,
            },
            resume,
            horizon=args.phase_schedule_horizon,
            allow_conditioned_repair=(
                args.allow_conditioned_schedule_repair_resume
            ),
        )
    )
    rgb_schedule = camera_schedules["rgb"]
    conditioned_schedule = camera_schedules["conditioned"]
    geometry_schedule = camera_schedules["geometry"]
    topology_schedule = camera_schedules["topology"]
    static_detail_schedule = camera_schedules["static_detail"]
    static_skeleton_schedule = camera_schedules["static_skeleton"]
    static_volume_schedule = camera_schedules["static_volume"]
    if resume is not None:
        saved_contract = resume.get("training_contract", {})
        if "conditioned" in restored_schedule_names:
            conditioned_schedule_audit = copy.deepcopy(
                saved_contract.get(
                    "conditioned_sampling", conditioned_schedule_audit
                )
            )
        saved_static_audits = {
            "static_detail": "static_detail_isolated_supervision",
            "static_skeleton": "static_skeleton_isolated_supervision",
            "static_volume": "static_volume_isolated_supervision",
        }
        for schedule_name, contract_name in saved_static_audits.items():
            if schedule_name not in restored_schedule_names:
                continue
            saved_audit = saved_contract.get(contract_name, {}).get(
                "view_schedule"
            )
            if saved_audit is not None:
                if schedule_name == "static_detail":
                    static_detail_schedule_audit = copy.deepcopy(saved_audit)
                elif schedule_name == "static_skeleton":
                    static_skeleton_schedule_audit = copy.deepcopy(saved_audit)
                else:
                    static_volume_schedule_audit = copy.deepcopy(saved_audit)
    schedule_hash = _schedule_digest(
        rgb_schedule,
        conditioned_schedule,
        geometry_schedule,
        topology_schedule,
        static_detail_schedule,
        static_skeleton_schedule,
        static_volume_schedule,
    )
    deployment_optical_replacement_policy = (
        "disabled"
        if args.reconstruction_target == "static"
        else args.deployment_optical_replacement_policy
    )
    deployment_optical_responsibility_prior = (
        0.0
        if args.reconstruction_target == "static"
        else float(args.deployment_optical_responsibility_prior)
    )
    training_contract = {
        "reconstruction_target": args.reconstruction_target,
        "foliage_representation": {
            "dynamic_rank": int(foliage.dynamic_rank),
            "temporal_parameters_allocated": bool(
                foliage.dynamic_rank > 0
            ),
            "contract": (
                "rank_zero_unconditional_static_volume"
                if args.reconstruction_target == "static"
                and foliage.dynamic_rank == 0
                else "legacy_sequence_conditioned_compatibility"
            ),
        },
        "pointmap_cross_sequence_posterior_preflight": (
            pointmap_posterior_preflight
        ),
        "task_field_ownership": fields.audit(),
        "static_fusion": static_fusion_audit,
        "static_replacement_group_association": (
            static_replacement_group_association
        ),
        "integrated_optical_mass_compensation": bool(
            args.integrated_optical_mass_compensation
        ),
        "static_optical_policy_contract": (
            STATIC_OPTICAL_POLICY_CONTRACT
            if args.reconstruction_target == "static"
            else None
        ),
        "static_ray_local_mass_handoff": {
            "evidence_every": int(args.static_replacement_evidence_every),
            "ema_decay": float(args.static_replacement_ema_decay),
            "maximum_fraction_per_event": float(
                args.static_replacement_mass_fraction_per_event
            ),
            "candidate_contract": (
                "verified_same_group_exact_support_camera__primitive_center_"
                "cannot_veto_pixel_ray_evidence"
            ),
            "audit_camera_schedule": (
                "complete_cycle_over_verified_detail_exact_support_cameras"
            ),
            "authority_contract": (
                "native_t_before_alpha_pixel_coverage_depth_rigid_safe_"
                "distinct_view_persistent"
            ),
            "transition_contract": (
                "reversible_integrated_optical_mass_reference"
            ),
        },
        "child_verification_lifecycle": {
            "grace_iterations": int(
                args.child_verification_grace_iterations
            ),
            "timeout_iterations": int(
                args.child_verification_timeout_iterations
            ),
            "debt_soft_fraction": float(
                args.volume_verification_debt_soft_fraction
            ),
            "debt_hard_fraction": float(
                args.volume_verification_debt_hard_fraction
            ),
            "minimum_split_capacity_scale": float(
                args.volume_verification_debt_minimum_split_scale
            ),
            "minimum_strict_ray_birth_capacity_scale": float(
                args.volume_verification_debt_minimum_birth_scale
            ),
            "new_child_state": "unverified",
            "parent_observations": "candidates_not_proof",
            "verification": (
                "native_real_ray_contribution_plus_candidate_camera_"
                "multiview_sequence"
            ),
            "unverified_permissions": {
                "dc_and_mass_growth": True,
                "higher_order_sh": False,
                "split": False,
                "ordinary_prune_during_grace": False,
            },
        },
        "static_training_stages": (
            {
                "rgb_role_contract": STATIC_STAGE_RGB_ROLE_CONTRACT,
                "rigid_stage": "validated_native_2dgs_handoff",
                "envelope_only_through_iteration": dict(
                    _resolved_phase_schedule(
                        args.training_profile,
                        args.phase_schedule_horizon,
                    )
                ).get("topology"),
                "detail_activation": (
                    "static_foliage_phase__no_sequence_visibility_gate"
                ),
                "envelope_training_signals": [
                    "all_view_ray_free_space",
                    "cross_sequence_hit_geometry",
                    "stage2_canonical_low_frequency_rgb_geometry_mass",
                    "stage3_all_view_dc_only_no_positive_mass_growth",
                    "persistent_view_depth_local_replacement",
                    "envelope_only_all_view_rigid_free_counterfactual_cleanup",
                ],
                "detail_training_signals": [
                    "positive_evidence_sequence_ray_free_space",
                    "positive_evidence_sequence_or_verified_consensus_ray_hit_interval",
                    "exact_support_verified_canonical_rgb_geometry",
                    "exact_support_plus_soft_same_sequence_detail_sh_appearance",
                    "exact_support_plus_soft_positive_evidence_sequence_optical_mass",
                    "exact_support_verified_surface_plus_detail_geometry",
                    "surface_plus_detail_isolated_screen_gradient",
                    "positive_evidence_sequence_rigid_free_counterfactual_cleanup",
                    "robust_multiview_color",
                    "ray_local_optical_depth_conserving_replacement",
                ],
                "joint_polish": (
                    "freeze_xyz_scale_rotation_mass_topology_together__"
                    "train_sh_sky_only"
                ),
            }
            if args.reconstruction_target == "static"
            else None
        ),
        "static_detail_isolated_supervision": {
            "contract": STATIC_DETAIL_ISOLATED_CONTRACT,
            "every": int(args.static_detail_isolated_every),
            "weight": float(args.static_detail_isolated_weight),
            "active_only_after_static_detail_stage": True,
            "view_schedule": static_detail_schedule_audit,
            "gradient_owners": [
                "verified_exact_support_static_detail_xyz_scale_rotation",
                "verified_exact_support_static_detail_geometry_rgb",
                "verified_soft_same_sequence_static_detail_sh_rgb",
                "verified_consensus_static_detail_ray_optical_mass",
                "verified_exact_support_static_detail_sh",
                "verified_exact_support_static_detail_means2d_topology",
            ],
            "excluded_owners": [
                "persistent_envelope",
                "skeleton",
                "static_detail_optical_mass",
                "surface",
                "sky",
                "uncertainty",
            ],
        },
        "static_skeleton_isolated_supervision": {
            "contract": (
                "surface_plus_static_skeleton__cross_sequence_track_role_"
                "posterior_or_detached_role_projection_clipped_by_real_"
                "canopy_mask__rgb_edge_and_screen_bandwidth"
            ),
            "every": int(args.static_skeleton_isolated_every),
            "weight": float(args.static_skeleton_isolated_weight),
            "view_schedule": static_skeleton_schedule_audit,
            "gradient_owners": [
                "static_skeleton_xyz_scale_rotation",
                "static_skeleton_optical_mass_sh",
                "static_skeleton_means2d_topology",
            ],
            "excluded_owners": [
                "persistent_envelope",
                "static_detail",
                "surface",
                "sky",
                "uncertainty",
            ],
        },
        "static_volume_isolated_supervision": {
            "contract": (
                "volume_only_intrinsic_rgb_on_real_canopy_and_detached_"
                "projected_support__geometry_sh_screen_bandwidth_only__"
                "opacity_read_only"
            ),
            "every": int(args.static_volume_isolated_every),
            "weight": float(args.static_volume_isolated_weight),
            "view_schedule": static_volume_schedule_audit,
            "gradient_owners": [
                "verified_exact_support_static_detail_xyz_scale_rotation",
                "verified_soft_same_sequence_static_detail_sh",
                "verified_exact_support_static_detail_means2d_topology",
            ],
            "excluded_owners": [
                "static_volume_opacity",
                "persistent_envelope",
                "skeleton",
                "surface",
                "sky",
                "uncertainty",
            ],
        },
        "static_detail_global_cleanup": {
            "contract": STATIC_DETAIL_GLOBAL_CLEANUP_CONTRACT,
            "every": int(args.static_detail_global_cleanup_every),
            "weight": float(args.static_detail_global_cleanup_weight),
            "counterfactual_weight": float(
                args.static_detail_global_counterfactual_weight
            ),
            "counterfactual_weight_is_independent": True,
            "camera_schedule": (
                "uniform_authoritative_rgb_epoch_with_per_primitive_"
                "positive_evidence_sequence_permission"
            ),
            "forward_roles": ["surface", "static_detail"],
            "gradient_owners": [
                "static_detail_xyz_scale_rotation",
                "static_detail_optical_mass",
            ],
            "excluded_owners": [
                "static_detail_sh",
                "persistent_envelope",
                "skeleton",
                "surface",
                "sky",
                "uncertainty",
                "topology_statistics",
            ],
        },
        "persistent_envelope_global_cleanup": {
            "contract": PERSISTENT_ENVELOPE_GLOBAL_CLEANUP_CONTRACT,
            "every": int(args.persistent_envelope_global_cleanup_every),
            "weight": float(
                args.persistent_envelope_global_cleanup_weight
            ),
            "counterfactual_weight": float(
                args.persistent_envelope_global_counterfactual_weight
            ),
            "counterfactual_weight_is_independent": True,
            "camera_schedule": "uniform_authoritative_rgb_epoch",
            "forward_roles": ["surface", "persistent_envelope"],
            "gradient_owners": [
                "persistent_envelope_xyz_scale_rotation",
                "persistent_envelope_optical_mass",
            ],
            "excluded_owners": [
                "persistent_envelope_sh",
                "static_detail",
                "skeleton",
                "surface",
                "sky",
                "uncertainty",
                "topology_statistics",
            ],
        },
        "static_ray_birth": {
            "contract": STATIC_RAY_BIRTH_VISUAL_HULL_CONTRACT,
            "midpoint_voxel_size": float(
                args.static_ray_birth_voxel_size
            ),
            "visual_hull_voxel_size": float(
                args.static_ray_birth_visual_hull_voxel_size
            ),
            "maximum_segment_samples": int(
                args.static_ray_birth_maximum_segment_samples
            ),
            "minimum_cameras": 2,
            "minimum_sequences": 2,
            "maximum_births_per_topology_event": int(
                args.maximum_static_ray_births_per_event
            ),
            "verification_debt_backpressure": (
                "same_continuous_smoothstep_capacity_as_ordinary_split"
            ),
            "initial_opacity": 0.04,
            "consumed_cell_policy": (
                "persistent_midpoint_and_visual_hull_tombstones"
            ),
            "legacy_tombstone_recovery": (
                "extant_initialization_source_6_centers"
            ),
            "initial_mass_policy": (
                "support_ray_local_transfer_from_verified_envelope__"
                "bounded_additive_fallback_for_strict_visual_hull_without_"
                "donor"
            ),
            "donor_transfer_component_is_mass_conserving": True,
            "unfunded_visual_hull_remainder_uses_shared_additive_budget": True,
            "ownerless_additive_mass_fraction_per_event": float(
                args.static_ray_birth_additive_mass_fraction_per_event
            ),
            "ownerless_additive_mass_is_explicitly_bounded": True,
            "initial_mass_reversible_failure_rollback": True,
            "capacity_order": (
                "cross_sequence_uncovered_birth_before_ordinary_split"
            ),
            "positive_permission": "cross_sequence_hit_intersection_only",
            "negative_permission": "all_calibrated_free_and_rigid_spill",
        },
        "static_detail_topology": {
            "exclusive_after_envelope_stage": bool(
                args.static_detail_exclusive_topology
            ),
            "envelope_topology_phase": "topology",
            "detail_topology_phase": "static_foliage",
            "envelope_after_stage": (
                "mass_trainable_and_contradiction_prunable_no_split"
            ),
            "detail_screen_gradient_source": (
                "authoritative_full_render_plus_surface_detail_isolated"
            ),
            "capacity_contract": (
                "same_global_budget_reallocated_not_added"
            ),
        },
        "static_detail_receiver_materialization": {
            "contract": (
                "one_visible_evidence_supported_receiver_per_missing_"
                "replacement_group"
            ),
            "maximum_per_event": int(
                args.maximum_static_detail_materializations_per_event
            ),
            "initial_optical_mass_fraction": float(
                args.static_detail_materialization_mass_fraction
            ),
            "birth_forward_contract": (
                "co_located_same_covariance_same_color_tau_partition"
            ),
            "integrated_optical_mass_conserved": True,
            "existing_detail_group_repeat_forbidden": True,
            "selection": (
                "continuous_real_render_residual_footprint_rigidity_"
                "occupancy_priority_with_tree_spatial_fairness"
            ),
            "capacity_order": (
                "missing_group_receiver_before_strict_ray_birth_and_"
                "ordinary_detail_split"
            ),
        },
        "static_detail_canonical_ownership": {
            "enabled": bool(args.static_detail_canonical_ownership),
            "forward_visibility": "unconditional_static",
            "geometry_topology_owner": (
                "persisted_exact_support_cameras_and_verified_multiview"
            ),
            "appearance_owner": (
                "exact_support_camera_weight1_plus_same_acquisition_"
                "sequence_soft_weight"
            ),
            "same_sequence_appearance_weight": float(
                args.static_detail_same_sequence_appearance_weight
            ),
            "optical_mass_owner": (
                "exact_support_camera_weight1_plus_positive_evidence_"
                "sequence_soft_weight"
            ),
            "same_sequence_optical_weight": float(
                args.static_detail_same_sequence_optical_weight
            ),
            "positive_ray_owner": (
                "persisted_positive_evidence_sequences_or_verified_two_"
                "sequence_consensus"
            ),
            "owned_signals": [
                "canonical_rgb",
                "canonical_high_frequency",
                "detail_isolated_rgb_high_frequency",
                "positive_evidence_sequence_or_verified_consensus_positive_"
                "ray_hit_interval",
                "screen_gradient_topology",
            ],
            "noncanonical_views": (
                "render_unconditionally__positive_and_negative_optical_"
                "evidence_are_symmetric_within_persisted_evidence_"
                "sequences__same_sequence_soft_appearance_and_mass__no_"
                "positive_geometry_or_topology"
            ),
        },
        "parameter_loss_permission_matrix": {
            "chart_inverse_depth": [
                "track",
                "chart",
                "plane",
                "rigid_rgb",
            ],
            "surface_sh": ["rigid_rgb", "rigid_edge"],
            "envelope_xyz_scale": [
                "ray_free_hit",
                "cross_sequence_geometry",
                "stage2_canopy_rgb_only",
                "all_view_rigid_free_counterfactual_cleanup",
            ],
            "envelope_optical_mass": [
                "ray_free_hit",
                "stage2_canopy_rgb_only",
                "persistent_local_replacement",
                "all_view_rigid_free_counterfactual_cleanup",
            ],
            "envelope_sh": [
                "stage2_all_view_canopy_rgb_all_coefficients",
                "stage3_all_view_canopy_rgb_dc_only",
            ],
            "static_leaf_xyz_scale": [
                "positive_evidence_sequence_ray_free_space",
                "positive_evidence_sequence_or_verified_consensus_ray_hit",
                "verified_exact_support_canonical_rgb",
                "verified_exact_support_detail_isolated_rgb_high_frequency",
                "positive_evidence_sequence_rigid_free_counterfactual_cleanup",
            ],
            "static_leaf_optical_mass": [
                "positive_evidence_sequence_ray_free_space",
                "positive_evidence_sequence_or_verified_consensus_ray_hit",
                "exact_support_plus_soft_positive_evidence_sequence_"
                "canonical_rgb_dc_mass",
                "rigid_spill",
                "positive_evidence_sequence_rigid_free_counterfactual_cleanup",
            ],
            "static_leaf_sh": [
                "robust_canonical_rgb",
                "verified_exact_plus_soft_same_sequence_high_frequency",
                "verified_exact_plus_soft_same_sequence_detail_isolated_"
                "rgb_high_frequency",
            ],
            "uncertainty": ["photometric_likelihood_only"],
        },
        # The method schedule is resolved independently of an early
        # diagnostic stop. Checkpoints may therefore be resumed to a later
        # stop without changing phases, camera samples or topology cadence.
        "schedule_horizon": int(args.phase_schedule_horizon),
        "training_profile": args.training_profile,
        "resolved_phase_schedule": _resolved_phase_schedule(
            args.training_profile, args.phase_schedule_horizon
        ),
        # Evidence hashes alone do not identify the renderer seeds: different
        # candidate budgets, view counts, DAV2 hole filling, or random seeds
        # can all derive distinct initializations from the same evidence
        # store.  Resuming across them silently mixes two causal experiments.
        "initialization_version": initialization.get("version"),
        "initialization_manifest_sha256": _file_sha256(
            args.initialization / "initialization_manifest.json"
        ),
        "surface_seed_sha256": _file_sha256(
            Path(initialization["surface_seed"])
        ),
        "surface_initialization_mode": (
            "native_rigid_surface_handoff"
            if surface_warmstart is not None
            else "role_aware_evidence_seed"
        ),
        "rigid_background_completion": {
            "maximum_seeds": int(
                args.maximum_rigid_completion_seeds
            ),
            "selection_contract": (
                "independent_cross_sequence_or_persistent_evidence__"
                "continuous_existing_footprint_deficit_times_precision"
            ),
            "source": "current_initialization_surface_seed",
        },
        "surface_warmstart_ply_sha256": (
            surface_warmstart["surface_ply_sha256"]
            if surface_warmstart is not None
            else None
        ),
        "surface_warmstart_manifest_sha256": (
            surface_warmstart["handoff_manifest_sha256"]
            if surface_warmstart is not None
            else None
        ),
        "surface_learning_rate_continuation": {
            "absolute_iteration_offset": int(
                surface_learning_rate_offset
            ),
            "position_schedule": (
                "continue_rigid_absolute_iteration"
                if surface_learning_rate_offset > 0
                else "start_at_zero"
            ),
            "optimizer_moments": "reset",
            "optimizer_schedule_source": (
                surface_warmstart["surface_optimizer_source"]
                if surface_warmstart is not None
                else "current_training_arguments"
            ),
            "producer_schedule_inherited": bool(
                inherited_surface_optimizer is not None
            ),
            "non_position_learning_rates": (
                "inherit_rigid_producer"
                if inherited_surface_optimizer is not None
                else "current_training_arguments"
            ),
        },
        "mature_handoff_surface_policy": (
            args.mature_handoff_surface_policy
        ),
        "mature_handoff_surface_partition": {
            "rigid_prefix_rows": int(rigid_surface_residual_start),
            "evidence_completion_suffix_rows": int(
                len(surface.get_xyz) - rigid_surface_residual_start
            ),
            "chart_atlas_learning_rate": float(
                chart_atlas_learning_rate
            ),
            "contract": (
                "mature_rigid_prefix_fixed__chart_inverse_depth_low_lr__"
                "evidence_completion_suffix_geometry_opacity_trainable"
                if args.mature_handoff_surface_policy == "atlas_residual"
                else "policy_specific_legacy_surface_ownership"
            ),
        },
        "surface_ownership_contract": (
            (
                "mature_rigid_prefix_fixed__chart_atlas_low_lr__"
                "evidence_completion_suffix_trainable"
                if args.mature_handoff_surface_policy == "atlas_residual"
                else SURFACE_OWNERSHIP_REPAIR_TARGET[
                    "surface_ownership_contract"
                ]
            )
            if (
                args.mature_handoff_surface_policy == "appearance_only"
                or args.mature_handoff_surface_policy == "atlas_residual"
            )
            and args.surface_retirement_optical_mass_fraction_per_event > 0
            else "surface_opacity_follows_mature_handoff_policy"
        ),
        "chart_atlas_lifecycle": {
            "geometry_owner": "continuous_uv_atlas",
            "ordinary_opacity_is_retirement_authority": False,
            "allowed_retirement": [
                "uv_quadtree_replace_and_retire_with_receiver",
                "independent_geometric_contradiction",
            ],
            "export_cleanup_preserves_live_bound_cells": True,
        },
        "surface_screen_topology": dict(
            SURFACE_SCREEN_TOPOLOGY_CONTRACT
        ),
        "foliage_seed_sha256": _file_sha256(
            Path(initialization["foliage_seed"])
        ),
        # Resolution changes both the RGB objective and every screen-space
        # topology threshold.  It therefore belongs to the immutable resume
        # contract, not merely to runtime provenance.
        "render_resolution": {
            "width": int(views[0].image_width) if views else 0,
            "height": int(views[0].image_height) if views else 0,
        },
        "rgb_source": training_rgb_source,
        "dynamic_replacement_contract": (
            SURFACE_OWNERSHIP_REPAIR_TARGET[
                "dynamic_replacement_contract"
            ]
        ),
        "rgb_gradient_ownership_contract": (
            SURFACE_OWNERSHIP_REPAIR_TARGET[
                "rgb_gradient_ownership_contract"
            ]
        ),
        "semantic_ownership_contract": (
            SURFACE_OWNERSHIP_REPAIR_TARGET[
                "semantic_ownership_contract"
            ]
        ),
        "surface_retirement_optical_mass_fraction_per_event": float(
            args.surface_retirement_optical_mass_fraction_per_event
        ),
        "fixed_camera_validation": fixed_camera_validation,
        "foliage_rigid_calibration_alignment": (
            foliage_rigid_calibration_audit
        ),
        "camera_geometry_sha256": camera_runtime_contract[
            "camera_geometry_sha256"
        ],
        "camera_intrinsics_contract_sha256": _file_sha256(
            output / "camera_intrinsics_contract.json"
        ),
        "seed": int(args.seed),
        "sampling_schedule_sha256": schedule_hash,
        "conditioned_sampling": conditioned_schedule_audit,
        "dynamic_evidence_view_fraction": float(
            args.dynamic_evidence_view_fraction
        ),
        "dynamic_visibility_contract": (
            "exact_owner_plus_treewise_continuous_temporal_fallback_"
            "max035_decay"
        ),
        "foliage_sampling_policy": (
            "per_camera_instance_dephased_deterministic_blue_noise_"
            "coverage_plus_full_resolution_rgb_detail"
        ),
        "appearance_grid_policy": "disabled_no_camera_plane_grid",
        "static_spatial_uncertainty": {
            "contract": (
                "training_only_per_image_degree2_legendre_"
                "heteroscedastic_canopy_sky_field"
            ),
            "active_after_static_detail_visibility": True,
            "rgb_residual": False,
            "geometry_or_opacity_owner": False,
            "canonical_weight_uses_detached_sigma": True,
            "likelihood_uses_detached_render": True,
            "deployment_parameter": False,
            "camera_plane_grid": False,
        },
        # Kept only as a legacy/diagnostic knob. The static authoritative
        # forward and deployment compositor both use a disabled-policy native
        # mixed pass; persistent replacement is a parameter lifecycle below.
        "optical_replacement_policy": args.optical_replacement_policy,
        "training_optical_replacement_policy": (
            "disabled"
            if args.reconstruction_target == "static"
            else args.optical_replacement_policy
        ),
        "static_replacement_lifecycle": {
            "enabled": bool(
                args.reconstruction_target == "static"
                and args.static_replacement_evidence_every > 0
                and args.static_replacement_mass_fraction_per_event > 0
            ),
            "measurement": (
                "separate_disabled_policy_envelope_and_detail_role_renders"
            ),
            "commit": "persistent_reversible_parameter_space_mass_handoff",
            "authoritative_forward_policy": "disabled",
        },
        "chart_surface_policy": (
            "continuous_learnable_inverse_depth_atlas_uv_quadtree_"
            "live_native_2dgs_binding"
        ),
        "dynamic_policy": (
            "sequence_metadata_is_training_evidence_only__no_sequence_owned_"
            "primitive_no_temporal_code_one_static_map"
            if args.reconstruction_target == "static"
            else "legacy_sequence_conditioned_database_fit_branch"
        ),
        "deployment_static_contract": {
            "render_mode": "canonical",
            "include_dynamic_leaf_rows": False,
            "include_temporal_code": False,
            "include_camera_plane_spatial_appearance": False,
            "conditioned_render_role": (
                "absent"
                if args.reconstruction_target == "static"
                else "database_fit_training_diagnostic_only"
            ),
            "optical_replacement_policy": (
                deployment_optical_replacement_policy
            ),
            "optical_responsibility_prior": (
                deployment_optical_responsibility_prior
            ),
            "training_and_deployment_image_formation_identical": True,
            "persistent_handoff_is_parameter_lifecycle": True,
        },
        "dynamic_optical_mass_contract": (
            "exact_owner_support_unknown_free_space_initial_optical_mass_"
            "separate_from_occupancy_depth_geometry_floor__spatial_"
            "uncertainty_routes_geometry_ray_and_topology_authority__"
            "rgb_can_reduce_or_raise_optical_prior__role_wide_safety_"
            "ceiling_only__canonical_dynamic_handoff_measured_in_current_"
            "camera_projected_cross_section"
        ),
        "dynamic_gradient_ownership_contract": (
            "conditioned_base_gradients_exact_dynamic_owner_only__"
            "foliage_photo_gradients_canopy_pixels_only__"
            "rigid_sky_context_updates_appearance_only__"
            "canonical_group_pools_visible_dynamic_low_rank_state__"
            "skeleton_forward_occluders_read_only__"
            "temporal_fallback_updates_low_rank_residuals_only"
        ),
        "shared_canonical_conditioning_contract": (
            "canonical_base_state_plus_visibility_weighted_group_"
            "sequence_time_deformation_feature_opacity__"
            "local_dynamic_birth_death_residual_with_optical_mass_handoff"
        ),
        "dynamic_lifecycle_contract": (
            DYNAMIC_LIFECYCLE_REPAIR_TARGET
        ),
        "boundary_supervision_contract": BOUNDARY_SUPERVISION_CONTRACT,
        "volume_reallocation_contract": VOLUME_REALLOCATION_CONTRACT,
        "branch_activation": {
            "foliage": float(
                TRAINING_PROFILES[args.training_profile].get(
                    "foliage_start", 0.0
                )
            ),
            "dynamic": float(
                TRAINING_PROFILES[args.training_profile]["dynamic_start"]
            ),
            "foliage_iteration": _activation_iteration(
                float(
                    TRAINING_PROFILES[args.training_profile].get(
                        "foliage_start", 0.0
                    )
                ),
                args.phase_schedule_horizon,
            ),
            "dynamic_iteration": _activation_iteration(
                float(
                    TRAINING_PROFILES[args.training_profile][
                        "dynamic_start"
                    ]
                ),
                args.phase_schedule_horizon,
            ),
        },
        "dense_ray_scale_ceiling_contract": (
            "source_footprint_x1.10_inherited__exact_ray_split_shrinks_"
            "camera_plane_axes_and_preserves_depth_extent"
        ),
        "exact_ray_render_footprint_contract": (
            (
                "not_applicable__static_model_contains_no_dynamic_exact_"
                "ray_render_rows__actual_parameter_scale_is_shared_by_"
                "rasterization_topology_mass_audit_and_checkpoint"
            )
            if args.reconstruction_target == "static"
            else (
                "metric_depth_posterior_retained_in_position_covariance__"
                "visible_ewa_depth_axis_bounded_to_4x_middle_tangent_scale__"
                "tangent_colour_opacity_unchanged"
            )
        ),
        "exact_ray_topology_contract": (
            "owner_conditioned_gradient_plus_screen_bandwidth__depth_"
            "posterior_limits_3d_growth_but_not_depth_preserving_camera_"
            "plane_subdivision__owner_then_lineage_family_capacity"
        ),
        "surface_optimizer": {
            "position_lr_init": float(opt.position_lr_init),
            "position_lr_final": float(opt.position_lr_final),
            "position_lr_delay_mult": float(opt.position_lr_delay_mult),
            "position_lr_max_steps": int(opt.position_lr_max_steps),
            "non_position_lr_decay_from": int(
                opt.non_position_lr_decay_from
            ),
            "non_position_lr_decay_until": int(opt.iterations),
            "non_position_lr_final_mult": float(
                opt.non_position_lr_final_mult
            ),
            "feature_lr": float(opt.feature_lr),
            "opacity_lr": float(opt.opacity_lr),
            "scaling_lr": float(opt.scaling_lr),
            "rotation_lr": float(opt.rotation_lr),
        },
        "training_profile_optimizer_resolution": dict(
            args.training_profile_optimizer_resolution
        ),
        "lambda_dssim": float(opt.lambda_dssim),
        "geometry_every": int(args.geometry_every),
        "topology_every": int(args.topology_every),
        "densify_from_iter": int(args.densify_from_iter),
        "densify_until_iter": int(args.densify_until_iter),
        "densification_interval": int(args.densification_interval),
        "opacity_reset_interval": int(args.opacity_reset_interval),
        "opacity_cull": float(args.opacity_cull),
        "maximum_surface_scale": float(args.maximum_surface_scale),
        "maximum_surface_radius_pixels": float(
            args.maximum_surface_radius_pixels
        ),
        "maximum_volume_scale": float(args.maximum_volume_scale),
        "maximum_dynamic_volume_scale": float(
            args.maximum_dynamic_volume_scale
        ),
        "volume_split_radius": float(args.volume_split_radius),
        "volume_opacity_lr": float(args.volume_opacity_lr),
        "volume_appearance_polish": {
            "contract": (
                "static_geometry_mass_frozen__remaining_sh_appearance_sky_"
                "exponential_low_lr_polish"
            ),
            "final_lr_multiplier": float(
                args.volume_polish_final_lr_multiplier
            ),
        },
        "volume_opacity_settle": {
            "contract": VOLUME_OPACITY_SETTLE_CONTRACT,
            "policy": str(args.volume_opacity_settle_policy),
            "start_iteration": int(
                args.volume_opacity_settle_start_iteration
            ),
            "retirement_until_iteration": int(
                args.volume_opacity_retirement_until_iteration
            ),
            "trainable_after_settle": (
                "geometry_scale_rotation_sh_and_dynamic_deformation_feature"
            ),
        },
        "volume_topology_ramp_iterations": int(
            args.volume_topology_ramp_iterations
        ),
        "role_opacity_ceilings": {
            "static_skeleton": float(args.skeleton_opacity_ceiling),
            "canonical_crown": float(
                args.canonical_crown_opacity_ceiling
            ),
            "dynamic_leaf": float(args.dynamic_leaf_opacity_ceiling),
        },
        "bootstrap_evidence_release_fraction": float(
            args.bootstrap_evidence_release_fraction
        ),
        "bootstrap_evidence_scale_growth": float(
            args.bootstrap_evidence_scale_growth
        ),
        "bootstrap_evidence_opacity_ceiling": float(
            args.bootstrap_evidence_opacity_ceiling
        ),
        "geometry_weight": float(args.geometry_weight),
        "projected_rigid_geometry": {
            "weight": float(args.projected_rigid_depth_weight),
            "contract": PROJECTED_OCCLUDED_RIGID_GEOMETRY_CONTRACT,
            "rgb_authority": False,
        },
        "dav2_observation_patch": {
            "weight": float(args.dav2_observation_patch_weight),
            "camera_count": int(len(dav2_observation_patches)),
            "observation_count": int(
                sum(len(value) for value in dav2_observation_patches.values())
            ),
            "sampling": (
                "five_by_five_real_rgb_patch_at_persisted_source_uv"
            ),
            "topology_independent": True,
        },
        "pointmap_weight": float(args.pointmap_weight),
        "plane_weight": float(args.plane_weight),
        "normal_weight": float(args.normal_weight),
        "ordinal_weight": float(args.ordinal_weight),
        "track_weight": float(args.track_weight),
        "geometry_gradient_ratio": float(
            args.geometry_gradient_ratio
        ),
        "chart_anchor_weight": float(args.chart_anchor_weight),
        "chart_atlas": {
            "representation": (
                "bounded_coarse_fine_inverse_depth_residual"
            ),
            "configured_learning_rate": float(args.chart_atlas_lr),
            "effective_learning_rate": float(
                chart_atlas_learning_rate
            ),
            "learning_rate_lifecycle": (
                "producer_absolute_non_position_polish"
            ),
            "mature_handoff_learning_rate_scale": float(
                args.mature_chart_atlas_lr_scale
            ),
            "regularization_weight": float(
                args.chart_atlas_regularization_weight
            ),
            "quadtree_growth_fraction": float(
                args.chart_quadtree_growth_fraction
            ),
            "quadtree_maximum_level": int(
                args.chart_quadtree_maximum_level
            ),
            "renderer_binding": (
                "native_2d_surfel_geometry_same_mixed_cuda_sort"
            ),
            "final_export": "baked_standard_2dgs_geometry",
        },
        "structure_weight": float(args.structure_weight),
        "ownership_weight": float(args.ownership_weight),
        "conditioned_ownership_weight": float(
            args.conditioned_ownership_weight
        ),
        "counterfactual_transparency": {
            "contract": COUNTERFACTUAL_TRANSPARENCY_CONTRACT,
            "weight": float(args.counterfactual_transparency_weight),
            "margin": float(args.counterfactual_transparency_margin),
            "temperature": float(
                args.counterfactual_transparency_temperature
            ),
            "conditioned_every": int(
                args.conditioned_counterfactual_every
            ),
            "gradient_owner": "volume_opacity_only",
            "responsibility_gradient": "detached",
        },
        "occupancy_weight": float(args.occupancy_weight),
        "dynamic_weight": float(args.dynamic_weight),
        "appearance_weight": float(args.appearance_weight),
        "high_frequency_weight": float(args.high_frequency_weight),
        "rigid_residual_patch": {
            "weight": float(args.rigid_residual_patch_weight),
            "maximum_patches_per_view": 64,
            "patch_size": 11,
            "selection": (
                "detached_rigid_residual_times_target_edge_spatial_nms_pool"
            ),
            "targets": (
                "facade_window_stone_small_object_tree_boundary_rigid_side"
            ),
        },
        "maximum_surface_gaussians": int(surface_budget),
        "maximum_surface_growth_per_event": int(
            args.maximum_surface_growth_per_event
        ),
        "maximum_volume_gaussians": int(volume_budget),
        "maximum_volume_splits_per_event": int(
            args.maximum_volume_splits
        ),
        "volume_densify_every": int(args.volume_densify_every),
        "volume_densify_until_iteration": int(
            args.volume_densify_until_iteration
        ),
        "volume_topology_phase_contract": VOLUME_TOPOLOGY_SETTLE_CONTRACT,
        "conditioned_settle_contract": CONDITIONED_SETTLE_CONTRACT,
        "maximum_volume_radius_pixels": float(
            args.maximum_volume_radius_pixels
        ),
        "maximum_volume_radius_pixels_by_role": {
            "static_skeleton": float(
                args.maximum_skeleton_radius_pixels
            ),
            "persistent_envelope": float(
                args.maximum_envelope_radius_pixels
            ),
            "static_detail": float(
                args.maximum_static_detail_radius_pixels
            ),
        },
        "maximum_volume_family_rollbacks_per_event": int(
            args.maximum_volume_family_rollbacks_per_event
        ),
        "ray_posterior_every": int(args.ray_posterior_every),
        "ray_posterior_maximum_rays": int(
            effective_ray_posterior_maximum_rays
        ),
        "ray_posterior_configured_maximum_rays": int(
            args.ray_posterior_maximum_rays
        ),
        "ray_evidence_epoch_capacity": {
            "batch_boundary_contract": RAY_EPOCH_BOUNDARY_CONTRACT,
            "capacity_contract": RAY_EPOCH_CAPACITY_CONTRACT,
            "capacity_scope": (
                "clean_epoch_start"
                if prior_ray_factor_calls == 0
                else "resume_remaining_unvisited_rows"
            ),
            "scheduled_factor_calls": int(scheduled_ray_factor_calls),
            "prior_factor_calls": int(prior_ray_factor_calls),
            "available_factor_calls": int(capacity_ray_factor_calls),
            "camera_table_count": int(len(per_camera_ray_rows)),
            "full_effective_row_count": int(
                foliage_ray_evidence_audit[
                    "interval_effective_rows"
                ]
            ),
            "remaining_unvisited_row_count": int(
                remaining_unvisited_ray_rows
            ),
            "clean_epoch_aggregate_minimum_batch_with_five_percent_margin": int(
                clean_epoch_aggregate_minimum_batch
            ),
            "clean_epoch_minimum_batch_for_complete_per_camera_epoch": int(
                clean_epoch_minimum_batch
            ),
            "clean_epoch_effective_batch": int(clean_epoch_batch),
            "runtime_aggregate_minimum_batch_with_five_percent_margin": int(
                runtime_epoch_aggregate_minimum_batch
            ),
            "minimum_batch_for_runtime_completion": int(
                ray_epoch_minimum_batch
            ),
            "required_factor_calls": int(required_ray_factor_calls),
            "unused_factor_calls": int(
                capacity_ray_factor_calls - required_ray_factor_calls
            ),
            "complete_epoch_capacity": bool(
                required_ray_factor_calls <= capacity_ray_factor_calls
            ),
        },
        "ray_posterior_maximum_candidates": int(
            args.ray_posterior_maximum_candidates
        ),
    }
    implementation_hashes = {
        "trainer": _file_sha256(Path(__file__)),
        "static_foliage": _file_sha256(
            REPO_ROOT / "outdoor/static_foliage.py"
        ),
        "role_aware_initialization": _file_sha256(
            REPO_ROOT / "outdoor/role_aware_initialization.py"
        ),
        "foliage_geometry": _file_sha256(
            REPO_ROOT / "outdoor/foliage_geometry.py"
        ),
        "hybrid_teacher_api": _file_sha256(
            REPO_ROOT / "outdoor/hybrid_teacher_api.py"
        ),
        "appearance_uncertainty": _file_sha256(
            REPO_ROOT / "outdoor/appearance_uncertainty.py"
        ),
        "chart_surface_model": _file_sha256(
            REPO_ROOT / "outdoor/chart_surface_model.py"
        ),
        "lazy_scene": _file_sha256(
            REPO_ROOT / "outdoor/lazy_scene.py"
        ),
        "intrinsics_utils": _file_sha256(
            SURFEL_ROOT / "utils/intrinsics_utils.py"
        ),
        "dataset_reader": _file_sha256(
            SURFEL_ROOT / "scene/dataset_readers.py"
        ),
        "gaussian_model": _file_sha256(
            SURFEL_ROOT / "scene/gaussian_model.py"
        ),
        "task_fields": _file_sha256(
            REPO_ROOT / "outdoor/task_fields.py"
        ),
        "projected_role_posterior": _file_sha256(
            REPO_ROOT / "outdoor/projected_role_posterior.py"
        ),
        "mask_lookup": _file_sha256(
            REPO_ROOT / "matcha/cambridge_masks.py"
        ),
        "training_evidence": _file_sha256(
            REPO_ROOT / "outdoor/training_evidence.py"
        ),
        "static_ray_birth": _file_sha256(
            REPO_ROOT / "outdoor/static_ray_birth.py"
        ),
        "hybrid_renderer": _file_sha256(
            REPO_ROOT / "outdoor/hybrid_gaussian_renderer.py"
        ),
        "mixed_forward_cuda": _file_sha256(
            SURFEL_ROOT
            / "submodules/diff-surfel-rasterization/"
            "cuda_rasterizer/forward.cu"
        ),
        "mixed_backward_cuda": _file_sha256(
            SURFEL_ROOT
            / "submodules/diff-surfel-rasterization/"
            "cuda_rasterizer/mixed_backward.cu"
        ),
        "mixed_cuda_config": _file_sha256(
            SURFEL_ROOT
            / "submodules/diff-surfel-rasterization/"
            "cuda_rasterizer/config.h"
        ),
        "mixed_rasterizer_impl_cuda": _file_sha256(
            SURFEL_ROOT
            / "submodules/diff-surfel-rasterization/"
            "cuda_rasterizer/rasterizer_impl.cu"
        ),
    }
    runtime_provenance = collect_runtime_provenance(
        REPO_ROOT,
        python_modules=(
            "scripts.train_unified_outdoor_teacher",
            "outdoor.static_foliage",
            "outdoor.hybrid_gaussian_renderer",
            "outdoor.foliage_geometry",
            "outdoor.role_aware_initialization",
            "outdoor.projected_role_posterior",
            "outdoor.task_fields",
            "outdoor.appearance_uncertainty",
            "outdoor.chart_surface_model",
            "outdoor.hybrid_teacher_api",
            "scene.gaussian_model",
            "diff_surfel_rasterization",
        ),
        extension_roots=(
            SURFEL_ROOT / "submodules/diff-surfel-rasterization",
            SURFEL_ROOT / "submodules/simple-knn",
        ),
    )
    runtime_provenance["cpu_parallelism"] = cpu_parallelism_audit
    print(
        json.dumps(
            {
                "runtime_provenance": runtime_provenance,
                "implementation_hashes": implementation_hashes,
                "policies": {
                    key: training_contract[key]
                    for key in (
                        "foliage_sampling_policy",
                        "appearance_grid_policy",
                        "optical_replacement_policy",
                        "chart_surface_policy",
                        "dynamic_policy",
                    )
                },
            },
            indent=2,
        ),
        flush=True,
    )
    if resume is not None:
        resume_hashes = resume.get("implementation_hashes", {})
        if resume_hashes != implementation_hashes:
            changed = {
                key
                for key in set(resume_hashes) | set(implementation_hashes)
                if resume_hashes.get(key) != implementation_hashes.get(key)
            }
            performance_only = (
                args.allow_performance_resume
                and changed == set(PERFORMANCE_RESUME_PREDECESSOR)
                and all(
                    resume_hashes.get(key) == value
                    for key, value in PERFORMANCE_RESUME_PREDECESSOR.items()
                )
            )
            trainer_repair = _trainer_repair_hash_change_is_allowed(
                changed,
                enabled=bool(args.allow_trainer_repair_resume),
                allow_hybrid_renderer_topology_extension=bool(
                    args.allow_trainer_repair_resume
                    and PROTOCOL
                    == "cambridge_native_hybrid_teacher_v83_spatial_detail_receivers"
                ),
            )
            static_detail_isolated_repair = (
                args.allow_static_detail_isolated_repair_resume
                and changed == {"trainer", "hybrid_teacher_api"}
                and int(resume.get("iteration", -1))
                == STATIC_DETAIL_ISOLATED_REPAIR_PREDECESSOR["iteration"]
                and resume.get("protocol")
                == STATIC_DETAIL_ISOLATED_REPAIR_PREDECESSOR["protocol"]
                and all(
                    resume_hashes.get(key) == value
                    for key, value in (
                        STATIC_DETAIL_ISOLATED_REPAIR_PREDECESSOR.items()
                    )
                    if key not in {"protocol", "iteration"}
                )
            )
            static_canonical_ownership_repair = (
                args.allow_static_canonical_ownership_repair_resume
                and changed
                == {"trainer", "hybrid_teacher_api"}
                and int(resume.get("iteration", -1))
                == STATIC_CANONICAL_OWNERSHIP_REPAIR_PREDECESSOR[
                    "iteration"
                ]
                and resume.get("protocol")
                == STATIC_CANONICAL_OWNERSHIP_REPAIR_PREDECESSOR[
                    "protocol"
                ]
                and all(
                    resume_hashes.get(key) == value
                    for key, value in (
                        STATIC_CANONICAL_OWNERSHIP_REPAIR_PREDECESSOR.items()
                    )
                    if key not in {"protocol", "iteration"}
                )
            )
            volume_capacity_repair = (
                args.allow_volume_capacity_repair_resume
                and changed == {"trainer"}
                and resume_hashes.get("trainer")
                == VOLUME_CAPACITY_REPAIR_PREDECESSOR["trainer"]
                and int(resume.get("iteration", -1))
                == VOLUME_CAPACITY_REPAIR_PREDECESSOR["iteration"]
            )
            surface_ownership_repair = (
                args.allow_surface_ownership_repair_resume
                and changed == {"trainer"}
                and resume_hashes.get("trainer")
                == SURFACE_OWNERSHIP_REPAIR_PREDECESSOR["trainer"]
                and int(resume.get("iteration", -1))
                == SURFACE_OWNERSHIP_REPAIR_PREDECESSOR["iteration"]
            )
            volume_split_repair = (
                args.allow_volume_split_repair_resume
                and changed == {"trainer", "hybrid_renderer"}
                and all(
                    resume_hashes.get(key) == value
                    for key, value in VOLUME_SPLIT_REPAIR_PREDECESSOR.items()
                )
            )
            surface_split_repair = (
                args.allow_surface_split_repair_resume
                and changed == {"trainer", "gaussian_model"}
                and all(
                    resume_hashes.get(key) == value
                    for key, value in SURFACE_SPLIT_REPAIR_PREDECESSOR.items()
                )
            )
            surface_screen_evidence_repair = (
                args.allow_surface_screen_evidence_repair_resume
                and changed
                == {"trainer", "gaussian_model", "hybrid_teacher_api"}
                and int(resume.get("iteration", -1))
                == SURFACE_SCREEN_EVIDENCE_REPAIR_PREDECESSOR["iteration"]
                and resume.get("protocol")
                == SURFACE_SCREEN_EVIDENCE_REPAIR_PREDECESSOR["protocol"]
                and all(
                    resume_hashes.get(key) == value
                    for key, value in (
                        SURFACE_SCREEN_EVIDENCE_REPAIR_PREDECESSOR.items()
                    )
                    if key not in {"protocol", "iteration"}
                )
            )
            if (
                not performance_only
                and not trainer_repair
                and not static_detail_isolated_repair
                and not static_canonical_ownership_repair
                and not volume_capacity_repair
                and not surface_ownership_repair
                and not volume_split_repair
                and not surface_split_repair
                and not surface_screen_evidence_repair
            ):
                raise RuntimeError(
                    "Resume implementation/CUDA hash mismatch; start a new "
                    "unified run"
                )
            if v38_causal_repair_resume:
                print(
                    "Resuming the exact v35 9k checkpoint into v39: ray "
                    "epochs stop at camera-table boundaries with capacity "
                    "derived from the persisted sampler cursors, detached "
                    "surface counterfactuals route only volume transparency, "
                    "and canonical_polish freezes topology while conditioned "
                    "leaf optimization continues."
                )
            elif static_detail_isolated_repair:
                print(
                    "Resuming the exact v45 3k prefix into v46: the newly "
                    "active surface+detail counterfactual routes RGB, high-"
                    "frequency and means2D gradients only to static detail; "
                    "evidence, schedules, model/optimizer state and CUDA "
                    "kernels are identical."
                )
            elif static_canonical_ownership_repair:
                print(
                    "Resuming the exact v50 3k envelope-only prefix into "
                    "v51: static detail remains globally visible and "
                    "support-sequence-owned, while its isolated RGB/HF pass "
                    "now uniformly covers visible canonical support cameras."
                )
            elif surface_ownership_repair:
                print(
                    "Resuming the exact retained v83 10k checkpoint into "
                    "cross-sequence local surface retirement with a bounded "
                    "per-event optical-mass budget; surface opacity remains "
                    "frozen in Adam and rigid xyz/scale/rotation, evidence "
                    "and CUDA kernels are identical."
                )
            elif surface_split_repair:
                print(
                    "Resuming the exact audited predecessor across the "
                    "tangent-area/optical-mass preserving surface split "
                    "repair; evidence, schedules, optimizers and CUDA kernels "
                    "are identical."
                )
            elif surface_screen_evidence_repair:
                print(
                    "Resuming the exact v67 2k pre-topology prefix into "
                    "residual-conditioned screen topology: accumulated "
                    "loss gradient remains necessary evidence and projected "
                    "footprint only modulates its priority. Cameras, evidence, "
                    "model/optimizer state and CUDA kernels are identical."
                )
            elif volume_split_repair:
                print(
                    "Resuming the exact audited predecessor across the "
                    "mass-safe sqrt(2) volume split repair; evidence, "
                    "schedules, optimizers and CUDA kernels are identical."
                )
            elif volume_capacity_repair:
                print(
                    "Resuming the exact retained v82 8k checkpoint across "
                    "the audited role-conserving boundary/capacity repair; "
                    "camera schedules, evidence, model state and CUDA "
                    "kernels are identical."
                )
            elif trainer_repair:
                print(
                    "Resuming across an explicit trainer-only causal repair; "
                    "all evidence/model/CUDA hashes remain identical."
                )
            else:
                print(
                    "Resuming the exact pre-fast-path checkpoint with "
                    "I/O/allocator-only compatibility enabled."
                )
    if resume is not None and resume["schedule_hash"] != schedule_hash:
        saved_schedules = resume.get("schedules", {})
        if args.allow_trainer_repair_resume:
            unchanged_schedule_names = (
                "rgb",
                "conditioned",
                "geometry",
                "topology",
                "static_skeleton",
                "static_volume",
            )
        elif args.allow_conditioned_schedule_repair_resume:
            unchanged_schedule_names = (
                "rgb",
                "geometry",
                "topology",
                "static_detail",
                "static_skeleton",
                "static_volume",
            )
        else:
            raise RuntimeError("Resume camera schedules changed")
        current_schedules = {
            "rgb": rgb_schedule,
            "conditioned": conditioned_schedule,
            "geometry": geometry_schedule,
            "topology": topology_schedule,
            "static_detail": static_detail_schedule,
            "static_skeleton": static_skeleton_schedule,
            "static_volume": static_volume_schedule,
        }
        changed_non_conditioned = [
            name
            for name in unchanged_schedule_names
            if not np.array_equal(
                np.asarray(saved_schedules.get(name, []), dtype=np.int64),
                current_schedules[name],
            )
        ]
        if changed_non_conditioned:
            raise RuntimeError(
                "Resume schedule repair changed unrelated camera "
                f"schedules: {changed_non_conditioned}"
            )
        if args.allow_trainer_repair_resume:
            print(
                "Resuming with the support-owned static-detail camera "
                "schedule; every non-detail schedule is identical."
            )
        else:
            print(
                "Resuming with the coverage-preserving conditioned-camera "
                "schedule; all unrelated schedules are identical."
            )
    contract_differences = (
        _resume_training_contract_differences(
            resume.get("training_contract", {}),
            training_contract,
            allow_trainer_repair_migration=bool(
                args.allow_trainer_repair_resume
            ),
            allow_conditioned_schedule_repair_migration=bool(
                args.allow_conditioned_schedule_repair_resume
            ),
            allow_volume_capacity_repair_migration=bool(
                args.allow_volume_capacity_repair_resume
            ),
            allow_volume_topology_settle_migration=bool(
                args.allow_volume_topology_settle_resume
            ),
            allow_volume_opacity_settle_migration=bool(
                args.allow_volume_opacity_settle_resume
            ),
            allow_surface_ownership_repair_migration=bool(
                args.allow_surface_ownership_repair_resume
            ),
            allow_v38_causal_repair_migration=bool(
                v38_causal_repair_resume
            ),
            allow_static_detail_isolated_repair_migration=bool(
                args.allow_static_detail_isolated_repair_resume
            ),
            allow_static_canonical_ownership_repair_migration=bool(
                args.allow_static_canonical_ownership_repair_resume
            ),
            allow_surface_screen_evidence_repair_migration=bool(
                args.allow_surface_screen_evidence_repair_resume
            ),
        )
        if resume is not None
        else set()
    )
    if contract_differences:
        raise RuntimeError(
            "Resume optimization/evidence-ownership contract changed "
            f"({', '.join(sorted(contract_differences))}); start a clean run"
        )
    # Contract comparison above intentionally uses the clean immutable
    # evidence preflight. Runtime counters are cumulative state and are
    # restored only after that comparison.
    _restore_geometry_audit_counters_from_checkpoint(geometry, resume)
    if (
        resume is not None
        and resume.get("training_contract", {}).get(
            "camera_intrinsics_contract_sha256"
        )
        != training_contract.get("camera_intrinsics_contract_sha256")
    ):
        print(
            "Resuming with identical K/pose geometry and a different "
            "performance-only camera-container cache setting."
        )
    if resume is not None:
        surface_ownership_extension_resume = bool(
            args.allow_surface_ownership_repair_resume
            and int(resume.get("iteration", -1))
            == SURFACE_OWNERSHIP_REPAIR_PREDECESSOR["iteration"]
            and resume.get("training_contract", {}).get(
                "mature_handoff_surface_policy"
            )
            == SURFACE_OWNERSHIP_REPAIR_PREDECESSOR[
                "mature_handoff_surface_policy"
            ]
            and args.mature_handoff_surface_policy
            == SURFACE_OWNERSHIP_REPAIR_TARGET[
                "mature_handoff_surface_policy"
            ]
        )
        if (
            args.allow_surface_ownership_repair_resume
            and not surface_ownership_extension_resume
        ):
            raise RuntimeError(
                "Surface ownership repair resume does not match the exact "
                "v83-10k event-bounded local-retirement migration"
            )
        capacity_extension_resume = bool(
            args.allow_volume_capacity_repair_resume
            and int(resume.get("iteration", -1))
            == VOLUME_CAPACITY_REPAIR_PREDECESSOR["iteration"]
            and int(
                resume.get("maximum_volume_gaussians", -1)
            )
            == VOLUME_CAPACITY_REPAIR_PREDECESSOR[
                "maximum_volume_gaussians"
            ]
            and int(
                resume.get(
                    "maximum_volume_splits_per_event", -1
                )
            )
            == VOLUME_CAPACITY_REPAIR_PREDECESSOR[
                "maximum_volume_splits_per_event"
            ]
            and int(volume_budget)
            == VOLUME_CAPACITY_REPAIR_TARGET[
                "maximum_volume_gaussians"
            ]
            and int(args.maximum_volume_splits)
            == VOLUME_CAPACITY_REPAIR_TARGET[
                "maximum_volume_splits_per_event"
            ]
            and int(args.volume_densify_until_iteration)
            == VOLUME_CAPACITY_REPAIR_TARGET[
                "volume_densify_until_iteration"
            ]
        )
        if (
            args.allow_volume_capacity_repair_resume
            and not capacity_extension_resume
        ):
            raise RuntimeError(
                "Volume capacity repair resume does not match the exact "
                "v82-8k -> v83-2M/20k/10k audited migration"
            )
        resume_profile = resume.get("training_profile", "quality")
        if resume_profile != args.training_profile:
            raise RuntimeError(
                "Resume training profile changed: "
                f"{resume_profile} != {args.training_profile}"
            )
        resume_budget = resume.get("maximum_surface_gaussians")
        if (
            resume_budget is not None
            and int(resume_budget) != surface_budget
        ):
            raise RuntimeError(
                "Resume surface budget changed: "
                f"{resume_budget} != "
                f"{surface_budget}"
            )
        resume_growth = resume.get(
            "maximum_surface_growth_per_event"
        )
        if (
            resume_growth is not None
            and int(resume_growth)
            != args.maximum_surface_growth_per_event
        ):
            raise RuntimeError(
                "Resume per-event surface budget changed: "
                f"{resume_growth} != "
                f"{args.maximum_surface_growth_per_event}"
            )
        resume_volume_budget = resume.get("maximum_volume_gaussians")
        if (
            resume_volume_budget is not None
            and int(resume_volume_budget) != volume_budget
            and not capacity_extension_resume
        ):
            raise RuntimeError(
                "Resume volume budget changed: "
                f"{resume_volume_budget} != {volume_budget}"
            )
        resume_volume_growth = resume.get(
            "maximum_volume_splits_per_event"
        )
        if (
            resume_volume_growth is not None
            and int(resume_volume_growth)
            != args.maximum_volume_splits
            and not capacity_extension_resume
        ):
            raise RuntimeError(
                "Resume per-event volume budget changed: "
                f"{resume_volume_growth} != "
                f"{args.maximum_volume_splits}"
            )

    background = torch.ones(3, device="cuda")
    anchor_release_iteration = (
        int(
            np.floor(
                args.phase_schedule_horizon
                * float(args.bootstrap_evidence_release_fraction)
            )
        )
        + 1
    )
    if resume is None:
        start_step = 0
        conditioned_visit_counts = np.zeros(
            len(views), dtype=np.int64
        )
        conditioned_visit_provenance = {
            "source": "new_runtime_ledger",
            "carried_conditioned_steps": 0,
        }
        volume_stats = _volume_stats(foliage)
        replacement_stats = None
        topology_events = []
        replacement_events = []
        anchors_released = False
        anchor_release_events = []
        geometry_topology_factor_calls = 0
        geometry_topology_primitive_updates = 0
        bootstrap_scale_limit = torch.minimum(
            surface.get_scaling.detach()
            * float(args.bootstrap_evidence_scale_growth),
            surface.get_scaling.new_full(
                surface.get_scaling.shape,
                float(args.maximum_surface_scale),
            ),
        )
    else:
        start_step = int(resume["iteration"])
        (
            conditioned_visit_counts,
            conditioned_visit_provenance,
        ) = _resume_conditioned_visit_counts(resume, len(views))
        volume_stats = {
            name: value.cuda()
            for name, value in resume["volume_stats"].items()
        }
        for name in (
            "observation_gradient",
            "observation_count",
            "conditioned_gradient",
            "conditioned_gradient_count",
            "conditioned_radius",
            "conditioned_context_score",
        ):
            volume_stats.setdefault(
                name, torch.zeros(len(foliage), device="cuda")
            )
        volume_stats.setdefault(
            "conditioned_context_id",
            torch.full(
                (len(foliage),),
                -1,
                dtype=torch.int32,
                device="cuda",
            ),
        )
        replacement_stats = (
            None
            if resume["replacement_stats"] is None
            else {
                name: value.cuda()
                for name, value in resume["replacement_stats"].items()
            }
        )
        if replacement_stats is not None:
            replacement_stats.setdefault(
                "retirement_fraction",
                torch.zeros(
                    len(surface.get_xyz),
                    device=surface.get_xyz.device,
                ),
            )
            replacement_stats.setdefault(
                "semantic_evidence",
                torch.zeros(
                    len(surface.get_xyz),
                    3,
                    device=surface.get_xyz.device,
                ),
            )
        topology_events = list(resume["topology_events"])
        replacement_events = list(resume["replacement_events"])
        anchors_released = bool(
            resume.get(
                "anchors_released",
                not bool(surface._protected_flag.any()),
            )
        )
        anchor_release_events = list(
            resume.get("anchor_release_events", [])
        )
        geometry_topology_factor_calls = int(
            resume.get("geometry_topology_factor_calls", 0)
        )
        geometry_topology_primitive_updates = int(
            resume.get("geometry_topology_primitive_updates", 0)
        )
        saved_bootstrap_limit = resume.get("bootstrap_scale_limit")
        bootstrap_scale_limit = (
            saved_bootstrap_limit.cuda()
            if saved_bootstrap_limit is not None
            else torch.minimum(
                surface.get_scaling.detach()
                * float(args.bootstrap_evidence_scale_growth),
                surface.get_scaling.new_full(
                    surface.get_scaling.shape,
                    float(args.maximum_surface_scale),
                ),
            )
        )
        random.setstate(resume["python_rng_state"])
        np.random.set_state(resume["numpy_rng_state"])
        torch.set_rng_state(resume["torch_rng_state"])
        torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
    sequence_names = sorted(
        {sequence_id(view.image_name) for view in views}
    )
    if len(sequence_names) > 62:
        raise RuntimeError("Replacement sequence bitset exceeds 62")
    sequence_bits = {
        name: 1 << index for index, name in enumerate(sequence_names)
    }

    trace = output / "training_trace.jsonl"
    started = time.time()
    graceful_stop: dict[str, int | str | float] = {}

    def request_graceful_stop(signum, _frame) -> None:
        # A Python signal handler must remain side-effect free with respect to
        # CUDA and torch serialization. The training loop observes this flag
        # only after the current optimizer/topology transaction is complete,
        # then writes the same atomic checkpoint used by normal milestones.
        if graceful_stop:
            return
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(int(signum))
        graceful_stop.update(
            {
                "signal": name,
                "signal_number": int(signum),
                "requested_at_unix": float(time.time()),
            }
        )

    for stop_signal in (
        signal.SIGHUP,
        signal.SIGINT,
        signal.SIGTERM,
    ):
        signal.signal(stop_signal, request_graceful_stop)
    geometry_audit_warnings: set[str] = set()
    # These audits are normally overwritten after every optimizer step.  A
    # checkpoint resumed exactly at its requested stop iteration executes no
    # loop body but must still be able to re-run export/evaluation after a
    # renderer-only repair.
    surface_spatial_confidence_audit = (
        _apply_surface_spatial_confidence_gradients(surface)
    )
    volume_opacity_settle_audit = {
        "policy": "not_executed_no_optimization_steps",
        "resume_iteration": int(start_step),
        "requested_stop_iteration": int(args.iterations),
    }
    static_replacement_audit = {
        "contract": "persistent_distinct_view_real_ray_replacement_ema",
        "scheduled": False,
    }
    static_mass_handoff_audit = {
        "contract": "reversible_integrated_optical_mass_handoff",
        "changed_rows": 0,
    }
    static_child_verification_audit = {
        "contract": "real_ray_child_reverification",
        "candidate_rows": 0,
    }
    progress = tqdm(
        range(start_step, args.iterations),
        initial=start_step,
        total=args.iterations,
        desc="unified outdoor teacher",
        # A dynamic progress bar emits one write per iteration even when stderr
        # is an NFS-backed log file.  That made the GPU wait on remote metadata
        # for most of v96c. Structured log rows below remain authoritative.
        disable=not sys.stderr.isatty(),
    )

    def prefetch_training_images(begin: int) -> None:
        if args.image_prefetch_depth <= 0:
            return
        requested = []
        end = min(args.iterations, begin + args.image_prefetch_depth)
        for future_step in range(max(begin, start_step), end):
            requested.append(
                views[int(rgb_schedule[future_step])]
            )
            if _conditioned_branch_active(
                future_step,
                args.phase_schedule_horizon,
                args.training_profile,
                args.reconstruction_target,
            ):
                requested.append(
                    views[int(conditioned_schedule[future_step])]
                )
            if (
                args.reconstruction_target == "static"
                and args.static_detail_isolated_every > 0
                and args.static_detail_isolated_weight > 0.0
                and (future_step + 1)
                % args.static_detail_isolated_every
                == 0
                and _static_detail_stage_visible(
                    _phase(
                        future_step,
                        args.phase_schedule_horizon,
                        args.training_profile,
                    )
                )
            ):
                requested.append(
                    views[
                        int(
                            static_detail_schedule[
                                _periodic_schedule_index(
                                    future_step,
                                    args.static_detail_isolated_every,
                                )
                            ]
                        )
                    ]
                )
            if (
                args.reconstruction_target == "static"
                and args.static_replacement_evidence_every > 0
                and args.static_replacement_mass_fraction_per_event > 0
                and (future_step + 1)
                % args.static_replacement_evidence_every
                == 0
                and _static_detail_stage_visible(
                    _phase(
                        future_step,
                        args.phase_schedule_horizon,
                        args.training_profile,
                    )
                )
            ):
                requested.append(
                    views[
                        int(
                            static_detail_schedule[
                                _periodic_schedule_index(
                                    future_step,
                                    args.static_replacement_evidence_every,
                                )
                            ]
                        )
                    ]
                )
            if (
                args.reconstruction_target == "static"
                and args.static_skeleton_isolated_every > 0
                and args.static_skeleton_isolated_weight > 0.0
                and static_skeleton_view_indices
                and (future_step + 1)
                % args.static_skeleton_isolated_every
                == 0
            ):
                requested.append(
                    views[
                        int(
                            static_skeleton_schedule[
                                _periodic_schedule_index(
                                    future_step,
                                    args.static_skeleton_isolated_every,
                                )
                            ]
                        )
                    ]
                )
            if (
                args.reconstruction_target == "static"
                and args.static_volume_isolated_every > 0
                and args.static_volume_isolated_weight > 0.0
                and static_volume_view_indices
                and (future_step + 1)
                % args.static_volume_isolated_every
                == 0
            ):
                requested.append(
                    views[
                        int(
                            static_volume_schedule[
                                _periodic_schedule_index(
                                    future_step,
                                    args.static_volume_isolated_every,
                                )
                            ]
                        )
                    ]
                )
            if (
                (
                    _surface_topology_active(future_step, args)
                    or _chart_topology_active(future_step, args)
                )
                and (future_step + 1) >= anchor_release_iteration
                and args.topology_every > 0
                and (future_step + 1) % args.topology_every == 0
            ):
                requested.append(
                    views[int(topology_schedule[future_step])]
                )
        scene.prefetch_images(requested)

    prefetch_training_images(start_step)
    for step in progress:
        if (
            not anchors_released
            and (step + 1) >= anchor_release_iteration
        ):
            with torch.no_grad():
                released = int(surface._protected_flag.sum())
                surface._protected_flag.zero_()
            anchors_released = True
            anchor_release_events.append(
                {
                    "iteration": step + 1,
                    "released_renderer_witnesses": released,
                    "permanent_track_evidence_nodes": int(
                        sum(
                            len(archive["track_id"])
                            for archive in geometry.track_archives.values()
                        )
                    ),
                    "permanent_chart_evidence_nodes": int(
                        chart_surface.audit()["anchor_count"]
                    ),
                }
            )
        phase = _phase(
            step, args.phase_schedule_horizon, args.training_profile
        )
        volume_learning_rate_audit = _update_volume_learning_rates(
            volume_optimizer, step, args, phase
        )
        # Static detail is persistent, but it is intentionally introduced
        # only after the broad crown envelope has learned depth, volume and
        # transmittance.  This is a training-stage ownership rule, not a
        # sequence visibility gate: the final model contains one static map.
        static_detail_active = (
            args.reconstruction_target != "static"
            or _static_detail_stage_visible(phase)
        )
        foliage_active = _foliage_enabled(
            step, args.phase_schedule_horizon, args.training_profile
        )
        dynamic_active = _conditioned_branch_active(
            step,
            args.phase_schedule_horizon,
            args.training_profile,
            args.reconstruction_target,
        )
        static_spatial_uncertainty_active = (
            _static_spatial_uncertainty_active(
                args.reconstruction_target,
                foliage_active=foliage_active,
                static_detail_active=static_detail_active,
            )
        )
        # A 24k rigid handoff is a mature reconstruction, not a fresh seed.
        # Restarting its exponential xyz schedule at iteration one raised the
        # position LR by roughly 40x and destroyed the scaffold within the
        # first mixed-training kilostep. Continue at the handoff's absolute
        # training iteration while keeping the independently initialized
        # volume/appearance schedules local to this run.
        surface_lr_iteration = _continued_surface_iteration(
            step + 1,
            surface_learning_rate_offset,
        )
        surface_xyz_learning_rate = surface.update_learning_rate(
            surface_lr_iteration
        )
        chart_atlas_lr_audit = _update_chart_atlas_learning_rate(
            chart_atlas_optimizer, surface_lr_iteration, opt
        )
        if (step + 1) % 1000 == 0:
            surface.oneupSHdegree()
        surface.optimizer.zero_grad(set_to_none=True)
        volume_optimizer.zero_grad(set_to_none=True)
        chart_atlas_optimizer.zero_grad(set_to_none=True)
        view = views[int(rgb_schedule[step])]
        task = fields.fields(
            view.image_name,
            (view.image_height, view.image_width),
            torch.device("cuda"),
        )
        target = view.original_image.cuda(non_blocking=True)
        canopy_topology_signal = _canopy_topology_signal(
            target, task
        )
        # Decode upcoming random-schedule images while CUDA executes the
        # current forward/backward passes.
        prefetch_training_images(step + 1)
        static_training_gate = None
        if (
            args.reconstruction_target == "static"
            and not static_detail_active
        ):
            static_training_gate = torch.ones_like(foliage.opacities)
            static_training_gate[foliage.static_leaf_mask] = 0.0
        static_detail_gradient_gate = None
        static_detail_positive_evidence_sequence_gate = None
        static_detail_appearance_gradient_gate = None
        static_detail_optical_gradient_gate = None
        static_detail_refinement_gradient_gate = None
        static_detail_ownership_audit = {
            "enabled": False,
            "camera_id": int(view.colmap_id),
            "owned_detail_rows": 0,
            "total_detail_rows": int(foliage.static_leaf_mask.sum()),
            "forward_visibility": "unconditional_static",
            "owned_positive_signals": (
                "geometry_opacity_sh_hit_topology"
            ),
            "appearance_signals": (
                "exact_support_plus_soft_same_acquisition_sequence"
            ),
            "soft_same_sequence_appearance_rows": 0,
            "soft_same_sequence_optical_rows": 0,
            "same_sequence_appearance_weight": float(
                args.static_detail_same_sequence_appearance_weight
            ),
            "same_sequence_optical_weight": float(
                args.static_detail_same_sequence_optical_weight
            ),
            "refinement_rows": 0,
            "same_sequence_non_support_rows": 0,
            "global_negative_signals": "free_rigid_counterfactual_cleanup",
        }
        if (
            args.reconstruction_target == "static"
            and static_detail_active
            and args.static_detail_canonical_ownership
        ):
            static_detail_gradient_gate = (
                _static_detail_canonical_ownership_gate(
                    foliage,
                    int(view.colmap_id),
                    camera_sequence_lookup,
                )
            )
            static_detail_positive_evidence_sequence_gate = (
                _static_detail_positive_evidence_sequence_gate(
                    foliage,
                    int(view.colmap_id),
                    camera_sequence_lookup,
                )
            )
            static_detail_appearance_gradient_gate = (
                _static_detail_same_sequence_appearance_gate(
                    foliage,
                    int(view.colmap_id),
                    camera_sequence_lookup,
                    static_detail_gradient_gate,
                    fallback_weight=(
                        args.static_detail_same_sequence_appearance_weight
                    ),
                    positive_evidence_sequence_gate=(
                        static_detail_positive_evidence_sequence_gate
                    ),
                )
            )
            static_detail_optical_gradient_gate = (
                _static_detail_same_sequence_optical_gate(
                    foliage,
                    int(view.colmap_id),
                    camera_sequence_lookup,
                    static_detail_gradient_gate,
                    fallback_weight=(
                        args.static_detail_same_sequence_optical_weight
                    ),
                    positive_evidence_sequence_gate=(
                        static_detail_positive_evidence_sequence_gate
                    ),
                )
            )
            owned_detail = (
                (static_detail_gradient_gate > 0)
                & foliage.static_leaf_mask
            )
            static_detail_refinement_gradient_gate = (
                _static_detail_refinement_gate(
                    foliage, static_detail_gradient_gate
                )
            )
            refinement_detail = (
                (static_detail_refinement_gradient_gate > 0)
                & foliage.static_leaf_mask
            )
            same_sequence_non_support_rows = 0
            if step == 0 or (step + 1) % args.log_every == 0:
                sequence_owned = _static_detail_support_sequence_audit_gate(
                    foliage,
                    int(view.colmap_id),
                    camera_sequence_lookup,
                )
                same_sequence_non_support_rows = int(
                    (sequence_owned & ~owned_detail).sum()
                )
            static_detail_ownership_audit = {
                "enabled": True,
                "camera_id": int(view.colmap_id),
                "owned_detail_rows": int(owned_detail.sum()),
                "refinement_rows": int(refinement_detail.sum()),
                "same_sequence_non_support_rows": (
                    same_sequence_non_support_rows
                ),
                "total_detail_rows": int(foliage.static_leaf_mask.sum()),
                "forward_visibility": "unconditional_static",
                "owned_positive_signals": (
                    "geometry_opacity_sh_hit_topology"
                ),
                "appearance_signals": (
                    "exact_support_plus_soft_same_acquisition_sequence"
                ),
                "soft_same_sequence_appearance_rows": int(
                    (
                        foliage.static_leaf_mask
                        & (static_detail_appearance_gradient_gate > 0)
                        & ~owned_detail
                    ).sum()
                ),
                "soft_same_sequence_optical_rows": int(
                    (
                        foliage.static_leaf_mask
                        & (static_detail_optical_gradient_gate > 0)
                        & ~owned_detail
                    ).sum()
                ),
                "same_sequence_appearance_weight": float(
                    args.static_detail_same_sequence_appearance_weight
                ),
                "same_sequence_optical_weight": float(
                    args.static_detail_same_sequence_optical_weight
                ),
                "global_negative_signals": (
                    "free_rigid_counterfactual_cleanup"
                ),
            }
        canonical_volume_geometry_gate = static_detail_gradient_gate
        canonical_volume_appearance_gate = static_detail_gradient_gate
        canonical_volume_opacity_gate = static_detail_gradient_gate
        if args.reconstruction_target == "static":
            (
                canonical_volume_geometry_gate,
                canonical_volume_appearance_gate,
                canonical_volume_opacity_gate,
            ) = _static_stage_rgb_gradient_gates(
                foliage,
                static_detail_gradient_gate,
                detail_stage_active=static_detail_active,
                appearance_gate=static_detail_appearance_gradient_gate,
            )
            if (
                static_detail_active
                and static_detail_optical_gradient_gate is not None
            ):
                # Stage 3 freezes broad envelope RGB-opacity growth, but
                # positive-evidence sequence cameras must be able to restore
                # local detail coverage. Geometry/topology remain governed by
                # the stricter verified exact-owner gate below.
                detail = foliage.static_leaf_mask
                canonical_volume_opacity_gate[detail] = (
                    static_detail_optical_gradient_gate[detail]
                )
            if (
                static_detail_active
                and static_detail_refinement_gradient_gate is not None
            ):
                canonical_volume_geometry_gate = (
                    canonical_volume_geometry_gate.clone()
                )
                canonical_volume_geometry_gate[
                    foliage.static_leaf_mask
                ] = static_detail_refinement_gradient_gate[
                    foliage.static_leaf_mask
                ]
        static_replacement_lifecycle_enabled = bool(
            args.reconstruction_target == "static"
            and foliage_active
            and static_detail_active
            and args.static_replacement_evidence_every > 0
            and args.static_replacement_mass_fraction_per_event > 0
        )
        static_replacement_candidate_scheduled = bool(
            static_replacement_lifecycle_enabled
            and (step + 1) % args.static_replacement_evidence_every == 0
        )
        package = render_hybrid(
            view,
            surface,
            foliage,
            background=background,
            include_dynamic=False,
            volume_gate=static_training_gate,
            volume_geometry_gradient_gate=(
                canonical_volume_geometry_gate
            ),
            # Positive geometry, optical mass and SH are calibrated-support
            # owned. All rows remain in the forward render, and independent
            # global negative evidence remains able to remove contradictions.
            volume_appearance_gradient_gate=(
                canonical_volume_appearance_gate
            ),
            volume_opacity_gradient_gate=canonical_volume_opacity_gate,
            volume_opacity_scale=1.0 if foliage_active else 0.0,
            structural_trainable_start=0,
            # The authoritative train/deployment image is always the same
            # one-pass mixed render. Local takeover is measured separately by
            # two disabled-policy role renders and then committed as a
            # persistent, reversible parameter-space mass handoff.
            optical_replacement_policy="disabled",
            audit_fields=torch.cat(
                [
                    torch.stack(
                        [
                            task["p_canopy_core"],
                            task["p_rigid"],
                            canopy_topology_signal,
                        ]
                    ),
                    target.detach(),
                ],
                dim=0,
            ),
        )
        # The mixed image is the authoritative image-formation model, but it
        # is not a valid source of rigid RGB gradients for the structural
        # branch: a wrong translucent crown in front of a facade attenuates
        # the surface gradient and encourages the 2D surfel to compensate for
        # the volume's colour/alpha error. Render the exact same structural
        # branch without volume solely to route rigid/sky RGB gradients to
        # their real owners. Canopy RGB below still uses the jointly sorted
        # mixed render and is differentiated only with respect to foliage.
        if foliage_active:
            structural_package = render_hybrid(
                view,
                surface,
                foliage,
                background=background,
                include_dynamic=False,
                volume_opacity_scale=0.0,
                structural_trainable_start=0,
                audit_fields=torch.stack(
                    [
                        task["p_canopy_core"],
                        task["p_rigid"],
                        canopy_topology_signal,
                    ]
                ),
            )
        else:
            # With exact-zero compacted volume the canonical package is
            # already the native surface-only render.  The former duplicate
            # call rasterized the same 2DGS population twice throughout the
            # entire rigid pretrain, roughly halving throughput without
            # changing either the image or any gradient.
            structural_package = package
        canonical = composite_white_background(
            package.render, package.alpha, sky(view)
        )
        structural_prediction = composite_white_background(
            structural_package.render,
            structural_package.alpha,
            sky(view),
        )
        owner_weight = (
            task["w_gaussian_rgb"] + task["w_sky_rgb"]
        ).clamp(0, 1.5)
        # The canonical crown owns stable low-frequency colour and average
        # transmittance only. Dynamic high-frequency observations are never
        # allowed to dominate it, including before the conditioned branch
        # starts.
        # Before the volume branch is active, canopy has no valid renderer
        # owner.  Giving those pixels to the surface/sky during bootstrap
        # creates exactly the low-frequency foreground paint that ownership
        # cleanup later struggles to retire.
        canonical_canopy_weight = 0.50 if foliage_active else 0.0
        background_weight = owner_weight * (
            task["p_rigid"] + task["p_sky"]
        ).clamp(0, 1)
        canopy_weight = (
            owner_weight
            * canonical_canopy_weight
            * task["p_canopy"]
        )
        static_confidence_mean = owner_weight.new_tensor(1.0)
        static_sigma_mean = owner_weight.new_zeros(())
        canonical_sigma = None
        if dynamic_active or static_spatial_uncertainty_active:
            # Pixels that repeatedly disagree across calibrated traversals no
            # longer drag the canonical crown into a broad average. Sigma is
            # detached here: uncertainty is trained by its proper likelihood
            # below and cannot reduce this loss by simply inflating itself.
            # Static reconstruction uses the same training-only confidence
            # without activating temporal leaves or conditioned rendering.
            canonical_sigma = appearance.spatial_uncertainty(
                view.image_name,
                (view.image_height, view.image_width),
            )
            sigma = canonical_sigma.detach()
            initial_sigma = sigma.new_tensor(
                math.exp(-4.5)
                + (math.exp(-1.5) - math.exp(-4.5))
                / (1.0 + math.exp(3.0))
            )
            # Confidence responds continuously to uncertainty above its
            # calibrated initialization instead of remaining exactly one up
            # to the former sigma=0.05 threshold. The asymptotic floor keeps
            # every real RGB observation informative without a binary gate.
            canopy_confidence = 0.35 + 0.65 * torch.minimum(
                initial_sigma / sigma[0].clamp_min(1.0e-6),
                torch.ones_like(sigma[0]),
            )
            sky_confidence = 0.25 + 0.75 * torch.minimum(
                initial_sigma / sigma[1].clamp_min(1.0e-6),
                torch.ones_like(sigma[1]),
            )
            background_weight = background_weight * (
                task["p_rigid"] + task["p_sky"] * sky_confidence
            ).clamp(0, 1)
            canopy_weight = canopy_weight * canopy_confidence
            static_confidence_mean = (
                canopy_confidence * task["p_canopy"]
            ).sum() / task["p_canopy"].sum().clamp_min(1)
            static_sigma_mean = (
                sigma[0] * task["p_canopy"]
            ).sum() / task["p_canopy"].sum().clamp_min(1)
        isolated_rgb = _branch_isolated_rgb_losses(
            structural_prediction,
            canonical,
            target,
            background_weight=background_weight,
            canopy_weight=canopy_weight,
            rigid_weight=task["p_rigid"],
            rigid_high_frequency_weight=(
                task["p_rigid"]
                * _boundary_evidence_weight(task)
                * (1.0 - task["p_transient"])
            ),
            lambda_dssim=opt.lambda_dssim,
        )
        photo = isolated_rgb["background"]
        canopy_photo = (
            isolated_rgb["canopy"]
            if foliage_active
            else canonical.new_zeros(())
        )
        static_canopy_high_frequency = (
            # High-frequency foliage supervision is role-isolated below.
            # Applying it to the authoritative all-foliage render lets broad
            # envelope and unverified one-view candidates reproduce arbitrary
            # image edges by moving geometry.  Detail/skeleton/volume isolated
            # streams retain the same real RGB target with explicit owners.
            canonical.new_zeros(())
        )
        static_detail_isolated_package = None
        static_detail_isolated_ownership_gate = None
        static_detail_isolated_refinement_gate = None
        static_detail_isolated_qualified_rows = 0
        static_detail_isolated_appearance_rows = 0
        static_detail_isolated_view = None
        static_detail_isolated_photo = canonical.new_zeros(())
        static_detail_isolated_high_frequency = canonical.new_zeros(())
        static_detail_isolated_scheduled = bool(
            args.reconstruction_target == "static"
            and foliage_active
            and static_detail_active
            and args.static_detail_isolated_every > 0
            and args.static_detail_isolated_weight > 0.0
            and (step + 1) % args.static_detail_isolated_every == 0
            and bool(foliage.static_leaf_mask.any())
        )
        if static_detail_isolated_scheduled:
            # The persistent envelope is deliberately absent in this
            # counterfactual. In the authoritative full render it can already
            # explain a low-frequency tree pixel and therefore attenuate the
            # gradient of a detail primitive behind it. That creates a
            # self-locking ownership failure: detail cannot gain optical mass
            # until it replaces the envelope, yet replacement authority is
            # itself measured from detail mass. Surface+detail exposes the
            # real RGB/occlusion residual while preserving the exact same
            # camera, native mixed rasterizer and depth sort.
            static_detail_isolated_view = views[
                int(
                    static_detail_schedule[
                        _periodic_schedule_index(
                            step, args.static_detail_isolated_every
                        )
                    ]
                )
            ]
            static_detail_isolated_task = fields.fields(
                static_detail_isolated_view.image_name,
                (
                    static_detail_isolated_view.image_height,
                    static_detail_isolated_view.image_width,
                ),
                torch.device("cuda"),
            )
            static_detail_isolated_target = (
                static_detail_isolated_view.original_image.cuda(
                    non_blocking=True
                )
            )
            static_detail_isolated_ownership_gate = (
                _static_detail_canonical_ownership_gate(
                    foliage,
                    int(static_detail_isolated_view.colmap_id),
                    camera_sequence_lookup,
                )
                if args.static_detail_canonical_ownership
                else None
            )
            static_detail_isolated_refinement_gate = (
                _static_detail_refinement_gate(
                    foliage, static_detail_isolated_ownership_gate
                )
            )
            static_detail_isolated_appearance_gate = (
                _static_detail_same_sequence_appearance_gate(
                    foliage,
                    int(static_detail_isolated_view.colmap_id),
                    camera_sequence_lookup,
                    static_detail_isolated_ownership_gate,
                    fallback_weight=(
                        args.static_detail_same_sequence_appearance_weight
                    ),
                )
            )
            static_detail_isolated_qualified_rows = int(
                (
                    foliage.static_leaf_mask
                    & (static_detail_isolated_refinement_gate > 0)
                ).sum()
            )
            static_detail_isolated_appearance_rows = int(
                (
                    foliage.static_leaf_mask
                    & (static_detail_isolated_appearance_gate > 0)
                ).sum()
            )
            static_detail_gate = foliage.static_leaf_mask.to(
                dtype=foliage.opacities.dtype
            )
            (
                static_detail_isolated_geometry_gate,
                static_detail_isolated_appearance_gate,
                static_detail_isolated_opacity_gate,
            ) = _static_detail_isolated_gradient_gates(
                static_detail_isolated_refinement_gate,
                static_detail_isolated_appearance_gate,
            )
            static_detail_isolated_package = render_hybrid(
                static_detail_isolated_view,
                surface,
                foliage,
                background=background,
                include_dynamic=False,
                volume_gate=static_detail_gate,
                volume_geometry_gradient_gate=(
                    static_detail_isolated_geometry_gate
                ),
                volume_appearance_gradient_gate=(
                    static_detail_isolated_appearance_gate
                ),
                volume_opacity_gradient_gate=(
                    static_detail_isolated_opacity_gate
                ),
                volume_opacity_scale=1.0,
                structural_trainable_start=0,
                audit_fields=torch.stack(
                    [
                        static_detail_isolated_task["p_canopy_core"],
                        static_detail_isolated_task["p_rigid"],
                        _canopy_topology_signal(
                            static_detail_isolated_target,
                            static_detail_isolated_task,
                        ),
                    ]
                ),
            )
            static_detail_isolated_prediction = composite_white_background(
                static_detail_isolated_package.render,
                static_detail_isolated_package.alpha,
                sky(static_detail_isolated_view),
            )
            static_detail_isolated_weight = (
                (
                    static_detail_isolated_task["w_gaussian_rgb"]
                    + static_detail_isolated_task["w_sky_rgb"]
                ).clamp(0, 1.5)
                * 0.50
                * static_detail_isolated_task["p_canopy"]
                * (1.0 - static_detail_isolated_task["p_transient"])
            )
            static_detail_isolated_photo = _photo_loss(
                static_detail_isolated_prediction,
                static_detail_isolated_target,
                static_detail_isolated_weight,
                opt.lambda_dssim,
            )
            static_detail_isolated_high_frequency = _high_frequency_loss(
                static_detail_isolated_prediction,
                static_detail_isolated_target,
                static_detail_isolated_task["p_canopy"]
                * (1.0 - static_detail_isolated_task["p_transient"])
                * static_detail_isolated_task["w_rgb"]
                * _boundary_evidence_weight(static_detail_isolated_task),
            )
        static_skeleton_isolated_package = None
        static_skeleton_isolated_photo = canonical.new_zeros(())
        static_skeleton_isolated_high_frequency = canonical.new_zeros(())
        static_skeleton_isolated_view = None
        static_skeleton_isolated_pixels = 0
        static_skeleton_isolated_scheduled = bool(
            args.reconstruction_target == "static"
            and foliage_active
            and args.static_skeleton_isolated_every > 0
            and args.static_skeleton_isolated_weight > 0.0
            and (step + 1) % args.static_skeleton_isolated_every == 0
            and bool(foliage.static_skeleton_mask.any())
            and len(static_skeleton_view_indices) > 0
        )
        if static_skeleton_isolated_scheduled:
            static_skeleton_isolated_view = views[
                int(
                    static_skeleton_schedule[
                        _periodic_schedule_index(
                            step, args.static_skeleton_isolated_every
                        )
                    ]
                )
            ]
            skeleton_task = fields.fields(
                static_skeleton_isolated_view.image_name,
                (
                    static_skeleton_isolated_view.image_height,
                    static_skeleton_isolated_view.image_width,
                ),
                torch.device("cuda"),
            )
            skeleton_target = (
                static_skeleton_isolated_view.original_image.cuda(
                    non_blocking=True
                )
            )
            skeleton_role = foliage.static_skeleton_mask
            static_skeleton_isolated_package = render_hybrid(
                static_skeleton_isolated_view,
                surface,
                foliage,
                background=background,
                include_dynamic=False,
                volume_role_mask=skeleton_role,
                volume_geometry_gradient_gate=skeleton_role,
                volume_appearance_gradient_gate=skeleton_role,
                volume_opacity_gradient_gate=skeleton_role,
                volume_opacity_scale=1.0,
                structural_trainable_start=None,
                optical_replacement_policy="disabled",
                audit_fields=torch.stack(
                    [
                        skeleton_task["p_tree_surface"],
                        skeleton_task["p_rigid"],
                        skeleton_task["w_skeleton"],
                    ]
                ),
            )
            skeleton_prediction = composite_white_background(
                static_skeleton_isolated_package.render,
                static_skeleton_isolated_package.alpha,
                sky(static_skeleton_isolated_view).detach(),
            )
            # MASt3R descriptor tracks cover only the selected graph cameras,
            # while optional SfM coverage can introduce independently stable
            # trunk/branch rows observed by any of the 1,487 fixed cameras.
            # Use the immutable row role projected by the native renderer as
            # additional *support*, then clip it through the real canopy mask.
            # The alpha is detached so a primitive cannot enlarge its own
            # supervision by becoming opaque, and no background/building ray
            # can be relabelled as skeleton evidence.
            projected_skeleton_support = (
                static_skeleton_isolated_package.volume_alpha[0].detach()
                * skeleton_task["p_canopy"]
                * skeleton_task["w_rgb"]
            )
            skeleton_weight = torch.maximum(
                skeleton_task["w_skeleton"],
                projected_skeleton_support,
            )
            static_skeleton_isolated_pixels = int(
                (skeleton_weight > 0).sum()
            )
            static_skeleton_isolated_photo = _photo_loss(
                skeleton_prediction,
                skeleton_target,
                skeleton_weight,
                0.10,
            )
            static_skeleton_isolated_high_frequency = (
                _high_frequency_loss(
                    skeleton_prediction,
                    skeleton_target,
                    skeleton_weight,
                )
            )
        static_volume_isolated_package = None
        static_volume_isolated_photo = canonical.new_zeros(())
        static_volume_isolated_high_frequency = canonical.new_zeros(())
        static_volume_isolated_pixels = 0
        static_volume_isolated_view = None
        static_volume_refinement_gate = None
        static_volume_isolated_qualified_rows = 0
        static_volume_isolated_appearance_rows = 0
        static_volume_isolated_scheduled = bool(
            args.reconstruction_target == "static"
            and foliage_active
            and args.static_volume_isolated_every > 0
            and args.static_volume_isolated_weight > 0.0
            and (step + 1) % args.static_volume_isolated_every == 0
            and bool(verified_static_detail.any())
            and len(static_volume_view_indices) > 0
        )
        if static_volume_isolated_scheduled:
            # This is the missing third branch forward: surface-only protects
            # rigid gradients, full mixed is authoritative for occlusion, and
            # volume-only exposes intrinsic foliage colour/bandwidth when a
            # wrongly front-running surface would otherwise cut its gradient.
            # Opacity is explicitly detached at the renderer input. Optical
            # existence remains owned by full mixed RGB and free/hit/behind
            # ray factors, so this pass cannot turn a colour residual into an
            # opaque low-frequency canopy wall.
            static_volume_isolated_view = views[
                int(
                    static_volume_schedule[
                        _periodic_schedule_index(
                            step, args.static_volume_isolated_every
                        )
                    ]
                )
            ]
            static_volume_isolated_task = fields.fields(
                static_volume_isolated_view.image_name,
                (
                    static_volume_isolated_view.image_height,
                    static_volume_isolated_view.image_width,
                ),
                torch.device("cuda"),
            )
            static_volume_isolated_target = (
                static_volume_isolated_view.original_image.cuda(
                    non_blocking=True
                )
            )
            static_volume_ownership_gate = (
                _static_detail_canonical_ownership_gate(
                    foliage,
                    int(static_volume_isolated_view.colmap_id),
                    camera_sequence_lookup,
                )
                if static_detail_active
                and args.static_detail_canonical_ownership
                else None
            )
            static_volume_appearance_ownership_gate = (
                _static_detail_same_sequence_appearance_gate(
                    foliage,
                    int(static_volume_isolated_view.colmap_id),
                    camera_sequence_lookup,
                    static_volume_ownership_gate,
                    fallback_weight=(
                        args.static_detail_same_sequence_appearance_weight
                    ),
                )
                if static_volume_ownership_gate is not None
                else None
            )
            (
                static_volume_geometry_gate,
                static_volume_appearance_gate,
                static_volume_opacity_gate,
            ) = _static_volume_isolated_gradient_gates(
                foliage,
                static_volume_ownership_gate,
                static_volume_appearance_ownership_gate,
            )
            static_volume_refinement_gate = static_volume_geometry_gate
            static_volume_isolated_qualified_rows = int(
                (static_volume_refinement_gate > 0).sum()
            )
            static_volume_isolated_appearance_rows = int(
                (static_volume_appearance_gate > 0).sum()
            )
            static_volume_isolated_package = render_hybrid(
                static_volume_isolated_view,
                surface,
                foliage,
                background=background,
                include_dynamic=False,
                surface_gate=torch.zeros_like(
                    surface.get_opacity.reshape(-1)
                ),
                volume_role_mask=foliage.static_leaf_mask,
                volume_geometry_gradient_gate=(
                    static_volume_geometry_gate
                ),
                volume_appearance_gradient_gate=(
                    static_volume_appearance_gate
                ),
                volume_opacity_gradient_gate=(
                    static_volume_opacity_gate
                ),
                volume_opacity_scale=1.0,
                structural_trainable_start=None,
                optical_replacement_policy="disabled",
            )
            isolated_alpha = (
                static_volume_isolated_package.volume_alpha
            )
            isolated_emission = (
                static_volume_isolated_package.render
                - (1.0 - isolated_alpha)
                * background[:, None, None]
            )
            isolated_intrinsic_rgb = (
                isolated_emission
                / isolated_alpha.detach().clamp_min(0.02)
            ).clamp(0.0, 1.0)
            static_volume_isolated_weight = (
                static_volume_isolated_task["p_canopy"]
                * static_volume_isolated_task["w_rgb"]
                * (isolated_alpha[0].detach() / 0.05).clamp(0.0, 1.0)
            )
            static_volume_isolated_pixels = int(
                (static_volume_isolated_weight > 0).sum()
            )
            static_volume_isolated_photo = _photo_loss(
                isolated_intrinsic_rgb,
                static_volume_isolated_target,
                static_volume_isolated_weight,
                0.10,
            )
            static_volume_isolated_high_frequency = _high_frequency_loss(
                isolated_intrinsic_rgb,
                static_volume_isolated_target,
                static_volume_isolated_weight
                * _boundary_evidence_weight(
                    static_volume_isolated_task
                ),
            )
        static_detail_global_cleanup_package = None
        static_detail_global_cleanup = canonical.new_zeros(())
        static_detail_global_cleanup_audit: dict[str, object] = {
            "contract": STATIC_DETAIL_GLOBAL_CLEANUP_CONTRACT,
            "scheduled": False,
            "rigid_free": 0.0,
            "rigid_free_supported_pixels": 0,
            "counterfactual_weight": float(
                args.static_detail_global_counterfactual_weight
            ),
            "counterfactual": {
                "contract": COUNTERFACTUAL_TRANSPARENCY_CONTRACT,
                "supported_pixels": 0,
                "mean_responsibility": 0.0,
                "mean_relative_rgb_advantage": 0.0,
                "mean_volume_alpha_on_support": 0.0,
                "canopy_rgb_retirement_blocked_pixels": 0,
                "canopy_rgb_retirement_blocked_mass": 0.0,
            },
            "gradient_permissions": {
                "static_detail_xyz_scale_rotation": True,
                "static_detail_optical_mass": True,
                "static_detail_sh": False,
                "surface_sky_uncertainty": False,
                "topology_statistics": False,
            },
            "evidence_sequence_rows": 0,
            "unrelated_sequence_rows_blocked": 0,
        }
        static_detail_global_cleanup_scheduled = bool(
            args.reconstruction_target == "static"
            and foliage_active
            and static_detail_active
            and _static_detail_stage_trainable(phase)
            and args.static_detail_global_cleanup_every > 0
            and args.static_detail_global_cleanup_weight > 0.0
            and (step + 1)
            % args.static_detail_global_cleanup_every
            == 0
            and bool(foliage.static_leaf_mask.any())
        )
        if static_detail_global_cleanup_scheduled:
            # A missing leaf in an unrelated traversal is seasonal absence,
            # not calibrated free space for this primitive. Route negative
            # cleanup through exactly the same persisted positive-evidence
            # sequences that may restore optical coverage. Surface and SH
            # values participate in the forward comparison but are read-only;
            # this package remains excluded from densification statistics.
            detail_rows = foliage.static_leaf_mask
            if args.static_detail_canonical_ownership:
                cleanup_rows = (
                    _static_detail_positive_evidence_sequence_gate(
                        foliage,
                        int(view.colmap_id),
                        camera_sequence_lookup,
                    )
                )
            else:
                cleanup_rows = detail_rows
            detail_gate = cleanup_rows.to(
                dtype=foliage.opacities.dtype
            )
            no_appearance_gradient = torch.zeros_like(
                foliage.opacities
            )
            static_detail_global_cleanup_package = render_hybrid(
                view,
                surface,
                foliage,
                background=background,
                include_dynamic=False,
                volume_gate=detail_gate,
                volume_geometry_gradient_gate=None,
                volume_appearance_gradient_gate=no_appearance_gradient,
                volume_opacity_gradient_gate=None,
                volume_opacity_scale=1.0,
                optical_replacement_policy="disabled",
                structural_trainable_start=None,
            )
            static_detail_global_cleanup_prediction = (
                composite_white_background(
                    static_detail_global_cleanup_package.render,
                    static_detail_global_cleanup_package.alpha,
                    sky(view).detach(),
                )
            )
            (
                static_detail_global_cleanup,
                static_detail_global_cleanup_audit,
            ) = _static_detail_global_cleanup_loss(
                static_detail_global_cleanup_prediction,
                structural_prediction,
                target,
                static_detail_global_cleanup_package.volume_alpha,
                structural_package.surface_alpha,
                task,
                counterfactual_weight=(
                    args.static_detail_global_counterfactual_weight
                ),
                margin=args.counterfactual_transparency_margin,
                temperature=args.counterfactual_transparency_temperature,
            )
            static_detail_global_cleanup_audit["scheduled"] = True
            static_detail_global_cleanup_audit[
                "evidence_sequence_rows"
            ] = int(cleanup_rows.sum())
            static_detail_global_cleanup_audit[
                "unrelated_sequence_rows_blocked"
            ] = int((detail_rows & ~cleanup_rows).sum())
        persistent_envelope_global_cleanup_package = None
        persistent_envelope_global_cleanup = canonical.new_zeros(())
        persistent_envelope_global_cleanup_audit: dict[str, object] = {
            "contract": PERSISTENT_ENVELOPE_GLOBAL_CLEANUP_CONTRACT,
            "scheduled": False,
            "rigid_free": 0.0,
            "rigid_free_supported_pixels": 0,
            "counterfactual_weight": float(
                args.persistent_envelope_global_counterfactual_weight
            ),
            "counterfactual": {
                "contract": COUNTERFACTUAL_TRANSPARENCY_CONTRACT,
                "supported_pixels": 0,
                "mean_responsibility": 0.0,
                "mean_relative_rgb_advantage": 0.0,
                "mean_volume_alpha_on_support": 0.0,
                "canopy_rgb_retirement_blocked_pixels": 0,
                "canopy_rgb_retirement_blocked_mass": 0.0,
            },
            "gradient_permissions": {
                "persistent_envelope_xyz_scale_rotation": True,
                "persistent_envelope_optical_mass": True,
                "persistent_envelope_sh": False,
                "surface_sky_uncertainty": False,
                "topology_statistics": False,
            },
        }
        persistent_envelope_global_cleanup_scheduled = bool(
            args.reconstruction_target == "static"
            and foliage_active
            and static_detail_active
            and _static_detail_stage_trainable(phase)
            and args.persistent_envelope_global_cleanup_every > 0
            and args.persistent_envelope_global_cleanup_weight > 0.0
            and (step + 1)
            % args.persistent_envelope_global_cleanup_every
            == 0
            and bool(foliage.persistent_envelope_mask.any())
        )
        if persistent_envelope_global_cleanup_scheduled:
            # The envelope is globally visible, so every calibrated camera
            # owns negative evidence against it.  Isolating the role prevents
            # leaf/detail alpha from hiding the envelope's responsibility and
            # prevents the same pass from eroding verified leaf coverage.
            envelope_gate = foliage.persistent_envelope_mask.to(
                dtype=foliage.opacities.dtype
            )
            envelope_no_appearance_gradient = torch.zeros_like(
                foliage.opacities
            )
            persistent_envelope_global_cleanup_package = render_hybrid(
                view,
                surface,
                foliage,
                background=background,
                include_dynamic=False,
                volume_gate=envelope_gate,
                volume_geometry_gradient_gate=None,
                volume_appearance_gradient_gate=(
                    envelope_no_appearance_gradient
                ),
                volume_opacity_gradient_gate=None,
                volume_opacity_scale=1.0,
                optical_replacement_policy="disabled",
                structural_trainable_start=None,
            )
            persistent_envelope_global_cleanup_prediction = (
                composite_white_background(
                    persistent_envelope_global_cleanup_package.render,
                    persistent_envelope_global_cleanup_package.alpha,
                    sky(view).detach(),
                )
            )
            (
                persistent_envelope_global_cleanup,
                persistent_envelope_global_cleanup_audit,
            ) = _persistent_envelope_global_cleanup_loss(
                persistent_envelope_global_cleanup_prediction,
                structural_prediction,
                target,
                persistent_envelope_global_cleanup_package.volume_alpha,
                structural_package.surface_alpha,
                task,
                counterfactual_weight=(
                    args.persistent_envelope_global_counterfactual_weight
                ),
                margin=args.counterfactual_transparency_margin,
                temperature=args.counterfactual_transparency_temperature,
            )
            persistent_envelope_global_cleanup_audit["scheduled"] = True
        rigid_photo = isolated_rgb["rigid"]
        rigid_high_frequency = isolated_rgb["rigid_high_frequency"]
        (
            rigid_residual_patch,
            rigid_residual_patch_audit,
        ) = _rigid_residual_patch_loss(
            structural_prediction,
            target,
            task["p_rigid"]
            * (1.0 - task["p_transient"])
            * _boundary_evidence_weight(task),
        )
        dav2_patch_uv = dav2_observation_patches.get(
            str(Path(str(view.image_name)).stem)
        )
        dav2_observation_photo, dav2_observation_pixels = (
            _dav2_observation_patch_loss(
                structural_prediction,
                target,
                dav2_patch_uv,
            )
        )
        surface_in_canopy = _soft_surface_canopy_conflict(
            package,
            task,
        )
        counterfactual_transparency = canonical.new_zeros(())
        counterfactual_transparency_audit = {
            "contract": COUNTERFACTUAL_TRANSPARENCY_CONTRACT,
            "supported_pixels": 0,
            "mean_responsibility": 0.0,
            "mean_relative_rgb_advantage": 0.0,
            "mean_volume_alpha_on_support": 0.0,
            "canopy_rgb_retirement_blocked_pixels": 0,
            "canopy_rgb_retirement_blocked_mass": 0.0,
        }
        if (
            foliage_active
            and args.counterfactual_transparency_weight > 0
        ):
            (
                counterfactual_transparency,
                counterfactual_transparency_audit,
            ) = _counterfactual_volume_transparency_loss(
                canonical,
                structural_prediction,
                target,
                package.volume_alpha,
                structural_package.surface_alpha,
                task,
                margin=args.counterfactual_transparency_margin,
                temperature=args.counterfactual_transparency_temperature,
            )
        volume_in_rigid = _weighted_mean(
            package.volume_alpha,
            task["p_rigid"],
        )
        finite_in_sky = _weighted_mean(
            package.surface_alpha + package.volume_alpha,
            task["p_sky"],
        )
        finite_in_transient = _weighted_mean(
            package.surface_alpha + package.volume_alpha,
            task["p_transient"]
            * task["w_rgb"]
            * _boundary_evidence_weight(task),
        )
        ownership = (
            surface_in_canopy
            + volume_in_rigid
            + finite_in_sky
            + finite_in_transient
        )
        alpha = package.volume_alpha[0].clamp(1e-5, 1 - 1e-5)
        negative = (
            task["p_rigid"] + 0.35 * task["p_sky"]
        ).clamp(0, 1) * _boundary_evidence_weight(task)
        free = (
            (-torch.log1p(-alpha) * negative).sum()
            / negative.sum().clamp_min(1)
        )
        # A tree mask is not a target alpha.  Occupancy is supervised by the
        # per-candidate visual-hull ray/depth posterior produced during
        # initialization, while projected rigid/sky pixels supply free-space
        # contradictions.  This removes the former fixed alpha=0.35 bias that
        # broadened crowns into translucent paint.
        posterior = foliage.occupancy_probability.clamp(0.01, 0.99)
        posterior_weight = (
            foliage.support_view_count.float()
            / (
                foliage.support_view_count.float()
                + foliage.unknown_view_count.float()
            ).clamp_min(1)
        ).clamp(0.05, 1.0)
        posterior_reliability = torch.rsqrt(
            1.0 + foliage.ray_depth_nll.clamp_min(0)
        ).clamp(0.05, 1.0)
        posterior_weight = posterior_weight * posterior_reliability
        active_canonical_volume = ~foliage.dynamic_leaf_mask
        if (
            args.reconstruction_target == "static"
            and not static_detail_active
        ):
            active_canonical_volume = (
                active_canonical_volume & ~foliage.static_leaf_mask
            )
        canonical_volume = active_canonical_volume.to(
            posterior_weight.dtype
        )
        canonical_posterior_weight = (
            posterior_weight * canonical_volume
        )
        # Visual-hull occupancy is the probability that a candidate exists;
        # it is not the target alpha of every overlapping Gaussian. The old
        # BCE drove each supported leaf toward opacity~occupancy, so dozens of
        # valid candidates on one ray inevitably became a solid paint layer.
        false_positive_opacity = (
            foliage.opacities
            * (1.0 - posterior)
            * canonical_posterior_weight
        ).sum() / canonical_posterior_weight.sum().clamp_min(1)
        opacity_complexity = (
            foliage.opacities.square() * canonical_posterior_weight
        ).sum() / canonical_posterior_weight.sum().clamp_min(1)
        ray_loss = package.depth.new_zeros(())
        ray_audit = {
            "rays": 0,
            "candidate_evaluations": 0,
            "canonical_candidate_evaluations": 0,
            "canonical_hit_candidate_evaluations": 0,
            "exact_dynamic_candidate_evaluations": 0,
            "confirmed_free_rays": 0,
            "hit_rays": 0,
            "unknown_rays": 0,
            "free": 0.0,
            "hit": 0.0,
        }
        if (
            foliage_active
            and (step + 1) % args.ray_posterior_every == 0
        ):
            ray_update = (step + 1) // args.ray_posterior_every - 1
            ray_camera_id = foliage_rays.scheduled_camera_id(ray_update)
            if ray_camera_id is not None:
                (
                    ray_free_candidate_mask,
                    ray_hit_candidate_mask,
                ) = _static_ray_candidate_masks(
                    args,
                    foliage,
                    active_canonical_volume,
                    phase=phase,
                    camera_id=int(ray_camera_id),
                    camera_sequence_lookup=camera_sequence_lookup,
                )
                ray_loss, ray_audit = foliage_rays.interval_factor(
                    view_by_camera_id[ray_camera_id],
                    foliage,
                    maximum_rays=effective_ray_posterior_maximum_rays,
                    maximum_candidates_per_ray=(
                        args.ray_posterior_maximum_candidates
                    ),
                    sample_update=ray_update,
                    canonical_candidate_mask=ray_free_candidate_mask,
                    canonical_hit_candidate_mask=ray_hit_candidate_mask,
                )
                if args.reconstruction_target == "static":
                    # Consensus construction is independent of render phase.
                    # New detail remains hidden until static_foliage, but the
                    # first ray epoch must already contribute proposals.
                    proposals = foliage_rays.pop_uncovered_hit_proposals()
                    static_ray_births.add(
                        proposals,
                        sequence_id=sequence_id(
                            view_by_camera_id[ray_camera_id].image_name
                        ),
                    )
        occupancy = (
            free
            + 0.05 * false_positive_opacity
            + 0.002 * opacity_complexity
            + ray_loss
            if foliage_active
            else package.depth.new_zeros(())
        )
        delta = foliage.xyz - foliage.initialization_center
        variance = foliage.position_covariance.diagonal(
            dim1=-2, dim2=-1
        ).clamp_min(1e-6)
        position_nll = (delta.square() / variance).sum(-1)
        skeleton_confidence = getattr(
            foliage,
            "static_skeleton_confidence",
            torch.ones_like(position_nll),
        ).clamp(0.0, 1.0)
        skeleton_rigidity = (
            skeleton_confidence
            / (skeleton_confidence + 0.20)
        ).clamp(0.0, 1.0)
        position_weight = position_nll.new_full(
            position_nll.shape, 0.20
        )
        position_weight = torch.where(
            foliage.static_skeleton_mask,
            0.20 + 1.80 * skeleton_rigidity,
            position_weight,
        )
        geometry_floor = (
            position_nll * position_weight * canonical_volume
        ).sum() / canonical_volume.sum().clamp_min(1)
        canonical_loss = (
            photo
            + 0.45 * rigid_photo
            + 0.12 * rigid_high_frequency
            + args.rigid_residual_patch_weight
            * rigid_residual_patch
            + args.dav2_observation_patch_weight
            * dav2_observation_photo
            + args.ownership_weight * ownership
            + args.counterfactual_transparency_weight
            * counterfactual_transparency
            + args.occupancy_weight * occupancy
            + (0.015 * geometry_floor if foliage_active else 0.0)
            + args.chart_atlas_regularization_weight
            * chart_surface.regularizer()
        )
        parameter_loss_gradient_audit = (
            _parameter_loss_gradient_audit(
                foliage,
                {
                    "ray_free_hit": occupancy,
                    "canopy_rgb": canopy_photo,
                    "canopy_high_frequency": static_canopy_high_frequency,
                    "detail_isolated_rgb": (
                        static_detail_isolated_photo
                    ),
                    "detail_isolated_high_frequency": (
                        static_detail_isolated_high_frequency
                    ),
                    "volume_isolated_rgb": (
                        static_volume_isolated_photo
                    ),
                    "volume_isolated_high_frequency": (
                        static_volume_isolated_high_frequency
                    ),
                    "rigid_spill": volume_in_rigid,
                    "counterfactual_transparency": (
                        counterfactual_transparency
                    ),
                    "global_static_detail_negative_cleanup": (
                        static_detail_global_cleanup
                    ),
                    "global_persistent_envelope_negative_cleanup": (
                        persistent_envelope_global_cleanup
                    ),
                    "position_prior": geometry_floor,
                },
            )
            if step == 0 or (step + 1) % args.log_every == 0
            else None
        )
        # Rigid/sky RGB, geometry and ownership update their normal owners.
        # Canopy RGB is differentiated only with respect to the foliage
        # branch.  This keeps the exact same jointly sorted image formation,
        # but prevents tree pixels from painting structural surfels or sky.
        canonical_loss.backward(retain_graph=foliage_active)
        canonical_evidence_opacity_gradient = (
            foliage.opacity_logits.grad.detach().clone()
            if foliage_active
            and foliage.opacity_logits.grad is not None
            else None
        )
        if foliage_active:
            torch.autograd.backward(
                canopy_photo
                + args.high_frequency_weight
                * static_canopy_high_frequency,
                inputs=tuple(foliage.parameters()),
            )
        if static_detail_isolated_package is not None:
            static_detail_isolated_objective = (
                args.static_detail_isolated_weight
                * (
                    static_detail_isolated_photo
                    + args.high_frequency_weight
                    * static_detail_isolated_high_frequency
                )
            )
            static_detail_inputs = tuple(foliage.parameters())
            if (
                static_detail_isolated_package.volume_means2d is not None
                and static_detail_isolated_package.volume_means2d.requires_grad
            ):
                static_detail_inputs = (
                    *static_detail_inputs,
                    static_detail_isolated_package.volume_means2d,
                )
            torch.autograd.backward(
                static_detail_isolated_objective,
                inputs=static_detail_inputs,
            )
        if static_skeleton_isolated_package is not None:
            static_skeleton_isolated_objective = (
                args.static_skeleton_isolated_weight
                * (
                    static_skeleton_isolated_photo
                    + args.high_frequency_weight
                    * static_skeleton_isolated_high_frequency
                )
            )
            static_skeleton_inputs = tuple(foliage.parameters())
            if (
                static_skeleton_isolated_package.volume_means2d is not None
                and static_skeleton_isolated_package.volume_means2d.requires_grad
            ):
                static_skeleton_inputs = (
                    *static_skeleton_inputs,
                    static_skeleton_isolated_package.volume_means2d,
                )
            torch.autograd.backward(
                static_skeleton_isolated_objective,
                inputs=static_skeleton_inputs,
            )
        if static_volume_isolated_package is not None:
            static_volume_isolated_objective = (
                args.static_volume_isolated_weight
                * (
                    static_volume_isolated_photo
                    + args.high_frequency_weight
                    * static_volume_isolated_high_frequency
                )
            )
            static_volume_isolated_inputs = tuple(foliage.parameters())
            if (
                static_volume_isolated_package.volume_means2d is not None
                and static_volume_isolated_package.volume_means2d.requires_grad
            ):
                static_volume_isolated_inputs = (
                    *static_volume_isolated_inputs,
                    static_volume_isolated_package.volume_means2d,
                )
            torch.autograd.backward(
                static_volume_isolated_objective,
                inputs=static_volume_isolated_inputs,
            )
        if static_detail_global_cleanup_package is not None:
            torch.autograd.backward(
                args.static_detail_global_cleanup_weight
                * static_detail_global_cleanup,
                inputs=(
                    foliage.xyz,
                    foliage.log_scales,
                    foliage.quaternions,
                    foliage.opacity_logits,
                ),
            )
            # This evidence stream is deliberately absent from
            # ``_accumulate_volume_stats``: a rigid/free-space contradiction
            # may move, shrink or retire detail, but can never request a
            # split/birth or spend topology capacity.
            del static_detail_global_cleanup_package
        if persistent_envelope_global_cleanup_package is not None:
            torch.autograd.backward(
                args.persistent_envelope_global_cleanup_weight
                * persistent_envelope_global_cleanup,
                inputs=(
                    foliage.xyz,
                    foliage.log_scales,
                    foliage.quaternions,
                    foliage.opacity_logits,
                ),
            )
            # This role-isolated contradiction stream is excluded from all
            # screen-gradient/topology statistics. It can only clean the
            # current envelope population, never create replacement demand.
            del persistent_envelope_global_cleanup_package
        # Surface topology only consumes rigid-dominant responsibility.  A
        # canopy pixel can improve color/opacity, but never create a new 2D
        # surfel or a giant facade/tree bridge.
        surface_responsibility = structural_package.responsibility[
            : structural_package.structural_count
        ]
        total_responsibility = surface_responsibility[:, 0].clamp_min(
            1e-8
        )
        rigid_fraction = (
            surface_responsibility[:, 2] / total_responsibility
        )
        canopy_fraction = (
            surface_responsibility[:, 1] / total_responsibility
        )
        surface_topology_weight = (
            rigid_fraction * (1.0 - canopy_fraction)
        ).clamp(0.0, 1.0)
        surface_topology_weight *= (
            structural_package.radii[
                : structural_package.structural_count
            ]
            > 0
        ).to(surface_topology_weight)
        if args.mature_handoff_surface_policy == "atlas_residual":
            # Screen-space gradients from the immutable handoff prefix are
            # useful for diagnostics but cannot request suffix capacity.  A
            # tree pixel or exposed facade must never turn a mature prefix
            # row into a hidden topology donor.
            surface_topology_weight[
                :rigid_surface_residual_start
            ] = 0
        surface_visible = surface_topology_weight > 0
        if structural_package.surface_means2d.grad is not None:
            surface.add_densification_stats(
                structural_package.surface_means2d,
                surface_topology_weight,
            )
            weighted_radius = structural_package.radii[
                : structural_package.structural_count
            ] * torch.sqrt(surface_topology_weight)
            surface.max_radii2D[surface_visible] = torch.maximum(
                surface.max_radii2D[surface_visible],
                weighted_radius[surface_visible],
            )
        surface_rgb_means2d_gradient_norm = (
            0.0
            if structural_package.surface_means2d.grad is None
            else float(
                torch.nan_to_num(
                    structural_package.surface_means2d.grad
                )
                .norm()
                .detach()
            )
        )
        del structural_package, structural_prediction
        _accumulate_volume_stats(
            volume_stats,
            package,
            (
                static_detail_refinement_gradient_gate
                if static_detail_active
                else static_detail_gradient_gate
            ),
        )
        if static_detail_isolated_package is not None:
            _accumulate_volume_stats(
                volume_stats,
                static_detail_isolated_package,
                foliage.static_leaf_mask
                & (static_detail_isolated_refinement_gate > 0),
            )
            del static_detail_isolated_package
        if static_skeleton_isolated_package is not None:
            _accumulate_volume_stats(
                volume_stats,
                static_skeleton_isolated_package,
                foliage.static_skeleton_mask,
            )
            del static_skeleton_isolated_package
        if static_volume_isolated_package is not None:
            static_volume_isolated_gate = foliage.static_leaf_mask.clone()
            if static_volume_geometry_gate is not None:
                static_volume_isolated_gate &= (
                    static_volume_geometry_gate > 0
                )
            _accumulate_volume_stats(
                volume_stats,
                static_volume_isolated_package,
                static_volume_isolated_gate,
            )
            del static_volume_isolated_package
        static_child_verification_audit = (
            _update_static_child_verification_from_render(
                foliage,
                package,
                camera_id=int(view.colmap_id),
                camera_sequence_lookup=camera_sequence_lookup,
                volume_optimizer=volume_optimizer,
            )
            if args.reconstruction_target == "static" and foliage_active
            else {
                "contract": "real_ray_child_reverification",
                "candidate_rows": 0,
                "new_camera_witnesses": 0,
                "new_color_witnesses": 0,
                "newly_verified": 0,
            }
        )
        static_replacement_scheduled = (
            static_replacement_candidate_scheduled
        )
        if static_replacement_scheduled:
            # Replacement evidence must visit the exact support cameras of
            # verified detail. Reusing the random authoritative RGB camera
            # made almost every audit a no-candidate event even though tens of
            # thousands of supported envelope/detail groups were visible.
            # Cycle the existing complete exact-support schedule; this changes
            # neither the training sample nor the one-pass authoritative
            # render, and the role-isolated audit remains read-only.
            replacement_view = views[
                int(
                    static_detail_schedule[
                        _periodic_schedule_index(
                            step,
                            args.static_replacement_evidence_every,
                        )
                    ]
                )
            ]
            replacement_task = fields.fields(
                replacement_view.image_name,
                (
                    replacement_view.image_height,
                    replacement_view.image_width,
                ),
                torch.device("cuda"),
            )
            (
                ray_local_authority,
                ray_local_visible,
                ray_local_candidate_visible,
                ray_local_measurement,
            ) = _measure_static_ray_local_replacement(
                replacement_view,
                surface,
                foliage,
                background,
                replacement_task,
                package,
            )
            static_replacement_audit = {
                **ray_local_measurement,
                "training_camera_id": int(view.colmap_id),
                "audit_camera_id": int(replacement_view.colmap_id),
                **_accumulate_static_replacement_evidence(
                    foliage,
                    ray_local_authority,
                    ray_local_visible,
                    candidate_visible=ray_local_candidate_visible,
                    camera_id=int(replacement_view.colmap_id),
                    decay=args.static_replacement_ema_decay,
                ),
            }
        else:
            static_replacement_audit = {
                "contract": (
                    "persistent_distinct_view_real_ray_replacement_ema"
                    if static_replacement_lifecycle_enabled
                    else "persistent_lifecycle_disabled_by_zero_cadence_or_mass"
                ),
                "scheduled": False,
                "observed_envelope_rows": 0,
                "overlapping_envelope_rows": 0,
                "mean_authority": 0.0,
            }

        conditioned_loss = canonical_loss.new_zeros(())
        conditioned_counterfactual_transparency = canonical_loss.new_zeros(())
        conditioned_counterfactual_audit = {
            "contract": COUNTERFACTUAL_TRANSPARENCY_CONTRACT,
            "scheduled": False,
            "supported_pixels": 0,
            "mean_responsibility": 0.0,
            "mean_relative_rgb_advantage": 0.0,
            "mean_volume_alpha_on_support": 0.0,
            "canopy_rgb_retirement_blocked_pixels": 0,
            "canopy_rgb_retirement_blocked_mass": 0.0,
        }
        uncertainty_loss = canonical_loss.new_zeros(())
        if static_spatial_uncertainty_active:
            # A proper heteroscedastic likelihood learns where the immutable
            # static map cannot explain traversal-dependent foliage. The RGB
            # prediction is detached so uncertainty cannot move geometry,
            # colour or opacity; its detached sigma only robustifies the next
            # canonical visits through the confidence field above.
            uncertainty_loss = (
                appearance.heteroscedastic_loss(
                    canonical.detach(),
                    target,
                    task,
                    image_name=view.image_name,
                    sigma=canonical_sigma,
                )
                + 0.05
                * appearance.uncertainty_regularization(
                    view.image_name,
                    (view.image_height, view.image_width),
                    sigma=canonical_sigma,
                )
            )
        high_frequency_loss = canonical_loss.new_zeros(())
        conditioned_ownership = canonical_loss.new_zeros(())
        dynamic_observation_loss = canonical_loss.new_zeros(())
        dynamic_observed = torch.zeros(
            len(foliage), dtype=torch.bool, device=foliage.xyz.device
        )
        dynamic_observation_audit = {"matched": 0}
        dynamic_gate_audit = {
            "exact": 0,
            "fallback": 0,
            "mean": 0.0,
        }
        conditioned_volume_radii = None
        conditioned_view = None
        if dynamic_active:
            conditioned_view_index = int(conditioned_schedule[step])
            conditioned_visit_counts[conditioned_view_index] += 1
            conditioned_view = views[conditioned_view_index]
            conditioned_task_fields = fields.fields(
                conditioned_view.image_name,
                (
                    conditioned_view.image_height,
                    conditioned_view.image_width,
                ),
                torch.device("cuda"),
            )
            conditioned_target = conditioned_view.original_image.cuda(
                non_blocking=True
            )
            conditioned_topology_signal = _canopy_topology_signal(
                conditioned_target, conditioned_task_fields
            )
            dynamic_gate = _dynamic_visibility_gate(
                foliage,
                conditioned_view,
                camera_sequence_lookup,
                camera_frame_lookup,
            )
            # Exact observation cameras own every dynamic primitive's base
            # xyz/colour/opacity.  Neighbouring frames own only the low-rank
            # deformation/feature/opacity residuals that were introduced for
            # temporal interpolation.  The former one-gate implementation
            # forced a false choice between cross-view base-state averaging
            # (v52) and a completely read-only temporal branch (v54).
            exact_dynamic_gradient_gate = (
                _conditioned_base_gradient_gate(
                    foliage, dynamic_gate
                )
            )
            temporal_residual_gradient_gate = torch.where(
                foliage.dynamic_leaf_mask,
                (dynamic_gate > 0).to(dynamic_gate.dtype),
                torch.ones_like(dynamic_gate),
            )
            dynamic_gate_values = dynamic_gate[
                foliage.dynamic_leaf_mask
            ]
            dynamic_gate_audit = {
                "exact": int((dynamic_gate_values >= 0.999).sum()),
                "fallback": int(
                    (
                        (dynamic_gate_values > 0)
                        & (dynamic_gate_values < 0.999)
                    ).sum()
                ),
                "mean": (
                    float(dynamic_gate_values.mean())
                    if len(dynamic_gate_values)
                    else 0.0
                ),
            }
            temporal_code = appearance.temporal_code(
                conditioned_view.image_name
            )
            conditioned_package = render_hybrid(
                conditioned_view,
                surface,
                foliage,
                background=background,
                temporal_code=temporal_code,
                include_dynamic=True,
                volume_gate=dynamic_gate,
                volume_geometry_gradient_gate=(
                    exact_dynamic_gradient_gate
                ),
                volume_appearance_gradient_gate=(
                    exact_dynamic_gradient_gate
                ),
                volume_opacity_gradient_gate=(
                    exact_dynamic_gradient_gate
                ),
                volume_dynamic_geometry_gradient_gate=(
                    temporal_residual_gradient_gate
                ),
                volume_dynamic_appearance_gradient_gate=(
                    temporal_residual_gradient_gate
                ),
                volume_dynamic_opacity_gradient_gate=(
                    temporal_residual_gradient_gate
                ),
                optical_replacement_policy=(
                    args.optical_replacement_policy
                ),
                structural_trainable_start=None,
                audit_fields=torch.stack(
                    [
                        conditioned_task_fields["p_canopy_core"],
                        conditioned_task_fields["p_rigid"],
                        conditioned_topology_signal,
                    ]
                ),
            )
            conditioned_base = composite_white_background(
                conditioned_package.render,
                conditioned_package.alpha,
                sky(conditioned_view),
            )
            if (
                args.counterfactual_transparency_weight > 0
                and (step + 1)
                % args.conditioned_counterfactual_every
                == 0
            ):
                # This render is evidence only.  Keeping it out of autograd
                # avoids a second surface/sky owner and leaves the mixed
                # package's volume alpha as the sole trainable quantity.
                with torch.no_grad():
                    conditioned_surface_package = render_hybrid(
                        conditioned_view,
                        surface,
                        foliage,
                        background=background,
                        include_dynamic=False,
                        volume_opacity_scale=0.0,
                        structural_trainable_start=None,
                    )
                    conditioned_surface_prediction = (
                        composite_white_background(
                            conditioned_surface_package.render,
                            conditioned_surface_package.alpha,
                            sky(conditioned_view),
                        )
                    )
                (
                    conditioned_counterfactual_transparency,
                    conditioned_counterfactual_audit,
                ) = _counterfactual_volume_transparency_loss(
                    conditioned_base,
                    conditioned_surface_prediction,
                    conditioned_target,
                    conditioned_package.volume_alpha,
                    conditioned_surface_package.surface_alpha,
                    conditioned_task_fields,
                    margin=args.counterfactual_transparency_margin,
                    temperature=args.counterfactual_transparency_temperature,
                )
                conditioned_counterfactual_audit["scheduled"] = True
                del (
                    conditioned_surface_package,
                    conditioned_surface_prediction,
                )
            # Appearance routing must use the same deployable primitive
            # ownership as HybridTeacherAPI.  Ground-truth masks define loss
            # support only; feeding them into the appearance decoder during
            # training created a train/deploy ownership conflict.
            surface_owner = conditioned_package.surface_alpha[0].clamp(0, 1)
            volume_owner = conditioned_package.volume_alpha[0].clamp(0, 1)
            sky_owner = (1.0 - conditioned_package.alpha[0]).clamp(0, 1)
            owner_sum = (
                surface_owner + volume_owner + sky_owner
            ).clamp_min(1e-6)
            conditioned_task = dict(conditioned_task_fields)
            conditioned_task.update(
                {
                    "p_rigid": surface_owner / owner_sum,
                    "p_canopy": volume_owner / owner_sum,
                    "p_canopy_core": volume_owner / owner_sum,
                    "p_sky": sky_owner / owner_sum,
                }
            )
            conditioned = appearance(
                conditioned_base, conditioned_view, conditioned_task
            )
            conditioned_ownership = (
                _weighted_mean(
                    conditioned_package.volume_alpha,
                    conditioned_task_fields["p_rigid"]
                    * _boundary_evidence_weight(
                        conditioned_task_fields
                    ),
                )
                + _weighted_mean(
                    conditioned_package.surface_alpha
                    + conditioned_package.volume_alpha,
                    conditioned_task_fields["p_sky"]
                    * _boundary_evidence_weight(
                        conditioned_task_fields
                    ),
                )
            )
            conditioned_photo_parts = _conditioned_photo_losses(
                conditioned,
                conditioned_target,
                conditioned_task_fields,
            )
            conditioned_photo = conditioned_photo_parts["combined"]
            conditioned_canopy_photo = conditioned_photo_parts["canopy"]
            if phase in {"dynamic_appearance", "ownership_cleanup"}:
                uncertainty_loss = (
                    appearance.heteroscedastic_loss(
                        # Direct conditioned RGB already supplies the model
                        # gradient.  The uncertainty likelihood estimates a
                        # spatial residual scale and influences subsequent
                        # canonical routing; letting its initial tiny sigma
                        # also backpropagate into geometry multiplied early
                        # image gradients by O(100).
                        conditioned.detach(),
                        conditioned_target,
                        conditioned_task,
                        image_name=conditioned_view.image_name,
                    )
                    + 0.05
                    * appearance.uncertainty_regularization(
                        conditioned_view.image_name,
                        (
                            conditioned_view.image_height,
                            conditioned_view.image_width,
                        ),
                    )
                )
            high_frequency_loss = _high_frequency_loss(
                conditioned_base,
                conditioned_target,
                conditioned_task_fields["p_canopy"]
                * (1.0 - conditioned_task_fields["p_transient"])
                * conditioned_task_fields["w_rgb"]
                * _boundary_evidence_weight(
                    conditioned_task_fields
                ),
            )
            owner_dynamic = (
                foliage.dynamic_leaf_mask
                & (exact_dynamic_gradient_gate >= 0.999)
            )
            dynamic_regularization = canonical_loss.new_zeros(())
            if bool(owner_dynamic.any()):
                dynamic_regularization = (
                    foliage.deformation_basis[owner_dynamic]
                    .square()
                    .mean()
                )
            dynamic_presence = canonical_loss.new_zeros(())
            if bool(owner_dynamic.any()):
                minimum_logit = torch.logit(
                    _dynamic_opacity_floor(foliage)[owner_dynamic]
                    .clamp(1e-6, 1.0 - 1e-6)
                )
                dynamic_presence = (
                    minimum_logit
                    - foliage.opacity_logits[owner_dynamic, 0]
                ).clamp_min(0).square().mean()
            (
                dynamic_observation_loss,
                dynamic_observed,
                dynamic_observation_audit,
            ) = _dynamic_observation_factor(
                foliage,
                conditioned_view,
                temporal_code,
                sample_update=step,
            )
            conditioned_loss = (
                args.dynamic_weight * conditioned_photo
                + args.appearance_weight * uncertainty_loss
                + args.high_frequency_weight * high_frequency_loss
                + args.conditioned_ownership_weight
                * conditioned_ownership
                + 1e-3 * appearance.regularization()
                + 0.02 * dynamic_regularization
                + 0.02 * dynamic_presence
                + 0.10 * dynamic_observation_loss
            )
            # The full combined image objective updates only the low-capacity
            # appearance/uncertainty and temporal-code module. Foliage base
            # geometry, colour and opacity receive photometric gradients only
            # where immutable task evidence labels canopy. Ownership,
            # high-frequency and observation factors still update foliage and
            # keep the same jointly sorted forward render.
            foliage_conditioned_loss = (
                args.dynamic_weight * conditioned_canopy_photo
                + args.high_frequency_weight * high_frequency_loss
                + args.conditioned_ownership_weight
                * conditioned_ownership
                + args.counterfactual_transparency_weight
                * conditioned_counterfactual_transparency
                + 0.02 * dynamic_regularization
                + 0.02 * dynamic_presence
                + 0.10 * dynamic_observation_loss
            )
            _backward_conditioned_foliage(
                foliage_conditioned_loss,
                foliage,
                conditioned_package.volume_means2d,
            )
            torch.autograd.backward(
                conditioned_loss,
                inputs=_trainable_parameters(appearance),
            )
            dynamic_gate_audit.update(
                {
                    "base_owner_rows": int(
                        (
                            foliage.dynamic_leaf_mask
                            & (exact_dynamic_gradient_gate > 0)
                        ).sum()
                    ),
                    "temporal_residual_rows": int(
                        (
                            foliage.dynamic_leaf_mask
                            & (temporal_residual_gradient_gate > 0)
                        ).sum()
                    ),
                    "ownership_contract": (
                        "exact_base_soft_temporal_residual_plus_"
                        "shared_canonical_group_conditioning"
                    ),
                }
            )
            _accumulate_dynamic_observation_stats(
                volume_stats, foliage, dynamic_observed
            )
            _accumulate_volume_stats(
                volume_stats,
                conditioned_package,
                ownership_gate=exact_dynamic_gradient_gate,
                exact_conditioned_dynamic=True,
                owner_context_id=int(conditioned_view.colmap_id),
            )
            conditioned_volume_radii = conditioned_package.radii[
                conditioned_package.structural_count :
            ].detach()
            del (
                conditioned_package,
                conditioned_base,
                conditioned,
                conditioned_task,
                conditioned_task_fields,
                conditioned_target,
                conditioned_topology_signal,
            )

        if static_spatial_uncertainty_active:
            torch.autograd.backward(
                args.appearance_weight * uncertainty_loss,
                inputs=appearance.uncertainty_parameters(),
            )

        geometry_loss = canonical_loss.new_zeros(())
        geometry_surface_means2d = None
        geometry_surface_radii = None
        geometry_values = {
            "chart": 0.0,
            "plane": 0.0,
            "normal": 0.0,
            "inverse": 0.0,
            "ordinal": 0.0,
            "track": 0.0,
            "track_matched": 0,
            "structure": 0.0,
            "structure_matched": 0,
            "chart_anchor": 0.0,
            "chart_anchor_matched": 0,
            "pointmap_native": 0.0,
            "pointmap_native_pixels": 0,
            "projected_rigid_geometry": 0.0,
            "projected_rigid_supported_pixels": 0,
            "projected_rigid_depth_matched_pixels": 0,
            "plane_pixels": 0,
            "chart_pixels": 0,
            "inverse_pixels": 0,
        }
        if (
            args.geometry_every > 0
            and (step + 1) % args.geometry_every == 0
            and geometry_schedule[step] >= 0
        ):
            geometry_view = views[int(geometry_schedule[step])]
            geometry_task = fields.fields(
                geometry_view.image_name,
                (
                    geometry_view.image_height,
                    geometry_view.image_width,
                ),
                torch.device("cuda"),
            )
            source_fields = geometry.fields(
                geometry_view.image_name,
                device=torch.device("cuda"),
                shape=(
                    geometry_view.image_height,
                    geometry_view.image_width,
                ),
            )
            geometry_loss = canonical_loss.new_zeros(())
            track_observation_loss = canonical_loss.new_zeros(())
            track_observation_audit = {"matched": 0}
            chart_native_loss = canonical_loss.new_zeros(())
            chart_native_audit = {"pixels": 0}
            pointmap_native_loss = canonical_loss.new_zeros(())
            pointmap_native_audit = {"pixels": 0}
            projected_rigid_geometry_loss = canonical_loss.new_zeros(())
            projected_rigid_geometry_audit = {
                "contract": PROJECTED_OCCLUDED_RIGID_GEOMETRY_CONTRACT,
                "supported_pixels": 0,
                "depth_matched_pixels": 0,
                "depth": 0.0,
                "coverage": 0.0,
            }
            has_pointmap_factor = (
                Path(str(geometry_view.image_name)).stem
                in geometry.pointmap_records
            )
            has_projected_rigid_geometry = bool(
                (
                    geometry_task["p_projected_rigid_geometry"] > 0.0
                ).any()
            )
            if (
                source_fields
                or has_pointmap_factor
                or has_projected_rigid_geometry
            ):
                geometry_package = render_hybrid(
                    geometry_view,
                    surface,
                    foliage,
                    background=background,
                    include_dynamic=False,
                    structural_trainable_start=0,
                    volume_opacity_scale=0.0,
                )
                geometry_surface_means2d = (
                    geometry_package.surface_means2d
                )
                geometry_surface_radii = geometry_package.radii[
                    : geometry_package.structural_count
                ]
                if source_fields:
                    geometry_loss, source_geometry_values = _geometry_losses(
                        geometry_package,
                        source_fields,
                        geometry_task["p_rigid"],
                        args,
                        geometry_weight=geometry_task["w_geometry"],
                        plane_task_weight=geometry_task["w_plane"],
                        native_chart_factor_active=(
                            "chart_depth" in source_fields
                        ),
                    )
                    geometry_values.update(source_geometry_values)
                if (
                    geometry_values["chart_pixels"] > 0
                    and geometry_values["inverse_pixels"] <= 0
                    and "chart_depth" not in source_fields
                ):
                    geometry_audit_warnings.add(
                        "chart_owner_with_zero_effective_inverse_depth"
                    )
                if (
                    geometry_values["chart_pixels"] > 0
                    and "chart_depth" not in source_fields
                    and (
                        geometry_values["inverse_pixels"]
                        / geometry_values["chart_pixels"]
                    )
                    < 0.95
                ):
                    geometry_audit_warnings.add(
                        "chart_inverse_depth_coverage_below_95_percent"
                    )
                track_observation_loss, track_observation_audit = (
                    geometry.track_observation_factor(
                        geometry_view.image_name, geometry_package
                    )
                )
                chart_native_loss, chart_native_audit = (
                    geometry.chart_native_factor(
                        geometry_view.image_name,
                        geometry_package,
                        geometry_task["p_rigid"],
                    )
                )
                (
                    pointmap_native_loss,
                    pointmap_native_audit,
                ) = geometry.mast3r_pointmap_native_factor(
                    geometry_view,
                    geometry_package,
                    geometry_task["p_rigid"],
                )
                if has_projected_rigid_geometry:
                    (
                        projected_rigid_geometry_loss,
                        projected_rigid_geometry_audit,
                    ) = _projected_occluded_rigid_geometry_loss(
                        geometry_package, geometry_task
                    )
                del geometry_package
            # Centre priors are a bootstrap stabilizer only.  Once renderer
            # witnesses are released, geometry is supervised exclusively by
            # observation/ray factors and remains valid even if every seed is
            # locally replaced.
            if not anchors_released:
                track_loss, track_audit = geometry.track_factor(surface)
                chart_anchor_loss, chart_anchor_audit = (
                    chart_surface.factor(surface)
                )
            else:
                track_loss = canonical_loss.new_zeros(())
                chart_anchor_loss = canonical_loss.new_zeros(())
                track_audit = {
                    "matched": 0,
                    "disabled_after_renderer_seed_release": True,
                }
                chart_anchor_audit = {
                    "matched": 0,
                    "disabled_after_renderer_seed_release": True,
                }
            if args.structure_weight > 0:
                structure_loss, structure_audit = geometry.structure_factor(
                    surface
                )
            else:
                structure_loss = canonical_loss.new_zeros(())
                structure_audit = {
                    "matched": 0,
                    "blocks": 0,
                    "disabled_invalid_center_ownership": True,
                }
            geometry_values["track"] = float(track_loss.detach())
            geometry_values["track_observation"] = float(
                track_observation_loss.detach()
            )
            geometry_values["track_observation_matched"] = int(
                track_observation_audit["matched"]
            )
            geometry_values["chart_native"] = float(
                chart_native_loss.detach()
            )
            geometry_values["chart_native_pixels"] = int(
                chart_native_audit["pixels"]
            )
            geometry_values["pointmap_native"] = float(
                pointmap_native_loss.detach()
            )
            geometry_values["pointmap_native_pixels"] = int(
                pointmap_native_audit["pixels"]
            )
            geometry_values["projected_rigid_geometry"] = float(
                projected_rigid_geometry_loss.detach()
            )
            geometry_values["projected_rigid_supported_pixels"] = int(
                projected_rigid_geometry_audit["supported_pixels"]
            )
            geometry_values[
                "projected_rigid_depth_matched_pixels"
            ] = int(
                projected_rigid_geometry_audit["depth_matched_pixels"]
            )
            geometry_values["projected_rigid_depth"] = float(
                projected_rigid_geometry_audit["depth"]
            )
            geometry_values["projected_rigid_coverage"] = float(
                projected_rigid_geometry_audit["coverage"]
            )
            geometry_values["track_matched"] = int(
                track_audit["matched"]
            )
            geometry_values["structure"] = float(
                structure_loss.detach()
            )
            geometry_values["structure_matched"] = int(
                structure_audit["matched"]
            )
            geometry_values["chart_anchor"] = float(
                chart_anchor_loss.detach()
            )
            geometry_values["chart_anchor_matched"] = int(
                chart_anchor_audit["matched"]
            )
            geometry_values["chart_uv_edges_sampled"] = int(
                chart_anchor_audit.get("uv_edges_sampled", 0)
            )
            geometry_values["chart_uv_edges_matched"] = int(
                chart_anchor_audit.get("uv_edges_matched", 0)
            )
            geometry_loss = geometry_loss + (
                args.track_weight
                * (track_observation_loss + 0.10 * track_loss)
                + args.geometry_weight * chart_native_loss
                + args.pointmap_weight * pointmap_native_loss
                + args.projected_rigid_depth_weight
                * projected_rigid_geometry_loss
                + args.structure_weight * structure_loss
                + args.chart_anchor_weight * chart_anchor_loss
            )
            phase_floor = (
                1.0
                if phase in {"canonical_bootstrap", "topology"}
                else 0.65
            )
            geometry_loss = phase_floor * geometry_loss
            if geometry_loss.requires_grad:
                geometry_gradient_scale = 1.0
                geometry_xyz_gradient_norm = 0.0
                rgb_xyz_gradient_norm = _gradient_norm(surface._xyz)
                if args.geometry_gradient_ratio > 0:
                    geometry_xyz_gradient = torch.autograd.grad(
                        geometry_loss,
                        surface._xyz,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                    if geometry_xyz_gradient is not None:
                        geometry_xyz_gradient_norm = float(
                            torch.nan_to_num(
                                geometry_xyz_gradient
                            ).norm().detach()
                        )
                        geometry_gradient_scale = (
                            _adaptive_geometry_scale(
                                rgb_xyz_gradient_norm,
                                geometry_xyz_gradient_norm,
                                target_ratio=(
                                    args.geometry_gradient_ratio
                                ),
                            )
                        )
                        geometry_loss = (
                            geometry_loss * geometry_gradient_scale
                        )
                geometry_values["adaptive_gradient_scale"] = float(
                    geometry_gradient_scale
                )
                geometry_values["rgb_xyz_gradient_norm_before"] = float(
                    rgb_xyz_gradient_norm
                )
                geometry_values["geometry_xyz_gradient_norm_before"] = float(
                    geometry_xyz_gradient_norm
                )
                geometry_loss.backward()
                # Metric ray/depth residuals must influence where the surface
                # gains bandwidth.  Previously only RGB/topology-image
                # gradients reached the 2DGS densification accumulator, so a
                # persistently wrong Chart/MASt3R surface could be optimized
                # but never split.  Evidence remains external; only its
                # rendered screen-space gradient is accumulated here.
                if (
                    geometry_surface_means2d is not None
                    and geometry_surface_means2d.grad is not None
                    and geometry_surface_radii is not None
                ):
                    geometry_visible = geometry_surface_radii > 0
                    surface.add_densification_stats(
                        geometry_surface_means2d,
                        geometry_visible,
                    )
                    surface.max_radii2D[
                        geometry_visible
                    ] = torch.maximum(
                        surface.max_radii2D[geometry_visible],
                        geometry_surface_radii[geometry_visible],
                    )
                    geometry_topology_factor_calls += 1
                    geometry_topology_primitive_updates += int(
                        geometry_visible.sum()
                    )
                    geometry_values[
                        "topology_factor_visible_primitives"
                    ] = int(geometry_visible.sum())

        topology_loss = canonical_loss.new_zeros(())
        if (
            (
                _surface_topology_active(step, args)
                or _chart_topology_active(step, args)
            )
            and anchors_released
            and args.topology_every > 0
            and (step + 1) % args.topology_every == 0
        ):
            topology_view = views[int(topology_schedule[step])]
            topology_task = fields.fields(
                topology_view.image_name,
                (
                    topology_view.image_height,
                    topology_view.image_width,
                ),
                torch.device("cuda"),
            )
            topology_target = topology_view.original_image.cuda(
                non_blocking=True
            )
            topology_package = render_hybrid(
                topology_view,
                surface,
                foliage,
                background=background,
                include_dynamic=False,
                volume_opacity_scale=0.0,
                structural_trainable_start=0,
                audit_fields=torch.stack(
                    [
                        topology_task["p_canopy_core"],
                        topology_task["p_rigid"],
                        torch.zeros_like(topology_task["w_topology"]),
                    ]
                ),
            )
            topology_prediction = composite_white_background(
                topology_package.render,
                topology_package.alpha,
                # This stream exists to add rigid surface bandwidth.  A
                # detached sky prevents missing surfels from being explained
                # by a view-dependent background update.
                sky(topology_view).detach(),
            )
            topology_loss = _photo_loss(
                topology_prediction,
                topology_target,
                # ``w_topology`` is intentionally the canopy-volume weight in
                # the task contract; multiplying it by p_rigid is identically
                # zero.  Rigid surface refinement uses the independent plane/
                # rigid-valid weight instead.
                topology_task["w_plane"],
                0.10,
            )
            (0.20 * topology_loss).backward()
            topology_surface_responsibility = (
                topology_package.responsibility[
                    : topology_package.structural_count
                ]
            )
            topology_total = topology_surface_responsibility[
                :, 0
            ].clamp_min(1e-8)
            topology_rigid_fraction = (
                topology_surface_responsibility[:, 2]
                / topology_total
            )
            topology_canopy_fraction = (
                topology_surface_responsibility[:, 1]
                / topology_total
            )
            topology_weight = (
                topology_rigid_fraction
                * (1.0 - topology_canopy_fraction)
            ).clamp(0.0, 1.0)
            topology_weight *= (
                topology_package.radii[
                    : topology_package.structural_count
                ]
                > 0
            ).to(topology_weight)
            topology_visible = topology_weight > 0
            if topology_package.surface_means2d.grad is not None:
                surface.add_densification_stats(
                    topology_package.surface_means2d,
                    topology_weight,
                )
                topology_weighted_radius = topology_package.radii[
                    : topology_package.structural_count
                ] * torch.sqrt(topology_weight)
                surface.max_radii2D[topology_visible] = torch.maximum(
                    surface.max_radii2D[topology_visible],
                    topology_weighted_radius[topology_visible],
                )
            del (
                topology_package,
                topology_prediction,
                topology_target,
            )

        replacement_event = None
        # Surface topology must be stable before per-candidate statistics can
        # retain row identity, but waiting for the named cleanup phase left
        # structural/tree ownership conflicts untouched for most of training.
        # Start as soon as the independent topology stream has ended and keep
        # accumulating the continuous posterior through all later phases.
        replacement_active = (
            foliage_active
            and anchors_released
            and (step + 1) > int(args.densify_until_iter)
        )
        if replacement_active:
            if replacement_stats is None:
                replacement_stats = _replacement_stats(
                    len(surface.get_xyz), surface.get_xyz.device
                )
            if len(replacement_stats["weight"]) != len(surface.get_xyz):
                raise RuntimeError(
                    "Surface topology changed after cleanup started"
                )
            if (
                args.replacement_every > 0
                and (step + 1) % args.replacement_every == 0
            ):
                replacement_view = view
                replacement_task = task
                replacement_target = target
                replacement_temporal_code = None
                replacement_volume_gate = None
                replacement_include_dynamic = False
                if dynamic_active and conditioned_view is not None:
                    replacement_view = conditioned_view
                    replacement_task = fields.fields(
                        replacement_view.image_name,
                        (
                            replacement_view.image_height,
                            replacement_view.image_width,
                        ),
                        torch.device("cuda"),
                    )
                    replacement_target = (
                        replacement_view.original_image.cuda(
                            non_blocking=True
                        )
                    )
                    replacement_temporal_code = (
                        appearance.temporal_code(
                            replacement_view.image_name
                        )
                    )
                    visibility = _dynamic_visibility_gate(
                        foliage,
                        replacement_view,
                        camera_sequence_lookup,
                        camera_frame_lookup,
                    )
                    # Only calibrated exact observations may prove that a
                    # dynamic leaf should replace a structural owner.
                    # Temporal fallback remains useful for rendering, but is
                    # deliberately excluded from irreversible retirement.
                    replacement_volume_gate = torch.where(
                        foliage.dynamic_leaf_mask,
                        (visibility >= 0.999).to(visibility.dtype),
                        torch.ones_like(visibility),
                    )
                    replacement_include_dynamic = True
                replacement_event = {
                    "iteration": step + 1,
                    **_replacement_audit(
                        replacement_view,
                        surface,
                        foliage,
                        background,
                        replacement_task,
                        replacement_target,
                        sky(replacement_view).detach(),
                        replacement_stats,
                        sequence_bits,
                        include_dynamic=(
                            replacement_include_dynamic
                        ),
                        temporal_code=replacement_temporal_code,
                        volume_gate=replacement_volume_gate,
                        maximum_retirement_mass_fraction=(
                            args.surface_retirement_optical_mass_fraction_per_event
                        ),
                    ),
                }
                replacement_events.append(replacement_event)

        gradient_audit = _teacher_gradient_audit(
            surface,
            foliage,
            package,
            surface_means2d_gradient_norm=(
                surface_rgb_means2d_gradient_norm
            ),
        )
        if chart_surface.atlas is not None:
            gradient_audit.update(
                {
                    "chart_atlas_coarse_inverse_depth": _gradient_norm(
                        chart_surface.atlas.coarse_residual
                    ),
                    "chart_atlas_fine_inverse_depth": _gradient_norm(
                        chart_surface.atlas.fine_residual
                    ),
                }
            )
        if (
            step == 0
            and (
                gradient_audit["surface_xyz"] <= 0
                or gradient_audit["surface_feature_dc"] <= 0
            )
        ):
            raise RuntimeError(
                "Structural 2DGS received no xyz/DC gradient while "
                "structural_trainable_start=0"
            )
        _apply_volume_role_gradients(
            foliage, phase, dynamic_active=dynamic_active
        )
        volume_opacity_settle_audit = (
            _apply_static_optical_policy(
                foliage,
                volume_optimizer,
                phase,
                canonical_evidence_opacity_gradient=(
                    canonical_evidence_opacity_gradient
                ),
            )
            if args.reconstruction_target == "static"
            else _apply_volume_opacity_settle_policy(
                foliage,
                volume_optimizer,
                step,
                args,
            )
        )
        # The parameter-space handoff must be applied after Adam and all
        # scale/mass compensation below.  Applying it here changed the logit
        # and cleared Adam state, but ``volume_optimizer.step()`` immediately
        # consumed the already-computed positive opacity gradient and grew
        # the same envelope back in the very same iteration.
        static_mass_handoff_audit = {
            "contract": "reversible_integrated_optical_mass_handoff",
            "changed_rows": 0,
            "removed_mass": 0.0,
            "restored_mass": 0.0,
            "scheduled_after_optimizer_step": bool(
                static_replacement_scheduled
            ),
        }
        surface_spatial_confidence_audit = (
            _apply_surface_spatial_confidence_gradients(surface)
        )
        mature_surface_policy_audit = (
            _apply_mature_surface_gradient_policy(
                surface,
                args.mature_handoff_surface_policy,
                chart_surface=chart_surface,
                residual_start=rigid_surface_residual_start,
            )
        )
        pre_step_optical_area = (
            projected_gaussian_cross_section(foliage.scales.detach())
            if args.integrated_optical_mass_compensation
            and len(foliage)
            else None
        )
        surface.optimizer.step()
        volume_optimizer.step()
        if args.mature_handoff_surface_policy in {"joint", "atlas_residual"}:
            chart_atlas_optimizer.step()
        surface.optimizer.zero_grad(set_to_none=True)
        volume_optimizer.zero_grad(set_to_none=True)
        chart_atlas_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            foliage.quaternions.copy_(
                F.normalize(foliage.quaternions, dim=-1)
            )
            foliage.opacity_logits.clamp_(-10, 2)
            skeleton_opacity_ceiling = foliage.opacity_logits.new_tensor(
                float(args.skeleton_opacity_ceiling)
            ).logit()
            crown_opacity_ceiling = foliage.opacity_logits.new_tensor(
                float(args.canonical_crown_opacity_ceiling)
            ).logit()
            dynamic_opacity_ceiling = foliage.opacity_logits.new_tensor(
                float(args.dynamic_leaf_opacity_ceiling)
            ).logit()
            _masked_clamp_max_(
                foliage.opacity_logits,
                foliage.static_skeleton_mask,
                skeleton_opacity_ceiling,
            )
            _masked_clamp_max_(
                foliage.opacity_logits,
                foliage.canonical_crown_mask,
                crown_opacity_ceiling,
            )
            # Geometry uncertainty controls initialization, ray-factor
            # strength, visibility and topology authority.  It must not be a
            # hard upper bound on optical existence: a broad single-view
            # depth posterior can still contain a very real opaque leaf in
            # its exact owner image.  The former per-row projection pinned
            # weak DAV2 births near alpha=0.02 (versus ~0.09 before the
            # projection), so RGB could never fill the crown even when its
            # residual supplied direct evidence.  Retain only the ordinary
            # role-wide safety bound here; free-space/ray evidence remains a
            # differentiable spatial constraint instead of a representational
            # gate.
            _masked_clamp_max_(
                foliage.opacity_logits,
                foliage.dynamic_leaf_mask,
                dynamic_opacity_ceiling,
            )
            target_integrated_optical_mass = None
            if pre_step_optical_area is not None:
                post_opacity_tau = -torch.log1p(
                    -foliage.opacities.clamp(0.0, 1.0 - 1.0e-6)
                )
                target_integrated_optical_mass = (
                    post_opacity_tau * pre_step_optical_area
                )
            if args.mature_handoff_surface_policy == "joint":
                surface._opacity.clamp_(-12, 4)
            elif args.mature_handoff_surface_policy == "atlas_residual":
                surface._opacity[rigid_surface_residual_start:].clamp_(
                    -12, 4
                )
            if not anchors_released:
                if len(bootstrap_scale_limit) != len(surface.get_xyz):
                    raise RuntimeError(
                        "Surface topology changed before evidence-seed release"
                    )
                protected = surface._protected_flag
                surface._scaling[protected] = torch.minimum(
                    surface._scaling[protected],
                    bootstrap_scale_limit[protected].clamp_min(1e-6).log(),
                )
                bootstrap_opacity_ceiling = surface._opacity.new_tensor(
                    float(args.bootstrap_evidence_opacity_ceiling)
                ).logit()
                _masked_clamp_max_(
                    surface._opacity,
                    protected,
                    bootstrap_opacity_ceiling,
                )
            # Physical metre caps alone are not meaningful across depth.
            # Project the current-view footprint back to a tangent-scale
            # correction so a sparse distant seed cannot become a 100-pixel
            # paint splat while waiting for the next split event.
            surface_limit_rows = _surface_screen_limit_rows(
                surface,
                chart_surface,
                args.mature_handoff_surface_policy,
                residual_start=rigid_surface_residual_start,
                structural_count=package.structural_count,
                device=package.radii.device,
            )
            if len(surface_limit_rows):
                selected_surface_radii = package.radii[
                    surface_limit_rows
                ].float()
                surface_radius_shrink = torch.ones_like(
                    selected_surface_radii
                )
                oversized = (
                    selected_surface_radii
                    > float(args.maximum_surface_radius_pixels)
                )
                if bool(oversized.any()):
                    surface_radius_shrink[oversized] = torch.sqrt(
                        float(args.maximum_surface_radius_pixels)
                        / selected_surface_radii[oversized].clamp_min(1.0)
                    ).clamp(0.50, 1.0)
                surface_screen_space_optical_audit = (
                    _apply_surface_scale_limits_preserve_optical_mass(
                        surface,
                        surface_limit_rows,
                        surface_radius_shrink,
                        maximum_scale=args.maximum_surface_scale,
                    )
                )
            else:
                surface_screen_space_optical_audit = {
                    "contract": "local_tau_times_tangent_area_conservation",
                    "rows": 0,
                    "changed_rows": 0,
                    "target_mass": 0.0,
                    "realized_mass": 0.0,
                    "unrealized_mass": 0.0,
                    "reason": "surface_geometry_frozen",
                }
            volume_limit = torch.full(
                (len(foliage), 1),
                float(args.maximum_volume_scale),
                device=foliage.log_scales.device,
                dtype=foliage.log_scales.dtype,
            )
            volume_limit[foliage.dynamic_leaf_mask] = float(
                args.maximum_dynamic_volume_scale
            )
            volume_limit[foliage.static_skeleton_mask] = min(
                float(args.maximum_volume_scale), 0.08
            )
            foliage.log_scales.copy_(
                torch.minimum(
                    foliage.log_scales,
                    torch.minimum(
                        volume_limit.expand(-1, 3),
                        foliage.scale_ceiling.clamp_min(1e-6),
                    ).log(),
                )
            )
            volume_radii = package.radii[
                package.structural_count :
            ].float()
            if conditioned_volume_radii is not None:
                volume_radii = torch.maximum(
                    volume_radii, conditioned_volume_radii.float()
                )
            if volume_radii.shape != (len(foliage),):
                raise RuntimeError(
                    "Mixed renderer volume radii no longer align with the "
                    "foliage model"
                )
            role_radius_limit = torch.full_like(
                volume_radii, float(args.maximum_volume_radius_pixels)
            )
            role_radius_limit[foliage.static_skeleton_mask] = float(
                args.maximum_skeleton_radius_pixels
            )
            role_radius_limit[foliage.persistent_envelope_mask] = float(
                args.maximum_envelope_radius_pixels
            )
            role_radius_limit[foliage.detail_leaf_mask] = float(
                args.maximum_static_detail_radius_pixels
            )
            oversized_volume = volume_radii > role_radius_limit
            volume_role_radius_audit = {
                "contract": (
                    "role_specific_screen_footprint_with_integrated_optical_"
                    "mass_conservation"
                ),
                "limits_pixels": {
                    "static_skeleton": float(
                        args.maximum_skeleton_radius_pixels
                    ),
                    "persistent_envelope": float(
                        args.maximum_envelope_radius_pixels
                    ),
                    "static_detail": float(
                        args.maximum_static_detail_radius_pixels
                    ),
                },
                "oversized_rows": int(oversized_volume.sum()),
                "oversized_by_role": {
                    "static_skeleton": int(
                        (oversized_volume & foliage.static_skeleton_mask).sum()
                    ),
                    "persistent_envelope": int(
                        (
                            oversized_volume
                            & foliage.persistent_envelope_mask
                        ).sum()
                    ),
                    "static_detail": int(
                        (oversized_volume & foliage.detail_leaf_mask).sum()
                    ),
                },
            }
            if bool(oversized_volume.any()):
                # Apply a gradual screen-space correction. The recorded large
                # radius remains in volume_stats and is therefore prioritized
                # by the next adaptive split; the correction prevents it from
                # becoming a 100-pixel paint splat while waiting for that
                # topology event.
                volume_shrink = torch.sqrt(
                    role_radius_limit[oversized_volume]
                    / volume_radii[oversized_volume].clamp_min(1.0)
                ).clamp(0.50, 1.0)
                foliage.log_scales[oversized_volume] += (
                    volume_shrink.log()[:, None]
                )
            optical_mass_compensation_audit = {
                "enabled": False,
                "contract": "tau_times_world_projected_cross_section",
            }
            if target_integrated_optical_mass is not None:
                optical_mass_compensation_audit = {
                    "enabled": True,
                    "contract": (
                        "opacity_update_changes_mass__all_scale_updates_and_"
                        "clamps_preserve_mass__role_alpha_ceilings_are_final_"
                        "safety_bounds"
                    ),
                    **foliage.restore_integrated_optical_mass(
                        target_integrated_optical_mass
                    ),
                }
                # A peak-alpha safety bound may intentionally discard mass
                # when the requested mass cannot fit a physically bounded
                # footprint. It must never be bypassed by compensation.
                _masked_clamp_max_(
                    foliage.opacity_logits,
                    foliage.static_skeleton_mask,
                    skeleton_opacity_ceiling,
                )
                _masked_clamp_max_(
                    foliage.opacity_logits,
                    foliage.canonical_crown_mask,
                    crown_opacity_ceiling,
                )
                _masked_clamp_max_(
                    foliage.opacity_logits,
                    foliage.dynamic_leaf_mask,
                    dynamic_opacity_ceiling,
                )
            # First fold this iteration's legitimate hit/opacity update into
            # the unretired reference.  Then apply the local transfer after
            # Adam and scale compensation so it cannot be undone by a stale
            # same-step gradient.  Finally synchronize the reference with the
            # new retired fraction, preserving reversibility.
            static_mass_handoff_audit = (
                _finalize_static_ray_local_mass_handoff(
                    foliage,
                    volume_optimizer,
                    scheduled=static_replacement_scheduled,
                    maximum_fraction_per_event=(
                        args.static_replacement_mass_fraction_per_event
                    ),
                )
            )

        topology_event = None
        if (
            _surface_topology_active(step, args)
            or _chart_topology_active(step, args)
        ) and anchors_released:
            if (step + 1) % args.densification_interval == 0:
                before_count = len(surface.get_xyz)
                chart_priority_snapshot = (
                    chart_surface.topology_priority_snapshot(
                        surface,
                        gradient_threshold=args.densify_grad_threshold,
                        radius_target=8.0,
                    )
                )
                chart_topology_active = _chart_topology_active(step, args)
                chart_growth_budget = (
                    int(
                        round(
                            args.maximum_surface_growth_per_event
                            * args.chart_quadtree_growth_fraction
                        )
                    )
                    if chart_topology_active
                    else 0
                )
                ordinary_growth_budget = max(
                    args.maximum_surface_growth_per_event
                    - chart_growth_budget,
                    0,
                )
                if (
                    _surface_topology_active(step, args)
                    and ordinary_growth_budget > 0
                ):
                    surface_event = surface.densify_and_prune_bounded(
                        args.densify_grad_threshold,
                        args.opacity_cull,
                        scene.cameras_extent,
                        64,
                        max_points=surface_budget,
                        max_growth=ordinary_growth_budget,
                        observation_reference_count=max(len(views), 1),
                        mutable_start=(
                            rigid_surface_residual_start
                            if args.mature_handoff_surface_policy
                            == "atlas_residual"
                            else 0
                        ),
                    )
                else:
                    surface_event = {
                        "enabled": False,
                        "reason": (
                            "mature_surface_world_topology_frozen"
                            if not _surface_topology_active(step, args)
                            else "entire_budget_reserved_for_chart_uv"
                        ),
                        "before": len(surface.get_xyz),
                        "after": len(surface.get_xyz),
                        "net_growth": 0,
                    }
                chart_event = chart_surface.adapt_uv_quadtree(
                    surface,
                    maximum_net_growth=(
                        chart_growth_budget
                        if chart_topology_active
                        else 0
                    ),
                    maximum_points=surface_budget,
                    gradient_threshold=args.densify_grad_threshold,
                    radius_target=8.0,
                    maximum_level=args.chart_quadtree_maximum_level,
                    priority_snapshot=chart_priority_snapshot,
                )
                topology_event = {
                    "iteration": step + 1,
                    "surface_before": before_count,
                    "surface_after": len(surface.get_xyz),
                    "surface": surface_event,
                    "chart_uv_quadtree": chart_event,
                    "renderer_evidence_witnesses_released": (
                        anchors_released
                    ),
                }
                if (
                    anchors_released
                    and bool(surface._protected_flag.any())
                ):
                    raise RuntimeError(
                        "Released renderer topology recreated a protected "
                        "primitive; external evidence must not be inherited "
                        "as an immutable Gaussian"
                    )
            if topology_event is not None:
                topology_events.append(topology_event)
        if (
            _volume_topology_active(step, args, phase)
            and foliage_active
            and (step + 1) % args.volume_densify_every == 0
            and len(foliage)
        ):
            # First create a local high-frequency receiver in every observed
            # envelope cell that has none.  This is an exact optical-depth
            # factorization (same centre/covariance/colour), so it cannot
            # create a birth-time hole, fog layer or brightness jump.  It is
            # deliberately scheduled before ordinary detail splits: otherwise
            # the same ~14k represented groups repeatedly consume capacity
            # while the remaining crown stays envelope-only forever.
            receiver_audit = {
                "contract": "inactive_outside_static_detail_stage",
                "selected": 0,
                "materialized": 0,
            }
            if (
                args.reconstruction_target == "static"
                and _static_detail_stage_trainable(phase)
                and args.maximum_static_detail_materializations_per_event > 0
            ):
                receiver_parents, receiver_selection = (
                    _select_missing_static_detail_receiver_parents(
                        foliage,
                        volume_stats,
                        min(
                            int(
                                args.maximum_static_detail_materializations_per_event
                            ),
                            max(int(volume_budget) - len(foliage), 0),
                        ),
                    )
                )
                receiver_event = (
                    foliage.materialize_static_detail_receivers(
                        receiver_parents,
                        optical_mass_fraction=(
                            args.static_detail_materialization_mass_fraction
                        ),
                        birth_iteration=step + 1,
                    )
                )
                receiver_mapping = receiver_event.pop("_new_to_old")
                receiver_start = int(receiver_event.pop("_new_start"))
                changed_parent_rows = receiver_event.pop("_parent_rows")
                if receiver_event["materialized"]:
                    volume_optimizer = _migrate_volume_optimizer(
                        args,
                        foliage,
                        appearance,
                        sky,
                        volume_optimizer,
                        receiver_mapping,
                    )
                    _zero_new_volume_optimizer_rows(
                        volume_optimizer, receiver_start
                    )
                    _zero_volume_opacity_optimizer_rows(
                        volume_optimizer, changed_parent_rows
                    )
                    volume_stats = _extend_volume_stats_for_births(
                        volume_stats, len(foliage)
                    )
                receiver_audit = {
                    **receiver_selection,
                    **receiver_event,
                    "configured_limit": int(
                        args.maximum_static_detail_materializations_per_event
                    ),
                    "optical_mass_fraction": float(
                        args.static_detail_materialization_mass_fraction
                    ),
                    "capacity_contract": (
                        "global_volume_budget_before_ray_birth_and_ordinary_"
                        "split"
                    ),
                }
            # Allocate newly confirmed uncovered rays before ordinary
            # residual/footprint splits.  The old order let existing broad
            # envelope lineages fill the global budget first and then called
            # drain() with zero capacity, even when independent cameras and
            # sequences had already confirmed a missing leaf volume.
            birth_audit = {
                "contract": "static_target_only",
                "born": 0,
                "eligible_cells": 0,
            }
            birth_event = {"appended": 0}
            configured_birth_limit = 0
            effective_birth_limit = 0
            verification_capacity_scale = 1.0
            if args.reconstruction_target == "static":
                current_verification_state = getattr(
                    foliage,
                    "verification_state",
                    torch.full_like(
                        foliage.layer_role, VERIFICATION_VERIFIED
                    ),
                )
                current_verification_debt = float(
                    (
                        current_verification_state
                        == VERIFICATION_UNVERIFIED
                    ).float().mean()
                )
                verification_capacity_scale = (
                    _verification_debt_capacity_scale(
                        current_verification_debt,
                        soft_fraction=float(
                            args.volume_verification_debt_soft_fraction
                        ),
                        hard_fraction=float(
                            args.volume_verification_debt_hard_fraction
                        ),
                        minimum_scale=float(
                            args.volume_verification_debt_minimum_birth_scale
                        ),
                    )
                )
                configured_birth_limit = int(
                    args.maximum_static_ray_births_per_event
                )
                effective_birth_limit = int(
                    np.floor(
                        configured_birth_limit
                        * np.clip(
                            verification_capacity_scale, 0.0, 1.0
                        )
                        + 1.0e-6
                    )
                )
                (
                    birth_centers,
                    birth_colors,
                    birth_support_camera_ids,
                    birth_support_sequence_count,
                    birth_audit,
                ) = static_ray_births.drain(
                    maximum_births=min(
                        effective_birth_limit,
                        max(int(volume_budget) - len(foliage), 0),
                    )
                )
                if len(birth_centers):
                    birth_event = foliage.append_static_ray_births(
                        birth_centers,
                        colors=birth_colors,
                        support_camera_ids=birth_support_camera_ids,
                        support_sequence_count=(
                            birth_support_sequence_count
                        ),
                        birth_iteration=step + 1,
                    )
                    birth_mapping = birth_event.pop("_new_to_old")
                    birth_start = int(birth_event.pop("_new_start"))
                    replacement_rows = birth_event.pop(
                        "_replacement_rows",
                        torch.empty(
                            0, dtype=torch.long, device=foliage.xyz.device
                        ),
                    )
                    replacement_owner_rows = birth_event.pop(
                        "_replacement_owner_rows",
                        torch.empty(
                            0, dtype=torch.long, device=foliage.xyz.device
                        ),
                    )
                    volume_optimizer = _migrate_volume_optimizer(
                        args,
                        foliage,
                        appearance,
                        sky,
                        volume_optimizer,
                        birth_mapping,
                    )
                    _zero_new_volume_optimizer_rows(
                        volume_optimizer, birth_start
                    )
                    all_birth_rows = torch.arange(
                        birth_start,
                        len(foliage),
                        dtype=torch.long,
                        device=foliage.xyz.device,
                    )
                    all_birth_owners = torch.full_like(all_birth_rows, -1)
                    if len(replacement_rows):
                        local_replacement_rows = replacement_rows - birth_start
                        valid_replacement_rows = (
                            (local_replacement_rows >= 0)
                            & (local_replacement_rows < len(all_birth_rows))
                        )
                        all_birth_owners[
                            local_replacement_rows[valid_replacement_rows]
                        ] = replacement_owner_rows[valid_replacement_rows]
                    birth_event["initial_optical_mass_transfer"] = (
                        _initialize_static_ray_birth_mass_handoff(
                            foliage,
                            volume_optimizer,
                            birth_rows=all_birth_rows,
                            owner_rows=all_birth_owners,
                            view_by_camera_id=view_by_camera_id,
                            maximum_fraction_per_event=(
                                args.static_replacement_mass_fraction_per_event
                            ),
                            maximum_additive_fraction_per_event=(
                                args.static_ray_birth_additive_mass_fraction_per_event
                            ),
                        )
                    )
                    volume_stats = _extend_volume_stats_for_births(
                        volume_stats, len(foliage)
                    )
            configured_topology_scale = _volume_topology_ramp_scale(
                step, args
            )
            receiver_growth = int(receiver_audit.get("materialized", 0))
            remaining_event_growth = max(
                int(args.maximum_volume_splits) - receiver_growth, 0
            )
            ordinary_topology_scale = min(
                configured_topology_scale,
                (
                    remaining_event_growth
                    / max(int(args.maximum_volume_splits), 1)
                ),
            )
            event = _adapt_volume(
                args,
                foliage,
                volume_stats,
                volume_budget=volume_budget,
                phase=phase,
                split_capacity_scale=ordinary_topology_scale,
                camera_forward_lookup=camera_forward_lookup,
                view_by_camera_id=view_by_camera_id,
                geometry_evidence=geometry,
                foliage_ray_evidence=foliage_rays,
                current_iteration=step + 1,
            )
            rollback_old_rows = event.pop(
                "_rollback_old_rows",
                torch.empty(
                    0, dtype=torch.long, device=foliage.xyz.device
                ),
            )
            new_to_old = event.pop("_new_to_old", None)
            if new_to_old is not None:
                volume_optimizer = _migrate_volume_optimizer(
                    args,
                    foliage,
                    appearance,
                    sky,
                    volume_optimizer,
                    new_to_old,
                )
                if len(rollback_old_rows):
                    rollback_new_rows = torch.nonzero(
                        torch.isin(new_to_old, rollback_old_rows),
                        as_tuple=False,
                    ).flatten()
                    _zero_volume_optimizer_rows(
                        volume_optimizer, rollback_new_rows
                    )
            if args.reconstruction_target == "static":
                event["static_detail_receiver_materialization"] = (
                    {
                        **receiver_audit,
                        "remaining_ordinary_growth_limit": int(
                            remaining_event_growth
                        ),
                        "ordinary_topology_scale_after_receiver": float(
                            ordinary_topology_scale
                        ),
                    }
                )
                event["ray_driven_birth"] = {
                    **birth_audit,
                    **birth_event,
                    "configured_birth_limit": configured_birth_limit,
                    "effective_birth_limit": effective_birth_limit,
                    "verification_capacity_scale": (
                        verification_capacity_scale
                    ),
                    "capacity_contract": (
                        "cross_sequence_uncovered_birth_before_ordinary_"
                        "split__strict_evidence_birth_has_independent_nonzero_"
                        "debt_floor_and_global_budget_bound"
                    ),
                }
            volume_stats = _volume_stats(foliage)
            if topology_event is None:
                topology_event = {
                    "iteration": step + 1,
                    "volume": event,
                }
                topology_events.append(topology_event)
            else:
                # Surface and volume topology often fire on the same step.
                # The former trace assignment was overwritten here even
                # though both mutations reached the final result list, making
                # per-step causal audits falsely report only volume growth.
                topology_event["volume"] = event
        if (
            _surface_topology_active(step, args)
            and anchors_released
            and args.opacity_reset_interval > 0
            and (step + 1) % args.opacity_reset_interval == 0
        ):
            # Kept only for explicit experimental overrides.  The unified
            # profile defaults this interval to zero because observation
            # factors and local mass-conserving replacement already control
            # topology without periodically erasing all surface alpha.
            surface.reset_opacity(~surface._protected_flag)

        total_loss = (
            canonical_loss.detach()
            + canopy_photo.detach()
            + args.high_frequency_weight
            * static_canopy_high_frequency.detach()
            + args.static_detail_isolated_weight
            * (
                static_detail_isolated_photo.detach()
                + args.high_frequency_weight
                * static_detail_isolated_high_frequency.detach()
            )
            + args.static_skeleton_isolated_weight
            * (
                static_skeleton_isolated_photo.detach()
                + args.high_frequency_weight
                * static_skeleton_isolated_high_frequency.detach()
            )
            + args.static_volume_isolated_weight
            * (
                static_volume_isolated_photo.detach()
                + args.high_frequency_weight
                * static_volume_isolated_high_frequency.detach()
            )
            + args.static_detail_global_cleanup_weight
            * static_detail_global_cleanup.detach()
            + args.persistent_envelope_global_cleanup_weight
            * persistent_envelope_global_cleanup.detach()
            + conditioned_loss.detach()
            + args.counterfactual_transparency_weight
            * conditioned_counterfactual_transparency.detach()
            + geometry_loss.detach()
            + 0.20 * topology_loss.detach()
        )
        if step == 0 or (step + 1) % args.log_every == 0:
            public_phase = _public_phase_name(
                phase,
                args.reconstruction_target,
                args.training_profile,
            )
            ownership_sum = (
                task["p_rigid"]
                + task["p_canopy"]
                + task["p_sky"]
                + task["p_transient"]
            ).clamp(0.0, 1.0)
            ownership_valid = task["p_distortion_valid"] > 0.5
            ownerless = (1.0 - ownership_sum)[ownership_valid]
            render_owner = (
                task["p_rigid"] + task["p_canopy"] + task["p_sky"]
            ).clamp(0.0, 1.0)
            unmodelled_observation = (1.0 - render_owner)[
                ownership_valid
            ]
            task_ownership_audit = {
                "contract": (
                    "semantic_prior_mass_transferred_continuously_by_"
                    "positive_multiview_evidence"
                ),
                "valid_pixels": int(ownership_valid.sum()),
                "ownerless_mass": float(ownerless.sum()),
                "ownerless_fraction": float(ownerless.mean()),
                "unmodelled_observation_mass": float(
                    unmodelled_observation.sum()
                ),
                "unmodelled_observation_fraction": float(
                    unmodelled_observation.mean()
                ),
                "unknown_mass": float(
                    task["p_unknown_ownership"][ownership_valid].sum()
                ),
                "projected_rigid_geometry_mass": float(
                    task["p_projected_rigid_geometry"][
                        ownership_valid
                    ].sum()
                ),
                "projected_rigid_visible_mass": float(
                    task["p_projected_rigid_visible"][
                        ownership_valid
                    ].sum()
                ),
                "direct_rigid_rescue_mass": float(
                    task["p_direct_rigid_rescue"][ownership_valid].sum()
                ),
            }
            row = {
                "iteration": step + 1,
                "phase": public_phase,
                "internal_phase": phase,
                "loss": float(total_loss),
                "task_ownership": task_ownership_audit,
                "canonical_photo": float(photo.detach()),
                "canonical_canopy_photo": float(
                    canopy_photo.detach()
                ),
                "static_canopy_high_frequency": float(
                    static_canopy_high_frequency.detach()
                ),
                "static_detail_isolated": {
                    "contract": STATIC_DETAIL_ISOLATED_CONTRACT,
                    "scheduled": bool(
                        static_detail_isolated_scheduled
                    ),
                    "photo": float(
                        static_detail_isolated_photo.detach()
                    ),
                    "high_frequency": float(
                        static_detail_isolated_high_frequency.detach()
                    ),
                    "every": int(args.static_detail_isolated_every),
                    "weight": float(args.static_detail_isolated_weight),
                    "opacity_gradient_permission": False,
                    # Snapshot before any topology mutation changes the row
                    # count later in this iteration.
                    "qualified_detail_rows": int(
                        static_detail_isolated_qualified_rows
                    ),
                    "appearance_detail_rows": int(
                        static_detail_isolated_appearance_rows
                    ),
                    "view_index": (
                        None
                        if static_detail_isolated_view is None
                        else int(
                            static_detail_schedule[
                                _periodic_schedule_index(
                                    step,
                                    args.static_detail_isolated_every,
                                )
                            ]
                        )
                    ),
                    "image_name": (
                        None
                        if static_detail_isolated_view is None
                        else str(
                            static_detail_isolated_view.image_name
                        )
                    ),
                },
                "static_skeleton_isolated": {
                    "scheduled": bool(
                        static_skeleton_isolated_scheduled
                    ),
                    "photo": float(
                        static_skeleton_isolated_photo.detach()
                    ),
                    "high_frequency": float(
                        static_skeleton_isolated_high_frequency.detach()
                    ),
                    "pixels": int(static_skeleton_isolated_pixels),
                    "every": int(args.static_skeleton_isolated_every),
                    "weight": float(args.static_skeleton_isolated_weight),
                    "image_name": (
                        None
                        if static_skeleton_isolated_view is None
                        else str(static_skeleton_isolated_view.image_name)
                    ),
                },
                "static_volume_isolated": {
                    "scheduled": bool(static_volume_isolated_scheduled),
                    "photo": float(
                        static_volume_isolated_photo.detach()
                    ),
                    "high_frequency": float(
                        static_volume_isolated_high_frequency.detach()
                    ),
                    "pixels": int(static_volume_isolated_pixels),
                    "every": int(args.static_volume_isolated_every),
                    "weight": float(args.static_volume_isolated_weight),
                    "opacity_gradient_permission": False,
                    "gradient_roles": [
                        "verified_exact_owner_static_detail_geometry",
                        "verified_soft_same_sequence_static_detail_sh",
                    ],
                    "qualified_detail_rows": int(
                        static_volume_isolated_qualified_rows
                    ),
                    "appearance_detail_rows": int(
                        static_volume_isolated_appearance_rows
                    ),
                    "image_name": (
                        None
                        if static_volume_isolated_view is None
                        else str(static_volume_isolated_view.image_name)
                    ),
                },
                "static_detail_global_cleanup": {
                    **static_detail_global_cleanup_audit,
                    "scheduled": bool(
                        static_detail_global_cleanup_scheduled
                    ),
                    "loss": float(
                        static_detail_global_cleanup.detach()
                    ),
                    "every": int(
                        args.static_detail_global_cleanup_every
                    ),
                    "weight": float(
                        args.static_detail_global_cleanup_weight
                    ),
                },
                "persistent_envelope_global_cleanup": {
                    **persistent_envelope_global_cleanup_audit,
                    "scheduled": bool(
                        persistent_envelope_global_cleanup_scheduled
                    ),
                    "loss": float(
                        persistent_envelope_global_cleanup.detach()
                    ),
                    "every": int(
                        args.persistent_envelope_global_cleanup_every
                    ),
                    "weight": float(
                        args.persistent_envelope_global_cleanup_weight
                    ),
                },
                "static_detail_canonical_ownership": (
                    static_detail_ownership_audit
                ),
                "rigid_photo": float(rigid_photo.detach()),
                "rigid_high_frequency": float(
                    rigid_high_frequency.detach()
                ),
                "rigid_residual_patch": {
                    "loss": float(rigid_residual_patch.detach()),
                    **rigid_residual_patch_audit,
                },
                "dav2_observation_photo": float(
                    dav2_observation_photo.detach()
                ),
                "dav2_observation_pixels": int(
                    dav2_observation_pixels
                ),
                "surface_learning_rate": {
                    "absolute_iteration": int(surface_lr_iteration),
                    "xyz": float(surface_xyz_learning_rate),
                    "chart_atlas": chart_atlas_lr_audit,
                },
                "volume_learning_rate": volume_learning_rate_audit,
                "dynamic_active": dynamic_active,
                "conditioned_view": (
                    None
                    if conditioned_view is None
                    else str(conditioned_view.image_name)
                ),
                "conditioned_view_has_exact_evidence": (
                    False
                    if conditioned_view is None
                    else int(conditioned_schedule[step])
                    in dynamic_evidence_index_set
                ),
                "conditioned": float(conditioned_loss.detach()),
                "uncertainty": float(uncertainty_loss.detach()),
                "high_frequency": float(
                    high_frequency_loss.detach()
                ),
                "canonical_canopy_confidence": float(
                    static_confidence_mean.detach()
                ),
                "canonical_canopy_sigma": float(
                    static_sigma_mean.detach()
                ),
                "ownership": float(ownership.detach()),
                "surface_canopy_front_conflict": float(
                    surface_in_canopy.detach()
                ),
                "counterfactual_transparency": {
                    "loss": float(counterfactual_transparency.detach()),
                    **counterfactual_transparency_audit,
                },
                "conditioned_ownership": float(
                    conditioned_ownership.detach()
                ),
                "conditioned_counterfactual_transparency": {
                    "loss": float(
                        conditioned_counterfactual_transparency.detach()
                    ),
                    **conditioned_counterfactual_audit,
                },
                "dynamic_observation": {
                    "loss": float(dynamic_observation_loss.detach()),
                    **dynamic_observation_audit,
                },
                "dynamic_visibility": dynamic_gate_audit,
                "occupancy": float(occupancy.detach()),
                "foliage_ray_factor": ray_audit,
                "foliage_ray_consumption": foliage_rays.audit(),
                "geometry": geometry_values,
                "gradient_norms": gradient_audit,
                "parameter_loss_gradient_audit": (
                    parameter_loss_gradient_audit
                ),
                "volume_opacity_settle": volume_opacity_settle_audit,
                "static_local_replacement": static_replacement_audit,
                "static_optical_mass_handoff": static_mass_handoff_audit,
                "static_child_verification": (
                    static_child_verification_audit
                ),
                "integrated_optical_mass_compensation": (
                    optical_mass_compensation_audit
                ),
                "surface_screen_space_optical_mass": (
                    surface_screen_space_optical_audit
                ),
                "volume_role_screen_space": volume_role_radius_audit,
                "surface_spatial_confidence": (
                    surface_spatial_confidence_audit
                ),
                "mature_surface_policy": mature_surface_policy_audit,
                "surface_count": len(surface.get_xyz),
                "surface_track_anchors": int(
                    (surface._track_id >= 0).sum()
                ),
                "surface_chart_anchors": int(
                    (
                        surface._source_type
                        == GaussianModel.SOURCE_CHART_RESIDUAL
                    )
                    .sum()
                ),
                "surface_protected": int(
                    surface._protected_flag.sum()
                ),
                "renderer_evidence_witnesses_released": anchors_released,
                "foliage_count": len(foliage),
                "static_skeleton": int(
                    foliage.static_skeleton_mask.sum()
                ),
                "canonical_crown": int(
                    foliage.canonical_crown_mask.sum()
                ),
                "persistent_envelope": int(
                    foliage.persistent_envelope_mask.sum()
                ),
                "static_leaf": int(foliage.static_leaf_mask.sum()),
                "dynamic_leaf": int(
                    foliage.dynamic_leaf_mask.sum()
                ),
                "topology_event": topology_event,
                "replacement_event": replacement_event,
                "elapsed_sec": time.time() - started,
            }
            with trace.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            progress.set_postfix(
                phase=public_phase,
                loss=f"{float(total_loss):.4f}",
                surface=len(surface.get_xyz),
                volume=len(foliage),
            )
        iteration = step + 1
        retain_checkpoint = iteration in retained_checkpoint_iterations
        graceful_stop_requested = bool(graceful_stop)
        if _checkpoint_due(
            iteration,
            final_iteration=args.iterations,
            checkpoint_every=args.checkpoint_every,
            early_checkpoint_every=args.early_checkpoint_every,
            early_checkpoint_until=args.early_checkpoint_until,
            retained_iterations=retained_checkpoint_iterations,
            graceful_stop_requested=graceful_stop_requested,
        ):
            rolling_checkpoint = (
                output / "hybrid_teacher_checkpoint.pth"
            )
            _save_checkpoint(
                rolling_checkpoint,
                {
                    "protocol": PROTOCOL,
                    "iteration": iteration,
                    "training_profile": args.training_profile,
                    "phase_schedule": TRAINING_PROFILES[
                        args.training_profile
                    ]["phases"],
                    "resolved_phase_schedule": _resolved_phase_schedule(
                        args.training_profile,
                        args.phase_schedule_horizon,
                    ),
                    "schedule_horizon": int(
                        args.phase_schedule_horizon
                    ),
                    "maximum_surface_gaussians": surface_budget,
                    "maximum_surface_growth_per_event": (
                        args.maximum_surface_growth_per_event
                    ),
                    "maximum_volume_gaussians": volume_budget,
                    "maximum_volume_splits_per_event": (
                        args.maximum_volume_splits
                    ),
                    "evidence_hash": evidence_store["evidence_hash"],
                    "schedule_hash": schedule_hash,
                    "training_contract": training_contract,
                    "schedules": {
                        "rgb": rgb_schedule,
                        "conditioned": conditioned_schedule,
                        "geometry": geometry_schedule,
                        "topology": topology_schedule,
                        "static_detail": static_detail_schedule,
                        "static_skeleton": static_skeleton_schedule,
                        "static_volume": static_volume_schedule,
                    },
                    "conditioned_visit_counts": (
                        conditioned_visit_counts.copy()
                    ),
                    "conditioned_visit_provenance": (
                        conditioned_visit_provenance
                    ),
                    "implementation_hashes": implementation_hashes,
                    "runtime_provenance": runtime_provenance,
                    "geometry_evidence_audit": geometry.audit(),
                    "phase": phase,
                    "surface": surface.capture(),
                    "deployment_surface_geometry": (
                        _deployment_surface_geometry(
                            surface, chart_surface
                        )
                    ),
                    "surface_audit": surface_audit,
                    "chart_surface": chart_surface.capture(),
                    "chart_atlas_optimizer": (
                        chart_atlas_optimizer.state_dict()
                    ),
                    "anchors_released": anchors_released,
                    "anchor_release_iteration": anchor_release_iteration,
                    "anchor_release_events": anchor_release_events,
                    "bootstrap_scale_limit": (
                        None
                        if anchors_released
                        else bootstrap_scale_limit.cpu()
                    ),
                    "foliage": foliage.capture(),
                    "foliage_ray_runtime_state": (
                        foliage_rays.capture_runtime_state()
                    ),
                    "static_ray_birth_state": static_ray_births.capture(),
                    "camera_ownership_contract": {
                        "sequence_lookup": camera_sequence_lookup.cpu(),
                        "frame_lookup": camera_frame_lookup.cpu(),
                    },
                    "dynamic_seed_count": dynamic_seed_count,
                    "sky": sky.state_dict(),
                    "appearance": appearance.capture(),
                    "volume_optimizer": volume_optimizer.state_dict(),
                    "volume_stats": volume_stats,
                    "replacement_stats": replacement_stats,
                    "topology_events": topology_events,
                    "replacement_events": replacement_events,
                    "geometry_topology_factor_calls": (
                        geometry_topology_factor_calls
                    ),
                    "geometry_topology_primitive_updates": (
                        geometry_topology_primitive_updates
                    ),
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all(),
                },
            )
            if retain_checkpoint:
                _retain_checkpoint_snapshot(
                    rolling_checkpoint, iteration
                )
            if graceful_stop_requested:
                interrupted_state = {
                    "status": "interrupted_checkpointed",
                    "iteration": int(iteration),
                    "checkpoint": str(rolling_checkpoint),
                    "checkpoint_atomic": True,
                    "resume_required": True,
                    **graceful_stop,
                }
                _write_json_atomic(
                    output / "interrupted_training_state.json",
                    interrupted_state,
                )
                print(
                    "Graceful stop checkpointed at iteration "
                    f"{iteration}: {graceful_stop['signal']}",
                    flush=True,
                )
        del package, canonical, target
        if (
            args.maintenance_every > 0
            and (step + 1) % args.maintenance_every == 0
        ):
            gc.collect()
        # Densification replaces large parameter tensors.  Release only those
        # stale allocator blocks; emptying the CUDA cache every iteration
        # previously serialized the entire pipeline and defeated reuse.
        if topology_event is not None:
            torch.cuda.empty_cache()
        if graceful_stop_requested:
            progress.close()
            # Conventional 128+signal status tells a supervising runner that
            # training did not finish, while the atomic checkpoint and state
            # sidecar prove that it is safe to resume.
            raise SystemExit(
                128 + int(graceful_stop["signal_number"])
            )

    (output / "interrupted_training_state.json").unlink(missing_ok=True)

    # Bootstrap centre-prior samplers stop being authoritative after renderer
    # witnesses are released.  Audit the permanent observation-level factors
    # instead; otherwise a healthy run is incorrectly reported as having
    # starved every track/Chart merely because their temporary XYZ priors were
    # intentionally disabled.
    evidence_coverage = geometry.coverage_audit()
    chart_coverage = chart_surface.audit()["coverage"]
    if not anchors_released:
        if any(
            audit["never_visited"]
            for audit in evidence_coverage["tracks"].values()
        ):
            geometry_audit_warnings.add(
                "bootstrap_track_center_prior_not_fully_visited"
            )
        if chart_coverage["never_visited"]:
            geometry_audit_warnings.add(
                "bootstrap_chart_center_prior_not_fully_visited"
            )
    else:
        if geometry.consumed["track_observation_factor"] <= 0:
            geometry_audit_warnings.add(
                "track_observation_factor_never_consumed"
            )
        if geometry.consumed["chart_native_factor"] <= 0:
            geometry_audit_warnings.add(
                "chart_native_factor_never_consumed"
            )
        if (
            geometry.pointmap_records
            and geometry.consumed[
                "mast3r_pointmap_native_factor"
            ]
            <= 0
        ):
            geometry_audit_warnings.add(
                "mast3r_pointmap_native_factor_never_consumed"
            )
    if (
        args.structure_weight > 0
        and evidence_coverage["structure"] is not None
        and evidence_coverage["structure"]["never_visited"]
    ):
        geometry_audit_warnings.add("structure_evidence_not_fully_visited")
    if args.structure_weight > 0 and geometry.consumed["structure_factor"] <= 0:
        geometry_audit_warnings.add("structure_center_factor_never_consumed")
    pointmap_posterior_audit = geometry.audit()[
        "pointmap_cross_sequence_posterior"
    ]
    if (
        pointmap_posterior_audit["single_sequence_fallback_factor_calls"] > 0
    ):
        geometry_audit_warnings.add(
            "mast3r_pointmap_single_sequence_low_precision_fallback_consumed"
        )
    if (
        geometry.pointmap_records
        and pointmap_posterior_audit["factor_calls"] > 0
        and pointmap_posterior_audit["supported_pixels"] == 0
    ):
        geometry_audit_warnings.add(
            "mast3r_pointmap_no_cross_sequence_supported_pixels"
        )
    # Export is a geometric model, not an optimizer graveyard. Physically
    # remove proven replacements and unprotected transparent descendants so
    # downstream point-cloud tools do not expose retired floaters as vertices.
    with torch.no_grad():
        # Use the same continuous observation-maturity contract as online
        # topology.  A final unconditional alpha threshold otherwise deletes
        # the DAV2 hole witnesses that online culling correctly retained long
        # enough to receive representative camera evidence.
        (
            final_remove,
            final_cull_maturity,
            final_chart_opacity_retirement_blocked,
        ) = (
            surface._ordinary_opacity_prune_mask(
                float(args.opacity_cull),
                observation_reference_count=max(len(views), 1),
            )
        )
        final_remove &= ~surface._protected_flag
        retired_before_compaction = 0
        if replacement_stats is not None:
            retired_before_compaction = int(
                (
                    replacement_stats["retired"]
                    & ~surface._protected_flag
                ).sum()
            )
            final_remove |= (
                replacement_stats["retired"]
                & ~surface._protected_flag
            )
        final_cleanup = {
            "before": len(surface.get_xyz),
            "retired": retired_before_compaction,
            "transparent_unprotected": int(final_remove.sum()),
            "opacity_cull_policy": (
                "source_aware_continuous_observation_maturity__chart_"
                "atlas_requires_explicit_receiver_or_contradiction"
            ),
            "chart_atlas_ordinary_opacity_retirement_blocked": int(
                final_chart_opacity_retirement_blocked.sum()
            ),
            "mean_cull_maturity": (
                float(final_cull_maturity.mean())
                if len(final_cull_maturity)
                else 1.0
            ),
        }
        if bool(final_remove.any()):
            surface.prune_points(final_remove)
        final_cleanup["after"] = len(surface.get_xyz)
        final_cleanup["removed"] = (
            final_cleanup["before"] - final_cleanup["after"]
        )
        # Persist the current atlas geometry into ordinary 2DGS tensors.  The
        # training checkpoint retains the atlas for exact continuation, while
        # point_cloud.ply is now directly consumable by an unmodified native
        # 2DGS renderer/localization stack.
        chart_bake = chart_surface.bake_into_surface(surface)
        final_cleanup["chart_atlas_bake"] = chart_bake
        final_scale = surface.get_scaling.max(dim=1).values
        final_dav2 = (
            surface._source_type
            == GaussianModel.SOURCE_DAV2_RIGID_HOLE
        )
        final_surface_health = {
            "count": len(surface.get_xyz),
            "surviving_mast3r_seed_primitives": int(
                (
                    surface._source_type
                    == GaussianModel.SOURCE_MAST3R_TRACK
                )
                .logical_and(surface._track_id >= 0)
                .sum()
            ),
            "surviving_chart_seed_primitives": int(
                (
                    surface._source_type
                    == GaussianModel.SOURCE_CHART_RESIDUAL
                )
                .sum()
            ),
            "surviving_dav2_hole_completion_primitives": int(
                (
                    surface._source_type
                    == GaussianModel.SOURCE_DAV2_RIGID_HOLE
                ).sum()
            ),
            "free_renderer_residuals": int(
                (
                    surface._source_type
                    == GaussianModel.SOURCE_FREE_RESIDUAL
                ).sum()
            ),
            "source_role_histogram": {
                str(int(value)): int(
                    (surface._source_type == value).sum()
                )
                for value in surface._source_type.unique().cpu()
            },
            "dav2_lineage_observation_mass": {
                "mean": (
                    float(surface._observation_mass[final_dav2].mean())
                    if bool(final_dav2.any())
                    else 0.0
                ),
                "median": (
                    float(surface._observation_mass[final_dav2].median())
                    if bool(final_dav2.any())
                    else 0.0
                ),
                "maximum": (
                    float(surface._observation_mass[final_dav2].max())
                    if bool(final_dav2.any())
                    else 0.0
                ),
                "persistent_per_primitive": True,
            },
            "external_mast3r_evidence_nodes": int(
                sum(
                    len(archive["track_id"])
                    for archive in geometry.track_archives.values()
                )
            ),
            "external_chart_evidence_nodes": int(
                chart_surface.audit()["anchor_count"]
            ),
            "protected": int(surface._protected_flag.sum()),
            "unprotected": int((~surface._protected_flag).sum()),
            "maximum_scale": (
                float(final_scale.max()) if len(final_scale) else 0.0
            ),
            "scale_over_half_meter": int(
                (final_scale > 0.5).sum()
            ),
            "minimum_opacity": (
                float(surface.get_opacity.min())
                if len(surface.get_xyz)
                else 0.0
            ),
        }

    teacher_state = output / "hybrid_teacher_state.pth"
    torch.save(
        {
            "protocol": PROTOCOL,
            "iteration": args.iterations,
            "training_profile": args.training_profile,
            "phase_schedule": TRAINING_PROFILES[
                args.training_profile
            ]["phases"],
            "resolved_phase_schedule": _resolved_phase_schedule(
                args.training_profile, args.phase_schedule_horizon
            ),
            "schedule_horizon": int(args.phase_schedule_horizon),
            "maximum_surface_gaussians": surface_budget,
            "maximum_surface_growth_per_event": (
                args.maximum_surface_growth_per_event
            ),
            "maximum_volume_gaussians": volume_budget,
            "maximum_volume_splits_per_event": (
                args.maximum_volume_splits
            ),
            "evidence_hash": evidence_store["evidence_hash"],
            "training_contract": training_contract,
            "schedule_hash": schedule_hash,
            "schedules": {
                "rgb": rgb_schedule,
                "conditioned": conditioned_schedule,
                "geometry": geometry_schedule,
                "topology": topology_schedule,
                "static_detail": static_detail_schedule,
                "static_skeleton": static_skeleton_schedule,
                "static_volume": static_volume_schedule,
            },
            "conditioned_visit_counts": (
                conditioned_visit_counts.copy()
            ),
            "conditioned_visit_provenance": (
                conditioned_visit_provenance
            ),
            "implementation_hashes": implementation_hashes,
            "runtime_provenance": runtime_provenance,
            "geometry_evidence_audit": geometry.audit(),
            "surface": surface.capture(),
            "surface_audit": surface_audit,
            "surface_warmstart": surface_warmstart,
            "chart_surface": chart_surface.capture(),
            "foliage": foliage.capture(),
            "foliage_ray_evidence_audit": foliage_rays.audit(),
            "static_ray_birth_state": static_ray_births.capture(),
            "camera_ownership_contract": {
                "sequence_lookup": camera_sequence_lookup.cpu(),
                "frame_lookup": camera_frame_lookup.cpu(),
            },
            "dynamic_seed_count": dynamic_seed_count,
            "sky": sky.state_dict(),
            "sky_degree": sky.degree,
            "appearance": appearance.capture(),
            "topology_events": topology_events,
            "replacement_events": replacement_events,
            "anchor_release_iteration": anchor_release_iteration,
            "anchor_release_events": anchor_release_events,
            "anchors_released": anchors_released,
            "final_cleanup": final_cleanup,
            "final_surface_health": final_surface_health,
            "geometry_topology_factor_calls": (
                geometry_topology_factor_calls
            ),
            "geometry_topology_primitive_updates": (
                geometry_topology_primitive_updates
            ),
        },
        teacher_state,
    )
    scene.save(args.iterations)
    rigid_surface_handoff = None
    if args.training_profile == "hybrid_rigid_stage1":
        chart_surface_state = output / "chart_surface_state.pth"
        torch.save(chart_surface.capture(), chart_surface_state)
        geometry_audit_for_handoff = geometry.audit()
        geometry_audit_for_handoff.update(
            {
                "geometry_topology_factor_calls": int(
                    geometry_topology_factor_calls
                ),
                "geometry_topology_primitive_updates": int(
                    geometry_topology_primitive_updates
                ),
            }
        )
        rigid_surface_handoff = _write_rigid_stage_surface_handoff(
            output=output,
            iteration=args.iterations,
            surface_iteration=_continued_surface_iteration(
                args.iterations, surface_learning_rate_offset
            ),
            surface_point_count=len(surface.get_xyz),
            initialization=initialization,
            initialization_directory=args.initialization,
            evidence_store=evidence_store,
            camera_runtime_contract=camera_runtime_contract,
            fixed_camera_validation=fixed_camera_validation,
            geometry_audit=geometry_audit_for_handoff,
            surface_optimizer=training_contract["surface_optimizer"],
            chart_surface_state=chart_surface_state,
        )
    eval_indices = [
        int(value)
        for value in args.eval_indices.split(",")
        if value.strip()
    ]
    evaluation = _evaluate(
        views,
        eval_indices,
        surface,
        foliage,
        sky,
        appearance,
        background,
        fields,
        camera_sequence_lookup,
        camera_frame_lookup,
        output / "visualization",
        canonical_canopy_enabled=(
            args.training_profile != "hybrid_rigid_stage1"
        ),
        conditioned_enabled=(
            args.training_profile != "hybrid_rigid_stage1"
            and args.reconstruction_target
            == "sequence_conditioned_legacy"
        ),
        training_optical_replacement_policy=(
            "disabled"
            if args.reconstruction_target == "static"
            else args.optical_replacement_policy
        ),
        deployment_optical_replacement_policy=(
            deployment_optical_replacement_policy
        ),
        deployment_optical_responsibility_prior=(
            deployment_optical_responsibility_prior
        ),
    )
    scene.close()
    result = {
        "protocol": PROTOCOL,
        "iterations": args.iterations,
        "schedule_horizon": int(args.phase_schedule_horizon),
        "training_profile": args.training_profile,
        "phase_schedule": TRAINING_PROFILES[
            args.training_profile
        ]["phases"],
        "resolved_phase_schedule": _resolved_phase_schedule(
            args.training_profile, args.phase_schedule_horizon
        ),
        "maximum_surface_gaussians": surface_budget,
        "maximum_surface_growth_per_event": (
            args.maximum_surface_growth_per_event
        ),
        "maximum_volume_gaussians": volume_budget,
        "maximum_volume_splits_per_event": args.maximum_volume_splits,
        "evidence_hash": evidence_store["evidence_hash"],
        "training_contract": training_contract,
        "teacher_state": str(teacher_state),
        "teacher_state_sha256": _file_sha256(teacher_state),
        "authoritative_final_model": "native_hybrid_teacher",
        "deployment_render_mode": "canonical_static",
        "standard_student_trained": False,
        "surface_ply": str(
            output
            / "point_cloud"
            / f"iteration_{args.iterations}"
            / "point_cloud.ply"
        ),
        "surface_initialization": surface_audit,
        "surface_warmstart": surface_warmstart,
        "rigid_surface_handoff": (
            str(rigid_surface_handoff)
            if rigid_surface_handoff is not None
            else None
        ),
        "surface_final_count": len(surface.get_xyz),
        "foliage_final_count": len(foliage),
        "layer_counts": {
            "static_skeleton": int(
                foliage.static_skeleton_mask.sum()
            ),
            "canonical_crown": int(
                foliage.canonical_crown_mask.sum()
            ),
            "persistent_envelope": int(
                foliage.persistent_envelope_mask.sum()
            ),
            "static_leaf": int(foliage.static_leaf_mask.sum()),
            "dynamic_leaf": int(foliage.dynamic_leaf_mask.sum()),
        },
        "geometry_evidence": {
            **geometry.audit(),
            "geometry_topology_factor_calls": int(
                geometry_topology_factor_calls
            ),
            "geometry_topology_primitive_updates": int(
                geometry_topology_primitive_updates
            ),
            "structure_center_factor": {
                "weight": float(args.structure_weight),
                "status": (
                    "enabled_legacy_center_factor"
                    if args.structure_weight > 0
                    else "disabled_invalid_center_ownership"
                ),
                "replacement_factors": [
                    "chart_native_factor",
                    "chart_uv_atlas_factor",
                    "track_observation_factor",
                ],
            },
        },
        "chart_surface": chart_surface.audit(),
        "representation_audit_warnings": representation_audit_warnings,
        "geometry_audit_warnings": sorted(geometry_audit_warnings),
        "implementation_hashes": implementation_hashes,
        "runtime_provenance": runtime_provenance,
        "topology_events": topology_events,
        "replacement_events": replacement_events,
        "anchor_release_iteration": anchor_release_iteration,
        "anchor_release_events": anchor_release_events,
        "anchors_released": anchors_released,
        "final_cleanup": final_cleanup,
        "final_surface_health": final_surface_health,
        "surface_spatial_confidence": surface_spatial_confidence_audit,
        "foliage_ray_evidence_audit": foliage_rays.audit(),
        "volume_opacity_settle": volume_opacity_settle_audit,
        "static_local_replacement": static_replacement_audit,
        "static_optical_mass_handoff": static_mass_handoff_audit,
        "static_child_verification": static_child_verification_audit,
        "targeted_evaluation": evaluation,
        "evaluation_contract": {
            "deployment_mode": "canonical",
            "conditioned_role": (
                "absent"
                if args.reconstruction_target == "static"
                else "database_fit_training_diagnostic_only"
            ),
            "valid_render_modes": (
                ["canonical"]
                if (
                    args.training_profile == "hybrid_rigid_stage1"
                    or args.reconstruction_target == "static"
                )
                else ["canonical", "conditioned"]
            ),
            "conditioned_valid": (
                args.training_profile != "hybrid_rigid_stage1"
                and args.reconstruction_target
                == "sequence_conditioned_legacy"
            ),
            "canopy_quality_valid": (
                args.training_profile != "hybrid_rigid_stage1"
            ),
        },
        "historical_parent_ply_used": False,
        "colmap_points_or_tracks_used": bool(
            evidence_store.get("colmap_points_or_tracks_used", False)
        ),
        "sfm_track_usage_mode": evidence_store.get(
            "sfm_track_usage_mode", "disabled"
        ),
        "all_real_rgb_from_iteration_one": True,
        "canopy_surface_topology_gradient": False,
        "canopy_volume_topology_gradient": True,
        "elapsed_sec": time.time() - started,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    renderer_manifest = {
        "version": "native-hybrid-teacher-renderer-manifest-v2",
        "protocol": PROTOCOL,
        "teacher_state": str(teacher_state),
        "teacher_state_sha256": result["teacher_state_sha256"],
        "surface_ply": result["surface_ply"],
        "evidence_hash": evidence_store["evidence_hash"],
        "canonical_render": {
            "include_dynamic": False,
            "appearance_conditioning": False,
            "include_canonical_volume": (
                args.training_profile != "hybrid_rigid_stage1"
            ),
            "deployment_authoritative": True,
            "uses_temporal_or_image_conditioning": False,
            "optical_replacement_policy": (
                deployment_optical_replacement_policy
            ),
            "optical_responsibility_prior": (
                deployment_optical_responsibility_prior
            ),
            "native_passes": {
                STATIC_RAY_NORMALIZED_OPTICAL_POLICY: 2,
                STATIC_RAY_SURFACE_EVIDENCE_POLICY: 3,
            }.get(deployment_optical_replacement_policy, 1),
            "training_render_policy": (
                "disabled"
                if args.reconstruction_target == "static"
                else args.optical_replacement_policy
            ),
            "diagnostic_renderer_policy": args.optical_replacement_policy,
            "persistent_handoff_is_parameter_lifecycle": True,
        },
        "conditioned_render": {
            "available": (
                args.training_profile != "hybrid_rigid_stage1"
                and args.reconstruction_target
                == "sequence_conditioned_legacy"
            ),
            "include_dynamic": (
                args.training_profile != "hybrid_rigid_stage1"
                and args.reconstruction_target
                == "sequence_conditioned_legacy"
            ),
            "temporal_code": (
                "known database image name"
                if args.reconstruction_target
                == "sequence_conditioned_legacy"
                else None
            ),
            "appearance_conditioning": bool(
                args.reconstruction_target
                == "sequence_conditioned_legacy"
            ),
            "deployment_authoritative": False,
            "role": (
                "database_fit_training_diagnostic_only"
                if args.reconstruction_target
                == "sequence_conditioned_legacy"
                else "absent_single_static_map"
            ),
        },
        "mixed_kernel": (
            "native perspective-correct 2D surfel and 3D EWA in one "
            "tile list and one depth/alpha compositing order"
        ),
        "implementation_hashes": implementation_hashes,
        "downstream_contract": (
            "load the state with outdoor.hybrid_teacher_api; do not treat "
            "surface_ply alone as the complete scene"
        ),
    }
    (output / "renderer_manifest.json").write_text(
        json.dumps(renderer_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
