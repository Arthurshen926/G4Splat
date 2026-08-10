"""Small public loading/rendering API for the authoritative mixed Teacher."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SURFEL_ROOT = REPO_ROOT / "2d-gaussian-splatting"
sys.path[:0] = [str(REPO_ROOT), str(SURFEL_ROOT)]

from outdoor.appearance_uncertainty import OutdoorAppearanceUncertainty  # noqa: E402
from outdoor.evidence_store import sha256_file  # noqa: E402
from outdoor.directional_sky import (  # noqa: E402
    CanonicalDirectionalSky,
    composite_white_background,
)
from outdoor.hybrid_gaussian_renderer import (  # noqa: E402
    HybridRenderOutput,
    VolumetricFoliageModel,
    dynamic_visibility_gate,
    render_hybrid,
)
from scene import GaussianModel  # noqa: E402


SUPPORTED_TEACHER_PROTOCOLS = {
    "cambridge_native_hybrid_teacher_v2",
    "cambridge_native_hybrid_teacher_v3_causal_repair",
    "cambridge_native_hybrid_teacher_v4_causal_repair",
    "cambridge_native_hybrid_teacher_v5_contract_closed",
    "cambridge_native_hybrid_teacher_v6_independent_observation_contract",
    "cambridge_native_hybrid_teacher_v7_candidate_ray_posterior_contract",
    "cambridge_native_hybrid_teacher_v8_global_ray_transmittance_contract",
    "cambridge_native_hybrid_teacher_v9_optical_mass_handoff_contract",
    "cambridge_native_hybrid_teacher_v10_dense_ray_bandwidth_contract",
    "cambridge_native_hybrid_teacher_v11_demand_limited_volume_topology",
    "cambridge_native_hybrid_teacher_v12_owner_exclusive_dynamic_gradients",
    "cambridge_native_hybrid_teacher_v13_soft_temporal_dynamic_gradients",
    "cambridge_native_hybrid_teacher_v14_parameter_family_ownership",
    "cambridge_native_hybrid_teacher_v15_retirement_only_fallback_opacity",
    "cambridge_native_hybrid_teacher_v16_base_residual_dynamic_ownership",
    "cambridge_native_hybrid_teacher_v17_sequence_graph_continuous_retirement",
    "cambridge_native_hybrid_teacher_v18_optical_mass_dynamic_replacement",
    "cambridge_native_hybrid_teacher_v19_branch_isolated_rgb_ownership",
    "cambridge_native_hybrid_teacher_v20_mature_surface_continuation",
    "cambridge_native_hybrid_teacher_v22_conditioned_owner_isolation",
    "cambridge_native_hybrid_teacher_v23_evidence_adaptive_role_capacity",
    "cambridge_native_hybrid_teacher_v24_population_normalized_role_capacity",
    "cambridge_native_hybrid_teacher_v25_schedule_preserving_surface_handoff",
    "cambridge_native_hybrid_teacher_v26_absolute_schedule_independent_posterior",
    "cambridge_native_hybrid_teacher_v27_pixel_and_primitive_owned_conditioning",
    "cambridge_native_hybrid_teacher_v28_local_temporal_mass_conservation",
    "cambridge_native_hybrid_teacher_v29_context_adaptive_volume_topology",
    "cambridge_native_hybrid_teacher_v30_exact_dynamic_ray_likelihood",
    "cambridge_native_hybrid_teacher_v31_lineage_complete_optical_schedule",
    "cambridge_native_hybrid_teacher_v32_preserved_handoff_topology_contract",
    "cambridge_native_hybrid_teacher_v33_exact_ray_camera_plane_topology",
    "cambridge_native_hybrid_teacher_v34_exact_owner_optical_mass_calibration",
    "cambridge_native_hybrid_teacher_v35_atomic_volume_replace_split",
    "cambridge_native_hybrid_teacher_v36_counterfactual_transparency_complete_ray_epoch",
    "cambridge_native_hybrid_teacher_v37_exact_per_camera_ray_epoch_topology_settle",
    "cambridge_native_hybrid_teacher_v38_resume_aware_ray_epoch_topology_settle",
    "cambridge_native_hybrid_teacher_v39_conditioned_topology_settle",
    "cambridge_native_hybrid_teacher_v40_projected_optical_footprint",
    "cambridge_native_hybrid_teacher_v41_blue_noise_static_optical_handoff",
    "cambridge_native_hybrid_teacher_v42_blue_noise_static_optical_handoff_"
    "owner_color_refresh",
    "cambridge_native_hybrid_teacher_v43_single_static_map_local_mass_"
    "conservation",
    "cambridge_native_hybrid_teacher_v44_static_staged_coverage_topology_"
    "optical_audit",
    "cambridge_native_hybrid_teacher_v45_static_canonical_viewset_"
    "multimode_optical_audit",
    "cambridge_native_hybrid_teacher_v46_static_detail_isolated_"
    "ownership_topology",
    "cambridge_native_hybrid_teacher_v47_static_staged_detail_"
    "ownership_topology",
    "cambridge_native_hybrid_teacher_v48_static_scene_snapshot_"
    "detail_ownership",
    "cambridge_native_hybrid_teacher_v49_static_scene_snapshot_"
    "canonical_detail_ownership",
    "cambridge_native_hybrid_teacher_v50_static_scene_snapshot_"
    "canonical_detail_schedule",
    "cambridge_native_hybrid_teacher_v51_static_scene_snapshot_"
    "visible_canonical_detail_schedule",
    "cambridge_native_hybrid_teacher_v52_static_ray_local_optical_handoff",
    "cambridge_native_hybrid_teacher_v53_posterior_deployment_contract_closed",
    "cambridge_native_hybrid_teacher_v54_static_ray_ownership_prefit_closed",
    "cambridge_native_hybrid_teacher_v55_static_ray_birth_capacity_closed",
    "cambridge_native_hybrid_teacher_v56_static_birth_camera_union_closed",
    "cambridge_native_hybrid_teacher_v57_static_ray_prefit_closed",
    "cambridge_native_hybrid_teacher_v58_static_sh_ownership_decoupled",
    "cambridge_native_hybrid_teacher_v59_static_rgb_appearance_coverage_closed",
    "cambridge_native_hybrid_teacher_v60_ray_local_optical_lifecycle_closed",
    "cambridge_native_hybrid_teacher_v61_static_role_atlas_residual",
    "cambridge_native_hybrid_teacher_v64_static_role_evidence_mature_"
    "replace_split_surface",
    "cambridge_native_hybrid_teacher_v65_static_residual_topology_"
    "and_weak_rigid_completion",
    "cambridge_native_hybrid_teacher_v66_all_camera_role_posterior_"
    "mask_soft_arbitration_and_surface_optical_mass",
    "cambridge_native_hybrid_teacher_v67_all_camera_role_posterior_"
    "chart_atlas_replace_only_retirement",
    "cambridge_native_hybrid_teacher_v68_all_camera_role_posterior_"
    "residual_conditioned_screen_topology",
    "cambridge_native_hybrid_teacher_v69_local_negative_permission_"
    "topology_stable_optical_polish",
    "cambridge_native_hybrid_teacher_v70_persistent_detail_"
    "lineage_debt_safe_local_handoff",
    "cambridge_native_hybrid_teacher_v71_static_optical_mass_"
    "ownership_closed",
    "cambridge_native_hybrid_teacher_v72_evidence_continuous_"
    "static_detail_mass",
    "cambridge_native_hybrid_teacher_v73_topology_safe_"
    "static_detail_audit",
    "cambridge_native_hybrid_teacher_v74_single_view_static_"
    "occupancy_lifecycle",
    "cambridge_native_hybrid_teacher_v75_single_view_birth_"
    "deferred_retirement",
    "unified_outdoor_mixed_teacher_v1",
}


STATIC_RAY_NORMALIZED_OPTICAL_POLICY = "ray_normalized_two_pass"
STATIC_RAY_SURFACE_EVIDENCE_POLICY = (
    "ray_normalized_surface_evidence_three_pass"
)
STATIC_DEPLOYMENT_OPTICAL_PRIOR = 0.25
SUPPORTED_OPTICAL_REPLACEMENT_POLICIES = {
    "view_depth_local",
    "group_projected",
    "disabled",
    STATIC_RAY_NORMALIZED_OPTICAL_POLICY,
    STATIC_RAY_SURFACE_EVIDENCE_POLICY,
}


def resolve_deployment_optical_contract(
    state: dict,
    *,
    requested_policy: str | None = None,
    requested_prior: float | None = None,
) -> dict[str, object]:
    """Resolve one immutable render contract for API and formal evaluation.

    The optimizer's single-pass envelope replacement rule is not a deployment
    compositor.  New checkpoints persist both contracts separately.  Static
    checkpoints written before that field existed use the retained formal
    two-pass policy rather than silently reverting to the optimizer default.
    An explicit caller override remains possible and is labelled as such.
    """
    training_contract = state.get("training_contract", {})
    reconstruction_target = training_contract.get("reconstruction_target")
    deployment = training_contract.get("deployment_static_contract", {})
    recorded_policy = deployment.get(
        "optical_replacement_policy",
        deployment.get("optical_compositing_policy"),
    )
    recorded_prior = deployment.get("optical_responsibility_prior")
    if recorded_policy is None:
        if reconstruction_target == "static":
            recorded_policy = STATIC_RAY_NORMALIZED_OPTICAL_POLICY
            recorded_prior = STATIC_DEPLOYMENT_OPTICAL_PRIOR
            source = "legacy_static_safe_default"
        else:
            recorded_policy = training_contract.get(
                "optical_replacement_policy", "view_depth_local"
            )
            recorded_prior = 0.0
            source = "legacy_training_policy_fallback"
    else:
        source = "checkpoint_deployment_contract"
    explicit_override = requested_policy is not None
    policy = recorded_policy if requested_policy is None else requested_policy
    if requested_prior is None:
        prior = (
            float(recorded_prior or 0.0)
            if not explicit_override
            else 0.0
        )
    else:
        prior = float(requested_prior)
        explicit_override = True
    policy = str(policy)
    if policy not in SUPPORTED_OPTICAL_REPLACEMENT_POLICIES:
        raise ValueError(f"Unsupported optical replacement policy {policy!r}")
    ray_normalized = policy in {
        STATIC_RAY_NORMALIZED_OPTICAL_POLICY,
        STATIC_RAY_SURFACE_EVIDENCE_POLICY,
    }
    if ray_normalized and reconstruction_target != "static":
        raise ValueError(
            f"{policy} requires a static reconstruction checkpoint"
        )
    if prior < 0:
        raise ValueError("optical_responsibility_prior must be non-negative")
    if prior != 0.0 and not ray_normalized:
        raise ValueError(
            "optical_responsibility_prior is valid only for a "
            "ray-normalized static compositor"
        )
    return {
        "policy": policy,
        "optical_responsibility_prior": prior,
        "source": "explicit_override" if explicit_override else source,
        "checkpoint_policy": str(recorded_policy),
        "checkpoint_optical_responsibility_prior": float(
            recorded_prior or 0.0
        ),
        "explicit_override": bool(explicit_override),
    }


def _static_ray_normalized_optical_mixture(
    envelope: HybridRenderOutput,
    detail: HybridRenderOutput,
    *,
    epsilon: float = 1.0e-6,
    symmetric_optical_prior: float = 0.0,
) -> tuple[HybridRenderOutput, torch.Tensor]:
    """Choose between two static canopy explanations per image ray.

    Envelope and detail Gaussians describe the same static crown rather than
    two independent translucent media.  Adding their alpha in one pass
    therefore counts extinction twice.  Each input here is still an ordinary
    native mixed 2DGS/3DGS render (including the static trunk/skeleton); the
    optical-depth ratio is the posterior responsibility of the detail
    explanation at each pixel.

    This intentionally has no learned gate, semantic mask, view id or time
    code.  A split of one explanation into more Gaussians leaves its summed
    optical depth, and hence the responsibility, unchanged.
    """
    envelope_alpha = envelope.volume_alpha.clamp(0.0, 1.0 - epsilon)
    detail_alpha = detail.volume_alpha.clamp(0.0, 1.0 - epsilon)
    envelope_tau = -torch.log1p(-envelope_alpha)
    detail_tau = -torch.log1p(-detail_alpha)
    total_tau = envelope_tau + detail_tau
    if symmetric_optical_prior < 0:
        raise ValueError("symmetric_optical_prior must be non-negative")
    raw_detail_weight = torch.where(
        total_tau > epsilon,
        detail_tau / total_tau.clamp_min(epsilon),
        torch.full_like(total_tau, 0.5),
    )
    # Peak alpha is an extinction estimate, not a calibrated model posterior.
    # Where both explanations are present, a symmetric Beta-style optical
    # pseudo-count prevents either branch from becoming certain merely by
    # carrying more split descendants.  A genuinely single-owner ray remains
    # exact rather than being faded by the prior.
    both_present = (envelope_tau > epsilon) & (detail_tau > epsilon)
    if symmetric_optical_prior > 0:
        prior = float(symmetric_optical_prior)
        regularized_weight = (detail_tau + prior) / (
            total_tau + 2.0 * prior
        )
        detail_weight = torch.where(
            both_present, regularized_weight, raw_detail_weight
        )
    else:
        detail_weight = raw_detail_weight
    detail_weight = detail_weight.detach()

    def pixel_mixture(
        envelope_value: torch.Tensor | None,
        detail_value: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if envelope_value is None or detail_value is None:
            return envelope_value if detail_value is None else detail_value
        if envelope_value.shape[-2:] != detail_weight.shape[-2:]:
            raise ValueError("Static optical mixture received non-image fields")
        return torch.lerp(envelope_value, detail_value, detail_weight)

    def primitive_union(
        envelope_value: torch.Tensor | None,
        detail_value: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if envelope_value is None or detail_value is None:
            return envelope_value if detail_value is None else detail_value
        if envelope_value.shape != detail_value.shape:
            raise ValueError("Static optical passes no longer align by primitive")
        return torch.maximum(envelope_value, detail_value)

    mixed = HybridRenderOutput(
        render=pixel_mixture(envelope.render, detail.render),
        alpha=pixel_mixture(envelope.alpha, detail.alpha),
        depth=pixel_mixture(envelope.depth, detail.depth),
        normal_world=pixel_mixture(
            envelope.normal_world, detail.normal_world
        ),
        median_depth=pixel_mixture(
            envelope.median_depth, detail.median_depth
        ),
        distortion=pixel_mixture(
            envelope.distortion, detail.distortion
        ),
        radii=primitive_union(envelope.radii, detail.radii),
        # Screen-space gradients from two independent raster passes cannot be
        # represented by one tensor. Training therefore keeps its single
        # native image-formation pass for optimization/topology, while a
        # scheduled no-grad two-pass audit uses these exact alpha/depth fields
        # to drive persistent integrated-mass hand-off.
        means2d=None,
        structural_count=envelope.structural_count,
        responsibility=primitive_union(
            envelope.responsibility, detail.responsibility
        ),
        gate_responsibility=primitive_union(
            envelope.gate_responsibility, detail.gate_responsibility
        ),
        surface_alpha=pixel_mixture(
            envelope.surface_alpha, detail.surface_alpha
        ),
        volume_alpha=pixel_mixture(
            envelope.volume_alpha, detail.volume_alpha
        ),
        surface_depth=pixel_mixture(
            envelope.surface_depth, detail.surface_depth
        ),
        volume_depth=pixel_mixture(
            envelope.volume_depth, detail.volume_depth
        ),
        surface_means2d=None,
        volume_means2d=None,
        volume_replacement=primitive_union(
            envelope.volume_replacement, detail.volume_replacement
        ),
        volume_replacement_candidate=primitive_union(
            envelope.volume_replacement_candidate,
            detail.volume_replacement_candidate,
        ),
    )
    return mixed, detail_weight


def _static_surface_evidence_mixture(
    canopy: HybridRenderOutput,
    surface: HybridRenderOutput,
    *,
    epsilon: float = 1.0e-6,
) -> tuple[HybridRenderOutput, torch.Tensor]:
    """Restore rigid evidence where a standalone 2D surface owns the ray.

    This is a soft, deployment-valid arbitration: no semantic mask, target
    RGB, camera id or sequence code participates.  Surface support is derived
    from native accumulated 2DGS alpha and is attenuated continuously when
    the foliage posterior is clearly in front in metric camera depth.
    """
    surface_alpha = surface.alpha.clamp(0.0, 1.0 - epsilon)
    volume_alpha = canopy.volume_alpha.clamp(0.0, 1.0 - epsilon)
    surface_tau = -torch.log1p(-surface_alpha)
    volume_tau = -torch.log1p(-volume_alpha)
    evidence_weight = surface_tau / (
        surface_tau + volume_tau
    ).clamp_min(epsilon)

    surface_present = surface_alpha > epsilon
    volume_present = volume_alpha > epsilon
    surface_depth = surface.depth
    volume_depth = canopy.volume_depth
    valid_depth = (
        surface_present
        & volume_present
        & torch.isfinite(surface_depth)
        & torch.isfinite(volume_depth)
        & (surface_depth > 0)
        & (volume_depth > 0)
    )
    depth_scale = (
        0.05
        + 0.02
        * torch.minimum(surface_depth, volume_depth).clamp_min(0.0)
    )
    depth_compatibility = torch.sigmoid(
        (volume_depth - surface_depth) / depth_scale.clamp_min(0.05)
    )
    depth_compatibility = torch.where(
        valid_depth,
        depth_compatibility,
        torch.where(
            surface_present & ~volume_present,
            torch.ones_like(depth_compatibility),
            torch.zeros_like(depth_compatibility),
        ),
    )
    surface_weight = (evidence_weight * depth_compatibility).detach()

    def pixels(a: torch.Tensor | None, b: torch.Tensor | None):
        if a is None or b is None:
            return a if b is None else b
        return torch.lerp(a, b, surface_weight)

    def rows(a: torch.Tensor | None, b: torch.Tensor | None):
        if a is None or b is None:
            return a if b is None else b
        return torch.maximum(a, b)

    mixed = HybridRenderOutput(
        render=pixels(canopy.render, surface.render),
        alpha=pixels(canopy.alpha, surface.alpha),
        depth=pixels(canopy.depth, surface.depth),
        normal_world=pixels(canopy.normal_world, surface.normal_world),
        median_depth=pixels(canopy.median_depth, surface.median_depth),
        distortion=pixels(canopy.distortion, surface.distortion),
        radii=rows(canopy.radii, surface.radii),
        means2d=None,
        structural_count=canopy.structural_count,
        responsibility=rows(
            canopy.responsibility, surface.responsibility
        ),
        gate_responsibility=rows(
            canopy.gate_responsibility, surface.gate_responsibility
        ),
        surface_alpha=pixels(canopy.surface_alpha, surface.surface_alpha),
        volume_alpha=pixels(canopy.volume_alpha, surface.volume_alpha),
        surface_depth=pixels(canopy.surface_depth, surface.surface_depth),
        volume_depth=pixels(canopy.volume_depth, surface.volume_depth),
        surface_means2d=None,
        volume_means2d=None,
        volume_replacement=rows(
            canopy.volume_replacement, surface.volume_replacement
        ),
        volume_replacement_candidate=rows(
            canopy.volume_replacement_candidate,
            surface.volume_replacement_candidate,
        ),
    )
    return mixed, surface_weight

# These pairs are audited render-equivalent, not a general stale-code escape
# hatch.  The newer VolumetricFoliageModel revision only adds split-time
# evidence bookkeeping and changes how *future training* creates children.
# It does not change the mixed rasterizer, SH evaluation, compositing, or any
# tensor consumed by forward rendering.  Keeping both hashes in the pair
# makes the exception self-invalidating as soon as either implementation
# changes again.
RENDER_EQUIVALENT_IMPLEMENTATION_PAIRS = {
    "gaussian_model": {
        (
            "16ee30778e0cf2db71a1d0e138cc01ca4e14ae1d78cf54c5eaddcbea38d4830c",
            "e0439788449ba69e8590a37383373f38d7ba63752a952c4ac2032a2440868015",
        ): (
            "checkpoint inference is unchanged; the newer implementation "
            "only makes pre-optimizer evidence-seed append preserve complete "
            "metadata and per-row initial opacity"
        ),
        (
            "e0439788449ba69e8590a37383373f38d7ba63752a952c4ac2032a2440868015",
            "1faf9387e35d8cfe9d77444155ce434631c235ade9d5f25de59fb2339cdda473",
        ): (
            "checkpoint inference is byte-for-byte unchanged; the newer "
            "implementation only reserves source role 4 for future DAV2 "
            "rigid-hole births so they cannot alias renderer-owned residuals"
        ),
        (
            "1faf9387e35d8cfe9d77444155ce434631c235ade9d5f25de59fb2339cdda473",
            "d52dd5fd1c55bf063a6b16b6f75871e1fa50346de280cbae1faf815839aa91a4",
        ): (
            "checkpoint inference is unchanged; the newer implementation "
            "only multiplies future densification priority by continuous "
            "opacity maturity and does not alter any forward-render tensor"
        ),
        (
            "e0439788449ba69e8590a37383373f38d7ba63752a952c4ac2032a2440868015",
            "d52dd5fd1c55bf063a6b16b6f75871e1fa50346de280cbae1faf815839aa91a4",
        ): (
            "checkpoint inference is unchanged; the runtime only reserves "
            "source role 4 for future DAV2 births and continuously delays "
            "future low-opacity densification"
        ),
        (
            "d52dd5fd1c55bf063a6b16b6f75871e1fa50346de280cbae1faf815839aa91a4",
            "209e13bed7647a95fa4682c13b0693cb07d68b010e4f85e489e1822b6afadb2a",
        ): (
            "checkpoint inference is unchanged; the runtime only persists "
            "cumulative observation mass, preserves source provenance in "
            "future topology, and applies both solely to future culling"
        ),
        (
            "1faf9387e35d8cfe9d77444155ce434631c235ade9d5f25de59fb2339cdda473",
            "209e13bed7647a95fa4682c13b0693cb07d68b010e4f85e489e1822b6afadb2a",
        ): (
            "checkpoint inference is unchanged; the runtime reserves source "
            "role 4 and adds training-only optical maturity, cumulative "
            "observation mass and source-lineage-preserving topology"
        ),
        (
            "e0439788449ba69e8590a37383373f38d7ba63752a952c4ac2032a2440868015",
            "209e13bed7647a95fa4682c13b0693cb07d68b010e4f85e489e1822b6afadb2a",
        ): (
            "checkpoint inference is unchanged; the runtime reserves source "
            "role 4 for future DAV2 births and adds only training-time "
            "observation maturity and source-lineage-preserving topology"
        ),
        (
            "2d3f0a2a9ec9f93531428d220e123f58031d2f1bddf0d8c3d34670ed78f4d322",
            "209e13bed7647a95fa4682c13b0693cb07d68b010e4f85e489e1822b6afadb2a",
        ): (
            "checkpoint inference is unchanged; future DAV2 cull maturity "
            "is normalized by the training camera reference-set size rather "
            "than a fixed observation count"
        ),
        (
            "e0439788449ba69e8590a37383373f38d7ba63752a952c4ac2032a2440868015",
            "96391ac05a15bfc8fa62602da43c3ff67a793d5f30d23786850bbd68a06dc186",
        ): (
            "checkpoint inference is unchanged; the newer GaussianModel "
            "reserves source role 4 for future DAV2 births, preserves the "
            "later training-only maturity metadata, and excludes UV-bound "
            "Chart cells from future world-space clone/split/reallocation "
            "so a separate atlas can refine them"
        ),
        (
            "e0439788449ba69e8590a37383373f38d7ba63752a952c4ac2032a2440868015",
            "139b8ff76beda6c9786f7c8b6fe8285ad988185f135893b08ac5a44955cb5b49",
        ): (
            "checkpoint inference is unchanged; the newer GaussianModel "
            "only changes future training topology: it reserves source role 4, "
            "adds source-local maturity and Chart ownership, retains oversized "
            "parents for replace-split, and defers at capacity instead of "
            "unrelated global eviction"
        ),
        (
            "139b8ff76beda6c9786f7c8b6fe8285ad988185f135893b08ac5a44955cb5b49",
            "f680766125613effeeac372864ce52f78cff91ed4879135a5c3ef842a44b3be4",
        ): (
            "checkpoint inference is unchanged; the newer GaussianModel "
            "only restricts future opacity culling and clone/replace-split "
            "to an explicitly mutable residual suffix and applies smooth "
            "evidence-lineage maturity during future training"
        ),
        (
            "e0439788449ba69e8590a37383373f38d7ba63752a952c4ac2032a2440868015",
            "f680766125613effeeac372864ce52f78cff91ed4879135a5c3ef842a44b3be4",
        ): (
            "checkpoint inference is unchanged; the runtime reserves source "
            "role 4 and changes only future training topology, including "
            "source-local maturity, Chart ownership, replacing splits, "
            "capacity deferral and an explicitly mutable residual suffix"
        ),
        (
            "f680766125613effeeac372864ce52f78cff91ed4879135a5c3ef842a44b3be4",
            "ba7cbf24f0eac44498f689d3f73dfc768999276aa820676eb021a3e5ac887fb8",
        ): (
            "checkpoint inference is unchanged; the newer GaussianModel "
            "only lets measured screen-footprint deficit contribute to "
            "future replace-split priority"
        ),
        (
            "e0439788449ba69e8590a37383373f38d7ba63752a952c4ac2032a2440868015",
            "ba7cbf24f0eac44498f689d3f73dfc768999276aa820676eb021a3e5ac887fb8",
        ): (
            "checkpoint inference is unchanged; all differences reserve "
            "source role 4 or alter only future evidence-mature topology, "
            "including screen-footprint-driven replace-split priority"
        ),
        (
            "ba7cbf24f0eac44498f689d3f73dfc768999276aa820676eb021a3e5ac887fb8",
            "77e9902a6788e8ee09dfb9f3c1f56926d23a077cb817cc331b9ed5705ec4e9f3",
        ): (
            "checkpoint inference is unchanged; the newer GaussianModel "
            "only removes ordinary opacity as future retirement authority "
            "for UV-bound Chart atlas cells"
        ),
        (
            "e0439788449ba69e8590a37383373f38d7ba63752a952c4ac2032a2440868015",
            "77e9902a6788e8ee09dfb9f3c1f56926d23a077cb817cc331b9ed5705ec4e9f3",
        ): (
            "checkpoint inference is unchanged; all differences reserve "
            "source role 4 or alter only future evidence-owned topology, "
            "including replace-only retirement of bound Chart atlas cells"
        ),
        (
            "77e9902a6788e8ee09dfb9f3c1f56926d23a077cb817cc331b9ed5705ec4e9f3",
            "b55bc847aa638b07c06f5a9e12d9fd5fc6e81a77ba977ba9d3605415fc821618",
        ): (
            "checkpoint inference is unchanged; the newer GaussianModel "
            "only makes future screen footprint multiply an observed "
            "residual instead of independently creating split eligibility"
        ),
        (
            "ba7cbf24f0eac44498f689d3f73dfc768999276aa820676eb021a3e5ac887fb8",
            "b55bc847aa638b07c06f5a9e12d9fd5fc6e81a77ba977ba9d3605415fc821618",
        ): (
            "checkpoint inference is unchanged; future training additionally "
            "requires explicit Chart retirement authority and conditions "
            "screen-driven split priority on an observed residual"
        ),
        (
            "e0439788449ba69e8590a37383373f38d7ba63752a952c4ac2032a2440868015",
            "b55bc847aa638b07c06f5a9e12d9fd5fc6e81a77ba977ba9d3605415fc821618",
        ): (
            "checkpoint inference is unchanged; differences reserve source "
            "role 4 or alter only future evidence-owned topology, including "
            "Chart retirement and residual-conditioned screen splitting"
        ),
    },
    "hybrid_renderer": {
        (
            "be76b3e033a48751a6bd57d4d91f426de1e9a52c3bef4af49474959373813389",
            "540c15345f655ddacbedecc07ace2db52fb555b3aedc099f9365cf4b39132bd9",
        ): (
            "forward render is unchanged; the newer implementation only "
            "repairs future volume split metadata and optical-depth mass"
        ),
        (
            "4d52977d338c2d1c5b61a8e2a1bb0921b5a56861531f2898ff2de73fd8c0dbc8",
            "6eb9d91360bf9830c7e37e9ce8500588bfb391f777f89728be4ac4434550fe25",
        ): (
            "forward render is unchanged when the new optional gradient-only "
            "ownership gate is absent and replacement detachment changes only "
            "future training backward routing, not checkpoint inference"
        ),
        (
            "6eb9d91360bf9830c7e37e9ce8500588bfb391f777f89728be4ac4434550fe25",
            "beb20803725c2c3e323835f145c0f40c3161ca034dec2b2eae79342a02158497",
        ): (
            "support-camera visibility reduction is now evaluated in bounded "
            "row chunks; every primitive uses the same inputs and reduction, "
            "so checkpoint rendering is numerically unchanged while peak "
            "temporary memory is bounded"
        ),
        (
            "beb20803725c2c3e323835f145c0f40c3161ca034dec2b2eae79342a02158497",
            "785d66b6952b26480ecfc9eddd1e0646ef91e64c7ca7889359157f1cab34b1fe",
        ): (
            "exact-zero volume rows are removed before mixed CUDA "
            "preprocessing and restored in topology/audit outputs; the CUDA "
            "kernel already discarded those rows before tile emission, so "
            "rendered values and parameter gradients are unchanged"
        ),
        (
            "785d66b6952b26480ecfc9eddd1e0646ef91e64c7ca7889359157f1cab34b1fe",
            "684190359c14e128d9d62bd885f646af929e675221f5f4126a3a4bc1dd562e32",
        ): (
            "checkpoint inference is unchanged; the new optional "
            "parameter-family gradient gates only separate future training "
            "backward ownership and default to the historical common gate"
        ),
        (
            "684190359c14e128d9d62bd885f646af929e675221f5f4126a3a4bc1dd562e32",
            "d6eeba808a8fb07e1d3a5b940d1d10a5868b6b7e3342173a7a818eec0ec4d67b",
        ): (
            "checkpoint inference is unchanged; base and temporal-residual "
            "gradient gates are now applied before their forward-identical "
            "sum and default to the prior parameter-family gate"
        ),
        (
            "ae7984e63425df4e512b49868c86b7a94a2a7303fc0592a71f818894495c3d13",
            "f78bef52f44c962e723b5785b26b206569aa4459da21c40b0dd77d0df6ffe8bc",
        ): (
            "checkpoint inference is unchanged; the newer implementation "
            "only adds an optional camera-plane direction to future "
            "adaptive volume splits and leaves every existing forward "
            "render tensor and mixed CUDA call unchanged"
        ),
        (
            "f78bef52f44c962e723b5785b26b206569aa4459da21c40b0dd77d0df6ffe8bc",
            "5b210eeb406502d30b0c0d83c2361221677a17e68417aee228386382303f6876",
        ): (
            "checkpoint inference is unchanged; the newer implementation "
            "only fuses future volume retirement and adaptive splitting "
            "into one parameter materialization before the same single "
            "Adam-state migration"
        ),
    },
}

PROJECTED_OPTICAL_FOOTPRINT_REPAIR_PREDECESSOR = {
    "protocol": "cambridge_native_hybrid_teacher_v39_conditioned_topology_settle",
    "hybrid_renderer": (
        "5b210eeb406502d30b0c0d83c2361221677a17e68417aee228386382303f6876"
    ),
}
STATIC_FOLIAGE_RENDER_REPAIR_PREDECESSOR = {
    "protocol": "cambridge_native_hybrid_teacher_v40_projected_optical_footprint",
    "appearance_uncertainty": (
        "9e0591affdaea7ebe286d3c95985926b1b027255679211328645ba5e585090bf"
    ),
    "hybrid_renderer": (
        "73d425a03926f65a494c6ba6b81aa72f19e44b006024918e4bda061203fb8251"
    ),
}
STATIC_OPTICAL_HANDOFF_REPAIR_PREDECESSOR = {
    "protocol": (
        "cambridge_native_hybrid_teacher_v51_static_scene_snapshot_"
        "visible_canonical_detail_schedule"
    ),
    "hybrid_renderer": (
        "c48518cda08c776ccd47fac435fa8c1739ae133f351b8f859cc7604b333f06eb"
    ),
}


def _validate_render_implementation(
    state: dict,
    *,
    allow_projected_optical_footprint_repair: bool = False,
    allow_static_foliage_render_repair: bool = False,
    allow_static_optical_handoff_repair: bool = False,
) -> dict:
    """Reject silently reinterpreting a state with different render code."""
    expected = state.get("implementation_hashes")
    if not expected:
        # Historical states predate embedded implementation hashes. Their
        # state-file hash still makes repeated evaluations identifiable, but
        # they cannot claim source-level renderer reproducibility.
        return {
            "status": "historical_state_without_implementation_hashes",
            "exact": False,
            "render_equivalent_migrations": {},
        }
    paths = {
        "appearance_uncertainty": REPO_ROOT
        / "outdoor/appearance_uncertainty.py",
        "dataset_reader": SURFEL_ROOT / "scene/dataset_readers.py",
        "gaussian_model": SURFEL_ROOT / "scene/gaussian_model.py",
        "hybrid_renderer": REPO_ROOT
        / "outdoor/hybrid_gaussian_renderer.py",
        "mixed_forward_cuda": SURFEL_ROOT
        / "submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu",
    }
    changed = []
    equivalent = {}
    actual_hashes = {}
    for name, path in paths.items():
        actual = sha256_file(path)
        actual_hashes[name] = actual
        stored = expected.get(name)
        if stored == actual:
            continue
        migration = RENDER_EQUIVALENT_IMPLEMENTATION_PAIRS.get(
            name, {}
        ).get((stored, actual))
        if migration is None:
            changed.append(name)
        else:
            equivalent[name] = {
                "state_hash": stored,
                "runtime_hash": actual,
                "reason": migration,
            }
    causal_repair = None
    deployment_static = state.get("training_contract", {}).get(
        "deployment_static_contract", {}
    )
    if (
        changed == ["hybrid_renderer"]
        and state.get("protocol")
        == "cambridge_native_hybrid_teacher_v59_static_rgb_appearance_coverage_closed"
        and expected.get("hybrid_renderer")
        == "da392232d3186c6cab145d72b537474cf5a8175607758f47ed25b33e66be768e"
        and deployment_static.get("optical_replacement_policy")
        in {
            STATIC_RAY_NORMALIZED_OPTICAL_POLICY,
            STATIC_RAY_SURFACE_EVIDENCE_POLICY,
        }
    ):
        # v60 changes only the *training* single-pass static hand-off: the
        # centre proxy becomes candidate-only and persistent mass is updated
        # by scheduled real-ray evidence. v59 deployment already used the
        # separate two/three-pass ray-normalized compositor, whose inputs and
        # pixel equations are unchanged. This exact predecessor/policy pair is
        # therefore safe to load for a zero-training CUDA compatibility audit.
        equivalent["hybrid_renderer"] = {
            "state_hash": expected["hybrid_renderer"],
            "runtime_hash": actual_hashes["hybrid_renderer"],
            "reason": (
                "v59 ray-normalized deployment is unchanged; v60 modifies "
                "only future training evidence/lifecycle and adds metadata"
            ),
        }
        changed.clear()
    if (
        allow_static_optical_handoff_repair
        and changed == ["hybrid_renderer"]
        and state.get("protocol")
        == STATIC_OPTICAL_HANDOFF_REPAIR_PREDECESSOR["protocol"]
        and expected.get("hybrid_renderer")
        == STATIC_OPTICAL_HANDOFF_REPAIR_PREDECESSOR["hybrid_renderer"]
    ):
        causal_repair = {
            "state_hash": expected["hybrid_renderer"],
            "runtime_hash": actual_hashes["hybrid_renderer"],
            "reason": (
                "explicit v51 optical-handoff diagnostic: replace the "
                "split-sensitive group-centroid/radius heuristic with a "
                "same-group, view/depth-local Gaussian-mixture allocation "
                "that conserves optical depth"
            ),
        }
        changed.clear()
    if (
        allow_static_foliage_render_repair
        and set(changed) == {"appearance_uncertainty", "hybrid_renderer"}
        and state.get("protocol")
        == STATIC_FOLIAGE_RENDER_REPAIR_PREDECESSOR["protocol"]
        and all(
            expected.get(name)
            == STATIC_FOLIAGE_RENDER_REPAIR_PREDECESSOR[name]
            for name in ("appearance_uncertainty", "hybrid_renderer")
        )
    ):
        causal_repair = {
            "state_hashes": {
                name: expected[name]
                for name in ("appearance_uncertainty", "hybrid_renderer")
            },
            "runtime_hashes": {
                name: actual_hashes[name]
                for name in ("appearance_uncertainty", "hybrid_renderer")
            },
            "reason": (
                "explicit v40->v41 diagnostic: camera-plane canopy RGB and "
                "uncertainty grids are bypassed and group replacement is "
                "localized by active-view projection and depth interval"
            ),
        }
        changed.clear()
    if (
        allow_projected_optical_footprint_repair
        and changed == ["hybrid_renderer"]
        and state.get("protocol")
        == PROJECTED_OPTICAL_FOOTPRINT_REPAIR_PREDECESSOR["protocol"]
        and expected.get("hybrid_renderer")
        == PROJECTED_OPTICAL_FOOTPRINT_REPAIR_PREDECESSOR[
            "hybrid_renderer"
        ]
    ):
        causal_repair = {
            "state_hash": expected["hybrid_renderer"],
            "runtime_hash": actual_hashes["hybrid_renderer"],
            "reason": (
                "explicit v39->v40 render repair: exact-ray metric depth "
                "posterior is decoupled from non-owner EWA footprint and "
                "local hand-off uses view-projected optical mass"
            ),
        }
        changed.clear()
    if changed:
        raise RuntimeError(
            "Teacher state render implementation hash mismatch: "
            + ", ".join(changed)
        )
    return {
        "status": (
            "static_foliage_render_causal_repair"
            if causal_repair is not None
            and state.get("protocol")
            == STATIC_FOLIAGE_RENDER_REPAIR_PREDECESSOR["protocol"]
            else "projected_optical_footprint_causal_repair"
            if causal_repair is not None
            else (
                "render_equivalent_migration"
                if equivalent
                else "exact_implementation_match"
            )
        ),
        "exact": not equivalent and causal_repair is None,
        "render_equivalent_migrations": equivalent,
        "causal_render_repair": causal_repair,
        "runtime_hashes": actual_hashes,
    }


def teacher_branch_validity(state: dict) -> dict[str, bool | str]:
    """Report which learned branches are valid at this exact checkpoint.

    A training *profile* describes the eventual model, not the state reached
    by an intermediate checkpoint.  In particular, hybrid checkpoints saved
    during ``canonical_bootstrap`` contain initialized foliage tensors that
    have never received a gradient.  Treating those tensors as a trained
    conditioned model made early visualizations and metrics look authoritative
    when they were only seed diagnostics.
    """
    profile = str(state.get("training_profile", "unknown"))
    phase = str(state.get("phase", ""))
    if state.get("training_contract", {}).get(
        "reconstruction_target"
    ) == "static":
        iteration = int(state.get("iteration", 0))
        configured = state.get("training_contract", {}).get(
            "branch_activation", {}
        )
        foliage_iteration = int(configured.get("foliage_iteration", 1))
        return {
            "canonical_surface": True,
            "canonical_canopy": iteration >= foliage_iteration,
            "conditioned": False,
            "reason": "single_static_localization_map",
        }
    if profile == "hybrid_rigid_stage1":
        return {
            "canonical_surface": True,
            "canonical_canopy": False,
            "conditioned": False,
            "reason": "rigid_stage_profile",
        }
    # Named phases and actual branch activation are independent schedules.
    # The handoff profile, for example, trains dynamic leaves from 8% while
    # the phase is still named ``topology`` until 25%.  Phase-only validation
    # therefore hid a genuinely trained conditioned branch at 3k/30k and
    # made the evaluator report a canonical render as if it tested the new
    # dynamic path. New checkpoints should persist these fractions in their
    # training contract; this table is the exact legacy/current-profile
    # fallback for states written before that field existed.
    branch_starts = {
        "quality": {"foliage": 0.0, "dynamic": 0.06},
        "fast": {"foliage": 0.0, "dynamic": 0.08},
        "hybrid_quality": {"foliage": 0.12, "dynamic": 0.50},
        "hybrid_fast": {"foliage": 0.08, "dynamic": 0.68},
        "hybrid_handoff_quality": {
            "foliage": 0.0,
            # Checkpoints old enough to omit branch_activation used the
            # original 8% handoff schedule. Current checkpoints persist their
            # exact 2% activation in the immutable training contract.
            "dynamic": 0.08,
        },
    }
    contract = state.get("training_contract", {})
    total = int(
        contract.get(
            "schedule_horizon",
            contract.get("iterations", state.get("iterations", 0)),
        )
    )
    iteration = int(state.get("iteration", 0))
    configured = contract.get("branch_activation", branch_starts.get(profile))
    if (
        isinstance(configured, dict)
        and total > 0
        and iteration > 0
    ):
        foliage_iteration = configured.get("foliage_iteration")
        dynamic_iteration = configured.get("dynamic_iteration")
        if foliage_iteration is not None and dynamic_iteration is not None:
            foliage_iteration = int(foliage_iteration)
            dynamic_iteration = int(dynamic_iteration)
            return {
                "canonical_surface": True,
                "canonical_canopy": iteration >= foliage_iteration,
                "conditioned": iteration >= dynamic_iteration,
                "reason": (
                    "checkpoint_absolute_branch_schedule:"
                    f"{iteration}/{total},dynamic>={dynamic_iteration}"
                ),
            }
        progress = iteration / float(total)
        foliage_start = float(
            configured.get("foliage", configured.get("foliage_start", 0.0))
        )
        dynamic_start = float(
            configured.get(
                "dynamic", configured.get("dynamic_start", 2.0)
            )
        )
        return {
            "canonical_surface": True,
            "canonical_canopy": progress > foliage_start,
            "conditioned": progress > dynamic_start,
            "reason": (
                "checkpoint_branch_schedule:"
                f"{iteration}/{total},dynamic>{dynamic_start:.6g}"
            ),
        }
    if phase:
        canonical_canopy = phase != "canonical_bootstrap"
        conditioned = phase in {
            "dynamic_appearance",
            "ownership_cleanup",
            "canonical_polish",
        }
        return {
            "canonical_surface": True,
            "canonical_canopy": canonical_canopy,
            "conditioned": conditioned,
            "reason": f"checkpoint_phase:{phase}",
        }
    # Older final states may not persist a phase.  Accept them only when the
    # checkpoint itself proves that the complete requested schedule finished.
    iteration = int(state.get("iteration", 0))
    total = int(
        state.get("training_contract", {}).get(
            "schedule_horizon",
            state.get("training_contract", {}).get(
                "iterations", state.get("iterations", 0)
            ),
        )
    )
    complete = total > 0 and iteration >= total
    return {
        "canonical_surface": True,
        "canonical_canopy": complete,
        "conditioned": complete,
        "reason": (
            "legacy_complete_schedule"
            if complete
            else "legacy_checkpoint_without_phase"
        ),
    }


@dataclass
class HybridTeacher:
    surface: GaussianModel
    foliage: VolumetricFoliageModel
    sky: CanonicalDirectionalSky
    appearance: OutdoorAppearanceUncertainty
    state: dict

    def render(
        self,
        camera,
        *,
        task: dict[str, torch.Tensor] | None = None,
        conditioned: bool = False,
        surface_only: bool = False,
        volume_only: bool = False,
        volume_layer: str = "all",
        background: torch.Tensor | None = None,
        exact_ray_render_aspect_limit: float = 4.0,
        optical_replacement_policy: str | None = None,
        optical_responsibility_prior: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Render the mixed teacher or its native structural counterfactual.

        ``surface_only`` is an evaluation/debug counterfactual.  It disables
        every 3D volume in the forward pass and therefore means something
        stricter than ``conditioned=False``, which still renders the canonical
        crown.  Keeping the distinction explicit prevents evaluators from
        silently labelling a canonical mixed image as a rigid scaffold.
        """
        if background is None:
            background = torch.ones(3, device=self.surface.get_xyz.device)
        if surface_only and conditioned:
            raise ValueError(
                "surface_only and conditioned are mutually exclusive"
            )
        if surface_only and volume_only:
            raise ValueError("surface_only and volume_only are mutually exclusive")
        if volume_layer not in {
            "all", "skeleton", "envelope", "detail", "none"
        }:
            raise ValueError(
                "volume_layer must be all, skeleton, envelope, detail or none"
            )
        optical_contract = resolve_deployment_optical_contract(
            self.state,
            requested_policy=optical_replacement_policy,
            requested_prior=optical_responsibility_prior,
        )
        optical_replacement_policy = str(optical_contract["policy"])
        optical_responsibility_prior = float(
            optical_contract["optical_responsibility_prior"]
        )
        ray_normalized_static = optical_replacement_policy in {
            STATIC_RAY_NORMALIZED_OPTICAL_POLICY,
            STATIC_RAY_SURFACE_EVIDENCE_POLICY,
        }
        surface_evidence_static = (
            optical_replacement_policy
            == STATIC_RAY_SURFACE_EVIDENCE_POLICY
        )
        if ray_normalized_static:
            if conditioned:
                raise ValueError(
                    "ray_normalized_two_pass is a single-static-map policy; "
                    "it cannot render a sequence-conditioned branch"
                )
            if self.state.get("training_contract", {}).get(
                "reconstruction_target"
            ) != "static":
                raise ValueError(
                    "ray_normalized_two_pass requires a static Teacher"
                )
            if volume_layer != "all":
                raise ValueError(
                    "ray_normalized_two_pass owns the envelope/detail split; "
                    "explicit volume_layer counterfactuals must use a "
                    "single-pass replacement policy"
                )
            if optical_responsibility_prior < 0:
                raise ValueError(
                    "optical_responsibility_prior must be non-negative"
                )
        elif optical_responsibility_prior != 0:
            raise ValueError(
                "optical_responsibility_prior is valid only with "
                "ray_normalized_two_pass"
            )
        # Preserve the ordinary teacher API: the default all-volume path does
        # not require the new layer metadata and passes no diagnostic mask.
        # Layer metadata is consulted only for an explicit counterfactual.
        volume_role_mask = None
        if volume_layer == "skeleton":
            volume_role_mask = self.foliage.static_skeleton_mask
        elif volume_layer == "envelope":
            volume_role_mask = self.foliage.persistent_envelope_mask
        elif volume_layer == "detail":
            volume_role_mask = self.foliage.detail_leaf_mask
        elif volume_layer == "none":
            volume_role_mask = torch.zeros_like(
                self.foliage.layer_role, dtype=torch.bool
            )
        validity = teacher_branch_validity(self.state)
        if conditioned and not bool(validity["conditioned"]):
            raise RuntimeError(
                "This checkpoint does not yet contain a trained conditioned "
                f"branch ({validity['reason']})"
            )
        temporal_code = (
            self.appearance.temporal_code(camera.image_name)
            if conditioned
            else None
        )
        volume_gate = None
        if conditioned:
            contract = self.state.get("camera_ownership_contract")
            if contract is not None:
                volume_gate = dynamic_visibility_gate(
                    self.foliage,
                    int(camera.colmap_id),
                    contract["sequence_lookup"].to(
                        device=self.foliage.xyz.device
                    ),
                    contract["frame_lookup"].to(
                        device=self.foliage.xyz.device
                    ),
                )
            else:
                volume_gate = dynamic_visibility_gate(
                    self.foliage, int(camera.colmap_id), None, None
                )
        def render_package(
            role_mask: torch.Tensor | None,
            replacement_policy: str,
        ) -> HybridRenderOutput:
            return render_hybrid(
                camera,
                self.surface,
                self.foliage,
                background=background,
                temporal_code=temporal_code,
                include_dynamic=conditioned,
                volume_opacity_scale=(
                    0.0
                    if surface_only
                    else (
                        1.0
                        if bool(validity["canonical_canopy"])
                        else 0.0
                    )
                ),
                volume_role_mask=role_mask,
                surface_gate=(
                    torch.zeros_like(self.surface.get_opacity.reshape(-1))
                    if volume_only
                    else None
                ),
                volume_gate=volume_gate,
                exact_ray_render_aspect_limit=(
                    exact_ray_render_aspect_limit
                ),
                optical_replacement_policy=replacement_policy,
            )

        optical_detail_responsibility = None
        optical_surface_responsibility = None
        if ray_normalized_static and not surface_only:
            skeleton = self.foliage.static_skeleton_mask
            envelope_package = render_package(
                skeleton | self.foliage.persistent_envelope_mask,
                "disabled",
            )
            detail_package = render_package(
                skeleton | self.foliage.static_leaf_mask,
                "disabled",
            )
            package, optical_detail_responsibility = (
                _static_ray_normalized_optical_mixture(
                    envelope_package,
                    detail_package,
                    symmetric_optical_prior=(
                        optical_responsibility_prior
                    ),
                )
            )
            if surface_evidence_static:
                surface_package = render_package(
                    torch.zeros_like(
                        self.foliage.layer_role, dtype=torch.bool
                    ),
                    "disabled",
                )
                package, optical_surface_responsibility = (
                    _static_surface_evidence_mixture(
                        package, surface_package
                    )
                )
        else:
            package = render_package(
                volume_role_mask,
                (
                    "disabled"
                    if ray_normalized_static
                    else optical_replacement_policy
                ),
            )
        rgb = composite_white_background(
            package.render, package.alpha, self.sky(camera)
        )
        if conditioned:
            if task is None:
                # Deployment routing must come from the rendered owners, not
                # ground-truth semantic masks.  surface_alpha and
                # volume_alpha are contribution alphas accumulated in the
                # same tile/depth-sorted mixed pass; the remaining
                # transmittance is sky.  Keep these fields soft at boundaries
                # so the appearance branch cannot create mask-shaped seams.
                surface_owner = package.surface_alpha[0].clamp(0, 1)
                volume_owner = package.volume_alpha[0].clamp(0, 1)
                sky_owner = (1.0 - package.alpha[0]).clamp(0, 1)
                owner_sum = (
                    surface_owner + volume_owner + sky_owner
                ).clamp_min(1e-6)
                task = {
                    "p_rigid": surface_owner / owner_sum,
                    "p_canopy": volume_owner / owner_sum,
                    "p_sky": sky_owner / owner_sum,
                    "p_transient": torch.zeros_like(surface_owner),
                    "semantic_condition_source": "rendered_owner_alpha",
                }
            rgb = self.appearance(rgb, camera, task)
        package_radii = getattr(
            package,
            "radii",
            torch.empty(
                0,
                device=package.render.device,
                dtype=package.render.dtype,
            ),
        )
        structural_count = min(
            int(getattr(package, "structural_count", 0)),
            int(len(package_radii)),
        )
        return {
            "rgb": rgb.clamp(0, 1),
            "alpha": package.alpha,
            "depth": package.depth,
            "surface_depth": package.surface_depth,
            "volume_depth": package.volume_depth,
            "normal_world": package.normal_world,
            "surface_alpha": package.surface_alpha,
            "volume_alpha": package.volume_alpha,
            # Expose footprint diagnostics without changing the rendered
            # representation. Mixed tile order is keyed by centre depth,
            # while a 2D surfel's final ray-intersection depth is pixel
            # dependent. Large projected surfels are therefore the subset
            # for which that standard approximation deserves explicit audit.
            "surface_radii": package_radii[:structural_count],
            "volume_radii": package_radii[structural_count:],
            "volume_means2d": getattr(package, "volume_means2d", None),
            "volume_replacement": getattr(
                package, "volume_replacement", None
            ),
            "volume_replacement_candidate": getattr(
                package, "volume_replacement_candidate", None
            ),
            "optical_detail_responsibility": (
                optical_detail_responsibility
            ),
            "optical_surface_responsibility": (
                optical_surface_responsibility
            ),
            "optical_compositing_policy": (
                optical_replacement_policy
                if ray_normalized_static and not surface_only
                else optical_replacement_policy
            ),
            "optical_responsibility_prior": float(
                optical_responsibility_prior
            ),
            "optical_compositing_contract": dict(optical_contract),
        }


def _restore_surface(model: GaussianModel, capture: tuple) -> None:
    (
        active,
        xyz,
        features_dc,
        features_rest,
        scaling,
        rotation,
        opacity,
    ) = capture[:7]
    model.active_sh_degree = int(active)
    model._xyz = torch.nn.Parameter(xyz.detach().cuda(), requires_grad=False)
    model._features_dc = torch.nn.Parameter(
        features_dc.detach().cuda(), requires_grad=False
    )
    model._features_rest = torch.nn.Parameter(
        features_rest.detach().cuda(), requires_grad=False
    )
    model._scaling = torch.nn.Parameter(
        scaling.detach().cuda(), requires_grad=False
    )
    model._rotation = torch.nn.Parameter(
        rotation.detach().cuda(), requires_grad=False
    )
    model._opacity = torch.nn.Parameter(
        opacity.detach().cuda(), requires_grad=False
    )
    model._restore_point_metadata(capture[12] if len(capture) >= 13 else None)


@torch.no_grad()
def _apply_deployment_surface_geometry(
    model: GaussianModel, payload: dict | None
) -> dict:
    """Apply a checkpoint's baked live-atlas geometry for read-only render."""
    if payload is None:
        return {"applied": False, "reason": "legacy_or_final_baked_surface"}
    if payload.get("version") != "native-2dgs-deployment-geometry-v1":
        raise RuntimeError("Unsupported deployment surface geometry protocol")
    point_count = int(payload.get("point_count", -1))
    if point_count != len(model.get_xyz):
        raise RuntimeError(
            "Deployment surface geometry point count differs from resume "
            "surface capture"
        )
    expected = {
        "xyz": model._xyz.shape,
        "scaling": model._scaling.shape,
        "rotation": model._rotation.shape,
    }
    for name, shape in expected.items():
        value = torch.as_tensor(payload[name])
        if value.shape != shape:
            raise RuntimeError(
                f"Deployment surface {name} shape is {tuple(value.shape)}, "
                f"expected {tuple(shape)}"
            )
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(
                f"Deployment surface {name} contains non-finite values"
            )
        getattr(model, f"_{name}").copy_(
            value.to(
                device=model.get_xyz.device,
                dtype=getattr(model, f"_{name}").dtype,
            )
        )
    return {
        "applied": True,
        "point_count": point_count,
        "chart_baked_rows": int(payload.get("chart_baked_rows", 0)),
        "resume_surface_unchanged": bool(
            payload.get("resume_surface_unchanged", False)
        ),
    }


def load_hybrid_teacher(
    state_path: Path,
    *,
    sh_degree: int,
    allow_projected_optical_footprint_repair: bool = False,
    allow_static_foliage_render_repair: bool = False,
    allow_static_optical_handoff_repair: bool = False,
) -> HybridTeacher:
    """Load a Teacher state. No student conversion or COLMAP geometry is used."""
    try:
        state = torch.load(
            Path(state_path), map_location="cpu", weights_only=False
        )
    except TypeError:
        state = torch.load(Path(state_path), map_location="cpu")
    protocol = str(state.get("protocol", ""))
    if protocol not in SUPPORTED_TEACHER_PROTOCOLS:
        raise RuntimeError(f"Unsupported hybrid Teacher protocol {protocol!r}")
    implementation_validation = _validate_render_implementation(
        state,
        allow_projected_optical_footprint_repair=(
            allow_projected_optical_footprint_repair
        ),
        allow_static_foliage_render_repair=(
            allow_static_foliage_render_repair
        ),
        allow_static_optical_handoff_repair=(
            allow_static_optical_handoff_repair
        ),
    )
    state["_render_implementation_validation"] = implementation_validation
    surface = GaussianModel(sh_degree)
    _restore_surface(surface, state["surface"])
    state["_deployment_surface_geometry"] = (
        _apply_deployment_surface_geometry(
            surface, state.get("deployment_surface_geometry")
        )
    )
    foliage = VolumetricFoliageModel(
        sh_degree,
        dynamic_rank=int(state["foliage"].get("dynamic_rank", 4)),
    ).cuda()
    foliage.restore(state["foliage"])
    if allow_static_optical_handoff_repair:
        state["_static_optical_handoff_group_association"] = (
            foliage.associate_static_detail_replacement_groups()
        )
    sky = CanonicalDirectionalSky(
        degree=int(state.get("sky_degree", 2))
    ).cuda()
    sky.load_state_dict(state["sky"])
    appearance_payload = state["appearance"]
    appearance = OutdoorAppearanceUncertainty(
        appearance_payload["image_names"],
        rank=int(appearance_payload["rank"]),
        sky_degree=int(appearance_payload["sky_degree"]),
        maximum_rgb_residual=float(
            appearance_payload["maximum_rgb_residual"]
        ),
        spatial_grid_size=int(appearance_payload["spatial_grid_size"]),
        temporal_code_norm=float(
            appearance_payload.get("temporal_code_norm", 0.25)
        ),
        device="cuda",
    )
    appearance.restore(appearance_payload)
    surface.active_sh_degree = min(surface.active_sh_degree, sh_degree)
    for parameter in foliage.parameters():
        parameter.requires_grad_(False)
    for parameter in sky.parameters():
        parameter.requires_grad_(False)
    for parameter in appearance.parameters():
        parameter.requires_grad_(False)
    return HybridTeacher(surface, foliage, sky, appearance, state)
