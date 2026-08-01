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
    "unified_outdoor_mixed_teacher_v1",
}

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


def _validate_render_implementation(
    state: dict,
    *,
    allow_projected_optical_footprint_repair: bool = False,
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
            "projected_optical_footprint_causal_repair"
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
        background: torch.Tensor | None = None,
        exact_ray_render_aspect_limit: float = 4.0,
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
        package = render_hybrid(
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
            volume_gate=volume_gate,
            exact_ray_render_aspect_limit=(
                exact_ray_render_aspect_limit
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


def load_hybrid_teacher(
    state_path: Path,
    *,
    sh_degree: int,
    allow_projected_optical_footprint_repair: bool = False,
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
    )
    state["_render_implementation_validation"] = implementation_validation
    surface = GaussianModel(sh_degree)
    _restore_surface(surface, state["surface"])
    foliage = VolumetricFoliageModel(
        sh_degree,
        dynamic_rank=int(state["foliage"].get("dynamic_rank", 4)),
    ).cuda()
    foliage.restore(state["foliage"])
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
