#!/usr/bin/env python
"""Train one from-scratch mixed outdoor teacher from unified evidence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
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
from outdoor.evidence_store import load_evidence_store  # noqa: E402
from outdoor.foliage_geometry import (  # noqa: E402
    evidence_conditioned_dynamic_opacity_ceiling,
    evidence_conditioned_leaf_optical_mass,
)
from outdoor.foliage_view_graph import sequence_id  # noqa: E402
from outdoor.hybrid_gaussian_renderer import (  # noqa: E402
    LAYER_CANONICAL_CROWN,
    LAYER_DYNAMIC_LEAF,
    VolumetricFoliageModel,
    dynamic_visibility_gate,
    render_hybrid,
)
from outdoor.lazy_scene import LazyScene, rgb_source_contract  # noqa: E402
from outdoor.runtime_provenance import collect_runtime_provenance  # noqa: E402
from outdoor.task_fields import OutdoorTaskFieldLookup  # noqa: E402
from outdoor.training_evidence import (  # noqa: E402
    FoliageRayEvidence,
    OutdoorGeometryEvidence,
)
from scene import GaussianModel  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402


PREDECESSOR_PROTOCOL = (
    "cambridge_native_hybrid_teacher_v35_atomic_volume_replace_split"
)
PROTOCOL = (
    "cambridge_native_hybrid_teacher_v40_projected_optical_footprint"
)
COUNTERFACTUAL_TRANSPARENCY_CONTRACT = (
    "detached_surface_only_relative_rgb_advantage_routes_only_volume_"
    "optical_depth"
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
            ("topology", 0.85),
            ("canonical_polish", 1.00),
        ),
        "foliage_start": 2.0,
        "dynamic_start": 2.0,
    },
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
    changed: set[str], *, enabled: bool
) -> bool:
    """Authorize only model-state-compatible Python repair resumes.

    A causal repair can live entirely in the Chart/evidence module while the
    trainer entrypoint itself stays byte-identical.  Requiring ``trainer`` to
    change made the explicit repair flag unusable for exactly that case.
    CUDA, renderer and Gaussian-model changes remain excluded.
    """
    return bool(
        enabled
        and changed
        and changed.issubset(
            {
                "trainer",
                "training_evidence",
                "chart_surface_model",
            }
        )
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
    parser.add_argument("--volume-scale-lr", type=float, default=4e-4)
    parser.add_argument("--volume-rotation-lr", type=float, default=2e-4)
    parser.add_argument("--dynamic-lr", type=float, default=3e-4)
    parser.add_argument("--appearance-lr", type=float, default=8e-4)
    parser.add_argument("--sky-lr", type=float, default=2e-3)
    parser.add_argument("--geometry-every", type=int, default=2)
    parser.add_argument("--topology-every", type=int, default=8)
    parser.add_argument("--replacement-every", type=int, default=100)
    parser.add_argument("--geometry-weight", type=float, default=0.12)
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
            "appearance_only",
            "frozen",
        ),
        default="joint",
        help=(
            "Optimization ownership for a validated rigid handoff. "
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
        "--skeleton-opacity-ceiling", type=float, default=0.25
    )
    parser.add_argument(
        "--canonical-crown-opacity-ceiling", type=float, default=0.35
    )
    parser.add_argument(
        "--dynamic-leaf-opacity-ceiling", type=float, default=0.40
    )
    parser.add_argument(
        "--maximum-volume-radius-pixels", type=float, default=32.0
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
        "--allow-volume-capacity-repair-resume",
        action="store_true",
        help=(
            "Resume only the exact retained v82 8k checkpoint into the "
            "audited 2M/20k/10k role-conserving capacity repair. This does "
            "not permit arbitrary trainer or optimization-contract changes."
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
    args = parser.parse_args()
    profile = TRAINING_PROFILES[args.training_profile]
    if args.iterations is None:
        args.iterations = int(profile["iterations"])
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.phase_schedule_horizon is None:
        args.phase_schedule_horizon = int(profile["iterations"])
    if args.phase_schedule_horizon <= 0:
        parser.error("--phase-schedule-horizon must be positive")
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
    if not (
        0.0
        <= args.surface_retirement_optical_mass_fraction_per_event
        <= 0.05
    ):
        parser.error(
            "--surface-retirement-optical-mass-fraction-per-event must "
            "lie in [0, 0.05]"
        )
    if args.volume_opacity_lr is None:
        args.volume_opacity_lr = 4e-3
    if args.volume_opacity_lr <= 0:
        parser.error("--volume-opacity-lr must be positive")
    if args.volume_topology_ramp_iterations is None:
        args.volume_topology_ramp_iterations = (
            int(round(0.08 * args.phase_schedule_horizon))
            if args.training_profile == "hybrid_handoff_quality"
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
    if args.image_prefetch_workers < 0:
        parser.error("--image-prefetch-workers must be non-negative")
    if args.image_prefetch_depth < 0:
        parser.error("--image-prefetch-depth must be non-negative")
    if args.maintenance_every < 0:
        parser.error("--maintenance-every must be non-negative")
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
            if args.training_profile in {"hybrid_quality", "hybrid_fast"}
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
    return (
        args,
        model.extract(args),
        optimization.extract(args),
        pipeline.extract(args),
    )


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
    step: int, schedule_horizon: int, training_profile: str
) -> bool:
    """Keep conditioned leaves trainable while final topology settles.

    ``canonical_polish`` is a topology-stable convergence phase, not a
    request to freeze the conditioned image-formation branch.  Freezing both
    at once left dynamic children created by the final replace/split event
    with only a few dozen owner-view updates while canonical volume continued
    to compensate for them during the remaining polish window.
    """
    return _dynamic_enabled(step, schedule_horizon, training_profile)


def _surface_topology_active(step: int, args) -> bool:
    """Return the explicit surface-topology schedule, independent of phase."""
    if getattr(
        args, "mature_handoff_surface_policy", "joint"
    ) != "joint":
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
    dynamic_start = float(
        TRAINING_PROFILES[args.training_profile]["dynamic_start"]
    )
    activation = _activation_iteration(
        dynamic_start, args.phase_schedule_horizon
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resume_training_contract_differences(
    saved: dict,
    current: dict,
    *,
    allow_trainer_repair_migration: bool = False,
    allow_conditioned_schedule_repair_migration: bool = False,
    allow_volume_capacity_repair_migration: bool = False,
    allow_surface_ownership_repair_migration: bool = False,
    allow_v38_causal_repair_migration: bool = False,
) -> set[str]:
    """Compare optimization identity without treating an RGB cache as K.

    ``camera_intrinsics_contract.json`` also records the size of the CPU
    image cache.  Its whole-file digest remains useful provenance, but that
    performance-only field cannot make an otherwise exact K/pose checkpoint
    ineligible for resume.  Camera identity is independently bound by the
    geometry digest and the exact per-camera validation payload.
    """
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
            "contract": COUNTERFACTUAL_TRANSPARENCY_CONTRACT,
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
    explanation.  Only ``volume_alpha`` remains differentiable, so neither
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
    static_semantic = (
        task["p_rigid"] + task.get("p_canopy", task["p_canopy_core"])
    ).clamp(0.0, 1.0)
    evidence_weight = (
        responsibility
        * surface_alpha.detach().reshape_as(responsibility).clamp(0.0, 1.0)
        * static_semantic
        * task.get("w_rgb", torch.ones_like(static_semantic))
        * (1.0 - task.get("p_transient", torch.zeros_like(static_semantic)))
        * _boundary_evidence_weight(task)
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
    """Validate a no-COLMAP, no-historical rigid surface provenance chain."""
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
    if bool(payload.get("colmap_points_or_tracks_used", True)):
        raise RuntimeError(
            "Rigid surface handoff used forbidden COLMAP point/track geometry"
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
    if surface_optimizer is None:
        # Compatibility path for already-produced v1 handoffs.  Their
        # content-addressed PLY and sibling result.json were written by the
        # same rigid run, but the early handoff schema omitted the schedule.
        # Validate the sibling's model identity before accepting its contract.
        producer_result_path = handoff_manifest.parent / "result.json"
        if not producer_result_path.is_file():
            raise RuntimeError(
                "Rigid surface handoff is missing its optimizer schedule and "
                "has no validated producer result.json"
            )
        producer_result = json.loads(
            producer_result_path.read_text(encoding="utf-8")
        )
        result_surface = Path(
            producer_result.get("surface_ply", "")
        ).expanduser().resolve()
        if result_surface != surface_ply:
            raise RuntimeError(
                "Rigid producer result surface identity mismatch"
            )
        if int(producer_result.get("iterations", -1)) != int(
            payload.get("surface_iteration", -2)
        ):
            raise RuntimeError(
                "Rigid producer result iteration mismatch"
            )
        surface_optimizer = producer_result.get(
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


def _inherit_surface_optimizer_contract(opt, surface_warmstart: dict) -> dict:
    """Make the handoff's complete surface optimizer schedule authoritative."""
    contract = dict(surface_warmstart["surface_optimizer"])
    for key, value in contract.items():
        setattr(opt, key, value)
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
    if evidence_store.get("geometry_source") != "mast3r_only":
        rejection_reasons.append("geometry source is not MASt3R-only")
    if evidence_colmap_geometry_used:
        rejection_reasons.append("Evidence Store used COLMAP point/track geometry")
    if bool(initialization.get("historical_model_initialization", True)):
        rejection_reasons.append("historical Gaussian initialization was used")
    if bool(surface_audit.get("historical_trained_ply_used", True)):
        rejection_reasons.append("surface seed used a historical trained PLY")
    if bool(surface_audit.get("colmap_points_or_tracks_used", True)):
        rejection_reasons.append("surface seed used COLMAP point/track geometry")
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
    """Select independently supported coverage missing from a rigid handoff.

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
        supported
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
            "independent_cross_sequence_or_persistent_evidence__"
            "continuous_existing_footprint_deficit_times_precision"
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


def _volume_optimizer(args, foliage, appearance, sky):
    return torch.optim.Adam(
        [
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
                "params": [
                    foliage.deformation_basis,
                    foliage.dynamic_feature_basis,
                    foliage.dynamic_opacity_basis,
                ],
                "lr": args.dynamic_lr,
                "name": "dynamic",
            },
            {
                "params": list(appearance.parameters()),
                "lr": args.appearance_lr,
                "name": "appearance",
            },
            {
                "params": list(sky.parameters()),
                "lr": args.sky_lr,
                "name": "sky",
            },
        ],
        eps=1e-15,
    )


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
    """Select high-score candidates without context/spatial starvation.

    Tree masks frequently connect several touching crowns into one dominant
    instance.  Reserving only one head per instance then still allocates
    almost an entire split event to one residual hotspot.  When positions are
    available, use (instance, coarse world cell) as the balancing group and
    give every group an equal first allocation before global score refill.

    Sequence-conditioned leaves have one additional physical axis: their
    calibrated owner camera.  Pooling candidates from a hundred conditioned
    views and then balancing only in world space lets several views of the
    same crown compete for the same cells; the highest-gradient view can take
    an entire event while another exact owner gets no topology update.  When
    ``context_id`` is supplied, allocate capacity fairly across the observed
    owner contexts first.  ``lineage_family_id`` adds a second hierarchy
    inside each owner so a high-confidence DAV2 lineage cannot consume every
    slot while a broad exact-ray RGB basis receives none.  The final level is
    the same instance/spatial rule.  Unused capacity is always refilled, so
    this is neither a fixed per-camera quota nor an eligibility gate.
    """
    quota = min(max(int(quota), 0), int(len(indices)))
    if quota == 0:
        return indices[:0]
    if quota == len(indices):
        return indices
    if lineage_family_id is not None:
        lineage_family_id = torch.as_tensor(
            lineage_family_id,
            device=indices.device,
            dtype=torch.int64,
        ).reshape(-1)
        if len(lineage_family_id) != len(tree_instance_id):
            raise ValueError(
                "lineage_family_id must have one value per topology row"
            )
    if context_id is not None:
        context_id = torch.as_tensor(
            context_id,
            device=indices.device,
            dtype=torch.int64,
        ).reshape(-1)
        if len(context_id) != len(tree_instance_id):
            raise ValueError(
                "context_id must have one value per topology row"
            )
        candidate_context = context_id[indices]
        contexts, inverse, context_counts = torch.unique(
            candidate_context,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        context_score = torch.full(
            (len(contexts),),
            -torch.inf,
            device=score.device,
            dtype=score.dtype,
        )
        safe_candidate_score = torch.nan_to_num(
            score[indices], nan=-torch.inf, neginf=-torch.inf
        )
        context_score.scatter_reduce_(
            0,
            inverse,
            safe_candidate_score,
            reduce="amax",
            include_self=True,
        )
        allocations = torch.zeros_like(context_counts)
        remaining = quota
        # Equal water filling is deterministic and capacity preserving.  If
        # fewer slots than contexts remain, use the best unresolved context
        # heads rather than silently favoring the lowest camera id.
        while remaining:
            active = torch.nonzero(
                allocations < context_counts, as_tuple=False
            ).flatten()
            if not len(active):
                break
            if remaining < len(active):
                ranked = active[
                    torch.topk(
                        context_score[active],
                        remaining,
                        sorted=True,
                    ).indices
                ]
                allocations[ranked] += 1
                remaining = 0
                break
            level = max(remaining // len(active), 1)
            addition = torch.minimum(
                context_counts[active] - allocations[active],
                torch.full_like(allocations[active], level),
            )
            assigned = int(addition.sum())
            allocations[active] += addition
            remaining -= assigned
            if assigned == 0:
                break
        selected_parts = []
        for context_offset in torch.nonzero(
            allocations > 0, as_tuple=False
        ).flatten():
            local = indices[inverse == context_offset]
            selected_parts.append(
                _balanced_instance_topk(
                    local,
                    score,
                    tree_instance_id,
                    int(allocations[context_offset]),
                    xyz=xyz,
                    spatial_cell_size=spatial_cell_size,
                    lineage_family_id=lineage_family_id,
                )
            )
        selected = (
            torch.cat(selected_parts)
            if selected_parts
            else indices[:0]
        )
        if len(selected) < quota:
            remaining_indices = indices[
                ~torch.isin(indices, selected)
            ]
            extra = min(quota - len(selected), len(remaining_indices))
            if extra:
                remaining_score = torch.nan_to_num(
                    score[remaining_indices],
                    nan=-torch.inf,
                    neginf=-torch.inf,
                )
                selected = torch.cat(
                    [
                        selected,
                        remaining_indices[
                            torch.topk(remaining_score, extra).indices
                        ],
                    ]
                )
        return selected
    if lineage_family_id is not None:
        candidate_family = lineage_family_id[indices]
        families, inverse, family_counts = torch.unique(
            candidate_family,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        family_score = torch.full(
            (len(families),),
            -torch.inf,
            device=score.device,
            dtype=score.dtype,
        )
        safe_candidate_score = torch.nan_to_num(
            score[indices], nan=-torch.inf, neginf=-torch.inf
        )
        family_score.scatter_reduce_(
            0,
            inverse,
            safe_candidate_score,
            reduce="amax",
            include_self=True,
        )
        allocations = torch.zeros_like(family_counts)
        remaining = quota
        while remaining:
            active = torch.nonzero(
                allocations < family_counts, as_tuple=False
            ).flatten()
            if not len(active):
                break
            if remaining < len(active):
                ranked = active[
                    torch.topk(
                        family_score[active], remaining, sorted=True
                    ).indices
                ]
                allocations[ranked] += 1
                remaining = 0
                break
            level = max(remaining // len(active), 1)
            addition = torch.minimum(
                family_counts[active] - allocations[active],
                torch.full_like(allocations[active], level),
            )
            assigned = int(addition.sum())
            allocations[active] += addition
            remaining -= assigned
            if assigned == 0:
                break
        selected_parts = []
        for family_offset in torch.nonzero(
            allocations > 0, as_tuple=False
        ).flatten():
            local = indices[inverse == family_offset]
            selected_parts.append(
                _balanced_instance_topk(
                    local,
                    score,
                    tree_instance_id,
                    int(allocations[family_offset]),
                    xyz=xyz,
                    spatial_cell_size=spatial_cell_size,
                )
            )
        selected = (
            torch.cat(selected_parts) if selected_parts else indices[:0]
        )
        if len(selected) < quota:
            remaining_indices = indices[
                ~torch.isin(indices, selected)
            ]
            extra = min(quota - len(selected), len(remaining_indices))
            if extra:
                remaining_score = torch.nan_to_num(
                    score[remaining_indices],
                    nan=-torch.inf,
                    neginf=-torch.inf,
                )
                selected = torch.cat(
                    [
                        selected,
                        remaining_indices[
                            torch.topk(remaining_score, extra).indices
                        ],
                    ]
                )
        return selected
    candidate_score = torch.nan_to_num(
        score[indices], nan=-torch.inf, neginf=-torch.inf
    )
    instances = tree_instance_id[indices]
    if xyz is None:
        groups = instances
    else:
        cells = torch.floor(
            xyz[indices] / max(float(spatial_cell_size), 1e-4)
        ).to(torch.int64)
        _, groups = torch.unique(
            torch.cat(
                [instances.to(torch.int64)[:, None], cells], dim=1
            ),
            dim=0,
            sorted=True,
            return_inverse=True,
        )
    unique_groups = torch.unique(groups, sorted=True)
    heads = []
    per_group = max(1, quota // max(len(unique_groups), 1))
    for group in unique_groups:
        rows = torch.nonzero(
            groups == group, as_tuple=False
        ).flatten()
        take = min(per_group, len(rows))
        best = rows[torch.topk(candidate_score[rows], take).indices]
        heads.append(indices[best])
    heads = torch.cat(heads)
    if len(heads) >= quota:
        return heads[
            torch.topk(score[heads], quota).indices
        ]
    remaining = indices[~torch.isin(indices, heads)]
    extra = quota - len(heads)
    if extra:
        remaining_score = torch.nan_to_num(
            score[remaining], nan=-torch.inf, neginf=-torch.inf
        )
        heads = torch.cat(
            [
                heads,
                remaining[
                    torch.topk(remaining_score, extra).indices
                ],
            ]
        )
    return heads


def _evidence_adaptive_role_quotas(
    eligible_counts: dict[str, int],
    population_counts: dict[str, int],
    capacity: int,
    *,
    observable_counts: dict[str, int] | None = None,
    effective_eligible_mass: dict[str, float] | None = None,
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
    rendered contribution and role-specific observation support. Their
    optional effective mass additionally carries continuous spatial geometry
    authority; normalize that mass by the *observable population in this
    topology epoch*, then use deterministic capped weighted apportionment.
    This distinction matters for the per-camera dynamic branch: rows belonging
    to cameras not sampled in the last epoch are not evidence that the sampled
    owner views have no unresolved bandwidth, while a weak single-view depth
    posterior is not equivalent to a multi-view canonical lineage. Falling
    back to raw eligible counts preserves legacy callers. Every role remains
    eligible in every phase; there is no fixed role percentage or pass/fail
    gate.
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
        value < 0
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
        if effective_mass[role] > counts[role] + 1e-5:
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
            effective_mass[role] / float(observable[role])
            if observable[role] > 0
            else 0.0
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
    lineage_id = torch.where(uses_evidence_id, evidence_id, track_id)
    candidates = (
        (foliage.split_generation > 0)
        & (lineage_id != -1)
        & ~foliage.static_skeleton_mask
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
def _adapt_volume(
    args,
    foliage,
    stats,
    *,
    volume_budget: int | None = None,
    phase: str | None = None,
    split_capacity_scale: float = 1.0,
    camera_forward_lookup: torch.Tensor | None = None,
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
    contradiction_remove = remove.clone()
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
        - int(contradiction_remove.sum()),
        0,
    )
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
    )
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
    ) | (foliage.split_generation > 0)
    canonical_eligible = (
        evidence_eligible
        & foliage.canonical_crown_mask
        & (working_stats["radius"] >= args.volume_split_radius)
        & canonical_lineage_supported
    )
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
    effective_eligible_mass = {
        "static_skeleton": float(
            topology_authority[skeleton_eligible].sum()
        ),
        "canonical_crown": float(
            topology_authority[canonical_eligible].sum()
        ),
        "dynamic_leaf": float(
            topology_authority[dynamic_eligible].sum()
        ),
    }
    role_quotas = _evidence_adaptive_role_quotas(
        eligible_counts,
        population_counts,
        capacity,
        observable_counts=observable_counts,
        effective_eligible_mass=effective_eligible_mass,
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
                    else None
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
        )
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
        )
        final_to_old = prune_event["_new_to_old"]
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
        "capacity_reallocation_contract": VOLUME_REALLOCATION_CONTRACT,
        "topology_mutation_contract": VOLUME_TOPOLOGY_MUTATION_CONTRACT,
        "integrated_split_demand": float(integrated_demand),
        "requested_splits": int(requested_splits),
        "configured_split_limit": int(args.maximum_volume_splits),
        "effective_split_limit": effective_split_limit,
        "topology_ramp_scale": split_capacity_scale,
        "budget": int(effective_budget),
        "remaining": int(max(effective_budget - len(foliage), 0)),
        "role_split_parents": role_split_counts,
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
        "mean_topology_authority_by_eligible_role": {
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
        "selection_contract": (
            "continuous_screen_demand_then_lineage_safe_replace_retire_then_"
            "posterior_authority_weighted_observable_population_normalized_"
            "role_capacity_then_"
            "owner_context_instance_spatial_utility_then_"
            "lineage_family_then_screen_severity_adaptive_2_or_4_child_"
            "split_with_exact_ray_camera_plane_refinement"
        ),
        "phase": phase,
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


def _apply_mature_surface_gradient_policy(
    surface,
    policy: str,
    *,
    chart_surface=None,
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
    if policy not in {"joint", "appearance_only", "frozen"}:
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
    conditioned_enabled: bool,
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
        package = render_hybrid(
            view,
            surface,
            foliage,
            background=background,
            include_dynamic=False,
            volume_opacity_scale=(
                1.0 if conditioned_enabled else 0.0
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
    retained_checkpoint_iterations = set(
        args.retain_checkpoint_iterations
    )
    output = Path(dataset.model_path).resolve()
    evidence_store = load_evidence_store(args.evidence_store)
    if evidence_store.get("geometry_source") != "mast3r_only":
        raise RuntimeError(
            "Native hybrid Teacher v2 requires geometry_source=mast3r_only"
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
    surface_warmstart = (
        _validate_surface_warmstart(
            args.surface_warmstart_ply,
            args.surface_warmstart_manifest,
        )
        if args.surface_warmstart_ply is not None
        else None
    )
    if (
        args.training_profile == "hybrid_handoff_quality"
        and surface_warmstart is None
    ):
        raise RuntimeError(
            "hybrid_handoff_quality requires a validated native rigid "
            "surface handoff"
        )
    if (
        args.mature_handoff_surface_policy != "joint"
        and surface_warmstart is None
    ):
        raise RuntimeError(
            "A non-joint mature surface policy requires a validated rigid "
            "surface handoff"
        )
    if (
        args.mature_handoff_surface_policy != "joint"
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
    if evidence_store.get("geometry_source") == "mast3r_only":
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
        if int(
            foliage_initialization_audit.get(
                "sequence_local_dynamic_tracks", 0
            )
        ) <= 0:
            gate_failures.append("sequence-conditioned leaf branch is empty")
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

    foliage = VolumetricFoliageModel(
        dataset.sh_degree, dynamic_rank=args.dynamic_rank
    ).cuda()
    seed_payload = _load(Path(initialization["foliage_seed"]))
    if resume is None:
        foliage.initialize_from_volume_state(seed_payload)
        # Dynamic leaves already come from real sequence-local pointmaps.
        # Cloning canonical crown primitives created two co-located alpha
        # owners and the opaque green paint layer seen in conditioned views.
        dynamic_seed_count = int(foliage.dynamic_leaf_mask.sum())
    else:
        foliage.restore(resume["foliage"])
        dynamic_seed_count = int(resume["dynamic_seed_count"])
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
        max_cached_views=0,
    )
    geometry = OutdoorGeometryEvidence(args.evidence_store)
    foliage_rays = FoliageRayEvidence(seed_payload.get("ray_evidence"))
    if resume is not None:
        foliage_rays.restore_runtime_state(
            resume.get("foliage_ray_runtime_state")
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
    chart_atlas_optimizer = torch.optim.Adam(
        list(chart_surface.parameters()), lr=args.chart_atlas_lr
    )
    if (
        resume is not None
        and resume.get("chart_atlas_optimizer") is not None
    ):
        chart_atlas_optimizer.load_state_dict(
            resume["chart_atlas_optimizer"]
        )
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
        volume_optimizer.load_state_dict(resume["volume_optimizer"])
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
            _dynamic_enabled(
                step,
                args.phase_schedule_horizon,
                args.training_profile,
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
    schedule_hash = _schedule_digest(
        rgb_schedule,
        conditioned_schedule,
        geometry_schedule,
        topology_schedule,
    )
    training_contract = {
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
        "surface_ownership_contract": (
            SURFACE_OWNERSHIP_REPAIR_TARGET[
                "surface_ownership_contract"
            ]
            if (
                args.mature_handoff_surface_policy == "appearance_only"
                and args.surface_retirement_optical_mass_fraction_per_event
                > 0
            )
            else "surface_opacity_follows_mature_handoff_policy"
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
            "metric_depth_posterior_retained_in_position_covariance__"
            "visible_ewa_depth_axis_bounded_to_4x_middle_tangent_scale__"
            "tangent_colour_opacity_unchanged"
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
            "feature_lr": float(opt.feature_lr),
            "opacity_lr": float(opt.opacity_lr),
            "scaling_lr": float(opt.scaling_lr),
            "rotation_lr": float(opt.rotation_lr),
        },
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
            "learning_rate": float(args.chart_atlas_lr),
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
        "mask_lookup": _file_sha256(
            REPO_ROOT / "matcha/cambridge_masks.py"
        ),
        "training_evidence": _file_sha256(
            REPO_ROOT / "outdoor/training_evidence.py"
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
    }
    runtime_provenance = collect_runtime_provenance(
        REPO_ROOT,
        python_modules=(
            "scripts.train_unified_outdoor_teacher",
            "outdoor.hybrid_gaussian_renderer",
            "scene.gaussian_model",
            "diff_surfel_rasterization",
        ),
        extension_roots=(
            SURFEL_ROOT / "submodules/diff-surfel-rasterization",
            SURFEL_ROOT / "submodules/simple-knn",
        ),
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
            if (
                not performance_only
                and not trainer_repair
                and not volume_capacity_repair
                and not surface_ownership_repair
                and not volume_split_repair
                and not surface_split_repair
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
        if not args.allow_conditioned_schedule_repair_resume:
            raise RuntimeError("Resume camera schedules changed")
        saved_schedules = resume.get("schedules", {})
        unchanged_schedule_names = ("rgb", "geometry", "topology")
        changed_non_conditioned = [
            name
            for name in unchanged_schedule_names
            if not np.array_equal(
                np.asarray(saved_schedules.get(name, []), dtype=np.int64),
                {
                    "rgb": rgb_schedule,
                    "geometry": geometry_schedule,
                    "topology": topology_schedule,
                }[name],
            )
        ]
        if changed_non_conditioned:
            raise RuntimeError(
                "Conditioned schedule repair changed unrelated camera "
                f"schedules: {changed_non_conditioned}"
            )
        print(
            "Resuming with the coverage-preserving conditioned-camera "
            "schedule; RGB, geometry and topology schedules are identical."
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
            allow_surface_ownership_repair_migration=bool(
                args.allow_surface_ownership_repair_resume
            ),
            allow_v38_causal_repair_migration=bool(
                v38_causal_repair_resume
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
    geometry_audit_warnings: set[str] = set()
    progress = tqdm(
        range(start_step, args.iterations),
        initial=start_step,
        total=args.iterations,
        desc="unified outdoor teacher",
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
            ):
                requested.append(
                    views[int(conditioned_schedule[future_step])]
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
        foliage_active = _foliage_enabled(
            step, args.phase_schedule_horizon, args.training_profile
        )
        dynamic_active = _conditioned_branch_active(
            step, args.phase_schedule_horizon, args.training_profile
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
        package = render_hybrid(
            view,
            surface,
            foliage,
            background=background,
            include_dynamic=False,
            volume_opacity_scale=1.0 if foliage_active else 0.0,
            structural_trainable_start=0,
            audit_fields=torch.stack(
                [
                    task["p_canopy_core"],
                    task["p_rigid"],
                    canopy_topology_signal,
                ]
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
        if dynamic_active:
            # Once the conditioned branch is active, pixels that repeatedly
            # disagree in space/time no longer drag the canonical crown into
            # a broad average.  Sigma is detached here: uncertainty is trained
            # by its proper likelihood below and cannot reduce this loss by
            # simply inflating itself.
            sigma = appearance.spatial_uncertainty(
                view.image_name,
                (view.image_height, view.image_width),
            ).detach()
            canopy_confidence = (
                0.10 / (sigma[0] + 0.05)
            ).clamp(0.35, 1.0)
            sky_confidence = (
                0.10 / (sigma[1] + 0.05)
            ).clamp(0.25, 1.0)
            background_weight = background_weight * (
                task["p_rigid"] + task["p_sky"] * sky_confidence
            ).clamp(0, 1)
            canopy_weight = canopy_weight * canopy_confidence
            static_confidence_mean = (
                canopy_confidence * task["p_canopy"]
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
        canonical_volume = (~foliage.dynamic_leaf_mask).to(
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
                ray_loss, ray_audit = foliage_rays.interval_factor(
                    view_by_camera_id[ray_camera_id],
                    foliage,
                    maximum_rays=effective_ray_posterior_maximum_rays,
                    maximum_candidates_per_ray=(
                        args.ray_posterior_maximum_candidates
                    ),
                    sample_update=ray_update,
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
            position_nll * position_weight
        ).mean()
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
        # Rigid/sky RGB, geometry and ownership update their normal owners.
        # Canopy RGB is differentiated only with respect to the foliage
        # branch.  This keeps the exact same jointly sorted image formation,
        # but prevents tree pixels from painting structural surfels or sky.
        canonical_loss.backward(retain_graph=foliage_active)
        if foliage_active:
            torch.autograd.backward(
                canopy_photo,
                inputs=tuple(foliage.parameters()),
            )
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
        _accumulate_volume_stats(volume_stats, package)

        conditioned_loss = canonical_loss.new_zeros(())
        conditioned_counterfactual_transparency = canonical_loss.new_zeros(())
        conditioned_counterfactual_audit = {
            "contract": COUNTERFACTUAL_TRANSPARENCY_CONTRACT,
            "scheduled": False,
            "supported_pixels": 0,
            "mean_responsibility": 0.0,
            "mean_relative_rgb_advantage": 0.0,
            "mean_volume_alpha_on_support": 0.0,
        }
        uncertainty_loss = canonical_loss.new_zeros(())
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
                inputs=tuple(appearance.parameters()),
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
            has_pointmap_factor = (
                Path(str(geometry_view.image_name)).stem
                in geometry.pointmap_records
            )
            if source_fields or has_pointmap_factor:
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
        surface_spatial_confidence_audit = (
            _apply_surface_spatial_confidence_gradients(surface)
        )
        mature_surface_policy_audit = (
            _apply_mature_surface_gradient_policy(
                surface,
                args.mature_handoff_surface_policy,
                chart_surface=chart_surface,
            )
        )
        surface.optimizer.step()
        volume_optimizer.step()
        if args.mature_handoff_surface_policy == "joint":
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
            if args.mature_handoff_surface_policy == "joint":
                surface._opacity.clamp_(-12, 4)
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
            if args.mature_handoff_surface_policy == "joint":
                structural_radii = package.radii[
                    : package.structural_count
                ].float()
                oversized = (
                    structural_radii
                    > float(args.maximum_surface_radius_pixels)
                )
                if bool(oversized.any()):
                    shrink = (
                        float(args.maximum_surface_radius_pixels)
                        / structural_radii[oversized].clamp_min(1.0)
                    ).clamp(0.1, 1.0)
                    surface._scaling[oversized] += shrink.log()[:, None]
                surface_scale_ceiling = surface._scaling.new_tensor(
                    float(args.maximum_surface_scale)
                ).log()
                surface._scaling.clamp_(max=surface_scale_ceiling)
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
            oversized_volume = (
                volume_radii
                > float(args.maximum_volume_radius_pixels)
            )
            if bool(oversized_volume.any()):
                # Apply a gradual screen-space correction. The recorded large
                # radius remains in volume_stats and is therefore prioritized
                # by the next adaptive split; the correction prevents it from
                # becoming a 100-pixel paint splat while waiting for that
                # topology event.
                volume_shrink = torch.sqrt(
                    float(args.maximum_volume_radius_pixels)
                    / volume_radii[oversized_volume].clamp_min(1.0)
                ).clamp(0.50, 1.0)
                foliage.log_scales[oversized_volume] += (
                    volume_shrink.log()[:, None]
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
                chart_growth_budget = int(
                    round(
                        args.maximum_surface_growth_per_event
                        * args.chart_quadtree_growth_fraction
                    )
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
                        if _chart_topology_active(step, args)
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
            event = _adapt_volume(
                args,
                foliage,
                volume_stats,
                volume_budget=volume_budget,
                phase=phase,
                split_capacity_scale=_volume_topology_ramp_scale(
                    step, args
                ),
                camera_forward_lookup=camera_forward_lookup,
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
            + conditioned_loss.detach()
            + args.counterfactual_transparency_weight
            * conditioned_counterfactual_transparency.detach()
            + geometry_loss.detach()
            + 0.20 * topology_loss.detach()
        )
        if step == 0 or (step + 1) % args.log_every == 0:
            row = {
                "iteration": step + 1,
                "phase": phase,
                "loss": float(total_loss),
                "canonical_photo": float(photo.detach()),
                "canonical_canopy_photo": float(
                    canopy_photo.detach()
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
                },
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
                phase=phase,
                loss=f"{float(total_loss):.4f}",
                surface=len(surface.get_xyz),
                volume=len(foliage),
            )
        iteration = step + 1
        retain_checkpoint = iteration in retained_checkpoint_iterations
        if (
            args.checkpoint_every > 0
            and iteration % args.checkpoint_every == 0
        ) or retain_checkpoint:
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
                    },
                    "conditioned_visit_counts": (
                        conditioned_visit_counts.copy()
                    ),
                    "conditioned_visit_provenance": (
                        conditioned_visit_provenance
                    ),
                    "implementation_hashes": implementation_hashes,
                    "runtime_provenance": runtime_provenance,
                    "phase": phase,
                    "surface": surface.capture(),
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
    # Export is a geometric model, not an optimizer graveyard. Physically
    # remove proven replacements and unprotected transparent descendants so
    # downstream point-cloud tools do not expose retired floaters as vertices.
    with torch.no_grad():
        # Use the same continuous observation-maturity contract as online
        # topology.  A final unconditional alpha threshold otherwise deletes
        # the DAV2 hole witnesses that online culling correctly retained long
        # enough to receive representative camera evidence.
        final_remove, final_cull_maturity = (
            surface._evidence_mature_opacity_prune_mask(
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
                "source_aware_continuous_observation_maturity"
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
            },
            "conditioned_visit_counts": (
                conditioned_visit_counts.copy()
            ),
            "conditioned_visit_provenance": (
                conditioned_visit_provenance
            ),
            "implementation_hashes": implementation_hashes,
            "runtime_provenance": runtime_provenance,
            "surface": surface.capture(),
            "surface_audit": surface_audit,
            "surface_warmstart": surface_warmstart,
            "chart_surface": chart_surface.capture(),
            "foliage": foliage.capture(),
            "foliage_ray_evidence_audit": foliage_rays.audit(),
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
        conditioned_enabled=(
            args.training_profile != "hybrid_rigid_stage1"
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
        "targeted_evaluation": evaluation,
        "evaluation_contract": {
            "valid_render_modes": (
                ["canonical"]
                if args.training_profile == "hybrid_rigid_stage1"
                else ["canonical", "conditioned"]
            ),
            "conditioned_valid": (
                args.training_profile != "hybrid_rigid_stage1"
            ),
            "canopy_quality_valid": (
                args.training_profile != "hybrid_rigid_stage1"
            ),
        },
        "historical_parent_ply_used": False,
        "colmap_points_or_tracks_used": False,
        "all_real_rgb_from_iteration_one": True,
        "canopy_surface_topology_gradient": False,
        "canopy_volume_topology_gradient": True,
        "elapsed_sec": time.time() - started,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    renderer_manifest = {
        "version": "native-hybrid-teacher-renderer-manifest-v1",
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
        },
        "conditioned_render": {
            "available": (
                args.training_profile != "hybrid_rigid_stage1"
            ),
            "include_dynamic": (
                args.training_profile != "hybrid_rigid_stage1"
            ),
            "temporal_code": "known database image name",
            "appearance_conditioning": True,
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
