#!/usr/bin/env python
"""Create disjoint rigid-surface and layered-foliage seeds from evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SURFEL_ROOT = REPO_ROOT / "2d-gaussian-splatting"
sys.path[:0] = [str(REPO_ROOT), str(SURFEL_ROOT)]

from outdoor.evidence_store import load_evidence_store  # noqa: E402
from outdoor.lazy_scene import rgb_source_contract  # noqa: E402
from outdoor.role_aware_initialization import (  # noqa: E402
    INITIALIZATION_VERSION,
    build_foliage_seed,
    build_surface_seed,
)
from outdoor.scene_contract import sha256_file  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rgb-root",
        type=Path,
        help=(
            "Immutable RGB target raster used by teacher training. Defaults "
            "to DATASET/images; pass the teacher's --images path when using "
            "an external resized target."
        ),
    )
    parser.add_argument("--maximum-chart-seeds", type=int, default=120_000)
    parser.add_argument("--chart-seeds-per-view", type=int, default=3000)
    parser.add_argument(
        "--mast3r-pointmap-seeds-per-view", type=int, default=6000
    )
    parser.add_argument(
        "--maximum-mast3r-pointmap-seeds", type=int, default=240_000
    )
    parser.add_argument(
        "--mast3r-pointmap-minimum-confidence",
        type=float,
        default=1.25,
    )
    parser.add_argument(
        "--mast3r-cross-sequence-radius", type=float, default=0.15
    )
    parser.add_argument(
        "--mast3r-pointmap-voxel-size", type=float, default=0.018
    )
    parser.add_argument(
        "--mast3r-maximum-single-sequence-seed-fraction",
        type=float,
        default=0.25,
        help=(
            "Maximum fraction of the dense pointmap seed budget that may "
            "lack support from an independent Cambridge traversal."
        ),
    )
    parser.add_argument(
        "--maximum-dav2-rigid-seeds",
        type=int,
        default=0,
        help=(
            "Maximum low-opacity rigid hole-completion births obtained by "
            "calibrating DAV2 to the MASt3R/MAtCha metric scaffold."
        ),
    )
    parser.add_argument(
        "--dav2-rigid-selected-views",
        type=int,
        default=0,
        help=(
            "Number of sequence/pose-diverse real cameras used for DAV2 "
            "rigid hole completion."
        ),
    )
    parser.add_argument(
        "--dav2-rigid-seeds-per-view", type=int, default=256
    )
    parser.add_argument(
        "--dav2-rigid-cross-sequence-radius",
        type=float,
        default=0.25,
    )
    parser.add_argument("--maximum-foliage-voxels", type=int, default=400_000)
    parser.add_argument(
        "--selected-foliage-views",
        type=int,
        default=64,
        help=(
            "Number of sequence/pose-diverse cameras used by the foliage "
            "posterior. Use 0 for every fixed database camera; the global "
            "dense-ray budget still controls ownerless observation-space "
            "evidence cardinality."
        ),
    )
    parser.add_argument(
        "--maximum-dense-rays-per-foliage-view",
        type=int,
        default=16_384,
        help=(
            "Maximum calibrated observation-space foliage rays per selected "
            "view. This controls sequence-local leaf bandwidth independently "
            "of the canonical visual-hull voxel budget."
        ),
    )
    parser.add_argument(
        "--maximum-dense-rays-total",
        type=int,
        default=None,
        help=(
            "Optional global foliage-ray budget shared by all selected views. "
            "When set, preserves camera coverage while allocating more rays to "
            "views with greater measured tree-pixel support."
        ),
    )
    parser.add_argument(
        "--minimum-dense-rays-per-foliage-view",
        type=int,
        default=0,
        help=(
            "Per-view coverage floor used with --maximum-dense-rays-total."
        ),
    )
    parser.add_argument(
        "--maximum-bound-rays-per-foliage-view",
        type=int,
        default=8192,
        help=(
            "Maximum persistent canonical candidate-bound interval factors "
            "per selected camera. Visual-hull verification still uses every "
            "candidate/camera projection; this bounds only the finite "
            "runtime factor basis."
        ),
    )
    parser.add_argument(
        "--maximum-dynamic-births-per-foliage-view",
        type=int,
        default=1024,
        help=(
            "Maximum sequence-local renderer basis size per selected view. "
            "The complete dense-ray posterior remains external evidence; "
            "nearby rays share finite-footprint births and adaptive splits."
        ),
    )
    parser.add_argument(
        "--maximum-weak-continuous-dynamic-births-per-foliage-view",
        type=int,
        default=None,
        help=(
            "Optional larger renderer-basis cap for views whose observed "
            "foliage tracks only have a weak-continuous DAV2 posterior. "
            "Metric geometry confidence remains weak; this only restores "
            "the optical bandwidth needed to explain measured tree pixels."
        ),
    )
    parser.add_argument(
        "--dynamic-birth-target-source-pixels-per-basis",
        type=float,
        default=None,
        help=(
            "Optional continuous local renderer-bandwidth target measured "
            "in eligible source-raster tree pixels per dynamic basis row."
        ),
    )
    parser.add_argument("--voxel-size", type=float, default=0.12)
    parser.add_argument(
        "--rigid-calibration-ply",
        type=Path,
        help=(
            "Optional trained native rigid 2DGS surface used only to render "
            "per-camera depth for DAV2 metric alignment and foliage "
            "occlusion. It is not copied into the mixed initialization."
        ),
    )
    parser.add_argument(
        "--rigid-calibration-resolution-scale",
        type=float,
        default=0.125,
    )
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument(
        "--reuse-foliage-initialization",
        type=Path,
        help=(
            "Reuse an immutable foliage seed from another initialization "
            "with the same Evidence Store and exact RGB contract. This is "
            "intended for causal surface-front-end repairs; provenance and "
            "content hashes are persisted in the new manifest."
        ),
    )
    parser.add_argument(
        "--rigid-stage-placeholder-foliage",
        action="store_true",
        help=(
            "Write one inert, explicitly non-deployable foliage row for a "
            "hybrid_rigid_stage1 run. That profile disables the volume "
            "renderer and all foliage losses, so building the expensive "
            "visual hull would not affect its surface optimization. Mixed "
            "training must use a normal initialization without this flag."
        ),
    )
    parser.add_argument(
        "--resume-existing-surface",
        action="store_true",
        help=(
            "Resume an interrupted initialization after surface_seed.npz "
            "and surface_seed.json were written. The seed must use the "
            "current protocol and the same Evidence Store; foliage and the "
            "final manifest are rebuilt normally."
        ),
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def _validated_reused_foliage(
    initialization_path: Path,
    *,
    evidence_hash: str,
    rgb_contract: dict,
) -> tuple[Path, dict, dict]:
    initialization_path = initialization_path.expanduser().resolve()
    manifest_path = (
        initialization_path / "initialization_manifest.json"
        if initialization_path.is_dir()
        else initialization_path
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("evidence_hash") != evidence_hash:
        raise RuntimeError(
            "Reused foliage initialization has a different Evidence Store"
        )
    if bool(payload.get("historical_model_initialization", True)):
        raise RuntimeError(
            "Historical-model foliage initialization cannot be reused"
        )
    foliage = payload.get("foliage")
    if not isinstance(foliage, dict):
        raise RuntimeError("Reused initialization has no foliage audit")
    if foliage.get("evidence_hash") != evidence_hash:
        raise RuntimeError("Reused foliage seed/evidence hash mismatch")
    if foliage.get("geometry_source") != "mast3r_only":
        raise RuntimeError("Reused foliage is not MASt3R-only")
    if bool(foliage.get("historical_trained_ply_used", True)):
        raise RuntimeError("Reused foliage used a historical trained PLY")
    prior_rgb = payload.get("rgb_source")
    identity_fields = (
        "image_root",
        "image_count",
        "name_set_sha256",
        "content_mapping_sha256",
        "canonical_image_size_wh",
    )
    mismatched = [
        name
        for name in identity_fields
        if prior_rgb is None
        or prior_rgb.get(name) != rgb_contract.get(name)
    ]
    if mismatched:
        raise RuntimeError(
            "Reused foliage RGB source differs from the requested training "
            "target: " + ", ".join(mismatched)
        )
    foliage_path = Path(payload["foliage_seed"]).expanduser().resolve()
    if not foliage_path.is_file():
        raise FileNotFoundError(foliage_path)
    provenance = {
        "policy": "immutable_same_evidence_same_rgb_foliage_reuse",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "foliage_seed_sha256": sha256_file(foliage_path),
        "surface_front_end_is_only_changed_variable": True,
    }
    return foliage_path, foliage, provenance


def _write_rigid_stage_placeholder_foliage(
    output_path: Path, *, evidence_hash: str
) -> dict:
    """Write a schema-valid but optically inert rigid-stage placeholder.

    The placeholder exists only because the unified checkpoint schema always
    carries both branches.  Its provenance deliberately fails the normal
    foliage-quality contract and must never be reused by a mixed trainer.
    """
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.zeros((1, 3), dtype=torch.float32),
        "scales": torch.full((1, 3), 1.0e-3, dtype=torch.float32),
        "quaternions": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        ),
        "colors": torch.zeros((1, 3), dtype=torch.float32),
        "opacities": torch.full((1, 1), 1.0e-6, dtype=torch.float32),
        "primitive_role": torch.zeros(1, dtype=torch.int8),
        "layer_role": torch.zeros(1, dtype=torch.int8),
        "tree_instance_id": torch.full((1,), -1, dtype=torch.int32),
        "support_camera_ids": torch.full((1, 0), -1, dtype=torch.int32),
        "replacement_group": torch.zeros(1, dtype=torch.int64),
        "ray_evidence": {"depth_coordinate": "camera_z"},
        "audit": {
            "protocol": INITIALIZATION_VERSION,
            "evidence_hash": evidence_hash,
            "geometry_source": "none_inert_rigid_stage_placeholder",
            "selected_views": [],
            "rigid_stage_only": True,
            "mixed_training_eligible": False,
            "deployment_eligible": False,
        },
    }
    torch.save(payload, output_path)
    return {
        "version": "rigid-stage-inert-foliage-placeholder-v1",
        "evidence_hash": evidence_hash,
        "geometry_source": "none_inert_rigid_stage_placeholder",
        "historical_trained_ply_used": False,
        "all_real_rgb_initialization_used": False,
        "colmap_points_or_tracks_used": False,
        "seed_count": 1,
        "optical_opacity": 1.0e-6,
        "rigid_stage_only": True,
        "mixed_training_eligible": False,
        "deployment_eligible": False,
    }


def main() -> None:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    if args.resume_existing_surface and args.replace:
        raise ValueError(
            "--resume-existing-surface and --replace are mutually exclusive"
        )
    if (
        args.rigid_stage_placeholder_foliage
        and args.reuse_foliage_initialization is not None
    ):
        raise ValueError(
            "--rigid-stage-placeholder-foliage and "
            "--reuse-foliage-initialization are mutually exclusive"
        )
    if output.exists() and not args.resume_existing_surface:
        if not args.replace:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=args.resume_existing_surface)
    store = load_evidence_store(args.evidence_store)
    dataset = Path(store["dataset"]).expanduser().resolve()
    rgb_root = (
        args.rgb_root.expanduser().resolve()
        if args.rgb_root is not None
        else (dataset / "images").resolve()
    )
    rgb_contract = rgb_source_contract(
        SimpleNamespace(source_path=str(dataset), images=str(rgb_root))
    )
    surface_path = output / "surface_seed.npz"
    surface_resume = None
    if args.resume_existing_surface:
        surface_audit_path = surface_path.with_suffix(".json")
        if not surface_path.is_file() or not surface_audit_path.is_file():
            raise FileNotFoundError(
                "Interrupted initialization has no complete surface seed"
            )
        surface = json.loads(
            surface_audit_path.read_text(encoding="utf-8")
        )
        if (
            surface.get("version") != INITIALIZATION_VERSION
            or surface.get("evidence_hash") != store["evidence_hash"]
        ):
            raise RuntimeError(
                "Interrupted surface seed does not match the current "
                "initialization/evidence contract"
            )
        forbidden = {
            "historical_trained_ply_used": False,
            "all_real_rgb_initialization_used": False,
            "colmap_points_or_tracks_used": False,
            "geometry_source": "mast3r_only",
        }
        mismatched = [
            name
            for name, expected in forbidden.items()
            if surface.get(name) != expected
        ]
        if mismatched:
            raise RuntimeError(
                "Interrupted surface seed violates the no-history/no-COLMAP "
                "contract: " + ", ".join(mismatched)
            )
        surface_resume = {
            "policy": "validated_current_protocol_interrupted_surface_resume",
            "surface_seed_sha256": sha256_file(surface_path),
            "surface_audit_sha256": sha256_file(surface_audit_path),
        }
    else:
        surface = build_surface_seed(
            args.evidence_store,
            surface_path,
            rgb_root=rgb_root,
            chart_seeds_per_view=args.chart_seeds_per_view,
            maximum_chart_seeds=args.maximum_chart_seeds,
            mast3r_pointmap_seeds_per_view=(
                args.mast3r_pointmap_seeds_per_view
            ),
            maximum_mast3r_pointmap_seeds=(
                args.maximum_mast3r_pointmap_seeds
            ),
            mast3r_pointmap_minimum_confidence=(
                args.mast3r_pointmap_minimum_confidence
            ),
            mast3r_cross_sequence_radius=args.mast3r_cross_sequence_radius,
            mast3r_pointmap_voxel_size=args.mast3r_pointmap_voxel_size,
            mast3r_maximum_single_sequence_fraction=(
                args.mast3r_maximum_single_sequence_seed_fraction
            ),
            maximum_dav2_rigid_seeds=args.maximum_dav2_rigid_seeds,
            dav2_rigid_selected_views=args.dav2_rigid_selected_views,
            dav2_rigid_seeds_per_view=args.dav2_rigid_seeds_per_view,
            dav2_rigid_cross_sequence_radius=(
                args.dav2_rigid_cross_sequence_radius
            ),
            seed=args.seed,
        )
    foliage_reuse = None
    if args.rigid_stage_placeholder_foliage:
        foliage_path = output / "foliage_seed_gaussians.pth"
        foliage = _write_rigid_stage_placeholder_foliage(
            foliage_path, evidence_hash=store["evidence_hash"]
        )
        foliage_reuse = {
            "policy": "inert_rigid_stage_schema_placeholder",
            "foliage_seed_sha256": sha256_file(foliage_path),
            "mixed_training_eligible": False,
        }
    elif args.reuse_foliage_initialization is None:
        foliage_path = output / "foliage_seed_gaussians.pth"
        foliage = build_foliage_seed(
            args.evidence_store,
            surface_path,
            foliage_path,
            rgb_root=rgb_root,
            voxel_size=args.voxel_size,
            maximum_voxels=args.maximum_foliage_voxels,
            selected_view_count=args.selected_foliage_views,
            maximum_dense_rays_per_view=(
                args.maximum_dense_rays_per_foliage_view
            ),
            maximum_dense_rays_total=args.maximum_dense_rays_total,
            minimum_dense_rays_per_view=(
                args.minimum_dense_rays_per_foliage_view
            ),
            maximum_bound_rays_per_view=(
                args.maximum_bound_rays_per_foliage_view
            ),
            maximum_dynamic_births_per_view=(
                args.maximum_dynamic_births_per_foliage_view
            ),
            maximum_weak_continuous_dynamic_births_per_view=(
                args.maximum_weak_continuous_dynamic_births_per_foliage_view
            ),
            dynamic_birth_target_source_pixels_per_basis=(
                args.dynamic_birth_target_source_pixels_per_basis
            ),
            rigid_calibration_ply=args.rigid_calibration_ply,
            rigid_calibration_resolution_scale=(
                args.rigid_calibration_resolution_scale
            ),
            seed=args.seed,
        )
    else:
        foliage_path, foliage, foliage_reuse = _validated_reused_foliage(
            args.reuse_foliage_initialization,
            evidence_hash=store["evidence_hash"],
            rgb_contract=rgb_contract,
        )
    manifest = {
        "version": INITIALIZATION_VERSION,
        "evidence_hash": store["evidence_hash"],
        "rgb_source": rgb_contract,
        "initialization_contract": {
            "rgb_root": str(rgb_root),
            "maximum_chart_seeds": int(args.maximum_chart_seeds),
            "chart_seeds_per_view": int(args.chart_seeds_per_view),
            "mast3r_pointmap_seeds_per_view": int(
                args.mast3r_pointmap_seeds_per_view
            ),
            "maximum_mast3r_pointmap_seeds": int(
                args.maximum_mast3r_pointmap_seeds
            ),
            "mast3r_pointmap_minimum_confidence": float(
                args.mast3r_pointmap_minimum_confidence
            ),
            "mast3r_cross_sequence_radius": float(
                args.mast3r_cross_sequence_radius
            ),
            "mast3r_pointmap_voxel_size": float(
                args.mast3r_pointmap_voxel_size
            ),
            "mast3r_maximum_single_sequence_seed_fraction": float(
                args.mast3r_maximum_single_sequence_seed_fraction
            ),
            "maximum_dav2_rigid_seeds": int(
                args.maximum_dav2_rigid_seeds
            ),
            "dav2_rigid_selected_views": int(
                args.dav2_rigid_selected_views
            ),
            "dav2_rigid_seeds_per_view": int(
                args.dav2_rigid_seeds_per_view
            ),
            "dav2_rigid_cross_sequence_radius": float(
                args.dav2_rigid_cross_sequence_radius
            ),
            "maximum_foliage_voxels": int(args.maximum_foliage_voxels),
            "selected_foliage_views": int(args.selected_foliage_views),
            "maximum_dense_rays_per_foliage_view": int(
                args.maximum_dense_rays_per_foliage_view
            ),
            "maximum_dense_rays_total": (
                None
                if args.maximum_dense_rays_total is None
                else int(args.maximum_dense_rays_total)
            ),
            "minimum_dense_rays_per_foliage_view": int(
                args.minimum_dense_rays_per_foliage_view
            ),
            "maximum_bound_rays_per_foliage_view": int(
                args.maximum_bound_rays_per_foliage_view
            ),
            "maximum_dynamic_births_per_foliage_view": int(
                args.maximum_dynamic_births_per_foliage_view
            ),
            "maximum_weak_continuous_dynamic_births_per_foliage_view": (
                None
                if args.maximum_weak_continuous_dynamic_births_per_foliage_view
                is None
                else int(
                    args.maximum_weak_continuous_dynamic_births_per_foliage_view
                )
            ),
            "dynamic_birth_target_source_pixels_per_basis": (
                None
                if args.dynamic_birth_target_source_pixels_per_basis is None
                else float(
                    args.dynamic_birth_target_source_pixels_per_basis
                )
            ),
            "voxel_size": float(args.voxel_size),
            "rigid_calibration_ply": (
                None
                if args.rigid_calibration_ply is None
                else str(
                    args.rigid_calibration_ply.expanduser().resolve()
                )
            ),
            "rigid_calibration_ply_sha256": (
                None
                if args.rigid_calibration_ply is None
                else sha256_file(
                    args.rigid_calibration_ply.expanduser().resolve()
                )
            ),
            "rigid_calibration_resolution_scale": float(
                args.rigid_calibration_resolution_scale
            ),
            "seed": int(args.seed),
            "foliage_reuse": foliage_reuse,
            "rigid_stage_placeholder_foliage": bool(
                args.rigid_stage_placeholder_foliage
            ),
        },
        "surface_seed": str(surface_path),
        "foliage_seed": str(foliage_path),
        "surface": surface,
        "surface_resume": surface_resume,
        "foliage": foliage,
        "ownership": {
            "surface_canopy_seed_count": 0,
            "trunk_branch": "static_skeleton_3dgs",
            "canonical_crown": "ray_depth_visual_hull_3dgs",
        },
        "historical_model_initialization": False,
        "causal_reuse": foliage_reuse,
    }
    (output / "initialization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
