#!/usr/bin/env python
"""Rebuild only foliage after the native rigid 2DGS stage is trained."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SURFEL_ROOT = REPO_ROOT / "2d-gaussian-splatting"
sys.path[:0] = [str(REPO_ROOT), str(SURFEL_ROOT)]

from outdoor.evidence_store import load_evidence_store  # noqa: E402
from outdoor.moge3_evidence import atomic_write_json  # noqa: E402
from outdoor.role_aware_initialization import (  # noqa: E402
    RIGID_CALIBRATED_INITIALIZATION_VERSION,
    build_foliage_seed,
)
from outdoor.runtime_provenance import (  # noqa: E402
    collect_runtime_provenance,
)
from outdoor.scene_contract import sha256_file  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-initialization", type=Path, required=True)
    parser.add_argument("--evidence-store", type=Path, required=True)
    parser.add_argument("--rigid-calibration-ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path)
    parser.add_argument("--maximum-foliage-voxels", type=int, default=400_000)
    parser.add_argument("--selected-foliage-views", type=int, default=0)
    parser.add_argument(
        "--maximum-dense-rays-per-foliage-view",
        type=int,
        default=2_048,
    )
    parser.add_argument("--maximum-dense-rays-total", type=int)
    parser.add_argument(
        "--minimum-dense-rays-per-foliage-view",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--maximum-bound-rays-per-foliage-view",
        type=int,
        default=8_192,
    )
    parser.add_argument(
        "--maximum-dynamic-births-per-foliage-view",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--maximum-weak-continuous-dynamic-births-per-foliage-view",
        type=int,
    )
    parser.add_argument(
        "--dynamic-birth-target-source-pixels-per-basis",
        type=float,
        default=None,
        help=(
            "Optional source-raster tree-pixel bandwidth target. Accepted "
            "views receive ceil(eligible_pixels / target) local births, "
            "bounded by the ordinary and adaptive per-view caps."
        ),
    )
    parser.add_argument("--voxel-size", type=float, default=0.12)
    parser.add_argument(
        "--maximum-sfm-static-tree-tracks", type=int, default=120_000
    )
    parser.add_argument(
        "--sfm-tree-coverage-radius", type=float, default=0.018
    )
    parser.add_argument(
        "--rigid-calibration-resolution-scale",
        type=float,
        default=0.125,
    )
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _full_foliage_initialization_contract(
    base_contract: dict,
    *,
    foliage_path: Path,
) -> dict:
    """Retire the rigid-stage placeholder after a real foliage rebuild.

    The surface seed is intentionally inherited from the rigid-only stage,
    but its inert foliage row is not.  Copying the base contract verbatim
    leaves ``rigid_stage_placeholder_foliage`` true and makes the mixed
    trainer correctly reject an otherwise complete visual-hull/ray-posterior
    initialization.  Make that ownership transition explicit and auditable.
    """
    contract = dict(base_contract)
    source_was_placeholder = bool(
        contract.get("rigid_stage_placeholder_foliage", False)
    )
    contract["foliage_reuse"] = {
        "policy": (
            "full_rigid_depth_calibrated_visual_hull_ray_posterior_rebuild"
        ),
        "foliage_seed_sha256": sha256_file(foliage_path),
        "mixed_training_eligible": True,
        "source_placeholder_retired": source_was_placeholder,
    }
    contract["rigid_stage_placeholder_foliage"] = False
    return contract


def main() -> None:
    args = _parse_args()
    runtime_provenance = collect_runtime_provenance(
        REPO_ROOT,
        python_modules=(
            "scripts.rebuild_foliage_with_rigid_depth",
            "outdoor.role_aware_initialization",
            "outdoor.foliage_geometry",
            "outdoor.rigid_occlusion",
            "outdoor.blue_noise_sampling",
        ),
    )
    print(
        json.dumps(
            {"runtime_provenance": runtime_provenance}, indent=2
        ),
        flush=True,
    )
    base = args.base_initialization.expanduser().resolve()
    base_manifest_path = base / "initialization_manifest.json"
    if not base_manifest_path.is_file():
        raise FileNotFoundError(base_manifest_path)
    base_manifest = json.loads(
        base_manifest_path.read_text(encoding="utf-8")
    )
    store = load_evidence_store(args.evidence_store)
    if base_manifest.get("evidence_hash") != store["evidence_hash"]:
        raise RuntimeError(
            "Base initialization and Evidence Store differ"
        )
    if bool(base_manifest.get("historical_model_initialization", True)):
        raise RuntimeError(
            "Rigid-calibrated foliage cannot inherit historical model state"
        )
    surface_source = Path(
        base_manifest["surface_seed"]
    ).expanduser().resolve()
    surface_audit_source = surface_source.with_suffix(".json")
    if not surface_source.is_file() or not surface_audit_source.is_file():
        raise FileNotFoundError(
            "Base initialization has no complete surface seed"
        )
    rigid_ply = args.rigid_calibration_ply.expanduser().resolve()
    if not rigid_ply.is_file():
        raise FileNotFoundError(rigid_ply)
    rgb_root = (
        args.rgb_root.expanduser().resolve()
        if args.rgb_root is not None
        else Path(base_manifest["rgb_source"]["image_root"]).resolve()
    )
    if str(rgb_root) != str(
        Path(base_manifest["rgb_source"]["image_root"]).resolve()
    ):
        raise RuntimeError(
            "Rigid-calibrated foliage must use the base initialization's "
            "exact RGB target raster"
        )

    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.replace:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    surface_path = output / "surface_seed.npz"
    _link_or_copy(surface_source, surface_path)
    _link_or_copy(surface_audit_source, surface_path.with_suffix(".json"))

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
        maximum_sfm_static_tree_tracks=(
            args.maximum_sfm_static_tree_tracks
        ),
        sfm_tree_coverage_radius=args.sfm_tree_coverage_radius,
        rigid_calibration_ply=rigid_ply,
        rigid_calibration_resolution_scale=(
            args.rigid_calibration_resolution_scale
        ),
        seed=args.seed,
    )
    rigid_contract = {
        "base_initialization_manifest": str(base_manifest_path),
        "base_initialization_manifest_sha256": sha256_file(
            base_manifest_path
        ),
        "rigid_calibration_ply": str(rigid_ply),
        "rigid_calibration_ply_sha256": sha256_file(rigid_ply),
        "rigid_calibration_resolution_scale": float(
            args.rigid_calibration_resolution_scale
        ),
        "role": (
            "chart_metric_alignment_and_rigid_occlusion_only__"
            "not_renderer_parameter_initialization"
        ),
    }
    initialization_contract = _full_foliage_initialization_contract(
        base_manifest.get("initialization_contract", {}),
        foliage_path=foliage_path,
    )
    initialization_contract.update(
        {
            "maximum_foliage_voxels": int(
                args.maximum_foliage_voxels
            ),
            "selected_foliage_views": int(
                args.selected_foliage_views
            ),
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
            "maximum_sfm_static_tree_tracks": int(
                args.maximum_sfm_static_tree_tracks
            ),
            "sfm_tree_coverage_radius": float(
                args.sfm_tree_coverage_radius
            ),
            "seed": int(args.seed),
            "rigid_depth_calibration": rigid_contract,
        }
    )
    manifest = {
        **base_manifest,
        "version": RIGID_CALIBRATED_INITIALIZATION_VERSION,
        "initialization_contract": initialization_contract,
        "surface_seed": str(surface_path),
        "foliage_seed": str(foliage_path),
        "foliage": foliage,
        "surface_resume": {
            "policy": (
                "immutable_evidence_surface_seed_reused_after_native_"
                "rigid_stage"
            ),
            "source_surface_seed": str(surface_source),
            "surface_seed_sha256": sha256_file(surface_source),
        },
        "historical_model_initialization": False,
        "causal_reuse": {
            "surface_only": True,
            "foliage_rebuilt": True,
            **rigid_contract,
        },
        "runtime_provenance": runtime_provenance,
    }
    manifest_path = output / "initialization_manifest.json"
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "version": manifest["version"],
                "output": str(output),
                "surface_seed_sha256": sha256_file(surface_path),
                "foliage_seed_sha256": sha256_file(foliage_path),
                "moge3_exact_k_front_hit": foliage.get(
                    "moge3_exact_k_front_hit"
                ),
                "legacy_dav2_depth_alignment": foliage.get(
                    "dav2_depth_alignment"
                ),
                "dense_ray_budget": foliage.get("dense_ray_budget"),
                "rigid_depth_calibration": foliage.get(
                    "rigid_depth_calibration"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
