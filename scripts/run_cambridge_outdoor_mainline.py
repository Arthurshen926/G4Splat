#!/usr/bin/env python3
"""One-command structural outdoor reconstruction for Cambridge Landmarks.

This runner intentionally composes the proven QC/MASt3R/2DGS infrastructure
instead of creating a second training stack.  It changes the mainline
contract: fixed calibrated cameras, task-specific semantics, broad candidates
followed by mandatory structural selection, inverse-depth fusion, all-real
topology growth and held-out evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outdoor.inverse_depth import fuse_inverse_depth_directory
from outdoor.scene_contract import build_scene_contract
from outdoor.structure_graph import build_static_structure_graph
from outdoor.structural_selection import select_structural_charts, validate_structural_selection
from outdoor.task_semantics import build_task_semantic_manifest
from scripts.run_cambridge_g4splat import (
    CambridgeG4Config,
    apply_active_chart_selection,
    evaluate_train_fit,
    prepare_audit,
    prepare_candidate_allowlist,
    prepare_qc,
    prepare_selection,
    run_logged,
    scene_paths,
    train_command,
    validate_scene,
    write_run_manifest,
)


MAINLINE_POLICY_VERSION = "cambridge-outdoor-structural-mainline-v1"


def build_mainline_config(args: argparse.Namespace) -> CambridgeG4Config:
    """Return the non-ablative defaults prescribed by the outdoor design."""
    return CambridgeG4Config(
        datasets_root=args.datasets_root,
        mask_root=args.mask_root,
        output_root=args.output_root,
        depth_checkpoint_dir=args.depth_checkpoint_dir,
        run_tag=args.run_tag,
        requested_charts=args.broad_charts,
        minimum_charts=args.min_charts,
        view_clusters=args.view_clusters,
        candidate_min_views_per_cluster=2,
        joint_min_views_per_cluster=2,
        gate_reserves_per_vulnerable_support=1,
        gate_failure_budget=1,
        # Cameras are a fixed task constraint, not a second SfM problem.
        strict_calibrated_poses=True,
        per_view_calibrated_intrinsics=True,
        rgb_supervision_profile="g4_tree_masked",
        rgb_sampling_policy="all_train_importance",
        chart_geometry_sampling_policy="active_only",
        # Only real dense views allocate split/prune topology.
        densification_view_policy="dense_only",
        # Sequence is a diversity signal in the broad pool, not a hard proxy
        # for final facade coverage.
        chart_sequence_coverage_mode="soft",
        min_global_plane_views=3,
        final_iterations=args.final_iterations,
        final_non_position_lr_decay_from=args.final_non_position_lr_decay_from,
        final_non_position_lr_final_mult=args.final_non_position_lr_final_mult,
        # See3D is not part of this real-view outdoor mainline.  Keep the
        # proposal knobs false as well as setting disable_see3d, so manifests
        # and child commands cannot imply a latent pseudo-view branch.
        scene_aligned_see3d_cameras=False,
        preserve_visible_see3d_render=False,
        disable_see3d=True,
        pseudo_initialization_mode="none",
        pseudo_geometry_mask_mode="none",
        pseudo_rgb_weight=0.0,
        pseudo_geometry_weight=0.0,
        pseudo_geometry_final_weight=0.0,
        joint_chart_count=None,
        require_post_gate_coverage=False,
    )


def _serialized_config(config: CambridgeG4Config) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _reuse_compatible_frontend(
    scene: str,
    config: CambridgeG4Config,
    paths: Any,
) -> Any:
    """Reuse an in-flight frontend whose only difference is legacy See3D knobs.

    The first real run of this branch began before the proposal knobs were
    made explicitly false.  Those values participate in the legacy run hash,
    although ``disable_see3d`` already prevented generation.  Match the
    immutable outdoor contract while intentionally ignoring just those two
    unreachable knobs, so later stages consume the completed fixed-camera
    alignment rather than spending another MASt3R frontend run.
    """
    runs_root = config.output_root / "runs"
    if not runs_root.is_dir():
        return paths
    expected = _serialized_config(config)
    ignored = {"scene_aligned_see3d_cameras", "preserve_visible_see3d_render"}
    pattern = (
        f"{scene}_g4_qc_n{config.requested_charts}_strictclean_planeonly_"
        "*_screen7k_v3/outdoor_mainline_manifest.json"
    )
    compatible: list[Path] = []
    for manifest_path in runs_root.glob(pattern):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("policy_version") != MAINLINE_POLICY_VERSION:
            continue
        recorded = payload.get("config")
        if not isinstance(recorded, dict):
            continue
        if all(
            recorded.get(key) == value
            for key, value in expected.items()
            if key not in ignored
        ):
            compatible.append(manifest_path.parent)
    if not compatible:
        return paths
    selected = max(compatible, key=lambda path: path.stat().st_mtime)
    if selected != paths.screen_output:
        print(
            "[INFO] Reusing compatible fixed-camera frontend from "
            f"{selected} (legacy See3D proposal identity ignored).",
            flush=True,
        )
    return replace(paths, screen_output=selected)


def _train_with_mainline_contract(command: list[str], *, semantic_manifest: Path, fused_depth: Path | None = None) -> list[str]:
    result = [
        *command,
        "--tree-missing-support-policy",
        "neutral",
        "--tree-neutral-support-value",
        "0.5",
        "--dense-view-sampling-policy",
        "spatial_block_balanced",
        "--dense-view-block-bins",
        "4",
        "--cambridge-task-semantic-policy",
        "outdoor_task_specific_v1",
        "--cambridge-task-semantic-manifest",
        str(semantic_manifest),
    ]
    if fused_depth is not None:
        result.extend(["--refine_depth_path_override", str(fused_depth)])
    return result


def _write_mainline_manifest(
    output: Path,
    *,
    scene: str,
    config: CambridgeG4Config,
    paths: Any,
    stage: str,
    scene_contract: Path,
    semantic_manifest: Path,
    structural_graph: Path | None = None,
    structural_selection: Path | None = None,
    fused_depth: Path | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_version": MAINLINE_POLICY_VERSION,
        "scene": scene,
        "stage": stage,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "camera_policy": "calibrated_fixed",
        "semantic_policy": "outdoor_task_specific_v1",
        "chart_policy": "hierarchical_structure_v1",
        "depth_policy": "inverse_depth_fusion_v1",
        "representation_policy": "hybrid_surface_volume_sky_v1",
        "densification_policy": "real_block_balanced_v1",
        "generation_policy": "none",
        "evaluation_policy": "trainfit_and_heldout_v1",
        "scene_contract": str(scene_contract),
        "task_semantic_manifest": str(semantic_manifest),
        "structural_graph": str(structural_graph) if structural_graph is not None else None,
        "structural_selection": str(structural_selection) if structural_selection is not None else None,
        "fused_inverse_depth": str(fused_depth) if fused_depth is not None else None,
        "dense_real_dataset": str(paths.qc_dataset),
        "query_images_excluded_from_training": True,
    }
    (output / "outdoor_mainline_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _run_heldout_evaluation(
    paths: Any,
    config: CambridgeG4Config,
    env: dict[str, str],
    *,
    scene: str,
    scene_contract: Path,
    output: Path,
    dry_run: bool,
) -> None:
    heldout_dataset = config.datasets_root / scene / "heldout_eval"
    if not heldout_dataset.is_dir():
        raise FileNotFoundError(
            "Held-out trajectory is required by the outdoor mainline but missing: "
            + str(heldout_dataset)
        )
    heldout_contract = paths.audit_dir.parent / "heldout_scene_manifest.json"
    if not heldout_contract.exists() and not dry_run:
        build_scene_contract(
            heldout_dataset,
            heldout_contract,
            mask_pickle=paths.mask_pickle,
            split="query_heldout",
        )
    evaluation_output = output / "evaluation" / "trajectory_heldout"
    command = [
        sys.executable,
        "scripts/evaluate_cambridge_heldout.py",
        "--model-path",
        str(output / "free_gaussians"),
        "--dataset-path",
        str(heldout_dataset),
        "--mask-pickle",
        str(paths.mask_pickle),
        "--iteration",
        str(config.final_iterations),
        "--output",
        str(evaluation_output),
        "--database-contract",
        str(scene_contract),
        "--heldout-contract",
        str(heldout_contract),
        "--gpu",
        env["CUDA_VISIBLE_DEVICES"],
    ]
    run_logged(
        command,
        cwd=REPO_ROOT,
        env=env,
        log_path=evaluation_output / "heldout_eval.log",
        dry_run=dry_run,
    )


def run_mainline(args: argparse.Namespace) -> None:
    config = build_mainline_config(args)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    conda_lib = str(Path(sys.prefix) / "lib")
    entries = [entry for entry in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if entry]
    if conda_lib not in entries:
        env["LD_LIBRARY_PATH"] = os.pathsep.join([conda_lib, *entries])
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Large Chart batches allocate many similarly sized reprojection tensors.
    # This does not change the objective, but avoids avoidable allocator
    # fragmentation when a restart must reuse the exact same frontend.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.dry_run:
        print(json.dumps({
            "policy_version": MAINLINE_POLICY_VERSION,
            "scene": args.scene,
            "phase": args.phase,
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
            "note": "Dry run validates the immutable mainline configuration without writing/reusing artifacts.",
        }, indent=2))
        return

    paths = scene_paths(args.scene, config)
    paths = _reuse_compatible_frontend(args.scene, config, paths)
    validate_scene(args.scene, paths)
    if paths.tree_mask_pickle is None:
        raise FileNotFoundError(
            "outdoor_task_specific_v1 requires an explicit masks_with_tree.pkl; "
            "do not silently switch semantic policies because a file is absent."
        )
    prepared = paths.audit_dir.parent
    scene_contract = prepared / "scene_manifest.json"
    semantic_manifest = prepared / "task_semantics_outdoor_v1.json"

    prepare_audit(paths, config, env, False)
    prepare_candidate_allowlist(paths, dry_run=False)
    prepare_qc(paths, env, False)
    if not scene_contract.is_file():
        build_scene_contract(paths.qc_dataset, scene_contract, mask_pickle=paths.mask_pickle)
    if not semantic_manifest.is_file():
        build_task_semantic_manifest(
            paths.qc_dataset,
            paths.mask_pickle,
            semantic_manifest,
            tree_mask_pickle=paths.tree_mask_pickle,
            tree_support_policy="neutral",
        )
    if args.phase == "prepare":
        print(f"[DONE] Prepared outdoor contracts: {prepared}")
        return

    selection = prepare_selection(paths, config, env, False, args.force_selection)
    gate_report = paths.screen_output / "mast3r_sfm" / "aligned_chart_conflict_gate.json"
    if not gate_report.is_file():
        frontend_command = train_command(
            paths,
            config,
            selection,
            screen_only=True,
            stop_after_alignment_gate=True,
        )
        _write_mainline_manifest(
            paths.screen_output,
            scene=args.scene,
            config=config,
            paths=paths,
            stage="frontend_gate",
            scene_contract=scene_contract,
            semantic_manifest=semantic_manifest,
        )
        run_logged(
            frontend_command,
            cwd=REPO_ROOT,
            env=env,
            log_path=paths.screen_output / "logs" / "outdoor_frontend_gate.log",
        )
    if not gate_report.is_file():
        raise FileNotFoundError(f"Frontend completed without an alignment gate: {gate_report}")
    if args.phase == "frontend":
        print(f"[DONE] Fixed-camera alignment gate: {gate_report}")
        return

    mast3r_scene = paths.screen_output / "mast3r_sfm"
    structural_graph = mast3r_scene / "static_structure_graph.npz"
    if not structural_graph.is_file():
        build_static_structure_graph(
            mast3r_scene,
            gate_report,
            paths.tree_mask_pickle,
            paths.qc_dataset,
            structural_graph,
            mask_indices=[0, 1, 2, 3],
            stride=args.structure_stride,
            bins=args.structure_voxel_bins,
            max_units=args.max_structure_units,
            min_unit_views=2,
            block_bins=args.block_bins,
            min_angle_degrees=args.min_triangulation_angle_degrees,
        )
    structural_selection = mast3r_scene / "quality_aware_selection" / "structural_chart_selection.json"
    if not structural_selection.is_file() or args.force_structural_selection:
        select_structural_charts(
            structural_graph,
            gate_report,
            paths.qc_dataset,
            structural_selection,
            base_selection=paths.selection_json,
            min_charts=args.min_charts,
            max_charts=args.max_final_charts,
            target_structural_coverage=args.target_structural_coverage,
            target_multiview_coverage=args.target_multiview_coverage,
            min_block_views=2,
            min_triangulation_angle_degrees=args.min_triangulation_angle_degrees,
            target_triangulation_angle_degrees=args.target_triangulation_angle_degrees,
        )
    structural_summary = validate_structural_selection(structural_selection, gate_report)
    apply_active_chart_selection(
        paths,
        paths.screen_output,
        structural_selection,
        env=env,
        dry_run=False,
    )
    _write_mainline_manifest(
        paths.screen_output,
        scene=args.scene,
        config=config,
        paths=paths,
        stage="structural_chart_selection",
        scene_contract=scene_contract,
        semantic_manifest=semantic_manifest,
        structural_graph=structural_graph,
        structural_selection=structural_selection,
    )
    print("[INFO] Structural Charts: " + json.dumps(structural_summary, sort_keys=True))
    if args.phase == "select":
        return

    plane_root = mast3r_scene / "plane-refine-depths"
    first_plane = plane_root / "refine_depth_frame000000.tiff"
    if not first_plane.is_file():
        plane_command = train_command(
            paths,
            config,
            selection,
            screen_only=True,
            continue_after_alignment=True,
        )
        plane_command = _train_with_mainline_contract(
            plane_command,
            semantic_manifest=semantic_manifest,
        )
        plane_command.append("--stop_after_plane_refinement")
        run_logged(
            plane_command,
            cwd=REPO_ROOT,
            env=env,
            log_path=paths.screen_output / "logs" / "outdoor_plane_refinement.log",
        )
    if not first_plane.is_file():
        raise FileNotFoundError(f"Plane stage did not produce {first_plane}")
    fused_depth = mast3r_scene / "inverse_depth_fusion"
    fusion_manifest = fused_depth / "inverse_depth_fusion_manifest.json"
    if not fusion_manifest.is_file():
        fuse_inverse_depth_directory(mast3r_scene, plane_root, fused_depth)
    _write_mainline_manifest(
        paths.screen_output,
        scene=args.scene,
        config=config,
        paths=paths,
        stage="inverse_depth_fusion",
        scene_contract=scene_contract,
        semantic_manifest=semantic_manifest,
        structural_graph=structural_graph,
        structural_selection=structural_selection,
        fused_depth=fused_depth,
    )
    if args.phase == "planes":
        return

    screen_model = paths.screen_output / "free_gaussians" / "point_cloud" / "iteration_7000" / "point_cloud.ply"
    if not screen_model.is_file():
        screen_command = train_command(
            paths,
            config,
            selection,
            screen_only=True,
            continue_after_plane=True,
        )
        screen_command = _train_with_mainline_contract(
            screen_command,
            semantic_manifest=semantic_manifest,
            fused_depth=fused_depth,
        )
        write_run_manifest(
            args.scene,
            paths,
            config,
            selection,
            screen_command,
            screen_only=True,
            active_chart_selection=structural_selection,
        )
        run_logged(
            screen_command,
            cwd=REPO_ROOT,
            env=env,
            log_path=paths.screen_output / "logs" / "outdoor_screen_train.log",
        )
    if not screen_model.is_file():
        raise FileNotFoundError(f"Screen training did not produce {screen_model}")
    if args.phase == "screen":
        return

    full_model = paths.full_output / "free_gaussians" / "point_cloud" / f"iteration_{config.final_iterations}" / "point_cloud.ply"
    if not full_model.is_file():
        if not paths.full_output.exists():
            paths.full_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(paths.screen_output, paths.full_output)
        full_command = train_command(
            paths,
            config,
            selection,
            screen_only=False,
            continue_after_initial=True,
        )
        full_command = _train_with_mainline_contract(
            full_command,
            semantic_manifest=semantic_manifest,
            fused_depth=fused_depth,
        )
        write_run_manifest(
            args.scene,
            paths,
            config,
            selection,
            full_command,
            screen_only=False,
            active_chart_selection=structural_selection,
        )
        _write_mainline_manifest(
            paths.full_output,
            scene=args.scene,
            config=config,
            paths=paths,
            stage="full_real_view_refinement",
            scene_contract=scene_contract,
            semantic_manifest=semantic_manifest,
            structural_graph=structural_graph,
            structural_selection=structural_selection,
            fused_depth=fused_depth,
        )
        run_logged(
            full_command,
            cwd=REPO_ROOT,
            env=env,
            log_path=paths.full_output / "logs" / "outdoor_full_train.log",
        )
    if not full_model.is_file():
        raise FileNotFoundError(f"Full training did not produce {full_model}")
    if args.phase == "full":
        return

    evaluate_train_fit(paths, config, env, screen_only=False, dry_run=False)
    _run_heldout_evaluation(
        paths,
        config,
        env,
        scene=args.scene,
        scene_contract=scene_contract,
        output=paths.full_output,
        dry_run=False,
    )
    print(f"[DONE] Outdoor structural mainline: {paths.full_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="StMarysChurch", choices=["GreatCourt", "KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch"])
    parser.add_argument(
        "--phase",
        choices=["prepare", "frontend", "select", "planes", "screen", "full", "evaluate", "mainline"],
        default="mainline",
    )
    parser.add_argument("--datasets-root", type=Path, default=Path("/root/MAtCha/output_cambridge/datasets_full"))
    parser.add_argument("--mask-root", type=Path, default=Path("/mnt/pool/sqy/Cambridge_stdloc"))
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/pool/sqy/G4Splat_runs/cambridge_outdoor_mainline_v1"))
    parser.add_argument("--depth-checkpoint-dir", type=Path, default=Path("/root/MAtCha/Depth-Anything-V2/checkpoints"))
    parser.add_argument("--run-tag", default="outdoorstructv1")
    parser.add_argument("--broad-charts", type=int, default=56)
    parser.add_argument("--min-charts", type=int, default=24)
    parser.add_argument("--max-final-charts", type=int, default=None)
    parser.add_argument("--view-clusters", type=int, default=8)
    parser.add_argument("--structure-stride", type=int, default=8)
    parser.add_argument("--structure-voxel-bins", type=int, default=40)
    parser.add_argument("--max-structure-units", type=int, default=4096)
    parser.add_argument("--block-bins", type=int, default=4)
    parser.add_argument("--min-triangulation-angle-degrees", type=float, default=1.5)
    parser.add_argument("--target-triangulation-angle-degrees", type=float, default=8.0)
    parser.add_argument("--target-structural-coverage", type=float, default=0.92)
    parser.add_argument("--target-multiview-coverage", type=float, default=0.70)
    parser.add_argument("--final-iterations", type=int, default=40_000)
    parser.add_argument("--final-non-position-lr-decay-from", type=int, default=30_000)
    parser.add_argument("--final-non-position-lr-final-mult", type=float, default=0.10)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--force-selection", action="store_true")
    parser.add_argument("--force-structural-selection", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.broad_charts < args.min_charts:
        raise ValueError("--broad-charts must be at least --min-charts")
    if args.max_final_charts is not None and not args.min_charts <= args.max_final_charts <= args.broad_charts:
        raise ValueError("--max-final-charts must lie in [min-charts, broad-charts]")
    if args.final_iterations <= 0:
        raise ValueError("--final-iterations must be positive")
    return args


if __name__ == "__main__":
    run_mainline(parse_args())
