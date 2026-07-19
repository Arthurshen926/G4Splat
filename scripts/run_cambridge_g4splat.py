#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from view_quality_control.poses import (  # noqa: E402
    COLMAP_POSE_POLICY_VERSION,
    VIEW_AUDIT_POLICY_VERSION,
)


DEFAULT_SCENES = [
    "GreatCourt",
    "KingsCollege",
    "OldHospital",
    "ShopFacade",
    "StMarysChurch",
]
RUN_POLICY_VERSION = "cambridge-g4-v12-all-train-tree-hard-crossview-plane-geometry-screen-schedule"
SELECTION_POLICY_VERSION = "target-kcenter-mask0-tree15-qc-v5-alltrain-candidate-feedback"


@dataclass(frozen=True)
class CambridgeG4Config:
    datasets_root: Path
    mask_root: Path
    output_root: Path
    depth_checkpoint_dir: Path
    run_tag: str = "base"
    requested_charts: int = 24
    minimum_charts: int = 16
    max_reject_fraction: float = 0.05
    view_clusters: int = 8
    min_views_per_cluster: int = 2
    min_baseline_ratio: float = 0.02
    min_global_pose_distance: float = 0.05
    max_borrowed_support_distance: float = 0.25
    min_sharpness: float = 1e-4
    max_extreme_ratio: float = 0.20
    max_shadow_ratio: float = 0.35
    min_lower_mean_intensity: float = 0.15
    resolution: int = 2
    rgb_loss_type: str = "l1"
    use_color_correction: bool = True
    color_correction_lr: float = 1e-3
    color_correction_reg: float = 1e-2
    white_background: bool = True
    semantic_alpha_weight: float = 0.01
    warp_downsample_pixel_grid_size: int = 2
    downweight_input_view_color_loss: bool = False
    scene_aligned_see3d_cameras: bool = True
    preserve_visible_see3d_render: bool = True
    disable_see3d: bool = True
    pseudo_initialization_mode: str = "none"
    pseudo_geometry_mask_mode: str = "none"
    pseudo_rgb_weight: float = 0.0
    pseudo_geometry_weight: float = 0.0
    pseudo_geometry_final_weight: float = 0.0
    pseudo_geometry_decay_until: int = 5000
    max_chart_tree_ratio: float = 0.15
    gate_aligned_chart_conflicts: bool = True
    aligned_chart_min_valid_fraction: float = 0.50
    aligned_chart_max_relative_p90: float = 0.50
    aligned_chart_max_gt25_fraction: float = 0.50
    gate_plane_refinement: bool = True
    plane_gate_min_support_fraction: float = 0.30
    plane_gate_max_relative_p90: float = 0.25
    plane_gate_max_gt25_fraction: float = 0.25
    plane_gate_max_pixel_relative_change: float = 0.50
    min_global_plane_views: int = 2
    # The 7k screen is a real capacity/schedule ablation, not an implicit
    # fixed constant.  Keep it in the run identity so a later full pass never
    # silently treats a differently densified screen PLY as the same gate.
    screen_free_gaussians_config: str = "default"
    final_iterations: int = 30_000
    final_non_position_lr_decay_from: int = 30_000
    final_non_position_lr_final_mult: float = 0.10


@dataclass(frozen=True)
class ScenePaths:
    full_dataset: Path
    fit_dataset: Path
    mask_pickle: Path
    tree_mask_pickle: Path | None
    audit_dir: Path
    audit_json: Path
    candidate_allowlist: Path
    qc_dataset: Path
    selection_json: Path
    semantic_cache: Path
    tree_ratio_cache: Path
    tree_support_dir: Path
    screen_output: Path
    full_output: Path


def _config_digest(
    config: CambridgeG4Config,
    *,
    include_final_schedule: bool = True,
) -> str:
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
        if key != "output_root"
    }
    if not include_final_schedule:
        for key in (
            "final_iterations",
            "final_non_position_lr_decay_from",
            "final_non_position_lr_final_mult",
        ):
            payload.pop(key, None)
    payload["run_policy_version"] = RUN_POLICY_VERSION
    # The alignment YAML controls model capacity and memory behavior.  Hash it
    # into the run identity so a frontend never silently reuses pointmaps or
    # charts produced with a different optimizer/memory implementation.
    alignment_config = REPO_ROOT / "configs" / "charts_alignment" / "semantic_masked_strong_lowmem.yaml"
    if alignment_config.is_file():
        payload["alignment_config_sha256"] = hashlib.sha256(
            alignment_config.read_bytes()
        ).hexdigest()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:8]


def _pseudo_policy_name(config: CambridgeG4Config) -> str:
    pseudo_disabled = (
        config.pseudo_initialization_mode == "none"
        and config.pseudo_geometry_mask_mode == "none"
        and config.pseudo_rgb_weight == 0.0
        and config.pseudo_geometry_weight == 0.0
        and config.pseudo_geometry_final_weight == 0.0
    )
    return "planeonly" if pseudo_disabled else "pseudo"


def scene_paths(scene: str, config: CambridgeG4Config) -> ScenePaths:
    prepared = config.output_root / "prepared" / scene
    tree_mask_pickle = config.mask_root / scene / "processed" / "masks_with_tree.pkl"
    if not tree_mask_pickle.exists():
        tree_mask_pickle = None
    run_stem = (
        f"{scene}_g4_qc_n{config.requested_charts}_strictclean_"
        f"{_pseudo_policy_name(config)}"
    )
    screen_stem = f"{run_stem}_{_config_digest(config, include_final_schedule=False)}"
    full_stem = f"{run_stem}_{_config_digest(config)}"
    final_label = (
        f"{config.final_iterations // 1000}k"
        if config.final_iterations % 1000 == 0
        else str(config.final_iterations)
    )
    return ScenePaths(
        # Cambridge reconstruction is deliberately transductive in this project:
        # every image under train contributes to QC, chart selection, and RGB/depth
        # supervision.  Do not silently revive the historical train_opt/heldout split.
        full_dataset=config.datasets_root / scene / "train",
        fit_dataset=config.datasets_root / scene / "train",
        mask_pickle=config.mask_root / scene / "processed" / "masks.pkl",
        tree_mask_pickle=tree_mask_pickle,
        audit_dir=prepared / "audit",
        audit_json=prepared / "audit" / "view_audit.json",
        candidate_allowlist=prepared / "chart_candidate_allowlist.txt",
        qc_dataset=prepared / "dataset_qc_tree_v5" / "train_all",
        selection_json=prepared / f"chart_selection_n{config.requested_charts}.json",
        semantic_cache=prepared / "mask0_invalid_ratio.json",
        tree_ratio_cache=prepared / "tree_invalid_ratio.json",
        tree_support_dir=prepared / "tree_support",
        screen_output=config.output_root / "runs" / f"{screen_stem}_screen7k_v3",
        full_output=config.output_root / "runs" / f"{full_stem}_full{final_label}_v3",
    )


def command_text(command: Iterable[object]) -> str:
    return " ".join(str(value) for value in command)


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path | None = None,
    capture: bool = False,
    dry_run: bool = False,
) -> subprocess.CompletedProcess:
    print(f"[CMD] {command_text(command)}", flush=True)
    if dry_run:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    if capture:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    if log_path is None:
        return subprocess.run(command, cwd=cwd, env=env, check=True)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[RUN] {command_text(command)}\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return subprocess.CompletedProcess(command, return_code)


def validate_scene(scene: str, paths: ScenePaths) -> None:
    missing = [
        path
        for path in (paths.full_dataset, paths.fit_dataset, paths.mask_pickle)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"{scene}: missing input(s): {', '.join(map(str, missing))}")
    required_dataset_files = [
        paths.full_dataset / "images",
        paths.full_dataset / "sparse" / "0" / "images.bin",
        paths.full_dataset / "name_mapping.json",
    ]
    missing_dataset = [path for path in required_dataset_files if not path.exists()]
    if missing_dataset:
        raise FileNotFoundError(
            f"{scene}: incomplete staged dataset: {', '.join(map(str, missing_dataset))}"
        )


def train_fit_paths(
    paths: ScenePaths,
    *,
    screen_only: bool,
    config: CambridgeG4Config | None = None,
) -> tuple[Path, Path, int]:
    output = paths.screen_output if screen_only else paths.full_output
    iteration = 7000 if screen_only else int(
        config.final_iterations if config is not None else 30_000
    )
    render_output = output / "evaluation" / "train_fit" / f"ours_{iteration}"
    metrics_output = render_output / "rgb_metrics.json"
    return render_output, metrics_output, iteration


def train_fit_render_command(paths: ScenePaths, config: CambridgeG4Config, *, screen_only: bool) -> list[str]:
    output = paths.screen_output if screen_only else paths.full_output
    render_output, _, iteration = train_fit_paths(
        paths,
        screen_only=screen_only,
        config=config,
    )
    return [
        sys.executable,
        "2d-gaussian-splatting/render.py",
        "-s",
        # Render against the exact normalized all-train camera set consumed by
        # dense reconstruction.  Rendering against the raw source would
        # silently reintroduce the handful of non-unit COLMAP quaternions that
        # QC corrected, so it is not an in-sample measurement of this model.
        str(paths.qc_dataset),
        "-m",
        str(output / "free_gaussians"),
        "--iteration",
        str(iteration),
        "--resolution",
        str(config.resolution),
        *(["--white_background"] if config.white_background else []),
        "--data_device",
        "cpu",
        "--rgb_only",
        "--skip_test",
        "--skip_mesh",
        "--output_dir",
        str(render_output),
    ]


def train_fit_metric_command(
    paths: ScenePaths,
    *,
    screen_only: bool,
    config: CambridgeG4Config | None = None,
) -> list[str]:
    render_output, metrics_output, _ = train_fit_paths(
        paths,
        screen_only=screen_only,
        config=config,
    )
    return [
        sys.executable,
        "scripts/evaluate_render_dir.py",
        str(render_output),
        "--dataset-path",
        str(paths.qc_dataset),
        "--mask-pickle",
        str(paths.mask_pickle),
        "--output",
        str(metrics_output),
        "--device",
        "cuda",
    ]


def evaluate_train_fit(
    paths: ScenePaths,
    config: CambridgeG4Config,
    env: dict[str, str],
    *,
    screen_only: bool,
    dry_run: bool,
) -> Path:
    render_output, metrics_output, iteration = train_fit_paths(
        paths,
        screen_only=screen_only,
        config=config,
    )
    output = paths.screen_output if screen_only else paths.full_output
    model = output / "free_gaussians" / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not model.exists() and not dry_run:
        raise FileNotFoundError(f"Cannot evaluate missing model: {model}")
    if metrics_output.exists() and not dry_run:
        print(f"[SKIP] Reusing in-sample train-fit metrics: {metrics_output}")
        return metrics_output

    run_logged(
        train_fit_render_command(paths, config, screen_only=screen_only),
        cwd=REPO_ROOT,
        env=env,
        log_path=output / "logs" / "train_fit_render.log",
        dry_run=dry_run,
    )
    run_logged(
        train_fit_metric_command(paths, screen_only=screen_only, config=config),
        cwd=REPO_ROOT,
        env=env,
        log_path=output / "logs" / "train_fit_metrics.log",
        dry_run=dry_run,
    )
    return metrics_output


def audit_command(paths: ScenePaths, config: CambridgeG4Config) -> list[str]:
    return [
        sys.executable,
        "scripts/audit_cambridge_train_views.py",
        "--dataset",
        str(paths.full_dataset),
        "--mask-pickle",
        str(paths.tree_mask_pickle or paths.mask_pickle),
        "--output",
        str(paths.audit_dir),
        "--quality-mask-indices",
        "0",
        "1",
        "2",
        "--tree-mask-index",
        "3" if paths.tree_mask_pickle is not None else "-1",
        "--pose-clusters",
        str(config.view_clusters),
        "--max-reject-fraction",
        str(config.max_reject_fraction),
    ]


def qc_command(paths: ScenePaths) -> list[str]:
    return [
        sys.executable,
        "scripts/build_cambridge_qc_dataset.py",
        "--source",
        str(paths.full_dataset),
        "--audit",
        str(paths.audit_json),
        "--output",
        str(paths.qc_dataset),
        "--retain-all-images",
    ]


def selection_command(
    paths: ScenePaths,
    config: CambridgeG4Config,
    chart_count: int,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/select_chart_views.py",
        "--scene-path",
        str(paths.qc_dataset),
        "--n-images",
        str(chart_count),
        "--semantic-filter",
        "--semantic-mask-indices",
        "0",
        "--max-semantic-invalid-ratio",
        "0.0",
        "--semantic-ratio-cache",
        str(paths.semantic_cache),
        "--quality-filter",
        "--quality-mask-indices",
        "0",
        "1",
        "2",
        "--min-sharpness",
        str(config.min_sharpness),
        "--max-extreme-ratio",
        str(config.max_extreme_ratio),
        "--max-shadow-ratio",
        str(config.max_shadow_ratio),
        "--min-lower-mean-intensity",
        str(config.min_lower_mean_intensity),
        "--min-valid-ratio",
        "0.0",
        "--view-clusters",
        str(config.view_clusters),
        "--min-views-per-cluster",
        str(config.min_views_per_cluster),
        "--min-baseline-ratio",
        str(config.min_baseline_ratio),
        "--min-global-pose-distance",
        str(config.min_global_pose_distance),
        "--max-borrowed-support-distance",
        str(config.max_borrowed_support_distance),
        "--coverage-objective",
        "target_kcenter",
        "--no-include-ends",
        "--mask-pickle",
        str(paths.mask_pickle),
        "--mask-dataset-path",
        str(paths.qc_dataset),
        "--json",
        "--allowed-name-file",
        str(paths.candidate_allowlist),
    ]
    if paths.tree_mask_pickle is not None:
        command.extend([
            "--tree-mask-pickle", str(paths.tree_mask_pickle),
            "--tree-mask-dataset-path", str(paths.qc_dataset),
            "--tree-mask-index", "3",
            "--max-tree-invalid-ratio", str(config.max_chart_tree_ratio),
            "--tree-ratio-cache", str(paths.tree_ratio_cache),
        ])
    return command


def train_command(
    paths: ScenePaths,
    config: CambridgeG4Config,
    selection: dict,
    *,
    screen_only: bool,
    continue_after_initial: bool = False,
    continue_after_sfm: bool = False,
    continue_after_alignment: bool = False,
    continue_after_plane: bool = False,
    continue_after_see3d_plane_stage: int | None = None,
    stop_after_alignment_gate: bool = False,
) -> list[str]:
    output_path = paths.screen_output if screen_only else paths.full_output
    command = [
        sys.executable,
        "train.py",
        "--source_path",
        str(paths.qc_dataset),
        "--output_path",
        str(output_path),
        "--sfm_config",
        "posed",
        "--image_idx",
        *(str(index) for index in selection["image_idx"]),
        "--alignment_config",
        "semantic_masked_strong_lowmem",
        "--depthanythingv2_checkpoint_dir",
        str(config.depth_checkpoint_dir),
        "--dense_supervision",
        "--dense_data_path",
        str(paths.qc_dataset),
        "--dense_regul",
        "strong_decay",
        "--dense_depth_cache",
        str(paths.qc_dataset.parent / "depth_anything_vitl_fp16.pt"),
        "--dense_final_only",
        "--free_gaussians_config",
        config.screen_free_gaussians_config,
        "--final_free_gaussians_config",
        "long",
        "--final_free_gaussians_iterations",
        str(config.final_iterations),
        "--final_non_position_lr_decay_from",
        str(config.final_non_position_lr_decay_from),
        "--final_non_position_lr_final_mult",
        str(config.final_non_position_lr_final_mult),
        "--data_device",
        "cpu",
        "--resolution",
        str(config.resolution),
        *(["--white_background"] if config.white_background else []),
        "--cambridge_mask_pickle",
        str(paths.mask_pickle),
        "--cambridge_mask_dataset_path",
        str(paths.qc_dataset),
        "--cambridge_mask_indices",
        "0",
        "1",
        "2",
        "--cambridge_geometry_mask_indices",
        "0",
        "1",
        "2",
        "--cambridge_alpha_mask_indices",
        "1",
        "--semantic_alpha_weight",
        str(config.semantic_alpha_weight),
        "--rgb_loss_type",
        config.rgb_loss_type,
        *(["--use_color_correction"] if config.use_color_correction else []),
        "--color_correction_lr",
        str(config.color_correction_lr),
        "--color_correction_reg",
        str(config.color_correction_reg),
        "--use_downsample_gaussians",
        "--downsample_gaussians_type",
        "warp",
        "--warp_downsample_pixel_grid_size",
        str(config.warp_downsample_pixel_grid_size),
        *(
            ["--downweight_input_view_color_loss"]
            if config.downweight_input_view_color_loss
            else []
        ),
        *(
            ["--scene_aligned_see3d_cameras"]
            if config.scene_aligned_see3d_cameras
            else []
        ),
        *(
            ["--preserve_visible_see3d_render"]
            if config.preserve_visible_see3d_render
            else []
        ),
        "--pseudo_initialization_mode",
        config.pseudo_initialization_mode,
        "--pseudo_geometry_mask_mode",
        config.pseudo_geometry_mask_mode,
        "--pseudo_rgb_weight",
        str(config.pseudo_rgb_weight),
        "--pseudo_geometry_weight",
        str(config.pseudo_geometry_weight),
        "--pseudo_geometry_final_weight",
        str(config.pseudo_geometry_final_weight),
        "--pseudo_geometry_decay_until",
        str(config.pseudo_geometry_decay_until),
        *( ["--gate_aligned_chart_conflicts"] if config.gate_aligned_chart_conflicts else [] ),
        "--aligned_chart_min_valid_fraction",
        str(config.aligned_chart_min_valid_fraction),
        "--aligned_chart_max_relative_p90",
        str(config.aligned_chart_max_relative_p90),
        "--aligned_chart_max_gt25_fraction",
        str(config.aligned_chart_max_gt25_fraction),
        *( ["--gate_plane_refinement"] if config.gate_plane_refinement else [] ),
        "--plane_gate_min_support_fraction",
        str(config.plane_gate_min_support_fraction),
        "--plane_gate_max_relative_p90",
        str(config.plane_gate_max_relative_p90),
        "--plane_gate_max_gt25_fraction",
        str(config.plane_gate_max_gt25_fraction),
        "--plane_gate_max_pixel_relative_change",
        str(config.plane_gate_max_pixel_relative_change),
        "--min_global_plane_views",
        str(config.min_global_plane_views),
        "--no_interpolated_views",
        *(["--disable_see3d"] if config.disable_see3d else []),
        "--skip_default_render_eval",
        *( ["--stop_after_alignment_gate"] if stop_after_alignment_gate else [] ),
    ]
    if paths.tree_mask_pickle is not None:
        command.extend([
            "--cambridge_tree_mask_pickle", str(paths.tree_mask_pickle),
            "--cambridge_tree_mask_dataset_path", str(paths.qc_dataset),
            "--cambridge_tree_mask_index", "3",
            "--cambridge_tree_support_dir", str(paths.tree_support_dir),
            "--tree_rgb_floor", "0.25",
            "--tree_rgb_support_gain", "0.50",
            "--tree_geometry_floor", "0.05",
            "--tree_geometry_support_gain", "0.25",
            "--tree_planar_weight", "0.0",
            "--tree_sky_feather", "4",
            "--tree_boundary_feather", "6",
        ])
    if screen_only:
        command.append("--stop_after_initial_refinement")
    if continue_after_initial:
        command.append("--continue_after_initial_refinement")
    if continue_after_sfm:
        command.append("--continue_after_sfm")
    if continue_after_alignment:
        command.append("--continue_after_alignment")
    if continue_after_plane:
        command.append("--continue_after_plane_refinement")
    if continue_after_see3d_plane_stage is not None:
        command.extend([
            "--continue_after_see3d_plane_stage",
            str(continue_after_see3d_plane_stage),
        ])
    return command


def prepare_audit(
    paths: ScenePaths,
    config: CambridgeG4Config,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    if paths.audit_json.exists():
        audit = json.loads(paths.audit_json.read_text())
        audit_inputs_match = (
            audit.get("dataset") == str(paths.full_dataset)
            and audit.get("mask_pickle") == str(paths.tree_mask_pickle or paths.mask_pickle)
            and audit.get("quality_mask_indices") == [0, 1, 2]
        )
        if (
            audit.get("audit_policy_version") == VIEW_AUDIT_POLICY_VERSION
            and audit_inputs_match
        ):
            print(f"[SKIP] Reusing audit: {paths.audit_json}")
            return
        print(f"[INFO] Recomputing stale audit: {paths.audit_json}")
    run_logged(
        audit_command(paths, config),
        cwd=REPO_ROOT,
        env=env,
        log_path=paths.audit_dir / "audit.log",
        dry_run=dry_run,
    )


def prepare_qc(
    paths: ScenePaths,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    manifest = paths.qc_dataset / "qc_manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        current_audit_sha256 = hashlib.sha256(paths.audit_json.read_bytes()).hexdigest()
        if (
            payload.get("pose_policy_version") == COLMAP_POSE_POLICY_VERSION
            and payload.get("audit_sha256") == current_audit_sha256
            and payload.get("reconstruction_image_policy")
            == "all_audited_train_images_retained"
            and payload.get("input_images") == payload.get("output_images")
        ):
            print(f"[SKIP] Reusing QC dataset: {paths.qc_dataset}")
            return
        raise RuntimeError(f"QC dataset has an incompatible pose policy: {manifest}")
    run_logged(qc_command(paths), cwd=REPO_ROOT, env=env, dry_run=dry_run)


def prepare_candidate_allowlist(paths: ScenePaths, *, dry_run: bool) -> None:
    """Make QC an eligibility signal, never an implicit reconstruction split.

    The full 1487-image posed set remains the target set in the chart coverage
    objective and remains the dense reconstruction set.  Only hard-reject
    records are denied *Chart* status.  Persisting the list makes that
    distinction inspectable and prevents later selection calls from accidentally
    treating the QC-copy's image count as the candidate pool.
    """
    audit = json.loads(paths.audit_json.read_text())
    if audit.get("dataset") != str(paths.full_dataset):
        raise RuntimeError(
            "View audit belongs to a different reconstruction dataset: "
            f"{audit.get('dataset')} != {paths.full_dataset}"
        )
    rejected = {
        str(record["image_name"])
        for record in audit.get("records", [])
        if record.get("status") == "hard_reject"
    }
    all_names = sorted(str(record["image_name"]) for record in audit.get("records", []))
    if not all_names:
        raise RuntimeError("View audit contains no train images")
    allowed = [name for name in all_names if name not in rejected]
    if not allowed:
        raise RuntimeError("QC rejected every view as a Chart candidate")
    if dry_run:
        return
    paths.candidate_allowlist.parent.mkdir(parents=True, exist_ok=True)
    paths.candidate_allowlist.write_text("\n".join(allowed) + "\n", encoding="utf-8")
    policy_path = paths.candidate_allowlist.with_suffix(".json")
    policy_path.write_text(
        json.dumps(
            {
                "version": 1,
                "policy": "all_train_target_coverage__hard_reject_chart_ineligible_only",
                "all_train_view_count": len(all_names),
                "candidate_chart_count": len(allowed),
                "hard_reject_chart_count": len(rejected),
                "audit": str(paths.audit_json),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def prepare_selection(
    paths: ScenePaths,
    config: CambridgeG4Config,
    env: dict[str, str],
    dry_run: bool,
    force: bool,
) -> dict:
    if paths.selection_json.exists() and not force:
        existing = json.loads(paths.selection_json.read_text())
        if (
            existing.get("audit_status_filter") == "hard_reject_removed_then_explicit_quality"
            and existing.get("requested_n_images") == config.requested_charts
            and existing.get("selection_policy_version") == SELECTION_POLICY_VERSION
        ):
            print(f"[SKIP] Reusing chart selection: {paths.selection_json}")
            return existing
        print(f"[INFO] Replacing stale or incompatible chart selection: {paths.selection_json}")

    failures = []
    minimum_count = max(
        config.minimum_charts,
        config.view_clusters * config.min_views_per_cluster,
    )
    retryable_messages = (
        "fewer than requested n_images",
        "Allowed-name filter kept",
        "candidate views remain for n_images",
        "Strict semantic cleanliness and trajectory coverage cannot both be satisfied",
        "no remaining clean support view",
        "none provides the required camera baseline",
        "clean, globally diverse chart views can be selected",
    )
    for count in range(config.requested_charts, minimum_count - 1, -1):
        result = run_logged(
            selection_command(paths, config, count),
            cwd=REPO_ROOT,
            env=env,
            capture=True,
            dry_run=dry_run,
        )
        if dry_run:
            return {"image_idx": list(range(count)), "n_images": count, "dry_run": True}
        if result.returncode == 0:
            selection = json.loads(result.stdout)
            selection["requested_n_images"] = config.requested_charts
            selection["actual_n_images"] = count
            selection["audit_status_filter"] = "hard_reject_removed_then_explicit_quality"
            selection["selection_policy_version"] = SELECTION_POLICY_VERSION
            selection["strict_fallback_used"] = count != config.requested_charts
            selection["failed_larger_counts"] = failures
            paths.selection_json.parent.mkdir(parents=True, exist_ok=True)
            paths.selection_json.write_text(json.dumps(selection, indent=2))
            print(
                f"[INFO] Selected {count} strict clean charts; "
                f"requested={config.requested_charts}.",
                flush=True,
            )
            return selection
        error = result.stderr.strip()[-4000:]
        if not any(message in error for message in retryable_messages):
            raise RuntimeError(
                f"Chart selector failed for an implementation/input reason at n={count}:\n{error}"
            )
        failures.append({"n_images": count, "error": error[-2000:]})

    raise RuntimeError(
        "Strict chart selection failed without admitting mask_0-contaminated frames:\n"
        + "\n".join(
            f"n={failure['n_images']}: {failure['error']}" for failure in failures
        )
    )


def load_external_selection(paths: ScenePaths, selection_path: Path) -> dict:
    """Validate a feedback/quality selection before it reaches ``train.py``."""
    selection_path = selection_path.expanduser().resolve()
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    names = [str(name) for name in selection.get("image_names", [])]
    indices = selection.get("image_idx", [])
    if not names or len(names) != len(indices):
        raise ValueError(
            f"External selection must contain equally sized image_names/image_idx: {selection_path}"
        )
    if len(set(names)) != len(names) or len(set(indices)) != len(indices):
        raise ValueError(f"External selection contains duplicate Charts: {selection_path}")
    available = {
        path.name
        for path in (paths.qc_dataset / "images").iterdir()
        if path.is_file() or path.is_symlink()
    }
    missing = sorted(set(names) - available)
    if missing:
        raise ValueError(
            "External selection is not expressed in the current full-train QC dataset: "
            + ", ".join(missing[:5])
        )
    if any(not isinstance(index, int) or index < 0 for index in indices):
        raise ValueError(f"External selection has invalid image_idx values: {selection_path}")
    # ``train.py --image_idx`` is positional.  Checking only that a name is
    # present is insufficient: a quality/joint selector may have been produced
    # against another QC copy with the same filenames but a different camera
    # order.  Refuse such a silent geometry mismatch before MASt3R starts.
    from scripts.select_chart_views import load_scene_poses, normalize_required_name

    ordered_names, _ = load_scene_poses(paths.qc_dataset)
    bad_coordinates = [
        (name, index, ordered_names[index] if index < len(ordered_names) else None)
        for name, index in zip(names, indices)
        if index >= len(ordered_names)
        or normalize_required_name(name) != normalize_required_name(ordered_names[index])
    ]
    if bad_coordinates:
        details = ", ".join(
            f"{name}@{index}!= {actual}" for name, index, actual in bad_coordinates[:3]
        )
        raise ValueError(
            "External selection image_idx does not match image_names in the current "
            f"full-train QC scene: {details}"
        )
    selection = dict(selection)
    selection["external_selection_path"] = str(selection_path)
    selection["selection_consumed_by_training"] = True
    return selection


def apply_active_chart_selection(
    paths: ScenePaths,
    output: Path,
    selection_path: Path,
    *,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    """Hard-zero Charts outside the quality-aware subset before plane building."""
    mast3r_scene = output / "mast3r_sfm"
    gate = mast3r_scene / "aligned_chart_conflict_gate.json"
    if not gate.is_file():
        raise FileNotFoundError(
            "An active Chart selection may only be applied after the alignment gate: "
            + str(gate)
        )
    report = mast3r_scene / "quality_aware_chart_filter.json"
    selection_path = selection_path.expanduser().resolve()
    if report.is_file():
        existing = json.loads(report.read_text(encoding="utf-8"))
        if existing.get("selection") == str(selection_path):
            print(f"[SKIP] Active quality Chart subset already applied: {report}")
            return
        raise RuntimeError(
            "A different quality-aware Chart subset is already active in this run. "
            "Use a new --run-tag/output rather than mutating geometry history."
        )
    run_logged(
        [
            sys.executable,
            "scripts/apply_chart_selection.py",
            "--mast3r-scene",
            str(mast3r_scene),
            "--selection",
            str(selection_path),
            "--gate-report",
            str(gate),
        ],
        cwd=REPO_ROOT,
        env=env,
        log_path=output / "logs" / "apply_active_chart_selection.log",
        dry_run=dry_run,
    )


def write_run_manifest(
    scene: str,
    paths: ScenePaths,
    config: CambridgeG4Config,
    selection: dict,
    command: list[str],
    *,
    screen_only: bool,
) -> Path:
    output = paths.screen_output if screen_only else paths.full_output
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "scene": scene,
        "mode": "screen7k" if screen_only else f"full{config.final_iterations}",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "mask_policy": {
            "strict_chart_frame_indices": [0],
            "quality_indices": [0, 1, 2],
            "alignment_indices": [0, 1, 2, 3] if paths.tree_mask_pickle is not None else [0, 1, 2],
            "rgb_loss_indices": [0, 1, 2],
            "geometry_loss_indices": [0, 1, 2],
            "alpha_suppression_indices": [1],
            "semantic_alpha_weight": config.semantic_alpha_weight,
            "white_background": config.white_background,
            "warp_downsample_pixel_grid_size": config.warp_downsample_pixel_grid_size,
            "downweight_input_view_color_loss": config.downweight_input_view_color_loss,
            "scene_aligned_see3d_cameras": config.scene_aligned_see3d_cameras,
            "preserve_visible_see3d_render": config.preserve_visible_see3d_render,
            "disable_see3d": config.disable_see3d,
            "pseudo_initialization_mode": config.pseudo_initialization_mode,
            "pseudo_geometry_mask_mode": config.pseudo_geometry_mask_mode,
            "pseudo_rgb_weight": config.pseudo_rgb_weight,
            "pseudo_geometry_weight": config.pseudo_geometry_weight,
            "pseudo_geometry_final_weight": config.pseudo_geometry_final_weight,
            "tree_index_3_hard_masked_for_chart_geometry": paths.tree_mask_pickle is not None,
            "tree_candidate_max_ratio": config.max_chart_tree_ratio,
            "canonical_tree_soft_weighting": paths.tree_mask_pickle is not None,
        },
        "audit_policy_version": VIEW_AUDIT_POLICY_VERSION,
        "pose_policy_version": COLMAP_POSE_POLICY_VERSION,
        "dense_depth_schedule": "strong_decay",
        "data_partition_policy": "all_train_no_validation_split",
        "qc_reconstruction_policy": "all_audited_train_images_retained",
        "chart_candidate_policy": "hard_reject_chart_ineligible_only",
        "evaluation_kind": "in_sample_train_fit",
        "reconstruction_dataset": str(paths.full_dataset),
        "dense_dataset": str(paths.qc_dataset),
        "evaluation_dataset": str(paths.qc_dataset),
        "evaluation_camera_policy": "all_train_qc_normalized_poses",
        "chart_candidate_allowlist": str(paths.candidate_allowlist),
        "dense_view_count": (
            len(list((paths.qc_dataset / "images").iterdir()))
            if (paths.qc_dataset / "images").exists()
            else None
        ),
        "selection": selection,
        "command": command,
    }
    manifest_path = output / "cambridge_g4_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and run quality-controlled Cambridge G4Splat experiments."
    )
    parser.add_argument("--scenes", nargs="+", default=["StMarysChurch"], choices=DEFAULT_SCENES)
    parser.add_argument(
        "--phase",
        choices=[
            "audit", "qc", "select", "prepare", "frontend", "screen", "full",
            "evaluate-screen", "evaluate-full",
        ],
        default="prepare",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("/root/MAtCha/output_cambridge/datasets_full"),
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=Path("/mnt/pool/sqy/Cambridge_stdloc"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/G4Splat/output_cambridge_qc_n24_30k"),
    )
    parser.add_argument(
        "--depth-checkpoint-dir",
        type=Path,
        default=Path("/root/MAtCha/Depth-Anything-V2/checkpoints"),
    )
    parser.add_argument("--n-images", type=int, default=24)
    parser.add_argument(
        "--run-tag",
        default="base",
        help="Change this for each chart-feedback round so stale SfM/alignment artifacts cannot be reused.",
    )
    parser.add_argument("--minimum-charts", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--rgb-loss-type", choices=["l1", "charbonnier"], default="l1")
    parser.add_argument(
        "--color-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Learn a regularized affine RGB correction for each real training image.",
    )
    parser.add_argument("--color-correction-lr", type=float, default=1e-3)
    parser.add_argument("--color-correction-reg", type=float, default=1e-2)
    parser.add_argument(
        "--white-background",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--semantic-alpha-weight", type=float, default=0.01)
    parser.add_argument(
        "--screen-free-gaussians-config",
        default="default",
        help=(
            "7k screen refinement configuration from "
            "configs/free_gaussians_refinement (for example default or screen_long7k)."
        ),
    )
    parser.add_argument(
        "--final-iterations",
        type=int,
        default=30_000,
        help="Final all-train refinement length; screen remains a separate 7k gate.",
    )
    parser.add_argument(
        "--final-non-position-lr-decay-from",
        type=int,
        default=30_000,
        help="Final-stage iteration at which appearance/opacity/scale/rotation rates start decaying.",
    )
    parser.add_argument(
        "--final-non-position-lr-final-mult",
        type=float,
        default=0.10,
        help="Non-position learning-rate multiplier reached at the end of the final stage.",
    )
    parser.add_argument(
        "--min-global-plane-views",
        type=int,
        default=2,
        help=(
            "Minimum distinct Chart views that must support a global plane "
            "before it may alter aligned depth."
        ),
    )
    parser.add_argument("--max-chart-tree-ratio", type=float, default=0.15)
    parser.add_argument("--warp-downsample-pixel-grid-size", type=int, default=2)
    parser.add_argument(
        "--downweight-input-view-color-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reduce RGB weight for real chart views when dense supervision is enabled. "
            "Disabled by default so selected charts remain appearance anchors."
        ),
    )
    parser.add_argument(
        "--scene-aligned-see3d-cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pose-neighbor camera proposals instead of G4Splat's fixed world axes.",
    )
    parser.add_argument(
        "--preserve-visible-see3d-render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve observed GS pixels and generate only disoccluded pseudo-view regions.",
    )
    parser.add_argument(
        "--pseudo-initialization-mode",
        choices=["all", "inpaint_only", "none"],
        default="none",
    )
    parser.add_argument(
        "--pseudo-geometry-mask-mode",
        choices=["all", "inpaint_only", "none"],
        default="none",
    )
    parser.add_argument("--pseudo-rgb-weight", type=float, default=0.0)
    parser.add_argument("--pseudo-geometry-weight", type=float, default=0.0)
    parser.add_argument("--pseudo-geometry-final-weight", type=float, default=0.0)
    parser.add_argument("--pseudo-geometry-decay-until", type=int, default=5000)
    parser.add_argument("--gpu", default="0", help="Exactly one physical GPU index.")
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=None,
        help=(
            "Use a previously audited joint Chart selection instead of selecting "
            "a fresh independent set.  The selection is validated against the "
            "current all-train QC dataset before training."
        ),
    )
    parser.add_argument(
        "--active-chart-selection-json",
        type=Path,
        default=None,
        help=(
            "After the aligned-chart gate, hard-zero every Chart outside this "
            "quality-aware subset before plane/2DGS geometry construction."
        ),
    )
    parser.add_argument(
        "--resume-after-see3d-plane-stage",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Recover a full run after the selected See3D plane stage completed.",
    )
    parser.add_argument("--force-selection", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "," in args.gpu or not args.gpu.isdigit():
        raise ValueError("--gpu must identify exactly one GPU, for example --gpu 0")
    if args.minimum_charts <= 0 or args.minimum_charts > args.n_images:
        raise ValueError("minimum chart count must be in [1, n_images]")
    if args.semantic_alpha_weight < 0:
        raise ValueError("--semantic-alpha-weight must be non-negative")
    if args.final_iterations <= 0:
        raise ValueError("--final-iterations must be positive")
    if args.final_non_position_lr_decay_from < -1:
        raise ValueError("--final-non-position-lr-decay-from must be -1 or non-negative")
    if not 0.0 < args.final_non_position_lr_final_mult <= 1.0:
        raise ValueError("--final-non-position-lr-final-mult must be in (0, 1]")
    if args.min_global_plane_views < 1:
        raise ValueError("--min-global-plane-views must be at least one")
    screen_config_path = (
        REPO_ROOT
        / "configs"
        / "free_gaussians_refinement"
        / f"{args.screen_free_gaussians_config}.yaml"
    )
    if not screen_config_path.is_file():
        raise ValueError(
            "--screen-free-gaussians-config must name an existing configuration: "
            f"{screen_config_path}"
        )
    if min(
        args.pseudo_rgb_weight,
        args.pseudo_geometry_weight,
        args.pseudo_geometry_final_weight,
    ) < 0:
        raise ValueError("pseudo-view weights must be non-negative")
    if args.pseudo_geometry_decay_until < 0:
        raise ValueError("--pseudo-geometry-decay-until must be non-negative")
    if args.warp_downsample_pixel_grid_size == 0 or args.warp_downsample_pixel_grid_size < -1:
        raise ValueError("--warp-downsample-pixel-grid-size must be -1 or a positive integer")
    if args.resume_after_see3d_plane_stage is not None and args.phase != "full":
        raise ValueError("--resume-after-see3d-plane-stage is only valid with --phase full")

    config = CambridgeG4Config(
        datasets_root=args.datasets_root,
        mask_root=args.mask_root,
        output_root=args.output_root,
        depth_checkpoint_dir=args.depth_checkpoint_dir,
        run_tag=args.run_tag,
        requested_charts=args.n_images,
        minimum_charts=args.minimum_charts,
        resolution=args.resolution,
        rgb_loss_type=args.rgb_loss_type,
        use_color_correction=args.color_correction,
        color_correction_lr=args.color_correction_lr,
        color_correction_reg=args.color_correction_reg,
        white_background=args.white_background,
        semantic_alpha_weight=args.semantic_alpha_weight,
        screen_free_gaussians_config=args.screen_free_gaussians_config,
        final_iterations=args.final_iterations,
        final_non_position_lr_decay_from=args.final_non_position_lr_decay_from,
        final_non_position_lr_final_mult=args.final_non_position_lr_final_mult,
        min_global_plane_views=args.min_global_plane_views,
        warp_downsample_pixel_grid_size=args.warp_downsample_pixel_grid_size,
        downweight_input_view_color_loss=args.downweight_input_view_color_loss,
        scene_aligned_see3d_cameras=args.scene_aligned_see3d_cameras,
        preserve_visible_see3d_render=args.preserve_visible_see3d_render,
        pseudo_initialization_mode=args.pseudo_initialization_mode,
        pseudo_geometry_mask_mode=args.pseudo_geometry_mask_mode,
        pseudo_rgb_weight=args.pseudo_rgb_weight,
        pseudo_geometry_weight=args.pseudo_geometry_weight,
        pseudo_geometry_final_weight=args.pseudo_geometry_final_weight,
        pseudo_geometry_decay_until=args.pseudo_geometry_decay_until,
        max_chart_tree_ratio=args.max_chart_tree_ratio,
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    conda_lib = str(Path(sys.prefix) / "lib")
    current_library_path = env.get("LD_LIBRARY_PATH", "")
    library_entries = [entry for entry in current_library_path.split(os.pathsep) if entry]
    if conda_lib not in library_entries:
        env["LD_LIBRARY_PATH"] = os.pathsep.join([conda_lib, *library_entries])
    env.setdefault("PYTHONUNBUFFERED", "1")

    for scene in args.scenes:
        paths = scene_paths(scene, config)
        validate_scene(scene, paths)
        print(f"\n[SCENE] {scene}", flush=True)

        prepare_audit(paths, config, env, args.dry_run)
        if args.phase == "audit":
            continue
        prepare_candidate_allowlist(paths, dry_run=args.dry_run)
        prepare_qc(paths, env, args.dry_run)
        if args.phase == "qc":
            continue
        selection = (
            load_external_selection(paths, args.selection_json)
            if args.selection_json is not None
            else prepare_selection(
                paths,
                config,
                env,
                args.dry_run,
                args.force_selection,
            )
        )
        if args.phase in {"select", "prepare"}:
            continue

        if args.phase in {"evaluate-screen", "evaluate-full"}:
            evaluate_train_fit(
                paths,
                config,
                env,
                screen_only=args.phase == "evaluate-screen",
                dry_run=args.dry_run,
            )
            continue

        frontend_only = args.phase == "frontend"
        screen_only = args.phase in {"frontend", "screen"}
        continue_after_initial = False
        resume_after_see3d = args.resume_after_see3d_plane_stage
        if not screen_only and resume_after_see3d is None:
            screen_iteration = (
                paths.screen_output
                / "free_gaussians"
                / "point_cloud"
                / "iteration_7000"
                / "point_cloud.ply"
            )
            full_iteration = (
                paths.full_output
                / "free_gaussians"
                / "point_cloud"
                / "iteration_7000"
                / "point_cloud.ply"
            )
            if screen_iteration.exists() and not paths.full_output.exists():
                paths.full_output.parent.mkdir(parents=True, exist_ok=True)
                # Keep the screened 7k artifact immutable so 20k/40k/80k
                # final schedules can all start from the identical geometry
                # gate rather than silently rebuilding a different frontend.
                shutil.copytree(paths.screen_output, paths.full_output)
                print(
                    f"[INFO] Cloned screened output for final schedule: {paths.full_output}",
                    flush=True,
                )
                continue_after_initial = True
            elif full_iteration.exists():
                continue_after_initial = True
        output = paths.screen_output if screen_only else paths.full_output
        continue_after_plane = (
            resume_after_see3d is None
            and
            not continue_after_initial
            and (output / "mast3r_sfm" / "charts_data.npz").exists()
            and (
                output
                / "mast3r_sfm"
                / "plane-refine-depths"
                / "refine_depth_frame000000.tiff"
            ).exists()
        )
        continue_after_alignment = (
            resume_after_see3d is None
            and
            not continue_after_initial
            and not continue_after_plane
            and (output / "mast3r_sfm" / "charts_data.npz").exists()
        )
        continue_after_sfm = (
            resume_after_see3d is None
            and
            not continue_after_initial
            and not continue_after_plane
            and not continue_after_alignment
            and (output / "mast3r_sfm" / "sparse" / "0" / "images.bin").exists()
            and (output / "mast3r_sfm" / "pointmaps").exists()
        )
        if args.active_chart_selection_json is not None:
            if not (continue_after_initial or continue_after_alignment or continue_after_plane):
                raise RuntimeError(
                    "--active-chart-selection-json requires an already aligned "
                    "frontend. Run --phase frontend first, inspect the gate, then "
                    "resume the same --run-tag with --phase screen/full."
                )
            if continue_after_plane:
                raise RuntimeError(
                    "Refusing to change the active Chart subset after plane refinement. "
                    "Use a new --run-tag/output so the geometry provenance remains valid."
                )
            apply_active_chart_selection(
                paths,
                output,
                args.active_chart_selection_json,
                env=env,
                dry_run=args.dry_run,
            )
        command = train_command(
            paths,
            config,
            selection,
            screen_only=screen_only,
            continue_after_initial=continue_after_initial,
            continue_after_sfm=continue_after_sfm,
            continue_after_alignment=continue_after_alignment,
            continue_after_plane=continue_after_plane,
            continue_after_see3d_plane_stage=resume_after_see3d,
            stop_after_alignment_gate=frontend_only,
        )
        final_iteration = output / "free_gaussians" / "point_cloud" / (
            "iteration_7000" if screen_only else f"iteration_{config.final_iterations}"
        )
        if final_iteration.exists():
            print(f"[SKIP] Completed model already exists: {final_iteration}")
        else:
            manifest = write_run_manifest(
                scene,
                paths,
                config,
                selection,
                command,
                screen_only=screen_only,
            )
            run_logged(
                command,
                cwd=REPO_ROOT,
                env=env,
                log_path=output / "logs" / "train.log",
                dry_run=args.dry_run,
            )
            print(f"[DONE] Manifest: {manifest}")
        if frontend_only:
            gate_report = output / "mast3r_sfm" / "aligned_chart_conflict_gate.json"
            print(f"[FRONTEND] Chart feedback report: {gate_report}")
            continue
        metrics_path = evaluate_train_fit(
            paths,
            config,
            env,
            screen_only=screen_only,
            dry_run=args.dry_run,
        )
        print(f"[DONE] In-sample train-fit metrics: {metrics_path}")


if __name__ == "__main__":
    main()
