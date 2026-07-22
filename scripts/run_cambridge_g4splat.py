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
from view_quality_control.provenance import audit_input_provenance  # noqa: E402


DEFAULT_SCENES = [
    "GreatCourt",
    "KingsCollege",
    "OldHospital",
    "ShopFacade",
    "StMarysChurch",
]
RUN_POLICY_VERSION = "cambridge-g4-v21-outdoor-structural-sequence-resilient"
SELECTION_POLICY_VERSION = "outdoor-static-support-sequence-joint-gate-v11-resilient"
AUDIT_THING_MASK_INDEX = 0
AUDIT_SKY_MASK_INDEX = 1
AUDIT_NEIGHBOR_COUNT = 8
AUDIT_MAX_IMAGE_WIDTH = 960


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
    # The broad MASt3R candidate set and the post-gate active set serve
    # different purposes.  Two baseline-separated primary views are sufficient
    # for geometry.  A temporal near-duplicate is not a third independent
    # baseline, but it is exactly the correct clean substitute if the hard
    # depth gate rejects one of those two primary views.  Keep the counts and
    # explicit replacement-reserve policy separate.
    candidate_min_views_per_cluster: int = 2
    joint_min_views_per_cluster: int = 2
    gate_reserves_per_vulnerable_support: int = 1
    # The target-depth gate can reject correlated charts in one pose cell.
    # ``1`` reproduces the legacy per-primary reserve policy; larger values
    # require a jointly certified primary-plus-reserve support set.
    gate_failure_budget: int = 1
    min_baseline_ratio: float = 0.02
    min_global_pose_distance: float = 0.05
    max_borrowed_support_distance: float = 0.25
    min_sharpness: float = 1e-4
    max_extreme_ratio: float = 0.20
    max_shadow_ratio: float = 0.35
    min_lower_mean_intensity: float = 0.15
    # Whole-frame mask purity is too brittle for outdoor trajectories: a small
    # pedestrian, tree branch or image border should not remove the only
    # usable baseline.  Use it only to reject heavily contaminated frames,
    # then require a continuous amount of static support at pixel level.
    max_chart_semantic_invalid_ratio: float = 0.10
    min_chart_static_support_ratio: float = 0.10
    chart_tree_candidate_policy: str = "soft_penalty"
    chart_quality_candidate_policy: str = "soft_penalty"
    chart_candidate_reliability_weight: float = 0.03
    # Same-sequence support is deliberately independent from global pose
    # coverage: a pose-near view from another traversal is not evidence that a
    # temporal corridor has a valid local triangulation pair.
    # Three input Charts are reserved so that a single hard target-depth-gate
    # rejection still leaves two independently-baselined same-trajectory
    # structural supports.  This is separate from the pose-cluster reserve
    # policy because a sequence can disappear while global clusters still
    # appear numerically covered by another traversal.
    min_chart_views_per_sequence: int = 3
    post_gate_min_chart_views_per_sequence: int = 2
    chart_sequence_gate_failure_budget: int = 1
    chart_sequence_coverage_mode: str = "strict"
    resolution: int = 2
    # This is deliberately an explicit frontend ablation rather than an
    # implicit consequence of ``--fix-translation``.  The historical MASt3R
    # path still optimized translation/scale and only restored calibrated
    # camera centers at export time, which can leave its point maps in a
    # slightly different coordinate frame.  A true fixed-COLMAP experiment
    # freezes those variables inside global alignment.
    strict_calibrated_poses: bool = False
    # A separate intrinsic-camera ablation: with calibrated focal values, a
    # shared MASt3R focal is not semantically a fixed calibration.
    per_view_calibrated_intrinsics: bool = False
    # Chart alignment initializes learned encodings and deformation layers.
    # Keep this fixed across camera-front-end ablations.
    chart_alignment_seed: int = 0
    # MASt3R pointmaps remain full resolution.  This applies only to the
    # diagnostic COLMAP export, whose historical one-Point3D-per-pixel loop is
    # otherwise a substantial CPU/RAM/disk bottleneck as the outdoor Chart set
    # grows beyond a few images.
    mast3r_sparse_export_stride: int = 4
    rgb_loss_type: str = "l1"
    # RGB pixels are a separate experimental variable from Chart geometry.
    # ``full_rgb`` and ``ulfloc_legacy`` both keep Chart/tree/plane constraints
    # intact. The former uses every RGB pixel; the latter adopts ULF-Loc's
    # object+distortion mask with white sky target.
    rgb_supervision_profile: str = "g4_tree_masked"
    # Charts remain geometry anchors, but RGB importance weights remove the
    # accidental Chart oversampling induced by the interleaved geometry loop.
    rgb_sampling_policy: str = "all_train_importance"
    # A hard-rejected or quality-excluded Chart must not consume a geometry
    # iteration.  The legacy mode remains available only for a paired ablation.
    chart_geometry_sampling_policy: str = "active_only"
    # Chart renders may remain geometry anchors without also dominating the
    # screen-space gradient/radius statistics that allocate 2DGS topology.
    densification_view_policy: str = "legacy_current"
    # A strict causal control can retain the exact Chart schedule while
    # removing only the Chart-exclusive geometric priors.
    chart_geometry_prior_weight: float = 1.0
    use_color_correction: bool = True
    color_correction_lr: float = 1e-3
    color_correction_reg: float = 1e-2
    white_background: bool = True
    semantic_alpha_weight: float = 0.01
    warp_downsample_pixel_grid_size: int = 2
    # Cambridge reconstruction is transductive here: all real train cameras
    # contribute from iteration one.  A Chart-only screen is useful only as a
    # named ablation, because otherwise it silently turns the Chart subset
    # into an unintended RGB-training split.
    dense_final_only: bool = False
    # ``requested_charts`` is the broad candidate set used for MASt3R
    # alignment.  This optional second count activates the post-gate joint
    # selector (clarity + cross-view agreement + normal stability + 3-D
    # marginal coverage) and hard-writes that verified subset before planes
    # or Gaussian initialization consume it.
    joint_chart_count: int | None = None
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
    require_post_gate_coverage: bool = True
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
    # Keep the final refinement schedule explicit too.  The outdoor mainline
    # needs to be able to use a capacity-safe continuation without silently
    # inheriting the legacy generic ``long`` profile.
    final_free_gaussians_config: str = "long"
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
    static_support_ratio_cache: Path
    quality_score_cache: Path
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
    # Keep the legacy (false) run identity stable so an in-flight legacy
    # frontend remains resumable.  The strict mode is a material intervention
    # and therefore participates in the digest when enabled.
    if not payload.get("strict_calibrated_poses", False):
        payload.pop("strict_calibrated_poses", None)
    if not payload.get("per_view_calibrated_intrinsics", False):
        payload.pop("per_view_calibrated_intrinsics", None)
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
        selection_json=(
            prepared
            / (
                f"chart_selection_n{config.requested_charts}_"
                f"clusters{config.view_clusters}_"
                f"support{config.candidate_min_views_per_cluster}.json"
            )
        ),
        semantic_cache=prepared / "mask0_invalid_ratio.json",
        tree_ratio_cache=prepared / "tree_invalid_ratio.json",
        static_support_ratio_cache=(
            prepared / "static_support_invalid_ratio_mask0123.json"
        ),
        # Quality features are independent of requested Chart count.  Persist
        # them so a strict fallback/reserve search does not rescan all 1487
        # images for every attempted count.
        quality_score_cache=prepared / "chart_quality_scores_mask012.json",
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
    source_names = staged_image_names(paths.full_dataset)
    mapping = json.loads((paths.full_dataset / "name_mapping.json").read_text())
    mapping_names = {str(name) for name in mapping}
    if source_names != mapping_names:
        missing_mapping = sorted(source_names - mapping_names)
        extra_mapping = sorted(mapping_names - source_names)
        raise RuntimeError(
            "Staged image directory and name_mapping.json disagree; refusing to "
            "silently construct a partial reconstruction set. "
            f"missing_mapping={missing_mapping[:3]}, extra_mapping={extra_mapping[:3]}"
        )


def staged_image_names(dataset: Path) -> set[str]:
    """Return the exact camera-image names a staged COLMAP scene can consume."""
    image_dir = dataset / "images"
    names = {
        path.name
        for path in image_dir.iterdir()
        if path.is_file() or path.is_symlink()
    }
    if not names:
        raise RuntimeError(f"No staged images found in {image_dir}")
    return names


def _name_set_sha256(names: Iterable[str]) -> str:
    """Stable identity for a camera set, independent of filesystem ordering."""
    payload = "\n".join(sorted(str(name) for name in names)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _colmap_text_image_names(dataset: Path) -> set[str]:
    """Read the exact image names represented by the QC pose text file."""
    images_txt = dataset / "sparse" / "0" / "images.txt"
    if not images_txt.is_file():
        raise FileNotFoundError(f"QC pose text file is missing: {images_txt}")
    names: set[str] = set()
    for raw_line in images_txt.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # COLMAP image records have IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME.
        # The interleaved 2D-point records have no image name and are ignored.
        if len(parts) >= 10 and parts[0].lstrip("-").isdigit():
            names.add(Path(parts[9]).name)
    if not names:
        raise RuntimeError(f"No camera records found in QC pose text file: {images_txt}")
    return names


def assert_audit_covers_full_dataset(paths: ScenePaths, audit: dict) -> set[str]:
    """Reject an audit that would turn an all-train run into a hidden subset.

    A hard-reject is allowed to make a view Chart-ineligible, but the audit
    itself must enumerate every source image.  Comparing only count is not
    enough: an omitted trajectory segment can have the same cardinality.
    """
    if audit.get("dataset") != str(paths.full_dataset):
        raise RuntimeError(
            "View audit belongs to a different reconstruction dataset: "
            f"{audit.get('dataset')} != {paths.full_dataset}"
        )
    audit_names = {
        str(record["image_name"])
        for record in audit.get("records", [])
        if "image_name" in record
    }
    source_names = staged_image_names(paths.full_dataset)
    if audit_names != source_names:
        missing_audit = sorted(source_names - audit_names)
        extra_audit = sorted(audit_names - source_names)
        raise RuntimeError(
            "View audit does not cover the complete staged training set; "
            "do not reuse it for an all-train reconstruction. "
            f"source={len(source_names)}, audit={len(audit_names)}, "
            f"missing={missing_audit[:5]}, extra={extra_audit[:5]}"
        )
    return source_names


def assert_qc_retains_full_dataset(paths: ScenePaths, manifest: dict) -> None:
    """Validate that QC is metadata/pose normalization, never a train split.

    Counts alone are insufficient here: a same-sized QC copy could still have
    replaced a difficult frame with a duplicate, or have lost its COLMAP pose.
    The dense reconstruction objective must consume exactly the original image
    *identity set* and exactly one normalized pose for each one.
    """
    source_names = staged_image_names(paths.full_dataset)
    qc_names = staged_image_names(paths.qc_dataset)
    qc_pose_names = _colmap_text_image_names(paths.qc_dataset)
    expected = len(source_names)
    if (
        manifest.get("source") != str(paths.full_dataset)
        or manifest.get("reconstruction_image_policy")
        != "all_audited_train_images_retained"
        or manifest.get("input_images") != expected
        or manifest.get("output_images") != expected
        or qc_names != source_names
        or qc_pose_names != source_names
    ):
        missing_qc = sorted(source_names - qc_names)
        extra_qc = sorted(qc_names - source_names)
        missing_pose = sorted(source_names - qc_pose_names)
        extra_pose = sorted(qc_pose_names - source_names)
        raise RuntimeError(
            "QC manifest is not a full-train reconstruction copy: "
            f"expected={expected}, source={manifest.get('source')}, "
            f"input={manifest.get('input_images')}, output={manifest.get('output_images')}, "
            f"policy={manifest.get('reconstruction_image_policy')}, "
            f"missing_qc={missing_qc[:3]}, extra_qc={extra_qc[:3]}, "
            f"missing_pose={missing_pose[:3]}, extra_pose={extra_pose[:3]}"
        )


def _expected_chart_candidate_names(paths: ScenePaths, audit: dict) -> list[str]:
    source_names = assert_audit_covers_full_dataset(paths, audit)
    rejected = {
        str(record["image_name"])
        for record in audit.get("records", [])
        if record.get("status") == "hard_reject"
    }
    unknown_rejected = sorted(rejected - source_names)
    if unknown_rejected:
        raise RuntimeError(
            "View audit marks a non-training image as hard-reject: "
            f"{unknown_rejected[:3]}"
        )
    return sorted(source_names - rejected)


def _read_allowed_chart_names(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Chart candidate allowlist is missing: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise RuntimeError(f"Chart candidate allowlist contains duplicate names: {path}")
    return names


def assert_candidate_allowlist_matches_audit(paths: ScenePaths) -> list[str]:
    """Ensure hard-reject affects Chart eligibility only and nothing else."""
    audit = json.loads(paths.audit_json.read_text(encoding="utf-8"))
    expected = _expected_chart_candidate_names(paths, audit)
    actual = _read_allowed_chart_names(paths.candidate_allowlist)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(
            "Chart candidate allowlist no longer matches the current full-train audit; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return actual


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
    tree_mask_index = 3 if paths.tree_mask_pickle is not None else -1
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
        str(tree_mask_index),
        "--thing-mask-index",
        str(AUDIT_THING_MASK_INDEX),
        "--sky-mask-index",
        str(AUDIT_SKY_MASK_INDEX),
        "--pose-clusters",
        str(config.view_clusters),
        "--neighbor-count",
        str(AUDIT_NEIGHBOR_COUNT),
        "--max-image-width",
        str(AUDIT_MAX_IMAGE_WIDTH),
        "--max-reject-fraction",
        str(config.max_reject_fraction),
    ]


def expected_audit_input_provenance(paths: ScenePaths, config: CambridgeG4Config) -> dict:
    return audit_input_provenance(
        paths.full_dataset,
        paths.tree_mask_pickle or paths.mask_pickle,
        quality_mask_indices=[0, 1, 2],
        thing_mask_index=AUDIT_THING_MASK_INDEX,
        sky_mask_index=AUDIT_SKY_MASK_INDEX,
        tree_mask_index=3 if paths.tree_mask_pickle is not None else -1,
        pose_clusters=config.view_clusters,
        neighbor_count=AUDIT_NEIGHBOR_COUNT,
        max_image_width=AUDIT_MAX_IMAGE_WIDTH,
        max_reject_fraction=config.max_reject_fraction,
    )


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
    # Structural Chart eligibility is measured from the static portion of a
    # frame.  When available, channel 3 excludes foliage; it is never a
    # license to let foliage into plane/depth alignment, only a way to avoid
    # rejecting a useful frame that has foliage elsewhere in the image.
    static_support_mask = paths.tree_mask_pickle or paths.mask_pickle
    static_support_indices = (
        [0, 1, 2, 3] if paths.tree_mask_pickle is not None else [0, 1, 2]
    )
    # The tree-aware pickle is an extension of the base validity tuple.  Use
    # that one payload consistently for semantic, quality and static-support
    # measurements so the selector can share one deserialized large mask
    # dictionary rather than loading the base and tree versions concurrently.
    selector_mask = paths.tree_mask_pickle or paths.mask_pickle
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
        str(config.max_chart_semantic_invalid_ratio),
        "--semantic-ratio-cache",
        str(paths.semantic_cache),
        "--static-support-mask-pickle",
        str(static_support_mask),
        "--static-support-mask-dataset-path",
        str(paths.qc_dataset),
        "--static-support-mask-indices",
        *(str(index) for index in static_support_indices),
        "--min-static-support-ratio",
        str(config.min_chart_static_support_ratio),
        "--static-support-ratio-cache",
        str(paths.static_support_ratio_cache),
        "--quality-filter",
        "--quality-candidate-policy",
        config.chart_quality_candidate_policy,
        "--quality-mask-indices",
        "0",
        "1",
        "2",
        "--quality-score-mask-pickle",
        # Quality only consumes channels 0/1/2; the tree-aware payload retains
        # those exact channels and lets the selector reuse its already-loaded
        # mask dictionary/cache instead of scanning every outdoor RGB frame a
        # second time merely because the pickle filename differs.
        str(selector_mask),
        "--quality-score-mask-dataset-path",
        str(paths.qc_dataset),
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
        "--quality-score-cache",
        str(paths.quality_score_cache),
        "--view-clusters",
        str(config.view_clusters),
        "--min-views-per-cluster",
        str(config.candidate_min_views_per_cluster),
        "--min-baseline-ratio",
        str(config.min_baseline_ratio),
        "--min-global-pose-distance",
        str(config.min_global_pose_distance),
        "--max-borrowed-support-distance",
        str(config.max_borrowed_support_distance),
        "--post-gate-min-views-per-cluster",
        str(config.joint_min_views_per_cluster),
        "--reserve-max-pose-distance",
        str(config.min_global_pose_distance),
        "--reserves-per-vulnerable-support",
        str(config.gate_reserves_per_vulnerable_support),
        "--gate-failure-budget",
        str(config.gate_failure_budget),
        "--coverage-objective",
        "target_kcenter",
        "--candidate-reliability-weight",
        str(config.chart_candidate_reliability_weight),
        "--min-views-per-sequence",
        str(config.min_chart_views_per_sequence),
        "--post-gate-min-views-per-sequence",
        str(config.post_gate_min_chart_views_per_sequence),
        "--sequence-gate-failure-budget",
        str(config.chart_sequence_gate_failure_budget),
        "--sequence-coverage-mode",
        config.chart_sequence_coverage_mode,
        "--no-include-ends",
        "--mask-pickle",
        str(selector_mask),
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
            "--tree-candidate-policy", config.chart_tree_candidate_policy,
        ])
    if config.gate_reserves_per_vulnerable_support > 0:
        command.append("--require-gate-replacement-reserves")
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
    gate_reserves = selection.get("gate_replacement_reserves", {})
    alignment_indices = gate_reserves.get("alignment_image_idx", selection["image_idx"])
    command = [
        sys.executable,
        "train.py",
        "--source_path",
        str(paths.qc_dataset),
        "--output_path",
        str(output_path),
        "--sfm_config",
        "posed",
        *(["--strict_calibrated_poses"] if config.strict_calibrated_poses else []),
        *(
            ["--per_view_calibrated_intrinsics"]
            if config.per_view_calibrated_intrinsics
            else []
        ),
        "--image_idx",
        *(str(index) for index in alignment_indices),
        "--alignment_config",
        "semantic_masked_strong_lowmem",
        "--chart-alignment-seed",
        str(config.chart_alignment_seed),
        "--mast3r-sparse-export-stride",
        str(config.mast3r_sparse_export_stride),
        "--depthanythingv2_checkpoint_dir",
        str(config.depth_checkpoint_dir),
        "--dense_supervision",
        "--dense_data_path",
        str(paths.qc_dataset),
        "--dense_regul",
        "strong_decay",
        "--dense_depth_cache",
        str(paths.qc_dataset.parent / "depth_anything_vitl_fp16.pt"),
        *( ["--dense_final_only"] if config.dense_final_only else [] ),
        "--free_gaussians_config",
        config.screen_free_gaussians_config,
        "--final_free_gaussians_config",
        config.final_free_gaussians_config,
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
        "--rgb-supervision-profile",
        config.rgb_supervision_profile,
        "--rgb-sampling-policy",
        config.rgb_sampling_policy,
        "--chart-geometry-sampling-policy",
        config.chart_geometry_sampling_policy,
        "--densification-view-policy",
        config.densification_view_policy,
        "--chart-geometry-prior-weight",
        str(config.chart_geometry_prior_weight),
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
    if screen_only and not stop_after_alignment_gate:
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
    expected_provenance = None if dry_run else expected_audit_input_provenance(paths, config)
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
            and audit.get("input_provenance") == expected_provenance
        ):
            if not dry_run:
                assert_audit_covers_full_dataset(paths, audit)
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
    if not dry_run:
        assert_audit_covers_full_dataset(
            paths,
            json.loads(paths.audit_json.read_text()),
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
        ):
            assert_qc_retains_full_dataset(paths, payload)
            print(f"[SKIP] Reusing QC dataset: {paths.qc_dataset}")
            return
        raise RuntimeError(f"QC dataset has an incompatible pose policy: {manifest}")
    run_logged(qc_command(paths), cwd=REPO_ROOT, env=env, dry_run=dry_run)
    if not dry_run:
        assert_qc_retains_full_dataset(paths, json.loads(manifest.read_text()))


def prepare_candidate_allowlist(paths: ScenePaths, *, dry_run: bool) -> None:
    """Make QC an eligibility signal, never an implicit reconstruction split.

    The full 1487-image posed set remains the target set in the chart coverage
    objective and remains the dense reconstruction set.  Only hard-reject
    records are denied *Chart* status.  Persisting the list makes that
    distinction inspectable and prevents later selection calls from accidentally
    treating the QC-copy's image count as the candidate pool.
    """
    audit = json.loads(paths.audit_json.read_text(encoding="utf-8"))
    all_names = sorted(assert_audit_covers_full_dataset(paths, audit))
    allowed = _expected_chart_candidate_names(paths, audit)
    if not all_names:
        raise RuntimeError("View audit contains no train images")
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
                "hard_reject_chart_count": len(all_names) - len(allowed),
                "audit": str(paths.audit_json),
                "audit_sha256": _file_sha256(paths.audit_json),
                "all_train_image_set_sha256": _name_set_sha256(all_names),
                "candidate_image_set_sha256": _name_set_sha256(allowed),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _selection_config_sha256(paths: ScenePaths, config: CambridgeG4Config) -> str:
    """Hash precisely the selector knobs, excluding unrelated trainer settings."""
    payload = {
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "requested_charts": config.requested_charts,
        "minimum_charts": config.minimum_charts,
        "semantic_mask_indices": [0],
        "max_semantic_invalid_ratio": config.max_chart_semantic_invalid_ratio,
        "static_support_mask_indices": [0, 1, 2, 3] if paths.tree_mask_pickle is not None else [0, 1, 2],
        "min_static_support_ratio": config.min_chart_static_support_ratio,
        "quality_mask_indices": [0, 1, 2],
        "min_sharpness": config.min_sharpness,
        "max_extreme_ratio": config.max_extreme_ratio,
        "max_shadow_ratio": config.max_shadow_ratio,
        "min_lower_mean_intensity": config.min_lower_mean_intensity,
        "min_valid_ratio": 0.0,
        "view_clusters": config.view_clusters,
        "candidate_min_views_per_cluster": config.candidate_min_views_per_cluster,
        "joint_min_views_per_cluster": config.joint_min_views_per_cluster,
        "gate_reserves_per_vulnerable_support": config.gate_reserves_per_vulnerable_support,
        "gate_failure_budget": config.gate_failure_budget,
        "min_baseline_ratio": config.min_baseline_ratio,
        "min_global_pose_distance": config.min_global_pose_distance,
        "max_borrowed_support_distance": config.max_borrowed_support_distance,
        "coverage_objective": "target_kcenter",
        "candidate_reliability_weight": config.chart_candidate_reliability_weight,
        "min_views_per_sequence": config.min_chart_views_per_sequence,
        "post_gate_min_views_per_sequence": config.post_gate_min_chart_views_per_sequence,
        "sequence_gate_failure_budget": config.chart_sequence_gate_failure_budget,
        "sequence_coverage_mode": config.chart_sequence_coverage_mode,
        "include_ends": False,
        "tree_mask_index": 3 if paths.tree_mask_pickle is not None else -1,
        "max_tree_invalid_ratio": config.max_chart_tree_ratio,
        "tree_candidate_policy": (
            config.chart_tree_candidate_policy if paths.tree_mask_pickle is not None else "ignore"
        ),
        "quality_candidate_policy": config.chart_quality_candidate_policy,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def selection_input_provenance(paths: ScenePaths, config: CambridgeG4Config) -> dict:
    """Immutable inputs which must be identical for a reusable Chart set.

    This is intentionally stricter than a policy-version check.  The selector
    operates on normalized poses, semantic/tree masks and the audit-derived
    candidate set; reusing a prior result after any one of those changes is a
    hidden multi-variable experiment.
    """
    qc_manifest_path = paths.qc_dataset / "qc_manifest.json"
    if not qc_manifest_path.is_file():
        raise FileNotFoundError(f"QC manifest is missing: {qc_manifest_path}")
    assert_qc_retains_full_dataset(
        paths,
        json.loads(qc_manifest_path.read_text(encoding="utf-8")),
    )
    source_names = staged_image_names(paths.full_dataset)
    qc_names = staged_image_names(paths.qc_dataset)
    allowed_names = assert_candidate_allowlist_matches_audit(paths)
    sparse = paths.qc_dataset / "sparse" / "0"
    tree_mask = paths.tree_mask_pickle
    return {
        "schema_version": 1,
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "selection_config_sha256": _selection_config_sha256(paths, config),
        "source_image_set_sha256": _name_set_sha256(source_names),
        "qc_image_set_sha256": _name_set_sha256(qc_names),
        "qc_pose_image_set_sha256": _name_set_sha256(_colmap_text_image_names(paths.qc_dataset)),
        "qc_images_txt_sha256": _file_sha256(sparse / "images.txt"),
        "qc_images_bin_sha256": _file_sha256(sparse / "images.bin"),
        "audit_sha256": _file_sha256(paths.audit_json),
        "candidate_allowlist_sha256": _file_sha256(paths.candidate_allowlist),
        "mask_pickle_size": paths.mask_pickle.stat().st_size,
        "mask_pickle_mtime_ns": paths.mask_pickle.stat().st_mtime_ns,
        "tree_mask_pickle_size": tree_mask.stat().st_size if tree_mask is not None else None,
        "tree_mask_pickle_mtime_ns": tree_mask.stat().st_mtime_ns if tree_mask is not None else None,
    }


def assert_valid_chart_selection(
    paths: ScenePaths,
    config: CambridgeG4Config,
    selection: dict,
    *,
    expected_provenance: dict | None,
) -> None:
    """Reject a syntactically valid but causally stale or incomplete selection."""
    names = [str(name) for name in selection.get("image_names", [])]
    indices = selection.get("image_idx", [])
    if not names or len(names) != len(indices):
        raise RuntimeError("Chart selection must contain equally sized non-empty image_names/image_idx")
    if len(names) != len(set(names)):
        raise RuntimeError("Chart selection contains duplicate image names")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise RuntimeError("Chart selection image_idx must contain integer camera indices")
    actual_count = selection.get("actual_n_images", selection.get("n_images"))
    if actual_count != len(names):
        raise RuntimeError(
            "Chart selection count does not match its image list: "
            f"declared={actual_count}, names={len(names)}"
        )
    if not config.minimum_charts <= len(names) <= config.requested_charts:
        raise RuntimeError(
            "Chart selection is outside the strict requested/fallback range: "
            f"selected={len(names)}, minimum={config.minimum_charts}, requested={config.requested_charts}"
        )

    qc_ordered_names = sorted(staged_image_names(paths.qc_dataset))
    invalid_indices = [index for index in indices if index < 0 or index >= len(qc_ordered_names)]
    if invalid_indices:
        raise RuntimeError(f"Chart selection has out-of-range camera index: {invalid_indices[:3]}")
    mismatched = [
        (index, name, qc_ordered_names[index])
        for index, name in zip(indices, names)
        if qc_ordered_names[index] != name
    ]
    if mismatched:
        raise RuntimeError(
            "Chart selection indices no longer name the current QC cameras: "
            f"{mismatched[:1]}"
        )
    allowed = set(assert_candidate_allowlist_matches_audit(paths))
    disallowed = sorted(set(names) - allowed)
    if disallowed:
        raise RuntimeError(
            "Chart selection contains an audit-ineligible view: "
            f"{disallowed[:3]}"
        )

    # The primary set remains the object selected for coverage.  A reserve is
    # only allowed to enter MASt3R alignment if it is a separately audited
    # clean, pose-near substitute for a gate-critical primary support.  Do not
    # let an arbitrary extra Chart bypass the semantic/tree/quality filters by
    # hiding it in ``--image_idx``.
    if (
        config.gate_reserves_per_vulnerable_support > 0
        and config.joint_min_views_per_cluster >= 2
    ):
        reserves = selection.get("gate_replacement_reserves")
        if not isinstance(reserves, dict):
            raise RuntimeError("Chart selection is missing its gate-replacement reserve audit")
        # A feedback re-selection can keep the global floor at one rejected
        # Chart while raising the certified failure budget only in clusters
        # where the preceding target-depth gate actually observed correlated
        # failures.  Treat that as the joint certificate it is; otherwise a
        # perfectly valid feedback selection would be rejected merely because
        # its global floor remains one.
        failure_budget_by_cluster = reserves.get("gate_failure_budget_by_cluster", {})
        adaptive_joint_budget = False
        if isinstance(failure_budget_by_cluster, dict):
            adaptive_joint_budget = any(
                isinstance(value, int) and not isinstance(value, bool) and value > 1
                for value in failure_budget_by_cluster.values()
            )
        expected_reserve_method = (
            "baseline_primary_plus_joint_gate_resilience"
            if config.gate_failure_budget > 1 or adaptive_joint_budget
            else "baseline_primary_plus_pose_near_gate_replacement"
        )
        if reserves.get("method") != expected_reserve_method:
            raise RuntimeError("Chart selection used an unknown gate-replacement reserve method")
        expected_reserve_policy = {
            "post_gate_min_views_per_cluster": config.joint_min_views_per_cluster,
            "min_baseline_ratio": config.min_baseline_ratio,
            "reserve_max_pose_distance": config.min_global_pose_distance,
            "gate_failure_budget": config.gate_failure_budget,
        }
        if expected_reserve_method == "baseline_primary_plus_pose_near_gate_replacement":
            expected_reserve_policy["reserves_per_vulnerable_support"] = (
                config.gate_reserves_per_vulnerable_support
            )
        mismatched_reserve_policy = {
            key: (reserves.get(key), value)
            for key, value in expected_reserve_policy.items()
            if reserves.get(key) != value
        }
        if mismatched_reserve_policy:
            raise RuntimeError(
                "Chart gate-replacement reserve policy does not match this run: "
                f"{mismatched_reserve_policy}"
            )
        certified = reserves.get("certified_for_gate_failure_budget")
        if certified is None and config.gate_failure_budget == 1:
            certified = reserves.get("certified_for_one_primary_gate_rejection")
        if certified is not True:
            raise RuntimeError(
                "Chart gate-replacement reserve audit is not certified for the configured "
                f"gate-failure budget ({config.gate_failure_budget})"
            )
        reserve_indices = reserves.get("reserve_image_idx", [])
        reserve_names = [str(name) for name in reserves.get("reserve_image_names", [])]
        alignment_indices = reserves.get("alignment_image_idx", [])
        alignment_names = [str(name) for name in reserves.get("alignment_image_names", [])]
        if len(reserve_indices) != len(reserve_names) or len(reserve_indices) != len(set(reserve_indices)):
            raise RuntimeError("Chart gate-replacement reserves have duplicate or mismatched name/index entries")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in reserve_indices):
            raise RuntimeError("Chart gate-replacement reserve indices must be integers")
        reserve_coordinate_errors = [
            (index, name, qc_ordered_names[index] if 0 <= index < len(qc_ordered_names) else None)
            for index, name in zip(reserve_indices, reserve_names)
            if index < 0 or index >= len(qc_ordered_names) or qc_ordered_names[index] != name
        ]
        if reserve_coordinate_errors:
            raise RuntimeError(
                "Chart gate-replacement reserve indices do not match the current QC cameras: "
                f"{reserve_coordinate_errors[:1]}"
            )
        if set(reserve_indices) & set(indices):
            raise RuntimeError("Chart gate-replacement reserve duplicates a primary Chart")
        candidate_pool = {str(name) for name in selection.get("candidate_pool_names", [])}
        if not candidate_pool:
            raise RuntimeError("Chart gate-replacement reserve audit has no clean candidate pool")
        ineligible_reserves = sorted(set(reserve_names) - candidate_pool)
        if ineligible_reserves:
            raise RuntimeError(
                "Chart gate-replacement reserve bypassed the clean candidate filters: "
                f"{ineligible_reserves[:3]}"
            )
        expected_alignment_indices = sorted(set(indices) | set(reserve_indices))
        expected_alignment_names = [qc_ordered_names[index] for index in expected_alignment_indices]
        if alignment_indices != expected_alignment_indices or alignment_names != expected_alignment_names:
            raise RuntimeError(
                "Chart gate-replacement alignment set is not exactly primary Charts plus verified reserves"
            )

    coverage = selection.get("coverage")
    clusters = coverage.get("clusters", []) if isinstance(coverage, dict) else []
    failed_clusters = [
        diagnostic.get("cluster")
        for diagnostic in clusters
        if diagnostic.get("coverage_satisfied") is not True
    ]
    if len(clusters) != config.view_clusters or failed_clusters:
        raise RuntimeError(
            "Chart coverage audit is absent or failed: "
            f"clusters={len(clusters)}, expected={config.view_clusters}, failed={failed_clusters}"
        )
    if coverage.get("coverage_objective") != "target_kcenter":
        raise RuntimeError("Chart selection did not use the required target_kcenter objective")
    if coverage.get("min_views_per_cluster") != config.candidate_min_views_per_cluster:
        raise RuntimeError(
            "Chart selection candidate reserve does not match the requested policy: "
            f"selected={coverage.get('min_views_per_cluster')}, "
            f"requested={config.candidate_min_views_per_cluster}"
        )
    if coverage.get("candidate_reliability_weight") != config.chart_candidate_reliability_weight:
        raise RuntimeError("Chart selection used a different candidate reliability weight")
    if coverage.get("min_views_per_sequence") != config.min_chart_views_per_sequence:
        raise RuntimeError("Chart selection used a different same-sequence support count")
    if (
        coverage.get("post_gate_min_views_per_sequence")
        != config.post_gate_min_chart_views_per_sequence
    ):
        raise RuntimeError("Chart selection used a different post-gate same-sequence support count")
    if (
        coverage.get("sequence_gate_failure_budget")
        != config.chart_sequence_gate_failure_budget
    ):
        raise RuntimeError("Chart selection used a different same-sequence gate-failure budget")
    if coverage.get("sequence_coverage_mode") != config.chart_sequence_coverage_mode:
        raise RuntimeError("Chart selection used a different same-sequence coverage mode")
    sequence_records = coverage.get("sequences", [])
    if config.min_chart_views_per_sequence > 0:
        if not isinstance(sequence_records, list) or not sequence_records:
            raise RuntimeError("Chart selection is missing its same-sequence coverage audit")
        failed_sequences = [
            record.get("sequence")
            for record in sequence_records
            if record.get("applicable") is not False and record.get("passed") is not True
        ]
        resilience_failures = [
            record.get("sequence")
            for record in sequence_records
            if (
                record.get("applicable") is not False
                and record.get("gate_resilience", {}).get("passed") is not True
            )
        ]
        if config.chart_sequence_coverage_mode == "strict" and failed_sequences:
            raise RuntimeError(
                "Chart selection has a temporal corridor without an independent "
                f"static baseline: {failed_sequences}"
            )
        if config.chart_sequence_coverage_mode == "strict" and resilience_failures:
            raise RuntimeError(
                "Chart selection is not resilient to the configured same-sequence "
                f"hard-gate failures: {resilience_failures}"
            )
    static_support = selection.get("static_support_filter", {})
    if static_support.get("min_support_ratio") != config.min_chart_static_support_ratio:
        raise RuntimeError("Chart selection used a different static-support threshold")
    expected_static_indices = [0, 1, 2, 3] if paths.tree_mask_pickle is not None else [0, 1, 2]
    if static_support.get("mask_indices") != expected_static_indices:
        raise RuntimeError("Chart selection used a different static-support mask policy")
    if paths.tree_mask_pickle is not None:
        tree_filter = selection.get("tree_filter", {})
        if tree_filter.get("candidate_policy") != config.chart_tree_candidate_policy:
            raise RuntimeError("Chart selection used a different tree candidate policy")
    quality_filter = selection.get("quality_filter", {})
    if quality_filter.get("candidate_policy") != config.chart_quality_candidate_policy:
        raise RuntimeError("Chart selection used a different quality candidate policy")
    allowed_audit = selection.get("allowed_name_filter", {})
    if allowed_audit.get("selected_all_allowed") is not True:
        raise RuntimeError("Chart selection does not prove that every selected view was allowlisted")
    if expected_provenance is not None and selection.get("input_provenance") != expected_provenance:
        raise RuntimeError("Chart selection input provenance does not match the current QC/audit inputs")


def prepare_selection(
    paths: ScenePaths,
    config: CambridgeG4Config,
    env: dict[str, str],
    dry_run: bool,
    force: bool,
) -> dict:
    expected_provenance = None if dry_run else selection_input_provenance(paths, config)
    if paths.selection_json.exists() and not force:
        existing = json.loads(paths.selection_json.read_text())
        if (
            existing.get("audit_status_filter") == "hard_reject_removed_then_explicit_quality"
            and existing.get("requested_n_images") == config.requested_charts
            and existing.get("selection_policy_version") == SELECTION_POLICY_VERSION
            and existing.get("input_provenance") == expected_provenance
        ):
            if not dry_run:
                assert_valid_chart_selection(
                    paths,
                    config,
                    existing,
                    expected_provenance=expected_provenance,
                )
            print(f"[SKIP] Reusing chart selection: {paths.selection_json}")
            return existing
        print(f"[INFO] Replacing stale or incompatible chart selection: {paths.selection_json}")

    failures = []
    minimum_count = max(
        config.minimum_charts,
        config.view_clusters * config.candidate_min_views_per_cluster,
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
            selection["candidate_reserve_policy"] = {
                "primary_min_views_per_cluster": config.candidate_min_views_per_cluster,
                "post_gate_min_views_per_cluster": config.joint_min_views_per_cluster,
                "min_views_per_sequence": config.min_chart_views_per_sequence,
                "sequence_coverage_mode": config.chart_sequence_coverage_mode,
                "min_static_support_ratio": config.min_chart_static_support_ratio,
                "tree_candidate_policy": config.chart_tree_candidate_policy,
                "gate_reserves_per_vulnerable_support": (
                    config.gate_reserves_per_vulnerable_support
                ),
                "gate_failure_budget": config.gate_failure_budget,
                "rationale": (
                    "preserve an independently-baselined post-gate pair under the "
                    "configured correlated hard-gate failure budget"
                ),
            }
            selection["strict_fallback_used"] = count != config.requested_charts
            selection["failed_larger_counts"] = failures
            selection["input_provenance"] = expected_provenance
            assert_valid_chart_selection(
                paths,
                config,
                selection,
                expected_provenance=expected_provenance,
            )
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


def joint_selection_output_path(output: Path, config: CambridgeG4Config) -> Path:
    if config.joint_chart_count is None:
        raise ValueError("joint_chart_count is required for a quality-aware Chart selection")
    return (
        output
        / "mast3r_sfm"
        / "quality_aware_selection"
        / f"chart_selection_quality_n{config.joint_chart_count}.json"
    )


def joint_selection_command(
    paths: ScenePaths,
    config: CambridgeG4Config,
    *,
    output: Path,
) -> list[str]:
    """Run the post-alignment selection whose result is consumed by geometry."""
    selection_path = joint_selection_output_path(output, config)
    # The hard gate aligned Charts with the tree channel included.  Reverting
    # to only [0,1,2] here would let tree pixels affect cross-view reliability
    # and 3-D coverage even though they were deliberately excluded from the
    # geometric anchor policy.
    selection_mask = paths.tree_mask_pickle or paths.mask_pickle
    selection_mask_indices = [0, 1, 2, 3] if paths.tree_mask_pickle is not None else [0, 1, 2]
    return [
        sys.executable,
        "scripts/select_quality_aware_charts.py",
        "--scene-path",
        str(output / "mast3r_sfm"),
        "--target-scene-path",
        str(paths.qc_dataset),
        "--selection",
        str(paths.selection_json),
        "--gate-report",
        str(output / "mast3r_sfm" / "aligned_chart_conflict_gate.json"),
        "--view-audit-csv",
        str(paths.audit_dir / "view_metrics.csv"),
        "--mask-pickle",
        str(selection_mask),
        "--mask-dataset-path",
        str(paths.qc_dataset),
        "--mask-indices",
        *(str(index) for index in selection_mask_indices),
        "--counts",
        str(config.joint_chart_count),
        "--min-pose-support",
        str(config.joint_min_views_per_cluster),
        "--min-sequence-support",
        str(
            config.post_gate_min_chart_views_per_sequence
            if config.chart_sequence_coverage_mode == "strict"
            else 0
        ),
        # Measured overlap avoids treating pose-near but occlusion-disjoint
        # charts as photometric evidence for each other.
        "--neighbor-selection",
        "overlap",
        "--min-neighbor-overlap",
        "0.05",
        "--output-dir",
        str(selection_path.parent),
    ]


def joint_selection_input_provenance(
    paths: ScenePaths,
    config: CambridgeG4Config,
    *,
    output: Path,
) -> dict:
    """Identity of all geometry evidence behind a post-gate chart subset."""
    mast3r_scene = output / "mast3r_sfm"
    gate = mast3r_scene / "aligned_chart_conflict_gate.json"
    charts = mast3r_scene / "charts_data.npz"
    cameras = mast3r_scene / "cameras.json"
    if not gate.is_file() or not charts.is_file() or not cameras.is_file():
        raise FileNotFoundError(
            "A joint Chart selection requires completed alignment and gate files under "
            f"{mast3r_scene}"
        )
    base_selection = paths.selection_json
    audit_csv = paths.audit_dir / "view_metrics.csv"
    if not base_selection.is_file() or not audit_csv.is_file():
        raise FileNotFoundError(
            "Joint Chart selection is missing its base selection or view audit: "
            f"selection={base_selection}, audit={audit_csv}"
        )
    command = joint_selection_command(paths, config, output=output)
    return {
        "schema_version": 1,
        "joint_chart_count": config.joint_chart_count,
        "base_selection_sha256": _file_sha256(base_selection),
        "base_selection_input_provenance": selection_input_provenance(paths, config),
        "gate_report_sha256": _file_sha256(gate),
        "charts_data_sha256": _file_sha256(charts),
        "cameras_json_sha256": _file_sha256(cameras),
        "view_audit_csv_sha256": _file_sha256(audit_csv),
        "joint_command_sha256": hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def assert_valid_joint_chart_selection(
    paths: ScenePaths,
    config: CambridgeG4Config,
    selection: dict,
    *,
    output: Path,
    expected_provenance: dict | None,
) -> None:
    """Verify that the final active charts are gate-clean and coverage-valid."""
    if config.joint_chart_count is None:
        raise ValueError("No joint chart count was configured")
    names = [str(name) for name in selection.get("image_names", [])]
    indices = selection.get("image_idx", [])
    if len(names) != config.joint_chart_count or len(indices) != len(names):
        raise RuntimeError(
            "Joint Chart selection has the wrong number of name/index entries: "
            f"names={len(names)}, indices={len(indices)}, expected={config.joint_chart_count}"
        )
    if len(set(names)) != len(names) or len(set(indices)) != len(indices):
        raise RuntimeError("Joint Chart selection contains duplicate views")
    ordered_names = sorted(staged_image_names(paths.qc_dataset))
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise RuntimeError("Joint Chart selection has non-integer image_idx entries")
    coordinate_errors = [
        (name, index, ordered_names[index] if 0 <= index < len(ordered_names) else None)
        for name, index in zip(names, indices)
        if index < 0 or index >= len(ordered_names) or ordered_names[index] != name
    ]
    if coordinate_errors:
        raise RuntimeError(
            "Joint Chart selection is not expressed in the current QC camera order: "
            f"{coordinate_errors[:1]}"
        )
    allowed = set(assert_candidate_allowlist_matches_audit(paths))
    invalid = sorted(set(names) - allowed)
    if invalid:
        raise RuntimeError(f"Joint Chart selection contains audit-ineligible view(s): {invalid[:3]}")

    gate = json.loads(
        (output / "mast3r_sfm" / "aligned_chart_conflict_gate.json").read_text(
            encoding="utf-8"
        )
    )
    rejected = {
        Path(str(record["image_name"])).name
        for record in gate.get("records", [])
        if record.get("rejected")
    }
    conflict = sorted(set(names) & rejected)
    if conflict:
        raise RuntimeError(
            "Joint Chart selection tried to reactivate a depth-conflict or unsupported chart: "
            f"{conflict[:3]}"
        )

    joint = selection.get("quality_aware_selection", {})
    constraints = joint.get("constraints", {})
    diagnostics = constraints.get("diagnostics", {})
    clusters = diagnostics.get("clusters", [])
    min_support = constraints.get("min_views_per_cluster")
    if isinstance(min_support, bool) or not isinstance(min_support, int) or min_support < 1:
        raise RuntimeError("Joint Chart selection has no valid minimum-support constraint")
    if min_support != config.joint_min_views_per_cluster:
        raise RuntimeError(
            "Joint Chart selection used the wrong post-gate support requirement: "
            f"selected={min_support}, requested={config.joint_min_views_per_cluster}"
        )
    invalid_clusters = [
        record.get("cluster")
        for record in clusters
        if record.get("support_count", 0) < min_support
        or record.get("baseline_satisfied") is not True
    ]
    references = diagnostics.get("references", [])
    invalid_references = [
        record.get("reference") for record in references if record.get("passed") is not True
    ]
    invalid_sequences: list[str | None] = []
    missing_sequences: list[str] = []
    if config.chart_sequence_coverage_mode == "strict":
        requested_sequence_support = config.post_gate_min_chart_views_per_sequence
        selected_sequence_support = constraints.get("min_views_per_sequence")
        if (
            isinstance(selected_sequence_support, bool)
            or not isinstance(selected_sequence_support, int)
            or selected_sequence_support != requested_sequence_support
        ):
            raise RuntimeError(
                "Joint Chart selection used the wrong same-sequence post-gate support "
                f"requirement: selected={selected_sequence_support}, "
                f"requested={requested_sequence_support}"
            )
        sequence_records = diagnostics.get("sequences", [])
        if not isinstance(sequence_records, list):
            raise RuntimeError("Joint Chart selection has no same-sequence support diagnostics")
        invalid_sequences = [
            record.get("sequence")
            for record in sequence_records
            if record.get("support_count", 0) < requested_sequence_support
            or record.get("baseline_satisfied") is not True
            or record.get("passed") is not True
        ]
        base_selection = json.loads(paths.selection_json.read_text(encoding="utf-8"))
        base_sequence_records = base_selection.get("coverage", {}).get("sequences", [])
        required_sequences = {
            str(record["sequence"])
            for record in base_sequence_records
            if isinstance(record, dict)
            and record.get("applicable", True) is not False
            and isinstance(record.get("sequence"), str)
        }
        selected_sequences = {
            str(record["sequence"])
            for record in sequence_records
            if isinstance(record, dict) and isinstance(record.get("sequence"), str)
        }
        if not required_sequences:
            raise RuntimeError(
                "Strict same-sequence coverage was requested but the broad Chart "
                "selection has no applicable sequence audit"
            )
        missing_sequences = sorted(required_sequences - selected_sequences)
    if (
        len(clusters) != config.view_clusters
        or invalid_clusters
        or invalid_references
        or invalid_sequences
        or missing_sequences
    ):
        raise RuntimeError(
            "Joint Chart coverage/support audit failed: "
            f"clusters={len(clusters)}, expected={config.view_clusters}, "
            f"invalid_clusters={invalid_clusters}, invalid_references={invalid_references}, "
            f"invalid_sequences={invalid_sequences}, missing_sequences={missing_sequences}"
        )
    if joint.get("hard_gate_active_count", 0) < len(names):
        raise RuntimeError("Joint Chart selection claims more active charts than passed the depth gate")
    if not Path(str(selection.get("quality_score_path", ""))).is_file():
        raise RuntimeError("Joint Chart selection is missing its cross-view quality audit")
    if expected_provenance is not None and selection.get("joint_input_provenance") != expected_provenance:
        raise RuntimeError("Joint Chart selection provenance is stale relative to aligned geometry")


def prepare_joint_chart_selection(
    paths: ScenePaths,
    config: CambridgeG4Config,
    env: dict[str, str],
    *,
    output: Path,
    dry_run: bool,
) -> Path | None:
    """Produce/reuse a verified quality-aware subset after the hard depth gate."""
    if config.joint_chart_count is None:
        return None
    selection_path = joint_selection_output_path(output, config)
    expected = None if dry_run else joint_selection_input_provenance(paths, config, output=output)
    if selection_path.is_file() and not dry_run:
        existing = json.loads(selection_path.read_text(encoding="utf-8"))
        if existing.get("joint_input_provenance") == expected:
            assert_valid_joint_chart_selection(
                paths,
                config,
                existing,
                output=output,
                expected_provenance=expected,
            )
            print(f"[SKIP] Reusing joint Chart selection: {selection_path}")
            return selection_path
        print(f"[INFO] Recomputing stale joint Chart selection: {selection_path}")

    run_logged(
        joint_selection_command(paths, config, output=output),
        cwd=REPO_ROOT,
        env=env,
        log_path=selection_path.parent / "joint_selection.log",
        dry_run=dry_run,
    )
    if dry_run:
        return selection_path
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["joint_input_provenance"] = expected
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    assert_valid_joint_chart_selection(
        paths,
        config,
        selection,
        output=output,
        expected_provenance=expected,
    )
    print(
        f"[INFO] Verified {config.joint_chart_count} active Charts after depth gate: {selection_path}",
        flush=True,
    )
    return selection_path


def post_gate_coverage_path(output: Path) -> Path:
    return output / "mast3r_sfm" / "post_gate_coverage_audit.json"


def post_gate_coverage_command(
    paths: ScenePaths,
    config: CambridgeG4Config,
    *,
    output: Path,
    selection_path: Path,
) -> list[str]:
    """Audit the *accepted* Charts before any plane or Gaussian stage consumes them."""
    return [
        sys.executable,
        "scripts/audit_gated_chart_coverage.py",
        "--selection",
        str(selection_path),
        "--gate-report",
        str(output / "mast3r_sfm" / "aligned_chart_conflict_gate.json"),
        "--scene-path",
        str(paths.qc_dataset),
        "--minimum-active",
        str(max(
            config.minimum_charts,
            config.view_clusters * config.joint_min_views_per_cluster,
        )),
        "--output",
        str(post_gate_coverage_path(output)),
    ]


def require_valid_post_gate_coverage(
    paths: ScenePaths,
    config: CambridgeG4Config,
    *,
    output: Path,
    selection_path: Path,
    env: dict[str, str],
    dry_run: bool,
) -> Path | None:
    """Run a fail-closed active-Chart audit after the target-depth gate."""
    if not config.require_post_gate_coverage:
        return None
    gate = output / "mast3r_sfm" / "aligned_chart_conflict_gate.json"
    if not dry_run and not gate.is_file():
        raise FileNotFoundError(
            "The Chart frontend has no hard depth-gate report; refusing to let "
            f"planes/2DGS consume unaudited geometry: {gate}"
        )
    audit_path = post_gate_coverage_path(output)
    run_logged(
        post_gate_coverage_command(
            paths,
            config,
            output=output,
            selection_path=selection_path,
        ),
        cwd=REPO_ROOT,
        env=env,
        log_path=output / "logs" / "post_gate_coverage.log",
        dry_run=dry_run,
    )
    if dry_run:
        return audit_path
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if payload.get("passed") is not True:
        failed = [
            name for name, passed in payload.get("checks", {}).items()
            if passed is not True
        ]
        raise RuntimeError(
            "Accepted Chart set failed post-gate coverage; do not run plane or Gaussian "
            f"training. Failed checks: {failed}. Audit: {audit_path}"
        )
    print(f"[INFO] Post-gate Chart coverage passed: {audit_path}", flush=True)
    return audit_path


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
    disallowed = sorted(set(names) - set(assert_candidate_allowlist_matches_audit(paths)))
    if disallowed:
        raise ValueError(
            "External selection contains chart-ineligible hard-reject view(s): "
            + ", ".join(disallowed[:5])
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
    active_chart_selection: Path | None = None,
) -> Path:
    output = paths.screen_output if screen_only else paths.full_output
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "scene": scene,
        "mode": "screen7k" if screen_only else f"full{config.final_iterations}",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "mask_policy": {
            "chart_frame_indices": [0],
            "chart_max_semantic_invalid_ratio": config.max_chart_semantic_invalid_ratio,
            "chart_static_support_indices": (
                [0, 1, 2, 3] if paths.tree_mask_pickle is not None else [0, 1, 2]
            ),
            "chart_min_static_support_ratio": config.min_chart_static_support_ratio,
            "chart_tree_candidate_policy": config.chart_tree_candidate_policy,
            "chart_quality_candidate_policy": config.chart_quality_candidate_policy,
            "chart_candidate_reliability_weight": config.chart_candidate_reliability_weight,
            "chart_min_views_per_sequence": config.min_chart_views_per_sequence,
            "chart_post_gate_min_views_per_sequence": config.post_gate_min_chart_views_per_sequence,
            "chart_sequence_gate_failure_budget": config.chart_sequence_gate_failure_budget,
            "chart_sequence_coverage_mode": config.chart_sequence_coverage_mode,
            "require_post_gate_coverage": config.require_post_gate_coverage,
            "quality_indices": [0, 1, 2],
            "alignment_indices": [0, 1, 2, 3] if paths.tree_mask_pickle is not None else [0, 1, 2],
            "rgb_loss_indices": [0, 1, 2],
            "rgb_supervision_profile": config.rgb_supervision_profile,
            "rgb_sampling_policy": config.rgb_sampling_policy,
            "chart_geometry_sampling_policy": config.chart_geometry_sampling_policy,
            "densification_view_policy": config.densification_view_policy,
            "chart_geometry_prior_weight": config.chart_geometry_prior_weight,
            "geometry_loss_indices": [0, 1, 2],
            "alpha_suppression_indices": [1],
            "semantic_alpha_weight": config.semantic_alpha_weight,
            "white_background": config.white_background,
            "warp_downsample_pixel_grid_size": config.warp_downsample_pixel_grid_size,
            "downweight_input_view_color_loss": config.downweight_input_view_color_loss,
            "dense_final_only": config.dense_final_only,
            "real_rgb_supervision_from_iteration": (
                "final_refinement_only" if config.dense_final_only else "iteration_1"
            ),
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
        "alignment_chart_policy": {
            "primary_chart_count": len(selection.get("image_idx", [])),
            "gate_replacement_reserve_count": len(
                selection.get("gate_replacement_reserves", {}).get("reserve_image_idx", [])
            ),
            "alignment_chart_count": len(
                selection.get("gate_replacement_reserves", {}).get(
                    "alignment_image_idx", selection.get("image_idx", [])
                )
            ),
            "gate_replacement_certified": selection.get(
                "gate_replacement_reserves", {}
            ).get(
                "certified_for_gate_failure_budget",
                selection.get("gate_replacement_reserves", {}).get(
                    "certified_for_one_primary_gate_rejection"
                ),
            ),
            "gate_failure_budget": selection.get(
                "gate_replacement_reserves", {}
            ).get("gate_failure_budget"),
        },
        "selection": selection,
        "active_chart_selection": (
            str(active_chart_selection.resolve())
            if active_chart_selection is not None
            else None
        ),
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
            "audit", "qc", "select", "prepare", "frontend", "quality-select", "screen", "full",
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
    parser.add_argument(
        "--view-clusters",
        type=int,
        default=8,
        help="Pose/direction coverage cells used by both candidate and post-gate selection.",
    )
    parser.add_argument(
        "--candidate-min-views-per-cluster",
        type=int,
        default=2,
        help=(
            "Redundant clean Chart supports required in every pose cluster before "
            "MASt3R alignment and its hard depth gate."
        ),
    )
    parser.add_argument(
        "--joint-min-views-per-cluster",
        type=int,
        default=2,
        help=(
            "Baseline-separated hard-gated supports required in every pose cluster "
            "for the active post-gate geometry subset."
        ),
    )
    parser.add_argument(
        "--gate-reserves-per-vulnerable-support",
        type=int,
        default=1,
        help=(
            "Clean pose-near alternate Charts appended to MASt3R alignment for "
            "each primary support whose single gate rejection would remove its "
            "baseline pair. Set to 0 only for an explicit legacy ablation."
        ),
    )
    parser.add_argument(
        "--gate-failure-budget",
        type=int,
        default=1,
        help=(
            "Correlated aligned-Chart target-gate failures that every pose cluster's "
            "primary-plus-reserve set must survive. Values above one activate the "
            "joint reserve certification."
        ),
    )
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument(
        "--strict-calibrated-poses",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Freeze calibrated camera translations and global scale inside MASt3R "
            "alignment; use only as an explicit frontend ablation."
        ),
    )
    parser.add_argument(
        "--per-view-calibrated-intrinsics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep per-camera calibrated intrinsics inside MASt3R instead of "
            "the historical shared-focal parameterization."
        ),
    )
    parser.add_argument(
        "--chart-alignment-seed",
        type=int,
        default=0,
        help="Fixed Chart-alignment initialization seed for paired ablations.",
    )
    parser.add_argument(
        "--mast3r-sparse-export-stride",
        type=int,
        default=4,
        help=(
            "Regular pixel stride for MASt3R's diagnostic COLMAP sparse export. "
            "The full Chart pointmaps are unchanged."
        ),
    )
    parser.add_argument("--rgb-loss-type", choices=["l1", "charbonnier"], default="l1")
    parser.add_argument(
        "--rgb-supervision-profile",
        choices=["g4_tree_masked", "full_rgb", "ulfloc_legacy"],
        default="g4_tree_masked",
        help=(
            "RGB-only supervision protocol. Geometry Chart/tree/plane masks and "
            "alpha terms remain unchanged under full_rgb and ulfloc_legacy."
        ),
    )
    parser.add_argument(
        "--rgb-sampling-policy",
        choices=["legacy_interleaved", "all_train_importance"],
        default="all_train_importance",
        help=(
            "Correct the RGB expectation to uniform all-train cameras while retaining "
            "Chart geometry updates, or run the legacy interleaved sampling ablation."
        ),
    )
    parser.add_argument(
        "--chart-geometry-sampling-policy",
        choices=["legacy_all_input", "active_only"],
        default="active_only",
        help=(
            "Sample every aligned Chart only for a legacy ablation, or ensure every "
            "geometry iteration uses a gate/quality-active Chart."
        ),
    )
    parser.add_argument(
        "--densification-view-policy",
        choices=["legacy_current", "dense_only"],
        default="legacy_current",
        help=(
            "Keep historical current-view topology statistics, or retain Chart "
            "geometry losses while collecting split/prune statistics only from "
            "the all-real dense camera stream."
        ),
    )
    parser.add_argument(
        "--chart-geometry-prior-weight",
        type=float,
        default=1.0,
        help=(
            "Scale Chart-exclusive geometric priors while retaining the exact Chart "
            "sampling schedule; zero is the strict no-Chart-prior control."
        ),
    )
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
        "--dense-final-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Explicit Chart-only 7k ablation: delay all-train dense RGB/depth "
            "supervision until final refinement. The default uses every real "
            "training camera from iteration 1."
        ),
    )
    parser.add_argument(
        "--joint-chart-count",
        type=int,
        default=None,
        help=(
            "After the alignment depth gate, select and hard-apply this many "
            "quality-aware Charts before plane/2DGS geometry. Run --phase frontend, "
            "then --phase quality-select, then screen/full with the same value."
        ),
    )
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
    parser.add_argument(
        "--max-chart-semantic-invalid-ratio",
        type=float,
        default=0.10,
        help=(
            "Whole-frame semantic contamination ceiling. Small occluders are handled "
            "by pixel-level static support instead of rejecting the entire Chart."
        ),
    )
    parser.add_argument(
        "--min-chart-static-support-ratio",
        type=float,
        default=0.10,
        help="Minimum static building/ground-like support fraction in a Chart frame.",
    )
    parser.add_argument(
        "--chart-tree-candidate-policy",
        choices=["hard_ratio", "soft_penalty", "ignore"],
        default="soft_penalty",
        help=(
            "Treat foliage as a hard Chart veto, a weak quality penalty, or ignore it "
            "at candidate-selection time. Geometry masking remains hard in all modes."
        ),
    )
    parser.add_argument(
        "--chart-quality-candidate-policy",
        choices=["hard_filter", "soft_penalty"],
        default="soft_penalty",
        help=(
            "Reject low-quality Chart frames only in the explicit hard-filter ablation; "
            "the outdoor default uses quality as a coverage tie-breaker."
        ),
    )
    parser.add_argument(
        "--chart-candidate-reliability-weight",
        type=float,
        default=0.03,
        help="Weak tie-break weight for static support and image quality after pose coverage.",
    )
    parser.add_argument(
        "--min-chart-views-per-sequence",
        type=int,
        default=3,
        help=(
            "Pre-gate static Chart supports reserved for each non-singleton "
            "trajectory segment."
        ),
    )
    parser.add_argument(
        "--post-gate-min-chart-views-per-sequence",
        type=int,
        default=2,
        help=(
            "Same-sequence structural supports that must survive the "
            "target-depth hard gate."
        ),
    )
    parser.add_argument(
        "--chart-sequence-gate-failure-budget",
        type=int,
        default=1,
        help=(
            "Worst-case target-depth gate rejections every temporal Chart "
            "seed set must survive."
        ),
    )
    parser.add_argument(
        "--chart-sequence-coverage-mode",
        choices=["off", "soft", "strict"],
        default="strict",
        help="Audit or hard-require same-trajectory Chart baselines before global coverage.",
    )
    parser.add_argument("--max-chart-tree-ratio", type=float, default=0.15)
    parser.add_argument(
        "--require-post-gate-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Block plane/2DGS stages unless the accepted hard-gated Charts still cover clusters and trajectories.",
    )
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
    if args.view_clusters < 1:
        raise ValueError("--view-clusters must be positive")
    if args.candidate_min_views_per_cluster < 1:
        raise ValueError("--candidate-min-views-per-cluster must be positive")
    if args.joint_min_views_per_cluster < 1:
        raise ValueError("--joint-min-views-per-cluster must be positive")
    if args.gate_reserves_per_vulnerable_support < 0:
        raise ValueError("--gate-reserves-per-vulnerable-support must be non-negative")
    if args.gate_failure_budget < 1:
        raise ValueError("--gate-failure-budget must be positive")
    if args.gate_failure_budget > 1 and args.gate_reserves_per_vulnerable_support < 1:
        raise ValueError(
            "A correlated --gate-failure-budget requires "
            "--gate-reserves-per-vulnerable-support >= 1"
        )
    if args.candidate_min_views_per_cluster < args.joint_min_views_per_cluster:
        raise ValueError(
            "--candidate-min-views-per-cluster must be at least "
            "--joint-min-views-per-cluster"
        )
    candidate_coverage_minimum = args.view_clusters * args.candidate_min_views_per_cluster
    if candidate_coverage_minimum > args.n_images:
        raise ValueError(
            "--n-images is too small for the requested candidate reserve: "
            f"{args.view_clusters} * {args.candidate_min_views_per_cluster} > {args.n_images}"
        )
    if args.semantic_alpha_weight < 0:
        raise ValueError("--semantic-alpha-weight must be non-negative")
    if not 0.0 <= args.max_chart_semantic_invalid_ratio <= 1.0:
        raise ValueError("--max-chart-semantic-invalid-ratio must be in [0, 1]")
    if not 0.0 <= args.min_chart_static_support_ratio <= 1.0:
        raise ValueError("--min-chart-static-support-ratio must be in [0, 1]")
    if args.chart_candidate_reliability_weight < 0.0:
        raise ValueError("--chart-candidate-reliability-weight must be non-negative")
    if args.min_chart_views_per_sequence < 0:
        raise ValueError("--min-chart-views-per-sequence must be non-negative")
    if (
        args.min_chart_views_per_sequence == 0
        and args.chart_sequence_coverage_mode != "off"
    ):
        raise ValueError(
            "--chart-sequence-coverage-mode requires --min-chart-views-per-sequence > 0"
        )
    if (
        args.min_chart_views_per_sequence > 0
        and args.chart_sequence_coverage_mode == "off"
    ):
        raise ValueError(
            "--min-chart-views-per-sequence requires coverage mode soft or strict"
        )
    if args.post_gate_min_chart_views_per_sequence < 0:
        raise ValueError("--post-gate-min-chart-views-per-sequence must be non-negative")
    if args.chart_sequence_gate_failure_budget < 0:
        raise ValueError("--chart-sequence-gate-failure-budget must be non-negative")
    if args.min_chart_views_per_sequence > 0:
        if args.post_gate_min_chart_views_per_sequence < 1:
            raise ValueError(
                "same-sequence coverage requires positive post-gate support"
            )
        if (
            args.min_chart_views_per_sequence
            < args.post_gate_min_chart_views_per_sequence
            + args.chart_sequence_gate_failure_budget
        ):
            raise ValueError(
                "--min-chart-views-per-sequence must contain post-gate support "
                "plus --chart-sequence-gate-failure-budget"
            )
    if args.chart_geometry_prior_weight < 0:
        raise ValueError("--chart-geometry-prior-weight must be non-negative")
    if args.final_iterations <= 0:
        raise ValueError("--final-iterations must be positive")
    if args.final_non_position_lr_decay_from < -1:
        raise ValueError("--final-non-position-lr-decay-from must be -1 or non-negative")
    if not 0.0 < args.final_non_position_lr_final_mult <= 1.0:
        raise ValueError("--final-non-position-lr-final-mult must be in (0, 1]")
    if args.min_global_plane_views < 1:
        raise ValueError("--min-global-plane-views must be at least one")
    if args.joint_chart_count is not None:
        # ``minimum_charts`` belongs to the broad candidate set.  Coupling it
        # to the final subset would make a deliberately redundant 48->24
        # candidate/active policy impossible even when every active cluster
        # satisfies the real post-gate support constraint.
        minimum_joint = args.joint_min_views_per_cluster * args.view_clusters
        if args.joint_chart_count < minimum_joint or args.joint_chart_count > args.n_images:
            raise ValueError(
                "--joint-chart-count must be at least the coverage minimum "
                f"({minimum_joint}) and no larger than --n-images ({args.n_images})"
            )
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
    if args.mast3r_sparse_export_stride < 1:
        raise ValueError("--mast3r-sparse-export-stride must be at least one")
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
        view_clusters=args.view_clusters,
        candidate_min_views_per_cluster=args.candidate_min_views_per_cluster,
        joint_min_views_per_cluster=args.joint_min_views_per_cluster,
        gate_reserves_per_vulnerable_support=args.gate_reserves_per_vulnerable_support,
        gate_failure_budget=args.gate_failure_budget,
        resolution=args.resolution,
        strict_calibrated_poses=args.strict_calibrated_poses,
        per_view_calibrated_intrinsics=args.per_view_calibrated_intrinsics,
        chart_alignment_seed=args.chart_alignment_seed,
        mast3r_sparse_export_stride=args.mast3r_sparse_export_stride,
        rgb_loss_type=args.rgb_loss_type,
        rgb_supervision_profile=args.rgb_supervision_profile,
        rgb_sampling_policy=args.rgb_sampling_policy,
        chart_geometry_sampling_policy=args.chart_geometry_sampling_policy,
        densification_view_policy=args.densification_view_policy,
        chart_geometry_prior_weight=args.chart_geometry_prior_weight,
        use_color_correction=args.color_correction,
        color_correction_lr=args.color_correction_lr,
        color_correction_reg=args.color_correction_reg,
        white_background=args.white_background,
        semantic_alpha_weight=args.semantic_alpha_weight,
        dense_final_only=args.dense_final_only,
        joint_chart_count=args.joint_chart_count,
        screen_free_gaussians_config=args.screen_free_gaussians_config,
        final_iterations=args.final_iterations,
        final_non_position_lr_decay_from=args.final_non_position_lr_decay_from,
        final_non_position_lr_final_mult=args.final_non_position_lr_final_mult,
        min_global_plane_views=args.min_global_plane_views,
        max_chart_semantic_invalid_ratio=args.max_chart_semantic_invalid_ratio,
        min_chart_static_support_ratio=args.min_chart_static_support_ratio,
        chart_tree_candidate_policy=args.chart_tree_candidate_policy,
        chart_quality_candidate_policy=args.chart_quality_candidate_policy,
        chart_candidate_reliability_weight=args.chart_candidate_reliability_weight,
        min_chart_views_per_sequence=args.min_chart_views_per_sequence,
        post_gate_min_chart_views_per_sequence=args.post_gate_min_chart_views_per_sequence,
        chart_sequence_gate_failure_budget=args.chart_sequence_gate_failure_budget,
        chart_sequence_coverage_mode=args.chart_sequence_coverage_mode,
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
        require_post_gate_coverage=args.require_post_gate_coverage,
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
        selection_artifact = (
            args.selection_json.expanduser().resolve()
            if args.selection_json is not None
            else paths.selection_json
        )
        if args.phase in {"select", "prepare"}:
            continue

        if args.phase == "quality-select":
            if config.joint_chart_count is None:
                raise ValueError("--phase quality-select requires --joint-chart-count")
            require_valid_post_gate_coverage(
                paths,
                config,
                output=paths.screen_output,
                selection_path=selection_artifact,
                env=env,
                dry_run=args.dry_run,
            )
            selection_path = prepare_joint_chart_selection(
                paths,
                config,
                env,
                output=paths.screen_output,
                dry_run=args.dry_run,
            )
            print(f"[DONE] Joint Chart selection: {selection_path}")
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
        gate_report = output / "mast3r_sfm" / "aligned_chart_conflict_gate.json"
        if config.require_post_gate_coverage and not frontend_only:
            if not gate_report.exists() and not args.dry_run:
                # Do not allow a direct screen/full invocation to move from
                # MASt3R alignment into planes before the *accepted* charts
                # have passed coverage.  Bootstrap exactly to the hard gate,
                # then resume the same output after the audit succeeds.
                bootstrap_command = train_command(
                    paths,
                    config,
                    selection,
                    screen_only=screen_only,
                    continue_after_initial=continue_after_initial,
                    continue_after_sfm=continue_after_sfm,
                    continue_after_alignment=continue_after_alignment,
                    continue_after_plane=False,
                    continue_after_see3d_plane_stage=None,
                    stop_after_alignment_gate=True,
                )
                print("[INFO] Bootstrapping frontend to the hard Chart gate before training.", flush=True)
                run_logged(
                    bootstrap_command,
                    cwd=REPO_ROOT,
                    env=env,
                    log_path=output / "logs" / "frontend_gate.log",
                )
                continue_after_initial = False
                continue_after_sfm = False
                continue_after_plane = False
                continue_after_alignment = True
            require_valid_post_gate_coverage(
                paths,
                config,
                output=output,
                selection_path=selection_artifact,
                env=env,
                dry_run=args.dry_run,
            )
        if config.joint_chart_count is not None and args.active_chart_selection_json is not None:
            raise ValueError(
                "Use either --joint-chart-count or --active-chart-selection-json, not both. "
                "The former is the provenance-checked automatic path."
            )
        active_chart_selection = args.active_chart_selection_json
        if config.joint_chart_count is not None and not frontend_only:
            if not (continue_after_initial or continue_after_alignment or continue_after_plane):
                raise RuntimeError(
                    "--joint-chart-count requires a completed gated frontend. Run --phase "
                    "frontend, then --phase quality-select, before screen/full."
                )
            if continue_after_plane:
                raise RuntimeError(
                    "Refusing to change the active Chart subset after plane refinement. "
                    "Use a new --run-tag/output so the geometry provenance remains valid."
                )
            active_chart_selection = prepare_joint_chart_selection(
                paths,
                config,
                env,
                output=output,
                dry_run=args.dry_run,
            )
        if active_chart_selection is not None:
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
                active_chart_selection,
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
                active_chart_selection=active_chart_selection,
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
            # A frontend artifact is not usable merely because MASt3R wrote a
            # gate report.  Validate the *accepted* Charts immediately, so a
            # later screen/full invocation cannot mistake a failed temporal
            # corridor for a completed frontend.
            require_valid_post_gate_coverage(
                paths,
                config,
                output=output,
                selection_path=selection_artifact,
                env=env,
                dry_run=args.dry_run,
            )
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
