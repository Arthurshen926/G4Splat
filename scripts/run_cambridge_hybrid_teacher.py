#!/usr/bin/env python
"""Restartable Cambridge fixed-camera MASt3R→native mixed Teacher mainline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SURFEL_ROOT = REPO_ROOT / "2d-gaussian-splatting"
if str(SURFEL_ROOT) not in sys.path:
    sys.path.insert(0, str(SURFEL_ROOT))

from outdoor.evidence_store import (  # noqa: E402
    EVIDENCE_STORE_VERSION,
    load_evidence_store,
    sha256_file,
)
from outdoor.mast3r_track_graph import validate_track_gate  # noqa: E402
from outdoor.lazy_scene import rgb_source_contract  # noqa: E402
from outdoor.role_aware_initialization import (  # noqa: E402
    INITIALIZATION_VERSION,
)
from scripts.build_crossview_chart_consensus import (  # noqa: E402
    CONSENSUS_VERSION,
)


PIPELINE_VERSION = (
    "cambridge-native-hybrid-teacher-mainline-v20-"
    "rigid-first-dav2-cross-ray-posterior-exact-rgb-contract"
)
STAGES = (
    "prepare_cameras",
    "build_mast3r_tracks",
    "build_charts",
    "build_evidence",
    "initialize_teacher",
    "train_teacher",
    "evaluate_teacher",
    "export_geometry",
)
PROFILES = {
    "quality": {
        # The final mixed stage consumes a geometry-trained native 2DGS
        # handoff.  From-seed mixed training asked canopy ownership and rigid
        # coverage to emerge simultaneously and repeatedly wrote foliage
        # residuals into the facade scaffold.
        "iterations": 30_000,
        "training_profile": "hybrid_handoff_quality",
        # The no-COLMAP v74 scaffold reached 400k at 13.2k and then retained
        # 25k--35k eligible coverage candidates per event until 20.4k.  The
        # old cap therefore froze a demonstrably under-covered rigid map and
        # the mixed stage faithfully inherited that blur.  The larger bound
        # is still demand driven: the trainer admits only primitives whose
        # accumulated screen-space gradient exceeds the native threshold.
        "rigid_pretrain_iterations": 32_000,
        "rigid_pretrain_surface_gaussians": 800_000,
        "rigid_pretrain_surface_growth_per_event": 5000,
        "rigid_pretrain_densify_until_iteration": 16_000,
        "rigid_geometry_gradient_ratio": 0.15,
        "geometry_gradient_ratio": 0.15,
        "checkpoint_every": 1000,
        "track_stride": 5,
        # Keep the complete source-resolution Chart/pointmap rasters as
        # external factors, but do not turn every sample into a renderer
        # primitive.  The older 240k/180k birth family occupied most of the
        # surface budget before topology had measured a screen-bandwidth
        # deficit and regressed the rigid scaffold.
        "maximum_chart_seeds": 180_000,
        # Preserve the controlled v86 Chart bandwidth.  The first DAV2 pilot
        # accidentally reduced this to 5k and removed about 6k supported
        # structural births, confounding the coverage comparison.
        "chart_seeds_per_view": 6000,
        "maximum_mast3r_pointmap_seeds": 240_000,
        "mast3r_pointmap_seeds_per_view": 7000,
        "mast3r_maximum_single_sequence_seed_fraction": 0.20,
        # MASt3R/MAtCha remain the metric authority.  DAV2 is calibrated to
        # that scaffold per fixed camera and may only propose low-opacity
        # births in measured rigid coverage holes.  Cross-traversal distance
        # is a continuous posterior, not a binary PSNR gate.
        "maximum_dav2_rigid_seeds": 80_000,
        "dav2_rigid_selected_views": 512,
        "dav2_rigid_seeds_per_view": 384,
        "dav2_rigid_cross_sequence_radius": 0.25,
        # Keep camera-manifold coverage independent from observation-space
        # bandwidth.  A fixed 96 x 16k allocation made a few exact keyframes
        # look strong while leaving most interpolation views unsupported.
        "selected_foliage_views": 256,
        "maximum_dense_rays_per_foliage_view": 8_192,
        "maximum_dense_rays_total": 1_572_864,
        "minimum_dense_rays_per_foliage_view": 4_096,
        # The historical 18.756 dB 2DGS contains 350,998 surfels.  Capping a
        # from-scratch rigid branch below that known-good capacity made the
        # unified method an implicit low-capacity ablation.
        "maximum_surface_gaussians": 800_000,
        "maximum_surface_growth_per_event": 5000,
        # The evidence seed already contains about 1.04M volumes.  The former
        # v82 saturated 1.5M with ~1.5M unresolved screen-demand rows and
        # only 12k net growth slots/event. Use the measured safe 2M budget,
        # retain role-matched retirement at capacity, and keep topology active
        # through the first half of ownership cleanup.
        "maximum_volume_gaussians": 2_000_000,
        "maximum_volume_splits": 20_000,
        "volume_densify_until_fraction": 0.85,
        "mature_handoff_surface_policy": "appearance_only",
        "surface_retirement_optical_mass_fraction_per_event": 0.0025,
        "maximum_rigid_completion_seeds": 0,
        # Owner rows are sparse, but accelerating opacity alone makes an
        # opaque low-frequency layer before colour/topology can catch up.
        # The measured v80 500->1k collapse falsified that shortcut.
        "volume_opacity_lr": 4.0e-3,
    },
    "fast": {
        "iterations": 12_000,
        "training_profile": "hybrid_handoff_quality",
        "rigid_pretrain_iterations": 12_000,
        "rigid_pretrain_surface_gaussians": 600_000,
        "rigid_pretrain_surface_growth_per_event": 5000,
        "rigid_pretrain_densify_until_iteration": 8_000,
        "rigid_geometry_gradient_ratio": 0.15,
        "geometry_gradient_ratio": 0.15,
        "checkpoint_every": 500,
        "track_stride": 7,
        "maximum_chart_seeds": 80_000,
        "chart_seeds_per_view": 3000,
        "maximum_mast3r_pointmap_seeds": 80_000,
        "mast3r_pointmap_seeds_per_view": 3500,
        "mast3r_maximum_single_sequence_seed_fraction": 0.25,
        "maximum_dav2_rigid_seeds": 20_000,
        "dav2_rigid_selected_views": 128,
        "dav2_rigid_seeds_per_view": 160,
        "dav2_rigid_cross_sequence_radius": 0.25,
        "selected_foliage_views": 128,
        "maximum_dense_rays_per_foliage_view": 8_192,
        "maximum_dense_rays_total": 786_432,
        "minimum_dense_rays_per_foliage_view": 4_096,
        "maximum_surface_gaussians": 600_000,
        "maximum_surface_growth_per_event": 5000,
        "maximum_volume_gaussians": 2_000_000,
        "maximum_volume_splits": 20_000,
        "volume_densify_until_fraction": 5.0 / 6.0,
        "mature_handoff_surface_policy": "appearance_only",
        "surface_retirement_optical_mass_fraction_per_event": 0.0025,
        "maximum_rigid_completion_seeds": 0,
        "volume_opacity_lr": 4.0e-3,
    },
    "rigid": {
        "iterations": 24_000,
        "training_profile": "hybrid_rigid_stage1",
        "checkpoint_every": 500,
        "track_stride": 5,
        "maximum_chart_seeds": 100_000,
        "chart_seeds_per_view": 3500,
        "maximum_mast3r_pointmap_seeds": 120_000,
        "mast3r_pointmap_seeds_per_view": 5000,
        "mast3r_maximum_single_sequence_seed_fraction": 0.25,
        "maximum_dav2_rigid_seeds": 0,
        "dav2_rigid_selected_views": 0,
        "dav2_rigid_seeds_per_view": 128,
        "dav2_rigid_cross_sequence_radius": 0.25,
        "selected_foliage_views": 72,
        "maximum_dense_rays_per_foliage_view": 8_192,
        "maximum_dense_rays_total": 294_912,
        "minimum_dense_rays_per_foliage_view": 2_048,
        "maximum_surface_gaussians": 400_000,
        "maximum_surface_growth_per_event": 2500,
        "maximum_volume_gaussians": 800_000,
        "maximum_volume_splits": 1500,
        "geometry_gradient_ratio": 0.25,
    },
}


def _trainer_implementation_hashes() -> dict[str, str]:
    surfel_root = REPO_ROOT / "2d-gaussian-splatting"
    paths = {
        "trainer": REPO_ROOT
        / "scripts/train_unified_outdoor_teacher.py",
        "appearance_uncertainty": REPO_ROOT
        / "outdoor/appearance_uncertainty.py",
        "chart_surface_model": REPO_ROOT
        / "outdoor/chart_surface_model.py",
        "lazy_scene": REPO_ROOT / "outdoor/lazy_scene.py",
        "intrinsics_utils": surfel_root / "utils/intrinsics_utils.py",
        "dataset_reader": surfel_root / "scene/dataset_readers.py",
        "gaussian_model": surfel_root / "scene/gaussian_model.py",
        "task_fields": REPO_ROOT / "outdoor/task_fields.py",
        "mask_lookup": REPO_ROOT / "matcha/cambridge_masks.py",
        "training_evidence": REPO_ROOT
        / "outdoor/training_evidence.py",
        "hybrid_renderer": REPO_ROOT
        / "outdoor/hybrid_gaussian_renderer.py",
        "mixed_forward_cuda": surfel_root
        / "submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu",
        "mixed_backward_cuda": surfel_root
        / (
            "submodules/diff-surfel-rasterization/"
            "cuda_rasterizer/mixed_backward.cu"
        ),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _teacher_result_is_current(
    result_path: Path,
    *,
    evidence_hash: str,
    initialization: Path,
    iterations: int,
    training_profile: str,
    maximum_surface_gaussians: int,
    maximum_surface_growth_per_event: int,
    maximum_volume_gaussians: int,
    maximum_volume_splits: int,
    geometry_gradient_ratio: float,
    volume_densify_until_iteration: int | None = None,
    surface_densify_until_iteration: int | None = None,
    surface_warmstart_ply: Path | None = None,
    surface_warmstart_manifest: Path | None = None,
    mature_handoff_surface_policy: str = "joint",
    maximum_rigid_completion_seeds: int = 20_000,
    volume_opacity_lr: float = 4.0e-3,
    surface_retirement_optical_mass_fraction_per_event: float = 0.0025,
) -> bool:
    """Accept a completed Teacher only when every causal input still matches."""
    if (surface_warmstart_ply is None) != (
        surface_warmstart_manifest is None
    ):
        raise ValueError(
            "Surface warm-start PLY and manifest must be supplied together"
        )
    if not result_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        manifest_path = initialization / "initialization_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = result["training_contract"]
        teacher_state = Path(result["teacher_state"]).resolve()
        expected_surface_ply_sha256 = (
            sha256_file(surface_warmstart_ply)
            if surface_warmstart_ply is not None
            else None
        )
        expected_handoff_sha256 = (
            sha256_file(surface_warmstart_manifest)
            if surface_warmstart_manifest is not None
            else None
        )
        return bool(
            result.get("evidence_hash") == evidence_hash
            and int(result.get("iterations", -1)) == int(iterations)
            and result.get("training_profile") == training_profile
            and int(result.get("maximum_surface_gaussians", -1))
            == int(maximum_surface_gaussians)
            and int(
                result.get(
                    "maximum_surface_growth_per_event", -1
                )
            )
            == int(maximum_surface_growth_per_event)
            and int(result.get("maximum_volume_gaussians", -1))
            == int(maximum_volume_gaussians)
            and int(
                result.get("maximum_volume_splits_per_event", -1)
            )
            == int(maximum_volume_splits)
            and (
                volume_densify_until_iteration is None
                or int(
                    contract.get(
                        "volume_densify_until_iteration", -1
                    )
                )
                == int(volume_densify_until_iteration)
            )
            and (
                surface_densify_until_iteration is None
                or int(contract.get("densify_until_iter", -1))
                == int(surface_densify_until_iteration)
            )
            and int(contract.get("schedule_horizon", -1))
            == int(iterations)
            and np.isclose(
                float(contract.get("geometry_gradient_ratio", np.nan)),
                float(geometry_gradient_ratio),
            )
            and contract.get("mature_handoff_surface_policy")
            == mature_handoff_surface_policy
            and int(
                contract.get("rigid_background_completion", {}).get(
                    "maximum_seeds", -1
                )
            )
            == int(maximum_rigid_completion_seeds)
            and np.isclose(
                float(contract.get("volume_opacity_lr", np.nan)),
                float(volume_opacity_lr),
            )
            and np.isclose(
                float(
                    contract.get(
                        "surface_retirement_optical_mass_fraction_per_event",
                        np.nan,
                    )
                ),
                float(
                    surface_retirement_optical_mass_fraction_per_event
                ),
            )
            and contract.get("initialization_version")
            == manifest.get("version")
            and contract.get("initialization_manifest_sha256")
            == sha256_file(manifest_path)
            and contract.get("surface_seed_sha256")
            == sha256_file(Path(manifest["surface_seed"]))
            and contract.get("foliage_seed_sha256")
            == sha256_file(Path(manifest["foliage_seed"]))
            and contract.get("surface_warmstart_ply_sha256")
            == expected_surface_ply_sha256
            and contract.get("surface_warmstart_manifest_sha256")
            == expected_handoff_sha256
            and result.get("implementation_hashes")
            == _trainer_implementation_hashes()
            and teacher_state.is_file()
            and result.get("teacher_state_sha256")
            == sha256_file(teacher_state)
        )
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


def _evaluation_is_current(
    metrics_path: Path,
    *,
    teacher_state: Path,
    evaluation_mode: str,
    expected_view_count: int,
    expected_rgb_source: dict | None = None,
    expected_camera_geometry_sha256: str | None = None,
    expected_database_contract: Path | None = None,
    expected_query_contract: Path | None = None,
) -> bool:
    if not metrics_path.is_file() or not teacher_state.is_file():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metric_protocol = metrics.get("metric_protocol", {})
        recorded_rgb_source = metric_protocol.get("rgb_source", {})
        rgb_identity_fields = (
            "image_root",
            "image_count",
            "name_set_sha256",
            "content_mapping_sha256",
            "content_bytes",
            "producer_manifest_sha256",
            "target_storage",
            "canonical_image_size_wh",
        )
        rgb_current = (
            expected_rgb_source is None
            or all(
                recorded_rgb_source.get(key)
                == expected_rgb_source.get(key)
                for key in rgb_identity_fields
            )
        )
        camera_current = (
            expected_camera_geometry_sha256 is None
            or metric_protocol.get("camera_geometry_sha256")
            == expected_camera_geometry_sha256
        )
        query_validation = metrics.get("query_contract_validation", {})
        database_contract_current = (
            expected_database_contract is None
            or query_validation.get("database_contract_sha256")
            == sha256_file(expected_database_contract)
        )
        query_contract_current = (
            expected_query_contract is None
            or query_validation.get("query_contract_sha256")
            == sha256_file(expected_query_contract)
        )
        return bool(
            Path(metrics.get("teacher_state", "")).resolve()
            == teacher_state.resolve()
            and metrics.get("teacher_state_sha256")
            == sha256_file(teacher_state)
            and metrics.get("evaluation_implementation_sha256")
            == sha256_file(
                REPO_ROOT / "scripts/evaluate_hybrid_teacher.py"
            )
            and metrics.get("evaluation_mode") == evaluation_mode
            and int(metrics.get("view_count", -1))
            == int(expected_view_count)
            and rgb_current
            and camera_current
            and database_contract_current
            and query_contract_current
            and metrics.get("student_or_standard_renderer_used") is False
        )
    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="StMarysChurch")
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument(
        "--camera-source",
        choices=("cambridge_fixed",),
        default="cambridge_fixed",
    )
    parser.add_argument(
        "--geometry-source",
        choices=("mast3r_only",),
        default="mast3r_only",
    )
    parser.add_argument(
        "--final-model",
        choices=("hybrid_teacher",),
        default="hybrid_teacher",
    )
    parser.add_argument(
        "--frontend-root",
        type=Path,
        default=Path(
            "/mnt/pool/sqy/G4Splat_runs/cambridge_outdoor_mainline_v1"
        ),
    )
    parser.add_argument("--frontend-run", type=Path)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "/mnt/pool/sqy/G4Splat_runs/cambridge_hybrid_teacher_v1"
        ),
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=Path("/mnt/pool/sqy/Cambridge_stdloc"),
    )
    parser.add_argument("--dav2-root", type=Path)
    parser.add_argument(
        "--rgb-images",
        type=Path,
        help=(
            "Optional pre-materialized RGB target directory. Use the shared "
            "640x360 torch-bilinear Cambridge targets for pixel-identical "
            "comparison with the historical 18.756 protocol."
        ),
    )
    parser.add_argument(
        "--query-dataset",
        type=Path,
        help=(
            "Optional disjoint Cambridge localization-query adapter. For "
            "StMarysChurch the verified official query64 adapter is "
            "auto-discovered when this option is omitted."
        ),
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="quality")
    parser.add_argument("--iterations", type=int)
    parser.add_argument(
        "--rigid-iterations",
        type=int,
        help=(
            "Override the geometry-first native 2DGS pretraining horizon for "
            "staged quality/fast profiles. The rigid-only profile continues "
            "to use --iterations."
        ),
    )
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--minimum-free-gpu-memory-mib", type=int, default=20_000)
    parser.add_argument("--maximum-surface-gaussians", type=int)
    parser.add_argument(
        "--maximum-surface-growth-per-event", type=int
    )
    parser.add_argument("--maximum-volume-gaussians", type=int)
    parser.add_argument("--maximum-volume-splits", type=int)
    parser.add_argument(
        "--volume-densify-until-iteration",
        type=int,
        help=(
            "Last mixed-stage iteration allowed to adapt 3D volume topology. "
            "Defaults to the selected profile fraction of --iterations."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-trainer-repair-resume", action="store_true"
    )
    args = parser.parse_args()
    if args.iterations is None:
        args.iterations = PROFILES[args.profile]["iterations"]
    if args.rigid_iterations is None:
        args.rigid_iterations = PROFILES[args.profile].get(
            "rigid_pretrain_iterations"
        )
    if (
        args.rigid_iterations is not None
        and int(args.rigid_iterations) <= 0
    ):
        parser.error("--rigid-iterations must be positive")
    if args.maximum_surface_gaussians is None:
        args.maximum_surface_gaussians = PROFILES[args.profile][
            "maximum_surface_gaussians"
        ]
    if args.maximum_surface_growth_per_event is None:
        args.maximum_surface_growth_per_event = PROFILES[args.profile][
            "maximum_surface_growth_per_event"
        ]
    if args.maximum_volume_gaussians is None:
        args.maximum_volume_gaussians = PROFILES[args.profile][
            "maximum_volume_gaussians"
        ]
    if args.maximum_volume_splits is None:
        args.maximum_volume_splits = PROFILES[args.profile][
            "maximum_volume_splits"
        ]
    if args.volume_densify_until_iteration is None:
        fraction = PROFILES[args.profile].get(
            "volume_densify_until_fraction"
        )
        if fraction is not None:
            args.volume_densify_until_iteration = int(
                round(float(args.iterations) * float(fraction))
            )
    if (
        args.volume_densify_until_iteration is not None
        and (
            int(args.volume_densify_until_iteration) <= 0
            or int(args.volume_densify_until_iteration)
            > int(args.iterations)
        )
    ):
        parser.error(
            "--volume-densify-until-iteration must be in [1, --iterations]"
        )
    return args


def _gpu(requested: str, minimum_free: int) -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = {}
    for line in result.stdout.splitlines():
        index, free, total = [value.strip() for value in line.split(",")]
        rows[index] = (int(free), int(total))
    if requested not in rows:
        raise RuntimeError(f"Physical GPU {requested} is unavailable")
    free, total = rows[requested]
    if free < int(minimum_free):
        raise RuntimeError(
            f"Physical GPU {requested} has {free}/{total} MiB free, "
            f"below the {minimum_free} MiB launch gate"
        )
    return requested


def _run(command: list[str], *, env: dict, log: Path, dry_run: bool) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(" ".join(command))
        return
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace")[-16000:]
        raise RuntimeError(
            f"Stage failed with exit {completed.returncode}: {' '.join(command)}\n"
            + tail
        )


def _teacher_command(
    *,
    python: str,
    dataset: Path,
    teacher: Path,
    evidence: Path,
    initialization: Path,
    iterations: int,
    training_profile: str,
    maximum_surface_gaussians: int,
    maximum_surface_growth_per_event: int,
    maximum_volume_gaussians: int,
    maximum_volume_splits: int,
    checkpoint_every: int,
    geometry_gradient_ratio: float,
    rgb_images: Path | None,
    volume_densify_until_iteration: int | None = None,
    surface_densify_until_iteration: int | None = None,
    surface_warmstart_ply: Path | None = None,
    surface_warmstart_manifest: Path | None = None,
    mature_handoff_surface_policy: str = "joint",
    maximum_rigid_completion_seeds: int = 20_000,
    volume_opacity_lr: float | None = None,
    surface_retirement_optical_mass_fraction_per_event: float = 0.0025,
) -> list[str]:
    if (surface_warmstart_ply is None) != (
        surface_warmstart_manifest is None
    ):
        raise ValueError(
            "Surface warm-start PLY and manifest must be supplied together"
        )
    command = [
        python,
        str(REPO_ROOT / "scripts/train_unified_outdoor_teacher.py"),
        "-s",
        str(dataset),
        "-m",
        str(teacher),
        "--evidence-store",
        str(evidence),
        "--initialization",
        str(initialization),
        "--resolution",
        "640",
        "--iterations",
        str(int(iterations)),
        "--phase-schedule-horizon",
        str(int(iterations)),
        "--training-profile",
        str(training_profile),
        "--geometry-gradient-ratio",
        str(float(geometry_gradient_ratio)),
        "--maximum-surface-gaussians",
        str(int(maximum_surface_gaussians)),
        "--maximum-surface-growth-per-event",
        str(int(maximum_surface_growth_per_event)),
        # Keep world-space coverage and projected-bandwidth controls
        # independent.  The former 0.12 m surface ceiling was inherited from
        # the foliage voxel size and forced distant facades into clone-heavy
        # under-coverage.
        "--maximum-surface-scale",
        "0.5",
        "--maximum-surface-radius-pixels",
        "24",
        "--maximum-volume-gaussians",
        str(int(maximum_volume_gaussians)),
        "--maximum-volume-splits",
        str(int(maximum_volume_splits)),
        "--volume-split-radius",
        "2.0",
        "--maximum-volume-radius-pixels",
        "24",
        "--checkpoint-every",
        str(int(checkpoint_every)),
        "--mature-handoff-surface-policy",
        str(mature_handoff_surface_policy),
        "--maximum-rigid-completion-seeds",
        str(int(maximum_rigid_completion_seeds)),
        "--surface-retirement-optical-mass-fraction-per-event",
        str(
            float(
                surface_retirement_optical_mass_fraction_per_event
            )
        ),
    ]
    if volume_densify_until_iteration is not None:
        command.extend(
            [
                "--volume-densify-until-iteration",
                str(int(volume_densify_until_iteration)),
            ]
        )
    if surface_densify_until_iteration is not None:
        command.extend(
            [
                "--densify_until_iter",
                str(int(surface_densify_until_iteration)),
            ]
        )
    if volume_opacity_lr is not None:
        command.extend(
            ["--volume-opacity-lr", str(float(volume_opacity_lr))]
        )
    if rgb_images is not None:
        command.extend(["--images", str(rgb_images)])
    if training_profile == "hybrid_rigid_stage1":
        command.extend(
            [
                "--position_lr_init",
                "1.6e-5",
                "--position_lr_final",
                "1.6e-6",
                "--position_lr_delay_mult",
                "0.01",
                "--position_lr_max_steps",
                str(min(max(int(iterations), 20_000), 30_000)),
            ]
        )
    if surface_warmstart_ply is not None:
        command.extend(
            [
                "--surface-warmstart-ply",
                str(surface_warmstart_ply),
                "--surface-warmstart-manifest",
                str(surface_warmstart_manifest),
            ]
        )
    return command


def _find_frontend(args: argparse.Namespace) -> Path:
    if args.frontend_run is not None:
        return args.frontend_run.expanduser().resolve()
    candidates = []
    for manifest in args.frontend_root.rglob("outdoor_mainline_manifest.json"):
        try:
            payload = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            Path(payload.get("dense_real_dataset", "")).name
            and args.scene in str(manifest)
            and (manifest.parent / "mast3r_sfm/charts_data.npz").is_file()
            and (manifest.parent / "mast3r_sfm/cameras.json").is_file()
            and (manifest.parent / "mast3r_sfm/pointmaps").is_dir()
        ):
            candidates.append(manifest.parent)
    if not candidates:
        raise FileNotFoundError("No completed MAtCha/G4Splat frontend was found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _write_manifest(path: Path, payload: dict) -> None:
    payload["updated_at_unix"] = time.time()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = _args()
    profile = PROFILES[args.profile]
    run = (
        args.run_root.expanduser().resolve()
        / f"{args.scene}_hybrid_teacher_{args.profile}_v20"
    )
    run.mkdir(parents=True, exist_ok=True)
    manifest_path = run / "pipeline_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.is_file()
        else {
            "version": PIPELINE_VERSION,
            "scene": args.scene,
            "camera_source": args.camera_source,
            "geometry_source": args.geometry_source,
            "final_model": args.final_model,
            "student_enabled": False,
            "stages": {},
        }
    )
    selected = STAGES if args.stage == "all" else (args.stage,)
    python = sys.executable
    env = dict(os.environ)
    conda_lib = str(Path(sys.executable).resolve().parent.parent / "lib")
    library_entries = [
        value
        for value in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if value and value != conda_lib
    ]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [conda_lib, *library_entries]
    )

    frontend_ref = run / "frontend.json"
    if "prepare_cameras" in selected:
        frontend = _find_frontend(args)
        frontend_payload = json.loads(
            (frontend / "outdoor_mainline_manifest.json").read_text()
        )
        dataset = Path(frontend_payload["dense_real_dataset"]).resolve()
        required = [
            dataset / "sparse/0/cameras.bin",
            dataset / "sparse/0/images.bin",
            dataset / "images",
            dataset / "name_mapping.json",
            frontend / "mast3r_sfm/cameras.json",
            frontend / "mast3r_sfm/charts_data.npz",
            frontend / "mast3r_sfm/pointmaps",
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Camera/chart preparation is incomplete: "
                + ", ".join(map(str, missing))
            )
        frontend_ref.write_text(
            json.dumps(
                {
                    "frontend": str(frontend),
                    "dataset": str(dataset),
                    "points3D_required": False,
                    "points3D_read": False,
                },
                indent=2,
            )
            + "\n"
        )
        manifest["stages"]["prepare_cameras"] = {
            "status": "complete",
            "dataset": str(dataset),
            "frontend": str(frontend),
        }
        _write_manifest(manifest_path, manifest)
    if not frontend_ref.is_file():
        raise FileNotFoundError(
            f"Run prepare_cameras first; missing {frontend_ref}"
        )
    frontend_payload = json.loads(frontend_ref.read_text())
    frontend = Path(frontend_payload["frontend"])
    dataset = Path(frontend_payload["dataset"])
    mast3r = frontend / "mast3r_sfm"
    tree_mask = args.mask_root / args.scene / "processed/masks_with_tree.pkl"
    base_mask = args.mask_root / args.scene / "processed/masks.pkl"

    chart_selection = (
        mast3r
        / "quality_aware_selection"
        / "structural_chart_selection.json"
    )
    if not chart_selection.is_file():
        raise FileNotFoundError(
            "The final Teacher requires the immutable structural Chart "
            f"selection: {chart_selection}"
        )
    chart_consensus = (
        run / "geometry/chart_crossview_consensus_v2.npz"
    )
    consensus_summary = chart_consensus.with_suffix(".json")
    rebuild_consensus = not (
        chart_consensus.is_file() and consensus_summary.is_file()
    )
    if not rebuild_consensus:
        try:
            with np.load(chart_consensus, allow_pickle=False) as archive:
                schema = str(archive["schema_version"].item())
            summary = json.loads(consensus_summary.read_text())
            expected_hashes = {
                "charts_data": sha256_file(mast3r / "charts_data.npz"),
                "chart_cameras": sha256_file(mast3r / "cameras.json"),
                "gate_report": sha256_file(
                    mast3r / "aligned_chart_conflict_gate.json"
                ),
                "chart_selection": sha256_file(chart_selection),
                "mask_pickle": sha256_file(tree_mask),
            }
            rebuild_consensus = (
                schema != CONSENSUS_VERSION
                or summary.get("input_sha256") != expected_hashes
            )
        except (KeyError, OSError, RuntimeError, ValueError):
            rebuild_consensus = True
    if (
        rebuild_consensus
        and (
            "build_mast3r_tracks" in selected
            or "build_charts" in selected
            or "build_evidence" in selected
        )
    ):
        _run(
            [
                python,
                str(
                    REPO_ROOT
                    / "scripts/build_crossview_chart_consensus.py"
                ),
                "--scene-path",
                str(mast3r),
                "--gate-report",
                str(mast3r / "aligned_chart_conflict_gate.json"),
                "--chart-selection-json",
                str(chart_selection),
                "--output",
                str(chart_consensus),
                "--mask-pickle",
                str(tree_mask),
                "--mask-dataset-path",
                str(dataset),
                "--mask-indices",
                "0",
                "1",
                "2",
                "--neighbors",
                "5",
                "--neighbor-selection",
                "overlap",
                "--min-neighbor-overlap",
                "0.03",
                "--min-consensus-views",
                "2",
                "--max-relative-error",
                "0.20",
                "--weight-floor",
                "0.10",
                "--relative-error-sigma",
                "0.15",
                "--rejected-supported-weight",
                "0.05",
                "--diagnostic-dir",
                str(run / "geometry/chart_consensus_diagnostics"),
            ],
            env=env,
            log=run / "logs/build_chart_consensus.log",
            dry_run=args.dry_run,
        )

    tracks = run / "tracks/mast3r_multiview_tracks.npz"
    if "build_mast3r_tracks" in selected:
        rebuild_tracks = not tracks.is_file()
        if not rebuild_tracks:
            try:
                validate_track_gate(tracks)
                track_summary = json.loads(
                    tracks.with_suffix(".json").read_text()
                )
                if track_summary.get(
                    "chart_consensus_sha256"
                ) != sha256_file(chart_consensus):
                    rebuild_tracks = True
            except (RuntimeError, OSError, KeyError, ValueError) as error:
                rebuild_tracks = True
                print(f"Rebuilding stale/invalid MASt3R tracks: {error}")
        if rebuild_tracks:
            command = [
                python,
                str(REPO_ROOT / "scripts/build_mast3r_multiview_tracks.py"),
                "--mast3r-scene",
                str(mast3r),
                "--dataset",
                str(dataset),
                "--tree-mask-pickle",
                str(tree_mask),
                "--chart-consensus",
                str(chart_consensus),
                "--output",
                str(tracks),
                "--stride",
                str(profile["track_stride"]),
            ]
            if tracks.exists():
                command.append("--replace")
            _run(
                command,
                env=env,
                log=run / "logs/build_mast3r_tracks.log",
                dry_run=args.dry_run,
            )
        if args.dry_run:
            return
        gate = validate_track_gate(tracks)
        manifest["stages"]["build_mast3r_tracks"] = {
            "status": "complete",
            "archive": str(tracks),
            "gate": gate,
        }
        _write_manifest(manifest_path, manifest)

    if "build_charts" in selected:
        required = [
            mast3r / "charts_data.npz",
            mast3r / "cameras.json",
            mast3r / "aligned_chart_conflict_gate.json",
        ]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "MAtCha chart atlas is incomplete: "
                + ", ".join(map(str, missing))
            )
        manifest["stages"]["build_charts"] = {
            "status": "complete",
            "atlas": str(required[0]),
            "crossview_consensus": str(chart_consensus),
            "native_resolution_observation_factor": True,
            "learnable_continuous_chart_atlas": False,
        }
        _write_manifest(manifest_path, manifest)

    evidence = run / "evidence"
    if "build_evidence" in selected:
        dav2_root = (
            args.dav2_root.expanduser().resolve()
            if args.dav2_root is not None
            else None
        )
        packed_candidates = (
            dataset / "depth_anything_vitl_fp16.pt",
            dataset.parent / "depth_anything_vitl_fp16.pt",
        )
        packed_dav2 = next(
            (path for path in packed_candidates if path.is_file()),
            packed_candidates[0],
        )
        if dav2_root is None and packed_dav2.is_file():
            dav2_root = run / "dav2_real_views"
            dav2_manifest = dav2_root / "dav2_cache_manifest.json"
            if not dav2_manifest.is_file():
                _run(
                    [
                        python,
                        str(
                            REPO_ROOT
                            / "scripts/unpack_dav2_real_view_cache.py"
                        ),
                        "--packed-cache",
                        str(packed_dav2),
                        "--output",
                        str(dav2_root),
                    ],
                    env=env,
                    log=run / "logs/unpack_dav2.log",
                    dry_run=args.dry_run,
                )
        rebuild_evidence = not (
            evidence / "evidence_manifest.json"
        ).is_file()
        if not rebuild_evidence:
            try:
                current_store = load_evidence_store(evidence)
                if (
                    current_store.get("schema_version")
                    != EVIDENCE_STORE_VERSION
                ):
                    rebuild_evidence = True
                artifact_names = {
                    item["name"]
                    for item in current_store.get("artifacts", [])
                }
                track_artifact = next(
                    (
                        item
                        for item in current_store.get("artifacts", [])
                        if item["name"] == "mast3r_multiview_tracks"
                    ),
                    None,
                )
                if (
                    dav2_root is not None
                    and "dav2_index" not in artifact_names
                ):
                    rebuild_evidence = True
                if (
                    track_artifact is None
                    or track_artifact.get("sha256") != sha256_file(tracks)
                ):
                    rebuild_evidence = True
                consensus_artifact = next(
                    (
                        item
                        for item in current_store.get("artifacts", [])
                        if item["name"] == "chart_crossview_consensus"
                    ),
                    None,
                )
                if (
                    consensus_artifact is None
                    or consensus_artifact.get("sha256")
                    != sha256_file(chart_consensus)
                ):
                    rebuild_evidence = True
            except (FileNotFoundError, RuntimeError):
                rebuild_evidence = True
        if rebuild_evidence:
            command = [
                python,
                str(REPO_ROOT / "scripts/build_hybrid_teacher_evidence.py"),
                "--dataset",
                str(dataset),
                "--mask-pickle",
                str(base_mask),
                "--tree-mask-pickle",
                str(tree_mask),
                "--mast3r-scene",
                str(mast3r),
                "--mast3r-tracks",
                str(tracks),
                "--chart-consensus",
                str(chart_consensus),
                "--output",
                str(evidence),
            ]
            if evidence.exists():
                command.append("--replace")
            if dav2_root is not None:
                command.extend(["--dav2-root", str(dav2_root)])
            _run(
                command,
                env=env,
                log=run / "logs/build_evidence.log",
                dry_run=args.dry_run,
            )
        if args.dry_run:
            return
        store = load_evidence_store(evidence)
        manifest["stages"]["build_evidence"] = {
            "status": "complete",
            "evidence_hash": store["evidence_hash"],
            "colmap_tracks": False,
        }
        _write_manifest(manifest_path, manifest)

    initialization = run / "initialization"
    if "initialize_teacher" in selected:
        store_hash = load_evidence_store(evidence)["evidence_hash"]
        initialization_contract = {
            "rgb_root": str(
                (
                    args.rgb_images.expanduser().resolve()
                    if args.rgb_images is not None
                    else (dataset / "images").resolve()
                )
            ),
            "maximum_chart_seeds": int(
                profile["maximum_chart_seeds"]
            ),
            "chart_seeds_per_view": int(
                profile["chart_seeds_per_view"]
            ),
            "mast3r_pointmap_seeds_per_view": int(
                profile["mast3r_pointmap_seeds_per_view"]
            ),
            "maximum_mast3r_pointmap_seeds": int(
                profile["maximum_mast3r_pointmap_seeds"]
            ),
            "mast3r_pointmap_minimum_confidence": 1.25,
            "mast3r_cross_sequence_radius": 0.15,
            "mast3r_pointmap_voxel_size": 0.018,
            "mast3r_maximum_single_sequence_seed_fraction": float(
                profile[
                    "mast3r_maximum_single_sequence_seed_fraction"
                ]
            ),
            "maximum_dav2_rigid_seeds": int(
                profile["maximum_dav2_rigid_seeds"]
            ),
            "dav2_rigid_selected_views": int(
                profile["dav2_rigid_selected_views"]
            ),
            "dav2_rigid_seeds_per_view": int(
                profile["dav2_rigid_seeds_per_view"]
            ),
            "dav2_rigid_cross_sequence_radius": float(
                profile["dav2_rigid_cross_sequence_radius"]
            ),
            "maximum_foliage_voxels": 400_000,
            "selected_foliage_views": int(
                profile["selected_foliage_views"]
            ),
            "maximum_dense_rays_per_foliage_view": int(
                profile["maximum_dense_rays_per_foliage_view"]
            ),
            "maximum_dense_rays_total": int(
                profile["maximum_dense_rays_total"]
            ),
            "minimum_dense_rays_per_foliage_view": int(
                profile["minimum_dense_rays_per_foliage_view"]
            ),
            "voxel_size": 0.12,
            "seed": 73,
        }
        rebuild_initialization = not (
            initialization / "initialization_manifest.json"
        ).is_file()
        if not rebuild_initialization:
            current_initialization = json.loads(
                (
                    initialization / "initialization_manifest.json"
                ).read_text()
            )
            rebuild_initialization = (
                current_initialization.get("evidence_hash") != store_hash
                or current_initialization.get("version")
                != INITIALIZATION_VERSION
                or current_initialization.get("initialization_contract")
                != initialization_contract
            )
        if rebuild_initialization:
            command = [
                    python,
                    str(REPO_ROOT / "scripts/initialize_unified_outdoor_scene.py"),
                    "--evidence-store",
                    str(evidence),
                    "--output",
                    str(initialization),
                    "--rgb-root",
                    str(
                        (
                            args.rgb_images.expanduser().resolve()
                            if args.rgb_images is not None
                            else (dataset / "images").resolve()
                        )
                    ),
                    "--maximum-chart-seeds",
                    str(profile["maximum_chart_seeds"]),
                    "--chart-seeds-per-view",
                    str(profile["chart_seeds_per_view"]),
                    "--mast3r-pointmap-seeds-per-view",
                    str(profile["mast3r_pointmap_seeds_per_view"]),
                    "--maximum-mast3r-pointmap-seeds",
                    str(profile["maximum_mast3r_pointmap_seeds"]),
                    "--mast3r-pointmap-minimum-confidence",
                    "1.25",
                    "--mast3r-cross-sequence-radius",
                    "0.15",
                    "--mast3r-pointmap-voxel-size",
                    "0.018",
                    "--mast3r-maximum-single-sequence-seed-fraction",
                    str(
                        profile[
                            "mast3r_maximum_single_sequence_seed_fraction"
                        ]
                    ),
                    "--maximum-dav2-rigid-seeds",
                    str(profile["maximum_dav2_rigid_seeds"]),
                    "--dav2-rigid-selected-views",
                    str(profile["dav2_rigid_selected_views"]),
                    "--dav2-rigid-seeds-per-view",
                    str(profile["dav2_rigid_seeds_per_view"]),
                    "--dav2-rigid-cross-sequence-radius",
                    str(profile["dav2_rigid_cross_sequence_radius"]),
                    "--selected-foliage-views",
                    str(profile["selected_foliage_views"]),
                    "--maximum-dense-rays-per-foliage-view",
                    str(
                        profile[
                            "maximum_dense_rays_per_foliage_view"
                        ]
                    ),
                    "--maximum-dense-rays-total",
                    str(profile["maximum_dense_rays_total"]),
                    "--minimum-dense-rays-per-foliage-view",
                    str(
                        profile[
                            "minimum_dense_rays_per_foliage_view"
                        ]
                    ),
                    "--maximum-foliage-voxels",
                    "400000",
                    "--voxel-size",
                    "0.12",
                    "--seed",
                    "73",
                ]
            if initialization.exists():
                command.append("--replace")
            _run(
                command,
                env=env,
                log=run / "logs/initialize_teacher.log",
                dry_run=args.dry_run,
            )
        if args.dry_run:
            return
        init = json.loads(
            (initialization / "initialization_manifest.json").read_text()
        )
        if init.get("historical_model_initialization") is not False:
            raise RuntimeError("Historical Gaussian initialization is forbidden")
        if init["surface"].get("colmap_points_or_tracks_used") is not False:
            raise RuntimeError("Initialization consumed COLMAP geometry")
        manifest["stages"]["initialize_teacher"] = {
            "status": "complete",
            "surface": init["surface"],
            "foliage": init["foliage"],
        }
        _write_manifest(manifest_path, manifest)

    teacher = run / "teacher"
    if "train_teacher" in selected:
        physical_gpu = _gpu(args.gpu, args.minimum_free_gpu_memory_mib)
        train_env = dict(env)
        train_env["CUDA_VISIBLE_DEVICES"] = physical_gpu
        store_hash = load_evidence_store(evidence)["evidence_hash"]
        rgb_images = (
            args.rgb_images.expanduser().resolve()
            if args.rgb_images is not None
            else None
        )
        if rgb_images is not None and not rgb_images.is_dir():
            raise FileNotFoundError(rgb_images)

        surface_warmstart_ply = None
        surface_warmstart_manifest = None
        rigid_result = None
        if args.rigid_iterations is not None:
            rigid_teacher = run / "teacher_rigid"
            rigid_result = rigid_teacher / "result.json"
            rigid_surface_gaussians = int(
                profile["rigid_pretrain_surface_gaussians"]
            )
            rigid_surface_growth = int(
                profile["rigid_pretrain_surface_growth_per_event"]
            )
            rigid_result_current = _teacher_result_is_current(
                rigid_result,
                evidence_hash=store_hash,
                initialization=initialization,
                iterations=int(args.rigid_iterations),
                training_profile="hybrid_rigid_stage1",
                maximum_surface_gaussians=rigid_surface_gaussians,
                maximum_surface_growth_per_event=rigid_surface_growth,
                maximum_volume_gaussians=800_000,
                maximum_volume_splits=1500,
                geometry_gradient_ratio=float(
                    profile["rigid_geometry_gradient_ratio"]
                ),
                surface_densify_until_iteration=int(
                    profile["rigid_pretrain_densify_until_iteration"]
                ),
            )
            if rigid_result.is_file() and not rigid_result_current:
                raise RuntimeError(
                    "Existing rigid Teacher is stale relative to the current "
                    "evidence, initialization, implementation, or state "
                    "file. Use a fresh --run-root so a mixed result cannot "
                    "silently inherit the wrong scaffold."
                )
            if not rigid_result_current:
                rigid_command = _teacher_command(
                    python=python,
                    dataset=dataset,
                    teacher=rigid_teacher,
                    evidence=evidence,
                    initialization=initialization,
                    iterations=int(args.rigid_iterations),
                    training_profile="hybrid_rigid_stage1",
                    maximum_surface_gaussians=rigid_surface_gaussians,
                    maximum_surface_growth_per_event=rigid_surface_growth,
                    maximum_volume_gaussians=800_000,
                    maximum_volume_splits=1500,
                    checkpoint_every=int(profile["checkpoint_every"]),
                    geometry_gradient_ratio=float(
                        profile["rigid_geometry_gradient_ratio"]
                    ),
                    surface_densify_until_iteration=int(
                        profile["rigid_pretrain_densify_until_iteration"]
                    ),
                    rgb_images=rgb_images,
                )
                rigid_checkpoint = (
                    rigid_teacher / "hybrid_teacher_checkpoint.pth"
                )
                if rigid_checkpoint.is_file():
                    rigid_command.extend(
                        ["--resume", str(rigid_checkpoint)]
                    )
                    if args.allow_trainer_repair_resume:
                        rigid_command.append(
                            "--allow-trainer-repair-resume"
                        )
                    else:
                        rigid_command.append("--allow-performance-resume")
                _run(
                    rigid_command,
                    env=train_env,
                    log=run / "logs/train_rigid_teacher.log",
                    dry_run=args.dry_run,
                )
            if not args.dry_run:
                rigid_payload = json.loads(
                    rigid_result.read_text(encoding="utf-8")
                )
                surface_warmstart_ply = Path(
                    rigid_payload["surface_ply"]
                ).resolve()
                surface_warmstart_manifest = (
                    rigid_teacher / "rigid_surface_handoff.json"
                ).resolve()
                if not surface_warmstart_manifest.is_file():
                    raise FileNotFoundError(surface_warmstart_manifest)
            else:
                surface_warmstart_ply = (
                    rigid_teacher
                    / "point_cloud"
                    / f"iteration_{int(args.rigid_iterations)}"
                    / "point_cloud.ply"
                ).resolve()
                surface_warmstart_manifest = (
                    rigid_teacher / "rigid_surface_handoff.json"
                ).resolve()

        result = teacher / "result.json"
        result_current = _teacher_result_is_current(
            result,
            evidence_hash=store_hash,
            initialization=initialization,
            iterations=args.iterations,
            training_profile=profile["training_profile"],
            maximum_surface_gaussians=args.maximum_surface_gaussians,
            maximum_surface_growth_per_event=(
                args.maximum_surface_growth_per_event
            ),
            maximum_volume_gaussians=args.maximum_volume_gaussians,
            maximum_volume_splits=args.maximum_volume_splits,
            geometry_gradient_ratio=float(
                profile["geometry_gradient_ratio"]
            ),
            volume_densify_until_iteration=(
                args.volume_densify_until_iteration
            ),
            surface_warmstart_ply=surface_warmstart_ply,
            surface_warmstart_manifest=surface_warmstart_manifest,
            mature_handoff_surface_policy=profile[
                "mature_handoff_surface_policy"
            ],
            maximum_rigid_completion_seeds=int(
                profile["maximum_rigid_completion_seeds"]
            ),
            volume_opacity_lr=float(profile["volume_opacity_lr"]),
            surface_retirement_optical_mass_fraction_per_event=float(
                profile[
                    "surface_retirement_optical_mass_fraction_per_event"
                ]
            ),
        )
        if result.is_file() and not result_current:
            raise RuntimeError(
                "Existing Teacher result is stale relative to the current "
                "evidence, initialization, implementation, or state file. "
                "Refusing to report or resume it as a completed run; use a "
                "fresh --run-root so the causal experiment remains intact."
            )
        if not result_current:
            command = _teacher_command(
                python=python,
                dataset=dataset,
                teacher=teacher,
                evidence=evidence,
                initialization=initialization,
                iterations=int(args.iterations),
                training_profile=profile["training_profile"],
                maximum_surface_gaussians=(
                    args.maximum_surface_gaussians
                ),
                maximum_surface_growth_per_event=(
                    args.maximum_surface_growth_per_event
                ),
                maximum_volume_gaussians=args.maximum_volume_gaussians,
                maximum_volume_splits=args.maximum_volume_splits,
                checkpoint_every=int(profile["checkpoint_every"]),
                geometry_gradient_ratio=float(
                    profile["geometry_gradient_ratio"]
                ),
                rgb_images=rgb_images,
                volume_densify_until_iteration=(
                    args.volume_densify_until_iteration
                ),
                surface_warmstart_ply=surface_warmstart_ply,
                surface_warmstart_manifest=surface_warmstart_manifest,
                mature_handoff_surface_policy=profile[
                    "mature_handoff_surface_policy"
                ],
                maximum_rigid_completion_seeds=int(
                    profile["maximum_rigid_completion_seeds"]
                ),
                volume_opacity_lr=float(profile["volume_opacity_lr"]),
                surface_retirement_optical_mass_fraction_per_event=float(
                    profile[
                        "surface_retirement_optical_mass_fraction_per_event"
                    ]
                ),
            )
            checkpoint = teacher / "hybrid_teacher_checkpoint.pth"
            if checkpoint.is_file():
                command.extend(["--resume", str(checkpoint)])
                if args.allow_trainer_repair_resume:
                    command.append("--allow-trainer-repair-resume")
                else:
                    command.append("--allow-performance-resume")
            _run(
                command,
                env=train_env,
                log=run / "logs/train_teacher.log",
                dry_run=args.dry_run,
            )
        if args.dry_run:
            return
        trained = json.loads(result.read_text())
        manifest["stages"]["train_teacher"] = {
            "status": "complete",
            "staged_rigid_handoff": args.rigid_iterations is not None,
            "rigid_result": (
                str(rigid_result)
                if rigid_result is not None
                else None
            ),
            "surface_warmstart_ply": (
                str(surface_warmstart_ply)
                if surface_warmstart_ply is not None
                else None
            ),
            "surface_warmstart_manifest": (
                str(surface_warmstart_manifest)
                if surface_warmstart_manifest is not None
                else None
            ),
            "result": str(result),
            "teacher_state": trained["teacher_state"],
        }
        _write_manifest(manifest_path, manifest)

    evaluation = run / "evaluation/full_train_fit"
    if "evaluate_teacher" in selected:
        physical_gpu = _gpu(args.gpu, args.minimum_free_gpu_memory_mib)
        eval_env = dict(env)
        eval_env["CUDA_VISIBLE_DEVICES"] = physical_gpu
        trained = json.loads((teacher / "result.json").read_text())
        teacher_state = Path(trained["teacher_state"]).resolve()
        evaluation_mode = (
            "rigid"
            if profile["training_profile"]
            == "hybrid_rigid_stage1"
            else "hybrid"
        )
        evaluation_store = load_evidence_store(evidence)
        semantic_contract = Path(
            evaluation_store["semantic_contract"]
        ).resolve()
        if not semantic_contract.is_file():
            raise FileNotFoundError(semantic_contract)
        scene_contract = json.loads(
            Path(evaluation_store["scene_contract"]).read_text(
                encoding="utf-8"
            )
        )
        expected_view_count = len(scene_contract["records"])
        evaluation_rgb_source = rgb_source_contract(
            SimpleNamespace(
                images=(
                    str(args.rgb_images.expanduser().resolve())
                    if args.rgb_images is not None
                    else "images"
                ),
                source_path=str(dataset),
            )
        )
        expected_camera_geometry_sha256 = trained[
            "training_contract"
        ]["camera_geometry_sha256"]
        if not _evaluation_is_current(
            evaluation / "metrics.json",
            teacher_state=teacher_state,
            evaluation_mode=evaluation_mode,
            expected_view_count=expected_view_count,
            expected_rgb_source=evaluation_rgb_source,
            expected_camera_geometry_sha256=(
                expected_camera_geometry_sha256
            ),
        ):
            evaluation_command = [
                    python,
                    str(REPO_ROOT / "scripts/evaluate_hybrid_teacher.py"),
                    "-s",
                    str(dataset),
                    "-m",
                    str(run / "evaluation_scene"),
                    "--resolution",
                    "640",
                    "--teacher-state",
                    trained["teacher_state"],
                    "--semantic-contract",
                    str(semantic_contract),
                    "--tree-mask-pickle",
                    str(tree_mask),
                    "--output",
                    str(evaluation),
                    "--all",
                    "--render-only",
                    "--evaluation-mode",
                    evaluation_mode,
                ]
            if args.rgb_images is not None:
                evaluation_command.extend(
                    [
                        "--images",
                        str(args.rgb_images.expanduser().resolve()),
                    ]
                )
            _run(
                evaluation_command,
                env=eval_env,
                log=run / "logs/evaluate_teacher.log",
                dry_run=args.dry_run,
            )
        if args.dry_run:
            return
        metrics = json.loads((evaluation / "metrics.json").read_text())
        query_dataset = (
            args.query_dataset.expanduser().resolve()
            if args.query_dataset is not None
            else (
                Path(
                    "/mnt/pool/sqy/G4Splat_runs/"
                    "cambridge_hybrid_contract_closed_v7/query_controls"
                )
                / (
                    f"{args.scene}_official_query64_"
                    "shared640_bilinear_v1"
                )
            )
        )
        query_metrics = None
        if args.query_dataset is not None and not query_dataset.is_dir():
            raise FileNotFoundError(query_dataset)
        if query_dataset.is_dir():
            query_contract = (
                query_dataset / "query_scene_contract.json"
            ).resolve()
            # The evidence store owns the content-addressed fixed-camera
            # database contract. Prepared dataset adapters do not all place a
            # duplicate `scene_manifest.json` directly under their split
            # directory, so reconstructing this path from `dataset` is not a
            # valid provenance lookup.
            database_contract = Path(
                evaluation_store["scene_contract"]
            ).resolve()
            if not query_contract.is_file():
                raise FileNotFoundError(query_contract)
            if not database_contract.is_file():
                raise FileNotFoundError(database_contract)
            query_payload = json.loads(
                query_contract.read_text(encoding="utf-8")
            )
            query_evaluation = run / "evaluation/official_query64"
            query_evaluation_mode = (
                "rigid" if evaluation_mode == "rigid" else "canonical"
            )
            query_rgb_source = rgb_source_contract(
                SimpleNamespace(
                    images=str(query_dataset / "images"),
                    source_path=str(query_dataset),
                )
            )
            if not _evaluation_is_current(
                query_evaluation / "metrics.json",
                teacher_state=teacher_state,
                evaluation_mode=query_evaluation_mode,
                expected_view_count=int(query_payload["image_count"]),
                expected_rgb_source=query_rgb_source,
                expected_database_contract=database_contract,
                expected_query_contract=query_contract,
            ):
                _run(
                    [
                        python,
                        str(
                            REPO_ROOT
                            / "scripts/evaluate_hybrid_teacher.py"
                        ),
                        "-s",
                        str(query_dataset),
                        "-m",
                        str(query_evaluation),
                        "--images",
                        str(query_dataset / "images"),
                        "--resolution",
                        "640",
                        "--teacher-state",
                        str(teacher_state),
                        "--semantic-contract",
                        str(semantic_contract),
                        "--tree-mask-pickle",
                        str(tree_mask),
                        "--output",
                        str(query_evaluation),
                        "--all",
                        "--render-only",
                        "--evaluation-mode",
                        query_evaluation_mode,
                        "--evaluation-split",
                        "localization_query",
                        "--database-contract",
                        str(database_contract),
                        "--query-contract",
                        str(query_contract),
                    ],
                    env=eval_env,
                    log=run / "logs/evaluate_official_query64.log",
                    dry_run=args.dry_run,
                )
            if not args.dry_run:
                query_metrics = json.loads(
                    (query_evaluation / "metrics.json").read_text(
                        encoding="utf-8"
                    )
                )
        manifest["stages"]["evaluate_teacher"] = {
            "status": "complete",
            "metrics": str(evaluation / "metrics.json"),
            "aggregate": metrics["aggregate"],
            "protocol_aggregate": metrics.get("protocol_aggregate"),
            "metric_protocol": metrics.get("metric_protocol"),
            "official_query64": (
                {
                    "metrics": str(
                        run / "evaluation/official_query64/metrics.json"
                    ),
                    "protocol_aggregate": query_metrics.get(
                        "protocol_aggregate"
                    ),
                    "metric_protocol": query_metrics.get(
                        "metric_protocol"
                    ),
                }
                if query_metrics is not None
                else {
                    "status": "not_available",
                    "query_dataset": str(query_dataset),
                }
            ),
        }
        _write_manifest(manifest_path, manifest)

    if "export_geometry" in selected:
        trained = json.loads((teacher / "result.json").read_text())
        store = load_evidence_store(evidence)
        export = run / "export"
        export.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "hybrid-teacher-final-assets-v1",
            "authoritative_model": trained["teacher_state"],
            "renderer_manifest": str(teacher / "renderer_manifest.json"),
            "structural_2dgs_ply": trained["surface_ply"],
            "stable_mast3r_tracks": str(tracks),
            "scene_contract": store["scene_contract"],
            "canonical_render_for_localization": True,
            "excluded_localization_roles": [
                "canonical_crown",
                "dynamic_leaf",
                "sky",
                "transient",
            ],
            "standard_student": None,
            "points3D_or_colmap_tracks_used": False,
            "mesh": {
                "status": "deferred_until_teacher_depth_render_complete",
                "method": "rigid-only adaptive TSDF from canonical surface depth",
            },
        }
        export_path = export / "assets.json"
        export_path.write_text(json.dumps(payload, indent=2) + "\n")
        manifest["stages"]["export_geometry"] = {
            "status": "complete",
            "assets": str(export_path),
        }
        _write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
