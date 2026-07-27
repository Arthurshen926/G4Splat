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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outdoor.evidence_store import load_evidence_store  # noqa: E402
from outdoor.mast3r_track_graph import validate_track_gate  # noqa: E402
from outdoor.role_aware_initialization import (  # noqa: E402
    INITIALIZATION_VERSION,
)


PIPELINE_VERSION = "cambridge-native-hybrid-teacher-mainline-v3-causal-repair"
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
        "iterations": 50_000,
        "training_profile": "hybrid_quality",
        "checkpoint_every": 1000,
        "track_stride": 5,
    },
    "fast": {
        "iterations": 18_000,
        "training_profile": "hybrid_fast",
        "checkpoint_every": 500,
        "track_stride": 7,
    },
}


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
    parser.add_argument("--profile", choices=tuple(PROFILES), default="quality")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--minimum-free-gpu-memory-mib", type=int, default=20_000)
    parser.add_argument("--maximum-surface-gaussians", type=int, default=300_000)
    parser.add_argument(
        "--maximum-surface-growth-per-event", type=int, default=1000
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-trainer-repair-resume", action="store_true"
    )
    args = parser.parse_args()
    if args.iterations is None:
        args.iterations = PROFILES[args.profile]["iterations"]
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
        / f"{args.scene}_hybrid_teacher_{args.profile}_v3"
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

    tracks = run / "tracks/mast3r_multiview_tracks.npz"
    if "build_mast3r_tracks" in selected:
        if not tracks.is_file():
            _run(
                [
                    python,
                    str(REPO_ROOT / "scripts/build_mast3r_multiview_tracks.py"),
                    "--mast3r-scene",
                    str(mast3r),
                    "--dataset",
                    str(dataset),
                    "--tree-mask-pickle",
                    str(tree_mask),
                    "--output",
                    str(tracks),
                    "--stride",
                    str(profile["track_stride"]),
                ],
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
            "continuous_training_factors": True,
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
                artifact_names = {
                    item["name"]
                    for item in current_store.get("artifacts", [])
                }
                if (
                    dav2_root is not None
                    and "dav2_index" not in artifact_names
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
            )
        if rebuild_initialization:
            command = [
                    python,
                    str(REPO_ROOT / "scripts/initialize_unified_outdoor_scene.py"),
                    "--evidence-store",
                    str(evidence),
                    "--output",
                    str(initialization),
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
        result = teacher / "result.json"
        if not result.is_file():
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
                "--iterations",
                str(args.iterations),
                "--training-profile",
                profile["training_profile"],
                "--maximum-surface-gaussians",
                str(args.maximum_surface_gaussians),
                "--maximum-surface-growth-per-event",
                str(args.maximum_surface_growth_per_event),
                "--checkpoint-every",
                str(profile["checkpoint_every"]),
            ]
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
            "result": str(result),
            "teacher_state": trained["teacher_state"],
        }
        _write_manifest(manifest_path, manifest)

    evaluation = run / "evaluation/full_train_fit"
    if "evaluate_teacher" in selected:
        physical_gpu = _gpu(args.gpu, args.minimum_free_gpu_memory_mib)
        eval_env = dict(env)
        eval_env["CUDA_VISIBLE_DEVICES"] = physical_gpu
        if not (evaluation / "metrics.json").is_file():
            trained = json.loads((teacher / "result.json").read_text())
            _run(
                [
                    python,
                    str(REPO_ROOT / "scripts/evaluate_hybrid_teacher.py"),
                    "-s",
                    str(dataset),
                    "-m",
                    str(run / "evaluation_scene"),
                    "--teacher-state",
                    trained["teacher_state"],
                    "--semantic-contract",
                    str(evidence / "task_semantics.json"),
                    "--tree-mask-pickle",
                    str(tree_mask),
                    "--output",
                    str(evaluation),
                    "--all",
                    "--render-only",
                ],
                env=eval_env,
                log=run / "logs/evaluate_teacher.log",
                dry_run=args.dry_run,
            )
        if args.dry_run:
            return
        metrics = json.loads((evaluation / "metrics.json").read_text())
        manifest["stages"]["evaluate_teacher"] = {
            "status": "complete",
            "metrics": str(evaluation / "metrics.json"),
            "aggregate": metrics["aggregate"],
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
