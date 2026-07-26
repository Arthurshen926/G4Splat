#!/usr/bin/env python
"""Single restartable Cambridge evidence→teacher→standard-3DGS mainline."""

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
from outdoor.standard_3dgs import validate_standard_3dgs_ply  # noqa: E402


PIPELINE_VERSION = "cambridge-unified-reconstruction-v2"
EXPERIMENT_PROFILES = {
    "quality": {
        "teacher_iterations": 50_000,
        "student_iterations": 30_000,
        "teacher_profile": "quality",
        "checkpoint_every": 1000,
    },
    "fast": {
        "teacher_iterations": 20_000,
        "student_iterations": 15_000,
        "teacher_profile": "fast",
        "checkpoint_every": 500,
    },
}
STAGES = (
    "prepare",
    "evidence",
    "initialize",
    "train_teacher",
    "distill_student",
    "evaluate",
    "localization",
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="StMarysChurch")
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
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
        "--frontend-output-root",
        type=Path,
        default=Path(
            "/mnt/pool/sqy/G4Splat_runs/cambridge_outdoor_mainline_v1"
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "/mnt/pool/sqy/G4Splat_runs/cambridge_unified_v1"
        ),
    )
    parser.add_argument("--frontend-run", type=Path)
    parser.add_argument(
        "--dav2-root",
        type=Path,
        help=(
            "Optional all-real-view Depth Anything V2 cache. Files are "
            "indexed by database image stem and consumed as rigid-only "
            "ordinal evidence."
        ),
    )
    parser.add_argument(
        "--build-dav2-cache",
        action="store_true",
        help=(
            "Generate/resume a DAV2 cache under the run directory before "
            "building the immutable evidence store."
        ),
    )
    parser.add_argument("--frontend-run-tag", default="unified-evidence-v1")
    parser.add_argument("--broad-charts", type=int, default=56)
    parser.add_argument("--min-charts", type=int, default=24)
    parser.add_argument(
        "--profile",
        choices=tuple(EXPERIMENT_PROFILES),
        default="quality",
    )
    parser.add_argument("--teacher-iterations", type=int)
    parser.add_argument("--student-iterations", type=int)
    parser.add_argument(
        "--maximum-surface-gaussians",
        type=int,
        default=1_200_000,
    )
    parser.add_argument(
        "--maximum-surface-growth-per-event",
        type=int,
        default=20_000,
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--minimum-free-gpu-memory-mib",
        type=int,
        default=12_000,
        help=(
            "Refuse to start the mixed teacher below this physical-GPU "
            "free-memory threshold; use --gpu auto to select the freest GPU."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    profile = EXPERIMENT_PROFILES[args.profile]
    if args.teacher_iterations is None:
        args.teacher_iterations = profile["teacher_iterations"]
    if args.student_iterations is None:
        args.student_iterations = profile["student_iterations"]
    if args.teacher_iterations <= 0 or args.student_iterations <= 0:
        parser.error("Teacher/student iterations must be positive")
    if args.minimum_free_gpu_memory_mib < 0:
        parser.error("--minimum-free-gpu-memory-mib must be non-negative")
    return args


def _gpu_memory_rows() -> list[tuple[str, int, int]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        index, free, total = [
            value.strip() for value in line.split(",")
        ]
        rows.append((index, int(free), int(total)))
    if not rows:
        raise RuntimeError("nvidia-smi returned no physical GPUs")
    return rows


def _resolve_training_gpu(requested: str, minimum_free_mib: int) -> str:
    rows = _gpu_memory_rows()
    by_index = {index: (free, total) for index, free, total in rows}
    requested = str(requested).strip().lower()
    if requested == "auto":
        selected = max(rows, key=lambda row: row[1])
        index, free, total = selected
    else:
        if requested not in by_index:
            raise RuntimeError(
                f"Physical GPU {requested!r} was not found; available "
                f"indices are {sorted(by_index)}"
            )
        index = requested
        free, total = by_index[index]
    if free < int(minimum_free_mib):
        availability = ", ".join(
            f"GPU {gpu}: {gpu_free}/{gpu_total} MiB free"
            for gpu, gpu_free, gpu_total in rows
        )
        raise RuntimeError(
            f"GPU {index} has only {free} MiB free; the mixed teacher "
            f"requires at least {minimum_free_mib} MiB by policy. "
            f"Current availability: {availability}. Choose a free GPU or "
            "pass --gpu auto."
        )
    print(
        f"Selected physical GPU {index}: {free}/{total} MiB free",
        flush=True,
    )
    return index


def _log_tail(path: Path, max_bytes: int = 24_000) -> str:
    if not path.is_file():
        return "(stage log was not created)"
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(size - max_bytes, 0))
        return handle.read().decode("utf-8", errors="replace")


def _run(command: list[str], *, env: dict, log: Path, dry_run: bool):
    log.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(" ".join(command))
        return
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        try:
            subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            handle.flush()
            raise RuntimeError(
                "Pipeline stage failed. The final stage log follows:\n\n"
                f"{_log_tail(log)}"
            ) from error


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_frontend_run(args) -> Path:
    if args.frontend_run is not None:
        root = args.frontend_run.expanduser().resolve()
        if not (root / "mast3r_sfm" / "charts_data.npz").is_file():
            raise FileNotFoundError(
                f"Frontend run has no Chart geometry: {root}"
            )
        return root
    runs = args.frontend_output_root / "runs"
    candidates = []
    for manifest in runs.glob(
        f"{args.scene}_g4_qc_n{args.broad_charts}_*/outdoor_mainline_manifest.json"
    ):
        try:
            payload = _load_json(manifest)
        except (OSError, json.JSONDecodeError):
            continue
        mast3r = manifest.parent / "mast3r_sfm"
        if (
            payload.get("scene") == args.scene
            and (mast3r / "charts_data.npz").is_file()
            and (
                mast3r
                / "inverse_depth_fusion"
                / "inverse_depth_fusion_manifest.json"
            ).is_file()
            and (
                mast3r
                / "plane-refine-depths"
                / "plane_residual_source_manifest.json"
            ).is_file()
        ):
            candidates.append(manifest.parent)
    if not candidates:
        raise FileNotFoundError(
            "No completed fixed-camera evidence frontend was found"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _write_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _mark_stage(
    manifest: dict,
    path: Path,
    stage: str,
    *,
    status: str,
    artifacts: dict | None = None,
) -> None:
    manifest.setdefault("stages", {})[stage] = {
        "status": status,
        "updated_unix": time.time(),
        "artifacts": artifacts or {},
    }
    _write_manifest(path, manifest)


def _stages_to_run(requested: str) -> tuple[str, ...]:
    if requested == "all":
        return STAGES
    # A requested terminal stage includes its prerequisites.  Existing
    # completed outputs are validated and skipped, so this is restartable.
    return STAGES[: STAGES.index(requested) + 1]


def main():
    args = _parse_args()
    selected = _stages_to_run(args.stage)
    if "train_teacher" in selected and not args.dry_run:
        args.gpu = _resolve_training_gpu(
            args.gpu, args.minimum_free_gpu_memory_mib
        )
    run_suffix = (
        "unified_v2"
        if args.profile == "quality"
        else f"unified_{args.profile}_v2"
    )
    run = (args.run_root / f"{args.scene}_{run_suffix}").resolve()
    run.mkdir(parents=True, exist_ok=True)
    manifest_path = run / "pipeline_manifest.json"
    manifest = (
        _load_json(manifest_path)
        if manifest_path.is_file()
        else {
            "version": PIPELINE_VERSION,
            "scene": args.scene,
            "run_root": str(run),
            "database_query_policy": "database_only_until_export",
            "historical_parent_ply_allowed": False,
            "experiment_profile": args.profile,
            "stages": {},
        }
    )
    if manifest.get("version") != PIPELINE_VERSION:
        raise RuntimeError("Pipeline manifest version mismatch")
    manifest_profile = manifest.get("experiment_profile", "quality")
    if manifest_profile != args.profile:
        raise RuntimeError(
            "Pipeline experiment profile mismatch: "
            f"{manifest_profile} != {args.profile}"
        )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    conda_lib = str(Path(sys.prefix) / "lib")
    library_entries = [
        value
        for value in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if value
    ]
    if conda_lib not in library_entries:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [conda_lib, *library_entries]
        )
    env.setdefault("PYTHONUNBUFFERED", "1")
    python = sys.executable
    frontend_file = run / "frontend_run.json"

    if "prepare" in selected:
        if not frontend_file.is_file():
            command = [
                python,
                str(REPO_ROOT / "scripts/run_cambridge_outdoor_mainline.py"),
                "--scene",
                args.scene,
                "--phase",
                "planes",
                "--datasets-root",
                str(args.datasets_root),
                "--mask-root",
                str(args.mask_root),
                "--output-root",
                str(args.frontend_output_root),
                "--run-tag",
                args.frontend_run_tag,
                "--broad-charts",
                str(args.broad_charts),
                "--min-charts",
                str(args.min_charts),
                "--gpu",
                str(args.gpu),
            ]
            if args.frontend_run is None:
                _run(
                    command,
                    env=env,
                    log=run / "logs/prepare.log",
                    dry_run=args.dry_run,
                )
            if args.dry_run:
                return
            frontend = _find_frontend_run(args)
            frontend_file.write_text(
                json.dumps({"path": str(frontend)}, indent=2) + "\n",
                encoding="utf-8",
            )
        frontend = Path(_load_json(frontend_file)["path"])
        frontend_manifest = _load_json(
            frontend / "outdoor_mainline_manifest.json"
        )
        dataset = Path(frontend_manifest["dense_real_dataset"])
        if not (dataset / "sparse" / "0" / "points3D.bin").is_file():
            raise FileNotFoundError(
                f"Prepared all-real database is incomplete: {dataset}"
            )
        _mark_stage(
            manifest,
            manifest_path,
            "prepare",
            status="complete",
            artifacts={
                "frontend_run": str(frontend),
                "dataset": str(dataset),
                "mast3r_scene": str(frontend / "mast3r_sfm"),
            },
        )
    else:
        frontend = Path(_load_json(frontend_file)["path"])
        frontend_manifest = _load_json(
            frontend / "outdoor_mainline_manifest.json"
        )
        dataset = Path(frontend_manifest["dense_real_dataset"])

    evidence = run / "evidence"
    if "evidence" in selected:
        evidence_manifest = evidence / "evidence_manifest.json"
        if not evidence_manifest.is_file():
            dav2_root = args.dav2_root
            if args.build_dav2_cache:
                dav2_root = run / "dav2_real_views"
                _run(
                    [
                        python,
                        str(
                            REPO_ROOT
                            / "scripts/cache_dav2_real_views.py"
                        ),
                        "--dataset",
                        str(dataset),
                        "--output",
                        str(dav2_root),
                    ],
                    env=env,
                    log=run / "logs/dav2.log",
                    dry_run=args.dry_run,
                )
            command = [
                python,
                str(
                    REPO_ROOT
                    / "scripts/build_unified_scene_evidence.py"
                ),
                    "--dataset",
                    str(dataset),
                    "--mask-pickle",
                    str(
                        args.mask_root
                        / args.scene
                        / "processed/masks.pkl"
                    ),
                    "--tree-mask-pickle",
                    str(
                        args.mask_root
                        / args.scene
                        / "processed/masks_with_tree.pkl"
                    ),
                    "--mast3r-scene",
                    str(frontend / "mast3r_sfm"),
                    "--output",
                    str(evidence),
            ]
            if dav2_root is not None:
                command.extend(
                    ["--dav2-root", str(dav2_root.resolve())]
                )
            _run(
                command,
                env=env,
                log=run / "logs/evidence.log",
                dry_run=args.dry_run,
            )
        if args.dry_run:
            return
        evidence_payload = load_evidence_store(evidence)
        _mark_stage(
            manifest,
            manifest_path,
            "evidence",
            status="complete",
            artifacts={
                "manifest": str(evidence_manifest),
                "evidence_hash": evidence_payload["evidence_hash"],
            },
        )

    initialization = run / "initialization"
    if "initialize" in selected:
        initialization_manifest = (
            initialization / "initialization_manifest.json"
        )
        if not initialization_manifest.is_file():
            _run(
                [
                    python,
                    str(
                        REPO_ROOT
                        / "scripts/initialize_unified_outdoor_scene.py"
                    ),
                    "--evidence-store",
                    str(evidence),
                    "--output",
                    str(initialization),
                ],
                env=env,
                log=run / "logs/initialize.log",
                dry_run=args.dry_run,
            )
        if args.dry_run:
            return
        init = _load_json(initialization_manifest)
        if init.get("historical_model_initialization") is not False:
            raise RuntimeError("Unified initialization used a historical model")
        _mark_stage(
            manifest,
            manifest_path,
            "initialize",
            status="complete",
            artifacts={
                "manifest": str(initialization_manifest),
                "surface_seed": init["surface_seed"],
                "foliage_seed": init["foliage_seed"],
            },
        )

    teacher = run / "teacher"
    if "train_teacher" in selected:
        teacher_result = teacher / "result.json"
        if not teacher_result.is_file():
            command = [
                python,
                str(
                    REPO_ROOT
                    / "scripts/train_unified_outdoor_teacher.py"
                ),
                "-s",
                str(dataset),
                "-m",
                str(teacher),
                "--evidence-store",
                str(evidence),
                "--initialization",
                str(initialization),
                "--iterations",
                str(args.teacher_iterations),
                "--training-profile",
                EXPERIMENT_PROFILES[args.profile]["teacher_profile"],
                "--maximum-surface-gaussians",
                str(args.maximum_surface_gaussians),
                "--maximum-surface-growth-per-event",
                str(args.maximum_surface_growth_per_event),
                "--checkpoint-every",
                str(
                    EXPERIMENT_PROFILES[args.profile][
                        "checkpoint_every"
                    ]
                ),
            ]
            checkpoint = teacher / "unified_teacher_checkpoint.pth"
            if checkpoint.is_file():
                command.extend(
                    [
                        "--resume",
                        str(checkpoint),
                        "--allow-performance-resume",
                    ]
                )
            _run(
                command,
                env=env,
                log=run / "logs/train_teacher.log",
                dry_run=args.dry_run,
            )
        if args.dry_run:
            return
        result = _load_json(teacher_result)
        if result.get("historical_parent_ply_used") is not False:
            raise RuntimeError("Teacher violated from-scratch initialization")
        _mark_stage(
            manifest,
            manifest_path,
            "train_teacher",
            status="complete",
            artifacts={
                "result": str(teacher_result),
                "state": result["teacher_state"],
            },
        )

    student = run / "student_3dgs"
    if "distill_student" in selected:
        distill_manifest = student / "distillation_manifest.json"
        if not distill_manifest.is_file():
            teacher_result = _load_json(teacher / "result.json")
            command = [
                python,
                str(REPO_ROOT / "scripts/distill_standard_3dgs.py"),
                "-s",
                str(dataset),
                "-m",
                str(student),
                "--teacher-state",
                teacher_result["teacher_state"],
                "--evidence-store",
                str(evidence),
                "--iterations",
                str(args.student_iterations),
            ]
            checkpoint = student / "student_checkpoint.pth"
            if checkpoint.is_file():
                command.extend(["--resume", str(checkpoint)])
            _run(
                command,
                env=env,
                log=run / "logs/distill_student.log",
                dry_run=args.dry_run,
            )
        if args.dry_run:
            return
        result = _load_json(distill_manifest)
        schema = validate_standard_3dgs_ply(
            Path(result["point_cloud"]), sh_degree=3
        )
        if any(
            result[key]
            for key in (
                "requires_custom_cuda",
                "requires_uv_gate",
                "requires_temporal_code",
                "requires_separate_sky",
                "requires_mixed_renderer",
            )
        ):
            raise RuntimeError("Student export still depends on teacher code")
        _mark_stage(
            manifest,
            manifest_path,
            "distill_student",
            status="complete",
            artifacts={
                "manifest": str(distill_manifest),
                "point_cloud": result["point_cloud"],
                "schema": schema,
            },
        )

    evaluation = run / "evaluation"
    if "evaluate" in selected:
        metrics = evaluation / "metrics.json"
        if not metrics.is_file():
            distill = _load_json(
                student / "distillation_manifest.json"
            )
            _run(
                [
                    python,
                    str(REPO_ROOT / "scripts/evaluate_standard_3dgs.py"),
                    "-s",
                    str(dataset),
                    "-m",
                    str(evaluation / "camera_cache"),
                    "--point-cloud",
                    distill["point_cloud"],
                    "--semantic-contract",
                    str(evidence / "task_semantics.json"),
                    "--tree-mask-pickle",
                    str(
                        args.mask_root
                        / args.scene
                        / "processed/masks_with_tree.pkl"
                    ),
                    "--output",
                    str(evaluation),
                ],
                env=env,
                log=run / "logs/evaluate.log",
                dry_run=args.dry_run,
            )
        if args.dry_run:
            return
        _mark_stage(
            manifest,
            manifest_path,
            "evaluate",
            status="complete",
            artifacts={"metrics": str(metrics)},
        )

    if "localization" in selected:
        distill = _load_json(student / "distillation_manifest.json")
        localization = {
            "version": "database-only-localization-export-v1",
            "database_evidence_hash": load_evidence_store(evidence)[
                "evidence_hash"
            ],
            "query_inputs_used": False,
            "standard_3dgs": distill["point_cloud"],
            "cameras": distill["cameras"],
            "matching_roles": ["rigid", "static_skeleton"],
            "excluded_feature_roles": [
                "dynamic_leaf",
                "sky",
                "transient",
            ],
            "downstream_interface": (
                "standard Graphdeco 3DGS PLY; no repository-specific "
                "teacher module is required"
            ),
        }
        localization_path = run / "localization/export.json"
        localization_path.parent.mkdir(parents=True, exist_ok=True)
        localization_path.write_text(
            json.dumps(localization, indent=2) + "\n",
            encoding="utf-8",
        )
        _mark_stage(
            manifest,
            manifest_path,
            "localization",
            status="complete",
            artifacts={"export": str(localization_path)},
        )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
