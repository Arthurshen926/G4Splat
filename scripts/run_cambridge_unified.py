#!/usr/bin/env python
"""Compatibility entry point for the single Cambridge Hybrid Teacher mainline.

The former script launched a separate from-seed Teacher, a standard-3DGS
student and a second evaluator.  That path had diverged from the native mixed
Teacher and even looked for a checkpoint filename that the active trainer
never writes.  Keep the familiar command name, but translate its supported
arguments to :mod:`run_cambridge_hybrid_teacher` so there is one producer,
checkpoint contract and evaluator.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTHORITATIVE_RUNNER = (
    REPO_ROOT / "scripts" / "run_cambridge_hybrid_teacher.py"
)

LEGACY_STAGE_MAP = {
    "prepare": "prepare_cameras",
    "evidence": "build_evidence",
    "initialize": "initialize_teacher",
    "train_teacher": "train_teacher",
    "evaluate": "evaluate_teacher",
    # The authoritative evaluator already produces the disjoint canonical
    # query result; there is no separate model or conditioning path here.
    "localization": "evaluate_teacher",
    "all": "all",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="StMarysChurch")
    parser.add_argument(
        "--stage",
        choices=(*LEGACY_STAGE_MAP, "distill_student"),
        default="all",
    )
    parser.add_argument(
        "--frontend-output-root",
        "--frontend-root",
        dest="frontend_root",
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
        "--rgb-root", "--rgb-images", dest="rgb_images", type=Path
    )
    parser.add_argument("--query-dataset", type=Path)
    parser.add_argument(
        "--profile", choices=("quality", "fast", "rigid"), default="quality"
    )
    parser.add_argument(
        "--teacher-iterations", "--iterations", dest="iterations", type=int
    )
    parser.add_argument("--rigid-iterations", type=int)
    parser.add_argument("--maximum-surface-gaussians", type=int)
    parser.add_argument("--maximum-surface-growth-per-event", type=int)
    parser.add_argument("--maximum-volume-gaussians", type=int)
    parser.add_argument("--maximum-volume-splits", type=int)
    parser.add_argument("--gpu", default="2")
    parser.add_argument(
        "--minimum-free-gpu-memory-mib", type=int, default=20_000
    )
    parser.add_argument("--allow-trainer-repair-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    # Parse removed options only to return an actionable migration error
    # instead of silently changing the experiment.
    parser.add_argument("--student-iterations", type=int)
    parser.add_argument("--build-dav2-cache", action="store_true")
    parser.add_argument("--datasets-root", type=Path)
    parser.add_argument("--frontend-run-tag")
    parser.add_argument("--broad-charts", type=int)
    parser.add_argument("--min-charts", type=int)
    args = parser.parse_args()

    if args.stage == "distill_student" or args.student_iterations is not None:
        parser.error(
            "The student/distillation branch was removed from the requested "
            "Teacher-only mainline. Use --stage train_teacher/evaluate, or "
            "invoke an exporter explicitly as a separately labelled task."
        )
    removed = []
    if args.build_dav2_cache:
        removed.append("--build-dav2-cache")
    argv = set(sys.argv[1:])
    for name in (
        "--datasets-root",
        "--frontend-run-tag",
        "--broad-charts",
        "--min-charts",
    ):
        if name in argv:
            removed.append(name)
    if removed:
        parser.error(
            "These options belonged to the retired duplicate front-end: "
            + ", ".join(removed)
            + ". Build/choose the immutable front-end with --frontend-run, "
            "then run the authoritative Teacher pipeline."
        )
    return args


def _append(command: list[str], flag: str, value) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def main() -> None:
    args = _args()
    command = [
        sys.executable,
        str(AUTHORITATIVE_RUNNER),
        "--scene",
        args.scene,
        "--stage",
        LEGACY_STAGE_MAP[args.stage],
        "--frontend-root",
        str(args.frontend_root),
        "--run-root",
        str(args.run_root),
        "--mask-root",
        str(args.mask_root),
        "--profile",
        args.profile,
        "--gpu",
        str(args.gpu),
        "--minimum-free-gpu-memory-mib",
        str(args.minimum_free_gpu_memory_mib),
    ]
    _append(command, "--frontend-run", args.frontend_run)
    _append(command, "--dav2-root", args.dav2_root)
    _append(command, "--rgb-images", args.rgb_images)
    _append(command, "--query-dataset", args.query_dataset)
    _append(command, "--iterations", args.iterations)
    _append(command, "--rigid-iterations", args.rigid_iterations)
    _append(
        command,
        "--maximum-surface-gaussians",
        args.maximum_surface_gaussians,
    )
    _append(
        command,
        "--maximum-surface-growth-per-event",
        args.maximum_surface_growth_per_event,
    )
    _append(
        command,
        "--maximum-volume-gaussians",
        args.maximum_volume_gaussians,
    )
    _append(command, "--maximum-volume-splits", args.maximum_volume_splits)
    if args.allow_trainer_repair_resume:
        command.append("--allow-trainer-repair-resume")
    if args.dry_run:
        command.append("--dry-run")
    completed = subprocess.run(command, cwd=REPO_ROOT)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
