#!/usr/bin/env python3
"""Run the frozen-MAtCha, 3D-ROI artifact repair pipeline in auditable phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, dry_run: bool) -> None:
    print("[CMD] " + " ".join(map(str, command)), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    adapter = args.output.expanduser().resolve()
    artifact_source = adapter / "mast3r_sfm"
    baseline_model = adapter / "baseline_model"
    diagnostics = adapter / "roi_diagnostics"
    stage_root = artifact_source / "see3d_render" / f"stage{args.stage}"
    edited_model = adapter / "roi_edited_model"
    python = sys.executable
    common_model = [
        "-s", str(args.dense_dataset),
        "-m", str(baseline_model),
        "--data_device", args.data_device,
        "--resolution", str(args.resolution),
        "--iteration", str(args.iteration),
        "--artifact_source_path", str(artifact_source),
        "--diagnostics_dir", str(diagnostics),
        "--mask_pickle", str(args.mask_pickle),
        "--mask_dataset_path", str(args.dense_dataset),
        "--semantic_mask_indices", *(str(index) for index in args.semantic_mask_indices),
        "--stage", str(args.stage),
    ]
    if args.target_image_names_file is not None:
        common_model.extend(
            ["--target_image_names_file", str(args.target_image_names_file)]
        )
    return [
        (
            "prepare",
            [
                python,
                "scripts/prepare_matcha_warmstart.py",
                "--matcha_scene", str(args.matcha_scene),
                "--matcha_model", str(args.matcha_model),
                "--output", str(adapter),
                "--iteration", str(args.iteration),
            ],
        ),
        (
            "diagnose",
            [
                python,
                "scripts/propose_artifact_guided_views.py",
                *common_model,
                "--diagnostics_only",
            ],
        ),
        (
            "propose",
            [
                python,
                "scripts/propose_artifact_guided_views.py",
                *common_model,
                "--reuse_diagnostics",
                "--replace_stage",
            ],
        ),
        (
            "projective",
            [
                python,
                "scripts/render_projective_artifact_repair.py",
                "--stage_root", str(stage_root),
                "--diagnostics_dir", str(diagnostics),
                "--replace_output",
            ],
        ),
        (
            "filter",
            [
                python,
                "scripts/filter_artifact_guided_views.py",
                "--stage_root", str(stage_root),
                "--repair_source", "projective",
                "--max_synthesized_fraction", "0.15",
                "--apply_depth_policy",
                "--apply_repair_policy",
            ],
        ),
        (
            "edit",
            [
                python,
                "scripts/edit_wrong_geometry_gaussians.py",
                "--input_model", str(baseline_model),
                "--input_iteration", str(args.iteration),
                "--stage_root", str(stage_root),
                "--diagnostics_dir", str(diagnostics),
                "--output_model", str(edited_model),
                "--replace_output",
            ],
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["prepare", "diagnose", "propose", "projective", "filter", "edit", "all"],
        default="all",
    )
    parser.add_argument("--matcha_scene", type=Path, required=True)
    parser.add_argument("--matcha_model", type=Path, required=True)
    parser.add_argument("--dense_dataset", type=Path, required=True)
    parser.add_argument("--mask_pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--semantic-mask-indices",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Validity-mask entries required for artifact diagnosis; include 3 for tree-aware masks.",
    )
    parser.add_argument(
        "--target-image-names-file",
        type=Path,
        default=None,
        help="Optional newline-delimited target views for support attribution after diagnostics.",
    )
    parser.add_argument(
        "--data-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Keep diagnostic camera images on CPU instead of filling VRAM.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=2,
        help="Diagnostic image downscale factor forwarded to ModelParams.",
    )
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = commands(args)
    if args.phase != "all":
        selected = [item for item in selected if item[0] == args.phase]
    for name, command in selected:
        print(f"[PHASE] {name}", flush=True)
        run(command, dry_run=args.dry_run)
    manifest = {
        "mode": "frozen_matcha_roi_repair",
        "phase": args.phase,
        "matcha_scene": str(args.matcha_scene.resolve()),
        "matcha_model": str(args.matcha_model.resolve()),
        "dense_dataset": str(args.dense_dataset.resolve()),
        "semantic_mask_indices": list(args.semantic_mask_indices),
        "target_image_names_file": (
            None
            if args.target_image_names_file is None
            else str(args.target_image_names_file.resolve())
        ),
        "output": str(args.output.resolve()),
        "baseline_frozen_by_policy": True,
        "see3d_policy": "projective_real_support_first; residual generation remains optional",
    }
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "roi_repair_run_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )


if __name__ == "__main__":
    main()
