import argparse
from pathlib import Path
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))


def build_command(args, config):
    command = [
        sys.executable,
        "mast3r/run_mast3r.py",
        "--scene_path",
        args.source_path,
        "--output_dir",
        args.output_path,
        "--weights_path",
        str(config["weights_path"]),
        "--retrieval_model",
        str(config["retrieval_model"]),
        "--min_conf_thr",
        str(config["min_conf_thr"]),
        "--matching_conf_thr",
        str(config["matching_conf_thr"]),
        "--n_coarse_iterations",
        str(config["n_coarse_iterations"]),
        "--n_refinement_iterations",
        str(config["n_refinement_iterations"]),
        "--TSDF_thresh",
        str(config["TSDF_thresh"]),
        "--n_images",
        str(args.n_images),
    ]

    for key, flag in (
        ("fix_focal", "--fix_focal"),
        ("fix_principal_point", "--fix_principal_point"),
        ("fix_rotation", "--fix_rotation"),
        ("fix_translation", "--fix_translation"),
    ):
        if config.get(key, False):
            command.append(flag)

    if args.use_all_images:
        command.append("--use_all_images")
    if args.image_idx is not None:
        command.append("--image_idx")
        command.extend(str(index) for index in args.image_idx)
    if args.randomize_images:
        command.append("--randomize_images")

    command.extend(
        [
            "--image_size",
            str(config["image_size"]),
            "--max_window_size",
            str(config["max_window_size"]),
            "--max_refid",
            str(config["max_refid"]),
            "--output_conf_thr",
            str(config["output_conf_thr"]),
            "--sparse_export_stride",
            str(args.sparse_export_stride),
        ]
    )
    for key, flag in (
        ("use_calibrated_poses", "--use_calibrated_poses"),
        ("save_glb", "--save_glb"),
        ("align_camera_locations", "--align_camera_locations"),
    ):
        if config.get(key, False):
            command.append(flag)
    if getattr(args, "strict_calibrated_poses", False):
        command.append("--strict_calibrated_poses")
    if getattr(args, "per_view_calibrated_intrinsics", False):
        command.append("--per_view_calibrated_intrinsics")
    return command


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--source_path",
        type=str,
        required=True,
        help="Directory containing images or a COLMAP dataset.",
    )
    parser.add_argument("-o", "--output_path", type=str, default=None)
    parser.add_argument(
        "--n_images",
        type=int,
        default=None,
        help="Number of uniformly sampled views. Prefer explicit --image_idx.",
    )
    parser.add_argument(
        "--image_idx",
        type=int,
        nargs="+",
        default=None,
        help="Explicit zero-based view indices; mutually exclusive with --n_images.",
    )
    parser.add_argument("--randomize_images", action="store_true")
    parser.add_argument(
        "--sparse-export-stride",
        type=int,
        default=1,
        help=(
            "Regular image-lattice stride for MASt3R's diagnostic COLMAP sparse export. "
            "It does not downsample the pointmaps consumed by Chart alignment."
        ),
    )
    parser.add_argument(
        "--strict_calibrated_poses",
        action="store_true",
        help=(
            "Keep calibrated camera translation and scale fixed during MASt3R "
            "global alignment (an explicit single-variable ablation)."
        ),
    )
    parser.add_argument(
        "--per_view_calibrated_intrinsics",
        action="store_true",
        help=(
            "Keep individual calibrated intrinsics rather than the historical "
            "shared-focal MASt3R parameterization."
        ),
    )
    parser.add_argument("-c", "--config", type=str, default="unposed")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_images is not None and args.image_idx is not None:
        raise ValueError("Cannot provide both --n_images and --image_idx.")
    if args.sparse_export_stride < 1:
        raise ValueError("--sparse-export-stride must be at least one")

    if args.image_idx is not None:
        args.use_all_images = False
        args.randomize_images = False
        args.n_images = len(args.image_idx)
        print(f"[INFO] Using {args.n_images} explicitly selected images.")
    elif args.n_images is None:
        args.use_all_images = True
        args.randomize_images = False
        args.n_images = -1
        print("[INFO] Using all images for optimization.")
    else:
        args.use_all_images = False
        print(f"[INFO] Using {args.n_images} sampled images.")

    if args.output_path is None:
        source_name = Path(args.source_path).resolve().name
        args.output_path = str(Path("output") / source_name / "mast3r_sfm")
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Scene will be saved to: {args.output_path}")

    config_path = REPO_ROOT / "configs" / "mast3r" / f"{args.config}.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    command = build_command(args, config)
    print("[INFO] Running command:\n", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
