import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import subprocess
from pathlib import Path
import yaml

from rich.console import Console


def _jsonable_argument(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable_argument(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_argument(item) for key, item in value.items()}
    return value


def write_refinement_manifest(args, config: dict, command: list[str]) -> Path:
    """Record every effective refinement input before the subprocess starts.

    A Gaussian refinement has enough interacting options that a directory name
    alone cannot establish a single-variable ablation.  The manifest is
    intentionally written before launch, so a failed/partial run still has a
    complete declaration of its intended intervention.
    """
    output_path = Path(args.output_path).resolve()
    payload = {
        "version": 1,
        "mast3r_scene": str(Path(args.mast3r_scene).resolve()),
        "output_path": str(output_path),
        "free_gaussians_config": args.config,
        "effective_iterations": int(args.iterations or config["iterations"]),
        "config": _jsonable_argument(config),
        "arguments": _jsonable_argument(vars(args)),
        "command": [str(argument) for argument in command],
    }
    manifest_path = output_path / "refinement_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    return manifest_path


def apply_screen_manifest_dense_defaults(args):
    """Recover explicit screen settings without overriding its dense policy.

    New Cambridge manifests always record ``dense_final_only``.  A Chart-only
    screen must stay Chart-only all the way into train_with_refine_depth;
    otherwise the apparent Chart-vs-dense ablation has no intervention.
    Legacy manifests did not record this field, so retain their historical
    dense recovery only for backwards compatibility.
    """
    if args.output_path is None:
        return args
    manifest_path = Path(args.output_path).resolve().parent / "cambridge_g4_manifest.json"
    if not manifest_path.exists():
        return args
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("mode") != "screen7k":
        return args
    manifest_config = manifest.get("config", {})
    if manifest_config.get("use_color_correction", False):
        args.use_color_correction = True
        args.color_correction_lr = float(
            manifest_config.get("color_correction_lr", args.color_correction_lr)
        )
        args.color_correction_reg = float(
            manifest_config.get("color_correction_reg", args.color_correction_reg)
        )
        print(
            "[INFO] screen7k manifest enabled affine color correction: "
            f"lr={args.color_correction_lr}, reg={args.color_correction_reg}"
        )
    manifest_sampling_policy = manifest_config.get("chart_geometry_sampling_policy")
    if manifest_sampling_policy is not None:
        if manifest_sampling_policy not in {"legacy_all_input", "active_only"}:
            raise RuntimeError(
                "screen7k manifest has an invalid chart geometry sampling policy: "
                f"{manifest_sampling_policy!r}"
            )
        args.chart_geometry_sampling_policy = manifest_sampling_policy
        print(
            "[INFO] screen7k manifest chart geometry scheduler: "
            f"{args.chart_geometry_sampling_policy}"
        )
    manifest_densification_policy = manifest_config.get("densification_view_policy")
    if manifest_densification_policy is not None:
        if manifest_densification_policy not in {"legacy_current", "dense_only"}:
            raise RuntimeError(
                "screen7k manifest has an invalid densification view policy: "
                f"{manifest_densification_policy!r}"
            )
        args.densification_view_policy = manifest_densification_policy
        print(
            "[INFO] screen7k manifest densification statistics policy: "
            f"{args.densification_view_policy}"
        )
    if args.dense_data_path is not None:
        return args
    if manifest_config.get("dense_final_only") is True:
        print("[INFO] screen7k manifest requests chart-only preliminary supervision.")
        return args
    dense_dataset = manifest.get("dense_dataset")
    if not dense_dataset:
        raise RuntimeError(f"screen7k manifest has no dense_dataset: {manifest_path}")
    if not Path(dense_dataset).exists():
        raise FileNotFoundError(f"screen7k dense dataset does not exist: {dense_dataset}")
    args.dense_data_path = dense_dataset
    if args.dense_regul == "none":
        args.dense_regul = "strong"
    print(f"[INFO] screen7k manifest enabled dense supervision: {dense_dataset}")
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Scene arguments
    parser.add_argument('-s', '--mast3r_scene', type=str, required=True)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    parser.add_argument('--white_background', action='store_true')
    
    # For dense RGB and depth supervision from a COLMAP dataset (optional)
    parser.add_argument("--dense_data_path", type=str, default=None)
    parser.add_argument('--depthanythingv2_checkpoint_dir', type=str, default='./Depth-Anything-V2/checkpoints/')
    parser.add_argument('--depthanything_encoder', type=str, default='vitl')
    parser.add_argument('--dense_regul', type=str, default='default', help='Dense depth schedule: default, strong, strong_decay, weak, or none.')
    parser.add_argument('--dense_depth_cache', type=str, default=None)
    parser.add_argument('--data_device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--resolution', type=int, default=-1)
    
    # Config
    parser.add_argument('-c', '--config', type=str, default='default')
    parser.add_argument('--iterations', type=int, default=None)
    parser.add_argument(
        '--continue-opacity-resets-after-densify',
        action='store_true',
        help=(
            'Experimental causal-control switch: preserve the configured opacity '
            'reset cadence after densification ends.'
        ),
    )
    parser.add_argument(
        '--opacity-cull', '--opacity_cull',
        dest='opacity_cull',
        type=float,
        default=None,
        help=(
            'Optional 2DGS opacity pruning threshold.  Leave unset to retain '
            'the model default; set explicitly for a strict protocol ablation.'
        ),
    )
    parser.add_argument('--non-position-lr-decay-from', type=int, default=-1)
    parser.add_argument('--non-position-lr-final-mult', type=float, default=1.0)

    parser.add_argument('--refine_depth_path', type=str, default=None, help='Path to the refine depth directory')
    parser.add_argument('--use_downsample_gaussians', action='store_true', help='Use downsample gaussians')
    parser.add_argument('--downsample_gaussians_type', type=str, default='warp', choices=['warp', 'voxel'],
                        help='Downsample method used when --use_downsample_gaussians is set')
    parser.add_argument('--warp_depth_error_thresh', type=float, default=0.01,
                        help='Relative depth error threshold for warp-based Gaussian downsample')
    parser.add_argument('--warp_downsample_pixel_grid_size', type=int, default=-1,
                        help='Pixel grid stride for warp-based Gaussian initialization')
    parser.add_argument('--downweight_input_view_color_loss', action='store_true',
                        help='Also reduce color loss weight for input views; See3D views are always reduced')
    parser.add_argument('--cambridge_mask_pickle', type=str, default=None)
    parser.add_argument('--cambridge_mask_dataset_path', type=str, default=None)
    parser.add_argument('--cambridge_mask_indices', nargs='*', type=int, default=None)
    parser.add_argument('--cambridge_geometry_mask_pickle', type=str, default=None)
    parser.add_argument('--cambridge_geometry_mask_dataset_path', type=str, default=None)
    parser.add_argument('--cambridge_geometry_mask_indices', nargs='*', type=int, default=None)
    parser.add_argument('--cambridge_alpha_mask_indices', nargs='*', type=int, default=None)
    parser.add_argument('--semantic_alpha_weight', type=float, default=0.0)
    parser.add_argument('--cambridge_tree_mask_pickle', type=str, default=None)
    parser.add_argument('--cambridge_tree_mask_dataset_path', type=str, default=None)
    parser.add_argument('--cambridge_tree_mask_index', type=int, default=3)
    parser.add_argument('--cambridge_tree_support_dir', type=str, default=None)
    parser.add_argument('--tree_rgb_floor', type=float, default=0.25)
    parser.add_argument('--tree_rgb_support_gain', type=float, default=0.50)
    parser.add_argument('--tree_geometry_floor', type=float, default=0.05)
    parser.add_argument('--tree_geometry_support_gain', type=float, default=0.25)
    parser.add_argument('--tree_planar_weight', type=float, default=0.0)
    parser.add_argument('--tree_sky_feather', type=int, default=4)
    parser.add_argument('--tree_boundary_feather', type=int, default=6)
    parser.add_argument('--rgb_loss_type', choices=['l1', 'charbonnier'], default='l1')
    parser.add_argument('--rgb_charbonnier_eps', type=float, default=1e-3)
    parser.add_argument(
        '--rgb-supervision-profile',
        choices=['g4_tree_masked', 'full_rgb', 'ulfloc_legacy'],
        default='g4_tree_masked',
        help=(
            'RGB pixel protocol; full_rgb changes RGB pixels only to all-pixel '
            'supervision, while ulfloc_legacy is an explicit ULF-Loc protocol ablation.'
        ),
    )
    parser.add_argument(
        '--rgb-sampling-policy',
        choices=['legacy_interleaved', 'all_train_importance'],
        default='all_train_importance',
        help='Keep legacy Chart RGB oversampling or importance-correct it to all real train cameras.',
    )
    parser.add_argument(
        '--chart-geometry-sampling-policy',
        choices=['legacy_all_input', 'active_only'],
        default='active_only',
        help='Sample all aligned Charts only for a legacy ablation, or only gate/quality-active Charts.',
    )
    parser.add_argument(
        '--densification-view-policy',
        choices=['legacy_current', 'dense_only'],
        default='legacy_current',
        help=(
            'Choose whether Chart geometry views also feed 2DGS split/prune statistics, '
            'or reserve topology statistics for the dense all-real camera stream.'
        ),
    )
    parser.add_argument(
        '--chart-geometry-prior-weight',
        type=float,
        default=1.0,
        help=(
            'Common multiplier for Chart-exclusive geometric priors; zero keeps '
            'the same Chart schedule while disabling only those priors.'
        ),
    )
    parser.add_argument('--use_color_correction', action='store_true')
    parser.add_argument('--color_correction_lr', type=float, default=1e-3)
    parser.add_argument('--color_correction_reg', type=float, default=1e-2)
    parser.add_argument('--geometry_view_every_n_iter', type=int, default=5)
    parser.add_argument('--dense_only_from_iter', type=int, default=3000)
    parser.add_argument('--pseudo_rgb_weight', type=float, default=0.01)
    parser.add_argument('--pseudo_geometry_weight', type=float, default=0.25)
    parser.add_argument('--pseudo_geometry_final_weight', type=float, default=0.02)
    parser.add_argument('--pseudo_geometry_decay_until', type=int, default=7000)
    parser.add_argument(
        '--pseudo_initialization_mode',
        choices=['all', 'inpaint_only', 'none'],
        default='all',
    )
    parser.add_argument(
        '--pseudo_geometry_mask_mode',
        choices=['all', 'inpaint_only', 'none'],
        default='all',
    )
    parser.add_argument('--max_plane_abs_depth', type=float, default=50.0)
    parser.add_argument('--init_fill_unsupported_with_prior', action='store_true')
    parser.add_argument('--init_ply', type=str, default=None)
    parser.add_argument('--freeze_init_ply', action='store_true')
    parser.add_argument('--warmstart_reseed_pixel_stride', type=int, default=8)
    parser.add_argument('--warmstart_reseed_max_scale', type=float, default=0.05)
    parser.add_argument('--warmstart_max_opacity', type=float, default=1.0)
    parser.add_argument('--warmstart_max_scale', type=float, default=0.0)
    parser.add_argument('--warmstart_max_position_delta', type=float, default=0.0)
    parser.add_argument('--warmstart_diffuse_only', action='store_true')
    parser.add_argument('--warmstart_clamp_dc', action='store_true')
    
    args = parser.parse_args()
    args = apply_screen_manifest_dense_defaults(args)
    
    # Set console
    CONSOLE = Console(width=120)
    
    # Set output path
    if args.output_path is None:
        if args.mast3r_scene.endswith(os.sep):
            output_dir_name = args.mast3r_scene.split(os.sep)[-2]
        else:
            output_dir_name = args.mast3r_scene.split(os.sep)[-1]
        args.output_path = os.path.join('output', output_dir_name)
        args.output_path = os.path.join(args.output_path, 'refined_free_gaussians')
    os.makedirs(args.output_path, exist_ok=True)
    CONSOLE.print(f"[INFO] Refined free gaussians will be saved to: {args.output_path}")
    
    # Load config
    config_path = os.path.join('configs/free_gaussians_refinement', args.config + '.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Define command
    if args.refine_depth_path is not None:
        print(f'refine depth path {args.refine_depth_path}, train gs use refine depth')
        command = [
            sys.executable, "2d-gaussian-splatting/train_with_refine_depth.py",
            "-s", args.mast3r_scene,
            "-m", args.output_path,
            "--iterations", str(args.iterations or config['iterations']),
            "--non_position_lr_decay_from", str(args.non_position_lr_decay_from),
            "--non_position_lr_final_mult", str(args.non_position_lr_final_mult),
            "--data_device", args.data_device,
            "--resolution", str(args.resolution),
            "--white_background" if args.white_background else "",
            "--densify_until_iter", str(config['densify_until_iter']),
            "--opacity_reset_interval", str(config['opacity_reset_interval']),
            "--continue-opacity-resets-after-densify"
            if args.continue_opacity_resets_after_densify else "",
            "--depth_ratio", str(config['depth_ratio']),
            "--use_mip_filter" if config['use_mip_filter'] else "",
            "--normal_consistency_from", str(config['normal_consistency_from']),
            "--distortion_from", str(config['distortion_from']),
            "--depthanythingv2_checkpoint_dir", args.depthanythingv2_checkpoint_dir,
            "--depthanything_encoder", args.depthanything_encoder,
            "--dense_regul", args.dense_regul,
            "--refine_depth_path", args.refine_depth_path,
            "--use_downsample_gaussians" if args.use_downsample_gaussians else "",
            "--downsample_gaussians_type", args.downsample_gaussians_type,
            "--warp_depth_error_thresh", str(args.warp_depth_error_thresh),
            "--warp_downsample_pixel_grid_size", str(args.warp_downsample_pixel_grid_size),
            "--downweight_input_view_color_loss" if args.downweight_input_view_color_loss else "",
            "--rgb_loss_type", args.rgb_loss_type,
            "--rgb_charbonnier_eps", str(args.rgb_charbonnier_eps),
            "--rgb-supervision-profile", args.rgb_supervision_profile,
            "--rgb-sampling-policy", args.rgb_sampling_policy,
            "--chart-geometry-sampling-policy", args.chart_geometry_sampling_policy,
            "--densification-view-policy", args.densification_view_policy,
            "--chart-geometry-prior-weight", str(args.chart_geometry_prior_weight),
            "--use_color_correction" if args.use_color_correction else "",
            "--color_correction_lr", str(args.color_correction_lr),
            "--color_correction_reg", str(args.color_correction_reg),
            "--geometry_view_every_n_iter", str(args.geometry_view_every_n_iter),
            "--dense_only_from_iter", str(args.dense_only_from_iter),
            "--pseudo_rgb_weight", str(args.pseudo_rgb_weight),
            "--pseudo_geometry_weight", str(args.pseudo_geometry_weight),
            "--pseudo_geometry_final_weight", str(args.pseudo_geometry_final_weight),
            "--pseudo_geometry_decay_until", str(args.pseudo_geometry_decay_until),
            "--pseudo_initialization_mode", args.pseudo_initialization_mode,
            "--pseudo_geometry_mask_mode", args.pseudo_geometry_mask_mode,
            "--max_plane_abs_depth", str(args.max_plane_abs_depth),
            "--init_fill_unsupported_with_prior" if args.init_fill_unsupported_with_prior else "",
            "--semantic_alpha_weight", str(args.semantic_alpha_weight),
            "--init_ply" if args.init_ply else "",
            args.init_ply or "",
            "--freeze_init_ply" if args.freeze_init_ply else "",
            "--warmstart_reseed_pixel_stride", str(args.warmstart_reseed_pixel_stride),
            "--warmstart_reseed_max_scale", str(args.warmstart_reseed_max_scale),
            "--warmstart_max_opacity", str(args.warmstart_max_opacity),
            "--warmstart_max_scale", str(args.warmstart_max_scale),
            "--warmstart_max_position_delta", str(args.warmstart_max_position_delta),
            "--warmstart_diffuse_only" if args.warmstart_diffuse_only else "",
            "--warmstart_clamp_dc" if args.warmstart_clamp_dc else "",
        ]
        if args.opacity_cull is not None:
            command.extend(["--opacity_cull", str(args.opacity_cull)])
        if args.dense_data_path is not None:
            command.extend(["--dense_data_path", args.dense_data_path])
        if args.dense_depth_cache is not None:
            command.extend(["--dense_depth_cache", args.dense_depth_cache])
        if args.cambridge_mask_pickle is not None:
            command.extend(["--cambridge_mask_pickle", args.cambridge_mask_pickle])
        if args.cambridge_mask_dataset_path is not None:
            command.extend(["--cambridge_mask_dataset_path", args.cambridge_mask_dataset_path])
        if args.cambridge_mask_indices is not None:
            command.append("--cambridge_mask_indices")
            command.extend(str(index) for index in args.cambridge_mask_indices)
        if args.cambridge_geometry_mask_pickle is not None:
            command.extend(["--cambridge_geometry_mask_pickle", args.cambridge_geometry_mask_pickle])
        if args.cambridge_geometry_mask_dataset_path is not None:
            command.extend([
                "--cambridge_geometry_mask_dataset_path",
                args.cambridge_geometry_mask_dataset_path,
            ])
        if args.cambridge_geometry_mask_indices is not None:
            command.append("--cambridge_geometry_mask_indices")
            command.extend(str(index) for index in args.cambridge_geometry_mask_indices)
        if args.cambridge_alpha_mask_indices is not None:
            command.append("--cambridge_alpha_mask_indices")
            command.extend(str(index) for index in args.cambridge_alpha_mask_indices)
        if args.cambridge_tree_mask_pickle is not None:
            command.extend(["--cambridge_tree_mask_pickle", args.cambridge_tree_mask_pickle])
            command.extend([
                "--cambridge_tree_mask_dataset_path",
                args.cambridge_tree_mask_dataset_path or args.cambridge_mask_dataset_path,
                "--cambridge_tree_mask_index", str(args.cambridge_tree_mask_index),
                "--tree_rgb_floor", str(args.tree_rgb_floor),
                "--tree_rgb_support_gain", str(args.tree_rgb_support_gain),
                "--tree_geometry_floor", str(args.tree_geometry_floor),
                "--tree_geometry_support_gain", str(args.tree_geometry_support_gain),
                "--tree_planar_weight", str(args.tree_planar_weight),
                "--tree_sky_feather", str(args.tree_sky_feather),
                "--tree_boundary_feather", str(args.tree_boundary_feather),
            ])
            if args.cambridge_tree_support_dir is not None:
                command.extend(["--cambridge_tree_support_dir", args.cambridge_tree_support_dir])
        command = [argument for argument in command if argument]
    else:
        raise ValueError('refine depth path is required')

    manifest_path = write_refinement_manifest(args, config, command)
    CONSOLE.print(f"[INFO] Wrote refinement manifest: {manifest_path}")
    
    # Run command
    CONSOLE.print(f"[INFO] Running command:\n{' '.join(command)}")
    subprocess.run(command, check=True)
