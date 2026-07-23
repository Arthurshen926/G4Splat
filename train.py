import os
import sys
import argparse
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import shutil
import shlex
import numpy as np

from matcha.cambridge_training import preliminary_uses_dense_supervision


def prefer_conda_runtime_libraries():
    """Let compiled CUDA extensions use the C++ runtime they were built against."""
    conda_lib = os.path.join(sys.prefix, "lib")
    current = os.environ.get("LD_LIBRARY_PATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if conda_lib not in entries:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([conda_lib, *entries])

def run_command_safe(command):
    if command.startswith("python "):
        command = f"{shlex.quote(sys.executable)} {command[len('python '):]}"
    print(f"Running command: {command}")
    exit_code = os.system(command)
    if exit_code != 0:
        print("Command failed!")
        sys.exit(1)
    else:
        print("Command succeeded!")

if __name__ == '__main__':
    prefer_conda_runtime_libraries()
    parser = argparse.ArgumentParser()
    
    # Scene arguments
    parser.add_argument('-s', '--source_path', type=str, required=True, help='Path to the source directory')
    parser.add_argument('-o', '--output_path', type=str, default=None, help='Path to the output directory')
    
    # Image selection parameters
    parser.add_argument('--n_images', type=int, default=None, 
        help='Number of images to use for optimization, sampled with constant spacing. If not provided, all images will be used.')
    parser.add_argument('--use_view_config', action='store_true', 
        help='Use view config file to select images for optimization. If provided, this will override the --n_images and --image_idx arguments.')
    parser.add_argument('--config_view_num', type=int, default=10, 
        help='View number of the config file. If provided, this will override the --n_images.')
    parser.add_argument('--image_idx', type=int, nargs='*', default=None, 
        help='View indices to use for optimization (zero-based indexing). If provided, this will override the --n_images.')
    parser.add_argument('--randomize_images', action='store_true', 
        help='Shuffle training images before sampling with constant spacing. If image_idx is provided, this will be ignored.')
    
    # Dense supervision (Optional)
    parser.add_argument('--dense_supervision', action='store_true', 
        help='Use dense RGB supervision with a COLMAP dataset. Should only be used with --sfm_config posed.')
    parser.add_argument('--dense_data_path', type=str, default=None,
        help='Posed COLMAP dataset used for real dense RGB/depth supervision. Defaults to source_path.')
    parser.add_argument('--dense_regul', type=str, default='default', help='Dense depth schedule: default, strong, strong_decay, weak, or none.')
    parser.add_argument('--dense_depth_cache', type=str, default=None)
    parser.add_argument(
        '--dense-view-sampling-policy',
        choices=['uniform', 'spatial_block_balanced'],
        default='uniform',
        help='Sampling policy for all-real dense cameras during Gaussian refinement.',
    )
    parser.add_argument('--dense-view-block-bins', type=int, default=4)
    parser.add_argument(
        '--dense_final_only',
        action='store_true',
        help=(
            'Use chart/See3D supervision for preliminary refinements and enable the full '
            'dense dataset only for the final refinement.'
        ),
    )
    
    # Output mesh parameters
    parser.add_argument('--use_multires_tsdf', action='store_true', help='Use multi-resolution TSDF fusion instead of adaptive tetrahedralization for mesh extraction (not recommended).')
    parser.add_argument('--no_interpolated_views', action='store_true', help='Disable interpolated views for mesh extraction.')
    
    # SfM config
    parser.add_argument('--sfm_config', type=str, default='unposed', help='Config for SfM. Should be "unposed" or "posed".')
    parser.add_argument(
        '--strict_calibrated_poses',
        action='store_true',
        help=(
            'Keep calibrated camera translation and scale fixed during MASt3R '
            'global alignment instead of using the legacy post-alignment restore mode.'
        ),
    )
    parser.add_argument(
        '--per_view_calibrated_intrinsics',
        action='store_true',
        help=(
            'Keep calibrated focal/principal-point values per Chart camera during '
            'MASt3R alignment rather than using the historical shared intrinsics.'
        ),
    )
    parser.add_argument(
        '--mast3r-sparse-export-stride',
        type=int,
        default=1,
        help=(
            'Regular pixel stride for MASt3R diagnostic sparse export only; '
            'full pointmaps used by Chart alignment are never downsampled.'
        ),
    )
    
    # Chart alignment config
    parser.add_argument('--alignment_config', type=str, default='default', help='Config for charts alignment')
    parser.add_argument(
        '--chart-alignment-seed',
        type=int,
        default=0,
        help='Fixed seed for Chart alignment; use the same value for paired front-end ablations.',
    )
    parser.add_argument('--depth_model', type=str, default="depthanythingv2")
    parser.add_argument('--depthanythingv2_checkpoint_dir', type=str, default='./Depth-Anything-V2/checkpoints/')
    parser.add_argument('--depthanything_encoder', type=str, default='vitl')
    parser.add_argument(
        '--gate_aligned_chart_conflicts',
        action='store_true',
        help='Disable chart geometry with extreme aligned-depth/prior disagreement.',
    )
    parser.add_argument('--aligned_chart_max_relative_p90', type=float, default=0.50)
    parser.add_argument('--aligned_chart_max_gt25_fraction', type=float, default=0.50)
    parser.add_argument('--aligned_chart_min_valid_fraction', type=float, default=0.05)
    
    # Free Gaussians config
    parser.add_argument('--free_gaussians_config', type=str, default=None, 
        help=(
            'Config for preliminary Free Gaussians refinement. Defaults to "default"; '
            'the final dense pass is controlled separately.'
        )
    )
    parser.add_argument('--final_free_gaussians_config', type=str, default=None,
        help='Final refinement config after the last See3D stage. Defaults to long with dense supervision.')
    parser.add_argument(
        '--final_free_gaussians_iterations',
        type=int,
        default=None,
        help='Override the final refinement length without changing the screen schedule.',
    )
    parser.add_argument(
        '--final_non_position_lr_decay_from',
        type=int,
        default=-1,
        help='Start decaying non-xyz learning rates at this final-stage iteration.',
    )
    parser.add_argument(
        '--final_non_position_lr_final_mult',
        type=float,
        default=1.0,
        help='Final multiplier for non-xyz learning rates after the decay begins.',
    )
    parser.add_argument('--data_device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--white_background', action='store_true')
    parser.add_argument('--cambridge_mask_pickle', type=str, default=None)
    parser.add_argument('--cambridge_mask_dataset_path', type=str, default=None)
    parser.add_argument('--cambridge_mask_indices', type=int, nargs='*', default=[0, 1, 2])
    parser.add_argument('--cambridge_geometry_mask_indices', type=int, nargs='*', default=[0, 1, 2])
    parser.add_argument('--cambridge_alpha_mask_indices', type=int, nargs='*', default=None)
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
    parser.add_argument(
        '--tree-missing-support-policy',
        choices=['error', 'neutral', 'legacy_zero'],
        default='legacy_zero',
        help='How a dense real view without a tree support map is interpreted.',
    )
    parser.add_argument('--tree-neutral-support-value', type=float, default=0.5)
    parser.add_argument(
        '--cambridge-task-semantic-policy',
        choices=['legacy', 'outdoor_task_specific_v1'],
        default='legacy',
    )
    parser.add_argument('--cambridge-task-semantic-manifest', type=str, default=None)
    parser.add_argument('--rgb_loss_type', choices=['l1', 'charbonnier'], default='l1')
    parser.add_argument(
        '--rgb-supervision-profile',
        choices=['g4_tree_masked', 'full_rgb', 'ulfloc_legacy'],
        default='g4_tree_masked',
        help=(
            'RGB pixel protocol. full_rgb changes only RGB supervision to all '
            'pixels; ulfloc_legacy changes only RGB supervision to the standard '
            'ULF-Loc object/distortion/sky ordering.'
        ),
    )
    parser.add_argument(
        '--rgb-sampling-policy',
        choices=['legacy_interleaved', 'all_train_importance'],
        default='all_train_importance',
        help=(
            'Keep legacy Chart RGB oversampling or importance-correct RGB loss '
            'to the uniform all-real-train camera objective.'
        ),
    )
    parser.add_argument(
        '--chart-geometry-sampling-policy',
        choices=['legacy_all_input', 'active_only'],
        default='active_only',
        help=(
            'Sample every aligned Chart only for a legacy ablation, or restrict '
            'geometry iterations to gate/quality-active Charts.'
        ),
    )
    parser.add_argument(
        '--densification-view-policy',
        choices=['legacy_current', 'dense_only'],
        default='legacy_current',
        help=(
            'Whether Chart geometry renders also update 2DGS split/prune statistics, '
            'or whether topology allocation is restricted to all-real dense views.'
        ),
    )
    parser.add_argument('--use_color_correction', action='store_true')
    parser.add_argument('--color_correction_lr', type=float, default=1e-3)
    parser.add_argument('--color_correction_reg', type=float, default=1e-2)
    
    # Multi-resolution TSDF config
    parser.add_argument('--tsdf_config', type=str, default='default', help='Config for multi-resolution TSDF fusion')
    
    # Tetrahedralization config
    parser.add_argument('--tetra_config', type=str, default='default', help='Config for adaptive tetrahedralization')
    parser.add_argument('--tetra_downsample_ratio', type=float, default=0.5, 
        help='Downsample ratio for tetrahedralization. We recommend starting with 0.5 and then decreasing to 0.25 '\
        'if the mesh is too dense, or increasing to 1.0 if the mesh is too sparse.'
    )

    # G4Splat config
    parser.add_argument('--select_inpaint_num', type=int, default=20, help='Number of views to select for inpainting.')
    parser.add_argument(
        '--see3d_iteration',
        type=int,
        default=7000,
        help='Gaussian iteration used to render See3D conditioning views.',
    )
    parser.add_argument(
        '--scene_aligned_see3d_cameras',
        action='store_true',
        help='Use pose-graph See3D camera proposals for trajectory-style scenes.',
    )
    parser.add_argument(
        '--preserve_visible_see3d_render',
        action='store_true',
        help='Keep the current GS render in observed pseudo-view pixels.',
    )
    parser.add_argument(
        '--skip_plane_see3d_views',
        action='store_true',
        help='Do not use globally refined planes to propose See3D cameras.',
    )
    parser.add_argument(
        '--disable_see3d',
        action='store_true',
        help='Skip all See3D stages and run the final refinement from real-view plane priors only.',
    )
    parser.add_argument(
        '--quality_filter_see3d',
        action='store_true',
        help='Reject high-synthesis, blurry, or depth-inconsistent pseudo-views.',
    )
    parser.add_argument(
        '--skip_see3d_plane_extraction',
        action='store_true',
        help='Skip SAM plane extraction in conservative warm-start experiments.',
    )
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
    parser.add_argument('--pseudo_rgb_weight', type=float, default=0.01)
    parser.add_argument('--pseudo_geometry_weight', type=float, default=0.25)
    parser.add_argument('--pseudo_geometry_final_weight', type=float, default=0.02)
    parser.add_argument('--pseudo_geometry_decay_until', type=int, default=7000)
    parser.add_argument(
        '--chart-geometry-prior-weight',
        '--chart_geometry_prior_weight',
        dest='chart_geometry_prior_weight',
        type=float,
        default=1.0,
        help=(
            'Multiplier for Chart-exclusive geometry priors in a strict causal '
            'ablation. The underscore spelling remains a backward-compatible alias.'
        ),
    )
    parser.add_argument(
        '--artifact_mask_see3d',
        action='store_true',
        help='Augment See3D visibility masks with conservative no-reference artifact masks.',
    )
    parser.add_argument('--artifact_detector_repo', type=str, default='/root/STDLoc')
    parser.add_argument('--artifact_max_edit_fraction', type=float, default=0.35)
    parser.add_argument('--use_downsample_gaussians', action='store_true', help='Use downsample gaussians for training')
    parser.add_argument('--downsample_gaussians_type', type=str, default='warp', choices=['warp', 'voxel'],
        help='Downsample method used when --use_downsample_gaussians is set')
    parser.add_argument('--warp_depth_error_thresh', type=float, default=0.01,
        help='Relative depth error threshold for warp-based Gaussian downsample')
    parser.add_argument('--warp_downsample_pixel_grid_size', type=int, default=-1,
        help='Pixel grid stride for warp-based Gaussian initialization')
    parser.add_argument(
        '--gate_plane_refinement',
        action='store_true',
        help='Fall back to aligned chart depth when plane support or depth changes are unsafe.',
    )
    parser.add_argument('--plane_gate_min_support_fraction', type=float, default=0.50)
    parser.add_argument('--plane_gate_max_relative_p90', type=float, default=0.25)
    parser.add_argument('--plane_gate_max_gt25_fraction', type=float, default=0.25)
    parser.add_argument('--plane_gate_max_pixel_relative_change', type=float, default=0.50)
    parser.add_argument(
        '--max_plane_abs_depth',
        type=float,
        default=None,
        help='Optional legacy depth cap; unset for the outdoor inverse-depth mainline.',
    )
    parser.add_argument(
        '--max_chart_abs_depth',
        '--max-chart-abs-depth',
        dest='max_chart_abs_depth',
        type=float,
        default=50.0,
        help=(
            'Absolute aligned-Chart depth cap before plane construction. '
            'Set to 0 to disable the legacy cap for the outdoor inverse-depth mainline.'
        ),
    )
    parser.add_argument(
        '--max_chart_abs_point',
        '--max-chart-abs-point',
        dest='max_chart_abs_point',
        type=float,
        default=50.0,
        help=(
            'Absolute aligned-Chart point-coordinate cap before plane construction. '
            'Set to 0 to disable the legacy cap for the outdoor inverse-depth mainline.'
        ),
    )
    parser.add_argument(
        '--min_global_plane_views',
        type=int,
        default=2,
        help='Require cross-view support before a fused plane rewrites Chart depth.',
    )
    parser.add_argument(
        '--plane-min-size-ratio',
        type=float,
        default=0.01,
        help='Relative minimum plane area measured over semantic structural support.',
    )
    parser.add_argument(
        '--plane-min-pixels',
        type=int,
        default=1,
        help='Absolute minimum local plane area after semantic gating.',
    )
    parser.add_argument(
        '--plane-normal-clusters',
        type=int,
        default=6,
        help='Number of normal modes retained for local plane proposals.',
    )
    parser.add_argument('--downweight_input_view_color_loss', action='store_true',
        help='Also reduce color loss weight for input views; See3D views are always reduced')
    parser.add_argument('--use_mesh_filter', action='store_true', help='Use mesh filter')
    parser.add_argument('--use_dense_view', action='store_true', help='Use dense view for training')                    # Add an additional input stage to extend plane-aware depth estimation across all input views
    parser.add_argument(
        '--skip_default_render_eval',
        action='store_true',
        help='Skip dense train-view export and the repository default evaluator; use an explicit heldout evaluator.',
    )
    parser.add_argument(
        '--stop_after_initial_refinement',
        action='store_true',
        help='Stop after the first refinement pass. Intended for 7k geometry screening.',
    )
    parser.add_argument(
        '--stop_after_alignment_gate',
        action='store_true',
        help='Stop after alignment and its chart gate so rejected charts can be replaced.',
    )
    parser.add_argument(
        '--stop_after_plane_refinement',
        action='store_true',
        help='Stop after real-view plane refinement so inverse-depth fusion can run before Gaussian training.',
    )
    parser.add_argument(
        '--refine_depth_path_override',
        type=str,
        default=None,
        help='Use a provenance-carrying fused depth directory instead of plane-refine-depths for Gaussian training.',
    )
    parser.add_argument(
        '--continue_after_initial_refinement',
        action='store_true',
        help=(
            'Reuse an existing mast3r_sfm/plane-refine-depths and initial 7k model in '
            'output_path, then continue with G4Splat stages and the final refinement.'
        ),
    )
    parser.add_argument(
        '--continue_after_see3d_plane_stage',
        type=int,
        choices=[1, 2, 3],
        default=None,
        help=(
            'Resume after the selected See3D generation and plane-refinement stage, '
            'before its Gaussian refinement. Intended for recovering a completed '
            'expensive generation stage after a downstream failure.'
        ),
    )
    parser.add_argument(
        '--continue_after_sfm',
        action='store_true',
        help='Reuse completed mast3r_sfm pointmaps, then rerun alignment and all later stages.',
    )
    parser.add_argument(
        '--continue_after_alignment',
        action='store_true',
        help='Reuse completed charts_data.npz, then run plane construction and later stages.',
    )
    parser.add_argument(
        '--continue_after_plane_refinement',
        action='store_true',
        help='Reuse completed chart alignment and plane-refined depths, then restart Gaussian refinement.',
    )
    args = parser.parse_args()
    if args.stop_after_initial_refinement and (
        args.continue_after_initial_refinement
        or args.continue_after_see3d_plane_stage is not None
    ):
        raise ValueError(
            '--stop_after_initial_refinement cannot be combined with a later-stage continuation'
        )
    continuation_modes = sum((
        args.continue_after_sfm,
        args.continue_after_alignment,
        args.continue_after_plane_refinement,
        args.continue_after_initial_refinement,
        args.continue_after_see3d_plane_stage is not None,
    ))
    if continuation_modes > 1:
        raise ValueError('Only one continuation mode may be selected')
    if not 0.0 < args.plane_min_size_ratio <= 1.0:
        raise ValueError('--plane-min-size-ratio must be in (0, 1]')
    if args.plane_min_pixels < 1:
        raise ValueError('--plane-min-pixels must be positive')
    if args.plane_normal_clusters < 1:
        raise ValueError('--plane-normal-clusters must be positive')
    
    # Set output paths
    if args.output_path is None:
        if args.source_path.endswith(os.sep):
            output_dir_name = args.source_path.split(os.sep)[-2]
        else:
            output_dir_name = args.source_path.split(os.sep)[-1]
        args.output_path = os.path.join('output', output_dir_name)
    mast3r_scene_path = os.path.join(args.output_path, 'mast3r_sfm')
    aligned_charts_path = os.path.join(args.output_path, 'mast3r_sfm')
    free_gaussians_path = os.path.join(args.output_path, 'free_gaussians')
    tsdf_meshes_path = os.path.join(args.output_path, 'tsdf_meshes')
    tetra_meshes_path = os.path.join(args.output_path, 'tetra_meshes')

    if args.use_dense_view:
        dense_view_json_path = os.path.join(args.source_path, 'dense_view.json')
        print(f'[INFO]: Search dense view json in {dense_view_json_path}')
        if not os.path.exists(dense_view_json_path):
            source_img_path = os.path.join(args.source_path, 'images')
            source_img_num = len(os.listdir(source_img_path))
            dense_view_idx_list = range(source_img_num)
            print(f"Use all {source_img_num} views in the source path as dense view")
            with open(dense_view_json_path, 'w') as f:
                json.dump({'train': list(dense_view_idx_list)}, f)
            print(f"Save dense view index list to {dense_view_json_path}")
        else:
            with open(dense_view_json_path, 'r') as f:
                dense_view_json = json.load(f)
            dense_view_idx_list = dense_view_json['train']
            print(f"Use {len(dense_view_idx_list)} views from {dense_view_json_path} as dense view")
    
    dense_data_path = args.dense_data_path or args.source_path
    dense_arg = ""
    if args.dense_supervision:
        dense_arg = " ".join(["--dense_data_path", dense_data_path])

    if args.free_gaussians_config is None:
        args.free_gaussians_config = 'default'
    if args.final_free_gaussians_config is None:
        args.final_free_gaussians_config = 'long' if args.dense_supervision else args.free_gaussians_config

    if args.use_view_config:
        n_images = None
        view_config_path = os.path.join(args.source_path, f'split-{args.config_view_num}views.json')
        if os.path.exists(view_config_path):
            with open(view_config_path, 'r') as f:
                view_config = json.load(f)
            image_idx_list = view_config['train']
        else:
            view_config_path = os.path.join(args.source_path, f'train_test_split_{args.config_view_num}.json')
            with open(view_config_path, 'r') as f:
                view_config = json.load(f)
            image_idx_list = view_config['train_ids']
    else:
        n_images = args.n_images
        image_idx_list = args.image_idx
    
    # Defining commands
    if args.mast3r_sparse_export_stride < 1:
        raise ValueError('--mast3r-sparse-export-stride must be at least one')
    sfm_command = " ".join([
        "python", "scripts/run_sfm.py",
        "--source_path", args.source_path,
        "--output_path", mast3r_scene_path,
        "--config", args.sfm_config,
        # "--env", args.sfm_env,
        "--n_images" if n_images is not None else "", str(n_images) if n_images is not None else "",
        "--image_idx" if image_idx_list is not None else "", " ".join([str(i) for i in image_idx_list]) if image_idx_list is not None else "",
        "--randomize_images" if args.randomize_images else "",
        "--sparse-export-stride", str(args.mast3r_sparse_export_stride),
        "--strict_calibrated_poses" if args.strict_calibrated_poses else "",
        "--per_view_calibrated_intrinsics" if args.per_view_calibrated_intrinsics else "",
    ])
    
    # Charts seed the entire geometry pipeline, so canonical trees must be
    # excluded *before* chart alignment and the aligned-depth conflict audit.
    # The later free-Gaussian stage can retain its softer tree weighting, but
    # an occluding tree pixel must never become a hard chart/plane anchor.
    alignment_mask_pickle = args.cambridge_tree_mask_pickle or args.cambridge_mask_pickle
    alignment_mask_dataset_path = (
        args.cambridge_tree_mask_dataset_path
        or args.cambridge_mask_dataset_path
        or dense_data_path
    )
    alignment_mask_indices = list(args.cambridge_geometry_mask_indices)
    if args.cambridge_tree_mask_pickle:
        alignment_mask_indices = list(
            dict.fromkeys([
                *alignment_mask_indices,
                int(args.cambridge_tree_mask_index),
            ])
        )

    align_charts_command = " ".join([
        "python", "scripts/align_charts.py",
        "--source_path", mast3r_scene_path,
        "--mast3r_scene", mast3r_scene_path,
        "--output_path", aligned_charts_path,
        "--config", args.alignment_config,
        "--seed", str(args.chart_alignment_seed),
        "--depth_model", args.depth_model,
        "--depthanythingv2_checkpoint_dir", args.depthanythingv2_checkpoint_dir,
        "--depthanything_encoder", args.depthanything_encoder,
        "--cambridge_mask_pickle" if alignment_mask_pickle else "",
        alignment_mask_pickle or "",
        "--cambridge_mask_dataset_path" if alignment_mask_pickle else "",
        alignment_mask_dataset_path if alignment_mask_pickle else "",
        "--cambridge_mask_indices" if alignment_mask_pickle else "",
        " ".join(str(index) for index in alignment_mask_indices)
        if alignment_mask_pickle else "",
    ])
    pointmap_coordinate_audit_command = None
    if args.strict_calibrated_poses:
        pointmap_coordinate_audit_command = " ".join([
            "python", "scripts/audit_mast3r_pointmaps.py",
            "--mast3r-root", mast3r_scene_path,
            "--output", os.path.join(mast3r_scene_path, "pointmap_coordinate_audit_raw"),
            "--mask-pickle" if alignment_mask_pickle else "",
            alignment_mask_pickle or "",
            "--mask-dataset-path" if alignment_mask_pickle else "",
            alignment_mask_dataset_path if alignment_mask_pickle else "",
            "--mask-indices" if alignment_mask_pickle else "",
            " ".join(str(index) for index in alignment_mask_indices)
            if alignment_mask_pickle else "",
            "--require-coordinate-contract",
            "--audit-sparse-export",
            "--require-sparse-export-coordinate-contract",
            "--require-sparse-export-track-contract",
        ])
    aligned_chart_gate_command = None
    if args.gate_aligned_chart_conflicts:
        aligned_chart_gate_command = " ".join([
            "python", "scripts/gate_aligned_charts.py",
            "--mast3r-scene", mast3r_scene_path,
            "--mask-pickle" if alignment_mask_pickle else "",
            alignment_mask_pickle or "",
            "--mask-dataset-path" if alignment_mask_pickle else "",
            alignment_mask_dataset_path if alignment_mask_pickle else "",
            "--mask-indices" if alignment_mask_pickle else "",
            " ".join(str(index) for index in alignment_mask_indices)
            if alignment_mask_pickle else "",
            "--max-relative-p90", str(args.aligned_chart_max_relative_p90),
            "--max-gt25-fraction", str(args.aligned_chart_max_gt25_fraction),
            "--min-valid-fraction", str(args.aligned_chart_min_valid_fraction),
            "--require-reference-depths",
        ])
    tree_support_command = None
    if args.cambridge_tree_mask_pickle and args.cambridge_tree_support_dir:
        tree_support_command = " ".join([
            "python", "scripts/build_tree_support_maps.py",
            "--mast3r-scene", mast3r_scene_path,
            "--mask-pickle", args.cambridge_tree_mask_pickle,
            "--mask-dataset-path",
            args.cambridge_tree_mask_dataset_path or args.cambridge_mask_dataset_path or dense_data_path,
            "--output", args.cambridge_tree_support_dir,
            "--tree-mask-index", str(args.cambridge_tree_mask_index),
        ])
    
    # NOTE: hard code plane-refine-depths path
    plane_root_path = os.path.join(mast3r_scene_path, 'plane-refine-depths')
    refine_depth_input_path = args.refine_depth_path_override or plane_root_path

    def get_refine_free_gaussians_command(
        config_name,
        use_dense_data,
        *,
        init_ply=None,
        start_checkpoint=None,
        freeze_init_ply=False,
        iterations=None,
        non_position_lr_decay_from=-1,
        non_position_lr_final_mult=1.0,
    ):
        stage_dense_arg = dense_arg if use_dense_data else ""
        stage_dense_regul = args.dense_regul if use_dense_data else "none"
        schedule_args = []
        if iterations is not None:
            schedule_args.extend(["--iterations", str(iterations)])
        if non_position_lr_decay_from >= 0 or non_position_lr_final_mult != 1.0:
            schedule_args.extend([
                "--non-position-lr-decay-from", str(non_position_lr_decay_from),
                "--non-position-lr-final-mult", str(non_position_lr_final_mult),
            ])
        return " ".join([
            "python", "scripts/refine_free_gaussians.py",
            "--mast3r_scene", mast3r_scene_path,
            "--output_path", free_gaussians_path,
            "--config", config_name,
            *schedule_args,
            "--data_device", args.data_device,
            "--resolution", str(args.resolution),
            "--white_background" if args.white_background else "",
            stage_dense_arg,
            "--dense_regul", stage_dense_regul,
            "--dense-view-sampling-policy", args.dense_view_sampling_policy,
            "--dense-view-block-bins", str(args.dense_view_block_bins),
            "--depthanythingv2_checkpoint_dir", args.depthanythingv2_checkpoint_dir,
            "--depthanything_encoder", args.depthanything_encoder,
            "--dense_depth_cache" if use_dense_data and args.dense_depth_cache else "",
            args.dense_depth_cache if use_dense_data and args.dense_depth_cache else "",
            "--refine_depth_path", refine_depth_input_path,
            "--use_downsample_gaussians" if args.use_downsample_gaussians else "",
            "--downsample_gaussians_type", args.downsample_gaussians_type,
            "--warp_depth_error_thresh", str(args.warp_depth_error_thresh),
            "--warp_downsample_pixel_grid_size", str(args.warp_downsample_pixel_grid_size),
            "--downweight_input_view_color_loss" if args.downweight_input_view_color_loss else "",
            "--cambridge_mask_pickle" if args.cambridge_mask_pickle else "",
            args.cambridge_mask_pickle or "",
            "--cambridge_mask_dataset_path" if args.cambridge_mask_pickle else "",
            (args.cambridge_mask_dataset_path or dense_data_path) if args.cambridge_mask_pickle else "",
            "--cambridge_mask_indices" if args.cambridge_mask_pickle else "",
            " ".join(str(index) for index in args.cambridge_mask_indices)
            if args.cambridge_mask_pickle else "",
            "--cambridge_geometry_mask_pickle" if args.cambridge_mask_pickle else "",
            args.cambridge_mask_pickle or "",
            "--cambridge_geometry_mask_dataset_path" if args.cambridge_mask_pickle else "",
            (args.cambridge_mask_dataset_path or dense_data_path) if args.cambridge_mask_pickle else "",
            "--cambridge_geometry_mask_indices" if args.cambridge_mask_pickle else "",
            " ".join(str(index) for index in args.cambridge_geometry_mask_indices)
            if args.cambridge_mask_pickle else "",
            "--cambridge_alpha_mask_indices"
            if args.cambridge_mask_pickle and args.cambridge_alpha_mask_indices else "",
            " ".join(str(index) for index in args.cambridge_alpha_mask_indices)
            if args.cambridge_mask_pickle and args.cambridge_alpha_mask_indices else "",
            "--semantic_alpha_weight", str(args.semantic_alpha_weight),
            "--cambridge_tree_mask_pickle" if args.cambridge_tree_mask_pickle else "",
            args.cambridge_tree_mask_pickle or "",
            "--cambridge_tree_mask_dataset_path" if args.cambridge_tree_mask_pickle else "",
            (args.cambridge_tree_mask_dataset_path or args.cambridge_mask_dataset_path or dense_data_path)
            if args.cambridge_tree_mask_pickle else "",
            "--cambridge_tree_mask_index" if args.cambridge_tree_mask_pickle else "",
            str(args.cambridge_tree_mask_index) if args.cambridge_tree_mask_pickle else "",
            "--cambridge_tree_support_dir" if args.cambridge_tree_support_dir else "",
            args.cambridge_tree_support_dir or "",
            "--tree_rgb_floor", str(args.tree_rgb_floor),
            "--tree_rgb_support_gain", str(args.tree_rgb_support_gain),
            "--tree_geometry_floor", str(args.tree_geometry_floor),
            "--tree_geometry_support_gain", str(args.tree_geometry_support_gain),
            "--tree_planar_weight", str(args.tree_planar_weight),
            "--tree_sky_feather", str(args.tree_sky_feather),
            "--tree_boundary_feather", str(args.tree_boundary_feather),
            "--tree-missing-support-policy", args.tree_missing_support_policy,
            "--tree-neutral-support-value", str(args.tree_neutral_support_value),
            "--cambridge-task-semantic-policy", args.cambridge_task_semantic_policy,
            "--cambridge-task-semantic-manifest" if args.cambridge_task_semantic_manifest else "",
            args.cambridge_task_semantic_manifest or "",
            "--rgb_loss_type", args.rgb_loss_type,
            "--rgb-supervision-profile", args.rgb_supervision_profile,
            "--rgb-sampling-policy", args.rgb_sampling_policy,
            "--chart-geometry-sampling-policy", args.chart_geometry_sampling_policy,
            "--densification-view-policy", args.densification_view_policy,
            "--chart-geometry-prior-weight", str(args.chart_geometry_prior_weight),
            "--use_color_correction" if args.use_color_correction else "",
            "--color_correction_lr", str(args.color_correction_lr),
            "--color_correction_reg", str(args.color_correction_reg),
            "--pseudo_initialization_mode", args.pseudo_initialization_mode,
            "--pseudo_geometry_mask_mode", args.pseudo_geometry_mask_mode,
            "--pseudo_rgb_weight", str(args.pseudo_rgb_weight),
            "--pseudo_geometry_weight", str(args.pseudo_geometry_weight),
            "--pseudo_geometry_final_weight", str(args.pseudo_geometry_final_weight),
            "--pseudo_geometry_decay_until", str(args.pseudo_geometry_decay_until),
            "--max_plane_abs_depth" if args.max_plane_abs_depth is not None else "",
            str(args.max_plane_abs_depth) if args.max_plane_abs_depth is not None else "",
            "--init_ply" if init_ply else "",
            init_ply or "",
            "--start-checkpoint" if start_checkpoint else "",
            start_checkpoint or "",
            "--freeze_init_ply" if freeze_init_ply else "",
        ])

    render_all_img_command = " ".join([
        "python", "2d-gaussian-splatting/render_multires.py",
        "--source_path", mast3r_scene_path,
        "--model_path", free_gaussians_path,
        "--skip_test",
        "--skip_mesh",
        "--render_all_img",
        "--use_default_output_dir",
    ])
    
    tsdf_command = " ".join([
        "python", "scripts/extract_tsdf_mesh.py",
        "--mast3r_scene", mast3r_scene_path,
        "--model_path", free_gaussians_path,
        "--output_path", tsdf_meshes_path,
        "--config", args.tsdf_config,
    ])
    
    tetra_command = " ".join([
        sys.executable, "scripts/extract_tetra_mesh.py",
        "--mast3r_scene", mast3r_scene_path,
        "--model_path", free_gaussians_path,
        "--output_path", tetra_meshes_path,
        "--config", args.tetra_config,
        "--downsample_ratio", str(args.tetra_downsample_ratio),
        "--interpolate_views" if not args.no_interpolated_views else "",
        dense_arg,
    ])

    def get_see3d_inpaint_command(stage, select_inpaint_num):
        return " ".join([
        "python", "scripts/see3d_inpaint.py",
        "--source_path", mast3r_scene_path,
        "--model_path", free_gaussians_path,
        "--plane_root_dir", plane_root_path,
        "--iteration", str(args.see3d_iteration),
            "--see3d_stage", str(stage),
            "--select_inpaint_num", str(select_inpaint_num),
            "--depthanythingv2_checkpoint_dir", args.depthanythingv2_checkpoint_dir,
            "--depthanything_encoder", args.depthanything_encoder,
            "--scene_aligned_cameras" if args.scene_aligned_see3d_cameras else "",
            "--preserve_visible_render" if args.preserve_visible_see3d_render else "",
            "--skip_plane_views" if args.skip_plane_see3d_views else "",
            "--quality_filter" if args.quality_filter_see3d else "",
            "--skip_plane_extraction" if args.skip_see3d_plane_extraction else "",
            "--artifact_mask_detector" if args.artifact_mask_see3d else "",
            "--artifact_detector_repo" if args.artifact_mask_see3d else "",
            args.artifact_detector_repo if args.artifact_mask_see3d else "",
            "--artifact_max_edit_fraction" if args.artifact_mask_see3d else "",
            str(args.artifact_max_edit_fraction) if args.artifact_mask_see3d else "",
        ])

    eval_command = " ".join([
        "python", "2d-gaussian-splatting/eval/eval.py",
        "--source_path", args.source_path,
        "--model_path", args.output_path,
        "--sparse_view_num", str(args.config_view_num),
    ])

    render_charts_command = " ".join([
        "python", "2d-gaussian-splatting/render_chart_views.py",
        "--source_path", mast3r_scene_path,
        "--save_root_path", plane_root_path,
        "--data_device", args.data_device,
        "--resolution", str(args.resolution),
        "--max_chart_abs_depth", str(args.max_chart_abs_depth),
        "--max_chart_abs_point", str(args.max_chart_abs_point),
        "--cambridge_mask_pickle" if alignment_mask_pickle else "",
        alignment_mask_pickle or "",
        "--cambridge_mask_dataset_path" if alignment_mask_pickle else "",
        alignment_mask_dataset_path if alignment_mask_pickle else "",
        "--cambridge_geometry_mask_indices" if alignment_mask_pickle else "",
        " ".join(str(index) for index in alignment_mask_indices)
        if alignment_mask_pickle else "",
    ])

    generate_2Dplane_command = " ".join([
        "python", "2d-gaussian-splatting/planes/plane_excavator.py",
        "--plane_root_path", plane_root_path,
        "--min-size-ratio", str(args.plane_min_size_ratio),
        "--min-plane-pixels", str(args.plane_min_pixels),
        "--normal-clusters", str(args.plane_normal_clusters),
    ])

    plane_safety_gate_command = " ".join([
        "python", "scripts/gate_plane_refinement.py",
        "--plane-root", plane_root_path,
        "--min-support-fraction", str(args.plane_gate_min_support_fraction),
        "--max-relative-p90", str(args.plane_gate_max_relative_p90),
        "--max-gt25-fraction", str(args.plane_gate_max_gt25_fraction),
        "--max-pixel-relative-change", str(args.plane_gate_max_pixel_relative_change),
    ])

    pnts_path = os.path.join(mast3r_scene_path, 'chart_pcd.ply')
    vis_plane_path = os.path.join(mast3r_scene_path, 'vis_plane')

    def get_plane_refine_depth_command(anchor_view_id_json_path=None, see3d_root_path=None):
        camera_args = [
            "--resolution", str(args.resolution),
            "--data_device", args.data_device,
            "--min_global_plane_views", str(args.min_global_plane_views),
        ]
        if see3d_root_path is not None:
            if anchor_view_id_json_path is not None:
                return " ".join([
                    "python", "scripts/plane_refine_depth.py",
                    "--source_path", mast3r_scene_path,
                    "--plane_root_path", plane_root_path,
                    "--pnts_path", pnts_path,
                    "--anchor_view_id_json_path", anchor_view_id_json_path,
                    "--see3d_root_path", see3d_root_path,
                    *camera_args,
                ])
            else:
                return " ".join([
                    "python", "scripts/plane_refine_depth.py",
                    "--source_path", mast3r_scene_path,
                    "--plane_root_path", plane_root_path,
                    "--pnts_path", pnts_path,
                    "--see3d_root_path", see3d_root_path,
                    *camera_args,
                ])
        else:
            return " ".join([
                "python", "scripts/plane_refine_depth.py",
                "--source_path", mast3r_scene_path,
                "--plane_root_path", plane_root_path,
                "--pnts_path", pnts_path,
                *camera_args,
            ])
        
    see3d_root_path = os.path.join(mast3r_scene_path, 'see3d_render')

    render_eval_path = os.path.join(free_gaussians_path, 'train', 'ours_7000', 'renders')

    t1 = time.time()
    
    preliminary_uses_dense = preliminary_uses_dense_supervision(
        dense_supervision=args.dense_supervision,
        dense_final_only=args.dense_final_only,
    )
    if (
        args.continue_after_initial_refinement
        or args.continue_after_see3d_plane_stage is not None
    ):
        required_resume_paths = [
            os.path.join(mast3r_scene_path, 'charts_data.npz'),
            os.path.join(plane_root_path, 'refine_depth_frame000000.tiff'),
            os.path.join(free_gaussians_path, 'point_cloud', 'iteration_7000', 'point_cloud.ply'),
        ]
        if args.continue_after_see3d_plane_stage is not None:
            stage = args.continue_after_see3d_plane_stage
            required_resume_paths.extend([
                os.path.join(see3d_root_path, 'see3d_cameras.npz'),
                os.path.join(see3d_root_path, 'inpainted_images'),
                os.path.join(see3d_root_path, f'stage{stage}', 'anchor_view_id.json'),
            ])
        missing_resume_paths = [path for path in required_resume_paths if not os.path.exists(path)]
        if missing_resume_paths:
            raise FileNotFoundError(
                'Cannot continue from the requested refinement stage; missing: '
                + ', '.join(missing_resume_paths)
            )
        if args.continue_after_see3d_plane_stage is not None:
            print(
                '[INFO] Reusing completed See3D/plane stage '
                f'{args.continue_after_see3d_plane_stage} artifacts.'
            )
        else:
            print('[INFO] Reusing completed MASt3R/alignment/plane/initial-refinement artifacts.')
    else:
        if args.continue_after_plane_refinement:
            required_plane_paths = [
                os.path.join(mast3r_scene_path, 'charts_data.npz'),
                os.path.join(plane_root_path, 'refine_depth_frame000000.tiff'),
                os.path.join(plane_root_path, 'global_3Dplane_ID_dict.json'),
            ]
            missing_plane_paths = [path for path in required_plane_paths if not os.path.exists(path)]
            if missing_plane_paths:
                raise FileNotFoundError(
                    'Cannot continue after plane refinement; missing: '
                    + ', '.join(missing_plane_paths)
                )
            print('[INFO] Reusing completed MASt3R/alignment/plane-refinement artifacts.')
        else:
            if args.continue_after_alignment:
                charts_data_path = os.path.join(mast3r_scene_path, 'charts_data.npz')
                if not os.path.exists(charts_data_path):
                    raise FileNotFoundError(
                        'Cannot continue after alignment; missing: ' + charts_data_path
                    )
                print('[INFO] Reusing completed MASt3R SfM and chart alignment artifacts.')
            elif args.continue_after_sfm:
                required_sfm_paths = [
                    os.path.join(mast3r_scene_path, 'sparse', '0', 'images.bin'),
                    os.path.join(mast3r_scene_path, 'pointmaps'),
                ]
                missing_sfm_paths = [path for path in required_sfm_paths if not os.path.exists(path)]
                if missing_sfm_paths:
                    raise FileNotFoundError(
                        'Cannot continue after SfM; missing: ' + ', '.join(missing_sfm_paths)
                    )
                print('[INFO] Reusing completed MASt3R SfM artifacts.')
            else:
                run_command_safe(sfm_command)
            if pointmap_coordinate_audit_command is not None:
                run_command_safe(pointmap_coordinate_audit_command)
            if not args.continue_after_alignment:
                run_command_safe(align_charts_command)
            reuse_existing_alignment_gate = False
            if args.continue_after_alignment:
                charts_data_path = os.path.join(mast3r_scene_path, 'charts_data.npz')
                with np.load(charts_data_path) as existing_charts:
                    reuse_existing_alignment_gate = (
                        'alignment_gate_valid' in existing_charts.files
                    )
                if reuse_existing_alignment_gate:
                    print(
                        '[INFO] Reusing persisted aligned-chart gate; '
                        'resume will not overwrite the audited active set.'
                    )
            if aligned_chart_gate_command is not None and not reuse_existing_alignment_gate:
                run_command_safe(aligned_chart_gate_command)
            if args.stop_after_alignment_gate:
                print('[INFO] Stopping after aligned-chart gate for feedback reselection.')
                sys.exit(0)
            if tree_support_command is not None:
                run_command_safe(tree_support_command)

            # generate 2D planes and refine depth for input views
            run_command_safe(render_charts_command)
            run_command_safe(generate_2Dplane_command)
            run_command_safe(get_plane_refine_depth_command(anchor_view_id_json_path=None, see3d_root_path=None))
            if args.gate_plane_refinement:
                run_command_safe(plane_safety_gate_command)
            if args.stop_after_plane_refinement:
                print('[INFO] Stopping after plane refinement for inverse-depth fusion.')
                sys.exit(0)

        run_command_safe(
            get_refine_free_gaussians_command(
                args.free_gaussians_config,
                preliminary_uses_dense,
            )
        )

    if args.stop_after_initial_refinement:
        print("Finished initial refinement screening pass.")
        print(f"Total running time: {time.time() - t1} seconds")
        sys.exit(0)

    if args.use_dense_view:
        # replace the sparse/0 with dense-view-sparse/0, use dense view for training
        # copy point3D files from sparse/0 to dense-view-sparse/0
        shutil.copy(f'{mast3r_scene_path}/sparse/0/points3D.bin', f'{mast3r_scene_path}/dense-view-sparse/0/points3D.bin')
        shutil.copy(f'{mast3r_scene_path}/sparse/0/points3D.txt', f'{mast3r_scene_path}/dense-view-sparse/0/points3D.txt')
        shutil.copy(f'{mast3r_scene_path}/sparse/0/points3D.ply', f'{mast3r_scene_path}/dense-view-sparse/0/points3D.ply')

        # render dense views
        render_dense_views_command = " ".join([
            "python", "2d-gaussian-splatting/render_dense_views.py",
            "--source_path", mast3r_scene_path,
            "--model_path", free_gaussians_path,
            "--iteration", "7000",
        ])
        run_command_safe(render_dense_views_command)

        # generate depth and normal for dense views
        gen_dn_dense_views_command = " ".join([
            "python", "2d-gaussian-splatting/guidance/dense_dn_util.py",
            "--source_path", mast3r_scene_path,
            "--model_path", free_gaussians_path,
            "--iteration", "7000",
        ])
        run_command_safe(gen_dn_dense_views_command)

        run_command_safe(generate_2Dplane_command)
        run_command_safe(get_plane_refine_depth_command(anchor_view_id_json_path=None, see3d_root_path=None))
        mv_cmd = f'mv {free_gaussians_path}/point_cloud {free_gaussians_path}/point_cloud-chart-views'
        run_command_safe(mv_cmd)
        run_command_safe(
            get_refine_free_gaussians_command(
                args.final_free_gaussians_config,
                args.dense_supervision,
                iterations=args.final_free_gaussians_iterations,
                non_position_lr_decay_from=args.final_non_position_lr_decay_from,
                non_position_lr_final_mult=args.final_non_position_lr_final_mult,
            )
        )

        # render all images, export mesh, and evaluate
        if not args.skip_default_render_eval:
            run_command_safe(render_all_img_command)
        run_command_safe(tetra_command)

        print("Finished training dense view without See3D prior!")

        t2 = time.time()
        print(f"Total running time: {t2 - t1} seconds")
        exit()


    if args.disable_see3d:
        print('[INFO] See3D disabled: continuing the 7k Gaussian state with real-view priors.')
        initial_ply = os.path.join(
            free_gaussians_path,
            'point_cloud',
            'iteration_7000',
            'point_cloud.ply',
        )
        if not os.path.exists(initial_ply):
            raise FileNotFoundError(
                'The final real-view refinement requires the initial 7k PLY: '
                + initial_ply
            )
        continuation_checkpoint = os.path.join(
            free_gaussians_path,
            'chkpnt7000.pth',
        )
        if os.path.exists(continuation_checkpoint):
            print(
                '[INFO] Continuing full refinement from the complete 7k checkpoint: '
                + continuation_checkpoint
            )
        else:
            print(
                '[WARNING] 7k continuation checkpoint is absent; falling back to protected PLY warm-start. '
                'New screen runs automatically save chkpnt7000.pth.'
            )
        if os.path.exists(continuation_checkpoint):
            final_refinement_command = get_refine_free_gaussians_command(
                args.final_free_gaussians_config,
                args.dense_supervision,
                start_checkpoint=continuation_checkpoint,
                iterations=args.final_free_gaussians_iterations,
                non_position_lr_decay_from=args.final_non_position_lr_decay_from,
                non_position_lr_final_mult=args.final_non_position_lr_final_mult,
            )
        else:
            final_refinement_command = get_refine_free_gaussians_command(
                args.final_free_gaussians_config,
                args.dense_supervision,
                init_ply=initial_ply,
                iterations=args.final_free_gaussians_iterations,
                non_position_lr_decay_from=args.final_non_position_lr_decay_from,
                non_position_lr_final_mult=args.final_non_position_lr_final_mult,
            )
        run_command_safe(final_refinement_command)
        if not args.skip_default_render_eval:
            run_command_safe(render_all_img_command)
        run_command_safe(tetra_command)
        print('Finished training without See3D.')
        print(f'Total running time: {time.time() - t1} seconds')
        sys.exit(0)

    # see3d inpainting stage 1 + refine depth with 2D planes + continue gaussian training
    resume_see3d_stage = args.continue_after_see3d_plane_stage
    if resume_see3d_stage is None or resume_see3d_stage < 1:
        run_command_safe(get_see3d_inpaint_command(1, args.select_inpaint_num))
        run_command_safe(get_plane_refine_depth_command(anchor_view_id_json_path=None, see3d_root_path=see3d_root_path))
    if resume_see3d_stage is None or resume_see3d_stage <= 1:
        mv_cmd = f'mv {free_gaussians_path}/point_cloud {free_gaussians_path}/point_cloud-ori'
        run_command_safe(mv_cmd)
        run_command_safe(
            get_refine_free_gaussians_command(
                args.free_gaussians_config,
                preliminary_uses_dense,
            )
        )

    # see3d inpainting stage 2 + refine depth with 2D planes + continue gaussian training
    if resume_see3d_stage is None or resume_see3d_stage < 2:
        run_command_safe(get_see3d_inpaint_command(2, args.select_inpaint_num))
        run_command_safe(get_plane_refine_depth_command(anchor_view_id_json_path=None, see3d_root_path=see3d_root_path))
    if resume_see3d_stage is None or resume_see3d_stage <= 2:
        mv_cmd = f'mv {free_gaussians_path}/point_cloud {free_gaussians_path}/point_cloud-s1'
        run_command_safe(mv_cmd)
        run_command_safe(
            get_refine_free_gaussians_command(
                args.free_gaussians_config,
                preliminary_uses_dense,
            )
        )

    # see3d inpainting stage 3 + refine depth with 2D planes + continue gaussian training
    if resume_see3d_stage is None or resume_see3d_stage < 3:
        run_command_safe(get_see3d_inpaint_command(3, args.select_inpaint_num))
        anchor_view_id_json_path = os.path.join(see3d_root_path, 'stage3', 'anchor_view_id.json')
        run_command_safe(get_plane_refine_depth_command(anchor_view_id_json_path=anchor_view_id_json_path, see3d_root_path=see3d_root_path))
    if resume_see3d_stage is None or resume_see3d_stage <= 3:
        mv_cmd = f'mv {free_gaussians_path}/point_cloud {free_gaussians_path}/point_cloud-s2'
        run_command_safe(mv_cmd)
        run_command_safe(
            get_refine_free_gaussians_command(
                args.final_free_gaussians_config,
                args.dense_supervision,
                iterations=args.final_free_gaussians_iterations,
                non_position_lr_decay_from=args.final_non_position_lr_decay_from,
                non_position_lr_final_mult=args.final_non_position_lr_final_mult,
            )
        )

    # render all images, export mesh, and evaluate
    if not args.skip_default_render_eval:
        run_command_safe(render_all_img_command)
    run_command_safe(tetra_command)

    if args.use_mesh_filter:
        # use mesh filter for forward facing scene
        mesh_path = os.path.join(tetra_meshes_path, 'tetra_mesh_binary_search_7_iter_7000.ply')
        length_threshold = 0.5
        filtered_mesh_path = os.path.join(tetra_meshes_path, f'tetra_mesh_binary_search_7_iter_7000_filtered_t{length_threshold}.ply')
        filter_mesh_command = " ".join([
            "python", "2d-gaussian-splatting/utils/mesh_filter.py",
            "--mesh_path", mesh_path,
            "--output_path", filtered_mesh_path,
        ])
        run_command_safe(filter_mesh_command)
        mv_cmd = f'mv {mesh_path} {tetra_meshes_path}/tetra_mesh_binary_search_7_iter_7000_ori.ply'
        run_command_safe(mv_cmd)
        mv_cmd = f'mv {filtered_mesh_path} {mesh_path}'
        run_command_safe(mv_cmd)

    if not args.skip_default_render_eval:
        run_command_safe(eval_command)

    # # vis global 3D plane by mesh (NOTE: slightly slow)
    # mesh_list = os.listdir(tetra_meshes_path)
    # mesh_list = [mesh_name for mesh_name in mesh_list if mesh_name.endswith('.ply')]
    # mesh_list.sort()
    # mesh_name = mesh_list[-1]
    # mesh_path = os.path.join(tetra_meshes_path, mesh_name)
    # print(f"Mesh path: {mesh_path}")
    # vis_global_3Dplane_by_mesh_command = " ".join([
    #     "python", "2d-gaussian-splatting/planes/vis_global_3Dplane_by_mesh.py",
    #     "--source_path", mast3r_scene_path,
    #     "--mesh_path", mesh_path,
    #     "--plane_root_path", plane_root_path,
    #     "--see3d_root_path", see3d_root_path,
    #     "--output_path", os.path.join(args.output_path, 'vis_global_plane_color_mesh.ply'),
    # ])
    # run_command_safe(vis_global_3Dplane_by_mesh_command)

    t2 = time.time()
    print(f"Total running time: {t2 - t1} seconds")
