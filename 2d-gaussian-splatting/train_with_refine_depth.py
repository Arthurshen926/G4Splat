#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.getcwd(), '2d-gaussian-splatting'))
from scene.dataset_readers import load_see3d_cameras

import gc
import copy
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.gaussian_model import get_gaussian_parameters_by_warp_from_depths
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from matcha.dm_scene.charts import (
    load_charts_data, 
    schedule_regularization_factor_2,
    get_gaussian_parameters_from_pa_data,
    depths_to_points_parallel,
    depth2normal_parallel,
    normal2curv_parallel,
    voxel_downsample_gaussians,
)
from matcha.dm_utils.rendering import normal2curv
from matcha.cambridge_masks import CambridgeMaskLookup, CambridgeTreeWeightLookup
from matcha.cambridge_training import (
    compute_masked_depth_order_loss,
    compute_rgb_loss,
    dense_depth_weight,
    geometry_iteration,
    linear_weight,
    load_depth_cache,
    masked_mean,
    PerImageAffineColorCorrection,
    rgb_supervision_weight,
    sanitize_depth,
    save_depth_cache,
)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np

# Old confidence to increasing weight function
# def confidence_to_weight(confidence:torch.Tensor):
#     conf_weights = confidence - 1.
#     return 1. - torch.exp(-conf_weights**2 / 2)

# New confidence to increasing weight function
def confidence_to_weight(confidence:torch.Tensor):
    conf_weights = confidence - 1.
    return torch.sigmoid((conf_weights - 2.) * 2.)


def load_pseudo_inpaint_masks(see3d_root_path, view_shapes, device):
    """Load masks where True denotes pixels synthesized by See3D."""
    mask_root = os.path.join(see3d_root_path, "visible_masks")
    masks = []
    for idx, (height, width) in enumerate(view_shapes):
        mask_path = os.path.join(mask_root, f"visible_mask_frame{idx:06d}.png")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"Pseudo-view masking requires {mask_path}. Regenerate See3D with the updated merge step."
            )
        visible_image = Image.open(mask_path).convert("L")
        if visible_image.size != (width, height):
            visible_image = visible_image.resize((width, height), Image.Resampling.NEAREST)
        visible = torch.from_numpy(np.asarray(visible_image).copy()).to(device) > 127
        masks.append(~visible)
    return masks


def training(
    dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, 
    use_refined_charts, use_mip_filter, dense_data_path, use_chart_view_every_n_iter,
    normal_consistency_from, distortion_from,
    depthanythingv2_checkpoint_dir, depthanything_encoder, 
    dense_regul, refine_depth_path, use_downsample_gaussians,
    downsample_gaussians_type, warp_depth_error_thresh, warp_downsample_pixel_grid_size,
    downweight_input_view_color_loss,
    cambridge_mask_pickle=None, cambridge_mask_dataset_path=None, cambridge_mask_indices=None,
    cambridge_geometry_mask_pickle=None, cambridge_geometry_mask_dataset_path=None,
    cambridge_geometry_mask_indices=None, rgb_loss_type="l1", rgb_charbonnier_eps=1e-3,
    dense_depth_cache=None, geometry_view_every_n_iter=5, dense_only_from_iter=3000,
    pseudo_rgb_weight=0.01, pseudo_geometry_weight=0.25,
    pseudo_geometry_final_weight=0.02, pseudo_geometry_decay_until=7000,
    pseudo_initialization_mode="all", pseudo_geometry_mask_mode="all",
    max_plane_abs_depth=50.0,
    init_fill_unsupported_with_prior=False,
    use_color_correction=False, color_correction_lr=1e-3,
    color_correction_reg=1e-2,
    cambridge_alpha_mask_indices=None, semantic_alpha_weight=0.0,
    cambridge_tree_mask_pickle=None, cambridge_tree_mask_dataset_path=None,
    cambridge_tree_mask_index=3, cambridge_tree_support_dir=None,
    tree_rgb_floor=0.25, tree_rgb_support_gain=0.50,
    tree_geometry_floor=0.05, tree_geometry_support_gain=0.25,
    tree_planar_weight=0.0, tree_sky_feather=4, tree_boundary_feather=6,
    init_ply=None, freeze_init_ply=False, warmstart_reseed_pixel_stride=8,
    warmstart_reseed_max_scale=0.05,
    warmstart_max_opacity=1.0, warmstart_max_scale=0.0,
    warmstart_max_position_delta=0.0, warmstart_diffuse_only=False,
    warmstart_clamp_dc=False,
):
    
    save_log_images = False
    save_log_images_every_n_iter = 200

    gaussian_points_count = []
    gaussian_points_iterations = []
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    
    # Sparse data
    scene = Scene(dataset, gaussians, shuffle=False)
    input_cams = scene.getTrainCameras()

    rgb_mask_lookup = None
    geometry_mask_lookup = None
    alpha_mask_lookup = None
    mask_dataset_path = cambridge_mask_dataset_path or dense_data_path or dataset.source_path
    if cambridge_mask_pickle is not None:
        rgb_mask_lookup = CambridgeMaskLookup(
            mask_dataset_path,
            cambridge_mask_pickle,
            mask_indices=cambridge_mask_indices or [0, 1, 2],
        )
        print(
            f"[INFO] Cambridge RGB mask indices: {rgb_mask_lookup.mask_indices}; "
            f"source: {cambridge_mask_pickle}"
        )
    if cambridge_geometry_mask_pickle is None and cambridge_geometry_mask_indices is not None:
        cambridge_geometry_mask_pickle = cambridge_mask_pickle
    if cambridge_geometry_mask_pickle is not None:
        geometry_dataset_path = cambridge_geometry_mask_dataset_path or mask_dataset_path
        geometry_mask_indices = cambridge_geometry_mask_indices or [0, 1, 2]
        shares_rgb_masks = (
            rgb_mask_lookup is not None
            and os.path.realpath(os.fspath(geometry_dataset_path))
            == os.path.realpath(os.fspath(mask_dataset_path))
            and os.path.realpath(os.fspath(cambridge_geometry_mask_pickle))
            == os.path.realpath(os.fspath(cambridge_mask_pickle))
            and list(geometry_mask_indices) == rgb_mask_lookup.mask_indices
        )
        if shares_rgb_masks:
            geometry_mask_lookup = rgb_mask_lookup
            print("[INFO] Sharing the Cambridge RGB mask lookup with geometry losses.")
        else:
            geometry_mask_lookup = CambridgeMaskLookup(
                geometry_dataset_path,
                cambridge_geometry_mask_pickle,
                mask_indices=geometry_mask_indices,
            )
        print(
            f"[INFO] Cambridge geometry mask indices: {geometry_mask_lookup.mask_indices}; "
            f"source: {cambridge_geometry_mask_pickle}"
        )
    if rgb_mask_lookup is not None and cambridge_alpha_mask_indices and semantic_alpha_weight > 0:
        alpha_mask_lookup = rgb_mask_lookup.with_indices(cambridge_alpha_mask_indices)
        print(
            f"[INFO] Cambridge invalid-region alpha suppression: "
            f"indices={alpha_mask_lookup.mask_indices}, weight={semantic_alpha_weight}."
        )

    tree_weight_lookup = None
    if cambridge_tree_mask_pickle is not None:
        tree_dataset_path = cambridge_tree_mask_dataset_path or mask_dataset_path
        tree_base_lookup = CambridgeMaskLookup(
            tree_dataset_path,
            cambridge_tree_mask_pickle,
            mask_indices=cambridge_mask_indices or [0, 1, 2],
        )
        tree_weight_lookup = CambridgeTreeWeightLookup(
            tree_base_lookup,
            tree_mask_index=cambridge_tree_mask_index,
            support_dir=cambridge_tree_support_dir,
            rgb_floor=tree_rgb_floor,
            rgb_support_gain=tree_rgb_support_gain,
            geometry_floor=tree_geometry_floor,
            geometry_support_gain=tree_geometry_support_gain,
            planar_tree_weight=tree_planar_weight,
            sky_feather_pixels=tree_sky_feather,
            tree_feather_pixels=tree_boundary_feather,
        )
        print(
            "[INFO] Canonical-tree soft weighting enabled: "
            f"index={cambridge_tree_mask_index}, support={cambridge_tree_support_dir}, "
            f"rgb=({tree_rgb_floor}+{tree_rgb_support_gain}*S), "
            f"geometry=({tree_geometry_floor}+{tree_geometry_support_gain}*S), "
            f"planar={tree_planar_weight}."
        )

    use_dense_supervision = dense_data_path is not None
    dense_viewpoint_cams = []
    dense_viewpoint_idx_stack = None
    if use_dense_supervision:
        print(f"[INFO] Loading real dense supervision from: {dense_data_path}")
        dense_dataset = copy.deepcopy(dataset)
        dense_dataset.source_path = dense_data_path
        dense_dataset.model_path = os.path.join(dataset.model_path, "dense_data")
        os.makedirs(dense_dataset.model_path, exist_ok=True)
        dense_gaussians = GaussianModel(dataset.sh_degree)
        dense_scene = Scene(dense_dataset, dense_gaussians, shuffle=False)
        dense_viewpoint_cams = dense_scene.getTrainCameras()
        gaussians.spatial_lr_scale = dense_gaussians.spatial_lr_scale
        scene.cameras_extent = dense_scene.cameras_extent
        print(
            f"[INFO] Dense cameras: {len(dense_viewpoint_cams)}; "
            f"resolution: {tuple(dense_viewpoint_cams[0].original_image.shape)}"
        )
        del dense_gaussians, dense_scene
        gc.collect()
        torch.cuda.empty_cache()

    # NOTE: hard code for See3D root path
    see3d_root_path = os.path.join(dataset.source_path, 'see3d_render')
    see3d_cam_path = os.path.join(see3d_root_path, 'see3d_cameras.npz')
    inpaint_root_dir = os.path.join(see3d_root_path, 'inpainted_images')
    if os.path.exists(see3d_cam_path):
        see3d_gs_cameras_list, _ = load_see3d_cameras(see3d_cam_path, inpaint_root_dir)
    else:
        see3d_gs_cameras_list = []
    
    print(
        f"[INFO] Plane/chart views every {geometry_view_every_n_iter} iteration(s) through "
        f"iteration {dense_only_from_iter}; dense supervision enabled={use_dense_supervision}."
    )
    
    # ===================================================================================
    # Create gaussians from charts data
    print("[INFO] Loading charts data...")
    charts_data_path = f'{dataset.source_path}/charts_data.npz'
    if use_refined_charts:
        charts_data_path = f'{dataset.source_path}/refined_charts_data.npz'
    print("Using charts data from: ", charts_data_path)
    charts_data = load_charts_data(charts_data_path)
    charts_data['confs'] = charts_data['confs'] # - 1.  # Was not there before
    print("[WARNING] Confidence values are not being subtracted by 1.0 as in the original implementation.")
    print("Minimum confidence: ", charts_data['confs'].min())
    print("Maximum confidence: ", charts_data['confs'].max())

    print(f'[INFO]: Load plane-aware depth from: {refine_depth_path}')
    pa_depths = []
    pa_confident_maps_list = []

    input_view_num = len(scene.getTrainCameras())
    see3d_view_num = len(see3d_gs_cameras_list)
    training_view_num = input_view_num + see3d_view_num
    for idx in range(training_view_num):
        pa_depth_path = os.path.join(refine_depth_path, f'refine_depth_frame{idx:06d}.tiff')
        pa_depth = Image.open(pa_depth_path)
        pa_depth = np.array(pa_depth)
        pa_depth = torch.from_numpy(pa_depth).cuda()
        pa_depths.append(pa_depth)

        pa_confident_map_path = os.path.join(refine_depth_path, f'confident_map_frame{idx:06d}.png')
        pa_confident_map = Image.open(pa_confident_map_path)
        pa_confident_map = np.array(pa_confident_map) / 255                   # 0 or 1
        pa_confident_map = torch.from_numpy(pa_confident_map).cuda()
        pa_confident_maps_list.append(pa_confident_map)

    plane_valid_masks = []
    for idx, (depth, confidence) in enumerate(zip(pa_depths, pa_confident_maps_list)):
        semantic_mask = None
        if idx < input_view_num and geometry_mask_lookup is not None:
            semantic_mask = geometry_mask_lookup.get_mask(
                input_cams[idx].image_name,
                depth.shape[-2:],
                depth.device,
            )
        clean_depth, valid_mask = sanitize_depth(
            depth,
            confidence=confidence,
            semantic_mask=semantic_mask,
            max_abs_depth=max_plane_abs_depth,
        )
        pa_depths[idx] = clean_depth
        plane_valid_masks.append(valid_mask)

    pseudo_inpaint_masks = None
    if see3d_view_num > 0 and (
        pseudo_initialization_mode == "inpaint_only"
        or pseudo_geometry_mask_mode == "inpaint_only"
    ):
        pseudo_inpaint_masks = load_pseudo_inpaint_masks(
            see3d_root_path,
            [tuple(pa_depths[input_view_num + idx].shape[-2:]) for idx in range(see3d_view_num)],
            pa_depths[0].device,
        )
        print(
            "[INFO] Loaded pseudo inpaint masks: "
            f"mean synthesized ratio={torch.stack(pseudo_inpaint_masks).float().mean().item():.3f}."
        )
    input_valid_ratio = torch.stack(plane_valid_masks[:input_view_num]).float().mean().item()
    print(
        f"[INFO] Plane depth validity after confidence/outlier/semantic filtering: "
        f"{input_valid_ratio * 100:.2f}% over real chart views."
    )

    # ===================================================================================
    # Initialize gaussians
    max_gaussians_num = 10_000_000
    print(
        f"Max gaussians num: {max_gaussians_num}, use downsample gaussians: {use_downsample_gaussians}, "
        f"downsample type: {downsample_gaussians_type}"
    )

    input_view_depths = list(pa_depths[:input_view_num])
    initialization_valid_masks = list(plane_valid_masks[:input_view_num])
    if init_fill_unsupported_with_prior:
        prior_depths = torch.nn.functional.interpolate(
            charts_data['prior_depths'][:input_view_num, None]
            / charts_data['scale_factor'],
            size=input_view_depths[0].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        filled_pixels = 0
        static_pixels = 0
        for idx in range(input_view_num):
            semantic_mask = torch.ones_like(initialization_valid_masks[idx])
            if geometry_mask_lookup is not None:
                semantic_mask = geometry_mask_lookup.get_mask(
                    input_cams[idx].image_name,
                    input_view_depths[idx].shape[-2:],
                    input_view_depths[idx].device,
                )
            fallback_valid = semantic_mask & torch.isfinite(prior_depths[idx]) & (prior_depths[idx] > 0)
            if max_plane_abs_depth is not None and max_plane_abs_depth > 0:
                fallback_valid &= prior_depths[idx] <= max_plane_abs_depth
            fill_mask = fallback_valid & (~initialization_valid_masks[idx])
            input_view_depths[idx] = torch.where(
                fill_mask,
                prior_depths[idx],
                input_view_depths[idx],
            )
            initialization_valid_masks[idx] = initialization_valid_masks[idx] | fill_mask
            filled_pixels += int(fill_mask.sum())
            static_pixels += int(semantic_mask.sum())
        print(
            "[INFO] Prior-only initialization filled "
            f"{filled_pixels / max(static_pixels, 1) * 100:.2f}% of static chart pixels; "
            "chart geometry losses remain alignment-supported only."
        )
    warmstart_requested = init_ply is not None
    _images = (
        []
        if warmstart_requested
        else [cam.original_image.cuda().permute(1, 2, 0) for cam in scene.getTrainCameras()]
    )
    # A warm start does not need to rebuild and immediately discard the real-chart
    # initialization. Only warp-sample pseudo holes that will be appended later.
    use_warp_downsample = warmstart_requested or (
        use_downsample_gaussians and downsample_gaussians_type == "warp"
    )
    voxel_max_init_gs_input_view_num = 50
    warp_max_init_gs_input_view_num = None
    max_init_gs_input_view_num = (
        0
        if warmstart_requested
        else (
            warp_max_init_gs_input_view_num
            if use_warp_downsample
            else voxel_max_init_gs_input_view_num
        )
    )
    if max_init_gs_input_view_num is not None and input_view_num > max_init_gs_input_view_num:
        print(f'[INFO]: Input view num is too large: {input_view_num}, use {max_init_gs_input_view_num} views for gs initialization')
        init_view_ids = np.linspace(0, input_view_num - 1, max_init_gs_input_view_num, dtype=int)
        init_input_view_depths = [input_view_depths[i] for i in init_view_ids]
        init_input_views = [scene.getTrainCameras()[i] for i in init_view_ids]
        init_images = [_images[i] for i in init_view_ids]
    else:
        init_input_view_depths = input_view_depths
        init_input_views = scene.getTrainCameras()
        init_images = _images

    warp_init_depths = list(init_input_view_depths)
    warp_init_views = list(init_input_views)
    warp_init_valid_masks = [
        initialization_valid_masks[i]
        for i in (
            init_view_ids
            if max_init_gs_input_view_num is not None and input_view_num > max_init_gs_input_view_num
            else range(input_view_num)
        )
    ]
    
    use_pseudo_initialization = see3d_view_num > 0 and pseudo_initialization_mode != "none"
    if use_pseudo_initialization:
        see3d_view_depths = pa_depths[input_view_num:]
        _images = (
            []
            if warmstart_requested
            else [cam.original_image.cuda().permute(1, 2, 0) for cam in see3d_gs_cameras_list]
        )
        pseudo_init_masks = plane_valid_masks[input_view_num:]
        if pseudo_initialization_mode == "inpaint_only":
            pseudo_init_masks = [
                valid & inpaint
                for valid, inpaint in zip(pseudo_init_masks, pseudo_inpaint_masks)
            ]

        if see3d_view_num > 30:
            print(f'[INFO]: See3D view num is too large: {see3d_view_num}, use 30 views for gs initialization')
            # NOTE: hard code for 15 select inpaint views, use 0 - 9, 15 - 24, 30 - 39 (in see3d view id)
            used_see3d_init_gs_view_list = list(range(10)) + list(range(15, 25)) + list(range(30, see3d_view_num))
            init_gs_see3d_view_depths = [see3d_view_depths[i] for i in used_see3d_init_gs_view_list]
            init_gs_see3d_gs_cameras_list = [see3d_gs_cameras_list[i] for i in used_see3d_init_gs_view_list]
            init_gs_see3d_images = (
                []
                if warmstart_requested
                else [_images[i] for i in used_see3d_init_gs_view_list]
            )
            warp_init_depths.extend(init_gs_see3d_view_depths)
            warp_init_views.extend(init_gs_see3d_gs_cameras_list)
            warp_init_valid_masks.extend(
                pseudo_init_masks[i] for i in used_see3d_init_gs_view_list
            )

            if not use_warp_downsample:
                init_gs_see3d_view_depths_stack = torch.stack(init_gs_see3d_view_depths, dim=0).cuda()
                init_gs_see3d_points = depths_to_points_parallel(init_gs_see3d_view_depths_stack, init_gs_see3d_gs_cameras_list)
                N, H, W = init_gs_see3d_view_depths_stack.shape
                init_gs_see3d_points = init_gs_see3d_points.reshape(N, H, W, 3)
                see3d_gaussian_params = get_gaussian_parameters_from_pa_data(
                    pa_points=init_gs_see3d_points,
                    images=init_gs_see3d_images,
                    conf_th=-1.,  # TODO: Try higher values
                    ratio_th=5.,
                    normal_scale=1e-10,
                    normalized_scales=0.5,
                    visibility_masks=[pseudo_init_masks[i] for i in used_see3d_init_gs_view_list],
                )

        else:
            warp_init_depths.extend(see3d_view_depths)
            warp_init_views.extend(see3d_gs_cameras_list)
            warp_init_valid_masks.extend(pseudo_init_masks)
            if not use_warp_downsample:
                see3d_view_depths_stack = torch.stack(see3d_view_depths, dim=0).cuda()
                see3d_points = depths_to_points_parallel(see3d_view_depths_stack, see3d_gs_cameras_list)
                N, H, W = see3d_view_depths_stack.shape
                see3d_points = see3d_points.reshape(N, H, W, 3)
                see3d_gaussian_params = get_gaussian_parameters_from_pa_data(
                    pa_points=see3d_points,
                    images=_images,
                    conf_th=-1.,  # TODO: Try higher values
                    ratio_th=5.,
                    normal_scale=1e-10,
                    normalized_scales=0.5,
                    visibility_masks=pseudo_init_masks,
                )

    warmstart_reseed_params = None
    if warmstart_requested and not use_pseudo_initialization:
        print("[INFO] Warm-start has no pseudo views to reseed; skipping initialization.")
    elif use_warp_downsample:
        _means, _scales, _quaternions, _colors = get_gaussian_parameters_by_warp_from_depths(
            depths=warp_init_depths,
            views=warp_init_views,
            depth_error_thresh=warp_depth_error_thresh,
            max_scale=(
                float(warmstart_reseed_max_scale)
                if warmstart_requested
                else 0.05
            ),
            downsample_pixel_grid_size=(
                max(int(warmstart_reseed_pixel_stride), 1)
                if warmstart_requested
                else warp_downsample_pixel_grid_size
            ),
            valid_masks=warp_init_valid_masks,
        )
        print(f"Warp-downsampled Gaussian initialization produced {len(_means)} gaussians.")
    else:
        input_view_depths_stack = torch.stack(init_input_view_depths, dim=0).cuda()
        pa_points = depths_to_points_parallel(input_view_depths_stack, init_input_views)
        N, H, W = input_view_depths_stack.shape
        pa_points = pa_points.reshape(N, H, W, 3)
        input_view_gaussian_params = get_gaussian_parameters_from_pa_data(
            pa_points=pa_points,
            images=init_images,
            conf_th=-1.,  # TODO: Try higher values
            ratio_th=5.,
            normal_scale=1e-10,
            normalized_scales=0.5,
            visibility_masks=[initialization_valid_masks[i] for i in init_view_ids]
            if max_init_gs_input_view_num is not None and input_view_num > max_init_gs_input_view_num
            else initialization_valid_masks,
        )

        if use_pseudo_initialization:
            gaussian_params = {}
            for key in input_view_gaussian_params.keys():
                gaussian_params[key] = torch.cat([input_view_gaussian_params[key], see3d_gaussian_params[key]], dim=0)
        else:
            gaussian_params = input_view_gaussian_params

        # Downsample gaussians
        if len(gaussian_params['means']) > max_gaussians_num and use_downsample_gaussians:
            sample_idx, downsample_factor = voxel_downsample_gaussians(gaussian_params, voxel_size=0.01)
            print(f"Voxel-downsampled {len(gaussian_params['means'])} gaussians to {len(sample_idx)} gaussians...")
        else:
            sample_idx = torch.arange(len(gaussian_params['means']), device=gaussian_params['means'].device)
            downsample_factor = 1.0
            print(f"Not downsampling gaussians, using all {len(gaussian_params['means'])} gaussians...")

        _means = gaussian_params['means'][sample_idx]
        _scales = gaussian_params['scales'][..., :2][sample_idx] * downsample_factor
        _quaternions = gaussian_params['quaternions'][sample_idx]
        _colors = gaussian_params['colors'][sample_idx]

    if warmstart_requested and use_pseudo_initialization:
        warmstart_reseed_params = (_means, _scales, _quaternions, _colors)
        print(f"Warm-start reseed candidates: {len(_means)}")
    elif not warmstart_requested:
        print(f"Final number of gaussians: {len(_means)}")
        gaussians.create_from_parameters(
            _means, _scales, _quaternions, _colors, gaussians.spatial_lr_scale
        )
        print("[INFO] Gaussians created from pnts data.")
        del _means, _scales, _quaternions, _colors
    gc.collect()
    torch.cuda.empty_cache()

    warmstart_baseline_count = 0
    if init_ply is not None:
        init_ply = os.path.abspath(os.path.expanduser(init_ply))
        if not os.path.isfile(init_ply):
            raise FileNotFoundError(f"Warm-start PLY does not exist: {init_ply}")
        gaussians.load_ply(init_ply)
        gaussians.max_radii2D = torch.zeros(
            (gaussians.get_xyz.shape[0]),
            device=gaussians.get_xyz.device,
        )
        warmstart_baseline_count = len(gaussians.get_xyz)
        print(
            f"[INFO] Warm-started from {init_ply} with "
            f"{warmstart_baseline_count} baseline Gaussians."
        )

        if warmstart_reseed_params is not None:
            try:
                reseed_means, reseed_scales, reseed_quaternions, reseed_colors = (
                    warmstart_reseed_params
                )
                reseed_count = gaussians.append_from_parameters(
                    reseed_means,
                    reseed_scales,
                    reseed_quaternions,
                    reseed_colors,
                    initial_opacity=0.05,
                )
                print(
                    f"[INFO] Appended {reseed_count} low-opacity See3D hole Gaussians "
                    f"with pixel stride {warmstart_reseed_pixel_stride}."
                )
                del reseed_means, reseed_scales, reseed_quaternions, reseed_colors
            except RuntimeError as error:
                print(f"[WARNING] Warm-start reseed skipped: {error}")
            gc.collect()
            torch.cuda.empty_cache()
    
    # ===================================================================================
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    if freeze_init_ply:
        if warmstart_baseline_count <= 0:
            raise ValueError("--freeze_init_ply requires --init_ply")
        # restore() replaces the optimized tensors, so install hooks and capture
        # the suffix anchors only after a checkpoint has been restored.
        gaussians.freeze_prefix_gradients(warmstart_baseline_count)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_idx_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_prior_depth_for_log = 0.0
    ema_prior_normal_for_log = 0.0
    ema_prior_curvature_for_log = 0.0
    ema_prior_anisotropy_for_log = 0.0
    ema_alpha_suppression_for_log = 0.0
    
    # ===================================================================================

    # geometry supervision for input views
    charts_scale_factor = 1.0
    input_refine_depths = pa_depths[:input_view_num]
    input_refine_depths = torch.stack(input_refine_depths, dim=0).cuda()
    input_pseudo_confs = pa_confident_maps_list[:input_view_num]
    input_pseudo_confs = torch.stack(input_pseudo_confs, dim=0).cuda()
    input_world_view_transforms = torch.stack([input_cams[i].world_view_transform for i in range(len(input_cams))])
    input_full_proj_transforms = torch.stack([input_cams[i].full_proj_transform for i in range(len(input_cams))])
    input_prior_normals = depth2normal_parallel(
        input_refine_depths, 
        world_view_transforms=input_world_view_transforms, 
        full_proj_transforms=input_full_proj_transforms
    ).permute(0, 3, 1, 2)  # Shape (n_charts, 3, h ,w)
    input_prior_curvs = normal2curv_parallel(input_prior_normals, torch.ones_like(input_prior_normals[:, 0:1]))
    input_geometry_masks = torch.stack(plane_valid_masks[:input_view_num], dim=0)
    print('Input pointmap loaded!')

    # geometry supervision for see3d views
    if see3d_view_num > 0:
        see3d_refine_depths = pa_depths[input_view_num:]
        see3d_refine_depths = torch.stack(see3d_refine_depths, dim=0).cuda()        # [n_views, h, w]
        see3d_pseudo_confs = pa_confident_maps_list[input_view_num:]
        see3d_pseudo_confs = torch.stack(see3d_pseudo_confs, dim=0).cuda()
        see3d_world_view_transforms = torch.stack([see3d_gs_cameras_list[i].world_view_transform for i in range(len(see3d_gs_cameras_list))])
        see3d_full_proj_transforms = torch.stack([see3d_gs_cameras_list[i].full_proj_transform for i in range(len(see3d_gs_cameras_list))])
        see3d_prior_normals = depth2normal_parallel(
            see3d_refine_depths, 
            world_view_transforms=see3d_world_view_transforms, 
            full_proj_transforms=see3d_full_proj_transforms
        ).permute(0, 3, 1, 2)  # Shape (n_charts, 3, h ,w)
        see3d_prior_curvs = normal2curv_parallel(see3d_prior_normals, torch.ones_like(see3d_prior_normals[:, 0:1]))
        see3d_geometry_masks = torch.stack(plane_valid_masks[input_view_num:], dim=0)
        if pseudo_geometry_mask_mode == "inpaint_only":
            see3d_geometry_masks = see3d_geometry_masks & torch.stack(
                pseudo_inpaint_masks,
                dim=0,
            )
        elif pseudo_geometry_mask_mode == "none":
            see3d_geometry_masks = torch.zeros_like(see3d_geometry_masks)
        print('See3D pointmap loaded!')

    # Keep pseudo geometry generation independent from whether it supervises GS.
    # This lets plane fusion improve real charts without spending iterations on
    # generated images that failed scene-specific quality screening.
    use_pseudo_geometry_supervision = (
        pseudo_geometry_mask_mode != "none"
        and max(pseudo_geometry_weight, pseudo_geometry_final_weight) > 0.0
    )
    use_pseudo_supervision = (
        see3d_view_num > 0
        and (pseudo_rgb_weight > 0.0 or use_pseudo_geometry_supervision)
    )
    if use_pseudo_supervision:
        total_views_list = input_cams + see3d_gs_cameras_list
        total_confs_list = [input_pseudo_confs[idx] for idx in range(len(input_pseudo_confs))] + [see3d_pseudo_confs[idx].unsqueeze(0) for idx in range(len(see3d_pseudo_confs))]
        total_depths_list = [input_refine_depths[idx] for idx in range(len(input_refine_depths))] + [see3d_refine_depths[idx].unsqueeze(0) for idx in range(len(see3d_refine_depths))]
        total_normals_list = [input_prior_normals[idx] for idx in range(len(input_prior_normals))] + [see3d_prior_normals[idx] for idx in range(len(see3d_prior_normals))]
        total_curvs_list = [input_prior_curvs[idx] for idx in range(len(input_prior_curvs))] + [see3d_prior_curvs[idx] for idx in range(len(see3d_prior_curvs))]
        total_geometry_masks_list = [input_geometry_masks[idx] for idx in range(len(input_geometry_masks))] + [see3d_geometry_masks[idx] for idx in range(len(see3d_geometry_masks))]
    else:
        total_views_list = input_cams
        total_confs_list = [input_pseudo_confs[idx] for idx in range(len(input_pseudo_confs))]
        total_depths_list = [input_refine_depths[idx] for idx in range(len(input_refine_depths))]
        total_normals_list = [input_prior_normals[idx] for idx in range(len(input_prior_normals))]
        total_curvs_list = [input_prior_curvs[idx] for idx in range(len(input_prior_curvs))]
        total_geometry_masks_list = [input_geometry_masks[idx] for idx in range(len(input_geometry_masks))]

    print(
        f"[INFO] Total number of supervised views: {len(total_views_list)}, "
        f"input views: {len(input_cams)}, enabled See3D views: "
        f"{len(see3d_gs_cameras_list) if use_pseudo_supervision else 0}/"
        f"{len(see3d_gs_cameras_list)}"
    )

    color_correction = None
    color_correction_optimizer = None
    if use_color_correction:
        real_image_names = [camera.image_name for camera in input_cams]
        real_image_names.extend(camera.image_name for camera in dense_viewpoint_cams)
        color_correction = PerImageAffineColorCorrection(real_image_names).cuda()
        color_correction_optimizer = torch.optim.Adam(
            color_correction.parameters(),
            lr=color_correction_lr,
        )
        print(
            f"[INFO] Using per-image affine color correction for "
            f"{len(color_correction.image_name_to_index)} real image(s), "
            f"lr={color_correction_lr}, reg={color_correction_reg}."
        )

    # Set mip filter
    if use_mip_filter:
        print("[INFO] Using mip filter during training.")
        gaussians.set_mip_filter(use_mip_filter)
        gaussians.compute_mip_filter(cameras=dense_viewpoint_cams if use_dense_supervision else total_views_list)

    dense_depth_priors = None
    if use_dense_supervision and dense_regul != "none":
        dense_names = [camera.image_name for camera in dense_viewpoint_cams]
        cache_path = dense_depth_cache or os.path.join(
            dataset.model_path,
            f"dense_depth_priors_{depthanything_encoder}.pt",
        )
        if os.path.exists(cache_path):
            print(f"[INFO] Loading dense Depth Anything priors from: {cache_path}")
            dense_depth_priors = load_depth_cache(cache_path, dense_names)
        else:
            print(f"[INFO] Building dense Depth Anything priors for {len(dense_names)} views...")
            from matcha.pointmap.depthanythingv2 import apply_depthanything
            from matcha.pointmap.depthanythingv2 import load_model as load_depthanythingv2

            dav2 = load_depthanythingv2(
                checkpoint_dir=depthanythingv2_checkpoint_dir,
                encoder=depthanything_encoder,
                device="cuda",
            )
            dav2.eval()
            dense_depth_priors = []
            with torch.no_grad():
                for index, camera in enumerate(dense_viewpoint_cams):
                    gt_image = camera.original_image.permute(1, 2, 0)
                    disparity = apply_depthanything(dav2, image=gt_image)
                    disparity_range = (disparity.max() - disparity.min()).clamp_min(1e-6)
                    disparity = (disparity - disparity.min()) / disparity_range
                    depth = 1.0 / (0.1 + 0.9 * disparity)
                    dense_depth_priors.append(depth.squeeze().unsqueeze(0).half().cpu())
                    if (index + 1) % 100 == 0:
                        print(f"[INFO] Dense depth priors: {index + 1}/{len(dense_names)}")
            del dav2
            gc.collect()
            torch.cuda.empty_cache()
            save_depth_cache(cache_path, dense_names, dense_depth_priors)
            print(f"[INFO] Saved dense Depth Anything priors to: {cache_path}")

    # # ===================================================================================
    
    print(f"\n[INFO] Normal consistency from iteration {normal_consistency_from} with lambda_normal {opt.lambda_normal}")
    print(f"[INFO] Distortion from iteration {distortion_from} with lambda_dist {opt.lambda_dist}")

    # TODO: Should the sparse depth order regularization be used during dense supervision? It's not clear.
    # use_depth_order_regularization = not use_dense_supervision
    use_depth_order_regularization = True
    if use_depth_order_regularization:
        print(f"[INFO] Using depth order regularization for charts.")

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        use_geometry_view = geometry_iteration(
            iteration,
            use_dense_supervision=use_dense_supervision,
            every_n=geometry_view_every_n_iter,
            dense_only_from_iter=dense_only_from_iter,
        )
        if use_geometry_view:
            if not viewpoint_idx_stack:
                viewpoint_idx_stack = list(range(len(total_views_list)))
            viewpoint_idx = viewpoint_idx_stack.pop(randint(0, len(viewpoint_idx_stack) - 1))
            viewpoint_cam = total_views_list[viewpoint_idx]
            dense_viewpoint_idx = None
            is_pseudo_view = viewpoint_idx >= input_view_num
            current_geometry_mask = total_geometry_masks_list[viewpoint_idx]
        else:
            if not dense_viewpoint_idx_stack:
                dense_viewpoint_idx_stack = list(range(len(dense_viewpoint_cams)))
            dense_viewpoint_idx = dense_viewpoint_idx_stack.pop(
                randint(0, len(dense_viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = dense_viewpoint_cams[dense_viewpoint_idx]
            viewpoint_idx = None
            is_pseudo_view = False
            current_geometry_mask = None
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        rgb_mask = None
        current_tree_geometry_weight = None
        current_tree_planar_weight = None
        if tree_weight_lookup is not None and not is_pseudo_view:
            rgb_mask, current_tree_geometry_weight, current_tree_planar_weight = (
                tree_weight_lookup.weights(
                    viewpoint_cam.image_name,
                    gt_image.shape[-2:],
                    gt_image.device,
                )
            )
        elif rgb_mask_lookup is not None and not is_pseudo_view:
            rgb_mask = rgb_mask_lookup.get_mask(
                viewpoint_cam.image_name,
                gt_image.shape[-2:],
                gt_image.device,
            )
        image_for_rgb_loss = image
        if color_correction is not None and not is_pseudo_view:
            image_for_rgb_loss = color_correction(image, viewpoint_cam.image_name)
        Ll1, loss = compute_rgb_loss(
            image_for_rgb_loss,
            gt_image,
            mask=rgb_mask,
            lambda_dssim=opt.lambda_dssim,
            ssim_fn=ssim,
            loss_type=rgb_loss_type,
            charbonnier_eps=rgb_charbonnier_eps,
        )
        loss = loss * rgb_supervision_weight(
            is_pseudo_view=is_pseudo_view,
            is_geometry_view=use_geometry_view,
            use_dense_supervision=use_dense_supervision,
            downweight_input_view_color_loss=downweight_input_view_color_loss,
            pseudo_rgb_weight=pseudo_rgb_weight,
        )

        if current_geometry_mask is None and geometry_mask_lookup is not None and not is_pseudo_view:
            current_geometry_mask = geometry_mask_lookup.get_mask(
                viewpoint_cam.image_name,
                image.shape[-2:],
                image.device,
            )
        base_geometry_mask = current_geometry_mask
        if current_tree_geometry_weight is not None:
            if base_geometry_mask is None:
                current_geometry_mask = current_tree_geometry_weight
            else:
                geometry_weight = current_tree_geometry_weight
                if geometry_weight.shape[-2:] != base_geometry_mask.shape[-2:]:
                    geometry_weight = F.interpolate(
                        geometry_weight[None, None],
                        size=base_geometry_mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                current_geometry_mask = base_geometry_mask.to(dtype=geometry_weight.dtype) * geometry_weight
        current_planar_mask = base_geometry_mask
        if current_tree_planar_weight is not None:
            if base_geometry_mask is None:
                current_planar_mask = current_tree_planar_weight
            else:
                planar_weight = current_tree_planar_weight
                if planar_weight.shape[-2:] != base_geometry_mask.shape[-2:]:
                    planar_weight = F.interpolate(
                        planar_weight[None, None],
                        size=base_geometry_mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                current_planar_mask = base_geometry_mask.to(dtype=planar_weight.dtype) * planar_weight

        alpha_suppression_loss = loss.detach() * 0.0
        if alpha_mask_lookup is not None and not is_pseudo_view:
            alpha_keep_mask = alpha_mask_lookup.get_mask(
                viewpoint_cam.image_name,
                image.shape[-2:],
                image.device,
            )
            alpha_suppression_loss = semantic_alpha_weight * masked_mean(
                render_pkg["rend_alpha"],
                ~alpha_keep_mask,
            )
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > normal_consistency_from else 0.0
        lambda_dist = opt.lambda_dist if iteration > distortion_from else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * masked_mean(normal_error, current_planar_mask)
        dist_loss = lambda_dist * masked_mean(rend_dist, current_geometry_mask)
        
        # loss
        total_loss = loss + dist_loss + normal_loss + alpha_suppression_loss
        if color_correction is not None:
            total_loss = total_loss + (
                color_correction_reg * color_correction.identity_regularization()
            )
        
        # ===================================================================================

        surf_depth = render_pkg['surf_depth']
        zero_loss = total_loss.detach() * 0.0
        depth_prior_loss = zero_loss
        normal_prior_loss = zero_loss
        curv_prior_loss = zero_loss
        anisotropy_loss = zero_loss
        lambda_anisotropy = 0.0
        anisotropy_max_ratio = 5.

        if use_geometry_view:
            current_depth = total_depths_list[viewpoint_idx].to(surf_depth.device)
            current_normal = total_normals_list[viewpoint_idx]
            current_curv = total_curvs_list[viewpoint_idx]
            current_conf = total_confs_list[viewpoint_idx].to(surf_depth.device)
            current_geometry_mask = current_geometry_mask.to(surf_depth.device)
            rend_curvature = normal2curv(
                render_pkg['rend_normal'],
                torch.ones_like(render_pkg['rend_normal'][0:1]),
            )

            regularization_factor = schedule_regularization_factor_2(iteration, 0.5)
            lambda_prior_depth = regularization_factor * 0.75
            lambda_prior_depth_derivative = regularization_factor * 0.5
            lambda_prior_normal = regularization_factor * 0.5
            lambda_prior_curvature = regularization_factor * 0.25
            confidence_weight = 0.5 * current_conf.clamp(0.0, 1.0)

            depth_term = confidence_weight * torch.log(
                1.0 + charts_scale_factor * (current_depth - surf_depth).abs()
            )
            depth_prior_loss = lambda_prior_depth * masked_mean(
                depth_term,
                current_geometry_mask,
            )
            if lambda_prior_depth_derivative > 0:
                depth_prior_loss = depth_prior_loss + lambda_prior_depth_derivative * masked_mean(
                    1.0 - (surf_normal * current_normal).sum(dim=0),
                    current_planar_mask,
                )
            normal_prior_loss = lambda_prior_normal * masked_mean(
                1.0 - (rend_normal * current_normal).sum(dim=0),
                current_planar_mask,
            )
            curv_prior_loss = lambda_prior_curvature * masked_mean(
                (current_curv - rend_curvature).abs(),
                current_planar_mask,
            )

            lambda_depth_order = 0.0
            if iteration > 1500:
                lambda_depth_order = 1.0
            if iteration > 3000:
                lambda_depth_order = 0.1
            if iteration > 4500:
                lambda_depth_order = 0.01
            if iteration > 6000:
                lambda_depth_order = 0.001
            if use_depth_order_regularization and lambda_depth_order > 0:
                depth_prior_loss = depth_prior_loss + lambda_depth_order * compute_masked_depth_order_loss(
                    depth=surf_depth,
                    prior_depth=current_depth,
                    mask=current_geometry_mask,
                    scene_extent=gaussians.spatial_lr_scale,
                    max_pixel_shift_ratio=0.05,
                    normalize_loss=True,
                    log_space=True,
                    log_scale=20.0,
                    debug=False,
                )

            geometry_weight = 1.0
            if is_pseudo_view:
                geometry_weight = linear_weight(
                    iteration,
                    pseudo_geometry_weight,
                    pseudo_geometry_final_weight,
                    pseudo_geometry_decay_until,
                )
            depth_prior_loss = depth_prior_loss * geometry_weight
            normal_prior_loss = normal_prior_loss * geometry_weight
            curv_prior_loss = curv_prior_loss * geometry_weight
            lambda_anisotropy = 0.1
        elif dense_depth_priors is not None:
            lambda_dense_depth = dense_depth_weight(iteration, dense_regul)
            if lambda_dense_depth > 0:
                dense_prior = dense_depth_priors[dense_viewpoint_idx].float().to(surf_depth.device)
                depth_prior_loss = lambda_dense_depth * compute_masked_depth_order_loss(
                    depth=surf_depth,
                    prior_depth=dense_prior,
                    mask=current_geometry_mask,
                    scene_extent=gaussians.spatial_lr_scale,
                    max_pixel_shift_ratio=0.05,
                    normalize_loss=True,
                    log_space=True,
                    log_scale=20.0,
                    debug=False,
                )

        total_regularization_loss = depth_prior_loss + normal_prior_loss + curv_prior_loss
        if lambda_anisotropy > 0.0:
            gaussians_scaling = gaussians.get_scaling
            anisotropy_loss = lambda_anisotropy * (
                torch.clamp_min(gaussians_scaling.max(dim=1).values / gaussians_scaling.min(dim=1).values, anisotropy_max_ratio) 
                - anisotropy_max_ratio
            ).mean()
            total_regularization_loss = total_regularization_loss + anisotropy_loss

        total_loss = total_loss + total_regularization_loss
        
        # ===================================================================================

        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_prior_depth_for_log = 0.4 * depth_prior_loss.item() + 0.6 * ema_prior_depth_for_log
            ema_prior_normal_for_log = 0.4 * normal_prior_loss.item() + 0.6 * ema_prior_normal_for_log
            ema_prior_curvature_for_log = 0.4 * curv_prior_loss.item() + 0.6 * ema_prior_curvature_for_log
            ema_alpha_suppression_for_log = (
                0.4 * alpha_suppression_loss.item()
                + 0.6 * ema_alpha_suppression_for_log
            )
            if lambda_anisotropy > 0.:
                ema_prior_anisotropy_for_log = 0.4 * anisotropy_loss.item() + 0.6 * ema_prior_anisotropy_for_log

            if iteration % 10 == 0:

                current_points = len(gaussians.get_xyz.detach())
                gaussian_points_count.append(current_points)
                gaussian_points_iterations.append(iteration)

                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz.detach())}",
                    "p_depth": f"{ema_prior_depth_for_log:.{5}f}",
                    "p_normal": f"{ema_prior_normal_for_log:.{5}f}",
                    "pc": f"{ema_prior_curvature_for_log:.{5}f}",
                    "alpha": f"{ema_alpha_suppression_for_log:.{5}f}",
                }
                if lambda_anisotropy > 0:
                    loss_dict["aniso"] = f"{ema_prior_anisotropy_for_log:.{5}f}"
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
                
            if (iteration % save_log_images_every_n_iter == 0) or (iteration == 1):
                # Save log image with rgb, depth, normal, curvature
                if use_geometry_view:
                    supervision_depth = current_depth
                    supervision_normal = current_normal
                else:
                    supervision_depth = (
                        dense_depth_priors[dense_viewpoint_idx].float()
                        if dense_depth_priors is not None
                        else surf_depth.detach().cpu()
                    )
                    supervision_normal = surf_normal.detach()
                
                if save_log_images:
                    figsize = 30
                    height, width = gt_image.shape[-2:]
                    nrows = 2
                    ncols = 3
                    plt.figure(figsize=(figsize, figsize * height / width * nrows / ncols))
                    plt.subplot(nrows, ncols, 1)
                    plt.title("GT Image")
                    plt.imshow(gt_image.permute(1, 2, 0).clamp(0, 1).cpu().numpy())
                    plt.subplot(nrows, ncols, 2)
                    plt.title("Charts Depth")
                    plt.imshow(supervision_depth[0].cpu().numpy(), cmap="Spectral")
                    plt.colorbar()
                    plt.subplot(nrows, ncols, 3)
                    plt.title("Charts Normal")
                    plt.imshow((-supervision_normal + 1).permute(1, 2, 0).clamp(0, 2).cpu().numpy() / 2)
                    plt.subplot(nrows, ncols, 4)
                    plt.title("Rendered Image")
                    plt.imshow(image.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy())
                    plt.subplot(nrows, ncols, 5)
                    plt.title("Rendered Depth")
                    plt.imshow(surf_depth.detach()[0].cpu().numpy(), cmap="Spectral")
                    plt.colorbar()
                    plt.subplot(nrows, ncols, 6)
                    plt.title("Rendered Normal")
                    plt.imshow((-rend_normal.detach() + 1).permute(1, 2, 0).clamp(0, 2).cpu().numpy() / 2)
                    # save image
                    plt.savefig(f"{dataset.model_path}/{iteration}.png")
                    plt.close()
                
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar(
                    'train_loss_patches/invalid_region_alpha_loss',
                    ema_alpha_suppression_for_log,
                    iteration,
                )

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                    if gaussians.use_mip_filter:
                        gaussians.compute_mip_filter(
                            cameras=dense_viewpoint_cams if use_dense_supervision else total_views_list
                        )
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    
            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:  # don't update in the end of training
                    torch.cuda.empty_cache()
                    if gaussians.use_mip_filter:
                        gaussians.compute_mip_filter(
                            cameras=dense_viewpoint_cams if use_dense_supervision else total_views_list
                        )

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                if freeze_init_ply:
                    gaussians.constrain_trainable_suffix(
                        warmstart_baseline_count,
                        max_opacity=warmstart_max_opacity,
                        max_scale=warmstart_max_scale,
                        max_position_delta=warmstart_max_position_delta,
                        diffuse_only=warmstart_diffuse_only,
                        clamp_dc=warmstart_clamp_dc,
                    )
                if color_correction_optimizer is not None:
                    color_correction_optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if color_correction_optimizer is not None:
                    color_correction_optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

    if len(gaussian_points_count) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(gaussian_points_iterations, gaussian_points_count)
        plt.xlabel('Iterations')
        plt.ylabel('Number of Gaussian Points')
        plt.title('Gaussian Points Count During Training')
        plt.grid(True)
        plt.savefig(f"{dataset.model_path}/gaussian_points_count.png")
        plt.close()
    print("Training complete.")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--refine_depth_path", type=str, required=True)
    parser.add_argument("--use_downsample_gaussians", action="store_true", help="Use downsample gaussians")
    parser.add_argument("--downsample_gaussians_type", type=str, default="warp", choices=['warp', 'voxel'],
                        help="Downsample method used when --use_downsample_gaussians is set")
    parser.add_argument("--warp_depth_error_thresh", type=float, default=0.01,
                        help="Relative depth error threshold for warp-based Gaussian downsample")
    parser.add_argument("--warp_downsample_pixel_grid_size", type=int, default=-1,
                        help="Pixel grid stride for warp-based Gaussian initialization")
    parser.add_argument("--downweight_input_view_color_loss", action="store_true",
                        help="Also reduce color loss weight for input views; See3D views are always reduced")
    parser.add_argument("--cambridge_mask_pickle", type=str, default=None)
    parser.add_argument("--cambridge_mask_dataset_path", type=str, default=None)
    parser.add_argument("--cambridge_mask_indices", nargs="*", type=int, default=None)
    parser.add_argument("--cambridge_geometry_mask_pickle", type=str, default=None)
    parser.add_argument("--cambridge_geometry_mask_dataset_path", type=str, default=None)
    parser.add_argument("--cambridge_geometry_mask_indices", nargs="*", type=int, default=None)
    parser.add_argument("--cambridge_alpha_mask_indices", nargs="*", type=int, default=None)
    parser.add_argument("--semantic_alpha_weight", type=float, default=0.0)
    parser.add_argument("--cambridge_tree_mask_pickle", type=str, default=None)
    parser.add_argument("--cambridge_tree_mask_dataset_path", type=str, default=None)
    parser.add_argument("--cambridge_tree_mask_index", type=int, default=3)
    parser.add_argument("--cambridge_tree_support_dir", type=str, default=None)
    parser.add_argument("--tree_rgb_floor", type=float, default=0.25)
    parser.add_argument("--tree_rgb_support_gain", type=float, default=0.50)
    parser.add_argument("--tree_geometry_floor", type=float, default=0.05)
    parser.add_argument("--tree_geometry_support_gain", type=float, default=0.25)
    parser.add_argument("--tree_planar_weight", type=float, default=0.0)
    parser.add_argument("--tree_sky_feather", type=int, default=4)
    parser.add_argument("--tree_boundary_feather", type=int, default=6)
    parser.add_argument("--rgb_loss_type", choices=["l1", "charbonnier"], default="l1")
    parser.add_argument("--rgb_charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--dense_depth_cache", type=str, default=None)
    parser.add_argument("--geometry_view_every_n_iter", type=int, default=5)
    parser.add_argument("--dense_only_from_iter", type=int, default=3000)
    parser.add_argument("--pseudo_rgb_weight", type=float, default=0.01)
    parser.add_argument("--pseudo_geometry_weight", type=float, default=0.25)
    parser.add_argument("--pseudo_geometry_final_weight", type=float, default=0.02)
    parser.add_argument("--pseudo_geometry_decay_until", type=int, default=7000)
    parser.add_argument(
        "--pseudo_initialization_mode",
        choices=["all", "inpaint_only", "none"],
        default="all",
    )
    parser.add_argument(
        "--pseudo_geometry_mask_mode",
        choices=["all", "inpaint_only", "none"],
        default="all",
    )
    parser.add_argument("--max_plane_abs_depth", type=float, default=50.0)
    parser.add_argument(
        "--init_fill_unsupported_with_prior",
        action="store_true",
        help="Seed static alignment holes from scaled monocular prior depth for initialization only.",
    )
    parser.add_argument("--use_color_correction", action="store_true")
    parser.add_argument("--color_correction_lr", type=float, default=1e-3)
    parser.add_argument("--color_correction_reg", type=float, default=1e-2)
    parser.add_argument(
        "--init_ply",
        type=str,
        default=None,
        help="Warm-start from an existing MAtCha/2DGS PLY instead of the regenerated chart model.",
    )
    parser.add_argument(
        "--freeze_init_ply",
        action="store_true",
        help="Freeze warm-start points and optimize only appended hole reseeds.",
    )
    parser.add_argument(
        "--warmstart_reseed_pixel_stride",
        type=int,
        default=8,
        help="Pixel stride used when appending See3D hole Gaussians to a warm start.",
    )
    parser.add_argument(
        "--warmstart_reseed_max_scale",
        type=float,
        default=0.05,
        help="Maximum linear scale retained for appended warm-start Gaussians.",
    )
    parser.add_argument("--warmstart_max_opacity", type=float, default=1.0)
    parser.add_argument("--warmstart_max_scale", type=float, default=0.0)
    parser.add_argument("--warmstart_max_position_delta", type=float, default=0.0)
    parser.add_argument("--warmstart_diffuse_only", action="store_true")
    parser.add_argument("--warmstart_clamp_dc", action="store_true")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=None)  # 6009
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--use_refined_charts", action="store_true", default=False)
    parser.add_argument("--use_mip_filter", action="store_true", default=False)
    parser.add_argument("--dense_data_path", type=str, default=None)
    parser.add_argument("--use_chart_view_every_n_iter", type=int, default=999_999)
    parser.add_argument("--normal_consistency_from", type=int, default=3500)
    parser.add_argument("--distortion_from", type=int, default=1500)
    parser.add_argument('--depthanythingv2_checkpoint_dir', type=str, default='../Depth-Anything-V2/checkpoints/')
    parser.add_argument('--depthanything_encoder', type=str, default='vitl')
    parser.add_argument('--dense_regul', type=str, default='default', help='Dense depth schedule: default, strong, strong_decay, weak, or none.')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if args.port is None:
        import time
        current_time = time.strftime("%H%M%S", time.localtime())[2:]
        args.port = int(current_time)
        print(f"Randomly selected port: {args.port}")
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args), op.extract(args), pp.extract(args), 
        args.test_iterations, args.save_iterations, args.checkpoint_iterations, 
        args.start_checkpoint, args.use_refined_charts, args.use_mip_filter, 
        args.dense_data_path, args.use_chart_view_every_n_iter,
        args.normal_consistency_from, args.distortion_from,
        args.depthanythingv2_checkpoint_dir, args.depthanything_encoder,
        args.dense_regul, args.refine_depth_path, args.use_downsample_gaussians,
        args.downsample_gaussians_type, args.warp_depth_error_thresh, args.warp_downsample_pixel_grid_size,
        args.downweight_input_view_color_loss,
        args.cambridge_mask_pickle, args.cambridge_mask_dataset_path, args.cambridge_mask_indices,
        args.cambridge_geometry_mask_pickle, args.cambridge_geometry_mask_dataset_path,
        args.cambridge_geometry_mask_indices, args.rgb_loss_type, args.rgb_charbonnier_eps,
        args.dense_depth_cache, args.geometry_view_every_n_iter, args.dense_only_from_iter,
        args.pseudo_rgb_weight, args.pseudo_geometry_weight,
        args.pseudo_geometry_final_weight, args.pseudo_geometry_decay_until,
        args.pseudo_initialization_mode, args.pseudo_geometry_mask_mode,
        args.max_plane_abs_depth,
        args.init_fill_unsupported_with_prior,
        args.use_color_correction, args.color_correction_lr,
        args.color_correction_reg,
        args.cambridge_alpha_mask_indices, args.semantic_alpha_weight,
        args.cambridge_tree_mask_pickle, args.cambridge_tree_mask_dataset_path,
        args.cambridge_tree_mask_index, args.cambridge_tree_support_dir,
        args.tree_rgb_floor, args.tree_rgb_support_gain,
        args.tree_geometry_floor, args.tree_geometry_support_gain,
        args.tree_planar_weight, args.tree_sky_feather, args.tree_boundary_feather,
        args.init_ply, args.freeze_init_ply, args.warmstart_reseed_pixel_stride,
        args.warmstart_reseed_max_scale,
        args.warmstart_max_opacity, args.warmstart_max_scale,
        args.warmstart_max_position_delta, args.warmstart_diffuse_only,
        args.warmstart_clamp_dc,
    )

    # All done
    print("\nTraining complete.")
