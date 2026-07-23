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
import json
import argparse
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.getcwd(), '2d-gaussian-splatting'))
from scene.dataset_readers import load_see3d_cameras

import gc
import copy
import torch
import torch.nn.functional as F
from collections import Counter
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
    build_spatial_camera_blocks,
    compute_masked_depth_order_loss,
    compute_rgb_loss,
    dense_depth_weight,
    densification_stats_from_view,
    fused_geometry_validity_mask,
    fused_inverse_depth_nll,
    geometry_chart_sampling_indices,
    geometry_iteration,
    geometry_prior_schedule_weight,
    linear_weight,
    apply_per_image_affine_color_correction,
    load_per_image_affine_color_correction,
    load_depth_cache,
    masked_mean,
    opacity_reset_due,
    PerImageAffineColorCorrection,
    rgb_sampling_importance_weights,
    rgb_sampling_importance_weights_by_camera,
    resolve_active_chart_indices,
    rgb_supervision_weight,
    scale_chart_geometry_priors,
    sanitize_depth,
    save_per_image_affine_color_correction,
    save_depth_cache,
    ulfloc_masked_supervision,
    validate_chart_camera_order,
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
    rgb_supervision_profile="g4_tree_masked", rgb_sampling_policy="all_train_importance",
    chart_geometry_sampling_policy="active_only",
    densification_view_policy="legacy_current",
    continue_opacity_resets_after_densify=False,
    chart_geometry_prior_weight=1.0,
    dense_depth_cache=None, geometry_view_every_n_iter=5, dense_only_from_iter=3000,
    geometry_schedule="persistent", geometry_phase1_until=12_000,
    geometry_phase2_until=25_000, geometry_final_weight_floor=0.2,
    dense_view_sampling_policy="uniform", dense_view_block_bins=4,
    pseudo_rgb_weight=0.01, pseudo_geometry_weight=0.25,
    pseudo_geometry_final_weight=0.02, pseudo_geometry_decay_until=7000,
    pseudo_initialization_mode="all", pseudo_geometry_mask_mode="all",
    max_plane_abs_depth=None,
    init_fill_unsupported_with_prior=False,
    use_color_correction=False, color_correction_lr=1e-3,
    color_correction_reg=1e-2,
    cambridge_alpha_mask_indices=None, semantic_alpha_weight=0.0,
    cambridge_tree_mask_pickle=None, cambridge_tree_mask_dataset_path=None,
    cambridge_tree_mask_index=3, cambridge_tree_support_dir=None,
    tree_rgb_floor=0.25, tree_rgb_support_gain=0.50,
    tree_geometry_floor=0.05, tree_geometry_support_gain=0.25,
    tree_planar_weight=0.0, tree_sky_feather=4, tree_boundary_feather=6,
    tree_missing_support_policy="legacy_zero", tree_neutral_support_value=0.5,
    cambridge_task_semantic_policy="legacy", cambridge_task_semantic_manifest=None,
    init_ply=None, freeze_init_ply=False, warmstart_reseed_pixel_stride=8,
    warmstart_reseed_max_scale=0.05,
    warmstart_max_opacity=1.0, warmstart_max_scale=0.0,
    warmstart_max_position_delta=0.0, warmstart_diffuse_only=False,
    warmstart_clamp_dc=False,
    warmstart_preserve_topology=True,
    warmstart_allow_residual_densification=False,
    warmstart_residual_densification_mode="clone_only",
    warmstart_residual_densify_max_per_event=5_000,
    warmstart_residual_clone_opacity=0.02,
    warmstart_freeze_baseline=False,
    warmstart_residual_lr_restart=False,
):
    
    save_log_images = False
    save_log_images_every_n_iter = 200

    gaussian_points_count = []
    gaussian_points_iterations = []
    
    first_iter = 0
    # RGB importance weights must be conditioned on the iterations that this
    # process will actually execute.  A 7k -> 40k checkpoint continuation
    # begins at iteration 7001, not at the bootstrap, so its Chart/dense
    # sampler mixture differs from a fresh 40k run.
    sampling_start_iteration = 0
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
    if rgb_supervision_profile not in {
        "g4_tree_masked",
        "full_rgb",
        "ulfloc_legacy",
    }:
        raise ValueError(
            "rgb_supervision_profile must be 'g4_tree_masked', 'full_rgb', or "
            f"'ulfloc_legacy', got {rgb_supervision_profile!r}"
        )
    if rgb_supervision_profile == "ulfloc_legacy" and rgb_mask_lookup is None:
        raise ValueError(
            "ulfloc_legacy RGB supervision requires --cambridge_mask_pickle "
            "with object/sky/distortion channels [0, 1, 2]"
        )
    print(f"[INFO] Cambridge RGB supervision profile: {rgb_supervision_profile}")
    if chart_geometry_prior_weight < 0.0:
        raise ValueError("--chart-geometry-prior-weight must be non-negative")
    if geometry_schedule not in {"legacy_cutoff", "persistent"}:
        raise ValueError("--geometry-schedule must be legacy_cutoff or persistent")
    if geometry_phase1_until < 1 or geometry_phase2_until < geometry_phase1_until:
        raise ValueError("geometry phase bounds must satisfy 1 <= phase1 <= phase2")
    if not 0.0 <= geometry_final_weight_floor <= 1.0:
        raise ValueError("--geometry-final-weight-floor must lie in [0, 1]")
    if warmstart_residual_densification_mode not in {"clone_only", "split_only"}:
        raise ValueError(
            "--warmstart-residual-densification-mode must be clone_only or split_only"
        )
    print(
        "[INFO] Chart-exclusive geometry-prior weight: "
        f"{chart_geometry_prior_weight}."
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

    if cambridge_task_semantic_policy not in {"legacy", "outdoor_task_specific_v1"}:
        raise ValueError(
            "cambridge_task_semantic_policy must be legacy or outdoor_task_specific_v1"
        )
    if cambridge_task_semantic_policy == "outdoor_task_specific_v1":
        if cambridge_task_semantic_manifest is None:
            raise ValueError(
                "outdoor_task_specific_v1 requires --cambridge-task-semantic-manifest"
            )
        semantic_manifest_path = Path(cambridge_task_semantic_manifest)
        if not semantic_manifest_path.is_file():
            raise FileNotFoundError(semantic_manifest_path)
        semantic_manifest = json.loads(semantic_manifest_path.read_text(encoding="utf-8"))
        if semantic_manifest.get("schema_version") != "outdoor_task_specific_v1":
            raise RuntimeError(
                "Task semantic manifest does not declare outdoor_task_specific_v1"
            )
        print(
            "[INFO] Outdoor task-specific semantic fields enabled: "
            f"{semantic_manifest_path}."
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
            missing_support_policy=tree_missing_support_policy,
            neutral_support_value=tree_neutral_support_value,
        )
        print(
            "[INFO] Canonical-tree soft weighting enabled: "
            f"index={cambridge_tree_mask_index}, support={cambridge_tree_support_dir}, "
            f"rgb=({tree_rgb_floor}+{tree_rgb_support_gain}*S), "
            f"geometry=({tree_geometry_floor}+{tree_geometry_support_gain}*S), "
            f"planar={tree_planar_weight}, missing_support={tree_missing_support_policy}."
        )

    use_dense_supervision = dense_data_path is not None
    dense_viewpoint_cams = []
    dense_viewpoint_idx_stack = None
    dense_block_idx_stack = None
    dense_block_view_idx_stacks = []
    dense_view_blocks = []
    if dense_view_sampling_policy not in {"uniform", "spatial_block_balanced"}:
        raise ValueError(
            "dense_view_sampling_policy must be uniform or spatial_block_balanced"
        )
    if dense_view_sampling_policy == "spatial_block_balanced" and not use_dense_supervision:
        raise ValueError(
            "spatial_block_balanced dense sampling requires --dense_data_path"
        )
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
        if dense_view_sampling_policy == "spatial_block_balanced":
            dense_view_blocks = build_spatial_camera_blocks(
                torch.stack([
                    camera.camera_center.detach().cpu()
                    for camera in dense_viewpoint_cams
                ]),
                bins=dense_view_block_bins,
            )
            dense_block_view_idx_stacks = [None for _ in dense_view_blocks]
            print(
                "[INFO] Dense view sampling: spatial_block_balanced; "
                f"blocks={len(dense_view_blocks)}, bins={dense_view_block_bins}, "
                f"min/max cameras per block={min(map(len, dense_view_blocks))}/"
                f"{max(map(len, dense_view_blocks))}."
            )
        del dense_gaussians, dense_scene
        gc.collect()
        torch.cuda.empty_cache()

    # Keep the *realized* sampler trace separate from its analytic schedule.
    # The latter proves the intended expectation; the former catches an
    # implementation drift such as a broken block cycle or an accidental
    # Chart-only topology stream in an actual reconstruction run.
    dense_camera_block_by_name = {
        dense_viewpoint_cams[camera_index].image_name: int(block_index)
        for block_index, block in enumerate(dense_view_blocks)
        for camera_index in block
    }
    runtime_real_camera_names = sorted(
        camera.image_name
        for camera in (dense_viewpoint_cams if use_dense_supervision else input_cams)
    )
    runtime_rgb_samples = Counter()
    runtime_rgb_weighted_mass = Counter()
    runtime_rgb_samples_by_sequence = Counter()
    runtime_rgb_weighted_mass_by_sequence = Counter()
    runtime_rgb_samples_by_block = Counter()
    runtime_rgb_weighted_mass_by_block = Counter()
    runtime_geometry_samples = Counter()
    runtime_dense_samples = Counter()
    runtime_topology_samples = Counter()
    runtime_topology_samples_by_block = Counter()
    runtime_topology_samples_by_sequence = Counter()

    # A chart-only support directory used to be silently interpreted as zero
    # canonical support for every remaining dense camera.  Preserve that
    # conservative numerical fallback for reproducibility, but make the
    # missing-evidence condition explicit and auditable.
    if tree_weight_lookup is not None:
        support_audit_views = (
            dense_viewpoint_cams if use_dense_supervision else input_cams
        )
        tree_support_audit = tree_weight_lookup.support_coverage_audit(
            [camera.image_name for camera in support_audit_views]
        )
        tree_support_audit_path = os.path.join(
            dataset.model_path, "tree_support_coverage.json"
        )
        with open(tree_support_audit_path, "w", encoding="utf-8") as handle:
            json.dump(tree_support_audit, handle, indent=2)
        if tree_support_audit["missing_map_count"]:
            print(
                "[WARNING] Tree-support maps cover "
                f"{tree_support_audit['present_map_count']}/"
                f"{tree_support_audit['requested_view_count']} training cameras. "
                f"Missing maps follow explicit policy={tree_missing_support_policy}; "
                "unknown evidence is not interpreted as zero canonical support."
            )
        else:
            print(
                "[INFO] Explicit canonical-tree support maps cover every requested "
                "training camera."
            )
        if (
            tree_support_audit["present_map_count"]
            and tree_support_audit["nonzero_map_count"] == 0
        ):
            print(
                "[WARNING] Every explicit tree-support map is numerically zero; "
                "tree RGB/geometry weights therefore use their zero-support floors, "
                "not measured canonical support."
            )

    # NOTE: hard code for See3D root path
    see3d_root_path = os.path.join(dataset.source_path, 'see3d_render')
    see3d_cam_path = os.path.join(see3d_root_path, 'see3d_cameras.npz')
    inpaint_root_dir = os.path.join(see3d_root_path, 'inpainted_images')
    if os.path.exists(see3d_cam_path):
        see3d_gs_cameras_list, _ = load_see3d_cameras(see3d_cam_path, inpaint_root_dir)
    else:
        see3d_gs_cameras_list = []
    
    geometry_schedule_description = (
        f"persistent phases <= {geometry_phase1_until} / <= {geometry_phase2_until} / later "
        f"at every {geometry_view_every_n_iter}/{geometry_view_every_n_iter * 2}/"
        f"{geometry_view_every_n_iter * 4} iter(s), floor={geometry_final_weight_floor}"
        if geometry_schedule == "persistent"
        else f"legacy cutoff at iteration {dense_only_from_iter} every {geometry_view_every_n_iter} iter(s)"
    )
    print(
        f"[INFO] Plane/chart geometry schedule: {geometry_schedule_description}; "
        f"dense supervision enabled={use_dense_supervision}."
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
    # The outdoor fusion directory optionally carries per-pixel inverse-depth
    # uncertainty and source provenance.  Keep this optional so historical
    # plane-only experiments remain byte-for-byte compatible.
    pa_rho_variances_list = []
    pa_source_bitmasks_list = []
    fused_inverse_depth = os.path.isfile(
        os.path.join(refine_depth_path, "inverse_depth_fusion_manifest.json")
    )
    if fused_inverse_depth:
        print(
            "[INFO] Inverse-depth fusion detected; plane initialization validity "
            "will use plane/Chart source provenance while retaining continuous "
            "fusion confidence for loss weighting."
        )

    input_view_num = len(scene.getTrainCameras())
    see3d_view_num = len(see3d_gs_cameras_list)
    training_view_num = input_view_num + see3d_view_num
    if "depths" not in charts_data or charts_data["depths"].ndim < 1:
        raise RuntimeError("Aligned charts_data is missing a camera-indexed depths tensor")
    validate_chart_camera_order(
        dataset.source_path,
        [camera.image_name for camera in input_cams],
        chart_tensor_count=int(charts_data["depths"].shape[0]),
    )
    active_input_chart_indices = resolve_active_chart_indices(
        input_view_num,
        alignment_gate_valid=charts_data.get("alignment_gate_valid"),
        quality_selection_active=charts_data.get("quality_selection_active"),
    )
    geometry_input_chart_indices = geometry_chart_sampling_indices(
        input_view_num,
        active_chart_indices=active_input_chart_indices,
        policy=chart_geometry_sampling_policy,
    )
    active_input_chart_set = set(active_input_chart_indices)
    scheduler_audit = {
        "version": 3,
        "policy": chart_geometry_sampling_policy,
        "chart_geometry_prior_weight": float(chart_geometry_prior_weight),
        "input_chart_count": input_view_num,
        "active_chart_indices": active_input_chart_indices,
        "active_chart_names": [
            input_cams[index].image_name for index in active_input_chart_indices
        ],
        "scheduled_chart_indices": geometry_input_chart_indices,
        "scheduled_chart_names": [
            input_cams[index].image_name for index in geometry_input_chart_indices
        ],
        "alignment_gate_present": "alignment_gate_valid" in charts_data,
        "quality_selection_present": "quality_selection_active" in charts_data,
        "tree_support_coverage_path": (
            os.path.join(dataset.model_path, "tree_support_coverage.json")
            if tree_weight_lookup is not None
            else None
        ),
        "task_semantic_policy": cambridge_task_semantic_policy,
        "task_semantic_manifest": cambridge_task_semantic_manifest,
        "tree_missing_support_policy": tree_missing_support_policy,
    }
    if rgb_sampling_policy == "all_train_importance" and see3d_view_num > 0:
        raise ValueError(
            "all_train_importance requires See3D to be disabled because generated "
            "views have no all-real-camera sampling probability"
        )
    input_chart_names = {
        input_cams[index].image_name for index in geometry_input_chart_indices
    }
    if use_dense_supervision:
        dense_names = {camera.image_name for camera in dense_viewpoint_cams}
        missing_dense_chart_names = sorted(input_chart_names - dense_names)
        if missing_dense_chart_names:
            raise RuntimeError(
                "The all-train dense set is missing Chart camera(s), so RGB sampling "
                f"cannot be audited: {missing_dense_chart_names[:3]}"
            )
    chart_rgb_sampling_weight, non_chart_rgb_sampling_weight = rgb_sampling_importance_weights(
        policy=rgb_sampling_policy,
        total_iterations=opt.iterations,
        dense_view_count=len(dense_viewpoint_cams),
        chart_view_count=len(geometry_input_chart_indices),
        use_dense_supervision=use_dense_supervision,
        geometry_view_every_n_iter=geometry_view_every_n_iter,
        dense_only_from_iter=dense_only_from_iter,
        geometry_schedule=geometry_schedule,
        geometry_phase1_until=geometry_phase1_until,
        geometry_phase2_until=geometry_phase2_until,
    )
    dense_camera_names = [camera.image_name for camera in dense_viewpoint_cams]
    rgb_sampling_weight_by_dense_name = rgb_sampling_importance_weights_by_camera(
        policy=rgb_sampling_policy,
        total_iterations=opt.iterations,
        dense_camera_names=dense_camera_names,
        chart_camera_names=input_chart_names,
        use_dense_supervision=use_dense_supervision,
        geometry_view_every_n_iter=geometry_view_every_n_iter,
        dense_only_from_iter=dense_only_from_iter,
        dense_view_sampling_policy=dense_view_sampling_policy,
        dense_view_blocks=dense_view_blocks,
        geometry_schedule=geometry_schedule,
        geometry_phase1_until=geometry_phase1_until,
        geometry_phase2_until=geometry_phase2_until,
    )
    print(
        "[INFO] RGB sampling policy: "
        f"{rgb_sampling_policy}; chart_weight={chart_rgb_sampling_weight:.6f}, "
        f"non_chart_weight={non_chart_rgb_sampling_weight:.6f}; exact per-camera "
        f"range={min(rgb_sampling_weight_by_dense_name.values(), default=1.0):.6f}-"
        f"{max(rgb_sampling_weight_by_dense_name.values(), default=1.0):.6f}."
    )
    geometry_iteration_count = sum(
        geometry_iteration(
            iteration,
            use_dense_supervision=use_dense_supervision,
            every_n=geometry_view_every_n_iter,
            dense_only_from_iter=dense_only_from_iter,
            geometry_schedule=geometry_schedule,
            phase1_until=geometry_phase1_until,
            phase2_until=geometry_phase2_until,
        )
        for iteration in range(1, opt.iterations + 1)
    )
    # Densification is a separate sampling process from the RGB objective.
    # Its window is open on exactly ``iteration < densify_until_iter``.  Keep
    # an explicit audit of which rendered views may feed screen-space radii
    # and position-gradient statistics, otherwise a Chart-heavy geometry
    # schedule can silently become a Chart-heavy capacity allocator.
    topology_last_iteration = min(
        int(opt.iterations),
        max(int(opt.densify_until_iter) - 1, 0),
    )
    topology_geometry_iteration_count = sum(
        geometry_iteration(
            iteration,
            use_dense_supervision=use_dense_supervision,
            every_n=geometry_view_every_n_iter,
            dense_only_from_iter=dense_only_from_iter,
            geometry_schedule=geometry_schedule,
            phase1_until=geometry_phase1_until,
            phase2_until=geometry_phase2_until,
        )
        for iteration in range(1, topology_last_iteration + 1)
    )
    topology_dense_iteration_count = (
        topology_last_iteration - topology_geometry_iteration_count
    )
    opacity_reset_iterations = [
        iteration
        for iteration in range(1, int(opt.iterations) + 1)
        if opacity_reset_due(
            iteration,
            opacity_reset_interval=int(opt.opacity_reset_interval),
            densify_from_iter=int(opt.densify_from_iter),
            densify_until_iter=int(opt.densify_until_iter),
            white_background=bool(dataset.white_background),
            continue_after_densify=bool(continue_opacity_resets_after_densify),
        )
    ]
    topology_geometry_stats_iteration_count = (
        topology_geometry_iteration_count
        if densification_stats_from_view(
            policy=densification_view_policy,
            use_dense_supervision=use_dense_supervision,
            is_geometry_view=True,
        )
        else 0
    )
    topology_dense_stats_iteration_count = (
        topology_dense_iteration_count
        if densification_stats_from_view(
            policy=densification_view_policy,
            use_dense_supervision=use_dense_supervision,
            is_geometry_view=False,
        )
        else 0
    )
    expected_chart_topology_stats = (
        topology_geometry_stats_iteration_count / float(len(geometry_input_chart_indices))
    )
    expected_non_chart_topology_stats = None
    if use_dense_supervision:
        expected_chart_topology_stats += (
            topology_dense_stats_iteration_count / float(len(dense_viewpoint_cams))
        )
        expected_non_chart_topology_stats = (
            topology_dense_stats_iteration_count / float(len(dense_viewpoint_cams))
        )
    scheduler_audit["rgb_sampling"] = {
        "policy": rgb_sampling_policy,
        "dense_camera_count": len(dense_viewpoint_cams),
        "sampling_start_iteration": sampling_start_iteration,
        "executed_iteration_count": int(opt.iterations) - sampling_start_iteration,
        "geometry_iteration_count": geometry_iteration_count,
        "total_iteration_count": int(opt.iterations),
        "geometry_view_every_n_iter": int(geometry_view_every_n_iter),
        "dense_only_from_iter": int(dense_only_from_iter),
        "geometry_schedule": geometry_schedule,
        "geometry_phase1_until": int(geometry_phase1_until),
        "geometry_phase2_until": int(geometry_phase2_until),
        "geometry_final_weight_floor": float(geometry_final_weight_floor),
        "chart_importance_weight": chart_rgb_sampling_weight,
        "non_chart_importance_weight": non_chart_rgb_sampling_weight,
        "per_camera_importance_weight": rgb_sampling_weight_by_dense_name,
        "all_scheduled_chart_names_present_in_dense_set": (
            not use_dense_supervision or not missing_dense_chart_names
        ),
    }
    scheduler_audit["dense_view_sampling"] = {
        "policy": dense_view_sampling_policy,
        "spatial_block_bins": int(dense_view_block_bins),
        "occupied_block_count": len(dense_view_blocks),
        "views_per_block": [len(block) for block in dense_view_blocks],
        "sampling_contract": (
            "one_random_real_camera_per_occupied_block_per_cycle"
            if dense_view_sampling_policy == "spatial_block_balanced"
            else "uniform_without_replacement_over_all_real_cameras_per_cycle"
        ),
        "rgb_objective_correction": (
            "exact_per_camera_importance_weight_from_chart_dense_mixture"
            if rgb_sampling_policy == "all_train_importance"
            else "legacy_unweighted"
        ),
    }
    scheduler_audit["densification_stats"] = {
        "policy": densification_view_policy,
        "topology_last_iteration": topology_last_iteration,
        "topology_iteration_count": topology_last_iteration,
        "scheduled_geometry_iteration_count": topology_geometry_iteration_count,
        "scheduled_dense_iteration_count": topology_dense_iteration_count,
        "geometry_stats_iteration_count": topology_geometry_stats_iteration_count,
        "dense_stats_iteration_count": topology_dense_stats_iteration_count,
        "skipped_stats_iteration_count": (
            topology_last_iteration
            - topology_geometry_stats_iteration_count
            - topology_dense_stats_iteration_count
        ),
        "expected_stats_per_scheduled_chart": expected_chart_topology_stats,
        "expected_stats_per_non_chart_dense_camera": expected_non_chart_topology_stats,
    }
    scheduler_audit["opacity_resets"] = {
        "native_window_bounded": not bool(continue_opacity_resets_after_densify),
        "continue_after_densify": bool(continue_opacity_resets_after_densify),
        "densify_until_iter": int(opt.densify_until_iter),
        "opacity_reset_interval": int(opt.opacity_reset_interval),
        "opacity_cull": float(opt.opacity_cull),
        "iterations": opacity_reset_iterations,
    }
    scheduler_audit_path = os.path.join(dataset.model_path, "chart_geometry_scheduler.json")
    with open(scheduler_audit_path, "w", encoding="utf-8") as handle:
        json.dump(scheduler_audit, handle, indent=2)
        handle.write("\n")
    print(
        "[INFO] Chart geometry scheduler: "
        f"{chart_geometry_sampling_policy}; active={len(active_input_chart_indices)}/"
        f"{input_view_num}, scheduled_input_charts={len(geometry_input_chart_indices)}; "
        f"audit={scheduler_audit_path}."
    )
    print(
        "[INFO] Densification statistics: "
        f"{densification_view_policy}; topology_iters={topology_last_iteration}, "
        f"chart_stats={topology_geometry_stats_iteration_count}, "
        f"dense_stats={topology_dense_stats_iteration_count}, "
        f"expected_per_chart={expected_chart_topology_stats:.6f}, "
        "expected_per_non_chart="
        f"{expected_non_chart_topology_stats if expected_non_chart_topology_stats is not None else 'n/a'}."
    )
    print(
        "[INFO] Opacity reset schedule: "
        f"{opacity_reset_iterations}; "
        f"continue_after_densify={bool(continue_opacity_resets_after_densify)}."
    )
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

        rho_variance_path = os.path.join(refine_depth_path, f'rho_variance_frame{idx:06d}.npy')
        source_bitmask_path = os.path.join(refine_depth_path, f'source_bitmask_frame{idx:06d}.npy')
        if os.path.isfile(rho_variance_path):
            rho_variance = np.load(rho_variance_path)
            if rho_variance.shape != pa_depth.shape:
                raise RuntimeError(
                    f'Inverse-depth variance shape mismatch for frame {idx}: '
                    f'{rho_variance.shape} vs {tuple(pa_depth.shape)}'
                )
            pa_rho_variances_list.append(torch.from_numpy(rho_variance.astype(np.float32, copy=False)).cuda())
        else:
            pa_rho_variances_list.append(None)
        if os.path.isfile(source_bitmask_path):
            source_bitmask = np.load(source_bitmask_path)
            if source_bitmask.shape != pa_depth.shape:
                raise RuntimeError(
                    f'Inverse-depth source map shape mismatch for frame {idx}: '
                    f'{source_bitmask.shape} vs {tuple(pa_depth.shape)}'
                )
            pa_source_bitmasks_list.append(
                torch.from_numpy(source_bitmask.astype(np.uint8, copy=False)).cuda()
            )
        else:
            pa_source_bitmasks_list.append(None)

    plane_valid_masks = []
    for idx, (depth, confidence) in enumerate(zip(pa_depths, pa_confident_maps_list)):
        semantic_mask = None
        if idx < input_view_num and geometry_mask_lookup is not None:
            semantic_mask = geometry_mask_lookup.get_mask(
                input_cams[idx].image_name,
                depth.shape[-2:],
                depth.device,
            )
        validity_confidence = confidence
        if fused_inverse_depth:
            source_bitmask = pa_source_bitmasks_list[idx]
            if source_bitmask is None:
                raise RuntimeError(
                    "Inverse-depth fusion manifest requires source_bitmask_frame*.npy "
                    f"for frame {idx}."
                )
            validity_confidence = fused_geometry_validity_mask(source_bitmask).to(
                dtype=depth.dtype
            )
        clean_depth, valid_mask = sanitize_depth(
            depth,
            confidence=validity_confidence,
            semantic_mask=semantic_mask,
            max_abs_depth=max_plane_abs_depth,
        )
        pa_depths[idx] = clean_depth
        plane_valid_masks.append(valid_mask)
        if pa_rho_variances_list[idx] is not None:
            variance = pa_rho_variances_list[idx]
            pa_rho_variances_list[idx] = torch.where(
                valid_mask & torch.isfinite(variance) & (variance > 0.0),
                variance,
                torch.full_like(variance, float('inf')),
            )

    # The quality selector preserves input-camera order for provenance, so
    # enforce its hard exclusions again at the plane interface.  In
    # particular, ``--init_fill_unsupported_with_prior`` must never revive a
    # Chart that the selection intentionally removed.
    for idx in range(input_view_num):
        if idx in active_input_chart_set:
            continue
        pa_depths[idx] = torch.zeros_like(pa_depths[idx])
        pa_confident_maps_list[idx] = torch.zeros_like(pa_confident_maps_list[idx])
        plane_valid_masks[idx] = torch.zeros_like(plane_valid_masks[idx], dtype=torch.bool)
        if pa_rho_variances_list[idx] is not None:
            pa_rho_variances_list[idx] = torch.full_like(
                pa_rho_variances_list[idx], float('inf')
            )
        if pa_source_bitmasks_list[idx] is not None:
            pa_source_bitmasks_list[idx] = torch.zeros_like(pa_source_bitmasks_list[idx])

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
    active_input_valid_ratio = torch.stack(
        [plane_valid_masks[idx] for idx in active_input_chart_indices]
    ).float().mean().item()
    print(
        f"[INFO] Plane depth validity after confidence/outlier/semantic filtering: "
        f"{input_valid_ratio * 100:.2f}% over all aligned charts; "
        f"{active_input_valid_ratio * 100:.2f}% over active charts."
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
        for idx in active_input_chart_indices:
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
    checkpoint_continuation_requested = checkpoint is not None
    if warmstart_requested and checkpoint_continuation_requested:
        raise ValueError("Use either --init_ply or --start_checkpoint, not both")
    _images = (
        []
        if warmstart_requested or checkpoint_continuation_requested
        else [cam.original_image.cuda().permute(1, 2, 0) for cam in scene.getTrainCameras()]
    )
    # A warm start is already a complete structural model.  Never route it
    # through the old warp-downsample bootstrap: that path was responsible for
    # replacing a 7k PLY with a smaller reinitialised topology.
    use_warp_downsample = (
        not warmstart_requested
        and not checkpoint_continuation_requested
        and (
        use_downsample_gaussians and downsample_gaussians_type == "warp"
        )
    )
    voxel_max_init_gs_input_view_num = 50
    warp_max_init_gs_input_view_num = None
    max_init_gs_input_view_num = (
        0
        if warmstart_requested or checkpoint_continuation_requested
        else (
            warp_max_init_gs_input_view_num
            if use_warp_downsample
            else voxel_max_init_gs_input_view_num
        )
    )
    if (
        max_init_gs_input_view_num is not None
        and len(active_input_chart_indices) > max_init_gs_input_view_num
    ):
        print(
            f'[INFO]: Active input Chart num is too large: {len(active_input_chart_indices)}, '
            f'use {max_init_gs_input_view_num} views for gs initialization'
        )
        active_positions = np.linspace(
            0,
            len(active_input_chart_indices) - 1,
            max_init_gs_input_view_num,
            dtype=int,
        )
        init_view_ids = [active_input_chart_indices[position] for position in active_positions]
        init_input_view_depths = [input_view_depths[i] for i in init_view_ids]
        init_input_views = [scene.getTrainCameras()[i] for i in init_view_ids]
        init_images = [_images[i] for i in init_view_ids]
    else:
        init_view_ids = list(active_input_chart_indices)
        init_input_view_depths = [input_view_depths[i] for i in init_view_ids]
        init_input_views = [scene.getTrainCameras()[i] for i in init_view_ids]
        init_images = [_images[i] for i in init_view_ids]

    warp_init_depths = list(init_input_view_depths)
    warp_init_views = list(init_input_views)
    warp_init_valid_masks = [initialization_valid_masks[i] for i in init_view_ids]
    
    use_pseudo_initialization = (
        not checkpoint_continuation_requested
        and see3d_view_num > 0
        and pseudo_initialization_mode != "none"
    )
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
    if checkpoint_continuation_requested:
        print("[INFO] Full checkpoint continuation requested; skipping Gaussian reinitialization.")
    elif warmstart_requested and not use_pseudo_initialization:
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
            # ``init_input_views`` may be a hard-selected subset even when
            # the original Chart count is below the legacy 50-view cap.  Its
            # depth/image/mask lists must therefore use the same indices in
            # every branch; passing all masks here could attach an excluded
            # Chart mask to a selected Chart's depth map.
            visibility_masks=[initialization_valid_masks[i] for i in init_view_ids],
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
    elif not warmstart_requested and not checkpoint_continuation_requested:
        print(f"Final number of gaussians: {len(_means)}")
        gaussians.create_from_parameters(
            _means, _scales, _quaternions, _colors, gaussians.spatial_lr_scale
        )
        print("[INFO] Gaussians created from pnts data.")
        del _means, _scales, _quaternions, _colors
    gc.collect()
    torch.cuda.empty_cache()

    warmstart_baseline_count = 0
    warmstart_reseed_count = 0
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
                warmstart_reseed_count = int(reseed_count)
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
        checkpoint = os.path.abspath(os.path.expanduser(checkpoint))
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Continuation checkpoint does not exist: {checkpoint}")
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        first_iter = int(first_iter)
        sampling_start_iteration = first_iter
        warmstart_baseline_count = len(gaussians.get_xyz)
        # Recondition the all-real RGB importance correction on the remaining
        # continuation range.  Using the fresh-run 0..40k mixture here would
        # misweight every camera after a checkpoint because the persistent
        # geometry cadence changes across phases.
        chart_rgb_sampling_weight, non_chart_rgb_sampling_weight = (
            rgb_sampling_importance_weights(
                policy=rgb_sampling_policy,
                total_iterations=opt.iterations,
                dense_view_count=len(dense_viewpoint_cams),
                chart_view_count=len(geometry_input_chart_indices),
                use_dense_supervision=use_dense_supervision,
                geometry_view_every_n_iter=geometry_view_every_n_iter,
                dense_only_from_iter=dense_only_from_iter,
                geometry_schedule=geometry_schedule,
                geometry_phase1_until=geometry_phase1_until,
                geometry_phase2_until=geometry_phase2_until,
                start_iteration=sampling_start_iteration,
            )
        )
        rgb_sampling_weight_by_dense_name = rgb_sampling_importance_weights_by_camera(
            policy=rgb_sampling_policy,
            total_iterations=opt.iterations,
            dense_camera_names=dense_camera_names,
            chart_camera_names=input_chart_names,
            use_dense_supervision=use_dense_supervision,
            geometry_view_every_n_iter=geometry_view_every_n_iter,
            dense_only_from_iter=dense_only_from_iter,
            dense_view_sampling_policy=dense_view_sampling_policy,
            dense_view_blocks=dense_view_blocks,
            geometry_schedule=geometry_schedule,
            geometry_phase1_until=geometry_phase1_until,
            geometry_phase2_until=geometry_phase2_until,
            start_iteration=sampling_start_iteration,
        )
        geometry_iteration_count = sum(
            geometry_iteration(
                iteration,
                use_dense_supervision=use_dense_supervision,
                every_n=geometry_view_every_n_iter,
                dense_only_from_iter=dense_only_from_iter,
                geometry_schedule=geometry_schedule,
                phase1_until=geometry_phase1_until,
                phase2_until=geometry_phase2_until,
            )
            for iteration in range(sampling_start_iteration + 1, opt.iterations + 1)
        )
        scheduler_audit["rgb_sampling"].update({
            "sampling_start_iteration": sampling_start_iteration,
            "executed_iteration_count": int(opt.iterations) - sampling_start_iteration,
            "geometry_iteration_count": geometry_iteration_count,
            "chart_importance_weight": chart_rgb_sampling_weight,
            "non_chart_importance_weight": non_chart_rgb_sampling_weight,
            "per_camera_importance_weight": rgb_sampling_weight_by_dense_name,
        })
        print(
            "[INFO] Reconditioned RGB sampling weights for checkpoint continuation: "
            f"iterations {sampling_start_iteration + 1}-{opt.iterations}; "
            f"chart_weight={chart_rgb_sampling_weight:.6f}, "
            f"non_chart_weight={non_chart_rgb_sampling_weight:.6f}."
        )
    if freeze_init_ply:
        if warmstart_baseline_count <= 0:
            raise ValueError("--freeze_init_ply requires --init_ply")
        # restore() replaces the optimized tensors, so install hooks and capture
        # the suffix anchors only after a checkpoint has been restored.
        gaussians.freeze_prefix_gradients(warmstart_baseline_count)

    warmstart_topology_protected = bool(
        (warmstart_requested and warmstart_preserve_topology and not freeze_init_ply)
        or checkpoint_continuation_requested
    )
    warmstart_residual_densification_active = bool(
        warmstart_topology_protected and warmstart_allow_residual_densification
    )
    if warmstart_allow_residual_densification and not warmstart_topology_protected:
        raise ValueError(
            "--warmstart-allow-residual-densification requires a protected "
            "--init_ply or --start_checkpoint continuation"
        )
    if warmstart_residual_densification_active:
        if warmstart_residual_densify_max_per_event <= 0:
            raise ValueError(
                "--warmstart-residual-densify-max-per-event must be positive "
                "when residual densification is enabled"
            )
        if not 0.0 < warmstart_residual_clone_opacity < 1.0:
            raise ValueError(
                "--warmstart-residual-clone-opacity must lie strictly between 0 and 1"
            )
        # Checkpoints retain historic gradient accumulators.  A residual stage
        # must allocate only from observations made after continuation, rather
        # than replaying stale Chart-only allocation evidence at its first event.
        gaussians.xyz_gradient_accum = torch.zeros_like(gaussians.xyz_gradient_accum)
        gaussians.denom = torch.zeros_like(gaussians.denom)
        gaussians.max_radii2D = torch.zeros_like(gaussians.max_radii2D)
    if warmstart_freeze_baseline and not (
        checkpoint_continuation_requested and warmstart_residual_densification_active
    ):
        raise ValueError(
            "--warmstart-freeze-baseline requires a checkpoint residual continuation"
        )
    if warmstart_residual_lr_restart and not (
        checkpoint_continuation_requested
        and warmstart_residual_densification_active
        and warmstart_freeze_baseline
    ):
        raise ValueError(
            "--warmstart-residual-lr-restart requires a frozen checkpoint residual continuation"
        )
    warmstart_audit = {
        "schema_version": "warmstart-topology-audit-v3",
        "init_ply": init_ply,
        "continuation_checkpoint": checkpoint if checkpoint_continuation_requested else None,
        "continuation_kind": (
            "checkpoint" if checkpoint_continuation_requested
            else ("ply" if warmstart_requested else "none")
        ),
        "baseline_gaussian_count": int(warmstart_baseline_count),
        "reseed_gaussian_count": int(warmstart_reseed_count),
        "initial_total_gaussian_count": int(len(gaussians.get_xyz)),
        "preserve_topology": warmstart_topology_protected,
        "freeze_init_ply": bool(freeze_init_ply),
        "topology_operations": {
            "densify_and_prune": "disabled" if warmstart_topology_protected else "enabled",
            "additive_residual_clone": (
                "enabled"
                if warmstart_residual_densification_active
                and warmstart_residual_densification_mode == "clone_only"
                else "disabled"
            ),
            "additive_residual_split": (
                "enabled"
                if warmstart_residual_densification_active
                and warmstart_residual_densification_mode == "split_only"
                else "disabled"
            ),
            "opacity_reset": "disabled" if warmstart_topology_protected else "enabled",
        },
        "baseline_parameter_updates": (
            "frozen" if warmstart_freeze_baseline else "enabled"
        ),
        "residual_learning_rate_schedule": (
            "restarted_at_continuation" if warmstart_residual_lr_restart else "global_iteration"
        ),
        "residual_densification": {
            "enabled": warmstart_residual_densification_active,
            "mode": (
                "bounded_additive_" + warmstart_residual_densification_mode
                if warmstart_residual_densification_active
                else "disabled"
            ),
            "max_new_points_per_event": int(warmstart_residual_densify_max_per_event),
            "clone_opacity_ceiling": float(warmstart_residual_clone_opacity),
            "events": [],
            "total_gaussians_added": 0,
        },
    }
    if warmstart_requested or checkpoint_continuation_requested:
        print(
            "[INFO] Existing-topology protection: "
            f"{warmstart_topology_protected}; initial={warmstart_audit['initial_total_gaussian_count']} "
            f"(baseline={warmstart_baseline_count}, reseed={warmstart_reseed_count})."
        )
    if warmstart_residual_densification_active:
        print(
            "[INFO] Protected residual topology stage: additive "
            f"{warmstart_residual_densification_mode}; "
            f"max/event={warmstart_residual_densify_max_per_event}, "
            f"opacity<={warmstart_residual_clone_opacity}.",
            flush=True,
        )

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
    input_rho_variances = pa_rho_variances_list[:input_view_num]
    input_source_bitmasks = pa_source_bitmasks_list[:input_view_num]
    input_has_fused_uncertainty = all(value is not None for value in input_rho_variances)
    input_has_fused_sources = all(value is not None for value in input_source_bitmasks)
    if (
        cambridge_task_semantic_policy == "outdoor_task_specific_v1"
        and (not input_has_fused_uncertainty or not input_has_fused_sources)
    ):
        raise RuntimeError(
            "outdoor_task_specific_v1 requires rho_variance_frame*.npy and "
            "source_bitmask_frame*.npy for every real Chart; run inverse-depth fusion first."
        )
    if input_has_fused_uncertainty:
        input_rho_variances = torch.stack(input_rho_variances, dim=0).cuda()
    else:
        input_rho_variances = None
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
        total_rho_variances_list = [
            input_rho_variances[idx] if input_rho_variances is not None else None
            for idx in range(len(input_refine_depths))
        ] + [None for _ in range(len(see3d_refine_depths))]
        total_source_bitmasks_list = list(input_source_bitmasks) + [None for _ in range(len(see3d_refine_depths))]
        total_normals_list = [input_prior_normals[idx] for idx in range(len(input_prior_normals))] + [see3d_prior_normals[idx] for idx in range(len(see3d_prior_normals))]
        total_curvs_list = [input_prior_curvs[idx] for idx in range(len(input_prior_curvs))] + [see3d_prior_curvs[idx] for idx in range(len(see3d_prior_curvs))]
        total_geometry_masks_list = [input_geometry_masks[idx] for idx in range(len(input_geometry_masks))] + [see3d_geometry_masks[idx] for idx in range(len(see3d_geometry_masks))]
    else:
        total_views_list = input_cams
        total_confs_list = [input_pseudo_confs[idx] for idx in range(len(input_pseudo_confs))]
        total_depths_list = [input_refine_depths[idx] for idx in range(len(input_refine_depths))]
        total_rho_variances_list = [
            input_rho_variances[idx] if input_rho_variances is not None else None
            for idx in range(len(input_refine_depths))
        ]
        total_source_bitmasks_list = list(input_source_bitmasks)
        total_normals_list = [input_prior_normals[idx] for idx in range(len(input_prior_normals))]
        total_curvs_list = [input_prior_curvs[idx] for idx in range(len(input_prior_curvs))]
        total_geometry_masks_list = [input_geometry_masks[idx] for idx in range(len(input_geometry_masks))]

    geometry_view_indices = list(geometry_input_chart_indices)
    if use_pseudo_supervision:
        geometry_view_indices.extend(range(input_view_num, len(total_views_list)))
    if not geometry_view_indices:
        raise RuntimeError("No geometry-supervision views remain after Chart filtering")

    print(
        f"[INFO] Total number of supervised views: {len(total_views_list)}, "
        f"input views: {len(input_cams)}, enabled See3D views: "
        f"{len(see3d_gs_cameras_list) if use_pseudo_supervision else 0}/"
        f"{len(see3d_gs_cameras_list)}"
    )
    if cambridge_task_semantic_policy == "outdoor_task_specific_v1":
        fused_count = sum(value is not None for value in total_rho_variances_list)
        scheduler_audit["inverse_depth_fusion"] = {
            "loss": "variance_aware_inverse_depth_nll",
            "fused_variance_views": fused_count,
            "total_geometry_views": len(total_rho_variances_list),
            "mono_only_weight": 0.15,
            "source_bitmask_required_for_real_charts": True,
        }
        with open(scheduler_audit_path, "w", encoding="utf-8") as handle:
            json.dump(scheduler_audit, handle, indent=2)
            handle.write("\n")
        print(
            "[INFO] Outdoor inverse-depth likelihood: "
            f"variance/source evidence for {fused_count}/{len(total_rho_variances_list)} geometry view(s)."
        )

    color_correction = None
    color_correction_optimizer = None
    if use_color_correction:
        real_image_names = [camera.image_name for camera in input_cams]
        real_image_names.extend(camera.image_name for camera in dense_viewpoint_cams)
        continuation_color_path = None
        if checkpoint_continuation_requested:
            # A genuine checkpoint continuation must restore the appearance
            # state from the checkpoint's *own* model directory.  The former
            # code only looked under the new output path, forcing callers to
            # pre-copy a sidecar file and making an incomplete continuation
            # silently start with identity affine parameters.
            checkpoint_model_dir = os.path.dirname(checkpoint)
            candidate_paths = [
                os.path.join(
                    checkpoint_model_dir,
                    "point_cloud",
                    f"iteration_{first_iter}",
                    "color_correction.pth",
                ),
                # Keep support for an explicit sidecar staged at the new
                # output path by older launchers, but never require it.
                os.path.join(
                    dataset.model_path,
                    "point_cloud",
                    f"iteration_{first_iter}",
                    "color_correction.pth",
                ),
            ]
            for candidate in dict.fromkeys(candidate_paths):
                if not os.path.isfile(candidate):
                    continue
                restored = load_per_image_affine_color_correction(
                    candidate, device="cuda"
                )
                expected_names = set(dict.fromkeys(real_image_names))
                restored_names = set(restored.image_name_to_index)
                if restored_names != expected_names:
                    raise RuntimeError(
                        "Continuation color-correction camera contract differs from "
                        f"the current real RGB set: checkpoint={len(restored_names)}, "
                        f"current={len(expected_names)}"
                    )
                color_correction = restored.train()
                continuation_color_path = candidate
                print(
                    "[INFO] Restored per-image affine color correction from "
                    + candidate
                )
                break
        if color_correction is None:
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
        if continuation_color_path is not None:
            warmstart_audit["color_correction_checkpoint"] = continuation_color_path

    # Set the MIP state explicitly in both directions.  ``load_ply`` restores
    # an embedded filter when a warm-start PLY contains one; only setting this
    # flag in the true branch would therefore make a later ``--no MIP``
    # ablation silently keep using the inherited renderer filter.
    gaussians.set_mip_filter(bool(use_mip_filter))
    if use_mip_filter:
        print("[INFO] Using mip filter during training.")
        gaussians.compute_mip_filter(cameras=dense_viewpoint_cams if use_dense_supervision else total_views_list)
    if warmstart_freeze_baseline:
        gaussians.freeze_prefix_gradients(warmstart_baseline_count)
        warmstart_audit["baseline_optimizer_state"] = {
            "prefix_momentum": "cleared",
            "cleared_fields_by_group": getattr(
                gaussians, "_frozen_prefix_optimizer_state", {}
            ),
        }

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

        learning_rate_iteration = (
            iteration - sampling_start_iteration
            if warmstart_residual_lr_restart
            else iteration
        )
        gaussians.update_learning_rate(learning_rate_iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        use_geometry_view = geometry_iteration(
            iteration,
            use_dense_supervision=use_dense_supervision,
            every_n=geometry_view_every_n_iter,
            dense_only_from_iter=dense_only_from_iter,
            geometry_schedule=geometry_schedule,
            phase1_until=geometry_phase1_until,
            phase2_until=geometry_phase2_until,
        )
        if use_geometry_view:
            if not viewpoint_idx_stack:
                viewpoint_idx_stack = list(geometry_view_indices)
            viewpoint_idx = viewpoint_idx_stack.pop(randint(0, len(viewpoint_idx_stack) - 1))
            viewpoint_cam = total_views_list[viewpoint_idx]
            dense_viewpoint_idx = None
            is_pseudo_view = viewpoint_idx >= input_view_num
            current_geometry_mask = total_geometry_masks_list[viewpoint_idx]
        else:
            if dense_view_sampling_policy == "spatial_block_balanced":
                if not dense_block_idx_stack:
                    dense_block_idx_stack = list(range(len(dense_view_blocks)))
                block_idx = dense_block_idx_stack.pop(
                    randint(0, len(dense_block_idx_stack) - 1)
                )
                if not dense_block_view_idx_stacks[block_idx]:
                    dense_block_view_idx_stacks[block_idx] = list(dense_view_blocks[block_idx])
                dense_viewpoint_idx = dense_block_view_idx_stacks[block_idx].pop(
                    randint(0, len(dense_block_view_idx_stacks[block_idx]) - 1)
                )
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

        # Correct the *actual* RGB sampler.  In spatial block balancing a
        # non-Chart camera's probability depends on its block population, so
        # a chart/non-chart scalar alone would bias the all-real objective.
        rgb_sampling_weight = (
            1.0
            if is_pseudo_view
            else rgb_sampling_weight_by_dense_name.get(
                viewpoint_cam.image_name,
                1.0,
            )
        )
        if not is_pseudo_view:
            sampled_name = viewpoint_cam.image_name
            sampled_sequence = sampled_name.split("__", 1)[0]
            sampled_block = dense_camera_block_by_name.get(sampled_name)
            runtime_rgb_samples[sampled_name] += 1
            runtime_rgb_weighted_mass[sampled_name] += float(rgb_sampling_weight)
            runtime_rgb_samples_by_sequence[sampled_sequence] += 1
            runtime_rgb_weighted_mass_by_sequence[sampled_sequence] += float(
                rgb_sampling_weight
            )
            if sampled_block is not None:
                runtime_rgb_samples_by_block[sampled_block] += 1
                runtime_rgb_weighted_mass_by_block[sampled_block] += float(
                    rgb_sampling_weight
                )
            if use_geometry_view:
                runtime_geometry_samples[sampled_name] += 1
            else:
                runtime_dense_samples[sampled_name] += 1
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        rgb_mask = None
        current_tree_geometry_weight = None
        current_tree_planar_weight = None
        if tree_weight_lookup is not None and not is_pseudo_view:
            tree_rgb_weight, current_tree_geometry_weight, current_tree_planar_weight = (
                tree_weight_lookup.weights(
                    viewpoint_cam.image_name,
                    gt_image.shape[-2:],
                    gt_image.device,
                )
            )
            if rgb_supervision_profile == "g4_tree_masked":
                rgb_mask = tree_rgb_weight
        elif (
            rgb_supervision_profile == "g4_tree_masked"
            and rgb_mask_lookup is not None
            and not is_pseudo_view
        ):
            rgb_mask = rgb_mask_lookup.get_mask(
                viewpoint_cam.image_name,
                gt_image.shape[-2:],
                gt_image.device,
            )
        image_for_rgb_loss = image
        if color_correction is not None and not is_pseudo_view:
            image_for_rgb_loss = apply_per_image_affine_color_correction(
                color_correction, image, viewpoint_cam.image_name
            )
        if rgb_supervision_profile == "ulfloc_legacy" and not is_pseudo_view:
            object_mask = rgb_mask_lookup.get_index_mask(
                viewpoint_cam.image_name, 0, gt_image.shape[-2:], gt_image.device
            )
            sky_mask = rgb_mask_lookup.get_index_mask(
                viewpoint_cam.image_name, 1, gt_image.shape[-2:], gt_image.device
            )
            distortion_mask = rgb_mask_lookup.get_index_mask(
                viewpoint_cam.image_name, 2, gt_image.shape[-2:], gt_image.device
            )
            legacy_image, legacy_target, _ = ulfloc_masked_supervision(
                image_for_rgb_loss,
                gt_image,
                object_mask=object_mask,
                sky_mask=sky_mask,
                distortion_mask=distortion_mask,
            )
            Ll1 = l1_loss(legacy_image, legacy_target)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                1.0 - ssim(legacy_image, legacy_target)
            )
        else:
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
        ) * rgb_sampling_weight

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
            alpha_suppression_loss = rgb_sampling_weight * semantic_alpha_weight * masked_mean(
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
            current_rho_variance = total_rho_variances_list[viewpoint_idx]
            current_source_bitmask = total_source_bitmasks_list[viewpoint_idx]
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

            if (
                cambridge_task_semantic_policy == "outdoor_task_specific_v1"
                and current_rho_variance is not None
            ):
                # The fused outdoor evidence is calibrated in inverse depth.
                # Its source map preserves the intended hierarchy: bounded
                # planes and aligned Charts dominate; mono-only support is a
                # weak fallback rather than an equally strong anchor.
                depth_prior_loss = lambda_prior_depth * fused_inverse_depth_nll(
                    surf_depth,
                    current_depth,
                    current_rho_variance,
                    confidence=confidence_weight * current_geometry_mask,
                    source_bitmask=current_source_bitmask,
                )
            else:
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

        if use_geometry_view:
            (
                depth_prior_loss,
                normal_prior_loss,
                curv_prior_loss,
                anisotropy_loss,
            ) = scale_chart_geometry_priors(
                (
                    depth_prior_loss,
                    normal_prior_loss,
                    curv_prior_loss,
                    anisotropy_loss,
                ),
                weight=(
                    chart_geometry_prior_weight
                    * geometry_prior_schedule_weight(
                        iteration,
                        geometry_schedule=geometry_schedule,
                        phase1_until=geometry_phase1_until,
                        phase2_until=geometry_phase2_until,
                        final_weight_floor=geometry_final_weight_floor,
                    )
                ),
            )
            total_regularization_loss = (
                depth_prior_loss
                + normal_prior_loss
                + curv_prior_loss
                + anisotropy_loss
            )

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

            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                report_train_cameras=(
                    dense_viewpoint_cams if use_dense_supervision else None
                ),
                report_train_name=(
                    "dense_train_samples" if use_dense_supervision else "chart_train_samples"
                ),
                color_correction=color_correction,
            )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if color_correction is not None:
                    save_per_image_affine_color_correction(
                        color_correction,
                        os.path.join(
                            scene.model_path,
                            "point_cloud",
                            f"iteration_{iteration}",
                            "color_correction.pth",
                        ),
                    )


            # Densification
            topology_densification_active = (
                iteration < opt.densify_until_iter
                and (
                    not warmstart_topology_protected
                    or warmstart_residual_densification_active
                )
            )
            if topology_densification_active:
                update_densification_stats = densification_stats_from_view(
                    policy=densification_view_policy,
                    use_dense_supervision=use_dense_supervision,
                    is_geometry_view=use_geometry_view,
                )
                if update_densification_stats:
                    if not is_pseudo_view:
                        topology_name = viewpoint_cam.image_name
                        topology_sequence = topology_name.split("__", 1)[0]
                        topology_block = dense_camera_block_by_name.get(topology_name)
                        runtime_topology_samples[topology_name] += 1
                        runtime_topology_samples_by_sequence[topology_sequence] += 1
                        if topology_block is not None:
                            runtime_topology_samples_by_block[topology_block] += 1
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    gaussians.add_densification_stats(
                        viewspace_point_tensor,
                        visibility_filter,
                    )

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if warmstart_residual_densification_active:
                        residual_grads = gaussians.xyz_gradient_accum / gaussians.denom
                        if warmstart_residual_densification_mode == "clone_only":
                            added_count = gaussians.densify_and_clone_limited(
                                residual_grads,
                                opt.densify_grad_threshold,
                                scene.cameras_extent,
                                warmstart_residual_densify_max_per_event,
                                opacity_ceiling=warmstart_residual_clone_opacity,
                            )
                        else:
                            added_count = gaussians.densify_and_split_limited(
                                residual_grads,
                                opt.densify_grad_threshold,
                                scene.cameras_extent,
                                warmstart_residual_densify_max_per_event,
                                opacity_ceiling=warmstart_residual_clone_opacity,
                            )
                        residual_audit = warmstart_audit["residual_densification"]
                        residual_audit["events"].append(
                            {
                                "iteration": int(iteration),
                                "mode": warmstart_residual_densification_mode,
                                "added_gaussian_count": int(added_count),
                                "total_gaussian_count": int(len(gaussians.get_xyz)),
                            }
                        )
                        residual_audit["total_gaussians_added"] += int(added_count)
                        print(
                            "[INFO] Protected residual densification at "
                            f"iteration {iteration} ({warmstart_residual_densification_mode}): "
                            f"+{added_count}, "
                            f"total={len(gaussians.get_xyz)}.",
                            flush=True,
                        )
                    else:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold,
                            opt.opacity_cull,
                            scene.cameras_extent,
                            size_threshold,
                        )
                    if gaussians.use_mip_filter and not warmstart_residual_densification_active:
                        gaussians.compute_mip_filter(
                            cameras=dense_viewpoint_cams if use_dense_supervision else total_views_list
                        )
                
            if (
                not warmstart_topology_protected
                and opacity_reset_due(
                iteration,
                opacity_reset_interval=int(opt.opacity_reset_interval),
                densify_from_iter=int(opt.densify_from_iter),
                densify_until_iter=int(opt.densify_until_iter),
                white_background=bool(dataset.white_background),
                continue_after_densify=bool(continue_opacity_resets_after_densify),
                )
            ):
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

    def _counter_payload(counter, *, names=None):
        if names is None:
            items = counter.items()
        else:
            items = ((name, counter[name]) for name in names)
        return {
            str(key): value
            for key, value in sorted(items, key=lambda item: str(item[0]))
        }

    # Rewrite the scheduler audit with empirical counts once training is
    # complete.  It deliberately includes zero-count real cameras: omitting
    # them would make a failed block cycle indistinguishable from a camera
    # that was merely not listed in the report.
    scheduler_audit["runtime_sampling"] = {
        "contract": "actual_rendered_real_view_counts_and_importance_weighted_rgb_mass",
        "sampling_start_iteration": sampling_start_iteration,
        "executed_iteration_count": int(opt.iterations) - sampling_start_iteration,
        "rgb": {
            "actual_sample_count_by_camera": _counter_payload(
                runtime_rgb_samples, names=runtime_real_camera_names
            ),
            "effective_weighted_mass_by_camera": _counter_payload(
                runtime_rgb_weighted_mass, names=runtime_real_camera_names
            ),
            "actual_sample_count_by_block": _counter_payload(runtime_rgb_samples_by_block),
            "effective_weighted_mass_by_block": _counter_payload(
                runtime_rgb_weighted_mass_by_block
            ),
            "actual_sample_count_by_sequence": _counter_payload(
                runtime_rgb_samples_by_sequence
            ),
            "effective_weighted_mass_by_sequence": _counter_payload(
                runtime_rgb_weighted_mass_by_sequence
            ),
        },
        "geometry": {
            "actual_sample_count_by_camera": _counter_payload(
                runtime_geometry_samples, names=runtime_real_camera_names
            ),
        },
        "dense": {
            "actual_sample_count_by_camera": _counter_payload(
                runtime_dense_samples, names=runtime_real_camera_names
            ),
        },
        "topology": {
            "actual_stats_sample_count_by_camera": _counter_payload(
                runtime_topology_samples, names=runtime_real_camera_names
            ),
            "actual_stats_sample_count_by_block": _counter_payload(
                runtime_topology_samples_by_block
            ),
            "actual_stats_sample_count_by_sequence": _counter_payload(
                runtime_topology_samples_by_sequence
            ),
        },
    }
    with open(scheduler_audit_path, "w", encoding="utf-8") as handle:
        json.dump(scheduler_audit, handle, indent=2)
        handle.write("\n")

    if len(gaussian_points_count) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(gaussian_points_iterations, gaussian_points_count)
        plt.xlabel('Iterations')
        plt.ylabel('Number of Gaussian Points')
        plt.title('Gaussian Points Count During Training')
        plt.grid(True)
        plt.savefig(f"{dataset.model_path}/gaussian_points_count.png")
        plt.close()
    if warmstart_requested or checkpoint_continuation_requested:
        final_count = int(len(gaussians.get_xyz))
        warmstart_audit.update(
            {
                "final_total_gaussian_count": final_count,
                "net_gaussian_count_change": final_count - warmstart_audit["initial_total_gaussian_count"],
                "baseline_gaussians_removed": max(
                    0,
                    warmstart_baseline_count - final_count,
                ),
            }
        )
        audit_path = os.path.join(dataset.model_path, "continuation_topology_audit.json")
        with open(audit_path, "w", encoding="utf-8") as handle:
            json.dump(warmstart_audit, handle, indent=2)
            handle.write("\n")
        if warmstart_topology_protected and final_count < warmstart_baseline_count:
            raise RuntimeError(
                "Warm-start topology protection lost baseline Gaussians; inspect audit "
                + audit_path
            )
        print(f"[INFO] Continuation topology audit: {audit_path}")
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
def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    *,
    report_train_cameras=None,
    report_train_name="chart_train_samples",
    color_correction=None,
):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and a small, explicitly-labelled training sample.  With
    # dense RGB supervision, ``scene`` contains only the Chart cameras, so
    # silently calling that sample "train" makes it look like an all-train
    # metric even though it is not representative of the 1,487 real views.
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        train_cameras = (
            scene.getTrainCameras()
            if report_train_cameras is None
            else report_train_cameras
        )
        train_samples = [
            train_cameras[idx % len(train_cameras)]
            for idx in range(5, 30, 5)
        ] if train_cameras else []
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': report_train_name, 'cameras': train_samples},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = apply_per_image_affine_color_correction(
                        color_correction,
                        render_pkg["render"],
                        viewpoint.image_name,
                    )
                    image = torch.clamp(image, 0.0, 1.0)
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
    parser.add_argument(
        "--tree-missing-support-policy",
        choices=["error", "neutral", "legacy_zero"],
        default="legacy_zero",
    )
    parser.add_argument("--tree-neutral-support-value", type=float, default=0.5)
    parser.add_argument(
        "--cambridge-task-semantic-policy",
        choices=["legacy", "outdoor_task_specific_v1"],
        default="legacy",
    )
    parser.add_argument("--cambridge-task-semantic-manifest", type=str, default=None)
    parser.add_argument("--rgb_loss_type", choices=["l1", "charbonnier"], default="l1")
    parser.add_argument("--rgb_charbonnier_eps", type=float, default=1e-3)
    parser.add_argument(
        "--rgb-supervision-profile",
        choices=["g4_tree_masked", "full_rgb", "ulfloc_legacy"],
        default="g4_tree_masked",
        help=(
            "g4_tree_masked keeps the existing semantic/tree-weighted objective; "
            "full_rgb supervises every real RGB pixel while leaving geometry and alpha "
            "terms unchanged; ulfloc_legacy changes RGB pixels only to ULF-Loc's "
            "object/distortion/sky protocol."
        ),
    )
    parser.add_argument(
        "--rgb-sampling-policy",
        choices=["legacy_interleaved", "all_train_importance"],
        default="all_train_importance",
        help=(
            "Whether RGB losses preserve the legacy Chart oversampling or use "
            "importance weights whose expectation is uniform over all real train cameras."
        ),
    )
    parser.add_argument(
        "--chart-geometry-sampling-policy",
        choices=["legacy_all_input", "active_only"],
        default="active_only",
        help=(
            "Whether chart geometry iterations sample every aligned camera (legacy) "
            "or only Charts that survived gate and quality filtering."
        ),
    )
    parser.add_argument(
        "--densification-view-policy",
        choices=["legacy_current", "dense_only"],
        default="legacy_current",
        help=(
            "Which rendered views may update 2DGS split/prune statistics. "
            "dense_only retains Chart geometry losses but prevents their "
            "oversampling from allocating disproportionate topology."
        ),
    )
    parser.add_argument(
        "--chart-geometry-prior-weight",
        type=float,
        default=1.0,
        help=(
            "Common multiplier for Chart-exclusive aligned depth/normal/curvature/"
            "depth-order/anisotropy priors. Zero preserves the Chart camera schedule "
            "but removes those extra geometry residuals for a strict ablation."
        ),
    )
    parser.add_argument("--dense_depth_cache", type=str, default=None)
    parser.add_argument("--geometry_view_every_n_iter", type=int, default=5)
    parser.add_argument("--dense_only_from_iter", type=int, default=3000)
    parser.add_argument(
        "--geometry-schedule",
        choices=["legacy_cutoff", "persistent"],
        default="persistent",
        help=(
            "Persistent keeps Chart geometry active with decreasing cadence after "
            "the bootstrap; legacy_cutoff reproduces dense_only_from_iter behavior."
        ),
    )
    parser.add_argument("--geometry-phase1-until", type=int, default=12_000)
    parser.add_argument("--geometry-phase2-until", type=int, default=25_000)
    parser.add_argument("--geometry-final-weight-floor", type=float, default=0.2)
    parser.add_argument(
        "--dense-view-sampling-policy",
        choices=["uniform", "spatial_block_balanced"],
        default="uniform",
        help="Sampling policy for all-real dense cameras.",
    )
    parser.add_argument("--dense-view-block-bins", type=int, default=4)
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
    parser.add_argument(
        "--max_plane_abs_depth",
        type=float,
        default=None,
        help="Optional legacy absolute depth cap. The outdoor mainline leaves it unset.",
    )
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
    parser.add_argument(
        "--warmstart-preserve-topology",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Preserve every Gaussian in an init PLY by disabling prune/reset and "
            "topology rewrites during warm-start refinement."
        ),
    )
    parser.add_argument(
        "--warmstart-allow-residual-densification",
        action="store_true",
        help=(
            "For a protected continuation only, append bounded low-opacity residual "
            "primitives from post-continuation gradients; never prune or reset baseline points."
        ),
    )
    parser.add_argument(
        "--warmstart-residual-densification-mode",
        choices=["clone_only", "split_only"],
        default="clone_only",
        help=(
            "Use small-parent clones or compact children of large parents for a "
            "protected residual topology stage."
        ),
    )
    parser.add_argument(
        "--warmstart-residual-densify-max-per-event",
        type=int,
        default=5_000,
        help="Maximum residual Gaussians appended at one protected densification event.",
    )
    parser.add_argument(
        "--warmstart-residual-clone-opacity",
        type=float,
        default=0.02,
        help="Upper bound on initial opacity of a protected residual primitive.",
    )
    parser.add_argument(
        "--warmstart-freeze-baseline",
        action="store_true",
        help=(
            "Freeze the checkpoint prefix parameters so only later residual "
            "Gaussians are optimized."
        ),
    )
    parser.add_argument(
        "--warmstart-residual-lr-restart",
        action="store_true",
        help=(
            "Restart the Gaussian learning-rate clock at a frozen checkpoint "
            "continuation so appended residual primitives can move."
        ),
    )
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
    parser.add_argument(
        "--continue-opacity-resets-after-densify",
        action="store_true",
        help=(
            "Experimental causal-control switch: keep the opacity reset cadence "
            "after topology densification stops. Default preserves native 2DGS/"
            "ULF/STDLoc behavior, where reset ends with densification."
        ),
    )
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
        args.rgb_supervision_profile, args.rgb_sampling_policy,
        args.chart_geometry_sampling_policy,
        args.densification_view_policy,
        args.continue_opacity_resets_after_densify,
        args.chart_geometry_prior_weight,
        args.dense_depth_cache, args.geometry_view_every_n_iter, args.dense_only_from_iter,
        args.geometry_schedule, args.geometry_phase1_until,
        args.geometry_phase2_until, args.geometry_final_weight_floor,
        args.dense_view_sampling_policy, args.dense_view_block_bins,
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
        args.tree_missing_support_policy, args.tree_neutral_support_value,
        args.cambridge_task_semantic_policy, args.cambridge_task_semantic_manifest,
        args.init_ply, args.freeze_init_ply, args.warmstart_reseed_pixel_stride,
        args.warmstart_reseed_max_scale,
        args.warmstart_max_opacity, args.warmstart_max_scale,
        args.warmstart_max_position_delta, args.warmstart_diffuse_only,
        args.warmstart_clamp_dc,
        args.warmstart_preserve_topology,
        args.warmstart_allow_residual_densification,
        args.warmstart_residual_densification_mode,
        args.warmstart_residual_densify_max_per_event,
        args.warmstart_residual_clone_opacity,
        args.warmstart_freeze_baseline,
        args.warmstart_residual_lr_restart,
    )

    # All done
    print("\nTraining complete.")
