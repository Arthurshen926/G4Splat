import os
import numpy as np
import torch

from matcha.dm_scene.parallel_aligner import (
    ParallelAligner,
    MLPParams,
    ChartsEncodingParams,
    DepthEncodingParams,
    MultiResChartsEncodingParams,
)
from matcha.dm_scene.meshes import get_manifold_meshes_from_pointmaps
from matcha.dm_scene.cameras import CamerasWrapper, create_gs_cameras_from_pointmap, rescale_cameras
from matcha.pointmap.mast3r import load_mast3r_matches


def restore_unaligned_geometry(
    output_verts,
    output_depths,
    output_confs,
    initial_verts,
    initial_depths,
    masks,
):
    """Restore the pointmap prior where alignment has no supervision.

    Alignment masks identify pixels where a deformation is constrained; they
    are not visibility masks. Deleting confidence outside them removes useful
    MASt3R geometry and creates coverage holes. Use aligned geometry inside the
    mask and the original pointmap everywhere else.
    """
    if masks is None:
        return output_verts, output_depths, output_confs

    valid = masks.to(device=output_depths.device, dtype=torch.bool)
    if valid.shape != output_depths.shape:
        raise ValueError(
            f"Alignment mask shape {tuple(valid.shape)} does not match depth "
            f"shape {tuple(output_depths.shape)}"
        )
    initial_verts = initial_verts.reshape(*output_depths.shape, 3)
    output_verts = output_verts.reshape(*output_depths.shape, 3)
    output_verts = torch.where(valid[..., None], output_verts, initial_verts)
    output_depths = torch.where(valid, output_depths, initial_depths)
    return output_verts, output_depths, output_confs


@torch.no_grad()
def reject_catastrophic_alignment_confidences(
    output_confs,
    output_depths,
    prior_depths,
    masks=None,
    *,
    reference_depths=None,
    median_relative_error_threshold=0.75,
    bad_pixel_relative_error=0.25,
    bad_pixel_fraction_threshold=0.90,
    minimum_valid_pixels=64,
    return_reference_metrics=False,
):
    """Disable a Chart only when it contradicts its alignment target.

    ``prior_depths`` is the DepthAnything initialization.  A large change from
    that initialization is not itself evidence of failure: the alignment loss
    is explicitly trying to bring it toward the MASt3R observation.  When the
    latter is supplied through ``reference_depths``, use it as the rejection
    target and retain the prior-relative measurements as diagnostics only.

    The legacy prior-only behaviour remains available for old archives that do
    not contain a persisted MASt3R target.
    """
    confidences = output_confs.clone()
    rejected = torch.zeros(
        output_depths.shape[0],
        dtype=torch.bool,
        device=output_depths.device,
    )
    relative_medians = torch.full(
        (output_depths.shape[0],),
        float("nan"),
        dtype=output_depths.dtype,
        device=output_depths.device,
    )
    bad_fractions = torch.full_like(relative_medians, float("nan"))
    reference_relative_medians = torch.full_like(relative_medians, float("nan"))
    reference_bad_fractions = torch.full_like(relative_medians, float("nan"))

    if reference_depths is not None:
        if reference_depths.shape != output_depths.shape:
            raise ValueError(
                "reference_depths must match output_depths, got "
                f"{tuple(reference_depths.shape)} and {tuple(output_depths.shape)}"
            )
        reference_depths = reference_depths.to(
            device=output_depths.device, dtype=output_depths.dtype
        )

    for chart_index in range(output_depths.shape[0]):
        aligned = output_depths[chart_index]
        prior = prior_depths[chart_index]
        # A DAV2 prior is useful for its diagnostic, but it is not part of the
        # MASt3R-target validity contract.  In particular, a missing/invalid
        # prior must not make a Chart invisible to the reference-based safety
        # check below.
        valid = torch.isfinite(aligned) & (aligned > 0)
        if masks is not None:
            valid &= masks[chart_index].to(device=valid.device, dtype=torch.bool)
        prior_valid = valid & torch.isfinite(prior) & (prior > 0)
        if int(prior_valid.sum()) >= minimum_valid_pixels:
            relative_error = (
                (aligned[prior_valid] - prior[prior_valid]).abs()
                / prior[prior_valid].clamp_min(1e-6)
            )
            median_error = relative_error.median()
            bad_fraction = (relative_error > bad_pixel_relative_error).float().mean()
            relative_medians[chart_index] = median_error
            bad_fractions[chart_index] = bad_fraction
            if (
                reference_depths is None
                and median_error > median_relative_error_threshold
                and bad_fraction > bad_pixel_fraction_threshold
            ):
                confidences[chart_index].zero_()
                rejected[chart_index] = True

        if reference_depths is not None:
            reference = reference_depths[chart_index]
            reference_valid = valid & torch.isfinite(reference) & (reference > 0)
            if int(reference_valid.sum()) < minimum_valid_pixels:
                # The subsequent hard gate treats this as insufficient support;
                # do not invent a prior-change failure label here.
                continue
            reference_relative_error = (
                (aligned[reference_valid] - reference[reference_valid]).abs()
                / reference[reference_valid].clamp_min(1e-6)
            )
            reference_median = reference_relative_error.median()
            reference_bad_fraction = (
                reference_relative_error > bad_pixel_relative_error
            ).float().mean()
            reference_relative_medians[chart_index] = reference_median
            reference_bad_fractions[chart_index] = reference_bad_fraction
            if (
                reference_median > median_relative_error_threshold
                and reference_bad_fraction > bad_pixel_fraction_threshold
            ):
                confidences[chart_index].zero_()
                rejected[chart_index] = True

    if return_reference_metrics:
        return (
            confidences,
            rejected,
            relative_medians,
            bad_fractions,
            reference_relative_medians,
            reference_bad_fractions,
        )
    return confidences, rejected, relative_medians, bad_fractions


# TODO: Update the default values of the parameters
def align_charts_in_parallel(
    # Scene
    scene_pm,
    # Data parameters
    reference_data,
    reference_depths_for_audit=None,
    masks=None,
    rendering_size=1600,
    target_scale=5.,
    # ParallelAligner architecture parameters
    use_learnable_depth_encoding=True,
    learnable_depth_encoding_mode='add',
    predict_in_disparity_space=False,
    use_learnable_confidence=True,
    use_meta_mlp=False,
    use_lora_mlp=False,
    lora_rank=4,
    n_lora_layers=2,
    use_multi_res_charts_encoding=True,
    # ParallelAligner optimization parameters
    n_iterations=1000,
    use_gradient_loss=False,
    use_hessian_loss=False,
    use_normal_loss=True,
    use_curvature_loss=True,
    use_matching_loss=True,
    use_reprojection_loss=False,
    matching_thr_factor=1./20.,
    matching_update_iters=None,
    use_confidence_in_matching_loss=False,
    weight_encodings_with_confidence=False,
    regularize_chart_encodings_norms=False,
    use_total_variation_on_depth_encodings=False,
    gradient_loss_weight=50.,
    hessian_loss_weight=100.,
    normal_loss_weight=4.,
    curvature_loss_weight=1.,
    reprojection_loss_weight=2.,
    reprojection_loss_power=0.5,
    reprojection_matches_file=None,
    matching_loss_weight=5.,
    chart_encodings_norm_loss_weight=2.,
    total_variation_on_depth_encodings_weight=5.0,
    projection_chunk_size=262_144,
    matching_pixel_stride=1,
    matching_checkpoint_chunks=False,
    chart_encoding_norm_chunk_rows=0,
    encodings_lr=1e-2,
    mlp_lr=1e-3,
    confidence_lr=1e-3,
    lr_update_iters=[1000],
    lr_update_factor=0.1,
    verbose=True,
    return_training_losses=False,
    save_charts_data=True,
    charts_data_path='./',
):
    device = scene_pm.points3d.device
    
    charts_encoding_params=ChartsEncodingParams()
    depth_encoding_params=DepthEncodingParams()
    mlp_params=MLPParams()
    
    if verbose:
        print("===== ParallelAligner parameters =====\n")
        print("Charts encoding dim", charts_encoding_params.encoding_dim)
        print("Charts encoding resolution factor", charts_encoding_params.resolution_factor)
        print("Charts encoding initialization range", charts_encoding_params.initialization_range, '\n')
        print("Depth encoding dim", depth_encoding_params.encoding_dim)
        print("Depth encoding n bins", depth_encoding_params.n_bins)
        print("Depth encoding initialization range", depth_encoding_params.initialization_range, '\n')
        print("MLP input dim", mlp_params.n_deformation_layers)
        print("MLP deformation layer size", mlp_params.deformation_layer_size, '\n')
    
    # Build cameras
    cam_list = create_gs_cameras_from_pointmap(
        scene_pm,
        image_resolution=1, 
        load_gt_images=True, 
        max_img_size=rendering_size, 
        use_original_image_size=True,
        average_focal_distances=False,
        verbose=False,
    )
    pointmap_cameras = CamerasWrapper(cam_list, no_p3d_cameras=False)
    if target_scale is not None:
        scale_factor = target_scale / pointmap_cameras.get_spatial_extent()
        pointmap_cameras = rescale_cameras(pointmap_cameras, scale_factor)
    else:
        scale_factor = 1.
    pm_h, pm_w = scene_pm.points3d.shape[1:3]
    lowres_cameras = CamerasWrapper.from_p3d_cameras(
        p3d_cameras=pointmap_cameras.p3d_cameras,
        height=pm_h,
        width=pm_w,
    )
    matching_thr = matching_thr_factor * pointmap_cameras.get_spatial_extent()
        
    # Build initial depths
    pt_maps = scale_factor * scene_pm.points3d
    imgs = scene_pm.images
    # masks = scene_pm.masks
    manifolds, _ = get_manifold_meshes_from_pointmaps(
        pt_maps, imgs, masks=None, return_single_mesh_object=True, return_manifold_idx=True
    )
    _verts =  torch.nn.Parameter(manifolds.verts_packed().clone().to(device), requires_grad=False).to(device)
    _verts_idx = torch.arange(_verts.shape[0], device=device)

    initial_depths = torch.cat([
        pointmap_cameras.p3d_cameras[i_chart].get_world_to_view_transform().transform_points(
            _verts.reshape(scene_pm.points3d.shape)[i_chart].reshape(-1, 3)
        )[..., 2].reshape(1, pm_h, pm_w) for i_chart in range(len(pointmap_cameras))
    ], dim=0)
    
    # Load matches if needed
    if use_reprojection_loss and reprojection_matches_file is not None:
        match_to_img, match_to_pix, idx_to_image = load_mast3r_matches(reprojection_matches_file)
    else:
        match_to_img, match_to_pix, idx_to_image = None, None, None
    
    pa = ParallelAligner(
        depths=initial_depths,
        cameras=lowres_cameras,
        charts_encoding_params=ChartsEncodingParams(),
        depth_encoding_params=DepthEncodingParams(),
        mlp_params=MLPParams(),
        use_learnable_depth_encoding=use_learnable_depth_encoding,
        learnable_depth_encoding_mode=learnable_depth_encoding_mode,
        use_learnable_confidence=use_learnable_confidence,
        device=device,
        predict_in_disparity_space=predict_in_disparity_space,
        use_meta_mlp=use_meta_mlp,
        use_lora_mlp=use_lora_mlp,
        lora_rank=lora_rank,
        n_lora_layers=n_lora_layers,
        use_multi_res_charts_encoding=use_multi_res_charts_encoding,
        multi_res_charts_encoding_params=MultiResChartsEncodingParams(),
        weight_encodings_with_confidence=weight_encodings_with_confidence,
    )
    
    pa.optimize(
        reference_data=reference_data,
        masks=masks,
        n_iterations=n_iterations,
        use_gradient_loss=use_gradient_loss,
        use_hessian_loss=use_hessian_loss,
        use_normal_loss=use_normal_loss,
        use_matching_loss=use_matching_loss,
        use_curvature_loss=use_curvature_loss,
        use_reprojection_loss=use_reprojection_loss,
        regularize_chart_encodings_norms=regularize_chart_encodings_norms,
        use_total_variation_on_depth_encodings=use_total_variation_on_depth_encodings,
        matching_thr=matching_thr,
        use_confidence_in_matching_loss=use_confidence_in_matching_loss,
        matching_update_iters=matching_update_iters,
        gradient_loss_weight=gradient_loss_weight,
        hessian_loss_weight=hessian_loss_weight,
        normal_loss_weight=normal_loss_weight,
        curvature_loss_weight=curvature_loss_weight,
        matching_loss_weight=matching_loss_weight,
        reprojection_loss_weight=reprojection_loss_weight,
        reprojection_loss_power=reprojection_loss_power,
        chart_encodings_norm_loss_weight=chart_encodings_norm_loss_weight,
        total_variation_on_depth_encodings_weight=total_variation_on_depth_encodings_weight,
        projection_chunk_size=projection_chunk_size,
        matching_pixel_stride=matching_pixel_stride,
        matching_checkpoint_chunks=matching_checkpoint_chunks,
        chart_encoding_norm_chunk_rows=chart_encoding_norm_chunk_rows,
        encodings_lr=encodings_lr,
        mlp_lr=mlp_lr,
        confidence_lr=confidence_lr,
        lr_update_iters=lr_update_iters,
        lr_update_factor=lr_update_factor,
        verbose=verbose,
        match_to_img=match_to_img,
        match_to_pix=match_to_pix,
    )
    
    output_verts = pa._deformed_verts.clone()
    output_depths = torch.cat([
    pa.cameras.p3d_cameras[i_chart].get_world_to_view_transform().transform_points(
        output_verts[i_chart].reshape(-1, 3)
    )[..., 2].reshape(1, pm_h, pm_w) for i_chart in range(len(pa.cameras))
    ], dim=0)
    
    if use_learnable_confidence:
        with torch.no_grad():
            output_confs = pa.confidence
    else:
        output_confs = 4. * torch.ones_like(output_depths)
    output_verts, output_depths, output_confs = restore_unaligned_geometry(
        output_verts,
        output_depths,
        output_confs,
        _verts,
        initial_depths,
        masks,
    )
    (
        output_confs,
        rejected_alignment_charts,
        alignment_relative_medians,
        alignment_bad_fractions,
        alignment_reference_relative_medians,
        alignment_reference_bad_fractions,
    ) = reject_catastrophic_alignment_confidences(
        output_confs,
        output_depths,
        initial_depths,
        masks,
        reference_depths=reference_depths_for_audit,
        # The internal safeguard catches a genuine global departure from the
        # MASt3R target.  The stricter per-chart .50/.50 decision remains in
        # gate_aligned_charts.py, where support and coverage are audited.
        median_relative_error_threshold=0.75,
        bad_pixel_fraction_threshold=0.90,
        return_reference_metrics=True,
    )
    for chart_index in torch.nonzero(rejected_alignment_charts).flatten().tolist():
        if reference_depths_for_audit is None:
            basis = "prior"
            median = alignment_relative_medians[chart_index].item()
            bad_fraction = alignment_bad_fractions[chart_index].item()
        else:
            basis = "MASt3R reference"
            median = alignment_reference_relative_medians[chart_index].item()
            bad_fraction = alignment_reference_bad_fractions[chart_index].item()
        print(
            "[WARN] Rejecting catastrophic aligned chart "
            f"{chart_index} relative to {basis}: median relative depth error="
            f"{median:.3f}, bad-pixel fraction={bad_fraction:.3f}."
        )
    
    if save_charts_data:
        save_path = os.path.join(charts_data_path, "charts_data.npz")
        
        print(f"[INFO] Saving charts data to {save_path}")
        charts_prior_depths = initial_depths
        charts_depths = output_depths
        charts_pts = output_verts
        charts_confs = output_confs
        charts_scale_factor = scale_factor

        # Save the charts data with numpy
        payload = {
            "prior_depths": charts_prior_depths.cpu().numpy(),
            "depths": charts_depths.cpu().numpy(),
            "pts": charts_pts.cpu().numpy(),
            "confs": charts_confs.cpu().numpy(),
            "scale_factor": charts_scale_factor,
            "alignment_chart_valid": (~rejected_alignment_charts).cpu().numpy(),
            "alignment_prior_relative_median": alignment_relative_medians.cpu().numpy(),
            "alignment_prior_gt25_fraction": alignment_bad_fractions.cpu().numpy(),
            "alignment_reference_relative_median": alignment_reference_relative_medians.cpu().numpy(),
            "alignment_reference_gt25_fraction": alignment_reference_bad_fractions.cpu().numpy(),
        }
        if reference_depths_for_audit is not None:
            if reference_depths_for_audit.shape != charts_depths.shape:
                raise ValueError(
                    "reference_depths_for_audit must match chart depths, got "
                    f"{tuple(reference_depths_for_audit.shape)} and {tuple(charts_depths.shape)}"
                )
            payload["reference_depths"] = reference_depths_for_audit.detach().cpu().numpy()
            payload["alignment_reference_mask"] = (
                torch.ones_like(charts_depths, dtype=torch.bool)
                if masks is None
                else masks.to(device=charts_depths.device, dtype=torch.bool)
            ).cpu().numpy()
            payload["alignment_rejection_basis"] = np.asarray("mast3r_reference_depths")
        else:
            payload["alignment_rejection_basis"] = np.asarray("depthanything_prior_legacy")
        np.savez(save_path, **payload)
    
    if use_learnable_confidence:
        if return_training_losses:
            return output_verts, output_depths, output_confs, pa.train_losses
        else:
            return output_verts, output_depths, output_confs
    else:
        if return_training_losses:
            return output_verts, output_depths, pa.train_losses
        else:
            return output_verts, output_depths
        
