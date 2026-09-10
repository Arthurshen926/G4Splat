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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, depth

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_depth,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([]).cuda()
        if colors_precomp is None:
            colors_precomp = torch.Tensor([]).cuda()

        if scales is None:
            scales = torch.Tensor([]).cuda()
        if rotations is None:
            rotations = torch.Tensor([]).cuda()
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([]).cuda()
        

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )


def rasterize_mixed_gaussians(
    surface_means3D,
    surface_means2D,
    surface_scales,
    surface_rotations,
    volume_means3D,
    volume_means2D,
    volume_scales,
    volume_rotations,
    colors,
    opacities,
    surface_gate_indices,
    surface_gate_atlas,
    depth_query_bounds,
    audit_fields,
    raster_settings,
):
    """Rasterize native 2D surfels and 3D EWA volumes in one visibility pass.

    Primitive ids ``[0, N_surface)`` are 2D surfels and the remaining ids
    are 3D volumes. Both representations emit into one tile list and are
    radix-sorted together. Volumes use camera-space centre depth; surfels use
    the perspective-correct ray/plane intersection at each touched tile
    centre so an oblique facade is not assigned one global depth key.
    """
    return _RasterizeMixedGaussians.apply(
        surface_means3D,
        surface_means2D,
        surface_scales,
        surface_rotations,
        volume_means3D,
        volume_means2D,
        volume_scales,
        volume_rotations,
        colors,
        opacities,
        surface_gate_indices,
        surface_gate_atlas,
        depth_query_bounds,
        audit_fields,
        raster_settings,
    )


class _RasterizeMixedGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        surface_means3D,
        surface_means2D,
        surface_scales,
        surface_rotations,
        volume_means3D,
        volume_means2D,
        volume_scales,
        volume_rotations,
        colors,
        opacities,
        surface_gate_indices,
        surface_gate_atlas,
        depth_query_bounds,
        audit_fields,
        raster_settings,
    ):
        args = (
            raster_settings.bg,
            surface_means3D,
            surface_scales,
            surface_rotations,
            volume_means3D,
            volume_scales,
            volume_rotations,
            colors,
            opacities,
            surface_gate_indices,
            surface_gate_atlas,
            depth_query_bounds,
            raster_settings.scale_modifier,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            audit_fields,
            raster_settings.prefiltered,
            raster_settings.debug,
        )
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                outputs = _C.rasterize_mixed_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_mixed_fw.dump")
                print(
                    "\nMixed rasterizer forward failed; "
                    "wrote snapshot_mixed_fw.dump."
                )
                raise ex
        else:
            outputs = _C.rasterize_mixed_gaussians(*args)
        (
            rendered,
            color,
            others,
            radii,
            responsibility,
            gate_responsibility,
            geom_buffer,
            binning_buffer,
            image_buffer,
        ) = outputs
        ctx.raster_settings = raster_settings
        ctx.rendered = rendered
        ctx.surface_count = surface_means3D.shape[0]
        ctx.mark_non_differentiable(
            radii, responsibility, gate_responsibility
        )
        ctx.save_for_backward(
            surface_means3D,
            surface_scales,
            surface_rotations,
            volume_means3D,
            volume_scales,
            volume_rotations,
            colors,
            opacities,
            surface_gate_indices,
            surface_gate_atlas,
            depth_query_bounds,
            radii,
            others,
            geom_buffer,
            binning_buffer,
            image_buffer,
        )
        return (
            color,
            radii,
            others,
            responsibility,
            gate_responsibility,
        )

    @staticmethod
    def backward(
        ctx,
        grad_color,
        grad_radii,
        grad_others,
        grad_responsibility,
        grad_gate_responsibility,
    ):
        del grad_radii, grad_responsibility, grad_gate_responsibility
        settings = ctx.raster_settings
        (
            surface_means3D,
            surface_scales,
            surface_rotations,
            volume_means3D,
            volume_scales,
            volume_rotations,
            colors,
            opacities,
            surface_gate_indices,
            surface_gate_atlas,
            depth_query_bounds,
            radii,
            forward_others,
            geom_buffer,
            binning_buffer,
            image_buffer,
        ) = ctx.saved_tensors
        if grad_color is None:
            grad_color = torch.zeros(
                (3, settings.image_height, settings.image_width),
                dtype=colors.dtype,
                device=colors.device,
            )
        if grad_others is None:
            grad_others = torch.zeros(
                (13, settings.image_height, settings.image_width),
                dtype=colors.dtype,
                device=colors.device,
            )
        args = (
            settings.bg,
            surface_means3D,
            surface_scales,
            surface_rotations,
            volume_means3D,
            volume_scales,
            volume_rotations,
            colors,
            opacities,
            surface_gate_indices,
            surface_gate_atlas,
            depth_query_bounds,
            settings.scale_modifier,
            settings.viewmatrix,
            settings.projmatrix,
            settings.tanfovx,
            settings.tanfovy,
            radii,
            grad_color,
            grad_others,
            forward_others,
            geom_buffer,
            ctx.rendered,
            binning_buffer,
            image_buffer,
            settings.debug,
        )
        if settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                grads = _C.rasterize_mixed_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_mixed_bw.dump")
                print(
                    "\nMixed rasterizer backward failed; "
                    "wrote snapshot_mixed_bw.dump."
                )
                raise ex
        else:
            grads = _C.rasterize_mixed_gaussians_backward(*args)
        (
            grad_surface_means3D,
            grad_surface_scales,
            grad_surface_rotations,
            grad_volume_means3D,
            grad_volume_scales,
            grad_volume_rotations,
            grad_colors,
            grad_opacities,
            grad_surface_gate_atlas,
            grad_means2D,
        ) = grads
        split = ctx.surface_count
        return (
            grad_surface_means3D,
            grad_means2D[:split],
            grad_surface_scales,
            grad_surface_rotations,
            grad_volume_means3D,
            grad_means2D[split:],
            grad_volume_scales,
            grad_volume_rotations,
            grad_colors,
            grad_opacities,
            None,
            grad_surface_gate_atlas,
            None,
            None,
            None,
        )


class MixedGaussianRasterizer(nn.Module):
    """PyTorch module for the native, jointly sorted mixed CUDA path."""

    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(
        self,
        surface_means3D,
        surface_means2D,
        surface_scales,
        surface_rotations,
        volume_means3D,
        volume_means2D,
        volume_scales,
        volume_rotations,
        colors,
        opacities,
        surface_gate_indices=None,
        surface_gate_atlas=None,
        depth_query_bounds=None,
        audit_fields=None,
    ):
        expected = surface_means3D.shape[0] + volume_means3D.shape[0]
        if colors.shape != (expected, 3):
            raise ValueError(
                "colors must have shape "
                f"({expected}, 3), received {tuple(colors.shape)}"
            )
        if opacities.numel() != expected:
            raise ValueError(
                f"opacities must contain {expected} values, "
                f"received {opacities.numel()}"
            )
        if audit_fields is None:
            audit_fields = colors.new_empty(
                (0, self.raster_settings.image_height,
                 self.raster_settings.image_width)
            )
        if depth_query_bounds is None:
            depth_query_bounds = colors.new_empty(
                (0, self.raster_settings.image_height,
                 self.raster_settings.image_width)
            )
        if surface_gate_indices is None:
            surface_gate_indices = torch.full(
                (surface_means3D.shape[0],),
                -1,
                dtype=torch.int32,
                device=surface_means3D.device,
            )
        if surface_gate_atlas is None:
            surface_gate_atlas = colors.new_empty((0, 0, 0))
        if surface_gate_indices.shape != (surface_means3D.shape[0],):
            raise ValueError(
                "surface_gate_indices must contain one int per surfel"
            )
        surface_gate_indices = surface_gate_indices.to(
            device=surface_means3D.device, dtype=torch.int32
        )
        if (
            surface_gate_atlas.ndim != 3
            or (
                surface_gate_atlas.numel()
                and surface_gate_atlas.shape[1]
                != surface_gate_atlas.shape[2]
            )
        ):
            raise ValueError("surface_gate_atlas must have shape (K, G, G)")
        if surface_gate_atlas.device != surface_means3D.device:
            raise ValueError("surface_gate_atlas must be on the render device")
        if surface_gate_atlas.dtype != colors.dtype:
            raise ValueError("surface_gate_atlas must match the render dtype")
        if depth_query_bounds.shape not in {
            (0, self.raster_settings.image_height,
             self.raster_settings.image_width),
            (2, self.raster_settings.image_height,
             self.raster_settings.image_width),
        }:
            raise ValueError(
                "depth_query_bounds must be empty or have shape (2,H,W)"
            )
        if depth_query_bounds.device != surface_means3D.device:
            raise ValueError("depth_query_bounds must be on the render device")
        if depth_query_bounds.dtype != colors.dtype:
            raise ValueError("depth_query_bounds must match the render dtype")
        if surface_gate_indices.numel():
            minimum = int(surface_gate_indices.min())
            maximum = int(surface_gate_indices.max())
            if minimum < -1 or maximum >= surface_gate_atlas.shape[0]:
                raise ValueError(
                    "surface_gate_indices contains an invalid atlas row"
                )
        return rasterize_mixed_gaussians(
            surface_means3D,
            surface_means2D,
            surface_scales,
            surface_rotations,
            volume_means3D,
            volume_means2D,
            volume_scales,
            volume_rotations,
            colors,
            opacities,
            surface_gate_indices,
            surface_gate_atlas,
            depth_query_bounds,
            audit_fields,
            self.raster_settings,
        )
