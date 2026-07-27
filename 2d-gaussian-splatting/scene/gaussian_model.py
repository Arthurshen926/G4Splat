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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
import torch.nn.functional as F
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import C0, RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.point_utils import depths_to_points
from tqdm import tqdm

class GaussianModel:
    PRIMITIVE_STRUCTURAL = 0
    PRIMITIVE_FOLIAGE = 1
    PRIMITIVE_SKY = 2

    SOURCE_SFM_OR_BASE = 0
    SOURCE_CANOPY_RESIDUAL = 1
    # Unified Teacher initialization uses explicit evidence source codes.
    SOURCE_MAST3R_TRACK = 1
    SOURCE_CHART_RESIDUAL = 2

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        # Non-optimised provenance fields.  They make protected residual
        # continuations auditable without changing rasterizer inputs.
        self._primitive_class = torch.empty(0, dtype=torch.int16)
        self._source_type = torch.empty(0, dtype=torch.int16)
        self._track_id = torch.empty(0, dtype=torch.int64)
        self._geometry_confidence = torch.empty(0)
        self._protected_flag = torch.empty(0, dtype=torch.bool)
        self._block_id = torch.empty(0, dtype=torch.int32)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.use_mip_filter = False
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            {
                "primitive_class": self._primitive_class,
                "source_type": self._source_type,
                "track_id": self._track_id,
                "geometry_confidence": self._geometry_confidence,
                "protected_flag": self._protected_flag,
                "block_id": self._block_id,
            },
        )
    
    def restore(self, model_args, training_args):
        if len(model_args) not in {12, 13}:
            raise RuntimeError(
                f"Unsupported GaussianModel capture with {len(model_args)} fields"
            )
        metadata = model_args[12] if len(model_args) == 13 else None
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args[:12]
        self._restore_point_metadata(metadata)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def _initialize_point_metadata(
        self,
        count,
        *,
        primitive_class=PRIMITIVE_STRUCTURAL,
        source_type=SOURCE_SFM_OR_BASE,
        track_id=-1,
        geometry_confidence=1.0,
        protected_flag=False,
        block_id=-1,
        device=None,
    ):
        count = int(count)
        device = self._xyz.device if device is None else torch.device(device)
        self._primitive_class = torch.full(
            (count,), int(primitive_class), dtype=torch.int16, device=device
        )
        self._source_type = torch.full(
            (count,), int(source_type), dtype=torch.int16, device=device
        )
        self._track_id = torch.full(
            (count,), int(track_id), dtype=torch.int64, device=device
        )
        self._geometry_confidence = torch.full(
            (count,), float(geometry_confidence), dtype=torch.float32, device=device
        )
        self._protected_flag = torch.full(
            (count,), bool(protected_flag), dtype=torch.bool, device=device
        )
        self._block_id = torch.full(
            (count,), int(block_id), dtype=torch.int32, device=device
        )

    def _restore_point_metadata(self, metadata):
        point_count = int(len(self._xyz))
        if metadata is None:
            self._initialize_point_metadata(point_count, device=self._xyz.device)
            return
        required = {
            "primitive_class",
            "source_type",
            "geometry_confidence",
            "protected_flag",
            "block_id",
        }
        missing = sorted(required - set(metadata))
        if missing:
            raise RuntimeError(f"Gaussian metadata is missing fields: {missing}")
        fields = {
            "_primitive_class": (metadata["primitive_class"], torch.int16),
            "_source_type": (metadata["source_type"], torch.int16),
            "_track_id": (
                metadata.get(
                    "track_id",
                    torch.full((point_count,), -1, dtype=torch.int64),
                ),
                torch.int64,
            ),
            "_geometry_confidence": (metadata["geometry_confidence"], torch.float32),
            "_protected_flag": (metadata["protected_flag"], torch.bool),
            "_block_id": (metadata["block_id"], torch.int32),
        }
        for attribute, (value, dtype) in fields.items():
            tensor = torch.as_tensor(value, device=self._xyz.device, dtype=dtype).reshape(-1)
            if len(tensor) != point_count:
                raise RuntimeError(
                    f"Gaussian metadata {attribute} has {len(tensor)} rows for "
                    f"{point_count} points"
                )
            setattr(self, attribute, tensor)

    def _point_metadata_from_indices(self, indices, *, repeat=1, **overrides):
        indices = indices.reshape(-1)
        if int(repeat) > 1:
            indices = indices.repeat_interleave(int(repeat))
        metadata = {
            "primitive_class": self._primitive_class[indices],
            "source_type": self._source_type[indices],
            "track_id": self._track_id[indices],
            "geometry_confidence": self._geometry_confidence[indices],
            "protected_flag": self._protected_flag[indices],
            "block_id": self._block_id[indices],
        }
        dtype_by_name = {
            "primitive_class": torch.int16,
            "source_type": torch.int16,
            "track_id": torch.int64,
            "geometry_confidence": torch.float32,
            "protected_flag": torch.bool,
            "block_id": torch.int32,
        }
        for name, value in overrides.items():
            if value is not None:
                metadata[name] = torch.full(
                    (len(indices),),
                    value,
                    dtype=dtype_by_name[name],
                    device=self._xyz.device,
                )
        return metadata

    @property
    def get_primitive_class(self):
        return self._primitive_class

    @property
    def get_source_type(self):
        return self._source_type

    @property
    def get_track_id(self):
        return self._track_id

    @property
    def get_geometry_confidence(self):
        return self._geometry_confidence

    @property
    def get_protected_flag(self):
        return self._protected_flag

    @property
    def get_block_id(self):
        return self._block_id

    @torch.no_grad()
    def mark_prefix_protected(self, point_count):
        point_count = int(point_count)
        if point_count < 0 or point_count > len(self._xyz):
            raise ValueError(
                f"Cannot protect {point_count} of {len(self._xyz)} Gaussian points"
            )
        self._protected_flag[:point_count] = True

    def freeze_params(self):
        """
        Freeze all parameters of the GaussianModel to prevent gradient updates
        
        This is useful for hierarchical model training or transfer learning scenarios,
        such as when you want to add new points based on an existing model without 
        changing the original points.
        
        The freezing operation affects the following parameters:
        - Positions (_xyz)
        - Features (_features_dc, _features_rest)
        - Scaling (_scaling)
        - Rotation (_rotation)
        - Opacity (_opacity)
        
        Note: This operation modifies the original parameters to no longer require gradients
        """
        # Record the original parameter count
        original_point_count = self._xyz.shape[0]
        print(f"Freezing {original_point_count} points in GaussianModel")
        
        # Freeze each parameter
        self._xyz.requires_grad_(False)
        self._features_dc.requires_grad_(False)
        self._features_rest.requires_grad_(False)
        self._scaling.requires_grad_(False)
        self._rotation.requires_grad_(False)
        self._opacity.requires_grad_(False)
        
        # If optimizer is already set, update the parameters in the optimizer
        if self.optimizer is not None:
            # Replace parameters with frozen versions
            updated_param_groups = []
            for group in self.optimizer.param_groups:
                if group["name"] == "xyz":
                    group["params"] = [self._xyz]
                elif group["name"] == "f_dc":
                    group["params"] = [self._features_dc]
                elif group["name"] == "f_rest":
                    group["params"] = [self._features_rest]
                elif group["name"] == "opacity":
                    group["params"] = [self._opacity]
                elif group["name"] == "scaling":
                    group["params"] = [self._scaling]
                elif group["name"] == "rotation":
                    group["params"] = [self._rotation]
                
                # Set learning rate to 0 to ensure no updates
                group["lr"] = 0.0
                updated_param_groups.append(group)
            
            # Recreate the optimizer with updated parameter groups while retaining state
            self.optimizer = torch.optim.Adam(
                updated_param_groups,
                lr=0.0,
                eps=1e-15
            )

    @property
    def get_scaling(self):
        scales = self.scaling_activation(self._scaling)
        if self.use_mip_filter:
            scales = torch.square(scales) + torch.square(self.mip_filter)
            scales = torch.sqrt(scales)
        return scales #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        opacity = self.opacity_activation(self._opacity) 
        if self.use_mip_filter:
            scales = self.scaling_activation(self._scaling)
            
            scales_square = torch.square(scales)
            det1 = scales_square.prod(dim=1)
            
            scales_after_square = scales_square + torch.square(self.mip_filter) 
            det2 = scales_after_square.prod(dim=1) 
            coef = torch.sqrt(det1 / det2)
            opacity = opacity * coef[..., None]
        return opacity
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._initialize_point_metadata(len(self._xyz), device=self._xyz.device)
        
    def create_from_parameters(self, _means, _scales, _quaternions, _colors, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = _means
        fused_color = RGB2SH(_colors)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        scales = torch.log(_scales)
        rots = _quaternions

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._initialize_point_metadata(len(self._xyz), device=self._xyz.device)

    def append_from_parameters(
        self,
        means,
        scales,
        quaternions,
        colors,
        *,
        initial_opacity=0.05,
    ):
        """Append conservative reseed Gaussians before optimizer construction."""
        if self.optimizer is not None:
            raise RuntimeError("append_from_parameters must run before training_setup")
        if len(means) == 0:
            return 0
        if not (len(means) == len(scales) == len(quaternions) == len(colors)):
            raise ValueError("Reseed Gaussian parameter lengths do not match")

        means = means.to(device=self._xyz.device, dtype=self._xyz.dtype)
        scales = scales.to(device=self._scaling.device, dtype=self._scaling.dtype).clamp_min(1e-8)
        quaternions = F.normalize(
            quaternions.to(device=self._rotation.device, dtype=self._rotation.dtype),
            dim=-1,
        )
        colors = colors.to(device=self._features_dc.device, dtype=self._features_dc.dtype)
        features_dc = RGB2SH(colors)[:, None, :]
        features_rest = torch.zeros(
            (len(means), (self.max_sh_degree + 1) ** 2 - 1, 3),
            device=self._features_rest.device,
            dtype=self._features_rest.dtype,
        )
        opacities = self.inverse_opacity_activation(
            torch.full(
                (len(means), 1),
                float(initial_opacity),
                device=self._opacity.device,
                dtype=self._opacity.dtype,
            )
        )

        self._xyz = nn.Parameter(torch.cat([self._xyz.detach(), means], dim=0).requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.cat([self._features_dc.detach(), features_dc], dim=0).requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.cat([self._features_rest.detach(), features_rest], dim=0).requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.cat([self._scaling.detach(), torch.log(scales)], dim=0).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.cat([self._rotation.detach(), quaternions], dim=0).requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.cat([self._opacity.detach(), opacities], dim=0).requires_grad_(True)
        )
        old_count = len(self._xyz) - len(means)
        if len(self._primitive_class) != old_count:
            self._initialize_point_metadata(old_count, device=self._xyz.device)
        appended_metadata = {
            "primitive_class": torch.full(
                (len(means),), self.PRIMITIVE_STRUCTURAL,
                dtype=torch.int16, device=self._xyz.device
            ),
            "source_type": torch.full(
                (len(means),), self.SOURCE_SFM_OR_BASE,
                dtype=torch.int16, device=self._xyz.device
            ),
            "geometry_confidence": torch.ones(
                len(means), dtype=torch.float32, device=self._xyz.device
            ),
            "protected_flag": torch.zeros(
                len(means), dtype=torch.bool, device=self._xyz.device
            ),
            "block_id": torch.full(
                (len(means),), -1, dtype=torch.int32, device=self._xyz.device
            ),
        }
        for name, attribute in (
            ("primitive_class", "_primitive_class"),
            ("source_type", "_source_type"),
            ("geometry_confidence", "_geometry_confidence"),
            ("protected_flag", "_protected_flag"),
            ("block_id", "_block_id"),
        ):
            setattr(
                self,
                attribute,
                torch.cat([getattr(self, attribute), appended_metadata[name]], dim=0),
            )
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self._xyz.device)

        if self.use_mip_filter:
            old_filter = getattr(self, "mip_filter", None)
            if old_filter is None or len(old_filter) != len(self._xyz) - len(means):
                old_filter = torch.zeros(
                    (len(self._xyz) - len(means), 1),
                    device=self._xyz.device,
                    dtype=self._xyz.dtype,
                )
            new_filter = torch.zeros(
                (len(means), 1),
                device=old_filter.device,
                dtype=old_filter.dtype,
            )
            self.mip_filter = torch.cat([old_filter, new_filter], dim=0)
        return len(means)

    def freeze_prefix_gradients(self, point_count):
        """Freeze a loaded baseline prefix while allowing appended points to train.

        A zero gradient alone is not sufficient after a checkpoint restore:
        Adam will still apply a non-zero update from the restored ``exp_avg``
        buffer.  Clear the prefix state at the same time as installing the
        gradient mask so a protected continuation is genuinely immutable.
        """
        point_count = int(point_count)
        if point_count < 0 or point_count > len(self._xyz):
            raise ValueError(
                f"Cannot freeze {point_count} of {len(self._xyz)} Gaussian points"
            )
        self._frozen_prefix_count = point_count
        self._refresh_prefix_gradient_hooks()
        self._frozen_prefix_optimizer_state = self.clear_prefix_optimizer_state(
            point_count
        )
        self._warmstart_suffix_initial_xyz = self._xyz[point_count:].detach().clone()
        if hasattr(self, "mip_filter") and len(self.mip_filter) >= point_count:
            self._frozen_prefix_mip_filter = self.mip_filter[:point_count].detach().clone()
        print(
            f"[INFO] Frozen {point_count} baseline Gaussians; "
            f"{len(self._xyz) - point_count} appended Gaussians remain trainable."
        )

    @torch.no_grad()
    def clear_prefix_optimizer_state(self, point_count=None):
        """Zero checkpoint momentum for an immutable Gaussian prefix.

        The model stores every Gaussian field as one concatenated parameter.
        Consequently a parameter-level ``requires_grad=False`` cannot freeze
        only the loaded prefix.  Gradient hooks block new gradients, while
        this helper removes Adam's *old* per-row momentum.  It deliberately
        leaves scalar optimizer bookkeeping (such as ``step``) and every
        appended suffix row intact.
        """
        if point_count is None:
            point_count = int(getattr(self, "_frozen_prefix_count", 0))
        point_count = int(point_count)
        if point_count <= 0 or self.optimizer is None:
            return {}
        if point_count > len(self._xyz):
            raise ValueError(
                f"Cannot clear optimizer state for {point_count} of {len(self._xyz)} points"
            )

        cleared = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) != 1:
                raise RuntimeError("Expected one tensor per Gaussian optimizer group")
            state = self.optimizer.state.get(group["params"][0])
            if not state:
                continue
            fields = []
            for field_name, value in state.items():
                if not torch.is_tensor(value) or value.ndim == 0:
                    continue
                if value.shape[0] < point_count:
                    raise RuntimeError(
                        "Optimizer state is shorter than the frozen Gaussian prefix"
                    )
                value[:point_count].zero_()
                fields.append(field_name)
            if fields:
                cleared[str(group.get("name", "unnamed"))] = fields
        return cleared

    def _refresh_prefix_gradient_hooks(self):
        """Reinstall an existing baseline-freeze mask after a topology append.

        Every append replaces optimizer tensors with fresh ``nn.Parameter``
        instances.  Tensor hooks do not survive that replacement, so a
        continuation that promises to keep its checkpoint baseline fixed must
        explicitly restore the prefix mask after ``densification_postfix``.
        """
        point_count = int(getattr(self, "_frozen_prefix_count", 0))
        if point_count <= 0:
            return
        if point_count > len(self._xyz):
            raise RuntimeError(
                "Frozen baseline is longer than the current Gaussian topology"
            )
        handles = getattr(self, "_prefix_gradient_hook_handles", [])
        for handle in handles:
            handle.remove()
        self._prefix_gradient_hook_handles = []

        for parameter in (
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._opacity,
            self._scaling,
            self._rotation,
        ):
            trainable = torch.ones(
                (len(parameter),) + (1,) * (parameter.ndim - 1),
                device=parameter.device,
                dtype=parameter.dtype,
            )
            trainable[:point_count] = 0
            self._prefix_gradient_hook_handles.append(
                parameter.register_hook(lambda gradient, mask=trainable: gradient * mask)
            )

    @torch.no_grad()
    def constrain_trainable_suffix(
        self,
        point_count,
        max_opacity=1.0,
        max_scale=0.0,
        max_position_delta=0.0,
        diffuse_only=False,
        clamp_dc=False,
    ):
        """Project sparse warm-start additions back into conservative bounds."""
        point_count = int(point_count)
        if point_count < 0 or point_count > len(self._xyz):
            raise ValueError(f"Invalid warm-start prefix size: {point_count}")
        suffix = slice(point_count, None)
        if 0.0 < float(max_opacity) < 1.0:
            cap = self.inverse_opacity_activation(
                self._opacity.new_tensor(float(max_opacity))
            )
            self._opacity[suffix].clamp_(max=cap)
        if float(max_scale) > 0.0:
            self._scaling[suffix].clamp_(
                max=float(np.log(float(max_scale)))
            )
        if float(max_position_delta) > 0.0:
            initial = getattr(self, "_warmstart_suffix_initial_xyz", None)
            if initial is None or len(initial) != len(self._xyz) - point_count:
                raise RuntimeError("Warm-start position constraints require a frozen suffix snapshot")
            delta = self._xyz[suffix] - initial
            norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-12)
            scale = torch.clamp(float(max_position_delta) / norm, max=1.0)
            self._xyz[suffix].copy_(initial + delta * scale)
        if diffuse_only:
            self._features_rest[suffix].zero_()
        if clamp_dc:
            dc_limit = 0.5 / C0
            self._features_dc[suffix].clamp_(min=-dc_limit, max=dc_limit)

    @torch.no_grad()
    def restore_frozen_prefix_mip_filter(self):
        frozen = getattr(self, "_frozen_prefix_mip_filter", None)
        point_count = int(getattr(self, "_frozen_prefix_count", 0))
        if frozen is None or point_count <= 0 or not hasattr(self, "mip_filter"):
            return
        if len(self.mip_filter) < point_count:
            raise RuntimeError("MIP filter is shorter than the frozen warm-start prefix")
        self.mip_filter[:point_count].copy_(frozen)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.non_position_base_lrs = {
            group["name"]: group["lr"] for group in self.optimizer.param_groups
            if group["name"] != "xyz"
        }
        self.non_position_lr_decay_from = int(training_args.non_position_lr_decay_from)
        self.non_position_lr_decay_until = max(int(training_args.iterations), 1)
        self.non_position_lr_final_mult = float(training_args.non_position_lr_final_mult)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.non_position_lr_decay_from >= 0:
            span = max(
                self.non_position_lr_decay_until - self.non_position_lr_decay_from,
                1,
            )
            progress = min(
                max((iteration - self.non_position_lr_decay_from) / span, 0.0),
                1.0,
            )
            final_mult = max(self.non_position_lr_final_mult, 1e-8)
            multiplier = final_mult ** progress
            for param_group in self.optimizer.param_groups:
                name = param_group["name"]
                if name != "xyz":
                    param_group["lr"] = self.non_position_base_lrs[name] * multiplier
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self.use_mip_filter:
            l.append('mip_filter')
        l.extend(
            [
                "primitive_class",
                "source_type",
                "track_id",
                "geometry_confidence",
                "protected_flag",
                "block_id",
            ]
        )
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        metadata = np.stack(
            [
                self._primitive_class.detach().cpu().numpy(),
                self._source_type.detach().cpu().numpy(),
                self._track_id.detach().cpu().numpy(),
                self._geometry_confidence.detach().cpu().numpy(),
                self._protected_flag.detach().cpu().numpy().astype(np.float32),
                self._block_id.detach().cpu().numpy(),
            ],
            axis=1,
        ).astype(np.float32)
        
        if self.use_mip_filter:
            mip_filter = self.mip_filter.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if self.use_mip_filter:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, mip_filter, metadata), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, metadata), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
    @torch.no_grad()
    def get_tetra_points(
        self, 
        downsample_ratio : float = None, 
        gaussian_flatness : float = 1e-3, 
        return_idx : bool = False,
        points_idx : torch.Tensor = None,
    ):
        import trimesh
        M = trimesh.creation.box()
        M.vertices *= 2
        
        rots = build_rotation(self._rotation)
        scales_3d = torch.nn.functional.pad(
            self.get_scaling, 
            (0, 1), 
            mode="constant", 
            value=gaussian_flatness,
        )
        print(f"[INFO] Padding 2D scaling with {gaussian_flatness} for tetra points: {scales_3d[0]}")
        
        if (downsample_ratio is None) and (points_idx is None):
            xyz = self.get_xyz
            scale = scales_3d * 3. # TODO test
            # filter points with small opacity for bicycle scene
            # opacity = self.get_opacity_with_3D_filter
            # mask = (opacity > 0.1).squeeze(-1)
            # xyz = xyz[mask]
            # scale = scale[mask]
            # rots = rots[mask]
        else:
            if points_idx is None:
                print(f"[INFO] Downsampling tetra points by {downsample_ratio}.")
                xyz_idx = torch.randperm(self.get_xyz.shape[0])[:int(self.get_xyz.shape[0] * downsample_ratio)]
                xyz = self.get_xyz[xyz_idx]
                scale = scales_3d[xyz_idx] * 3. / (downsample_ratio ** (1/3))
                rots = rots[xyz_idx]
                print(f"[INFO] Number of tetra points after downsampling: {xyz.shape[0]}.")
            else:
                downsample_ratio = len(points_idx) / len(self.get_xyz)
                xyz_idx = points_idx
                xyz = self.get_xyz[xyz_idx]
                scale = scales_3d[xyz_idx] * 3. / (downsample_ratio ** (1/3))
                rots = rots[xyz_idx]
                print(f"[INFO] Number of tetra points after downsampling: {xyz.shape[0]}.")
                
        vertices = M.vertices.T    
        vertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)
        # scale vertices first
        vertices = vertices * scale.unsqueeze(-1)
        vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)
        vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()
        # concat center points
        vertices = torch.cat([vertices, xyz], dim=0)
        
        # scale is not a good solution but use it for now
        scale = scale.max(dim=-1, keepdim=True)[0]
        scale_corner = scale.repeat(1, 8).reshape(-1, 1)
        vertices_scale = torch.cat([scale_corner, scale], dim=0)
        if return_idx:
            if downsample_ratio is None:
                print("[WARNING] return_idx might not be needed when downsample_ratio is None")
                xyz_idx = torch.arange(self.get_xyz.shape[0])
            return vertices, vertices_scale, xyz_idx
        else:
            return vertices, vertices_scale
    
    def set_mip_filter(self, use_mip_filter: bool):
        self.use_mip_filter = use_mip_filter
    
    @torch.no_grad()
    def compute_mip_filter(self, cameras, znear=0.2, filter_variance=0.2):
        # Set the flag to use the mip filter
        if not self.use_mip_filter:
            print("[WARNING] Computing mip filter but mip filter is currently disabled.")
        
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # We should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
            # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            xyz_to_cam = torch.norm(xyz_cam, dim=1)
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > znear
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            # Mip support must use the same off-axis camera as rendering.
            # Falling back to a centred principal point here changes which
            # splats are filtered near an image border in calibrated scenes.
            x = x / z * camera.focal_x + camera.cx
            y = y / z * camera.focal_y + camera.cy
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x
        
        distance[~valid_points] = distance[valid_points].max()
        
        mip_filter = distance / focal_length * (filter_variance ** 0.5)
        self.mip_filter = mip_filter[..., None]
        self.restore_frozen_prefix_mip_filter()

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        if "mip_filter" in [p.name for p in plydata.elements[0].properties]:
            mip_filter = np.asarray(plydata.elements[0]["mip_filter"])[..., np.newaxis]
            use_mip_filter = True
            self.set_mip_filter(use_mip_filter)
            print("[INFO] Loading mip filter from ply file.")
        else:
            print("[INFO] No mip filter found in ply file.")
            use_mip_filter = False

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        property_names = {item.name for item in plydata.elements[0].properties}
        if {
            "primitive_class",
            "source_type",
            "geometry_confidence",
            "protected_flag",
            "block_id",
        }.issubset(property_names):
            self._primitive_class = torch.tensor(
                np.asarray(plydata.elements[0]["primitive_class"]),
                dtype=torch.int16,
                device="cuda",
            )
            self._source_type = torch.tensor(
                np.asarray(plydata.elements[0]["source_type"]),
                dtype=torch.int16,
                device="cuda",
            )
            self._track_id = torch.tensor(
                (
                    np.asarray(plydata.elements[0]["track_id"])
                    if "track_id" in property_names
                    else np.full(len(xyz), -1)
                ),
                dtype=torch.int64,
                device="cuda",
            )
            self._geometry_confidence = torch.tensor(
                np.asarray(plydata.elements[0]["geometry_confidence"]),
                dtype=torch.float32,
                device="cuda",
            )
            self._protected_flag = torch.tensor(
                np.asarray(plydata.elements[0]["protected_flag"]) > 0.5,
                dtype=torch.bool,
                device="cuda",
            )
            self._block_id = torch.tensor(
                np.asarray(plydata.elements[0]["block_id"]),
                dtype=torch.int32,
                device="cuda",
            )
        else:
            self._initialize_point_metadata(len(self._xyz), device=self._xyz.device)
        if use_mip_filter:
            self.mip_filter = torch.tensor(mip_filter, dtype=torch.float, device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self._primitive_class = self._primitive_class[valid_points_mask]
        self._source_type = self._source_type[valid_points_mask]
        self._track_id = self._track_id[valid_points_mask]
        self._geometry_confidence = self._geometry_confidence[valid_points_mask]
        self._protected_flag = self._protected_flag[valid_points_mask]
        self._block_id = self._block_id[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        *,
        metadata=None,
    ):
        previous_point_count = self.get_xyz.shape[0]
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        new_count = int(len(new_xyz))
        if metadata is None:
            metadata = {
                "primitive_class": torch.full(
                    (new_count,), self.PRIMITIVE_STRUCTURAL,
                    dtype=torch.int16, device=self._xyz.device
                ),
                "source_type": torch.full(
                    (new_count,), self.SOURCE_SFM_OR_BASE,
                    dtype=torch.int16, device=self._xyz.device
                ),
                "track_id": torch.full(
                    (new_count,), -1, dtype=torch.int64, device=self._xyz.device
                ),
                "geometry_confidence": torch.ones(
                    new_count, dtype=torch.float32, device=self._xyz.device
                ),
                "protected_flag": torch.zeros(
                    new_count, dtype=torch.bool, device=self._xyz.device
                ),
                "block_id": torch.full(
                    (new_count,), -1, dtype=torch.int32, device=self._xyz.device
                ),
            }
        for name, attribute in (
            ("primitive_class", "_primitive_class"),
            ("source_type", "_source_type"),
            ("track_id", "_track_id"),
            ("geometry_confidence", "_geometry_confidence"),
            ("protected_flag", "_protected_flag"),
            ("block_id", "_block_id"),
        ):
            extension = torch.as_tensor(
                metadata[name],
                dtype=getattr(self, attribute).dtype,
                device=self._xyz.device,
            ).reshape(-1)
            if len(extension) != new_count:
                raise RuntimeError(
                    f"Appended metadata {name} has {len(extension)} rows for "
                    f"{new_count} new Gaussians"
                )
            setattr(
                self,
                attribute,
                torch.cat([getattr(self, attribute), extension], dim=0),
            )

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # Normal densification temporarily disables the Mip filter and then
        # recomputes it after split/prune.  Additive continuation deliberately
        # never prunes its protected baseline, so it needs a shape-consistent
        # filter immediately after appending points.  The caller may replace
        # these inherited/zero values later, but rendering must never observe a
        # filter whose length belongs to the pre-densification topology.
        if self.use_mip_filter:
            old_filter = getattr(self, "mip_filter", None)
            if old_filter is None or len(old_filter) != previous_point_count:
                old_filter = torch.zeros(
                    (previous_point_count, 1),
                    device=self._xyz.device,
                    dtype=self._xyz.dtype,
                )
            extension = torch.zeros(
                (self.get_xyz.shape[0] - previous_point_count, 1),
                device=old_filter.device,
                dtype=old_filter.dtype,
            )
            self.mip_filter = torch.cat([old_filter, extension], dim=0)
        self._refresh_prefix_gradient_hooks()
        # ``cat_tensors_to_optimizer`` retains old prefix momentum when it
        # swaps parameter tensors.  Reassert the invariant after every
        # additive append, even though the initial protected-stage clear makes
        # this normally a no-op.
        self.clear_prefix_optimizer_state()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        selected_indices = torch.nonzero(selected_pts_mask, as_tuple=False).flatten()
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            metadata=self._point_metadata_from_indices(
                selected_indices,
                repeat=N,
                track_id=-1,
                protected_flag=False,
            ),
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        selected_indices = torch.nonzero(selected_pts_mask, as_tuple=False).flatten()
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            metadata=self._point_metadata_from_indices(
                selected_indices,
                track_id=-1,
                protected_flag=False,
            ),
        )

    def densify_and_clone_limited(
        self,
        grads,
        grad_threshold,
        scene_extent,
        max_new_points,
        *,
        opacity_ceiling=0.02,
        eligibility_mask=None,
        primitive_class=None,
        source_type=None,
        geometry_confidence=None,
        protected_flag=None,
        block_id=None,
    ):
        """Append a bounded, conservative residual clone set without pruning.

        This is intentionally different from the native 2DGS
        ``densify_and_prune`` path: it never removes or replaces a protected
        checkpoint Gaussian.  It is used for continuation when a Chart-led
        bootstrap lacks capacity in real camera views, not for a new-model
        initialization.  New clones inherit their parent's Mip filter and are
        opacity-capped so an append cannot double an established surface's
        alpha in one step.
        """
        max_new_points = int(max_new_points)
        if max_new_points <= 0:
            return 0
        if not 0.0 < float(opacity_ceiling) < 1.0:
            raise ValueError("opacity_ceiling must lie strictly between 0 and 1")

        grad_norm = torch.nan_to_num(torch.norm(grads, dim=-1), nan=0.0)
        selected_pts_mask = grad_norm >= grad_threshold
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )
        if eligibility_mask is not None:
            eligibility_mask = torch.as_tensor(
                eligibility_mask, dtype=torch.bool, device=selected_pts_mask.device
            ).reshape(-1)
            if len(eligibility_mask) != len(selected_pts_mask):
                raise ValueError(
                    "eligibility_mask must have one value per Gaussian point"
                )
            selected_pts_mask = torch.logical_and(
                selected_pts_mask, eligibility_mask
            )
        selected_indices = torch.nonzero(selected_pts_mask, as_tuple=False).flatten()
        if selected_indices.numel() == 0:
            return 0
        if selected_indices.numel() > max_new_points:
            _, ranking = torch.topk(grad_norm[selected_indices], k=max_new_points)
            selected_indices = selected_indices[ranking]

        parent_mip_filter = None
        if self.use_mip_filter and hasattr(self, "mip_filter"):
            parent_mip_filter = self.mip_filter[selected_indices].detach().clone()

        new_xyz = self._xyz[selected_indices]
        new_features_dc = self._features_dc[selected_indices]
        new_features_rest = self._features_rest[selected_indices]
        opacity_limit = self.inverse_opacity_activation(
            torch.full_like(self._opacity[selected_indices], float(opacity_ceiling))
        )
        new_opacities = torch.minimum(self._opacity[selected_indices], opacity_limit)
        new_scaling = self._scaling[selected_indices]
        new_rotation = self._rotation[selected_indices]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            metadata=self._point_metadata_from_indices(
                selected_indices,
                primitive_class=primitive_class,
                source_type=source_type,
                geometry_confidence=geometry_confidence,
                protected_flag=protected_flag,
                block_id=block_id,
                track_id=-1,
            ),
        )
        if parent_mip_filter is not None:
            self.mip_filter[-len(parent_mip_filter):] = parent_mip_filter
        return int(selected_indices.numel())

    def densify_and_split_limited(
        self,
        grads,
        grad_threshold,
        scene_extent,
        max_new_points,
        *,
        children_per_parent=2,
        opacity_ceiling=0.02,
        max_parent_scale_fraction=0.1,
        eligibility_mask=None,
        primitive_class=None,
        source_type=None,
        geometry_confidence=None,
        protected_flag=None,
        block_id=None,
    ):
        """Append compact children for large, high-gradient Gaussians.

        Unlike native ``densify_and_split``, this continuation primitive never
        deletes the large parent.  It is for a protected model whose broad
        Chart-led surfels still cover a real camera view but cannot express its
        local facade/foliage detail.  Children receive a local randomized
        offset and reduced scales, while their opacity is capped so the parent
        remains the stable low-frequency explanation during refinement.
        """
        max_new_points = int(max_new_points)
        children_per_parent = int(children_per_parent)
        if max_new_points <= 0:
            return 0
        if children_per_parent < 1:
            raise ValueError("children_per_parent must be positive")
        if not 0.0 < float(opacity_ceiling) < 1.0:
            raise ValueError("opacity_ceiling must lie strictly between 0 and 1")
        if not 0.0 < float(max_parent_scale_fraction) <= 1.0:
            raise ValueError("max_parent_scale_fraction must lie in (0, 1]")

        # ``grads`` belongs to the topology at the start of a densification
        # event.  Never let a stale, shorter gradient tensor silently address
        # points appended by another residual primitive.
        point_count = min(int(grads.shape[0]), int(self.get_xyz.shape[0]))
        if point_count == 0:
            return 0
        grad_norm = torch.nan_to_num(torch.norm(grads[:point_count], dim=-1), nan=0.0)
        parent_scales = torch.max(self.get_scaling[:point_count], dim=1).values
        selected_pts_mask = torch.logical_and(grad_norm >= grad_threshold, parent_scales > self.percent_dense * scene_extent)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            parent_scales <= float(max_parent_scale_fraction) * scene_extent,
        )
        if eligibility_mask is not None:
            eligibility_mask = torch.as_tensor(
                eligibility_mask, dtype=torch.bool, device=selected_pts_mask.device
            ).reshape(-1)
            if len(eligibility_mask) < point_count:
                raise ValueError(
                    "eligibility_mask must cover every Gaussian addressed by grads"
                )
            selected_pts_mask = torch.logical_and(
                selected_pts_mask, eligibility_mask[:point_count]
            )
        selected_indices = torch.nonzero(selected_pts_mask, as_tuple=False).flatten()
        max_parents = max_new_points // children_per_parent
        if selected_indices.numel() == 0 or max_parents <= 0:
            return 0
        if selected_indices.numel() > max_parents:
            _, ranking = torch.topk(grad_norm[selected_indices], k=max_parents)
            selected_indices = selected_indices[ranking]

        repeated_indices = selected_indices.repeat_interleave(children_per_parent)
        parent_scales = self.get_scaling[repeated_indices]
        stds = torch.cat(
            [parent_scales, torch.zeros_like(parent_scales[:, :1])], dim=-1
        )
        offsets = torch.normal(mean=torch.zeros_like(stds), std=stds)
        rotations = build_rotation(self._rotation[repeated_indices])
        new_xyz = torch.bmm(rotations, offsets.unsqueeze(-1)).squeeze(-1)
        new_xyz = new_xyz + self.get_xyz[repeated_indices]
        new_scaling = self.scaling_inverse_activation(
            parent_scales / (0.8 * children_per_parent)
        )
        opacity_limit = self.inverse_opacity_activation(
            torch.full_like(self._opacity[repeated_indices], float(opacity_ceiling))
        )
        new_opacities = torch.minimum(self._opacity[repeated_indices], opacity_limit)

        parent_mip_filter = None
        if self.use_mip_filter and hasattr(self, "mip_filter"):
            parent_mip_filter = self.mip_filter[repeated_indices].detach().clone()
        self.densification_postfix(
            new_xyz,
            self._features_dc[repeated_indices],
            self._features_rest[repeated_indices],
            new_opacities,
            new_scaling,
            self._rotation[repeated_indices],
            metadata=self._point_metadata_from_indices(
                selected_indices,
                repeat=children_per_parent,
                primitive_class=primitive_class,
                source_type=source_type,
                geometry_confidence=geometry_confidence,
                protected_flag=protected_flag,
                block_id=block_id,
                track_id=-1,
            ),
        )
        if parent_mip_filter is not None:
            self.mip_filter[-len(parent_mip_filter):] = parent_mip_filter
        return int(repeated_indices.numel())

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        use_mip_filter = self.use_mip_filter
        if use_mip_filter:
            self.set_mip_filter(False)
            
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
        if use_mip_filter:
            self.set_mip_filter(True)

    def densify_and_prune_bounded(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        *,
        max_points,
        max_growth,
    ):
        """Run native clone/split topology under a strict point budget.

        Native 2DGS densification selects every point above the gradient
        threshold.  Large outdoor scenes can therefore add hundreds of
        thousands of surfels in one event and exhaust the mixed rasterizer
        before the dynamic foliage phases begin.  This variant first applies
        the native culling rules, then admits only the highest-gradient
        clone/split candidates that fit both the global and per-event budgets.
        Split parents are still retired exactly as in native 2DGS.
        """
        max_points = int(max_points)
        max_growth = int(max_growth)
        if max_points <= 0 or max_growth <= 0:
            raise ValueError("Surface topology budgets must be positive")

        use_mip_filter = self.use_mip_filter
        if use_mip_filter:
            self.set_mip_filter(False)

        before = int(self.get_xyz.shape[0])
        grads = torch.nan_to_num(
            self.xyz_gradient_accum / self.denom,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = (
                self.get_scaling.max(dim=1).values > 0.1 * extent
            )
            prune_mask = torch.logical_or(
                prune_mask,
                torch.logical_or(big_points_vs, big_points_ws),
            )
        # Explicitly protected points are never removed by routine topology
        # maintenance.  Descendants inherit this metadata contract.
        if len(self._protected_flag) == len(prune_mask):
            prune_mask = torch.logical_and(
                prune_mask, ~self._protected_flag
            )
        pruned = int(prune_mask.sum())
        if pruned:
            grads = grads[~prune_mask]
            self.prune_points(prune_mask)

        point_count = int(self.get_xyz.shape[0])
        remaining = min(
            max(max_points - point_count, 0),
            max_growth,
        )
        cloned = 0
        split_parents = 0
        if remaining > 0 and point_count:
            grad_norm = torch.norm(grads, dim=-1)
            max_scale = self.get_scaling.max(dim=1).values
            clone_candidates = torch.logical_and(
                grad_norm >= max_grad,
                max_scale <= self.percent_dense * extent,
            )
            split_candidates = torch.logical_and(
                grads.squeeze(-1) >= max_grad,
                max_scale > self.percent_dense * extent,
            )
            if len(self._protected_flag) == point_count:
                # Metric anchors may seed conservative clones, but a native
                # split retires its parent. Never replace a fixed SfM/MAtCha
                # measurement with unanchored children.
                split_candidates = torch.logical_and(
                    split_candidates, ~self._protected_flag
                )
            candidate_mask = torch.logical_or(
                clone_candidates, split_candidates
            )
            candidate_indices = torch.nonzero(
                candidate_mask, as_tuple=False
            ).flatten()
            if candidate_indices.numel() > remaining:
                ranking = torch.topk(
                    grad_norm[candidate_indices], k=remaining
                ).indices
                candidate_indices = candidate_indices[ranking]

            admitted = torch.zeros(
                point_count, dtype=torch.bool, device=self.get_xyz.device
            )
            admitted[candidate_indices] = True
            clone_admitted = torch.logical_and(
                admitted, clone_candidates
            )
            split_admitted = torch.logical_and(
                admitted, split_candidates
            )
            cloned = int(clone_admitted.sum())
            split_parents = int(split_admitted.sum())

            clone_grads = torch.zeros_like(grads)
            clone_grads[clone_admitted] = grads[clone_admitted]
            split_grads = torch.zeros_like(grads)
            split_grads[split_admitted] = grads[split_admitted]
            if cloned:
                self.densify_and_clone(
                    clone_grads, max_grad, extent
                )
            if split_parents:
                self.densify_and_split(
                    split_grads, max_grad, extent
                )

        after = int(self.get_xyz.shape[0])
        if after > max_points:
            raise RuntimeError(
                f"Bounded densification exceeded its budget: "
                f"{after} > {max_points}"
            )
        torch.cuda.empty_cache()
        if use_mip_filter:
            self.set_mip_filter(True)
        return {
            "before": before,
            "pruned": pruned,
            "cloned": cloned,
            "split_parents": split_parents,
            "after": after,
            "budget": max_points,
            "remaining": max(max_points - after, 0),
        }

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def gs_scale_loss(self, max_scale_thresh=0.05):
        max_scale = self.get_scaling.max(dim=1).values
        excess = torch.clamp(max_scale - max_scale_thresh, min=0.0)
        loss = torch.sum(excess ** 2)
        return loss

def combine_gslist(gslist):
    """
    Combine a list of GaussianModel objects into a single GaussianModel object.
    
    Args:
        gslist: List of GaussianModel objects to combine
        
    Returns:
        A new GaussianModel instance containing all parameters from the input models
    """
    # Initialize a new GaussianModel object with the same SH degree as the first model in the list
    combined_model = GaussianModel(gslist[0].max_sh_degree)
    
    # Prepare lists to hold parameters from all models
    xyz_list = []
    features_dc_list = []
    features_rest_list = []
    opacity_list = []
    scaling_list = []
    rotation_list = []
    mip_filter_list = []
    # Collect parameters from each model
    for model in gslist:
        xyz_list.append(model.get_xyz.detach())
        features_dc_list.append(model._features_dc.detach())
        features_rest_list.append(model._features_rest.detach())
        opacity_list.append(model._opacity.detach())
        scaling_list.append(model._scaling.detach())
        rotation_list.append(model._rotation.detach())

        if hasattr(model, "mip_filter"):
            mip_filter_list.append(model.mip_filter.detach())
    
    # Concatenate all parameters
    combined_model._xyz = nn.Parameter(torch.cat(xyz_list, dim=0))
    combined_model._features_dc = nn.Parameter(torch.cat(features_dc_list, dim=0))
    combined_model._features_rest = nn.Parameter(torch.cat(features_rest_list, dim=0))
    combined_model._opacity = nn.Parameter(torch.cat(opacity_list, dim=0))
    combined_model._scaling = nn.Parameter(torch.cat(scaling_list, dim=0))
    combined_model._rotation = nn.Parameter(torch.cat(rotation_list, dim=0))

    if len(mip_filter_list) > 0:
        combined_model.set_mip_filter(True)
        combined_model.mip_filter = nn.Parameter(torch.cat(mip_filter_list, dim=0))
    assert combined_model.mip_filter.shape[0] == combined_model._xyz.shape[0]
    
    # Also initialize/combine other necessary properties
    combined_model.active_sh_degree = gslist[0].active_sh_degree
    
    # Initialize max_radii2D with the right size
    n_points = combined_model._xyz.shape[0]
    combined_model.max_radii2D = torch.zeros(n_points, device=combined_model._xyz.device)
    
    # Initialize other buffers with the appropriate sizes
    combined_model.xyz_gradient_accum = torch.zeros((n_points, 1), device=combined_model._xyz.device)
    combined_model.denom = torch.zeros((n_points, 1), device=combined_model._xyz.device)

    # NOTE: hard code pseudo optimizer
    l = [
        {'params': [combined_model._xyz], 'lr': 0.001, "name": "xyz"},
        {'params': [combined_model._features_dc], 'lr': 0.001, "name": "f_dc"},
        {'params': [combined_model._features_rest], 'lr': 0.001 / 20.0, "name": "f_rest"},
        {'params': [combined_model._opacity], 'lr': 0.001, "name": "opacity"},
        {'params': [combined_model._scaling], 'lr': 0.001, "name": "scaling"},
        {'params': [combined_model._rotation], 'lr': 0.001, "name": "rotation"}
    ]
    combined_model.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    
    # Copy spatial_lr_scale from the first model
    combined_model.spatial_lr_scale = gslist[0].spatial_lr_scale
    
    # Set percent_dense to the same as the first model
    combined_model.percent_dense = gslist[0].percent_dense
    
    # Print information about the combined model
    print(f"Combined {len(gslist)} models with a total of {n_points} points")
    for i, model in enumerate(gslist):
        print(f"  Model {i}: {model.get_xyz.shape[0]} points")
    
    return combined_model

def combine_gslist_simple(gslist):
    """
    Combine a list of GaussianModel objects into a single GaussianModel object.
    
    Args:
        gslist: List of GaussianModel objects to combine
        
    Returns:
        A new GaussianModel instance containing all parameters from the input models
    """
    # Initialize a new GaussianModel object with the same SH degree as the first model in the list
    combined_model = GaussianModel(gslist[0].max_sh_degree)
    
    # Prepare lists to hold parameters from all models
    xyz_list = []
    features_dc_list = []
    features_rest_list = []
    opacity_list = []
    scaling_list = []
    rotation_list = []

    # Collect parameters from each model
    for model in gslist:
        xyz_list.append(model.get_xyz.detach())
        features_dc_list.append(model._features_dc.detach())
        features_rest_list.append(model._features_rest.detach())
        opacity_list.append(model._opacity.detach())
        scaling_list.append(model._scaling.detach())
        rotation_list.append(model._rotation.detach())
    
    # Concatenate all parameters
    combined_model._xyz = nn.Parameter(torch.cat(xyz_list, dim=0))
    combined_model._features_dc = nn.Parameter(torch.cat(features_dc_list, dim=0))
    combined_model._features_rest = nn.Parameter(torch.cat(features_rest_list, dim=0))
    combined_model._opacity = nn.Parameter(torch.cat(opacity_list, dim=0))
    combined_model._scaling = nn.Parameter(torch.cat(scaling_list, dim=0))
    combined_model._rotation = nn.Parameter(torch.cat(rotation_list, dim=0))

    combined_model.max_radii2D = torch.zeros(combined_model._xyz.shape[0], device=combined_model._xyz.device)
    
    # Count of old Gaussians (for setting learning rates)
    gaussians_num = len(gslist)
    total_count = combined_model._xyz.shape[0]
    
    print(f'{gaussians_num} Gaussians are combined into one model, with {total_count} points')

    return combined_model

def get_obj_gaussian_by_mask(gaussian, obj_gs_mask):
    """
    Extract object-specific Gaussians based on a binary mask.
    
    Args:
        gaussian: Source GaussianModel containing all Gaussians
        obj_gs_mask: Binary mask indicating which Gaussians belong to the object
        
    Returns:
        A new GaussianModel instance containing only the Gaussians of the object
    """
    # Create a new GaussianModel with the same SH degree
    obj_gaussian = GaussianModel(gaussian.max_sh_degree)
    
    # Extract parameters for the selected Gaussians
    obj_gaussian._xyz = torch.nn.Parameter(gaussian._xyz[obj_gs_mask].clone().detach())
    obj_gaussian._features_dc = torch.nn.Parameter(gaussian._features_dc[obj_gs_mask].clone().detach())
    obj_gaussian._features_rest = torch.nn.Parameter(gaussian._features_rest[obj_gs_mask].clone().detach())
    obj_gaussian._scaling = torch.nn.Parameter(gaussian._scaling[obj_gs_mask].clone().detach())
    obj_gaussian._rotation = torch.nn.Parameter(gaussian._rotation[obj_gs_mask].clone().detach())
    obj_gaussian._opacity = torch.nn.Parameter(gaussian._opacity[obj_gs_mask].clone().detach())
    
    # Copy other necessary properties
    obj_gaussian.active_sh_degree = gaussian.active_sh_degree
    obj_gaussian.max_sh_degree = gaussian.max_sh_degree
    obj_gaussian.spatial_lr_scale = gaussian.spatial_lr_scale
    
    # Handle MIP filtering if used
    if hasattr(gaussian, 'use_mip_filter') and gaussian.use_mip_filter:
        obj_gaussian.set_mip_filter(True)
        if hasattr(gaussian, 'mip_filter'):
            obj_gaussian.mip_filter = gaussian.mip_filter[obj_gs_mask].clone().detach()
    
    # Initialize max_radii2D with the right size
    obj_gaussian.max_radii2D = torch.zeros(
        obj_gaussian._xyz.shape[0], device=obj_gaussian._xyz.device
    )
    
    # Print statistics
    print(f"Extracted {obj_gaussian._xyz.shape[0]} Gaussians for the object "
          f"(out of {gaussian._xyz.shape[0]} total Gaussians)")
    
    return obj_gaussian

def get_gaussian_normal(rotation, scaling, scale_modifier=1.0):

    q = torch.nn.functional.normalize(rotation, dim=-1)
    scales_3d = torch.cat([scaling * scale_modifier, torch.ones_like(scaling[:, :1])], dim=-1)
    L = build_scaling_rotation(scales_3d, q)  # (N, 3, 3), L = R * S
    normal = L[:, :, 2]

    return normal


def _camera_intrinsics_from_view(view):
    c2w = (view.world_view_transform.T).inverse()
    width, height = view.image_width, view.image_height
    ndc2pix = torch.tensor([
        [width / 2, 0, 0, width / 2],
        [0, height / 2, 0, height / 2],
        [0, 0, 0, 1],
    ], dtype=torch.float32, device=c2w.device).T
    projection_matrix = c2w.T @ view.full_proj_transform
    return (projection_matrix @ ndc2pix)[:3, :3].T


def _project_world_points_to_view(points_world, target_view):
    c2w = (target_view.world_view_transform.T).inverse()
    w2c = c2w.inverse()
    points_h = torch.cat([points_world, torch.ones_like(points_world[..., :1])], dim=-1)
    points_cam = points_h @ w2c.T
    points_cam = points_cam[..., :3]

    intrinsics = _camera_intrinsics_from_view(target_view)
    projected = points_cam @ intrinsics.T
    z = projected[..., 2]
    xy = projected[..., :2] / torch.clamp(z[..., None], min=1e-6)
    return xy, z


def _warp_coverage_mask(points_world, source_valid_mask, target_view, target_depth, depth_error_thresh):
    projected_xy, projected_depth = _project_world_points_to_view(points_world, target_view)
    height, width = target_depth.shape[-2:]
    in_image = (
        (projected_xy[..., 0] >= 0.0)
        & (projected_xy[..., 0] <= width - 1)
        & (projected_xy[..., 1] >= 0.0)
        & (projected_xy[..., 1] <= height - 1)
        & (projected_depth > 0.0)
        & source_valid_mask
    )

    u_int = torch.clamp(torch.round(projected_xy[..., 0]).long(), 0, width - 1)
    v_int = torch.clamp(torch.round(projected_xy[..., 1]).long(), 0, height - 1)
    target_depth = target_depth.squeeze()
    target_depth_sampled = target_depth[v_int, u_int]
    target_valid = target_depth_sampled > 0.0
    relative_depth_error = torch.abs(projected_depth - target_depth_sampled) / (projected_depth.abs() + 1e-6)
    depth_match = relative_depth_error < depth_error_thresh
    return in_image & target_valid & depth_match


def _points_to_normal_map(points_world):
    normals = torch.zeros_like(points_world)
    if points_world.shape[0] < 3 or points_world.shape[1] < 3:
        normals[..., 2] = 1.0
        return normals

    dx = points_world[2:, 1:-1] - points_world[:-2, 1:-1]
    dy = points_world[1:-1, 2:] - points_world[1:-1, :-2]
    normal_map = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    normals[1:-1, 1:-1] = normal_map
    normals[0, 1:-1] = normals[1, 1:-1]
    normals[-1, 1:-1] = normals[-2, 1:-1]
    normals[:, 0] = normals[:, 1]
    normals[:, -1] = normals[:, -2]
    return normals


def _matrix_to_quaternion(rotation_matrices):
    m = rotation_matrices
    qw = torch.sqrt(torch.clamp(1.0 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2], min=0.0)) / 2.0
    qx = torch.sqrt(torch.clamp(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], min=0.0)) / 2.0
    qy = torch.sqrt(torch.clamp(1.0 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], min=0.0)) / 2.0
    qz = torch.sqrt(torch.clamp(1.0 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2], min=0.0)) / 2.0

    qx = qx * torch.sign(m[:, 2, 1] - m[:, 1, 2] + 1e-12)
    qy = qy * torch.sign(m[:, 0, 2] - m[:, 2, 0] + 1e-12)
    qz = qz * torch.sign(m[:, 1, 0] - m[:, 0, 1] + 1e-12)
    return F.normalize(torch.stack([qw, qx, qy, qz], dim=-1), dim=-1)


def _normals_to_quaternions(normals):
    z_axis = F.normalize(normals, dim=-1)
    ref = torch.tensor([1.0, 0.0, 0.0], dtype=z_axis.dtype, device=z_axis.device).expand(z_axis.shape[0], -1).clone()
    is_parallel = torch.abs(z_axis[:, 0]) > 0.9
    ref[is_parallel] = torch.tensor([0.0, 1.0, 0.0], dtype=z_axis.dtype, device=z_axis.device)
    x_axis = F.normalize(torch.cross(ref, z_axis, dim=-1), dim=-1)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    rot_matrices = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    return _matrix_to_quaternion(rot_matrices)


def _points_to_distance_map(points_world):
    dist_h = torch.norm(points_world[:, 1:, :] - points_world[:, :-1, :], dim=-1)
    dist_v = torch.norm(points_world[1:, :, :] - points_world[:-1, :, :], dim=-1)

    if dist_h.shape[1] == 0 or dist_v.shape[0] == 0:
        return torch.full(points_world.shape[:2], 1e-3, dtype=points_world.dtype, device=points_world.device)

    dist_r = torch.cat([dist_h, dist_h[:, -1:]], dim=1)
    dist_l = torch.cat([dist_h[:, :1], dist_h], dim=1)
    dist_d = torch.cat([dist_v, dist_v[-1:, :]], dim=0)
    dist_u = torch.cat([dist_v[:1, :], dist_v], dim=0)
    return torch.stack([dist_r, dist_l, dist_d, dist_u], dim=0).min(dim=0).values


def get_gaussian_parameters_by_warp_from_depths(
    depths,
    views,
    depth_error_thresh=0.01,
    min_scale=0.0005,
    max_scale=0.05,
    downsample_pixel_grid_size=-1,
    valid_masks=None,
):
    """
    Initialize one Gaussian only for pixels that are not already covered by earlier
    initialized views under depth-consistent warping.
    """
    means = []
    scales = []
    quaternions = []
    colors = []
    initialized_view_ids = []

    if valid_masks is not None and len(valid_masks) != len(depths):
        raise ValueError("valid_masks must have one entry per depth/view")

    masked_depths = []
    for idx, depth in enumerate(depths):
        masked_depth = depth.squeeze().cuda()
        if valid_masks is not None:
            mask = valid_masks[idx].squeeze().to(masked_depth.device, dtype=torch.bool)
            if mask.shape != masked_depth.shape:
                raise ValueError(
                    f"valid mask {idx} has shape {tuple(mask.shape)}, expected {tuple(masked_depth.shape)}"
                )
            masked_depth = torch.where(mask, masked_depth, torch.zeros_like(masked_depth))
        masked_depths.append(masked_depth)

    for idx, (depth, view) in enumerate(tqdm(list(zip(masked_depths, views)), desc="Initializing Gaussians by warp")):
        depth = depth.squeeze().cuda()
        points_world = depths_to_points(view, depth).reshape(depth.shape[0], depth.shape[1], 3)
        valid_mask = depth > 0.0

        if downsample_pixel_grid_size > 0:
            downsample_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
            downsample_mask[::downsample_pixel_grid_size, ::downsample_pixel_grid_size] = True
        else:
            downsample_mask = torch.ones_like(valid_mask, dtype=torch.bool)

        covered_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        for initialized_idx in initialized_view_ids:
            covered_mask |= _warp_coverage_mask(
                points_world,
                valid_mask,
                views[initialized_idx],
                masked_depths[initialized_idx],
                depth_error_thresh,
            )

        keep_mask = ((~covered_mask) & downsample_mask & valid_mask).reshape(-1)
        if not keep_mask.any():
            initialized_view_ids.append(idx)
            continue

        distance_map = _points_to_distance_map(points_world)
        scale_values = distance_map.reshape(-1)[keep_mask] / 2.0
        if downsample_pixel_grid_size > 0:
            scale_values = scale_values * downsample_pixel_grid_size
        view_scales = scale_values[..., None].repeat(1, 2)

        normal_map = _points_to_normal_map(points_world)
        view_normals = normal_map.reshape(-1, 3)[keep_mask]
        view_quaternions = _normals_to_quaternions(view_normals)
        view_colors = view.original_image.cuda().permute(1, 2, 0).reshape(-1, 3)[keep_mask]

        means.append(points_world.reshape(-1, 3)[keep_mask])
        scales.append(view_scales)
        quaternions.append(view_quaternions)
        colors.append(view_colors)
        initialized_view_ids.append(idx)

    if len(means) == 0:
        raise RuntimeError("Warp-based Gaussian initialization produced no valid points.")

    means = torch.cat(means, dim=0)
    scales = torch.cat(scales, dim=0)
    quaternions = torch.cat(quaternions, dim=0)
    colors = torch.cat(colors, dim=0)

    valid_scale_mask = scales[..., 0] < max_scale
    return (
        means[valid_scale_mask],
        scales[valid_scale_mask].clamp(min=min_scale),
        quaternions[valid_scale_mask],
        colors[valid_scale_mask],
    )
