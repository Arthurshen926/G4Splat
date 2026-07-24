"""Native jointly sorted structural surfels and volumetric foliage.

Structural primitives retain the exact perspective-correct 2DGS ray/surfel
projection. Foliage uses a true three-axis 3D EWA footprint. Both families
emit into one CUDA tile list, share one center-depth radix sort, and are
interleaved in the same front-to-back pixel loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

PRIMITIVE_HULL = 0
PRIMITIVE_SFM_TRACK = 1

LAYER_CANONICAL_CROWN = 0
LAYER_STATIC_SKELETON = 1
LAYER_DYNAMIC_LEAF = 2


@dataclass
class HybridRenderOutput:
    render: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor
    median_depth: torch.Tensor
    distortion: torch.Tensor
    radii: torch.Tensor
    means2d: torch.Tensor | None
    structural_count: int
    responsibility: torch.Tensor | None = None
    surface_alpha: torch.Tensor | None = None
    volume_alpha: torch.Tensor | None = None
    surface_depth: torch.Tensor | None = None
    volume_depth: torch.Tensor | None = None
    surface_means2d: torch.Tensor | None = None
    volume_means2d: torch.Tensor | None = None


class VolumetricFoliageModel(nn.Module):
    """Layered foliage 3DGS with persistent geometry-evidence metadata."""

    def __init__(
        self,
        sh_degree: int,
        *,
        dynamic_rank: int = 4,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.sh_degree = int(sh_degree)
        self.dynamic_rank = int(dynamic_rank)
        device = torch.device(device)
        coefficients = (self.sh_degree + 1) ** 2
        self.xyz = nn.Parameter(torch.empty(0, 3, device=device))
        self.log_scales = nn.Parameter(torch.empty(0, 3, device=device))
        self.quaternions = nn.Parameter(torch.empty(0, 4, device=device))
        self.opacity_logits = nn.Parameter(torch.empty(0, 1, device=device))
        self.features = nn.Parameter(
            torch.empty(0, coefficients, 3, device=device)
        )
        self.deformation_basis = nn.Parameter(
            torch.empty(0, self.dynamic_rank, 3, device=device)
        )
        self.dynamic_feature_basis = nn.Parameter(
            torch.empty(0, self.dynamic_rank, 3, device=device)
        )
        self.dynamic_opacity_basis = nn.Parameter(
            torch.empty(0, self.dynamic_rank, 1, device=device)
        )
        self.register_buffer(
            "primitive_role", torch.empty(0, dtype=torch.int8, device=device)
        )
        self.register_buffer(
            "layer_role", torch.empty(0, dtype=torch.int8, device=device)
        )
        self.register_buffer(
            "track_id", torch.empty(0, dtype=torch.int64, device=device)
        )
        self.register_buffer(
            "tree_instance_id",
            torch.empty(0, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "support_camera_ids",
            torch.empty(0, 0, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "support_view_count",
            torch.empty(0, dtype=torch.int16, device=device),
        )
        self.register_buffer(
            "support_sequence_count",
            torch.empty(0, dtype=torch.int16, device=device),
        )
        self.register_buffer(
            "track_linearity", torch.empty(0, device=device)
        )
        self.register_buffer(
            "occupancy_probability", torch.empty(0, device=device)
        )
        self.register_buffer(
            "position_covariance", torch.empty(0, 3, 3, device=device)
        )
        self.register_buffer(
            "reprojection_error", torch.empty(0, device=device)
        )
        self.register_buffer(
            "initialization_source",
            torch.empty(0, dtype=torch.int8, device=device),
        )
        self.register_buffer(
            "initialization_center", torch.empty(0, 3, device=device)
        )

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def scales(self) -> torch.Tensor:
        return self.log_scales.exp()

    @property
    def opacities(self) -> torch.Tensor:
        return self.opacity_logits.sigmoid().squeeze(-1)

    @property
    def normalized_quaternions(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self.quaternions, dim=-1)

    @property
    def static_skeleton_mask(self) -> torch.Tensor:
        return self.layer_role == LAYER_STATIC_SKELETON

    @property
    def dynamic_leaf_mask(self) -> torch.Tensor:
        return self.layer_role == LAYER_DYNAMIC_LEAF

    @property
    def canonical_crown_mask(self) -> torch.Tensor:
        return self.layer_role == LAYER_CANONICAL_CROWN

    def conditioned_state(
        self,
        temporal_code: torch.Tensor | None,
        *,
        include_dynamic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return xyz/features/opacities for canonical or conditioned output."""
        xyz = self.xyz
        features = self.features
        opacity_logits = self.opacity_logits.squeeze(-1)
        dynamic = self.dynamic_leaf_mask
        if temporal_code is not None and include_dynamic and bool(dynamic.any()):
            code = temporal_code.to(device=xyz.device, dtype=xyz.dtype)
            if code.shape != (self.dynamic_rank,):
                raise ValueError(
                    f"temporal code must have shape {(self.dynamic_rank,)}, "
                    f"got {tuple(code.shape)}"
                )
            xyz = xyz + torch.einsum(
                "nrc,r->nc", self.deformation_basis, code
            ) * dynamic[:, None]
            feature_delta = torch.einsum(
                "nrc,r->nc", self.dynamic_feature_basis, code
            )
            features = features.clone()
            features[:, 0] = features[:, 0] + feature_delta * dynamic[:, None]
            opacity_logits = opacity_logits + torch.einsum(
                "nrc,r->nc", self.dynamic_opacity_basis, code
            ).squeeze(-1) * dynamic
        opacities = opacity_logits.sigmoid()
        if not include_dynamic and bool(dynamic.any()):
            opacities = opacities * (~dynamic).to(opacities.dtype)
        return xyz, features, opacities

    @torch.no_grad()
    def initialize_from_volume_state(self, payload: dict) -> int:
        """Initialize from independent SfM/visual-hull evidence."""
        if len(self):
            raise RuntimeError("Volumetric foliage is already initialized")
        if payload.get("version") != "independent_sfm_semantic_canopy_volume_v1":
            raise RuntimeError("Unsupported independent foliage volume state")
        device, dtype = self.xyz.device, self.xyz.dtype
        xyz = payload["centers"].to(device=device, dtype=dtype)
        scales = payload["scales"].to(device=device, dtype=dtype).clamp_min(1e-6)
        colors = payload["colors"].to(device=device, dtype=dtype).clamp(0, 1)
        coefficients = (self.sh_degree + 1) ** 2
        features = torch.zeros(
            len(xyz), coefficients, 3, device=device, dtype=dtype
        )
        features[:, 0] = (colors - 0.5) / 0.28209479177387814
        opacity = payload["opacities"].to(device=device, dtype=dtype)
        count = len(xyz)
        primitive_role = payload.get(
            "primitive_role", torch.zeros(count, dtype=torch.int8)
        ).to(device=device, dtype=torch.int8)
        track_linearity = payload.get(
            "track_linearity", torch.zeros(count)
        ).to(device=device, dtype=dtype)
        layer_role = payload.get("layer_role")
        if layer_role is None:
            layer_role = torch.full(
                (count,),
                LAYER_CANONICAL_CROWN,
                dtype=torch.int8,
                device=device,
            )
            skeleton = (
                (primitive_role == PRIMITIVE_SFM_TRACK)
                & (track_linearity >= 2.0)
            )
            layer_role[skeleton] = LAYER_STATIC_SKELETON
        else:
            layer_role = layer_role.to(device=device, dtype=torch.int8)
        support_camera_ids = payload.get("support_camera_ids")
        if support_camera_ids is None:
            support_camera_ids = torch.full(
                (count, 0), -1, dtype=torch.int32
            )
        track_id = payload.get("track_id")
        if track_id is None:
            track_id = torch.full((count,), -1, dtype=torch.int64)
            track = primitive_role.cpu() == PRIMITIVE_SFM_TRACK
            track_id[track] = torch.arange(
                int(track.sum()), dtype=torch.int64
            )
        position_covariance = payload.get(
            "position_covariance", torch.diag_embed(scales.square())
        )
        self._replace(
            xyz=xyz,
            log_scales=scales.log(),
            quaternions=payload["quaternions"].to(device=device, dtype=dtype),
            opacity_logits=torch.logit(opacity.clamp(1e-6, 1 - 1e-6)),
            features=features,
            deformation_basis=torch.zeros(
                count, self.dynamic_rank, 3, device=device, dtype=dtype
            ),
            dynamic_feature_basis=torch.zeros(
                count, self.dynamic_rank, 3, device=device, dtype=dtype
            ),
            dynamic_opacity_basis=torch.zeros(
                count, self.dynamic_rank, 1, device=device, dtype=dtype
            ),
        )
        metadata = {
            "primitive_role": primitive_role,
            "layer_role": layer_role,
            "track_id": track_id.to(device=device, dtype=torch.int64),
            "tree_instance_id": payload.get(
                "tree_instance_id",
                torch.full((count,), -1, dtype=torch.int32),
            ).to(device=device, dtype=torch.int32),
            "support_camera_ids": support_camera_ids.to(
                device=device, dtype=torch.int32
            ),
            "support_view_count": payload.get(
                "support_view_count", torch.zeros(count, dtype=torch.int16)
            ).to(device=device, dtype=torch.int16),
            "support_sequence_count": payload.get(
                "support_sequence_count", torch.zeros(count, dtype=torch.int16)
            ).to(device=device, dtype=torch.int16),
            "track_linearity": track_linearity,
            "occupancy_probability": payload.get(
                "occupancy_probability", torch.ones(count)
            ).to(device=device, dtype=dtype),
            "position_covariance": position_covariance.to(
                device=device, dtype=dtype
            ),
            "reprojection_error": payload.get(
                "reprojection_error", torch.full((count,), float("nan"))
            ).to(device=device, dtype=dtype),
            "initialization_source": payload.get(
                "initialization_source", primitive_role
            ).to(device=device, dtype=torch.int8),
            "initialization_center": payload.get(
                "initialization_center", xyz
            ).to(device=device, dtype=dtype),
        }
        self._replace_buffers(**metadata)
        return int(len(xyz))

    def _replace(self, **values: torch.Tensor) -> None:
        for name, value in values.items():
            setattr(self, name, nn.Parameter(value.requires_grad_(True)))

    def _replace_buffers(self, **values: torch.Tensor) -> None:
        for name, value in values.items():
            setattr(self, name, value)

    @property
    def metadata_names(self) -> tuple[str, ...]:
        return (
            "primitive_role",
            "layer_role",
            "track_id",
            "tree_instance_id",
            "support_camera_ids",
            "support_view_count",
            "support_sequence_count",
            "track_linearity",
            "occupancy_probability",
            "position_covariance",
            "reprojection_error",
            "initialization_source",
            "initialization_center",
        )

    @torch.no_grad()
    def append_dynamic_leaves(
        self,
        indices: torch.Tensor,
        *,
        opacity_scale: float = 0.35,
        scale_factor: float = 0.65,
        seed: int = 1701,
    ) -> int:
        """Clone evidence-backed crown/track points into a dynamic residual layer."""
        indices = torch.as_tensor(
            indices, device=self.xyz.device, dtype=torch.long
        ).unique()
        if not indices.numel():
            return 0
        generator = torch.Generator(device=self.xyz.device)
        generator.manual_seed(int(seed))
        parameter_values = {}
        for name in (
            "xyz",
            "log_scales",
            "quaternions",
            "opacity_logits",
            "features",
            "deformation_basis",
            "dynamic_feature_basis",
            "dynamic_opacity_basis",
        ):
            value = getattr(self, name).detach()
            clone = value[indices].clone()
            if name == "log_scales":
                clone = clone + math.log(float(scale_factor))
            elif name == "opacity_logits":
                opacity = clone.sigmoid() * float(opacity_scale)
                clone = torch.logit(opacity.clamp(1e-6, 1 - 1e-6))
            elif name == "deformation_basis":
                clone.normal_(0.0, 0.002, generator=generator)
            parameter_values[name] = torch.cat([value, clone], dim=0)
        metadata = {}
        for name in self.metadata_names:
            value = getattr(self, name)
            clone = value[indices].clone()
            if name == "layer_role":
                clone.fill_(LAYER_DYNAMIC_LEAF)
            metadata[name] = torch.cat([value, clone], dim=0)
        self._replace(**parameter_values)
        self._replace_buffers(**metadata)
        return int(indices.numel())

    @torch.no_grad()
    def split(
        self,
        indices: torch.Tensor,
        *,
        shrink: float = 1.6,
    ) -> dict[str, int]:
        """Replace selected non-skeleton volumes with two oriented children."""
        indices = torch.as_tensor(
            indices, device=self.xyz.device, dtype=torch.long
        ).unique()
        if indices.numel():
            indices = indices[~self.static_skeleton_mask[indices]]
        if not indices.numel():
            return {"split_parents": 0, "children": 0}
        keep = torch.ones(len(self), dtype=torch.bool, device=self.xyz.device)
        keep[indices] = False
        scales = self.scales.detach()[indices]
        axis = scales.argmax(dim=-1)
        local = torch.zeros_like(scales)
        local.scatter_(1, axis[:, None], 0.35 * scales.gather(1, axis[:, None]))
        quaternion = self.normalized_quaternions.detach()[indices]
        w, x, y, z = quaternion.unbind(-1)
        rotation = torch.stack(
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ],
            dim=-1,
        ).reshape(-1, 3, 3)
        offset = torch.bmm(rotation, local[:, :, None]).squeeze(-1)
        parameter_values = {}
        for name in (
            "xyz",
            "log_scales",
            "quaternions",
            "opacity_logits",
            "features",
            "deformation_basis",
            "dynamic_feature_basis",
            "dynamic_opacity_basis",
        ):
            value = getattr(self, name).detach()
            child = value[indices].repeat_interleave(2, dim=0)
            if name == "xyz":
                child[0::2] -= offset
                child[1::2] += offset
            elif name == "log_scales":
                child -= math.log(float(shrink))
            elif name == "opacity_logits":
                opacity = 1.0 - torch.sqrt(
                    (1.0 - child.sigmoid()).clamp_min(1e-6)
                )
                child = torch.logit(opacity.clamp(1e-6, 1 - 1e-6))
            parameter_values[name] = torch.cat([value[keep], child], dim=0)
        metadata = {
            name: torch.cat(
                [
                    getattr(self, name)[keep],
                    getattr(self, name)[indices].repeat_interleave(2, dim=0),
                ],
                dim=0,
            )
            for name in self.metadata_names
        }
        self._replace(**parameter_values)
        self._replace_buffers(**metadata)
        return {
            "split_parents": int(indices.numel()),
            "children": int(2 * indices.numel()),
        }

    @torch.no_grad()
    def prune(self, remove: torch.Tensor) -> int:
        remove = torch.as_tensor(
            remove, device=self.xyz.device, dtype=torch.bool
        )
        if remove.shape != (len(self),):
            raise ValueError("prune mask shape mismatch")
        remove = remove & ~self.static_skeleton_mask
        keep = ~remove
        count = int(remove.sum())
        if not count:
            return 0
        self._replace(
            **{
                name: getattr(self, name).detach()[keep]
                for name in (
                    "xyz",
                    "log_scales",
                    "quaternions",
                    "opacity_logits",
                    "features",
                    "deformation_basis",
                    "dynamic_feature_basis",
                    "dynamic_opacity_basis",
                )
            }
        )
        self._replace_buffers(
            **{
                name: getattr(self, name)[keep]
                for name in self.metadata_names
            }
        )
        return count

    def capture(self) -> dict:
        return {
            "sh_degree": self.sh_degree,
            "dynamic_rank": self.dynamic_rank,
            "xyz": self.xyz.detach(),
            "log_scales": self.log_scales.detach(),
            "quaternions": self.quaternions.detach(),
            "opacity_logits": self.opacity_logits.detach(),
            "features": self.features.detach(),
            "deformation_basis": self.deformation_basis.detach(),
            "dynamic_feature_basis": self.dynamic_feature_basis.detach(),
            "dynamic_opacity_basis": self.dynamic_opacity_basis.detach(),
            **{
                name: getattr(self, name).detach()
                for name in self.metadata_names
            },
        }

    def restore(self, payload: dict) -> None:
        if int(payload["sh_degree"]) != self.sh_degree:
            raise RuntimeError("Foliage SH degree mismatch")
        if int(payload.get("dynamic_rank", self.dynamic_rank)) != self.dynamic_rank:
            raise RuntimeError("Foliage dynamic rank mismatch")
        device = self.xyz.device
        count = int(payload["xyz"].shape[0])
        dtype = payload["xyz"].dtype
        dynamic_defaults = {
            "deformation_basis": torch.zeros(
                count, self.dynamic_rank, 3, dtype=dtype
            ),
            "dynamic_feature_basis": torch.zeros(
                count, self.dynamic_rank, 3, dtype=dtype
            ),
            "dynamic_opacity_basis": torch.zeros(
                count, self.dynamic_rank, 1, dtype=dtype
            ),
        }
        self._replace(
            **{
                name: payload.get(name, dynamic_defaults.get(name)).to(
                    device=device
                )
                for name in (
                    "xyz",
                    "log_scales",
                    "quaternions",
                    "opacity_logits",
                    "features",
                    "deformation_basis",
                    "dynamic_feature_basis",
                    "dynamic_opacity_basis",
                )
            }
        )
        metadata_defaults = {
            "primitive_role": torch.zeros(count, dtype=torch.int8),
            "layer_role": torch.zeros(count, dtype=torch.int8),
            "track_id": torch.full((count,), -1, dtype=torch.int64),
            "tree_instance_id": torch.full(
                (count,), -1, dtype=torch.int32
            ),
            "support_camera_ids": torch.empty(count, 0, dtype=torch.int32),
            "support_view_count": torch.zeros(count, dtype=torch.int16),
            "support_sequence_count": torch.zeros(count, dtype=torch.int16),
            "track_linearity": torch.zeros(count),
            "occupancy_probability": torch.ones(count),
            "position_covariance": torch.diag_embed(
                payload["log_scales"].exp().square()
            ),
            "reprojection_error": torch.full((count,), float("nan")),
            "initialization_source": torch.zeros(count, dtype=torch.int8),
            "initialization_center": payload["xyz"].clone(),
        }
        self._replace_buffers(
            **{
                name: payload.get(name, metadata_defaults[name]).to(
                    device=device
                )
                for name in self.metadata_names
            }
        )


def render_hybrid(
    camera,
    structural,
    foliage: VolumetricFoliageModel,
    *,
    background: torch.Tensor,
    structural_thickness_ratio: float = 0.02,
    structural_scale_ceiling: float | None = None,
    radius_clip: float = 0.0,
    surface_gate: torch.Tensor | None = None,
    audit_fields: torch.Tensor | None = None,
    temporal_code: torch.Tensor | None = None,
    include_dynamic: bool = False,
) -> HybridRenderOutput:
    """Rasterise exact 2D surfels and 3D volumes with native mixed CUDA."""
    del structural_thickness_ratio, radius_clip
    from diff_surfel_rasterization import (
        GaussianRasterizationSettings,
        MixedGaussianRasterizer,
    )
    from utils.sh_utils import eval_sh

    structural_count = int(len(structural.get_xyz))
    structural_tangent = structural.get_scaling
    if structural_scale_ceiling is not None:
        structural_tangent = structural_tangent.clamp_max(
            float(structural_scale_ceiling)
        )
    surface_means = structural.get_xyz.detach()
    surface_scales = structural_tangent.detach()
    surface_rotations = structural.get_rotation.detach()
    surface_base_opacity = structural.get_opacity.detach().reshape(-1)
    if surface_gate is None:
        surface_gate = torch.ones_like(surface_base_opacity)
    if surface_gate.shape != surface_base_opacity.shape:
        raise ValueError(
            f"surface_gate has shape {tuple(surface_gate.shape)}, expected "
            f"{tuple(surface_base_opacity.shape)}"
        )
    # Direct multiplicative gates initialize to an exactly representable 1.
    # Clamping is intentionally outside the parameter state so the parent
    # opacity remains immutable and bit-identical at zero step.
    surface_opacity = surface_base_opacity * surface_gate.clamp(0.0, 1.0)
    volume_means, volume_features, volume_opacities = (
        foliage.conditioned_state(
            temporal_code,
            include_dynamic=include_dynamic,
        )
    )
    volume_scales = foliage.scales
    volume_rotations = foliage.normalized_quaternions

    def sh_colors(
        xyz: torch.Tensor,
        features: torch.Tensor,
        degree: int,
    ) -> torch.Tensor:
        if xyz.shape[0] == 0:
            return xyz.new_empty((0, 3))
        coefficients = features.transpose(1, 2).reshape(
            -1, 3, features.shape[1]
        )
        direction = xyz - camera.camera_center.reshape(1, 3)
        direction = torch.nn.functional.normalize(direction, dim=-1)
        return torch.clamp_min(eval_sh(degree, coefficients, direction) + 0.5, 0)

    degree = min(int(structural.active_sh_degree), int(foliage.sh_degree))
    surface_colors = sh_colors(
        surface_means,
        structural.get_features.detach(),
        degree,
    ).detach()
    volume_colors = sh_colors(volume_means, volume_features, degree)
    colors = torch.cat([surface_colors, volume_colors], dim=0)
    opacities = torch.cat(
        [surface_opacity, volume_opacities],
        dim=0,
    ).reshape(-1, 1)

    surface_means2D = torch.zeros_like(
        surface_means, requires_grad=surface_gate.requires_grad
    )
    volume_means2D = torch.zeros_like(volume_means, requires_grad=True)
    try:
        surface_means2D.retain_grad()
        volume_means2D.retain_grad()
    except RuntimeError:
        pass
    tanfovx = math.tan(float(camera.FoVx) * 0.5)
    tanfovy = math.tan(float(camera.FoVy) * 0.5)
    settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background,
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=False,
    )
    rgb, radii, allmap, responsibility = MixedGaussianRasterizer(settings)(
        surface_means,
        surface_means2D,
        surface_scales,
        surface_rotations,
        volume_means,
        volume_means2D,
        volume_scales,
        volume_rotations,
        colors,
        opacities,
        audit_fields,
    )
    alpha_chw = allmap[1:2]
    depth = torch.nan_to_num(
        allmap[0:1] / alpha_chw.clamp_min(1e-8), 0.0, 0.0
    )
    median_depth = torch.nan_to_num(allmap[5:6], 0.0, 0.0)
    surface_alpha = allmap[7:8]
    volume_alpha = allmap[8:9]
    surface_depth = torch.nan_to_num(
        allmap[9:10] / surface_alpha.clamp_min(1e-8), 0.0, 0.0
    )
    volume_depth = torch.nan_to_num(
        allmap[10:11] / volume_alpha.clamp_min(1e-8), 0.0, 0.0
    )
    return HybridRenderOutput(
        render=rgb,
        alpha=alpha_chw,
        depth=depth,
        median_depth=median_depth,
        distortion=allmap[6:7],
        radii=radii,
        means2d=torch.cat([surface_means2D, volume_means2D], dim=0),
        structural_count=structural_count,
        responsibility=(
            responsibility if responsibility.numel() else None
        ),
        surface_alpha=surface_alpha,
        volume_alpha=volume_alpha,
        surface_depth=surface_depth,
        volume_depth=volume_depth,
        surface_means2d=surface_means2D,
        volume_means2d=volume_means2D,
    )
