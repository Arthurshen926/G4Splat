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


class VolumetricFoliageModel(nn.Module):
    """Independent volumetric 3DGS foliage branch."""

    def __init__(self, sh_degree: int, *, device: str | torch.device = "cuda"):
        super().__init__()
        self.sh_degree = int(sh_degree)
        device = torch.device(device)
        coefficients = (self.sh_degree + 1) ** 2
        self.xyz = nn.Parameter(torch.empty(0, 3, device=device))
        self.log_scales = nn.Parameter(torch.empty(0, 3, device=device))
        self.quaternions = nn.Parameter(torch.empty(0, 4, device=device))
        self.opacity_logits = nn.Parameter(torch.empty(0, 1, device=device))
        self.features = nn.Parameter(
            torch.empty(0, coefficients, 3, device=device)
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
        self._replace(
            xyz=xyz,
            log_scales=scales.log(),
            quaternions=payload["quaternions"].to(device=device, dtype=dtype),
            opacity_logits=torch.logit(opacity.clamp(1e-6, 1 - 1e-6)),
            features=features,
        )
        return int(len(xyz))

    def _replace(self, **values: torch.Tensor) -> None:
        for name, value in values.items():
            setattr(self, name, nn.Parameter(value.requires_grad_(True)))

    def capture(self) -> dict:
        return {
            "sh_degree": self.sh_degree,
            "xyz": self.xyz.detach(),
            "log_scales": self.log_scales.detach(),
            "quaternions": self.quaternions.detach(),
            "opacity_logits": self.opacity_logits.detach(),
            "features": self.features.detach(),
        }

    def restore(self, payload: dict) -> None:
        if int(payload["sh_degree"]) != self.sh_degree:
            raise RuntimeError("Foliage SH degree mismatch")
        self._replace(
            **{
                name: payload[name].to(device=self.xyz.device)
                for name in (
                    "xyz",
                    "log_scales",
                    "quaternions",
                    "opacity_logits",
                    "features",
                )
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
    volume_means = foliage.xyz
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
    volume_colors = sh_colors(volume_means, foliage.features, degree)
    colors = torch.cat([surface_colors, volume_colors], dim=0)
    opacities = torch.cat(
        [surface_opacity, foliage.opacities],
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
    )
