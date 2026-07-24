"""Jointly depth-sorted structural surfels and volumetric foliage.

The structural model remains the audited 2DGS parameterisation (two tangent
scales).  For joint rasterisation it is embedded as a very thin 3D covariance.
Foliage uses three independently trainable scales.  Both primitive families
are passed to one gsplat call, so occlusion is resolved by one global depth
sort rather than by a fixed post-render compositing order.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys

import torch
from torch import nn

@dataclass
class HybridRenderOutput:
    render: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor
    radii: torch.Tensor
    means2d: torch.Tensor | None
    structural_count: int


class VolumetricFoliageModel(nn.Module):
    """Small append-only 3DGS residual branch."""

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
    def append_from_structural(
        self,
        structural,
        indices: torch.Tensor,
        *,
        normal_scale_ratio: float = 0.65,
        initial_opacity: float = 0.02,
        jitter_fraction: float = 0.35,
    ) -> int:
        indices = indices.reshape(-1).long()
        if indices.numel() == 0:
            return 0
        tangent_scales = structural.get_scaling[indices].detach()
        base_scale = tangent_scales.mean(dim=-1, keepdim=True)
        volume_scales = torch.cat(
            [
                tangent_scales,
                base_scale * float(normal_scale_ratio),
            ],
            dim=-1,
        ).clamp_min(1e-6)
        rotation = structural.get_rotation[indices].detach()
        # Jitter in all three local axes.  This breaks the inherited sheet
        # degeneracy while keeping every seed near multi-view residual support.
        noise_local = torch.randn_like(volume_scales) * volume_scales
        rotation_matrix = _quaternion_to_rotation(rotation)
        jitter = torch.bmm(
            rotation_matrix, noise_local.unsqueeze(-1)
        ).squeeze(-1) * float(jitter_fraction)
        xyz = structural.get_xyz[indices].detach() + jitter
        features = structural.get_features[indices].detach()
        opacity = torch.full(
            (len(indices), 1),
            float(initial_opacity),
            device=xyz.device,
            dtype=xyz.dtype,
        )
        opacity_logits = torch.logit(opacity.clamp(1e-6, 1.0 - 1e-6))
        self._replace(
            xyz=torch.cat([self.xyz.detach(), xyz], dim=0),
            log_scales=torch.cat(
                [self.log_scales.detach(), volume_scales.log()], dim=0
            ),
            quaternions=torch.cat(
                [self.quaternions.detach(), rotation], dim=0
            ),
            opacity_logits=torch.cat(
                [self.opacity_logits.detach(), opacity_logits], dim=0
            ),
            features=torch.cat([self.features.detach(), features], dim=0),
        )
        return int(len(indices))

    @torch.no_grad()
    def initialize_from_surfel_residual(
        self,
        surfels,
        mask: torch.Tensor,
        *,
        normal_scale_ratio: float = 0.65,
    ) -> int:
        """Lift audited persistent-residual surfels into three-axis 3DGS."""
        if len(self):
            raise RuntimeError("Volumetric foliage is already initialized")
        indices = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
        tangent = surfels.get_scaling[indices].detach()
        normal = (
            tangent.mean(dim=-1, keepdim=True) * float(normal_scale_ratio)
        ).clamp_min(1e-6)
        self._replace(
            xyz=surfels.get_xyz[indices].detach().clone(),
            log_scales=torch.cat([tangent, normal], dim=-1).log(),
            quaternions=surfels.get_rotation[indices].detach().clone(),
            opacity_logits=surfels._opacity[indices].detach().clone(),
            features=surfels.get_features[indices].detach().clone(),
        )
        return int(len(indices))

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


def _quaternion_to_rotation(quaternion: torch.Tensor) -> torch.Tensor:
    q = torch.nn.functional.normalize(quaternion, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
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


def render_hybrid(
    camera,
    structural,
    foliage: VolumetricFoliageModel,
    *,
    background: torch.Tensor,
    structural_thickness_ratio: float = 0.02,
    structural_scale_ceiling: float | None = None,
    radius_clip: float = 0.0,
) -> HybridRenderOutput:
    """Rasterise both branches with one covariance projection and depth sort."""
    try:
        import gsplat

        # Binary-less gsplat wheels JIT-build ``gsplat_cuda`` but then probe
        # for ``gsplat.csrc`` again in every fresh process.  Reuse the cached
        # extension when present; otherwise gsplat retains its normal JIT
        # fallback and produces the actionable build error.
        if "csrc" not in gsplat.__dict__:
            from torch.utils.cpp_extension import _get_build_directory

            build_directory = _get_build_directory("gsplat_cuda", verbose=False)
            if build_directory not in sys.path:
                sys.path.insert(0, build_directory)
            try:
                extension = importlib.import_module("gsplat_cuda")
            except ImportError:
                extension = None
            if extension is not None:
                gsplat.csrc = extension
                sys.modules["gsplat.csrc"] = extension
        rasterization = gsplat.rasterization
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("Hybrid foliage requires gsplat") from exc

    structural_count = int(len(structural.get_xyz))
    structural_tangent = structural.get_scaling
    if structural_scale_ceiling is not None:
        structural_tangent = structural_tangent.clamp_max(
            float(structural_scale_ceiling)
        )
    structural_thickness = (
        structural_tangent.amin(dim=-1, keepdim=True)
        * float(structural_thickness_ratio)
    ).clamp_min(1e-6)
    structural_scales = torch.cat(
        [structural_tangent, structural_thickness], dim=-1
    )
    means = torch.cat([structural.get_xyz.detach(), foliage.xyz], dim=0)
    scales = torch.cat([structural_scales.detach(), foliage.scales], dim=0)
    quaternions = torch.cat(
        [structural.get_rotation.detach(), foliage.normalized_quaternions],
        dim=0,
    )
    opacities = torch.cat(
        [structural.get_opacity.detach().squeeze(-1), foliage.opacities],
        dim=0,
    )
    features = torch.cat(
        [structural.get_features.detach(), foliage.features], dim=0
    )
    viewmat = camera.world_view_transform.transpose(0, 1).unsqueeze(0)
    intrinsics = torch.tensor(
        [
            [camera.focal_x, 0.0, camera.cx],
            [0.0, camera.focal_y, camera.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=means.dtype,
        device=means.device,
    ).unsqueeze(0)
    rendered, alpha, meta = rasterization(
        means=means,
        quats=quaternions,
        scales=scales,
        opacities=opacities,
        colors=features,
        viewmats=viewmat,
        Ks=intrinsics,
        width=int(camera.image_width),
        height=int(camera.image_height),
        near_plane=float(camera.znear),
        far_plane=float(camera.zfar),
        radius_clip=float(radius_clip),
        sh_degree=int(structural.active_sh_degree),
        # gsplat 1.5.3's packed path drops the camera batch dimension before
        # validating a non-None background.  The one-camera dense projection
        # keeps exact white-background semantics and remains bounded here.
        packed=False,
        backgrounds=background.reshape(1, 3),
        render_mode="RGB+ED",
    )
    rgb = rendered[0, ..., :3].permute(2, 0, 1)
    depth = rendered[0, ..., 3:4].permute(2, 0, 1)
    alpha_chw = alpha[0].permute(2, 0, 1)
    return HybridRenderOutput(
        render=rgb,
        alpha=alpha_chw,
        depth=depth,
        radii=meta["radii"],
        means2d=meta.get("means2d"),
        structural_count=structural_count,
    )
