"""Protected-structure depth buffers for occlusion-aware foliage evidence."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch


class ProtectedDepthRenderer:
    """Reuse one trained rigid surface for several camera batches.

    Foliage initialization first requests z-buffers for DAV2 alignment and
    can later discover a few additional posterior cameras.  Reloading a
    million-surfel PLY for the second request costs both disk bandwidth and a
    second CUDA allocation even though the model is immutable.  This small
    callable owns the model for the lifetime of one initialization and keeps
    the public functional wrapper backward compatible.
    """

    def __init__(
        self,
        *,
        structural_ply: Path,
        replacement_audit: Path | None = None,
        sh_degree: int = 3,
        resolution_scale: float = 0.5,
    ) -> None:
        from scene import GaussianModel
        from types import SimpleNamespace

        self.structural_ply = Path(structural_ply).resolve()
        self.resolution_scale = float(resolution_scale)
        self.model = GaussianModel(int(sh_degree))
        self.model.load_ply(str(self.structural_ply))
        if replacement_audit is not None:
            try:
                payload = torch.load(
                    Path(replacement_audit).resolve(),
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                payload = torch.load(
                    Path(replacement_audit).resolve(), map_location="cpu"
                )
            candidates = payload["candidate_indices"].to(
                device=self.model._opacity.device, dtype=torch.long
            )
            self.model._opacity[candidates] = torch.logit(
                self.model._opacity.new_full((len(candidates), 1), 1e-8)
            )
        self.pipe = SimpleNamespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False,
            depth_ratio=0.0,
        )
        self.background = torch.ones(3, device="cuda")

    @torch.no_grad()
    def __call__(self, views) -> dict[int, np.ndarray]:
        from gaussian_renderer import render
        from scene.cameras import Camera

        output = {}
        scale = self.resolution_scale
        for offset, view in enumerate(views):
            width = max(32, int(round(view["width"] * scale)))
            height = max(32, int(round(view["height"] * scale)))
            image = torch.zeros(3, height, width)
            fx = float(view["fx"]) * width / float(view["width"])
            fy = float(view["fy"]) * height / float(view["height"])
            cx = float(view["cx"]) * width / float(view["width"])
            cy = float(view["cy"]) * height / float(view["height"])
            camera = Camera(
                colmap_id=int(view["image_id"]),
                R=np.asarray(view["rotation"]).T,
                T=np.asarray(view["translation"]),
                FoVx=2.0 * math.atan(width / (2.0 * fx)),
                FoVy=2.0 * math.atan(height / (2.0 * fy)),
                image=image,
                gt_alpha_mask=None,
                image_name=str(view["image_name"]),
                uid=offset,
                data_device="cpu",
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                zfar=200.0,
            )
            package = render(
                camera,
                self.model,
                self.pipe,
                self.background,
            )
            depth = package["rend_depth"][0].detach().cpu().numpy()
            alpha = package["rend_alpha"][0].detach().cpu().numpy()
            depth[alpha < 0.05] = np.inf
            output[int(view["image_id"])] = depth.astype(np.float32)
        return output


@torch.no_grad()
def render_protected_depth_maps(
    views,
    *,
    structural_ply: Path,
    replacement_audit: Path | None = None,
    sh_degree: int = 3,
    resolution_scale: float = 0.5,
) -> dict[int, np.ndarray]:
    """Render rigid z-buffers after suppressing audited legacy canopy."""
    renderer = ProtectedDepthRenderer(
        structural_ply=structural_ply,
        replacement_audit=replacement_audit,
        sh_degree=sh_degree,
        resolution_scale=resolution_scale,
    )
    return renderer(views)
