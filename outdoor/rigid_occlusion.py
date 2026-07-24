"""Protected-structure depth buffers for occlusion-aware foliage evidence."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch


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
    from gaussian_renderer import render
    from scene import GaussianModel
    from scene.cameras import Camera
    from types import SimpleNamespace

    model = GaussianModel(int(sh_degree))
    model.load_ply(str(Path(structural_ply).resolve()))
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
            device=model._opacity.device, dtype=torch.long
        )
        model._opacity[candidates] = torch.logit(
            model._opacity.new_full((len(candidates), 1), 1e-8)
        )
    pipe = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        depth_ratio=0.0,
    )
    background = torch.ones(3, device="cuda")
    output = {}
    scale = float(resolution_scale)
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
        package = render(camera, model, pipe, background)
        depth = package["rend_depth"][0].detach().cpu().numpy()
        alpha = package["rend_alpha"][0].detach().cpu().numpy()
        depth[alpha < 0.05] = np.inf
        output[int(view["image_id"])] = depth.astype(np.float32)
    return output
