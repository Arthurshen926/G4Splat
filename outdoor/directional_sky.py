"""Small camera-independent directional sky field for outdoor rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

SKY_MODEL_VERSION = "canonical_directional_sky_sh_v1"
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)


def _eval_sh(degree: int, coefficients: torch.Tensor, directions: torch.Tensor):
    result = C0 * coefficients[..., 0] + directions[..., :1] * 0.0
    if degree > 0:
        x, y, z = (
            directions[..., 0:1],
            directions[..., 1:2],
            directions[..., 2:3],
        )
        result = (
            result
            - C1 * y * coefficients[..., 1]
            + C1 * z * coefficients[..., 2]
            - C1 * x * coefficients[..., 3]
        )
        if degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * coefficients[..., 4]
                + C2[1] * yz * coefficients[..., 5]
                + C2[2] * (2.0 * zz - xx - yy) * coefficients[..., 6]
                + C2[3] * xz * coefficients[..., 7]
                + C2[4] * (xx - yy) * coefficients[..., 8]
            )
    return result


class CanonicalDirectionalSky(nn.Module):
    """A low-capacity world-direction SH sky, evaluated behind raster alpha."""

    def __init__(self, degree: int = 2, initial_rgb: float = 0.95):
        super().__init__()
        if degree < 0 or degree > 2:
            raise ValueError("Sky SH degree must be in [0, 2]")
        if not 0.0 < float(initial_rgb) < 1.0:
            raise ValueError("initial_rgb must be in (0, 1)")
        self.degree = int(degree)
        coefficients = torch.zeros(3, (self.degree + 1) ** 2)
        initial_logit = float(np.log(initial_rgb / (1.0 - initial_rgb)))
        coefficients[:, 0] = initial_logit / C0
        self.logit_sh = nn.Parameter(coefficients)

    @staticmethod
    def world_directions(camera, *, device, dtype) -> torch.Tensor:
        height = int(camera.image_height)
        width = int(camera.image_width)
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        camera_rays = torch.stack(
            [
                (x - float(camera.cx)) / float(camera.focal_x),
                (y - float(camera.cy)) / float(camera.focal_y),
                torch.ones_like(x),
            ],
            dim=-1,
        )
        rotation = torch.as_tensor(camera.R, device=device, dtype=dtype)
        world_rays = camera_rays @ rotation.transpose(0, 1)
        return torch.nn.functional.normalize(world_rays, dim=-1)

    def forward(self, camera) -> torch.Tensor:
        directions = self.world_directions(
            camera, device=self.logit_sh.device, dtype=self.logit_sh.dtype
        )
        logits = _eval_sh(self.degree, self.logit_sh, directions)
        return torch.sigmoid(logits).permute(2, 0, 1)

    def audit(self) -> dict:
        return {
            "version": SKY_MODEL_VERSION,
            "degree": self.degree,
            "coefficient_count": int(self.logit_sh.numel()),
            "coordinate_frame": "COLMAP_world_direction",
            "compositing": "behind_gaussian_raster_alpha",
            "camera_specific_parameters": False,
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": SKY_MODEL_VERSION,
                "degree": self.degree,
                "state_dict": self.state_dict(),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path, *, device) -> "CanonicalDirectionalSky":
        try:
            payload = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=device)
        if payload.get("version") != SKY_MODEL_VERSION:
            raise RuntimeError(f"Unsupported directional sky checkpoint: {path}")
        model = cls(degree=int(payload["degree"])).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        return model


def composite_white_background(
    gaussian_white_rgb: torch.Tensor,
    raster_alpha: torch.Tensor,
    sky_rgb: torch.Tensor,
) -> torch.Tensor:
    """Replace a white renderer background by a directional sky."""
    if gaussian_white_rgb.shape != sky_rgb.shape:
        raise ValueError(
            f"RGB/sky shapes differ: {gaussian_white_rgb.shape} vs {sky_rgb.shape}"
        )
    if raster_alpha.ndim == 2:
        raster_alpha = raster_alpha[None]
    if raster_alpha.shape[0] != 1 or raster_alpha.shape[-2:] != sky_rgb.shape[-2:]:
        raise ValueError(
            f"Unexpected raster alpha shape {raster_alpha.shape} for {sky_rgb.shape}"
        )
    return gaussian_white_rgb + (1.0 - raster_alpha) * (sky_rgb - 1.0)
