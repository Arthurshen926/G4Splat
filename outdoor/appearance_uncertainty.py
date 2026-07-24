"""Low-capacity transient appearance and uncertainty for outdoor scenes.

Geometry is rendered canonically first.  A shared low-rank model may then
condition canopy/sky pixels for a known training image, while the canonical
render remains available for mapping and localization evaluation.
"""

from __future__ import annotations

import re

import torch
from torch import nn

from outdoor.directional_sky import CanonicalDirectionalSky, _eval_sh


APPEARANCE_VERSION = "outdoor_low_rank_appearance_uncertainty_v1"


def _natural_key(value: str):
    return [
        int(token) if token.isdigit() else token
        for token in re.split(r"(\d+)", str(value))
    ]


class OutdoorAppearanceUncertainty(nn.Module):
    """Shared decoder plus tiny per-image codes; no per-pixel free parameters."""

    def __init__(
        self,
        image_names,
        *,
        rank: int = 4,
        sky_degree: int = 2,
        maximum_rgb_residual: float = 0.08,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        names = [str(name) for name in image_names]
        if len(names) != len(set(names)):
            raise ValueError("Appearance image names must be unique")
        if rank <= 0:
            raise ValueError("Appearance rank must be positive")
        self.image_names = tuple(names)
        self.index = {name: offset for offset, name in enumerate(names)}
        self.rank = int(rank)
        self.sky_degree = int(sky_degree)
        self.maximum_rgb_residual = float(maximum_rgb_residual)
        device = torch.device(device)

        generator = torch.Generator(device=device)
        generator.manual_seed(9173)
        self.codes = nn.Parameter(
            torch.randn(
                len(names), self.rank, generator=generator, device=device
            )
            * 0.01
        )
        # Shared canopy affine decoder: three log-scales and three biases.
        self.canopy_decoder = nn.Parameter(
            torch.zeros(self.rank, 6, device=device)
        )
        # Directional low-rank sky basis B(d), in logit-residual space.
        self.sky_basis = nn.Parameter(
            torch.zeros(
                self.rank,
                3,
                (self.sky_degree + 1) ** 2,
                device=device,
            )
        )
        # Canopy and sky observation log-scales.  These describe aleatoric
        # mismatch, not geometry confidence, and are tightly bounded in loss.
        self.log_uncertainty = nn.Parameter(
            torch.full((len(names), 2), -3.0, device=device)
        )
        order = sorted(range(len(names)), key=lambda i: _natural_key(names[i]))
        self.register_buffer(
            "temporal_pairs",
            torch.tensor(
                list(zip(order[:-1], order[1:])),
                dtype=torch.long,
                device=device,
            )
            if len(order) > 1
            else torch.empty(0, 2, dtype=torch.long, device=device),
        )

    def image_index(self, image_name: str) -> int:
        try:
            return self.index[str(image_name)]
        except KeyError as error:
            raise KeyError(
                f"Unknown appearance image {image_name!r}; query views must use "
                "the canonical render, not an invented training code"
            ) from error

    def forward(self, rgb, camera, task):
        image_index = self.image_index(camera.image_name)
        code = self.codes[image_index]
        affine = code @ self.canopy_decoder
        scale = torch.exp(
            0.25 * torch.tanh(affine[:3])
        ).reshape(3, 1, 1)
        bias = (
            self.maximum_rgb_residual * torch.tanh(affine[3:])
        ).reshape(3, 1, 1)
        canopy = task["p_canopy"][None]
        conditioned = rgb + canopy * (rgb * (scale - 1.0) + bias)

        directions = CanonicalDirectionalSky.world_directions(
            camera, device=rgb.device, dtype=rgb.dtype
        )
        sky_residual = rgb.new_zeros((*directions.shape[:2], 3))
        for component in range(self.rank):
            coefficient = self.sky_basis[component].to(dtype=rgb.dtype)
            sky_residual = sky_residual + code[component] * _eval_sh(
                self.sky_degree, coefficient, directions
            )
        sky_residual = self.maximum_rgb_residual * torch.tanh(
            sky_residual
        ).permute(2, 0, 1)
        conditioned = conditioned + task["p_sky"][None] * sky_residual
        return conditioned.clamp(0.0, 1.0)

    def heteroscedastic_loss(self, prediction, target, task, epsilon=1e-3):
        image_index = self.image_index(task["image_name"])
        sigma = self.log_uncertainty[image_index].clamp(-4.5, -1.0).exp()
        robust = torch.sqrt((prediction - target).square() + epsilon**2).mean(0)
        canopy = task["p_canopy"] * (1.0 - task["p_transient"])
        sky = task["p_sky"] * (1.0 - task["p_transient"])
        rigid = task["p_rigid"]

        def region(mask, value):
            return (value * mask).sum() / mask.sum().clamp_min(1.0)

        # Positive robust uncertainty objective.  The additive floor prevents
        # vanishing sigma from amplifying sub-pixel residuals without bound;
        # log1p keeps increasing uncertainty costly while preserving readable
        # non-negative training traces.
        canopy_loss = region(
            canopy, robust / (sigma[0] + 0.05) + torch.log1p(sigma[0])
        )
        sky_loss = region(
            sky, robust / (sigma[1] + 0.05) + torch.log1p(sigma[1])
        )
        rigid_loss = region(rigid, robust)
        return canopy_loss + 0.5 * sky_loss + 0.25 * rigid_loss

    def regularization(self):
        value = self.codes.square().mean()
        value = value + 4.0 * self.codes.mean(dim=0).square().mean()
        value = value + self.canopy_decoder.square().mean()
        value = value + self.sky_basis.square().mean()
        value = value + 0.05 * self.log_uncertainty.square().mean()
        if self.temporal_pairs.numel():
            first, second = self.temporal_pairs.unbind(-1)
            value = value + 0.25 * (
                self.codes[first] - self.codes[second]
            ).square().mean()
        return value

    def capture(self):
        return {
            "version": APPEARANCE_VERSION,
            "image_names": self.image_names,
            "rank": self.rank,
            "sky_degree": self.sky_degree,
            "maximum_rgb_residual": self.maximum_rgb_residual,
            "state_dict": self.state_dict(),
        }

    def audit(self):
        return {
            "version": APPEARANCE_VERSION,
            "rank": self.rank,
            "image_count": len(self.image_names),
            "canonical_render_available": True,
            "query_fallback": "canonical_only",
            "canopy_model": "shared_low_rank_affine",
            "sky_model": "low_rank_directional_SH_temporal_residual",
            "uncertainty_regions": ["canopy", "sky"],
            "per_pixel_free_parameters": False,
        }
