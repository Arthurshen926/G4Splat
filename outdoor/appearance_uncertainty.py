"""Low-capacity transient appearance and uncertainty for outdoor scenes.

Geometry is rendered canonically first.  A shared low-rank model may then
condition canopy/sky pixels for a known training image, while the canonical
render remains available for mapping and localization evaluation.
"""

from __future__ import annotations

import math
import re

import torch
from torch import nn

from outdoor.directional_sky import CanonicalDirectionalSky, _eval_sh
from outdoor.foliage_view_graph import sequence_id


APPEARANCE_VERSION = (
    "outdoor_stripe_free_spatial_sequence_appearance_uncertainty_v5"
)
GLOBAL_APPEARANCE_VERSION = (
    "outdoor_global_sequence_appearance_uncertainty_v4"
)
SPATIAL_APPEARANCE_VERSION = (
    "outdoor_spatial_sequence_appearance_uncertainty_v3"
)
LEGACY_APPEARANCE_VERSION = "outdoor_spatial_sequence_appearance_uncertainty_v2"


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
        spatial_grid_size: int = 24,
        temporal_code_norm: float = 0.25,
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
        self.spatial_grid_size = int(spatial_grid_size)
        self.temporal_code_norm = float(temporal_code_norm)
        if self.temporal_code_norm <= 0:
            raise ValueError("temporal_code_norm must be positive")
        device = torch.device(device)

        generator = torch.Generator(device=device)
        generator.manual_seed(9173)
        self.codes = nn.Parameter(
            torch.randn(
                len(names), self.rank, generator=generator, device=device
            )
            * 0.01
        )
        # Geometry, foliage appearance, sky and uncertainty do not share one
        # latent direction.  Sharing previously let a sky/exposure update
        # move leaf geometry and encouraged low-frequency compromise.
        self.foliage_codes = nn.Parameter(self.codes.detach().clone())
        self.sky_codes = nn.Parameter(self.codes.detach().clone())
        self.uncertainty_codes = nn.Parameter(self.codes.detach().clone())
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
        # Compatibility tensors for v2/v3 checkpoints.  They are deliberately
        # frozen and ignored by v4: a fixed camera-plane lattice imprints its
        # cell frequency on foliage RGB and uncertainty (24 columns at the
        # Cambridge training resolution produced the measured 26.7 px bands).
        # Keeping the tensors lets an old checkpoint load losslessly while the
        # runtime image-formation policy is explicit and artifact-free.
        self.local_canopy_basis = nn.Parameter(
            torch.zeros(
                self.rank,
                3,
                self.spatial_grid_size,
                self.spatial_grid_size,
                device=device,
            ),
            requires_grad=False,
        )
        self.spatial_uncertainty_basis = nn.Parameter(
            torch.zeros(
                self.rank,
                2,
                self.spatial_grid_size,
                self.spatial_grid_size,
                device=device,
            ),
            requires_grad=False,
        )
        self.uncertainty_base = nn.Parameter(
            torch.full((2,), -3.0, device=device)
        )
        # Training-only heteroscedasticity needs spatial support even for a
        # static reconstruction: foliage captured in different traversals is
        # not spatially consistent.  The former 24x24 camera-plane lattice
        # visibly imprinted its column frequency on the render.  A bounded
        # low-order Legendre field has no cells or periodic frequencies and
        # is used only to weight the canonical loss; it never adds RGB to the
        # deployed model.  Register this after all v4 trainable parameters so
        # a v4 Adam group can be migrated by appending exactly one parameter.
        self.uncertainty_decoder = nn.Parameter(
            torch.zeros(self.rank, 12, device=device)
        )
        self._coordinate_basis_cache: dict[
            tuple[int, int, str, int | None, torch.dtype], torch.Tensor
        ] = {}
        grouped: dict[str, list[int]] = {}
        for index, name in enumerate(names):
            grouped.setdefault(sequence_id(name), []).append(index)
        pairs = []
        for indices in grouped.values():
            order = sorted(indices, key=lambda i: _natural_key(names[i]))
            pairs.extend(zip(order[:-1], order[1:]))
        self.register_buffer(
            "temporal_pairs",
            torch.tensor(
                pairs,
                dtype=torch.long,
                device=device,
            )
            if pairs
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

    def _normalized_code(
        self, values: torch.Tensor, image_name: str
    ) -> torch.Tensor:
        code = values[self.image_index(image_name)]
        # Low-rank deformation has a scale ambiguity between the per-frame
        # code and the per-Gaussian basis.  Raw L2 regularization previously
        # collapsed code norms to ~1e-2, leaving sub-micrometre "dynamic"
        # motion.  Optimize direction with a fixed effective norm while the
        # raw codes still receive within-sequence smoothness regularization.
        denominator = code.detach().norm().clamp_min(1e-3)
        return code * (self.temporal_code_norm / denominator)

    def temporal_code(self, image_name: str) -> torch.Tensor:
        """Geometry/deformation code consumed by dynamic 3D leaves."""
        return self._normalized_code(self.codes, image_name)

    def foliage_code(self, image_name: str) -> torch.Tensor:
        return self._normalized_code(self.foliage_codes, image_name)

    def sky_code(self, image_name: str) -> torch.Tensor:
        return self._normalized_code(self.sky_codes, image_name)

    def uncertainty_code(self, image_name: str) -> torch.Tensor:
        return self._normalized_code(self.uncertainty_codes, image_name)

    def _spatial_fields(
        self,
        image_name: str,
        shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = tuple(map(int, shape))
        local_rgb = self.uncertainty_base.new_zeros((3, *size))
        height, width = size
        key = (
            height,
            width,
            self.uncertainty_base.device.type,
            self.uncertainty_base.device.index,
            self.uncertainty_base.dtype,
        )
        basis = self._coordinate_basis_cache.get(key)
        if basis is None:
            basis = self._continuous_coordinate_basis(height, width)
            self._coordinate_basis_cache[key] = basis
        coefficients = (
            self.uncertainty_code(image_name) @ self.uncertainty_decoder
        ).reshape(2, 6)
        spatial_logit = torch.einsum("rc,chw->rhw", coefficients, basis)
        # A smooth bounded offset prevents a difficult image from escaping
        # the proper scale likelihood by sending sigma to infinity.
        log_sigma = self.uncertainty_base[:, None, None] + 1.5 * torch.tanh(
            spatial_logit / math.sqrt(6.0)
        )
        return (
            self.maximum_rgb_residual * torch.tanh(local_rgb),
            # Smooth bounded parameterization preserves a recovery gradient
            # at both limits. A hard clamp stranded raw parameters outside
            # the valid range while displaying an apparently healthy sigma.
            math.exp(-4.5)
            + (math.exp(-1.5) - math.exp(-4.5))
            * torch.sigmoid(log_sigma),
        )

    def _continuous_coordinate_basis(
        self, height: int, width: int
    ) -> torch.Tensor:
        """Build one immutable non-periodic basis per render resolution."""

        y = torch.linspace(
            -1.0,
            1.0,
            height,
            device=self.uncertainty_base.device,
            dtype=self.uncertainty_base.dtype,
        )
        x = torch.linspace(
            -1.0,
            1.0,
            width,
            device=self.uncertainty_base.device,
            dtype=self.uncertainty_base.dtype,
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack(
            [
                torch.ones_like(xx),
                xx,
                yy,
                xx * yy,
                0.5 * (3.0 * xx.square() - 1.0),
                0.5 * (3.0 * yy.square() - 1.0),
            ],
            dim=0,
        )

    def spatial_uncertainty(
        self,
        image_name: str,
        shape: tuple[int, int],
    ) -> torch.Tensor:
        """Return learned canopy/sky sigma maps for robust static supervision."""
        return self._spatial_fields(image_name, shape)[1]

    def uncertainty_regularization(
        self,
        image_name: str,
        shape: tuple[int, int],
        *,
        sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prior and total variation for the late uncertainty curriculum."""
        if sigma is None:
            sigma = self.spatial_uncertainty(image_name, shape)
        log_sigma = sigma.clamp_min(1e-6).log()
        prior = (log_sigma + 3.0).square().mean()
        horizontal = (
            log_sigma[:, :, 1:] - log_sigma[:, :, :-1]
        ).abs().mean()
        vertical = (
            log_sigma[:, 1:, :] - log_sigma[:, :-1, :]
        ).abs().mean()
        return prior + 0.25 * (horizontal + vertical)

    def forward(self, rgb, camera, task):
        image_index = self.image_index(camera.image_name)
        foliage_code = self.foliage_code(camera.image_name)
        sky_code = self.sky_code(camera.image_name)
        affine = foliage_code @ self.canopy_decoder
        scale = torch.exp(
            0.25 * torch.tanh(affine[:3])
        ).reshape(3, 1, 1)
        bias = (
            self.maximum_rgb_residual * torch.tanh(affine[3:])
        ).reshape(3, 1, 1)
        canopy = task["p_canopy"][None]
        conditioned = rgb + canopy * (rgb * (scale - 1.0) + bias)
        local_canopy, _ = self._spatial_fields(
            camera.image_name, (rgb.shape[-2], rgb.shape[-1])
        )
        conditioned = conditioned + canopy * local_canopy

        directions = CanonicalDirectionalSky.world_directions(
            camera, device=rgb.device, dtype=rgb.dtype
        )
        sky_residual = rgb.new_zeros((*directions.shape[:2], 3))
        for component in range(self.rank):
            coefficient = self.sky_basis[component].to(dtype=rgb.dtype)
            sky_residual = sky_residual + sky_code[component] * _eval_sh(
                self.sky_degree, coefficient, directions
            )
        sky_residual = self.maximum_rgb_residual * torch.tanh(
            sky_residual
        ).permute(2, 0, 1)
        conditioned = conditioned + task["p_sky"][None] * sky_residual
        return conditioned.clamp(0.0, 1.0)

    def heteroscedastic_loss(
        self,
        prediction,
        target,
        task,
        *,
        image_name: str | None = None,
        sigma: torch.Tensor | None = None,
        epsilon=1e-3,
    ):
        if image_name is None:
            image_name = task.get("image_name")
        if image_name is None:
            raise ValueError(
                "Spatial uncertainty requires the calibrated training image name"
            )
        if sigma is None:
            sigma = self.spatial_uncertainty(
                image_name,
                (prediction.shape[-2], prediction.shape[-1]),
            )
        robust = torch.sqrt((prediction - target).square() + epsilon**2).mean(0)
        canopy = task["p_canopy"] * (1.0 - task["p_transient"])
        sky = task["p_sky"] * (1.0 - task["p_transient"])
        rigid = task["p_rigid"]

        def region(mask, value):
            return (value * mask).sum() / mask.sum().clamp_min(1.0)

        # Proper robust scale likelihood. The direct conditioned photo loss is
        # optimized separately, so sigma cannot weaken geometry or topology.
        effective_sigma = sigma + 0.02
        canopy_loss = region(
            canopy,
            0.5 * (robust / effective_sigma[0]).square()
            + effective_sigma[0].log(),
        )
        sky_loss = region(
            sky,
            0.5 * (robust / effective_sigma[1]).square()
            + effective_sigma[1].log(),
        )
        rigid_loss = region(rigid, robust)
        return canopy_loss + 0.5 * sky_loss + 0.25 * rigid_loss

    def regularization(self):
        code_groups = (
            self.codes,
            self.foliage_codes,
            self.sky_codes,
            self.uncertainty_codes,
        )
        value = sum(group.square().mean() for group in code_groups)
        value = value + 4.0 * sum(
            group.mean(dim=0).square().mean() for group in code_groups
        )
        value = value + self.canopy_decoder.square().mean()
        value = value + self.sky_basis.square().mean()
        value = value + self.uncertainty_decoder.square().mean()
        value = value + 0.05 * (
            self.uncertainty_base + 3.0
        ).square().mean()
        if self.temporal_pairs.numel():
            first, second = self.temporal_pairs.unbind(-1)
            value = value + 0.25 * sum(
                (group[first] - group[second]).square().mean()
                for group in code_groups
            )
        return value

    @torch.no_grad()
    def restore(self, payload):
        """Restore v3 or migrate the former shared-code v2 checkpoint."""
        version = payload.get("version", LEGACY_APPEARANCE_VERSION)
        if version not in {
            APPEARANCE_VERSION,
            GLOBAL_APPEARANCE_VERSION,
            SPATIAL_APPEARANCE_VERSION,
            LEGACY_APPEARANCE_VERSION,
        }:
            raise RuntimeError(f"Unsupported appearance state {version!r}")
        state = dict(payload["state_dict"])
        shared = state.get("codes")
        if shared is not None:
            for name in (
                "foliage_codes",
                "sky_codes",
                "uncertainty_codes",
            ):
                state.setdefault(name, shared.clone())
        missing, unexpected = self.load_state_dict(state, strict=False)
        if (
            version != APPEARANCE_VERSION
            and missing == ["uncertainty_decoder"]
        ):
            missing = []
        if missing or unexpected:
            raise RuntimeError(
                "Appearance state mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )

    def capture(self):
        return {
            "version": APPEARANCE_VERSION,
            "image_names": self.image_names,
            "rank": self.rank,
            "sky_degree": self.sky_degree,
            "maximum_rgb_residual": self.maximum_rgb_residual,
            "spatial_grid_size": self.spatial_grid_size,
            "temporal_code_norm": self.temporal_code_norm,
            "state_dict": self.state_dict(),
        }

    def audit(self):
        return {
            "version": APPEARANCE_VERSION,
            "rank": self.rank,
            "image_count": len(self.image_names),
            "canonical_render_available": True,
            "query_fallback": "canonical_only",
            "canopy_model": "shared_low_rank_global_affine",
            "sky_model": "low_rank_directional_SH_temporal_residual",
            "spatial_canopy_rgb_residual": "disabled",
            "appearance_grid_policy": (
                "no_camera_plane_grid_stripe_free_legendre_uncertainty"
            ),
            "uncertainty_regions": ["spatial_canopy", "spatial_sky"],
            "uncertainty_grid": "continuous_degree2_legendre_no_cells",
            "uncertainty_usage": "training_loss_weight_only_no_rgb_residual",
            "temporal_regularization": "within_sequence_only",
            "temporal_code_parameterization": "fixed_norm_direction",
            "temporal_code_branches": [
                "geometry",
                "foliage_appearance",
                "sky",
                "spatial_uncertainty",
            ],
            "temporal_code_norm": self.temporal_code_norm,
            "per_pixel_free_parameters": False,
        }

    def uncertainty_parameters(self) -> tuple[nn.Parameter, ...]:
        """Parameters owned only by the training-time robust scale model."""

        return (
            self.uncertainty_codes,
            self.uncertainty_base,
            self.uncertainty_decoder,
        )
