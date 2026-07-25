"""Memory-efficient optimisation of an appended 2DGS residual suffix."""

from __future__ import annotations

import torch


class TrainableSurfelSuffix:
    """Expose a frozen GaussianModel prefix and leaf-parameter suffix.

    Optimising slices of a monolithic GaussianModel parameter makes Adam
    allocate moments for the million-row frozen prefix.  This adapter owns
    only suffix parameters while presenting the renderer's GaussianModel
    property contract.
    """

    def __init__(self, structural, start: int):
        self.structural = structural
        self.start = int(start)
        count = int(len(structural.get_xyz))
        if self.start < 0 or self.start > count:
            raise ValueError("suffix start must be within the structural model")
        self.max_sh_degree = int(structural.max_sh_degree)
        self.active_sh_degree = int(structural.active_sh_degree)
        self.use_mip_filter = bool(structural.use_mip_filter)
        self._xyz = torch.nn.Parameter(
            structural._xyz[self.start :].detach().clone()
        )
        self._features_dc = torch.nn.Parameter(
            structural._features_dc[self.start :].detach().clone()
        )
        self._features_rest = torch.nn.Parameter(
            structural._features_rest[self.start :].detach().clone()
        )
        self._opacity = torch.nn.Parameter(
            structural._opacity[self.start :].detach().clone()
        )
        self._scaling = torch.nn.Parameter(
            structural._scaling[self.start :].detach().clone()
        )
        self._rotation = torch.nn.Parameter(
            structural._rotation[self.start :].detach().clone()
        )
        self.initial_xyz = self._xyz.detach().clone()
        self.initial_scaling = self._scaling.detach().clone()

    def _join(self, original: torch.Tensor, suffix: torch.Tensor) -> torch.Tensor:
        if self.start == 0:
            return suffix
        return torch.cat([original[: self.start].detach(), suffix], dim=0)

    @property
    def get_xyz(self):
        return self._join(self.structural._xyz, self._xyz)

    @property
    def get_features(self):
        dc = self._join(self.structural._features_dc, self._features_dc)
        rest = self._join(
            self.structural._features_rest, self._features_rest
        )
        return torch.cat([dc, rest], dim=1)

    @property
    def get_rotation(self):
        rotation = torch.nn.functional.normalize(self._rotation, dim=-1)
        return self._join(self.structural.get_rotation, rotation)

    @property
    def get_scaling(self):
        scaling = torch.exp(self._scaling)
        if self.use_mip_filter:
            mip = self.structural.mip_filter[self.start :].detach()
            scaling = torch.sqrt(scaling.square() + mip.square())
        return self._join(self.structural.get_scaling, scaling)

    @property
    def get_opacity(self):
        opacity = torch.sigmoid(self._opacity)
        if self.use_mip_filter:
            raw_scale = torch.exp(self._scaling)
            mip = self.structural.mip_filter[self.start :].detach()
            filtered_scale = torch.sqrt(raw_scale.square() + mip.square())
            opacity = opacity * torch.sqrt(
                raw_scale.square().prod(dim=1)
                / filtered_scale.square().prod(dim=1).clamp_min(1e-12)
            )[:, None]
        return self._join(self.structural.get_opacity, opacity)

    def parameter_groups(
        self,
        *,
        position_lr: float,
        feature_lr: float,
        opacity_lr: float,
        scale_lr: float,
        rotation_lr: float,
    ) -> list[dict]:
        return [
            {"params": [self._xyz], "lr": position_lr, "name": "rigid_xyz"},
            {
                "params": [self._features_dc],
                "lr": feature_lr,
                "name": "rigid_dc",
            },
            {
                "params": [self._features_rest],
                "lr": feature_lr / 20.0,
                "name": "rigid_sh",
            },
            {
                "params": [self._opacity],
                "lr": opacity_lr,
                "name": "rigid_opacity",
            },
            {
                "params": [self._scaling],
                "lr": scale_lr,
                "name": "rigid_scale",
            },
            {
                "params": [self._rotation],
                "lr": rotation_lr,
                "name": "rigid_rotation",
            },
        ]

    @torch.no_grad()
    def project(
        self,
        *,
        maximum_opacity: float,
        maximum_scale: float,
        maximum_position_delta: float,
        diffuse_only: bool = True,
    ) -> None:
        if 0.0 < maximum_opacity < 1.0:
            ceiling = torch.logit(
                self._opacity.new_tensor(float(maximum_opacity))
            )
            self._opacity.clamp_(max=ceiling)
        if maximum_scale > 0:
            self._scaling.clamp_(max=float(torch.log(
                self._scaling.new_tensor(maximum_scale)
            )))
        if maximum_position_delta > 0:
            delta = self._xyz - self.initial_xyz
            norm = torch.linalg.vector_norm(
                delta, dim=-1, keepdim=True
            ).clamp_min(1e-12)
            multiplier = torch.clamp(
                float(maximum_position_delta) / norm, max=1.0
            )
            self._xyz.copy_(self.initial_xyz + delta * multiplier)
        self._rotation.copy_(
            torch.nn.functional.normalize(self._rotation, dim=-1)
        )
        if diffuse_only:
            self._features_rest.zero_()

    @torch.no_grad()
    def write_back(self) -> None:
        suffix = slice(self.start, None)
        self.structural._xyz[suffix].copy_(self._xyz)
        self.structural._features_dc[suffix].copy_(self._features_dc)
        self.structural._features_rest[suffix].copy_(self._features_rest)
        self.structural._opacity[suffix].copy_(self._opacity)
        self.structural._scaling[suffix].copy_(self._scaling)
        self.structural._rotation[suffix].copy_(self._rotation)
