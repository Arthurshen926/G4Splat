"""Small public loading/rendering API for the authoritative mixed Teacher."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SURFEL_ROOT = REPO_ROOT / "2d-gaussian-splatting"
sys.path[:0] = [str(REPO_ROOT), str(SURFEL_ROOT)]

from outdoor.appearance_uncertainty import OutdoorAppearanceUncertainty  # noqa: E402
from outdoor.directional_sky import (  # noqa: E402
    CanonicalDirectionalSky,
    composite_white_background,
)
from outdoor.hybrid_gaussian_renderer import (  # noqa: E402
    VolumetricFoliageModel,
    dynamic_visibility_gate,
    render_hybrid,
)
from scene import GaussianModel  # noqa: E402


SUPPORTED_TEACHER_PROTOCOLS = {
    "cambridge_native_hybrid_teacher_v2",
    "cambridge_native_hybrid_teacher_v3_causal_repair",
    "cambridge_native_hybrid_teacher_v4_causal_repair",
    "unified_outdoor_mixed_teacher_v1",
}


@dataclass
class HybridTeacher:
    surface: GaussianModel
    foliage: VolumetricFoliageModel
    sky: CanonicalDirectionalSky
    appearance: OutdoorAppearanceUncertainty
    state: dict

    def render(
        self,
        camera,
        *,
        task: dict[str, torch.Tensor] | None = None,
        conditioned: bool = False,
        background: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Render canonical geometry or a known sequence/time-conditioned view."""
        if background is None:
            background = torch.ones(3, device=self.surface.get_xyz.device)
        if conditioned and task is None:
            raise ValueError("Conditioned rendering requires semantic task fields")
        temporal_code = (
            self.appearance.temporal_code(camera.image_name)
            if conditioned
            else None
        )
        volume_gate = None
        if conditioned:
            contract = self.state.get("camera_ownership_contract")
            if contract is not None:
                volume_gate = dynamic_visibility_gate(
                    self.foliage,
                    int(camera.colmap_id),
                    contract["sequence_lookup"].to(
                        device=self.foliage.xyz.device
                    ),
                    contract["frame_lookup"].to(
                        device=self.foliage.xyz.device
                    ),
                )
            else:
                volume_gate = dynamic_visibility_gate(
                    self.foliage, int(camera.colmap_id), None, None
                )
        package = render_hybrid(
            camera,
            self.surface,
            self.foliage,
            background=background,
            temporal_code=temporal_code,
            include_dynamic=conditioned,
            volume_gate=volume_gate,
        )
        rgb = composite_white_background(
            package.render, package.alpha, self.sky(camera)
        )
        if conditioned:
            rgb = self.appearance(rgb, camera, task)
        return {
            "rgb": rgb.clamp(0, 1),
            "alpha": package.alpha,
            "depth": package.depth,
            "surface_depth": package.surface_depth,
            "volume_depth": package.volume_depth,
            "normal_world": package.normal_world,
            "surface_alpha": package.surface_alpha,
            "volume_alpha": package.volume_alpha,
        }


def _restore_surface(model: GaussianModel, capture: tuple) -> None:
    (
        active,
        xyz,
        features_dc,
        features_rest,
        scaling,
        rotation,
        opacity,
    ) = capture[:7]
    model.active_sh_degree = int(active)
    model._xyz = torch.nn.Parameter(xyz.detach().cuda(), requires_grad=False)
    model._features_dc = torch.nn.Parameter(
        features_dc.detach().cuda(), requires_grad=False
    )
    model._features_rest = torch.nn.Parameter(
        features_rest.detach().cuda(), requires_grad=False
    )
    model._scaling = torch.nn.Parameter(
        scaling.detach().cuda(), requires_grad=False
    )
    model._rotation = torch.nn.Parameter(
        rotation.detach().cuda(), requires_grad=False
    )
    model._opacity = torch.nn.Parameter(
        opacity.detach().cuda(), requires_grad=False
    )
    model._restore_point_metadata(capture[12] if len(capture) >= 13 else None)


def load_hybrid_teacher(
    state_path: Path,
    *,
    sh_degree: int,
) -> HybridTeacher:
    """Load a Teacher state. No student conversion or COLMAP geometry is used."""
    try:
        state = torch.load(
            Path(state_path), map_location="cpu", weights_only=False
        )
    except TypeError:
        state = torch.load(Path(state_path), map_location="cpu")
    protocol = str(state.get("protocol", ""))
    if protocol not in SUPPORTED_TEACHER_PROTOCOLS:
        raise RuntimeError(f"Unsupported hybrid Teacher protocol {protocol!r}")
    surface = GaussianModel(sh_degree)
    _restore_surface(surface, state["surface"])
    foliage = VolumetricFoliageModel(
        sh_degree,
        dynamic_rank=int(state["foliage"].get("dynamic_rank", 4)),
    ).cuda()
    foliage.restore(state["foliage"])
    sky = CanonicalDirectionalSky(
        degree=int(state.get("sky_degree", 2))
    ).cuda()
    sky.load_state_dict(state["sky"])
    appearance_payload = state["appearance"]
    appearance = OutdoorAppearanceUncertainty(
        appearance_payload["image_names"],
        rank=int(appearance_payload["rank"]),
        sky_degree=int(appearance_payload["sky_degree"]),
        maximum_rgb_residual=float(
            appearance_payload["maximum_rgb_residual"]
        ),
        spatial_grid_size=int(appearance_payload["spatial_grid_size"]),
        temporal_code_norm=float(
            appearance_payload.get("temporal_code_norm", 0.25)
        ),
        device="cuda",
    )
    appearance.restore(appearance_payload)
    surface.active_sh_degree = min(surface.active_sh_degree, sh_degree)
    for parameter in foliage.parameters():
        parameter.requires_grad_(False)
    for parameter in sky.parameters():
        parameter.requires_grad_(False)
    for parameter in appearance.parameters():
        parameter.requires_grad_(False)
    return HybridTeacher(surface, foliage, sky, appearance, state)
