"""Canonical standard-3DGS initialization, PLY export and schema audit."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


STANDARD_3DGS_VERSION = "canonical-standard-3dgs-v1"
C0 = 0.28209479177387814
STUDENT_ROLE_RIGID = 10
STUDENT_ROLE_SKELETON = 11
STUDENT_ROLE_CROWN = 12
STUDENT_ROLE_DYNAMIC_BAKED = 13
STUDENT_ROLE_SKY = 14


def _rotation_from_z(directions: np.ndarray) -> np.ndarray:
    directions = np.asarray(directions, dtype=np.float64)
    directions /= np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
    )
    z = np.asarray([0.0, 0.0, 1.0])
    axes = np.cross(np.broadcast_to(z, directions.shape), directions)
    dots = np.clip(directions[:, 2], -1.0, 1.0)
    norms = np.linalg.norm(axes, axis=1)
    rotations = np.zeros((len(directions), 3), dtype=np.float64)
    valid = norms > 1e-8
    rotations[valid] = (
        axes[valid]
        / norms[valid, None]
        * np.arccos(dots[valid])[:, None]
    )
    opposite = (~valid) & (dots < 0)
    rotations[opposite, 0] = math.pi
    xyzw = Rotation.from_rotvec(rotations).as_quat()
    return np.column_stack(
        [xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]]
    ).astype(np.float32)


def _surface_subdivisions(
    scales: torch.Tensor,
    *,
    maximum_gaussians: int,
) -> torch.Tensor:
    count = len(scales)
    subdivisions = torch.ones(
        count, dtype=torch.int64, device=scales.device
    )
    if maximum_gaussians <= count:
        return subdivisions
    footprint = scales.prod(dim=1)
    median = footprint.median().clamp_min(1e-12)
    desired = torch.where(
        footprint >= 9.0 * median,
        torch.full_like(subdivisions, 3),
        torch.where(
            footprint >= 3.0 * median,
            torch.full_like(subdivisions, 2),
            subdivisions,
        ),
    )
    budget = int(maximum_gaussians - count)
    # Allocate extra children to the largest footprints first.  Every surfel
    # remains represented even under a tight export budget.
    for value in (3, 2):
        candidates = torch.nonzero(
            desired >= value, as_tuple=False
        ).flatten()
        extra = value * value - 1
        maximum = min(len(candidates), budget // extra)
        if maximum <= 0:
            continue
        chosen = candidates[
            torch.topk(footprint[candidates], maximum).indices
        ]
        subdivisions[chosen] = value
        desired[chosen] = 1
        budget -= maximum * extra
    return subdivisions


@torch.no_grad()
def canonical_student_seed(
    surface,
    foliage,
    sky,
    camera_centers: torch.Tensor,
    *,
    maximum_surface_gaussians: int = 1_500_000,
    sky_gaussians: int = 8192,
    temporal_codes: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Sample thin 3D volumes over teacher surfels, then append foliage/sky."""
    from utils.general_utils import build_rotation

    device = surface.get_xyz.device
    dtype = surface.get_xyz.dtype
    scales_2d = surface.get_scaling.detach()
    subdivisions = _surface_subdivisions(
        scales_2d,
        maximum_gaussians=maximum_surface_gaussians,
    )
    rotation = build_rotation(surface.get_rotation.detach())
    xyz_values = []
    scale_values = []
    quaternion_values = []
    opacity_values = []
    feature_values = []
    role_values = []
    confidence = surface._geometry_confidence.detach().clamp(0, 1)
    for value in (1, 2, 3):
        indices = torch.nonzero(
            subdivisions == value, as_tuple=False
        ).flatten()
        if not len(indices):
            continue
        if value == 1:
            grid = torch.zeros(1, 2, device=device, dtype=dtype)
        else:
            axis = torch.linspace(
                -(1.0 - 1.0 / value),
                1.0 - 1.0 / value,
                value,
                device=device,
                dtype=dtype,
            )
            yy, xx = torch.meshgrid(axis, axis, indexing="ij")
            grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        child_count = value * value
        parent_scale = scales_2d[indices]
        local = torch.zeros(
            len(indices), child_count, 3, device=device, dtype=dtype
        )
        local[..., :2] = (
            grid[None] * parent_scale[:, None]
        )
        offset = torch.einsum(
            "nij,nkj->nki", rotation[indices], local
        )
        xyz_values.append(
            (
                surface.get_xyz.detach()[indices, None] + offset
            ).reshape(-1, 3)
        )
        role_values.append(
            torch.full(
                (len(indices) * child_count,),
                STUDENT_ROLE_RIGID,
                dtype=torch.int8,
                device=device,
            )
        )
        tangent = (
            parent_scale[:, None, :].expand(-1, child_count, -1)
            / float(value)
        )
        thickness = (
            parent_scale.min(dim=1).values
            * (0.04 + 0.12 * (1.0 - confidence[indices]))
        ).clamp(0.0015, 0.05)
        thickness = thickness[:, None, None].expand(
            -1, child_count, 1
        )
        scale_values.append(
            torch.cat([tangent, thickness], dim=2).reshape(-1, 3)
        )
        quaternion_values.append(
            surface.get_rotation.detach()[indices, None]
            .expand(-1, child_count, -1)
            .reshape(-1, 4)
        )
        alpha = surface.get_opacity.detach()[indices]
        child_alpha = 1.0 - (
            1.0 - alpha
        ).clamp_min(1e-6).pow(1.0 / child_count)
        opacity_values.append(
            child_alpha[:, None]
            .expand(-1, child_count, -1)
            .reshape(-1, 1)
        )
        feature_values.append(
            surface.get_features.detach()[indices, None]
            .expand(-1, child_count, -1, -1)
            .reshape(
                -1,
                surface.get_features.shape[1],
                3,
            )
        )

    canonical = ~foliage.dynamic_leaf_mask
    xyz_values.append(foliage.xyz.detach()[canonical])
    scale_values.append(foliage.scales.detach()[canonical])
    quaternion_values.append(
        foliage.normalized_quaternions.detach()[canonical]
    )
    opacity_values.append(
        foliage.opacities.detach()[canonical, None]
    )
    feature_values.append(foliage.features.detach()[canonical])
    canonical_roles = torch.full(
        (int(canonical.sum()),),
        STUDENT_ROLE_CROWN,
        dtype=torch.int8,
        device=device,
    )
    canonical_roles[
        foliage.static_skeleton_mask[canonical]
    ] = STUDENT_ROLE_SKELETON
    role_values.append(canonical_roles)

    dynamic = foliage.dynamic_leaf_mask
    baked_dynamic_count = 0
    temporal_variance_mean = 0.0
    if bool(dynamic.any()) and temporal_codes is not None and len(temporal_codes):
        codes = temporal_codes.to(device=device, dtype=dtype)
        median_code = codes.median(dim=0).values
        centered = codes - codes.mean(dim=0, keepdim=True)
        code_covariance = centered.T @ centered / max(len(codes) - 1, 1)
        basis = foliage.deformation_basis.detach()[dynamic]
        displacement_variance = torch.einsum(
            "nrc,rs,nsc->n", basis, code_covariance, basis
        ).clamp_min(0)
        dynamic_scale = foliage.scales.detach()[dynamic].square().mean(1)
        stability = torch.exp(
            -displacement_variance / dynamic_scale.clamp_min(1e-6)
        ).clamp(0.10, 1.0)
        conditioned_xyz, conditioned_features, conditioned_opacity = (
            foliage.conditioned_state(
                median_code, include_dynamic=True
            )
        )
        xyz_values.append(conditioned_xyz.detach()[dynamic])
        scale_values.append(
            foliage.scales.detach()[dynamic] * 0.75
        )
        quaternion_values.append(
            foliage.normalized_quaternions.detach()[dynamic]
        )
        opacity_values.append(
            (
                conditioned_opacity.detach()[dynamic]
                * stability
                * 0.22
            )[:, None]
        )
        feature_values.append(conditioned_features.detach()[dynamic])
        role_values.append(
            torch.full(
                (int(dynamic.sum()),),
                STUDENT_ROLE_DYNAMIC_BAKED,
                dtype=torch.int8,
                device=device,
            )
        )
        baked_dynamic_count = int(dynamic.sum())
        temporal_variance_mean = float(displacement_variance.mean())

    centers = camera_centers.to(device=device, dtype=dtype)
    scene_center = torch.median(centers, dim=0).values
    support = torch.cat(
        [surface.get_xyz.detach(), foliage.xyz.detach()[canonical]], dim=0
    )
    extent = torch.quantile(
        torch.linalg.vector_norm(
            support - scene_center[None], dim=1
        ),
        0.98,
    ).clamp_min(1.0)
    shell_radius = extent * 8.0
    count = int(sky_gaussians)
    index = torch.arange(count, device=device, dtype=dtype) + 0.5
    z = 1.0 - 2.0 * index / count
    phi = index * (math.pi * (3.0 - math.sqrt(5.0)))
    radius_xy = torch.sqrt((1.0 - z * z).clamp_min(0))
    directions = torch.stack(
        [radius_xy * torch.cos(phi), radius_xy * torch.sin(phi), z],
        dim=1,
    )
    shell_xyz = scene_center[None] + shell_radius * directions
    angular_scale = (
        shell_radius * math.sqrt(4.0 * math.pi / count) * 0.72
    )
    shell_scales = torch.empty(count, 3, device=device, dtype=dtype)
    shell_scales[:, :2] = angular_scale
    shell_scales[:, 2] = shell_radius * 0.018
    shell_quaternion = torch.from_numpy(
        _rotation_from_z(directions.detach().cpu().numpy())
    ).to(device=device, dtype=dtype)
    from outdoor.directional_sky import _eval_sh

    sky_rgb = torch.sigmoid(
        _eval_sh(sky.degree, sky.logit_sh, directions)
    )
    coefficients = surface.get_features.shape[1]
    shell_features = torch.zeros(
        count, coefficients, 3, device=device, dtype=dtype
    )
    shell_features[:, 0] = (sky_rgb - 0.5) / C0
    xyz_values.append(shell_xyz)
    scale_values.append(shell_scales)
    quaternion_values.append(shell_quaternion)
    opacity_values.append(
        torch.full((count, 1), 0.08, device=device, dtype=dtype)
    )
    feature_values.append(shell_features)
    role_values.append(
        torch.full(
            (count,),
            STUDENT_ROLE_SKY,
            dtype=torch.int8,
            device=device,
        )
    )

    xyz = torch.cat(xyz_values)
    scales = torch.cat(scale_values)
    quaternion = torch.cat(quaternion_values)
    opacity = torch.cat(opacity_values).clamp(1e-6, 1 - 1e-6)
    features = torch.cat(feature_values)
    seed = {
        "xyz": xyz,
        "log_scales": scales.clamp_min(1e-7).log(),
        "quaternions": quaternion,
        "opacity_logits": torch.logit(opacity),
        "features": features,
        "primitive_role": torch.cat(role_values),
    }
    surface_student_count = int(
        sum(
            int((roles == STUDENT_ROLE_RIGID).sum())
            for roles in role_values
        )
    )
    audit = {
        "surface_parent_count": len(surface.get_xyz),
        "surface_student_count": surface_student_count,
        "surface_subdivision_counts": {
            str(value): int((subdivisions == value).sum())
            for value in (1, 2, 3)
        },
        "canonical_foliage_count": int(canonical.sum()),
        "dynamic_foliage_omitted": int(
            foliage.dynamic_leaf_mask.sum()
        ) - baked_dynamic_count,
        "dynamic_foliage_baked": baked_dynamic_count,
        "dynamic_temporal_variance_mean": temporal_variance_mean,
        "dynamic_bake_policy": (
            "median_temporal_code_with_variance_opacity_attenuation"
            if baked_dynamic_count
            else "not_available"
        ),
        "sky_shell_count": count,
        "sky_shell_radius": float(shell_radius),
        "total_count": len(xyz),
        "training_roles_stripped_on_export": True,
    }
    return seed, audit


def standard_property_names(sh_degree: int) -> list[str]:
    coefficient_count = (int(sh_degree) + 1) ** 2
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names.extend(f"f_dc_{index}" for index in range(3))
    names.extend(
        f"f_rest_{index}"
        for index in range(3 * (coefficient_count - 1))
    )
    names.append("opacity")
    names.extend(f"scale_{index}" for index in range(3))
    names.extend(f"rot_{index}" for index in range(4))
    return names


@torch.no_grad()
def save_standard_3dgs_ply(
    path: Path,
    *,
    xyz: torch.Tensor,
    log_scales: torch.Tensor,
    quaternions: torch.Tensor,
    opacity_logits: torch.Tensor,
    features: torch.Tensor,
    sh_degree: int,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_coefficients = (int(sh_degree) + 1) ** 2
    if features.shape != (len(xyz), expected_coefficients, 3):
        raise ValueError(
            f"Expected SH shape {(len(xyz), expected_coefficients, 3)}, "
            f"got {tuple(features.shape)}"
        )
    if log_scales.shape != (len(xyz), 3):
        raise ValueError("Standard 3DGS requires exactly three scales")
    values = [
        xyz.detach().cpu().numpy(),
        np.zeros((len(xyz), 3), dtype=np.float32),
        features[:, 0]
        .detach()
        .cpu()
        .numpy(),
        features[:, 1:]
        .transpose(1, 2)
        .reshape(len(xyz), -1)
        .detach()
        .cpu()
        .numpy(),
        opacity_logits.reshape(-1, 1).detach().cpu().numpy(),
        log_scales.detach().cpu().numpy(),
        quaternions.detach().cpu().numpy(),
    ]
    attributes = np.concatenate(values, axis=1).astype(np.float32)
    names = standard_property_names(sh_degree)
    if attributes.shape[1] != len(names):
        raise RuntimeError("Standard PLY attribute count mismatch")
    table = np.empty(len(xyz), dtype=[(name, "f4") for name in names])
    table[:] = list(map(tuple, attributes))
    PlyData(
        [PlyElement.describe(table, "vertex")], text=False
    ).write(path)
    return path


def validate_standard_3dgs_ply(
    path: Path, *, sh_degree: int
) -> dict[str, Any]:
    path = Path(path)
    ply = PlyData.read(path)
    if len(ply.elements) != 1 or ply.elements[0].name != "vertex":
        raise RuntimeError("Standard 3DGS PLY must contain one vertex element")
    actual = [property.name for property in ply.elements[0].properties]
    expected = standard_property_names(sh_degree)
    if actual != expected:
        extra = sorted(set(actual) - set(expected))
        missing = sorted(set(expected) - set(actual))
        raise RuntimeError(
            f"Non-standard 3DGS schema: extra={extra}, missing={missing}"
        )
    return {
        "version": STANDARD_3DGS_VERSION,
        "path": str(path.resolve()),
        "vertex_count": int(ply.elements[0].count),
        "sh_degree": int(sh_degree),
        "property_count": len(actual),
        "custom_property_count": 0,
        "three_scales": True,
        "single_quaternion": True,
        "standard_graphdeco_schema": True,
    }


def load_standard_3dgs_ply(
    path: Path,
    *,
    sh_degree: int,
    device: str | torch.device = "cuda",
) -> dict[str, torch.Tensor]:
    """Load only the canonical Graphdeco fields; reject custom schemas."""
    validate_standard_3dgs_ply(path, sh_degree=sh_degree)
    vertex = PlyData.read(Path(path)).elements[0]
    names = [property.name for property in vertex.properties]
    values = np.column_stack(
        [np.asarray(vertex[name], dtype=np.float32) for name in names]
    )
    coefficient_count = (int(sh_degree) + 1) ** 2
    offset = 0
    xyz = values[:, offset : offset + 3]
    offset += 6  # xyz plus placeholder normals
    dc = values[:, offset : offset + 3]
    offset += 3
    rest_count = 3 * (coefficient_count - 1)
    rest = values[:, offset : offset + rest_count]
    offset += rest_count
    opacity = values[:, offset : offset + 1]
    offset += 1
    scales = values[:, offset : offset + 3]
    offset += 3
    rotation = values[:, offset : offset + 4]
    features = np.zeros(
        (len(values), coefficient_count, 3), dtype=np.float32
    )
    features[:, 0] = dc
    if coefficient_count > 1:
        features[:, 1:] = rest.reshape(
            len(values), 3, coefficient_count - 1
        ).transpose(0, 2, 1)
    target = torch.device(device)
    return {
        "xyz": torch.from_numpy(xyz).to(target),
        "log_scales": torch.from_numpy(scales).to(target),
        "quaternions": torch.from_numpy(rotation).to(target),
        "opacity_logits": torch.from_numpy(opacity).to(target),
        "features": torch.from_numpy(features).to(target),
    }
