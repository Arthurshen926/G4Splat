"""Bake intrinsic UV ownership into smaller native 2D surfels.

The mixed renderer can mask a legacy surfel in its intrinsic tangent-plane
coordinates.  A clean structural model should not depend on the contaminated
parent forever, though.  This module converts the retained UV cells into
ordinary 2DGS primitives and records their parent/cell provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


ROLE_UNCHANGED = 0
ROLE_UV_RETAINED = 1
ROLE_RIGID_EVIDENCE = 2


@dataclass(frozen=True)
class BakedSurfelResult:
    vertices: np.ndarray
    source_parent_index: np.ndarray
    source_uv_x: np.ndarray
    source_uv_y: np.ndarray
    ownership_role: np.ndarray
    audit: dict


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    output = np.empty_like(value, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert scalar-first 2DGS quaternions to row-major rotation matrices."""
    value = np.asarray(quaternion, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError("quaternion must have shape [N,4]")
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError("quaternion contains a zero-norm row")
    w, x, y, z = (value / norm).T
    result = np.empty((len(value), 3, 3), dtype=np.float64)
    result[:, 0, 0] = 1 - 2 * (y * y + z * z)
    result[:, 0, 1] = 2 * (x * y - w * z)
    result[:, 0, 2] = 2 * (x * z + w * y)
    result[:, 1, 0] = 2 * (x * y + w * z)
    result[:, 1, 1] = 1 - 2 * (x * x + z * z)
    result[:, 1, 2] = 2 * (y * z - w * x)
    result[:, 2, 0] = 2 * (x * z - w * y)
    result[:, 2, 1] = 2 * (y * z + w * x)
    result[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return result


def _required_statistics(
    statistics: Mapping[str, np.ndarray], count: int
) -> dict[str, np.ndarray]:
    names = (
        "canopy_responsibility",
        "rigid_responsibility",
        "maximum_view_rigid_responsibility",
        "support_views",
        "maximum_projected_radius",
        "minimum_camera_depth",
    )
    output = {}
    for name in names:
        if name not in statistics:
            raise ValueError(f"responsibility statistics missing {name}")
        value = np.asarray(statistics[name]).reshape(-1)
        if len(value) != count:
            raise ValueError(
                f"responsibility statistic {name} has {len(value)} rows, "
                f"expected {count}"
            )
        output[name] = value
    return output


def _field_names(vertices: np.ndarray, prefix: str) -> list[str]:
    return sorted(
        (name for name in vertices.dtype.names or () if name.startswith(prefix)),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )


def bake_surfel_uv_ownership(
    vertices: np.ndarray,
    candidate_indices: np.ndarray,
    gate_atlas: np.ndarray,
    evidence_sum: np.ndarray,
    evidence_weight: np.ndarray,
    responsibility_statistics: Mapping[str, np.ndarray],
    *,
    mode: str,
    rigid_threshold: float = 0.05,
    rigid_dominance: float = 1.0,
    global_canopy_threshold: float = 0.45,
    global_rigid_threshold: float = 0.20,
    minimum_global_support_views: int = 3,
    maximum_bake_radius: float = 4096.0,
    minimum_bake_camera_depth: float = 1.0,
    child_sigma_over_spacing: float = 0.55,
    minimum_child_peak_alpha: float = 1.0 / 255.0,
    remove_nonstructural: bool = True,
) -> BakedSurfelResult:
    """Replace candidate parents by local children or produce a rigid-only base.

    ``uv-equivalent`` approximates the learned intrinsic gate.  ``rigid-clean``
    keeps only candidate cells with direct rigid-dominant evidence and removes
    other audited canopy-only primitives.  Geometrically unsafe candidate
    parents are never inherited as children in rigid-clean mode.
    """
    if mode not in {"uv-equivalent", "rigid-clean"}:
        raise ValueError(f"unsupported bake mode: {mode}")
    if vertices.dtype.names is None:
        raise ValueError("vertices must be a structured PLY vertex array")
    required_vertex_fields = {
        "x",
        "y",
        "z",
        "opacity",
        "scale_0",
        "scale_1",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    missing = sorted(required_vertex_fields - set(vertices.dtype.names))
    if missing:
        raise ValueError(f"vertices missing fields: {missing}")
    candidates = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    if len(np.unique(candidates)) != len(candidates):
        raise ValueError("candidate_indices contains duplicates")
    if len(candidates) and (
        int(candidates.min()) < 0 or int(candidates.max()) >= len(vertices)
    ):
        raise ValueError("candidate_indices is outside the vertex array")
    atlas = np.asarray(gate_atlas, dtype=np.float64)
    evidence = np.asarray(evidence_sum, dtype=np.float64)
    weight = np.asarray(evidence_weight, dtype=np.float64)
    if atlas.ndim != 3 or atlas.shape[1] != atlas.shape[2]:
        raise ValueError("gate_atlas must have shape [K,G,G]")
    if atlas.shape[0] != len(candidates):
        raise ValueError("gate_atlas and candidate_indices disagree")
    if evidence.shape != (*atlas.shape, 4):
        raise ValueError("evidence_sum must have shape [K,G,G,4]")
    if weight.shape != atlas.shape:
        raise ValueError("evidence_weight must have shape [K,G,G]")
    if not 0 < child_sigma_over_spacing <= 1:
        raise ValueError("child_sigma_over_spacing must be in (0,1]")

    statistics = _required_statistics(
        responsibility_statistics, len(vertices)
    )
    candidate_mask = np.zeros(len(vertices), dtype=bool)
    candidate_mask[candidates] = True
    retain_original = ~candidate_mask
    global_canopy_only = (
        (
            statistics["canopy_responsibility"]
            >= float(global_canopy_threshold)
        )
        & (
            statistics["rigid_responsibility"]
            <= float(global_rigid_threshold)
        )
        & (
            statistics["maximum_view_rigid_responsibility"]
            <= float(global_rigid_threshold)
        )
        & (
            statistics["support_views"]
            >= int(minimum_global_support_views)
        )
    )
    if mode == "rigid-clean":
        retain_original &= ~global_canopy_only
    nonstructural = np.zeros(len(vertices), dtype=bool)
    if (
        mode == "rigid-clean"
        and remove_nonstructural
        and "primitive_class" in vertices.dtype.names
    ):
        nonstructural = (
            np.asarray(vertices["primitive_class"]).reshape(-1) != 0
        )
        retain_original &= ~nonstructural

    observed = weight > 1e-8
    average = evidence / np.maximum(weight[..., None], 1e-8)
    canopy = average[..., 0]
    rigid = average[..., 1]
    if mode == "uv-equivalent":
        local_gate = np.clip(atlas, 0.0, 1.0)
        local_role = np.full(atlas.shape, ROLE_UV_RETAINED, dtype=np.int8)
        geometry_safe = np.ones(len(candidates), dtype=bool)
    else:
        rigid_cell = (
            observed
            & (rigid >= float(rigid_threshold))
            & (rigid >= float(rigid_dominance) * canopy)
        )
        local_gate = rigid_cell.astype(np.float64)
        local_role = np.full(
            atlas.shape, ROLE_RIGID_EVIDENCE, dtype=np.int8
        )
        geometry_safe = (
            statistics["maximum_projected_radius"][candidates]
            <= float(maximum_bake_radius)
        ) & (
            statistics["minimum_camera_depth"][candidates]
            >= float(minimum_bake_camera_depth)
        )
        local_gate *= geometry_safe[:, None, None]

    grid_size = int(atlas.shape[1])
    coordinates = np.linspace(-3.0, 3.0, grid_size, dtype=np.float64)
    grid_y, grid_x = np.meshgrid(coordinates, coordinates, indexing="ij")
    spacing = 6.0 / max(grid_size - 1, 1)
    child_sigma = float(child_sigma_over_spacing) * spacing
    if grid_size == 1:
        child_sigma = float(child_sigma_over_spacing)

    parent_opacity = _sigmoid(
        np.asarray(vertices["opacity"][candidates], dtype=np.float64)
    )
    base_falloff = np.exp(-0.5 * (grid_x * grid_x + grid_y * grid_y))
    target_alpha = (
        parent_opacity[:, None, None] * base_falloff * local_gate
    )

    # Partition-of-unity calibration on the infinite regular lattice.  Optical
    # density is used so overlapping coplanar children reproduce high alpha
    # more faithfully than a linear opacity split.
    offsets = np.arange(-5, 6, dtype=np.float64)
    one_dimensional_sum = np.exp(
        -0.5 * (offsets / float(child_sigma_over_spacing)) ** 2
    ).sum()
    partition_scale = 1.0 / float(one_dimensional_sum**2)
    target_tau = -np.log1p(-np.clip(target_alpha, 0.0, 1.0 - 1e-6))
    child_peak_opacity = 1.0 - np.exp(-partition_scale * target_tau)
    active = child_peak_opacity >= float(minimum_child_peak_alpha)

    parent_rows, uv_y, uv_x = np.nonzero(active)
    source_parent = candidates[parent_rows]
    children = vertices[source_parent].copy()
    rotations = np.stack(
        [children[f"rot_{index}"] for index in range(4)], axis=1
    )
    matrices = quaternion_to_matrix(rotations)
    scales = np.exp(
        np.stack(
            [children["scale_0"], children["scale_1"]], axis=1
        ).astype(np.float64)
    )
    u = coordinates[uv_x]
    v = coordinates[uv_y]
    centers = np.stack(
        [children["x"], children["y"], children["z"]], axis=1
    ).astype(np.float64)
    centers += (
        matrices[:, :, 0] * (scales[:, 0] * u)[:, None]
        + matrices[:, :, 1] * (scales[:, 1] * v)[:, None]
    )
    for column, name in enumerate(("x", "y", "z")):
        children[name] = centers[:, column].astype(children[name].dtype)
    log_child_sigma = math.log(child_sigma)
    children["scale_0"] = (
        children["scale_0"].astype(np.float64) + log_child_sigma
    ).astype(children["scale_0"].dtype)
    children["scale_1"] = (
        children["scale_1"].astype(np.float64) + log_child_sigma
    ).astype(children["scale_1"].dtype)
    selected_opacity = child_peak_opacity[parent_rows, uv_y, uv_x]
    children["opacity"] = np.log(
        selected_opacity / np.maximum(1.0 - selected_opacity, 1e-12)
    ).astype(children["opacity"].dtype)
    if "protected_flag" in vertices.dtype.names and mode == "rigid-clean":
        children["protected_flag"] = 1.0
    if "geometry_confidence" in vertices.dtype.names:
        confidence = rigid[parent_rows, uv_y, uv_x]
        children["geometry_confidence"] = np.maximum(
            children["geometry_confidence"].astype(np.float64),
            confidence,
        ).astype(children["geometry_confidence"].dtype)

    original_indices = np.flatnonzero(retain_original)
    output_vertices = np.concatenate(
        [vertices[original_indices].copy(), children]
    )
    source_parent_index = np.concatenate(
        [original_indices, source_parent]
    ).astype(np.int64)
    source_uv_x = np.concatenate(
        [
            np.full(len(original_indices), -1, dtype=np.int16),
            uv_x.astype(np.int16),
        ]
    )
    source_uv_y = np.concatenate(
        [
            np.full(len(original_indices), -1, dtype=np.int16),
            uv_y.astype(np.int16),
        ]
    )
    ownership_role = np.concatenate(
        [
            np.full(
                len(original_indices), ROLE_UNCHANGED, dtype=np.int8
            ),
            local_role[parent_rows, uv_y, uv_x],
        ]
    )
    audit = {
        "mode": mode,
        "input_surfel_count": int(len(vertices)),
        "candidate_parent_count": int(len(candidates)),
        "retained_unchanged_count": int(len(original_indices)),
        "baked_child_count": int(len(children)),
        "output_surfel_count": int(len(output_vertices)),
        "removed_global_canopy_only_count": int(
            (global_canopy_only & ~candidate_mask).sum()
            if mode == "rigid-clean"
            else 0
        ),
        "removed_nonstructural_count": int(
            (nonstructural & ~candidate_mask).sum()
        ),
        "unsafe_candidate_parent_count": int((~geometry_safe).sum()),
        "observed_uv_texel_count": int(observed.sum()),
        "active_uv_texel_count": int(active.sum()),
        "rigid_dominant_texel_count": int(
            (
                observed
                & (rigid >= float(rigid_threshold))
                & (rigid >= float(rigid_dominance) * canopy)
            ).sum()
        ),
        "child_sigma": float(child_sigma),
        "partition_scale": float(partition_scale),
        "thresholds": {
            "rigid": float(rigid_threshold),
            "rigid_dominance": float(rigid_dominance),
            "global_canopy": float(global_canopy_threshold),
            "global_rigid": float(global_rigid_threshold),
            "minimum_global_support_views": int(
                minimum_global_support_views
            ),
            "maximum_bake_radius": float(maximum_bake_radius),
            "minimum_bake_camera_depth": float(
                minimum_bake_camera_depth
            ),
            "minimum_child_peak_alpha": float(
                minimum_child_peak_alpha
            ),
            "remove_nonstructural": bool(remove_nonstructural),
        },
    }
    return BakedSurfelResult(
        vertices=output_vertices,
        source_parent_index=source_parent_index,
        source_uv_x=source_uv_x,
        source_uv_y=source_uv_y,
        ownership_role=ownership_role,
        audit=audit,
    )
