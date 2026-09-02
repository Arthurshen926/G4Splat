"""Source-separated MoGe3 base-depth evidence for the MAtCha Chart atlas.

The original MAtCha archive remains immutable.  This module builds two
explicit alternatives in Cambridge metric camera-z units: a direct MoGe3
base and a confidence-adaptive base.  Neither variant may silently replace
the Chart source; the training contract selects one by name.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


MOGE3_CHART_BASE_VERSION = "moge3-matcha-chart-base-v1-source-separated"
MOGE3_CHART_BASE_SOURCES = (
    "matcha",
    "moge3",
    "moge3_adaptive",
)


def robust_scene_scale(
    log_ratios_by_view: list[np.ndarray],
    *,
    consensus_log_radius: float = 0.18,
    minimum_views: int = 2,
    minimum_pixels: int = 256,
    maximum_log_sigma: float = 0.12,
) -> dict[str, Any]:
    """Fit one positive MoGe-to-Cambridge scale from independent views."""
    values: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    for view, raw in enumerate(log_ratios_by_view):
        value = np.asarray(raw, dtype=np.float64).reshape(-1)
        value = value[np.isfinite(value)]
        if len(value) > 4096:
            value = value[:: max(len(value) // 4096, 1)][:4096]
        if len(value):
            values.append(value)
            owners.append(np.full(len(value), view, dtype=np.int32))
    if not values:
        raise RuntimeError("MoGe3 Chart calibration has no finite overlap")
    value = np.concatenate(values)
    owner = np.concatenate(owners)
    if len(value) < int(minimum_pixels):
        raise RuntimeError("MoGe3 Chart calibration has insufficient pixels")
    radius = float(consensus_log_radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("consensus_log_radius must be positive")
    ordered = np.sort(value, kind="mergesort")
    left = best_left = best_right = 0
    for right in range(len(ordered)):
        while ordered[right] - ordered[left] > 2.0 * radius:
            left += 1
        if right + 1 - left > best_right - best_left:
            best_left, best_right = left, right + 1
    center = float(np.median(ordered[best_left:best_right]))
    inlier = np.abs(value - center) <= radius
    for _ in range(3):
        center = float(np.median(value[inlier]))
        inlier = np.abs(value - center) <= radius
    minimum_per_view = max(
        1,
        min(
            32,
            int(math.ceil(int(minimum_pixels) / max(8 * minimum_views, 1))),
        ),
    )
    unique, counts = np.unique(owner[inlier], return_counts=True)
    supported = unique[counts >= minimum_per_view]
    inlier &= np.isin(owner, supported)
    if int(inlier.sum()) < int(minimum_pixels) or len(supported) < int(
        minimum_views
    ):
        raise RuntimeError(
            "MoGe3 Chart calibration lacks cross-view scale consensus"
        )
    center = float(np.median(value[inlier]))
    sigma = float(1.4826 * np.median(np.abs(value[inlier] - center)))
    if not math.isfinite(sigma) or sigma > float(maximum_log_sigma):
        raise RuntimeError(
            "MoGe3 Chart calibration consensus is too broad: "
            f"sigma={sigma:.6f}"
        )
    return {
        "metric_to_cambridge_scale": float(math.exp(center)),
        "global_log_scale_sigma": sigma,
        "scale_support_pixels": int(inlier.sum()),
        "scale_support_views": int(len(supported)),
        "scale_rejected_pixels": int(len(value) - inlier.sum()),
        "scale_consensus_log_radius": radius,
    }


def chart_base_variants(
    matcha_depth_m: np.ndarray,
    moge3_depth_m: np.ndarray,
    *,
    rigid_valid: np.ndarray,
    refinement_sigma: np.ndarray,
    normal_direct_camera: np.ndarray,
    normal_depth_camera: np.ndarray,
    depth_normal_valid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build direct and adaptive Chart bases without erasing MAtCha holes."""
    matcha = np.asarray(matcha_depth_m, dtype=np.float32)
    moge = np.asarray(moge3_depth_m, dtype=np.float32)
    rigid = np.asarray(rigid_valid, dtype=bool)
    sigma = np.asarray(refinement_sigma, dtype=np.float32)
    direct = np.asarray(normal_direct_camera, dtype=np.float32)
    derived = np.asarray(normal_depth_camera, dtype=np.float32)
    derived_valid = np.asarray(depth_normal_valid, dtype=bool)
    if matcha.shape != moge.shape or matcha.shape != rigid.shape:
        raise ValueError("MAtCha, MoGe3 and rigid masks must align")
    if sigma.shape != matcha.shape or derived_valid.shape != matcha.shape:
        raise ValueError("MoGe3 confidence rasters must align with depth")
    if direct.shape != (*matcha.shape, 3) or derived.shape != direct.shape:
        raise ValueError("MoGe3 normal rasters must be HxWx3")
    matcha_valid = np.isfinite(matcha) & (matcha > 0.05)
    moge_valid = np.isfinite(moge) & (moge > 0.05)
    direct_norm = np.linalg.norm(direct, axis=-1)
    derived_norm = np.linalg.norm(derived, axis=-1)
    normal_valid = (
        np.isfinite(direct).all(-1)
        & np.isfinite(derived).all(-1)
        & (direct_norm > 1.0e-5)
        & (derived_norm > 1.0e-5)
        & derived_valid
    )
    direct_unit = np.zeros_like(direct)
    derived_unit = np.zeros_like(derived)
    direct_unit[normal_valid] = (
        direct[normal_valid] / direct_norm[normal_valid, None]
    )
    derived_unit[normal_valid] = (
        derived[normal_valid] / derived_norm[normal_valid, None]
    )
    normal_agreement = np.clip(
        np.sum(direct_unit * derived_unit, axis=-1), 0.0, 1.0
    ) ** 2
    refinement_precision = 1.0 / (
        1.0 + np.square(np.maximum(sigma, 0.0) / 0.12)
    )
    log_matcha = np.log(np.maximum(matcha, 1.0e-6))
    log_moge = np.log(np.maximum(moge, 1.0e-6))
    edge_x = np.zeros_like(log_moge)
    edge_y = np.zeros_like(log_moge)
    if log_moge.shape[1] > 1:
        delta_x = np.abs(log_moge[:, 1:] - log_moge[:, :-1])
        edge_x[:, 1:] = np.maximum(edge_x[:, 1:], delta_x)
        edge_x[:, :-1] = np.maximum(edge_x[:, :-1], delta_x)
    if log_moge.shape[0] > 1:
        delta_y = np.abs(log_moge[1:, :] - log_moge[:-1, :])
        edge_y[1:, :] = np.maximum(edge_y[1:, :], delta_y)
        edge_y[:-1, :] = np.maximum(edge_y[:-1, :], delta_y)
    boundary_precision = 1.0 / (
        1.0 + np.square(np.maximum(edge_x, edge_y) / 0.08)
    )
    agreement = 1.0 / (
        1.0 + np.square((log_matcha - log_moge) / 0.12)
    )
    valid = rigid & moge_valid & normal_valid
    precision = np.where(
        valid,
        refinement_precision * normal_agreement * boundary_precision,
        0.0,
    ).astype(np.float32)
    # MAtCha/MASt3R remains the multiview authority.  Disagreement therefore
    # reduces, but never hard-thresholds, the amount of MoGe detail admitted.
    adaptive_weight = np.where(
        valid & matcha_valid,
        precision * agreement,
        np.where(valid & ~matcha_valid, precision, 0.0),
    ).astype(np.float32)
    direct_base = matcha.copy()
    direct_base[valid] = moge[valid]
    adaptive = matcha.copy()
    both = valid & matcha_valid
    adaptive[both] = np.exp(
        (1.0 - adaptive_weight[both]) * log_matcha[both]
        + adaptive_weight[both] * log_moge[both]
    )
    only_moge = valid & ~matcha_valid
    adaptive[only_moge] = moge[only_moge]
    output_valid = matcha_valid | valid
    direct_base[~output_valid] = 0.0
    adaptive[~output_valid] = 0.0
    return {
        "depth_moge3": direct_base.astype(np.float32),
        "depth_moge3_adaptive": adaptive.astype(np.float32),
        "validity": output_valid.astype(np.uint8),
        "moge3_precision": precision,
        "moge3_adaptive_weight": adaptive_weight,
    }


def load_chart_base_metadata(
    path: Path,
    *,
    expected_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Load the Chart/scale contract without materializing depth rasters.

    The MoGe-to-Cambridge scene gauge is evidence metadata, not a property of
    whichever depth raster is selected as the current Chart base. A MAtCha
    control can therefore consume the same MoGe scale for auxiliary factors
    without silently replacing its Chart geometry.
    """
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {"schema_version", "image_names", "metadata_json"}
        missing = required - set(archive.files)
        if missing:
            raise RuntimeError(
                "MoGe3 Chart base lacks metadata fields: "
                + ", ".join(sorted(missing))
            )
        schema = str(archive["schema_version"].item())
        if schema != MOGE3_CHART_BASE_VERSION:
            raise RuntimeError(f"Unsupported MoGe3 Chart schema: {schema}")
        names = [str(value) for value in archive["image_names"].tolist()]
        metadata = json.loads(str(archive["metadata_json"].item()))
    if not isinstance(metadata, dict):
        raise RuntimeError("MoGe3 Chart metadata must be a JSON object")
    metadata_schema = metadata.get("schema_version")
    if metadata_schema is not None and metadata_schema != schema:
        raise RuntimeError("MoGe3 Chart metadata schema does not match archive")
    if expected_names is not None:
        expected = [Path(str(value)).stem for value in expected_names]
        actual = [Path(value).stem for value in names]
        if actual != expected:
            raise RuntimeError("MoGe3 Chart camera order does not match MAtCha")
    return {"image_names": names, "metadata": metadata}


def load_chart_base(
    path: Path,
    *,
    source: str,
    expected_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Load and validate one explicitly selected Chart base variant."""
    if source not in MOGE3_CHART_BASE_SOURCES[1:]:
        raise ValueError(f"Unsupported MoGe3 Chart source: {source!r}")
    contract = load_chart_base_metadata(path, expected_names=expected_names)
    with np.load(Path(path), allow_pickle=False) as archive:
        depth_key = "depth_moge3" if source == "moge3" else (
            "depth_moge3_adaptive"
        )
        raw_precision = archive["moge3_precision"].astype(np.float32)
        result = {
            "depth": archive[depth_key].astype(np.float32),
            "validity": archive["validity"].astype(bool),
            # Pixels without an admitted MoGe observation retain the original
            # MAtCha base and therefore retain unit source precision.  A zero
            # here must not erase the explicit fallback geometry.
            "precision": np.where(
                raw_precision > 0.0, raw_precision, 1.0
            ).astype(np.float32),
            "adaptive_weight": archive[
                "moge3_adaptive_weight"
            ].astype(np.float32),
            "image_names": contract["image_names"],
            "metadata": contract["metadata"],
        }
    if result["depth"].shape != result["validity"].shape or (
        result["precision"].shape != result["depth"].shape
    ):
        raise RuntimeError("MoGe3 Chart base arrays do not align")
    return result
