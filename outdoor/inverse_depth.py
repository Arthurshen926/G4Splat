"""Precision-weighted inverse-depth fusion for fixed-camera Chart geometry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


INVERSE_DEPTH_FUSION_VERSION = "outdoor-inverse-depth-fusion-v1"


def _resize_float(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if array.shape == (height, width):
        return array.astype(np.float32, copy=False)
    image = Image.fromarray(array.astype(np.float32, copy=False), mode="F")
    return np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)


def _normalise_confidence(confidence: np.ndarray) -> np.ndarray:
    valid = confidence[np.isfinite(confidence) & (confidence > 0.0)]
    if valid.size == 0:
        return np.zeros_like(confidence, dtype=np.float32)
    high = max(float(np.quantile(valid, 0.95)), 1e-6)
    return np.clip(confidence / high, 0.0, 1.0).astype(np.float32)


def fuse_inverse_depth_sources(
    plane_depth: np.ndarray,
    plane_confidence: np.ndarray,
    chart_depth: np.ndarray | None,
    chart_confidence: np.ndarray | None,
    mono_depth: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Fuse bounded plane, aligned Chart and mono depth in inverse-depth space.

    The result retains source bitmasks and variance, so unsupported regions are
    emitted with zero confidence instead of being converted into opaque
    geometry by a global absolute-depth threshold.
    """
    plane_depth = np.asarray(plane_depth, dtype=np.float32)
    shape = plane_depth.shape
    if plane_depth.ndim != 2:
        raise ValueError("plane_depth must be two-dimensional")
    plane_confidence = _resize_float(np.asarray(plane_confidence, dtype=np.float32), shape)
    sources: list[tuple[np.ndarray, np.ndarray, int]] = []

    plane_valid = np.isfinite(plane_depth) & (plane_depth > 0.0) & np.isfinite(plane_confidence)
    plane_weight = np.clip(plane_confidence, 0.0, 1.0) * plane_valid
    sources.append((1.0 / np.maximum(plane_depth, 1e-6), plane_weight.astype(np.float32), 1))

    def add_optional(depth: np.ndarray | None, confidence: np.ndarray | None, bit: int, base_precision: float) -> None:
        if depth is None:
            return
        value = _resize_float(np.asarray(depth, dtype=np.float32), shape)
        if confidence is None:
            confidence_value = np.ones(shape, dtype=np.float32)
        else:
            confidence_value = _normalise_confidence(_resize_float(np.asarray(confidence, dtype=np.float32), shape))
        valid = np.isfinite(value) & (value > 0.0) & np.isfinite(confidence_value)
        # Plane and aligned-chart disagreement is recorded in the variance;
        # it downweights the lower-priority source rather than hard overwriting
        # a distant building plane with monocular depth.
        weight = base_precision * confidence_value * valid
        sources.append((1.0 / np.maximum(value, 1e-6), weight.astype(np.float32), bit))

    add_optional(chart_depth, chart_confidence, 2, 0.45)
    add_optional(mono_depth, None, 4, 0.06)

    numerator = np.zeros(shape, dtype=np.float64)
    precision = np.zeros(shape, dtype=np.float64)
    source_bitmask = np.zeros(shape, dtype=np.uint8)
    support_view_count = np.zeros(shape, dtype=np.uint8)
    source_values: list[tuple[np.ndarray, np.ndarray]] = []
    for rho, weight, bit in sources:
        valid = weight > 0.0
        numerator += weight * rho
        precision += weight
        source_bitmask[valid] |= np.uint8(bit)
        support_view_count[valid] += 1
        source_values.append((rho, weight))
    valid = precision > 1e-8
    rho_mean = np.zeros(shape, dtype=np.float32)
    rho_mean[valid] = (numerator[valid] / precision[valid]).astype(np.float32)
    variance = np.full(shape, np.inf, dtype=np.float32)
    if np.any(valid):
        residual = np.zeros(shape, dtype=np.float64)
        for rho, weight in source_values:
            residual += weight * (rho - rho_mean) ** 2
        variance[valid] = (1.0 / precision[valid] + residual[valid] / precision[valid]).astype(np.float32)
    depth = np.zeros(shape, dtype=np.float32)
    depth[valid] = 1.0 / np.maximum(rho_mean[valid], 1e-8)
    relative_variance = variance / np.maximum(rho_mean * rho_mean, 1e-12)
    confidence = np.zeros(shape, dtype=np.float32)
    confidence[valid] = np.clip(
        precision[valid] / 1.25 * np.exp(-np.sqrt(np.maximum(relative_variance[valid], 0.0))),
        0.0,
        1.0,
    )
    return {
        "depth": depth,
        "confidence": confidence,
        "rho_mean": rho_mean,
        "rho_variance": variance,
        "source_bitmask": source_bitmask,
        "support_view_count": support_view_count,
    }


def fuse_inverse_depth_directory(
    mast3r_scene: Path,
    plane_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Fuse every real Chart depth map and write train-compatible TIFF/PNG maps."""
    mast3r_scene = Path(mast3r_scene).resolve()
    plane_root = Path(plane_root).resolve()
    output = Path(output)
    charts_path = mast3r_scene / "charts_data.npz"
    if not charts_path.is_file():
        raise FileNotFoundError(charts_path)
    if not plane_root.is_dir():
        raise FileNotFoundError(plane_root)
    charts = np.load(charts_path)
    chart_depths = charts["depths"] if "depths" in charts.files else None
    chart_confs = charts["confs"] if "confs" in charts.files else None
    mono_depths = charts["prior_depths"] if "prior_depths" in charts.files else None
    count = int(chart_depths.shape[0]) if chart_depths is not None else 0
    if count <= 0:
        raise RuntimeError("charts_data has no camera-indexed depth tensor")
    output.mkdir(parents=True, exist_ok=True)
    frame_summaries: list[dict[str, Any]] = []
    for index in range(count):
        depth_path = plane_root / f"refine_depth_frame{index:06d}.tiff"
        confidence_path = plane_root / f"confident_map_frame{index:06d}.png"
        if not depth_path.is_file() or not confidence_path.is_file():
            raise FileNotFoundError(
                "Plane refinement is incomplete; expected " + str(depth_path) + " and " + str(confidence_path)
            )
        plane_depth = np.asarray(Image.open(depth_path), dtype=np.float32)
        plane_confidence = np.asarray(Image.open(confidence_path), dtype=np.float32) / 255.0
        fused = fuse_inverse_depth_sources(
            plane_depth,
            plane_confidence,
            chart_depths[index] if chart_depths is not None else None,
            chart_confs[index] if chart_confs is not None else None,
            mono_depths[index] if mono_depths is not None else None,
        )
        Image.fromarray(fused["depth"].astype(np.float32), mode="F").save(
            output / f"refine_depth_frame{index:06d}.tiff"
        )
        Image.fromarray(np.rint(fused["confidence"] * 255.0).astype(np.uint8)).save(
            output / f"confident_map_frame{index:06d}.png"
        )
        np.save(output / f"rho_mean_frame{index:06d}.npy", fused["rho_mean"])
        np.save(output / f"rho_variance_frame{index:06d}.npy", fused["rho_variance"])
        np.save(output / f"source_bitmask_frame{index:06d}.npy", fused["source_bitmask"])
        np.save(output / f"support_view_count_frame{index:06d}.npy", fused["support_view_count"])
        frame_summaries.append(
            {
                "frame": index,
                "valid_fraction": float(np.mean(fused["confidence"] > 0.0)),
                "mean_confidence": float(np.mean(fused["confidence"])),
                "source_bitmask_values": [int(value) for value in np.unique(fused["source_bitmask"])],
            }
        )
    payload = {
        "schema_version": INVERSE_DEPTH_FUSION_VERSION,
        "mast3r_scene": str(mast3r_scene),
        "plane_root": str(plane_root),
        "source_priority": ["bounded_plane", "aligned_chart", "calibrated_mono"],
        "fusion_space": "inverse_depth",
        "fixed_absolute_depth_limit": None,
        "frame_count": count,
        "frames": frame_summaries,
    }
    (output / "inverse_depth_fusion_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload
