"""Precision-weighted inverse-depth fusion for fixed-camera Chart geometry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


INVERSE_DEPTH_FUSION_VERSION = (
    "outdoor-inverse-depth-fusion-v4-crossview-confirmed-world-metric"
)


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


def _resize_bool(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    value = np.asarray(array, dtype=bool)
    if value.shape == (height, width):
        return value
    image = Image.fromarray(value.astype(np.uint8) * 255)
    return np.asarray(
        image.resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8
    ) > 0


def _resize_support_count(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    value = np.asarray(array, dtype=np.uint8)
    if value.shape == (height, width):
        return value
    image = Image.fromarray(value)
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8)


def fuse_inverse_depth_sources(
    plane_depth: np.ndarray,
    plane_confidence: np.ndarray,
    chart_depth: np.ndarray | None,
    chart_confidence: np.ndarray | None,
    mono_depth: np.ndarray | None,
    *,
    plane_valid_mask: np.ndarray | None = None,
    plane_support_view_count: np.ndarray | None = None,
    chart_support_view_count: np.ndarray | None = None,
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
    if plane_valid_mask is None:
        # Compatibility for unit-level callers and explicitly old artifacts.
        # The directory fusion path requires an explicit residual-plane mask.
        plane_valid = np.isfinite(plane_depth) & (plane_depth > 0.0)
    else:
        plane_valid = _resize_bool(plane_valid_mask, shape)
    if plane_support_view_count is None:
        plane_support = np.where(plane_valid, 1, 0).astype(np.uint8)
    else:
        plane_support = _resize_support_count(plane_support_view_count, shape)
        plane_support = np.where(plane_valid, plane_support, 0).astype(np.uint8)
    sources: list[tuple[np.ndarray, np.ndarray, int, np.ndarray]] = []

    plane_valid = plane_valid & np.isfinite(plane_depth) & (plane_depth > 0.0) & np.isfinite(plane_confidence)
    plane_weight = np.clip(plane_confidence, 0.0, 1.0) * plane_valid
    sources.append((
        1.0 / np.maximum(plane_depth, 1e-6),
        plane_weight.astype(np.float32),
        1,
        plane_support,
    ))

    def add_optional(
        depth: np.ndarray | None,
        confidence: np.ndarray | None,
        bit: int,
        base_precision: float,
        support_count: np.ndarray | None,
    ) -> None:
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
        if support_count is None:
            support = np.where(valid, 1, 0).astype(np.uint8)
        else:
            support = _resize_support_count(support_count, shape)
            support = np.where(valid, support, 0).astype(np.uint8)
        sources.append((1.0 / np.maximum(value, 1e-6), weight.astype(np.float32), bit, support))

    add_optional(chart_depth, chart_confidence, 2, 0.45, chart_support_view_count)
    # Monocular fallback is not independent multi-view evidence.
    add_optional(mono_depth, None, 4, 0.06, np.zeros(shape, dtype=np.uint8))

    numerator = np.zeros(shape, dtype=np.float64)
    precision = np.zeros(shape, dtype=np.float64)
    source_bitmask = np.zeros(shape, dtype=np.uint8)
    support_view_count = np.zeros(shape, dtype=np.uint8)
    source_values: list[tuple[np.ndarray, np.ndarray]] = []
    for rho, weight, bit, independent_support in sources:
        valid = weight > 0.0
        numerator += weight * rho
        precision += weight
        source_bitmask[valid] |= np.uint8(bit)
        # Source families are not independent camera observations.  A plane
        # can already include the target Chart view, so report the strongest
        # proven distinct-view support instead of double-counting plane+Chart.
        support_view_count[valid] = np.maximum(
            support_view_count[valid], independent_support[valid]
        )
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
    *,
    chart_consensus: Path | None = None,
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
    # Synthetic/unit-level archives historically omitted scale_factor and are
    # already metric. Production MAtCha archives always provide it.
    scale_factor = float(
        charts["scale_factor"] if "scale_factor" in charts.files else 1.0
    )
    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise RuntimeError(
            f"Invalid MAtCha chart scale_factor {scale_factor}"
        )
    # Plane refinement is emitted in the fixed Cambridge camera/world scale.
    # MAtCha keeps its atlas depths in the normalized optimization scale.
    # Convert every optional dense source to Cambridge metric depth *before*
    # inverse-depth fusion; a mixed-unit rho cache cannot be repaired later.
    chart_depths = (
        charts["depths"].astype(np.float32) / scale_factor
        if "depths" in charts.files
        else None
    )
    chart_confs = charts["confs"] if "confs" in charts.files else None
    chart_support_counts = None
    for key in ("support_view_count", "support_view_counts", "chart_support_view_count"):
        if key in charts.files:
            chart_support_counts = charts[key]
            break
    chart_consensus_contract = None
    if chart_consensus is not None:
        chart_consensus = Path(chart_consensus).resolve()
        with np.load(chart_consensus, allow_pickle=False) as sidecar:
            required = {
                "depths",
                "support_counts",
                "consistency_weights",
                "correction_mask",
                "image_names",
            }
            missing = required - set(sidecar.files)
            if missing:
                raise RuntimeError(
                    "Chart consensus lacks required arrays: "
                    + ", ".join(sorted(missing))
                )
            candidate_depths = sidecar["depths"].astype(np.float32)
            support = sidecar["support_counts"].astype(np.uint8)
            accepted = sidecar["correction_mask"].astype(bool)
            consistency = sidecar["consistency_weights"].astype(np.float32)
            minimum_support = int(
                sidecar["minimum_consensus_views"].item()
                if "minimum_consensus_views" in sidecar
                else 2
            )
            consensus_names = [
                Path(str(value)).stem for value in sidecar["image_names"]
            ]
        camera_names = [
            Path(str(value)).stem
            for value in json.loads(
                (mast3r_scene / "cameras.json").read_text(encoding="utf-8")
            )["filepaths"]
        ]
        if consensus_names != camera_names:
            raise RuntimeError(
                "Chart consensus camera order does not match Chart atlas"
            )
        if chart_depths is None or candidate_depths.shape != chart_depths.shape:
            raise RuntimeError("Chart consensus depth shape does not match atlas")
        confirmed = accepted & (support >= minimum_support)
        chart_depths = candidate_depths / scale_factor
        base_confidence = (
            np.asarray(chart_confs, dtype=np.float32)
            if chart_confs is not None
            else np.ones_like(chart_depths, dtype=np.float32)
        )
        chart_confs = (
            base_confidence
            * np.clip(consistency, 0.0, 1.0)
            * confirmed.astype(np.float32)
        )
        chart_support_counts = np.where(
            confirmed, support, 0
        ).astype(np.uint8)
        chart_consensus_contract = {
            "path": str(chart_consensus),
            "minimum_consensus_views": minimum_support,
            "confirmed_pixel_count": int(confirmed.sum()),
            "contradicted_supported_pixel_count": int(
                ((support >= minimum_support) & ~accepted).sum()
            ),
            "unsupported_chart_pixels_are_metric_targets": False,
        }
    mono_depths = (
        charts["prior_depths"].astype(np.float32) / scale_factor
        if "prior_depths" in charts.files
        else None
    )
    count = int(chart_depths.shape[0]) if chart_depths is not None else 0
    if count <= 0:
        raise RuntimeError("charts_data has no camera-indexed depth tensor")
    output.mkdir(parents=True, exist_ok=True)
    frame_summaries: list[dict[str, Any]] = []
    for index in range(count):
        plane_depth_path = plane_root / f"plane_depth_frame{index:06d}.npy"
        plane_mask_path = plane_root / f"plane_valid_mask_frame{index:06d}.npy"
        plane_confidence_path = plane_root / f"plane_confidence_frame{index:06d}.npy"
        plane_support_path = plane_root / f"plane_support_view_count_frame{index:06d}.npy"
        if not all(path.is_file() for path in (
            plane_depth_path, plane_mask_path, plane_confidence_path, plane_support_path,
        )):
            raise FileNotFoundError(
                "Plane residual source is incomplete; expected explicit plane_depth, "
                "plane_valid_mask, plane_confidence, and plane_support_view_count for frame "
                f"{index:06d}. Do not fuse a legacy mixed refine_depth map."
            )
        plane_depth = np.load(plane_depth_path).astype(np.float32, copy=False)
        plane_mask = np.load(plane_mask_path).astype(bool, copy=False)
        plane_confidence = np.load(plane_confidence_path).astype(np.float32, copy=False)
        plane_support = np.load(plane_support_path).astype(np.uint8, copy=False)
        fused = fuse_inverse_depth_sources(
            plane_depth,
            plane_confidence,
            chart_depths[index] if chart_depths is not None else None,
            chart_confs[index] if chart_confs is not None else None,
            mono_depths[index] if mono_depths is not None else None,
            plane_valid_mask=plane_mask,
            plane_support_view_count=plane_support,
            chart_support_view_count=(
                chart_support_counts[index]
                if chart_support_counts is not None
                else None
            ),
        )
        plane_mask_resized = _resize_bool(plane_mask, fused["depth"].shape)
        plane_bit = (fused["source_bitmask"] & np.uint8(1)) != 0
        if np.any(plane_bit & ~plane_mask_resized):
            raise RuntimeError(
                f"Plane source provenance escaped its accepted residual mask in frame {index}"
            )
        plane_support_resized = _resize_support_count(plane_support, fused["depth"].shape)
        if np.any(plane_mask_resized & (plane_support_resized == 0)):
            raise RuntimeError(
                f"Plane residual mask has zero independent support count in frame {index}"
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
                "accepted_plane_core_fraction": float(np.mean(plane_mask_resized)),
                "fused_plane_bit_fraction": float(np.mean(plane_bit)),
                "plane_support_view_count_values": [
                    int(value)
                    for value in np.unique(plane_support_resized[plane_mask_resized])
                ],
                "source_bitmask_values": [int(value) for value in np.unique(fused["source_bitmask"])],
            }
        )
    payload = {
        "schema_version": INVERSE_DEPTH_FUSION_VERSION,
        "mast3r_scene": str(mast3r_scene),
        "plane_root": str(plane_root),
        "source_priority": ["bounded_plane_residual_core", "aligned_chart", "calibrated_mono"],
        "fusion_space": "inverse_depth",
        "metric_coordinate_frame": "cambridge_fixed_camera_depth",
        "rho_units": "inverse_cambridge_world_unit",
        "plane_depth_input_units": "cambridge_fixed_camera_depth",
        "chart_depth_input_units": "matcha_normalized_camera_depth",
        "chart_to_cambridge_depth_multiplier": float(1.0 / scale_factor),
        "chart_crossview_consensus": chart_consensus_contract,
        "plane_source_contract": "explicit_residual_plane_depth_mask_confidence_support_v2",
        "support_view_count_contract": "maximum_proven_distinct_camera_support_not_source_family_count",
        "fixed_absolute_depth_limit": None,
        "frame_count": count,
        "frames": frame_summaries,
    }
    (output / "inverse_depth_fusion_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload
