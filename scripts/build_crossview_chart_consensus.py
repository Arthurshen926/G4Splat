#!/usr/bin/env python3
"""Build a conservative pixelwise cross-view depth-consensus sidecar.

The chart hard gate rejects a completely unreliable view, but an accepted
chart can still contain a local floater, an occlusion boundary, or a wrong
facade depth.  A scalar chart score cannot distinguish those pixels.  This
tool reprojects *neighbouring* hard-gated pointmaps into every chart camera,
z-buffers them, and uses a multi-view median only where independently
projected support exists.  Unsupported pixels deliberately fall back to the
chart's own depth: lack of a neighbour must never become a coverage hole.

The emitted ``depths`` array can be consumed by ``train_with_charts.py`` as a
bounded direct chart-prior correction, while ``consistency_weights`` supports
a separate trust-only ablation.  Neither route changes RGB supervision or the
initialized Gaussian set.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any
import warnings

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
# This script depends on the G4Splat pose helpers.  Put its root ahead of an
# optional MAtCha checkout in PYTHONPATH; both repositories expose a
# ``view_quality_control`` package but only the former owns ``poses.py``.
for path in (str(REPO_ROOT), str(REPO_ROOT / "mast3r")):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

from colmap.read_write_model import read_cameras_binary, read_images_binary  # noqa: E402
from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402
from view_quality_control.poses import qvec_to_rotmat  # noqa: E402


def _canonical_name(value: str) -> str:
    return Path(str(value).replace("\\", "/")).stem


def _intrinsics(camera: Any, width: int, height: int) -> tuple[float, float, float, float]:
    params = np.asarray(camera.params, dtype=np.float64)
    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        fx = fy = params[0]
        cx, cy = params[1:3]
    elif camera.model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        fx, fy, cx, cy = params[:4]
    else:
        raise RuntimeError(f"Unsupported COLMAP camera model: {camera.model}")
    return (
        float(fx * width / camera.width),
        float(fy * height / camera.height),
        float(cx * width / camera.width),
        float(cy * height / camera.height),
    )


def _active_indices(
    gate_report: Path,
    image_names: list[str],
    chart_selection_json: Path | None = None,
) -> np.ndarray:
    """Return Charts that are both hard-gated and, optionally, selected.

    A quality-aware subset must constrain every consumer of cross-view
    evidence, not just Gaussian initialization.  In particular, allowing an
    excluded Chart to remain a consensus *neighbour* would leak its geometry
    back into otherwise selected Chart priors.  The output arrays still keep
    their original chart-axis shape, but only the returned indices may source
    or receive consensus evidence.
    """
    payload = json.loads(gate_report.read_text())
    accepted = {
        _canonical_name(record["image_name"])
        for record in payload.get("records", [])
        if not record.get("rejected", False)
    }
    names = {_canonical_name(name) for name in image_names}
    unknown = accepted - names
    if unknown:
        raise RuntimeError(
            "Hard-gate report contains chart(s) absent from COLMAP: "
            + ", ".join(sorted(unknown)[:5])
        )
    if chart_selection_json is not None:
        selection_payload = json.loads(chart_selection_json.read_text())
        selected_values = selection_payload.get(
            "selected_image_names", selection_payload.get("image_names")
        )
        if not isinstance(selected_values, list) or not selected_values:
            raise RuntimeError(
                f"Chart selection {chart_selection_json} has no non-empty image_names list"
            )
        selected = {_canonical_name(str(value)) for value in selected_values}
        unknown_selected = selected - names
        if unknown_selected:
            raise RuntimeError(
                "Chart selection contains chart(s) absent from COLMAP: "
                + ", ".join(sorted(unknown_selected)[:5])
            )
        rejected_selected = selected - accepted
        if rejected_selected:
            raise RuntimeError(
                "Chart selection contains hard-gate rejected chart(s): "
                + ", ".join(sorted(rejected_selected)[:5])
            )
        accepted &= selected
    result = np.asarray(
        [index for index, name in enumerate(image_names) if _canonical_name(name) in accepted],
        dtype=np.int64,
    )
    if len(result) < 2:
        raise RuntimeError("Cross-view consensus requires at least two enabled charts")
    return result


def _neighbour_indices(
    rotations: np.ndarray,
    translations: np.ndarray,
    active: np.ndarray,
    count: int,
    max_distance: float,
) -> dict[int, np.ndarray]:
    """Select nearby camera poses in joint position/orientation space."""
    centers = np.stack(
        [-rotations[index].T @ translations[index] for index in range(len(rotations))]
    )
    forward = np.stack([rotations[index].T[:, 2] for index in range(len(rotations))])
    center_active = centers[active]
    extent = max(float(np.linalg.norm(center_active.max(0) - center_active.min(0))), 1e-8)
    features = np.concatenate(
        [(centers - center_active.mean(0)) / extent, 0.35 * forward], axis=1
    )
    result: dict[int, np.ndarray] = {}
    for source in active:
        candidates = active[active != source]
        distances = np.linalg.norm(features[candidates] - features[source][None], axis=1)
        order = np.argsort(distances, kind="stable")
        selected = candidates[order]
        if math.isfinite(max_distance):
            selected = selected[distances[order] <= max_distance]
        result[int(source)] = selected[:count]
    return result


def select_overlap_neighbours(
    candidates: np.ndarray,
    overlap_fractions: np.ndarray,
    count: int,
    min_overlap: float,
) -> np.ndarray:
    """Choose deterministic peers by actual target-image support.

    Pose distance is only a proxy for co-visibility.  This small pure helper
    makes the ranking explicit and testable: candidates below the requested
    overlap are not silently substituted by a distant view.
    """
    if candidates.ndim != 1 or overlap_fractions.ndim != 1:
        raise ValueError("candidates and overlap_fractions must be one-dimensional")
    if len(candidates) != len(overlap_fractions):
        raise ValueError("candidates and overlap_fractions must have equal length")
    if count < 1:
        raise ValueError("count must be positive")
    eligible = np.isfinite(overlap_fractions) & (overlap_fractions >= min_overlap)
    candidates = candidates[eligible]
    overlap_fractions = overlap_fractions[eligible]
    if len(candidates) == 0:
        return np.empty(0, dtype=np.int64)
    # Stable id tie-breaking makes the sidecar reproducible across NumPy
    # versions and avoids an accidental dependence on hash iteration order.
    order = np.lexsort((candidates, -overlap_fractions))
    return candidates[order[:count]].astype(np.int64, copy=False)


def _overlap_neighbour_indices(
    points: np.ndarray,
    valid: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    intrinsics: list[tuple[float, float, float, float]],
    active: np.ndarray,
    count: int,
    max_distance: float,
    min_overlap: float,
) -> tuple[dict[int, np.ndarray], dict[int, dict[int, float]]]:
    """Pick peers by measured reprojection overlap rather than pose proxy.

    A candidate peer contributes only where its valid pointmap z-buffers into
    a valid source-chart pixel.  This is deliberately the same visibility
    operation used later for the consensus target, so a selected neighbour is
    guaranteed to carry usable evidence for that target chart.
    """
    centers = np.stack(
        [-rotations[index].T @ translations[index] for index in range(len(rotations))]
    )
    forward = np.stack([rotations[index].T[:, 2] for index in range(len(rotations))])
    center_active = centers[active]
    extent = max(float(np.linalg.norm(center_active.max(0) - center_active.min(0))), 1e-8)
    features = np.concatenate(
        [(centers - center_active.mean(0)) / extent, 0.35 * forward], axis=1
    )
    selected: dict[int, np.ndarray] = {}
    selected_overlaps: dict[int, dict[int, float]] = {}
    for source in active:
        candidates = active[active != source]
        distances = np.linalg.norm(features[candidates] - features[source][None], axis=1)
        if math.isfinite(max_distance):
            candidates = candidates[distances <= max_distance]
        target_valid = valid[int(source)]
        target_count = max(int(target_valid.sum()), 1)
        fractions = np.empty(len(candidates), dtype=np.float32)
        for offset, peer in enumerate(candidates):
            projected = _zbuffer_projected_depth(
                points[int(peer)],
                valid[int(peer)],
                rotations[int(source)],
                translations[int(source)],
                intrinsics[int(source)],
            )
            fractions[offset] = float(
                (np.isfinite(projected) & target_valid).sum() / target_count
            )
        peers = select_overlap_neighbours(candidates, fractions, count, min_overlap)
        selected[int(source)] = peers
        selected_overlaps[int(source)] = {
            int(peer): float(fractions[np.flatnonzero(candidates == peer)[0]])
            for peer in peers
        }
    return selected, selected_overlaps


def _zbuffer_projected_depth(
    points: np.ndarray,
    valid: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    intrinsics: tuple[float, float, float, float],
) -> np.ndarray:
    """Project one source pointmap into a target chart and retain nearest depth."""
    height, width = valid.shape
    source_points = points[valid]
    if len(source_points) == 0:
        return np.full((height, width), np.nan, dtype=np.float32)
    camera_points = source_points @ rotation.T + translation[None]
    z = camera_points[:, 2]
    fx, fy, cx, cy = intrinsics
    safe_z = np.maximum(z, 1e-8)
    x = np.rint(fx * camera_points[:, 0] / safe_z + cx).astype(np.int64)
    y = np.rint(fy * camera_points[:, 1] / safe_z + cy).astype(np.int64)
    inside = (z > 0.0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    flat = np.full(height * width, np.inf, dtype=np.float32)
    if np.any(inside):
        np.minimum.at(flat, y[inside] * width + x[inside], z[inside].astype(np.float32))
    flat[~np.isfinite(flat)] = np.nan
    return flat.reshape(height, width)


def consensus_candidate_depth(
    source_depth: np.ndarray,
    source_valid: np.ndarray,
    neighbour_depths: np.ndarray,
    min_consensus_views: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fallback-safe consensus depth, support count, and relative error."""
    if neighbour_depths.ndim != 3:
        raise ValueError("neighbour_depths must have shape (K,H,W)")
    if neighbour_depths.shape[1:] != source_depth.shape:
        raise ValueError("neighbour_depths and source_depth spatial shapes differ")
    support_count = np.isfinite(neighbour_depths).sum(axis=0).astype(np.uint8)
    # Nanmedian is safe here because unsupported pixels fall back immediately.
    # Suppress only its expected all-NaN warning for pixels no peer observes.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        median = np.nanmedian(neighbour_depths, axis=0)
    consensus = source_depth.astype(np.float32, copy=True)
    eligible = source_valid & (support_count >= min_consensus_views) & np.isfinite(median)
    consensus[eligible] = median[eligible].astype(np.float32, copy=False)
    relative_error = np.full(source_depth.shape, np.nan, dtype=np.float32)
    denominator = np.maximum(np.abs(median), 1e-6)
    relative_error[eligible] = np.abs(source_depth[eligible] - median[eligible]) / denominator[eligible]
    return consensus, support_count, relative_error


def gate_consensus_by_relative_error(
    source_depth: np.ndarray,
    consensus_depth: np.ndarray,
    relative_error: np.ndarray,
    max_relative_error: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only geometrically self-consistent consensus corrections.

    Reprojection consensus can otherwise cross an occlusion boundary: three
    peers may agree on a different surface than the source-chart pixel.  The
    raw residual is retained for audit, but a correction that exceeds the
    configured relative-depth bound falls back exactly to the source depth.
    """
    if source_depth.shape != consensus_depth.shape or source_depth.shape != relative_error.shape:
        raise ValueError("source, consensus, and relative-error shapes must match")
    if max_relative_error < 0.0:
        raise ValueError("max_relative_error must be non-negative")
    eligible = (
        np.isfinite(source_depth)
        & np.isfinite(consensus_depth)
        & np.isfinite(relative_error)
        & (source_depth > 0.0)
        & (consensus_depth > 0.0)
        & (relative_error <= max_relative_error)
    )
    gated = source_depth.astype(np.float32, copy=True)
    gated[eligible] = consensus_depth[eligible].astype(np.float32, copy=False)
    return gated, eligible


def consistency_weights_from_relative_error(
    relative_error: np.ndarray,
    support_count: np.ndarray,
    source_valid: np.ndarray,
    min_consensus_views: int,
    floor: float,
    sigma: float,
) -> np.ndarray:
    """Convert supported geometric disagreement into an explicit soft weight.

    Unsupported pixels are exactly one.  A caller may separately attenuate
    *supported but rejected* pixels: those have independent cross-view
    evidence of a disagreement, unlike pixels for which no peer observed the
    source surface at all.
    """
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must lie in [0, 1]")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    weights = np.ones(relative_error.shape, dtype=np.float32)
    eligible = (
        source_valid
        & (support_count >= min_consensus_views)
        & np.isfinite(relative_error)
    )
    disagreement = relative_error[eligible].astype(np.float32, copy=False)
    weights[eligible] = floor + (1.0 - floor) * np.exp(
        -(disagreement * disagreement) / (2.0 * sigma * sigma)
    )
    return weights


def apply_rejected_supported_weight(
    weights: np.ndarray,
    correction_mask: np.ndarray,
    support_count: np.ndarray,
    source_valid: np.ndarray,
    min_consensus_views: int,
    rejected_supported_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Optionally attenuate only cross-view-disagreed source pixels.

    A rejected direct correction still retains the source depth for safety.
    This helper controls whether its *prior-loss weight* stays legacy-neutral
    or is attenuated because independent peers observed a disagreement.
    """
    if not 0.0 <= rejected_supported_weight <= 1.0:
        raise ValueError("rejected_supported_weight must lie in [0, 1]")
    if not (
        weights.shape == correction_mask.shape == support_count.shape == source_valid.shape
    ):
        raise ValueError("weights, masks, and support_count must share a shape")
    adjusted = weights.astype(np.float32, copy=True)
    supported = source_valid & (support_count >= min_consensus_views)
    rejected_supported = supported & ~correction_mask
    if rejected_supported_weight == 1.0:
        # Preserve the exact legacy behaviour, including a neutral weight for
        # unsupported fallback pixels.
        adjusted[~correction_mask] = 1.0
    else:
        adjusted[rejected_supported] = rejected_supported_weight
    return adjusted, rejected_supported


def _make_montage(
    image_names: list[str],
    active: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    output: Path,
    *,
    kind: str,
    columns: int = 5,
) -> None:
    """Write a compact chart diagnostic without retaining full render buffers."""
    tiles: list[np.ndarray] = []
    for index in active:
        value = values[index]
        if kind == "support":
            scaled = np.clip(value.astype(np.float32) / max(float(np.nanmax(values)), 1.0), 0.0, 1.0)
            image = cv2.applyColorMap(np.rint(scaled * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            label = "peer support"
        else:
            scaled = np.nan_to_num(value.astype(np.float32), nan=0.0) / 0.25
            image = cv2.applyColorMap(np.rint(np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            label = "relative error"
        image[~valid[index]] = 0
        image = cv2.resize(image, (192, 108), interpolation=cv2.INTER_NEAREST)
        cv2.putText(
            image,
            f"{_canonical_name(image_names[index])}",
            (4, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(image, label, (4, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(image)
    rows = []
    blank = np.zeros_like(tiles[0])
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        row += [blank] * (columns - len(row))
        rows.append(np.concatenate(row, axis=1))
    cv2.imwrite(str(output), np.concatenate(rows, axis=0))


def build_consensus(args: argparse.Namespace) -> dict[str, Any]:
    scene_path = args.scene_path.resolve()
    chart_path = scene_path / "charts_data.npz"
    gate_path = args.gate_report.resolve()
    if not chart_path.exists():
        raise FileNotFoundError(chart_path)
    if not gate_path.exists():
        raise FileNotFoundError(gate_path)
    charts = np.load(chart_path)
    required = {"pts", "depths", "confs", "scale_factor"}
    missing = required - set(charts.files)
    if missing:
        raise RuntimeError(f"{chart_path} lacks required arrays: {sorted(missing)}")
    points = charts["pts"].astype(np.float32, copy=False)
    depths = charts["depths"].astype(np.float32, copy=False)
    confs = charts["confs"].astype(np.float32, copy=False)
    if points.ndim != 4 or depths.shape != points.shape[:3] or confs.shape != depths.shape:
        raise RuntimeError("charts_data point/depth/confidence shapes are inconsistent")
    n_charts, height, width = depths.shape

    # ``all-sparse`` holds the full dense Cambridge trajectory, whereas the
    # chart pointmaps are indexed by the compact chart reconstruction.
    sparse_dir = scene_path / "sparse" / "0"
    cameras = read_cameras_binary(str(sparse_dir / "cameras.bin"))
    colmap_images = sorted(read_images_binary(str(sparse_dir / "images.bin")).values(), key=lambda image: image.id)
    if len(colmap_images) != n_charts:
        raise RuntimeError(f"COLMAP has {len(colmap_images)} charts but NPZ has {n_charts}")
    image_names = [str(image.name) for image in colmap_images]
    hard_gated = _active_indices(gate_path, image_names)
    active = _active_indices(gate_path, image_names, args.chart_selection_json)
    scale_factor = float(charts["scale_factor"])
    rotations = np.stack([qvec_to_rotmat(image.qvec) for image in colmap_images]).astype(np.float32)
    translations = np.stack([image.tvec * scale_factor for image in colmap_images]).astype(np.float32)
    intrinsics = [_intrinsics(cameras[image.camera_id], width, height) for image in colmap_images]

    static_masks = np.ones((n_charts, height, width), dtype=bool)
    if args.mask_pickle is not None:
        if args.mask_dataset_path is None:
            raise ValueError("--mask-dataset-path is required with --mask-pickle")
        lookup = CambridgeMaskLookup(args.mask_dataset_path, args.mask_pickle, args.mask_indices)
        static_masks = np.stack(
            [
                lookup.get_mask(name, (height, width), torch.device("cpu")).numpy()
                for name in image_names
            ]
        )
    base_valid = (
        static_masks
        & np.isfinite(points).all(axis=3)
        & np.isfinite(depths)
        & np.isfinite(confs)
        & (confs > 0.0)
        & (depths > 0.0)
    )
    active_mask = np.zeros(n_charts, dtype=bool)
    active_mask[active] = True
    base_valid[~active_mask] = False
    selected_overlaps: dict[int, dict[int, float]] | None = None
    if args.neighbor_selection == "pose":
        neighbors = _neighbour_indices(
            rotations,
            translations,
            active,
            args.neighbors,
            args.max_neighbor_pose_distance,
        )
    else:
        neighbors, selected_overlaps = _overlap_neighbour_indices(
            points,
            base_valid,
            rotations,
            translations,
            intrinsics,
            active,
            args.neighbors,
            args.max_neighbor_pose_distance,
            args.min_neighbor_overlap,
        )

    candidate_depths = depths.astype(np.float32, copy=True)
    support_counts = np.zeros_like(depths, dtype=np.uint8)
    relative_errors = np.full_like(depths, np.nan, dtype=np.float32)
    consistency_weights = np.ones_like(depths, dtype=np.float32)
    correction_masks = np.zeros_like(depths, dtype=bool)
    records: list[dict[str, Any]] = []
    for offset, source in enumerate(active, start=1):
        source_neighbours = neighbors[int(source)]
        maps = [
            _zbuffer_projected_depth(
                points[int(peer)],
                base_valid[int(peer)],
                rotations[int(source)],
                translations[int(source)],
                intrinsics[int(source)],
            )
            for peer in source_neighbours
        ]
        if maps:
            neighbour_depths = np.stack(maps)
        else:
            neighbour_depths = np.empty((0, height, width), dtype=np.float32)
        consensus, support, relative_error = consensus_candidate_depth(
            depths[int(source)],
            base_valid[int(source)],
            neighbour_depths,
            args.min_consensus_views,
        )
        consensus, correction_mask = gate_consensus_by_relative_error(
            depths[int(source)],
            consensus,
            relative_error,
            args.max_relative_error,
        )
        candidate_depths[int(source)] = consensus
        support_counts[int(source)] = support
        relative_errors[int(source)] = relative_error
        weights = consistency_weights_from_relative_error(
            relative_error,
            support,
            base_valid[int(source)],
            args.min_consensus_views,
            args.weight_floor,
            args.relative_error_sigma,
        )
        # A rejected direct correction deliberately falls back to the source
        # depth, but that does not make the source trustworthy: this pixel was
        # independently observed and disagreed with the peer consensus.  Keep
        # the legacy-neutral behaviour by default, while allowing a conservative
        # attenuation for exactly these ambiguous supported pixels.  Unsupported
        # pixels remain neutral because they provide no such evidence.
        weights, rejected_supported = apply_rejected_supported_weight(
            weights,
            correction_mask,
            support,
            base_valid[int(source)],
            args.min_consensus_views,
            args.rejected_supported_weight,
        )
        consistency_weights[int(source)] = weights
        correction_masks[int(source)] = correction_mask
        valid = base_valid[int(source)]
        supported = valid & (support >= args.min_consensus_views)
        corrected = correction_mask & (np.abs(consensus - depths[int(source)]) > 1e-6)
        errors = relative_error[supported]
        records.append(
            {
                "index": int(source),
                "image_name": image_names[int(source)],
                "neighbors": [image_names[int(peer)] for peer in source_neighbours],
                "neighbor_overlap_fractions": (
                    [
                        selected_overlaps[int(source)][int(peer)]
                        for peer in source_neighbours
                    ]
                    if selected_overlaps is not None
                    else None
                ),
                "valid_fraction": float(valid.mean()),
                "supported_fraction_of_valid": float(supported.sum() / max(int(valid.sum()), 1)),
                "supported_pixel_count": int(supported.sum()),
                "rejected_supported_fraction_of_valid": float(
                    rejected_supported.sum() / max(int(valid.sum()), 1)
                ),
                "rejected_supported_pixel_count": int(rejected_supported.sum()),
                "accepted_fraction_of_valid": float(correction_mask.sum() / max(int(valid.sum()), 1)),
                "accepted_pixel_count": int(correction_mask.sum()),
                "corrected_fraction_of_valid": float(corrected.sum() / max(int(valid.sum()), 1)),
                "corrected_pixel_count": int(corrected.sum()),
                "relative_error_median": float(np.nanmedian(errors)) if len(errors) else None,
                "relative_error_p90": float(np.nanpercentile(errors, 90)) if len(errors) else None,
                "relative_error_gt25_fraction": (
                    float(np.mean(errors > 0.25)) if len(errors) else None
                ),
            }
        )
        print(
            f"[{offset:02d}/{len(active):02d}] {image_names[int(source)]}: "
            f"support={records[-1]['supported_fraction_of_valid'] * 100:.1f}% "
            f"accepted={records[-1]['accepted_fraction_of_valid'] * 100:.1f}% "
            f"rejected={records[-1]['rejected_supported_fraction_of_valid'] * 100:.1f}% "
            f"p90={records[-1]['relative_error_p90']}"
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        depths=candidate_depths,
        support_counts=support_counts,
        relative_errors=relative_errors,
        consistency_weights=consistency_weights,
        correction_mask=correction_masks,
        image_names=np.asarray(image_names),
    )
    summary = {
        "method": "hard_gated_selection_filtered_neighbour_reprojection_zbuffer_median",
        "scene_path": str(scene_path),
        "gate_report": str(gate_path),
        "chart_selection_json": (
            str(args.chart_selection_json.resolve())
            if args.chart_selection_json is not None
            else None
        ),
        "output": str(output),
        "chart_shape": [int(n_charts), int(height), int(width)],
        "hard_gated_chart_count": int(len(hard_gated)),
        "active_chart_count": int(len(active)),
        "neighbors_requested": int(args.neighbors),
        "neighbor_selection": args.neighbor_selection,
        "min_neighbor_overlap": (
            float(args.min_neighbor_overlap)
            if args.neighbor_selection == "overlap"
            else None
        ),
        "min_consensus_views": int(args.min_consensus_views),
        "max_relative_error": (
            float(args.max_relative_error)
            if math.isfinite(args.max_relative_error)
            else None
        ),
        "weight_floor": float(args.weight_floor),
        "relative_error_sigma": float(args.relative_error_sigma),
        "rejected_supported_weight": float(args.rejected_supported_weight),
        "max_neighbor_pose_distance": (
            float(args.max_neighbor_pose_distance)
            if math.isfinite(args.max_neighbor_pose_distance)
            else None
        ),
        "mask_indices": args.mask_indices if args.mask_pickle is not None else None,
        "records": records,
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    if args.diagnostic_dir is not None:
        diagnostic_dir = args.diagnostic_dir.resolve()
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        _make_montage(
            image_names,
            active,
            support_counts,
            base_valid,
            diagnostic_dir / "crossview_support_montage.jpg",
            kind="support",
        )
        _make_montage(
            image_names,
            active,
            relative_errors,
            base_valid,
            diagnostic_dir / "crossview_relative_error_montage.jpg",
            kind="error",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-path", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument(
        "--chart-selection-json",
        type=Path,
        default=None,
        help=(
            "Optional strict quality/joint selection. Enabled Charts are the "
            "intersection of this list and the hard gate; excluded Charts "
            "cannot contribute consensus-neighbour evidence."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-pickle", type=Path, default=None)
    parser.add_argument("--mask-dataset-path", type=Path, default=None)
    parser.add_argument("--mask-indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument(
        "--neighbor-selection",
        choices=("pose", "overlap"),
        default="pose",
        help=(
            "Use pose proximity (legacy) or actual reprojected valid-pixel overlap "
            "to choose peer charts."
        ),
    )
    parser.add_argument(
        "--min-neighbor-overlap",
        type=float,
        default=0.05,
        help="Minimum valid-source-pixel fraction required for an overlap-selected peer.",
    )
    parser.add_argument("--min-consensus-views", type=int, default=2)
    parser.add_argument(
        "--weight-floor",
        type=float,
        default=0.65,
        help="Minimum prior multiplier at a strongly inconsistent supported pixel.",
    )
    parser.add_argument(
        "--relative-error-sigma",
        type=float,
        default=0.25,
        help="Relative depth error at which the soft consistency multiplier decays.",
    )
    parser.add_argument(
        "--rejected-supported-weight",
        type=float,
        default=1.0,
        help=(
            "Prior multiplier for a source pixel with enough reprojection support "
            "whose direct consensus correction was rejected.  The legacy-neutral "
            "default 1.0 preserves prior behaviour; values below one attenuate only "
            "cross-view-disagreed pixels, never unsupported pixels."
        ),
    )
    parser.add_argument(
        "--max-relative-error",
        type=float,
        default=float("inf"),
        help=(
            "Reject a direct consensus depth correction when its source-to-median "
            "relative error exceeds this bound; infinity preserves legacy behavior."
        ),
    )
    parser.add_argument("--max-neighbor-pose-distance", type=float, default=float("inf"))
    parser.add_argument("--diagnostic-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.neighbors < 1:
        raise ValueError("--neighbors must be positive")
    if not 0.0 <= args.min_neighbor_overlap <= 1.0:
        raise ValueError("--min-neighbor-overlap must lie in [0, 1]")
    if args.min_consensus_views < 1 or args.min_consensus_views > args.neighbors:
        raise ValueError("--min-consensus-views must be in [1, --neighbors]")
    if not 0.0 <= args.weight_floor <= 1.0:
        raise ValueError("--weight-floor must lie in [0, 1]")
    if args.relative_error_sigma <= 0.0:
        raise ValueError("--relative-error-sigma must be positive")
    if not 0.0 <= args.rejected_supported_weight <= 1.0:
        raise ValueError("--rejected-supported-weight must lie in [0, 1]")
    if args.max_relative_error < 0.0:
        raise ValueError("--max-relative-error must be non-negative")
    build_consensus(args)


if __name__ == "__main__":
    main()
