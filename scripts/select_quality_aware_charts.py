#!/usr/bin/env python3
"""Select a geometrically valid chart subset with quality and 3D coverage.

The hard chart gate answers a binary question: can a chart participate in
geometry?  It intentionally does not answer the appearance question.  This
tool operates *after* that gate and makes the latter explicit:

* sharpness/exposure/tracks provide a per-image appearance prior;
* aligned pointmaps are reprojected into geometrically overlapping chart
  views to measure photometric agreement, depth support, and normal
  stability;
* every candidate contributes a set of world-space voxels, so marginal
  coverage is measured in 3D rather than by copying a 2D mask between views;
* a constrained greedy pass plus local swaps keeps the original pose-cluster
  support and baseline guarantees while optimizing the joint objective.

The output never changes ``charts_data.npz``.  It is an initialization subset
and a reliability sidecar; the full hard-gated chart set remains available for
geometry priors downstream.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Optional

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "mast3r") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "mast3r"))

from colmap.read_write_model import read_cameras_binary, read_images_binary  # noqa: E402
from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402
from scripts.select_chart_views import (  # noqa: E402
    _farthest_point_indices,
    build_pose_geometry,
    load_scene_poses,
    normalize_required_name,
)
from scripts.audit_gated_chart_coverage import archived_reference_features  # noqa: E402
from view_quality_control.poses import qvec_to_rotmat  # noqa: E402


def _name(value: str) -> str:
    return normalize_required_name(Path(str(value)).name)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _rank01(values: Iterable[float], higher_is_better: bool = True) -> np.ndarray:
    """Stable [0, 1] ranks; robust to heterogeneous input units."""
    array = np.asarray(list(values), dtype=np.float64)
    finite = np.isfinite(array)
    result = np.full(len(array), 0.0, dtype=np.float64)
    if not finite.any():
        return result
    finite_values = array[finite]
    if np.ptp(finite_values) <= 1e-12:
        result[finite] = 0.5
        return result
    order = np.argsort(finite_values, kind="stable")
    ranks = np.empty(len(finite_values), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(finite_values))
    if not higher_is_better:
        ranks = 1.0 - ranks
    result[finite] = ranks
    return result


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _read_view_audit(path: Optional[Path]) -> dict[str, dict[str, float]]:
    if path is None or not path.exists():
        return {}
    output: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_name = row.get("image_name")
            if not raw_name:
                continue
            output[_name(raw_name)] = {
                key: _safe_float(value)
                for key, value in row.items()
                if key not in {"image_name", "source_name", "sequence", "status"}
            }
    return output


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


def _read_chart_rgb(image_path: Path, width: int, height: int) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read chart image {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def _world_normals(
    points: np.ndarray,
    valid: np.ndarray,
    camera_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate view-facing world normals from an aligned point map."""
    horizontal = np.zeros_like(points, dtype=np.float32)
    vertical = np.zeros_like(points, dtype=np.float32)
    horizontal[:, 1:-1] = points[:, 2:] - points[:, :-2]
    horizontal[:, 0] = points[:, 1] - points[:, 0]
    horizontal[:, -1] = points[:, -1] - points[:, -2]
    vertical[1:-1] = points[2:] - points[:-2]
    vertical[0] = points[1] - points[0]
    vertical[-1] = points[-1] - points[-2]
    normals = np.cross(horizontal, vertical)
    norms = np.linalg.norm(normals, axis=2)
    normal_valid = valid & np.isfinite(norms) & (norms > 1e-8)
    normals[normal_valid] /= norms[normal_valid][:, None]
    normals[~normal_valid] = 0.0
    facing = np.sum(normals * (camera_center[None, None] - points), axis=2)
    normals[facing < 0.0] *= -1.0
    return normals.astype(np.float32, copy=False), normal_valid


def _affine_photometric_score(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return illumination-compensated RGB agreement and raw MAE."""
    if len(source) < 16:
        return 0.0, 1.0
    transformed = np.empty_like(target)
    for channel in range(3):
        x = target[:, channel]
        y = source[:, channel]
        x_centered = x - np.median(x)
        y_centered = y - np.median(y)
        denom = float(np.dot(x_centered, x_centered))
        slope = float(np.dot(x_centered, y_centered) / denom) if denom > 1e-8 else 1.0
        slope = float(np.clip(slope, 0.5, 2.0))
        intercept = float(np.median(y - slope * x))
        transformed[:, channel] = np.clip(slope * x + intercept, 0.0, 1.0)
    mae = float(np.mean(np.abs(source - transformed)))
    return float(np.exp(-mae / 0.10)), mae


def select_overlap_neighbours(
    candidates: np.ndarray,
    overlap_fractions: np.ndarray,
    count: int,
    min_overlap: float,
) -> np.ndarray:
    """Choose peer Charts by measured co-visibility with deterministic ties.

    Pose proximity is useful as a speed heuristic but is not evidence that two
    facade pixels observe the same surface.  In particular, Cambridge camera
    trajectories regularly pass occluders at nearly identical poses.  Do not
    backfill a missing high-overlap neighbour with a pose-near chart: a
    smaller valid peer set is safer than contaminated cross-view quality
    measurements.
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
    order = np.lexsort((candidates, -overlap_fractions))
    return candidates[order[:count]].astype(np.int64, copy=False)


def _projection_overlap_fraction(
    source_index: int,
    target_index: int,
    *,
    points: np.ndarray,
    valid_masks: np.ndarray,
    normal_valid: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    intrinsics: list[tuple[float, float, float, float]],
    stride: int,
) -> float:
    """Measure target-valid co-visibility without assuming depth agreement.

    This deliberately does *not* gate by relative depth: geometric agreement
    is what the subsequent quality metric should measure.  Requiring it at
    neighbour-selection time would hide a bad Chart by denying it all peers.
    """
    source_valid = valid_masks[source_index] & normal_valid[source_index]
    sampled = np.zeros_like(source_valid)
    sampled[::stride, ::stride] = True
    source_valid &= sampled
    source_y, source_x = np.nonzero(source_valid)
    source_count = len(source_y)
    if source_count == 0:
        return 0.0
    source_points = points[source_index, source_y, source_x]
    target_camera_points = (
        source_points @ rotations[target_index].T + translations[target_index][None]
    )
    z = target_camera_points[:, 2]
    fx, fy, cx, cy = intrinsics[target_index]
    safe_z = np.maximum(z, 1e-8)
    target_x = np.rint(fx * target_camera_points[:, 0] / safe_z + cx).astype(np.int64)
    target_y = np.rint(fy * target_camera_points[:, 1] / safe_z + cy).astype(np.int64)
    height, width = valid_masks.shape[-2:]
    inside = (
        (z > 0.0)
        & (target_x >= 0)
        & (target_x < width)
        & (target_y >= 0)
        & (target_y < height)
    )
    if not np.any(inside):
        return 0.0
    rows, cols = target_y[inside], target_x[inside]
    target_valid = valid_masks[target_index, rows, cols] & normal_valid[target_index, rows, cols]
    return float(target_valid.sum() / source_count)


def _pair_consistency(
    source_index: int,
    target_index: int,
    *,
    points: np.ndarray,
    depths: np.ndarray,
    valid_masks: np.ndarray,
    normals: np.ndarray,
    normal_valid: np.ndarray,
    rgbs: list[np.ndarray],
    rotations: np.ndarray,
    translations: np.ndarray,
    intrinsics: list[tuple[float, float, float, float]],
    stride: int,
    depth_threshold: float,
) -> dict[str, float]:
    """Reproject one chart into another and compare only depth-visible pixels."""
    source_valid = valid_masks[source_index] & normal_valid[source_index]
    sampled = np.zeros_like(source_valid)
    sampled[::stride, ::stride] = True
    source_valid &= sampled
    source_y, source_x = np.nonzero(source_valid)
    source_count = int(len(source_y))
    if source_count == 0:
        return {
            "source_index": source_index,
            "target_index": target_index,
            "sample_count": 0,
            "support_fraction": 0.0,
            "depth_relative_median": 1.0,
            "depth_score": 0.0,
            "photo_score": 0.0,
            "photo_mae": 1.0,
            "normal_score": 0.0,
            "pair_score": 0.0,
        }

    source_points = points[source_index, source_y, source_x]
    target_camera_points = (
        source_points @ rotations[target_index].T + translations[target_index][None]
    )
    z = target_camera_points[:, 2]
    fx, fy, cx, cy = intrinsics[target_index]
    safe_z = np.maximum(z, 1e-8)
    target_x = np.rint(fx * target_camera_points[:, 0] / safe_z + cx).astype(np.int64)
    target_y = np.rint(fy * target_camera_points[:, 1] / safe_z + cy).astype(np.int64)
    height, width = depths.shape[-2:]
    inside = (
        (z > 0.0)
        & (target_x >= 0)
        & (target_x < width)
        & (target_y >= 0)
        & (target_y < height)
    )
    accepted = np.zeros(source_count, dtype=bool)
    relative = np.full(source_count, np.inf, dtype=np.float32)
    if np.any(inside):
        rows, cols = target_y[inside], target_x[inside]
        target_depth = depths[target_index, rows, cols]
        target_ok = valid_masks[target_index, rows, cols] & normal_valid[target_index, rows, cols]
        current_relative = np.abs(z[inside] - target_depth) / np.maximum(np.abs(target_depth), 1e-6)
        relative[inside] = current_relative
        accepted[inside] = target_ok & np.isfinite(current_relative) & (current_relative <= depth_threshold)

    support_fraction = float(accepted.mean())
    if not np.any(accepted):
        return {
            "source_index": source_index,
            "target_index": target_index,
            "sample_count": source_count,
            "support_fraction": support_fraction,
            "depth_relative_median": 1.0,
            "depth_score": 0.0,
            "photo_score": 0.0,
            "photo_mae": 1.0,
            "normal_score": 0.0,
            "pair_score": 0.0,
        }

    source_keep = np.flatnonzero(accepted)
    source_rgb = rgbs[source_index][source_y[source_keep], source_x[source_keep]]
    target_rgb = rgbs[target_index][target_y[source_keep], target_x[source_keep]]
    photo_score, photo_mae = _affine_photometric_score(source_rgb, target_rgb)
    source_normal = normals[source_index, source_y[source_keep], source_x[source_keep]]
    target_normal = normals[target_index, target_y[source_keep], target_x[source_keep]]
    normal_score = float(
        np.mean(np.clip(np.abs(np.sum(source_normal * target_normal, axis=1)), 0.0, 1.0))
    )
    depth_relative_median = float(np.median(relative[source_keep]))
    depth_score = float(np.exp(-depth_relative_median / max(depth_threshold, 1e-6)))
    pair_score = support_fraction * (
        0.45 * photo_score + 0.30 * normal_score + 0.25 * depth_score
    )
    return {
        "source_index": source_index,
        "target_index": target_index,
        "sample_count": source_count,
        "support_fraction": support_fraction,
        "depth_relative_median": depth_relative_median,
        "depth_score": depth_score,
        "photo_score": photo_score,
        "photo_mae": photo_mae,
        "normal_score": normal_score,
        "pair_score": float(pair_score),
    }


def _voxel_coverage(
    points: np.ndarray,
    valid_masks: np.ndarray,
    stride: int,
    bins: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Return per-chart compact voxel ids and inverse-frequency cell weights."""
    samples: list[np.ndarray] = []
    for index in range(len(points)):
        mask = valid_masks[index].copy()
        sampled = np.zeros_like(mask)
        sampled[::stride, ::stride] = True
        value = points[index][mask & sampled]
        samples.append(value)
    nonempty = [value for value in samples if len(value)]
    if not nonempty:
        raise RuntimeError("No valid chart points are available for 3D coverage")
    all_points = np.concatenate(nonempty, axis=0)
    lower = np.quantile(all_points, 0.002, axis=0)
    upper = np.quantile(all_points, 0.998, axis=0)
    span = np.maximum(upper - lower, 1e-6)
    raw_keys: list[np.ndarray] = []
    for value in samples:
        if not len(value):
            raw_keys.append(np.empty(0, dtype=np.int64))
            continue
        quantized = np.floor((value - lower) / span * bins).astype(np.int64)
        quantized = np.clip(quantized, 0, bins - 1)
        key = quantized[:, 0] * bins * bins + quantized[:, 1] * bins + quantized[:, 2]
        raw_keys.append(np.unique(key))
    all_keys = np.unique(np.concatenate(raw_keys))
    compact = [np.searchsorted(all_keys, value) for value in raw_keys]
    frequency = np.zeros(len(all_keys), dtype=np.int32)
    for value in compact:
        frequency[value] += 1
    # Singleton cells are useful but may originate from an uncertain boundary;
    # sqrt rather than inverse weighting gives them a bounded advantage.
    weights = 1.0 / np.sqrt(np.maximum(frequency, 1))
    return compact, weights.astype(np.float64), np.stack([lower, upper])


def _coverage_fraction(selected: Iterable[int], cells: list[np.ndarray], weights: np.ndarray) -> float:
    selected = list(selected)
    if not selected:
        return 0.0
    used = np.zeros(len(weights), dtype=bool)
    for index in selected:
        used[cells[index]] = True
    return float(weights[used].sum() / max(float(weights.sum()), 1e-12))


def _exclusive_coverage_fraction(
    selected: list[int], cells: list[np.ndarray], weights: np.ndarray
) -> dict[int, float]:
    total = max(float(weights.sum()), 1e-12)
    counts = np.zeros(len(weights), dtype=np.int32)
    for index in selected:
        counts[cells[index]] += 1
    return {
        index: float(weights[cells[index]][counts[cells[index]] == 1].sum() / total)
        for index in selected
    }


class JointSelector:
    """Constrained joint quality / 3D coverage / pose coverage selection."""

    def __init__(
        self,
        reliabilities: np.ndarray,
        cells: list[np.ndarray],
        cell_weights: np.ndarray,
        target_distances: np.ndarray,
        centers: np.ndarray,
        support_matrix: np.ndarray,
        pair_scores: np.ndarray,
        reference_matrix: Optional[np.ndarray] = None,
        reference_names: Optional[list[str]] = None,
        sequence_matrix: Optional[np.ndarray] = None,
        sequence_names: Optional[list[str]] = None,
        *,
        min_support: int,
        min_baseline: float,
        min_reference_support: int = 1,
        min_sequence_support: int = 0,
    ) -> None:
        self.reliabilities = reliabilities
        self.cells = cells
        self.cell_weights = cell_weights
        self.target_distances = target_distances
        self.centers = centers
        self.support_matrix = support_matrix
        self.pair_scores = pair_scores
        self.reference_matrix = (
            reference_matrix
            if reference_matrix is not None
            else np.zeros((0, len(reliabilities)), dtype=bool)
        )
        self.reference_names = list(reference_names or [])
        if self.reference_matrix.shape != (len(self.reference_names), len(reliabilities)):
            raise ValueError("reference matrix must have shape [n_references, n_charts]")
        self.sequence_matrix = (
            sequence_matrix
            if sequence_matrix is not None
            else np.zeros((0, len(reliabilities)), dtype=bool)
        )
        self.sequence_names = list(sequence_names or [])
        if self.sequence_matrix.shape != (len(self.sequence_names), len(reliabilities)):
            raise ValueError("sequence matrix must have shape [n_sequences, n_charts]")
        self.min_support = int(min_support)
        self.min_baseline = float(min_baseline)
        if min_reference_support < 1:
            raise ValueError("min_reference_support must be positive")
        self.min_reference_support = int(min_reference_support)
        if min_sequence_support < 0:
            raise ValueError("min_sequence_support must be non-negative")
        self.min_sequence_support = int(min_sequence_support)
        if self.min_sequence_support > 0 and not self.sequence_names:
            raise ValueError(
                "min_sequence_support requires one or more same-sequence support groups"
            )
        # A retained MAtCha camera is an anchor, not merely a point that needs
        # one nearby replacement.  Ask for redundant hard-gated support where
        # the source actually offers it, but never make a sparse trajectory
        # impossible by requiring two views when only one can geometrically
        # cover that anchor.
        self.reference_available_support = self.reference_matrix.sum(axis=1).astype(np.int32)
        self.reference_required_support = np.minimum(
            self.reference_available_support,
            self.min_reference_support,
        ).astype(np.int32)
        self.pose_normalizer = max(
            float(np.quantile(target_distances, 0.90)), 1e-6
        )

    def _baseline_satisfied(self, support_indices: list[int]) -> bool:
        return any(
            float(np.linalg.norm(self.centers[first] - self.centers[second]))
            >= self.min_baseline
            for offset, first in enumerate(support_indices)
            for second in support_indices[offset + 1 :]
        )

    def _support_diagnostics(
        self,
        selected: list[int],
        matrix: np.ndarray,
        labels: list[Any],
        *,
        label_key: str,
        required_support: int,
    ) -> list[dict[str, Any]]:
        diagnostics = []
        for group, support in enumerate(matrix):
            support_indices = [index for index in selected if support[index]]
            baseline = self._baseline_satisfied(support_indices)
            diagnostics.append(
                {
                    label_key: labels[group],
                    "required_support_count": required_support,
                    "support_count": len(support_indices),
                    "support_indices": support_indices,
                    "baseline_satisfied": baseline,
                    "passed": len(support_indices) >= required_support and baseline,
                }
            )
        return diagnostics

    def constraints(self, selected: Iterable[int]) -> tuple[bool, dict[str, list[dict[str, Any]]]]:
        chosen = list(selected)
        diagnostics = self._support_diagnostics(
            chosen,
            self.support_matrix,
            list(range(len(self.support_matrix))),
            label_key="cluster",
            required_support=self.min_support,
        )
        sequence_diagnostics = self._support_diagnostics(
            chosen,
            self.sequence_matrix,
            self.sequence_names,
            label_key="sequence",
            required_support=self.min_sequence_support,
        )
        reference_diagnostics = []
        for reference_index, support in enumerate(self.reference_matrix):
            support_indices = [index for index in chosen if support[index]]
            available_count = int(self.reference_available_support[reference_index])
            required_count = int(self.reference_required_support[reference_index])
            reference_diagnostics.append(
                {
                    "reference": self.reference_names[reference_index],
                    "available_support_count": available_count,
                    "required_support_count": required_count,
                    "support_count": len(support_indices),
                    "support_indices": support_indices,
                    "passed": len(support_indices) >= required_count,
                }
            )
        passed = (
            all(value["passed"] for value in diagnostics)
            and all(value["passed"] for value in sequence_diagnostics)
            and all(value["passed"] for value in reference_diagnostics)
        )
        return passed, {
            "clusters": diagnostics,
            "sequences": sequence_diagnostics,
            "references": reference_diagnostics,
        }

    def objective(self, selected: Iterable[int]) -> dict[str, float]:
        chosen = list(selected)
        if not chosen:
            return {"score": -float("inf"), "quality": 0.0, "coverage": 0.0, "pose": 0.0, "pair": 0.0}
        coverage = _coverage_fraction(chosen, self.cells, self.cell_weights)
        nearest = self.target_distances[:, chosen].min(axis=1)
        pose = _clamp01(1.0 - float(nearest.mean()) / self.pose_normalizer)
        quality = float(self.reliabilities[chosen].mean())
        if len(chosen) <= 1:
            pair = 0.0
        else:
            matrix = self.pair_scores[np.ix_(chosen, chosen)].copy()
            np.fill_diagonal(matrix, 0.0)
            pair = float(np.max(matrix, axis=1).mean())
        score = 0.35 * quality + 0.35 * coverage + 0.20 * pose + 0.10 * pair
        return {
            "score": float(score),
            "quality": quality,
            "coverage": coverage,
            "pose": pose,
            "pair": pair,
        }

    def _candidate_scores(self, selected: list[int], candidates: list[int]) -> dict[int, float]:
        if not candidates:
            return {}
        used = np.zeros(len(self.cell_weights), dtype=bool)
        for index in selected:
            used[self.cells[index]] = True
        if selected:
            nearest = self.target_distances[:, selected].min(axis=1)
        else:
            nearest = np.full(self.target_distances.shape[0], self.pose_normalizer)
        coverage_gain = np.asarray(
            [self.cell_weights[self.cells[index][~used[self.cells[index]]]].sum() for index in candidates],
            dtype=np.float64,
        )
        pose_gain = np.asarray(
            [np.maximum(nearest - self.target_distances[:, index], 0.0).mean() for index in candidates],
            dtype=np.float64,
        )
        coverage_rank = _rank01(coverage_gain)
        pose_rank = _rank01(pose_gain)
        output: dict[int, float] = {}
        for offset, index in enumerate(candidates):
            pair = (
                float(self.pair_scores[index, selected].max())
                if selected
                else 0.0
            )
            output[index] = float(
                0.45 * self.reliabilities[index]
                + 0.30 * coverage_rank[offset]
                + 0.18 * pose_rank[offset]
                + 0.07 * pair
            )
        return output

    def _constraint_bonus(self, selected: list[int], candidate: int) -> float:
        bonus = 0.0
        support_groups = [(self.support_matrix, self.min_support)]
        if self.min_sequence_support > 0:
            support_groups.append((self.sequence_matrix, self.min_sequence_support))
        for matrix, required_support in support_groups:
            for support in matrix:
                if not support[candidate]:
                    continue
                current = [index for index in selected if support[index]]
                if len(current) < required_support:
                    bonus += 2.0
                has_baseline = self._baseline_satisfied(current)
                adds_baseline = any(
                    float(np.linalg.norm(self.centers[candidate] - self.centers[other]))
                    >= self.min_baseline
                    for other in current
                )
                if current and not has_baseline and adds_baseline:
                    bonus += 1.0
        for reference_index, support in enumerate(self.reference_matrix):
            current_count = sum(bool(support[index]) for index in selected)
            if (
                support[candidate]
                and current_count < int(self.reference_required_support[reference_index])
            ):
                bonus += 2.0
        return bonus

    def _seed_pose_constraints(self, count: int) -> list[int]:
        """Construct a feasible pose/trajectory seed before optimizing quality.

        The old all-in-one greedy loop could spend its fixed budget on very
        high-quality charts that each helped an under-supported pose cell, but
        never formed the *baseline-separated pair* required by that cell.  It
        then reported an infeasible 24-chart request even when the active
        hard-gated pool contained a valid pair for every cell.  Seed one
        admissible pair per cell first; pairs are deliberately scored for
        cross-cell reuse, so this preserves room for the later quality/3-D
        coverage objective rather than relaxing any constraint.
        """
        selected: list[int] = []
        support_groups: list[tuple[np.ndarray, list[Any], int, str]] = [
            (
                self.support_matrix,
                list(range(len(self.support_matrix))),
                self.min_support,
                "pose cluster",
            )
        ]
        if self.min_sequence_support > 0:
            support_groups.append(
                (
                    self.sequence_matrix,
                    self.sequence_names,
                    self.min_sequence_support,
                    "same-sequence trajectory",
                )
            )
        for matrix, labels, required_support, group_label in support_groups:
            for group, support in enumerate(matrix):
                candidates = np.flatnonzero(support).astype(np.int64).tolist()
                label = labels[group]
                pairs = [
                    (first, second)
                    for offset, first in enumerate(candidates)
                    for second in candidates[offset + 1 :]
                    if float(np.linalg.norm(self.centers[first] - self.centers[second]))
                    >= self.min_baseline
                ]
                if not pairs:
                    raise RuntimeError(
                        "Hard-gated Chart pool has no baseline-separated support pair "
                        f"for {group_label} {label}; do not relax the gate, replace its candidates."
                    )

                def pair_key(pair: tuple[int, int]) -> tuple[float, float, float, int, int]:
                    first, second = pair
                    added = int(first not in selected) + int(second not in selected)
                    # A pair that covers more pose cells can satisfy future
                    # constraints too. Prefer reuse before raw quality so this is
                    # a feasibility seed, not a hidden quality objective.
                    shared_cells = float(
                        (self.support_matrix[:, first] & self.support_matrix[:, second]).sum()
                    )
                    if self.min_sequence_support > 0:
                        shared_cells += float(
                            (self.sequence_matrix[:, first] & self.sequence_matrix[:, second]).sum()
                        )
                    quality = float(self.reliabilities[first] + self.reliabilities[second])
                    return (-float(added), shared_cells, quality, -first, -second)

                first, second = max(pairs, key=pair_key)
                for index in (first, second):
                    if index not in selected:
                        selected.append(index)

                # The current Cambridge policy asks for two support views. Keep
                # this general for future configurations that request more.
                while sum(bool(support[index]) for index in selected) < required_support:
                    remaining = [index for index in candidates if index not in selected]
                    if not remaining:
                        raise RuntimeError(
                            f"Hard-gated Chart pool has fewer than {required_support} "
                            f"support views for {group_label} {label}."
                        )
                    selected.append(
                        max(remaining, key=lambda index: (self.reliabilities[index], -index))
                    )

                if len(selected) > count:
                    raise RuntimeError(
                        "Pose/trajectory-support seed needs "
                        f"{len(selected)} Charts, exceeding requested count {count}."
                    )
        return sorted(selected)

    def select(self, count: int, local_swap_rounds: int = 4) -> tuple[list[int], dict[str, Any]]:
        if count > len(self.reliabilities):
            raise ValueError(f"Requested {count} charts but only {len(self.reliabilities)} passed the hard gate")
        selected = self._seed_pose_constraints(count)
        # First satisfy the exact constraints audited downstream.  A view can
        # support multiple pose cells, so the feasibility seed leaves room for
        # reference-anchor constraints and the quality/coverage objective.
        while True:
            valid, _ = self.constraints(selected)
            if valid:
                break
            if len(selected) >= count:
                raise RuntimeError(
                    f"Could not satisfy pose/trajectory support/baseline constraints within {count} charts"
                )
            candidates = [index for index in range(len(self.reliabilities)) if index not in selected]
            scores = self._candidate_scores(selected, candidates)
            next_index = max(
                candidates,
                key=lambda index: (self._constraint_bonus(selected, index), scores[index], -index),
            )
            selected.append(next_index)

        while len(selected) < count:
            candidates = [index for index in range(len(self.reliabilities)) if index not in selected]
            scores = self._candidate_scores(selected, candidates)
            selected.append(max(candidates, key=lambda index: (scores[index], -index)))

        selected.sort()
        swaps = []
        current = self.objective(selected)
        for _ in range(local_swap_rounds):
            best_selected = selected
            best_objective = current
            selected_set = set(selected)
            for removed in selected:
                retained = [index for index in selected if index != removed]
                for added in range(len(self.reliabilities)):
                    if added in selected_set:
                        continue
                    proposal = sorted(retained + [added])
                    valid, _ = self.constraints(proposal)
                    if not valid:
                        continue
                    objective = self.objective(proposal)
                    if objective["score"] > best_objective["score"] + 1e-8:
                        best_selected = proposal
                        best_objective = objective
            if best_selected == selected:
                break
            swaps.append(
                {
                    "removed": sorted(set(selected) - set(best_selected)),
                    "added": sorted(set(best_selected) - set(selected)),
                    "score_before": current["score"],
                    "score_after": best_objective["score"],
                }
            )
            selected, current = best_selected, best_objective
        valid, constraints = self.constraints(selected)
        if not valid:
            raise AssertionError("Quality-aware local search broke a hard coverage constraint")
        return selected, {"objective": current, "constraints": constraints, "local_swaps": swaps}


def _build_reliability(
    names: list[str],
    gate_records: dict[str, dict[str, Any]],
    base_scores: dict[str, dict[str, Any]],
    audit_scores: dict[str, dict[str, float]],
    static_support: np.ndarray,
    pair_summary: list[dict[str, float]],
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """Combine independently meaningful terms only after robust normalization."""
    rows = []
    for index, name in enumerate(names):
        gate = gate_records[name]
        base = base_scores.get(name, {})
        audit = audit_scores.get(name, {})
        laplacian = _safe_float(
            audit.get("laplacian_variance"), _safe_float(base.get("sharpness"))
        )
        gradient = _safe_float(
            audit.get("gradient_variance"), _safe_float(base.get("sharpness"))
        )
        # Older joint-selection payloads did not retain a standalone
        # sharpness scalar.  Treating that missing field as zero turns one
        # third of the clarity term into a constant.  The audit has two
        # complementary, physically meaningful measurements instead:
        # Laplacian energy captures fine edges and gradient energy captures
        # broader contrast.  Their geometric mean rewards a chart only when
        # both are present, while retaining an explicit upstream sharpness
        # score when a newer selector supplies one.
        upstream_sharpness = _safe_float(base.get("sharpness"), float("nan"))
        if not math.isfinite(upstream_sharpness) or upstream_sharpness <= 0.0:
            sharpness = math.sqrt(max(laplacian, 0.0) * max(gradient, 0.0))
            sharpness_from_audit = 1.0
        else:
            sharpness = upstream_sharpness
            sharpness_from_audit = 0.0
        rows.append(
            {
                "name": name,
                "laplacian": laplacian,
                "gradient": gradient,
                "sharpness": sharpness,
                "sharpness_from_audit": sharpness_from_audit,
                "risk": _safe_float(audit.get("risk_score")),
                "exposure": max(_safe_float(audit.get("exposure_warning")), _safe_float(base.get("extreme_ratio"))),
                "tracks": _safe_float(audit.get("track_count")),
                "view_valid": _safe_float(audit.get("valid_ratio"), _safe_float(base.get("valid_ratio"))),
                "static_support": float(static_support[index]),
                "gate_valid": _safe_float(gate.get("valid_fraction")),
                "gate_support": _safe_float(gate.get("support_ratio"), 1.0),
                "gate_p90": _safe_float(gate.get("relative_p90")),
                "gate_gt25": _safe_float(gate.get("gt25_fraction")),
            }
        )

    def rank(field: str, higher: bool = True) -> np.ndarray:
        return _rank01([row[field] for row in rows], higher)

    clarity = (
        0.25 * rank("laplacian")
        + 0.25 * rank("gradient")
        + 0.50 * rank("sharpness")
    )
    appearance = (clarity + rank("risk", False) + rank("exposure", False)) / 3.0
    geometry = (
        rank("gate_valid")
        + rank("gate_support")
        + rank("gate_p90", False)
        + rank("gate_gt25", False)
    ) / 4.0
    support = (rank("tracks") + rank("view_valid") + rank("static_support")) / 3.0

    pair_by_source: list[list[dict[str, float]]] = [[] for _ in names]
    for record in pair_summary:
        pair_by_source[int(record["source_index"])].append(record)
    photo = np.zeros(len(names), dtype=np.float64)
    normals = np.zeros(len(names), dtype=np.float64)
    cross_support = np.zeros(len(names), dtype=np.float64)
    depth = np.zeros(len(names), dtype=np.float64)
    for index, records in enumerate(pair_by_source):
        if not records:
            continue
        weights = np.asarray([record["support_fraction"] for record in records], dtype=np.float64)
        # Unsupported projections cannot make a chart look good.  For charts
        # with valid overlap, support-weighted aggregation avoids a tiny shared
        # sliver dominating the consistency term.
        if weights.sum() <= 1e-8:
            continue
        weights /= weights.sum()
        photo[index] = float(np.dot(weights, [record["photo_score"] for record in records]))
        normals[index] = float(np.dot(weights, [record["normal_score"] for record in records]))
        depth[index] = float(np.dot(weights, [record["depth_score"] for record in records]))
        cross_support[index] = float(np.mean([record["support_fraction"] for record in records]))
    photo_rank = _rank01(photo)
    normal_rank = _rank01(normals)
    depth_rank = _rank01(depth)
    cross_support_rank = _rank01(cross_support)
    reliability = (
        0.20 * clarity
        + 0.15 * appearance
        + 0.20 * geometry
        + 0.15 * support
        + 0.15 * photo_rank
        + 0.10 * normal_rank
        + 0.03 * depth_rank
        + 0.02 * cross_support_rank
    )
    # Keep every hard-gated chart selectable, but retain a non-zero floor for
    # voxel sampling tie breaks when it is the only chart covering a region.
    reliability = 0.05 + 0.90 * np.clip(reliability, 0.0, 1.0)

    output = {}
    for index, row in enumerate(rows):
        output[row["name"]] = {
            **{key: float(value) if key != "name" else value for key, value in row.items() if key != "name"},
            "clarity": float(clarity[index]),
            "appearance": float(appearance[index]),
            "geometry": float(geometry[index]),
            "support": float(support[index]),
            "crossview_photo": float(photo[index]),
            "crossview_normal": float(normals[index]),
            "crossview_depth": float(depth[index]),
            "crossview_support": float(cross_support[index]),
            "reliability": float(reliability[index]),
        }
    return reliability.astype(np.float64), output


def _active_gate_records(gate_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output = {}
    for record in gate_payload.get("records", []):
        if record.get("rejected"):
            continue
        name = _name(str(record["image_name"]))
        if name in output:
            raise ValueError(f"Duplicate hard-gate record for {name}")
        output[name] = record
    return output


def _build_masks(
    names: list[str],
    shape: tuple[int, int],
    mask_pickle: Optional[Path],
    mask_dataset_path: Optional[Path],
    mask_indices: list[int],
) -> np.ndarray:
    if mask_pickle is None:
        return np.ones((len(names), *shape), dtype=bool)
    if mask_dataset_path is None:
        raise ValueError("--mask-dataset-path is required together with --mask-pickle")
    lookup = CambridgeMaskLookup(mask_dataset_path, mask_pickle, mask_indices=mask_indices)
    return np.stack(
        [
            lookup.get_mask(name, shape, torch.device("cpu")).cpu().numpy().astype(bool)
            for name in names
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-path", type=Path, required=True, help="Hard-gated MASt3R chart scene.")
    parser.add_argument(
        "--target-scene-path",
        type=Path,
        help=(
            "Full training COLMAP scene used for pose coverage/audit. Defaults to "
            "the original selection's scene_path."
        ),
    )
    parser.add_argument("--selection", type=Path, required=True, help="Original joint chart selection JSON.")
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument(
        "--reference-cameras-json",
        type=Path,
        help=(
            "Proven MAtCha chart cameras that must retain a same-sequence "
            "nearby hard-gated replacement."
        ),
    )
    parser.add_argument("--reference-max-pose-distance", type=float, default=0.30)
    parser.add_argument(
        "--min-reference-support",
        type=int,
        default=1,
        help=(
            "Require this many same-sequence hard-gated charts for each retained "
            "MAtCha anchor when available; the requirement is capped by the "
            "number of geometrically eligible charts for that anchor."
        ),
    )
    parser.add_argument("--view-audit-csv", type=Path)
    parser.add_argument("--mask-pickle", type=Path)
    parser.add_argument("--mask-dataset-path", type=Path)
    parser.add_argument("--mask-indices", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--counts", nargs="*", type=int, default=[24, 28, 32])
    parser.add_argument(
        "--min-pose-support",
        type=int,
        default=None,
        help=(
            "Override the original candidate-set support requirement for the "
            "post-gate joint selection.  This must be explicit when the broad "
            "candidate set carries additional gate-reserve views."
        ),
    )
    parser.add_argument(
        "--min-sequence-support",
        type=int,
        default=None,
        help=(
            "Require this many baseline-separated Charts for every applicable "
            "trajectory sequence after the hard gate. Defaults to the source "
            "selection's post-gate same-sequence requirement. Set 0 only for "
            "an explicit legacy/soft-coverage ablation."
        ),
    )
    parser.add_argument("--neighbors", type=int, default=6)
    parser.add_argument(
        "--neighbor-selection",
        choices=("pose", "overlap"),
        default="pose",
        help=(
            "Use legacy nearest-pose peers or rank every hard-gated peer by "
            "measured valid-pixel overlap before scoring cross-view quality."
        ),
    )
    parser.add_argument(
        "--min-neighbor-overlap",
        type=float,
        default=0.05,
        help="Minimum source valid-pixel fraction required for an overlap peer.",
    )
    parser.add_argument("--pair-stride", type=int, default=6)
    parser.add_argument("--coverage-stride", type=int, default=6)
    parser.add_argument("--coverage-voxel-bins", type=int, default=56)
    parser.add_argument("--depth-relative-threshold", type=float, default=0.12)
    parser.add_argument("--local-swap-rounds", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.neighbors < 1 or args.pair_stride < 1 or args.coverage_stride < 1:
        raise ValueError("neighbors and strides must be positive")
    if not 0.0 <= args.min_neighbor_overlap <= 1.0:
        raise ValueError("--min-neighbor-overlap must lie in [0, 1]")
    if args.coverage_voxel_bins < 2:
        raise ValueError("coverage voxel bins must be at least 2")
    if args.min_reference_support < 1:
        raise ValueError("--min-reference-support must be positive")
    if args.min_pose_support is not None and args.min_pose_support < 1:
        raise ValueError("--min-pose-support must be positive")
    if args.min_sequence_support is not None and args.min_sequence_support < 0:
        raise ValueError("--min-sequence-support must be non-negative")
    requested_counts = sorted(set(int(value) for value in args.counts))
    if not requested_counts or min(requested_counts) <= 0:
        raise ValueError("--counts must contain one or more positive integers")

    scene_path = args.scene_path.resolve()
    base_selection = json.loads(args.selection.read_text())
    target_scene_path = (
        args.target_scene_path.resolve()
        if args.target_scene_path is not None
        else Path(base_selection["scene_path"]).resolve()
    )
    gate_payload = json.loads(args.gate_report.read_text())
    gate_records = _active_gate_records(gate_payload)
    charts = np.load(scene_path / "charts_data.npz")
    points_all = charts["pts"].astype(np.float32, copy=False)
    depths_all = charts["depths"].astype(np.float32, copy=False)
    confs_all = charts["confs"].astype(np.float32, copy=False)
    height, width = depths_all.shape[-2:]
    scale_factor = float(charts["scale_factor"])

    cameras = read_cameras_binary(str(scene_path / "sparse" / "0" / "cameras.bin"))
    colmap_images = sorted(
        read_images_binary(str(scene_path / "sparse" / "0" / "images.bin")).values(),
        key=lambda image: image.id,
    )
    scene_names = [_name(image.name) for image in colmap_images]
    if len(scene_names) != len(points_all):
        raise RuntimeError("COLMAP image count does not match charts_data")
    if not set(gate_records).issubset(set(scene_names)):
        missing = sorted(set(gate_records) - set(scene_names))
        extra = sorted(set(scene_names) - set(gate_records))
        raise RuntimeError(
            "Hard-gate / chart scene mismatch: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    # Preserve chart order in all tensors; names are only a stable key.
    active_indices = [index for index, name in enumerate(scene_names) if name in gate_records]
    names = [scene_names[index] for index in active_indices]
    points = points_all[active_indices]
    depths = depths_all[active_indices]
    confs = confs_all[active_indices]
    images = [colmap_images[index] for index in active_indices]
    static_masks = _build_masks(
        names, (height, width), args.mask_pickle, args.mask_dataset_path, args.mask_indices
    )
    finite = np.isfinite(points).all(axis=3) & np.isfinite(depths) & np.isfinite(confs)
    valid_masks = static_masks & finite & (confs > 0.0) & (depths > 0.0)

    rotations = np.stack([qvec_to_rotmat(image.qvec) for image in images]).astype(np.float32)
    translations = np.stack([image.tvec * scale_factor for image in images]).astype(np.float32)
    centers = np.stack(
        [-qvec_to_rotmat(image.qvec).T @ (image.tvec * scale_factor) for image in images]
    ).astype(np.float32)
    chart_intrinsics = [_intrinsics(cameras[image.camera_id], width, height) for image in images]
    rgb_dir = scene_path / "images"
    rgbs = [_read_chart_rgb(rgb_dir / name, width, height) for name in names]
    normals_and_valid = [_world_normals(points[index], valid_masks[index], centers[index]) for index in range(len(names))]
    normals = np.stack([value[0] for value in normals_and_valid])
    normal_valid = np.stack([value[1] for value in normals_and_valid])

    # Neighboring poses are the most diagnostic: they share enough surface to
    # identify blur/floaters while avoiding false disagreement from extreme
    # occlusions.  They are selected in trajectory pose+direction space.
    all_scene_names, all_scene_poses = load_scene_poses(target_scene_path)
    missing_chart_poses = sorted(set(scene_names) - set(all_scene_names))
    if missing_chart_poses:
        raise RuntimeError(
            "The full target scene has no pose for chart(s): "
            + ", ".join(missing_chart_poses[:5])
        )
    normalized_centers, directions, _ = build_pose_geometry(all_scene_names, all_scene_poses)
    all_features = np.concatenate([normalized_centers, 0.35 * directions], axis=1)
    all_name_to_index = {name: index for index, name in enumerate(all_scene_names)}
    active_scene_indices = np.asarray([all_name_to_index[name] for name in names], dtype=np.int64)
    active_features = all_features[active_scene_indices]
    reference_names: list[str] = []
    reference_matrix = np.zeros((0, len(names)), dtype=bool)
    if args.reference_cameras_json is not None:
        archived_features = archived_reference_features(
            args.reference_cameras_json, all_scene_names, all_scene_poses
        )
        reference_rows = []
        for raw_name, feature in sorted(archived_features.items()):
            reference_name = _name(raw_name)
            same_sequence = np.asarray(
                [name.split("__", 1)[0] == reference_name.split("__", 1)[0] for name in names],
                dtype=bool,
            )
            supported = same_sequence & (
                np.linalg.norm(active_features - feature[None], axis=1)
                <= args.reference_max_pose_distance
            )
            if not np.any(supported):
                raise RuntimeError(
                    "No hard-gated chart can cover retained reference "
                    f"{reference_name} within {args.reference_max_pose_distance:.3f}"
                )
            reference_names.append(reference_name)
            reference_rows.append(supported)
        reference_matrix = np.stack(reference_rows)
    active_pair_distances = np.linalg.norm(
        active_features[:, None] - active_features[None, :], axis=2
    )
    np.fill_diagonal(active_pair_distances, np.inf)
    pair_summary: list[dict[str, float]] = []
    selected_pair_overlaps: dict[int, dict[int, float]] = {}
    for source_index in range(len(names)):
        candidates = np.asarray(
            [index for index in range(len(names)) if index != source_index], dtype=np.int64
        )
        if args.neighbor_selection == "pose":
            neighbors = candidates[
                np.argsort(active_pair_distances[source_index, candidates], kind="stable")
            ][: min(args.neighbors, len(candidates))]
            overlaps = None
        else:
            candidate_overlaps = np.asarray(
                [
                    _projection_overlap_fraction(
                        source_index,
                        int(target_index),
                        points=points,
                        valid_masks=valid_masks,
                        normal_valid=normal_valid,
                        rotations=rotations,
                        translations=translations,
                        intrinsics=chart_intrinsics,
                        stride=args.pair_stride,
                    )
                    for target_index in candidates
                ],
                dtype=np.float64,
            )
            neighbors = select_overlap_neighbours(
                candidates,
                candidate_overlaps,
                args.neighbors,
                args.min_neighbor_overlap,
            )
            overlaps = {
                int(target_index): float(candidate_overlaps[offset])
                for offset, target_index in enumerate(candidates)
            }
        selected_pair_overlaps[source_index] = (
            {int(target_index): overlaps[int(target_index)] for target_index in neighbors}
            if overlaps is not None
            else {}
        )
        for target_index in neighbors:
            record = _pair_consistency(
                source_index,
                int(target_index),
                points=points,
                depths=depths,
                valid_masks=valid_masks,
                normals=normals,
                normal_valid=normal_valid,
                rgbs=rgbs,
                rotations=rotations,
                translations=translations,
                intrinsics=chart_intrinsics,
                stride=args.pair_stride,
                depth_threshold=args.depth_relative_threshold,
            )
            record["neighbor_overlap_fraction"] = (
                selected_pair_overlaps[source_index].get(int(target_index))
                if args.neighbor_selection == "overlap"
                else None
            )
            pair_summary.append(record)

    base_scores = {
        _name(name): value
        for name, value in base_selection.get("quality_scores", {}).items()
    }
    audit_scores = _read_view_audit(args.view_audit_csv)
    static_support = valid_masks.reshape(len(names), -1).mean(axis=1)
    reliabilities, chart_scores = _build_reliability(
        names, gate_records, base_scores, audit_scores, static_support, pair_summary
    )
    cells, cell_weights, coverage_bounds = _voxel_coverage(
        points, valid_masks, args.coverage_stride, args.coverage_voxel_bins
    )

    pair_matrix = np.zeros((len(names), len(names)), dtype=np.float64)
    pair_counts = np.zeros_like(pair_matrix, dtype=np.int32)
    for record in pair_summary:
        source = int(record["source_index"])
        target = int(record["target_index"])
        pair_matrix[source, target] += record["pair_score"]
        pair_counts[source, target] += 1
    nonzero_pairs = pair_counts > 0
    pair_matrix[nonzero_pairs] /= pair_counts[nonzero_pairs]
    pair_matrix = np.maximum(pair_matrix, pair_matrix.T)

    coverage_config = base_selection.get("coverage", {})
    n_clusters = int(coverage_config.get("view_clusters", 8))
    min_support = int(
        args.min_pose_support
        if args.min_pose_support is not None
        else coverage_config.get("min_views_per_cluster", 2)
    )
    min_support_source = (
        "explicit_post_gate_override"
        if args.min_pose_support is not None
        else "base_candidate_selection_coverage"
    )
    min_baseline = float(coverage_config.get("min_baseline_ratio", 0.02))
    sequence_coverage_mode = str(coverage_config.get("sequence_coverage_mode", "off"))
    if args.min_sequence_support is not None:
        min_sequence_support = int(args.min_sequence_support)
        min_sequence_support_source = "explicit_post_gate_override"
    elif sequence_coverage_mode == "off":
        min_sequence_support = 0
        min_sequence_support_source = "base_selection_disabled"
    else:
        min_sequence_support = int(
            coverage_config.get("post_gate_min_views_per_sequence", 0)
        )
        min_sequence_support_source = "base_candidate_selection_coverage"
    sequence_names: list[str] = []
    sequence_rows: list[np.ndarray] = []
    if min_sequence_support > 0:
        base_sequence_records = coverage_config.get("sequences")
        if not isinstance(base_sequence_records, list) or not base_sequence_records:
            raise RuntimeError(
                "The source Chart selection requires same-sequence post-gate support "
                "but has no sequence coverage audit. Re-select the broad Chart set; "
                "do not silently drop the trajectory constraint."
            )
        for record in base_sequence_records:
            if not isinstance(record, dict) or record.get("applicable", True) is False:
                continue
            sequence = record.get("sequence")
            if not isinstance(sequence, str) or not sequence:
                raise RuntimeError("Source sequence coverage audit contains an invalid sequence name")
            if sequence in sequence_names:
                raise RuntimeError(f"Source sequence coverage audit duplicates {sequence}")
            support = np.asarray(
                [name.split("__", 1)[0] == sequence for name in names], dtype=bool
            )
            if int(support.sum()) < min_sequence_support:
                raise RuntimeError(
                    "Hard-gated Chart pool cannot preserve the source same-sequence "
                    f"requirement for {sequence}: active={int(support.sum())}, "
                    f"required={min_sequence_support}. Do not relax the quality selector; "
                    "replace/re-align its broad Chart candidates."
                )
            sequence_names.append(sequence)
            sequence_rows.append(support)
        if not sequence_rows:
            raise RuntimeError(
                "The source Chart selection has no applicable same-sequence support groups"
            )
    sequence_matrix = (
        np.stack(sequence_rows)
        if sequence_rows
        else np.zeros((0, len(names)), dtype=bool)
    )
    max_borrowed = float(coverage_config.get("max_borrowed_support_distance", 0.32))
    anchors = _farthest_point_indices(all_features, n_clusters)
    anchor_features = all_features[np.asarray(anchors)]
    labels = np.argmin(
        np.linalg.norm(all_features[:, None] - anchor_features[None], axis=2), axis=1
    )
    support_matrix = np.stack(
        [
            (labels[active_scene_indices] == cluster)
            | (
                np.linalg.norm(active_features - anchor_features[cluster], axis=1)
                <= max_borrowed
            )
            for cluster in range(n_clusters)
        ]
    )
    target_distances = np.linalg.norm(
        all_features[:, None] - active_features[None], axis=2
    )
    selector = JointSelector(
        reliabilities,
        cells,
        cell_weights,
        target_distances,
        centers,
        support_matrix,
        pair_matrix,
        reference_matrix=reference_matrix,
        reference_names=reference_names,
        sequence_matrix=sequence_matrix,
        sequence_names=sequence_names,
        min_support=min_support,
        min_baseline=min_baseline,
        min_reference_support=args.min_reference_support,
        min_sequence_support=min_sequence_support,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    quality_path = args.output_dir / "chart_quality_scores.json"
    quality_payload = {
        "version": 3,
        "method": "hard_gate_then_quality_crossview_normal_3d_coverage_sequence_support",
        "scene_path": str(target_scene_path),
        "chart_source_scene_path": str(scene_path),
        "hard_gate_report": str(args.gate_report.resolve()),
        "reference_cameras_json": (
            str(args.reference_cameras_json.resolve())
            if args.reference_cameras_json is not None
            else None
        ),
        "reference_max_pose_distance": args.reference_max_pose_distance,
        "min_reference_support_requested": args.min_reference_support,
        "reference_count": len(reference_names),
        "active_chart_count": len(names),
        "min_pose_support": min_support,
        "min_pose_support_source": min_support_source,
        "sequence_coverage_mode": sequence_coverage_mode,
        "min_sequence_support": min_sequence_support,
        "min_sequence_support_source": min_sequence_support_source,
        "applicable_sequences": sequence_names,
        "pair_count": len(pair_summary),
        "pair_stride": args.pair_stride,
        "neighbor_selection": args.neighbor_selection,
        "min_neighbor_overlap": (
            args.min_neighbor_overlap if args.neighbor_selection == "overlap" else None
        ),
        "depth_relative_threshold": args.depth_relative_threshold,
        "coverage": {
            "stride": args.coverage_stride,
            "voxel_bins": args.coverage_voxel_bins,
            "occupied_voxels": int(len(cell_weights)),
            "bounds": coverage_bounds.tolist(),
        },
        "chart_scores": chart_scores,
        "pairs": [
            {
                **record,
                "source_name": names[int(record["source_index"])],
                "target_name": names[int(record["target_index"])],
            }
            for record in pair_summary
        ],
    }
    quality_path.write_text(json.dumps(quality_payload, indent=2))

    exclusive_all = _exclusive_coverage_fraction(list(range(len(names))), cells, cell_weights)
    for index, name in enumerate(names):
        chart_scores[name]["exclusive_coverage_all_active"] = float(exclusive_all[index])

    results = []
    for count in requested_counts:
        selected, diagnostics = selector.select(count, local_swap_rounds=args.local_swap_rounds)
        selected_names = [names[index] for index in selected]
        exclusive = _exclusive_coverage_fraction(selected, cells, cell_weights)
        for index in selected:
            chart_scores[names[index]][f"exclusive_coverage_selected_n{count}"] = float(exclusive[index])
        selected_scene_indices = [int(active_scene_indices[index]) for index in selected]
        selection_payload = {
            **base_selection,
            "scene_path": str(target_scene_path),
            "chart_source_scene_path": str(scene_path),
            "n_images": int(count),
            "requested_n_images": int(count),
            "actual_n_images": int(count),
            "image_names": selected_names,
            "selected_image_names": selected_names,
            "image_idx": selected_scene_indices,
            "image_idx_coordinate_system": "full_target_scene_order",
            "audit_active_from_selection": True,
            "quality_score_path": str(quality_path),
            "quality_aware_selection": {
                "version": 3,
                "joint_objective": "0.35 quality + 0.35 weighted_3d_coverage + 0.20 pose_coverage + 0.10 crossview_pair",
                "hard_gate_active_count": len(names),
                "selected_chart_indices": selected_scene_indices,
                "selected_reliabilities": {
                    name: float(chart_scores[name]["reliability"]) for name in selected_names
                },
                "exclusive_coverage": {
                    name: float(exclusive[index]) for index, name in zip(selected, selected_names)
                },
                "constraints": {
                    "view_clusters": n_clusters,
                    "min_views_per_cluster": min_support,
                    "min_views_per_cluster_source": min_support_source,
                    "min_views_per_sequence": min_sequence_support,
                    "min_views_per_sequence_source": min_sequence_support_source,
                    "sequence_coverage_mode": sequence_coverage_mode,
                    "min_baseline_ratio": min_baseline,
                    "max_borrowed_support_distance": max_borrowed,
                    "min_reference_support_requested": args.min_reference_support,
                    "diagnostics": diagnostics["constraints"],
                },
                "objective": diagnostics["objective"],
                "local_swaps": diagnostics["local_swaps"],
            },
        }
        path = args.output_dir / f"chart_selection_quality_n{count}.json"
        path.write_text(json.dumps(selection_payload, indent=2))
        results.append(
            {
                "count": count,
                "path": str(path),
                "selected_names": selected_names,
                **diagnostics["objective"],
            }
        )

    # Re-write after selected-set marginal coverage has been added to the
    # sidecar.  The training initializer only needs ``reliability`` but the
    # extra fields make selection decisions auditable.
    quality_payload["chart_scores"] = chart_scores
    quality_path.write_text(json.dumps(quality_payload, indent=2))
    summary_path = args.output_dir / "quality_aware_selection_summary.json"
    summary_path.write_text(json.dumps({"results": results}, indent=2))
    print(
        json.dumps(
            {
                "active_charts": len(names),
                "quality_scores": str(quality_path),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
