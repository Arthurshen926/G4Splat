"""Sequence-consistent metric depth for static foliage ray posteriors.

Per-image monocular inverse-depth alignment is ill-conditioned when a tree
fills most of a Cambridge frame and only a narrow rigid strip remains.  The
fixed camera trajectory supplies a stronger constraint: a static ray birth
projected into a nearby frame must agree with the nearby frame's birth depth.
This module solves those constraints as a robust graph of *continuous* depth
scale corrections.  Independent rigid alignments are retained as soft priors;
they are never converted into a binary camera admission gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree


SEQUENCE_METRIC_DEPTH_VERSION = (
    "sequence-reprojection-static-foliage-metric-depth-v1"
)


@dataclass(frozen=True)
class SequenceDepthCamera:
    """Fixed-camera quantities needed by the reprojection graph."""

    camera_id: int
    sequence: str
    frame: int
    rotation: np.ndarray
    translation: np.ndarray
    center: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class SequenceDepthView:
    """One camera's immutable DAV2 ray births and soft rigid-fit prior."""

    camera: SequenceDepthCamera
    points: np.ndarray
    normalized_uv: np.ndarray
    relative_rmse: float


def _scaled_world_points(view: SequenceDepthView, scale: float) -> np.ndarray:
    center = np.asarray(view.camera.center, dtype=np.float64)
    return center[None] + float(scale) * (
        np.asarray(view.points, dtype=np.float64) - center[None]
    )


def _camera_points(points: np.ndarray, camera: SequenceDepthCamera) -> np.ndarray:
    return (
        np.asarray(points, dtype=np.float64)
        @ np.asarray(camera.rotation, dtype=np.float64).T
        + np.asarray(camera.translation, dtype=np.float64)[None]
    )


def _canonical_pixels(
    camera_points: np.ndarray,
    camera: SequenceDepthCamera,
    *,
    canonical_width: float,
) -> np.ndarray:
    depth = camera_points[:, 2]
    u = camera.fx * camera_points[:, 0] / np.maximum(depth, 1.0e-8)
    v = camera.fy * camera_points[:, 1] / np.maximum(depth, 1.0e-8)
    u = (u + camera.cx + 0.5) / float(camera.width)
    v = (v + camera.cy + 0.5) / float(camera.height)
    canonical_height = canonical_width * camera.height / float(camera.width)
    return np.column_stack([u * canonical_width, v * canonical_height])


def _edge_measurements(
    views: list[SequenceDepthView],
    log_scales: np.ndarray,
    *,
    maximum_frame_gap: int,
    maximum_pixel_distance: float,
    minimum_matches: int,
    canonical_width: float,
) -> list[dict[str, float | int]]:
    """Measure robust adjacent-view log-depth residuals.

    The source point is first transformed into the target camera.  Its
    projected location is then matched against a target exact-ray birth.  A
    low-MAD median over many such pairs is a physical static-scene constraint,
    whereas a raw same-pixel depth comparison would ignore parallax.
    """

    scaled = [
        _scaled_world_points(view, float(np.exp(log_scales[index])))
        for index, view in enumerate(views)
    ]
    target_pixels = []
    target_depths = []
    target_trees = []
    for index, view in enumerate(views):
        points = _camera_points(scaled[index], view.camera)
        height = canonical_width * view.camera.height / float(view.camera.width)
        pixels = np.asarray(view.normalized_uv, dtype=np.float64) * np.asarray(
            [canonical_width, height], dtype=np.float64
        )
        target_pixels.append(pixels)
        target_depths.append(points[:, 2])
        target_trees.append(cKDTree(pixels))

    edges: list[dict[str, float | int]] = []
    for source_index, source in enumerate(views):
        for target_index in range(
            source_index + 1, min(source_index + 4, len(views))
        ):
            target = views[target_index]
            frame_gap = int(target.camera.frame - source.camera.frame)
            if frame_gap <= 0:
                continue
            if frame_gap > int(maximum_frame_gap):
                break
            source_in_target = _camera_points(
                scaled[source_index], target.camera
            )
            source_pixels = _canonical_pixels(
                source_in_target,
                target.camera,
                canonical_width=canonical_width,
            )
            distance, nearest = target_trees[target_index].query(
                source_pixels, k=1
            )
            canonical_height = (
                canonical_width
                * target.camera.height
                / float(target.camera.width)
            )
            valid = (
                np.isfinite(distance)
                & (distance <= float(maximum_pixel_distance))
                & np.isfinite(source_in_target[:, 2])
                & (source_in_target[:, 2] > 0.05)
                & (source_pixels[:, 0] >= 0)
                & (source_pixels[:, 0] < canonical_width)
                & (source_pixels[:, 1] >= 0)
                & (source_pixels[:, 1] < canonical_height)
            )
            match_count = int(valid.sum())
            if match_count < int(minimum_matches):
                continue
            source_depth = source_in_target[valid, 2]
            target_depth = target_depths[target_index][nearest[valid]]
            finite = (
                np.isfinite(target_depth)
                & (target_depth > 0.05)
                & np.isfinite(source_depth)
                & (source_depth > 0.05)
            )
            if int(finite.sum()) < int(minimum_matches):
                continue
            log_ratio = np.log(
                np.clip(source_depth[finite] / target_depth[finite], 1e-3, 1e3)
            )
            median = float(np.median(log_ratio))
            mad = float(np.median(np.abs(log_ratio - median)))
            # Match support and residual spread are continuous authority, not
            # pass/fail quality gates.  A small floor keeps a weak connected
            # edge numerically visible without allowing it to dominate.
            weight = float(
                max(
                    1.0e-6,
                    min(len(log_ratio) / 256.0, 1.0)
                    * np.exp(-0.5 * (mad / 0.15) ** 2)
                    * np.exp(-0.15 * max(frame_gap - 1, 0)),
                )
            )
            edges.append(
                {
                    "source": int(source_index),
                    "target": int(target_index),
                    "frame_gap": frame_gap,
                    "matches": int(len(log_ratio)),
                    "median_log_depth_ratio": median,
                    "mad_log_depth_ratio": mad,
                    "weight": weight,
                }
            )
    return edges


def solve_sequence_depth_scales(
    views: list[SequenceDepthView],
    *,
    maximum_frame_gap: int = 3,
    maximum_pixel_distance: float = 3.0,
    minimum_matches: int = 64,
    canonical_width: float = 640.0,
    iterations: int = 2,
) -> tuple[dict[int, float], dict[str, object]]:
    """Solve continuous per-camera scale corrections for one sequence."""

    ordered = sorted(views, key=lambda value: value.camera.frame)
    if not ordered:
        return {}, {
            "camera_count": 0,
            "edge_count": 0,
            "iterations": 0,
        }
    log_scales = np.zeros(len(ordered), dtype=np.float64)
    iteration_audit: list[dict[str, object]] = []
    for iteration in range(max(1, int(iterations))):
        edges = _edge_measurements(
            ordered,
            log_scales,
            maximum_frame_gap=maximum_frame_gap,
            maximum_pixel_distance=maximum_pixel_distance,
            minimum_matches=minimum_matches,
            canonical_width=canonical_width,
        )
        if not edges:
            break
        system = lil_matrix(
            (len(edges) + len(ordered) + 1, len(ordered)),
            dtype=np.float64,
        )
        target = np.zeros(system.shape[0], dtype=np.float64)
        row = 0
        before = []
        for edge in edges:
            source = int(edge["source"])
            destination = int(edge["target"])
            residual = float(edge["median_log_depth_ratio"])
            weight = float(edge["weight"])
            coefficient = np.sqrt(weight)
            system[row, source] = coefficient
            system[row, destination] = -coefficient
            target[row] = -coefficient * residual
            before.append(abs(residual))
            row += 1
        # A trustworthy rigid fit stays close to its independently measured
        # metric depth.  Tree-dominant, high-RMSE frames retain only a very
        # weak gauge prior and are corrected by the reprojection graph.
        for index, view in enumerate(ordered):
            rmse = max(float(view.relative_rmse), 0.0)
            weight = 2.0 * np.exp(-0.5 * (rmse / 0.15) ** 2) + 0.002
            coefficient = np.sqrt(weight)
            system[row, index] = coefficient
            target[row] = -coefficient * log_scales[index]
            row += 1
        coefficient = 0.03 / np.sqrt(len(ordered))
        system[row, :] = coefficient
        target[row] = -coefficient * float(log_scales.sum())
        delta = lsqr(
            system.tocsr(), target, atol=1.0e-10, btol=1.0e-10
        )[0]
        log_scales += delta
        after = []
        for edge in edges:
            source = int(edge["source"])
            destination = int(edge["target"])
            residual = float(edge["median_log_depth_ratio"])
            after.append(abs(residual + delta[source] - delta[destination]))
        iteration_audit.append(
            {
                "iteration": iteration + 1,
                "edge_count": len(edges),
                "absolute_log_residual_before_quantiles": np.quantile(
                    before, [0.0, 0.5, 0.9, 0.99, 1.0]
                ).tolist(),
                "linearized_absolute_log_residual_after_quantiles": (
                    np.quantile(after, [0.0, 0.5, 0.9, 0.99, 1.0]).tolist()
                ),
            }
        )

    scales = np.exp(log_scales)
    result = {
        int(view.camera.camera_id): float(scales[index])
        for index, view in enumerate(ordered)
    }
    final_edges = _edge_measurements(
        ordered,
        log_scales,
        maximum_frame_gap=maximum_frame_gap,
        maximum_pixel_distance=maximum_pixel_distance,
        minimum_matches=minimum_matches,
        canonical_width=canonical_width,
    )
    final_residual = [
        abs(float(edge["median_log_depth_ratio"])) for edge in final_edges
    ]
    audit: dict[str, object] = {
        "contract": SEQUENCE_METRIC_DEPTH_VERSION,
        "sequence": ordered[0].camera.sequence,
        "camera_count": len(ordered),
        "edge_count": len(final_edges),
        "iterations": len(iteration_audit),
        "maximum_frame_gap": int(maximum_frame_gap),
        "maximum_pixel_distance_at_640": float(maximum_pixel_distance),
        "minimum_correspondences": int(minimum_matches),
        "scale_quantiles": np.quantile(
            scales, [0.0, 0.1, 0.5, 0.9, 1.0]
        ).tolist(),
        "final_absolute_log_depth_residual_quantiles": (
            np.quantile(final_residual, [0.0, 0.5, 0.9, 0.99, 1.0]).tolist()
            if final_residual
            else []
        ),
        "optimization": iteration_audit,
    }
    return result, audit

