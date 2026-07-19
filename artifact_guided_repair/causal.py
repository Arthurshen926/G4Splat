"""Primitive-level causal diagnostics for a frozen 2D Gaussian scene.

The module deliberately keeps a 2D artifact mask in image space.  It calls the
mask an *anomaly-ray set* and only creates a 3D repair region after primitive
attribution, a counterfactual intervention, and independent real-image
geometry agree.  The first attribution backend is masked differentiation; the
JSON schema records this explicitly so a future rasterizer ``T * alpha``
accumulator can replace it without changing downstream decisions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.spatial import cKDTree
import torch

from .core import CameraGeometry, fit_support_plane, project_points, robust_affine_color


def _normalize(value: np.ndarray, quantile: float = 0.95) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        return np.zeros_like(value)
    scale = max(float(np.quantile(finite, quantile)), 1e-6)
    return np.clip(value / scale, 0.0, 1.0)


def build_anomaly_rays(
    render_rgb: np.ndarray,
    target_rgb: np.ndarray,
    alpha: np.ndarray,
    depth: np.ndarray,
    semantic_keep: np.ndarray,
    *,
    distortion: np.ndarray | None = None,
    expected_depth: np.ndarray | None = None,
    median_depth: np.ndarray | None = None,
    residual_threshold: float = 0.12,
    score_threshold: float = 0.38,
    min_component_pixels: int = 48,
) -> dict[str, Any]:
    """Detect anomalous *rays*, without asserting a 3D ROI.

    Exposure is fitted only once on static valid pixels.  The anomaly score
    joins photometric, structural, renderer-distortion, low-alpha, and weak
    expected-vs-median depth-layer ambiguity evidence.
    Connected components are reported solely for visual audit and target-mask
    extraction; their component identity is never used as a 3D identity.
    """
    render = np.asarray(render_rgb, dtype=np.float32)
    target = np.asarray(target_rgb, dtype=np.float32)
    keep = np.asarray(semantic_keep, dtype=bool)
    alpha = np.asarray(alpha, dtype=np.float32).squeeze()
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if render.shape != target.shape or render.ndim != 3 or render.shape[2] != 3:
        raise ValueError("render_rgb and target_rgb must be equally-sized HxWx3 images")
    height, width = render.shape[:2]
    if keep.shape != (height, width) or alpha.shape != (height, width):
        raise ValueError("semantic_keep/alpha do not match image dimensions")
    if depth.shape != (height, width):
        depth = np.zeros((height, width), dtype=np.float32)

    fit_mask = keep & np.isfinite(depth) & (depth > 1e-6) & (alpha > 0.30)
    corrected, affine = robust_affine_color(render, target, fit_mask)
    rgb_error = np.mean(np.abs(corrected - target), axis=2)
    corrected_u8 = np.uint8(np.clip(corrected, 0.0, 1.0) * 255.0)
    target_u8 = np.uint8(np.clip(target, 0.0, 1.0) * 255.0)
    corrected_gray = cv2.cvtColor(corrected_u8, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target_u8, cv2.COLOR_RGB2GRAY)
    gx_r = cv2.Sobel(corrected_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_r = cv2.Sobel(corrected_gray, cv2.CV_32F, 0, 1, ksize=3)
    gx_t = cv2.Sobel(target_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_t = cv2.Sobel(target_gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_error = np.hypot(gx_r - gx_t, gy_r - gy_t) / 255.0
    target_gradient = np.hypot(gx_t, gy_t) / 255.0

    if distortion is None:
        distortion_score = np.zeros((height, width), dtype=np.float32)
    else:
        distortion_value = np.asarray(distortion, dtype=np.float32).squeeze()
        if distortion_value.shape != (height, width):
            distortion_value = cv2.resize(
                distortion_value, (width, height), interpolation=cv2.INTER_LINEAR
            )
        distortion_score = _normalize(np.abs(distortion_value), 0.97)

    # Expected and median compositing depths disagree when a ray mixes
    # materially separated layers.  That is helpful for ranking a residual
    # ray that may contain a semi-transparent floater, but it is *not* an
    # independent surface measurement: both maps come from the model under
    # diagnosis.  It can therefore only be a weak anomaly term, never input
    # to Chart/track/plane recovery.
    if expected_depth is None or median_depth is None:
        depth_layer_ambiguity = np.zeros((height, width), dtype=np.float32)
    else:
        expected = np.asarray(expected_depth, dtype=np.float32).squeeze()
        median = np.asarray(median_depth, dtype=np.float32).squeeze()
        if expected.shape != (height, width):
            expected = cv2.resize(expected, (width, height), interpolation=cv2.INTER_LINEAR)
        if median.shape != (height, width):
            median = cv2.resize(median, (width, height), interpolation=cv2.INTER_LINEAR)
        valid_depth_layers = (
            np.isfinite(expected)
            & np.isfinite(median)
            & (expected > 1e-6)
            & (median > 1e-6)
            & (alpha > 0.20)
        )
        depth_layer_ambiguity = np.zeros((height, width), dtype=np.float32)
        denominator = np.maximum(np.maximum(np.abs(expected), np.abs(median)), 1e-6)
        depth_layer_ambiguity[valid_depth_layers] = np.clip(
            np.abs(expected[valid_depth_layers] - median[valid_depth_layers])
            / denominator[valid_depth_layers],
            0.0,
            1.0,
        )
    low_alpha_edge = (alpha < 0.50) & (target_gradient > 0.06)
    score = (
        0.54 * _normalize(rgb_error, 0.95)
        + 0.23 * _normalize(gradient_error, 0.97)
        + 0.10 * distortion_score
        + 0.07 * low_alpha_edge.astype(np.float32)
        + 0.06 * _normalize(depth_layer_ambiguity, 0.95)
    )
    score *= keep.astype(np.float32)
    raw = keep & (
        ((rgb_error >= float(residual_threshold)) & (score >= float(score_threshold)))
        | ((score >= 0.55) & low_alpha_edge)
    )
    # Small morphology joins noisy raster samples, but no large closing is used:
    # a broad 2D component is not evidence of a common 3D cause.
    raw = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8), 8)
    anomaly = np.zeros_like(raw, dtype=bool)
    components: list[dict[str, Any]] = []
    label_map = np.zeros_like(labels, dtype=np.uint16)
    component_id = 1
    for label in range(1, count):
        pixels = int(stats[label, cv2.CC_STAT_AREA])
        if pixels < int(min_component_pixels):
            continue
        mask = labels == label
        anomaly[mask] = True
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        label_map[mask] = component_id
        components.append(
            {
                "component_id": component_id,
                "pixel_count": pixels,
                "area_fraction": float(pixels / (height * width)),
                "bbox_xyxy": [x, y, x + w, y + h],
                "score_mean": float(score[mask].mean()),
                "rgb_error_mean": float(rgb_error[mask].mean()),
                "alpha_mean": float(alpha[mask].mean()),
                "distortion_mean": float(distortion_score[mask].mean()),
                "depth_layer_ambiguity_mean": float(depth_layer_ambiguity[mask].mean()),
            }
        )
        component_id += 1
    return {
        "mask": anomaly,
        "labels": label_map,
        "components": components,
        "score": score.astype(np.float32),
        "rgb_error": rgb_error.astype(np.float32),
        "depth_layer_ambiguity": depth_layer_ambiguity.astype(np.float32),
        "corrected_render": corrected.astype(np.float32),
        "affine": affine.astype(np.float32),
        "summary": {
            "anomaly_ray_fraction": float(anomaly.mean()),
            "component_count": len(components),
            "mean_static_rgb_error": float(rgb_error[keep].mean()) if np.any(keep) else 0.0,
            "mean_anomaly_rgb_error": float(rgb_error[anomaly].mean()) if np.any(anomaly) else 0.0,
            "mean_anomaly_depth_layer_ambiguity": float(depth_layer_ambiguity[anomaly].mean()) if np.any(anomaly) else 0.0,
        },
    }


def affine_correct_torch(render: torch.Tensor, affine: np.ndarray) -> torch.Tensor:
    """Apply a detached per-channel affine calibration to a renderer tensor."""
    calibration = torch.as_tensor(affine, dtype=render.dtype, device=render.device)
    gain = calibration[:, 0].view(3, 1, 1)
    bias = calibration[:, 1].view(3, 1, 1)
    return torch.clamp(render * gain + bias, 0.0, 1.0)


def masked_charbonnier(
    render: torch.Tensor,
    target: torch.Tensor,
    anomaly_mask: torch.Tensor,
    score: torch.Tensor | None = None,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Differentiable anomaly-ray loss for the minimum viable attribution path."""
    if anomaly_mask.dtype != torch.bool:
        anomaly_mask = anomaly_mask > 0
    if not torch.any(anomaly_mask):
        return render.sum() * 0.0
    error = torch.sqrt((render - target).square() + float(epsilon) ** 2).mean(dim=0)
    if score is None:
        weights = anomaly_mask.to(error.dtype)
    else:
        weights = anomaly_mask.to(error.dtype) * torch.clamp(score, min=0.05)
    return (error * weights).sum() / weights.sum().clamp_min(1.0)


def counterfactual_causal_rays(
    baseline_rgb: np.ndarray,
    counterfactual_rgb: np.ndarray,
    anomaly_rays: np.ndarray,
    *,
    minimum_influence: float = 0.01,
    causal_quantile: float = 0.50,
    surface_core_quantile: float = 0.90,
    minimum_causal_pixels: int = 32,
    minimum_surface_pixels: int = 32,
) -> dict[str, Any]:
    """Split anomaly rays by a cluster's observed counterfactual influence.

    The first mask records every materially affected anomaly ray.  The second
    ``surface_core_mask`` is intentionally tighter and is the only mask a
    later Chart/track surface recovery may consume.  Neither mask is a 3-D
    ROI: they are intervention-conditioned image rays that still require
    independent geometry evidence.
    """
    baseline = np.asarray(baseline_rgb, dtype=np.float32)
    intervention = np.asarray(counterfactual_rgb, dtype=np.float32)
    anomaly = np.asarray(anomaly_rays, dtype=bool)
    if baseline.shape != intervention.shape or baseline.ndim != 3 or baseline.shape[-1] != 3:
        raise ValueError("baseline_rgb and counterfactual_rgb must be equally-sized HxWx3")
    if anomaly.shape != baseline.shape[:2]:
        raise ValueError("anomaly_rays do not match RGB dimensions")
    influence = np.mean(np.abs(baseline - intervention), axis=2)
    values = influence[anomaly]
    if not len(values):
        return {
            "influence": influence.astype(np.float32),
            "causal_mask": np.zeros_like(anomaly),
            "surface_core_mask": np.zeros_like(anomaly),
            "causal_threshold": None,
            "surface_core_threshold": None,
            "causal_fallback_to_full_anomaly": True,
            "surface_core_fallback_to_causal": True,
        }
    causal_threshold = max(float(minimum_influence), float(np.quantile(values, causal_quantile)))
    causal = anomaly & (influence >= causal_threshold)
    causal_fallback = int(causal.sum()) < int(minimum_causal_pixels)
    if causal_fallback:
        causal = anomaly.copy()
    core_threshold = max(float(minimum_influence), float(np.quantile(values, surface_core_quantile)))
    surface_core = anomaly & (influence >= core_threshold)
    core_fallback = float(np.max(values)) < float(minimum_influence) or int(surface_core.sum()) < int(minimum_surface_pixels)
    if core_fallback:
        surface_core = causal.copy()
    return {
        "influence": influence.astype(np.float32),
        "causal_mask": causal,
        "surface_core_mask": surface_core,
        "causal_threshold": causal_threshold,
        "surface_core_threshold": core_threshold,
        "causal_fallback_to_full_anomaly": bool(causal_fallback),
        "surface_core_fallback_to_causal": bool(core_fallback),
    }


def select_full_train_anomaly_targets(
    records: Iterable[dict[str, Any]],
    *,
    count: int,
) -> list[str]:
    """Select causal-diagnostic targets from an all-real-view prescan.

    The prescan is deliberately only a *ranking* pass: it has RGB and
    structural residuals from every real training image, but not the full
    renderer buffers required to infer a 3-D cause.  This helper keeps that
    distinction explicit.  A selected image still has to go through the
    alpha/depth-aware anomaly-ray extraction, primitive attribution, and
    counterfactual test before it can authorize an edit.

    ``records`` are JSON-friendly dictionaries containing ``name``,
    ``anomaly_ray_pixel_count`` and ``priority``.  A deterministic ordering is
    important because a full 1487-image audit should be reproducible even when
    two views have nearly equal residuals.
    """
    if int(count) < 0:
        raise ValueError("count must be non-negative")
    eligible = [
        item
        for item in records
        if str(item.get("name", ""))
        and int(item.get("anomaly_ray_pixel_count", 0)) > 0
        and math.isfinite(float(item.get("priority", 0.0)))
    ]
    ranked = sorted(
        eligible,
        key=lambda item: (
            -float(item.get("priority", 0.0)),
            -int(item.get("anomaly_ray_pixel_count", 0)),
            str(item["name"]),
        ),
    )
    return [str(item["name"]) for item in ranked[: int(count)]]


def select_diverse_full_train_anomaly_targets(
    records: Iterable[dict[str, Any]],
    view_features: dict[str, np.ndarray],
    *,
    count: int,
    candidate_multiplier: int = 12,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Greedily select high-residual targets with real-camera coverage.

    The all-train prescan deliberately has only residual evidence, so it
    should not turn model depth or virtual-render confidence into a geometry
    prior.  It *can*, however, avoid spending every expensive attribution pass
    on nearly identical neighbouring frames.  This selector therefore ranks
    the residual-positive views first, limits the pool to high-priority
    candidates, then balances residual strength against pose/direction novelty
    and sequence repetition.  The caller persists the returned rows as an
    audit trail; no row selected here is allowed to authorize a 3-D edit.

    ``view_features`` is intentionally a generic, pre-normalized real-camera
    feature (for this project: centre plus a lightly weighted forward axis),
    so the policy remains independent of the 2DGS scene being diagnosed.
    """
    if int(count) < 0:
        raise ValueError("count must be non-negative")
    if int(candidate_multiplier) <= 0:
        raise ValueError("candidate_multiplier must be positive")
    if int(count) == 0:
        return [], []

    ranked_records = [
        item
        for item in records
        if str(item.get("name", ""))
        and int(item.get("anomaly_ray_pixel_count", 0)) > 0
        and math.isfinite(float(item.get("priority", 0.0)))
        and str(item.get("name", "")) in view_features
        and np.asarray(view_features[str(item.get("name", ""))]).ndim == 1
        and np.all(np.isfinite(np.asarray(view_features[str(item.get("name", ""))], dtype=np.float64)))
    ]
    ranked_records.sort(
        key=lambda item: (
            -float(item.get("priority", 0.0)),
            -int(item.get("anomaly_ray_pixel_count", 0)),
            str(item["name"]),
        )
    )
    if not ranked_records:
        return [], []

    pool_size = min(len(ranked_records), max(32, int(count) * int(candidate_multiplier)))
    pool = ranked_records[:pool_size]
    vectors = np.stack(
        [np.asarray(view_features[str(item["name"])], dtype=np.float64) for item in pool]
    )
    # Coverage is evaluated against every residual-positive real camera, not
    # just against the other candidates.  That makes a selected target useful
    # when it explains a previously distant *set* of anomaly views, rather
    # than merely being far from the immediately preceding frame.
    coverage_vectors = np.stack(
        [np.asarray(view_features[str(item["name"])], dtype=np.float64) for item in ranked_records]
    )
    # Normalize the residual term robustly.  A rank-free range prevents a
    # single exposure-failed frame from assigning zero value to all remaining
    # high-residual candidates, while retaining the deterministic priority
    # ordering as the leading objective term.
    priorities = np.asarray([float(item.get("priority", 0.0)) for item in pool], dtype=np.float64)
    low, high = np.quantile(priorities, [0.05, 0.95]) if len(priorities) > 1 else (priorities[0], priorities[0])
    priority_scale = max(float(high - low), 1e-8)
    priority_scores = np.clip((priorities - low) / priority_scale, 0.0, 1.0)
    coverage_priorities = np.asarray(
        [float(item.get("priority", 0.0)) for item in ranked_records], dtype=np.float64
    )
    coverage_low, coverage_high = (
        np.quantile(coverage_priorities, [0.05, 0.95])
        if len(coverage_priorities) > 1
        else (coverage_priorities[0], coverage_priorities[0])
    )
    coverage_priority_scale = max(float(coverage_high - coverage_low), 1e-8)
    # Even a moderate residual camera contributes to coverage, but the
    # highest-residual rays carry three times the mass of the floor term.
    coverage_weights = 0.25 + 0.75 * np.clip(
        (coverage_priorities - coverage_low) / coverage_priority_scale,
        0.0,
        1.0,
    )
    rank_by_name = {str(item["name"]): index + 1 for index, item in enumerate(ranked_records)}

    selected_indices: list[int] = []
    sequence_counts: dict[str, int] = {}
    audit_rows: list[dict[str, Any]] = []
    nearest_coverage_distance: np.ndarray | None = None
    for step in range(min(int(count), len(pool))):
        candidates = [index for index in range(len(pool)) if index not in selected_indices]
        if not selected_indices:
            chosen = candidates[0]
            minimum_distance = None
            coverage_gain = 0.0
            coverage_gain_raw = 0.0
            sequence_count = 0
            sequence_novelty = 1.0
            objective = float(priority_scores[chosen])
        else:
            if nearest_coverage_distance is None:
                raise RuntimeError("Coverage state is missing after initial target selection")
            candidates_with_gain: list[tuple[int, float, float, int, float]] = []
            chosen_vectors = vectors[np.asarray(selected_indices, dtype=np.int64)]
            for index in candidates:
                minimum_distance = float(
                    np.linalg.norm(chosen_vectors - vectors[index][None, :], axis=1).min()
                )
                candidate_distance = np.linalg.norm(
                    coverage_vectors - vectors[index][None, :], axis=1
                )
                reduced_distance = np.minimum(nearest_coverage_distance, candidate_distance)
                denominator = max(
                    float(np.sum(coverage_weights * nearest_coverage_distance)), 1e-12
                )
                coverage_gain_raw = float(
                    np.sum(coverage_weights * (nearest_coverage_distance - reduced_distance))
                    / denominator
                )
                sequence = str(pool[index]["name"]).split("__", 1)[0]
                sequence_count = int(sequence_counts.get(sequence, 0))
                sequence_novelty = 1.0 / float(1 + sequence_count)
                candidates_with_gain.append(
                    (index, minimum_distance, coverage_gain_raw, sequence_count, sequence_novelty)
                )
            max_coverage_gain = max(item[2] for item in candidates_with_gain)
            objectives: list[tuple[float, int, float, float, float, int, float]] = []
            for index, minimum_distance, coverage_gain_raw, sequence_count, sequence_novelty in candidates_with_gain:
                coverage_gain = float(
                    coverage_gain_raw / max(max_coverage_gain, 1e-12)
                )
                objective = float(
                    # The candidate pool already constrains all choices to
                    # high residual views. Within that pool, a second nearly
                    # identical frame should not beat a view that adds broad
                    # weighted real-camera coverage for a tiny residual
                    # difference.
                    0.45 * priority_scores[index]
                    + 0.40 * coverage_gain
                    + 0.15 * sequence_novelty
                )
                objectives.append(
                    (
                        objective,
                        index,
                        minimum_distance,
                        coverage_gain,
                        coverage_gain_raw,
                        sequence_count,
                        sequence_novelty,
                    )
                )
            # A stable tie-breaker preserves the input residual order, then
            # the camera name, so repeated audits pick the same expensive
            # attribution batch.
            objectives.sort(key=lambda item: (-item[0], -priority_scores[item[1]], str(pool[item[1]]["name"])))
            (
                objective,
                chosen,
                minimum_distance,
                coverage_gain,
                coverage_gain_raw,
                sequence_count,
                sequence_novelty,
            ) = objectives[0]
        item = pool[chosen]
        name = str(item["name"])
        sequence = name.split("__", 1)[0]
        selected_indices.append(chosen)
        selected_distance = np.linalg.norm(
            coverage_vectors - vectors[chosen][None, :], axis=1
        )
        nearest_coverage_distance = (
            selected_distance
            if nearest_coverage_distance is None
            else np.minimum(nearest_coverage_distance, selected_distance)
        )
        sequence_counts[sequence] = sequence_counts.get(sequence, 0) + 1
        audit_rows.append(
            {
                "selection_step": int(step + 1),
                "name": name,
                "priority_rank_in_all_positive_prescan_records": int(rank_by_name[name]),
                "priority": float(item.get("priority", 0.0)),
                "priority_normalized_within_high_residual_pool": float(priority_scores[chosen]),
                "minimum_real_camera_feature_distance_to_previous_selection": minimum_distance,
                "marginal_weighted_real_camera_coverage_gain": float(coverage_gain),
                "marginal_weighted_real_camera_coverage_gain_raw": float(coverage_gain_raw),
                "same_sequence_selected_before": int(sequence_count),
                "sequence_novelty": float(sequence_novelty),
                "joint_objective": float(objective),
            }
        )
    return [str(pool[index]["name"]) for index in selected_indices], audit_rows


def gradient_primitive_attribution(
    gaussians: Any,
    render_package: dict[str, torch.Tensor],
    *,
    max_candidates: int = 384,
    opacity_suppression_factor: float = 0.05,
    min_screen_radius: float = 1.0,
    xyz_weight: float = 0.20,
) -> dict[str, Any]:
    """Rank candidates from masked opacity/position gradients.

    A positive raw-opacity gradient means decreasing opacity decreases the
    masked loss locally.  This is only a candidate generator; the caller must
    counterfactually render each resulting cluster before accepting an edit.
    """
    raw_opacity_grad = gaussians._opacity.grad
    xyz_grad = gaussians._xyz.grad
    if raw_opacity_grad is None:
        raise RuntimeError("Masked attribution did not produce an opacity gradient")
    opacity_grad = raw_opacity_grad.detach().reshape(-1)
    position = (
        xyz_grad.detach().norm(dim=1)
        if xyz_grad is not None
        else torch.zeros_like(opacity_grad)
    )
    visible = render_package["visibility_filter"].detach().reshape(-1)
    radii = render_package["radii"].detach().reshape(-1)
    logits = gaussians._opacity.detach().reshape(-1)
    old_opacity = torch.sigmoid(logits)
    new_opacity = torch.clamp(
        old_opacity * float(opacity_suppression_factor), 1e-5, 1.0 - 1e-5
    )
    logit_delta = (logits - torch.logit(new_opacity)).clamp_min(0.0)
    opacity_benefit = torch.relu(opacity_grad) * logit_delta
    eligible = visible & (radii >= float(min_screen_radius)) & (opacity_benefit > 0)
    if torch.any(eligible):
        normalizer = torch.quantile(position[eligible], 0.95).clamp_min(1e-12)
    else:
        normalizer = position.new_tensor(1.0)
    score = opacity_benefit + float(xyz_weight) * (position / normalizer) * eligible
    score = torch.where(eligible, score, torch.zeros_like(score))
    positive = int(torch.count_nonzero(score > 0).item())
    count = min(int(max_candidates), positive)
    if count == 0:
        return {
            "primitive_ids": np.empty(0, dtype=np.int64),
            "score": np.empty(0, dtype=np.float32),
            "opacity_benefit": np.empty(0, dtype=np.float32),
            "position_gradient": np.empty(0, dtype=np.float32),
            "screen_radius": np.empty(0, dtype=np.float32),
            "positive_candidate_count": 0,
        }
    values, ids = torch.topk(score, count, largest=True, sorted=True)
    return {
        "primitive_ids": ids.detach().cpu().numpy().astype(np.int64),
        "score": values.detach().cpu().numpy().astype(np.float32),
        "opacity_benefit": opacity_benefit[ids].detach().cpu().numpy().astype(np.float32),
        "position_gradient": position[ids].detach().cpu().numpy().astype(np.float32),
        "screen_radius": radii[ids].detach().cpu().numpy().astype(np.float32),
        "positive_candidate_count": positive,
    }


@dataclass
class PrimitiveEvidence:
    primitive_id: int
    score: float = 0.0
    opacity_benefit: float = 0.0
    position_gradient: float = 0.0
    views: set[str] | None = None
    per_view_score: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.views is None:
            self.views = set()
        if self.per_view_score is None:
            self.per_view_score = {}

    def add(self, view: str, score: float, opacity_benefit: float, position_gradient: float) -> None:
        self.score += float(score)
        self.opacity_benefit += float(opacity_benefit)
        self.position_gradient = max(self.position_gradient, float(position_gradient))
        self.views.add(str(view))
        self.per_view_score[str(view)] = self.per_view_score.get(str(view), 0.0) + float(score)


def aggregate_attribution(
    records: Iterable[tuple[str, dict[str, Any]]],
) -> dict[int, PrimitiveEvidence]:
    output: dict[int, PrimitiveEvidence] = {}
    for view_name, result in records:
        ids = np.asarray(result["primitive_ids"], dtype=np.int64)
        scores = np.asarray(result["score"], dtype=np.float64)
        benefits = np.asarray(result["opacity_benefit"], dtype=np.float64)
        positions = np.asarray(result["position_gradient"], dtype=np.float64)
        for primitive_id, score, benefit, position in zip(ids, scores, benefits, positions):
            evidence = output.setdefault(int(primitive_id), PrimitiveEvidence(int(primitive_id)))
            evidence.add(view_name, float(score), float(benefit), float(position))
    return output


class _UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        first, second = self.find(first), self.find(second)
        if first != second:
            self.parent[second] = first


def _component_diameter(points: np.ndarray) -> float:
    """Return the Euclidean diameter of a small candidate component."""
    value = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(value) <= 1:
        return 0.0
    return float(np.max(np.linalg.norm(value[:, None] - value[None, :], axis=2)))


def _split_spatial_chain(
    member_indices: list[int],
    points: np.ndarray,
    *,
    max_diameter: float,
) -> list[list[int]]:
    """Break single-linkage chains into compact counterfactual interventions.

    Graph connectivity is useful evidence that two primitives co-contribute to
    anomalous rays, but a long same-view chain is not evidence that they form
    one removable cause.  In particular, a facade can produce a connected
    candidate graph over many metres simply because all of its surfels occur in
    one target image.  A counterfactual needs a compact intervention, so split
    any over-wide component with deterministic farthest-point bisection until
    every child fits the diameter budget.  This preserves all candidates while
    preventing an image-space co-visibility edge from silently becoming a
    scene-scale deletion.
    """
    pending = [sorted(map(int, member_indices))]
    compact: list[list[int]] = []
    while pending:
        members = pending.pop()
        member_points = np.asarray(points, dtype=np.float64)[members]
        if len(members) <= 1 or _component_diameter(member_points) <= float(max_diameter):
            compact.append(members)
            continue
        distances = np.linalg.norm(
            member_points[:, None] - member_points[None, :], axis=2
        )
        first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
        if first == second or float(distances[first, second]) <= 1e-12:
            compact.append(members)
            continue
        to_first = distances[:, first]
        to_second = distances[:, second]
        # Stable ties make the partition reproducible across runs.
        left_local = (to_first < to_second) | (
            np.isclose(to_first, to_second) & (np.asarray(members) <= int(members[first]))
        )
        left = [members[index] for index in np.flatnonzero(left_local)]
        right = [members[index] for index in np.flatnonzero(~left_local)]
        if not left or not right:
            compact.append(members)
            continue
        pending.extend([left, right])
    return sorted(compact, key=lambda values: (min(values), len(values)))


def cluster_attributed_primitives(
    evidence: dict[int, PrimitiveEvidence],
    xyz: np.ndarray,
    normals: np.ndarray,
    scales: np.ndarray,
    *,
    max_primitives: int = 1600,
    max_cluster_primitives: int = 160,
    normal_cosine: float = 0.20,
    max_component_diameter_factor: float = 2.5,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Group candidates by co-contribution, 3D proximity, and normal stability.

    ``max_component_diameter_factor`` limits the final intervention diameter
    relative to the adaptive local graph radius.  It is deliberately applied
    after graph construction: shared anomaly views may connect a real chain,
    but no later opacity-zero test should delete the entire chain as one cause.
    """
    if float(max_component_diameter_factor) <= 0.0:
        raise ValueError("max_component_diameter_factor must be positive")
    ranked = sorted(evidence.values(), key=lambda item: item.score, reverse=True)[:max_primitives]
    if not ranked:
        return [], {"candidate_count": 0.0, "radius": 0.0}
    ids = np.asarray([item.primitive_id for item in ranked], dtype=np.int64)
    points = np.asarray(xyz, dtype=np.float64)[ids]
    normal_values = np.asarray(normals, dtype=np.float64)[ids]
    scale_values = np.asarray(scales, dtype=np.float64)[ids]
    normal_values /= np.linalg.norm(normal_values, axis=1, keepdims=True).clip(min=1e-12)
    local_scale = np.maximum(np.linalg.norm(scale_values, axis=1), 1e-6)
    tree = cKDTree(points)
    if len(points) > 1:
        nearest = tree.query(points, k=2)[0][:, 1]
        nearest = nearest[np.isfinite(nearest) & (nearest > 1e-9)]
    else:
        nearest = np.empty(0, dtype=np.float64)
    scene_extent = max(float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))), 1e-5)
    nearest_radius = float(np.quantile(nearest, 0.75) * 2.5) if len(nearest) else 0.0
    scale_radius = float(np.quantile(local_scale, 0.75) * 4.0)
    radius = float(np.clip(max(nearest_radius, scale_radius, scene_extent * 5e-4), scene_extent * 1e-5, scene_extent * 0.04))
    unions = _UnionFind(len(ids))
    for first, second in tree.query_pairs(radius):
        first_evidence, second_evidence = ranked[first], ranked[second]
        shared_views = bool(first_evidence.views & second_evidence.views)
        distance = float(np.linalg.norm(points[first] - points[second]))
        local_radius = max(radius * 0.45, 3.0 * max(local_scale[first], local_scale[second]))
        if not shared_views and distance > local_radius:
            continue
        if abs(float(np.dot(normal_values[first], normal_values[second]))) < float(normal_cosine):
            continue
        unions.union(first, second)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(ids)):
        grouped[unions.find(index)].append(index)
    max_component_diameter = max(float(radius) * float(max_component_diameter_factor), 1e-6)
    clusters = []
    pre_compaction_component_count = len(grouped)
    spatial_split_count = 0
    for parent_component_id, members in grouped.items():
        compact_groups = _split_spatial_chain(
            members,
            points,
            max_diameter=max_component_diameter,
        )
        spatial_split_count += max(len(compact_groups) - 1, 0)
        for spatial_partition_index, compact_members in enumerate(compact_groups):
            ordered = sorted(compact_members, key=lambda index: ranked[index].score, reverse=True)
            kept = ordered[:max_cluster_primitives]
            member_ids = ids[kept]
            member_points = points[kept]
            views = sorted(set().union(*(ranked[index].views for index in kept)))
            score = float(sum(ranked[index].score for index in kept))
            clusters.append(
                {
                    "cluster_id": int(parent_component_id),
                    "parent_connectivity_component_id": int(parent_component_id),
                    "spatial_partition_index": int(spatial_partition_index),
                    "primitive_ids": member_ids.astype(np.int64),
                    "score": score,
                    "opacity_benefit": float(sum(ranked[index].opacity_benefit for index in kept)),
                    "view_names": views,
                    "primitive_count": int(len(member_ids)),
                    "center_world": np.median(member_points, axis=0),
                    "diameter_world": _component_diameter(member_points),
                    "position_gradient_max": float(max(ranked[index].position_gradient for index in kept)),
                }
            )
    clusters.sort(key=lambda item: (-float(item["score"]), int(item["cluster_id"])))
    for index, cluster in enumerate(clusters):
        cluster["cluster_id"] = index
    return clusters, {
        "candidate_count": float(len(ids)),
        "radius": radius,
        "scene_extent": scene_extent,
        "pre_compaction_component_count": float(pre_compaction_component_count),
        "spatial_chain_split_count": float(spatial_split_count),
        "max_component_diameter": max_component_diameter,
        "max_component_diameter_factor": float(max_component_diameter_factor),
        "cluster_count": float(len(clusters)),
    }


def _reprojection_error(point: np.ndarray, observations: list[tuple[CameraGeometry, np.ndarray]]) -> np.ndarray:
    errors = []
    for camera, pixel in observations:
        projected, _, inside = project_points(point[None], camera)
        if not bool(inside[0]):
            errors.append(float("inf"))
        else:
            errors.append(float(np.linalg.norm(projected[0] - pixel)))
    return np.asarray(errors, dtype=np.float64)


def _triangulate_multiview(observations: list[tuple[CameraGeometry, np.ndarray]]) -> np.ndarray | None:
    if len(observations) < 2:
        return None
    rows = []
    for camera, pixel in observations:
        projection = camera.projection
        x, y = map(float, pixel)
        rows.extend([x * projection[2] - projection[0], y * projection[2] - projection[1]])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64), full_matrices=False)
    homogeneous = vh[-1]
    if abs(float(homogeneous[3])) <= 1e-10:
        return None
    point = homogeneous[:3] / homogeneous[3]
    return point if np.isfinite(point).all() else None


def build_multiview_feature_tracks(
    target_rgb: np.ndarray,
    target_mask: np.ndarray,
    target_camera: CameraGeometry,
    supports: list[tuple[str, np.ndarray, np.ndarray, CameraGeometry]],
    *,
    max_features: int = 2500,
    ratio_threshold: float = 0.75,
    max_reprojection_error: float = 3.0,
    min_parallax_degrees: float = 0.5,
    min_support_observations: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build target-anchored tracks and require at least three real observations.

    This is intentionally stronger than concatenating pairwise SIFT
    triangulations: a point has one target keypoint identity, multiple support
    observations, a single multi-view DLT estimate, cheirality, and a joint
    reprojection check.
    """
    target = np.uint8(np.clip(target_rgb, 0.0, 1.0) * 255.0)
    target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    target_mask_u8 = np.uint8(np.asarray(target_mask, dtype=bool)) * 255
    sift = cv2.SIFT_create(nfeatures=int(max_features), contrastThreshold=0.02)
    keypoints_target, descriptors_target = sift.detectAndCompute(target_gray, target_mask_u8)
    metrics: dict[str, Any] = {
        "target_keypoint_count": len(keypoints_target),
        "support_count": len(supports),
        "pair_ratio_matches": {},
        "track_count_before_geometry": 0,
        "accepted_track_count": 0,
        "rejected_track_count": 0,
        "method": "target_anchored_sift_union_find_equivalent_multiview_dlt",
    }
    if descriptors_target is None or len(descriptors_target) < 2:
        return np.empty((0, 3), dtype=np.float64), metrics
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    tracks: dict[int, list[tuple[CameraGeometry, np.ndarray]]] = {
        index: [(target_camera, np.asarray(point.pt, dtype=np.float64))]
        for index, point in enumerate(keypoints_target)
    }
    for support_name, support_rgb, support_mask, support_camera in supports:
        support = np.uint8(np.clip(support_rgb, 0.0, 1.0) * 255.0)
        support_gray = cv2.cvtColor(support, cv2.COLOR_RGB2GRAY)
        support_mask_u8 = np.uint8(np.asarray(support_mask, dtype=bool)) * 255
        keypoints, descriptors = sift.detectAndCompute(support_gray, support_mask_u8)
        if descriptors is None or len(descriptors) < 2:
            metrics["pair_ratio_matches"][support_name] = 0
            continue
        pairs = matcher.knnMatch(descriptors_target, descriptors, k=2)
        good = [first for first, second in pairs if first.distance < float(ratio_threshold) * second.distance]
        metrics["pair_ratio_matches"][support_name] = len(good)
        for match in good:
            tracks[int(match.queryIdx)].append(
                (support_camera, np.asarray(keypoints[match.trainIdx].pt, dtype=np.float64))
            )
    candidate_tracks = [
        observations
        for observations in tracks.values()
        if len(observations) >= int(min_support_observations) + 1
    ]
    metrics["track_count_before_geometry"] = len(candidate_tracks)
    points: list[np.ndarray] = []
    reprojections: list[float] = []
    parallaxes: list[float] = []
    for observations in candidate_tracks:
        point = _triangulate_multiview(observations)
        if point is None:
            metrics["rejected_track_count"] += 1
            continue
        _, depths, inside = project_points(point[None], target_camera)
        if not bool(inside[0]) or float(depths[0]) <= 0:
            metrics["rejected_track_count"] += 1
            continue
        positive = True
        for camera, _ in observations[1:]:
            _, support_depth, support_inside = project_points(point[None], camera)
            positive &= bool(support_inside[0]) and float(support_depth[0]) > 0
        if not positive:
            metrics["rejected_track_count"] += 1
            continue
        errors = _reprojection_error(point, observations)
        rays = np.asarray([point - camera.center for camera, _ in observations])
        rays /= np.linalg.norm(rays, axis=1, keepdims=True).clip(min=1e-12)
        pair_angles = np.rad2deg(np.arccos(np.clip(rays @ rays.T, -1.0, 1.0)))
        parallax = float(np.max(pair_angles))
        if not np.isfinite(errors).all() or float(np.max(errors)) > float(max_reprojection_error) or parallax < float(min_parallax_degrees):
            metrics["rejected_track_count"] += 1
            continue
        points.append(point)
        reprojections.append(float(np.median(errors)))
        parallaxes.append(parallax)
    output = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    metrics.update(
        {
            "accepted_track_count": int(len(output)),
            "median_reprojection_error": float(np.median(reprojections)) if reprojections else None,
            "median_max_parallax_degrees": float(np.median(parallaxes)) if parallaxes else None,
        }
    )
    return output, metrics


def recover_planar_surface(
    points: np.ndarray,
    *,
    min_points: int = 8,
    min_inlier_fraction: float = 0.60,
    max_normalized_residual: float = 0.03,
) -> tuple[np.ndarray | None, np.ndarray, dict[str, Any]]:
    """Fit a local plane only when independent multi-view tracks warrant it."""
    try:
        plane, inliers, metrics = fit_support_plane(points, min_points=int(min_points))
    except ValueError as error:
        return None, np.zeros(len(points), dtype=bool), {"accepted": False, "reason": str(error)}
    accepted = (
        float(metrics["inlier_fraction"]) >= float(min_inlier_fraction)
        and float(metrics["normalized_median_residual"]) <= float(max_normalized_residual)
    )
    return (
        plane if accepted else None,
        inliers,
        {**metrics, "accepted": bool(accepted), "reason": None if accepted else "plane_quality_gate"},
    )


def build_plane_footprint(
    points: np.ndarray,
    plane: np.ndarray,
    inliers: np.ndarray | None = None,
    *,
    samples_per_edge: int = 4,
    max_edge_multiple: float = 3.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Turn sparse planar evidence into a bounded, density-filtered footprint.

    The return value is a point sampling of a local 2-D alpha-like footprint,
    not a world AABB or the convex hull of all observations.  We triangulate in
    plane coordinates, reject triangles that bridge gaps larger than the local
    feature-track spacing, then sample only the remaining triangles.  It is a
    deliberately conservative fallback for a full alpha-shape implementation:
    an unsupported gap stays a gap when projected into another camera.
    """
    value = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    valid = np.isfinite(value).all(axis=1)
    if inliers is not None:
        support = np.asarray(inliers, dtype=bool).reshape(-1)
        if len(support) != len(value):
            raise ValueError("Plane inlier mask does not match point count")
        valid &= support
    support_points = value[valid]
    plane = np.asarray(plane, dtype=np.float64).reshape(4)
    normal = plane[:3]
    normal /= np.linalg.norm(normal).clip(min=1e-12)
    # Ensure numerical plane consistency even if the DLT points have tiny
    # signed-depth noise.
    signed = support_points @ normal + float(plane[3])
    support_points = support_points - signed[:, None] * normal[None]
    if len(support_points) < 3:
        return support_points, {
            "method": "sparse_track_points_no_triangulation",
            "input_point_count": int(len(support_points)),
            "footprint_point_count": int(len(support_points)),
        }

    anchor = support_points.mean(axis=0)
    seed_axis = np.array([1.0, 0.0, 0.0])
    if abs(float(seed_axis @ normal)) > 0.9:
        seed_axis = np.array([0.0, 1.0, 0.0])
    axis_u = seed_axis - normal * float(seed_axis @ normal)
    axis_u /= np.linalg.norm(axis_u).clip(min=1e-12)
    axis_v = np.cross(normal, axis_u)
    uv = np.stack([(support_points - anchor) @ axis_u, (support_points - anchor) @ axis_v], axis=1)
    tree = cKDTree(uv)
    nearest = tree.query(uv, k=2)[0][:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 1e-9)]
    if not len(nearest):
        return support_points, {
            "method": "degenerate_track_points_no_triangulation",
            "input_point_count": int(len(support_points)),
            "footprint_point_count": int(len(support_points)),
        }
    spacing = float(np.median(nearest))
    max_edge = max(float(max_edge_multiple) * spacing, 1e-5)
    try:
        from scipy.spatial import Delaunay

        simplices = Delaunay(uv).simplices
    except Exception as error:  # Qhull errors are evidence of a degenerate patch.
        return support_points, {
            "method": "delaunay_failed_sparse_track_points",
            "reason": str(error),
            "input_point_count": int(len(support_points)),
            "footprint_point_count": int(len(support_points)),
            "median_spacing": spacing,
        }

    accepted_triangles = []
    for triangle in simplices:
        triangle_uv = uv[triangle]
        edges = np.linalg.norm(
            triangle_uv[:, None, :] - triangle_uv[None, :, :], axis=-1
        )
        area = 0.5 * abs(
            np.cross(triangle_uv[1] - triangle_uv[0], triangle_uv[2] - triangle_uv[0])
        )
        if float(edges.max()) <= max_edge and area >= max(spacing * spacing * 0.01, 1e-10):
            accepted_triangles.append(np.asarray(triangle, dtype=np.int64))
    if not accepted_triangles:
        return support_points, {
            "method": "all_delaunay_bridges_rejected_sparse_track_points",
            "input_point_count": int(len(support_points)),
            "footprint_point_count": int(len(support_points)),
            "triangle_count": int(len(simplices)),
            "accepted_triangle_count": 0,
            "median_spacing": spacing,
            "max_edge": max_edge,
        }

    resolution = max(1, int(samples_per_edge))
    barycentric = []
    for first in range(resolution + 1):
        for second in range(resolution + 1 - first):
            third = resolution - first - second
            barycentric.append(np.asarray([first, second, third], dtype=np.float64) / resolution)
    samples = [support_points]
    for triangle in accepted_triangles:
        samples.append(np.asarray(barycentric) @ support_points[triangle])
    footprint = np.concatenate(samples, axis=0)
    # Pixel-sized duplicates from adjacent triangles make no geometric
    # difference but inflate projected masks and support statistics.
    rounded = np.round(footprint / max(spacing * 0.05, 1e-6)).astype(np.int64)
    _, unique_ids = np.unique(rounded, axis=0, return_index=True)
    footprint = footprint[np.sort(unique_ids)]
    return footprint, {
        "method": "density_filtered_planar_delaunay_footprint",
        "input_point_count": int(len(support_points)),
        "footprint_point_count": int(len(footprint)),
        "triangle_count": int(len(simplices)),
        "accepted_triangle_count": int(len(accepted_triangles)),
        "median_spacing": spacing,
        "max_edge": max_edge,
        "samples_per_edge": resolution,
        "plane_anchor": anchor,
        "plane_axis_u": axis_u,
        "plane_axis_v": axis_v,
    }


_COLMAP_CAMERA_PARAMETER_COUNTS = {
    0: 3,   # SIMPLE_PINHOLE
    1: 4,   # PINHOLE
    2: 4,   # SIMPLE_RADIAL
    3: 5,   # RADIAL
    4: 8,   # OPENCV
    5: 8,   # OPENCV_FISHEYE
    6: 12,  # FULL_OPENCV
    7: 5,   # FOV
    8: 4,   # SIMPLE_RADIAL_FISHEYE
    9: 5,   # RADIAL_FISHEYE
    10: 12, # THIN_PRISM_FISHEYE
}


def _colmap_name(value: str) -> str:
    """Normalize a COLMAP image path to the 2DGS camera-name convention."""
    text = str(value).replace("\\", "/")
    suffix = Path(text).suffix
    if suffix:
        text = text[: -len(suffix)]
    return text.replace("/", "__")


def _read_colmap_camera_shapes(path: Path) -> dict[int, tuple[int, int]]:
    """Read only image sizes from a COLMAP ``cameras.bin`` file."""
    shapes: dict[int, tuple[int, int]] = {}
    with path.open("rb") as handle:
        raw_count = handle.read(8)
        if len(raw_count) != 8:
            raise ValueError("COLMAP cameras.bin is truncated")
        for _ in range(struct.unpack("<Q", raw_count)[0]):
            raw = handle.read(24)
            if len(raw) != 24:
                raise ValueError("COLMAP cameras.bin has a truncated camera record")
            camera_id, model_id, width, height = struct.unpack("<iiQQ", raw)
            parameter_count = _COLMAP_CAMERA_PARAMETER_COUNTS.get(int(model_id))
            if parameter_count is None:
                raise ValueError(f"Unsupported COLMAP camera model id: {model_id}")
            handle.seek(8 * parameter_count, 1)
            shapes[int(camera_id)] = (int(width), int(height))
    return shapes


def _colmap_qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP's scalar-first unit quaternion to a world-to-camera rotation."""
    value = np.asarray(qvec, dtype=np.float64).reshape(4)
    value /= np.linalg.norm(value).clip(min=1e-12)
    qw, qx, qy, qz = value
    return np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _read_colmap_images(path: Path) -> dict[int, dict[str, Any]]:
    """Read sparse image observations without importing a global COLMAP SDK."""
    output: dict[int, dict[str, Any]] = {}
    point_dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("point_id", "<i8")])
    with path.open("rb") as handle:
        raw_count = handle.read(8)
        if len(raw_count) != 8:
            raise ValueError("COLMAP images.bin is truncated")
        for _ in range(struct.unpack("<Q", raw_count)[0]):
            raw = handle.read(64)
            if len(raw) != 64:
                raise ValueError("COLMAP images.bin has a truncated image record")
            values = struct.unpack("<idddddddi", raw)
            image_id, camera_id = int(values[0]), int(values[-1])
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = _colmap_qvec_to_rotmat(np.asarray(values[1:5], dtype=np.float64))
            w2c[:3, 3] = np.asarray(values[5:8], dtype=np.float64)
            name_bytes = bytearray()
            while True:
                character = handle.read(1)
                if not character:
                    raise ValueError("COLMAP images.bin has an unterminated image name")
                if character == b"\x00":
                    break
                name_bytes.extend(character)
            raw_observation_count = handle.read(8)
            if len(raw_observation_count) != 8:
                raise ValueError("COLMAP images.bin has a truncated observation count")
            observation_count = int(struct.unpack("<Q", raw_observation_count)[0])
            raw_observations = handle.read(24 * observation_count)
            if len(raw_observations) != 24 * observation_count:
                raise ValueError("COLMAP images.bin has truncated image observations")
            values_array = np.frombuffer(raw_observations, dtype=point_dtype)
            output[image_id] = {
                "name": _colmap_name(name_bytes.decode("utf-8")),
                "camera_id": camera_id,
                "w2c": w2c,
                "xys": np.stack([values_array["x"], values_array["y"]], axis=1).astype(np.float64, copy=False),
                "point_ids": values_array["point_id"].astype(np.int64, copy=False),
            }
    return output


def _scale_colmap_pixels(
    pixels: np.ndarray,
    source_shape: tuple[int, int],
    camera: CameraGeometry,
) -> np.ndarray:
    """Map full-resolution COLMAP pixels to the actual 2DGS render raster."""
    source_width, source_height = source_shape
    value = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    output = value.copy()
    output[:, 0] = (output[:, 0] + 0.5) * float(camera.width) / float(source_width) - 0.5
    output[:, 1] = (output[:, 1] + 0.5) * float(camera.height) / float(source_height) - 0.5
    return output


def _colmap_pose_frame_audit(
    images: dict[int, dict[str, Any]],
    real_cameras: dict[str, CameraGeometry],
    *,
    min_views: int = 2,
    max_rotation_p90_degrees: float = 1.0,
    max_center_relative_p90: float = 0.02,
) -> dict[str, Any]:
    """Verify that a sparse model and frozen 2DGS share a world frame."""
    angles: list[float] = []
    center_errors: list[float] = []
    common = []
    for image in images.values():
        name = str(image["name"])
        camera = real_cameras.get(name)
        if camera is None:
            continue
        sparse_w2c = np.asarray(image["w2c"], dtype=np.float64)
        sparse_center = np.linalg.inv(sparse_w2c)[:3, 3]
        relative_rotation = sparse_w2c[:3, :3] @ camera.w2c[:3, :3].T
        cosine = np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
        angles.append(float(np.rad2deg(np.arccos(cosine))))
        center_errors.append(float(np.linalg.norm(sparse_center - camera.center)))
        common.append(camera.center)
    extent = (
        max(float(np.linalg.norm(np.max(common, axis=0) - np.min(common, axis=0))), 1e-6)
        if common
        else 1e-6
    )
    rotation_p90 = float(np.quantile(angles, 0.90)) if angles else None
    center_p90 = float(np.quantile(center_errors, 0.90)) if center_errors else None
    center_relative_p90 = None if center_p90 is None else float(center_p90 / extent)
    accepted = bool(
        len(common) >= int(min_views)
        and rotation_p90 is not None
        and center_relative_p90 is not None
        and rotation_p90 <= float(max_rotation_p90_degrees)
        and center_relative_p90 <= float(max_center_relative_p90)
    )
    return {
        "method": "colmap_camera_pose_to_frozen_2dgs_coordinate_audit",
        "common_camera_count": int(len(common)),
        "rotation_p90_degrees": rotation_p90,
        "center_error_p90": center_p90,
        "center_relative_p90": center_relative_p90,
        "camera_extent": extent,
        "thresholds": {
            "min_views": int(min_views),
            "max_rotation_p90_degrees": float(max_rotation_p90_degrees),
            "max_center_relative_p90": float(max_center_relative_p90),
        },
        "accepted": accepted,
    }


def sample_colmap_track_surface_points(
    sparse_path: str | Path,
    target_camera: CameraGeometry,
    target_anomaly_rays: np.ndarray,
    real_cameras: dict[str, CameraGeometry],
    *,
    source_static_masks: dict[str, np.ndarray] | None = None,
    min_support_observations: int = 2,
    min_parallax_degrees: float = 0.5,
    max_reprojection_p90_pixels: float = 4.0,
    max_reprojection_pixels: float = 8.0,
    max_points: int = 12_000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover independent real-image surface anchors from COLMAP tracks.

    Unlike the dense MAtCha point maps, these are original feature tracks: a
    returned point must be observed by the target anomaly rays and at least two
    other static real training images.  The function streams ``points3D.bin``
    rather than materializing millions of points; when an export retained
    point-to-image associations it reads only target-linked records, and when
    it did not it projects target-visible tracks to the anomaly rays.  A
    camera-frame audit against the frozen 2DGS cameras is mandatory before its
    points can become a plane/footprint input.
    """
    sparse = Path(sparse_path).expanduser().resolve()
    images_path = sparse / "images.bin"
    cameras_path = sparse / "cameras.bin"
    points_path = sparse / "points3D.bin"
    empty = np.empty((0, 3), dtype=np.float64)
    report: dict[str, Any] = {
        "method": "original_colmap_multiview_feature_tracks",
        "sparse_path": str(sparse),
        "available": False,
        "accepted": False,
        "reason": None,
        "source_semantic_mask_applied": bool(source_static_masks is not None),
        "required_support_observations": int(min_support_observations),
    }
    if not images_path.is_file() or not cameras_path.is_file() or not points_path.is_file():
        report["reason"] = "colmap_sparse_images_cameras_or_points_missing"
        return empty, report
    anomaly = np.asarray(target_anomaly_rays, dtype=bool)
    if anomaly.shape != (target_camera.height, target_camera.width):
        raise ValueError("target_anomaly_rays do not match target camera")
    if int(min_support_observations) < 2:
        raise ValueError("COLMAP evidence must require two independent support views")
    try:
        camera_shapes = _read_colmap_camera_shapes(cameras_path)
        images = _read_colmap_images(images_path)
    except (OSError, ValueError, struct.error) as error:
        report["reason"] = "colmap_sparse_metadata_load_failed"
        report["error"] = str(error)
        return empty, report
    by_name = {value["name"]: (image_id, value) for image_id, value in images.items()}
    target_entry = by_name.get(target_camera.name)
    if target_entry is None:
        report["reason"] = "target_camera_absent_from_colmap_sparse_model"
        return empty, report
    target_image_id, target_image = target_entry
    target_shape = camera_shapes.get(int(target_image["camera_id"]))
    if target_shape is None:
        report["reason"] = "target_colmap_camera_shape_missing"
        return empty, report
    pose_audit = _colmap_pose_frame_audit(images, real_cameras)
    target_pixels = _scale_colmap_pixels(target_image["xys"], target_shape, target_camera)
    rounded = np.rint(target_pixels).astype(np.int64)
    has_point2d_associations = bool(
        len(target_image["point_ids"]) and np.any(target_image["point_ids"] > 0)
    )
    candidate_point_ids: set[int] | None = None
    if has_point2d_associations:
        in_target = (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < target_camera.width)
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < target_camera.height)
            & (target_image["point_ids"] > 0)
        )
        target_ray_ids = np.flatnonzero(in_target)
        if len(target_ray_ids):
            keep = anomaly[rounded[target_ray_ids, 1], rounded[target_ray_ids, 0]]
            target_ray_ids = target_ray_ids[keep]
        candidate_point_ids = set(map(int, target_image["point_ids"][target_ray_ids].tolist()))
    report.update(
        {
            "available": True,
            "sparse_image_count": int(len(images)),
            "target_colmap_image_id": int(target_image_id),
            "coordinate_frame_pose_audit": pose_audit,
            "target_track_selection_mode": (
                "stored_target_2d_track_associations"
                if has_point2d_associations
                else "project_all_target_visible_tracks_to_anomaly_rays"
            ),
            "target_anomaly_linked_feature_count": (
                int(len(candidate_point_ids)) if candidate_point_ids is not None else None
            ),
        }
    )
    if not pose_audit["accepted"]:
        report["reason"] = "colmap_to_frozen_2dgs_camera_coordinate_audit_failed"
        return empty, report
    if candidate_point_ids is not None and not candidate_point_ids:
        report["reason"] = "no_colmap_feature_track_intersects_target_anomaly_rays"
        return empty, report

    rows: list[np.ndarray] = []
    reprojection_errors: list[float] = []
    support_counts: list[int] = []
    parallaxes: list[float] = []
    scanned = 0
    target_observed = 0
    static_observed = 0
    try:
        with points_path.open("rb") as handle:
            raw_count = handle.read(8)
            if len(raw_count) != 8:
                raise ValueError("COLMAP points3D.bin is truncated")
            for _ in range(struct.unpack("<Q", raw_count)[0]):
                raw = handle.read(43)
                if len(raw) != 43:
                    raise ValueError("COLMAP points3D.bin has a truncated point record")
                values = struct.unpack("<QdddBBBd", raw)
                point_id = int(values[0])
                raw_track_length = handle.read(8)
                if len(raw_track_length) != 8:
                    raise ValueError("COLMAP points3D.bin has a truncated track length")
                track_length = int(struct.unpack("<Q", raw_track_length)[0])
                raw_track = handle.read(8 * track_length)
                if len(raw_track) != 8 * track_length:
                    raise ValueError("COLMAP points3D.bin has a truncated track")
                if candidate_point_ids is not None and point_id not in candidate_point_ids:
                    continue
                track = np.frombuffer(raw_track, dtype="<i4").reshape(-1, 2)
                if not np.any(track[:, 0] == int(target_image_id)):
                    continue
                scanned += 1
                point = np.asarray(values[1:4], dtype=np.float64)
                # Test target-ray membership before touching support masks.
                # A sparse track graph may connect one target point to many
                # cameras; loading every semantic mask for a point outside the
                # causal core would turn a local repair into an all-view cache.
                target_point2d_index = int(track[np.flatnonzero(track[:, 0] == int(target_image_id))[0], 1])
                projected_target, target_depth, target_inside = project_points(point[None], target_camera)
                if not bool(target_inside[0]) or not np.isfinite(target_depth[0]) or target_depth[0] <= 1e-6:
                    continue
                rounded_target = np.rint(projected_target[0]).astype(np.int64)
                if not (
                    0 <= rounded_target[0] < target_camera.width
                    and 0 <= rounded_target[1] < target_camera.height
                    and bool(anomaly[rounded_target[1], rounded_target[0]])
                ):
                    continue
                target_error: float | None = None
                target_pixel_for_static = rounded_target
                if 0 <= target_point2d_index < len(target_image["xys"]):
                    observed_target = _scale_colmap_pixels(
                        target_image["xys"][target_point2d_index:target_point2d_index + 1],
                        target_shape,
                        target_camera,
                    )[0]
                    rounded_observed_target = np.rint(observed_target).astype(np.int64)
                    if not (
                        0 <= rounded_observed_target[0] < target_camera.width
                        and 0 <= rounded_observed_target[1] < target_camera.height
                    ):
                        continue
                    target_pixel_for_static = rounded_observed_target
                    target_error = float(np.linalg.norm(projected_target[0] - observed_target))
                if source_static_masks is not None:
                    target_static = np.asarray(source_static_masks[target_camera.name], dtype=bool)
                    if target_static.shape != (target_camera.height, target_camera.width):
                        raise ValueError(f"Static mask for COLMAP target {target_camera.name} has the wrong shape")
                    if not bool(target_static[target_pixel_for_static[1], target_pixel_for_static[0]]):
                        continue
                observations: list[tuple[str, CameraGeometry, float | None]] = [
                    (target_camera.name, target_camera, target_error)
                ]
                for image_id, point2d_index in track:
                    if int(image_id) == int(target_image_id):
                        continue
                    image = images.get(int(image_id))
                    if image is None:
                        continue
                    name = str(image["name"])
                    camera = real_cameras.get(name)
                    index = int(point2d_index)
                    source_shape = camera_shapes.get(int(image["camera_id"]))
                    if camera is None or source_shape is None:
                        continue
                    projected, projected_depth, inside = project_points(point[None], camera)
                    if not bool(inside[0]) or not np.isfinite(projected_depth[0]) or projected_depth[0] <= 1e-6:
                        continue
                    rounded_projected = np.rint(projected[0]).astype(np.int64)
                    if not (0 <= rounded_projected[0] < camera.width and 0 <= rounded_projected[1] < camera.height):
                        continue
                    reprojection_error: float | None = None
                    pixel_for_static = rounded_projected
                    if 0 <= index < len(image["xys"]):
                        observed = _scale_colmap_pixels(image["xys"][index:index + 1], source_shape, camera)[0]
                        rounded_observed = np.rint(observed).astype(np.int64)
                        if not (0 <= rounded_observed[0] < camera.width and 0 <= rounded_observed[1] < camera.height):
                            continue
                        pixel_for_static = rounded_observed
                        reprojection_error = float(np.linalg.norm(projected[0] - observed))
                    if source_static_masks is not None:
                        static_mask = np.asarray(source_static_masks[name], dtype=bool)
                        if static_mask.shape != (camera.height, camera.width):
                            raise ValueError(f"Static mask for COLMAP track view {name} has the wrong shape")
                        if not bool(static_mask[pixel_for_static[1], pixel_for_static[0]]):
                            continue
                    observations.append((name, camera, reprojection_error))
                names = {item[0] for item in observations}
                target_observed += 1
                supports = [item for item in observations if item[0] != target_camera.name]
                if len({item[0] for item in supports}) < int(min_support_observations):
                    continue
                static_observed += 1
                errors = np.asarray(
                    [item[2] for item in observations if item[2] is not None], dtype=np.float64
                )
                centers = np.asarray([item[1].center for item in observations], dtype=np.float64)
                rays = point[None] - centers
                rays /= np.linalg.norm(rays, axis=1, keepdims=True).clip(min=1e-12)
                cosine = np.clip(rays @ rays.T, -1.0, 1.0)
                parallax = float(np.rad2deg(np.arccos(cosine)).max())
                if (
                    (len(errors) and not np.isfinite(errors).all())
                    or (len(errors) and float(np.quantile(errors, 0.90)) > float(max_reprojection_p90_pixels))
                    or (len(errors) and float(np.max(errors)) > float(max_reprojection_pixels))
                    or parallax < float(min_parallax_degrees)
                ):
                    continue
                rows.append(point)
                reprojection_errors.extend(errors.tolist())
                support_counts.append(len({item[0] for item in supports}))
                parallaxes.append(parallax)
    except (OSError, ValueError, struct.error) as error:
        report["reason"] = "colmap_sparse_track_load_failed"
        report["error"] = str(error)
        return empty, report
    points = np.asarray(rows, dtype=np.float64).reshape(-1, 3)
    raw_count = len(points)
    if len(points) > int(max_points):
        ids = np.random.default_rng(0).choice(len(points), size=int(max_points), replace=False)
        points = points[ids]
    p90 = float(np.quantile(reprojection_errors, 0.90)) if reprojection_errors else None
    report.update(
        {
            "candidate_point_records_scanned": int(scanned),
            "target_visible_track_count": int(target_observed),
            "target_plus_static_support_track_count": int(static_observed),
            "returned_point_count_before_cap": int(raw_count),
            "returned_point_count": int(len(points)),
            "coordinate_audit_reprojection_p90_pixels": p90,
            "coordinate_audit_reprojection_max_pixels": float(max(reprojection_errors)) if reprojection_errors else None,
            "support_count_p50": float(np.quantile(support_counts, 0.50)) if support_counts else None,
            "support_count_p90": float(np.quantile(support_counts, 0.90)) if support_counts else None,
            "parallax_p50_degrees": float(np.quantile(parallaxes, 0.50)) if parallaxes else None,
            "thresholds": {
                "min_support_observations": int(min_support_observations),
                "min_parallax_degrees": float(min_parallax_degrees),
                "max_reprojection_p90_pixels": float(max_reprojection_p90_pixels),
                "max_reprojection_pixels": float(max_reprojection_pixels),
            },
        }
    )
    # Some Cambridge sparse exports retain the 3-D track graph but omit the
    # per-image xys arrays.  In that case the independently audited camera
    # frame, positive-depth projection, target-ray intersection, and static
    # multi-view track support are the available coordinate checks; do not
    # falsely reject them merely because a deleted 2-D observation cannot be
    # reprojected.  When xys exists, retain the stricter pixel audit too.
    pixel_audit_passed = p90 is None or p90 <= float(max_reprojection_p90_pixels)
    report["accepted"] = bool(
        len(points) >= 8 and bool(pose_audit["accepted"]) and pixel_audit_passed
    )
    report["reason"] = None if report["accepted"] else "insufficient_coordinate_audited_multiview_colmap_tracks"
    return points, report


def _chart_name(value: str) -> str:
    """Normalize a chart image path to the repository camera-name convention."""
    name = Path(str(value).replace("\\", "/")).name
    suffix = Path(name).suffix
    return name[: -len(suffix)] if suffix else name


def _bounded_chart_sample(
    points: np.ndarray,
    chart_ids: np.ndarray,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically cap dense point-map evidence without changing support."""
    if len(points) <= int(max_points):
        return points, chart_ids
    # A fixed RNG makes an evidence report repeatable.  Sampling happens only
    # after target-ray and multi-chart support gates, so it cannot create an
    # unsupported surface.
    ids = np.random.default_rng(0).choice(len(points), size=int(max_points), replace=False)
    return points[ids], chart_ids[ids]


def sample_matcha_chart_surface_points(
    chart_scene: str | Path,
    target_camera: CameraGeometry,
    target_anomaly_rays: np.ndarray,
    real_cameras: dict[str, CameraGeometry],
    *,
    source_static_masks: dict[str, np.ndarray] | None = None,
    chart_stride: int = 4,
    coordinate_audit_stride: int = 32,
    min_chart_confidence: float = 1.0,
    min_chart_views: int = 2,
    target_cell_pixels: int = 8,
    depth_cluster_relative_width: float = 0.08,
    max_points: int = 12_000,
    min_coordinate_audit_charts: int = 2,
    max_coordinate_reprojection_p90_pixels: float = 16.0,
    max_coordinate_depth_relative_p90: float = 0.20,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract independently aligned MAtCha Chart support for anomaly rays.

    ``charts_data.npz`` stores point maps in the MAtCha alignment coordinate
    frame.  Before using them as geometry evidence, this function proves that
    that frame is compatible with the frozen 2DGS camera frame by projecting
    chart samples back into their own real images.  It then projects only
    confidence-gated chart points into a *target anomaly-ray set*.  A target
    cell/depth layer must be supported by points from at least two distinct
    Charts.  No 2DGS depth is consumed here: a bad renderer cannot certify its
    own repair surface.

    The returned points are candidates for a later RANSAC plane and
    density-filtered footprint.  The report is deliberately exhaustive enough
    to make a coordinate-frame or support failure observable rather than
    silently falling back to a 2-D mask.
    """
    scene = Path(chart_scene).expanduser().resolve()
    chart_path = scene / "charts_data.npz"
    cameras_path = scene / "cameras.json"
    empty = np.empty((0, 3), dtype=np.float64)
    report: dict[str, Any] = {
        "method": "aligned_matcha_chart_multiview_support",
        "chart_scene": str(scene),
        "available": False,
        "accepted": False,
        "reason": None,
        "source_semantic_mask_applied": bool(source_static_masks is not None),
    }
    if not chart_path.is_file() or not cameras_path.is_file():
        report["reason"] = "charts_data_or_cameras_json_missing"
        return empty, report
    anomaly = np.asarray(target_anomaly_rays, dtype=bool)
    if anomaly.shape != (target_camera.height, target_camera.width):
        raise ValueError("target_anomaly_rays do not match target camera")
    if int(chart_stride) < 1 or int(coordinate_audit_stride) < 1:
        raise ValueError("Chart sample strides must be positive")
    if int(min_chart_views) < 2:
        raise ValueError("min_chart_views must require independent multi-Chart support")
    if not 0.0 < float(depth_cluster_relative_width) < 1.0:
        raise ValueError("depth_cluster_relative_width must lie in (0, 1)")
    if int(target_cell_pixels) < 1:
        raise ValueError("target_cell_pixels must be positive")

    try:
        metadata = json.loads(cameras_path.read_text(encoding="utf-8"))
        filepaths = list(metadata["filepaths"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["reason"] = "invalid_chart_cameras_json"
        report["error"] = str(error)
        return empty, report
    try:
        archive = np.load(chart_path)
        required = {"pts", "depths", "confs", "scale_factor"}
        missing = sorted(required - set(archive.files))
        if missing:
            report["reason"] = "charts_data_missing_required_arrays"
            report["missing_arrays"] = missing
            return empty, report
        points_all = archive["pts"].astype(np.float64, copy=False)
        depths_all = archive["depths"].astype(np.float64, copy=False)
        confs_all = archive["confs"].astype(np.float64, copy=False)
        scale_factor = float(archive["scale_factor"])
        gate = (
            archive["alignment_gate_valid"].astype(bool, copy=False)
            if "alignment_gate_valid" in archive.files
            else np.ones(len(points_all), dtype=bool)
        )
    except (OSError, ValueError, KeyError) as error:
        report["reason"] = "charts_data_load_failed"
        report["error"] = str(error)
        return empty, report
    finally:
        try:
            archive.close()
        except UnboundLocalError:
            pass

    if (
        points_all.ndim != 4
        or points_all.shape[-1] != 3
        or depths_all.shape != points_all.shape[:3]
        or confs_all.shape != depths_all.shape
        or len(filepaths) != len(points_all)
        or gate.shape != (len(points_all),)
        or not math.isfinite(scale_factor)
        or scale_factor <= 0.0
    ):
        report["reason"] = "charts_data_shape_or_scale_invalid"
        return empty, report
    points_all = points_all / scale_factor
    depths_all = depths_all / scale_factor
    chart_names = [_chart_name(path) for path in filepaths]
    height, width = depths_all.shape[1:]
    active = np.asarray(gate, dtype=bool)
    report.update(
        {
            "available": True,
            "chart_count": int(len(points_all)),
            "active_chart_count": int(active.sum()),
            "chart_height": int(height),
            "chart_width": int(width),
            "scale_factor": scale_factor,
            "min_chart_confidence": float(min_chart_confidence),
            "min_chart_views": int(min_chart_views),
        }
    )
    if not np.any(active):
        report["reason"] = "no_alignment_gate_valid_chart"
        return empty, report

    # Coordinate-frame audit.  Chart pixels have a 512x288-style raster while
    # the real train camera can be a differently scaled version of the same
    # image, hence the half-pixel-respecting scale conversion below.
    audit_rows = []
    audit_y, audit_x = np.mgrid[0:height:int(coordinate_audit_stride), 0:width:int(coordinate_audit_stride)]
    expected_x_base = (audit_x.reshape(-1).astype(np.float64) + 0.5)
    expected_y_base = (audit_y.reshape(-1).astype(np.float64) + 0.5)
    for chart_index, chart_name in enumerate(chart_names):
        if not active[chart_index] or chart_name not in real_cameras:
            continue
        camera = real_cameras[chart_name]
        samples = points_all[chart_index, audit_y, audit_x].reshape(-1, 3)
        chart_depth = depths_all[chart_index, audit_y, audit_x].reshape(-1)
        chart_conf = confs_all[chart_index, audit_y, audit_x].reshape(-1)
        valid = (
            np.isfinite(samples).all(axis=1)
            & np.isfinite(chart_depth)
            & np.isfinite(chart_conf)
            & (chart_depth > 1e-6)
            & (chart_conf >= float(min_chart_confidence))
        )
        if int(valid.sum()) < 16:
            continue
        projected, projected_depth, inside = project_points(samples[valid], camera)
        expected = np.stack(
            [
                expected_x_base[valid] * float(camera.width) / float(width) - 0.5,
                expected_y_base[valid] * float(camera.height) / float(height) - 0.5,
            ],
            axis=1,
        )
        good = inside & np.isfinite(projected_depth) & (projected_depth > 1e-6)
        if int(good.sum()) < 16:
            continue
        reprojection = np.linalg.norm(projected[good] - expected[good], axis=1)
        relative_depth = np.abs(projected_depth[good] - chart_depth[valid][good]) / np.maximum(
            chart_depth[valid][good], 1e-6
        )
        audit_rows.append(
            {
                "chart_index": int(chart_index),
                "chart_name": chart_name,
                "sample_count": int(valid.sum()),
                "inside_count": int(good.sum()),
                "reprojection_p50_pixels": float(np.quantile(reprojection, 0.50)),
                "reprojection_p90_pixels": float(np.quantile(reprojection, 0.90)),
                "depth_relative_p50": float(np.quantile(relative_depth, 0.50)),
                "depth_relative_p90": float(np.quantile(relative_depth, 0.90)),
            }
        )
    report["coordinate_audit"] = audit_rows
    passing_audits = [
        row
        for row in audit_rows
        if row["reprojection_p90_pixels"] <= float(max_coordinate_reprojection_p90_pixels)
        and row["depth_relative_p90"] <= float(max_coordinate_depth_relative_p90)
    ]
    report["coordinate_audit_passing_chart_count"] = int(len(passing_audits))
    report["coordinate_audit_thresholds"] = {
        "min_charts": int(min_coordinate_audit_charts),
        "max_reprojection_p90_pixels": float(max_coordinate_reprojection_p90_pixels),
        "max_depth_relative_p90": float(max_coordinate_depth_relative_p90),
    }
    if len(passing_audits) < int(min_coordinate_audit_charts):
        report["reason"] = "chart_to_real_camera_coordinate_audit_failed"
        return empty, report
    passing_chart_indices = {int(row["chart_index"]) for row in passing_audits}

    sample_y, sample_x = np.mgrid[0:height:int(chart_stride), 0:width:int(chart_stride)]
    projected_points: list[np.ndarray] = []
    projected_cells: list[np.ndarray] = []
    projected_depths: list[np.ndarray] = []
    projected_charts: list[np.ndarray] = []
    for chart_index in sorted(passing_chart_indices):
        samples = points_all[chart_index, sample_y, sample_x].reshape(-1, 3)
        confidence = confs_all[chart_index, sample_y, sample_x].reshape(-1)
        valid = np.isfinite(samples).all(axis=1) & np.isfinite(confidence) & (confidence >= float(min_chart_confidence))
        chart_name = chart_names[chart_index]
        # Alignment masks are intentionally not geometry visibility masks in
        # MAtCha (outside-mask prior geometry is restored).  Apply the real
        # Cambridge static mask again here so an occluding tree/sky chart pixel
        # cannot become independent facade support merely by projection.
        if source_static_masks is not None and chart_name in real_cameras:
            source_mask = np.asarray(source_static_masks[chart_name], dtype=bool)
            source_camera = real_cameras[chart_name]
            if source_mask.shape != (source_camera.height, source_camera.width):
                raise ValueError(f"Static mask for Chart {chart_name} has the wrong shape")
            source_x = np.rint(
                (sample_x.reshape(-1).astype(np.float64) + 0.5)
                * float(source_camera.width)
                / float(width)
                - 0.5
            ).astype(np.int64)
            source_y = np.rint(
                (sample_y.reshape(-1).astype(np.float64) + 0.5)
                * float(source_camera.height)
                / float(height)
                - 0.5
            ).astype(np.int64)
            source_x = np.clip(source_x, 0, source_camera.width - 1)
            source_y = np.clip(source_y, 0, source_camera.height - 1)
            valid &= source_mask[source_y, source_x]
        if not np.any(valid):
            continue
        pixels, depths, inside = project_points(samples[valid], target_camera)
        rounded = np.rint(pixels).astype(np.int64)
        inside &= (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < target_camera.width)
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < target_camera.height)
            & np.isfinite(depths)
            & (depths > 1e-6)
        )
        if not np.any(inside):
            continue
        rounded = rounded[inside]
        target_keep = anomaly[rounded[:, 1], rounded[:, 0]]
        if not np.any(target_keep):
            continue
        rounded = rounded[target_keep]
        projected_points.append(samples[valid][inside][target_keep])
        projected_depths.append(depths[inside][target_keep])
        projected_charts.append(np.full(int(target_keep.sum()), chart_index, dtype=np.int32))
        projected_cells.append(
            np.stack(
                [
                    rounded[:, 0] // int(target_cell_pixels),
                    rounded[:, 1] // int(target_cell_pixels),
                ],
                axis=1,
            ).astype(np.int32)
        )
    if not projected_points:
        report["reason"] = "no_confident_chart_point_projects_to_target_anomaly_rays"
        return empty, report
    points = np.concatenate(projected_points, axis=0)
    cells = np.concatenate(projected_cells, axis=0)
    depths = np.concatenate(projected_depths, axis=0)
    chart_ids = np.concatenate(projected_charts, axis=0)
    report["target_anomaly_projected_point_count_before_multiview"] = int(len(points))

    # Quantized target pixel/depth layers intentionally model *support*, not
    # a 2-D bounding box.  A near/far layer at the same pixel remains separate.
    depth_bins = np.floor(np.log(depths) / math.log1p(float(depth_cluster_relative_width))).astype(np.int32)
    order = np.lexsort((chart_ids, depth_bins, cells[:, 1], cells[:, 0]))
    ordered_cells = cells[order]
    ordered_bins = depth_bins[order]
    ordered_charts = chart_ids[order]
    group_start = np.r_[True, (ordered_cells[1:] != ordered_cells[:-1]).any(axis=1) | (ordered_bins[1:] != ordered_bins[:-1])]
    starts = np.flatnonzero(group_start)
    stops = np.r_[starts[1:], len(order)]
    keep = np.zeros(len(points), dtype=bool)
    support_values = np.zeros(len(points), dtype=np.int16)
    for start, stop in zip(starts, stops):
        source_ids = np.unique(ordered_charts[start:stop])
        if len(source_ids) >= int(min_chart_views):
            member = order[start:stop]
            keep[member] = True
            support_values[member] = len(source_ids)
    if not np.any(keep):
        report["reason"] = "target_anomaly_has_no_multichart_depth_layer"
        return empty, report
    supported_points = points[keep]
    supported_charts = chart_ids[keep]
    supported_values = support_values[keep]
    raw_count = len(supported_points)
    supported_points, supported_charts = _bounded_chart_sample(
        supported_points, supported_charts, max_points=max_points
    )
    report.update(
        {
            "target_anomaly_projected_point_count_after_multiview": int(raw_count),
            "returned_point_count": int(len(supported_points)),
            "returned_source_chart_count": int(len(np.unique(supported_charts))),
            "returned_source_chart_indices": [
                int(index) for index in sorted(set(supported_charts.tolist()))
            ],
            "returned_source_chart_names": [
                chart_names[int(index)] for index in sorted(set(supported_charts.tolist()))
            ],
            "target_cell_pixels": int(target_cell_pixels),
            "depth_cluster_relative_width": float(depth_cluster_relative_width),
            "support_count_p50": float(np.quantile(supported_values, 0.50)),
            "support_count_p90": float(np.quantile(supported_values, 0.90)),
            "accepted": bool(len(supported_points) >= 8),
            "reason": None if len(supported_points) >= 8 else "insufficient_multichart_surface_points",
        }
    )
    return supported_points, report


def virtual_reprojection_consistency(
    source_rgb: np.ndarray,
    source_depth: np.ndarray,
    source_alpha: np.ndarray,
    source_normal_world: np.ndarray,
    source_camera: CameraGeometry,
    target_rgb: np.ndarray,
    target_depth: np.ndarray,
    target_alpha: np.ndarray,
    target_normal_world: np.ndarray,
    target_camera: CameraGeometry,
    *,
    source_mask: np.ndarray | None = None,
    stride: int = 4,
    depth_relative_tolerance: float = 0.10,
) -> dict[str, float]:
    """Measure model stability across two nearby virtual cameras.

    This is deliberately a *diagnostic* rather than a source of geometry or
    supervision.  A point rendered in the first virtual camera is projected
    into the second; only depth-consistent correspondences participate in the
    RGB/normal comparison.  Thus normal image parallax is not counted as a
    flicker, while a rapid visibility/depth switch remains measurable through
    the depth-consistent fraction and depth error tails.
    """
    source = np.asarray(source_rgb, dtype=np.float32)
    target = np.asarray(target_rgb, dtype=np.float32)
    source_depth = np.asarray(source_depth, dtype=np.float32).squeeze()
    target_depth = np.asarray(target_depth, dtype=np.float32).squeeze()
    source_alpha = np.asarray(source_alpha, dtype=np.float32).squeeze()
    target_alpha = np.asarray(target_alpha, dtype=np.float32).squeeze()
    source_normal = np.asarray(source_normal_world, dtype=np.float32)
    target_normal = np.asarray(target_normal_world, dtype=np.float32)
    height, width = source_depth.shape
    if (
        source.shape != (height, width, 3)
        or source_alpha.shape != (height, width)
        or source_normal.shape != (height, width, 3)
        or target.shape != (target_camera.height, target_camera.width, 3)
        or target_depth.shape != (target_camera.height, target_camera.width)
        or target_alpha.shape != (target_camera.height, target_camera.width)
        or target_normal.shape != (target_camera.height, target_camera.width, 3)
        or (height, width) != (source_camera.height, source_camera.width)
    ):
        raise ValueError("Virtual render maps do not match their camera geometries")
    if int(stride) < 1:
        raise ValueError("stride must be positive")

    yy, xx = np.mgrid[0:height:int(stride), 0:width:int(stride)]
    sampled_x = xx.reshape(-1).astype(np.float64)
    sampled_y = yy.reshape(-1).astype(np.float64)
    source_z = source_depth[yy, xx].reshape(-1).astype(np.float64)
    valid = (
        np.isfinite(source_z)
        & (source_z > 1e-6)
        & np.isfinite(source_alpha[yy, xx].reshape(-1))
        & (source_alpha[yy, xx].reshape(-1) >= 0.50)
    )
    if source_mask is not None:
        mask = np.asarray(source_mask, dtype=bool)
        if mask.shape != (height, width):
            raise ValueError("source_mask does not match source render")
        valid &= mask[yy, xx].reshape(-1)
    sampled_count = int(valid.sum())
    empty = {
        "sampled_pixel_count": sampled_count,
        "projected_pixel_count": 0,
        "depth_consistent_pixel_count": 0,
        "depth_consistent_fraction": 0.0,
        "depth_relative_p50": None,
        "depth_relative_p90": None,
        "rgb_mae": None,
        "normal_angle_p90_degrees": None,
    }
    if not sampled_count:
        return empty

    x = sampled_x[valid]
    y = sampled_y[valid]
    z = source_z[valid]
    camera_points = np.stack(
        [
            (x - float(source_camera.cx)) * z / float(source_camera.fx),
            (y - float(source_camera.cy)) * z / float(source_camera.fy),
            z,
        ],
        axis=1,
    )
    homogeneous = np.concatenate(
        [camera_points, np.ones((len(camera_points), 1), dtype=np.float64)], axis=1
    )
    world = (source_camera.c2w @ homogeneous.T).T[:, :3]
    pixels, projected_depth, inside = project_points(world, target_camera)
    rounded = np.rint(pixels).astype(np.int64)
    inside &= (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < target_camera.width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < target_camera.height)
    )
    if not np.any(inside):
        return empty
    source_x = x[inside].astype(np.int64)
    source_y = y[inside].astype(np.int64)
    x2 = rounded[inside, 0]
    y2 = rounded[inside, 1]
    observed_depth = target_depth[y2, x2].astype(np.float64)
    observed_alpha = target_alpha[y2, x2]
    relative = np.abs(projected_depth[inside] - observed_depth) / np.maximum(np.abs(observed_depth), 1e-6)
    target_valid = (
        np.isfinite(observed_depth)
        & (observed_depth > 1e-6)
        & np.isfinite(observed_alpha)
        & (observed_alpha >= 0.50)
    )
    projected_count = int(target_valid.sum())
    consistent = target_valid & (relative <= float(depth_relative_tolerance))
    result = {
        "sampled_pixel_count": sampled_count,
        "projected_pixel_count": projected_count,
        "depth_consistent_pixel_count": int(consistent.sum()),
        "depth_consistent_fraction": float(consistent.sum() / max(sampled_count, 1)),
        "depth_relative_p50": float(np.quantile(relative[target_valid], 0.50)) if projected_count else None,
        "depth_relative_p90": float(np.quantile(relative[target_valid], 0.90)) if projected_count else None,
        "rgb_mae": None,
        "normal_angle_p90_degrees": None,
    }
    if not np.any(consistent):
        return result
    keep_y = y2[consistent]
    keep_x = x2[consistent]
    source_color = source[source_y[consistent], source_x[consistent]]
    target_color = target[keep_y, keep_x]
    result["rgb_mae"] = float(np.mean(np.abs(source_color - target_color)))
    source_normals = source_normal[source_y[consistent], source_x[consistent]]
    target_normals = target_normal[keep_y, keep_x]
    source_norms = np.linalg.norm(source_normals, axis=1)
    target_norms = np.linalg.norm(target_normals, axis=1)
    normal_valid = (source_norms > 1e-6) & (target_norms > 1e-6)
    if np.any(normal_valid):
        cosine = np.sum(
            source_normals[normal_valid] * target_normals[normal_valid], axis=1
        ) / (source_norms[normal_valid] * target_norms[normal_valid])
        # A surfel may flip its arbitrary disk orientation without becoming a
        # different physical plane; compare unoriented normals for stability.
        angle = np.rad2deg(np.arccos(np.clip(np.abs(cosine), 0.0, 1.0)))
        result["normal_angle_p90_degrees"] = float(np.quantile(angle, 0.90))
    return result


def see3d_gate(
    *,
    surface_known: bool,
    real_rgb_support_fraction: float,
    reliable_model_fraction: float,
    max_real_support_fraction: float = 0.15,
    max_reliable_model_fraction: float = 0.20,
) -> dict[str, Any]:
    """Strictly decide whether a residual has the right to reach See3D."""
    eligible = (
        bool(surface_known)
        and float(real_rgb_support_fraction) < float(max_real_support_fraction)
        and float(reliable_model_fraction) < float(max_reliable_model_fraction)
    )
    reasons = []
    if not surface_known:
        reasons.append("surface_not_independently_known")
    if real_rgb_support_fraction >= float(max_real_support_fraction):
        reasons.append("real_rgb_support_available")
    if reliable_model_fraction >= float(max_reliable_model_fraction):
        reasons.append("reliable_model_appearance_available")
    return {
        "eligible": bool(eligible),
        "formula": "surface_known AND not_real_rgb_support AND not_reliable_model_appearance",
        "surface_known": bool(surface_known),
        "real_rgb_support_fraction": float(real_rgb_support_fraction),
        "reliable_model_fraction": float(reliable_model_fraction),
        "reasons_not_eligible": reasons,
        "allowed_parameters_if_generated": ["appearance", "opacity"],
        "geometry_frozen": True,
    }
