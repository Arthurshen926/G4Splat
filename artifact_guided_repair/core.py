from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class CameraGeometry:
    name: str
    width: int
    height: int
    w2c: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        w2c = np.asarray(self.w2c, dtype=np.float64)
        if w2c.shape != (4, 4):
            raise ValueError(f"w2c must be 4x4, got {w2c.shape}")
        object.__setattr__(self, "w2c", w2c)

    @property
    def c2w(self) -> np.ndarray:
        return np.linalg.inv(self.w2c)

    @property
    def center(self) -> np.ndarray:
        return self.c2w[:3, 3]

    @property
    def forward(self) -> np.ndarray:
        return self.c2w[:3, 2]

    @property
    def K(self) -> np.ndarray:
        return np.asarray(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def projection(self) -> np.ndarray:
        return self.K @ self.w2c[:3]

    def as_json(self) -> dict:
        return {
            "name": self.name,
            "width": int(self.width),
            "height": int(self.height),
            "w2c": self.w2c.tolist(),
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
        }

    @classmethod
    def from_json(cls, value: dict) -> "CameraGeometry":
        return cls(
            name=str(value["name"]),
            width=int(value["width"]),
            height=int(value["height"]),
            w2c=np.asarray(value["w2c"], dtype=np.float64),
            fx=float(value["fx"]),
            fy=float(value["fy"]),
            cx=float(value["cx"]),
            cy=float(value["cy"]),
        )


def _as_rgb_float(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {value.shape}")
    value = value.astype(np.float32, copy=False)
    if value.size and float(np.nanmax(value)) > 2.0:
        value = value / 255.0
    return np.clip(value, 0.0, 1.0)


def robust_affine_color(
    render_rgb: np.ndarray,
    target_rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    max_samples: int = 200_000,
    trim_quantile: float = 0.80,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a bounded per-channel affine map while trimming transient residuals."""
    render = _as_rgb_float(render_rgb)
    target = _as_rgb_float(target_rgb)
    if render.shape != target.shape:
        raise ValueError(f"RGB shapes differ: {render.shape} vs {target.shape}")
    keep = np.asarray(valid_mask, dtype=bool)
    if keep.shape != render.shape[:2]:
        raise ValueError(f"Mask shape {keep.shape} does not match RGB {render.shape[:2]}")

    flat_ids = np.flatnonzero(keep.reshape(-1))
    if len(flat_ids) < 32:
        params = np.asarray([[1.0, 0.0]] * 3, dtype=np.float32)
        return render.copy(), params
    if len(flat_ids) > max_samples:
        stride = max(1, len(flat_ids) // max_samples)
        flat_ids = flat_ids[::stride][:max_samples]

    x_all = render.reshape(-1, 3)[flat_ids]
    y_all = target.reshape(-1, 3)[flat_ids]
    params = []
    corrected = np.empty_like(render)
    for channel in range(3):
        x = x_all[:, channel]
        y = y_all[:, channel]
        fit_keep = np.ones(len(x), dtype=bool)
        gain, bias = 1.0, 0.0
        for _ in range(3):
            design = np.stack([x[fit_keep], np.ones(np.count_nonzero(fit_keep))], axis=1)
            if len(design) < 16:
                break
            gain, bias = np.linalg.lstsq(design, y[fit_keep], rcond=None)[0]
            gain = float(np.clip(gain, 0.5, 2.0))
            bias = float(np.clip(bias, -0.25, 0.25))
            residual = np.abs(gain * x + bias - y)
            threshold = float(np.quantile(residual, trim_quantile))
            fit_keep = residual <= max(threshold, 1e-4)
        params.append([gain, bias])
        corrected[..., channel] = np.clip(gain * render[..., channel] + bias, 0.0, 1.0)
    return corrected, np.asarray(params, dtype=np.float32)


def _normalize_map(value: np.ndarray, quantile: float = 0.95) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        return np.zeros_like(value)
    scale = max(float(np.quantile(finite, quantile)), 1e-6)
    return np.clip(value / scale, 0.0, 1.0)


def build_artifact_components(
    render_rgb: np.ndarray,
    target_rgb: np.ndarray,
    alpha: np.ndarray,
    depth: np.ndarray,
    semantic_keep: np.ndarray,
    *,
    no_reference_invalid: Optional[np.ndarray] = None,
    residual_threshold: float = 0.16,
    alpha_threshold: float = 0.50,
    min_area_fraction: float = 0.004,
    max_area_fraction: float = 0.40,
) -> dict:
    """Extract large, coherent train-view rendering failures after exposure fitting."""
    render = _as_rgb_float(render_rgb)
    target = _as_rgb_float(target_rgb)
    height, width = render.shape[:2]
    alpha = np.asarray(alpha, dtype=np.float32).squeeze()
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    semantic_keep = np.asarray(semantic_keep, dtype=bool)
    if alpha.shape != (height, width) or depth.shape != (height, width):
        raise ValueError("alpha/depth shape does not match RGB")
    if semantic_keep.shape != (height, width):
        raise ValueError("semantic mask shape does not match RGB")

    fit_mask = semantic_keep & np.isfinite(depth) & (depth > 1e-6) & (alpha > 0.5)
    corrected, affine = robust_affine_color(render, target, fit_mask)
    residual = np.mean(np.abs(corrected - target), axis=2)
    residual_smooth = cv2.GaussianBlur(residual, (0, 0), 2.0)

    target_gray = cv2.cvtColor(np.uint8(np.clip(target, 0.0, 1.0) * 255), cv2.COLOR_RGB2GRAY)
    render_gray = cv2.cvtColor(np.uint8(np.clip(corrected, 0.0, 1.0) * 255), cv2.COLOR_RGB2GRAY)
    target_grad_x = cv2.Sobel(target_gray, cv2.CV_32F, 1, 0, ksize=3)
    target_grad_y = cv2.Sobel(target_gray, cv2.CV_32F, 0, 1, ksize=3)
    render_grad_x = cv2.Sobel(render_gray, cv2.CV_32F, 1, 0, ksize=3)
    render_grad_y = cv2.Sobel(render_gray, cv2.CV_32F, 0, 1, ksize=3)
    target_grad = np.hypot(target_grad_x, target_grad_y) / 255.0
    gradient_error = np.hypot(target_grad_x - render_grad_x, target_grad_y - render_grad_y) / 255.0

    valid_depth = np.isfinite(depth) & (depth > 1e-6)
    log_depth = np.zeros_like(depth)
    log_depth[valid_depth] = np.log(np.maximum(depth[valid_depth], 1e-6))
    depth_dx = cv2.Sobel(log_depth, cv2.CV_32F, 1, 0, ksize=3)
    depth_dy = cv2.Sobel(log_depth, cv2.CV_32F, 0, 1, ksize=3)
    depth_edge = np.hypot(depth_dx, depth_dy)

    no_ref = np.zeros((height, width), dtype=np.float32)
    if no_reference_invalid is not None:
        no_ref_value = np.asarray(no_reference_invalid)
        if no_ref_value.shape != (height, width):
            no_ref_value = cv2.resize(
                no_ref_value.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST
            )
        no_ref = np.clip(no_ref_value.astype(np.float32), 0.0, 1.0)

    coverage_warning = (alpha < alpha_threshold) & (target_grad > 0.08)
    depth_warning = (_normalize_map(depth_edge, 0.97) > 0.75) & (residual_smooth > 0.10)
    score = (
        0.55 * np.clip((residual_smooth - 0.06) / 0.24, 0.0, 1.0)
        + 0.20 * _normalize_map(gradient_error, 0.97)
        + 0.15 * no_ref
        + 0.10 * coverage_warning.astype(np.float32)
    )
    score = np.clip(score, 0.0, 1.0) * semantic_keep.astype(np.float32)
    candidate = semantic_keep & (
        (residual_smooth >= residual_threshold)
        | coverage_warning
        | depth_warning
        | (no_ref >= 0.5)
    ) & (score >= 0.30)

    kernel3 = np.ones((3, 3), np.uint8)
    kernel9 = np.ones((9, 9), np.uint8)
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN, kernel3)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel9)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    output_labels = np.zeros((height, width), dtype=np.uint16)
    components = []
    total = float(height * width)
    output_id = 1
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        area_fraction = area / total
        if area_fraction < min_area_fraction:
            continue
        oversized = area_fraction > max_area_fraction
        mask = labels == label
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        output_labels[mask] = output_id
        components.append(
            {
                "component_id": output_id,
                "area": area,
                "area_fraction": float(area_fraction),
                "bbox_xyxy": [x, y, x + w, y + h],
                "score_mean": float(score[mask].mean()),
                "residual_mean": float(residual[mask].mean()),
                "alpha_mean": float(alpha[mask].mean()),
                "valid_depth_fraction": float(valid_depth[mask].mean()),
                "coverage_warning_fraction": float(coverage_warning[mask].mean()),
                "depth_warning_fraction": float(depth_warning[mask].mean()),
                "no_reference_invalid_fraction": float((no_ref[mask] >= 0.5).mean()),
                # Large failures must remain visible to diagnostics, but they are
                # too broad and usually too mixed for an automatic local edit.
                "oversized": bool(oversized),
                "repair_eligible": bool(not oversized),
            }
        )
        output_id += 1

    components.sort(
        key=lambda item: (item["score_mean"] * np.sqrt(item["area_fraction"])),
        reverse=True,
    )
    return {
        "corrected_render": corrected,
        "affine": affine,
        "residual": residual,
        "score": score,
        "labels": output_labels,
        "components": components,
        "summary": {
            "component_count": len(components),
            "artifact_fraction": float((output_labels > 0).mean()),
            "repair_eligible_artifact_fraction": float(
                sum(
                    item["area_fraction"]
                    for item in components
                    if item["repair_eligible"]
                )
            ),
            "oversized_component_count": int(
                sum(item["oversized"] for item in components)
            ),
            "largest_component_fraction": float(
                max((item["area_fraction"] for item in components), default=0.0)
            ),
            "residual_mean": float(residual[semantic_keep].mean())
            if np.any(semantic_keep)
            else 0.0,
        },
    }


def backproject_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    camera: CameraGeometry,
    *,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if depth.shape != (camera.height, camera.width) or mask.shape != depth.shape:
        raise ValueError("depth/mask shape does not match camera")
    yy, xx = np.nonzero(mask & np.isfinite(depth) & (depth > 1e-6))
    if stride > 1 and len(xx):
        keep = np.arange(len(xx)) % int(stride) == 0
        xx, yy = xx[keep], yy[keep]
    if len(xx) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    z = depth[yy, xx]
    camera_points = np.stack(
        [
            (xx.astype(np.float64) - camera.cx) / camera.fx * z,
            (yy.astype(np.float64) - camera.cy) / camera.fy * z,
            z,
            np.ones_like(z),
        ],
        axis=1,
    )
    world = camera_points @ camera.c2w.T
    return world[:, :3], np.stack([xx, yy], axis=1).astype(np.float64)


def project_points(
    points_world: np.ndarray,
    camera: CameraGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=bool),
        )
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    view = homogeneous @ camera.w2c.T
    z = view[:, 2]
    safe_z = np.where(np.abs(z) > 1e-12, z, 1e-12)
    pixels = np.stack(
        [camera.fx * view[:, 0] / safe_z + camera.cx, camera.fy * view[:, 1] / safe_z + camera.cy],
        axis=1,
    )
    inside = (
        (z > 1e-6)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] <= camera.width - 1)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] <= camera.height - 1)
    )
    return pixels, z, inside


def depth_support_metrics(
    points_world: np.ndarray,
    camera: CameraGeometry,
    candidate_depth: np.ndarray,
    candidate_alpha: np.ndarray,
    candidate_clean_mask: np.ndarray,
    *,
    relative_depth_threshold: float = 0.08,
) -> dict:
    pixels, projected_depth, inside = project_points(points_world, camera)
    total = max(len(points_world), 1)
    if not np.any(inside):
        return {
            "projected_count": 0,
            "overlap_fraction": 0.0,
            "depth_consistent_fraction": 0.0,
            "clean_fraction": 0.0,
        }
    ids = np.flatnonzero(inside)
    x = np.clip(np.rint(pixels[ids, 0]).astype(np.int64), 0, camera.width - 1)
    y = np.clip(np.rint(pixels[ids, 1]).astype(np.int64), 0, camera.height - 1)
    depth = np.asarray(candidate_depth, dtype=np.float64)[y, x]
    alpha = np.asarray(candidate_alpha, dtype=np.float64)[y, x]
    clean = np.asarray(candidate_clean_mask, dtype=bool)[y, x]
    finite = np.isfinite(depth) & (depth > 1e-6) & (alpha > 0.5)
    relative_error = np.full(len(ids), np.inf, dtype=np.float64)
    denominator = np.maximum(np.maximum(np.abs(depth), np.abs(projected_depth[ids])), 1e-6)
    relative_error[finite] = np.abs(depth[finite] - projected_depth[ids][finite]) / denominator[finite]
    consistent = finite & (relative_error <= relative_depth_threshold)
    return {
        "projected_count": int(len(ids)),
        "overlap_fraction": float(len(ids) / total),
        "depth_consistent_fraction": float(consistent.mean()),
        "clean_fraction": float((clean & finite).mean()),
        "median_relative_depth_error": float(np.median(relative_error[finite]))
        if np.any(finite)
        else float("inf"),
    }


def multiview_depth_supported_points(
    points_world: np.ndarray,
    views: list[tuple[CameraGeometry, np.ndarray, np.ndarray, np.ndarray]],
    *,
    relative_depth_threshold: float = 0.08,
    min_views: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep 3D points that agree with rendered depth in several clean views."""
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return points, np.empty(0, dtype=np.int32)
    support_counts = np.zeros(len(points), dtype=np.int32)
    for camera, depth_map, alpha_map, clean_mask in views:
        pixels, projected_depth, inside = project_points(points, camera)
        ids = np.flatnonzero(inside)
        if len(ids) == 0:
            continue
        x = np.clip(np.rint(pixels[ids, 0]).astype(np.int64), 0, camera.width - 1)
        y = np.clip(np.rint(pixels[ids, 1]).astype(np.int64), 0, camera.height - 1)
        depth = np.asarray(depth_map, dtype=np.float64)[y, x]
        alpha = np.asarray(alpha_map, dtype=np.float64)[y, x]
        clean = np.asarray(clean_mask, dtype=bool)[y, x]
        denominator = np.maximum(np.maximum(np.abs(depth), np.abs(projected_depth[ids])), 1e-6)
        relative_error = np.abs(depth - projected_depth[ids]) / denominator
        supported = (
            np.isfinite(depth)
            & (depth > 1e-6)
            & (alpha > 0.5)
            & clean
            & (relative_error <= relative_depth_threshold)
        )
        support_counts[ids[supported]] += 1
    return points[support_counts >= int(min_views)], support_counts


def multiview_clean_visible_points(
    points_world: np.ndarray,
    views: list[tuple[CameraGeometry, np.ndarray]],
    *,
    min_views: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep points visible in independent clean-image masks.

    Unlike :func:`multiview_depth_supported_points`, this function never reads
    depth rendered by the model under diagnosis. It is therefore suitable for
    validating feature-triangulated geometry without circular evidence.
    """
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return points, np.empty(0, dtype=np.int32)
    support_counts = np.zeros(len(points), dtype=np.int32)
    for camera, clean_mask in views:
        pixels, _, inside = project_points(points, camera)
        ids = np.flatnonzero(inside)
        if len(ids) == 0:
            continue
        x = np.clip(np.rint(pixels[ids, 0]).astype(np.int64), 0, camera.width - 1)
        y = np.clip(np.rint(pixels[ids, 1]).astype(np.int64), 0, camera.height - 1)
        clean = np.asarray(clean_mask, dtype=bool)
        if clean.shape != (camera.height, camera.width):
            raise ValueError(
                f"Clean mask shape {clean.shape} does not match camera "
                f"{camera.name} ({camera.height}, {camera.width})"
            )
        support_counts[ids[clean[y, x]]] += 1
    return points[support_counts >= int(min_views)], support_counts


def match_component_geometry(
    target_rgb: np.ndarray,
    target_mask: np.ndarray,
    target_camera: CameraGeometry,
    candidate_rgb: np.ndarray,
    candidate_mask: np.ndarray,
    candidate_camera: CameraGeometry,
    *,
    max_features: int = 2000,
    ratio_threshold: float = 0.75,
    max_reprojection_error: float = 3.0,
    min_parallax_degrees: float = 0.5,
) -> tuple[dict, np.ndarray]:
    """Match a target component and triangulate it using calibrated cameras."""
    target = np.uint8(_as_rgb_float(target_rgb) * 255)
    candidate = np.uint8(_as_rgb_float(candidate_rgb) * 255)
    target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY)
    target_mask_u8 = np.uint8(np.asarray(target_mask, dtype=bool)) * 255
    candidate_mask_u8 = np.uint8(np.asarray(candidate_mask, dtype=bool)) * 255

    sift = cv2.SIFT_create(nfeatures=int(max_features), contrastThreshold=0.02)
    keypoints_1, descriptors_1 = sift.detectAndCompute(target_gray, target_mask_u8)
    keypoints_2, descriptors_2 = sift.detectAndCompute(candidate_gray, candidate_mask_u8)
    empty_metrics = {
        "target_keypoints": len(keypoints_1),
        "candidate_keypoints": len(keypoints_2),
        "ratio_match_count": 0,
        "triangulated_count": 0,
        "median_reprojection_error": float("inf"),
        "median_parallax_degrees": 0.0,
    }
    if descriptors_1 is None or descriptors_2 is None or len(descriptors_1) < 2 or len(descriptors_2) < 2:
        return empty_metrics, np.empty((0, 3), dtype=np.float64)

    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors_1, descriptors_2, k=2)
    good = [first for first, second in matches if first.distance < ratio_threshold * second.distance]
    if len(good) < 4:
        empty_metrics["ratio_match_count"] = len(good)
        return empty_metrics, np.empty((0, 3), dtype=np.float64)

    points_1 = np.asarray([keypoints_1[item.queryIdx].pt for item in good], dtype=np.float64)
    points_2 = np.asarray([keypoints_2[item.trainIdx].pt for item in good], dtype=np.float64)
    homogeneous = cv2.triangulatePoints(
        target_camera.projection,
        candidate_camera.projection,
        points_1.T,
        points_2.T,
    )
    valid_w = np.abs(homogeneous[3]) > 1e-10
    points_world = np.zeros((len(good), 3), dtype=np.float64)
    points_world[valid_w] = (homogeneous[:3, valid_w] / homogeneous[3:4, valid_w]).T

    projection_1, depth_1, inside_1 = project_points(points_world, target_camera)
    projection_2, depth_2, inside_2 = project_points(points_world, candidate_camera)
    error_1 = np.linalg.norm(projection_1 - points_1, axis=1)
    error_2 = np.linalg.norm(projection_2 - points_2, axis=1)
    reprojection_error = np.maximum(error_1, error_2)
    ray_1 = points_world - target_camera.center[None]
    ray_2 = points_world - candidate_camera.center[None]
    ray_1 /= np.linalg.norm(ray_1, axis=1, keepdims=True).clip(min=1e-12)
    ray_2 /= np.linalg.norm(ray_2, axis=1, keepdims=True).clip(min=1e-12)
    parallax = np.rad2deg(np.arccos(np.clip(np.sum(ray_1 * ray_2, axis=1), -1.0, 1.0)))
    valid = (
        valid_w
        & inside_1
        & inside_2
        & (depth_1 > 0)
        & (depth_2 > 0)
        & np.isfinite(reprojection_error)
        & (reprojection_error <= max_reprojection_error)
        & (parallax >= min_parallax_degrees)
    )
    valid_points = points_world[valid]
    metrics = {
        "target_keypoints": len(keypoints_1),
        "candidate_keypoints": len(keypoints_2),
        "ratio_match_count": len(good),
        "triangulated_count": int(np.count_nonzero(valid)),
        "median_reprojection_error": float(np.median(reprojection_error[valid]))
        if np.any(valid)
        else float("inf"),
        "median_parallax_degrees": float(np.median(parallax[valid]))
        if np.any(valid)
        else 0.0,
    }
    return metrics, valid_points


def interpolate_c2w(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    fraction = float(fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    rotation = Slerp(
        [0.0, 1.0], Rotation.from_matrix(np.stack([first[:3, :3], second[:3, :3]]))
    )([fraction]).as_matrix()[0]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = (1.0 - fraction) * first[:3, 3] + fraction * second[:3, 3]
    return result


def select_wrong_geometry_interpolation(
    scan: list[dict],
    *,
    min_median_relative_depth_error: float = 0.30,
    min_overlap_fraction: float = 0.50,
    min_projected_points: int = 6,
) -> Optional[dict]:
    """Choose the clean-support-nearest pose that still exposes wrong geometry."""
    eligible = []
    for item in scan:
        metrics = item.get("depth_metrics", {})
        median_error = float(metrics.get("median_relative_depth_error", float("inf")))
        overlap = float(metrics.get("overlap_fraction", 0.0))
        projected = int(metrics.get("projected_count", 0))
        fraction = float(item.get("fraction", 0.0))
        if not np.isfinite(median_error):
            continue
        if median_error < float(min_median_relative_depth_error):
            continue
        if overlap < float(min_overlap_fraction) or projected < int(min_projected_points):
            continue
        if not 0.0 < fraction < 1.0:
            continue
        eligible.append(item)
    if not eligible:
        return None
    return max(eligible, key=lambda item: float(item["fraction"]))


def select_conservative_edit_trial(
    trials: list[dict],
    *,
    min_depth_improvement: float,
    max_outside_rgb_mae: float,
    max_validation_psnr_drop: float,
    near_best_ratio: float = 0.95,
) -> Optional[dict]:
    """Select the smallest edit whose gain is near the best quality-gated trial."""
    accepted = []
    for trial in trials:
        if float(trial.get("depth_improvement", -float("inf"))) < float(min_depth_improvement):
            continue
        if float(trial.get("outside_rgb_mae", float("inf"))) > float(max_outside_rgb_mae):
            continue
        if float(trial.get("min_validation_psnr_delta", -float("inf"))) < -float(
            max_validation_psnr_drop
        ):
            continue
        accepted.append(trial)
    if not accepted:
        return None
    best_improvement = max(float(trial["depth_improvement"]) for trial in accepted)
    near_best = [
        trial
        for trial in accepted
        if float(trial["depth_improvement"]) >= float(near_best_ratio) * best_improvement
    ]
    return min(
        near_best,
        key=lambda trial: (
            int(trial.get("edited_points", 0)),
            float(trial.get("outside_rgb_mae", float("inf"))),
        ),
    )


def fit_support_plane(
    points_world: np.ndarray,
    *,
    min_points: int = 6,
    trim_iterations: int = 3,
    ransac_trials: int = 512,
    relative_inlier_threshold: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Robustly fit a local plane to sparse clean-view triangulated points."""
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    unique_count = len(np.unique(np.round(points, decimals=7), axis=0)) if len(points) else 0
    if unique_count < int(min_points):
        raise ValueError(
            f"Need at least {int(min_points)} unique support points, got {unique_count}"
        )

    scene_center = np.median(points, axis=0)
    scene_extent = max(float(np.linalg.norm(points - scene_center, axis=1).max()), 1e-6)
    base_threshold = max(float(relative_inlier_threshold) * scene_extent, 1e-4)
    rng = np.random.default_rng(0)
    best_active = np.zeros(len(points), dtype=bool)
    best_score = (-1, -float("inf"))
    for _ in range(max(1, int(ransac_trials))):
        ids = rng.choice(len(points), size=3, replace=False)
        first, second, third = points[ids]
        normal = np.cross(second - first, third - first)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-10:
            continue
        normal /= norm
        offset = -float(normal @ first)
        residual = np.abs(points @ normal + offset)
        candidate = residual <= base_threshold
        count = int(np.count_nonzero(candidate))
        if count < int(min_points):
            continue
        score = (count, -float(np.median(residual[candidate])))
        if score > best_score:
            best_score = score
            best_active = candidate
    if np.count_nonzero(best_active) < int(min_points):
        raise ValueError("Could not find a non-degenerate support plane")

    active = best_active
    plane = np.zeros(4, dtype=np.float64)
    singular_values = np.zeros(3, dtype=np.float64)
    for _ in range(max(1, int(trim_iterations))):
        selected = points[active]
        center = selected.mean(axis=0)
        _, singular_values, vh = np.linalg.svd(selected - center, full_matrices=False)
        normal = vh[-1]
        normal /= np.linalg.norm(normal).clip(min=1e-12)
        plane = np.concatenate([normal, [-float(normal @ center)]])
        residual = np.abs(points @ plane[:3] + plane[3])
        extent = max(float(np.linalg.norm(selected - center, axis=1).max()), 1e-6)
        active_residual = residual[active]
        threshold = max(
            3.0 * float(np.median(active_residual)),
            base_threshold,
        )
        updated = residual <= threshold
        if np.count_nonzero(updated) < int(min_points) or np.array_equal(updated, active):
            break
        active = updated

    selected = points[active]
    center = selected.mean(axis=0)
    _, singular_values, vh = np.linalg.svd(selected - center, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal).clip(min=1e-12)
    plane = np.concatenate([normal, [-float(normal @ center)]])
    residual = np.abs(points @ plane[:3] + plane[3])
    extent = max(float(np.linalg.norm(selected - center, axis=1).max()), 1e-6)
    metrics = {
        "input_count": int(len(points)),
        "unique_count": int(unique_count),
        "inlier_count": int(np.count_nonzero(active)),
        "inlier_fraction": float(active.mean()),
        "extent": extent,
        "median_absolute_residual": float(np.median(residual[active])),
        "max_absolute_residual": float(np.max(residual[active])),
        "normalized_median_residual": float(np.median(residual[active]) / extent),
        "planarity_ratio": float(
            singular_values[-1] / max(float(singular_values[-2]), 1e-12)
        ),
    }
    return plane, active, metrics


def render_support_plane_depth(
    camera: CameraGeometry,
    plane_world: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect camera rays with a world-space plane and return camera-z depth."""
    plane = np.asarray(plane_world, dtype=np.float64).reshape(4)
    keep = np.asarray(mask, dtype=bool)
    if keep.shape != (camera.height, camera.width):
        raise ValueError(f"Mask shape {keep.shape} does not match camera")

    depth = np.zeros(keep.shape, dtype=np.float32)
    valid = np.zeros(keep.shape, dtype=bool)
    yy, xx = np.nonzero(keep)
    if len(xx) == 0:
        return depth, valid
    rays_camera = np.stack(
        [
            (xx.astype(np.float64) - camera.cx) / camera.fx,
            (yy.astype(np.float64) - camera.cy) / camera.fy,
            np.ones(len(xx), dtype=np.float64),
        ],
        axis=1,
    )
    rays_world = rays_camera @ camera.c2w[:3, :3].T
    denominator = rays_world @ plane[:3]
    numerator = -(float(camera.center @ plane[:3]) + plane[3])
    z = np.full(len(xx), np.nan, dtype=np.float64)
    stable = np.abs(denominator) > 1e-8
    z[stable] = numerator / denominator[stable]
    ray_valid = stable & np.isfinite(z) & (z > 1e-6)
    depth[yy[ray_valid], xx[ray_valid]] = z[ray_valid].astype(np.float32)
    valid[yy[ray_valid], xx[ray_valid]] = True
    return depth, valid


def blend_projective_reference_colors(
    colors: np.ndarray,
    semantic_valid: np.ndarray,
    depth_valid: np.ndarray,
    artifact_scores: np.ndarray,
    *,
    max_fallback_rgb_disagreement: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fuse posed reference RGB, preferring clean depth-consistent observations."""
    colors = np.asarray(colors, dtype=np.float32)
    semantic_valid = np.asarray(semantic_valid, dtype=bool)
    depth_valid = np.asarray(depth_valid, dtype=bool)
    artifact_scores = np.asarray(artifact_scores, dtype=np.float32)
    if colors.ndim != 3 or colors.shape[2] != 3:
        raise ValueError(f"Expected reference colors VxNx3, got {colors.shape}")
    expected = colors.shape[:2]
    if semantic_valid.shape != expected or depth_valid.shape != expected:
        raise ValueError("Reference validity shape does not match colors")
    if artifact_scores.shape != expected:
        raise ValueError("Artifact score shape does not match colors")

    point_count = colors.shape[1]
    output = np.zeros((point_count, 3), dtype=np.float32)
    accepted = np.zeros(point_count, dtype=bool)
    source_kind = np.zeros(point_count, dtype=np.uint8)

    depth_weights = depth_valid.astype(np.float32) / np.maximum(artifact_scores + 0.05, 1e-3)
    depth_weight_sum = depth_weights.sum(axis=0)
    has_depth = depth_weight_sum > 0
    if np.any(has_depth):
        output[has_depth] = (
            (colors * depth_weights[..., None]).sum(axis=0)[has_depth]
            / depth_weight_sum[has_depth, None]
        )
        accepted[has_depth] = True
        source_kind[has_depth] = 1

    fallback_ids = np.flatnonzero(~has_depth & (semantic_valid.sum(axis=0) >= 2))
    fallback_count = 0
    for point_id in fallback_ids:
        view_ids = np.flatnonzero(semantic_valid[:, point_id])
        values = colors[view_ids, point_id]
        disagreement = np.max(np.mean(np.abs(values[:, None] - values[None]), axis=2))
        if disagreement > float(max_fallback_rgb_disagreement):
            continue
        weights = 1.0 / np.maximum(artifact_scores[view_ids, point_id] + 0.05, 1e-3)
        output[point_id] = np.average(values, axis=0, weights=weights)
        accepted[point_id] = True
        source_kind[point_id] = 2
        fallback_count += 1

    diagnostics = {
        "point_count": int(point_count),
        "depth_supported_count": int(np.count_nonzero(has_depth)),
        "consistent_multiview_fallback_count": int(fallback_count),
        "accepted_count": int(np.count_nonzero(accepted)),
        "depth_supported_fraction": float(np.mean(has_depth)) if point_count else 0.0,
        "accepted_fraction": float(np.mean(accepted)) if point_count else 0.0,
        "source_kind": source_kind,
    }
    return output, accepted, diagnostics


def rasterize_projected_component(
    points_world: np.ndarray,
    camera: CameraGeometry,
    *,
    point_radius: int = 8,
    close_radius: int = 12,
    max_area_fraction: float = 0.35,
) -> np.ndarray:
    pixels, _, inside = project_points(points_world, camera)
    mask = np.zeros((camera.height, camera.width), dtype=np.uint8)
    valid_pixels = pixels[inside]
    if len(valid_pixels) == 0:
        return mask.astype(bool)
    for x, y in np.rint(valid_pixels).astype(np.int64):
        cv2.circle(mask, (int(x), int(y)), int(point_radius), 1, thickness=-1)
    if close_radius > 0:
        size = int(close_radius) * 2 + 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))

    if len(valid_pixels) >= 12:
        center = np.median(valid_pixels, axis=0)
        distance = np.linalg.norm(valid_pixels - center[None], axis=1)
        cutoff = max(float(np.quantile(distance, 0.95)), 1.0)
        hull_points = valid_pixels[distance <= cutoff]
        if len(hull_points) >= 3:
            hull = cv2.convexHull(np.rint(hull_points).astype(np.int32))
            hull_mask = np.zeros_like(mask)
            cv2.fillConvexPoly(hull_mask, hull, 1)
            if float(hull_mask.mean()) <= float(max_area_fraction):
                mask = np.maximum(mask, hull_mask)
    if float(mask.mean()) > float(max_area_fraction):
        return np.zeros_like(mask, dtype=bool)
    return mask.astype(bool)
