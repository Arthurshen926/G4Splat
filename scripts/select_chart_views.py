from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from matcha.cambridge_masks import CambridgeMaskLookup
from view_quality_control.poses import qvec_to_rotmat


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def sorted_scene_image_names(scene_path: Path) -> list[str]:
    image_dir = scene_path / "images"
    if not image_dir.exists():
        image_dir = scene_path
    return sorted(p.name for p in image_dir.iterdir() if p.suffix in IMAGE_SUFFIXES)


def parse_colmap_images_text(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    poses = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        if not parts[0].lstrip("-").isdigit():
            continue
        name = parts[9]
        if Path(name).suffix not in IMAGE_SUFFIXES:
            continue
        qvec = np.array([float(v) for v in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(v) for v in parts[5:8]], dtype=np.float64)
        poses[name] = (qvec, tvec)
    return poses


def camera_center_and_direction(qvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = qvec_to_rotmat(qvec)
    center = -rotation.T @ tvec
    direction = rotation.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    direction_norm = np.linalg.norm(direction)
    if direction_norm > 0:
        direction = direction / direction_norm
    return center, direction


def build_pose_features(
    image_names: list[str],
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
    position_weight: float = 1.0,
    direction_weight: float = 0.25,
) -> np.ndarray:
    centers = []
    directions = []
    for name in image_names:
        center, direction = camera_center_and_direction(*poses[name])
        centers.append(center)
        directions.append(direction)

    centers = np.stack(centers, axis=0)
    directions = np.stack(directions, axis=0)
    bbox_diag = np.linalg.norm(centers.max(axis=0) - centers.min(axis=0))
    if bbox_diag <= 1e-12:
        bbox_diag = 1.0
    normalized_centers = (centers - centers.mean(axis=0, keepdims=True)) / bbox_diag
    return np.concatenate(
        [position_weight * normalized_centers, direction_weight * directions],
        axis=1,
    )


def build_pose_geometry(
    image_names: list[str],
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, float]:
    centers = []
    directions = []
    for name in image_names:
        center, direction = camera_center_and_direction(*poses[name])
        centers.append(center)
        directions.append(direction)

    centers_array = np.stack(centers, axis=0)
    directions_array = np.stack(directions, axis=0)
    bbox_diag = float(np.linalg.norm(centers_array.max(axis=0) - centers_array.min(axis=0)))
    if bbox_diag <= 1e-12:
        bbox_diag = 1.0
    normalized_centers = (centers_array - centers_array.mean(axis=0, keepdims=True)) / bbox_diag
    return normalized_centers, directions_array, bbox_diag


def normalize_required_name(name: str) -> str:
    return name.replace("/", "__")


def image_quality_score(
    image_path: Path,
    image_name: str,
    mask_lookup: Optional[CambridgeMaskLookup] = None,
    low_intensity: float = 0.03,
    high_intensity: float = 0.97,
) -> dict[str, float]:
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    analysis_mask = np.ones(gray.shape, dtype=np.bool_)
    valid_ratio = 1.0
    if mask_lookup is not None:
        mask = mask_lookup.get_mask(image_name, gray.shape, torch.device("cpu"))
        analysis_mask = mask.detach().cpu().numpy().astype(np.bool_, copy=False)
        valid_ratio = float(np.count_nonzero(analysis_mask) / analysis_mask.size)

    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    dx_values = dx[analysis_mask[:, 1:] & analysis_mask[:, :-1]]
    dy_values = dy[analysis_mask[1:, :] & analysis_mask[:-1, :]]
    sharpness = float(
        (dx_values.var() if dx_values.size else 0.0)
        + (dy_values.var() if dy_values.size else 0.0)
    )

    valid_gray = gray[analysis_mask]
    if valid_gray.size == 0:
        extreme_ratio = 1.0
        shadow_ratio = 1.0
    else:
        extreme_ratio = float(
            ((valid_gray <= low_intensity) | (valid_gray >= high_intensity)).mean()
        )
        shadow_ratio = float((valid_gray <= 0.10).mean())

    lower_start = int(0.25 * gray.shape[0])
    lower_mask = analysis_mask[lower_start:]
    lower_values = gray[lower_start:][lower_mask]
    lower_mean_intensity = float(lower_values.mean()) if lower_values.size else 0.0
    return {
        "sharpness": sharpness,
        "extreme_ratio": extreme_ratio,
        "shadow_ratio": shadow_ratio,
        "lower_mean_intensity": lower_mean_intensity,
        "valid_ratio": valid_ratio,
    }


def compute_quality_scores(
    scene_path: Path,
    image_names: list[str],
    mask_lookup: Optional[CambridgeMaskLookup] = None,
) -> dict[str, dict[str, float]]:
    image_dir = scene_path / "images"
    if not image_dir.exists():
        image_dir = scene_path
    return {
        name: image_quality_score(image_dir / name, name, mask_lookup)
        for name in image_names
    }


def load_quality_score_cache(
    cache_path: Path,
    scene_path: Path,
    image_names: list[str],
    mask_pickle: Optional[Path],
    mask_indices: list[int],
) -> Optional[dict[str, dict[str, float]]]:
    if not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text())
    mask_stat = mask_pickle.stat() if mask_pickle is not None else None
    expected = {
        "scene_path": str(scene_path.resolve()),
        "mask_pickle": str(mask_pickle.resolve()) if mask_pickle is not None else None,
        "mask_size": mask_stat.st_size if mask_stat is not None else None,
        "mask_mtime_ns": mask_stat.st_mtime_ns if mask_stat is not None else None,
        "mask_indices": mask_indices,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    scores = payload.get("scores", {})
    if any(name not in scores for name in image_names):
        return None
    return {
        name: {key: float(value) for key, value in scores[name].items()}
        for name in image_names
    }


def write_quality_score_cache(
    cache_path: Path,
    scene_path: Path,
    scores: dict[str, dict[str, float]],
    mask_pickle: Optional[Path],
    mask_indices: list[int],
) -> None:
    mask_stat = mask_pickle.stat() if mask_pickle is not None else None
    payload = {
        "scene_path": str(scene_path.resolve()),
        "mask_pickle": str(mask_pickle.resolve()) if mask_pickle is not None else None,
        "mask_size": mask_stat.st_size if mask_stat is not None else None,
        "mask_mtime_ns": mask_stat.st_mtime_ns if mask_stat is not None else None,
        "mask_indices": mask_indices,
        "scores": scores,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload))


def quality_filtered_names(
    image_names: list[str],
    scores: dict[str, dict[str, float]],
    min_sharpness: float,
    max_extreme_ratio: float,
    min_valid_ratio: float,
    required_names: Optional[list[str]] = None,
    max_shadow_ratio: float = 1.0,
    min_lower_mean_intensity: float = 0.0,
) -> list[str]:
    required = {normalize_required_name(name) for name in (required_names or [])}
    filtered = []
    for name in image_names:
        score = scores[name]
        keep = (
            score["sharpness"] >= min_sharpness
            and score["extreme_ratio"] <= max_extreme_ratio
            and score.get("shadow_ratio", 0.0) <= max_shadow_ratio
            and score.get("lower_mean_intensity", 1.0) >= min_lower_mean_intensity
            and score["valid_ratio"] >= min_valid_ratio
        )
        if keep or name in required:
            filtered.append(name)
    return filtered


def semantic_filtered_names(
    image_names: list[str],
    mask_lookup: CambridgeMaskLookup,
    mask_indices: list[int],
    max_invalid_ratio: float,
) -> tuple[list[str], dict[str, float]]:
    if max_invalid_ratio < 0.0 or max_invalid_ratio > 1.0:
        raise ValueError("max_invalid_ratio must be in [0, 1]")
    if not mask_indices:
        raise ValueError("At least one semantic mask index is required")

    invalid_ratios = {
        name: mask_lookup.invalid_ratio_for_indices(name, mask_indices)
        for name in image_names
    }
    epsilon = 1e-12
    filtered = [
        name for name in image_names
        if invalid_ratios[name] <= max_invalid_ratio + epsilon
    ]
    return filtered, invalid_ratios


def load_semantic_ratio_cache(
    cache_path: Path,
    image_names: list[str],
    mask_pickle: Path,
    mask_indices: list[int],
) -> Optional[dict[str, float]]:
    if not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text())
    mask_stat = mask_pickle.stat()
    expected = {
        "mask_pickle": str(mask_pickle.resolve()),
        "mask_size": mask_stat.st_size,
        "mask_mtime_ns": mask_stat.st_mtime_ns,
        "mask_indices": mask_indices,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    ratios = payload.get("invalid_ratios", {})
    if any(name not in ratios for name in image_names):
        return None
    return {name: float(ratios[name]) for name in image_names}


def write_semantic_ratio_cache(
    cache_path: Path,
    invalid_ratios: dict[str, float],
    mask_pickle: Path,
    mask_indices: list[int],
) -> None:
    mask_stat = mask_pickle.stat()
    payload = {
        "mask_pickle": str(mask_pickle.resolve()),
        "mask_size": mask_stat.st_size,
        "mask_mtime_ns": mask_stat.st_mtime_ns,
        "mask_indices": mask_indices,
        "invalid_ratios": invalid_ratios,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2))


def _farthest_point_indices(features: np.ndarray, count: int) -> list[int]:
    if count <= 0 or count > len(features):
        raise ValueError(f"count must be in [1, {len(features)}], got {count}")
    first = int(np.argmax(np.linalg.norm(features - features.mean(axis=0), axis=1)))
    selected = [first]
    min_distances = np.linalg.norm(features - features[first], axis=1)
    min_distances[first] = -1.0
    while len(selected) < count:
        next_index = int(np.argmax(min_distances))
        selected.append(next_index)
        min_distances = np.minimum(
            min_distances,
            np.linalg.norm(features - features[next_index], axis=1),
        )
        min_distances[selected] = -1.0
    return selected


def _sequence_name(image_name: str) -> str:
    return image_name.split("__", 1)[0] if "__" in image_name else "default"


def _target_coverage_score(
    image_names: list[str],
    features: np.ndarray,
    target_indices: np.ndarray,
    selected_indices: list[int],
) -> tuple[float, float, float, float, float]:
    """Rank a chart set by sequence-balanced nearest-pose coverage."""
    if not selected_indices:
        return (float("inf"),) * 5
    target_features = features[target_indices]
    selected_features = features[np.asarray(selected_indices, dtype=np.int64)]
    nearest = np.min(
        np.linalg.norm(target_features[:, None, :] - selected_features[None, :, :], axis=2),
        axis=1,
    )
    sequence_p95 = []
    target_names = [image_names[int(index)] for index in target_indices]
    for sequence in sorted({_sequence_name(name) for name in target_names}):
        sequence_mask = np.asarray(
            [_sequence_name(name) == sequence for name in target_names],
            dtype=bool,
        )
        sequence_p95.append(float(np.quantile(nearest[sequence_mask], 0.95)))
    return (
        max(sequence_p95),
        float(np.quantile(nearest, 0.95)),
        float(np.quantile(nearest, 0.99)),
        float(nearest.max()),
        float(nearest.mean()),
    )


def _target_coverage_diagnostics(
    image_names: list[str],
    features: np.ndarray,
    selected_indices: list[int],
) -> dict:
    selected_array = np.asarray(selected_indices, dtype=np.int64)
    distances = np.linalg.norm(
        features[:, None, :] - features[selected_array][None, :, :],
        axis=2,
    )
    nearest_chart_offsets = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(len(image_names)), nearest_chart_offsets]

    def summarize(values: np.ndarray) -> dict[str, float]:
        return {
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }

    per_sequence = {}
    for sequence in sorted({_sequence_name(name) for name in image_names}):
        sequence_indices = np.asarray(
            [index for index, name in enumerate(image_names) if _sequence_name(name) == sequence],
            dtype=np.int64,
        )
        per_sequence[sequence] = {
            "target_count": int(len(sequence_indices)),
            **summarize(nearest_distances[sequence_indices]),
        }

    worst_indices = np.argsort(nearest_distances)[::-1][:20]
    return {
        "all_targets": summarize(nearest_distances),
        "per_sequence": per_sequence,
        "worst_targets": [
            {
                "image_name": image_names[int(index)],
                "distance": float(nearest_distances[index]),
                "nearest_chart": image_names[
                    selected_indices[int(nearest_chart_offsets[index])]
                ],
            }
            for index in worst_indices
        ],
    }


def select_clustered_coverage_indices(
    image_names: list[str],
    candidate_image_names: list[str],
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
    n_images: int,
    n_view_clusters: int = 8,
    min_views_per_cluster: int = 2,
    min_baseline_ratio: float = 0.02,
    min_global_pose_distance: float = 0.05,
    max_borrowed_support_distance: float = 0.20,
    allow_undercovered_clusters: bool = False,
    position_weight: float = 1.0,
    direction_weight: float = 0.35,
    required_names: Optional[list[str]] = None,
    coverage_objective: str = "target_kcenter",
) -> tuple[list[int], dict]:
    """Select clean chart views while covering pose/direction cells with multiple baselines."""
    if n_images <= 0:
        raise ValueError("n_images must be positive")
    if n_view_clusters <= 0:
        raise ValueError("n_view_clusters must be positive")
    if min_views_per_cluster <= 0:
        raise ValueError("min_views_per_cluster must be positive")
    if n_view_clusters * min_views_per_cluster > n_images:
        raise ValueError(
            "n_view_clusters * min_views_per_cluster cannot exceed n_images: "
            f"{n_view_clusters} * {min_views_per_cluster} > {n_images}"
        )
    if min_baseline_ratio < 0.0:
        raise ValueError("min_baseline_ratio must be non-negative")
    if min_global_pose_distance < 0.0:
        raise ValueError("min_global_pose_distance must be non-negative")
    if max_borrowed_support_distance < 0.0:
        raise ValueError("max_borrowed_support_distance must be non-negative")
    if coverage_objective not in {"anchor_fps", "target_kcenter"}:
        raise ValueError(
            "coverage_objective must be one of: anchor_fps, target_kcenter"
        )

    missing_poses = [name for name in image_names if name not in poses]
    if missing_poses:
        preview = ", ".join(missing_poses[:5])
        raise ValueError(f"{len(missing_poses)} images are missing COLMAP poses: {preview}")
    if n_images > len(candidate_image_names):
        raise RuntimeError(
            f"Only {len(candidate_image_names)} candidate views remain for n_images={n_images}"
        )

    name_to_index = {name: index for index, name in enumerate(image_names)}
    unknown_candidates = [name for name in candidate_image_names if name not in name_to_index]
    if unknown_candidates:
        raise ValueError(f"Candidate image is not in the scene list: {unknown_candidates[0]}")
    candidate_indices = np.array([name_to_index[name] for name in candidate_image_names], dtype=np.int64)
    candidate_set = set(candidate_indices.tolist())

    required_indices = []
    for raw_name in required_names or []:
        name = normalize_required_name(raw_name)
        if name not in name_to_index:
            raise ValueError(f"Required image {raw_name!r} is not in the scene image list")
        index = name_to_index[name]
        if index not in candidate_set:
            raise ValueError(
                f"Required image {raw_name!r} was rejected by semantic/quality filtering"
            )
        required_indices.append(index)
    required_indices = list(dict.fromkeys(required_indices))
    if len(required_indices) > n_images:
        raise ValueError("More required images were supplied than requested chart views")

    normalized_centers, directions, _ = build_pose_geometry(image_names, poses)
    features = np.concatenate(
        [position_weight * normalized_centers, direction_weight * directions],
        axis=1,
    )
    for required_offset, required_index in enumerate(required_indices):
        for other_index in required_indices[required_offset + 1:]:
            distance = float(np.linalg.norm(features[required_index] - features[other_index]))
            if distance < min_global_pose_distance:
                raise ValueError(
                    f"Required images are pose-near-duplicates ({distance:.4f} < "
                    f"{min_global_pose_distance:.4f})"
                )

    anchor_indices = _farthest_point_indices(features, n_view_clusters)
    anchor_features = features[np.array(anchor_indices)]
    cluster_labels = np.argmin(
        np.linalg.norm(features[:, None, :] - anchor_features[None, :, :], axis=2),
        axis=1,
    )

    selected = list(required_indices)
    cluster_diagnostics = []
    for cluster_index in range(n_view_clusters):
        target_indices = np.flatnonzero(cluster_labels == cluster_index)
        cluster_candidates = [
            int(index) for index in candidate_indices
            if cluster_labels[index] == cluster_index
        ]
        support_distances = {
            int(index): float(np.linalg.norm(features[index] - anchor_features[cluster_index]))
            for index in candidate_indices
        }
        # Required/reference anchors may lie just across a Voronoi cluster
        # boundary while still being valid borrowed support for this cluster.
        # Count them here exactly as the final coverage audit does; otherwise
        # the selector tries to add a pose-near duplicate and reports a false
        # coverage failure.
        already_selected = [
            index
            for index in selected
            if cluster_labels[index] == cluster_index
            or support_distances.get(index, float("inf"))
            <= max_borrowed_support_distance
        ]
        support_candidates = sorted(
            (
                int(index) for index in candidate_indices
                if int(index) in cluster_candidates
                or support_distances[int(index)] <= max_borrowed_support_distance
            ),
            key=lambda index: (index not in cluster_candidates, support_distances[index], index),
        )
        nearest_distances = sorted(support_distances.values())[: min_views_per_cluster + 2]
        if len(support_candidates) < min_views_per_cluster and not allow_undercovered_clusters:
            raise RuntimeError(
                f"View cluster {cluster_index} contains {len(target_indices)} target frames and "
                f"{len(cluster_candidates)} native clean candidate(s), but only "
                f"{len(support_candidates)} clean support view(s) are within pose distance "
                f"{max_borrowed_support_distance:.4f}. Strict semantic cleanliness and trajectory "
                f"coverage cannot both be satisfied. Nearest clean distances: {nearest_distances}"
            )

        required_support_count = min(min_views_per_cluster, len(support_candidates))
        while len(already_selected) < required_support_count:
            available = [
                index for index in support_candidates
                if index not in selected
                and (
                    not selected
                    or min(
                        float(np.linalg.norm(features[index] - features[other]))
                        for other in selected
                    ) >= min_global_pose_distance
                )
            ]
            if not available:
                if allow_undercovered_clusters:
                    break
                raise RuntimeError(
                    f"View cluster {cluster_index} has no remaining clean support view satisfying "
                    f"global pose distance {min_global_pose_distance:.4f}."
                )
            if not already_selected:
                if coverage_objective == "target_kcenter":
                    next_index = min(
                        available,
                        key=lambda index: (
                            _target_coverage_score(
                                image_names,
                                features,
                                target_indices,
                                already_selected + [index],
                            ),
                            index not in cluster_candidates,
                            support_distances[index],
                            index,
                        ),
                    )
                else:
                    next_index = min(available, key=lambda index: support_distances[index])
            else:
                baseline_candidates = [
                    index for index in available
                    if min(
                        float(np.linalg.norm(normalized_centers[index] - normalized_centers[other]))
                        for other in already_selected
                    ) >= min_baseline_ratio
                ]
                if not baseline_candidates:
                    if allow_undercovered_clusters:
                        break
                    raise RuntimeError(
                        f"View cluster {cluster_index} has clean candidates, but none provides the required "
                        f"camera baseline ratio {min_baseline_ratio:.4f}."
                    )
                if coverage_objective == "target_kcenter":
                    next_index = min(
                        baseline_candidates,
                        key=lambda index: (
                            _target_coverage_score(
                                image_names,
                                features,
                                target_indices,
                                already_selected + [index],
                            ),
                            index not in cluster_candidates,
                            support_distances[index],
                            index,
                        ),
                    )
                else:
                    next_index = max(
                        baseline_candidates,
                        key=lambda index: (
                            index in cluster_candidates,
                            -support_distances[index],
                            min(
                                float(np.linalg.norm(features[index] - features[other]))
                                for other in already_selected
                            ),
                        ),
                    )
            selected.append(next_index)
            already_selected.append(next_index)

        cluster_diagnostics.append(
            {
                "cluster": cluster_index,
                "target_count": int(len(target_indices)),
                "candidate_count": len(cluster_candidates),
                "support_candidate_count": len(support_candidates),
                "nearest_clean_distances": nearest_distances,
            }
        )

    while len(selected) < n_images:
        available = [
            int(index) for index in candidate_indices
            if int(index) not in selected
            and min(
                float(np.linalg.norm(features[int(index)] - features[other]))
                for other in selected
            ) >= min_global_pose_distance
        ]
        if not available:
            raise RuntimeError(
                f"Only {len(selected)} clean, globally diverse chart views can be selected; "
                f"requested {n_images} with min_global_pose_distance={min_global_pose_distance:.4f}."
            )
        if coverage_objective == "target_kcenter":
            all_target_indices = np.arange(len(image_names), dtype=np.int64)
            next_index = min(
                available,
                key=lambda index: (
                    _target_coverage_score(
                        image_names,
                        features,
                        all_target_indices,
                        selected + [index],
                    ),
                    index,
                ),
            )
        else:
            next_index = max(
                available,
                key=lambda index: min(
                    float(np.linalg.norm(features[index] - features[other]))
                    for other in selected
                ),
            )
        selected.append(next_index)

    selected = selected[:n_images]
    for diagnostic in cluster_diagnostics:
        cluster_index = diagnostic["cluster"]
        diagnostic["selected_count"] = int(
            sum(bool(cluster_labels[index] == cluster_index) for index in selected)
        )
        selected_for_cluster = sorted(
            selected,
            key=lambda index: float(np.linalg.norm(features[index] - anchor_features[cluster_index])),
        )[:min_views_per_cluster]
        diagnostic["selected_support_count"] = int(
            sum(
                bool(
                    cluster_labels[index] == cluster_index
                    or float(np.linalg.norm(features[index] - anchor_features[cluster_index]))
                    <= max_borrowed_support_distance
                )
                for index in selected
            )
        )
        diagnostic["coverage_satisfied"] = (
            diagnostic["selected_support_count"] >= min_views_per_cluster
        )
        diagnostic["max_selected_support_distance"] = max(
            float(np.linalg.norm(features[index] - anchor_features[cluster_index]))
            for index in selected_for_cluster
        )

    diagnostics = {
        "method": "pose_direction_clustered_coverage",
        "coverage_objective": coverage_objective,
        "view_clusters": n_view_clusters,
        "min_views_per_cluster": min_views_per_cluster,
        "min_baseline_ratio": min_baseline_ratio,
        "min_global_pose_distance": min_global_pose_distance,
        "max_borrowed_support_distance": max_borrowed_support_distance,
        "clusters": cluster_diagnostics,
        "target_coverage": _target_coverage_diagnostics(
            image_names,
            features,
            selected,
        ),
    }
    return sorted(int(index) for index in selected), diagnostics


def select_farthest_indices(
    image_names: list[str],
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
    n_images: int,
    position_weight: float = 1.0,
    direction_weight: float = 0.25,
    include_ends: bool = True,
    required_names: Optional[list[str]] = None,
) -> list[int]:
    if n_images <= 0:
        raise ValueError("n_images must be positive")

    available = [name for name in image_names if name in poses]
    missing = sorted(set(image_names) - set(available))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} images are missing COLMAP poses: {preview}")

    if n_images >= len(image_names):
        return list(range(len(image_names)))

    selected: list[int] = []
    if include_ends:
        selected.extend([0, len(image_names) - 1])

    for raw_name in required_names or []:
        name = normalize_required_name(raw_name)
        if name not in image_names:
            raise ValueError(f"Required image {raw_name!r} is not in the scene image list")
        selected.append(image_names.index(name))

    selected = list(dict.fromkeys(selected))[:n_images]
    features = build_pose_features(image_names, poses, position_weight, direction_weight)

    if not selected:
        selected.append(int(np.argmax(np.linalg.norm(features - features.mean(axis=0), axis=1))))

    min_distances = np.min(
        np.linalg.norm(features[:, None, :] - features[np.array(selected)][None, :, :], axis=2),
        axis=1,
    )
    min_distances[selected] = -1.0

    while len(selected) < n_images:
        next_index = int(np.argmax(min_distances))
        selected.append(next_index)
        candidate_distances = np.linalg.norm(features - features[next_index], axis=1)
        min_distances = np.minimum(min_distances, candidate_distances)
        min_distances[selected] = -1.0

    return sorted(selected)


def load_scene_poses(scene_path: Path) -> tuple[list[str], dict[str, tuple[np.ndarray, np.ndarray]]]:
    image_names = sorted_scene_image_names(scene_path)
    images_txt = scene_path / "sparse" / "0" / "images.txt"
    if not images_txt.exists():
        raise FileNotFoundError(f"COLMAP text pose file not found: {images_txt}")
    poses = parse_colmap_images_text(images_txt)
    return image_names, poses


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select clean MAtCha chart views with clustered camera-position and viewing-direction coverage."
        )
    )
    parser.add_argument("-s", "--scene-path", type=Path, required=True, help="Staged COLMAP scene path.")
    parser.add_argument("-n", "--n-images", type=int, required=True, help="Number of chart views to select.")
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--direction-weight", type=float, default=0.35)
    parser.add_argument("--no-include-ends", action="store_true", help="Do not force the first and last sorted images.")
    parser.add_argument("--require-name", nargs="*", default=[], help="Image names that must be included.")
    parser.add_argument(
        "--view-clusters",
        type=int,
        default=8,
        help="Pose/direction cells used for coverage. Set to 0 for legacy global pose FPS.",
    )
    parser.add_argument("--min-views-per-cluster", type=int, default=2)
    parser.add_argument(
        "--min-baseline-ratio",
        type=float,
        default=0.02,
        help="Minimum camera-center baseline divided by the full trajectory bounding-box diagonal.",
    )
    parser.add_argument("--min-global-pose-distance", type=float, default=0.05)
    parser.add_argument("--max-borrowed-support-distance", type=float, default=0.20)
    parser.add_argument("--allow-undercovered-clusters", action="store_true")
    parser.add_argument(
        "--coverage-objective",
        choices=["target_kcenter", "anchor_fps"],
        default="target_kcenter",
        help=(
            "Optimize nearest-chart coverage over every training camera, or use the legacy "
            "cluster-anchor plus candidate FPS objective."
        ),
    )
    parser.add_argument(
        "--semantic-filter",
        action="store_true",
        help="Reject entire chart candidates whose selected semantic keep masks contain invalid pixels.",
    )
    parser.add_argument("--semantic-mask-indices", nargs="*", type=int, default=[0])
    parser.add_argument("--max-semantic-invalid-ratio", type=float, default=0.0)
    parser.add_argument("--semantic-ratio-cache", type=Path, default=None)
    parser.add_argument("--occlusion-mask-indices", nargs="*", type=int, default=[])
    parser.add_argument("--max-occlusion-invalid-ratio", type=float, default=0.15)
    parser.add_argument("--occlusion-ratio-cache", type=Path, default=None)
    parser.add_argument("--tree-mask-pickle", type=Path, default=None)
    parser.add_argument("--tree-mask-dataset-path", type=Path, default=None)
    parser.add_argument("--tree-mask-index", type=int, default=3)
    parser.add_argument("--max-tree-invalid-ratio", type=float, default=0.15)
    parser.add_argument("--tree-ratio-cache", type=Path, default=None)
    parser.add_argument("--quality-filter", action="store_true", help="Filter low-quality images before pose FPS.")
    parser.add_argument("--min-sharpness", type=float, default=1e-4)
    parser.add_argument("--max-extreme-ratio", type=float, default=0.35)
    parser.add_argument("--max-shadow-ratio", type=float, default=1.0)
    parser.add_argument("--min-lower-mean-intensity", type=float, default=0.0)
    parser.add_argument("--min-valid-ratio", type=float, default=0.75)
    parser.add_argument("--mask-pickle", type=Path, default=None)
    parser.add_argument("--mask-dataset-path", type=Path, default=None)
    parser.add_argument("--mask-indices", nargs="*", type=int, default=None)
    parser.add_argument("--quality-mask-indices", nargs="*", type=int, default=None)
    parser.add_argument("--quality-score-cache", type=Path, default=None)
    parser.add_argument(
        "--allowed-name-file",
        type=Path,
        default=None,
        help="Optional newline-delimited staged/source names allowed to become chart views.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to persist the complete selection and coverage audit JSON.",
    )
    args = parser.parse_args()

    image_names, poses = load_scene_poses(args.scene_path)
    candidate_image_names = image_names
    mask_lookup = None
    semantic_invalid_ratios = None
    if args.semantic_filter:
        if args.mask_pickle is None:
            raise ValueError("--mask-pickle is required when --semantic-filter is enabled")
        if args.semantic_ratio_cache is not None:
            semantic_invalid_ratios = load_semantic_ratio_cache(
                args.semantic_ratio_cache,
                image_names,
                args.mask_pickle,
                args.semantic_mask_indices,
            )
        if semantic_invalid_ratios is None:
            mask_lookup = CambridgeMaskLookup(
                args.mask_dataset_path or args.scene_path,
                args.mask_pickle,
                mask_indices=args.semantic_mask_indices,
            )
            candidate_image_names, semantic_invalid_ratios = semantic_filtered_names(
                image_names,
                mask_lookup,
                mask_indices=args.semantic_mask_indices,
                max_invalid_ratio=args.max_semantic_invalid_ratio,
            )
            if args.semantic_ratio_cache is not None:
                write_semantic_ratio_cache(
                    args.semantic_ratio_cache,
                    semantic_invalid_ratios,
                    args.mask_pickle,
                    args.semantic_mask_indices,
                )
        else:
            candidate_image_names = [
                name for name in image_names
                if semantic_invalid_ratios[name] <= args.max_semantic_invalid_ratio + 1e-12
            ]
        required = {normalize_required_name(name) for name in args.require_name}
        rejected_required = sorted(required - set(candidate_image_names))
        if rejected_required:
            raise RuntimeError(
                "Required chart view(s) contain forbidden semantic pixels: "
                + ", ".join(rejected_required)
            )

    occlusion_invalid_ratios = None
    if args.occlusion_mask_indices:
        if args.mask_pickle is None:
            raise ValueError("--mask-pickle is required when occlusion mask indices are supplied")
        if args.occlusion_ratio_cache is not None:
            occlusion_invalid_ratios = load_semantic_ratio_cache(
                args.occlusion_ratio_cache,
                image_names,
                args.mask_pickle,
                args.occlusion_mask_indices,
            )
        if occlusion_invalid_ratios is None:
            if mask_lookup is None:
                mask_lookup = CambridgeMaskLookup(
                    args.mask_dataset_path or args.scene_path,
                    args.mask_pickle,
                    mask_indices=args.occlusion_mask_indices,
                )
            _, occlusion_invalid_ratios = semantic_filtered_names(
                image_names,
                mask_lookup,
                mask_indices=args.occlusion_mask_indices,
                max_invalid_ratio=args.max_occlusion_invalid_ratio,
            )
            if args.occlusion_ratio_cache is not None:
                write_semantic_ratio_cache(
                    args.occlusion_ratio_cache,
                    occlusion_invalid_ratios,
                    args.mask_pickle,
                    args.occlusion_mask_indices,
                )
        candidate_image_names = [
            name for name in candidate_image_names
            if occlusion_invalid_ratios[name] <= args.max_occlusion_invalid_ratio + 1e-12
        ]

    tree_invalid_ratios = None
    if args.tree_mask_pickle is not None:
        tree_indices = [args.tree_mask_index]
        if args.tree_ratio_cache is not None:
            tree_invalid_ratios = load_semantic_ratio_cache(
                args.tree_ratio_cache,
                image_names,
                args.tree_mask_pickle,
                tree_indices,
            )
        tree_lookup = CambridgeMaskLookup(
            args.tree_mask_dataset_path or args.mask_dataset_path or args.scene_path,
            args.tree_mask_pickle,
            mask_indices=tree_indices,
        )
        if tree_invalid_ratios is None:
            _, tree_invalid_ratios = semantic_filtered_names(
                image_names,
                tree_lookup,
                mask_indices=tree_indices,
                max_invalid_ratio=args.max_tree_invalid_ratio,
            )
            if args.tree_ratio_cache is not None:
                write_semantic_ratio_cache(
                    args.tree_ratio_cache,
                    tree_invalid_ratios,
                    args.tree_mask_pickle,
                    tree_indices,
                )
        candidate_image_names = [
            name for name in candidate_image_names
            if tree_invalid_ratios[name] <= args.max_tree_invalid_ratio + 1e-12
        ]

    allowed_names = None
    if args.allowed_name_file is not None:
        allowed_names = {
            normalize_required_name(line.strip())
            for line in args.allowed_name_file.read_text().splitlines()
            if line.strip()
        }
        candidate_image_names = [
            name for name in candidate_image_names if name in allowed_names
        ]
        if len(candidate_image_names) < args.n_images:
            raise RuntimeError(
                f"Allowed-name filter kept {len(candidate_image_names)} image(s), "
                f"fewer than requested n_images={args.n_images}."
            )

    quality_scores = None
    if args.quality_filter:
        quality_mask_indices = args.quality_mask_indices or args.mask_indices or [0, 1, 2]
        if args.quality_score_cache is not None:
            quality_scores = load_quality_score_cache(
                args.quality_score_cache,
                args.scene_path,
                candidate_image_names,
                args.mask_pickle,
                quality_mask_indices,
            )
        quality_mask_lookup = None
        if quality_scores is None and (
            args.mask_pickle is not None
            and (args.min_valid_ratio > 0.0 or args.quality_mask_indices is not None)
        ):
            quality_mask_lookup = CambridgeMaskLookup(
                args.mask_dataset_path or args.scene_path,
                args.mask_pickle,
                mask_indices=quality_mask_indices,
            )
        if quality_scores is None:
            quality_scores = compute_quality_scores(
                args.scene_path,
                candidate_image_names,
                quality_mask_lookup,
            )
            if args.quality_score_cache is not None:
                write_quality_score_cache(
                    args.quality_score_cache,
                    args.scene_path,
                    quality_scores,
                    args.mask_pickle,
                    quality_mask_indices,
                )
        candidate_image_names = quality_filtered_names(
            candidate_image_names,
            quality_scores,
            min_sharpness=args.min_sharpness,
            max_extreme_ratio=args.max_extreme_ratio,
            min_valid_ratio=args.min_valid_ratio,
            required_names=args.require_name,
            max_shadow_ratio=args.max_shadow_ratio,
            min_lower_mean_intensity=args.min_lower_mean_intensity,
        )
        if len(candidate_image_names) < args.n_images:
            raise RuntimeError(
                f"Quality filter kept {len(candidate_image_names)} image(s), "
                f"fewer than requested n_images={args.n_images}. Relax thresholds."
            )

    coverage_diagnostics = None
    if args.view_clusters > 0:
        selected, coverage_diagnostics = select_clustered_coverage_indices(
            image_names=image_names,
            candidate_image_names=candidate_image_names,
            poses=poses,
            n_images=args.n_images,
            n_view_clusters=args.view_clusters,
            min_views_per_cluster=args.min_views_per_cluster,
            min_baseline_ratio=args.min_baseline_ratio,
            min_global_pose_distance=args.min_global_pose_distance,
            max_borrowed_support_distance=args.max_borrowed_support_distance,
            allow_undercovered_clusters=args.allow_undercovered_clusters,
            position_weight=args.position_weight,
            direction_weight=args.direction_weight,
            required_names=args.require_name,
            coverage_objective=args.coverage_objective,
        )
    else:
        candidate_selected = select_farthest_indices(
            image_names=candidate_image_names,
            poses=poses,
            n_images=args.n_images,
            position_weight=args.position_weight,
            direction_weight=args.direction_weight,
            include_ends=not args.no_include_ends,
            required_names=args.require_name,
        )
        selected = [image_names.index(candidate_image_names[index]) for index in candidate_selected]
    selected_names = [image_names[index] for index in selected]
    result = {
        "scene_path": str(args.scene_path),
        "n_images": args.n_images,
        "image_idx": selected,
        "image_names": selected_names,
        # Preserve the fully filtered reserve pool so an alignment failure can
        # be replaced without silently relaxing semantic/image-quality gates.
        "candidate_pool_names": candidate_image_names,
        "candidate_pool_size": len(candidate_image_names),
        "run_sfm_arg": "--image_idx " + " ".join(str(i) for i in selected),
    }
    if coverage_diagnostics is not None:
        result["coverage"] = coverage_diagnostics
    if semantic_invalid_ratios is not None:
        result["semantic_filter"] = {
            "kept": len([
                ratio for ratio in semantic_invalid_ratios.values()
                if ratio <= args.max_semantic_invalid_ratio + 1e-12
            ]),
            "total": len(image_names),
            "mask_indices": args.semantic_mask_indices,
            "max_invalid_ratio": args.max_semantic_invalid_ratio,
        }
        result["semantic_invalid_ratios"] = {
            name: semantic_invalid_ratios[name] for name in selected_names
        }
    if occlusion_invalid_ratios is not None:
        result["occlusion_filter"] = {
            "kept_after_semantic_filter": len([
                name for name in image_names
                if semantic_invalid_ratios is None
                or semantic_invalid_ratios[name] <= args.max_semantic_invalid_ratio + 1e-12
                if occlusion_invalid_ratios[name] <= args.max_occlusion_invalid_ratio + 1e-12
            ]),
            "mask_indices": args.occlusion_mask_indices,
            "max_invalid_ratio": args.max_occlusion_invalid_ratio,
        }
        result["occlusion_invalid_ratios"] = {
            name: occlusion_invalid_ratios[name] for name in selected_names
        }
    if tree_invalid_ratios is not None:
        result["tree_filter"] = {
            "mask_pickle": str(args.tree_mask_pickle),
            "mask_index": args.tree_mask_index,
            "max_invalid_ratio": args.max_tree_invalid_ratio,
            "kept": sum(
                ratio <= args.max_tree_invalid_ratio + 1e-12
                for ratio in tree_invalid_ratios.values()
            ),
            "total": len(image_names),
        }
        result["tree_invalid_ratios"] = {
            name: tree_invalid_ratios[name] for name in selected_names
        }
    if quality_scores is not None:
        result["quality_filter"] = {
            "kept": len(candidate_image_names),
            "total": len(image_names),
            "min_sharpness": args.min_sharpness,
            "max_extreme_ratio": args.max_extreme_ratio,
            "max_shadow_ratio": args.max_shadow_ratio,
            "min_lower_mean_intensity": args.min_lower_mean_intensity,
            "min_valid_ratio": args.min_valid_ratio,
        }
        result["quality_scores"] = {name: quality_scores[name] for name in selected_names}
    if allowed_names is not None:
        result["allowed_name_filter"] = {
            "file": str(args.allowed_name_file),
            "allowed": len(allowed_names),
            "selected_all_allowed": all(name in allowed_names for name in selected_names),
        }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"Selected {len(selected)} chart views from {len(image_names)} images.")
    print(result["run_sfm_arg"])
    for idx, name in zip(selected, selected_names):
        print(f"{idx:04d} {name}")


if __name__ == "__main__":
    main()
