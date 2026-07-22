from __future__ import annotations

import argparse
from itertools import combinations
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


def static_support_ratios(
    image_names: list[str],
    mask_lookup: CambridgeMaskLookup,
    mask_indices: list[int],
) -> dict[str, float]:
    """Measure usable *static* support without making an image-level veto.

    Outdoor trajectories routinely contain a small amount of foliage, sky, or
    a transient object.  Those pixels must remain excluded from Chart
    alignment, but rejecting the whole image because of them removes the only
    useful baseline for parts of the trajectory.  The caller can still impose
    a minimum usable fraction; this helper deliberately reports a continuous
    support value so selection can trade image reliability against coverage.
    """
    if not mask_indices:
        raise ValueError("At least one static-support mask index is required")
    return {
        name: float(1.0 - mask_lookup.invalid_ratio_for_indices(name, mask_indices))
        for name in image_names
    }


def _rank_unit(values: list[float], *, higher_is_better: bool) -> np.ndarray:
    """Stable [0, 1] ranks for heterogeneous candidate-quality quantities."""
    array = np.asarray(values, dtype=np.float64)
    result = np.full(len(array), 0.5, dtype=np.float64)
    finite = np.isfinite(array)
    if not np.any(finite):
        return result
    finite_values = array[finite]
    if np.ptp(finite_values) <= 1e-12:
        return result
    order = np.argsort(finite_values, kind="stable")
    ranks = np.empty(len(finite_values), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(finite_values))
    if not higher_is_better:
        ranks = 1.0 - ranks
    result[finite] = ranks
    return result


def candidate_reliability_scores(
    image_names: list[str],
    *,
    quality_scores: Optional[dict[str, dict[str, float]]] = None,
    static_support: Optional[dict[str, float]] = None,
    tree_invalid_ratios: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Combine image quality with usable static support into a bounded prior.

    This is intentionally a *weak* prior.  It should choose a clearer view
    when two cameras give equivalent coverage, never erase a trajectory
    corridor merely because it contains a tree in one corner.
    """
    if not image_names:
        return {}
    if quality_scores is None:
        quality = np.full(len(image_names), 0.5, dtype=np.float64)
    else:
        sharpness = _rank_unit(
            [quality_scores[name].get("sharpness", 0.0) for name in image_names],
            higher_is_better=True,
        )
        extremes = _rank_unit(
            [quality_scores[name].get("extreme_ratio", 1.0) for name in image_names],
            higher_is_better=False,
        )
        shadows = _rank_unit(
            [quality_scores[name].get("shadow_ratio", 1.0) for name in image_names],
            higher_is_better=False,
        )
        lower = _rank_unit(
            [quality_scores[name].get("lower_mean_intensity", 0.0) for name in image_names],
            higher_is_better=True,
        )
        quality = 0.40 * sharpness + 0.25 * extremes + 0.20 * shadows + 0.15 * lower
    static = np.asarray(
        [static_support.get(name, 1.0) if static_support is not None else 1.0 for name in image_names],
        dtype=np.float64,
    )
    tree_clear = np.asarray(
        [
            1.0 - tree_invalid_ratios.get(name, 0.0)
            if tree_invalid_ratios is not None
            else 1.0
            for name in image_names
        ],
        dtype=np.float64,
    )
    reliability = 0.55 * np.clip(static, 0.0, 1.0) + 0.35 * quality + 0.10 * np.clip(tree_clear, 0.0, 1.0)
    return {
        name: float(np.clip(value, 0.0, 1.0))
        for name, value in zip(image_names, reliability)
    }


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
    distance_matrix: Optional[np.ndarray] = None,
) -> tuple[float, float, float, float, float]:
    """Rank a chart set by sequence-balanced nearest-pose coverage."""
    if not selected_indices:
        return (float("inf"),) * 5
    selected_array = np.asarray(selected_indices, dtype=np.int64)
    if distance_matrix is None:
        target_features = features[target_indices]
        selected_features = features[selected_array]
        nearest = np.min(
            np.linalg.norm(target_features[:, None, :] - selected_features[None, :, :], axis=2),
            axis=1,
        )
    else:
        nearest = np.min(distance_matrix[np.ix_(target_indices, selected_array)], axis=1)
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


def _has_baseline(
    indices: list[int],
    normalized_centers: np.ndarray,
    minimum: float,
) -> bool:
    return any(
        float(np.linalg.norm(normalized_centers[first] - normalized_centers[second])) >= minimum
        for offset, first in enumerate(indices)
        for second in indices[offset + 1 :]
    )


def _coverage_choice_key(
    image_names: list[str],
    features: np.ndarray,
    target_indices: np.ndarray,
    proposed_indices: list[int],
    candidate_index: int,
    reliabilities: dict[int, float],
    reliability_weight: float,
    distance_matrix: Optional[np.ndarray] = None,
) -> tuple[float, float, float, float, float, float, int]:
    """Coverage remains primary; reliability only resolves a close trade-off."""
    coverage = _target_coverage_score(
        image_names,
        features,
        target_indices,
        proposed_indices + [candidate_index],
        distance_matrix=distance_matrix,
    )
    reliability = reliabilities.get(candidate_index, 0.5)
    return (
        coverage[0] + reliability_weight * (1.0 - reliability),
        coverage[1],
        coverage[2],
        coverage[3],
        coverage[4],
        -reliability,
        candidate_index,
    )


def _best_coverage_candidate(
    image_names: list[str],
    target_indices: np.ndarray,
    selected_indices: list[int],
    available_indices: list[int],
    reliabilities: dict[int, float],
    reliability_weight: float,
    distance_matrix: np.ndarray,
) -> int:
    """Vectorized exact target-k-center choice for a large Chart pool.

    The legacy implementation evaluated ``_target_coverage_score`` once per
    candidate.  Each invocation allocated a target-by-selected distance array,
    which made an otherwise modest 1,487-frame outdoor scene spend most of its
    time repeatedly recomputing the same camera geometry.  Evaluate every
    available candidate against the current nearest-chart distances in one
    array instead.  The returned lexicographic objective is intentionally the
    same as ``_coverage_choice_key``.
    """
    if not available_indices:
        raise ValueError("At least one coverage candidate is required")
    candidates = np.asarray(available_indices, dtype=np.int64)
    candidate_distances = distance_matrix[np.ix_(target_indices, candidates)]
    if selected_indices:
        current = np.min(
            distance_matrix[
                np.ix_(target_indices, np.asarray(selected_indices, dtype=np.int64))
            ],
            axis=1,
        )
        nearest = np.minimum(candidate_distances, current[:, None])
    else:
        nearest = candidate_distances

    target_sequences = np.asarray(
        [_sequence_name(image_names[int(index)]) for index in target_indices],
        dtype=object,
    )
    sequence_p95 = np.zeros(len(candidates), dtype=np.float64)
    for sequence in np.unique(target_sequences):
        sequence_values = nearest[target_sequences == sequence]
        sequence_p95 = np.maximum(
            sequence_p95,
            np.quantile(sequence_values, 0.95, axis=0),
        )
    p95 = np.quantile(nearest, 0.95, axis=0)
    p99 = np.quantile(nearest, 0.99, axis=0)
    maximum = np.max(nearest, axis=0)
    mean = np.mean(nearest, axis=0)
    reliability = np.asarray(
        [reliabilities.get(int(index), 0.5) for index in candidates],
        dtype=np.float64,
    )
    order = np.lexsort(
        (
            candidates,
            -reliability,
            mean,
            maximum,
            p99,
            p95,
            sequence_p95 + reliability_weight * (1.0 - reliability),
        )
    )
    return int(candidates[int(order[0])])


def _seed_sequence_support(
    image_names: list[str],
    candidate_indices: np.ndarray,
    features: np.ndarray,
    normalized_centers: np.ndarray,
    selected: list[int],
    *,
    max_selection_count: int,
    min_views_per_sequence: int,
    post_gate_min_views_per_sequence: int,
    sequence_gate_failure_budget: int,
    sequence_coverage_mode: str,
    min_baseline_ratio: float,
    reliabilities: dict[int, float],
    reliability_weight: float,
    distance_matrix: Optional[np.ndarray] = None,
) -> tuple[list[int], list[dict]]:
    """Reserve independent same-trajectory evidence before global k-center.

    Pose-near Charts from another capture sequence may be useful appearance
    evidence, but are not proof that a short, partially occluded temporal
    corridor has a two-view geometric basis.  Seed each sequence first, then
    let the existing global/pose-cluster coverage objective spend the remaining
    budget.  ``soft`` records an impossible sequence; ``strict`` refuses to
    label it as covered.
    """
    if min_views_per_sequence <= 0:
        return selected, []
    if sequence_coverage_mode not in {"soft", "strict"}:
        raise ValueError("sequence_coverage_mode must be 'soft' or 'strict' when sequence support is enabled")

    by_sequence: dict[str, list[int]] = {}
    target_by_sequence: dict[str, list[int]] = {}
    for index, name in enumerate(image_names):
        target_by_sequence.setdefault(_sequence_name(name), []).append(index)
    for index in candidate_indices.tolist():
        by_sequence.setdefault(_sequence_name(image_names[int(index)]), []).append(int(index))

    # A one-frame capture cannot, by definition, supply an independent
    # same-trajectory baseline.  Treat it as an explicitly reported exception
    # rather than making every outdoor scene with a singleton sequence
    # impossible to run.  Every temporal corridor that *can* contain the
    # requested number of real views remains subject to the hard constraint.
    required_sequences = {
        sequence
        for sequence, targets in target_by_sequence.items()
        if len(targets) >= min_views_per_sequence
    }

    required_budget = 0
    for sequence in required_sequences:
        available = len(by_sequence.get(sequence, []))
        already_selected = sum(
            _sequence_name(image_names[index]) == sequence for index in selected
        )
        if sequence_coverage_mode == "strict" and available < min_views_per_sequence:
            raise RuntimeError(
                "Sequence has insufficient static-support Chart candidates: "
                f"sequence={sequence}, candidates={available}, required={min_views_per_sequence}."
            )
        required_budget += max(0, min(min_views_per_sequence, available) - already_selected)
    if len(selected) + required_budget > max_selection_count:
        raise RuntimeError(
            "Sequence support seed exceeds requested Chart budget: "
            f"required={len(selected) + required_budget}, requested={max_selection_count}."
        )

    diagnostics: list[dict] = []
    for sequence in sorted(target_by_sequence):
        sequence_candidates = by_sequence.get(sequence, [])
        target_count = len(target_by_sequence[sequence])
        applicable = sequence in required_sequences
        required = min_views_per_sequence if applicable else 0
        desired = min(required, len(sequence_candidates))
        chosen = [index for index in selected if _sequence_name(image_names[index]) == sequence]
        targets = np.asarray(target_by_sequence[sequence], dtype=np.int64)
        failure = None
        while len(chosen) < desired:
            available = [index for index in sequence_candidates if index not in selected]
            if chosen:
                require_all_pairwise_baselines = (
                    sequence_gate_failure_budget > 0
                    and required - sequence_gate_failure_budget <= 2
                    and post_gate_min_views_per_sequence >= 2
                )
                baseline_candidates = [
                    index
                    for index in available
                    if (
                        all(
                            float(np.linalg.norm(normalized_centers[index] - normalized_centers[other]))
                            >= min_baseline_ratio
                            for other in chosen
                        )
                        if require_all_pairwise_baselines
                        else any(
                            float(np.linalg.norm(normalized_centers[index] - normalized_centers[other]))
                            >= min_baseline_ratio
                            for other in chosen
                        )
                    )
                ]
                if not baseline_candidates:
                    failure = "no_same_sequence_baseline"
                    break
                available = baseline_candidates
            if not available:
                failure = "no_remaining_candidate"
                break
            if distance_matrix is None:
                raise RuntimeError("Sequence coverage requires the precomputed pose-distance matrix")
            next_index = _best_coverage_candidate(
                image_names,
                targets,
                chosen,
                available,
                reliabilities,
                reliability_weight,
                distance_matrix,
            )
            selected.append(next_index)
            chosen.append(next_index)

        baseline = _has_baseline(chosen, normalized_centers, min_baseline_ratio)
        resilience = (
            _failure_budget_support_audit(
                chosen,
                normalized_centers,
                post_gate_min_views_per_sequence,
                min_baseline_ratio,
                sequence_gate_failure_budget,
            )
            if applicable
            else {
                "passed": True,
                "failure_budget": sequence_gate_failure_budget,
                "failure_subset_size": 0,
                "failure_scenario_count": 0,
                "failing_scenario_count": 0,
                "failing_scenario_examples": [],
                "worst_surviving_count": len(chosen),
            }
        )
        passed = (
            len(chosen) >= required
            and (required < 2 or baseline)
            and resilience["passed"]
        )
        if not passed and failure is None:
            failure = "insufficient_candidates"
        if not applicable:
            failure = "not_applicable_short_sequence"
        record = {
            "sequence": sequence,
            "target_count": target_count,
            "candidate_count": len(sequence_candidates),
            "required_count": required,
            "post_gate_required_count": post_gate_min_views_per_sequence,
            "gate_failure_budget": sequence_gate_failure_budget,
            "applicable": applicable,
            "selected_count": len(chosen),
            "selected_names": [image_names[index] for index in chosen],
            "baseline_satisfied": baseline if required >= 2 else True,
            "gate_resilience": resilience,
            "passed": passed,
            "failure": failure,
        }
        diagnostics.append(record)
        if applicable and not passed and sequence_coverage_mode == "strict":
            raise RuntimeError(
                "Sequence lacks independently-baselined static Chart support: "
                f"sequence={sequence}, candidates={len(sequence_candidates)}, "
                f"selected={len(chosen)}, required={required}, failure={failure}."
            )
    return selected, diagnostics


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
    candidate_reliability: Optional[dict[str, float]] = None,
    reliability_weight: float = 0.0,
    min_views_per_sequence: int = 0,
    post_gate_min_views_per_sequence: Optional[int] = None,
    sequence_gate_failure_budget: int = 0,
    sequence_coverage_mode: str = "off",
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
    if reliability_weight < 0.0:
        raise ValueError("reliability_weight must be non-negative")
    if min_views_per_sequence < 0:
        raise ValueError("min_views_per_sequence must be non-negative")
    if post_gate_min_views_per_sequence is None:
        post_gate_min_views_per_sequence = min_views_per_sequence
    if post_gate_min_views_per_sequence < 0:
        raise ValueError("post_gate_min_views_per_sequence must be non-negative")
    if sequence_gate_failure_budget < 0:
        raise ValueError("sequence_gate_failure_budget must be non-negative")
    if sequence_coverage_mode not in {"off", "soft", "strict"}:
        raise ValueError("sequence_coverage_mode must be one of: off, soft, strict")
    if min_views_per_sequence == 0 and sequence_coverage_mode != "off":
        raise ValueError("sequence coverage mode requires min_views_per_sequence > 0")
    if min_views_per_sequence > 0 and sequence_coverage_mode == "off":
        raise ValueError("min_views_per_sequence requires sequence coverage mode soft or strict")
    if min_views_per_sequence > 0:
        if post_gate_min_views_per_sequence < 1:
            raise ValueError("sequence coverage requires a positive post-gate support count")
        if min_views_per_sequence < post_gate_min_views_per_sequence + sequence_gate_failure_budget:
            raise ValueError(
                "min_views_per_sequence must contain post-gate support plus its failure budget"
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
    reliability_by_index = {
        index: float(np.clip((candidate_reliability or {}).get(name, 0.5), 0.0, 1.0))
        for name, index in name_to_index.items()
        if index in candidate_set
    }

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
    # Coverage selection calls the objective for many candidate/target pairs.
    # Recomputing the same pose distances in each call made a 1.5k-frame
    # outdoor sequence effectively quadratic in Python.  The full matrix is
    # small at this scale (~18 MiB float64 for Cambridge) and makes every
    # subsequent coverage score an indexed reduction.
    pairwise_distances = np.linalg.norm(
        features[:, None, :] - features[None, :, :], axis=2
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
    selected, sequence_diagnostics = _seed_sequence_support(
        image_names,
        candidate_indices,
        features,
        normalized_centers,
        selected,
        max_selection_count=n_images,
        min_views_per_sequence=min_views_per_sequence,
        post_gate_min_views_per_sequence=post_gate_min_views_per_sequence,
        sequence_gate_failure_budget=sequence_gate_failure_budget,
        sequence_coverage_mode=sequence_coverage_mode,
        min_baseline_ratio=min_baseline_ratio,
        reliabilities=reliability_by_index,
        reliability_weight=reliability_weight,
        distance_matrix=pairwise_distances,
    )
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
                    next_index = _best_coverage_candidate(
                        image_names,
                        target_indices,
                        already_selected,
                        available,
                        reliability_by_index,
                        reliability_weight,
                        pairwise_distances,
                    )
                else:
                    next_index = min(
                        available,
                        key=lambda index: (
                            support_distances[index],
                            -reliability_by_index.get(index, 0.5),
                            index,
                        ),
                    )
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
                    next_index = _best_coverage_candidate(
                        image_names,
                        target_indices,
                        already_selected,
                        baseline_candidates,
                        reliability_by_index,
                        reliability_weight,
                        pairwise_distances,
                    )
                else:
                    next_index = max(
                        baseline_candidates,
                        key=lambda index: (
                            index in cluster_candidates,
                            -support_distances[index],
                            -reliability_by_index.get(index, 0.5),
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
            next_index = _best_coverage_candidate(
                image_names,
                all_target_indices,
                selected,
                available,
                reliability_by_index,
                reliability_weight,
                pairwise_distances,
            )
        else:
            next_index = max(
                available,
                key=lambda index: (
                    min(
                        float(np.linalg.norm(features[index] - features[other]))
                        for other in selected
                    ),
                    reliability_by_index.get(index, 0.5),
                    -index,
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
        "candidate_reliability_weight": reliability_weight,
        "sequence_coverage_mode": sequence_coverage_mode,
        "min_views_per_sequence": min_views_per_sequence,
        "post_gate_min_views_per_sequence": post_gate_min_views_per_sequence,
        "sequence_gate_failure_budget": sequence_gate_failure_budget,
        "clusters": cluster_diagnostics,
        "sequences": sequence_diagnostics,
        "target_coverage": _target_coverage_diagnostics(
            image_names,
            features,
            selected,
        ),
    }
    return sorted(int(index) for index in selected), diagnostics


def _post_gate_support_satisfied(
    indices: list[int],
    normalized_centers: np.ndarray,
    min_views_per_cluster: int,
    min_baseline_ratio: float,
) -> tuple[bool, tuple[int, int] | None]:
    """Mirror the downstream joint-selector pose support constraint.

    A Chart cluster needs ``min_views_per_cluster`` surviving views and at
    least one independently-baselined pair.  In the Cambridge trajectories a
    third *temporally adjacent* frame is not a third baseline; it is useful
    only as a gate replacement for the corresponding primary support.  Keep
    this predicate identical to the quality-aware selector rather than
    inventing a stronger, incompatible notion of support here.
    """
    if min_views_per_cluster < 1:
        raise ValueError("min_views_per_cluster must be positive")
    if len(indices) < min_views_per_cluster:
        return False, None
    if min_views_per_cluster == 1:
        return True, None
    for first, second in combinations(indices, 2):
        baseline = float(
            np.linalg.norm(normalized_centers[first] - normalized_centers[second])
        )
        if baseline >= min_baseline_ratio:
            return True, (int(first), int(second))
    return False, None


def _failure_budget_support_audit(
    indices: list[int],
    normalized_centers: np.ndarray,
    min_views_per_cluster: int,
    min_baseline_ratio: float,
    failure_budget: int,
) -> dict:
    """Audit every worst-case hard-gate failure subset for one pose cluster.

    The old reserve policy only asked whether each *single* primary Chart could
    be removed in isolation.  Alignment conflicts can be correlated, however:
    several views in the same pose cell can all fail the target-depth gate.
    Support is monotone in the surviving view set, so checking every subset of
    exactly ``failure_budget`` rejected Charts is sufficient (and avoids
    presenting a probabilistic heuristic as a coverage guarantee).
    """
    if failure_budget < 0:
        raise ValueError("failure_budget must be non-negative")
    failure_size = min(int(failure_budget), len(indices))
    scenarios = list(combinations(indices, failure_size))
    if not scenarios:
        scenarios = [()]

    failed_examples: list[list[int]] = []
    failing_scenario_count = 0
    worst_surviving_count = len(indices)
    for failed in scenarios:
        failed_set = set(failed)
        surviving = [index for index in indices if index not in failed_set]
        worst_surviving_count = min(worst_surviving_count, len(surviving))
        passed, _ = _post_gate_support_satisfied(
            surviving,
            normalized_centers,
            min_views_per_cluster,
            min_baseline_ratio,
        )
        if not passed:
            failing_scenario_count += 1
            if len(failed_examples) < 8:
                failed_examples.append([int(index) for index in failed])

    return {
        "passed": failing_scenario_count == 0,
        "failure_budget": int(failure_budget),
        "failure_subset_size": int(failure_size),
        "failure_scenario_count": len(scenarios),
        "failing_scenario_count": failing_scenario_count,
        "failing_scenario_examples": failed_examples,
        "worst_surviving_count": int(worst_surviving_count),
    }


def _select_joint_gate_reserves(
    *,
    image_names: list[str],
    candidate_indices: list[int],
    selected: list[int],
    normalized_centers: np.ndarray,
    features: np.ndarray,
    anchor_features: np.ndarray,
    cluster_labels: np.ndarray,
    n_view_clusters: int,
    post_gate_min_views_per_cluster: int,
    min_baseline_ratio: float,
    max_borrowed_support_distance: float,
    reserve_max_pose_distance: float,
    failure_budget: int,
    failure_budget_by_cluster: dict[int, int] | None,
    joint_reserve_candidate_limit: int,
) -> dict:
    """Choose a jointly robust reserve set for correlated gate failures.

    The search is intentionally small and deterministic: it considers the
    clean, pose-local candidates nearest to a cluster's primary support, then
    finds the smallest reserve set that survives every ``failure_budget``
    hard-gate failure pattern.  A candidate may not be silently used merely
    because it makes one currently failing view look replaceable.
    """
    if failure_budget < 1:
        raise ValueError("Joint reserve selection requires a positive failure budget")
    if joint_reserve_candidate_limit < 1:
        raise ValueError("joint_reserve_candidate_limit must be positive")

    reserve_indices: list[int] = []
    cluster_records: list[dict] = []
    certified = True
    selected_set = set(selected)
    for cluster_index in range(n_view_clusters):
        cluster_failure_budget = int(
            (failure_budget_by_cluster or {}).get(cluster_index, failure_budget)
        )
        if cluster_failure_budget < 1:
            raise ValueError("Each cluster failure budget must be positive")
        support_distances = {
            index: float(np.linalg.norm(features[index] - anchor_features[cluster_index]))
            for index in candidate_indices
        }
        support_candidates = [
            index
            for index in candidate_indices
            if cluster_labels[index] == cluster_index
            or support_distances[index] <= max_borrowed_support_distance
        ]
        primary_supports = [
            index
            for index in selected
            if cluster_labels[index] == cluster_index
            or float(np.linalg.norm(features[index] - anchor_features[cluster_index]))
            <= max_borrowed_support_distance
        ]
        primary_satisfied, primary_pair = _post_gate_support_satisfied(
            primary_supports,
            normalized_centers,
            post_gate_min_views_per_cluster,
            min_baseline_ratio,
        )
        if not primary_satisfied:
            raise RuntimeError(
                "Primary Chart selection does not contain a baseline-separated post-gate "
                f"support set for view cluster {cluster_index}; do not add reserves to hide "
                "an invalid primary selection."
            )

        scored_candidates = []
        for candidate_index in support_candidates:
            if candidate_index in selected_set:
                continue
            pose_distance = min(
                float(np.linalg.norm(features[candidate_index] - features[primary_index]))
                for primary_index in primary_supports
            )
            if pose_distance > reserve_max_pose_distance + 1e-12:
                continue
            scored_candidates.append((pose_distance, candidate_index))
        scored_candidates.sort(key=lambda value: (value[0], value[1]))
        searched_candidates = scored_candidates[:joint_reserve_candidate_limit]

        primary_audit = _failure_budget_support_audit(
            primary_supports,
            normalized_centers,
            post_gate_min_views_per_cluster,
            min_baseline_ratio,
            cluster_failure_budget,
        )
        chosen: tuple[int, ...] | None = None
        chosen_audit: dict | None = None
        chosen_score: tuple[float, float, tuple[int, ...]] | None = None
        # A primary set that is already robust should not acquire gratuitous
        # dense Chart evidence.  Otherwise, search the smallest feasible set.
        # A two-view baseline needs two temporal substitutes even for a single
        # failure (one substitute near each endpoint), so the cardinality
        # lower bound alone is not an upper bound on reserves.
        candidate_indices_for_search = [value[1] for value in searched_candidates]
        distance_by_index = {value[1]: float(value[0]) for value in searched_candidates}
        max_reserve_count = min(
            len(candidate_indices_for_search),
            cluster_failure_budget + post_gate_min_views_per_cluster - 1,
        )
        for reserve_count in range(
            0, max_reserve_count + 1
        ):
            if len(primary_supports) + reserve_count < (
                post_gate_min_views_per_cluster + cluster_failure_budget
            ):
                continue
            for candidate_combo in combinations(candidate_indices_for_search, reserve_count):
                audit = _failure_budget_support_audit(
                    [*primary_supports, *candidate_combo],
                    normalized_centers,
                    post_gate_min_views_per_cluster,
                    min_baseline_ratio,
                    cluster_failure_budget,
                )
                if not audit["passed"]:
                    continue
                score = (
                    float(sum(distance_by_index[index] for index in candidate_combo)),
                    float(max((distance_by_index[index] for index in candidate_combo), default=0.0)),
                    tuple(int(index) for index in candidate_combo),
                )
                if chosen_score is None or score < chosen_score:
                    chosen = tuple(int(index) for index in candidate_combo)
                    chosen_audit = audit
                    chosen_score = score
            if chosen is not None:
                break

        if chosen is None:
            chosen = ()
            chosen_audit = primary_audit
        assert chosen_audit is not None
        certified = certified and bool(chosen_audit["passed"])
        reserve_indices.extend(chosen)
        cluster_records.append(
            {
                "cluster": cluster_index,
                "support_candidate_count": len(support_candidates),
                "primary_support_image_idx": [int(index) for index in primary_supports],
                "primary_support_image_names": [image_names[index] for index in primary_supports],
                "primary_baseline_pair": (
                    [int(index) for index in primary_pair]
                    if primary_pair is not None
                    else None
                ),
                "primary_failure_budget_audit": primary_audit,
                "gate_failure_budget": cluster_failure_budget,
                "eligible_joint_reserve_count": len(scored_candidates),
                "searched_joint_reserve_candidate_count": len(searched_candidates),
                "joint_reserve_image_idx": [int(index) for index in chosen],
                "joint_reserve_image_names": [image_names[index] for index in chosen],
                "joint_reserve_pose_distances": [
                    float(distance_by_index[index]) for index in chosen
                ],
                "joint_failure_budget_audit": chosen_audit,
                "passed": bool(chosen_audit["passed"]),
            }
        )

    reserve_indices = sorted(set(reserve_indices))
    alignment_indices = sorted(set(selected) | set(reserve_indices))
    return {
        "method": "baseline_primary_plus_joint_gate_resilience",
        "post_gate_min_views_per_cluster": post_gate_min_views_per_cluster,
        "min_baseline_ratio": min_baseline_ratio,
        "reserve_max_pose_distance": reserve_max_pose_distance,
        "gate_failure_budget": int(failure_budget),
        "gate_failure_budget_by_cluster": {
            str(cluster_index): int(
                (failure_budget_by_cluster or {}).get(cluster_index, failure_budget)
            )
            for cluster_index in range(n_view_clusters)
        },
        "joint_reserve_candidate_limit": int(joint_reserve_candidate_limit),
        "certified_for_gate_failure_budget": bool(certified),
        "reserve_image_idx": reserve_indices,
        "reserve_image_names": [image_names[index] for index in reserve_indices],
        "alignment_image_idx": alignment_indices,
        "alignment_image_names": [image_names[index] for index in alignment_indices],
        "substitutions": [],
        "clusters": cluster_records,
    }


def select_gate_replacement_reserves(
    image_names: list[str],
    candidate_image_names: list[str],
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
    selected_indices: list[int],
    *,
    n_view_clusters: int,
    post_gate_min_views_per_cluster: int,
    min_baseline_ratio: float,
    max_borrowed_support_distance: float,
    reserve_max_pose_distance: float,
    reserves_per_vulnerable_support: int = 1,
    gate_failure_budget: int = 1,
    observed_rejected_image_names: list[str] | None = None,
    joint_reserve_candidate_limit: int = 48,
    position_weight: float = 1.0,
    direction_weight: float = 0.35,
) -> dict:
    """Add clean, pose-near substitutes for gate-critical primary Charts.

    The primary selection is globally diverse, so it deliberately rejects
    temporal near-duplicates.  That is right for geometry but wrong for a
    hard post-alignment gate: if one of a cluster's two independent camera
    modes fails, the remaining primary cannot form a baseline pair.  This
    routine retains the primary set unchanged and appends a clean nearby
    alternate for each *critical* primary support.  The alternate is not
    counted as an independent support until it replaces a gated-out primary.
    """
    if n_view_clusters <= 0:
        raise ValueError("n_view_clusters must be positive")
    if post_gate_min_views_per_cluster < 1:
        raise ValueError("post_gate_min_views_per_cluster must be positive")
    if min_baseline_ratio < 0.0:
        raise ValueError("min_baseline_ratio must be non-negative")
    if max_borrowed_support_distance < 0.0:
        raise ValueError("max_borrowed_support_distance must be non-negative")
    if reserve_max_pose_distance < 0.0:
        raise ValueError("reserve_max_pose_distance must be non-negative")
    if reserves_per_vulnerable_support < 0:
        raise ValueError("reserves_per_vulnerable_support must be non-negative")
    if gate_failure_budget < 1:
        raise ValueError("gate_failure_budget must be positive when reserves are requested")

    name_to_index = {name: index for index, name in enumerate(image_names)}
    unknown_candidates = [name for name in candidate_image_names if name not in name_to_index]
    if unknown_candidates:
        raise ValueError(f"Candidate image is not in the scene list: {unknown_candidates[0]}")
    candidate_indices = sorted({name_to_index[name] for name in candidate_image_names})
    selected = sorted({int(index) for index in selected_indices})
    unknown_selected = [index for index in selected if index < 0 or index >= len(image_names)]
    if unknown_selected:
        raise ValueError(f"Selected chart index is outside the scene list: {unknown_selected[0]}")
    non_candidate_selected = [index for index in selected if index not in set(candidate_indices)]
    if non_candidate_selected:
        raise ValueError(
            "Primary Chart is not in the clean candidate pool: "
            f"{image_names[non_candidate_selected[0]]}"
        )

    normalized_centers, directions, _ = build_pose_geometry(image_names, poses)
    features = np.concatenate(
        [position_weight * normalized_centers, direction_weight * directions],
        axis=1,
    )
    anchor_indices = _farthest_point_indices(features, n_view_clusters)
    anchor_features = features[np.asarray(anchor_indices)]
    cluster_labels = np.argmin(
        np.linalg.norm(features[:, None, :] - anchor_features[None, :, :], axis=2),
        axis=1,
    )

    failure_budget_by_cluster = {cluster_index: int(gate_failure_budget) for cluster_index in range(n_view_clusters)}
    observed_rejected_count = 0
    if observed_rejected_image_names:
        unknown_rejected = [
            name for name in observed_rejected_image_names if name not in name_to_index
        ]
        if unknown_rejected:
            raise ValueError(
                "Observed rejected chart is not in the scene list: "
                f"{unknown_rejected[0]}"
            )
        observed_counts = {cluster_index: 0 for cluster_index in range(n_view_clusters)}
        for name in sorted(set(observed_rejected_image_names)):
            observed_counts[int(cluster_labels[name_to_index[name]])] += 1
            observed_rejected_count += 1
        failure_budget_by_cluster = {
            cluster_index: max(int(gate_failure_budget), observed_counts[cluster_index])
            for cluster_index in range(n_view_clusters)
        }

    if max(failure_budget_by_cluster.values(), default=gate_failure_budget) > 1:
        return _select_joint_gate_reserves(
            image_names=image_names,
            candidate_indices=candidate_indices,
            selected=selected,
            normalized_centers=normalized_centers,
            features=features,
            anchor_features=anchor_features,
            cluster_labels=cluster_labels,
            n_view_clusters=n_view_clusters,
            post_gate_min_views_per_cluster=post_gate_min_views_per_cluster,
            min_baseline_ratio=min_baseline_ratio,
            max_borrowed_support_distance=max_borrowed_support_distance,
            reserve_max_pose_distance=reserve_max_pose_distance,
            failure_budget=gate_failure_budget,
            failure_budget_by_cluster=failure_budget_by_cluster,
            joint_reserve_candidate_limit=joint_reserve_candidate_limit,
        )

    reserve_indices: list[int] = []
    substitution_records: list[dict] = []
    cluster_records: list[dict] = []
    certified = True
    for cluster_index in range(n_view_clusters):
        support_distances = {
            index: float(np.linalg.norm(features[index] - anchor_features[cluster_index]))
            for index in candidate_indices
        }
        support_candidates = [
            index
            for index in candidate_indices
            if cluster_labels[index] == cluster_index
            or support_distances[index] <= max_borrowed_support_distance
        ]
        primary_supports = [
            index
            for index in selected
            if cluster_labels[index] == cluster_index
            or float(np.linalg.norm(features[index] - anchor_features[cluster_index]))
            <= max_borrowed_support_distance
        ]
        primary_satisfied, primary_pair = _post_gate_support_satisfied(
            primary_supports,
            normalized_centers,
            post_gate_min_views_per_cluster,
            min_baseline_ratio,
        )
        if not primary_satisfied:
            raise RuntimeError(
                "Primary Chart selection does not contain a baseline-separated post-gate "
                f"support set for view cluster {cluster_index}; do not add reserves to hide "
                "an invalid primary selection."
            )

        vulnerable = []
        for primary_index in primary_supports:
            retained = [index for index in primary_supports if index != primary_index]
            retained_satisfied, _ = _post_gate_support_satisfied(
                retained,
                normalized_centers,
                post_gate_min_views_per_cluster,
                min_baseline_ratio,
            )
            if not retained_satisfied:
                vulnerable.append(primary_index)

        cluster_substitutions = []
        for primary_index in vulnerable:
            eligible = []
            for candidate_index in support_candidates:
                if candidate_index in selected:
                    continue
                pose_distance = float(
                    np.linalg.norm(features[candidate_index] - features[primary_index])
                )
                if pose_distance > reserve_max_pose_distance + 1e-12:
                    continue
                substituted = [
                    index for index in primary_supports if index != primary_index
                ] + [candidate_index]
                substitution_satisfied, replacement_pair = _post_gate_support_satisfied(
                    substituted,
                    normalized_centers,
                    post_gate_min_views_per_cluster,
                    min_baseline_ratio,
                )
                if not substitution_satisfied:
                    continue
                eligible.append((pose_distance, candidate_index, replacement_pair))
            eligible.sort(key=lambda value: (value[0], value[1]))
            chosen = eligible[:reserves_per_vulnerable_support]
            passed = len(chosen) >= reserves_per_vulnerable_support
            certified = certified and passed
            record = {
                "cluster": cluster_index,
                "primary_image_idx": int(primary_index),
                "primary_image_name": image_names[primary_index],
                "eligible_replacement_count": len(eligible),
                "reserve_image_idx": [int(value[1]) for value in chosen],
                "reserve_image_names": [image_names[value[1]] for value in chosen],
                "reserve_pose_distances": [float(value[0]) for value in chosen],
                "replacement_baseline_pairs": [
                    [int(index) for index in value[2]] if value[2] is not None else None
                    for value in chosen
                ],
                "passed": passed,
            }
            substitution_records.append(record)
            cluster_substitutions.append(record)
            reserve_indices.extend(record["reserve_image_idx"])

        cluster_records.append(
            {
                "cluster": cluster_index,
                "support_candidate_count": len(support_candidates),
                "primary_support_image_idx": [int(index) for index in primary_supports],
                "primary_support_image_names": [image_names[index] for index in primary_supports],
                "primary_baseline_pair": (
                    [int(index) for index in primary_pair]
                    if primary_pair is not None
                    else None
                ),
                "vulnerable_primary_image_idx": [int(index) for index in vulnerable],
                "vulnerable_primary_image_names": [image_names[index] for index in vulnerable],
                "substitutions": cluster_substitutions,
            }
        )

    reserve_indices = sorted(set(reserve_indices))
    alignment_indices = sorted(set(selected) | set(reserve_indices))
    return {
        "method": "baseline_primary_plus_pose_near_gate_replacement",
        "post_gate_min_views_per_cluster": post_gate_min_views_per_cluster,
        "min_baseline_ratio": min_baseline_ratio,
        "reserve_max_pose_distance": reserve_max_pose_distance,
        "reserves_per_vulnerable_support": reserves_per_vulnerable_support,
        "gate_failure_budget": 1,
        "gate_failure_budget_by_cluster": {str(index): 1 for index in range(n_view_clusters)},
        "observed_gate_rejection_count": observed_rejected_count,
        "certified_for_one_primary_gate_rejection": certified,
        "certified_for_gate_failure_budget": certified,
        "reserve_image_idx": reserve_indices,
        "reserve_image_names": [image_names[index] for index in reserve_indices],
        "alignment_image_idx": alignment_indices,
        "alignment_image_names": [image_names[index] for index in alignment_indices],
        "substitutions": substitution_records,
        "clusters": cluster_records,
    }


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
    parser.add_argument(
        "--post-gate-min-views-per-cluster",
        type=int,
        default=None,
        help=(
            "Support count required after the aligned-Chart hard gate.  This may be "
            "lower than the broad candidate-set coverage requirement when pose-near "
            "gate replacements are explicitly reserved."
        ),
    )
    parser.add_argument(
        "--reserve-max-pose-distance",
        type=float,
        default=None,
        help=(
            "Maximum pose+direction feature distance between a gate-critical primary "
            "Chart and its clean replacement reserve."
        ),
    )
    parser.add_argument(
        "--reserves-per-vulnerable-support",
        type=int,
        default=0,
        help=(
            "Append this many clean pose-near substitutes for every primary support "
            "whose removal would violate post-gate coverage."
        ),
    )
    parser.add_argument(
        "--gate-failure-budget",
        type=int,
        default=1,
        help=(
            "Maximum correlated hard-gate failures that every pose cluster's "
            "primary-plus-reserve set must survive. Values above one use the "
            "joint reserve search instead of independent per-primary substitutes."
        ),
    )
    parser.add_argument(
        "--joint-reserve-candidate-limit",
        type=int,
        default=48,
        help=(
            "Maximum clean pose-local reserve candidates searched per cluster "
            "by the correlated-failure reserve solver."
        ),
    )
    parser.add_argument(
        "--require-gate-replacement-reserves",
        action="store_true",
        help="Fail selection unless every gate-critical primary has the requested replacement reserve.",
    )
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
    parser.add_argument(
        "--static-support-mask-pickle",
        type=Path,
        default=None,
        help=(
            "Binary keep-mask source used to measure usable static surface support. "
            "Unlike --semantic-filter this is a continuous per-image criterion."
        ),
    )
    parser.add_argument("--static-support-mask-dataset-path", type=Path, default=None)
    parser.add_argument(
        "--static-support-mask-indices",
        nargs="*",
        type=int,
        default=None,
        help="Keep-mask channels intersected when measuring static support.",
    )
    parser.add_argument(
        "--min-static-support-ratio",
        type=float,
        default=0.0,
        help="Minimum usable static-pixel fraction for a Chart candidate.",
    )
    parser.add_argument("--static-support-ratio-cache", type=Path, default=None)
    parser.add_argument("--occlusion-mask-indices", nargs="*", type=int, default=[])
    parser.add_argument("--max-occlusion-invalid-ratio", type=float, default=0.15)
    parser.add_argument("--occlusion-ratio-cache", type=Path, default=None)
    parser.add_argument("--tree-mask-pickle", type=Path, default=None)
    parser.add_argument("--tree-mask-dataset-path", type=Path, default=None)
    parser.add_argument("--tree-mask-index", type=int, default=3)
    parser.add_argument("--max-tree-invalid-ratio", type=float, default=0.15)
    parser.add_argument("--tree-ratio-cache", type=Path, default=None)
    parser.add_argument(
        "--tree-candidate-policy",
        choices=["hard_ratio", "soft_penalty", "ignore"],
        default="hard_ratio",
        help=(
            "Whether tree coverage vetoes a whole Chart, only lowers its weak "
            "reliability prior, or is ignored at candidate-selection time."
        ),
    )
    parser.add_argument("--quality-filter", action="store_true", help="Filter low-quality images before pose FPS.")
    parser.add_argument(
        "--quality-candidate-policy",
        choices=["hard_filter", "soft_penalty"],
        default="hard_filter",
        help=(
            "Reject low-quality frames outright, or retain them and use quality "
            "only as a weak coverage tie-breaker."
        ),
    )
    parser.add_argument("--min-sharpness", type=float, default=1e-4)
    parser.add_argument("--max-extreme-ratio", type=float, default=0.35)
    parser.add_argument("--max-shadow-ratio", type=float, default=1.0)
    parser.add_argument("--min-lower-mean-intensity", type=float, default=0.0)
    parser.add_argument("--min-valid-ratio", type=float, default=0.75)
    parser.add_argument("--mask-pickle", type=Path, default=None)
    parser.add_argument("--mask-dataset-path", type=Path, default=None)
    parser.add_argument("--mask-indices", nargs="*", type=int, default=None)
    parser.add_argument("--quality-mask-indices", nargs="*", type=int, default=None)
    parser.add_argument("--quality-score-mask-pickle", type=Path, default=None)
    parser.add_argument("--quality-score-mask-dataset-path", type=Path, default=None)
    parser.add_argument("--quality-score-cache", type=Path, default=None)
    parser.add_argument(
        "--candidate-reliability-weight",
        type=float,
        default=0.0,
        help=(
            "Weak coverage-tie penalty for low static support/photometric quality. "
            "Zero reproduces pure pose coverage."
        ),
    )
    parser.add_argument(
        "--min-views-per-sequence",
        type=int,
        default=0,
        help=(
            "Independent same-sequence static Charts reserved for every temporal "
            "segment with at least this many training views."
        ),
    )
    parser.add_argument(
        "--post-gate-min-views-per-sequence",
        type=int,
        default=None,
        help="Same-sequence supports that must remain after accepted hard-gate rejections.",
    )
    parser.add_argument(
        "--sequence-gate-failure-budget",
        type=int,
        default=0,
        help="Worst-case hard-gate rejections that each same-sequence seed set must survive.",
    )
    parser.add_argument(
        "--sequence-coverage-mode",
        choices=["off", "soft", "strict"],
        default="off",
        help="Record, or hard-require, same-trajectory Chart support before global coverage.",
    )
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

    if args.reserves_per_vulnerable_support < 0:
        raise ValueError("--reserves-per-vulnerable-support must be non-negative")
    if args.gate_failure_budget < 1:
        raise ValueError("--gate-failure-budget must be positive")
    if args.joint_reserve_candidate_limit < 1:
        raise ValueError("--joint-reserve-candidate-limit must be positive")
    if not 0.0 <= args.min_static_support_ratio <= 1.0:
        raise ValueError("--min-static-support-ratio must be in [0, 1]")
    if args.static_support_mask_pickle is None:
        if args.static_support_mask_indices is not None or args.min_static_support_ratio > 0.0:
            raise ValueError(
                "--static-support-mask-pickle is required when static support filtering is enabled"
            )
    elif not args.static_support_mask_indices:
        raise ValueError(
            "--static-support-mask-indices is required with --static-support-mask-pickle"
        )
    if args.candidate_reliability_weight < 0.0:
        raise ValueError("--candidate-reliability-weight must be non-negative")
    if args.min_views_per_sequence < 0:
        raise ValueError("--min-views-per-sequence must be non-negative")
    if args.min_views_per_sequence == 0 and args.sequence_coverage_mode != "off":
        raise ValueError("--sequence-coverage-mode requires --min-views-per-sequence > 0")
    if args.min_views_per_sequence > 0 and args.sequence_coverage_mode == "off":
        raise ValueError(
            "--min-views-per-sequence requires --sequence-coverage-mode soft or strict"
        )
    post_gate_sequence_support = (
        args.min_views_per_sequence
        if args.post_gate_min_views_per_sequence is None
        else args.post_gate_min_views_per_sequence
    )
    if post_gate_sequence_support < 0:
        raise ValueError("--post-gate-min-views-per-sequence must be non-negative")
    if args.sequence_gate_failure_budget < 0:
        raise ValueError("--sequence-gate-failure-budget must be non-negative")
    if args.min_views_per_sequence > 0:
        if post_gate_sequence_support < 1:
            raise ValueError("same-sequence coverage needs positive post-gate support")
        if args.min_views_per_sequence < post_gate_sequence_support + args.sequence_gate_failure_budget:
            raise ValueError(
                "--min-views-per-sequence must be at least post-gate support plus "
                "--sequence-gate-failure-budget"
            )
    if args.require_gate_replacement_reserves and args.reserves_per_vulnerable_support < 1:
        raise ValueError(
            "--require-gate-replacement-reserves requires "
            "--reserves-per-vulnerable-support >= 1"
        )
    if args.reserves_per_vulnerable_support > 0:
        if args.view_clusters <= 0:
            raise ValueError("Gate replacement reserves require --view-clusters > 0")
        if args.post_gate_min_views_per_cluster is None:
            raise ValueError(
                "--post-gate-min-views-per-cluster is required when gate replacement reserves are enabled"
            )
        if args.reserve_max_pose_distance is None:
            raise ValueError(
                "--reserve-max-pose-distance is required when gate replacement reserves are enabled"
            )

    image_names, poses = load_scene_poses(args.scene_path)
    candidate_image_names = image_names
    # A CambridgeMaskLookup owns the complete pickle payload.  Several
    # criteria below commonly use different channels of the same (large)
    # semantic file; reuse one backing object instead of deserializing it once
    # per criterion and exhausting host memory on outdoor captures.
    lookup_roots: dict[tuple[Path, Path], CambridgeMaskLookup] = {}

    def lookup_for(
        mask_pickle: Path,
        dataset_path: Path,
        mask_indices: list[int],
    ) -> CambridgeMaskLookup:
        key = (mask_pickle.expanduser().resolve(), dataset_path.expanduser().resolve())
        root = lookup_roots.get(key)
        if root is None:
            root = CambridgeMaskLookup(dataset_path, mask_pickle, mask_indices=mask_indices)
            lookup_roots[key] = root
        return root.with_indices(mask_indices)

    mask_lookup: Optional[CambridgeMaskLookup] = None
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
            mask_lookup = lookup_for(
                args.mask_pickle,
                args.mask_dataset_path or args.scene_path,
                args.semantic_mask_indices,
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
            occlusion_lookup = lookup_for(
                args.mask_pickle,
                args.mask_dataset_path or args.scene_path,
                args.occlusion_mask_indices,
            )
            _, occlusion_invalid_ratios = semantic_filtered_names(
                image_names,
                occlusion_lookup,
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
    tree_lookup: Optional[CambridgeMaskLookup] = None
    if args.tree_mask_pickle is not None:
        tree_indices = [args.tree_mask_index]
        if args.tree_ratio_cache is not None:
            tree_invalid_ratios = load_semantic_ratio_cache(
                args.tree_ratio_cache,
                image_names,
                args.tree_mask_pickle,
                tree_indices,
            )
        tree_lookup = lookup_for(
            args.tree_mask_pickle,
            args.tree_mask_dataset_path or args.mask_dataset_path or args.scene_path,
            tree_indices,
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
        if args.tree_candidate_policy == "hard_ratio":
            candidate_image_names = [
                name for name in candidate_image_names
                if tree_invalid_ratios[name] <= args.max_tree_invalid_ratio + 1e-12
            ]

    static_support = None
    static_support_invalid_ratios = None
    if args.static_support_mask_pickle is not None:
        static_indices = list(args.static_support_mask_indices or [])
        if args.static_support_ratio_cache is not None:
            static_support_invalid_ratios = load_semantic_ratio_cache(
                args.static_support_ratio_cache,
                image_names,
                args.static_support_mask_pickle,
                static_indices,
            )
        if static_support_invalid_ratios is None:
            static_lookup = lookup_for(
                args.static_support_mask_pickle,
                args.static_support_mask_dataset_path or args.scene_path,
                static_indices,
            )
            static_support = static_support_ratios(
                image_names,
                static_lookup,
                static_indices,
            )
            static_support_invalid_ratios = {
                name: float(1.0 - support) for name, support in static_support.items()
            }
            if args.static_support_ratio_cache is not None:
                write_semantic_ratio_cache(
                    args.static_support_ratio_cache,
                    static_support_invalid_ratios,
                    args.static_support_mask_pickle,
                    static_indices,
                )
        else:
            static_support = {
                name: float(1.0 - invalid)
                for name, invalid in static_support_invalid_ratios.items()
            }
        candidate_image_names = [
            name for name in candidate_image_names
            if static_support[name] >= args.min_static_support_ratio - 1e-12
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
        quality_mask_pickle = args.quality_score_mask_pickle or args.mask_pickle
        quality_mask_dataset_path = (
            args.quality_score_mask_dataset_path
            or args.mask_dataset_path
            or args.scene_path
        )
        if args.quality_score_cache is not None:
            quality_scores = load_quality_score_cache(
                args.quality_score_cache,
                args.scene_path,
                candidate_image_names,
                quality_mask_pickle,
                quality_mask_indices,
            )
        quality_mask_lookup = None
        if quality_scores is None and (
            quality_mask_pickle is not None
            and (args.min_valid_ratio > 0.0 or args.quality_mask_indices is not None)
        ):
            quality_mask_lookup = lookup_for(
                quality_mask_pickle,
                quality_mask_dataset_path,
                quality_mask_indices,
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
                    quality_mask_pickle,
                    quality_mask_indices,
                )
        if args.quality_candidate_policy == "hard_filter":
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

    candidate_reliability = candidate_reliability_scores(
        candidate_image_names,
        quality_scores=quality_scores,
        static_support=static_support,
        tree_invalid_ratios=(
            tree_invalid_ratios
            if args.tree_candidate_policy == "soft_penalty"
            else None
        ),
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
            candidate_reliability=candidate_reliability,
            reliability_weight=args.candidate_reliability_weight,
            min_views_per_sequence=args.min_views_per_sequence,
            post_gate_min_views_per_sequence=post_gate_sequence_support,
            sequence_gate_failure_budget=args.sequence_gate_failure_budget,
            sequence_coverage_mode=args.sequence_coverage_mode,
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
    gate_replacement_reserves = None
    if args.reserves_per_vulnerable_support > 0:
        gate_replacement_reserves = select_gate_replacement_reserves(
            image_names=image_names,
            candidate_image_names=candidate_image_names,
            poses=poses,
            selected_indices=selected,
            n_view_clusters=args.view_clusters,
            post_gate_min_views_per_cluster=args.post_gate_min_views_per_cluster,
            min_baseline_ratio=args.min_baseline_ratio,
            max_borrowed_support_distance=args.max_borrowed_support_distance,
            reserve_max_pose_distance=args.reserve_max_pose_distance,
            reserves_per_vulnerable_support=args.reserves_per_vulnerable_support,
            gate_failure_budget=args.gate_failure_budget,
            joint_reserve_candidate_limit=args.joint_reserve_candidate_limit,
            position_weight=args.position_weight,
            direction_weight=args.direction_weight,
        )
        if (
            args.require_gate_replacement_reserves
            and not gate_replacement_reserves["certified_for_gate_failure_budget"]
        ):
            failed = []
            for record in gate_replacement_reserves["clusters"]:
                if record.get("passed"):
                    continue
                failed.extend(record.get("primary_support_image_names", []))
            raise RuntimeError(
                "No clean jointly robust gate-reserve set exists for critical Chart support(s): "
                + ", ".join(failed[:5])
            )
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
    if gate_replacement_reserves is not None:
        result["gate_replacement_reserves"] = gate_replacement_reserves
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
            "candidate_policy": args.tree_candidate_policy,
            "hard_ratio_kept": sum(
                ratio <= args.max_tree_invalid_ratio + 1e-12
                for ratio in tree_invalid_ratios.values()
            ),
            "total": len(image_names),
        }
        result["tree_invalid_ratios"] = {
            name: tree_invalid_ratios[name] for name in selected_names
        }
    if static_support is not None:
        result["static_support_filter"] = {
            "mask_pickle": str(args.static_support_mask_pickle),
            "mask_indices": list(args.static_support_mask_indices or []),
            "min_support_ratio": args.min_static_support_ratio,
            "kept": sum(
                support >= args.min_static_support_ratio - 1e-12
                for support in static_support.values()
            ),
            "total": len(image_names),
        }
        result["static_support_ratios"] = {
            name: static_support[name] for name in selected_names
        }
    if quality_scores is not None:
        result["quality_filter"] = {
            "candidate_policy": args.quality_candidate_policy,
            "kept": len(candidate_image_names),
            "total": len(image_names),
            "min_sharpness": args.min_sharpness,
            "max_extreme_ratio": args.max_extreme_ratio,
            "max_shadow_ratio": args.max_shadow_ratio,
            "min_lower_mean_intensity": args.min_lower_mean_intensity,
            "min_valid_ratio": args.min_valid_ratio,
        }
        result["quality_scores"] = {name: quality_scores[name] for name in selected_names}
    if candidate_reliability:
        result["candidate_reliability"] = {
            "weight": args.candidate_reliability_weight,
            # Feedback re-selection must score replacement views by the same
            # static/photometric evidence as the initial set.  Persisting only
            # ``selected_scores`` made every previously unselected candidate
            # fall back to a neutral 0.5 score, so an outdoor feedback round
            # could replace a failed clear facade with an equally pose-useful
            # but visibly poor neighbour.  This map is small (one float per
            # eligible Chart) and keeps the full candidate decision auditable.
            "scores": {
                name: candidate_reliability[name] for name in candidate_image_names
            },
            "selected_scores": {
                name: candidate_reliability[name] for name in selected_names
            },
            "candidate_score_summary": {
                "min": float(min(candidate_reliability.values())),
                "mean": float(np.mean(list(candidate_reliability.values()))),
                "max": float(max(candidate_reliability.values())),
            },
        }
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
