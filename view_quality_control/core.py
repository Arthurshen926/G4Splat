from __future__ import annotations

from collections import Counter
from typing import Iterable

import cv2
import numpy as np


def compute_image_quality(image_bgr: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    """Compute blur/exposure statistics on a static, non-sky image mask."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape HxWx3, got {image_bgr.shape}")
    if valid_mask.shape != image_bgr.shape[:2]:
        raise ValueError(
            f"Mask shape {valid_mask.shape} does not match image {image_bgr.shape[:2]}"
        )

    keep = valid_mask.astype(np.bool_, copy=False)
    valid_count = int(np.count_nonzero(keep))
    if valid_count == 0:
        return {
            "valid_ratio": 0.0,
            "laplacian_variance": 0.0,
            "gradient_variance": 0.0,
            "gray_mean": 0.0,
            "gray_p05": 0.0,
            "gray_p50": 0.0,
            "gray_p95": 0.0,
            "contrast_p90": 0.0,
            "dark_ratio": 1.0,
            "highlight_ratio": 0.0,
            "clipped_ratio": 0.0,
            "saturated_ratio": 0.0,
        }

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    eroded_keep = cv2.erode(keep.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if not np.any(eroded_keep):
        eroded_keep = keep

    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    values = gray[keep]
    p05, p50, p95 = np.percentile(values, [5, 50, 95])
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[..., 1].astype(np.float32) / 255.0

    return {
        "valid_ratio": float(valid_count / keep.size),
        "laplacian_variance": float(np.var(laplacian[eroded_keep])),
        "gradient_variance": float(np.var(gradient_magnitude[eroded_keep])),
        "gray_mean": float(values.mean()),
        "gray_p05": float(p05),
        "gray_p50": float(p50),
        "gray_p95": float(p95),
        "contrast_p90": float(p95 - p05),
        "dark_ratio": float((values <= 0.10).mean()),
        "highlight_ratio": float((values >= 0.90).mean()),
        "clipped_ratio": float(((values <= 0.02) | (values >= 0.98)).mean()),
        "saturated_ratio": float((saturation[keep] >= 0.95).mean()),
    }


def percentile_badness(values: Iterable[float], higher_is_bad: bool) -> np.ndarray:
    values_array = np.asarray(list(values), dtype=np.float64)
    if values_array.ndim != 1:
        raise ValueError("Expected a one-dimensional metric array")
    if len(values_array) == 0:
        return values_array

    if len(values_array) == 1:
        ranks = np.full(1, 0.5, dtype=np.float64)
    else:
        _, inverse, counts = np.unique(values_array, return_inverse=True, return_counts=True)
        starts = np.cumsum(np.r_[0, counts[:-1]])
        average_positions = starts + (counts - 1) / 2.0
        ranks = average_positions[inverse] / (len(values_array) - 1)
    return ranks if higher_is_bad else 1.0 - ranks


def _quantile(records: list[dict], key: str, q: float) -> float:
    return float(np.quantile([float(record[key]) for record in records], q))


def classify_view_records(
    records: list[dict],
    *,
    max_reject_fraction: float = 0.05,
    max_group_reject_fraction: float = 0.10,
) -> tuple[list[dict], dict]:
    """Classify views conservatively using independent image/geometry signals.

    The reject list is capped globally and per pose/sequence group. A frame is
    never rejected solely because it occupies the tail of one metric.
    """
    if not records:
        raise ValueError("No view records were supplied")
    if not 0.0 <= max_reject_fraction < 1.0:
        raise ValueError("max_reject_fraction must be in [0, 1)")

    output = [dict(record) for record in records]
    metric_specs = [
        ("laplacian_variance", False),
        ("track_count", False),
        ("shared_track_max", False),
        ("track_projection_valid_ratio", False),
        ("exposure_warning", True),
        ("semantic_warning", True),
        ("intrinsics_robust_z", True),
    ]
    badness_by_metric = {
        key: percentile_badness((record[key] for record in output), higher_is_bad)
        for key, higher_is_bad in metric_specs
    }

    thresholds = {
        "laplacian_q02": _quantile(output, "laplacian_variance", 0.02),
        "laplacian_q10": _quantile(output, "laplacian_variance", 0.10),
        "track_q02": _quantile(output, "track_count", 0.02),
        "track_q10": _quantile(output, "track_count", 0.10),
        "shared_q02": _quantile(output, "shared_track_max", 0.02),
        "shared_q10": _quantile(output, "shared_track_max", 0.10),
        "projection_q02": _quantile(output, "track_projection_valid_ratio", 0.02),
        "projection_q10": _quantile(output, "track_projection_valid_ratio", 0.10),
    }
    thresholds["projection_severe"] = min(0.90, thresholds["projection_q02"])
    thresholds["projection_mild"] = min(0.98, thresholds["projection_q10"])

    def evidence_groups(reasons: list[str]) -> set[str]:
        return {
            "geometry_support" if reason in {"low_tracks", "low_neighbor_support"} else reason
            for reason in reasons
        }

    for index, record in enumerate(output):
        component_badness = {
            key: float(values[index]) for key, values in badness_by_metric.items()
        }
        ordered_badness = sorted(component_badness.values(), reverse=True)
        risk_score = (
            0.55 * ordered_badness[0]
            + 0.30 * ordered_badness[1]
            + 0.15 * ordered_badness[2]
        )

        severe = []
        mild = []
        pose_recoverable = bool(record.get("pose_recoverable", True))
        quaternion_norm_error = float(record.get("quaternion_norm_error", 0.0))
        hard_invariant_failure = not pose_recoverable
        if hard_invariant_failure:
            severe.append("invalid_pose")
        elif quaternion_norm_error > 1e-3:
            severe.append("pose_requires_normalization")
        elif quaternion_norm_error > 1e-4:
            mild.append("pose_requires_normalization")
        if record["laplacian_variance"] < thresholds["laplacian_q02"]:
            severe.append("blur")
        elif record["laplacian_variance"] < thresholds["laplacian_q10"]:
            mild.append("blur")

        if record["track_count"] < thresholds["track_q02"]:
            severe.append("low_tracks")
        elif record["track_count"] < thresholds["track_q10"]:
            mild.append("low_tracks")

        if record["shared_track_max"] < thresholds["shared_q02"]:
            severe.append("low_neighbor_support")
        elif record["shared_track_max"] < thresholds["shared_q10"]:
            mild.append("low_neighbor_support")

        projection_ratio = record["track_projection_valid_ratio"]
        if projection_ratio < thresholds["projection_severe"]:
            severe.append("pose_projection")
        elif projection_ratio < thresholds["projection_mild"]:
            mild.append("pose_projection")

        if (
            record["gray_mean"] < 0.12
            or record["gray_mean"] > 0.88
            or record["dark_ratio"] > 0.60
            or record["clipped_ratio"] > 0.20
        ):
            severe.append("exposure")
        elif (
            record["gray_mean"] < 0.18
            or record["gray_mean"] > 0.82
            or record["dark_ratio"] > 0.35
            or record["clipped_ratio"] > 0.08
            or record["contrast_p90"] < 0.12
        ):
            mild.append("exposure")

        if record["thing_invalid_ratio"] > 0.40 or record["tree_invalid_ratio"] > 0.70:
            severe.append("occlusion")
        elif record["thing_invalid_ratio"] > 0.20 or record["tree_invalid_ratio"] > 0.50:
            mild.append("occlusion")

        if record["intrinsics_robust_z"] > 8.0:
            severe.append("intrinsics")
        elif record["intrinsics_robust_z"] > 4.0:
            mild.append("intrinsics")

        reject_candidate = hard_invariant_failure or (
            len(evidence_groups(severe)) >= 2
            or (risk_score >= 0.97 and len(evidence_groups(severe + mild)) >= 2)
        )
        record["risk_score"] = float(risk_score)
        record["risk_components"] = component_badness
        record["severe_reasons"] = sorted(set(severe))
        record["mild_reasons"] = sorted(set(mild))
        record["reject_candidate"] = bool(reject_candidate)
        record["hard_invariant_failure"] = hard_invariant_failure
        record["status"] = "clean"

    max_reject = int(np.floor(len(output) * max_reject_fraction))
    cluster_counts = Counter(record["pose_cluster"] for record in output)
    sequence_counts = Counter(record["sequence"] for record in output)
    cluster_rejected = Counter()
    sequence_rejected = Counter()

    invariant_rejected = [record for record in output if record["hard_invariant_failure"]]
    for record in invariant_rejected:
        record["status"] = "hard_reject"
        cluster_rejected[record["pose_cluster"]] += 1
        sequence_rejected[record["sequence"]] += 1

    candidates = sorted(
        (
            record
            for record in output
            if record["reject_candidate"] and not record["hard_invariant_failure"]
        ),
        key=lambda item: (-item["risk_score"], item["image_name"]),
    )
    rejected = list(invariant_rejected)
    for record in candidates:
        if len(rejected) >= max_reject:
            break
        cluster = record["pose_cluster"]
        sequence = record["sequence"]
        cluster_cap = max(1, int(np.floor(cluster_counts[cluster] * max_group_reject_fraction)))
        sequence_cap = max(1, int(np.floor(sequence_counts[sequence] * max_group_reject_fraction)))
        if cluster_rejected[cluster] >= cluster_cap or sequence_rejected[sequence] >= sequence_cap:
            record["coverage_guarded"] = True
            continue
        record["status"] = "hard_reject"
        cluster_rejected[cluster] += 1
        sequence_rejected[sequence] += 1
        rejected.append(record)

    for record in output:
        if record["status"] == "hard_reject":
            continue
        if record["reject_candidate"] or record["risk_score"] >= 0.85 or record["severe_reasons"]:
            record["status"] = "soft_keep"

    summary = {
        "total": len(output),
        "hard_reject": sum(record["status"] == "hard_reject" for record in output),
        "soft_keep": sum(record["status"] == "soft_keep" for record in output),
        "clean": sum(record["status"] == "clean" for record in output),
        "max_reject_fraction": max_reject_fraction,
        "actual_reject_fraction": float(len(rejected) / len(output)),
        "hard_invariant_reject": len(invariant_rejected),
        "pose_normalization_required": sum(
            bool(record.get("pose_normalization_required", False)) for record in output
        ),
        "thresholds": thresholds,
        "hard_reject_by_cluster": dict(cluster_rejected),
        "hard_reject_by_sequence": dict(sequence_rejected),
    }
    return output, summary
