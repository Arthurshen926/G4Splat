"""Contracts for calibrated-camera MASt3R runs.

The sparse-GA camera representation reconstructs camera centres from depth and
intrinsics.  Consequently, disabling the explicit translation optimiser is
not, by itself, evidence that the *effective* cameras stayed calibrated.  This
small, dependency-free audit is deliberately separate from image reprojection
checks: it compares the in-memory poses before any legacy output-side pose
replacement is allowed to hide a mismatch.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
    }


def calibrated_pose_contract(
    estimated_cam2world: np.ndarray,
    calibrated_cam2world: np.ndarray,
    image_names: Sequence[str],
    *,
    max_relative_center_error: float = 1e-5,
    max_rotation_error_deg: float = 1e-3,
) -> dict:
    """Measure whether effective MASt3R cameras still equal calibration.

    ``estimated_cam2world`` must be sampled immediately after sparse global
    alignment and before any output-only camera restoration.  The center error
    is normalized by the median non-zero calibrated inter-camera baseline so
    the contract is meaningful across scene scales.
    """
    estimated = np.asarray(estimated_cam2world, dtype=np.float64)
    calibrated = np.asarray(calibrated_cam2world, dtype=np.float64)
    if estimated.shape != calibrated.shape or estimated.ndim != 3 or estimated.shape[1:] != (4, 4):
        raise ValueError(
            "Expected matching [N,4,4] camera-to-world arrays, got "
            f"{estimated.shape} and {calibrated.shape}"
        )
    if len(image_names) != estimated.shape[0]:
        raise ValueError(
            f"camera/name count mismatch: {estimated.shape[0]} cameras, "
            f"{len(image_names)} image names"
        )
    if not np.isfinite(estimated).all() or not np.isfinite(calibrated).all():
        raise ValueError("Calibrated-pose contract received non-finite camera values")

    estimated_centers = estimated[:, :3, 3]
    calibrated_centers = calibrated[:, :3, 3]
    center_errors = np.linalg.norm(estimated_centers - calibrated_centers, axis=1)

    if calibrated_centers.shape[0] > 1:
        pairwise = np.linalg.norm(
            calibrated_centers[:, None, :] - calibrated_centers[None, :, :],
            axis=-1,
        )
        baselines = pairwise[np.triu_indices(pairwise.shape[0], k=1)]
        baselines = baselines[baselines > np.finfo(np.float64).eps]
        baseline = float(np.median(baselines)) if baselines.size else 1.0
    else:
        baseline = 1.0
    relative_center_errors = center_errors / max(baseline, np.finfo(np.float64).eps)

    relative_rotations = np.einsum(
        "nji,njk->nik",
        calibrated[:, :3, :3],
        estimated[:, :3, :3],
    )
    trace = np.trace(relative_rotations, axis1=1, axis2=2)
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    rotation_errors_deg = np.degrees(np.arccos(cosine))

    center_summary = _summary(center_errors)
    relative_center_summary = _summary(relative_center_errors)
    rotation_summary = _summary(rotation_errors_deg)
    failing_indices = np.flatnonzero(
        (relative_center_errors > max_relative_center_error)
        | (rotation_errors_deg > max_rotation_error_deg)
    )
    return {
        "camera_count": int(estimated.shape[0]),
        "image_names": list(image_names),
        "calibrated_median_baseline": baseline,
        "center_error": center_summary,
        "relative_center_error": relative_center_summary,
        "rotation_error_deg": rotation_summary,
        "thresholds": {
            "max_relative_center_error": float(max_relative_center_error),
            "max_rotation_error_deg": float(max_rotation_error_deg),
        },
        "passed": bool(failing_indices.size == 0),
        "failing_image_names": [str(image_names[index]) for index in failing_indices],
    }
