from __future__ import annotations

import numpy as np


VIEW_AUDIT_POLICY_VERSION = "cambridge-view-audit-v2-pose-integrity"
COLMAP_POSE_POLICY_VERSION = "normalize-colmap-qvec-v1"


def normalize_qvec(qvec: np.ndarray) -> np.ndarray:
    """Return the canonical unit quaternion required by COLMAP camera math."""
    quaternion = np.asarray(qvec, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"Expected qvec with shape (4,), got {quaternion.shape}")
    if not np.isfinite(quaternion).all():
        raise ValueError("Quaternion contains non-finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("Quaternion norm is zero")
    return quaternion / norm


def qvec_to_rotmat(qvec: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    quaternion = normalize_qvec(qvec) if normalize else np.asarray(qvec, dtype=np.float64)
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def pose_integrity_metrics(qvec: np.ndarray) -> dict[str, float | bool]:
    """Measure whether raw COLMAP quaternion data is directly renderable."""
    quaternion = np.asarray(qvec, dtype=np.float64)
    finite = bool(quaternion.shape == (4,) and np.isfinite(quaternion).all())
    if not finite:
        return {
            "pose_finite": False,
            "quaternion_norm": float("nan"),
            "quaternion_norm_error": float("inf"),
            "rotation_orthogonality_error": float("inf"),
            "rotation_determinant_error": float("inf"),
            "pose_normalization_required": False,
            "pose_recoverable": False,
        }

    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        return {
            "pose_finite": True,
            "quaternion_norm": norm,
            "quaternion_norm_error": 1.0,
            "rotation_orthogonality_error": float("inf"),
            "rotation_determinant_error": float("inf"),
            "pose_normalization_required": False,
            "pose_recoverable": False,
        }

    raw_rotation = qvec_to_rotmat(quaternion, normalize=False)
    orthogonality_error = float(
        np.max(np.abs(raw_rotation.T @ raw_rotation - np.eye(3, dtype=np.float64)))
    )
    determinant_error = float(abs(np.linalg.det(raw_rotation) - 1.0))
    norm_error = float(abs(norm - 1.0))
    return {
        "pose_finite": True,
        "quaternion_norm": norm,
        "quaternion_norm_error": norm_error,
        "rotation_orthogonality_error": orthogonality_error,
        "rotation_determinant_error": determinant_error,
        "pose_normalization_required": norm_error > 1e-4,
        "pose_recoverable": True,
    }
