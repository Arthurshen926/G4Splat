"""Small contracts for calibrated-image rectification.

MASt3R's image preprocessing may require a rectified image whose principal
point and focal differ from the source camera.  The image and its intrinsics
are one indivisible contract: changing only one of them silently creates a
geometrically inconsistent SfM scene.
"""

from __future__ import annotations

import numpy as np


def calibrated_camera_to_world_for_output(
    estimated_camera_to_world: np.ndarray,
    calibrated_world_to_camera: np.ndarray,
    *,
    strict: bool,
) -> np.ndarray:
    """Select the camera matrix advertised by a calibrated pointmap export.

    A strict fixed-camera run must not serialize the float32 tensor used by
    the neural optimizer as its camera contract.  Restore the original
    float64 calibration after optimization and retarget dense points against
    that exact matrix.  Non-strict/legacy runs retain their estimated poses.
    """
    estimated = np.asarray(estimated_camera_to_world)
    calibrated = np.asarray(calibrated_world_to_camera)
    if estimated.ndim != 3 or estimated.shape[1:] != (4, 4):
        raise ValueError("estimated_camera_to_world must have shape [N,4,4]")
    if calibrated.shape != estimated.shape:
        raise ValueError(
            "calibrated_world_to_camera must match estimated cameras"
        )
    if not bool(strict):
        return estimated
    exact = np.linalg.inv(
        calibrated.astype(np.float64, copy=False)
    )
    if not np.isfinite(exact).all():
        raise ValueError("Calibrated output cameras are not finite")
    return exact


def rectification_is_identity(
    source_intrinsics: np.ndarray,
    target_intrinsics: np.ndarray,
    *,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    atol: float = 1e-7,
) -> bool:
    """Whether a calibrated rectification leaves image pixels unchanged.

    Shapes use OpenCV's ``(height, width)`` convention.  A no-op transform is
    intentionally detected before calling ``cv2.remap`` so identity-calibrated
    datasets retain byte-identical source imagery in strict ablations.
    """
    return (
        tuple(map(int, source_shape)) == tuple(map(int, target_shape))
        and np.allclose(
            np.asarray(source_intrinsics, dtype=np.float64),
            np.asarray(target_intrinsics, dtype=np.float64),
            rtol=0.0,
            atol=float(atol),
        )
    )


def retarget_pointmap_camera_rays(
    points_world: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: np.ndarray,
    *,
    raster_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Keep camera-z depth while enforcing the advertised pixel/K rays.

    MASt3R can produce dense points using an internal focal estimate even
    when the final pose is restored to a calibrated camera.  A rigid pose
    transform cannot repair that ray mismatch.  This operation preserves the
    network's per-pixel camera-z measurement and re-unprojects it through the
    exact ``fx, fy, cx, cy`` contract.  It is idempotent for an already
    calibrated pointmap.
    """
    points = np.asarray(points_world)
    original_shape = points.shape
    if points.ndim == 2 and points.shape[-1] == 3:
        if raster_shape is None:
            raise ValueError(
                "Flattened pointmaps require raster_shape=(height,width)"
            )
        height, width = map(int, raster_shape)
        if height <= 0 or width <= 0 or height * width != len(points):
            raise ValueError(
                "Flattened pointmap length does not match raster_shape"
            )
        points = points.reshape(height, width, 3)
    elif points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(
            f"Expected pointmap [H,W,3], got {points.shape}"
        )
    camera_to_world = np.asarray(camera_to_world, dtype=np.float64)
    if camera_to_world.shape != (4, 4):
        raise ValueError("camera_to_world must have shape [4,4]")
    values = np.asarray(intrinsics, dtype=np.float64).reshape(-1)
    if len(values) != 4:
        raise ValueError("intrinsics must be [fx,fy,cx,cy]")
    fx, fy, cx, cy = map(float, values)
    if not np.isfinite(values).all() or fx <= 0 or fy <= 0:
        raise ValueError("Pointmap intrinsics must be finite and positive")

    world_to_camera = np.linalg.inv(camera_to_world)
    flat = points.reshape(-1, 3).astype(np.float64, copy=False)
    old_camera = (
        flat @ world_to_camera[:3, :3].T
        + world_to_camera[:3, 3]
    )
    depth = old_camera[:, 2]
    height, width = points.shape[:2]
    rows, columns = np.mgrid[:height, :width]
    exact_camera = np.column_stack(
        [
            (columns.reshape(-1) - cx) * depth / fx,
            (rows.reshape(-1) - cy) * depth / fy,
            depth,
        ]
    )
    retargeted = (
        exact_camera @ camera_to_world[:3, :3].T
        + camera_to_world[:3, 3]
    ).reshape(points.shape)
    return retargeted.reshape(original_shape).astype(
        points.dtype, copy=False
    )
