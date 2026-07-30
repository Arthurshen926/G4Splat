"""Pure helpers for mapping calibrated pinhole K into an RGB raster."""

from __future__ import annotations

from utils.graphics_utils import fov2focal


def scale_calibrated_intrinsics(cam_info, resolution):
    """Scale K from its calibration raster into a requested RGB raster.

    ``cam_info.width/height`` describe the coordinate system in which
    explicit COLMAP ``fx/fy/cx/cy`` are expressed.  They need not equal the
    dimensions of a materialized training RGB target.
    """
    width, height = (int(resolution[0]), int(resolution[1]))
    calibration_w = int(getattr(cam_info, "width", width))
    calibration_h = int(getattr(cam_info, "height", height))
    if min(width, height, calibration_w, calibration_h) <= 0:
        raise ValueError("Camera and calibration dimensions must be positive")
    scale_x = float(width) / float(calibration_w)
    scale_y = float(height) / float(calibration_h)
    source_fx = getattr(cam_info, "fx", None)
    source_fy = getattr(cam_info, "fy", None)
    source_cx = getattr(cam_info, "cx", None)
    source_cy = getattr(cam_info, "cy", None)
    fx = (
        float(source_fx) * scale_x
        if source_fx is not None
        else fov2focal(cam_info.FovX, width)
    )
    fy = (
        float(source_fy) * scale_y
        if source_fy is not None
        else fov2focal(cam_info.FovY, height)
    )
    cx = (
        float(source_cx) * scale_x
        if source_cx is not None
        else float(width) / 2.0
    )
    cy = (
        float(source_cy) * scale_y
        if source_cy is not None
        else float(height) / 2.0
    )
    return fx, fy, cx, cy
