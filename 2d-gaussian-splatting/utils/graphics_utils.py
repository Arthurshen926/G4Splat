#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(
    znear,
    zfar,
    fovX,
    fovY,
    *,
    fx=None,
    fy=None,
    cx=None,
    cy=None,
    image_width=None,
    image_height=None,
):
    """Return a row-vector projection matrix with optional exact pinhole K.

    The historical FoV-only path remains centered and bit-compatible.  When
    all K fields are supplied, the x/y NDC coordinates are
    ``2 * (f * x / z + c) / size - 1``, preserving COLMAP's principal point
    through rasterisation instead of pretending every camera is centered.
    """
    use_intrinsics = all(value is not None for value in (fx, fy, cx, cy, image_width, image_height))
    if use_intrinsics:
        if float(fx) <= 0.0 or float(fy) <= 0.0 or float(image_width) <= 0.0 or float(image_height) <= 0.0:
            raise ValueError("Pinhole intrinsics and image dimensions must be positive")
        scale_x = 2.0 * float(fx) / float(image_width)
        scale_y = 2.0 * float(fy) / float(image_height)
        offset_x = 2.0 * float(cx) / float(image_width) - 1.0
        offset_y = 2.0 * float(cy) / float(image_height) - 1.0
    else:
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right
        scale_x = 2.0 * znear / (right - left)
        scale_y = 2.0 * znear / (top - bottom)
        offset_x = (right + left) / (right - left)
        offset_y = (top + bottom) / (top - bottom)

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = scale_x
    P[1, 1] = scale_y
    P[0, 2] = offset_x
    P[1, 2] = offset_y
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
