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
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
from utils.intrinsics_utils import scale_calibrated_intrinsics

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            # if orig_w > 1600:
            #     global WARNED
            #     if not WARNED:
            #         print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
            #             "If this is not desired, please explicitly specify '--resolution/-r' as 1")
            #         WARNED = True
            #     global_down = orig_w / 1600
            # else:
            #     global_down = 1

            global_down = 1
            # print("[ INFO ] Not rescaling images")
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if len(cam_info.image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb

    # Intrinsics must be scaled with the image.  Reusing an FoV alone loses an
    # off-centre principal point and makes Chart/plane/renderer rays disagree
    # after a requested image resize.
    # The loaded RGB can be a pre-materialized target raster whose dimensions
    # differ from cameras.bin.  Explicit COLMAP K is defined in the latter,
    # so scale it from calibration dimensions rather than from the RGB raster.
    fx, fy, cx, cy = scale_calibrated_intrinsics(cam_info, resolution)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=focal2fov(fx, resolution[0]), FoVy=focal2fov(fy, resolution[1]),
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  fx=fx, fy=fy, cx=cx, cy=cy, zfar=getattr(cam_info, "zfar", None))

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    width = getattr(camera, "image_width", None)
    height = getattr(camera, "image_height", None)
    if width is None:
        width = camera.width
    if height is None:
        height = camera.height
    focal_x = getattr(camera, "focal_x", None)
    focal_y = getattr(camera, "focal_y", None)
    if focal_x is None:
        focal_x = fov2focal(camera.FovX, width)
    if focal_y is None:
        focal_y = fov2focal(camera.FovY, height)
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : width,
        'height' : height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : float(focal_y),
        'fx' : float(focal_x),
        'cx' : float(getattr(camera, "cx", width / 2.0)),
        'cy' : float(getattr(camera, "cy", height / 2.0)),
        'zfar' : float(getattr(camera, "zfar", 100.0)),
    }
    return camera_entry
