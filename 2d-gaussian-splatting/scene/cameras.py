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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov, fov2focal

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 fx=None, fy=None, cx=None, cy=None, zfar=None,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.image_name = image_name
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.focal_x = float(fx) if fx is not None else float(fov2focal(FoVx, self.image_width))
        self.focal_y = float(fy) if fy is not None else float(fov2focal(FoVy, self.image_height))
        self.cx = float(cx) if cx is not None else self.image_width / 2.0
        self.cy = float(cy) if cy is not None else self.image_height / 2.0
        self.FoVx = focal2fov(self.focal_x, self.image_width)
        self.FoVy = focal2fov(self.focal_y, self.image_height)

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None
        
        # Per-camera sparse support provides an adaptive zfar in outdoor
        # scenes.  Keep the historical fallback for legacy inputs that have
        # no calibrated depth-range evidence.
        self.zfar = float(zfar) if zfar is not None else 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy,
            fx=self.focal_x, fy=self.focal_y, cx=self.cx, cy=self.cy,
            image_width=self.image_width, image_height=self.image_height,
        ).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        

    def set_depth_range(self, *, znear=None, zfar=None):
        """Update clipping planes without losing the exact calibrated K."""
        if znear is not None:
            self.znear = float(znear)
        if zfar is not None:
            self.zfar = float(zfar)
        if self.zfar <= self.znear:
            raise ValueError("Camera zfar must be larger than znear")
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy,
            fx=self.focal_x, fy=self.focal_y, cx=self.cx, cy=self.cy,
            image_width=self.image_width, image_height=self.image_height,
        ).transpose(0, 1).to(self.data_device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
