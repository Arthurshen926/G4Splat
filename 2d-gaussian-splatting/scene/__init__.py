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

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # Persist the exact calibrated camera contract consumed by 2DGS.  This
        # catches a principal-point or resize mismatch before it becomes an
        # opaque reconstruction failure, and records the adaptive far planes
        # selected from sparse scene support.
        primary_cameras = self.train_cameras.get(1.0, [])
        principal_point_offsets = np.asarray(
            [
                [
                    float(camera.cx - camera.image_width / 2.0),
                    float(camera.cy - camera.image_height / 2.0),
                ]
                for camera in primary_cameras
            ],
            dtype=np.float64,
        )
        centered_projection_error = (
            np.linalg.norm(principal_point_offsets, axis=1)
            if len(principal_point_offsets)
            else np.empty((0,), dtype=np.float64)
        )
        zfar_values = np.asarray(
            [float(camera.zfar) for camera in primary_cameras], dtype=np.float64
        )
        camera_z_statistics = []
        sparse_points = getattr(scene_info.point_cloud, "points", None)
        sparse_point_count = 0
        sparse_audit_point_count = 0
        if sparse_points is not None:
            sparse_points = np.asarray(sparse_points, dtype=np.float64)
            if sparse_points.ndim == 2 and sparse_points.shape[1] == 3:
                sparse_point_count = int(len(sparse_points))
                if len(sparse_points) > 50_000:
                    audit_indices = np.linspace(
                        0, len(sparse_points) - 1, num=50_000, dtype=np.int64
                    )
                    sparse_points = sparse_points[audit_indices]
                sparse_audit_point_count = int(len(sparse_points))
                for camera in primary_cameras:
                    # The renderer and COLMAP loader use the same row-vector
                    # convention: X_cam = X_world @ R + T.  This audit is
                    # intentionally computed in that contract, rather than
                    # through a transposed CUDA matrix.
                    depth = (
                        sparse_points @ np.asarray(camera.R, dtype=np.float64)
                        + np.asarray(camera.T, dtype=np.float64)
                    )[:, 2]
                    positive_depth = depth[np.isfinite(depth) & (depth > 1e-4)]
                    if positive_depth.size:
                        camera_z_statistics.append(
                            {
                                "image_name": camera.image_name,
                                "positive_point_count": int(positive_depth.size),
                                "p99": float(np.quantile(positive_depth, 0.99)),
                                "p995": float(np.quantile(positive_depth, 0.995)),
                                "p999": float(np.quantile(positive_depth, 0.999)),
                                "max": float(np.max(positive_depth)),
                            }
                        )
        def _distribution(values):
            if not len(values):
                return {"min": None, "median": None, "p95": None, "max": None}
            return {
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
                "max": float(np.max(values)),
            }

        camera_contract = {
            "schema_version": "calibrated-camera-contract-v2",
            "projection": "off_axis_colmap_pinhole",
            "camera_count": len(primary_cameras),
            "audit": {
                "principal_point_offset_pixels": {
                    "absolute_x": _distribution(np.abs(principal_point_offsets[:, 0]))
                    if len(principal_point_offsets) else _distribution([]),
                    "absolute_y": _distribution(np.abs(principal_point_offsets[:, 1]))
                    if len(principal_point_offsets) else _distribution([]),
                    "euclidean": _distribution(centered_projection_error),
                },
                # For a fixed pose and 3-D static point, replacing only
                # (cx, cy) by the centered approximation shifts its projected
                # pixel by exactly this norm; it is therefore a direct
                # reprojection error, not a proxy based on FoV.
                "static_point_reprojection_error_pixels": {
                    "exact_k": _distribution(np.zeros_like(centered_projection_error)),
                    "centered_k": _distribution(centered_projection_error),
                    "reference": "same fixed pose and pinhole focal lengths",
                },
                "adaptive_zfar": {
                    "sparse_point_count": sparse_point_count,
                    "audit_point_sample_count": sparse_audit_point_count,
                    "renderer_zfar": _distribution(zfar_values),
                    "camera_positive_z": camera_z_statistics,
                },
            },
            "cameras": [
                {
                    "image_name": camera.image_name,
                    "width": int(camera.image_width),
                    "height": int(camera.image_height),
                    "fx": float(camera.focal_x),
                    "fy": float(camera.focal_y),
                    "cx": float(camera.cx),
                    "cy": float(camera.cy),
                    "znear": float(camera.znear),
                    "zfar": float(camera.zfar),
                    "principal_point_offset_pixels": [
                        float(camera.cx - camera.image_width / 2.0),
                        float(camera.cy - camera.image_height / 2.0),
                    ],
                }
                for camera in primary_cameras
            ],
        }
        with open(os.path.join(self.model_path, "camera_intrinsics_contract.json"), "w") as handle:
            json.dump(camera_contract, handle, indent=2)
        
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
