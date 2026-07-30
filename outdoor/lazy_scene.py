"""Memory-bounded calibrated camera loading for unified outdoor training.

The upstream 2DGS ``Scene`` eagerly decodes every RGB image.  That is
convenient for small benchmarks, but a Cambridge sequence with more than one
thousand full-resolution frames can consume over 100 GB before iteration one.
This module keeps the calibrated camera tensors resident and decodes only the
few RGB images that are active in the current optimization step.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from scene.dataset_readers import (
    readColmapSceneInfo,
    sceneLoadTypeCallbacks,
)
from utils.general_utils import PILtoTorch
from utils.intrinsics_utils import scale_calibrated_intrinsics
from utils.graphics_utils import (
    focal2fov,
    getProjectionMatrix,
    getWorld2View2,
)


def rgb_source_contract(args) -> dict:
    """Describe the immutable RGB target raster used by training/evaluation.

    Camera calibration and render resolution do not uniquely identify the
    optimization target: the historical Cambridge control materializes a
    shared 640x360 torch-bilinear raster, whereas resizing the 1920x1080
    source at runtime with Pillow produces different edge pixels.  Persist
    both the image inventory and the producer manifest so a later evaluator
    cannot silently compare against a different target raster.
    """
    image_root = Path(args.images)
    if not image_root.is_absolute():
        image_root = Path(args.source_path) / image_root
    image_root = image_root.expanduser().resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"RGB image root does not exist: {image_root}")
    image_paths = sorted(
        (
            path
            for path in image_root.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ),
        key=lambda path: path.name,
    )
    names = [path.name for path in image_paths]
    name_set_sha256 = hashlib.sha256(
        ("\n".join(names) + "\n").encode("utf-8")
    ).hexdigest()
    # A name-set digest does not detect an RGB raster overwritten under the
    # correct filename (or two named rasters accidentally swapped).  Bind
    # every camera-facing identity to its exact encoded bytes.  The digest is
    # deliberately over ``name -> file SHA256`` rather than file iteration
    # order, so it is stable across filesystems while still detecting a
    # content/identity permutation.
    content_mapping = hashlib.sha256()
    content_bytes = 0
    for path in image_paths:
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                file_digest.update(block)
                content_bytes += len(block)
        content_mapping.update(path.name.encode("utf-8"))
        content_mapping.update(b"\0")
        content_mapping.update(file_digest.hexdigest().encode("ascii"))
        content_mapping.update(b"\n")
    producer_manifest = image_root.parent / "input_manifest.json"
    producer = {}
    producer_manifest_sha256 = None
    if producer_manifest.is_file():
        raw = producer_manifest.read_bytes()
        producer_manifest_sha256 = hashlib.sha256(raw).hexdigest()
        try:
            producer = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            producer = {}
    return {
        "image_root": str(image_root),
        "image_count": len(names),
        "name_set_sha256": name_set_sha256,
        "content_mapping_sha256": content_mapping.hexdigest(),
        "content_bytes": int(content_bytes),
        "producer_manifest": (
            str(producer_manifest)
            if producer_manifest.is_file()
            else None
        ),
        "producer_manifest_sha256": producer_manifest_sha256,
        "producer_contract": producer.get("contract"),
        "target_storage": producer.get("rgb_target_storage"),
        "canonical_image_size_wh": producer.get(
            "canonical_image_size_wh"
        ),
    }


class _ImageLRU:
    def __init__(self, maximum: int):
        if maximum < 1:
            raise ValueError("Image cache size must be positive")
        self.maximum = int(maximum)
        self._entries: OrderedDict[int, LazyCamera] = OrderedDict()

    def touch(self, camera: "LazyCamera") -> None:
        key = id(camera)
        self._entries.pop(key, None)
        self._entries[key] = camera
        while len(self._entries) > self.maximum:
            _, victim = self._entries.popitem(last=False)
            victim._drop_cached_image()

    def clear(self) -> None:
        for camera in tuple(self._entries.values()):
            camera._drop_cached_image()
        self._entries.clear()

    @property
    def cached_count(self) -> int:
        return len(self._entries)


def _scaled_resolution(args, width: int, height: int, scale: float):
    if args.resolution in (1, 2, 4, 8):
        return (
            round(width / (scale * args.resolution)),
            round(height / (scale * args.resolution)),
        )
    global_down = (
        1.0
        if args.resolution == -1
        else float(width) / float(args.resolution)
    )
    total_scale = float(global_down) * float(scale)
    return int(width / total_scale), int(height / total_scale)


class LazyCamera:
    """Renderer-compatible calibrated camera with an on-demand RGB tensor."""

    def __init__(
        self,
        cam_info,
        uid: int,
        args,
        cache: _ImageLRU,
        resolution_scale: float = 1.0,
    ):
        # COLMAP intrinsics are expressed in the calibration raster, while a
        # materialized training target may deliberately use another raster
        # (Cambridge's shared target is 640x360 although cameras.bin remains
        # 1920x1080).  Resolution arguments operate on the RGB raster, but K
        # must be scaled from the calibration raster.  Conflating these two
        # coordinate systems made ``--resolution 1`` decode a 640-wide target
        # at 1920 pixels and, in the eager Scene path, could leave fx three
        # times too large.
        calibration_width = int(cam_info.width)
        calibration_height = int(cam_info.height)
        with Image.open(cam_info.image_path) as source_image:
            source_width, source_height = source_image.size
        source_width = int(source_width)
        source_height = int(source_height)
        if min(
            calibration_width,
            calibration_height,
            source_width,
            source_height,
        ) <= 0:
            raise RuntimeError(
                "Camera calibration and RGB raster dimensions must be positive"
            )
        width, height = _scaled_resolution(
            args, source_width, source_height, resolution_scale
        )
        focal_x, focal_y, principal_x, principal_y = (
            scale_calibrated_intrinsics(cam_info, (width, height))
        )

        self.uid = int(uid)
        self.colmap_id = cam_info.uid
        self.R = np.asarray(cam_info.R)
        self.T = np.asarray(cam_info.T)
        self.image_name = str(cam_info.image_name)
        self.image_path = str(cam_info.image_path)
        self.calibration_width = calibration_width
        self.calibration_height = calibration_height
        self.source_raster_width = source_width
        self.source_raster_height = source_height
        self.image_width = int(width)
        self.image_height = int(height)
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.cx = principal_x
        self.cy = principal_y
        self.FoVx = focal2fov(self.focal_x, self.image_width)
        self.FoVy = focal2fov(self.focal_y, self.image_height)
        self.znear = 0.01
        self.zfar = (
            float(cam_info.zfar)
            if getattr(cam_info, "zfar", None) is not None
            else 100.0
        )
        self.trans = np.zeros(3)
        self.scale = 1.0
        self.data_device = torch.device("cpu")
        self._cache = cache
        self._original_image: torch.Tensor | None = None
        self._prefetch_future: Future | None = None
        self.gt_alpha_mask: torch.Tensor | None = None

        self.world_view_transform = torch.tensor(
            getWorld2View2(self.R, self.T, self.trans, self.scale)
        ).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.FoVx,
            fovY=self.FoVy,
            fx=self.focal_x,
            fy=self.focal_y,
            cx=self.cx,
            cy=self.cy,
            image_width=self.image_width,
            image_height=self.image_height,
        ).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def _decode_image(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        with Image.open(self.image_path) as image:
            if "A" in image.getbands():
                alpha = PILtoTorch(
                    image.getchannel("A"),
                    (self.image_width, self.image_height),
                )
                rgb = PILtoTorch(
                    image.convert("RGB"),
                    (self.image_width, self.image_height),
                )
                alpha = alpha.clamp(0, 1).contiguous()
            else:
                rgb = PILtoTorch(
                    image.convert("RGB"),
                    (self.image_width, self.image_height),
                )
                alpha = None
        rgb = rgb.clamp(0, 1).contiguous()
        # Pinned tensors let the main thread enqueue the H2D copy without
        # waiting for it after the background decoder has completed.
        if torch.cuda.is_available():
            rgb = rgb.pin_memory()
            if alpha is not None:
                alpha = alpha.pin_memory()
        return rgb, alpha

    def prefetch(self, executor: ThreadPoolExecutor | None) -> None:
        if (
            executor is None
            or self._original_image is not None
            or self._prefetch_future is not None
        ):
            return
        self._prefetch_future = executor.submit(self._decode_image)

    def _load_image(self) -> None:
        future = self._prefetch_future
        self._prefetch_future = None
        if future is None:
            rgb, alpha = self._decode_image()
        else:
            rgb, alpha = future.result()
        self._original_image = rgb
        self.gt_alpha_mask = alpha
        self._cache.touch(self)

    @property
    def original_image(self) -> torch.Tensor:
        if self._original_image is None:
            self._load_image()
        else:
            self._cache.touch(self)
        return self._original_image

    def _drop_cached_image(self) -> None:
        self._original_image = None
        self.gt_alpha_mask = None

    def release_image(self) -> None:
        self._cache._entries.pop(id(self), None)
        self._drop_cached_image()

    def cancel_prefetch(self) -> None:
        future = self._prefetch_future
        self._prefetch_future = None
        if future is not None:
            future.cancel()

    @property
    def image_is_cached(self) -> bool:
        return self._original_image is not None


def _camera_json(camera: LazyCamera) -> dict:
    world_to_camera = np.zeros((4, 4))
    world_to_camera[:3, :3] = camera.R.transpose()
    world_to_camera[:3, 3] = camera.T
    world_to_camera[3, 3] = 1.0
    camera_to_world = np.linalg.inv(world_to_camera)
    return {
        "id": camera.uid,
        "img_name": camera.image_name,
        "width": camera.image_width,
        "height": camera.image_height,
        "position": camera_to_world[:3, 3].tolist(),
        "rotation": camera_to_world[:3, :3].tolist(),
        "fy": camera.focal_y,
        "fx": camera.focal_x,
        "cx": camera.cx,
        "cy": camera.cy,
    }


def _camera_geometry_digest(cameras: Iterable[LazyCamera]) -> str:
    """Hash only calibrated image geometry, excluding cache/runtime knobs."""
    payload = [
        {
            "image_name": camera.image_name,
            "colmap_id": int(camera.colmap_id),
            "width": int(camera.image_width),
            "height": int(camera.image_height),
            "fx": float(camera.focal_x),
            "fy": float(camera.focal_y),
            "cx": float(camera.cx),
            "cy": float(camera.cy),
            "R_camera_to_world": np.asarray(
                camera.R, dtype=np.float64
            ).tolist(),
            "T_world_to_camera": np.asarray(
                camera.T, dtype=np.float64
            ).tolist(),
            "znear": float(camera.znear),
            "zfar": float(camera.zfar),
        }
        for camera in cameras
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class LazyScene:
    """Small ``Scene`` replacement used by the unified mainline."""

    def __init__(
        self,
        args,
        gaussians,
        *,
        image_cache_size: int = 4,
        image_prefetch_workers: int = 0,
        resolution_scales: Iterable[float] = (1.0,),
    ):
        self.model_path = str(args.model_path)
        self.gaussians = gaussians
        source_path = Path(args.source_path)
        if (source_path / "sparse").exists():
            scene_info = readColmapSceneInfo(
                args.source_path,
                args.images,
                args.eval,
                load_images=False,
                load_point_cloud=False,
            )
        elif (source_path / "transforms_train.json").exists():
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval
            )
        else:
            raise ValueError(f"Unrecognized scene type: {source_path}")

        self.cameras_extent = float(
            scene_info.nerf_normalization["radius"]
        )
        self._image_cache = _ImageLRU(image_cache_size)
        self._image_executor = (
            ThreadPoolExecutor(
                max_workers=int(image_prefetch_workers),
                thread_name_prefix="g4splat-image",
            )
            if int(image_prefetch_workers) > 0
            else None
        )
        self.train_cameras = {}
        self.test_cameras = {}
        for scale in resolution_scales:
            self.train_cameras[scale] = [
                LazyCamera(info, index, args, self._image_cache, scale)
                for index, info in enumerate(scene_info.train_cameras)
            ]
            self.test_cameras[scale] = [
                LazyCamera(info, index, args, self._image_cache, scale)
                for index, info in enumerate(scene_info.test_cameras)
            ]

        # PIL objects in CameraInfo are no longer needed; close their file
        # handles after extracting paths and calibration.
        for info in (
            list(scene_info.train_cameras)
            + list(scene_info.test_cameras)
        ):
            close = getattr(info.image, "close", None)
            if close is not None:
                close()

        output = Path(self.model_path)
        output.mkdir(parents=True, exist_ok=True)
        input_ply = output / "input.ply"
        if (
            not input_ply.exists()
            and scene_info.point_cloud is not None
            and scene_info.ply_path is not None
        ):
            shutil.copyfile(scene_info.ply_path, input_ply)
        primary = self.train_cameras.get(1.0, [])
        (output / "cameras.json").write_text(
            json.dumps(
                [_camera_json(camera) for camera in primary],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        offsets = np.asarray(
            [
                [
                    camera.cx - camera.image_width / 2.0,
                    camera.cy - camera.image_height / 2.0,
                ]
                for camera in primary
            ],
            dtype=np.float64,
        )
        camera_contract = {
            "schema_version": "calibrated-lazy-camera-contract-v3",
            "projection": "off_axis_colmap_pinhole",
            "camera_container_only": True,
            "colmap_points_or_tracks_opened": False,
            "raster_coordinate_contract": (
                "resolution_applies_to_rgb_raster_intrinsics_scale_from_"
                "calibration_raster"
            ),
            "zfar_policy": "fixed_renderer_fallback_100m",
            "camera_count": len(primary),
            "camera_geometry_sha256": _camera_geometry_digest(primary),
            "image_loading": {
                "mode": "on_demand_lru",
                "maximum_cached_views": image_cache_size,
                "eager_rgb_tensor_count": 0,
            },
            "principal_point_offset_pixels": {
                "median": (
                    float(np.median(np.linalg.norm(offsets, axis=1)))
                    if len(offsets)
                    else None
                ),
                "maximum": (
                    float(np.max(np.linalg.norm(offsets, axis=1)))
                    if len(offsets)
                    else None
                ),
            },
            "cameras": [
                {
                    "image_name": camera.image_name,
                    "calibration_width": camera.calibration_width,
                    "calibration_height": camera.calibration_height,
                    "source_raster_width": camera.source_raster_width,
                    "source_raster_height": camera.source_raster_height,
                    "width": camera.image_width,
                    "height": camera.image_height,
                    "fx": camera.focal_x,
                    "fy": camera.focal_y,
                    "cx": camera.cx,
                    "cy": camera.cy,
                    "znear": camera.znear,
                    "zfar": camera.zfar,
                }
                for camera in primary
            ],
        }
        (output / "camera_intrinsics_contract.json").write_text(
            json.dumps(camera_contract, indent=2) + "\n",
            encoding="utf-8",
        )

    def getTrainCameras(self, scale: float = 1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale: float = 1.0):
        return self.test_cameras[scale]

    def release_images(self) -> None:
        self._image_cache.clear()

    def prefetch_images(self, cameras: Iterable[LazyCamera]) -> None:
        if self._image_executor is None:
            return
        for camera in cameras:
            camera.prefetch(self._image_executor)

    def close(self) -> None:
        self.release_images()
        for camera_set in (
            *self.train_cameras.values(),
            *self.test_cameras.values(),
        ):
            for camera in camera_set:
                camera.cancel_prefetch()
        if self._image_executor is not None:
            self._image_executor.shutdown(wait=True, cancel_futures=True)
            self._image_executor = None

    @property
    def cached_image_count(self) -> int:
        return self._image_cache.cached_count

    def save(self, iteration: int) -> None:
        output = (
            Path(self.model_path)
            / "point_cloud"
            / f"iteration_{iteration}"
            / "point_cloud.ply"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        self.gaussians.save_ply(str(output))
