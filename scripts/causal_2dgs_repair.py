#!/usr/bin/env python3
"""Causal post-reconstruction repair for a frozen 2DGS/MAtCha checkpoint.

The command implements the conservative causal order:

``anomaly rays -> masked-gradient candidates -> 3D/co-view clusters ->
counterfactual removal -> real-image tracks/plane -> constrained local edit ->
validation``.

It intentionally has no unconditional See3D call.  The output contains a
strict See3D request only when a real surface is known while both real RGB and
reliable model appearance are absent.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import sys
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GS_ROOT = REPO_ROOT / "2d-gaussian-splatting"
for root in (REPO_ROOT, GS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from artifact_guided_repair import (  # noqa: E402
    ArtifactROI3D,
    CameraGeometry,
    affine_correct_torch,
    aggregate_attribution,
    build_anomaly_rays,
    build_multiview_feature_tracks,
    build_plane_footprint,
    cluster_attributed_primitives,
    counterfactual_causal_rays,
    gradient_primitive_attribution,
    interpolate_c2w,
    masked_charbonnier,
    project_points,
    recover_planar_surface,
    sample_colmap_track_surface_points,
    sample_matcha_chart_surface_points,
    select_diverse_full_train_anomaly_targets,
    select_full_train_anomaly_targets,
    see3d_gate,
    virtual_reprojection_consistency,
)
from gaussian_renderer import render  # noqa: E402
from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402
from scene import GaussianModel  # noqa: E402
from scene.colmap_loader import qvec2rotmat, read_intrinsics_binary  # noqa: E402
from scene.dataset_readers import CameraInfo  # noqa: E402
from scene.cameras import MiniCam  # noqa: E402
from scene.gaussian_model import get_gaussian_normal  # noqa: E402
from utils.camera_utils import loadCam  # noqa: E402
from utils.graphics_utils import focal2fov, getWorld2View2  # noqa: E402


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(value), indent=2) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str | None:
    """Record immutable frozen-checkpoint provenance without copying it."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_name(value: str) -> str:
    """Map source/staged image paths to Scene camera ``image_name`` values.

    COLMAP cameras in this repository deliberately store ``image_name`` without
    the filename suffix (for example ``seq12__frame00097``), whereas metric
    reports and manually supplied target lists retain ``.png``.  Keeping a
    suffix here silently made every metric-selected target miss the loaded
    camera dictionary after the expensive full mask dictionary had been read.
    """
    text = str(value).strip().replace("\\", "/")
    suffix = Path(text).suffix
    if suffix:
        text = text[: -len(suffix)]
    return text.replace("/", "__")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _camera_geometry(camera: Any) -> CameraGeometry:
    return CameraGeometry(
        name=str(camera.image_name),
        width=int(camera.image_width),
        height=int(camera.image_height),
        w2c=camera.world_view_transform.T.detach().cpu().numpy(),
        fx=float(camera.focal_x),
        fy=float(camera.focal_y),
        cx=float(camera.image_width) / 2.0,
        cy=float(camera.image_height) / 2.0,
    )


def _camera_gt(camera: Any) -> np.ndarray:
    value = camera.original_image.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(value.astype(np.float32), 0.0, 1.0)


def _to_tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value)).to(device=device)


def _save_rgb(path: Path, value: np.ndarray | torch.Tensor) -> None:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().permute(1, 2, 0).numpy()
    image = np.asarray(value, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255.0 + 0.5), mode="RGB").save(path)


def _save_mask(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.asarray(value, dtype=bool)) * 255, mode="L").save(path)


def _save_heatmap(path: Path, score: np.ndarray) -> None:
    value = np.asarray(score, dtype=np.float32)
    color = cv2.applyColorMap(np.uint8(np.clip(value, 0.0, 1.0) * 255), cv2.COLORMAP_TURBO)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB), mode="RGB").save(path)


def _copy_cfg(input_model: Path, output_model: Path) -> None:
    cfg = input_model / "cfg_args"
    if cfg.is_file():
        shutil.copy2(cfg, output_model / "cfg_args")


@dataclass
class TargetContext:
    name: str
    camera: Any
    geometry: CameraGeometry
    gt: np.ndarray
    static_mask: np.ndarray
    anomaly_mask: np.ndarray
    score: np.ndarray
    rgb_error: np.ndarray
    affine: np.ndarray
    baseline_render: np.ndarray
    baseline_alpha: np.ndarray
    baseline_depth: np.ndarray
    baseline_expected_depth: np.ndarray
    baseline_median_depth: np.ndarray
    baseline_distortion: np.ndarray
    components: list[dict[str, Any]]
    attribution: dict[str, Any]


@dataclass(frozen=True)
class _LazyCameraRecord:
    """Pose/intrinsic metadata for a real image not yet decoded into a tensor."""

    list_index: int
    uid: int
    R: np.ndarray
    T: np.ndarray
    fov_x: float
    fov_y: float
    image_path: Path
    image_name: str
    width: int
    height: int
    geometry: CameraGeometry


def _scaled_camera_resolution(width: int, height: int, requested_resolution: int) -> tuple[int, int]:
    """Match ``utils.camera_utils.loadCam`` without decoding an image first."""
    if int(requested_resolution) in {1, 2, 4, 8}:
        return (
            round(int(width) / int(requested_resolution)),
            round(int(height) / int(requested_resolution)),
        )
    if int(requested_resolution) == -1:
        return int(width), int(height)
    scale = float(width) / float(requested_resolution)
    return int(width / scale), int(height / scale)


def _read_colmap_pose_headers(path: Path) -> list[tuple[int, np.ndarray, np.ndarray, int, str]]:
    """Read only image poses from COLMAP binary without materializing tracks.

    The standard loader decodes every 2-D keypoint/track array.  The causal
    front-end needs the all-train camera graph, not those arrays; skipping them
    keeps the 1,487-view diagnostic lightweight and preserves the original
    track file for the later, independently audited surface-evidence branch.
    """
    rows: list[tuple[int, np.ndarray, np.ndarray, int, str]] = []
    with path.open("rb") as handle:
        count_bytes = handle.read(8)
        if len(count_bytes) != 8:
            raise ValueError(f"Invalid COLMAP images.bin header: {path}")
        image_count = struct.unpack("<Q", count_bytes)[0]
        for _ in range(int(image_count)):
            payload = handle.read(64)
            if len(payload) != 64:
                raise ValueError(f"Truncated COLMAP image pose entry: {path}")
            values = struct.unpack("<idddddddi", payload)
            image_id = int(values[0])
            qvec = np.asarray(values[1:5], dtype=np.float64)
            tvec = np.asarray(values[5:8], dtype=np.float64)
            camera_id = int(values[8])
            name_bytes = bytearray()
            while True:
                character = handle.read(1)
                if not character:
                    raise ValueError(f"Truncated COLMAP image name: {path}")
                if character == b"\x00":
                    break
                name_bytes.extend(character)
            point_count_bytes = handle.read(8)
            if len(point_count_bytes) != 8:
                raise ValueError(f"Truncated COLMAP point count: {path}")
            point_count = struct.unpack("<Q", point_count_bytes)[0]
            handle.seek(int(point_count) * 24, 1)
            rows.append((image_id, qvec, tvec, camera_id, name_bytes.decode("utf-8")))
    return rows


class _LazyCameraStore(Mapping[str, Any]):
    """All real camera geometry with image tensors decoded only on demand.

    ``Scene`` eagerly loads every image into a ``Camera`` even when causal
    diagnosis needs full-train coverage only for cached RGB prescan and pose
    graph construction.  This mapping preserves the exact Camera API for the
    few targets, controls, and Chart-support views that require rendering,
    while retaining only pose/intrinsic metadata for the other 1,400+ views.
    """

    def __init__(
        self,
        records: dict[str, _LazyCameraRecord],
        *,
        requested_resolution: int,
        data_device: str,
    ) -> None:
        self._records = records
        self._requested_resolution = int(requested_resolution)
        self._data_device = str(data_device)
        self._cache: dict[str, Any] = {}
        self.geometries = {name: record.geometry for name, record in records.items()}

    @classmethod
    def from_colmap(
        cls,
        source_path: Path,
        *,
        requested_resolution: int,
        data_device: str,
    ) -> "_LazyCameraStore":
        sparse = source_path / "sparse" / "0"
        images_bin = sparse / "images.bin"
        cameras_bin = sparse / "cameras.bin"
        images_root = source_path / "images"
        if not images_bin.is_file() or not cameras_bin.is_file() or not images_root.is_dir():
            raise FileNotFoundError(
                "Causal repair needs source sparse/0 images.bin, cameras.bin, and images/: "
                f"{source_path}"
            )
        intrinsics = read_intrinsics_binary(str(cameras_bin))
        records: dict[str, _LazyCameraRecord] = {}
        for list_index, (_, qvec, tvec, camera_id, image_file) in enumerate(
            _read_colmap_pose_headers(images_bin)
        ):
            intrinsic = intrinsics.get(int(camera_id))
            if intrinsic is None:
                raise ValueError(f"COLMAP pose references missing camera_id {camera_id}")
            if intrinsic.model == "SIMPLE_PINHOLE":
                focal_x = focal_y = float(intrinsic.params[0])
            elif intrinsic.model == "PINHOLE":
                focal_x, focal_y = float(intrinsic.params[0]), float(intrinsic.params[1])
            else:
                raise ValueError(
                    f"Unsupported COLMAP camera model {intrinsic.model}; expected PINHOLE or SIMPLE_PINHOLE"
                )
            R = np.transpose(qvec2rotmat(qvec))
            T = np.asarray(tvec, dtype=np.float64)
            image_path = images_root / Path(image_file).name
            if not image_path.is_file():
                raise FileNotFoundError(f"COLMAP image missing from staged source: {image_path}")
            with Image.open(image_path) as image:
                staged_width, staged_height = image.size
            width, height = _scaled_camera_resolution(
                int(staged_width), int(staged_height), int(requested_resolution)
            )
            fov_x = float(focal2fov(focal_x, int(intrinsic.width)))
            fov_y = float(focal2fov(focal_y, int(intrinsic.height)))
            image_name = image_path.stem
            if image_name in records:
                raise ValueError(f"Duplicate staged camera image name: {image_name}")
            geometry = CameraGeometry(
                name=image_name,
                width=int(width),
                height=int(height),
                w2c=getWorld2View2(R, T),
                fx=float(width) / (2.0 * math.tan(fov_x / 2.0)),
                fy=float(height) / (2.0 * math.tan(fov_y / 2.0)),
                cx=float(width) / 2.0,
                cy=float(height) / 2.0,
            )
            records[image_name] = _LazyCameraRecord(
                list_index=int(list_index),
                uid=int(intrinsic.id),
                R=R,
                T=T,
                fov_x=fov_x,
                fov_y=fov_y,
                image_path=image_path,
                image_name=image_name,
                width=int(width),
                height=int(height),
                geometry=geometry,
            )
        if not records:
            raise ValueError(f"No COLMAP train cameras found in {images_bin}")
        return cls(records, requested_resolution=requested_resolution, data_device=data_device)

    def __getitem__(self, name: str) -> Any:
        if name in self._cache:
            return self._cache[name]
        record = self._records[name]
        # ``loadCam`` is the same image/resolution/Camera construction path as
        # normal 2DGS training.  Only its invocation is deferred until a real
        # render or GT comparison actually needs this image.
        with Image.open(record.image_path) as image:
            info = CameraInfo(
                uid=record.uid,
                R=record.R,
                T=record.T,
                FovY=record.fov_y,
                FovX=record.fov_x,
                image=image.copy(),
                image_path=str(record.image_path),
                image_name=record.image_name,
                width=record.width,
                height=record.height,
            )
        args = SimpleNamespace(resolution=self._requested_resolution, data_device=self._data_device)
        camera = loadCam(args, record.list_index, info, 1.0)
        self._cache[name] = camera
        return camera

    def __iter__(self) -> Iterator[str]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)


class _LazyStaticMasks(dict[str, np.ndarray]):
    """Load full-resolution semantic masks only for target/control views used."""

    def __init__(self, lookup: CambridgeMaskLookup, geometries: dict[str, CameraGeometry]) -> None:
        super().__init__()
        self.lookup = lookup
        self.geometries = geometries

    def __missing__(self, name: str) -> np.ndarray:
        geometry = self.geometries[name]
        value = self.lookup.get_mask(
            name, (geometry.height, geometry.width), "cpu"
        ).numpy().astype(bool)
        self[name] = value
        return value


def _read_metric_quality(path: Path | None) -> dict[str, float]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    output = {}
    for item in payload.get("per_view", {}).values():
        name = _canonical_name(item.get("source_image", ""))
        static = item.get("masked", {}).get("static_valid", {}).get("psnr")
        if static is not None:
            output[name] = float(static)
    return output


def _read_rgb_file(path: Path) -> np.ndarray:
    """Load a saved renderer/GT image in the same [0, 1] RGB gauge as Scene."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _full_train_anomaly_prescan(
    *,
    metrics_path: Path | None,
    model_path: Path,
    cameras: dict[str, Any],
    geometries: dict[str, CameraGeometry],
    lookup: CambridgeMaskLookup,
    residual_threshold: float,
    score_threshold: float,
    min_component_pixels: int,
    output: Path,
) -> dict[str, Any]:
    """Audit RGB/structure residuals on every saved real training render.

    This is the first half of stage 1 in the causal design.  It intentionally
    uses the frozen all-train render cache only to decide *where to spend the
    expensive alpha/depth/primitive attribution pass*.  It never creates a
    3-D ROI or authorizes an edit.  Every accepted edit is still guarded by
    the full renderer buffers and an opacity-zero intervention below.

    The evaluation cache must live under ``model_path``.  Otherwise it could
    accidentally describe a different checkpoint and turn a seemingly global
    audit into an invalid causal comparison.
    """
    expected = int(len(cameras))
    disabled = {
        "status": "unavailable",
        "policy": "all_real_train_views_rgb_structure_prescan_before_causal_attribution",
        "expected_full_train_view_count": expected,
        "processed_view_count": 0,
        "selected_target_names": [],
    }
    if metrics_path is None:
        return {**disabled, "reason": "metrics_json_not_provided"}
    metrics_path = metrics_path.expanduser().resolve()
    if not metrics_path.is_file():
        return {**disabled, "reason": "metrics_json_not_found", "metrics_json": str(metrics_path)}
    if not _path_is_within(metrics_path, model_path):
        return {
            **disabled,
            "reason": "metrics_json_is_not_nested_under_frozen_model_path",
            "metrics_json": str(metrics_path),
            "model_path": str(model_path),
        }
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {**disabled, "reason": "metrics_json_unreadable", "detail": str(exc)}
    per_view = payload.get("per_view", {})
    if not isinstance(per_view, dict):
        return {**disabled, "reason": "metrics_json_has_no_per_view_mapping"}
    renders_root = metrics_path.parent / "renders"
    gt_root = metrics_path.parent / "gt"
    if not renders_root.is_dir() or not gt_root.is_dir():
        return {
            **disabled,
            "reason": "frozen_all_train_render_or_gt_cache_missing",
            "renders_root": str(renders_root),
            "gt_root": str(gt_root),
        }

    index: dict[str, tuple[str, dict[str, Any]]] = {}
    duplicate_names: list[str] = []
    unknown_metric_names: list[str] = []
    for rendered_name, row in per_view.items():
        if not isinstance(row, dict):
            continue
        name = _canonical_name(row.get("source_image", ""))
        if name not in cameras:
            unknown_metric_names.append(name or str(rendered_name))
            continue
        if name in index:
            duplicate_names.append(name)
            continue
        index[name] = (str(rendered_name), row)

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    progress_path = output / "full_train_anomaly_prescan.in_progress.json"
    # CambridgeMaskLookup caches resized masks.  A full 1487 x 1080p audit
    # must not retain all of them: the current item is converted to NumPy then
    # the cache is explicitly released before advancing to the next image.
    for ordinal, name in enumerate(sorted(cameras), start=1):
        indexed = index.get(name)
        if indexed is None:
            skipped.append({"name": name, "reason": "missing_metric_per_view_entry"})
            continue
        rendered_name, metric_row = indexed
        render_path = renders_root / rendered_name
        gt_path = gt_root / rendered_name
        if not render_path.is_file() or not gt_path.is_file():
            skipped.append(
                {
                    "name": name,
                    "reason": "missing_saved_render_or_gt",
                    "render_path": str(render_path),
                    "gt_path": str(gt_path),
                }
            )
            continue
        try:
            render_rgb = _read_rgb_file(render_path)
            gt_rgb = _read_rgb_file(gt_path)
            if render_rgb.shape != gt_rgb.shape:
                raise ValueError(f"render/GT shape mismatch: {render_rgb.shape} vs {gt_rgb.shape}")
            geometry = geometries[name]
            static_mask = lookup.get_mask(
                name, (geometry.height, geometry.width), "cpu"
            ).numpy().astype(bool)
            # Do not let the all-view prescan turn a lazy cache into a 3+ GB
            # resident tensor collection.  Target/control masks are cached
            # later by _LazyStaticMasks after the candidate set is fixed.
            lookup._resized_mask_cache.clear()
            height, width = render_rgb.shape[:2]
            if static_mask.shape != (height, width):
                static_mask = cv2.resize(
                    static_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            # There are no alpha/depth buffers in a PNG cache.  Unit alpha and
            # depth explicitly make this a RGB/gradient *prescan* rather than
            # a geometry signal; the later target pass rerenders all maps.
            prescan = build_anomaly_rays(
                render_rgb,
                gt_rgb,
                np.ones((height, width), dtype=np.float32),
                np.ones((height, width), dtype=np.float32),
                static_mask,
                residual_threshold=float(residual_threshold),
                score_threshold=float(score_threshold),
                min_component_pixels=int(min_component_pixels),
            )
            mask = np.asarray(prescan["mask"], dtype=bool)
            valid = np.asarray(static_mask, dtype=bool)
            valid_count = max(int(valid.sum()), 1)
            anomaly_count = int(mask.sum())
            anomaly_fraction = float(anomaly_count / valid_count)
            score = np.asarray(prescan["score"], dtype=np.float32)
            rgb_error = np.asarray(prescan["rgb_error"], dtype=np.float32)
            mean_anomaly_score = float(score[mask].mean()) if anomaly_count else 0.0
            p95_static_error = float(np.quantile(rgb_error[valid], 0.95)) if np.any(valid) else 0.0
            # Area alone over-ranks a global exposure failure; residual alone
            # over-ranks an isolated noisy pixel.  This bounded combination is
            # only a deterministic triage score, never an edit criterion.
            priority = float(
                0.70 * math.sqrt(max(anomaly_fraction, 0.0)) * mean_anomaly_score
                + 0.30 * p95_static_error
            )
            records.append(
                {
                    "name": name,
                    "rendered_name": rendered_name,
                    "source_image": metric_row.get("source_image"),
                    "static_valid_pixel_count": int(valid.sum()),
                    "anomaly_ray_pixel_count": anomaly_count,
                    "anomaly_ray_fraction_of_static_pixels": anomaly_fraction,
                    "component_count": int(prescan["summary"]["component_count"]),
                    "mean_anomaly_score": mean_anomaly_score,
                    "mean_static_rgb_error": float(prescan["summary"]["mean_static_rgb_error"]),
                    "p95_static_rgb_error": p95_static_error,
                    "priority": priority,
                    "metric_static_psnr": metric_row.get("masked", {}).get("static_valid", {}).get("psnr"),
                }
            )
        except Exception as exc:  # Continue auditing the remaining real views.
            lookup._resized_mask_cache.clear()
            skipped.append({"name": name, "reason": "prescan_exception", "detail": str(exc)})
        if ordinal % 64 == 0 or ordinal == expected:
            print(
                f"[full-train-prescan] {ordinal}/{expected}: processed={len(records)}, skipped={len(skipped)}",
                flush=True,
            )
            # Persist an incremental audit trail while the long all-view pass
            # is still running. This file has no selection authority; the final
            # report below is written only after all required images finish.
            _write_json(
                progress_path,
                {
                    "status": "in_progress",
                    "policy": "all_real_train_views_rgb_structure_prescan_before_causal_attribution",
                    "expected_full_train_view_count": expected,
                    "last_ordinal": ordinal,
                    "processed_view_count": len(records),
                    "skipped_view_count": len(skipped),
                    "records": records,
                    "skipped": skipped,
                },
            )

    ranked = select_full_train_anomaly_targets(records, count=len(records))
    status = "completed" if len(records) == expected and not skipped else "completed_with_gaps"
    result = {
        "status": status,
        "policy": "all_real_train_views_rgb_structure_prescan_before_causal_attribution",
        "metrics_json": str(metrics_path),
        "metrics_json_sha256": _sha256_file(metrics_path),
        "renders_root": str(renders_root),
        "gt_root": str(gt_root),
        "expected_full_train_view_count": expected,
        "metric_per_view_count": int(len(per_view)),
        "processed_view_count": int(len(records)),
        "skipped_view_count": int(len(skipped)),
        "unknown_metric_name_count": int(len(unknown_metric_names)),
        "duplicate_metric_name_count": int(len(duplicate_names)),
        "records": records,
        "ranked_target_names": ranked,
        "selected_target_names": [],
        "skipped": skipped,
        "unknown_metric_names": unknown_metric_names[:32],
        "duplicate_metric_names": duplicate_names[:32],
        "geometry_authority": "none_rgb_structure_prescan_only",
        "next_stage": "selected_views_rerendered_with_alpha_depth_distortion_then_primitive_counterfactual",
    }
    _write_json(output / "full_train_anomaly_prescan.json", result)
    if progress_path.exists():
        progress_path.unlink()
    return result


def _reuse_full_train_anomaly_prescan(
    *,
    report_path: Path,
    metrics_path: Path | None,
    model_path: Path,
    cameras: dict[str, Any],
) -> dict[str, Any]:
    """Reuse a completed prescan only when it names this exact frozen cache.

    Re-running the same immutable 1,487-view RGB prescan for each different
    cluster budget wastes time without adding evidence.  Reuse is valid for
    diagnosis-only variants of one frozen checkpoint, but is deliberately
    rejected for an edited model or a changed evaluation cache.
    """
    path = report_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"--full-train-prescan-report does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = int(len(cameras))
    if payload.get("status") != "completed":
        raise ValueError("Reusable full-train prescan is not complete")
    if int(payload.get("expected_full_train_view_count", 0)) != expected:
        raise ValueError("Reusable full-train prescan has the wrong expected camera count")
    if int(payload.get("processed_view_count", 0)) != expected:
        raise ValueError("Reusable full-train prescan did not process every real training view")
    if metrics_path is None or not metrics_path.is_file():
        raise ValueError("Reusable full-train prescan still requires its frozen --metrics-json")
    metrics_path = metrics_path.expanduser().resolve()
    if not _path_is_within(metrics_path, model_path):
        raise ValueError("Reusable prescan metrics must be nested under the frozen model path")
    if payload.get("metrics_json_sha256") != _sha256_file(metrics_path):
        raise ValueError("Reusable full-train prescan belongs to a different metrics cache")
    records = payload.get("records", [])
    names = [str(item.get("name", "")) for item in records if isinstance(item, dict)]
    if len(names) != expected or len(set(names)) != expected or set(names) != set(cameras):
        raise ValueError("Reusable full-train prescan record names do not exactly cover the loaded cameras")
    result = dict(payload)
    result["reused_from"] = str(path)
    result["reuse_policy"] = "same_frozen_checkpoint_metrics_sha256_and_exact_all_train_camera_coverage"
    return result


def _infer_chart_scene(model_path: Path) -> Path | None:
    """Recover the exact MAtCha scene that initialized a frozen 2DGS model.

    A chart point map is only admissible if it shares the model coordinate
    frame.  The saved training ``cfg_args`` is the most reliable provenance
    record for that relation.  Failure to infer it is harmless: the causal
    path simply uses real-image tracks instead of guessing a nearby run.
    """
    cfg = model_path / "cfg_args"
    if not cfg.is_file():
        return None
    try:
        match = re.search(r"source_path='([^']+)'", cfg.read_text(encoding="utf-8"))
    except OSError:
        return None
    if match is None:
        return None
    candidate = Path(match.group(1)).expanduser()
    return candidate if (candidate / "charts_data.npz").is_file() else None


def _infer_sfm_sparse(mask_pickle: Path) -> Path | None:
    """Prefer the original Cambridge sparse tracks when the mask root has them.

    The staged MAtCha dataset intentionally keeps poses but not the global
    ``points3D.bin`` tracks.  Cambridge STDLoc stores that sparse model next
    to the semantic masks, in exactly the same COLMAP coordinate frame.  The
    later projection audit remains mandatory, so a merely similarly named
    directory can never become surface evidence by inference alone.
    """
    candidate = mask_pickle.expanduser().resolve().parent.parent / "sparse" / "0"
    required = ("cameras.bin", "images.bin", "points3D.bin")
    return candidate if all((candidate / name).is_file() for name in required) else None


def _choose_targets(
    requested: Path | None,
    cameras: dict[str, Any],
    metric_quality: dict[str, float],
    count: int,
    prescan_ranked_names: list[str] | None = None,
) -> list[str]:
    if requested is not None:
        values = [
            _canonical_name(line)
            for line in requested.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        missing = sorted(set(values) - set(cameras))
        if missing:
            raise ValueError("Target list has views absent from source path: " + ", ".join(missing[:5]))
        return list(dict.fromkeys(values))
    ranked = []
    if prescan_ranked_names:
        ranked.extend(name for name in prescan_ranked_names if name in cameras)
    ranked.extend(
        name
        for name in sorted(metric_quality, key=lambda name: (metric_quality[name], name))
        if name in cameras and name not in ranked
    )
    if not ranked:
        raise ValueError("Provide --target-image-names-file or a valid --metrics-json")
    return ranked[: int(count)]


def _view_features(geometries: dict[str, CameraGeometry]) -> dict[str, np.ndarray]:
    names = sorted(geometries)
    centers = np.stack([geometries[name].center for name in names])
    extent = max(float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0))), 1e-6)
    normalized = (centers - centers.mean(axis=0, keepdims=True)) / extent
    directions = np.stack([geometries[name].forward for name in names])
    return {
        name: np.concatenate([position, 0.35 * direction])
        for name, position, direction in zip(names, normalized, directions)
    }


def _render_context(
    gaussians: GaussianModel,
    camera: Any,
    pipe: Any,
    background: torch.Tensor,
    static_mask: np.ndarray,
    *,
    affine: np.ndarray | None = None,
) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray]:
    package = render(camera, gaussians, pipe, background)
    rgb = package["render"].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    gt = _camera_gt(camera)
    if affine is None:
        # Control calibration is intentionally independent from anomaly
        # extraction, but is frozen before/after the intervention.
        from artifact_guided_repair.core import robust_affine_color

        _, affine = robust_affine_color(
            rgb,
            gt,
            static_mask & (package["rend_alpha"][0].detach().cpu().numpy() > 0.30),
        )
    return package, rgb, np.asarray(affine, dtype=np.float32)


def _fixed_mae(render_rgb: np.ndarray, gt: np.ndarray, mask: np.ndarray, affine: np.ndarray) -> float:
    corrected = np.clip(
        render_rgb * affine[None, None, :, 0] + affine[None, None, :, 1], 0.0, 1.0
    )
    if not np.any(mask):
        return float("inf")
    return float(np.abs(corrected - gt)[mask].mean())


def _alpha_summary(alpha: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if not np.any(mask):
        return {"mean": 0.0, "low_fraction": 1.0}
    selected = alpha[mask]
    return {"mean": float(selected.mean()), "low_fraction": float((selected < 0.20).mean())}


def _candidate_control_order(
    target_names: list[str],
    features: dict[str, np.ndarray],
    metric_quality: dict[str, float],
) -> list[str]:
    target_set = set(target_names)
    target_features = np.stack([features[name] for name in target_names])
    qualities = np.asarray(list(metric_quality.values()), dtype=np.float64)
    median_quality = float(np.median(qualities)) if len(qualities) else -float("inf")
    target_sequences = {name.split("__", 1)[0] for name in target_names}
    candidates = [name for name in features if name not in target_set]
    return sorted(
        candidates,
        key=lambda name: (
            name.split("__", 1)[0] not in target_sequences,
            metric_quality.get(name, median_quality - 1.0) < median_quality,
            float(np.min(np.linalg.norm(target_features - features[name][None], axis=1))),
            -metric_quality.get(name, median_quality),
            name,
        ),
    )


def _sequence_and_frame_index(name: str) -> tuple[str, int | None]:
    """Extract the real capture sequence and optional frame number.

    Camera poses alone are not enough to choose a causal hold-out: a nearby
    low-quality real frame can be precisely the observation that disproves a
    proposed deletion.  Cambridge names are stable ``seqX__frameNNNNN``
    identifiers, but retain a graceful fallback for datasets without that
    convention.
    """
    sequence, separator, suffix = str(name).partition("__")
    match = re.search(r"(?:^|__)frame(\d+)$", str(name))
    if match is None:
        return (sequence if separator else str(name), None)
    return (sequence, int(match.group(1)))


def _real_camera_envelope_control_order(
    target_names: list[str],
    features: dict[str, np.ndarray],
    metric_quality: dict[str, float],
    *,
    control_count: int,
    temporal_radius: int,
) -> list[dict[str, Any]]:
    """Return an auditable real-camera counterfactual envelope.

    A deletion must be challenged by the adjacent observations that view the
    same surface, even when the frozen model renders those observations badly.
    The former control ordering used metric quality as an early sorting key,
    which silently excluded precisely such contrary evidence.  This order is
    deliberately *not* a clean-image ranking:

    * 75% of the requested panel is a temporal neighbourhood around every
      target sequence, balanced by frame offset and independent of quality;
    * the rest is a pose-near panel from all real cameras, again with quality
      used only as a deterministic tie-breaker.

    Renderer visibility remains an exact per-primitive check in
    ``_counterfactual_cluster``.  The returned rows preserve the reason each
    camera entered the envelope for later audit.
    """
    if not target_names:
        return []
    target_set = set(target_names)
    target_features = np.stack([features[name] for name in target_names])
    target_sequence_frames: dict[str, list[int]] = {}
    for name in target_names:
        sequence, frame = _sequence_and_frame_index(name)
        if frame is not None:
            target_sequence_frames.setdefault(sequence, []).append(frame)
    qualities = np.asarray(list(metric_quality.values()), dtype=np.float64)
    median_quality = float(np.median(qualities)) if len(qualities) else 0.0

    rows: list[dict[str, Any]] = []
    for name in sorted(features):
        if name in target_set:
            continue
        sequence, frame = _sequence_and_frame_index(name)
        reference_frames = target_sequence_frames.get(sequence, [])
        temporal_offset = (
            min(abs(int(frame) - int(reference)) for reference in reference_frames)
            if frame is not None and reference_frames
            else None
        )
        pose_distance = float(np.min(np.linalg.norm(target_features - features[name][None], axis=1)))
        rows.append(
            {
                "name": name,
                "temporal_offset_frames": temporal_offset,
                "minimum_pose_feature_distance": pose_distance,
                # Never use this to filter a real observation.  It is retained
                # solely to make ties reproducible and to expose old bias in
                # the report should it ever reappear.
                "baseline_metric_quality": float(metric_quality.get(name, median_quality)),
            }
        )

    temporal = [
        row
        for row in rows
        if row["temporal_offset_frames"] is not None
        and int(row["temporal_offset_frames"]) <= int(temporal_radius)
    ]
    temporal.sort(
        key=lambda row: (
            int(row["temporal_offset_frames"]),
            float(row["minimum_pose_feature_distance"]),
            -float(row["baseline_metric_quality"]),
            str(row["name"]),
        )
    )
    temporal_budget = max(1, int(np.ceil(0.75 * int(control_count))))
    selected: list[dict[str, Any]] = [
        {**row, "selection_stage": "same_sequence_temporal_envelope"}
        for row in temporal[:temporal_budget]
    ]
    selected_names = {str(row["name"]) for row in selected}

    pose = [row for row in rows if str(row["name"]) not in selected_names]
    pose.sort(
        key=lambda row: (
            float(row["minimum_pose_feature_distance"]),
            -float(row["baseline_metric_quality"]),
            str(row["name"]),
        )
    )
    pose_budget = max(1, int(control_count) - len(selected))
    selected.extend(
        {**row, "selection_stage": "pose_near_real_camera_envelope"}
        for row in pose[:pose_budget]
    )
    selected_names = {str(row["name"]) for row in selected}

    # If some proposed cameras are not visible, continue with the remaining
    # real views rather than treating an undersized control set as harmless.
    selected.extend(
        {**row, "selection_stage": "pose_near_real_camera_envelope_fallback"}
        for row in pose[pose_budget:]
        if str(row["name"]) not in selected_names
    )
    return selected


def _counterfactual_cluster(
    cluster: dict[str, Any],
    contexts: dict[str, TargetContext],
    cameras: dict[str, Any],
    static_masks: dict[str, np.ndarray],
    geometries: dict[str, CameraGeometry],
    features: dict[str, np.ndarray],
    metric_quality: dict[str, float],
    gaussians: GaussianModel,
    pipe: Any,
    background: torch.Tensor,
    *,
    control_count: int,
    control_search: int,
    control_policy: str,
    control_temporal_radius: int,
    min_visible_clean_controls: int,
    min_bad_improvement: float,
    max_clean_increase: float,
    visuals_root: Path,
) -> dict[str, Any]:
    ids = torch.as_tensor(cluster["primitive_ids"], dtype=torch.long, device=gaussians._opacity.device)
    target_names = [name for name in cluster["view_names"] if name in contexts]
    target_names = sorted(target_names, key=lambda name: -float(contexts[name].attribution["per_view_score"].get(name, 0.0)))[:3]
    if not target_names:
        return {**cluster, "accepted": False, "classification": "unattributed", "reason": "no_target_context"}
    if control_policy == "real_camera_envelope_v1":
        candidate_controls = _real_camera_envelope_control_order(
            target_names,
            features,
            metric_quality,
            control_count=int(control_count),
            temporal_radius=int(control_temporal_radius),
        )
    elif control_policy == "legacy_metric_nearby":
        candidate_controls = [
            {
                "name": name,
                "selection_stage": "legacy_metric_nearby",
                "temporal_offset_frames": None,
                "minimum_pose_feature_distance": float(
                    np.min(
                        np.linalg.norm(
                            np.stack([features[target] for target in target_names]) - features[name][None],
                            axis=1,
                        )
                    )
                ),
                "baseline_metric_quality": float(metric_quality.get(name, 0.0)),
            }
            for name in _candidate_control_order(target_names, features, metric_quality)
        ]
    else:
        raise ValueError(f"Unknown counterfactual control policy: {control_policy}")
    controls = []
    for candidate in candidate_controls[: int(control_search)]:
        name = str(candidate["name"])
        # Visibility is checked before treating a view as a clean counterfactual
        # control.  A view that cannot see the candidate cluster cannot certify
        # that its removal is harmless.
        with torch.no_grad():
            package = render(cameras[name], gaussians, pipe, background)
        visible = package["visibility_filter"][ids].detach().float().mean().item()
        if visible <= 0.0:
            continue
        rgb = package["render"].detach().cpu().permute(1, 2, 0).numpy()
        gt = _camera_gt(cameras[name])
        from artifact_guided_repair.core import robust_affine_color

        _, affine = robust_affine_color(
            rgb,
            gt,
            static_masks[name] & (package["rend_alpha"][0].detach().cpu().numpy() > 0.30),
        )
        controls.append(
            {
                "name": name,
                "baseline_rgb": rgb,
                "affine": affine,
                "visible_fraction": visible,
                "selection_stage": candidate["selection_stage"],
                "temporal_offset_frames": candidate["temporal_offset_frames"],
                "minimum_pose_feature_distance": candidate["minimum_pose_feature_distance"],
                "baseline_metric_quality": candidate["baseline_metric_quality"],
            }
        )
        if len(controls) >= int(control_count):
            break

    baseline_targets = []
    for name in target_names:
        context = contexts[name]
        baseline_targets.append(
            {
                "name": name,
                "mae": _fixed_mae(context.baseline_render, context.gt, context.anomaly_mask, context.affine),
                "alpha": _alpha_summary(context.baseline_alpha, context.anomaly_mask),
            }
        )
    baseline_controls = []
    for control in controls:
        name = control["name"]
        baseline_controls.append(
            {
                "name": name,
                "mae": _fixed_mae(control["baseline_rgb"], _camera_gt(cameras[name]), static_masks[name], control["affine"]),
                "visible_fraction": control["visible_fraction"],
            }
        )

    original = gaussians._opacity.detach()[ids].clone()
    with torch.no_grad():
        gaussians._opacity[ids] = -16.0
    after_targets = []
    after_controls = []
    after_images: dict[str, np.ndarray] = {}
    try:
        with torch.no_grad():
            for name in target_names:
                context = contexts[name]
                package = render(context.camera, gaussians, pipe, background)
                rgb = package["render"].detach().cpu().permute(1, 2, 0).numpy()
                alpha = package["rend_alpha"][0].detach().cpu().numpy()
                after_images[name] = rgb
                after_targets.append(
                    {
                        "name": name,
                        "mae": _fixed_mae(rgb, context.gt, context.anomaly_mask, context.affine),
                        "alpha": _alpha_summary(alpha, context.anomaly_mask),
                    }
                )
            for control in controls:
                name = control["name"]
                package = render(cameras[name], gaussians, pipe, background)
                rgb = package["render"].detach().cpu().permute(1, 2, 0).numpy()
                after_images[name] = rgb
                after_controls.append(
                    {
                        "name": name,
                        "mae": _fixed_mae(rgb, _camera_gt(cameras[name]), static_masks[name], control["affine"]),
                    }
                )
    finally:
        with torch.no_grad():
            gaussians._opacity[ids] = original

    baseline_by_name = {item["name"]: item for item in baseline_targets}
    improvements = []
    alpha_holes = []
    peel_explanation_fractions = []
    target_rows = []
    causal_ray_masks: dict[str, np.ndarray] = {}
    causal_surface_ray_masks: dict[str, np.ndarray] = {}
    for after in after_targets:
        before = baseline_by_name[after["name"]]
        improvement = (before["mae"] - after["mae"]) / max(before["mae"], 1e-6)
        improvements.append(improvement)
        alpha_holes.append(after["alpha"]["low_fraction"] - before["alpha"]["low_fraction"])
        context = contexts[after["name"]]
        baseline_corrected = np.clip(
            context.baseline_render * context.affine[None, None, :, 0]
            + context.affine[None, None, :, 1],
            0.0,
            1.0,
        )
        counterfactual_corrected = np.clip(
            after_images[after["name"]] * context.affine[None, None, :, 0]
            + context.affine[None, None, :, 1],
            0.0,
            1.0,
        )
        baseline_error = np.mean(np.abs(baseline_corrected - context.gt), axis=2)
        counterfactual_error = np.mean(np.abs(counterfactual_corrected - context.gt), axis=2)
        significantly_explained = context.anomaly_mask & (
            baseline_error - counterfactual_error >= 0.01
        )
        peel_fraction = float(
            significantly_explained.sum() / max(int(context.anomaly_mask.sum()), 1)
        )
        peel_explanation_fractions.append(peel_fraction)
        # The whole residual component can cover several depth layers or even
        # several physical surfaces.  For geometry recovery keep only rays
        # whose rendered color actually changes under this cluster's opacity
        # intervention.  This remains an image-space anomaly-ray subset, not
        # a 3-D ROI; real chart/track evidence is still required downstream.
        causal_split = counterfactual_causal_rays(
            context.baseline_render,
            after_images[after["name"]],
            context.anomaly_mask,
            minimum_causal_pixels=max(32, int(0.005 * context.anomaly_mask.sum())),
            minimum_surface_pixels=max(32, int(0.001 * context.anomaly_mask.sum())),
        )
        influence = causal_split["influence"]
        causal_rays = causal_split["causal_mask"]
        threshold = causal_split["causal_threshold"]
        fallback_to_full_anomaly = bool(causal_split["causal_fallback_to_full_anomaly"])
        causal_ray_masks[after["name"]] = causal_rays
        # Surface recovery is stricter still: use only the causal core (top
        # 10% influence on anomaly rays) so one cluster cannot turn a broad
        # mixed residual into a spurious facade plane.  If the intervention
        # has no measurable image support, retain the full anomaly and record
        # that geometry remains weak rather than silently inventing a core.
        surface_threshold = causal_split["surface_core_threshold"]
        causal_surface_rays = causal_split["surface_core_mask"]
        surface_fallback_to_causal = bool(causal_split["surface_core_fallback_to_causal"])
        causal_surface_ray_masks[after["name"]] = causal_surface_rays
        target_rows.append(
            {
                **after,
                "baseline_mae": before["mae"],
                "relative_improvement": improvement,
                "peel_explained_bad_ray_fraction": peel_fraction,
                "baseline_alpha": before["alpha"],
                "causal_ray_pixel_count": int(causal_rays.sum()),
                "causal_ray_fraction_of_image": float(causal_rays.mean()),
                "causal_ray_fraction_of_anomaly": float(
                    causal_rays.sum() / max(int(context.anomaly_mask.sum()), 1)
                ),
                "causal_ray_influence_threshold": threshold,
                "causal_ray_fallback_to_full_anomaly": fallback_to_full_anomaly,
                "causal_surface_ray_pixel_count": int(causal_surface_rays.sum()),
                "causal_surface_ray_fraction_of_anomaly": float(
                    causal_surface_rays.sum() / max(int(context.anomaly_mask.sum()), 1)
                ),
                "causal_surface_ray_influence_threshold": surface_threshold,
                "causal_surface_ray_fallback_to_causal_rays": surface_fallback_to_causal,
            }
        )
    baseline_control_by_name = {item["name"]: item for item in baseline_controls}
    control_by_name = {item["name"]: item for item in controls}
    clean_rows = []
    increases = []
    for after in after_controls:
        before = baseline_control_by_name[after["name"]]
        increase = after["mae"] - before["mae"]
        increases.append(increase)
        clean_rows.append(
            {
                **after,
                "baseline_mae": before["mae"],
                "mae_increase": increase,
                "visible_fraction": before["visible_fraction"],
                "selection_stage": control_by_name[after["name"]]["selection_stage"],
                "temporal_offset_frames": control_by_name[after["name"]]["temporal_offset_frames"],
                "minimum_pose_feature_distance": control_by_name[after["name"]]["minimum_pose_feature_distance"],
                "baseline_metric_quality": control_by_name[after["name"]]["baseline_metric_quality"],
                # Persist the frozen calibration used for the original
                # opacity-zero comparison so later geometry counterfactuals
                # cannot refit exposure and hide a clean-view regression.
                "affine": np.asarray(control_by_name[after["name"]]["affine"], dtype=np.float32),
            }
        )
    median_improvement = float(np.median(improvements)) if improvements else -float("inf")
    worst_clean_increase = float(max(increases)) if increases else 0.0
    # A permanent opacity edit needs an observed counterfactual control.  An
    # empty control set is not evidence that the intervention has zero harm.
    has_visible_clean_control = len(controls) >= int(min_visible_clean_controls)
    foreground = (
        has_visible_clean_control
        and median_improvement >= float(min_bad_improvement)
        and worst_clean_increase <= float(max_clean_increase)
    )
    mean_alpha = float(np.mean([item["alpha"]["mean"] for item in after_targets])) if after_targets else 0.0
    hole_increase = float(np.mean(alpha_holes)) if alpha_holes else 0.0
    if not has_visible_clean_control and median_improvement >= float(min_bad_improvement):
        classification = "foreground_unverified_insufficient_real_camera_envelope"
    elif foreground:
        classification = "foreground_plus_hole" if mean_alpha < 0.50 or hole_increase > 0.15 else "wrong_foreground"
    elif float(np.mean([contexts[name].baseline_alpha[contexts[name].anomaly_mask].mean() for name in target_names])) < 0.45:
        classification = "missing_surface"
    elif float(cluster["position_gradient_max"]) > 0:
        classification = "surface_misalignment_or_appearance"
    else:
        classification = "appearance_or_renderer"
    accepted = bool(foreground)
    report = {
        **cluster,
        "primitive_ids": np.asarray(cluster["primitive_ids"], dtype=np.int64),
        "counterfactual_method": "opacity_zero_intervention",
        "targets": target_rows,
        "clean_controls": clean_rows,
        "median_bad_relative_improvement": median_improvement,
        "worst_clean_mae_increase": worst_clean_increase,
        "visible_clean_control_count": int(len(controls)),
        "control_envelope": {
            "policy": control_policy,
            "requested_control_count": int(control_count),
            "searched_candidate_count": int(min(len(candidate_controls), int(control_search))),
            "temporal_radius_frames": int(control_temporal_radius),
            "minimum_visible_control_count": int(min_visible_clean_controls),
            "visible_controls_by_stage": {
                stage: int(sum(1 for item in controls if item["selection_stage"] == stage))
                for stage in sorted({str(item["selection_stage"]) for item in controls})
            },
            "quality_is_not_a_filter": bool(control_policy == "real_camera_envelope_v1"),
        },
        "mean_counterfactual_alpha": mean_alpha,
        "mean_alpha_hole_increase": hole_increase,
        "median_peel_explained_bad_ray_fraction": float(np.median(peel_explanation_fractions)) if peel_explanation_fractions else 0.0,
        "acceptance_thresholds": {
            "min_bad_relative_improvement": min_bad_improvement,
            "max_clean_mae_increase": max_clean_increase,
            "min_visible_clean_controls": int(min_visible_clean_controls),
        },
        "classification": classification,
        "accepted": accepted,
        "culprit_confidence": "counterfactual_verified" if accepted else "not_verified",
    }
    cluster_dir = visuals_root / f"cluster_{int(cluster['cluster_id']):03d}"
    for name in target_names:
        context = contexts[name]
        _save_rgb(cluster_dir / f"{_safe_name(name)}.baseline.png", context.baseline_render)
        _save_rgb(cluster_dir / f"{_safe_name(name)}.counterfactual.png", after_images[name])
        _save_rgb(cluster_dir / f"{_safe_name(name)}.gt.png", context.gt)
        _save_mask(cluster_dir / f"{_safe_name(name)}.anomaly.png", context.anomaly_mask)
        causal_mask_path = cluster_dir / f"{_safe_name(name)}.causal_rays.png"
        _save_mask(causal_mask_path, causal_ray_masks[name])
        surface_mask_path = cluster_dir / f"{_safe_name(name)}.causal_surface_rays.png"
        _save_mask(surface_mask_path, causal_surface_ray_masks[name])
        target_row = next(item for item in target_rows if item["name"] == name)
        target_row["causal_ray_mask_path"] = str(causal_mask_path)
        target_row["causal_surface_ray_mask_path"] = str(surface_mask_path)
    for control in controls:
        name = control["name"]
        _save_rgb(cluster_dir / f"{_safe_name(name)}.clean.baseline.png", control["baseline_rgb"])
        _save_rgb(cluster_dir / f"{_safe_name(name)}.clean.counterfactual.png", after_images[name])
    _write_json(cluster_dir / "counterfactual.json", report)
    return report


def _recover_surface_evidence(
    counterfactual: dict[str, Any],
    contexts: dict[str, TargetContext],
    cameras: dict[str, Any],
    static_masks: dict[str, np.ndarray],
    geometries: dict[str, CameraGeometry],
    visuals_root: Path,
    *,
    chart_scene: Path | None,
    sfm_sparse_path: Path | None,
) -> dict[str, Any]:
    """Recover a surface from aligned Charts first, then real-image tracks.

    The MAtCha branch is not a shortcut from a 2-D mask to a 3-D box: it
    first audits the Chart/world coordinate transform against real cameras,
    then requires a target ray/depth layer to be backed by multiple Charts.
    Feature tracks remain an independent cross-check when available and the
    sole fallback when the chart evidence is absent or fails audit.
    """
    target_names = [item["name"] for item in counterfactual.get("targets", []) if item["name"] in contexts]
    if not target_names:
        return {"surface_known": False, "reason": "no_target_context"}
    target_name = target_names[0]
    context = contexts[target_name]
    target_counterfactual = next(
        (item for item in counterfactual.get("targets", []) if item.get("name") == target_name),
        {},
    )
    surface_rays = context.anomaly_mask
    causal_ray_path = target_counterfactual.get("causal_surface_ray_mask_path") or target_counterfactual.get("causal_ray_mask_path")
    if causal_ray_path:
        path = Path(str(causal_ray_path))
        if path.is_file():
            loaded = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127
            if loaded.shape == context.anomaly_mask.shape and int(loaded.sum()) >= 32:
                surface_rays = loaded & context.anomaly_mask
    # The same culprit cluster can be visible in several bad training views.
    # Their *real* RGB observations remain valid track/color evidence even
    # though their current renders are anomalous, so use them before adding
    # clean controls.  This is still never a copied 2-D mask: each support
    # image receives its own feature matches and camera projection.
    support_names = list(
        dict.fromkeys(
            [name for name in target_names[1:] if name in cameras]
            + [item["name"] for item in counterfactual.get("clean_controls", []) if item["name"] in cameras]
        )
    )
    # Gather the original global SfM tracks as independent evidence.  These
    # observations predate the 2DGS model and have a stable point identity
    # across all 1,487 real cameras.  The selection below still gives a
    # coordinate-audited MAtCha Chart first priority; SfM validates it or
    # provides the real-geometry fallback. SIFT remains a last fallback, not
    # a substitute for an existing MASt3R/SfM track graph.
    sfm_points = np.empty((0, 3), dtype=np.float64)
    sfm_tracks: dict[str, Any] = {
        "method": "not_requested",
        "available": False,
        "accepted": False,
        "reason": "no_original_colmap_sparse_path",
    }
    sfm_plane = None
    sfm_inliers = np.empty(0, dtype=bool)
    sfm_plane_metrics: dict[str, Any] = {"accepted": False, "reason": "no_sfm_tracks"}
    if sfm_sparse_path is not None:
        sfm_points, sfm_tracks = sample_colmap_track_surface_points(
            sfm_sparse_path,
            context.geometry,
            surface_rays,
            geometries,
            source_static_masks=static_masks,
        )
        if sfm_tracks.get("accepted"):
            sfm_plane, sfm_inliers, sfm_plane_metrics = recover_planar_surface(sfm_points)
        sfm_tracks["plane_metrics"] = sfm_plane_metrics

    supports = [
        (name, _camera_gt(cameras[name]), static_masks[name], geometries[name])
        for name in support_names
    ]
    sift_points, sift_tracks = build_multiview_feature_tracks(
        context.gt,
        surface_rays,
        context.geometry,
        supports,
        min_support_observations=2,
    )
    sift_plane, sift_inliers, sift_plane_metrics = recover_planar_surface(sift_points)
    if sfm_plane is not None:
        points, tracks = sfm_points, {
            "primary_source": "original_colmap_multiview_tracks",
            "primary": sfm_tracks,
            "sift_complement": sift_tracks,
            "sift_complement_plane_metrics": sift_plane_metrics,
        }
        track_plane, track_inliers, track_plane_metrics = sfm_plane, sfm_inliers, sfm_plane_metrics
        track_source = "original_colmap_multiview_tracks"
    else:
        points, tracks = sift_points, {
            "primary_source": "target_anchored_sift_fallback",
            "original_colmap": sfm_tracks,
            "sift_fallback": sift_tracks,
            "sift_fallback_plane_metrics": sift_plane_metrics,
        }
        track_plane, track_inliers, track_plane_metrics = sift_plane, sift_inliers, sift_plane_metrics
        track_source = "target_anchored_sift_fallback"

    chart_points = np.empty((0, 3), dtype=np.float64)
    chart_report: dict[str, Any] = {
        "method": "not_requested",
        "available": False,
        "accepted": False,
        "reason": "no_coordinate-matched_chart_scene",
    }
    chart_plane = None
    chart_inliers = np.empty(0, dtype=bool)
    chart_plane_metrics: dict[str, Any] = {"accepted": False, "reason": "no_chart_points"}
    if chart_scene is not None:
        chart_points, chart_report = sample_matcha_chart_surface_points(
            chart_scene,
            context.geometry,
            surface_rays,
            geometries,
            source_static_masks=static_masks,
        )
        if chart_report.get("accepted"):
            chart_plane, chart_inliers, chart_plane_metrics = recover_planar_surface(
                chart_points,
                min_points=16,
                min_inlier_fraction=0.65,
                max_normalized_residual=0.025,
            )
        chart_report["plane_metrics"] = chart_plane_metrics

    agreement: dict[str, Any] = {
        "available": bool(chart_plane is not None and track_plane is not None),
        "accepted": None,
    }
    if chart_plane is not None and track_plane is not None:
        chart_normal = np.asarray(chart_plane[:3], dtype=np.float64)
        chart_normal /= np.linalg.norm(chart_normal).clip(min=1e-12)
        track_normal = np.asarray(track_plane[:3], dtype=np.float64)
        track_normal /= np.linalg.norm(track_normal).clip(min=1e-12)
        angle = float(np.rad2deg(np.arccos(np.clip(abs(float(chart_normal @ track_normal)), 0.0, 1.0))))
        track_center = np.median(points[track_inliers], axis=0)
        chart_residual_at_track_center = abs(float(track_center @ chart_normal + chart_plane[3]))
        extent = max(
            float(chart_plane_metrics.get("extent", 0.0)),
            float(track_plane_metrics.get("extent", 0.0)),
            1e-6,
        )
        normalized_offset = chart_residual_at_track_center / extent
        agreement = {
            "available": True,
            "normal_angle_degrees": angle,
            "normalized_center_offset": normalized_offset,
            "accepted": bool(angle <= 20.0 and normalized_offset <= 0.10),
            "thresholds": {"max_normal_angle_degrees": 20.0, "max_normalized_center_offset": 0.10},
        }

    selected_source: str | None = None
    selected_points = np.empty((0, 3), dtype=np.float64)
    selected_inliers = np.empty(0, dtype=bool)
    selected_plane = None
    selected_plane_metrics: dict[str, Any] = {"accepted": False, "reason": "no_real_surface"}
    selection_reason = "no_chart_or_track_plane_passed"
    if chart_plane is not None and track_plane is not None:
        if bool(agreement.get("accepted")):
            # MAtCha's dense aligned Chart is preferred for a compact
            # footprint, while independently triangulated real tracks certify
            # that it has not drifted into a different surface.
            selected_source = "aligned_matcha_chart_validated_by_real_tracks"
            selected_points, selected_inliers = chart_points, chart_inliers
            selected_plane, selected_plane_metrics = chart_plane, chart_plane_metrics
            selection_reason = "chart_and_track_planes_agree"
        else:
            selection_reason = "chart_track_plane_disagreement"
    elif chart_plane is not None:
        selected_source = "aligned_matcha_chart_multiview"
        selected_points, selected_inliers = chart_points, chart_inliers
        selected_plane, selected_plane_metrics = chart_plane, chart_plane_metrics
        selection_reason = "chart_coordinate_audit_and_multiview_support_passed"
    elif track_plane is not None:
        selected_source = track_source
        selected_points, selected_inliers = points, track_inliers
        selected_plane, selected_plane_metrics = track_plane, track_plane_metrics
        selection_reason = "chart_unavailable_or_rejected_real_track_plane_passed"

    footprint_points = np.empty((0, 3), dtype=np.float64)
    footprint_metrics: dict[str, Any] = {"method": "no_accepted_plane"}
    slab_thickness = 0.0
    if selected_plane is not None:
        # A dense chart can contain thousands of supported samples.  Its
        # footprint is geometrically identical after a deterministic spread
        # sample, but avoids turning an ROI audit into an unbounded point
        # rasterization workload.
        footprint_support = selected_points[selected_inliers]
        if len(footprint_support) > 800:
            footprint_support = footprint_support[_farthest_sample_indices(footprint_support, 800)]
        footprint_points, footprint_metrics = build_plane_footprint(
            footprint_support,
            selected_plane,
            np.ones(len(footprint_support), dtype=bool),
        )
        # The slab is tied to independently triangulated residuals and never
        # inferred from the current 2DGS depth.  Chart and track evidence use
        # the same conservative rule; a small floor avoids a numerically zero
        # slab while the upper term keeps it local.
        extent = float(selected_plane_metrics.get("extent", 0.0))
        residual = float(selected_plane_metrics.get("median_absolute_residual", 0.0))
        slab_thickness = max(3.0 * residual, 0.002 * extent, 1e-4)
    evidence: dict[str, Any] = {
        "target_name": target_name,
        "causal_surface_ray_mask_path": causal_ray_path,
        "causal_ray_pixel_count": int(surface_rays.sum()),
        "causal_ray_fraction_of_target_anomaly": float(
            surface_rays.sum() / max(int(context.anomaly_mask.sum()), 1)
        ),
        "support_names": support_names,
        "track_source": track_source,
        "tracks": tracks,
        "original_colmap_tracks": sfm_tracks,
        "original_colmap_plane": sfm_plane,
        "original_colmap_plane_metrics": sfm_plane_metrics,
        "sift_tracks": sift_tracks,
        "sift_plane": sift_plane,
        "sift_plane_metrics": sift_plane_metrics,
        "track_plane": track_plane,
        "track_plane_metrics": track_plane_metrics,
        "chart": chart_report,
        "chart_plane": chart_plane,
        "chart_plane_metrics": chart_plane_metrics,
        "chart_track_agreement": agreement,
        "surface_source": selected_source,
        "selection_reason": selection_reason,
        "plane": selected_plane,
        "plane_metrics": selected_plane_metrics,
        "surface_known": bool(selected_plane is not None),
        "points": selected_points,
        "inlier_mask": selected_inliers,
        "footprint_points": footprint_points,
        "footprint_metrics": footprint_metrics,
        "slab_thickness": slab_thickness,
    }
    if selected_plane is not None:
        roi = ArtifactROI3D(
            roi_id=f"cluster_{int(counterfactual['cluster_id']):03d}",
            points=footprint_points,
            source_view_ids=(),
            classification=str(counterfactual["classification"]),
            plane_world=selected_plane,
            slab_thickness=slab_thickness,
            evidence_view_ids=(),
            chart_ids=tuple(int(value) for value in chart_report.get("returned_source_chart_indices", [])),
            gaussian_ids=tuple(int(value) for value in counterfactual["primitive_ids"]),
            evidence={
                "tracks": tracks,
                "track_source": track_source,
                "original_colmap_tracks": sfm_tracks,
                "original_colmap_plane_metrics": sfm_plane_metrics,
                "sift_tracks": sift_tracks,
                "sift_plane_metrics": sift_plane_metrics,
                "track_plane_metrics": track_plane_metrics,
                "chart": chart_report,
                "chart_plane_metrics": chart_plane_metrics,
                "chart_track_agreement": agreement,
                "surface_source": selected_source,
                "plane_metrics": selected_plane_metrics,
                "footprint_metrics": footprint_metrics,
                "evidence_source": selected_source,
            },
        )
        roi_path = roi.save(visuals_root / "rois" / f"{roi.roi_id}.npz")
        evidence["roi_path"] = str(roi_path)
        evidence["roi"] = roi.summary()
    return evidence


def _farthest_sample_indices(points: np.ndarray, count: int) -> np.ndarray:
    """Deterministically spread a small set of seeds over a support patch."""
    value = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(value) or count <= 0:
        return np.empty(0, dtype=np.int64)
    count = min(int(count), len(value))
    chosen = [int(np.argmin(np.linalg.norm(value - np.median(value, axis=0), axis=1)))]
    nearest_sq = np.sum((value - value[chosen[0]]) ** 2, axis=1)
    while len(chosen) < count:
        candidate = int(np.argmax(nearest_sq))
        if candidate in chosen:
            break
        chosen.append(candidate)
        nearest_sq = np.minimum(nearest_sq, np.sum((value - value[candidate]) ** 2, axis=1))
    return np.asarray(chosen, dtype=np.int64)


def _plane_local_cluster_ids(
    primitive_ids: np.ndarray,
    surface: dict[str, Any],
    xyz: np.ndarray,
    normals: np.ndarray,
    scales: np.ndarray,
    *,
    max_count: int,
) -> np.ndarray:
    """Keep only attributed primitives that actually live on the real patch."""
    plane = surface.get("plane")
    footprint = np.asarray(surface.get("footprint_points", []), dtype=np.float64).reshape(-1, 3)
    if plane is None or not len(footprint):
        return np.empty(0, dtype=np.int64)
    candidates = np.unique(np.asarray(primitive_ids, dtype=np.int64))
    candidates = candidates[(candidates >= 0) & (candidates < len(xyz))]
    if not len(candidates):
        return candidates
    plane_value = np.asarray(plane, dtype=np.float64).reshape(4)
    normal = plane_value[:3] / np.linalg.norm(plane_value[:3]).clip(min=1e-12)
    slab = max(float(surface.get("slab_thickness", 0.0)), 1e-4)
    candidate_xyz = np.asarray(xyz, dtype=np.float64)[candidates]
    candidate_normals = np.asarray(normals, dtype=np.float64)[candidates]
    candidate_scales = np.asarray(scales, dtype=np.float64)[candidates]
    signed_distance = np.abs(candidate_xyz @ normal + float(plane_value[3]))
    scale_radius = np.linalg.norm(candidate_scales, axis=1)
    tree = None
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(footprint)
        footprint_distance = tree.query(candidate_xyz, k=1)[0]
    except Exception:
        footprint_distance = np.linalg.norm(
            candidate_xyz[:, None] - footprint[None], axis=2
        ).min(axis=1)
    footprint_metrics = surface.get("footprint_metrics", {})
    spacing = float(footprint_metrics.get("median_spacing", 0.0))
    if not math.isfinite(spacing) or spacing <= 0:
        spacing = max(float(np.median(scale_radius)), slab)
    local_radius = max(2.5 * spacing, 3.0 * slab, 2e-4)
    normal_length = np.linalg.norm(candidate_normals, axis=1).clip(min=1e-12)
    normal_alignment = np.abs(candidate_normals @ normal) / normal_length
    valid = (
        (signed_distance <= slab + 2.0 * scale_radius)
        & (footprint_distance <= local_radius + 2.0 * scale_radius)
        & (normal_alignment >= 0.20)
    )
    # Do not fabricate a repair set merely because an attributed primitive
    # happens to be nearby: empty is safer than a large unbounded edit.
    output = candidates[valid]
    if len(output) > int(max_count):
        order = np.lexsort((footprint_distance[valid], signed_distance[valid]))
        output = output[order[: int(max_count)]]
    return np.asarray(output, dtype=np.int64)


def _plane_projection_counterfactual(
    counterfactual: dict[str, Any],
    surface: dict[str, Any],
    contexts: dict[str, TargetContext],
    cameras: dict[str, Any],
    static_masks: dict[str, np.ndarray],
    gaussians: GaussianModel,
    pipe: Any,
    background: torch.Tensor,
    *,
    max_update_per_cluster: int,
    min_target_improvement: float,
    max_clean_increase: float,
    visuals_root: Path,
) -> dict[str, Any]:
    """Counterfactually test the actual plane-projection geometry action.

    A non-zero position gradient is only a local indication that moving a
    primitive might change an anomaly.  It does not establish that projecting
    the primitive to the independently recovered Chart/track plane is a useful
    edit.  This intervention temporarily performs exactly that projection,
    re-renders the same target rays and visible clean controls used by the
    opacity-zero test, and restores the frozen model bit-for-bit afterwards.
    ``G_update`` is allowed only when this direct geometry counterfactual
    passes, not merely when a gradient happens to be non-zero.
    """
    cluster_id = int(counterfactual.get("cluster_id", -1))
    disabled = {
        "cluster_id": cluster_id,
        "method": "project_attributed_primitives_to_independent_surface_plane",
        "accepted": False,
        "reason": None,
        "candidate_count": 0,
    }
    if str(counterfactual.get("classification")) != "surface_misalignment_or_appearance":
        return {**disabled, "reason": "not_surface_misalignment_class"}
    if not bool(surface.get("surface_known")) or surface.get("plane") is None:
        return {**disabled, "reason": "no_independent_surface_plane"}
    target_names = [
        str(item["name"])
        for item in counterfactual.get("targets", [])
        if str(item.get("name", "")) in contexts
    ]
    if not target_names:
        return {**disabled, "reason": "no_target_context"}
    with torch.no_grad():
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        normals = get_gaussian_normal(
            gaussians.get_rotation, gaussians.get_scaling
        ).detach().cpu().numpy()
        scales = gaussians.get_scaling.detach().cpu().numpy()
    local_ids = _plane_local_cluster_ids(
        np.asarray(counterfactual.get("primitive_ids", []), dtype=np.int64),
        surface,
        xyz,
        normals,
        scales,
        max_count=max_update_per_cluster,
    )
    if not len(local_ids):
        return {**disabled, "reason": "no_attributed_primitive_on_surface_footprint"}
    ids = torch.as_tensor(local_ids, dtype=torch.long, device=gaussians._xyz.device)
    plane = torch.as_tensor(
        np.asarray(surface["plane"], dtype=np.float32).reshape(4),
        dtype=gaussians._xyz.dtype,
        device=gaussians._xyz.device,
    )
    normal = plane[:3] / plane[:3].norm().clamp_min(1e-12)

    target_before = {
        str(item["name"]): float(item["baseline_mae"])
        for item in counterfactual.get("targets", [])
        if str(item.get("name", "")) in target_names
    }
    controls = []
    with torch.no_grad():
        for item in counterfactual.get("clean_controls", []):
            name = str(item.get("name", ""))
            if name not in cameras or name not in static_masks:
                continue
            package = render(cameras[name], gaussians, pipe, background)
            visible_fraction = float(
                package["visibility_filter"][ids].detach().float().mean().item()
            )
            # An invisible view cannot certify that a local geometry update is
            # harmless.  Retain only controls that actually see this subset.
            if visible_fraction <= 0.0:
                continue
            controls.append(
                {
                    "name": name,
                    "baseline_mae": float(item["baseline_mae"]),
                    "affine": np.asarray(item["affine"], dtype=np.float32),
                    "visible_fraction": visible_fraction,
                }
            )
    if not controls:
        return {
            **disabled,
            "reason": "no_visible_clean_control_for_local_projection",
            "candidate_count": int(len(local_ids)),
        }

    original_xyz = gaussians._xyz.detach()[ids].clone()
    signed = original_xyz @ normal + plane[3]
    projected_xyz = original_xyz - signed[:, None] * normal[None]
    displacement = (projected_xyz - original_xyz).norm(dim=1)
    after_targets: dict[str, np.ndarray] = {}
    after_controls: dict[str, np.ndarray] = {}
    try:
        with torch.no_grad():
            gaussians._xyz[ids] = projected_xyz
            for name in target_names:
                after_targets[name] = render(contexts[name].camera, gaussians, pipe, background)["render"].detach().cpu().permute(1, 2, 0).numpy()
            for control in controls:
                name = control["name"]
                after_controls[name] = render(cameras[name], gaussians, pipe, background)["render"].detach().cpu().permute(1, 2, 0).numpy()
    finally:
        with torch.no_grad():
            gaussians._xyz[ids] = original_xyz

    target_rows = []
    target_improvements = []
    for name in target_names:
        context = contexts[name]
        before = target_before.get(
            name,
            _fixed_mae(context.baseline_render, context.gt, context.anomaly_mask, context.affine),
        )
        after = _fixed_mae(after_targets[name], context.gt, context.anomaly_mask, context.affine)
        improvement = (before - after) / max(before, 1e-6)
        target_improvements.append(improvement)
        target_rows.append(
            {
                "name": name,
                "baseline_mae": before,
                "projected_mae": after,
                "relative_improvement": improvement,
            }
        )
        root = visuals_root / f"cluster_{cluster_id:03d}"
        _save_rgb(root / f"{_safe_name(name)}.baseline.png", context.baseline_render)
        _save_rgb(root / f"{_safe_name(name)}.plane_projected.png", after_targets[name])
        _save_rgb(root / f"{_safe_name(name)}.gt.png", context.gt)
        _save_mask(root / f"{_safe_name(name)}.anomaly.png", context.anomaly_mask)
    control_rows = []
    control_increases = []
    for control in controls:
        name = control["name"]
        after = _fixed_mae(
            after_controls[name],
            _camera_gt(cameras[name]),
            static_masks[name],
            control["affine"],
        )
        increase = after - float(control["baseline_mae"])
        control_increases.append(increase)
        control_rows.append(
            {
                "name": name,
                "baseline_mae": float(control["baseline_mae"]),
                "projected_mae": after,
                "mae_increase": increase,
                "visible_fraction": float(control["visible_fraction"]),
            }
        )
        root = visuals_root / f"cluster_{cluster_id:03d}"
        _save_rgb(root / f"{_safe_name(name)}.clean.plane_projected.png", after_controls[name])
    median_improvement = float(np.median(target_improvements)) if target_improvements else -float("inf")
    worst_clean_increase = float(max(control_increases)) if control_increases else float("inf")
    accepted = bool(
        median_improvement >= float(min_target_improvement)
        and worst_clean_increase <= float(max_clean_increase)
    )
    report = {
        **disabled,
        "reason": None if accepted else "plane_projection_target_or_clean_gate",
        "candidate_count": int(len(local_ids)),
        "primitive_ids": local_ids,
        "plane": np.asarray(surface["plane"], dtype=np.float64),
        "mean_projection_displacement": float(displacement.mean().item()),
        "max_projection_displacement": float(displacement.max().item()),
        "targets": target_rows,
        "clean_controls": control_rows,
        "median_target_relative_improvement": median_improvement,
        "worst_clean_mae_increase": worst_clean_increase,
        "acceptance_thresholds": {
            "min_target_relative_improvement": float(min_target_improvement),
            "max_clean_mae_increase": float(max_clean_increase),
            "min_visible_clean_controls": 1,
        },
        "accepted": accepted,
    }
    _write_json(visuals_root / f"cluster_{cluster_id:03d}" / "plane_projection_counterfactual.json", report)
    return report


def _plane_quaternions(normal: np.ndarray, count: int) -> np.ndarray:
    """Return wxyz rotations whose local +z normal lies on a known plane."""
    target = np.asarray(normal, dtype=np.float64).reshape(3)
    target /= np.linalg.norm(target).clip(min=1e-12)
    source = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(source @ target, -1.0, 1.0))
    if dot < -0.999999:
        quaternion = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    else:
        cross = np.cross(source, target)
        quaternion = np.asarray([1.0 + dot, cross[0], cross[1], cross[2]], dtype=np.float64)
        quaternion /= np.linalg.norm(quaternion).clip(min=1e-12)
    return np.repeat(quaternion[None], int(count), axis=0).astype(np.float32)


def _sample_real_colors(
    points: np.ndarray,
    view_names: list[str],
    cameras: dict[str, Any],
    geometries: dict[str, CameraGeometry],
    static_masks: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse only real, semantic-valid observations for conservative G_create."""
    value = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    colors = np.full((len(value), 3), 0.5, dtype=np.float32)
    support_count = np.zeros(len(value), dtype=np.int32)
    for point_index, point in enumerate(value):
        samples = []
        for name in view_names:
            if name not in cameras or name not in geometries:
                continue
            pixels, _, inside = project_points(point[None], geometries[name])
            if not bool(inside[0]):
                continue
            x = int(np.clip(round(float(pixels[0, 0])), 0, geometries[name].width - 1))
            y = int(np.clip(round(float(pixels[0, 1])), 0, geometries[name].height - 1))
            if not bool(static_masks[name][y, x]):
                continue
            samples.append(_camera_gt(cameras[name])[y, x])
        if samples:
            colors[point_index] = np.median(np.asarray(samples, dtype=np.float32), axis=0)
            support_count[point_index] = len(samples)
    return colors, support_count


def _build_local_edit_plan(
    counterfactuals: list[dict[str, Any]],
    surface_evidence: dict[str, dict[str, Any]],
    geometry_projection_counterfactuals: dict[str, dict[str, Any]],
    gaussians: GaussianModel,
    cameras: dict[str, Any],
    static_masks: dict[str, np.ndarray],
    geometries: dict[str, CameraGeometry],
    *,
    max_update_per_cluster: int,
    max_create_per_cluster: int,
) -> dict[str, Any]:
    """Bind causal reports to G_remove/G_update/G_create with real support.

    ``G_update`` is restricted to already-attributed primitives that intersect
    the independently recovered planar footprint *and* pass a direct
    plane-projection counterfactual.  ``G_create`` is seeded only on
    feature-track-supported points for a verified missing surface (or a hole
    exposed by a verified foreground removal).  The returned metadata carries
    the plane group for every editable row, so later optimization can project
    positions back to the proxy after every gradient step.
    """
    with torch.no_grad():
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        normals = get_gaussian_normal(gaussians.get_rotation, gaussians.get_scaling).detach().cpu().numpy()
        scales = gaussians.get_scaling.detach().cpu().numpy()
    remove_ids = np.unique(
        np.concatenate(
            [np.asarray(item["primitive_ids"], dtype=np.int64) for item in counterfactuals if item.get("accepted")]
        )
    ) if any(item.get("accepted") for item in counterfactuals) else np.empty(0, dtype=np.int64)
    remove_set = set(map(int, remove_ids.tolist()))
    update_ids: list[int] = []
    seed_points: list[np.ndarray] = []
    seed_colors: list[np.ndarray] = []
    seed_scales: list[np.ndarray] = []
    seed_rotations: list[np.ndarray] = []
    seed_support: list[int] = []
    plane_groups: list[dict[str, Any]] = []
    audit_groups: list[dict[str, Any]] = []
    for item in counterfactuals:
        cluster_id = str(item.get("cluster_id"))
        surface = surface_evidence.get(cluster_id, {})
        if not bool(surface.get("surface_known")):
            continue
        plane = np.asarray(surface["plane"], dtype=np.float64).reshape(4)
        classification = str(item.get("classification", ""))
        local_ids = _plane_local_cluster_ids(
            np.asarray(item["primitive_ids"], dtype=np.int64),
            surface,
            xyz,
            normals,
            scales,
            max_count=max_update_per_cluster,
        )
        local_ids = np.asarray([value for value in local_ids if int(value) not in remove_set], dtype=np.int64)
        group_update = np.empty(0, dtype=np.int64)
        projection_trial = geometry_projection_counterfactuals.get(cluster_id, {})
        # A masked xyz gradient only proposes a surface-attached mismatch.  A
        # direct temporary projection to the independent plane must then help
        # target rays without harming a visible clean control before the same
        # rows can enter G_update.
        if (
            classification == "surface_misalignment_or_appearance"
            and bool(projection_trial.get("accepted"))
        ):
            approved_ids = np.asarray(projection_trial.get("primitive_ids", []), dtype=np.int64)
            group_update = np.intersect1d(local_ids, approved_ids, assume_unique=False)
            update_ids.extend(map(int, group_update.tolist()))

        group_seed_start = len(seed_points)
        if classification in {"missing_surface", "foreground_plus_hole"}:
            support_points = np.asarray(surface.get("points", []), dtype=np.float64).reshape(-1, 3)
            inliers = np.asarray(surface.get("inlier_mask", []), dtype=bool)
            if len(inliers) == len(support_points):
                support_points = support_points[inliers]
            sample_ids = _farthest_sample_indices(support_points, max_create_per_cluster)
            selected_points = support_points[sample_ids]
            view_names = [str(surface.get("target_name", "")), *map(str, surface.get("support_names", []))]
            colors, support_count = _sample_real_colors(
                selected_points, view_names, cameras, geometries, static_masks
            )
            # A newly created surfel needs at least one real appearance sample;
            # unsupported appearance is left for the strict See3D gate.
            valid_seed = support_count > 0
            selected_points = selected_points[valid_seed]
            colors = colors[valid_seed]
            support_count = support_count[valid_seed]
            if len(selected_points):
                scale_source = local_ids if len(local_ids) else np.asarray(item["primitive_ids"], dtype=np.int64)
                scale_source = scale_source[(scale_source >= 0) & (scale_source < len(scales))]
                footprint_spacing = float(surface.get("footprint_metrics", {}).get("median_spacing", 0.0))
                fallback = max(0.5 * footprint_spacing, 1e-4)
                prototype_scale = (
                    np.median(scales[scale_source], axis=0)
                    if len(scale_source)
                    else np.asarray([fallback, fallback], dtype=np.float64)
                )
                prototype_scale = np.maximum(prototype_scale, fallback)
                seed_points.extend(selected_points)
                seed_colors.extend(colors)
                seed_scales.extend(np.repeat(prototype_scale[None], len(selected_points), axis=0))
                seed_rotations.extend(_plane_quaternions(plane[:3], len(selected_points)))
                seed_support.extend(map(int, support_count.tolist()))
        group_seed_stop = len(seed_points)
        plane_groups.append(
            {
                "cluster_id": int(item["cluster_id"]),
                "plane": plane,
                "slab_thickness": float(surface.get("slab_thickness", 0.0)),
                "update_ids": group_update,
                "create_seed_range": [group_seed_start, group_seed_stop],
            }
        )
        audit_groups.append(
            {
                "cluster_id": int(item["cluster_id"]),
                "classification": classification,
                "surface_known": True,
                "local_attributed_primitive_count": int(len(local_ids)),
                "plane_projection_counterfactual": projection_trial,
                "G_update_count": int(len(group_update)),
                "G_create_seed_count": int(group_seed_stop - group_seed_start),
            }
        )
    update = np.unique(np.asarray(update_ids, dtype=np.int64)) if update_ids else np.empty(0, dtype=np.int64)
    return {
        "G_remove": remove_ids,
        "G_update": update,
        "G_create_points": np.asarray(seed_points, dtype=np.float32).reshape(-1, 3),
        "G_create_colors": np.asarray(seed_colors, dtype=np.float32).reshape(-1, 3),
        "G_create_scales": np.asarray(seed_scales, dtype=np.float32).reshape(-1, 2),
        "G_create_rotations": np.asarray(seed_rotations, dtype=np.float32).reshape(-1, 4),
        "G_create_real_support_count": np.asarray(seed_support, dtype=np.int32),
        "plane_groups": plane_groups,
        "audit_groups": audit_groups,
    }


def _save_roi_projection_audit(
    surface_evidence: dict[str, dict[str, Any]],
    contexts: dict[str, TargetContext],
    cameras: dict[str, Any],
    geometries: dict[str, CameraGeometry],
    gaussians: GaussianModel,
    pipe: Any,
    background: torch.Tensor,
    output: Path,
) -> dict[str, Any]:
    """Project a real 3-D footprint separately into every evidence camera."""
    rows = []
    for cluster_id, evidence in surface_evidence.items():
        roi_path = evidence.get("roi_path")
        if not roi_path:
            continue
        roi = ArtifactROI3D.load(Path(roi_path))
        names = list(dict.fromkeys([
            str(evidence.get("target_name", "")),
            *[str(name) for name in evidence.get("support_names", [])],
        ]))
        for name in names:
            if name not in cameras:
                continue
            if name in contexts:
                depth = contexts[name].baseline_depth
            else:
                with torch.no_grad():
                    package = render(cameras[name], gaussians, pipe, background)
                depth = package["surf_depth"][0].detach().cpu().numpy()
            raw_mask = roi.project_mask(geometries[name])
            visible_mask = roi.project_mask(geometries[name], reference_depth=depth)
            root = output / "roi_projections" / f"cluster_{int(cluster_id):03d}"
            _save_mask(root / f"{_safe_name(name)}.surface_raw.png", raw_mask)
            _save_mask(root / f"{_safe_name(name)}.surface_visible.png", visible_mask)
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "view_name": name,
                    "raw_surface_fraction": float(raw_mask.mean()),
                    "depth_visible_surface_fraction": float(visible_mask.mean()),
                    "raw_pixel_count": int(raw_mask.sum()),
                    "visible_pixel_count": int(visible_mask.sum()),
                }
            )
    payload = {"projection_count": len(rows), "projections": rows}
    _write_json(output / "roi_projection_audit.json", payload)
    return payload


def _virtual_camera(anchor: Any, c2w: np.ndarray, name: str) -> tuple[MiniCam, CameraGeometry]:
    """Create a render-only interpolated camera; it never enters repair loss."""
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    world_view = torch.as_tensor(
        w2c.T, dtype=anchor.world_view_transform.dtype, device=anchor.world_view_transform.device
    )
    full_projection = (
        world_view.unsqueeze(0).bmm(anchor.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera = MiniCam(
        int(anchor.image_width),
        int(anchor.image_height),
        float(anchor.FoVy),
        float(anchor.FoVx),
        float(anchor.znear),
        float(anchor.zfar),
        world_view,
        full_projection,
    )
    geometry = CameraGeometry(
        name=name,
        width=int(anchor.image_width),
        height=int(anchor.image_height),
        w2c=w2c,
        fx=float(anchor.focal_x),
        fy=float(anchor.focal_y),
        cx=float(anchor.image_width) / 2.0,
        cy=float(anchor.image_height) / 2.0,
    )
    return camera, geometry


def _virtual_probe_metrics(package: dict[str, torch.Tensor], mask: np.ndarray) -> dict[str, float]:
    keep = np.asarray(mask, dtype=bool)
    if not np.any(keep):
        return {
            "roi_pixel_count": 0,
            "alpha_low_fraction": 1.0,
            "log_depth_edge_mean": 0.0,
            "normal_edge_mean": 0.0,
            "distortion_mean": 0.0,
        }
    alpha = package["rend_alpha"][0].detach().cpu().numpy()
    depth = package["surf_depth"][0].detach().cpu().numpy()
    normal = package["rend_normal"].detach().cpu().permute(1, 2, 0).numpy()
    distortion = package["rend_dist"][0].detach().cpu().numpy()
    log_depth = np.zeros_like(depth, dtype=np.float32)
    valid_depth = np.isfinite(depth) & (depth > 1e-6)
    log_depth[valid_depth] = np.log(depth[valid_depth])
    depth_edge = np.hypot(
        cv2.Sobel(log_depth, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(log_depth, cv2.CV_32F, 0, 1, ksize=3),
    )
    normal_edge = np.zeros_like(depth_edge)
    for channel in range(3):
        normal_edge += cv2.Sobel(normal[..., channel], cv2.CV_32F, 1, 0, ksize=3) ** 2
        normal_edge += cv2.Sobel(normal[..., channel], cv2.CV_32F, 0, 1, ksize=3) ** 2
    return {
        "roi_pixel_count": int(keep.sum()),
        "alpha_low_fraction": float((alpha[keep] < 0.50).mean()),
        "log_depth_edge_mean": float(depth_edge[keep].mean()),
        "normal_edge_mean": float(np.sqrt(normal_edge[keep]).mean()),
        "distortion_mean": float(np.abs(distortion[keep]).mean()),
    }


def _virtual_instability_rays(
    source_package: dict[str, torch.Tensor],
    source_camera: CameraGeometry,
    target_package: dict[str, torch.Tensor],
    target_camera: CameraGeometry,
    *,
    stride: int = 3,
    depth_relative_threshold: float = 0.10,
    normal_angle_threshold_degrees: float = 30.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mark diagnostic-only virtual rays with an unstable depth/normal warp.

    These are intentionally *not* an ROI and never enter a repair loss.  They
    are the virtual-view counterpart of real anomaly rays: an instability can
    nominate a region for real-view inspection, but cannot establish geometry
    or authorize an edit without the later primitive counterfactual.
    """
    source_depth = source_package["surf_depth"][0].detach().cpu().numpy().astype(np.float64)
    source_alpha = source_package["rend_alpha"][0].detach().cpu().numpy().astype(np.float64)
    source_normal = source_package["rend_normal"].detach().cpu().permute(1, 2, 0).numpy().astype(np.float64)
    target_depth = target_package["surf_depth"][0].detach().cpu().numpy().astype(np.float64)
    target_alpha = target_package["rend_alpha"][0].detach().cpu().numpy().astype(np.float64)
    target_normal = target_package["rend_normal"].detach().cpu().permute(1, 2, 0).numpy().astype(np.float64)
    height, width = source_depth.shape
    if (height, width) != (source_camera.height, source_camera.width):
        raise ValueError("Source virtual maps do not match source camera")
    if target_depth.shape != (target_camera.height, target_camera.width):
        raise ValueError("Target virtual maps do not match target camera")
    mask = np.zeros((height, width), dtype=bool)
    yy, xx = np.mgrid[0:height:int(stride), 0:width:int(stride)]
    x = xx.reshape(-1).astype(np.float64)
    y = yy.reshape(-1).astype(np.float64)
    z = source_depth[yy, xx].reshape(-1)
    source_valid = np.isfinite(z) & (z > 1e-6) & (source_alpha[yy, xx].reshape(-1) >= 0.50)
    if not np.any(source_valid):
        return mask, {
            "source_valid_count": 0,
            "projected_count": 0,
            "risk_count": 0,
            "risk_fraction": 0.0,
            "used_for_causal_selection": False,
            "used_for_repair_loss": False,
        }
    camera_points = np.stack(
        [
            (x[source_valid] - source_camera.cx) * z[source_valid] / source_camera.fx,
            (y[source_valid] - source_camera.cy) * z[source_valid] / source_camera.fy,
            z[source_valid],
        ],
        axis=1,
    )
    homogeneous = np.concatenate([camera_points, np.ones((len(camera_points), 1))], axis=1)
    world = (source_camera.c2w @ homogeneous.T).T[:, :3]
    pixels, projected_depth, inside = project_points(world, target_camera)
    rounded = np.rint(pixels).astype(np.int64)
    inside &= (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < target_camera.width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < target_camera.height)
    )
    if not np.any(inside):
        return mask, {
            "source_valid_count": int(source_valid.sum()),
            "projected_count": 0,
            "risk_count": 0,
            "risk_fraction": 0.0,
            "used_for_causal_selection": False,
            "used_for_repair_loss": False,
        }
    source_x = x[source_valid][inside].astype(np.int64)
    source_y = y[source_valid][inside].astype(np.int64)
    target_x = rounded[inside, 0]
    target_y = rounded[inside, 1]
    observed_depth = target_depth[target_y, target_x]
    observed_alpha = target_alpha[target_y, target_x]
    target_valid = np.isfinite(observed_depth) & (observed_depth > 1e-6) & (observed_alpha >= 0.50)
    relative_depth = np.full(len(observed_depth), np.inf, dtype=np.float64)
    relative_depth[target_valid] = np.abs(projected_depth[inside][target_valid] - observed_depth[target_valid]) / np.maximum(
        observed_depth[target_valid], 1e-6
    )
    source_normals = source_normal[source_y, source_x]
    target_normals = target_normal[target_y, target_x]
    source_norm = np.linalg.norm(source_normals, axis=1)
    target_norm = np.linalg.norm(target_normals, axis=1)
    normal_valid = target_valid & (source_norm > 1e-6) & (target_norm > 1e-6)
    normal_angle = np.full(len(observed_depth), 90.0, dtype=np.float64)
    cosine = np.sum(source_normals[normal_valid] * target_normals[normal_valid], axis=1) / (
        source_norm[normal_valid] * target_norm[normal_valid]
    )
    normal_angle[normal_valid] = np.rad2deg(np.arccos(np.clip(np.abs(cosine), 0.0, 1.0)))
    risk = (
        ~target_valid
        | (relative_depth > float(depth_relative_threshold))
        | (normal_angle > float(normal_angle_threshold_degrees))
    )
    mask[source_y[risk], source_x[risk]] = True
    return mask, {
        "source_valid_count": int(source_valid.sum()),
        "projected_count": int(len(observed_depth)),
        "target_depth_valid_count": int(target_valid.sum()),
        "risk_count": int(risk.sum()),
        "risk_fraction": float(risk.sum() / max(int(source_valid.sum()), 1)),
        "depth_relative_p90": float(np.quantile(relative_depth[target_valid], 0.90)) if np.any(target_valid) else None,
        "normal_angle_p90_degrees": float(np.quantile(normal_angle[normal_valid], 0.90)) if np.any(normal_valid) else None,
        "thresholds": {
            "depth_relative": float(depth_relative_threshold),
            "normal_angle_degrees": float(normal_angle_threshold_degrees),
        },
        "used_for_causal_selection": False,
        "used_for_repair_loss": False,
    }


def _run_pre_roi_virtual_diagnostics(
    contexts: dict[str, TargetContext],
    cameras: dict[str, Any],
    geometries: dict[str, CameraGeometry],
    features: dict[str, np.ndarray],
    metric_quality: dict[str, float],
    gaussians: GaussianModel,
    pipe: Any,
    background: torch.Tensor,
    output: Path,
    fractions: list[float],
) -> dict[str, Any]:
    """Probe nearby virtual views before any 3-D ROI is assumed.

    Unlike the later ROI diagnostics, this branch renders the whole virtual
    image and records unstable *rays* only.  It is deliberately detached from
    candidate ranking and optimization, because model-only virtual evidence
    cannot distinguish a true occlusion from a floater without real-image
    corroboration.
    """
    rows = []
    for target_name, context in sorted(contexts.items()):
        peers = _candidate_control_order([target_name], features, metric_quality)
        support_name = next((name for name in peers if name in cameras), None)
        if support_name is None:
            continue
        probes = []
        for fraction in fractions:
            if not 0.0 < float(fraction) < 1.0:
                continue
            c2w = interpolate_c2w(context.geometry.c2w, geometries[support_name].c2w, float(fraction))
            virtual_name = f"pre_roi_{_safe_name(target_name)}_to_{_safe_name(support_name)}_{float(fraction):.2f}"
            camera, geometry = _virtual_camera(context.camera, c2w, virtual_name)
            with torch.no_grad():
                package = render(camera, gaussians, pipe, background)
            probes.append((virtual_name, float(fraction), geometry, package))
            root = output / "pre_roi_virtual_diagnostics" / _safe_name(target_name)
            _save_rgb(root / f"{virtual_name}.render.png", package["render"])
        for previous, current in zip(probes[:-1], probes[1:]):
            risk_mask, risk = _virtual_instability_rays(
                previous[3], previous[2], current[3], current[2]
            )
            root = output / "pre_roi_virtual_diagnostics" / _safe_name(target_name)
            _save_mask(root / f"{previous[0]}_to_{current[0]}.risk_rays.png", risk_mask)
            rows.append(
                {
                    "target_real_view": target_name,
                    "support_real_view": support_name,
                    "source_virtual_name": previous[0],
                    "target_virtual_name": current[0],
                    "source_fraction": previous[1],
                    "target_fraction": current[1],
                    "risk": risk,
                    "warp_metrics": virtual_reprojection_consistency(
                        previous[3]["render"].detach().cpu().permute(1, 2, 0).numpy(),
                        previous[3]["surf_depth"][0].detach().cpu().numpy(),
                        previous[3]["rend_alpha"][0].detach().cpu().numpy(),
                        previous[3]["rend_normal"].detach().cpu().permute(1, 2, 0).numpy(),
                        previous[2],
                        current[3]["render"].detach().cpu().permute(1, 2, 0).numpy(),
                        current[3]["surf_depth"][0].detach().cpu().numpy(),
                        current[3]["rend_alpha"][0].detach().cpu().numpy(),
                        current[3]["rend_normal"].detach().cpu().permute(1, 2, 0).numpy(),
                        current[2],
                    ),
                    "used_for_causal_selection": False,
                    "used_for_repair_loss": False,
                }
            )
    payload = {
        "mode": "pre_roi_virtual_instability_rays_diagnostic_only",
        "virtual_views_are_not_geometry_evidence_or_supervision": True,
        "pair_count": len(rows),
        "pairs": rows,
    }
    _write_json(output / "pre_roi_virtual_diagnostics.json", payload)
    return payload


def _run_virtual_diagnostics(
    surface_evidence: dict[str, dict[str, Any]],
    contexts: dict[str, TargetContext],
    cameras: dict[str, Any],
    geometries: dict[str, CameraGeometry],
    gaussians: GaussianModel,
    pipe: Any,
    background: torch.Tensor,
    output: Path,
    fractions: list[float],
) -> dict[str, Any]:
    """Render interpolation probes as diagnostics, never as repair supervision.

    Adjacent probes are depth-warped into one another and audited for
    RGB/normal/depth stability.  These diagnostics may expose a floater or a
    visibility switch, but are intentionally never used as a repair loss or
    generated-image condition.
    """
    rows = []
    stability_pairs = []
    for cluster_id, evidence in surface_evidence.items():
        target_name = str(evidence.get("target_name", ""))
        support_names = [str(name) for name in evidence.get("support_names", [])]
        if not evidence.get("roi_path") or target_name not in cameras:
            continue
        support_name = next((name for name in support_names if name in cameras), None)
        if support_name is None:
            continue
        roi = ArtifactROI3D.load(Path(evidence["roi_path"]))
        first = geometries[target_name]
        second = geometries[support_name]
        previous_probe = None
        for fraction in fractions:
            if not 0.0 < float(fraction) < 1.0:
                continue
            c2w = interpolate_c2w(first.c2w, second.c2w, float(fraction))
            virtual_name = f"cluster{int(cluster_id):03d}_{_safe_name(target_name)}_to_{_safe_name(support_name)}_{float(fraction):.2f}"
            camera, geometry = _virtual_camera(cameras[target_name], c2w, virtual_name)
            with torch.no_grad():
                package = render(camera, gaussians, pipe, background)
            depth = package["surf_depth"][0].detach().cpu().numpy()
            raw_mask = roi.project_mask(geometry)
            visible_mask = roi.project_mask(geometry, reference_depth=depth)
            root = output / "virtual_diagnostics" / f"cluster_{int(cluster_id):03d}"
            _save_rgb(root / f"{virtual_name}.render.png", package["render"])
            _save_mask(root / f"{virtual_name}.surface_raw.png", raw_mask)
            _save_mask(root / f"{virtual_name}.surface_visible.png", visible_mask)
            current_probe = {
                "name": virtual_name,
                "fraction": float(fraction),
                "geometry": geometry,
                "visible_mask": visible_mask,
                "render": package["render"].detach().cpu().permute(1, 2, 0).numpy(),
                "depth": package["surf_depth"][0].detach().cpu().numpy(),
                "alpha": package["rend_alpha"][0].detach().cpu().numpy(),
                "normal": package["rend_normal"].detach().cpu().permute(1, 2, 0).numpy(),
            }
            if previous_probe is not None:
                stability_pairs.append(
                    {
                        "cluster_id": int(cluster_id),
                        "source_virtual_name": previous_probe["name"],
                        "target_virtual_name": current_probe["name"],
                        "source_fraction": previous_probe["fraction"],
                        "target_fraction": current_probe["fraction"],
                        "metrics": virtual_reprojection_consistency(
                            previous_probe["render"],
                            previous_probe["depth"],
                            previous_probe["alpha"],
                            previous_probe["normal"],
                            previous_probe["geometry"],
                            current_probe["render"],
                            current_probe["depth"],
                            current_probe["alpha"],
                            current_probe["normal"],
                            current_probe["geometry"],
                            source_mask=previous_probe["visible_mask"],
                        ),
                        "used_for_repair_selection": False,
                        "used_for_repair_loss": False,
                    }
                )
            previous_probe = current_probe
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "target_name": target_name,
                    "support_name": support_name,
                    "fraction": float(fraction),
                    "camera": geometry.as_json(),
                    "metrics": _virtual_probe_metrics(package, visible_mask),
                    "used_for_repair_selection": False,
                    "used_for_repair_loss": False,
                }
            )
    payload = {
        "mode": "interpolated_real_camera_pair_diagnostic_only",
        "probe_count": len(rows),
        "virtual_views_are_not_training_or_generation_supervision": True,
        "probes": rows,
        "depth_warp_stability_pairs": stability_pairs,
    }
    _write_json(output / "virtual_diagnostics.json", payload)
    return payload


def _virtual_stability_nonregression(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    max_depth_consistent_drop: float = 0.05,
    max_depth_relative_p90_increase: float = 0.03,
    max_normal_angle_p90_increase_degrees: float = 5.0,
) -> dict[str, Any]:
    """Compare only matching diagnostic probes before and after a local edit."""
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("cluster_id"),
            row.get("source_virtual_name"),
            row.get("target_virtual_name"),
        )

    before_pairs = {key(row): row.get("metrics", {}) for row in before.get("depth_warp_stability_pairs", [])}
    after_pairs = {key(row): row.get("metrics", {}) for row in after.get("depth_warp_stability_pairs", [])}
    if not before_pairs:
        return {
            "available": False,
            "passed": True,
            "reason": "no_real_surface_roi_virtual_probe",
            "comparable_pair_count": 0,
        }
    shared = sorted(set(before_pairs) & set(after_pairs), key=str)
    if not shared:
        return {
            "available": True,
            "passed": False,
            "reason": "post_edit_virtual_probe_missing",
            "comparable_pair_count": 0,
        }
    depth_fraction_deltas = []
    depth_p90_increases = []
    normal_p90_increases = []
    for item in shared:
        previous = before_pairs[item]
        current = after_pairs[item]
        if previous.get("depth_consistent_fraction") is not None and current.get("depth_consistent_fraction") is not None:
            depth_fraction_deltas.append(
                float(current["depth_consistent_fraction"]) - float(previous["depth_consistent_fraction"])
            )
        if previous.get("depth_relative_p90") is not None and current.get("depth_relative_p90") is not None:
            depth_p90_increases.append(
                float(current["depth_relative_p90"]) - float(previous["depth_relative_p90"])
            )
        if previous.get("normal_angle_p90_degrees") is not None and current.get("normal_angle_p90_degrees") is not None:
            normal_p90_increases.append(
                float(current["normal_angle_p90_degrees"])
                - float(previous["normal_angle_p90_degrees"])
            )
    worst_depth_fraction_delta = min(depth_fraction_deltas) if depth_fraction_deltas else 0.0
    worst_depth_p90_increase = max(depth_p90_increases) if depth_p90_increases else 0.0
    worst_normal_p90_increase = max(normal_p90_increases) if normal_p90_increases else 0.0
    passed = bool(
        worst_depth_fraction_delta >= -float(max_depth_consistent_drop)
        and worst_depth_p90_increase <= float(max_depth_relative_p90_increase)
        and worst_normal_p90_increase <= float(max_normal_angle_p90_increase_degrees)
    )
    return {
        "available": True,
        "passed": passed,
        "reason": None if passed else "virtual_depth_or_normal_stability_regression",
        "comparable_pair_count": len(shared),
        "worst_depth_consistent_fraction_delta": float(worst_depth_fraction_delta),
        "worst_depth_relative_p90_increase": float(worst_depth_p90_increase),
        "worst_normal_angle_p90_increase_degrees": float(worst_normal_p90_increase),
        "thresholds": {
            "max_depth_consistent_drop": float(max_depth_consistent_drop),
            "max_depth_relative_p90_increase": float(max_depth_relative_p90_increase),
            "max_normal_angle_p90_increase_degrees": float(max_normal_angle_p90_increase_degrees),
        },
    }


def _parameter_map(gaussians: GaussianModel) -> dict[str, torch.Tensor]:
    return {
        "xyz": gaussians._xyz,
        "f_dc": gaussians._features_dc,
        "f_rest": gaussians._features_rest,
        "opacity": gaussians._opacity,
        "scaling": gaussians._scaling,
        "rotation": gaussians._rotation,
    }


def _indexed_adam_step(
    parameters: dict[str, torch.Tensor],
    row_ids: dict[str, torch.Tensor],
    state: dict[str, dict[str, torch.Tensor]],
    learning_rates: dict[str, float],
    step: int,
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> None:
    """Adam over selected rows only; no global Gaussian tensor is an optimizer parameter."""
    with torch.no_grad():
        for name, parameter in parameters.items():
            ids = row_ids.get(name)
            if ids is None or not len(ids) or parameter.grad is None:
                continue
            gradient = parameter.grad[ids].detach()
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Non-finite selected gradient for {name}")
            if name not in state:
                state[name] = {"m": torch.zeros_like(gradient), "v": torch.zeros_like(gradient)}
            moment = state[name]["m"]
            variance = state[name]["v"]
            moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            variance.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            corrected_moment = moment / (1.0 - beta1 ** int(step))
            corrected_variance = variance / (1.0 - beta2 ** int(step))
            delta = float(learning_rates[name]) * corrected_moment / (corrected_variance.sqrt() + epsilon)
            parameter[ids] = parameter[ids] - delta


def _materialize_plane_groups(edit_plan: dict[str, Any], base_count: int) -> list[dict[str, Any]]:
    groups = []
    for source in edit_plan.get("plane_groups", []):
        ids = list(map(int, np.asarray(source.get("update_ids", []), dtype=np.int64).tolist()))
        start, stop = map(int, source.get("create_seed_range", [0, 0]))
        ids.extend(range(int(base_count) + start, int(base_count) + stop))
        if ids:
            groups.append({**source, "editable_ids": np.unique(np.asarray(ids, dtype=np.int64))})
    return groups


def _project_selected_to_planes(gaussians: GaussianModel, groups: list[dict[str, Any]]) -> None:
    with torch.no_grad():
        for group in groups:
            ids = torch.as_tensor(group["editable_ids"], dtype=torch.long, device=gaussians._xyz.device)
            plane = torch.as_tensor(group["plane"], dtype=gaussians._xyz.dtype, device=gaussians._xyz.device)
            normal = plane[:3] / torch.linalg.vector_norm(plane[:3]).clamp_min(1e-12)
            points = gaussians._xyz[ids]
            signed = points @ normal + plane[3]
            gaussians._xyz[ids] = points - signed[:, None] * normal[None]
            # The local patch proxy constrains both the surfel center and its
            # disk normal.  Leaving rotations free would let a selected row
            # remain a grazing/tilted floater even though its center is on the
            # recovered facade plane.  Tangential rotation is immaterial for a
            # plane normal, so use the deterministic +z -> plane-normal frame.
            rotations = torch.from_numpy(
                _plane_quaternions(normal.detach().cpu().numpy(), len(ids))
            ).to(device=gaussians._rotation.device, dtype=gaussians._rotation.dtype)
            gaussians._rotation[ids] = rotations


def _geometry_constraint_audit(
    gaussians: GaussianModel,
    plane_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify the actual editable rows obey their independently recovered plane."""
    rows = []
    with torch.no_grad():
        for group in plane_groups:
            ids = np.asarray(group.get("editable_ids", []), dtype=np.int64)
            ids = ids[(ids >= 0) & (ids < len(gaussians._xyz))]
            if not len(ids):
                continue
            tensor_ids = torch.as_tensor(ids, dtype=torch.long, device=gaussians._xyz.device)
            plane = torch.as_tensor(group["plane"], dtype=gaussians._xyz.dtype, device=gaussians._xyz.device)
            normal = plane[:3] / torch.linalg.vector_norm(plane[:3]).clamp_min(1e-12)
            signed = (gaussians._xyz[tensor_ids] @ normal + plane[3]).abs().detach().cpu().numpy()
            surfel_normals = get_gaussian_normal(
                gaussians._rotation[tensor_ids], gaussians.get_scaling[tensor_ids]
            )
            cosine = torch.abs(surfel_normals @ normal).clamp(0.0, 1.0).detach().cpu().numpy()
            angle = np.rad2deg(np.arccos(cosine))
            slab = float(group.get("slab_thickness", 0.0))
            rows.append(
                {
                    "cluster_id": int(group["cluster_id"]),
                    "editable_count": int(len(ids)),
                    "plane_abs_residual_p90": float(np.quantile(signed, 0.90)),
                    "plane_abs_residual_max": float(np.max(signed)),
                    "normal_angle_p90_degrees": float(np.quantile(angle, 0.90)),
                    "normal_angle_max_degrees": float(np.max(angle)),
                    "plane_residual_tolerance": max(1e-4, 0.05 * slab),
                }
            )
    passed = all(
        row["plane_abs_residual_p90"] <= row["plane_residual_tolerance"]
        and row["normal_angle_p90_degrees"] <= 1.0
        for row in rows
    )
    return {
        "available": bool(rows),
        "passed": bool(passed),
        "reason": None if rows else "no_plane_constrained_G_update_or_G_create",
        "groups": rows,
    }


def _frozen_prefix_audit(
    gaussians: GaussianModel,
    snapshots: dict[str, torch.Tensor],
    base_count: int,
    editable_existing: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Prove that rows outside the causal sets remain bit-identical."""
    output = {}
    for name, parameter in _parameter_map(gaussians).items():
        current = parameter[:base_count].detach()
        reference = snapshots[name]
        editable = torch.zeros(base_count, dtype=torch.bool, device=current.device)
        ids = np.asarray(editable_existing.get(name, np.empty(0, dtype=np.int64)), dtype=np.int64)
        if len(ids):
            editable[torch.as_tensor(ids, dtype=torch.long, device=current.device)] = True
        frozen = ~editable
        if torch.any(frozen):
            difference = (current[frozen] - reference[frozen]).abs()
            changed_rows = int(torch.count_nonzero(difference.reshape(len(difference), -1).amax(dim=1) > 0).item())
            max_delta = float(difference.max().item())
        else:
            changed_rows = 0
            max_delta = 0.0
        output[name] = {
            "frozen_row_count": int(frozen.sum().item()),
            "changed_frozen_row_count": changed_rows,
            "max_frozen_abs_delta": max_delta,
        }
    output["all_frozen_rows_bit_identical"] = bool(
        all(item["changed_frozen_row_count"] == 0 for key, item in output.items() if key != "all_frozen_rows_bit_identical")
    )
    return output


def _local_constrained_edit(
    edit_plan: dict[str, Any],
    contexts: dict[str, TargetContext],
    cameras: dict[str, Any],
    static_masks: dict[str, np.ndarray],
    gaussians: GaussianModel,
    pipe: Any,
    background: torch.Tensor,
    *,
    control_names: list[str],
    steps: int,
    clean_every: int,
    position_lr: float,
    appearance_lr: float,
    opacity_lr: float,
    plane_weight: float,
    normal_weight: float,
    preserve_weight: float,
    distortion_weight: float,
    regularization_weight: float,
    initial_opacity: float,
) -> dict[str, Any]:
    """Apply G_remove/G_update/G_create with an indexed, constrained optimizer.

    This intentionally does *not* instantiate ``torch.optim.Adam`` over a
    global Gaussian tensor.  Gradients are read only from selected rows, Adam
    state exists only for those rows, and a post-edit bitwise audit verifies
    that every baseline row outside the approved causal sets stayed unchanged.
    """
    base_count = int(len(gaussians._xyz))
    snapshots = {name: value[:base_count].detach().clone() for name, value in _parameter_map(gaussians).items()}
    remove_ids = np.asarray(edit_plan["G_remove"], dtype=np.int64)
    update_ids = np.asarray(edit_plan["G_update"], dtype=np.int64)
    if len(remove_ids):
        _apply_opacity_factor(gaussians, remove_ids, initial_opacity)

    create_points = np.asarray(edit_plan["G_create_points"], dtype=np.float32).reshape(-1, 3)
    if len(create_points):
        gaussians.append_from_parameters(
            torch.from_numpy(create_points),
            torch.from_numpy(np.asarray(edit_plan["G_create_scales"], dtype=np.float32)),
            torch.from_numpy(np.asarray(edit_plan["G_create_rotations"], dtype=np.float32)),
            torch.from_numpy(np.asarray(edit_plan["G_create_colors"], dtype=np.float32)),
            initial_opacity=float(initial_opacity),
        )
    new_ids = np.arange(base_count, len(gaussians._xyz), dtype=np.int64)
    plane_groups = _materialize_plane_groups(edit_plan, base_count)
    # Plane projection is a geometric constraint, not a soft pseudo-view cue.
    _project_selected_to_planes(gaussians, plane_groups)
    train_ids = np.unique(np.concatenate([update_ids, new_ids])) if len(update_ids) or len(new_ids) else np.empty(0, dtype=np.int64)
    train_ids = train_ids[(train_ids >= 0) & (train_ids < len(gaussians._xyz))]
    parameter_rows = {
        name: torch.as_tensor(train_ids, dtype=torch.long, device=gaussians._xyz.device)
        for name in _parameter_map(gaussians)
    }
    initial_rows = {
        name: value[parameter_rows[name]].detach().clone()
        for name, value in _parameter_map(gaussians).items()
    }
    # Calibrate real clean controls once before editing.  Refitting this
    # affine map after an edit would hide a color regression.
    control_cache: dict[str, tuple[torch.Tensor, torch.Tensor, np.ndarray]] = {}
    for name in control_names:
        if name not in cameras:
            continue
        with torch.no_grad():
            _, rgb, affine = _render_context(gaussians, cameras[name], pipe, background, static_masks[name])
        control_cache[name] = (
            _to_tensor(_camera_gt(cameras[name]).transpose(2, 0, 1), gaussians.get_xyz.device),
            _to_tensor(static_masks[name], gaussians.get_xyz.device).bool(),
            affine,
        )
    target_cache = {
        name: (
            _to_tensor(context.gt.transpose(2, 0, 1), gaussians.get_xyz.device),
            _to_tensor(context.anomaly_mask, gaussians.get_xyz.device).bool(),
            _to_tensor(context.score, gaussians.get_xyz.device),
        )
        for name, context in contexts.items()
    }
    trace = []
    state: dict[str, dict[str, torch.Tensor]] = {}
    learning_rates = {
        "xyz": float(position_lr),
        "f_dc": float(appearance_lr),
        "f_rest": float(appearance_lr) * 0.10,
        "opacity": float(opacity_lr),
        "scaling": float(position_lr) * 0.50,
        "rotation": float(position_lr) * 0.50,
    }
    # Per-group position anchors limit tangent drift even though normal motion
    # is already hard-projected away after every selected-row update.
    plane_initial_xyz = {
        int(group["cluster_id"]): gaussians._xyz[
            torch.as_tensor(group["editable_ids"], dtype=torch.long, device=gaussians._xyz.device)
        ].detach().clone()
        for group in plane_groups
    }
    for iteration in range(1, int(steps) + 1):
        for parameter in _parameter_map(gaussians).values():
            parameter.grad = None
        target_name = sorted(contexts)[(iteration - 1) % len(contexts)]
        context = contexts[target_name]
        target_rgb, target_mask, target_score = target_cache[target_name]
        package = render(context.camera, gaussians, pipe, background)
        corrected = affine_correct_torch(package["render"], context.affine)
        real_loss = masked_charbonnier(corrected, target_rgb, target_mask, target_score)
        loss = real_loss
        plane_loss = loss * 0.0
        normal_loss = loss * 0.0
        for group in plane_groups:
            ids = torch.as_tensor(group["editable_ids"], dtype=torch.long, device=gaussians._xyz.device)
            if not len(ids):
                continue
            plane = torch.as_tensor(group["plane"], dtype=gaussians._xyz.dtype, device=gaussians._xyz.device)
            normal = plane[:3] / torch.linalg.vector_norm(plane[:3]).clamp_min(1e-12)
            signed = gaussians._xyz[ids] @ normal + plane[3]
            plane_loss = plane_loss + signed.square().mean()
            primitive_normals = get_gaussian_normal(
                gaussians._rotation[ids], gaussians.get_scaling[ids]
            )
            alignment = torch.abs(primitive_normals @ normal)
            normal_loss = normal_loss + (1.0 - alignment).mean()
        loss = loss + float(plane_weight) * plane_loss + float(normal_weight) * normal_loss
        distortion_loss = torch.abs(package["rend_dist"][0][target_mask]).mean() if torch.any(target_mask) else loss * 0.0
        loss = loss + float(distortion_weight) * distortion_loss
        preserve_loss = loss * 0.0
        if control_cache and iteration % max(1, int(clean_every)) == 0:
            control_name = sorted(control_cache)[(iteration // max(1, int(clean_every))) % len(control_cache)]
            control_gt, control_mask, control_affine = control_cache[control_name]
            control_package = render(cameras[control_name], gaussians, pipe, background)
            control_render = affine_correct_torch(control_package["render"], control_affine)
            if torch.any(control_mask):
                preserve_loss = torch.abs(control_render - control_gt).mean(dim=0)[control_mask].mean()
                loss = loss + float(preserve_weight) * preserve_loss
        regularization = loss * 0.0
        if len(train_ids):
            for name, parameter in _parameter_map(gaussians).items():
                regularization = regularization + (parameter[parameter_rows[name]] - initial_rows[name]).square().mean()
            loss = loss + float(regularization_weight) * regularization
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite local constrained-edit loss")
        loss.backward()
        _indexed_adam_step(_parameter_map(gaussians), parameter_rows, state, learning_rates, iteration)
        _project_selected_to_planes(gaussians, plane_groups)
        with torch.no_grad():
            for group in plane_groups:
                ids = torch.as_tensor(group["editable_ids"], dtype=torch.long, device=gaussians._xyz.device)
                if not len(ids):
                    continue
                initial = plane_initial_xyz[int(group["cluster_id"])]
                # The selected row may move along the plane, but not farther
                # than a small geometry-evidence-derived trust region.
                limit = max(5.0 * float(group.get("slab_thickness", 0.0)), 0.01)
                delta = gaussians._xyz[ids] - initial
                norm = torch.linalg.vector_norm(delta, dim=1, keepdim=True).clamp_min(1e-12)
                gaussians._xyz[ids] = initial + delta * torch.clamp(float(limit) / norm, max=1.0)
            for name, parameter in _parameter_map(gaussians).items():
                ids = parameter_rows[name]
                if len(ids) and name in {"scaling", "opacity"}:
                    parameter[ids] = torch.maximum(
                        torch.minimum(parameter[ids], initial_rows[name] + math.log(2.0)),
                        initial_rows[name] - math.log(2.0),
                    )
        if iteration == 1 or iteration == int(steps) or iteration % max(1, int(steps) // 20) == 0:
            trace.append(
                {
                    "iteration": iteration,
                    "loss": float(loss.detach().item()),
                    "real_loss": float(real_loss.detach().item()),
                    "plane_loss": float(plane_loss.detach().item()),
                    "normal_loss": float(normal_loss.detach().item()),
                    "distortion_loss": float(distortion_loss.detach().item()),
                    "preserve_loss": float(preserve_loss.detach().item()),
                    "regularization": float(regularization.detach().item()),
                }
            )

    editable_existing = {
        "xyz": update_ids,
        "f_dc": update_ids,
        "f_rest": update_ids,
        "scaling": update_ids,
        "rotation": update_ids,
        "opacity": np.unique(np.concatenate([remove_ids, update_ids])) if len(remove_ids) or len(update_ids) else np.empty(0, dtype=np.int64),
    }
    geometry_constraint_audit = _geometry_constraint_audit(gaussians, plane_groups)
    frozen_audit = _frozen_prefix_audit(gaussians, snapshots, base_count, editable_existing)
    if not frozen_audit["all_frozen_rows_bit_identical"]:
        raise RuntimeError("Local edit mutated baseline rows outside its causal sets")
    return {
        "mode": "counterfactual_remove_plus_plane_constrained_indexed_adam",
        "baseline_gaussian_count": base_count,
        "G_remove_count": int(len(remove_ids)),
        "G_update_count": int(len(update_ids)),
        "G_create_count": int(len(new_ids)),
        "refinement_steps": int(steps),
        "optimizer_scope": "selected_rows_only_no_global_optimizer_parameters",
        "loss_trace": trace,
        "geometry_constraint_audit": geometry_constraint_audit,
        "frozen_prefix_audit": frozen_audit,
    }


def _apply_opacity_factor(gaussians: GaussianModel, ids: np.ndarray, factor: float) -> None:
    tensor_ids = torch.as_tensor(ids, dtype=torch.long, device=gaussians._opacity.device)
    with torch.no_grad():
        old = torch.sigmoid(gaussians._opacity[tensor_ids])
        gaussians._opacity[tensor_ids] = torch.logit(
            torch.clamp(old * float(factor), 1e-5, 1.0 - 1e-5)
        )


def _validate_edit(
    gaussians: GaussianModel,
    contexts: dict[str, TargetContext],
    cameras: dict[str, Any],
    static_masks: dict[str, np.ndarray],
    pipe: Any,
    background: torch.Tensor,
    *,
    control_names: list[str],
    target_names: list[str] | None = None,
    control_affines: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Evaluate target and clean controls with a fixed photometric gauge.

    The clean-view guard must not refit its affine exposure/color correction
    after an edit: doing so can disguise a genuine appearance regression as a
    camera exposure change.  The pre-edit call estimates and returns one
    affine map per control; the post-edit call receives those exact maps.
    """
    target_scope = (
        sorted({str(name) for name in target_names if name in contexts})
        if target_names is not None
        else sorted(contexts)
    )
    targets = []
    controls = []
    with torch.no_grad():
        for name in target_scope:
            context = contexts[name]
            package = render(context.camera, gaussians, pipe, background)
            rgb = package["render"].detach().cpu().permute(1, 2, 0).numpy()
            targets.append(
                {
                    "name": name,
                    "anomaly_mae": _fixed_mae(rgb, context.gt, context.anomaly_mask, context.affine),
                    "anomaly_alpha": _alpha_summary(package["rend_alpha"][0].detach().cpu().numpy(), context.anomaly_mask),
                }
            )
        for name in control_names:
            affine = None if control_affines is None else control_affines.get(name)
            package, rgb, affine = _render_context(
                gaussians,
                cameras[name],
                pipe,
                background,
                static_masks[name],
                affine=affine,
            )
            controls.append(
                {
                    "name": name,
                    "static_mae": _fixed_mae(rgb, _camera_gt(cameras[name]), static_masks[name], affine),
                    "affine": affine,
                }
            )
    return {
        "targets": targets,
        "clean_controls": controls,
        "target_scope": {
            "policy": (
                "accepted_counterfactual_causal_targets_only"
                if target_names is not None
                else "all_diagnostic_triage_targets"
            ),
            "names": target_scope,
        },
    }


def _accepted_causal_target_names(
    counterfactuals: list[dict[str, Any]],
    contexts: dict[str, TargetContext],
) -> list[str]:
    """Return only real rays whose own intervention authorized this edit.

    The all-train prescan and its small attribution batch are deliberately
    broad: they identify several independent suspects.  A local edit to one
    verified cluster must be scored on its attributed target rays, not on the
    median of unrelated diagnostic views that cannot change.  Cross-view
    safety remains governed by the full real-camera envelope controls.
    """
    return sorted(
        {
            str(row["name"])
            for cluster in counterfactuals
            if bool(cluster.get("accepted"))
            for row in cluster.get("targets", [])
            if str(row.get("name", "")) in contexts
        }
    )


def _strict_see3d_residual_masks(
    roi: ArtifactROI3D,
    context: TargetContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute M_surface, M_real, M_model, M_gen for an audited target view.

    A real target image is itself valid RGB evidence, so it is deliberately
    counted as ``M_real``.  This makes See3D unavailable for an observed train
    surface instead of using generation merely because the current 2DGS render
    is poor.  A later novel-view exporter may evaluate the same formula using
    warped *other* real views, but may not weaken it.
    """
    surface = roi.project_mask(context.geometry) & context.static_mask
    real = surface.copy()
    model = surface & (context.baseline_alpha >= 0.70) & (context.rgb_error <= 0.08)
    generation = surface & ~real & ~model
    return surface, real, model, generation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True, help="Full 1487-image train dataset.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--mask-pickle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--chart-scene",
        type=Path,
        help=(
            "Aligned MAtCha mast3r_sfm directory used as an independent surface source. "
            "Default: infer the exact source_path recorded in the frozen model cfg_args."
        ),
    )
    parser.add_argument(
        "--use-matcha-chart-surface",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Audit and prefer coordinate-matched, multi-Chart surface evidence before SIFT tracks.",
    )
    parser.add_argument(
        "--sfm-sparse-path",
        type=Path,
        help=(
            "Original COLMAP/MASt3R sparse/0 directory containing images.bin, cameras.bin, "
            "and points3D.bin. Default: infer it from the Cambridge mask root."
        ),
    )
    parser.add_argument(
        "--use-original-sfm-tracks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use coordinate-audited original multi-view SfM tracks before the SIFT fallback.",
    )
    parser.add_argument("--target-image-names-file", type=Path)
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--auto-target-count", type=int, default=12)
    parser.add_argument(
        "--auto-target-selection",
        choices=("priority", "joint_diverse"),
        default="joint_diverse",
        help=(
            "When no manual target list is supplied, choose high-residual views by "
            "priority alone or balance priority with real-camera pose coverage and "
            "sequence novelty. This is triage only, never geometry evidence."
        ),
    )
    parser.add_argument(
        "--auto-target-candidate-multiplier",
        type=int,
        default=12,
        help="High-residual prescan pool multiplier used by --auto-target-selection=joint_diverse.",
    )
    parser.add_argument(
        "--full-train-prescan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Audit every saved real training render before selecting causal targets. "
            "This is RGB/structure triage only; alpha/depth attribution is rerun "
            "on the selected views."
        ),
    )
    parser.add_argument(
        "--full-train-prescan-report",
        type=Path,
        help=(
            "Completed prescan from the same frozen model/metrics cache to reuse for "
            "another causal cluster experiment. Exact 1,487-name coverage and the "
            "metrics SHA-256 are revalidated before use."
        ),
    )
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--white-background", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--semantic-mask-indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-candidates-per-view", type=int, default=384)
    parser.add_argument("--max-clusters", type=int, default=12)
    parser.add_argument(
        "--max-cluster-diameter-factor",
        type=float,
        default=2.5,
        help=(
            "Maximum causal-intervention diameter relative to the adaptive "
            "3-D candidate graph radius; longer single-linkage chains are "
            "split before opacity-zero testing."
        ),
    )
    parser.add_argument(
        "--control-policy",
        choices=("real_camera_envelope_v1", "legacy_metric_nearby"),
        default="real_camera_envelope_v1",
        help=(
            "Counterfactual protection panel. The default includes temporal and "
            "pose-near real observations without rejecting low-quality frames; "
            "the legacy mode exists only for historical reproduction."
        ),
    )
    parser.add_argument(
        "--control-count",
        type=int,
        default=32,
        help="Requested real-camera counterfactual envelope size before exact renderer visibility filtering.",
    )
    parser.add_argument(
        "--control-search",
        type=int,
        default=192,
        help="Maximum real-camera envelope candidates rendered to satisfy the visible-control quota.",
    )
    parser.add_argument(
        "--control-temporal-radius",
        type=int,
        default=16,
        help="Same-sequence frame radius included before pose-near counterfactual controls.",
    )
    parser.add_argument(
        "--min-visible-clean-controls",
        type=int,
        default=12,
        help="Minimum exactly visible real-camera controls required before G_remove can be causal-approved.",
    )
    parser.add_argument(
        "--virtual-probe-fractions",
        nargs="+",
        type=float,
        default=[0.25, 0.50, 0.75],
        help="Interpolated real-camera-pair probes for diagnostics only; never repair supervision.",
    )
    parser.add_argument(
        "--pre-roi-virtual-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Audit whole-image virtual instability rays before any 3-D ROI is assumed.",
    )
    parser.add_argument("--residual-threshold", type=float, default=0.12)
    parser.add_argument("--score-threshold", type=float, default=0.38)
    parser.add_argument("--min-component-pixels", type=int, default=48)
    parser.add_argument("--min-bad-improvement", type=float, default=0.04)
    parser.add_argument("--max-clean-mae-increase", type=float, default=0.006)
    parser.add_argument(
        "--min-geometry-update-improvement",
        type=float,
        default=0.005,
        help=(
            "Required relative target-ray gain from a temporary projection of "
            "attributed primitives to an independent real-surface plane before "
            "they can enter G_update."
        ),
    )
    parser.add_argument("--opacity-factor", type=float, default=0.05)
    parser.add_argument("--max-update-per-cluster", type=int, default=160)
    parser.add_argument("--max-create-per-cluster", type=int, default=48)
    parser.add_argument(
        "--local-refine-steps",
        type=int,
        default=0,
        help="Optional selected-row real-image refinement steps after causal approval.",
    )
    parser.add_argument("--local-clean-every", type=int, default=4)
    parser.add_argument("--local-position-lr", type=float, default=5e-4)
    parser.add_argument("--local-appearance-lr", type=float, default=2e-3)
    parser.add_argument("--local-opacity-lr", type=float, default=1e-3)
    parser.add_argument("--local-plane-weight", type=float, default=5.0)
    parser.add_argument("--local-normal-weight", type=float, default=0.10)
    parser.add_argument("--local-preserve-weight", type=float, default=0.20)
    parser.add_argument("--local-distortion-weight", type=float, default=0.01)
    parser.add_argument("--local-regularization-weight", type=float, default=1e-4)
    parser.add_argument(
        "--min-local-target-improvement",
        type=float,
        default=1e-4,
        help=(
            "Required median anomaly-MAE improvement after an applied local "
            "edit; prevents writing geometry edits that merely avoid regression."
        ),
    )
    parser.add_argument("--apply-approved", action="store_true")
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0.0 < args.opacity_factor < 1.0:
        raise ValueError("--opacity-factor must lie in (0, 1)")
    if float(args.max_cluster_diameter_factor) <= 0.0:
        raise ValueError("--max-cluster-diameter-factor must be positive")
    if float(args.min_local_target_improvement) < 0.0:
        raise ValueError("--min-local-target-improvement must be non-negative")
    if float(args.min_geometry_update_improvement) < 0.0:
        raise ValueError("--min-geometry-update-improvement must be non-negative")
    if int(args.auto_target_count) < 0:
        raise ValueError("--auto-target-count must be non-negative")
    if int(args.auto_target_candidate_multiplier) <= 0:
        raise ValueError("--auto-target-candidate-multiplier must be positive")
    if int(args.control_count) <= 0:
        raise ValueError("--control-count must be positive")
    if int(args.control_search) <= 0:
        raise ValueError("--control-search must be positive")
    if int(args.control_temporal_radius) < 0:
        raise ValueError("--control-temporal-radius must be non-negative")
    if int(args.min_visible_clean_controls) <= 0:
        raise ValueError("--min-visible-clean-controls must be positive")
    if int(args.min_visible_clean_controls) > int(args.control_count):
        raise ValueError("--min-visible-clean-controls cannot exceed --control-count")
    if args.apply_approved and args.control_policy != "real_camera_envelope_v1":
        raise ValueError(
            "--apply-approved requires --control-policy real_camera_envelope_v1; "
            "legacy control ordering is diagnostic-only historical reproduction"
        )
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.replace_output:
        raise FileExistsError(f"Output is not empty: {output}; use --replace-output")
    if output.exists() and args.replace_output:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    source = args.source_path.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    mask_pickle = args.mask_pickle.expanduser().resolve()
    if args.chart_scene is not None:
        chart_scene = args.chart_scene.expanduser().resolve()
        if not (chart_scene / "charts_data.npz").is_file():
            raise FileNotFoundError(f"--chart-scene lacks charts_data.npz: {chart_scene}")
    elif args.use_matcha_chart_surface:
        chart_scene = _infer_chart_scene(model_path)
    else:
        chart_scene = None
    if args.sfm_sparse_path is not None:
        sfm_sparse_path = args.sfm_sparse_path.expanduser().resolve()
        required_sfm = ("cameras.bin", "images.bin", "points3D.bin")
        if not all((sfm_sparse_path / name).is_file() for name in required_sfm):
            raise FileNotFoundError(f"--sfm-sparse-path lacks required COLMAP files: {sfm_sparse_path}")
    elif args.use_original_sfm_tracks:
        sfm_sparse_path = _infer_sfm_sparse(mask_pickle)
    else:
        sfm_sparse_path = None
    frozen_ply = model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    frozen_checkpoint_sha256 = _sha256_file(frozen_ply)
    gaussians = GaussianModel(3)
    if not frozen_ply.is_file():
        raise FileNotFoundError(f"Frozen checkpoint PLY does not exist: {frozen_ply}")
    print(f"Loading frozen Gaussian model at iteration {args.iteration}", flush=True)
    gaussians.load_ply(str(frozen_ply))
    cameras = _LazyCameraStore.from_colmap(
        source,
        requested_resolution=int(args.resolution),
        data_device="cpu",
    )
    geometries = cameras.geometries
    # Resolve target names before deserializing the historical 1.8 GB semantic
    # pickle.  Apart from making configuration failures immediate, this keeps
    # the expensive mask path out of camera/metric bookkeeping.
    lookup = CambridgeMaskLookup(source, mask_pickle, args.semantic_mask_indices)
    static_masks = _LazyStaticMasks(lookup, geometries)
    metric_quality = _read_metric_quality(args.metrics_json)
    if args.full_train_prescan_report is not None:
        if not args.full_train_prescan:
            raise ValueError("--full-train-prescan-report cannot be combined with --no-full-train-prescan")
        full_train_prescan = _reuse_full_train_anomaly_prescan(
            report_path=args.full_train_prescan_report,
            metrics_path=args.metrics_json,
            model_path=model_path,
            cameras=cameras,
        )
    elif args.full_train_prescan:
        full_train_prescan = _full_train_anomaly_prescan(
            metrics_path=args.metrics_json,
            model_path=model_path,
            cameras=cameras,
            geometries=geometries,
            lookup=lookup,
            residual_threshold=args.residual_threshold,
            score_threshold=args.score_threshold,
            min_component_pixels=args.min_component_pixels,
            output=output,
        )
    else:
        full_train_prescan = {
            "status": "disabled_by_user",
            "policy": "all_real_train_views_rgb_structure_prescan_before_causal_attribution",
            "expected_full_train_view_count": len(cameras),
            "processed_view_count": 0,
            "selected_target_names": [],
        }
    prescan_ranked_names = list(full_train_prescan.get("ranked_target_names", []))
    camera_features = _view_features(geometries)
    if args.target_image_names_file is not None:
        target_names = _choose_targets(
            args.target_image_names_file,
            cameras,
            metric_quality,
            args.auto_target_count,
            prescan_ranked_names=prescan_ranked_names,
        )
        target_selection = {
            "mode": "explicit_manual_target_file",
            "manual_target_file": str(args.target_image_names_file),
            "selection_is_causal_triage_not_geometry_evidence": True,
            "rows": [
                {"selection_step": int(index + 1), "name": name}
                for index, name in enumerate(target_names)
            ],
        }
    elif args.auto_target_selection == "joint_diverse":
        target_names, target_selection_rows = select_diverse_full_train_anomaly_targets(
            full_train_prescan.get("records", []),
            camera_features,
            count=int(args.auto_target_count),
            candidate_multiplier=int(args.auto_target_candidate_multiplier),
        )
        if target_names:
            target_selection = {
                "mode": "all_train_residual_priority_pose_coverage_sequence_novelty_greedy_v1",
                "candidate_multiplier": int(args.auto_target_candidate_multiplier),
                "selection_is_causal_triage_not_geometry_evidence": True,
                "rows": target_selection_rows,
            }
        else:
            # If the caller intentionally disabled the prescan or supplied a
            # cache without usable anomaly records, retain the existing metric
            # fallback rather than silently exiting with no diagnostic views.
            target_names = _choose_targets(
                None,
                cameras,
                metric_quality,
                args.auto_target_count,
                prescan_ranked_names=prescan_ranked_names,
            )
            target_selection = {
                "mode": "metric_quality_fallback_no_usable_prescan_records",
                "selection_is_causal_triage_not_geometry_evidence": True,
                "rows": [
                    {"selection_step": int(index + 1), "name": name}
                    for index, name in enumerate(target_names)
                ],
            }
    else:
        target_names = _choose_targets(
            None,
            cameras,
            metric_quality,
            args.auto_target_count,
            prescan_ranked_names=prescan_ranked_names,
        )
        target_selection = {
            "mode": "all_train_prescan_priority_v1",
            "selection_is_causal_triage_not_geometry_evidence": True,
            "rows": [
                {"selection_step": int(index + 1), "name": name}
                for index, name in enumerate(target_names)
            ],
        }
    full_train_prescan["selected_target_names"] = target_names
    full_train_prescan["target_selection"] = target_selection
    # Rewrite after final target selection so the all-view audit records the
    # exact hand-off into the expensive causal stage.  A manual target list
    # remains an explicit experiment override, never a hidden selection.
    if args.full_train_prescan:
        _write_json(output / "full_train_anomaly_prescan.json", full_train_prescan)
    if args.apply_approved and full_train_prescan.get("status") != "completed":
        raise RuntimeError(
            "Refusing to apply a causal edit before the required all-real-training-view "
            "prescan completes; inspect full_train_anomaly_prescan.json or rerun with "
            "a valid frozen-model --metrics-json."
        )
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, depth_ratio=0.0)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=gaussians.get_xyz.device,
    )
    manifest = {
        "version": 7,
        "mode": "post_reconstruction_causal_2dgs_repair",
        "attribution_backend": "masked_gradient_opacity_plus_xyz__counterfactual_required",
        "source_path": str(source),
        "model_path": str(model_path),
        "iteration": int(args.iteration),
        "frozen_checkpoint": {
            "ply_path": str(frozen_ply),
            "sha256": frozen_checkpoint_sha256,
            "source_is_never_modified": True,
        },
        "full_train_view_count": len(cameras),
        "full_train_anomaly_prescan": {
            "status": full_train_prescan.get("status"),
            "expected_full_train_view_count": full_train_prescan.get("expected_full_train_view_count"),
            "processed_view_count": full_train_prescan.get("processed_view_count"),
            "skipped_view_count": full_train_prescan.get("skipped_view_count", 0),
            "report_path": str(output / "full_train_anomaly_prescan.json")
            if args.full_train_prescan
            else None,
            "reused_from": full_train_prescan.get("reused_from"),
            "manual_target_override": args.target_image_names_file is not None,
        },
        "target_view_names": target_names,
        "target_selection": target_selection,
        "matcha_chart_surface_scene": None if chart_scene is None else str(chart_scene),
        "matcha_chart_surface_policy": (
            "coordinate_audit + target_ray_multichart_support + plane_gate; "
            "fallback=real_multiview_feature_tracks"
            if args.use_matcha_chart_surface
            else "disabled"
        ),
        "original_sfm_track_scene": None if sfm_sparse_path is None else str(sfm_sparse_path),
        "original_sfm_track_policy": (
            "target anomaly rays + target and two static real support observations + "
            "2DGS-camera coordinate reprojection audit; fallback=target_anchored_sift"
            if args.use_original_sfm_tracks
            else "disabled"
        ),
        "semantic_mask_indices": list(args.semantic_mask_indices),
        "see3d_policy": "M_gen=surface_known_footprint AND no_real_rgb_support AND no_reliable_model_appearance; appearance_opacity_only; geometry_frozen",
        "local_edit_policy": (
            "G_remove=counterfactual_verified; "
            "G_update=attributed_and_real_surface_bounded_and_plane_projection_counterfactual_verified_and_post_edit_positive_target_gain; "
            "G_create=real_track_supported_surface_only; selected_row_optimizer_only"
        ),
        "causal_cluster_policy": {
            "max_component_diameter_factor": float(args.max_cluster_diameter_factor),
            "single_linkage_spatial_chain_split_before_counterfactual": True,
            "real_camera_counterfactual_envelope": {
                "policy": args.control_policy,
                "requested_control_count": int(args.control_count),
                "control_search": int(args.control_search),
                "temporal_radius_frames": int(args.control_temporal_radius),
                "minimum_visible_control_count": int(args.min_visible_clean_controls),
                "quality_is_not_a_filter": bool(args.control_policy == "real_camera_envelope_v1"),
            },
        },
        "post_edit_acceptance_policy": {
            "min_local_target_improvement": float(args.min_local_target_improvement),
        },
        "geometry_update_counterfactual_policy": {
            "method": "temporary_projection_to_independent_chart_track_plane",
            "min_target_relative_improvement": float(args.min_geometry_update_improvement),
            "max_clean_mae_increase": float(args.max_clean_mae_increase),
            "min_visible_clean_controls": 1,
        },
    }
    _write_json(output / "run_manifest.json", manifest)

    attribution_records = []
    contexts: dict[str, TargetContext] = {}
    for ordinal, name in enumerate(target_names, start=1):
        camera = cameras[name]
        with torch.no_grad():
            package = render(camera, gaussians, pipe, background)
            baseline_render = package["render"].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
            alpha = package["rend_alpha"][0].detach().cpu().numpy().astype(np.float32)
            depth = package["surf_depth"][0].detach().cpu().numpy().astype(np.float32)
            expected_depth = package["rend_depth"][0].detach().cpu().numpy().astype(np.float32)
            # Older frozen environments may not yet expose the median map.
            # Falling back to expected depth leaves the ambiguity term at zero
            # rather than changing causal decisions silently.
            median_depth = package.get("rend_depth_median", package["rend_depth"])[0].detach().cpu().numpy().astype(np.float32)
            distortion = package["rend_dist"][0].detach().cpu().numpy().astype(np.float32)
        gt = _camera_gt(camera)
        anomaly = build_anomaly_rays(
            baseline_render,
            gt,
            alpha,
            depth,
            static_masks[name],
            distortion=distortion,
            expected_depth=expected_depth,
            median_depth=median_depth,
            residual_threshold=args.residual_threshold,
            score_threshold=args.score_threshold,
            min_component_pixels=args.min_component_pixels,
        )
        visual = output / "anomaly_rays" / f"{ordinal:03d}_{_safe_name(name)}"
        _save_rgb(visual / "render.png", baseline_render)
        _save_rgb(visual / "gt.png", gt)
        _save_rgb(visual / "affine_render.png", anomaly["corrected_render"])
        _save_mask(visual / "static_mask.png", static_masks[name])
        _save_mask(visual / "anomaly_rays.png", anomaly["mask"])
        _save_heatmap(visual / "anomaly_score.png", anomaly["score"])
        _save_heatmap(visual / "depth_layer_ambiguity.png", anomaly["depth_layer_ambiguity"])
        if not np.any(anomaly["mask"]):
            attribution = {"primitive_ids": np.empty(0, dtype=np.int64), "score": np.empty(0), "opacity_benefit": np.empty(0), "position_gradient": np.empty(0), "screen_radius": np.empty(0), "positive_candidate_count": 0}
        else:
            for parameter in (gaussians._xyz, gaussians._features_dc, gaussians._features_rest, gaussians._scaling, gaussians._rotation, gaussians._opacity):
                parameter.grad = None
            package = render(camera, gaussians, pipe, background)
            corrected = affine_correct_torch(package["render"], anomaly["affine"])
            target_tensor = _to_tensor(gt.transpose(2, 0, 1), gaussians.get_xyz.device)
            mask_tensor = _to_tensor(anomaly["mask"], gaussians.get_xyz.device).bool()
            score_tensor = _to_tensor(anomaly["score"], gaussians.get_xyz.device)
            loss = masked_charbonnier(corrected, target_tensor, mask_tensor, score_tensor)
            loss.backward()
            attribution = gradient_primitive_attribution(
                gaussians,
                package,
                max_candidates=args.max_candidates_per_view,
            )
            attribution["loss"] = float(loss.detach().item())
            for parameter in (gaussians._xyz, gaussians._features_dc, gaussians._features_rest, gaussians._scaling, gaussians._rotation, gaussians._opacity):
                parameter.grad = None
        np.savez_compressed(visual / "gradient_attribution.npz", **{key: value for key, value in attribution.items() if isinstance(value, np.ndarray)})
        attribution["per_view_score"] = {name: float(np.asarray(attribution["score"]).sum())}
        context = TargetContext(
            name=name,
            camera=camera,
            geometry=geometries[name],
            gt=gt,
            static_mask=static_masks[name],
            anomaly_mask=anomaly["mask"],
            score=anomaly["score"],
            rgb_error=anomaly["rgb_error"],
            affine=anomaly["affine"],
            baseline_render=baseline_render,
            baseline_alpha=alpha,
            baseline_depth=depth,
            baseline_expected_depth=expected_depth,
            baseline_median_depth=median_depth,
            baseline_distortion=distortion,
            components=anomaly["components"],
            attribution=attribution,
        )
        contexts[name] = context
        attribution_records.append((name, attribution))
        _write_json(visual / "anomaly_report.json", {"name": name, "summary": anomaly["summary"], "components": anomaly["components"], "attribution": attribution})
        print(f"[anomaly] {ordinal}/{len(target_names)} {name}: rays={anomaly['summary']['anomaly_ray_fraction']:.4f}, candidates={attribution['positive_candidate_count']}", flush=True)

    evidence = aggregate_attribution(attribution_records)
    with torch.no_grad():
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        normals = get_gaussian_normal(gaussians.get_rotation, gaussians.get_scaling).detach().cpu().numpy()
        scales = gaussians.get_scaling.detach().cpu().numpy()
    clusters, clustering = cluster_attributed_primitives(
        evidence,
        xyz,
        normals,
        scales,
        max_component_diameter_factor=args.max_cluster_diameter_factor,
    )
    clusters = clusters[: int(args.max_clusters)]
    # Attach exact per-view attribution score to each target context so the
    # counterfactual target ordering is deterministic and inspectable.
    for cluster in clusters:
        scores: dict[str, float] = {}
        for primitive_id in np.asarray(cluster["primitive_ids"], dtype=np.int64):
            item = evidence[int(primitive_id)]
            for name, score in item.per_view_score.items():
                scores[name] = scores.get(name, 0.0) + float(score)
        cluster["per_view_score"] = scores
    _write_json(output / "clustering.json", {"diagnostics": clustering, "clusters": clusters})
    features = camera_features
    pre_roi_virtual_diagnostics = (
        _run_pre_roi_virtual_diagnostics(
            contexts,
            cameras,
            geometries,
            features,
            metric_quality,
            gaussians,
            pipe,
            background,
            output,
            list(args.virtual_probe_fractions),
        )
        if args.pre_roi_virtual_diagnostics
        else {
            "mode": "disabled",
            "virtual_views_are_not_geometry_evidence_or_supervision": True,
            "pair_count": 0,
            "pairs": [],
        }
    )
    counterfactuals = []
    for cluster in clusters:
        report = _counterfactual_cluster(
            cluster,
            contexts,
            cameras,
            static_masks,
            geometries,
            features,
            metric_quality,
            gaussians,
            pipe,
            background,
            control_count=args.control_count,
            control_search=args.control_search,
            control_policy=args.control_policy,
            control_temporal_radius=args.control_temporal_radius,
            min_visible_clean_controls=args.min_visible_clean_controls,
            min_bad_improvement=args.min_bad_improvement,
            max_clean_increase=args.max_clean_mae_increase,
            visuals_root=output / "counterfactuals",
        )
        counterfactuals.append(report)
        print(f"[counterfactual] c{report.get('cluster_id')} {report.get('classification')}: bad={report.get('median_bad_relative_improvement')}, clean={report.get('worst_clean_mae_increase')}, accepted={report.get('accepted')}", flush=True)

    surface_evidence = {}
    for report in counterfactuals:
        if report.get("classification") not in {"foreground_plus_hole", "missing_surface", "surface_misalignment_or_appearance"}:
            continue
        geometry = _recover_surface_evidence(
            report,
            contexts,
            cameras,
            static_masks,
            geometries,
            output,
            chart_scene=chart_scene if args.use_matcha_chart_surface else None,
            sfm_sparse_path=sfm_sparse_path if args.use_original_sfm_tracks else None,
        )
        surface_evidence[str(report["cluster_id"])] = geometry
    geometry_projection_counterfactuals: dict[str, dict[str, Any]] = {}
    for report in counterfactuals:
        cluster_id = str(report.get("cluster_id"))
        surface = surface_evidence.get(cluster_id, {})
        if str(report.get("classification")) != "surface_misalignment_or_appearance":
            continue
        trial = _plane_projection_counterfactual(
            report,
            surface,
            contexts,
            cameras,
            static_masks,
            gaussians,
            pipe,
            background,
            max_update_per_cluster=args.max_update_per_cluster,
            min_target_improvement=args.min_geometry_update_improvement,
            max_clean_increase=args.max_clean_mae_increase,
            visuals_root=output / "geometry_counterfactuals",
        )
        geometry_projection_counterfactuals[cluster_id] = trial
        print(
            "[plane-projection] "
            f"c{report.get('cluster_id')} target={trial.get('median_target_relative_improvement')}, "
            f"clean={trial.get('worst_clean_mae_increase')}, accepted={trial.get('accepted')}",
            flush=True,
        )
    _write_json(
        output / "geometry_projection_counterfactuals.json",
        geometry_projection_counterfactuals,
    )
    roi_projection_audit = _save_roi_projection_audit(
        surface_evidence, contexts, cameras, geometries, gaussians, pipe, background, output
    )
    virtual_diagnostics = _run_virtual_diagnostics(
        surface_evidence,
        contexts,
        cameras,
        geometries,
        gaussians,
        pipe,
        background,
        output,
        list(args.virtual_probe_fractions),
    )

    # A generated pixel is allowed only in the explicitly computed residual
    # M_surface & ~M_real & ~M_model.  For observed train surfaces M_real is
    # deliberately non-empty, which makes the correct outcome "no See3D".
    see3d_requests = []
    for cluster_id, geometry in surface_evidence.items():
        target_name = geometry.get("target_name")
        if target_name in contexts:
            context = contexts[target_name]
            if not geometry.get("roi_path"):
                continue
            roi = ArtifactROI3D.load(Path(geometry["roi_path"]))
            surface_mask, real_mask, model_mask, generation_mask = _strict_see3d_residual_masks(roi, context)
            surface_pixels = max(int(surface_mask.sum()), 1)
            real_support = float((real_mask & surface_mask).sum() / surface_pixels)
            reliable_model = float((model_mask & surface_mask).sum() / surface_pixels)
            gate = see3d_gate(
                surface_known=bool(geometry.get("surface_known")),
                real_rgb_support_fraction=real_support,
                reliable_model_fraction=reliable_model,
            )
            mask_root = output / "see3d_requests" / f"cluster_{int(cluster_id):03d}_{_safe_name(target_name)}"
            _save_mask(mask_root / "M_surface.png", surface_mask)
            _save_mask(mask_root / "M_real.png", real_mask)
            _save_mask(mask_root / "M_model.png", model_mask)
            _save_mask(mask_root / "M_gen.png", generation_mask)
            corrected = np.clip(
                context.baseline_render * context.affine[None, None, :, 0]
                + context.affine[None, None, :, 1],
                0.0,
                1.0,
            )
            condition = np.zeros_like(corrected)
            condition[real_mask] = context.gt[real_mask]
            condition[model_mask & ~real_mask] = corrected[model_mask & ~real_mask]
            _save_rgb(mask_root / "condition_real_then_model.png", condition)
            geometry["see3d_gate"] = gate
            geometry["see3d_masks"] = {
                "M_surface_path": str(mask_root / "M_surface.png"),
                "M_real_path": str(mask_root / "M_real.png"),
                "M_model_path": str(mask_root / "M_model.png"),
                "M_gen_path": str(mask_root / "M_gen.png"),
                "surface_fraction": float(surface_mask.mean()),
                "generation_fraction": float(generation_mask.mean()),
            }
            if gate["eligible"] and np.any(generation_mask):
                see3d_requests.append(
                    {
                        "cluster_id": int(cluster_id),
                        "target_name": target_name,
                        "target_camera": context.geometry.as_json(),
                        "target_mask_path": str(mask_root / "M_gen.png"),
                        "condition_path": str(mask_root / "condition_real_then_model.png"),
                        "roi_path": geometry.get("roi_path"),
                        "real_support_views": geometry.get("support_names", []),
                        "gate": gate,
                        "geometry_frozen": True,
                        "allowed_update_parameters": ["appearance", "opacity"],
                    }
                )
    _write_json(
        output / "see3d_requests.json",
        {
            "requests": see3d_requests,
            "surface_evidence": surface_evidence,
            "policy": "M_gen = M_surface AND NOT M_real AND NOT M_model; geometry_frozen",
        },
    )

    edit_plan = _build_local_edit_plan(
        counterfactuals,
        surface_evidence,
        geometry_projection_counterfactuals,
        gaussians,
        cameras,
        static_masks,
        geometries,
        max_update_per_cluster=args.max_update_per_cluster,
        max_create_per_cluster=args.max_create_per_cluster,
    )
    causal_sets_root = output / "causal_sets"
    causal_sets_root.mkdir(parents=True, exist_ok=True)
    np.save(causal_sets_root / "G_remove.npy", edit_plan["G_remove"])
    np.save(causal_sets_root / "G_update.npy", edit_plan["G_update"])
    np.savez_compressed(
        causal_sets_root / "G_create_seeds.npz",
        points=edit_plan["G_create_points"],
        colors=edit_plan["G_create_colors"],
        scales=edit_plan["G_create_scales"],
        rotations=edit_plan["G_create_rotations"],
        real_support_count=edit_plan["G_create_real_support_count"],
    )
    _write_json(
        causal_sets_root / "local_edit_plan.json",
        {
            "G_remove": edit_plan["G_remove"],
            "G_update": edit_plan["G_update"],
            "G_create_seed_count": int(len(edit_plan["G_create_points"])),
            "G_create_seed_file": str(causal_sets_root / "G_create_seeds.npz"),
            "plane_groups": edit_plan["plane_groups"],
            "audit_groups": edit_plan["audit_groups"],
        },
    )

    accepted = [item for item in counterfactuals if item.get("accepted")]
    remove_ids = np.asarray(edit_plan["G_remove"], dtype=np.int64)
    report = {
        **manifest,
        "anomaly_views": {
            name: {"ray_fraction": float(context.anomaly_mask.mean()), "components": context.components, "positive_gradient_candidates": int(context.attribution["positive_candidate_count"])}
            for name, context in contexts.items()
        },
        "clustering": clustering,
        "pre_roi_virtual_diagnostics": pre_roi_virtual_diagnostics,
        "counterfactual_clusters": counterfactuals,
        "surface_evidence": surface_evidence,
        "geometry_projection_counterfactuals": geometry_projection_counterfactuals,
        "roi_projection_audit": roi_projection_audit,
        "virtual_diagnostics": virtual_diagnostics,
        "sets": {
            "G_remove": remove_ids,
            "G_update": edit_plan["G_update"],
            "G_create": {
                "seed_count": int(len(edit_plan["G_create_points"])),
                "seed_file": str(causal_sets_root / "G_create_seeds.npz"),
                "real_rgb_support_count": edit_plan["G_create_real_support_count"],
            },
        },
        "edit_plan": {
            "local_edit_plan_path": str(causal_sets_root / "local_edit_plan.json"),
            "plane_groups": edit_plan["plane_groups"],
            "audit_groups": edit_plan["audit_groups"],
        },
        "accepted_culprit_cluster_count": len(accepted),
        "accepted_remove_primitive_count": int(len(remove_ids)),
        "selected_update_primitive_count": int(len(edit_plan["G_update"])),
        "selected_create_primitive_count": int(len(edit_plan["G_create_points"])),
        "see3d_request_count": len(see3d_requests),
    }

    has_edit = bool(len(remove_ids) or len(edit_plan["G_update"]) or len(edit_plan["G_create_points"]))
    control_names = sorted(
        {
            item["name"]
            for report_item in counterfactuals
            for item in report_item.get("clean_controls", [])
            if item["name"] in cameras
        }
    )
    report["visible_clean_control_count_for_edit"] = int(len(control_names))
    accepted_causal_target_names = _accepted_causal_target_names(counterfactuals, contexts)
    report["accepted_causal_target_names"] = accepted_causal_target_names
    if args.apply_approved and has_edit and control_names and accepted_causal_target_names:
        before_validation = _validate_edit(
            gaussians,
            contexts,
            cameras,
            static_masks,
            pipe,
            background,
            control_names=control_names,
            target_names=accepted_causal_target_names,
        )
        validation_control_affines = {
            item["name"]: np.asarray(item["affine"], dtype=np.float32)
            for item in before_validation["clean_controls"]
        }
        edit_result = _local_constrained_edit(
            edit_plan,
            contexts,
            cameras,
            static_masks,
            gaussians,
            pipe,
            background,
            control_names=control_names,
            steps=args.local_refine_steps,
            clean_every=args.local_clean_every,
            position_lr=args.local_position_lr,
            appearance_lr=args.local_appearance_lr,
            opacity_lr=args.local_opacity_lr,
            plane_weight=args.local_plane_weight,
            normal_weight=args.local_normal_weight,
            preserve_weight=args.local_preserve_weight,
            distortion_weight=args.local_distortion_weight,
            regularization_weight=args.local_regularization_weight,
            initial_opacity=args.opacity_factor,
        )
        post_edit_virtual_diagnostics = _run_virtual_diagnostics(
            surface_evidence,
            contexts,
            cameras,
            geometries,
            gaussians,
            pipe,
            background,
            output / "post_edit_validation",
            list(args.virtual_probe_fractions),
        )
        virtual_stability_gate = _virtual_stability_nonregression(
            virtual_diagnostics,
            post_edit_virtual_diagnostics,
        )
        after_validation = _validate_edit(
            gaussians,
            contexts,
            cameras,
            static_masks,
            pipe,
            background,
            control_names=control_names,
            target_names=accepted_causal_target_names,
            control_affines=validation_control_affines,
        )
        before_target = {item["name"]: item for item in before_validation["targets"]}
        target_improvements = [
            before_target[item["name"]]["anomaly_mae"] - item["anomaly_mae"]
            for item in after_validation["targets"]
            if item["name"] in before_target
        ]
        before_control = {item["name"]: item for item in before_validation["clean_controls"]}
        clean_increases = [
            item["static_mae"] - before_control[item["name"]]["static_mae"]
            for item in after_validation["clean_controls"]
            if item["name"] in before_control
        ]
        validation_gate = {
            "target_scope": before_validation["target_scope"],
            "median_target_anomaly_mae_improvement": float(np.median(target_improvements)) if target_improvements else 0.0,
            "min_local_target_improvement": float(args.min_local_target_improvement),
            "worst_clean_static_mae_increase": float(max(clean_increases)) if clean_increases else 0.0,
            "max_clean_mae_increase": float(args.max_clean_mae_increase),
            "frozen_rows_intact": bool(edit_result["frozen_prefix_audit"]["all_frozen_rows_bit_identical"]),
            "geometry_constraints": edit_result["geometry_constraint_audit"],
            "virtual_depth_normal_stability": virtual_stability_gate,
        }
        # Counterfactual accepted G_remove already has a strict local gain
        # test.  Plane edits additionally need a measurable real-target gain:
        # avoiding a regression alone is not evidence that a bounded geometry
        # modification repaired the diagnosed cause.
        validation_gate["passed"] = bool(
            validation_gate["median_target_anomaly_mae_improvement"]
            >= float(args.min_local_target_improvement)
            and validation_gate["worst_clean_static_mae_increase"] <= float(args.max_clean_mae_increase)
            and validation_gate["frozen_rows_intact"]
            and bool(edit_result["geometry_constraint_audit"]["passed"])
            and bool(virtual_stability_gate["passed"])
        )
        if validation_gate["passed"]:
            edited = output / "edited_model"
            ply_dir = edited / "point_cloud" / f"iteration_{args.iteration}"
            ply_dir.mkdir(parents=True, exist_ok=True)
            gaussians.save_ply(str(ply_dir / "point_cloud.ply"))
            _copy_cfg(model_path, edited)
            shutil.copy2(causal_sets_root / "G_remove.npy", edited / "G_remove.npy")
            shutil.copy2(causal_sets_root / "G_update.npy", edited / "G_update.npy")
            shutil.copy2(causal_sets_root / "G_create_seeds.npz", edited / "G_create_seeds.npz")
            report["applied_edit"] = {
                **edit_result,
                "opacity_factor": args.opacity_factor,
                "output_model": str(edited),
                "validation_before": before_validation,
                "validation_after": after_validation,
                "validation_gate": validation_gate,
                "post_edit_virtual_diagnostics": post_edit_virtual_diagnostics,
            }
        else:
            report["applied_edit"] = {
                **edit_result,
                "mode": "rejected_after_targeted_validation_not_written",
                "validation_before": before_validation,
                "validation_after": after_validation,
                "validation_gate": validation_gate,
                "post_edit_virtual_diagnostics": post_edit_virtual_diagnostics,
            }
    elif args.apply_approved and has_edit:
        report["applied_edit"] = {
            "mode": "no_op",
            "reason": (
                "no_visible_clean_control_for_causal_edit"
                if not control_names
                else "no_accepted_counterfactual_causal_targets_for_edit"
            ),
            "candidate_edit_sets": {
                "G_remove": int(len(remove_ids)),
                "G_update": int(len(edit_plan["G_update"])),
                "G_create": int(len(edit_plan["G_create_points"])),
            },
        }
    elif args.apply_approved:
        report["applied_edit"] = {"mode": "no_op", "reason": "no_counterfactual_verified_or_real_surface_supported_causal_set"}
    else:
        report["applied_edit"] = {"mode": "not_requested"}
    _write_json(output / "causal_repair_report.json", report)
    print(
        json.dumps(
            _json_ready(
                {
                    "output": output,
                    "targets": len(contexts),
                    "clusters": len(counterfactuals),
                    "accepted": len(accepted),
                    "G_remove": len(remove_ids),
                    "G_update": len(edit_plan["G_update"]),
                    "G_create": len(edit_plan["G_create_points"]),
                    "see3d_requests": len(see3d_requests),
                    "applied": bool(args.apply_approved and has_edit and report["applied_edit"].get("output_model")),
                }
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
