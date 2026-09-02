"""Strict, exact-camera evidence contract for offline MoGe3 predictions.

MoGe3 predicts metric camera-z depth, masks and normals.  Its public v3
inference helper deliberately builds a normalized pinhole camera with a
centred principal point, even when the horizontal field of view is supplied.
Cambridge cameras are fixed task inputs, so G4Splat never consumes that
point-map as calibrated geometry.  This module reconstructs the point-map
from the predicted metric depth and the exact Cambridge pixel intrinsics.

The module intentionally has no torch or MoGe dependency.  Training can read
and validate an immutable cache without importing the model environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


MOGE3_VIEW_VERSION = (
    "g4splat-moge3-view-v2-exact-k-frozen-generator-contract"
)
MOGE3_INDEX_VERSION = (
    "g4splat-moge3-index-v2-exact-k-frozen-generator-contract"
)
MOGE3_RUNTIME_CACHE_VERSION = (
    "g4splat-moge3-runtime-cache-v1-content-addressed-mmap"
)
MOGE3_OFFICIAL_REPOSITORY = "https://github.com/microsoft/MoGe"
MOGE3_AUDITED_COMMIT = "74fbce054ebed49800de42d0ad0e83495065719a"


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def horizontal_fov_degrees(*, width: int, fx: float) -> float:
    """Return the horizontal FOV of an exact pixel-space pinhole camera."""
    width = int(width)
    fx = float(fx)
    if width <= 0 or not math.isfinite(fx) or fx <= 0:
        raise ValueError(f"Invalid pinhole width/fx: width={width}, fx={fx}")
    return math.degrees(2.0 * math.atan(width / (2.0 * fx)))


def exact_pixel_intrinsics(camera: Mapping[str, Any]) -> np.ndarray:
    """Build exact pixel K and reject distorted/non-pinhole camera records."""
    model = str(camera.get("model"))
    if model not in {"SIMPLE_PINHOLE", "PINHOLE"}:
        raise ValueError(
            "MoGe3 exact-K evidence requires an already-undistorted "
            f"SIMPLE_PINHOLE/PINHOLE camera, got {model!r}"
        )
    distortion = np.asarray(camera.get("distortion", []), dtype=np.float64)
    if distortion.size and np.any(np.abs(distortion) > 1e-12):
        raise ValueError(
            "MoGe3 evidence refuses non-zero lens distortion; rectify the "
            "fixed raster and update its camera contract first"
        )
    fx, fy, cx, cy = (
        float(camera[key]) for key in ("fx", "fy", "cx", "cy")
    )
    values = np.asarray([fx, fy, cx, cy], dtype=np.float64)
    if not np.isfinite(values).all() or fx <= 0 or fy <= 0:
        raise ValueError(f"Invalid exact camera intrinsics: {values.tolist()}")
    return np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def exact_k_unproject_depth(depth_m: np.ndarray, K_px: np.ndarray) -> np.ndarray:
    """Unproject camera-z metric depth with the exact pixel-space K."""
    depth = np.asarray(depth_m, dtype=np.float32)
    K = np.asarray(K_px, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must be HxW, got {depth.shape}")
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError(f"K_px must be a finite 3x3 matrix, got {K.shape}")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("K_px focal lengths must be positive")
    rows, columns = np.indices(depth.shape, dtype=np.float32)
    x = (columns - np.float32(K[0, 2])) / np.float32(K[0, 0]) * depth
    y = (rows - np.float32(K[1, 2])) / np.float32(K[1, 1]) * depth
    return np.stack((x, y, depth), axis=-1).astype(np.float32, copy=False)


def _normalise_vectors(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(vectors, axis=-1)
    valid = np.isfinite(vectors).all(axis=-1) & np.isfinite(norm) & (norm > 1e-6)
    result = np.zeros_like(vectors, dtype=np.float32)
    result[valid] = vectors[valid] / norm[valid, None]
    return result, valid


def depth_normals_exact_k(
    depth_m: np.ndarray,
    K_px: np.ndarray,
    valid_mask: np.ndarray,
    *,
    orient_like: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate camera-space normals from exact-K points using central rays."""
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != depth.shape:
        raise ValueError("valid_mask shape does not match depth")
    points = exact_k_unproject_depth(depth, K_px)
    normals = np.zeros((*depth.shape, 3), dtype=np.float32)
    interior = np.zeros(depth.shape, dtype=bool)
    if depth.shape[0] >= 3 and depth.shape[1] >= 3:
        tangent_x = points[1:-1, 2:] - points[1:-1, :-2]
        tangent_y = points[2:, 1:-1] - points[:-2, 1:-1]
        estimate = np.cross(tangent_x, tangent_y)
        estimate, finite_normal = _normalise_vectors(estimate)
        supported = (
            valid[1:-1, 1:-1]
            & valid[1:-1, 2:]
            & valid[1:-1, :-2]
            & valid[2:, 1:-1]
            & valid[:-2, 1:-1]
            & finite_normal
        )
        normals[1:-1, 1:-1] = estimate
        interior[1:-1, 1:-1] = supported
    normals[~interior] = 0.0
    if orient_like is not None:
        reference, reference_valid = _normalise_vectors(orient_like)
        flip = (
            interior
            & reference_valid
            & ((normals * reference).sum(axis=-1) < 0)
        )
        normals[flip] *= -1.0
    return normals, interior


def refinement_stability(
    depth_per_step_m: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return log-depth std, final update magnitude and valid-step mask."""
    steps = np.asarray(depth_per_step_m, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if steps.ndim != 3 or steps.shape[1:] != valid.shape:
        raise ValueError(
            "depth_per_step_m must be SxHxW and match valid_mask, got "
            f"{steps.shape} and {valid.shape}"
        )
    if steps.shape[0] < 1:
        raise ValueError("At least one MoGe3 depth refinement output is required")
    stable_valid = (
        valid
        & np.isfinite(steps).all(axis=0)
        & (steps > 0).all(axis=0)
    )
    log_steps = np.zeros_like(steps, dtype=np.float32)
    np.log(steps, out=log_steps, where=steps > 0)
    log_std = np.zeros(valid.shape, dtype=np.float32)
    log_std[stable_valid] = np.std(
        log_steps[:, stable_valid], axis=0, dtype=np.float64
    ).astype(np.float32)
    final_delta = np.zeros(valid.shape, dtype=np.float32)
    if steps.shape[0] >= 2:
        final_delta[stable_valid] = np.abs(
            log_steps[-1, stable_valid] - log_steps[-2, stable_valid]
        )
    return log_std, final_delta, stable_valid


def make_view_payload(
    *,
    depth_m: np.ndarray,
    normal_direct_camera: np.ndarray,
    model_valid_mask: np.ndarray,
    depth_per_step_m: np.ndarray,
    K_px: np.ndarray,
    model_intrinsics_normalized: np.ndarray,
    camera_record: Mapping[str, Any],
    image_sha256: str,
    model_id: str,
    model_revision: str,
    model_checkpoint_sha256: str,
    refine_steps: int,
    resolution_level: int,
    use_fp16: bool,
) -> dict[str, np.ndarray]:
    """Construct one strict, self-describing MoGe3 view archive."""
    depth = np.asarray(depth_m, dtype=np.float32)
    normal, normal_valid = _normalise_vectors(normal_direct_camera)
    model_mask = np.asarray(model_valid_mask, dtype=bool)
    per_step = np.asarray(depth_per_step_m, dtype=np.float32)
    K = np.asarray(K_px, dtype=np.float64)
    camera = dict(camera_record["camera"])
    height, width = int(camera["height"]), int(camera["width"])
    expected_shape = (height, width)
    if depth.shape != expected_shape:
        raise ValueError(
            "MoGe3 output must retain the exact fixed-camera raster; got "
            f"{depth.shape}, expected {expected_shape}"
        )
    if normal.shape != (*expected_shape, 3):
        raise ValueError(f"MoGe3 normal has wrong shape: {normal.shape}")
    if model_mask.shape != expected_shape:
        raise ValueError(f"MoGe3 mask has wrong shape: {model_mask.shape}")
    if per_step.ndim != 3 or per_step.shape[1:] != expected_shape:
        raise ValueError(f"MoGe3 per-step depth has wrong shape: {per_step.shape}")
    if per_step.shape[0] != int(refine_steps) + 1:
        raise ValueError(
            f"Expected initial+{refine_steps} refinement depths, got "
            f"{per_step.shape[0]}"
        )
    if not np.allclose(per_step[-1], depth, rtol=2e-5, atol=2e-6):
        raise ValueError("Final MoGe3 depth differs from final per-step output")
    finite_depth = np.isfinite(depth) & (depth > 0)
    valid = model_mask & finite_depth & normal_valid
    exact_points = exact_k_unproject_depth(depth, K)
    depth_normal, depth_normal_valid = depth_normals_exact_k(
        depth, K, valid, orient_like=normal
    )
    log_std, final_delta, refinement_valid = refinement_stability(
        per_step, valid
    )
    metadata = {
        "schema_version": MOGE3_VIEW_VERSION,
        "coordinate_frame": "exact_cambridge_camera_x_right_y_down_z_forward",
        "depth_semantics": "metric_camera_z_depth",
        "pointmap_policy": "recomputed_from_depth_with_exact_pixel_K",
        "model_pointmap_is_geometry_authority": False,
        "image_name": str(camera_record["image_name"]),
        "frame_index": int(camera_record["frame_index"]),
        "camera_id": int(camera["camera_id"]),
        "image_sha256": str(image_sha256),
        "camera_record_sha256": canonical_json_sha256(camera_record),
        "model_id": str(model_id),
        "model_revision": str(model_revision),
        "model_checkpoint_sha256": str(model_checkpoint_sha256),
        "audited_moge_commit": MOGE3_AUDITED_COMMIT,
        "refine_steps": int(refine_steps),
        "resolution_level": int(resolution_level),
        "use_fp16": bool(use_fp16),
        "width": width,
        "height": height,
        "fov_x_degrees": horizontal_fov_degrees(width=width, fx=K[0, 0]),
    }
    return {
        "metadata_json": np.asarray(_canonical_json(metadata)),
        "depth_m": depth,
        "pointmap_exact_k_camera": exact_points,
        "normal_direct_camera": normal,
        "normal_depth_exact_k_camera": depth_normal,
        "valid_mask": valid.astype(np.uint8),
        "depth_normal_valid_mask": depth_normal_valid.astype(np.uint8),
        "depth_per_step_m": per_step,
        "refinement_log_depth_std": log_std,
        "refinement_final_delta_log_depth": final_delta,
        "refinement_valid_mask": refinement_valid.astype(np.uint8),
        "K_pixel_exact": K,
        "model_intrinsics_normalized": np.asarray(
            model_intrinsics_normalized, dtype=np.float32
        ),
    }


def atomic_save_view(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_view(path: Path, *, verify_exact_k: bool = True) -> dict[str, Any]:
    """Load one view fail-closed and optionally verify its stored point-map."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "metadata_json",
            "depth_m",
            "pointmap_exact_k_camera",
            "normal_direct_camera",
            "normal_depth_exact_k_camera",
            "valid_mask",
            "depth_normal_valid_mask",
            "depth_per_step_m",
            "refinement_log_depth_std",
            "refinement_final_delta_log_depth",
            "refinement_valid_mask",
            "K_pixel_exact",
            "model_intrinsics_normalized",
        }
        missing = required - set(archive.files)
        if missing:
            raise RuntimeError(
                f"MoGe3 archive {path} lacks: {', '.join(sorted(missing))}"
            )
        result = {key: np.array(archive[key], copy=True) for key in required}
    metadata = json.loads(str(result.pop("metadata_json").item()))
    if metadata.get("schema_version") != MOGE3_VIEW_VERSION:
        raise RuntimeError(
            f"Unsupported MoGe3 view schema: {metadata.get('schema_version')!r}"
        )
    depth = result["depth_m"]
    shape = (int(metadata["height"]), int(metadata["width"]))
    if depth.shape != shape or result["valid_mask"].shape != shape:
        raise RuntimeError(f"MoGe3 view raster shape does not match metadata: {path}")
    if result["pointmap_exact_k_camera"].shape != (*shape, 3):
        raise RuntimeError(f"MoGe3 exact-K point-map has wrong shape: {path}")
    if result["depth_per_step_m"].shape[1:] != shape:
        raise RuntimeError(f"MoGe3 refinement depth has wrong shape: {path}")
    if verify_exact_k:
        expected = exact_k_unproject_depth(depth, result["K_pixel_exact"])
        if not np.allclose(
            result["pointmap_exact_k_camera"],
            expected,
            rtol=2e-5,
            atol=2e-6,
            equal_nan=True,
        ):
            raise RuntimeError(
                f"MoGe3 point-map is not an exact-K unprojection: {path}"
            )
    result["metadata"] = metadata
    return result


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_index(
    *,
    view_root: Path,
    scene_contract: Mapping[str, Any],
    output: Path,
    model_id: str,
    model_revision: str,
    generator: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and content-address every exact camera view in scene order."""
    view_root = Path(view_root).resolve()
    records: dict[str, dict[str, Any]] = {}
    camera_order: list[str] = []
    for camera_record in scene_contract.get("records", []):
        stem = Path(str(camera_record["image_name"])).stem
        if stem in records:
            raise RuntimeError(f"Duplicate scene-contract image stem: {stem}")
        path = (view_root / f"{stem}.npz").resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        view = load_view(path)
        metadata = view["metadata"]
        expected_camera_hash = canonical_json_sha256(camera_record)
        if metadata.get("camera_record_sha256") != expected_camera_hash:
            raise RuntimeError(f"MoGe3 camera contract changed for {stem}")
        if metadata.get("image_sha256") != camera_record.get("image_sha256"):
            raise RuntimeError(f"MoGe3 image content changed for {stem}")
        if metadata.get("model_id") != str(model_id):
            raise RuntimeError(f"Mixed MoGe3 model ids in view cache: {stem}")
        if metadata.get("model_revision") != str(model_revision):
            raise RuntimeError(f"Mixed MoGe3 model revisions in view cache: {stem}")
        expected_generator = {
            "model_checkpoint_sha256": str(
                generator.get("checkpoint_sha256", "")
            ),
            "refine_steps": int(generator.get("refine_steps", -1)),
            "resolution_level": int(
                generator.get("resolution_level", -1)
            ),
            "use_fp16": bool(generator.get("use_fp16", False)),
            "audited_moge_commit": str(
                generator.get("moge_git_revision", "")
            ),
        }
        actual_generator = {
            key: metadata.get(key) for key in expected_generator
        }
        if actual_generator != expected_generator:
            raise RuntimeError(
                "Mixed MoGe3 generator contracts in view cache: "
                f"{stem}: {actual_generator} != {expected_generator}"
            )
        records[stem] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "image_name": str(camera_record["image_name"]),
            "frame_index": int(camera_record["frame_index"]),
            "camera_record_sha256": expected_camera_hash,
        }
        camera_order.append(stem)
    if not records:
        raise RuntimeError("Cannot build an empty MoGe3 evidence index")
    payload: dict[str, Any] = {
        "schema_version": MOGE3_INDEX_VERSION,
        "coordinate_frame": "cambridge_fixed_exact_K_camera",
        "depth_semantics": "metric_camera_z_depth",
        "pointmap_policy": "recomputed_from_depth_with_exact_pixel_K",
        "model_pointmap_is_geometry_authority": False,
        "camera_scope": "all_scene_contract_database_views",
        "scene_view_count": len(camera_order),
        "indexed_view_count": len(records),
        "camera_order": camera_order,
        "records": records,
        "model": {
            "id": str(model_id),
            "revision": str(model_revision),
            "official_repository": MOGE3_OFFICIAL_REPOSITORY,
            "audited_commit": MOGE3_AUDITED_COMMIT,
        },
        "generator": dict(generator),
    }
    payload["index_hash"] = canonical_json_sha256(payload)
    atomic_write_json(output, payload)
    return payload


def load_index(
    path: Path,
    *,
    verify_views: bool = True,
    verify_payloads: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MOGE3_INDEX_VERSION:
        raise RuntimeError(
            f"Unsupported MoGe3 index schema: {payload.get('schema_version')!r}"
        )
    unhashed = dict(payload)
    expected_hash = unhashed.pop("index_hash", None)
    if expected_hash != canonical_json_sha256(unhashed):
        raise RuntimeError("MoGe3 index hash does not match its content")
    order = list(payload.get("camera_order", []))
    records = dict(payload.get("records", {}))
    if len(order) != len(set(order)) or set(order) != set(records):
        raise RuntimeError("MoGe3 camera order and record set disagree")
    if int(payload.get("scene_view_count", -1)) != len(order):
        raise RuntimeError("MoGe3 scene view count is inconsistent")
    if int(payload.get("indexed_view_count", -1)) != len(records):
        raise RuntimeError("MoGe3 indexed view count is inconsistent")
    if verify_views:
        for stem in order:
            record = records[stem]
            view_path = Path(record["path"])
            if not view_path.is_file():
                raise FileNotFoundError(view_path)
            if view_path.stat().st_size != int(record["bytes"]):
                raise RuntimeError(f"MoGe3 view size changed: {view_path}")
            if sha256_file(view_path) != record["sha256"]:
                raise RuntimeError(f"MoGe3 view content changed: {view_path}")
            if verify_payloads:
                view = load_view(view_path)
                if view["metadata"]["camera_record_sha256"] != record[
                    "camera_record_sha256"
                ]:
                    raise RuntimeError(
                        f"MoGe3 indexed camera hash changed: {stem}"
                    )
    return payload


def load_runtime_cache(
    path: Path,
    *,
    expected_source_index_sha256: str | None = None,
    verify_arrays: bool = True,
) -> dict[str, Any]:
    """Open the compact training raster cache without loading it into RAM."""
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MOGE3_RUNTIME_CACHE_VERSION:
        raise RuntimeError(
            "Unsupported MoGe3 runtime cache schema: "
            f"{payload.get('schema_version')!r}"
        )
    unhashed = dict(payload)
    expected_hash = unhashed.pop("index_hash", None)
    if expected_hash != canonical_json_sha256(unhashed):
        raise RuntimeError("MoGe3 runtime cache index hash is invalid")
    if (
        expected_source_index_sha256 is not None
        and payload.get("source_index_sha256")
        != str(expected_source_index_sha256)
    ):
        raise RuntimeError("MoGe3 runtime cache belongs to another source index")
    order = [str(value) for value in payload.get("camera_order", [])]
    if len(order) != len(set(order)) or int(
        payload.get("camera_count", -1)
    ) != len(order):
        raise RuntimeError("MoGe3 runtime cache camera order is invalid")
    arrays: dict[str, np.ndarray] = {}
    for name, record in payload.get("arrays", {}).items():
        array_path = (path.parent / str(record["path"])).resolve()
        if not array_path.is_file():
            raise FileNotFoundError(array_path)
        if array_path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"MoGe3 runtime array size changed: {array_path}")
        if verify_arrays and sha256_file(array_path) != record["sha256"]:
            raise RuntimeError(
                f"MoGe3 runtime array content changed: {array_path}"
            )
        value = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != list(record["shape"]) or str(value.dtype) != str(
            record["dtype"]
        ):
            raise RuntimeError(
                f"MoGe3 runtime array contract changed: {array_path}"
            )
        arrays[str(name)] = value
    required = {
        "depth_m",
        "normal_direct_camera",
        "normal_depth_exact_k_camera",
        "valid_mask",
        "depth_normal_valid_mask",
        "refinement_log_depth_std",
        "refinement_final_delta_log_depth",
    }
    if set(arrays) != required:
        raise RuntimeError(
            "MoGe3 runtime cache fields disagree: "
            f"{sorted(set(arrays) ^ required)}"
        )
    shape = tuple(int(value) for value in payload.get("raster_shape", []))
    if len(shape) != 2 or any(
        array.shape[0] != len(order) for array in arrays.values()
    ):
        raise RuntimeError("MoGe3 runtime cache array count/shape is invalid")
    return {
        "path": path,
        "camera_order": order,
        "camera_index": {stem: index for index, stem in enumerate(order)},
        "raster_shape": shape,
        "arrays": arrays,
        "index_hash": expected_hash,
        "source_index_sha256": payload["source_index_sha256"],
    }
