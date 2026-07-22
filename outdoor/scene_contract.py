"""Immutable camera/data contracts for Cambridge outdoor reconstructions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MAST3R_ROOT = REPO_ROOT / "mast3r"
if str(MAST3R_ROOT) not in sys.path:
    sys.path.insert(0, str(MAST3R_ROOT))

from colmap.read_write_model import read_cameras_binary, read_images_binary  # noqa: E402
from view_quality_control.poses import qvec_to_rotmat  # noqa: E402


SCENE_CONTRACT_VERSION = "cambridge-outdoor-scene-contract-v1"


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return a stable content hash without materialising an image in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _camera_intrinsics(camera: Any) -> dict[str, Any]:
    params = [float(value) for value in np.asarray(camera.params).reshape(-1)]
    model = str(camera.model)
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        fx = fy = params[0]
        cx, cy = params[1:3]
        distortion = params[3:]
    elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"}:
        fx, fy, cx, cy = params[:4]
        distortion = params[4:]
    else:
        # The matrix still records the exact calibrated model/parameters.  A
        # downstream module must opt in before approximating an unfamiliar
        # model instead of silently treating it as a pinhole camera.
        fx = fy = cx = cy = None
        distortion = params
    return {
        "camera_id": int(camera.id),
        "model": model,
        "width": int(camera.width),
        "height": int(camera.height),
        "params": params,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "distortion": distortion,
    }


def _pose_matrices(qvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    qvec = np.asarray(qvec, dtype=np.float64)
    norm = float(np.linalg.norm(qvec))
    if norm <= 1e-12:
        raise ValueError("COLMAP camera has a zero quaternion")
    normalized_qvec = qvec / norm
    rotation = qvec_to_rotmat(normalized_qvec)
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = np.asarray(tvec, dtype=np.float64)
    camera_to_world = np.linalg.inv(world_to_camera)
    return world_to_camera, camera_to_world, norm


def _scene_scale(camera_centers: np.ndarray) -> dict[str, float]:
    if len(camera_centers) < 2:
        return {
            "nearest_camera_distance_median": 1.0,
            "camera_extent_p90": 1.0,
            "normalization_scale": 1.0,
        }
    distances = np.linalg.norm(
        camera_centers[:, None, :] - camera_centers[None, :, :], axis=2
    )
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    finite_pairs = distances[np.isfinite(distances)]
    nearest_median = float(np.median(nearest))
    extent_p90 = float(np.percentile(finite_pairs, 90)) if finite_pairs.size else nearest_median
    scale = max(nearest_median, extent_p90 * 0.05, 1e-6)
    return {
        "nearest_camera_distance_median": nearest_median,
        "camera_extent_p90": extent_p90,
        "normalization_scale": float(scale),
    }


def _sequence_id(name: str) -> str:
    stem = Path(name).stem
    return stem.split("__", 1)[0] if "__" in stem else "default"


def build_scene_contract(
    dataset: Path,
    output: Path,
    *,
    mask_pickle: Path | None = None,
    split: str = "database_train",
    image_hashes: bool = True,
) -> dict[str, Any]:
    """Create the sole immutable camera/data contract consumed by the mainline.

    The function intentionally records both ``T_world_to_camera`` and
    ``T_camera_to_world``.  That makes coordinate mistakes observable at the
    hand-off boundaries between COLMAP, MASt3R, plane fitting and rendering.
    """
    dataset = Path(dataset).resolve()
    output = Path(output)
    sparse = dataset / "sparse" / "0"
    images_path = sparse / "images.bin"
    cameras_path = sparse / "cameras.bin"
    mapping_path = dataset / "name_mapping.json"
    image_root = dataset / "images"
    required = [images_path, cameras_path, mapping_path, image_root]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing scene-contract input(s): " + ", ".join(map(str, missing)))

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    cameras = read_cameras_binary(str(cameras_path))
    colmap_images = sorted(read_images_binary(str(images_path)).values(), key=lambda image: image.name)
    if not colmap_images:
        raise RuntimeError(f"No calibrated images found in {images_path}")

    records: list[dict[str, Any]] = []
    centers: list[np.ndarray] = []
    non_unit_quaternions: list[str] = []
    for frame_index, image in enumerate(colmap_images):
        name = Path(str(image.name)).name
        image_path = image_root / name
        if not image_path.is_file():
            raise FileNotFoundError(f"COLMAP camera references a missing image: {image_path}")
        if image.camera_id not in cameras:
            raise RuntimeError(f"Image {name} references unknown camera id {image.camera_id}")
        world_to_camera, camera_to_world, quaternion_norm = _pose_matrices(image.qvec, image.tvec)
        if not np.isclose(quaternion_norm, 1.0, rtol=0.0, atol=1e-5):
            non_unit_quaternions.append(name)
        centers.append(camera_to_world[:3, 3])
        records.append(
            {
                "frame_index": frame_index,
                "image_name": name,
                "source_image_name": mapping.get(name, name),
                "camera": _camera_intrinsics(cameras[image.camera_id]),
                "qvec_input_norm": quaternion_norm,
                "T_world_to_camera": world_to_camera.tolist(),
                "T_camera_to_world": camera_to_world.tolist(),
                "camera_center_world": camera_to_world[:3, 3].tolist(),
                "sequence_id": _sequence_id(name),
                "split": split,
                "image_sha256": sha256_file(image_path) if image_hashes else None,
            }
        )

    centers_array = np.stack(centers, axis=0)
    payload: dict[str, Any] = {
        "schema_version": SCENE_CONTRACT_VERSION,
        "dataset": str(dataset),
        "split": split,
        "camera_policy": "calibrated_fixed",
        "intrinsics_policy": "dataset_camera_id",
        "image_count": len(records),
        "scene_scale": _scene_scale(centers_array),
        "records": records,
        "input_hashes": {
            "images_bin": sha256_file(images_path),
            "cameras_bin": sha256_file(cameras_path),
            "name_mapping": sha256_file(mapping_path),
            "mask_pickle": sha256_file(mask_pickle) if mask_pickle is not None else None,
        },
        "projection_checks": {
            "all_referenced_images_present": True,
            "all_camera_ids_resolved": True,
            "non_unit_input_quaternion_count": len(non_unit_quaternions),
            "non_unit_input_quaternion_examples": non_unit_quaternions[:20],
            "transform_convention": "T_world_to_camera is COLMAP qvec/tvec; T_camera_to_world is its inverse",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def validate_disjoint_contracts(database_contract: Path, query_contract: Path) -> dict[str, Any]:
    """Fail closed when a held-out trajectory leaks into map construction."""
    database = json.loads(Path(database_contract).read_text(encoding="utf-8"))
    query = json.loads(Path(query_contract).read_text(encoding="utf-8"))
    database_names = {record["source_image_name"] for record in database["records"]}
    query_names = {record["source_image_name"] for record in query["records"]}
    overlap = sorted(database_names & query_names)
    if overlap:
        raise RuntimeError(
            "Database/query camera contracts overlap; held-out evaluation would leak: "
            + ", ".join(overlap[:10])
        )
    return {
        "database_image_count": len(database_names),
        "query_image_count": len(query_names),
        "overlap_count": 0,
        "passed": True,
    }
