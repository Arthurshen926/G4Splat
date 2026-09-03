#!/usr/bin/env python3
"""Build a compact mmap cache for per-step MoGe3 training evidence."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outdoor.moge3_evidence import (  # noqa: E402
    MOGE3_RUNTIME_CACHE_VERSION,
    MOGE3_VIEW_VERSION,
    atomic_write_json,
    canonical_json_sha256,
    load_index,
    sha256_file,
)


FIELDS = {
    "depth_m": np.float32,
    "normal_direct_camera": np.float16,
    "normal_depth_exact_k_camera": np.float16,
    "valid_mask": np.uint8,
    "depth_normal_valid_mask": np.uint8,
    "refinement_log_depth_std": np.float16,
    "refinement_final_delta_log_depth": np.float16,
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moge3-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def _resize(
    value: np.ndarray, shape: tuple[int, int], *, nearest: bool
) -> np.ndarray:
    height, width = shape
    return cv2.resize(
        np.asarray(value),
        (width, height),
        interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA,
    )


def _resize_weighted(
    value: np.ndarray,
    support: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Area-resample finite measurements without bleeding invalid pixels."""
    value = np.asarray(value, dtype=np.float32)
    support = np.asarray(support, dtype=bool)
    finite = np.isfinite(value)
    if value.ndim == 3:
        valid = support[..., None] & finite
        scalar_support = support & finite.all(axis=-1)
        clean = np.where(valid, value, 0.0)
    else:
        scalar_support = support & finite
        clean = np.where(scalar_support, value, 0.0)
    numerator = _resize(clean, shape, nearest=False).astype(np.float32)
    denominator = _resize(
        scalar_support.astype(np.float32), shape, nearest=False
    ).astype(np.float32)
    if numerator.ndim == 3:
        denominator = denominator[..., None]
    return np.divide(
        numerator,
        np.maximum(denominator, 1.0e-6),
        out=np.zeros_like(numerator),
        where=denominator > 1.0e-6,
    )


def _compact_one(
    index: int,
    stem: str,
    record: dict,
    shape: tuple[int, int],
) -> tuple[int, dict[str, np.ndarray]]:
    path = Path(record["path"])
    if not path.is_file() or path.stat().st_size != int(record["bytes"]):
        raise RuntimeError(f"MoGe3 indexed view is missing or changed: {path}")
    required = {
        "metadata_json",
        "depth_m",
        "normal_direct_camera",
        "normal_depth_exact_k_camera",
        "valid_mask",
        "depth_normal_valid_mask",
        "refinement_valid_mask",
        "refinement_log_depth_std",
        "refinement_final_delta_log_depth",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise RuntimeError(f"MoGe3 view {path} lacks {sorted(missing)}")
        metadata = json.loads(str(archive["metadata_json"].item()))
        source = {
            name: np.asarray(archive[name])
            for name in required
            if name != "metadata_json"
        }
    if metadata.get("schema_version") != MOGE3_VIEW_VERSION:
        raise RuntimeError(f"Unsupported MoGe3 view schema: {path}")
    if metadata.get("camera_record_sha256") != record["camera_record_sha256"]:
        raise RuntimeError(f"MoGe3 camera contract changed: {stem}")
    source_shape = (int(metadata["height"]), int(metadata["width"]))
    if source["depth_m"].shape != source_shape:
        raise RuntimeError(f"MoGe3 depth shape changed: {path}")
    valid = source["valid_mask"].astype(bool) & source[
        "refinement_valid_mask"
    ].astype(bool)
    depth_normal_valid = valid & source[
        "depth_normal_valid_mask"
    ].astype(bool)
    result = {
        "depth_m": _resize_weighted(source["depth_m"], valid, shape),
        "normal_direct_camera": _resize_weighted(
            source["normal_direct_camera"], valid, shape
        ),
        "normal_depth_exact_k_camera": _resize_weighted(
            source["normal_depth_exact_k_camera"], depth_normal_valid, shape
        ),
        "valid_mask": _resize(
            valid.astype(np.uint8), shape, nearest=True
        ).astype(np.uint8),
        "depth_normal_valid_mask": _resize(
            depth_normal_valid.astype(np.uint8), shape, nearest=True
        ).astype(np.uint8),
        "refinement_log_depth_std": _resize_weighted(
            source["refinement_log_depth_std"], valid, shape
        ),
        "refinement_final_delta_log_depth": _resize_weighted(
            source["refinement_final_delta_log_depth"], valid, shape
        ),
    }
    for name in (
        "normal_direct_camera",
        "normal_depth_exact_k_camera",
    ):
        normal = result[name]
        norm = np.linalg.norm(normal, axis=-1, keepdims=True)
        result[name] = np.divide(
            normal,
            np.maximum(norm, 1.0e-6),
            out=np.zeros_like(normal),
            where=np.isfinite(norm) & (norm > 1.0e-6),
        )
    result["depth_m"][~np.isfinite(result["depth_m"])] = 0.0
    result["valid_mask"] &= (result["depth_m"] > 0.0).astype(np.uint8)
    for name in (
        "normal_direct_camera",
        "normal_depth_exact_k_camera",
        "refinement_log_depth_std",
        "refinement_final_delta_log_depth",
    ):
        result[name][~np.isfinite(result[name])] = 0.0
    depth_normal_norm = np.linalg.norm(
        result["normal_depth_exact_k_camera"], axis=-1
    )
    result["depth_normal_valid_mask"] &= (
        depth_normal_norm > 1.0e-5
    ).astype(np.uint8)
    return index, result


def main() -> None:
    args = _args()
    if args.width <= 0 or args.height <= 0 or args.workers <= 0:
        raise ValueError("width, height and workers must be positive")
    source_index = args.moge3_index.expanduser().resolve()
    index = load_index(source_index, verify_views=False)
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.replace:
            raise FileExistsError(output)
        shutil.rmtree(output)
    temporary = output.with_name(
        f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    order = [str(value) for value in index["camera_order"]]
    shape = (int(args.height), int(args.width))
    arrays: dict[str, np.memmap] = {}
    try:
        for name, dtype in FIELDS.items():
            suffix = (
                (3,)
                if name
                in {
                    "normal_direct_camera",
                    "normal_depth_exact_k_camera",
                }
                else ()
            )
            arrays[name] = np.lib.format.open_memmap(
                temporary / f"{name}.npy",
                mode="w+",
                dtype=dtype,
                shape=(len(order), *shape, *suffix),
            )
        with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = {
                executor.submit(
                    _compact_one,
                    position,
                    stem,
                    index["records"][stem],
                    shape,
                ): stem
                for position, stem in enumerate(order)
            }
            completed = 0
            for future in as_completed(futures):
                position, values = future.result()
                for name, value in values.items():
                    arrays[name][position] = value.astype(
                        FIELDS[name], copy=False
                    )
                completed += 1
                if completed % 50 == 0 or completed == len(order):
                    print(f"runtime-cache {completed}/{len(order)}", flush=True)
        for value in arrays.values():
            value.flush()
        del arrays
        array_contract = {}
        for name, dtype in FIELDS.items():
            path = temporary / f"{name}.npy"
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            array_contract[name] = {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "shape": list(value.shape),
                "dtype": str(np.dtype(dtype)),
            }
        payload = {
            "schema_version": MOGE3_RUNTIME_CACHE_VERSION,
            "source_index": str(source_index),
            "source_index_sha256": sha256_file(source_index),
            "source_index_hash": index["index_hash"],
            "camera_count": len(order),
            "camera_order": order,
            "raster_shape": list(shape),
            "resampling": {
                "depth_and_confidence": "opencv_area",
                "normals": "opencv_area_then_unit_normalize",
                "validity": "nearest",
            },
            "arrays": array_contract,
        }
        payload["index_hash"] = canonical_json_sha256(payload)
        atomic_write_json(temporary / "runtime_index.json", payload)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        print(json.dumps(payload, indent=2))
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
