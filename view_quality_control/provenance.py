"""Lightweight, deterministic input identities for Cambridge QC stages.

The masks used by the project are gigabyte-scale, so hashing every byte on
every launch is needlessly expensive.  This module instead fingerprints the
complete image identity/stat manifest plus all sparse-model and mask file
identities.  It is strong enough to prevent an accidental stale audit from
silently entering a supposedly single-variable reconstruction experiment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def _file_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _image_manifest_sha256(image_dir: Path) -> str:
    entries: list[dict[str, int | str]] = []
    for path in sorted(image_dir.iterdir(), key=lambda value: value.name):
        if not (path.is_file() or path.is_symlink()):
            continue
        stat = path.stat()
        entries.append(
            {
                "name": path.name,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    if not entries:
        raise RuntimeError(f"No image files found for provenance: {image_dir}")
    encoded = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def audit_input_provenance(
    dataset: Path,
    mask_pickle: Path,
    *,
    quality_mask_indices: Iterable[int],
    thing_mask_index: int,
    sky_mask_index: int,
    tree_mask_index: int,
    pose_clusters: int,
    neighbor_count: int,
    max_image_width: int,
    max_reject_fraction: float,
) -> dict:
    """Return every input/configuration identity that changes a view audit."""
    dataset = dataset.resolve()
    mask_pickle = mask_pickle.resolve()
    sparse = dataset / "sparse" / "0"
    required_sparse = ("cameras.bin", "images.bin", "points3D.bin")
    missing = [name for name in required_sparse if not (sparse / name).is_file()]
    if missing:
        raise FileNotFoundError(f"QC provenance missing sparse file(s): {missing}")
    mapping = dataset / "name_mapping.json"
    if not mapping.is_file():
        raise FileNotFoundError(mapping)
    if not mask_pickle.is_file():
        raise FileNotFoundError(mask_pickle)
    return {
        "schema_version": 1,
        "dataset": str(dataset),
        "image_manifest_sha256": _image_manifest_sha256(dataset / "images"),
        "name_mapping": _file_stat(mapping),
        "cameras_bin": _file_stat(sparse / "cameras.bin"),
        "images_bin": _file_stat(sparse / "images.bin"),
        "points3d_bin": _file_stat(sparse / "points3D.bin"),
        "mask_pickle": {"path": str(mask_pickle), **_file_stat(mask_pickle)},
        "quality_mask_indices": [int(index) for index in quality_mask_indices],
        "thing_mask_index": int(thing_mask_index),
        "sky_mask_index": int(sky_mask_index),
        "tree_mask_index": int(tree_mask_index),
        "pose_clusters": int(pose_clusters),
        "neighbor_count": int(neighbor_count),
        "max_image_width": int(max_image_width),
        "max_reject_fraction": float(max_reject_fraction),
    }
