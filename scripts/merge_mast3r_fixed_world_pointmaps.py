#!/usr/bin/env python3
"""Merge complementary fixed-camera MASt3R pointmap runs into one evidence store.

The Cambridge G4/MASt3R front-end is normally run on a selected keyframe set.
Several independently selected runs can therefore contain complementary real
geometry while sharing the exact calibrated Cambridge world frame.  This tool
merges unique camera observations only after proving that every producer
camera matrix matches the immutable scene contract.  Earlier roots have
priority for duplicate image names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from outdoor.evidence_store import load_evidence_store
from outdoor.role_aware_initialization import (
    _load_mast3r_pointmap_geometry,
    _native_pointmap_shape,
)
from outdoor.scene_contract import sha256_file


INDEX_VERSION = "mast3r-pointmap-index-v2-multi-producer-fixed-world"


def _canonical_json_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scene_cameras(scene_contract: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    payload = json.loads(Path(scene_contract).read_text(encoding="utf-8"))
    cameras = {}
    order = {}
    for row in payload["records"]:
        stem = Path(row["image_name"]).stem
        cameras[stem] = np.asarray(
            row["T_camera_to_world"], dtype=np.float64
        )
        order[stem] = int(row["frame_index"])
    return cameras, order


def _producer_cameras(
    pointmap_root: Path,
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    camera_path = pointmap_root.parent / "cameras.json"
    payload = json.loads(camera_path.read_text(encoding="utf-8"))
    stems = [Path(value).stem for value in payload["filepaths"]]
    cameras = np.asarray(payload["cams2world"], dtype=np.float64)
    if cameras.shape != (len(stems), 4, 4):
        raise RuntimeError(
            f"Unexpected producer camera shape {cameras.shape}: {camera_path}"
        )
    if len(stems) != len(set(stems)):
        raise RuntimeError(f"Duplicate producer camera names: {camera_path}")
    return stems, cameras, payload


def _pointmap_projection_error(
    *,
    pointmap_path: Path,
    camera_to_world: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    maximum_samples: int = 8192,
) -> tuple[float, float]:
    """Check that a pointmap pixel reprojects through its adjacent camera."""
    points, confidence = _load_mast3r_pointmap_geometry(pointmap_path)
    height, width = _native_pointmap_shape(len(confidence), image_size)
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(confidence)
        & (confidence >= 1.25)
    )
    selected = np.flatnonzero(valid)
    if not len(selected):
        raise RuntimeError(f"Pointmap has no finite confident pixels: {pointmap_path}")
    if len(selected) > int(maximum_samples):
        selected = selected[
            np.linspace(
                0, len(selected) - 1, int(maximum_samples), dtype=np.int64
            )
        ]
    world_to_camera = np.linalg.inv(camera_to_world)
    camera = (
        points[selected] @ world_to_camera[:3, :3].T
        + world_to_camera[:3, 3]
    )
    positive = camera[:, 2] > 0.05
    # Dense predictors can retain a small invalid tail even above their local
    # confidence threshold.  Those rows are not camera measurements and are
    # removed by the posterior/initialization loaders.  Reject the producer
    # only when the tail is large enough to indicate a coordinate mismatch.
    if float(positive.mean()) < 0.95:
        raise RuntimeError(
            f"Pointmap has points behind its stored camera: {pointmap_path}"
        )
    selected = selected[positive]
    camera = camera[positive]
    source_width, source_height = map(int, image_size)
    fx, fy, cx, cy = map(float, intrinsics)
    scale_x = width / max(source_width, 1)
    scale_y = height / max(source_height, 1)
    projected = np.column_stack(
        [
            fx * scale_x * camera[:, 0] / camera[:, 2] + cx * scale_x,
            fy * scale_y * camera[:, 1] / camera[:, 2] + cy * scale_y,
        ]
    )
    rows, columns = np.divmod(selected, width)
    expected = np.column_stack([columns, rows])
    error = np.linalg.norm(projected - expected, axis=1)
    return tuple(map(float, np.percentile(error, [50, 90])))


def _validate_producer(
    pointmap_root: Path,
    fixed_cameras: dict[str, np.ndarray],
    *,
    tolerance: float,
    skip_pointmap_stems: set[str] | None = None,
) -> dict[str, Any]:
    stems, cameras, camera_payload = _producer_cameras(pointmap_root)
    missing = [stem for stem in stems if stem not in fixed_cameras]
    if missing:
        raise RuntimeError(
            f"Producer contains cameras outside the scene contract: {missing[:5]}"
        )
    errors = np.asarray(
        [
            np.max(np.abs(camera - fixed_cameras[stem]))
            for stem, camera in zip(stems, cameras)
        ],
        dtype=np.float64,
    )
    if len(errors) and float(errors.max()) > float(tolerance):
        raise RuntimeError(
            f"Producer is not in the fixed Cambridge world: {pointmap_root}; "
            f"maximum camera error={float(errors.max()):.3e}"
        )
    files = {path.stem: path.resolve() for path in pointmap_root.glob("*.json")}
    unknown = sorted(set(files) - set(stems))
    if unknown:
        raise RuntimeError(
            f"Pointmaps lack producer camera records: {unknown[:5]}"
        )
    intrinsics = camera_payload.get("intrinsics")
    image_sizes = camera_payload.get("image_sizes")
    if intrinsics is None or image_sizes is None:
        focals = camera_payload.get("focals")
        if focals is None:
            raise RuntimeError(
                f"Producer has no pointmap camera intrinsics: {pointmap_root}"
            )
        # Legacy MASt3R pointmaps used a centered square-pixel camera.
        image_sizes = []
        intrinsics = []
        for stem, focal in zip(stems, focals):
            points, confidence = _load_mast3r_pointmap_geometry(files[stem])
            height, width = _native_pointmap_shape(
                len(confidence), (512, 288)
            )
            image_sizes.append([width, height])
            intrinsics.append(
                [float(focal), float(focal), width / 2.0, height / 2.0]
            )
    camera_index = {stem: index for index, stem in enumerate(stems)}
    validate_stems = sorted(
        set(files) - set(skip_pointmap_stems or ())
    )
    projection = []
    for stem in validate_stems:
        index = camera_index[stem]
        p50, p90 = _pointmap_projection_error(
            pointmap_path=files[stem],
            camera_to_world=cameras[index],
            intrinsics=np.asarray(intrinsics[index], dtype=np.float64),
            image_size=tuple(map(int, image_sizes[index])),
        )
        projection.append((stem, p50, p90))
    if projection:
        maximum_p50 = max(value[1] for value in projection)
        maximum_p90 = max(value[2] for value in projection)
        if maximum_p50 > 2.0 or maximum_p90 > 4.0:
            worst = max(projection, key=lambda value: value[2])
            raise RuntimeError(
                "Producer pointmaps do not share their advertised fixed "
                f"camera/world frame: {pointmap_root}; worst={worst}; "
                f"maximum_p50={maximum_p50:.3f}px, "
                f"maximum_p90={maximum_p90:.3f}px"
            )
    else:
        maximum_p50 = 0.0
        maximum_p90 = 0.0
    return {
        "pointmap_root": str(pointmap_root.resolve()),
        "camera_manifest": str((pointmap_root.parent / "cameras.json").resolve()),
        "camera_manifest_sha256": sha256_file(
            pointmap_root.parent / "cameras.json"
        ),
        "producer_camera_count": len(stems),
        "pointmap_count": len(files),
        "maximum_camera_matrix_absolute_error": (
            float(errors.max()) if len(errors) else 0.0
        ),
        "pointmap_projection_validated_count": len(projection),
        "maximum_pointmap_projection_p50_px": float(maximum_p50),
        "maximum_pointmap_projection_p90_px": float(maximum_p90),
        "files": files,
    }


def _load_base_pointmap_records(
    evidence_manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, Any]]:
    """Recover already-validated pointmaps from the base Evidence Store.

    The former merge command replaced the base pointmap index with only the
    newly supplied producers.  A complementary run could therefore appear to
    add evidence while silently dropping every prior camera.  Preserve the
    content-addressed base records and let explicit new producers take
    priority only for duplicate camera names.
    """
    artifact = next(
        (
            row
            for row in evidence_manifest.get("artifacts", [])
            if row.get("name") == "mast3r_pointmap_index"
        ),
        None,
    )
    if artifact is None:
        raise RuntimeError("Base store has no mast3r_pointmap_index artifact")
    index_path = Path(artifact["path"]).expanduser().resolve()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("coordinate_frame") != "cambridge_fixed_world":
        raise RuntimeError(
            "Base pointmap index is not in the fixed Cambridge world: "
            f"{index_path}"
        )
    raw_records = payload.get("records", {})
    records: dict[str, dict[str, Any]] = {}
    source_by_camera: dict[str, str] = {}
    for stem in payload.get("camera_order", sorted(raw_records)):
        if stem not in raw_records:
            raise RuntimeError(
                f"Base pointmap order references a missing record: {stem}"
            )
        raw = raw_records[stem]
        path = Path(raw["path"]).expanduser()
        if not path.is_absolute():
            path = (index_path.parent / path).resolve()
        else:
            path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_bytes = int(path.stat().st_size)
        if observed_bytes != int(raw["bytes"]):
            raise RuntimeError(
                f"Base pointmap byte count changed: {path}; "
                f"expected={raw['bytes']}, observed={observed_bytes}"
            )
        observed_sha = sha256_file(path)
        if observed_sha != raw["sha256"]:
            raise RuntimeError(
                f"Base pointmap content hash changed: {path}"
            )
        records[stem] = {
            "path": str(path),
            "bytes": observed_bytes,
            "sha256": observed_sha,
            "producer_priority": -1,
        }
        source_by_camera[stem] = str(
            payload.get("source_by_camera", {}).get(
                stem, f"base:{index_path}"
            )
        )
    return records, source_by_camera, {
        "pointmap_root": str(index_path),
        "camera_manifest": None,
        "camera_manifest_sha256": None,
        "producer_camera_count": len(records),
        "pointmap_count": len(records),
        "maximum_camera_matrix_absolute_error": 0.0,
        "pointmap_projection_validated_count": int(
            sum(
                int(row.get("pointmap_projection_validated_count", 0))
                for row in payload.get("producer_audit", [])
            )
        ),
        "maximum_pointmap_projection_p50_px": float(
            max(
                [
                    float(row.get("maximum_pointmap_projection_p50_px", 0.0))
                    for row in payload.get("producer_audit", [])
                ]
                or [0.0]
            )
        ),
        "maximum_pointmap_projection_p90_px": float(
            max(
                [
                    float(row.get("maximum_pointmap_projection_p90_px", 0.0))
                    for row in payload.get("producer_audit", [])
                ]
                or [0.0]
            )
        ),
        "priority": -1,
        "accepted_unique_pointmaps": len(records),
        "source": "content_addressed_base_evidence_store",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-evidence-store", type=Path, required=True)
    parser.add_argument("--output-evidence-store", type=Path, required=True)
    parser.add_argument(
        "--pointmap-root",
        type=Path,
        action="append",
        required=True,
        help=(
            "MASt3R pointmaps directory. Repeat in descending priority; "
            "the first producer wins duplicate camera names."
        ),
    )
    parser.add_argument("--camera-tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    base = load_evidence_store(args.base_evidence_store)
    output = args.output_evidence_store.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    fixed_cameras, scene_order = _scene_cameras(Path(base["scene_contract"]))

    base_records, base_sources, base_audit = _load_base_pointmap_records(base)
    producer_audits = []
    records: dict[str, dict[str, Any]] = {}
    source_by_camera = {}
    duplicate_count = 0
    for priority, root in enumerate(args.pointmap_root):
        root = root.expanduser().resolve()
        audit = _validate_producer(
            root,
            fixed_cameras,
            tolerance=args.camera_tolerance,
            skip_pointmap_stems=set(records),
        )
        files = audit.pop("files")
        accepted = 0
        for stem, path in files.items():
            if stem in records:
                duplicate_count += 1
                continue
            records[stem] = {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
                "producer_priority": int(priority),
            }
            source_by_camera[stem] = str(root)
            accepted += 1
        audit["priority"] = int(priority)
        audit["accepted_unique_pointmaps"] = int(accepted)
        producer_audits.append(audit)

    base_accepted = 0
    for stem, record in base_records.items():
        if stem in records:
            duplicate_count += 1
            continue
        records[stem] = record
        source_by_camera[stem] = base_sources[stem]
        base_accepted += 1
    base_audit["accepted_unique_pointmaps"] = int(base_accepted)
    producer_audits.append(base_audit)

    camera_order = sorted(records, key=lambda stem: scene_order[stem])
    index = {
        "schema_version": INDEX_VERSION,
        "coordinate_frame": "cambridge_fixed_world",
        "camera_scope": "complementary_real_keyframes",
        "camera_order": camera_order,
        "records": {stem: records[stem] for stem in camera_order},
        "producer_audit": producer_audits,
        "duplicate_camera_observations_skipped": int(duplicate_count),
        "selection_policy": "first_exact_camera_producer_wins",
        "consumer_ray_policy": (
            "preserve_camera_z_then_unproject_exact_fixed_K"
        ),
        "source_by_camera": {
            stem: source_by_camera[stem] for stem in camera_order
        },
    }
    index_path = output / "mast3r_pointmap_index.json"
    index_path.write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )

    manifest = dict(base)
    manifest.pop("evidence_hash", None)
    if manifest.get("geometry_source") != "mast3r_only":
        raise RuntimeError(
            "Complementary fixed-world pointmaps require a MASt3R-only base"
        )
    manifest["colmap_points_or_tracks_used"] = False
    manifest.setdefault("historical_gaussian_initialization_used", False)
    artifacts = [dict(row) for row in manifest["artifacts"]]
    replaced = False
    for row in artifacts:
        if row["name"] != "mast3r_pointmap_index":
            continue
        row.update(
            {
                "source_type": "mast3r_dense_geometry_multi_producer",
                "path": str(index_path),
                "sha256": sha256_file(index_path),
                "bytes": int(index_path.stat().st_size),
                "measurement": (
                    "content-addressed complementary dense point/conf maps "
                    "with camera-z retargeting to exact fixed Cambridge rays"
                ),
                "coordinate_frame": "cambridge_fixed_world",
                "camera_scope": "complementary_real_keyframes",
                "validity": (
                    "all producer cameras validated, all pointmap files "
                    "hashed, and consumers retarget each camera-z sample "
                    "onto its exact fixed-K pixel ray"
                ),
            }
        )
        replaced = True
    if not replaced:
        raise RuntimeError("Base store has no mast3r_pointmap_index artifact")
    manifest["artifacts"] = artifacts
    manifest["evidence_hash"] = _canonical_json_digest(manifest)
    manifest_path = output / "evidence_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    # Re-open through the normal contract checker before reporting success.
    load_evidence_store(output)
    print(
        json.dumps(
            {
                "evidence_store": str(output),
                "evidence_hash": manifest["evidence_hash"],
                "pointmap_count": len(records),
                "producer_count": len(producer_audits),
                "duplicate_camera_observations_skipped": duplicate_count,
                "pointmap_index": str(index_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
