#!/usr/bin/env python3
"""Attach cross-traversal precision posteriors to MASt3R pointmap evidence.

Raw MASt3R confidence is local to one image pair/run.  In Cambridge outdoor
scenes, a locally confident pointmap can still put a facade or foreground
object at a different world depth in another traversal.  This tool constructs
an independent-sequence metric reference and persists a native-raster
precision map for every pointmap.  Unsupported pixels remain available at a
small uncertainty floor; they do not act as equally strong metric anchors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402
from outdoor.evidence_store import (  # noqa: E402
    artifact_path,
    load_evidence_store,
    mast3r_is_geometry_authority,
)
from outdoor.foliage_view_graph import sequence_id  # noqa: E402
from outdoor.role_aware_initialization import (  # noqa: E402
    SINGLE_SEQUENCE_POINTMAP_PRECISION,
    _load_mast3r_pointmap_geometry,
    _native_pointmap_shape,
    _pointmap_fixed_camera_validity,
    _retarget_pointmap_fixed_camera_rays,
    _uniform_confidence_samples,
)
from outdoor.scene_contract import sha256_file  # noqa: E402


POSTERIOR_VERSION = "mast3r-cross-sequence-pointmap-posterior-v1"


def _canonical_json_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_native_record(
    *,
    stem: str,
    record: dict,
    dataset: Path,
    lookup: CambridgeMaskLookup,
    minimum_confidence: float,
    scene_record: dict,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[int, int],
    dict[str, int],
]:
    path = Path(record["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or sha256_file(path) != record["sha256"]
    ):
        raise RuntimeError(f"Pointmap identity mismatch: {path}")
    staged_name = lookup.staged_by_stem.get(stem, f"{stem}.png")
    image_path = dataset / "images" / staged_name
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as image:
        source_size = image.size
    points_flat, confidence_flat = _load_mast3r_pointmap_geometry(path)
    native_shape = _native_pointmap_shape(
        len(confidence_flat), source_size
    )
    masks = lookup.get_index_masks(
        staged_name,
        (0, 1, 2, 3),
        native_shape,
        torch.device("cpu"),
    ).numpy()
    rigid = masks.all(axis=0).reshape(-1)
    local_candidate = (
        np.isfinite(points_flat).all(axis=1)
        & (np.linalg.norm(points_flat, axis=1) > 1e-5)
        & np.isfinite(confidence_flat)
        & (confidence_flat >= float(minimum_confidence))
    )
    original_fixed_ray = _pointmap_fixed_camera_validity(
        points_flat,
        native_shape,
        scene_record,
    )
    points_flat = _retarget_pointmap_fixed_camera_rays(
        points_flat,
        native_shape,
        scene_record,
    )
    fixed_camera_valid = _pointmap_fixed_camera_validity(
        points_flat,
        native_shape,
        scene_record,
    )
    valid = rigid & local_candidate & fixed_camera_valid
    return (
        points_flat,
        confidence_flat,
        valid,
        native_shape,
        {
            "finite_confident_pixels": int(local_candidate.sum()),
            "finite_confident_original_ray_mismatch_pixels": int(
                (local_candidate & ~original_fixed_ray).sum()
            ),
            "finite_confident_nonpositive_or_invalid_depth_pixels": int(
                (local_candidate & ~fixed_camera_valid).sum()
            ),
            "consumer_ray_policy": (
                "preserve_camera_z_then_unproject_exact_fixed_K"
            ),
        },
    )


def _voxel_compact(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if not len(points):
        return points.astype(np.float32)
    voxel = np.floor(
        points / max(float(voxel_size), 1e-5)
    ).astype(np.int64)
    _, first = np.unique(voxel, axis=0, return_index=True)
    return points[np.sort(first)].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-evidence-store", type=Path, required=True)
    parser.add_argument("--output-evidence-store", type=Path, required=True)
    parser.add_argument(
        "--reference-samples-per-view", type=int, default=4000
    )
    parser.add_argument("--reference-voxel-size", type=float, default=0.025)
    parser.add_argument("--minimum-confidence", type=float, default=1.25)
    parser.add_argument("--support-radius", type=float, default=0.15)
    parser.add_argument(
        "--full-trust-radius", type=float, default=0.06
    )
    parser.add_argument(
        "--supported-minimum-precision", type=float, default=0.20
    )
    parser.add_argument(
        "--unsupported-precision-floor",
        type=float,
        default=SINGLE_SEQUENCE_POINTMAP_PRECISION,
    )
    args = parser.parse_args()

    base = load_evidence_store(args.base_evidence_store)
    if not mast3r_is_geometry_authority(base):
        raise RuntimeError(
            "Pointmap posterior requires MASt3R/MAtCha-primary geometry"
        )
    output = args.output_evidence_store.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    posterior_root = output / "pointmap_cross_sequence_posteriors"
    posterior_root.mkdir()

    pointmap_index_path = artifact_path(base, "mast3r_pointmap_index")
    index = json.loads(pointmap_index_path.read_text(encoding="utf-8"))
    records = {
        str(stem): dict(record)
        for stem, record in index["records"].items()
    }
    camera_order = [str(stem) for stem in index["camera_order"]]
    if set(camera_order) != set(records):
        raise RuntimeError("Pointmap index camera order is incomplete")

    semantic = json.loads(
        Path(base["semantic_contract"]).read_text(encoding="utf-8")
    )
    dataset = Path(base["dataset"])
    lookup = CambridgeMaskLookup(
        dataset,
        Path(semantic["tree_mask_pickle"]),
        mask_indices=[0, 1, 2, 3],
    )
    scene_payload = json.loads(
        Path(base["scene_contract"]).read_text(encoding="utf-8")
    )
    scene_records = {
        Path(str(row["image_name"])).stem: row
        for row in scene_payload["records"]
    }
    missing_scene_cameras = sorted(set(camera_order) - set(scene_records))
    if missing_scene_cameras:
        raise RuntimeError(
            "Pointmap cameras are missing from the immutable scene contract: "
            + ", ".join(missing_scene_cameras[:5])
        )

    references: dict[str, list[np.ndarray]] = {}
    native_audit: dict[str, dict[str, Any]] = {}
    for stem in camera_order:
        points, confidence, valid, native_shape, coordinate_audit = (
            _load_native_record(
            stem=stem,
            record=records[stem],
            dataset=dataset,
            lookup=lookup,
            minimum_confidence=args.minimum_confidence,
            scene_record=scene_records[stem],
        )
        )
        selected = _uniform_confidence_samples(
            valid.reshape(native_shape),
            confidence.reshape(native_shape),
            args.reference_samples_per_view,
        )
        current_sequence = str(sequence_id(stem))
        references.setdefault(current_sequence, []).append(points[selected])
        native_audit[stem] = {
            "shape": list(native_shape),
            "valid_pixels": int(valid.sum()),
            "reference_samples": int(len(selected)),
            "sequence": current_sequence,
            **coordinate_audit,
        }

    compact_references = {
        current_sequence: _voxel_compact(
            np.concatenate(parts),
            args.reference_voxel_size,
        )
        for current_sequence, parts in references.items()
        if parts and sum(map(len, parts))
    }
    if len(compact_references) < 2:
        raise RuntimeError(
            "Cross-sequence posterior needs at least two traversals"
        )

    support_radius = float(args.support_radius)
    full_trust_radius = min(
        max(float(args.full_trust_radius), 0.0), support_radius
    )
    supported_minimum = float(
        np.clip(args.supported_minimum_precision, 0.0, 1.0)
    )
    unsupported_floor = float(
        np.clip(args.unsupported_precision_floor, 0.0, supported_minimum)
    )
    aggregate = {
        "valid_pixels": 0,
        "supported_pixels": 0,
        "precision_sum": 0.0,
    }
    for current_sequence in sorted(compact_references):
        other = [
            points
            for name, points in compact_references.items()
            if name != current_sequence and len(points)
        ]
        if not other:
            raise RuntimeError(
                f"No independent traversal reference for {current_sequence}"
            )
        reference_tree = cKDTree(np.concatenate(other))
        for stem in camera_order:
            if str(sequence_id(stem)) != current_sequence:
                continue
            points, _confidence, valid, native_shape, _ = _load_native_record(
                stem=stem,
                record=records[stem],
                dataset=dataset,
                lookup=lookup,
                minimum_confidence=args.minimum_confidence,
                scene_record=scene_records[stem],
            )
            distance = np.full(len(points), np.inf, dtype=np.float32)
            distance[valid] = reference_tree.query(
                points[valid], k=1, workers=-1
            )[0].astype(np.float32)
            supported = valid & (distance <= support_radius)
            precision = np.zeros(len(points), dtype=np.float32)
            precision[valid] = unsupported_floor
            precision[supported] = supported_minimum
            full = supported & (distance <= full_trust_radius)
            precision[full] = 1.0
            transition = supported & ~full
            if bool(transition.any()) and support_radius > full_trust_radius:
                fraction = (
                    support_radius - distance[transition]
                ) / (support_radius - full_trust_radius)
                precision[transition] = (
                    supported_minimum
                    + (1.0 - supported_minimum)
                    * np.clip(fraction, 0.0, 1.0)
                )
            destination = posterior_root / f"{stem}.npz"
            np.savez_compressed(
                destination,
                schema_version=np.asarray(POSTERIOR_VERSION),
                image_name=np.asarray(stem),
                native_shape=np.asarray(native_shape, dtype=np.int32),
                precision=precision.reshape(native_shape).astype(np.float16),
                cross_sequence_supported=supported.reshape(native_shape),
                cross_sequence_distance=np.where(
                    np.isfinite(distance), distance, -1.0
                ).reshape(native_shape).astype(np.float16),
                support_radius=np.asarray(support_radius, dtype=np.float32),
                full_trust_radius=np.asarray(
                    full_trust_radius, dtype=np.float32
                ),
                unsupported_precision_floor=np.asarray(
                    unsupported_floor, dtype=np.float32
                ),
            )
            records[stem]["cross_sequence_posterior"] = {
                "schema_version": POSTERIOR_VERSION,
                "path": str(destination),
                "bytes": int(destination.stat().st_size),
                "sha256": sha256_file(destination),
            }
            audit = native_audit[stem]
            audit["supported_pixels"] = int(supported.sum())
            audit["supported_fraction"] = float(
                supported.sum() / max(valid.sum(), 1)
            )
            audit["mean_precision"] = float(
                precision[valid].mean() if bool(valid.any()) else 0.0
            )
            aggregate["valid_pixels"] += int(valid.sum())
            aggregate["supported_pixels"] += int(supported.sum())
            aggregate["precision_sum"] += float(precision[valid].sum())

    index["records"] = {stem: records[stem] for stem in camera_order}
    index["cross_sequence_posterior"] = {
        "schema_version": POSTERIOR_VERSION,
        "reference_samples_per_view": int(
            args.reference_samples_per_view
        ),
        "reference_voxel_size": float(args.reference_voxel_size),
        "minimum_confidence": float(args.minimum_confidence),
        "support_radius": support_radius,
        "full_trust_radius": full_trust_radius,
        "supported_minimum_precision": supported_minimum,
        "unsupported_precision_floor": unsupported_floor,
        "sequence_reference_counts": {
            name: int(len(points))
            for name, points in compact_references.items()
        },
        "camera_audit": native_audit,
        "aggregate": {
            **aggregate,
            "supported_fraction": (
                aggregate["supported_pixels"]
                / max(aggregate["valid_pixels"], 1)
            ),
            "mean_precision": (
                aggregate["precision_sum"]
                / max(aggregate["valid_pixels"], 1)
            ),
        },
    }
    output_index = output / "mast3r_pointmap_index.json"
    output_index.write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )

    manifest = dict(base)
    manifest.pop("evidence_hash", None)
    manifest["derived_from_evidence_hash"] = base["evidence_hash"]
    manifest["pointmap_cross_sequence_posterior_contract"] = {
        "schema_version": POSTERIOR_VERSION,
        "unsupported_precision_floor": unsupported_floor,
        "support_is_explicit_boolean_not_precision_threshold": True,
        "single_sequence_observations_remain_low_precision": True,
    }
    # Preserve optional SfM coverage provenance. The posterior itself reads
    # only MASt3R pointmaps and must not erase a sibling evidence source from
    # the derived store's immutable contract.
    manifest["colmap_points_or_tracks_used"] = bool(
        base.get("colmap_points_or_tracks_used", False)
    )
    manifest.setdefault("historical_gaussian_initialization_used", False)
    artifacts = [dict(row) for row in manifest["artifacts"]]
    for row in artifacts:
        if row["name"] != "mast3r_pointmap_index":
            continue
        row.update(
            {
                "source_type": (
                    "mast3r_dense_geometry_cross_sequence_posterior"
                ),
                "path": str(output_index),
                "sha256": sha256_file(output_index),
                "bytes": int(output_index.stat().st_size),
                "measurement": (
                    "content-addressed dense pointmaps with native-raster "
                    "independent-traversal metric precision posterior"
                ),
                "validity": (
                    "exact fixed cameras, rigid task mask, local confidence, "
                    "and cross-traversal world-space support"
                ),
            }
        )
        break
    else:
        raise RuntimeError("Evidence Store has no pointmap index artifact")
    manifest["artifacts"] = artifacts
    manifest["evidence_hash"] = _canonical_json_digest(manifest)
    (output / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    load_evidence_store(output)
    print(
        json.dumps(
            {
                "evidence_store": str(output),
                "evidence_hash": manifest["evidence_hash"],
                "pointmap_count": len(camera_order),
                "cross_sequence_posterior": (
                    index["cross_sequence_posterior"]["aggregate"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
