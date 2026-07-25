#!/usr/bin/env python
"""LEGACY: build the pre-v6 silhouette-only canopy volume.

Use ``build_layered_foliage_geometry.py`` for instance-aware rigid occlusion,
ray/depth posterior, and free-space evidence.  This entry point now requires
an explicit acknowledgement so it cannot silently create incompatible seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from matcha.cambridge_masks import CambridgeMaskLookup
from outdoor.foliage_geometry import (
    build_canopy_volume,
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary_with_tracks,
    save_volume_state,
    semantic_tree_tracks,
)


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--i-understand-this-is-legacy",
        action="store_true",
        help="Acknowledge that this omits v6 occlusion/ray-depth evidence.",
    )
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.2)
    parser.add_argument("--dilation-steps", type=int, default=2)
    parser.add_argument("--maximum-voxels", type=int, default=250_000)
    parser.add_argument("--selected-view-count", type=int, default=64)
    parser.add_argument("--minimum-track-observations", type=int, default=3)
    parser.add_argument("--minimum-track-sequences", type=int, default=2)
    parser.add_argument("--minimum-track-tree-fraction", type=float, default=0.7)
    parser.add_argument("--maximum-reprojection-error", type=float, default=2.0)
    parser.add_argument("--minimum-voxel-support-views", type=int, default=3)
    parser.add_argument("--minimum-voxel-support-sequences", type=int, default=2)
    parser.add_argument("--minimum-occupancy", type=float, default=0.6)
    parser.add_argument("--minimum-baseline", type=float, default=0.75)
    parser.add_argument("--minimum-angle-degrees", type=float, default=1.5)
    parser.add_argument("--minimum-camera-distance", type=float, default=0.75)
    parser.add_argument("--maximum-projected-radius", type=float, default=128.0)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()
    if not args.i_understand_this_is_legacy:
        parser.error(
            "legacy builder disabled by default; use "
            "scripts/build_layered_foliage_geometry.py"
        )
    return args


def main():
    args = _parse_args()
    source = args.source_path.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sparse = source / "sparse" / "0"
    camera_path = sparse / "cameras.bin"
    image_path = sparse / "images.bin"
    point_path = sparse / "points3D.bin"
    cameras = read_cameras_binary(camera_path)
    images = read_images_binary(image_path)
    points = read_points3d_binary_with_tracks(point_path)
    masks = CambridgeMaskLookup(source, args.tree_mask_pickle.resolve())
    tracks = semantic_tree_tracks(
        points,
        images,
        cameras,
        masks,
        minimum_tree_observations=args.minimum_track_observations,
        minimum_sequences=args.minimum_track_sequences,
        minimum_tree_fraction=args.minimum_track_tree_fraction,
        maximum_reprojection_error=args.maximum_reprojection_error,
    )
    if not tracks:
        raise RuntimeError("No semantic multi-sequence tree track survived")
    np.savez_compressed(
        output / "semantic_tree_tracks.npz",
        xyz=np.stack([point["xyz"] for point in tracks]).astype(np.float32),
        rgb=np.stack([point["rgb"] for point in tracks]),
        reprojection_error=np.asarray(
            [point["error"] for point in tracks], dtype=np.float32
        ),
        track_length=np.asarray(
            [len(point["image_ids"]) for point in tracks], dtype=np.int16
        ),
        tree_observation_count=np.asarray(
            [len(point["tree_image_ids"]) for point in tracks], dtype=np.int16
        ),
        tree_sequence_count=np.asarray(
            [point["tree_sequence_count"] for point in tracks], dtype=np.int16
        ),
        tree_fraction=np.asarray(
            [point["tree_fraction"] for point in tracks], dtype=np.float32
        ),
    )
    volume = build_canopy_volume(
        tracks,
        images,
        cameras,
        masks,
        voxel_size=args.voxel_size,
        dilation_steps=args.dilation_steps,
        maximum_voxels=args.maximum_voxels,
        selected_view_count=args.selected_view_count,
        minimum_support_views=args.minimum_voxel_support_views,
        minimum_support_sequences=args.minimum_voxel_support_sequences,
        minimum_occupancy=args.minimum_occupancy,
        minimum_baseline=args.minimum_baseline,
        minimum_triangulation_angle_degrees=args.minimum_angle_degrees,
        minimum_camera_distance=args.minimum_camera_distance,
        maximum_projected_radius=args.maximum_projected_radius,
        seed=args.seed,
    )
    selected_views = [
        {
            "image_id": int(view["image_id"]),
            "image_name": view["image_name"],
            "sequence_id": view["sequence_id"],
            "canopy_fraction": float(view["canopy_fraction"]),
            "camera_center": np.asarray(view["camera_center"]).tolist(),
        }
        for view in volume.pop("selected_views")
    ]
    audit = {
        "protocol": "independent_sfm_semantic_canopy_volume_v1",
        "source_path": str(source),
        "inputs": {
            "cameras_bin_sha256": _digest(camera_path),
            "images_bin_sha256": _digest(image_path),
            "points3d_bin_sha256": _digest(point_path),
            "tree_mask_pickle": str(args.tree_mask_pickle.resolve()),
            "tree_mask_pickle_sha256": _digest(args.tree_mask_pickle.resolve()),
        },
        "source_counts": {
            "cameras": len(cameras),
            "images": len(images),
            "stored_image_observations": int(
                sum(len(image["xys"]) for image in images.values())
            ),
            "sfm_points": len(points),
            "semantic_tree_tracks": len(tracks),
            "candidate_voxels": int(volume["candidate_count"]),
            "accepted_voxels": int(len(volume["centers"])),
            "support_camera_id_capacity": len(selected_views),
        },
        "gates": {
            "minimum_track_observations": args.minimum_track_observations,
            "minimum_track_sequences": args.minimum_track_sequences,
            "minimum_track_tree_fraction": args.minimum_track_tree_fraction,
            "maximum_reprojection_error": args.maximum_reprojection_error,
            "minimum_voxel_support_views": args.minimum_voxel_support_views,
            "minimum_voxel_support_sequences": args.minimum_voxel_support_sequences,
            "minimum_occupancy": args.minimum_occupancy,
            "minimum_baseline": args.minimum_baseline,
            "minimum_angle_degrees": args.minimum_angle_degrees,
            "minimum_camera_distance": args.minimum_camera_distance,
            "maximum_projected_radius": args.maximum_projected_radius,
        },
        "voxel_size": args.voxel_size,
        "selected_views": selected_views,
        "semantics": {
            "semantic_tree_tracks": (
                "SfM tracks consistently inside available tree mask; trunk/branch "
                "labels remain unavailable and are not invented"
            ),
            "canopy_voxels": "semantic visual hull around independent SfM tree anchors",
        },
        "initialization_from_legacy_surfel_xyz": False,
        "historical_full_real_rgb_initialization": False,
        "track_observation_coordinate_policy": (
            "use stored point2D coordinate when present; otherwise calibrated "
            "reprojection of the same points3D track because Cambridge "
            "images.bin preserves image IDs but stores zero point2D rows"
        ),
        "support_camera_ids_truncated": False,
        "occlusion_policy": (
            "non-tree silhouette evidence is conservative but not yet "
            "depth-buffer occlusion aware; candidates remain bounded around "
            "independent semantic SfM anchors"
        ),
    }
    save_volume_state(output / "foliage_seed_gaussians.pth", volume, audit)
    np.savez_compressed(
        output / "canopy_voxels.npz",
        **{
            key: value
            for key, value in volume.items()
            if isinstance(value, np.ndarray)
        },
    )
    (output / "manifest.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        **audit["source_counts"],
        "center_bounds_min": volume["centers"].min(axis=0).tolist(),
        "center_bounds_max": volume["centers"].max(axis=0).tolist(),
        "support_views": {
            "minimum": int(volume["support_view_count"].min()),
            "median": float(np.median(volume["support_view_count"])),
            "maximum": int(volume["support_view_count"].max()),
        },
        "support_sequences": {
            "minimum": int(volume["support_sequence_count"].min()),
            "median": float(np.median(volume["support_sequence_count"])),
            "maximum": int(volume["support_sequence_count"].max()),
        },
        "triangulation_angle_degrees": {
            "minimum": float(volume["maximum_triangulation_angle_degrees"].min()),
            "median": float(np.median(volume["maximum_triangulation_angle_degrees"])),
            "maximum": float(volume["maximum_triangulation_angle_degrees"].max()),
        },
        "nearest_camera_distance": {
            "minimum": float(volume["nearest_camera_distance"].min()),
            "median": float(np.median(volume["nearest_camera_distance"])),
        },
        "largest_projected_radius": {
            "median": float(np.median(volume["largest_projected_radius"])),
            "maximum": float(volume["largest_projected_radius"].max()),
        },
        "occupancy_probability": {
            "minimum": float(volume["occupancy_probability"].min()),
            "median": float(np.median(volume["occupancy_probability"])),
            "maximum": float(volume["occupancy_probability"].max()),
        },
        "inverse_depth_variance": {
            "median": float(np.median(volume["inverse_depth_variance"])),
            "p95": float(np.quantile(volume["inverse_depth_variance"], 0.95)),
            "maximum": float(volume["inverse_depth_variance"].max()),
        },
    }
    (output / "result.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
