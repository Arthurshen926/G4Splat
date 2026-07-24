#!/usr/bin/env python
"""Build instance/ray/depth-aware crown and static SfM skeleton seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SURFEL_ROOT = REPO_ROOT / "2d-gaussian-splatting"
sys.path[:0] = [str(REPO_ROOT), str(SURFEL_ROOT)]

from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402
from outdoor.foliage_geometry import (  # noqa: E402
    _camera_record,
    build_instance_aware_canopy_volume,
    cluster_tree_instances,
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary_with_tracks,
    semantic_tree_tracks,
)
from outdoor.foliage_view_graph import greedy_diverse_views  # noqa: E402
from outdoor.rigid_occlusion import render_protected_depth_maps  # noqa: E402
from scripts.augment_foliage_seed_with_sfm_tracks import (  # noqa: E402
    _track_frames,
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path, required=True)
    parser.add_argument("--structural-ply", type=Path, required=True)
    parser.add_argument("--replacement-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.12)
    parser.add_argument("--maximum-voxels", type=int, default=400_000)
    parser.add_argument("--selected-view-count", type=int, default=64)
    parser.add_argument("--depth-resolution-scale", type=float, default=0.5)
    parser.add_argument("--minimum-track-observations", type=int, default=3)
    parser.add_argument("--minimum-track-sequences", type=int, default=2)
    parser.add_argument("--maximum-reprojection-error", type=float, default=2.0)
    parser.add_argument("--skeleton-linearity", type=float, default=1.8)
    parser.add_argument("--seed", type=int, default=73)
    return parser.parse_args()


def _merge(hull, tracks, track_instances, skeleton_linearity):
    xyz = np.stack([point["xyz"] for point in tracks]).astype(np.float32)
    rgb = np.stack([point["rgb"] for point in tracks]).astype(np.float32) / 255.0
    scales, quaternions, linearity = _track_frames(
        xyz, 0.008, 0.035, 0.10
    )
    track_count = len(tracks)
    capacity = hull["support_camera_ids"].shape[1]
    track_camera_ids = np.full((track_count, capacity), -1, dtype=np.int32)
    for index, point in enumerate(tracks):
        values = np.unique(point["tree_image_ids"])[:capacity]
        track_camera_ids[index, : len(values)] = values
    error = np.asarray([point["error"] for point in tracks], dtype=np.float32)
    covariance_scale = np.maximum(
        0.008, 0.012 + 0.01 * error
    ).astype(np.float32)
    track_covariance = np.eye(3, dtype=np.float32)[None] * (
        covariance_scale[:, None, None] ** 2
    )
    layer_role = np.where(
        linearity >= float(skeleton_linearity), 1, 0
    ).astype(np.int8)
    track_values = {
        "centers": xyz,
        "colors": rgb,
        "scales": scales,
        "opacities": np.full((track_count, 1), 0.025, dtype=np.float32),
        "quaternions": quaternions,
        "primitive_role": np.ones(track_count, dtype=np.int8),
        "layer_role": layer_role,
        "track_id": np.asarray(
            [point["id"] for point in tracks], dtype=np.int64
        ),
        "tree_instance_id": track_instances.astype(np.int32),
        "initialization_source": np.ones(track_count, dtype=np.int8),
        "occupancy_probability": np.asarray(
            [point["tree_fraction"] for point in tracks], dtype=np.float32
        ),
        "position_covariance": track_covariance,
        "reprojection_error": error,
        "track_linearity": linearity,
        "support_camera_ids": track_camera_ids,
        "support_view_count": np.asarray(
            [len(point["tree_image_ids"]) for point in tracks],
            dtype=np.int16,
        ),
        "support_sequence_count": np.asarray(
            [point["tree_sequence_count"] for point in tracks],
            dtype=np.int16,
        ),
    }
    output = {}
    required = tuple(track_values)
    for key in required:
        if key not in hull:
            raise RuntimeError(f"Ray/depth hull is missing metadata {key}")
        output[key] = np.concatenate([hull[key], track_values[key]], axis=0)
    # Preserve hull-only diagnostics with aligned neutral track values.
    for key, value in hull.items():
        if key in output or not isinstance(value, np.ndarray):
            continue
        if len(value) != len(hull["centers"]):
            continue
        fill_shape = (track_count,) + value.shape[1:]
        fill = np.zeros(fill_shape, dtype=value.dtype)
        output[key] = np.concatenate([value, fill], axis=0)
    return output, {
        "hull_count": int(len(hull["centers"])),
        "track_count": track_count,
        "static_skeleton_count": int((layer_role == 1).sum()),
        "track_crown_anchor_count": int((layer_role == 0).sum()),
        "tree_instance_count": int(track_instances.max()) + 1,
    }


def main():
    args = _parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source = args.source_path.resolve()
    sparse = source / "sparse" / "0"
    cameras = read_cameras_binary(sparse / "cameras.bin")
    images = read_images_binary(sparse / "images.bin")
    points = read_points3d_binary_with_tracks(sparse / "points3D.bin")
    masks = CambridgeMaskLookup(source, args.tree_mask_pickle.resolve())
    tracks = semantic_tree_tracks(
        points,
        images,
        cameras,
        masks,
        minimum_tree_observations=args.minimum_track_observations,
        minimum_sequences=args.minimum_track_sequences,
        maximum_reprojection_error=args.maximum_reprojection_error,
    )
    xyz = np.stack([point["xyz"] for point in tracks])
    instances = cluster_tree_instances(xyz)
    view_records = [
        _camera_record(image, cameras[image["camera_id"]], masks)
        for image in images.values()
    ]
    selected = greedy_diverse_views(
        view_records,
        limit=args.selected_view_count,
        minimum_center_distance=0.75,
    )
    rigid_depth = render_protected_depth_maps(
        selected,
        structural_ply=args.structural_ply,
        replacement_audit=args.replacement_audit,
        resolution_scale=args.depth_resolution_scale,
    )
    hull = build_instance_aware_canopy_volume(
        tracks,
        images,
        cameras,
        masks,
        rigid_depth_maps=rigid_depth,
        selected_views=selected,
        voxel_size=args.voxel_size,
        maximum_voxels=args.maximum_voxels,
        seed=args.seed,
    )
    selected_audit = [
        {
            "image_id": int(view["image_id"]),
            "image_name": str(view["image_name"]),
            "sequence_id": str(view["sequence_id"]),
        }
        for view in hull.pop("selected_views")
    ]
    candidate_count = int(hull.pop("candidate_count"))
    rigid_depth_view_count = int(hull.pop("rigid_depth_view_count"))
    tree_instance_count = int(hull.pop("tree_instance_count"))
    merged, counts = _merge(
        hull, tracks, instances, args.skeleton_linearity
    )
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "geometry_version": "instance_occlusion_ray_depth_layered_v2",
        "audit": {
            "protocol": "layered_foliage_geometry_v2",
            "tree_instances": tree_instance_count,
            "selected_views": selected_audit,
            "candidate_voxels": candidate_count,
            "rigid_depth_view_count": rigid_depth_view_count,
            "positive_negative_unknown_evidence": True,
            "depth_posterior": "local_same-instance_SfM_track_ray_likelihood",
            "cross_camera_depth_variance_used": False,
            "protected_rigid_zbuffer": True,
            "historical_full_real_rgb_initialization": False,
            **counts,
        },
        **{key: torch.from_numpy(value) for key, value in merged.items()},
    }
    torch.save(payload, output / "foliage_seed_gaussians.pth")
    np.savez_compressed(output / "layered_foliage_geometry.npz", **merged)
    result = {
        **payload["audit"],
        "total_count": int(len(merged["centers"])),
        "support_views_median": float(
            np.median(merged["support_view_count"])
        ),
        "support_sequences_median": float(
            np.median(merged["support_sequence_count"])
        ),
        "position_sigma_median": float(
            np.median(
                np.sqrt(
                    np.linalg.eigvalsh(merged["position_covariance"]).max(1)
                )
            )
        ),
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
