#!/usr/bin/env python3
"""Build canonical-tree support maps from aligned MAtCha chart geometry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "mast3r"))

from colmap.read_write_model import read_cameras_binary, read_images_binary  # noqa: E402
from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402
from view_quality_control.poses import qvec_to_rotmat  # noqa: E402


def intrinsics(camera, width: int, height: int) -> tuple[float, float, float, float]:
    params = np.asarray(camera.params, dtype=np.float64)
    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        fx = fy = params[0]
        cx, cy = params[1:3]
    elif camera.model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        fx, fy, cx, cy = params[:4]
    else:
        raise RuntimeError(f"Unsupported camera model for tree support: {camera.model}")
    return (
        fx * width / camera.width,
        fy * height / camera.height,
        cx * width / camera.width,
        cy * height / camera.height,
    )


def pose_features(images: list) -> np.ndarray:
    centers, directions = [], []
    for image in images:
        rotation = qvec_to_rotmat(image.qvec)
        center = -rotation.T @ image.tvec
        direction = rotation.T @ np.array([0.0, 0.0, 1.0])
        centers.append(center)
        directions.append(direction / max(np.linalg.norm(direction), 1e-12))
    centers = np.stack(centers)
    extent = max(float(np.linalg.norm(centers.max(0) - centers.min(0))), 1e-12)
    centers = (centers - centers.mean(0, keepdims=True)) / extent
    return np.concatenate([centers, 0.35 * np.stack(directions)], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mast3r-scene", type=Path, required=True)
    parser.add_argument("--mask-pickle", type=Path, required=True)
    parser.add_argument("--mask-dataset-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tree-mask-index", type=int, default=3)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--min-support-views", type=int, default=2)
    parser.add_argument("--relative-depth-threshold", type=float, default=0.10)
    args = parser.parse_args()

    charts = np.load(args.mast3r_scene / "charts_data.npz")
    points = charts["pts"].astype(np.float32, copy=False)
    depths = charts["depths"].astype(np.float32, copy=False)
    confidences = charts["confs"].astype(np.float32, copy=False)
    scale_factor = float(charts["scale_factor"])
    height, width = depths.shape[-2:]

    sparse = args.mast3r_scene / "sparse" / "0"
    cameras = read_cameras_binary(str(sparse / "cameras.bin"))
    images = sorted(read_images_binary(str(sparse / "images.bin")).values(), key=lambda item: item.id)
    if len(images) != len(depths):
        raise RuntimeError(f"Expected {len(depths)} chart cameras, found {len(images)}")

    lookup = CambridgeMaskLookup(args.mask_dataset_path, args.mask_pickle, mask_indices=[0, 1, 2])
    tree_masks = np.stack([
        (~lookup.get_index_mask(image.name, args.tree_mask_index, (height, width), torch.device("cpu")))
        .numpy()
        for image in images
    ])
    features = pose_features(images)
    pairwise = np.linalg.norm(features[:, None] - features[None], axis=2)
    np.fill_diagonal(pairwise, np.inf)

    output_records = []
    args.output.mkdir(parents=True, exist_ok=True)
    for source_index, source_image in enumerate(images):
        neighbor_indices = np.argsort(pairwise[source_index])[: min(args.neighbors, len(images) - 1)]
        source_valid = (
            tree_masks[source_index]
            & np.isfinite(points[source_index]).all(axis=-1)
            & np.isfinite(depths[source_index])
            & (depths[source_index] > 0)
            & np.isfinite(confidences[source_index])
            & (confidences[source_index] > 0)
        )
        flat_indices = np.flatnonzero(source_valid.ravel())
        support_count = np.zeros((height * width,), dtype=np.uint8)
        source_points = points[source_index].reshape(-1, 3)[flat_indices]

        for target_index in neighbor_indices:
            target_image = images[int(target_index)]
            target_camera = cameras[target_image.camera_id]
            rotation = qvec_to_rotmat(target_image.qvec)
            camera_points = source_points @ rotation.T + target_image.tvec[None] * scale_factor
            z = camera_points[:, 2]
            fx, fy, cx, cy = intrinsics(target_camera, width, height)
            safe_z = np.maximum(z, 1e-8)
            x = np.rint(fx * camera_points[:, 0] / safe_z + cx).astype(np.int64)
            y = np.rint(fy * camera_points[:, 1] / safe_z + cy).astype(np.int64)
            inside = (z > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
            accepted = np.zeros(len(flat_indices), dtype=bool)
            if np.any(inside):
                rows, cols = y[inside], x[inside]
                target_depth = depths[target_index, rows, cols]
                relative = np.abs(z[inside] - target_depth) / np.maximum(np.abs(target_depth), 1e-6)
                accepted[inside] = (
                    tree_masks[target_index, rows, cols]
                    & np.isfinite(target_depth)
                    & (target_depth > 0)
                    & (confidences[target_index, rows, cols] > 0)
                    & (relative <= args.relative_depth_threshold)
                )
            support_count[flat_indices[accepted]] += 1

        support = np.zeros((height * width,), dtype=np.float32)
        canonical = support_count >= args.min_support_views
        support[canonical] = np.minimum(
            support_count[canonical] / max(args.min_support_views, 1),
            1.0,
        )
        support = support.reshape(height, width)
        output_path = args.output / f"{Path(source_image.name).stem}.npy"
        np.save(output_path, support.astype(np.float16))
        tree_pixels = int(tree_masks[source_index].sum())
        output_records.append({
            "image_name": source_image.name,
            "tree_fraction": float(tree_masks[source_index].mean()),
            "canonical_fraction_of_tree": float((support > 0).sum() / max(tree_pixels, 1)),
            "mean_support_on_tree": float(support[tree_masks[source_index]].mean()) if tree_pixels else 0.0,
            "neighbors": [images[int(index)].name for index in neighbor_indices],
        })

    report = {
        "method": "aligned_chart_bidirectional_semantic_depth_support",
        "neighbors": args.neighbors,
        "min_support_views": args.min_support_views,
        "relative_depth_threshold": args.relative_depth_threshold,
        "records": output_records,
    }
    (args.output / "tree_support_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "views": len(output_records),
        "mean_tree_fraction": float(np.mean([row["tree_fraction"] for row in output_records])),
        "mean_canonical_fraction_of_tree": float(np.mean([row["canonical_fraction_of_tree"] for row in output_records])),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
