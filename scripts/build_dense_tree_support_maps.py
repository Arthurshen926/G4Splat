#!/usr/bin/env python3
"""Build canonical-tree support from posed dense views and scaled mono depth."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402
from matcha.see3d_geometry import robust_align_inverse_depth  # noqa: E402


def _load_colmap_loader():
    path = REPO_ROOT / "2d-gaussian-splatting" / "scene" / "colmap_loader.py"
    spec = importlib.util.spec_from_file_location("g4_colmap_loader", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--reference-render", type=Path, required=True)
    parser.add_argument("--depth-cache", type=Path, required=True)
    parser.add_argument("--base-mask-pickle", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-mask-indices", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--tree-mask-index", type=int, default=3)
    parser.add_argument("--height", type=int, default=135)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--min-support-views", type=int, default=2)
    parser.add_argument("--relative-depth-threshold", type=float, default=0.10)
    parser.add_argument("--max-alignment-rmse", type=float, default=0.20)
    return parser.parse_args()


def camera_intrinsics(camera, width: int, height: int) -> np.ndarray:
    params = np.asarray(camera.params, dtype=np.float64)
    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        fx = fy = params[0]
        cx, cy = params[1:3]
    elif camera.model in {"PINHOLE", "OPENCV", "FULL_OPENCV"}:
        fx, fy, cx, cy = params[:4]
    else:
        raise ValueError(f"Unsupported COLMAP camera model for tree support: {camera.model}")
    sx = width / float(camera.width)
    sy = height / float(camera.height)
    return np.asarray([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])


def resize_bool(mask: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(
        mask.cpu().numpy().astype(np.uint8),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def pose_neighbors(
    names: list[str], centers: np.ndarray, directions: np.ndarray, count: int
) -> list[list[int]]:
    center_scale = np.linalg.norm(centers.max(0) - centers.min(0)) + 1e-6
    output = []
    for index, name in enumerate(names):
        sequence = name.split("__", 1)[0]
        center_distance = np.linalg.norm(centers - centers[index], axis=1) / center_scale
        view_distance = 1.0 - np.clip(directions @ directions[index], -1.0, 1.0)
        score = center_distance + 0.25 * view_distance
        score[index] = np.inf
        same_sequence = np.asarray(
            [candidate.split("__", 1)[0] == sequence for candidate in names]
        )
        order = np.argsort(score + np.where(same_sequence, 0.0, 0.10))
        output.append(order[:count].tolist())
    return output


def main() -> None:
    args = parse_args()
    if args.min_support_views > args.neighbors:
        raise ValueError("min support views cannot exceed neighbor count")
    args.output.mkdir(parents=True, exist_ok=True)
    shape = (args.height, args.width)

    colmap = _load_colmap_loader()
    sparse = args.dataset_path / "sparse" / "0"
    extrinsics = colmap.read_extrinsics_binary(str(sparse / "images.bin"))
    intrinsics = colmap.read_intrinsics_binary(str(sparse / "cameras.bin"))
    ordered_images = sorted(extrinsics.values(), key=lambda image: image.name)
    names = [Path(image.name).stem for image in ordered_images]

    cache = torch.load(args.depth_cache, map_location="cpu")
    cache_names = [Path(name).stem for name in cache["image_names"]]
    depth_by_name = dict(zip(cache_names, cache["depths"]))
    if set(names) != set(cache_names):
        raise RuntimeError("Dense depth cache and COLMAP image names do not match")

    base_lookup = CambridgeMaskLookup(
        args.dataset_path, args.base_mask_pickle, mask_indices=args.base_mask_indices
    )
    tree_lookup = CambridgeMaskLookup(
        args.dataset_path, args.tree_mask_pickle, mask_indices=[args.tree_mask_index]
    )

    rotations = np.stack([image.qvec2rotmat() for image in ordered_images])
    translations = np.stack([image.tvec for image in ordered_images])
    centers = np.einsum("nij,nj->ni", rotations.transpose(0, 2, 1), -translations)
    directions = np.einsum(
        "nij,j->ni", rotations.transpose(0, 2, 1), np.asarray([0.0, 0.0, 1.0])
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(1e-8)
    neighbors = pose_neighbors(names, centers, directions, args.neighbors)
    camera_matrices = [
        camera_intrinsics(intrinsics[image.camera_id], args.width, args.height)
        for image in ordered_images
    ]

    aligned_depths: list[np.ndarray] = []
    tree_masks: list[np.ndarray] = []
    alignment_records = []
    depth_root = args.reference_render / "vis"
    for index, name in enumerate(names):
        render_path = depth_root / f"depth_{index:05d}.tiff"
        if not render_path.exists():
            raise FileNotFoundError(render_path)
        render_depth = cv2.resize(
            np.asarray(Image.open(render_path), dtype=np.float32),
            (args.width, args.height),
            interpolation=cv2.INTER_NEAREST,
        )
        relative_depth = cv2.resize(
            depth_by_name[name].squeeze().float().numpy(),
            (args.width, args.height),
            interpolation=cv2.INTER_LINEAR,
        )
        source_name = base_lookup.source_name_for(name)
        source_shape = tuple(base_lookup.masks[source_name][0].shape[-2:])
        base = resize_bool(base_lookup.get_mask(name, source_shape, "cpu"), shape)
        tree_keep = resize_bool(
            tree_lookup.get_mask(name, source_shape, "cpu"), shape
        )
        tree = ~tree_keep
        static_support = (
            base & (~tree) & np.isfinite(render_depth) & (render_depth > 0.0)
        )
        aligned, diagnostics = robust_align_inverse_depth(
            torch.from_numpy(1.0 / np.maximum(relative_depth, 1e-6)),
            torch.from_numpy(render_depth),
            torch.from_numpy(static_support),
            min_samples=256,
            max_samples=50_000,
            max_relative_rmse=args.max_alignment_rmse,
        )
        aligned_depths.append(aligned.numpy().astype(np.float32))
        tree_masks.append(tree)
        alignment_records.append({"image_name": name, **diagnostics.to_dict()})

    records = []
    yy, xx = np.indices(shape)
    for index, name in enumerate(names):
        tree = tree_masks[index]
        source_depth = aligned_depths[index]
        source_valid = tree & np.isfinite(source_depth) & (source_depth > 0.0)
        support_count = np.zeros(shape, dtype=np.uint8)
        pixels = np.flatnonzero(source_valid)
        if pixels.size:
            u = xx.reshape(-1)[pixels].astype(np.float64)
            v = yy.reshape(-1)[pixels].astype(np.float64)
            z = source_depth.reshape(-1)[pixels].astype(np.float64)
            k = camera_matrices[index]
            camera_points = np.stack(
                [(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z],
                axis=1,
            )
            world_points = (camera_points - translations[index]) @ rotations[index]
            for neighbor in neighbors[index]:
                neighbor_points = world_points @ rotations[neighbor].T + translations[neighbor]
                nz = neighbor_points[:, 2]
                nk = camera_matrices[neighbor]
                nu = np.rint(nk[0, 0] * neighbor_points[:, 0] / nz + nk[0, 2]).astype(int)
                nv = np.rint(nk[1, 1] * neighbor_points[:, 1] / nz + nk[1, 2]).astype(int)
                in_frame = (
                    (nz > 0.0)
                    & (nu >= 0)
                    & (nu < args.width)
                    & (nv >= 0)
                    & (nv < args.height)
                )
                valid_indices = np.flatnonzero(in_frame)
                if not valid_indices.size:
                    continue
                neighbor_depth = aligned_depths[neighbor][nv[valid_indices], nu[valid_indices]]
                neighbor_tree = tree_masks[neighbor][nv[valid_indices], nu[valid_indices]]
                relative_error = np.abs(neighbor_depth - nz[valid_indices]) / np.maximum(
                    np.abs(nz[valid_indices]), 1e-6
                )
                supported = (
                    neighbor_tree
                    & np.isfinite(neighbor_depth)
                    & (neighbor_depth > 0.0)
                    & (relative_error <= args.relative_depth_threshold)
                )
                support_count.reshape(-1)[pixels[valid_indices[supported]]] += 1

        support = support_count.astype(np.float32) / max(args.neighbors, 1)
        support[support_count < args.min_support_views] = 0.0
        np.save(args.output / f"{name}.npy", support.astype(np.float16))
        records.append(
            {
                "image_name": name,
                "tree_fraction": float(tree.mean()),
                "canonical_fraction_of_tree": float(
                    (support[tree] > 0).mean() if np.any(tree) else 0.0
                ),
                "mean_support_on_tree": float(support[tree].mean() if np.any(tree) else 0.0),
                "neighbors": [names[item] for item in neighbors[index]],
            }
        )

    report = {
        "method": "dense_posed_scaled_dav2_tree_support",
        "views": len(names),
        "shape": list(shape),
        "neighbors": args.neighbors,
        "min_support_views": args.min_support_views,
        "relative_depth_threshold": args.relative_depth_threshold,
        "accepted_depth_alignments": sum(record["accepted"] for record in alignment_records),
        "mean_canonical_fraction_of_tree": float(
            np.mean([record["canonical_fraction_of_tree"] for record in records])
        ),
        "alignments": alignment_records,
        "records": records,
    }
    (args.output / "tree_support_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in (
        "views", "accepted_depth_alignments", "mean_canonical_fraction_of_tree"
    )}, indent=2))


if __name__ == "__main__":
    main()
