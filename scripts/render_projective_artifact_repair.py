#!/usr/bin/env python3
"""Render artifact patches from posed clean support views on verified planes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from artifact_guided_repair import (  # noqa: E402
    CameraGeometry,
    backproject_depth,
    blend_projective_reference_colors,
    project_points,
    render_support_plane_depth,
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _read_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def _read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255.0 + 0.5), mode="RGB").save(path)


def _save_depth_visualization(path: Path, depth: np.ndarray) -> None:
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size:
        lower, upper = np.quantile(valid, [0.01, 0.99])
        normalized = np.clip((depth - lower) / max(float(upper - lower), 1e-6), 0.0, 1.0)
    else:
        normalized = np.zeros_like(depth)
    plt.imsave(path, normalized, cmap="viridis")


def _sample_reference(
    points: np.ndarray,
    record: dict,
    diagnostics_root: Path,
    max_artifact_score: float,
    relative_depth_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    camera = CameraGeometry.from_json(record["camera"])
    pixels, projected_depth, inside = project_points(points, camera)
    image = _read_rgb(Path(record["image_path"]), (camera.width, camera.height))
    # OpenCV remap uses signed-short image dimensions internally.  A large
    # planar ROI can contain well over 32k samples, so remap point rows in
    # bounded chunks while preserving the exact interpolation operation.
    color_chunks = []
    for start in range(0, len(pixels), 30_000):
        chunk = pixels[start : start + 30_000]
        map_x = chunk[:, 0].astype(np.float32).reshape(-1, 1)
        map_y = chunk[:, 1].astype(np.float32).reshape(-1, 1)
        color_chunks.append(
            cv2.remap(
                image,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            ).reshape(-1, 3)
        )
    colors = np.concatenate(color_chunks, axis=0) if color_chunks else np.empty((0, 3))

    files = record["files"]
    labels = np.asarray(Image.open(diagnostics_root / files["labels"]))
    semantic = _read_gray(diagnostics_root / files["semantic"]) > 127
    score_map = _read_gray(diagnostics_root / files["score"]).astype(np.float32) / 255.0
    alpha = _read_gray(diagnostics_root / files["alpha"]).astype(np.float32) / 255.0
    depth = np.load(diagnostics_root / files["depth"]).astype(np.float32)
    x = np.clip(np.rint(pixels[:, 0]).astype(np.int64), 0, camera.width - 1)
    y = np.clip(np.rint(pixels[:, 1]).astype(np.int64), 0, camera.height - 1)
    scores = score_map[y, x]
    semantic_valid = (
        inside
        & (labels[y, x] == 0)
        & semantic[y, x]
        & (scores <= float(max_artifact_score))
        & (alpha[y, x] >= 0.5)
    )
    denominator = np.maximum(
        np.maximum(np.abs(depth[y, x]), np.abs(projected_depth)),
        1e-6,
    )
    relative_error = np.abs(depth[y, x] - projected_depth) / denominator
    depth_valid = (
        semantic_valid
        & np.isfinite(depth[y, x])
        & (depth[y, x] > 0)
        & (relative_error <= float(relative_depth_threshold))
    )
    metrics = {
        "image_name": record["image_name"],
        "inside_fraction": float(inside.mean()),
        "semantic_clean_fraction": float(semantic_valid.mean()),
        "depth_consistent_fraction": float(depth_valid.mean()),
        "median_relative_depth_error_on_semantic_clean": float(
            np.median(relative_error[semantic_valid])
        )
        if np.any(semantic_valid)
        else None,
    }
    return colors, semantic_valid, depth_valid, scores, metrics


def _nearest_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if valid.all() or not valid.any():
        return values
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = values.copy()
    filled[~valid] = values[nearest[0][~valid], nearest[1][~valid]]
    return filled


def _depth_normals_world(depth: np.ndarray, camera: CameraGeometry) -> np.ndarray:
    yy, xx = np.mgrid[: camera.height, : camera.width]
    rays = np.stack(
        [
            (xx - camera.cx) / camera.fx,
            (yy - camera.cy) / camera.fy,
            np.ones_like(depth),
        ],
        axis=2,
    )
    points_camera = rays * depth[..., None]
    points_world = points_camera @ camera.c2w[:3, :3].T + camera.center
    dx = np.gradient(points_world, axis=1)
    dy = np.gradient(points_world, axis=0)
    normals = np.cross(dx, dy)
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / np.maximum(norm, 1e-8)
    toward_camera = camera.center - points_world
    flip = np.sum(normals * toward_camera, axis=2) < 0
    normals[flip] *= -1.0
    normals[~np.isfinite(normals).all(axis=2)] = 0.0
    return normals.astype(np.float32)


def _write_geometry_bundle(
    output_root: Path,
    index: int,
    rgb: np.ndarray,
    baseline_depth: np.ndarray,
    clean_depth: np.ndarray,
    repair_mask: np.ndarray,
    known_mask: np.ndarray,
    alpha: np.ndarray,
    camera: CameraGeometry,
    plane_world: np.ndarray,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    stem = f"{index:06d}"
    _save_rgb(output_root / f"rgb_frame{stem}.png", rgb)
    merged_depth = baseline_depth.astype(np.float32).copy()
    merged_depth[repair_mask] = clean_depth[repair_mask]
    Image.fromarray(merged_depth, mode="F").save(output_root / f"depth_frame{stem}.tiff")
    Image.fromarray(merged_depth, mode="F").save(output_root / f"mono_depth_frame{stem}.tiff")
    _save_depth_visualization(output_root / f"depth_frame{stem}.png", merged_depth)
    _save_depth_visualization(output_root / f"mono_depth_frame{stem}.png", merged_depth)

    normals_world = _depth_normals_world(merged_depth, camera)
    plane_normal = np.asarray(plane_world[:3], dtype=np.float32)
    center_point = camera.center + camera.forward * float(np.median(clean_depth[repair_mask]))
    if float(np.dot(plane_normal, camera.center - center_point)) < 0:
        plane_normal *= -1.0
    normals_world[repair_mask] = plane_normal
    normals_camera = normals_world @ camera.w2c[:3, :3].T
    for prefix, normal in (
        ("depth_normal_world", normals_world),
        ("mono_normal_world", normals_world),
        ("mono_normal", normals_camera),
    ):
        np.save(output_root / f"{prefix}_frame{stem}.npy", normal.astype(np.float32))
        _save_rgb(output_root / f"{prefix}_frame{stem}.png", normal * 0.5 + 0.5)
    visibility = known_mask & (alpha >= 0.9)
    np.save(output_root / f"visibility_frame{stem}.npy", visibility)
    Image.fromarray(np.uint8(visibility) * 255, mode="L").save(
        output_root / f"visibility_frame{stem}.png"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage_root", type=Path, required=True)
    parser.add_argument("--diagnostics_dir", type=Path, required=True)
    parser.add_argument("--max_clean_artifact_score", type=float, default=0.20)
    parser.add_argument("--relative_depth_threshold", type=float, default=0.10)
    parser.add_argument("--max_fallback_rgb_disagreement", type=float, default=0.15)
    parser.add_argument("--min_depth_supported_fraction", type=float, default=0.85)
    parser.add_argument("--max_nearest_fill_fraction", type=float, default=0.02)
    parser.add_argument("--feather_radius", type=float, default=2.0)
    parser.add_argument("--replace_output", action="store_true")
    args = parser.parse_args()

    stage_root = args.stage_root.expanduser().resolve()
    diagnostics_root = args.diagnostics_dir.expanduser().resolve()
    output_root = stage_root / "select-gs-projective-merged"
    geometry_root = stage_root / "select-gs-projective-geometry"
    condition_root = stage_root / "select-gs-projective-condition"
    if args.replace_output:
        for path in (output_root, geometry_root, condition_root):
            if path.exists():
                shutil.rmtree(path)
    for path in (output_root, geometry_root, condition_root):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(path)
        path.mkdir(parents=True, exist_ok=True)

    stage_manifest = json.loads((stage_root / "artifact_guided_manifest.json").read_text())
    diagnostics = json.loads((diagnostics_root / "diagnostic_manifest.json").read_text())
    reports = []
    accepted_indices = []
    for view in stage_manifest["pseudo_views"]:
        index = int(view["pseudo_view_index"])
        stem = f"{index:06d}"
        camera = CameraGeometry.from_json(view["pseudo_camera"])
        known = _read_gray(stage_root / "select-gs" / f"mask_frame{stem}.png") > 127
        repair = ~known
        baseline = _read_rgb(
            stage_root / "select-gs" / f"ori_warp_frame{stem}.png",
            (camera.width, camera.height),
        )
        baseline_depth = np.asarray(
            Image.open(stage_root / "select-gs" / f"depth_frame{stem}.tiff"),
            dtype=np.float32,
        )
        alpha = np.load(stage_root / "select-gs" / f"alpha_{stem}.npy").astype(np.float32)
        plane = np.asarray(view["clean_support_plane"]["plane_world"], dtype=np.float64)
        clean_depth, clean_valid = render_support_plane_depth(camera, plane, repair)
        points, pixels = backproject_depth(clean_depth, clean_valid, camera, stride=1)
        pixel_xy = np.rint(pixels).astype(np.int64)

        color_rows = []
        semantic_rows = []
        depth_rows = []
        score_rows = []
        support_metrics = []
        for support in view["clean_reference_support"]:
            record = diagnostics["views"][int(support["view_index"])]
            colors, semantic_valid, depth_valid, scores, metrics = _sample_reference(
                points,
                record,
                diagnostics_root,
                float(args.max_clean_artifact_score),
                float(args.relative_depth_threshold),
            )
            color_rows.append(colors)
            semantic_rows.append(semantic_valid)
            depth_rows.append(depth_valid)
            score_rows.append(scores)
            support_metrics.append(metrics)

        fused, accepted, fusion = blend_projective_reference_colors(
            np.stack(color_rows),
            np.stack(semantic_rows),
            np.stack(depth_rows),
            np.stack(score_rows),
            max_fallback_rgb_disagreement=float(args.max_fallback_rgb_disagreement),
        )
        patch = np.zeros_like(baseline)
        patch_valid = np.zeros(repair.shape, dtype=bool)
        patch[pixel_xy[accepted, 1], pixel_xy[accepted, 0]] = fused[accepted]
        patch_valid[pixel_xy[accepted, 1], pixel_xy[accepted, 0]] = True
        missing = repair & ~patch_valid
        fill_fraction = float(missing.sum() / max(repair.sum(), 1))

        # Real train-view reprojections are stronger conditions than the
        # current GS render. Keep them visible to See3D and expose only the
        # residual coverage hole as unknown.
        condition_known = known | patch_valid
        condition = baseline.copy()
        condition[patch_valid] = patch[patch_valid]
        condition[~condition_known] = 0.0
        _save_rgb(condition_root / f"ori_warp_frame{stem}.png", baseline)
        _save_rgb(condition_root / f"warp_frame{stem}.png", condition)
        Image.fromarray(np.uint8(condition_known) * 255, mode="L").save(
            condition_root / f"mask_frame{stem}.png"
        )
        Image.fromarray(np.uint8(patch_valid) * 255, mode="L").save(
            condition_root / f"projective_support_frame{stem}.png"
        )

        patch = _nearest_fill(patch, patch_valid)

        distance = cv2.distanceTransform(np.uint8(repair), cv2.DIST_L2, 3)
        blend_weight = np.clip(distance / max(float(args.feather_radius), 1e-6), 0.0, 1.0)
        merged = baseline.copy()
        merged[repair] = (
            patch[repair] * blend_weight[repair, None]
            + baseline[repair] * (1.0 - blend_weight[repair, None])
        )
        output_path = output_root / f"predict_warp_frame{stem}.png"
        _save_rgb(output_path, merged)

        accepted_view = (
            fusion["depth_supported_fraction"] >= float(args.min_depth_supported_fraction)
            and fill_fraction <= float(args.max_nearest_fill_fraction)
        )
        if accepted_view:
            accepted_indices.append(index)
        # The downstream quality gate evaluates both accepted and rejected
        # projective candidates and therefore needs a complete geometry
        # bundle for every view.  Its accepted list remains the authority for
        # whether any bundle can be applied.
        _write_geometry_bundle(
            geometry_root,
            index,
            merged,
            baseline_depth,
            clean_depth,
            repair,
            known,
            alpha,
            camera,
            plane,
        )
        source_kind = np.zeros(repair.shape, dtype=np.uint8)
        source_kind[pixel_xy[:, 1], pixel_xy[:, 0]] = fusion.pop("source_kind")
        Image.fromarray(source_kind * 100, mode="L").save(
            output_root / f"support_kind_frame{stem}.png"
        )
        reports.append(
            {
                "index": index,
                "classification": view["classification"],
                "accepted": accepted_view,
                "output_path": str(output_path),
                "repair_fraction": float(repair.mean()),
                "nearest_fill_fraction": fill_fraction,
                "projective_support_fraction_of_repair": float(
                    patch_valid.sum() / max(repair.sum(), 1)
                ),
                "residual_hole_fraction": float(missing.mean()),
                "see3d_needed": bool(missing.mean() >= 0.001),
                "protected_pixels_exact": bool(
                    np.array_equal(
                        np.uint8(np.clip(merged[known], 0.0, 1.0) * 255.0 + 0.5),
                        np.uint8(np.clip(baseline[known], 0.0, 1.0) * 255.0 + 0.5),
                    )
                ),
                "fusion": fusion,
                "support_views": support_metrics,
            }
        )

    report = {
        "version": 1,
        "mode": "clean_support_plane_projective_rgb_repair",
        "accepted_indices": accepted_indices,
        "views": reports,
        "geometry_root": str(geometry_root),
        "see3d_condition_root": str(condition_root),
    }
    (stage_root / "projective_repair_report.json").write_text(
        json.dumps(_json_ready(report), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_ready(report), indent=2))


if __name__ == "__main__":
    main()
