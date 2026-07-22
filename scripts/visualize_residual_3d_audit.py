#!/usr/bin/env python3
"""Visualize residual-to-3D evidence for frozen 2DGS hard cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
from types import SimpleNamespace
from typing import Any

import cv2
import matplotlib
import numpy as np
from PIL import Image
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
GS_ROOT = REPO_ROOT / "2d-gaussian-splatting"
for import_root in (REPO_ROOT, GS_ROOT, REPO_ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from artifact_guided_repair import ArtifactROI3D, project_points  # noqa: E402
from causal_2dgs_repair import _LazyCameraStore, _camera_gt  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import GaussianModel  # noqa: E402


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 127


def _save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    Image.fromarray(np.uint8(clipped * 255.0 + 0.5), mode="RGB").save(path)


def _depth_preview(depth: np.ndarray) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(values) & (values > 1e-6)
    result = np.zeros_like(values, dtype=np.float32)
    if np.any(valid):
        lower, upper = np.quantile(values[valid], [0.02, 0.98])
        result[valid] = np.clip((values[valid] - lower) / max(float(upper - lower), 1e-6), 0.0, 1.0)
    return result


def _heatmap(values: np.ndarray, *, upper: float, cmap: int = cv2.COLORMAP_MAGMA) -> np.ndarray:
    normalized = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=upper, neginf=0.0)
    image = np.uint8(np.clip(normalized / max(float(upper), 1e-6), 0.0, 1.0) * 255.0)
    colored = cv2.applyColorMap(image, cmap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _overlay_contours(
    image: np.ndarray,
    masks: list[tuple[np.ndarray, tuple[int, int, int]]],
) -> np.ndarray:
    result = np.uint8(np.clip(image, 0.0, 1.0) * 255.0).copy()
    for mask, color in masks:
        contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2, lineType=cv2.LINE_AA)
    return result


def _overlay_points(image: np.ndarray, pixels: np.ndarray, inside: np.ndarray) -> np.ndarray:
    result = np.uint8(np.clip(image, 0.0, 1.0) * 255.0).copy()
    rounded = np.rint(pixels[np.asarray(inside, dtype=bool)]).astype(np.int64)
    height, width = result.shape[:2]
    for horizontal, vertical in rounded:
        if 0 <= horizontal < width and 0 <= vertical < height:
            cv2.circle(result, (int(horizontal), int(vertical)), 1, (255, 70, 40), -1, lineType=cv2.LINE_AA)
    return result


def _parse_neighbor_names(name: str, available: set[str]) -> list[str]:
    match = re.fullmatch(r"(seq\d+)__frame(\d+)", name)
    if match is None:
        return []
    sequence, frame_text = match.groups()
    frame_number = int(frame_text)
    candidates = [
        f"{sequence}__frame{frame_number - 1:05d}",
        f"{sequence}__frame{frame_number + 1:05d}",
    ]
    return [candidate for candidate in candidates if candidate in available]


def _backproject_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    geometry: Any,
) -> np.ndarray:
    vertical, horizontal = np.where(mask)
    sampled_depth = depth[vertical, horizontal].astype(np.float64)
    valid = np.isfinite(sampled_depth) & (sampled_depth > 1e-6)
    horizontal = horizontal[valid].astype(np.float64)
    vertical = vertical[valid].astype(np.float64)
    sampled_depth = sampled_depth[valid]
    if not len(sampled_depth):
        return np.empty((0, 3), dtype=np.float64)
    camera_points = np.stack(
        [
            (horizontal - geometry.cx) * sampled_depth / geometry.fx,
            (vertical - geometry.cy) * sampled_depth / geometry.fy,
            sampled_depth,
        ],
        axis=1,
    )
    homogeneous = np.concatenate([camera_points, np.ones((len(camera_points), 1))], axis=1)
    return (geometry.c2w @ homogeneous.T).T[:, :3]


def _render_maps(camera: Any, gaussians: GaussianModel, pipe: Any, background: torch.Tensor) -> dict[str, np.ndarray]:
    with torch.no_grad():
        package = render(camera, gaussians, pipe, background)
    return {
        "render": package["render"].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32),
        "alpha": package["rend_alpha"][0].detach().cpu().numpy().astype(np.float32),
        "surf_depth": package["surf_depth"][0].detach().cpu().numpy().astype(np.float32),
        "expected_depth": package["rend_depth"][0].detach().cpu().numpy().astype(np.float32),
        "median_depth": package.get("rend_depth_median", package["rend_depth"])[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
    }


def _depth_consistency(
    source_points: np.ndarray,
    target_geometry: Any,
    target_maps: dict[str, np.ndarray],
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    pixels, projected_depth, inside = project_points(source_points, target_geometry)
    rounded = np.rint(pixels).astype(np.int64)
    height, width = target_maps["surf_depth"].shape
    inside &= (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    valid_indices = np.flatnonzero(inside)
    if not len(valid_indices):
        return (
            {
                "projected_point_count": int(len(source_points)),
                "inside_count": 0,
                "model_depth_valid_count": 0,
                "relative_depth_median": None,
                "within_10_percent_fraction": 0.0,
            },
            pixels,
            inside,
        )
    observed_depth = target_maps["surf_depth"][rounded[valid_indices, 1], rounded[valid_indices, 0]]
    observed_alpha = target_maps["alpha"][rounded[valid_indices, 1], rounded[valid_indices, 0]]
    valid_depth = np.isfinite(observed_depth) & (observed_depth > 1e-6) & (observed_alpha >= 0.70)
    relative = np.full(len(valid_indices), np.nan, dtype=np.float64)
    relative[valid_depth] = np.abs(projected_depth[valid_indices][valid_depth] - observed_depth[valid_depth]) / np.maximum(
        projected_depth[valid_indices][valid_depth], 1e-6
    )
    return (
        {
            "projected_point_count": int(len(source_points)),
            "inside_count": int(len(valid_indices)),
            "model_depth_valid_count": int(valid_depth.sum()),
            "relative_depth_median": None if not np.any(valid_depth) else float(np.nanmedian(relative)),
            "within_10_percent_fraction": 0.0
            if not np.any(valid_depth)
            else float((relative[valid_depth] <= 0.10).mean()),
        },
        pixels,
        inside,
    )


def _roi_depth_difference(
    roi: ArtifactROI3D,
    geometry: Any,
    maps: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, float | int]]:
    pixels, point_depth, inside = project_points(roi.points, geometry)
    rounded = np.rint(pixels).astype(np.int64)
    height, width = maps["surf_depth"].shape
    inside &= (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    values = np.full((height, width), np.nan, dtype=np.float32)
    indices = np.flatnonzero(inside)
    if not len(indices):
        return values, {"point_count": int(len(roi.points)), "inside_count": 0, "depth_valid_count": 0}
    model_depth = maps["surf_depth"][rounded[indices, 1], rounded[indices, 0]]
    model_alpha = maps["alpha"][rounded[indices, 1], rounded[indices, 0]]
    valid = np.isfinite(model_depth) & (model_depth > 1e-6) & (model_alpha >= 0.70)
    if np.any(valid):
        relative = (point_depth[indices][valid] - model_depth[valid]) / np.maximum(point_depth[indices][valid], 1e-6)
        for source_index, value in zip(indices[valid], relative):
            horizontal, vertical = rounded[source_index]
            previous = values[vertical, horizontal]
            if not np.isfinite(previous) or abs(value) > abs(previous):
                values[vertical, horizontal] = float(value)
    return values, {
        "point_count": int(len(roi.points)),
        "inside_count": int(len(indices)),
        "depth_valid_count": int(valid.sum()),
        "independent_depth_median": float(np.median(point_depth[indices])),
        "model_depth_median": None if not np.any(valid) else float(np.median(model_depth[valid])),
        "relative_gap_median": None
        if not np.any(valid)
        else float(np.median(np.abs(point_depth[indices][valid] - model_depth[valid]) / np.maximum(point_depth[indices][valid], 1e-6))),
        "independent_behind_model_fraction": 0.0
        if not np.any(valid)
        else float((point_depth[indices][valid] > model_depth[valid]).mean()),
        "within_10_percent_fraction": 0.0
        if not np.any(valid)
        else float(
            (np.abs(point_depth[indices][valid] - model_depth[valid]) / np.maximum(point_depth[indices][valid], 1e-6) <= 0.10).mean()
        ),
    }


def _roi_masks(
    rois: list[tuple[int, str, ArtifactROI3D]],
    geometry: Any,
    maps: dict[str, np.ndarray],
) -> tuple[list[tuple[np.ndarray, tuple[int, int, int]]], list[tuple[np.ndarray, tuple[int, int, int]]], dict[str, dict[str, float | int]], np.ndarray]:
    raw_entries: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    visible_entries: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    summary: dict[str, dict[str, float | int]] = {}
    difference = np.full_like(maps["surf_depth"], np.nan, dtype=np.float32)
    colors = ((70, 220, 70), (55, 170, 255), (255, 210, 50))
    for ordinal, (cluster_id, _, roi) in enumerate(rois):
        color = colors[ordinal % len(colors)]
        raw_mask = roi.project_mask(geometry)
        visible_mask = roi.project_mask(geometry, reference_depth=maps["surf_depth"])
        raw_entries.append((raw_mask, color))
        visible_entries.append((visible_mask, (255, 55, 55)))
        relative_map, row = _roi_depth_difference(roi, geometry, maps)
        summary[str(cluster_id)] = {
            **row,
            "raw_pixel_count": int(raw_mask.sum()),
            "visible_pixel_count": int(visible_mask.sum()),
        }
        finite_relative = relative_map[np.isfinite(relative_map)]
        if len(finite_relative):
            representative_relative = float(np.median(finite_relative))
            merge = raw_mask & (~np.isfinite(difference) | (abs(representative_relative) > np.abs(difference)))
            difference[merge] = representative_relative
    return raw_entries, visible_entries, summary, difference


def _load_rois(report: dict[str, Any], target_name: str) -> list[tuple[int, str, ArtifactROI3D]]:
    result: list[tuple[int, str, ArtifactROI3D]] = []
    for cluster_text, evidence in report.get("surface_evidence", {}).items():
        if str(evidence.get("target_name")) != target_name or not bool(evidence.get("surface_known")):
            continue
        roi_path = evidence.get("roi_path")
        if not roi_path:
            continue
        result.append((int(cluster_text), str(evidence.get("surface_source", "unknown")), ArtifactROI3D.load(Path(roi_path))))
    return sorted(result, key=lambda item: item[0])


def _render_neighbor(
    name: str,
    cameras: _LazyCameraStore,
    gaussians: GaussianModel,
    pipe: Any,
    background: torch.Tensor,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    camera = cameras[name]
    return _camera_gt(camera), _render_maps(camera, gaussians, pipe, background)


def _case_panel(
    *,
    name: str,
    target_geometry: Any,
    target_gt: np.ndarray,
    target_maps: dict[str, np.ndarray],
    anomaly_mask: np.ndarray,
    static_mask: np.ndarray,
    rois: list[tuple[int, str, ArtifactROI3D]],
    neighbors: list[tuple[str, np.ndarray, dict[str, np.ndarray], dict[str, Any], np.ndarray, np.ndarray]],
    output_path: Path,
) -> dict[str, Any]:
    residual = np.mean(np.abs(target_gt - target_maps["render"]), axis=2)
    expected = target_maps["expected_depth"]
    median = target_maps["median_depth"]
    ambiguity = np.abs(expected - median) / np.maximum(np.maximum(expected, median), 1e-6)
    model_3d_mask = anomaly_mask & static_mask & (target_maps["alpha"] >= 0.70) & np.isfinite(target_maps["surf_depth"]) & (target_maps["surf_depth"] > 1e-6)
    raw_entries, visible_entries, roi_summary, roi_difference = _roi_masks(rois, target_geometry, target_maps)
    target_overlay = _overlay_contours(target_maps["render"], [(anomaly_mask, (255, 70, 40))])
    residual_3d = np.where(model_3d_mask, residual, np.nan)
    figure, axes = plt.subplots(3, 4, figsize=(20, 13), constrained_layout=True)
    top_panels = [
        (target_gt, "Ground truth"),
        (target_overlay, "G4 render + anomaly rays"),
        (_heatmap(residual, upper=0.50), f"Residual |GT-G4|, mean={residual.mean():.3f}"),
        (target_maps["alpha"], f"G4 alpha, anomaly alpha={target_maps['alpha'][anomaly_mask].mean() if np.any(anomaly_mask) else 0.0:.3f}", "viridis", 0.0, 1.0),
        (_depth_preview(target_maps["surf_depth"]), "Current G4 surf depth", "viridis", 0.0, 1.0),
        (_heatmap(ambiguity, upper=0.50), f"Expected/median depth ambiguity, anomaly={ambiguity[anomaly_mask].mean() if np.any(anomaly_mask) else 0.0:.3f}"),
        (_heatmap(residual_3d, upper=0.50), f"Residual with usable current depth: {model_3d_mask.mean():.2%}"),
        (_overlay_contours(target_gt, [(model_3d_mask, (55, 170, 255)), *raw_entries, *visible_entries]), "blue=current-depth residual; green/cyan=independent ROI; red=depth-visible"),
    ]
    for axis, panel in zip(axes.flat[:8], top_panels):
        image, title, *style = panel
        options: dict[str, Any] = {}
        if style:
            options["cmap"] = style[0]
            if len(style) > 1:
                options["vmin"] = style[1]
            if len(style) > 2:
                options["vmax"] = style[2]
        axis.imshow(image, **options)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    if rois:
        roi_overlay = _overlay_contours(target_gt, [*raw_entries, *visible_entries])
        diff_preview = np.clip((np.nan_to_num(roi_difference, nan=0.0) + 1.0) * 0.5, 0.0, 1.0)
        panels = [
            (roi_overlay, f"Independent 3D ROI in target ({len(rois)} clusters)"),
            (diff_preview, "orange: independent surface lies behind shallow G4 layer", "coolwarm", 0.0, 1.0),
        ]
        for neighbor_name, neighbor_gt, neighbor_maps, _, _, _ in neighbors[:2]:
            neighbor_raw, neighbor_visible, neighbor_summary, _ = _roi_masks(rois, neighbor_maps["geometry"], neighbor_maps)
            raw_total = sum(int(row[0].sum()) for row in neighbor_raw)
            visible_total = sum(int(row[0].sum()) for row in neighbor_visible)
            panels.append(
                (
                    _overlay_contours(neighbor_gt, [*neighbor_raw, *neighbor_visible]),
                    f"{neighbor_name}: ROI reprojection raw={raw_total}, visible={visible_total}",
                )
            )
            roi_summary[f"neighbor_{neighbor_name}"] = neighbor_summary
    else:
        panels = []
        for neighbor_name, neighbor_gt, neighbor_maps, neighbor_meta, pixels, inside in neighbors[:2]:
            panels.append(
                (
                    _overlay_points(neighbor_gt, pixels, inside),
                    f"{neighbor_name}: current-depth projected residual ({neighbor_meta['projection']['inside_count']} inside)",
                )
            )
        while len(panels) < 2:
            panels.append((np.zeros_like(target_gt), "No adjacent real frame available"))
        panels.append((_heatmap(residual_3d, upper=0.50), "Current-depth residual only; no independent 3D surface"))
        panels.append((_overlay_contours(target_gt, [(anomaly_mask, (255, 70, 40))]), "No MAtCha/track ROI admitted: retain as photometric triage"))
    while len(panels) < 4:
        panels.append((np.zeros_like(target_gt), "No panel"))
    for axis, panel in zip(axes.flat[8:], panels[:4]):
        image, title, *style = panel
        options = {}
        if style:
            options["cmap"] = style[0]
            if len(style) > 1:
                options["vmin"] = style[1]
            if len(style) > 2:
                options["vmax"] = style[2]
        axis.imshow(image, **options)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    figure.suptitle(
        f"{name}: residual-to-3D audit | anomaly={anomaly_mask.mean():.2%}, "
        f"current-depth eligible={model_3d_mask.mean():.2%}, independent ROI clusters={len(rois)}",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return {
        "residual_mean": float(residual.mean()),
        "anomaly_pixel_count": int(anomaly_mask.sum()),
        "anomaly_fraction": float(anomaly_mask.mean()),
        "anomaly_alpha_mean": float(target_maps["alpha"][anomaly_mask].mean()) if np.any(anomaly_mask) else None,
        "anomaly_depth_ambiguity_mean": float(ambiguity[anomaly_mask].mean()) if np.any(anomaly_mask) else None,
        "current_depth_eligible_pixel_count": int(model_3d_mask.sum()),
        "current_depth_eligible_fraction": float(model_3d_mask.mean()),
        "roi_clusters": roi_summary,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--causal-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--white-background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    causal_root = args.causal_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not args.replace_output:
            raise FileExistsError(f"Output is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((causal_root / "causal_repair_report.json").read_text(encoding="utf-8"))
    target_names = [str(name) for name in report["target_view_names"]]
    source_path = args.source_path.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    point_cloud = model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    if not point_cloud.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {point_cloud}")
    gaussians = GaussianModel(3)
    gaussians.load_ply(str(point_cloud))
    cameras = _LazyCameraStore.from_colmap(source_path, requested_resolution=int(args.resolution), data_device="cpu")
    for name in target_names:
        geometry = cameras.geometries[name]
        if (geometry.height, geometry.width) != (360, 640):
            raise ValueError(
                f"Expected shared640 diagnostics but got {(geometry.height, geometry.width)} for {name}; "
                "check --resolution against the frozen render cache"
            )
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, depth_ratio=0.0)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=gaussians.get_xyz.device,
    )
    summary: dict[str, Any] = {
        "protocol": "frozen_g4_current_depth_plus_independent_matcha_track_roi_v1",
        "source_path": str(source_path),
        "model_path": str(model_path),
        "iteration": int(args.iteration),
        "target_count": len(target_names),
        "cases": {},
    }
    rendered_neighbors: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    available_names = set(cameras)
    for ordinal, name in enumerate(target_names, start=1):
        camera = cameras[name]
        target_gt = _camera_gt(camera)
        target_maps = _render_maps(camera, gaussians, pipe, background)
        target_maps["geometry"] = cameras.geometries[name]
        target_dir = causal_root / "anomaly_rays" / f"{ordinal:03d}_{_safe_name(name)}"
        anomaly_mask = _read_mask(target_dir / "anomaly_rays.png")
        static_mask = _read_mask(target_dir / "static_mask.png")
        if anomaly_mask.shape != target_maps["surf_depth"].shape:
            raise ValueError(f"Anomaly map shape mismatch for {name}: {anomaly_mask.shape}")
        rois = _load_rois(report, name)
        model_3d_mask = anomaly_mask & static_mask & (target_maps["alpha"] >= 0.70) & (target_maps["surf_depth"] > 1e-6)
        model_points = _backproject_mask(model_3d_mask, target_maps["surf_depth"], cameras.geometries[name])
        neighbor_rows = []
        for neighbor_name in _parse_neighbor_names(name, available_names):
            if neighbor_name not in rendered_neighbors:
                rendered_neighbors[neighbor_name] = _render_neighbor(neighbor_name, cameras, gaussians, pipe, background)
                rendered_neighbors[neighbor_name][1]["geometry"] = cameras.geometries[neighbor_name]
            neighbor_gt, neighbor_maps = rendered_neighbors[neighbor_name]
            projection, pixels, inside = _depth_consistency(model_points, cameras.geometries[neighbor_name], neighbor_maps)
            neighbor_rows.append(
                (
                    neighbor_name,
                    neighbor_gt,
                    neighbor_maps,
                    {"target_geometry": cameras.geometries[name], "projection": projection},
                    pixels,
                    inside,
                )
            )
        case_dir = output / _safe_name(name)
        case_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            case_dir / "current_g4_maps.npz",
            render=target_maps["render"],
            alpha=target_maps["alpha"],
            surf_depth=target_maps["surf_depth"],
            expected_depth=target_maps["expected_depth"],
            median_depth=target_maps["median_depth"],
            anomaly_mask=anomaly_mask,
            static_mask=static_mask,
            gt=target_gt,
        )
        _save_rgb(case_dir / "current_render.png", target_maps["render"])
        row = _case_panel(
            name=name,
            target_geometry=cameras.geometries[name],
            target_gt=target_gt,
            target_maps=target_maps,
            anomaly_mask=anomaly_mask,
            static_mask=static_mask,
            rois=rois,
            neighbors=neighbor_rows,
            output_path=case_dir / "residual_3d_panel.png",
        )
        row["independent_roi_sources"] = [
            {"cluster_id": int(cluster_id), "source": source, "point_count": int(len(roi.points))}
            for cluster_id, source, roi in rois
        ]
        row["neighbor_model_depth_reprojection"] = {
            neighbor_name: metadata["projection"]
            for neighbor_name, _, _, metadata, _, _ in neighbor_rows
        }
        summary["cases"][name] = row
        print(
            f"[{ordinal}/{len(target_names)}] {name}: anomaly={row['anomaly_fraction']:.3%}, "
            f"current_depth={row['current_depth_eligible_fraction']:.3%}, rois={len(rois)}",
            flush=True,
        )
    (output / "residual_3d_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
