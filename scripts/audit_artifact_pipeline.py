#!/usr/bin/env python3
"""First-principles audit and large-scale visualization for artifact repair.

The script is intentionally read-only with respect to reconstruction outputs. It
checks dataset invariants, summarizes dense train diagnostics, compares two
frozen models on identical cameras, and audits whether repair decisions are
supported by independent multi-view evidence at the same spatial scope.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mast3r.colmap.read_write_model import (  # noqa: E402
    read_cameras_binary,
    read_images_binary,
)
from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def quantiles(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: None for key in ("min", "p25", "p50", "p75", "p90", "p95", "p99", "max")}
    levels = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    keys = ["min", "p25", "p50", "p75", "p90", "p95", "p99", "max"]
    return {key: float(value) for key, value in zip(keys, np.quantile(array, levels))}


def read_bgr(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None and image.size != size:
            # This matches the repository's PILtoTorch loader.
            image = image.resize(size, Image.Resampling.NEAREST)
        rgb = np.asarray(image, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if image.shape[1::-1] == size:
        return image
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def heatmap(values: np.ndarray, *, scale: float | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values)
    if scale is None:
        scale = float(np.quantile(values[valid], 0.98)) if np.any(valid) else 1.0
    normalized = np.zeros(values.shape, dtype=np.uint8)
    if scale > 1e-12:
        normalized[valid] = np.uint8(np.clip(values[valid] / scale, 0.0, 1.0) * 255.0)
    result = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    result[~valid] = 0
    return result


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 1e-6)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        log_depth = np.log(np.maximum(depth, 1e-6))
        low, high = np.quantile(log_depth[valid], [0.02, 0.98])
        high = max(float(high), float(low) + 1e-6)
        normalized[valid] = np.uint8(
            np.clip((log_depth[valid] - low) / (high - low), 0.0, 1.0) * 255.0
        )
    result = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    result[~valid] = 0
    return result


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = image.copy()
    mask = np.asarray(mask, dtype=bool)
    tint = np.empty_like(output)
    tint[:] = color
    output[mask] = cv2.addWeighted(output, 0.45, tint, 0.55, 0.0)[mask]
    return output


def make_tile(image: np.ndarray, title: str, width: int = 240, height: int = 135) -> np.ndarray:
    image = resize(image, (width, height))
    canvas = np.full((height + 32, width, 3), 245, dtype=np.uint8)
    canvas[32:] = image
    cv2.putText(
        canvas,
        title[:44],
        (5, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return canvas


def save_rows(rows: list[tuple[str, list[np.ndarray]]], root: Path, stem: str, rows_per_sheet: int = 10) -> list[str]:
    if not rows:
        return []
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for sheet_index, offset in enumerate(range(0, len(rows), rows_per_sheet), start=1):
        batch = rows[offset : offset + rows_per_sheet]
        tile_height, tile_width = batch[0][1][0].shape[:2]
        columns = max(len(tiles) for _, tiles in batch)
        row_header = 28
        sheet = np.full(
            ((tile_height + row_header) * len(batch), tile_width * columns, 3),
            245,
            dtype=np.uint8,
        )
        for row_index, (label, tiles) in enumerate(batch):
            y = row_index * (tile_height + row_header)
            cv2.putText(
                sheet,
                label[:180],
                (6, y + 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (15, 15, 15),
                1,
                cv2.LINE_AA,
            )
            for column, tile in enumerate(tiles):
                sheet[y + row_header : y + row_header + tile_height, column * tile_width : (column + 1) * tile_width] = tile
        path = root / f"{stem}_{sheet_index:02d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths.append(str(path))
    return paths


class DiagnosticSet:
    def __init__(self, root: Path):
        self.root = root
        self.manifest = read_json(root / "diagnostic_manifest.json")
        if self.manifest is None:
            raise FileNotFoundError(root / "diagnostic_manifest.json")
        self.records = self.manifest["views"]
        self.by_name = {record["image_name"]: record for record in self.records}

    def arrays(self, record: dict[str, Any]) -> dict[str, np.ndarray]:
        files = record["files"]
        render = read_bgr(self.root / files["render"])
        height, width = render.shape[:2]
        return {
            "gt": read_bgr(Path(record["image_path"]), (width, height)),
            "render": render,
            "depth": np.load(self.root / files["depth"]).astype(np.float32),
            "labels": read_gray(self.root / files["labels"]),
            "score": read_gray(self.root / files["score"]).astype(np.float32) / 255.0,
            "semantic": read_gray(self.root / files["semantic"]) > 127,
        }


def diagnostic_tiles(dataset: DiagnosticSet, record: dict[str, Any]) -> list[np.ndarray]:
    data = dataset.arrays(record)
    error = np.mean(np.abs(data["render"].astype(np.float32) - data["gt"].astype(np.float32)), axis=2) / 255.0
    component = overlay_mask(data["render"], data["labels"] > 0, (32, 32, 245))
    semantic = overlay_mask(data["gt"], ~data["semantic"], (220, 32, 220))
    return [
        make_tile(data["gt"], "GT"),
        make_tile(data["render"], "render"),
        make_tile(heatmap(error, scale=0.35), f"|RGB error| mean={error.mean():.3f}"),
        make_tile(colorize_depth(data["depth"]), "rendered log depth"),
        make_tile(component, "detected components"),
        make_tile(semantic, "excluded semantics"),
    ]


def compare_tiles(base: DiagnosticSet, other: DiagnosticSet, name: str) -> list[np.ndarray]:
    first = base.arrays(base.by_name[name])
    second = other.arrays(other.by_name[name])
    base_error = np.mean(np.abs(first["render"].astype(np.float32) - first["gt"].astype(np.float32)), axis=2) / 255.0
    other_error = np.mean(np.abs(second["render"].astype(np.float32) - second["gt"].astype(np.float32)), axis=2) / 255.0
    return [
        make_tile(first["gt"], "GT"),
        make_tile(first["render"], "retained_v2"),
        make_tile(second["render"], "comparison"),
        make_tile(heatmap(base_error, scale=0.35), f"base error={base_error.mean():.3f}"),
        make_tile(heatmap(other_error, scale=0.35), f"other error={other_error.mean():.3f}"),
        make_tile(heatmap(np.abs(other_error - base_error), scale=0.20), "absolute error change"),
    ]


def camera_intrinsics(camera: Any) -> tuple[float, float, float, float]:
    params = np.asarray(camera.params, dtype=np.float64)
    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        return float(params[0]), float(params[0]), float(params[1]), float(params[2])
    if camera.model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        return tuple(float(value) for value in params[:4])  # type: ignore[return-value]
    raise ValueError(f"Unsupported camera model for audit: {camera.model}")


def audit_dataset_invariants(dataset: DiagnosticSet) -> dict[str, Any]:
    source = Path(dataset.manifest["source_path"])
    images = read_images_binary(str(source / "sparse/0/images.bin"))
    cameras = read_cameras_binary(str(source / "sparse/0/cameras.bin"))
    colmap_by_name = {image.name: image for image in images.values()}
    mask_lookup = CambridgeMaskLookup(
        source,
        Path(dataset.manifest["mask_pickle"]),
        list(dataset.manifest["semantic_mask_indices"]),
    )

    unresolved_source = []
    camera_missing = []
    semantic_mismatch = []
    max_w2c_error = 0.0
    max_intrinsic_error = 0.0
    for record in dataset.records:
        image_name = record["image_name"]
        staged_candidates = list((source / "images").glob(f"{image_name}.*"))
        if len(staged_candidates) != 1 or staged_candidates[0].resolve() != Path(record["image_path"]).resolve():
            unresolved_source.append(image_name)

        colmap = colmap_by_name.get(f"{image_name}.png")
        if colmap is None:
            camera_missing.append(image_name)
            continue
        camera = cameras[colmap.camera_id]
        expected_w2c = np.eye(4, dtype=np.float64)
        expected_w2c[:3, :3] = colmap.qvec2rotmat()
        expected_w2c[:3, 3] = colmap.tvec
        actual_camera = record["camera"]
        max_w2c_error = max(
            max_w2c_error,
            float(np.max(np.abs(expected_w2c - np.asarray(actual_camera["w2c"], dtype=np.float64)))),
        )
        fx, fy, cx, cy = camera_intrinsics(camera)
        sx = float(actual_camera["width"]) / float(camera.width)
        sy = float(actual_camera["height"]) / float(camera.height)
        expected_intrinsics = np.asarray([fx * sx, fy * sy, cx * sx, cy * sy])
        actual_intrinsics = np.asarray(
            [actual_camera["fx"], actual_camera["fy"], actual_camera["cx"], actual_camera["cy"]],
            dtype=np.float64,
        )
        max_intrinsic_error = max(max_intrinsic_error, float(np.max(np.abs(expected_intrinsics - actual_intrinsics))))

        semantic_path = dataset.root / record["files"]["semantic"]
        saved = read_gray(semantic_path) > 127
        expected = mask_lookup.get_mask(
            image_name,
            saved.shape,
            torch.device("cpu"),
        ).numpy()
        if not np.array_equal(saved, expected):
            semantic_mismatch.append(
                {"image_name": image_name, "different_fraction": float(np.mean(saved != expected))}
            )

    return {
        "view_count": len(dataset.records),
        "staged_source_mismatch_count": len(unresolved_source),
        "staged_source_mismatch_examples": unresolved_source[:10],
        "missing_colmap_camera_count": len(camera_missing),
        "missing_colmap_camera_examples": camera_missing[:10],
        "semantic_mask_mismatch_count": len(semantic_mismatch),
        "semantic_mask_mismatch_examples": semantic_mismatch[:10],
        "max_w2c_absolute_error": max_w2c_error,
        "max_scaled_intrinsic_absolute_error": max_intrinsic_error,
        "passed": not unresolved_source
        and not camera_missing
        and not semantic_mismatch
        and max_w2c_error < 1e-5
        and max_intrinsic_error < 1e-4,
    }


def summarize_diagnostics(dataset: DiagnosticSet) -> dict[str, Any]:
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in dataset.records:
        by_sequence[record["image_name"].split("__", 1)[0]].append(record)

    def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
        artifact = [record["summary"]["artifact_fraction"] for record in records]
        largest = [record["summary"]["largest_component_fraction"] for record in records]
        residual = [record["summary"]["residual_mean"] for record in records]
        return {
            "count": len(records),
            "artifact_fraction": quantiles(artifact),
            "largest_component_fraction": quantiles(largest),
            "residual_mean": quantiles(residual),
            "views_artifact_gt_10pct": int(np.sum(np.asarray(artifact) > 0.10)),
            "views_artifact_gt_25pct": int(np.sum(np.asarray(artifact) > 0.25)),
        }

    component_count = sum(len(record["components"]) for record in dataset.records)
    no_reference_positive = sum(
        component["no_reference_invalid_fraction"] > 0.01
        for record in dataset.records
        for component in record["components"]
    )
    return {
        "global": summarize(dataset.records),
        "by_sequence": {sequence: summarize(records) for sequence, records in sorted(by_sequence.items())},
        "component_count": component_count,
        "components_with_no_reference_invalid_gt_1pct": no_reference_positive,
    }


def make_train_sheets(dataset: DiagnosticSet, output: Path, top_k: int) -> dict[str, list[str]]:
    ranked = sorted(dataset.records, key=lambda record: record["summary"]["artifact_fraction"], reverse=True)
    global_rows = []
    for record in ranked[:top_k]:
        summary = record["summary"]
        global_rows.append(
            (
                f"{record['index']:03d} {record['image_name']} | artifact={summary['artifact_fraction']:.3f} "
                f"largest={summary['largest_component_fraction']:.3f} residual={summary['residual_mean']:.3f}",
                diagnostic_tiles(dataset, record),
            )
        )

    per_sequence = defaultdict(list)
    for record in dataset.records:
        per_sequence[record["image_name"].split("__", 1)[0]].append(record)
    sequence_rows = []
    for sequence, records in sorted(per_sequence.items()):
        worst = sorted(records, key=lambda record: record["summary"]["artifact_fraction"], reverse=True)[:2]
        for record in worst:
            summary = record["summary"]
            sequence_rows.append(
                (
                    f"{sequence} | {record['image_name']} | artifact={summary['artifact_fraction']:.3f} "
                    f"largest={summary['largest_component_fraction']:.3f}",
                    diagnostic_tiles(dataset, record),
                )
            )
    return {
        "global_worst": save_rows(global_rows, output, "train_global_worst"),
        "per_sequence_worst": save_rows(sequence_rows, output, "train_per_sequence_worst"),
    }


def compare_diagnostics(base: DiagnosticSet, other: DiagnosticSet, output: Path, top_k: int) -> dict[str, Any]:
    common = sorted(set(base.by_name) & set(other.by_name))
    records = []
    for name in common:
        first = base.by_name[name]["summary"]
        second = other.by_name[name]["summary"]
        records.append(
            {
                "image_name": name,
                "base_artifact": float(first["artifact_fraction"]),
                "other_artifact": float(second["artifact_fraction"]),
                "artifact_delta": float(second["artifact_fraction"] - first["artifact_fraction"]),
                "base_residual": float(first["residual_mean"]),
                "other_residual": float(second["residual_mean"]),
                "residual_delta": float(second["residual_mean"] - first["residual_mean"]),
            }
        )

    def rows_for(items: list[dict[str, Any]]) -> list[tuple[str, list[np.ndarray]]]:
        return [
            (
                f"{item['image_name']} | artifact {item['base_artifact']:.3f}->{item['other_artifact']:.3f} "
                f"({item['artifact_delta']:+.3f}) | residual {item['base_residual']:.3f}->{item['other_residual']:.3f}",
                compare_tiles(base, other, item["image_name"]),
            )
            for item in items
        ]

    improvements = sorted(records, key=lambda item: item["artifact_delta"])[:top_k]
    regressions = sorted(records, key=lambda item: item["artifact_delta"], reverse=True)[:top_k]
    deltas = np.asarray([item["artifact_delta"] for item in records])
    residual_deltas = np.asarray([item["residual_delta"] for item in records])
    return {
        "common_view_count": len(common),
        "artifact_delta": quantiles(deltas),
        "residual_delta": quantiles(residual_deltas),
        "artifact_better_count": int(np.sum(deltas < -1e-9)),
        "artifact_worse_count": int(np.sum(deltas > 1e-9)),
        "residual_better_count": int(np.sum(residual_deltas < -1e-9)),
        "residual_worse_count": int(np.sum(residual_deltas > 1e-9)),
        "largest_improvements": improvements,
        "largest_regressions": regressions,
        "sheets": {
            "improvements": save_rows(rows_for(improvements), output, "train_ab_improvements"),
            "regressions": save_rows(rows_for(regressions), output, "train_ab_regressions"),
        },
    }


def plot_sequence_summary(summary: dict[str, Any], output: Path) -> str:
    sequences = list(summary["by_sequence"])
    medians = [summary["by_sequence"][name]["artifact_fraction"]["p50"] for name in sequences]
    p90 = [summary["by_sequence"][name]["artifact_fraction"]["p90"] for name in sequences]
    x = np.arange(len(sequences))
    figure, axis = plt.subplots(figsize=(max(10, len(sequences) * 0.75), 4.8))
    axis.bar(x - 0.2, medians, 0.4, label="median")
    axis.bar(x + 0.2, p90, 0.4, label="p90")
    axis.set_xticks(x, sequences, rotation=45, ha="right")
    axis.set_ylabel("detected artifact fraction")
    axis.set_ylim(0.0, max(p90 + [0.1]) * 1.15)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = output / "train_sequence_artifact_summary.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return str(path)


def plot_camera_trajectory(dataset: DiagnosticSet, output: Path) -> str:
    centers = []
    scores = []
    names = []
    for record in dataset.records:
        w2c = np.asarray(record["camera"]["w2c"], dtype=np.float64)
        centers.append(np.linalg.inv(w2c)[:3, 3])
        scores.append(record["summary"]["artifact_fraction"])
        names.append(record["image_name"])
    centers = np.asarray(centers)
    scores = np.asarray(scores)
    figure, axis = plt.subplots(figsize=(8.5, 7.0))
    points = axis.scatter(centers[:, 0], centers[:, 2], c=scores, cmap="turbo", s=18, vmin=0.0, vmax=max(0.25, float(np.quantile(scores, 0.98))))
    for index in np.argsort(scores)[-12:]:
        axis.annotate(names[index], centers[index, [0, 2]], fontsize=6)
    axis.set_xlabel("camera center X")
    axis.set_ylabel("camera center Z")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.2)
    figure.colorbar(points, ax=axis, label="artifact fraction")
    figure.tight_layout()
    path = output / "train_trajectory_artifact_map.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return str(path)


def compare_image_dirs(first: Path | None, second: Path | None) -> dict[str, Any] | None:
    if first is None or second is None:
        return None
    first_files = {path.name: path for path in first.glob("*.png")}
    second_files = {path.name: path for path in second.glob("*.png")}
    names = sorted(set(first_files) & set(second_files))
    exact = 0
    maes = []
    max_differences = []
    for name in names:
        first_image = read_bgr(first_files[name])
        second_image = read_bgr(second_files[name])
        if first_image.shape != second_image.shape:
            second_image = resize(second_image, first_image.shape[1::-1])
        difference = np.abs(first_image.astype(np.int16) - second_image.astype(np.int16))
        exact += int(not np.any(difference))
        maes.append(float(difference.mean() / 255.0))
        max_differences.append(int(difference.max()))
    return {
        "common_image_count": len(names),
        "exact_image_count": exact,
        "mean_pixel_mae": float(np.mean(maes)) if maes else None,
        "max_pixel_difference": int(max(max_differences, default=0)),
    }


def heldout_sheets(
    base_root: Path | None,
    comparison_root: Path | None,
    refined_root: Path | None,
    output: Path,
    top_k: int,
) -> dict[str, Any] | None:
    if base_root is None:
        return None
    metrics = read_json(base_root / "rgb_metrics.json")
    if metrics is None:
        return None
    other_metrics = read_json(comparison_root / "rgb_metrics.json") if comparison_root else None
    refined_metrics = read_json(refined_root / "rgb_metrics.json") if refined_root else None
    ranked = sorted(metrics["per_view"].items(), key=lambda item: item[1]["psnr"])

    def image(root: Path | None, folder: str, name: str, fallback: np.ndarray) -> np.ndarray:
        path = root / folder / name if root else None
        return read_bgr(path) if path is not None and path.is_file() else fallback.copy()

    rows = []
    for name, entry in ranked[:top_k]:
        gt = read_bgr(base_root / "gt" / name)
        base_render = read_bgr(base_root / "renders" / name)
        other_render = image(comparison_root, "renders", name, base_render)
        refined_render = image(refined_root, "renders", name, base_render)
        error = np.mean(np.abs(base_render.astype(np.float32) - gt.astype(np.float32)), axis=2) / 255.0
        other_psnr = other_metrics["per_view"][name]["psnr"] if other_metrics else float("nan")
        refined_psnr = refined_metrics["per_view"][name]["psnr"] if refined_metrics else float("nan")
        rows.append(
            (
                f"{name} {entry.get('source_image', '')} | PSNR base={entry['psnr']:.2f} "
                f"comparison={other_psnr:.2f} refined={refined_psnr:.2f}",
                [
                    make_tile(gt, "heldout GT"),
                    make_tile(base_render, "retained_v2"),
                    make_tile(other_render, "comparison"),
                    make_tile(refined_render, "repair 3k"),
                    make_tile(heatmap(error, scale=0.35), "retained error"),
                ],
            )
        )
    return {
        "base_mean": metrics["mean"],
        "comparison_mean": other_metrics["mean"] if other_metrics else None,
        "refined_mean": refined_metrics["mean"] if refined_metrics else None,
        "sheets": save_rows(rows, output, "heldout_base_worst"),
    }


def audit_support_logic(
    support_report: dict[str, Any] | None,
    stage_manifest: dict[str, Any] | None,
    projective_report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if support_report is None:
        return None
    stage_by_target = {
        view["target_image_name"]: view for view in (stage_manifest or {}).get("pseudo_views", [])
    }
    projective_by_index = {
        int(view["index"]): view for view in (projective_report or {}).get("views", [])
    }
    findings = []
    components = []
    for report in support_report.get("reports", []):
        component = report["component"]
        area = int(component["area"])
        triangulated = int(report.get("triangulated_consensus_count", 0))
        depth_only = [
            support["image_name"]
            for support in report.get("selected_support", [])
            if bool(support.get("model_depth_ok", support.get("depth_ok", False)))
            and not bool(support.get("feature_ok"))
        ]
        stage_view = stage_by_target.get(report["target_image_name"])
        repair_fraction = float(stage_view["repair_mask_fraction"]) if stage_view else None
        component_fraction = float(component["area_fraction"])
        repair_to_component = (
            repair_fraction / component_fraction
            if repair_fraction is not None and component_fraction > 0
            else None
        )
        record = {
            "target_image_name": report["target_image_name"],
            "classification": report["classification"],
            "component_area_pixels": area,
            "component_area_fraction": component_fraction,
            "triangulated_consensus_count": triangulated,
            "triangulated_points_per_component_pixel": triangulated / max(area, 1),
            "dense_consensus_fraction": float(report.get("dense_consensus_fraction", 0.0)),
            "selected_depth_only_support_views": depth_only,
            "repair_mask_fraction": repair_fraction,
            "repair_to_component_area_ratio": repair_to_component,
        }
        components.append(record)
        if depth_only:
            findings.append(
                {
                    "severity": "error",
                    "code": "circular_current_model_depth_support",
                    "target": report["target_image_name"],
                    "detail": "Current-model depth agreement was accepted without independent feature geometry: "
                    + ", ".join(depth_only),
                }
            )
        if report["classification"] in {"wrong_geometry", "supported_static_patch"} and triangulated / max(area, 1) < 0.001:
            findings.append(
                {
                    "severity": "error",
                    "code": "component_evidence_scope_mismatch",
                    "target": report["target_image_name"],
                    "detail": f"{triangulated} triangulated points cannot classify a {area}-pixel mixed component.",
                }
            )
        if repair_to_component is not None and repair_to_component < 0.25:
            findings.append(
                {
                    "severity": "error",
                    "code": "repair_scope_metric_scope_mismatch",
                    "target": report["target_image_name"],
                    "detail": f"Repair covers {repair_to_component:.1%} of the component, so whole-component metrics do not test the repair hypothesis.",
                }
            )

    for index, view in projective_by_index.items():
        for support in view.get("support_views", []):
            if support.get("depth_consistent_fraction", 0.0) <= 0.05:
                findings.append(
                    {
                        "severity": "error",
                        "code": "selected_support_fails_clean_plane_validation",
                        "target": f"pseudo_{index:06d}",
                        "detail": f"{support['image_name']} was selected as clean support but has "
                        f"{support.get('depth_consistent_fraction', 0.0):.1%} clean-plane depth consistency and "
                        f"median relative error {support.get('median_relative_depth_error_on_semantic_clean')}.",
                    }
                )
    return {"components": components, "findings": findings}


def target_first_principles_sheet(
    diagnostics: DiagnosticSet,
    stage_manifest: dict[str, Any] | None,
    stage_root: Path | None,
    evaluation_root: Path | None,
    output: Path,
) -> str | None:
    if stage_manifest is None or not stage_manifest.get("pseudo_views"):
        return None
    view = stage_manifest["pseudo_views"][0]
    target_name = view["target_image_name"]
    record = diagnostics.by_name[target_name]
    data = diagnostics.arrays(record)
    component_id = int(view["component"]["component_id"])
    component = data["labels"] == component_id
    tiles = [
        make_tile(data["gt"], "target GT"),
        make_tile(data["render"], "target retained_v2"),
        make_tile(overlay_mask(data["render"], component, (32, 32, 245)), f"component {component.mean():.1%}"),
        make_tile(colorize_depth(data["depth"]), "target rendered depth"),
    ]
    if evaluation_root is not None:
        component_root = evaluation_root / "component_000"
        for suffix, title in (
            ("retained", "target eval retained"),
            ("edit_k3", "target eval opacity edit"),
            ("projective_k3_3k", "target eval reseed 3k"),
        ):
            path = component_root / f"{target_name}.{suffix}.png"
            if path.is_file():
                tiles.append(make_tile(read_bgr(path), title))
    if stage_root is not None:
        paths = [
            (stage_root / "select-gs/ori_warp_frame000000.png", "pseudo retained render"),
            (stage_root / "select-gs/mask_frame000000.png", "pseudo repair mask"),
            (stage_root / "select-gs/clean_support_depth_frame000000.tiff", "clean plane depth"),
            (stage_root / "select-gs-inpainted/predict_warp_frame000000.png", "See3D output"),
            (stage_root / "select-gs-projective-merged/predict_warp_frame000000.png", "projective clean RGB"),
        ]
        for path, title in paths:
            if not path.is_file():
                continue
            if path.suffix.lower() in {".tiff", ".tif"}:
                depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                image = colorize_depth(depth.astype(np.float32))
            elif "mask" in path.name:
                mask = read_gray(path)
                image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            else:
                image = read_bgr(path)
            tiles.append(make_tile(image, title))
    rows = [(f"{target_name} | component={view['component']['area_fraction']:.1%} repair={view.get('repair_mask_fraction', 0.0):.1%}", tiles)]
    paths = save_rows(rows, output, "target_first_principles", rows_per_sheet=1)
    return paths[0] if paths else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--comparison-diagnostics", type=Path)
    parser.add_argument("--support-report", type=Path)
    parser.add_argument("--stage-manifest", type=Path)
    parser.add_argument("--projective-report", type=Path)
    parser.add_argument("--stage-root", type=Path)
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--base-heldout", type=Path)
    parser.add_argument("--comparison-heldout", type=Path)
    parser.add_argument("--refined-heldout", type=Path)
    parser.add_argument("--g4-parity-heldout-renders", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=24)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = DiagnosticSet(args.diagnostics.expanduser().resolve())
    summary = summarize_diagnostics(diagnostics)
    report: dict[str, Any] = {
        "version": 1,
        "mode": "artifact_pipeline_first_principles_audit",
        "diagnostics": str(diagnostics.root),
        "dataset_invariants": audit_dataset_invariants(diagnostics),
        "diagnostic_summary": summary,
        "visualizations": make_train_sheets(diagnostics, output, int(args.top_k)),
    }
    report["visualizations"]["sequence_summary"] = plot_sequence_summary(summary, output)
    report["visualizations"]["trajectory"] = plot_camera_trajectory(diagnostics, output)

    comparison = DiagnosticSet(args.comparison_diagnostics.expanduser().resolve()) if args.comparison_diagnostics else None
    if comparison is not None:
        report["comparison_dataset_invariants"] = audit_dataset_invariants(comparison)
        report["comparison_summary"] = summarize_diagnostics(comparison)
        report["model_comparison"] = compare_diagnostics(diagnostics, comparison, output, int(args.top_k))

    support_report = read_json(args.support_report)
    stage_manifest = read_json(args.stage_manifest)
    projective_report = read_json(args.projective_report)
    report["support_logic_audit"] = audit_support_logic(support_report, stage_manifest, projective_report)
    report["visualizations"]["target_first_principles"] = target_first_principles_sheet(
        diagnostics,
        stage_manifest,
        args.stage_root,
        args.evaluation_root,
        output,
    )
    report["heldout"] = heldout_sheets(
        args.base_heldout,
        args.comparison_heldout,
        args.refined_heldout,
        output,
        int(args.top_k),
    )
    report["renderer_parity"] = compare_image_dirs(
        args.base_heldout / "renders" if args.base_heldout else None,
        args.g4_parity_heldout_renders,
    )
    write_json(output / "first_principles_audit.json", report)
    print(json.dumps({
        "output": str(output),
        "invariants_passed": report["dataset_invariants"]["passed"],
        "train_views": len(diagnostics.records),
        "comparison_views": len(comparison.records) if comparison else 0,
        "support_findings": len((report.get("support_logic_audit") or {}).get("findings", [])),
    }, indent=2))


if __name__ == "__main__":
    main()
