#!/usr/bin/env python3
"""Audit whether chart masks and geometry survive into Gaussian initialization.

The script is read-only. Each ``--run-spec`` is a JSON object describing one
MAtCha run, which makes it possible to compare historical runs whose alignment
and initialization mask policies differed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if mask.shape == shape:
        return mask.astype(bool)
    return cv2.resize(
        mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)


def finite_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def masked_quantile(values: np.ndarray, mask: np.ndarray, quantile: float) -> float | None:
    selected = values[mask & np.isfinite(values)]
    return finite_number(np.quantile(selected, quantile)) if selected.size else None


def depth_color(depth: np.ndarray, low: float, high: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= high * 4.0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        log_low = math.log(max(low, 1e-6))
        log_high = math.log(max(high, low + 1e-6))
        values = (np.log(np.maximum(depth, 1e-6)) - log_low) / (log_high - log_low)
        normalized[valid] = np.round(np.clip(values[valid], 0.0, 1.0) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def relative_delta_color(prior: np.ndarray, aligned: np.ndarray) -> np.ndarray:
    valid = np.isfinite(prior) & np.isfinite(aligned) & (prior > 0.0) & (aligned > 0.0)
    delta = np.zeros(prior.shape, dtype=np.float32)
    delta[valid] = np.abs(aligned[valid] - prior[valid]) / np.maximum(prior[valid], 1e-6)
    normalized = np.round(np.clip(delta / 0.75, 0.0, 1.0) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def mask_overlay(image: np.ndarray, keep: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    keep = resize_mask(keep, image.shape[:2])
    overlay = image.copy()
    solid = np.empty_like(image)
    solid[:] = color
    overlay[~keep] = cv2.addWeighted(image, 0.25, solid, 0.75, 0)[~keep]
    return overlay


def heatmap(values: np.ndarray, maximum: float) -> np.ndarray:
    normalized = np.round(np.clip(values / maximum, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def tile(image: np.ndarray, title: str, width: int = 240, height: int = 135) -> np.ndarray:
    interpolation = cv2.INTER_AREA if image.shape[1] > width else cv2.INTER_LINEAR
    image = cv2.resize(image, (width, height), interpolation=interpolation)
    canvas = np.full((height + 42, width, 3), 245, dtype=np.uint8)
    canvas[42:] = image
    lines = [title[index : index + 38] for index in range(0, len(title), 38)][:2]
    for line_index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (5, 16 + 17 * line_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (15, 15, 15),
            1,
            cv2.LINE_AA,
        )
    return canvas


def save_pages(rows: list[np.ndarray], output_root: Path, prefix: str, rows_per_page: int = 6) -> list[str]:
    paths = []
    for start in range(0, len(rows), rows_per_page):
        page = np.vstack(rows[start : start + rows_per_page])
        path = output_root / f"{prefix}_{start // rows_per_page + 1:02d}.jpg"
        cv2.imwrite(str(path), page, [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths.append(str(path))
    return paths


def load_diagnostics(root: Path | None) -> tuple[dict[str, dict[str, Any]], Path | None]:
    if root is None:
        return {}, None
    manifest_path = root / "diagnostic_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return {record["image_name"]: record for record in manifest["views"]}, root


def pointmap_confidence(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    return np.asarray(payload["confs"], dtype=np.float32)


def load_run(spec: dict[str, Any]) -> dict[str, Any]:
    mast3r_root = Path(spec["mast3r_root"])
    charts = np.load(mast3r_root / "charts_data.npz")
    cameras = json.loads((mast3r_root / "cameras.json").read_text())
    names = [Path(path).stem for path in cameras["filepaths"]]
    if len(names) != charts["depths"].shape[0]:
        raise RuntimeError(f"{spec['label']}: camera/chart count mismatch")
    mask_lookup = CambridgeMaskLookup(
        Path(spec["dataset_path"]),
        Path(spec["mask_pickle"]),
        mask_indices=[int(index) for index in spec["mask_indices"]],
    )
    diagnostics, diagnostics_root = load_diagnostics(
        Path(spec["diagnostics_root"]) if spec.get("diagnostics_root") else None
    )
    return {
        "spec": spec,
        "root": mast3r_root,
        "charts": charts,
        "cameras": cameras,
        "names": names,
        "mask_lookup": mask_lookup,
        "diagnostics": diagnostics,
        "diagnostics_root": diagnostics_root,
    }


def chart_depth_range(run: dict[str, Any]) -> tuple[float, float]:
    charts = run["charts"]
    scale = float(charts["scale_factor"])
    values = np.concatenate(
        [charts["prior_depths"].reshape(-1), charts["depths"].reshape(-1)]
    ) / scale
    valid = np.isfinite(values) & (values > 0.0) & (values < 100.0)
    if not np.any(valid):
        return 0.1, 10.0
    return tuple(float(value) for value in np.quantile(values[valid], [0.01, 0.99]))


def effective_masks(
    run: dict[str, Any], index: int, sfm_conf: np.ndarray, semantic: np.ndarray
) -> dict[str, np.ndarray]:
    spec = run["spec"]
    charts = run["charts"]
    aligned = charts["depths"][index]
    points = charts["pts"][index]
    confidence = charts["confs"][index]

    sfm_keep = sfm_conf > float(spec.get("sfm_mask_threshold", 0.25))
    alignment_keep = np.ones_like(sfm_keep)
    if spec.get("use_sfm_alignment_mask", False):
        alignment_keep &= sfm_keep
    if spec.get("use_semantic_alignment_mask", False):
        alignment_keep &= semantic

    outlier_keep = np.isfinite(aligned) & np.isfinite(points).all(axis=-1)
    max_depth = spec.get("max_abs_depth")
    max_point = spec.get("max_abs_point")
    if max_depth is not None:
        outlier_keep &= np.abs(aligned) <= float(max_depth)
    if max_point is not None:
        outlier_keep &= np.max(np.abs(points), axis=-1) <= float(max_point)

    # A zero confidence is the persisted alignment/semantic rejection marker.
    # Keep this strict even when the requested quantile itself evaluates to 0.
    confidence_keep = np.isfinite(confidence) & (confidence > 0.0)
    quantile = spec.get("min_conf_quantile")
    if quantile is not None:
        positive_confidence = confidence[confidence_keep]
        threshold = (
            np.quantile(positive_confidence, float(quantile))
            if positive_confidence.size
            else float("inf")
        )
        confidence_keep &= confidence >= threshold

    init_keep = outlier_keep & confidence_keep
    if spec.get("use_semantic_init_mask", False):
        init_keep &= semantic
    leaked = init_keep & ~alignment_keep
    return {
        "sfm": sfm_keep,
        "semantic": semantic,
        "alignment": alignment_keep,
        "outlier": outlier_keep,
        "confidence": confidence_keep,
        "init": init_keep,
        "leaked": leaked,
    }


def diagnostic_images(
    run: dict[str, Any], name: str, fallback_rgb: np.ndarray, low: float, high: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None, float | None]:
    record = run["diagnostics"].get(name)
    root = run["diagnostics_root"]
    if record is None or root is None:
        blank = np.zeros_like(fallback_rgb)
        return blank, blank, blank, None, None
    rendered = read_image(root / record["files"]["render"])
    rendered = cv2.resize(rendered, (fallback_rgb.shape[1], fallback_rgb.shape[0]))
    depth = np.load(root / record["files"]["depth"]).astype(np.float32)
    depth_vis = depth_color(depth, low, high)
    depth_vis = cv2.resize(depth_vis, (fallback_rgb.shape[1], fallback_rgb.shape[0]))
    difference = np.abs(rendered.astype(np.float32) - fallback_rgb.astype(np.float32)).mean(axis=2) / 255.0
    summary = record["summary"]
    return (
        rendered,
        heatmap(difference, 0.35),
        depth_vis,
        float(summary["artifact_fraction"]),
        float(summary["residual_mean"]),
    )


def analyze_run(run: dict[str, Any], output_root: Path) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    spec = run["spec"]
    charts = run["charts"]
    scale = float(charts["scale_factor"])
    low, high = chart_depth_range(run)
    records: list[dict[str, Any]] = []
    rows: list[np.ndarray] = []

    for index, name in enumerate(run["names"]):
        image_path = run["root"] / "images" / f"{name}.png"
        rgb = read_image(image_path)
        height, width = charts["depths"][index].shape
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        sfm_conf = pointmap_confidence(run["root"] / "pointmaps" / f"{name}.json")
        semantic = run["mask_lookup"].get_mask(
            name, (height, width), torch.device("cpu")
        ).numpy()
        masks = effective_masks(run, index, sfm_conf, semantic)

        prior = charts["prior_depths"][index]
        aligned = charts["depths"][index]
        valid = (
            masks["alignment"]
            & np.isfinite(prior)
            & np.isfinite(aligned)
            & (prior > 0.0)
            & (aligned > 0.0)
        )
        relative = np.zeros_like(prior, dtype=np.float32)
        relative[valid] = np.abs(aligned[valid] - prior[valid]) / np.maximum(prior[valid], 1e-6)
        rendered, residual_vis, rendered_depth, artifact_fraction, residual_mean = diagnostic_images(
            run, name, rgb, low, high
        )

        record = {
            "label": spec["label"],
            "index": index,
            "image_name": name,
            "sequence": name.split("__", 1)[0],
            "sfm_keep_fraction": float(masks["sfm"].mean()),
            "semantic_keep_fraction": float(masks["semantic"].mean()),
            "alignment_keep_fraction": float(masks["alignment"].mean()),
            "init_keep_fraction": float(masks["init"].mean()),
            "leaked_fraction": float(masks["leaked"].mean()),
            "leaked_of_init_fraction": float(
                masks["leaked"].sum() / max(int(masks["init"].sum()), 1)
            ),
            "saved_conf_zero_fraction": float(np.mean(charts["confs"][index] <= 0.0)),
            "prior_aligned_relative_median": masked_quantile(relative, valid, 0.5),
            "prior_aligned_relative_p90": masked_quantile(relative, valid, 0.9),
            "prior_aligned_gt25_fraction": float(np.mean(relative[valid] > 0.25)) if np.any(valid) else None,
            "depth_outlier_fraction": float(np.mean(np.abs(aligned) > 50.0)),
            "artifact_fraction": artifact_fraction,
            "residual_mean": residual_mean,
        }
        records.append(record)

        leak_overlay = mask_overlay(rgb, ~masks["leaked"], (0, 255, 255))
        row_tiles = [
            tile(rgb, f"{name} GT"),
            tile(rendered, f"final render art={artifact_fraction if artifact_fraction is not None else -1:.3f}"),
            tile(residual_vis, f"RGB residual mean={residual_mean if residual_mean is not None else -1:.3f}"),
            tile(rendered_depth, "final rendered depth"),
            tile(depth_color(prior / scale, low, high), "chart prior depth"),
            tile(depth_color(aligned / scale, low, high), "aligned chart depth"),
            tile(relative_delta_color(prior, aligned), f"aligned/prior p90={record['prior_aligned_relative_p90'] or 0:.3f}"),
            tile(leak_overlay, f"init<-masked leak={record['leaked_fraction']:.3f}"),
        ]
        rows.append(np.hstack(row_tiles))

    return records, rows


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "sfm_keep_fraction",
        "semantic_keep_fraction",
        "alignment_keep_fraction",
        "init_keep_fraction",
        "leaked_fraction",
        "leaked_of_init_fraction",
        "saved_conf_zero_fraction",
        "prior_aligned_relative_median",
        "prior_aligned_relative_p90",
        "prior_aligned_gt25_fraction",
        "depth_outlier_fraction",
        "artifact_fraction",
        "residual_mean",
    ]
    output: dict[str, Any] = {"chart_count": len(records)}
    for key in numeric_keys:
        values = [record[key] for record in records if record.get(key) is not None]
        output[f"mean_{key}"] = float(np.mean(values)) if values else None
        output[f"max_{key}"] = float(np.max(values)) if values else None
    paired = [
        (record["leaked_fraction"], record["artifact_fraction"])
        for record in records
        if record.get("artifact_fraction") is not None
    ]
    if len(paired) > 2 and np.std([item[0] for item in paired]) > 1e-9:
        output["leak_artifact_correlation"] = float(
            np.corrcoef(np.asarray(paired).T)[0, 1]
        )
    else:
        output["leak_artifact_correlation"] = None
    output["worst_leak_charts"] = sorted(
        records, key=lambda record: record["leaked_fraction"], reverse=True
    )[:8]
    output["worst_artifact_charts"] = sorted(
        [record for record in records if record["artifact_fraction"] is not None],
        key=lambda record: record["artifact_fraction"],
        reverse=True,
    )[:8]
    return output


def target_neighbors(run: dict[str, Any], target_name: str) -> list[dict[str, Any]]:
    target_record = run["diagnostics"].get(target_name)
    if target_record is None:
        return []
    target_c2w = np.linalg.inv(np.asarray(target_record["camera"]["w2c"], dtype=np.float64))
    target_center = target_c2w[:3, 3]
    target_direction = target_c2w[:3, 2]
    neighbors = []
    for name, c2w_list in zip(run["names"], run["cameras"]["cams2world"]):
        c2w = np.asarray(c2w_list, dtype=np.float64)
        center_distance = float(np.linalg.norm(c2w[:3, 3] - target_center))
        cosine = float(np.clip(np.dot(c2w[:3, 2], target_direction), -1.0, 1.0))
        angle = float(math.degrees(math.acos(cosine)))
        neighbors.append(
            {"image_name": name, "center_distance": center_distance, "view_angle_degrees": angle}
        )
    return sorted(neighbors, key=lambda item: (item["center_distance"], item["view_angle_degrees"]))[:8]


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    fields = list(records[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-spec", action="append", required=True, help="JSON run specification")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-view", default="seq2__frame00109")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run_specs = [json.loads(value) for value in args.run_spec]
    all_records: list[dict[str, Any]] = []
    report: dict[str, Any] = {"runs": {}, "target_view": args.target_view}

    for spec in run_specs:
        run = load_run(spec)
        records, rows = analyze_run(run, args.output)
        all_records.extend(records)
        label = spec["label"]
        ordered_by_leak = sorted(
            zip(records, rows), key=lambda pair: pair[0]["leaked_fraction"], reverse=True
        )
        ordered_by_artifact = sorted(
            zip(records, rows),
            key=lambda pair: pair[0]["artifact_fraction"] or 0.0,
            reverse=True,
        )
        pages = save_pages(rows, args.output, f"{label}_all_charts")
        leak_pages = save_pages([row for _, row in ordered_by_leak], args.output, f"{label}_worst_leak")
        artifact_pages = save_pages(
            [row for _, row in ordered_by_artifact], args.output, f"{label}_worst_final_artifact"
        )
        report["runs"][label] = {
            "spec": spec,
            "summary": summarize(records),
            "target_nearest_charts": target_neighbors(run, args.target_view),
            "pages": pages,
            "leak_pages": leak_pages,
            "artifact_pages": artifact_pages,
        }

    write_csv(all_records, args.output / "chart_geometry_audit.csv")
    (args.output / "chart_geometry_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "output": str(args.output),
        "chart_count": len(all_records),
        "report": str(args.output / "chart_geometry_audit.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
