#!/usr/bin/env python3
"""Create aligned GT/G4Splat/ULFLoc/STDLoc hard-case diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


METHOD_ORDER = ("g4", "ulfloc", "stdloc")
METHOD_LABELS = {"g4": "G4Splat / 2DGS-40k", "ulfloc": "ULFLoc / 3DGS-30k", "stdloc": "STDLoc / 3DGS-7k"}


def _canonical_name(value: str) -> str:
    text = str(value).strip().replace("\\", "/")
    suffix = Path(text).suffix
    if suffix:
        text = text[: -len(suffix)]
    return text.replace("/", "__")


def _safe_name(value: str) -> str:
    return _canonical_name(value).replace("__", "_")


def _read_rgb(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if shape is not None and value.shape[:2] != shape:
        value = cv2.resize(value, (shape[1], shape[0]), interpolation=cv2.INTER_LANCZOS4)
    return value


def _metric_index(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for render_name, row in payload.get("per_view", {}).items():
        if not isinstance(row, dict):
            continue
        source = str(row.get("source_image", ""))
        if source:
            result[_canonical_name(source)] = {"render_name": str(render_name), "row": row}
    return result


def _metric_summary(row: dict[str, Any]) -> dict[str, Any]:
    static = row.get("masked", {}).get("static_valid", {})
    return {
        "raw_psnr": row.get("psnr"),
        "raw_ssim": row.get("ssim"),
        "raw_mae": row.get("mae"),
        "static_psnr": static.get("psnr"),
        "static_ssim": static.get("ssim"),
        "static_mae": static.get("mae"),
        "static_valid_pixel_ratio": static.get("valid_pixel_ratio"),
    }


def _error_stats(render: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    difference = render - gt
    mae = float(np.mean(np.abs(difference)))
    mse = float(np.mean(np.square(difference)))
    return {
        "mae": mae,
        "rmse": float(math.sqrt(mse)),
        "psnr": float(-10.0 * math.log10(max(mse, 1e-12))),
    }


def _fit_affine(render: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = render.shape[:2]
    valid = np.ones((height, width), dtype=bool)
    valid &= np.max(gt, axis=2) < 0.995
    corrected = render.copy()
    affine = np.zeros((3, 2), dtype=np.float32)
    for _ in range(3):
        for channel in range(3):
            source = render[..., channel][valid]
            target = gt[..., channel][valid]
            if source.size < 32 or float(np.std(source)) < 1e-6:
                gain, bias = 1.0, 0.0
            else:
                design = np.stack([source, np.ones_like(source)], axis=1)
                gain, bias = np.linalg.lstsq(design, target, rcond=None)[0]
                gain = float(np.clip(gain, 0.25, 4.0))
                bias = float(np.clip(bias, -0.50, 0.50))
            affine[channel] = (gain, bias)
            corrected[..., channel] = np.clip(render[..., channel] * gain + bias, 0.0, 1.0)
        residual = np.mean(np.abs(corrected - gt), axis=2)
        cutoff = float(np.quantile(residual[valid], 0.90)) if np.any(valid) else float("inf")
        valid &= residual <= cutoff
    return corrected, affine


def _phase_alignment(render: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    render_gray = cv2.cvtColor(np.uint8(np.clip(render, 0.0, 1.0) * 255.0), cv2.COLOR_RGB2GRAY)
    gt_gray = cv2.cvtColor(np.uint8(np.clip(gt, 0.0, 1.0) * 255.0), cv2.COLOR_RGB2GRAY)
    render_edge = np.hypot(
        cv2.Sobel(render_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(render_gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    gt_edge = np.hypot(
        cv2.Sobel(gt_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gt_gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    window = cv2.createHanningWindow((render.shape[1], render.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(render_edge * window, gt_edge * window)
    translations = ((float(shift[0]), float(shift[1])), (-float(shift[0]), -float(shift[1])))
    baseline_mae = _error_stats(render, gt)["mae"]
    best: tuple[float, tuple[float, float], np.ndarray] | None = None
    for dx, dy in translations:
        matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        warped = cv2.warpAffine(
            render,
            matrix,
            (render.shape[1], render.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        candidate_mae = _error_stats(warped, gt)["mae"]
        if best is None or candidate_mae < best[0]:
            best = (candidate_mae, (dx, dy), warped)
    assert best is not None
    return {
        "shift_xy": [float(best[1][0]), float(best[1][1])],
        "shift_norm_px": float(math.hypot(*best[1])),
        "phase_response": float(response),
        "raw_mae": float(baseline_mae),
        "translated_mae": float(best[0]),
        "relative_mae_reduction": float((baseline_mae - best[0]) / max(baseline_mae, 1e-9)),
    }


def _residual_components(error: np.ndarray) -> tuple[list[dict[str, Any]], tuple[int, int, int, int]]:
    height, width = error.shape
    threshold = max(0.12, float(np.quantile(error, 0.92)))
    binary = np.uint8(error >= threshold)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[dict[str, Any]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 24:
            continue
        mask = labels == index
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        component_width = int(stats[index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        mean_error = float(error[mask].mean())
        components.append(
            {
                "bbox_xyxy": [x, y, x + component_width, y + component_height],
                "pixel_count": area,
                "mean_error": mean_error,
                "score": float(mean_error * math.sqrt(area)),
            }
        )
    components.sort(key=lambda item: (-float(item["score"]), -int(item["pixel_count"])))
    selected = components[:3]
    if not selected:
        return components, (0, 0, width, height)
    x0 = min(item["bbox_xyxy"][0] for item in selected)
    y0 = min(item["bbox_xyxy"][1] for item in selected)
    x1 = max(item["bbox_xyxy"][2] for item in selected)
    y1 = max(item["bbox_xyxy"][3] for item in selected)
    pad = 20
    return components, (max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad))


def _error_heatmap(error: np.ndarray) -> np.ndarray:
    normalized = np.uint8(np.clip(error / 0.35, 0.0, 1.0) * 255.0)
    return cv2.cvtColor(cv2.applyColorMap(normalized, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)


def _with_boxes(image: np.ndarray, components: list[dict[str, Any]]) -> np.ndarray:
    output = np.uint8(np.clip(image, 0.0, 1.0) * 255.0).copy()
    for ordinal, component in enumerate(components[:3], start=1):
        x0, y0, x1, y1 = component["bbox_xyxy"]
        cv2.rectangle(output, (x0, y0), (x1 - 1, y1 - 1), (255, 70, 40), 2, lineType=cv2.LINE_AA)
        cv2.putText(output, str(ordinal), (x0 + 3, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 70, 40), 1, cv2.LINE_AA)
    return output


def _static_title(label: str, metric: dict[str, Any]) -> str:
    psnr = metric.get("static_psnr")
    ssim = metric.get("static_ssim")
    mae = metric.get("static_mae")
    if all(value is not None for value in (psnr, ssim, mae)):
        return f"{label}\nstatic PSNR {float(psnr):.2f}, SSIM {float(ssim):.3f}, MAE {float(mae):.3f}"
    return label


def _save_case_panel(
    output: Path,
    name: str,
    gt: np.ndarray,
    renders: dict[str, np.ndarray],
    metrics: dict[str, dict[str, Any]],
    corrected_g4: np.ndarray,
    g4_error: np.ndarray,
    components: list[dict[str, Any]],
    crop: tuple[int, int, int, int],
    affine_stats: dict[str, float],
    phase: dict[str, Any],
) -> None:
    errors = {method: np.mean(np.abs(render - gt), axis=2) for method, render in renders.items()}
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    top = [
        (gt, "Ground truth"),
        (_with_boxes(renders["g4"], components), _static_title(METHOD_LABELS["g4"], metrics["g4"])),
        (renders["ulfloc"], _static_title(METHOD_LABELS["ulfloc"], metrics["ulfloc"])),
        (renders["stdloc"], _static_title(METHOD_LABELS["stdloc"], metrics["stdloc"])),
    ]
    bottom = [
        (_error_heatmap(g4_error), "|GT − G4| (raw)"),
        (_error_heatmap(errors["ulfloc"]), "|GT − ULFLoc|"),
        (_error_heatmap(errors["stdloc"]), "|GT − STDLoc|"),
        (_error_heatmap(np.mean(np.abs(corrected_g4 - gt), axis=2)), "|GT − G4| (affine-corrected)"),
    ]
    for axis, (image, title) in zip(axes[0], top):
        axis.imshow(image)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    for axis, (image, title) in zip(axes[1], bottom):
        axis.imshow(image)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.suptitle(
        f"{name}: G4 affine MAE {affine_stats['mae']:.3f}; phase shift "
        f"({phase['shift_xy'][0]:.2f}, {phase['shift_xy'][1]:.2f}) px; "
        f"translation MAE gain {phase['relative_mae_reduction']:.1%}",
        fontsize=13,
    )
    fig.savefig(output / f"{_safe_name(name)}_comparison.png", dpi=160)
    plt.close(fig)

    x0, y0, x1, y1 = crop
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    top_crops = [
        (gt[y0:y1, x0:x1], "Ground truth crop"),
        (renders["g4"][y0:y1, x0:x1], "G4 crop"),
        (renders["ulfloc"][y0:y1, x0:x1], "ULFLoc crop"),
        (renders["stdloc"][y0:y1, x0:x1], "STDLoc crop"),
    ]
    bottom_crops = [
        (_error_heatmap(g4_error[y0:y1, x0:x1]), "G4 residual"),
        (_error_heatmap(errors["ulfloc"][y0:y1, x0:x1]), "ULFLoc residual"),
        (_error_heatmap(errors["stdloc"][y0:y1, x0:x1]), "STDLoc residual"),
        (_error_heatmap(np.mean(np.abs(corrected_g4 - gt), axis=2)[y0:y1, x0:x1]), "G4 affine residual"),
    ]
    for axis, (image, title) in zip(axes[0], top_crops):
        axis.imshow(image)
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    for axis, (image, title) in zip(axes[1], bottom_crops):
        axis.imshow(image)
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    fig.suptitle(f"{name}: highest G4 residual components, crop={list(crop)}", fontsize=13)
    fig.savefig(output / f"{_safe_name(name)}_crop.png", dpi=180)
    plt.close(fig)


def _save_overview(output: Path, cases: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(len(cases), 4, figsize=(16, 4.2 * len(cases)), constrained_layout=True)
    if len(cases) == 1:
        axes = np.asarray([axes])
    for row, case in enumerate(cases):
        panels = [(case["gt"], "GT")] + [(case["renders"][method], METHOD_LABELS[method]) for method in METHOD_ORDER]
        for axis, (image, label) in zip(axes[row], panels):
            axis.imshow(image)
            axis.set_title(label, fontsize=10)
            axis.axis("off")
        axes[row, 0].set_ylabel(case["source_name"], fontsize=11)
    fig.savefig(output / "all_cases_overview.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-file", type=Path, required=True)
    parser.add_argument("--g4-render-dir", type=Path, required=True)
    parser.add_argument("--g4-gt-dir", type=Path, required=True)
    parser.add_argument("--g4-metrics-json", type=Path, required=True)
    parser.add_argument("--ulfloc-render-dir", type=Path, required=True)
    parser.add_argument("--ulfloc-metrics-json", type=Path, required=True)
    parser.add_argument("--stdloc-render-dir", type=Path, required=True)
    parser.add_argument("--stdloc-metrics-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = [line.strip() for line in args.target_file.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    metrics_index = {
        "g4": _metric_index(args.g4_metrics_json),
        "ulfloc": _metric_index(args.ulfloc_metrics_json),
        "stdloc": _metric_index(args.stdloc_metrics_json),
    }
    render_dirs = {"g4": args.g4_render_dir, "ulfloc": args.ulfloc_render_dir, "stdloc": args.stdloc_render_dir}
    cases: list[dict[str, Any]] = []
    report_cases: list[dict[str, Any]] = []
    for source_name in targets:
        name = _canonical_name(source_name)
        entries = {method: metrics_index[method].get(name) for method in METHOD_ORDER}
        missing = [method for method, entry in entries.items() if entry is None]
        if missing:
            raise FileNotFoundError(f"Missing metric identity for {source_name}: {', '.join(missing)}")
        assert all(entry is not None for entry in entries.values())
        g4_entry = entries["g4"]
        assert g4_entry is not None
        gt = _read_rgb(args.g4_gt_dir / g4_entry["render_name"])
        renders = {}
        cached_metrics = {}
        gt_differences = {}
        for method in METHOD_ORDER:
            entry = entries[method]
            assert entry is not None
            renders[method] = _read_rgb(render_dirs[method] / entry["render_name"], gt.shape[:2])
            cached_metrics[method] = _metric_summary(entry["row"])
            if method != "g4":
                candidate_gt_dir = render_dirs[method].parent / "gt"
                candidate_gt = candidate_gt_dir / entry["render_name"]
                if candidate_gt.is_file():
                    gt_differences[method] = _error_stats(_read_rgb(candidate_gt, gt.shape[:2]), gt)
        g4_error = np.mean(np.abs(renders["g4"] - gt), axis=2)
        corrected_g4, affine = _fit_affine(renders["g4"], gt)
        affine_stats = _error_stats(corrected_g4, gt)
        phase = _phase_alignment(renders["g4"], gt)
        components, crop = _residual_components(g4_error)
        _save_case_panel(output, source_name, gt, renders, cached_metrics, corrected_g4, g4_error, components, crop, affine_stats, phase)
        direct_metrics = {method: _error_stats(renders[method], gt) for method in METHOD_ORDER}
        case_report = {
            "source_name": source_name,
            "canonical_name": name,
            "cached_metrics": cached_metrics,
            "direct_rgb_metrics": direct_metrics,
            "g4_affine": affine.tolist(),
            "g4_affine_direct_metrics": affine_stats,
            "g4_phase_translation_probe": phase,
            "g4_residual_components": components[:12],
            "g4_selected_crop_xyxy": list(crop),
            "gt_differences_against_g4_cache": gt_differences,
        }
        report_cases.append(case_report)
        cases.append({"source_name": source_name, "gt": gt, "renders": renders})
    _save_overview(output, cases)
    report = {
        "protocol": "same_camera_shared640_gt_g4_ulfloc_stdloc_visual_comparison_v1",
        "methods": METHOD_LABELS,
        "case_count": len(report_cases),
        "cases": report_cases,
    }
    (output / "multimethod_visual_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "case_count": len(report_cases)}, indent=2))


if __name__ == "__main__":
    main()
