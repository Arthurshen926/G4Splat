#!/usr/bin/env python3
"""Audit raw MASt3R pointmaps against the cameras written beside them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402


def quantiles(values: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    selected = values[mask & np.isfinite(values)]
    if not selected.size:
        return {key: None for key in ("p01", "p50", "p90", "p99", "max")}
    result = np.quantile(selected, [0.01, 0.50, 0.90, 0.99])
    return {
        "p01": float(result[0]),
        "p50": float(result[1]),
        "p90": float(result[2]),
        "p99": float(result[3]),
        "max": float(selected.max()),
    }


def colorize(values: np.ndarray, mask: np.ndarray, high: float, log: bool = False) -> np.ndarray:
    normalized = np.zeros(values.shape, dtype=np.uint8)
    valid = mask & np.isfinite(values)
    if np.any(valid):
        display = np.log(np.maximum(values, 1e-6)) if log else values
        low = float(np.quantile(display[valid], 0.01))
        upper = float(np.quantile(display[valid], high)) if high <= 1.0 else float(high)
        denominator = max(upper - low, 1e-6)
        normalized[valid] = np.round(
            np.clip((display[valid] - low) / denominator, 0.0, 1.0) * 255.0
        ).astype(np.uint8)
    output = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    output[~valid] = 0
    return output


def tile(image: np.ndarray, title: str, width: int = 256, height: int = 144) -> np.ndarray:
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.full((height + 24, width, 3), 245, dtype=np.uint8)
    canvas[24:] = image
    cv2.putText(canvas, title[:40], (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mast3r-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-pickle", type=Path)
    parser.add_argument("--mask-dataset-path", type=Path)
    parser.add_argument("--mask-indices", type=int, nargs="*", default=[0, 1, 2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cameras = json.loads((args.mast3r_root / "cameras.json").read_text())
    mask_lookup = None
    if args.mask_pickle:
        mask_lookup = CambridgeMaskLookup(
            args.mask_dataset_path,
            args.mask_pickle,
            mask_indices=args.mask_indices,
        )

    records = []
    rows = []
    for index, filepath in enumerate(cameras["filepaths"]):
        name = Path(filepath).stem
        payload = json.loads((args.mast3r_root / "pointmaps" / f"{name}.json").read_text())
        confidence = np.asarray(payload["confs"], dtype=np.float32)
        height, width = confidence.shape
        points = np.asarray(payload["points"], dtype=np.float64).reshape(height, width, 3)
        rgb = np.asarray(payload["rgb"], dtype=np.float32)
        rgb_bgr = cv2.cvtColor(np.round(np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        semantic = np.ones((height, width), dtype=bool)
        if mask_lookup is not None:
            semantic = mask_lookup.get_mask(
                name, (height, width), torch.device("cpu")
            ).cpu().numpy().astype(bool)

        camera_to_world = np.asarray(cameras["cams2world"][index], dtype=np.float64)
        world_to_camera = np.linalg.inv(camera_to_world)
        camera_points = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        depth = camera_points[..., 2]
        finite = np.isfinite(points).all(axis=2) & np.isfinite(confidence)
        valid = finite & semantic & (confidence > 0.0) & (depth > 0.0)

        yy, xx = np.mgrid[:height, :width]
        focal = float(cameras["focals"][index])
        projected_x = focal * camera_points[..., 0] / np.maximum(depth, 1e-12) + width / 2.0
        projected_y = focal * camera_points[..., 1] / np.maximum(depth, 1e-12) + height / 2.0
        delta_x = projected_x - xx
        delta_y = projected_y - yy
        projection_error = np.hypot(delta_x, delta_y)

        record = {
            "index": index,
            "name": name,
            "shape": [height, width],
            "finite_fraction": float(finite.mean()),
            "semantic_keep_fraction": float(semantic.mean()),
            "positive_confidence_fraction": float((confidence > 0.0).mean()),
            "valid_projection_fraction": float(valid.mean()),
            "confidence": quantiles(confidence, valid),
            "camera_depth": quantiles(depth, valid),
            "projection_error_px": quantiles(projection_error, valid),
            "projection_delta_x_p50": float(np.median(delta_x[valid])) if np.any(valid) else None,
            "projection_delta_y_p50": float(np.median(delta_y[valid])) if np.any(valid) else None,
        }
        records.append(record)

        semantic_overlay = rgb_bgr.copy()
        semantic_overlay[~semantic] = (180, 30, 180)
        rows.append(np.hstack([
            tile(rgb_bgr, name),
            tile(colorize(depth, valid, 0.99, log=True), "camera log depth"),
            tile(colorize(confidence, valid, 0.99), "MASt3R confidence"),
            tile(colorize(projection_error, valid, 256.0), f"projection p50={record['projection_error_px']['p50']:.1f}px"),
            tile(semantic_overlay, f"semantic keep={semantic.mean():.3f}"),
        ]))

    projection_p50 = np.asarray([r["projection_error_px"]["p50"] for r in records], dtype=float)
    projection_p90 = np.asarray([r["projection_error_px"]["p90"] for r in records], dtype=float)
    report = {
        "mast3r_root": str(args.mast3r_root),
        "chart_count": len(records),
        "summary": {
            "all_finite": all(record["finite_fraction"] == 1.0 for record in records),
            "projection_p50_median_px": float(np.median(projection_p50)),
            "projection_p50_max_px": float(projection_p50.max()),
            "projection_p90_median_px": float(np.median(projection_p90)),
            "projection_p90_max_px": float(projection_p90.max()),
            "charts_projection_p50_gt_32px": int((projection_p50 > 32.0).sum()),
            "interpretation": "Raw pointmaps require chart alignment; these residuals are not a post-alignment pass criterion.",
        },
        "charts": records,
    }
    (args.output / "pointmap_audit.json").write_text(json.dumps(report, indent=2))
    for start in range(0, len(rows), 6):
        cv2.imwrite(str(args.output / f"pointmap_audit_{start // 6 + 1:02d}.jpg"), np.vstack(rows[start:start + 6]))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
