#!/usr/bin/env python3
"""Audit dense training renders against RGB GT and monocular depth priors."""

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

from matcha.cambridge_masks import combine_masks, load_mask_dict  # noqa: E402


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 16 or np.ptp(x) <= 1e-8 or np.ptp(y) <= 1e-8:
        return 0.0
    x_rank = np.empty(x.size, dtype=np.float32)
    y_rank = np.empty(y.size, dtype=np.float32)
    x_rank[np.argsort(x)] = np.arange(x.size, dtype=np.float32)
    y_rank[np.argsort(y)] = np.arange(y.size, dtype=np.float32)
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def colorize_depth(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    shown = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        lo, hi = np.quantile(depth[valid], [0.02, 0.98])
        shown = np.clip((depth - lo) / max(float(hi - lo), 1e-6) * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(shown, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(result, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("render_output", type=Path)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--mask-pickle", type=Path, required=True)
    parser.add_argument("--depth-cache", type=Path, required=True)
    parser.add_argument("--rgb-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--sample-stride", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb_report = json.loads(args.rgb_metrics.read_text())
    masks = load_mask_dict(args.mask_pickle)
    depth_cache = torch.load(args.depth_cache, map_location="cpu")
    staged_mapping = json.loads((args.dataset_path / "name_mapping.json").read_text())
    staged_names = sorted(staged_mapping, key=lambda name: Path(name).stem)
    cache_index = {name: index for index, name in enumerate(depth_cache["image_names"])}
    render_names = sorted(rgb_report["per_view"])
    if len(render_names) != len(staged_names):
        raise RuntimeError(f"Render/dataset count mismatch: {len(render_names)} vs {len(staged_names)}")

    records = []
    for index, render_name in enumerate(render_names):
        stem = Path(render_name).stem
        staged_name = staged_names[index]
        source_name = staged_mapping[staged_name]
        depth_path = args.render_output / "vis" / f"depth_{stem}.tiff"
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(depth_path)
        prior = depth_cache["depths"][cache_index[Path(staged_name).stem]].float().squeeze().numpy()
        if prior.shape != depth.shape:
            prior = cv2.resize(prior, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        static_keep = combine_masks(masks[source_name], [0, 1, 2], depth.shape, torch.device("cpu")).numpy()
        finite_positive = np.isfinite(depth) & (depth > 1e-6)
        valid = static_keep & finite_positive & np.isfinite(prior) & (prior > 1e-6)
        sampled = valid[:: args.sample_stride, :: args.sample_stride]
        pred_values = np.log(depth[:: args.sample_stride, :: args.sample_stride][sampled])
        prior_values = np.log(prior[:: args.sample_stride, :: args.sample_stride][sampled])
        correlation = rank_correlation(pred_values, prior_values)
        if pred_values.size:
            matrix = np.stack([pred_values, np.ones_like(pred_values)], axis=1)
            scale, offset = np.linalg.lstsq(matrix, prior_values, rcond=None)[0]
            aligned_residual = float(np.median(np.abs(scale * pred_values + offset - prior_values)))
        else:
            aligned_residual = float("inf")
        static_psnr = rgb_report["per_view"][render_name]["masked"]["static_valid"]["psnr"]
        positive_fraction = float((finite_positive & static_keep).sum() / max(int(static_keep.sum()), 1))
        record = {
            "render": render_name,
            "source_image": source_name,
            "static_psnr": float(static_psnr),
            "static_positive_depth_fraction": positive_fraction,
            "depth_prior_rank_correlation": correlation,
            "depth_prior_log_affine_median_residual": aligned_residual,
        }
        record["risk_score"] = float(
            max(0.0, 18.0 - static_psnr) / 6.0
            + 2.0 * (1.0 - positive_fraction)
            + max(0.0, 0.5 - correlation)
            + min(aligned_residual, 1.0)
        )
        records.append(record)

    report = {
        "num_views": len(records),
        "summary": {
            key: quantiles([record[key] for record in records])
            for key in (
                "static_psnr",
                "static_positive_depth_fraction",
                "depth_prior_rank_correlation",
                "depth_prior_log_affine_median_residual",
                "risk_score",
            )
        },
        "worst_views": sorted(records, key=lambda item: item["risk_score"], reverse=True)[: args.top_k],
        "per_view": records,
    }
    (args.output_dir / "training_depth_audit.json").write_text(json.dumps(report, indent=2))

    rows = []
    for record in report["worst_views"]:
        render_name = record["render"]
        stem = Path(render_name).stem
        index = render_names.index(render_name)
        staged_name = staged_names[index]
        depth = cv2.imread(str(args.render_output / "vis" / f"depth_{stem}.tiff"), cv2.IMREAD_UNCHANGED)
        prior = depth_cache["depths"][cache_index[Path(staged_name).stem]].float().squeeze().numpy()
        if prior.shape != depth.shape:
            prior = cv2.resize(prior, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        gt = cv2.imread(str(args.render_output / "gt" / render_name), cv2.IMREAD_COLOR)
        render = cv2.imread(str(args.render_output / "renders" / render_name), cv2.IMREAD_COLOR)
        error = cv2.absdiff(gt, render)
        valid_depth = np.isfinite(depth) & (depth > 1e-6)
        tiles = [
            label(gt, f"GT {record['source_image']}"),
            label(render, f"render static PSNR {record['static_psnr']:.2f}"),
            label(error, "absolute RGB error"),
            label(colorize_depth(depth, valid_depth), f"render depth valid {record['static_positive_depth_fraction']:.3f}"),
            label(colorize_depth(prior, np.isfinite(prior) & (prior > 0)), f"DAV2 rho {record['depth_prior_rank_correlation']:.3f}"),
        ]
        tile_width = 384
        tiles = [cv2.resize(tile, (tile_width, int(tile.shape[0] * tile_width / tile.shape[1]))) for tile in tiles]
        rows.append(np.concatenate(tiles, axis=1))
    if rows:
        sheet = np.concatenate(rows, axis=0)
        cv2.imwrite(str(args.output_dir / "training_depth_worst.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
