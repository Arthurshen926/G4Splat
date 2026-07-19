#!/usr/bin/env python3
"""Audit plane-refined chart depths before Gaussian initialization."""

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


def read(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    value = cv2.imread(str(path), flags)
    if value is None:
        raise FileNotFoundError(path)
    return value


def summarize(values: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    selected = values[mask & np.isfinite(values)]
    if not selected.size:
        return {key: None for key in ("p01", "p50", "p90", "p99", "max")}
    q = np.quantile(selected, [0.01, 0.50, 0.90, 0.99])
    return {"p01": float(q[0]), "p50": float(q[1]), "p90": float(q[2]), "p99": float(q[3]), "max": float(selected.max())}


def colorize(values: np.ndarray, mask: np.ndarray, *, relative: bool = False) -> np.ndarray:
    valid = mask & np.isfinite(values)
    output = np.zeros(values.shape, dtype=np.uint8)
    if np.any(valid):
        display = values if relative else np.log(np.maximum(values, 1e-6))
        upper = 0.75 if relative else float(np.quantile(display[valid], 0.99))
        lower = 0.0 if relative else float(np.quantile(display[valid], 0.01))
        output[valid] = np.round(np.clip((display[valid] - lower) / max(upper - lower, 1e-6), 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(output, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def tile(image: np.ndarray, title: str, width: int = 240, height: int = 135) -> np.ndarray:
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    canvas = np.full((height + 24, width, 3), 245, dtype=np.uint8)
    canvas[24:] = image
    cv2.putText(canvas, title[:38], (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mast3r-root", type=Path, required=True)
    parser.add_argument("--plane-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-pickle", type=Path, required=True)
    parser.add_argument("--mask-dataset-path", type=Path, required=True)
    parser.add_argument("--mask-indices", type=int, nargs="*", default=[0, 1, 2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cameras = json.loads((args.mast3r_root / "cameras.json").read_text())
    lookup = CambridgeMaskLookup(args.mask_dataset_path, args.mask_pickle, mask_indices=args.mask_indices)
    records, rows = [], []

    for index, filepath in enumerate(cameras["filepaths"]):
        name = Path(filepath).stem
        source = read(Path(filepath), cv2.IMREAD_COLOR)
        aligned = read(args.plane_root / f"depth_frame{index:06d}.tiff").astype(np.float32)
        refined = read(args.plane_root / f"refine_depth_frame{index:06d}.tiff").astype(np.float32)
        confidence = read(args.plane_root / f"confident_map_frame{index:06d}.png")
        if confidence.ndim == 3:
            confidence = confidence[..., 0]
        confidence = confidence > 127
        semantic = lookup.get_mask(name, refined.shape, torch.device("cpu")).cpu().numpy().astype(bool)
        finite = np.isfinite(refined) & np.isfinite(aligned)
        valid = finite & semantic & confidence & (refined > 0) & (aligned > 0)
        relative = np.zeros_like(refined)
        relative[valid] = np.abs(refined[valid] - aligned[valid]) / np.maximum(aligned[valid], 1e-6)
        excluded = ~semantic
        record = {
            "index": index,
            "name": name,
            "finite_fraction": float(finite.mean()),
            "semantic_keep_fraction": float(semantic.mean()),
            "confidence_fraction": float(confidence.mean()),
            "valid_fraction": float(valid.mean()),
            "excluded_nonzero_fraction": float(((refined > 0) & excluded).sum() / max(int(excluded.sum()), 1)),
            "refined_depth": summarize(refined, valid),
            "relative_change": summarize(relative, valid),
        }
        records.append(record)
        overlay = cv2.resize(source, (refined.shape[1], refined.shape[0]), interpolation=cv2.INTER_AREA)
        overlay[~semantic] = (180, 30, 180)
        rows.append(np.hstack([
            tile(source, name),
            tile(colorize(aligned, valid), "aligned chart depth"),
            tile(colorize(refined, valid), "plane-refined depth"),
            tile(colorize(relative, valid, relative=True), f"change p90={record['relative_change']['p90']:.3f}"),
            tile((confidence.astype(np.uint8) * 255), f"confidence={confidence.mean():.3f}"),
            tile(overlay, f"semantic keep={semantic.mean():.3f}"),
        ]))

    p90 = np.asarray([record["relative_change"]["p90"] for record in records], dtype=float)
    report = {
        "chart_count": len(records),
        "summary": {
            "all_finite": all(record["finite_fraction"] == 1.0 for record in records),
            "mean_confidence_fraction": float(np.mean([record["confidence_fraction"] for record in records])),
            "mean_valid_fraction": float(np.mean([record["valid_fraction"] for record in records])),
            "max_excluded_nonzero_fraction": float(max(record["excluded_nonzero_fraction"] for record in records)),
            "relative_change_p90_median": float(np.median(p90)),
            "relative_change_p90_max": float(p90.max()),
        },
        "charts": records,
    }
    (args.output / "plane_refinement_audit.json").write_text(json.dumps(report, indent=2))
    ordered = sorted(zip(records, rows), key=lambda pair: pair[0]["relative_change"]["p90"], reverse=True)
    for start in range(0, len(ordered), 6):
        cv2.imwrite(str(args.output / f"plane_refinement_worst_{start // 6 + 1:02d}.jpg"), np.vstack([row for _, row in ordered[start:start + 6]]))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
