#!/usr/bin/env python3
"""Keep plane refinement local and fall back to aligned chart depth when unsafe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _read_depth(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise FileNotFoundError(path)
    return np.asarray(value, dtype=np.float32)


def _write_depth(path: Path, value: np.ndarray) -> None:
    if not cv2.imwrite(str(path), np.asarray(value, dtype=np.float32)):
        raise RuntimeError(f"Failed to write {path}")


def gate_plane_depth(
    aligned: np.ndarray,
    refined: np.ndarray,
    support: np.ndarray,
    *,
    min_support_fraction: float,
    max_relative_p90: float,
    max_gt25_fraction: float,
    max_pixel_relative_change: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    aligned = np.asarray(aligned, dtype=np.float32).squeeze()
    refined = np.asarray(refined, dtype=np.float32).squeeze()
    support = np.asarray(support).squeeze() > 0.5
    if aligned.shape != refined.shape or aligned.shape != support.shape:
        raise ValueError(
            f"Shape mismatch: aligned={aligned.shape}, refined={refined.shape}, support={support.shape}"
        )

    valid = (
        support
        & np.isfinite(aligned)
        & np.isfinite(refined)
        & (aligned > 0.0)
        & (refined > 0.0)
    )
    support_fraction = float(valid.mean())
    relative = np.full(aligned.shape, np.inf, dtype=np.float32)
    relative[valid] = np.abs(refined[valid] - aligned[valid]) / np.maximum(
        np.abs(aligned[valid]), 1e-6
    )
    if np.any(valid):
        relative_p90 = float(np.quantile(relative[valid], 0.9))
        gt25_fraction = float(np.mean(relative[valid] > 0.25))
    else:
        relative_p90 = None
        gt25_fraction = None

    reasons = []
    if support_fraction < min_support_fraction:
        reasons.append("insufficient_support")
    if relative_p90 is None or relative_p90 > max_relative_p90:
        reasons.append("relative_p90")
    if gt25_fraction is None or gt25_fraction > max_gt25_fraction:
        reasons.append("gt25_fraction")
    chart_fallback = bool(reasons)

    output = aligned.copy() if chart_fallback else refined.copy()
    output_support = support & np.isfinite(aligned) & (aligned > 0.0)
    local_fallback = np.zeros_like(support, dtype=bool)
    if not chart_fallback:
        local_fallback = (~valid) | (relative > max_pixel_relative_change)
        output[local_fallback] = aligned[local_fallback]
        output_support &= np.isfinite(output) & (output > 0.0)

    record = {
        "support_fraction": support_fraction,
        "relative_p90": relative_p90,
        "gt25_fraction": gt25_fraction,
        "chart_fallback": chart_fallback,
        "reasons": reasons,
        "local_fallback_fraction": float(local_fallback.mean()),
    }
    return output, output_support, record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plane-root", type=Path, required=True)
    parser.add_argument("--min-support-fraction", type=float, default=0.50)
    parser.add_argument("--max-relative-p90", type=float, default=0.25)
    parser.add_argument("--max-gt25-fraction", type=float, default=0.25)
    parser.add_argument("--max-pixel-relative-change", type=float, default=0.50)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.plane_root.expanduser().resolve()
    refined_paths = sorted(root.glob("refine_depth_frame*.tiff"))
    if not refined_paths:
        raise FileNotFoundError(f"No refine_depth_frame*.tiff in {root}")

    records = []
    for refined_path in refined_paths:
        stem = refined_path.stem.removeprefix("refine_depth_frame")
        aligned_path = root / f"depth_frame{stem}.tiff"
        support_path = root / f"visibility_frame{stem}.npy"
        if not support_path.is_file():
            raise FileNotFoundError(support_path)
        aligned = _read_depth(aligned_path)
        refined = _read_depth(refined_path)
        support = np.load(support_path)
        output, output_support, record = gate_plane_depth(
            aligned,
            refined,
            support,
            min_support_fraction=args.min_support_fraction,
            max_relative_p90=args.max_relative_p90,
            max_gt25_fraction=args.max_gt25_fraction,
            max_pixel_relative_change=args.max_pixel_relative_change,
        )
        backup = root / f"refine_depth_frame{stem}.pre_safety_gate.tiff"
        if not backup.exists():
            _write_depth(backup, refined)
        _write_depth(refined_path, output)
        confidence_path = root / f"confident_map_frame{stem}.png"
        Image.fromarray(np.uint8(output_support) * 255, mode="L").save(confidence_path)
        records.append({"frame": int(stem), **record})

    report = {
        "version": 1,
        "thresholds": {
            "min_support_fraction": args.min_support_fraction,
            "max_relative_p90": args.max_relative_p90,
            "max_gt25_fraction": args.max_gt25_fraction,
            "max_pixel_relative_change": args.max_pixel_relative_change,
        },
        "chart_fallback_count": sum(record["chart_fallback"] for record in records),
        "records": records,
    }
    report_path = args.report or root / "plane_refinement_safety_gate.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "chart_fallback_count": report["chart_fallback_count"],
        "report": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
