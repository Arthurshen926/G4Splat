#!/usr/bin/env python3
"""Paired statistical comparison for Cambridge held-out RGB metrics.

The renderer writes a metric per held-out camera.  Comparing only two global
means can turn sampling noise or an isolated failure into a promotion
decision, so this utility keeps the camera pairing intact and reports a
bootstrap confidence interval for each delta.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = {
    "raw_psnr": ("psnr",),
    "raw_ssim": ("ssim",),
    "raw_mae": ("mae",),
    "static_psnr": ("masked", "static_valid", "psnr"),
    "static_ssim": ("masked", "static_valid", "ssim"),
    "static_mae": ("masked", "static_valid", "mae"),
}


def _nested(record: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = record
    for key in path:
        value = value[key]
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"Non-finite metric at {'.'.join(path)}")
    return value


def _bootstrap_ci(
    deltas: np.ndarray,
    samples: int,
    seed: int,
    groups: np.ndarray | None = None,
) -> list[float]:
    if samples < 1:
        return [float(deltas.mean()), float(deltas.mean())]
    rng = np.random.default_rng(seed)
    if groups is None:
        indices = rng.integers(0, len(deltas), size=(samples, len(deltas)))
        means = deltas[indices].mean(axis=1)
    else:
        unique_groups, inverse = np.unique(groups, return_inverse=True)
        group_sums = np.bincount(inverse, weights=deltas)
        group_counts = np.bincount(inverse)
        # Sample sequences as units and retain every held-out frame drawn
        # with that sequence.  Adjacent frames are strongly correlated, so a
        # view-level bootstrap alone can overstate precision on Cambridge.
        draws = rng.integers(0, len(unique_groups), size=(samples, len(unique_groups)))
        means = group_sums[draws].sum(axis=1) / group_counts[draws].sum(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _sequence_key(record: dict[str, Any], fallback_view_name: str) -> str:
    source = str(record.get("source_image", fallback_view_name)).replace("\\", "/")
    parts = [part for part in source.split("/") if part]
    if len(parts) >= 2:
        return parts[-2]
    stem = Path(parts[-1] if parts else fallback_view_name).stem
    return stem.split("__", 1)[0]


def compare(
    baseline_path: Path,
    candidate_path: Path,
    *,
    bootstrap_samples: int,
    seed: int,
    bootstrap_unit: str = "view",
) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    common = sorted(set(baseline["per_view"]) & set(candidate["per_view"]))
    if not common:
        raise RuntimeError("No common held-out render names")
    if bootstrap_unit not in {"view", "sequence"}:
        raise ValueError(f"Unknown bootstrap unit: {bootstrap_unit}")
    groups = (
        np.asarray([_sequence_key(candidate["per_view"][view], view) for view in common])
        if bootstrap_unit == "sequence"
        else None
    )

    report: dict[str, Any] = {
        "baseline": str(baseline_path.resolve()),
        "candidate": str(candidate_path.resolve()),
        "common_view_count": len(common),
        "bootstrap": {
            "unit": bootstrap_unit,
            "unit_count": int(len(np.unique(groups))) if groups is not None else len(common),
        },
        "metrics": {},
    }
    for metric_index, (name, path) in enumerate(METRICS.items()):
        before = np.asarray(
            [_nested(baseline["per_view"][view], path) for view in common], dtype=np.float64
        )
        after = np.asarray(
            [_nested(candidate["per_view"][view], path) for view in common], dtype=np.float64
        )
        delta = after - before
        lower_is_better = name.endswith("mae")
        wins = delta < 0.0 if lower_is_better else delta > 0.0
        ci = _bootstrap_ci(delta, bootstrap_samples, seed + metric_index, groups)
        report["metrics"][name] = {
            "baseline_mean": float(before.mean()),
            "candidate_mean": float(after.mean()),
            "delta": float(delta.mean()),
            "candidate_win_fraction": float(wins.mean()),
            "bootstrap_95ci": ci,
            "direction": "lower" if lower_is_better else "higher",
            "positive_95ci": bool(ci[1] < 0.0 if lower_is_better else ci[0] > 0.0),
            "non_regressing_95ci": bool(ci[1] <= 0.0 if lower_is_better else ci[0] >= 0.0),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bootstrap-unit",
        choices=("view", "sequence"),
        default="view",
        help="Resample individual views or whole source sequences (more conservative).",
    )
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")

    report = compare(
        args.baseline,
        args.candidate,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        bootstrap_unit=args.bootstrap_unit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    compact = {
        name: {
            "delta": values["delta"],
            "ci95": values["bootstrap_95ci"],
            "wins": values["candidate_win_fraction"],
        }
        for name, values in report["metrics"].items()
    }
    print(json.dumps({"common_view_count": report["common_view_count"], "metrics": compact}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
