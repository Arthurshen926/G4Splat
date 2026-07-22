#!/usr/bin/env python3
"""Compare two all-view render-metric reports with paired uncertainty.

The aggregate values in ``evaluate_render_dir.py`` are means over the same
training cameras, so an ablation should be judged from paired per-view
differences rather than from two rounded global numbers.  This utility keeps
the comparison protocol explicit, including the smaller valid subset of the
tree metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "raw": (),
    "dynamic_valid": ("masked", "dynamic_valid"),
    "static_valid": ("masked", "static_valid"),
    "ulfloc_legacy": ("ulfloc_legacy",),
    "non_tree_static": ("tree_stratified", "non_tree_static"),
    "tree_static": ("tree_stratified", "tree_static"),
}
_SCALAR_METRICS = ("psnr", "ssim", "mae", "rmse")


def _read_report(path: Path) -> dict[str, Any]:
    path = path.resolve()
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read render metrics at {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("per_view"), dict):
        raise RuntimeError(f"Render metrics at {path} do not contain a per_view mapping")
    return payload


def _descend(mapping: dict[str, Any], path: Iterable[str]) -> dict[str, Any] | None:
    value: Any = mapping
    for component in path:
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    return value if isinstance(value, dict) else None


def _aggregate_mapping(report: dict[str, Any], metric_path: tuple[str, ...]) -> dict[str, Any] | None:
    if metric_path == ():
        value = report.get("mean")
    elif metric_path == ("masked", "dynamic_valid"):
        value = report.get("masked", {}).get("dynamic_valid", {}).get("mean")
    elif metric_path == ("masked", "static_valid"):
        value = report.get("masked", {}).get("static_valid", {}).get("mean")
    elif metric_path == ("ulfloc_legacy",):
        value = report.get("protocol", {}).get("ulfloc_legacy", {}).get("mean")
    else:
        value = _descend(report, metric_path)
        value = value.get("mean") if isinstance(value, dict) else None
    return value if isinstance(value, dict) else None


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty metric series")
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")

    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    # Chunking avoids materializing a samples-by-views matrix for every metric.
    remaining = int(samples)
    while remaining:
        batch = min(remaining, 512)
        indices = rng.integers(0, values.size, size=(batch, values.size))
        means.append(values[indices].mean(axis=1))
        remaining -= batch
    draws = np.concatenate(means)
    tail = (1.0 - confidence) * 0.5
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))


def _per_view_values(
    view: dict[str, Any],
    metric_path: tuple[str, ...],
    scalar: str,
) -> float | None:
    value = _descend(view, metric_path)
    if value is None:
        return None
    raw = value.get(scalar)
    if raw is None or not np.isfinite(float(raw)):
        return None
    return float(raw)


def _view_identity_index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index metrics by real source image, not renderer-specific filename.

    The native streaming renderer writes numeric sorted-camera indices while a
    clean external gsplat renderer writes staged image names.  Both evaluators
    record ``source_image`` after resolving those filenames, which is the only
    identity suitable for a same-PLY cross-renderer paired comparison.
    """
    # MAtCha's historical ``rgb_metrics_tree_masks.json`` stores masked
    # per-view values in root-level parallel maps, while the current G4
    # evaluator nests them in each ``per_view`` item.  Normalize the former
    # shape here so a paired comparison is based on the same source images
    # rather than silently dropping every masked metric from an otherwise
    # valid MAtCha-vs-G4 comparison.
    root_masked = report.get("masked", {})
    if not isinstance(root_masked, dict):
        root_masked = {}

    indexed: dict[str, dict[str, Any]] = {}
    for render_name, view in report["per_view"].items():
        if not isinstance(view, dict):
            raise RuntimeError(f"Per-view metric entry {render_name!r} is not a mapping")
        normalized = dict(view)
        nested_masked = normalized.get("masked", {})
        if not isinstance(nested_masked, dict):
            nested_masked = {}
        else:
            nested_masked = dict(nested_masked)
        for section in ("dynamic_valid", "static_valid"):
            if section in nested_masked:
                continue
            parallel = root_masked.get(section, {})
            if not isinstance(parallel, dict):
                continue
            per_view = parallel.get("per_view", {})
            if not isinstance(per_view, dict):
                continue
            metric = per_view.get(render_name)
            if isinstance(metric, dict):
                nested_masked[section] = metric
        if nested_masked:
            normalized["masked"] = nested_masked
        identity = str(view.get("source_image", render_name))
        if identity in indexed:
            raise RuntimeError(
                "Render metric report maps multiple outputs to the same source image: "
                f"{identity!r}"
            )
        indexed[identity] = normalized
    return indexed


def build_metric_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    bootstrap_samples: int = 20_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Build paired deltas and bootstrap intervals for every shared scalar."""
    baseline_index = _view_identity_index(baseline)
    candidate_index = _view_identity_index(candidate)
    baseline_views = set(baseline_index)
    candidate_views = set(candidate_index)
    shared_views = sorted(baseline_views & candidate_views)
    if not shared_views:
        raise RuntimeError("The two reports do not share any per-view image names")

    aggregate_comparable = baseline_views == candidate_views
    sections: dict[str, Any] = {}
    for section, metric_path in _METRIC_PATHS.items():
        scalar_results: dict[str, Any] = {}
        aggregate_before = _aggregate_mapping(baseline, metric_path) or {}
        aggregate_after = _aggregate_mapping(candidate, metric_path) or {}
        for scalar in _SCALAR_METRICS:
            paired = [
                (
                    _per_view_values(baseline_index[name], metric_path, scalar),
                    _per_view_values(candidate_index[name], metric_path, scalar),
                )
                for name in shared_views
            ]
            paired = [
                (before, after)
                for before, after in paired
                if before is not None and after is not None
            ]
            before_values = np.asarray([before for before, _ in paired], dtype=np.float64)
            after_values = np.asarray([after for _, after in paired], dtype=np.float64)
            values = after_values - before_values
            if values.size == 0:
                continue
            low, high = _bootstrap_mean_ci(
                values,
                samples=bootstrap_samples,
                seed=seed + len(sections) * 17 + len(scalar_results),
                confidence=confidence,
            )
            before = aggregate_before.get(scalar)
            after = aggregate_after.get(scalar)
            scalar_results[scalar] = {
                "paired_view_count": int(values.size),
                # Root-level means are only directly comparable when both
                # reports cover the same camera set.  In particular, the
                # retained_v2 MAtCha report has 64 renders while a G4 full
                # control has all 1,487; subtracting those two global means
                # would be a false cross-method result.  Always expose the
                # matched-view means, and expose a root aggregate delta only
                # when the two report populations coincide.
                "paired_baseline_mean": float(before_values.mean()),
                "paired_candidate_mean": float(after_values.mean()),
                "reported_baseline_aggregate": float(before) if before is not None else None,
                "reported_candidate_aggregate": float(after) if after is not None else None,
                "aggregate_comparable": bool(aggregate_comparable),
                "baseline_aggregate": (
                    float(before) if aggregate_comparable and before is not None else None
                ),
                "candidate_aggregate": (
                    float(after) if aggregate_comparable and after is not None else None
                ),
                "aggregate_delta": (
                    float(after) - float(before)
                    if aggregate_comparable and before is not None and after is not None
                    else None
                ),
                "paired_mean_delta": float(values.mean()),
                "paired_median_delta": float(np.median(values)),
                "candidate_win_fraction": float(np.mean(values > 0.0)),
                "bootstrap_mean_ci": [low, high],
                "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
            }
        if scalar_results:
            sections[section] = scalar_results

    return {
        "protocol": "paired_render_metric_comparison_v1",
        "view_identity": {
            "baseline_view_count": len(baseline_views),
            "candidate_view_count": len(candidate_views),
            "shared_view_count": len(shared_views),
            "baseline_only_views": sorted(baseline_views - candidate_views),
            "candidate_only_views": sorted(candidate_views - baseline_views),
        },
        "bootstrap": {
            "samples": int(bootstrap_samples),
            "seed": int(seed),
            "confidence": float(confidence),
            "unit": "paired camera view",
        },
        "metrics": sections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite metric-comparison report: {output}")
    comparison = build_metric_comparison(
        _read_report(args.baseline),
        _read_report(args.candidate),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        confidence=args.confidence,
    )
    comparison["baseline"] = str(args.baseline.resolve())
    comparison["candidate"] = str(args.candidate.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
