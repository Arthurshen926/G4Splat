#!/usr/bin/env python3
"""Sweep a temporary Gaussian opacity against a verified full-view baseline.

This is the efficient second stage after
``audit_global_opacity_intervention.py``.  The baseline report proves the
checkpoint, camera sets, masks, and original metrics.  For each requested
logit this tool re-renders every camera in which the selected Gaussian had a
non-zero screen radius, reuses the exact frozen baseline for the rest, and
reports raw reconstruction metrics.  It never writes a PLY.

Fixed-affine target/control acceptance remains the responsibility of the
local causal audit.  This sweep deliberately uses raw metrics because the
choice among already locally safe opacity settings must improve the actual
reconstruction objective, not an exposure-compensated proxy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GS_ROOT = REPO_ROOT / "2d-gaussian-splatting"
for import_root in (REPO_ROOT, GS_ROOT, REPO_ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from audit_global_opacity_intervention import _metrics, _render, _sha256  # noqa: E402
from causal_2dgs_repair import _LazyCameraStore, _LazyStaticMasks, _camera_gt, _json_ready  # noqa: E402
from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402
from scene import GaussianModel  # noqa: E402


def _discard_camera(cameras: _LazyCameraStore, name: str) -> None:
    cameras._cache.pop(name, None)


def _source_rows(report: dict[str, Any], split_name: str, expected_count: int) -> dict[str, dict[str, Any]]:
    values = report.get("per_view", {}).get(split_name)
    if not isinstance(values, list) or len(values) != expected_count:
        raise ValueError(
            f"Baseline report has {0 if not isinstance(values, list) else len(values)} {split_name} rows; "
            f"expected {expected_count}"
        )
    rows = {str(row["name"]): row for row in values}
    if len(rows) != expected_count:
        raise ValueError(f"Baseline report has duplicate {split_name} camera names")
    for name, row in rows.items():
        if "raw" not in row.get("baseline", {}):
            raise ValueError(f"Baseline report lacks raw metrics for {split_name}/{name}")
    return rows


def _summarize(
    baseline: dict[str, dict[str, Any]],
    intervention: dict[str, dict[str, float]],
    *,
    changed_mae_epsilon: float,
) -> dict[str, Any]:
    names = sorted(baseline)
    metric_keys = ("mae", "mse", "rmse", "psnr", "ssim")
    before = {key: np.asarray([baseline[name]["baseline"]["raw"][key] for name in names], dtype=np.float64) for key in metric_keys}
    after = {
        key: np.asarray(
            [intervention.get(name, baseline[name]["baseline"]["raw"])[key] for name in names],
            dtype=np.float64,
        )
        for key in metric_keys
    }
    weights = np.asarray(
        [baseline[name]["baseline"]["raw"]["pixel_count"] for name in names], dtype=np.float64
    )
    mae_delta = after["mae"] - before["mae"]
    worst = int(np.argmax(mae_delta))
    best = int(np.argmin(mae_delta))
    weighted_before_mse = float(np.average(before["mse"], weights=weights))
    weighted_after_mse = float(np.average(after["mse"], weights=weights))
    return {
        "view_count": int(len(names)),
        "candidate_visible_view_count": int(sum(bool(baseline[name]["candidate_visible_before"]) for name in names)),
        "rerendered_visible_view_count": int(len(intervention)),
        "changed_view_count": int(np.count_nonzero(np.abs(mae_delta) > float(changed_mae_epsilon))),
        "mean_per_view": {
            "baseline": {key: float(before[key].mean()) for key in metric_keys},
            "intervention": {key: float(after[key].mean()) for key in metric_keys},
            "delta": {key: float((after[key] - before[key]).mean()) for key in metric_keys},
        },
        "pixel_weighted": {
            "baseline_mae": float(np.average(before["mae"], weights=weights)),
            "intervention_mae": float(np.average(after["mae"], weights=weights)),
            "delta_mae": float(np.average(after["mae"] - before["mae"], weights=weights)),
            "baseline_psnr": float(-10.0 * np.log10(max(weighted_before_mse, 1e-12))),
            "intervention_psnr": float(-10.0 * np.log10(max(weighted_after_mse, 1e-12))),
            "delta_psnr": float(
                -10.0 * np.log10(max(weighted_after_mse, 1e-12))
                + 10.0 * np.log10(max(weighted_before_mse, 1e-12))
            ),
        },
        "worst_mae_increase": {"name": names[worst], "delta": float(mae_delta[worst])},
        "best_mae_change": {"name": names[best], "delta": float(mae_delta[best])},
    }


def _render_logit(
    *,
    split_name: str,
    cameras: _LazyCameraStore,
    masks: _LazyStaticMasks,
    baseline: dict[str, dict[str, Any]],
    gaussians: GaussianModel,
    pipe: Any,
    background: torch.Tensor,
    primitive_ids: np.ndarray,
    logit: float,
) -> dict[str, dict[str, float]]:
    visible_names = [name for name in sorted(baseline) if bool(baseline[name]["candidate_visible_before"])]
    rows: dict[str, dict[str, float]] = {}
    for index, name in enumerate(visible_names, start=1):
        camera = cameras[name]
        rgb, _, _ = _render(camera, gaussians, pipe, background)
        rows[name] = _metrics(rgb, _camera_gt(camera), masks[name])
        _discard_camera(cameras, name)
        if index == 1 or index % 100 == 0 or index == len(visible_names):
            print(f"[{split_name} logit={logit:g}] {index}/{len(visible_names)}", flush=True)
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--train-source-path", type=Path, required=True)
    parser.add_argument("--train-mask-pickle", type=Path, required=True)
    parser.add_argument("--test-source-path", type=Path, required=True)
    parser.add_argument("--test-mask-pickle", type=Path, required=True)
    parser.add_argument("--primitive-ids", type=int, nargs="+", required=True)
    parser.add_argument("--opacity-logits", type=float, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train-views", type=int, default=1487)
    parser.add_argument("--expected-test-views", type=int, default=530)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--changed-mae-epsilon", type=float, default=1e-8)
    parser.add_argument("--white-background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    primitive_ids = np.unique(np.asarray(args.primitive_ids, dtype=np.int64))
    if not len(primitive_ids) or np.any(primitive_ids < 0):
        raise ValueError("--primitive-ids must contain non-negative IDs")
    logits = np.asarray(args.opacity_logits, dtype=np.float64)
    if not len(logits) or not np.isfinite(logits).all():
        raise ValueError("--opacity-logits must contain finite values")
    if int(args.expected_train_views) <= 0 or int(args.expected_test_views) <= 0:
        raise ValueError("Expected view counts must be positive")
    baseline_path = args.baseline_report.expanduser().resolve()
    baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline_report.get("protocol") != "frozen_2dgs_global_opacity_intervention_v1":
        raise ValueError(f"Unsupported baseline protocol: {baseline_report.get('protocol')}")
    baseline = {
        "train": _source_rows(baseline_report, "train", int(args.expected_train_views)),
        "test": _source_rows(baseline_report, "test", int(args.expected_test_views)),
    }
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.replace_output:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_path.expanduser().resolve()
    point_cloud = model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    if not point_cloud.is_file():
        raise FileNotFoundError(f"Frozen checkpoint does not exist: {point_cloud}")
    hash_before = _sha256(point_cloud)
    if hash_before != baseline_report.get("point_cloud_sha256_before"):
        raise ValueError("Baseline report was not produced from this exact frozen point cloud")
    gaussians = GaussianModel(3)
    gaussians.load_ply(str(point_cloud))
    if np.any(primitive_ids >= len(gaussians.get_xyz)):
        raise ValueError(f"A requested primitive ID is outside [0, {len(gaussians.get_xyz)})")
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, depth_ratio=0.0)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=gaussians.get_xyz.device,
    )
    source_paths = {
        "train": args.train_source_path.expanduser().resolve(),
        "test": args.test_source_path.expanduser().resolve(),
    }
    mask_pickles = {
        "train": args.train_mask_pickle.expanduser().resolve(),
        "test": args.test_mask_pickle.expanduser().resolve(),
    }
    stores: dict[str, tuple[_LazyCameraStore, _LazyStaticMasks]] = {}
    for split_name, expected_count in (("train", int(args.expected_train_views)), ("test", int(args.expected_test_views))):
        source = source_paths[split_name]
        mask_pickle = mask_pickles[split_name]
        if not source.is_dir() or not mask_pickle.is_file():
            raise FileNotFoundError(f"Missing {split_name} source or mask input")
        cameras = _LazyCameraStore.from_colmap(source, requested_resolution=int(args.resolution), data_device="cpu")
        if len(cameras) != expected_count or set(cameras) != set(baseline[split_name]):
            raise ValueError(f"{split_name} camera names do not exactly match the baseline report")
        lookup = CambridgeMaskLookup(source, mask_pickle, [0, 1, 2])
        stores[split_name] = (cameras, _LazyStaticMasks(lookup, cameras.geometries))

    ids = torch.as_tensor(primitive_ids, dtype=torch.long, device=gaussians._opacity.device)
    original = gaussians._opacity.detach()[ids].clone()
    trials = []
    try:
        for logit in logits.tolist():
            with torch.no_grad():
                gaussians._opacity[ids] = float(logit)
            per_split = {}
            per_view = {}
            for split_name in ("train", "test"):
                cameras, masks = stores[split_name]
                rows = _render_logit(
                    split_name=split_name,
                    cameras=cameras,
                    masks=masks,
                    baseline=baseline[split_name],
                    gaussians=gaussians,
                    pipe=pipe,
                    background=background,
                    primitive_ids=primitive_ids,
                    logit=float(logit),
                )
                per_split[split_name] = _summarize(
                    baseline[split_name], rows, changed_mae_epsilon=float(args.changed_mae_epsilon)
                )
                per_view[split_name] = {
                    name: {
                        "baseline": baseline[split_name][name]["baseline"]["raw"],
                        "intervention": value,
                        "delta": {
                            key: float(value[key] - baseline[split_name][name]["baseline"]["raw"][key])
                            for key in ("mae", "mse", "rmse", "psnr", "ssim")
                        },
                    }
                    for name, value in rows.items()
                }
            trials.append(
                {
                    "opacity_logit": float(logit),
                    "opacity": float(1.0 / (1.0 + np.exp(-float(logit)))),
                    "summaries": per_split,
                    "rerendered_per_view": per_view,
                }
            )
    finally:
        with torch.no_grad():
            gaussians._opacity[ids] = original
    hash_after = _sha256(point_cloud)
    if hash_before != hash_after:
        raise RuntimeError("The supposedly read-only sweep changed the frozen point cloud")
    report = {
        "protocol": "frozen_2dgs_visible_view_raw_opacity_sweep_v1",
        "read_only": True,
        "baseline_report": str(baseline_path),
        "model_path": str(model_path),
        "point_cloud": str(point_cloud),
        "point_cloud_sha256_before": hash_before,
        "point_cloud_sha256_after": hash_after,
        "iteration": int(args.iteration),
        "primitive_ids": [int(value) for value in primitive_ids],
        "opacity_logits": [float(value) for value in logits],
        "changed_mae_epsilon": float(args.changed_mae_epsilon),
        "trials": trials,
    }
    (output_dir / "global_raw_opacity_sweep.json").write_text(
        json.dumps(_json_ready(report), indent=2) + "\n", encoding="utf-8"
    )
    for trial in trials:
        train_delta = trial["summaries"]["train"]["mean_per_view"]["delta"]["mae"]
        test_delta = trial["summaries"]["test"]["mean_per_view"]["delta"]["mae"]
        print(f"[summary] logit={trial['opacity_logit']:g} train_mae_delta={train_delta:.8f} test_mae_delta={test_delta:.8f}")


if __name__ == "__main__":
    main()
