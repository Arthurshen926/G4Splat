#!/usr/bin/env python3
"""Audit an opacity-only Gaussian intervention over every real Cambridge view.

The local causal pass deliberately uses a small, geometry-aware control set to
identify a plausible repair.  This second pass is the non-negotiable global
regression gate: it compares a frozen model with a temporary opacity-zero
intervention over all staged train and official-test cameras.  It neither
writes a PLY nor changes the source checkpoint.

Per-view affine colour calibration is fitted on the baseline render only and
then held fixed for the intervention.  That isolates geometry/visibility
effects from per-view exposure compensation while preserving a raw metric in
the report for compatibility with ordinary reconstruction evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
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

from artifact_guided_repair.core import robust_affine_color  # noqa: E402
from causal_2dgs_repair import _LazyCameraStore, _LazyStaticMasks, _camera_gt, _json_ready  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402
from scene import GaussianModel  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render(camera: Any, gaussians: GaussianModel, pipe: Any, background: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        package = render(camera, gaussians, pipe, background)
    return (
        package["render"].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32),
        package["rend_alpha"][0].detach().cpu().numpy().astype(np.float32),
        package["visibility_filter"].detach().cpu().numpy().astype(bool),
    )


def _apply_affine(render_rgb: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return np.clip(
        render_rgb * affine[None, None, :, 0] + affine[None, None, :, 1],
        0.0,
        1.0,
    )


def _metrics(render_rgb: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Use the same scalar global-SSIM definition as evaluate_render_dir.py."""
    valid = np.asarray(mask, dtype=bool)
    if valid.shape != render_rgb.shape[:2] or gt.shape != render_rgb.shape:
        raise ValueError("Metric RGB/mask shapes disagree")
    if not np.any(valid):
        raise ValueError("Static semantic mask removed every pixel")
    prediction = render_rgb[valid].astype(np.float64, copy=False)
    target = gt[valid].astype(np.float64, copy=False)
    residual = prediction - target
    mse = float(np.mean(residual * residual))
    mu_x = float(prediction.mean())
    mu_y = float(target.mean())
    sigma_x = float(np.mean((prediction - mu_x) ** 2))
    sigma_y = float(np.mean((target - mu_y) ** 2))
    sigma_xy = float(np.mean((prediction - mu_x) * (target - mu_y)))
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return {
        "mae": float(np.mean(np.abs(residual))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "psnr": float(-10.0 * np.log10(max(mse, 1e-12))),
        "ssim": float(ssim),
        "pixel_count": int(valid.sum()),
    }


def _view_row(
    *,
    name: str,
    render_rgb: np.ndarray,
    alpha: np.ndarray,
    visible: np.ndarray,
    camera: Any,
    static_mask: np.ndarray,
    primitive_ids: np.ndarray,
    affine: np.ndarray | None,
) -> tuple[dict[str, Any], np.ndarray]:
    gt = _camera_gt(camera)
    if affine is None:
        _, affine = robust_affine_color(render_rgb, gt, static_mask & (alpha > 0.30))
    affine = np.asarray(affine, dtype=np.float32)
    return (
        {
            "name": str(name),
            "static_pixel_count": int(static_mask.sum()),
            "candidate_visible": bool(np.any(visible[primitive_ids])),
            "raw": _metrics(render_rgb, gt, static_mask),
            "fixed_affine": _metrics(_apply_affine(render_rgb, affine), gt, static_mask),
        },
        affine,
    )


def _discard_camera(cameras: _LazyCameraStore, name: str) -> None:
    """Keep the full-view audit bounded instead of retaining 2,017 RGB tensors."""
    cameras._cache.pop(name, None)


def _baseline_split(
    *,
    split_name: str,
    source_path: Path,
    mask_pickle: Path,
    gaussians: GaussianModel,
    pipe: Any,
    background: torch.Tensor,
    primitive_ids: np.ndarray,
    expected_view_count: int,
    resolution: int,
) -> tuple[_LazyCameraStore, _LazyStaticMasks, dict[str, dict[str, Any]]]:
    cameras = _LazyCameraStore.from_colmap(source_path, requested_resolution=resolution, data_device="cpu")
    if len(cameras) != expected_view_count:
        raise ValueError(
            f"{split_name} source has {len(cameras)} cameras, expected {expected_view_count}: {source_path}"
        )
    lookup = CambridgeMaskLookup(source_path, mask_pickle, [0, 1, 2])
    masks = _LazyStaticMasks(lookup, cameras.geometries)
    rows: dict[str, dict[str, Any]] = {}
    names = sorted(cameras)
    for index, name in enumerate(names, start=1):
        camera = cameras[name]
        rgb, alpha, visible = _render(camera, gaussians, pipe, background)
        row, affine = _view_row(
            name=name,
            render_rgb=rgb,
            alpha=alpha,
            visible=visible,
            camera=camera,
            static_mask=masks[name],
            primitive_ids=primitive_ids,
            affine=None,
        )
        row["affine"] = affine
        rows[name] = row
        _discard_camera(cameras, name)
        if index == 1 or index % 100 == 0 or index == len(names):
            print(f"[{split_name} baseline] {index}/{len(names)}", flush=True)
    return cameras, masks, rows


def _intervention_split(
    *,
    split_name: str,
    cameras: _LazyCameraStore,
    masks: _LazyStaticMasks,
    baseline: dict[str, dict[str, Any]],
    gaussians: GaussianModel,
    pipe: Any,
    background: torch.Tensor,
    primitive_ids: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = sorted(cameras)
    for index, name in enumerate(names, start=1):
        before = baseline[name]
        # ``visibility_filter`` is the rasterizer's non-zero screen-radius
        # predicate.  Opacity cannot change a primitive's radius, so an
        # invisible primitive cannot affect this camera under the intervention.
        # Reusing the baseline here is exact, while avoiding a second full
        # render of every back-facing Cambridge observation.
        if not bool(before["candidate_visible"]):
            rows.append(
                {
                    "name": name,
                    "candidate_visible_before": False,
                    "candidate_visible_after": False,
                    "intervention_rendered": False,
                    "baseline": {
                        "raw": before["raw"],
                        "fixed_affine": before["fixed_affine"],
                    },
                    "intervention": {
                        "raw": before["raw"],
                        "fixed_affine": before["fixed_affine"],
                    },
                    "delta": {
                        domain: {key: 0.0 for key in ("mae", "mse", "rmse", "psnr", "ssim")}
                        for domain in ("raw", "fixed_affine")
                    },
                }
            )
            if index == 1 or index % 100 == 0 or index == len(names):
                print(f"[{split_name} intervention] {index}/{len(names)} (baseline reuse)", flush=True)
            continue
        camera = cameras[name]
        rgb, alpha, visible = _render(camera, gaussians, pipe, background)
        after, _ = _view_row(
            name=name,
            render_rgb=rgb,
            alpha=alpha,
            visible=visible,
            camera=camera,
            static_mask=masks[name],
            primitive_ids=primitive_ids,
            affine=np.asarray(baseline[name]["affine"], dtype=np.float32),
        )
        row = {
            "name": name,
            "candidate_visible_before": bool(before["candidate_visible"]),
            "candidate_visible_after": bool(after["candidate_visible"]),
            "intervention_rendered": True,
            "baseline": {
                "raw": before["raw"],
                "fixed_affine": before["fixed_affine"],
            },
            "intervention": {
                "raw": after["raw"],
                "fixed_affine": after["fixed_affine"],
            },
        }
        row["delta"] = {
            domain: {
                key: float(after[domain][key] - before[domain][key])
                for key in ("mae", "mse", "rmse", "psnr", "ssim")
            }
            for domain in ("raw", "fixed_affine")
        }
        rows.append(row)
        _discard_camera(cameras, name)
        if index == 1 or index % 100 == 0 or index == len(names):
            print(f"[{split_name} intervention] {index}/{len(names)}", flush=True)
    return rows


def _domain_summary(rows: list[dict[str, Any]], domain: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty split")
    metric_keys = ("mae", "mse", "rmse", "psnr", "ssim")
    before = {
        key: float(np.mean([row["baseline"][domain][key] for row in rows]))
        for key in metric_keys
    }
    after = {
        key: float(np.mean([row["intervention"][domain][key] for row in rows]))
        for key in metric_keys
    }
    deltas = {
        key: float(np.mean([row["delta"][domain][key] for row in rows]))
        for key in metric_keys
    }
    weights = np.asarray([row["baseline"][domain]["pixel_count"] for row in rows], dtype=np.float64)
    before_mae = np.asarray([row["baseline"][domain]["mae"] for row in rows], dtype=np.float64)
    after_mae = np.asarray([row["intervention"][domain]["mae"] for row in rows], dtype=np.float64)
    before_mse = np.asarray([row["baseline"][domain]["mse"] for row in rows], dtype=np.float64)
    after_mse = np.asarray([row["intervention"][domain]["mse"] for row in rows], dtype=np.float64)
    weighted_before_mse = float(np.average(before_mse, weights=weights))
    weighted_after_mse = float(np.average(after_mse, weights=weights))
    mae_changes = np.asarray([row["delta"][domain]["mae"] for row in rows], dtype=np.float64)
    worst_index = int(np.argmax(mae_changes))
    best_index = int(np.argmin(mae_changes))
    return {
        "mean_per_view": {
            "baseline": before,
            "intervention": after,
            "delta": deltas,
        },
        "pixel_weighted": {
            "baseline_mae": float(np.average(before_mae, weights=weights)),
            "intervention_mae": float(np.average(after_mae, weights=weights)),
            "delta_mae": float(np.average(after_mae - before_mae, weights=weights)),
            "baseline_psnr": float(-10.0 * np.log10(max(weighted_before_mse, 1e-12))),
            "intervention_psnr": float(-10.0 * np.log10(max(weighted_after_mse, 1e-12))),
            "delta_psnr": float(-10.0 * np.log10(max(weighted_after_mse, 1e-12)) - -10.0 * np.log10(max(weighted_before_mse, 1e-12))),
        },
        "worst_mae_increase": {
            "name": rows[worst_index]["name"],
            "delta": float(mae_changes[worst_index]),
        },
        "best_mae_change": {
            "name": rows[best_index]["name"],
            "delta": float(mae_changes[best_index]),
        },
    }


def _split_summary(rows: list[dict[str, Any]], affected_mae_epsilon: float) -> dict[str, Any]:
    summary = {
        "view_count": int(len(rows)),
        "candidate_visible_before_count": int(sum(row["candidate_visible_before"] for row in rows)),
        "candidate_visible_after_count": int(sum(row["candidate_visible_after"] for row in rows)),
        "intervention_rendered_view_count": int(sum(row["intervention_rendered"] for row in rows)),
        "changed_view_count": int(
            sum(abs(float(row["delta"]["fixed_affine"]["mae"])) > affected_mae_epsilon for row in rows)
        ),
        "raw": _domain_summary(rows, "raw"),
        "fixed_affine": _domain_summary(rows, "fixed_affine"),
    }
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--train-source-path", type=Path, required=True)
    parser.add_argument("--train-mask-pickle", type=Path, required=True)
    parser.add_argument("--test-source-path", type=Path, required=True)
    parser.add_argument("--test-mask-pickle", type=Path, required=True)
    parser.add_argument("--primitive-ids", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train-views", type=int, default=1487)
    parser.add_argument("--expected-test-views", type=int, default=530)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--opacity-logit", type=float, default=-16.0)
    parser.add_argument("--max-mean-mae-increase", type=float, default=1e-4)
    parser.add_argument("--max-worst-view-mae-increase", type=float, default=0.006)
    parser.add_argument("--affected-mae-epsilon", type=float, default=1e-5)
    parser.add_argument("--white-background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if int(args.iteration) < 0:
        raise ValueError("--iteration must be non-negative")
    if int(args.expected_train_views) <= 0 or int(args.expected_test_views) <= 0:
        raise ValueError("Expected view counts must be positive")
    if float(args.max_mean_mae_increase) < 0.0 or float(args.max_worst_view_mae_increase) < 0.0:
        raise ValueError("MAE regression allowances must be non-negative")
    primitive_ids = np.unique(np.asarray(args.primitive_ids, dtype=np.int64))
    if not len(primitive_ids) or np.any(primitive_ids < 0):
        raise ValueError("--primitive-ids must contain non-negative IDs")
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
    source_paths = {
        "train": args.train_source_path.expanduser().resolve(),
        "test": args.test_source_path.expanduser().resolve(),
    }
    mask_pickles = {
        "train": args.train_mask_pickle.expanduser().resolve(),
        "test": args.test_mask_pickle.expanduser().resolve(),
    }
    for split_name, path in source_paths.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {split_name} source directory: {path}")
    for split_name, path in mask_pickles.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split_name} mask pickle: {path}")

    frozen_hash = _sha256(point_cloud)
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

    stores: dict[str, tuple[_LazyCameraStore, _LazyStaticMasks, dict[str, dict[str, Any]]]] = {}
    for split_name, expected_count in (("train", int(args.expected_train_views)), ("test", int(args.expected_test_views))):
        stores[split_name] = _baseline_split(
            split_name=split_name,
            source_path=source_paths[split_name],
            mask_pickle=mask_pickles[split_name],
            gaussians=gaussians,
            pipe=pipe,
            background=background,
            primitive_ids=primitive_ids,
            expected_view_count=expected_count,
            resolution=int(args.resolution),
        )

    ids = torch.as_tensor(primitive_ids, dtype=torch.long, device=gaussians._opacity.device)
    original_opacity = gaussians._opacity.detach()[ids].clone()
    split_rows: dict[str, list[dict[str, Any]]] = {}
    try:
        with torch.no_grad():
            gaussians._opacity[ids] = float(args.opacity_logit)
        for split_name in ("train", "test"):
            cameras, masks, baseline = stores[split_name]
            split_rows[split_name] = _intervention_split(
                split_name=split_name,
                cameras=cameras,
                masks=masks,
                baseline=baseline,
                gaussians=gaussians,
                pipe=pipe,
                background=background,
                primitive_ids=primitive_ids,
            )
    finally:
        with torch.no_grad():
            gaussians._opacity[ids] = original_opacity

    summaries = {
        split_name: _split_summary(rows, float(args.affected_mae_epsilon))
        for split_name, rows in split_rows.items()
    }
    gates = {}
    for split_name, summary in summaries.items():
        fixed = summary["fixed_affine"]
        mean_delta = float(fixed["mean_per_view"]["delta"]["mae"])
        worst_delta = float(fixed["worst_mae_increase"]["delta"])
        gates[split_name] = {
            "mean_mae_non_regression": bool(mean_delta <= float(args.max_mean_mae_increase)),
            "worst_view_mae_budget": bool(worst_delta <= float(args.max_worst_view_mae_increase)),
            "accepted": bool(
                mean_delta <= float(args.max_mean_mae_increase)
                and worst_delta <= float(args.max_worst_view_mae_increase)
            ),
        }
    after_hash = _sha256(point_cloud)
    if frozen_hash != after_hash:
        raise RuntimeError("The supposedly read-only audit changed the frozen point cloud")
    report = {
        "protocol": "frozen_2dgs_global_opacity_intervention_v1",
        "read_only": True,
        "model_path": str(model_path),
        "point_cloud": str(point_cloud),
        "point_cloud_sha256_before": frozen_hash,
        "point_cloud_sha256_after": after_hash,
        "iteration": int(args.iteration),
        "primitive_ids": [int(value) for value in primitive_ids],
        "opacity_logit": float(args.opacity_logit),
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "mask_pickles": {name: str(path) for name, path in mask_pickles.items()},
        "thresholds": {
            "max_mean_mae_increase": float(args.max_mean_mae_increase),
            "max_worst_view_mae_increase": float(args.max_worst_view_mae_increase),
            "affected_mae_epsilon": float(args.affected_mae_epsilon),
        },
        "summaries": summaries,
        "gates": gates,
        "accepted": bool(all(row["accepted"] for row in gates.values())),
        "per_view": split_rows,
    }
    (output_dir / "global_opacity_intervention_audit.json").write_text(
        json.dumps(_json_ready(report), indent=2) + "\n", encoding="utf-8"
    )
    print(f"[gate] accepted={report['accepted']} train={gates['train']['accepted']} test={gates['test']['accepted']}")


if __name__ == "__main__":
    main()
