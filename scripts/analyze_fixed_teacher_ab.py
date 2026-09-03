#!/usr/bin/env python
"""Analyze fixed-view canonical Teacher renders across two exact implementations.

The historical side is intentionally consumed from PNGs emitted when that
Teacher's own implementation hash was current.  This avoids reinterpreting an
old state with a newer renderer merely to make an A/B convenient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_hybrid_teacher import (
    _high_frequency_metrics,
    _protocol_metrics,
    _tree_boundary_masks,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _metric_bundle(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    protocol = _protocol_metrics(prediction, target, mask)
    high_frequency = _high_frequency_metrics(prediction, target, mask)
    return {
        **(protocol or {}),
        "high_frequency": high_frequency,
        "pixels": int(mask.sum()),
    }


def _delta(new: dict, old: dict) -> dict:
    result = {}
    for key in ("psnr", "ssim", "mae", "rmse"):
        if key in new and key in old:
            result[key] = float(new[key] - old[key])
    new_hf = new.get("high_frequency") or {}
    old_hf = old.get("high_frequency") or {}
    result["high_frequency"] = {
        key: float(new_hf[key] - old_hf[key])
        for key in new_hf.keys() & old_hf.keys()
    }
    return result


def _masked_error(error: np.ndarray, mask: np.ndarray, scale: float) -> np.ndarray:
    rgb = plt.get_cmap("magma")(np.clip(error / scale, 0.0, 1.0))[..., :3]
    return np.where(mask[..., None], rgb, 0.92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-render-dir", type=Path, required=True)
    parser.add_argument("--new-render-dir", type=Path, required=True)
    parser.add_argument("--old-result", type=Path, required=True)
    parser.add_argument("--old-metrics", type=Path)
    parser.add_argument("--new-metrics", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--indices",
        default="",
        help=(
            "Comma-separated camera indices. By default consume every view "
            "listed by --new-metrics; no historical three-frame fallback."
        ),
    )
    parser.add_argument("--old-label", default="old")
    parser.add_argument("--new-label", default="new")
    args = parser.parse_args()

    old_dir = args.old_render_dir.resolve()
    new_dir = args.new_render_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    new_metrics = json.loads(args.new_metrics.read_text())
    if args.indices.strip():
        indices = [
            int(value)
            for value in args.indices.split(",")
            if value.strip()
        ]
    else:
        per_view = new_metrics.get("per_view")
        if not isinstance(per_view, list) or not per_view:
            raise ValueError(
                "--new-metrics has no non-empty per_view list; pass "
                "--indices explicitly"
            )
        indices = [int(row["index"]) for row in per_view]
    old_cameras = {
        int(row["id"]): str(row["img_name"])
        for row in json.loads((old_dir / "cameras.json").read_text())
    }
    image_names = {
        int(row["id"]): str(row["img_name"])
        for row in json.loads((new_dir / "cameras.json").read_text())
    }
    missing_names = sorted(set(indices) - set(image_names))
    if missing_names:
        raise ValueError(f"No immutable image-name binding for indices {missing_names}")
    mismatched_names = [
        index for index in indices if old_cameras.get(index) != image_names[index]
    ]
    if mismatched_names:
        raise RuntimeError(
            "Camera/image binding differs across A/B for views "
            f"{mismatched_names}"
        )

    old_result = json.loads(args.old_result.read_text())
    old_metrics_report = (
        json.loads(args.old_metrics.read_text()) if args.old_metrics else None
    )
    lookup = CambridgeMaskLookup(
        args.dataset.resolve(), args.tree_mask_pickle.resolve(), [0, 1, 2, 3]
    )
    rows = []
    panel_rows = []
    region_panel_rows = []
    causal_panel_rows = []

    for index in indices:
        prefix = f"{index:05d}"
        paths = {
            "old": old_dir / f"{prefix}_canonical.png",
            "new": new_dir / f"{prefix}_canonical.png",
            "gt_old": old_dir / f"{prefix}_gt.png",
            "gt_new": new_dir / f"{prefix}_gt.png",
            "alpha": new_dir / f"{prefix}_canonical_volume_alpha.png",
            "surface_only": new_dir / f"{prefix}_surface_only.png",
            "volume_only": new_dir / f"{prefix}_volume_only.png",
        }
        absent = [str(path) for path in paths.values() if not path.exists()]
        if absent:
            raise FileNotFoundError(f"Missing fixed-view artifacts: {absent}")
        if _sha256(paths["gt_old"]) != _sha256(paths["gt_new"]):
            raise RuntimeError(f"Ground truth differs across A/B for view {index}")

        target = _image(paths["gt_new"])
        old = _image(paths["old"])
        new = _image(paths["new"])
        alpha = _image(paths["alpha"])[0]
        surface = _image(paths["surface_only"])
        volume = _image(paths["volume_only"])
        height, width = target.shape[-2:]
        keep = lookup.get_index_masks(
            image_names[index], (0, 1, 2, 3), (height, width), torch.device("cpu")
        )
        object_keep, sky_keep, distortion_keep, tree_keep = keep
        static_keep = object_keep & sky_keep & distortion_keep
        boundary_inside, boundary_outside, boundary_radius = _tree_boundary_masks(tree_keep)
        regions = {
            "canopy_core": static_keep & ~tree_keep & ~boundary_inside,
            "canopy_all": static_keep & ~tree_keep,
            "canopy_boundary_inside": static_keep & boundary_inside,
            "rigid_boundary_outside": static_keep & boundary_outside,
            "known_rigid": static_keep & tree_keep,
        }
        region_metrics = {}
        for name, mask in regions.items():
            old_region_metrics = _metric_bundle(old, target, mask)
            new_metrics_for_region = _metric_bundle(new, target, mask)
            region_metrics[name] = {
                "old": old_region_metrics,
                "new": new_metrics_for_region,
                "delta_new_minus_old": _delta(
                    new_metrics_for_region, old_region_metrics
                ),
            }

        new_error = (new - target).abs().mean(0)
        old_error = (old - target).abs().mean(0)
        surface_error = (surface - target).abs().mean(0)
        rigid_boundary = regions["rigid_boundary_outside"]
        visible_volume = alpha > (1.0 / 255.0)
        material_volume = alpha >= 0.05
        harmful = rigid_boundary & material_volume & (new_error > surface_error + 1.0 / 255.0)
        beneficial = rigid_boundary & material_volume & (surface_error > new_error + 1.0 / 255.0)
        rigid_count = int(rigid_boundary.sum())
        causal = {
            "semantics": (
                "new-side causal attribution on semantic rigid proxy immediately outside "
                "the authoritative canopy; not a building-instance segmentation"
            ),
            "rigid_boundary_pixels": rigid_count,
            "mean_volume_alpha": float(alpha[rigid_boundary].mean()) if rigid_count else None,
            "visible_volume_fraction_alpha_gt_1_over_255": (
                float((rigid_boundary & visible_volume).sum() / max(rigid_count, 1))
            ),
            "material_volume_fraction_alpha_ge_0_05": (
                float((rigid_boundary & material_volume).sum() / max(rigid_count, 1))
            ),
            "harmful_volume_fraction": float(harmful.sum() / max(rigid_count, 1)),
            "beneficial_volume_fraction": float(beneficial.sum() / max(rigid_count, 1)),
            "mixed_mae": float(new_error[rigid_boundary].mean()) if rigid_count else None,
            "surface_only_mae": float(surface_error[rigid_boundary].mean()) if rigid_count else None,
            "positive_mixed_minus_surface_mae": (
                float((new_error - surface_error).clamp_min(0)[rigid_boundary].mean())
                if rigid_count
                else None
            ),
        }
        rows.append(
            {
                "index": index,
                "image_name": image_names[index],
                "ground_truth_sha256": _sha256(paths["gt_new"]),
                "boundary_radius_pixels": boundary_radius,
                "regions": region_metrics,
                "new_rigid_boundary_volume_causal": causal,
            }
        )

        target_np = target.permute(1, 2, 0).numpy()
        old_np = old.permute(1, 2, 0).numpy()
        new_np = new.permute(1, 2, 0).numpy()
        error_scale = max(float(torch.quantile(torch.cat([old_error, new_error]), 0.98)), 1e-6)
        improvement = (old_error - new_error).numpy()
        improvement_scale = max(float(np.quantile(np.abs(improvement), 0.98)), 1e-6)
        delta_rgb = plt.get_cmap("coolwarm")(
            np.clip(improvement / improvement_scale * 0.5 + 0.5, 0.0, 1.0)
        )[..., :3]
        panel_rows.append(
            [
                (target_np, "Ground truth"),
                (old_np, f"{args.old_label} exact render"),
                (new_np, f"{args.new_label} exact render"),
                (plt.get_cmap("magma")(np.clip(old_error.numpy() / error_scale, 0, 1))[..., :3], f"{args.old_label} |error|"),
                (plt.get_cmap("magma")(np.clip(new_error.numpy() / error_scale, 0, 1))[..., :3], f"{args.new_label} |error|"),
                (delta_rgb, f"red: {args.new_label} better / blue: worse"),
            ]
        )
        region_panel_rows.append(
            [
                (_masked_error(old_error.numpy(), regions["canopy_core"].numpy(), error_scale), f"{args.old_label} canopy core error"),
                (_masked_error(new_error.numpy(), regions["canopy_core"].numpy(), error_scale), f"{args.new_label} canopy core error"),
                (_masked_error(old_error.numpy(), regions["canopy_boundary_inside"].numpy(), error_scale), f"{args.old_label} inner boundary"),
                (_masked_error(new_error.numpy(), regions["canopy_boundary_inside"].numpy(), error_scale), f"{args.new_label} inner boundary"),
                (_masked_error(old_error.numpy(), rigid_boundary.numpy(), error_scale), f"{args.old_label} rigid-side boundary"),
                (_masked_error(new_error.numpy(), rigid_boundary.numpy(), error_scale), f"{args.new_label} rigid-side boundary"),
            ]
        )
        causal_panel_rows.append(
            [
                (target_np, "Ground truth"),
                (new_np, f"{args.new_label} mixed"),
                (surface.permute(1, 2, 0).numpy(), f"{args.new_label} surface-only"),
                (volume.permute(1, 2, 0).numpy(), f"{args.new_label} volume-only"),
                (plt.get_cmap("viridis")(alpha.numpy())[..., :3], f"{args.new_label} intrinsic volume alpha"),
                (np.where(harmful.numpy()[..., None], np.array([1.0, 0.0, 0.0]), 0.9), "harmful alpha on rigid boundary"),
            ]
        )

    def save_panel(rows_to_plot, filename: str, title: str) -> None:
        figure, axes = plt.subplots(
            len(rows_to_plot), len(rows_to_plot[0]), figsize=(24, 4.2 * len(rows_to_plot)),
            constrained_layout=True,
        )
        axes = np.asarray(axes).reshape(len(rows_to_plot), len(rows_to_plot[0]))
        for row_index, items in enumerate(rows_to_plot):
            for axis, (image, label) in zip(axes[row_index], items):
                axis.imshow(np.clip(image, 0.0, 1.0))
                axis.set_title(label)
                axis.axis("off")
            axes[row_index, 0].set_ylabel(str(indices[row_index]), fontsize=13)
        figure.suptitle(title, fontsize=16)
        figure.savefig(output / filename, dpi=150)
        plt.close(figure)

    save_panel(panel_rows, "fixed_views_ab.png", "Exact-render fixed-view A/B")
    save_panel(region_panel_rows, "region_errors_ab.png", "Semantic region error A/B")
    save_panel(causal_panel_rows, "new_rigid_boundary_causal.png", f"{args.new_label} rigid-boundary volume causal audit")

    aggregate = {}
    for region in rows[0]["regions"]:
        aggregate[region] = {}
        for side in ("old", "new"):
            all_metrics = [row["regions"][region][side] for row in rows]
            metrics = [metric for metric in all_metrics if "mae" in metric]
            if not metrics:
                aggregate[region][side] = None
                continue
            aggregate[region][side] = {
                key: float(np.mean([metric[key] for metric in metrics]))
                for key in ("psnr", "ssim", "mae", "rmse")
            }
            aggregate[region][side]["high_frequency"] = {
                key: float(np.mean([metric["high_frequency"][key] for metric in metrics]))
                for key in metrics[0]["high_frequency"]
            }
            aggregate[region][side]["mean_pixels_per_view"] = float(
                np.mean([metric["pixels"] for metric in metrics])
            )
            aggregate[region][side]["evaluated_view_count"] = len(metrics)
        aggregate[region]["delta_new_minus_old"] = (
            _delta(aggregate[region]["new"], aggregate[region]["old"])
            if aggregate[region]["new"] is not None
            and aggregate[region]["old"] is not None
            else None
        )

    causal_keys = [
        "mean_volume_alpha",
        "visible_volume_fraction_alpha_gt_1_over_255",
        "material_volume_fraction_alpha_ge_0_05",
        "harmful_volume_fraction",
        "beneficial_volume_fraction",
        "mixed_mae",
        "surface_only_mae",
        "positive_mixed_minus_surface_mae",
    ]
    aggregate_causal = {
        key: float(
            np.mean(
                [
                    row["new_rigid_boundary_volume_causal"][key]
                    for row in rows
                    if row["new_rigid_boundary_volume_causal"][key] is not None
                ]
            )
        )
        for key in causal_keys
    }
    payload = {
        "schema_version": 2,
        "comparison": f"{args.new_label}_minus_{args.old_label}",
        "fixed_indices": indices,
        "canonical_sequence": "seq2",
        "contract": {
            "old_render_source": "old exact-implementation saved render",
            "new_render_source": "new exact-implementation direct evaluation",
            "old_iteration": (
                old_metrics_report.get("checkpoint_iteration")
                if old_metrics_report is not None
                else old_result.get("iterations")
            ),
            "new_iteration": new_metrics.get("checkpoint_iteration"),
            "new_implementation_validation": new_metrics.get("render_implementation_validation"),
            "camera_geometry_sha256": new_metrics["metric_protocol"]["camera_geometry_sha256"],
            "rgb_content_mapping_sha256": new_metrics["metric_protocol"]["rgb_source"]["content_mapping_sha256"],
            "tree_mask_sha256": new_metrics["metric_protocol"]["tree_mask_sha256"],
            "all_ground_truth_pngs_byte_identical": True,
            "metric_raster": "saved_uint8_png",
            "building_limitation": (
                "No building-instance GT is present. rigid_boundary_outside is the exact "
                "static-valid semantic rigid proxy immediately outside the canopy mask."
            ),
        },
        "aggregate": aggregate,
        "new_rigid_boundary_volume_causal_aggregate": aggregate_causal,
        "per_view": rows,
        "labels": {"old": args.old_label, "new": args.new_label},
        "figures": {
            "fixed_views": str(output / "fixed_views_ab.png"),
            "region_errors": str(output / "region_errors_ab.png"),
            "new_rigid_boundary_causal": str(output / "new_rigid_boundary_causal.png"),
        },
    }
    report = output / "fixed_ab_report.json"
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report), "aggregate": aggregate, "causal": aggregate_causal}, indent=2))


if __name__ == "__main__":
    main()
