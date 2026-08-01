#!/usr/bin/env python
"""Causal branch isolation for canonical/dynamic foliage ownership.

This is deliberately a diagnostic, not an evaluation protocol.  It loads one
native mixed Teacher checkpoint once, renders the same calibrated database
views with selected volume roles optically disabled, and reports identical
tree-region measurements for every counterfactual.  No checkpoint is changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SURFEL_ROOT = REPO_ROOT / "2d-gaussian-splatting"
sys.path[:0] = [str(REPO_ROOT), str(SURFEL_ROOT)]

from arguments import ModelParams  # noqa: E402
from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402
from outdoor.hybrid_teacher_api import load_hybrid_teacher  # noqa: E402
from outdoor.lazy_scene import LazyScene  # noqa: E402
from scene import GaussianModel  # noqa: E402
from scripts.evaluate_hybrid_teacher import (  # noqa: E402
    _high_frequency_metrics,
    _historical_uint8_raster,
    _protocol_metrics,
)


def _parse_indices(value: str, count: int) -> list[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise ValueError("At least one diagnostic view index is required")
    if min(indices) < 0 or max(indices) >= count:
        raise IndexError(
            f"Diagnostic indices {indices} exceed a {count}-view scene"
        )
    return indices


def _save_image(path: Path, value: torch.Tensor) -> None:
    pixels = (
        torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)
        .clamp(0, 1)
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(pixels).save(path)


def _branch_disable_mask(foliage, branch: str) -> torch.Tensor:
    dynamic = foliage.dynamic_leaf_mask
    canonical = foliage.canonical_crown_mask
    base = branch.split("_plus_")[0]
    if base in {"baseline", "mixed_dynamic"}:
        return torch.zeros_like(dynamic)
    if base == "canonical_only":
        return dynamic
    if base == "dynamic_only":
        return canonical
    if base == "exact_dynamic_only":
        return canonical | (dynamic & (foliage.initialization_source != 4))
    if base == "nonexact_dynamic_only":
        return canonical | (dynamic & (foliage.initialization_source == 4))
    if base == "skeleton_only":
        return canonical | dynamic
    raise ValueError(f"Unknown ownership counterfactual {branch!r}")


def _dynamic_logit_offset(branch: str) -> float:
    if "_plus_" not in branch:
        return 0.0
    encoded = branch.rsplit("_plus_", 1)[1]
    return float(encoded.replace("p", "."))


def _mean_rows(rows: list[dict], branch: str, key: str) -> float:
    return float(np.mean([row[branch][key] for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    parser.set_defaults(data_device="cpu", white_background=True)
    parser.add_argument("--teacher-state", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", default="408,409,410")
    parser.add_argument(
        "--opacity-sweep",
        action="store_true",
        help=(
            "Also render continuous dynamic-opacity logit offsets. This is "
            "an oracle sensitivity test, never a checkpoint mutation."
        ),
    )
    args = parser.parse_args()
    dataset = model.extract(args)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    teacher = load_hybrid_teacher(
        args.teacher_state.expanduser().resolve(),
        sh_degree=dataset.sh_degree,
    )
    scene = LazyScene(
        dataset,
        GaussianModel(dataset.sh_degree),
        image_cache_size=1,
    )
    views = scene.getTrainCameras()
    indices = _parse_indices(args.indices, len(views))
    masks = CambridgeMaskLookup(
        Path(dataset.source_path),
        args.tree_mask_pickle.expanduser().resolve(),
        mask_indices=[0, 1, 2, 3],
    )

    base_branches = (
        "baseline",
        "canonical_only",
        "dynamic_only",
        "exact_dynamic_only",
        "nonexact_dynamic_only",
        "skeleton_only",
    )
    sweep_branches = (
        "mixed_dynamic_plus_0p5",
        "mixed_dynamic_plus_1p0",
        "mixed_dynamic_plus_1p5",
        "mixed_dynamic_plus_2p0",
        "mixed_dynamic_plus_2p5",
        "mixed_dynamic_plus_3p0",
        "dynamic_only_plus_1p0",
        "dynamic_only_plus_2p0",
        "dynamic_only_plus_3p0",
        "exact_dynamic_only_plus_2p0",
        "exact_dynamic_only_plus_3p0",
    )
    branches = base_branches + (sweep_branches if args.opacity_sweep else ())
    original_opacity = teacher.foliage.opacity_logits.detach().clone()
    rows: list[dict] = []
    with torch.no_grad():
        for index in indices:
            view = views[index]
            keep = masks.get_index_masks(
                view.image_name,
                (0, 1, 2, 3),
                (view.image_height, view.image_width),
                torch.device("cuda"),
            )
            object_keep, sky_keep, distortion_keep, tree_keep = keep
            static_valid = object_keep & sky_keep & distortion_keep
            tree_static = static_valid & ~tree_keep
            target = view.original_image.cuda(non_blocking=True)
            historical_target = _historical_uint8_raster(target)
            row: dict[str, object] = {
                "index": int(index),
                "image_name": str(view.image_name),
                "tree_static_pixels": int(tree_static.sum()),
            }
            _save_image(output / f"{index:05d}_gt.png", target)
            for branch in branches:
                teacher.foliage.opacity_logits.copy_(original_opacity)
                disabled = _branch_disable_mask(teacher.foliage, branch)
                teacher.foliage.opacity_logits[disabled] = -30.0
                logit_offset = _dynamic_logit_offset(branch)
                if logit_offset:
                    teacher.foliage.opacity_logits[
                        teacher.foliage.dynamic_leaf_mask
                    ] += logit_offset
                render = teacher.render(view, task=None, conditioned=True)
                prediction = render["rgb"]
                historical = _historical_uint8_raster(prediction)
                tree_metric = _protocol_metrics(
                    historical, historical_target, tree_static
                )
                static_metric = _protocol_metrics(
                    historical, historical_target, static_valid
                )
                high_frequency = _high_frequency_metrics(
                    prediction, target, tree_static
                )
                alpha = render["volume_alpha"][0, tree_static]
                luma_weight = prediction.new_tensor([0.299, 0.587, 0.114])
                prediction_luma = (
                    prediction[:, tree_static].T * luma_weight
                ).sum(1)
                target_luma = (target[:, tree_static].T * luma_weight).sum(1)
                row[branch] = {
                    "tree_psnr": float(tree_metric["psnr"]),
                    "tree_mae": float(tree_metric["mae"]),
                    "static_psnr": float(static_metric["psnr"]),
                    "tree_gradient_cosine": float(
                        high_frequency["gradient_cosine"]
                    ),
                    "tree_edge_energy_ratio": float(
                        high_frequency["edge_energy_ratio"]
                    ),
                    "tree_prediction_luma": float(prediction_luma.mean()),
                    "tree_target_luma": float(target_luma.mean()),
                    "tree_luma_bias": float(
                        prediction_luma.mean() - target_luma.mean()
                    ),
                    "tree_volume_alpha_mean": float(alpha.mean()),
                    "tree_volume_alpha_median": float(alpha.median()),
                    "tree_volume_alpha_above_0_8": float(
                        (alpha > 0.8).float().mean()
                    ),
                    "disabled_primitive_count": int(disabled.sum()),
                    "dynamic_opacity_logit_offset": logit_offset,
                }
                _save_image(
                    output / f"{index:05d}_{branch}.png", prediction
                )
                _save_image(
                    output / f"{index:05d}_{branch}_volume_alpha.png",
                    render["volume_alpha"].expand(3, -1, -1),
                )
            rows.append(row)
            scene.release_images()
            masks._resized_mask_cache.clear()
    teacher.foliage.opacity_logits.copy_(original_opacity)

    aggregate = {
        branch: {
            key: _mean_rows(rows, branch, key)
            for key in (
                "tree_psnr",
                "tree_mae",
                "static_psnr",
                "tree_gradient_cosine",
                "tree_edge_energy_ratio",
                "tree_prediction_luma",
                "tree_target_luma",
                "tree_luma_bias",
                "tree_volume_alpha_mean",
                "tree_volume_alpha_median",
                "tree_volume_alpha_above_0_8",
            )
        }
        for branch in branches
    }
    payload = {
        "protocol": "hybrid-volume-ownership-counterfactual-v1",
        "diagnostic_only": True,
        "teacher_state": str(args.teacher_state.expanduser().resolve()),
        "checkpoint_iteration": int(teacher.state.get("iteration", 0)),
        "branch_semantics": {
            "baseline": "native conditioned mixed render",
            "canonical_only": "dynamic leaves disabled; skeleton and canonical crown retained",
            "dynamic_only": "canonical crown disabled; skeleton and every visible dynamic leaf retained",
            "exact_dynamic_only": "canonical crown and non-source4 dynamic leaves disabled",
            "nonexact_dynamic_only": "canonical crown and source4 exact-owner leaves disabled",
            "skeleton_only": "canonical crown and every dynamic leaf disabled",
            "*_plus_N": (
                "same isolation with N added to visible dynamic opacity "
                "log-odds; diagnostic sensitivity only"
            ),
        },
        "population": {
            "total": len(teacher.foliage),
            "static_skeleton": int(teacher.foliage.static_skeleton_mask.sum()),
            "canonical_crown": int(teacher.foliage.canonical_crown_mask.sum()),
            "dynamic_leaf": int(teacher.foliage.dynamic_leaf_mask.sum()),
            "exact_dynamic_source4": int(
                (
                    teacher.foliage.dynamic_leaf_mask
                    & (teacher.foliage.initialization_source == 4)
                ).sum()
            ),
            "grouped_dynamic": int(
                (
                    teacher.foliage.dynamic_leaf_mask
                    & (teacher.foliage.replacement_group >= 0)
                ).sum()
            ),
        },
        "aggregate": aggregate,
        "per_view": rows,
    }
    (output / "ownership_counterfactual.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
