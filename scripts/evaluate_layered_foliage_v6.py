#!/usr/bin/env python
"""Streaming all-view evaluation for layered canonical and conditioned foliage."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SURFEL_ROOT = REPO_ROOT / "2d-gaussian-splatting"
sys.path[:0] = [str(REPO_ROOT), str(SURFEL_ROOT)]

from arguments import ModelParams, PipelineParams  # noqa: E402
from matcha.cambridge_training import (  # noqa: E402
    apply_per_image_affine_color_correction,
    load_per_image_affine_color_correction,
)
from outdoor.appearance_uncertainty import (  # noqa: E402
    OutdoorAppearanceUncertainty,
)
from outdoor.directional_sky import (  # noqa: E402
    CanonicalDirectionalSky,
    composite_white_background,
)
from outdoor.foliage_view_graph import sequence_id  # noqa: E402
from outdoor.hybrid_gaussian_renderer import (  # noqa: E402
    VolumetricFoliageModel,
    render_hybrid,
)
from outdoor.task_fields import OutdoorTaskFieldLookup  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from scripts.train_layered_foliage_v6 import (  # noqa: E402
    _gate_vector,
    _load,
    _surface_atlas_indices,
)
from utils.loss_utils import ssim  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.set_defaults(data_device="cpu", white_background=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--structural-ply", type=Path, required=True)
    parser.add_argument("--sky-model", type=Path, required=True)
    parser.add_argument("--color-correction", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path, required=True)
    parser.add_argument("--task-semantic-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    return args, model.extract(args), pipeline.extract(args)


def _region(prediction, target, mask):
    denominator = mask.sum()
    if float(denominator) < 1.0:
        return None, None
    mse = (
        (prediction - target).square().mean(0) * mask
    ).sum() / denominator.clamp_min(1)
    mae = (
        (prediction - target).abs().mean(0) * mask
    ).sum() / denominator.clamp_min(1)
    return float(-10 * torch.log10(mse.clamp_min(1e-12))), float(mae)


def _metrics(prediction, target, task):
    error = prediction - target
    mse = error.square().mean()
    gray = prediction.mean(0)
    target_gray = target.mean(0)
    gradient = 0.5 * (
        (gray[:, 1:] - gray[:, :-1]).abs().mean()
        + (gray[1:] - gray[:-1]).abs().mean()
    )
    target_gradient = 0.5 * (
        (target_gray[:, 1:] - target_gray[:, :-1]).abs().mean()
        + (target_gray[1:] - target_gray[:-1]).abs().mean()
    )
    canopy_psnr, canopy_mae = _region(
        prediction, target, task["p_canopy"]
    )
    rigid_psnr, rigid_mae = _region(
        prediction, target, task["p_rigid"]
    )
    return {
        "psnr": float(-10 * torch.log10(mse.clamp_min(1e-12))),
        "ssim": float(ssim(prediction, target)),
        "mae": float(error.abs().mean()),
        "canopy_psnr": canopy_psnr,
        "canopy_mae": canopy_mae,
        "rigid_psnr": rigid_psnr,
        "rigid_mae": rigid_mae,
        "high_frequency_energy_ratio": float(
            gradient / target_gradient.clamp_min(1e-8)
        ),
    }


def _mean(rows):
    output = {}
    for key in rows[0]:
        if key in {"index", "image_name", "sequence"}:
            continue
        values = [row[key] for row in rows if row[key] is not None]
        output[key] = float(np.mean(values)) if values else None
    return output


@torch.no_grad()
def main():
    args, dataset, _ = _parse_args()
    Path(dataset.model_path).mkdir(parents=True, exist_ok=True)
    structural = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, structural, shuffle=False)
    structural.load_ply(str(args.structural_ply.resolve()))
    views = scene.getTrainCameras()
    state = _load(args.state.resolve())
    rank = int(state["foliage"].get("dynamic_rank", 4))
    foliage = VolumetricFoliageModel(
        dataset.sh_degree, dynamic_rank=rank
    ).cuda()
    foliage.restore(state["foliage"])
    appearance_capture = state["appearance"]
    appearance = OutdoorAppearanceUncertainty(
        [view.image_name for view in views],
        rank=int(appearance_capture["rank"]),
        sky_degree=int(appearance_capture["sky_degree"]),
        maximum_rgb_residual=float(
            appearance_capture["maximum_rgb_residual"]
        ),
        spatial_grid_size=int(
            appearance_capture.get("spatial_grid_size", 24)
        ),
        temporal_code_norm=float(
            appearance_capture.get("temporal_code_norm", 0.25)
        ),
        device="cuda",
    )
    appearance.restore(appearance_capture)
    sky = CanonicalDirectionalSky.load(args.sky_model.resolve(), device="cuda")
    color = load_per_image_affine_color_correction(
        args.color_correction.resolve(), device="cuda"
    )
    fields = OutdoorTaskFieldLookup(
        Path(dataset.source_path),
        args.tree_mask_pickle,
        args.task_semantic_manifest,
        max_cached_views=0,
    )
    background = torch.ones(3, device="cuda")
    empty = VolumetricFoliageModel(
        dataset.sh_degree, dynamic_rank=rank
    ).cuda()
    candidates = state["legacy_candidate_indices"].cuda().long()
    if "surfel_uv_gate_atlas" in state:
        final_gate = torch.ones_like(
            structural.get_opacity.reshape(-1)
        )
        surface_atlas_indices = _surface_atlas_indices(
            structural, candidates
        )
        surface_gate_atlas = state["surfel_uv_gate_atlas"].cuda()
    else:
        final_gate = _gate_vector(
            structural,
            candidates,
            state["legacy_candidate_gates"].cuda(),
        )
        surface_atlas_indices = None
        surface_gate_atlas = None
    rows = []
    for index in tqdm(
        range(0, len(views), max(1, args.stride)),
        desc="all-view layered evaluation",
    ):
        view = views[index]
        task = fields.fields(
            view.image_name,
            (view.image_height, view.image_width),
            background.device,
        )
        task["image_name"] = str(view.image_name)
        target = view.original_image.cuda()

        def rendered(
            model,
            gate,
            include_dynamic,
            code=None,
            *,
            use_uv_gate=False,
        ):
            package = render_hybrid(
                view,
                structural,
                model,
                background=background,
                surface_gate=gate,
                surface_gate_indices=(
                    surface_atlas_indices if use_uv_gate else None
                ),
                surface_gate_atlas=(
                    surface_gate_atlas if use_uv_gate else None
                ),
                temporal_code=code,
                include_dynamic=include_dynamic,
            )
            return apply_per_image_affine_color_correction(
                color,
                composite_white_background(
                    package.render, package.alpha, sky(view)
                ),
                view.image_name,
            ).clamp(0, 1)

        parent = rendered(
            empty,
            torch.ones_like(structural.get_opacity.reshape(-1)),
            False,
        )
        canonical = rendered(
            foliage, final_gate, False, use_uv_gate=True
        )
        conditioned_base = rendered(
            foliage,
            final_gate,
            True,
            appearance.temporal_code(view.image_name),
            use_uv_gate=True,
        )
        conditioned = appearance(
            conditioned_base, view, task
        ).clamp(0, 1)
        row = {
            "index": int(index),
            "image_name": str(view.image_name),
            "sequence": sequence_id(view.image_name),
            "parent": _metrics(parent, target, task),
            "canonical": _metrics(canonical, target, task),
            "conditioned": _metrics(conditioned, target, task),
        }
        rows.append(row)
        del parent, canonical, conditioned_base, conditioned, target, task
        if (index + 1) % 25 == 0:
            torch.cuda.empty_cache()
    flattened = {}
    for mode in ("parent", "canonical", "conditioned"):
        flattened[mode] = _mean([row[mode] for row in rows])
    deltas = {
        mode: {
            key: flattened[mode][key] - flattened["parent"][key]
            for key in flattened["parent"]
        }
        for mode in ("canonical", "conditioned")
    }
    per_sequence = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sequence"]].append(row)
    for sequence, group in grouped.items():
        per_sequence[sequence] = {
            mode: _mean([row[mode] for row in group])
            for mode in ("parent", "canonical", "conditioned")
        }
    worst = sorted(
        (
            {
                "index": row["index"],
                "image_name": row["image_name"],
                "canonical_psnr_delta": (
                    row["canonical"]["psnr"] - row["parent"]["psnr"]
                ),
                "conditioned_psnr_delta": (
                    row["conditioned"]["psnr"] - row["parent"]["psnr"]
                ),
            }
            for row in rows
        ),
        key=lambda item: item["canonical_psnr_delta"],
    )
    result = {
        "protocol": "layered_foliage_all_view_evaluation_v1",
        "view_count": len(rows),
        "stride": args.stride,
        "mean": flattened,
        "delta": deltas,
        "per_sequence": per_sequence,
        "worst_five_percent": worst[: max(1, int(0.05 * len(worst)))],
        "per_view": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: result[key] for key in ("view_count", "mean", "delta")}, indent=2))


if __name__ == "__main__":
    main()
