#!/usr/bin/env python
"""Evaluate UV baking and a clean structural PLY without loading task masks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SURFEL_ROOT = REPO_ROOT / "2d-gaussian-splatting"
sys.path[:0] = [str(REPO_ROOT), str(SURFEL_ROOT)]

from arguments import ModelParams  # noqa: E402
from matcha.cambridge_training import (  # noqa: E402
    apply_per_image_affine_color_correction,
    load_per_image_affine_color_correction,
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
from scene import GaussianModel, Scene  # noqa: E402
from scripts.train_layered_foliage_v6 import (  # noqa: E402
    _load,
    _surface_atlas_indices,
)
from utils.loss_utils import ssim  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    parser.set_defaults(data_device="cpu", white_background=True)
    parser.add_argument("--original-structural-ply", type=Path, required=True)
    parser.add_argument("--baked-structural-ply", type=Path, required=True)
    parser.add_argument("--layered-state", type=Path, required=True)
    parser.add_argument("--sky-model", type=Path, required=True)
    parser.add_argument("--color-correction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--indices", default="408,409,410")
    args = parser.parse_args()
    return args, model.extract(args)


def _metrics(prediction, target):
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
    return {
        "psnr": float(-10 * torch.log10(mse.clamp_min(1e-12))),
        "ssim": float(ssim(prediction, target)),
        "mae": float(error.abs().mean()),
        "high_frequency_energy_ratio": float(
            gradient / target_gradient.clamp_min(1e-8)
        ),
    }


def _mean(rows):
    if not rows:
        return {}
    return {
        name: float(np.mean([row[name] for row in rows]))
        for name in rows[0]
    }


def _save(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = (
        image.detach()
        .clamp(0, 1)
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(value).save(path)


@torch.no_grad()
def main() -> None:
    args, dataset = _parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = {
        int(value) for value in args.indices.split(",") if value.strip()
    }

    original = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, original, shuffle=False)
    views = scene.getTrainCameras()
    original.load_ply(str(args.original_structural_ply.resolve()))
    if args.stride > 0:
        indices = set(range(0, len(views), args.stride)) | selected
    else:
        indices = selected
    indices = sorted(index for index in indices if 0 <= index < len(views))
    state = _load(args.layered_state.resolve())
    if state.get("protocol") != "layered_dynamic_foliage_surfel_uv_replace_v7":
        raise RuntimeError("Layered state does not contain intrinsic UV gates")
    candidates = state["legacy_candidate_indices"].cuda().long()
    surface_atlas_indices = _surface_atlas_indices(original, candidates)
    surface_gate_atlas = state["surfel_uv_gate_atlas"].cuda()
    background = torch.ones(3, device="cuda")
    sky = CanonicalDirectionalSky.load(
        args.sky_model.resolve(), device="cuda"
    )
    color = load_per_image_affine_color_correction(
        args.color_correction.resolve(), device="cuda"
    )
    empty = VolumetricFoliageModel(dataset.sh_degree).cuda()
    rows: dict[int, dict] = defaultdict(dict)
    selected_uv: dict[int, torch.Tensor] = {}

    def corrected(package, view):
        return apply_per_image_affine_color_correction(
            color,
            composite_white_background(
                package.render, package.alpha, sky(view)
            ),
            view.image_name,
        ).clamp(0, 1)

    for index in tqdm(indices, desc="original structural branches"):
        view = views[index]
        target = view.original_image.cuda()
        torch.cuda.empty_cache()
        parent = corrected(
            render_hybrid(
                view, original, empty, background=background
            ),
            view,
        )
        rows[index]["original_parent"] = _metrics(parent, target)
        if index in selected:
            _save(
                output.parent / "visualization" / f"{index:05d}_gt.png",
                target,
            )
            _save(
                output.parent
                / "visualization"
                / f"{index:05d}_original_parent.png",
                parent,
            )
        del parent
        torch.cuda.empty_cache()
        uv_gated = corrected(
            render_hybrid(
                view,
                original,
                empty,
                background=background,
                surface_gate_indices=surface_atlas_indices,
                surface_gate_atlas=surface_gate_atlas,
            ),
            view,
        )
        rows[index]["original_uv_gated"] = _metrics(uv_gated, target)
        if index in selected:
            selected_uv[index] = uv_gated.cpu()
            _save(
                output.parent
                / "visualization"
                / f"{index:05d}_original_uv_gated.png",
                uv_gated,
            )
        del uv_gated, target

    scene.gaussians = None
    del original, empty, surface_atlas_indices, surface_gate_atlas
    torch.cuda.empty_cache()
    baked = GaussianModel(dataset.sh_degree)
    baked.load_ply(str(args.baked_structural_ply.resolve()))
    empty = VolumetricFoliageModel(dataset.sh_degree).cuda()
    foliage = VolumetricFoliageModel(
        dataset.sh_degree,
        dynamic_rank=int(state["foliage"].get("dynamic_rank", 4)),
    ).cuda()
    foliage.restore(state["foliage"])

    for index in tqdm(indices, desc="baked clean branches"):
        view = views[index]
        target = view.original_image.cuda()
        torch.cuda.empty_cache()
        surface = corrected(
            render_hybrid(view, baked, empty, background=background),
            view,
        )
        rows[index]["baked_surface"] = _metrics(surface, target)
        if index in selected:
            reference = selected_uv[index].to(surface.device)
            bake_error = (surface - reference).abs()
            rows[index]["bake_approximation"] = {
                "mae": float(bake_error.mean()),
                "max_abs": float(bake_error.max()),
            }
            _save(
                output.parent
                / "visualization"
                / f"{index:05d}_baked_surface.png",
                surface,
            )
            _save(
                output.parent
                / "visualization"
                / f"{index:05d}_bake_error_x4.png",
                bake_error * 4,
            )
        del surface
        torch.cuda.empty_cache()
        canonical = corrected(
            render_hybrid(
                view,
                baked,
                foliage,
                background=background,
                include_dynamic=False,
            ),
            view,
        )
        rows[index]["baked_canonical"] = _metrics(canonical, target)
        if index in selected:
            _save(
                output.parent
                / "visualization"
                / f"{index:05d}_baked_canonical.png",
                canonical,
            )
            _save(
                output.parent
                / "visualization"
                / f"{index:05d}_baked_canonical_error_x4.png",
                (canonical - target).abs() * 4,
            )
        del canonical, target

    modes = (
        "original_parent",
        "original_uv_gated",
        "baked_surface",
        "baked_canonical",
    )
    mean = {
        mode: _mean([rows[index][mode] for index in indices])
        for mode in modes
    }
    per_sequence = {}
    for sequence in sorted(
        {sequence_id(views[index].image_name) for index in indices}
    ):
        group = [
            index
            for index in indices
            if sequence_id(views[index].image_name) == sequence
        ]
        per_sequence[sequence] = {
            mode: _mean([rows[index][mode] for index in group])
            for mode in modes
        }
    result = {
        "protocol": "baked_clean_hybrid_evaluation_v1",
        "view_count": len(indices),
        "stride": args.stride,
        "original_structural_ply": str(
            args.original_structural_ply.resolve()
        ),
        "baked_structural_ply": str(
            args.baked_structural_ply.resolve()
        ),
        "mean": mean,
        "per_sequence": per_sequence,
        "per_view": [
            {
                "index": index,
                "image_name": str(views[index].image_name),
                "sequence": sequence_id(views[index].image_name),
                **rows[index],
            }
            for index in indices
        ],
    }
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"view_count": len(indices), "mean": mean}, indent=2))


if __name__ == "__main__":
    main()
