#!/usr/bin/env python
"""Evaluate a canonical standard 3DGS without teacher-only interfaces."""

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
from outdoor.hybrid_gaussian_renderer import (  # noqa: E402
    VolumetricFoliageModel,
    render_hybrid,
)
from outdoor.lazy_scene import LazyScene  # noqa: E402
from outdoor.standard_3dgs import load_standard_3dgs_ply  # noqa: E402
from outdoor.task_fields import OutdoorTaskFieldLookup  # noqa: E402
from scene import GaussianModel  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    parser.set_defaults(data_device="cpu", white_background=True)
    parser.add_argument("--point-cloud", type=Path, required=True)
    parser.add_argument("--semantic-contract", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", default="408,409,410")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help=(
            "Save every standard-3DGS prediction but omit duplicate GT, "
            "error and alpha PNGs. Metrics are unchanged and auxiliary "
            "visualizations can still be produced by a targeted evaluation."
        ),
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help=(
            "Render predictions without decoding GT or semantic fields. "
            "Use evaluate_render_dir.py afterwards for the complete metric "
            "protocol."
        ),
    )
    parser.add_argument(
        "--resume-render",
        action="store_true",
        help=(
            "With --skip-metrics, reuse prediction PNGs already present in "
            "the output directory."
        ),
    )
    parser.add_argument("--view-cache-size", type=int, default=2)
    args = parser.parse_args()
    if args.skip_metrics and not args.render_only:
        parser.error("--skip-metrics requires --render-only")
    if args.resume_render and not args.skip_metrics:
        parser.error("--resume-render requires --skip-metrics")
    return args, model.extract(args)


def _empty_surface(sh_degree: int) -> GaussianModel:
    model = GaussianModel(sh_degree)
    model.active_sh_degree = sh_degree
    coefficients = (sh_degree + 1) ** 2
    model._xyz = torch.nn.Parameter(torch.empty(0, 3, device="cuda"))
    model._features_dc = torch.nn.Parameter(
        torch.empty(0, 1, 3, device="cuda")
    )
    model._features_rest = torch.nn.Parameter(
        torch.empty(0, coefficients - 1, 3, device="cuda")
    )
    model._scaling = torch.nn.Parameter(
        torch.empty(0, 2, device="cuda")
    )
    model._rotation = torch.nn.Parameter(
        torch.empty(0, 4, device="cuda")
    )
    model._opacity = torch.nn.Parameter(
        torch.empty(0, 1, device="cuda")
    )
    model._initialize_point_metadata(0, device="cuda")
    return model


def _region_metrics(prediction, target, weight):
    weight = weight.clamp(0, 1)
    mass = weight.sum().clamp_min(1)
    mae = (
        (prediction - target).abs().mean(0) * weight
    ).sum() / mass
    mse = (
        (prediction - target).square().mean(0) * weight
    ).sum() / mass
    return {
        "psnr": float(-10 * torch.log10(mse.clamp_min(1e-12))),
        "mae": float(mae),
    }


def main():
    args, dataset = _parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    Path(dataset.model_path).mkdir(parents=True, exist_ok=True)
    dummy = GaussianModel(dataset.sh_degree)
    scene = LazyScene(
        dataset,
        dummy,
        image_cache_size=args.view_cache_size,
    )
    views = scene.getTrainCameras()
    seed = load_standard_3dgs_ply(
        args.point_cloud, sh_degree=dataset.sh_degree
    )
    count = len(seed["xyz"])
    model = VolumetricFoliageModel(
        dataset.sh_degree, dynamic_rank=1
    ).cuda()
    model.restore(
        {
            "sh_degree": dataset.sh_degree,
            "dynamic_rank": 1,
            **seed,
            "deformation_basis": torch.zeros(count, 1, 3),
            "dynamic_feature_basis": torch.zeros(count, 1, 3),
            "dynamic_opacity_basis": torch.zeros(count, 1, 1),
        }
    )
    empty = _empty_surface(dataset.sh_degree)
    fields = None
    if not args.skip_metrics:
        fields = OutdoorTaskFieldLookup(
            Path(dataset.source_path),
            args.tree_mask_pickle,
            args.semantic_contract,
            max_cached_views=0,
        )
    if args.all:
        indices = list(range(len(views)))
    else:
        indices = [
            int(value)
            for value in args.indices.split(",")
            if value.strip()
        ]
    background = torch.ones(3, device="cuda")
    rows = []
    reused_renders = 0
    with torch.no_grad():
        for index in indices:
            view = views[index]
            prediction_path = output / f"{index:05d}_standard_3dgs.png"
            if (
                args.skip_metrics
                and args.resume_render
                and prediction_path.is_file()
            ):
                rows.append(
                    {
                        "index": index,
                        "image_name": str(view.image_name),
                    }
                )
                reused_renders += 1
                continue
            package = render_hybrid(
                view,
                empty,
                model,
                background=background,
                include_dynamic=False,
            )
            prediction = package.render.clamp(0, 1)
            row = {
                "index": index,
                "image_name": str(view.image_name),
            }
            if not args.skip_metrics:
                target = view.original_image.cuda()
                task = fields.fields(
                    view.image_name,
                    (view.image_height, view.image_width),
                    torch.device("cuda"),
                )
                mse = (prediction - target).square().mean()
                row.update(
                    {
                        "psnr": float(
                            -10 * torch.log10(mse.clamp_min(1e-12))
                        ),
                        "ssim": float(ssim(prediction, target)),
                        "mae": float((prediction - target).abs().mean()),
                        "rigid": _region_metrics(
                            prediction, target, task["p_rigid"]
                        ),
                        "canopy": _region_metrics(
                            prediction, target, task["p_canopy"]
                        ),
                    }
                )
            rows.append(row)
            images_to_save = [("standard_3dgs", prediction)]
            if not args.render_only:
                images_to_save.extend(
                    (
                        ("gt", target),
                        ("error_x4", (prediction - target).abs() * 4),
                        ("alpha", package.alpha.expand(3, -1, -1)),
                    )
                )
            for name, value in images_to_save:
                pixels = (
                    value.clamp(0, 1)
                    .mul(255)
                    .byte()
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
                Image.fromarray(pixels).save(
                    output / f"{index:05d}_{name}.png"
                )
            scene.release_images()
    result = {
        "protocol": (
            "standard_3dgs_direct_render_v1"
            if args.skip_metrics
            else "standard_3dgs_direct_evaluation_v1"
        ),
        "point_cloud": str(args.point_cloud.resolve()),
        "view_count": len(rows),
        "custom_teacher_interface_used": False,
    }
    if args.skip_metrics:
        result.update(
            {
                "rendered_view_indices": [row["index"] for row in rows],
                "reused_render_count": reused_renders,
                "metrics_computed": False,
            }
        )
        result_path = output / "render_manifest.json"
    else:
        result.update(
            {
                "aggregate": {
                    key: float(np.mean([row[key] for row in rows]))
                    for key in ("psnr", "ssim", "mae")
                },
                "per_view": rows,
            }
        )
        result_path = output / "metrics.json"
    result_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
