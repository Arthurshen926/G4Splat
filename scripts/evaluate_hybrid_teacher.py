#!/usr/bin/env python
"""Render and evaluate the authoritative native mixed Teacher directly."""

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
from outdoor.hybrid_teacher_api import load_hybrid_teacher  # noqa: E402
from outdoor.lazy_scene import LazyScene  # noqa: E402
from outdoor.task_fields import OutdoorTaskFieldLookup  # noqa: E402
from scene import GaussianModel  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402


def _region(prediction, target, weight):
    weight = weight.clamp(0, 1)
    mass = weight.sum().clamp_min(1)
    mae = ((prediction - target).abs().mean(0) * weight).sum() / mass
    mse = ((prediction - target).square().mean(0) * weight).sum() / mass
    return {
        "psnr": float(-10 * torch.log10(mse.clamp_min(1e-12))),
        "mae": float(mae),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    parser.set_defaults(data_device="cpu", white_background=True)
    parser.add_argument("--teacher-state", type=Path, required=True)
    parser.add_argument("--semantic-contract", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", default="408,409,410")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--view-cache-size", type=int, default=2)
    args = parser.parse_args()
    dataset = model.extract(args)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    Path(dataset.model_path).mkdir(parents=True, exist_ok=True)
    scene = LazyScene(
        dataset,
        GaussianModel(dataset.sh_degree),
        image_cache_size=args.view_cache_size,
    )
    views = scene.getTrainCameras()
    teacher = load_hybrid_teacher(
        args.teacher_state, sh_degree=dataset.sh_degree
    )
    fields = OutdoorTaskFieldLookup(
        Path(dataset.source_path),
        args.tree_mask_pickle,
        args.semantic_contract,
        max_cached_views=0,
    )
    indices = (
        list(range(len(views)))
        if args.all
        else [
            int(value)
            for value in args.indices.split(",")
            if value.strip()
        ]
    )
    rows = []
    with torch.no_grad():
        for index in indices:
            view = views[index]
            task = fields.fields(
                view.image_name,
                (view.image_height, view.image_width),
                torch.device("cuda"),
            )
            target = view.original_image.cuda(non_blocking=True)
            canonical = teacher.render(
                view, task=task, conditioned=False
            )
            conditioned = teacher.render(
                view, task=task, conditioned=True
            )
            row = {"index": index, "image_name": str(view.image_name)}
            for name, render in (
                ("canonical", canonical),
                ("conditioned", conditioned),
            ):
                prediction = render["rgb"]
                mse = (prediction - target).square().mean()
                row[name] = {
                    "psnr": float(
                        -10 * torch.log10(mse.clamp_min(1e-12))
                    ),
                    "ssim": float(ssim(prediction, target)),
                    "mae": float((prediction - target).abs().mean()),
                    "rigid": _region(
                        prediction, target, task["p_rigid"]
                    ),
                    "canopy": _region(
                        prediction, target, task["p_canopy"]
                    ),
                }
            rows.append(row)
            if not args.render_only or not args.all:
                images = {
                    "gt": target,
                    "canonical": canonical["rgb"],
                    "conditioned": conditioned["rgb"],
                    "error_x4": (
                        conditioned["rgb"] - target
                    ).abs()
                    * 4,
                    "surface_alpha": canonical[
                        "surface_alpha"
                    ].expand(3, -1, -1),
                    "volume_alpha": conditioned[
                        "volume_alpha"
                    ].expand(3, -1, -1),
                }
                for name, value in images.items():
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
    aggregate = {
        mode: {
            metric: float(
                np.mean([row[mode][metric] for row in rows])
            )
            for metric in ("psnr", "ssim", "mae")
        }
        for mode in ("canonical", "conditioned")
    }
    payload = {
        "protocol": "native-hybrid-teacher-direct-evaluation-v1",
        "teacher_state": str(args.teacher_state.resolve()),
        "view_count": len(rows),
        "aggregate": aggregate,
        "per_view": rows,
        "student_or_standard_renderer_used": False,
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
