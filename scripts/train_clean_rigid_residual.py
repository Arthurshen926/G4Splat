#!/usr/bin/env python
"""Refine only appended Chart/SfM 2D surfels behind a frozen clean base."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SURFEL_ROOT = REPO_ROOT / "2d-gaussian-splatting"
sys.path[:0] = [str(REPO_ROOT), str(SURFEL_ROOT)]

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from matcha.cambridge_training import (  # noqa: E402
    apply_per_image_affine_color_correction,
    load_per_image_affine_color_correction,
)
from outdoor.directional_sky import (  # noqa: E402
    CanonicalDirectionalSky,
    composite_white_background,
)
from outdoor.task_fields import OutdoorTaskFieldLookup  # noqa: E402
from outdoor.trainable_surfel_suffix import TrainableSurfelSuffix  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402


PROTOCOL = "clean_rigid_chart_residual_v1"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.set_defaults(
        iterations=1600,
        data_device="cpu",
        white_background=True,
    )
    parser.add_argument("--initial-ply", type=Path, required=True)
    parser.add_argument("--base-count", type=int, required=True)
    parser.add_argument("--sky-model", type=Path, required=True)
    parser.add_argument("--color-correction", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path, required=True)
    parser.add_argument("--task-semantic-manifest", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1600)
    parser.add_argument("--eval-indices", default="408,409,410")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--position-lr", type=float, default=2e-4)
    parser.add_argument("--feature-lr", type=float, default=1e-3)
    parser.add_argument("--opacity-lr", type=float, default=8e-3)
    parser.add_argument("--scale-lr", type=float, default=8e-4)
    parser.add_argument("--rotation-lr", type=float, default=3e-4)
    parser.add_argument("--opacity-sparsity-weight", type=float, default=2e-4)
    parser.add_argument("--position-prior-weight", type=float, default=2e-2)
    parser.add_argument("--scale-prior-weight", type=float, default=2e-3)
    parser.add_argument("--maximum-opacity", type=float, default=0.35)
    parser.add_argument("--maximum-scale", type=float, default=0.08)
    parser.add_argument("--maximum-position-delta", type=float, default=0.08)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.base_count < 0:
        parser.error("--base-count must be non-negative")
    if not dataset.source_path or not dataset.model_path:
        parser.error("-s/--source_path and -m/--model_path are required")
    if not dataset.white_background:
        parser.error("This refinement requires --white_background")
    # The suffix adapter exposes activated scales and cannot provide the
    # Python covariance helper. Native CUDA covariance remains exact.
    pipe.compute_cov3D_python = False
    return args, dataset, pipe


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _weighted_charbonnier(prediction, target, weight, epsilon=1e-3):
    weight = weight[None].to(prediction)
    denominator = weight.sum().clamp_min(1.0) * prediction.shape[0]
    return (
        torch.sqrt((prediction - target).square() + epsilon**2) * weight
    ).sum() / denominator


def _metrics(prediction, target, weight):
    weight = weight[None].to(prediction)
    denominator = weight.sum().clamp_min(1.0) * prediction.shape[0]
    mae = ((prediction - target).abs() * weight).sum() / denominator
    mse = ((prediction - target).square() * weight).sum() / denominator
    return {
        "psnr": float(-10.0 * torch.log10(mse.clamp_min(1e-12))),
        "mae": float(mae),
    }


def _save_rgb(path: Path, value: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (
        value.detach()
        .clamp(0, 1)
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(array).save(path)


@torch.no_grad()
def _evaluate(
    views,
    indices,
    model,
    pipe,
    sky,
    color,
    fields,
    output: Path,
):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    background = torch.ones(3, device="cuda")
    for index in indices:
        view = views[index]
        package = render(
            view, model, pipe, background, rgb_only=False
        )
        prediction = apply_per_image_affine_color_correction(
            color,
            composite_white_background(
                package["render"],
                package["rend_alpha"],
                sky(view),
            ),
            view.image_name,
        ).clamp(0, 1)
        target = view.original_image.cuda()
        task = fields.fields(
            view.image_name,
            (view.image_height, view.image_width),
            torch.device("cuda"),
        )
        rigid = (
            task["p_rigid"] * (1.0 - task["p_boundary_uncertain"])
        )
        row = {
            "index": int(index),
            "image_name": str(view.image_name),
            "full": _metrics(prediction, target, torch.ones_like(rigid)),
            "rigid": _metrics(prediction, target, rigid),
            "ssim": float(ssim(prediction, target)),
        }
        rows.append(row)
        stem = f"{index:05d}"
        _save_rgb(output / f"{stem}_render.png", prediction)
        _save_rgb(output / f"{stem}_ground_truth.png", target)
        _save_rgb(
            output / f"{stem}_error_x4.png",
            (prediction - target).abs() * 4,
        )
    return rows


def main():
    args, dataset, pipe = _parse_args()
    output = Path(dataset.model_path).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    initial_ply = args.initial_ply.resolve()
    if not initial_ply.is_file():
        raise FileNotFoundError(initial_ply)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    structural = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, structural, shuffle=False)
    structural.load_ply(str(initial_ply))
    if args.base_count >= len(structural.get_xyz):
        raise RuntimeError("The initial PLY contains no trainable suffix")
    residual = TrainableSurfelSuffix(structural, args.base_count)
    views = scene.getTrainCameras()
    sky = CanonicalDirectionalSky.load(args.sky_model.resolve(), device="cuda")
    color = load_per_image_affine_color_correction(
        args.color_correction.resolve(), device="cuda"
    )
    for module in (sky, color):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    fields = OutdoorTaskFieldLookup(
        Path(dataset.source_path),
        args.tree_mask_pickle.resolve(),
        args.task_semantic_manifest.resolve(),
        max_cached_views=0,
    )
    eval_indices = [
        int(value) for value in args.eval_indices.split(",") if value.strip()
    ]
    if any(index < 0 or index >= len(views) for index in eval_indices):
        raise IndexError("An evaluation index is outside the training cameras")
    before = _evaluate(
        views,
        eval_indices,
        residual,
        pipe,
        sky,
        color,
        fields,
        output / "visualization" / "before",
    )

    optimizer = torch.optim.Adam(
        residual.parameter_groups(
            position_lr=args.position_lr,
            feature_lr=args.feature_lr,
            opacity_lr=args.opacity_lr,
            scale_lr=args.scale_lr,
            rotation_lr=args.rotation_lr,
        ),
        eps=1e-15,
    )
    rng = np.random.default_rng(args.seed + 1)
    schedule = []
    while len(schedule) < args.iterations:
        schedule.extend(rng.permutation(len(views)).tolist())
    schedule = schedule[: args.iterations]
    trace = output / "training_trace.jsonl"
    background = torch.ones(3, device="cuda")
    started = time.time()
    progress = tqdm(schedule, desc="clean rigid residual")
    for step, camera_index in enumerate(progress, start=1):
        view = views[camera_index]
        task = fields.fields(
            view.image_name,
            (view.image_height, view.image_width),
            torch.device("cuda"),
        )
        rigid = task["p_rigid"] * (
            1.0 - task["p_boundary_uncertain"]
        )
        package = render(view, residual, pipe, background)
        prediction = apply_per_image_affine_color_correction(
            color,
            composite_white_background(
                package["render"], package["rend_alpha"], sky(view)
            ),
            view.image_name,
        )
        target = view.original_image.cuda()
        photo = _weighted_charbonnier(prediction, target, rigid)
        position_prior = (
            (residual._xyz - residual.initial_xyz).square().sum(-1).mean()
        )
        scale_growth = torch.relu(
            residual._scaling - residual.initial_scaling
        ).square().mean()
        opacity_sparsity = torch.sigmoid(residual._opacity).mean()
        loss = (
            photo
            + args.position_prior_weight * position_prior
            + args.scale_prior_weight * scale_growth
            + args.opacity_sparsity_weight * opacity_sparsity
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        residual.project(
            maximum_opacity=args.maximum_opacity,
            maximum_scale=args.maximum_scale,
            maximum_position_delta=args.maximum_position_delta,
        )
        if step == 1 or step % args.log_every == 0:
            row = {
                "iteration": step,
                "loss": float(loss),
                "photo": float(photo),
                "position_prior": float(position_prior),
                "scale_growth": float(scale_growth),
                "opacity_mean": float(torch.sigmoid(residual._opacity).mean()),
                "elapsed_sec": time.time() - started,
            }
            with trace.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            progress.set_postfix(
                loss=f"{row['loss']:.4f}",
                opacity=f"{row['opacity_mean']:.3f}",
            )

    residual.write_back()
    point_cloud = output / "point_cloud" / f"iteration_{args.iterations}"
    point_cloud.mkdir(parents=True, exist_ok=True)
    final_ply = point_cloud / "point_cloud.ply"
    structural.save_ply(str(final_ply))
    after = _evaluate(
        views,
        eval_indices,
        residual,
        pipe,
        sky,
        color,
        fields,
        output / "visualization" / "after",
    )
    paired = []
    for first, last in zip(before, after):
        paired.append(
            {
                **last,
                "delta_full_psnr": (
                    last["full"]["psnr"] - first["full"]["psnr"]
                ),
                "delta_rigid_psnr": (
                    last["rigid"]["psnr"] - first["rigid"]["psnr"]
                ),
                "delta_ssim": last["ssim"] - first["ssim"],
            }
        )
    result = {
        "protocol": PROTOCOL,
        "initial_ply": str(initial_ply),
        "initial_ply_sha256": _digest(initial_ply),
        "base_count": int(args.base_count),
        "residual_count": int(len(structural.get_xyz) - args.base_count),
        "iterations": int(args.iterations),
        "constraints": {
            "maximum_opacity": args.maximum_opacity,
            "maximum_scale": args.maximum_scale,
            "maximum_position_delta": args.maximum_position_delta,
            "diffuse_only": True,
        },
        "before": before,
        "targeted_evaluation": paired,
        "output_ply": str(final_ply),
        "output_ply_sha256": _digest(final_ply),
        "elapsed_sec": time.time() - started,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
