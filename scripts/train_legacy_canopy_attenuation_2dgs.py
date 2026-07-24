#!/usr/bin/env python
"""Conservatively attenuate legacy 2D tree sheets before 3DGS replacement."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
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
from scene import GaussianModel, Scene  # noqa: E402
from scripts.train_standard_full_2dgs import (  # noqa: E402
    _file_digest,
    _fixed_camera_schedule,
)

PROTOCOL = "persistent_multiview_legacy_canopy_attenuation_v1"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--parent-ply", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--sky-model", type=Path, required=True)
    parser.add_argument("--color-correction", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path, required=True)
    parser.add_argument("--task-semantic-manifest", type=Path, required=True)
    parser.add_argument("--classification-views", type=int, default=64)
    parser.add_argument("--minimum-canopy-views", type=int, default=3)
    parser.add_argument("--canopy-to-rigid-ratio", type=float, default=2.0)
    parser.add_argument("--minimum-world-scale", type=float, default=0.02)
    parser.add_argument("--minimum-screen-radius", type=float, default=128.0)
    parser.add_argument("--gate-lr", type=float, default=0.02)
    parser.add_argument("--preservation-weight", type=float, default=0.02)
    parser.add_argument("--minimum-gate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--eval-indices", default="408,409,410")
    args = parser.parse_args()
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    if args.iterations <= 0 or args.classification_views < 3:
        parser.error("iterations and classification views must be positive")
    return args, dataset, pipe


def _project(points, view):
    rotation = torch.as_tensor(view.R, device=points.device, dtype=points.dtype)
    translation = torch.as_tensor(
        view.T, device=points.device, dtype=points.dtype
    )
    camera = points @ rotation + translation
    z = camera[:, 2]
    u = camera[:, 0] / z.clamp_min(1e-6) * view.focal_x + view.cx
    v = camera[:, 1] / z.clamp_min(1e-6) * view.focal_y + view.cy
    valid = (
        (z > view.znear)
        & (z < view.zfar)
        & (u >= 0)
        & (u < view.image_width)
        & (v >= 0)
        & (v < view.image_height)
    )
    return u, v, valid


@torch.no_grad()
def _classify_legacy_canopy(
    gaussians,
    views,
    task_fields,
    protected_count,
    *,
    view_count,
    minimum_views,
    canopy_to_rigid_ratio,
    minimum_world_scale,
    minimum_screen_radius,
    pipe,
    background,
):
    scored = []
    for index, view in enumerate(views):
        fields = task_fields.fields(
            view.image_name,
            (view.image_height, view.image_width),
            gaussians.get_xyz.device,
        )
        scored.append((float(fields["p_canopy"].mean().item()), index))
    selected = [index for _, index in sorted(scored, reverse=True)[:view_count]]
    canopy_hits = torch.zeros(protected_count, dtype=torch.int16, device="cuda")
    rigid_hits = torch.zeros_like(canopy_hits)
    points = gaussians.get_xyz[:protected_count]
    for index in tqdm(selected, desc="classify legacy canopy"):
        view = views[index]
        fields = task_fields.fields(
            view.image_name,
            (view.image_height, view.image_width),
            points.device,
        )
        u, v, valid = _project(points, view)
        rendered = render(
            view, gaussians, pipe, background, rgb_only=True
        )
        abnormal_visible_sheet = (
            rendered["radii"][:protected_count]
            >= float(minimum_screen_radius)
        )
        valid = valid & abnormal_visible_sheet
        rows = v.round().long().clamp(0, view.image_height - 1)
        cols = u.round().long().clamp(0, view.image_width - 1)
        canopy_hits += (
            valid & (fields["p_canopy"][rows, cols] > 0.5)
        ).to(torch.int16)
        rigid_hits += (
            valid & (fields["p_rigid"][rows, cols] > 0.5)
        ).to(torch.int16)
    scale = gaussians.get_scaling[:protected_count].amax(dim=-1)
    candidate = (
        (canopy_hits >= int(minimum_views))
        & (
            canopy_hits.float()
            >= float(canopy_to_rigid_ratio) * rigid_hits.float()
        )
        & (scale >= float(minimum_world_scale))
    )
    return torch.nonzero(candidate, as_tuple=False).flatten(), {
        "selected_camera_indices": selected,
        "candidate_count": int(candidate.sum().item()),
        "canopy_hits_max": int(canopy_hits.max().item()),
        "rigid_hits_max": int(rigid_hits.max().item()),
        "minimum_screen_radius": float(minimum_screen_radius),
    }


def _weighted_loss(prediction, target, weight):
    weight = weight[None].expand_as(prediction)
    return (
        torch.sqrt((prediction - target).square() + 1e-6) * weight
    ).sum() / weight.sum().clamp_min(1.0)


@torch.no_grad()
def _evaluate(
    views, indices, gaussians, pipe, sky, color, background, opacity_override
):
    rows = []
    for index in indices:
        view = views[index]
        package = render(
            view,
            gaussians,
            pipe,
            background,
            opacity_override=opacity_override,
        )
        prediction = apply_per_image_affine_color_correction(
            color,
            composite_white_background(
                package["render"], package["rend_alpha"], sky(view)
            ),
            view.image_name,
        ).clamp(0, 1)
        target = view.original_image.cuda()
        mse = (prediction - target).square().mean()
        rows.append(
            {
                "index": index,
                "image_name": view.image_name,
                "psnr": float((-10 * torch.log10(mse)).item()),
                "mae": float((prediction - target).abs().mean().item()),
            }
        )
    return rows


def main():
    args, dataset, pipe = _parse_args()
    output = Path(dataset.model_path).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.load_ply(str(args.parent_ply.resolve()))
    gaussians.set_mip_filter(False)
    views = scene.getTrainCameras()
    parent_result = json.loads(args.parent_result.read_text(encoding="utf-8"))
    protected_count = int(parent_result["protected_parent_gaussians"])
    task_fields = OutdoorTaskFieldLookup(
        Path(dataset.source_path).resolve(),
        args.tree_mask_pickle.resolve(),
        args.task_semantic_manifest.resolve(),
        max_cached_views=0,
    )
    candidates, classification = _classify_legacy_canopy(
        gaussians,
        views,
        task_fields,
        protected_count,
        view_count=args.classification_views,
        minimum_views=args.minimum_canopy_views,
        canopy_to_rigid_ratio=args.canopy_to_rigid_ratio,
        minimum_world_scale=args.minimum_world_scale,
        minimum_screen_radius=args.minimum_screen_radius,
        pipe=pipe,
        background=torch.ones(3, device="cuda"),
    )
    if not len(candidates):
        raise RuntimeError("Legacy canopy classifier selected no Gaussians")

    sky = CanonicalDirectionalSky.load(args.sky_model.resolve(), device="cuda")
    color = load_per_image_affine_color_correction(
        args.color_correction.resolve(), device="cuda"
    )
    for module in (sky, color):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    base_opacity = gaussians.get_opacity.detach()
    minimum_logit = float(
        torch.logit(torch.tensor(float(args.minimum_gate))).item()
    )
    gate_logits = torch.nn.Parameter(
        torch.full((len(candidates),), 6.9, device="cuda")
    )
    optimizer = torch.optim.Adam([gate_logits], lr=args.gate_lr)
    schedule, schedule_audit = _fixed_camera_schedule(
        views, iterations=args.iterations, seed=args.seed
    )
    background = torch.ones(3, device="cuda")

    def effective_opacity():
        multiplier = torch.ones(
            len(base_opacity), device="cuda", dtype=base_opacity.dtype
        )
        multiplier = multiplier.index_copy(
            0, candidates, gate_logits.sigmoid().to(base_opacity.dtype)
        )
        return base_opacity * multiplier[:, None]

    eval_indices = [int(value) for value in args.eval_indices.split(",")]
    before = _evaluate(
        views,
        eval_indices,
        gaussians,
        pipe,
        sky,
        color,
        background,
        base_opacity,
    )
    trace = output / "training_trace.jsonl"
    start = time.time()
    for step in tqdm(range(args.iterations), desc="legacy canopy attenuation"):
        view = views[int(schedule[step])]
        opacity = effective_opacity()
        package = render(
            view,
            gaussians,
            pipe,
            background,
            opacity_override=opacity,
        )
        prediction = apply_per_image_affine_color_correction(
            color,
            composite_white_background(
                package["render"], package["rend_alpha"], sky(view)
            ),
            view.image_name,
        )
        target = view.original_image.cuda()
        fields = task_fields.fields(
            view.image_name, prediction.shape[-2:], prediction.device
        )
        weight = fields["w_topology"] + 0.1 * fields["p_rigid"]
        gate = gate_logits.sigmoid()
        photo = _weighted_loss(prediction, target, weight)
        preservation = (1.0 - gate).mean()
        loss = photo + args.preservation_weight * preservation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            gate_logits.clamp_(min=minimum_logit, max=6.9)
        if step == 0 or (step + 1) % 100 == 0:
            with trace.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "iteration": step + 1,
                            "loss": float(loss.item()),
                            "mean_gate": float(gate.mean().item()),
                            "min_gate": float(gate.min().item()),
                        }
                    )
                    + "\n"
                )

    final_opacity = effective_opacity().detach()
    after = _evaluate(
        views,
        eval_indices,
        gaussians,
        pipe,
        sky,
        color,
        background,
        final_opacity,
    )
    paired = []
    for initial, final in zip(before, after):
        paired.append(
            {
                **final,
                "initial_psnr": initial["psnr"],
                "initial_mae": initial["mae"],
                "delta_psnr": final["psnr"] - initial["psnr"],
                "delta_mae": final["mae"] - initial["mae"],
            }
        )

    checkpoint = output / "point_cloud" / f"iteration_{68000 + args.iterations}"
    checkpoint.mkdir(parents=True)
    with torch.no_grad():
        gaussians._opacity.copy_(
            gaussians.inverse_opacity_activation(
                final_opacity.clamp(1e-6, 1 - 1e-6)
            )
        )
    gaussians.save_ply(str(checkpoint / "point_cloud.ply"))
    shutil.copy2(args.sky_model.resolve(), checkpoint / "sky_model.pth")
    shutil.copy2(
        args.color_correction.resolve(), checkpoint / "color_correction.pth"
    )
    manifest = {
        "protocol": PROTOCOL,
        "run_role": "legacy_canopy_replacement_preparation",
        "parent_ply": str(args.parent_ply.resolve()),
        "parent_ply_sha256": _file_digest(args.parent_ply.resolve()),
        "structural_geometry_changed": False,
        "new_gaussians_changed": False,
        "legacy_prefix_change": "opacity_attenuation_only_for_multiview_canopy",
        "classification": classification,
        "schedule": schedule_audit,
        "historical_full_real_rgb_initialization": False,
    }
    (output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "protocol": PROTOCOL,
        "validation_status": (
            "accepted_targeted"
            if all(row["delta_psnr"] >= 0.0 for row in paired)
            else "rejected_targeted_regression"
        ),
        "mainline_eligible": all(
            row["delta_psnr"] >= 0.0 for row in paired
        ),
        "candidate_count": len(candidates),
        "gate": {
            "mean": float(gate_logits.sigmoid().mean().item()),
            "minimum": float(gate_logits.sigmoid().min().item()),
            "attenuated_below_0_99": int(
                (gate_logits.sigmoid() < 0.99).sum().item()
            ),
        },
        "targeted_evaluation": paired,
        "point_cloud": str(checkpoint / "point_cloud.ply"),
        "elapsed_sec": time.time() - start,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
