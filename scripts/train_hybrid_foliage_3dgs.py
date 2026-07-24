#!/usr/bin/env python
"""Lift persistent canopy residuals into a jointly sorted volumetric 3DGS branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
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
from outdoor.hybrid_gaussian_renderer import (  # noqa: E402
    VolumetricFoliageModel,
    render_hybrid,
)
from outdoor.task_fields import OutdoorTaskFieldLookup  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from scripts.train_standard_full_2dgs import (  # noqa: E402
    _file_digest,
    _fixed_camera_schedule,
    _input_audit,
)


PROTOCOL = "joint_2d_surfel_volumetric_foliage_3dgs_v1"


def _loss(prediction, target, weight, epsilon=1e-3):
    weight = weight[None] if weight.ndim == 2 else weight
    weight = weight.expand_as(prediction)
    return (
        torch.sqrt((prediction - target).square() + epsilon**2) * weight
    ).sum() / weight.sum().clamp_min(1.0)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    parser.set_defaults(
        iterations=800,
        data_device="cpu",
        white_background=True,
    )
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--structural-ply", type=Path, required=True)
    parser.add_argument("--residual-ply", type=Path, required=True)
    parser.add_argument("--residual-result", type=Path, required=True)
    parser.add_argument("--sky-model", type=Path, required=True)
    parser.add_argument("--color-correction", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path, required=True)
    parser.add_argument("--task-semantic-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--camera-schedule-seed", type=int, default=29)
    parser.add_argument("--normal-scale-ratio", type=float, default=0.65)
    parser.add_argument("--structural-thickness-ratio", type=float, default=0.005)
    parser.add_argument("--position-lr", type=float, default=2e-5)
    parser.add_argument("--feature-lr", type=float, default=5e-4)
    parser.add_argument("--opacity-lr", type=float, default=2e-3)
    parser.add_argument("--scale-lr", type=float, default=2e-4)
    parser.add_argument("--rotation-lr", type=float, default=1e-4)
    parser.add_argument("--opacity-ceiling", type=float, default=0.08)
    parser.add_argument("--max-displacement", type=float, default=0.35)
    parser.add_argument("--eval-indices", default="408,409,410")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    dataset = model.extract(args)
    if not dataset.source_path or not dataset.model_path:
        parser.error("--source_path and --model_path are required")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if not 0 < args.normal_scale_ratio <= 2:
        parser.error("--normal-scale-ratio must be in (0, 2]")
    return args, dataset


@torch.no_grad()
def _evaluate(
    views,
    indices,
    structural,
    foliage,
    sky,
    color,
    background,
    thickness,
):
    rows = []
    for index in indices:
        view = views[index]
        package = render_hybrid(
            view,
            structural,
            foliage,
            background=background,
            structural_thickness_ratio=thickness,
        )
        prediction = composite_white_background(
            package.render, package.alpha, sky(view)
        )
        prediction = apply_per_image_affine_color_correction(
            color, prediction, view.image_name
        ).clamp(0, 1)
        target = view.original_image.cuda()
        mse = (prediction - target).square().mean()
        rows.append(
            {
                "index": int(index),
                "image_name": str(view.image_name),
                "psnr": float((-10 * torch.log10(mse)).item()),
                "mae": float((prediction - target).abs().mean().item()),
            }
        )
    return rows


def main():
    args, dataset = _parse_args()
    output = Path(dataset.model_path).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Scene is used for the exact fixed cameras only.  Its temporary sparse
    # initialization is immediately replaced by the audited structural PLY.
    structural = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, structural, shuffle=False)
    structural.load_ply(str(args.structural_ply.resolve()))
    structural.set_mip_filter(False)
    views = scene.getTrainCameras()
    input_audit = _input_audit(Path(dataset.source_path).resolve(), views)

    residual = GaussianModel(dataset.sh_degree)
    residual.load_ply(str(args.residual_ply.resolve()))
    result = json.loads(args.residual_result.read_text(encoding="utf-8"))
    expected_foliage = int(result["class_counts"]["foliage"])
    foliage_mask = (
        residual.get_primitive_class == GaussianModel.PRIMITIVE_FOLIAGE
    )
    if int(foliage_mask.sum().item()) != expected_foliage:
        raise RuntimeError("Residual PLY foliage count disagrees with result.json")
    foliage = VolumetricFoliageModel(dataset.sh_degree).cuda()
    lifted = foliage.initialize_from_surfel_residual(
        residual,
        foliage_mask,
        normal_scale_ratio=args.normal_scale_ratio,
    )
    del residual
    torch.cuda.empty_cache()

    sky = CanonicalDirectionalSky.load(args.sky_model.resolve(), device="cuda")
    for parameter in sky.parameters():
        parameter.requires_grad_(False)
    color = load_per_image_affine_color_correction(
        args.color_correction.resolve(), device="cuda"
    )
    for parameter in color.parameters():
        parameter.requires_grad_(False)
    fields = OutdoorTaskFieldLookup(
        Path(dataset.source_path).resolve(),
        args.tree_mask_pickle.resolve(),
        args.task_semantic_manifest.resolve(),
        max_cached_views=0,
    )
    schedule, schedule_audit = _fixed_camera_schedule(
        views, iterations=args.iterations, seed=args.camera_schedule_seed
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [foliage.xyz], "lr": args.position_lr, "name": "xyz"},
            {
                "params": [foliage.features],
                "lr": args.feature_lr,
                "name": "features",
            },
            {
                "params": [foliage.opacity_logits],
                "lr": args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [foliage.log_scales],
                "lr": args.scale_lr,
                "name": "scale",
            },
            {
                "params": [foliage.quaternions],
                "lr": args.rotation_lr,
                "name": "rotation",
            },
        ],
        eps=1e-15,
    )
    background = torch.ones(3, device="cuda")
    eval_indices = [
        int(value) for value in args.eval_indices.split(",") if value.strip()
    ]
    if any(index < 0 or index >= len(views) for index in eval_indices):
        raise IndexError("Evaluation index outside the training camera set")
    initial_xyz = foliage.xyz.detach().clone()
    initial_scale_bounds = (
        foliage.log_scales.detach().amin(dim=0) - np.log(2.0),
        foliage.log_scales.detach().amax(dim=0) + np.log(2.0),
    )
    before = _evaluate(
        views,
        eval_indices,
        structural,
        foliage,
        sky,
        color,
        background,
        args.structural_thickness_ratio,
    )

    manifest = {
        "protocol": PROTOCOL,
        "run_role": "volumetric_foliage_candidate",
        "input": input_audit,
        "parents": {
            "structural_ply": str(args.structural_ply.resolve()),
            "structural_ply_sha256": _file_digest(args.structural_ply.resolve()),
            "persistent_residual_ply": str(args.residual_ply.resolve()),
            "persistent_residual_ply_sha256": _file_digest(
                args.residual_ply.resolve()
            ),
            "persistent_residual_result": str(args.residual_result.resolve()),
        },
        "active_components": {
            "structural_parameters_inherited_from_2d_surfels": True,
            "exact_2d_surfel_projection_preserved": False,
            "true_volumetric_foliage_3dgs": True,
            "single_global_covariance_projection_and_depth_sort": True,
            "fixed_order_branch_compositing": False,
            "canonical_directional_sky": True,
            "frozen_inherited_train_camera_affine": True,
            "historical_full_real_rgb_initialization": False,
        },
        "foliage": {
            "seed_count": lifted,
            "seed_provenance": (
                "68k append-only residual with at least three distinct real views"
            ),
            "scale_dimensions": 3,
            "normal_scale_ratio": args.normal_scale_ratio,
            "topology": "frozen_during_3d_covariance_lift",
        },
        "structural": {
            "parameters_trainable": False,
            "thickness_ratio_for_joint_3d_covariance": (
                args.structural_thickness_ratio
            ),
        },
        "schedule": schedule_audit,
    }
    (output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    trace = output / "training_trace.jsonl"
    start = time.time()
    progress = tqdm(range(args.iterations), desc="volumetric foliage 3DGS")
    for step in progress:
        view = views[int(schedule[step])]
        package = render_hybrid(
            view,
            structural,
            foliage,
            background=background,
            structural_thickness_ratio=args.structural_thickness_ratio,
        )
        prediction = composite_white_background(
            package.render, package.alpha, sky(view)
        )
        prediction = apply_per_image_affine_color_correction(
            color, prediction, view.image_name
        )
        target = view.original_image.cuda()
        task = fields.fields(
            view.image_name, prediction.shape[-2:], prediction.device
        )
        # Canopy drives learning; a small rigid-region term prevents large
        # volume tails from spilling onto the protected building.
        weight = task["w_topology"] + 0.05 * task["p_rigid"]
        photometric = _loss(prediction, target, weight)
        opacity_regularizer = foliage.opacities.mean() * 2e-4
        loss = photometric + opacity_regularizer
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            displacement = foliage.xyz - initial_xyz
            distance = displacement.norm(dim=-1, keepdim=True)
            foliage.xyz.copy_(
                initial_xyz
                + displacement
                * (args.max_displacement / distance.clamp_min(args.max_displacement))
            )
            foliage.log_scales.copy_(
                torch.maximum(
                    torch.minimum(
                        foliage.log_scales, initial_scale_bounds[1]
                    ),
                    initial_scale_bounds[0],
                )
            )
            foliage.opacity_logits.clamp_(
                max=float(torch.logit(torch.tensor(args.opacity_ceiling)))
            )
            foliage.quaternions.copy_(
                torch.nn.functional.normalize(foliage.quaternions, dim=-1)
            )
        if step == 0 or (step + 1) % args.log_every == 0:
            row = {
                "iteration": step + 1,
                "camera": view.image_name,
                "loss": float(loss.item()),
                "photometric": float(photometric.item()),
                "mean_opacity": float(foliage.opacities.mean().item()),
                "elapsed_sec": time.time() - start,
            }
            with trace.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            progress.set_postfix(loss=f"{loss.item():.5f}")

    after = _evaluate(
        views,
        eval_indices,
        structural,
        foliage,
        sky,
        color,
        background,
        args.structural_thickness_ratio,
    )
    state_path = output / "volumetric_foliage_state.pth"
    torch.save(
        {
            "protocol": PROTOCOL,
            "foliage": foliage.capture(),
            "optimizer": optimizer.state_dict(),
            "manifest_sha256": hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode()
            ).hexdigest(),
        },
        state_path,
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
    result_payload = {
        "protocol": PROTOCOL,
        "validation_scope": "within_hybrid_training_only",
        "mainline_eligible": False,
        "mainline_blocker": (
            "requires direct comparison against exact 2DGS parent because "
            "structural projection semantics changed"
        ),
        "iterations": args.iterations,
        "structural_gaussians": int(len(structural.get_xyz)),
        "volumetric_foliage_gaussians": int(len(foliage)),
        "targeted_evaluation": paired,
        "state": str(state_path),
        "elapsed_sec": time.time() - start,
    }
    (output / "result.json").write_text(
        json.dumps(result_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result_payload, indent=2))


if __name__ == "__main__":
    main()
