#!/usr/bin/env python
"""Train native mixed 2D-surface/3D-volume foliage replacement."""

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
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SURFEL_ROOT = REPO_ROOT / "2d-gaussian-splatting"
sys.path[:0] = [str(REPO_ROOT), str(SURFEL_ROOT)]

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render as render_2dgs  # noqa: E402
from matcha.cambridge_training import (  # noqa: E402
    apply_per_image_affine_color_correction,
    load_per_image_affine_color_correction,
)
from outdoor.directional_sky import (  # noqa: E402
    CanonicalDirectionalSky,
    composite_white_background,
)
from outdoor.appearance_uncertainty import (  # noqa: E402
    OutdoorAppearanceUncertainty,
)
from outdoor.hybrid_gaussian_renderer import (  # noqa: E402
    VolumetricFoliageModel,
    render_hybrid,
)
from outdoor.foliage_responsibility import (  # noqa: E402
    LegacyResponsibilityAudit,
    RESPONSIBILITY_VERSION,
    audit_legacy_surface_responsibility,
)
from outdoor.task_fields import OutdoorTaskFieldLookup  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from scripts.train_standard_full_2dgs import (  # noqa: E402
    _file_digest,
    _fixed_camera_schedule,
    _input_audit,
)
from utils.loss_utils import ssim  # noqa: E402


PROTOCOL = "native_mixed_replace_and_retire_foliage_v2"


def _loss(prediction, target, weight, epsilon=1e-3):
    weight = weight[None] if weight.ndim == 2 else weight
    weight = weight.expand_as(prediction)
    return (
        torch.sqrt((prediction - target).square() + epsilon**2) * weight
    ).sum() / weight.sum().clamp_min(1.0)


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


def _save_scalar(path: Path, value: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    scalar = value.detach().squeeze().float().cpu()
    finite = torch.isfinite(scalar)
    if bool(finite.any()):
        low = torch.quantile(scalar[finite], 0.02)
        high = torch.quantile(scalar[finite], 0.98)
        scalar = ((scalar - low) / (high - low).clamp_min(1e-8)).clamp(0, 1)
    else:
        scalar = torch.zeros_like(scalar)
    Image.fromarray(scalar.mul(255).byte().numpy()).save(path)


def _region_metrics(prediction, target, mask):
    mask = mask[None].expand_as(prediction)
    denominator = mask.sum().clamp_min(1.0)
    mse = ((prediction - target).square() * mask).sum() / denominator
    mae = ((prediction - target).abs() * mask).sum() / denominator
    return {
        "psnr": float((-10 * torch.log10(mse.clamp_min(1e-12))).item()),
        "mae": float(mae.item()),
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.set_defaults(
        iterations=800,
        data_device="cpu",
        white_background=True,
    )
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--structural-ply", type=Path, required=True)
    parser.add_argument("--foliage-seed-state", type=Path, required=True)
    parser.add_argument("--replacement-audit", type=Path)
    parser.add_argument("--audit-views", type=int, default=24)
    parser.add_argument("--minimum-support-views", type=int, default=3)
    parser.add_argument("--minimum-support-sequences", type=int, default=2)
    parser.add_argument(
        "--maximum-rigid-responsibility",
        type=float,
        default=0.25,
        help=(
            "Maximum aggregate and per-view rigid responsibility for any "
            "legacy surfel eligible for replacement"
        ),
    )
    parser.add_argument(
        "--audit-mandatory-indices",
        default="408,409,410",
        help="Fixed difficult views always included in replacement responsibility",
    )
    parser.add_argument("--sky-model", type=Path, required=True)
    parser.add_argument("--color-correction", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path, required=True)
    parser.add_argument("--task-semantic-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--camera-schedule-seed", type=int, default=29)
    parser.add_argument("--position-lr", type=float, default=2e-5)
    parser.add_argument("--feature-lr", type=float, default=5e-4)
    parser.add_argument("--opacity-lr", type=float, default=2e-3)
    parser.add_argument("--scale-lr", type=float, default=2e-4)
    parser.add_argument("--rotation-lr", type=float, default=1e-4)
    parser.add_argument("--gate-lr", type=float, default=2e-3)
    parser.add_argument("--minimum-gate", type=float, default=0.0)
    parser.add_argument("--counterfactual-gate", type=float, default=0.05)
    parser.add_argument("--volume-pretrain-fraction", type=float, default=0.40)
    parser.add_argument("--joint-replacement-fraction", type=float, default=0.45)
    parser.add_argument("--counterfactual-weight", type=float, default=0.5)
    parser.add_argument("--silhouette-weight", type=float, default=0.05)
    parser.add_argument("--rigid-spill-weight", type=float, default=0.15)
    parser.add_argument("--rigid-alpha-spill-weight", type=float, default=0.20)
    parser.add_argument("--normal-rigid-weight", type=float, default=0.15)
    parser.add_argument("--canopy-gradient-weight", type=float, default=0.08)
    parser.add_argument("--depth-prior-weight", type=float, default=0.01)
    parser.add_argument("--free-space-weight", type=float, default=0.03)
    parser.add_argument("--scale-weight", type=float, default=0.01)
    parser.add_argument("--radius-weight", type=float, default=0.01)
    parser.add_argument("--gate-prior-weight", type=float, default=2e-4)
    parser.add_argument("--opacity-sparsity-weight", type=float, default=2e-4)
    parser.add_argument("--maximum-axis-ratio", type=float, default=4.0)
    parser.add_argument("--maximum-radius", type=float, default=96.0)
    parser.add_argument("--opacity-ceiling", type=float, default=0.08)
    parser.add_argument("--max-displacement", type=float, default=0.35)
    parser.add_argument("--appearance-rank", type=int, default=4)
    parser.add_argument("--appearance-lr", type=float, default=5e-4)
    parser.add_argument("--appearance-weight", type=float, default=0.08)
    parser.add_argument("--appearance-regularization-weight", type=float, default=1e-3)
    parser.add_argument("--appearance-start-fraction", type=float, default=0.55)
    parser.add_argument("--eval-indices", default="408,409,410")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--identity-rgb-tolerance", type=float, default=2e-5)
    parser.add_argument("--identity-aux-tolerance", type=float, default=1e-6)
    parser.add_argument("--identity-depth-tolerance", type=float, default=3e-5)
    args = parser.parse_args()
    dataset = model.extract(args)
    if not dataset.source_path or not dataset.model_path:
        parser.error("--source_path and --model_path are required")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if not 0 < args.volume_pretrain_fraction < 1:
        parser.error("--volume-pretrain-fraction must be in (0, 1)")
    if not 0 < args.joint_replacement_fraction < 1:
        parser.error("--joint-replacement-fraction must be in (0, 1)")
    if args.volume_pretrain_fraction + args.joint_replacement_fraction >= 1:
        parser.error("pretrain + joint fractions must leave a retirement stage")
    if not 0 <= args.minimum_gate <= args.counterfactual_gate <= 1:
        parser.error("gate bounds must satisfy 0 <= minimum <= counterfactual <= 1")
    if not 0 <= args.appearance_start_fraction < 1:
        parser.error("--appearance-start-fraction must be in [0, 1)")
    return args, dataset, pipeline.extract(args)


@torch.no_grad()
def _evaluate(
    views,
    indices,
    structural,
    foliage,
    sky,
    color,
    background,
    surface_gate,
    appearance=None,
    task_fields=None,
    output_dir=None,
):
    rows = []
    for index in indices:
        view = views[index]
        package = render_hybrid(
            view,
            structural,
            foliage,
            background=background,
            surface_gate=surface_gate,
        )
        prediction = composite_white_background(
            package.render, package.alpha, sky(view)
        )
        prediction = apply_per_image_affine_color_correction(
            color, prediction, view.image_name
        ).clamp(0, 1)
        conditioned = prediction
        task = None
        if appearance is not None:
            task = task_fields.fields(
                view.image_name,
                (view.image_height, view.image_width),
                prediction.device,
            )
            task["image_name"] = str(view.image_name)
            conditioned = appearance(prediction, view, task)
        target = view.original_image.cuda()
        mse = (prediction - target).square().mean()
        conditioned_mse = (conditioned - target).square().mean()
        rows.append(
            {
                "index": int(index),
                "image_name": str(view.image_name),
                "psnr": float((-10 * torch.log10(mse)).item()),
                "mae": float((prediction - target).abs().mean().item()),
                "conditioned_psnr": float(
                    (-10 * torch.log10(conditioned_mse)).item()
                ),
                "conditioned_mae": float(
                    (conditioned - target).abs().mean().item()
                ),
                "ssim": float(ssim(prediction, target).item()),
                "conditioned_ssim": float(ssim(conditioned, target).item()),
                "canopy": (
                    _region_metrics(prediction, target, task["p_canopy"])
                    if task is not None
                    else None
                ),
                "rigid": (
                    _region_metrics(prediction, target, task["p_rigid"])
                    if task is not None
                    else None
                ),
            }
        )
        if output_dir is not None:
            stem = f"{index:05d}_{view.image_name}"
            root = Path(output_dir)
            _save_rgb(root / f"{stem}_ground_truth.png", target)
            _save_rgb(root / f"{stem}_canonical.png", prediction)
            _save_rgb(root / f"{stem}_conditioned.png", conditioned)
            _save_rgb(
                root / f"{stem}_error_x4.png",
                (prediction - target).abs() * 4.0,
            )
            _save_scalar(root / f"{stem}_volume_alpha.png", package.volume_alpha)
            _save_scalar(root / f"{stem}_volume_depth.png", package.volume_depth)
    return rows


def main():
    args, dataset, pipe = _parse_args()
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
    views = scene.getTrainCameras()
    input_audit = _input_audit(Path(dataset.source_path).resolve(), views)
    background = torch.ones(3, device="cuda")
    empty_foliage = VolumetricFoliageModel(dataset.sh_degree).cuda()
    projection_identity_rows = []
    for index in sorted({0, len(views) // 2, len(views) - 1}):
        native = render_2dgs(views[index], structural, pipe, background)
        hybrid = render_hybrid(
            views[index],
            structural,
            empty_foliage,
            background=background,
        )
        fields = {
            "raw_rgb": (native["render"] - hybrid.render).abs(),
            "alpha": (native["rend_alpha"] - hybrid.alpha).abs(),
            "expected_depth": (native["rend_depth"] - hybrid.depth).abs(),
            "median_depth": (
                native["rend_depth_median"] - hybrid.median_depth
            ).abs(),
            "radii": (
                native["radii"].to(torch.int64)
                - hybrid.radii.to(torch.int64)
            ).abs().float(),
        }
        projection_identity_rows.append(
            {
                "index": index,
                "image_name": views[index].image_name,
                "fields": {
                    name: {
                        "max_abs": float(value.max().item()),
                        "mean_abs": float(value.mean().item()),
                    }
                    for name, value in fields.items()
                },
            }
        )
    structural_projection_identity = {
        "passed": all(
            row["fields"]["raw_rgb"]["max_abs"]
            <= args.identity_rgb_tolerance
            and row["fields"]["alpha"]["max_abs"]
            <= args.identity_aux_tolerance
            and row["fields"]["expected_depth"]["max_abs"]
            <= args.identity_depth_tolerance
            and row["fields"]["median_depth"]["max_abs"]
            <= args.identity_aux_tolerance
            and row["fields"]["radii"]["max_abs"] == 0.0
            for row in projection_identity_rows
        ),
        "complete_parent_contract": True,
        "missing_hybrid_outputs": [],
        "rgb_tolerance": args.identity_rgb_tolerance,
        "aux_tolerance": args.identity_aux_tolerance,
        "expected_depth_tolerance": args.identity_depth_tolerance,
        "kernel": "native_2d_surfel_plus_3d_ewa_shared_tile_sort",
        "views": projection_identity_rows,
    }
    if not structural_projection_identity["passed"]:
        (output / "rejected_zero_step_identity.json").write_text(
            json.dumps(structural_projection_identity, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            "Native mixed zero-volume path does not reproduce the parent "
            "2DGS contract within the declared tolerances"
        )

    foliage = VolumetricFoliageModel(dataset.sh_degree).cuda()
    try:
        seed_payload = torch.load(
            args.foliage_seed_state.resolve(),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        seed_payload = torch.load(
            args.foliage_seed_state.resolve(), map_location="cpu"
        )
    lifted = foliage.initialize_from_volume_state(seed_payload)
    seed_provenance = "independent_sfm_semantic_canopy_volume"
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
    appearance = OutdoorAppearanceUncertainty(
        [view.image_name for view in views],
        rank=args.appearance_rank,
        device="cuda",
    )
    if args.replacement_audit is None:
        mandatory_audit_indices = tuple(
            int(value)
            for value in args.audit_mandatory_indices.split(",")
            if value.strip()
        )
        responsibility_audit = audit_legacy_surface_responsibility(
            views,
            structural,
            empty_foliage,
            fields,
            background=background,
            view_count=args.audit_views,
            minimum_support_views=args.minimum_support_views,
            minimum_support_sequences=args.minimum_support_sequences,
            mandatory_view_indices=mandatory_audit_indices,
            maximum_rigid_responsibility=args.maximum_rigid_responsibility,
        )
        responsibility_path = responsibility_audit.save(
            output / "legacy_responsibility_audit.pth"
        )
    else:
        try:
            audit_payload = torch.load(
                args.replacement_audit.resolve(),
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            audit_payload = torch.load(
                args.replacement_audit.resolve(), map_location="cpu"
            )
        if audit_payload.get("version") != RESPONSIBILITY_VERSION:
            raise RuntimeError("Unsupported replacement audit version")
        responsibility_audit = LegacyResponsibilityAudit(
            candidate_indices=audit_payload["candidate_indices"],
            statistics=audit_payload["statistics"],
            selected_view_indices=audit_payload["selected_view_indices"],
            selected_view_names=audit_payload["selected_view_names"],
            thresholds=audit_payload["thresholds"],
        )
        responsibility_path = args.replacement_audit.resolve()
    candidates = responsibility_audit.candidate_indices.to(
        device="cuda", dtype=torch.long
    )
    if candidates.numel() == 0:
        raise RuntimeError("Replacement audit contains no candidates")
    candidate_gates = torch.nn.Parameter(
        torch.ones(candidates.numel(), device="cuda")
    )

    def effective_surface_gate(*, counterfactual: bool = False):
        base = torch.ones(
            structural.get_xyz.shape[0],
            device="cuda",
            dtype=structural.get_xyz.dtype,
        )
        values = (
            torch.full_like(candidate_gates, args.counterfactual_gate)
            if counterfactual
            else candidate_gates.clamp(args.minimum_gate, 1.0)
        )
        return base.index_copy(0, candidates, values)

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
            {
                "params": [candidate_gates],
                "lr": args.gate_lr,
                "name": "legacy_canopy_gate",
            },
            {
                "params": list(appearance.parameters()),
                "lr": args.appearance_lr,
                "name": "appearance_uncertainty",
            },
        ],
        eps=1e-15,
    )
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
        empty_foliage,
        sky,
        color,
        background,
        torch.ones_like(structural.get_opacity.reshape(-1)),
        appearance,
        fields,
        output / "visualization" / "before",
    )

    manifest = {
        "protocol": PROTOCOL,
        "run_role": "volumetric_foliage_candidate",
        "input": input_audit,
        "zero_foliage_structural_projection_identity": (
            structural_projection_identity
        ),
        "parents": {
            "structural_ply": str(args.structural_ply.resolve()),
            "structural_ply_sha256": _file_digest(args.structural_ply.resolve()),
            "foliage_seed_state": str(args.foliage_seed_state.resolve()),
            "foliage_seed_state_sha256": _file_digest(
                args.foliage_seed_state.resolve()
            ),
            "legacy_responsibility_audit": str(responsibility_path),
        },
        "active_components": {
            "structural_parameters_inherited_from_2d_surfels": True,
            "exact_2d_surfel_projection_preserved": True,
            "true_volumetric_foliage_3dgs": True,
            "native_2d_surfel_and_3d_ewa_shared_tile_depth_sort": True,
            "fixed_order_branch_compositing": False,
            "contribution_normalized_legacy_responsibility": True,
            "learnable_legacy_canopy_gate": True,
            "three_stage_replace_and_retire": True,
            "canonical_directional_sky": True,
            "low_rank_canopy_and_temporal_sky_appearance": True,
            "canopy_sky_heteroscedastic_uncertainty": True,
            "canonical_and_conditioned_metrics_separated": True,
            "frozen_inherited_train_camera_affine": True,
            "historical_full_real_rgb_initialization": False,
        },
        "foliage": {
            "seed_count": lifted,
            "seed_provenance": seed_provenance,
            "replacement_policy": (
                "independent SfM/semantic visual-hull volumes are pretrained, "
                "then contribution-audited legacy canopy gates retire"
            ),
            "legacy_foliage_surfel_active": True,
            "legacy_replacement_candidate_count": int(candidates.numel()),
            "scale_dimensions": 3,
            "topology": "frozen_during_3d_covariance_lift",
        },
        "structural": {
            "parameters_trainable": False,
            "representation": "native_perspective_correct_2d_surfel",
            "protected_except_candidate_opacity_gate": True,
        },
        "replacement_audit": {
            "version": RESPONSIBILITY_VERSION,
            "candidate_count": int(candidates.numel()),
            "thresholds": responsibility_audit.thresholds,
            "selected_views": responsibility_audit.selected_view_names,
        },
        "stages": {
            "volume_pretrain_end": int(
                args.iterations * args.volume_pretrain_fraction
            ),
            "joint_replacement_end": int(
                args.iterations
                * (
                    args.volume_pretrain_fraction
                    + args.joint_replacement_fraction
                )
            ),
            "final": args.iterations,
        },
        "schedule": schedule_audit,
        "appearance": appearance.audit(),
        "renderer_contract": {
            "mip_filter_inherited_from_ply": bool(structural.use_mip_filter),
            "projection": "exact_2d_ray_surfel_plus_3d_EWA",
            "background": "white_then_canonical_directional_sky",
            "cuda_sources": {
                str(path.relative_to(REPO_ROOT)): _file_digest(path)
                for path in sorted(
                    (
                        SURFEL_ROOT
                        / "submodules/diff-surfel-rasterization/cuda_rasterizer"
                    ).glob("*.cu")
                )
            },
        },
    }
    (output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    trace = output / "training_trace.jsonl"
    start = time.time()
    pretrain_end = max(
        1, int(args.iterations * args.volume_pretrain_fraction)
    )
    joint_end = max(
        pretrain_end + 1,
        int(
            args.iterations
            * (
                args.volume_pretrain_fraction
                + args.joint_replacement_fraction
            )
        ),
    )
    initial_scales = foliage.scales.detach().clone()
    appearance_start = int(
        args.iterations * args.appearance_start_fraction
    )
    posterior_variance = seed_payload["inverse_depth_variance"].to(
        device="cuda", dtype=foliage.xyz.dtype
    ).clamp_min(1e-6)
    position_weight = (
        1.0 / (
            initial_scales.mean(dim=-1).square()
            + posterior_variance
        )
    ).clamp_max(1e3)
    replacement_support_views: set[str] = set()
    replacement_support_sequences: set[str] = set()
    low_gate_steps = torch.zeros_like(
        candidate_gates, dtype=torch.int32
    )
    progress = tqdm(range(args.iterations), desc="native mixed replacement")
    for step in progress:
        view = views[int(schedule[step])]
        if step < pretrain_end:
            stage = "A_volume_pretrain"
            render_gate = effective_surface_gate().detach()
        elif step < joint_end:
            stage = "B_joint_replacement"
            render_gate = effective_surface_gate()
        else:
            stage = "C_delayed_retirement"
            render_gate = effective_surface_gate()
        target = view.original_image.cuda()
        task = fields.fields(
            view.image_name,
            (view.image_height, view.image_width),
            target.device,
        )
        package = render_hybrid(
            view,
            structural,
            foliage,
            background=background,
            surface_gate=render_gate,
        )
        prediction = composite_white_background(
            package.render, package.alpha, sky(view)
        )
        prediction = apply_per_image_affine_color_correction(
            color, prediction, view.image_name
        )
        task["image_name"] = str(view.image_name)
        conditioned_prediction = appearance(prediction, view, task)
        weight = task["w_topology"] + 0.05 * task["p_rigid"]
        photometric = _loss(prediction, target, weight)
        appearance_loss = appearance.heteroscedastic_loss(
            conditioned_prediction, target, task
        )
        appearance_regularization = appearance.regularization()
        counterfactual = render_hybrid(
            view,
            structural,
            foliage,
            background=background,
            surface_gate=effective_surface_gate(counterfactual=True),
            audit_fields=torch.stack(
                [task["p_canopy"], task["p_rigid"]], dim=0
            ),
        )
        counterfactual_prediction = apply_per_image_affine_color_correction(
            color,
            composite_white_background(
                counterfactual.render,
                counterfactual.alpha,
                sky(view),
            ),
            view.image_name,
        )
        counterfactual_photo = _loss(
            counterfactual_prediction,
            target,
            task["w_topology"] + 0.1 * task["p_rigid"],
        )
        gradient_x = _loss(
            counterfactual_prediction[:, :, 1:]
            - counterfactual_prediction[:, :, :-1],
            target[:, :, 1:] - target[:, :, :-1],
            task["p_canopy_core"][:, 1:]
            * task["p_canopy_core"][:, :-1],
        )
        gradient_y = _loss(
            counterfactual_prediction[:, 1:, :]
            - counterfactual_prediction[:, :-1, :],
            target[:, 1:, :] - target[:, :-1, :],
            task["p_canopy_core"][1:, :]
            * task["p_canopy_core"][:-1, :],
        )
        canopy_gradient = 0.5 * (gradient_x + gradient_y)
        silhouette = (
            (
                (1.0 - counterfactual.volume_alpha)
                * task["p_canopy"][None]
            ).square().mean()
        )
        rigid_spill = _loss(
            counterfactual_prediction,
            target,
            task["p_rigid"],
        )
        normal_rigid = _loss(prediction, target, task["p_rigid"])
        rigid_alpha_spill = (
            counterfactual.volume_alpha
            * task["p_rigid"][None]
        ).square().mean()
        displacement = foliage.xyz - initial_xyz
        depth_prior = (
            displacement.square().sum(dim=-1) * position_weight
        ).mean()
        rotation = torch.as_tensor(
            view.R, device="cuda", dtype=foliage.xyz.dtype
        )
        translation = torch.as_tensor(
            view.T, device="cuda", dtype=foliage.xyz.dtype
        )
        current_depth = (foliage.xyz @ rotation + translation)[:, 2]
        initial_depth = (initial_xyz @ rotation + translation)[:, 2]
        free_space = torch.relu(
            initial_depth - 0.25 - current_depth
        ).square().mean()
        scales = foliage.scales
        axis_ratio = scales.amax(dim=-1) / scales.amin(dim=-1).clamp_min(1e-6)
        scale_loss = (
            torch.relu(axis_ratio / args.maximum_axis_ratio - 1.0).square()
            + torch.relu(
                scales.amax(dim=-1)
                / (2.0 * initial_scales.amax(dim=-1)).clamp_min(1e-6)
                - 1.0
            ).square()
        ).mean()
        focal = max(float(view.focal_x), float(view.focal_y))
        projected_radius = (
            3.0
            * focal
            * scales.amax(dim=-1)
            / current_depth.clamp_min(float(view.znear))
        )
        volume_visible = (
            package.radii[package.structural_count :] > 0
        ) & (current_depth > float(view.znear))
        radius_penalty = torch.relu(
            projected_radius / args.maximum_radius - 1.0
        ).square()
        radius_loss = (
            radius_penalty[volume_visible].mean()
            if bool(volume_visible.any())
            else radius_penalty.new_zeros(())
        )
        gate_values = candidate_gates.clamp(args.minimum_gate, 1.0)
        gate_prior = (
            (1.0 - gate_values).mean()
            if stage == "B_joint_replacement"
            else gate_values.mean()
            if stage == "C_delayed_retirement"
            else gate_values.new_zeros(())
        )
        opacity_regularizer = foliage.opacities.mean()
        loss = (
            (0.1 if stage == "A_volume_pretrain" else 1.0) * photometric
            + args.counterfactual_weight * counterfactual_photo
            + args.silhouette_weight * silhouette
            + args.rigid_spill_weight * rigid_spill
            + args.normal_rigid_weight * normal_rigid
            + args.rigid_alpha_spill_weight * rigid_alpha_spill
            + args.canopy_gradient_weight * canopy_gradient
            + args.depth_prior_weight * depth_prior
            + args.free_space_weight * free_space
            + args.scale_weight * scale_loss
            + args.radius_weight * radius_loss
            + args.gate_prior_weight * gate_prior
            + args.opacity_sparsity_weight * opacity_regularizer
            + (
                args.appearance_weight * appearance_loss
                + args.appearance_regularization_weight
                * appearance_regularization
                if step >= appearance_start
                else appearance_loss.new_zeros(())
            )
        )
        loss.backward()
        responsibility = counterfactual.responsibility
        if responsibility is not None:
            volume = responsibility[package.structural_count :]
            canopy_support = float(volume[:, 1].sum().item())
            rigid_support = float(volume[:, 2].sum().item())
            if (
                canopy_support >= 1.0
                and rigid_support <= 0.5 * canopy_support
            ):
                replacement_support_views.add(str(view.image_name))
                replacement_support_sequences.add(
                    str(view.image_name).split("__", 1)[0]
                )
        gate_unlocked = (
            len(replacement_support_views) >= args.minimum_support_views
            and len(replacement_support_sequences)
            >= args.minimum_support_sequences
            and stage != "A_volume_pretrain"
        )
        if not gate_unlocked:
            candidate_gates.grad = None
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
            candidate_gates.clamp_(args.minimum_gate, 1.0)
            if stage == "C_delayed_retirement":
                low_gate_steps += (candidate_gates < 0.05).to(torch.int32)
        if step == 0 or (step + 1) % args.log_every == 0:
            row = {
                "iteration": step + 1,
                "camera": view.image_name,
                "stage": stage,
                "loss": float(loss.item()),
                "photometric": float(photometric.item()),
                "counterfactual_photo": float(counterfactual_photo.item()),
                "silhouette": float(silhouette.item()),
                "rigid_spill": float(rigid_spill.item()),
                "normal_rigid": float(normal_rigid.item()),
                "rigid_alpha_spill": float(rigid_alpha_spill.item()),
                "canopy_gradient": float(canopy_gradient.item()),
                "free_space": float(free_space.item()),
                "scale": float(scale_loss.item()),
                "radius": float(radius_loss.item()),
                "appearance": float(appearance_loss.item()),
                "appearance_active": step >= appearance_start,
                "mean_opacity": float(foliage.opacities.mean().item()),
                "mean_legacy_gate": float(candidate_gates.mean().item()),
                "minimum_legacy_gate": float(candidate_gates.min().item()),
                "gate_unlocked": gate_unlocked,
                "replacement_support_view_count": len(
                    replacement_support_views
                ),
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
        effective_surface_gate().detach(),
        appearance,
        fields,
        output / "visualization" / "after",
    )
    state_path = output / "volumetric_foliage_state.pth"
    retirement_window = max(1, (args.iterations - joint_end) // 2)
    rigid_responsibility = responsibility_audit.statistics[
        "rigid_responsibility"
    ].to(device="cuda")[candidates]
    retirement_eligible = (
        (candidate_gates.detach() < 0.05)
        & (low_gate_steps >= retirement_window)
        & (rigid_responsibility < 0.10)
    )
    torch.save(
        {
            "protocol": PROTOCOL,
            "foliage": foliage.capture(),
            "legacy_candidate_indices": candidates.detach(),
            "legacy_candidate_gates": candidate_gates.detach(),
            "retirement_eligible_indices": candidates[
                retirement_eligible
            ].detach(),
            "optimizer": optimizer.state_dict(),
            "appearance": appearance.capture(),
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
                "initial_conditioned_psnr": initial["conditioned_psnr"],
                "initial_conditioned_mae": initial["conditioned_mae"],
                "delta_conditioned_psnr": (
                    final["conditioned_psnr"]
                    - initial["conditioned_psnr"]
                ),
                "delta_conditioned_mae": (
                    final["conditioned_mae"]
                    - initial["conditioned_mae"]
                ),
                "delta_canopy_psnr": (
                    final["canopy"]["psnr"] - initial["canopy"]["psnr"]
                ),
                "delta_rigid_psnr": (
                    final["rigid"]["psnr"] - initial["rigid"]["psnr"]
                ),
            }
        )
    fixed_views_pass = all(
        row["delta_psnr"] >= 0.0
        and row["delta_rigid_psnr"] >= -0.01
        for row in paired
    )
    result_payload = {
        "protocol": PROTOCOL,
        "validation_scope": (
            "exact-parent-zero-volume-plus_targeted_native-mixed"
        ),
        "mainline_eligible": bool(
            structural_projection_identity["passed"]
            and fixed_views_pass
        ),
        "mainline_blocker": (
            None
            if fixed_views_pass
            else "one or more fixed difficult/rigid regions regressed"
        ),
        "iterations": args.iterations,
        "structural_gaussians": int(len(structural.get_xyz)),
        "volumetric_foliage_gaussians": int(len(foliage)),
        "legacy_replacement_candidates": int(candidates.numel()),
        "legacy_gate": {
            "mean": float(candidate_gates.mean().item()),
            "minimum": float(candidate_gates.min().item()),
            "below_0_5": int((candidate_gates < 0.5).sum().item()),
            "retirement_eligible": int(retirement_eligible.sum().item()),
        },
        "replacement_support": {
            "distinct_views": len(replacement_support_views),
            "distinct_sequences": len(replacement_support_sequences),
        },
        "zero_volume_identity": structural_projection_identity,
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
