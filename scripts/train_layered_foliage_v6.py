#!/usr/bin/env python
"""Train static skeleton, porous crown and sequence-conditioned dynamic leaves."""

from __future__ import annotations

import argparse
import gc
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

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render as render_2dgs  # noqa: E402
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
from outdoor.foliage_responsibility import (  # noqa: E402
    RESPONSIBILITY_VERSION,
)
from outdoor.foliage_view_graph import sequence_id  # noqa: E402
from outdoor.hybrid_gaussian_renderer import (  # noqa: E402
    LAYER_CANONICAL_CROWN,
    LAYER_DYNAMIC_LEAF,
    LAYER_STATIC_SKELETON,
    VolumetricFoliageModel,
    render_hybrid,
)
from outdoor.task_fields import OutdoorTaskFieldLookup  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from scripts.train_hybrid_foliage_3dgs import (  # noqa: E402
    _evaluate,
    _loss,
    _region_metrics,
    _save_rgb,
    _save_scalar,
)
from utils.loss_utils import ssim  # noqa: E402


PROTOCOL = "layered_dynamic_foliage_surfel_uv_replace_v7"
LEGACY_PROTOCOL = "layered_dynamic_foliage_local_replace_v6"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.set_defaults(
        iterations=20_000,
        data_device="cpu",
        white_background=True,
    )
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--structural-ply", type=Path, required=True)
    parser.add_argument("--foliage-seed-state", type=Path, required=True)
    parser.add_argument("--replacement-audit", type=Path, required=True)
    parser.add_argument("--sky-model", type=Path, required=True)
    parser.add_argument("--color-correction", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path, required=True)
    parser.add_argument("--task-semantic-manifest", type=Path, required=True)
    parser.add_argument("--eval-indices", default="408,409,410")
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument("--dynamic-rank", type=int, default=4)
    parser.add_argument("--dynamic-seed-count", type=int, default=12_000)
    parser.add_argument("--position-lr", type=float, default=8e-5)
    parser.add_argument("--feature-lr", type=float, default=8e-4)
    parser.add_argument("--opacity-lr", type=float, default=4e-3)
    parser.add_argument("--scale-lr", type=float, default=4e-4)
    parser.add_argument("--rotation-lr", type=float, default=2e-4)
    parser.add_argument("--dynamic-lr", type=float, default=3e-4)
    parser.add_argument("--gate-lr", type=float, default=6e-3)
    parser.add_argument("--uv-gate-size", type=int, default=8)
    parser.add_argument("--exposure-weight", type=float, default=0.35)
    parser.add_argument("--exposure-start-fraction", type=float, default=0.05)
    parser.add_argument("--posterior-depth-sigma", type=float, default=0.45)
    parser.add_argument("--ray-hit-weight", type=float, default=0.03)
    parser.add_argument("--free-space-weight", type=float, default=0.04)
    parser.add_argument("--appearance-lr", type=float, default=8e-4)
    parser.add_argument("--pretrain-fraction", type=float, default=0.30)
    parser.add_argument("--retirement-fraction", type=float, default=0.65)
    parser.add_argument("--counterfactual-gate", type=float, default=0.03)
    parser.add_argument("--local-confidence-threshold", type=float, default=0.20)
    parser.add_argument("--retirement-confidence", type=float, default=0.45)
    parser.add_argument("--local-audit-every", type=int, default=10)
    parser.add_argument("--densify-start", type=int, default=1200)
    parser.add_argument("--densify-until-fraction", type=float, default=0.78)
    parser.add_argument("--densify-every", type=int, default=600)
    parser.add_argument("--maximum-splits", type=int, default=4000)
    parser.add_argument(
        "--dynamic-split-fraction",
        type=float,
        default=0.40,
        help="Reserved fraction of each adaptive split event for dynamic leaves.",
    )
    parser.add_argument("--split-radius", type=float, default=8.0)
    parser.add_argument("--prune-opacity", type=float, default=0.0025)
    parser.add_argument("--maximum-gaussians", type=int, default=240_000)
    parser.add_argument("--soft-crown-alpha", type=float, default=0.48)
    parser.add_argument("--occupancy-weight", type=float, default=0.08)
    parser.add_argument("--canonical-weight", type=float, default=0.45)
    parser.add_argument("--conditioned-weight", type=float, default=0.75)
    parser.add_argument("--rigid-weight", type=float, default=0.30)
    parser.add_argument("--depth-weight", type=float, default=0.03)
    parser.add_argument("--dynamic-regularization", type=float, default=0.02)
    parser.add_argument("--appearance-weight", type=float, default=0.06)
    parser.add_argument("--gate-weight", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume an interrupted v6 run from a periodic checkpoint.",
    )
    parser.add_argument(
        "--initial-layered-state",
        type=Path,
        help=(
            "Warm-start foliage/appearance and remap legacy gates into a new "
            "candidate audit for a second-stage local retirement pass."
        ),
    )
    parser.add_argument(
        "--certified-initial-gates",
        action="store_true",
        help=(
            "On a warm start, preserve low gates only for structural indices "
            "listed in retirement_eligible_indices; reset all others to one."
        ),
    )
    parser.add_argument(
        "--reset-initial-replacement-proof",
        action="store_true",
        help=(
            "Warm-start foliage/appearance but reset every legacy gate and "
            "local proof accumulator. Use after changing the replacement "
            "proof definition so stale retirements cannot be inherited."
        ),
    )
    args = parser.parse_args()
    dataset = model.extract(args)
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.resume and args.initial_layered_state:
        parser.error("--resume and --initial-layered-state are mutually exclusive")
    if args.reset_initial_replacement_proof and not args.initial_layered_state:
        parser.error(
            "--reset-initial-replacement-proof requires "
            "--initial-layered-state"
        )
    if args.reset_initial_replacement_proof and args.certified_initial_gates:
        parser.error(
            "--reset-initial-replacement-proof and "
            "--certified-initial-gates are mutually exclusive"
        )
    if not 0 <= args.dynamic_split_fraction <= 1:
        parser.error("--dynamic-split-fraction must be in [0, 1]")
    if args.uv_gate_size < 2:
        parser.error("--uv-gate-size must be at least 2")
    if not 0 < args.pretrain_fraction < args.retirement_fraction < 1:
        parser.error("stage fractions must satisfy 0 < pretrain < retirement < 1")
    return args, dataset, pipeline.extract(args)


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _save_checkpoint(path, payload):
    """Atomically replace the restart checkpoint after a completed step."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def _restore_geometry_evidence(foliage, seed_payload, captured_foliage):
    """Backfill v6 states that predate persistent ray-posterior buffers.

    Dynamic clones and historical split children retain the independent seed
    centre, so one stable centre lookup restores evidence without consulting
    RGB or the old structural surfel geometry.
    """
    names = (
        "ray_depth_nll",
        "free_space_violation_count",
        "unknown_view_count",
    )
    missing = [name for name in names if name not in captured_foliage]
    if not missing:
        return {"needed": False, "matched": len(foliage), "unmatched": 0}
    seed_centres = seed_payload["centers"].detach().cpu().float()
    current_centres = foliage.initialization_center.detach().cpu().float()
    exact = {
        tuple(float(value) for value in centre): index
        for index, centre in enumerate(seed_centres.tolist())
    }
    # Quantized fallback only handles serialization roundoff; collisions keep
    # the first independent primitive and are reported.
    quantized = {}
    collisions = 0
    for index, centre in enumerate(seed_centres.tolist()):
        key = tuple(round(float(value), 5) for value in centre)
        if key in quantized:
            collisions += 1
        else:
            quantized[key] = index
    mapped = []
    for centre in current_centres.tolist():
        key = tuple(float(value) for value in centre)
        index = exact.get(key)
        if index is None:
            index = quantized.get(
                tuple(round(float(value), 5) for value in centre)
            )
        mapped.append(-1 if index is None else int(index))
    mapped = torch.tensor(mapped, dtype=torch.long, device=foliage.xyz.device)
    valid = mapped >= 0
    for name in missing:
        destination = getattr(foliage, name)
        source = seed_payload[name].to(
            device=destination.device, dtype=destination.dtype
        )
        destination[valid] = source[mapped[valid]]
    return {
        "needed": True,
        "matched": int(valid.sum()),
        "unmatched": int((~valid).sum()),
        "quantized_seed_collisions": int(collisions),
        "restored_fields": missing,
    }


def _remap_candidate_values(
    new_candidates,
    old_candidates,
    old_values,
    default,
):
    """Map per-candidate state by stable structural primitive index."""
    result = default.clone()
    lookup = {
        int(candidate): index
        for index, candidate in enumerate(old_candidates.cpu().tolist())
    }
    destination = []
    source = []
    for index, candidate in enumerate(new_candidates.cpu().tolist()):
        old_index = lookup.get(int(candidate))
        if old_index is not None:
            destination.append(index)
            source.append(old_index)
    if destination:
        result[torch.as_tensor(destination, device=result.device)] = old_values[
            torch.as_tensor(source, device=old_values.device)
        ].to(device=result.device, dtype=result.dtype)
    return result


def _make_schedule(views, foliage, iterations, seed):
    """Full shuffled epochs interleaved with direct support-camera samples."""
    rng = np.random.default_rng(seed)
    camera_to_index = {
        int(getattr(view, "colmap_id", -1)): index
        for index, view in enumerate(views)
    }
    support = foliage.support_camera_ids.detach().cpu().numpy().reshape(-1)
    support_indices = np.asarray(
        sorted(
            {
                camera_to_index[int(value)]
                for value in support
                if int(value) in camera_to_index
            }
        ),
        dtype=np.int64,
    )
    output = []
    while len(output) < iterations:
        epoch = rng.permutation(len(views)).tolist()
        if len(support_indices):
            direct = rng.permutation(support_indices).tolist()
            interleaved = []
            for offset, index in enumerate(epoch):
                interleaved.append(index)
                if offset % 2 == 0 and direct:
                    interleaved.append(direct.pop())
            epoch = interleaved
        output.extend(epoch)
    return np.asarray(output[:iterations], dtype=np.int64), {
        "policy": "full_epochs_interleaved_with_metadata_support_views",
        "direct_support_view_count": int(len(support_indices)),
        "minimum_complete_epochs": int(iterations // max(len(views), 1)),
    }


def _optimizer(args, foliage, gate_atlas, appearance):
    return torch.optim.Adam(
        [
            {"params": [foliage.xyz], "lr": args.position_lr, "name": "xyz"},
            {"params": [foliage.features], "lr": args.feature_lr, "name": "features"},
            {"params": [foliage.opacity_logits], "lr": args.opacity_lr, "name": "opacity"},
            {"params": [foliage.log_scales], "lr": args.scale_lr, "name": "scale"},
            {"params": [foliage.quaternions], "lr": args.rotation_lr, "name": "rotation"},
            {
                "params": [
                    foliage.deformation_basis,
                    foliage.dynamic_feature_basis,
                    foliage.dynamic_opacity_basis,
                ],
                "lr": args.dynamic_lr,
                "name": "dynamic_leaf",
            },
            {
                "params": [gate_atlas],
                "lr": args.gate_lr,
                "name": "surfel_uv_gate",
            },
            {
                "params": list(appearance.parameters()),
                "lr": args.appearance_lr,
                "name": "spatial_sequence_conditioner",
            },
        ],
        eps=1e-15,
    )


def _optimizer_with_migrated_state(
    args,
    foliage,
    gate_atlas,
    appearance,
    previous,
    new_to_old,
):
    """Rebuild Adam after topology surgery without discarding its history."""
    current = _optimizer(args, foliage, gate_atlas, appearance)
    previous_groups = {
        group.get("name"): group for group in previous.param_groups
    }
    primitive_groups = {
        "xyz",
        "features",
        "opacity",
        "scale",
        "rotation",
        "dynamic_leaf",
    }
    for group in current.param_groups:
        name = group.get("name")
        old_group = previous_groups.get(name)
        if old_group is None or len(old_group["params"]) != len(group["params"]):
            continue
        for old_parameter, new_parameter in zip(
            old_group["params"], group["params"]
        ):
            old_state = previous.state.get(old_parameter)
            if not old_state:
                continue
            state = {}
            for key, value in old_state.items():
                if (
                    name in primitive_groups
                    and torch.is_tensor(value)
                    and value.ndim > 0
                    and value.shape[0] == old_parameter.shape[0]
                ):
                    state[key] = value[new_to_old].clone()
                elif torch.is_tensor(value):
                    state[key] = value.clone()
                else:
                    state[key] = value
            current.state[new_parameter] = state
    return current


def _gate_vector(structural, candidates, values):
    gate = torch.ones(
        structural.get_xyz.shape[0],
        device=values.device,
        dtype=values.dtype,
    )
    return gate.index_copy(0, candidates, values.clamp(0.0, 1.0))


def _surface_atlas_indices(structural, candidates):
    """Map immutable structural primitive ids to compact UV atlas rows."""
    indices = torch.full(
        (structural.get_xyz.shape[0],),
        -1,
        dtype=torch.int32,
        device=candidates.device,
    )
    return indices.index_copy(
        0,
        candidates,
        torch.arange(
            len(candidates), dtype=torch.int32, device=candidates.device
        ),
    )


def _uv_total_variation(atlas, mask):
    """Regularize only certified texel pairs; rigid texels stay untouched."""
    horizontal = mask[:, :, 1:] & mask[:, :, :-1]
    vertical = mask[:, 1:, :] & mask[:, :-1, :]
    terms = []
    if bool(horizontal.any()):
        terms.append(
            (atlas[:, :, 1:] - atlas[:, :, :-1]).abs()[horizontal].mean()
        )
    if bool(vertical.any()):
        terms.append(
            (atlas[:, 1:, :] - atlas[:, :-1, :]).abs()[vertical].mean()
        )
    return sum(terms, atlas.new_zeros(()))


def _atlas_candidate_means(atlas):
    return atlas.detach().mean(dim=(1, 2))


def _joint_replacement_match(depth_match, removal_improvement):
    """AND-like local proof for geometry and photometric counterfactual."""
    return depth_match.clamp(0, 1) * removal_improvement.clamp(0, 1)


def _spawn_dynamic_layer(foliage, maximum):
    eligible = (
        (foliage.layer_role == LAYER_CANONICAL_CROWN)
        & (foliage.support_sequence_count >= 2)
    )
    indices = torch.nonzero(eligible, as_tuple=False).flatten()
    if len(indices) > int(maximum):
        score = (
            foliage.occupancy_probability[indices]
            * foliage.support_sequence_count[indices].float()
            / foliage.position_covariance[indices]
            .diagonal(dim1=-2, dim2=-1)
            .sum(-1)
            .sqrt()
            .clamp_min(1e-4)
        )
        indices = indices[torch.topk(score, int(maximum)).indices]
    return foliage.append_dynamic_leaves(indices)


def _reset_topology_statistics(foliage):
    device = foliage.xyz.device
    count = len(foliage)
    return {
        "gradient": torch.zeros(count, device=device),
        "gradient_count": torch.zeros(count, device=device),
        "radius": torch.zeros(count, device=device),
        "contribution": torch.zeros(count, device=device),
        "residual": torch.zeros(count, device=device),
        "rigid": torch.zeros(count, device=device),
    }


@torch.no_grad()
def _accumulate_topology(stats, package, responsibility):
    offset = package.structural_count
    radii = package.radii[offset:].float()
    stats["radius"] = torch.maximum(stats["radius"], radii)
    if (
        package.volume_means2d is not None
        and package.volume_means2d.grad is not None
    ):
        gradient = package.volume_means2d.grad[:, :2].norm(dim=-1)
        visible = radii > 0
        stats["gradient"][visible] += gradient[visible]
        stats["gradient_count"][visible] += 1
    if responsibility is not None:
        volume = responsibility[offset:]
        total = volume[:, 0]
        stats["contribution"] += total
        if volume.shape[1] >= 4:
            stats["rigid"] += volume[:, 2]
            stats["residual"] += volume[:, 3]


@torch.no_grad()
def _adaptive_topology(args, foliage, stats):
    count = len(foliage)
    gradient = stats["gradient"] / stats["gradient_count"].clamp_min(1)
    residual = stats["residual"] / stats["contribution"].clamp_min(1e-8)
    rigid = stats["rigid"] / stats["contribution"].clamp_min(1e-8)
    base_eligible = (
        ~foliage.static_skeleton_mask
        & (foliage.support_sequence_count >= 2)
        & (stats["radius"] >= args.split_radius)
        & (rigid <= 0.15)
        & (stats["contribution"] > 0)
    )
    if bool(base_eligible.any()) and count < args.maximum_gaussians:
        def qualified(role_mask):
            role = base_eligible & role_mask
            if not bool(role.any()):
                return torch.empty(
                    0, device=gradient.device, dtype=torch.long
                )
            gradient_threshold = torch.quantile(gradient[role], 0.60)
            residual_threshold = torch.quantile(residual[role], 0.50)
            return torch.nonzero(
                role
                & (gradient >= gradient_threshold)
                & (residual >= residual_threshold),
                as_tuple=False,
            ).flatten()

        dynamic_indices = qualified(foliage.dynamic_leaf_mask)
        canonical_indices = qualified(~foliage.dynamic_leaf_mask)
        capacity = min(
            int(args.maximum_splits),
            int(args.maximum_gaussians - count),
        )

        def top(indices, maximum):
            maximum = min(int(maximum), len(indices))
            if maximum <= 0:
                return indices[:0]
            score = gradient[indices] * residual[indices]
            return indices[torch.topk(score, maximum).indices]

        dynamic_capacity = min(
            len(dynamic_indices),
            int(round(capacity * args.dynamic_split_fraction)),
        )
        selected_dynamic = top(dynamic_indices, dynamic_capacity)
        selected_canonical = top(
            canonical_indices, capacity - len(selected_dynamic)
        )
        remaining = capacity - len(selected_dynamic) - len(selected_canonical)
        if remaining > 0 and len(dynamic_indices) > len(selected_dynamic):
            chosen = torch.zeros(
                len(foliage), device=gradient.device, dtype=torch.bool
            )
            chosen[selected_dynamic] = True
            dynamic_remainder = dynamic_indices[~chosen[dynamic_indices]]
            selected_dynamic = torch.cat(
                [selected_dynamic, top(dynamic_remainder, remaining)]
            )
        indices = torch.cat([selected_dynamic, selected_canonical])
        split = foliage.split(indices)
        split["dynamic_split_parents"] = int(len(selected_dynamic))
        split["canonical_split_parents"] = int(len(selected_canonical))
    else:
        split = {
            "split_parents": 0,
            "children": 0,
            "dynamic_split_parents": 0,
            "canonical_split_parents": 0,
        }
    # Pruning is evidence-based and never applies to the protected skeleton.
    if split["split_parents"] == 0:
        weak_contribution = stats["contribution"] <= torch.quantile(
            stats["contribution"], 0.10
        )
        positive = foliage.support_view_count.float()
        confirmed_free = foliage.free_space_violation_count.float()
        free_space_rate = confirmed_free / (
            positive + confirmed_free
        ).clamp_min(1)
        finite_depth = torch.isfinite(foliage.ray_depth_nll)
        depth_threshold = (
            torch.quantile(foliage.ray_depth_nll[finite_depth], 0.90)
            if bool(finite_depth.any())
            else foliage.ray_depth_nll.new_tensor(float("inf"))
        )
        contradiction = (
            (rigid >= 0.35)
            | (free_space_rate >= 0.30)
            | (foliage.occupancy_probability <= 0.15)
            | (finite_depth & (foliage.ray_depth_nll >= depth_threshold))
            | (
                torch.isfinite(foliage.reprojection_error)
                & (foliage.reprojection_error >= 3.0)
            )
        )
        persistently_transparent = (
            foliage.opacities < args.prune_opacity
        )
        remove = (
            weak_contribution
            & (contradiction | persistently_transparent)
        )
        pruned, new_to_old = foliage.prune(remove)
        if pruned:
            split["_new_to_old"] = new_to_old
    else:
        pruned = 0
    return {**split, "pruned": int(pruned)}


def _apply_role_gradient_policy(foliage):
    skeleton = foliage.static_skeleton_mask
    dynamic = foliage.dynamic_leaf_mask
    crown = foliage.canonical_crown_mask
    if foliage.xyz.grad is not None:
        foliage.xyz.grad[skeleton] *= 0.08
        foliage.xyz.grad[dynamic] *= 1.5
    if foliage.log_scales.grad is not None:
        foliage.log_scales.grad[skeleton] *= 0.05
        foliage.log_scales.grad[dynamic] *= 1.25
    if foliage.opacity_logits.grad is not None:
        foliage.opacity_logits.grad[skeleton] *= 0.25
        foliage.opacity_logits.grad[crown] *= 0.75
    for parameter in (
        foliage.deformation_basis,
        foliage.dynamic_feature_basis,
        foliage.dynamic_opacity_basis,
    ):
        if parameter.grad is not None:
            parameter.grad[~dynamic] = 0


@torch.no_grad()
def _evaluate_layered(
    views,
    indices,
    structural,
    foliage,
    sky,
    color,
    background,
    surface_gate,
    appearance,
    fields,
    output,
    surface_gate_indices=None,
    surface_gate_atlas=None,
):
    rows = []
    for index in indices:
        view = views[index]
        task = fields.fields(
            view.image_name,
            (view.image_height, view.image_width),
            background.device,
        )
        task["image_name"] = str(view.image_name)
        canonical_package = render_hybrid(
            view,
            structural,
            foliage,
            background=background,
            surface_gate=surface_gate,
            surface_gate_indices=surface_gate_indices,
            surface_gate_atlas=surface_gate_atlas,
            include_dynamic=False,
        )
        conditioned_package = render_hybrid(
            view,
            structural,
            foliage,
            background=background,
            surface_gate=surface_gate,
            surface_gate_indices=surface_gate_indices,
            surface_gate_atlas=surface_gate_atlas,
            temporal_code=appearance.temporal_code(view.image_name),
            include_dynamic=True,
        )
        structural_only_package = render_hybrid(
            view,
            structural,
            foliage,
            background=background,
            surface_gate=torch.ones_like(surface_gate),
            include_dynamic=False,
            volume_opacity_scale=0.0,
        )
        gated_structural_package = render_hybrid(
            view,
            structural,
            foliage,
            background=background,
            surface_gate=surface_gate,
            surface_gate_indices=surface_gate_indices,
            surface_gate_atlas=surface_gate_atlas,
            include_dynamic=False,
            volume_opacity_scale=0.0,
        )
        volume_only_package = render_hybrid(
            view,
            structural,
            foliage,
            background=background,
            surface_gate=torch.zeros_like(surface_gate),
            temporal_code=appearance.temporal_code(view.image_name),
            include_dynamic=True,
        )
        canonical = apply_per_image_affine_color_correction(
            color,
            composite_white_background(
                canonical_package.render,
                canonical_package.alpha,
                sky(view),
            ),
            view.image_name,
        ).clamp(0, 1)
        conditioned_base = apply_per_image_affine_color_correction(
            color,
            composite_white_background(
                conditioned_package.render,
                conditioned_package.alpha,
                sky(view),
            ),
            view.image_name,
        )
        conditioned = appearance(
            conditioned_base, view, task
        ).clamp(0, 1)
        def corrected(package):
            return apply_per_image_affine_color_correction(
                color,
                composite_white_background(
                    package.render, package.alpha, sky(view)
                ),
                view.image_name,
            ).clamp(0, 1)

        structural_only = corrected(structural_only_package)
        gated_structural = corrected(gated_structural_package)
        volume_only = corrected(volume_only_package)
        target = view.original_image.cuda()
        row = {
            "index": int(index),
            "image_name": str(view.image_name),
            "canonical_psnr": float(
                (-10 * torch.log10((canonical - target).square().mean())).item()
            ),
            "conditioned_psnr": float(
                (-10 * torch.log10((conditioned - target).square().mean())).item()
            ),
            "canonical_mae": float((canonical - target).abs().mean()),
            "conditioned_mae": float((conditioned - target).abs().mean()),
            "canonical_ssim": float(ssim(canonical, target)),
            "conditioned_ssim": float(ssim(conditioned, target)),
            "canonical_canopy": _region_metrics(
                canonical, target, task["p_canopy"]
            ),
            "conditioned_canopy": _region_metrics(
                conditioned, target, task["p_canopy"]
            ),
            "rigid": _region_metrics(canonical, target, task["p_rigid"]),
        }
        rows.append(row)
        stem = f"{index:05d}_{view.image_name}"
        _save_rgb(output / f"{stem}_ground_truth.png", target)
        _save_rgb(output / f"{stem}_canonical.png", canonical)
        _save_rgb(output / f"{stem}_conditioned.png", conditioned)
        _save_rgb(
            output / f"{stem}_branch_structural_ungated.png",
            structural_only,
        )
        _save_rgb(
            output / f"{stem}_branch_structural_uv_gated.png",
            gated_structural,
        )
        _save_rgb(
            output / f"{stem}_branch_volume_only.png",
            volume_only,
        )
        _save_rgb(
            output / f"{stem}_canonical_error_x4.png",
            (canonical - target).abs() * 4,
        )
        _save_rgb(
            output / f"{stem}_conditioned_error_x4.png",
            (conditioned - target).abs() * 4,
        )
        _save_scalar(
            output / f"{stem}_canonical_volume_alpha.png",
            canonical_package.volume_alpha,
        )
        _save_scalar(
            output / f"{stem}_conditioned_volume_alpha.png",
            conditioned_package.volume_alpha,
        )
    return rows


def main():
    args, dataset, pipe = _parse_args()
    output = Path(dataset.model_path).resolve()
    resume_payload = _load(args.resume.resolve()) if args.resume else None
    initial_payload = (
        _load(args.initial_layered_state.resolve())
        if args.initial_layered_state
        else None
    )
    if output.exists() and any(output.iterdir()) and resume_payload is None:
        raise FileExistsError(f"Refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    structural = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, structural, shuffle=False)
    structural.load_ply(str(args.structural_ply.resolve()))
    views = scene.getTrainCameras()
    background = torch.ones(3, device="cuda")
    seed_payload = _load(args.foliage_seed_state.resolve())
    foliage = VolumetricFoliageModel(
        dataset.sh_degree, dynamic_rank=args.dynamic_rank
    ).cuda()
    if resume_payload is None and initial_payload is None:
        foliage.initialize_from_volume_state(seed_payload)
        dynamic_seed_count = _spawn_dynamic_layer(
            foliage, args.dynamic_seed_count
        )
        geometry_evidence_restore = {
            "needed": False,
            "matched": len(foliage),
            "unmatched": 0,
        }
    elif initial_payload is not None:
        if initial_payload.get("protocol") not in {
            PROTOCOL,
            LEGACY_PROTOCOL,
        }:
            raise RuntimeError("Initial layered state protocol mismatch")
        foliage.restore(initial_payload["foliage"])
        geometry_evidence_restore = _restore_geometry_evidence(
            foliage, seed_payload, initial_payload["foliage"]
        )
        dynamic_seed_count = int(foliage.dynamic_leaf_mask.sum())
    else:
        if resume_payload.get("protocol") != PROTOCOL:
            raise RuntimeError("Resume checkpoint protocol mismatch")
        foliage.restore(resume_payload["foliage"])
        geometry_evidence_restore = _restore_geometry_evidence(
            foliage, seed_payload, resume_payload["foliage"]
        )
        dynamic_seed_count = int(resume_payload["dynamic_seed_count"])
    audit = _load(args.replacement_audit.resolve())
    if audit.get("version") != RESPONSIBILITY_VERSION:
        raise RuntimeError("Replacement audit must use per-view rigid-safe v2")
    candidates = audit["candidate_indices"].cuda().long()
    def restored_vector(
        resume_name,
        initial_name,
        default,
    ):
        if resume_payload is not None:
            return resume_payload[resume_name].to(device="cuda")
        if initial_payload is None or initial_name not in initial_payload:
            return default
        return _remap_candidate_values(
            candidates,
            initial_payload["legacy_candidate_indices"].long(),
            initial_payload[initial_name],
            default,
        )

    if resume_payload is not None and "gate_atlas" in resume_payload:
        initial_gate_values = resume_payload["gate_atlas"].to(
            device="cuda"
        ).mean(dim=(1, 2))
    else:
        initial_gate_values = restored_vector(
            "gates",
            "legacy_candidate_gates",
            torch.ones(len(candidates), device="cuda"),
        )
    if args.certified_initial_gates:
        if initial_payload is None:
            raise RuntimeError(
                "--certified-initial-gates requires --initial-layered-state"
            )
        certified = {
            int(value)
            for value in initial_payload["retirement_eligible_indices"]
            .cpu()
            .tolist()
        }
        certified_mask = torch.as_tensor(
            [
                int(candidate) in certified
                for candidate in candidates.cpu().tolist()
            ],
            device="cuda",
            dtype=torch.bool,
        )
        initial_gate_values[~certified_mask] = 1.0
    atlas_shape = (
        len(candidates),
        args.uv_gate_size,
        args.uv_gate_size,
    )
    if resume_payload is not None:
        restored_atlas = resume_payload["gate_atlas"].to(device="cuda")
        if tuple(restored_atlas.shape) != atlas_shape:
            raise RuntimeError(
                "Resume UV gate atlas shape does not match --uv-gate-size"
            )
    elif (
        initial_payload is not None
        and "surfel_uv_gate_atlas" in initial_payload
    ):
        old_atlas = initial_payload["surfel_uv_gate_atlas"]
        if tuple(old_atlas.shape[1:]) != atlas_shape[1:]:
            old_atlas = torch.nn.functional.interpolate(
                old_atlas[:, None].float(),
                size=atlas_shape[1:],
                mode="bilinear",
                align_corners=True,
            )[:, 0]
        restored_atlas = _remap_candidate_values(
            candidates,
            initial_payload["legacy_candidate_indices"].long(),
            old_atlas,
            torch.ones(atlas_shape, device="cuda"),
        )
    else:
        restored_atlas = initial_gate_values[:, None, None].expand(
            atlas_shape
        ).clone()
    gate_atlas = torch.nn.Parameter(restored_atlas)
    surface_atlas_indices = _surface_atlas_indices(structural, candidates)
    scalar_surface_gate = torch.ones(
        structural.get_xyz.shape[0], device="cuda"
    )
    local_confidence = restored_vector(
        "local_confidence",
        "surfel_uv_replacement_confidence",
        torch.zeros(atlas_shape, device="cuda"),
    )
    local_observations = restored_vector(
        "local_observations",
        "surfel_uv_replacement_observations",
        torch.zeros(atlas_shape, dtype=torch.int16, device="cuda"),
    )
    local_sequence_bits = restored_vector(
        "local_sequence_bits",
        "surfel_uv_replacement_sequence_bits",
        torch.zeros(atlas_shape, dtype=torch.int64, device="cuda"),
    )
    sequence_support = restored_vector(
        "sequence_support",
        "surfel_uv_replacement_sequence_support",
        torch.zeros(atlas_shape, dtype=torch.int16, device="cuda"),
    )
    if initial_payload is not None and not bool(sequence_support.any()):
        flat_bits = local_sequence_bits.detach().cpu().reshape(-1).tolist()
        sequence_support.copy_(
            torch.as_tensor(
                [bin(int(value)).count("1") for value in flat_bits],
                device="cuda",
                dtype=torch.int16,
            ).reshape_as(sequence_support)
        )
    low_gate_steps = restored_vector(
        "low_gate_steps",
        "surfel_uv_replacement_low_gate_steps",
        torch.zeros(atlas_shape, dtype=torch.int32, device="cuda"),
    )
    evidence_sum = restored_vector(
        "evidence_sum",
        "surfel_uv_evidence_sum",
        torch.zeros((*atlas_shape, 4), device="cuda"),
    )
    evidence_weight = restored_vector(
        "evidence_weight",
        "surfel_uv_evidence_weight",
        torch.zeros(atlas_shape, device="cuda"),
    )
    if args.reset_initial_replacement_proof:
        with torch.no_grad():
            gate_atlas.fill_(1.0)
            local_confidence.zero_()
            local_observations.zero_()
            local_sequence_bits.zero_()
            sequence_support.zero_()
            low_gate_steps.zero_()
            evidence_sum.zero_()
            evidence_weight.zero_()

    sky = CanonicalDirectionalSky.load(args.sky_model.resolve(), device="cuda")
    for parameter in sky.parameters():
        parameter.requires_grad_(False)
    color = load_per_image_affine_color_correction(
        args.color_correction.resolve(), device="cuda"
    )
    for parameter in color.parameters():
        parameter.requires_grad_(False)
    fields = OutdoorTaskFieldLookup(
        Path(dataset.source_path),
        args.tree_mask_pickle,
        args.task_semantic_manifest,
        max_cached_views=0,
    )
    appearance = OutdoorAppearanceUncertainty(
        [view.image_name for view in views],
        rank=args.dynamic_rank,
        spatial_grid_size=24,
        device="cuda",
    )
    if resume_payload is not None:
        appearance.restore(resume_payload["appearance"])
    elif initial_payload is not None:
        appearance.restore(initial_payload["appearance"])
    schedule, schedule_audit = _make_schedule(
        views, foliage, args.iterations, args.seed + 1
    )
    optimizer = _optimizer(args, foliage, gate_atlas, appearance)
    if resume_payload is None:
        topology_stats = _reset_topology_statistics(foliage)
        topology_events = []
        start_step = 0
    else:
        optimizer.load_state_dict(resume_payload["optimizer"])
        topology_stats = {
            name: value.to(device="cuda")
            for name, value in resume_payload["topology_stats"].items()
        }
        topology_events = list(resume_payload["topology_events"])
        start_step = int(resume_payload["iteration"])
        random.setstate(resume_payload["python_rng_state"])
        np.random.set_state(resume_payload["numpy_rng_state"])
        torch.set_rng_state(resume_payload["torch_rng_state"])
        cuda_rng_state = resume_payload["cuda_rng_state"]
        if len(cuda_rng_state) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_rng_state)
        elif torch.cuda.device_count() == 1 and cuda_rng_state:
            # Checkpoints may have been written while all host GPUs were
            # visible and resumed with CUDA_VISIBLE_DEVICES restricted to the
            # original training GPU.  Restore that stream without requiring
            # unrelated devices to remain visible.
            torch.cuda.set_rng_state(cuda_rng_state[0], device=0)
        else:
            raise RuntimeError(
                "CUDA RNG checkpoint/device-count mismatch: "
                f"{len(cuda_rng_state)} saved, "
                f"{torch.cuda.device_count()} currently visible"
            )
    sequence_names = sorted({sequence_id(view.image_name) for view in views})
    sequence_bits = {
        name: 1 << index for index, name in enumerate(sequence_names)
    }
    if len(sequence_bits) > 62:
        raise RuntimeError("Local replacement sequence bitset exceeds 62 sequences")

    all_one_gate = torch.ones_like(scalar_surface_gate)
    eval_indices = [
        int(value) for value in args.eval_indices.split(",") if value.strip()
    ]
    if resume_payload is None:
        before = _evaluate_layered(
            views,
            eval_indices,
            structural,
            VolumetricFoliageModel(
                dataset.sh_degree, dynamic_rank=args.dynamic_rank
            ).cuda(),
            sky,
            color,
            background,
            torch.ones_like(structural.get_opacity.reshape(-1)),
            appearance,
            fields,
            output / "visualization" / "before",
        )
        # Zero-volume identity remains a hard executable contract.
        identity = []
        empty = VolumetricFoliageModel(
            dataset.sh_degree, dynamic_rank=args.dynamic_rank
        ).cuda()
        for index in (0, len(views) // 2, len(views) - 1):
            native = render_2dgs(views[index], structural, pipe, background)
            mixed = render_hybrid(
                views[index], structural, empty, background=background
            )
            identity.append(
                {
                    "index": index,
                    "rgb_max_abs": float(
                        (native["render"] - mixed.render).abs().max()
                    ),
                    "alpha_max_abs": float(
                        (native["rend_alpha"] - mixed.alpha).abs().max()
                    ),
                }
            )
    else:
        before = resume_payload["before"]
        identity = resume_payload["identity"]
    if any(row["rgb_max_abs"] > 2e-5 for row in identity):
        raise RuntimeError("Layered v6 failed zero-volume identity")

    trace_path = output / "training_trace.jsonl"
    pretrain_end = int(args.iterations * args.pretrain_fraction)
    retirement_start = int(args.iterations * args.retirement_fraction)
    densify_end = int(args.iterations * args.densify_until_fraction)
    resumed_elapsed = (
        float(resume_payload.get("elapsed_sec", 0.0))
        if resume_payload is not None
        else 0.0
    )
    start = time.time() - resumed_elapsed
    # All restart tensors have now been restored onto their live owners.
    # Releasing the CPU checkpoint avoids retaining a second model/optimizer
    # copy for the rest of a long resumed run.
    del resume_payload, initial_payload
    progress = tqdm(
        range(start_step, args.iterations),
        initial=start_step,
        total=args.iterations,
        desc="layered foliage v6",
    )
    for step in progress:
        view = views[int(schedule[step])]
        task = fields.fields(
            view.image_name,
            (view.image_height, view.image_width),
            torch.device("cuda"),
        )
        task["image_name"] = str(view.image_name)
        target = view.original_image.cuda()
        render_gates = _gate_vector(
            structural, candidates, torch.ones(len(candidates), device="cuda")
        )
        # Rendering consumes a stop-gradient ownership decision.  Atlas
        # updates come only from certified local replace-and-retire evidence,
        # never from an unconstrained photometric shortcut.
        render_atlas = gate_atlas.detach().clamp(0.0, 1.0)
        evidence_average = evidence_sum / evidence_weight.clamp_min(
            1e-8
        )[..., None]
        exposure_safe = (
            (evidence_average[..., 0] >= 0.45)
            & (evidence_average[..., 1] <= 0.08)
            & (evidence_average[..., 2] >= 0.05)
            & (local_observations >= 1)
        )
        code = appearance.temporal_code(view.image_name)
        canonical_package = render_hybrid(
            view,
            structural,
            foliage,
            background=background,
            surface_gate=render_gates,
            surface_gate_indices=surface_atlas_indices,
            surface_gate_atlas=render_atlas,
            include_dynamic=False,
        )
        canonical = apply_per_image_affine_color_correction(
            color,
            composite_white_background(
                canonical_package.render,
                canonical_package.alpha,
                sky(view),
            ),
            view.image_name,
        )
        canonical_photo = _loss(
            canonical,
            target,
            0.35 * task["p_canopy_core"] + task["p_rigid"],
        )
        rigid_loss = _loss(canonical, target, task["p_rigid"])

        # Tree masks are weak ray-hit evidence, never a fixed opaque target.
        alpha = canonical_package.volume_alpha[0].clamp(1e-5, 1 - 1e-5)
        positive = task["p_canopy_core"] * (1.0 - task["p_transient"])
        negative = (
            task["p_rigid"]
            * (1.0 - task["p_boundary_uncertain"])
            + 0.35 * task["p_sky"] * (1.0 - task["p_boundary_uncertain"])
        ).clamp(0, 1)
        occupancy_positive = (
            (-torch.log(alpha) * positive)
        ).sum() / positive.sum().clamp_min(1)
        silhouette_floor = (
            torch.relu(0.08 - alpha) * positive
        ).sum() / positive.sum().clamp_min(1)
        occupancy_negative = (
            -torch.log1p(-alpha) * negative
        ).sum() / negative.sum().clamp_min(1)
        occupancy_loss = (
            0.15 * occupancy_positive
            + silhouette_floor
            + occupancy_negative
        )

        delta = foliage.xyz - foliage.initialization_center
        position_variance = foliage.position_covariance.diagonal(
            dim1=-2, dim2=-1
        ).clamp_min(1e-6)
        position_nll = (delta.square() / position_variance).sum(-1)
        position_weight = torch.where(
            foliage.static_skeleton_mask,
            position_nll.new_tensor(2.0),
            position_nll.new_tensor(0.25),
        )
        finite_ray_nll = torch.nan_to_num(
            foliage.ray_depth_nll, nan=0.0, posinf=10.0
        ).clamp(0, 10)
        position_weight = position_weight * (
            1.0 + finite_ray_nll / 5.0
        )
        depth_prior = (position_nll * position_weight).mean()
        metadata_support = foliage.support_view_count.float()
        metadata_free = foliage.free_space_violation_count.float()
        free_space_rate = (
            metadata_free
            / (metadata_support + metadata_free).clamp_min(1)
        ).clamp(0, 1)
        free_space_loss = (
            foliage.opacities * free_space_rate
        ).mean()
        dynamic_regularizer = (
            foliage.deformation_basis[foliage.dynamic_leaf_mask]
            .square()
            .mean()
            if bool(foliage.dynamic_leaf_mask.any())
            else foliage.xyz.new_zeros(())
        )
        canonical_loss = (
            args.canonical_weight * canonical_photo
            + args.rigid_weight * rigid_loss
            + args.occupancy_weight * occupancy_loss
            + args.ray_hit_weight * occupancy_positive
            + args.free_space_weight * free_space_loss
            + args.depth_weight * depth_prior
            + args.dynamic_regularization * dynamic_regularizer
        )
        canonical_loss.backward()
        del canonical_package, canonical, alpha
        torch.cuda.empty_cache()

        conditioned_package = render_hybrid(
            view,
            structural,
            foliage,
            background=background,
            surface_gate=render_gates,
            surface_gate_indices=surface_atlas_indices,
            surface_gate_atlas=render_atlas,
            temporal_code=code,
            include_dynamic=True,
            audit_fields=torch.stack(
                [
                    task["p_canopy_core"],
                    task["p_rigid"],
                    task["p_canopy_core"],
                ],
                dim=0,
            ),
        )
        conditioned_base = apply_per_image_affine_color_correction(
            color,
            composite_white_background(
                conditioned_package.render,
                conditioned_package.alpha,
                sky(view),
            ),
            view.image_name,
        )
        conditioned = appearance(conditioned_base, view, task)
        conditioned_photo = _loss(
            conditioned,
            target,
            task["p_canopy_core"] + 0.25 * task["p_rigid"],
        )
        appearance_loss = appearance.heteroscedastic_loss(
            conditioned, target, task
        )
        conditioned_loss = (
            args.conditioned_weight * conditioned_photo
            + args.appearance_weight * appearance_loss
            + 1e-3 * appearance.regularization()
        )
        conditioned_loss.backward()
        _accumulate_topology(
            topology_stats,
            conditioned_package,
            conditioned_package.responsibility,
        )
        conditioned_surface_alpha = (
            conditioned_package.surface_alpha[0].detach()
        )
        conditioned_volume_depth = (
            conditioned_package.volume_depth[0].detach()
        )
        conditioned_surface_depth = (
            conditioned_package.surface_depth[0].detach()
        )
        normal_error = (conditioned.detach() - target).abs().mean(0)
        del conditioned_package, conditioned_base, conditioned
        torch.cuda.empty_cache()

        exposure_prediction_detached = None
        exposure_loss = canonical_photo.new_zeros(())
        exposure_region = positive.new_zeros(positive.shape)
        exposure_start = int(
            args.iterations * args.exposure_start_fraction
        )
        if step >= exposure_start and bool(exposure_safe.any()):
            # Differentiable replacement exposure: the old branch and UV
            # decision are stop-grad, while the newly visible volume remains
            # fully differentiable and receives the reconstruction gradient.
            exposure_atlas = torch.where(
                exposure_safe,
                render_atlas.new_tensor(args.counterfactual_gate),
                render_atlas.detach(),
            ).detach()
            exposure_package = render_hybrid(
                view,
                structural,
                foliage,
                background=background,
                surface_gate=render_gates.detach(),
                surface_gate_indices=surface_atlas_indices,
                surface_gate_atlas=exposure_atlas,
                # The conditioned pass has already consumed and freed its
                # normalized-code graph; exposure owns an independent graph
                # while accumulating into the same leaf parameters.
                temporal_code=appearance.temporal_code(view.image_name),
                include_dynamic=True,
            )
            exposure_prediction = apply_per_image_affine_color_correction(
                color,
                composite_white_background(
                    exposure_package.render,
                    exposure_package.alpha,
                    sky(view),
                ),
                view.image_name,
            )
            exposure_region = (
                (
                    conditioned_surface_alpha
                    - exposure_package.surface_alpha[0].detach()
                ).clamp_min(0)
                * positive
            )
            exposure_loss = _loss(
                exposure_prediction, target, exposure_region
            )
            (args.exposure_weight * exposure_loss).backward()
            exposure_prediction_detached = exposure_prediction.detach()
            del exposure_package, exposure_prediction
            torch.cuda.empty_cache()
        local_q = local_confidence.detach()
        if step % args.local_audit_every == 0:
            # Independent ray/depth posterior: initialization centres come
            # from SfM tracks / visual-hull rays, not the old surfel depth.
            with torch.no_grad():
                posterior = render_hybrid(
                    view,
                    structural,
                    foliage,
                    background=background,
                    surface_gate=torch.zeros_like(render_gates),
                    volume_means_override=foliage.initialization_center,
                    include_dynamic=False,
                )
            posterior_alpha = posterior.volume_alpha[0].detach()
            posterior_depth = posterior.volume_depth[0].detach()
            sigma = float(args.posterior_depth_sigma)
            new_depth_match = torch.exp(
                -(
                    conditioned_volume_depth - posterior_depth
                ).abs()
                / sigma
            )
            old_depth_match = torch.exp(
                -(
                    conditioned_surface_depth - posterior_depth
                ).abs()
                / sigma
            )
            posterior_valid = (
                (posterior_alpha >= 0.03)
                & (posterior_depth > 0)
            ).to(posterior_alpha.dtype)
            replacement_depth_proof = (
                new_depth_match
                * (1.0 - old_depth_match)
                * posterior_valid
            ).clamp(0, 1)
            if exposure_prediction_detached is None:
                removal_improvement = normal_error.new_full(
                    normal_error.shape, 0.5
                )
            else:
                exposure_error = (
                    exposure_prediction_detached - target
                ).abs().mean(0)
                removal_improvement = torch.sigmoid(
                    (normal_error - exposure_error) / 0.02
                )
            with torch.no_grad():
                local_audit = render_hybrid(
                    view,
                    structural,
                    foliage,
                    background=background,
                    surface_gate=render_gates.detach(),
                    surface_gate_indices=surface_atlas_indices,
                    surface_gate_atlas=render_atlas.detach(),
                    audit_fields=torch.stack(
                        [
                            task["p_canopy_core"],
                            task["p_rigid"],
                            replacement_depth_proof,
                            removal_improvement,
                        ],
                        dim=0,
                    ),
                ).gate_responsibility
            if local_audit is not None:
                total = local_audit[..., 0].clamp_min(1e-8)
                per_view = local_audit[..., 1:5] / total[..., None]
                canopy = per_view[..., 0]
                rigid = per_view[..., 1]
                depth_proof = per_view[..., 2]
                improvement = per_view[..., 3]
                view_q = (
                    canopy
                    * (1.0 - rigid)
                    * depth_proof
                    * improvement
                ).clamp(0, 1)
                observed = local_audit[..., 0] > 0.02
                with torch.no_grad():
                    evidence_sum.add_(local_audit[..., 1:5])
                    evidence_weight.add_(local_audit[..., 0])
                    # Confidence is a retained lower-cost local proof, not a
                    # per-view average.  The previous EMA needed dozens of
                    # repeat observations before even a very strong candidate
                    # could unlock, while the support-camera schedule only
                    # revisits a particular primitive a few times per epoch.
                    # Multi-view/sequence counters below remain the safeguards
                    # against a one-frame false positive.
                    local_confidence[observed] = torch.maximum(
                        0.995 * local_confidence[observed],
                        view_q[observed],
                    )
                    proven = observed & (view_q >= args.local_confidence_threshold)
                    local_observations[proven] += 1
                    bit = sequence_bits[sequence_id(view.image_name)]
                    local_sequence_bits[proven] |= bit
                    sequence_support.zero_()
                    for sequence_bit in sequence_bits.values():
                        sequence_support.add_(
                            (
                                (local_sequence_bits & sequence_bit) != 0
                            ).to(torch.int16)
                        )
                local_q = local_confidence.detach()
        evidence_average = evidence_sum / evidence_weight.clamp_min(
            1e-8
        )[..., None]
        local_unlocked = (
            (local_q >= args.local_confidence_threshold)
            & (local_observations >= 3)
            & (sequence_support >= 2)
            & (evidence_average[..., 1] <= 0.05)
        )
        gate_trainable = local_unlocked & (
            local_q >= args.retirement_confidence
        )
        gate_values = gate_atlas.clamp(0, 1)
        retire = (
            local_q * gate_values
            + 0.15 * (1 - local_q) * (1 - gate_values)
        )
        local_gate_loss = (
            retire[gate_trainable].mean()
            if bool(gate_trainable.any())
            else gate_values.new_zeros(())
        )
        local_gate_loss = (
            local_gate_loss
            + 0.05 * _uv_total_variation(
                gate_values, gate_trainable
            )
        )
        gate_objective = (
            args.gate_weight * local_gate_loss
            if step >= pretrain_end
            else local_gate_loss.new_zeros(())
        )
        if gate_objective.requires_grad:
            gate_objective.backward()
        loss = (
            canonical_loss.detach()
            + conditioned_loss.detach()
            + args.exposure_weight * exposure_loss.detach()
            + gate_objective.detach()
        )
        if gate_atlas.grad is not None:
            gate_atlas.grad[~gate_trainable] = 0
        _apply_role_gradient_policy(foliage)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            foliage.quaternions.copy_(
                torch.nn.functional.normalize(foliage.quaternions, dim=-1)
            )
            foliage.opacity_logits.clamp_(-10.0, 1.5)
            gate_atlas.clamp_(0, 1)
            # A rigid-contaminated texel is an absolute ownership lock.
            rigid_locked = (
                evidence_weight > 0
            ) & (evidence_average[..., 1] > 0.05)
            gate_atlas[rigid_locked] = 1.0
            if step >= retirement_start:
                low_gate_steps += (
                    (gate_atlas < 0.05)
                    & (local_q >= args.retirement_confidence)
                    & local_unlocked
                ).to(torch.int32)

        topology_event = None
        if (
            step + 1 >= args.densify_start
            and step + 1 <= densify_end
            and (step + 1) % args.densify_every == 0
        ):
            topology_event = _adaptive_topology(
                args, foliage, topology_stats
            )
            new_to_old = topology_event.pop("_new_to_old", None)
            topology_event["iteration"] = step + 1
            topology_events.append(topology_event)
            if new_to_old is not None:
                optimizer = _optimizer_with_migrated_state(
                    args,
                    foliage,
                    gate_atlas,
                    appearance,
                    optimizer,
                    new_to_old,
                )
            topology_stats = _reset_topology_statistics(foliage)

        if step == 0 or (step + 1) % args.log_every == 0:
            row = {
                "iteration": step + 1,
                "loss": float(loss),
                "canonical_photo": float(canonical_photo),
                "conditioned_photo": float(conditioned_photo),
                "occupancy": float(occupancy_loss),
                "depth_prior": float(depth_prior),
                "local_gate_loss": float(local_gate_loss),
                "gaussians": len(foliage),
                "skeleton": int(foliage.static_skeleton_mask.sum()),
                "canonical_crown": int(foliage.canonical_crown_mask.sum()),
                "dynamic_leaf": int(foliage.dynamic_leaf_mask.sum()),
                "local_unlocked": int(local_unlocked.sum()),
                "local_gate_trainable": int(gate_trainable.sum()),
                "uv_texels_below_0_5": int(
                    (gate_atlas < 0.5).sum()
                ),
                "uv_candidates_changed": int(
                    (_atlas_candidate_means(gate_atlas) < 0.99).sum()
                ),
                "exposure_safe_texels": int(exposure_safe.sum()),
                "exposure_loss": float(exposure_loss),
                "local_confidence_mean": float(local_confidence.mean()),
                "local_confidence_p95": float(
                    torch.quantile(local_confidence, 0.95)
                ),
                "local_confidence_max": float(local_confidence.max()),
                "local_observed_three_views": int(
                    (local_observations >= 3).sum()
                ),
                "local_supported_two_sequences": int(
                    (sequence_support >= 2).sum()
                ),
                "topology_event": topology_event,
                "elapsed_sec": time.time() - start,
            }
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            progress.set_postfix(
                loss=f"{float(loss):.4f}",
                n=len(foliage),
                unlocked=int(local_unlocked.sum()),
            )
        if (
            args.checkpoint_every > 0
            and (step + 1) % args.checkpoint_every == 0
        ):
            _save_checkpoint(
                output / "training_checkpoint.pth",
                {
                    "protocol": PROTOCOL,
                    "iteration": step + 1,
                    "elapsed_sec": time.time() - start,
                    "dynamic_seed_count": dynamic_seed_count,
                    "foliage": foliage.capture(),
                    "appearance": appearance.capture(),
                    "gate_atlas": gate_atlas.detach(),
                    "local_confidence": local_confidence,
                    "local_observations": local_observations,
                    "local_sequence_bits": local_sequence_bits,
                    "sequence_support": sequence_support,
                    "low_gate_steps": low_gate_steps,
                    "evidence_sum": evidence_sum,
                    "evidence_weight": evidence_weight,
                    "optimizer": optimizer.state_dict(),
                    "topology_stats": topology_stats,
                    "topology_events": topology_events,
                    "before": before,
                    "identity": identity,
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all(),
                },
            )
        # Release every remaining graph-facing scalar/map before the next
        # giant-footprint view requests its tile buffer.
        del (
            canonical_photo,
            conditioned_photo,
            rigid_loss,
            appearance_loss,
            canonical_loss,
            conditioned_loss,
            loss,
            conditioned_surface_alpha,
            conditioned_volume_depth,
            conditioned_surface_depth,
            normal_error,
            exposure_prediction_detached,
        )
        if step % args.local_audit_every == 0:
            del posterior, local_audit
        torch.cuda.empty_cache()

    # The mixed hard views can require several additional GiB.  Training
    # optimizer moments and the last autograd render packages are no longer
    # needed; release them before final no-grad evaluation to avoid allocator
    # fragmentation/OOM after densification.
    del optimizer, topology_stats
    gc.collect()
    torch.cuda.empty_cache()
    final_gate = torch.ones_like(scalar_surface_gate)
    after = _evaluate_layered(
        views,
        eval_indices,
        structural,
        foliage,
        sky,
        color,
        background,
        final_gate,
        appearance,
        fields,
        output / "visualization" / "after",
        surface_atlas_indices,
        gate_atlas.detach(),
    )
    paired = []
    for initial, final in zip(before, after):
        paired.append(
            {
                **final,
                "delta_canonical_psnr": (
                    final["canonical_psnr"] - initial["canonical_psnr"]
                ),
                "delta_conditioned_psnr": (
                    final["conditioned_psnr"] - initial["conditioned_psnr"]
                ),
                "delta_canonical_canopy_psnr": (
                    final["canonical_canopy"]["psnr"]
                    - initial["canonical_canopy"]["psnr"]
                ),
                "delta_conditioned_canopy_psnr": (
                    final["conditioned_canopy"]["psnr"]
                    - initial["conditioned_canopy"]["psnr"]
                ),
                "delta_rigid_psnr": (
                    final["rigid"]["psnr"] - initial["rigid"]["psnr"]
                ),
            }
        )
    retirement_texels = (
        (gate_atlas.detach() < 0.05)
        & (local_confidence >= args.retirement_confidence)
        & (low_gate_steps >= max(20, int(0.02 * args.iterations)))
    )
    retirement_eligible = retirement_texels.float().mean(
        dim=(1, 2)
    ) >= 0.80
    state_path = output / "layered_foliage_state.pth"
    torch.save(
        {
            "protocol": PROTOCOL,
            "foliage": foliage.capture(),
            "appearance": appearance.capture(),
            "legacy_candidate_indices": candidates,
            "legacy_candidate_gates": _atlas_candidate_means(gate_atlas),
            "surfel_uv_gate_atlas": gate_atlas.detach(),
            "surfel_uv_replacement_confidence": local_confidence,
            "surfel_uv_replacement_observations": local_observations,
            "surfel_uv_replacement_sequence_bits": local_sequence_bits,
            "surfel_uv_replacement_sequence_support": sequence_support,
            "surfel_uv_replacement_low_gate_steps": low_gate_steps,
            "surfel_uv_evidence_sum": evidence_sum,
            "surfel_uv_evidence_weight": evidence_weight,
            "retirement_eligible_indices": candidates[retirement_eligible],
        },
        state_path,
    )
    hard_pass = all(
        row["delta_canonical_psnr"] >= 0
        and row["delta_rigid_psnr"] >= -0.01
        and row["delta_conditioned_canopy_psnr"] > 0
        for row in paired
    )
    audit_statistics = audit.get("statistics", {})
    candidate_indices_cpu = candidates.detach().cpu()
    candidate_radius = audit_statistics.get("maximum_projected_radius")
    candidate_giant_views = audit_statistics.get("giant_radius_view_count")
    footprint_audit = {}
    if candidate_radius is not None and candidate_giant_views is not None:
        candidate_radius = candidate_radius[candidate_indices_cpu].float()
        candidate_giant = (
            candidate_giant_views[candidate_indices_cpu] > 0
        )
        retired_cpu = retirement_eligible.detach().cpu()
        retired_radius = candidate_radius[retired_cpu]
        footprint_audit = {
            "giant_candidate_count": int(candidate_giant.sum()),
            "giant_retired_count": int(
                (candidate_giant & retired_cpu).sum()
            ),
            "candidate_radius_median": float(candidate_radius.median()),
            "candidate_radius_p95": float(
                torch.quantile(candidate_radius, 0.95)
            ),
            "retired_radius_median": (
                float(retired_radius.median())
                if retired_radius.numel()
                else None
            ),
            "retired_radius_p95": (
                float(torch.quantile(retired_radius, 0.95))
                if retired_radius.numel()
                else None
            ),
        }
    result = {
        "protocol": PROTOCOL,
        "mainline_eligible": bool(hard_pass),
        "iterations": args.iterations,
        "zero_volume_identity": identity,
        "schedule": schedule_audit,
        "geometry_version": seed_payload.get("geometry_version"),
        "geometry_evidence_restore": geometry_evidence_restore,
        "initial_dynamic_leaf_count": dynamic_seed_count,
        "final_gaussians": len(foliage),
        "layer_counts": {
            "static_skeleton": int(foliage.static_skeleton_mask.sum()),
            "canonical_crown": int(foliage.canonical_crown_mask.sum()),
            "dynamic_leaf": int(foliage.dynamic_leaf_mask.sum()),
        },
        "topology_events": topology_events,
        "local_replacement": {
            "proof_definition": (
                "per_texel_canopy_times_independent_new_depth_correct_"
                "times_old_depth_wrong_times_exposure_improvement"
            ),
            "initial_proof_reset": bool(
                args.reset_initial_replacement_proof
            ),
            "candidate_count": len(candidates),
            "unlocked": int(
                (
                    (local_confidence >= args.local_confidence_threshold)
                    & (local_observations >= 3)
                    & (sequence_support >= 2)
                ).sum()
            ),
            "confidence_mean": float(local_confidence.mean()),
            "confidence_p95": float(
                torch.quantile(local_confidence, 0.95)
            ),
            "gate_mean": float(gate_atlas.mean()),
            "uv_texels_below_0_5": int((gate_atlas < 0.5).sum()),
            "uv_texels_below_0_05": int((gate_atlas < 0.05).sum()),
            "uv_candidates_changed": int(
                (_atlas_candidate_means(gate_atlas) < 0.99).sum()
            ),
            "retirement_eligible": int(retirement_eligible.sum()),
            "footprint_audit": footprint_audit,
        },
        "targeted_evaluation": paired,
        "state": str(state_path),
        "elapsed_sec": time.time() - start,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
