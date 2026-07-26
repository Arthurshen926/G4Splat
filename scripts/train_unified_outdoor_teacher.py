#!/usr/bin/env python
"""Train one from-scratch mixed outdoor teacher from unified evidence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SURFEL_ROOT = REPO_ROOT / "2d-gaussian-splatting"
sys.path[:0] = [str(REPO_ROOT), str(SURFEL_ROOT)]

from arguments import (  # noqa: E402
    ModelParams,
    OptimizationParams,
    PipelineParams,
)
from outdoor.appearance_uncertainty import (  # noqa: E402
    OutdoorAppearanceUncertainty,
)
from outdoor.directional_sky import (  # noqa: E402
    CanonicalDirectionalSky,
    composite_white_background,
)
from outdoor.evidence_store import load_evidence_store  # noqa: E402
from outdoor.foliage_view_graph import sequence_id  # noqa: E402
from outdoor.hybrid_gaussian_renderer import (  # noqa: E402
    LAYER_CANONICAL_CROWN,
    LAYER_DYNAMIC_LEAF,
    VolumetricFoliageModel,
    render_hybrid,
)
from outdoor.lazy_scene import LazyScene  # noqa: E402
from outdoor.task_fields import OutdoorTaskFieldLookup  # noqa: E402
from outdoor.training_evidence import OutdoorGeometryEvidence  # noqa: E402
from scene import GaussianModel  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402


PROTOCOL = "unified_outdoor_mixed_teacher_v1"
# Exact hashes of the checkpoint implementation that preceded the
# I/O/allocator-only fast path.  This narrow allow-list lets the active formal
# run resume without weakening the normal implementation/CUDA provenance
# guard for arbitrary code changes.
PERFORMANCE_RESUME_PREDECESSOR = {
    "trainer": "b662682037b4f67a1439a6867a1bf1925520871ebf84efff1602dd1a3bd75357",
    "lazy_scene": "eb1f67b0a0813b55d89933fa710d486e3515548510c2c8cd9811a4103c0f925b",
    "task_fields": "e654646d46ee172d32f96a8540173000327a470dd0bc9097d644ddffe8e155ea",
    # Older checkpoints did not hash the shared mask lookup separately.
    "mask_lookup": None,
}
TRAINING_PROFILES = {
    # Final quality schedule.  Keep this profile stable for the eventual
    # benchmark run.
    "quality": {
        "iterations": 80_000,
        "phases": (
            ("canonical_bootstrap", 0.20),
            ("topology", 0.60),
            ("dynamic_appearance", 0.85),
            ("ownership_cleanup", 0.95),
            ("canonical_polish", 1.00),
        ),
    },
    # Reconstruction-quality validation schedule: the dynamic foliage branch
    # starts at 12k instead of 48k, while every phase still sees several full
    # passes over the 1,487 Cambridge training cameras.
    "fast": {
        "iterations": 30_000,
        "phases": (
            ("canonical_bootstrap", 0.20),
            ("topology", 0.40),
            ("dynamic_appearance", 0.73),
            ("ownership_cleanup", 0.90),
            ("canonical_polish", 1.00),
        ),
    },
}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.set_defaults(
        iterations=None,
        data_device="cpu",
        white_background=True,
        densify_until_iter=None,
        opacity_cull=0.004,
        lambda_dssim=0.20,
    )
    parser.add_argument("--evidence-store", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument(
        "--training-profile",
        choices=tuple(TRAINING_PROFILES),
        default="quality",
    )
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--dynamic-rank", type=int, default=4)
    parser.add_argument("--dynamic-seed-count", type=int, default=16_000)
    parser.add_argument("--volume-position-lr", type=float, default=8e-5)
    parser.add_argument("--volume-feature-lr", type=float, default=8e-4)
    parser.add_argument("--volume-opacity-lr", type=float, default=4e-3)
    parser.add_argument("--volume-scale-lr", type=float, default=4e-4)
    parser.add_argument("--volume-rotation-lr", type=float, default=2e-4)
    parser.add_argument("--dynamic-lr", type=float, default=3e-4)
    parser.add_argument("--appearance-lr", type=float, default=8e-4)
    parser.add_argument("--sky-lr", type=float, default=2e-3)
    parser.add_argument("--geometry-every", type=int, default=4)
    parser.add_argument("--topology-every", type=int, default=8)
    parser.add_argument("--replacement-every", type=int, default=20)
    parser.add_argument("--geometry-weight", type=float, default=0.08)
    parser.add_argument("--plane-weight", type=float, default=0.06)
    parser.add_argument("--normal-weight", type=float, default=0.025)
    parser.add_argument("--ordinal-weight", type=float, default=0.015)
    parser.add_argument("--ownership-weight", type=float, default=0.08)
    parser.add_argument("--occupancy-weight", type=float, default=0.06)
    parser.add_argument("--dynamic-weight", type=float, default=0.65)
    parser.add_argument("--appearance-weight", type=float, default=0.06)
    parser.add_argument("--high-frequency-weight", type=float, default=0.08)
    parser.add_argument(
        "--maximum-surface-gaussians",
        type=int,
        default=1_200_000,
    )
    parser.add_argument(
        "--maximum-surface-growth-per-event",
        type=int,
        default=20_000,
    )
    parser.add_argument("--maximum-volume-gaussians", type=int, default=300_000)
    parser.add_argument("--maximum-volume-splits", type=int, default=4000)
    parser.add_argument("--volume-split-radius", type=float, default=8.0)
    parser.add_argument("--volume-densify-every", type=int, default=600)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-indices", default="408,409,410")
    parser.add_argument("--view-cache-size", type=int, default=4)
    parser.add_argument("--image-prefetch-workers", type=int, default=2)
    parser.add_argument("--image-prefetch-depth", type=int, default=4)
    parser.add_argument("--maintenance-every", type=int, default=1000)
    parser.add_argument("--allow-performance-resume", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    profile = TRAINING_PROFILES[args.training_profile]
    if args.iterations is None:
        args.iterations = int(profile["iterations"])
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.maximum_surface_gaussians <= 0:
        parser.error("--maximum-surface-gaussians must be positive")
    if args.maximum_surface_growth_per_event <= 0:
        parser.error(
            "--maximum-surface-growth-per-event must be positive"
        )
    if args.image_prefetch_workers < 0:
        parser.error("--image-prefetch-workers must be non-negative")
    if args.image_prefetch_depth < 0:
        parser.error("--image-prefetch-depth must be non-negative")
    if args.maintenance_every < 0:
        parser.error("--maintenance-every must be non-negative")
    topology_end = next(
        end for name, end in profile["phases"] if name == "topology"
    )
    if args.densify_until_iter is None:
        args.densify_until_iter = int(args.iterations * topology_end)
    elif args.densify_until_iter > int(args.iterations * topology_end):
        args.densify_until_iter = int(args.iterations * topology_end)
    return (
        args,
        model.extract(args),
        optimization.extract(args),
        pipeline.extract(args),
    )


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _surface_capture_to_device(capture, device: torch.device):
    """Move a CPU-mapped 2DGS capture to its training device.

    Upstream ``GaussianModel.restore`` assumes ``torch.load`` retained CUDA
    tensors.  Unified checkpoints are intentionally loaded on CPU to bound
    restart memory, so make that implicit contract explicit before restore.
    Optimizer state is left in the state dict; ``Optimizer.load_state_dict``
    casts it to the parameter device.
    """
    values = list(capture)
    for index in range(1, 10):
        if torch.is_tensor(values[index]):
            source = values[index]
            moved = source.detach().to(device=device)
            values[index] = (
                torch.nn.Parameter(
                    moved, requires_grad=source.requires_grad
                )
                if isinstance(source, torch.nn.Parameter)
                else moved
            )
    if len(values) >= 13 and values[12] is not None:
        values[12] = {
            name: value.to(device=device)
            if torch.is_tensor(value)
            else value
            for name, value in values[12].items()
        }
    return tuple(values)


def _save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _phase(
    step: int, iterations: int, training_profile: str = "quality"
) -> str:
    progress = (step + 1) / float(iterations)
    phases = TRAINING_PROFILES[training_profile]["phases"]
    for name, end in phases:
        if progress <= end:
            return name
    return phases[-1][0]


def _full_epoch_schedule(count: int, iterations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = []
    while len(result) < iterations:
        result.extend(rng.permutation(count).tolist())
    return np.asarray(result[:iterations], dtype=np.int64)


def _cycle_schedule(
    indices: list[int], iterations: int, seed: int
) -> np.ndarray:
    if not indices:
        return np.full(iterations, -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    result = []
    values = np.asarray(indices, dtype=np.int64)
    while len(result) < iterations:
        result.extend(rng.permutation(values).tolist())
    return np.asarray(result[:iterations], dtype=np.int64)


def _schedule_digest(*schedules: np.ndarray) -> str:
    digest = hashlib.sha256()
    for schedule in schedules:
        digest.update(np.ascontiguousarray(schedule).view(np.uint8))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim == 2:
        weight = weight[None]
    return (value * weight).sum() / (
        weight.sum() * value.shape[0]
    ).clamp_min(1.0)


def _photo_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    lambda_dssim: float,
) -> torch.Tensor:
    l1 = _weighted_mean((prediction - target).abs(), weight)
    structural = 1.0 - ssim(
        prediction * weight[None],
        target * weight[None],
    )
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * structural


def _high_frequency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Masked gradient loss for the sequence-conditioned leaf branch."""
    horizontal_weight = torch.minimum(weight[:, 1:], weight[:, :-1])
    vertical_weight = torch.minimum(weight[1:, :], weight[:-1, :])
    horizontal = (
        (prediction[:, :, 1:] - prediction[:, :, :-1])
        - (target[:, :, 1:] - target[:, :, :-1])
    ).abs().mean(0)
    vertical = (
        (prediction[:, 1:, :] - prediction[:, :-1, :])
        - (target[:, 1:, :] - target[:, :-1, :])
    ).abs().mean(0)
    return (
        (horizontal * horizontal_weight).sum()
        / horizontal_weight.sum().clamp_min(1)
        + (vertical * vertical_weight).sum()
        / vertical_weight.sum().clamp_min(1)
    )


def _initialize_surface(
    surface: GaussianModel,
    surface_seed: Path,
    scene_extent: float,
    opt,
) -> dict:
    with np.load(surface_seed, allow_pickle=False) as seed:
        xyz = torch.from_numpy(seed["xyz"]).float().cuda()
        scales = torch.from_numpy(seed["scales"]).float().cuda()
        quaternions = torch.from_numpy(
            seed["quaternions"]
        ).float().cuda()
        rgb = torch.from_numpy(seed["rgb"]).float().cuda()
        opacity = torch.from_numpy(
            seed["initial_opacity"]
        ).float().cuda()
        source_type = torch.from_numpy(
            seed["source_type"]
        ).to(device="cuda", dtype=torch.int16)
        confidence = torch.from_numpy(
            seed["geometry_confidence"]
        ).float().cuda()
    surface.create_from_parameters(
        xyz, scales, quaternions, rgb, scene_extent
    )
    surface._opacity = torch.nn.Parameter(
        torch.logit(opacity[:, None].clamp(1e-6, 1 - 1e-6))
    )
    surface._source_type = source_type
    surface._geometry_confidence = confidence
    surface._primitive_class.fill_(GaussianModel.PRIMITIVE_STRUCTURAL)
    surface._protected_flag.zero_()
    surface._block_id.fill_(-1)
    surface.training_setup(opt)
    return {
        "surface_count": len(surface.get_xyz),
        "source_counts": {
            str(int(value)): int((source_type == value).sum())
            for value in source_type.unique().cpu()
        },
    }


def _volume_optimizer(args, foliage, appearance, sky):
    return torch.optim.Adam(
        [
            {
                "params": [foliage.xyz],
                "lr": args.volume_position_lr,
                "name": "xyz",
            },
            {
                "params": [foliage.features],
                "lr": args.volume_feature_lr,
                "name": "features",
            },
            {
                "params": [foliage.opacity_logits],
                "lr": args.volume_opacity_lr,
                "name": "opacity",
            },
            {
                "params": [foliage.log_scales],
                "lr": args.volume_scale_lr,
                "name": "scale",
            },
            {
                "params": [foliage.quaternions],
                "lr": args.volume_rotation_lr,
                "name": "rotation",
            },
            {
                "params": [
                    foliage.deformation_basis,
                    foliage.dynamic_feature_basis,
                    foliage.dynamic_opacity_basis,
                ],
                "lr": args.dynamic_lr,
                "name": "dynamic",
            },
            {
                "params": list(appearance.parameters()),
                "lr": args.appearance_lr,
                "name": "appearance",
            },
            {
                "params": list(sky.parameters()),
                "lr": args.sky_lr,
                "name": "sky",
            },
        ],
        eps=1e-15,
    )


def _migrate_volume_optimizer(
    args,
    foliage,
    appearance,
    sky,
    previous,
    new_to_old,
):
    current = _volume_optimizer(args, foliage, appearance, sky)
    previous_groups = {
        group["name"]: group for group in previous.param_groups
    }
    primitive = {
        "xyz",
        "features",
        "opacity",
        "scale",
        "rotation",
        "dynamic",
    }
    for group in current.param_groups:
        old = previous_groups.get(group["name"])
        if old is None or len(old["params"]) != len(group["params"]):
            continue
        for old_parameter, new_parameter in zip(
            old["params"], group["params"]
        ):
            state = previous.state.get(old_parameter)
            if not state:
                continue
            migrated = {}
            for key, value in state.items():
                if (
                    group["name"] in primitive
                    and torch.is_tensor(value)
                    and value.ndim > 0
                    and value.shape[0] == old_parameter.shape[0]
                ):
                    migrated[key] = value[new_to_old].clone()
                elif torch.is_tensor(value):
                    migrated[key] = value.clone()
                else:
                    migrated[key] = value
            current.state[new_parameter] = migrated
    return current


def _volume_stats(foliage) -> dict[str, torch.Tensor]:
    count = len(foliage)
    device = foliage.xyz.device
    return {
        "gradient": torch.zeros(count, device=device),
        "gradient_count": torch.zeros(count, device=device),
        "radius": torch.zeros(count, device=device),
        "contribution": torch.zeros(count, device=device),
        "residual": torch.zeros(count, device=device),
        "rigid": torch.zeros(count, device=device),
    }


@torch.no_grad()
def _accumulate_volume_stats(stats, package) -> None:
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
    if package.responsibility is not None:
        volume = package.responsibility[offset:]
        stats["contribution"] += volume[:, 0]
        if volume.shape[1] >= 4:
            stats["rigid"] += volume[:, 2]
            stats["residual"] += volume[:, 3]


@torch.no_grad()
def _adapt_volume(args, foliage, stats):
    count = len(foliage)
    gradient = stats["gradient"] / stats["gradient_count"].clamp_min(1)
    residual = stats["residual"] / stats["contribution"].clamp_min(1e-8)
    rigid = stats["rigid"] / stats["contribution"].clamp_min(1e-8)
    eligible = (
        ~foliage.static_skeleton_mask
        & (foliage.support_sequence_count >= 2)
        & (stats["radius"] >= args.volume_split_radius)
        & (stats["contribution"] > 0)
        & (rigid < 0.15)
    )
    capacity = min(
        args.maximum_volume_splits,
        max(args.maximum_volume_gaussians - count, 0),
    )
    if bool(eligible.any()) and capacity > 0:
        indices = torch.nonzero(eligible, as_tuple=False).flatten()
        score = gradient[indices] * residual[indices]
        selected = indices[
            torch.topk(score, min(capacity, len(indices))).indices
        ]
        event = foliage.split(selected)
        event["pruned"] = 0
        return event
    support = foliage.support_view_count.float().clamp_min(1)
    contradiction = (
        (
            foliage.free_space_violation_count.float()
            / support
        )
        >= 0.30
    ) | (foliage.occupancy_probability < 0.12)
    weak = (
        foliage.opacities < 0.002
    ) & (
        stats["contribution"]
        <= torch.quantile(stats["contribution"], 0.10)
    )
    pruned, new_to_old = foliage.prune(weak & contradiction)
    return {
        "split_parents": 0,
        "children": 0,
        "pruned": pruned,
        "_new_to_old": new_to_old if pruned else None,
    }


def _apply_volume_role_gradients(foliage, phase: str) -> None:
    skeleton = foliage.static_skeleton_mask
    dynamic = foliage.dynamic_leaf_mask
    if foliage.xyz.grad is not None:
        foliage.xyz.grad[skeleton] *= 0.08
        if phase != "canonical_bootstrap":
            foliage.xyz.grad[dynamic] *= 1.4
    if foliage.log_scales.grad is not None:
        foliage.log_scales.grad[skeleton] *= 0.05
    if foliage.opacity_logits.grad is not None:
        foliage.opacity_logits.grad[skeleton] *= 0.25
    for parameter in (
        foliage.deformation_basis,
        foliage.dynamic_feature_basis,
        foliage.dynamic_opacity_basis,
    ):
        if parameter.grad is not None:
            parameter.grad[~dynamic] = 0
            if phase not in {
                "dynamic_appearance",
                "ownership_cleanup",
            }:
                parameter.grad[dynamic] = 0


def _geometry_losses(
    package,
    evidence: dict[str, torch.Tensor],
    rigid: torch.Tensor,
    args,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = package.depth.new_zeros(())
    losses = {
        "chart": zero,
        "plane": zero,
        "normal": zero,
        "inverse": zero,
        "ordinal": zero,
    }
    predicted_depth = package.depth[0]
    valid_prediction_mask = (
        torch.isfinite(predicted_depth) & (predicted_depth > 0)
    )
    safe_predicted_depth = torch.where(
        valid_prediction_mask,
        predicted_depth,
        torch.ones_like(predicted_depth),
    )
    valid_prediction = valid_prediction_mask.float()
    if "chart_depth" in evidence:
        raw_target = evidence["chart_depth"][0]
        valid_target = torch.isfinite(raw_target) & (raw_target > 0)
        target = torch.where(
            valid_target, raw_target, torch.ones_like(raw_target)
        )
        weight = (
            evidence["chart_weight"][0]
            * rigid
            * valid_prediction
            * valid_target.float()
        )
        relative = (
            safe_predicted_depth - target
        ).abs() / target.clamp_min(0.1)
        losses["chart"] = (
            torch.log1p(relative) * weight
        ).sum() / weight.sum().clamp_min(1)
    if "plane_depth" in evidence:
        raw_target = evidence["plane_depth"][0]
        valid_target = torch.isfinite(raw_target) & (raw_target > 0)
        target = torch.where(
            valid_target, raw_target, torch.ones_like(raw_target)
        )
        weight = (
            evidence["plane_weight"][0]
            * rigid
            * valid_prediction
            * valid_target.float()
        )
        relative = (
            safe_predicted_depth - target
        ).abs() / target.clamp_min(0.1)
        losses["plane"] = (
            torch.log1p(relative) * weight
        ).sum() / weight.sum().clamp_min(1)
        if "plane_normal_world" in evidence:
            raw_predicted_normal = package.normal_world
            valid_predicted_normal = torch.isfinite(
                raw_predicted_normal
            ).all(0)
            safe_predicted_normal = torch.where(
                valid_predicted_normal[None],
                raw_predicted_normal,
                torch.zeros_like(raw_predicted_normal),
            )
            predicted_normal = F.normalize(
                safe_predicted_normal, dim=0, eps=1e-6
            )
            raw_target_normal = evidence["plane_normal_world"]
            valid_normal = torch.isfinite(raw_target_normal).all(0)
            safe_target_normal = torch.where(
                valid_normal[None],
                raw_target_normal,
                torch.zeros_like(raw_target_normal),
            )
            target_normal = F.normalize(
                safe_target_normal, dim=0, eps=1e-6
            )
            cosine = (
                predicted_normal * target_normal
            ).sum(0).abs()
            normal_weight = (
                weight
                * valid_normal.float()
                * valid_predicted_normal.float()
            )
            losses["normal"] = (
                (1.0 - cosine) * normal_weight
            ).sum() / normal_weight.sum().clamp_min(1)
    if "rho_mean" in evidence:
        rho = 1.0 / safe_predicted_depth.clamp_min(1e-3)
        raw_target = evidence["rho_mean"][0]
        raw_variance = evidence["rho_variance"][0]
        valid_target = (
            torch.isfinite(raw_target)
            & (raw_target > 0)
            & torch.isfinite(raw_variance)
            & (raw_variance > 0)
        )
        target = torch.where(
            valid_target, raw_target, torch.zeros_like(raw_target)
        )
        variance = torch.where(
            valid_target,
            raw_variance.clamp_min(1e-6),
            torch.ones_like(raw_variance),
        )
        support = (evidence["support_view_count"][0] >= 2).float()
        weight = (
            rigid
            * support
            * valid_prediction
            * valid_target.float()
        )
        nll = torch.log1p((rho - target).square() / variance)
        losses["inverse"] = (
            nll * weight
        ).sum() / weight.sum().clamp_min(1)
    if "mono_depth" in evidence:
        mono = evidence["mono_depth"][0]
        # DAV2 contributes only ordering.  Adjacent samples with negligible
        # monocular separation are ignored instead of treated as metric depth.
        pred_a = predicted_depth[::8, ::8]
        pred_b = predicted_depth[4::8, 4::8]
        mono_a = mono[::8, ::8]
        mono_b = mono[4::8, 4::8]
        height = min(pred_a.shape[0], pred_b.shape[0])
        width = min(pred_a.shape[1], pred_b.shape[1])
        delta = mono_a[:height, :width] - mono_b[:height, :width]
        sign = torch.sign(delta)
        finite_mono = mono[torch.isfinite(mono)]
        mono_scale = (
            finite_mono.abs().median().clamp_min(1e-3)
            if finite_mono.numel()
            else mono.new_tensor(1.0)
        )
        valid = (
            torch.isfinite(delta)
            & torch.isfinite(pred_a[:height, :width])
            & torch.isfinite(pred_b[:height, :width])
            & (delta.abs() > 0.01 * mono_scale)
        )
        order = sign * (
            pred_a[:height, :width] - pred_b[:height, :width]
        )
        losses["ordinal"] = (
            F.softplus(-order[valid]).mean()
            if bool(valid.any())
            else zero
        )
    total = (
        args.geometry_weight
        * (losses["chart"] + losses["inverse"])
        + args.plane_weight * losses["plane"]
        + args.normal_weight * losses["normal"]
        + args.ordinal_weight * losses["ordinal"]
    )
    return total, {
        name: float(value.detach()) for name, value in losses.items()
    }


def _replacement_stats(count: int, device) -> dict[str, torch.Tensor]:
    return {
        "evidence": torch.zeros(count, 4, device=device),
        "weight": torch.zeros(count, device=device),
        "observations": torch.zeros(
            count, dtype=torch.int16, device=device
        ),
        "sequence_bits": torch.zeros(
            count, dtype=torch.int64, device=device
        ),
        "retired": torch.zeros(
            count, dtype=torch.bool, device=device
        ),
    }


@torch.no_grad()
def _replacement_audit(
    view,
    structural,
    foliage,
    background,
    task,
    target,
    prediction,
    package,
    stats,
    sequence_bits,
) -> dict[str, int]:
    posterior = render_hybrid(
        view,
        structural,
        foliage,
        background=background,
        structural_trainable_start=None,
        volume_means_override=foliage.initialization_center,
        volume_opacity_scale=1.0,
        include_dynamic=False,
        surface_gate=torch.zeros_like(
            structural.get_opacity.reshape(-1)
        ),
    )
    volume_only = render_hybrid(
        view,
        structural,
        foliage,
        background=background,
        structural_trainable_start=None,
        include_dynamic=False,
        surface_gate=torch.zeros_like(
            structural.get_opacity.reshape(-1)
        ),
    )
    posterior_valid = (
        (posterior.volume_alpha[0] > 0.03)
        & (posterior.volume_depth[0] > 0)
    )
    new_match = torch.exp(
        -(
            package.volume_depth[0] - posterior.volume_depth[0]
        ).abs()
        / 0.45
    )
    old_match = torch.exp(
        -(
            package.surface_depth[0] - posterior.volume_depth[0]
        ).abs()
        / 0.45
    )
    depth_proof = (
        new_match
        * (1.0 - old_match)
        * posterior_valid
    ).clamp(0, 1)
    full_error = (prediction - target).abs().mean(0)
    volume_error = (volume_only.render - target).abs().mean(0)
    improvement = torch.sigmoid(
        (full_error - volume_error) / 0.02
    )
    audit = render_hybrid(
        view,
        structural,
        foliage,
        background=background,
        structural_trainable_start=None,
        include_dynamic=False,
        audit_fields=torch.stack(
            [
                task["p_canopy_core"],
                task["p_rigid"],
                depth_proof,
                improvement,
            ]
        ),
    ).responsibility
    if audit is None:
        return {"observed": 0, "newly_retired": 0}
    surface = audit[: package.structural_count]
    observed = surface[:, 0] > 0.02
    stats["evidence"] += surface[:, 1:5]
    stats["weight"] += surface[:, 0]
    stats["observations"][observed] += 1
    bit = sequence_bits[sequence_id(view.image_name)]
    stats["sequence_bits"][observed] |= bit
    average = stats["evidence"] / stats["weight"].clamp_min(1e-8)[:, None]
    sequence_count = torch.zeros_like(stats["observations"])
    for value in sequence_bits.values():
        sequence_count += (
            (stats["sequence_bits"] & value) != 0
        ).to(torch.int16)
    confidence = (
        average[:, 0]
        * (1.0 - average[:, 1])
        * average[:, 2]
        * average[:, 3]
    )
    eligible = (
        (confidence >= 0.45)
        & (stats["observations"] >= 3)
        & (sequence_count >= 2)
        & ~stats["retired"]
    )
    # Per-candidate replace-and-retire: a candidate is attenuated only after
    # its own cross-sequence ray/depth and RGB counterfactual proof passes.
    structural._opacity[eligible] -= 0.45
    newly = int(eligible.sum())
    stats["retired"] |= eligible & (
        structural.get_opacity.reshape(-1) < 0.02
    )
    return {"observed": int(observed.sum()), "newly_retired": newly}


@torch.no_grad()
def _evaluate(
    views,
    indices,
    surface,
    foliage,
    sky,
    appearance,
    background,
    fields,
    output: Path,
) -> list[dict]:
    from PIL import Image as PILImage

    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in indices:
        if index < 0 or index >= len(views):
            continue
        view = views[index]
        task = fields.fields(
            view.image_name,
            (view.image_height, view.image_width),
            background.device,
        )
        package = render_hybrid(
            view,
            surface,
            foliage,
            background=background,
            include_dynamic=False,
        )
        canonical = composite_white_background(
            package.render, package.alpha, sky(view)
        ).clamp(0, 1)
        conditioned_package = render_hybrid(
            view,
            surface,
            foliage,
            background=background,
            temporal_code=appearance.temporal_code(view.image_name),
            include_dynamic=True,
        )
        conditioned = appearance(
            composite_white_background(
                conditioned_package.render,
                conditioned_package.alpha,
                sky(view),
            ),
            view,
            task,
        ).clamp(0, 1)
        target = view.original_image.cuda()
        def metrics(value):
            mse = (value - target).square().mean().clamp_min(1e-12)
            return {
                "psnr": float(-10 * torch.log10(mse)),
                "ssim": float(ssim(value, target)),
                "mae": float((value - target).abs().mean()),
            }
        rows.append(
            {
                "index": index,
                "image_name": str(view.image_name),
                "canonical": metrics(canonical),
                "conditioned": metrics(conditioned),
            }
        )
        for name, value in (
            ("gt", target),
            ("canonical", canonical),
            ("conditioned", conditioned),
            ("error_x4", (conditioned - target).abs() * 4),
        ):
            image = (
                value.detach()
                .clamp(0, 1)
                .mul(255)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            PILImage.fromarray(image).save(
                output / f"{index:05d}_{name}.png"
            )
        release = getattr(view, "release_image", None)
        if release is not None:
            release()
    return rows


def main():
    args, dataset, opt, _pipe = _parse_args()
    output = Path(dataset.model_path).resolve()
    evidence_store = load_evidence_store(args.evidence_store)
    initialization = json.loads(
        (args.initialization / "initialization_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if initialization["evidence_hash"] != evidence_store["evidence_hash"]:
        raise RuntimeError("Initialization/evidence hash mismatch")
    resume = _load(args.resume.resolve()) if args.resume else None
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(f"Refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    surface = GaussianModel(dataset.sh_degree)
    scene = LazyScene(
        dataset,
        surface,
        image_cache_size=args.view_cache_size,
        image_prefetch_workers=args.image_prefetch_workers,
    )
    views = scene.getTrainCameras()
    scene_contract = json.loads(
        Path(evidence_store["scene_contract"]).read_text(encoding="utf-8")
    )
    expected_names = {
        Path(record["image_name"]).stem
        for record in scene_contract["records"]
    }
    loaded_names = {Path(str(view.image_name)).stem for view in views}
    if loaded_names != expected_names:
        raise RuntimeError(
            "Teacher cameras do not match the evidence contract: "
            f"loaded={len(loaded_names)}, expected={len(expected_names)}"
        )
    if resume is None:
        surface_audit = _initialize_surface(
            surface,
            Path(initialization["surface_seed"]),
            scene.cameras_extent,
            opt,
        )
    else:
        if resume.get("protocol") != PROTOCOL:
            raise RuntimeError("Unified teacher resume protocol mismatch")
        if resume["evidence_hash"] != evidence_store["evidence_hash"]:
            raise RuntimeError("Resume/evidence hash mismatch")
        surface.restore(
            _surface_capture_to_device(
                resume["surface"], torch.device("cuda")
            ),
            opt,
        )
        surface_audit = resume["surface_audit"]
    if len(surface.get_xyz) > args.maximum_surface_gaussians:
        raise RuntimeError(
            "Surface checkpoint/initialization exceeds the configured budget: "
            f"{len(surface.get_xyz)} > {args.maximum_surface_gaussians}. "
            "Use a clean pre-topology checkpoint or compact it before resume."
        )

    foliage = VolumetricFoliageModel(
        dataset.sh_degree, dynamic_rank=args.dynamic_rank
    ).cuda()
    seed_payload = _load(Path(initialization["foliage_seed"]))
    if resume is None:
        foliage.initialize_from_volume_state(seed_payload)
        eligible = torch.nonzero(
            (foliage.layer_role == LAYER_CANONICAL_CROWN)
            & (foliage.support_sequence_count >= 2),
            as_tuple=False,
        ).flatten()
        if len(eligible) > args.dynamic_seed_count:
            score = (
                foliage.occupancy_probability[eligible]
                * foliage.support_sequence_count[eligible].float()
            )
            eligible = eligible[
                torch.topk(score, args.dynamic_seed_count).indices
            ]
        dynamic_seed_count = foliage.append_dynamic_leaves(eligible)
    else:
        foliage.restore(resume["foliage"])
        dynamic_seed_count = int(resume["dynamic_seed_count"])

    semantic_contract = Path(evidence_store["semantic_contract"])
    semantic_payload = json.loads(
        semantic_contract.read_text(encoding="utf-8")
    )
    fields = OutdoorTaskFieldLookup(
        Path(dataset.source_path),
        Path(semantic_payload["tree_mask_pickle"]),
        semantic_contract,
        max_cached_views=0,
    )
    geometry = OutdoorGeometryEvidence(args.evidence_store)
    appearance = OutdoorAppearanceUncertainty(
        [view.image_name for view in views],
        rank=args.dynamic_rank,
        spatial_grid_size=24,
        device="cuda",
    )
    sky = CanonicalDirectionalSky(degree=2).cuda()
    if resume is not None:
        appearance.restore(resume["appearance"])
        sky.load_state_dict(resume["sky"])
    volume_optimizer = _volume_optimizer(
        args, foliage, appearance, sky
    )
    if resume is not None:
        volume_optimizer.load_state_dict(resume["volume_optimizer"])

    rgb_schedule = _full_epoch_schedule(
        len(views), args.iterations, args.seed + 1
    )
    by_stem = {
        Path(str(view.image_name)).stem: index
        for index, view in enumerate(views)
    }
    geometry_indices = [
        by_stem[stem]
        for stem in geometry.geometry_view_stems
        if stem in by_stem
    ]
    geometry_schedule = _cycle_schedule(
        geometry_indices, args.iterations, args.seed + 2
    )
    canopy_indices = sorted(
        range(len(views)),
        key=lambda index: fields.canopy_fraction(
            views[index].image_name,
            # This is a view-priority statistic, not a supervision field.
            # Computing it at full 2 MP resolution for 1.5k cameras delayed
            # every cold start by minutes without changing the ordering.
            (128, 128),
        ),
        reverse=True,
    )[: max(64, min(256, len(views)))]
    topology_schedule = _cycle_schedule(
        canopy_indices, args.iterations, args.seed + 3
    )
    schedule_hash = _schedule_digest(
        rgb_schedule, geometry_schedule, topology_schedule
    )
    implementation_hashes = {
        "trainer": _file_sha256(Path(__file__)),
        "appearance_uncertainty": _file_sha256(
            REPO_ROOT / "outdoor/appearance_uncertainty.py"
        ),
        "lazy_scene": _file_sha256(
            REPO_ROOT / "outdoor/lazy_scene.py"
        ),
        "dataset_reader": _file_sha256(
            SURFEL_ROOT / "scene/dataset_readers.py"
        ),
        "gaussian_model": _file_sha256(
            SURFEL_ROOT / "scene/gaussian_model.py"
        ),
        "task_fields": _file_sha256(
            REPO_ROOT / "outdoor/task_fields.py"
        ),
        "mask_lookup": _file_sha256(
            REPO_ROOT / "matcha/cambridge_masks.py"
        ),
        "training_evidence": _file_sha256(
            REPO_ROOT / "outdoor/training_evidence.py"
        ),
        "hybrid_renderer": _file_sha256(
            REPO_ROOT / "outdoor/hybrid_gaussian_renderer.py"
        ),
        "mixed_forward_cuda": _file_sha256(
            SURFEL_ROOT
            / "submodules/diff-surfel-rasterization/"
            "cuda_rasterizer/forward.cu"
        ),
        "mixed_backward_cuda": _file_sha256(
            SURFEL_ROOT
            / "submodules/diff-surfel-rasterization/"
            "cuda_rasterizer/mixed_backward.cu"
        ),
    }
    if resume is not None:
        resume_hashes = resume.get("implementation_hashes", {})
        if resume_hashes != implementation_hashes:
            changed = {
                key
                for key in set(resume_hashes) | set(implementation_hashes)
                if resume_hashes.get(key) != implementation_hashes.get(key)
            }
            performance_only = (
                args.allow_performance_resume
                and changed == set(PERFORMANCE_RESUME_PREDECESSOR)
                and all(
                    resume_hashes.get(key) == value
                    for key, value in PERFORMANCE_RESUME_PREDECESSOR.items()
                )
            )
            if not performance_only:
                raise RuntimeError(
                    "Resume implementation/CUDA hash mismatch; start a new "
                    "unified run"
                )
            print(
                "Resuming the exact pre-fast-path checkpoint with "
                "I/O/allocator-only compatibility enabled."
            )
    if resume is not None and resume["schedule_hash"] != schedule_hash:
        raise RuntimeError("Resume camera schedules changed")
    if resume is not None:
        resume_profile = resume.get("training_profile", "quality")
        if resume_profile != args.training_profile:
            raise RuntimeError(
                "Resume training profile changed: "
                f"{resume_profile} != {args.training_profile}"
            )
        resume_budget = resume.get("maximum_surface_gaussians")
        if (
            resume_budget is not None
            and int(resume_budget) != args.maximum_surface_gaussians
        ):
            raise RuntimeError(
                "Resume surface budget changed: "
                f"{resume_budget} != "
                f"{args.maximum_surface_gaussians}"
            )
        resume_growth = resume.get(
            "maximum_surface_growth_per_event"
        )
        if (
            resume_growth is not None
            and int(resume_growth)
            != args.maximum_surface_growth_per_event
        ):
            raise RuntimeError(
                "Resume per-event surface budget changed: "
                f"{resume_growth} != "
                f"{args.maximum_surface_growth_per_event}"
            )

    background = torch.ones(3, device="cuda")
    if resume is None:
        start_step = 0
        volume_stats = _volume_stats(foliage)
        replacement_stats = None
        topology_events = []
        replacement_events = []
    else:
        start_step = int(resume["iteration"])
        volume_stats = {
            name: value.cuda()
            for name, value in resume["volume_stats"].items()
        }
        replacement_stats = (
            None
            if resume["replacement_stats"] is None
            else {
                name: value.cuda()
                for name, value in resume["replacement_stats"].items()
            }
        )
        topology_events = list(resume["topology_events"])
        replacement_events = list(resume["replacement_events"])
        random.setstate(resume["python_rng_state"])
        np.random.set_state(resume["numpy_rng_state"])
        torch.set_rng_state(resume["torch_rng_state"])
        torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
    sequence_names = sorted(
        {sequence_id(view.image_name) for view in views}
    )
    if len(sequence_names) > 62:
        raise RuntimeError("Replacement sequence bitset exceeds 62")
    sequence_bits = {
        name: 1 << index for index, name in enumerate(sequence_names)
    }

    trace = output / "training_trace.jsonl"
    started = time.time()
    progress = tqdm(
        range(start_step, args.iterations),
        initial=start_step,
        total=args.iterations,
        desc="unified outdoor teacher",
    )

    def prefetch_training_images(begin: int) -> None:
        if args.image_prefetch_depth <= 0:
            return
        requested = []
        end = min(args.iterations, begin + args.image_prefetch_depth)
        for future_step in range(max(begin, start_step), end):
            requested.append(
                views[int(rgb_schedule[future_step])]
            )
            if (
                _phase(
                    future_step,
                    args.iterations,
                    args.training_profile,
                )
                == "topology"
                and args.topology_every > 0
                and (future_step + 1) % args.topology_every == 0
            ):
                requested.append(
                    views[int(topology_schedule[future_step])]
                )
        scene.prefetch_images(requested)

    prefetch_training_images(start_step)
    for step in progress:
        phase = _phase(step, args.iterations, args.training_profile)
        surface.update_learning_rate(step + 1)
        if (step + 1) % 1000 == 0:
            surface.oneupSHdegree()
        surface.optimizer.zero_grad(set_to_none=True)
        volume_optimizer.zero_grad(set_to_none=True)
        view = views[int(rgb_schedule[step])]
        task = fields.fields(
            view.image_name,
            (view.image_height, view.image_width),
            torch.device("cuda"),
        )
        target = view.original_image.cuda(non_blocking=True)
        # Decode upcoming random-schedule images while CUDA executes the
        # current forward/backward passes.
        prefetch_training_images(step + 1)
        package = render_hybrid(
            view,
            surface,
            foliage,
            background=background,
            include_dynamic=False,
            structural_trainable_start=0,
            audit_fields=torch.stack(
                [
                    task["p_canopy_core"],
                    task["p_rigid"],
                    (target - 0.5).abs().mean(0),
                ]
            ),
        )
        canonical = composite_white_background(
            package.render, package.alpha, sky(view)
        )
        canonical_weight = task["w_rgb"]
        static_confidence_mean = canonical_weight.new_tensor(1.0)
        if phase in {
            "dynamic_appearance",
            "ownership_cleanup",
            "canonical_polish",
        }:
            # Once the conditioned branch is active, pixels that repeatedly
            # disagree in space/time no longer drag the canonical crown into
            # a broad average.  Sigma is detached here: uncertainty is trained
            # by its proper likelihood below and cannot reduce this loss by
            # simply inflating itself.
            sigma = appearance.spatial_uncertainty(
                view.image_name,
                (view.image_height, view.image_width),
            ).detach()
            canopy_confidence = (
                0.10 / (sigma[0] + 0.05)
            ).clamp(0.15, 1.0)
            sky_confidence = (
                0.10 / (sigma[1] + 0.05)
            ).clamp(0.25, 1.0)
            canonical_weight = canonical_weight * (
                task["p_rigid"]
                + task["p_canopy"] * canopy_confidence
                + task["p_sky"] * sky_confidence
            ).clamp(0, 1)
            static_confidence_mean = (
                canopy_confidence * task["p_canopy"]
            ).sum() / task["p_canopy"].sum().clamp_min(1)
        photo = _photo_loss(
            canonical,
            target,
            canonical_weight,
            opt.lambda_dssim,
        )
        rigid_photo = _photo_loss(
            canonical,
            target,
            task["p_rigid"],
            0.10,
        )
        surface_in_canopy = _weighted_mean(
            package.surface_alpha,
            task["p_canopy_core"],
        )
        volume_in_rigid = _weighted_mean(
            package.volume_alpha,
            task["p_rigid"],
        )
        finite_in_sky = _weighted_mean(
            package.surface_alpha + package.volume_alpha,
            task["p_sky"],
        )
        ownership = (
            surface_in_canopy + volume_in_rigid + finite_in_sky
        )
        alpha = package.volume_alpha[0].clamp(1e-5, 1 - 1e-5)
        positive = (
            task["p_canopy_core"] * (1.0 - task["p_transient"])
        )
        negative = (
            task["p_rigid"] + 0.35 * task["p_sky"]
        ).clamp(0, 1) * (1.0 - task["p_boundary_uncertain"])
        hit = (
            (-torch.log(alpha) * positive).sum()
            / positive.sum().clamp_min(1)
        )
        free = (
            (-torch.log1p(-alpha) * negative).sum()
            / negative.sum().clamp_min(1)
        )
        occupancy = 0.15 * hit + free
        delta = foliage.xyz - foliage.initialization_center
        variance = foliage.position_covariance.diagonal(
            dim1=-2, dim2=-1
        ).clamp_min(1e-6)
        position_nll = (delta.square() / variance).sum(-1)
        position_weight = torch.where(
            foliage.static_skeleton_mask,
            position_nll.new_tensor(2.0),
            position_nll.new_tensor(0.20),
        )
        geometry_floor = (
            position_nll * position_weight
        ).mean()
        canonical_loss = (
            photo
            + 0.25 * rigid_photo
            + args.ownership_weight * ownership
            + args.occupancy_weight * occupancy
            + 0.015 * geometry_floor
        )
        canonical_loss.backward()
        # Surface topology only consumes rigid-dominant responsibility.  A
        # canopy pixel can improve color/opacity, but never create a new 2D
        # surfel or a giant facade/tree bridge.
        surface_responsibility = package.responsibility[
            : package.structural_count
        ]
        total_responsibility = surface_responsibility[:, 0].clamp_min(
            1e-8
        )
        rigid_fraction = (
            surface_responsibility[:, 2] / total_responsibility
        )
        canopy_fraction = (
            surface_responsibility[:, 1] / total_responsibility
        )
        surface_visible = (
            (package.radii[: package.structural_count] > 0)
            & (rigid_fraction >= 0.70)
            & (canopy_fraction <= 0.15)
        )
        if package.surface_means2d.grad is not None:
            surface.add_densification_stats(
                package.surface_means2d, surface_visible
            )
            surface.max_radii2D[surface_visible] = torch.maximum(
                surface.max_radii2D[surface_visible],
                package.radii[: package.structural_count][surface_visible],
            )
        _accumulate_volume_stats(volume_stats, package)

        conditioned_loss = canonical_loss.new_zeros(())
        uncertainty_loss = canonical_loss.new_zeros(())
        high_frequency_loss = canonical_loss.new_zeros(())
        if phase in {"dynamic_appearance", "ownership_cleanup"}:
            conditioned_package = render_hybrid(
                view,
                surface,
                foliage,
                background=background,
                temporal_code=appearance.temporal_code(view.image_name),
                include_dynamic=True,
                structural_trainable_start=None,
                audit_fields=torch.stack(
                    [
                        task["p_canopy_core"],
                        task["p_rigid"],
                        (target - canonical.detach()).abs().mean(0),
                    ]
                ),
            )
            conditioned_base = composite_white_background(
                conditioned_package.render,
                conditioned_package.alpha,
                sky(view),
            )
            conditioned = appearance(
                conditioned_base, view, task
            )
            conditioned_photo = _photo_loss(
                conditioned,
                target,
                (
                    task["p_canopy"]
                    + task["p_sky"]
                    + 0.15 * task["p_rigid"]
                ).clamp(0, 1),
                0.15,
            )
            uncertainty_loss = appearance.heteroscedastic_loss(
                conditioned,
                target,
                task,
                image_name=view.image_name,
            )
            high_frequency_loss = _high_frequency_loss(
                conditioned,
                target,
                task["p_canopy_core"]
                * (1.0 - task["p_transient"])
                * (1.0 - task["p_boundary_uncertain"]),
            )
            dynamic_regularization = (
                foliage.deformation_basis[
                    foliage.dynamic_leaf_mask
                ]
                .square()
                .mean()
            )
            conditioned_loss = (
                args.dynamic_weight * conditioned_photo
                + args.appearance_weight * uncertainty_loss
                + args.high_frequency_weight * high_frequency_loss
                + 1e-3 * appearance.regularization()
                + 0.02 * dynamic_regularization
            )
            conditioned_loss.backward()
            _accumulate_volume_stats(
                volume_stats, conditioned_package
            )
            del conditioned_package, conditioned_base, conditioned

        geometry_loss = canonical_loss.new_zeros(())
        geometry_values = {
            "chart": 0.0,
            "plane": 0.0,
            "normal": 0.0,
            "inverse": 0.0,
            "ordinal": 0.0,
        }
        if (
            args.geometry_every > 0
            and (step + 1) % args.geometry_every == 0
            and geometry_schedule[step] >= 0
        ):
            geometry_view = views[int(geometry_schedule[step])]
            geometry_task = fields.fields(
                geometry_view.image_name,
                (
                    geometry_view.image_height,
                    geometry_view.image_width,
                ),
                torch.device("cuda"),
            )
            source_fields = geometry.fields(
                geometry_view.image_name,
                device=torch.device("cuda"),
                shape=(
                    geometry_view.image_height,
                    geometry_view.image_width,
                ),
            )
            if source_fields:
                geometry_package = render_hybrid(
                    geometry_view,
                    surface,
                    foliage,
                    background=background,
                    include_dynamic=False,
                    structural_trainable_start=0,
                    volume_opacity_scale=0.0,
                )
                geometry_loss, geometry_values = _geometry_losses(
                    geometry_package,
                    source_fields,
                    geometry_task["p_rigid"],
                    args,
                )
                phase_floor = (
                    1.0
                    if phase
                    in {"canonical_bootstrap", "topology"}
                    else 0.35
                )
                geometry_loss = phase_floor * geometry_loss
                geometry_loss.backward()
                del geometry_package

        topology_loss = canonical_loss.new_zeros(())
        if (
            phase == "topology"
            and args.topology_every > 0
            and (step + 1) % args.topology_every == 0
        ):
            topology_view = views[int(topology_schedule[step])]
            topology_task = fields.fields(
                topology_view.image_name,
                (
                    topology_view.image_height,
                    topology_view.image_width,
                ),
                torch.device("cuda"),
            )
            topology_target = topology_view.original_image.cuda(
                non_blocking=True
            )
            topology_package = render_hybrid(
                topology_view,
                surface,
                foliage,
                background=background,
                include_dynamic=False,
                structural_trainable_start=None,
                audit_fields=torch.stack(
                    [
                        topology_task["p_canopy_core"],
                        topology_task["p_rigid"],
                        (
                            topology_target
                            - topology_target.mean(dim=(1, 2), keepdim=True)
                        )
                        .abs()
                        .mean(0),
                    ]
                ),
            )
            topology_prediction = composite_white_background(
                topology_package.render,
                topology_package.alpha,
                sky(topology_view),
            )
            topology_loss = _photo_loss(
                topology_prediction,
                topology_target,
                topology_task["p_canopy"],
                0.10,
            )
            (0.20 * topology_loss).backward()
            _accumulate_volume_stats(
                volume_stats, topology_package
            )
            del topology_package, topology_prediction, topology_target

        replacement_event = None
        if phase == "ownership_cleanup":
            if replacement_stats is None:
                replacement_stats = _replacement_stats(
                    len(surface.get_xyz), surface.get_xyz.device
                )
            if len(replacement_stats["weight"]) != len(surface.get_xyz):
                raise RuntimeError(
                    "Surface topology changed after cleanup started"
                )
            if (
                args.replacement_every > 0
                and (step + 1) % args.replacement_every == 0
            ):
                replacement_event = {
                    "iteration": step + 1,
                    **_replacement_audit(
                        view,
                        surface,
                        foliage,
                        background,
                        task,
                        target,
                        canonical.detach(),
                        package,
                        replacement_stats,
                        sequence_bits,
                    ),
                }
                replacement_events.append(replacement_event)

        _apply_volume_role_gradients(foliage, phase)
        surface.optimizer.step()
        volume_optimizer.step()
        surface.optimizer.zero_grad(set_to_none=True)
        volume_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            foliage.quaternions.copy_(
                F.normalize(foliage.quaternions, dim=-1)
            )
            foliage.opacity_logits.clamp_(-10, 2)
            surface._opacity.clamp_(-12, 4)

        topology_event = None
        if (
            phase == "topology"
            and (step + 1) >= max(args.densify_from_iter, 1)
            and (step + 1) <= args.densify_until_iter
        ):
            if (step + 1) % args.densification_interval == 0:
                before_count = len(surface.get_xyz)
                surface_event = surface.densify_and_prune_bounded(
                    args.densify_grad_threshold,
                    args.opacity_cull,
                    scene.cameras_extent,
                    64,
                    max_points=args.maximum_surface_gaussians,
                    max_growth=args.maximum_surface_growth_per_event,
                )
                topology_event = {
                    "iteration": step + 1,
                    "surface_before": before_count,
                    "surface_after": len(surface.get_xyz),
                    "surface": surface_event,
                }
            if (
                (step + 1) % args.volume_densify_every == 0
                and len(foliage)
            ):
                event = _adapt_volume(args, foliage, volume_stats)
                new_to_old = event.pop("_new_to_old", None)
                if new_to_old is not None:
                    volume_optimizer = _migrate_volume_optimizer(
                        args,
                        foliage,
                        appearance,
                        sky,
                        volume_optimizer,
                        new_to_old,
                    )
                volume_stats = _volume_stats(foliage)
                topology_event = {
                    **(topology_event or {"iteration": step + 1}),
                    "volume": event,
                }
            if topology_event is not None:
                topology_events.append(topology_event)
        if (
            phase == "topology"
            and args.opacity_reset_interval > 0
            and (step + 1) % args.opacity_reset_interval == 0
            and (step + 1) <= args.densify_until_iter
        ):
            surface.reset_opacity()

        total_loss = (
            canonical_loss.detach()
            + conditioned_loss.detach()
            + geometry_loss.detach()
            + 0.20 * topology_loss.detach()
        )
        if step == 0 or (step + 1) % args.log_every == 0:
            row = {
                "iteration": step + 1,
                "phase": phase,
                "loss": float(total_loss),
                "canonical_photo": float(photo.detach()),
                "rigid_photo": float(rigid_photo.detach()),
                "conditioned": float(conditioned_loss.detach()),
                "uncertainty": float(uncertainty_loss.detach()),
                "high_frequency": float(
                    high_frequency_loss.detach()
                ),
                "canonical_canopy_confidence": float(
                    static_confidence_mean.detach()
                ),
                "ownership": float(ownership.detach()),
                "occupancy": float(occupancy.detach()),
                "geometry": geometry_values,
                "surface_count": len(surface.get_xyz),
                "foliage_count": len(foliage),
                "static_skeleton": int(
                    foliage.static_skeleton_mask.sum()
                ),
                "canonical_crown": int(
                    foliage.canonical_crown_mask.sum()
                ),
                "dynamic_leaf": int(
                    foliage.dynamic_leaf_mask.sum()
                ),
                "topology_event": topology_event,
                "replacement_event": replacement_event,
                "elapsed_sec": time.time() - started,
            }
            with trace.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            progress.set_postfix(
                phase=phase,
                loss=f"{float(total_loss):.4f}",
                surface=len(surface.get_xyz),
                volume=len(foliage),
            )
        if (
            args.checkpoint_every > 0
            and (step + 1) % args.checkpoint_every == 0
        ):
            _save_checkpoint(
                output / "unified_teacher_checkpoint.pth",
                {
                    "protocol": PROTOCOL,
                    "iteration": step + 1,
                    "training_profile": args.training_profile,
                    "phase_schedule": TRAINING_PROFILES[
                        args.training_profile
                    ]["phases"],
                    "maximum_surface_gaussians": (
                        args.maximum_surface_gaussians
                    ),
                    "maximum_surface_growth_per_event": (
                        args.maximum_surface_growth_per_event
                    ),
                    "evidence_hash": evidence_store["evidence_hash"],
                    "schedule_hash": schedule_hash,
                    "schedules": {
                        "rgb": rgb_schedule,
                        "geometry": geometry_schedule,
                        "topology": topology_schedule,
                    },
                    "implementation_hashes": implementation_hashes,
                    "phase": phase,
                    "surface": surface.capture(),
                    "surface_audit": surface_audit,
                    "foliage": foliage.capture(),
                    "dynamic_seed_count": dynamic_seed_count,
                    "sky": sky.state_dict(),
                    "appearance": appearance.capture(),
                    "volume_optimizer": volume_optimizer.state_dict(),
                    "volume_stats": volume_stats,
                    "replacement_stats": replacement_stats,
                    "topology_events": topology_events,
                    "replacement_events": replacement_events,
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all(),
                },
            )
        del package, canonical, target
        if (
            args.maintenance_every > 0
            and (step + 1) % args.maintenance_every == 0
        ):
            gc.collect()
        # Densification replaces large parameter tensors.  Release only those
        # stale allocator blocks; emptying the CUDA cache every iteration
        # previously serialized the entire pipeline and defeated reuse.
        if topology_event is not None:
            torch.cuda.empty_cache()

    teacher_state = output / "unified_teacher_state.pth"
    torch.save(
        {
            "protocol": PROTOCOL,
            "iteration": args.iterations,
            "training_profile": args.training_profile,
            "phase_schedule": TRAINING_PROFILES[
                args.training_profile
            ]["phases"],
            "maximum_surface_gaussians": (
                args.maximum_surface_gaussians
            ),
            "maximum_surface_growth_per_event": (
                args.maximum_surface_growth_per_event
            ),
            "evidence_hash": evidence_store["evidence_hash"],
            "surface": surface.capture(),
            "surface_audit": surface_audit,
            "foliage": foliage.capture(),
            "dynamic_seed_count": dynamic_seed_count,
            "sky": sky.state_dict(),
            "sky_degree": sky.degree,
            "appearance": appearance.capture(),
            "topology_events": topology_events,
            "replacement_events": replacement_events,
        },
        teacher_state,
    )
    scene.save(args.iterations)
    eval_indices = [
        int(value)
        for value in args.eval_indices.split(",")
        if value.strip()
    ]
    evaluation = _evaluate(
        views,
        eval_indices,
        surface,
        foliage,
        sky,
        appearance,
        background,
        fields,
        output / "visualization",
    )
    scene.close()
    result = {
        "protocol": PROTOCOL,
        "iterations": args.iterations,
        "training_profile": args.training_profile,
        "phase_schedule": TRAINING_PROFILES[
            args.training_profile
        ]["phases"],
        "maximum_surface_gaussians": args.maximum_surface_gaussians,
        "maximum_surface_growth_per_event": (
            args.maximum_surface_growth_per_event
        ),
        "evidence_hash": evidence_store["evidence_hash"],
        "teacher_state": str(teacher_state),
        "surface_ply": str(
            output
            / "point_cloud"
            / f"iteration_{args.iterations}"
            / "point_cloud.ply"
        ),
        "surface_initialization": surface_audit,
        "surface_final_count": len(surface.get_xyz),
        "foliage_final_count": len(foliage),
        "layer_counts": {
            "static_skeleton": int(
                foliage.static_skeleton_mask.sum()
            ),
            "canonical_crown": int(
                foliage.canonical_crown_mask.sum()
            ),
            "dynamic_leaf": int(foliage.dynamic_leaf_mask.sum()),
        },
        "geometry_evidence": geometry.audit(),
        "implementation_hashes": implementation_hashes,
        "topology_events": topology_events,
        "replacement_events": replacement_events,
        "targeted_evaluation": evaluation,
        "historical_parent_ply_used": False,
        "all_real_rgb_from_iteration_one": True,
        "canopy_structural_topology_gradient": False,
        "elapsed_sec": time.time() - started,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
