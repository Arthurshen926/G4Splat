#!/usr/bin/env python
"""Protected semantic residual refinement of a full all-real 2DGS state.

This is an explicitly declared post-checkpoint schedule fork.  It restores the
full 60k Gaussian/Adam/RNG/appearance state, freezes that complete Gaussian
prefix, and only appends bounded foliage-class children after persistent
canopy residuals from multiple distinct real cameras.  It also trains a small
camera-independent directional sky behind raster alpha.  The foliage branch
still uses the existing 2D surfel rasterizer; it is not labelled volumetric.
"""

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
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SURFEL_ROOT))

from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from matcha.cambridge_training import (  # noqa: E402
    PerImageAffineColorCorrection,
    apply_per_image_affine_color_correction,
    save_per_image_affine_color_correction,
)
from outdoor.directional_sky import (  # noqa: E402
    CanonicalDirectionalSky,
    composite_white_background,
)
from outdoor.render_contract import (  # noqa: E402
    assert_render_contract,
    capture_render_contract,
    compare_render_contract,
    declared_mip_filter,
    directory_sha256,
)
from outdoor.task_fields import OutdoorTaskFieldLookup  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from scripts.train_standard_full_2dgs import (  # noqa: E402
    _capture_rng_state,
    _file_digest,
    _fixed_camera_schedule,
    _input_audit,
    _json_digest,
    _load_training_state,
    _restore_color_correction_state,
    _restore_rng_state,
    _scene_input_ply_audit,
)


RUN_SCHEMA_VERSION = "protected_semantic_residual_schedule_fork_v1"


def _weighted_charbonnier(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    if weight.ndim == 2:
        weight = weight[None]
    weight = weight.to(device=prediction.device, dtype=prediction.dtype)
    if weight.shape[0] == 1:
        weight = weight.expand_as(prediction)
    denominator = weight.sum()
    if float(denominator.detach()) <= 0.0:
        return prediction.sum() * 0.0
    residual = torch.sqrt((prediction - target).square() + epsilon**2)
    return (residual * weight).sum() / denominator


@torch.no_grad()
def _record_distinct_view_hits(
    slots: torch.Tensor,
    hit_mask: torch.Tensor,
    camera_index: int,
) -> None:
    """Store exact camera identities until each Gaussian reaches the threshold."""
    indices = torch.nonzero(hit_mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        return
    current = slots[indices]
    unseen = torch.all(current != int(camera_index), dim=1)
    candidates = indices[unseen]
    if candidates.numel() == 0:
        return
    empty = slots[candidates] < 0
    has_capacity = torch.any(empty, dim=1)
    candidates = candidates[has_capacity]
    if candidates.numel() == 0:
        return
    empty = slots[candidates] < 0
    first_empty = torch.argmax(empty.to(torch.int8), dim=1)
    slots[candidates, first_empty] = int(camera_index)


@torch.no_grad()
def _projected_center_mask(
    xyz: torch.Tensor,
    view,
    pixel_mask: torch.Tensor,
) -> torch.Tensor:
    rotation = torch.as_tensor(view.R, device=xyz.device, dtype=xyz.dtype)
    translation = torch.as_tensor(view.T, device=xyz.device, dtype=xyz.dtype)
    camera = xyz @ rotation + translation
    depth = camera[:, 2]
    u = camera[:, 0] / depth.clamp_min(1e-6) * view.focal_x + view.cx
    v = camera[:, 1] / depth.clamp_min(1e-6) * view.focal_y + view.cy
    valid = (
        (depth > view.znear)
        & (depth < view.zfar)
        & (u >= 0)
        & (u < view.image_width)
        & (v >= 0)
        & (v < view.image_height)
    )
    rows = v.round().long().clamp(0, view.image_height - 1)
    columns = u.round().long().clamp(0, view.image_width - 1)
    return valid & (pixel_mask[rows, columns] > 0.5)


def _sequence_label(image_name: str) -> str:
    value = str(image_name)
    return value.split("__", 1)[0].split("/", 1)[0]


def _capture_protected_parameters(
    gaussians: GaussianModel, point_count: int
) -> dict[str, torch.Tensor]:
    return {
        # Protection auditing must not compete with the renderer and Adam for
        # scarce GPU memory.  The immutable reference lives on CPU and is
        # compared back in bounded chunks at allocation/save barriers.
        name: value[:point_count].detach().cpu().clone()
        for name, value in {
            "xyz": gaussians._xyz,
            "features_dc": gaussians._features_dc,
            "features_rest": gaussians._features_rest,
            "scaling": gaussians._scaling,
            "rotation": gaussians._rotation,
            "opacity": gaussians._opacity,
        }.items()
    }


def _protected_parameters_are_bit_exact(
    gaussians: GaussianModel,
    snapshots: dict[str, torch.Tensor],
    *,
    chunk_rows: int = 65_536,
) -> bool:
    current = {
        "xyz": gaussians._xyz,
        "features_dc": gaussians._features_dc,
        "features_rest": gaussians._features_rest,
        "scaling": gaussians._scaling,
        "rotation": gaussians._rotation,
        "opacity": gaussians._opacity,
    }
    for name, reference in snapshots.items():
        for start in range(0, len(reference), int(chunk_rows)):
            end = min(start + int(chunk_rows), len(reference))
            candidate = current[name][start:end].detach().cpu()
            if not torch.equal(candidate, reference[start:end]):
                return False
    return True


def _sha256_files(paths: list[Path]) -> dict:
    records = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        records[str(path.relative_to(REPO_ROOT))] = {
            "sha256": _file_digest(path),
            "bytes": int(path.stat().st_size),
        }
    return records


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Protected canopy/sky residual schedule fork for Cambridge 2DGS."
    )
    model_params = ModelParams(parser)
    optimization_params = OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.set_defaults(
        iterations=12_000,
        position_lr_init=4e-5,
        position_lr_final=4e-7,
        position_lr_delay_mult=0.1,
        position_lr_max_steps=12_000,
        feature_lr=5e-4,
        opacity_lr=5e-3,
        scaling_lr=5e-4,
        rotation_lr=2e-4,
        data_device="cpu",
        non_position_lr_decay_from=6_000,
        non_position_lr_final_mult=0.2,
        percent_dense=0.01,
    )
    parser.add_argument("--parent-training-state", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path, required=True)
    parser.add_argument("--task-semantic-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-schedule-seed", type=int, default=13)
    parser.add_argument("--sky-degree", type=int, default=2)
    parser.add_argument("--sky-lr", type=float, default=5e-3)
    parser.add_argument("--sky-loss-weight", type=float, default=1.0)
    parser.add_argument("--charbonnier-epsilon", type=float, default=1e-3)
    parser.add_argument("--boundary-radius", type=int, default=3)
    parser.add_argument("--task-field-cache-views", type=int, default=0)
    parser.add_argument("--topology-start", type=int, default=4_000)
    parser.add_argument("--topology-interval", type=int, default=4_500)
    parser.add_argument("--minimum-persistent-views", type=int, default=3)
    parser.add_argument("--canopy-grad-hit-threshold", type=float, default=1e-10)
    parser.add_argument("--topology-grad-threshold", type=float, default=0.0)
    parser.add_argument("--topology-grad-quantile", type=float, default=0.75)
    parser.add_argument("--minimum-support-sequences", type=int, default=2)
    parser.add_argument("--minimum-support-baseline", type=float, default=0.75)
    parser.add_argument("--max-new-gaussians", type=int, default=24_000)
    parser.add_argument("--max-new-per-event", type=int, default=12_000)
    parser.add_argument("--split-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--children-per-parent", type=int, default=2)
    parser.add_argument("--opacity-ceiling", type=float, default=0.025)
    parser.add_argument("--max-parent-scale-fraction", type=float, default=0.08)
    parser.add_argument("--foliage-geometry-confidence", type=float, default=0.35)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    dataset = model_params.extract(args)
    opt = optimization_params.extract(args)
    pipe = pipeline_params.extract(args)
    if not dataset.source_path or not dataset.model_path:
        parser.error("Both --source_path/-s and --model_path/-m are required")
    if not dataset.white_background:
        parser.error("Directional sky residual refinement requires --white_background")
    if opt.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.topology_interval < 1:
        parser.error("--topology-interval must be positive")
    if args.minimum_persistent_views < 2:
        parser.error("--minimum-persistent-views must be at least 2")
    if not 0.0 <= args.topology_grad_quantile <= 1.0:
        parser.error("--topology-grad-quantile must be in [0, 1]")
    if args.minimum_support_sequences < 1:
        parser.error("--minimum-support-sequences must be positive")
    if args.minimum_support_baseline < 0:
        parser.error("--minimum-support-baseline must be non-negative")
    if args.max_new_gaussians < 0 or args.max_new_per_event < 0:
        parser.error("Gaussian append caps must be non-negative")
    if not 0.0 <= args.split_fraction <= 1.0:
        parser.error("--split-fraction must be in [0, 1]")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    if args.task_field_cache_views < 0:
        parser.error("--task-field-cache-views must be non-negative")
    return args, dataset, opt, pipe


def _validate_parent_contract(
    payload: dict,
    *,
    source_path: Path,
    dataset,
    input_audit: dict,
) -> dict:
    parent_contract = payload["contract"]
    if parent_contract.get("protocol") != "standard_2dgs_training_state_contract_v1":
        raise RuntimeError("Parent is not a standard all-real 2DGS full training state")
    parent_input = parent_contract.get("input", {})
    expected = {
        "source_path": str(source_path),
        "image_count": int(input_audit["image_count"]),
        "camera_set_sha256_input": input_audit["camera_set_sha256_input"],
        "name_mapping_sha256": input_audit["name_mapping_sha256"],
        "white_background": bool(dataset.white_background),
        "resolution": int(dataset.resolution),
        "renderer": "diff_surfel_rasterization",
    }
    mismatches = {
        key: {"parent": parent_input.get(key), "current": value}
        for key, value in expected.items()
        if parent_input.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Parent/data immutable contract mismatch:\n"
            + json.dumps(mismatches, indent=2)
        )
    appearance = parent_input.get("per_image_affine_color", {})
    if not appearance.get("enabled", False) or payload.get("color_correction") is None:
        raise RuntimeError(
            "This residual fork requires the inherited 60k affine appearance state"
        )
    return parent_contract


def _save_residual_state(
    destination: Path,
    *,
    global_iteration: int,
    gaussians: GaussianModel,
    sky: CanonicalDirectionalSky,
    sky_optimizer,
    color_correction,
    contract: dict,
    trace_state: dict,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    torch.save(
        {
            "version": 3,
            "protocol": RUN_SCHEMA_VERSION,
            "global_iteration": int(global_iteration),
            "model_capture": gaussians.capture(),
            "sky": {
                "model_state": sky.state_dict(),
                "optimizer_state": sky_optimizer.state_dict(),
                "degree": sky.degree,
            },
            "frozen_color_correction": {
                "model_state": color_correction.state_dict(),
                "image_name_to_index": color_correction.image_name_to_index,
            },
            "rng_state": _capture_rng_state(),
            "trace_state": dict(trace_state),
            "contract": contract,
        },
        destination,
    )
    return destination


def main() -> None:
    args, dataset, opt, pipe = _parse_args()
    source_path = Path(dataset.source_path).resolve()
    model_path = Path(dataset.model_path).resolve()
    if model_path.exists() and any(model_path.iterdir()):
        raise FileExistsError(f"Refusing to mix output with existing files: {model_path}")
    model_path.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)

    parent_state_path = args.parent_training_state.resolve()
    parent_state_sha256 = _file_digest(parent_state_path)
    parent_payload = _load_training_state(parent_state_path)
    parent_iteration = int(parent_payload["iteration"])
    global_iteration = parent_iteration + int(opt.iterations)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    views = scene.getTrainCameras()
    input_audit = _input_audit(source_path, views)
    input_audit.update(_scene_input_ply_audit(model_path))
    parent_contract = _validate_parent_contract(
        parent_payload,
        source_path=source_path,
        dataset=dataset,
        input_audit=input_audit,
    )
    schedule, schedule_audit = _fixed_camera_schedule(
        views,
        iterations=int(opt.iterations),
        seed=int(args.camera_schedule_seed),
    )
    sequence_names = sorted({_sequence_label(view.image_name) for view in views})
    sequence_to_code = {
        name: index for index, name in enumerate(sequence_names)
    }
    camera_sequence_codes = torch.tensor(
        [sequence_to_code[_sequence_label(view.image_name)] for view in views],
        dtype=torch.int16,
        device="cuda",
    )
    camera_centers = torch.stack(
        [view.camera_center.detach().to(device="cuda") for view in views]
    )
    task_fields = OutdoorTaskFieldLookup(
        source_path,
        args.tree_mask_pickle.resolve(),
        args.task_semantic_manifest.resolve(),
        boundary_radius=args.boundary_radius,
        max_cached_views=args.task_field_cache_views,
    )

    train_names = [str(view.image_name) for view in views]
    color_correction = PerImageAffineColorCorrection(train_names).cuda()
    color_optimizer = torch.optim.Adam(color_correction.parameters(), lr=1e-3)
    gaussians.restore(parent_payload["model_capture"], opt)
    background = torch.ones(3, dtype=torch.float32, device="cuda")
    identity_indices = sorted({0, len(views) // 2, len(views) - 1})
    zero_step_before = capture_render_contract(
        render, views, gaussians, pipe, background, identity_indices
    )
    mip_filter_audit = declared_mip_filter(parent_contract, gaussians)
    zero_step_after = capture_render_contract(
        render, views, gaussians, pipe, background, identity_indices
    )
    zero_step_identity = compare_render_contract(
        zero_step_before, zero_step_after, atol=0.0, rtol=0.0
    )
    assert_render_contract(zero_step_identity)
    del zero_step_before, zero_step_after
    _restore_color_correction_state(
        parent_payload,
        color_correction=color_correction,
        color_correction_optimizer=color_optimizer,
    )
    _restore_rng_state(parent_payload["rng_state"])
    for parameter in color_correction.parameters():
        parameter.requires_grad_(False)
    color_correction.eval()
    del color_optimizer

    protected_count = int(len(gaussians.get_xyz))
    gaussians.mark_prefix_protected(protected_count)
    gaussians.freeze_prefix_gradients(protected_count)
    protected_parameters = _capture_protected_parameters(
        gaussians, protected_count
    )

    sky = CanonicalDirectionalSky(degree=args.sky_degree).cuda()
    sky_optimizer = torch.optim.Adam(sky.parameters(), lr=args.sky_lr)

    implementation_files = _sha256_files(
        [
            REPO_ROOT / "scripts" / "train_semantic_residual_2dgs.py",
            REPO_ROOT / "outdoor" / "task_fields.py",
            REPO_ROOT / "outdoor" / "directional_sky.py",
            REPO_ROOT / "matcha" / "cambridge_masks.py",
            REPO_ROOT / "matcha" / "cambridge_training.py",
            SURFEL_ROOT / "gaussian_renderer" / "__init__.py",
            SURFEL_ROOT / "scene" / "gaussian_model.py",
        ]
    )
    rasterizer_implementation = directory_sha256(
        SURFEL_ROOT / "submodules" / "diff-surfel-rasterization",
        suffixes={".cu", ".h", ".cpp", ".py"},
    )
    post_resume_schedule = {
        "stage_iterations": int(opt.iterations),
        "global_iteration_start_exclusive": parent_iteration,
        "global_iteration_end": global_iteration,
        "camera_schedule": schedule_audit,
        "loss": {
            "gaussian": "task_weighted_charbonnier_w_gaussian_rgb",
            "sky": "task_weighted_charbonnier_w_sky_rgb",
            "charbonnier_epsilon": args.charbonnier_epsilon,
            "sky_loss_weight": args.sky_loss_weight,
        },
        "topology": {
            "policy": "append_only_clone_split_no_prune_no_reset",
            "start_stage_iteration": args.topology_start,
            "interval": args.topology_interval,
            "minimum_distinct_schedule_views": args.minimum_persistent_views,
            "minimum_distinct_sequences": args.minimum_support_sequences,
            "minimum_camera_center_baseline": args.minimum_support_baseline,
            "distinct_view_evidence": (
                "area_normalized_screen_gradient_proxy_at_canopy_projected_center"
            ),
            "gradient_quantile_per_view": args.topology_grad_quantile,
            "canopy_grad_hit_threshold": args.canopy_grad_hit_threshold,
            "topology_grad_threshold": args.topology_grad_threshold,
            "total_append_cap": args.max_new_gaussians,
            "per_event_cap": args.max_new_per_event,
        },
    }
    residual_contract = {
        "protocol": RUN_SCHEMA_VERSION,
        "run_role": "protected_structure_semantic_residual_candidate",
        "resume_mode": "exact_full_state_restore_then_declared_schedule_fork",
        "trajectory_identity_claim": False,
        "parent": {
            "training_state": str(parent_state_path),
            "training_state_sha256": parent_state_sha256,
            "iteration": parent_iteration,
            "contract_sha256": _json_digest(parent_contract),
            "protected_gaussians": protected_count,
        },
        "input": input_audit,
        "runtime_storage": {
            "camera_rgb": str(dataset.data_device),
            "sample_transfer": "one_target_rgb_to_cuda_per_iteration",
            "protected_reference_snapshots": "cpu_chunked_bit_exact_audit",
        },
        "zero_step_renderer_identity": zero_step_identity,
        "mip_filter": mip_filter_audit,
        "rasterizer_implementation": rasterizer_implementation,
        "active_components": {
            "protected_structural_2d_surfels": True,
            "class_conditioned_foliage_2d_surfels": True,
            "true_volumetric_foliage": False,
            "canonical_directional_sky": True,
            "frozen_inherited_train_camera_affine": True,
            "charts": False,
            "planes": False,
            "see3d": False,
            "facade_atlas": False,
            "historical_full_real_rgb_initialization": False,
        },
        "semantic_fields": task_fields.audit(),
        "post_resume_schedule": post_resume_schedule,
        "post_resume_schedule_sha256": _json_digest(post_resume_schedule),
        "implementation_files": implementation_files,
        "implementation_sha256": _json_digest(implementation_files),
    }
    (model_path / "input_manifest.json").write_text(
        json.dumps(residual_contract, indent=2) + "\n", encoding="utf-8"
    )
    (model_path / "protected_base.json").write_text(
        json.dumps(residual_contract["parent"], indent=2) + "\n", encoding="utf-8"
    )
    cfg_values = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    cfg_values.update(
        {
            "source_path": str(source_path),
            "model_path": str(model_path),
            "use_color_correction": True,
            "semantic_residual_sky": True,
        }
    )
    (model_path / "cfg_args").write_text(
        str(argparse.Namespace(**cfg_values)) + "\n", encoding="utf-8"
    )

    gradient_sum = torch.zeros(len(gaussians.get_xyz), device="cuda")
    distinct_view_slots = torch.full(
        (len(gaussians.get_xyz), int(args.minimum_persistent_views)),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    appended_total = 0
    topology_events: list[dict] = []
    trace_path = model_path / "training_trace.jsonl"
    trace_path.touch()
    ema = 0.0
    start_time = time.time()
    progress = tqdm(range(1, int(opt.iterations) + 1), desc="semantic residual fork")

    for stage_iteration in progress:
        gaussians.update_learning_rate(stage_iteration)
        camera_index = int(schedule[stage_iteration - 1])
        view = views[camera_index]
        package = render(view, gaussians, pipe, background)
        raw_gaussian = package["render"]
        alpha = package["rend_alpha"]
        target = view.original_image.cuda()
        fields = task_fields.fields(
            view.image_name, raw_gaussian.shape[-2:], raw_gaussian.device
        )
        sky_rgb = sky(view)

        gaussian_composite = composite_white_background(
            raw_gaussian, alpha, sky_rgb.detach()
        )
        sky_composite = composite_white_background(
            raw_gaussian.detach(), alpha.detach(), sky_rgb
        )
        gaussian_prediction = apply_per_image_affine_color_correction(
            color_correction, gaussian_composite, view.image_name
        )
        sky_prediction = apply_per_image_affine_color_correction(
            color_correction, sky_composite, view.image_name
        )
        gaussian_loss = _weighted_charbonnier(
            gaussian_prediction,
            target,
            fields["w_gaussian_rgb"],
            epsilon=args.charbonnier_epsilon,
        )
        topology_loss = _weighted_charbonnier(
            gaussian_prediction,
            target,
            fields["w_topology"],
            epsilon=args.charbonnier_epsilon,
        )
        sky_loss = _weighted_charbonnier(
            sky_prediction,
            target,
            fields["w_sky_rgb"],
            epsilon=args.charbonnier_epsilon,
        )
        topology_gradient = torch.autograd.grad(
            topology_loss,
            package["viewspace_points"],
            retain_graph=True,
            allow_unused=True,
        )[0]
        loss = gaussian_loss + float(args.sky_loss_weight) * sky_loss
        loss.backward()

        with torch.no_grad():
            normalized_gradient_threshold = float(
                args.canopy_grad_hit_threshold
            )
            if topology_gradient is not None:
                norm = torch.nan_to_num(
                    torch.linalg.vector_norm(topology_gradient, dim=-1),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                projected_area = (
                    torch.pi
                    * package["radii"].float().clamp_min(1.0).square()
                )
                normalized_norm = norm / projected_area
                canopy_center = _projected_center_mask(
                    gaussians.get_xyz[: len(norm)],
                    view,
                    fields["p_canopy"],
                )
                threshold_population = normalized_norm[
                    package["visibility_filter"] & canopy_center
                ]
                if threshold_population.numel():
                    robust_threshold = torch.quantile(
                        threshold_population,
                        float(args.topology_grad_quantile),
                    )
                    normalized_gradient_threshold = max(
                        normalized_gradient_threshold,
                        float(robust_threshold.item()),
                        float(args.topology_grad_threshold),
                    )
                visible_hit = (
                    package["visibility_filter"]
                    & canopy_center
                    & (normalized_norm >= normalized_gradient_threshold)
                )
                gradient_sum[: len(norm)] += torch.where(
                    visible_hit, normalized_norm, torch.zeros_like(normalized_norm)
                )
                _record_distinct_view_hits(
                    distinct_view_slots[: len(norm)],
                    visible_hit,
                    camera_index,
                )

            gaussians.optimizer.step()
            sky_optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            sky_optimizer.zero_grad(set_to_none=True)
            gaussians.constrain_trainable_suffix(
                protected_count,
                max_opacity=max(args.opacity_ceiling * 3.0, args.opacity_ceiling),
            )
            gaussians.clear_prefix_optimizer_state(protected_count)

            should_allocate = (
                stage_iteration >= args.topology_start
                and stage_iteration % args.topology_interval == 0
                and appended_total < args.max_new_gaussians
            )
            if should_allocate:
                valid_slots = distinct_view_slots >= 0
                view_hits = valid_slots.sum(dim=1)
                safe_slots = distinct_view_slots.clamp_min(0).long()
                slot_sequences = camera_sequence_codes[safe_slots]
                sequence_count = torch.zeros_like(view_hits)
                for slot_index in range(distinct_view_slots.shape[1]):
                    novel = valid_slots[:, slot_index]
                    if slot_index:
                        earlier_match = (
                            valid_slots[:, :slot_index]
                            & (
                                slot_sequences[:, :slot_index]
                                == slot_sequences[:, slot_index : slot_index + 1]
                            )
                        ).any(dim=1)
                        novel = novel & ~earlier_match
                    sequence_count += novel.to(sequence_count.dtype)
                maximum_baseline = torch.zeros(
                    len(distinct_view_slots), device="cuda"
                )
                for first in range(distinct_view_slots.shape[1]):
                    for second in range(first + 1, distinct_view_slots.shape[1]):
                        valid_pair = (
                            valid_slots[:, first] & valid_slots[:, second]
                        )
                        pair_baseline = torch.linalg.vector_norm(
                            camera_centers[safe_slots[:, first]]
                            - camera_centers[safe_slots[:, second]],
                            dim=1,
                        )
                        maximum_baseline = torch.maximum(
                            maximum_baseline,
                            torch.where(
                                valid_pair,
                                pair_baseline,
                                torch.zeros_like(pair_baseline),
                            ),
                        )
                eligible = (
                    (view_hits >= int(args.minimum_persistent_views))
                    & (sequence_count >= int(args.minimum_support_sequences))
                    & (maximum_baseline >= float(args.minimum_support_baseline))
                )
                average_gradient = gradient_sum / view_hits.clamp_min(1).float()
                event_cap = min(
                    int(args.max_new_per_event),
                    int(args.max_new_gaussians - appended_total),
                )
                split_cap = int(event_cap * float(args.split_fraction))
                split_cap -= split_cap % int(args.children_per_parent)
                clone_cap = event_cap - split_cap
                block_id = len(topology_events)
                before = int(len(gaussians.get_xyz))
                cloned = gaussians.densify_and_clone_limited(
                    average_gradient[:, None],
                    float(args.topology_grad_threshold),
                    scene.cameras_extent,
                    clone_cap,
                    opacity_ceiling=args.opacity_ceiling,
                    eligibility_mask=eligible,
                    primitive_class=GaussianModel.PRIMITIVE_FOLIAGE,
                    source_type=GaussianModel.SOURCE_CANOPY_RESIDUAL,
                    geometry_confidence=args.foliage_geometry_confidence,
                    protected_flag=False,
                    block_id=block_id,
                )
                split = gaussians.densify_and_split_limited(
                    average_gradient[:, None],
                    float(args.topology_grad_threshold),
                    scene.cameras_extent,
                    split_cap,
                    children_per_parent=args.children_per_parent,
                    opacity_ceiling=args.opacity_ceiling,
                    max_parent_scale_fraction=args.max_parent_scale_fraction,
                    eligibility_mask=eligible,
                    primitive_class=GaussianModel.PRIMITIVE_FOLIAGE,
                    source_type=GaussianModel.SOURCE_CANOPY_RESIDUAL,
                    geometry_confidence=args.foliage_geometry_confidence,
                    protected_flag=False,
                    block_id=block_id,
                )
                appended = int(cloned + split)
                appended_total += appended
                event = {
                    "stage_iteration": stage_iteration,
                    "global_iteration": parent_iteration + stage_iteration,
                    "eligible_persistent_gaussians": int(eligible.sum().item()),
                    "view_hits_max": int(view_hits.max().item()),
                    "support_sequences_max": int(sequence_count.max().item()),
                    "support_baseline_max": float(maximum_baseline.max().item()),
                    "clone_children": int(cloned),
                    "split_children": int(split),
                    "appended": appended,
                    "gaussians_before": before,
                    "gaussians_after": int(len(gaussians.get_xyz)),
                }
                topology_events.append(event)
                with (model_path / "topology_events.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(json.dumps(event) + "\n")
                gradient_sum = torch.zeros(len(gaussians.get_xyz), device="cuda")
                distinct_view_slots = torch.full(
                    (
                        len(gaussians.get_xyz),
                        int(args.minimum_persistent_views),
                    ),
                    -1,
                    dtype=torch.int32,
                    device="cuda",
                )

            verify_prefix = (
                should_allocate
                or stage_iteration == 1
            )
            if verify_prefix and not _protected_parameters_are_bit_exact(
                gaussians, protected_parameters
            ):
                raise RuntimeError("Protected 60k Gaussian prefix changed")

            ema = 0.4 * float(loss.item()) + 0.6 * ema
            if stage_iteration == 1 or stage_iteration % args.log_every == 0:
                row = {
                    "stage_iteration": stage_iteration,
                    "global_iteration": parent_iteration + stage_iteration,
                    "camera": view.image_name,
                    "gaussian_loss": float(gaussian_loss.item()),
                    "topology_loss": float(topology_loss.item()),
                    "sky_loss": float(sky_loss.item()),
                    "total_loss": float(loss.item()),
                    "ema_total_loss": ema,
                    "gaussians": int(len(gaussians.get_xyz)),
                    "appended_total": appended_total,
                    "canopy_pixel_fraction": float(fields["p_canopy"].mean().item()),
                    "sky_pixel_fraction": float(fields["p_sky"].mean().item()),
                    "elapsed_sec": time.time() - start_time,
                }
                with trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                progress.set_postfix(
                    loss=f"{ema:.5f}",
                    points=int(len(gaussians.get_xyz)),
                    added=appended_total,
                )

    if not _protected_parameters_are_bit_exact(
        gaussians, protected_parameters
    ):
        raise RuntimeError("Protected 60k Gaussian prefix changed before save")
    scene.save(global_iteration)
    checkpoint_dir = model_path / "point_cloud" / f"iteration_{global_iteration}"
    save_per_image_affine_color_correction(
        color_correction, checkpoint_dir / "color_correction.pth"
    )
    sky.save(checkpoint_dir / "sky_model.pth")
    class_counts = {
        "structural": int(
            (gaussians.get_primitive_class == GaussianModel.PRIMITIVE_STRUCTURAL)
            .sum()
            .item()
        ),
        "foliage": int(
            (gaussians.get_primitive_class == GaussianModel.PRIMITIVE_FOLIAGE)
            .sum()
            .item()
        ),
        "sky_gaussians": int(
            (gaussians.get_primitive_class == GaussianModel.PRIMITIVE_SKY)
            .sum()
            .item()
        ),
    }
    final_state = _save_residual_state(
        model_path / "training_state" / f"iteration_{global_iteration}.pth",
        global_iteration=global_iteration,
        gaussians=gaussians,
        sky=sky,
        sky_optimizer=sky_optimizer,
        color_correction=color_correction,
        contract=residual_contract,
        trace_state={"ema": ema, "appended_total": appended_total},
    )
    result = {
        "protocol": RUN_SCHEMA_VERSION,
        "global_iteration": global_iteration,
        "protected_parent_gaussians": protected_count,
        "final_gaussians": int(len(gaussians.get_xyz)),
        "class_counts": class_counts,
        "topology_events": topology_events,
        "protected_prefix_bit_exact": True,
        "sky": sky.audit(),
        "frozen_inherited_color_correction": True,
        "point_cloud": str((checkpoint_dir / "point_cloud.ply").resolve()),
        "sky_model": str((checkpoint_dir / "sky_model.pth").resolve()),
        "training_state": str(final_state.resolve()),
        "elapsed_sec": time.time() - start_time,
    }
    (model_path / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
