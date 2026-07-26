#!/usr/bin/env python
"""Distill a mixed outdoor teacher into one canonical standard 3DGS PLY."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SURFEL_ROOT = REPO_ROOT / "2d-gaussian-splatting"
sys.path[:0] = [str(REPO_ROOT), str(SURFEL_ROOT)]

from arguments import ModelParams, PipelineParams  # noqa: E402
from outdoor.appearance_uncertainty import (  # noqa: E402
    OutdoorAppearanceUncertainty,
)
from outdoor.directional_sky import (  # noqa: E402
    CanonicalDirectionalSky,
    composite_white_background,
)
from outdoor.evidence_store import load_evidence_store  # noqa: E402
from outdoor.hybrid_gaussian_renderer import (  # noqa: E402
    VolumetricFoliageModel,
    render_hybrid,
)
from outdoor.lazy_scene import LazyScene  # noqa: E402
from outdoor.runtime_provenance import collect_runtime_provenance  # noqa: E402
from outdoor.standard_3dgs import (  # noqa: E402
    STANDARD_3DGS_VERSION,
    STUDENT_ROLE_CROWN,
    STUDENT_ROLE_DYNAMIC_BAKED,
    STUDENT_ROLE_RIGID,
    STUDENT_ROLE_SKELETON,
    STUDENT_ROLE_SKY,
    canonical_student_seed,
    save_standard_3dgs_ply,
    validate_standard_3dgs_ply,
)
from outdoor.task_fields import OutdoorTaskFieldLookup  # noqa: E402
from scene import GaussianModel  # noqa: E402
from scene.cameras import Camera  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402
from utils.point_utils import depth_to_normal  # noqa: E402


PROTOCOL = "role_aware_standard_3dgs_optimization_v2"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    PipelineParams(parser)
    parser.set_defaults(data_device="cpu", white_background=True)
    parser.add_argument("--teacher-state", type=Path, required=True)
    parser.add_argument("--evidence-store", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--maximum-surface-gaussians", type=int, default=1_500_000)
    parser.add_argument("--sky-gaussians", type=int, default=8192)
    parser.add_argument("--position-lr", type=float, default=5e-5)
    parser.add_argument("--feature-lr", type=float, default=1e-3)
    parser.add_argument("--opacity-lr", type=float, default=4e-3)
    parser.add_argument("--scale-lr", type=float, default=3e-4)
    parser.add_argument("--rotation-lr", type=float, default=2e-4)
    parser.add_argument("--novel-every", type=int, default=0)
    parser.add_argument("--densify-from-iter", type=int, default=500)
    parser.add_argument("--densify-until-fraction", type=float, default=0.65)
    parser.add_argument("--densification-interval", type=int, default=500)
    parser.add_argument("--maximum-gaussians", type=int, default=2_000_000)
    parser.add_argument("--maximum-splits-per-event", type=int, default=8_000)
    parser.add_argument("--split-radius", type=float, default=7.0)
    parser.add_argument("--split-gradient-threshold", type=float, default=2e-5)
    parser.add_argument("--prune-opacity", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--view-cache-size", type=int, default=4)
    parser.add_argument("--image-prefetch-workers", type=int, default=2)
    parser.add_argument("--image-prefetch-depth", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    return args, model.extract(args)


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _restore_surface_inference(model: GaussianModel, capture) -> None:
    (
        active,
        xyz,
        features_dc,
        features_rest,
        scaling,
        rotation,
        opacity,
    ) = capture[:7]
    model.active_sh_degree = int(active)
    model._xyz = torch.nn.Parameter(xyz.detach().cuda())
    model._features_dc = torch.nn.Parameter(features_dc.detach().cuda())
    model._features_rest = torch.nn.Parameter(
        features_rest.detach().cuda()
    )
    model._scaling = torch.nn.Parameter(scaling.detach().cuda())
    model._rotation = torch.nn.Parameter(rotation.detach().cuda())
    model._opacity = torch.nn.Parameter(opacity.detach().cuda())
    metadata = capture[12] if len(capture) >= 13 else None
    model._restore_point_metadata(metadata)


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


def _student_from_seed(seed: dict, sh_degree: int):
    student = VolumetricFoliageModel(
        sh_degree, dynamic_rank=1
    ).cuda()
    count = len(seed["xyz"])
    payload = {
        "sh_degree": sh_degree,
        "dynamic_rank": 1,
        **seed,
        "deformation_basis": torch.zeros(count, 1, 3),
        "dynamic_feature_basis": torch.zeros(count, 1, 3),
        "dynamic_opacity_basis": torch.zeros(count, 1, 1),
    }
    student.restore(payload)
    return student


def _optimizer(args, student):
    return torch.optim.Adam(
        [
            {
                "params": [student.xyz],
                "lr": args.position_lr,
                "name": "xyz",
            },
            {
                "params": [student.features],
                "lr": args.feature_lr,
                "name": "features",
            },
            {
                "params": [student.opacity_logits],
                "lr": args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [student.log_scales],
                "lr": args.scale_lr,
                "name": "scale",
            },
            {
                "params": [student.quaternions],
                "lr": args.rotation_lr,
                "name": "rotation",
            },
        ],
        eps=1e-15,
    )


def _migrate_optimizer(args, student, previous, new_to_old):
    current = _optimizer(args, student)
    old_groups = {group["name"]: group for group in previous.param_groups}
    for group in current.param_groups:
        old_group = old_groups[group["name"]]
        old_parameter = old_group["params"][0]
        new_parameter = group["params"][0]
        state = previous.state.get(old_parameter)
        if not state:
            continue
        migrated = {}
        for key, value in state.items():
            if (
                torch.is_tensor(value)
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


def _jitter_camera(view, rng, scene_extent, step):
    translation = np.asarray(view.T).copy()
    translation += rng.normal(
        0.0, 0.003 * float(scene_extent), size=3
    )
    return Camera(
        colmap_id=-1,
        R=np.asarray(view.R).copy(),
        T=translation,
        FoVx=view.FoVx,
        FoVy=view.FoVy,
        image=torch.zeros(
            3, view.image_height, view.image_width
        ),
        gt_alpha_mask=None,
        image_name=f"distill_novel_{step:06d}",
        uid=-1,
        data_device="cpu",
        fx=view.focal_x,
        fy=view.focal_y,
        cx=view.cx,
        cy=view.cy,
        zfar=view.zfar,
    )


def _mean(value, weight):
    if weight.ndim == 2:
        weight = weight[None]
    return (value * weight).sum() / (
        weight.sum() * value.shape[0]
    ).clamp_min(1)


def _gradient_loss(prediction, target, weight):
    horizontal_weight = torch.minimum(weight[:, 1:], weight[:, :-1])
    vertical_weight = torch.minimum(weight[1:, :], weight[:-1, :])
    horizontal = (
        prediction[:, :, 1:]
        - prediction[:, :, :-1]
        - target[:, :, 1:]
        + target[:, :, :-1]
    ).abs().mean(0)
    vertical = (
        prediction[:, 1:, :]
        - prediction[:, :-1, :]
        - target[:, 1:, :]
        + target[:, :-1, :]
    ).abs().mean(0)
    return (
        (horizontal * horizontal_weight).sum()
        / horizontal_weight.sum().clamp_min(1)
        + (vertical * vertical_weight).sum()
        / vertical_weight.sum().clamp_min(1)
    )


def _masked_ssim_loss(prediction, target, weight, window_size=11):
    prediction = prediction[None]
    target = target[None]
    weight = weight[None, None]
    padding = window_size // 2
    mu_prediction = F.avg_pool2d(
        prediction, window_size, 1, padding
    )
    mu_target = F.avg_pool2d(target, window_size, 1, padding)
    variance_prediction = F.avg_pool2d(
        prediction.square(), window_size, 1, padding
    ) - mu_prediction.square()
    variance_target = F.avg_pool2d(
        target.square(), window_size, 1, padding
    ) - mu_target.square()
    covariance = F.avg_pool2d(
        prediction * target, window_size, 1, padding
    ) - mu_prediction * mu_target
    similarity = (
        (2 * mu_prediction * mu_target + 0.01**2)
        * (2 * covariance + 0.03**2)
        / (
            (mu_prediction.square() + mu_target.square() + 0.01**2)
            * (variance_prediction + variance_target + 0.03**2)
        ).clamp_min(1e-8)
    ).mean(1, keepdim=True)
    valid = -F.max_pool2d(-weight, window_size, 1, padding)
    valid[..., :padding, :] = 0
    valid[..., -padding:, :] = 0
    valid[..., :, :padding] = 0
    valid[..., :, -padding:] = 0
    return ((1 - similarity) * valid).sum() / valid.sum().clamp_min(1)


def _topology_stats(student):
    count = len(student)
    device = student.xyz.device
    return {
        "gradient": torch.zeros(count, device=device),
        "gradient_count": torch.zeros(count, device=device),
        "radius": torch.zeros(count, device=device),
        "contribution": torch.zeros(count, device=device),
        "residual": torch.zeros(count, device=device),
    }


@torch.no_grad()
def _accumulate_student_stats(stats, package):
    radii = package.radii.float()
    visible = radii > 0
    stats["radius"] = torch.maximum(stats["radius"], radii)
    if package.volume_means2d is not None and package.volume_means2d.grad is not None:
        gradient = torch.nan_to_num(
            package.volume_means2d.grad[:, :2]
        ).norm(dim=-1)
        stats["gradient"][visible] += gradient[visible]
        stats["gradient_count"][visible] += 1
    if package.responsibility is not None:
        responsibility = package.responsibility
        stats["contribution"] += responsibility[:, 0]
        if responsibility.shape[1] >= 4:
            stats["residual"] += responsibility[:, 3]


@torch.no_grad()
def _adapt_student_topology(args, student, stats):
    old_count = len(student)
    role = student.primitive_role
    protected = (role == STUDENT_ROLE_SKY) | (
        role == STUDENT_ROLE_SKELETON
    )
    remove = (student.opacities < args.prune_opacity) & ~protected
    pruned, kept_old = student.prune(remove)
    working = {name: value[kept_old] for name, value in stats.items()}
    role = student.primitive_role
    gradient = (
        working["gradient"]
        / working["gradient_count"].clamp_min(1)
    )
    residual = (
        working["residual"]
        / working["contribution"].clamp_min(1e-8)
    )
    eligible = (
        (role != STUDENT_ROLE_SKY)
        & (role != STUDENT_ROLE_SKELETON)
        & (working["radius"] >= args.split_radius)
        & (gradient >= args.split_gradient_threshold)
        & (working["contribution"] > 0)
    )
    capacity = min(
        args.maximum_splits_per_event,
        max(args.maximum_gaussians - len(student), 0),
    )
    split_parents = 0
    children = 0
    final_to_kept = torch.arange(len(student), device=student.xyz.device)
    if bool(eligible.any()) and capacity > 0:
        indices = torch.nonzero(eligible, as_tuple=False).flatten()
        score = gradient[indices] * residual[indices]
        chosen = indices[
            torch.topk(score, min(capacity, len(indices))).indices
        ]
        event = student.split(chosen)
        split_parents = int(event["split_parents"])
        children = int(event["children"])
        final_to_kept = event["_new_to_old"]
    final_to_old = kept_old[final_to_kept]
    return {
        "old_count": old_count,
        "new_count": len(student),
        "pruned": int(pruned),
        "split_parents": split_parents,
        "children": children,
        "_new_to_old": (
            final_to_old
            if pruned or split_parents
            else None
        ),
    }


def main():
    args, dataset = _parse_args()
    output = Path(dataset.model_path).resolve()
    resume = _load(args.resume) if args.resume else None
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    store = load_evidence_store(args.evidence_store)
    teacher_state = _load(args.teacher_state)
    if teacher_state["evidence_hash"] != store["evidence_hash"]:
        raise RuntimeError("Teacher/evidence hash mismatch")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    teacher_surface = GaussianModel(dataset.sh_degree)
    scene = LazyScene(
        dataset,
        teacher_surface,
        image_cache_size=args.view_cache_size,
        image_prefetch_workers=args.image_prefetch_workers,
    )
    _restore_surface_inference(
        teacher_surface, teacher_state["surface"]
    )
    views = scene.getTrainCameras()
    teacher_foliage = VolumetricFoliageModel(
        dataset.sh_degree,
        dynamic_rank=int(
            teacher_state["foliage"].get("dynamic_rank", 4)
        ),
    ).cuda()
    teacher_foliage.restore(teacher_state["foliage"])
    sky = CanonicalDirectionalSky(
        degree=int(teacher_state.get("sky_degree", 2))
    ).cuda()
    sky.load_state_dict(teacher_state["sky"])
    appearance = OutdoorAppearanceUncertainty(
        [view.image_name for view in views],
        rank=teacher_foliage.dynamic_rank,
        spatial_grid_size=24,
        device="cuda",
    )
    appearance.restore(teacher_state["appearance"])
    for parameter in (
        teacher_surface._xyz,
        teacher_surface._features_dc,
        teacher_surface._features_rest,
        teacher_surface._scaling,
        teacher_surface._rotation,
        teacher_surface._opacity,
    ):
        parameter.requires_grad_(False)
    for parameter in teacher_foliage.parameters():
        parameter.requires_grad_(False)
    for parameter in sky.parameters():
        parameter.requires_grad_(False)
    centers = torch.stack([view.camera_center for view in views])
    if resume is None:
        temporal_codes = torch.stack(
            [
                appearance.temporal_code(view.image_name).detach()
                for view in views
            ]
        )
        seed, initialization_audit = canonical_student_seed(
            teacher_surface,
            teacher_foliage,
            sky,
            centers,
            maximum_surface_gaussians=args.maximum_surface_gaussians,
            sky_gaussians=args.sky_gaussians,
            temporal_codes=temporal_codes,
        )
        student = _student_from_seed(seed, dataset.sh_degree)
        start = 0
        topology_stats = _topology_stats(student)
        topology_events = []
    else:
        if resume.get("protocol") != PROTOCOL:
            raise RuntimeError("Student resume protocol mismatch")
        if resume.get("evidence_hash") != store["evidence_hash"]:
            raise RuntimeError("Student resume/evidence hash mismatch")
        student = VolumetricFoliageModel(
            dataset.sh_degree, dynamic_rank=1
        ).cuda()
        student.restore(resume["student"])
        initialization_audit = resume["initialization_audit"]
        start = int(resume["iteration"])
        topology_stats = {
            name: value.cuda()
            for name, value in resume["topology_stats"].items()
        }
        topology_events = list(resume.get("topology_events", []))
    optimizer = _optimizer(args, student)
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
    empty = _empty_surface(dataset.sh_degree)
    semantic = json.loads(
        Path(store["semantic_contract"]).read_text(encoding="utf-8")
    )
    fields = OutdoorTaskFieldLookup(
        Path(dataset.source_path),
        Path(semantic["tree_mask_pickle"]),
        Path(store["semantic_contract"]),
        max_cached_views=0,
    )
    background = torch.ones(3, device="cuda")
    schedule_rng = np.random.default_rng(args.seed + 1)
    schedule = []
    while len(schedule) < args.iterations:
        schedule.extend(schedule_rng.permutation(len(views)).tolist())
    schedule = np.asarray(schedule[: args.iterations], dtype=np.int64)
    schedule_hash = hashlib.sha256(
        np.ascontiguousarray(schedule).view(np.uint8)
    ).hexdigest()
    implementation_hashes = {
        "distiller": _file_sha256(Path(__file__)),
        "standard_3dgs": _file_sha256(
            REPO_ROOT / "outdoor/standard_3dgs.py"
        ),
        "hybrid_renderer": _file_sha256(
            REPO_ROOT / "outdoor/hybrid_gaussian_renderer.py"
        ),
        "lazy_scene": _file_sha256(
            REPO_ROOT / "outdoor/lazy_scene.py"
        ),
        "dataset_reader": _file_sha256(
            SURFEL_ROOT / "scene/dataset_readers.py"
        ),
        "mixed_forward_cuda": _file_sha256(
            SURFEL_ROOT
            / "submodules/diff-surfel-rasterization/"
            "cuda_rasterizer/forward.cu"
        ),
    }
    runtime_provenance = collect_runtime_provenance(
        REPO_ROOT,
        python_modules=(
            "scripts.distill_standard_3dgs",
            "outdoor.hybrid_gaussian_renderer",
            "outdoor.standard_3dgs",
            "diff_surfel_rasterization",
        ),
        extension_roots=(
            SURFEL_ROOT / "submodules/diff-surfel-rasterization",
            SURFEL_ROOT / "submodules/simple-knn",
        ),
    )
    jitter_rng = np.random.default_rng(args.seed + 2)
    if resume is not None:
        if resume.get("schedule_hash") != schedule_hash:
            raise RuntimeError("Student resume camera schedule changed")
        if resume.get("implementation_hashes") != implementation_hashes:
            raise RuntimeError(
                "Student resume implementation hash mismatch; start a new export"
            )
        random.setstate(resume["python_rng_state"])
        np.random.set_state(resume["numpy_rng_state"])
        torch.set_rng_state(resume["torch_rng_state"])
        torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
        jitter_rng.bit_generator.state = resume["jitter_rng_state"]
    trace = output / "distillation_trace.jsonl"
    progress = tqdm(
        range(start, args.iterations),
        initial=start,
        total=args.iterations,
        desc="standard 3DGS distillation",
    )

    def prefetch_training_images(begin: int) -> None:
        requested = []
        end = min(args.iterations, begin + args.image_prefetch_depth)
        for future_step in range(begin, end):
            if (
                args.novel_every <= 0
                or (future_step + 1) % args.novel_every != 0
            ):
                requested.append(
                    views[int(schedule[future_step])]
                )
        scene.prefetch_images(requested)

    prefetch_training_images(start)
    for step in progress:
        real_view = views[int(schedule[step])]
        novel = (
            args.novel_every > 0
            and (step + 1) % args.novel_every == 0
        )
        view = (
            _jitter_camera(
                real_view, jitter_rng, scene.cameras_extent, step
            )
            if novel
            else real_view
        )
        prefetch_training_images(step + 1)
        task = None
        target = None
        if not novel:
            task = fields.fields(
                real_view.image_name,
                (
                    real_view.image_height,
                    real_view.image_width,
                ),
                torch.device("cuda"),
            )
            target = real_view.original_image.cuda(non_blocking=True)
        with torch.no_grad():
            teacher_package = render_hybrid(
                view,
                teacher_surface,
                teacher_foliage,
                background=background,
                temporal_code=(
                    appearance.temporal_code(real_view.image_name)
                    if not novel
                    else None
                ),
                include_dynamic=not novel,
            )
            teacher_rgb = composite_white_background(
                teacher_package.render,
                teacher_package.alpha,
                sky(view),
            ).clamp(0, 1)
            if not novel:
                teacher_rgb = appearance(
                    teacher_rgb, real_view, task
                ).clamp(0, 1)
        student_package = render_hybrid(
            view,
            empty,
            student,
            background=background,
            include_dynamic=False,
        )
        prediction = student_package.render.clamp(0, 1)
        if not novel:
            rigid = task["p_rigid"] * (1.0 - task["p_transient"])
            crown = task["p_canopy"] * (1.0 - task["p_transient"])
            sky_weight = task["p_sky"] * (1.0 - task["p_transient"])
            teacher_rgb_weight = (
                rigid + 0.35 * crown + sky_weight
            ).clamp(0, 1)
        else:
            rigid = (teacher_package.surface_alpha[0] > 0.05).float()
            crown = (teacher_package.volume_alpha[0] > 0.05).float()
            sky_weight = torch.zeros_like(rigid)
            teacher_rgb_weight = torch.ones_like(rigid)
        teacher_rgb_loss = _mean(
            (prediction - teacher_rgb).abs(), teacher_rgb_weight
        )
        foreground = (rigid + crown).clamp(0, 1)
        alpha_loss = _mean(
            (
                student_package.alpha
                - teacher_package.alpha
            ).abs(),
            foreground,
        )
        depth_valid = (
            (teacher_package.alpha > 0.05)
            & (teacher_package.depth > 0)
            & (student_package.depth > 0)
        ).float() * rigid[None]
        depth_loss = _mean(
            torch.log1p(
                (
                    student_package.depth
                    - teacher_package.depth
                ).abs()
                / teacher_package.depth.clamp_min(0.1)
            ),
            depth_valid[0],
        )
        student_normal = depth_to_normal(
            view, student_package.depth
        ).permute(2, 0, 1)
        student_normal = F.normalize(
            student_normal, dim=0, eps=1e-6
        )
        teacher_normal = F.normalize(
            teacher_package.normal_world, dim=0, eps=1e-6
        )
        normal_loss = _mean(
            1.0 - (student_normal * teacher_normal).sum(0).abs(),
            depth_valid[0] * rigid,
        )
        gt_rigid = prediction.new_zeros(())
        gt_ssim = prediction.new_zeros(())
        gt_edge = prediction.new_zeros(())
        gt_crown_low = prediction.new_zeros(())
        gt_sky = prediction.new_zeros(())
        if not novel:
            gt_rigid = _mean(
                (prediction - target).abs(), rigid
            )
            gt_ssim = _masked_ssim_loss(
                prediction, target, rigid
            )
            gt_edge = _gradient_loss(
                prediction,
                target,
                rigid * (1.0 - task["p_boundary_uncertain"]),
            )
            low_prediction = F.avg_pool2d(
                prediction[None], 9, 1, 4
            )[0]
            low_target = F.avg_pool2d(target[None], 9, 1, 4)[0]
            gt_crown_low = _mean(
                (low_prediction - low_target).abs(), crown
            )
            gt_sky = _mean(
                (prediction - target).abs(), sky_weight
            )
            loss = (
                0.70 * gt_rigid
                + 0.20 * gt_ssim
                + 0.12 * gt_edge
                + 0.25 * gt_crown_low
                + 0.30 * gt_sky
                + 0.20 * teacher_rgb_loss
                + 0.10 * depth_loss
                + 0.03 * normal_loss
                + 0.04 * alpha_loss
            )
        else:
            loss = (
                teacher_rgb_loss
                + 0.08 * depth_loss
                + 0.02 * normal_loss
                + 0.04 * alpha_loss
            )
        loss.backward()
        _accumulate_student_stats(topology_stats, student_package)
        gradient_audit = {
            name: float(
                torch.nan_to_num(parameter.grad).norm()
                if parameter.grad is not None
                else 0.0
            )
            for name, parameter in (
                ("xyz", student.xyz),
                ("features", student.features),
                ("opacity", student.opacity_logits),
                ("scale", student.log_scales),
                ("rotation", student.quaternions),
            )
        }
        if step == 0 and (
            gradient_audit["xyz"] <= 0
            or gradient_audit["features"] <= 0
        ):
            raise RuntimeError("Standard 3DGS received no xyz/feature gradient")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            student.quaternions.copy_(
                F.normalize(student.quaternions, dim=-1)
            )
            student.opacity_logits.clamp_(-12, 5)
            student.log_scales.clamp_(-12, 5)
        topology_event = None
        topology_end = int(
            args.iterations * args.densify_until_fraction
        )
        if (
            not novel
            and step + 1 >= args.densify_from_iter
            and step + 1 <= topology_end
            and args.densification_interval > 0
            and (step + 1) % args.densification_interval == 0
        ):
            with torch.no_grad():
                topology_package = render_hybrid(
                    view,
                    empty,
                    student,
                    background=background,
                    include_dynamic=False,
                    audit_fields=torch.stack(
                        [
                            rigid,
                            crown,
                            (prediction.detach() - target).abs().mean(0),
                        ]
                    ),
                )
                _accumulate_student_stats(
                    topology_stats, topology_package
                )
                topology_event = _adapt_student_topology(
                    args, student, topology_stats
                )
                new_to_old = topology_event.pop("_new_to_old")
                if new_to_old is not None:
                    optimizer = _migrate_optimizer(
                        args,
                        student,
                        optimizer,
                        new_to_old,
                    )
                topology_events.append(
                    {"iteration": step + 1, **topology_event}
                )
                topology_stats = _topology_stats(student)
        if step == 0 or (step + 1) % 100 == 0:
            row = {
                "iteration": step + 1,
                "novel_camera": novel,
                "loss": float(loss),
                "teacher_rgb": float(teacher_rgb_loss),
                "gt_rigid": float(gt_rigid),
                "gt_masked_ssim": float(gt_ssim),
                "gt_rigid_edge": float(gt_edge),
                "gt_crown_low": float(gt_crown_low),
                "gt_sky": float(gt_sky),
                "depth": float(depth_loss),
                "normal": float(normal_loss),
                "alpha": float(alpha_loss),
                "gaussians": len(student),
                "gradient_norms": gradient_audit,
                "topology_event": topology_event,
            }
            with trace.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            progress.set_postfix(loss=f"{float(loss):.4f}")
        if (
            args.checkpoint_every > 0
            and (step + 1) % args.checkpoint_every == 0
        ):
            torch.save(
                {
                    "protocol": PROTOCOL,
                    "iteration": step + 1,
                    "student": student.capture(),
                    "optimizer": optimizer.state_dict(),
                    "initialization_audit": initialization_audit,
                    "evidence_hash": store["evidence_hash"],
                    "schedule_hash": schedule_hash,
                    "implementation_hashes": implementation_hashes,
                    "runtime_provenance": runtime_provenance,
                    "topology_stats": topology_stats,
                    "topology_events": topology_events,
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all(),
                    "jitter_rng_state": jitter_rng.bit_generator.state,
                },
                output / "student_checkpoint.pth",
            )
    ply_path = (
        output
        / "point_cloud"
        / f"iteration_{args.iterations}"
        / "point_cloud.ply"
    )
    save_standard_3dgs_ply(
        ply_path,
        xyz=student.xyz,
        log_scales=student.log_scales,
        quaternions=student.normalized_quaternions,
        opacity_logits=student.opacity_logits,
        features=student.features,
        sh_degree=dataset.sh_degree,
    )
    schema = validate_standard_3dgs_ply(
        ply_path, sh_degree=dataset.sh_degree
    )
    scene.close()
    cameras_source = output / "cameras.json"
    cameras_source.write_text(
        (Path(dataset.model_path) / "cameras.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    result = {
        "protocol": PROTOCOL,
        "version": STANDARD_3DGS_VERSION,
        "mode": "canonical",
        "teacher_state": str(args.teacher_state.resolve()),
        "evidence_hash": store["evidence_hash"],
        "initialization": initialization_audit,
        "iterations": args.iterations,
        "point_cloud": str(ply_path),
        "cameras": str(cameras_source),
        "schema": schema,
        "implementation_hashes": implementation_hashes,
        "runtime_provenance": runtime_provenance,
        "topology_events": topology_events,
        "density_control": {
            "from_iteration": args.densify_from_iter,
            "until_fraction": args.densify_until_fraction,
            "interval": args.densification_interval,
            "maximum_gaussians": args.maximum_gaussians,
        },
        "requires_custom_cuda": False,
        "requires_uv_gate": False,
        "requires_temporal_code": False,
        "requires_separate_sky": False,
        "requires_mixed_renderer": False,
    }
    (output / "distillation_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
