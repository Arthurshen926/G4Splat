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
from outdoor.standard_3dgs import (  # noqa: E402
    STANDARD_3DGS_VERSION,
    canonical_student_seed,
    save_standard_3dgs_ply,
    validate_standard_3dgs_ply,
)
from outdoor.task_fields import OutdoorTaskFieldLookup  # noqa: E402
from scene import GaussianModel  # noqa: E402
from scene.cameras import Camera  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402
from utils.point_utils import depth_to_normal  # noqa: E402


PROTOCOL = "mixed_teacher_render_space_standard_3dgs_distillation_v1"


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
    parser.add_argument("--novel-every", type=int, default=8)
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
            {"params": [student.xyz], "lr": args.position_lr},
            {"params": [student.features], "lr": args.feature_lr},
            {"params": [student.opacity_logits], "lr": args.opacity_lr},
            {"params": [student.log_scales], "lr": args.scale_lr},
            {"params": [student.quaternions], "lr": args.rotation_lr},
        ],
        eps=1e-15,
    )


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
        seed, initialization_audit = canonical_student_seed(
            teacher_surface,
            teacher_foliage,
            sky,
            centers,
            maximum_surface_gaussians=args.maximum_surface_gaussians,
            sky_gaussians=args.sky_gaussians,
        )
        student = _student_from_seed(seed, dataset.sh_degree)
        start = 0
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
        with torch.no_grad():
            teacher_package = render_hybrid(
                view,
                teacher_surface,
                teacher_foliage,
                background=background,
                include_dynamic=False,
            )
            teacher_rgb = composite_white_background(
                teacher_package.render,
                teacher_package.alpha,
                sky(view),
            ).clamp(0, 1)
        student_package = render_hybrid(
            view,
            empty,
            student,
            background=background,
            include_dynamic=False,
        )
        prediction = student_package.render.clamp(0, 1)
        teacher_rgb_loss = (prediction - teacher_rgb).abs().mean()
        alpha_loss = (
            student_package.alpha - teacher_package.alpha
        ).abs().mean()
        depth_valid = (
            (teacher_package.alpha > 0.05)
            & (teacher_package.depth > 0)
            & (student_package.depth > 0)
        ).float()
        depth_loss = _mean(
            (
                student_package.depth
                - teacher_package.depth
            ).abs()
            / teacher_package.depth.clamp_min(0.1),
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
            depth_valid[0],
        )
        gt_loss = prediction.new_zeros(())
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
            static_weight = (
                task["p_rigid"]
                + 0.20
                * task["p_canopy"]
                * (1.0 - task["p_transient"])
            ).clamp(0, 1)
            gt_loss = _mean(
                (prediction - target).abs(), static_weight
            )
        loss = (
            0.55 * teacher_rgb_loss
            + 0.55 * gt_loss
            + 0.12 * depth_loss
            + 0.04 * normal_loss
            + 0.06 * alpha_loss
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            student.quaternions.copy_(
                F.normalize(student.quaternions, dim=-1)
            )
            student.opacity_logits.clamp_(-12, 5)
            student.log_scales.clamp_(-12, 5)
        if step == 0 or (step + 1) % 100 == 0:
            row = {
                "iteration": step + 1,
                "novel_camera": novel,
                "loss": float(loss),
                "teacher_rgb": float(teacher_rgb_loss),
                "gt_static": float(gt_loss),
                "depth": float(depth_loss),
                "normal": float(normal_loss),
                "alpha": float(alpha_loss),
                "gaussians": len(student),
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
