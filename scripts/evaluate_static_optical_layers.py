#!/usr/bin/env python
"""Render static Teacher counterfactual layers and optical coverage audits."""

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
from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402
from outdoor.hybrid_teacher_api import load_hybrid_teacher  # noqa: E402
from outdoor.lazy_scene import LazyScene  # noqa: E402
from scene import GaussianModel  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402


def _save(path: Path, image: torch.Tensor) -> None:
    raster = (
        image.detach()
        .clamp(0, 1)
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(raster).save(path)


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse = (prediction - target).square().mean().clamp_min(1e-12)
    return {
        "psnr": float(-10.0 * torch.log10(mse)),
        "ssim": float(ssim(prediction, target)),
        "mae": float((prediction - target).abs().mean()),
    }


def _replacement_group_rows(
    teacher,
    render: dict[str, torch.Tensor],
    *,
    view_index: int,
) -> list[dict[str, float | int | bool]]:
    """Build a projected, view/depth-local replacement audit per group."""
    radii = render["volume_radii"].detach().float().cpu().numpy()
    replacement_tensor = render.get("volume_replacement")
    if replacement_tensor is None:
        replacement = np.zeros_like(radii)
    else:
        replacement = (
            replacement_tensor.detach().float().cpu().numpy()
        )
    group = teacher.foliage.replacement_group.detach().cpu().numpy()
    envelope = (
        teacher.foliage.persistent_envelope_mask.detach().cpu().numpy()
    )
    detail = teacher.foliage.static_leaf_mask.detach().cpu().numpy()
    support = teacher.foliage.support_view_count.detach().cpu().numpy()
    mass = (
        teacher.foliage.integrated_optical_mass().detach().cpu().numpy()
    )
    valid = (group >= 0) & (radii > 0) & (envelope | detail)
    if not np.any(valid):
        return []
    size = int(group[valid].max()) + 1
    area = np.pi * np.square(radii)

    def summed(values, mask):
        return np.bincount(
            group[mask], weights=values[mask], minlength=size
        )

    visible_envelope = valid & envelope
    visible_detail = valid & detail
    envelope_area = summed(area, visible_envelope)
    detail_area = summed(area, visible_detail)
    intersection = summed(
        area * np.clip(replacement, 0.0, 1.0), visible_envelope
    )
    envelope_mass = summed(mass, visible_envelope)
    detail_mass = summed(mass, visible_detail)
    supporting_views = np.zeros(size, dtype=np.int64)
    if np.any(visible_detail):
        np.maximum.at(
            supporting_views,
            group[visible_detail],
            support[visible_detail].astype(np.int64),
        )
    present = np.flatnonzero((envelope_area + detail_area) > 0)
    rows = []
    for group_id in present.tolist():
        overlap = float(
            min(intersection[group_id], envelope_area[group_id])
        )
        union = max(
            float(envelope_area[group_id] + detail_area[group_id] - overlap),
            1e-8,
        )
        fraction = overlap / max(float(envelope_area[group_id]), 1e-8)
        iou = overlap / union
        rows.append(
            {
                "view_index": int(view_index),
                "group": int(group_id),
                "canonical_projected_area": float(envelope_area[group_id]),
                "detail_projected_area": float(detail_area[group_id]),
                "projected_intersection": overlap,
                "projected_iou": iou,
                "canonical_optical_mass": float(envelope_mass[group_id]),
                "detail_optical_mass": float(detail_mass[group_id]),
                "replacement_fraction": fraction,
                "supporting_views": int(supporting_views[group_id]),
                "violation": bool(fraction > 0.5 and iou < 0.2),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    parser.add_argument("--teacher-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", default="408,409,410")
    parser.add_argument("--tree-mask-pickle", type=Path)
    parser.add_argument(
        "--allow-static-optical-handoff-repair",
        action="store_true",
        help=(
            "Explicitly reinterpret a v51 checkpoint with the conserved "
            "same-group Gaussian-mixture hand-off for zero-training diagnosis."
        ),
    )
    args = parser.parse_args()
    dataset = model.extract(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    teacher = load_hybrid_teacher(
        args.teacher_state.resolve(),
        sh_degree=dataset.sh_degree,
        allow_static_optical_handoff_repair=(
            args.allow_static_optical_handoff_repair
        ),
    )
    if teacher.state.get("training_contract", {}).get(
        "reconstruction_target"
    ) != "static":
        raise RuntimeError("Static optical-layer audit requires a static Teacher")
    scene = LazyScene(
        dataset,
        GaussianModel(dataset.sh_degree),
        image_cache_size=1,
    )
    views = scene.getTrainCameras()
    if views and min(
        int(views[0].image_height), int(views[0].image_width)
    ) < 32:
        raise RuntimeError(
            "Optical-layer audit resolution is implausibly small: "
            f"{views[0].image_width}x{views[0].image_height}. Pass the "
            "explicit target width (for example '-r 640'), not a custom "
            "downsampling factor."
        )
    indices = [int(value) for value in args.indices.split(",") if value]
    masks = (
        CambridgeMaskLookup(
            Path(dataset.source_path),
            args.tree_mask_pickle,
            mask_indices=[3],
        )
        if args.tree_mask_pickle is not None
        else None
    )
    configurations = {
        "surface_only": {
            "surface_only": True,
            "volume_layer": "none",
            "optical_replacement_policy": "disabled",
        },
        "envelope_only": {
            "volume_only": True,
            "volume_layer": "envelope",
            "optical_replacement_policy": "disabled",
        },
        "detail_only": {
            "volume_only": True,
            "volume_layer": "detail",
            "optical_replacement_policy": "disabled",
        },
        "surface_envelope": {
            "volume_layer": "envelope",
            "optical_replacement_policy": "disabled",
        },
        "surface_detail": {
            "volume_layer": "detail",
            "optical_replacement_policy": "disabled",
        },
        "full_local_replacement": {
            "volume_layer": "all",
            "optical_replacement_policy": "view_depth_local",
        },
        "full_ray_normalized": {
            "volume_layer": "all",
            "optical_replacement_policy": "ray_normalized_two_pass",
        },
        "full_ray_prior025": {
            "volume_layer": "all",
            "optical_replacement_policy": "ray_normalized_two_pass",
            "optical_responsibility_prior": 0.25,
        },
        "full_ray_prior025_surface_evidence": {
            "volume_layer": "all",
            "optical_replacement_policy": (
                "ray_normalized_surface_evidence_three_pass"
            ),
            "optical_responsibility_prior": 0.25,
        },
        "full_ray_prior_log2": {
            "volume_layer": "all",
            "optical_replacement_policy": "ray_normalized_two_pass",
            "optical_responsibility_prior": float(np.log(2.0)),
        },
        "full_without_replacement": {
            "volume_layer": "all",
            "optical_replacement_policy": "disabled",
        },
    }
    rows = []
    replacement_group_rows = []
    with torch.no_grad():
        for index in indices:
            view = views[index]
            target = view.original_image.cuda(non_blocking=True)
            tree = None
            if masks is not None:
                keep = masks.get_mask(
                    view.image_name,
                    (view.image_height, view.image_width),
                    torch.device("cuda"),
                )
                tree = ~keep
            row = {
                "index": index,
                "image_name": str(view.image_name),
                "height": int(view.image_height),
                "width": int(view.image_width),
            }
            _save(output / f"{index:05d}_gt.png", target)
            for name, kwargs in configurations.items():
                render = teacher.render(view, task=None, **kwargs)
                prediction = render["rgb"]
                alpha = render["volume_alpha"][0]
                volume_radii = render["volume_radii"]
                item = {
                    **_metrics(prediction, target),
                    "mean_volume_alpha": float(alpha.mean()),
                    "alpha_coverage_001": float((alpha > 0.01).float().mean()),
                    "contributing_volume_primitives": int(
                        (volume_radii > 0).sum()
                    ),
                }
                if tree is not None and bool(tree.any()):
                    item.update(
                        {
                            "tree_mean_alpha": float(alpha[tree].mean()),
                            "tree_alpha_coverage_001": float(
                                (alpha[tree] > 0.01).float().mean()
                            ),
                            "tree_hole_rate_001": float(
                                (alpha[tree] <= 0.01).float().mean()
                            ),
                        }
                    )
                row[name] = item
                if name == "full_local_replacement":
                    replacement_group_rows.extend(
                        _replacement_group_rows(
                            teacher, render, view_index=index
                        )
                    )
                _save(output / f"{index:05d}_{name}.png", prediction)
                responsibility = render.get(
                    "optical_detail_responsibility"
                )
                if responsibility is not None:
                    _save(
                        output
                        / f"{index:05d}_{name}_detail_responsibility.png",
                        responsibility.expand(3, -1, -1),
                    )
            rows.append(row)
            release = getattr(view, "release_image", None)
            if release is not None:
                release()
    mass = teacher.foliage.integrated_optical_mass()
    role_mass = {
        "skeleton": float(mass[teacher.foliage.static_skeleton_mask].sum()),
        "envelope": float(mass[teacher.foliage.persistent_envelope_mask].sum()),
        "detail": float(mass[teacher.foliage.detail_leaf_mask].sum()),
    }
    group_audit_path = output / "replacement_groups.jsonl"
    with group_audit_path.open("w", encoding="utf-8") as handle:
        for item in replacement_group_rows:
            handle.write(json.dumps(item) + "\n")
    violations = sum(item["violation"] for item in replacement_group_rows)
    report = {
        "protocol": "static-optical-layer-audit-v2_explicit_resolution",
        "teacher_state": str(args.teacher_state.resolve()),
        "static_optical_handoff_group_association": teacher.state.get(
            "_static_optical_handoff_group_association"
        ),
        "evaluation_contract": {
            "loader_resolution": int(dataset.resolution),
            "image_root_argument": str(dataset.images),
            "indices": indices,
            "static_optical_handoff_reinterpretation": bool(
                args.allow_static_optical_handoff_repair
            ),
        },
        "integrated_optical_mass_by_role": role_mass,
        "replacement_overlap": {
            "mean_envelope_ema": float(
                teacher.foliage.replacement_overlap_ema[
                    teacher.foliage.persistent_envelope_mask
                ].mean()
            ),
            "observed_envelope_rows": int(
                (
                    teacher.foliage.replacement_observation_count
                    [teacher.foliage.persistent_envelope_mask]
                    > 0
                ).sum()
            ),
            "projected_group_view_rows": len(replacement_group_rows),
            "replacement_gt_05_iou_lt_02_violations": int(violations),
            "group_audit_jsonl": str(group_audit_path),
        },
        "views": rows,
    }
    (output / "static_optical_layers.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    scene.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
