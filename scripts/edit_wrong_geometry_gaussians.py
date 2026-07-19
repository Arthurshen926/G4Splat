#!/usr/bin/env python3
"""Conservatively suppress Gaussians that cause clean-depth conflicts.

This experimental adapter leaves the base MAtCha/G4Splat implementation
unchanged. It attributes a local clean-depth error through the differentiable
surfel renderer, searches bounded opacity edits, validates them on the target
and clean-support train views, and writes a separate edited checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GS_ROOT = REPO_ROOT / "2d-gaussian-splatting"
for import_root in (REPO_ROOT, GS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from artifact_guided_repair import (  # noqa: E402
    CameraGeometry,
    select_conservative_edit_trial,
)
from gaussian_renderer import render  # noqa: E402
from guidance.cam_utils import MiniCam  # noqa: E402
from scene import GaussianModel  # noqa: E402


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(value), indent=2) + "\n", encoding="utf-8")


def _make_camera(geometry: CameraGeometry) -> MiniCam:
    return MiniCam(
        geometry.c2w.astype(np.float32),
        geometry.width,
        geometry.height,
        2.0 * math.atan(geometry.height / (2.0 * geometry.fy)),
        2.0 * math.atan(geometry.width / (2.0 * geometry.fx)),
    )


def _read_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def _save_rgb(path: Path, value: torch.Tensor) -> None:
    image = value.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(image * 255.0 + 0.5), mode="RGB").save(path)


def _psnr(rendered: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    pixel_mse = (rendered - target).square().mean(dim=0)
    return float((-10.0 * torch.log10(pixel_mse[mask].mean().clamp_min(1e-12))).item())


def _load_validation_views(
    diagnostic_manifest: dict,
    diagnostics_root: Path,
    view: dict,
    device: torch.device,
) -> list[dict]:
    indices = [int(view["target_view_index"])]
    indices.extend(int(item["view_index"]) for item in view["clean_reference_support"])
    output = []
    seen = set()
    for index in indices:
        if index in seen:
            continue
        seen.add(index)
        record = diagnostic_manifest["views"][index]
        geometry = CameraGeometry.from_json(record["camera"])
        gt = torch.from_numpy(
            _read_rgb(Path(record["image_path"]), (geometry.width, geometry.height)).copy()
        ).permute(2, 0, 1).to(device=device)
        semantic = np.asarray(
            Image.open(diagnostics_root / record["files"]["semantic"]).convert("L")
        ) > 127
        labels = np.asarray(Image.open(diagnostics_root / record["files"]["labels"]))
        component = None
        if index == int(view["target_view_index"]):
            component = labels == int(view["component"]["component_id"])
        output.append(
            {
                "index": index,
                "name": record["image_name"],
                "camera": _make_camera(geometry),
                "gt": gt,
                "static_mask": torch.from_numpy(semantic.copy()).to(device=device),
                "component_mask": torch.from_numpy(component.copy()).to(device=device)
                if component is not None
                else None,
            }
        )
    return output


@torch.no_grad()
def _validation_metrics(gaussians, views, pipe, background) -> tuple[list[dict], dict[str, torch.Tensor]]:
    rows = []
    renders = {}
    for view in views:
        image = render(view["camera"], gaussians, pipe, background)["render"].clamp(0.0, 1.0)
        all_pixels = torch.ones_like(view["static_mask"], dtype=torch.bool)
        row = {
            "name": view["name"],
            "full_psnr": _psnr(image, view["gt"], all_pixels),
            "static_psnr": _psnr(image, view["gt"], view["static_mask"]),
        }
        if view["component_mask"] is not None and view["component_mask"].any():
            row["component_psnr"] = _psnr(image, view["gt"], view["component_mask"])
        rows.append(row)
        renders[view["name"]] = image.detach().cpu()
    return rows, renders


def _minimum_psnr_delta(current: list[dict], baseline: list[dict]) -> float:
    baseline_by_name = {row["name"]: row for row in baseline}
    deltas = []
    for row in current:
        reference = baseline_by_name[row["name"]]
        for key in ("full_psnr", "static_psnr", "component_psnr"):
            if key in row and key in reference:
                deltas.append(float(row[key] - reference[key]))
    return min(deltas) if deltas else -float("inf")


def _pseudo_metrics(package, baseline_rgb, clean_depth, repair_mask) -> dict:
    depth = package["surf_depth"][0]
    denominator = torch.maximum(
        torch.maximum(depth.abs(), clean_depth.abs()),
        torch.as_tensor(1e-6, device=depth.device),
    )
    relative_error = torch.abs(depth - clean_depth) / denominator
    protected = ~repair_mask
    return {
        "depth_relative_mean": float(relative_error[repair_mask].mean().item()),
        "depth_relative_median": float(relative_error[repair_mask].median().item()),
        "alpha_mean": float(package["rend_alpha"][0][repair_mask].mean().item()),
        "outside_rgb_mae": float(
            torch.abs(package["render"] - baseline_rgb)[:, protected].mean().item()
        ),
        "inside_rgb_mae": float(
            torch.abs(package["render"] - baseline_rgb)[:, repair_mask].mean().item()
        ),
    }


def _apply_opacity_factor(gaussians, indices: torch.Tensor, factor: float) -> None:
    opacity = torch.sigmoid(gaussians._opacity.data[indices, 0])
    gaussians._opacity.data[indices, 0] = torch.logit(
        (opacity * float(factor)).clamp(1e-4, 1.0 - 1e-4)
    )


def _prepare_output(output_model: Path, input_model: Path, replace: bool) -> None:
    output_model = output_model.resolve()
    if output_model == input_model.resolve():
        raise ValueError("Output model must differ from the frozen input model")
    if output_model.exists() and any(output_model.iterdir()):
        if not replace:
            raise FileExistsError(f"Output model is not empty: {output_model}")
        shutil.rmtree(output_model)
    output_model.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_model", type=Path, required=True)
    parser.add_argument("--input_iteration", type=int, default=30000)
    parser.add_argument("--stage_root", type=Path, required=True)
    parser.add_argument("--diagnostics_dir", type=Path, required=True)
    parser.add_argument("--output_model", type=Path, required=True)
    parser.add_argument("--opacity_factor", type=float, default=0.20)
    parser.add_argument("--candidate_point_counts", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--min_screen_radius", type=float, default=8.0)
    parser.add_argument("--min_depth_improvement", type=float, default=0.05)
    parser.add_argument("--max_outside_rgb_mae", type=float, default=0.015)
    parser.add_argument("--max_validation_psnr_drop", type=float, default=0.025)
    parser.add_argument("--near_best_ratio", type=float, default=0.95)
    parser.add_argument("--replace_output", action="store_true")
    args = parser.parse_args()

    input_model = args.input_model.expanduser().resolve()
    stage_root = args.stage_root.expanduser().resolve()
    diagnostics_root = args.diagnostics_dir.expanduser().resolve()
    output_model = args.output_model.expanduser().resolve()
    _prepare_output(output_model, input_model, args.replace_output)

    stage_manifest = json.loads((stage_root / "artifact_guided_manifest.json").read_text())
    accepted_path = stage_root / "accepted_view_indices.json"
    accepted_view_indices = {
        int(index) for index in json.loads(accepted_path.read_text())
    }
    diagnostic_manifest = json.loads((diagnostics_root / "diagnostic_manifest.json").read_text())
    checkpoint = input_model / "point_cloud" / f"iteration_{args.input_iteration}" / "point_cloud.ply"
    gaussians = GaussianModel(3)
    gaussians.load_ply(str(checkpoint))
    device = gaussians.get_xyz.device
    pipe = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        depth_ratio=0.0,
    )
    background = torch.zeros(3, dtype=torch.float32, device=device)
    visual_root = output_model / "artifact_edit_visuals"
    component_reports = []
    selected_index_chunks = []

    for view in stage_manifest["pseudo_views"]:
        if view["classification"] != "wrong_geometry":
            continue
        pseudo_index = int(view["pseudo_view_index"])
        if pseudo_index not in accepted_view_indices:
            continue
        geometry = CameraGeometry.from_json(view["pseudo_camera"])
        camera = _make_camera(geometry)
        stem = f"{pseudo_index:06d}"
        known = np.asarray(
            Image.open(stage_root / "select-gs" / f"mask_frame{stem}.png").convert("L")
        ) > 127
        repair_mask = torch.from_numpy((~known).copy()).to(device=device)
        clean_depth = torch.from_numpy(
            np.asarray(Image.open(view["clean_support_depth_path"]), dtype=np.float32).copy()
        ).to(device=device)

        gaussians._opacity.grad = None
        baseline_package = render(camera, gaussians, pipe, background)
        baseline_rgb = baseline_package["render"].detach()
        relative = (baseline_package["surf_depth"][0] - clean_depth) / clean_depth.clamp_min(1e-6)
        attribution_loss = torch.sqrt(relative.square() + 1e-6)[repair_mask].mean()
        attribution_loss.backward()
        opacity_gradient = gaussians._opacity.grad[:, 0].detach().clone()
        gaussians._opacity.grad = None
        visibility = baseline_package["visibility_filter"].detach()
        radii = baseline_package["radii"].detach()
        original_opacity = gaussians._opacity.detach().clone()
        opacity = torch.sigmoid(original_opacity[:, 0])
        reduced_logits = torch.logit(
            (opacity * float(args.opacity_factor)).clamp(1e-4, 1.0 - 1e-4)
        )
        benefit = opacity_gradient.clamp_min(0.0) * (
            original_opacity[:, 0] - reduced_logits
        ).clamp_min(0.0)
        eligible = visibility & (radii >= float(args.min_screen_radius)) & (benefit > 0)
        benefit = torch.where(eligible, benefit, torch.zeros_like(benefit))
        order = torch.argsort(benefit, descending=True)
        positive_count = int((benefit > 0).sum().item())
        if positive_count == 0:
            gaussians._opacity.data.copy_(original_opacity)
            component_reports.append({"pseudo_view_index": pseudo_index, "accepted": False, "reason": "no_positive_depth_attribution"})
            continue

        validation_views = _load_validation_views(
            diagnostic_manifest,
            diagnostics_root,
            view,
            device,
        )
        baseline_validation, baseline_validation_renders = _validation_metrics(
            gaussians, validation_views, pipe, background
        )
        baseline_metrics = _pseudo_metrics(
            baseline_package,
            baseline_rgb,
            clean_depth,
            repair_mask,
        )
        trials = []
        for requested_count in sorted(set(args.candidate_point_counts)):
            count = min(int(requested_count), positive_count)
            if count <= 0:
                continue
            indices = order[:count]
            gaussians._opacity.data.copy_(original_opacity)
            _apply_opacity_factor(gaussians, indices, float(args.opacity_factor))
            with torch.no_grad():
                trial_package = render(camera, gaussians, pipe, background)
            pseudo_metrics = _pseudo_metrics(
                trial_package,
                baseline_rgb,
                clean_depth,
                repair_mask,
            )
            validation, _ = _validation_metrics(gaussians, validation_views, pipe, background)
            trials.append(
                {
                    "edited_points": count,
                    "depth_improvement": baseline_metrics["depth_relative_mean"]
                    - pseudo_metrics["depth_relative_mean"],
                    "outside_rgb_mae": pseudo_metrics["outside_rgb_mae"],
                    "min_validation_psnr_delta": _minimum_psnr_delta(
                        validation, baseline_validation
                    ),
                    "pseudo_metrics": pseudo_metrics,
                    "validation": validation,
                }
            )

        selected = select_conservative_edit_trial(
            trials,
            min_depth_improvement=float(args.min_depth_improvement),
            max_outside_rgb_mae=float(args.max_outside_rgb_mae),
            max_validation_psnr_drop=float(args.max_validation_psnr_drop),
            near_best_ratio=float(args.near_best_ratio),
        )
        gaussians._opacity.data.copy_(original_opacity)
        if selected is None:
            component_reports.append(
                {
                    "pseudo_view_index": pseudo_index,
                    "accepted": False,
                    "reason": "no_edit_trial_passed_quality_budget",
                    "baseline_pseudo_metrics": baseline_metrics,
                    "baseline_validation": baseline_validation,
                    "trials": trials,
                }
            )
            continue

        selected_indices = order[: int(selected["edited_points"])]
        _apply_opacity_factor(gaussians, selected_indices, float(args.opacity_factor))
        with torch.no_grad():
            edited_package = render(camera, gaussians, pipe, background)
        edited_validation, edited_validation_renders = _validation_metrics(
            gaussians, validation_views, pipe, background
        )
        _save_rgb(visual_root / f"pseudo_{stem}.baseline.png", baseline_rgb)
        _save_rgb(visual_root / f"pseudo_{stem}.edited.png", edited_package["render"])
        for name, baseline_image in baseline_validation_renders.items():
            _save_rgb(visual_root / f"{name}.baseline.png", baseline_image)
            _save_rgb(visual_root / f"{name}.edited.png", edited_validation_renders[name])
        selected_index_chunks.append(selected_indices.detach().cpu().numpy())
        component_reports.append(
            {
                "pseudo_view_index": pseudo_index,
                "target_image_name": view["target_image_name"],
                "accepted": True,
                "positive_attribution_points": positive_count,
                "selected_points": int(selected_indices.numel()),
                "selected_indices": selected_indices.detach().cpu().numpy(),
                "selected_radii": radii[selected_indices].detach().cpu().numpy(),
                "selected_benefit": benefit[selected_indices].detach().cpu().numpy(),
                "baseline_pseudo_metrics": baseline_metrics,
                "edited_pseudo_metrics": _pseudo_metrics(
                    edited_package, baseline_rgb, clean_depth, repair_mask
                ),
                "baseline_validation": baseline_validation,
                "edited_validation": edited_validation,
                "trials": trials,
            }
        )

    if not selected_index_chunks:
        _write_json(
            output_model / "artifact_edit_rejection_report.json",
            {
                "version": 1,
                "mode": "clean_depth_gradient_attribution_opacity_edit_rejected",
                "input_model": str(input_model),
                "input_iteration": int(args.input_iteration),
                "component_reports": component_reports,
                "quality_budget": {
                    "min_depth_improvement": float(args.min_depth_improvement),
                    "max_outside_rgb_mae": float(args.max_outside_rgb_mae),
                    "max_validation_psnr_drop": float(args.max_validation_psnr_drop),
                    "near_best_ratio": float(args.near_best_ratio),
                },
            },
        )
        raise RuntimeError("No wrong-geometry opacity edit passed the quality gate")

    checkpoint_root = output_model / "point_cloud" / f"iteration_{args.input_iteration}"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(str(checkpoint_root / "point_cloud.ply"))
    cfg_path = input_model / "cfg_args"
    if cfg_path.is_file():
        shutil.copy2(cfg_path, output_model / "cfg_args")
    all_selected = np.unique(np.concatenate(selected_index_chunks)).astype(np.int64)
    np.save(output_model / "edited_gaussian_indices.npy", all_selected)
    report = {
        "version": 1,
        "mode": "clean_depth_gradient_attribution_opacity_edit",
        "input_model": str(input_model),
        "input_iteration": int(args.input_iteration),
        "stage_root": str(stage_root),
        "diagnostics_dir": str(diagnostics_root),
        "output_model": str(output_model),
        "opacity_factor": float(args.opacity_factor),
        "selected_point_count": int(len(all_selected)),
        "component_reports": component_reports,
        "quality_budget": {
            "min_depth_improvement": float(args.min_depth_improvement),
            "max_outside_rgb_mae": float(args.max_outside_rgb_mae),
            "max_validation_psnr_drop": float(args.max_validation_psnr_drop),
            "near_best_ratio": float(args.near_best_ratio),
        },
    }
    _write_json(output_model / "artifact_edit_report.json", report)
    print(
        json.dumps(
            {
                "output_model": str(output_model),
                "selected_point_count": int(len(all_selected)),
                "component_count": len(component_reports),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
