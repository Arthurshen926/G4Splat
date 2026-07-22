#!/usr/bin/env python3
"""Audit conservative shallow-occluder subsets for an established 3-D ROI.

This is deliberately a read-only diagnostic for the failure mode where an
independent Chart/SfM surface is known, but the frozen 2DGS surface depth is
substantially in front of it.  The existing plane-update path correctly keeps
only primitives *on* the independent plane.  That cannot identify an
incorrect shallow layer which occludes the known surface.

For each causal cluster with accepted independent surface evidence, this tool
therefore:

1. keeps attributed primitive centres which project inside the 3-D ROI and
   lie materially in front of the independently recovered plane;
2. tests each candidate by opacity-zero intervention on the original target
   anomaly rays and the exact real-camera control envelope; and
3. greedily tests only joint subsets whose every intermediate intervention
   remains inside the clean-view budget.

It never writes a Gaussian PLY and it never treats a selected subset as an
approved edit.  A later repair stage must still revalidate a proposed local
optimization or split against all required real observations.
"""

from __future__ import annotations

import argparse
import json
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
for import_root in (REPO_ROOT, GS_ROOT, REPO_ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from artifact_guided_repair import ArtifactROI3D, project_points, render_support_plane_depth  # noqa: E402
from causal_2dgs_repair import (  # noqa: E402
    _LazyCameraStore,
    _LazyStaticMasks,
    _camera_gt,
    _fixed_mae,
    _json_ready,
)
from artifact_guided_repair.core import robust_affine_color  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402
from scene import GaussianModel  # noqa: E402


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 127


def _save_overlay(
    path: Path,
    image: np.ndarray,
    roi_mask: np.ndarray,
    candidates: list[dict[str, Any]],
    selected_ids: set[int],
) -> None:
    """Persist a compact visual audit without making a 2-D mask a 3-D claim."""
    result = np.uint8(np.clip(image, 0.0, 1.0) * 255.0).copy()
    contours, _ = cv2.findContours(np.uint8(roi_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (50, 220, 50), 2, lineType=cv2.LINE_AA)
    for row in candidates:
        x, y = row["pixel_xy"]
        color = (50, 220, 50) if int(row["primitive_id"]) in selected_ids else (255, 185, 45)
        cv2.circle(result, (int(x), int(y)), 4, color, 1, lineType=cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result, mode="RGB").save(path)


def _front_roi_candidates(
    primitive_ids: np.ndarray,
    xyz: np.ndarray,
    roi: ArtifactROI3D,
    geometry: Any,
    plane: np.ndarray,
    *,
    minimum_relative_gap: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Return attributed centres that are both ROI-aligned and clearly shallow.

    Centre inclusion is intentionally conservative.  A Gaussian whose broad
    footprint overlaps the ROI but whose centre does not is not admitted by
    this diagnostic; later rasterizer-level contribution accounting can widen
    the candidate set without weakening this initial contract.
    """
    roi_mask = roi.project_mask(geometry)
    plane_depth, plane_valid = render_support_plane_depth(geometry, plane, roi_mask)
    ids = np.unique(np.asarray(primitive_ids, dtype=np.int64))
    ids = ids[(ids >= 0) & (ids < len(xyz))]
    if not len(ids):
        return roi_mask, plane_depth, []
    pixels, candidate_depth, inside = project_points(xyz[ids], geometry)
    rounded = np.rint(pixels).astype(np.int64)
    inside &= (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < geometry.width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < geometry.height)
    )
    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(inside):
        x, y = rounded[index]
        if not (roi_mask[y, x] and plane_valid[y, x]):
            continue
        support_depth = float(plane_depth[y, x])
        relative_gap = (support_depth - float(candidate_depth[index])) / max(support_depth, 1e-6)
        if relative_gap < float(minimum_relative_gap):
            continue
        rows.append(
            {
                "primitive_id": int(ids[index]),
                "pixel_xy": [int(x), int(y)],
                "candidate_depth": float(candidate_depth[index]),
                "independent_plane_depth": support_depth,
                "relative_front_gap": float(relative_gap),
            }
        )
    return roi_mask, plane_depth, rows


def _render_rgb(camera: Any, gaussians: GaussianModel, pipe: Any, background: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        package = render(camera, gaussians, pipe, background)
    return (
        package["render"].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32),
        package["rend_alpha"][0].detach().cpu().numpy().astype(np.float32),
        package["visibility_filter"].detach().cpu().numpy().astype(bool),
    )


def _evaluate_opacity_subset(
    ids: list[int],
    *,
    target_name: str,
    target_camera: Any,
    target_gt: np.ndarray,
    target_mask: np.ndarray,
    target_affine: np.ndarray,
    baseline_target_mae: float,
    controls: list[dict[str, Any]],
    cameras: _LazyCameraStore,
    static_masks: _LazyStaticMasks,
    gaussians: GaussianModel,
    pipe: Any,
    background: torch.Tensor,
    intervention_opacity_logit: float,
    min_target_improvement: float,
    max_clean_increase: float,
) -> dict[str, Any]:
    """Intervene on one candidate set and restore the frozen opacity tensor."""
    tensor_ids = torch.as_tensor(ids, dtype=torch.long, device=gaussians._opacity.device)
    original = gaussians._opacity.detach()[tensor_ids].clone()
    try:
        with torch.no_grad():
            gaussians._opacity[tensor_ids] = float(intervention_opacity_logit)
        target_rgb, _, _ = _render_rgb(target_camera, gaussians, pipe, background)
        target_mae = _fixed_mae(target_rgb, target_gt, target_mask, target_affine)
        target_relative_improvement = (baseline_target_mae - target_mae) / max(baseline_target_mae, 1e-6)
        control_rows = []
        for control in controls:
            name = str(control["name"])
            rgb, _, _ = _render_rgb(cameras[name], gaussians, pipe, background)
            mae = _fixed_mae(rgb, _camera_gt(cameras[name]), static_masks[name], control["affine"])
            control_rows.append(
                {
                    "name": name,
                    "baseline_mae": float(control["baseline_mae"]),
                    "mae": float(mae),
                    "mae_increase": float(mae - float(control["baseline_mae"])),
                    "visible_fraction": float(control["visible_fraction"]),
                }
            )
    finally:
        with torch.no_grad():
            gaussians._opacity[tensor_ids] = original
    worst_clean = max((float(row["mae_increase"]) for row in control_rows), default=float("inf"))
    return {
        "primitive_ids": [int(value) for value in ids],
        "target_mae": float(target_mae),
        "baseline_target_mae": float(baseline_target_mae),
        "target_relative_improvement": float(target_relative_improvement),
        "worst_clean_mae_increase": float(worst_clean),
        "clean_controls": control_rows,
        "accepted": bool(
            target_relative_improvement >= float(min_target_improvement)
            and worst_clean <= float(max_clean_increase)
        ),
    }


def _target_counterfactual_row(counterfactual: dict[str, Any], target_name: str) -> dict[str, Any]:
    row = next((item for item in counterfactual.get("targets", []) if str(item.get("name")) == target_name), None)
    if row is None:
        raise ValueError(f"Causal cluster {counterfactual.get('cluster_id')} lacks target row for {target_name}")
    return row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--mask-pickle", type=Path, required=True)
    parser.add_argument("--causal-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cluster-ids", nargs="+", type=int)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--minimum-relative-front-gap", type=float, default=0.10)
    parser.add_argument("--max-candidates-per-cluster", type=int, default=24)
    parser.add_argument("--min-target-relative-improvement", type=float)
    parser.add_argument("--max-clean-mae-increase", type=float)
    parser.add_argument(
        "--intervention-opacity-logit",
        type=float,
        default=-16.0,
        help="Temporary logit assigned to each tested candidate; the frozen PLY is never written.",
    )
    parser.add_argument("--white-background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0.0 < float(args.minimum_relative_front_gap) < 1.0:
        raise ValueError("--minimum-relative-front-gap must lie in (0, 1)")
    if int(args.max_candidates_per_cluster) <= 0:
        raise ValueError("--max-candidates-per-cluster must be positive")
    if not np.isfinite(float(args.intervention_opacity_logit)):
        raise ValueError("--intervention-opacity-logit must be finite")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not args.replace_output:
            raise FileExistsError(f"Output is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    causal_root = args.causal_root.expanduser().resolve()
    report = json.loads((causal_root / "causal_repair_report.json").read_text(encoding="utf-8"))
    source = args.source_path.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    point_cloud = model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    if not point_cloud.is_file():
        raise FileNotFoundError(f"Frozen checkpoint does not exist: {point_cloud}")
    gaussians = GaussianModel(3)
    gaussians.load_ply(str(point_cloud))
    cameras = _LazyCameraStore.from_colmap(source, requested_resolution=int(args.resolution), data_device="cpu")
    lookup = CambridgeMaskLookup(source, args.mask_pickle.expanduser().resolve(), [0, 1, 2])
    static_masks = _LazyStaticMasks(lookup, cameras.geometries)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, depth_ratio=0.0)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=gaussians.get_xyz.device,
    )
    with torch.no_grad():
        xyz = gaussians.get_xyz.detach().cpu().numpy()

    counterfactuals = {int(row["cluster_id"]): row for row in report.get("counterfactual_clusters", [])}
    requested_clusters = None if args.cluster_ids is None else {int(value) for value in args.cluster_ids}
    results: dict[str, Any] = {
        "protocol": "independent_roi_shallow_occluder_joint_subset_counterfactual_v1",
        "read_only": True,
        "model_path": str(model_path),
        "iteration": int(args.iteration),
        "causal_root": str(causal_root),
        "minimum_relative_front_gap": float(args.minimum_relative_front_gap),
        "max_candidates_per_cluster": int(args.max_candidates_per_cluster),
        "intervention_opacity_logit": float(args.intervention_opacity_logit),
        "clusters": {},
    }
    for cluster_text, surface in sorted(report.get("surface_evidence", {}).items(), key=lambda item: int(item[0])):
        cluster_id = int(cluster_text)
        if requested_clusters is not None and cluster_id not in requested_clusters:
            continue
        if not bool(surface.get("surface_known")) or not surface.get("roi_path") or surface.get("plane") is None:
            continue
        counterfactual = counterfactuals.get(cluster_id)
        if counterfactual is None:
            continue
        target_name = str(surface.get("target_name", ""))
        if target_name not in cameras:
            continue
        target_row = _target_counterfactual_row(counterfactual, target_name)
        target_camera = cameras[target_name]
        target_gt = _camera_gt(target_camera)
        target_mask_path = causal_root / "counterfactuals" / f"cluster_{cluster_id:03d}" / f"{target_name}.anomaly.png"
        if not target_mask_path.is_file():
            raise FileNotFoundError(f"Missing frozen target anomaly mask: {target_mask_path}")
        target_mask = _read_mask(target_mask_path)
        if target_mask.shape != target_gt.shape[:2]:
            raise ValueError(f"Target mask shape mismatch for {target_name}: {target_mask.shape} vs {target_gt.shape[:2]}")
        target_baseline_rgb, target_baseline_alpha, target_visibility = _render_rgb(
            target_camera, gaussians, pipe, background
        )
        # Version-7 target rows retained the frozen target MAE but not the
        # affine itself.  Refit exactly once on the frozen render and static
        # valid rays, which is the same rule used by build_anomaly_rays.
        _, target_affine = robust_affine_color(
            target_baseline_rgb,
            target_gt,
            static_masks[target_name] & (target_baseline_alpha > 0.30),
        )
        target_baseline_mae = _fixed_mae(target_baseline_rgb, target_gt, target_mask, target_affine)

        controls = []
        for row in counterfactual.get("clean_controls", []):
            name = str(row["name"])
            if name not in cameras:
                continue
            baseline_rgb, _, visibility = _render_rgb(cameras[name], gaussians, pipe, background)
            baseline_mae = _fixed_mae(
                baseline_rgb,
                _camera_gt(cameras[name]),
                static_masks[name],
                np.asarray(row["affine"], dtype=np.float32),
            )
            controls.append(
                {
                    "name": name,
                    "affine": np.asarray(row["affine"], dtype=np.float32),
                    "baseline_mae": float(baseline_mae),
                    "reported_baseline_mae": float(row["baseline_mae"]),
                    "visible_fraction": float(row["visible_fraction"]),
                    "candidate_visibility": visibility,
                }
            )
        thresholds = counterfactual.get("acceptance_thresholds", {})
        min_target = float(
            args.min_target_relative_improvement
            if args.min_target_relative_improvement is not None
            else thresholds.get("min_bad_relative_improvement", thresholds.get("min_bad_improvement", 0.04))
        )
        max_clean = float(
            args.max_clean_mae_increase
            if args.max_clean_mae_increase is not None
            else thresholds.get("max_clean_mae_increase", 0.006)
        )
        roi = ArtifactROI3D.load(Path(surface["roi_path"]))
        roi_mask, _, candidates = _front_roi_candidates(
            np.asarray(counterfactual.get("primitive_ids", []), dtype=np.int64),
            xyz,
            roi,
            cameras.geometries[target_name],
            np.asarray(surface["plane"], dtype=np.float64),
            minimum_relative_gap=float(args.minimum_relative_front_gap),
        )
        candidates.sort(key=lambda row: (-float(row["relative_front_gap"]), int(row["primitive_id"])))
        candidates = candidates[: int(args.max_candidates_per_cluster)]
        for row in candidates:
            primitive_id = int(row["primitive_id"])
            row["target_visible"] = bool(target_visibility[primitive_id])
            row["visible_control_count"] = int(
                sum(bool(control["candidate_visibility"][primitive_id]) for control in controls)
            )
            row["trial"] = _evaluate_opacity_subset(
                [primitive_id],
                target_name=target_name,
                target_camera=target_camera,
                target_gt=target_gt,
                target_mask=target_mask,
                target_affine=target_affine,
                baseline_target_mae=target_baseline_mae,
                controls=controls,
                cameras=cameras,
                static_masks=static_masks,
                gaussians=gaussians,
                pipe=pipe,
                background=background,
                intervention_opacity_logit=float(args.intervention_opacity_logit),
                min_target_improvement=min_target,
                max_clean_increase=max_clean,
            )

        # The greedy ordering is only a proposal mechanism.  Every proposed
        # prefix is rendered jointly and must itself satisfy the strict clean
        # envelope, so no additive independence assumption reaches the report.
        ranked = sorted(
            candidates,
            key=lambda row: (
                -float(row["trial"]["target_relative_improvement"]),
                float(row["trial"]["worst_clean_mae_increase"]),
                -float(row["relative_front_gap"]),
                int(row["primitive_id"]),
            ),
        )
        selected: list[int] = []
        selected_trial: dict[str, Any] | None = None
        for row in ranked:
            individual = row["trial"]
            if float(individual["worst_clean_mae_increase"]) > max_clean:
                continue
            proposal = [*selected, int(row["primitive_id"])]
            joint = _evaluate_opacity_subset(
                proposal,
                target_name=target_name,
                target_camera=target_camera,
                target_gt=target_gt,
                target_mask=target_mask,
                target_affine=target_affine,
                baseline_target_mae=target_baseline_mae,
                controls=controls,
                cameras=cameras,
                static_masks=static_masks,
                gaussians=gaussians,
                pipe=pipe,
                background=background,
                intervention_opacity_logit=float(args.intervention_opacity_logit),
                min_target_improvement=min_target,
                max_clean_increase=max_clean,
            )
            current_gain = -float("inf") if selected_trial is None else float(selected_trial["target_relative_improvement"])
            if (
                float(joint["worst_clean_mae_increase"]) <= max_clean
                and float(joint["target_relative_improvement"]) > current_gain + 1e-8
            ):
                selected = proposal
                selected_trial = joint
        _save_overlay(
            output / f"cluster_{cluster_id:03d}_{target_name}_candidate_overlay.png",
            target_baseline_rgb,
            roi_mask,
            candidates,
            set(selected),
        )
        results["clusters"][str(cluster_id)] = {
            "target_name": target_name,
            "surface_source": surface.get("surface_source"),
            "roi_path": surface.get("roi_path"),
            "target_baseline_mae": float(target_baseline_mae),
            "reported_target_baseline_mae": float(target_row["baseline_mae"]),
            "target_baseline_mae_delta": float(target_baseline_mae - float(target_row["baseline_mae"])),
            "recomputed_target_affine": target_affine,
            "target_anomaly_pixels": int(target_mask.sum()),
            "roi_pixel_count": int(roi_mask.sum()),
            "control_count": int(len(controls)),
            "thresholds": {
                "min_target_relative_improvement": min_target,
                "max_clean_mae_increase": max_clean,
            },
            "front_roi_candidate_count": int(len(candidates)),
            "candidates": candidates,
            "greedy_joint_subset": {
                "primitive_ids": selected,
                "trial": selected_trial,
                "accepted": bool(selected_trial is not None and selected_trial["accepted"]),
            },
        }
        print(
            f"[cluster {cluster_id}] front candidates={len(candidates)}, "
            f"joint={selected}, accepted={False if selected_trial is None else selected_trial['accepted']}",
            flush=True,
        )
    (output / "occluder_subset_audit.json").write_text(
        json.dumps(_json_ready(results), indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
