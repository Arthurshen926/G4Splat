#!/usr/bin/env python3
"""Build aligned diagnostic panels for artifact-guided retained-model repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def read_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return np.asarray(image)


def read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def read_depth(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.float32)


def robust_depth(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    result = np.zeros_like(depth, dtype=np.float32)
    if np.any(valid):
        low, high = np.percentile(depth[valid], [2, 98])
        result[valid] = np.clip((depth[valid] - low) / max(float(high - low), 1e-6), 0, 1)
    return result


def overlay_contour(image: np.ndarray, mask: np.ndarray, color=(255, 48, 48)) -> np.ndarray:
    output = image.copy()
    contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, color, 2, lineType=cv2.LINE_AA)
    return output


def local_plane_coordinates(points: np.ndarray, plane: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = plane[:3].astype(np.float64)
    normal /= max(np.linalg.norm(normal), 1e-12)
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(seed @ normal)) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    axis_u = np.cross(normal, seed)
    axis_u /= max(np.linalg.norm(axis_u), 1e-12)
    axis_v = np.cross(normal, axis_u)
    centered = points - points.mean(axis=0, keepdims=True)
    return centered @ axis_u, centered @ axis_v, centered @ normal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    stage = args.stage_root.expanduser().resolve()
    diagnostics_root = args.diagnostics_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((stage / "artifact_guided_manifest.json").read_text())
    diagnostics = json.loads((diagnostics_root / "diagnostic_manifest.json").read_text())
    projective = json.loads((stage / "projective_repair_report.json").read_text())
    policy = json.loads((stage / "artifact_training_policy.json").read_text())
    projective_by_index = {int(item["index"]): item for item in projective["views"]}
    policy_by_index = {int(item["index"]): item for item in policy["views"]}

    summary_rows = []
    for view in manifest["pseudo_views"]:
        index = int(view["pseudo_view_index"])
        stem = f"{index:06d}"
        target = diagnostics["views"][int(view["target_view_index"])]
        width = int(view["pseudo_camera"]["width"])
        height = int(view["pseudo_camera"]["height"])
        size = (width, height)

        known = read_gray(stage / "select-gs" / f"mask_frame{stem}.png") > 127
        repair = ~known
        raw = read_rgb(Path(target["image_path"]), size)
        render = read_rgb(diagnostics_root / target["files"]["render"], size)
        score = read_gray(diagnostics_root / target["files"]["score"])
        labels = read_gray(diagnostics_root / target["files"]["labels"])
        baseline = read_rgb(stage / "select-gs" / f"ori_warp_frame{stem}.png", size)
        repaired = read_rgb(
            stage / "select-gs-projective-merged" / f"predict_warp_frame{stem}.png", size
        )
        baseline_depth = read_depth(stage / "select-gs" / f"depth_frame{stem}.tiff")
        clean_depth = read_depth(Path(view["clean_support_depth_path"]))
        generated_depth = read_depth(
            stage / "select-gs-projective-geometry" / f"depth_frame{stem}.tiff"
        )
        delta = np.abs(repaired.astype(np.float32) - baseline.astype(np.float32)).mean(axis=2)
        denominator = np.maximum(np.maximum(np.abs(generated_depth), np.abs(clean_depth)), 1e-6)
        relative_depth_error = np.abs(generated_depth - clean_depth) / denominator
        relative_depth_error[~repair] = np.nan

        roi = np.load(stage / f"roi3d_frame{stem}.npz", allow_pickle=True)
        u, v, residual = local_plane_coordinates(roi["points"], roi["plane_world"])
        projection = projective_by_index[index]
        decision = policy_by_index[index]

        fig, axes = plt.subplots(3, 4, figsize=(20, 11), constrained_layout=True)
        panels = [
            (overlay_contour(raw, repair), "Real target + ROI"),
            (overlay_contour(render, repair), "Frozen retained_v2 render"),
            (score, "Artifact score", "magma", 0, 255),
            (labels, "Detected components", "tab20", None, None),
            (baseline, "Pseudo-view baseline"),
            (repair, f"Repair mask ({repair.mean():.1%})", "gray", 0, 1),
            (repaired, "Projective RGB repair"),
            (delta, f"RGB change, mean={delta[repair].mean():.1f}", "inferno", 0, None),
            (robust_depth(baseline_depth), "Baseline rendered depth", "viridis", 0, 1),
            (robust_depth(clean_depth), "Clean-support plane depth", "viridis", 0, 1),
            (relative_depth_error, "Projective depth rel. error", "magma", 0, 0.15),
        ]
        for axis, panel in zip(axes.flat[:11], panels):
            image, title, *style = panel
            kwargs = {}
            if style:
                kwargs["cmap"] = style[0]
                if len(style) > 1 and style[1] is not None:
                    kwargs["vmin"] = style[1]
                if len(style) > 2 and style[2] is not None:
                    kwargs["vmax"] = style[2]
            axis.imshow(image, **kwargs)
            axis.set_title(title, fontsize=11)
            axis.axis("off")

        roi_axis = axes.flat[11]
        scatter = roi_axis.scatter(u, v, c=residual, cmap="coolwarm", s=28)
        roi_axis.set_aspect("equal", adjustable="box")
        roi_axis.set_xlabel("plane axis u")
        roi_axis.set_ylabel("plane axis v")
        roi_axis.set_title(
            f"3D ROI: {len(u)} points, slab={view['roi3d']['slab_thickness']:.4f}"
        )
        fig.colorbar(scatter, ax=roi_axis, fraction=0.046, label="normal residual")

        reasons = decision.get("reasons", [])
        fig.suptitle(
            f"ROI {index} | target={view['target_image_name']} | "
            f"projective={'PASS' if projection['accepted'] else 'FAIL'} | "
            f"final={'PASS' if decision['accepted'] else 'FAIL'}\n"
            f"depth support={projection['fusion']['depth_supported_fraction']:.3f}, "
            f"nearest fill={projection['nearest_fill_fraction']:.3f}, "
            f"baseline depth err={decision['baseline_depth_median_relative_error']:.3f}, "
            f"repaired depth err={decision['generated_depth_median_relative_error']:.3f}; "
            f"reasons={reasons or ['none']}",
            fontsize=13,
        )
        fig.savefig(output / f"roi_{stem}_diagnostic_panel.png", dpi=140)
        plt.close(fig)

        target_bbox = [int(value) for value in view["component"]["bbox_xyxy"]]
        tx0, ty0, tx1, ty1 = target_bbox
        target_pad = 18
        tx0, ty0 = max(tx0 - target_pad, 0), max(ty0 - target_pad, 0)
        tx1, ty1 = min(tx1 + target_pad, width), min(ty1 + target_pad, height)
        yy, xx = np.where(repair)
        pseudo_pad = 18
        px0, py0 = max(int(xx.min()) - pseudo_pad, 0), max(int(yy.min()) - pseudo_pad, 0)
        px1, py1 = min(int(xx.max()) + pseudo_pad + 1, width), min(int(yy.max()) + pseudo_pad + 1, height)
        crop_fig, crop_axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        crop_items = [
            (raw[ty0:ty1, tx0:tx1], "Real target crop", {}),
            (render[ty0:ty1, tx0:tx1], "retained_v2 crop", {}),
            (score[ty0:ty1, tx0:tx1], "Artifact score crop", {"cmap": "magma", "vmin": 0, "vmax": 255}),
            (baseline[py0:py1, px0:px1], "Pseudo baseline crop", {}),
            (repaired[py0:py1, px0:px1], "Projective repair crop", {}),
            (relative_depth_error[py0:py1, px0:px1], "Depth error crop", {"cmap": "magma", "vmin": 0, "vmax": 0.15}),
        ]
        for axis, (crop, title, kwargs) in zip(crop_axes.flat, crop_items):
            axis.imshow(crop, **kwargs)
            axis.set_title(title)
            axis.axis("off")
        crop_fig.suptitle(
            f"ROI {index} local comparison | final={'PASS' if decision['accepted'] else 'FAIL'}",
            fontsize=14,
        )
        crop_fig.savefig(output / f"roi_{stem}_crop_comparison.png", dpi=160)
        plt.close(crop_fig)

        summary_rows.append(
            {
                "index": index,
                "accepted": bool(decision["accepted"]),
                "depth_support": float(projection["fusion"]["depth_supported_fraction"]),
                "nearest_fill": float(projection["nearest_fill_fraction"]),
                "baseline_error": float(decision["baseline_depth_median_relative_error"]),
                "generated_error": float(decision["generated_depth_median_relative_error"]),
                "repair_fraction": float(decision["synthesized_fraction"]),
                "reasons": reasons,
            }
        )

    indices = [str(row["index"]) for row in summary_rows]
    x = np.arange(len(indices))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].bar(x - 0.18, [r["baseline_error"] for r in summary_rows], 0.36, label="baseline")
    axes[0].bar(x + 0.18, [r["generated_error"] for r in summary_rows], 0.36, label="projective")
    axes[0].set_title("Median relative depth error")
    axes[0].legend()
    axes[1].bar(x - 0.18, [r["depth_support"] for r in summary_rows], 0.36, label="depth support")
    axes[1].bar(x + 0.18, [r["nearest_fill"] for r in summary_rows], 0.36, label="nearest fill")
    axes[1].axhline(0.85, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(0.02, color="red", linestyle=":", linewidth=1)
    axes[1].set_title("Projective support / fallback")
    axes[1].legend()
    colors = ["#2ca02c" if r["accepted"] else "#d62728" for r in summary_rows]
    axes[2].bar(x, [r["repair_fraction"] for r in summary_rows], color=colors)
    axes[2].axhline(0.15, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("Repair fraction and final decision")
    for axis in axes:
        axis.set_xticks(x, indices)
        axis.set_xlabel("ROI index")
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output / "roi_gate_summary.png", dpi=160)
    plt.close(fig)

    # Visualize likely duplicate 3D ROIs before novel-view generation.
    roi_records = []
    for view in manifest["pseudo_views"]:
        index = int(view["pseudo_view_index"])
        data = np.load(stage / f"roi3d_frame{index:06d}.npz", allow_pickle=True)
        roi_records.append((index, data["points"], data["plane_world"], view["roi3d"]))
    for left in range(len(roi_records)):
        for right in range(left + 1, len(roi_records)):
            li, lp, lplane, lmeta = roi_records[left]
            ri, rp, rplane, rmeta = roi_records[right]
            center_distance = float(
                np.linalg.norm(np.asarray(lmeta["center_world"]) - np.asarray(rmeta["center_world"]))
            )
            ln = lplane[:3] / max(float(np.linalg.norm(lplane[:3])), 1e-12)
            rn = rplane[:3] / max(float(np.linalg.norm(rplane[:3])), 1e-12)
            plane_angle = float(np.degrees(np.arccos(np.clip(abs(float(ln @ rn)), -1, 1))))
            if center_distance > 5.0 or plane_angle > 10.0:
                continue
            pair_fig, pair_axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
            for axis, dims, labels in zip(
                pair_axes,
                [(0, 1), (0, 2), (1, 2)],
                [("world x", "world y"), ("world x", "world z"), ("world y", "world z")],
            ):
                axis.scatter(lp[:, dims[0]], lp[:, dims[1]], s=24, alpha=0.7, label=f"ROI {li}")
                axis.scatter(rp[:, dims[0]], rp[:, dims[1]], s=24, alpha=0.7, label=f"ROI {ri}")
                axis.set_xlabel(labels[0])
                axis.set_ylabel(labels[1])
                axis.grid(alpha=0.25)
                axis.legend()
            pair_fig.suptitle(
                f"Likely duplicate 3D ROIs {li}/{ri}: center distance={center_distance:.2f}, "
                f"plane angle={plane_angle:.2f} deg"
            )
            pair_fig.savefig(output / f"roi_{li:06d}_{ri:06d}_3d_overlap.png", dpi=160)
            plt.close(pair_fig)

    (output / "visualization_summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n")
    print(json.dumps({"output": str(output), "views": len(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()
