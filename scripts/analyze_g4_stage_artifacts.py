#!/usr/bin/env python3
"""Audit G4Splat chart, See3D, and global-plane artifacts without training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def read_depth(path: Path) -> np.ndarray:
    return read_image(path, cv2.IMREAD_UNCHANGED).astype(np.float32)


def finite_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def format_metric(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def relative_depth_stats(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float | None]:
    valid = (
        np.isfinite(reference)
        & np.isfinite(estimate)
        & (reference > 1e-6)
        & (estimate > 1e-6)
    )
    if not np.any(valid):
        return {
            "valid_ratio": 0.0,
            "median": None,
            "p95": None,
            "gt_10pct": None,
            "gt_25pct": None,
            "log_correlation": None,
        }
    relative = np.abs(estimate[valid] - reference[valid]) / np.maximum(
        np.abs(reference[valid]), 1e-6
    )
    log_reference = np.log(reference[valid])
    log_estimate = np.log(estimate[valid])
    correlation = None
    if np.std(log_reference) > 1e-8 and np.std(log_estimate) > 1e-8:
        correlation = float(np.corrcoef(log_reference, log_estimate)[0, 1])
    return {
        "valid_ratio": float(valid.mean()),
        "median": finite_float(np.median(relative)),
        "p95": finite_float(np.percentile(relative, 95)),
        "gt_10pct": float(np.mean(relative > 0.10)),
        "gt_25pct": float(np.mean(relative > 0.25)),
        "log_correlation": finite_float(correlation) if correlation is not None else None,
    }


def plane_stats(mask: np.ndarray) -> dict[str, float | int]:
    labels, counts = np.unique(mask, return_counts=True)
    plane_counts = counts[labels != 0]
    pixel_count = mask.size
    return {
        "plane_count": int(len(plane_counts)),
        "plane_coverage": float(plane_counts.sum() / pixel_count) if len(plane_counts) else 0.0,
        "largest_plane": float(plane_counts.max() / pixel_count) if len(plane_counts) else 0.0,
    }


def image_stats(image_bgr: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return {
        "rgb_mean": float(gray.mean()),
        "rgb_std": float(gray.std()),
        "dark_fraction": float(np.mean(gray < 0.03)),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
    }


def visible_mae(first: np.ndarray, second: np.ndarray, visible: np.ndarray) -> float | None:
    if not np.any(visible):
        return None
    difference = np.abs(first.astype(np.float32) - second.astype(np.float32)) / 255.0
    return float(difference[visible].mean())


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 1e-6)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth[valid], [2, 98])
        if high <= low:
            high = low + 1e-6
        values = np.clip((depth - low) / (high - low), 0.0, 1.0)
        normalized[valid] = np.round(values[valid] * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def colorize_relative_delta(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    valid = (
        np.isfinite(reference)
        & np.isfinite(estimate)
        & (reference > 1e-6)
        & (estimate > 1e-6)
    )
    relative = np.zeros(reference.shape, dtype=np.float32)
    relative[valid] = np.abs(estimate[valid] - reference[valid]) / np.maximum(
        np.abs(reference[valid]), 1e-6
    )
    normalized = np.clip(relative / 0.50, 0.0, 1.0)
    colored = cv2.applyColorMap(np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def overlay_invalid(image: np.ndarray, keep: np.ndarray) -> np.ndarray:
    keep = keep.astype(bool)
    if keep.shape != image.shape[:2]:
        keep = cv2.resize(
            keep.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    overlay = image.copy()
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    overlay[~keep] = cv2.addWeighted(overlay, 0.35, red, 0.65, 0)[~keep]
    return overlay


def make_tile(image: np.ndarray, title: str, width: int = 260, height: int = 165) -> np.ndarray:
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.full((height + 30, width, 3), 245, dtype=np.uint8)
    canvas[30:] = image
    cv2.putText(
        canvas,
        title[:42],
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (15, 15, 15),
        1,
        cv2.LINE_AA,
    )
    return canvas


def save_contact_sheet(rows: list[list[np.ndarray]], path: Path) -> None:
    if not rows:
        return
    width = max(len(row) for row in rows)
    tile_shape = rows[0][0].shape
    blank = np.full(tile_shape, 245, dtype=np.uint8)
    padded = [row + [blank] * (width - len(row)) for row in rows]
    sheet = np.vstack([np.hstack(row) for row in padded])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        return
    fields: list[str] = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def mean_record_value(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record[key] for record in records if record.get(key) is not None]
    return float(np.mean(values)) if values else None


def load_normal_for_plane(plane_root: Path, view_id: int, plane_id: int) -> np.ndarray | None:
    mask = np.load(plane_root / f"plane_mask_frame{view_id:06d}.npy")
    normal = np.load(plane_root / f"depth_normal_world_frame{view_id:06d}.npy")
    selected = normal[mask == plane_id]
    selected = selected[np.all(np.isfinite(selected), axis=1)]
    selected = selected[np.linalg.norm(selected, axis=1) > 1e-6]
    if not len(selected):
        return None
    representative = np.median(selected, axis=0)
    norm = np.linalg.norm(representative)
    if not np.isfinite(norm) or norm <= 1e-6:
        return None
    return representative / norm


def max_undirected_normal_angle(normals: Iterable[np.ndarray]) -> float | None:
    normals = list(normals)
    if len(normals) < 2:
        return None
    maximum = 0.0
    for first_index, first in enumerate(normals[:-1]):
        for second in normals[first_index + 1 :]:
            cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
            maximum = max(maximum, math.degrees(math.acos(cosine)))
    return maximum


def analyze_global_planes(plane_root: Path, train_views: int) -> dict[str, Any]:
    mapping_path = plane_root / "global_3Dplane_ID_dict.json"
    mapping = json.loads(mapping_path.read_text())
    groups: list[dict[str, Any]] = []
    for global_id, entries in mapping.items():
        normals = []
        areas = []
        normalized_entries = []
        for view_id, plane_id in entries:
            view_id, plane_id = int(view_id), int(plane_id)
            mask = np.load(plane_root / f"plane_mask_frame{view_id:06d}.npy")
            areas.append(float(np.mean(mask == plane_id)))
            normal = load_normal_for_plane(plane_root, view_id, plane_id)
            if normal is not None:
                normals.append(normal)
            normalized_entries.append([view_id, plane_id])
        real_count = sum(view_id < train_views for view_id, _ in normalized_entries)
        pseudo_count = len(normalized_entries) - real_count
        groups.append(
            {
                "global_plane_id": int(global_id),
                "local_plane_count": len(normalized_entries),
                "real_local_plane_count": real_count,
                "pseudo_local_plane_count": pseudo_count,
                "mixed_real_pseudo": bool(real_count and pseudo_count),
                "area_sum": float(sum(areas)),
                "largest_local_area": float(max(areas, default=0.0)),
                "max_normal_angle_deg": finite_float(max_undirected_normal_angle(normals))
                if len(normals) >= 2
                else None,
                "entries": normalized_entries,
            }
        )

    mixed = [group for group in groups if group["mixed_real_pseudo"]]
    mixed_multi = [
        group
        for group in mixed
        if group["max_normal_angle_deg"] is not None
    ]
    angles = np.asarray([group["max_normal_angle_deg"] for group in mixed_multi], dtype=np.float64)
    return {
        "global_plane_count": len(groups),
        "groups_with_pseudo": sum(group["pseudo_local_plane_count"] > 0 for group in groups),
        "pseudo_only_groups": sum(
            group["pseudo_local_plane_count"] > 0 and group["real_local_plane_count"] == 0
            for group in groups
        ),
        "mixed_real_pseudo_groups": len(mixed),
        "mixed_groups_with_normal_comparison": len(mixed_multi),
        "mixed_angle_gt_15_deg": int(np.sum(angles > 15.0)) if len(angles) else 0,
        "mixed_angle_gt_30_deg": int(np.sum(angles > 30.0)) if len(angles) else 0,
        "mixed_angle_gt_45_deg": int(np.sum(angles > 45.0)) if len(angles) else 0,
        "mixed_angle_gt_60_deg": int(np.sum(angles > 60.0)) if len(angles) else 0,
        "mixed_angle_median_deg": finite_float(np.median(angles)) if len(angles) else None,
        "mixed_angle_p90_deg": finite_float(np.percentile(angles, 90)) if len(angles) else None,
        "mixed_angle_max_deg": finite_float(np.max(angles)) if len(angles) else None,
        "worst_groups": sorted(
            mixed_multi,
            key=lambda group: group["max_normal_angle_deg"],
            reverse=True,
        )[:20],
    }


def analyze_pseudo_views(
    mast3r_root: Path,
    guarded_root: Path | None,
    stage: int,
) -> tuple[list[dict[str, Any]], list[list[np.ndarray]]]:
    stage_root = mast3r_root / "see3d_render" / f"stage{stage}"
    selection_root = stage_root / "select-gs"
    raw_inpaint_root = stage_root / "select-gs-inpainted"
    plane_root = mast3r_root / "plane-refine-depths"
    camera_file = stage_root / f"stage{stage}_see3d_cameras.npz"
    camera_data = np.load(camera_file)
    train_views = int(camera_data["train_views"])
    pseudo_views = int(camera_data["n_views"])
    protected_root = (
        guarded_root / "see3d_render" / "inpainted_images" if guarded_root is not None else None
    )

    records: list[dict[str, Any]] = []
    rows: list[list[np.ndarray]] = []
    for local_id in range(pseudo_views):
        global_id = train_views + local_id
        raw = read_image(selection_root / f"ori_warp_frame{local_id:06d}.png")
        visible = read_image(
            selection_root / f"mask_frame{local_id:06d}.png", cv2.IMREAD_GRAYSCALE
        ) > 127
        generated = read_image(raw_inpaint_root / f"predict_warp_frame{local_id:06d}.png")
        alpha = np.load(selection_root / f"alpha_{local_id:06d}.npy")
        depth_alignment_visible = alpha > 0.90
        protected = None
        if protected_root is not None:
            protected_path = protected_root / f"predict_warp_frame{local_id:06d}.png"
            if protected_path.exists():
                protected = read_image(protected_path)
        plane_mask = np.load(plane_root / f"plane_mask_frame{global_id:06d}.npy")
        raw_depth = read_depth(selection_root / f"depth_frame{local_id:06d}.tiff")
        refined_depth = read_depth(plane_root / f"refine_depth_frame{global_id:06d}.tiff")
        confidence = read_image(
            plane_root / f"confident_map_frame{global_id:06d}.png", cv2.IMREAD_GRAYSCALE
        ) > 127
        record: dict[str, Any] = {
            "local_view_id": local_id,
            "global_view_id": global_id,
            "visible_ratio": float(visible.mean()),
            "depth_alignment_alpha_ratio": float(depth_alignment_visible.mean()),
            "depth_alignment_unsupported_ratio": float(
                np.mean(depth_alignment_visible & ~visible)
            ),
            **image_stats(raw),
            **plane_stats(plane_mask),
            "confidence_ratio": float(confidence.mean()),
            "see3d_visible_mae": visible_mae(raw, generated, visible),
            "protected_visible_mae": visible_mae(raw, protected, visible)
            if protected is not None
            else None,
        }
        record.update(
            {
                f"see3d_{key}": value
                for key, value in image_stats(generated).items()
            }
        )
        record.update(
            {
                f"refined_vs_gs_depth_{key}": value
                for key, value in relative_depth_stats(raw_depth, refined_depth).items()
            }
        )
        records.append(record)

        row_title = (
            f"p{local_id}/g{global_id} vis={record['visible_ratio']:.2f} "
            f"planes={record['plane_count']}"
        )
        rows.append(
            [
                make_tile(raw, f"{row_title} | raw GS"),
                make_tile(overlay_invalid(raw, visible), "red = unobserved"),
                make_tile(generated, f"raw See3D | vis MAE={record['see3d_visible_mae']:.3f}"),
                make_tile(
                    protected if protected is not None else generated,
                    f"protected | vis MAE={record['protected_visible_mae'] or 0.0:.3f}",
                ),
                make_tile(colorize_depth(raw_depth), "GS depth"),
                make_tile(colorize_depth(refined_depth), "plane-refined depth"),
                make_tile(
                    read_image(plane_root / f"plane_vis_frame{global_id:06d}.png"),
                    f"plane cov={record['plane_coverage']:.2f}",
                ),
            ]
        )
    return records, rows


def analyze_real_charts(
    mast3r_root: Path,
    mask_pickle: Path,
    mask_dataset_path: Path,
    mask_indices: list[int],
    chart_ids: list[int],
) -> tuple[list[dict[str, Any]], list[list[np.ndarray]]]:
    plane_root = mast3r_root / "plane-refine-depths"
    cameras = json.loads((mast3r_root / "cameras.json").read_text())
    filepaths = cameras["filepaths"]
    lookup = CambridgeMaskLookup(mask_dataset_path, mask_pickle, mask_indices=mask_indices)
    records: list[dict[str, Any]] = []
    rows: list[list[np.ndarray]] = []
    visual_chart_ids = set(chart_ids)
    for chart_id in range(len(filepaths)):
        image_name = Path(filepaths[chart_id]).name
        rgb = read_image(plane_root / f"rgb_frame{chart_id:06d}.png")
        height, width = rgb.shape[:2]
        pure_semantic = lookup.get_mask(
            image_name, (height, width), torch.device("cpu")
        ).numpy()
        combined = np.load(plane_root / f"semantic_keep_frame{chart_id:06d}.npy") > 0
        chart_depth = read_depth(plane_root / f"depth_frame{chart_id:06d}.tiff")
        mono_depth = read_depth(plane_root / f"mono_depth_frame{chart_id:06d}.tiff")
        refined_depth = read_depth(plane_root / f"refine_depth_frame{chart_id:06d}.tiff")
        plane_mask = np.load(plane_root / f"plane_mask_frame{chart_id:06d}.npy")
        record: dict[str, Any] = {
            "chart_id": chart_id,
            "image_name": image_name,
            "pure_semantic_keep_ratio": float(pure_semantic.mean()),
            "combined_geometry_keep_ratio": float(combined.mean()),
            "extra_geometry_reject_ratio": float(pure_semantic.mean() - combined.mean()),
            **plane_stats(plane_mask),
        }
        record.update(
            {
                f"refined_vs_chart_{key}": value
                for key, value in relative_depth_stats(chart_depth, refined_depth).items()
            }
        )
        record.update(
            {
                f"mono_vs_chart_{key}": value
                for key, value in relative_depth_stats(chart_depth, mono_depth).items()
            }
        )
        records.append(record)
        if chart_id not in visual_chart_ids:
            continue
        rows.append(
            [
                make_tile(rgb, f"chart {chart_id}: {image_name}"),
                make_tile(
                    overlay_invalid(rgb, pure_semantic),
                    f"semantic keep={record['pure_semantic_keep_ratio']:.2f}",
                ),
                make_tile(
                    overlay_invalid(rgb, combined),
                    f"semantic+pointmap={record['combined_geometry_keep_ratio']:.2f}",
                ),
                make_tile(colorize_depth(chart_depth), "aligned chart depth"),
                make_tile(colorize_depth(mono_depth), "aligned DAV2 depth"),
                make_tile(
                    colorize_relative_delta(chart_depth, refined_depth),
                    f"plane delta p95={format_metric(record['refined_vs_chart_p95'])}",
                ),
                make_tile(
                    read_image(plane_root / f"plane_vis_frame{chart_id:06d}.png"),
                    f"planes={record['plane_count']} cov={record['plane_coverage']:.2f}",
                ),
            ]
        )
    return records, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mast3r-root", type=Path, required=True)
    parser.add_argument(
        "--guarded-mast3r-root",
        type=Path,
        help="Optional protected-See3D mast3r_sfm directory.",
    )
    parser.add_argument("--mask-pickle", type=Path, required=True)
    parser.add_argument("--mask-dataset-path", type=Path, required=True)
    parser.add_argument("--mask-indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument(
        "--chart-ids",
        type=int,
        nargs="+",
        default=[0, 5, 9, 13, 14, 17, 20],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pseudo_records, pseudo_rows = analyze_pseudo_views(
        args.mast3r_root,
        args.guarded_mast3r_root,
        args.stage,
    )
    real_records, real_rows = analyze_real_charts(
        args.mast3r_root,
        args.mask_pickle,
        args.mask_dataset_path,
        args.mask_indices,
        args.chart_ids,
    )
    camera_data = np.load(
        args.mast3r_root
        / "see3d_render"
        / f"stage{args.stage}"
        / f"stage{args.stage}_see3d_cameras.npz"
    )
    plane_report = analyze_global_planes(
        args.mast3r_root / "plane-refine-depths",
        int(camera_data["train_views"]),
    )

    write_csv(pseudo_records, args.output_dir / "pseudo_view_metrics.csv")
    write_csv(real_records, args.output_dir / "real_chart_metrics.csv")
    save_contact_sheet(pseudo_rows, args.output_dir / "pseudo_stage_contact_sheet.jpg")
    save_contact_sheet(real_rows, args.output_dir / "real_chart_contact_sheet.jpg")
    report = {
        "inputs": {
            "mast3r_root": str(args.mast3r_root.resolve()),
            "guarded_mast3r_root": str(args.guarded_mast3r_root.resolve())
            if args.guarded_mast3r_root
            else None,
            "mask_pickle": str(args.mask_pickle.resolve()),
            "mask_dataset_path": str(args.mask_dataset_path.resolve()),
            "mask_indices": args.mask_indices,
            "stage": args.stage,
        },
        "pseudo_views": pseudo_records,
        "pseudo_summary": {
            "view_count": len(pseudo_records),
            "mean_visible_ratio": mean_record_value(pseudo_records, "visible_ratio"),
            "mean_depth_alignment_alpha_ratio": mean_record_value(
                pseudo_records, "depth_alignment_alpha_ratio"
            ),
            "mean_depth_alignment_unsupported_ratio": mean_record_value(
                pseudo_records, "depth_alignment_unsupported_ratio"
            ),
            "mean_see3d_visible_mae": mean_record_value(
                pseudo_records, "see3d_visible_mae"
            ),
            "mean_protected_visible_mae": mean_record_value(
                pseudo_records, "protected_visible_mae"
            ),
        },
        "real_charts": real_records,
        "real_chart_summary": {
            "chart_count": len(real_records),
            "mean_pure_semantic_keep_ratio": mean_record_value(
                real_records, "pure_semantic_keep_ratio"
            ),
            "mean_combined_geometry_keep_ratio": mean_record_value(
                real_records, "combined_geometry_keep_ratio"
            ),
            "mean_extra_geometry_reject_ratio": mean_record_value(
                real_records, "extra_geometry_reject_ratio"
            ),
        },
        "global_planes": plane_report,
    }
    (args.output_dir / "stage_artifact_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(plane_report, indent=2))
    print(f"Wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
