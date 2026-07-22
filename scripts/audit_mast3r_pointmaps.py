#!/usr/bin/env python3
"""Audit raw MASt3R pointmaps against the cameras written beside them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MAST3R_ROOT = REPO_ROOT / "mast3r"
if str(MAST3R_ROOT) not in sys.path:
    sys.path.insert(0, str(MAST3R_ROOT))

from colmap.read_write_model import (  # noqa: E402
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
)
from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402


def quantiles(values: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    selected = values[mask & np.isfinite(values)]
    if not selected.size:
        return {key: None for key in ("p01", "p50", "p90", "p99", "max")}
    result = np.quantile(selected, [0.01, 0.50, 0.90, 0.99])
    return {
        "p01": float(result[0]),
        "p50": float(result[1]),
        "p90": float(result[2]),
        "p99": float(result[3]),
        "max": float(selected.max()),
    }


def colorize(values: np.ndarray, mask: np.ndarray, high: float, log: bool = False) -> np.ndarray:
    normalized = np.zeros(values.shape, dtype=np.uint8)
    valid = mask & np.isfinite(values)
    if np.any(valid):
        display = np.log(np.maximum(values, 1e-6)) if log else values
        low = float(np.quantile(display[valid], 0.01))
        upper = float(np.quantile(display[valid], high)) if high <= 1.0 else float(high)
        denominator = max(upper - low, 1e-6)
        normalized[valid] = np.round(
            np.clip((display[valid] - low) / denominator, 0.0, 1.0) * 255.0
        ).astype(np.uint8)
    output = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    output[~valid] = 0
    return output


def tile(image: np.ndarray, title: str, width: int = 256, height: int = 144) -> np.ndarray:
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.full((height + 24, width, 3), 245, dtype=np.uint8)
    canvas[24:] = image
    cv2.putText(canvas, title[:40], (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)
    return canvas


def assess_coordinate_contract(
    projection_p50: np.ndarray,
    projection_p90: np.ndarray,
    *,
    all_finite: bool,
    max_chart_projection_p50_px: float,
    max_chart_projection_p90_px: float,
) -> dict[str, object]:
    """Check that every stored pointmap is expressed in its stored camera frame.

    Pointmap pixel ``(u, v)`` is a 3D point whose reprojection through the
    adjacent camera must return ``(u, v)``.  A two/four-pixel allowance is
    deliberately much looser than numerical precision, but rules out a
    similarity transform or camera/pointmap permutation before chart fitting
    can hide it.
    """
    failures = []
    if not all_finite:
        failures.append("non_finite_pointmap_values")
    if float(projection_p50.max()) > max_chart_projection_p50_px:
        failures.append("chart_projection_p50_exceeds_limit")
    if float(projection_p90.max()) > max_chart_projection_p90_px:
        failures.append("chart_projection_p90_exceeds_limit")
    return {
        "passed": not failures,
        "failures": failures,
        "max_chart_projection_p50_px": max_chart_projection_p50_px,
        "max_chart_projection_p90_px": max_chart_projection_p90_px,
        "observed_max_chart_projection_p50_px": float(projection_p50.max()),
        "observed_max_chart_projection_p90_px": float(projection_p90.max()),
    }


def read_sparse_export(mast3r_root: Path):
    sparse_root = mast3r_root / "sparse" / "0"
    return (
        read_cameras_binary(str(sparse_root / "cameras.bin")),
        read_images_binary(str(sparse_root / "images.bin")),
        read_points3D_binary(str(sparse_root / "points3D.bin")),
    )


def sparse_export_projection_records(
    mast3r_root: Path,
    *,
    sparse_export=None,
) -> list[dict[str, object]]:
    """Audit the COLMAP sparse export consumed by the Chart aligner.

    Raw pointmaps can be self-consistent while a later export accidentally
    attaches them to a different camera model.  The chart pipeline consumes
    ``sparse/0`` directly, so each stored 3-D point must also reproject to the
    2-D observation stored in that export.
    """
    cameras, images, points = sparse_export or read_sparse_export(mast3r_root)
    point_ids = np.fromiter((int(point_id) for point_id in points), dtype=np.int64)
    point_xyz = np.stack(
        [np.asarray(point.xyz, dtype=np.float64) for point in points.values()]
    )
    order = np.argsort(point_ids)
    point_ids = point_ids[order]
    point_xyz = point_xyz[order]

    records = []
    for image in images.values():
        observation_ids = np.asarray(image.point3D_ids, dtype=np.int64)
        keep = observation_ids > 0
        observation_ids = observation_ids[keep]
        observations = np.asarray(image.xys, dtype=np.float64)[keep]
        locations = np.searchsorted(point_ids, observation_ids)
        present = (
            (locations < point_ids.size)
            & (point_ids[np.minimum(locations, point_ids.size - 1)] == observation_ids)
        )
        locations = locations[present]
        observations = observations[present]
        points_xyz = point_xyz[locations]

        camera = cameras[image.camera_id]
        if camera.model == "PINHOLE":
            fx, fy, cx, cy = map(float, camera.params[:4])
        elif camera.model == "SIMPLE_PINHOLE":
            fx = fy = float(camera.params[0])
            cx, cy = map(float, camera.params[1:3])
        else:
            raise ValueError(
                f"Unsupported sparse-export camera model: {camera.model}"
            )
        camera_xyz = points_xyz @ qvec2rotmat(image.qvec).T + image.tvec[None]
        positive_depth = camera_xyz[:, 2] > 0.0
        projected = np.column_stack([
            fx * camera_xyz[:, 0] / np.maximum(camera_xyz[:, 2], 1e-12) + cx,
            fy * camera_xyz[:, 1] / np.maximum(camera_xyz[:, 2], 1e-12) + cy,
        ])
        errors = np.linalg.norm(projected - observations, axis=1)
        finite_values = (
            np.isfinite(points_xyz).all(axis=1)
            & np.isfinite(observations).all(axis=1)
            & np.isfinite(errors)
        )
        valid = finite_values & positive_depth
        records.append({
            "name": image.name,
            "observation_count": int(observation_ids.size),
            "finite_fraction": float(finite_values.mean()) if finite_values.size else 0.0,
            "positive_depth_fraction": float(positive_depth.mean()) if positive_depth.size else 0.0,
            "valid_projection_fraction": float(valid.mean()) if valid.size else 0.0,
            "projection_error_px": quantiles(errors, valid),
        })
    return records


def sparse_export_track_contract(
    mast3r_root: Path,
    *,
    sparse_export=None,
) -> dict[str, object]:
    """Verify that every Point3D track indexes its Image observation exactly.

    MASt3R retains a confidence-filtered subset of a dense pixel grid.  COLMAP
    tracks must index the resulting compact Image.xys list, rather than that
    original dense grid.  Projection-only checks cannot see this distinction.
    """
    _, images, points = sparse_export or read_sparse_export(mast3r_root)
    image_observations = {
        int(image_id): np.asarray(image.point3D_ids, dtype=np.int64)
        for image_id, image in images.items()
    }
    total = 0
    invalid = 0
    examples: list[dict[str, object]] = []
    for point in points.values():
        for image_id, point2d_index in zip(point.image_ids, point.point2D_idxs):
            total += 1
            observations = image_observations.get(int(image_id))
            index = int(point2d_index)
            valid = (
                observations is not None
                and 0 <= index < observations.size
                and int(observations[index]) == int(point.id)
            )
            if valid:
                continue
            invalid += 1
            if len(examples) < 20:
                examples.append({
                    "point3D_id": int(point.id),
                    "image_id": int(image_id),
                    "point2D_idx": index,
                    "image_observation_count": (
                        int(observations.size) if observations is not None else None
                    ),
                    "point3D_at_track_index": (
                        int(observations[index])
                        if observations is not None and 0 <= index < observations.size
                        else None
                    ),
                })
    return {
        "track_count": int(total),
        "invalid_track_count": int(invalid),
        "invalid_track_fraction": float(invalid / total) if total else 0.0,
        "passed": bool(invalid == 0),
        "examples": examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mast3r-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-pickle", type=Path)
    parser.add_argument("--mask-dataset-path", type=Path)
    parser.add_argument("--mask-indices", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument(
        "--require-coordinate-contract",
        action="store_true",
        help=(
            "Fail after writing the report unless every pointmap reprojects "
            "through its adjacent stored camera within the supplied limits."
        ),
    )
    parser.add_argument(
        "--max-chart-projection-p50-px",
        type=float,
        default=2.0,
        help="Maximum per-chart median reprojection error for the coordinate contract.",
    )
    parser.add_argument(
        "--max-chart-projection-p90-px",
        type=float,
        default=4.0,
        help="Maximum per-chart p90 reprojection error for the coordinate contract.",
    )
    parser.add_argument(
        "--audit-sparse-export",
        action="store_true",
        help="Also audit sparse/0 point tracks against the sparse-export cameras.",
    )
    parser.add_argument(
        "--require-sparse-export-coordinate-contract",
        action="store_true",
        help="Fail unless the sparse/0 export itself is camera/point consistent.",
    )
    parser.add_argument(
        "--require-sparse-export-track-contract",
        action="store_true",
        help="Fail unless every Point3D track indexes its Image.xys observation.",
    )
    parser.add_argument(
        "--max-sparse-export-projection-p50-px",
        type=float,
        default=4.0,
        help="Maximum per-chart median track reprojection error for sparse/0.",
    )
    parser.add_argument(
        "--max-sparse-export-projection-p90-px",
        type=float,
        default=8.0,
        help="Maximum per-chart p90 track reprojection error for sparse/0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.max_chart_projection_p50_px,
        args.max_chart_projection_p90_px,
        args.max_sparse_export_projection_p50_px,
        args.max_sparse_export_projection_p90_px,
    ) <= 0:
        raise ValueError("Coordinate-contract projection limits must be positive.")
    if (
        args.require_sparse_export_coordinate_contract
        or args.require_sparse_export_track_contract
    ):
        args.audit_sparse_export = True
    args.output.mkdir(parents=True, exist_ok=True)
    cameras = json.loads((args.mast3r_root / "cameras.json").read_text())
    mask_lookup = None
    if args.mask_pickle:
        mask_lookup = CambridgeMaskLookup(
            args.mask_dataset_path,
            args.mask_pickle,
            mask_indices=args.mask_indices,
        )

    records = []
    rows = []
    for index, filepath in enumerate(cameras["filepaths"]):
        name = Path(filepath).stem
        payload = json.loads((args.mast3r_root / "pointmaps" / f"{name}.json").read_text())
        confidence = np.asarray(payload["confs"], dtype=np.float32)
        height, width = confidence.shape
        points = np.asarray(payload["points"], dtype=np.float64).reshape(height, width, 3)
        rgb = np.asarray(payload["rgb"], dtype=np.float32)
        rgb_bgr = cv2.cvtColor(np.round(np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        semantic = np.ones((height, width), dtype=bool)
        if mask_lookup is not None:
            semantic = mask_lookup.get_mask(
                name, (height, width), torch.device("cpu")
            ).cpu().numpy().astype(bool)

        camera_to_world = np.asarray(cameras["cams2world"][index], dtype=np.float64)
        world_to_camera = np.linalg.inv(camera_to_world)
        camera_points = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        depth = camera_points[..., 2]
        finite = np.isfinite(points).all(axis=2) & np.isfinite(confidence)
        valid = finite & semantic & (confidence > 0.0) & (depth > 0.0)

        yy, xx = np.mgrid[:height, :width]
        focal = float(cameras["focals"][index])
        projected_x = focal * camera_points[..., 0] / np.maximum(depth, 1e-12) + width / 2.0
        projected_y = focal * camera_points[..., 1] / np.maximum(depth, 1e-12) + height / 2.0
        delta_x = projected_x - xx
        delta_y = projected_y - yy
        projection_error = np.hypot(delta_x, delta_y)

        record = {
            "index": index,
            "name": name,
            "shape": [height, width],
            "finite_fraction": float(finite.mean()),
            "semantic_keep_fraction": float(semantic.mean()),
            "positive_confidence_fraction": float((confidence > 0.0).mean()),
            "valid_projection_fraction": float(valid.mean()),
            "confidence": quantiles(confidence, valid),
            "camera_depth": quantiles(depth, valid),
            "projection_error_px": quantiles(projection_error, valid),
            "projection_delta_x_p50": float(np.median(delta_x[valid])) if np.any(valid) else None,
            "projection_delta_y_p50": float(np.median(delta_y[valid])) if np.any(valid) else None,
        }
        records.append(record)

        semantic_overlay = rgb_bgr.copy()
        semantic_overlay[~semantic] = (180, 30, 180)
        rows.append(np.hstack([
            tile(rgb_bgr, name),
            tile(colorize(depth, valid, 0.99, log=True), "camera log depth"),
            tile(colorize(confidence, valid, 0.99), "MASt3R confidence"),
            tile(colorize(projection_error, valid, 256.0), f"projection p50={record['projection_error_px']['p50']:.1f}px"),
            tile(semantic_overlay, f"semantic keep={semantic.mean():.3f}"),
        ]))

    projection_p50 = np.asarray([r["projection_error_px"]["p50"] for r in records], dtype=float)
    projection_p90 = np.asarray([r["projection_error_px"]["p90"] for r in records], dtype=float)
    all_finite = all(record["finite_fraction"] == 1.0 for record in records)
    coordinate_contract = assess_coordinate_contract(
        projection_p50,
        projection_p90,
        all_finite=all_finite,
        max_chart_projection_p50_px=args.max_chart_projection_p50_px,
        max_chart_projection_p90_px=args.max_chart_projection_p90_px,
    )
    sparse_export_records = None
    sparse_export_coordinate_contract = None
    sparse_export_track = None
    if args.audit_sparse_export:
        sparse_export = read_sparse_export(args.mast3r_root)
        sparse_export_records = sparse_export_projection_records(
            args.mast3r_root,
            sparse_export=sparse_export,
        )
        sparse_export_track = sparse_export_track_contract(
            args.mast3r_root,
            sparse_export=sparse_export,
        )
        sparse_p50 = np.asarray(
            [record["projection_error_px"]["p50"] for record in sparse_export_records],
            dtype=float,
        )
        sparse_p90 = np.asarray(
            [record["projection_error_px"]["p90"] for record in sparse_export_records],
            dtype=float,
        )
        sparse_export_coordinate_contract = assess_coordinate_contract(
            sparse_p50,
            sparse_p90,
            all_finite=all(
                record["finite_fraction"] == 1.0
                for record in sparse_export_records
            ),
            max_chart_projection_p50_px=args.max_sparse_export_projection_p50_px,
            max_chart_projection_p90_px=args.max_sparse_export_projection_p90_px,
        )
    report = {
        "mast3r_root": str(args.mast3r_root),
        "chart_count": len(records),
        "summary": {
            "all_finite": all_finite,
            "projection_p50_median_px": float(np.median(projection_p50)),
            "projection_p50_max_px": float(projection_p50.max()),
            "projection_p90_median_px": float(np.median(projection_p90)),
            "projection_p90_max_px": float(projection_p90.max()),
            "charts_projection_p50_gt_32px": int((projection_p50 > 32.0).sum()),
            "interpretation": (
                "This is a pre-alignment coordinate-frame contract: a pointmap "
                "pixel must reproject through its adjacent stored camera."
            ),
        },
        "coordinate_contract": coordinate_contract,
        "sparse_export_coordinate_contract": sparse_export_coordinate_contract,
        "sparse_export_track_contract": sparse_export_track,
        "sparse_export_charts": sparse_export_records,
        "charts": records,
    }
    (args.output / "pointmap_audit.json").write_text(json.dumps(report, indent=2))
    for start in range(0, len(rows), 6):
        cv2.imwrite(str(args.output / f"pointmap_audit_{start // 6 + 1:02d}.jpg"), np.vstack(rows[start:start + 6]))
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps({"coordinate_contract": coordinate_contract}, indent=2))
    if sparse_export_coordinate_contract is not None:
        print(
            json.dumps(
                {"sparse_export_coordinate_contract": sparse_export_coordinate_contract},
                indent=2,
            )
        )
    if sparse_export_track is not None:
        print(json.dumps({"sparse_export_track_contract": sparse_export_track}, indent=2))
    if args.require_coordinate_contract and not coordinate_contract["passed"]:
        raise SystemExit(
            "MASt3R pointmap/camera coordinate contract failed: "
            + ", ".join(coordinate_contract["failures"])
        )
    if (
        args.require_sparse_export_coordinate_contract
        and not sparse_export_coordinate_contract["passed"]
    ):
        raise SystemExit(
            "MASt3R sparse export/camera coordinate contract failed: "
            + ", ".join(sparse_export_coordinate_contract["failures"])
        )
    if args.require_sparse_export_track_contract and not sparse_export_track["passed"]:
        raise SystemExit(
            "MASt3R sparse export/track contract failed: "
            f"invalid={sparse_export_track['invalid_track_count']}/"
            f"{sparse_export_track['track_count']}"
        )


if __name__ == "__main__":
    main()
