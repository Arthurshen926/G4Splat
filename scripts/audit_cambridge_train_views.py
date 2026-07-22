#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "mast3r"))

from colmap.read_write_model import read_cameras_binary, read_images_binary, read_points3D_binary  # noqa: E402
from matcha.cambridge_masks import load_mask_dict  # noqa: E402
from view_quality_control import (  # noqa: E402
    VIEW_AUDIT_POLICY_VERSION,
    classify_view_records,
    compute_image_quality,
    pose_integrity_metrics,
    qvec_to_rotmat,
)
from view_quality_control.provenance import audit_input_provenance  # noqa: E402


def camera_center_and_direction(image) -> tuple[np.ndarray, np.ndarray]:
    rotation = qvec_to_rotmat(image.qvec)
    center = -rotation.T @ image.tvec
    direction = rotation.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    direction /= max(np.linalg.norm(direction), 1e-12)
    return center, direction


def build_pose_features(ordered_images: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers, directions = zip(*(camera_center_and_direction(image) for image in ordered_images))
    centers = np.stack(centers)
    directions = np.stack(directions)
    extent = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
    extent = max(extent, 1e-12)
    normalized_centers = (centers - centers.mean(axis=0, keepdims=True)) / extent
    features = np.concatenate([normalized_centers, 0.35 * directions], axis=1)
    return features, normalized_centers, directions


def farthest_point_indices(features: np.ndarray, count: int) -> list[int]:
    first = int(np.argmax(np.linalg.norm(features - features.mean(axis=0), axis=1)))
    selected = [first]
    minimum = np.linalg.norm(features - features[first], axis=1)
    minimum[first] = -1.0
    while len(selected) < count:
        index = int(np.argmax(minimum))
        selected.append(index)
        minimum = np.minimum(minimum, np.linalg.norm(features - features[index], axis=1))
        minimum[selected] = -1.0
    return selected


def resize_keep_mask(mask, shape: tuple[int, int]) -> np.ndarray:
    array = mask.detach().cpu().numpy().astype(np.uint8, copy=False)
    height, width = shape
    if array.shape != (height, width):
        array = cv2.resize(array, (width, height), interpolation=cv2.INTER_NEAREST)
    return array > 0


def combined_keep_mask(mask_tuple: tuple, indices: tuple[int, ...], shape: tuple[int, int]) -> np.ndarray:
    keep = np.ones(shape, dtype=np.bool_)
    for index in indices:
        if index < 0 or index >= len(mask_tuple):
            raise RuntimeError(f"Mask index {index} is outside tuple length {len(mask_tuple)}")
        keep &= resize_keep_mask(mask_tuple[index], shape)
    return keep


def invalid_ratio(mask_tuple: tuple, index: int) -> float:
    if index < 0 or index >= len(mask_tuple):
        return 0.0
    mask = mask_tuple[index].detach().cpu().numpy().astype(np.bool_, copy=False)
    return float(1.0 - np.count_nonzero(mask) / mask.size)


def pinhole_parameters(camera) -> tuple[float, float, float, float]:
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = map(float, camera.params[:4])
    elif camera.model == "SIMPLE_PINHOLE":
        focal, cx, cy = map(float, camera.params[:3])
        fx = fy = focal
    else:
        raise RuntimeError(f"Unsupported staged camera model: {camera.model}")
    return fx, fy, cx, cy


def projection_valid_ratio(image, camera, point_ids: set[int], points3d: dict) -> float:
    if not point_ids:
        return 0.0
    xyz = np.stack([points3d[point_id].xyz for point_id in point_ids])
    camera_xyz = xyz @ qvec_to_rotmat(image.qvec).T + image.tvec[None]
    depth = camera_xyz[:, 2]
    fx, fy, cx, cy = pinhole_parameters(camera)
    positive = depth > 1e-6
    safe_depth = np.where(positive, depth, 1.0)
    pixel_x = fx * camera_xyz[:, 0] / safe_depth + cx
    pixel_y = fy * camera_xyz[:, 1] / safe_depth + cy
    valid = (
        positive
        & (pixel_x >= 0.0)
        & (pixel_x < camera.width)
        & (pixel_y >= 0.0)
        & (pixel_y < camera.height)
    )
    return float(valid.mean())


def robust_intrinsics_z(records: list[dict]) -> dict[str, float]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["sequence"]].append(record)
    result = {}
    for sequence_records in grouped.values():
        values = np.asarray(
            [
                [
                    record["fx_over_width"],
                    record["fy_over_height"],
                    record["cx_over_width"],
                    record["cy_over_height"],
                ]
                for record in sequence_records
            ],
            dtype=np.float64,
        )
        median = np.median(values, axis=0)
        mad = np.median(np.abs(values - median), axis=0)
        scale = np.maximum(1.4826 * mad, 1e-6)
        zscores = np.max(np.abs(values - median) / scale, axis=1)
        for record, zscore in zip(sequence_records, zscores):
            result[record["image_name"]] = float(zscore)
    return result


def write_csv(path: Path, records: list[dict]) -> None:
    scalar_keys = sorted(
        key for key in records[0]
        if not isinstance(records[0][key], (dict, list))
    )
    list_keys = ["severe_reasons", "mild_reasons"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys + list_keys)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in scalar_keys}
            for key in list_keys:
                row[key] = ";".join(record.get(key, []))
            writer.writerow(row)


def render_contact_sheet(dataset_path: Path, records: list[dict], output_path: Path, count: int = 50) -> None:
    selected = sorted(records, key=lambda item: (-item["risk_score"], item["image_name"]))[:count]
    columns, tile_width, image_height, label_height = 5, 320, 180, 58
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * (image_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for order, record in enumerate(selected):
        with Image.open(dataset_path / "images" / record["image_name"]) as source:
            tile = ImageOps.contain(source.convert("RGB"), (tile_width, image_height))
        canvas = Image.new("RGB", (tile_width, image_height), (24, 24, 24))
        canvas.paste(tile, ((tile_width - tile.width) // 2, (image_height - tile.height) // 2))
        x = (order % columns) * tile_width
        y = (order // columns) * (image_height + label_height)
        sheet.paste(canvas, (x, y))
        reasons = ",".join(record["severe_reasons"] + record["mild_reasons"])
        draw.text((x + 4, y + image_height + 2), f"{record['status']} r={record['risk_score']:.3f}", fill="black")
        draw.text((x + 4, y + image_height + 18), record["image_name"], fill="black")
        draw.text((x + 4, y + image_height + 34), reasons[:48], fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit posed Cambridge training views without training a model.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mask-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-mask-indices", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--thing-mask-index", type=int, default=0)
    parser.add_argument("--sky-mask-index", type=int, default=1)
    parser.add_argument("--tree-mask-index", type=int, default=3)
    parser.add_argument("--pose-clusters", type=int, default=8)
    parser.add_argument("--neighbor-count", type=int, default=8)
    parser.add_argument("--max-image-width", type=int, default=960)
    parser.add_argument("--max-reject-fraction", type=float, default=0.05)
    parser.add_argument("--chart-scene", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sparse = args.dataset / "sparse" / "0"
    mapping = json.loads((args.dataset / "name_mapping.json").read_text())
    masks = load_mask_dict(args.mask_pickle)
    cameras = read_cameras_binary(str(sparse / "cameras.bin"))
    images_by_id = read_images_binary(str(sparse / "images.bin"))
    points3d = read_points3D_binary(str(sparse / "points3D.bin"))
    images_by_name = {image.name: image for image in images_by_id.values()}
    image_names = sorted(mapping)
    ordered_images = [images_by_name[name] for name in image_names]

    track_sets = {image.id: set() for image in ordered_images}
    for point_id, point in points3d.items():
        for image_id in point.image_ids:
            image_id = int(image_id)
            if image_id in track_sets:
                track_sets[image_id].add(int(point_id))

    features, _, _ = build_pose_features(ordered_images)
    anchor_indices = farthest_point_indices(features, min(args.pose_clusters, len(features)))
    anchor_features = features[np.asarray(anchor_indices)]
    cluster_labels = np.argmin(
        np.linalg.norm(features[:, None] - anchor_features[None], axis=2),
        axis=1,
    )
    pairwise = np.linalg.norm(features[:, None] - features[None], axis=2)
    np.fill_diagonal(pairwise, np.inf)
    neighbor_count = min(args.neighbor_count, len(features) - 1)
    if neighbor_count > 0:
        neighbor_indices = np.argpartition(
            pairwise,
            neighbor_count - 1,
            axis=1,
        )[:, :neighbor_count]
    else:
        neighbor_indices = np.empty((len(features), 0), dtype=np.int64)

    chart_names = set()
    if args.chart_scene is not None:
        chart_names = {path.name for path in (args.chart_scene / "images").iterdir() if path.is_file()}

    records = []
    for index, (image_name, image) in enumerate(zip(image_names, ordered_images)):
        source_name = mapping[image_name]
        if source_name not in masks:
            raise RuntimeError(f"{source_name} is absent from {args.mask_pickle}")
        image_bgr = cv2.imread(str(args.dataset / "images" / image_name), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(args.dataset / "images" / image_name)
        if args.max_image_width > 0 and image_bgr.shape[1] > args.max_image_width:
            scale = args.max_image_width / image_bgr.shape[1]
            image_bgr = cv2.resize(
                image_bgr,
                (args.max_image_width, round(image_bgr.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        mask_tuple = masks[source_name]
        quality_mask = combined_keep_mask(
            mask_tuple,
            tuple(args.quality_mask_indices),
            image_bgr.shape[:2],
        )
        quality = compute_image_quality(image_bgr, quality_mask)

        camera = cameras[image.camera_id]
        fx, fy, cx, cy = pinhole_parameters(camera)
        current_tracks = track_sets[image.id]
        shared_counts = [
            len(current_tracks.intersection(track_sets[ordered_images[neighbor].id]))
            for neighbor in neighbor_indices[index]
        ]
        thing_ratio = invalid_ratio(mask_tuple, args.thing_mask_index)
        sky_ratio = invalid_ratio(mask_tuple, args.sky_mask_index)
        tree_ratio = invalid_ratio(mask_tuple, args.tree_mask_index)
        exposure_warning = max(
            quality["dark_ratio"],
            quality["clipped_ratio"],
            abs(quality["gray_mean"] - 0.5) * 2.0,
        )
        semantic_warning = max(thing_ratio, tree_ratio)
        record = {
            "image_name": image_name,
            "source_name": source_name,
            "image_id": int(image.id),
            "sequence": source_name.split("/", 1)[0],
            "pose_cluster": int(cluster_labels[index]),
            "is_chart": image_name in chart_names,
            **quality,
            "thing_invalid_ratio": thing_ratio,
            "sky_invalid_ratio": sky_ratio,
            "tree_invalid_ratio": tree_ratio,
            "track_count": len(current_tracks),
            "shared_track_max": max(shared_counts) if shared_counts else 0,
            "shared_track_median": float(np.median(shared_counts)) if shared_counts else 0.0,
            "shared_supported_neighbors": sum(count > 0 for count in shared_counts),
            "track_projection_valid_ratio": projection_valid_ratio(
                image,
                camera,
                current_tracks,
                points3d,
            ),
            "fx_over_width": fx / camera.width,
            "fy_over_height": fy / camera.height,
            "cx_over_width": cx / camera.width,
            "cy_over_height": cy / camera.height,
            "exposure_warning": exposure_warning,
            "semantic_warning": semantic_warning,
            **pose_integrity_metrics(image.qvec),
        }
        records.append(record)
        if (index + 1) % 100 == 0:
            print(f"Audited {index + 1}/{len(image_names)} views", flush=True)

    intrinsics_z = robust_intrinsics_z(records)
    for record in records:
        record["intrinsics_robust_z"] = intrinsics_z[record["image_name"]]

    classified, summary = classify_view_records(
        records,
        max_reject_fraction=args.max_reject_fraction,
    )
    summary["hard_reject_charts"] = [
        record["image_name"]
        for record in classified
        if record["status"] == "hard_reject" and record["is_chart"]
    ]
    summary["soft_keep_charts"] = [
        record["image_name"]
        for record in classified
        if record["status"] == "soft_keep" and record["is_chart"]
    ]

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "audit_policy_version": VIEW_AUDIT_POLICY_VERSION,
        "dataset": str(args.dataset),
        "mask_pickle": str(args.mask_pickle),
        "quality_mask_indices": args.quality_mask_indices,
        "input_provenance": audit_input_provenance(
            args.dataset,
            args.mask_pickle,
            quality_mask_indices=args.quality_mask_indices,
            thing_mask_index=args.thing_mask_index,
            sky_mask_index=args.sky_mask_index,
            tree_mask_index=args.tree_mask_index,
            pose_clusters=args.pose_clusters,
            neighbor_count=args.neighbor_count,
            max_image_width=args.max_image_width,
            max_reject_fraction=args.max_reject_fraction,
        ),
        "summary": summary,
        "records": classified,
    }
    (args.output / "view_audit.json").write_text(json.dumps(payload, indent=2))
    write_csv(args.output / "view_metrics.csv", classified)
    for status in ["hard_reject", "soft_keep", "clean"]:
        names = [record["image_name"] for record in classified if record["status"] == status]
        (args.output / f"{status}.txt").write_text("\n".join(names) + ("\n" if names else ""))
    render_contact_sheet(args.dataset, classified, args.output / "worst_views.jpg")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
