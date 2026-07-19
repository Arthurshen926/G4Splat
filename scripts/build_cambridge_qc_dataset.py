#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "mast3r"))

from colmap.read_write_model import (  # noqa: E402
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    write_cameras_binary,
    write_cameras_text,
    write_images_binary,
    write_images_text,
    write_points3D_binary,
)
from view_quality_control.colmap_filter import filter_points3d_tracks  # noqa: E402
from view_quality_control.poses import COLMAP_POSE_POLICY_VERSION, normalize_qvec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a filtered posed COLMAP dataset from a view audit.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-point-track-length", type=int, default=2)
    parser.add_argument(
        "--retain-all-images",
        action="store_true",
        help=(
            "Keep every source training image and pose in the reconstruction "
            "dataset.  Audit hard-rejects are then recorded only as ineligible "
            "Chart candidates, rather than silently becoming a train split."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing QC dataset: {args.output}")

    audit = json.loads(args.audit.read_text())
    rejected_names = {
        record["image_name"] for record in audit["records"] if record["status"] == "hard_reject"
    }
    source_mapping = json.loads((args.source / "name_mapping.json").read_text())
    sparse = args.source / "sparse" / "0"
    cameras = read_cameras_binary(str(sparse / "cameras.bin"))
    images = read_images_binary(str(sparse / "images.bin"))
    points3d = read_points3D_binary(str(sparse / "points3D.bin"))

    kept_images = {}
    materially_normalized_names = []
    max_quaternion_norm_error = 0.0
    for image_id, image in images.items():
        if image.name in rejected_names and not args.retain_all_images:
            continue
        try:
            normalized_qvec = normalize_qvec(image.qvec)
        except ValueError as error:
            raise RuntimeError(f"Cannot normalize pose for {image.name}: {error}") from error
        norm_error = abs(float(sum(value * value for value in image.qvec)) ** 0.5 - 1.0)
        max_quaternion_norm_error = max(max_quaternion_norm_error, norm_error)
        if norm_error > 1e-4:
            materially_normalized_names.append(image.name)
        kept_images[image_id] = image._replace(qvec=normalized_qvec)
    missing_rejects = rejected_names - {image.name for image in images.values()}
    if missing_rejects:
        raise RuntimeError(f"Audit reject is absent from source dataset: {sorted(missing_rejects)[0]}")
    used_camera_ids = {image.camera_id for image in kept_images.values()}
    kept_cameras = {camera_id: cameras[camera_id] for camera_id in used_camera_ids}
    filtered_points, point_summary = filter_points3d_tracks(
        points3d,
        kept_images,
        min_track_length=args.min_point_track_length,
    )

    image_dir = args.output / "images"
    sparse_dir = args.output / "sparse" / "0"
    image_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)
    for image in kept_images.values():
        source_image = (args.source / "images" / image.name).resolve()
        (image_dir / image.name).symlink_to(source_image)

    write_cameras_binary(kept_cameras, str(sparse_dir / "cameras.bin"))
    write_cameras_text(kept_cameras, str(sparse_dir / "cameras.txt"))
    write_images_binary(kept_images, str(sparse_dir / "images.bin"))
    write_images_text(kept_images, str(sparse_dir / "images.txt"))
    write_points3D_binary(filtered_points, str(sparse_dir / "points3D.bin"))

    kept_mapping = {
        staged_name: source_name
        for staged_name, source_name in source_mapping.items()
        if args.retain_all_images or staged_name not in rejected_names
    }
    (args.output / "name_mapping.json").write_text(json.dumps(kept_mapping, indent=2))
    (args.output / "source_split.txt").write_text("\n".join(kept_mapping.values()) + "\n")
    manifest = {
        "pose_policy_version": COLMAP_POSE_POLICY_VERSION,
        "source": str(args.source),
        "audit": str(args.audit),
        "audit_sha256": hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        "input_images": len(images),
        "output_images": len(kept_images),
        "rejected_images": sorted(rejected_names),
        "candidate_chart_excluded_images": sorted(rejected_names),
        "reconstruction_image_policy": (
            "all_audited_train_images_retained"
            if args.retain_all_images
            else "hard_reject_images_removed"
        ),
        "normalized_quaternion_count": len(kept_images),
        "materially_normalized_quaternion_count": len(materially_normalized_names),
        "materially_normalized_quaternion_names": sorted(materially_normalized_names),
        "max_input_quaternion_norm_error": max_quaternion_norm_error,
        "point_filter": point_summary,
    }
    (args.output / "qc_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
