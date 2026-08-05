#!/usr/bin/env python3
"""Add safe sequence-local DAV2 witnesses without rebuilding the visual hull.

Strict per-view inverse-depth alignment can fail when a foreground crown
leaves too little rigid support.  This tool recovers the exact affine
alignment of already accepted DAV2 rows from an immutable initialization,
interpolates only between two accepted frames in the same sequence, and adds
exact-camera dynamic leaves plus ownerless hit rays.  Interpolated rows never
become canonical geometry or localization landmarks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "2d-gaussian-splatting"))

from outdoor.evidence_store import artifact_path, load_evidence_store
from outdoor.foliage_geometry import (
    enforce_strict_foliage_ray_intervals,
    half_open_raster_coordinates,
    quaternion_to_rotation,
    read_cameras_binary,
    read_images_binary,
    strict_foliage_hit_interval_bounds,
)
from outdoor.role_aware_initialization import (
    CambridgeMaskLookup,
    TEMPORAL_DAV2_AUGMENTATION_VERSION,
)


PROTOCOL = TEMPORAL_DAV2_AUGMENTATION_VERSION
FRAME_PATTERN = re.compile(r"^(?P<sequence>.+)__frame(?P<frame>[0-9]+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_view(name: str) -> tuple[str, int] | None:
    match = FRAME_PATTERN.match(Path(name).stem)
    if match is None:
        return None
    return match.group("sequence"), int(match.group("frame"))


def _camera_parameters(camera: dict, shape: tuple[int, int]):
    height, width = map(int, shape)
    if camera["model"] == "SIMPLE_PINHOLE":
        fx = fy = float(camera["params"][0])
        cx, cy = map(float, camera["params"][1:3])
    else:
        fx, fy, cx, cy = map(float, camera["params"][:4])
    sx = width / float(camera["width"])
    sy = height / float(camera["height"])
    return fx * sx, fy * sy, cx * sx, cy * sy


def _resize_depth(path: Path) -> np.ndarray:
    depth = np.squeeze(np.asarray(np.load(path), dtype=np.float32))
    if depth.ndim != 2:
        raise RuntimeError(f"DAV2 depth is not two-dimensional: {path}")
    return (
        F.interpolate(
            torch.from_numpy(depth)[None, None],
            size=(135, 240),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        .numpy()
        .astype(np.float32)
    )


def _recover_alignments(
    payload: dict,
    camera_views: list[dict],
    dav2_records: dict,
    *,
    audit: dict | None = None,
) -> dict[str, dict]:
    """Recover ``rho = alpha + beta / dav2`` from accepted DAV2 rows."""
    source = payload["initialization_source"]
    rows = torch.nonzero(source == 3, as_tuple=False).flatten().numpy()
    camera_ids = payload["observation_camera_ids"][rows, 0].numpy()
    uv = payload["observation_uv"][rows, 0].numpy()
    depth = payload["observation_depth"][rows, 0].numpy()
    error = payload["reprojection_error"][rows].numpy()
    names = {
        int(row["image_id"]): str(row["image_name"])
        for row in camera_views
    }
    recovered = {}
    source_camera_ids = sorted(set(map(int, camera_ids.tolist())))
    unmapped_camera_ids = []
    for camera_id in source_camera_ids:
        chosen = np.flatnonzero(camera_ids == camera_id)
        if len(chosen) < 8:
            continue
        name = names.get(camera_id)
        if name is None:
            unmapped_camera_ids.append(camera_id)
            continue
        record = dav2_records.get(Path(name).stem)
        if record is None:
            continue
        ordinal = _resize_depth(Path(record["path"]))
        # Observation UV stores exact normalized pixel centres,
        # ``(index + 0.5) / size``.  The inverse is the half-open raster
        # lookup ``floor(uv * size)``.  Rounding shifts every centre in the
        # upper half of a pixel into its neighbour and made an otherwise
        # exact affine fit fail for almost every accepted camera.
        row = np.clip(
            np.floor(uv[chosen, 1] * ordinal.shape[0]).astype(np.int64),
            0,
            ordinal.shape[0] - 1,
        )
        column = np.clip(
            np.floor(uv[chosen, 0] * ordinal.shape[1]).astype(np.int64),
            0,
            ordinal.shape[1] - 1,
        )
        x = 1.0 / np.maximum(ordinal[row, column], 1e-6)
        y = 1.0 / np.maximum(depth[chosen], 1e-6)
        design = np.column_stack([np.ones(len(x)), x])
        alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = float(
            np.sqrt(np.mean(np.square(y - design @ [alpha, beta])))
        )
        parsed = _parse_view(name)
        if (
            parsed is None
            or not np.isfinite(alpha)
            or not np.isfinite(beta)
            or beta <= 1e-6
            or residual > 1e-4
        ):
            continue
        recovered[name] = {
            "sequence": parsed[0],
            "frame": parsed[1],
            "alpha": float(alpha),
            "beta": float(beta),
            "relative_rmse": float(np.median(error[chosen])),
            "depth_p01": float(np.quantile(depth[chosen], 0.01)),
            "depth_p99": float(np.quantile(depth[chosen], 0.99)),
            "sample_count": int(len(chosen)),
        }
    if audit is not None:
        audit.update(
            {
                "camera_name_source_count": int(len(names)),
                "dav2_source_camera_count": int(len(source_camera_ids)),
                "unmapped_camera_count": int(len(unmapped_camera_ids)),
                "unmapped_camera_ids": unmapped_camera_ids,
                "recovered_alignment_count": int(len(recovered)),
                "camera_name_source_contract": (
                    "fixed_camera_id_to_name__missing_ids_are_audited_"
                    "and_excluded_not_reindexed"
                ),
            }
        )
    return recovered


def _temporal_interpolation(
    name: str,
    by_sequence: dict[str, list[dict]],
    *,
    maximum_gap: int,
) -> dict | None:
    parsed = _parse_view(name)
    if parsed is None:
        return None
    sequence, frame = parsed
    records = by_sequence.get(sequence, ())
    before = [row for row in records if int(row["frame"]) < frame]
    after = [row for row in records if int(row["frame"]) > frame]
    if not before or not after:
        return None
    left = before[-1]
    right = after[0]
    left_gap = frame - int(left["frame"])
    right_gap = int(right["frame"]) - frame
    if (
        left_gap > int(maximum_gap)
        or right_gap > int(maximum_gap)
        or left_gap + right_gap > 2 * int(maximum_gap)
    ):
        return None
    weight = left_gap / float(left_gap + right_gap)
    interpolate = lambda key: (1.0 - weight) * float(left[key]) + weight * float(right[key])
    confidence = float(
        np.clip(
            np.exp(-(left_gap + right_gap) / max(float(maximum_gap), 1.0))
            * np.exp(
                -0.5
                * (
                    float(left["relative_rmse"])
                    + float(right["relative_rmse"])
                )
            ),
            0.10,
            0.85,
        )
    )
    return {
        "alpha": interpolate("alpha"),
        "beta": interpolate("beta"),
        "depth_min": 0.5
        * min(float(left["depth_p01"]), float(right["depth_p01"])),
        "depth_max": 2.0
        * max(float(left["depth_p99"]), float(right["depth_p99"])),
        "confidence": confidence,
        "left_frame": int(left["frame"]),
        "right_frame": int(right["frame"]),
        "maximum_gap": max(left_gap, right_gap),
    }


def _detail_rows(
    rows: np.ndarray,
    columns: np.ndarray,
    score: np.ndarray,
    count: int,
) -> np.ndarray:
    if count <= 0 or not len(rows):
        return np.empty(0, dtype=np.int64)
    count = min(int(count), len(rows))
    side = max(1, int(np.ceil(np.sqrt(2 * count))))
    cell = (
        np.minimum(rows * side // 135, side - 1) * side
        + np.minimum(columns * side // 240, side - 1)
    )
    winners = []
    for cell_id in np.unique(cell):
        candidates = np.flatnonzero(cell == cell_id)
        order = np.lexsort((candidates, -score[candidates]))
        winners.append(int(candidates[order[0]]))
    winners = np.asarray(winners, dtype=np.int64)
    order = np.lexsort((winners, -score[winners]))
    selected = winners[order[:count]]
    if len(selected) < count:
        remaining = np.setdiff1d(
            np.arange(len(rows)), selected, assume_unique=False
        )
        order = np.lexsort((remaining, -score[remaining]))
        selected = np.concatenate(
            [selected, remaining[order[: count - len(selected)]]]
        )
    return selected


def _spatial_coverage_rows(
    rows: np.ndarray,
    columns: np.ndarray,
    count: int,
) -> np.ndarray:
    """Choose deterministic image-space coverage, independent of row order.

    ``np.nonzero`` is raster ordered, so subsampling it with ``linspace``
    repeatedly selects similar columns and leaves large canopy holes.  Start
    with one point nearest each regular cell centre, then complete a k-centre
    traversal.  The latter only handles empty/irregular cells and therefore
    remains cheap for the small per-view budgets used here.
    """
    if count <= 0 or not len(rows):
        return np.empty(0, dtype=np.int64)
    count = min(int(count), len(rows))
    points = np.column_stack(
        [
            (columns.astype(np.float64) + 0.5) / 240.0,
            (rows.astype(np.float64) + 0.5) / 135.0,
        ]
    )
    grid_rows = max(
        1, int(round(np.sqrt(count * 135.0 / 240.0)))
    )
    # Keep the initial grid at or below the requested count. Any missing
    # occupied cells are completed by the unbiased k-centre traversal below.
    grid_columns = max(1, int(count // grid_rows))
    cell_row = np.minimum(
        (points[:, 1] * grid_rows).astype(np.int64),
        grid_rows - 1,
    )
    cell_column = np.minimum(
        (points[:, 0] * grid_columns).astype(np.int64),
        grid_columns - 1,
    )
    cell = cell_row * grid_columns + cell_column
    winners = []
    for cell_id in np.unique(cell):
        candidates = np.flatnonzero(cell == cell_id)
        target = np.asarray(
            [
                (int(cell_id) % grid_columns + 0.5) / grid_columns,
                (int(cell_id) // grid_columns + 0.5) / grid_rows,
            ]
        )
        distance = np.square(points[candidates] - target).sum(axis=1)
        winners.append(int(candidates[np.argmin(distance)]))
    selected = np.asarray(winners, dtype=np.int64)
    selected_mask = np.zeros(len(rows), dtype=bool)
    selected_mask[selected] = True
    if len(selected):
        minimum_distance = np.full(len(rows), np.inf, dtype=np.float64)
        for index in selected:
            minimum_distance = np.minimum(
                minimum_distance,
                np.square(points - points[index]).sum(axis=1),
            )
    else:
        start = int(
            np.argmin(
                np.square(points - points.mean(axis=0)).sum(axis=1)
            )
        )
        selected = np.asarray([start], dtype=np.int64)
        selected_mask[start] = True
        minimum_distance = np.square(
            points - points[start]
        ).sum(axis=1)
    minimum_distance[selected_mask] = -1.0
    while len(selected) < count:
        index = int(np.argmax(minimum_distance))
        selected = np.append(selected, index)
        selected_mask[index] = True
        minimum_distance = np.minimum(
            minimum_distance,
            np.square(points - points[index]).sum(axis=1),
        )
        minimum_distance[selected_mask] = -1.0
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-initialization", type=Path, required=True)
    parser.add_argument(
        "--coverage-initialization",
        type=Path,
        default=None,
        help=(
            "Optional immutable all-camera initialization used only for its "
            "selected-view/depth-acceptance contract. Geometry, dense rays, "
            "and renderer births are always retained from "
            "--source-initialization. This permits adding exact missing-view "
            "witnesses without thinning an already validated dense posterior."
        ),
    )
    parser.add_argument("--evidence-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-temporal-gap", type=int, default=12)
    parser.add_argument("--maximum-rays-per-view", type=int, default=512)
    parser.add_argument("--maximum-births-per-view", type=int, default=512)
    parser.add_argument("--maximum-canonical-distance", type=float, default=3.0)
    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Replace only an existing output that contains an initialization "
            "manifest. This is intended for the restartable pipeline."
        ),
    )
    args = parser.parse_args()

    source = args.source_initialization.resolve()
    coverage_source = (
        source
        if args.coverage_initialization is None
        else args.coverage_initialization.resolve()
    )
    output = args.output.resolve()
    if output.exists():
        if not args.replace:
            raise FileExistsError(output)
        if not (output / "initialization_manifest.json").is_file():
            raise RuntimeError(
                "Refusing to replace a directory without an initialization "
                f"manifest: {output}"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True)
    manifest = json.loads(
        (source / "initialization_manifest.json").read_text()
    )
    store = load_evidence_store(args.evidence_store)
    if manifest["evidence_hash"] != store["evidence_hash"]:
        raise RuntimeError("Initialization and Evidence Store hashes differ")
    payload = torch.load(
        source / "foliage_seed_gaussians.pth", map_location="cpu"
    )
    # The compact tensor archive keeps runtime-critical audit fields, while
    # aggregate source counts live in the adjacent initialization manifest.
    # Rejoin them before extending either representation.
    payload["audit"] = {
        **manifest.get("foliage", {}),
        **payload.get("audit", {}),
    }
    if coverage_source == source:
        coverage_manifest = manifest
        coverage_payload = payload
    else:
        coverage_manifest = json.loads(
            (
                coverage_source / "initialization_manifest.json"
            ).read_text()
        )
        if coverage_manifest["evidence_hash"] != store["evidence_hash"]:
            raise RuntimeError(
                "Coverage initialization and Evidence Store hashes differ"
            )
        coverage_payload = torch.load(
            coverage_source / "foliage_seed_gaussians.pth",
            map_location="cpu",
        )
        coverage_payload["audit"] = {
            **coverage_manifest.get("foliage", {}),
            **coverage_payload.get("audit", {}),
        }
        source_rgb = manifest.get("rgb_source", {})
        coverage_rgb = coverage_manifest.get("rgb_source", {})
        for key in ("name_set_sha256", "content_mapping_sha256"):
            if source_rgb.get(key) != coverage_rgb.get(key):
                raise RuntimeError(
                    "Source and coverage RGB contracts differ at "
                    f"{key}"
                )
    evidence = payload["ray_evidence"]
    (
        repaired_free_end,
        repaired_hit_start,
        repaired_hit_end,
        strict_interval_audit,
    ) = enforce_strict_foliage_ray_intervals(
        evidence["free_end_depth"].numpy(),
        evidence["hit_start_depth"].numpy(),
        evidence["hit_end_depth"].numpy(),
        evidence["observation_type"].numpy(),
    )
    evidence["free_end_depth"] = torch.from_numpy(repaired_free_end)
    evidence["hit_start_depth"] = torch.from_numpy(repaired_hit_start)
    evidence["hit_end_depth"] = torch.from_numpy(repaired_hit_end)
    selected_views = coverage_payload["audit"]["selected_views"]
    camera_views = coverage_payload["audit"].get(
        "fixed_camera_sequences", selected_views
    )
    dav2_records = json.loads(
        artifact_path(store, "dav2_index").read_text()
    )["records"]
    alignment_camera_audit: dict = {}
    alignment = _recover_alignments(
        coverage_payload,
        camera_views,
        dav2_records,
        audit=alignment_camera_audit,
    )
    by_sequence: dict[str, list[dict]] = {}
    for row in alignment.values():
        by_sequence.setdefault(str(row["sequence"]), []).append(row)
    for rows in by_sequence.values():
        rows.sort(key=lambda row: int(row["frame"]))

    dataset = Path(store["dataset"])
    sparse = dataset / "sparse" / "0"
    cameras = read_cameras_binary(sparse / "cameras.bin")
    images = read_images_binary(sparse / "images.bin")
    image_by_name = {
        str(row["name"]): (int(image_id), row)
        for image_id, row in images.items()
    }
    semantic = json.loads(Path(store["semantic_contract"]).read_text())
    masks = CambridgeMaskLookup(
        dataset,
        Path(semantic["tree_mask_pickle"]),
        mask_indices=[0, 1, 2, 3],
    )
    rgb_root = Path(manifest["rgb_source"]["image_root"])

    canonical_rows = torch.nonzero(
        payload["layer_role"] < 2, as_tuple=False
    ).flatten().numpy()
    canonical_xyz = payload["centers"][canonical_rows].numpy()
    canonical_tree = cKDTree(canonical_xyz)
    canonical_instance = payload["tree_instance_id"][
        canonical_rows
    ].numpy()
    canonical_group = payload["replacement_group"][
        canonical_rows
    ].numpy()
    dense_by_camera = {
        int(row["image_id"]): row
        for row in coverage_payload["audit"]["dense_ray_budget"][
            "per_view"
        ]
    }
    if coverage_payload is not payload:
        del coverage_payload

    tensor_rows: dict[str, list[torch.Tensor]] = {
        key: [] for key, value in payload.items()
        if torch.is_tensor(value) and value.ndim > 0
        and len(value) == len(payload["centers"])
    }
    ray_rows = {
        key: [] for key in (
            "camera_ids",
            "pixels",
            "source_image_sizes",
            "free_end_depth",
            "hit_start_depth",
            "hit_end_depth",
            "observation_type",
            "confidence",
        )
    }
    next_track = int(payload["track_id"].min()) - 1
    per_view_audit = []
    candidate_views = coverage_payload["audit"].get(
        "fixed_camera_sequences", selected_views
    )
    candidate_audit = {
        "candidate_camera_count": int(len(candidate_views)),
        "existing_dense_ray_camera_count": 0,
        "missing_two_sided_alignment_count": 0,
        "missing_depth_or_camera_record_count": 0,
        "empty_tree_depth_support_count": 0,
        "outside_canonical_distance_count": 0,
    }
    target_names = {
        "seq1__frame00075.png",
        "seq1__frame00076.png",
        "seq1__frame00077.png",
        "seq1__frame00079.png",
    }

    for selected in candidate_views:
        name = str(selected["image_name"])
        camera_id = int(selected["image_id"])
        dense = dense_by_camera.get(camera_id, {})
        if int(dense.get("accepted_rays", 0)) > 0:
            candidate_audit["existing_dense_ray_camera_count"] += 1
            continue
        interpolation = _temporal_interpolation(
            name,
            by_sequence,
            maximum_gap=args.maximum_temporal_gap,
        )
        if interpolation is None:
            candidate_audit["missing_two_sided_alignment_count"] += 1
            continue
        depth_record = dav2_records.get(Path(name).stem)
        image_item = image_by_name.get(name)
        if depth_record is None or image_item is None:
            candidate_audit["missing_depth_or_camera_record_count"] += 1
            continue
        image_id, image = image_item
        if image_id != camera_id:
            raise RuntimeError("Scene camera id/name contract changed")
        ordinal = _resize_depth(Path(depth_record["path"]))
        key = masks.source_name_for(name)
        channels = masks.masks[key]
        resized = [
            F.interpolate(
                value.detach().float()[None, None],
                size=ordinal.shape,
                mode="nearest",
            )[0, 0]
            .bool()
            .numpy()
            for value in channels[:4]
        ]
        canopy = resized[0] & resized[1] & ~resized[3]
        denominator = (
            float(interpolation["alpha"])
            + float(interpolation["beta"])
            / np.maximum(ordinal, 1e-6)
        )
        metric_depth = np.zeros_like(ordinal, dtype=np.float32)
        valid_depth = (
            np.isfinite(denominator)
            & (denominator > 1e-6)
        )
        metric_depth[valid_depth] = 1.0 / denominator[valid_depth]
        valid = (
            canopy
            & valid_depth
            & (metric_depth >= max(0.05, interpolation["depth_min"]))
            & (metric_depth <= interpolation["depth_max"])
        )
        rows, columns = np.nonzero(valid)
        if not len(rows):
            candidate_audit["empty_tree_depth_support_count"] += 1
            continue
        camera = cameras[int(image["camera_id"])]
        fx, fy, cx, cy = _camera_parameters(camera, ordinal.shape)
        z = metric_depth[rows, columns].astype(np.float64)
        camera_xyz = np.column_stack(
            [
                (columns + 0.5 - cx) * z / fx,
                (rows + 0.5 - cy) * z / fy,
                z,
            ]
        )
        rotation = quaternion_to_rotation(image["qvec"])
        translation = np.asarray(image["tvec"], dtype=np.float64)
        world = (camera_xyz - translation[None]) @ rotation
        distance, nearest = canonical_tree.query(world, k=1)
        local = distance <= float(args.maximum_canonical_distance)
        rows, columns, z, world, nearest, distance = (
            value[local]
            for value in (rows, columns, z, world, nearest, distance)
        )
        if not len(rows):
            candidate_audit["outside_canonical_distance_count"] += 1
            continue

        with Image.open(rgb_root / name) as handle:
            rgb_image = np.asarray(handle.convert("RGB"))
        luminance = (
            0.2126 * rgb_image[..., 0]
            + 0.7152 * rgb_image[..., 1]
            + 0.0722 * rgb_image[..., 2]
        ).astype(np.float32)
        gx = np.zeros_like(luminance)
        gy = np.zeros_like(luminance)
        gx[:, 1:-1] = np.abs(luminance[:, 2:] - luminance[:, :-2])
        gy[1:-1] = np.abs(luminance[2:] - luminance[:-2])
        rgb_rows = np.clip(
            np.floor((rows + 0.5) / 135 * rgb_image.shape[0]).astype(int),
            0,
            rgb_image.shape[0] - 1,
        )
        rgb_columns = np.clip(
            np.floor((columns + 0.5) / 240 * rgb_image.shape[1]).astype(int),
            0,
            rgb_image.shape[1] - 1,
        )
        score = gx[rgb_rows, rgb_columns] + gy[rgb_rows, rgb_columns]
        birth_count = min(
            int(args.maximum_births_per_view), len(rows)
        )
        coverage_count = (3 * birth_count + 3) // 4
        coverage = _spatial_coverage_rows(
            rows, columns, coverage_count
        )
        remaining = np.setdiff1d(
            np.arange(len(rows)), coverage, assume_unique=False
        )
        detail_local = _detail_rows(
            rows[remaining],
            columns[remaining],
            score[remaining],
            birth_count - coverage_count,
        )
        detail = remaining[detail_local]
        births = np.concatenate([coverage, detail])
        is_detail = np.concatenate(
            [
                np.zeros(len(coverage), dtype=bool),
                np.ones(len(detail), dtype=bool),
            ]
        )
        # An ownerless hit without a birth at the same measured ray/depth is
        # a constant loss until another primitive happens to drift into its
        # interval. Spatial proximity alone is insufficient at a foliage
        # depth discontinuity. Bind every persisted ray to one exact dynamic
        # birth; when a caller requests fewer rays, retain a spatially
        # representative subset of the birth set.
        ray_count = min(
            int(args.maximum_rays_per_view), len(births)
        )
        if ray_count == len(births):
            ray_choice = births.copy()
        else:
            ray_choice = births[
                _spatial_coverage_rows(
                    rows[births], columns[births], ray_count
                )
            ]

        confidence = float(interpolation["confidence"])
        gap = float(interpolation["maximum_gap"])
        depth_sigma = np.maximum(
            0.30, 0.025 * z + 0.04 * gap
        ).astype(np.float32)
        native_width = int(camera["width"])
        native_height = int(camera["height"])
        pixel = np.column_stack(
            [
                (columns[ray_choice] + 0.5) / 240 * native_width,
                (rows[ray_choice] + 0.5) / 135 * native_height,
            ]
        ).astype(np.float32)
        image_size = np.tile(
            np.asarray([native_width, native_height], dtype=np.int32),
            (ray_count, 1),
        )
        ray_rows["camera_ids"].append(
            torch.full((ray_count,), camera_id, dtype=torch.int32)
        )
        ray_rows["pixels"].append(
            torch.from_numpy(
                half_open_raster_coordinates(pixel, image_size)
            )
        )
        ray_rows["source_image_sizes"].append(torch.from_numpy(image_size))
        free_end, hit_start, hit_end = strict_foliage_hit_interval_bounds(
            z[ray_choice],
            depth_sigma[ray_choice],
            free_space_margin=0.25,
        )
        ray_rows["free_end_depth"].append(torch.from_numpy(free_end))
        ray_rows["hit_start_depth"].append(torch.from_numpy(hit_start))
        ray_rows["hit_end_depth"].append(torch.from_numpy(hit_end))
        ray_rows["observation_type"].append(
            torch.ones(ray_count, dtype=torch.int8)
        )
        ray_rows["confidence"].append(
            torch.full((ray_count,), confidence, dtype=torch.float32)
        )

        count = len(births)
        birth_z = z[births].astype(np.float32)
        reference_downsample = max(native_width / 640.0, 1.0)
        represented_pixels = np.where(
            is_detail, 1.5 * reference_downsample, 4.0 * reference_downsample
        )
        native_fx = fx * native_width / 240.0
        native_fy = fy * native_height / 135.0
        footprint = np.clip(
            0.60
            * birth_z
            * represented_pixels
            / np.sqrt(native_fx * native_fy),
            0.008,
            0.12,
        ).astype(np.float32)
        sigma = (
            0.15 + 0.025 * birth_z + 0.04 * gap
        ).astype(np.float32)
        covariance = (
            np.eye(3, dtype=np.float32)[None]
            * np.square(sigma[:, None, None])
        )
        instance = canonical_instance[nearest[births]].astype(np.int32)
        replacement = np.where(
            distance[births] <= 0.36,
            canonical_group[nearest[births]],
            -1,
        ).astype(np.int64)
        support_width = payload["support_camera_ids"].shape[1]
        observation_width = payload["observation_camera_ids"].shape[1]
        support_ids = np.full(
            (count, support_width), -1, dtype=np.int32
        )
        support_ids[:, 0] = camera_id
        observation_ids = np.full(
            (count, observation_width), -1, dtype=np.int32
        )
        observation_ids[:, 0] = camera_id
        observation_uv = np.full(
            (count, observation_width, 2), np.nan, dtype=np.float32
        )
        observation_uv[:, 0] = np.column_stack(
            [(columns[births] + 0.5) / 240, (rows[births] + 0.5) / 135]
        )
        observation_depth = np.full(
            (count, observation_width), np.nan, dtype=np.float32
        )
        observation_depth[:, 0] = birth_z
        colors = (
            rgb_image[rgb_rows[births], rgb_columns[births]].astype(np.float32)
            / 255.0
        )
        reprojection = np.clip(
            sigma * max(native_fx, native_fy) / np.maximum(birth_z, 0.05),
            0.2,
            10.0,
        ).astype(np.float32)
        values = {
            "centers": torch.from_numpy(world[births].astype(np.float32)),
            "colors": torch.from_numpy(colors),
            "scales": torch.from_numpy(
                np.repeat(footprint[:, None], 3, axis=1)
            ),
            "opacities": torch.full(
                (count, 1), 0.035 + 0.040 * confidence
            ),
            "quaternions": torch.from_numpy(
                np.tile(
                    np.asarray([[1.0, 0.0, 0.0, 0.0]], np.float32),
                    (count, 1),
                )
            ),
            "primitive_role": torch.ones(count, dtype=torch.int8),
            "layer_role": torch.full((count,), 2, dtype=torch.int8),
            "track_id": torch.arange(
                next_track, next_track - count, -1, dtype=torch.int64
            ),
            "tree_instance_id": torch.from_numpy(instance),
            "initialization_source": torch.full(
                (count,), 3, dtype=torch.int8
            ),
            "occupancy_probability": torch.full(
                (count,), 0.55 + 0.30 * confidence
            ),
            "position_covariance": torch.from_numpy(covariance),
            "reprojection_error": torch.from_numpy(reprojection),
            "track_linearity": torch.ones(count),
            "support_camera_ids": torch.from_numpy(support_ids),
            "observation_camera_ids": torch.from_numpy(observation_ids),
            "observation_uv": torch.from_numpy(observation_uv),
            "observation_depth": torch.from_numpy(observation_depth),
            "support_view_count": torch.ones(count, dtype=torch.int16),
            "support_sequence_count": torch.ones(count, dtype=torch.int16),
            "ray_depth_nll": torch.full(
                (count,), float(-np.log(max(confidence, 1e-4)))
            ),
            "free_space_violation_count": torch.zeros(
                count, dtype=torch.int16
            ),
            "unknown_view_count": torch.zeros(count, dtype=torch.int16),
            "maximum_baseline": torch.zeros(count),
            "maximum_triangulation_angle_degrees": torch.zeros(count),
            "static_skeleton_confidence": torch.zeros(count),
            "scale_ceiling": torch.from_numpy(
                np.repeat((1.10 * footprint)[:, None], 3, axis=1)
            ),
            "replacement_group": torch.from_numpy(replacement),
        }
        for key, value in values.items():
            tensor_rows[key].append(value.to(dtype=payload[key].dtype))
        next_track -= count
        ray_to_birth_distance_at_640 = (
            cKDTree(
                np.column_stack([columns[births], rows[births]])
            ).query(
                np.column_stack(
                    [columns[ray_choice], rows[ray_choice]]
                ),
                k=1,
            )[0]
            * (640.0 / 240.0)
        )
        per_view_audit.append(
            {
                "image_id": camera_id,
                "image_name": name,
                "rays": ray_count,
                "births": count,
                "confidence": confidence,
                "left_frame": interpolation["left_frame"],
                "right_frame": interpolation["right_frame"],
                "maximum_gap": interpolation["maximum_gap"],
                "nearest_canonical_distance_median": float(
                    np.median(distance[births])
                ),
                "spatial_coverage_birth_count": int(len(coverage)),
                "high_frequency_birth_count": int(len(detail)),
                "exact_ray_depth_birth_count": int(ray_count),
                "every_added_ray_has_exact_birth": True,
                "ray_to_birth_distance_at_640_p50": float(
                    np.quantile(ray_to_birth_distance_at_640, 0.50)
                ),
                "ray_to_birth_distance_at_640_p90": float(
                    np.quantile(ray_to_birth_distance_at_640, 0.90)
                ),
                "ray_to_birth_distance_at_640_max": float(
                    np.max(ray_to_birth_distance_at_640)
                ),
                "target_diagnostic": name in target_names,
            }
        )

    added_births = int(sum(row["births"] for row in per_view_audit))
    added_rays = int(sum(row["rays"] for row in per_view_audit))
    if bool(added_births) != bool(added_rays):
        raise RuntimeError(
            "Temporal DAV2 augmentation must add births and rays together"
        )
    for key, rows in tensor_rows.items():
        if rows:
            payload[key] = torch.cat([payload[key], *rows], dim=0)
    for key, rows in ray_rows.items():
        evidence[key] = torch.cat([evidence[key], *rows], dim=0)
    (
        final_free_end,
        final_hit_start,
        final_hit_end,
        final_interval_audit,
    ) = enforce_strict_foliage_ray_intervals(
        evidence["free_end_depth"].numpy(),
        evidence["hit_start_depth"].numpy(),
        evidence["hit_end_depth"].numpy(),
        evidence["observation_type"].numpy(),
    )
    evidence["free_end_depth"] = torch.from_numpy(final_free_end)
    evidence["hit_start_depth"] = torch.from_numpy(final_hit_start)
    evidence["hit_end_depth"] = torch.from_numpy(final_hit_end)

    audit = payload["audit"]
    old_count = int(len(payload["centers"]) - added_births)
    audit["protocol"] = PROTOCOL
    audit["total_count"] = int(len(payload["centers"]))
    audit["sequence_local_track_count"] = int(
        audit["sequence_local_track_count"] + added_births
    )
    audit["dav2_sequence_local_sample_count"] = int(
        audit["dav2_sequence_local_sample_count"] + added_births
    )
    audit["dense_observation_ray_count"] = int(
        audit["dense_observation_ray_count"] + added_rays
    )
    audit["temporal_dav2_augmentation"] = {
        "protocol": PROTOCOL,
        "source_initialization": str(source),
        "source_foliage_sha256": _sha256(
            source / "foliage_seed_gaussians.pth"
        ),
        "coverage_initialization": str(coverage_source),
        "coverage_foliage_sha256": _sha256(
            coverage_source / "foliage_seed_gaussians.pth"
        ),
        "dense_source_preserved": True,
        "producer_implementation_sha256": _sha256(Path(__file__)),
        "accepted_alignment_cache_count": int(len(alignment)),
        "alignment_camera_contract": alignment_camera_audit,
        "candidate_camera_audit": candidate_audit,
        "interpolated_view_count": int(len(per_view_audit)),
        "added_dynamic_birth_count": added_births,
        "added_ownerless_hit_ray_count": added_rays,
        "canonical_birth_count": 0,
        "localization_landmark_count": 0,
        "same_sequence_two_sided_only": True,
        "identity_noop": not bool(added_births),
        "selection_contract": (
            "deterministic_image_grid_plus_k_center_coverage__"
            "three_quarter_coverage_one_quarter_rgb_detail__"
            "every_ray_bound_to_exact_same_pixel_depth_dynamic_birth"
        ),
        "every_added_ray_has_exact_birth": True,
        "maximum_temporal_gap": int(args.maximum_temporal_gap),
        "maximum_rays_per_view": int(args.maximum_rays_per_view),
        "maximum_births_per_view": int(args.maximum_births_per_view),
        "maximum_canonical_distance": float(
            args.maximum_canonical_distance
        ),
        "spatial_uncertainty": (
            "per_birth_gap_and_depth_conditioned_isotropic_covariance"
        ),
        "per_view": per_view_audit,
        "source_interval_repair": strict_interval_audit,
        "final_interval_validation": final_interval_audit,
    }

    for name in ("surface_seed.npz", "surface_seed.json"):
        shutil.copy2(source / name, output / name)
    foliage_path = output / "foliage_seed_gaussians.pth"
    torch.save(payload, foliage_path)
    summary = json.loads(
        (source / "foliage_seed_gaussians.json").read_text()
    )
    summary.update(audit)
    (output / "foliage_seed_gaussians.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    manifest["version"] = PROTOCOL
    manifest["surface_seed"] = str(output / "surface_seed.npz")
    manifest["foliage_seed"] = str(foliage_path)
    manifest["foliage"] = audit
    manifest["causal_reuse"] = {
        "policy": "same_evidence_untrained_temporal_dav2_augmentation",
        "source_initialization": str(source),
        "coverage_initialization": str(coverage_source),
        "source_total_count": old_count,
        "dense_source_preserved": True,
        "canonical_geometry_unchanged": True,
        "surface_geometry_unchanged": True,
        "historical_trained_rgb_model_used": False,
    }
    (output / "initialization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "accepted_alignment_cache_count": len(alignment),
                "interpolated_view_count": len(per_view_audit),
                "added_births": added_births,
                "added_rays": added_rays,
                "total_count": len(payload["centers"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
